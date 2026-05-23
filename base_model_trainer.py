import gc
from functools import partial

import torch
import wandb
import numpy as np
import pytorch_lightning as L

from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage

from data.cv.aggregate_dataloader import AggregateDataset
from data.cv.ego4d_dataloader import Ego4dDataset
from data.cv.finevideo_dataloader import FineVideoDataset
from data.cv.something_dataloader import SomethingDataset

from model.cv.tdv.tdv import TDV
from model.cv.tdv.utils import collate_data_and_cast, MaskingGenerator
from model.model_utils import get_cv_transforms

from optimization import LARS, WarmUpCosineAnnealingLR, exclude_bias_and_norm


class ModelTrainer(L.LightningModule):
	def __init__(self, hparams, trained_model = None):
		super().__init__()

		self.save_hyperparameters()

		# -- dict form when loading from checkpoint
		if isinstance(hparams, dict):
			self.hparams.update(hparams)
		else:
			self.hparams.update(vars(hparams))

		if self.hparams.modality == "CV":
			self.image_dims = self.hparams.image_dim
			self.transform, self.normal_lookup = get_cv_transforms(self.hparams.dataset_name, self.image_dims, self.hparams.custom_image_normalization, self.hparams.vae_normalization)
			self.to_pil = ToPILImage()

		self.full_ds = None

		if trained_model is not None:
			self.model = trained_model
		else:
			if self.hparams.model_name == "tdv":
				self.model = TDV(self.hparams)
			else:
				raise ValueError(f"do not recognize model name: {self.hparams.model_name}")

		if self.hparams.compile_model:
			self.model = torch.compile(self.model, fullgraph=True)

		self.torchmetrics_dict = nn.ModuleDict()
		self.metrics = []
		for metric in self.hparams.metrics_list:
			self.metrics.append(metric)
		if len(self.metrics) > 0:
			assert self.hparams.num_classes != -1, "please set num_classes to the appropriate amount for the in use metrics. if are using accuracy and num_classes varies just set it to something that makes sense (shouldnt matter in that case)"
			assert self.hparams.metrics_task != "", "please set metrics_task to the appropriate value for your metrics"

		if self.hparams.wandb_watch:
			for name, module in self.model.named_modules():
				module.name = name


	def on_train_start(self):
		if self.hparams.debug_unused_parameters:
			for name, param in self.model.named_parameters():
				# -- update frozen prefix exclusions here if model changes which params are frozen
				if param.requires_grad and "image_encoder" not in name:
					print(f"registering param - {name}")
					param.register_hook(self.create_hook(name))
				else:
					self.model.parameters_not_to_check.add(name)


	# -- only used when debug_unused_parameters=True
	def create_hook(self, name):
		def hook(_):
			self.model.used_parameters.add(name)
		return hook


	@staticmethod
	def wandb_activation_hook(run, step):
		# -- logs per-module output histograms to wandb
		def hook(module, _, output):
			# -- wandb.Histogram requires a plain tensor; skip compound outputs
			if isinstance(output, (tuple, list, dict)):
				pass
			else:
				run.experiment.log(
					{f"activations/{module.name}": wandb.Histogram(output.detach().cpu().float())},
					step=step
				)

		return hook


	def on_train_epoch_start(self):
		if self.hparams.reinit_difference_encoder_every_epoch:
			print(f"[Epoch {self.current_epoch}] Reinitializing difference encoder weights.")
			self.model.motion_encoder.init_weights()
			self.model.linear_fc.reset_parameters()


	def training_step(self, batch, _batch_idx):
		if not self.hparams.no_wandb and self.hparams.wandb_watch and self.global_step % self.hparams.wandb_watch_log_freq == 0:
			hook_handles = []
			hook_function = self.wandb_activation_hook(run=self.logger, step=self.global_step)
			for module in self.model.modules():
				# -- only hook unfrozen modules
				if any(param.requires_grad for param in module.parameters(recurse=False)):
					handle = module.register_forward_hook(hook_function)
					hook_handles.append(handle)

			eval_step_dict = self.eval_step(batch)
			for handle in hook_handles:
				handle.remove()

		else:
			eval_step_dict = self.eval_step(batch)

		self.log_metrics(eval_step_dict, "train")
		return eval_step_dict['loss']


	def on_after_backward(self):
		if self.hparams.log_gradients:
			things_to_log = {}
			total_norm = 0.0
			num_parameters = 0
			num_grads_exceeding_clip_val = 0
			# -- counts individual scalars, not parameter tensors
			total_gradients = 0
			for param in self.parameters():
				if param.grad is not None:
					param_norm = param.grad.data.norm(2)
					total_norm += param_norm
					num_parameters += 1

					total_gradients += torch.numel(param.grad)
					num_grads_exceeding_clip_val += torch.sum(param.grad.abs() > self.hparams.gradient_clip_val)

			# -- skip assertion when using conditional loss (rgb_diff_threshold > 0),
			# -- as some batches may produce no active gradients
			if not self.hparams.rgb_diff_threshold > 0:
				assert num_parameters > 0, "no gradients after backwards detected please investigate"

			if num_parameters > 0:
				average_norm = (total_norm / num_parameters).detach()
				things_to_log['avg_gradient_norms'] = average_norm

			if total_gradients > 0:
				percentage_clipped = ((num_grads_exceeding_clip_val / total_gradients) * 100).detach()
				things_to_log['pct_gradient_clipped'] = percentage_clipped

			self.log_metrics(things_to_log, "train", log_torchmetrics = False)


	def on_train_batch_end(self, _outputs, _batch, _batch_idx):
		# -- add frozen param prefix exclusions here when using with partially frozen models
		if self.hparams.debug_unused_parameters:
			all_parameters = {name for name, _ in self.model.named_parameters()}
			unused_parameters = all_parameters - self.model.used_parameters - self.model.parameters_not_to_check

			print(f"Number of total parameters: {len(all_parameters)}")
			print(f"Number of unused_parameters: {len(unused_parameters)}")
			print(f"Unused parameters: {unused_parameters}")

		if self.hparams.manual_gc_collect_every_n_steps != -1:
			if self.global_step % self.hparams.manual_gc_collect_every_n_steps == 0:
				print("calling GC manually")
				gc.collect()


	def on_before_zero_grad(self, _optimizer):
		if self.hparams.use_ema_for_frame_encoder:
			self.model.ema_update()


	def on_train_epoch_end(self):
		# -- e.g. for lars need to manually update epoch
		if self.hparams.optimizer != "adamw":
			optimizer = self.trainer.optimizers[0]
			optimizer.update_epoch(self.current_epoch)


	def validation_step(self, batch, _batch_idx):
		eval_step_dict = self.eval_step(batch)
		self.log_metrics(eval_step_dict, "valid")


	def test_step(self, batch, _batch_idx):
		eval_step_dict = self.eval_step(batch)
		self.log_metrics(eval_step_dict, "test")


	def eval_step(self, batch):
		# -- forward_loss_wrapper must return a dict containing at least 'loss' for backprop
		things_to_log = self.model.forward_loss_wrapper(batch)

		if len(self.metrics) > 0:
			raise NotImplementedError("Need to implement torchmetrics stuff, i.e. looping through self.torchmetrics_dict.keys(), checking to make sure 'phase in key', and updating based off predicted and labels i.e. self.torchmetrics_dict[key].update(logits, labels), more info https://lightning.ai/docs/torchmetrics/stable/pages/lightning.html (just be careful make sure to detach logits before using them and only update current phase). recommended to possibly return things_to_log and logits from forward_loss_wrapper to do this easily")

		return things_to_log


	def forward(self, batch):
		return self.model(batch)


	def configure_optimizers(self):
		if self.hparams.modality == "CV":
			return self.configure_optimizers_vision()
		else:
			raise NotImplementedError(f"Modality {self.hparams.modality} does not have configure optimizers supported yet")


	def on_warm_up_finished(self):
		if hasattr(self.model, 'warm_up_finished'):
			self.model.warm_up_finished()
			print("Warm up finished, calling self.model.warm_up_finished()")
		else:
			print("Warm up finished, no self.model.warm_up_finished() exists so not doing anything")


	def get_optimizer(self, optimizer_parameters):
		if self.hparams.optimizer == "lars":
			lars_exclude_bias_and_norm = None if not self.hparams.lars_exclude_bias_bn_wd else exclude_bias_and_norm
			optimizer = LARS(optimizer_parameters, lr=self.hparams.peak_learning_rate, weight_decay=self.hparams.weight_decay, momentum=self.hparams.beta1, eta=self.hparams.lars_trust_coeff, weight_decay_filter=lars_exclude_bias_and_norm, lars_adaptation_filter=lars_exclude_bias_and_norm)
		else:
			optimizer = torch.optim.AdamW(optimizer_parameters, betas=[self.hparams.beta1, self.hparams.beta2])
		return optimizer


	def get_lr_scheduler(self, optimizer):
		ipe = len(self.train_dataloader())

		# -- adjust ipe for effective batch size when using gradient accumulation
		if self.hparams.accumulate_grad_batches > 1:
			ipe = ipe // self.hparams.accumulate_grad_batches

		max_steps = self.hparams.max_scheduling_steps if self.hparams.max_scheduling_steps else (ipe * self.hparams.epochs)
		cosine_annealing_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps - self.hparams.warm_up_steps, eta_min=self.hparams.peak_learning_rate / self.hparams.min_lr_scale)
		lr_scheduler = WarmUpCosineAnnealingLR(optimizer, warm_up_steps = self.hparams.warm_up_steps, warm_up_base_lr_divider = self.hparams.warm_up_base_lr_divider, cosine_scheduler=cosine_annealing_scheduler, warm_up_finished_func=self.on_warm_up_finished)

		return lr_scheduler

	def get_optimizer_scheduler_dict(self, optimizer_parameters):
		optimizer = self.get_optimizer(optimizer_parameters)
		lr_scheduler = self.get_lr_scheduler(optimizer)
		return {
			'optimizer': optimizer,
			'lr_scheduler': {
				'scheduler': lr_scheduler,
				'interval': 'step',
				'frequency': 1
			}
		}

	def configure_optimizers_vision(self):
		if self.hparams.model_name == "tdv":
			optimizer_parameters = []

			# -- frame encoder
			frame_encoder_params = list(self.model.frame_encoder.parameters())
			if self.hparams.unfreeze_frame_encoder:
				optimizer_parameters.append({'params': frame_encoder_params, 'weight_decay': self.hparams.weight_decay, 'lr': self.hparams.peak_learning_rate})

			# -- dino head
			if self.hparams.use_dino_head:
				student_dino_head_params = list(self.model.dino_head.parameters())
				optimizer_parameters.append({'params': student_dino_head_params, 'weight_decay': self.hparams.weight_decay, 'lr': self.hparams.peak_learning_rate})

			# -- ibot head
			if self.hparams.use_seperate_ibot_head:
				student_ibot_head_params = list(self.model.ibot_head.parameters())
				optimizer_parameters.append({'params': student_ibot_head_params, 'weight_decay': self.hparams.weight_decay, 'lr': self.hparams.peak_learning_rate})

			# -- motion encoder
			if not self.hparams.remove_motion_encoder:
				motion_encoder_params = list(self.model.motion_encoder.parameters())
				motion_encoder_params += list(self.model.linear_fc.parameters())
				if not self.hparams.freeze_difference_encoder:
					motion_encoder_lr = self.hparams.peak_learning_rate * self.hparams.difference_encoder_lr_multiplier
					optimizer_parameters.append({'params': motion_encoder_params, 'weight_decay': self.hparams.weight_decay, 'lr': motion_encoder_lr})

			optimizer_dict = self.get_optimizer_scheduler_dict(optimizer_parameters)

		else:
			raise NotImplementedError(f"haven't implemented configure optimizers for model {self.hparams.model_name}")

		return optimizer_dict


	def setup(self, stage=None):
		assert self.hparams.test_split_pct == 0, "Haven't implemented nonzero value for test_split_pct yet"

		if stage == "fit":
			if self.hparams.dataset_name in ('something' , 'smth'):
				self.train_ds = SomethingDataset(self.hparams, split = 'train', transform = self.transform)
				self.val_ds = SomethingDataset(self.hparams, split = 'val', transform = self.transform)
			elif self.hparams.dataset_name in ('ego4d'):
				self.train_ds = Ego4dDataset(self.hparams, split = 'train', transform = self.transform)
				self.val_ds = Ego4dDataset(self.hparams, split = 'val', transform = self.transform)
			elif self.hparams.dataset_name in ('finevideo', 'fv'):
				self.train_ds = FineVideoDataset(self.hparams, split = 'train', transform = self.transform)
				self.val_ds = FineVideoDataset(self.hparams, split = 'val', transform = self.transform)
			elif self.hparams.dataset_name in ('aggr', 'aggregate'):
				self.train_ds = AggregateDataset(self.hparams, split = 'train', transform = self.transform)
				self.val_ds = AggregateDataset(self.hparams, split = 'val', transform = self.transform)
			else:
				raise NotImplementedError("Haven't implemented this dataset yet")

			print(f"{self.hparams.dataset_name} length of train_dataset: {len(self.train_ds)}")

			if self.hparams.train_data_pct < 1.0:
				n = max(1, int(len(self.train_ds) * self.hparams.train_data_pct))
				rng = torch.Generator().manual_seed(42)
				indices = torch.randperm(len(self.train_ds), generator=rng)[:n].tolist()
				self.train_ds = torch.utils.data.Subset(self.train_ds, indices)
				print(f"train_data_pct={self.hparams.train_data_pct}: using {n} training samples")

			if self.val_ds is not None:
				print(f"{self.hparams.dataset_name} length of val_dataset: {len(self.val_ds)}")


	def get_collate_fn(self):
		collate_fn = None
		if self.hparams.use_ibot_masking:
			img_size = self.hparams.image_dim[0]
			patch_size = self.hparams.patch_size
			n_tokens = (img_size // patch_size) ** 2
			mask_generator = MaskingGenerator(
				input_size=(img_size // patch_size, img_size // patch_size),
				max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
			)

			collate_fn = partial(
				collate_data_and_cast,
				mask_ratio_tuple=self.hparams.ibot_mask_ratio_min_max,
				mask_probability=self.hparams.ibot_mask_sample_probability,
				n_tokens=n_tokens,
				mask_generator=mask_generator,
			)
		return collate_fn


	def train_dataloader(self):
		return DataLoader(self.train_ds, batch_size=self.hparams.batch_size_per_device, num_workers=self.hparams.num_workers, persistent_workers=True, collate_fn = self.get_collate_fn(), pin_memory = True, drop_last = False, shuffle = not self.hparams.no_shuffle)

	def val_dataloader(self):
		return DataLoader(self.val_ds, batch_size=self.hparams.batch_size_per_device, num_workers=self.hparams.num_workers, persistent_workers=True, collate_fn = self.get_collate_fn(), pin_memory = True, drop_last = False, shuffle = False)

	def test_dataloader(self):
		return DataLoader(self.test_ds, batch_size=self.hparams.batch_size_per_device, num_workers=self.hparams.num_workers, persistent_workers=True, collate_fn = self.get_collate_fn(), pin_memory = True, drop_last = False, shuffle = False)


	def log_metrics(self, metrics_dict, phase, log_torchmetrics = True):
		if log_torchmetrics and len(self.metrics) > 0:
			phase_dict = {key : value for key, value in self.torchmetrics_dict.items() if phase in key}
			self.log_dict(phase_dict, on_step = False, on_epoch = True)

		# -- copy keys to avoid mutation during iteration
		scalar_metrics = {}
		keys = list(metrics_dict.keys())
		for key in keys:
			value = metrics_dict[key]
			if 'image' in key:
				image = self.to_pil(value)
				wandb_image = wandb.Image(image, mode="RGB")
				self.logger.experiment.log({f'{phase}/{key}': wandb_image})

			elif 'video' in key:
				video_np = value.cpu().numpy()
				assert video_np.ndim != 5, "video should not include batch dimension, either fix that or add support"
				if video_np.shape[1] in [1, 3]:
					# -- axes are already (frames, C, H, W)
					pass
				elif video_np.shape[-1] in [1, 3]:
					# -- (frames, H, W, C) to (frames, C, H, W)
					video_np = video_np.transpose(0, 3, 1, 2)
				else:
					raise ValueError(f"Unexpected video shape: {video_np.shape}")
				if video_np.dtype != np.uint8:
					video_np = (video_np * 255).astype(np.uint8)
				wandb_video = wandb.Video(video_np, fps=4, format="mp4")
				self.logger.experiment.log({f'{phase}/{key}': wandb_video})

			elif isinstance(value, torch.Tensor) and value.numel() > 1:
				self.logger.experiment.log({f"{phase}/{key}": wandb.Histogram(value.detach().cpu())})

			elif isinstance(value, torch.Tensor) and value.dim() == 0:
				scalar_metrics[f"{phase}/{key}"] = value.detach()
			elif isinstance(value, (int, float)):
				scalar_metrics[f"{phase}/{key}"] = value
			else:
				raise ValueError(f"unsupported type/format in log_metrics, type:, {type(value)}, key: {key}")

		if scalar_metrics:
			self.log_dict(scalar_metrics, sync_dist=True, prog_bar=True, on_epoch=True)

		if len(self.trainer.optimizers) == 0:
			# -- no optimizers during testing
			pass
		else:
			# -- relies on the main model's lr being in the first param group
			current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
			self.log("Global_LR", current_lr)

			if self.hparams.difference_encoder_lr_multiplier != 1.0:
				diff_encoder_lr = self.trainer.optimizers[0].param_groups[-1]['lr']
				self.log("Difference_Encoder_LR", diff_encoder_lr)
