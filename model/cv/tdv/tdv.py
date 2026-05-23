import torch
import pytorch_lightning as L
import torch.nn.functional as F

from model.cv.tdv.losses.center_sharp_mse_loss import CenterSharpReconstructionLoss
from model.cv.tdv.losses.dino_loss import DinoLoss
from model.cv.tdv.utils import *
from model.model_utils import calculate_var_covar, encode_images, create_motion_encoder, create_image_encoder


class TDVDifferenceEncoderXAttn(L.LightningModule):
	"""
	Temporal Difference Vision (TDV) difference encoder with cross attention.
	"""

	def __init__(self, hparams):
		super().__init__()

		if isinstance(hparams, dict): self.hparams.update(hparams)
		else: self.hparams.update(vars(hparams))
		self.EPS = 0.00001

		# MAE style masking
		self.mask_ratio = self.hparams.frame_encoder_mask_ratio
		self.mask_ratio = None if self.mask_ratio <= 0.0 else self.mask_ratio
		self.use_masking = self.mask_ratio is not None or self.hparams.use_ibot_masking

		# MODEL COMPONENTS INIT ####################

		# -- frame encoder
		self.encoder_name = self.hparams.backbone_type
		self.encoder_dim = self.hparams.vit_backbone_dim

		self.frame_encoder = create_image_encoder(
			self.hparams.backbone_type,
			self.hparams.vit_backbone_size,
			pretrained = not self.hparams.load_without_weights,
			use_rope = self.hparams.use_rope,
			use_ape = self.hparams.use_ape,
			use_masking = self.use_masking
		)
		set_trainable(self.frame_encoder, trainable=self.hparams.unfreeze_frame_encoder)
		print(f"Loaded frame encoder as {self.encoder_name}: {self.encoder_dim}")

		# -- motion encoder
		if not self.hparams.remove_motion_encoder:
			self.motion_encoder, output_embedding_dim = create_motion_encoder(
				self.hparams.change_encoder_type,
				depth=self.hparams.num_transformer_blocks,
				xattn_condition_dim=self.hparams.vit_backbone_dim,
				use_spatial_conditioning=self.hparams.use_spatial_conditioning,
				spatial_condn_gating=self.hparams.use_gating,
				ignore_prefix_tokens_in_condition=self.hparams.ignore_prefix_tokens_in_condition,
				use_rope=self.hparams.use_rope,
				use_ape=self.hparams.use_ape,
				use_masking=self.use_masking
			)
			set_trainable(self.motion_encoder, trainable=not self.hparams.freeze_difference_encoder)
			print(f"Loaded motion encoder as {self.hparams.change_encoder_type}: {output_embedding_dim}")

			# -- motion encoder dim to frame encoder dim
			self.linear_fc = torch.nn.Linear(output_embedding_dim, self.hparams.vit_backbone_dim)
			set_trainable(self.linear_fc, trainable= not self.hparams.freeze_difference_encoder)

		# -- dino head
		self.dino_head = None
		if self.hparams.use_dino_head:
			self.dino_head = init_dino_head(
				in_dim=self.hparams.vit_backbone_dim, 
				out_dim=self.hparams.dino_head_prototype_dim,
				layers=3)
			set_trainable(self.dino_head, trainable=True)

			if self.hparams.use_seperate_ibot_head:
				self.ibot_head = init_dino_head(
					in_dim=self.hparams.vit_backbone_dim, 
					out_dim=self.hparams.dino_head_prototype_dim,
					layers=3)
				set_trainable(self.ibot_head, trainable=True)

		# -- DINO-style student/teacher augmentations
		self.student_aug = None
		self.teacher_aug = None
		if self.hparams.use_dino_augmentation:
			image_size = self.hparams.image_dim[0]
			global_crops_scale = (0.4, 1.0)
			# global_transfo1 → teacher: always blur, no solarization
			self.teacher_aug = DINOClipAugmentation(
				crop_scale=global_crops_scale, image_size=image_size,
				gaussian_blur_p=1.0, solarization_p=0.0)
			# global_transfo2 → student: rare blur, solarization
			self.student_aug = DINOClipAugmentation(
				crop_scale=global_crops_scale, image_size=image_size,
				gaussian_blur_p=0.1, solarization_p=0.2)

		# -- teacher copies if using ema
		if self.hparams.use_ema_for_frame_encoder:
			self.teacher_frame_encoder, self.teacher_dino_head, self.teacher_ibot_head = get_teacher_encoder(
				hparams, 
				student_encoder=self.frame_encoder
			)
			self.ema_momentum = self.hparams.ema_momentum

		# LOSSES ###################################
		
		# -- actual losses for gradient
		self.recon_loss = CenterSharpReconstructionLoss(
			out_dim=self.hparams.vit_backbone_dim,
			predicted_temp=self.hparams.recon_predicted_temp,
			target_temp=self.hparams.recon_target_temp,
			center_momentum=0.99,
			loss=self.hparams.recon_loss_type
		 )
		self.dino_cce_loss = DinoLoss(
			out_dim=self.hparams.dino_head_prototype_dim, 
			student_temp=self.hparams.dino_student_temp, 
			teacher_temp=self.hparams.dino_teacher_temp, 
			center_momentum=self.hparams.dino_center_update_momentum
		)
		self.ibot_cce_loss = DinoLoss(
			out_dim=self.hparams.dino_head_prototype_dim, 
			student_temp=self.hparams.ibot_student_temp, 
			teacher_temp=self.hparams.ibot_teacher_temp, 
			center_momentum=self.hparams.ibot_center_update_momentum
		)

		# -- other losses just for logging
		self.smooth_l1_loss = torch.nn.SmoothL1Loss()
		self.l1_loss = torch.nn.L1Loss()
		self.baseline_mse_loss = torch.nn.MSELoss()

		# DEBUGGING CODE ##########################

		if self.hparams.debug_unused_parameters:
			self.used_parameters = set()
			self.parameters_not_to_check = set() 


	def forward(self, rgb_diff, condition, masks=None):

		# -- encode the RGB diff
		encoded_rgb_diff = self.encode_sequences(rgb_diff, encoder=self.motion_encoder, condition=condition, enable_grad=not self.hparams.freeze_difference_encoder, masks=masks)
		if self.hparams.use_only_cls_token: 
			encoded_rgb_diff = encoded_rgb_diff[:, :, 0]

		encoded_rgb_diff = self.linear_fc(encoded_rgb_diff)
		return encoded_rgb_diff
			

	def forward_loss_wrapper(self, batch):

		masks = None
		masks_weight = None

		# -- get encodings for frame sequences
		if self.hparams.use_preencoded_dataset:
			frame_sequences, frame_seq_encodings, _ = batch
		else:
			if self.hparams.use_ibot_masking:
				frame_sequences, ibot_masks = batch["frames"], batch["masks"]
				masks_weight = batch["masks_weight"][:, :-1]  # [B, T-1], match student frames
			else:
				frame_sequences = batch

			# -- apply separate DINO-style augmentations to student and teacher views
			if self.hparams.use_dino_augmentation:
				unaugmented_frames = frame_sequences
				student_frames = apply_clip_augmentation(frame_sequences, self.student_aug)
				teacher_frames = apply_clip_augmentation(frame_sequences, self.teacher_aug)
			else:
				student_frames = teacher_frames = unaugmented_frames = frame_sequences

			enable_grad = self.hparams.unfreeze_frame_encoder or self.hparams.use_ema_for_frame_encoder
			frame_seq_encodings, masks = self.encode_sequences(
				student_frames[:, :-1],
				self.frame_encoder,
				enable_grad,
				mask_ratio=self.mask_ratio,
				masks=ibot_masks[:, :-1] if self.hparams.use_ibot_masking else None,
				return_masks=True
			)
			# reshape from [B*(T-1), N_patches] → [B, T-1, N_patches] for downstream masking
			if masks is not None:
				masks = masks.reshape(frame_seq_encodings.shape[0], frame_seq_encodings.shape[1], -1)

		# -- seperate out into previous and next frame encodings
		previous_frame_encodings = frame_seq_encodings
		# next_frame_encodings = frame_seq_encodings[:, 1:].detach()

		if self.hparams.use_ema_for_frame_encoder:
			next_frame_encodings = self.encode_sequences(teacher_frames[:, 1:], self.teacher_frame_encoder, enable_grad=False).detach()

		# -- optionally use only cls token
		if self.hparams.use_only_cls_token:
			previous_frame_encodings = previous_frame_encodings[:, :, 0]
			next_frame_encodings = next_frame_encodings[:, :, 0]

		# -- compute encoded rgb diff
		rgb_diff_src = unaugmented_frames if self.hparams.rgb_diff_from_unaugmented else student_frames
		rgb_diff = get_rgb_diff(rgb_diff_src, self.hparams.use_full_frame)
		static_frames_mask = calculate_static_frames_mask(rgb_diff, threshold=self.hparams.rgb_diff_threshold)

		if skip_this_batch(static_frames_mask): return {"loss": torch.tensor(0., device=frame_sequences.device, requires_grad=True)}
		
		encoded_rgb_diff = None
		if not self.hparams.remove_motion_encoder:
			encoded_rgb_diff = self.forward(rgb_diff, previous_frame_encodings, masks=masks)

		# -- mask out the static frame sequences
		if static_frames_mask is not None:
			rgb_diff = rgb_diff[static_frames_mask]
			encoded_rgb_diff = encoded_rgb_diff[static_frames_mask]
			previous_frame_encodings = previous_frame_encodings[static_frames_mask]
			next_frame_encodings = next_frame_encodings[static_frames_mask]
			if masks is not None:
				masks = masks[static_frames_mask]
			if masks_weight is not None:
				masks_weight = masks_weight[static_frames_mask]

		# -- hierarchical prediction
		previous_frame_encodings, encoded_rgb_diff, next_frame_encodings = rollout_n_frames(
			previous=previous_frame_encodings,
			rgb_diff=encoded_rgb_diff,
			target=next_frame_encodings,
			n=self.hparams.rollout_n_frames
		)
		# align masks with the rolled-out time dimension
		T_out = previous_frame_encodings.shape[1]
		if masks is not None:
			masks = masks[:, :T_out, :]
		if masks_weight is not None:
			masks_weight = masks_weight[:, :T_out]

		# -- add difference in latent space
		if not self.hparams.remove_motion_encoder:
			predicted_next_frame_encodings = previous_frame_encodings + encoded_rgb_diff
		else:
			predicted_next_frame_encodings = previous_frame_encodings

		# -- calculate and return total loss with metrics
		total_loss, individual_losses = self.calculate_loss_for_backward(
			previous=previous_frame_encodings,
			predicted=predicted_next_frame_encodings,
			target=next_frame_encodings,
			rgb_diff=rgb_diff,
			masks=masks,
			masks_weight=masks_weight,
		)
		
		logging_metrics = self.get_visualization_metrics(
			previous=previous_frame_encodings,
			predicted=predicted_next_frame_encodings,
			target=next_frame_encodings
		)

		return {
			"loss": total_loss,				# used for backprop
			**individual_losses,			# logged for analysis
			**logging_metrics				# extra logging metrics
		}
	

	def ema_update(self):
		if self.hparams.use_fixed_dino_teacher:
			students = [self.dino_head]
			teachers = [self.teacher_dino_head]
		else:
			students = [self.frame_encoder]
			teachers = [self.teacher_frame_encoder]

			if self.hparams.use_dino_head:
				students.append(self.dino_head)
				teachers.append(self.teacher_dino_head)

			if self.hparams.use_seperate_ibot_head:
				students.append(self.ibot_head)
				teachers.append(self.teacher_ibot_head)

		update_teacher_using_ema(students, teachers, self.ema_momentum)
	

	def encode_sequences(self, frame_sequences, encoder, enable_grad=True, condition=None, masks=None, mask_ratio=None, return_masks=False):
		batch_size, num_frames, c, h, w = frame_sequences.shape

		if condition is not None:
			condition = condition.reshape(-1, condition.shape[-2], condition.shape[-1])
			if self.hparams.ignore_prefix_tokens_in_condition and not self.hparams.use_spatial_conditioning:
				# -- remove CLS token from condition
				condition = condition[:, 1:, :]
			assert condition.shape[0] == batch_size * num_frames, "Condition batch size does not match frame sequences batch size"

		# -- flatten batch and time dimensions together for masking
		if masks is not None and len(masks.shape) == 3:
			masks = masks.flatten(0,1)	
			
		context = torch.enable_grad if enable_grad else torch.no_grad
		with context():
			encoded_frames, masks = encode_images(
				frame_sequences.reshape(-1, c, h, w), 
				encoder=encoder, 
				encoder_name=self.encoder_name,
				condition=condition,
				masks=masks,
				mask_ratio=mask_ratio,
				return_masks=True
			)
		
		D = encoded_frames.shape[-1]
		
		if return_masks:
			return encoded_frames.reshape(batch_size, num_frames, -1, D), masks
		
		return encoded_frames.reshape(batch_size, num_frames, -1, D)
	

	def calculate_loss_for_backward(self, previous, predicted, target, rgb_diff, masks=None, masks_weight=None):
		total_loss = 0.0
		individual_losses = {}

		# RECON
		if self.hparams.use_mse_loss:
			recon_loss = self.recon_loss(predicted, target, 
				use_centering=self.hparams.recon_use_centering,
				use_sharpening=self.hparams.recon_use_sharpening,
			)
			total_loss += self.hparams.mse_loss_weight * recon_loss

			individual_losses.update({
				"mse_loss": recon_loss.detach(),
			})

		 # DINO
		if self.hparams.use_dino_loss:
			loss, metrics = self._compute_dino_style_loss(
				"dino",
				predicted[:, :, 0, :],  # CLS tokens
				target[:, :, 0, :],
				log_center=True,
			)
			total_loss += self.hparams.dino_loss_weight * loss
			individual_losses.update(metrics)

		# iBOT
		if self.hparams.use_ibot_loss:
			student_patches = predicted[:, :, 1:, :]   # [B, T, N_patches, D]
			teacher_patches = target[:, :, 1:, :]      # [B, T, N_patches, D]
			token_weights = None
			if masks is not None:
				# masks: [B, T, N_patches] bool, True = masked — compute loss only on masked positions
				# masks_weight: [B, T] = 1/n_masked_per_frame, pre-computed in collate; fall back to computing here if unavailable
				w = masks_weight if masks_weight is not None else (1.0 / masks.sum(dim=-1).clamp(min=1.0))
				token_weights = w.unsqueeze(-1).expand_as(masks)[masks]  # [total_masked]
				student_patches = student_patches[masks]   # [total_masked, D]
				teacher_patches = teacher_patches[masks]   # [total_masked, D]
			loss, metrics = self._compute_dino_style_loss(
				"ibot",
				student_patches,
				teacher_patches,
				log_center=True,
				token_weights=token_weights,
			)
			total_loss += self.hparams.ibot_loss_weight * loss
			individual_losses.update(metrics)

		# Motion
		if self.hparams.use_motion_loss:
			loss, metrics = self._compute_motion_loss(previous, target, rgb_diff)
			total_loss += self.hparams.motion_loss_weight * loss
			individual_losses.update(metrics)

		return total_loss, individual_losses


	# ----------------------------
	# Helpers
	# ----------------------------

	def _compute_dino_style_loss(self, prefix, student_tokens, teacher_tokens, log_center=False, token_weights=None):
		student_head, teacher_head = self.dino_head, self.teacher_dino_head

		if prefix == "ibot" and self.hparams.use_seperate_ibot_head:
			assert self.ibot_head is not None and self.teacher_ibot_head is not None
			student_head, teacher_head = self.ibot_head, self.teacher_ibot_head

		student_logits = student_head(student_tokens)
		with torch.no_grad():
			teacher_logits = teacher_head(teacher_tokens.detach())

		cce_loss = self.ibot_cce_loss if (prefix == "ibot" and self.hparams.use_seperate_ibot_center) else self.dino_cce_loss

		loss, entropy, kl = cce_loss(
			student_logits, teacher_logits,
			use_centering=self.hparams.use_centering,
			use_sharpening=self.hparams.use_sharpening,
			token_weights=token_weights,
		)

		metrics = {
			f"{prefix}_loss": loss.detach(),
			f"{prefix}_entropy": entropy.detach(),
			f"{prefix}_kl_div": kl.detach(),
		}
		if log_center:
			metrics[f"{prefix}_center_mean"] = cce_loss.center.mean().detach()
			metrics[f"{prefix}_center_std"] = cce_loss.center.std().detach()
			metrics[f"{prefix}_center_norm"] = cce_loss.center.norm().detach()

		return loss, metrics


	def _compute_motion_loss(self, previous, target, rgb_diff):
		pixel_diff_mean = torch.abs(rgb_diff).mean(dim=(-3, -2, -1)) + self.EPS
		embed_diff = torch.abs(previous - target)
		embed_diff_mean = embed_diff.mean(dim=(-2, -1))

		motion_loss = F.relu(
			self.hparams.min_embed_diff_per_pixel_diff - (embed_diff_mean / pixel_diff_mean)
		).mean()

		metrics = {
			"motion_capture_loss": motion_loss.detach(),
			"embed_diff_mean": embed_diff_mean.mean().detach(),
			"pixel_diff_mean": pixel_diff_mean.mean().detach(),
		}
		return motion_loss, metrics


	def get_visualization_metrics(self, previous, predicted, target):
		metrics = {}

		# -- variance and off diag covariance calculation
		if self.hparams.log_var_covar:
			variance, off_diag_covariance = calculate_var_covar(previous)
			metrics.update({
				"variance": variance.detach(),
				"off_diag_covariance": off_diag_covariance.detach()
			})

			variance, off_diag_covariance = calculate_var_covar(predicted)
			metrics.update({
				"predicted_variance": variance.detach(),
				"predicted_off_diag_covariance": off_diag_covariance.detach()
			})

			variance, off_diag_covariance = calculate_var_covar(target)
			metrics.update({
				"teacher_variance": variance.detach(),
				"teacher_off_diag_covariance": off_diag_covariance.detach()
			})

		# -- calculate (detached) l1 and smooth-l1 loss for evaluation
		with torch.no_grad():
			if self.hparams.log_baseline_losses:
				baseline_mse_loss = self.baseline_mse_loss(previous, target).detach()
				baseline_smooth_l1_loss = self.smooth_l1_loss(previous, target).detach()
				baseline_l1_loss = self.l1_loss(previous, target).detach()

				metrics.update({
					"baseline_mse_loss": baseline_mse_loss.detach(),
					"baseline_smooth_l1_loss": baseline_smooth_l1_loss.detach(),
					"baseline_l1_loss": baseline_l1_loss.detach()
				})

			smooth_l1_loss = self.smooth_l1_loss(predicted, target).detach()
			l1_loss = self.l1_loss(predicted, target).detach()


		metrics.update({
			"smooth_l1_loss": smooth_l1_loss.detach(),
			"l1_loss": l1_loss.detach(),
		})
			
		return metrics
	