from types import SimpleNamespace
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from data.cv.imagenet_dataloader import ImageNetDataset
from data.cv.something_dataloader import SomethingDataset
from model.model_utils import get_cv_transforms


class ProbeDataModule(pl.LightningDataModule):
	def __init__(self, args):
		super().__init__()
		self.args = args
		self.dataset = args.probe_eval_dataset
		self.image_dims = args.image_dim
		self.dataset_dir = args.probe_eval_data_dir
		self.bs = args.probe_eval_bs
		self.train_dataset_samples_per_class = args.eval_train_num_samples_per_class
		self.val_dataset_samples_per_class = args.eval_val_num_samples_per_class
		self.num_workers = 20

		transform, _ = get_cv_transforms(self.dataset, self.image_dims, custom_image_normalization=True)

		if self.dataset == 'imagenet1k':
			self.train_ds = ImageNetDataset(self.args, 'train', transform, self.dataset_dir, self.train_dataset_samples_per_class)
			self.val_ds   = ImageNetDataset(self.args, 'val',   transform, self.dataset_dir, self.val_dataset_samples_per_class)
		elif self.dataset in ["ssv2", "smth"]:
			hparams = SimpleNamespace(**{
				"dataset_dir": self.dataset_dir,
				"context_length": self.args.context_length,
				"time_between_frames": getattr(self.args, "temporal_diff", None) or getattr(self.args, "time_between_frames", 0.25),
				"sampling_rate": 0.0,
				"preencode_dataset": False,
				"use_preencoded_dataset": False,
				"debug_mode": False,
				"model_name": "",
				"return_labels": True,
				"preprocess_data": False,
				"crop_all_samples": False,
				"use_raw_framerate": False,
			})
			self.train_ds = SomethingDataset(hparams, 'train', transform)
			self.val_ds   = SomethingDataset(hparams, 'val',   transform)
		
		else:
			raise ValueError(f"Dataset {self.dataset} is not supported for online probe evaluation.")

	def train_dataloader(self):
		return DataLoader(self.train_ds, batch_size=self.bs, shuffle=True, num_workers=self.num_workers, pin_memory=True, drop_last=True)

	def val_dataloader(self):
		return DataLoader(self.val_ds, batch_size=self.bs, shuffle=False, num_workers=self.num_workers, pin_memory=True)
