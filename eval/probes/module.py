import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy

from model.model_utils import encode_images

class ProbeLightningModule(pl.LightningModule):
	def __init__(self, encoder, encoder_name, input_dim, num_classes, probe_type, pooling="average", num_queries=1, num_heads=4, lr=1e-3, frame_aggregation="mean", context_length=1):
		super().__init__()
		self.save_hyperparameters(ignore=['encoder'])

		self.encoder_name = encoder_name
		self.pooling = pooling
		self.lr = lr
		self.probe_type = probe_type
		self.frame_aggregation = frame_aggregation  # "mean" or "concat"
		if probe_type not in ["attentive", "linear"]:
			raise ValueError(f"type={self.probe_type} is not supported for online probe evaluation, try 'linear' or 'attentive'. ")
		if frame_aggregation not in ["mean", "concat"]:
			raise ValueError(f"frame_aggregation={frame_aggregation} not supported, try 'mean' or 'concat'.")

		# frozen encoder
		self.encoder = encoder
		for p in self.encoder.parameters():
			p.requires_grad = False

		if probe_type == "attentive":
			# learnable query tokens
			self.query = nn.Parameter(torch.randn(1, num_queries, input_dim))  # (1, Q, D)

			# cross-attention layer
			self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)

		# classifier — concat mode widens the input by context_length
		classifier_input_dim = input_dim * context_length if frame_aggregation == "concat" else input_dim
		self.classifier = nn.Linear(classifier_input_dim, num_classes)

		self.train_top1 = Accuracy(top_k=1, task="multiclass", num_classes=num_classes)
		self.train_top5 = Accuracy(top_k=5, task="multiclass", num_classes=num_classes)
		self.val_top1 = Accuracy(top_k=1, task="multiclass", num_classes=num_classes)
		self.val_top5 = Accuracy(top_k=5, task="multiclass", num_classes=num_classes)


	def _encode_and_pool_patches(self, x):
		"""Encode a (B, C, H, W) batch and pool spatial patches → (B, D)."""
		features = encode_images(x, encoder=self.encoder, encoder_name=self.encoder_name)  # (B, S, D)
		if self.probe_type == "attentive":
			B = features.shape[0]
			query = self.query.expand(B, -1, -1)          # (B, Q, D)
			features, _ = self.attn(query, features, features)  # (B, Q, D)
		if self.pooling == "average":
			features = features.mean(dim=1)               # (B, D)
		else:
			features = features[:, 0, :]                  # (B, D) — CLS token
		return features


	def forward(self, x):
		with torch.no_grad():
			if x.dim() == 5:
				# Video input: (B, T, C, H, W) — encode each frame, then aggregate
				B, T, C, H, W = x.shape
				frame_features = self._encode_and_pool_patches(x.view(B * T, C, H, W))  # (B*T, D)
				frame_features = frame_features.view(B, T, -1)                           # (B, T, D)
				if self.frame_aggregation == "mean":
					features = frame_features.mean(dim=1)                                # (B, D)
				else:  # concat
					features = frame_features.reshape(B, -1)                             # (B, T*D)
			else:
				# Image input: (B, C, H, W)
				features = self._encode_and_pool_patches(x)                              # (B, D)

		return self.classifier(features)                                                 # (B, num_classes)
	

	def training_step(self, batch, batch_idx):
		x, y = batch
		logits = self(x)
		loss = F.cross_entropy(logits, y)

		self.train_top1.update(logits, y)
		self.train_top5.update(logits, y)

		self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
		self.log("train_top1", self.train_top1, prog_bar=True, on_step=False, on_epoch=True)
		self.log("train_top5", self.train_top5, prog_bar=True, on_step=False, on_epoch=True)

		return loss
	

	def validation_step(self, batch, batch_idx):
		x, y = batch
		logits = self(x)
		loss = F.cross_entropy(logits, y)

		self.val_top1.update(logits, y)
		self.val_top5.update(logits, y)

		self.log("val_loss", loss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
		self.log("val_top1", self.val_top1, prog_bar=True, on_step=False, on_epoch=True)
		self.log("val_top5", self.val_top5, prog_bar=True, on_step=False, on_epoch=True)


	def on_train_epoch_start(self):
		self.train_top1.reset()
		self.train_top5.reset()


	def on_validation_epoch_start(self):
		self.val_top1.reset()
		self.val_top5.reset()


	def configure_optimizers(self):
		return torch.optim.Adam(self.parameters(), lr=self.lr)
