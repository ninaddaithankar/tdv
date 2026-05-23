import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)

from pytorch_lightning import LightningModule
from deep_sort_realtime.deepsort_tracker import DeepSort
from eval.tracking.deepsort.metrics import compute_mot_metrics
from model.model_utils import encode_images


class DeepSORTModule(LightningModule):
	def __init__(self, encoder, encoder_name, log_metrics, step, config):
		super().__init__()
		self.save_hyperparameters()
		self.config = config
		self.log_metrics = log_metrics
		self.step = step

		self.encoder_name = encoder_name
		self.encoder = encoder
		self.encoder.eval()
		for p in self.encoder.parameters():
			p.requires_grad = False

		# DeepSORT init
		self.tracker = DeepSort(
			max_age=30,
			n_init=3,
			max_cosine_distance=0.4,
			nn_budget=100
		)

	def extract_embedding(self, crops):
		with torch.no_grad():
			feats = encode_images(crops, self.encoder_name, self.encoder)  # (B, N+1, D) or (B, D, H, W)
			if feats.ndim == 3:  # ViT: use patch mean
				return F.normalize(feats[:, 1:].mean(dim=1), dim=1)
			elif feats.ndim == 4:
				return F.normalize(feats.mean(dim=(2, 3)), dim=1)
			else:
				raise ValueError("Unexpected encoder output")
			
	def validation_step(self, batch, batch_idx):
		frames, gt_tracks = batch["video"], batch["gt_tracks"]
		pred_tracks = []

		for t, frame in enumerate(frames):  # T x C x H x W
			detections = batch["boxes"][t]  # List of [x1, y1, x2, y2]
			crops = self.crop_and_resize(frame, detections)  # (N, C, 224, 224)
			crops = crops.to(self.device)
			embeddings = self.extract_embedding(crops)

			# convert boxes to xywh
			boxes_xywh = [([x1, y1, x2 - x1, y2 - y1], conf, None) for x1, y1, x2, y2, conf in detections]

			# run DeepSORT update
			tracks = self.tracker.update_tracks(
				raw_detections=boxes_xywh,
				embeds=embeddings.cpu().numpy()
			)

			for track in tracks:
				if track.is_confirmed():
					pred_tracks.append((t+1, track.track_id, *track.to_ltrb()))

		metrics = compute_mot_metrics(pred_tracks, gt_tracks)
		print(f"Validation metrics: {metrics}")
		self.log_metrics(metrics, step=self.step)

	def crop_and_resize(self, frame: torch.Tensor, detections, size=(224, 224)) -> torch.Tensor:
		"""
		frame: (C, H, W) uint8/float tensor
		detections: list of [x1, y1, x2, y2]
		size: resize target (H, W)

		Returns: (N, C, H, W) tensor of resized crops
		"""
		C, H, W = frame.shape
		crops = []

		for (x1, y1, x2, y2, conf) in detections:
			# Cast to int (detections may be floats)
			x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

			# Clamp to frame boundaries
			x1, y1 = max(0, x1), max(0, y1)
			x2, y2 = min(W, x2), min(H, y2)

			# Skip invalid/empty boxes
			if x2 <= x1 or y2 <= y1:
				continue

			# Crop region: (C, h, w)
			crop = frame[:, y1:y2, x1:x2].unsqueeze(0)  # (1, C, h, w)

			# Resize to target size
			crop = F.interpolate(crop.float(), size=size, mode="bilinear", align_corners=False)

			crops.append(crop.squeeze(0))  # (C, H, W)

		if len(crops) == 0:
			# Return empty tensor if no valid crops
			return torch.empty(0, C, *size, device=frame.device)

		return torch.stack(crops, dim=0)  # (N, C, H, W)

