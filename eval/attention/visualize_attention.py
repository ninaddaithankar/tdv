#!/usr/bin/env python3
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import wandb

from data.cv.imagenet_dataloader import ImageNetDataset
from hparams.args import get_args
from model import model_utils

import model.cv.dinov1.vision_transformer as vits


# ---------------------------------------------------------------------------
# Shared utils
# ---------------------------------------------------------------------------

def denorm_imagenet(x: torch.Tensor) -> torch.Tensor:
	mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(3, 1, 1)
	std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(3, 1, 1)
	return (x * std + mean).clamp(0, 1)


def _render_fig(fig) -> np.ndarray:
	fig.canvas.draw()
	w, h = fig.canvas.get_width_height()
	arr = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
	plt.close(fig)
	return arr


# ---------------------------------------------------------------------------
# Attention visualization helpers
# ---------------------------------------------------------------------------

def attn_to_heatmap(attn_grid: torch.Tensor, img_h: int, img_w: int) -> torch.Tensor:
	attn = attn_grid.unsqueeze(0).unsqueeze(0)
	attn = TF.resize(attn, (img_h, img_w), InterpolationMode.BILINEAR)[0, 0]
	attn = attn - attn.min()
	return attn / (attn.max() + 1e-6)


def make_attn_plot(image_tensor: torch.Tensor, heatmap: torch.Tensor = None) -> np.ndarray:
	img = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
	fig, ax = plt.subplots()
	ax.imshow(img)
	if heatmap is not None:
		ax.imshow(heatmap.detach().cpu().numpy(), alpha=0.4, cmap="jet")
	ax.axis("off")
	fig.tight_layout(pad=0)
	return _render_fig(fig)


# ---------------------------------------------------------------------------
# Patch PCA helpers
# ---------------------------------------------------------------------------

def extract_patch_features(model: nn.Module, images: torch.Tensor, encoder_name: str) -> torch.Tensor:
	"""Returns (B, N, D) last-layer patch tokens."""
	with torch.no_grad():
		if encoder_name == "dinov1":
			# get_intermediate_layers returns [(B, N+1, D)]; index 0 is CLS
			feats = model.get_intermediate_layers(images, n=1)[0]  # (B, N+1, D)
			return feats[:, 1:]  # drop CLS → (B, N, D)
		else:  # dinov2 / TDV
			out = model(images, is_training=True)
			return out["x_norm_patchtokens"]  # (B, N, D)


def pca_color_patches(patch_features: torch.Tensor) -> torch.Tensor:
	"""
	patch_features: (B, N, D)
	Returns: (B, H', W', 3) float32 in [0, 1], where H'*W' = N.

	Per-image mean is subtracted before PCA so the projection captures
	within-image spatial variation rather than between-image variation.
	This matters for models like TDV whose patch features have low spatial
	variance on static images but high variance across images.
	"""
	B, N, D = patch_features.shape
	side = int(N ** 0.5)
	assert side * side == N, f"N={N} patches is not a perfect square"

	# remove per-image mean so PCA focuses on within-image spatial structure
	centered = patch_features - patch_features.mean(dim=1, keepdim=True)  # (B, N, D)
	flat = centered.reshape(B * N, D).float()

	_, _, V = torch.pca_lowrank(flat, q=3, niter=4)
	pca = flat @ V  # (B*N, 3)

	# normalise each component independently to [0, 1]
	lo = pca.min(dim=0).values
	hi = pca.max(dim=0).values
	pca = (pca - lo) / (hi - lo + 1e-6)

	# flip each component per-image so the majority (background) always maps to low
	# values — makes colors consistent across checkpoints despite arbitrary PCA sign
	pca_3d = pca.reshape(B, N, 3)
	for c in range(3):
		flip = (pca_3d[:, :, c].median(dim=1).values > 0.5).view(B, 1)
		pca_3d[:, :, c] = torch.where(flip, 1.0 - pca_3d[:, :, c], pca_3d[:, :, c])

	return pca_3d.reshape(B, side, side, 3).cpu()


def make_combined_plot(orig_np: np.ndarray, attn_overlay_np: np.ndarray, pca_up_np: np.ndarray) -> np.ndarray:
	"""original | attention overlay | pure PCA — all as HxWx3 uint8 numpy arrays."""
	fig, axes = plt.subplots(1, 3, figsize=(12, 4))
	axes[0].imshow(orig_np);         axes[0].axis("off"); axes[0].set_title("original")
	axes[1].imshow(attn_overlay_np); axes[1].axis("off"); axes[1].set_title("attention")
	axes[2].imshow(pca_up_np);       axes[2].axis("off"); axes[2].set_title("patch PCA")
	fig.tight_layout(pad=0)
	return _render_fig(fig)


def make_pca_plot(pca_img: torch.Tensor, original_img: torch.Tensor) -> np.ndarray:
	"""
	pca_img:      (H', W', 3) float32 in [0, 1]
	original_img: (3, H, W)   float32 in [0, 1]
	Returns 2-panel: original | pure PCA (upsampled).
	"""
	H, W = original_img.shape[1], original_img.shape[2]

	pca_chw   = pca_img.permute(2, 0, 1).unsqueeze(0)
	pca_up_np = torch.nn.functional.interpolate(pca_chw, size=(H, W), mode="bilinear",
	                                             align_corners=False)[0].detach().cpu().permute(1, 2, 0).numpy()
	orig_np   = original_img.detach().cpu().permute(1, 2, 0).numpy()

	fig, axes = plt.subplots(1, 2, figsize=(8, 4))
	axes[0].imshow(orig_np);   axes[0].axis("off"); axes[0].set_title("original")
	axes[1].imshow(pca_up_np); axes[1].axis("off"); axes[1].set_title("patch PCA")
	fig.tight_layout(pad=0)
	return _render_fig(fig)


# ---------------------------------------------------------------------------
# Encoder loading
# ---------------------------------------------------------------------------

def load_encoder(model_name, pretrained_weights, backbone_type, backbone_size='base', patch_size=16, checkpoint_key="teacher", device='cuda'):
	if model_name.startswith("dinov1"):
		encoder = load_dino_v1(pretrained_weights, checkpoint_key=checkpoint_key, arch=f"vit_{backbone_size}", patch_size=patch_size, device=device)
	elif model_name.startswith("tdv"):
		encoder = model_utils.load_tdv_encoder_from_checkpoint(pretrained_weights, backbone_type, backbone_size, key="teacher_frame_encoder")
	else:
		raise ValueError(f"Unknown model name {model_name}")
	return encoder.to(device).eval()


def load_dino_v1(pretrained_weights, checkpoint_key="teacher", arch="vit_base", patch_size=16, device="cuda"):
	model = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
	print(f"Model {arch} {patch_size}x{patch_size} built.")
	ckpt = torch.load(pretrained_weights, map_location="cpu", weights_only=False)
	if checkpoint_key in ckpt:
		print(f"Taking key {checkpoint_key}")
		ckpt = ckpt[checkpoint_key]
	ckpt = {k.replace("module.", "").replace("backbone.", ""): v for k, v in ckpt.items()}
	msg = model.load_state_dict(ckpt, strict=False)
	print(f"Loaded pretrained weights: {msg}")
	return model.to(device).eval()


# ---------------------------------------------------------------------------
# Dataloader
# ---------------------------------------------------------------------------

def build_imagenet_dataloader(args, image_size, batch_size, num_workers, max_images=None, seed=42):
	transform = T.Compose([
		T.Resize(256, interpolation=3),
		T.CenterCrop(args.image_dim[0]),
		T.ToTensor(),
		T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
	])
	dataset = ImageNetDataset(args, split='val', transform=transform,
	                          n_samples_per_class=args.eval_val_num_samples_per_class,
	                          dataset_dir=args.probe_eval_data_dir)
	if max_images is not None and max_images < len(dataset):
		rng = random.Random(seed)
		indices = sorted(rng.sample(range(len(dataset)), max_images))
		dataset = torch.utils.data.Subset(dataset, indices)
	return DataLoader(dataset, batch_size=batch_size, shuffle=False,
	                  num_workers=num_workers, pin_memory=True)


# ---------------------------------------------------------------------------
# Visualization loops
# ---------------------------------------------------------------------------

def visualize_attention(model, dataloader, device, max_batches, log_every_n_batches,
                        max_images_per_batch, save_dir="", use_wandb=True):
	global_img_idx = 0
	for batch_idx, (images, targets) in enumerate(dataloader):
		if max_batches != -1 and batch_idx >= max_batches:
			print(f"[INFO] Reached max_batches={max_batches}. Stopping.")
			break

		images = images.to(device, non_blocking=True)
		B, C, H, W = images.shape

		with torch.no_grad():
			attn = model.get_last_selfattention(images)  # (B, heads, N, N)

		n_heads, N = attn.shape[1], attn.shape[2]
		num_patches = N - 1
		side = int(num_patches ** 0.5)
		if side * side != num_patches:
			raise ValueError(f"Cannot reshape {num_patches} patches into a square grid.")

		if batch_idx % log_every_n_batches == 0:
			log_imgs = []
			num_to_log = min(B, max_images_per_batch) if max_images_per_batch != -1 else B
			for i in range(num_to_log):
				cls_to_patches = attn[i, :, 0, 1:].mean(0).reshape(side, side)
				heatmap  = attn_to_heatmap(cls_to_patches, H, W)
				img_vis  = denorm_imagenet(images[i].clone())
				overlay  = make_attn_plot(img_vis, heatmap)
				original = make_attn_plot(img_vis)
				label    = int(targets[i])

				if save_dir:
					Image.fromarray(overlay).save(os.path.join(save_dir, f"img_{global_img_idx:04d}_overlay.png"))
					Image.fromarray(original).save(os.path.join(save_dir, f"img_{global_img_idx:04d}_original.png"))
				if use_wandb:
					log_imgs.append(wandb.Image(overlay,  caption=f"overlay, label={label}"))
					log_imgs.append(wandb.Image(original, caption=f"original, label={label}"))
				global_img_idx += 1

			if use_wandb and log_imgs:
				wandb.log({"attention_maps": log_imgs, "batch_idx": batch_idx})

	print("[INFO] Finished attention visualization.")


def visualize_patch_pca(model, dataloader, device, encoder_name, max_batches,
                        log_every_n_batches, max_images_per_batch, save_dir="", use_wandb=True):
	global_img_idx = 0
	for batch_idx, (images, targets) in enumerate(dataloader):
		if max_batches != -1 and batch_idx >= max_batches:
			print(f"[INFO] Reached max_batches={max_batches}. Stopping.")
			break

		images = images.to(device, non_blocking=True)
		B = images.shape[0]

		patch_feats = extract_patch_features(model, images, encoder_name)  # (B, N, D)
		pca_imgs    = pca_color_patches(patch_feats)                        # (B, H', W', 3)

		if batch_idx % log_every_n_batches == 0:
			H, W = images.shape[2], images.shape[3]
			log_imgs = []
			num_to_log = min(B, max_images_per_batch) if max_images_per_batch != -1 else B
			for i in range(num_to_log):
				img_vis = denorm_imagenet(images[i].clone())
				label   = int(targets[i])

				# upsample pca grid once, reuse for both saving and wandb
				pca_chw = pca_imgs[i].permute(2, 0, 1).unsqueeze(0)
				pca_up  = torch.nn.functional.interpolate(pca_chw, size=(H, W), mode="bilinear",
				                                          align_corners=False)[0]
				pca_up_np = (pca_up.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

				if save_dir:
					Image.fromarray(pca_up_np).save(os.path.join(save_dir, f"img_{global_img_idx:04d}_pca.png"))
				if use_wandb:
					plot = make_pca_plot(pca_imgs[i], img_vis)
					log_imgs.append(wandb.Image(plot, caption=f"pca, label={label}"))
				global_img_idx += 1

			if use_wandb and log_imgs:
				wandb.log({"patch_pca": log_imgs, "batch_idx": batch_idx})

	print("[INFO] Finished patch PCA visualization.")


def visualize_combined(model, dataloader, device, encoder_name, max_batches,
                       log_every_n_batches, max_images_per_batch, save_dir="", use_wandb=True):
	"""Single-pass loop saving: original, attention, pca3, pc1 grayscale, combined."""
	global_img_idx = 0
	interp = torch.nn.functional.interpolate
	for batch_idx, (images, targets) in enumerate(dataloader):
		if max_batches != -1 and batch_idx >= max_batches:
			print(f"[INFO] Reached max_batches={max_batches}. Stopping.")
			break

		images = images.to(device, non_blocking=True)
		B, C, H, W = images.shape

		with torch.no_grad():
			attn        = model.get_last_selfattention(images)              # (B, heads, N, N)
			patch_feats = extract_patch_features(model, images, encoder_name)  # (B, N, D)

		num_patches = attn.shape[2] - 1
		side_attn   = int(num_patches ** 0.5)
		pca_imgs    = pca_color_patches(patch_feats)                        # (B, H', W', 3)

		if batch_idx % log_every_n_batches == 0:
			log_imgs = []
			num_to_log = min(B, max_images_per_batch) if max_images_per_batch != -1 else B
			for i in range(num_to_log):
				img_vis = denorm_imagenet(images[i].clone())
				label   = int(targets[i])
				orig_np = (img_vis.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

				# attention overlay
				cls_to_patches = attn[i, :, 0, 1:].mean(0).reshape(side_attn, side_attn)
				heatmap        = attn_to_heatmap(cls_to_patches, H, W)
				attn_overlay   = make_attn_plot(img_vis, heatmap)

				# pca3 upsampled (RGB)
				pca_chw   = pca_imgs[i].permute(2, 0, 1).unsqueeze(0)       # (1, 3, H', W')
				pca_up    = interp(pca_chw, size=(H, W), mode="bilinear", align_corners=False)[0]
				pca_up_np = (pca_up.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

				# pc1 grayscale
				pc1_chw  = pca_imgs[i, :, :, 0].unsqueeze(0).unsqueeze(0)   # (1, 1, H', W')
				pc1_up   = interp(pc1_chw, size=(H, W), mode="bilinear", align_corners=False)[0, 0]
				pc1_np   = (pc1_up.detach().cpu().numpy() * 255).astype(np.uint8)

				if save_dir:
					Image.fromarray(orig_np).save(os.path.join(save_dir, f"img_{global_img_idx:04d}_original.png"))
					Image.fromarray(attn_overlay).save(os.path.join(save_dir, f"img_{global_img_idx:04d}_overlay.png"))
					Image.fromarray(pca_up_np).save(os.path.join(save_dir, f"img_{global_img_idx:04d}_pca3.png"))
					Image.fromarray(pc1_np, mode="L").save(os.path.join(save_dir, f"img_{global_img_idx:04d}_pc1.png"))
					combined = make_combined_plot(orig_np, attn_overlay, pca_up_np)
					Image.fromarray(combined).save(os.path.join(save_dir, f"img_{global_img_idx:04d}_combined.png"))

				if use_wandb:
					combined = make_combined_plot(orig_np, attn_overlay, pca_up_np)
					log_imgs.append(wandb.Image(combined, caption=f"combined, label={label}"))
				global_img_idx += 1

			if use_wandb and log_imgs:
				wandb.log({"combined": log_imgs, "batch_idx": batch_idx})

	print("[INFO] Finished combined visualization.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
	args = get_args()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"[INFO] Using device: {device}")

	use_wandb = not args.no_wandb
	if use_wandb:
		wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))

	if args.save_dir:
		os.makedirs(args.save_dir, exist_ok=True)

	encoder = load_encoder(args.model_name, args.resume_training_ckpt, args.backbone_type,
	                       args.vit_backbone_size, args.patch_size, args.dino_v1_checkpoint_key,
	                       device=device)
	for p in encoder.parameters():
		p.requires_grad = False

	encoder_name = "dinov1" if args.model_name.startswith("dinov1") else "dinov2"

	dataloader = build_imagenet_dataloader(
		args,
		image_size=args.image_dim[0],
		batch_size=args.batch_size_per_device,
		num_workers=args.num_workers,
		max_images=args.max_images,
		seed=args.viz_seed,
	)

	common = dict(max_batches=args.max_batches, log_every_n_batches=args.log_every_n_batches,
	              max_images_per_batch=args.max_images_per_batch, save_dir=args.save_dir,
	              use_wandb=use_wandb)

	if args.viz_mode == "both":
		visualize_combined(model=encoder, dataloader=dataloader, device=device,
		                   encoder_name=encoder_name, **common)
	elif args.viz_mode == "attention":
		visualize_attention(model=encoder, dataloader=dataloader, device=device, **common)
	elif args.viz_mode == "pca":
		visualize_patch_pca(model=encoder, dataloader=dataloader, device=device,
		                    encoder_name=encoder_name, **common)

	if use_wandb:
		wandb.finish()


if __name__ == "__main__":
	main()
