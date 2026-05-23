"""
Standalone linear/attentive probe evaluation script.

Supports:
  - DINOv1 checkpoints  (--model_name dinov1_*)
  - TDV / DINOv2 checkpoints  (--model_name tdv_*)

Example — TDV checkpoint on SSv2:
    python eval/probes/train.py \
        --model_name tdv \
        --backbone_type dinov2 \
        --vit_backbone_size base \
        --resume_training_ckpt /path/to/last.ckpt \
        --probe_eval_dataset ssv2 \
        --probe_eval_data_dir /path/to/ssv2 \
        --num_classes 174 \
        --context_length 8 \
        --time_between_frames 0.25 \
        --probe_eval_max_epochs 5 \
        --probe_eval_bs 128 \
        --gpus -1

Example — DINOv1 checkpoint on SSv2:
    python eval/probes/train.py \
        --model_name dinov1_vit_base \
        --vit_backbone_size base \
        --patch_size 8 \
        --dino_v1_checkpoint_key teacher \
        --resume_training_ckpt /path/to/dino_checkpoint.pth \
        --probe_eval_dataset ssv2 \
        --probe_eval_data_dir /path/to/ssv2 \
        --num_classes 174 \
        --context_length 8 \
        --time_between_frames 0.25 \
        --probe_eval_max_epochs 5 \
        --probe_eval_bs 128 \
        --gpus -1
"""

import os
import sys
import argparse

import torch
import pytorch_lightning as pl
import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import model.cv.dinov1.vision_transformer as dino_v1_vits
from model.model_utils import load_tdv_encoder_from_checkpoint
from eval.probes.module import ProbeLightningModule
from eval.data_utils.data_module import ProbeDataModule


# ---------------------------------------------------------------------------
# Embed-dim lookup tables
# ---------------------------------------------------------------------------

DINOV2_EMBED_DIMS = {
    "xsmall": 192,
    "small":   384,
    "base":    768,
    "large":  1024,
    "huge":   1280,
    "giant":  1536,
}

DINOV1_EMBED_DIMS = {
    "vit_tiny":  192,
    "vit_small": 384,
    "vit_base":  768,
}


# ---------------------------------------------------------------------------
# Encoder loading
# ---------------------------------------------------------------------------

def load_dinov1_encoder(pretrained_weights, arch, patch_size, checkpoint_key, device):
    vit_arch = f"vit_{arch}"
    model = dino_v1_vits.__dict__[vit_arch](patch_size=patch_size, num_classes=0)
    print(f"DINOv1 model {vit_arch} patch{patch_size} built.")

    ckpt = torch.load(pretrained_weights, map_location="cpu", weights_only=False)
    if checkpoint_key in ckpt:
        print(f"Taking key '{checkpoint_key}' from checkpoint.")
        ckpt = ckpt[checkpoint_key]

    ckpt = {k.replace("module.", "").replace("backbone.", ""): v for k, v in ckpt.items()}
    msg = model.load_state_dict(ckpt, strict=False)
    print(f"Loaded DINOv1 weights: {msg}")
    return model.to(device).eval()


def load_encoder(args, device):
    if args.model_name.startswith("dinov1"):
        encoder = load_dinov1_encoder(
            pretrained_weights=args.resume_training_ckpt,
            arch=args.vit_backbone_size,
            patch_size=args.patch_size,
            checkpoint_key=args.dino_v1_checkpoint_key,
            device=device,
        )
        input_dim = DINOV1_EMBED_DIMS[f"vit_{args.vit_backbone_size}"]
        encoder_name = "dinov1"

    elif args.model_name.startswith("tdv"):
        encoder = load_tdv_encoder_from_checkpoint(
            ckpt=args.resume_training_ckpt,
            backbone_type=args.backbone_type,
            backbone_size=args.vit_backbone_size,
            key="teacher_frame_encoder",
        )
        encoder = encoder.to(device)
        input_dim = DINOV2_EMBED_DIMS[args.vit_backbone_size]
        encoder_name = args.backbone_type

    else:
        raise ValueError(
            f"Unknown model_name '{args.model_name}'. "
            "Must start with 'dinov1' or 'tdv'."
        )

    return encoder, input_dim, encoder_name


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ProbeDataModule reads args.temporal_diff; args uses time_between_frames
    args.temporal_diff = args.time_between_frames

    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_global_zero = int(os.environ.get("LOCAL_RANK", 0)) == 0

    if not args.no_wandb and is_global_zero:
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config=vars(args),
        )

    # -- load encoder
    encoder, input_dim, encoder_name = load_encoder(args, device)
    print(f"Encoder loaded. input_dim={input_dim}, encoder_name={encoder_name}")

    # -- build data module
    datamodule = ProbeDataModule(args)

    # -- build probe module
    probe_module = ProbeLightningModule(
        encoder=encoder,
        encoder_name=encoder_name,
        input_dim=input_dim,
        num_classes=args.num_classes,
        probe_type=args.probe_eval_type,
        pooling=args.pooling,
        lr=args.lr,
        frame_aggregation=args.frame_aggregation,
        context_length=args.context_length,
    )

    # -- wandb logger for pl
    logger = False
    if not args.no_wandb and is_global_zero:
        logger = pl.loggers.WandbLogger(experiment=wandb.run)

    # -- trainer
    trainer = pl.Trainer(
        max_epochs=args.probe_eval_max_epochs,
        check_val_every_n_epoch=args.probe_eval_max_epochs,
        devices=args.gpus,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        logger=logger,
        enable_checkpointing=False,
        enable_model_summary=False,
    )

    trainer.fit(probe_module, datamodule)

    val_top1 = trainer.callback_metrics.get("val_top1")
    val_top5 = trainer.callback_metrics.get("val_top5")
    if val_top1 is not None:
        print(f"val_top1: {val_top1.item() * 100:.2f}%")
    if val_top5 is not None:
        print(f"val_top5: {val_top5.item() * 100:.2f}%")


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Standalone linear/attentive probe evaluation")

    # -- run
    parser.add_argument("--run_name", type=str, default="probe_eval")
    parser.add_argument("--no_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="eval")

    # -- encoder (shared)
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model name prefix: 'dinov1_*' or 'tdv_*'")
    parser.add_argument("--resume_training_ckpt", type=str, required=True,
                        help="Path to the checkpoint to evaluate")

    # -- encoder (shared)
    parser.add_argument("--backbone_type", type=str, default="dinov2",
                        choices=["dinov2", "dinov1"])
    parser.add_argument("--vit_backbone_size", type=str, default="base",
                        choices=["tiny", "small", "base", "large", "huge", "giant"])
    parser.add_argument("--patch_size", type=int, default=16, choices=[8, 14, 16],
                        help="ViT patch size (8 or 16 for DINOv1; 14 for DINOv2)")

    # -- encoder (DINOv1 specific)
    parser.add_argument("--dino_v1_checkpoint_key", type=str, default="teacher",
                        choices=["teacher", "student"])

    # -- data
    parser.add_argument("--probe_eval_dataset", type=str, default="ssv2",
                        choices=["ssv2", "smth", "imagenet1k"])
    parser.add_argument("--probe_eval_data_dir", type=str, required=True)
    parser.add_argument("--context_length", type=int, default=8)
    parser.add_argument("--time_between_frames", type=float, default=0.25)
    parser.add_argument("--image_dim", type=int, nargs="+", default=[224, 224])
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--eval_train_num_samples_per_class", type=int, default=-1)
    parser.add_argument("--eval_val_num_samples_per_class", type=int, default=-1)

    # -- probe
    parser.add_argument("--probe_eval_type", type=str, default="linear",
                        choices=["linear", "attentive"])
    parser.add_argument("--probe_eval_bs", type=int, default=128)
    parser.add_argument("--probe_eval_max_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pooling", type=str, default="average", choices=["average", "cls"],
                        help="How to pool encoder tokens: 'average' mean-pools CLS+patches, 'cls' uses CLS token only")
    parser.add_argument("--frame_aggregation", type=str, default="concat",
                        choices=["mean", "concat"],
                        help="How to combine per-frame embeddings: "
                             "'mean' averages across frames (classifier input = D), "
                             "'concat' concatenates them (classifier input = T*D)")

    # -- hardware
    parser.add_argument("--gpus", type=int, default=-1,
                        help="-1 uses all available GPUs")

    return parser.parse_args()


if __name__ == "__main__":
    main()
