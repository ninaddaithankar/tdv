# Optical Flow Evaluation

This code is adapted from the [CroCo v2](https://github.com/naver/croco) stereoflow codebase.
The original CroCo decoder has been replaced with a **Midway decoder** (`IterativeLatentMotion` + DPT head),
allowing a TDV/DINOv2/iBOT checkpoint to be used as the encoder.

## Files

| File | Description |
|---|---|
| `train_midway.py` | Main training script (use this) |
| `midway_flow_wrapper.py` | Builds the model from a TDV/DINOv2/iBot checkpoint |
| `croco/` | CroCo v2 repo — datasets, engine, criterion, DPT blocks |

## Model architecture

```
img1, img2
    ↓
frame_encoder  (shared, called separately for each image)
    ↓  all_layers: per-block patch features
IterativeLatentMotion  (student=img1, teacher=img2)
    ↓  dec_out: motion features
enc_layers + dec_out
    ↓
MidwayPixelwiseTaskWithDPT (DPT head)
    ↓
flow  [B, 2, H, W]
```

The encoder weights are loaded from a pretrained checkpoint; the decoder and DPT head are randomly initialised and trained.

## Data

Training uses the same datasets as CroCo-Flow. Default paths in `croco/stereoflow/datasets_flow.py` follow the `./data/stereoflow/` convention; override at runtime with `--dataset_dirs KEY=PATH`.

| Dataset | Default path |
|---|---|
| FlyingChairs | `./data/stereoflow/FlyingChairs` |
| FlyingThings (via SceneFlow) | `./data/stereoflow/SceneFlow/FlyingThings3D/...` |
| MPI-Sintel | `./data/stereoflow/MPI-Sintel` |
| TartanAir | `./data/stereoflow/TartanAir` |

## Training

Must be run from the repo root (`tdv-clean/`) so imports resolve correctly.

**Training dataset used for reported numbers:**
```
40*MPISintel('subtrain_cleanpass') + 40*MPISintel('subtrain_finalpass') +
4*FlyingThings('train_allpass') + 4*FlyingChairs('train')
```
Note: `SceneFlow` optical flow maps to the `FlyingThings` dataset class internally.

**Run command (2 GPUs):**
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --master_port=29797 --nproc_per_node=2 \
  eval/flow/train_midway.py flow \
  --pretrained /path/to/checkpoint \
  --output_dir /path/to/output \
  --dataset "40*MPISintel('subtrain_cleanpass')+40*MPISintel('subtrain_finalpass')+4*FlyingThings('train_allpass')+4*FlyingChairs('train')" \
  --val_dataset "MPISintel('subval_cleanpass')+MPISintel('subval_finalpass')" \
  --dataset_dirs \
    SceneFlow=/path/to/sceneflow \
    MPISintel=/path/to/mpi-sintel \
    FlyingChairs=/path/to/flying-chairs/data \
  --batch_size 8
```

All other hyperparameters match the defaults in `train_midway.py`, which are set to the best-performing configuration.

### Checkpoint types supported

`--pretrained` accepts any of:
- **TDV checkpoint** — loads `model.teacher_frame_encoder.*` (falls back to `model.frame_encoder.*`)
- **DINO/iBOT checkpoint** — loads `teacher.backbone.*`
- **Official DINOv2 pretrain** — loads flat `blocks.*` weights directly

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--pretrained` | required | Encoder checkpoint path |
| `--feature_levels` | `2 5 8 11` | Encoder block indices fed to `IterativeLatentMotion` |
| `--motion_dim` | `192` | Motion feature dimension |
| `--motion_depth` | `4` | Motion transformer depth |
| `--predictor_depth` | `4` | Predictor depth in `IterativeLatentMotion` |
| `--feature_block_type` | `cross` | Feature block type in `IterativeLatentMotion` |
| `--decoder_feature_mode` | `motion2-forward` | Feature mode fed to DPT |
| `--gating_type` | `pred-vector` | Gating type in predictor |
| `--gating_bias` | `4.0` | Gating bias |
| `--hooks_idx` | `15 19 31 35` | DPT hook indices into the combined feature list |
| `--patch_size` | `14` | ViT patch size for TDV (change to 16 for DINO/iBOT) |
| `--epochs` | `240` | Training epochs |
| `--img_per_epoch` | `30000` | Images sampled per epoch |
| `--amp` | `1` | Mixed precision |
| `--eval_every` | `5` | Validate every N epochs |
| `--val_overlap` | `0.5` | Tiled prediction overlap |
| `--freeze_encoder` | off | Freeze encoder, train only decoder + DPT |
| `--pretrained_encoder` | off | Use encoder's own pretrained weights instead of `--pretrained` |
| `--num_cls_tokens` | `1` | Tokens to strip from `all_layers` |
| `--dataset_dirs` | `[]` | Override dataset roots, e.g. `SceneFlow=/data/sf` |
