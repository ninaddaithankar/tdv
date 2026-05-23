# TDV Hyperparameters Reference

All configurable hyperparameters for the **Temporal Difference Vision (TDV)** framework.
Defined in [`get_args()`](args.py) and set via command-line arguments.

---

## Core Experiment Settings

| Argument | Description | Default |
|----------|-------------|---------|
| `--run_name` | Unique identifier for the run (used in logging and checkpoints). | `"test"` |
| `--modality` | Data modality. Most code assumes `CV`. | `"CV"` |
| `--model_name` | Model type or name. | `"tdv"` |

---

## Vision Parameters

| Argument | Description | Default |
|----------|-------------|---------|
| `--vit_backbone_size` | ViT backbone size: `xsmall`, `small`, `base`, `large`, `huge`, `giant`. | `"base"` |
| `--backbone_type` | Encoder backbone: `dinov1`, `dinov2`, `vae`, `mae`. | `"dinov2"` |
| `--time_between_frames` | Temporal spacing (seconds) between frames. | `0.25` |
| `--sampling_rate` | Frame sampling rate override; overrides `time_between_frames`. | `0.0` |
| `--use_raw_framerate` | Use native dataset framerate with sampling rate of 1. | `False` |
| `--vae_normalization` | Normalize input to `[-1, 1]` for VAEs. | `False` |

---

## Model Architecture

| Argument | Description | Default |
|----------|-------------|---------|
| `--context_length` | Total number of frames supported by the model. | `16` |
| `--change_encoder_type` | Architecture type for the motion/change encoder. | `"dinoViT_xattn_base14"` |
| `--embedding_dim` | Embedding dimension; auto-set from backbone size for DINOv2/MAE. | `768` |

---

## Hardware

| Argument | Description | Default |
|----------|-------------|---------|
| `--gpus` | GPU list; `-1` uses all GPUs. Use `"[0,1]"` for specific GPUs. | `"-1"` |
| `--distributed_strategy` | Parallel strategy: `ddp`, `ddp_spawn`, `fsdp_native`. | `"ddp"` |

---

## Training

| Argument | Description | Default |
|----------|-------------|---------|
| `--epochs` | Total training epochs; `-1` relies on `max_steps`. | `-1` |
| `--peak_learning_rate` | Peak LR after warmup. | `1e-4` |
| `--batch_size_per_device` | Per-GPU batch size. Effective BS = `gpus × bs × accumulate_grad_batches`. | `4` |
| `--accumulate_grad_batches` | Gradient accumulation steps. | `1` |
| `--gradient_clip_val` | Gradient clipping threshold; `0` disables. | `1.0` |
| `--is_random_seed` | Use a random seed instead of fixed seed `786`. | `False` |
| `--deterministic` | Force deterministic ops (slower but reproducible). | `False` |
| `--execution_mode` | `pretrain` or `finetune`. | `"pretrain"` |
| `--finetuning_model_ckpt` | Checkpoint path when finetuning. | `None` |

---

## Metrics

| Argument | Description | Default |
|----------|-------------|---------|
| `--num_classes` | Number of target classes (required if using metrics). | `-1` |

---

## Optimizer and LR Scheduler

| Argument | Description | Default |
|----------|-------------|---------|
| `--optimizer` | Optimizer: `adamw`, `lars`, or `stableadamw`. | `"adamw"` |
| `--weight_decay` | Weight decay. | `0.01` |
| `--beta1` | AdamW first moment coefficient. | `0.9` |
| `--beta2` | AdamW second moment coefficient. | `0.999` |
| `--lars_trust_coeff` | LARS trust coefficient. | `0.001` |
| `--lars_exclude_bias_bn_wd` | Exclude bias and batch norm from LARS adaptation and weight decay. | `False` |
| `--lr_scaling_rule` | Scale LR linearly: `LR = base_lr × effective_bs / 256`. | `False` |
| `--min_lr_scale` | Minimum LR divisor at end of cosine decay. | `10` |
| `--max_steps` | Maximum training steps. | `600,000` |
| `--max_scheduling_steps` | Steps used for LR scheduling; defaults to `max_steps` if `-1`. | `600,000` |
| `--warm_up_steps` | Linear warmup duration (steps). | `10,000` |
| `--warm_up_base_lr_divider` | Warmup starting LR divider; `-1` warms up from 0. | `-1` |

---

## Dataset and Dataloader

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset_name` | Dataset identifier (e.g. `ucf101`, `ssv2`, `k400`, `aggr`). | `"ucf101"` |
| `--dataset_dir` | Root directory for dataset. | `""` |
| `--aggr_datasets_list` | Comma-separated dataset names to aggregate. | `""` |
| `--aggr_dataset_dirs` | Comma-separated directories for each dataset listed in aggregate dataset list. | `""` |
| `--num_workers` | Dataloader workers per GPU. | `4` |
| `--image_dim` | Image size `[H W]`. | `[224, 224]` |
| `--patch_size` | ViT patch size. | `14` |
| `--no_randomness_dataloader` | Disable dataloader randomness (sample from start only). | `False` |
| `--preprocess_data` | Center crop images to remove black bars. | `False` |
| `--crop_all_samples` | Apply cropping to all samples (requires `--preprocess_data`). | `False` |
| `--custom_image_normalization` | Normalize per dataset mean/std. | `False` |
| `--ffprobe_path` | Path to ffprobe binary (for video loading). | `""` |
| `--test_split_pct` | Fraction held out as test set. | `0` |
| `--train_data_pct` | Fraction of training data to use (fixed subset). | `1.0` |
| `--limit_train_batches` | Fraction or count of train batches per epoch. | `1.0` |
| `--limit_val_batches` | Fraction or count of validation batches. | `1.0` |
| `--limit_test_batches` | Fraction or count of test batches. | `1.0` |
| `--val_sanity` | Number of sanity validation steps before training. | `0` |
| `--val_check_interval` | Validation frequency within an epoch (1.0 = once per epoch). | `1.0` |
| `--check_val_every_n_epoch` | Validate every N epochs. | `1` |
| `--preencode_dataset` | Encode and cache dataset features to disk. | `False` |
| `--use_preencoded_dataset` | Load from pre-encoded HDF5 dataset. | `False` |
| `--use_preprocessed_finevideo` | Use preprocessed FineVideo dataset. | `False` |
| `--use_preprocessed_ego4d` | Use preprocessed Ego4D dataset. | `False` |
| `--hdf5_file_path` | Path to pre-encoded HDF5 dataset file. | `""` |

---

## WandB

| Argument | Description | Default |
|----------|-------------|---------|
| `--no_wandb` | Disable W&B logging (logs to console file instead). | `False` |
| `--wandb_project` | W&B project name. | `"tdv"` |
| `--wandb_tags` | List of tags to attach to the W&B run. | `None` |
| `--wandb_offline` | Enable offline W&B mode. | `False` |
| `--wandb_watch` | Enable W&B gradient/weight watching (expensive). | `False` |
| `--wandb_watch_log_freq` | Steps between W&B watch logs. | `1000` |

---

## Logging

| Argument | Description | Default |
|----------|-------------|---------|
| `--console_log_filename` | Filename for console log (used when `--no_wandb` is set). | `"console.log"` |
| `--log_model_archi` | Print and log model architecture. | `False` |
| `--log_gradients` | Log gradient norms to W&B at every step. | `False` |
| `--log_every_n_steps` | Lightning logger step frequency. | `50` |

---

## Checkpointing

| Argument | Description | Default |
|----------|-------------|---------|
| `--resume_training_ckpt` | Checkpoint path to resume training from. | `""` |
| `--checkpoint_monitor_string` | Metric to monitor for saving best checkpoints. | `"valid/loss"` |
| `--checkpoint_monitor_mode` | `min` (loss) or `max` (accuracy). | `"min"` |

---

## Precision and Speed

| Argument | Description | Default |
|----------|-------------|---------|
| `--float_precision` | Precision mode: `16-mixed`, `bf16-mixed`, `32-true`. | `"32-true"` |
| `--set_matmul_precision` | Float32 matmul precision: `medium`, `high`, `highest`. | `None` |
| `--compile_model` | Compile model with `torch.compile()`. | `False` |

---

## Slurm

| Argument | Description | Default |
|----------|-------------|---------|
| `--is_slurm_run` | Set when running under Slurm. Enforces that debug flags are off. | `False` |

---

## Debugging

| Argument | Description | Default |
|----------|-------------|---------|
| `--debug_mode` | Enable debug mode: tiny dataset, no W&B, anomaly detection on. | `False` |
| `--fast_dev_run` | Run one train + one val batch for sanity check. | `False` |
| `--overfit_batches` | Overfit to a fixed number or fraction of batches. | `0.0` |
| `--profiler` | PyTorch Lightning profiler: `simple` or `advanced`. | `""` |
| `--no_shuffle` | Disable dataloader shuffling. | `False` |
| `--detect_anomaly` | Enable PyTorch autograd anomaly detection. | `False` |
| `--find_unused_parameters` | Enable DDP `find_unused_parameters` (debug only). | `False` |
| `--debug_unused_parameters` | Track which parameters are unused (for diagnosing DDP warnings). | `False` |
| `--manual_gc_collect_every_n_steps` | Manually call `gc.collect()` every N steps to reduce CPU RAM growth. | `-1` |

---

## TDV-Specific

### Architecture

| Argument | Description | Default |
|----------|-------------|---------|
| `--use_ema_for_frame_encoder` | Enable EMA teacher–student for the frame encoder. | `False` |
| `--ema_momentum` | EMA momentum for teacher updates. | `0.9999` |
| `--remove_motion_encoder` | Use only the frame encoder (no motion encoder). | `False` |
| `--use_rope` | Use RoPE positional encoding. | `False` |
| `--use_ape` | Use absolute positional encoding (APE); combinable with RoPE. | `False` |
| `--use_different_init_teacher` | Use a different random seed for teacher initialization. | `False` |
| `--load_teacher_from_checkpoint` | Load teacher frame encoder from a checkpoint path. | `""` |
| `--use_fixed_dino_teacher` | Use a frozen pretrained DINOv2 teacher (no EMA updates). | `False` |

### Masking

| Argument | Description | Default |
|----------|-------------|---------|
| `--frame_encoder_mask_ratio` | Fraction of frame encoder tokens to mask. | `0.0` |
| `--use_ibot_masking` | Use iBOT-style block masking for the frame encoder. | `False` |
| `--ibot_mask_sample_probability` | Probability of masking a sample in a batch. | `0.5` |
| `--ibot_mask_ratio_min_max` | Min/max mask ratio range for iBOT masking. | `[0.4, 0.8]` |


### Frame Encoder

| Argument | Description | Default |
|----------|-------------|---------|
| `--unfreeze_frame_encoder` | Allow the frame encoder to be trained. | `False` |
| `--use_only_cls_token` | Use only CLS token from the frame encoder for loss computation. | `False` |
| `--load_without_weights` | Load frame encoder architecture without pretrained weights. | `False` |

### Augmentation

| Argument | Description | Default |
|----------|-------------|---------|
| `--use_dino_augmentation` | Apply separate DINO-style augmentations to student and teacher inputs. | `False` |
| `--rgb_diff_from_unaugmented` | Compute RGB difference from unaugmented frames (requires `--use_dino_augmentation`). | `False` |

### Motion Encoder

| Argument | Description | Default |
|----------|-------------|---------|
| `--use_full_frame` | Pass full RGB frames instead of RGB differences to the motion encoder. | `False` |
| `--rgb_diff_threshold` | Skip loss computation when RGB diff is below this value. | `-1` |
| `--reinit_difference_encoder_every_epoch` | Reinitialize motion encoder weights after every epoch. | `False` |
| `--freeze_difference_encoder` | Freeze motion encoder weights during training. | `False` |
| `--difference_encoder_lr_multiplier` | Scale motion encoder learning rate relative to global LR. | `1.0` |
| `--use_spatial_conditioning` | Use spatial conditioning instead of cross-attention in motion encoder. | `False` |
| `--use_gating` | Use gated spatial conditioning. | `False` |
| `--ignore_prefix_tokens_in_condition` | Ignore CLS tokens when applying spatial conditioning. | `False` |

### Loss

| Argument | Description | Default |
|----------|-------------|---------|
| `--use_mse_loss` | Enable MSE loss on predicted frame latents. | `False` |
| `--mse_loss_weight` | Coefficient for MSE loss. | `1.0` |
| `--rollout_n_frames` | Prediction horizon (frames) for next-frame MSE loss. | `1` |
| `--recon_loss_type` | Reconstruction loss type: `mse` or `smoothl1`. | `"mse"` |
| `--recon_predicted_temp` | Temperature for softening student output in reconstruction loss. | `1.0` |
| `--recon_target_temp` | Temperature for sharpening teacher output in reconstruction loss. | `1.0` |
| `--recon_use_sharpening` | Apply sharpening/softening in reconstruction loss. | `False` |
| `--recon_use_centering` | Apply centering in reconstruction loss. | `False` |
| `--use_motion_loss` | Enable motion contrastive loss. | `False` |
| `--motion_loss_weight` | Coefficient for motion loss. | `1.0` |
| `--min_embed_diff_per_pixel_diff` | Motion loss constant (embedding-to-pixel diff ratio). | `0.44607` |
| `--use_dino_head` | Add DINO projection head after the frame encoder. | `False` |
| `--dino_head_prototype_dim` | Output dimension of the DINO head. | `768` |
| `--use_dino_loss` | Enable DINO loss on CLS tokens. | `False` |
| `--dino_loss_weight` | Coefficient for DINO loss. | `1.0` |
| `--dino_student_temp` | Student softening temperature for DINO loss. | `1` |
| `--dino_teacher_temp` | Teacher sharpening temperature for DINO loss. | `1` |
| `--use_sharpening` | Apply sharpening/softening in DINO loss. | `False` |
| `--use_centering` | Apply centering to teacher in DINO loss. | `False` |
| `--dino_center_update_momentum` | EMA momentum for DINO center updates. | `0.9` |
| `--use_ibot_loss` | Enable iBOT loss on patch tokens. | `False` |
| `--ibot_loss_weight` | Coefficient for iBOT loss. | `1.0` |
| `--ibot_student_temp` | Student softening temperature for iBOT loss. | `1` |
| `--ibot_teacher_temp` | Teacher sharpening temperature for iBOT loss. | `1` |
| `--ibot_center_update_momentum` | EMA momentum for iBOT center updates. | `0.9` |
| `--use_seperate_ibot_head` | Use a separate projection head for iBOT prototypes. | `False` |
| `--use_seperate_ibot_center` | Use a separate center for iBOT centering. | `False` |

### TDV Logging

| Argument | Description | Default |
|----------|-------------|---------|
| `--log_baseline_losses` | Log baseline loss metrics for comparison. | `False` |
| `--log_var_covar` | Log variance and covariance of computed embeddings. | `False` |

---

## Online Evaluation

| Argument | Description | Default |
|----------|-------------|---------|
| `--run_online_evaluations` | Comma-separated evaluations to run: `knn`, `probe`, `mot`. | `"probe"` |
| `--eval_once_before_training_start` | Run evaluation once before training begins. | `False` |
| `--run_eval_on_student` | Also evaluate the student encoder when using EMA. | `False` |

### KNN

| Argument | Description | Default |
|----------|-------------|---------|
| `--knn_eval_data_dir` | Directory containing KNN evaluation data. | `""` |
| `--knn_k_values` | List of K values for KNN. | `[10, 20]` |
| `--knn_temperature` | Temperature for KNN softmax. | `0.07` |
| `--knn_pooling` | Feature pooling before KNN: `cls` or `avg`. | `"cls"` |
| `--knn_batch_size` | Batch size for KNN feature extraction. | `32` |
| `--eval_train_num_samples_per_class` | Training samples per class for KNN fitting; `-1` uses all. | `50` |
| `--eval_val_num_samples_per_class` | Validation samples per class; `-1` uses all. | `-1` |

### Linear/Attentive Probe

| Argument | Description | Default |
|----------|-------------|---------|
| `--probe_eval_dataset` | Dataset for probe evaluation: `imagenet1k`, `ssv2`. | `"imagenet1k"` |
| `--probe_eval_data_dir` | Directory containing probe evaluation data. | `""` |
| `--probe_eval_type` | Probe type: `linear` or `attentive`. | `"linear"` |
| `--probe_eval_bs` | Batch size for probe evaluation. | `128` |
| `--probe_eval_max_epochs` | Epochs to train the probe. | `1` |

### MOT (Multi-Object Tracking)

| Argument | Description | Default |
|----------|-------------|---------|
| `--mot_eval_data_dir` | Directory containing MOT17 evaluation data. | `""` |
| `--mot_eval_max_epochs` | Epochs for MOT evaluation. | `1` |

### Attention Visualization

| Argument | Description | Default |
|----------|-------------|---------|
| `--max_images` | Max images to process; `-1` means no limit. | `-1` |
| `--max_batches` | Max batches to iterate through. | `-1` |
| `--log_every_n_batches` | Log attention maps every N batches. | `1` |
| `--max_images_per_batch` | Max images to visualize per logged batch. | `-1` |

---

## Notes

- Boolean flags default to `False` and are enabled by including them (e.g. `--use_dino_loss`).
- Use `--help` to print a concise version of this list.
- All hparams are logged to W&B (when enabled) for easy cross-referencing across runs.

---

**Example usage:**
```bash
python train_model.py \
--modality "CV" \
--model_name "tdv" \
--run_name "REPRODUCE_RESULTS" \
\
--backbone_type "dinov2" \
--use_dino_head \
--vit_backbone_size "base" \
--time_between_frames 0.25 \
--context_length 16 \
\
--unfreeze_frame_encoder \
--load_without_weights \
--use_ema_for_frame_encoder \
--ema_momentum 0.990025 \
\
--change_encoder_type "dinoViT_xattn_base14" \
--difference_encoder_lr_multiplier 3e-2 \
\
--rollout_n_frames 1 \
--use_mse_loss \
--mse_loss_weight 1.5 \
--use_dino_loss \
--dino_loss_weight 0.75 \
--use_ibot_loss \
--ibot_loss_weight 0.75 \
\
--use_centering \
--use_sharpening \
--dino_teacher_temp 0.1 \
--dino_student_temp 0.1 \
--dino_center_update_momentum 0.9 \
--dino_head_prototype_dim 32768 \
\
--gpus "-1" \
\
--batch_size_per_device 4 \
--accumulate_grad_batches 2 \
--gradient_clip_val 1.0 \
\
--epochs -1 \
--max_steps 300000 \
--peak_learning_rate 1e-4 \
--min_lr_scale 100000 \
--max_scheduling_steps 3000000 \
--warm_up_steps 10000 \
--weight_decay 0.01 \
\
--dataset_name "aggr" \
--aggr_datasets_list "ssv2" \
--aggr_dataset_dirs "/path/to/ssv2, /path/to/ego4d, /path/to/k400" \
--use_preprocessed_ego4d \
--num_workers 6 \
--image_dim 224 224 \
\
--run_online_evaluations "knn" \
--knn_eval_data_dir "/path/to/imagenet-1k" \
--eval_train_num_samples_per_class -1 \
--eval_val_num_samples_per_class -1 \
--run_eval_on_student \
--eval_once_before_training_start \
\
--wandb_project "tdv" \
\
--log_var_covar \
--log_model_archi \
--log_every_n_steps 50 \
\
--set_matmul_precision "medium" \
--is_slurm_run
```
