### ADDITIONAL RUN INFO ###
#SBATCH --array=0
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2

### LOG INFO ###
#SBATCH --job-name=YOUR_RUN_NAME
#SBATCH --output=logs/slurm/YOUR_RUN_NAME
export RUN_NAME="YOUR_RUN_NAME"
export MODEL_NAME="tdv_difference_encoder_x_attn"

export HF_DATASETS_CACHE=$HF_HOME/datasets

mkdir -p logs/slurm/
module purge

python train_model.py \
\
--modality "CV" \
--model_name ${MODEL_NAME} \
--run_name ${RUN_NAME} \
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
--ignore_prefix_tokens_in_condition \
\
--rollout_n_frames 1 \
--use_mse_loss \
--mse_loss_weight 1.5 \
--use_dino_loss \
--dino_loss_weight 0.75 \
--use_ibot_loss \
--ibot_loss_weight 0.75 \
\
--recon_loss_type "mse" \
\
--use_centering \
--use_sharpening \
--dino_teacher_temp 0.1 \
--dino_student_temp 0.1 \
--ibot_teacher_temp 0.1 \
--ibot_student_temp 0.1 \
--dino_center_update_momentum 0.9 \
--ibot_center_update_momentum 0.9 \
--dino_head_prototype_dim 32768 \
\
--gpus "[0, 1]" \
\
--batch_size_per_device 4 \
--accumulate_grad_batches 2 \
--gradient_clip_val 1.0 \
\
--epochs -1 \
--max_steps 600000 \
--peak_learning_rate 1e-4 \
--min_lr_scale 100000 \
--max_scheduling_steps 3000000 \
--warm_up_steps 10000 \
--weight_decay 0.01 \
\
--dataset_name "aggr" \
--aggr_datasets_list "ssv2" \
--aggr_dataset_dirs "/path/to/ssv2" \
--num_workers 6 \
--image_dim 224 224 \
\
--run_online_evaluations "knn" \
--eval_train_num_samples_per_class -1 \
--eval_val_num_samples_per_class -1 \
--run_eval_on_student \
\
--wandb_project "YOUR_PROJECT" \
--wandb_watch \
\
--log_var_covar \
--log_model_archi \
--log_every_n_steps 100 \
--log_baseline_losses \
--log_gradients \
\
--set_matmul_precision "medium" \
--float_precision "bf16-mixed" \
--is_slurm_run \
