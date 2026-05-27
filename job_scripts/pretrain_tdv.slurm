### SLURM ###
#SBATCH --array=0
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --job-name=YOUR_RUN_NAME
#SBATCH --output=logs/slurm/YOUR_RUN_NAME

export RUN_NAME="YOUR_RUN_NAME"
export MODEL_NAME="tdv"

export HF_DATASETS_CACHE=$HF_HOME/datasets

mkdir -p logs/slurm/
module purge

ARGS=(
    # Run identity
    --modality "CV"
    --model_name ${MODEL_NAME}
    --run_name ${RUN_NAME}

    # Backbone
    --backbone_type "dinov2"
    --vit_backbone_size "base"
    --use_dino_head
    --context_length 16
    --time_between_frames 0.25

    # Frame encoder
    --unfreeze_frame_encoder
    --load_without_weights
    --use_ema_for_frame_encoder
    --ema_momentum 0.990025

    # Difference encoder
    --change_encoder_type "dinoViT_xattn_base14"
    --difference_encoder_lr_multiplier 3e-2
    --ignore_prefix_tokens_in_condition

    # Loss
    --rollout_n_frames 1
    --use_mse_loss
    --mse_loss_weight 1.5
    --use_dino_loss
    --dino_loss_weight 0.75
    --use_ibot_loss
    --ibot_loss_weight 0.75
    --recon_loss_type "mse"

    # Teacher / centering / sharpening
    --use_centering
    --use_sharpening
    --dino_teacher_temp 0.1
    --dino_student_temp 0.1
    --ibot_teacher_temp 0.1
    --ibot_student_temp 0.1
    --dino_center_update_momentum 0.9
    --ibot_center_update_momentum 0.9
    --dino_head_prototype_dim 32768

    # Hardware
    --gpus "-1"
    --set_matmul_precision "medium"
    --float_precision "bf16-mixed"
    --is_slurm_run

    # Optimiser
    --batch_size_per_device 4
    --accumulate_grad_batches 2
    --gradient_clip_val 1.0
    --epochs -1
    --max_steps 600000
    --peak_learning_rate 1e-4
    --min_lr_scale 100000
    --max_scheduling_steps 3000000
    --warm_up_steps 10000
    --weight_decay 0.01

    # Dataset
    --dataset_name "aggr"
    --aggr_datasets_list "ssv2"
    --aggr_dataset_dirs "/path/to/ssv2"
    --num_workers 6
    --image_dim 224 224

    # Evaluation
    --run_online_evaluations "knn"
    --eval_train_num_samples_per_class -1
    --eval_val_num_samples_per_class -1
    --run_eval_on_student
    --knn_eval_data_dir "/path/to/imagenet"

    # Wandb
    --wandb_project "YOUR_PROJECT"
    --wandb_watch

    # Logging
    --log_var_covar
    --log_model_archi
    --log_every_n_steps 100
    --log_baseline_losses
    --log_gradients
)

python train_model.py "${ARGS[@]}"
