### SLURM ###
#SBATCH --array=0
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --job-name=linear_probe
#SBATCH --output=logs/slurm/eval/linear_probe_%A-%a.log

CHECKPOINT_PATH="/path/to/checkpoint.ckpt"
DATASET_DIR="/path/to/dataset"
DATASET="ssv2"              # ssv2 | imagenet1k
NUM_CLASSES=174             # 174 for ssv2, 1000 for imagenet1k
CONTEXT_LENGTH=8            # 8 for ssv2, 1 for imagenet1k
FRAME_AGGREGATION="concat"  # concat for ssv2, mean for imagenet1k

mkdir -p logs/slurm/eval/
module purge
ulimit -n 65535

ARGS=(
    # Run identity
    --model_name "tdv"
    --run_name "linear_probe"

    # Backbone
    --backbone_type "dinov2"
    --vit_backbone_size "base"

    # Checkpoint
    --resume_training_ckpt "${CHECKPOINT_PATH}"

    # Dataset
    --probe_eval_dataset "${DATASET}"
    --probe_eval_data_dir "${DATASET_DIR}"
    --num_classes ${NUM_CLASSES}
    --context_length ${CONTEXT_LENGTH}
    --time_between_frames 0.25

    # Probe
    --probe_eval_type "linear"
    --pooling "cls"
    --frame_aggregation "${FRAME_AGGREGATION}"
    --probe_eval_bs 128
    --probe_eval_max_epochs 5

    # Evaluation
    --eval_train_num_samples_per_class -1
    --eval_val_num_samples_per_class -1

    # Hardware & logging
    --gpus -1
    --wandb_project "linear_eval"
)

python eval/probes/train.py "${ARGS[@]}"
