### SLURM ###
#SBATCH --array=0
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --job-name=flow_eval
#SBATCH --output=logs/slurm/flow/flow_eval_%A-%a.log

export RUN_NAME="flow_eval"

REPO_ROOT="/path/to/tdv-clean"
OUTPUT_DIR="${REPO_ROOT}/logs/eval/flow/${RUN_NAME}"
NUM_GPUS=2

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${REPO_ROOT}/logs/slurm/flow"

# TDV checkpoint (teacher_frame_encoder used; falls back to frame_encoder if no EMA)
CHECKPOINT="/path/to/checkpoint.ckpt"
# CHECKPOINT="/path/to/dino/checkpoint.pth"   # DINO
# CHECKPOINT="/path/to/ibot/checkpoint.pth"   # iBOT

# SceneFlow optical flow maps to the FlyingThings dataset class in this codepath
TRAIN_DATASET="40*MPISintel('subtrain_cleanpass')+40*MPISintel('subtrain_finalpass')+4*FlyingThings('train_allpass')+4*FlyingChairs('train')"
VAL_DATASET="MPISintel('subval_cleanpass')+MPISintel('subval_finalpass')"

cd "${REPO_ROOT}" || exit 1

ARGS=(
    # Logging
    --wandb_run_name "${RUN_NAME}"
    --wandb_project "flow_eval"

    # Backbone
    --patch_size 14
    --pretrained "${CHECKPOINT}"

    # Decoder
    --motion_depth 4
    --feature_block_type cross
    --decoder_feature_mode motion2-forward
    --gating_type pred-vector
    --gating_bias 4
    --no_pred_pos_embed
    --hooks_idx 15 19 31 35

    # Dataset
    --dataset "${TRAIN_DATASET}"
    --val_dataset "${VAL_DATASET}"
    --dataset_dirs
        SceneFlow=/path/to/sceneflow
        MPISintel=/path/to/mpi-sintel
        FlyingChairs=/path/to/flying-chairs
    --img_per_epoch 30000

    # Training
    --batch_size 8
    --crop 224 224
    --epochs 240
    --criterion "LaplacianLossBounded()"
    --lr 2e-5
    --num_workers 16
    --amp 1

    # Evaluation
    --val_overlap 0.5
    --eval_every 5
    --log_flow_stack_every 10000

    --resume "${OUTPUT_DIR}/checkpoint-last.pth"
    --output_dir "${OUTPUT_DIR}"
)

CUDA_VISIBLE_DEVICES=0,1 torchrun --master_port=29797 --nproc_per_node=${NUM_GPUS} \
    "${REPO_ROOT}/eval/flow/train_midway.py" flow "${ARGS[@]}"
