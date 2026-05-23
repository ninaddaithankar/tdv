### ADDITIONAL RUN INFO ###
#SBATCH --array=0
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2

### LOG INFO ###
#SBATCH --job-name=flow_eval
#SBATCH --output=logs/slurm/flow/flow_eval_%A-%a.log

export RUN_NAME="flow_eval"

REPO_ROOT="/path/to/tdv-clean"
OUTPUT_DIR="${REPO_ROOT}/logs/eval/flow/${RUN_NAME}"
mkdir -p "${OUTPUT_DIR}"

# In this codepath, SceneFlow optical flow maps to the FlyingThings dataset class.
TRAIN_DATASET="40*MPISintel('subtrain_cleanpass')+40*MPISintel('subtrain_finalpass')+4*FlyingThings('train_allpass')+4*FlyingChairs('train')"
VAL_DATASET="MPISintel('subval_cleanpass')+MPISintel('subval_finalpass')"

NUM_GPUS=2

# TDV checkpoint (teacher_frame_encoder used; falls back to frame_encoder if no EMA)
CHECKPOINT="/path/to/checkpoint.ckpt"

# DINO
# CHECKPOINT="/path/to/dino/checkpoint.pth"

# iBOT
# CHECKPOINT="/path/to/ibot/checkpoint.pth"

cd "${REPO_ROOT}" || exit 1

CUDA_VISIBLE_DEVICES=0,1 torchrun --master_port=29797 --nproc_per_node=${NUM_GPUS} \
  "${REPO_ROOT}/eval/flow/train_midway.py" flow \
  --wandb_run_name "${RUN_NAME}" \
  --wandb_project "flow_eval" \
  \
  --patch_size 14 \
  --pretrained "${CHECKPOINT}" \
  \
  --motion_depth 4 \
  --feature_block_type cross \
  --decoder_feature_mode motion2-forward \
  --gating_type pred-vector \
  --gating_bias 4 \
  --no_pred_pos_embed \
  --hooks_idx 15 19 31 35 \
  \
  --dataset "${TRAIN_DATASET}" \
  --val_dataset "${VAL_DATASET}" \
  --dataset_dirs \
    SceneFlow=/path/to/sceneflow \
    MPISintel=/path/to/mpi-sintel \
    FlyingChairs=/path/to/flying-chairs \
  --img_per_epoch 30000 \
  --batch_size 8 \
  --crop 224 224 \
  --epochs 240 \
  --criterion "LaplacianLossBounded()" \
  --lr 2e-5 \
  --num_workers 16 \
  --log_flow_stack_every 10000 \
  --amp 1 \
  --val_overlap 0.5 \
  --eval_every 5 \
  --resume "${OUTPUT_DIR}/checkpoint-last.pth" \
  --output_dir "${OUTPUT_DIR}"
