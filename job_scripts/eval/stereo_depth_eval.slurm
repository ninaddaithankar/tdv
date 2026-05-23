### ADDITIONAL RUN INFO ###
#SBATCH --array=0
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2

### LOG INFO ###
#SBATCH --job-name=stereo_depth_eval
#SBATCH --output=logs/slurm/depth_eval/stereo_depth_eval_%A-%a.log

export RUN_NAME="stereo_depth_eval"

REPO_ROOT="/path/to/tdv-clean"
OUTPUT_DIR="${REPO_ROOT}/logs/eval/stereo/${RUN_NAME}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${REPO_ROOT}/logs/slurm/depth_eval"

NUM_GPUS=2

# TDV checkpoint (teacher_frame_encoder used; falls back to frame_encoder if no EMA)
CHECKPOINT="/path/to/checkpoint.ckpt"

# DINO
# CHECKPOINT="/path/to/dino/checkpoint.pth"

# iBOT
# CHECKPOINT="/path/to/ibot/checkpoint.pth"

TRAIN_DATASET="SceneFlow('train_finalpass')"
VAL_DATASET="SceneFlow('test1of100_finalpass')"

cd "${REPO_ROOT}" || exit 1

CUDA_VISIBLE_DEVICES=0,1 torchrun --master_port=29788 --nproc_per_node=${NUM_GPUS} \
  "${REPO_ROOT}/eval/flow/train_midway.py" stereo \
  --wandb_run_name "${RUN_NAME}" \
  --wandb_project "stereo_sceneflow" \
  \
  --patch_size 16 \
  --pretrained "${CHECKPOINT}" \
  \
  --motion_depth 4 \
  --feature_block_type cross \
  --decoder_feature_mode motion2-forward \
  --gating_type pred-vector \
  --gating_bias 4 \
  --no_pred_pos_embed \
  --hooks_idx 15 19 31 35 \
  --layer_dims 48 96 192 384 \
  \
  --dataset "${TRAIN_DATASET}" \
  --val_dataset "${VAL_DATASET}" \
  --dataset_dirs SceneFlow=/path/to/sceneflow \
  \
  --batch_size 3 \
  --crop 352 704 \
  --epochs 32 \
  --criterion "LaplacianLossBounded2()" \
  --lr 3e-5 \
  --num_workers 8 \
  \
  --output_dir "${OUTPUT_DIR}"
