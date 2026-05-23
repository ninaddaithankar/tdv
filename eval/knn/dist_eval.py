import os, sys, torch, torch.distributed as dist
import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from model import model_utils
from hparams.args import get_args
from eval.knn.callback import KNNEvalCallback

import model.cv.dinov1.vision_transformer as vits


# ---------- Distributed setup ----------
def init_distributed():
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank, world_size, local_rank = 0, 1, 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return rank, world_size, local_rank, device


def cleanup_distributed():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


# ---------- minimal dummy trainer (required by the pl knn callback) ----------
class DummyTrainer:
    def __init__(self, rank, world_size, logger=None):
        self.global_rank = rank
        self.world_size = world_size
        self.is_global_zero = (rank == 0)
        self.logger = logger
        self.global_step = 0  # can be ignored for pure eval
        self.strategy = self  # so .strategy.barrier() works

    def barrier(self):
        if dist.is_initialized():
            dist.barrier()


def load_encoder(model_name, pretrained_weights, backbone_type, backbone_size='base', device='cuda'):
    if model_name.startswith("dinov1"):
        encoder = load_dino_v1(pretrained_weights, device=device)
    elif model_name.startswith("tdv"):
        encoder = model_utils.load_tdv_encoder_from_checkpoint(pretrained_weights, backbone_type, backbone_size, key="teacher_frame_encoder")
    else:
        raise ValueError(f"Unknown model name {model_name} for k-NN evaluation")

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

    print(f"Loaded pretrained weights with msg: {msg}")
    return model.to(device).eval()


def main():
    args = get_args()
    rank, world_size, local_rank, device = init_distributed()

    if rank == 0:
        print(f"Rank {rank}/{world_size}, device={device}")

        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config=vars(args),
        )

    # load encoder
    encoder = load_encoder(args.model_name, args.resume_training_ckpt, args.backbone_type, args.vit_backbone_size, device=device)
    for p in encoder.parameters():
        p.requires_grad = False

    # instantiate callback and shadow trainer
    knn_callback = KNNEvalCallback(args)
    trainer = DummyTrainer(rank=rank, world_size=world_size)

    # run evaluation exactly like PL
    results_dict = knn_callback.run_evaluation_with_encoder(trainer, encoder)

    if rank == 0:
        for k, v in results_dict.items():
            wandb.log({k: v})

        wandb.finish()
        
    cleanup_distributed()



if __name__ == "__main__":
    main()
