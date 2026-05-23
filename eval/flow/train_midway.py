import argparse
import datetime
import json
import numpy as np
import os
import time
import sys

import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from model.model_utils import create_image_encoder
from croco.utils.misc import NativeScalerWithGradNormCount as NativeScaler
import croco.utils.misc as misc

from midway_flow_wrapper import build_midway_flow_model_from_checkpoint

from croco.stereoflow.datasets_stereo import get_train_dataset_stereo, get_test_datasets_stereo
from croco.stereoflow.datasets_flow import get_train_dataset_flow, get_test_datasets_flow
from croco.stereoflow.engine import train_one_epoch, validate_one_epoch
from croco.stereoflow.criterion import *


def get_args_parser():
    parser = argparse.ArgumentParser('Midway Decoder flow training', add_help=False)
    subparsers = parser.add_subparsers(title="Task (stereo or flow)", dest="task", required=True)
    parser_stereo = subparsers.add_parser('stereo', help='Training stereo model')
    parser_flow = subparsers.add_parser('flow', help='Training flow model')

    def add_arg(name_or_flags, default=None, default_stereo=None, default_flow=None, **kwargs):
        if default is not None:
            assert default_stereo is None and default_flow is None
        parser_stereo.add_argument(name_or_flags, default=default if default is not None else default_stereo, **kwargs)
        parser_flow.add_argument(name_or_flags, default=default if default is not None else default_flow, **kwargs)

    # output / checkpoint
    add_arg('--output_dir', required=True, type=str)
    add_arg('--pretrained', required=True, type=str,
            help="TDV checkpoint to load encoder weights from")

    # crop
    add_arg('--crop', type=int, nargs='+', default_stereo=[352, 704], default_flow=[224, 224])

    # criterion
    add_arg('--criterion', default_stereo='LaplacianLossBounded2()', default_flow='LaplacianLossBounded()', type=str)
    add_arg('--bestmetric', default_stereo='avgerr', default_flow='EPE', type=str)

    # dataset
    add_arg('--dataset', type=str, required=True)
    add_arg('--val_dataset', type=str, default='')
    add_arg('--dataset_dirs', nargs='*', default=[],
            help='Override dataset root dirs as KEY=PATH pairs, e.g. SceneFlow=/data/sceneflow')

    # training hyperparams
    add_arg('--seed', default=0, type=int)
    add_arg('--batch_size', default_stereo=6, default_flow=8, type=int)
    add_arg('--epochs', default=240, type=int)
    add_arg('--img_per_epoch', type=int, default=30000)
    add_arg('--accum_iter', default=1, type=int)
    add_arg('--weight_decay', type=float, default=0.05)
    add_arg('--lr', type=float, default_stereo=3e-5, default_flow=2e-5, metavar='LR')
    add_arg('--min_lr', type=float, default=0., metavar='LR')
    add_arg('--warmup_epochs', type=int, default=1, metavar='N')
    add_arg('--optimizer', default='AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))', type=str)
    add_arg('--amp', default=1, type=int, choices=[0, 1])

    # validation
    add_arg('--tile_conf_mode', type=str, default_stereo='conf_expsigmoid_15_3', default_flow='conf_expsigmoid_10_5')
    add_arg('--val_overlap', default=0.5, type=float)

    # IterativeLatentMotion decoder config
    add_arg('--feature_levels', type=int, nargs='+', default=[2, 5, 8, 11],
            help='Encoder block indices used as feature levels in IterativeLatentMotion')
    add_arg('--motion_dim', type=int, default=192)
    add_arg('--motion_depth', type=int, default=4)
    add_arg('--motion_tokens', type=int, default=10)
    add_arg('--predictor_depth', type=int, default=4)
    add_arg('--motion_agg_type', type=str, default='add')
    add_arg('--feature_block_type', type=str, default='cross',
            help='Feature block type in IterativeLatentMotion (identity or cross)')
    add_arg('--decoder_feature_mode', type=str, default='motion2-forward',
            help='Decoder feature mode (motion, motion2-forward, feature)')
    add_arg('--gating_type', type=str, default='pred-vector',
            help='Gating type for predictor (e.g. pred-vector)')
    add_arg('--gating_bias', type=float, default=4.0,
            help='Gating bias (e.g. 4.0)')
    add_arg('--no_pred_pos_embed', action='store_true', default=True,
            help='Disable positional embedding in the predictor branch')
    add_arg('--num_cls_tokens', type=int, default=1,
            help='Number of cls/register tokens to strip from DINOv2 all_layers (1 for standard DINOv2)')
    add_arg('--patch_size', type=int, default=14,
            help='DINOv2 patch size (14 for ViT-S/B/L with patch14)')

    # DPT
    add_arg('--hooks_idx', type=int, nargs='+', default=[15, 19, 31, 35],
            help='Manual DPT hook indices; None means auto-computed from decoder config')
    add_arg('--layer_dims', type=int, nargs='+', default=None,
            help='DPT layer dims (4 values); None means [96,192,384,768]')

    # freeze encoder
    add_arg('--freeze_encoder', action='store_true', default=False,
            help='Freeze DINOv2 encoder weights (train only decoder + DPT head)')
    add_arg('--pretrained_encoder', action='store_true', default=False,
            help='Load encoder with its own pretrained weights; skip loading from --pretrained checkpoint')

    # others
    add_arg('--num_workers', default=16, type=int)
    add_arg('--eval_every', type=int, default=5)
    add_arg('--save_every', type=int, default=1)
    add_arg('--tboard_log_step', type=int, default=100)
    add_arg('--log_flow_stack_every', type=int, default=10000)
    add_arg('--resume', type=str, default=None, help='Path to checkpoint to resume training from')
    add_arg('--dist_url', default='env://')
    add_arg('--wandb_project', type=str, default='debug')
    add_arg('--wandb_run_name', type=str, default='midway_flow')

    return parser


def main(args):
    misc.init_distributed_mode(args)
    global_rank = misc.get_rank()
    num_tasks = misc.get_world_size()

    assert os.path.isfile(args.pretrained), f"Checkpoint not found: {args.pretrained}"
    os.makedirs(args.output_dir, exist_ok=True)

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    metrics = (StereoMetrics if args.task == 'stereo' else FlowMetrics)().to(device)
    criterion = eval(args.criterion).to(device)
    print('Criterion:', args.criterion)

    num_channels = {'stereo': 1, 'flow': 2}[args.task]
    if criterion.with_conf:
        num_channels += 1

    print(f'Building MidwayDINOv2FlowWithDPT with {num_channels} channel(s)')

    # Compute num_patches from crop + patch_size so the decoder gets proper
    # spatial positional embeddings (not a degenerate scalar bias).
    num_patches = (args.crop[0] // args.patch_size) * (args.crop[1] // args.patch_size)
    print(f'num_patches={num_patches} (crop={args.crop}, patch_size={args.patch_size})')

    model, tdv_hparams = build_midway_flow_model_from_checkpoint(
        args.pretrained,
        num_channels=num_channels,
        feature_levels=args.feature_levels,
        motion_dim=args.motion_dim,
        motion_depth=args.motion_depth,
        motion_tokens=args.motion_tokens,
        predictor_depth=args.predictor_depth,
        motion_agg_type=args.motion_agg_type,
        feature_block_type=args.feature_block_type,
        decoder_feature_mode=args.decoder_feature_mode,
        gating_type=args.gating_type,
        gating_bias=args.gating_bias,
        use_pred_pos_embed=not args.no_pred_pos_embed,
        num_cls_tokens=args.num_cls_tokens,
        num_patches=num_patches,
        patch_size=args.patch_size,
        hooks_idx=args.hooks_idx,
        layer_dims=args.layer_dims,
        device=device.type,
        strict=False,
        pretrained_encoder=args.pretrained_encoder,
        create_image_encoder_fn=create_image_encoder,
    )

    # Optionally freeze the DINOv2 encoder (train only decoder + DPT head)
    if args.freeze_encoder:
        for p in model.backbone.frame_encoder.parameters():
            p.requires_grad = False
        print('DINOv2 encoder frozen.')

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}  |  Trainable: {trainable_params:,}")

    model_without_ddp = model.to(device)

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    print(f"lr: {args.lr:.2e}  |  accum_iter: {args.accum_iter}  |  eff batch size: {eff_batch_size}")

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    param_groups = misc.get_parameter_groups(model_without_ddp, args.weight_decay)
    optimizer = eval(f"torch.optim.{args.optimizer}")
    print(optimizer)
    loss_scaler = NativeScaler()

    best_so_far = misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    if best_so_far is None:
        best_so_far = np.inf

    log_writer = None
    if global_rank == 0:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args),
                   resume="allow" if args.resume else None)

    # datasets
    import croco.stereoflow.datasets_stereo as _ds_stereo
    import croco.stereoflow.datasets_flow as _ds_flow
    _ds_stereo.set_dataset_dirs(args.dataset_dirs)
    _ds_flow.set_dataset_dirs(args.dataset_dirs)

    print('Building train dataset:', args.dataset)
    train_dataset = (get_train_dataset_stereo if args.task == 'stereo' else get_train_dataset_flow)(
        args.dataset, crop_size=args.crop)
    print('  total length:', len(train_dataset))

    if args.distributed:
        sampler_train = torch.utils.data.DistributedSampler(
            train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    else:
        sampler_train = torch.utils.data.RandomSampler(train_dataset)

    data_loader_train = torch.utils.data.DataLoader(
        train_dataset, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    if args.val_dataset == '':
        data_loaders_val = None
    else:
        print('Building val datasets:', args.val_dataset)
        val_datasets = (get_test_datasets_stereo if args.task == 'stereo' else get_test_datasets_flow)(args.val_dataset)
        for vd in val_datasets:
            print(repr(vd))
        data_loaders_val = [
            DataLoader(vd, batch_size=1, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True, drop_last=False)
            for vd in val_datasets
        ]
        bestmetric = ("AVG_" if len(data_loaders_val) > 1 else str(data_loaders_val[0].dataset) + '_') + args.bestmetric

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        epoch_start = time.time()
        train_stats = train_one_epoch(
            model, criterion, metrics, data_loader_train, optimizer,
            device, epoch, loss_scaler, global_rank=global_rank,
            log_writer=log_writer, args=args)
        epoch_time = time.time() - epoch_start

        if args.distributed:
            dist.barrier()

        if data_loaders_val is not None and args.eval_every > 0 and (epoch + 1) % args.eval_every == 0:
            val_epoch_start = time.time()
            val_stats = validate_one_epoch(
                model, criterion, metrics, data_loaders_val,
                device, epoch, global_rank=global_rank,
                log_writer=log_writer, args=args)
            val_epoch_time = time.time() - val_epoch_start

            val_best = val_stats[bestmetric]
            if val_best <= best_so_far:
                best_so_far = val_best
                misc.save_model(args=args, model_without_ddp=model_without_ddp,
                                optimizer=optimizer, loss_scaler=loss_scaler,
                                epoch=epoch, best_so_far=best_so_far, fname='best')

            log_stats = {
                **{f'train_epoch/{k}': v for k, v in train_stats.items()},
                'epoch': epoch,
                **{f'val_epoch/{k}': v for k, v in val_stats.items()},
            }
        else:
            log_stats = {
                **{f'train_epoch/{k}': v for k, v in train_stats.items()},
                'epoch': epoch,
            }

        if args.distributed:
            dist.barrier()

        if global_rank == 0:
            wandb.log(log_stats)

        if args.output_dir and ((epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs):
            misc.save_model(args=args, model_without_ddp=model_without_ddp,
                            optimizer=optimizer, loss_scaler=loss_scaler,
                            epoch=epoch, best_so_far=best_so_far, fname='last')

        if args.output_dir:
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    if global_rank == 0:
        wandb.finish()

    total_time_str = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f'Training time {total_time_str}')


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
