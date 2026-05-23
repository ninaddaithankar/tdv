# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
from functools import partial
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models
import wandb

from data.ego4d_dataloader import Ego4DTasksDataset
from data.imagenet_dataloader import ImageNetDataset, ImageNetSequentialClips
from data.something_dataloader import SomethingDataset
from eval_knn import evaluate_knn, get_args
import utils
import vision_transformer as vits
from vision_transformer import DINOHead

torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base','vit_large', 'vit_huge', 'xcit', 'deit_tiny', 'deit_small'] \
                + torchvision_archs + torch.hub.list("facebookresearch/xcit:main"),
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=8, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--acc_grad_steps', default=1, type=int,
        help='Number of gradient accumulation steps.')
    parser.add_argument('--epochs', default=10, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument("--warmup_steps", default=10000, type=int,
        help="Number of steps for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")
    parser.add_argument('--minimal_augmentation', type=utils.bool_flag, default=False, help="""Whether to use minimal augmentation
        (only random resized crop and normalization) for the global views. Useful for debugging or
        to disable color-based augmentations when working with grayscale images.""")
    parser.add_argument('--num_student_views', default=2, type=int, help="Number of views fed to student.")
    parser.add_argument('--num_teacher_views', default=2, type=int, help="Number of views fed to teacher.")
    parser.add_argument('--mask_ratio', default=0.0, type=float, help="Proportion of the visible patches in the input.")
    parser.add_argument('--use_ibot_masking', default=False, type=utils.bool_flag, help="Whether to use iBOT style masking (different masks for student and teacher).")
    parser.add_argument('--ibot_mask_ratio_min_max', default=(0.7, 0.75), type=float, nargs=2, help="Min and max proportion of visible patches for iBOT style masking.")
    parser.add_argument('--ibot_mask_sample_probability', default=1.0, type=float, help="Probability of applying iBOT style masking to a sample.")
    
    parser.add_argument('--image_dim', default=224, type=int, help="Image dimension.")

    # Misc
    parser.add_argument('--datasets', default='ssv2', type=str, help='Comma separated list of datasets to train on.')
    parser.add_argument('--data_paths', default='/path/to/imagenet/train/', type=str, help='Please specify path to the training data.')
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=1, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--resume', default=False, type=utils.bool_flag, help='Resume training from checkpoint.pth in output_dir.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=6, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local-rank", default=0, type=int, help="Please ignore and do not set this argument.")

    parser.add_argument("--imagenet_samples_per_class", default=-1, type=int, help="Number of samples per class for imagenet. -1 uses all samples.")
    parser.add_argument("--non_random_filtering", default=True, type=utils.bool_flag, help="Whether to use non-random filtering for imagenet samples.")
    parser.add_argument("--data_pct", default=1.0, type=float, help="Fraction of each dataset to train on (0, 1]. Subset is chosen deterministically.")
    parser.add_argument("--context_length", default=16, type=int, help="Number of frames in the input clip.")
    parser.add_argument("--temporal_diff", default=0.25, type=float, help="Time difference between sampled frames in seconds.")
    parser.add_argument("--knn_freq", default=1, type=int, help="run knn evaluation every n epochs.")
    parser.add_argument("--run_name", required=True, type=str, help="Name of run on wandb.")
    parser.add_argument("--wandb_run_id", default=None, type=str, help="Wandb run id for resuming.")
    parser.add_argument("--wandb_resume", default="allow", type=str, help="Wandb resume mode (allow, must, never).")
    parser.add_argument("--project_name", default="dino_recipe", type=str, help="Wandb project name.")

    return parser


def train_dino(args):
    utils.init_distributed_mode(args)

    # init wandb
    if utils.is_main_process():
        wandb.init(project=args.project_name, id=args.wandb_run_id, resume=args.wandb_resume, name=args.run_name, config=vars(args))

    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    transform = DataAugmentationDINO(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
        use_minimal=args.minimal_augmentation,
        image_dim=args.image_dim,
        return_dict=args.use_ibot_masking,
    )

    hparams = SimpleNamespace(**{
        "dataset_dir": None,
        "context_length": args.context_length,
        "time_between_frames": args.temporal_diff,
        "sampling_rate": 0.0,
        "preencode_dataset": False,
        "use_preencoded_dataset": False,
        "debug_mode": False,
        "model_name": "linear_probe",
        "preprocess_data": False,
        "crop_all_samples": False,
        "use_raw_framerate": False,
        "use_preprocessed_ego4d": True,
    })

    datasets_list = [d.strip() for d in args.datasets.split(',')]
    data_paths = [p.strip() for p in args.data_paths.split(',')]

    dataset_classes = {
        'imagenet': partial(ImageNetDataset, n_samples_per_class=args.imagenet_samples_per_class, non_random_filtering=args.non_random_filtering),
        'imagenet-video': ImageNetSequentialClips,
        'ego4d': Ego4DTasksDataset,
        'ssv2': SomethingDataset,
    }

    datasets = []
    for ds_name, data_path in zip(datasets_list, data_paths[:len(datasets_list)]):
        dataset_class = dataset_classes.get(ds_name)
        if dataset_class is None:
            raise ValueError(f"Unknown dataset: {ds_name}. Skipping...")
        else:
            ds = dataset_class(hparams, "train", dataset_dir=data_path, transform=transform)
            if args.data_pct < 1.0:
                n_total = len(ds)
                n_keep = max(1, int(round(args.data_pct * n_total)))
                rng = np.random.default_rng(args.seed)
                indices = np.sort(rng.choice(n_total, size=n_keep, replace=False)).tolist()
                ds = torch.utils.data.Subset(ds, indices)
                print(f"Subset {ds_name} to {n_keep}/{n_total} samples ({args.data_pct*100:.1f}%)")
            datasets.append(ds)
            print(f"Loaded {ds_name} with {len(ds)} items.")
    
    dataset = torch.utils.data.ConcatDataset(datasets)
    print(f"Total aggregate dataset has {len(dataset)} items.")

    collate_fn = None
    if args.use_ibot_masking:
        img_size = args.image_dim
        patch_size = args.patch_size
        n_tokens = (img_size // patch_size) ** 2
        mask_generator = utils.MaskingGenerator(
            input_size=(img_size // patch_size, img_size // patch_size),
            max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
        )

        inputs_dtype = torch.float16 if args.use_fp16 else torch.float32
        collate_fn = partial(
            utils.collate_data_and_cast,
            mask_ratio_tuple=args.ibot_mask_ratio_min_max,
            mask_probability=args.ibot_mask_sample_probability,
            n_tokens=n_tokens,
            mask_generator=mask_generator,
            dtype=inputs_dtype,
        )

    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    print(f"Data loaded: there are {len(dataset)} images.")

    # ============ building student and teacher networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path_rate,  # stochastic depth
            use_masking=(args.mask_ratio > 0.0 or args.use_ibot_masking),
        )
        teacher = vits.__dict__[args.arch](patch_size=args.patch_size, use_masking=(args.mask_ratio > 0.0 or args.use_ibot_masking))
        embed_dim = student.embed_dim
    # if the network is a XCiT
    elif args.arch in torch.hub.list("facebookresearch/xcit:main"):
        student = torch.hub.load('facebookresearch/xcit:main', args.arch,
                                 pretrained=False, drop_path_rate=args.drop_path_rate)
        teacher = torch.hub.load('facebookresearch/xcit:main', args.arch, pretrained=False)
        embed_dim = student.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        teacher = torchvision_models.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]
    else:
        print(f"Unknown architecture: {args.arch}")

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper(student, DINOHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    ))
    teacher = utils.MultiCropWrapper(
        teacher,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
    )
    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============
    dino_loss = DINOLoss(
        args.out_dim,
        args.local_crops_number + args.num_student_views,  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
        args.num_teacher_views,
    ).cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
        warmup_steps=args.warmup_steps,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    start_epoch = 0
    knn_freq = args.knn_freq if hasattr(args, 'knn_freq') else 1
    acc_grad_steps = args.acc_grad_steps

    if args.resume:
        to_restore = {"epoch": 0}
        utils.restart_from_checkpoint(
            os.path.join(args.output_dir, "checkpoint.pth"),
            run_variables=to_restore,
            student=student,
            teacher=teacher,
            optimizer=optimizer,
            fp16_scaler=fp16_scaler,
            dino_loss=dino_loss,
        )
        start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting DINO training !")
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of DINO ... ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, acc_grad_steps, args)
        
        # ============ logging ... ============
        end_step = (epoch + 1) * len(data_loader) - 1
        if utils.is_main_process():
            # log scalar metrics
            wandb.log({f"train/{k}_epoch": v for k, v in train_stats.items()}, step=end_step)

        if (epoch % knn_freq == 0) or (epoch == args.epochs - 1):
            evaluate_knn(get_args(defaults=True), step=end_step, encoder=teacher_without_ddp.backbone) 

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        # if args.saveckp_freq and epoch % args.saveckp_freq == 0:
        #     utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")   
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    if utils.is_main_process():
        wandb.finish()


def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, acc_grad_steps, args):
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    
    for it, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration

        if (it + 1) % acc_grad_steps == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = lr_schedule[it]
                if i == 0:  # only the first group is regularized
                    param_group["weight_decay"] = wd_schedule[it]

        # flatten time dim if video and move images to gpu
        # prev_frames, next_frames = images
        # if prev_frames[0].dim() == 5:  # for video inputs
        #     prev_frames = [im.flatten(0, 1).cuda(non_blocking=True) for im in prev_frames]
        #     next_frames = [im.flatten(0, 1).cuda(non_blocking=True) for im in next_frames]
        # else:
        #     prev_frames = [im.cuda(non_blocking=True) for im in prev_frames]
        #     next_frames = [im.cuda(non_blocking=True) for im in next_frames]

        masks, mask_indices_list, masks_weight, n_masked_patches = None, None, None, None
        if isinstance(batch, dict):
            # new collate output
            global_crops = batch["collated_global_crops"].cuda(non_blocking=True)
            local_crops  = batch["collated_local_crops"].cuda(non_blocking=True) if batch["collated_local_crops"].numel() > 0 else None

            masks             = batch["collated_masks"].cuda(non_blocking=True)          # (G*B, N) bool
            mask_indices_list = batch["mask_indices_list"].cuda(non_blocking=True)       # (total_masked,)
            masks_weight      = batch["masks_weight"].cuda(non_blocking=True)            # (total_masked,)
            n_masked_patches  = batch["n_masked_patches"].cuda(non_blocking=True)        # (1,)

            B_img = global_crops.shape[0] // args.num_student_views   # if args.num_student_views == G
            G = args.num_student_views

            # reshape to (G, B_img, C, H, W) then split into list length G
            global_views = global_crops.view(G, B_img, *global_crops.shape[1:])  # (G,B,C,H,W)
            images = [global_views[i] for i in range(G)]

            # local crops: assume L == args.local_crops_number
            if args.local_crops_number > 0 and local_crops is not None:
                L = args.local_crops_number
                local_views = local_crops.view(L, B_img, *local_crops.shape[1:])  # (L,B,C,h,w)
                images += [local_views[i] for i in range(L)]
        else:
            images, _ = batch
            if images[0].dim() == 5:  # for video inputs
                images = [im.flatten(0, 1).cuda(non_blocking=True) for im in images]
            else:
                images = [im.cuda(non_blocking=True) for im in images]

        num_all_student_views = args.local_crops_number + args.num_student_views
        mask_ratio = None if (args.use_ibot_masking or args.mask_ratio <= 0.0) else args.mask_ratio

        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output = teacher(images[:args.num_teacher_views])  # only the first global view is passed to the teacher
            student_output = student(images[:num_all_student_views], mask_ratio, masks) # student sees all available view
            
            loss = dino_loss(student_output, teacher_output, epoch)
            # loss += args.ibot_weight * ibot_loss(student_output, teacher_output, mask_indices_list, masks_weight, n_masked_patches, epoch)

            if acc_grad_steps > 1:
                loss = loss / acc_grad_steps

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
        else:
            fp16_scaler.scale(loss).backward()
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)

        # update gradients according to gradient accumulation schedule
        if (it + 1) % acc_grad_steps == 0:
            if fp16_scaler is None:
                if args.clip_grad:
                    param_norms = utils.clip_gradients(student, args.clip_grad)
                optimizer.step()
            else:
                if args.clip_grad:
                    fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                    param_norms = utils.clip_gradients(student, args.clip_grad)
                fp16_scaler.step(optimizer)
                fp16_scaler.update()
        
            optimizer.zero_grad()

            # EMA update for the teacher
            with torch.no_grad():
                m = momentum_schedule[it]  # momentum parameter
                for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
        
        if utils.is_main_process():
            wandb.log(
                {
                    "train/loss": loss.item(), 
                    "train/epoch": epoch,
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/wd": optimizer.param_groups[0]["weight_decay"],
                    "train/momentum": momentum_schedule[it],
                    "train/global_step": it,
                },
                step=it,
            )

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, num_teacher_views=2,student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
        self.num_teacher_views = num_teacher_views  # we only apply the loss to the 2 global views

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(self.num_teacher_views)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number, use_minimal=False, image_dim=224, return_dict=False):
        self.return_dict = return_dict
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.use_minimal = use_minimal
        if use_minimal:
            minimal = transforms.Compose([
                # transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                # transforms.CenterCrop(224),

                transforms.Resize((image_dim, image_dim)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                utils.GaussianBlur(1.0),

                normalize,
            ])
            self.global_transfo1 = minimal
            self.global_transfo2 = minimal

            self.local_crops_number = 0
            self.local_transfo = None

        else:
            # first global crop
            self.global_transfo1 = transforms.Compose([
                transforms.RandomResizedCrop(image_dim, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(1.0),
                normalize,
            ])
            # second global crop
            self.global_transfo2 = transforms.Compose([
                transforms.RandomResizedCrop(image_dim, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(0.1),
                utils.Solarization(0.2),
                normalize,
            ])
            # transformation for the local small crops
            self.local_crops_number = local_crops_number
            self.local_transfo = transforms.Compose([
                transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(p=0.5),
                normalize,
            ])

    def __call__(self, image):
        output = {}
        output["local_crops"] = []

        if self.use_minimal:
            output["global_crops"] = [self.global_transfo1(image), self.global_transfo2(image)]
        else:
            output["global_crops"] = [self.global_transfo1(image), self.global_transfo2(image)]
            for _ in range(self.local_crops_number):
                output["local_crops"].append(self.local_transfo(image))

        return output if self.return_dict else (output["global_crops"] + output["local_crops"])


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_dino(args)
