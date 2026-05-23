# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

# --------------------------------------------------------
# Main function for training one epoch or testing
# --------------------------------------------------------

import math
import sys
from typing import Iterable
import numpy as np
import torch
import torch.distributed as dist
import torchvision
import wandb

import croco.utils.misc as misc
from eval.flow.croco.stereoflow.datasets_flow import flowMaxNorm, flowToColor
from eval.flow.croco.stereoflow.datasets_stereo import unnormalize_imagenet


def split_prediction_conf(predictions, with_conf=False):
    if not with_conf:
        return predictions, None
    conf = predictions[:,-1:,:,:]
    predictions = predictions[:,:-1,:,:]
    return predictions, conf

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module, metrics: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, global_rank: int,
                    log_writer=None, print_freq = 20,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    details = {}

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    if args.img_per_epoch:
        iter_per_epoch = args.img_per_epoch // args.batch_size + int(args.img_per_epoch % args.batch_size > 0)
        assert len(data_loader) >= iter_per_epoch, 'Dataset is too small for so many iterations'
        len_data_loader = iter_per_epoch
    else:
        len_data_loader, iter_per_epoch = len(data_loader), None

    for data_iter_step, (image1, image2, gt, pairname) in enumerate(metric_logger.log_every(data_loader, print_freq, header, max_iter=iter_per_epoch)):
        
        image1 = image1.to(device, non_blocking=True)
        image2 = image2.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)
        
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            misc.adjust_learning_rate(optimizer, data_iter_step / len_data_loader + epoch, args)

        with torch.cuda.amp.autocast(enabled=bool(args.amp)):
            prediction = model(image1, image2)
            prediction, conf = split_prediction_conf(prediction, criterion.with_conf)
            batch_metrics = metrics(prediction.detach(), gt)
            loss = criterion(prediction, gt) if conf is None else criterion(prediction, gt, conf)
            
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()
        
        metric_logger.update(loss=loss_value)
        for k,v in batch_metrics.items():
            metric_logger.update(**{k: v.item()})
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        #if args.distributed: loss_value_reduce = misc.all_reduce_mean(loss_value)
        time_to_log = ((data_iter_step + 1) % (args.tboard_log_step * accum_iter) == 0 or data_iter_step == len_data_loader-1)
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and time_to_log:
            epoch_1000x = int((data_iter_step / len_data_loader + epoch) * 1000)
            # We use epoch_1000x as the x-axis in tensorboard. This calibrates different curves when batch size changes.
            log_writer.add_scalar('train/loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
            for k,v in batch_metrics.items():
                log_writer.add_scalar('train/'+k, v.item(), epoch_1000x)
        elif time_to_log and global_rank == 0:
            wandb.log({
                'train/loss': loss_value_reduce,
                'lr': lr,
                'epoch': epoch,
                **{f'train/{k}': v.item() for k,v in batch_metrics.items()}
            })
            if args.log_flow_stack_every > 0 and (data_iter_step + 1) % (args.log_flow_stack_every * accum_iter) == 0:
                log_flow_stack_to_wandb(
                    image1=image1,
                    image2=image2,
                    gt=gt,
                    pred=prediction,
                    conf=conf,
                    pairname=pairname,
                    step=None,
                    split="train",
                    max_items=4,
                )

    # gather the stats from all processes
    if args.distributed: metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate_one_epoch(model: torch.nn.Module,
                   criterion: torch.nn.Module,
                   metrics: torch.nn.Module,
                   data_loaders: list[Iterable],
                   device: torch.device,
                   epoch: int,
                   global_rank: int,
                   log_writer=None,
                   args=None):

    model.eval()
    metric_loggers = []
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    conf_mode = args.tile_conf_mode
    crop = args.crop
    
    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    results = {}
    dnames = []
    image1, image2, gt, prediction = None, None, None, None

    # only rank 0 runs validation to avoid redundant computation across GPUs
    if global_rank == 0:
        for didx, data_loader in enumerate(data_loaders):
            dname = str(data_loader.dataset)
            dnames.append(dname)
            metric_loggers.append(misc.MetricLogger(delimiter="  "))
            for data_iter_step, (image1, image2, gt, pairname) in enumerate(metric_loggers[didx].log_every(data_loader, print_freq, header)):
                image1 = image1.to(device, non_blocking=True)
                image2 = image2.to(device, non_blocking=True)
                gt = gt.to(device, non_blocking=True)
                if dname.startswith('Spring'):
                    assert gt.size(2)==image1.size(2)*2 and gt.size(3)==image1.size(3)*2
                    gt = (gt[:,:,0::2,0::2] + gt[:,:,0::2,1::2] + gt[:,:,1::2,0::2] + gt[:,:,1::2,1::2] ) / 4.0 # we approximate the gt based on the 2x upsampled ones

                with torch.inference_mode():
                    prediction, tiled_loss, c = tiled_pred(model, criterion, image1, image2, gt, conf_mode=conf_mode, overlap=args.val_overlap, crop=crop, with_conf=criterion.with_conf)
                    batch_metrics = metrics(prediction.detach(), gt)
                    loss = criterion(prediction.detach(), gt) if not criterion.with_conf else criterion(prediction.detach(), gt, c)
                    loss_value = loss.item()
                    metric_loggers[didx].update(loss_tiled=tiled_loss.item())
                    metric_loggers[didx].update(**{f'loss': loss_value})

                    for k,v in batch_metrics.items():
                        metric_loggers[didx].update(**{dname+'_' + k: v.item()})

                    if data_iter_step in [1,2,10,20]:
                        try:
                            log_flow_stack_to_wandb(
                                image1=image1,
                                image2=image2,
                                gt=gt,
                                pred=prediction,
                                conf=c.unsqueeze(1) if c is not None else None,
                                pairname=pairname,
                                step=None,
                                split=f"val/{dname}",
                                max_items=2,
                            )
                        except Exception as e:
                            print(f"[WARN] flow visualization logging failed (val): {e}")

        results = {k: meter.global_avg for ml in metric_loggers for k, meter in ml.meters.items()}
        if len(dnames) > 1:
            # derive metric keys from the loggers rather than the last batch_metrics dict
            prefix = dnames[0] + '_'
            metric_keys = [k[len(prefix):] for k in metric_loggers[0].meters if k.startswith(prefix)]
            for k in metric_keys:
                results['AVG_' + k] = sum(results[dname + '_' + k] for dname in dnames) / len(dnames)

    # broadcast results from rank 0 to all other ranks
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        results_list = [results]
        dist.broadcast_object_list(results_list, src=0)
        results = results_list[0]
            
    if log_writer is not None :
        epoch_1000x = int((1 + epoch) * 1000)
        for k,v in results.items():
            log_writer.add_scalar('val/'+k, v, epoch_1000x)
    elif global_rank == 0:
        wandb.log({f'val/{k}': v for k,v in results.items()})

    print("Averaged stats:", results)
    return results

import torch.nn.functional as F
def _resize_img(img, new_size):
    return F.interpolate(img, size=new_size, mode='bicubic', align_corners=False)
def _resize_stereo_or_flow(data, new_size):
    assert data.ndim==4
    assert data.size(1) in [1,2]
    scale_x = new_size[1]/float(data.size(3))
    out = F.interpolate(data, size=new_size, mode='bicubic', align_corners=False)
    out[:,0,:,:] *= scale_x
    if out.size(1)==2:
        scale_y = new_size[0]/float(data.size(2))        
        out[:,1,:,:] *= scale_y
        # print(scale_x, new_size, data.shape)
    return out
    

@torch.no_grad()
def tiled_pred(model, criterion, img1, img2, gt,
               overlap=0.5, bad_crop_thr=0.05,
               downscale=False, crop=512, ret='loss',
               conf_mode='conf_expsigmoid_10_5', with_conf=False, 
               return_time=False):
                     
    # for each image, we are going to run inference on many overlapping patches
    # then, all predictions will be weighted-averaged
    if gt is not None:
        B, C, H, W = gt.shape
    else:
        B, _, H, W = img1.shape
        C = model.head.num_channels-int(with_conf)
    win_height, win_width = crop[0], crop[1]
    
    # upscale to be larger than the crop
    do_change_scale =  H<win_height or W<win_width
    if do_change_scale: 
        upscale_factor = max(win_width/W, win_height/H)
        original_size = (H,W)
        new_size = (round(H*upscale_factor),round(W*upscale_factor))
        img1 = _resize_img(img1, new_size)
        img2 = _resize_img(img2, new_size)
        # resize gt just for the computation of tiled losses
        if gt is not None: gt = _resize_stereo_or_flow(gt, new_size)
        H,W = img1.shape[2:4]
        
    if conf_mode.startswith('conf_expsigmoid_'): # conf_expsigmoid_30_10
        beta, betasigmoid = map(float, conf_mode[len('conf_expsigmoid_'):].split('_'))
    elif conf_mode.startswith('conf_expbeta'): # conf_expbeta3
        beta = float(conf_mode[len('conf_expbeta'):])
    else:
        raise NotImplementedError(f"conf_mode {conf_mode} is not implemented")

    def crop_generator():
        for sy in _overlapping(H, win_height, overlap):
          for sx in _overlapping(W, win_width, overlap):
            yield sy, sx, sy, sx, True

    # keep track of weighted sum of prediction*weights and weights
    accu_pred = img1.new_zeros((B, C, H, W)) # accumulate the weighted sum of predictions 
    accu_conf = img1.new_zeros((B, H, W)) + 1e-16 # accumulate the weights 
    accu_c = img1.new_zeros((B, H, W)) # accumulate the weighted sum of confidences ; not so useful except for computing some losses

    tiled_losses = []
    
    if return_time:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

    for sy1, sx1, sy2, sx2, aligned in crop_generator():
        # compute optical flow there
        pred =  model(_crop(img1,sy1,sx1), _crop(img2,sy2,sx2))
        pred, predconf = split_prediction_conf(pred, with_conf=with_conf)
        
        if gt is not None: gtcrop = _crop(gt,sy1,sx1)
        if criterion is not None and gt is not None: 
            tiled_losses.append( criterion(pred, gtcrop).item() if predconf is None else criterion(pred, gtcrop, predconf).item() )
        
        if conf_mode.startswith('conf_expsigmoid_'):
            conf = torch.exp(- beta * 2 * (torch.sigmoid(predconf / betasigmoid) - 0.5)).view(B,win_height,win_width)
        elif conf_mode.startswith('conf_expbeta'):
            conf = torch.exp(- beta * predconf).view(B,win_height,win_width)
        else:
            raise NotImplementedError
                        
        accu_pred[...,sy1,sx1] += pred * conf[:,None,:,:]
        accu_conf[...,sy1,sx1] += conf
        accu_c[...,sy1,sx1] += predconf.view(B,win_height,win_width) * conf 
        
    pred = accu_pred / accu_conf[:, None,:,:]
    c = accu_c / accu_conf
    assert not torch.any(torch.isnan(pred))

    if return_time:
        end.record()
        torch.cuda.synchronize()
        time = start.elapsed_time(end)/1000.0 # this was in milliseconds

    if do_change_scale:
        pred = _resize_stereo_or_flow(pred, original_size)
    
    if return_time:
        return pred, torch.mean(torch.tensor(tiled_losses)), c, time
    return pred, torch.mean(torch.tensor(tiled_losses)), c


def _overlapping(total, window, overlap=0.5):
    assert total >= window and 0 <= overlap < 1, (total, window, overlap)
    num_windows = 1 + int(np.ceil( (total - window) / ((1-overlap) * window) ))
    offsets = np.linspace(0, total-window, num_windows).round().astype(int)
    yield from (slice(x, x+window) for x in offsets)

def _crop(img, sy, sx):
    B, THREE, H, W = img.shape
    if 0 <= sy.start and sy.stop <= H and 0 <= sx.start and sx.stop <= W:
        return img[:,:,sy,sx]
    l, r = max(0,-sx.start), max(0,sx.stop-W)
    t, b = max(0,-sy.start), max(0,sy.stop-H)
    img = torch.nn.functional.pad(img, (l,r,t,b), mode='constant')
    return img[:, :, slice(sy.start+t,sy.stop+t), slice(sx.start+l,sx.stop+l)]


@torch.no_grad()
def log_flow_stack_to_wandb(
    image1,
    image2,
    gt,
    pred,
    conf=None,
    pairname=None,
    step=None,
    split="train",
    max_items=2,
):
    """
    Logs a stacked flow visualization to WandB.

    Inputs:
      image1, image2: [B,3,H,W] tensors (normalized to [0,1] or roughly image range)
      gt:             [B,2,H,W] tensor (flow GT)
      pred:           [B,2,H,W] tensor (flow prediction)
      conf:           [B,1,H,W] or [B,H,W] tensor or None
      pairname:       list[str] or tuple[str] or None
      step:           global step for wandb.log(...)
      split:          "train" or "val"
      max_items:      how many samples from batch to log
    """
    if image1 is None or image2 is None or gt is None or pred is None:
        return

    # Move to CPU once
    image1 = image1.detach().float().cpu()
    image2 = image2.detach().float().cpu()
    gt = gt.detach().float().cpu()
    pred = pred.detach().float().cpu()

    if conf is not None:
        conf = conf.detach().float().cpu()
        if conf.ndim == 4 and conf.shape[1] == 1:
            conf = conf[:, 0]  # [B,H,W]

    B = image1.shape[0]
    n = min(B, max_items)

    images_to_log = []
    captions = []

    for i in range(n):
        # --- tensors -> numpy ---
        img1_np = unnormalize_imagenet(image1[i]).permute(1, 2, 0).numpy()
        img2_np = unnormalize_imagenet(image2[i]).permute(1, 2, 0).numpy()

        # If normalized to [0,1], this is fine. If normalized differently, clamp still prevents garbage.
        img1_np = (np.clip(img1_np, 0.0, 1.0) * 255.0).astype(np.uint8)
        img2_np = (np.clip(img2_np, 0.0, 1.0) * 255.0).astype(np.uint8)

        gt_np = gt[i].permute(1, 2, 0).numpy().copy()    # [H,W,2]
        pred_np = pred[i].permute(1, 2, 0).numpy().copy()

        # valid mask from GT (KITTI-style invalids are inf)
        valid = np.isfinite(gt_np[..., 0]) & np.isfinite(gt_np[..., 1])

        # shared maxflow => fair color comparison
        # guard invalids when computing max
        gt_vis_for_norm = gt_np.copy()
        gt_vis_for_norm[~valid] = 0.0
        pred_vis_for_norm = pred_np.copy()
        pred_vis_for_norm[~valid] = 0.0 if valid.shape == pred_vis_for_norm[..., 0].shape else pred_vis_for_norm[~valid]

        try:
            maxflow = max(float(flowMaxNorm(gt_vis_for_norm)), float(flowMaxNorm(pred_vis_for_norm)))
        except Exception:
            maxflow = None
        if maxflow is not None and (not np.isfinite(maxflow) or maxflow <= 0):
            maxflow = None

        # flow color images
        gt_rgb = flowToColor(gt_np.copy(), maxflow=maxflow)
        pred_rgb = flowToColor(pred_np.copy(), maxflow=maxflow)

        # EPE heatmap-like grayscale (simple and robust)
        epe = np.linalg.norm(pred_np - gt_np, axis=2)
        epe[~valid] = np.nan
        epe_mean = float(np.nanmean(epe)) if np.any(valid) else float("nan")

        # robust normalize epe for display
        if np.any(valid):
            epe_vis = epe.copy()
            # cap at 95th percentile for readability
            epe_cap = np.nanpercentile(epe_vis, 95.0)
            if not np.isfinite(epe_cap) or epe_cap <= 1e-6:
                epe_cap = 1.0
            epe_vis = np.nan_to_num(epe_vis, nan=0.0, posinf=epe_cap, neginf=0.0)
            epe_vis = np.clip(epe_vis / epe_cap, 0.0, 1.0)
            epe_vis = (epe_vis * 255.0).astype(np.uint8)
            epe_vis = np.stack([epe_vis, epe_vis, epe_vis], axis=-1)  # RGB grayscale
        else:
            epe_vis = np.zeros((*gt_rgb.shape[:2], 3), dtype=np.uint8)

        # confidence visualization (if available)
        conf_rgb = None
        if conf is not None:
            c_np = conf[i].numpy()
            c_np = np.nan_to_num(c_np, nan=0.0, posinf=0.0, neginf=0.0)
            # normalize per-sample for display
            c_min, c_max = float(np.min(c_np)), float(np.max(c_np))
            if c_max > c_min:
                c_vis = (c_np - c_min) / (c_max - c_min)
            else:
                c_vis = np.zeros_like(c_np)
            c_vis = (c_vis * 255.0).astype(np.uint8)
            conf_rgb = np.stack([c_vis, c_vis, c_vis], axis=-1)

        # # make a single stacked panel (vertical stack)
        # panels = [img1_np, img2_np, gt_rgb, pred_rgb, epe_vis]
        # if conf_rgb is not None:
        #     panels.insert(4, conf_rgb)  # before EPE

        # # ensure same height/width already; concatenate vertically
        # stacked = np.concatenate(panels, axis=0)

        # build rows
        rows = []
        rows.append(np.concatenate([img1_np, img2_np], axis=1))     # row 1
        rows.append(np.concatenate([gt_rgb, pred_rgb], axis=1))     # row 2
        if conf_rgb is not None:
            rows.append(np.concatenate([conf_rgb, epe_vis], axis=1))  # row 3
        else:
            rows.append(np.concatenate([epe_vis, epe_vis], axis=1))   # optional placeholder row

        # stack rows vertically
        grid = np.concatenate(rows, axis=0)

        # caption
        name_str = ""
        if pairname is not None:
            try:
                name_str = str(pairname[i])
            except Exception:
                name_str = str(pairname)

        caption = f"{split} | idx={i}"
        if name_str:
            caption += f" | {name_str}"
        if np.isfinite(epe_mean):
            caption += f" | EPE={epe_mean:.3f}"
        if maxflow is not None:
            caption += f" | maxflow={maxflow:.2f}"

        images_to_log.append(wandb.Image(grid, caption=caption))
        captions.append(caption)

    if images_to_log:
        payload = {f"{split}/flow_stack": images_to_log}
        if step is None:
            wandb.log(payload)
        else:
            wandb.log(payload, step=step)