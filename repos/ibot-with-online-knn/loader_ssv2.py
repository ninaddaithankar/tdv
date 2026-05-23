# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import random
import argparse
import numpy as np

import torch
from torch.utils.data import Dataset

from data.something_dataloader import SomethingDataset


class SomethingDatasetMask(Dataset):
    """
    Wrapper around SomethingDataset for iBOT pre-training.

    SomethingDataset samples `num_frames` frames per video (spaced
    `time_between_frames` seconds apart) and applies the iBOT multi-crop
    transform to each frame. This wrapper flattens the resulting
    (num_videos × num_frames) frames into a flat index so each frame is
    an independent sample for the DataLoader, then generates iBOT block/random
    masks for each crop view.

    Returns (crops, label, masks) per item — identical format to ImageFolderMask.
    """

    def __init__(self, dataset_dir, split, transform,
                 patch_size, pred_ratio, pred_ratio_var, pred_aspect_ratio,
                 pred_shape='block', pred_start_epoch=0,
                 labels_dir='labels/', num_frames=16, time_between_frames=0.25):

        hparams = argparse.Namespace(
            context_length=num_frames,
            time_between_frames=time_between_frames,
            sampling_rate=0,
            use_raw_framerate=False,
            no_randomness_dataloader=False,
            preencode_dataset=False,
            use_preencoded_dataset=False,
            preprocess_data=False,
            crop_all_samples=False,
            ffprobe_path=None,
            hdf5_file_path=None,
            debug_mode=False,
            dataset_dir=dataset_dir,
        )

        self.inner = SomethingDataset(
            hparams=hparams,
            split=split,
            transform=transform,
            dataset_dir=dataset_dir,
            labels_dir=labels_dir,
        )
        self.num_frames = num_frames
        self.epoch = 0

        # ---- masking parameters (mirrors ImageFolderMask) ----
        self.psz = patch_size
        self.pred_ratio = (
            pred_ratio[0]
            if isinstance(pred_ratio, list) and len(pred_ratio) == 1
            else pred_ratio
        )
        self.pred_ratio_var = (
            pred_ratio_var[0]
            if isinstance(pred_ratio_var, list) and len(pred_ratio_var) == 1
            else pred_ratio_var
        )
        if isinstance(self.pred_ratio, list) and not isinstance(self.pred_ratio_var, list):
            self.pred_ratio_var = [self.pred_ratio_var] * len(self.pred_ratio)
        self.log_aspect_ratio = tuple(map(math.log, pred_aspect_ratio))
        self.pred_shape = pred_shape
        self.pred_start_epoch = pred_start_epoch

    # ------------------------------------------------------------------
    # Epoch tracking (called by main_ibot.py each epoch)
    # ------------------------------------------------------------------

    def set_epoch(self, epoch):
        self.epoch = epoch

    # ------------------------------------------------------------------
    # Mask helpers (identical logic to ImageFolderMask)
    # ------------------------------------------------------------------

    def get_pred_ratio(self):
        if self.epoch < self.pred_start_epoch:
            return 0

        if isinstance(self.pred_ratio, list):
            pred_ratio = []
            for prm, prv in zip(self.pred_ratio, self.pred_ratio_var):
                assert prm >= prv
                pr = random.uniform(prm - prv, prm + prv) if prv > 0 else prm
                pred_ratio.append(pr)
            return random.choice(pred_ratio)
        else:
            assert self.pred_ratio >= self.pred_ratio_var
            return (
                random.uniform(
                    self.pred_ratio - self.pred_ratio_var,
                    self.pred_ratio + self.pred_ratio_var,
                )
                if self.pred_ratio_var > 0
                else self.pred_ratio
            )

    def _make_mask(self, img_tensor):
        """Generate a 2-D boolean mask for one crop tensor [C, H, W]."""
        H = img_tensor.shape[1] // self.psz
        W = img_tensor.shape[2] // self.psz
        high = self.get_pred_ratio() * H * W

        if self.pred_shape == 'block':
            mask = np.zeros((H, W), dtype=bool)
            mask_count = 0
            while mask_count < high:
                max_mask_patches = high - mask_count
                delta = 0
                for _ in range(10):
                    low = (min(H, W) // 3) ** 2
                    target_area = random.uniform(low, max_mask_patches)
                    aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                    h = int(round(math.sqrt(target_area * aspect_ratio)))
                    w = int(round(math.sqrt(target_area / aspect_ratio)))
                    if w < W and h < H:
                        top = random.randint(0, H - h)
                        left = random.randint(0, W - w)
                        num_masked = mask[top: top + h, left: left + w].sum()
                        if 0 < h * w - num_masked <= max_mask_patches:
                            for i in range(top, top + h):
                                for j in range(left, left + w):
                                    if not mask[i, j]:
                                        mask[i, j] = True
                                        delta += 1
                    if delta > 0:
                        break
                if delta == 0:
                    break
                mask_count += delta

        elif self.pred_shape == 'rand':
            mask = np.hstack([
                np.zeros(H * W - int(high)),
                np.ones(int(high)),
            ]).astype(bool)
            np.random.shuffle(mask)
            mask = mask.reshape(H, W)

        else:
            raise ValueError(f"Unknown pred_shape: {self.pred_shape}")

        return mask

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        # Each video contributes num_frames independent samples
        return len(self.inner)

    def __getitem__(self, idx):
        try:
            # SomethingDataset returns:
            #   frames[j] = torch.stack of shape [num_frames, C, H, W]
            #              (crop-j view of every frame in the clip)
            frames, _ = self.inner[idx]
            label = self.inner.labels[idx]

            # Generate masks for every frame of every crop view.
            # masks[j] = np.ndarray [num_frames, H_grid, W_grid]
            masks = [
                np.stack([self._make_mask(frames[j][t]) for t in range(self.num_frames)])
                for j in range(len(frames))
            ]

            # frames[j]: [num_frames, C, H, W]  — returned as-is
            return frames, label, masks

        except Exception as exc:
            print(f"[SomethingDatasetMask] skipping video {idx}: {exc}")
            return self[random.randint(0, len(self) - 1)]
