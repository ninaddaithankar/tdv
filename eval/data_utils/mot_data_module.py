import os
import glob
import torch

from typing import List, Optional
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

from data.cv.mot17_dataloader import MOT17Sequence


class MOT17DataModule(LightningDataModule):
    """
    dataroot should contain MOT17 structure, e.g.:
      {dataroot}/train/MOT17-02-FRCNN
      {dataroot}/train/MOT17-04-FRCNN
      ...
    Split can be 'train' or 'test' depending on where your sequences live.
    """
    def __init__(self, args):
        super().__init__()

        self.dataset_root = args.mot_eval_data_dir
        self.split = "train"
        self.detection_source = "public"  # 'public' or 'gt'
        self.det_score_thresh = 0.4
        self.max_frames_per_seq = 500
        self.num_workers = args.num_workers

        # NEW: optionally restrict to one sequence
        self.seq_name = "MOT17-13-FRCNN"
        self.seq_paths: List[str] = []
        self.dataset: Optional[Dataset] = None

    def setup(self, stage=None):
        seq_glob = os.path.join(self.dataset_root, self.split, "*")
        all_seqs = sorted([p for p in glob.glob(seq_glob) if os.path.isdir(p)])

         # Filter if seq_name is provided
        if self.seq_name is not None:
            all_seqs = [s for s in all_seqs if self.seq_name in os.path.basename(s)]
            if not all_seqs:
                raise ValueError(f"Requested seq_name={self.seq_name}, but not found under {seq_glob}")

        if not all_seqs:
            raise FileNotFoundError(
                f"No MOT17 sequences found under: {seq_glob}\n"
                f"Expected layout like {self.dataset_root}/{self.split}/MOT17-02-FRCNN/img1/*.jpg"
            )

        # For validation/eval, we typically wrap ONE sequence per DataLoader item (batch_size=1).
        # If you want to iterate multiple sequences, we wrap them into a “multi-seq dataset”
        # whose __getitem__ returns a full sequence at a time.

        class _MultiSeq(torch.utils.data.Dataset):
            def __init__(self, seq_paths, ds_kwargs):
                self.seq_paths = seq_paths
                self.ds_kwargs = ds_kwargs

            def __len__(self):
                return len(self.seq_paths)

            def __getitem__(self, i):
                seq_ds = MOT17Sequence(self.seq_paths[i], **self.ds_kwargs)
                return seq_ds[0]

        ds_kwargs = dict(
            detection_source=self.detection_source,
            det_score_thresh=self.det_score_thresh,
            max_frames=self.max_frames_per_seq,
        )
        self.dataset = _MultiSeq(all_seqs, ds_kwargs)

    def val_dataloader(self):
        assert self.dataset is not None
        # batch_size=1 => one sequence per step
        return DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_one_seq,
            pin_memory=True,
        )

    @staticmethod
    def _collate_one_seq(batch_list):
        # batch_list length is 1 (batch_size=1). Unwrap cleanly.
        assert len(batch_list) == 1, "Set batch_size=1 for MOT sequence eval."
        return batch_list[0]

