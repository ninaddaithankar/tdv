import os
import glob
import csv

import torch
import torchvision.transforms.functional as TF

from torch.utils.data import Dataset
from PIL import Image
from typing import Dict, List, Optional


class MOT17Sequence(Dataset):
    """
    One sequence from MOT17.
    Provides:
      - frames: Tensor list (T, C, H, W) uint8
      - detections: List[List[boxes]] per frame (xyxy float)
      - gt_tracks: dict frame -> list of (id, x1,y1,x2,y2)
    Choose detection_source:
      - 'public' -> use det/det.txt
      - 'gt'     -> use gt/gt.txt as detections (good for appearance-only probing)
    """
    def __init__(self,
                 seq_root: str,
                 detection_source: str = "gt",
                 det_score_thresh: float = 0.7,
                 max_frames: Optional[int] = 100,
                 to_rgb: bool = True):
        """
        seq_root: path to sequence, e.g. /shared/dir/user/datasets/mot17/MOT17/train/MOT17-13-FRCNN
        """
        assert detection_source in {"public", "gt"}
        self.seq_root = seq_root
        self.img_dir = os.path.join(seq_root, "img1")
        self.gt_file = os.path.join(seq_root, "gt", "gt.txt")
        self.det_file = os.path.join(seq_root, "det", "det.txt")
        self.detection_source = detection_source
        self.det_score_thresh = det_score_thresh
        self.max_frames = max_frames
        self.to_rgb = to_rgb

        self.frame_paths = _frames_list(self.img_dir)
        if self.max_frames:
            self.frame_paths = self.frame_paths[: self.max_frames]
        self.T = len(self.frame_paths)

        # Load GT (for evaluation)
        if not os.path.isfile(self.gt_file):
            raise FileNotFoundError(f"Missing GT file: {self.gt_file}")
        gt_rows = _read_mot_csv(self.gt_file)
        self.gt_by_frame = _group_by_frame_boxes(gt_rows, score_thresh=0.0, use_gt_conf=True)

        # Load detections
        if self.detection_source == "public":
            if not os.path.isfile(self.det_file):
                raise FileNotFoundError(
                    f"Missing public detections: {self.det_file}\n"
                    f"To use public detections, download MOT17Det or the det/ folders for MOT17 sequences."
                )
            det_rows = _read_mot_csv(self.det_file)
            self.det_by_frame = _group_by_frame_boxes(det_rows, score_thresh=self.det_score_thresh, use_gt_conf=False)
        else:
            # Use GT boxes as detections (score=1.0)
            self.det_by_frame = {}
            for f, gts in self.gt_by_frame.items():
                dets = []
                for g in gts:
                    x1, y1, x2, y2, _, tid = g
                    dets.append([x1, y1, x2, y2, 1.0, -1])
                self.det_by_frame[f] = dets

    def __len__(self):
        # Each __getitem__ returns the WHOLE sequence (designed for batch_size=1)
        return 1

    def _load_frame_tensor(self, path: str) -> torch.Tensor:
        im = Image.open(path).convert("RGB" if self.to_rgb else "BGR")
        # uint8 C,H,W
        t = TF.pil_to_tensor(im)  # (C,H,W) uint8
        return t

    def __getitem__(self, idx):
        # Load all frames (can be heavy; keep num_workers low or add caching if needed)
        frames = [self._load_frame_tensor(p) for p in self.frame_paths]
        frames = torch.stack(frames, dim=0)  # (T, C, H, W), uint8

        # Per-frame detections (xyxy), list of lists
        detections = []
        gt_tracks = {}  # frame -> list of (id, x1,y1,x2,y2)

        for t in range(1, self.T + 1):  # MOT frames are 1-based
            dets = self.det_by_frame.get(t, [])
            gts = self.gt_by_frame.get(t, [])
            # for gt_tracks, keep (id, x1,y1,x2,y2)
            gt_tracks[t] = [(int(rec[5]), rec[0], rec[1], rec[2], rec[3]) for rec in gts]
            # for detections list, keep only boxes (and optionally scores)
            detections.append([[rec[0], rec[1], rec[2], rec[3], rec[4]] for rec in dets])

        sample = {
            "sequence_name": os.path.basename(self.seq_root.rstrip("/")),
            "video": frames,                # (T,C,H,W) uint8
            "boxes": detections,            # List[T] of List[xyxy]
            "gt_tracks": gt_tracks,         # Dict[int, List[(id,x1,y1,x2,y2)]]
            "orig_size": frames.shape[2:],  # (H,W)
        }
        return sample


def _read_mot_csv(path: str) -> List[List[float]]:
    """
    Reads a MOTChallenge-style CSV (gt.txt or det.txt).
    Returns a list of rows, each row = [frame, id, x, y, w, h, conf, class, vis]
    Some files (det.txt) may not have class/vis; we fill with defaults.
    """
    rows = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for r in reader:
            if not r:
                continue
            r = [float(x) for x in r]
            # Pad missing cols to length 9
            while len(r) < 9:
                r.append(-1.0)
            rows.append(r)
    return rows


def _group_by_frame_boxes(rows: List[List[float]],
                          score_thresh: float = 0.0,
                          use_gt_conf: bool = False) -> Dict[int, List[List[float]]]:
    """
    Group rows by frame index.
    For detections, we filter by 'conf' >= score_thresh.
    For gt, 'conf' is usually 1; use_gt_conf lets you still filter or ignore.
    Returns dict: frame -> list of [x1,y1,x2,y2,score,id(optional)]
    Note: coordinates in MOT are 1-based, top-left x,y, width, height in pixels.
    We'll convert to 0-based xyxy here.
    """
    by_f = {}
    for row in rows:
        f, tid, x, y, w, h, conf = int(row[0]), int(row[1]), row[2], row[3], row[4], row[5], row[6]
        if w <= 0 or h <= 0:
            continue
        if not use_gt_conf and score_thresh > 0.0 and conf < score_thresh:
            continue
        # Convert to 0-based xyxy
        x1 = max(0.0, x - 1.0)
        y1 = max(0.0, y - 1.0)
        x2 = x1 + w
        y2 = y1 + h
        # store: [x1,y1,x2,y2,score,id]
        if int(tid) < 0:  # public detections often have -1 id
            rec = [x1, y1, x2, y2, float(conf), -1]
        else:
            rec = [x1, y1, x2, y2, float(conf), int(tid)]
        by_f.setdefault(f, []).append(rec)
    return by_f


def _frames_list(img_dir: str) -> List[str]:
    """
    Returns sorted list of frame image paths in img1/*.jpg (MOT17 standard).
    """
    paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    if not paths:  # some releases use png
        paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    if not paths:
        raise FileNotFoundError(f"No frame images found in {img_dir}")
    return paths

