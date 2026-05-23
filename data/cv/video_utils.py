import math
import numpy as np
import torch, random, cv2
from PIL import Image

from data.data_preprocessor import crop_to_square


def sample_frame_indices(start_sec, end_sec, fps, time_between_frames, num_frames, sampling_rate = 0, total_frames=None) -> torch.Tensor:
    """
    Returns a tensor of length num_frames with frame indices
    (int) relative to the *whole* video file.
    """
    if total_frames is None:
        total_frames = math.floor((end_sec - start_sec) * fps)

    if total_frames < num_frames:
        raise ValueError("Moment too short for the number of frames requested")

    # -- strategy: sampling_rate (in frames) wins over time_between_sec
    if sampling_rate > 0:
        needed = sampling_rate * (num_frames - 1)
        start_frame = random.randint(0, max(0, total_frames - needed - 1))
        idxs = torch.arange(start_frame,
                            start_frame + needed + 1,
                            step=sampling_rate)
        return (start_sec * fps + idxs).long().tolist()

    # -- otherwise use constant seconds between frames
    needed_time = time_between_frames * (num_frames - 1)
    if needed_time > (end_sec - start_sec):
        # fallback: uniform on moment duration
        t = torch.linspace(start_sec, end_sec,
                            steps=num_frames)
    else:
        t0 = random.uniform(start_sec, end_sec - needed_time)
        t = torch.linspace(t0, t0 + needed_time,
                            steps=num_frames)
    return (t * fps).long().tolist()


def get_total_frames(filepath):
    cap = cv2.VideoCapture(filepath)
    estimate = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    count=0
    for i in range(estimate):
        ret, img = cap.read()
        if not ret: break
        count += 1
        
    return count


def sample_frame_indices_no_dups(
    start_sec,
    end_sec,
    fps,
    time_between_frames,
    num_frames,
    sampling_rate=0,
    total_frames=None,
):
    """
    Returns list[int] of length num_frames with NO duplicate indices.
    Raises if impossible.
    """
    duration = end_sec - start_sec
    if duration <= 0:
        raise ValueError(f"Bad segment duration: {duration}")

    # If user provided sampling_rate (in frames), this dominates.
    if sampling_rate > 0:
        if total_frames is None:
            total_frames = math.floor(duration * fps)

        needed = sampling_rate * (num_frames - 1)
        if total_frames <= needed:
            raise ValueError(
                f"Segment too short for sampling_rate={sampling_rate}: "
                f"total_frames={total_frames}, need>={needed+1}"
            )
        start_frame = random.randint(0, total_frames - needed - 1)
        idxs = torch.arange(start_frame, start_frame + needed + 1, step=sampling_rate)
        out = (start_sec * fps + idxs).long().tolist()

        # sanity: ensure strict uniqueness
        if len(out) != len(set(out)):
            raise RuntimeError("Unexpected duplicates with sampling_rate path")
        return out

    # --- strict time spacing path (NO fallback)
    needed_time = time_between_frames * (num_frames - 1)
    if duration < needed_time:
        raise ValueError(
            f"Segment too short: fps={fps} total_frames={total_frames} duration={duration:.4f}s < needed_time={needed_time:.4f}s "
            f"(num_frames={num_frames}, dt={time_between_frames})"
        )

    # Also ensure dt maps to >=1 frame step; otherwise duplicates are inevitable
    if time_between_frames * fps < 1.0:
        raise ValueError(
            f"time_between_frames too small for fps: dt*fps={time_between_frames*fps:.3f} < 1. "
            f"Use dt >= {1.0/fps:.6f} or set sampling_rate >= 1 frame."
        )

    t0 = random.uniform(start_sec, end_sec - needed_time)
    t = torch.linspace(t0, t0 + needed_time, steps=num_frames)

    # use round instead of floor to reduce boundary collisions
    idx = torch.round(t * fps).long()

    # clamp into valid frame range for segment
    if total_frames is None:
        total_frames = math.floor(duration * fps)
    seg0 = int(round(start_sec * fps))
    seg1 = seg0 + total_frames - 1
    idx = idx.clamp(seg0, seg1)

    out = idx.tolist()

    # final guarantee: raise if any duplicates remain
    if len(out) != len(set(out)):
        raise ValueError(
            f"Duplicate indices produced even though checks passed. "
            f"(This can happen with weird fps/timebase). idx={out}"
        )

    return out



def read_frame(video_path, cap, fi, transform=None, crop_all_samples=False):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
    ret, frame_bgr = cap.read()
    if not ret:
        raise RuntimeError(f"bad frame {fi} in {video_path}")

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    if crop_all_samples:
        frame_rgb = crop_to_square(frame_rgb)

    if transform:
        pil = Image.fromarray(frame_rgb)
        frame_tensor = transform(pil)
    else:
        frame_tensor = torch.from_numpy(frame_rgb).permute(2,0,1)/255.
    return frame_tensor


def read_selected_frames(video_path, container, stream, frame_indices, transform=None):
    frames_out = []
    frame_indices_set = set(frame_indices)
    max_index = max(frame_indices)

    for i, frame in enumerate(container.decode(stream)):
        if i > max_index:
            break
        if i in frame_indices_set:
            img = frame.to_image()
            if transform:
                frames_out.append(transform(img))
            else:
                arr = np.asarray(img)
                tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.
                frames_out.append(tensor)

    if not frames_out or len(frames_out) != len(frame_indices):
        print(frame_indices)
        print(len(frames_out), len(frame_indices))
        raise RuntimeError(f"No frames read from {video_path}. Check frame indices: {frame_indices}")

    return torch.stack(frames_out)