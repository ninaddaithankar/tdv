import math
import numpy as np
import torch, random, cv2, av
from PIL import Image

from data.utils.data_preprocessor import crop_to_square


def sample_frame_indices(start_sec, end_sec, fps, time_between_frames, num_frames, sampling_rate = 0) -> torch.Tensor:
    """
    Returns a tensor of length num_frames with frame indices
    (int) relative to the *whole* video file.
    """
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

def new_sample_frame_indices(
    total_frames: int,
    fps: float,
    time_between_frames: float,
    num_frames: int,
    sampling_rate: int = 0,
    use_raw_framerate: bool = False,
    no_randomness_dataloader: bool = False
) -> torch.Tensor:
    """
    Returns a list of frame indices robustly sampled from the video.

    Args:
        total_frames (int): total decodable frames in the video
        fps (float): video framerate
        time_between_frames (float): spacing in seconds (ignored if sampling_rate > 0)
        num_frames (int): number of frames to return
        sampling_rate (int): spacing in frames (overrides time_between_frames if > 0)
        use_raw_framerate (bool): if True, sample contiguous frames at raw framerate
        no_randomness_dataloader (bool): if True, always start at frame 0

    Returns:
        Tensor of shape (num_frames,) with valid frame indices
    """
    if total_frames <= 0:
        raise ValueError("Video has no decodable frames")

    # --- Case 1: too short video ---
    if total_frames < num_frames:
        # Duplicate frames uniformly to meet length
        return torch.linspace(0, total_frames-1, steps=num_frames).long()

    # --- Case 2: contiguous frames at raw fps ---
    if use_raw_framerate:
        if no_randomness_dataloader:
            start_frame = 0
        else:
            start_frame = random.randint(0, total_frames - num_frames)
        return torch.arange(start_frame, start_frame + num_frames).long()

    # --- Case 3: fixed sampling_rate (in frames) ---
    if sampling_rate > 0:
        required_frames = sampling_rate * (num_frames - 1) + 1
        if required_frames > total_frames:
            # fallback: uniform spacing
            return torch.linspace(0, total_frames-1, steps=num_frames).long()
        if no_randomness_dataloader:
            start_frame = 0
        else:
            start_frame = random.randint(0, total_frames - required_frames)
        return torch.arange(start_frame, start_frame + required_frames, step=sampling_rate).long()

    # --- Case 4: spacing in seconds (time_between_frames) ---
    required_length = time_between_frames * (num_frames - 1)
    video_length = total_frames / fps

    if required_length > video_length:
        # fallback: uniform spacing
        return torch.linspace(0, total_frames-1, steps=num_frames).long()

    if no_randomness_dataloader:
        start_time = 0
    else:
        start_time = random.uniform(0, video_length - required_length)
    end_time = start_time + required_length

    times = torch.linspace(start_time, end_time, steps=num_frames)
    indices = (times * fps).long().clamp(max=total_frames-1)
    return indices



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

    if transform:
        frames_out = [torch.stack(same_crops) for same_crops in zip(*frames_out)]
        return frames_out

    return torch.stack(frames_out)