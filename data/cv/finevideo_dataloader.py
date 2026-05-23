import os
import json
import random
import av

from torch.utils.data import Dataset
from torchvision import transforms
from datasets import load_dataset

from data.cv import video_utils
from data.data_preprocessor import handle_corrupt_file

class FineVideoDataset(Dataset):
	"""
	FineVideo Dataset Loader.
	- If preprocessed: Reads from local JSONL manifest and trimmed preprocessed clips.
	- If raw: Uses Hugging Face 'HuggingFaceFV/finevideo' for metadata and video access.
	"""
	def __init__(
		self,
		hparams,                     
		split: str,                                             
		dataset_dir: str = None,                                
		transform: transforms.Compose = None,
	):
		super().__init__()
		self.hparams = hparams
		self.dataset_dir = dataset_dir if dataset_dir is not None else hparams.dataset_dir
		self.use_processed = getattr(self.hparams, "use_preprocessed_finevideo", False)
		
		self.transform = transform
		self.num_frames = hparams.context_length
		self.time_between_frames = hparams.time_between_frames
		self.corrupt_files_path = "data/cv/corrupt_files/finevideo.txt"

		if self.use_processed:
			# local preprocessed clips
			self.base_dir = os.path.join(self.dataset_dir, f"preprocessed_train")
			self.manifest_path = os.path.join(self.base_dir, "manifest.jsonl")
			self.clips_dir = os.path.join(self.base_dir, "clips")
			self.meta = self._read_local_manifest()
		else:
			# hf dataset
			print(f"Loading FineVideo ({split}) from Hugging Face Hub...")
			self.hf_dataset = load_dataset("HuggingFaceFV/finevideo", split=split, streaming=False)
			self.meta = self.hf_dataset 

		if split == 'val':
			# if using streaming=True, slicing logic would need to change to .take(1000)
			self.meta = self.meta[:100]

	def __len__(self):
		return len(self.meta)
	
	def __getitem__(self, idx):
		try:
			if self.use_processed:
				info = self.meta[idx]
				video_path = info["video_path"]
				start_sec, end_sec = 0, info["duration"]
			else:
				# accessing raw FineVideo from HF
				sample = self.meta[idx]
				video_path = sample['mp4'] 
				
				# finevideo stores detailed metadata here
				metadata = sample['json']
				start_sec = 0
				end_sec = metadata.get('duration', 0)

			with av.open(video_path) as container:
				stream = container.streams.video[0]
				fps = float(stream.average_rate)

				total_frames = None
				if stream.frames > 0:
					total_frames = stream.frames
					end_sec = total_frames / fps
				
				frame_idxs = video_utils.sample_frame_indices(
					start_sec, 
					end_sec, 
					fps, 
					time_between_frames=self.time_between_frames, 
					num_frames=self.num_frames,
					total_frames=total_frames
				)
				
				clip = video_utils.read_selected_frames(video_path, container, stream, frame_idxs, self.transform)
				return clip

		except Exception as e:
			# for HF, this might trigger if download fails
			path_to_log = video_path if isinstance(video_path, str) else "hf_stream_error"
			handle_corrupt_file(e, path_to_log, log_path=self.corrupt_files_path)
			return self[random.randint(0, len(self)-1)]
		
		
	def _read_local_manifest(self):
		"""Parses the JSONL manifest for preprocessed FineVideo clips."""
		meta = []
		corrupt_paths = set()
		if os.path.exists(self.corrupt_files_path):
			with open(self.corrupt_files_path, "r") as f:
				corrupt_paths = {line.strip() for line in f if line.strip()}

		if not os.path.exists(self.manifest_path):
			raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

		with open(self.manifest_path, "r") as f:
			for line in f:
				item = json.loads(line)
				video_path = os.path.join(self.clips_dir, item["path"])
				
				if video_path in corrupt_paths:
					continue
					
				meta.append({
					"video_path": video_path,
					"duration": item["end_s"] - item["start_s"],
					"source_id": item["source_id"]
				})

		return meta