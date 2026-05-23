import os, json, random, av
from torch.utils.data import Dataset
from torchvision import transforms

from data.utils import video_utils
from data.utils.data_preprocessor import handle_corrupt_file

class Ego4DTasksDataset(Dataset):
	"""
	Load video clips from Ego4D '{task}_{split}*.json' annotations.

	Each __getitem__ returns:
		- frames  : (T, 3, H, W) tensor 
		- meta    : dict  {video_uid, clip_uid, class, ...}
	"""
	def __init__(
		self,
		hparams,                     
		split: str,                                             # "train", "val", "test"
		dataset_dir: str = None,                                # path to Ego4D dataset
		transform: transforms.Compose = None,
		task = "moments"                                        # "moments", "nlq", "va", "ego4d"
	):
		super().__init__()
		self.hparams = hparams
		self.dataset_dir = dataset_dir if dataset_dir is not None else hparams.dataset_dir
		self.use_processed_data = self.hparams.use_preprocessed_ego4d

		if self.use_processed_data:
			annotations_dir = f"{self.dataset_dir}/meta/"
			videos_dir = self.dataset_dir
			json_file_name = f"{task}_{split}_preprocessed.json"
		else:
			annotations_dir = f"{self.dataset_dir}/v2/annotations"
			videos_dir = f"{self.dataset_dir}/v2/full_scale"
			json_file_name = f"{task}_{split}.json"

		self.filter_corrupt_videos = True
		self.corrupt_files_path = "data/cv/corrupt_files/ego4d.txt"

		self.transform = transform
		self.num_frames = hparams.context_length
		self.sampling_rate = hparams.sampling_rate
		self.time_between_frames = hparams.time_between_frames
		
		# -- load the json with metadata
		json_path = os.path.join(annotations_dir, json_file_name)
		with open(json_path, "r") as f: annotations = json.load(f)

		# read metadata for clips
		self.meta = self._read_metadata(annotations, videos_dir)


	def __len__(self):
		return len(self.meta)
	

	def __getitem__(self, idx):

		info = self.meta[idx]
		video_path = info["video_path"]

		try:
			container = av.open(video_path)
			stream = container.streams.video[0]
			fps = float(stream.average_rate)

			if self.use_processed_data:
				start_sec, end_sec = 0, info["duration"]
			else:
				start_sec, end_sec = info["start_sec"], info["end_sec"]
			
			frame_idxs = video_utils.sample_frame_indices(start_sec, end_sec, fps, time_between_frames=self.time_between_frames, num_frames=self.num_frames)
			clip = video_utils.read_selected_frames(video_path, container, stream, frame_idxs, self.transform)
			
			return clip, "dummy"

		except Exception as e:
			handle_corrupt_file(e, video_path, log_path=self.corrupt_files_path)

			# recursion: try next sample
			return self[random.randint(0, len(self)-1)]
		

	def _read_metadata(self, annotations, videos_dir):
		meta = []

		corrupt_paths = {}
		if self.filter_corrupt_videos:
			with open(self.corrupt_files_path, "r") as f:
				corrupt_paths = {line.strip() for line in f if line.strip()}

		if self.use_processed_data:
			for segment in annotations:
				# if "filename" in segment:
				# 	# use full path to video stored under filename key
				# 	video_path = segment["filename"]
				# else:
				video_path = os.path.join(videos_dir, f"{segment['filename']}")

				if video_path in corrupt_paths: continue
				
				if os.path.exists(video_path):
					meta.append({
						"video_path": video_path,
						"video_uid": segment["video_uid"],
						"clip_uid": segment["clip_uid"],
						"segment_index": segment["segment_index"],
						"duration": segment["end_sec"] - segment["start_sec"]
					})
				else: raise FileNotFoundError(video_path)
		else:
			for video in annotations["videos"]:
				video_path = os.path.join(videos_dir, f"{video['video_uid']}.mp4")

				if video_path in corrupt_paths: continue

				if os.path.exists(video_path):
					for clip in video["clips"]:
						meta.append({
							"video_path" : video_path,
							"video_uid" : video["video_uid"],
							"clip_uid"  : clip["clip_uid"],
							"start_sec" : clip["clip_start_sec"],
							"end_sec"   : clip["clip_end_sec"],
						})
				else: raise FileNotFoundError(video_path)

		return meta