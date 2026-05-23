import os
import warnings
import h5py

import numpy as np
import pandas as pd

from decord import VideoReader, cpu

import torch

class GenericVideoDatasetFromCSV(torch.utils.data.Dataset):
	def __init__(self, hparams, split, transform, shared_transform=None):
		if split == "train":
			self.data_paths = [hparams.dataset_train_csv_path]
		else:
			self.data_paths = [hparams.dataset_val_csv_path]

		self.split = split
		self.hparams = hparams
		self.datasets_weights = None
		self.frames_per_clip = 16
		self.frame_step = 4
		self.num_clips = hparams.num_segments
		self.transform = transform
		self.shared_transform = shared_transform
		self.random_clip_sampling = True
		self.allow_clip_overlap = True
		self.filter_short_videos = False
		self.filter_long_videos = 1000000000
		self.duration = None

		if VideoReader is None:
			raise ImportError('Unable to import "decord" which is required to read videos.')

		# Load video paths and labels
		samples, labels = [], []
		self.num_samples_per_dataset = []
		for data_path in self.data_paths:

			if data_path[-4:] == '.csv':
				data = pd.read_csv(data_path, header=None, delimiter=" ")
				samples += list(data.values[:, 0])
				labels += list(data.values[:, 1])
				num_samples = len(data)
				self.num_samples_per_dataset.append(num_samples)

			elif data_path[-4:] == '.npy':
				data = np.load(data_path, allow_pickle=True)
				data = list(map(lambda x: repr(x)[1:-1], data))
				samples += data
				labels += [0] * len(data)
				num_samples = len(data)
				self.num_samples_per_dataset.append(len(data))

		# [Optional] Weights for each sample to be used by downstream
		# weighted video sampler
		self.sample_weights = None
		if self.datasets_weights is not None:
			self.sample_weights = []
			for dw, ns in zip(self.datasets_weights, self.num_samples_per_dataset):
				self.sample_weights += [dw / ns] * ns
		
		self.samples = samples
		self.labels = labels
			
		self.h5f = None
		if self.hparams.use_preencoded_dataset:
			hdf5_path = self.hparams.preencoded_train_ds_path if self.split == "train" else self.hparams.preencoded_val_ds_path

			# Load preencoded features file
			self.h5f = h5py.File(hdf5_path, "r")


		self.samples = samples
		self.labels = labels


	def __getitem__(self, index):
		sample = self.samples[index]

		if self.hparams.use_preencoded_dataset and self.h5f is not None:
			# get video id
			video_id = os.path.splitext(sample)[0]

			# fetch preencoded features
			embeddings = self.h5f[video_id][:]
			embeddings = torch.tensor(embeddings, dtype=torch.float32)
			# print(f"Inside dataloader {embeddings.shape=}")
			return embeddings, self.labels[index]

		# Keep trying to load videos until you find a valid sample
		loaded_video = False
		while not loaded_video:
			buffer, clip_indices = self.loadvideo_decord(sample)  # [T H W 3]
			loaded_video = len(buffer) > 0
			if not loaded_video:
				index = np.random.randint(self.__len__())
				sample = self.samples[index]

		# Label/annotations for video
		label = self.labels[index]

		def split_into_clips(video):
			""" Split video into a list of clips """
			fpc = self.frames_per_clip
			nc = self.num_clips
			return [video[i*fpc:(i+1)*fpc] for i in range(nc)]

		# Parse video into frames & apply data augmentations
		if self.shared_transform is not None:
			buffer = self.shared_transform(buffer)
		buffer = split_into_clips(buffer)
		if self.transform is not None:
			buffer = [self.transform(clip) for clip in buffer]

		return buffer, label, clip_indices, sample

	def loadvideo_decord(self, sample):
		""" Load video content using Decord """

		fname = sample
		if not os.path.exists(fname):
			warnings.warn(f'video path not found {fname=}')
			return [], None

		_fsize = os.path.getsize(fname)
		if _fsize < 1 * 1024:  # avoid hanging issue
			warnings.warn(f'video too short {fname=}')
			return [], None
		if _fsize > self.filter_long_videos:
			warnings.warn(f'skipping long video of size {_fsize=} (bytes)')
			return [], None

		try:
			vr = VideoReader(fname, num_threads=-1, ctx=cpu(0))
		except Exception:
			return [], None

		fpc = self.frames_per_clip
		fstp = self.frame_step
		if self.duration is not None:
			try:
				fps = vr.get_avg_fps()
				fstp = int(self.duration * fps / fpc)
			except Exception as e:
				warnings.warn(e)
		clip_len = int(fpc * fstp)

		if self.filter_short_videos and len(vr) < clip_len:
			warnings.warn(f'skipping video of length {len(vr)}')
			return [], None

		vr.seek(0)  # Go to start of video before sampling frames

		# Partition video into equal sized segments and sample each clip
		# from a different segment
		partition_len = len(vr) // self.num_clips

		all_indices, clip_indices = [], []
		for i in range(self.num_clips):

			if partition_len > clip_len:
				# If partition_len > clip len, then sample a random window of
				# clip_len frames within the segment
				end_indx = clip_len
				if self.random_clip_sampling:
					end_indx = np.random.randint(clip_len, partition_len)
				start_indx = end_indx - clip_len
				indices = np.linspace(start_indx, end_indx, num=fpc)
				indices = np.clip(indices, start_indx, end_indx-1).astype(np.int64)
				# --
				indices = indices + i * partition_len
			else:
				# If partition overlap not allowed and partition_len < clip_len
				# then repeatedly append the last frame in the segment until
				# we reach the desired clip length
				if not self.allow_clip_overlap:
					indices = np.linspace(0, partition_len, num=partition_len // fstp)
					indices = np.concatenate((indices, np.ones(fpc - partition_len // fstp) * partition_len,))
					indices = np.clip(indices, 0, partition_len-1).astype(np.int64)
					# --
					indices = indices + i * partition_len

				# If partition overlap is allowed and partition_len < clip_len
				# then start_indx of segment i+1 will lie within segment i
				else:
					sample_len = min(clip_len, len(vr)) - 1
					indices = np.linspace(0, sample_len, num=sample_len // fstp)
					indices = np.concatenate((indices, np.ones(fpc - sample_len // fstp) * sample_len,))
					indices = np.clip(indices, 0, sample_len-1).astype(np.int64)
					# --
					clip_step = 0
					if len(vr) > clip_len:
						clip_step = (len(vr) - clip_len) // (self.num_clips - 1)
					indices = indices + i * clip_step

			clip_indices.append(indices)
			all_indices.extend(list(indices))

		buffer = vr.get_batch(all_indices).asnumpy()
		return buffer, clip_indices

	def __len__(self):
		return len(self.samples)

