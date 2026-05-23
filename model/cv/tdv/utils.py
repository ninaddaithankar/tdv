import copy
import math
import torch
import torch.distributed as dist
import random
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from model.cv.dinov2.layers.dino_head import DINOHead
from model.model_utils import create_image_encoder, load_tdv_encoder_from_checkpoint


def set_trainable(module, trainable: bool):
	module.train(trainable)
	for p in module.parameters():
		p.requires_grad = trainable


def update_teacher_using_ema(students, teachers, ema_momentum):
	assert len(students) == len(teachers), "there should be a teacher for every student!"
	with torch.no_grad():
		for student, teacher in zip(students, teachers):
			student_params = dict(student.named_parameters())
			teacher_params = dict(teacher.named_parameters())

			assert student_params.keys() == teacher_params.keys(), "Student/teacher param mismatch"

			for name in teacher_params.keys():
				# teacher = m*teacher + (1-m)*student
				teacher_params[name].data.mul_(ema_momentum)
				teacher_params[name].data.add_((1.0 - ema_momentum) * student_params[name].data)


def init_dino_head(in_dim, out_dim, layers):
	dino_head = DINOHead(
		in_dim=in_dim, 
		out_dim=out_dim, 
		hidden_dim=2048, 
		bottleneck_dim=256, 
		nlayers=layers
	)
	return dino_head


def get_teacher_encoder(hparams, student_encoder):
	teacher_frame_encoder, teacher_dino_head, teacher_ibot_head = None, None, None

	# -- load from checkpoint
	if hparams.load_teacher_from_checkpoint:
		print(f"Loading teacher from checkpoint: {hparams.load_teacher_from_checkpoint}")
		teacher_frame_encoder = load_tdv_encoder_from_checkpoint(
					hparams.load_teacher_from_checkpoint, 
					hparams.backbone_type, 
					hparams.vit_backbone_size
				)

	# -- or load fresh with different seed
	elif hparams.use_different_init_teacher:
		prev_cpu_rng_state = torch.get_rng_state()
		prev_cuda_rng_state = torch.cuda.get_rng_state_all()

		torch.manual_seed(42)
		torch.cuda.manual_seed_all(42)		

		teacher_frame_encoder = create_image_encoder(
			hparams.backbone_type, 
			hparams.vit_backbone_size, 
			pretrained = not hparams.load_without_weights
		)

		torch.set_rng_state(prev_cpu_rng_state)
		torch.cuda.set_rng_state_all(prev_cuda_rng_state)

	elif hparams.use_fixed_dino_teacher:
		print("Using fixed DINOv2 pre-trained weights for teacher encoder.")
		
		teacher_frame_encoder = create_image_encoder(
			hparams.backbone_type, 
			hparams.vit_backbone_size, 
			pretrained = True
		)
	
	# -- or create a deepcopy
	else:
		teacher_frame_encoder = copy.deepcopy(student_encoder)

	if hparams.use_dino_head:
		teacher_dino_head = init_dino_head(
			in_dim=hparams.vit_backbone_dim, 
			out_dim=hparams.dino_head_prototype_dim,
			layers=3
		)

		if hparams.use_seperate_ibot_head:
			teacher_ibot_head = init_dino_head(
				in_dim=hparams.vit_backbone_dim, 
				out_dim=hparams.dino_head_prototype_dim,
				layers=3)
			set_trainable(teacher_ibot_head, trainable=False)
			

	# -- freeze weights
	if teacher_frame_encoder is not None:
		set_trainable(teacher_frame_encoder, trainable=False)
	if teacher_dino_head is not None:
		set_trainable(teacher_dino_head, trainable=False)

	return teacher_frame_encoder, teacher_dino_head, teacher_ibot_head


class DINOClipAugmentation:
	"""
	DINO global crop augmentation adapted for video clips [T, C, H, W].

	Mirrors the DINO repo's global_transfo1 / global_transfo2 exactly:
	  global_transfo1 (teacher): gaussian_blur_p=1.0, solarization_p=0.0
	  global_transfo2 (student): gaussian_blur_p=0.1, solarization_p=0.2
	Both use global_crops_scale=(0.4, 1.0).

	Spatial params (crop, flip) are sampled ONCE per clip so all T frames
	share the same spatial transform — required for rgb_diff to remain valid.
	Color transforms are applied uniformly across all frames in the clip.
	"""
	_IMAGENET_MEAN = (0.485, 0.456, 0.406)
	_IMAGENET_STD  = (0.229, 0.224, 0.225)

	def __init__(self, crop_scale, image_size, gaussian_blur_p, solarization_p=0.0):
		self.crop_scale = crop_scale
		self.image_size = image_size
		self.gaussian_blur_p = gaussian_blur_p
		self.solarization_p = solarization_p
		self.color_jitter = T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
		self.gaussian_blur = T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))
		self.normalize     = T.Normalize(mean=self._IMAGENET_MEAN, std=self._IMAGENET_STD)

	def __call__(self, clip):
		"""clip: [T, C, H, W] float tensor in [0, 1]"""
		# -- spatial: sample params once from first frame, apply to all T frames
		# i, j, h, w = T.RandomResizedCrop.get_params(clip[0], self.crop_scale, (3/4, 4/3))
		# clip = TF.resized_crop(clip, i, j, h, w, [self.image_size, self.image_size],
		# 					   interpolation=TF.InterpolationMode.BICUBIC)
		clip = TF.resize(clip, [self.image_size, self.image_size],
						 interpolation=TF.InterpolationMode.BICUBIC)
		# if torch.rand(1).item() < 0.5:
		# 	clip = TF.hflip(clip)

		# -- flip_and_color_jitter (same params applied uniformly across all T frames)
		if torch.rand(1).item() < 0.8:
			clip = self.color_jitter(clip)
		if torch.rand(1).item() < 0.2:
			clip = TF.rgb_to_grayscale(clip, num_output_channels=3)

		# -- gaussian blur (p=1.0 for global_transfo1, p=0.1 for global_transfo2)
		if torch.rand(1).item() < self.gaussian_blur_p:
			clip = self.gaussian_blur(clip)

		# -- solarization (p=0.0 for global_transfo1, p=0.2 for global_transfo2)
		if torch.rand(1).item() < self.solarization_p:
			clip = TF.solarize(clip, threshold=0.5)

		# -- normalize (ImageNet stats, matches DINO recipe)
		# clip = self.normalize(clip)

		return clip


def apply_clip_augmentation(frames, aug):
	"""Apply a DINOClipAugmentation independently to each clip in a batch.
	frames: [B, T, C, H, W]  →  [B, T, C, H, W]
	"""
	return torch.stack([aug(frames[b]) for b in range(frames.shape[0])])


def get_rgb_diff(frame_sequences, use_full_frame=False):
	if use_full_frame:
		# CAUTION: only for checking FULL FRAME PRED
		rgb_diff = frame_sequences[:, 1:]
	else:
		# -- (default) calculate RGB diff
		rgb_diff = frame_sequences[:, 1:] - frame_sequences[:, :-1]

	return rgb_diff 																			


def calculate_static_frames_mask(rgb_diff, threshold=0):
	mask = None

	if threshold > 0:
		pixel_diff = torch.abs(rgb_diff)                      			 	# [B, T-1, 3, H, W]
		mean_rgb_diff = pixel_diff.mean(dim=[1,2,3,4])        				# [B]
		mask = mean_rgb_diff > threshold            						# [B] boolean
	
	return mask


def skip_this_batch(mask):
	if mask is None: return False

	# -- if all frames in the batch are static, skip this training step
	local_skip = torch.tensor(
		1 if (mask.sum() == 0) else 0,          						# 1 → want to skip
		device=mask.device, dtype=torch.uint8
	)

	# sync this across gpus so everybody knows we need to skip this current step! 
	if dist.is_initialized() and dist.get_world_size() > 1:
		dist.all_reduce(local_skip, op=dist.ReduceOp.MAX)
	
	global_skip = bool(local_skip.item())
	if global_skip:
		print(f"All sequence of frames are static on device {mask.device}. No loss calculated for current step! #############################")
	
	return global_skip


def rollout_n_frames(previous, rgb_diff, target, n=1):
	'''
		if current timestep is t, then
		n=1 is normal next frame prediction (predict t+1),
		n=2 means skip a frame and predict the next (predict t+2)
		and so on..
	'''
	if n <= 1:
		return previous, rgb_diff, target

	assert previous.shape == rgb_diff.shape == target.shape, \
		"Shape mismatch while evaluating rollout"

	B, T = previous.shape[:2]

	csum = rgb_diff.cumsum(dim=1)
	pad = torch.zeros_like(csum[:, :1])
	csum = torch.cat([pad, csum], dim=1)

	n_rgb_diff = csum[:, n:] - csum[:, :-n]    # n-step diffs
	previous_n = previous[:, : T-n+1]          # F_t
	target_n   = target[:, n-1:]               # F_{t+n}

	return previous_n, n_rgb_diff, target_n


def collate_data_and_cast(samples_list, mask_ratio_tuple, mask_probability, n_tokens=None, mask_generator=None):
	samples = torch.stack([s[0] if isinstance(s, (tuple, list)) else s for s in samples_list])
	B, t, c, h, w = samples.shape

	N = n_tokens
	n_frames = B * t
	n_frames_masked = int(n_frames * mask_probability)
	probs = torch.linspace(*mask_ratio_tuple, n_frames_masked + 1)
	masks_list = []

	for i in range(n_frames_masked):
		prob_min = probs[i]
		prob_max = probs[i + 1]
		masks_list.append(torch.BoolTensor(mask_generator(int(N * random.uniform(prob_min, prob_max)))))

	for i in range(n_frames_masked, n_frames):
		masks_list.append(torch.BoolTensor(mask_generator(0)))

	random.shuffle(masks_list)

	collated_masks = torch.stack(masks_list).flatten(1)                       # [B*T, N_patches]
	masks_weight = (1.0 / collated_masks.sum(-1).clamp(min=1.0)).reshape(B, t)  # [B, T]

	return {
		"frames": samples,
		"masks": collated_masks.reshape(B, t, -1),  # [B, T, N_patches]
		"masks_weight": masks_weight,               # [B, T], weight = 1/n_masked per frame
	}


class MaskingGenerator:
	def __init__(
		self,
		input_size,
		num_masking_patches=None,
		min_num_patches=4,
		max_num_patches=None,
		min_aspect=0.3,
		max_aspect=None,
	):
		if not isinstance(input_size, tuple):
			input_size = (input_size,) * 2
		self.height, self.width = input_size

		self.num_patches = self.height * self.width
		self.num_masking_patches = num_masking_patches

		self.min_num_patches = min_num_patches
		self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches

		max_aspect = max_aspect or 1 / min_aspect
		self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

	def __repr__(self):
		repr_str = "Generator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
			self.height,
			self.width,
			self.min_num_patches,
			self.max_num_patches,
			self.num_masking_patches,
			self.log_aspect_ratio[0],
			self.log_aspect_ratio[1],
		)
		return repr_str

	def get_shape(self):
		return self.height, self.width

	def _mask(self, mask, max_mask_patches):
		delta = 0
		for _ in range(10):
			target_area = random.uniform(self.min_num_patches, max_mask_patches)
			aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
			h = int(round(math.sqrt(target_area * aspect_ratio)))
			w = int(round(math.sqrt(target_area / aspect_ratio)))
			if w < self.width and h < self.height:
				top = random.randint(0, self.height - h)
				left = random.randint(0, self.width - w)

				num_masked = mask[top : top + h, left : left + w].sum()
				# Overlap
				if 0 < h * w - num_masked <= max_mask_patches:
					for i in range(top, top + h):
						for j in range(left, left + w):
							if mask[i, j] == 0:
								mask[i, j] = 1
								delta += 1

				if delta > 0:
					break
		return delta

	def __call__(self, num_masking_patches=0):
		mask = np.zeros(shape=self.get_shape(), dtype=bool)
		mask_count = 0
		while mask_count < num_masking_patches:
			max_mask_patches = num_masking_patches - mask_count
			max_mask_patches = min(max_mask_patches, self.max_num_patches)

			delta = self._mask(mask, max_mask_patches)
			if delta == 0:
				break
			else:
				mask_count += delta

		return mask
