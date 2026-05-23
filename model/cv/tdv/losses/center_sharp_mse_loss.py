import torch

class CenterSharpReconstructionLoss(torch.nn.Module):
	def __init__(
		self,
		out_dim: int,
		predicted_temp: float = 1.0,
		target_temp: float = 1.0,
		center_momentum: float = 0.9,
		loss = "mse",
	):
		super().__init__()

		# params
		self.predicted_temp = predicted_temp
		self.target_temp = target_temp
		self.center_momentum = center_momentum
		
		# loss function
		self.loss_fn = torch.nn.MSELoss() if loss == "mse" else torch.nn.SmoothL1Loss()

		# running mean of teacher logits; register as buffer so it is saved in ckpt
		self.register_buffer("center", torch.zeros(1, 1, out_dim))

	@torch.no_grad()
	def _update_center(self, target_logits, sync_center=True):
		dims = tuple(range(target_logits.ndim - 1))
		batch_center = target_logits.mean(dim=dims, keepdim=False)
		batch_center = batch_center.reshape(1, 1, -1)
		assert batch_center.shape[-1] == self.center.shape[-1], f"Shape mismatch: {batch_center.shape} vs {self.center.shape}"

		# -- make it global if we are in DDP/FSDP
		if sync_center and torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
			torch.distributed.all_reduce(batch_center, op=torch.distributed.ReduceOp.SUM)
			batch_center /= torch.distributed.get_world_size()

		# -- in-place EMA keeps the buffer intact
		self.center.mul_(self.center_momentum).add_(batch_center * (1 - self.center_momentum))


	def forward(self, predicted: torch.Tensor, target: torch.Tensor, use_centering=True, use_sharpening=True, sync_center=True):
		raw_target_logits = target

		# -- center targets
		if use_centering:
			target = target - self.center
		
		# -- apply temperature sharpening
		if use_sharpening:
			target = target / self.target_temp
			predicted = predicted / self.predicted_temp

		# -- MSE between predicted and target
		loss = self.loss_fn(predicted, target)

		# -- update running center  (only after we used it)
		if use_centering:
			with torch.no_grad():
				self._update_center(raw_target_logits, sync_center)

		return loss
