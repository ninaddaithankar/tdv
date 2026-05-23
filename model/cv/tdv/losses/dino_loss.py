import torch
import torch.nn.functional as F

class DinoLoss(torch.nn.Module):
    def __init__(
        self,
        out_dim: int,
        student_temp: float = 0.1,
        teacher_temp: float = 0.04,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.eps = 1e-5

        # running mean of teacher logits; register as buffer so it is saved in ckpt
        self.register_buffer("center", torch.zeros(1, 1, out_dim))

    # @torch.no_grad()
    # def _update_center(self, teacher_logits: torch.Tensor):
    #     dims = tuple(range(0, teacher_logits.ndim - 1))  # all dims except feature dim
    #     batch_center = teacher_logits.mean(dim=dims, keepdim=True)
        
    #     # EMA update
    #     self.center = (
    #         self.center * self.center_momentum
    #         + batch_center * (1.0 - self.center_momentum)
    #     )

    @torch.no_grad()
    def _update_center(self, teacher_logits, sync_center=True):
        dims = tuple(range(teacher_logits.ndim - 1))
        batch_center = teacher_logits.mean(dim=dims, keepdim=False)
        batch_center = batch_center.view_as(self.center)

        # make it global if we are in DDP/FSDP
        if sync_center and torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            torch.distributed.all_reduce(batch_center, op=torch.distributed.ReduceOp.SUM)
            batch_center /= torch.distributed.get_world_size()

        # in-place EMA keeps the buffer intact
        self.center.mul_(self.center_momentum).add_(batch_center * (1 - self.center_momentum))


    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, use_centering=True, use_sharpening=True, sync_center=True, token_weights=None):
        raw_teacher_logits = teacher_logits

        # -- CENTER TEACHER
        if use_centering:
            teacher_logits = teacher_logits - self.center

        # -- SHARPEN TEACHER, SOFTEN STUDENT
        if use_sharpening:
            teacher_logits = teacher_logits / self.teacher_temp
            student_logits = student_logits / self.student_temp

        teacher_probs = F.softmax(teacher_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

        student_log_probs = F.log_softmax(student_logits, dim=-1)

        # -- per-token cross-entropy
        per_token_loss = -torch.sum(teacher_probs * student_log_probs, dim=-1)  # [N_tokens]
        
        weight_sum = None
        if token_weights is not None:
            weight_sum = token_weights.sum().clamp(min=1e-6)
            loss = (per_token_loss * token_weights).sum() / weight_sum
        else:
            loss = per_token_loss.mean()

        with torch.no_grad():
            per_token_entropy = -(teacher_probs * teacher_log_probs).sum(dim=-1)
            per_token_kl = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
            if token_weights is not None and weight_sum is not None:
                entropy = (per_token_entropy * token_weights).sum() / weight_sum
                kl_div = (per_token_kl * token_weights).sum() / weight_sum
            else:
                entropy = per_token_entropy.mean()
                kl_div = per_token_kl.mean()

            # -- Update running center  (only after we used it)
            if use_centering:
                self._update_center(raw_teacher_logits, sync_center)

        return loss, entropy.detach(), kl_div.detach()
