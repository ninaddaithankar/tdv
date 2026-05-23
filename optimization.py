import torch
from torch import optim
from torch.optim.lr_scheduler import _LRScheduler

class LARS(optim.Optimizer):
    def __init__(
        self,
        params,
        lr,
        weight_decay=0,
        momentum=0.9,
        eta=0.001,
        weight_decay_filter=None,
        lars_adaptation_filter=None,
        epoch=0, 
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter,
        )
        self.epoch = epoch
        super().__init__(params, defaults)

    def update_epoch(self, epoch):
        self.epoch = epoch

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                closure()

        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad

                if dp is None:
                    continue

                if g["weight_decay_filter"] is None or not g["weight_decay_filter"](p):
                    dp = dp.add(p, alpha=g["weight_decay"])

                if (g["lars_adaptation_filter"] is None or not g[
                    "lars_adaptation_filter"
                ](p)):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0, (g["eta"] * param_norm / update_norm), one
                        ),
                        one,
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)

                p.add_(mu, alpha=-g["lr"])

def exclude_bias_and_norm(p):
    return p.ndim == 1

class WarmUpCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, warm_up_steps, warm_up_base_lr_divider, cosine_scheduler, warm_up_finished_func=None):
        self.warm_up_steps = warm_up_steps
        self.cosine_scheduler = cosine_scheduler
        self.last_step = 0
        self.highest_lr = [group['lr'] for group in optimizer.param_groups]
        self.finished_warming_up = False
        self.warm_up_finished_func = warm_up_finished_func

        self.base_lr_divider = warm_up_base_lr_divider
        if self.base_lr_divider == -1:
            self._last_lr = [0.0 for _ in self.highest_lr]
        else:
            self._last_lr = [lr / self.base_lr_divider for lr in self.highest_lr]

        super(WarmUpCosineAnnealingLR, self).__init__(optimizer)
        
    def step(self):
        self.last_step += 1
        super().step()

    def get_lr(self):
        if self.warm_up_steps != 0 and self.last_step <= self.warm_up_steps:
            if self.base_lr_divider == -1: # Warm-up from 0 to highest_lr
                warmup_lr = [
                    lr * (self.last_step / self.warm_up_steps)
                    for lr in self.highest_lr
                ]
            else: # Warm-up from lr / base_lr_divider to highest_lr
                warmup_lr = [
                    (lr / self.base_lr_divider) +
                    (lr - (lr / self.base_lr_divider)) * (self.last_step / self.warm_up_steps)
                    for lr in self.highest_lr
                ]
            self._last_lr = warmup_lr
            return warmup_lr
        else:
            if not self.finished_warming_up:
                self.finished_warming_up = True
                if self.warm_up_finished_func is not None:
                    self.warm_up_finished_func()
            # Proceed with the cosine scheduler after warm-up
            self.cosine_scheduler.step()
            self._last_lr = self.cosine_scheduler.get_last_lr()
            return self._last_lr
            
    def get_last_lr(self):
        return self._last_lr

    def state_dict(self):
        return {
            'warm_up_steps': self.warm_up_steps,
            'last_lr': self._last_lr,
            'last_step': self.last_step,
            'finished_warming_up': self.finished_warming_up,
            'cosine_scheduler_state_dict': self.cosine_scheduler.state_dict()
        }

    def load_state_dict(self, state_dict):
        self.warm_up_steps = state_dict['warm_up_steps']
        self._last_lr = state_dict['last_lr']
        self.last_step = state_dict['last_step']
        self.finished_warming_up = state_dict['finished_warming_up']
        self.cosine_scheduler.load_state_dict(state_dict['cosine_scheduler_state_dict'])
 