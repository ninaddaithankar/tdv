# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os
import warnings

from torch import Tensor
from torch import nn
import torch

import torch.nn.functional as F


logger = logging.getLogger("dinov2")


XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind

        XFORMERS_AVAILABLE = True
        warnings.warn("xFormers is available (Attention)")
    else:
        warnings.warn("xFormers is disabled (Attention)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    warnings.warn("xFormers is not available (Attention)")


def rotate_queries_or_keys(x, pos, base=500):
    B, num_heads, N, D = x.size()
    assert D % 2 == 0, "Embedding dimension must be a multiple of 2 for block matrix rotation"

    # -- compute angle for each position
    omega = torch.arange(0, D, 2, dtype=torch.float32, device=x.device)
    omega = 1.0 / (base ** (omega / D))
    
    # freq: (..., N, D/2)
    freq = torch.einsum("..., f -> ... f", pos.float(), omega)

    # -- build rotation matrix and apply
    # We interleave so emb_cos/sin look like [c0, c0, c1, c1, ...]
    emb_sin = freq.sin().repeat_interleave(2, dim=-1)  
    emb_cos = freq.cos().repeat_interleave(2, dim=-1)

    # -- Optimized rotation logic [x1, x2] -> [-x2, x1]
    # This avoids the overhead of stack/unflatten and is very stable for FP16
    x_rotated = torch.empty_like(x)
    x_rotated[..., 0::2] = -x[..., 1::2]
    x_rotated[..., 1::2] = x[..., 0::2]
    
    return (x * emb_cos) + (x_rotated * emb_sin)


def rotate_queries_or_keys_jepa(x, pos):
    B, num_heads, N, D = x.size()
    assert D % 2 == 0, "Embedding dimension must be a multiple of 2 for block matrix rotation"

    # -- compute angle for each position
    omega = torch.arange(D // 2, dtype=x.dtype, device=x.device)
    omega /= D / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)
    freq = torch.einsum("..., f -> ... f", pos, omega)  # (..., N, D/2), outer product

    # -- build rotation matrix and apply
    emb_sin = freq.sin()  # (..., N, D/2)
    emb_cos = freq.cos()  # (..., N, D/2)
    # -- NOTE: This expansion has a subtle bug where frequencies are duplicated across the vector pair.
    # -- Fixing the bug would break compatibility with the pretrained model, but the fix can be applied by commenting
    # -- out the two lines below, and uncommenting the following two lines.
    # -- Thanks to @echosprint, original PR: https://github.com/facebookresearch/vjepa2/pull/15
    # emb_sin = emb_sin.squeeze(-1).repeat(1, 1, 1, 2)
    # emb_cos = emb_cos.squeeze(-1).repeat(1, 1, 1, 2)
    emb_sin = emb_sin.repeat_interleave(2, dim=-1)  # (..., N, D)
    emb_cos = emb_cos.repeat_interleave(2, dim=-1)  # (..., N, D)

    # --
    y = x.unflatten(-1, (-1, 2))
    y1, y2 = y.unbind(
        dim=-1,
    )
    y = torch.stack((-y2, y1), dim=-1)
    y = y.flatten(-2)
    return (x * emb_cos) + (y * emb_sin)



class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    

class RoPESelfAttention2D(nn.Module):
    """
        Implementation from VJEPA2: https://github.com/facebookresearch/vjepa2/blob/main/src/models/utils/modules.py
        Modified to just work with 2D spatial data instead of 3D.
    """
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        proj_bias=False,
        use_sdpa=True,
        grid_size=16,   # default H=W if not provided at forward
        is_causal=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop_prob = attn_drop
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa
        self.grid_size = grid_size
        self.is_causal = is_causal

        # --- 2D RoPE split: some dims for H, some for W ---
        # This mirrors your earlier style: we ensure even dims for each axis.
        # Ideally head_dim is divisible by 4 so we can use all dims cleanly.
        half = head_dim // 2
        self.h_dim = int(2 * (half // 2))  # even, <= half
        self.w_dim = int(2 * (half // 2))  # even, <= half

    # ---- 2D index helpers (no time / depth) ----

    def _get_height_pos(self, ids, H_patches=None, W_patches=None):
        """
        Given flattened patch indices (0..H*W-1), return height (row) indices.
        ids: (...,) int tensor
        """
        if H_patches is None or W_patches is None:
            tokens_per_row = self.grid_size
        else:
            tokens_per_row = W_patches
        return ids // tokens_per_row

    def separate_positions_2d(self, ids, H_patches=None, W_patches=None):
        """
        Map flattened ids (0..H*W-1) to (height_ids, width_ids).
        """
        if H_patches is None or W_patches is None:
            tokens_per_row = self.grid_size
        else:
            tokens_per_row = W_patches

        height_ids = self._get_height_pos(ids, H_patches, W_patches)
        width_ids = ids - tokens_per_row * height_ids
        return height_ids, width_ids

    def forward(
        self,
        x,
        attn_mask=None,
        H_patches=None,
        W_patches=None,
        p: int = 1,        # number of CLS/prefix tokens at the start to ignore for RoPE
    ):
        """
        x: (B, N, C) where
           N = p + H*W  (prefix tokens + patches)

        p: number of prefix tokens (CLS, registers, etc.) at the beginning.
           These will *not* be rotated.

        We:
          - keep tokens [0..p-1] unrotated
          - apply 2D RoPE over tokens [p..N-1]
        """
        B, N, C = x.size()
        assert N > p, "Need at least p prefix tokens and some patch tokens"

        # qkv: (B, N, 3*dim) -> (3, B, num_heads, N, head_dim)
        qkv = self.qkv(x).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, head_dim]

        q = q * self.scale

        # --- separate prefix and patch tokens ---
        # prefix tokens: indices [0..p-1]
        q_prefix, q_patches = q[:, :, :p, :], q[:, :, p:, :]  # [B, H, p, D], [B, H, N_p, D]
        k_prefix, k_patches = k[:, :, :p, :], k[:, :, p:, :]
        # v is NOT rotated; we keep full v (prefix + patches)
        N_patches = N - p

        # --- build linear positions for patch tokens only (0..N_patches-1) ---
        if H_patches is None or W_patches is None:
            # assume square grid if not provided
            H = W = self.grid_size
            assert N_patches == H * W, "N - p must be grid_size^2 if H_patches/W_patches not given"
        else:
            H, W = H_patches, W_patches
            assert N_patches == H * W, "N - p must equal H_patches * W_patches for 2D RoPE"

        patch_ids = torch.arange(N_patches, device=x.device)  # 0..H*W-1
        h_ids, w_ids = self.separate_positions_2d(patch_ids, H_patches, W_patches)

        # --- apply 2D RoPE: height chunk, then width chunk, on patches only ---
        s = 0

        # Rotate height dimensions
        qh = rotate_queries_or_keys(q_patches[..., s : s + self.h_dim], pos=h_ids)
        kh = rotate_queries_or_keys(k_patches[..., s : s + self.h_dim], pos=h_ids)
        s += self.h_dim

        # Rotate width dimensions
        qw = rotate_queries_or_keys(q_patches[..., s : s + self.w_dim], pos=w_ids)
        kw = rotate_queries_or_keys(k_patches[..., s : s + self.w_dim], pos=w_ids)
        s += self.w_dim

        # Any remaining dims (if head_dim not fully used) just pass them through
        if s < self.head_dim:
            qr = q_patches[..., s:]
            kr = k_patches[..., s:]
            q_patches = torch.cat([qh, qw, qr], dim=-1)
            k_patches = torch.cat([kh, kw, kr], dim=-1)
        else:
            q_patches = torch.cat([qh, qw], dim=-1)
            k_patches = torch.cat([kh, kw], dim=-1)

        # --- re-attach prefix tokens (unrotated) in front ---
        q = torch.cat([q_prefix, q_patches], dim=2)  # [B, H, N, D]
        k = torch.cat([k_prefix, k_patches], dim=2)

        if attn_mask is not None or self.use_sdpa:
            with torch.backends.cuda.sdp_kernel():
                x = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self.attn_drop_prob,
                    is_causal=self.is_causal,
                    attn_mask=attn_mask,
                    scale=1.0
                )
                attn = None
        else:
            attn = (q @ k.transpose(-2, -1))  # [B, num_heads, N, N]
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class MemEffSelfAttention(SelfAttention):
    def forward(self, x: Tensor, context: Tensor=None, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        # -- (cross attention) if context is available use its keys and values
        if context is not None:
            q = qkv[:, :, 0]
            kv = self.kv(context).reshape(B, N, 2, self.num_heads, C // self.num_heads)
            k, v = unbind(kv, 2)
        
        # -- (self attention) else use self keys and values
        else:
            q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    


# -- CROSS ATTENTION --

class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        kv_input_dim: int,
        num_heads: int = 8,
        q_bias: bool = False,
        kv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=q_bias)
        self.kv = nn.Linear(kv_input_dim, dim * 2, bias=kv_bias) 

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        B, N, C = x.shape
        B_c, N_c, C_c = condition.shape

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = q * self.scale
        
        kv = self.kv(condition).reshape(B_c, N_c, 2, self.num_heads, C_c // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    

class MemEffCrossAttention(CrossAttention):
    def forward(self, x: Tensor, condition: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape

        # print(f"{x.shape=}")
        # print(B, N, self.num_heads, C // self.num_heads)

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads)

        # print(f"Inside MemEFFXAttn {condition.shape=}")
        # print(B, N, 2, self.num_heads, C // self.num_heads)
        
        kv = self.kv(condition).reshape(B, N, 2, self.num_heads, C // self.num_heads)
        k, v = unbind(kv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    

class RoPECrossAttention2D(nn.Module):
    """
    2D RoPE cross-attention:
      - Queries from x_q
      - Keys/Values from x_kv
      - First p tokens are prefix tokens (CLS / registers) -> not rotated
      - Remaining tokens form an H x W grid -> 2D RoPE applied to Q and K
    """
    def __init__(
        self,
        dim,
        kv_input_dim,
        num_heads: int = 8,
        q_bias: bool = False,
        kv_bias: bool = False,
        proj_bias: bool = False,
        qk_scale=None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_sdpa: bool = True,
        grid_size: int = 16,   # default H=W if not provided at forward
        is_causal: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # Separate projections for cross-attn
        self.q_proj = nn.Linear(dim, dim, bias=q_bias)
        self.kv_proj = nn.Linear(kv_input_dim, dim * 2, bias=kv_bias)

        self.attn_drop_prob = attn_drop
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa
        self.grid_size = grid_size
        self.is_causal = is_causal

        # --- 2D RoPE split: some dims for H, some for W ---
        # Ideally head_dim % 4 == 0 so we can use all dims cleanly.
        half = head_dim // 2
        self.h_dim = int(2 * (half // 2))  # even, <= half
        self.w_dim = int(2 * (half // 2))  # even, <= half

    # ---- 2D index helpers (no time / depth) ----

    def _get_height_pos(self, ids, H_patches=None, W_patches=None):
        """
        Given flattened patch indices (0..H*W-1), return height (row) indices.
        ids: (...,) int tensor
        """
        if H_patches is None or W_patches is None:
            tokens_per_row = self.grid_size
        else:
            tokens_per_row = W_patches
        return ids // tokens_per_row

    def separate_positions_2d(self, ids, H_patches=None, W_patches=None):
        """
        Map flattened ids (0..H*W-1) to (height_ids, width_ids).
        """
        if H_patches is None or W_patches is None:
            tokens_per_row = self.grid_size
        else:
            tokens_per_row = W_patches

        height_ids = self._get_height_pos(ids, H_patches, W_patches)
        width_ids = ids - tokens_per_row * height_ids
        return height_ids, width_ids

    def forward(
        self,
        x,                 # (B, N_q, C)
        condition,         # (B, N_kv, C)
        attn_mask=None,
        H_patches=None,
        W_patches=None,
        p: int = 1,          # number of prefix tokens at the start to ignore for RoPE
    ):
        """
        x_q: (B, N_q, C)
        x_kv: (B, N_kv, C)

        We assume:
          N_q == N_kv == N
          N = p + H*W

        p: number of prefix tokens (CLS, registers, etc.) at the beginning.
           These will *not* be rotated.

        We:
          - keep tokens [0..p-1] in Q/K unrotated
          - apply 2D RoPE over tokens [p..N-1] (patch tokens)
          - never rotate V
        """
        B, N_q, C = x.size()
        B2, N_kv, C2 = condition.size()
        assert B == B2 and C == C2, "x_q and x_kv must have same batch and dim"
        assert N_q == N_kv, "For this implementation we assume N_q == N_kv"
        N = N_q
        assert N > p, "Need at least p prefix tokens and some patch tokens"

        # --- project Q, K, V ---
        # q: (B, N, dim)
        q = self.q_proj(x)
        # kv: (B, N, 2*dim)
        kv = self.kv_proj(condition)

        # reshape to multi-head
        # q: (B, N, num_heads, head_dim) -> (B, num_heads, N, head_dim)
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # kv: (B, N, 2, num_heads, head_dim) -> (2, B, num_heads, N, head_dim)
        kv = kv.view(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # [B, num_heads, N, head_dim]

        q = q * self.scale

        # --- separate prefix and patch tokens in Q and K ---
        # prefix: [0..p-1]
        q_prefix, q_patches = q[:, :, :p, :], q[:, :, p:, :]  # [B, H, p, D], [B, H, N_p, D]
        k_prefix, k_patches = k[:, :, :p, :], k[:, :, p:, :]
        N_patches = N - p

        # --- build linear positions for patch tokens only (0..N_patches-1) ---
        if H_patches is None or W_patches is None:
            # assume square grid if not provided
            H = W = self.grid_size
            assert N_patches == H * W, "N - p must be grid_size^2 if H_patches/W_patches not given"
        else:
            H, W = H_patches, W_patches
            assert N_patches == H * W, "N - p must equal H_patches * W_patches for 2D RoPE"

        patch_ids = torch.arange(N_patches, device=x.device)  # 0..H*W-1
        h_ids, w_ids = self.separate_positions_2d(patch_ids, H_patches, W_patches)

        # --- apply 2D RoPE: height chunk, then width chunk, on patch tokens only ---
        s = 0

        # Rotate height dims
        qh = rotate_queries_or_keys(q_patches[..., s : s + self.h_dim], pos=h_ids)
        kh = rotate_queries_or_keys(k_patches[..., s : s + self.h_dim], pos=h_ids)
        s += self.h_dim

        # Rotate width dims
        qw = rotate_queries_or_keys(q_patches[..., s : s + self.w_dim], pos=w_ids)
        kw = rotate_queries_or_keys(k_patches[..., s : s + self.w_dim], pos=w_ids)
        s += self.w_dim

        # Any remaining dims (if head_dim not fully used) just pass them through
        if s < self.head_dim:
            qr = q_patches[..., s:]
            kr = k_patches[..., s:]
            q_patches = torch.cat([qh, qw, qr], dim=-1)
            k_patches = torch.cat([kh, kw, kr], dim=-1)
        else:
            q_patches = torch.cat([qh, qw], dim=-1)
            k_patches = torch.cat([kh, kw], dim=-1)

        # --- re-attach unrotated prefix tokens ---
        q = torch.cat([q_prefix, q_patches], dim=2)  # [B, num_heads, N, head_dim]
        k = torch.cat([k_prefix, k_patches], dim=2)

        # --- attention ---
        if attn_mask is not None or self.use_sdpa:
            # use SDPA kernel if available
            with torch.backends.cuda.sdp_kernel():
                x = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self.attn_drop_prob,
                    is_causal=self.is_causal,
                    attn_mask=attn_mask,
                    scale=1.0
                )
                attn = None
        else:
            attn = (q @ k.transpose(-2, -1))  # [B, num_heads, N, N]
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v  # [B, num_heads, N, head_dim]

        # merge heads
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    

class SpatialCondition(nn.Module):
    def __init__(
            self,
            dim,
            condition_dim,
            gated=False,
    ):
        super().__init__()
        self.W = nn.Linear(condition_dim, dim)
        self.gated = gated

        if gated:
            self.gate = nn.Linear(condition_dim, dim)

    def forward(self, x:Tensor, condition: Tensor) -> Tensor:
        cond = self.W(condition)

        if self.gated:
            g = torch.sigmoid(self.gate(condition))
            cond = cond * g

        return x + cond
    

class FiLMSpatialCondition(nn.Module):
    def __init__(
            self,
            dim,
            condition_dim,
            gated=False,
    ):
        super().__init__()
        self.W_gamma = nn.Linear(condition_dim, dim)
        self.W_beta  = nn.Linear(condition_dim, dim)
        self.gated = gated
        if gated:
            self.gate = nn.Linear(dim, dim)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        gamma = self.W_gamma(condition)
        beta  = self.W_beta(condition)

        if self.gated:
            g = torch.sigmoid(self.gate(condition))
            gamma = gamma * g

        return (gamma*x) + beta
           

class AdaLNZero(nn.Module):
    """
    adaLN-Zero style per-token modulation from condition:
      cond_patches [B, N, cond_dim] -> shift, scale, gate each [B, N, dim]
    """
    def __init__(self, dim: int, condition_dim: int, bound_scale: bool = False, bound_gate: bool = False):
        super().__init__()
        self.proj = nn.Linear(condition_dim, 3 * dim, bias=True)
        self.bound_scale = bound_scale
        self.bound_gate = bound_gate

        # adaLN-Zero init: outputs start at 0 => no conditioning at init
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, cond_patches: Tensor):
        shift, scale, gate = self.proj(cond_patches).chunk(3, dim=-1)

        if self.bound_scale: 
            scale = torch.tanh(scale)

        scale = 1.0 + scale  # DiT convention

        if self.bound_gate: 
            gate = 1.0 + torch.tanh(gate)

        return shift, scale, gate