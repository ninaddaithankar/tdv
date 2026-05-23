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

import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn




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



class Attention(nn.Module):
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

    def forward(self, x: Tensor, return_attention=False) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        if return_attention:
            return x, attn
        
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
        proj_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_sdpa=True,
        grid_size=16,   # default H=W if not provided at forward
        is_causal=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
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
        return_attention=False,
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
                    dropout_p=self.proj_drop_prob,
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

        if return_attention:
            return x, attn
        
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, return_attention=False) -> Tensor:
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        attn = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        attn = attn.reshape([B, N, C])

        x = self.proj(attn)
        x = self.proj_drop(x)

        if return_attention:
            return x, attn

        return x
