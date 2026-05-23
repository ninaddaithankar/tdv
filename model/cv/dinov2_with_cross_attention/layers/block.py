# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py

import logging
import os
from typing import Callable, List, Any, Tuple, Dict
import warnings

import torch
from torch import nn, Tensor

from .attention import AdaLNZero, CrossAttention, FiLMSpatialCondition, MemEffCrossAttention, SelfAttention, MemEffSelfAttention, SpatialCondition
from .drop_path import DropPath
from .layer_scale import LayerScale
from .mlp import Mlp


logger = logging.getLogger("dinov2")


XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import fmha, scaled_index_add, index_select_cat

        XFORMERS_AVAILABLE = True
        warnings.warn("xFormers is available (Block)")
    else:
        warnings.warn("xFormers is disabled (Block)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False

    warnings.warn("xFormers is not available (Block)")


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = SelfAttention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
    ) -> None:
        super().__init__()
        # print(f"biases: qkv: {qkv_bias}, proj: {proj_bias}, ffn: {ffn_bias}")
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

    def forward(self, x: Tensor) -> Tensor:
        def attn_residual_func(x: Tensor) -> Tensor:
            return self.ls1(self.attn(self.norm1(x)))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x))
            x = x + self.drop_path1(ffn_residual_func(x))  # FIXME: drop_path2
        else:
            x = x + attn_residual_func(x)
            x = x + ffn_residual_func(x)
        return x
    

class BlockWithCrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        condition_dim: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        self_attn_class: Callable[..., nn.Module] = SelfAttention,
        cross_attn_class: Callable[..., nn.Module] = CrossAttention,        
        ffn_layer: Callable[..., nn.Module] = Mlp,
        use_gating: bool = False,
        ignore_prefix_tokens_in_condition: bool = False,
    ) -> None:
        super().__init__()
        # print(f"biases: qkv: {qkv_bias}, proj: {proj_bias}, ffn: {ffn_bias}")
        self.norm1 = norm_layer(dim)
        self.self_attn = self_attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # -- norm, layer scaling, and drop path for cross attention
        self.norm_cross_attn = norm_layer(dim)
        self.cross_attn = cross_attn_class(
            dim,
            kv_input_dim=condition_dim,
            num_heads=num_heads,
            q_bias=qkv_bias,
            kv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls_cross_attn = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path_cross_attn = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path


    def forward(self, x: Tensor, condition: Tensor=None) -> Tensor:
        def self_attn_residual_func(x: Tensor) -> Tensor:
            return self.ls1(self.self_attn(self.norm1(x)))

        def cross_attn_residual_func(x: Tensor, condition: Tensor) -> Tensor:
            return self.ls_cross_attn(self.cross_attn(self.norm_cross_attn(x), condition))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1

            # -- performs self attention
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=self_attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )

            # -- performs cross attention
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=cross_attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                condition=condition,
            )

            # -- normal feed forward network
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(self_attn_residual_func(x))
            x = x + self.drop_path_cross_attn(cross_attn_residual_func(x, condition))
            x = x + self.drop_path2(ffn_residual_func(x))
        else:
            x = x + self_attn_residual_func(x)
            x = x + cross_attn_residual_func(x, condition)
            x = x + ffn_residual_func(x)
        return x
    



class BlockWithFilMSpatialConditioning(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        condition_dim: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        self_attn_class: Callable[..., nn.Module] = SelfAttention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        use_gating: bool = False,    
    ) -> None:
        super().__init__()
        
        print("Use spatial conditioning with gating set to :", use_gating)

        self.norm1 = norm_layer(dim)
        self.self_attn = self_attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # -- norm, layer scaling, and drop path for spatial conditioning
        self.norm_spatial_condn = norm_layer(dim)
        self.spatial_condn = FiLMSpatialCondition(
            dim,
            condition_dim=condition_dim,
            gated=use_gating,
        )
        self.ls_spatial_condn = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path_spatial_condn = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path


    def forward(self, x: Tensor, condition: Tensor=None) -> Tensor:
        def self_attn_residual_func(x: Tensor) -> Tensor:
            return self.ls1(self.self_attn(self.norm1(x)))

        def spatial_condn_residual_func(x: Tensor, condition: Tensor) -> Tensor:
            assert condition is not None, "BlockWithSpatialConditioning requires a non-None condition tensor"
            assert x.shape[1] == condition.shape[1], "BlockWithSpatialConditioning requires condition to have same num of tokens as x"

            x_norm = self.norm_spatial_condn(x)

            # -- separate the prefix token and the patch tokens
            x_prefix, x_patches = x_norm[:, :1, :], x_norm[:, 1:, :]
            condn_patches = condition[:, 1:, :]
            
            # -- apply spatial conditioning only on the patch tokens
            out_patches = self.spatial_condn(x_patches, condn_patches)
            x = torch.cat([torch.zeros_like(x_prefix), out_patches], dim=1)

            return self.ls_spatial_condn(x)

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1

            # -- performs self attention
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=self_attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )

            # -- performs spatial conditioning
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=spatial_condn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                condition=condition,
            )

            # -- normal feed forward network
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(self_attn_residual_func(x))
            x = x + self.drop_path_spatial_condn(spatial_condn_residual_func(x, condition))
            x = x + self.drop_path2(ffn_residual_func(x))
        else:
            x = x + self_attn_residual_func(x)
            x = x + spatial_condn_residual_func(x, condition)
            x = x + ffn_residual_func(x)
        return x



class BlockWithAdaLN(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        condition_dim: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        self_attn_class: Callable[..., nn.Module] = SelfAttention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        use_gating: bool = True,
        use_gate_before_attn: bool = True,
        num_prefix_tokens = 1,
        ignore_prefix_tokens_in_condition: bool = False,    
    ) -> None:
        super().__init__()

        self.num_prefix_tokens = num_prefix_tokens
        self.ignore_prefix = ignore_prefix_tokens_in_condition
        self.use_gate_before_attn = use_gate_before_attn

        # -- self attention branch
        self.norm1 = norm_layer(dim, elementwise_affine=False)
        self.self_attn = self_attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.mod_msa = AdaLNZero(dim=dim, condition_dim=condition_dim)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # -- MLP branch
        self.norm2 = norm_layer(dim, elementwise_affine=False)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.mod_mlp = AdaLNZero(dim=dim, condition_dim=condition_dim)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path
    
    def forward(self, x: Tensor, condition: Tensor = None) -> Tensor:
        assert condition is not None
        assert x.shape[0] == condition.shape[0]
        assert x.shape[1] == condition.shape[1]

        P = self.num_prefix_tokens

        # ---- Self-attention ----
        x_norm1 = self.norm1(x)

        if self.ignore_prefix:
            x_prefix1, x_patches1 = x_norm1[:, :P, :], x_norm1[:, P:, :]
            c_patches1 = condition[:, P:, :]

            shift_msa, scale_msa, gate_msa = self.mod_msa(c_patches1)      # [B, N, D]
            msa_in_patches = modulate(x_patches1, shift_msa, scale_msa)    # [B, N, D]
            msa_in = torch.cat([x_prefix1, msa_in_patches], dim=1)         # [B, P+N, D]

            gate_prefix = torch.ones_like(x_prefix1)
            gate_full = torch.cat([gate_prefix, gate_msa], dim=1)          # [B, P+N, D]

        else:
            shift_msa, scale_msa, gate_full = self.mod_msa(condition)      # [B, P+N, D]
            msa_in = modulate(x_norm1, shift_msa, scale_msa)               # [B, P+N, D]

        if self.use_gate_before_attn:
            msa_out = self.self_attn(gate_full * msa_in)                   # [B, P+N, D]
        else:
            msa_out = gate_full * self.self_attn(msa_in)                   # [B, P+N, D]

        x = x + self.ls1(msa_out)

        # ---- MLP ----
        x_norm2 = self.norm2(x)

        if self.ignore_prefix:
            x_prefix2, x_patches2 = x_norm2[:, :P, :], x_norm2[:, P:, :]
            c_patches2 = condition[:, P:, :]

            shift_mlp, scale_mlp, gate_mlp = self.mod_mlp(c_patches2)      # [B, N, D]
            mlp_in_patches = modulate(x_patches2, shift_mlp, scale_mlp)    # [B, N, D]
            mlp_in = torch.cat([x_prefix2, mlp_in_patches], dim=1)

            gate_prefix = torch.ones_like(x_prefix2)
            gate_full = torch.cat([gate_prefix, gate_mlp], dim=1)
        else:
            shift_mlp, scale_mlp, gate_full = self.mod_mlp(condition)
            mlp_in = modulate(x_norm2, shift_mlp, scale_mlp)

        if self.use_gate_before_attn:
            mlp_out = self.mlp(gate_full * mlp_in)
        else:
            mlp_out = gate_full * self.mlp(mlp_in)

        x = x + self.ls2(mlp_out)

        return x


def modulate(x_norm: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    # DiT convention: (1 + scale) * x + shift
    return x_norm * (scale) + shift


def drop_add_residual_stochastic_depth(
    x: Tensor,
    residual_func: Callable[[Tensor], Tensor],
    sample_drop_ratio: float = 0.0,
    condition: Tensor=None,
) -> Tensor:
    # 1) extract subset using permutation
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[brange]

    # 2) apply residual_func to get residual -- if condition is available, use that for cross attention
    residual = residual_func(x_subset, condition) if condition is not None else residual_func(x_subset) 

    x_flat = x.flatten(1)
    residual = residual.flatten(1)

    residual_scale_factor = b / sample_subset_size

    # 3) add the residual
    x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    return x_plus_residual.view_as(x)


def get_branges_scales(x, sample_drop_ratio=0.0):
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    residual_scale_factor = b / sample_subset_size
    return brange, residual_scale_factor


def add_residual(x, brange, residual, residual_scale_factor, scaling_vector=None):
    if scaling_vector is None:
        x_flat = x.flatten(1)
        residual = residual.flatten(1)
        x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    else:
        x_plus_residual = scaled_index_add(
            x, brange, residual.to(dtype=x.dtype), scaling=scaling_vector, alpha=residual_scale_factor
        )
    return x_plus_residual


attn_bias_cache: Dict[Tuple, Any] = {}


def get_attn_bias_and_cat(x_list, branges=None):
    """
    this will perform the index select, cat the tensors, and provide the attn_bias from cache
    """
    batch_sizes = [b.shape[0] for b in branges] if branges is not None else [x.shape[0] for x in x_list]
    all_shapes = tuple((b, x.shape[1]) for b, x in zip(batch_sizes, x_list))
    if all_shapes not in attn_bias_cache.keys():
        seqlens = []
        for b, x in zip(batch_sizes, x_list):
            for _ in range(b):
                seqlens.append(x.shape[1])
        attn_bias = fmha.BlockDiagonalMask.from_seqlens(seqlens)
        attn_bias._batch_sizes = batch_sizes
        attn_bias_cache[all_shapes] = attn_bias

    if branges is not None:
        cat_tensors = index_select_cat([x.flatten(1) for x in x_list], branges).view(1, -1, x_list[0].shape[-1])
    else:
        tensors_bs1 = tuple(x.reshape([1, -1, *x.shape[2:]]) for x in x_list)
        cat_tensors = torch.cat(tensors_bs1, dim=1)

    return attn_bias_cache[all_shapes], cat_tensors


def drop_add_residual_stochastic_depth_list(
    x_list: List[Tensor],
    residual_func: Callable[[Tensor, Any], Tensor],
    sample_drop_ratio: float = 0.0,
    scaling_vector=None,
) -> Tensor:
    # 1) generate random set of indices for dropping samples in the batch
    branges_scales = [get_branges_scales(x, sample_drop_ratio=sample_drop_ratio) for x in x_list]
    branges = [s[0] for s in branges_scales]
    residual_scale_factors = [s[1] for s in branges_scales]

    # 2) get attention bias and index+concat the tensors
    attn_bias, x_cat = get_attn_bias_and_cat(x_list, branges)

    # 3) apply residual_func to get residual, and split the result
    residual_list = attn_bias.split(residual_func(x_cat, attn_bias=attn_bias))  # type: ignore

    outputs = []
    for x, brange, residual, residual_scale_factor in zip(x_list, branges, residual_list, residual_scale_factors):
        outputs.append(add_residual(x, brange, residual, residual_scale_factor, scaling_vector).view_as(x))
    return outputs


class NestedTensorBlock(Block):
    def forward_nested(self, x_list: List[Tensor]) -> List[Tensor]:
        """
        x_list contains a list of tensors to nest together and run
        """
        assert isinstance(self.attn, MemEffSelfAttention)

        if self.training and self.sample_drop_ratio > 0.0:

            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.attn(self.norm1(x), attn_bias=attn_bias)

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.mlp(self.norm2(x))

            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=self.ls1.gamma if isinstance(self.ls1, LayerScale) else None,
            )
            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=self.ls2.gamma if isinstance(self.ls1, LayerScale) else None,
            )
            return x_list
        else:

            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.ls1(self.attn(self.norm1(x), attn_bias=attn_bias))

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.ls2(self.mlp(self.norm2(x)))

            attn_bias, x = get_attn_bias_and_cat(x_list)
            x = x + attn_residual_func(x, attn_bias=attn_bias)
            x = x + ffn_residual_func(x)
            return attn_bias.split(x)

    def forward(self, x_or_x_list):
        if isinstance(x_or_x_list, Tensor):
            return super().forward(x_or_x_list)
        elif isinstance(x_or_x_list, list):
            if not XFORMERS_AVAILABLE:
                raise AssertionError("xFormers is required for using nested tensors")
            return self.forward_nested(x_or_x_list)
        else:
            raise AssertionError


class NestedTensorBlockWithCrossAttention(BlockWithCrossAttention):
    def forward_nested(self, x_list: List[Tensor], condition_list: List[Tensor]) -> List[Tensor]:
        """
        x_list contains a list of tensors to nest together and run
        """
        assert isinstance(self.self_attn, MemEffSelfAttention)
        assert isinstance(self.cross_attn, MemEffCrossAttention)
        assert isinstance(condition_list, list) and len(condition_list) == len(x_list)

        def cat_condition_tensors(cond_tensors: List[Tensor], branges=None) -> Tensor:
            if branges is not None:
                return index_select_cat([c.flatten(1) for c in cond_tensors], branges).view(1, -1, cond_tensors[0].shape[-1])
            tensors_bs1 = tuple(c.reshape([1, -1, *c.shape[2:]]) for c in cond_tensors)
            return torch.cat(tensors_bs1, dim=1)

        if self.training and self.sample_drop_ratio > 0.0:

            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.self_attn(self.norm1(x), attn_bias=attn_bias)

            def cross_attn_residual_func(x: Tensor, attn_bias=None, condition_cat: Tensor = None) -> Tensor:
                assert condition_cat is not None
                return self.cross_attn(self.norm_cross_attn(x), condition_cat, attn_bias=attn_bias)

            def ffn_residual_func(x: Tensor) -> Tensor:
                return self.mlp(self.norm2(x))

            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=self.ls1.gamma if isinstance(self.ls1, LayerScale) else None,
            )

            # cross attention with stochastic depth
            branges_scales = [get_branges_scales(x, sample_drop_ratio=self.sample_drop_ratio) for x in x_list]
            branges = [s[0] for s in branges_scales]
            residual_scale_factors = [s[1] for s in branges_scales]
            attn_bias, x_cat = get_attn_bias_and_cat(x_list, branges)
            condition_cat = cat_condition_tensors(condition_list, branges)
            residual_list = attn_bias.split(
                cross_attn_residual_func(x_cat, attn_bias=attn_bias, condition_cat=condition_cat)  # type: ignore
            )

            cross_scaling_vector = self.ls_cross_attn.gamma if isinstance(self.ls_cross_attn, LayerScale) else None
            outputs = []
            for x, brange, residual, residual_scale_factor in zip(x_list, branges, residual_list, residual_scale_factors):
                outputs.append(add_residual(x, brange, residual, residual_scale_factor, cross_scaling_vector).view_as(x))
            x_list = outputs

            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=self.ls2.gamma if isinstance(self.ls1, LayerScale) else None,
            )
            return x_list
        else:

            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.ls1(self.self_attn(self.norm1(x), attn_bias=attn_bias))

            def cross_attn_residual_func(x: Tensor, attn_bias=None, condition_cat: Tensor = None) -> Tensor:
                assert condition_cat is not None
                return self.ls_cross_attn(self.cross_attn(self.norm_cross_attn(x), condition_cat, attn_bias=attn_bias))

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.ls2(self.mlp(self.norm2(x)))

            attn_bias, x = get_attn_bias_and_cat(x_list)
            condition_cat = cat_condition_tensors(condition_list)
            x = x + attn_residual_func(x, attn_bias=attn_bias)
            x = x + cross_attn_residual_func(x, attn_bias=attn_bias, condition_cat=condition_cat)
            x = x + ffn_residual_func(x)
            return attn_bias.split(x)

    def forward(self, x_or_x_list, condition):
        if isinstance(x_or_x_list, Tensor):
            return super().forward(x_or_x_list, condition)
        elif isinstance(x_or_x_list, list):
            if not XFORMERS_AVAILABLE:
                raise AssertionError("xFormers is required for using nested tensors")
            return self.forward_nested(x_or_x_list, condition)
        else:
            raise AssertionError
