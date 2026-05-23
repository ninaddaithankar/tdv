# Copied from croco-chris/src/models/latent_motion.py
# Only change: import from .midway_blocks instead of src.models.vision_transformer

import copy
from functools import partial

import torch
import torch.distributed as dist
import torch.nn as nn

from .midway_blocks import Block, trunc_normal_, Mlp, CrossBlock


class GatingProj(nn.Module):
    def __init__(self, dim, gating_dim, bias=None, tau=None, type='mlp',
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm_layer = norm_layer(dim)
        if type == 'mlp':
            self.mlp = Mlp(dim, dim, gating_dim)
        elif type == 'linear':
            self.mlp = nn.Linear(dim, gating_dim)
        self.bias = bias
        self.tau = tau
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.norm_layer(x)
        x = self.mlp(x)
        if self.tau is not None:
            x = x / self.tau
        if self.bias is not None:
            x = x + self.bias
        return self.sigmoid(x)


class IterativeLatentMotion(nn.Module):
    def __init__(self,
                 decoder_feature_mode,
                 embed_dim,
                 num_patches=None,
                 # latent motion parameters
                 use_pos_embed=True,
                 feature_levels=[11],
                 feature_block_type='cross',
                 feature_depth=1,
                 num_feature_heads=6,
                 use_cls_token=False,
                 ema_teacher=True,   # False = use student_embed for both imgs (no EMA, no detach); for flow fine-tuning
                 motion_dim=192,
                 motion_depth=2,
                 num_motion_heads=6,
                 motion_tokens=10,
                 motion_agg_type='identity',
                 motion_pred_input=True,
                 no_teacher=False,
                 gating_type=None,
                 gating_bias=None,
                 gating_tau=None,
                 pred_type='self',
                 predictor_depth=4,
                 num_pred_heads=6,
                 pred_tau=None,
                 pred_registers=0,
                 use_pred_pos_embed=True,
                 patch_size=16,
                 mask_mode=None,
                 # transformer parameters
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = False,
                 qk_scale: float = None,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 norm_layer=nn.LayerNorm,
                 num_cls_tokens=1,
                 **kwargs):
        super().__init__()
        self.decoder_feature_mode = decoder_feature_mode
        self.num_cls_tokens = num_cls_tokens
        self.use_cls_token = use_cls_token
        self.ema_teacher = ema_teacher

        self.feature_depth = feature_depth
        self.feature_levels = feature_levels
        self.feature_block_type = feature_block_type
        if feature_block_type == 'cross':
            for latent_level in range(len(feature_levels) - 1):
                student_feature_blocks = nn.ModuleList([
                    CrossBlock(
                        dim=embed_dim, num_heads=num_feature_heads,
                        mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                        qk_scale=qk_scale, drop=drop_rate,
                        attn_drop=attn_drop_rate, drop_path=0,
                        norm_layer=norm_layer)
                    for _ in range(feature_depth)])
                setattr(self, f"student_feature_blocks_{latent_level}",
                        student_feature_blocks)

        if decoder_feature_mode != 'feature':
            num_pos = None
            if num_patches is not None:
                num_pos = num_patches
                if use_cls_token:
                    num_pos += 1

            self.use_pos_embed = use_pos_embed
            if use_pos_embed:
                if num_pos is None:
                    num_pos = 1
                if no_teacher:
                    self.pos_embed = nn.Parameter(
                        torch.zeros(1, num_pos, motion_dim))
                else:
                    self.pos_embed = nn.Parameter(
                        torch.zeros(1, num_pos * 2, motion_dim))

            self.motion_depth = motion_depth
            self.motion_dim = motion_dim
            self.student_embed = nn.Linear(embed_dim, motion_dim)
            dpr = [x.item() for x in torch.linspace(
                0, drop_path_rate, motion_depth)]

            self.no_teacher = no_teacher
            self.motion_pred_input = motion_pred_input
            for latent_level in range(len(feature_levels) - 1):
                motion_blocks = nn.ModuleList([
                    Block(
                        dim=motion_dim, num_heads=num_motion_heads,
                        mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                        qk_scale=qk_scale, drop=drop_rate,
                        attn_drop=attn_drop_rate, drop_path=dpr[j],
                        norm_layer=norm_layer)
                    for j in range(motion_depth)])
                setattr(self, f"motion_blocks_{latent_level}", motion_blocks)

            self.motion_tokens = nn.Parameter(
                torch.zeros(1, motion_tokens, motion_dim))
            self.motion_agg_type = motion_agg_type
            self.motion_proj = nn.Linear(motion_dim, embed_dim)

            self.gating_type = None
            if gating_type is not None:
                gating_type, gating_out = gating_type.split('-')
                if gating_out == 'vector':
                    gating_dim = embed_dim
                elif gating_out == 'scalar':
                    gating_dim = 1
                else:
                    raise NotImplementedError(
                        f"Gating type {gating_type} not implemented.")

                self.gating_type = gating_type
                for latent_level in range(len(feature_levels) - 1):
                    if gating_type == 'all_first_blk':
                        initial_gating_blk = Block(
                            dim=motion_dim, num_heads=num_motion_heads,
                            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                            qk_scale=qk_scale, drop=drop_rate,
                            attn_drop=attn_drop_rate, drop_path=0,
                            norm_layer=norm_layer)
                        setattr(self, f"initial_gating_blk_{latent_level}",
                                initial_gating_blk)
                    if gating_type in ['initial', 'all', 'all_first_blk']:
                        initial_gating_mlp = GatingProj(
                            motion_dim, gating_dim, gating_bias, gating_tau)
                        setattr(self, f"initial_gating_mlp_{latent_level}",
                                initial_gating_mlp)
                    if gating_type in ['pred', 'all', 'all_first_blk']:
                        gating_mlps = nn.ModuleList([
                            GatingProj(embed_dim, gating_dim, gating_bias,
                                       gating_tau)
                            for _ in range(predictor_depth - 1)])
                        setattr(self, f"gating_mlps_{latent_level}",
                                gating_mlps)
                    if gating_type == 'gating_blk':
                        gating_blks = nn.ModuleList([
                            Block(
                                dim=embed_dim, num_heads=num_pred_heads,
                                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                qk_scale=qk_scale, drop=drop_rate,
                                attn_drop=attn_drop_rate, drop_path=0,
                                norm_layer=norm_layer)
                            for _ in range(predictor_depth)])
                        setattr(self, f"gating_blks_{latent_level}",
                                gating_blks)
                        gating_mlps = nn.ModuleList([
                            GatingProj(embed_dim, gating_dim, gating_bias,
                                       gating_tau)
                            for _ in range(predictor_depth)])
                        setattr(self, f"gating_mlps_{latent_level}",
                                gating_mlps)

            dpr = [x.item() for x in torch.linspace(
                0, drop_path_rate, predictor_depth)]
            self.predictor_depth = predictor_depth

            self.pred_type = pred_type
            if pred_type != 'self':
                self.pred_tokens = nn.Parameter(
                    torch.zeros(1, 1, embed_dim))
            self.pred_pos_embed = None
            if use_pred_pos_embed:
                self.pred_pos_embed = nn.Parameter(
                    torch.zeros(1, num_pos, embed_dim))

            self.pred_registers = None
            if pred_registers:
                self.pred_registers = nn.Parameter(
                    torch.zeros(1, pred_registers, embed_dim))

            self.mask_mode = mask_mode
            if (pred_type == 'self' and mask_mode is not None) or \
                    mask_mode == 'mask-loss':
                self.mask_token = nn.Parameter(
                    torch.zeros(1, 1, embed_dim))
            if pred_type == 'self':
                pred_block = Block
            elif pred_type == 'cross':
                pred_block = partial(CrossBlock, tau=pred_tau)
            else:
                raise NotImplementedError(
                    f"Predictor type {pred_type} not implemented.")
            for latent_level in range(len(feature_levels) - 1):
                forward_predictor = nn.ModuleList([
                    pred_block(
                        dim=embed_dim, num_heads=num_pred_heads,
                        mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                        qk_scale=qk_scale, drop=drop_rate,
                        attn_drop=attn_drop_rate, drop_path=dpr[j],
                        norm_layer=norm_layer)
                    for j in range(self.predictor_depth)])
                setattr(self, f"forward_predictor_{latent_level}",
                        forward_predictor)

        self.apply(self._init_weights)

        if self.decoder_feature_mode != 'feature':
            self.teacher_embed = copy.deepcopy(self.student_embed)

        if self.feature_block_type == 'cross':
            for latent_level in range(len(feature_levels) - 1):
                setattr(self, f"teacher_feature_blocks_{latent_level}",
                        copy.deepcopy(
                            getattr(self,
                                    f"student_feature_blocks_{latent_level}")))

        self.patch_size = patch_size

        for p in self.teacher_parameters:
            p.requires_grad = False

    @property
    def student_parameters(self):
        if self.decoder_feature_mode != 'feature':
            for module in [self.student_embed]:
                for p in module.parameters():
                    yield p
        if self.feature_block_type == 'cross':
            for latent_level in range(len(self.feature_levels) - 1):
                for p in getattr(
                        self,
                        f"student_feature_blocks_{latent_level}").parameters():
                    yield p

    @property
    def teacher_parameters(self):
        if self.decoder_feature_mode != 'feature':
            for module in [self.teacher_embed]:
                for p in module.parameters():
                    yield p
        if self.feature_block_type == 'cross':
            for latent_level in range(len(self.feature_levels) - 1):
                for p in getattr(
                        self,
                        f"teacher_feature_blocks_{latent_level}").parameters():
                    yield p

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_backward_features(self, student_features):
        x1 = student_features[-1]
        out = []
        for i in range(len(self.feature_levels) - 2, -1, -1):
            lower_x1 = student_features[i]
            if self.feature_block_type == 'cross':
                student_feature_blocks = getattr(
                    self, f"student_feature_blocks_{i}")
                for b, blk in enumerate(student_feature_blocks):
                    lower_x1, x1 = blk(lower_x1, x1)
                    out.append(x1)
            x1 = lower_x1
        return out

    def forward(self, student_output, teacher_output):
        out = []

        student_features = []
        for i, feat in enumerate(student_output):
            if i in self.feature_levels:
                if not self.use_cls_token:
                    feat = feat[:, self.num_cls_tokens:]
                student_features.append(feat)
        teacher_features = []
        for i, feat in enumerate(teacher_output):
            if i in self.feature_levels:
                if not self.use_cls_token:
                    feat = feat[:, self.num_cls_tokens:]
                teacher_features.append(feat)

        if self.decoder_feature_mode == 'feature':
            return self.forward_backward_features(student_features)

        x1 = student_features[-1]
        _x1 = x1
        x2 = teacher_features[-1]
        B, L, D = x2.shape
        m = self.motion_tokens.expand(B, -1, -1)

        for i in range(len(self.feature_levels) - 2, -1, -1):
            if self.motion_pred_input:
                _x1 = self.student_embed(_x1)
            else:
                _x1 = self.student_embed(x1)
            if not self.no_teacher:
                if self.ema_teacher:
                    _x2 = self.teacher_embed(x2).detach()
                else:
                    # No EMA update in this training loop: use student_embed for both
                    # images so img2 features are projected through trainable weights
                    # and gradients flow back through img2.
                    _x2 = self.student_embed(x2)

            if self.use_pos_embed:
                if self.pos_embed.shape[1] == 2:
                    _x1 = _x1 + self.pos_embed[:, 0:1].expand(
                        B, _x1.shape[1], -1)
                    if not self.no_teacher:
                        _x2 = _x2 + self.pos_embed[:, 1:2].expand(
                            B, _x2.shape[1], -1)
                else:
                    pos_embed = self.pos_embed[:, :L].expand(B, -1, -1)
                    _x1 = _x1 + pos_embed
                    if not self.no_teacher:
                        pos_embed = self.pos_embed[:, L:].expand(B, -1, -1)
                        _x2 = _x2 + pos_embed

            old_m = m.clone()
            if self.no_teacher:
                motion_input = torch.cat([m, _x1], dim=1)
            else:
                motion_input = torch.cat([m, _x1, _x2], dim=1)
            motion_blocks = getattr(self, f"motion_blocks_{i}")
            for b, blk in enumerate(motion_blocks):
                motion_input = blk(motion_input)
                M = m.shape[1]
                if 'motion2' in self.decoder_feature_mode:
                    out.append(motion_input[:, M + L:M + 2 * L])
                else:
                    out.append(motion_input[:, M:M + L])
            m = motion_input[:, :m.shape[1]]

            if i != len(self.feature_levels) - 2:
                if self.motion_agg_type == 'identity':
                    pass
                elif self.motion_agg_type == 'add':
                    m = m + old_m
                elif self.motion_agg_type == 'concat':
                    m = torch.cat([old_m, m], dim=1)

            lower_x1 = student_features[i]
            lower_x2 = teacher_features[i]
            if self.feature_block_type == 'cross':
                student_feature_blocks = getattr(
                    self, f"student_feature_blocks_{i}")
                teacher_feature_blocks = getattr(
                    self, f"teacher_feature_blocks_{i}")

                for b, blk in enumerate(student_feature_blocks):
                    lower_x1, x1 = blk(lower_x1, x1)

                with torch.no_grad():
                    for blk in teacher_feature_blocks:
                        lower_x2, x2 = blk(lower_x2, x2)

            forward_predictor = getattr(self, f"forward_predictor_{i}")
            if self.pred_type == 'self':
                pred_tokens = lower_x1
                if self.pred_pos_embed is not None:
                    pred_tokens = pred_tokens + self.pred_pos_embed.expand(
                        B, -1, -1)
                _x1 = torch.cat([self.motion_proj(m), pred_tokens], dim=1)
                M = m.shape[1]
                if self.pred_registers is not None:
                    pred_registers = self.pred_registers.expand(B, -1, -1)
                    _x1 = torch.cat([pred_registers, _x1], dim=1)
                    M = M + self.pred_registers.shape[1]
                gating = None
                if self.gating_type in ['initial', 'all', 'all_first_blk']:
                    if self.gating_type == 'all_first_blk':
                        gating_input = getattr(
                            self, f"initial_gating_blk_{i}")(
                            motion_input[:, M:M + L])
                    else:
                        gating_input = motion_input[:, M:M + L]
                    gating_mlp = getattr(self, f"initial_gating_mlp_{i}")
                    gating = gating_mlp(gating_input)
                    D_g = gating.shape[2]
                    ones = torch.ones((B, M, D_g), device=_x1.device)
                    gating = torch.cat([ones, gating], dim=1)
                gating_mlps = []
                if self.gating_type in ['pred', 'all', 'all_first_blk']:
                    gating_mlps = getattr(self, f"gating_mlps_{i}")
                if self.gating_type == 'gating_blk':
                    gating_x1 = _x1
                    for gating_blk, gating_mlp, blk in zip(
                            getattr(self, f"gating_blks_{i}"),
                            getattr(self, f"gating_mlps_{i}"),
                            forward_predictor):
                        gating_x1 = gating_blk(gating_x1)
                        next_gating = gating_mlp(gating_x1[:, M:M + L])
                        D_g = next_gating.shape[2]
                        ones = torch.ones((B, M, D_g), device=_x1.device)
                        next_gating = torch.cat([ones, next_gating], dim=1)
                        _x1 = blk(_x1, gating=next_gating)
                        out.append(_x1[:, M:M + L])
                else:
                    for b, blk in enumerate(forward_predictor):
                        next_gating = None
                        if b < len(gating_mlps):
                            next_gating = gating_mlps[b](_x1[:, M:M + L])
                            D_g = next_gating.shape[2]
                            ones = torch.ones((B, M, D_g), device=_x1.device)
                            next_gating = torch.cat([ones, next_gating],
                                                    dim=1)
                        _x1 = blk(_x1, gating=gating)
                        out.append(_x1[:, M:M + L])
                        gating = next_gating
                _x1 = _x1[:, M:]
            elif self.pred_type == 'cross':
                cross_tokens = lower_x1
                pred_tokens = self.pred_tokens.expand(B, L, -1)
                if self.pred_pos_embed is not None:
                    pred_tokens = pred_tokens + self.pred_pos_embed.expand(
                        B, -1, -1)
                _x1 = torch.cat([self.motion_proj(m), pred_tokens], dim=1)
                M = m.shape[1]
                if self.pred_registers is not None:
                    pred_registers = self.pred_registers.expand(B, -1, -1)
                    _x1 = torch.cat([pred_registers, _x1], dim=1)
                    M = M + self.pred_registers.shape[1]
                for blk in forward_predictor:
                    _x1, cross_tokens = blk(_x1, cross_tokens)
                    out.append(_x1[:, M:M + L])
                _x1 = _x1[:, M:]
            x1 = lower_x1
            x2 = lower_x2
        return out
