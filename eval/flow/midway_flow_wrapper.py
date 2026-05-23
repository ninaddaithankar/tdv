"""
midway_flow_wrapper.py
======================
Midway-style optical flow model that uses DINOv2 as the shared encoder for
both img1 and img2, fuses the two feature streams with IterativeLatentMotion,
and feeds the combined feature list to a DPT head.

This is additive – it does NOT touch tdv_flow_wrapper.py or any existing code.

Pipeline
--------
  img1, img2
      |
  DINOv2 frame_encoder  (called for each image separately)
      |                   → all_layers: list of [B, N_patches, enc_embed_dim]
  IterativeLatentMotion  (student=img1, teacher=img2)
      |                   → dec_out:   list of [B, N_patches, motion_dim]
  combined: enc_all_layers + dec_out   (length = enc_depth + n_dec_out)
      |
  MidwayPixelwiseTaskWithDPT  (DPT with mixed-dim hook handling)
      |
  flow  [B, 2, H, W]  (or [B, 3, H, W] with confidence)

Note on all_layers format
-------------------------
encode_images() with return_all_layers=True returns:
    (last_token [B, 1+N, D], all_layers list of [B, 1+num_reg+N, D])
DINOv2 forward_features appends the raw block output x to all_layers, which
includes [cls, (register tokens), patch tokens].  We must strip num_cls_tokens
= 1 + num_register_tokens from the front before passing to DPT.
The default num_cls_tokens=1 covers standard DINOv2 (no register tokens).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from croco.models.midway_head import MidwayPixelwiseTaskWithDPT
from croco.models.latent_motion import IterativeLatentMotion
from model.model_utils import encode_images


# ---------------------------------------------------------------------------
# Small namespace so MidwayPixelwiseTaskWithDPT.setup() can read model attrs
# ---------------------------------------------------------------------------

class _EncoderStub:
    """Exposes the attrs that MidwayPixelwiseTaskWithDPT.setup() reads."""
    def __init__(self, n_blocks: int, embed_dim: int):
        self.blocks = [None] * n_blocks   # only len() is used
        self.embed_dim = embed_dim


class _DecoderStub:
    """Exposes the attrs that MidwayPixelwiseTaskWithDPT.setup() reads."""
    def __init__(self, feature_levels, motion_depth: int, motion_dim: int,
                 predictor_depth: int = 0):
        self.feature_levels = feature_levels
        self.motion_depth = motion_depth
        self.motion_dim = motion_dim
        self.predictor_depth = predictor_depth


class _ModelStub:
    """
    Passed to MidwayPixelwiseTaskWithDPT.setup() so it can auto-compute
    hooks_idx and dim_tokens without needing the real model object.
    """
    def __init__(self, encoder_stub, decoder_stub, decoder_feature_mode='motion'):
        self.encoder = encoder_stub
        self.decoder = decoder_stub
        self.decoder_feature_mode = decoder_feature_mode


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

class MidwayDINOv2Backbone(nn.Module):
    """
    Encodes img1 and img2 through the DINOv2 frame_encoder (separately),
    then fuses the two feature streams with IterativeLatentMotion.

    Returns a flat list of feature tensors (enc layers + dec motion layers)
    suitable for DPT hook selection.  All tensors have shape [B, N_patches, D]
    with cls/register tokens already stripped.

    cls-token note
    --------------
    DINOv2's forward_features appends the raw block output x to all_layers at
    each block.  That x has shape [B, 1 + num_register_tokens + N_patches, D].
    The DPT rearrange step expects exactly N_H*N_W patch tokens, so we must
    strip the non-patch prefix tokens before combining with dec_out.
    num_cls_tokens = 1 + num_register_tokens (default 1 for DINOv2 w/o regs).
    IterativeLatentMotion is built with use_cls_token=False, num_cls_tokens=N
    so it strips them internally; we then strip again from enc layers here.
    """

    def __init__(
        self,
        frame_encoder: nn.Module,
        decoder: IterativeLatentMotion,
        *,
        encoder_name: str,          # e.g. 'dinov2'
        enc_embed_dim: int,         # DINOv2 feature dim, e.g. 384
        enc_depth: int,             # number of DINOv2 transformer blocks
        num_cls_tokens: int = 1,    # tokens to strip from front of each all_layers element
                                    # = 1 + num_register_tokens (typically 1 for DINOv2)
    ):
        super().__init__()
        self.frame_encoder = frame_encoder
        self.decoder = decoder
        self.encoder_name = encoder_name
        self.enc_embed_dim = enc_embed_dim
        self.enc_depth = enc_depth
        self.num_cls_tokens = num_cls_tokens

    def forward_all_layers(
        self, img1: torch.Tensor, img2: torch.Tensor
    ):
        """
        Returns
        -------
        all_features : list of Tensor [B, N_patches, D]
            enc_all_layers_no_cls (length enc_depth)  +  dec_out (motion tensors)
            All tensors are patch tokens only (no cls/register).
        """
        # --- encode both images through DINOv2, get per-block features ------
        # all_layers[i]: [B, 1 + num_register_tokens + N_patches, D]  (raw block output, includes cls)
        _, all_layers1 = encode_images(
            img1,
            encoder_name=self.encoder_name,
            encoder=self.frame_encoder,
            condition=None,
            return_all_layers=True,
        )
        _, all_layers2 = encode_images(
            img2,
            encoder_name=self.encoder_name,
            encoder=self.frame_encoder,
            condition=None,
            return_all_layers=True,
        )

        # --- IterativeLatentMotion: student=img1, teacher=img2 --------------
        # The decoder is built with use_cls_token=False, num_cls_tokens=self.num_cls_tokens
        # so it strips cls/register tokens internally before processing.
        # dec_out elements: [B, N_patches, motion_dim]  (patch tokens only)
        dec_out = self.decoder(all_layers1, all_layers2)

        # --- strip cls/register from encoder layers -------------------------
        # DPT rearrange requires exactly N_H*N_W = N_patches tokens per layer.
        n = self.num_cls_tokens
        all_layers1_patches = [f[:, n:] for f in all_layers1]  # [B, N_patches, D] each

        # --- combine: enc patch layers + decoder motion outputs -------------
        all_features = all_layers1_patches + dec_out
        return all_features


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class MidwayDINOv2FlowWithDPT(nn.Module):
    """
    End-to-end flow model using the Midway approach:
      (img1, img2) → MidwayDINOv2Backbone → MidwayPixelwiseTaskWithDPT → flow
    """

    def __init__(
        self,
        backbone: MidwayDINOv2Backbone,
        head: MidwayPixelwiseTaskWithDPT,
    ):
        super().__init__()
        self.backbone = backbone
        self.head = head

        # Build the stub so setup() can auto-compute hooks without the real
        # MidwayNetwork internals.
        stub = _ModelStub(
            encoder_stub=_EncoderStub(
                n_blocks=backbone.enc_depth,
                embed_dim=backbone.enc_embed_dim,
            ),
            decoder_stub=_DecoderStub(
                feature_levels=backbone.decoder.feature_levels,
                motion_depth=backbone.decoder.motion_depth,
                motion_dim=backbone.decoder.motion_dim,
                predictor_depth=backbone.decoder.predictor_depth,
            ),
            decoder_feature_mode=backbone.decoder.decoder_feature_mode,
        )
        head.setup(stub)

        # refinenet4 has no skip connection at the deepest level; delete the
        # unused resConfUnit1 to avoid DDP unused-parameter errors.
        del self.head.dpt.scratch.refinenet4.resConfUnit1

    @torch.no_grad()
    def infer(self, img1, img2):
        return self.forward(img1, img2)

    def forward(self, img1, img2):
        all_features = self.backbone.forward_all_layers(img1, img2)
        img_info = {"height": img1.shape[-2], "width": img1.shape[-1]}
        out = self.head(all_features, img_info)
        # DINOv2 patch_size=14 causes DPT to output (H/14)*16 instead of H.
        # Resize back to input resolution so the criterion sees matching sizes.
        if out.shape[-2:] != img1.shape[-2:]:
            out = F.interpolate(out, size=img1.shape[-2:], mode='bilinear', align_corners=False)
        return out


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

_DINO_ARCH_TO_SIZE_DIM = {
    "vit_tiny":  ("tiny",  192),
    "vit_small": ("small", 384),
    "vit_base":  ("base",  768),
    "vit_large": ("large", 1024),
}


def _is_dino_recipe_ckpt(ckpt: dict) -> bool:
    """True if ckpt looks like a dino-recipe checkpoint (has 'teacher'+'student' keys)."""
    return "teacher" in ckpt and "student" in ckpt


def _is_dinov2_pretrain_ckpt(ckpt: dict) -> bool:
    """True if ckpt is an official DINOv2 pretrain file (flat state dict, keys start with cls_token / blocks.*)."""
    return "cls_token" in ckpt and "blocks.0.norm1.weight" in ckpt


def _encoder_state_from_dinov2_pretrain_ckpt(ckpt: dict) -> dict:
    """
    Extract weights from an official DINOv2 pretrain file (flat state dict).
    All keys → backbone.frame_encoder.*
    """
    return {"backbone.frame_encoder." + k: v for k, v in ckpt.items()}


def _hparams_from_dinov2_pretrain_ckpt(ckpt: dict) -> dict:
    """Infer hparams from tensor shapes in a flat DINOv2 pretrain checkpoint."""
    embed_dim = ckpt["norm.weight"].shape[0]
    size_map = {192: "tiny", 384: "small", 768: "base", 1024: "large"}
    return {
        "backbone_type": "dinov2",
        "vit_backbone_size": size_map.get(embed_dim, "base"),
        "vit_backbone_dim": embed_dim,
        "use_rope": False,
    }


def _encoder_state_from_tdv_ckpt(ckpt: dict) -> dict:
    """
    Extract frame_encoder weights from a TDV checkpoint.
    Prefers model.teacher_frame_encoder.* (EMA teacher); falls back to
    model.frame_encoder.* (student) if the run did not use EMA.
    Both are remapped to backbone.frame_encoder.*
    """
    state = ckpt.get("state_dict", ckpt.get("model", ckpt))
    encoder_state = {}

    for prefix in ("model.teacher_frame_encoder.", "model.frame_encoder."):
        for k, v in state.items():
            if k.startswith(prefix):
                new_k = "backbone.frame_encoder." + k[len(prefix):]
                encoder_state[new_k] = v
        if encoder_state:
            print(f"MidwayDINOv2FlowWithDPT: TDV checkpoint – loaded from prefix '{prefix}'")
            return encoder_state

    enc_keys = [k for k in state.keys() if "encoder" in k.lower()][:10]
    raise ValueError(
        "No weights found under 'model.teacher_frame_encoder.' or 'model.frame_encoder.' "
        f"in checkpoint. Sample encoder-related keys: {enc_keys}"
    )


def _encoder_state_from_dino_recipe_ckpt(ckpt: dict) -> dict:
    """
    Extract backbone weights from a dino-recipe checkpoint (EMA teacher).
    ckpt['teacher']['backbone.*'] → backbone.frame_encoder.*
    """
    teacher_state = ckpt["teacher"]
    encoder_state = {}
    for k, v in teacher_state.items():
        if k.startswith("backbone."):
            new_k = "backbone.frame_encoder." + k[len("backbone."):]
            encoder_state[new_k] = v
    if not encoder_state:
        keys = list(teacher_state.keys())[:10]
        raise ValueError(
            "No 'backbone.*' keys found in dino-recipe checkpoint['teacher']. "
            f"Sample keys: {keys}"
        )
    return encoder_state


def _interpolate_pos_embed(encoder_state: dict, model) -> dict:
    """
    Bicubic-interpolate pos_embed in encoder_state to match the model's shape.
    Needed when a checkpoint was trained at a different resolution
    (e.g. DINOv2 pretrain at 518×518 → [1,1370,768] vs model at 224×224 → [1,257,768]).
    Only the patch-position rows are interpolated; the cls-token row is kept as-is.
    """
    key = "backbone.frame_encoder.pos_embed"
    if key not in encoder_state:
        return encoder_state

    ckpt_pe = encoder_state[key]            # [1, N_src+1, D]
    model_pe = model.backbone.frame_encoder.pos_embed  # [1, N_tgt+1, D]
    if ckpt_pe.shape == model_pe.shape:
        return encoder_state

    print(f"[pos_embed] interpolating {tuple(ckpt_pe.shape)} → {tuple(model_pe.shape)}")
    cls  = ckpt_pe[:, :1, :]               # [1, 1, D]
    patches = ckpt_pe[:, 1:, :]            # [1, N_src, D]
    D = patches.shape[2]
    N_src = patches.shape[1]
    N_tgt = model_pe.shape[1] - 1          # strip cls

    src_h = src_w = int(N_src ** 0.5)
    tgt_h = tgt_w = int(N_tgt ** 0.5)

    patches = patches.reshape(1, src_h, src_w, D).permute(0, 3, 1, 2).float()
    patches = F.interpolate(patches, size=(tgt_h, tgt_w), mode='bicubic', align_corners=False)
    patches = patches.permute(0, 2, 3, 1).reshape(1, N_tgt, D).to(ckpt_pe.dtype)

    encoder_state = dict(encoder_state)    # don't mutate the original
    encoder_state[key] = torch.cat([cls, patches], dim=1)
    return encoder_state


def _encoder_state_from_ckpt(ckpt: dict) -> dict:
    """Auto-detect checkpoint type and return the encoder state dict."""
    if _is_dino_recipe_ckpt(ckpt):
        print("MidwayDINOv2FlowWithDPT: dino-recipe checkpoint detected – loading teacher backbone.")
        return _encoder_state_from_dino_recipe_ckpt(ckpt)
    if _is_dinov2_pretrain_ckpt(ckpt):
        print("MidwayDINOv2FlowWithDPT: official DINOv2 pretrain checkpoint detected – loading directly.")
        return _encoder_state_from_dinov2_pretrain_ckpt(ckpt)
    print("MidwayDINOv2FlowWithDPT: TDV checkpoint detected – loading teacher_frame_encoder.")
    return _encoder_state_from_tdv_ckpt(ckpt)


def _hparams_from_dino_recipe_ckpt(ckpt: dict) -> dict:
    """Build a hparams dict from the args namespace stored in a dino-recipe checkpoint."""
    args = ckpt.get("args")
    arch = getattr(args, "arch", "vit_base")
    size, dim = _DINO_ARCH_TO_SIZE_DIM.get(arch, ("base", 768))
    return {
        "backbone_type": "dinov2",
        "vit_backbone_size": size,
        "vit_backbone_dim": dim,
        "use_rope": False,
    }


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_midway_flow_model_from_checkpoint(
    tdv_ckpt_path: str,
    *,
    num_channels: int,
    # IterativeLatentMotion decoder config (match your training config)
    feature_levels=(2, 5, 8, 11),
    motion_dim: int = 192,
    motion_depth: int = 2,
    num_motion_heads: int = 3,
    motion_tokens: int = 10,
    motion_agg_type: str = 'add',
    motion_pred_input: bool = True,
    no_teacher: bool = False,
    predictor_depth: int = 4,
    num_pred_heads: int = 6,
    pred_type: str = 'self',
    use_pos_embed: bool = True,
    feature_block_type: str = 'identity',
    decoder_feature_mode: str = 'motion',
    gating_type: str = None,
    gating_bias: float = None,
    use_pred_pos_embed: bool = True,
    # Number of patch tokens per image for positional embeddings in the decoder.
    # Compute as (crop_H // patch_size) * (crop_W // patch_size).
    # e.g. crop 224×224 with DINOv2 patch_size=14 → 256 patches.
    # If None and use_pos_embed=True the decoder falls back to a degenerate
    # scalar bias (same embedding for all positions) — no spatial structure.
    num_patches: int = None,
    # cls-token handling in all_layers.
    # DINOv2 all_layers includes cls at index 0 (plus register tokens if any).
    # num_cls_tokens = 1 + num_register_tokens.  Default 1 for standard DINOv2.
    num_cls_tokens: int = 1,
    use_cls_token: bool = False,
    # DPT head config
    hooks_idx=None,          # None → auto-computed to match MidwayNetwork
    layer_dims=None,         # None → auto ([96,192,384,768])
    patch_size: int = 14,    # ViT patch size — must match the checkpoint (14 for DINOv2, 16 for DINOv1)
    # checkpoint / device
    map_location: str = "cpu",
    device: str = "cuda",
    strict: bool = False,
    # If True the encoder is initialised with its own pretrained weights and
    # no weights are loaded from tdv_ckpt_path (decoder + DPT still random).
    pretrained_encoder: bool = False,
    # TDV constructors (same ones used in tdv_flow_wrapper.py)
    create_image_encoder_fn=None,
):
    """
    Build a MidwayDINOv2FlowWithDPT from a TDV checkpoint.

    Only the frame_encoder weights are loaded; the IterativeLatentMotion
    decoder and DPT head are freshly initialised (you fine-tune them).

    Parameters
    ----------
    create_image_encoder_fn : callable
        Same factory used in tdv_flow_wrapper.build_tdv_flow_model_from_checkpoint.
        Signature: fn(backbone_type, vit_backbone_size, pretrained, use_rope,
                      use_masking) -> encoder
    """
    assert create_image_encoder_fn is not None, \
        "Pass create_image_encoder_fn (e.g. from model.model_utils import create_image_encoder)"

    ckpt = torch.load(tdv_ckpt_path, map_location=map_location, weights_only=False)
    _HPARAM_DEFAULTS = {
        "backbone_type": "dinov2",
        "vit_backbone_size": "base",
        "vit_backbone_dim": 768,   # ViT-B/14; change to 384 for ViT-S/14
        "use_rope": False,
    }
    if _is_dino_recipe_ckpt(ckpt):
        hparams = _hparams_from_dino_recipe_ckpt(ckpt)
        print(f"MidwayDINOv2FlowWithDPT: hparams from dino-recipe args: {hparams}")
    elif _is_dinov2_pretrain_ckpt(ckpt):
        hparams = _hparams_from_dinov2_pretrain_ckpt(ckpt)
        print(f"MidwayDINOv2FlowWithDPT: hparams inferred from DINOv2 pretrain shapes: {hparams}")
    else:
        hparams = ckpt.get("hyper_parameters", ckpt.get("hparams", None))
        if hparams is None:
            print(
                "[MidwayDINOv2FlowWithDPT WARNING] Checkpoint has no hyperparameters. "
                f"Falling back to defaults: {_HPARAM_DEFAULTS}"
            )
            hparams = _HPARAM_DEFAULTS
        else:
            # Fill in any keys missing from the checkpoint hparams
            for k, v in _HPARAM_DEFAULTS.items():
                hparams.setdefault(k, v)

    # ---- Build DINOv2 frame encoder ----------------------------------------
    frame_encoder = create_image_encoder_fn(
        hparams["backbone_type"],
        hparams["vit_backbone_size"],
        pretrained=pretrained_encoder,
        use_rope=hparams.get("use_rope", False),
        use_masking=False,
        patch_size=patch_size,
    )
    enc_embed_dim = hparams["vit_backbone_dim"]

    # Infer enc_depth from the encoder.
    # DINOv2 uses chunked_blocks=True so len(enc.blocks) = 1 (one chunk).
    # Use n_blocks (= actual block count) first; fall back to len(blocks).
    if hasattr(frame_encoder, "n_blocks"):
        enc_depth = frame_encoder.n_blocks
    elif hasattr(frame_encoder, "blocks"):
        enc_depth = len(frame_encoder.blocks)
    else:
        raise ValueError(
            "Cannot infer enc_depth from frame_encoder. "
            "Add an explicit enc_depth argument."
        )

    patch_size = getattr(frame_encoder, "patch_size", 14)
    if use_pos_embed and num_patches is None:
        print(
            "[MidwayDINOv2FlowWithDPT WARNING] use_pos_embed=True but num_patches=None. "
            "Positional embeddings will be a scalar bias (same for all spatial positions). "
            "Pass num_patches=(crop_H // patch_size) * (crop_W // patch_size) for correct behaviour."
        )

    # ---- Build IterativeLatentMotion decoder --------------------------------
    decoder = IterativeLatentMotion(
        decoder_feature_mode=decoder_feature_mode,
        embed_dim=enc_embed_dim,
        num_patches=num_patches,
        feature_levels=list(feature_levels),
        use_pos_embed=use_pos_embed,
        feature_block_type=feature_block_type,
        feature_depth=1,
        num_feature_heads=6,
        use_cls_token=use_cls_token,
        num_cls_tokens=num_cls_tokens,
        ema_teacher=False,   # no EMA update in flow fine-tuning; use student_embed for both images
        motion_dim=motion_dim,
        motion_depth=motion_depth,
        num_motion_heads=num_motion_heads,
        motion_tokens=motion_tokens,
        motion_agg_type=motion_agg_type,
        motion_pred_input=motion_pred_input,
        no_teacher=no_teacher,
        predictor_depth=predictor_depth,
        num_pred_heads=num_pred_heads,
        pred_type=pred_type,
        gating_type=gating_type,
        gating_bias=gating_bias,
        use_pred_pos_embed=use_pred_pos_embed,
        patch_size=patch_size,
    )

    # ---- Build DPT head ----------------------------------------------------
    head = MidwayPixelwiseTaskWithDPT(
        hooks_idx=hooks_idx,
        layer_dims=layer_dims or [96, 192, 384, 768],
        num_channels=num_channels,
        patch_size=patch_size,
    )

    # ---- Assemble -----------------------------------------------------------
    backbone = MidwayDINOv2Backbone(
        frame_encoder=frame_encoder,
        decoder=decoder,
        encoder_name=hparams["backbone_type"],
        enc_embed_dim=enc_embed_dim,
        enc_depth=enc_depth,
        num_cls_tokens=num_cls_tokens,
    )

    model = MidwayDINOv2FlowWithDPT(backbone=backbone, head=head).to(device)

    # ---- Load encoder weights from checkpoint ------------------------------
    if pretrained_encoder:
        print("MidwayDINOv2FlowWithDPT: pretrained_encoder=True, skipping checkpoint weight loading.")
    else:
        encoder_state = _encoder_state_from_ckpt(ckpt)
        encoder_state = _interpolate_pos_embed(encoder_state, model)
        # Official DINOv2 pretrain checkpoints store blocks flat: blocks.{i}.*
        # But the local ViT with block_chunks=1 stores them chunked: blocks.0.{i}.*
        # Remap so the block weights actually load instead of silently being dropped.
        if getattr(model.backbone.frame_encoder, 'chunked_blocks', False):
            import re
            remapped = {}
            for k, v in encoder_state.items():
                k_new = re.sub(
                    r'^(backbone\.frame_encoder\.blocks\.)(\d+)(\.)',
                    r'\g<1>0.\2\3',
                    k,
                )
                remapped[k_new] = v
            encoder_state = remapped
            print("[MidwayDINOv2FlowWithDPT] chunked_blocks=True: remapped flat block keys to chunked format.")
        msg = model.load_state_dict(encoder_state, strict=False)
        print("MidwayDINOv2FlowWithDPT – encoder weights loaded:", msg)

    return model, hparams
