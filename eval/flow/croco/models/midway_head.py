# MidwayPixelwiseTaskWithDPT — DPT head for the Midway pipeline.
# Copied from croco-chris/src/models/head_downstream.py.
# Lives here so we don't modify the existing head_downstream.py.

import logging
import torch.nn as nn
from .dpt_block import DPTOutputAdapter

log = logging.getLogger(__name__)


class MidwayPixelwiseTaskWithDPT(nn.Module):
    """
    DPT head that understands the Midway mixed-dim feature list
    (encoder layers at enc_embed_dim, decoder motion layers at motion_dim).

    setup(model_stub) auto-computes hooks_idx and dim_tokens from:
      model_stub.encoder.blocks           (list, len = enc_depth)
      model_stub.encoder.embed_dim        (int)
      model_stub.decoder.feature_levels   (list)
      model_stub.decoder.motion_depth     (int)
      model_stub.decoder.motion_dim       (int)
      model_stub.decoder_feature_mode     (str, expected 'motion')
    """

    def __init__(self, *, hooks_idx=None, layer_dims=[96, 192, 384, 768],
                 output_width_ratio=1, num_channels=1, postprocess=None,
                 dim_tokens=None, patch_size=14, **kwargs):
        super().__init__()
        self.return_all_blocks = True
        self.postprocess = postprocess
        self.output_width_ratio = output_width_ratio
        self.num_channels = num_channels
        self.hooks_idx = hooks_idx
        self.layer_dims = layer_dims
        self.dim_tokens = dim_tokens
        self.patch_size = patch_size  # must match the encoder patch size (14 for DINOv2)

    def setup(self, model):
        dpt_args = {
            'output_width_ratio': self.output_width_ratio,
            'num_channels': self.num_channels,
            'patch_size': self.patch_size,
        }

        if self.hooks_idx is None:
            if hasattr(model, 'decoder'):
                if model.decoder_feature_mode in ('motion', 'motion2', 'motion2-forward'):
                    pred_depth = getattr(model.decoder, 'predictor_depth', 0)
                    outputs_per_level = model.decoder.motion_depth + pred_depth
                    depth = (len(model.encoder.blocks) +
                             (len(model.decoder.feature_levels) - 1) *
                             outputs_per_level)
                    step = outputs_per_level
                elif model.decoder_feature_mode == 'feature':
                    depth = (len(model.encoder.blocks) +
                             (len(model.decoder.feature_levels) - 1) *
                             model.decoder.feature_depth)
                    step = model.decoder.feature_depth
                else:
                    raise ValueError(
                        f'Unknown decoder_feature_mode: {model.decoder_feature_mode}')
                hooks_idx = [depth - 1 - i * step for i in range(3, -1, -1)]
            else:
                step = len(model.encoder.blocks) // 4
                depth = len(model.encoder.blocks)
                hooks_idx = [depth - 1 - i * step for i in range(3, -1, -1)]
            self.hooks_idx = hooks_idx
            log.info(f'  MidwayPixelwiseTaskWithDPT: auto hooks_idx={self.hooks_idx}')

        dpt_args['hooks'] = self.hooks_idx
        dpt_args['layer_dims'] = self.layer_dims
        self.dpt = DPTOutputAdapter(**dpt_args)

        if self.dim_tokens is not None:
            dim_tokens = self.dim_tokens
        else:
            if hasattr(model, 'decoder') and model.decoder_feature_mode in ('motion', 'motion2', 'motion2-forward'):
                enc_depth = len(model.encoder.blocks)
                motion_depth = model.decoder.motion_depth
                # predictor blocks output embed_dim, not motion_dim
                pred_depth = getattr(model.decoder, 'predictor_depth', 0)
                outputs_per_level = motion_depth + pred_depth
                dim_tokens = []
                for hook in self.hooks_idx:
                    if hook < enc_depth:
                        dim_tokens.append(model.encoder.embed_dim)
                    else:
                        offset = (hook - enc_depth) % outputs_per_level
                        if offset < motion_depth:
                            dim_tokens.append(model.decoder.motion_dim)
                        else:
                            dim_tokens.append(model.encoder.embed_dim)
            else:
                dim_tokens = [model.encoder.embed_dim for _ in self.hooks_idx]

        self.dpt.init(dim_tokens_enc=dim_tokens)

    def forward(self, x, img_info):
        out = self.dpt(x, image_size=(img_info['height'], img_info['width']))
        if self.postprocess:
            out = self.postprocess(out)
        return out
