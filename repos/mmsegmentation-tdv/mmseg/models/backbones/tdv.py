import sys
import torch
from mmengine.model import BaseModule
from mmseg.registry import MODELS


_EMBED_DIMS = {
    'small': 384,
    'base': 768,
    'large': 1024,
    'huge': 1280,
    'giant': 1536,
}

# Default block indices to extract for each backbone size.
# For 12-block ViTs (small/base): evenly spaced across 12 blocks.
# For 24-block ViTs (large/huge): evenly spaced across 24 blocks.
# For 40-block ViTs (giant): evenly spaced across 40 blocks.
_DEFAULT_OUT_INDICES = {
    'small': (2, 5, 8, 11),
    'base': (2, 5, 8, 11),
    'large': (5, 11, 17, 23),
    'huge': (5, 11, 17, 23),
    'giant': (9, 19, 29, 39),
}


@MODELS.register_module()
class TDVBackbone(BaseModule):
    """DINOv2 ViT frame encoder extracted from a TDV training checkpoint.

    Loads the ``frame_encoder`` from a PyTorch Lightning checkpoint saved by
    the TDV training pipeline, or falls back to pretrained DINOv2 weights
    when no checkpoint is provided.

    The backbone returns intermediate layer features reshaped to 2-D spatial
    maps ``(B, embed_dim, H/patch_size, W/patch_size)`` at the chosen block
    indices, suitable for a Feature2Pyramid + UPerNet decode head.

    Args:
        backbone_size (str): One of ``'small'``, ``'base'``, ``'large'``,
            ``'huge'``, ``'giant'``. Determines embed_dim and the default
            out_indices. Default: ``'base'``.
        checkpoint_path (str | None): Path to a TDV PL checkpoint.  The
            ``frame_encoder`` weights are extracted using the key prefix
            ``model.frame_encoder.*``.  When ``None``, pretrained DINOv2
            weights are loaded from torch hub. Default: ``None``.
        out_indices (tuple[int] | None): Block indices from which to extract
            features.  ``None`` uses sensible defaults per backbone size.
        frozen (bool): Freeze backbone weights (no gradient). Default: True.
        tdv_repo_path (str): Filesystem path to the root of the TDV repo so
            that ``model.*`` imports resolve correctly.
            Required — set this to your local clone of the TDV repo.
        init_cfg (dict | list[dict] | None): Initialisation config.
    """

    def __init__(
        self,
        backbone_size='base',
        checkpoint_path=None,
        out_indices=None,
        frozen=True,
        patch_size=14,
        use_rope=False,
        random_init=False,
        tdv_repo_path=None,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)

        if tdv_repo_path is None:
            raise ValueError(
                "tdv_repo_path must be set to the root of your local TDV repo clone. "
                "Pass it in the backbone config: tdv_repo_path='/path/to/tdv-clean'"
            )
        if tdv_repo_path not in sys.path:
            sys.path.insert(0, tdv_repo_path)

        from model.model_utils import create_image_encoder

        self.embed_dim = _EMBED_DIMS[backbone_size]
        self.out_indices = out_indices if out_indices is not None else _DEFAULT_OUT_INDICES[backbone_size]

        if random_init:
            # No pretrained weights — purely random initialisation (baseline).
            self.encoder = create_image_encoder(
                'dinov2', backbone_size, pretrained=False,
                patch_size=patch_size, use_rope=use_rope,
            )
        elif checkpoint_path is not None:
            self.encoder = self._load_from_tdv_checkpoint(
                checkpoint_path, backbone_size, create_image_encoder,
                patch_size=patch_size, use_rope=use_rope,
            )
        else:
            self.encoder = create_image_encoder(
                'dinov2', backbone_size, pretrained=True
            )

        if frozen:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

    @staticmethod
    def _load_from_tdv_checkpoint(ckpt_path, backbone_size, create_image_encoder,
                                   patch_size=14, use_rope=False):
        """Load encoder weights from a TDV PL checkpoint or a DINO checkpoint.

        Supported formats:
        - TDV (PyTorch Lightning): has ``state_dict`` with ``model.frame_encoder.*`` keys.
        - DINO: has ``teacher`` with ``backbone.*`` keys.
        """
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

        if 'state_dict' in checkpoint:
            # TDV / PyTorch Lightning format
            prefix = 'model.frame_encoder.'
            encoder_sd = {
                k[len(prefix):]: v
                for k, v in checkpoint['state_dict'].items()
                if k.startswith(prefix)
            }
            if not encoder_sd:
                raise KeyError(
                    f'No weights with prefix "{prefix}" found in {ckpt_path}.'
                )
        elif 'teacher' in checkpoint:
            # DINO format — use teacher backbone
            # The chunked model (block_chunks=1) stores blocks as blocks.0.{i}.*
            # but DINO checkpoints use flat blocks.{i}.* — remap accordingly.
            prefix = 'backbone.'
            raw_sd = {
                k[len(prefix):]: v
                for k, v in checkpoint['teacher'].items()
                if k.startswith(prefix)
            }
            if not raw_sd:
                raise KeyError(
                    f'No weights with prefix "{prefix}" found under "teacher" in {ckpt_path}.'
                )
            encoder_sd = {}
            for k, v in raw_sd.items():
                if k.startswith('blocks.'):
                    # blocks.{i}.rest  →  blocks.0.{i}.rest
                    rest = k[len('blocks.'):]
                    block_idx, remainder = rest.split('.', 1)
                    k = f'blocks.0.{block_idx}.{remainder}'
                encoder_sd[k] = v
        else:
            raise KeyError(
                f'Unrecognised checkpoint format in {ckpt_path}. '
                f'Expected "state_dict" (TDV) or "teacher" (DINO). '
                f'Got keys: {list(checkpoint.keys())}'
            )

        encoder = create_image_encoder('dinov2', backbone_size, pretrained=False,
                                        patch_size=patch_size, use_rope=use_rope)
        encoder.load_state_dict(encoder_sd, strict=True)

        return encoder

    def forward(self, x):
        """Extract multi-scale spatial features from the DINOv2 encoder.

        Args:
            x (Tensor): Input images ``(B, 3, H, W)``.

        Returns:
            tuple[Tensor]: Feature maps at each ``out_indices`` block,
                each of shape ``(B, embed_dim, H//patch_size, W//patch_size)``.
        """
        features = self.encoder.get_intermediate_layers(
            x,
            n=self.out_indices,
            reshape=True,
            norm=True,
        )
        return tuple(features)
