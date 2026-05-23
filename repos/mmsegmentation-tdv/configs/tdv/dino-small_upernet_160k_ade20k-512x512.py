_base_ = [
    '../_base_/models/upernet_tdv.py',
    '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py',
]

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend')
]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')

data_root = '/path/to/ade20k/ADEChallengeData2016'  # set to your ADE20K root
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(
        type='RandomResize',
        scale=(2048, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 512), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs')
]

# -----------------------------------------------------------------------
# Model — DINO ViT-Small (embed_dim=384), patch_size=16, no RoPE
# -----------------------------------------------------------------------
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        backbone_size='small',
        checkpoint_path=None,  # set via --cfg-options or shell script
        patch_size=16,
        use_rope=False,
        frozen=True),
    neck=dict(embed_dim=384),
    decode_head=dict(
        in_channels=[384, 384, 384, 384],
        num_classes=150),
    auxiliary_head=dict(
        in_channels=384,
        num_classes=150),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(341, 341)))

# -----------------------------------------------------------------------
# Optimiser  –  only the neck + heads are trainable (backbone is frozen)
# -----------------------------------------------------------------------
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.05))

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        power=1.0,
        begin=1500,
        end=160000,
        eta_min=0.0,
        by_epoch=False),
]

train_dataloader = dict(
    batch_size=4, dataset=dict(data_root=data_root, pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1, dataset=dict(data_root=data_root, pipeline=test_pipeline))
test_dataloader = val_dataloader
