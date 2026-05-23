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
crop_size = (504, 504)
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
# Model
# -----------------------------------------------------------------------
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        # Set checkpoint_path to your TDV .ckpt file, e.g.:
        #   checkpoint_path='/path/to/your/tdv_checkpoint.ckpt'
        # Leave as None to use pretrained DINOv2 weights from torch hub.
        checkpoint_path=None,
        frozen=True),
    decode_head=dict(num_classes=150),
    auxiliary_head=dict(num_classes=150),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(336, 336)))

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
