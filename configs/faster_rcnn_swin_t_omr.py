import json
import os
from pathlib import Path

_base_ = [
    "mmdet::faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py",
]

work_dir = os.getenv("WORK_DIR", "./work_dirs/faster_rcnn_swin_t_omr")

data_root = os.getenv(
    "DATA_ROOT",
    "/home/users/jl1430/jl1430/OMR-training/datasets/muscima_coco/",
)
img_root = os.getenv(
    "IMG_ROOT",
    "/home/users/jl1430/muscima-pp/v2.0/data/images/",
)
train_ann_file = os.getenv("TRAIN_ANN_FILE", "train.json")
val_ann_file = os.getenv("VAL_ANN_FILE", "val.json")
category_source = Path(os.getenv("CATEGORY_SOURCE_JSON", os.path.join(data_root, train_ann_file)))
_cats = json.loads(category_source.read_text())["categories"]
classes = tuple(c["name"] for c in _cats)


def _optional_pretrained_init():
    checkpoint = os.getenv("PRETRAINED")
    if not checkpoint or checkpoint.lower() == "none":
        return None
    return dict(type="Pretrained", checkpoint=checkpoint)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Resize", scale=(1333, 800), keep_ratio=True),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PackDetInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(1333, 800), keep_ratio=True),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="PackDetInputs"),
]

train_dataloader = dict(
    batch_size=int(os.getenv("BATCH_SIZE", 1)),
    num_workers=int(os.getenv("NUM_WORKERS", 2)),
    persistent_workers=False,
    dataset=dict(
        type="CocoDataset",
        data_root=data_root,
        ann_file=train_ann_file,
        data_prefix=dict(img=img_root),
        metainfo=dict(classes=classes),
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=train_pipeline,
        serialize_data=False,
    ),
    sampler=dict(type="DefaultSampler", shuffle=True),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=False,
    dataset=dict(
        type="CocoDataset",
        data_root=data_root,
        ann_file=val_ann_file,
        data_prefix=dict(img=img_root),
        metainfo=dict(classes=classes),
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=test_pipeline,
        serialize_data=False,
    ),
    sampler=dict(type="DefaultSampler", shuffle=False),
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type="CocoMetric",
    ann_file=os.path.join(data_root, val_ann_file),
    metric="bbox",
)
test_evaluator = val_evaluator

model = dict(
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
    ),
    backbone=dict(
        _delete_=True,
        type="SwinTransformer",
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=_optional_pretrained_init(),
    ),
    neck=dict(
        _delete_=True,
        type="FPN",
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        num_outs=5,
    ),
    rpn_head=dict(
        anchor_generator=dict(
            scales=[4, 8],
            ratios=[0.25, 0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64],
        ),
    ),
    roi_head=dict(
        bbox_head=dict(num_classes=len(classes)),
    ),
    test_cfg=dict(
        rcnn=dict(score_thr=float(os.getenv("SCORE_THR", 0.001))),
    ),
)

optim_wrapper = dict(
    _delete_=True,
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=float(os.getenv("LR", 0.0001)), betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            "absolute_pos_embed": dict(decay_mult=0.0),
            "relative_position_bias_table": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
        }
    ),
)

max_epochs = int(os.getenv("EPOCHS", 24))
train_cfg = dict(max_epochs=max_epochs, val_interval=int(os.getenv("VAL_INTERVAL", 1)))

param_scheduler = [
    dict(type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=1000),
    dict(
        type="MultiStepLR",
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[int(max_epochs * 0.67), int(max_epochs * 0.89)],
        gamma=0.1,
    ),
]

default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        interval=int(os.getenv("CKPT_INTERVAL", 1)),
        save_best="coco/bbox_mAP",
        rule="greater",
        max_keep_ckpts=int(os.getenv("MAX_KEEP_CKPTS", 3)),
        save_last=True,
    )
)

train_cfg.setdefault("type", "EpochBasedTrainLoop")
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
