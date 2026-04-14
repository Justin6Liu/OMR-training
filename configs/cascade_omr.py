import os
import json

# Cascade R-CNN (ResNet-101 FPN) for MUSCIMA++ (115 classes)
# Adjust paths below for your environment.

_base_ = [
    "mmdet::cascade_rcnn/cascade-rcnn_r101_fpn_1x_coco.py",
]

work_dir = os.getenv("WORK_DIR", "./work_dirs/cascade_omr")

data_root = "/home/users/jl1430/jl1430/OMR-training/datasets/muscima_coco/"
img_root = "/home/users/jl1430/muscima-pp/v2.0/data/images/"
train_ann_file = os.getenv("TRAIN_ANN_FILE", "train.json")
val_ann_file = os.getenv("VAL_ANN_FILE", "val.json")
category_source = os.getenv("CATEGORY_SOURCE_JSON", os.path.join(data_root, train_ann_file))

import json
_cats = json.loads(open(category_source).read())["categories"]
classes = tuple([c["name"] for c in _cats])

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type="CocoDataset",
        data_root=data_root,
        ann_file=train_ann_file,
        data_prefix=dict(img=img_root),
        metainfo=dict(classes=classes),
        filter_cfg=dict(filter_empty_gt=False),
    ),
    sampler=dict(type="DefaultSampler", shuffle=True),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    dataset=dict(
        type="CocoDataset",
        data_root=data_root,
        ann_file=val_ann_file,
        data_prefix=dict(img=img_root),
        metainfo=dict(classes=classes),
        filter_cfg=dict(filter_empty_gt=False),
    ),
    sampler=dict(type="DefaultSampler", shuffle=False),
)

test_dataloader = val_dataloader

val_evaluator = dict(type="CocoMetric", ann_file=data_root + val_ann_file, metric="bbox")
test_evaluator = val_evaluator

# Override image scale for lower memory (further downscaled)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(640, 448), keep_ratio=True),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PackDetInputs"),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(640, 448), keep_ratio=True),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="PackDetInputs"),
]

model = dict(
    type="CascadeRCNN",
    test_cfg=dict(rcnn=dict(score_thr=0.001)),
    roi_head=dict(
        bbox_head=[
            dict(
                type="Shared2FCBBoxHead",
                num_classes=len(classes),
                loss_cls=dict(
                    type="FocalLoss",
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=1.0,
                ),
            ),
            dict(
                type="Shared2FCBBoxHead",
                num_classes=len(classes),
                loss_cls=dict(
                    type="FocalLoss",
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=1.0,
                ),
            ),
            dict(
                type="Shared2FCBBoxHead",
                num_classes=len(classes),
                loss_cls=dict(
                    type="FocalLoss",
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=1.0,
                ),
            ),
        ],
    )
)
# Smaller/skinny anchors and more proposals for tiny symbols
# Lighter anchor setting for lower memory
model["rpn_head"] = dict(
    type="RPNHead",
    anchor_generator=dict(
        type="AnchorGenerator",
        scales=[4, 8, 16, 32, 64],
        ratios=[0.25, 0.5, 1.0, 2.0],
        strides=[4, 8, 16, 32, 64],
    ),
    bbox_coder=dict(type="DeltaXYWHBBoxCoder", target_means=[0., 0., 0., 0.],
                    target_stds=[1.0, 1.0, 1.0, 1.0]),
    loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
    loss_bbox=dict(type="SmoothL1Loss", beta=1.0 / 9.0, loss_weight=1.0)
)
# Reduce proposals/detections for memory
model["rpn_head"]["train_cfg"] = dict(
    assigner=dict(type="MaxIoUAssigner", pos_iou_thr=0.7, neg_iou_thr=0.3,
                  min_pos_iou=0.3, match_low_quality=True, ignore_iof_thr=-1),
    sampler=dict(type="RandomSampler", num=256, pos_fraction=0.5,
                 neg_pos_ub=-1, add_gt_as_proposals=False),
    allowed_border=-1,
    pos_weight=-1,
    debug=False,
    nms_pre=128,
    max_per_img=64,
)
model["rpn_head"]["test_cfg"] = dict(nms_pre=128, max_per_img=64)

# ROI configs for lower memory
model["roi_head"]["train_cfg"] = dict(
    rpn_proposal=dict(
        nms_pre=128, max_per_img=64, nms=dict(type="nms", iou_threshold=0.7), min_bbox_size=0),
    rcnn=dict(
        assigner=dict(
            type="MaxIoUAssigner",
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            match_low_quality=False,
            ignore_iof_thr=-1),
        sampler=dict(
            type="RandomSampler",
            num=256,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        bbox_coder=dict(
            type="DeltaXYWHBBoxCoder",
            target_means=[0., 0., 0., 0.],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        stage_loss_weights=[1.0, 0.5, 0.25]
    )
)
model["roi_head"]["test_cfg"] = dict(
    rpn=dict(nms_pre=256, max_per_img=128, nms=dict(type="nms", iou_threshold=0.7), min_bbox_size=0),
    rcnn=dict(
        score_thr=0.001,
        nms=dict(type="nms", iou_threshold=0.5),
        max_per_img=30))

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="SGD", lr=0.002, momentum=0.9, weight_decay=0.0001),
)

# Allow overriding epochs via EPOCHS env var (default 36)
train_cfg = dict(max_epochs=int(os.getenv("EPOCHS", 36)))

# Keep a few checkpoints; save every 4 epochs by default
default_hooks = dict(checkpoint=dict(type="CheckpointHook", interval=4, max_keep_ckpts=3))

# Pretrained detector weights (COCO) — R101
# You can override via PRETRAINED env var; default matches cluster path.
load_from = os.getenv(
    "PRETRAINED",
    "/home/users/jl1430/jl1430/OMR-training/cascade_rcnn_r101_fpn_1x_coco_20200317-0b6a2fbf.pth",
)
