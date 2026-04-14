import os
import json

# Cascade R-CNN R50 FPN for MUSCIMA++ (115 classes)
# Adjust paths below for your environment.

_base_ = [
    # Switch to R101 backbone config
    "mmdet::cascade_rcnn/cascade-rcnn_r101_fpn_1x_coco.py",
]

data_root = "/home/users/jl1430/jl1430/OMR-training/datasets/muscima_coco/"
img_root = "/home/users/jl1430/muscima-pp/v2.0/data/images/"

import json
_cats = json.load(open(os.path.join(data_root, "train.json")))["categories"]
classes = tuple([c["name"] for c in _cats])

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type="CocoDataset",
        data_root=data_root,
        ann_file="train.json",
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
        ann_file="val.json",
        data_prefix=dict(img=img_root),
        metainfo=dict(classes=classes),
        filter_cfg=dict(filter_empty_gt=False),
    ),
    sampler=dict(type="DefaultSampler", shuffle=False),
)

test_dataloader = val_dataloader

val_evaluator = dict(type="CocoMetric", ann_file=data_root + "val.json", metric="bbox")
test_evaluator = val_evaluator

model = dict(
    type="CascadeRCNN",
    test_cfg=dict(rcnn=dict(score_thr=0.001)),
    roi_head=dict(
        bbox_head=[
            dict(type="Shared2FCBBoxHead", num_classes=len(classes)),
            dict(type="Shared2FCBBoxHead", num_classes=len(classes)),
            dict(type="Shared2FCBBoxHead", num_classes=len(classes)),
        ],
    )
)
# Smaller/skinny anchors and more proposals for tiny symbols
# Lighter anchor setting for lower memory
model["rpn_head"] = dict(
    type="RPNHead",
    anchor_generator=dict(
        type="AnchorGenerator",
        scales=[8, 16, 32, 64, 128],
        ratios=[0.5, 1.0, 2.0],
        strides=[4, 8, 16, 32, 64],
    ),
    bbox_coder=dict(type="DeltaXYWHBBoxCoder", target_means=[0., 0., 0., 0.],
                    target_stds=[1.0, 1.0, 1.0, 1.0]),
    loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
    loss_bbox=dict(type="SmoothL1Loss", beta=1.0 / 9.0, loss_weight=1.0)
)

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="SGD", lr=0.002, momentum=0.9, weight_decay=0.0001),
)

train_cfg = dict(max_epochs=12)
train_cfg = dict(max_epochs=24)

default_hooks = dict(checkpoint=dict(type="CheckpointHook", interval=1, max_keep_ckpts=1))
default_hooks = dict(checkpoint=dict(type="CheckpointHook", interval=4, max_keep_ckpts=3))

load_from = "/home/users/jl1430/jl1430/OMR-training/cascade_rcnn_r101_fpn_1x_coco_20200317-0b6a2fbf.pth"
