import os
import json

# Cascade R-CNN R50 FPN for MUSCIMA++ (115 classes)
# Adjust paths below for your environment.

_base_ = [
    # Use Cascade R-CNN backbone config (box only; our dataset lacks masks)
    "mmdet::cascade_rcnn/cascade-rcnn_r50_fpn_1x_coco.py",
]

data_root = "/home/users/jl1430/jl1430/OMR-training/datasets/muscima_coco/"
img_root = "/home/users/jl1430/muscima-pp/v2.0/data/images/"

import json
_cats = json.load(open(os.path.join(data_root, "train.json")))["categories"]
classes = tuple([c["name"] for c in _cats])

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type="CocoDataset",
        data_root=data_root,
        ann_file="train.json",
        data_prefix=dict(img=img_root),
        metainfo=dict(classes=classes),
        filter_cfg=dict(filter_empty_gt=False),
    ),
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

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="SGD", lr=0.002, momentum=0.9, weight_decay=0.0001),
)

train_cfg = dict(max_epochs=12)

default_hooks = dict(checkpoint=dict(type="CheckpointHook", interval=1, max_keep_ckpts=1))

load_from = "https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco/cascade_rcnn_r50_fpn_1x_coco_20200317-0b6a2fb4.pth"
