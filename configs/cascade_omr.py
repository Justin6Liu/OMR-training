# Cascade R-CNN R50 FPN for MUSCIMA++ detection (115 classes)
# Minimal training config; adjust epochs/batch for full runs.

base = [
    "mmdet::cascade_rcnn/cascade-rcnn_r50_fpn_1x_coco.py",
]

data_root = "/Users/justinliu/Documents/GitHub/OMR-training/datasets/muscima_coco/"

classes = tuple([f"class_{i}" for i in range(115)])  # placeholder names; ids map in COCO

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type="CocoDataset",
        data_root=data_root,
        ann_file="train.json",
        data_prefix=dict(img="images/"),
        metainfo=dict(classes=classes),
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    dataset=dict(
        type="CocoDataset",
        data_root=data_root,
        ann_file="val.json",
        data_prefix=dict(img="images/"),
        metainfo=dict(classes=classes),
    ),
)

test_dataloader = val_dataloader

val_evaluator = dict(type="CocoMetric", ann_file=data_root + "val.json", metric="bbox")
test_evaluator = val_evaluator

model = dict(
    roi_head=dict(
        bbox_head=[
            dict(num_classes=115),
            dict(num_classes=115),
            dict(num_classes=115),
        ]
    )
)

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="SGD", lr=0.002, momentum=0.9, weight_decay=0.0001),
)

train_cfg = dict(max_epochs=1)  # sanity; increase for real training

default_hooks = dict(checkpoint=dict(type="CheckpointHook", interval=1, max_keep_ckpts=1))

load_from = "https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco/cascade_rcnn_r50_fpn_1x_coco_20200317-0b6a2fbf.pth"
