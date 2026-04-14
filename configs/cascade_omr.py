# Cascade R-CNN R50 FPN for MUSCIMA++ (115 classes)
# Adjust paths below for your environment.

_base_ = [
    # Use the COCO-pretrained Cascade Mask R-CNN R50 FPN weights
    "mmdet::cascade_rcnn/cascade-mask-rcnn_r50_fpn_1x_coco.py",
]

data_root = "/home/users/jl1430/jl1430/OMR-training/datasets/muscima_coco/"
img_root = "/home/users/jl1430/muscima-pp/v2.0/data/images/"

classes = tuple([f"class_{i}" for i in range(115)])

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type="CocoDataset",
        data_root=data_root,
        ann_file="train.json",
        data_prefix=dict(img=img_root),
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
        data_prefix=dict(img=img_root),
        metainfo=dict(classes=classes),
    ),
)

test_dataloader = val_dataloader

val_evaluator = dict(type="CocoMetric", ann_file=data_root + "val.json", metric="bbox")
test_evaluator = val_evaluator

model = dict(
    type="CascadeRCNN",
    roi_head=dict(
        bbox_head=[
            dict(type="Shared2FCBBoxHead", num_classes=115),
            dict(type="Shared2FCBBoxHead", num_classes=115),
            dict(type="Shared2FCBBoxHead", num_classes=115),
        ],
        mask_head=[
            dict(type="FCNMaskHead", num_classes=115),
            dict(type="FCNMaskHead", num_classes=115),
            dict(type="FCNMaskHead", num_classes=115),
        ],
    )
)

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="SGD", lr=0.002, momentum=0.9, weight_decay=0.0001),
)

train_cfg = dict(max_epochs=12)

default_hooks = dict(checkpoint=dict(type="CheckpointHook", interval=1, max_keep_ckpts=1))

load_from = "/home/users/jl1430/jl1430/OMR-training/cascade_mask_rcnn_r50_fpn_1x_coco_20200203-9d4dcb24.pth"
