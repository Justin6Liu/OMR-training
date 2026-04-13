import json
from pathlib import Path

import torch
from PIL import Image

from scripts.train_rtdetrv3 import CocoDetectionDataset, collate_fn


def make_fake_coco(tmp_path: Path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    # create two tiny blank images that PIL can open
    for i in range(2):
        img = Image.new("RGB", (10, 10), color=(255, 255, 255))
        img.save(img_dir / f"{i}.jpg")
    ann = {
        "images": [
            {"id": 0, "file_name": "0.jpg", "width": 10, "height": 10},
            {"id": 1, "file_name": "1.jpg", "width": 10, "height": 10},
        ],
        "annotations": [
            {"id": 1, "image_id": 0, "bbox": [1, 1, 3, 3], "category_id": 1},
            {"id": 2, "image_id": 1, "bbox": [2, 2, 2, 2], "category_id": 1},
        ],
        "categories": [{"id": 1, "name": "notehead"}],
    }
    ann_path = tmp_path / "ann.json"
    ann_path.write_text(json.dumps(ann))
    return img_dir, ann_path


class DummyProcessor:
    def __call__(self, images, annotations, return_tensors):
        # mimic HF processor output
        return {
            "pixel_values": torch.zeros(3, 10, 10),
            "pixel_mask": torch.ones(10, 10),
            "labels": annotations,
        }


def test_dataset_loads(tmp_path):
    img_dir, ann_path = make_fake_coco(tmp_path)
    ds = CocoDetectionDataset(str(img_dir), str(ann_path), DummyProcessor())
    sample = ds[0]
    assert "pixel_values" in sample
    assert sample["pixel_values"].shape[-1] == 10


def test_collate_fn():
    batch = [{"pixel_values": torch.zeros(3, 10, 10)}, {"pixel_values": torch.ones(3, 10, 10)}]
    out = collate_fn(batch)
    assert isinstance(out["pixel_values"], list)
    assert len(out["pixel_values"]) == 2
