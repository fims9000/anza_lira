import numpy as np
from PIL import Image

from datasets.cracks import BLUE, CRACKSAnnotatedSectionDataset, WHITE


def test_annotated_dataset_maps_policy_and_deterministic_crop(tmp_path) -> None:
    image_root = tmp_path / "images"
    annotation_root = tmp_path / "annotations"
    image_root.mkdir()
    annotation_root.mkdir()
    image = np.zeros((255, 701, 3), dtype=np.uint8)
    mask = np.full((255, 701, 3), WHITE, dtype=np.uint8)
    mask[120:123, 330:370] = BLUE
    Image.fromarray(image).save(image_root / "section_001.png")
    Image.fromarray(mask).save(annotation_root / "section_001.png")
    dataset = CRACKSAnnotatedSectionDataset(
        image_root,
        annotation_root,
        [1],
        policy_name="paper_like",
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        crop_size=256,
        foreground_probability=1.0,
        seed=9,
    )
    first = dataset[0]
    second = dataset[0]
    assert first["image"].shape == (3, 256, 256)
    assert first["target"].shape == (1, 256, 256)
    assert int(first["valid"].sum()) == 255 * 256
    assert not first["valid"][0, -1].any()
    assert first["target"].sum() == 120
    assert first["crop_origin"] == second["crop_origin"]
