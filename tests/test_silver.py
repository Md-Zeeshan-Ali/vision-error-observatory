import json

import numpy as np
import pytest

from veo.silver import class_ids, region_silver, silver_maps

NAMES = ["wall", "mirror", "chair", "floor", "doorreflection"]
WALL, MIRROR, CHAIR, FLOOR, REFLECTION = 1, 2, 3, 4, 5


@pytest.fixture
def scene():
    """Wall at 3 m with a mirror on it; a chair at 1.5 m with a 2 px leg; floor below."""
    labels = np.full((80, 80), WALL, dtype=np.uint16)
    depth = np.full((80, 80), 3.0, dtype=np.float32)
    labels[5:25, 5:25] = MIRROR  # same depth as the wall: no occluding boundary
    labels[30:40, 40:60] = CHAIR  # seat (thick)
    labels[40:70, 45:47] = CHAIR  # leg (thin)
    depth[labels == CHAIR] = 1.5
    labels[70:, :] = FLOOR
    instances = np.where(labels == CHAIR, 1, 0).astype(np.uint8)
    return labels, instances, depth, class_ids(NAMES)


def _mask(rows, cols):
    mask = np.zeros((80, 80), dtype=bool)
    mask[rows, cols] = True
    return mask


def test_class_groups():
    groups = class_ids(NAMES)
    assert groups["reflective"] == {MIRROR, REFLECTION}
    assert groups["structure"] == {WALL, FLOOR}
    assert groups["flat"] == {WALL}


def test_reflection_and_textureless_follow_annotations(scene):
    labels, instances, depth, groups = scene
    maps = silver_maps(labels, instances, depth, groups)
    assert region_silver(maps, _mask(slice(8, 20), slice(8, 20)))["labels"] == ["reflection"]
    assert "textureless" in region_silver(maps, _mask(slice(40, 60), slice(5, 30)))["labels"]


def test_thin_parts_of_objects_but_not_thick_parts(scene):
    labels, instances, depth, groups = scene
    maps = silver_maps(labels, instances, depth, groups)
    assert maps["thin_object"][50, 45] and maps["thin_object"][55, 46]
    assert not maps["thin_object"][35, 52]  # middle of the seat
    assert not maps["thin_object"][50, 10]  # wall is structure, never thin


def test_occluding_boundaries_need_a_depth_jump(scene):
    labels, instances, depth, groups = scene
    maps = silver_maps(labels, instances, depth, groups)
    assert maps["boundary_mixing"][30, 50]  # chair seat edge against the wall: 1.5 m vs 3 m
    assert not maps["boundary_mixing"][5, 15]  # mirror edge on the wall: same depth


def test_region_silver_reports_shares(scene):
    labels, instances, depth, groups = scene
    result = region_silver(silver_maps(labels, instances, depth, groups), _mask(slice(0, 80), slice(0, 80)))
    assert set(result["shares"]) == {"reflection", "thin_object", "boundary_mixing", "textureless"}
    assert result["annotated_share"] == 1.0
    json.dumps(result)


def test_unlabelled_pixels_get_no_silver_label():
    labels = np.zeros((20, 20), dtype=np.uint16)
    maps = silver_maps(labels, None, np.ones((20, 20), dtype=np.float32), class_ids(NAMES))
    result = region_silver(maps, np.ones((20, 20), dtype=bool))
    assert result["labels"] == [] and result["annotated_share"] == 0.0
