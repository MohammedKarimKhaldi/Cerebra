"""Unit tests for postprocess.py — pure functions, no I/O."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cerebra.postprocess import (
    PROB_THRESHOLD,
    VOLUME_THRESHOLD_MM3,
    compute_confidence,
    compute_max_diameter_mm,
    compute_volume_mm3,
    largest_component,
    make_triage_decision,
    map_anatomical_region,
    postprocess,
    threshold_and_label,
    tumour_probability_map,
)
from cerebra.schemas import TriageDecision


# --- tumour_probability_map ---

def test_tumour_probability_map_all_background():
    probs = torch.zeros(1, 4, 10, 10, 10)
    probs[0, 0] = 1.0
    result = tumour_probability_map(probs)
    assert result.shape == (10, 10, 10)
    assert np.allclose(result, 0.0)


def test_tumour_probability_map_full_tumour():
    probs = torch.zeros(1, 4, 10, 10, 10)
    probs[0, 1] = 1.0
    result = tumour_probability_map(probs)
    assert np.allclose(result, 1.0)


def test_tumour_probability_map_uses_synthetic_probs(synthetic_probs):
    result = tumour_probability_map(synthetic_probs)
    assert result.shape == (120, 120, 80)
    # Should have high probability somewhere
    assert result.max() > 0.5


# --- threshold_and_label ---

def test_threshold_returns_zero_when_below():
    prob_map = np.full((10, 10, 10), 0.3, dtype=np.float32)
    labels = threshold_and_label(prob_map, threshold=0.5)
    assert labels.max() == 0


def test_threshold_labels_distinct_blobs():
    prob_map = np.zeros((20, 20, 20), dtype=np.float32)
    prob_map[2:5, 2:5, 2:5] = 0.9
    prob_map[14:17, 14:17, 14:17] = 0.9
    labels = threshold_and_label(prob_map)
    assert labels.max() == 2


# --- largest_component ---

def test_largest_component_empty():
    labels = np.zeros((10, 10, 10), dtype=np.int32)
    mask = largest_component(labels)
    assert not mask.any()


def test_largest_component_picks_bigger():
    prob_map = np.zeros((20, 20, 20), dtype=np.float32)
    prob_map[1:4, 1:4, 1:4] = 0.9    # small blob
    prob_map[10:16, 10:16, 10:16] = 0.9  # big blob
    labels = threshold_and_label(prob_map)
    mask = largest_component(labels)
    # centroid of big blob should be inside mask
    assert mask[13, 13, 13]
    # centroid of small blob should not be inside mask
    assert not mask[2, 2, 2]


# --- compute_volume_mm3 ---

def test_volume_mm3_zero():
    mask = np.zeros((10, 10, 10), dtype=bool)
    assert compute_volume_mm3(mask) == 0.0


def test_volume_mm3_known_value():
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[0:5, 0:5, 0:5] = True   # 125 voxels
    assert compute_volume_mm3(mask) == pytest.approx(125.0)


def test_volume_mm3_custom_voxel_size():
    mask = np.ones((10, 10, 10), dtype=bool)
    # 1000 voxels × 2.0 mm³ each
    assert compute_volume_mm3(mask, voxel_volume=2.0) == pytest.approx(2000.0)


# --- compute_max_diameter_mm ---

def test_max_diameter_empty():
    mask = np.zeros((10, 10, 10), dtype=bool)
    assert compute_max_diameter_mm(mask) == 0.0


def test_max_diameter_cube():
    mask = np.zeros((30, 30, 30), dtype=bool)
    mask[5:15, 5:20, 5:10] = True   # 10 × 15 × 5 — largest axial edge is 15
    assert compute_max_diameter_mm(mask) == pytest.approx(15.0)


# --- map_anatomical_region ---

def test_region_unknown_when_empty():
    mask = np.zeros((100, 100, 80), dtype=bool)
    assert map_anatomical_region(mask, (100, 100, 80)) == "unknown"


def test_region_right_frontal():
    mask = np.zeros((100, 100, 80), dtype=bool)
    # centroid at (70, 10, 40) → right (x>50), frontal (y/100 < 1/3)
    mask[65:75, 5:15, 35:45] = True
    region = map_anatomical_region(mask, (100, 100, 80))
    assert region == "right frontal"


def test_region_left_occipital():
    mask = np.zeros((100, 100, 80), dtype=bool)
    # centroid at (20, 90, 40) → left (x<50), occipital (y/100 > 2/3)
    mask[15:25, 85:95, 35:45] = True
    region = map_anatomical_region(mask, (100, 100, 80))
    assert region == "left occipital"


# --- compute_confidence ---

def test_confidence_zero_when_no_mask():
    prob_map = np.ones((10, 10, 10), dtype=np.float32) * 0.7
    mask = np.zeros((10, 10, 10), dtype=bool)
    assert compute_confidence(prob_map, mask) == 0.0


def test_confidence_clipped():
    prob_map = np.full((10, 10, 10), 1.5, dtype=np.float32)  # over 1
    mask = np.ones((10, 10, 10), dtype=bool)
    assert compute_confidence(prob_map, mask) == pytest.approx(1.0)


# --- make_triage_decision ---

def test_flag_when_both_thresholds_met():
    assert make_triage_decision(VOLUME_THRESHOLD_MM3, PROB_THRESHOLD) == TriageDecision.FLAG


def test_clear_when_volume_too_small():
    assert make_triage_decision(VOLUME_THRESHOLD_MM3 - 1, 0.9) == TriageDecision.CLEAR


def test_clear_when_confidence_too_low():
    assert make_triage_decision(10000.0, PROB_THRESHOLD - 0.01) == TriageDecision.CLEAR


# --- end-to-end postprocess ---

def test_postprocess_with_synthetic_probs(synthetic_probs):
    findings, decision, confidence, mask = postprocess(synthetic_probs, (120, 120, 80))
    assert findings.tumour_suspected
    assert findings.tumour_volume_mm3 > 0
    assert 0.0 <= confidence <= 1.0
    assert decision in (TriageDecision.FLAG, TriageDecision.CLEAR)
    assert mask.shape == (120, 120, 80)
