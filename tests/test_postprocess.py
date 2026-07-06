"""Unit tests for postprocess.py — pure functions, no I/O."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cerebra.postprocess import (
    PROB_THRESHOLD,
    VOLUME_THRESHOLD_MM3,
    compute_max_diameter_mm,
    compute_volume_mm3,
    largest_component,
    make_triage_decision,
    map_anatomical_region,
    postprocess,
    study_tumour_probability,
    threshold_and_label,
    to_prob_map,
)
from cerebra.schemas import TriageDecision


# --- to_prob_map ---

def test_to_prob_map_passthrough_3d():
    m = np.random.rand(10, 10, 10).astype(np.float32)
    out = to_prob_map(m)
    assert out.shape == (10, 10, 10)
    assert np.allclose(out, m)


def test_to_prob_map_from_4class_tensor():
    probs = torch.zeros(1, 4, 8, 8, 8)
    probs[0, 0] = 1.0  # all background
    out = to_prob_map(probs)
    assert out.shape == (8, 8, 8)
    assert np.allclose(out, 0.0)


def test_to_prob_map_4class_full_tumour():
    probs = torch.zeros(1, 4, 8, 8, 8)
    probs[0, 1] = 1.0  # class-1 tumour everywhere; background=0
    out = to_prob_map(probs)
    assert np.allclose(out, 1.0)


def test_to_prob_map_accepts_torch_3d():
    m = torch.rand(6, 6, 6)
    out = to_prob_map(m)
    assert out.shape == (6, 6, 6)


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
    prob_map[1:4, 1:4, 1:4] = 0.9
    prob_map[10:16, 10:16, 10:16] = 0.9
    labels = threshold_and_label(prob_map)
    mask = largest_component(labels)
    assert mask[13, 13, 13]
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
    mask[65:75, 5:15, 35:45] = True   # right (x>50), frontal (y/100 < 1/3)
    assert map_anatomical_region(mask, (100, 100, 80)) == "right frontal"


def test_region_left_occipital():
    mask = np.zeros((100, 100, 80), dtype=bool)
    mask[15:25, 85:95, 35:45] = True  # left (x<50), occipital (y/100 > 2/3)
    assert map_anatomical_region(mask, (100, 100, 80)) == "left occipital"


# --- study_tumour_probability ---

def test_study_probability_zero_when_no_mask():
    prob_map = np.ones((10, 10, 10), dtype=np.float32) * 0.7
    mask = np.zeros((10, 10, 10), dtype=bool)
    assert study_tumour_probability(prob_map, mask) == 0.0


def test_study_probability_clipped_and_high():
    prob_map = np.full((10, 10, 10), 1.5, dtype=np.float32)  # over 1
    mask = np.ones((10, 10, 10), dtype=bool)
    assert study_tumour_probability(prob_map, mask) == pytest.approx(1.0)


def test_study_probability_uses_top_decile():
    prob_map = np.zeros((10, 10, 10), dtype=np.float32)
    mask = np.ones((10, 10, 10), dtype=bool)
    prob_map[...] = 0.2
    # make 10% of voxels very confident
    flat = prob_map.reshape(-1)
    flat[:100] = 0.95
    val = study_tumour_probability(prob_map.reshape(10, 10, 10), mask)
    assert val > 0.5   # dominated by the confident decile, not the 0.2 background


# --- make_triage_decision ---

def test_flag_when_both_thresholds_met():
    assert make_triage_decision(VOLUME_THRESHOLD_MM3, PROB_THRESHOLD) == TriageDecision.FLAG


def test_clear_when_volume_too_small():
    assert make_triage_decision(VOLUME_THRESHOLD_MM3 - 1, 0.9) == TriageDecision.CLEAR


def test_clear_when_confidence_too_low():
    assert make_triage_decision(10000.0, PROB_THRESHOLD - 0.01) == TriageDecision.CLEAR


# --- end-to-end postprocess ---

def test_postprocess_with_prob_map(synthetic_prob_map):
    findings, decision, confidence, mask = postprocess(synthetic_prob_map, (120, 120, 80))
    assert findings.tumour_suspected
    assert findings.tumour_volume_mm3 > 0
    assert 0.0 <= confidence <= 1.0
    assert decision == TriageDecision.FLAG   # big, confident blob
    assert mask.shape == (120, 120, 80)
    assert findings.anatomical_region == "right frontal"


def test_postprocess_with_legacy_4class(synthetic_probs):
    findings, decision, confidence, mask = postprocess(synthetic_probs, (120, 120, 80))
    assert findings.tumour_suspected
    assert 0.0 <= confidence <= 1.0
    assert mask.shape == (120, 120, 80)


def test_postprocess_clear_when_empty():
    empty = np.zeros((60, 60, 40), dtype=np.float32)
    findings, decision, confidence, mask = postprocess(empty, (60, 60, 40))
    assert not findings.tumour_suspected
    assert decision == TriageDecision.CLEAR
    assert confidence == 0.0
