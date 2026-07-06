"""Post-process a tumour probability map into triage metrics.

All public functions are pure (no I/O, no global state) so they are easy to unit test.

Input contract: a single-channel calibrated P(tumour) map of shape (H, W, D), as
produced by `inference.run_inference`. Because the probability is temperature-scaled,
thresholds and the reported confidence are meaningful, not just monotonic scores.

Triage rule (POC):
  FLAG  if tumour_volume_mm3 >= 250 AND study_tumour_probability >= 0.5
  CLEAR otherwise

Confidence = calibrated study-level tumour probability (see `study_tumour_probability`).

Anatomical region: bounding-box heuristic splitting the volume into left/right ×
frontal/temporal/occipital thirds. This is a placeholder — see TODO below.

# TODO(real-product): replace bounding-box region mapping with atlas-based
#   registration (e.g., MNI152 via ANTs/SimpleITK) for clinically valid localisation.
"""

from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi
import torch

from cerebra.schemas import Findings, TriageDecision

# Triage thresholds
VOLUME_THRESHOLD_MM3 = 250.0
PROB_THRESHOLD = 0.5

# Voxel volume for 1 mm³ isotropic images (preprocessing resamples to 1,1,1)
VOXEL_VOLUME_MM3 = 1.0


def to_prob_map(probs: np.ndarray | torch.Tensor) -> np.ndarray:
    """Coerce model output into a single-channel (H, W, D) float32 P(tumour) map.

    Accepts either an already-collapsed (H, W, D) map (our binary model, the normal
    case) or a legacy 4-class (1, 4, H, W, D) tensor (bundle path), collapsing the
    latter to P(any tumour) = 1 - P(background).
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    probs = np.asarray(probs)
    if probs.ndim == 5:  # (1, C, H, W, D)
        return (1.0 - probs[0, 0]).astype(np.float32)
    if probs.ndim == 4:  # (C, H, W, D)
        return (1.0 - probs[0]).astype(np.float32)
    return probs.astype(np.float32)


def threshold_and_label(prob_map: np.ndarray, threshold: float = PROB_THRESHOLD) -> np.ndarray:
    """Threshold probability map and label connected components.

    Returns an int32 label map (0 = background, 1+ = distinct tumour regions).
    """
    binary = prob_map >= threshold
    labelled, _ = ndi.label(binary)
    return labelled.astype(np.int32)


def largest_component(label_map: np.ndarray) -> np.ndarray:
    """Return a binary mask of the single largest connected component.

    Returns an all-zero mask if no components exist.
    """
    if label_map.max() == 0:
        return np.zeros_like(label_map, dtype=bool)
    counts = ndi.sum(np.ones_like(label_map), label_map, range(1, label_map.max() + 1))
    best_label = int(np.argmax(counts)) + 1  # labels are 1-indexed
    return label_map == best_label


def compute_volume_mm3(mask: np.ndarray, voxel_volume: float = VOXEL_VOLUME_MM3) -> float:
    """Return tumour volume in mm³ for a binary mask with given voxel size."""
    return float(mask.sum()) * voxel_volume


def compute_max_diameter_mm(mask: np.ndarray) -> float:
    """Approximate max axial diameter as the longest bounding-box edge in mm.

    Uses the bounding box of the largest connected component rather than a
    full distance transform — sufficient for POC triage.
    # TODO(real-product): compute true max chord length via rotating callipers
    #   on the 2-D axial cross-section with the largest area.
    """
    if not mask.any():
        return 0.0
    coords = np.argwhere(mask)
    extents = coords.max(axis=0) - coords.min(axis=0) + 1  # +1 for inclusive
    return float(extents[:2].max())  # largest of x, y extent (axial plane)


def map_anatomical_region(mask: np.ndarray, volume_shape: tuple[int, int, int]) -> str:
    """Map the tumour centroid to a coarse anatomical region name.

    Splits the volume into left/right (x-axis) and frontal/temporal/occipital
    (y-axis thirds).  Z-axis (superior/inferior) is not classified in the POC.

    # TODO(real-product): register to MNI152 atlas and use Brodmann area labels
    #   or AAL parcellation for clinically valid localisation.
    """
    if not mask.any():
        return "unknown"

    centroid = np.array(ndi.center_of_mass(mask))
    h, w, d = volume_shape

    side = "left" if centroid[0] < h / 2 else "right"

    y_frac = centroid[1] / w
    if y_frac < 1 / 3:
        lobe = "frontal"
    elif y_frac < 2 / 3:
        lobe = "temporal"
    else:
        lobe = "occipital"

    return f"{side} {lobe}"


def study_tumour_probability(prob_map: np.ndarray, mask: np.ndarray) -> float:
    """Calibrated study-level P(tumour), clipped to [0, 1].

    Rather than a single hottest voxel (noisy) or the whole-region mean (diluted by
    partial-volume edges), we take the mean of the most-confident voxels inside the
    largest component. This is a stable summary of "how tumour-like is the flagged
    region", and because the map is temperature-calibrated it reads as a genuine
    probability. Empty mask → 0.0.
    """
    if not mask.any():
        return 0.0
    vals = prob_map[mask]
    # top decile (at least one voxel) of the component's probabilities
    k = max(1, int(vals.size * 0.1))
    top = np.sort(vals)[-k:]
    return float(np.clip(top.mean(), 0.0, 1.0))


def make_triage_decision(volume_mm3: float, probability: float) -> TriageDecision:
    """Apply the POC triage rule: FLAG if volume ≥ 250 mm³ AND probability ≥ 0.5."""
    if volume_mm3 >= VOLUME_THRESHOLD_MM3 and probability >= PROB_THRESHOLD:
        return TriageDecision.FLAG
    return TriageDecision.CLEAR


def postprocess(
    prob_map: np.ndarray | torch.Tensor,
    volume_shape: tuple[int, int, int],
) -> tuple[Findings, TriageDecision, float, np.ndarray]:
    """Run the full postprocessing pipeline on a calibrated tumour probability map.

    Args:
        prob_map: (H, W, D) calibrated P(tumour), or a legacy 4-class tensor.
        volume_shape: (H, W, D) of the preprocessed input (used for region mapping).

    Returns:
        findings, triage_decision, confidence, largest_component_mask
    """
    prob_map = to_prob_map(prob_map)
    label_map = threshold_and_label(prob_map)
    component = largest_component(label_map)

    volume_mm3 = compute_volume_mm3(component)
    diameter_mm = compute_max_diameter_mm(component)
    region = map_anatomical_region(component, volume_shape)
    probability = study_tumour_probability(prob_map, component)
    decision = make_triage_decision(volume_mm3, probability)

    findings = Findings(
        tumour_suspected=bool(component.any()),
        tumour_volume_mm3=volume_mm3,
        anatomical_region=region,
        max_diameter_mm=diameter_mm,
    )
    return findings, decision, probability, component
