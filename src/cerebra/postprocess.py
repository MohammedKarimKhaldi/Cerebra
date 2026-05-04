"""Post-process the model probability map into triage metrics.

All public functions are pure (no I/O, no global state) so they are easy to unit test.

Triage rule (POC):
  FLAG  if tumour_volume_mm3 >= 250 AND max_segmentation_probability >= 0.5
  CLEAR otherwise

Confidence = max softmax probability inside the largest connected component,
             clipped to [0, 1].

Anatomical region: bounding-box heuristic splitting the volume into left/right ×
frontal/parietal/occipital thirds. This is a placeholder — see TODO below.

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

# Voxel volume for 1 mm³ isotropic images (TARGET_SPACING = 1,1,1)
VOXEL_VOLUME_MM3 = 1.0


def tumour_probability_map(probs: torch.Tensor) -> np.ndarray:
    """Collapse 4-class prob map to a single 'tumour present' probability.

    probs: (1, 4, H, W, D)  — class 0 is background; 1–3 are tumour sub-regions.
    Returns (H, W, D) float32 array: P(any tumour class).
    """
    tumour_prob = 1.0 - probs[0, 0].numpy()  # P(not background) == P(any tumour)
    return tumour_prob.astype(np.float32)


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
    return (label_map == best_label)


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

    # Left / Right along x-axis
    side = "left" if centroid[0] < h / 2 else "right"

    # Frontal / Temporal / Occipital thirds along y-axis
    y_frac = centroid[1] / w
    if y_frac < 1 / 3:
        lobe = "frontal"
    elif y_frac < 2 / 3:
        lobe = "temporal"
    else:
        lobe = "occipital"

    return f"{side} {lobe}"


def compute_confidence(prob_map: np.ndarray, mask: np.ndarray) -> float:
    """Return max segmentation probability inside the component, clipped to [0, 1]."""
    if not mask.any():
        return 0.0
    return float(np.clip(prob_map[mask].max(), 0.0, 1.0))


def make_triage_decision(volume_mm3: float, confidence: float) -> TriageDecision:
    """Apply the POC triage rule: FLAG if volume ≥ 250 mm³ AND confidence ≥ 0.5."""
    if volume_mm3 >= VOLUME_THRESHOLD_MM3 and confidence >= PROB_THRESHOLD:
        return TriageDecision.FLAG
    return TriageDecision.CLEAR


def postprocess(
    probs: torch.Tensor,
    volume_shape: tuple[int, int, int],
) -> tuple[Findings, TriageDecision, float, np.ndarray]:
    """Run the full postprocessing pipeline on the model output.

    Args:
        probs: (1, 4, H, W, D) softmax probability tensor from the model.
        volume_shape: (H, W, D) of the preprocessed input (used for region mapping).

    Returns:
        findings, triage_decision, confidence, largest_component_mask
    """
    prob_map = tumour_probability_map(probs)
    label_map = threshold_and_label(prob_map)
    component = largest_component(label_map)

    volume_mm3 = compute_volume_mm3(component)
    diameter_mm = compute_max_diameter_mm(component)
    region = map_anatomical_region(component, volume_shape)
    confidence = compute_confidence(prob_map, component)
    decision = make_triage_decision(volume_mm3, confidence)

    findings = Findings(
        tumour_suspected=component.any(),
        tumour_volume_mm3=volume_mm3,
        anatomical_region=region,
        max_diameter_mm=diameter_mm,
    )
    return findings, decision, confidence, component
