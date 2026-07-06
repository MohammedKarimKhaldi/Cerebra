"""Shared pytest fixtures — builds the synthetic NIfTI study if not present."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "synthetic"


@pytest.fixture(scope="session")
def synthetic_study_dir() -> Path:
    """Return the path to the synthetic BraTS fixture, generating it if needed."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from make_synthetic_sample import generate   # type: ignore[import]

    generate(FIXTURE_DIR, seed=42)
    return FIXTURE_DIR


@pytest.fixture(scope="session")
def synthetic_prob_map() -> np.ndarray:
    """(H, W, D) calibrated P(tumour) map with a synthetic tumour blob (right-frontal)."""
    prob = np.zeros((120, 120, 80), dtype=np.float32)
    cx, cy, cz = 78, 20, 32   # right (x>60), frontal (y/120 < 1/3)
    prob[cx - 10:cx + 10, cy - 8:cy + 8, cz - 8:cz + 8] = 0.9
    return prob


@pytest.fixture(scope="session")
def synthetic_probs() -> torch.Tensor:
    """Legacy 4-class (1, 4, H, W, D) probability tensor (bundle-path coverage)."""
    probs = np.zeros((1, 4, 120, 120, 80), dtype=np.float32)
    probs[0, 0] = 0.9  # background
    cx, cy, cz = 78, 20, 32
    probs[0, 0, cx - 10:cx + 10, cy - 8:cy + 8, cz - 8:cz + 8] = 0.05
    probs[0, 1, cx - 10:cx + 10, cy - 8:cy + 8, cz - 8:cz + 8] = 0.80
    probs[0, 2, cx - 10:cx + 10, cy - 8:cy + 8, cz - 8:cz + 8] = 0.10
    probs[0, 3, cx - 10:cx + 10, cy - 8:cy + 8, cz - 8:cz + 8] = 0.05
    return torch.from_numpy(probs)


@pytest.fixture(scope="session")
def synthetic_input_tensor(synthetic_study_dir: Path) -> torch.Tensor:
    """Preprocessed (1, 4, H, W, D) tensor from the synthetic study."""
    from cerebra.preprocess import load_study, preprocess
    paths = load_study(synthetic_study_dir)
    return preprocess(paths)
