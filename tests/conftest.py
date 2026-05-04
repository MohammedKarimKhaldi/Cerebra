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
def synthetic_probs() -> torch.Tensor:
    """Fake (1, 4, 120, 120, 80) probability tensor with a synthetic tumour blob."""
    rng = np.random.default_rng(0)
    probs = np.zeros((1, 4, 120, 120, 80), dtype=np.float32)

    # Background dominates
    probs[0, 0] = 0.9

    # Inject a tumour blob in class 1 (necrotic core) in right-frontal region
    cx, cy, cz = 78, 78, 32
    for x in range(cx - 10, cx + 10):
        for y in range(cy - 10, cy + 10):
            for z in range(cz - 8, cz + 8):
                if 0 <= x < 120 and 0 <= y < 120 and 0 <= z < 80:
                    probs[0, 0, x, y, z] = 0.05
                    probs[0, 1, x, y, z] = 0.80
                    probs[0, 2, x, y, z] = 0.10
                    probs[0, 3, x, y, z] = 0.05

    return torch.from_numpy(probs)


@pytest.fixture(scope="session")
def synthetic_input_tensor(synthetic_study_dir: Path) -> torch.Tensor:
    """Preprocessed (1, 4, H, W, D) tensor from the synthetic study."""
    from cerebra.preprocess import load_study, preprocess
    paths = load_study(synthetic_study_dir)
    return preprocess(paths)
