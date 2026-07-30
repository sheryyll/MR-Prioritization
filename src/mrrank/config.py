"""
Central configuration for the MR-Rank project.

Single source of truth for: random seed, filesystem paths, CIFAR-10
constants, and the fixed evaluation-subset size.

This lives INSIDE the `mrrank` package (not a project-root-only file) so
it is importable as `from mrrank import config` in any environment where
the package is installed -- local machine, Kaggle, Colab -- regardless of
the current working directory.

Paths default to locations relative to the installed package's source
tree, but can be overridden via environment variables. This matters on
Kaggle, where you may want data/outputs written to /kaggle/working/
instead of wherever the repo happens to be cloned.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42


def set_seed(seed: int = SEED) -> None:
    """
    Fix random seeds across python's `random`, numpy, and torch (if
    installed), so that dataset sampling, model initialization, and mutant
    generation are reproducible.

    This directly implements the report's Reproducibility Constraint
    (Section 3.3): "All stochastic operations including model
    initialization, mutant generation, and data sampling must use fixed
    random seeds."
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        # torch may not be installed in every context that imports config
        # (unlikely here, but keeps this module lightweight/defensive).
        pass


# ---------------------------------------------------------------------------
# Filesystem paths
# ---------------------------------------------------------------------------
# This file lives at: <project_root>/src/mrrank/config.py
#   _PACKAGE_DIR       = <project_root>/src/mrrank
#   _PACKAGE_DIR.parents[0] = <project_root>/src
#   _PACKAGE_DIR.parents[1] = <project_root>
_PACKAGE_DIR = Path(__file__).resolve().parent
_DEFAULT_PROJECT_ROOT = _PACKAGE_DIR.parents[1]

PROJECT_ROOT = Path(os.environ.get("MRRANK_PROJECT_ROOT", str(_DEFAULT_PROJECT_ROOT)))
DATA_DIR = Path(os.environ.get("MRRANK_DATA_DIR", str(PROJECT_ROOT / "data")))
OUTPUTS_DIR = Path(os.environ.get("MRRANK_OUTPUTS_DIR", str(PROJECT_ROOT / "outputs")))
MUTANTS_DIR = Path(os.environ.get("MRRANK_MUTANTS_DIR", str(PROJECT_ROOT / "mutants")))
CHECKPOINTS_DIR = Path(
    os.environ.get("MRRANK_CHECKPOINTS_DIR", str(PROJECT_ROOT / "outputs" / "checkpoints"))
)


# ---------------------------------------------------------------------------
# CIFAR-10 dataset constants
# ---------------------------------------------------------------------------
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
NUM_CLASSES = len(CIFAR10_CLASSES)

# Per-channel mean/std used for normalization.
# Source: report Section 4.3 ("Model Module"), standard CIFAR-10 statistics.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


# ---------------------------------------------------------------------------
# Fixed evaluation subset
# ---------------------------------------------------------------------------
# Report Section 3.3 (Constraints and Assumptions):
#   "The fixed 500-image stratified evaluation subset (50 images per class)
#    must remain constant across all experiments... No training image may
#    appear in this evaluation subset."
EVAL_IMAGES_PER_CLASS = 50
EVAL_SUBSET_SIZE = EVAL_IMAGES_PER_CLASS * NUM_CLASSES  # 500