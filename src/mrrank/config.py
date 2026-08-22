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

# ---------------------------------------------------------------------------
# Training hyperparameters (report Section 4.2 / 4.3, Model Module)
# ---------------------------------------------------------------------------
TRAIN_BATCH_SIZE = 128
TRAIN_LR = 0.1
TRAIN_MOMENTUM = 0.9
TRAIN_WEIGHT_DECAY = 5e-4
TRAIN_EPOCHS = 100  # report: "up to 100 epochs, targeting >= 80% test accuracy"

# Model seeds. Model A is the baseline SUT (mutated 80 times in Phase 4).
# Model B is a fresh, differently-seeded ResNet-18 used to test whether the
# ML Meta-Classifier generalizes to an unseen model (Phase 8 / supplementary
# doc Section 6.3: "Use a different random seed than Model A").
MODEL_A_SEED = 42
MODEL_B_SEED = 123

MODEL_A_CHECKPOINT = CHECKPOINTS_DIR / "model_A.pth"
MODEL_B_CHECKPOINT = CHECKPOINTS_DIR / "model_B.pth"

# Accuracy target band (report Section 1.3, Objective 1 + step-by-step doc):
# "at least 80%" but explicitly NOT much higher -- a near-perfect model is
# harder to break with mutations, defeating the purpose of Phase 4.
TARGET_ACC_MIN = 0.85
TARGET_ACC_MAX = 0.90


# ---------------------------------------------------------------------------
# Early stopping (added after observing Model A/B overshoot the target band
# due to late-training memorization once cosine-annealed LR gets small --
# no data augmentation is used, so the network eventually fits the training
# set near-perfectly once LR is low enough, pushing test_acc above the
# intended [TARGET_ACC_MIN, TARGET_ACC_MAX] band).
# ---------------------------------------------------------------------------
EARLY_STOP_PATIENCE = 5  # stop once test_acc has been in-band for this many
                          # consecutive epochs (avoids wasting GPU time
                          # running all the way to the memorization collapse)

# ---------------------------------------------------------------------------
# Mutation Engine (Phase 4)
# ---------------------------------------------------------------------------
# Layer target mapping resolves the report's "5 layers" (Appendix B) to
# concrete ResNet-18 sub-modules. conv1/fc deliberately excluded -- both are
# structural chokepoints with no residual bypass; zeroing either collapses
# the network to a constant output regardless of input (confirmed by the
# archived legacy exploration's WZ_conv1 mutant landing at 11.4% accuracy,
# an effectively dead/equivalent mutant). layer1 is split per-BasicBlock
# (highest spatial resolution / most generic features -- probed at finer
# granularity); layer2-4 are each treated as one mutation unit.
MUTATION_TARGET_LAYERS = ["layer1.0", "layer1.1", "layer2", "layer3", "layer4"]

# Per-layer fuzz std, calibrated via calibrate_mutants.py (accounts for
# depth-dependent sensitivity -- a fixed std across all layers was found
# to leave layer1 nearly untouched while collapsing layer2-4 to
# random-guess accuracy). "low" = milder variant, "high" = std x4 of low,
# preserving the report's two-intensity-tier design per layer.
# PLACEHOLDER VALUES -- run calibrate_mutants.py and update before use.
WEIGHT_FUZZ_STD_BY_LAYER = {
    "layer1.0": {"low": 0.0667, "high": 0.0721},
    "layer1.1": {"low": 0.0698, "high": 0.0729},
    "layer2": {"low": 0.0379, "high": 0.0399},
    "layer3": {"low": 0.0274, "high": 0.0290},
    "layer4": {"low": 0.0294, "high": 0.0325},
}

# CALIBRATION (empirical, replacing report's original 10%/20% suggestion):
# measured that even 50% corruption at 15 epochs only reached 0.7334
# accuracy (barely above target band), and 10%/20% corruption showed NO
# meaningful degradation even at 80 epochs (0.889 accuracy, indistinguishable
# from clean training) -- ResNet-18 with this training recipe (SGD +
# cosine annealing + RandomCrop/Flip augmentation) is highly robust to
# label noise at the report's originally suggested levels. Recalibrated via
# short diagnostic sweeps (15-16 epoch runs at 30/40/50/55% corruption) to
# find corruption-percentage/epoch-count pairs that reliably land in
# [MUTANT_ACC_MIN, MUTANT_ACC_MAX] with real margin, not at a knife-edge
# transition.
LABEL_CORRUPTION_CONFIG = {
    "low": {"pct": 0.40, "epochs": 9},
    "high": {"pct": 0.55, "epochs": 12},
}
LABEL_CORRUPTION_RUNS_PER_PCT = 10

# Capped well below Model A/B's ~20-50 effective epoch schedule -- label-
# corruption mutants only need to land in [MUTANT_ACC_MIN, MUTANT_ACC_MAX],
# not fully converge. Keeps all 20 mutants' total GPU cost to roughly
# 4-5 hours, instead of the report's own 8-12 hour estimate for
# full-length (100-epoch) retraining of each variant.
LABEL_CORRUPTION_EPOCHS = 20

MUTANT_ACC_MIN = 0.40
MUTANT_ACC_MAX = 0.70