"""
Model Module for MR-Rank.

Encapsulates the ResNet-18 architecture adapted for CIFAR-10's 32x32
images (report Section 4.3 "Model Module", residual formula Eq. 4.5), plus
training, evaluation, checkpointing, and feature-extraction utilities used
throughout the rest of the pipeline:
    - Phase 2 (this phase): train Model A (baseline SUT) and Model B
      (unseen model, different seed).
    - Phase 4 (Mutation Engine): loads Model A's state_dict and mutates its
      tensors directly.
    - Phase 6-8 (ML Meta-Classifier): extract_feature_vector() describes a
      model's "fingerprint" WITHOUT mutating it, per the supplementary
      doc's Phase 2 ("Extract Features -- This is the Key Step").

NOTE on early stopping / in-band checkpointing (added after observing real
training runs): with no data augmentation, ResNet-18 on CIFAR-10 with a
100-epoch cosine-annealed schedule eventually memorizes the training set
once the LR gets small (loss collapses toward ~0), pushing test accuracy
well above the report's intended [0.80, 0.85] target band. fit() now
tracks the BEST epoch whose test_acc falls within [target_acc_min,
target_acc_max] and can stop early once that plateau is confirmed stable,
rather than always training to completion and saving overfit final
weights.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import resnet18

from mrrank import config


def build_resnet18_cifar(num_classes: int = 10) -> nn.Module:
    """
    Build a ResNet-18 adapted for CIFAR-10's 32x32 input.

    torchvision's stock resnet18 targets ImageNet's 224x224 images: its
    stem is a 7x7 stride-2 conv followed by a stride-2 max-pool, which
    would shrink a 32x32 CIFAR image to 4x4 before the first residual
    block even runs -- destroying almost all spatial detail.

    Standard practice for CIFAR-scale ResNets (and required for the
    architecture to be meaningful at 32x32 at all) is to:
      - replace the 7x7 stride-2 stem conv with a 3x3 stride-1 conv
      - remove the initial max-pool layer entirely (nn.Identity)
    This preserves full 32x32 resolution into the first residual block.
    """
    model = resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


@dataclass
class TrainConfig:
    epochs: int = config.TRAIN_EPOCHS
    lr: float = config.TRAIN_LR
    momentum: float = config.TRAIN_MOMENTUM
    weight_decay: float = config.TRAIN_WEIGHT_DECAY
    batch_size: int = config.TRAIN_BATCH_SIZE
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Limits batches/epoch -- ONLY for fast local smoke tests, never set
    # this on Kaggle for a real training run.
    max_batches_per_epoch: int | None = None
    # In-band checkpointing / early stopping (see module docstring)
    target_acc_min: float = config.TARGET_ACC_MIN
    target_acc_max: float = config.TARGET_ACC_MAX
    early_stop_patience: int = config.EARLY_STOP_PATIENCE
    enable_early_stopping: bool = True


class ModelWrapper:
    """
    Thin, consistent wrapper around a ResNet-18-for-CIFAR model: training,
    evaluation, checkpointing, inference, and feature extraction all live
    behind one interface -- matches the report's "unified ModelWrapper
    class" (Section 4.3).
    """

    def __init__(self, num_classes: int = 10, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_resnet18_cifar(num_classes).to(self.device)
        # Populated during fit() if a target-band epoch is found.
        self.best_in_band_state: dict | None = None
        self.best_in_band_acc: float | None = None
        self.best_in_band_epoch: int | None = None

    # ------------------------------------------------------------------
    # Training / evaluation
    # ------------------------------------------------------------------
    def fit(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        train_cfg: TrainConfig,
        verbose: bool = True,
    ) -> dict:
        """
        Train per report Section 4.2: SGD with momentum + weight decay,
        cosine-annealed LR schedule, up to `train_cfg.epochs`. Returns a
        history dict with per-epoch train loss and test accuracy.

        Also tracks the best epoch whose test_acc falls within
        [train_cfg.target_acc_min, train_cfg.target_acc_max] (stored on
        self.best_in_band_state / self.best_in_band_acc /
        self.best_in_band_epoch). If enable_early_stopping is True,
        training stops once test_acc has stayed in-band for
        early_stop_patience consecutive epochs -- this avoids burning
        GPU time training all the way to the point where an unaugmented
        CIFAR ResNet-18 starts memorizing the training set and drifting
        above the target band.
        """
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=train_cfg.lr,
            momentum=train_cfg.momentum,
            weight_decay=train_cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=train_cfg.epochs
        )

        history = {"train_loss": [], "test_acc": []}
        consecutive_in_band = 0

        for epoch in range(train_cfg.epochs):
            start = time.time()
            train_loss = self._train_one_epoch(
                train_loader, optimizer, train_cfg.max_batches_per_epoch
            )
            test_acc = self.evaluate(test_loader)
            scheduler.step()

            history["train_loss"].append(train_loss)
            history["test_acc"].append(test_acc)

            in_band = train_cfg.target_acc_min <= test_acc <= train_cfg.target_acc_max
            if in_band:
                consecutive_in_band += 1
                if self.best_in_band_acc is None or test_acc > self.best_in_band_acc:
                    self.best_in_band_acc = test_acc
                    self.best_in_band_epoch = epoch + 1
                    self.best_in_band_state = copy.deepcopy(self.model.state_dict())
            else:
                consecutive_in_band = 0

            if verbose:
                elapsed = time.time() - start
                band_marker = " [IN-BAND]" if in_band else ""
                print(
                    f"Epoch {epoch + 1}/{train_cfg.epochs} "
                    f"- loss: {train_loss:.4f} - test_acc: {test_acc:.4f} "
                    f"- {elapsed:.1f}s{band_marker}"
                )

            if (
                train_cfg.enable_early_stopping
                and consecutive_in_band >= train_cfg.early_stop_patience
            ):
                if verbose:
                    print(
                        f"Early stopping at epoch {epoch + 1}: test_acc has been "
                        f"in target band [{train_cfg.target_acc_min}, "
                        f"{train_cfg.target_acc_max}] for "
                        f"{train_cfg.early_stop_patience} consecutive epochs. "
                        f"Best in-band epoch: {self.best_in_band_epoch} "
                        f"(test_acc={self.best_in_band_acc:.4f})"
                    )
                break

        return history

    def _train_one_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        max_batches: int | None = None,
    ) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            images, labels = images.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader) -> float:
        """Return top-1 test accuracy as a fraction in [0, 1]."""
        self.model.eval()
        correct, total = 0, 0
        for images, labels in test_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        return correct / total if total > 0 else 0.0

    @torch.no_grad()
    def predict_batch(self, images: torch.Tensor) -> torch.Tensor:
        """
        Run inference on a batch of already-normalized image tensors
        (N, C, H, W). Returns predicted class indices, shape (N,). This is
        the interface the MR Engine (Phase 3) and Evaluation Module
        (Phase 5) will call to compare predictions on original vs.
        transformed images.
        """
        self.model.eval()
        images = images.to(self.device)
        outputs = self.model(images)
        return outputs.argmax(dim=1).cpu()

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def save(self, path) -> None:
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def save_best_in_band(self, path, fallback_to_final: bool = True) -> float:
        """
        Save the best in-band checkpoint tracked during fit(), if one was
        found. Returns the accuracy that was saved.

        If fit() never encountered an epoch within the target band (edge
        case -- e.g. accuracy jumped straight from below-band to
        above-band without landing in between), falls back to saving the
        model's CURRENT weights (whatever state it's in after fit()
        returned) and prints a warning, unless fallback_to_final=False,
        in which case this raises instead.
        """
        if self.best_in_band_state is not None:
            from pathlib import Path
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.best_in_band_state, path)
            return self.best_in_band_acc

        if not fallback_to_final:
            raise RuntimeError(
                "No in-band checkpoint was recorded during training, and "
                "fallback_to_final=False."
            )

        print(
            "WARNING: no epoch fell within the target accuracy band during "
            "training. Saving the model's current (final) weights instead -- "
            "verify this checkpoint's accuracy manually before using it."
        )
        self.save(path)
        return -1.0  # signals "no in-band value available" to the caller

    def load(self, path) -> None:
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)

    @classmethod
    def from_checkpoint(
        cls, path, num_classes: int = 10, device: str | None = None
    ) -> "ModelWrapper":
        wrapper = cls(num_classes=num_classes, device=device)
        wrapper.load(path)
        return wrapper

    # ------------------------------------------------------------------
    # Feature extraction (for ML Meta-Classifier, Phase 6-8)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def extract_feature_vector(self, test_loader: DataLoader | None = None) -> dict:
        """
        Extract a lightweight "fingerprint" describing this model WITHOUT
        mutating it -- per the supplementary doc's Phase 2 ("Extract
        Features -- This is the Key Step"). Used to build meta-classifier
        input rows for a new, unseen model (Model B) without running full
        mutation testing on it.
        """
        all_weights = torch.cat([p.detach().flatten() for p in self.model.parameters()])
        features = {
            "num_layers": 18,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "avg_weight_magnitude": all_weights.abs().mean().item(),
            "weight_std": all_weights.std().item(),
        }
        if test_loader is not None:
            features["test_accuracy"] = self.evaluate(test_loader)
        return features