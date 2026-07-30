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
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

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

        for epoch in range(train_cfg.epochs):
            start = time.time()
            train_loss = self._train_one_epoch(
                train_loader, optimizer, train_cfg.max_batches_per_epoch
            )
            test_acc = self.evaluate(test_loader)
            scheduler.step()

            history["train_loss"].append(train_loss)
            history["test_acc"].append(test_acc)

            if verbose:
                elapsed = time.time() - start
                print(
                    f"Epoch {epoch + 1}/{train_cfg.epochs} "
                    f"- loss: {train_loss:.4f} - test_acc: {test_acc:.4f} "
                    f"- {elapsed:.1f}s"
                )

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
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path: str | Path) -> None:
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, num_classes: int = 10, device: str | None = None
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