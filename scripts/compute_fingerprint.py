"""
compute_fingerprint.py — generate the model fingerprint JSON used by the
meta-classifier inference tab in the Streamlit app.

Run locally (requires torch + torchvision + the CIFAR-10 eval subset):

    python scripts/compute_fingerprint.py \\
        --checkpoint outputs/checkpoints/model_B.pth \\
        [--out outputs/model_b_fingerprint.json]

The script extracts three scalar features from the checkpoint:
  - avg_weight_magnitude  (mean of |all weights|)
  - weight_std            (std of all weights)
  - test_accuracy         (top-1 on the 500-image eval subset)

The resulting JSON can be uploaded in the Streamlit app's
"Rank a new model" tab, OR saved to outputs/model_b_fingerprint.json
so that scripts/export_artifacts.py can use it when regenerating artifacts.

Constraints
-----------
* Load checkpoints with weights_only=True for security.
* Only state_dict format checkpoints are supported.
* No training is performed — this is inference only.
* torch and torchvision are ONLY imported here, not in the Streamlit app.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract model fingerprint for MR-Rank meta-classifier.")
    p.add_argument(
        "--checkpoint",
        default="outputs/checkpoints/model_B.pth",
        help="Path to a ResNet-18 state_dict checkpoint (.pth).",
    )
    p.add_argument(
        "--out",
        default="outputs/model_b_fingerprint.json",
        help="Output JSON path (default: outputs/model_b_fingerprint.json).",
    )
    p.add_argument(
        "--data-dir",
        default="data",
        help="Root data directory containing CIFAR-10 batches (default: data/).",
    )
    p.add_argument(
        "--eval-indices",
        default="outputs/eval_subset_indices.npy",
        help="Path to eval_subset_indices.npy (500-image fixed subset).",
    )
    p.add_argument(
        "--no-accuracy",
        action="store_true",
        help="Skip accuracy computation (useful if CIFAR-10 data is not available locally).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    try:
        import torch
        import torch.nn as nn
        import numpy as np
    except ImportError as e:
        sys.exit(
            f"ERROR: {e}\n"
            "Install torch and torchvision before running this script.\n"
            "They are intentionally excluded from the Streamlit app requirements."
        )

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        sys.exit(f"ERROR: checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)

    if not isinstance(state_dict, dict):
        sys.exit(
            "ERROR: checkpoint must be a plain state_dict mapping (dict), "
            "not a full model object or training checkpoint."
        )

    # Compute weight statistics from the state_dict tensors
    all_weights = torch.cat([v.float().flatten() for v in state_dict.values() if v.dtype.is_floating_point])
    avg_weight_magnitude = float(all_weights.abs().mean().item())
    weight_std = float(all_weights.std().item())
    print(f"  num_parameters     : {len(all_weights):,}")
    print(f"  avg_weight_magnitude: {avg_weight_magnitude:.6f}")
    print(f"  weight_std         : {weight_std:.6f}")

    # Compute test accuracy on the fixed 500-image eval subset
    test_accuracy: float | None = None
    if not args.no_accuracy:
        try:
            from torchvision import datasets, transforms

            eval_indices_path = Path(args.eval_indices)
            data_dir = Path(args.data_dir)

            if not eval_indices_path.exists():
                print(f"  WARNING: eval_subset_indices.npy not found at {eval_indices_path}. "
                      "Skipping accuracy computation.")
            elif not data_dir.exists():
                print(f"  WARNING: data directory not found at {data_dir}. "
                      "Skipping accuracy computation.")
            else:
                import numpy as np

                # Build the same CIFAR-10 ResNet-18 architecture used in this project
                from torchvision.models import resnet18

                model = resnet18(weights=None, num_classes=10)
                model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                model.maxpool = nn.Identity()
                model.load_state_dict(state_dict)
                model.eval()

                CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
                CIFAR10_STD = (0.2023, 0.1994, 0.2010)
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
                ])
                full_test = datasets.CIFAR10(root=str(data_dir), train=False,
                                             download=False, transform=transform)
                indices = np.load(str(eval_indices_path))
                subset = torch.utils.data.Subset(full_test, indices.tolist())
                loader = torch.utils.data.DataLoader(subset, batch_size=256,
                                                     shuffle=False, num_workers=0)

                correct, total = 0, 0
                with torch.no_grad():
                    for images, labels in loader:
                        preds = model(images).argmax(dim=1)
                        correct += (preds == labels).sum().item()
                        total += labels.size(0)
                test_accuracy = correct / total if total > 0 else 0.0
                print(f"  test_accuracy (500-image eval subset): {test_accuracy:.4f}")

        except Exception as e:
            print(f"  WARNING: accuracy computation failed ({e}). "
                  "Proceeding without test_accuracy.")

    fingerprint = {
        "avg_weight_magnitude": avg_weight_magnitude,
        "weight_std": weight_std,
        "test_accuracy": test_accuracy if test_accuracy is not None else 0.0,
    }

    if test_accuracy is None:
        fingerprint["_test_accuracy_note"] = (
            "test_accuracy was not computed (CIFAR-10 data or eval indices not available). "
            "Set to 0.0 as a placeholder. The meta-classifier will still rank MRs, "
            "but accuracy-dependent patterns will be suppressed."
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(fingerprint, indent=2))
    print(f"\nFingerprint written to: {out_path}")
    print(json.dumps({k: v for k, v in fingerprint.items() if not k.startswith("_")}, indent=2))


if __name__ == "__main__":
    main()
