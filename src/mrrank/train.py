"""
Training entry point for Model A / Model B.

Full training run (Kaggle, GPU):
    python -m mrrank.train --which A --epochs 100 --num-workers 2
    python -m mrrank.train --which B --epochs 100 --num-workers 2

Local CPU smoke test (verifies the training loop runs end-to-end without
crashing -- NOT meant to reach real accuracy, just a correctness check):
    python -m mrrank.train --which A --epochs 1 --max-batches 5 --smoke-test
"""

from __future__ import annotations

import argparse

from mrrank import config, data_module
from mrrank.config import set_seed
from mrrank.model_module import ModelWrapper, TrainConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", choices=["A", "B"], required=True)
    parser.add_argument("--epochs", type=int, default=config.TRAIN_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.TRAIN_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=0,
                         help="DataLoader workers. Use 0 on Windows locally; "
                              "2-4 on Kaggle (Linux) for faster loading.")
    parser.add_argument("--max-batches", type=int, default=None,
                         help="Limit batches/epoch. Smoke testing ONLY -- "
                              "never set this for a real training run.")
    parser.add_argument("--smoke-test", action="store_true",
                         help="Skip the accuracy-target warning check.")
    args = parser.parse_args()

    seed = config.MODEL_A_SEED if args.which == "A" else config.MODEL_B_SEED
    set_seed(seed)

    train_loader = data_module.get_train_loader(
        batch_size=args.batch_size, num_workers=args.num_workers
    )
    test_loader = data_module.get_test_loader(
        batch_size=args.batch_size, num_workers=args.num_workers
    )

    wrapper = ModelWrapper()
    train_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_batches_per_epoch=args.max_batches,
    )
    print(f"Training Model {args.which} (seed={seed}) on device={train_cfg.device}")
    history = wrapper.fit(train_loader, test_loader, train_cfg)

    final_acc = history["test_acc"][-1]
    checkpoint_path = (
        config.MODEL_A_CHECKPOINT if args.which == "A" else config.MODEL_B_CHECKPOINT
    )
    wrapper.save(checkpoint_path)
    print(f"Saved Model {args.which} to {checkpoint_path} (final test_acc={final_acc:.4f})")

    if not args.smoke_test:
        if not (config.TARGET_ACC_MIN <= final_acc <= config.TARGET_ACC_MAX + 0.05):
            print(
                f"WARNING: final accuracy {final_acc:.4f} is outside the target "
                f"band [{config.TARGET_ACC_MIN}, {config.TARGET_ACC_MAX}]. "
                f"Consider adjusting epochs or the LR schedule."
            )


if __name__ == "__main__":
    main()