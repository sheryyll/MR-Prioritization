"""
Training entry point for Model A / Model B.

Full training run (Kaggle, GPU):
    python -m mrrank.train --which A --epochs 100 --num-workers 2
    python -m mrrank.train --which B --epochs 100 --num-workers 2

By default, saves the BEST IN-BAND checkpoint (the epoch whose test_acc
fell within [TARGET_ACC_MIN, TARGET_ACC_MAX] with the highest accuracy),
not necessarily the final epoch's weights -- see model_module.py's
early-stopping / in-band tracking logic. Training also stops early once
the target band has been reached and held stable, rather than always
running the full epoch count.

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
                         help="Skip the accuracy-target warning check and "
                              "disable early stopping/in-band checkpointing "
                              "(smoke tests are too short for either to be "
                              "meaningful).")
    parser.add_argument("--no-early-stop", action="store_true",
                         help="Disable early stopping; always train the full "
                              "--epochs count. The in-band checkpoint is still "
                              "tracked and saved unless --smoke-test is set.")
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
        enable_early_stopping=(not args.smoke_test and not args.no_early_stop),
    )
    print(f"Training Model {args.which} (seed={seed}) on device={train_cfg.device}")
    history = wrapper.fit(train_loader, test_loader, train_cfg)

    final_acc = history["test_acc"][-1]
    checkpoint_path = (
        config.MODEL_A_CHECKPOINT if args.which == "A" else config.MODEL_B_CHECKPOINT
    )

    if args.smoke_test:
        wrapper.save(checkpoint_path)
        print(f"[smoke-test] Saved final-epoch weights to {checkpoint_path} "
              f"(final test_acc={final_acc:.4f})")
        return

    saved_acc = wrapper.save_best_in_band(checkpoint_path)
    if saved_acc >= 0:
        print(
            f"Saved BEST IN-BAND checkpoint (epoch {wrapper.best_in_band_epoch}, "
            f"test_acc={saved_acc:.4f}) for Model {args.which} to {checkpoint_path}"
        )
    else:
        print(
            f"WARNING: Model {args.which} never landed in the target band "
            f"[{config.TARGET_ACC_MIN}, {config.TARGET_ACC_MAX}] during training. "
            f"Saved final-epoch weights instead (test_acc={final_acc:.4f}) -- "
            f"this checkpoint likely needs manual review before use in later phases."
        )


if __name__ == "__main__":
    main()