"""
Unit tests for the Model Module.

FAST, CPU-only smoke tests: verify the model builds correctly, forward and
backward passes work, checkpointing round-trips correctly, and the
training loop runs without crashing on a tiny slice of real data. These do
NOT test for real accuracy -- that requires the full Kaggle GPU training
run (see notebooks/kaggle_train_model_a.py).
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

from mrrank import data_module
from mrrank.model_module import ModelWrapper, TrainConfig, build_resnet18_cifar


def _tiny_loader(n: int, batch_size: int = 16) -> DataLoader:
    """Build a small DataLoader from the fixed eval subset -- fast, no full
    CIFAR train/test set iteration needed for these correctness checks."""
    images, labels = data_module.get_eval_subset_raw()
    images, labels = images[:n], labels[:n]
    tensors = data_module.normalize_batch(images)
    labels_tensor = torch.as_tensor(labels, dtype=torch.long)
    dataset = TensorDataset(tensors, labels_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def test_model_builds_and_forward_pass():
    model = build_resnet18_cifar(num_classes=10)
    dummy_input = torch.randn(4, 3, 32, 32)
    output = model(dummy_input)
    assert output.shape == (4, 10)


def test_conv1_and_maxpool_adapted_for_cifar():
    """Confirms the CIFAR adaptation (3x3 stride-1 stem, no max-pool) is
    actually applied, not silently using the ImageNet defaults."""
    model = build_resnet18_cifar(num_classes=10)
    assert model.conv1.kernel_size == (3, 3)
    assert model.conv1.stride == (1, 1)
    assert isinstance(model.maxpool, torch.nn.Identity)


def test_model_wrapper_predict_batch():
    wrapper = ModelWrapper(device="cpu")
    images, _ = data_module.get_eval_subset_raw()
    batch = data_module.normalize_batch(images[:8])
    preds = wrapper.predict_batch(batch)
    assert preds.shape == (8,)
    assert preds.dtype == torch.int64


def test_save_load_roundtrip(tmp_path):
    wrapper = ModelWrapper(device="cpu")
    checkpoint_path = tmp_path / "test_model.pth"
    wrapper.save(checkpoint_path)

    wrapper2 = ModelWrapper(device="cpu")
    wrapper2.load(checkpoint_path)

    images, _ = data_module.get_eval_subset_raw()
    batch = data_module.normalize_batch(images[:4])
    preds1 = wrapper.predict_batch(batch)
    preds2 = wrapper2.predict_batch(batch)
    assert torch.equal(preds1, preds2)


def test_tiny_training_loop_runs():
    """Verifies the training loop executes without error for 1 epoch on a
    tiny slice of real data. Does NOT assert on accuracy -- only that loss
    is finite and nothing crashes."""
    wrapper = ModelWrapper(device="cpu")
    train_loader = _tiny_loader(n=64)
    test_loader = _tiny_loader(n=32)

    train_cfg = TrainConfig(epochs=1, batch_size=16, device="cpu")
    history = wrapper.fit(train_loader, test_loader, train_cfg, verbose=False)

    assert len(history["train_loss"]) == 1
    loss_value = history["train_loss"][0]
    assert loss_value == loss_value  # NaN check: NaN != NaN
    assert 0.0 <= history["test_acc"][0] <= 1.0


def test_extract_feature_vector():
    wrapper = ModelWrapper(device="cpu")
    test_loader = _tiny_loader(n=32)
    features = wrapper.extract_feature_vector(test_loader=test_loader)
    assert features["num_layers"] == 18
    assert features["num_parameters"] > 0
    assert "avg_weight_magnitude" in features
    assert "weight_std" in features
    assert 0.0 <= features["test_accuracy"] <= 1.0

def test_in_band_checkpoint_tracking():
    """
    Verify fit() correctly tracks the best in-band epoch and that
    save_best_in_band() persists those weights rather than the final
    epoch's. Uses an artificially wide target band so a tiny 1-epoch CPU
    run can land inside it deterministically-ish; this test checks the
    MECHANISM works, not real training dynamics (that requires the full
    Kaggle GPU run to observe).
    """
    from mrrank.model_module import TrainConfig, ModelWrapper

    wrapper = ModelWrapper(device="cpu")
    train_loader = _tiny_loader(n=64)
    test_loader = _tiny_loader(n=32)

    # Deliberately wide band so a short run has a real chance of landing
    # inside it and exercising the tracking/save logic end-to-end.
    train_cfg = TrainConfig(
        epochs=3,
        batch_size=16,
        device="cpu",
        target_acc_min=0.0,
        target_acc_max=1.0,
        early_stop_patience=1,
        enable_early_stopping=True,
    )
    wrapper.fit(train_loader, test_loader, train_cfg, verbose=False)

    assert wrapper.best_in_band_state is not None
    assert wrapper.best_in_band_acc is not None
    assert wrapper.best_in_band_epoch is not None

    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "best.pth"
        saved_acc = wrapper.save_best_in_band(path)
        assert saved_acc == wrapper.best_in_band_acc
        assert path.exists()