# %% Cell 1 - Environment check (run first, confirm GPU is active)
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")

# %% Cell 2 - Clone the repo and install the package
# If MR-Prioritization is a PRIVATE repo, you'll need a GitHub PAT here:
#   !git clone https://<PAT>@github.com/sheryyll/MR-Prioritization.git
# If PUBLIC, this works as-is:
!git clone https://github.com/sheryyll/MR-Prioritization.git
%cd MR-Prioritization
!pip install -e . --no-deps -q
# --no-deps: Kaggle already has a CUDA-matched torch/torchvision/numpy/etc.
# installed. Reinstalling via our pyproject.toml dependencies could pull a
# CPU-only or mismatched build and break GPU access. We only want OUR
# package registered as importable; its deps are already satisfied.

# %% Cell 3 - Verify the package imports and CIFAR-10 loads correctly
from mrrank import config, data_module
print("SEED:", config.SEED)
train_loader = data_module.get_train_loader(batch_size=128, num_workers=2)
test_loader = data_module.get_test_loader(batch_size=128, num_workers=2)
images, labels = next(iter(train_loader))
print("Train batch shape:", images.shape, "labels shape:", labels.shape)

# %% Cell 4 - Train Model A (full 100-epoch run, expect ~30-60 min on T4 GPU)
!python -m mrrank.train --which A --epochs 100 --num-workers 2

# %% Cell 5 - Train Model B (different seed, for Phase 8 generalization test)
!python -m mrrank.train --which B --epochs 100 --num-workers 2

# %% Cell 6 - Copy checkpoints to Kaggle's persistent /kaggle/working/ output
# (so they survive after the session ends and can be downloaded)
!mkdir -p /kaggle/working/checkpoints
!cp outputs/checkpoints/model_A.pth /kaggle/working/checkpoints/
!cp outputs/checkpoints/model_B.pth /kaggle/working/checkpoints/
print("Checkpoints copied to /kaggle/working/checkpoints/ -- download from the Kaggle Output tab.")