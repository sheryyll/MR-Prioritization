"""
mutation_engine.py — MR-Rank Project (Member 2: Mutation Engineer)
===================================================================
Loads the base ResNet18 trained in the original notebook and generates
~60 mutants using 4 weight-level operators across 8 layer groups.

Outputs
-------
./checkpoints/mutants/      — one .pth per mutant (state_dict only)
./results/kill_matrix.pkl   — dict[mr_id][mutant_id] = True/False
./results/kill_matrix.csv   — human-readable summary
./results/mutation_summary.json — FDR per mutant, metadata

Usage
-----
    python mutation_engine.py

Then in prioritization/scoring.py:
    import pickle
    with open('./results/kill_matrix.pkl', 'rb') as f:
        kill_matrix = pickle.load(f)
"""

import os
import copy
import json
import pickle
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F

# ── Directories ────────────────────────────────────────────────────────────────
os.makedirs('./checkpoints/mutants', exist_ok=True)
os.makedirs('./results',             exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device : {device}')

NUM_CLASSES = 10
CLASSES     = ['airplane','automobile','bird','cat','deer',
                'dog','frog','horse','ship','truck']

# ── 1. Rebuild architecture (must match original notebook exactly) ─────────────
def build_original_model() -> nn.Module:
    """
    Recreates the exact architecture used in the original notebook:
      - conv1  : 3×3 kernel, stride 1, padding 1
      - maxpool: replaced with Identity
      - fc     : 512 → 10
    """
    model = resnet18(weights=None)
    model.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc      = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def load_base_model(path: str = 'resnet18_cifar10.pth') -> nn.Module:
    """Load the base model saved by the original notebook (state_dict only)."""
    model = build_original_model().to(device)
    state = torch.load(path, map_location=device)
    # Handle both raw state_dict and checkpoint dict formats
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict(state)
    model.eval()
    print(f'[load_base_model] Loaded from {path}')
    return model


# ── 2. Test data (1000 samples — fast but statistically valid) ─────────────────
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2023, 0.1994, 0.2010]),
])

full_test = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)

# Fixed 1000-sample subset — same indices every run (reproducible kill matrix)
torch.manual_seed(42)
subset_indices = torch.randperm(len(full_test))[:1000].tolist()
test_subset    = Subset(full_test, subset_indices)
test_loader    = DataLoader(test_subset, batch_size=256,
                            shuffle=False, num_workers=0)

print(f'Mutation test subset : {len(test_subset)} samples')


# ── 3. Inference helper ────────────────────────────────────────────────────────
def get_predictions(model: nn.Module) -> np.ndarray:
    """Returns predicted class indices for the full test subset (shape: [1000])."""
    model.eval()
    all_preds = []
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            preds  = model(images).argmax(1).cpu().numpy()
            all_preds.append(preds)
    return np.concatenate(all_preds)


def model_accuracy(model: nn.Module) -> float:
    """Quick accuracy check on the 1000-sample subset."""
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(1) == labels).sum().item()
    return 100.0 * correct / len(test_subset)


# ── 4. Mutation Operators ──────────────────────────────────────────────────────

def mutate_weight_fuzzing(base: nn.Module, layer_keyword: str,
                          noise_std: float) -> nn.Module:
    """Add Gaussian noise to weights of layers whose name contains layer_keyword."""
    mutant = copy.deepcopy(base)
    for name, param in mutant.named_parameters():
        if layer_keyword in name and 'weight' in name:
            param.data.add_(torch.randn_like(param) * noise_std)
    return mutant


def mutate_weight_negation(base: nn.Module, layer_keyword: str) -> nn.Module:
    """Negate all weights in matching layers — strong behavioural change."""
    mutant = copy.deepcopy(base)
    for name, param in mutant.named_parameters():
        if layer_keyword in name and 'weight' in name:
            param.data.mul_(-1)
    return mutant


def mutate_weight_zeroing(base: nn.Module, layer_keyword: str) -> nn.Module:
    """Zero out all weights — kills the layer entirely."""
    mutant = copy.deepcopy(base)
    for name, param in mutant.named_parameters():
        if layer_keyword in name and 'weight' in name:
            param.data.zero_()
    return mutant


def mutate_weight_scaling(base: nn.Module, layer_keyword: str,
                          scale: float) -> nn.Module:
    """Scale weights by a constant factor — moderate distortion."""
    mutant = copy.deepcopy(base)
    for name, param in mutant.named_parameters():
        if layer_keyword in name and 'weight' in name:
            param.data.mul_(scale)
    return mutant


def mutate_dropout_cripple(base: nn.Module, new_rate: float = 0.8) -> nn.Module:
    """Increase all Dropout rates to near-1 — cripples inference."""
    mutant = copy.deepcopy(base)
    for module in mutant.modules():
        if isinstance(module, nn.Dropout):
            module.p = new_rate
    mutant.eval()   # ensure dropout is inactive in eval (still alters trained behaviour)
    return mutant


def mutate_bn_reset(base: nn.Module, layer_keyword: str) -> nn.Module:
    """Reset BatchNorm running stats in matching layers — disrupts normalisation."""
    mutant = copy.deepcopy(base)
    for name, module in mutant.named_modules():
        if layer_keyword in name and isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
            module.running_mean.zero_()
            module.running_var.fill_(1)
    return mutant


# ── 5. Define all ~60 mutants ─────────────────────────────────────────────────
#
# ResNet18 layer keywords (8 groups):
#   'conv1'   — stem conv
#   'layer1'  — residual block 1 (2 basic blocks)
#   'layer2'  — residual block 2
#   'layer3'  — residual block 3
#   'layer4'  — residual block 4 (deepest)
#   'bn1'     — stem batch norm
#   'downsample' — skip connection projections
#   'fc'      — classification head
#
LAYER_GROUPS = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'bn1', 'downsample', 'fc']

def define_mutant_specs() -> list[dict]:
    """
    Returns a list of mutant specification dicts.
    Each dict has: id, description, operator, kwargs
    Total: 8 layers × 4 weight ops (32) + 8 layers × 1 BN reset (8)
           + 8 layers × 2 fuzz strengths (16) + 4 global ops (4) = ~60
    """
    specs = []

    # ── A) Weight fuzzing — mild (noise_std=0.05) ──────────────────────────────
    for layer in LAYER_GROUPS:
        specs.append({
            'id':          f'WF_mild_{layer}',
            'description': f'Weight fuzz std=0.05 on {layer}',
            'operator':    'weight_fuzz',
            'layer':       layer,
            'noise_std':   0.05,
        })

    # ── B) Weight fuzzing — strong (noise_std=0.30) ────────────────────────────
    for layer in LAYER_GROUPS:
        specs.append({
            'id':          f'WF_strong_{layer}',
            'description': f'Weight fuzz std=0.30 on {layer}',
            'operator':    'weight_fuzz',
            'layer':       layer,
            'noise_std':   0.30,
        })

    # ── C) Weight negation ────────────────────────────────────────────────────
    for layer in LAYER_GROUPS:
        specs.append({
            'id':          f'WN_{layer}',
            'description': f'Weight negation on {layer}',
            'operator':    'weight_negate',
            'layer':       layer,
        })

    # ── D) Weight zeroing ─────────────────────────────────────────────────────
    for layer in LAYER_GROUPS:
        specs.append({
            'id':          f'WZ_{layer}',
            'description': f'Weight zeroing on {layer}',
            'operator':    'weight_zero',
            'layer':       layer,
        })

    # ── E) Weight scaling (×2.0) ──────────────────────────────────────────────
    for layer in ['layer3', 'layer4', 'fc']:   # only deeper layers for diversity
        specs.append({
            'id':          f'WS_scale2_{layer}',
            'description': f'Weight scale ×2.0 on {layer}',
            'operator':    'weight_scale',
            'layer':       layer,
            'scale':       2.0,
        })

    # ── F) BatchNorm reset ────────────────────────────────────────────────────
    for layer in ['layer1', 'layer2', 'layer3', 'layer4', 'bn1']:
        specs.append({
            'id':          f'BN_reset_{layer}',
            'description': f'BatchNorm reset on {layer}',
            'operator':    'bn_reset',
            'layer':       layer,
        })

    # ── G) Global dropout cripple ─────────────────────────────────────────────
    specs.append({
        'id':          'DO_cripple_50',
        'description': 'Global dropout rate set to 0.50',
        'operator':    'dropout_cripple',
        'rate':        0.50,
    })
    specs.append({
        'id':          'DO_cripple_80',
        'description': 'Global dropout rate set to 0.80',
        'operator':    'dropout_cripple',
        'rate':        0.80,
    })

    return specs


# ── 6. Apply operator from spec ────────────────────────────────────────────────
def apply_mutation(base: nn.Module, spec: dict) -> nn.Module:
    op = spec['operator']
    if op == 'weight_fuzz':
        return mutate_weight_fuzzing(base, spec['layer'], spec['noise_std'])
    elif op == 'weight_negate':
        return mutate_weight_negation(base, spec['layer'])
    elif op == 'weight_zero':
        return mutate_weight_zeroing(base, spec['layer'])
    elif op == 'weight_scale':
        return mutate_weight_scaling(base, spec['layer'], spec['scale'])
    elif op == 'bn_reset':
        return mutate_bn_reset(base, spec['layer'])
    elif op == 'dropout_cripple':
        return mutate_dropout_cripple(base, spec['rate'])
    else:
        raise ValueError(f'Unknown operator: {op}')


# ── 7. MR definitions (image-level, applied to raw tensors) ───────────────────
#
# These are simplified inline MRs for the kill matrix builder.
# The full MR library using albumentations lives in mrs/image_mrs.py
# Here we define tensor-level versions for speed (no PIL round-trip needed).
#

def mr_horizontal_flip(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, dims=[-1])

def mr_brightness_up(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x + 0.3, -3.0, 3.0)   # operates in normalised space

def mr_brightness_down(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x - 0.3, -3.0, 3.0)

def mr_gaussian_noise(x: torch.Tensor, std: float = 0.1) -> torch.Tensor:
    return x + torch.randn_like(x) * std

def mr_negate_contrast(x: torch.Tensor) -> torch.Tensor:
    """Invert pixel intensities — class-preserving for most natural images."""
    return -x

def mr_channel_shuffle(x: torch.Tensor) -> torch.Tensor:
    """Permute RGB channels — semantic class should be unaffected."""
    return x[[2, 0, 1], ...]   # R→B, G→R, B→G (single image)

def mr_vertical_flip(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, dims=[-2])

def mr_add_salt_pepper(x: torch.Tensor, density: float = 0.02) -> torch.Tensor:
    noise = torch.zeros_like(x)
    mask  = torch.rand_like(x[0]) < density   # same mask for all channels
    noise[:, mask] = torch.where(
        torch.rand(mask.sum()) > 0.5,
        torch.ones(mask.sum()),
        -torch.ones(mask.sum())
    )
    return torch.clamp(x + noise, -3.0, 3.0)

def mr_center_crop_pad(x: torch.Tensor) -> torch.Tensor:
    """Crop 28×28 centre, pad back to 32×32 — mild spatial MR."""
    _, H, W = x.shape
    crop_h, crop_w = 28, 28
    top  = (H - crop_h) // 2
    left = (W - crop_w) // 2
    cropped = x[:, top:top+crop_h, left:left+crop_w]
    padded  = torch.nn.functional.pad(cropped, (2, 2, 2, 2), mode='constant', value=0)
    return padded

def mr_identity(x: torch.Tensor) -> torch.Tensor:
    """Null MR — no transformation. Any violation = base model instability."""
    return x.clone()

# ── 11. Translation (Shift Right) ──
def mr_translate_right(x: torch.Tensor) -> torch.Tensor:
    """Shifts the image right by 4 pixels, wrapping the overflow to the left."""
    return torch.roll(x, shifts=4, dims=-1)

# ── 12. Rotate 180 Degrees ──
def mr_rotate_180(x: torch.Tensor) -> torch.Tensor:
    """Flips the image both horizontally and vertically."""
    return torch.flip(x, dims=[-1, -2])

# ── 13. Grayscale Conversion ──
def mr_grayscale(x: torch.Tensor) -> torch.Tensor:
    """Converts RGB to grayscale using standard luminance weights."""
    # Weights: R: 0.299, G: 0.587, B: 0.114
    gray = 0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2]
    # Repeat the single gray channel 3 times to maintain C=3 shape
    return gray.unsqueeze(0).repeat(3, 1, 1)

# ── 14. Cutout (Occlusion) ──
def mr_cutout(x: torch.Tensor) -> torch.Tensor:
    """Masks out an 8x8 square in the top-left corner to test partial occlusion."""
    out = x.clone()
    out[:, :8, :8] = 0.0 # Set pixels to mean (0 in normalized space)
    return out

# ── 15. Channel Drop (Blindness) ──
def mr_drop_blue_channel(x: torch.Tensor) -> torch.Tensor:
    """Zeros out the blue channel completely to test color reliance."""
    out = x.clone()
    out[2, :, :] = 0.0
    return out

# ── 16. Contrast Increase ──
def mr_contrast_up(x: torch.Tensor, factor: float = 1.5) -> torch.Tensor:
    """Pushes pixel values further away from their local channel mean."""
    mean = x.mean(dim=[-1, -2], keepdim=True)
    out = (x - mean) * factor + mean
    return torch.clamp(out, -3.0, 3.0)

# ── 17. Simple Average Blur ──
def mr_simple_blur(x: torch.Tensor) -> torch.Tensor:
    """Applies a 3x3 average blur (low-pass filter)."""
    # avg_pool2d expects a batch dimension, so we unsqueeze(0) then squeeze(0)
    x_batched = x.unsqueeze(0)
    blurred = F.avg_pool2d(x_batched, kernel_size=3, stride=1, padding=1)
    return blurred.squeeze(0)

# ── 18. Red Tint (Color Shift) ──
def mr_red_tint(x: torch.Tensor) -> torch.Tensor:
    """Adds brightness strictly to the Red channel."""
    out = x.clone()
    out[0, :, :] = torch.clamp(out[0, :, :] + 0.5, -3.0, 3.0)
    return out

# ── 19. Pixelation (Downsample/Upsample) ──
def mr_pixelate(x: torch.Tensor) -> torch.Tensor:
    """Shrinks to 16x16 then scales back to 32x32, creating a blocky effect."""
    x_batched = x.unsqueeze(0)
    down = F.interpolate(x_batched, size=(16, 16), mode='nearest')
    up = F.interpolate(down, size=(32, 32), mode='nearest')
    return up.squeeze(0)

# ── 20. Diagonal Transpose ──
def mr_transpose(x: torch.Tensor) -> torch.Tensor:
    """Swaps the height and width axes (a diagonal reflection)."""
    return torch.transpose(x, 1, 2)

MR_FUNCTIONS = {
    'MR01_hflip':         mr_horizontal_flip,
    'MR02_brightness_up': mr_brightness_up,
    'MR03_brightness_dn': mr_brightness_down,
    'MR04_gauss_noise':   mr_gaussian_noise,
    'MR05_neg_contrast':  mr_negate_contrast,
    'MR06_ch_shuffle':    mr_channel_shuffle,
    'MR07_vflip':         mr_vertical_flip,
    'MR08_salt_pepper':   mr_add_salt_pepper,
    'MR09_crop_pad':      mr_center_crop_pad,
    'MR10_identity':      mr_identity,
    'MR11_translate':     mr_translate_right,
    'MR12_rotate180':     mr_rotate_180,
    'MR13_grayscale':     mr_grayscale,
    'MR14_cutout':        mr_cutout,
    'MR15_drop_blue':     mr_drop_blue_channel,
    'MR16_contrast_up':   mr_contrast_up,
    'MR17_blur':          mr_simple_blur,
    'MR18_red_tint':      mr_red_tint,
    'MR19_pixelate':      mr_pixelate,
    'MR20_transpose':     mr_transpose,
}


# ── 8. Kill matrix builder ─────────────────────────────────────────────────────
#
# kill_matrix[mr_id][mutant_id] = True  →  MR killed this mutant
#                                 False →  MR did NOT kill this mutant
#
# A mutant is "killed" by an MR if the MR produces significantly more
# prediction changes on the mutant than on the base model.
# Threshold = base_violation_rate + 0.05 (5 percentage point buffer)
#

def compute_violation_rate(model: nn.Module,
                           mr_fn,
                           n_samples: int = 200) -> float:
    """
    Fraction of test samples where model(x) ≠ model(mr(x)).
    Uses first n_samples from the subset for speed.
    """
    model.eval()
    violations = 0
    tested     = 0

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            batch_size = images.shape[0]

            preds_orig = model(images).argmax(1)

            # Apply MR per-image (MR functions operate on single C×H×W tensors)
            transformed = torch.stack([mr_fn(img.cpu()).to(device) for img in images])
            preds_trans = model(transformed).argmax(1)

            violations += (preds_orig != preds_trans).sum().item()
            tested     += batch_size

            if tested >= n_samples:
                break

    return violations / tested


def build_kill_matrix(base_model: nn.Module,
                      mutant_specs: list[dict]) -> tuple[dict, list[dict]]:
    """
    Main function: generates all mutants, computes violation rates,
    fills kill_matrix, and returns summary metadata.

    Returns
    -------
    kill_matrix  : dict[mr_id][mutant_id] = bool
    summary      : list of dicts with per-mutant metadata
    """
    print('\n── Step 1: Computing base model violation rates per MR ──')
    base_vr = {}
    for mr_id, mr_fn in MR_FUNCTIONS.items():
        vr = compute_violation_rate(base_model, mr_fn)
        base_vr[mr_id] = vr
        print(f'  {mr_id:<25} base_VR = {vr:.4f}')

    # Initialise kill matrix
    kill_matrix = {mr_id: {} for mr_id in MR_FUNCTIONS}
    summary     = []

    print(f'\n── Step 2: Generating & testing {len(mutant_specs)} mutants ──')
    base_acc = model_accuracy(base_model)
    print(f'  Base model accuracy (subset) : {base_acc:.2f}%\n')

    for i, spec in enumerate(mutant_specs):
        t0 = time.time()
        mutant_id = spec['id']

        # Generate mutant
        mutant = apply_mutation(base_model, spec).to(device)
        mutant.eval()

        # Accuracy drop vs base
        mutant_acc = model_accuracy(mutant)
        acc_drop   = base_acc - mutant_acc

        # Check if mutant is "equivalent" (< 1% accuracy drop = too similar to base)
        is_equivalent = acc_drop < 1.0

        # For each MR: does it kill this mutant?
        killed_by = []
        for mr_id, mr_fn in MR_FUNCTIONS.items():
            mutant_vr = compute_violation_rate(mutant, mr_fn)
            threshold = base_vr[mr_id] + 0.05   # 5pp buffer above base
            killed    = (mutant_vr > threshold) and (not is_equivalent)
            kill_matrix[mr_id][mutant_id] = killed
            if killed:
                killed_by.append(mr_id)

        # Save mutant checkpoint
        ckpt_path = f'./checkpoints/mutants/{mutant_id}.pth'
        torch.save(mutant.state_dict(), ckpt_path)

        elapsed = time.time() - t0
        status  = 'EQUIV' if is_equivalent else f'killed by {len(killed_by)} MRs'
        print(f'  [{i+1:02d}/{len(mutant_specs)}] {mutant_id:<30} '
              f'acc={mutant_acc:.1f}% drop={acc_drop:.1f}%  {status}  ({elapsed:.1f}s)')

        summary.append({
            'mutant_id':      mutant_id,
            'description':    spec['description'],
            'operator':       spec['operator'],
            'layer':          spec.get('layer', 'global'),
            'mutant_acc':     round(mutant_acc, 2),
            'acc_drop':       round(acc_drop, 2),
            'is_equivalent':  is_equivalent,
            'killed_by_count': len(killed_by),
            'killed_by':       killed_by,
            'checkpoint':      ckpt_path,
        })

        # Free GPU memory
        del mutant
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return kill_matrix, summary


# ── 9. Save outputs ────────────────────────────────────────────────────────────
def save_kill_matrix(kill_matrix: dict, summary: list[dict],
                     base_vr: dict = None):

    # ── kill_matrix.pkl (primary output for scoring.py) ───────────────────────
    pkl_path = './results/kill_matrix.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump(kill_matrix, f)
    print(f'\nSaved : {pkl_path}')

    # ── kill_matrix.csv (human-readable) ──────────────────────────────────────
    mutant_ids = list(next(iter(kill_matrix.values())).keys())
    mr_ids     = list(kill_matrix.keys())

    rows = []
    for mutant_id in mutant_ids:
        row = {'mutant_id': mutant_id}
        for mr_id in mr_ids:
            row[mr_id] = int(kill_matrix[mr_id][mutant_id])
        rows.append(row)

    df = pd.DataFrame(rows).set_index('mutant_id')
    csv_path = './results/kill_matrix.csv'
    df.to_csv(csv_path)
    print(f'Saved : {csv_path}')

    # ── mutation_summary.json (metadata for scoring.py feature vectors) ───────
    fdr_per_mr = {
        mr_id: round(sum(kill_matrix[mr_id].values()) / len(kill_matrix[mr_id]), 4)
        for mr_id in mr_ids
    }

    output = {
        'total_mutants':       len(mutant_ids),
        'total_mrs':           len(mr_ids),
        'equivalent_mutants':  sum(1 for s in summary if s['is_equivalent']),
        'fdr_per_mr':          fdr_per_mr,
        'mutant_summary':      summary,
    }

    json_path = './results/mutation_summary.json'
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'Saved : {json_path}')

    # ── Print FDR table ────────────────────────────────────────────────────────
    print(f"\n{'MR':<28} {'FDR':>6}  {'Killed':>8}/{len(mutant_ids)}")
    print('─' * 48)
    for mr_id, fdr in sorted(fdr_per_mr.items(), key=lambda x: -x[1]):
        killed_n = sum(kill_matrix[mr_id].values())
        print(f'{mr_id:<28} {fdr:>6.4f}  {killed_n:>8}')

    total_killed = len([s for s in summary if s['killed_by_count'] > 0])
    print(f'\nMutation Score : {total_killed}/{len(mutant_ids)} mutants killed '
          f'({100*total_killed/len(mutant_ids):.1f}%)')


# ── 10. Main ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('=' * 60)
    print(' MR-Rank Mutation Engine — SUT-2 (ResNet18 / CIFAR-10)')
    print('=' * 60)

    # Load base model from original notebook checkpoint
    BASE_MODEL_PATH = 'resnet18_cifar10.pth'
    if not os.path.exists(BASE_MODEL_PATH):
        raise FileNotFoundError(
            f'\n[ERROR] {BASE_MODEL_PATH} not found.\n'
            'Run the original ResNet18 notebook first to generate this file.'
        )

    base_model = load_base_model(BASE_MODEL_PATH)
    base_acc   = model_accuracy(base_model)
    print(f'Base model loaded  — subset accuracy: {base_acc:.2f}%')

    if base_acc < 60:
        print('[WARNING] Base accuracy < 60% — model may not have trained properly.')
    elif base_acc > 90:
        print('[INFO] Base accuracy > 90% — mutations may produce many equivalents.')

    # Generate mutant specifications (~60 total)
    specs = define_mutant_specs()
    print(f'\nTotal mutant specs defined : {len(specs)}')

    # Build kill matrix
    t_start = time.time()
    kill_matrix, summary = build_kill_matrix(base_model, specs)
    elapsed = time.time() - t_start

    print(f'\nTotal time : {elapsed/60:.1f} minutes')

    # Save all outputs
    # Recompute base_vr for the save function (already computed inside build_kill_matrix)
    save_kill_matrix(kill_matrix, summary)

    print('\n✓ Done. Next step: run prioritization/scoring.py')
    print('  scoring.py expects: ./results/kill_matrix.pkl')
