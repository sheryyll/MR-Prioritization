# MR-Rank: A Scoring Framework for Metamorphic Relation Prioritization in ML Systems

Final-year engineering project (Mangalore Institute of Technology & Engineering) implementing
MR-Rank, a systematic scoring framework for prioritizing Metamorphic Relations (MRs) when
testing Machine Learning models.

## Problem

ML models cannot be tested with traditional pass/fail assertions (the "Oracle Problem").
Metamorphic Testing (MT) solves this by checking that specific transformations (e.g., flipping
an image) should not change a model's prediction. But defining dozens of these relations leads
to the "MR Explosion" problem: exhaustively running all of them is computationally expensive.

MR-Rank ranks MRs by effectiveness (Fault Detection Rate, Execution Cost, Functional Diversity)
so the most valuable tests can be run first.

## Project Structure

mr-rank/
  src/mrrank/    - installable package: data, model, MR engine, mutation engine,
                   evaluation, ranking, metrics, CLI
  configs/       - seeds, paths, hyperparameters
  notebooks/     - Kaggle/exploratory notebooks
  outputs/       - generated artifacts: kill matrix, rankings, plots (not committed)
  mutants/       - generated mutant model weights (not committed)
  app/           - Streamlit results dashboard
  tests/         - unit tests
  legacy/        - archived earlier exploration (not tracked in git)

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/sheryyll/MR-Prioritization.git
cd MR-Prioritization
```

### 2. Create a virtual environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
```

### 3. Install the package (editable mode)
```bash
pip install -e .
```
This installs mrrank plus all dependencies declared in pyproject.toml. Local installs use
CPU-only PyTorch (see below); training runs on Kaggle/Colab GPU notebooks instead.

### CPU-only PyTorch (local development only)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Status

Under active development. See project report in docs/ (Phase 1 report) for full methodology,
including the MR library (Appendix A), mutation operator specifications (Appendix B), and
evaluation metrics (Appendix C).
