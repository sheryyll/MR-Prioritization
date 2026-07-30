# AI HANDOFF

**Project:** MR-Rank  
**Author:** Jenisha  
**Status:** Active Development (Phase 0 Complete ✅ → Phase 1 Ready 🚀)  

---

# 1. Project Overview

MR-Rank is a research-oriented machine learning testing framework that prioritizes Metamorphic Relations (MRs) instead of executing every available MR.

The project follows the methodology described in the project report.

### Problem
Machine Learning models suffer from the **Oracle Problem**. Traditional software testing cannot determine whether predictions are correct without an oracle for arbitrary inputs.

### Solution
Use **Metamorphic Testing (MT)**. MT sidesteps the oracle problem by defining rules/transformations (e.g., "flipping this image shouldn't change the predicted class").

### Problem with MT
Running every MR across all tests is expensive and redundant (**MR Explosion** problem).

### MR-Rank Solution
Score and rank the 20 MRs according to how useful they are at detecting faults so the best few can be run first. Two ranking algorithms are implemented:

* **Algorithm 1 (Greedy heuristic):** Ranks MRs round-by-round using:
  $$\text{Priority}(\text{MR}_i) = 0.5 \cdot \text{FDR} + 0.2 \cdot (1 - \text{NormCost}) + 0.3 \cdot \text{Diversity}$$
* **Algorithm 2 (Meta-learning model):** Gradient Boosting trained on feature vectors to predict MR usefulness for completely new models without generating mutants again.

---

# 2. Source of Truth

The following documents are the **ONLY** source of truth:
1. Project Report (`Project Phase Final Report 1.15.pdf`, `MR Prioritization - Final Report.pdf`)
2. Workflow Document / Architecture
3. Implementation Guide
4. Supplementary Explanation Documents ("step-by-step" and "new model" explainers)

* Do NOT redesign the methodology unless a major technical issue exists.
* If documentation conflicts with implementation, the documentation wins.

---

# 3. Development Philosophy & Mentoring Role

Treat this project like production-quality research software.

When acting as an AI assistant / mentor:
* Explain **WHY** before writing code.
* Explain every library, framework, tool, and command before using it.
* Keep modules independent and adhere to clean architecture.
* Use type hints, docstrings, and PEP8 standards.
* Write maintainable, production-quality code.
* Never skip implementation details or assume prior knowledge.
* Debug errors rather than simply rewriting code.
* Preserve research reproducibility at all costs.
* Follow the project report; do not over-engineer or alter research methodology unless explicitly instructed.
* Work strictly **one phase at a time** and wait for user confirmation before moving forward.

---

# 4. Current Project Structure

```text
mr-rank/
├── pyproject.toml              # Makes package installable (`pip install -e .`)
├── requirements.txt
├── .gitignore
├── README.md
├── configs/
│   └── config.py                # Single source of truth (SEED = 42, paths, hyperparams)
├── src/
│   └── mrrank/
│       ├── __init__.py
│       ├── data_module.py       # Phase 1
│       ├── model_module.py      # Phase 2 (ResNet-18 wrapper, train/eval/save/load)
│       ├── mr_engine.py          # Phase 3 (20 MR transform classes)
│       ├── mutation_engine.py    # Phase 4 (4 mutation operators)
│       ├── evaluation_module.py  # Phase 5 (Kill Matrix construction)
│       ├── ranking_module.py     # Phase 6 & 7 (Greedy + Meta-classifier)
│       ├── metrics.py            # Phase 9 (APFD, FD@k, Wilcoxon)
│       └── cli.py                # Exposes `mrrank train`, `mrrank build-kill-matrix`, etc.
├── notebooks/                  # Exploratory / Kaggle notebooks (.ipynb)
├── outputs/                    # kill_matrix.npy, ranked_lists.json, plots/ (gitignored)
├── mutants/                    # 80 mutant .pth files (gitignored)
└── app/
    └── dashboard.py             # Streamlit UI (Phase 10)

# 5. Current Progress & Session Log
Phase 0 ✅ Complete
Repository Cleaned: Legacy exploratory code archived into legacy/ and excluded from git.

Package Created: Modern src/mrrank/ structure with pyproject.toml created; pip install -e . and import mrrank verified.

Environment Setup: Local .venv configured with CPU-only PyTorch for fast development/testing (heavy GPU work assigned to Kaggle).

Version Control: Git repo configured, proper .gitignore added, authenticated via Git Credential Manager, and pushed to GitHub (sheryyll/MR-Prioritization).

Current Branch: main

Latest Commit: 5eba2cb..b27a8f3 main -> main

# 6. Next Phase: Phase 1 — Data ModuleGoals:Load CIFAR-10 dataset.Apply official normalization:$$\mu = [0.4914, 0.4822, 0.4465], \quad \sigma = [0.2023, 0.1994, 0.2010]$$Create a reproducible 500-image stratified evaluation subset.CRITICAL: Everything afterwards depends on this subset. It must NEVER change once generated.

# 7. Overall Roadmap & Compressed Timeline (2 Months)

Week,Phase,Milestone,Report Reference
1,Phase 0,"Environment setup, project structure, reproducibility (seeds)",Constraints section
1,Phase 1,"Data Module: CIFAR-10 loading, normalization, fixed 500-image subset",Sec 4.3
1–2,Phase 2,Model Module: ResNet-18 architecture + training loop → Model A,"Sec 4.3, Eq 4.5"
2,Phase 3,MR Engine: Implement all 20 MRs + validate <5% false-positive rate,Appendix A
3,Phase 4,Mutation Engine: 4 operators → 80 mutants (verify 40–70% accuracy),Appendix B
4,Phase 5,"Evaluation Module: Build 20×80 Kill Matrix (1,600 evaluations)",Sec 4.3
5,Phase 6,Ranking Module — Algorithm 1: Greedy prioritization,Eq 4.1–4.4
5,Phase 7,Ranking Module — Algorithm 2: ML Meta-Classifier (Gradient Boosting),Sec 4.1.2
6,Phase 8,"Model B: Train second ResNet, extract features, predict ranking",Sec 6.3–6.6 (supp doc)
7,Phase 9,"Evaluation: APFD, FD@k curves, Wilcoxon signed-rank test",Sec 4.3.1 Step 7
8,Phase 10,Streamlit UI / Dashboard + polish + report write-up support

# 8. Technical Stack
Language: Python 3.10+

Deep Learning Framework: PyTorch 2.0+, TorchVision

Data & Image Processing: NumPy, OpenCV (cv2)

Machine Learning & Stats: scikit-learn (GradientBoostingClassifier), SciPy (scipy.stats.wilcoxon)

Visualization & Web App: Matplotlib, Seaborn, Streamlit

CLI & Tooling: pyproject.toml, Setuptools

# 9. Important Constraints & Formulas
Fixed Seeds: Always set SEED = 42 across PyTorch, NumPy, and random.

Eval Subset: Exactly 500 images, stratified across classes, non-modifiable.

Mutant Health Criteria: Accuracy must fall strictly within 40–70% on the 500-image eval subset.

MR False Positive Rate: Clean model violation rate must be <5%.

Kill Condition: Mutant Violation Rate−Clean Model Violation Rate>5%.

Mutant Composition (80 Total):

15 Fuzz-Low

15 Fuzz-High

15 Negation

15 Zeroing

10 Label-Corrupt 10%

10 Label-Corrupt 20%

Diversity Formula:

Diversity(MR 
i
​
 ,S)=1− 
MR 
j
​
 ∈S
max
​
 Jaccard(KillSet(MR 
i
​
 ),KillSet(MR 
j
​
 ))
10. Architectural Pipeline
Train Base Model A⟶Validate MRs (<5% FP)⟶Generate 80 Mutants
↓
Run 20 MRs vs. 80 Mutants⟶Build 20×80 Kill Matrix
↓
Compute FDR, Cost, Diversity⟶Algorithm 1 (Greedy)  &  Algorithm 2 (Meta-Classifier)
↓
Evaluate Rankings (APFD, FD@k, Wilcoxon Test)
11. Coding Standards
Strictly adhere to PEP8.

Use explicit Type Hints and complete Docstrings.

Keep functions small and modular; no duplicated logic.

Centralize configuration values in configs/config.py (no magic numbers).

12. Instructions for AI Assistants
Any AI assistant continuing work on this project must:

Read this entire AI_HANDOFF.md file.

Read README.md.

Understand the exact current phase (Phase 1).

Focus solely on the current phase; never jump ahead.

Explain implementation decisions and command execution before asking the user to run them.

Wait for user confirmation before proceeding to the next milestone.

13. Additional Decisions & Clarifications (Post-Discussion Log)
NOTE: The sections below contain verbatim responses from technical mentorship sessions documenting key compute strategies, architecture decisions, and setup instructions.

1. Compute Strategy: Kaggle over Colab
Your Intel UHD graphics has no CUDA support, so all training happens in the cloud. Between Colab and Kaggle, I'd recommend Kaggle as primary:

Kaggle: 30 GPU-hours/week (resets weekly), sessions run up to 12 hours, and — critically — you get a persistent dataset/output system, so your Model A, mutants, and kill matrix survive between sessions without you re-uploading anything to Drive each time.

Colab (free): shorter session limits, more aggressive disconnects, and no native persistent storage (you'd have to mount Google Drive every session, which is workable but more fragile for something like 80 mutant .pth files).

My recommendation: Kaggle for anything GPU-heavy (training Model A, generating mutants, building the kill matrix, training Model B). Use Colab only as backup if you hit your weekly Kaggle GPU quota.

One real risk I want to flag now, not later: the report's own constraint says label-corruption mutants need 8–12 GPU hours total because each of the 20 variants requires full retraining. On a 30hrs/week budget shared with everything else, that's tight. My proposed fix (I'll confirm with you at Phase 4, not now): retrain those mutants for far fewer epochs than the full 100-epoch Model A schedule — since the goal is only to land them in the 40–70% accuracy range, not to fully converge. This is a legitimate engineering call, not a deviation from the report's intent. Flagging it now so it's not a surprise later.

2. Timeline: 2 months instead of 8
I'm compressing the 10 phases into roughly this cadence. Since you're comfortable with Python/PyTorch, I'll move faster and skip basics-explanations unless something is genuinely non-obvious.

Week	Phase(s)
1	Phase 0 (setup) + Phase 1 (data) + Phase 2 start (Model A training)
2	Phase 2 finish + Phase 3 (MR Engine, all 20 MRs + validation)
3	Phase 4 (80 mutants) — this is the heaviest compute week
4	Phase 5 (Kill Matrix — 1,600 evaluations)
5	Phase 6 (Greedy algorithm) + Phase 7 (Meta-classifier)
6	Phase 8 (Model B + generalization test)
7	Phase 9 (APFD, FD@k, Wilcoxon, comparison tables)
8	Phase 10 (UI/dashboard) + polish + report write-up support
We can compress further once we see how fast Phase 2–4 actually go on Kaggle's GPU.

3. Package Structure Rationale
Given the "no hand-holding needed" signal, here's the concrete structure I'll set up:

Plaintext
mr-rank/
├── pyproject.toml              # makes this pip install -e .-able
├── requirements.txt
├── .gitignore
├── README.md
├── configs/
│   └── config.py                # SEED, paths, hyperparams — single source of truth
├── src/
│   └── mrrank/
│       ├── __init__.py
│       ├── data_module.py
│       ├── model_module.py
│       ├── mr_engine.py
│       ├── mutation_engine.py
│       ├── evaluation_module.py
│       ├── ranking_module.py
│       ├── metrics.py
│       └── cli.py               # exposes mrrank train, mrrank build-kill-matrix, etc.
├── notebooks/
│   └── (Kaggle notebooks live here as .ipynb, synced back to repo)
├── outputs/                     # kill_matrix.npy, ranked_lists.json, plots/ — gitignored (large/binary)
├── mutants/                     # 80 .pth files — gitignored (too large for git)
└── app/
    └── dashboard.py              # Streamlit UI (Phase 10)
Why pyproject.toml + src/mrrank layout: this is the modern standard for a real installable Python package (as opposed to a loose folder of scripts). It means anywhere — Kaggle, Colab, your laptop — you can run pip install -e . once and then just import mrrank or run mrrank train-model-a from the CLI, instead of juggling sys.path hacks. It also makes your final GitHub repo look like an actual engineering artifact, which matters if this gets evaluated or you want to show it off later.

Why a CLI (cli.py): instead of re-running notebook cells by hand every session, you get repeatable commands (mrrank train --model A, mrrank build-kill-matrix). This matters a lot on Kaggle where sessions restart.

Why Streamlit for the UI: it's the fastest way to get a real interactive dashboard (kill-matrix heatmap, live-updating ranked MR list, APFD/FD@k charts) with almost no frontend code — just Python. It reads directly from your outputs/ artifacts, so it doesn't need to know anything about the training internals. I'll build this in Phase 10 once we actually have real results to show, but I'll design the output file formats from Phase 5 onward so they're dashboard-ready (no refactor needed later).

One question before we lock this in: do you want the dashboard to be purely a results viewer (loads the kill matrix, rankings, metrics you've already computed and displays them nicely — this is fast, ~1 day of work), or do you also want it to trigger/run parts of the pipeline interactively (e.g., "click here to re-rank with different α/β/γ weights and see the numbers update live")? The second is more impressive but adds real work. I'd suggest we default to the first (viewer) and treat live re-ranking as a stretch goal if time allows in week 8 — let me know if you disagree.

4. Phase 0 — Concrete Setup Steps
Step 0.1 — Local repo skeleton (on your Windows machine)
Open a terminal (PowerShell or your preferred shell) and run:   

Bash
mkdir mr-rank
cd mr-rank
git init
This creates the project folder and initializes git version control — even though heavy training happens on Kaggle, your code should live in git so it's versioned, and Kaggle notebooks will just pull or pip install from this repo (or you upload it as a Kaggle Dataset/utility script).   

Step 0.2 — Create the folder structure
Commands differ depending on shell (Windows PowerShell vs WSL/Git Bash).

Step 0.3 — Kaggle account check
Before we write any training code:

Confirm you have a Kaggle account and it's phone-verified (Kaggle requires phone verification to enable GPU access on notebooks — this trips people up if skipped).

Once verified, create a new Kaggle Notebook, and in the right sidebar under "Settings" → "Accelerator", confirm you can select GPU T4 x2 or GPU P100.

14. Key Decisions Summary
  
Primary Compute: Kaggle (30 GPU hours/week, persistent output storage).

Backup Compute: Google Drive / Colab.

Package Architecture: pyproject.toml with src/mrrank layout and CLI interface (cli.py).

Dashboard: Streamlit app in Phase 10 defaulting to results viewer.

Label Mutation Strategy: Epoch reduction for label corruption mutants to keep GPU runtime within budget while achieving the 40–70% accuracy requirement.