# Deploying MR-Rank to Streamlit Community Cloud

This document walks you through the exact steps to deploy the MR-Rank Streamlit
app from a **private** GitHub repository at zero cost.

---

## Prerequisites

- [ ] Python 3.10+ installed locally
- [ ] Git installed
- [ ] A GitHub account (the repo can be private)
- [ ] A [Streamlit Community Cloud](https://streamlit.io/cloud) account
  (free; sign in with GitHub)

---

## Step 0 — Before you deploy: generate artifacts

The app reads pre-computed files from `app_artifacts/`. These files must be
generated and committed **before** you deploy.

```bash
# 1. Activate your local virtual environment
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

# 2. Compute Model B's real fingerprint (requires torch + CIFAR-10 data)
python scripts/compute_fingerprint.py \
    --checkpoint outputs/checkpoints/model_B.pth \
    --out outputs/model_b_fingerprint.json

# 3. Export all app artifacts
python scripts/export_artifacts.py
# Expected output: "app_artifacts/ total size: ~0.15 MB  [OK]"
```

After this, `app_artifacts/` will contain:
- `mr_library.csv`, `mr_scores.csv`, `greedy_ranking.csv`, `kill_matrix.csv`
- `metaclassifier_ranking.csv`, `evaluation_summary.json`
- `meta_classifier.joblib`, `feature_columns.json`

---

## Step 1 — Confirm repo is private

1. Go to your GitHub repo → **Settings** → **General**.
2. Under **Danger Zone**, confirm **Visibility** is **Private**.
3. Leave it private. Streamlit Community Cloud can deploy private repos
   once you grant it access (Step 3 below).

---

## Step 2 — Merge the branch

```bash
git checkout main
git merge feature/streamlit-deploy
git push origin main
```

> If you prefer, open a Pull Request on GitHub and merge from there.

---

## Step 3 — Authorize Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   your GitHub account.
2. Click **New app** → **From existing repo**.
3. If this is your first time, Streamlit will ask to install its GitHub App.
   Click **Install & Authorize** and grant access to **all repositories**
   (or at minimum the MR-Prioritization repo).

---

## Step 4 — Deploy the app

Fill in the deployment form:

| Field | Value |
|-------|-------|
| Repository | `your-username/MR-Prioritization` |
| Branch | `main` |
| Main file path | `app.py` |
| Python version | `3.13` (or match your local version) |

Click **Deploy**.

Streamlit will:
1. Clone your private repo.
2. Install `requirements-app.txt` (not requirements.txt — see note below).
3. Run `app.py`.

> **Important**: Streamlit Community Cloud looks for a file named
> `requirements.txt` by default. Since your project's `requirements.txt`
> includes torch (not needed in the app), you have two options:
>
> **Option A (recommended)**: In the Streamlit deploy settings, under
> **Advanced settings → Packages file**, type `requirements-app.txt`.
>
> **Option B**: In the Streamlit deploy UI, click **Advanced settings** →
> paste the contents of `requirements-app.txt` directly.

---

## Step 5 — Set viewer access and share the link

After deployment succeeds:

1. In the Streamlit dashboard, click the ⚙️ gear icon next to your app.
2. Under **Sharing**, set **Who can view this app** to:
   - **Anyone with the link** — for sharing with evaluators/customers without
     requiring them to create a Streamlit account.
   - **Only specific people** — for restricted access (enter email addresses).
3. Copy the app URL (e.g., `https://your-app-name.streamlit.app`).
4. Share the URL with whoever you want to give access.

---

## Step 6 — When the app sleeps (and how to wake it)

Streamlit Community Cloud free-tier apps **sleep after ~20 minutes of
inactivity**. The first request after a sleep takes 30–60 seconds to
wake the app.

**What users see**: A "Wake up" screen with a button. Clicking it starts
the container.

**To prevent sleeping** (paid plans only): Upgrade to the Teams or Enterprise
plan which supports always-on instances.

**For your use case**: The sleep behavior is acceptable. Just warn users in
the "About" tab (already done) that the first load after a period of
inactivity may be slow.

---

## Re-deploying after code changes

```bash
# Make your changes on the feature branch
git add <files>
git commit -m "feat: ..."
git push origin feature/streamlit-deploy

# Merge to main to trigger a redeploy
git checkout main
git merge feature/streamlit-deploy
git push origin main
```

Streamlit Community Cloud **automatically redeploys** when it detects a push
to the configured branch (`main`).

---

## Re-exporting artifacts (if project outputs change)

If you retrain models or rebuild the kill matrix:

```bash
python scripts/compute_fingerprint.py \
    --checkpoint outputs/checkpoints/model_B.pth \
    --out outputs/model_b_fingerprint.json

python scripts/export_artifacts.py

git add app_artifacts/
git commit -m "chore: refresh app_artifacts"
git push origin main
```

---

## Local testing

```bash
# Create a fresh test environment
python -m venv .venv-test
.venv-test\Scripts\activate

pip install -r requirements-app.txt

# Generate artifacts first (if not already done)
# (needs the full .venv with torch)
.venv\Scripts\python.exe scripts/export_artifacts.py

# Run the app
streamlit run app.py
```

The app should open at `http://localhost:8501`.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: joblib` | Make sure Streamlit is using `requirements-app.txt`, not `requirements.txt` |
| `FileNotFoundError: app_artifacts/...` | Run `scripts/export_artifacts.py` and commit the `app_artifacts/` folder |
| `InconsistentVersionWarning: sklearn` | Ensure `scikit-learn==1.9.0` in `requirements-app.txt` matches the training environment |
| App sleeps immediately | Normal on free tier — wake it by visiting the URL |
| Private repo not visible in Streamlit | Re-authorize the GitHub App with access to the specific repo |
