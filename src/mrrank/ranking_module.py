"""
Ranking Module: Greedy MR prioritization (report Eq 4.1-4.4).

Priority(MR_i) = alpha*FDR(MR_i) + beta*(1-NormCost(MR_i)) + gamma*Diversity(MR_i, S)
alpha=0.5, beta=0.2, gamma=0.3
"""
from __future__ import annotations
from mrrank.model_module import ModelWrapper
import json

import numpy as np

from mrrank import config

ALPHA, BETA, GAMMA = 0.5, 0.2, 0.3


def load_kill_matrix():
    km = np.load(config.OUTPUTS_DIR / "kill_matrix.npy")
    meta = json.load(open(config.OUTPUTS_DIR / "kill_matrix_metadata.json"))
    return km, meta


def get_kill_sets(kill_matrix: np.ndarray, mr_ids: list[str]) -> dict[str, set]:
    return {
        mr_ids[i]: set(np.where(kill_matrix[i])[0].tolist())
        for i in range(len(mr_ids))
    }


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def compute_norm_cost(cost_by_mr: dict[str, float]) -> dict[str, float]:
    max_cost = max(cost_by_mr.values())
    return {mr: (c / max_cost if max_cost else 0.0) for mr, c in cost_by_mr.items()}


def diversity(mr_id: str, selected: list[str], kill_sets: dict[str, set]) -> float:
    if not selected:
        return 1.0
    max_sim = max(jaccard(kill_sets[mr_id], kill_sets[s]) for s in selected)
    return 1 - max_sim


def greedy_rank(mr_ids: list[str], fdr: dict[str, float], norm_cost: dict[str, float],
                 kill_sets: dict[str, set]) -> list[dict]:
    remaining = list(mr_ids)
    selected: list[str] = []
    ranking = []

    while remaining:
        scores = {}
        for mr in remaining:
            div = diversity(mr, selected, kill_sets)
            scores[mr] = ALPHA * fdr[mr] + BETA * (1 - norm_cost[mr]) + GAMMA * div

        best = max(scores, key=scores.get)
        ranking.append({
            "rank": len(ranking) + 1,
            "mr_id": best,
            "priority_score": scores[best],
            "fdr": fdr[best],
            "norm_cost": norm_cost[best],
            "diversity": diversity(best, selected, kill_sets),
        })
        selected.append(best)
        remaining.remove(best)

    return ranking


def main():
    km, meta = load_kill_matrix()
    mr_ids = meta["mr_ids"]
    fdr = meta["fdr_per_mr"]
    kill_sets = get_kill_sets(km, mr_ids)

    # Cost from mr_validation.json (cost_ms per MR)
    validation = json.load(open(config.OUTPUTS_DIR / "mr_validation.json"))
    cost_by_mr = {mr: validation[mr]["cost_ms"] for mr in mr_ids}
    norm_cost = compute_norm_cost(cost_by_mr)

    ranking = greedy_rank(mr_ids, fdr, norm_cost, kill_sets)

    print(f"{'Rank':<6}{'MR':<8}{'Priority':<12}{'FDR':<8}{'NormCost':<10}{'Diversity'}")
    print("-" * 55)
    for r in ranking:
        print(f"{r['rank']:<6}{r['mr_id']:<8}{r['priority_score']:<12.4f}"
              f"{r['fdr']:<8.4f}{r['norm_cost']:<10.4f}{r['diversity']:.4f}")

    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.OUTPUTS_DIR / "greedy_ranking.json", "w") as f:
        json.dump(ranking, f, indent=2)
    print(f"\nSaved to {config.OUTPUTS_DIR / 'greedy_ranking.json'}")


# --- Phase 7: ML Meta-Classifier ---

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from mrrank import mr_engine
from mrrank.mutation_engine import build_all_mutant_specs


def build_feature_table(kill_matrix, meta) -> pd.DataFrame:
    validation = json.load(open(config.OUTPUTS_DIR / "mr_validation.json"))
    mr_ids = meta["mr_ids"]
    mutant_ids = meta["mutant_ids"]

    specs_by_id = {s.mutant_id: s for s in build_all_mutant_specs()}
    op_encoding = {"weight_fuzz_low": 1, "weight_fuzz_high": 1, "weight_negate": 2,
                   "weight_zero": 3, "label_corrupt_40": 4, "label_corrupt_55": 4}
    layer_encoding = {"layer1.0": 1, "layer1.1": 1, "layer2": 2, "layer3": 3, "layer4": 4, None: 0}

    # Model A's own feature vector -- constant across all 1600 rows, but
    # required so the classifier learns to CONDITION on model identity,
    # not just MR type. Without this, predictions can't differ for Model B.
    from mrrank import data_module
    wrapper_a = ModelWrapper.from_checkpoint(config.MODEL_A_CHECKPOINT, device="cpu")
    test_loader = data_module.get_test_loader(batch_size=128, num_workers=0)
    model_a_features = wrapper_a.extract_feature_vector(test_loader=test_loader)

    rows = []
    for i, mr_id in enumerate(mr_ids):
        mr = mr_engine.MR_BY_ID[mr_id]
        mr_meta = mr.get_metadata()
        for j, mutant_id in enumerate(mutant_ids):
            spec = specs_by_id.get(mutant_id)
            operator = spec.operator if spec else "label_corrupt"
            layer = spec.layer if spec else None
            strength = spec.strength if spec else 0.0

            rows.append({
                "mr_id": mr_id, "mutant_id": mutant_id,
                "mr_type_encoded": mr_meta["mr_type_encoded"],
                "mr_magnitude": mr_meta["magnitude"],
                "mr_cost_ms": validation[mr_id]["cost_ms"],
                "mr_is_composite": mr_meta["is_composite"],
                "mr_category_encoded": mr_meta["category_encoded"],
                "mutation_type_encoded": op_encoding.get(operator, 0),
                "mutation_layer_encoded": layer_encoding.get(layer, 0),
                "mutation_strength": strength or 0.0,
                "model_avg_weight_magnitude": model_a_features["avg_weight_magnitude"],
                "model_weight_std": model_a_features["weight_std"],
                "model_test_accuracy": model_a_features["test_accuracy"],
                "label": int(kill_matrix[i, j]),
            })
    return pd.DataFrame(rows)


def train_meta_classifier(df: pd.DataFrame, seed: int = config.SEED):
    feature_cols = ["mr_type_encoded", "mr_magnitude", "mr_cost_ms", "mr_is_composite",
                     "mr_category_encoded", "mutation_type_encoded",
                     "mutation_layer_encoded", "mutation_strength",
                     "model_avg_weight_magnitude", "model_weight_std", "model_test_accuracy"]
    X, y = df[feature_cols], df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=seed)
    clf.fit(X_train, y_train)
    test_acc = accuracy_score(y_test, clf.predict(X_test))
    return clf, test_acc, feature_cols


def predict_ranking_for_model(clf, feature_cols, model_features: dict) -> list[dict]:
    validation = json.load(open(config.OUTPUTS_DIR / "mr_validation.json"))
    rows = []
    for mr in mr_engine.ALL_MRS:
        meta = mr.get_metadata()
        rows.append({
            "mr_type_encoded": meta["mr_type_encoded"], "mr_magnitude": meta["magnitude"],
            "mr_cost_ms": validation[mr.id]["cost_ms"], "mr_is_composite": meta["is_composite"],
            "mr_category_encoded": meta["category_encoded"],
            "mutation_type_encoded": 0, "mutation_layer_encoded": 0, "mutation_strength": 0.0,
            "model_avg_weight_magnitude": model_features["avg_weight_magnitude"],
            "model_weight_std": model_features["weight_std"],
            "model_test_accuracy": model_features["test_accuracy"],
        })
    X = pd.DataFrame(rows)[feature_cols]
    probs = clf.predict_proba(X)[:, 1]
    ranked = sorted(zip([m.id for m in mr_engine.ALL_MRS], probs), key=lambda x: -x[1])
    return [{"rank": i + 1, "mr_id": mr_id, "predicted_kill_prob": float(p)}
            for i, (mr_id, p) in enumerate(ranked)]


def main_meta_classifier():
    km, meta = load_kill_matrix()
    df = build_feature_table(km, meta)
    print(f"Feature table: {df.shape}")

    clf, test_acc, feature_cols = train_meta_classifier(df)
    print(f"Held-out test accuracy: {test_acc:.4f} (target: >0.70)")

    df.to_csv(config.OUTPUTS_DIR / "meta_classifier_features.csv", index=False)
    import joblib
    joblib.dump({"clf": clf, "feature_cols": feature_cols}, config.OUTPUTS_DIR / "meta_classifier.joblib")
    print(f"Saved model + features to {config.OUTPUTS_DIR}")


if __name__ == "__main__":
    main()
    main_meta_classifier()