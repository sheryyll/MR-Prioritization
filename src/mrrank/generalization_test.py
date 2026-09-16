from __future__ import annotations

import json

import joblib

from mrrank import config, data_module
from mrrank.model_module import ModelWrapper
from mrrank.ranking_module import predict_ranking_for_model


def main():
    print("Loading Model B checkpoint...")
    wrapper_b = ModelWrapper.from_checkpoint(config.MODEL_B_CHECKPOINT, device="cpu")

    print("Extracting Model B feature vector (no mutation)...")
    test_loader = data_module.get_test_loader(batch_size=128, num_workers=0)
    model_b_features = wrapper_b.extract_feature_vector(test_loader=test_loader)
    print("Model B features:", model_b_features)

    print("\nLoading trained meta-classifier...")
    bundle = joblib.load(config.OUTPUTS_DIR / "meta_classifier.joblib")
    clf, feature_cols = bundle["clf"], bundle["feature_cols"]

    predicted_ranking = predict_ranking_for_model(clf, feature_cols, model_b_features)

    print(f"\n{'Rank':<6}{'MR':<8}{'Predicted Kill Prob'}")
    print("-" * 35)
    for r in predicted_ranking:
        print(f"{r['rank']:<6}{r['mr_id']:<8}{r['predicted_kill_prob']:.4f}")

    with open(config.OUTPUTS_DIR / "model_b_predicted_ranking.json", "w") as f:
        json.dump(
            {
                "model_b_features": model_b_features,
                "predicted_ranking": predicted_ranking
            },
            f,
            indent=2
        )

    print(f"\nSaved to {config.OUTPUTS_DIR / 'model_b_predicted_ranking.json'}")


if __name__ == "__main__":
    main()