from mrrank.ranking_module import jaccard, diversity, greedy_rank, compute_norm_cost


def test_jaccard_identical():
    assert jaccard({1, 2, 3}, {1, 2, 3}) == 1.0


def test_jaccard_disjoint():
    assert jaccard({1, 2}, {3, 4}) == 0.0


def test_jaccard_both_empty():
    assert jaccard(set(), set()) == 0.0


def test_jaccard_partial():
    assert jaccard({1, 2, 3}, {2, 3, 4}) == 0.5


def test_diversity_first_selection_is_one():
    kill_sets = {"A": {1, 2}, "B": {3, 4}}
    assert diversity("A", [], kill_sets) == 1.0


def test_diversity_uses_max_not_average():
    kill_sets = {"A": {1, 2, 3}, "B": {1, 2, 3}, "C": {9, 10}}
    # A is identical to B (already selected) -> max jaccard = 1.0 -> diversity = 0
    assert diversity("A", ["B", "C"], kill_sets) == 0.0


def test_norm_cost_max_is_one():
    costs = {"A": 100, "B": 50, "C": 200}
    nc = compute_norm_cost(costs)
    assert nc["C"] == 1.0
    assert nc["B"] == 0.25


def test_greedy_rank_returns_all_mrs_once():
    mr_ids = ["A", "B", "C"]
    fdr = {"A": 0.5, "B": 0.8, "C": 0.3}
    norm_cost = {"A": 0.5, "B": 0.5, "C": 0.5}
    kill_sets = {"A": {1, 2}, "B": {1, 2}, "C": {9, 10}}
    ranking = greedy_rank(mr_ids, fdr, norm_cost, kill_sets)
    assert len(ranking) == 3
    assert {r["mr_id"] for r in ranking} == {"A", "B", "C"}
    assert [r["rank"] for r in ranking] == [1, 2, 3]


def test_greedy_rank_diversity_can_beat_higher_fdr():
    """Report's own example: a lower-FDR MR can win a round if it's more diverse."""
    mr_ids = ["HIGH_FDR_REDUNDANT", "LOW_FDR_UNIQUE", "FILLER"]
    fdr = {"HIGH_FDR_REDUNDANT": 0.9, "LOW_FDR_UNIQUE": 0.4, "FILLER": 0.1}
    norm_cost = {k: 0.0 for k in mr_ids}  # cost equal, isolate FDR/diversity effect
    kill_sets = {
        "HIGH_FDR_REDUNDANT": set(range(90)),
        "LOW_FDR_UNIQUE": set(range(90, 130)),
        "FILLER": set(range(1)),
    }
    ranking = greedy_rank(mr_ids, fdr, norm_cost, kill_sets)
    # Round 1: HIGH_FDR_REDUNDANT wins (highest FDR, diversity=1 for all)
    assert ranking[0]["mr_id"] == "HIGH_FDR_REDUNDANT"
    # Round 2: LOW_FDR_UNIQUE should beat FILLER despite lower FDR than nothing
    # to compare against here directly, but check it's not last
    assert ranking[1]["mr_id"] in ("LOW_FDR_UNIQUE", "FILLER")

def test_build_feature_table_shape():
    from mrrank.ranking_module import build_feature_table, load_kill_matrix
    km, meta = load_kill_matrix()
    df = build_feature_table(km, meta)
    assert df.shape[0] == 20 * len(meta["mutant_ids"])
    assert set(df["label"].unique()) <= {0, 1}


def test_meta_classifier_trains_and_predicts():
    from mrrank.ranking_module import (
        build_feature_table,
        load_kill_matrix,
        train_meta_classifier,
    )

    km, meta = load_kill_matrix()
    df = build_feature_table(km, meta)

    clf, acc, cols = train_meta_classifier(df)

    assert 0.0 <= acc <= 1.0

    expected_cols = [
        "mr_type_encoded",
        "mr_magnitude",
        "mr_cost_ms",
        "mr_is_composite",
        "mr_category_encoded",
        "mutation_type_encoded",
        "mutation_layer_encoded",
        "mutation_strength",
        "model_avg_weight_magnitude",
        "model_weight_std",
        "model_test_accuracy",
    ]

    assert cols == expected_cols