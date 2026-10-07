"""Smoke test for all app data loaders and inference logic."""
import json, sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

ARTIFACTS = Path('app_artifacts')
OK = True

def check(cond, msg):
    global OK
    if not cond:
        print(f'  FAIL: {msg}')
        OK = False

print('=== Smoke-testing all data loaders ===')

# 1. greedy_ranking.csv
greedy = pd.read_csv(ARTIFACTS / 'greedy_ranking.csv')
check(len(greedy) == 20, f'greedy: expected 20 rows, got {len(greedy)}')
check('rank' in greedy.columns and 'mr_id' in greedy.columns, 'greedy: missing columns')
print(f'  greedy_ranking.csv     OK ({len(greedy)} rows, top: {greedy.iloc[0]["mr_id"]})')

# 2. metaclassifier_ranking.csv
meta = pd.read_csv(ARTIFACTS / 'metaclassifier_ranking.csv')
check(len(meta) == 20, f'meta: expected 20 rows, got {len(meta)}')
print(f'  metaclassifier_ranking OK ({len(meta)} rows, top: {meta.iloc[0]["mr_id"]})')

# 3. mr_library.csv
lib = pd.read_csv(ARTIFACTS / 'mr_library.csv')
check(len(lib) == 20, f'lib: expected 20 rows')
required = {'mr_id', 'name', 'category', 'mr_type', 'magnitude', 'is_composite'}
check(required.issubset(lib.columns), f'lib: missing cols {required - set(lib.columns)}')
print(f'  mr_library.csv         OK ({len(lib)} rows)')

# 4. mr_scores.csv
scores = pd.read_csv(ARTIFACTS / 'mr_scores.csv')
check(len(scores) == 20, 'scores: expected 20 rows')
print(f'  mr_scores.csv          OK ({len(scores)} rows)')

# 5. kill_matrix.csv
km = pd.read_csv(ARTIFACTS / 'kill_matrix.csv', index_col=0)
check(km.shape == (20, 80), f'kill matrix: expected (20,80), got {km.shape}')
print(f'  kill_matrix.csv        OK shape={km.shape}')

# 6. evaluation_summary.json
ev = json.loads((ARTIFACTS / 'evaluation_summary.json').read_text())
check('apfd' in ev and 'fd_at_k' in ev and 'wilcoxon' in ev, 'eval: missing keys')
apfd_g = ev['apfd']['greedy']
apfd_r = ev['apfd']['random_mean']
check(apfd_g > apfd_r, f'APFD sanity: greedy ({apfd_g:.4f}) should beat random ({apfd_r:.4f})')
print(f'  evaluation_summary.json OK (APFD greedy={apfd_g:.4f}, random={apfd_r:.4f})')

# 7. feature_columns.json
fc = json.loads((ARTIFACTS / 'feature_columns.json').read_text())
check(len(fc) == 11, f'features: expected 11, got {len(fc)}')
print(f'  feature_columns.json   OK ({len(fc)} features: {fc[:3]}...)')

# 8. meta_classifier.joblib
bundle = joblib.load(ARTIFACTS / 'meta_classifier.joblib')
clf = bundle['clf']
check(hasattr(clf, 'predict_proba'), 'clf: no predict_proba')
print(f'  meta_classifier.joblib OK ({type(clf).__name__}, {clf.n_estimators} estimators)')

# 9. Model B fingerprint
fp = json.loads(Path('outputs/model_b_fingerprint.json').read_text())
check('avg_weight_magnitude' in fp, 'fp: missing avg_weight_magnitude')
print(f'  Model B fingerprint    OK (acc={fp["test_accuracy"]:.4f}, avg_w={fp["avg_weight_magnitude"]:.6f})')

# 10. Live inference test — verify matches the CSV
op_enc = {'weight_fuzz_low':1,'weight_fuzz_high':1,'weight_negate':2,'weight_zero':3,
          'label_corrupt_40':4,'label_corrupt_55':4}
layer_enc = {'layer1.0':1,'layer1.1':1,'layer2':2,'layer3':3,'layer4':4}
mutant_ids = list(km.columns)

def parse_mutant(mid):
    if mid.startswith('LC_'):
        pct = float(mid.split('_')[1].replace('pct',''))/100
        return {'mutation_type_encoded':4,'mutation_layer_encoded':0,'mutation_strength':pct}
    parts = mid.split('_'); op = parts[0]
    if op == 'WF':
        op_str = f'weight_fuzz_{parts[1]}'; lk = parts[2]
    elif op == 'WN':
        op_str = 'weight_negate'; lk = parts[1]
    else:
        op_str = 'weight_zero'; lk = parts[1]
    return {'mutation_type_encoded': op_enc.get(op_str, 0),
            'mutation_layer_encoded': layer_enc.get(lk, 0),
            'mutation_strength': 0.0}

mut_profiles = pd.DataFrame([parse_mutant(m) for m in mutant_ids])
validation = json.loads(Path('outputs/mr_validation.json').read_text())
mr_ids = list(km.index)

rows = []
for mr_id in mr_ids:
    vm = validation[mr_id]
    base = {'_mr': mr_id,
            'mr_type_encoded': vm['mr_type_encoded'],
            'mr_magnitude': vm['magnitude'],
            'mr_cost_ms': vm['cost_ms'],
            'mr_is_composite': vm['is_composite'],
            'mr_category_encoded': vm['category_encoded'],
            'model_avg_weight_magnitude': fp['avg_weight_magnitude'],
            'model_weight_std': fp['weight_std'],
            'model_test_accuracy': fp['test_accuracy']}
    for mu in mut_profiles.to_dict('records'):
        rows.append({**base, **mu})

X = pd.DataFrame(rows)
X['p'] = clf.predict_proba(X[fc])[:,1]
mean_p = X.groupby('_mr')['p'].mean().sort_values(ascending=False)
live_top5 = list(mean_p.index[:5])
csv_top5 = meta['mr_id'].tolist()[:5]

print(f'  Inference live top-5:  {live_top5}')
print(f'  CSV   top-5:           {csv_top5}')
check(live_top5 == csv_top5, f'Ranking mismatch: live={live_top5} vs csv={csv_top5}')
if live_top5 == csv_top5:
    print('  Ranking MATCHES metaclassifier_ranking.csv')

print()
if OK:
    print('All smoke tests PASSED')
else:
    print('Some tests FAILED — see above')
    sys.exit(1)
