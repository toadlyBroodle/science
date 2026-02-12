#!/usr/bin/env python3
"""Step 5: Supervised CV classification using Random Forest.

Replaces the unsupervised Isolation Forest approach with a supervised
binary classifier trained on known CVs. Evaluates three feature subsets:
  A) Variability + CMD (8 features)
  B) Spectral + CMD (13 features)
  C) All features combined (19 features)

Uses stratified 5-fold cross-validation for evaluation, then trains
a final model on the full dataset for candidate ranking.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (precision_recall_curve, roc_curve, auc,
                              average_precision_score, roc_auc_score)
from config import CONFIG, DATA_DIR, PLOT_DIR

os.makedirs(PLOT_DIR, exist_ok=True)

print("=" * 70)
print("STEP 5: SUPERVISED CV CLASSIFICATION")
print("=" * 70)

# --- Load features ---
df = pd.read_csv(os.path.join(DATA_DIR, '04_combined_features.csv'))
with open(os.path.join(DATA_DIR, '04_feature_groups.json')) as f:
    groups = json.load(f)

print(f"  {len(df)} sources loaded\n")

# --- Identify known CVs (with caching) ---
cv_cache_path = os.path.join(DATA_DIR, '05_known_cv_ids.json')

if os.path.exists(cv_cache_path):
    print("--- Loading cached CV labels ---")
    with open(cv_cache_path) as f:
        cv_data = json.load(f)
    known_cv_ids = set(cv_data['known_cv_ids'])
    print(f"  Loaded {len(known_cv_ids)} known CVs from cache")
    print(f"  (delete {cv_cache_path} to force re-query)")
else:
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from astroquery.simbad import Simbad
    from astroquery.vizier import Vizier

    print("--- Identifying known CVs via SIMBAD + VSX ---")
    cv_types_vsx = ['UG', 'UGSU', 'UGSS', 'UGZ', 'AM', 'DQ', 'NL',
                    'N:', 'NA', 'NB', 'NC', 'DN']
    cv_types_simbad = ['CV', 'No*', 'DN', 'DQ', 'AM', 'NL']

    # Batch SIMBAD TAP query
    source_ids = df['source_id'].astype(int).tolist()
    id_strings = ', '.join(f"'Gaia DR3 {sid}'" for sid in source_ids)
    tap_query = f"""
    SELECT i.id, b.main_id, b.otype
    FROM ident AS i
    JOIN basic AS b ON i.oidref = b.oid
    WHERE i.id IN ({id_strings})
    """

    simbad_cvs = set()
    try:
        print("  SIMBAD TAP query...", flush=True)
        result = Simbad.query_tap(tap_query)
        if result:
            for row in result:
                otype = str(row['otype']).strip()
                if any(t in otype for t in cv_types_simbad):
                    gaia_str = str(row['id']).strip()
                    sid = int(gaia_str.replace('Gaia DR3 ', ''))
                    simbad_cvs.add(sid)
        print(f"  SIMBAD CVs: {len(simbad_cvs)}")
    except Exception as e:
        print(f"  SIMBAD error: {e}")

    # VSX positional query (row-by-row)
    vsx_cvs = set()
    print("  VSX positional query...", flush=True)
    for i, row in df.iterrows():
        coord = SkyCoord(ra=row['ra'] * u.deg, dec=row['dec'] * u.deg,
                         frame='icrs')
        try:
            result = Vizier(columns=['**'], row_limit=3,
                            timeout=10).query_region(
                coord, radius=10 * u.arcsec, catalog='B/vsx/vsx')
            if result and len(result) > 0 and len(result[0]) > 0:
                vtype = str(result[0][0]['Type'])
                if any(vtype.startswith(t) for t in cv_types_vsx):
                    vsx_cvs.add(int(row['source_id']))
        except Exception:
            pass
        if (i + 1) % 500 == 0 or i + 1 == len(df):
            print(f"    {i + 1}/{len(df)}", flush=True)

    print(f"  VSX CVs: {len(vsx_cvs)}")

    known_cv_ids = simbad_cvs | vsx_cvs

    # Cache for future runs
    with open(cv_cache_path, 'w') as f:
        json.dump({
            'known_cv_ids': [int(x) for x in known_cv_ids],
            'n_simbad': len(simbad_cvs),
            'n_vsx': len(vsx_cvs),
        }, f)
    print(f"  Cached CV labels to {cv_cache_path}")

df['is_known_cv'] = df['source_id'].isin(known_cv_ids)
n_cv = df['is_known_cv'].sum()
print(f"  Total known CVs in sample: {n_cv}")

if n_cv < 10:
    print(f"\n  ERROR: Need >= 10 known CVs for classification, found {n_cv}.")
    raise SystemExit(1)


# --- Define feature subsets ---
modalities = {
    'variability+CMD': groups['variability'] + groups['cmd'],
    'spectral+CMD':    groups['spectral'] + groups['cmd'],
    'all':             groups['all'],
}

print(f"\n  Feature subsets:")
for name, feats in modalities.items():
    print(f"    {name}: {len(feats)} features")


# --- Stratified 5-fold cross-validation ---
print(f"\n{'=' * 70}")
print("CROSS-VALIDATION (stratified 5-fold)")
print(f"{'=' * 70}")

y = df['is_known_cv'].astype(int).values
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_results = {}

for name, features in modalities.items():
    X = df[features].values
    print(f"\n  {name} ({len(features)} features)")

    oof_proba = np.zeros(len(df))
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        clf = RandomForestClassifier(
            n_estimators=CONFIG['n_estimators'],
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X[train_idx], y[train_idx])
        proba = clf.predict_proba(X[test_idx])[:, 1]
        oof_proba[test_idx] = proba

        if y[test_idx].sum() > 0:
            fold_ap = average_precision_score(y[test_idx], proba)
            fold_auc = roc_auc_score(y[test_idx], proba)
            fold_metrics.append({'ap': fold_ap, 'auc': fold_auc})

    # Aggregate curves from out-of-fold predictions
    prec, rec, pr_thresh = precision_recall_curve(y, oof_proba)
    fpr, tpr, roc_thresh = roc_curve(y, oof_proba)
    pr_auc_val = auc(rec, prec)
    roc_auc_val = auc(fpr, tpr)

    # Recall at precision thresholds
    mask_50 = prec >= 0.50
    recall_at_50 = rec[mask_50].max() if mask_50.any() else 0.0
    mask_80 = prec >= 0.80
    recall_at_80 = rec[mask_80].max() if mask_80.any() else 0.0

    cv_results[name] = {
        'prec': prec, 'rec': rec,
        'fpr': fpr, 'tpr': tpr,
        'pr_auc': pr_auc_val, 'roc_auc': roc_auc_val,
        'recall_at_50': recall_at_50,
        'recall_at_80': recall_at_80,
        'oof_proba': oof_proba,
        'fold_metrics': fold_metrics,
    }

    mean_ap = np.mean([m['ap'] for m in fold_metrics])
    std_ap = np.std([m['ap'] for m in fold_metrics])
    print(f"    AUC-PR:  {pr_auc_val:.3f}  "
          f"(per-fold: {mean_ap:.3f} +/- {std_ap:.3f})")
    print(f"    AUC-ROC: {roc_auc_val:.3f}")
    print(f"    Recall @ 50% precision: {recall_at_50:.1%}")
    print(f"    Recall @ 80% precision: {recall_at_80:.1%}")


# --- Train final model on full dataset ---
print(f"\n--- Training final model (all features) ---")

features_all = groups['all']
X_full = df[features_all].values

final_clf = RandomForestClassifier(
    n_estimators=CONFIG['n_estimators'],
    class_weight='balanced',
    oob_score=True,
    random_state=42,
    n_jobs=-1,
)
final_clf.fit(X_full, y)

df['cv_probability'] = final_clf.predict_proba(X_full)[:, 1]
print(f"  Trained on {len(df)} sources ({n_cv} CVs, "
      f"{len(df) - n_cv} non-CVs)")
print(f"  OOB accuracy: {final_clf.oob_score_:.3f}")

# Store OOF probabilities from CV for honest evaluation
df['cv_probability_oof'] = cv_results['all']['oof_proba']


# --- Feature importances ---
importances = final_clf.feature_importances_
feat_imp = pd.DataFrame({
    'feature': features_all,
    'importance': importances,
}).sort_values('importance', ascending=False)

print(f"\n--- Feature Importances (top 15) ---")
for _, row in feat_imp.head(15).iterrows():
    bar = '#' * int(row['importance'] * 100)
    print(f"    {row['feature']:<25} {row['importance']:.4f} {bar}")


# --- 4-panel figure ---
print(f"\n--- Generating figures ---")

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

colors = {
    'variability+CMD': '#e74c3c',
    'spectral+CMD':    '#3498db',
    'all':             '#2ecc71',
}

# (a) Precision-Recall curves
ax = axes[0, 0]
for name, res in cv_results.items():
    ax.plot(res['rec'], res['prec'],
            label=f'{name} (AUC={res["pr_auc"]:.3f})',
            color=colors[name], lw=2)
baseline = n_cv / len(df)
ax.axhline(baseline, ls='--', color='gray', lw=1, alpha=0.5,
           label=f'Baseline ({baseline:.3f})')
ax.set_xlabel('Recall', fontsize=11)
ax.set_ylabel('Precision', fontsize=11)
ax.set_title('(a) Precision-Recall Curves', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# (b) ROC curves
ax = axes[0, 1]
for name, res in cv_results.items():
    ax.plot(res['fpr'], res['tpr'],
            label=f'{name} (AUC={res["roc_auc"]:.3f})',
            color=colors[name], lw=2)
ax.plot([0, 1], [0, 1], ls='--', color='gray', lw=1, alpha=0.5,
        label='Random')
ax.set_xlabel('False Positive Rate', fontsize=11)
ax.set_ylabel('True Positive Rate', fontsize=11)
ax.set_title('(b) ROC Curves', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='lower right')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# (c) Feature importances (top 15)
ax = axes[1, 0]
top_feats = feat_imp.head(15).iloc[::-1]  # reverse for horizontal bars
group_colors = []
for feat in top_feats['feature']:
    if feat in groups['variability']:
        group_colors.append('#e74c3c')
    elif feat in groups['cmd']:
        group_colors.append('#f39c12')
    else:
        group_colors.append('#3498db')
ax.barh(range(len(top_feats)), top_feats['importance'], color=group_colors)
ax.set_yticks(range(len(top_feats)))
ax.set_yticklabels(top_feats['feature'], fontsize=9)
ax.set_xlabel('Importance', fontsize=11)
ax.set_title('(c) Feature Importances (top 15)', fontsize=12,
             fontweight='bold')
ax.legend(handles=[
    Patch(facecolor='#e74c3c', label='Variability'),
    Patch(facecolor='#f39c12', label='CMD'),
    Patch(facecolor='#3498db', label='Spectral'),
], fontsize=9, loc='lower right')

# (d) CV probability distribution
ax = axes[1, 1]
cv_probs = df.loc[df['is_known_cv'], 'cv_probability']
non_cv_probs = df.loc[~df['is_known_cv'], 'cv_probability']
ax.hist(non_cv_probs, bins=50, alpha=0.6, color='gray',
        label=f'Non-CV ({len(non_cv_probs)})', density=True)
ax.hist(cv_probs, bins=30, alpha=0.7, color='#e74c3c',
        label=f'Known CVs ({len(cv_probs)})', density=True)
ax.set_xlabel('CV Probability', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('(d) CV Probability Distribution', fontsize=12,
             fontweight='bold')
ax.legend(fontsize=9)
ax.set_xlim(0, 1)

plt.suptitle('Supervised CV Classification: Random Forest',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'classification_results.png'), dpi=150)
plt.close()
print(f"  Saved: {PLOT_DIR}/classification_results.png")


# --- Save ---
out = os.path.join(DATA_DIR, '05_classification_results.csv')
df.to_csv(out, index=False)
print(f"\n  Saved {len(df)} sources to {out}")

feat_imp.to_csv(os.path.join(DATA_DIR, '05_feature_importances.csv'),
                index=False)
print(f"  Saved feature importances to 05_feature_importances.csv")


# --- Summary table ---
print(f"\n{'=' * 70}")
print("CLASSIFICATION SUMMARY")
print(f"{'=' * 70}")
print(f"{'Modality':<20} {'AUC-PR':>8} {'AUC-ROC':>8} "
      f"{'R@P50':>8} {'R@P80':>8}")
print("-" * 55)
for name, res in cv_results.items():
    print(f"{name:<20} {res['pr_auc']:>8.3f} {res['roc_auc']:>8.3f} "
          f"{res['recall_at_50']:>7.1%} {res['recall_at_80']:>7.1%}")

n_novel = ((~df['is_known_cv']) & (df['cv_probability'] > 0.5)).sum()
print(f"\nKnown CVs: {n_cv}")
print(f"Novel candidates (p > 0.5): {n_novel}")
print(f"Top features: {', '.join(feat_imp.head(5)['feature'].tolist())}")
