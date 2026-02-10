#!/usr/bin/env python3
"""Step 4: Priority scoring and final candidate table."""

import os
import numpy as np
import pandas as pd
from config import CONFIG, DATA_DIR

print("=" * 70)
print("STEP 4: PRIORITY SCORING")
print("=" * 70)

df = pd.read_csv(os.path.join(DATA_DIR, '03_crossmatched.csv'))
n = len(df)
print(f"  Loaded {n} candidates\n")


def priority_score(row):
    score = row['combined_score'] * 5  # base ML score

    # X-ray: strong accretion indicator
    if row.get('has_xray', False):
        score += 3

    # GALEX UV
    if pd.notna(row.get('galex_fuv')):
        score += 3   # FUV = very strong
    elif pd.notna(row.get('galex_nuv')):
        score += 1.5

    # SIMBAD: novel = high value
    if not row.get('in_simbad', True):
        score += 5   # NOT in SIMBAD - potentially novel
    else:
        otype = str(row.get('simbad_otype', ''))
        if any(t in otype for t in ['CV', 'No*', 'DN', 'DQ', 'AM', 'NL']):
            score += 1

    # Gaia variability class
    gc = str(row.get('gaia_vari_class', ''))
    if gc and gc not in ['None', 'nan', '']:
        if any(c in gc for c in ['CV', 'SYST', 'ECL']):
            score += 1.5

    # VSX
    if not row.get('in_vsx', True):
        score += 4   # Not catalogued
    elif row.get('vsx_type') in ['VAR', 'VAR:', None, '']:
        score += 2.5  # Generic type

    # ZTF
    ztf_obs = row.get('ztf_nobs', 0) or 0
    if ztf_obs > 100: score += 1.5
    elif ztf_obs > 50: score += 1
    elif ztf_obs > 0: score += 0.5

    ztf_p = row.get('ztf_period')
    vsx_p = row.get('vsx_period')
    if pd.notna(ztf_p) and pd.notna(vsx_p) and vsx_p > 0:
        if abs(ztf_p - vsx_p) / vsx_p < 0.1:
            score += 1

    # TIC match
    if pd.notna(row.get('tic_id')):
        score += 2

    # Short period
    period = row.get('vsx_period')
    if pd.notna(period) and period < 0.1:
        score += 2

    # Negative skewness (outburst-like)
    if row.get('skewness_mag_g_fov', 0) < -1:
        score += 1

    return score


df['priority_score'] = df.apply(priority_score, axis=1)
df = df.sort_values('priority_score', ascending=False)

threshold = CONFIG['priority_score_min']
high = df[df['priority_score'] >= threshold]

print(f"Priority scoring criteria:")
print(f"  +5*score (ML base), +3 X-ray, +3 FUV / +1.5 NUV")
print(f"  +5 not-in-SIMBAD, +4 not-in-VSX, +2.5 generic-VAR")
print(f"  +1.5 Gaia CV/SYST/ECL, +2 TIC match, +2 short period")
print(f"  +0.5-1.5 ZTF obs, +1 ZTF/VSX period agree, +1 neg skewness")
print(f"\n  Above threshold ({threshold}): {len(high)} / {n}")
print(f"  Score range: {df['priority_score'].min():.1f} - {df['priority_score'].max():.1f}\n")

# Print top candidates
print("=" * 70)
print(f"HIGH PRIORITY CANDIDATES (score >= {threshold})")
print("=" * 70)

for rank, (_, row) in enumerate(high.iterrows(), 1):
    print(f"\n  #{rank} | Score: {row['priority_score']:.1f} | Gaia DR3 {int(row['source_id'])}")
    print(f"    G={row['phot_g_mean_mag']:.2f}, BP-RP={row['bp_rp']:.2f}, Amp={row['amplitude']:.2f}")
    parts = []
    if row.get('in_vsx'):
        p = f"P={row['vsx_period']:.4f}d" if pd.notna(row.get('vsx_period')) else ""
        parts.append(f"VSX:{row.get('vsx_type','')} {p}")
    else:
        parts.append("NOT in VSX")
    if row.get('in_simbad'):
        parts.append(f"SIMBAD:{row.get('simbad_otype','')} ({row.get('simbad_name','')})")
    else:
        parts.append("NOT in SIMBAD")
    if row.get('has_xray'): parts.append(f"X-ray:{row.get('xray_count_rate',0):.3f}ct/s")
    if pd.notna(row.get('galex_fuv')): parts.append(f"FUV={row['galex_fuv']:.1f}")
    if pd.notna(row.get('galex_nuv')): parts.append(f"NUV={row['galex_nuv']:.1f}")
    gc = row.get('gaia_vari_class')
    if gc and str(gc) not in ['None', 'nan']: parts.append(f"GaiaV:{gc}")
    if row.get('ztf_nobs', 0) > 0: parts.append(f"ZTF:{int(row['ztf_nobs'])}obs")
    if pd.notna(row.get('tic_id')): parts.append(f"TIC:{int(row['tic_id'])}")
    print(f"    {' | '.join(parts)}")

# Save
out_all = os.path.join(DATA_DIR, '04_prioritized.csv')
out_high = os.path.join(DATA_DIR, '04_high_priority.csv')
df.to_csv(out_all, index=False)
high.to_csv(out_high, index=False)
print(f"\n  Saved all to {out_all}")
print(f"  Saved {len(high)} high-priority to {out_high}")
