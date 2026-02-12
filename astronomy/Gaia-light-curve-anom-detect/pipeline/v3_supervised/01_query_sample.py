#!/usr/bin/env python3
"""Step 1: Query Gaia DR3 for variable sources WITH XP spectra.

Selects two populations:
  A) Sources in the CMD MS-WD bridge region (where CVs live)
  B) High-variability sources (large photometric range)
Both require has_xp_continuous = True.
"""

import os
import numpy as np
from astroquery.gaia import Gaia
from config import CONFIG, DATA_DIR

os.makedirs(DATA_DIR, exist_ok=True)

print("=" * 70)
print("STEP 1: QUERY GAIA DR3 — VARIABLE SOURCES WITH XP SPECTRA")
print("=" * 70)

bridge = CONFIG['cmd_bridge']

# Query: variable sources in CMD bridge region with XP spectra
# The CMD bridge sits between the main sequence and white dwarf sequence.
# CVs occupy this region due to their composite SED (WD + disc + donor).
# We also grab high-variability sources outside the bridge.
query = f"""
SELECT TOP {CONFIG['sample_size']}
    vs.source_id,
    gs.ra, gs.dec,
    gs.phot_g_mean_mag,
    gs.bp_rp,
    gs.parallax, gs.parallax_error,
    gs.pmra, gs.pmdec,
    gs.teff_gspphot,
    gs.logg_gspphot,
    gs.ruwe,
    gs.phot_bp_rp_excess_factor,
    vs.mean_mag_g_fov,
    vs.std_dev_mag_g_fov,
    vs.range_mag_g_fov,
    vs.skewness_mag_g_fov,
    vs.kurtosis_mag_g_fov,
    vs.num_selected_g_fov,
    vs.mean_obs_time_g_fov,
    vs.time_duration_g_fov
FROM gaiadr3.vari_summary AS vs
JOIN gaiadr3.gaia_source AS gs ON vs.source_id = gs.source_id
WHERE gs.has_xp_continuous = 'True'
    AND gs.phot_g_mean_mag > {CONFIG['g_mag_min']}
    AND gs.phot_g_mean_mag < {CONFIG['g_mag_max']}
    AND vs.std_dev_mag_g_fov IS NOT NULL
    AND vs.num_selected_g_fov > 10
    AND gs.parallax IS NOT NULL
    AND gs.parallax > 0.1
    AND gs.bp_rp IS NOT NULL
ORDER BY vs.range_mag_g_fov DESC
"""

print(f"  Sample size: {CONFIG['sample_size']}")
print(f"  G: {CONFIG['g_mag_min']}–{CONFIG['g_mag_max']}")
print(f"  Requires: XP spectra, variability data, parallax > 0.1 mas")
print("  Ordered by: photometric range (descending)")
print("  Submitting async query...", flush=True)

job = Gaia.launch_job_async(query)
results = job.get_results()
df = results.to_pandas()

# Compute absolute magnitude
df['abs_mag_g'] = df['phot_g_mean_mag'] + 5 * np.log10(df['parallax'] / 100)

# Tag CMD bridge sources
in_bridge = (
    (df['bp_rp'] >= bridge['bp_rp_min']) &
    (df['bp_rp'] <= bridge['bp_rp_max']) &
    (df['abs_mag_g'] >= bridge['abs_g_min']) &
    (df['abs_mag_g'] <= bridge['abs_g_max'])
)
df['in_cmd_bridge'] = in_bridge

print(f"\n  Retrieved {len(df)} sources")
print(f"  G: {df['phot_g_mean_mag'].min():.1f} – {df['phot_g_mean_mag'].max():.1f}")
print(f"  BP-RP: {df['bp_rp'].min():.2f} – {df['bp_rp'].max():.2f}")
print(f"  Range: {df['range_mag_g_fov'].min():.2f} – {df['range_mag_g_fov'].max():.2f} mag")
print(f"  In CMD bridge: {in_bridge.sum()} / {len(df)}")
print(f"  With Teff: {df['teff_gspphot'].notna().sum()} / {len(df)}")

out = os.path.join(DATA_DIR, '01_sample.csv')
df.to_csv(out, index=False)
print(f"\n  Saved to {out}")
