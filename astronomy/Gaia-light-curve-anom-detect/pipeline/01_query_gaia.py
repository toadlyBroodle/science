#!/usr/bin/env python3
"""Step 1: Query Gaia DR3 for variable star candidates."""

import os
from astroquery.gaia import Gaia
from config import CONFIG, DATA_DIR

os.makedirs(DATA_DIR, exist_ok=True)

print("=" * 70)
print("STEP 1: QUERY GAIA DR3")
print("=" * 70)

query = f"""
SELECT TOP {CONFIG['sample_size']}
    vs.source_id,
    gs.ra, gs.dec,
    gs.phot_g_mean_mag,
    gs.bp_rp,
    gs.parallax, gs.parallax_error,
    gs.pmra, gs.pmdec,
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
WHERE gs.phot_g_mean_mag > {CONFIG['g_mag_min']}
    AND gs.phot_g_mean_mag < {CONFIG['g_mag_max']}
    AND vs.std_dev_mag_g_fov IS NOT NULL
    AND vs.num_selected_g_fov > 10
    AND gs.parallax IS NOT NULL
    AND gs.parallax > 0.1
ORDER BY vs.range_mag_g_fov DESC, vs.source_id ASC
"""

print(f"  Sample size: {CONFIG['sample_size']}")
print(f"  G: {CONFIG['g_mag_min']}-{CONFIG['g_mag_max']} (no color cut)")
print("  Submitting async query...", flush=True)

job = Gaia.launch_job_async(query)
results = job.get_results()
df = results.to_pandas()

print(f"  Retrieved {len(df)} sources")
print(f"  G: {df['phot_g_mean_mag'].min():.1f} - {df['phot_g_mean_mag'].max():.1f}")
print(f"  BP-RP: {df['bp_rp'].min():.2f} - {df['bp_rp'].max():.2f}")
print(f"  Range: {df['range_mag_g_fov'].min():.2f} - {df['range_mag_g_fov'].max():.2f} mag")

out = os.path.join(DATA_DIR, '01_gaia_sources.csv')
df.to_csv(out, index=False)
print(f"\n  Saved to {out}")
