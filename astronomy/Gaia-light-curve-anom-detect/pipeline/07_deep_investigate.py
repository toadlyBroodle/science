#!/usr/bin/env python3
"""Step 7: Deep candidate investigation - extended Gaia params, X-ray catalogs, classification."""

import os, signal, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.gaia import Gaia
from astroquery.vizier import Vizier
from astropy.timeseries import LombScargle
from config import CONFIG, DATA_DIR, PLOT_DIR
from lc_utils import collapse_gaps, plot_lc, isolate_quiescent, detrend, mask_tess_systematics

TIMEOUT = CONFIG['query_timeout']


def timeout_handler(signum, frame):
    raise TimeoutError("Query timed out")


def with_timeout(func, timeout=TIMEOUT):
    old = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    try:
        return func()
    except TimeoutError:
        return None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


print("=" * 70)
print("STEP 7: DEEP CANDIDATE INVESTIGATION")
print("=" * 70)

df = pd.read_csv(os.path.join(DATA_DIR, '04_high_priority.csv'))
n = len(df)
print(f"  {n} high-priority candidates\n")


# --- Extended Gaia DR3 parameters (single batch query) ---
print("--- Extended Gaia DR3 parameters ---")
gaia_ids = [str(int(s)) for s in df['source_id']]
id_list = ', '.join(gaia_ids)

ext_query = f"""
SELECT source_id, teff_gspphot, logg_gspphot, mh_gspphot,
       radial_velocity, ruwe, astrometric_excess_noise,
       phot_bp_rp_excess_factor, non_single_star
FROM gaiadr3.gaia_source
WHERE source_id IN ({id_list})
"""

ext_map = {}
try:
    job = Gaia.launch_job(ext_query)
    ext = job.get_results()
    for row in ext:
        sid = int(row['source_id'])
        ext_map[sid] = {
            'teff': float(row['teff_gspphot']) if not np.ma.is_masked(row['teff_gspphot']) else None,
            'logg': float(row['logg_gspphot']) if not np.ma.is_masked(row['logg_gspphot']) else None,
            'rv': float(row['radial_velocity']) if not np.ma.is_masked(row['radial_velocity']) else None,
            'ruwe': float(row['ruwe']) if not np.ma.is_masked(row['ruwe']) else None,
            'nss': int(row['non_single_star']) if not np.ma.is_masked(row['non_single_star']) else None,
        }
    print(f"  Retrieved for {len(ext_map)}/{n} sources")
except Exception as e:
    print(f"  Error: {e}")

# Assign to dataframe
for col in ['teff', 'logg', 'rv', 'ruwe', 'nss']:
    df[f'gaia_{col}'] = [ext_map.get(int(row['source_id']), {}).get(col) for _, row in df.iterrows()]


# --- Per-candidate catalog searches ---
print("\n--- CV-specific catalog searches ---")

xmm_flags, erosita_flags, sdss_flags = [], [], []
assessments, assessment_notes = [], []

for i, (_, row) in enumerate(df.iterrows()):
    gaia_id = int(row['source_id'])
    coord = SkyCoord(ra=row['ra']*u.deg, dec=row['dec']*u.deg, frame='icrs')
    ep = ext_map.get(gaia_id, {})

    # XMM-Newton
    xmm = False
    try:
        result = with_timeout(lambda: Vizier(columns=['*'], row_limit=3, timeout=10).query_region(
            coord, radius=10*u.arcsec, catalog='IX/68/xmm4d14s'))
        if result and len(result) > 0 and len(result[0]) > 0:
            xmm = True
    except:
        pass
    xmm_flags.append(xmm)

    # eROSITA
    erosita = False
    try:
        result = with_timeout(lambda: Vizier(columns=['*'], row_limit=3, timeout=10).query_region(
            coord, radius=15*u.arcsec, catalog='J/A+A/661/A1'))
        if result and len(result) > 0 and len(result[0]) > 0:
            erosita = True
    except:
        pass
    erosita_flags.append(erosita)

    # SDSS spectroscopy
    sdss = False
    try:
        result = with_timeout(lambda: Vizier(columns=['*'], row_limit=3, timeout=10).query_region(
            coord, radius=3*u.arcsec, catalog='V/154/sdss16'))
        if result and len(result) > 0 and len(result[0]) > 0:
            sdss = True
    except:
        pass
    sdss_flags.append(sdss)

    # --- Classify: known CV vs novel candidate ---
    # Check if already classified as CV in any catalog
    known_cv = False
    known_labels = []

    gc = str(row.get('gaia_vari_class', ''))
    if gc and gc not in ['None', 'nan', '']:
        if any(c in gc for c in ['CV', 'SYST']):
            known_labels.append(f'Gaia:{gc}')

    vsx_type = str(row.get('vsx_type', ''))
    cv_vsx_types = ['UG', 'UGSU', 'UGSS', 'UGZ', 'AM', 'DQ', 'NL', 'N:', 'NA', 'NB', 'NC', 'DN']
    if any(vsx_type.startswith(t) for t in cv_vsx_types):
        known_labels.append(f'VSX:{vsx_type}')

    simbad_otype = str(row.get('simbad_otype', ''))
    if any(t in simbad_otype for t in ['CV', 'No*', 'DN', 'DQ', 'AM', 'NL']):
        known_labels.append(f'SIMBAD:{simbad_otype}')

    if known_labels:
        known_cv = True

    # --- Physical CV indicators (for novel candidates) ---
    indicators = []

    bp_rp = row.get('bp_rp')
    if pd.notna(bp_rp) and bp_rp < 0.5:
        indicators.append('blue')

    has_xray = row.get('has_xray', False) or xmm
    has_fuv = pd.notna(row.get('galex_fuv')) and row.get('galex_fuv') == row.get('galex_fuv')
    if has_xray and has_fuv:
        indicators.append('xray+uv')
    elif has_xray:
        indicators.append('xray')
    elif has_fuv:
        indicators.append('fuv')

    ruwe = ep.get('ruwe')
    if ruwe and ruwe > 1.4:
        indicators.append('binary')

    teff = ep.get('teff')
    if teff and teff > 10000:
        indicators.append('hot')

    amp = row.get('amplitude', 0) or 0
    if amp > 2.0:
        indicators.append('high_amp')

    skew = row.get('skewness_mag_g_fov', 0) or 0
    if abs(skew) > 1:
        indicators.append('outburst')

    if not row.get('in_simbad', True):
        indicators.append('novel')
    if not row.get('in_vsx', True):
        indicators.append('no_vsx')

    # --- Assessment ---
    if known_cv:
        assess = f'KNOWN CV ({", ".join(known_labels)})'
    else:
        n_ind = len(indicators)
        if n_ind >= 4:
            assess = 'STRONG NEW CANDIDATE'
        elif n_ind >= 3:
            assess = 'MODERATE NEW CANDIDATE'
        elif n_ind >= 2:
            assess = 'WEAK NEW CANDIDATE'
        else:
            assess = 'UNCERTAIN'

    assessments.append(assess)
    assessment_notes.append(', '.join(indicators))

    if (i + 1) % 25 == 0 or i + 1 == n:
        print(f"  {i+1}/{n}", flush=True)

df['xmm_detected'] = xmm_flags
df['erosita_detected'] = erosita_flags
df['sdss_spec'] = sdss_flags
df['assessment'] = assessments
df['assessment_notes'] = assessment_notes


# --- Summary ---
known = df[df['assessment'].str.startswith('KNOWN')]
novel = df[~df['assessment'].str.startswith('KNOWN')]

print(f"\n{'='*70}")
print("DEEP INVESTIGATION SUMMARY")
print(f"{'='*70}")

# Known CVs (validation)
print(f"\n--- KNOWN CVs ({len(known)}) - pipeline validation ---")
for _, row in known.iterrows():
    sid = int(row['source_id'])
    name = row.get('simbad_name', '') or row.get('vsx_name', '')
    print(f"  {sid}  {name:<20} {row['assessment']}")

# Novel candidates (the science)
print(f"\n--- NOVEL CANDIDATES ({len(novel)}) - follow-up targets ---")
print(f"{'Gaia DR3 ID':<22} {'Teff':>6} {'RUWE':>6} {'XMM':>4} {'eRO':>4} {'SDSS':>4} {'Assessment':<24} Indicators")
print("-" * 110)
for _, row in novel.iterrows():
    sid = int(row['source_id'])
    teff_s = f"{row['gaia_teff']:.0f}" if pd.notna(row.get('gaia_teff')) else '---'
    ruwe_s = f"{row['gaia_ruwe']:.2f}" if pd.notna(row.get('gaia_ruwe')) else '---'
    xmm_s = 'Y' if row['xmm_detected'] else 'N'
    ero_s = 'Y' if row['erosita_detected'] else 'N'
    sdss_s = 'Y' if row['sdss_spec'] else 'N'
    notes = row.get('assessment_notes', '')
    print(f"{sid:<22} {teff_s:>6} {ruwe_s:>6} {xmm_s:>4} {ero_s:>4} {sdss_s:>4} {row['assessment']:<24} {notes}")

# Assessment breakdown
print(f"\nBreakdown:")
for assess, count in pd.Series(assessments).value_counts().items():
    print(f"  {assess}: {count}")

# Links for novel strong candidates
novel_strong = novel[novel['assessment'].str.contains('STRONG|MODERATE')]
if len(novel_strong) > 0:
    print(f"\nTOP NEW CANDIDATES - investigation links:")
    for _, row in novel_strong.iterrows():
        ra, dec = row['ra'], row['dec']
        name = row.get('simbad_name', 'unknown')
        print(f"  Gaia DR3 {int(row['source_id'])} ({name})")
        print(f"    Indicators: {row.get('assessment_notes', '')}")
        print(f"    Aladin: https://aladin.cds.unistra.fr/AladinLite/?target={ra:.6f}+{dec:.6f}&fov=0.05")
        print(f"    ESASky: https://sky.esa.int/?target={ra:.6f}%20{dec:.6f}&hips=DSS2+color&fov=0.05")

# Save
out = os.path.join(DATA_DIR, '07_deep_investigated.csv')
df.to_csv(out, index=False)
print(f"\n  Saved to {out}")


# --- Plots for novel candidates only ---
os.makedirs(PLOT_DIR, exist_ok=True)

lc_path = os.path.join(DATA_DIR, '05_lc_data.pkl')
pr_path = os.path.join(DATA_DIR, '06_period_results.pkl')

if not os.path.exists(lc_path):
    print("\n  No light curve data found, skipping novel candidate plots")
else:
    with open(lc_path, 'rb') as f:
        lc_data = pickle.load(f)
    period_results = {}
    if os.path.exists(pr_path):
        with open(pr_path, 'rb') as f:
            period_results = pickle.load(f)

    # Get TIC IDs for novel candidates
    novel_tics = []
    for _, row in novel.iterrows():
        tic = row.get('tic_id')
        if pd.notna(tic) and int(tic) in lc_data:
            novel_tics.append(int(tic))

    n_novel = len(novel_tics)
    print(f"\n--- Plotting {n_novel} novel candidates with TESS data ---")

    if n_novel > 0:

        # --- Light curves ---
        fig, axes = plt.subplots(n_novel, 1, figsize=(16, 4 * n_novel), squeeze=False)
        for idx, tic_id in enumerate(novel_tics):
            ax = axes[idx, 0]
            lc = lc_data[tic_id]
            t, flux = lc['time'], lc['flux']
            sectors = lc.get('sectors', [lc.get('sector', '?')])
            sector_str = ','.join(str(s) for s in sectors)

            # Get assessment info for title
            r = novel[novel['tic_id'] == tic_id]
            if len(r) == 0:
                r = novel[novel['tic_id'] == float(tic_id)]
            assess = r.iloc[0]['assessment'] if len(r) > 0 else ''
            notes = r.iloc[0]['assessment_notes'] if len(r) > 0 else ''
            gaia_id = lc['gaia_id']
            title = f'TIC {tic_id} | Gaia DR3 {gaia_id} | S{sector_str}\n{assess} [{notes}]'

            plot_lc(ax, t, flux, title, sector_str)

        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, 'novel_lightcurves.png'), dpi=150)
        plt.close()
        print(f"  Plot: {PLOT_DIR}/novel_lightcurves.png")

        # --- Period analysis for novel candidates ---
        novel_with_periods = [t for t in novel_tics if t in period_results]
        n_per = len(novel_with_periods)
        if n_per > 0:
            fig2, axes2 = plt.subplots(n_per, 2, figsize=(16, 4.5 * n_per), squeeze=False)
            for idx, tic_id in enumerate(novel_with_periods):
                lc = lc_data[tic_id]
                t, flux = lc['time'], lc['flux']
                pr = period_results[tic_id]

                # Reuse step 6 functions for quiescent isolation + detrending
                t_q, flux_q, _, _ = isolate_quiescent(t, flux)
                if len(t_q) < 20:
                    continue
                t_q, flux_detrend = detrend(t_q, flux_q)

                # Recompute periodogram (same approach as step 6)
                dt = np.diff(t_q)
                dt_clean = dt[dt > 0]
                med_cad = np.median(dt_clean) if len(dt_clean) > 0 else 30.0 / (24 * 60)
                min_period = 2 * med_cad
                max_period = min(0.5, (t_q[-1] - t_q[0]) / 3)
                if min_period < max_period:
                    t_ls, f_ls = t_q, flux_detrend
                    if len(t_q) > 30000:
                        step = len(t_q) // 30000
                        t_ls, f_ls = t_q[::step], flux_detrend[::step]
                    freq = np.geomspace(1/max_period, 1/min_period, 5000)
                    ls = LombScargle(t_ls, f_ls)
                    power_raw = ls.power(freq)
                    periods = 1 / freq
                    power = mask_tess_systematics(periods, power_raw)

                    # Detrended LC (clip y-axis to quiescent range)
                    ax_lc = axes2[idx, 0]
                    ax_lc.scatter(t_q, flux_detrend, s=1, alpha=0.5, c='steelblue')
                    ax_lc.axhline(1, color='gray', ls='--', alpha=0.3)
                    y_lo, y_hi = np.percentile(flux_detrend, [0.5, 99.5])
                    y_pad = (y_hi - y_lo) * 0.15
                    ax_lc.set_ylim(y_lo - y_pad, y_hi + y_pad)
                    ax_lc.set_xlabel('Time (BTJD)')
                    ax_lc.set_ylabel('Detrended Flux')
                    ax_lc.set_title(f'TIC {tic_id} (quiescent, detrended)', fontsize=10)

                    # Periodogram
                    ax_ls = axes2[idx, 1]
                    best_period = pr['best_period']
                    fap = pr['fap']
                    vsx_period = pr.get('vsx_period')
                    ax_ls.semilogx(periods * 24 * 60, power, 'k-', lw=0.5)
                    ax_ls.axvline(best_period * 24 * 60, color='red', ls='--', alpha=0.7,
                                  label=f'Best: {best_period*24*60:.1f}m')
                    if vsx_period:
                        ax_ls.axvline(vsx_period * 24 * 60, color='blue', ls=':', alpha=0.7,
                                      label=f'VSX: {vsx_period*24*60:.1f}m')
                    ax_ls.set_xlabel('Period (min, log)')
                    ax_ls.set_ylabel('L-S Power')
                    ax_ls.set_title(f'Periodogram (FAP={fap:.1e})', fontsize=10)
                    ax_ls.legend(fontsize=8)

            plt.tight_layout()
            plt.savefig(os.path.join(PLOT_DIR, 'novel_periods.png'), dpi=150)
            plt.close()
            print(f"  Plot: {PLOT_DIR}/novel_periods.png")
        else:
            print("  No novel candidates with period results")
    else:
        print("  No novel candidates with TESS light curves")
