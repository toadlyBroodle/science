#!/usr/bin/env python3
"""Step 3: Cross-match candidates with VSX, ROSAT, TIC, SIMBAD, Gaia vari, ZTF, GALEX."""

import os, signal
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.vizier import Vizier
from astroquery.simbad import Simbad
from astroquery.gaia import Gaia
from astroquery.mast import Catalogs
from config import CONFIG, DATA_DIR

TIMEOUT = CONFIG['query_timeout']


def timeout_handler(signum, frame):
    raise TimeoutError("Query timed out")


def with_timeout(func, timeout=TIMEOUT):
    """Run func with a Unix signal-based timeout."""
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
print("STEP 3: CROSS-MATCHING")
print("=" * 70)

df = pd.read_csv(os.path.join(DATA_DIR, '02_candidates.csv'))
n = len(df)
print(f"  Loaded {n} candidates\n")


# ==================== VSX ====================
print(f"--- VSX (Variable Star Index) ---")
vsx_types, vsx_names, vsx_periods, in_vsx = [], [], [], []

for i, row in df.iterrows():
    coord = SkyCoord(ra=row['ra']*u.deg, dec=row['dec']*u.deg, frame='icrs')
    try:
        result = with_timeout(lambda: Vizier(columns=['**'], row_limit=3).query_region(
            coord, radius=10*u.arcsec, catalog='B/vsx/vsx'))
        if result and len(result) > 0 and len(result[0]) > 0:
            r = result[0][0]
            vsx_types.append(str(r['Type']) if not np.ma.is_masked(r['Type']) else None)
            vsx_names.append(str(r['Name']) if not np.ma.is_masked(r['Name']) else None)
            vsx_periods.append(float(r['Period']) if 'Period' in r.colnames and not np.ma.is_masked(r['Period']) else None)
            in_vsx.append(True)
        else:
            vsx_types.append(None); vsx_names.append(None); vsx_periods.append(None); in_vsx.append(False)
    except:
        vsx_types.append(None); vsx_names.append(None); vsx_periods.append(None); in_vsx.append(False)
    if (i + 1) % 25 == 0 or i + 1 == n:
        print(f"  {i+1}/{n}", flush=True)

df['in_vsx'] = in_vsx; df['vsx_type'] = vsx_types; df['vsx_name'] = vsx_names; df['vsx_period'] = vsx_periods
print(f"  VSX matches: {sum(in_vsx)}/{n}\n")


# ==================== ROSAT ====================
print(f"--- ROSAT (X-ray) ---")
has_xray, xray_rates = [], []
rosat_cats = ['IX/10A/1rxs', 'IX/29/rass_fsc']

for i, row in df.iterrows():
    coord = SkyCoord(ra=row['ra']*u.deg, dec=row['dec']*u.deg, frame='icrs')
    found, rate = False, None
    for cat in rosat_cats:
        try:
            result = with_timeout(lambda c=cat: Vizier.query_region(
                coord, radius=CONFIG['rosat_search_radius']*u.arcsec, catalog=c))
            if result and len(result) > 0 and len(result[0]) > 0:
                found = True
                if 'Count' in result[0].colnames:
                    rate = float(result[0][0]['Count'])
                break
        except:
            pass
    has_xray.append(found); xray_rates.append(rate)
    if (i + 1) % 25 == 0 or i + 1 == n:
        print(f"  {i+1}/{n}", flush=True)

df['has_xray'] = has_xray; df['xray_count_rate'] = xray_rates
print(f"  X-ray detections: {sum(has_xray)}/{n}\n")


# ==================== TIC ====================
print(f"--- TIC (TESS Input Catalog) ---")
tic_ids = []

for i, row in df.iterrows():
    coord = SkyCoord(ra=row['ra']*u.deg, dec=row['dec']*u.deg, frame='icrs')
    try:
        result = with_timeout(lambda: Catalogs.query_region(
            coord, radius=CONFIG['tess_search_radius']*u.arcsec, catalog='TIC'))
        if result and len(result) > 0:
            tic_ids.append(int(result[0]['ID']))
        else:
            tic_ids.append(None)
    except:
        tic_ids.append(None)
    if (i + 1) % 25 == 0 or i + 1 == n:
        print(f"  {i+1}/{n}", flush=True)

df['tic_id'] = tic_ids
print(f"  TIC matches: {sum(1 for t in tic_ids if t)}/{n}\n")


# ==================== SIMBAD ====================
print(f"--- SIMBAD ---")

simbad_map = {}

# Batch TAP query using Gaia DR3 identifiers (not positional)
source_ids = [int(s) for s in df['source_id']]
id_strings = ', '.join(f"'Gaia DR3 {sid}'" for sid in source_ids)
tap_query = f"""
SELECT i.id, b.main_id, b.otype
FROM ident AS i
JOIN basic AS b ON i.oidref = b.oid
WHERE i.id IN ({id_strings})
"""

try:
    print(f"  SIMBAD TAP: querying {len(source_ids)} Gaia DR3 IDs...", flush=True)
    result = Simbad.query_tap(tap_query)
    if result and len(result) > 0:
        for row in result:
            gaia_str = str(row['id']).strip()
            sid = int(gaia_str.replace('Gaia DR3 ', ''))
            mid = str(row['main_id']).strip() if not np.ma.is_masked(row['main_id']) else None
            ot = str(row['otype']).strip() if not np.ma.is_masked(row['otype']) else None
            if mid and sid not in simbad_map:
                simbad_map[sid] = (mid, ot, None)
        print(f"  TAP: {len(simbad_map)} matches", flush=True)
    else:
        print(f"  TAP: no results", flush=True)
except Exception as e:
    print(f"  TAP failed ({e}), falling back to sequential...", flush=True)
    custom = Simbad()
    custom.add_votable_fields('otype', 'otypes')
    custom.TIMEOUT = 10

    for i, row in df.iterrows():
        sid = int(row['source_id'])
        if sid in simbad_map:
            continue
        try:
            def _q():
                r = custom.query_object(f"Gaia DR3 {sid}")
                if r is None:
                    c = SkyCoord(ra=row['ra']*u.deg, dec=row['dec']*u.deg, frame='icrs')
                    r = custom.query_region(c, radius=CONFIG['simbad_search_radius']*u.arcsec)
                return r
            result = with_timeout(_q, timeout=15)
            if result and len(result) > 0:
                cols = {c.lower(): c for c in result.colnames}
                def _v(variants):
                    for v in variants:
                        a = cols.get(v.lower())
                        if a:
                            val = result[0][a]
                            return str(val).strip() if not np.ma.is_masked(val) else None
                    return None
                mid = _v(['MAIN_ID', 'main_id'])
                if mid:
                    simbad_map[sid] = (mid, _v(['OTYPE', 'otype', 'main_type']),
                                       _v(['OTYPES', 'otypes', 'all_types']))
        except:
            pass
        if (i + 1) % 25 == 0 or i + 1 == n:
            print(f"  {i+1}/{n}", flush=True)

simbad_names, simbad_otypes, in_simbad = [], [], []
for _, row in df.iterrows():
    sid = int(row['source_id'])
    if sid in simbad_map:
        mid, ot, _ = simbad_map[sid]
        simbad_names.append(mid); simbad_otypes.append(ot); in_simbad.append(True)
    else:
        simbad_names.append(None); simbad_otypes.append(None); in_simbad.append(False)

df['simbad_name'] = simbad_names; df['simbad_otype'] = simbad_otypes; df['in_simbad'] = in_simbad
print(f"  SIMBAD matches: {sum(in_simbad)}/{n}\n")


# ==================== Gaia DR3 vari_classifier ====================
print(f"--- Gaia DR3 variability classifier ---")
gaia_ids = [str(int(s)) for s in df['source_id']]
id_list = ', '.join(gaia_ids)
vari_query = f"""
SELECT source_id, best_class_name, best_class_score
FROM gaiadr3.vari_classifier_result
WHERE source_id IN ({id_list})
"""
vari_map = {}
try:
    job = Gaia.launch_job(vari_query)
    vr = job.get_results()
    for row in vr:
        vari_map[int(row['source_id'])] = (str(row['best_class_name']), float(row['best_class_score']))
except Exception as e:
    print(f"  Error: {e}")

gaia_classes, gaia_scores = [], []
for _, row in df.iterrows():
    sid = int(row['source_id'])
    if sid in vari_map:
        gaia_classes.append(vari_map[sid][0]); gaia_scores.append(vari_map[sid][1])
    else:
        gaia_classes.append(None); gaia_scores.append(None)

df['gaia_vari_class'] = gaia_classes; df['gaia_vari_score'] = gaia_scores
print(f"  Classified: {sum(1 for c in gaia_classes if c)}/{n}\n")


# ==================== ZTF ====================
print(f"--- ZTF ---")
ztf_cats = ['J/ApJS/249/18', 'J/AJ/159/198', 'J/MNRAS/499/5782']
Vizier.ROW_LIMIT = 5

in_ztf, ztf_nobs, ztf_periods, ztf_mags = [], [], [], []

for i, row in df.iterrows():
    coord = SkyCoord(ra=row['ra']*u.deg, dec=row['dec']*u.deg, frame='icrs')
    found, nobs, period, mag = False, 0, None, None

    for cat in ztf_cats:
        try:
            result = with_timeout(lambda c=cat: Vizier(columns=['**'], row_limit=5, timeout=15).query_region(
                coord, radius=CONFIG['ztf_search_radius']*u.arcsec, catalog=c))
            if result and len(result) > 0 and len(result[0]) > 0:
                table = result[0]
                r0 = table[0]
                found = True
                for col in table.colnames:
                    cl = col.lower()
                    if cl in ('ng', 'nr', 'nobs', 'nepochs', 'numobs', 'ndet', 'n'):
                        try:
                            val = r0[col]
                            if not np.ma.is_masked(val): nobs += int(val)
                        except: pass
                    elif 'nobs' in cl or 'nepoch' in cl or 'ndet' in cl:
                        try:
                            val = r0[col]
                            if not np.ma.is_masked(val): nobs += int(val)
                        except: pass
                for col in table.colnames:
                    cl = col.lower()
                    if cl in ('per', 'period', 'p', 'per-g', 'per-r') or cl.startswith('per'):
                        try:
                            val = r0[col]
                            if not np.ma.is_masked(val) and float(val) > 0:
                                period = float(val); break
                        except: pass
                for col in table.colnames:
                    cl = col.lower()
                    if cl in ('gmag', 'rmag', 'magg', 'magr', 'mag'):
                        try:
                            val = r0[col]
                            if not np.ma.is_masked(val): mag = float(val); break
                        except: pass
                break
        except:
            pass

    in_ztf.append(found); ztf_nobs.append(nobs); ztf_periods.append(period); ztf_mags.append(mag)
    if (i + 1) % 25 == 0 or i + 1 == n:
        print(f"  {i+1}/{n}", flush=True)

df['in_ztf'] = in_ztf; df['ztf_nobs'] = ztf_nobs; df['ztf_period'] = ztf_periods; df['ztf_mag'] = ztf_mags
print(f"  ZTF matches: {sum(in_ztf)}/{n}\n")


# ==================== GALEX ====================
print(f"--- GALEX (UV) ---")
in_galex, galex_fuv, galex_nuv = [], [], []

for i, row in df.iterrows():
    coord = SkyCoord(ra=row['ra']*u.deg, dec=row['dec']*u.deg, frame='icrs')
    found, fuv, nuv = False, None, None
    try:
        result = with_timeout(lambda: Vizier.query_region(
            coord, radius=CONFIG['galex_search_radius']*u.arcsec, catalog='II/335/galex_ais'))
        if result and len(result) > 0 and len(result[0]) > 0:
            g = result[0][0]
            found = True
            fuv = float(g['FUVmag']) if 'FUVmag' in g.colnames and not np.ma.is_masked(g['FUVmag']) else None
            nuv = float(g['NUVmag']) if 'NUVmag' in g.colnames and not np.ma.is_masked(g['NUVmag']) else None
    except:
        pass
    in_galex.append(found); galex_fuv.append(fuv); galex_nuv.append(nuv)
    if (i + 1) % 25 == 0 or i + 1 == n:
        print(f"  {i+1}/{n}", flush=True)

df['in_galex'] = in_galex; df['galex_fuv'] = galex_fuv; df['galex_nuv'] = galex_nuv
print(f"  GALEX matches: {sum(in_galex)}/{n}\n")


# ==================== SAVE ====================
out = os.path.join(DATA_DIR, '03_crossmatched.csv')
df.to_csv(out, index=False)
print(f"Saved to {out}")

# Summary
print(f"\n{'='*70}")
print(f"CROSS-MATCH SUMMARY ({n} candidates)")
print(f"{'='*70}")
print(f"  VSX:    {sum(in_vsx)}")
print(f"  ROSAT:  {sum(has_xray)}")
print(f"  TIC:    {sum(1 for t in tic_ids if t)}")
print(f"  SIMBAD: {sum(in_simbad)}")
print(f"  Gaia V: {sum(1 for c in gaia_classes if c)}")
print(f"  ZTF:    {sum(in_ztf)}")
print(f"  GALEX:  {sum(in_galex)}")
