"""Shared configuration for the CV Hunter pipeline."""

CONFIG = {
    'sample_size': 5000,
    'g_mag_min': 10.0,
    'g_mag_max': 18.5,
    'anomaly_percentile': 5,
    'anomaly_score_min': 0.4,
    'require_consensus': True,
    'priority_score_min': 10.0,
    'tess_search_radius': 21,      # arcsec
    'rosat_search_radius': 30,
    'simbad_search_radius': 5,
    'ztf_search_radius': 3,
    'galex_search_radius': 5,
    'query_timeout': 30,           # seconds per network query
}

DATA_DIR = 'data'
PLOT_DIR = '../figs/pipeline'
