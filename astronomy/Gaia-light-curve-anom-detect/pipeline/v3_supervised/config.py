"""Shared configuration for the CV Hunter v3 supervised pipeline."""

import os
_BASE = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    # Sample selection
    'sample_size': 8000,
    'g_mag_min': 10.0,
    'g_mag_max': 20.0,

    # CMD bridge region (MS-WD gap where CVs live)
    # Approximate polygon in (BP-RP, M_G) space
    'cmd_bridge': {
        'bp_rp_min': -0.6,
        'bp_rp_max': 1.8,
        'abs_g_min': 4.0,
        'abs_g_max': 13.0,
    },

    # Classification
    'contamination': 0.05,      # 5% anomaly fraction (legacy, unused)
    'n_estimators': 300,

    # Spectral feature extraction
    'xp_batch_size': 500,       # sources per GaiaXPy batch call

    # Cross-match radii (arcsec)
    'rosat_search_radius': 30,
    'galex_search_radius': 5,
    'simbad_search_radius': 5,

    'query_timeout': 30,

    # TESS follow-up
    'n_tess_candidates': 20,
}

DATA_DIR = os.path.join(_BASE, 'data')
PLOT_DIR = os.path.join(_BASE, '..', '..', 'figs', 'v3_supervised')
