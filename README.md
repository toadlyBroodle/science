# Science Projects

A collection of scientific research projects using data science and machine learning.

## Projects

### [Gaia Light Curve Anomaly Detection](https://github.com/toadlyBroodle/science/tree/main/astronomy/Gaia-light-curve-anom-detect)

Machine learning-based discovery of overlooked variable stars in Gaia DR3.

**Key Result:** Characterized TIC 22888126 as a dwarf nova with a 57-minute orbital period (below the CV period gap) using TESS archival photometry.

- **Method:** Isolation Forest anomaly detection on Gaia DR3 light curve statistics
- **Data:** Gaia DR3 `vari_summary`, TESS FFI photometry, ROSAT X-ray
- **Status:** Draft research note ready for submission

See the [research note](https://github.com/toadlyBroodle/science/blob/main/astronomy/Gaia-light-curve-anom-detect/research_note_tic22888126_revised.md) for details.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/toadlyBroodle/science/blob/main/astronomy/Gaia-light-curve-anom-detect/Gaia_LightCurve_Anomaly_Detection.ipynb)

### [Longevity Research — Knowledge Tree](https://github.com/toadlyBroodle/science/tree/main/longevity)

A Karpathy-style markdown knowledge base of computational entry points
into longevity / reverse-aging research, with TF-IDF + wikilink-graph
indexing and a linter to keep it consistent.

- **Coverage:** 40 curated sources, 32 topics, 1 tier-ranked analysis
- **Pipeline:** `download.py` (HTML/PDF) → `convert.py` (markdown) →
  `index.py` (TF-IDF + graph) → `lint.py` (consistency checks)
- **Licensing:** every source tagged in `sources.json`; auto-generated
  [`LICENSES.md`](LICENSES.md) classifies the 40 sources by
  redistributability

Quick start:

```bash
cd longevity
pip install requests beautifulsoup4 markdownify trafilatura scikit-learn lxml
sudo apt-get install -y poppler-utils   # for pdftotext

# Build the index and lint the wiki (no network needed)
python3 scripts/index.py
python3 scripts/lint.py

# Download primary sources (needs unrestricted HTTP egress) and convert
python3 scripts/download.py
python3 scripts/convert.py

# Refresh per-source license tags + regenerate LICENSES.md
python3 scripts/licenses.py all
```

Read the wiki: start at [`longevity/wiki/index.md`](longevity/wiki/index.md)
or [`longevity/wiki/analysis/promising-reverse-aging.md`](longevity/wiki/analysis/promising-reverse-aging.md).
Full documentation: [`longevity/README.md`](longevity/README.md).

## Structure

```
science/
├── astronomy/
│   └── Gaia-light-curve-anom-detect/
│       ├── Gaia_LightCurve_Anomaly_Detection.ipynb
│       ├── research_note_tic22888126_revised.md
│       └── figs/
├── longevity/
│   ├── README.md            # full pipeline + querying docs
│   ├── sources.json         # 40-source manifest
│   ├── scripts/             # download, convert, index, lint, licenses
│   └── wiki/                # papers/, topics/, analysis/, index.md
├── LICENSES.md              # generated; per-source license table
└── README.md
```

## Requirements

- Python 3.10+
- astroquery
- astropy
- scikit-learn
- matplotlib
- pandas
- numpy

## License

MIT
