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

## Structure

```
science/
├── astronomy/
│   └── Gaia-light-curve-anom-detect/
│       ├── Gaia_LightCurve_Anomaly_Detection.ipynb
│       ├── research_note_tic22888126_revised.md
│       └── figs/
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
