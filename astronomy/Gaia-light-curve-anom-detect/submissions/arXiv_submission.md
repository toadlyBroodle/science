# arXiv Submission Guide

**Submission URL:** https://arxiv.org/submit

---

## Submission Details

**Category:** astro-ph.SR (Solar and Stellar Astrophysics)

**Title:** Characterization of TIC 22888126 as an Active Dwarf Nova Candidate from TESS Photometry

**Authors:** Landon Mutch; Opus 4.5

**Abstract:**
We characterize TIC 22888126 (Gaia DR3 5947829831449228800) as an active dwarf nova candidate based on TESS photometry revealing seven distinct outbursts across 6 years (Sectors 13, 39, 66, 93). Outburst amplitudes range from 1--5 mag with classic dwarf nova morphology. A Lomb-Scargle analysis of ~23,500 background-subtracted quiescent TESS data points finds a candidate photometric period of ~90 min (FAP = 2.5e-85), consistent across three of four sectors. A 57.3-min period reported by Gaia DR3 short-timescale variability analysis is not recovered (FAP = 1) and is likely a sampling alias. If the 90-min period reflects the orbital period, this system lies at the upper edge of the CV period gap. Combined with ROSAT X-ray detection, we propose this as a strong dwarf nova candidate and recommend spectroscopic follow-up.

---

## Files to Upload

1. **Main paper:** Convert `research_note_tic22888126_revised.md` to LaTeX or PDF
2. **Figures:** All files from `figs/` directory

---

## Conversion to LaTeX

For arXiv, you'll need LaTeX format. Use this template:

```latex
\documentclass[twocolumn]{aastex631}

\begin{document}

\title{Characterization of TIC 22888126 as an Active Dwarf Nova Candidate from TESS Photometry}

\author{Landon Mutch}

\begin{abstract}
We characterize TIC 22888126 (Gaia DR3 5947829831449228800) as an active dwarf nova candidate based on TESS photometry revealing seven outbursts across 6 years with 1--5 mag amplitudes. A candidate $\sim$90-min photometric period is detected (FAP $= 2.5 \times 10^{-85}$). The Gaia/VSX 57.3-min period is not recovered...
\end{abstract}

\keywords{cataclysmic variables --- novae, dwarf --- methods: statistical --- surveys}

\section{Introduction}
...

\end{document}
```

---

## Alternative: PDF Upload

arXiv accepts PDF directly. To convert from Markdown:

```bash
# Using pandoc
pandoc research_note_tic22888126_revised.md -o arxiv_submission.pdf

# Or use a Markdown-to-PDF converter online
```

---

## Submission Steps

1. **Create arXiv account** at https://arxiv.org/user/register
2. **Start new submission** at https://arxiv.org/submit
3. **Select category:** astro-ph.SR
4. **Upload files:** LaTeX source or PDF + figures
5. **Enter metadata:** Title, abstract, authors
6. **Preview and submit**
7. **Wait for processing:** Usually 1-2 days for first submission

---

## Cross-listing

Consider cross-listing to:
- astro-ph.IM (Instrumentation and Methods) - for the ML aspect
- astro-ph.GA (Astrophysics of Galaxies) - if galactic context relevant

---

## Expected Timeline

- Submission → Announcement: 1-2 business days
- arXiv ID assigned immediately upon acceptance
- Paper appears in next daily listing

---

## After Posting

1. Share arXiv link on social media / astronomy forums
2. Submit RNAAS version (can reference arXiv)
3. Monitor for community feedback
4. Update with v2 if needed after spectroscopic follow-up
