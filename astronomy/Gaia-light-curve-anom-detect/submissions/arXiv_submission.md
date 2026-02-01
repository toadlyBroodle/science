# arXiv Submission Guide

**Submission URL:** https://arxiv.org/submit

---

## Submission Details

**Category:** astro-ph.SR (Solar and Stellar Astrophysics)

**Title:** Characterization of a Dwarf Nova Candidate: TESS Reveals Outburst in the Ultra-Short Period Variable TIC 22888126

**Authors:** Landon Mutch; Opus 4.5

**Abstract:**
We report the characterization of TIC 22888126 (Gaia DR3 5947829831449229312) as a dwarf nova candidate based on archival TESS photometry showing a ~2.5 magnitude outburst. This object was previously catalogued in VSX as a generic variable ("VAR") with a 57.3-minute period, but its nature was never determined. Our machine learning analysis of Gaia DR3 variability statistics flagged this object as anomalous, prompting archival investigation. TESS Sector 13 data reveals classic dwarf nova outburst morphology: rapid rise (<1 day) and gradual decline (~5-7 days). Combined with archival X-ray detection (ROSAT) and the ultra-short orbital period (below the CV period gap), we propose this system as a strong dwarf nova candidate and recommend spectroscopic follow-up to confirm its cataclysmic variable nature.

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

\title{Characterization of a Dwarf Nova Candidate: TESS Reveals Outburst in the Ultra-Short Period Variable TIC 22888126}

\author{Landon Mutch}

\begin{abstract}
We report the characterization of TIC 22888126 (Gaia DR3 5947829831449229312) as a dwarf nova candidate based on archival TESS photometry showing a $\sim$2.5 magnitude outburst...
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
