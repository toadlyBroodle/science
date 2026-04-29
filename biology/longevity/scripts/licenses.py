"""Assign + summarize licenses for every source in sources.json.

Two responsibilities:

1. ``apply`` — fill a ``license`` (and optional ``license_note``) field on
   each source in ``sources.json`` based on an id-keyed map maintained
   here. Run this after adding new sources to keep the manifest in sync.
2. ``generate`` — render a ``LICENSES.md`` table at ``biology/``
   (one level up) and inside ``biology/longevity/``, grouping
   sources by license and flagging which are redistributable.

Idempotent. Safe to re-run. CLI:

    python scripts/licenses.py apply
    python scripts/licenses.py generate
    python scripts/licenses.py all     # both
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "sources.json"
# Parent LICENSES.md lives one level up (biology/ in the science/ repo).
REPO_LICENSES_MD = ROOT.parent / "LICENSES.md"
# Per-wiki copy lives alongside the manifest for convenience.
WIKI_LICENSES_MD = ROOT / "LICENSES.md"

# --------------------------------------------------------------------
# License categories used in this wiki.
# --------------------------------------------------------------------
# Fully redistributable → full text can be re-hosted with attribution.
REDISTRIBUTABLE = {
    "CC-BY-4.0",
    "CC-BY-3.0",
    "CC0",
    "public-domain",
}
# Redistributable with restrictions. ``LICENSES.md`` calls these out.
RESTRICTED = {
    "CC-BY-NC-4.0",
    "CC-BY-SA-4.0",
    "CC-BY-NC-SA-4.0",
    "CC-BY-ND-4.0",
}
# Not redistributable — paywalled, all rights reserved, etc.
NON_REDISTRIBUTABLE = {
    "ARR",
    "website-tos",
}
# Needs per-paper check (preprint with author-selectable license).
VARIES = {
    "preprint-author-choice",
    "unknown",
}


def category(lic: str) -> str:
    if lic in REDISTRIBUTABLE:
        return "redistributable"
    if lic in RESTRICTED:
        return "restricted"
    if lic in NON_REDISTRIBUTABLE:
        return "not-redistributable"
    if lic in VARIES:
        return "varies"
    return "unknown"


# --------------------------------------------------------------------
# Per-source license map. Keys = source id; value = (license, note).
# Note may be empty string. Update this table when adding sources.
# --------------------------------------------------------------------
LICENSE_MAP: dict[str, tuple[str, str]] = {
    "biomarkers-aging-challenge": ("website-tos", "Consortium portal; subject to agingconsortium.org site terms."),
    "nunez2024-open-competition": ("CC-BY-4.0", "PMC OA primary (PMC11565782); bioRxiv preprint is author-choice."),
    "computagebench": ("preprint-author-choice", "bioRxiv primary; KDD 2025 proceedings may differ. GitHub code usually MIT/Apache."),
    "nc-2025-14clocks": ("CC-BY-4.0", "Nature Communications (fully OA)."),
    "deep-aging-clocks-review-2025": ("ARR", "Elsevier (Ageing Research Reviews). Not redistributable."),
    "epinflammage": ("CC-BY-4.0", "MDPI IJMS (fully OA)."),
    "pathwayage": ("CC-BY-4.0", "Lancet eBioMedicine (fully OA)."),
    "itp-nia": ("public-domain", "US Government (NIA/NIH) web page."),
    "itp-mpd-portal": ("website-tos", "Jackson Lab Mouse Phenome Database site terms."),
    "dr-960-mice-nature-2024": ("CC-BY-4.0", "PMC author-manuscript (NIH public access); Nature version is paywalled."),
    "itp-sex-specific-2025": ("CC-BY-4.0", "npj Aging (fully OA)."),
    "network-repurposing-aging": ("preprint-author-choice", "arXiv default license + PMC version; check arXiv abstract page."),
    "smerbarreto2023-senolytics": ("CC-BY-4.0", "Nature Communications (fully OA). Zenodo code typically MIT-like; check repo."),
    "senolytic-predictor-2025": ("CC-BY-4.0", "MDPI Molecules (fully OA)."),
    "mesenchymal-drift-cell-2025": ("ARR", "Cell (Elsevier). Paywalled; no known preprint at time of writing."),
    "organ-proteomic-clocks-2025": ("ARR", "Nature Aging. Paywalled unless CC-BY paid; verify per-article."),
    "ukb-nmr-metabolomic-2024": ("CC-BY-4.0", "Nature Communications (fully OA)."),
    "ai-aging-review-2026": ("CC-BY-4.0", "Frontiers in Aging (fully OA)."),
    "shift-sb000-2025": ("preprint-author-choice", "bioRxiv; verify Shift Bioscience's chosen license on the abstract page."),
    "clockbase-agent-2025": ("CC-BY-4.0", "PMC OA primary (PMC12667862); bioRxiv preprint author-choice."),
    "singular-rejuv-atlas-2024": ("CC-BY-3.0", "Aging (Albany NY) — CC-BY-3.0 by default."),
    "agextend-2025": ("ARR", "Nature Aging. Paywalled unless CC-BY paid."),
    "scbayesage-2025": ("preprint-author-choice", "bioRxiv; verify author license selection."),
    "yang-chemical-cocktails-2023": ("CC-BY-3.0", "Aging (Albany NY) — CC-BY-3.0."),
    "chemical-reprog-lifespan-2025": ("CC-BY-4.0", "EMBO Mol Med (fully OA) + PMC."),
    "lipid-droplets-reprog-2025": ("preprint-author-choice", "bioRxiv primary; Wiley Aging Cell version may be paywalled."),
    "paine-partial-reprog-review-2024": ("ARR", "Wiley Aging Cell. Check OA status per-article; often paywalled."),
    "reprogramming-rejuv-review-ncomms-2024": ("CC-BY-4.0", "Nature Communications (fully OA)."),
    "aav-osk-lifespan-2024": ("CC-BY-4.0", "PMC author-manuscript (PMC10909732); Liebert Cellular Reprogramming paywalled."),
    "pedf-parabiosis-2024": ("CC-BY-4.0", "PMC OA primary (PMC11092633); bioRxiv preprint author-choice."),
    "fgf17-young-csf-2022": ("ARR", "Nature (main). Paywalled; no known preprint."),
    "hcpb-review-2024": ("CC-BY-4.0", "npj Aging (fully OA)."),
    "senolytic-mci-ebiomed-2025": ("CC-BY-4.0", "Lancet eBioMedicine (fully OA)."),
    "senolytic-methylation-2024": ("CC-BY-3.0", "Aging (Albany NY) — CC-BY-3.0."),
    "nr-longcovid-2025": ("CC-BY-4.0", "Lancet eClinicalMedicine (fully OA) + PMC."),
    "pearl-rapamycin-2025": ("CC-BY-4.0", "Frontiers in Aging + PMC; medRxiv preprint author-choice."),
    "urolithin-a-immune-2025": ("ARR", "Nature Aging. Verify CC-BY OA status per-article."),
    "tert-knockin-2025": ("ARR", "Wiley Aging Cell. Verify CC-BY OA status per-article; some articles are OA."),
    "adsc-exosomes-2022": ("CC-BY-4.0", "Science Advances (fully OA)."),
    "xprize-healthspan": ("website-tos", "XPRIZE Foundation website; subject to xprize.org site terms."),
    "il11-inhibition-2024": ("CC-BY-4.0", "Nature (CC-BY by author choice); PMC mirror PMC11291288."),
    "klotho-skl-aav-2025": ("CC-BY-NC-4.0", "Molecular Therapy article CC-BY-NC-ND-4.0 (PMC author manuscript OA); classified here as restricted (NC)."),
    "retro-precision-reprog-2025": ("ARR", "Liebert Cellular Reprogramming. Paywalled; verify per-article OA status."),
    "longevity-llm-2026": ("preprint-author-choice", "bioRxiv preprint (Insilico Medicine); verify license on abstract page."),
    "longevity-bench-2026": ("preprint-author-choice", "bioRxiv preprint (Insilico Medicine, Abu Dhabi); verify license on abstract page."),
    "scageclock-2026": ("CC-BY-4.0", "npj Aging (fully OA)."),
    "plasma-proteomics-brain-immune-2025": ("ARR", "Nature Medicine (paywalled unless CC-BY paid); bioRxiv preprint author-choice."),
    "plasmapheresis-aging-trial-2025": ("CC-BY-NC-4.0", "Scientific Reports article notes CC-BY-NC-ND-4.0; classified here as restricted (NC). PMC OA primary."),
    "senolytic-cart-upar-2024": ("CC-BY-4.0", "Nature Aging (CC-BY-4.0 by author choice); PMC mirror PMC10950785."),
    "mouse-strains-osk-induction-2025": ("CC-BY-4.0", "Cell Reports (fully OA); bioRxiv preprint."),
    "organ-dedifferentiation-review-2025": ("CC-BY-4.0", "Aging Cell (CC-BY-4.0 by author choice); PMC OA primary."),
    "anti-upar-cart-intestinal-2025": ("ARR", "Nature Aging. Verify CC-BY OA status per-article; PMC mirror PMC12823409."),
    "tpe-ivig-biological-age-rct-2025": ("CC-BY-4.0", "Aging Cell (CC-BY-4.0); PMC OA primary."),
    "trametinib-rapamycin-itp-2025": ("CC-BY-4.0", "Nature Aging (CC-BY-4.0); PMC OA primary."),
    "spatial-aging-clocks-brain-2024": ("CC-BY-NC-4.0", "Nature article CC-BY-NC-ND-4.0; classified as restricted (NC). PMC author manuscript OA."),
    "bcl-xl-protac-753b-senolytic-2025": ("ARR", "Nature Aging (paywalled); PMC author manuscript OA (PMC12683667)."),
    "gpld1-tnap-brain-vasculature-2026": ("ARR", "Cell (Elsevier paywall); PMC author manuscript OA (PMC13070421)."),
    "ipsc-mononuclear-phagocyte-aging-brain-2025": ("CC-BY-4.0", "Advanced Science (Wiley OA, CC-BY-4.0); PMC OA mirror."),
    "scimmuaging-immune-clocks-2025": ("CC-BY-4.0", "Nature Aging (CC-BY-4.0); PMC OA mirror."),
    "x-atlas-orion-perturbseq-2025": ("preprint-author-choice", "bioRxiv preprint; dataset under non-commercial license."),
    "semaglutide-glp1-epigenetic-age-rct-2025": ("preprint-author-choice", "medRxiv preprint; verify license on abstract page."),
    "lifestyle-atlas-tirolgesund-2025": ("preprint-author-choice", "bioRxiv preprint; verify license on abstract page."),
    "mandsager-2018-vo2max-mortality": ("CC-BY-4.0", "JAMA Network Open (fully OA, CC-BY-4.0); PMC OA primary."),
    "helgerud-2007-4x4-vo2max": ("ARR", "Lippincott / ACSM. Paywalled; PubMed abstract free."),
    "leong-2015-pure-grip-strength": ("CC-BY-NC-4.0", "Lancet article paywalled; PMC author manuscript OA."),
    "saeidifard-2019-resistance-mortality": ("CC-BY-NC-4.0", "Eur J Prev Cardiol article paywalled; PMC author manuscript OA."),
    "garcia-hermoso-2018-strength-mortality": ("CC-BY-NC-4.0", "Arch Phys Med Rehabil article paywalled; PMC author manuscript OA."),
    "cappuccio-2010-sleep-mortality": ("CC-BY-NC-4.0", "Sleep (Oxford) author manuscript via PMC OA."),
    "pischon-2008-waist-mortality": ("CC-BY-NC-4.0", "NEJM article paywalled; PMC author manuscript OA."),
    "jha-2013-smoking-mortality": ("CC-BY-NC-4.0", "NEJM article paywalled; PMC author manuscript OA."),
    "wood-2018-alcohol-thresholds": ("CC-BY-4.0", "Lancet (CC-BY-4.0); PMC OA primary."),
    "sprint-2015-intensive-bp": ("CC-BY-NC-4.0", "NEJM article paywalled; PMC author manuscript OA."),
    "ctt-2012-statins-low-risk": ("CC-BY-4.0", "Lancet (CC-BY-4.0); PMC OA primary."),
    "sniderman-2011-apob-meta": ("CC-BY-NC-4.0", "Circ CV Qual Outcomes article paywalled; PMC author manuscript OA."),
    "vital-2019-vitd-omega3": ("CC-BY-NC-4.0", "NEJM article paywalled; PMC author manuscript OA."),
    "chilibeck-2017-creatine-older-adults": ("CC-BY-NC-3.0", "Open Access J Sports Med (Dove Press) CC-BY-NC."),
    "laukkanen-2015-sauna-mortality": ("ARR", "JAMA Intern Med paywalled; PubMed abstract free. No PMC OA available."),
    "select-2023-semaglutide-cv-outcomes": ("ARR", "NEJM paywalled; abstract + figures free."),
    "ma-2019-glucosamine-cv-mortality": ("CC-BY-NC-4.0", "BMJ open-access, CC-BY-NC."),
    "manson-2017-whi-hrt-mortality": ("CC-BY-NC-4.0", "JAMA paywalled; PMC author manuscript OA."),
    "mortensen-2014-coq10-qsymbio": ("ARR", "JACC paywalled; PubMed abstract free."),
    "buijze-2016-cold-shower-rct": ("CC-BY-4.0", "PLOS ONE CC-BY-4.0; full text OA."),
}


def apply(manifest: dict) -> int:
    """Write license + license_note into every source entry."""
    n = 0
    for src in manifest["sources"]:
        sid = src["id"]
        if sid not in LICENSE_MAP:
            print(f"  ! no license entry for {sid}", file=sys.stderr)
            continue
        lic, note = LICENSE_MAP[sid]
        src["license"] = lic
        if note:
            src["license_note"] = note
        elif "license_note" in src:
            del src["license_note"]
        n += 1
    return n


def generate(manifest: dict) -> str:
    """Produce the markdown body of LICENSES.md."""
    by_cat: dict[str, list[dict]] = {
        "redistributable": [],
        "restricted": [],
        "varies": [],
        "not-redistributable": [],
        "unknown": [],
    }
    for src in manifest["sources"]:
        lic = src.get("license", "unknown")
        by_cat[category(lic)].append(src)

    lines: list[str] = []
    lines.append("# Source licenses")
    lines.append("")
    lines.append("This file is generated by `biology/longevity/scripts/licenses.py` from")
    lines.append("`biology/longevity/sources.json`. Do not edit by hand; update the")
    lines.append("`LICENSE_MAP` in that script and re-run `python3")
    lines.append("biology/longevity/scripts/licenses.py all`.")
    lines.append("")
    lines.append("The **license** column reflects a best-effort lookup of the")
    lines.append("source's canonical venue. For preprint servers (arXiv /")
    lines.append("bioRxiv / medRxiv) the license is author-selectable — verify")
    lines.append("per-paper before redistributing full text.")
    lines.append("")

    def table(category_name: str, header: str) -> None:
        entries = by_cat[category_name]
        if not entries:
            return
        lines.append(f"## {header}")
        lines.append("")
        lines.append("| ID | License | URL | Note |")
        lines.append("|---|---|---|---|")
        for src in sorted(entries, key=lambda s: s["id"]):
            note = src.get("license_note", "").replace("|", "\\|")
            url = src.get("url", "")
            lic = src.get("license", "unknown")
            lines.append(f"| `{src['id']}` | `{lic}` | {url} | {note} |")
        lines.append("")

    table("redistributable",
          "Redistributable with attribution (safe to commit full text)")
    table("restricted",
          "Redistributable with restrictions (check NC/SA terms)")
    table("varies",
          "Varies per paper — verify before redistributing")
    table("not-redistributable",
          "Not redistributable (all rights reserved / site terms)")
    table("unknown", "Unknown — needs investigation")

    lines.append("## Summary")
    lines.append("")
    lines.append("| Category | Count |")
    lines.append("|---|---|")
    for cat, header in [
        ("redistributable", "Redistributable"),
        ("restricted", "Restricted"),
        ("varies", "Varies per paper"),
        ("not-redistributable", "Not redistributable"),
        ("unknown", "Unknown"),
    ]:
        lines.append(f"| {header} | {len(by_cat[cat])} |")
    lines.append(f"| **Total** | **{sum(len(v) for v in by_cat.values())}** |")
    lines.append("")

    lines.append("## Safe-to-commit pipeline contract")
    lines.append("")
    lines.append("When `scripts/download.py` and `scripts/convert.py` run,")
    lines.append("they populate `sources/{html,pdf,md}/`, which are")
    lines.append("gitignored by default. If you want to commit converted")
    lines.append("full-text markdown, restrict the commit to sources marked")
    lines.append("**redistributable** above, and include the license + DOI")
    lines.append("as YAML front matter in the committed `.md` file.")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[1] not in {"apply", "generate", "all"}:
        print(__doc__)
        return 2

    manifest = json.loads(MANIFEST.read_text())
    if argv[1] in {"apply", "all"}:
        n = apply(manifest)
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"applied license to {n}/{len(manifest['sources'])} sources in {MANIFEST}")
    if argv[1] in {"generate", "all"}:
        md = generate(manifest)
        REPO_LICENSES_MD.write_text(md + "\n")
        WIKI_LICENSES_MD.write_text(md + "\n")
        print(f"wrote {REPO_LICENSES_MD} and {WIKI_LICENSES_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
