"""Build script for the lessons/ static learning site.

Reads distribution metadata (hand-written Tier 1 + Tier 2-multivariate, and
agent-drafted Tier 2 continuous/discrete JSON), renders one HTML page per
scipy.stats distribution plus index/definitions/connections, and generates a
chart per distribution by actually calling scipy.stats. Every code block and
number shown on a page is computed for real, not invented.

Run with: ./.venv/Scripts/python.exe tools/generate_lessons.py
"""

import contextlib
import html
import io
import json
import math
import sys
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

ROOT = Path(__file__).resolve().parent.parent
LESSONS = ROOT / "lessons"
DIST_DIR = LESSONS / "distributions"
IMG_DIR = LESSONS / "assets" / "img"
TOOLS = ROOT / "tools"

sys.path.insert(0, str(TOOLS))
from tier1_metadata import TIER1_METADATA  # noqa: E402
from multivariate_metadata import MULTIVARIATE_METADATA  # noqa: E402

MATRIX_NAMES = {
    "wishart", "invwishart", "matrix_normal", "ortho_group",
    "special_ortho_group", "unitary_group", "random_correlation",
}

# -------------------------------------------------------------------------
# Chart styling (dataviz skill palette — categorical slots 1 & 2, validated
# adjacent pair). Charts render inside a fixed light chart-card, so absolute
# hex (not CSS variables) is used regardless of the page's light/dark theme.
# -------------------------------------------------------------------------
CHART_SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SERIES_BLUE = "#2a78d6"
SERIES_ORANGE = "#eb6834"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "text.color": INK,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "figure.facecolor": CHART_SURFACE,
    "axes.facecolor": CHART_SURFACE,
    "savefig.facecolor": CHART_SURFACE,
})

BUILD_LOG = {"chart_fail": [], "code_fail": [], "page_fail": []}


def _new_fig():
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=110)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return fig, ax


def _save(fig, name):
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(IMG_DIR / f"{name}.svg", format="svg")
    plt.close(fig)


def chart_discrete(name, ctor, params):
    dist = ctor(**params)
    lo = int(math.floor(dist.ppf(0.001)))
    hi = int(math.ceil(dist.ppf(0.999)))
    lo = max(lo, int(getattr(dist, "a", lo)) if np.isfinite(dist.a) else lo)
    if hi - lo > 45:
        hi = lo + 45
    if hi <= lo:
        hi = lo + 5
    x = np.arange(lo, hi + 1)
    pmf = dist.pmf(x)
    cdf = dist.cdf(x)

    fig, ax1 = _new_fig()
    ax1.bar(x, pmf, color=SERIES_BLUE, width=0.6, label="PMF")
    ax1.set_ylabel("P(X = x)", color=SERIES_BLUE)
    ax1.set_xlabel("x")
    ax2 = ax1.twinx()
    ax2.plot(x, cdf, color=SERIES_ORANGE, marker="o", markersize=3, linewidth=1.6, label="CDF")
    ax2.set_ylabel("P(X \u2264 x)", color=SERIES_ORANGE)
    ax2.set_ylim(0, 1.05)
    ax2.grid(False)
    _save(fig, name)


def chart_continuous(name, ctor, params):
    dist = ctor(**params)
    lo = dist.ppf(0.001)
    hi = dist.ppf(0.999)
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        lo, hi = dist.ppf(0.01), dist.ppf(0.99)
    x = np.linspace(lo, hi, 300)
    pdf = dist.pdf(x)
    cdf = dist.cdf(x)

    fig, ax1 = _new_fig()
    ax1.plot(x, pdf, color=SERIES_BLUE, linewidth=2, label="PDF")
    ax1.fill_between(x, pdf, color=SERIES_BLUE, alpha=0.12)
    ax1.set_ylabel("density", color=SERIES_BLUE)
    ax1.set_xlabel("x")
    ax2 = ax1.twinx()
    ax2.plot(x, cdf, color=SERIES_ORANGE, linewidth=1.6, label="CDF")
    ax2.set_ylabel("P(X \u2264 x)", color=SERIES_ORANGE)
    ax2.set_ylim(0, 1.05)
    ax2.grid(False)
    _save(fig, name)


def chart_matrix(name, ctor, params):
    rs = np.random.RandomState(7)
    sample = ctor(**params).rvs(random_state=rs)
    sample = np.atleast_2d(np.asarray(sample))
    if np.iscomplexobj(sample):
        sample = np.abs(sample)
    sample = sample.astype(float)
    fig, ax = _new_fig()
    im = ax.imshow(sample, cmap="RdBu_r", vmin=-np.abs(sample).max(), vmax=np.abs(sample).max())
    ax.set_title("one sampled matrix", fontsize=10, color=MUTED)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.grid(False)
    _save(fig, name)


def chart_multivariate_vector(name, ctor, params):
    rs = np.random.RandomState(7)
    sample = np.atleast_2d(np.asarray(ctor(**params).rvs(size=800, random_state=rs)))
    if sample.shape[0] != 800 and sample.shape[-1] == 800:
        sample = sample.T
    if sample.ndim == 1 or sample.shape[1] < 2:
        sample = np.column_stack([sample.ravel(), np.roll(sample.ravel(), 1)])
    x, y = sample[:, 0], sample[:, 1]
    fig, ax = _new_fig()
    hb = ax.hexbin(x, y, gridsize=22, cmap="Blues", mincnt=1)
    ax.set_xlabel("dimension 1")
    ax.set_ylabel("dimension 2")
    fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04, label="sample count")
    ax.grid(False)
    _save(fig, name)


def make_chart(name, category, ctor, params):
    try:
        if name in MATRIX_NAMES:
            chart_matrix(name, ctor, params)
        elif category == "multivariate":
            chart_multivariate_vector(name, ctor, params)
        elif category == "discrete":
            chart_discrete(name, ctor, params)
        else:
            chart_continuous(name, ctor, params)
        return True
    except Exception:
        BUILD_LOG["chart_fail"].append((name, traceback.format_exc(limit=2)))
        plt.close("all")
        return False


def run_code(name, code):
    ns = {"stats": stats, "np": np}
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, ns)
        return buf.getvalue().strip()
    except Exception:
        BUILD_LOG["code_fail"].append((name, traceback.format_exc(limit=2)))
        return None


# -------------------------------------------------------------------------
# HTML shell
# -------------------------------------------------------------------------
KATEX_HEAD = """
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body, {delimiters: [
    {left: '\\\\[', right: '\\\\]', display: true},
    {left: '\\\\(', right: '\\\\)', display: false}
  ]});"></script>
""".strip()


def page_shell(title, description, body, prefix="", extra_head=""):
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<meta name="description" content="{html.escape(description)}">
<link rel="stylesheet" href="{prefix}assets/style.css">
{KATEX_HEAD}
{extra_head}
</head>
<body>
<nav class="nav">
  <span class="brand">\U0001F3B2 Probability Distributions</span>
  <a href="{prefix}index.html">Index</a>
  <a href="{prefix}definitions.html">Definitions</a>
  <a href="{prefix}connections.html">Connections</a>
</nav>
{body}
</body>
</html>
"""


CATEGORY_LABEL = {
    "discrete": "Discrete",
    "continuous": "Continuous",
    "multivariate": "Multivariate",
    "matrix": "Matrix / Group",
}


def badge_category(name, category):
    if name in MATRIX_NAMES:
        return "matrix"
    return category


def render_distribution_page(name, meta, category, tier):
    badge_cat = badge_category(name, category)
    badge_label = CATEGORY_LABEL[badge_cat]
    ctor = getattr(stats, name)
    params = {p["name"]: p["default"] for p in meta["params"]}

    chart_ok = make_chart(name, category, ctor, params)
    output = run_code(name, meta["code"])

    params_rows = "".join(
        f"<tr><td><code>{html.escape(p['name'])}</code></td>"
        f"<td><code>{html.escape(repr(p['default']))}</code></td>"
        f"<td>{html.escape(p['desc'])}</td></tr>"
        for p in meta["params"]
    )

    use_cases = "".join(f"<li>{html.escape(u)}</li>" for u in meta["use_cases"])

    chart_html = ""
    if chart_ok:
        chart_html = f"""
<div class="chart-card">
  <img src="../assets/img/{name}.svg" alt="{html.escape(meta['display'])} chart" loading="lazy">
  <div class="chart-caption">Theoretical shape at the default parameters above (blue = density/PMF, orange = CDF).</div>
</div>"""

    output_html = ""
    if output is not None:
        output_html = f'<div class="output">{html.escape(output)}</div>'
    else:
        output_html = '<div class="output">(example output unavailable in this build)</div>'

    tier_badge = '<span class="badge tier2">extended catalog</span>' if tier == 2 else ""

    body = f"""
<div class="container">
  <span class="badge {badge_cat}">{badge_label}</span> {tier_badge}
  <h1>{html.escape(meta['display'])}</h1>
  <p class="scipy-name">scipy.stats.{name}</p>

  <h2>Intuition</h2>
  <p>{html.escape(meta['intuition'])}</p>

  <h2>Formula</h2>
  <div class="formula">\\[ {meta['formula']} \\]</div>
  <table class="def-table">
    <tr><th>parameter</th><th>default used here</th><th>meaning</th></tr>
    {params_rows}
  </table>

  <h2>Use cases</h2>
  <ul class="use-cases">{use_cases}</ul>

  <h2>A funny example</h2>
  <div class="story">
    <p>{html.escape(meta['kid_story'])}</p>
    <p class="q">{html.escape(meta['kid_question'])}</p>
  </div>

  <h2>Solve it with <code>scipy.stats</code></h2>
  <pre><code>import scipy.stats as stats
import numpy as np

{html.escape(meta['code'])}</code></pre>
  {output_html}

  {chart_html}

  <div class="footer-nav">
    <a href="../index.html">&larr; Back to index</a>
    <a href="../connections.html">See how it connects to other distributions &rarr;</a>
  </div>
</div>
"""
    title = f"{meta['display']} distribution ({name}) \u2014 Probability Distributions"
    desc = meta["intuition"][:150]
    html_doc = page_shell(title, desc, body, prefix="../")
    (DIST_DIR / f"{name}.html").write_text(html_doc, encoding="utf-8")


def build_all_distribution_pages(catalog, all_meta):
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    for tier_key, tier_num in (("tier1", 1), ("tier2", 2)):
        for category, names in catalog[tier_key].items():
            for name in names:
                meta = all_meta.get(name)
                if meta is None:
                    BUILD_LOG["page_fail"].append((name, "no metadata"))
                    continue
                try:
                    render_distribution_page(name, meta, category, tier_num)
                except Exception:
                    BUILD_LOG["page_fail"].append((name, traceback.format_exc(limit=2)))


def load_all_metadata():
    tier2_json = json.loads((TOOLS / "tier2_metadata.json").read_text(encoding="utf-8"))
    merged = {}
    merged.update(TIER1_METADATA)
    merged.update(MULTIVARIATE_METADATA)
    merged.update(tier2_json)
    return merged


def main():
    catalog = json.loads((TOOLS / "dist_catalog.json").read_text(encoding="utf-8"))
    all_meta = load_all_metadata()

    expected = set()
    for tier in catalog.values():
        for names in tier.values():
            expected.update(names)
    missing = expected - set(all_meta.keys())
    extra = set(all_meta.keys()) - expected
    if missing:
        print(f"WARNING: {len(missing)} distributions missing metadata: {sorted(missing)}")
    if extra:
        print(f"NOTE: {len(extra)} metadata entries not in catalog (ignored): {sorted(extra)}")

    build_all_distribution_pages(catalog, all_meta)

    from build_pages import build_index, build_definitions, build_connections
    build_index(catalog, all_meta, page_shell, MATRIX_NAMES, CATEGORY_LABEL)
    build_definitions(page_shell)
    build_connections(page_shell)

    print("\n--- BUILD REPORT ---")
    print(f"chart failures: {len(BUILD_LOG['chart_fail'])}")
    for n, tb in BUILD_LOG["chart_fail"]:
        print(f"  [chart] {n}: {tb.strip().splitlines()[-1]}")
    print(f"code failures: {len(BUILD_LOG['code_fail'])}")
    for n, tb in BUILD_LOG["code_fail"]:
        print(f"  [code] {n}: {tb.strip().splitlines()[-1]}")
    print(f"page failures: {len(BUILD_LOG['page_fail'])}")
    for n, tb in BUILD_LOG["page_fail"]:
        print(f"  [page] {n}: {tb.strip().splitlines()[-1] if isinstance(tb, str) else tb}")
    total = len(expected)
    print(f"\nBuilt {total - len(BUILD_LOG['page_fail'])}/{total} distribution pages.")


if __name__ == "__main__":
    main()
