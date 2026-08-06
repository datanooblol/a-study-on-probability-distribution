"""index.html, definitions.html, connections.html builders for the lessons site."""

import html

import scipy.stats as stats


# ---------------------------------------------------------------------------
# index.html
# ---------------------------------------------------------------------------
def build_index(catalog, all_meta, page_shell, matrix_names, category_label):
    grouped = {"discrete": [], "continuous": [], "multivariate": [], "matrix": []}
    counts = {}
    for tier_key, tier_num in (("tier1", 1), ("tier2", 2)):
        for category, names in catalog[tier_key].items():
            for name in names:
                badge_cat = "matrix" if name in matrix_names else category
                grouped[badge_cat].append((name, tier_num))

    for k in grouped:
        grouped[k].sort(key=lambda t: t[0])
        counts[k] = len(grouped[k])

    total = sum(counts.values())

    section_blurbs = {
        "discrete": "Countable outcomes — number of successes, arrivals, defects, ranks.",
        "continuous": "Measurements on a continuum — time, distance, magnitude, ratios.",
        "multivariate": "Several random quantities at once, with their own joint shape.",
        "matrix": "Random matrices — covariance structures, rotations, correlation matrices.",
    }

    sections_html = ""
    for cat in ("discrete", "continuous", "multivariate", "matrix"):
        cards = ""
        for name, tier_num in grouped[cat]:
            meta = all_meta.get(name)
            if meta is None:
                continue
            blurb = meta["intuition"]
            blurb = blurb[:90] + ("…" if len(blurb) > 90 else "")
            tier_badge = ' <span class="badge tier2">ext.</span>' if tier_num == 2 else ""
            cards += f"""
      <div class="dist-card" data-name="{html.escape(name + ' ' + meta['display'])}">
        <a href="distributions/{name}.html">{html.escape(meta['display'])}</a>{tier_badge}
        <span class="scipy-name">scipy.stats.{name}</span>
        <div class="chart-caption">{html.escape(blurb)}</div>
      </div>"""
        sections_html += f"""
    <div class="category-section">
      <h2>{category_label[cat]} <span class="scipy-name">({counts[cat]})</span></h2>
      <p class="category-note">{section_blurbs[cat]}</p>
      <div class="dist-grid">{cards}
      </div>
    </div>"""

    body = f"""
<div class="container wide">
  <div class="hero">
    <h1>A Study on Probability Distributions</h1>
    <p class="lede">Every distribution in <code>scipy.stats</code> ({total} of them),
    explained from intuition to formula to a working <code>scipy.stats</code> example
    — with a kid-friendly story for each one.</p>
  </div>

  <div class="hub-links">
    <a href="definitions.html">
      <span class="hub-title">Definitions</span>
      <span class="hub-desc">Discrete vs. continuous, PMF vs. PDF vs. CDF, and how to use each one wisely.</span>
    </a>
    <a href="connections.html">
      <span class="hub-title">Connections</span>
      <span class="hub-desc">How distributions build on each other — Bernoulli → Binomial → Poisson, and more.</span>
    </a>
  </div>

  <input id="filter-box" type="search" placeholder="Filter {total} distributions by name…" autocomplete="off">

  {sections_html}
</div>
<script>
  const box = document.getElementById('filter-box');
  const cards = Array.from(document.querySelectorAll('.dist-card'));
  box.addEventListener('input', () => {{
    const q = box.value.trim().toLowerCase();
    cards.forEach(c => {{
      const hit = c.dataset.name.toLowerCase().includes(q);
      c.classList.toggle('hidden', !hit);
    }});
    document.querySelectorAll('.category-section').forEach(sec => {{
      const anyVisible = Array.from(sec.querySelectorAll('.dist-card')).some(c => !c.classList.contains('hidden'));
      sec.style.display = anyVisible ? '' : 'none';
    }});
  }});
</script>
"""
    doc = page_shell(
        "Probability Distributions — Table of Contents",
        "Full scipy.stats distribution catalog with intuition, formulas, and worked examples.",
        body,
        prefix="",
    )
    (_lessons_dir() / "index.html").write_text(doc, encoding="utf-8")


def _lessons_dir():
    from pathlib import Path
    return Path(__file__).resolve().parent.parent / "lessons"


# ---------------------------------------------------------------------------
# definitions.html
# ---------------------------------------------------------------------------
def build_definitions(page_shell):
    binom_dist = stats.binom(10, 0.5)
    norm_dist = stats.norm(0, 1)

    binom_pmf = round(float(binom_dist.pmf(5)), 4)
    binom_cdf = round(float(binom_dist.cdf(5)), 4)
    binom_sf = round(float(binom_dist.sf(5)), 4)

    norm_pdf = round(float(norm_dist.pdf(1.96)), 4)
    norm_cdf = round(float(norm_dist.cdf(1.96)), 4)
    norm_sf = round(float(norm_dist.sf(1.96)), 4)
    norm_ppf = round(float(norm_dist.ppf(0.975)), 4)

    body = f"""
<div class="container">
  <h1>Definitions</h1>
  <p class="lede">The vocabulary every distribution page uses — read this once and every
  other page will make sense.</p>

  <h2>Discrete vs. continuous</h2>
  <p>A <strong>discrete</strong> random variable can only take specific, separated values —
  usually whole numbers you could list: 0, 1, 2, 3, … (number of heads, number of claims,
  number of customers). A <strong>continuous</strong> random variable can take any value on a
  range — there's always another value between any two you pick (height, waiting time,
  temperature). This distinction is the first thing to check about any random quantity,
  because it decides which function you reach for: PMF for discrete, PDF for continuous.</p>

  <h2>PMF, PDF, CDF, SF, PPF</h2>
  <table class="def-table">
    <tr><th>Name</th><th>scipy method</th><th>Applies to</th><th>Answers</th></tr>
    <tr><td>Probability Mass Function (PMF)</td><td><code>dist.pmf(x)</code></td><td>discrete</td>
        <td>The exact probability that X equals x: P(X = x). This <em>is</em> a probability, always between 0 and 1.</td></tr>
    <tr><td>Probability Density Function (PDF)</td><td><code>dist.pdf(x)</code></td><td>continuous</td>
        <td>The <em>density</em> of probability at x — not a probability itself (it can exceed 1).
        Only an area under the curve, over some range, is a probability.</td></tr>
    <tr><td>Cumulative Distribution Function (CDF)</td><td><code>dist.cdf(x)</code></td><td>both</td>
        <td>P(X ≤ x) — the probability of x or anything smaller/earlier.</td></tr>
    <tr><td>Survival Function (SF)</td><td><code>dist.sf(x)</code></td><td>both</td>
        <td>P(X &gt; x) = 1 − CDF(x) — the upper tail.</td></tr>
    <tr><td>Percent Point Function (PPF)</td><td><code>dist.ppf(q)</code></td><td>both</td>
        <td>The inverse of the CDF: "what x gives cumulative probability q?" — i.e. a quantile.</td></tr>
  </table>

  <p><strong>Why the PDF is not a probability:</strong> for a continuous variable, the
  probability of hitting any single exact value is zero (there are infinitely many
  possible values). <code>pdf(x)</code> only tells you the relative likelihood <em>density</em>
  near x; you get an actual probability by integrating it over an interval — which is
  exactly what <code>cdf(b) - cdf(a)</code> does for you, i.e. P(a &lt; X ≤ b).</p>

  <h2>Worked side-by-side example</h2>
  <p>Ten coin flips, fair coin — discrete, <code>scipy.stats.binom(n=10, p=0.5)</code>:</p>
  <pre><code>dist = stats.binom(n=10, p=0.5)
dist.pmf(5)   # P(exactly 5 heads)
dist.cdf(5)   # P(5 or fewer heads)
dist.sf(5)    # P(more than 5 heads)</code></pre>
  <div class="output">P(exactly 5 heads) = {binom_pmf}
P(5 or fewer heads) = {binom_cdf}
P(more than 5 heads) = {binom_sf}</div>

  <p>Standard normal — continuous, <code>scipy.stats.norm(loc=0, scale=1)</code>:</p>
  <pre><code>dist = stats.norm(loc=0, scale=1)
dist.pdf(1.96)    # density at x=1.96 (NOT a probability)
dist.cdf(1.96)    # P(X <= 1.96)
dist.sf(1.96)     # P(X > 1.96), the upper tail
dist.ppf(0.975)   # the x whose CDF is 0.975</code></pre>
  <div class="output">density at x=1.96 = {norm_pdf}   (not a probability — can exceed 1 for narrow distributions)
P(X &lt;= 1.96) = {norm_cdf}
P(X &gt; 1.96) = {norm_sf}
x such that P(X &lt;= x) = 0.975  →  x = {norm_ppf}</div>

  <h2>Using them wisely</h2>
  <ul class="use-cases">
    <li><strong>Prefer <code>sf(x)</code> over <code>1 - cdf(x)</code> for tail probabilities.</strong>
        When x is far in the tail, <code>cdf(x)</code> rounds to 1.0 in floating point, and
        <code>1 - cdf(x)</code> silently becomes 0. <code>sf</code> is computed directly and
        keeps precision far into the tail.</li>
    <li><strong>Use <code>ppf</code> for critical values / quantiles</strong> — e.g. the 95th
        percentile, a confidence interval bound, or a value-at-risk threshold, instead of
        solving the CDF equation by hand.</li>
    <li><strong>Never compare a PDF value across distributions with different scales</strong>
        and call it a probability comparison — densities depend on the units of x. Compare
        CDF differences (areas) instead.</li>
    <li><strong>Every scipy distribution also exposes <code>.mean()</code>, <code>.var()</code>,
        <code>.std()</code>, <code>.median()</code>, and <code>.rvs(size=n)</code></strong> for
        random sampling — useful for simulation once you've picked the right shape.</li>
    <li><strong>Discrete CDFs are step functions</strong> — <code>P(X &lt; x)</code> and
        <code>P(X ≤ x)</code> are genuinely different for discrete variables (use
        <code>dist.cdf(x - 1)</code> for the strict version), whereas for continuous variables
        they're the same because any single point has probability zero.</li>
  </ul>

  <div class="footer-nav">
    <a href="index.html">&larr; Back to index</a>
    <a href="connections.html">Next: how distributions connect &rarr;</a>
  </div>
</div>
"""
    doc = page_shell(
        "Definitions — Probability Distributions",
        "Discrete vs continuous, PMF vs PDF vs CDF, and how to use scipy.stats wisely.",
        body,
        prefix="",
    )
    (_lessons_dir() / "definitions.html").write_text(doc, encoding="utf-8")


# ---------------------------------------------------------------------------
# connections.html
# ---------------------------------------------------------------------------
NODES = {
    "bernoulli":         (50,  50,  "bernoulli",  "discrete"),
    "binomial":           (250, 50,  "binom",      "discrete"),
    "multinomial":        (450, 50,  "multinomial","discrete"),
    "poisson":            (650, 50,  "poisson",    "discrete"),
    "geometric":          (50,  165, "geom",       "discrete"),
    "negative_binomial":  (250, 165, "nbinom",     "discrete"),
    "hypergeometric":     (450, 165, "hypergeom",  "discrete"),
    "uniform":            (50,  290, "uniform",    "continuous"),
    "beta":               (250, 290, "beta",       "continuous"),
    "normal":             (650, 290, "norm",       "continuous"),
    "lognormal":          (850, 290, "lognorm",    "continuous"),
    "exponential":        (50,  405, "expon",      "continuous"),
    "gamma":              (250, 405, "gamma",      "continuous"),
    "chi_square":         (450, 405, "chi2",       "continuous"),
    "t_dist":             (650, 405, "t",          "continuous"),
    "pareto":             (50,  520, "pareto",     "continuous"),
    "weibull":            (250, 520, "weibull_min","continuous"),
    "f_dist":             (450, 520, "f",          "continuous"),
}

EDGES = [
    ("bernoulli", "binomial", "n i.i.d. trials, sum of 0/1s"),
    ("bernoulli", "geometric", "# trials until 1st success"),
    ("binomial", "multinomial", "generalize to k>2 outcomes"),
    ("binomial", "negative_binomial", "generalize: failures before r successes"),
    ("geometric", "negative_binomial", "special case r = 1"),
    ("binomial", "hypergeometric", "sampling without replacement"),
    ("binomial", "poisson", "n→∞, p→0, np=λ"),
    ("binomial", "normal", "n large (CLT)"),
    ("poisson", "normal", "λ large (CLT)"),
    ("poisson", "exponential", "waiting time between events"),
    ("uniform", "exponential", "inverse-CDF transform"),
    ("uniform", "beta", "special case Beta(1,1)"),
    ("exponential", "pareto", "Y = e^X tail"),
    ("exponential", "gamma", "sum of k i.i.d. exponentials"),
    ("exponential", "weibull", "generalized with shape param"),
    ("gamma", "chi_square", "shape=df/2, scale=2"),
    ("normal", "chi_square", "sum of squared std normals"),
    ("normal", "lognormal", "Y = e^X"),
    ("normal", "t_dist", "with χ², small-sample mean test"),
    ("chi_square", "t_dist", "normal / sqrt(χ²/df)"),
    ("chi_square", "f_dist", "ratio of two independent χ²"),
    ("beta", "binomial", "conjugate prior (Bayesian)"),
]

NODE_W, NODE_H = 168, 54


def _node_center(nid):
    x, y, _, _ = NODES[nid]
    return x + NODE_W / 2, y + NODE_H / 2


def _edge_path(src, dst):
    x1, y1 = _node_center(src)
    x2, y2 = _node_center(dst)
    dx, dy = x2 - x1, y2 - y1
    dist = max((dx ** 2 + dy ** 2) ** 0.5, 1e-6)
    ux, uy = dx / dist, dy / dist
    sx, sy = x1 + ux * (NODE_W / 2.1), y1 + uy * (NODE_H / 2.1)
    ex, ey = x2 - ux * (NODE_W / 2.1), y2 - uy * (NODE_H / 2.1)
    return sx, sy, ex, ey


def _render_svg():
    max_x = max(x + NODE_W for x, y, _, _ in NODES.values()) + 40
    max_y = max(y + NODE_H for x, y, _, _ in NODES.values()) + 40

    parts = [f'<svg viewBox="0 0 {max_x} {max_y}" width="{max_x}" height="{max_y}" xmlns="http://www.w3.org/2000/svg">']
    parts.append("""
  <defs>
    <marker id="arrow" markerWidth="9" markerHeight="9" refX="7" refY="3.5" orient="auto">
      <path d="M0,0 L0,7 L8,3.5 z" fill="var(--muted)"></path>
    </marker>
  </defs>""")

    for src, dst, label in EDGES:
        sx, sy, ex, ey = _edge_path(src, dst)
        mx, my = (sx + ex) / 2, (sy + ey) / 2
        w = max(len(label) * 6.0 + 10, 20)
        parts.append(f'  <path class="g-edge" d="M{sx:.1f},{sy:.1f} L{ex:.1f},{ey:.1f}"></path>')
        parts.append(
            f'  <rect x="{mx - w / 2:.1f}" y="{my - 9:.1f}" width="{w:.1f}" height="16" '
            f'fill="var(--chart-surface)" opacity="0.92"></rect>'
        )
        parts.append(
            f'  <text class="g-edge-label" x="{mx:.1f}" y="{my + 3:.1f}" text-anchor="middle">'
            f'{html.escape(label)}</text>'
        )

    for nid, (x, y, scipy_name, cat) in NODES.items():
        label = nid.replace("_", " ").replace(" dist", "").title()
        parts.append(f'  <a href="distributions/{scipy_name}.html">')
        parts.append(f'  <g class="g-node {cat}">')
        parts.append(f'    <rect x="{x}" y="{y}" width="{NODE_W}" height="{NODE_H}" rx="9"></rect>')
        parts.append(
            f'    <text x="{x + NODE_W / 2}" y="{y + 22}" text-anchor="middle" font-weight="700">'
            f'{html.escape(label)}</text>'
        )
        parts.append(
            f'    <text x="{x + NODE_W / 2}" y="{y + 40}" text-anchor="middle" font-size="11" '
            f'fill="var(--muted)">scipy.stats.{scipy_name}</text>'
        )
        parts.append("  </g>")
        parts.append("  </a>")

    parts.append("</svg>")
    return "\n".join(parts)


def build_connections(page_shell):
    svg = _render_svg()
    body = f"""
<div class="container wide">
  <h1>Connections</h1>
  <p class="lede">Distributions aren't a random list — most of them are built from
  simpler ones, or arise as a limit of another. This map covers the core teaching set;
  click any node to jump to its full page.</p>

  <div class="graph-wrap">
    {svg}
  </div>

  <div class="g-legend" style="margin-top:1rem; display:flex; gap:1.5rem; font-size:0.85rem; color:var(--text-secondary);">
    <span><span style="display:inline-block;width:10px;height:10px;border:2px solid var(--series-blue);border-radius:3px;margin-right:6px;"></span>Discrete family</span>
    <span><span style="display:inline-block;width:10px;height:10px;border:2px solid var(--series-orange);border-radius:3px;margin-right:6px;"></span>Continuous family</span>
  </div>

  <h2>Reading the map</h2>
  <ul class="use-cases">
    <li><strong>Bernoulli</strong> is the atom — a single yes/no trial. Repeat it n times
    and sum the successes: that's <strong>Binomial</strong>. Count trials until the first
    success instead: that's <strong>Geometric</strong>.</li>
    <li><strong>Binomial</strong> generalizes three ways: more than two outcomes per trial
    (<strong>Multinomial</strong>), counting failures before r successes instead of a fixed
    n (<strong>Negative Binomial</strong>), or sampling without replacement from a finite
    population (<strong>Hypergeometric</strong>).</li>
    <li>As trials grow and success gets rare while n×p stays fixed, Binomial approaches
    <strong>Poisson</strong> — the distribution of rare-event counts. The gap in time
    <em>between</em> Poisson events is itself <strong>Exponential</strong>.</li>
    <li>Chain independent Exponential waits together and you get <strong>Gamma</strong>;
    Gamma with a particular shape is <strong>Chi-square</strong>; a ratio built from Normal
    and Chi-square gives the <strong>t</strong> distribution (small-sample inference), and a
    ratio of two Chi-squares gives <strong>F</strong> (comparing variances).</li>
    <li><strong>Normal</strong> shows up as a limit of both Binomial and Poisson for large
    samples (Central Limit Theorem) — and exponentiating a Normal gives
    <strong>Log-normal</strong>, the standard model for quantities that multiply rather
    than add (stock prices, incomes).</li>
    <li><strong>Uniform</strong> is the universal source: feeding it through the inverse
    CDF of any distribution is literally how <code>.rvs()</code> generates random samples
    for all of them under the hood.</li>
  </ul>

  <div class="footer-nav">
    <a href="definitions.html">&larr; Back to definitions</a>
    <a href="index.html">Browse the full index &rarr;</a>
  </div>
</div>
"""
    doc = page_shell(
        "Connections — Probability Distributions",
        "How probability distributions build on and relate to each other.",
        body,
        prefix="",
    )
    (_lessons_dir() / "connections.html").write_text(doc, encoding="utf-8")
