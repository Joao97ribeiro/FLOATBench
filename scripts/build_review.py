"""Build the anonymized copy of the project page under review/.

Run from the root of the `gh-pages` branch after any change to the site:

    python scripts/build_review.py

The copy drops everything that identifies the authors (names,
affiliations, e-mails, preprint, citation, repository and dataset links,
the live Hugging Face sections) and embeds the data in
`review/static/data/*.js` (compact, split per tower), so the page
works inside the sandboxed
viewer of anonymous.4open.science, where `fetch()` of sibling files is
blocked. The script fails if an identifying term survives.
"""

# pylint: disable=line-too-long  # HTML fragments
import array
import base64
import json
import pathlib
import re
import shutil

SITE = pathlib.Path(__file__).resolve().parents[1]
B52 = "0123456789bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ"
CODE_URL = "https://anonymous.4open.science/r/FLOATBench-84A1/"
CODE_ZIP = "https://anonymous.4open.science/api/repo/FLOATBench-84A1/zip"
DATA_URL = "https://osf.io/te9na/?view_only=224887b50912448b871620b3ef96cefc"
DATA_ZIP = ("https://osf.io/download/6ab722cf8c24072987b0bede/"
            "?view_only=224887b50912448b871620b3ef96cefc")
OUT = SITE / "review"
FIGURES = [
    "logo.png", "overview.png", "simulation_pipeline.png",
    "figure_geom_damage.png", "regime_partition.png", "crossover.png",
    "cross_tower.png", "app_heatmap_regimes.png", "app_family_regime.png",
    "app_scatter_global_exex.png", "app_scatter_e3.png",
    "app_split_sensitivity.png", "app_spacing_wind.png", "app_spacing_wave.png"
]
FORBIDDEN = [
    "joao", "ribeiro", "pimenta", "tavares", "faez", "ahmed", "decodelab",
    "mit.edu", "delft", "brown univ", "porto", "aveiro", "2605.25717", "arxiv",
    "github.io", "github.com/joao97", "iclr", "neurips", "openreview"
]

START = f'''<!-- ============ GET STARTED ============ -->
<section class="section" id="start">
  <div class="container is-max-desktop">
    <h2 class="title is-3">Get Started</h2>
    <div class="columns">
      <div class="column is-half">
        <div class="box">
          <p class="title is-5"><i class="fas fa-code"></i> &nbsp;<a href="{CODE_URL}">Code (anonymized)</a></p>
          <p>Evaluation harness, regime-aware splitter, leaderboards and every analysis of the paper. Browse it online or <a href="{CODE_ZIP}">download it as a zip</a>.</p>
        </div>
      </div>
      <div class="column is-half">
        <div class="box">
          <p class="title is-5"><i class="fas fa-database"></i> &nbsp;<a href="{DATA_URL}">Data (anonymized)</a></p>
          <p>The released CSVs of the three towers (<code>dataset/FLOATBench.zip</code>, 25.6 MB, with a README of the schema and split) and the raw time-series audit subset. <a href="{DATA_ZIP}">Direct download of the dataset</a>.</p>
        </div>
      </div>
    </div>
    <p class="has-text-weight-semibold" style="margin-bottom:0.4rem;">Download and run the benchmark</p>
<pre><code># 1. code
curl -L -o code.zip "{CODE_ZIP}"
unzip code.zip -d FLOATBench &amp;&amp; cd FLOATBench
conda env create -f environment.yml &amp;&amp; conda activate floatbench

# 2. data: downloads the archive, checks it and unpacks data/{{ref,opt1,opt2}}/
python scripts/download/run.py --flagfile=scripts/download/config.cfg

# 3. smoke test (~10 min on one GPU), then the full E2 + E3 benchmark
python scripts/run_benchmark.py --experiment=within --tower=ref --time_limit=120
python scripts/run_benchmark.py --experiment=all</code></pre>
    <p class="has-text-weight-semibold" style="margin:1rem 0 0.4rem;">Or just look at the data</p>
<pre><code>import pandas as pd

test = pd.read_csv("data/ref/test_damage.csv")
ex_ex = test[(test.wind_group == "Extrapolate") &amp; (test.wave_group == "Extrapolate")]
print(len(test), "test rows,", ex_ex.sim_id.nunique(), "EX_EX simulations")</code></pre>
  </div>
</section>

'''


def cut(html, start, end):
    """Remove html[start:end] where start/end are marker strings."""
    i = html.index(start)
    j = html.index(end, i) + len(end)
    return html[:i] + html[j:]


def section(html, name):
    """Remove one <!-- ==== NAME ==== --> section block."""
    return cut(html, f"<!-- ============ {name} ============ -->",
               "</section>\n")


def build_html(html):
    """Anonymized index.html."""
    # Hero: no authors, affiliations or external buttons.
    i = html.index('      <div class="is-size-5 publication-authors">')
    j = html.index("    </div>\n  </div>\n</section>", i)
    html = html[:i] + f'''      <div class="publication-links" style="margin-top:1.5rem;">
        <a href="{CODE_URL}" class="button is-rounded is-dark">
          <span class="icon"><i class="fas fa-code"></i></span><span>Code</span>
        </a>
        <a href="{DATA_URL}" class="button is-rounded is-dark">
          <span class="icon"><i class="fas fa-database"></i></span><span>Data</span>
        </a>
        <a href="#start" class="button is-rounded is-dark">
          <span class="icon"><i class="fas fa-download"></i></span><span>Get started</span>
        </a>
        <a href="#explorer" class="button is-rounded is-dark">
          <span class="icon"><i class="fas fa-cube"></i></span><span>Dataset explorer</span>
        </a>
        <a href="#leaderboard" class="button is-rounded is-dark">
          <span class="icon"><i class="fas fa-trophy"></i></span><span>Leaderboard</span>
        </a>
      </div>
''' + html[j:]
    # Hugging Face banner above the tagline.
    html = cut(html,
               '    <div class="has-text-centered" style="margin:0 0 1.5rem;',
               "    </div>\n\n")
    # Get started: anonymized code and data, with commands.
    html = html.replace("<!-- ============ SCOPE ============ -->",
                        START + "<!-- ============ SCOPE ============ -->", 1)
    # Sections that link to identifying resources.
    for name in ("LIVE ROWS", "RESOURCES", "CITATION"):
        html = section(html, name)
    # Per-simulation Hugging Face button.
    html = cut(
        html,
        '        <button class="button is-rounded is-dark is-small" id="sim-hf"',
        '<div class="is-size-7" id="sim-hf-status" style="margin-top:0.6rem;"></div>\n'
    )
    # Links to the sister framework and the dataset host.
    html = re.sub(
        r'<a href="https://joao97ribeiro\.github\.io/FLOAT/">FLOAT</a>',
        "FLOAT (cited in the paper)", html)
    html = re.sub(
        r'read directly from the Hugging Face dataset <a [^>]*>[^<]*</a> when this page opens',
        "from the released dataset", html)
    html = re.sub(
        r'<p class="is-size-7 has-text-grey" id="ex-source"[^>]*>.*?</p>',
        '<p class="is-size-7 has-text-grey" id="ex-source" '
        'style="margin-top:-0.75rem;margin-bottom:1rem;">'
        'Loading the dataset…</p>',
        html,
        flags=re.S)
    html = html.replace("Files (per tower, on Hugging Face)",
                        "Files (per tower)")
    html = re.sub(r'<meta property="og:image"[^>]*>\n', "", html)
    # Footer without the source link.
    html = re.sub(
        r'<footer class="footer">.*?</footer>',
        '<footer class="footer">\n  <div class="content has-text-centered">\n'
        '    <p>Anonymized project page for review. Page template adapted from '
        'the Academic Project Page Template.</p>\n  </div>\n</footer>',
        html,
        flags=re.S)
    html = alternate_backgrounds(html)
    # Embedded data before the app script.
    scripts = "".join(f'<script src="static/data/{f}.js"></script>\n  '
                      for f in ("data", "damage_ref", "damage_opt1",
                                "damage_opt2"))
    html = re.sub(r'(<script defer src="static/js/app\.js[^"]*"></script>)',
                  lambda m: scripts + m.group(1), html)
    # The public page sends visitors of the anonymized mirror here.
    html = re.sub(r"<script>/\* anon-redirect \*/.*?</script>\n",
                  "",
                  html,
                  flags=re.S)
    return html


def alternate_backgrounds(html):
    """White / light-grey alternation of the sections after the hero."""
    count = [0]

    def repl(_match):
        light = count[0] % 2 == 1
        count[0] += 1
        return ('<section class="section has-background-light"'
                if light else '<section class="section"')

    return re.sub(r'<section class="section(?: has-background-light)?"', repl,
                  html)


def build_js(js):
    """Anonymized app.js: no dataset host id."""
    js = re.sub(r'const HF_DATASET = "[^"]*";', 'const HF_DATASET = "";', js)
    return js


def check(path, text):
    """Fail if an identifying term is left."""
    low = text.lower()
    hits = [t for t in FORBIDDEN if t in low]
    if hits:
        raise SystemExit(f"{path}: identifying terms left: {hits}")


def compact_damage(towers):
    """Per-tower damage scripts: 2 vowel-free characters per value.

    The anonymizer rewrites every served text file (term replacement) and
    times out on one large file; small files without vowels are served
    quickly and can never match a (vowel-bearing) term. Modifies `towers`
    in place (drops the base64 field, adds the decoding parameters).
    """
    out = {}
    for name, tower in towers.items():
        vals = [
            v / 1000 for v in array.array(
                "h", base64.b64decode(tower.pop("log_damage_i16")))
        ]
        lo, hi = min(vals), max(vals)
        step = (hi - lo) / (52 * 52 - 1)
        codes = [round((v - lo) / step) for v in vals]
        tower["dmg_lo"], tower["dmg_step"] = lo, step
        enc = "".join(B52[c // 52] + B52[c % 52] for c in codes)
        out[name] = ("window.FB_DMG = window.FB_DMG || {};\n"
                     f'window.FB_DMG.{name} = "{enc}";\n')
    return out


def main():
    """Write review/ from the current site."""
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "static" / "css").mkdir(parents=True)
    (OUT / "static" / "js").mkdir(parents=True)
    (OUT / "static" / "data").mkdir(parents=True)
    (OUT / "figures").mkdir(parents=True)

    html = build_html((SITE / "index.html").read_text())
    js = build_js((SITE / "static" / "js" / "app.js").read_text())
    css = (SITE / "static" / "css" / "style.css").read_text()
    data = {
        name:
            json.loads((SITE / "static" / "data" / f"{name}.json").read_text())
        for name in ("dataset", "leaderboard")
    }
    dmg_js = compact_damage(data["dataset"]["towers"])
    datajs = "window.FB_DATA = " + json.dumps(data,
                                              separators=(",", ":")) + ";\n"
    files = [("index.html", html), ("static/js/app.js", js),
             ("static/css/style.css", css), ("static/data/data.js", datajs)]
    files += [(f"static/data/damage_{name}.js", text)
              for name, text in dmg_js.items()]
    for path, text in files:
        check(path, text)
        (OUT / path).write_text(text)
    for fig in FIGURES:
        shutil.copy(SITE / "figures" / fig, OUT / "figures" / fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
