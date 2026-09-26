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
    html = html[:i] + '''      <div class="publication-links" style="margin-top:1.5rem;">
        <a href="#explorer" class="button is-rounded is-dark">
          <span class="icon"><i class="fas fa-cube"></i></span><span>Dataset explorer</span>
        </a>
        <a href="#leaderboard" class="button is-rounded is-dark">
          <span class="icon"><i class="fas fa-trophy"></i></span><span>Leaderboard</span>
        </a>
        <a href="#findings" class="button is-rounded is-dark">
          <span class="icon"><i class="fas fa-chart-line"></i></span><span>Results</span>
        </a>
      </div>
''' + html[j:]
    # Hugging Face banner above the tagline.
    html = cut(html,
               '    <div class="has-text-centered" style="margin:0 0 1.5rem;',
               "    </div>\n\n")
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
