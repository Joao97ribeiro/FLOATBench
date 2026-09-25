# FLOATBench project page (`gh-pages`)

This branch is served at <https://joao97ribeiro.github.io/FLOATBench/> via
GitHub Pages.

## Enabling the page

In **Settings → Pages**, choose:

- **Source:** Deploy from a branch
- **Branch:** `gh-pages`
- **Folder:** `/ (root)`

The site is a single `index.html` with Bulma + Plotly loaded from CDN. No
build step. The "Live Rows" section and the per-simulation button read the
dataset directly from the Hugging Face dataset viewer API
(`datasets-server.huggingface.co`, `DeCoDELab/FLOATBench`).

## Updating the interactive plots

The Dataset Explorer and the Leaderboard read pre-extracted JSON files in
[`static/data/`](./static/data). Regenerate them from a FLOATBench checkout
with the released dataset in `data/` and a merged E2 benchmark:

```bash
python scripts/build_data.py \
    --repo_root=../FLOATBench \
    --bench_dir=outputs/within/{tower}/benchmark
```

`dataset.json` holds the 6,468 simulations of the envelope and the
per-section damage of each tower (log10, int16, base64); `leaderboard.json`
holds the E2 pools with Rel L² DEL globally, per regime and per section.
