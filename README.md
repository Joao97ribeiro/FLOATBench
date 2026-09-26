# FLOATBench project page (`gh-pages`)

This branch is served at <https://joao97ribeiro.github.io/FLOATBench/> via
GitHub Pages.

## Enabling the page

In **Settings → Pages**, choose:

- **Source:** Deploy from a branch
- **Branch:** `gh-pages`
- **Folder:** `/ (root)`

The site is a single `index.html` with Bulma + Plotly loaded from CDN. No
build step.

## Where the data comes from

- **Dataset Explorer:** reads the six train/test parquet files of
  [`DeCoDELab/FLOATBench`](https://huggingface.co/datasets/DeCoDELab/FLOATBench)
  from Hugging Face when the page opens (~13 MB, parsed in the browser with
  [hyparquet](https://github.com/hyparam/hyparquet)). If Hugging Face cannot
  be reached, it falls back to the bundled copy in
  `static/data/dataset.json` and says so on the page.
- **Live Rows** and the per-simulation button: the Hugging Face dataset
  viewer API (`datasets-server.huggingface.co`).
- **Leaderboard and Results:** `static/data/leaderboard.json`, the E2
  benchmark pools of the paper (not part of the Hugging Face release).

## Updating the bundled data

Regenerate `static/data/` from a FLOATBench checkout with the released
dataset in `data/` and a merged E2 benchmark:

```bash
python scripts/build_data.py \
    --repo_root=../FLOATBench \
    --bench_dir=outputs/within/{tower}/benchmark
```

`dataset.json` holds the 6,468 simulations of the envelope and the
per-section damage of each tower (log10, int16, base64); `leaderboard.json`
holds the E2 pools with Rel L² DEL globally, per regime and per section.

After changing `static/`, bump the `?v=` stamp in `index.html` and in the
`fetch()` calls of `static/js/app.js` so browsers do not keep an old copy.

## Anonymized copy for review

`review/` is an anonymized copy of the page (no authors, affiliations,
preprint, citation, repository or dataset links; data embedded in
`review/static/data/data.js`), served through anonymous.4open.science.
Regenerate it after any change to the site:

```bash
python scripts/build_review.py
```

The script fails if an identifying term is left in the copy.
