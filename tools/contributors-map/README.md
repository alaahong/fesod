# Contributor Heatmap Tool

This utility powers the automated contributor heatmap published via GitHub Actions.
It collects all public contributors to the repository, resolves their profile
locations, renders an ECharts thermal map backed by a bundled Natural
Earth-derived `world-geo.json` dataset, and uploads the resulting PNG to a
persistent GitHub release asset so it can be embedded in the project README
without committing generated files.

## Local usage


## Prerequisites

- Python 3.11+
- `playwright` Chromium dependencies (`python -m playwright install --with-deps chromium`)

```bash
python -m pip install -r tools/contributors-map/requirements.txt
GITHUB_TOKEN="<your-token>" python tools/contributors-map/generate.py \
  --owner apache \
  --repo fesod \
  --output-dir dist/contributor-heatmap \
  --intensity-scale 8
```


For a network-free smoke test you can generate a synthetic map：

```bash
python tools/contributors-map/generate.py --sample --output-dir dist/sample-map
```

Outputs are written to the chosen `--output-dir` and include:

- `contributors-heatmap.png`: static image for embedding (skipped with `--skip-screenshot`)
- `contributors-heatmap.html`: interactive map snapshot rendered with ECharts
- `contributors.json`: resolved contributor metadata
- `summary.txt`: run statistics

> The bundled `world-geo.json` is derived from [Natural Earth](https://www.naturalearthdata.com)
> (public domain) via the [datasets/geo-countries](https://github.com/datasets/geo-countries)
> project.

### Tuning the heatmap

- `--intensity-scale` (default `6.0`) amplifies each location's contributor count before feeding it
  into the heatmap. Increase this value if the gradients look too soft, or reduce it if the map
  appears oversaturated.
- `--skip-screenshot` is handy for quick checks that don't require the PNG output.
