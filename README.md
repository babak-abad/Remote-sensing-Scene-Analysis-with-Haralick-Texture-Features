# Remote-sensing scene analysis with Haralick texture features

A classical, texture-only pipeline for the NWPU-RESISC45 aerial-scene benchmark. Every
256×256 tile is reduced to a 26-number **Haralick fingerprint** — the 13 grey-level
co-occurrence descriptors, computed at four angles and summarised by their mean and range —
and that single fingerprint drives three tasks: nearest-neighbour **retrieval**, supervised
**classification** (an RBF SVM and a simple MLP), and unsupervised **clustering** (K-means
and OPTICS). No deep network, no GPU.

## Headline results

On a curated 8-class subset (150 images each, 1,200 tiles):

- Retrieval precision@5: **0.68**
- SVM (RBF): **0.79** accuracy / 0.79 macro-F1 — simple MLP: **0.73** / 0.73
- K-means: ARI 0.21, NMI 0.33 — OPTICS: ARI 0.10, NMI 0.20 (2 dense clusters, 57% noise)

Texture alone is a strong, interpretable baseline: it classifies scenes well but only
*orders* rather than cleanly *partitions* them.

## Dataset

**NWPU-RESISC45** — 45 classes × 700 images (256×256), Cheng, Han & Lu, 2017
(https://arxiv.org/abs/1703.00121). Distributed for research/educational use only.
Download the dataset and place the class folders under `NWPU-RESISC45/` in the project root:

```
NWPU-RESISC45/<class>/<class>_NNN.jpg
```

The subset of classes and images used is set in `config.py` (`selected_classes`,
`images_per_class`); switch to all 45 classes by editing that list.

## Run

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # Linux/macOS: .venv/bin/pip
.venv/Scripts/python -m src.pipeline             # Linux/macOS: .venv/bin/python
```

The first run extracts and caches the feature matrix under `assets/cache/`; later runs reuse
it. Metrics are printed and written to `assets/outputs/results.json`. All settings — feature
parameters, model hyper-parameters, retrieval metric — live in `config.py`.

## Layout

- `config.py` — all tunable settings
- `src/features.py` — Haralick/GLCM extraction and caching
- `src/retrieval.py` — best-match search by feature distance
- `src/classify.py` — SVM and MLP classifiers
- `src/cluster.py` — K-means and OPTICS clustering
- `src/pipeline.py` — end-to-end driver

Attribution for the dataset and every library is in [RESOURCES.md](RESOURCES.md).
