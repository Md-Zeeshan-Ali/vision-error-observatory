<div align="center">

# Vision Error Observatory (VEO)
### A Model-Agnostic Diagnostic and Observability Framework for Failure Analysis in Monocular Depth Estimation

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CI](https://github.com/MdZeeshanAli01/VEO_V2/actions/workflows/ci.yml/badge.svg?branch=new-version)](https://github.com/MdZeeshanAli01/VEO_V2/actions/workflows/ci.yml)
[![Dataset](https://img.shields.io/badge/Dataset-NYU%20Depth%20V2%20%7C%20Hypersim-blueviolet.svg)]()
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## Overview

Monocular Depth Estimation (MDE) foundation models have reached impressive global benchmark scores, yet standard evaluation protocols rely almost exclusively on dataset-wide scalar summaries (RMSE, MAE, AbsRel, $\delta_1$). These global metrics collapse thousands of 3D spatial predictions into monolithic averages, concealing severe, localized failure clusters (such as reflective glass doors, featureless walls, and slender furniture legs) while offering little insight into the physical or visual mechanisms that triggered the failure.

**Vision Error Observatory (VEO)** is a model-agnostic post-hoc diagnostic framework that aims to move depth evaluation from scalar scoring toward region-level failure attribution. Decoupled from model weights and training gradients, VEO ingests standard prediction arrays, performs inverse-depth scale-shift alignment, isolates high-error regions via connected-components analysis, extracts 7 hand-designed visual evidence features, and assigns each region a failure hypothesis using rule-based scores with a fixed-threshold ambiguity gate (abstains when the top score is < 0.40 or the margin is < 0.10).

> **Status: under revision (v2).** The v1 results below are reported in full, including the unfavourable ones. Several v1 claims were overstated and have been corrected here; the original state is preserved at the git tag `v1-archive`, and every correction is listed in [`docs/ERRATA_V1.md`](docs/ERRATA_V1.md). Known limitations of v1 are listed at the end of the findings.

<div align="center">
  <img src="web_app/assets/fig1_veo_architecture.png" alt="VEO Architecture" width="95%"/>
  <p><em><b>Fig. 1:</b> Modular diagnostic architecture of the Vision Error Observatory (VEO) framework across Modules M1 to M5. [Note: Architecture diagram generated using AI visualization tools].</em></p>
</div>

---

## v1 Empirical Results

All v1 results use 244 NYU Depth V2 images from a custom scene-disjoint test split (1,449 labeled samples partitioned with seed `42`; this is **not** the official NYU 654-image test split, so numbers are not directly comparable with published results). Result files are in [`results/v1/`](results/v1/).

1. **Alignment space (implementation note, not a contribution):**
   * Depth Anything V2 (relative) outputs inverse depth. Fitting it in depth space ($d = s \cdot p + b$) gives a negative scale ($s < 0$) in 244 / 244 test scenes, which is the expected consequence of fitting disparity to depth.
   * Fitting in inverse-depth space ($1/D \approx s \cdot P + b$), the standard practice since MiDaS, gives $s > 0$ in 244 / 244 scenes (observed, not guaranteed) with AbsRel = 0.1040 and $\delta_1$ = 0.8965.
2. **Failure-label distribution across two models (exploratory):**
   * **Depth Anything V2 Small** (5,236 regions): 19.5% `textureless`, 18.0% `boundary_mixing`, 39.3% `ambiguous`.
   * **DPT-Hybrid-MiDaS** (3,689 regions): 17.4% `reflection` vs 13.1% for Depth Anything V2.
   * Caveats: regions are the top 15% of errors *within each image*, so these are shares of each model's relative worst pixels, not absolute vulnerability. No significance test was run. The `reflection` label scored 0/7 against human raters (below), so the reflection difference is not established.
3. **Blinded human agreement (N = 50 test regions, 2 raters):**
   * Inter-rater agreement: Cohen's $\kappa$ = 0.348.
   * **VEO vs. rater 1: $\kappa$ = 0.259, accuracy 36% (18/50), macro-F1 = 0.262. VEO vs. rater 2: $\kappa$ = 0.138.**
   * VEO abstained (`ambiguous`) on 21 / 50 regions. On the 29 regions where it committed to a label, accuracy vs. rater 1 was 58.6% ($\kappa$ = 0.472). The majority-class baseline on that subset is 41.4%.
   * Per-class precision vs. rater 1: `textureless` 7/8, `thin_object` 5/7, `boundary_mixing` 5/7, `reflection` 0/7. VEO never predicted `occlusion` or `poor_lighting`.
4. **Scoring-function sanity check (not a faithfulness result):**
   * Adding noise to the bounding box of `scene_0049` region 8 dropped its `textureless` score from 0.646 to 0.000. Because that score is defined as $1 - \text{variance}/500$, this confirms the implementation behaves as written; it says nothing about the depth model. Tests that re-run the model on edited images are planned for v2.
5. **Oracle boundary-refinement experiment (not deployable):**
   * Bilateral smoothing inside `boundary_mixing` regions across 207 test scenes changed regional RMSE from 0.6629 m to 0.6618 m (−0.16%) and regional AbsRel from 0.2348 to 0.2340 (−0.35%); global RMSE was unchanged (0.4658 m).
   * The pixels to smooth were selected using the **ground-truth** error map, so this cannot run without ground truth. There is no control condition and no significance test.

### Known v1 limitations (being addressed in v2)
* Ground truth is NYU's in-painted `depths`; values in sensor-failure areas (glass, mirrors, dark regions) are filled in, which inflates apparent `reflection` / `poor_lighting` errors.
* Feature thresholds are hand-set, not calibrated against labels. The `occlusion` scale effectively disables that category (22 / 5,236 regions), even though depth discontinuity is the feature most correlated with error (r = 0.45).
* Some features do not isolate their category: `reflection` also fires on bright white walls, and `thin_object` uses bounding-box aspect ratio.
* The human study is small (50 regions, 2 raters) and the raw rating files are not in the repository.

---

## Repository Layout

```text
VEO_V2/
├── src/veo/                # Library: alignment, regions, features, scoring, pipeline, CLI
│   ├── reproduce.py        # `veo reproduce`: splits -> predictions -> analysis -> experiments -> summary
│   ├── provenance.py       # git commit, config hash and package versions stamped on every result
│   ├── summary.py          # renders results/<run>/SUMMARY.md from the result JSON files
│   └── fixtures.py         # synthetic NYU-format dataset and predictor for offline end-to-end runs
├── experiments/            # One script per reported result, each writing one JSON with provenance
│   ├── human_agreement.py  # VEO vs. raters: all-region and selective metrics, majority baseline
│   ├── signal_ablation.py  # labels changed when each feature is neutralised
│   ├── scoring_sanity_check.py   # formula check: added texture lowers the textureless score
│   └── oracle_boundary_filter.py # bilateral filtering in ground-truth-selected masks (oracle)
├── results/                # Committed results; v1/ holds the original study's summaries
├── configs/                # nyu.yaml, hypersim.yaml
├── tests/                  # unit, behaviour, golden-output, pipeline and web-server tests
├── scripts/                # generate_figures.py; site/ builds the web app, its data and video
├── web_app/                # static explorer site, serve.py, fetch_assets.py (deployed with the Dockerfile)
├── docs/                   # protocol, specification, errata
├── data/, outputs/         # local datasets and large run outputs (git-ignored)
└── .github/workflows/      # CI: lint, tests, fixture reproduction
```

---

## Quickstart

This project uses [`uv`](https://github.com/astral-sh/uv) for reproducible environments.

```bash
git clone https://github.com/MdZeeshanAli01/VEO_V2.git
cd VEO_V2
uv sync --extra dev            # add --extra model for real depth models (PyTorch, transformers)
python web_app/fetch_assets.py # site video and scene data (GitHub release assets, checksum-verified)
uv run pytest -q               # unit, behaviour, golden and pipeline tests
uv run veo reproduce --fixture --workdir /tmp/veo   # full pipeline on synthetic data, no downloads
```

The fixture run writes `results/fixture/SUMMARY.md` under the work directory. CI runs the same three steps on every push.

---

## Reproducing Results on NYU Depth V2

```bash
uv sync --extra dev --extra model
uv run veo download-nyu                       # ~2.8 GB official labelled set -> data/nyu/; prints its SHA-256
uv run veo reproduce --models depth-anything/Depth-Anything-V2-Small-hf Intel/dpt-hybrid-midas
```

This extracts the scene-disjoint test split, runs each model, compares alignment protocols, runs the signal ablation, scoring sanity check and oracle filter on the first model, and writes `results/nyu_test/SUMMARY.md`. Every result JSON records the git commit, configuration hash and package versions. Add `--rater1 <csv> --rater2 <csv>` to include a human-agreement study on a rating set exported with `veo prepare-rating`.

The v1 human-agreement figures above can be recomputed from the stored confusion matrix:

```bash
uv run python experiments/human_agreement.py --from-confusion results/v1/human_validation_results.json
```

Raw rater CSVs and per-region v1 outputs are not in the repository, so the other v1 numbers can be re-derived only by re-running the pipeline.

---

## License

This project is licensed under the [MIT License](LICENSE).
