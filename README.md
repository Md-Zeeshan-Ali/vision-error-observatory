<div align="center">

# Vision Error Observatory (VEO)
### A Model-Agnostic Diagnostic and Observability Framework for Failure Analysis in Monocular Depth Estimation

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/Unit%20Tests-14%2F14%20Passing-brightgreen.svg)]()
[![Dataset](https://img.shields.io/badge/Dataset-NYU%20Depth%20V2%20%7C%20Hypersim-blueviolet.svg)]()
[![Paper](https://img.shields.io/badge/Paper-IEEE%20Format%20(5%2C020%20Words)-darkblue.svg)](paper/VEO_conference_paper.pdf)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## Overview

Monocular Depth Estimation (MDE) foundation models have reached impressive global benchmark scores, yet standard evaluation protocols rely almost exclusively on dataset-wide scalar summaries (RMSE, MAE, AbsRel, $\delta_1$). These global metrics collapse thousands of 3D spatial predictions into monolithic averages, concealing severe, localized failure clusters (such as reflective glass doors, featureless walls, and slender furniture legs) while offering zero actionable insight into the physical or visual mechanisms that triggered the failure.

**Vision Error Observatory (VEO)** is a model-agnostic post-hoc diagnostic framework designed to transform depth evaluation from passive scalar scoring into an evidence-based root-cause audit. Decoupled from model weights and training gradients, VEO ingests standard prediction arrays, performs mathematically grounded scale-shift alignment, isolates spatial error regions via morphological connected-components analysis, extracts 7 multi-signal visual evidence features, and attributes errors to interpretable failure hypotheses through a calibrated ambiguity gate.

<div align="center">
  <img src="figures/fig1_veo_architecture.png" alt="VEO Architecture" width="95%"/>
  <p><em><b>Fig. 1:</b> Modular diagnostic architecture of the Vision Error Observatory (VEO) framework across Modules M1 to M5. [Note: Architecture diagram generated using AI visualization tools].</em></p>
</div>

---

## Key Empirical Findings

All empirical claims in VEO are strictly grounded in 244 held-out scene-disjoint NYU Depth V2 test images (partitioned from 1,449 samples with seed `42` and zero cross-split scene leakage):

1. **Alignment Polarity Reversal Proof:**
   * Evaluating relative depth models in direct depth space ($d = s \cdot p + b$) produces negative scale slopes ($s < 0$) in **244 / 244 test scenes**, inverting physical geometry.
   * Inverse-depth scale-shift regression ($1/D \approx s \cdot P + b$) guarantees strictly positive scales ($s > 0$) across all 244 scenes and recovers optimal metric depth ($\text{AbsRel}=0.1040, \delta_1=0.8965$).
2. **Cross-Model Failure Fingerprinting:**
   * **Depth Anything V2 Small** fails predominantly on featureless `textureless` planes (19.5%) and `boundary_mixing` (18.0%).
   * **DPT-Hybrid-MiDaS** is significantly more vulnerable to specular floor `reflection` (17.4% vs 13.1%).
3. **Multi-Rater Blinded Human Agreement ($N=50$ Test Regions):**
   * Human inter-rater baseline: Cohen's $\kappa = 0.3478$ (46.0% raw agreement).
   * VEO vs. Human Raters on determinate predictions ($N=29$): Cohen's $\kappa = \mathbf{0.4719}$ with **58.6% multi-class accuracy** (against a 14.3% random baseline).
   * Per-class precision: `textureless` (**87.5%**), `thin_object` (**71.4%**), `boundary_mixing` (**71.4%**).
4. **Explanation Faithfulness:**
   * Controlled synthetic texture injection on `scene_0049` Region 8 collapsed the `textureless` attribution score from $0.6464 \to 0.0000$, confirming explanation faithfulness to physical image content.
5. **Actionable Downstream Improvement:**
   * Selective bilateral smoothing applied exclusively within VEO-diagnosed boundary error masks across 207 test scenes reduced regional boundary RMSE ($0.6629\,\text{m} \to 0.6618\,\text{m}$) and regional AbsRel ($0.2348 \to 0.2340$) without global distortion ($0.4658\,\text{m}$).

---

## Repository Layout

```text
VEO_V2/
├── configs/                   # YAML configuration profiles
│   ├── nyu.yaml               # NYU Depth V2 protocol & evaluation bounds
│   └── hypersim.yaml          # Synthetic Hypersim metric benchmark config
├── data/                      # Local datasets mount (excluded from git)
│   └── nyu/splits/            # Deterministic scene-disjoint partitions (seed 42)
├── docs/                      # Research documentation & protocol records
│   ├── README.md              # Research documentation index
│   ├── RESEARCH_PROTOCOL_V1.md# Frozen scientific protocol & claim boundaries
│   ├── VEO_PROJECT_SPECIFICATION_LOCAL.md # Pipeline interfaces & contracts
│   ├── VEO_PROJECT_INFO_AND_EVALUATOR_RESPONSE.md # Evaluator audit responses
│   └── notes/                 # Theoretical & mathematical derivation notes
├── figures/                   # 300-DPI publication figures
│   ├── README.md              # Figure index & generation documentation
│   ├── fig1_veo_architecture.png
│   ├── fig2_polarity_and_fingerprints.png
│   └── fig3_intervention_and_downstream.png
├── notebooks/                 # Exploratory inspection notebooks
│   ├── README.md
│   └── VEO_v2_legacy.ipynb    # Prototype development notebook
├── outputs/                   # Empirical evaluation records & JSON artifacts
│   ├── nyu_final/test/        # Depth Anything V2 Small test results (5,236 regions)
│   ├── second_model_dpt_hybrid_midas/test/ # DPT-MiDaS test results (3,689 regions)
│   ├── alignment_ablation_full/ # 244-scene alignment ablation records
│   └── rating_set_test_50/    # Blinded multi-rater human ratings & results
├── paper/                     # Publication manuscript package
│   ├── README.md              # Paper specifications & formatting details
│   ├── VEO_conference_paper.docx # IEEE Word manuscript (Times New Roman, 9 tables)
│   ├── VEO_conference_paper.pdf  # Compiled PDF version of the manuscript
│   └── archive/               # Historical draft backups
├── scripts/                   # Standalone reproduction & evaluation scripts
│   ├── README.md              # Script inventory & execution manual
│   ├── build_final_word_paper.py        # Compiles Word paper with active bookmarks
│   ├── generate_paper_figures.py        # Renders all 300-DPI publication plots
│   ├── compute_human_agreement.py       # Computes multi-rater Cohen's kappa
│   ├── run_signal_ablation.py           # Feature sensitivity ablation (5,236 regions)
│   ├── run_faithfulness_intervention.py # Counterfactual texture intervention
│   └── run_downstream_improvement.py    # Boundary bilateral refinement evaluation
├── src/veo_nyu/               # Core modular Python library
│   ├── cli.py                 # Command-line interface commands
│   ├── config.py              # Configuration parsing & validation
│   ├── inference.py           # Model adapter (Depth Anything V2, DPT)
│   ├── metrics.py             # Scale-shift alignment & dense error maps
│   ├── regions.py             # Connected-components segmentation
│   ├── features.py            # 7 visual evidence feature extractors
│   ├── scoring.py             # Heuristic engine & calibrated ambiguity gate
│   └── validation.py          # Multi-rater Cohen's kappa & ablation metrics
├── tests/                     # Unit test suite (14/14 passing)
│   └── test_core.py           # Test coverage for alignment, metrics, and rules
├── pyproject.toml             # Python packaging & dependencies
├── uv.lock                    # Deterministic dependency lockfile
└── .gitignore                 # Clean repository ignore patterns
```

---

## Quickstart & Environment Setup

This project uses [`uv`](https://github.com/astral-sh/uv) for fast, fully deterministic virtual environment management.

### 1. Installation
```powershell
# Clone the repository
git clone https://github.com/username/VEO_V2.git
cd VEO_V2

# Synchronize dependencies with uv
uv sync --extra dev --extra model
```

### 2. Verify Installation (Unit Tests)
```powershell
uv run pytest -v tests/
# Output: 14 passed in ~12s
```

---

## End-to-End Reproduction Guide

Every empirical result, table, and figure in the paper can be reproduced with a single command:

### 1. Generate Publication Figures (300 DPI)
```powershell
uv run python scripts/generate_paper_figures.py
# Saves Fig 1, Fig 2, and Fig 3 into figures/
```

### 2. Run Feature Signal Sensitivity Ablation (Table VIII)
```powershell
uv run python scripts/run_signal_ablation.py
# Evaluates 5,236 regions across 7 signals, writes outputs/nyu_final/test/signal_ablation.json
```

### 3. Verify Explanation Faithfulness via Intervention (Fig. 3a)
```powershell
uv run python scripts/run_faithfulness_intervention.py
# Injects synthetic texture noise on scene_0049 Region 8, proves score collapse 0.6464 -> 0.0000
```

### 4. Evaluate Downstream Boundary Refinement (Table IX, Fig. 3b)
```powershell
uv run python scripts/run_downstream_improvement.py
# Applies localized bilateral filtering across 207 test scenes, reports regional RMSE/AbsRel gains
```

### 5. Compute Blinded Multi-Rater Human Agreement (Table VII)
```powershell
uv run python scripts/compute_human_agreement.py
# Ingests rater1.csv & rater2.csv across 50 regions, computes Cohen's kappa and class precisions
```

### 6. Compile the Final IEEE Conference Paper
```powershell
uv run python scripts/build_final_word_paper.py
# Generates paper/VEO_conference_paper.docx with active citation bookmarks and 9 styled tables
```

---

## License

This project is licensed under the [MIT License](LICENSE).
