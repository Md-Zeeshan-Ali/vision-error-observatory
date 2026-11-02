# VEO Label Model v3

Written on 2026-11-02, before the model was fitted. It amends `RESEARCH_PROTOCOL_V2.md`, whose hand-set rules failed validation for `reflection` and `textureless` and left `thin_object` almost unused.

## Why the test split is no longer clean

The v2 rules were scored on the official NYU test split, and that result motivated this rebuild. NYU test numbers for v3 are therefore **not confirmatory**, even though nothing is fitted on them. The clean checks are the datasets in Phase 6, which v3 has never seen, and the human study.

## Fitting

- **Data:** failure regions of Depth Anything V2 Small on the official NYU **train** split, with the silver labels of `src/veo/silver.py` unchanged.
- **Model per category:** logistic regression on at most 3 standardised features, chosen by forward selection on scene-grouped 5-fold cross-validated AUROC. A feature is kept only if it improves AUROC by at least 0.005.
- **Label threshold:** the probability that maximises out-of-fold F1.
- **Output:** `configs/label_model_v3.json`, committed before any test scoring.

## Allowed features

Each category may only use features with a physical reason to indicate that cause. Position in the image is excluded everywhere: "high in the image" would predict the wall/ceiling silver label while saying nothing about texture.

| Category | Candidate features |
|---|---|
| textureless | `grad_energy`, `coarse_grad`, `corner_strength`, `contrast`, `log_area` |
| boundary_mixing | `depth_edge_fraction`, `depth_step`, `log_area` |
| thin_object | `thin_fraction`, `thin_image_fraction`, `depth_edge_fraction`, `aspect_ratio`, `log_area` |
| reflection | `highlight_ratio`, `sensor_hole_fraction`, `signed_error`, `value_p95`, `saturation_mean`, `clipped_fraction` |

New features for this rebuild:

- `coarse_grad`: gradient energy after heavier smoothing, for large-scale structure
- `sensor_hole_fraction`: share of the neighbourhood with no sensor reading (Kinect often fails on glass and mirrors)
- `signed_error`: mean (prediction − truth) / truth; positive means the model puts the surface too far away, as when it sees the scene behind glass or in a mirror
- `value_p95`, `saturation_mean`: colour statistics
- `log_area`: log of region area

`sensor_hole_fraction` is specific to active depth sensors, and `signed_error` needs ground truth. Both are fine for an audit, but neither is available in deployment.

## Top label

- Each category's score is compared with its own threshold: ratio = probability / threshold.
- The top label is the category with the highest ratio. VEO abstains (`ambiguous`) when no ratio reaches 1.
- `labels` lists every category with a ratio of at least 1.
- `poor_lighting` has no silver label, so it keeps its v2 rule with threshold 0.5.
- This is what lets a rarer, more specific label such as `thin_object` win over `boundary_mixing`.

## Reporting

- v2 rules and the v3 model are reported side by side, along with a gradient-boosted model and a small CNN as learned references.
- The silver labels are a proxy. Wall/ceiling in particular is a crude stand-in for "textureless", so the human study remains the real test.
- Interventions and the Hypersim test previously used score cut-offs from the v2 rules to choose controls. With probabilities as scores, controls become regions without the label in question.

## Amendments

**2026-11-02, after fitting on train and before any test scoring:**

1. *Direction of `signed_error`.* This document expected positive values on reflective surfaces (the model seeing the scene behind them). The fit gives a negative coefficient: predictions on these surfaces are too near. The feature stays, since selection follows AUROC. A possible explanation, not established here: on mirrors and windows the sensor may measure through or into the reflection, so the "truth" lies behind the surface while the model predicts the surface. If so, some reflection "failures" are ground-truth errors rather than model errors.
2. *Where the labels no longer apply.* `thin_evidence` (protocol v2, amendment 2) only applies to the hand-set rules. The v3 model always uses both thin cues.
