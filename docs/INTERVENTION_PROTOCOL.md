# VEO Intervention Tests (Phase 5)

Recorded on 2026-11-02, **after** the results. The test code was committed before any real result existed (commits `84c276a` and `b529963`), but the design was only described in code and commit history, so this document is a record and not a pre-registration. No test, threshold or sample size was changed after seeing results.

## NYU tests (`experiments/causal_interventions.py`)

- **Design:** up to 120 regions per test, each paired with a matched control (similar area, depth and starting error; a different label). Both get the same edit, the depth model is re-run, and the whole prediction is re-aligned.
- **Outcome:** paired difference = (change in labelled-region AbsRel) minus (change in control AbsRel). Negative supports the label.
- **Statistics:** scene-level bootstrap 95% interval, sign-flip permutation test, Holm correction across the tests. A test supports its label only if the interval's upper bound is below 0 and the Holm-adjusted p is below 0.05.

| Test | Labelled regions | Edit |
|---|---|---|
| texture | top label `textureless` | fine zero-mean texture added inside the region |
| exposure | top label `poor_lighting` | gamma 0.5 lift inside the region |
| resolution_boundary | top label `boundary_mixing` | model run at 1.5x input size |
| resolution_thin | `thin_object` among labels | model run at 1.5x input size |

## Hypersim test (`experiments/hypersim_reflection.py`)

- **Data:** 3 seeded frames from each official test scene with the standard camera.
- **Edit:** the non-diffuse (residual) layer is removed; both versions are tone-mapped at the same exposure.
- **Outcomes:** paired test of `reflection` regions against matched controls; Spearman correlation of a region's specular light with its change in error; AUROC of VEO's reflection score for strongly specular regions (at least 10 grey levels) and annotated reflective surfaces; whole-image AbsRel change.

## Limits

These edits test specific interventions. A null result means the edit did not help labelled regions more than controls; it does not rule out the cause, since an edit may fail to remove it. Added noise is not real surface texture, and a larger input does not add detail the camera missed.
