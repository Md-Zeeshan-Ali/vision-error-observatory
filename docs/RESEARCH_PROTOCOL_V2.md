# VEO Research Protocol v2

Frozen on 2026-10-28, before any v2 result on the NYU official test split was computed. Changes after this point must be listed under "Amendments" with a date and a reason. The v1 protocol is superseded (available at the `v1-archive` tag); its corrections are in `ERRATA_V1.md`.

## Data

- **Dataset:** NYU Depth V2 labelled set (`nyu_depth_v2_labeled.mat`, 1,449 frames, 480×640). The SHA-256 of the file is recorded in each run's provenance.
- **Split:** the official train (795) / test (654) split from `splits.mat` (`trainNdxs`, `testNdxs`, 1-based). The loader refuses a split that shares scenes between train and test.
- **Use of splits:** the test split is used only for final reporting. Anything that is fitted or tuned (score scales, gate thresholds, calibration in Phase 3) uses the train split only.
- **Disclosure:** v1 analysed frames from all 1,449 images through a custom split, so v1 exploratory work saw some official-test images. No v2 parameter was chosen by looking at v1 results on those images.

## Ground truth and evaluation mask

- **Error target:** `rawDepths`, the sensor depth projected into the RGB frame. Pixels with no measurement (0) are excluded.
- **Geometry cues:** the dense, in-painted `depths` are used only for geometry features (depth edges, thin structures), never as the error target.
- **Mask:** finite, `0 < depth ≤ 10 m`, inside the Eigen crop (rows 45–470, columns 41–600).

## Models and alignment

- **Primary models:** Depth Anything V2 Small (`depth-anything/Depth-Anything-V2-Small-hf`) and DPT-Hybrid (`Intel/dpt-hybrid-midas`), the two v1 models. Both output relative disparity.
- **Alignment:** per-image least-squares scale and shift in inverse-depth space over the evaluation mask, then inversion to metric depth. Direct-depth, scale-only and no-alignment protocols are reported as an ablation.
- **Inference:** each model's Hugging Face `depth-estimation` pipeline, with the output resized to 480×640.

## Failure regions

- **Failing pixel:** `max(pred / gt, gt / pred) > 1.25` (fails δ1), inside the evaluation mask.
- **Regions:** 4-connected components (amendment 4; this line first said 8-connected) of failing pixels with area ≥ 25 px. No morphological opening, because it erases thin failure bands at edges and on thin objects.
- **Sensitivity analysis:** δ thresholds 1.10, 1.25 (primary) and 1.5625, plus v1's per-image 85th-percentile rule for comparison.

## Features and hypotheses

- **Definitions:** as implemented in `src/veo/features.py` and `src/veo/scoring.py` at the commit that introduced this document.
- **Categories:**

  | Category | Feature | Threshold |
  |---|---|---|
  | textureless | `grad_energy` | scale 8 grey levels/px |
  | boundary_mixing | `depth_edge_fraction`; log-depth edges with \|∇ log d\| > 0.05/px, within 3 px | scale 0.5 |
  | thin_object | `thin_fraction`; disparity top-hat (9 px) > 10% of disparity | scale 0.3 |
  | reflection | `highlight_ratio`; V ≥ 200, S ≤ 50, ≥ 25 above the 21-px local median | ×5 |
  | poor_lighting | brightness below 50, or clipped pixels | — |

- **Occlusion:** occlusion edges are part of `boundary_mixing`, because the two cannot be separated from one image.
- **Abstention:** `ambiguous` when the top score is < 0.40 or the margin to the runner-up is < 0.10.
- **Multiple labels:** `labels` lists every category scoring ≥ 0.5.
- **How the parameters were set:** by hand, from the synthetic behaviour tests in `tests/test_feature_behaviour.py`, before any NYU v2 run. They are not tuned on test data.

## Reported quantities (Phase 2)

- **Global metrics:** per model (AbsRel, RMSE, δ1–δ3), as means over images, with scene-level bootstrap 95% confidence intervals.
- **Failure extent:** the failure pixel fraction per image and the region count.
- **Labels:** per model, each top label's share of regions, multi-label frequencies, and failure pixels per image attributed to each label.
- **Not in scope yet:** statistical comparison between models is deferred to Phase 6 (scene-cluster bootstrap with Holm correction), and agreement with humans or objective labels is deferred to Phases 3–4. Phase 2 makes no claim about whether labels are correct.

## Amendments

**2026-10-28, before any NYU v2 result was computed:**

1. *Speckle-cleanup sensitivity.* Added a fifth failure rule, δ 1.25 with a 1 px morphological opening, to the sensitivity table. It measures how many thin failure regions an opening would erase. The primary rule is unchanged (no opening).
2. *Thin structures without ground truth.* Added `thin_image_fraction`, the share of region pixels on or next to a thin bright or dark image line (top-hat / black-hat of grey level, 9 px window, > 25 grey levels). When `geometry_source` is `prediction`, the thin-object evidence is the larger of the depth cue and this image cue, because a model that misses a thin object usually removes it from its own depth map too. Audit mode (ground-truth geometry) is unchanged.

**2026-10-28, after the first NYU v2 run (post-hoc, disclosed):**

3. *Cap on aligned predictions.* The first run showed DPT-Hybrid with mean AbsRel 12.5 and RMSE 421 m although δ1 was 0.918. In 2 of 654 images the inverse-depth fit maps some pixels to zero or negative disparity, which then became depths of up to 10^6 m. Aligned predictions are now capped at the evaluation range (`max_prediction_m: 10` for NYU; 20 m in the Hypersim test), as in standard NYU evaluation, and all v2 results were regenerated. This changes only pixels whose aligned depth exceeded the cap. It was decided after seeing results, so it is recorded here as a post-hoc correction of a numerical defect, not a pre-registered choice.

**2026-11-03, found while writing up (no result changes):**

4. *Region connectivity.* The "Failure regions" section said 8-connected components, but `src/veo/regions.py` calls `scipy.ndimage.label` with its default structure, which is 4-connected. Every v2 and v3 result was computed with 4-connectivity. The code is kept as it is so that published numbers stay reproducible, and this document now records what was actually done.
