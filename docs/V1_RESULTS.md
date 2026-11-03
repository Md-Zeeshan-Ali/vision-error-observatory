# VEO v1 Results

The original study, kept for the record. It is superseded: see the project README for current results and [ERRATA_V1.md](ERRATA_V1.md) for what was corrected.

## Results

All v1 results use 244 NYU Depth V2 images from a custom scene-disjoint test split (1,449 labeled samples partitioned with seed `42`; this is **not** the official NYU 654-image test split, so numbers are not directly comparable with published results). Result files are in [`results/v1/`](../results/v1/).

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

## Known limitations
* Ground truth is NYU's in-painted `depths`; values in sensor-failure areas (glass, mirrors, dark regions) are filled in, which inflates apparent `reflection` / `poor_lighting` errors.
* Feature thresholds are hand-set, not calibrated against labels. The `occlusion` scale effectively disables that category (22 / 5,236 regions), even though depth discontinuity is the feature most correlated with error (r = 0.45).
* Some features do not isolate their category: `reflection` also fires on bright white walls, and `thin_object` uses bounding-box aspect ratio.
* The human study is small (50 regions, 2 raters) and the raw rating files are not in the repository.
