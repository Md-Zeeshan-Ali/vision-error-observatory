# VEO Human Study Protocol (Phase 4)

Pre-registered on 2026-10-28, before any rating was collected. It extends `RESEARCH_PROTOCOL_V2.md`. Changes after data collection starts go under "Amendments" with a date and a reason.

## Purpose

Measure how well VEO's failure labels agree with careful human judgement. Also check the objective (silver) labels from Phase 3 against the same judgements. The v1 study (50 regions, 2 raters) was too small to support conclusions; this study replaces it.

## Items

- **Source:** failure regions of the primary model (Depth Anything V2 Small) on the official NYU test split, analysed under protocol v2.
- **Sample:** up to 40 regions per VEO top label (`textureless`, `boundary_mixing`, `thin_object`, `reflection`, `poor_lighting`, `ambiguous`), plus 60 regions drawn uniformly from the rest. That is about 300 items, with at most 3 per image (`veo prepare-study`, seed 42).
- **Weights:** each item records its inclusion probability. Unweighted results describe the sample; weighted results (1 / inclusion probability) estimate the whole test set.
- **Practice:** 20 regions from the **train** split whose annotation-based label is a single category covering at least 60% of the region, balanced across categories. Practice answers get feedback and are never analysed as ratings.

## What raters see

- Through the offline tool (`index.html` in the study folder): the whole image with the region outlined, and four aligned close-ups (colour image, true depth, predicted depth, error).
- Nothing from VEO. No scores, labels or file names that reveal the image. Item order is randomised per rater.
- **Answers:** one primary cause (the five categories or "Other / cannot tell"), every cause that applies, a confidence rating, and optional notes.

## Raters

- **Number:** at least 3, and at least 1 not involved in building VEO.
- **Preparation:** every rater reads `RATER_GUIDELINE.md` and completes the practice round.
- **Exclusion:** raters whose practice answers match the reference on fewer than 40% of items are excluded from the primary analysis. They are reported, and included in a sensitivity analysis.

## Analysis (`experiments/human_study.py`)

1. **Agreement among raters:** Krippendorff's α (nominal; tolerates skipped items), Fleiss' κ on items every included rater answered, and mean pairwise Cohen's κ.
2. **Consensus:** Dawid–Skene over the primary labels of included raters, as the primary method. Majority vote is reported as a check, with the share of items where it agrees and the number of ties.
3. **VEO against the consensus,** with 95% intervals from resampling scenes (1,000 replicates):
   - accuracy, Cohen's κ and macro-F1 over all items
   - accuracy weighted to the test set
   - coverage (share of items where VEO did not abstain) and selective accuracy on those items, against the majority-class baseline on the same items
   - per-class precision, against how often the consensus picks that class at all (its base rate)
4. **Multiple labels:** a label counts for an item when at least half its raters chose it. Per category, VEO's precision and recall, and the silver labels' precision and recall, against these human labels.
5. **Calibrated model:** the Phase 3 calibrated model's labels scored the same way, as a comparison with VEO's hand-set rules.

## Pre-registered checks

These are reported as pass or fail, whatever the outcome:

- **Raters agree enough for the task to be meaningful:** Krippendorff's α ≥ 0.4.
- **VEO's committed labels beat the majority-class baseline:** the lower bound of selective accuracy is above the upper bound of the baseline.
- **At least 3 classes where VEO's precision is clearly above chance:** the lower bound of precision is above the upper bound of that class's base rate.

A failed check is reported as a negative result, not explained away. If rater agreement itself is low, the main finding becomes how hard this attribution task is, with the released annotations as the contribution.

## Release

The rating files, manifest and analysis output are released together (CC-BY 4.0) with a short datasheet: sampling, rater background, tool version and instructions.

## Amendments

### 2026-11-04: what happened so far, and the analysis plan fixed before the last rater

Written before rater R5 starts, so the plan below cannot be shaped by R5's answers. Everything here will be reported.

**Ratings collected.** All on study `5ea5b0bb5eb3` (300 items, 20 practice items), with the round-1 guideline and tool version 1.

| Rater | When (UTC) | Background | Practice agreement | SHA-256 of the exported file |
|---|---|---|---|---|
| R1 | 2026-11-03 | Co-author | 0.50 | `5a17266008ecce50aad2cccef6c85f2666bdde3ad609e9865959dbb4bd037916` |
| R2 | 2026-11-03 | Co-author | 0.55 | `a81635ca617205a9e33a4407eea7592a958ff70db124e6b2e9411c36b2bd8efa` |
| R3 (file ID `r3`) | 2026-11-03 | Co-author | 0.65 | `89142088838ebb922f8b306dd0af6f10a2d1daabf2caa387373d860f51db2499` |
| R4 (file ID `r1`) | 2026-11-04 | Not involved in building VEO | 0.65 | `b513c5c3ede83020b89a3581b6e3e9e17e5472b1ae7c4b1bea1b8362051c6b68` |

The manifest's SHA-256 is `88e51ef02ca1b718a01e5820de19baedf96d63b4e38109d43aecebe2ea10840f`. The rating files are released when the study closes, not before, so a later rater cannot see earlier answers.

**Deviations, in order.**

1. All three planned raters were co-authors, against the requirement of at least one rater not involved in building VEO.
2. R1's file shows all 300 ratings saved within 1.3 minutes. The tool's review pass re-saves every answer with a new time, so R1's real pace cannot be checked. R1 started 3 hours 20 minutes before finishing.
3. The analysis of R1 to R3 was run on 2026-11-03: Krippendorff's α 0.193, VEO against the consensus accuracy 0.210 and κ 0.063, and all three pre-registered checks failed.
4. **After seeing that result,** the team had a person outside the project rate the same items (R4), first intended to replace R1. Both are kept. With R4 in place of R1, α was 0.338; with all four raters, α 0.279. All checks failed in every set.
5. **Also after seeing the results,** the team decided to add one more rater (R5), first intended to replace R2. R2 is kept.

Adding raters after seeing results is a deviation in itself. It is reported here so that readers can judge it.

**Plan for the final analysis, fixed now.**

- **R5** rates the same 300 items with the same package, guideline and tool, alone and without seeing other answers. R5's background is recorded with the results. No further raters are added and none is removed.
- **Primary analysis:** every rater who passes the practice threshold (the rule above), so R1 to R5.
- **Secondary analyses, all reported next to the primary:**
  - R3, R4 and R5, the set the team chose on 2026-11-04;
  - R1, R2 and R3, the roster as originally planned.
- The pre-registered checks are reported as pass or fail for every set, and a check that passes in one set but fails in the primary analysis is described as such.
