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

None yet.
