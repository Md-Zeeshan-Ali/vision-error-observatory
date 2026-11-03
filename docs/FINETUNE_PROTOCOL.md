# VEO Guided Fine-Tuning (Phase 7)

Written on 2026-11-02, after a pilot on the NYU train split and before any fine-tuned model was scored on the test split. Amendments go at the end with a date and a reason.

## Question

Can VEO's diagnosis make a model better where it matters? This tests whether up-weighting the regions VEO diagnosed reduces error on those kinds of pixels more than up-weighting the same amount of pixels at random. Ground truth is used to find the training regions, which is legitimate at training time. The resulting model needs no ground truth at inference.

## Model and training

- **Model:** Depth Anything V2 Small. The DINOv2 backbone is frozen, and only the decoder is trained.
- **Data:** the 795 NYU official train images. The test split is never used for training or for choosing settings.
- **Loss:** L1 error in disparity against sensor-measured pixels, after a per-image scale and shift fit, divided by the image's mean disparity. The fit is unweighted, as in evaluation.
- **Settings:** AdamW, learning rate 3e-5, weight decay 0.01, cosine decay, 4 epochs, batch size 4, half precision, random horizontal flips.
- **Seeds:** 0, 1 and 2. Within a seed, every condition sees the same batch order and flips.

**Pilot.** The learning rate and epoch count came from one run of the uniform condition on 90% of train scenes, scored on the other 10%. Learning rate 1e-5 reached AbsRel 0.0474 and 3e-5 reached 0.0465, from 0.0525 for the pretrained model. The higher rate was level by epoch 4. Final runs use all 795 images.

## Conditions

| Condition | Up-weighted pixels |
|---|---|
| `uniform` | None |
| `veo_boundary` | Regions the pretrained model failed on, with top label `boundary_mixing` |
| `random_boundary` | The same masks, each moved to a different training image |
| `veo_textureless` | Regions the pretrained model failed on, with top label `textureless` |
| `random_textureless` | The same masks, each moved to a different training image |

The blueprint targets "the categories VEO found dominant", and on the train split that depends on how it is counted. `textureless` has 62% of region pixels, and `boundary_mixing` has 46% of regions. Both are tested. `boundary_mixing` is the only label with support from external annotations and image edits, and `textureless` scores near chance, so comparing them also shows whether a label's validity matters here.

Moving masks to other images keeps their sizes and shapes but breaks their link to the scene. Each target's weight gives its pixels 20% of the training set's loss weight: 118.8 for `boundary_mixing` (0.21% of pixels) and 15.9 for `textureless` (1.55%). Random conditions use the same weight. Some moved pixels land on unmeasured areas, so random conditions carry slightly less extra weight.

## Outcomes on the test split

Outcomes use annotations, not VEO's labels, because regions differ between models and labels could improve by changing what they call things.

- **Boundary outcome:** AbsRel on pixels within the silver `boundary_mixing` map (near annotated occluding boundaries).
- **Textureless outcome:** AbsRel on pixels within the silver `textureless` map (walls and ceilings).
- **Overall:** AbsRel, δ1 and failing pixels, with the same alignment and failure rule as every other result.

## Tests

Each image's outcome is averaged over seeds, then over its scene. Statistics are the scene bootstrap and sign-flip permutation used elsewhere.

**Primary, Holm-corrected as a pair:**

1. `veo_boundary` minus `random_boundary` on the boundary outcome.
2. `veo_textureless` minus `random_textureless` on the textureless outcome.

**Claim rule.** VEO-guided weighting is supported for a target only if all of these hold: Holm-adjusted p below 0.05, the 95% interval entirely below zero, and all three seeds showing a difference in the same direction. The seed check is there because scene statistics don't capture variation between training runs.

**Secondary, Holm-corrected together, descriptive:**

- Each VEO condition against `uniform` on its own outcome.
- Each VEO condition against its random control on overall AbsRel, to show whether a gain elsewhere costs accuracy.
- Specificity: `veo_boundary` against `veo_textureless` on the boundary outcome, and the reverse on the textureless outcome. Their total up-weighting is matched by loss share, not by area.

**Context:** `uniform` against the pretrained model on overall AbsRel. The pretrained model's results from this code are also compared with the stored NYU results, to confirm that scoring matches.

## Limits

This is one model, one weighting strength and one dataset, and the training and test images come from the same dataset and sensor. A positive result would show that the diagnosis can steer fine-tuning on NYU, not that the labels name true causes. A negative result would not rule out other weightings.

## Amendments

### 2026-11-03: results

The rules above were applied as written, and VEO-guided weighting is not supported for either target ([results](../results/finetune/guided_finetune.json), [pilot](../results/finetune/pilot.json)).

| Test (first minus second) | Difference [95% CI] | Holm p | Seeds agree |
|---|---|---|---|
| Boundary outcome: `veo_boundary` vs `random_boundary` | +0.0202 [+0.0180, +0.0224] | 0.0002 | Yes |
| Textureless outcome: `veo_textureless` vs `random_textureless` | +0.0035 [+0.0019, +0.0051] | 0.0002 | Yes |

The direction is the opposite of the claim: the VEO-guided models were reliably worse than their random controls on their own targets. Overall AbsRel was also worse (+0.0028 and +0.0019), and the specificity contrasts point the same way. Random controls stayed close to `uniform`. Plain fine-tuning improved overall AbsRel against the pretrained model (−0.0051 [−0.0059, −0.0042]). This code's scoring of the pretrained model matches the stored NYU results (mean absolute per-image difference 0.00007).

**Added after seeing the results, and exploratory.** To tell a working mechanism from a bug, [`finetune_diagnostic.py`](../experiments/finetune_diagnostic.py) scored the seed 0 models on the training images. On its own targeted regions, `veo_boundary` reached AbsRel 0.173, against 0.312 for `uniform` and 0.320 for `random_boundary`, and on all other pixels it was worse than `uniform` (0.0427 against 0.0388). `veo_textureless` shows the same pattern (0.242 against 0.315). So the weighting acts as intended, and the gain does not carry over to test images. This points to the model fitting those specific regions, possibly including sensor errors at boundaries, though no test here separates those explanations.

`veo_textureless` had slightly fewer failing pixels than `uniform` (0.0219 against 0.0225), despite its worse AbsRel. This is an observation and was not a registered test.

**Consequence.** As the blueprint's fallback states, no downstream benefit is claimed. No other weighting strength was tried after seeing these results, because choosing one that way would not be a fair test.

### 2026-11-03: weaker weights and a hard-example baseline (fixed before these runs)

Two reviewers of the paper draft raised two points. The primary test used one, very strong weighting (118.8 per boundary pixel). It also lacked the comparison that isolates the diagnosis from the failure mask itself. Both are added here, with every rule fixed before any of these runs started. Every result will be reported, whatever it shows, and none of them replaces the primary test above.

- **Weaker weights.** The four targeted conditions are repeated with the targeted pixels carrying 5% and 10% of the loss, instead of 20%: `veo_boundary_share05`, `random_boundary_share05`, `veo_textureless_share05`, `random_textureless_share05`, and the same with `share10`. Everything else is unchanged: seeds 0 to 2, settings, masks and outcomes.
- **Hard-example baseline.** `failure_weighted` up-weights every failure region of the pretrained model on the train split, whatever its label, so those pixels carry 20% of the loss. It asks whether the label adds anything beyond "this region failed".
- **Tests (Holm within each family, same claim rule as the primary test).**
  - Sweep family (4 tests): VEO minus random at each weaker share, on the matching outcome (boundary pixels or wall and ceiling pixels).
  - Hard-example family (2 tests): `veo_boundary` minus `failure_weighted` on boundary pixels, and `veo_textureless` minus `failure_weighted` on wall and ceiling pixels, all at 20%.
  - Descriptive only: `failure_weighted` minus `uniform` on both outcomes and on overall AbsRel.
- **Reading.** A supported sweep test would show that a weaker weight helps where the strong one did not. Since the strengths were chosen after the primary result, that would be evidence for a dose effect, not a replacement for the failed primary claim. A supported hard-example test would show that the label adds value beyond the failure mask.

Command: `python experiments/guided_finetune.py --sweep`, which writes `results/finetune/weight_sweep.json`.

### 2026-11-03: weight sweep and hard-example results

The amendment above was committed (90a1317) before the first of these runs started, and its rules were applied as written ([results](../results/finetune/weight_sweep.json)). Differences are VEO-guided minus control on the matching outcome, pooled over seeds 0 to 2.

| Test | Difference [95% CI] | Holm p | Seeds agree | Supported |
|---|---|---|---|---|
| Boundary, 5%: `veo` vs `random` | +0.0053 [+0.0038, +0.0067] | 0.0004 | Yes | No |
| Textureless, 5%: `veo` vs `random` | +0.0003 [−0.0003, +0.0009] | 0.34 | Yes | No |
| Boundary, 10%: `veo` vs `random` | +0.0114 [+0.0096, +0.0133] | 0.0004 | Yes | No |
| Textureless, 10%: `veo` vs `random` | +0.0010 [−0.0003, +0.0022] | 0.25 | Yes | No |
| Boundary, 20%: `veo_boundary` vs `failure_weighted` | +0.0180 [+0.0167, +0.0193] | 0.0002 | Yes | No |
| Textureless, 20%: `veo_textureless` vs `failure_weighted` | +0.0013 [+0.0008, +0.0018] | 0.0002 | Yes | No |

No sweep test is supported. On boundary pixels the harm shrinks with the share (+0.0202 at 20%, +0.0114 at 10%, +0.0053 at 5%) and stays reliable; on wall and ceiling pixels the weaker shares make no reliable difference. No hard-example test is supported either: VEO guidance was worse than up-weighting every failure region. Descriptively, `failure_weighted` was itself worse than `uniform` on boundary pixels (+0.0027), wall and ceiling pixels (+0.0025) and overall AbsRel (+0.0015).

**Consequence.** No weighting strength or target tried here improves the model, and the label adds nothing over the failure mask as a training signal. No further strengths will be tried on this design.
