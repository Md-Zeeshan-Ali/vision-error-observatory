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

None yet.
