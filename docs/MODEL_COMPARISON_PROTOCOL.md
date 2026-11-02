# VEO Model Comparison and External Checks (Phase 6)

Written on 2026-11-02, before any result from these models or from DIODE was computed. Amendments go at the end with a date and a reason.

## Models

| Model | Hugging Face id | Output | Role |
|---|---|---|---|
| Depth Anything V2 Small | `depth-anything/Depth-Anything-V2-Small-hf` | disparity | primary; size contrast |
| Depth Anything V2 Base | `depth-anything/Depth-Anything-V2-Base-hf` | disparity | size contrast |
| Depth Anything V2 Large | `depth-anything/Depth-Anything-V2-Large-hf` | disparity | size contrast |
| DPT-Hybrid | `Intel/dpt-hybrid-midas` | disparity | backbone contrast |
| DPT-Large | `Intel/dpt-large` | disparity | backbone contrast |
| ZoeDepth NYU+KITTI | `Intel/zoedepth-nyu-kitti` | metric depth | descriptive only: it was trained on NYU |
| Depth Pro | `apple/DepthPro-hf` | metric depth | descriptive only |

Metric outputs are inverted to disparity, then aligned like the others, so every model is evaluated on structure rather than absolute scale. Each model's alignment table shows whether its fitted scale came out positive, which checks the declared output type.

## Contrasts on the NYU official test split

| Contrast | Models | Question |
|---|---|---|
| Size | DA-V2 Small vs Base; Base vs Large | Same family and training, different capacity |
| Backbone | DPT-Hybrid vs DPT-Large | Same decoder, convolutional hybrid vs pure ViT |
| Family | DA-V2 Small vs DPT-Hybrid | v1's "fingerprint" question |

These contrasts are the only inferential tests. Differences between families cannot be attributed to architecture alone, because training data also differs. Only the backbone contrast speaks to architecture, and even there it isn't controlled for parameter count.

## Outcomes and statistics

- **Per-image outcomes:** AbsRel, δ1, failing-pixel share, and for each label (5 causes plus `ambiguous`) the share of valid pixels in regions with that top label.
- **Unit of analysis:** the scene. Paired per-image differences are averaged within each scene, then scene means are compared.
- **Reported:** mean difference, scene-level bootstrap 95% interval, sign-flip permutation p-value, Holm correction across all outcomes within a contrast, and Cohen's d_z.
- **Claim rule:** a difference is called reliable only if the Holm-adjusted p is below 0.05 and the interval excludes zero. Label-share differences are descriptions of VEO's labels, whose validity is limited per category.

## External checks of label model v3

The v3 model was fitted on NYU train and has already been compared with NYU test, so these are its clean checks. It is not refitted on them.

- **Hypersim test scenes:** silver labels from Hypersim's own semantic and instance annotations (NYU40 classes), using the same definitions as NYU: window, mirror, television, whiteboard for reflection; wall and ceiling for textureless. AUROC of v2 rules vs v3. There are no sensor holes in synthetic depth, so `sensor_hole_fraction` is uninformative here, which handicaps v3 on reflection.
- **DIODE indoor validation (325 frames, laser scanner):** no semantic labels exist, so this checks model metrics, failure extent and label distributions on another sensor, not label correctness. Frames are 768×1024, the evaluation range is 10 m, and no in-painted depth exists, so geometry cues come from the scan itself.

## Datasets not included

iBims-1 was planned for its depth-edge annotations. Its servers refused automated download when this was written (HTTP 401 and 429), so it is not part of this run. See the 2026-11-02 amendment.

## Limits

No model here was retrained, and a single split and a single set of hand-chosen contrasts are reported. Results describe these checkpoints on these datasets.

## Amendments

### 2026-11-02: Marigold, iBims-1 and the fingerprint decision

Written after the seven models above were reported on NYU and DIODE, and before any Marigold or iBims-1 result was computed. iBims-1's annotation coverage was inspected first, without any model output. The earlier section "Datasets not included" no longer applies: the archive could be downloaded on this date.

**Marigold.** `prs-eth/marigold-depth-v1-1` adds a diffusion model, run on NYU test, DIODE and iBims-1. Settings: half precision, 4 denoising steps, ensemble size 1, processing resolution 768, seed 42 for every image. These are lighter than the 10-member ensemble of the Marigold paper, so results describe this configuration. Its output is affine-invariant depth rather than disparity, so it is aligned by scale and shift in depth space, unlike the others. One contrast is added: DA-V2 Small vs Marigold (`family_dav2_small_vs_marigold`), under the same rules. Training data differs, so it does not speak to architecture.

**iBims-1 (100 images, 11 room types, laser scanner).** Evaluation uses the same rules as DIODE: 10 m range, no crop, and geometry cues from the scan itself. Pixels marked invalid by the dataset are unmeasured. Transparent pixels stay measured wherever the laser returned a value, although iBims-1's own evaluation removes them, because the reflection check needs them. Metrics here therefore can't be compared with published iBims-1 numbers. All 8 models are run. The unit of analysis is the room type, which is conservative because iBims-1 does not identify individual rooms.

**External label check on iBims-1.** Depth Anything V2 Small only, as on Hypersim. Label model v3 is not refitted. A region gets a silver label when at least 30% of its pixels support it, the same share as NYU.

| Label | Support | Notes |
|---|---|---|
| `boundary_mixing` | Within 3 px of an annotated depth edge | Only the 85 images with edge annotations, since an empty map may mean "not annotated" |
| `textureless` | On a wall mask | Closest match to NYU's wall and ceiling. Sensitivity: wall, floor or table masks. Masks cover selected planes, so some true positives are missed |
| `reflection` | On a transparent mask | Differs from NYU's reflective classes. A laser may also miss glass, so AUROC of `sensor_hole_fraction` alone is reported to show how much of the result is that cue |
| `thin_object`, `poor_lighting` | Not checked | No annotation exists |

**Fingerprint decision.** The blueprint's rule is that v1's "failure fingerprint" claim either holds across datasets with confidence intervals, or is cut. A fingerprint means a different mix of failure types, not just more failure, so the outcome is composition: for each label, the share of an image's region pixels with that top label. Images where either model has no regions are left out. This is a separate family from the outcomes above, with its own Holm correction, so earlier results don't change.

- **Contrasts tested:** Family (DA-V2 Small vs DPT-Hybrid) and Backbone (DPT-Hybrid vs DPT-Large).
- **Datasets and units:** NYU test (scenes), iBims-1 (room types), and DIODE (scans). DIODE has only 3 scenes, too few for a sign-flip test to reach significance, so its 10 scans are used and are noted as not independent.
- **Pass rule:** a label passes only if it is reliable by the claim rule above on all three datasets, with the same sign. Otherwise it fails.
- **Verdict:** the claim is cut if no label passes in a contrast. A passing label supports only a difference in VEO's label shares. It is called a difference in failure type only if that label scored AUROC of at least 0.7 on both Hypersim and iBims-1.
