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

iBims-1 was planned for its depth-edge annotations. Its servers refused automated download when this was written (HTTP 401 and 429), so it is not part of this run.

## Limits

No model here was retrained, and a single split and a single set of hand-chosen contrasts are reported. Results describe these checkpoints on these datasets.

## Amendments

None yet.
