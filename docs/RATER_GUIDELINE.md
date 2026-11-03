# Rater Guideline: Why Did the Depth Model Fail Here?

Thank you for helping. Each item shows a region where a depth-estimation model got the distance to surfaces wrong. Your job is to judge **what in the scene most likely caused the mistake**. There are no trick questions, and "cannot tell" is a valid answer.

## Setting up

1. Open `index.html` from the study folder in a recent browser (Chrome, Edge, Firefox or Safari). It works offline.
2. Enter the rater ID the study team gave you.
3. Do the practice round first. After each practice item you will see a reference answer derived from the dataset's object annotations. It is a guide, not always the only reasonable answer.
4. Rate the main items. Your progress is saved in that browser. When you finish, click **Export ratings** and send the downloaded file to the study team.

Plan for about 45–60 minutes. Take breaks; consistent attention matters more than speed.

## What you see

- **Left:** the whole photo with the failure region outlined in yellow.
- **Right, four close-ups of the same area:**
  1. the colour photo
  2. the true depth
  3. the model's predicted depth
  4. the error (brighter means worse; grey means the depth sensor had no reading)

  In both depth panels, near surfaces are bright and far ones are dark.

Compare the true and predicted depth. Where do they differ, and what is in the photo there?

## The causes

Pick the **primary cause**: the one that best explains the error. Then tick **all causes that apply**.

| Cause | Choose it when | Typical examples |
|---|---|---|
| **Textureless surface** | The error sits on a large, plain area with little visual detail, so the model has nothing to anchor its estimate to | Painted walls, ceilings, plain doors, whiteboards seen flat-on |
| **Depth boundary / occlusion edge** | The error hugs the edge between a nearer and a farther surface; the prediction blurs or shifts the edge | The outline of a chair against a wall, a door frame, the edge of a table |
| **Thin object** | A narrow object (thinner than a finger's width in the image) is missing, thickened or blended into the background in the prediction | Chair and table legs, poles, lamp stands, cables, rails |
| **Reflective or transparent surface** | The surface mirrors or shows through other things, so the model sees the reflected or transmitted scene instead of the surface | Mirrors, windows, glass doors, TV or monitor screens, glossy floors with visible reflections |
| **Too dark or overexposed** | The area is so dark, or so washed-out, that the photo itself shows little | Dark corners, under furniture, bright windows that are pure white |
| **Other / cannot tell** | None of the above fits, several fit equally well, or the images do not let you decide | — |

## Tips

- **Judge from the images, not from expectations.** A wall is "textureless" only if it actually looks plain here; a wall covered in posters is not.
- **Brightness alone is not reflection.** A white wall is bright but reflects nothing. Choose reflection only when you can see, or strongly suspect, a mirror-like or see-through surface.
- **Edges vs. thin objects:** if the problem is one long edge between two surfaces, choose *depth boundary*. If the object itself is narrow (both of its sides are close together), choose *thin object*.
- **Use "Other / cannot tell" honestly.** It is better than a guess, and it is analysed separately.
- **Confidence:** *High* if you would bet on it, *Medium* if it is your best judgement, *Low* if you are mostly unsure.

## Keyboard shortcuts

`1`–`6` select the primary cause · `H` / `M` / `L` set confidence · `Enter` saves and moves on.

## Privacy

Your ratings are stored with your rater ID only. Please do not discuss items with other raters until everyone has finished, because the study measures independent judgements.
