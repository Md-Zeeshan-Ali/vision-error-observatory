import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def create_slide_image(width, height, bg_color, header_bg, title, subtitle, bullets, callout=None, footer="Vision Error Observatory (VEO)"):
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    # Header banner
    draw.rectangle([(0, 0), (width, 110)], fill=header_bg)
    
    # Try system fonts, fall back to default
    try:
        font_title = ImageFont.truetype("arialbd.ttf", 36)
        font_sub = ImageFont.truetype("ariali.ttf", 20)
        font_body = ImageFont.truetype("arial.ttf", 22)
        font_body_b = ImageFont.truetype("arialbd.ttf", 22)
        font_callout = ImageFont.truetype("arialbd.ttf", 20)
        font_footer = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font_title = ImageFont.load_default()
        font_sub = font_title
        font_body = font_title
        font_body_b = font_title
        font_callout = font_title
        font_footer = font_title

    # Title & Subtitle
    draw.text((40, 25), title, fill=(255, 255, 255), font=font_title)
    draw.text((42, 70), subtitle, fill=(210, 230, 255), font=font_sub)

    # Bullets
    y_pos = 145
    for b_bold, b_text in bullets:
        # bullet circle
        draw.ellipse([(45, y_pos + 6), (57, y_pos + 18)], fill=header_bg)
        draw.text((70, y_pos), b_bold, fill=(20, 35, 60), font=font_body_b)
        x_offset = draw.textlength(b_bold + " ", font=font_body_b) if hasattr(draw, 'textlength') else 200
        draw.text((70 + x_offset, y_pos), b_text, fill=(50, 65, 85), font=font_body)
        y_pos += 48

    # Callout Box if present
    if callout:
        c_title, c_text, c_border, c_bg = callout
        box_y1 = y_pos + 15
        box_y2 = height - 60
        draw.rounded_rectangle([(40, box_y1), (width - 40, box_y2)], radius=12, fill=c_bg, outline=c_border, width=2)
        draw.text((65, box_y1 + 20), c_title, fill=c_border, font=font_callout)
        draw.text((65, box_y1 + 55), c_text, fill=(40, 50, 65), font=font_body)

    # Footer
    draw.line([(0, height - 40), (width, height - 40)], fill=(220, 225, 235), width=1)
    draw.text((40, height - 30), footer, fill=(120, 135, 150), font=font_footer)

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def main():
    out_dir = Path("web_app")
    out_dir.mkdir(exist_ok=True)
    video_path = out_dir / "veo_explainer.mp4"

    width, height = 1280, 720
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        print("Error: Could not open VideoWriter.")
        return

    # Color themes
    c_blue = (15, 45, 95)
    c_red = (150, 25, 25)
    c_amber = (180, 95, 10)
    c_green = (15, 105, 65)

    slides_config = [
        # Slide 1: The Leaderboard Illusion
        {
            "header_bg": c_blue,
            "title": "SCENE 1: ONE NUMBER HIDES A LOT",
            "subtitle": "A good average says little about where and why a depth model fails",
            "bullets": [
                ("The Average:", "Depth Anything V2 Small scores AbsRel 0.051 and delta1 97.3% on NYU Depth V2."),
                ("What It Hides:", "2.7% of pixels still fail, in 6,325 regions across 654 test images."),
                ("The Question:", "Where do these failures sit, and what in the scene goes with them?")
            ],
            "callout": (
                "THE GOAL:",
                "Turn one score into a map of failure regions, with explanations that can be checked.",
                c_red, (255, 240, 240)
            ),
            "duration_sec": 7
        },
        {
            "header_bg": c_amber,
            "title": "SCENE 2: WHAT VEO DOES",
            "subtitle": "A model-agnostic audit of any depth prediction",
            "bullets": [
                ("1. Align:", "Fit each prediction to measured depth in disparity space."),
                ("2. Find Failures:", "Group pixels failing the delta1 test, on sensor-measured pixels only."),
                ("3. Measure:", "Compute image and geometry cues for each failure region."),
                ("4. Label:", "Small interpretable models name a likely cause, or abstain."),
                ("5. Test:", "Edit the image, re-run the model, and check whether the error responds.")
            ],
            "callout": (
                "THE DESIGN:",
                "Rules were fixed before results, and every reported number is computed by code.",
                c_green, (240, 253, 245)
            ),
            "duration_sec": 8
        },
        {
            "header_bg": c_green,
            "title": "SCENE 3: WHAT HELD UP",
            "subtitle": "Against NYU annotations, synthetic Hypersim scenes and model re-runs",
            "bullets": [
                ("Depth Boundaries:", "AUROC 0.85 on NYU and 0.81 on Hypersim, a dataset it never saw."),
                ("Thin Objects:", "AUROC 0.73 on NYU and 0.81 on Hypersim."),
                ("Higher Resolution:", "Helps boundary regions more than matched controls (exploratory)."),
                ("Model Size:", "Larger Depth Anything models reliably reduce error: 0.051, 0.046, 0.043."),
                ("Specular Light:", "The more of it a region has, the more error falls without it (rho = -0.26).")
            ],
            "callout": (
                "THE TAKEAWAY:",
                "VEO's boundary label is its strongest result, and specular light really does raise error.",
                c_blue, (240, 245, 255)
            ),
            "duration_sec": 8
        },
        {
            "header_bg": c_red,
            "title": "SCENE 4: WHAT DID NOT",
            "subtitle": "Reported as plainly as the successes",
            "bullets": [
                ("Reflection:", "AUROC 0.83 on NYU but 0.39 on Hypersim: it relies on Kinect sensor holes."),
                ("Textureless:", "Near chance: AUROC 0.58 on NYU and 0.50 on Hypersim."),
                ("Image Edits:", "Texture, brightening and thin-object resolution tests did not support the labels."),
                ("ViT vs Hybrid:", "No reliable difference between DPT-Large and DPT-Hybrid.")
            ],
            "callout": (
                "STILL OPEN:",
                "A 300-region blinded human study is ready and needs raters.",
                c_amber, (255, 250, 235)
            ),
            "duration_sec": 8
        },
        {
            "header_bg": c_blue,
            "title": "SCENE 5: EXPLORE THE RESULTS",
            "subtitle": "Real test images with their actual failure regions",
            "bullets": [
                ("Pick a Scene:", "Four NYU test images, one for each main failure label."),
                ("Compare Views:", "See the photo, the model's depth, the sensor's depth and the error."),
                ("Click a Region:", "Inspect its measurements and how VEO ranks each explanation."),
                ("Go 3D:", "Fly through true and predicted geometry, then see how the model works.")
            ],
            "callout": (
                "READ IT AS:",
                "A description of where a model fails and what goes with it, not proof of cause.",
                c_amber, (255, 250, 235)
            ),
            "duration_sec": 6
        }
    ]

    print("Generating VEO Explainer Video (720p HD, 30 FPS)...")

    # Render frames with smooth 15-frame cross-fades
    fade_frames = 15
    prev_bgr = None

    for idx, slide in enumerate(slides_config):
        curr_bgr = create_slide_image(
            width=width, height=height,
            bg_color=(250, 252, 255),
            header_bg=slide["header_bg"],
            title=slide["title"],
            subtitle=slide["subtitle"],
            bullets=slide["bullets"],
            callout=slide["callout"]
        )

        total_frames = int(slide["duration_sec"] * fps)
        
        # If there is a previous slide, cross-fade transition
        if prev_bgr is not None:
            for f in range(fade_frames):
                alpha = f / float(fade_frames)
                blended = cv2.addWeighted(prev_bgr, 1.0 - alpha, curr_bgr, alpha, 0)
                writer.write(blended)
            steady_frames = total_frames - fade_frames
        else:
            steady_frames = total_frames

        # Steady display
        for _ in range(steady_frames):
            writer.write(curr_bgr)

        prev_bgr = curr_bgr
        print(f"  - Rendered Slide {idx + 1}/5 ({slide['duration_sec']}s)")

    writer.release()

    print(f"\n[OK] Explainer video successfully generated and saved to:")
    print(f"     - {video_path.resolve()}")

if __name__ == "__main__":
    main()

