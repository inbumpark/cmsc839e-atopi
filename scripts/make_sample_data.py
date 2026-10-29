"""Write a tiny synthetic dataset to data/sample/ so the pipeline can be tested before real data arrives.

Images are generated blotches on a skin-tone background (not real lesions); the model's answers
on them are meaningless and only exercise the plumbing.
"""
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

OUT = Path(__file__).resolve().parent.parent / "data" / "sample"

CASES = [
    ("sample-1", "Eczema", 4, {"Eczema": 0.75, "Allergic Contact Dermatitis": 0.25},
     {"duration": "1 to 4 weeks", "body_location": "arm, inner elbow", "texture": "rough or flaky",
      "symptoms": "itching", "other_symptoms": "none", "age": "18-29", "sex": "female",
      "fitzpatrick": "FST2"}),
    ("sample-2", "Psoriasis", 3, {"Psoriasis": 1.0},
     {"duration": "more than one year", "body_location": "leg, knee", "texture": "raised or bumpy, rough or flaky",
      "symptoms": "itching", "age": "40-49", "sex": "male"}),
    ("sample-3", "Urticaria", 5, {"Urticaria": 0.6, "Insect Bite": 0.4},
     {"duration": "less than one day", "body_location": "torso front, back", "texture": "raised or bumpy",
      "symptoms": "itching, burning", "other_symptoms": "none", "fitzpatrick": "FST4"}),
    ("sample-4", "Eczema", 2, {"Eczema": 0.5, "Tinea": 0.5},
     {"body_location": "palm, back of hand", "texture": "rough or flaky", "symptoms": "itching, bleeding"}),
]


def fake_image(seed, path):
    rng = random.Random(seed)
    base = (rng.randint(150, 230), rng.randint(110, 180), rng.randint(90, 150))
    img = Image.new("RGB", (640, 480), base)
    draw = ImageDraw.Draw(img)
    for _ in range(rng.randint(8, 25)):
        x, y, r = rng.randint(80, 560), rng.randint(80, 400), rng.randint(10, 60)
        red = (min(255, base[0] + 20), max(0, base[1] - 60), max(0, base[2] - 50))
        draw.ellipse([x - r, y - r * 0.7, x + r, y + r * 0.7], fill=red)
    img.filter(ImageFilter.GaussianBlur(4)).save(path)


def main():
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    with (OUT / "cases.jsonl").open("w") as f:
        for i, (cid, label, conf, diff, ctx) in enumerate(CASES):
            img = f"images/{cid}_1.png"
            fake_image(i, OUT / img)
            f.write(json.dumps({"case_id": cid, "images": [img], "label": label, "label_confidence": conf,
                                "differential": diff, "context": ctx}) + "\n")
    print(f"wrote {len(CASES)} cases to {OUT / 'cases.jsonl'}")


if __name__ == "__main__":
    main()
