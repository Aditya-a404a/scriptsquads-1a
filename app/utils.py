import os
import fitz  # PyMuPDF
import cv2
import numpy as np
from ultralytics import YOLO
import pytesseract
import json

# Configuration
model_path = "./models/yolov11s_best.pt"
input_dir  = "./input"
output_dir = "./output"

# Make sure output dir exists
os.makedirs(output_dir, exist_ok=True)

# Initialize YOLO model once
torch_model = YOLO(model_path)

# Helper to process one PDF path and return the outline dict
def process_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    ans = {"title": "", "outline": []}
    outline = []
    dpi = 170 if len(doc) < 5 else 100

    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        pix = page.get_pixmap(dpi=dpi)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            image = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        elif pix.n == 1:
            image = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        else:
            image = arr[:, :, :3]

        results = torch_model(image, conf=0.2, iou=0.8)[0]
        boxes = results.boxes.xyxy.cpu().numpy().astype(int)
        class_ids = results.boxes.cls.cpu().numpy().astype(int)

        page_entries = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            crop = image[y1:y2, x1:x2]
            text = pytesseract.image_to_string(crop).strip()
            class_name = torch_model.names[class_ids[i]]
            area = abs(y2 - y1) + 30
            if page_index == 0 and ans["title"] == "" and class_name == "Title":
                ans["title"] = text
                continue
            if class_name == "Section-header":
                page_entries.append({
                    "page": page_index + 1,
                    "class": class_name,
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                    "text": text,
                    "area": area
                })

        # collect all headers so far to compute percentiles
        all_results = page_entries  # if you want to compute across pages, accumulate above instead
        areas = sorted([h["area"] for h in all_results], reverse=True)
        p75 = np.percentile(areas, 75) if areas else 0
        p50 = np.percentile(areas, 50) if areas else 0

        # sort and assign levels
        for h in sorted(all_results, key=lambda x: (x["page"], x["box"][1], x["box"][0])):
            
            if h["area"] >= p75:
                h["level"] = "H1"
            elif h["area"] >= p50:
                h["level"] = "H2"
            else:
                h["level"] = "H3"
            if h["text"]:
                outline.append({
                    "level": h["level"],
                    "text": h["text"],
                    "page": h["page"]
                })

    ans["outline"] = outline
    return ans

# Main loop: find all .pdf in input_dir
for fname in os.listdir(input_dir):
    if not fname.lower().endswith(".pdf"):
        continue
    input_path  = os.path.join(input_dir, fname)
    output_name = os.path.splitext(fname)[0] + ".json"
    output_path = os.path.join(output_dir, output_name)

    print(f"Processing {input_path} → {output_path}...")
    result = process_pdf(input_path)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"  → done: {output_path}")

print("All files processed.")