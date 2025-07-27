import os
import fitz  # PyMuPDF
import cv2
import numpy as np
from ultralytics import YOLO
import json

# Configuration
model_path = "./models/secretRecipe.pt"
input_dir  = "./input"
output_dir = "./output"

# Make sure output dir exists
os.makedirs(output_dir, exist_ok=True)

# Initialize YOLO model once
torch_model = YOLO(model_path)

def process_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    ans = {"title": "", "outline": []}
    
    page_images = []
    for page in doc:
        pix = page.get_pixmap(dpi=120)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            image = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        elif pix.n == 1:
            image = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        else:
            image = arr[:, :, :3]
        page_images.append(image)

    if not page_images:
        doc.close()
        return ans

    all_results = torch_model(page_images, conf=0.2, iou=0.8)

    all_headers = []
    for page_index, results in enumerate(all_results):
        page = doc[page_index]
        
        # Create a matrix to convert from pixel coords to PDF point coords
        mat = fitz.Matrix(72/120, 72/120)

        # Get all words on the page directly from the PDF data
        page_words = page.get_text("words")

        yolo_boxes = results.boxes.xyxy.cpu().numpy().astype(int)
        yolo_classes = results.boxes.cls.cpu().numpy().astype(int)
        
        for i, yolo_box in enumerate(yolo_boxes):
            class_name = torch_model.names[yolo_classes[i]]
            text = ""

            # Create a rectangle from YOLO's pixel coordinates
            yolo_pixel_rect = fitz.Rect(*yolo_box)
            
            # Transform the pixel rect to the PDF's point coordinate system
            yolo_pdf_rect = yolo_pixel_rect * mat

            # Use the transformed rect to find words in the same coordinate system
            words_in_box = [w[4] for w in page_words if fitz.Rect(w[:4]).intersects(yolo_pdf_rect)]
            if words_in_box:
                text = " ".join(words_in_box)

            if page_index == 0 and ans["title"] == "" and class_name == "Title":
                ans["title"] = text
                continue
            
            if class_name == "Section-header" and text:
                all_headers.append({
                    "page": page_index + 1,
                    "text": text,
                    "box": yolo_box.tolist(),
                    "area": abs(yolo_box[3] - yolo_box[1])
                })

    # Assign heading levels based on size
    if all_headers:
        areas = sorted([h["area"] for h in all_headers], reverse=True)
        p75 = np.percentile(areas, 75) if areas else 0
        p50 = np.percentile(areas, 50) if areas else 0

        final_outline = []
        for h in sorted(all_headers, key=lambda x: (x["page"], x["box"][1])):
            level = "H3"
            if h["area"] >= p75:
                level = "H1"
            elif h["area"] >= p50:
                level = "H2"
            final_outline.append({"level": level, "text": h["text"], "page": h["page"]})
        ans["outline"] = final_outline

    doc.close()
    return ans

# --- Main loop ---
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

print("\nAll files processed.")