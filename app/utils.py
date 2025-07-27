import os
import fitz  # PyMuPDF
import cv2
import numpy as np
from ultralytics import YOLO
import json
import multiprocessing

# --- Configuration ---
MODEL_PATH = "./models/secretRecipe.pt"
INPUT_DIR  = "./input"
OUTPUT_DIR = "./output"
DPI = 120  # Define DPI as a constant

def process_pdf(pdf_path):
    """
    This is the core worker function. It processes a single PDF from its path,
    extracts the outline, and saves it to a JSON file.
    """
    fname = os.path.basename(pdf_path)
    print(f"Starting: {fname}")
    
    # Initialize the model inside the worker for multiprocessing compatibility
    torch_model = YOLO(MODEL_PATH)
    doc = fitz.open(pdf_path)
    ans = {"title": "", "outline": []}
    
    page_images = []
    for page in doc:
        pix = page.get_pixmap(dpi=DPI)
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
        return f"Skipped (no pages): {fname}"

    all_results = torch_model(page_images, conf=0.35, iou=0.8)

    all_headers = []
    for page_index, results in enumerate(all_results):
        page = doc[page_index]
        
        # Create a matrix to convert from pixel coords to PDF point coords
        mat = fitz.Matrix(72/DPI, 72/DPI)

        # Get all words on the page directly from the PDF data
        page_words = page.get_text("words")

        yolo_boxes = results.boxes.xyxy.cpu().numpy().astype(int)
        yolo_classes = results.boxes.cls.cpu().numpy().astype(int)
        
        for i, yolo_box in enumerate(yolo_boxes):
            class_name = torch_model.names[yolo_classes[i]]
            text = ""

            yolo_pixel_rect = fitz.Rect(*yolo_box)
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
    
    # Save the output file
    output_name = os.path.splitext(fname)[0] + ".json"
    output_path = os.path.join(OUTPUT_DIR, output_name)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ans, f, indent=2, ensure_ascii=False)
        
    return f"Finished: {fname}"

def main():
    """
    Finds all PDFs in the input directory and distributes them to a pool of worker processes.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get a list of all PDF file paths to process
    pdf_files = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if f.lower().endswith(".pdf")]
    
    if not pdf_files:
        print("No PDF files found in the input directory.")
        return

    # Use all available CPU cores for processing
    num_processes = multiprocessing.cpu_count()
    print(f"Starting processing for {len(pdf_files)} files using {num_processes} cores...")

    # Create a pool of worker processes
    with multiprocessing.Pool(processes=num_processes) as pool:
        # map() distributes the list of pdf_files to the process_pdf function
        # and collects the results.
        results = pool.map(process_pdf, pdf_files)
    
    # Print the status message from each worker
    for res in results:
        print(res)

    print("\nAll files processed.")

# This guard is essential for multiprocessing to work correctly
if __name__ == "__main__":
    main()