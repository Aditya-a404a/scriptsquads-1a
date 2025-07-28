# Adobe Hackathon 2025: Round 1A – Document Outline Extractor

A high-performance, offline solution to extract structured outlines (Title, H1, H2, H3) from PDF documents. Developed for **Adobe Hackathon 2025: Round 1A – Understand Your Document**.

---

## Challenge

Develop a system that programmatically interprets PDF structure and outputs a structured JSON file containing:

- Title  
- Headings (H1, H2, H3)  
- Corresponding page numbers  

---

## Features

- **Accurate Heading Detection**: Utilizes computer vision techniques to identify Title, H1, H2, and H3 elements.  
- **High Performance**: Capable of processing 50-page PDFs in under 10 seconds on standard CPU hardware.  
- **Fully Offline**: Operates without any external network dependencies.  
- **Containerized Deployment**: Provided as a Docker image for consistent runtime environments.  
- **Multilingual Support**: Compatible with documents in multiple languages, including Japanese and other non-Latin scripts.

---

## Evolution of the Approach

### Phase 1: Rule-Based Extraction (PyMuPDF)
- Relied on font size, boldness, and positional heuristics.  
- Limited robustness across diverse document formats.

### Phase 2: Classical Machine Learning (Random Forest)
- Engineered features from DocLayNet annotations.  
- Improved generalization but constrained by extraction quality from PDF blocks.

### Phase 3: Hybrid Model (YOLO + Graph Network)
- Combined visual detection with graph-based hierarchy modeling.  
- Achieved accuracy but was too complex and slow for real-time requirements.

### Final Phase: Pure Visual Detection (YOLOv11)
- Rendered each PDF page as an image.  
- Employed a quantized YOLOv11 model to directly detect Title, H1, H2, and H3.  
- Achieved robust, high-speed, and language-agnostic performance.

---

## Rationale Against Multiprocessing

1. **Model Thread Safety**  
   - The YOLOv11 model is not thread-safe; concurrent inference can lead to race conditions and corrupted outputs.

2. **Resource Overhead**  
   - Multiprocessing requires loading a separate model instance (~165 MB) per process, resulting in excessive memory consumption and startup latency.

3. **Sufficient Performance**  
   - A single-process pipeline processes 50 pages in approximately 7–9 seconds, meeting the performance targets with simpler maintenance.

---

## Technology Stack

| Component        | Tool / Library           |
|------------------|--------------------------|
| Language         | Python 3.11              |
| Deep Learning    | PyTorch, ONNX Runtime    |
| Computer Vision  | OpenCV                   |
| PDF Processing   | PyMuPDF (fitz)           |
| Containerization | Docker (linux/amd64)     |
| Model Framework  | Ultralytics YOLOv11      |

---

## Setup & Usage

### Prerequisites
- Docker installed and configured on the host machine.

### Building the Docker Image
```bash
docker build --platform linux/amd64 -t adobe-hackathon-2025-round1a .
docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  adobe-hackathon-2025-round1a
