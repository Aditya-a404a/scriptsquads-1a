# 🧾 Adobe Hackathon 2025: Round 1A – Document Outline Extractor

A high-performance, offline solution to extract structured outlines (Title, H1, H2, H3) from PDF documents. Developed for **Adobe Hackathon 2025: Round 1A – Understand Your Document**.

---

## 🚀 Challenge

Build a system that can programmatically understand PDF structure. Given a PDF, extract:

- **Title**
- **Headings (H1, H2, H3)**
- **Their corresponding page numbers**

The output should be a structured **JSON** file.

---

## ✨ Features

- 🎯 **Accurate Heading Detection**: Title, H1, H2, H3 detection using computer vision.
- ⚡ **High Performance**: Processes 50-page PDFs in **< 10 seconds** on standard CPU.
- 🛜 **Fully Offline**: Works without internet connectivity.
- 📦 **Dockerized**: Easy to run anywhere via a container.
- 🌐 **Multilingual Capable**: Handles documents in multiple languages, including Japanese and other non-Latin scripts.

---

## 🧠 Evolution of the Approach

### Phase 1: Rule-Based (PyMuPDF)
- Relied on font size, boldness, position.
- ❌ Fragile and inconsistent across sources.

### Phase 2: Classical ML (Random Forest)
- Trained on DocLayNet with engineered features.
- ❌ Improved generalization but limited by PyMuPDF block quality.

### Phase 3: Hybrid (YOLO + Graph Network)
- YOLO for block detection + Graph for hierarchy.
- ❌ Too complex, slow for real-time constraints.

### ✅ Final Phase: Pure Visual – YOLOv11
- Rendered PDF pages as images.
- Used quantized **YOLOv11** to directly detect Title, H1, H2, H3 from visual layout.
- Robust, fast, multilingual, and accurate.

---

## ❌ Why We Don't Use Multiprocessing

While multiprocessing is a common choice for performance, we avoided it for these reasons:

### ⚠️ Model Not Thread-Safe
- Our YOLOv11 model is not thread-safe.
- Parallel inference within a shared model can lead to **race conditions** and **corrupted outputs**.

### 🧠 High Overhead for Multiple Processes
- Multiprocessing loads a separate model (~165MB) per process.
- On systems like M1 or amd64 cloud VMs, this causes:
  - High **memory usage**
  - Noticeable **startup latency**
- These negate the performance benefits for short tasks.

### 🏎️ Single-Process Is Enough
- Sequential pipeline processes 50 pages in **~7–9 seconds**.
- Simpler, more robust, and meets all performance constraints.

---

## 🛠️ Tech Stack

| Component        | Tool/Library              |
|------------------|---------------------------|
| Language         | Python 3.11               |
| Deep Learning    | Torch, ONNX Runtime       |
| Computer Vision  | OpenCV                    |
| PDF Processing   | PyMuPDF (fitz)            |
| Deployment       | Docker (linux/amd64)      |

---

## ⚙️ Setup & Usage

### 🧱 Prerequisites

- Docker installed and running.

### 🔨 Build Docker Image

```bash
docker build --platform linux/amd64 -t adobe-hackathon-2025:round1a .
