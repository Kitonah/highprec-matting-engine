# Gristine / Precision Matting Engine

> High-throughput, sub-pixel foreground extraction and illumination-invariant alpha matting engine designed for automated fashion and luxury product discovery pipelines.

---

## Overview

The Gristine Precision Matting Engine is an industrial-grade computer vision pipeline engineered to ingest raw catalog assets, user uploads, and editorial photography, transforming them into clean, unmultiplied 32-bit RGBA cutouts for dynamic mood boards, shoppable collage canvases, and visual search.

Unlike off-the-shelf consumer background removal tools that produce edge haloing, silhouette swelling from cast shadows, and edge erosion on low-contrast objects, this system couples deep bilateral reference transformers with physical optics modeling to resolve zero-contrast boundaries and sub-pixel contours natively on Apple Silicon.

---

## Architectural Highlights

* **Bilateral Reference Segmentation (BiRefNet):** Decoupled Localization (LM) and Reconstruction (RM) modules running gradient-supervised dichotomous segmentation at 1024 x 1024 native patch resolutions.
* **Fourier Phase-Preserving Boundary Recovery:** Reconstructs high-frequency geometric contours using 2D FFT Phase Spectrum analysis, preventing silhouette collapse on white garments, cosmetics, and jewelry placed against high-key white backgrounds.
* **Physical Illumination Invariance:** Employs Finlayson 1D log-chromaticity orthogonal projections to decouple material reflectance boundaries from external cast shadows, cleaving ground shadow penumbras while preserving 3D attached contours.
* **Dynamic Entropy-Directed Trimaps:** Adapts local morphological uncertainty windows from 3px (rigid mechanical edges) to 15px (loose hair fibers, sheer textiles, and porous accessories).
* **Intrinsic Backdrop Despill:** Reverses classical compositing equations to decontaminate ambient light reflection and color spill along semi-transparent perimeters.

---

## Pipeline Architecture

```
[Raw Commercial Asset / Editorial Photo]
                 │
                 ├──► Phase 1: Spectral & Illumination Deconstruction
                 │    ├─ 2D Fast Fourier Transform (Phase Boundary Extraction)
                 │    ├─ Discrete Wavelet Transform (High-Pass Sub-bands)
                 │    └─ Finlayson 1D Log-Chromaticity Invariant Projection
                 │
                 ├──► Phase 2: High-Resolution DIS Inference (BiRefNet)
                 │    └─ Deep Bilateral Reference Skip Aggregation (MPS Accelerated)
                 │
                 └──► Phase 3: Boundary-Adaptive Matting & Despill
                      ├─ Local Entropy-Driven Dynamic Uncertainty Trimap
                      ├─ Bilateral Sub-Pixel Alpha Refinement
                      └─ Analytical Inward Background Unmixing
                               │
                               ▼
               [Production 32-bit RGBA Cutout]
```

---

## Hardware Acceleration

Engineered natively for **Apple Silicon (ARM64)** leveraging PyTorch Metal Performance Shaders backend (`torch.device("mps")`). Full-resolution forward passes complete in **~1.18 seconds**, providing high-speed local inference with unified memory architecture.

---

## Engineering Ownership & Module Allocation

* **Mohit Lal:**
  * System Architecture & Orchestration (`pipeline.py`)
  * BiRefNet DIS Segmentation Engine (`core/segmentation.py`)
  * Boundary Adaptive Matting & Dynamic Trimaps (`core/matting.py`)
  * Intrinsic Despill & Backdrop Decontamination (`core/despill.py`)
  * Asynchronous Microservice Layer & Workbench UI (`server.py`, `index.html`)

* **Ekantika Kumari:**
  * Physical Optics & Illumination-Invariant Dissociation (`core/illumination.py`)
  * Spectral Decomposition: 2D FFT Phase & Wavelet Sub-band Analysis (`core/frequency.py`)
  * Zero-Contrast Camouflage Perimeter Calibration

---

## Installation & Setup

### Prerequisites
* macOS 13.0+ running on Apple Silicon (M-series) or Linux with CUDA
* Python 3.11+

### 1. Clone Repository & Initialize Environment
```bash
git clone [https://github.com/](https://github.com/)Kitonah/highprec-matting-engine.git
cd highprec-matting-engine

python3.11 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio
pip install timm "transformers<4.44.0" accelerate einops kornia
pip install numpy scipy opencv-contrib-python-headless pillow PyWavelets
pip install fastapi "uvicorn[standard]" python-multipart pydantic
```

---

## Running the Engine

### Option A: Launch Interactive Studio Web UI
Start the local FastAPI microservice:
```bash
uvicorn server:app --host 0.0.0.0 --port 8080 --reload
```
Open your browser and navigate to `http://localhost:8080` to access the drag-and-drop studio workbench.

### Option B: Headless Batch Processing
Drop test assets into the root directory and execute the direct pipeline script:
```bash
python test_run.py
```

---

## API Specifications

### `POST /v2/extract`
Streams an unmultiplied 32-bit transparent PNG buffer directly from memory.

* **Headers:** `Content-Type: multipart/form-data`
* **Body:** `file`: `Binary Image Data` (`.jpg`, `.png`, `.webp`)
* **Response:** `image/png` binary stream

#### Example cURL Request
```bash
curl -X POST "http://localhost:8080/v2/extract" \
     -H "Accept: image/png" \
     -F "file=@product_sample.jpg" \
     --output cutout_master.png
```

---

## Technical Stack

* **Compute & Deep Learning:** PyTorch (MPS/CUDA), Hugging Face Transformers, BiRefNet, Kornia
* **Digital Signal Processing:** OpenCV Contrib, NumPy, SciPy, PyWavelets
* **Serving Layer:** FastAPI, Uvicorn, ASGI Streaming
* **Interface:** Vanilla HTML5 Canvas, Dynamic File API, CSS Grid

---

## Author & License

* **Authors:** Mohit Lal & Ekantika Kumari
* **Project:** Core vision infrastructure for **Gristine**.
* **Copyright:** © 2026 Mohit Lal & Ekantika Kumari. All rights reserved.
* **License:** Proprietary. Unauthorized copying, modification, distribution, or commercial deployment of this software is strictly prohibited.
