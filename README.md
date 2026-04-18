# 2-Stage Neural Compression Pipeline

A hackathon project implementing an end-to-end pipeline that extracts text from noisy scanned document images using a custom CNN and compresses the output using a custom Adaptive Huffman encoder — with full lossless recovery.

```
Noisy Image → [Stage 1: OCR CNN] → Text → [Stage 2: Adaptive Huffman] → Bitstream → Decompressed Text
```

---

## Architecture Overview

### Stage 1 — OCR Microservice (ResVGG CNN)

A 4-block residual VGG-style CNN trained on EMNIST (byclass split, 62 classes: 0–9, A–Z, a–z).

```
Input: 32×32 grayscale character patch
│
├─ Block 1: ResVGGBlock(  1 →  64)   32×32 → 16×16
│   ├─ Conv2d(1,  64, 3×3) → BN → ReLU
│   ├─ Conv2d(64, 64, 3×3) → BN
│   ├─ Residual skip: Conv2d(1, 64, 1×1) → BN
│   ├─ (+skip) → ReLU → MaxPool2d(2) → Dropout(0.10)
│
├─ Block 2: ResVGGBlock( 64 → 128)   16×16 →  8×8
│   ├─ Conv2d(64,  128, 3×3) → BN → ReLU
│   ├─ Conv2d(128, 128, 3×3) → BN
│   ├─ Residual skip: Conv2d(64, 128, 1×1) → BN
│   ├─ (+skip) → ReLU → MaxPool2d(2) → Dropout(0.10)
│
├─ Block 3: ResVGGBlock(128 → 256)    8×8  →  4×4
│   ├─ Conv2d(128, 256, 3×3) → BN → ReLU
│   ├─ Conv2d(256, 256, 3×3) → BN
│   ├─ Residual skip: Conv2d(128, 256, 1×1) → BN
│   ├─ (+skip) → ReLU → MaxPool2d(2) → Dropout(0.15)
│
├─ Block 4: ResVGGBlock(256 → 512)    4×4  →  2×2
│   ├─ Conv2d(256, 512, 3×3) → BN → ReLU
│   ├─ Conv2d(512, 512, 3×3) → BN
│   ├─ Residual skip: Conv2d(256, 512, 1×1) → BN
│   ├─ (+skip) → ReLU → MaxPool2d(2) → Dropout(0.15)
│
├─ GlobalAvgPool2d → 512-dim vector
│
└─ Classifier Head
    ├─ Linear(512 → 512) → ReLU → Dropout(0.5)
    └─ Linear(512 → 62)  → logits
```

**Total parameters:** 5,154,622

**Design decisions:**
- **Residual skip connections** — stabilise gradient flow through 8 conv layers; critical for training with cuDNN disabled on BigRed200 (A100 + torch 2.2.0)
- **4 blocks instead of 3** — doubles effective receptive field depth; needed for 62 visually-similar classes (e.g. `0/O`, `1/l/I`, `5/S`)
- **Global Average Pooling** — eliminates spatial sensitivity to character centering, which varies in EMNIST
- **Dropout only in FC head (0.5)** — avoids under-training with the already challenging 62-class task

**Preprocessing pipeline:**
1. Convert to grayscale
2. DenoisingUNet (U-Net, base_ch=32) removes Gaussian / salt-and-pepper noise
3. Median filter (3×3) — removes residual isolated noise pixels before thresholding
4. Otsu thresholding — binarise to foreground/background
5. Horizontal projection profile → detect text line rows
6. Vertical projection profile → segment individual character bounding boxes
7. Resize each patch to 32×32 with aspect-ratio padding

---

### Stage 2 — Compression Microservice (Adaptive Huffman)

A fully custom Adaptive Huffman encoder — no zlib, gzip, or any external compression library.

**Algorithm (online / no pre-scan):**
```
For each incoming symbol:
  ┌─ Symbol seen before? ──YES──► Output current Huffman code
  │
  └─ New symbol? ──────────NO───► Output NYT escape code + raw 8-bit value

After each symbol:
  └─ Increment frequency table → rebuild Huffman tree (deterministic)

Encoder and decoder maintain identical frequency state → always synchronised
No frequency table needs to be transmitted with the compressed data
```

**Why adaptive (not static)?**
- No two-pass pre-scan required → true online / streaming compression
- Code lengths adapt as the symbol distribution is learned in real time
- Both encoder and decoder stay synchronised without side-channel data

**Compression results on OCR output (~141 chars):**
- Compression ratio: **1.87×**
- Encoding efficiency: **74%** of Shannon entropy limit
- Compress latency: **~8 ms**
- Decompress latency: **~9 ms**

---

## Training

**Dataset:** EMNIST byclass — 697,932 training samples, 116,323 validation samples, 62 classes

**Training configuration (BigRed200 A100):**
| Parameter | Value |
|-----------|-------|
| Epochs | 30 |
| Batch size | 512 |
| Learning rate | 2e-3 (OneCycleLR) |
| Optimizer | AdamW (weight_decay=1e-4) |
| Loss | CrossEntropyLoss (label_smoothing=0.05) |
| Scheduler | OneCycleLR (warmup 30%, cosine decay) |
| Train samples | 200,000 (stratified subsample) |
| Augmentation | RandomRotation(15°), RandomAffine, RandomPerspective, RandomErasing |

**Results:**
| Metric | Value |
|--------|-------|
| Best val accuracy | **87.86%** |
| Gaussian noise accuracy | 55.85% |
| Salt & pepper noise accuracy | 40.95% |
| Training time | ~90 min on A100 |

**BigRed200 note:** `torch.backends.cudnn.enabled = False` is required — torch 2.2.0+cu118 segfaults on the first cuDNN Conv2d call on A100. All ops still run on GPU via raw CUDA kernels.

---

## Project Structure

```
Hackathon/
├── ocr_service/
│   ├── api.py          # FastAPI endpoints: POST /ocr, POST /ocr/denoise, GET /health
│   ├── model.py        # CharClassifier (4-block ResVGG) + DenoisingUNet
│   ├── predict.py      # Inference: load_classifier(), predict_image()
│   └── preprocess.py   # Preprocessing pipeline: denoise → threshold → segment
│
├── compression_service/
│   ├── api.py          # FastAPI endpoints: POST /compress, POST /decompress, GET /health
│   └── huffman.py      # Adaptive Huffman: _encode_adaptive(), _decode_adaptive()
│
├── training/
│   ├── train_classifier.py   # Train CharClassifier on EMNIST
│   └── train_denoiser.py     # Train DenoisingUNet
│
├── models/
│   ├── char_classifier.pth   # Trained classifier weights (5.15M params, 87.86% val acc)
│   └── denoiser.pth          # Trained denoiser weights (U-Net, base_ch=32)
│
├── test_images/              # 15 test images (clean, gaussian, salt-and-pepper × 5 docs)
├── pipeline.py               # End-to-end orchestrator with benchmarking
├── demo.py                   # Streamlit UI demo
├── train_classifier.slurm    # SLURM job script for BigRed200
└── requirements.txt
```

---

## Setup

### Prerequisites
- Python 3.10+
- PyTorch 2.0+ (CPU or CUDA)

### Install dependencies
```bash
git clone <repo-url>
cd Hackathon
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Running the Pipeline

### Option 1 — Streamlit Demo (recommended)

Start both microservices in separate terminals:
```bash
# Terminal 1
uvicorn ocr_service.api:app --port 8000

# Terminal 2
uvicorn compression_service.api:app --port 8001
```

Launch the demo:
```bash
streamlit run demo.py
```
Open `http://localhost:8501` — upload any document image and run the full pipeline.

---

### Option 2 — Direct CLI

```bash
# Single image
python pipeline.py --image test_images/letter_gaussian.png --direct

# Benchmark all test images
python pipeline.py --benchmark --image-dir test_images/ --direct

# Save results to JSON
python pipeline.py --image test_images/letter_combined.png --direct --output results.json
```

---

### Option 3 — API calls

```bash
# OCR
curl -X POST http://localhost:8000/ocr \
  -F "file=@test_images/letter_clean.png"

# Compress OCR output
curl -X POST http://localhost:8001/compress \
  -F "file=@ocr_output.txt"

# Decompress
curl -X POST http://localhost:8001/decompress \
  -H "Content-Type: application/json" \
  -d '{"bitstring":"010101...","original_length":141}'
```

---

## API Reference

### OCR Service — `http://localhost:8000`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service liveness check |
| `/ocr` | POST | Upload image → returns extracted text + latency |
| `/ocr/denoise` | POST | Upload image → returns denoised PNG for inspection |

**POST /ocr response:**
```json
{
  "text": "Dear Dr. Johnson,",
  "characters": ["D","e","a","r",...],
  "num_chars": 18,
  "denoise_enabled": true,
  "latency_ms": 312.4
}
```

### Compression Service — `http://localhost:8001`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service liveness + algorithm name |
| `/compress` | POST | Upload file → adaptive Huffman compressed payload |
| `/decompress` | POST | JSON payload → decompressed text |

**POST /compress response:**
```json
{
  "original_bytes": 141,
  "compressed_bytes": 76,
  "compression_ratio": 1.87,
  "entropy_bits_per_symbol": 3.163,
  "avg_bits_per_symbol": 4.277,
  "encoding_efficiency": 0.74,
  "latency_ms": 7.8,
  "bitstring": "010010110...",
  "original_length": 141
}
```

---

## Reproducing Training

### Classifier (BigRed200 A100)
```bash
sbatch train_classifier.slurm
# or locally:
python training/train_classifier.py --epochs 30 --batch-size 512 --lr 2e-3
```

### Denoiser
```bash
python training/train_denoiser.py
```

---

## Noise Robustness

The pipeline handles two noise profiles on scanned documents:

| Noise type | Description | Handling |
|------------|-------------|----------|
| **Gaussian** | Additive Gaussian (σ = 0.10–0.15) | DenoisingUNet + MedianFilter |
| **Salt & Pepper** | Random pixel corruption (2–8% density) | MedianFilter before Otsu thresholding |

Both noise types are applied during training via `RandomNoise` augmentation and evaluated separately using `evaluate_per_noise()`.
