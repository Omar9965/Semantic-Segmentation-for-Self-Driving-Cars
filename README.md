# Semantic Segmentation for Self-Driving Cars

A FastAPI + PyTorch application for semantic segmentation using a U-Net model enhanced with CBAM attention. Trained on the Lyft-Udacity Challenge dataset for autonomous driving scenarios. It provides a simple web UI and a clean REST API for single or batch image inference.

---

## Features

- U-Net with CBAM attention for accurate 13-class semantic segmentation
- Detects: Background, Building, Fence, Other, Pedestrian, Pole, Road Line, Road, Sidewalk, Vegetation, Vehicle, Wall, Traffic Sign
- FastAPI backend with Swagger docs and CORS enabled
- Saves results (original, colored mask, overlay) to disk and returns public URLs
- Simple drag-and-drop web UI
- Batch processing endpoint
- Safety-relevant detection flags (vehicles, pedestrians, road)

---

## Project Structure

```
requirements.txt
car/                      # Python venv (local)
src/
  main.py                 # FastAPI app entry
  assets/
    files/               # Temp uploads
    output/              # Saved outputs (served at /output)
  controllers/
  models/
    Model/
      unet.py            # U-Net + CBAM (13 classes)
      inference.py       # Inference utilities
      best_model.pth     # Place weights here
  routes/
    data.py              # API routes
    schemas/DataSchema.py
  static/                # JS/CSS for UI
  templates/             # index.html
  Notebook/              # Training notebook
  utils/
```

---

## Model

- Architecture: U-Net with CBAM (Channel + Spatial attention)
- Classes: 13 semantic classes for self-driving car scenes
- Training Input: RGB, resized to 256×256, normalized with ImageNet mean/std
- Inference Input: RGB, resized to 256×256, same normalization
- Output: Multi-class segmentation mask (argmax over softmax)
- Weights: Place `best_model.pth` in `src/models/Model/`

### Class Labels
| ID | Class Name    | Color        |
|----|---------------|--------------|
| 0  | Background    | Black        |
| 1  | Building      | Gray         |
| 2  | Fence         | Light Gray   |
| 3  | Other         | Peach        |
| 4  | Pedestrian    | Red          |
| 5  | Pole          | Medium Gray  |
| 6  | Road Line     | Yellow-Green |
| 7  | Road          | Purple       |
| 8  | Sidewalk      | Pink         |
| 9  | Vegetation    | Olive        |
| 10 | Vehicle       | Blue         |
| 11 | Wall          | Blue-Gray    |
| 12 | Traffic Sign  | Yellow       |

Preprocessing (inference):
- Converts BGR/GRAY to RGB
- Resizes to 256×256
- Normalizes with ImageNet mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

---

## Run Locally (Windows)

```powershell
# 1) Activate virtual environment (provided)
.\car\Scripts\Activate.ps1

# 2) Install dependencies
pip install -r requirements.txt

# 3) Start the API
cd src
uvicorn main:app --reload
```

- UI: http://127.0.0.1:8000/
- Docs (Swagger): http://127.0.0.1:8000/docs
- Outputs served at: http://127.0.0.1:8000/output/

---

## API

### Endpoints

- POST `/api/v1/segment` — Perform semantic segmentation on a single image
- POST `/api/v1/segment-multiple` — Perform semantic segmentation on multiple images

Accepted types: JPG/JPEG, PNG, TIFF (`.tif`, `.tiff`). Max size defaults to 500MB in the UI; server-side limits may differ.

### Response Schema (single)

```json
{
  "filename": "street_001",
  "original_image_url": "/output/street_001_1702828800000_original.png",
  "mask_url": "/output/street_001_1702828800000_mask.png",
  "overlay_url": "/output/street_001_1702828800000_overlay.png",
  "width": 512,
  "height": 256,
  "detected_classes": ["Road", "Vehicle", "Building", "Vegetation", "Sidewalk"],
  "class_counts": {"Background": 1000, "Road": 50000, "Vehicle": 8000, ...},
  "has_vehicles": true,
  "has_pedestrians": false,
  "has_road": true,
  "num_classes_detected": 5
}
```

`overlay_url` may be null if overlays are disabled.

### cURL Examples

Single image:
```bash
curl -X POST \
  -F "file=@C:/path/to/your/street.jpg" \
  http://127.0.0.1:8000/api/v1/segment
```

Multiple images:
```bash
curl -X POST \
  -F "files=@C:/path/scene1.png" \
  -F "files=@C:/path/scene2.jpg" \
  http://127.0.0.1:8000/api/v1/segment-multiple
```

---

## Frontend

- Served at `/` with `index.html`
- Drag-and-drop or file picker uploads
- Displays Original, Colored Mask, and Overlay using API-returned URLs
- Shows safety-relevant detection badges (Vehicles, Pedestrians, Road)

---

## Storage & Paths

- Temp uploads: `src/assets/files/` (cleaned after processing)
- Saved results: `src/assets/output/` and exposed at `/output` via FastAPI static mount
- File naming: `{original_filename}_{timestamp}_{type}.png`

---

## Configuration

- Model weights: `src/models/Model/best_model.pth`
- Device: CUDA if available, else CPU
- Classes: 13 (multi-class semantic segmentation)

---

## Training

The training notebook is at `src/Notebook/Semantic Segmentation for Self Driving Cars.ipynb`:
- Dataset: Lyft-Udacity Challenge (Kaggle)
- Loss: `0.5 * CrossEntropyLoss + 0.5 * MulticlassDiceLoss`
- Augmentations: HorizontalFlip, Rotate(15°), RandomBrightnessContrast, GaussNoise
- Early stopping with patience=7
- Mixed precision training with gradient accumulation
- Metrics: Mean Dice Score, Mean IoU (mIoU)

---

## Troubleshooting

- Missing weights: Ensure `best_model.pth` exists in `src/models/Model/`
- 500 errors on upload: Check file type/size and server logs
- Images not visible: Confirm `/output` is mounted and URLs resolve
- CUDA issues: App will automatically fall back to CPU

---

## Notes

- This repo includes a project-specific venv folder (`car/`) for convenience; you may use your own environment instead.
- The included Jupyter notebook for training is at `src/Notebook/Semantic Segmentation for Self Driving Cars.ipynb`.
