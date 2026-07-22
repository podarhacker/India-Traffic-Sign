# Indian Traffic Sign Recognition — Web App

Runs 100% in the browser using `onnxruntime-web` (WASM). No backend, no server RAM needed.
Supports both **image upload** and **live camera scanning** (with an optional auto-scan "live mode").

---

## 1. Add your model files

This zip ships with **placeholder** files in `model/`. Replace them with your real exported files:

```
model/
├── best_model.onnx   <- replace with your real exported ONNX model
└── labels.json       <- replace with your real class labels
```

### How to generate `best_model.onnx`
Already covered in your notebook (section 9 — `torch.onnx.export`).

### How to generate `labels.json`
Run this in your notebook (after loading your checkpoint):

```python
import json

checkpoint = torch.load(
    CONFIG.checkpoint_dir / CONFIG.checkpoint_name,
    map_location=DEVICE,
    weights_only=False,
)
idx_to_class = checkpoint["idx_to_class"]

labels_list = [idx_to_class[i] for i in range(len(idx_to_class))]
with open(CONFIG.checkpoint_dir / "labels.json", "w") as f:
    json.dump(labels_list, f)

print(labels_list)
```

Download both files and drop them into this project's `model/` folder, overwriting the placeholders.

---

## 2. Check your image size / normalization match

Open `index.html` and check these constants near the top of the `<script>` block match your training config:

```js
const IMAGE_SIZE = 96;                          // CONFIG.image_size in your notebook
const IMAGENET_MEAN = [0.485, 0.456, 0.406];    // only change if you used different normalization
const IMAGENET_STD  = [0.229, 0.224, 0.225];
```

---

## 3. Test locally (optional but recommended)

Browsers block `fetch()` on local files opened directly, so serve it with a simple local server:

```bash
cd site
python3 -m http.server 8080
```

Then open `http://localhost:8080` in your browser.

Camera access requires either `localhost` or HTTPS (see deployment notes below) — it will NOT work over plain `http://` on a real IP address.

---

## 4. Deploy

### Option A: Vercel
```bash
npm install -g vercel
cd site
vercel deploy --prod
```
Or: go to vercel.com → New Project → drag and drop this folder.

### Option B: Cloudflare Pages
```bash
npm install -g wrangler
cd site
wrangler pages deploy .
```
Or: Cloudflare dashboard → Pages → Create a project → Direct Upload → drag and drop this folder.

Both give you free HTTPS automatically, which is required for camera access on mobile.

---

## 5. Using the app

- **Upload tab**: click or drag-and-drop any traffic sign photo.
- **Camera tab**:
  - Tap **📸 Capture & Scan** to take a single photo and classify it.
  - Tap **🔄 Flip** to switch between front/rear camera (useful on phones).
  - Toggle **Live mode** to auto-capture and classify every 1.5 seconds without tapping — good for real-time scanning while walking/driving (as a passenger, not while driving!).

---

## Notes / Troubleshooting

- **"labels.json not found" or model fails to load**: make sure `model/best_model.onnx` and `model/labels.json` actually replaced the placeholders (not just added alongside).
- **Camera doesn't open on phone**: must be served over HTTPS (Vercel/Cloudflare give you this automatically) — will not work over plain HTTP except on `localhost`.
- **Wrong/garbage predictions**: double check `IMAGE_SIZE` matches `CONFIG.image_size`, and mean/std match what you trained with.
- **Slow inference on old phones**: `onnxruntime-web` runs on CPU via WASM. Your model (~1.6MB CNN) should still be fast (<200ms) on most modern phones/laptops.
