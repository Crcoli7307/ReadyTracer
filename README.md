# ReadyTracer
### Ultra Image → SVG Line Art Converter 🖼️➡️🖋️

A single-file PyQt6 app that converts images into **SVG line art** with a **sleek purple/black flat UI**, rounded corners, and tons of tweakable knobs. Includes live preview, presets, and export scaling.

## ✨ Features
- Live preview with purple edge overlay
- Canny / adaptive / simple threshold modes
- CLAHE, bilateral & Gaussian blur, brightness/contrast, gamma, invert
- Ramer–Douglas–Peucker path simplification
- Min area filtering, close/fill paths, stroke width/color, background color
- One-click presets: **Crisp Lines**, **Sketchy**, **Bold Poster**
- Drag & drop images, SVG export

## 🧰 Install

#### Download the release for Windows or:

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
````

## ▶️ Run

```bash
python app.py
```

## 📦 Dependencies

* PyQt6, OpenCV, NumPy, Pillow, svgwrite

## 💾 Export

Use **Save SVG** to export at your chosen scale and style. Background can be set to transparent (use `#00000000`).

## 🧪 Presets

* **Crisp Lines:** balanced Canny, minimal blur
* **Sketchy:** softer blur, more simplification
* **Bold Poster:** adaptive threshold, thicker strokes, optional fill

## 🧩 Troubleshooting

* **macOS:** If you see “Qt platform plugin could not be initialized”, run inside a venv and reinstall PyQt6.
* **HiDPI scaling:** export uses `Export Scale` (SVG size), preview uses `Preview Scale` (speed).
* **Large images:** if preview is slow, reduce `Preview Scale`.

## 📄 License

MIT © 2025 Crayton Litton
