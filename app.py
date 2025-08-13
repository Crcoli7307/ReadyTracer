#!/usr/bin/env python3

"""
ReadyTracer - Image to SVG Line Art Converter
Single-file PyQt6 app with a modern, purple/black flat UI, rounded corners, and emoji UX.
Author: Crayton Litton (Crayton Technologies)
License: MIT
"""

import sys
import os
import math
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
import cv2
import svgwrite

from PyQt6 import QtCore, QtGui, QtWidgets

# -----------------------------
# Utilities & Geometry Helpers
# -----------------------------

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def np_img_to_qimage(img_bgr_or_gray: np.ndarray) -> QtGui.QImage:
    """
    Convert OpenCV image (BGR or GRAY) to QImage (RGB888/Indexed8).
    """
    if img_bgr_or_gray is None:
        return QtGui.QImage()
    if len(img_bgr_or_gray.shape) == 2:
        h, w = img_bgr_or_gray.shape
        qimg = QtGui.QImage(img_bgr_or_gray.data, w, h, w, QtGui.QImage.Format.Format_Grayscale8)
        return qimg.copy()
    h, w, ch = img_bgr_or_gray.shape
    rgb = cv2.cvtColor(img_bgr_or_gray, cv2.COLOR_BGR2RGB)
    qimg = QtGui.QImage(rgb.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)
    return qimg.copy()

def ramer_douglas_peucker(points: List[Tuple[float, float]], epsilon: float) -> List[Tuple[float,float]]:
    """
    Simplify a polyline using the Ramer-Douglas-Peucker algorithm.
    points: [(x,y), ...]
    epsilon: distance threshold.
    """
    if len(points) < 3:
        return points

    # Find the point with the maximum distance
    start, end = points[0], points[-1]

    def perpendicular_distance(p, a, b):
        # distance from p to line ab
        ax, ay = a
        bx, by = b
        px, py = p
        dx, dy = bx - ax, by - ay
        if dx == 0 and dy == 0:
            return math.dist(p, a)
        t = ((px - ax) * dx + (py - ay) * dy) / float(dx*dx + dy*dy)
        t = clamp(t, 0.0, 1.0)
        closest = (ax + t*dx, ay + t*dy)
        return math.dist(p, closest)

    max_dist = -1.0
    index = -1
    for i in range(1, len(points)-1):
        d = perpendicular_distance(points[i], start, end)
        if d > max_dist:
            index = i
            max_dist = d

    if max_dist > epsilon:
        # Recursive call
        left = ramer_douglas_peucker(points[:index+1], epsilon)
        right = ramer_douglas_peucker(points[index:], epsilon)
        return left[:-1] + right
    else:
        return [start, end]

def contour_to_path(contour: np.ndarray, closed: bool, epsilon: float, scale: float) -> List[Tuple[float,float]]:
    """
    Convert a contour (Nx1x2) to simplified path points (list of (x,y)) scaled.
    """
    cnt = contour.reshape(-1, 2)
    pts = [(float(x)*scale, float(y)*scale) for x, y in cnt.tolist()]
    if epsilon > 0:
        pts = ramer_douglas_peucker(pts, epsilon)
    if closed and (len(pts) == 0 or pts[0] != pts[-1]):
        pts.append(pts[0])
    return pts

def points_to_svg_path_d(points: List[Tuple[float,float]], closed: bool) -> str:
    """
    Create a SVG path 'd' string from list of points.
    """
    if not points:
        return ""
    d = [f"M {points[0][0]:.2f},{points[0][1]:.2f}"]
    for x, y in points[1:]:
        d.append(f"L {x:.2f},{y:.2f}")
    if closed:
        d.append("Z")
    return " ".join(d)

def auto_contrast(img_gray: np.ndarray, clip_limit=2.0, tile_grid=8) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    return clahe.apply(img_gray)

def gamma_correction(img_gray: np.ndarray, gamma: float) -> np.ndarray:
    gamma = max(0.01, gamma)
    inv = 1.0 / gamma
    lut = np.array([((i / 255.0) ** inv) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img_gray, lut)

def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    m, M = img.min(), img.max()
    if M - m < 1e-6:
        return np.zeros_like(img, dtype=np.uint8)
    out = (img - m) * 255.0 / (M - m)
    return out.astype(np.uint8)

# -----------------------------
# Processing Pipeline
# -----------------------------

@dataclass
class PipelineSettings:
    # Preprocess
    resize_scale: float = 1.0         # preview scale (for speed)
    denoise_bilateral_d: int = 5
    denoise_bilateral_sigmaColor: int = 50
    denoise_bilateral_sigmaSpace: int = 50
    gaussian_ksize: int = 1           # odd [1..31]
    clahe_clip: float = 2.0
    clahe_tile: int = 8
    brightness: int = 0               # [-100..100]
    contrast: int = 0                 # [-100..100]
    gamma: float = 1.0
    invert: bool = False

    # Edge/Threshold
    use_canny: bool = True
    canny_low: int = 80
    canny_high: int = 160
    canny_aperture: int = 3           # 3,5,7

    use_adaptive: bool = False
    adaptive_block: int = 11          # odd
    adaptive_C: int = 2

    use_simple_thresh: bool = False
    simple_thresh: int = 128

    # Contours → SVG
    min_contour_area: float = 20.0
    simplify_epsilon_px: float = 1.5
    close_paths: bool = False
    stroke_width: float = 1.5
    stroke_color: str = "#C084FC"     # light purple
    background: str = "#0B0B10"       # near-black
    fill_closed: bool = False
    fill_color: str = "#00000000"     # transparent

    # Export scaling
    export_scale: float = 1.0

def apply_brightness_contrast(img_gray, brightness=0, contrast=0):
    # brightness: -100..100, contrast: -100..100
    beta = brightness
    alpha = (contrast + 100) / 100.0
    # convert to float for precise adjustment then clip
    out = cv2.convertScaleAbs(img_gray, alpha=alpha, beta=beta)
    return out

def process_to_edges(src_bgr: np.ndarray, s: PipelineSettings) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns: (proc_preview_bgr, edges_uint8)
    """
    if src_bgr is None:
        return None, None

    # Resize for preview
    if s.resize_scale != 1.0:
        preview = cv2.resize(src_bgr, None, fx=s.resize_scale, fy=s.resize_scale, interpolation=cv2.INTER_AREA)
    else:
        preview = src_bgr.copy()

    gray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)

    # Denoise & Blur
    if s.denoise_bilateral_d > 0:
        gray = cv2.bilateralFilter(
            gray,
            d=s.denoise_bilateral_d,
            sigmaColor=s.denoise_bilateral_sigmaColor,
            sigmaSpace=s.denoise_bilateral_sigmaSpace
        )

    k = max(1, s.gaussian_ksize | 1) # ensure odd >=1
    if k > 1:
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    # Local contrast + tone
    gray = auto_contrast(gray, clip_limit=max(0.5, s.clahe_clip), tile_grid=max(2, s.clahe_tile))
    gray = apply_brightness_contrast(gray, brightness=s.brightness, contrast=s.contrast)
    gray = gamma_correction(gray, s.gamma)
    if s.invert:
        gray = 255 - gray

    # Edge/Thresh
    edges = np.zeros_like(gray)
    if s.use_canny:
        ap = 3 if s.canny_aperture not in (3,5,7) else s.canny_aperture
        edges = cv2.Canny(gray, threshold1=s.canny_low, threshold2=s.canny_high, apertureSize=ap, L2gradient=True)
    elif s.use_adaptive:
        bs = s.adaptive_block | 1
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, max(3, bs), s.adaptive_C)
        edges = 255 - edges  # edges white
    elif s.use_simple_thresh:
        _, edges = cv2.threshold(gray, s.simple_thresh, 255, cv2.THRESH_BINARY_INV)

    # Make a preview overlay (purple lines)
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    purple = (204, 132, 252)  # BGR-ish for preview strokes
    mask = edges > 0
    overlay[mask] = purple

    return overlay, edges

def trace_svg(edges: np.ndarray, s: PipelineSettings, base_size: Tuple[int,int]) -> Tuple[svgwrite.Drawing, int]:
    """
    Find contours and export to SVG paths. Returns (svg, count_paths)
    """
    if edges is None:
        return None, 0

    # find contours on a copy
    ed = edges.copy()
    contours, hierarchy = cv2.findContours(ed, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    H, W = base_size
    scale = s.export_scale
    dwg = svgwrite.Drawing(size=(f"{int(W*scale)}px", f"{int(H*scale)}px"))
    dwg.viewbox(0, 0, int(W*scale), int(H*scale))

    # background
    if s.background and s.background != "#00000000":
        dwg.add(dwg.rect(insert=(0,0), size=("100%","100%"), fill=s.background))

    count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < s.min_contour_area:
            continue
        closed = bool(s.close_paths)
        epsilon = max(0.0, s.simplify_epsilon_px)
        pts = contour_to_path(cnt, closed, epsilon, s.export_scale / s.resize_scale)

        if len(pts) < 2:
            continue

        d = points_to_svg_path_d(pts, closed)
        path = dwg.path(d=d,
                        stroke=s.stroke_color,
                        fill=(s.fill_color if s.fill_closed and closed else "none"),
                        stroke_width=s.stroke_width,
                        stroke_linecap="round",
                        stroke_linejoin="round")
        dwg.add(path)
        count += 1

    # a11y title/desc
    dwg.add(dwg.desc("Generated by ReadyTracer"))
    return dwg, count

# -----------------------------
# Stylish Qt Widgets
# -----------------------------

class Banner(QtWidgets.QFrame):
    def __init__(self, text="🎨 Image → SVG Line Art", parent=None):
        super().__init__(parent)
        self.setObjectName("Banner")
        self.setMinimumHeight(64)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(16, 10, 16, 10)
        self.lbl = QtWidgets.QLabel(text)
        self.lbl.setObjectName("BannerLabel")
        self.lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft)
        lay.addWidget(self.lbl)
        self.btnOpen = QtWidgets.QPushButton("🖼️  Open Image")
        self.btnOpen.setObjectName("PrimaryButton")
        lay.addStretch(1)
        lay.addWidget(self.btnOpen)

class ImageView(QtWidgets.QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setObjectName("ImageView")
        self.setMinimumSize(360, 240)
        self.setText("Drop an image here or click “Open” 👇")
        self.setAcceptDrops(True)
        self.pix = None

    def set_pixmap_from_cv(self, img_bgr_or_gray):
        qimg = np_img_to_qimage(img_bgr_or_gray)
        self.pix = QtGui.QPixmap.fromImage(qimg)
        self.setPixmap(self.pix.scaled(self.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.pix is not None:
            self.setPixmap(self.pix.scaled(self.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation))

    # Drag & drop
    def dragEnterEvent(self, e: QtGui.QDragEnterEvent) -> None:
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e: QtGui.QDropEvent) -> None:
        for url in e.mimeData().urls():
            path = url.toLocalFile()
            if os.path.isfile(path):
                self.parent().parent().open_image(path)
                break

class Knob(QtWidgets.QWidget):
    """
    A compact label + slider + spinbox control with emojis.
    """
    valueChanged = QtCore.pyqtSignal(object)

    def __init__(self, title:str, vmin:int, vmax:int, step:int=1, initial:int=None, emoji:str="🎚️", parent=None):
        super().__init__(parent)
        self.setObjectName("Knob")
        lay = QtWidgets.QGridLayout(self)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setHorizontalSpacing(8)
        self.label = QtWidgets.QLabel(f"{emoji} {title}")
        self.label.setObjectName("KnobLabel")
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setMinimum(vmin)
        self.slider.setMaximum(vmax)
        self.slider.setSingleStep(step)
        self.slider.setPageStep(step)
        self.spin = QtWidgets.QSpinBox()
        self.spin.setMinimum(vmin)
        self.spin.setMaximum(vmax)
        self.spin.setSingleStep(step)
        lay.addWidget(self.label, 0, 0, 1, 2)
        lay.addWidget(self.slider, 1, 0)
        lay.addWidget(self.spin, 1, 1)
        if initial is not None:
            self.slider.setValue(initial)
            self.spin.setValue(initial)
        self.slider.valueChanged.connect(self.spin.setValue)
        self.spin.valueChanged.connect(self.slider.setValue)
        self.spin.valueChanged.connect(lambda v: self.valueChanged.emit(v))

    def value(self) -> int:
        return self.spin.value()

    def setValue(self, v:int):
        self.spin.setValue(v)

class FKnob(QtWidgets.QWidget):
    """
    Float knob (double spin).
    """
    valueChanged = QtCore.pyqtSignal(object)

    def __init__(self, title:str, vmin:float, vmax:float, step:float=0.1, initial:float=None, emoji:str="🧪", parent=None, decimals:int=2):
        super().__init__(parent)
        self.setObjectName("Knob")
        lay = QtWidgets.QGridLayout(self)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setHorizontalSpacing(8)
        self.label = QtWidgets.QLabel(f"{emoji} {title}")
        self.label.setObjectName("KnobLabel")
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(1000)
        self.spin = QtWidgets.QDoubleSpinBox()
        self.spin.setDecimals(decimals)
        self.spin.setMinimum(vmin)
        self.spin.setMaximum(vmax)
        self.spin.setSingleStep(step)
        lay.addWidget(self.label, 0, 0, 1, 2)
        lay.addWidget(self.slider, 1, 0)
        lay.addWidget(self.spin, 1, 1)

        if initial is None:
            initial = vmin
        self.spin.setValue(initial)
        # map slider [0..1000] to [vmin..vmax]
        def s2v(sv): return vmin + (vmax - vmin) * (sv/1000.0)
        def v2s(v): return int( (v - vmin)/(vmax - vmin) * 1000.0 )

        self.slider.setValue(v2s(initial))
        self.slider.valueChanged.connect(lambda sv: self.spin.setValue(s2v(sv)))
        self.spin.valueChanged.connect(lambda v: self.slider.setValue(v2s(v)))
        self.spin.valueChanged.connect(lambda v: self.valueChanged.emit(v))

    def value(self) -> float:
        return self.spin.value()

    def setValue(self, v:float):
        self.spin.setValue(v)

class Toggle(QtWidgets.QCheckBox):
    def __init__(self, text:str, emoji="⚡", parent=None):
        super().__init__(f"{emoji} {text}", parent)
        self.setObjectName("Toggle")

class ColorChip(QtWidgets.QPushButton):
    colorChanged = QtCore.pyqtSignal(str)
    def __init__(self, title:str, initial:str="#FFFFFF", emoji="🎨", parent=None):
        super().__init__(f"{emoji} {title}: {initial}", parent)
        self.color = initial
        self.setObjectName("ColorChip")
        self.clicked.connect(self.pick)

    def pick(self):
        c = QtWidgets.QColorDialog.getColor(QtGui.QColor(self.color), self, "Pick Color")
        if c.isValid():
            self.color = c.name(QtGui.QColor.NameFormat.HexArgb if c.alpha() < 255 else QtGui.QColor.NameFormat.HexRgb)
            self.setText(f"🎨 Color: {self.color}")
            self.colorChanged.emit(self.color)

# -----------------------------
# Main Window
# -----------------------------

class App(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ReadyTracer - Image to SVG Line Art")
        self.setWindowIcon(QtGui.QIcon())  # add your icon path if desired
        self.setObjectName("Root")

        self.src_bgr: Optional[np.ndarray] = None
        self.preview_bgr: Optional[np.ndarray] = None
        self.edges: Optional[np.ndarray] = None

        self.settings = PipelineSettings()

        self.banner = Banner("🧩 ReadyTracer")
        self.banner.btnOpen.clicked.connect(self.on_open_clicked)

        # LEFT: preview
        self.imageView = ImageView()
        self.imageView.setToolTip("Drop image here. Preview shows purple edges over grayscale 🤖")

        # RIGHT: controls
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        self.controlsPane = QtWidgets.QWidget()
        scroll.setWidget(self.controlsPane)
        self.controlsLayout = QtWidgets.QVBoxLayout(self.controlsPane)
        self.controlsLayout.setContentsMargins(12,12,12,12)
        self.controlsLayout.setSpacing(10)

        # Build knobs
        self._build_controls()

        # Footer buttons
        self.btnExport = QtWidgets.QPushButton("💾 Save SVG")
        self.btnExport.setObjectName("PrimaryButton")
        self.btnExport.clicked.connect(self.on_export)
        self.btnPreset1 = QtWidgets.QPushButton("🧪 Preset: Crisp Lines")
        self.btnPreset2 = QtWidgets.QPushButton("🧪 Preset: Sketchy")
        self.btnPreset3 = QtWidgets.QPushButton("🧪 Preset: Bold Poster")
        for b in (self.btnPreset1, self.btnPreset2, self.btnPreset3):
            b.clicked.connect(self.on_preset)

        btnRow = QtWidgets.QHBoxLayout()
        btnRow.addWidget(self.btnPreset1)
        btnRow.addWidget(self.btnPreset2)
        btnRow.addWidget(self.btnPreset3)
        self.controlsLayout.addLayout(btnRow)
        self.controlsLayout.addWidget(self.btnExport)
        self.controlsLayout.addStretch(1)

        # Layout
        main = QtWidgets.QVBoxLayout(self)
        main.setContentsMargins(12,12,12,12)
        main.setSpacing(10)
        main.addWidget(self.banner)

        center = QtWidgets.QHBoxLayout()
        center.setSpacing(10)
        center.addWidget(self.imageView, 2)
        center.addWidget(scroll, 1)
        main.addLayout(center)

        # Style
        self.setStyleSheet(self._qss())

        # Timer for live updates (debounce)
        self.liveTimer = QtCore.QTimer(self)
        self.liveTimer.setInterval(120)
        self.liveTimer.setSingleShot(True)
        self.liveTimer.timeout.connect(self.reprocess)

    # ------------- UI Construction -------------
    def _build_controls(self):
        add = self.controlsLayout.addWidget

        # Preprocess section
        add(self._sectionLabel("✨ Preprocess"))
        self.knScale = FKnob("Preview Scale", 0.25, 1.0, step=0.05, initial=1.0, emoji="🔎", decimals=2)
        self.knBilaD = Knob("Bilateral D", 0, 15, step=1, initial=5, emoji="🫧")
        self.knBilaSigC = Knob("Bilateral σColor", 0, 150, step=5, initial=50, emoji="🎯")
        self.knBilaSigS = Knob("Bilateral σSpace", 0, 150, step=5, initial=50, emoji="🧭")
        self.knGaus = Knob("Gaussian ksize", 1, 31, step=2, initial=1, emoji="🌫️")
        self.knCLAHE = FKnob("CLAHE Clip", 0.5, 4.0, step=0.1, initial=2.0, emoji="🧪", decimals=2)
        self.knTile = Knob("CLAHE Tile", 2, 16, step=1, initial=8, emoji="🧩")
        self.knBright = Knob("Brightness", -100, 100, step=1, initial=0, emoji="🔆")
        self.knContrast = Knob("Contrast", -100, 100, step=1, initial=0, emoji="🌓")
        self.knGamma = FKnob("Gamma", 0.20, 3.0, step=0.05, initial=1.0, emoji="🕹️", decimals=2)
        self.tgInvert = Toggle("Invert", emoji="🔁")

        for w in [self.knScale, self.knBilaD, self.knBilaSigC, self.knBilaSigS, self.knGaus,
                  self.knCLAHE, self.knTile, self.knBright, self.knContrast, self.knGamma, self.tgInvert]:
            add(w)
            self._connect_live(w)

        # Edge section
        add(self._sectionLabel("🪄 Edge / Threshold"))
        self.tgCanny = Toggle("Use Canny", emoji="⚙️"); self.tgCanny.setChecked(True)
        self.knLow = Knob("Canny Low", 0, 255, 1, 80, emoji="🔻")
        self.knHigh = Knob("Canny High", 0, 255, 1, 160, emoji="🔺")
        self.knAp = Knob("Canny Aperture", 3, 7, 2, 3, emoji="📐")

        self.tgAdaptive = Toggle("Use Adaptive Thresh", emoji="🧠")
        self.knBlock = Knob("Adaptive Block", 3, 51, 2, 11, emoji="🧱")
        self.knC = Knob("Adaptive C", -20, 20, 1, 2, emoji="➖")

        self.tgSimple = Toggle("Use Simple Thresh", emoji="🎚️")
        self.knT = Knob("Threshold", 0, 255, 1, 128, emoji="⚫")

        for w in [self.tgCanny, self.knLow, self.knHigh, self.knAp,
                  self.tgAdaptive, self.knBlock, self.knC,
                  self.tgSimple, self.knT]:
            add(w)
            self._connect_live(w)

        # Post/Export section
        add(self._sectionLabel("🧵 Contours → SVG"))
        self.knMinArea = FKnob("Min Area", 0.0, 500.0, 5.0, 20.0, emoji="📐", decimals=1)
        self.knEps = FKnob("Simplify ε (px)", 0.0, 10.0, 0.1, 1.5, emoji="✂️", decimals=2)
        self.tgClosed = Toggle("Close Paths", emoji="🔒")
        self.tgFill = Toggle("Fill Closed", emoji="🪄")

        self.knStroke = FKnob("Stroke Width", 0.1, 10.0, 0.1, 1.5, emoji="🖊️", decimals=2)
        self.colStroke = ColorChip("Color", initial="#C084FC", emoji="🖌️")
        self.colFill = ColorChip("Fill", initial="#00000000", emoji="🧴")
        self.colBG = ColorChip("Background", initial="#0B0B10", emoji="🪐")
        self.knExportScale = FKnob("Export Scale", 0.25, 4.0, 0.05, 1.0, emoji="📏", decimals=2)

        for w in [self.knMinArea, self.knEps, self.tgClosed, self.tgFill, self.knStroke,
                  self.colStroke, self.colFill, self.colBG, self.knExportScale]:
            add(w)
            self._connect_live(w)

        hint = QtWidgets.QLabel("💡 Tip: Use Presets, then fine-tune knobs.\nExport scale boosts SVG resolution.")
        hint.setObjectName("Hint")
        add(hint)

    def _sectionLabel(self, text:str) -> QtWidgets.QLabel:
        lbl = QtWidgets.QLabel(text)
        lbl.setObjectName("Section")
        return lbl

    def _connect_live(self, widget: QtWidgets.QWidget):
        if isinstance(widget, (Knob, FKnob)):
            widget.valueChanged.connect(self._schedule_live_update)
        elif isinstance(widget, Toggle):
            widget.stateChanged.connect(self._schedule_live_update)
        elif isinstance(widget, ColorChip):
            widget.colorChanged.connect(self._schedule_live_update)

    def _schedule_live_update(self, *args):
        self.liveTimer.start()

    # ------------- Actions -------------

    def on_open_clicked(self):
        self._open_dialog()

    def _open_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open image", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff)")
        if path:
            self.open_image(path)

    def open_image(self, path:str):
        bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            QtWidgets.QMessageBox.warning(self, "Open failed", "Could not read the file. Try another image.")
            return
        self.src_bgr = bgr
        self.reprocess()

    def on_preset(self):
        sender = self.sender()
        if sender is self.btnPreset1:
            # Crisp Lines
            self.tgCanny.setChecked(True); self.tgAdaptive.setChecked(False); self.tgSimple.setChecked(False)
            self.knLow.setValue(60); self.knHigh.setValue(180); self.knAp.setValue(3)
            self.knGaus.setValue(1)
            self.knBilaD.setValue(6); self.knBilaSigC.setValue(40); self.knBilaSigS.setValue(60)
            self.knEps.setValue(1.0); self.knMinArea.setValue(15.0)
            self.knStroke.setValue(1.5)
            self.colStroke.color = "#C084FC"; self.colStroke.setText(f"🎨 Color: {self.colStroke.color}")
        elif sender is self.btnPreset2:
            # Sketchy
            self.tgCanny.setChecked(True); self.knLow.setValue(30); self.knHigh.setValue(120); self.knAp.setValue(3)
            self.knGaus.setValue(5)
            self.knEps.setValue(2.5); self.knMinArea.setValue(10.0)
            self.knStroke.setValue(1.0)
            self.tgClosed.setChecked(False); self.tgFill.setChecked(False)
        elif sender is self.btnPreset3:
            # Bold Poster
            self.tgCanny.setChecked(False); self.tgAdaptive.setChecked(True); self.knBlock.setValue(21); self.knC.setValue(2)
            self.knEps.setValue(3.0); self.knMinArea.setValue(50.0)
            self.knStroke.setValue(2.8)
            self.tgClosed.setChecked(True); self.tgFill.setChecked(True)
            self.colFill.color = "#1F0B3A"; self.colFill.setText(f"🎨 Color: {self.colFill.color}")
        self.reprocess()

    def gather_settings(self) -> PipelineSettings:
        s = self.settings
        # Preprocess
        s.resize_scale = float(self.knScale.value())
        s.denoise_bilateral_d = int(self.knBilaD.value())
        s.denoise_bilateral_sigmaColor = int(self.knBilaSigC.value())
        s.denoise_bilateral_sigmaSpace = int(self.knBilaSigS.value())
        s.gaussian_ksize = int(self.knGaus.value())
        s.clahe_clip = float(self.knCLAHE.value())
        s.clahe_tile = int(self.knTile.value())
        s.brightness = int(self.knBright.value())
        s.contrast = int(self.knContrast.value())
        s.gamma = float(self.knGamma.value())
        s.invert = bool(self.tgInvert.isChecked())

        # Edge
        s.use_canny = bool(self.tgCanny.isChecked())
        s.canny_low = int(self.knLow.value())
        s.canny_high = int(self.knHigh.value())
        s.canny_aperture = int(self.knAp.value())
        s.use_adaptive = bool(self.tgAdaptive.isChecked())
        s.adaptive_block = int(self.knBlock.value())
        s.adaptive_C = int(self.knC.value())
        s.use_simple_thresh = bool(self.tgSimple.isChecked())
        s.simple_thresh = int(self.knT.value())

        # SVG
        s.min_contour_area = float(self.knMinArea.value())
        s.simplify_epsilon_px = float(self.knEps.value())
        s.close_paths = bool(self.tgClosed.isChecked())
        s.fill_closed = bool(self.tgFill.isChecked())
        s.stroke_width = float(self.knStroke.value())
        s.stroke_color = self.colStroke.color
        s.fill_color = self.colFill.color
        s.background = self.colBG.color
        s.export_scale = float(self.knExportScale.value())
        return s

    def reprocess(self):
        if self.src_bgr is None:
            return
        s = self.gather_settings()
        overlay, edges = process_to_edges(self.src_bgr, s)
        self.preview_bgr = overlay
        self.edges = edges
        # Show preview image
        self.imageView.set_pixmap_from_cv(self.preview_bgr)

    def on_export(self):
        if self.src_bgr is None or self.edges is None:
            QtWidgets.QMessageBox.information(self, "Nothing to export", "Open an image first. 🖼️")
            return

        s = self.gather_settings()

        # edges currently at preview scale — rebuild at full input res before export
        s_full = PipelineSettings(**vars(s))
        s_full.resize_scale = 1.0  # full res processing for export
        overlay_full, edges_full = process_to_edges(self.src_bgr, s_full)
        if edges_full is None:
            QtWidgets.QMessageBox.warning(self, "Export failed", "Could not reprocess full-size edges.")
            return

        H, W = edges_full.shape
        svg, npaths = trace_svg(edges_full, s_full, base_size=(H, W))

        if svg is None:
            QtWidgets.QMessageBox.warning(self, "Export failed", "SVG generation failed.")
            return

        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save SVG", "lineart.svg", "SVG (*.svg)")
        if not path:
            return
        try:
            svg.saveas(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save error", f"Could not save SVG:\n{e}")
            return

        QtWidgets.QMessageBox.information(self, "Exported 🎉", f"Saved {npaths} path(s) to:\n{path}")

    # ------------- Styling -------------

    def _qss(self) -> str:
        # Purple/black, rounded, flat UI
        return """
        /* Base */
        #Root {
            background-color: #0B0B10;
            color: #EDE9FE;
            font-family: Inter, "SF Pro Display", Segoe UI, Roboto, Arial;
            font-size: 14px;
        }
        QScrollBar:vertical, QScrollBar:horizontal {
            background: #141320; border: none; border-radius: 8px; margin: 4px;
        }
        QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
            background: #6D28D9; border-radius: 8px; min-height: 24px; min-width: 24px;
        }
        QToolTip {
            background-color: #1B132F; color: #EDE9FE; border: 1px solid #7C3AED; border-radius: 8px;
        }

        /* Cards & banners */
        #Banner {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1B132F, stop:1 #0D0A1A);
            border: 1px solid #2A104E; border-radius: 18px;
        }
        #BannerLabel {
            color: #EDE9FE; font-weight: 600; font-size: 18px;
        }
        #ImageView {
            background: #0F0E1A; border: 1px solid #2A104E; border-radius: 18px; padding: 8px; color: #6B7280;
        }
        #Section {
            color: #C4B5FD; font-weight: 600; margin-top: 10px; margin-bottom: 4px;
            padding: 6px 10px; background: #141320; border: 1px solid #2A104E; border-radius: 12px;
        }
        #Hint {
            color: #9FA6B2; padding: 8px; background: #141320; border: 1px dashed #2A104E; border-radius: 12px;
        }

        /* Buttons */
        QPushButton {
            background: #141320; color: #EDE9FE; border: 1px solid #2A104E; border-radius: 12px; padding: 8px 12px;
        }
        QPushButton:hover { border-color: #7C3AED; }
        QPushButton:pressed { background: #1B132F; }
        #PrimaryButton {
            background: #7C3AED; color: white; border: none; border-radius: 14px; padding: 10px 16px;
        }
        #PrimaryButton:hover { background: #8B5CF6; }
        #PrimaryButton:pressed { background: #6D28D9; }

        /* Sliders */
        QSlider::groove:horizontal {
            height: 8px; background: #1B132F; border-radius: 6px;
        }
        QSlider::handle:horizontal {
            background: #8B5CF6; width: 18px; height: 18px; margin: -5px 0; border-radius: 10px;
            border: 1px solid #C4B5FD;
        }

        /* Spinboxes */
        QSpinBox, QDoubleSpinBox, QLineEdit, QTextEdit {
            background: #0F0E1A; color: #EDE9FE; border: 1px solid #2A104E; border-radius: 10px; padding: 6px 8px;
            selection-background-color: #7C3AED;
        }

        /* Labels & toggles */
        #KnobLabel { color: #E9D5FF; }
        #Toggle { color: #EDE9FE; spacing: 8px; }
        #ColorChip { font-weight: 600; }

        /* Frames */
        #Knob {
            background: #0F0E1A; border: 1px solid #1E1633; border-radius: 12px;
        }
        """

# -----------------------------
# Entry
# -----------------------------

def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("ReadyTracer")
    app.setStyle("Fusion")
    w = App()
    w.resize(1200, 720)
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
