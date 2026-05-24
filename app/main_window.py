"""
main_window.py — The main application window built with PySide6.

"""

from pathlib import Path

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QListWidget, QLabel, QPushButton,
    QComboBox, QProgressBar, QFileDialog, QTabWidget,
    QScrollArea, QStatusBar, QGroupBox, QMessageBox,
    QLineEdit, QMenu, QDialog, QTableWidget, QTableWidgetItem,
    QSpinBox, QHeaderView, QSizePolicy, QFrame,
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QPixmap, QImage, QPainter, QColor, QPen

from app.worker import TranslationWorker
from app import settings as _settings

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp",
    ".bmp", ".tiff", ".tif", ".gif",
}

SOURCE_LANGUAGES = {
    "Japanese":                    "ja",
    "Japanese (Manga OCR)":        "ja_manga",
    "Korean":                      "ko",
    "Chinese Simplified":          "ch_sim",
    "Chinese Traditional":         "ch_tra",
    "English":                     "en",
}

# Friendly label → BCP-47 code used by deep_translator targets
TARGET_LANGUAGES = {
    "English":              "en",
    "Japanese":             "ja",
    "Korean":               "ko",
    "Chinese Simplified":   "zh-CN",
    "Chinese Traditional":  "zh-TW",
    "Spanish":              "es",
    "French":               "fr",
    "German":               "de",
    "Portuguese":           "pt",
    "Russian":              "ru",
    "Italian":              "it",
    "Vietnamese":           "vi",
    "Thai":                 "th",
}

TRANSLATION_SERVICES = [
    "Google Translate",
    "OPUS-MT (Offline)",
    "Local LLM (GGUF)",
]

# Each entry: repo = HuggingFace repo path, file = filename inside the repo
GGUF_CATALOG = [
    {
        "name": "Sakura 14B v1.0  (Q4_K_M)",
        "desc": "Best for Japanese → Chinese/English manga. Purpose-built for manga/visual novels.",
        "repo": "SakuraLLM/Sakura-14B-Qwen2.5-v1.0-GGUF",
        "file": "sakura-14b-qwen2.5-v1.0-q4_k_m.gguf",
        "size": "8.9 GB",
    },
    {
        "name": "Qwen3 8B  (Q4_K_M)  — recommended",
        "desc": "Best general multilingual model. Excellent for all CJK language pairs.",
        "repo": "bartowski/Qwen3-8B-GGUF",
        "file": "Qwen3-8B-Q4_K_M.gguf",
        "size": "5.2 GB",
    },
    {
        "name": "Qwen2.5 7B  (Q4_K_M)",
        "desc": "Proven multilingual quality. Good for Japanese/Chinese to English.",
        "repo": "bartowski/Qwen2.5-7B-Instruct-GGUF",
        "file": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "size": "4.7 GB",
    },
    {
        "name": "Qwen2.5 7B Abliterated  (Q4_K_M)  — uncensored",
        "desc": "No safety restrictions. Use for adult manga content.",
        "repo": "mradermacher/Qwen2.5-7B-Instruct-abliterated-GGUF",
        "file": "Qwen2.5-7B-Instruct-abliterated.Q4_K_M.gguf",
        "size": "4.7 GB",
    },
    {
        "name": "LLaMA 3.1 8B Abliterated  (Q4_K_M)  — uncensored",
        "desc": "Uncensored LLaMA 3.1. Good uncensored fallback for non-CJK pairs.",
        "repo": "bartowski/Meta-Llama-3.1-8B-Instruct-abliterated-GGUF",
        "file": "Meta-Llama-3.1-8B-Instruct-abliterated-Q4_K_M.gguf",
        "size": "4.9 GB",
    },
]


# ── Custom image viewer ─────────────────────────────────────────────────── #

class ImageViewer(QLabel):
    """
    Zoomable image label inside a QScrollArea.
    - Fit mode (default): image scales to fill the available area.
    - Zoom mode: Ctrl+scroll to zoom in/out; double-click to return to fit.
    """

    def __init__(self, placeholder="No image"):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(50, 50)
        self.setStyleSheet("background-color: #1e1e1e; color: #666; font-size: 14px;")
        self.setText(placeholder)
        self._pixmap_original = None
        self._zoom     = 1.0
        self._fit_mode = True
        self._pan_start  = None
        self._pan_hval   = 0
        self._pan_vval   = 0

    def set_image_path(self, path):
        self._pixmap_original = QPixmap(str(path))
        self._fit_mode = True
        self._refresh()

    def set_pil_image(self, pil_img):
        rgb  = pil_img.convert("RGB")
        data = rgb.tobytes("raw", "RGB")
        qimg = QImage(data, rgb.width, rgb.height, rgb.width * 3, QImage.Format_RGB888)
        self._pixmap_original = QPixmap.fromImage(qimg)
        self._fit_mode = True
        self._refresh()

    def clear(self):
        self._pixmap_original = None
        self.setText("No image")

    # ── zoom helpers ── #

    def _scroll_area(self):
        vp = self.parentWidget()
        if vp:
            sa = vp.parentWidget()
            if isinstance(sa, QScrollArea):
                return sa
        return None

    def _refresh(self):
        if not self._pixmap_original:
            return
        sa = self._scroll_area()
        if self._fit_mode:
            if sa and not sa.widgetResizable():
                sa.setWidgetResizable(True)
            scaled = self._pixmap_original.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.setCursor(Qt.ArrowCursor)
        else:
            if sa and sa.widgetResizable():
                sa.setWidgetResizable(False)
            zw = max(1, int(self._pixmap_original.width()  * self._zoom))
            zh = max(1, int(self._pixmap_original.height() * self._zoom))
            scaled = self._pixmap_original.scaled(
                zw, zh, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.resize(scaled.width(), scaled.height())
            self.setCursor(Qt.OpenHandCursor)
        super().setPixmap(scaled)

    def wheelEvent(self, event):
        if self._pixmap_original and (event.modifiers() & Qt.ControlModifier):
            self._fit_mode = False
            factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            self._zoom = max(0.1, min(8.0, self._zoom * factor))
            self._refresh()
            event.accept()
        else:
            event.ignore()  # let the scroll area handle normal scrolling

    def mousePressEvent(self, event):
        if not self._fit_mode and self._pixmap_original and \
                event.button() in (Qt.LeftButton, Qt.MiddleButton):
            self._pan_start = event.globalPosition().toPoint()
            sa = self._scroll_area()
            if sa:
                self._pan_hval = sa.horizontalScrollBar().value()
                self._pan_vval = sa.verticalScrollBar().value()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._pan_start is not None:
            delta = event.globalPosition().toPoint() - self._pan_start
            sa = self._scroll_area()
            if sa:
                sa.horizontalScrollBar().setValue(self._pan_hval - delta.x())
                sa.verticalScrollBar().setValue(self._pan_vval - delta.y())
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._pan_start is not None:
            self._pan_start = None
            self.setCursor(Qt.OpenHandCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self._pixmap_original:
            self._pan_start = None
            self._fit_mode = True
            self._zoom = 1.0
            self._refresh()

    def resizeEvent(self, event):
        if self._fit_mode:
            self._refresh()
        super().resizeEvent(event)


# ── Interactive region canvas ───────────────────────────────────────────── #

class RegionCanvas(QWidget):
    """
    Displays a PIL image with coloured bounding-box overlays.
    Supports:
      • Click a box  → select it (emits region_selected(idx))
      • Drag a box   → move it
      • Drag a corner handle → resize it
      • Drag on empty area   → draw a brand-new box (emits region_added)
    All coordinates stored internally in image-pixel space.
    """

    region_selected = Signal(int)   # idx of selected region (-1 = none)
    regions_changed = Signal()      # any move / resize / add

    _HANDLE = 8   # half-size of corner handle in widget pixels

    def __init__(self):
        super().__init__()
        self._pixmap   = None
        self._regions  = []          # list of {'bbox', 'text', 'font_size'} dicts
        self._selected = -1
        self._mode     = None        # None | 'move' | 'resize' | 'draw'
        self._drag_start   = None    # (wx, wy) at mouse-down
        self._orig_bbox    = None    # copy of bbox at drag start
        self._resize_handle = None   # 'tl' | 'tr' | 'bl' | 'br'
        self._draw_end     = None    # current draw corner (image coords)
        self.setMouseTracking(True)
        self.setMinimumSize(300, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    # ── public API ── #

    def set_image(self, pil_img):
        rgb  = pil_img.convert("RGB")
        data = rgb.tobytes("raw", "RGB")
        qimg = QImage(data, rgb.width, rgb.height, rgb.width * 3, QImage.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qimg)
        self.update()

    def set_regions(self, regions):
        self._regions  = regions
        self._selected = -1
        self.update()

    def set_selected(self, idx):
        self._selected = idx
        self.update()

    def get_regions(self):
        return self._regions

    def get_selected(self):
        return self._selected

    def delete_selected(self):
        if 0 <= self._selected < len(self._regions):
            self._regions.pop(self._selected)
            self._selected = min(self._selected, len(self._regions) - 1)
            self.regions_changed.emit()
            self.update()

    # ── coordinate helpers ── #

    def _layout(self):
        """Return (x0, y0, scale) — image origin and scale in the widget."""
        if not self._pixmap:
            return 0.0, 0.0, 1.0
        pw, ph = self._pixmap.width(), self._pixmap.height()
        ww, wh = self.width(), self.height()
        scale  = min(ww / pw, wh / ph)
        return (ww - pw * scale) / 2, (wh - ph * scale) / 2, scale

    def _to_widget(self, ix, iy):
        x0, y0, s = self._layout()
        return x0 + ix * s, y0 + iy * s

    def _to_image(self, wx, wy):
        x0, y0, s = self._layout()
        return (wx - x0) / s, (wy - y0) / s

    def _bbox_rect(self, bbox):
        """Axis-aligned rect (wx1,wy1,wx2,wy2) for a bbox in widget coords."""
        xs = [p[0] for p in bbox]; ys = [p[1] for p in bbox]
        wx1, wy1 = self._to_widget(min(xs), min(ys))
        wx2, wy2 = self._to_widget(max(xs), max(ys))
        return wx1, wy1, wx2, wy2

    # ── paint ── #

    def paintEvent(self, _event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(30, 30, 30))

        if self._pixmap:
            x0, y0, s = self._layout()
            pw, ph = self._pixmap.width(), self._pixmap.height()
            p.drawPixmap(int(x0), int(y0), int(pw * s), int(ph * s), self._pixmap)

        h = self._HANDLE
        for i, r in enumerate(self._regions):
            x1, y1, x2, y2 = self._bbox_rect(r['bbox'])
            sel = (i == self._selected)
            box_col  = QColor(255, 165, 0) if sel else QColor(80, 200, 120)
            fill_col = QColor(box_col.red(), box_col.green(), box_col.blue(), 35)
            p.setPen(QPen(box_col, 2 if sel else 1))
            p.setBrush(fill_col)
            p.drawRect(int(x1), int(y1), int(x2 - x1), int(y2 - y1))

            if sel:
                p.setBrush(box_col)
                p.setPen(Qt.NoPen)
                for cx, cy in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
                    p.drawRect(int(cx - h), int(cy - h), 2 * h, 2 * h)

        # Show in-progress draw rectangle
        if self._mode == 'draw' and self._draw_end and self._drag_start:
            ix1, iy1 = self._to_image(*self._drag_start)
            ix2, iy2 = self._draw_end
            wx1, wy1 = self._to_widget(min(ix1, ix2), min(iy1, iy2))
            wx2, wy2 = self._to_widget(max(ix1, ix2), max(iy1, iy2))
            p.setPen(QPen(QColor(255, 80, 80), 2, Qt.DashLine))
            p.setBrush(QColor(255, 80, 80, 30))
            p.drawRect(int(wx1), int(wy1), int(wx2 - wx1), int(wy2 - wy1))

        p.end()

    # ── mouse events ── #

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        wx, wy = event.position().x(), event.position().y()

        # Corner handle of selected box?
        if self._selected >= 0:
            h = self._hit_handle(wx, wy, self._selected)
            if h:
                self._mode          = 'resize'
                self._resize_handle = h
                self._drag_start    = (wx, wy)
                self._orig_bbox     = [list(pt) for pt in self._regions[self._selected]['bbox']]
                return

        # Inside any box?
        for i in range(len(self._regions) - 1, -1, -1):
            if self._hit_region(wx, wy, i):
                self._selected  = i
                self._mode      = 'move'
                self._drag_start = (wx, wy)
                self._orig_bbox  = [list(pt) for pt in self._regions[i]['bbox']]
                self.region_selected.emit(i)
                self.update()
                return

        # Empty area → start drawing a new box
        self._selected   = -1
        self._mode       = 'draw'
        self._drag_start = (wx, wy)
        self._draw_end   = self._to_image(wx, wy)
        self.region_selected.emit(-1)
        self.update()

    def mouseMoveEvent(self, event):
        if not self._mode:
            return
        wx, wy = event.position().x(), event.position().y()
        _, _, s = self._layout()

        if self._mode == 'move':
            dx = (wx - self._drag_start[0]) / s
            dy = (wy - self._drag_start[1]) / s
            self._regions[self._selected]['bbox'] = [
                [pt[0] + dx, pt[1] + dy] for pt in self._orig_bbox
            ]
            self.update()

        elif self._mode == 'resize':
            ix, iy = self._to_image(wx, wy)
            orig   = self._orig_bbox
            ox1 = min(pt[0] for pt in orig); ox2 = max(pt[0] for pt in orig)
            oy1 = min(pt[1] for pt in orig); oy2 = max(pt[1] for pt in orig)
            coords = {
                'tl': (ix,  iy,  ox2, oy2),
                'tr': (ox1, iy,  ix,  oy2),
                'bl': (ix,  oy1, ox2, iy),
                'br': (ox1, oy1, ix,  iy),
            }[self._resize_handle]
            nx1, ny1, nx2, ny2 = coords
            if nx2 - nx1 > 8 and ny2 - ny1 > 8:
                self._regions[self._selected]['bbox'] = [
                    [nx1, ny1], [nx2, ny1], [nx2, ny2], [nx1, ny2]
                ]
            self.update()

        elif self._mode == 'draw':
            self._draw_end = self._to_image(wx, wy)
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.LeftButton:
            return

        if self._mode == 'draw' and self._drag_start and self._draw_end:
            ix1, iy1 = self._to_image(*self._drag_start)
            ix2, iy2 = self._draw_end
            if abs(ix2 - ix1) > 10 and abs(iy2 - iy1) > 10:
                nx1, nx2 = min(ix1, ix2), max(ix1, ix2)
                ny1, ny2 = min(iy1, iy2), max(iy1, iy2)
                self._regions.append({
                    'bbox': [[nx1, ny1], [nx2, ny1], [nx2, ny2], [nx1, ny2]],
                    'text': '',
                    'font_size': 12,
                })
                self._selected = len(self._regions) - 1
                self.region_selected.emit(self._selected)
                self.regions_changed.emit()

        elif self._mode in ('move', 'resize'):
            self.regions_changed.emit()

        self._mode = None
        self.update()

    # ── hit-testing ── #

    def _hit_region(self, wx, wy, idx):
        x1, y1, x2, y2 = self._bbox_rect(self._regions[idx]['bbox'])
        return x1 <= wx <= x2 and y1 <= wy <= y2

    def _hit_handle(self, wx, wy, idx):
        h  = self._HANDLE
        x1, y1, x2, y2 = self._bbox_rect(self._regions[idx]['bbox'])
        for name, cx, cy in [('tl', x1, y1), ('tr', x2, y1),
                               ('bl', x1, y2), ('br', x2, y2)]:
            if abs(wx - cx) <= h and abs(wy - cy) <= h:
                return name
        return None


# ── Text edit dialog ─────────────────────────────────────────────────────── #

class TextEditDialog(QDialog):
    """
    Lets the user:
      • Edit translated text and font size per region (right-side table)
      • Move / resize existing boxes by dragging them on the canvas
      • Draw a new box on the canvas, then type the desired translated text
      • Click "Update Preview" to re-render with current edits
      • Click "Save" to commit changes back to the main window
    """

    def __init__(self, parent, file_path, base_img, regions, target_lang):
        super().__init__(parent)
        self.setWindowTitle(f"Edit Texts — {Path(file_path).name}")
        self.setMinimumSize(960, 640)
        self.resize(1200, 760)

        self._base_img    = base_img
        self._target_lang = target_lang
        self._pipeline    = None
        self.result_image = None

        self._regions = [dict(r) for r in regions]

        # Debounce timer — updates the canvas preview 400 ms after the last edit
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(400)
        self._preview_timer.timeout.connect(self._update_preview)

        self._build_ui()
        self._refresh_table()
        self._update_preview()

    # ── pipeline (lightweight, render-only instance) ── #

    def _get_pipeline(self):
        if self._pipeline is None:
            from app.pipeline import ComicTranslationPipeline
            self._pipeline = ComicTranslationPipeline(
                source_lang="en",           # unused for rendering
                target_lang=self._target_lang,
                ocr_reader=None,
                translator=None,
            )
        return self._pipeline

    # ── UI construction ── #

    def _build_ui(self):
        main = QHBoxLayout(self)
        main.setSpacing(8)

        # ── left: canvas ── #
        left = QVBoxLayout()
        hint = QLabel("Click a box to select • Drag to move • Drag corner to resize • Drag empty area to draw a new box")
        hint.setStyleSheet("color: #aaa; font-size: 11px;")
        hint.setWordWrap(True)
        left.addWidget(hint)

        self._canvas = RegionCanvas()
        self._canvas.set_image(self._base_img)
        self._canvas.region_selected.connect(self._on_canvas_select)
        self._canvas.regions_changed.connect(self._on_regions_changed)

        canvas_scroll = QScrollArea()
        canvas_scroll.setWidget(self._canvas)
        canvas_scroll.setWidgetResizable(True)
        left.addWidget(canvas_scroll, stretch=1)

        left_w = QWidget()
        left_w.setLayout(left)
        left_w.setMinimumWidth(460)
        main.addWidget(left_w, stretch=3)

        # ── right: table + controls ── #
        right = QVBoxLayout()
        right.addWidget(QLabel("Text regions (edit text or font size, or draw a new box on the left):"))

        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["#", "Translated Text", "Size"])
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self._table.setColumnWidth(0, 32)
        self._table.setColumnWidth(2, 62)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.itemSelectionChanged.connect(self._on_table_select)
        self._table.itemChanged.connect(self._on_table_item_changed)
        right.addWidget(self._table, stretch=1)

        # Delete selected region
        btn_del = QPushButton("Delete Selected Region")
        btn_del.clicked.connect(self._delete_selected)
        right.addWidget(btn_del)

        # Global size row
        grow = QHBoxLayout()
        grow.addWidget(QLabel("Set all sizes to:"))
        self._global_spin = QSpinBox()
        self._global_spin.setRange(6, 72)
        self._global_spin.setValue(12)
        grow.addWidget(self._global_spin)
        btn_all = QPushButton("Apply to All")
        btn_all.clicked.connect(self._apply_size_all)
        grow.addWidget(btn_all)
        grow.addStretch()
        right.addLayout(grow)

        # Action buttons
        brow = QHBoxLayout()
        brow.addStretch()
        btn_preview = QPushButton("Update Preview")
        btn_save    = QPushButton("Save Changes")
        btn_cancel  = QPushButton("Cancel")
        btn_save.setStyleSheet("background-color: #5a9e5a; color: white; font-weight: bold; padding: 0 14px;")
        btn_preview.clicked.connect(self._update_and_preview)
        btn_save.clicked.connect(self._save_and_close)
        btn_cancel.clicked.connect(self.reject)
        for b in (btn_preview, btn_save, btn_cancel):
            b.setFixedHeight(32)
            brow.addWidget(b)
        right.addLayout(brow)

        right_w = QWidget()
        right_w.setLayout(right)
        right_w.setMinimumWidth(320)
        right_w.setMaximumWidth(440)
        main.addWidget(right_w, stretch=2)

    # ── table helpers ── #

    def _refresh_table(self):
        prev_sel = self._canvas.get_selected()
        self._table.blockSignals(True)
        self._table.setRowCount(len(self._regions))
        for i, r in enumerate(self._regions):
            num = QTableWidgetItem(str(i + 1))
            num.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self._table.setItem(i, 0, num)
            self._table.setItem(i, 1, QTableWidgetItem(r.get('text', '')))
            spin = QSpinBox()
            spin.setRange(6, 72)
            spin.setValue(r.get('font_size', 12))
            spin.valueChanged.connect(lambda v, idx=i: self._on_size_changed(idx, v))
            self._table.setCellWidget(i, 2, spin)
        self._table.blockSignals(False)
        self._canvas.set_regions(self._regions)
        # Restore selection so move/resize/delete don't lose the active box
        if 0 <= prev_sel < len(self._regions):
            self._canvas.set_selected(prev_sel)
            self._table.blockSignals(True)
            self._table.selectRow(prev_sel)
            self._table.blockSignals(False)

    def _collect_from_table(self):
        """Push table edits back into self._regions."""
        for i in range(len(self._regions)):
            item = self._table.item(i, 1)
            spin = self._table.cellWidget(i, 2)
            if item is not None:
                self._regions[i]['text'] = item.text()
            if spin is not None:
                self._regions[i]['font_size'] = spin.value()

    # ── signal handlers ── #

    def _on_canvas_select(self, idx):
        if 0 <= idx < self._table.rowCount():
            self._table.selectRow(idx)

    def _on_table_select(self):
        rows = self._table.selectedItems()
        if rows:
            self._canvas.set_selected(self._table.currentRow())

    def _on_table_item_changed(self, item):
        if item.column() == 1 and 0 <= item.row() < len(self._regions):
            self._regions[item.row()]['text'] = item.text()
            self._preview_timer.start()

    def _on_size_changed(self, idx, value):
        if 0 <= idx < len(self._regions):
            self._regions[idx]['font_size'] = value
            self._preview_timer.start()

    def _on_regions_changed(self):
        # A box was added, moved, or resized via the canvas
        self._collect_from_table()
        self._refresh_table()
        self._preview_timer.start()

    def _delete_selected(self):
        self._collect_from_table()
        self._canvas.delete_selected()
        # canvas.delete_selected already modifies self._regions in-place (shared list)
        self._refresh_table()

    def _apply_size_all(self):
        size = self._global_spin.value()
        for i in range(self._table.rowCount()):
            spin = self._table.cellWidget(i, 2)
            if spin:
                spin.setValue(size)
        for r in self._regions:
            r['font_size'] = size

    def _update_and_preview(self):
        self._collect_from_table()
        self._update_preview()

    def _update_preview(self):
        img = self._get_pipeline().render_from_regions(self._base_img, self._regions)
        self._canvas.set_image(img)

    def _save_and_close(self):
        self._preview_timer.stop()
        self._collect_from_table()
        try:
            self.result_image = self._get_pipeline().render_from_regions(
                self._base_img, self._regions
            )
        except Exception as e:
            print(f"[TextEditDialog] Render error: {e}")
            import traceback; traceback.print_exc()
        self.accept()


# ── Zone-picker dialog ──────────────────────────────────────────────────── #

class ZonePickDialog(QDialog):
    """
    Shows the original image in an interactive canvas so the user can draw
    rectangles over text regions that the auto-detector missed.  The saved
    bboxes are passed to the pipeline as forced regions, which are cropped
    and OCR'd separately then merged into the normal translate→render flow.
    """

    def __init__(self, parent, file_path, existing_zones=None):
        super().__init__(parent)
        self.setWindowTitle(f"Mark Zones — {Path(file_path).name}")
        self.setMinimumSize(700, 560)
        self.resize(960, 700)

        # Wrap existing bare-bbox lists as region dicts so RegionCanvas is happy
        self._regions = [
            {'bbox': z, 'text': '', 'font_size': 12}
            for z in (existing_zones or [])
        ]
        self._file_path = file_path
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        hint = QLabel(
            "Drag on the image to mark zones where the auto-detector missed text. "
            "Click a box to select it; drag to move; drag a corner handle to resize."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #aaa; font-size: 11px;")
        layout.addWidget(hint)

        self._canvas = RegionCanvas()
        scroll = QScrollArea()
        scroll.setWidget(self._canvas)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll, stretch=1)

        # Load image after canvas exists
        from PIL import Image as PilImage
        pil = PilImage.open(self._file_path).convert("RGB")
        self._canvas.set_image(pil)
        self._canvas.set_regions(self._regions)

        brow = QHBoxLayout()
        brow.addStretch()
        btn_clear = QPushButton("Clear All")
        btn_ok    = QPushButton("Save Zones")
        btn_cancel = QPushButton("Cancel")
        btn_ok.setStyleSheet(
            "background-color: #5a9e5a; color: white; font-weight: bold; padding: 0 14px;"
        )
        btn_clear.clicked.connect(self._clear_all)
        btn_ok.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)
        for b in (btn_clear, btn_ok, btn_cancel):
            b.setFixedHeight(32)
            brow.addWidget(b)
        layout.addLayout(brow)

    def _clear_all(self):
        self._regions.clear()
        self._canvas.set_regions(self._regions)

    def get_zones(self):
        """Return list of bboxes (image pixel coordinates)."""
        return [r['bbox'] for r in self._canvas.get_regions()]


# ── Model row widget (one entry in the GGUF Model Manager) ──────────────── #

class ModelRowWidget(QWidget):
    use_requested = Signal(str)   # path to the downloaded .gguf file

    def __init__(self, info: dict, models_dir: str, parent=None):
        super().__init__(parent)
        self._info       = info
        self._models_dir = models_dir
        self._worker     = None
        self._btn_connected = False
        self._build_ui()

    def _dest_path(self) -> str:
        return str(Path(self._models_dir) / self._info["file"])

    def _is_downloaded(self) -> bool:
        return Path(self._dest_path()).exists()

    def _build_ui(self):
        outer = QHBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)

        info_col = QVBoxLayout()
        name_lbl = QLabel(f"<b>{self._info['name']}</b>")
        repo_lbl = QLabel(f"<small>{self._info['repo']}</small>")
        repo_lbl.setStyleSheet("color: #888; font-size: 10px;")
        desc_lbl = QLabel(self._info["desc"])
        desc_lbl.setStyleSheet("color: #aaa; font-size: 11px;")
        desc_lbl.setWordWrap(True)
        for w in (name_lbl, repo_lbl, desc_lbl):
            info_col.addWidget(w)
        info_w = QWidget(); info_w.setLayout(info_col)
        outer.addWidget(info_w, stretch=3)

        right_col = QVBoxLayout()
        right_col.setSpacing(4)

        self._status_lbl = QLabel()
        self._status_lbl.setStyleSheet("font-size: 11px;")
        right_col.addWidget(self._status_lbl)

        self._progress = QProgressBar()
        self._progress.setFixedHeight(12)
        self._progress.setTextVisible(False)
        self._progress.setVisible(False)
        right_col.addWidget(self._progress)

        self._btn = QPushButton()
        self._btn.setFixedHeight(28)
        right_col.addWidget(self._btn)
        right_col.addStretch()

        right_w = QWidget(); right_w.setFixedWidth(220); right_w.setLayout(right_col)
        outer.addWidget(right_w)

        self._refresh_state()

    def _refresh_state(self):
        if self._is_downloaded():
            self._status_lbl.setText("Downloaded")
            self._status_lbl.setStyleSheet("color: #5a9e5a; font-size: 11px; font-weight: bold;")
            self._set_btn("Use This Model", self._use_current)
        else:
            self._status_lbl.setText(f"Not downloaded  ({self._info['size']})")
            self._status_lbl.setStyleSheet("color: #aaa; font-size: 11px;")
            self._set_btn("Download", self._start_download)

    def _set_btn(self, text, slot):
        self._btn.setText(text)
        if self._btn_connected:
            self._btn.clicked.disconnect()
        self._btn.clicked.connect(slot)
        self._btn_connected = True

    def _use_current(self):
        self.use_requested.emit(self._dest_path())

    def _start_download(self):
        from app.worker import ModelDownloadWorker
        url = (f"https://huggingface.co/{self._info['repo']}"
               f"/resolve/main/{self._info['file']}")
        self._worker = ModelDownloadWorker(url, self._dest_path())
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

        self._status_lbl.setText("Downloading…")
        self._status_lbl.setStyleSheet("color: #4a8eff; font-size: 11px;")
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setVisible(True)
        self._set_btn("Cancel", self._cancel)

    def _cancel(self):
        if self._worker:
            self._worker.stop()
            self._worker = None
        self._progress.setVisible(False)
        self._refresh_state()

    def _on_progress(self, done: int, total: int):
        if total > 0:
            self._progress.setValue(int(done * 100 / total))
            self._status_lbl.setText(f"Downloading…  {done} / {total} MB")
        else:
            self._progress.setRange(0, 0)

    def _on_done(self, path: str):
        self._worker = None
        self._progress.setVisible(False)
        self._refresh_state()

    def _on_error(self, msg: str):
        self._worker = None
        self._progress.setVisible(False)
        self._status_lbl.setText("Download failed")
        self._status_lbl.setStyleSheet("color: #cc4444; font-size: 11px;")
        self._set_btn("Retry", self._start_download)
        QMessageBox.warning(
            self, "Download failed",
            f"Could not download {self._info['file']}.\n\n{msg}\n\n"
            "The file may have been renamed on HuggingFace. Open the repo link below,\n"
            "download the Q4_K_M file manually, then use the Browse button in settings\n"
            "to point the app at it.\n\n"
            f"Repo: https://huggingface.co/{self._info['repo']}"
        )


# ── Model Manager dialog (GGUF) ──────────────────────────────────────────── #

class ModelManagerDialog(QDialog):
    def __init__(self, parent=None, models_dir: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Model Manager — Local GGUF")
        self.setMinimumSize(720, 500)
        self.resize(860, 640)
        self.chosen_path: str | None = None
        self._models_dir = models_dir
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        hdr = QLabel(
            "<b>Recommended models</b> — click <i>Download</i> to fetch from HuggingFace, "
            "then <i>Use This Model</i> to select it.  "
            "If a download fails with 404, the filename may have changed — "
            "download manually from the repo link and use <i>Browse existing file…</i> below."
        )
        hdr.setWordWrap(True)
        layout.addWidget(hdr)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine); sep.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setSpacing(4)
        vbox.setContentsMargins(2, 2, 2, 2)

        for info in GGUF_CATALOG:
            row = ModelRowWidget(info, self._models_dir)
            row.use_requested.connect(self._on_use)
            frame = QFrame()
            frame.setFrameShape(QFrame.StyledPanel)
            fl = QVBoxLayout(frame)
            fl.setContentsMargins(4, 4, 4, 4)
            fl.addWidget(row)
            vbox.addWidget(frame)

        vbox.addStretch()
        scroll.setWidget(container)
        layout.addWidget(scroll, stretch=1)

        sep2 = QFrame(); sep2.setFrameShape(QFrame.HLine); sep2.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep2)

        brow = QHBoxLayout()
        btn_browse = QPushButton("Browse existing .gguf file…")
        btn_browse.clicked.connect(self._browse_existing)
        btn_close  = QPushButton("Close")
        btn_close.setFixedHeight(32)
        btn_browse.setFixedHeight(32)
        btn_close.clicked.connect(self.reject)
        brow.addWidget(btn_browse)
        brow.addStretch()
        brow.addWidget(btn_close)
        layout.addLayout(brow)

    def _on_use(self, path: str):
        self.chosen_path = path
        self.accept()

    def _browse_existing(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select GGUF model file", self._models_dir,
            "GGUF models (*.gguf)"
        )
        if path:
            self.chosen_path = path
            self.accept()


# ── Main window ─────────────────────────────────────────────────────────── #

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Comic Translator")
        self.setMinimumSize(900, 640)
        self.resize(1200, 800)

        self.file_list      = []   # ordered list of absolute file path strings
        self.translated     = {}   # path → PIL Image (filled as images finish)
        self.edit_data      = {}   # path → {'base': PIL, 'regions': list, 'target_lang': str}
        self.manual_zones   = {}   # path → list of bboxes drawn by user (forced OCR regions)
        self.worker         = None
        self._cfg           = _settings.load()

        self._build_ui()
        self._apply_settings()
        self.statusBar().showMessage("Ready — add files or a folder to begin.")

    # ------------------------------------------------------------------ #
    # UI construction                                                      #
    # ------------------------------------------------------------------ #

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(8, 8, 8, 4)
        root_layout.setSpacing(6)

        root_layout.addLayout(self._build_toolbar())

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(1, 1)   # right panel takes all extra space
        root_layout.addWidget(splitter)

    def _build_toolbar(self):
        row = QHBoxLayout()

        self.btn_add_files      = QPushButton("Add Files")
        self.btn_add_folder     = QPushButton("Add Folder")
        self.btn_remove         = QPushButton("Remove Selected")
        self.btn_clear          = QPushButton("Clear All")
        self.btn_zones          = QPushButton("Mark Zones")
        self.btn_save_all       = QPushButton("Save All")
        self.btn_edit           = QPushButton("Edit Texts")
        self.btn_translate_one  = QPushButton("Translate Selected")
        self.btn_translate      = QPushButton("Translate All")

        for btn in (self.btn_add_files, self.btn_add_folder,
                    self.btn_remove, self.btn_clear):
            btn.setFixedHeight(32)
            row.addWidget(btn)

        row.addStretch()

        self.btn_zones.setStyleSheet(
            "background-color: #5a7ab0; color: white; font-weight: bold; padding: 0 12px;"
        )
        self.btn_save_all.setStyleSheet(
            "background-color: #5a9e5a; color: white; font-weight: bold; padding: 0 12px;"
        )
        self.btn_edit.setStyleSheet(
            "background-color: #b07a2a; color: white; font-weight: bold; padding: 0 12px;"
        )
        self.btn_translate_one.setStyleSheet(
            "background-color: #4a8eff; color: white; font-weight: bold; padding: 0 12px;"
        )
        self.btn_translate.setStyleSheet(
            "background-color: #2a5ecc; color: white; font-weight: bold; padding: 0 12px;"
        )
        for btn in (self.btn_zones, self.btn_save_all, self.btn_edit,
                    self.btn_translate_one, self.btn_translate):
            btn.setFixedHeight(32)
            row.addWidget(btn)

        # Start disabled; enabled in _on_row_changed as files are selected / translated
        self.btn_zones.setEnabled(False)
        self.btn_edit.setEnabled(False)
        self.btn_translate_one.setEnabled(False)

        # Wire up — use lambdas on buttons where QPushButton passes checked=bool as first arg
        self.btn_add_files.clicked.connect(self._add_files)
        self.btn_add_folder.clicked.connect(self._add_folder)
        self.btn_remove.clicked.connect(self._remove_selected)
        self.btn_clear.clicked.connect(self._clear_all)
        self.btn_zones.clicked.connect(lambda: self._open_zone_dialog())
        self.btn_save_all.clicked.connect(self._save_all)
        self.btn_edit.clicked.connect(lambda: self._open_edit_dialog())
        self.btn_translate_one.clicked.connect(lambda: self._start_translation_selected())
        self.btn_translate.clicked.connect(self._toggle_translation)

        return row

    def _build_left_panel(self):
        panel = QWidget()
        panel.setMaximumWidth(270)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 4, 0)
        layout.setSpacing(8)

        layout.addWidget(QLabel("Files:"))

        # The file list shows just the filename (not the full path)
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QListWidget.SingleSelection)
        self.list_widget.currentRowChanged.connect(self._on_row_changed)
        self.list_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list_widget.customContextMenuRequested.connect(self._show_file_context_menu)
        layout.addWidget(self.list_widget)

        layout.addWidget(self._build_settings_group())

        # Progress panel — hidden until a translation job starts
        self._progress_panel = QWidget()
        self._progress_panel.setVisible(False)
        prog_layout = QVBoxLayout(self._progress_panel)
        prog_layout.setContentsMargins(0, 4, 0, 2)
        prog_layout.setSpacing(2)

        self._progress_file_lbl = QLabel()
        self._progress_file_lbl.setStyleSheet("font-size: 11px; color: #ccc;")
        prog_layout.addWidget(self._progress_file_lbl)

        self._progress_step_lbl = QLabel()
        self._progress_step_lbl.setStyleSheet("font-size: 11px; color: #4a8eff;")
        prog_layout.addWidget(self._progress_step_lbl)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setTextVisible(False)
        prog_layout.addWidget(self.progress_bar)

        layout.addWidget(self._progress_panel)

        return panel

    def _build_settings_group(self):
        group = QGroupBox("Settings")
        g = QVBoxLayout(group)

        g.addWidget(QLabel("Source Language (what the comic is in):"))
        self.src_combo = QComboBox()
        self.src_combo.addItems(list(SOURCE_LANGUAGES.keys()))
        g.addWidget(self.src_combo)

        g.addWidget(QLabel("Target Language (translate to):"))
        self.tgt_combo = QComboBox()
        self.tgt_combo.addItems(list(TARGET_LANGUAGES.keys()))
        g.addWidget(self.tgt_combo)

        g.addWidget(QLabel("Translation Service:"))
        self.svc_combo = QComboBox()
        self.svc_combo.addItems(TRANSLATION_SERVICES)
        self.svc_combo.currentTextChanged.connect(self._on_service_changed)
        g.addWidget(self.svc_combo)

        # GGUF / Local LLM settings — only shown when "Local LLM (GGUF)" is selected
        self.gguf_hint = QLabel(
            "Requires <b>llama-cpp-python</b>.  "
            "CPU: <code>pip install llama-cpp-python</code><br>"
            "GPU: add <code>--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121</code>"
        )
        self.gguf_hint.setWordWrap(True)
        self.gguf_hint.setTextFormat(Qt.RichText)
        self.gguf_hint.setStyleSheet("color: #aaa; font-size: 10px;")

        self.gguf_manage_btn = QPushButton("Manage Models…")
        self.gguf_manage_btn.setStyleSheet(
            "background-color: #5a5a8a; color: white; font-weight: bold;"
        )
        self.gguf_manage_btn.setFixedHeight(28)
        self.gguf_manage_btn.clicked.connect(lambda: self._open_model_manager())

        self.gguf_models_label = QLabel("Models folder:")
        models_row = QHBoxLayout()
        self.gguf_models_edit = QLineEdit()
        self.gguf_models_edit.setPlaceholderText("AppData/Roaming/ComicTranslator/models")
        btn_browse_models = QPushButton("…")
        btn_browse_models.setFixedWidth(28)
        btn_browse_models.clicked.connect(self._browse_models_dir)
        models_row.addWidget(self.gguf_models_edit)
        models_row.addWidget(btn_browse_models)
        self._gguf_models_row_widget = QWidget()
        self._gguf_models_row_widget.setLayout(models_row)

        self.gguf_file_label = QLabel("GGUF model file:")
        file_row = QHBoxLayout()
        self.gguf_file_edit = QLineEdit()
        self.gguf_file_edit.setPlaceholderText("(none selected)")
        btn_browse_gguf = QPushButton("…")
        btn_browse_gguf.setFixedWidth(28)
        btn_browse_gguf.clicked.connect(self._browse_gguf)
        file_row.addWidget(self.gguf_file_edit)
        file_row.addWidget(btn_browse_gguf)
        self._gguf_file_row_widget = QWidget()
        self._gguf_file_row_widget.setLayout(file_row)

        self.gguf_gpu_label = QLabel("GPU layers (0 = CPU only):")
        self.gguf_gpu_spin = QSpinBox()
        self.gguf_gpu_spin.setRange(0, 999)
        self.gguf_gpu_spin.setValue(0)
        self.gguf_gpu_spin.setToolTip("Number of model layers to offload to GPU. 999 = fully on GPU.")

        for w in (self.gguf_hint, self.gguf_manage_btn,
                  self.gguf_models_label, self._gguf_models_row_widget,
                  self.gguf_file_label, self._gguf_file_row_widget,
                  self.gguf_gpu_label, self.gguf_gpu_spin):
            w.hide(); g.addWidget(w)

        g.addWidget(QLabel("Output Folder:"))
        out_row = QHBoxLayout()
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("(same folder as source, with _translated suffix)")
        self.btn_browse = QPushButton("…")
        self.btn_browse.setFixedWidth(28)
        self.btn_browse.clicked.connect(self._browse_output)
        out_row.addWidget(self.output_edit)
        out_row.addWidget(self.btn_browse)
        g.addLayout(out_row)

        return group

    def _build_right_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 0, 0, 0)

        zoom_hint = QLabel("Ctrl+scroll to zoom  •  Double-click to fit")
        zoom_hint.setStyleSheet("color: #555; font-size: 10px;")
        zoom_hint.setAlignment(Qt.AlignRight)
        layout.addWidget(zoom_hint)

        # Two tabs: original and translated side by side for easy comparison
        self.tabs = QTabWidget()

        self.viewer_orig  = ImageViewer("Select a file to preview")
        self.viewer_trans = ImageViewer("Not yet translated")

        # Wrap each viewer in a scroll area so images don't get clipped
        for viewer, label in ((self.viewer_orig, "Original"),
                               (self.viewer_trans, "Translated")):
            scroll = QScrollArea()
            scroll.setWidget(viewer)
            scroll.setWidgetResizable(True)
            self.tabs.addTab(scroll, label)

        layout.addWidget(self.tabs)
        return panel

    # ------------------------------------------------------------------ #
    # File management                                                      #
    # ------------------------------------------------------------------ #

    def _add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select comic pages", "",
            "Images (*.jpg *.jpeg *.png *.webp *.bmp *.tiff *.tif *.gif)"
        )
        added = 0
        for p in paths:
            if p not in self.file_list:
                self.file_list.append(p)
                self.list_widget.addItem(Path(p).name)
                added += 1
        if added:
            self.statusBar().showMessage(f"Added {added} file(s). Total: {len(self.file_list)}")

    def _add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select folder of comic pages")
        if not folder:
            return
        added = 0
        for ext in SUPPORTED_EXTENSIONS:
            # glob both lowercase and uppercase extensions (e.g. .JPG and .jpg)
            for pattern in (f"*{ext}", f"*{ext.upper()}"):
                for f in sorted(Path(folder).glob(pattern)):
                    if str(f) not in self.file_list:
                        self.file_list.append(str(f))
                        self.list_widget.addItem(f.name)
                        added += 1
        self.statusBar().showMessage(
            f"Added {added} file(s) from folder. Total: {len(self.file_list)}. "
            "Select one and click 'Translate Selected', or click 'Translate All'."
        )

    def _remove_selected(self):
        row = self.list_widget.currentRow()
        if row >= 0:
            self.list_widget.takeItem(row)
            path = self.file_list.pop(row)
            self.translated.pop(path, None)
            self.edit_data.pop(path, None)
            self.manual_zones.pop(path, None)

    def _clear_all(self):
        self.file_list.clear()
        self.list_widget.clear()
        self.translated.clear()
        self.edit_data.clear()
        self.manual_zones.clear()
        self.viewer_orig.clear()
        self.viewer_trans.clear()
        self.btn_zones.setEnabled(False)
        self.btn_edit.setEnabled(False)
        self.btn_translate_one.setEnabled(False)

    def _browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select output folder")
        if folder:
            self.output_edit.setText(folder)

    # ------------------------------------------------------------------ #
    # Preview update when user clicks a file in the list                  #
    # ------------------------------------------------------------------ #

    def _on_row_changed(self, row):
        running = self.worker and self.worker.isRunning()
        if 0 <= row < len(self.file_list):
            path = self.file_list[row]
            self.viewer_orig.set_image_path(path)
            self.btn_zones.setEnabled(True)
            self.btn_edit.setEnabled(path in self.edit_data)
            self.btn_translate_one.setEnabled(not running)

            if path in self.translated:
                self.viewer_trans.set_pil_image(self.translated[path])
                self.tabs.setCurrentIndex(1)
            else:
                self.viewer_trans.clear()
                self.tabs.setCurrentIndex(0)
        else:
            self.btn_zones.setEnabled(False)
            self.btn_edit.setEnabled(False)
            self.btn_translate_one.setEnabled(False)

    # ------------------------------------------------------------------ #
    # Settings panel helpers                                              #
    # ------------------------------------------------------------------ #

    def _apply_settings(self):
        cfg = self._cfg
        for combo, key in ((self.src_combo, "source_lang"),
                           (self.tgt_combo, "target_lang"),
                           (self.svc_combo, "service")):
            idx = combo.findText(cfg[key])
            if idx >= 0:
                combo.setCurrentIndex(idx)
        from app import settings as _s
        self.gguf_models_edit.setText(cfg.get("models_dir", str(_s.APPDATA_DIR / "models")))
        self.gguf_file_edit.setText(cfg.get("gguf_path", ""))
        self.gguf_gpu_spin.setValue(int(cfg.get("gguf_gpu_layers", 0)))
        self.output_edit.setText(cfg["output_dir"])

    def _collect_settings(self) -> dict:
        from app import settings as _s
        return {
            "source_lang":     self.src_combo.currentText(),
            "target_lang":     self.tgt_combo.currentText(),
            "service":         self.svc_combo.currentText(),
            "models_dir":      self.gguf_models_edit.text() or str(_s.APPDATA_DIR / "models"),
            "gguf_path":       self.gguf_file_edit.text(),
            "gguf_gpu_layers": self.gguf_gpu_spin.value(),
            "output_dir":      self.output_edit.text(),
        }

    def closeEvent(self, event):
        _settings.save(self._collect_settings())
        super().closeEvent(event)

    def _on_service_changed(self, text):
        show_gguf = (text == "Local LLM (GGUF)")
        for w in (self.gguf_hint, self.gguf_manage_btn,
                  self.gguf_models_label, self._gguf_models_row_widget,
                  self.gguf_file_label, self._gguf_file_row_widget,
                  self.gguf_gpu_label, self.gguf_gpu_spin):
            w.setVisible(show_gguf)

    def _open_model_manager(self):
        models_dir = self.gguf_models_edit.text().strip()
        if not models_dir:
            from app import settings as _s
            models_dir = str(_s.APPDATA_DIR / "models")
        dlg = ModelManagerDialog(self, models_dir=models_dir)
        if dlg.exec() == QDialog.Accepted and dlg.chosen_path:
            self.gguf_file_edit.setText(dlg.chosen_path)
            self.statusBar().showMessage(
                f"Model set: {Path(dlg.chosen_path).name}  — ready to translate."
            )

    def _browse_models_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Select models folder")
        if folder:
            self.gguf_models_edit.setText(folder)

    def _browse_gguf(self):
        start = self.gguf_models_edit.text().strip() or ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select GGUF model file", start, "GGUF models (*.gguf)"
        )
        if path:
            self.gguf_file_edit.setText(path)

    # ------------------------------------------------------------------ #
    # Translation control                                                  #
    # ------------------------------------------------------------------ #

    def _toggle_translation(self):
        if self.worker and self.worker.isRunning():
            self._stop_translation()
        else:
            self._start_translation()

    def _start_translation_selected(self):
        row = self.list_widget.currentRow()
        if 0 <= row < len(self.file_list):
            self._start_translation(files=[self.file_list[row]])

    def _show_file_context_menu(self, pos):
        row = self.list_widget.currentRow()
        if row < 0 or row >= len(self.file_list):
            return

        path    = self.file_list[row]
        running = self.worker and self.worker.isRunning()
        n_zones = len(self.manual_zones.get(path, []))

        menu = QMenu(self)
        act_translate = menu.addAction("Translate this file")
        act_translate.setEnabled(not running)
        zone_label = f"Mark zones… ({n_zones} saved)" if n_zones else "Mark zones…"
        act_zones = menu.addAction(zone_label)
        act_zones.setEnabled(not running)
        act_edit  = menu.addAction("Edit texts…")
        act_edit.setEnabled(path in self.edit_data)

        action = menu.exec(self.list_widget.mapToGlobal(pos))
        if action == act_translate:
            self._start_translation(files=[path])
        elif action == act_zones:
            self._open_zone_dialog(path)
        elif action == act_edit:
            self._open_edit_dialog(path)

    def _open_zone_dialog(self, path=None):
        if path is None:
            row = self.list_widget.currentRow()
            if row < 0 or row >= len(self.file_list):
                return
            path = self.file_list[row]

        existing = self.manual_zones.get(path, [])
        dlg = ZonePickDialog(self, path, existing)
        if dlg.exec() == QDialog.Accepted:
            zones = dlg.get_zones()
            if zones:
                self.manual_zones[path] = zones
                n = len(zones)
                reply = QMessageBox.question(
                    self, "Zones saved",
                    f"Saved {n} zone(s) for {Path(path).name}.\n\n"
                    "Re-translate this file now to apply them?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes,
                )
                if reply == QMessageBox.Yes:
                    self._start_translation(files=[path])
                else:
                    self.statusBar().showMessage(
                        f"Saved {n} zone(s) — click Translate Selected to apply them."
                    )
            else:
                self.manual_zones.pop(path, None)
                self.statusBar().showMessage("All zones cleared.")

    def _open_edit_dialog(self, path=None):
        if path is None:
            row = self.list_widget.currentRow()
            if row < 0 or row >= len(self.file_list):
                return
            path = self.file_list[row]

        if path not in self.edit_data:
            QMessageBox.information(self, "Not translated", "Translate this image first.")
            return

        ed = self.edit_data[path]
        dlg = TextEditDialog(self, path, ed['base'], ed['regions'], ed['target_lang'])
        if dlg.exec() == QDialog.Accepted and dlg.result_image is not None:
            self.translated[path] = dlg.result_image
            # Persist the edited regions so re-opening the dialog shows the correct state
            self.edit_data[path]['regions'] = [dict(r) for r in dlg._regions]
            # Refresh preview if this file is still selected
            row = self.list_widget.currentRow()
            if 0 <= row < len(self.file_list) and self.file_list[row] == path:
                self.viewer_trans.set_pil_image(dlg.result_image)
                self.tabs.setCurrentIndex(1)

    def _start_translation(self, files=None):
        # If no specific file list given, translate everything in the queue
        if files is None:
            files = self.file_list

        if not files:
            QMessageBox.warning(self, "No files", "Add at least one image file first.")
            return

        settings = {
            "source_lang":     SOURCE_LANGUAGES[self.src_combo.currentText()],
            "target_lang":     TARGET_LANGUAGES[self.tgt_combo.currentText()],
            "service":         self.svc_combo.currentText(),
            "gguf_path":       self.gguf_file_edit.text().strip(),
            "gguf_gpu_layers": self.gguf_gpu_spin.value(),
            "manual_zones":    self.manual_zones,
        }

        self._progress_panel.setVisible(True)
        self._progress_file_lbl.setText("Starting…")
        self._progress_step_lbl.setText("")
        self.progress_bar.setRange(0, len(files))
        self.progress_bar.setValue(0)

        self.btn_translate.setText("Stop")
        self.btn_translate_one.setEnabled(False)
        self.btn_add_files.setEnabled(False)
        self.btn_add_folder.setEnabled(False)

        self.worker = TranslationWorker(files, settings)
        self.worker.progress.connect(self._on_progress)
        self.worker.step.connect(self._on_step)
        self.worker.image_done.connect(self._on_image_done)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.fatal_error.connect(self._on_fatal_error)
        self.worker.start()

    def _stop_translation(self):
        if self.worker:
            self.worker.stop()
        # UI will reset via _on_finished when the thread exits

    # ------------------------------------------------------------------ #
    # Worker signal handlers                                               #
    # ------------------------------------------------------------------ #

    def _on_progress(self, count, message):
        self.progress_bar.setValue(count)
        self._progress_file_lbl.setText(message)
        self._progress_step_lbl.setText("")
        self.statusBar().showMessage(message)

    def _on_step(self, stage):
        self._progress_step_lbl.setText(f"  ↳  {stage}" if stage else "")

    def _on_image_done(self, path, payload):
        """Called each time a single image finishes — update preview if it's selected."""
        pil_image = payload['image']
        self.translated[path] = pil_image
        self.edit_data[path]  = {
            'base':        payload['base'],
            'regions':     payload['regions'],
            'target_lang': TARGET_LANGUAGES[self.tgt_combo.currentText()],
        }

        row = self.list_widget.currentRow()
        if 0 <= row < len(self.file_list) and self.file_list[row] == path:
            self.viewer_trans.set_pil_image(pil_image)
            self.tabs.setCurrentIndex(1)
        # Always enable Edit for this file now that it has been translated
        self.btn_edit.setEnabled(
            self.file_list[self.list_widget.currentRow()] == path
            if 0 <= self.list_widget.currentRow() < len(self.file_list) else False
        )

    def _on_finished(self):
        self.btn_translate.setText("Translate All")
        self.btn_add_files.setEnabled(True)
        self.btn_add_folder.setEnabled(True)
        self._progress_panel.setVisible(False)
        row = self.list_widget.currentRow()
        if 0 <= row < len(self.file_list):
            self.btn_translate_one.setEnabled(True)
        n = len(self.translated)
        self.statusBar().showMessage(f"Finished — {n} image(s) translated. Click 'Edit Texts' to adjust, then Save All.")

    def _on_error(self, msg):
        self.statusBar().showMessage(f"⚠  {msg}")

    def _on_fatal_error(self, msg):
        QMessageBox.critical(self, "Translation Error", msg)

    # ------------------------------------------------------------------ #
    # Saving                                                              #
    # ------------------------------------------------------------------ #

    def _save_all(self):
        if not self.translated:
            QMessageBox.warning(self, "Nothing to save", "Translate some images first.")
            return

        out_dir_text = self.output_edit.text().strip()
        saved = 0
        dest_dirs = set()

        for path, img in self.translated.items():
            p = Path(path)

            if out_dir_text:
                # User explicitly picked an output folder — use it for everything
                dest_dir = Path(out_dir_text)
            else:
                # Auto: create a sibling folder next to the source folder named
                # "{source_folder}_translated".  e.g. Comics/MangaXYZ/ → Comics/MangaXYZ_translated/
                dest_dir = p.parent.parent / f"{p.parent.name}_translated"

            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / p.name
            dest_dirs.add(str(dest_dir))

            # Save JPEG at high quality to avoid visible compression artefacts
            save_kwargs = {"quality": 95} if p.suffix.lower() in (".jpg", ".jpeg") else {}
            img.save(str(dest), **save_kwargs)
            saved += 1

        self.statusBar().showMessage(f"Saved {saved} image(s).")

        if len(dest_dirs) == 1:
            location_msg = list(dest_dirs)[0]
        else:
            location_msg = f"{len(dest_dirs)} output folders"

        QMessageBox.information(
            self, "Saved",
            f"Saved {saved} translated image(s) to:\n{location_msg}"
        )
