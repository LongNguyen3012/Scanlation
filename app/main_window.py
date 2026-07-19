from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
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
    "Auto-detect":                 "auto",
    "Japanese":                    "ja",
    "Japanese (Manga OCR)":        "ja_manga",
    "Korean":                      "ko",
    "Chinese Simplified":          "ch_sim",
    "Chinese Traditional":         "ch_tra",
    "English":                     "en",
}

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
    "NLLB (Offline)",
]

class ImageViewer(QLabel):
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
            event.ignore()

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

class RegionCanvas(QWidget):
    region_selected = Signal(int)
    regions_changed = Signal()

    _HANDLE = 8

    def __init__(self):
        super().__init__()
        self._pixmap   = None
        self._regions  = []
        self._selected = -1
        self._mode     = None
        self._drag_start   = None
        self._orig_bbox    = None
        self._resize_handle = None
        self._draw_end     = None
        self.setMouseTracking(True)
        self.setMinimumSize(300, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

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

    def _layout(self):
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
        xs = [p[0] for p in bbox]; ys = [p[1] for p in bbox]
        wx1, wy1 = self._to_widget(min(xs), min(ys))
        wx2, wy2 = self._to_widget(max(xs), max(ys))
        return wx1, wy1, wx2, wy2

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

        if self._mode == 'draw' and self._draw_end and self._drag_start:
            ix1, iy1 = self._to_image(*self._drag_start)
            ix2, iy2 = self._draw_end
            wx1, wy1 = self._to_widget(min(ix1, ix2), min(iy1, iy2))
            wx2, wy2 = self._to_widget(max(ix1, ix2), max(iy1, iy2))
            p.setPen(QPen(QColor(255, 80, 80), 2, Qt.DashLine))
            p.setBrush(QColor(255, 80, 80, 30))
            p.drawRect(int(wx1), int(wy1), int(wx2 - wx1), int(wy2 - wy1))

        p.end()

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        wx, wy = event.position().x(), event.position().y()

        if self._selected >= 0:
            h = self._hit_handle(wx, wy, self._selected)
            if h:
                self._mode          = 'resize'
                self._resize_handle = h
                self._drag_start    = (wx, wy)
                self._orig_bbox     = [list(pt) for pt in self._regions[self._selected]['bbox']]
                return

        for i in range(len(self._regions) - 1, -1, -1):
            if self._hit_region(wx, wy, i):
                self._selected  = i
                self._mode      = 'move'
                self._drag_start = (wx, wy)
                self._orig_bbox  = [list(pt) for pt in self._regions[i]['bbox']]
                self.region_selected.emit(i)
                self.update()
                return

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

class TextEditDialog(QDialog):
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

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(400)
        self._preview_timer.timeout.connect(self._update_preview)

        self._build_ui()
        self._refresh_table()
        self._update_preview()

    def _get_pipeline(self):
        if self._pipeline is None:
            from app.pipeline import ComicTranslationPipeline
            self._pipeline = ComicTranslationPipeline(
                source_lang="en",
                target_lang=self._target_lang,
                ocr_reader=None,
                translator=None,
            )
        return self._pipeline

    def _build_ui(self):
        main = QHBoxLayout(self)
        main.setSpacing(8)

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

        right = QVBoxLayout()
        right.addWidget(QLabel("Text regions – original text (read‑only) and translated text (editable):"))

        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["#", "Original Text", "Translated Text", "Size"])
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self._table.setColumnWidth(0, 32)
        self._table.setColumnWidth(3, 62)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.itemSelectionChanged.connect(self._on_table_select)
        self._table.itemChanged.connect(self._on_table_item_changed)

        self._table.setContextMenuPolicy(Qt.CustomContextMenu)
        self._table.customContextMenuRequested.connect(self._show_table_context_menu)

        right.addWidget(self._table, stretch=1)

        btn_del = QPushButton("Delete Selected Region")
        btn_del.clicked.connect(self._delete_selected)
        right.addWidget(btn_del)

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
        right_w.setMinimumWidth(420)
        right_w.setMaximumWidth(560)
        main.addWidget(right_w, stretch=2)

    def _show_table_context_menu(self, pos):
        row = self._table.currentRow()
        if row < 0:
            return
        menu = QMenu(self)
        act_copy_orig = menu.addAction("Copy Original Text")
        act_copy_trans = menu.addAction("Copy Translated Text")
        action = menu.exec(self._table.mapToGlobal(pos))
        if action == act_copy_orig:
            item = self._table.item(row, 1)
            if item:
                QApplication.clipboard().setText(item.text())
        elif action == act_copy_trans:
            item = self._table.item(row, 2)
            if item:
                QApplication.clipboard().setText(item.text())

    def _refresh_table(self):
        prev_sel = self._canvas.get_selected()
        self._table.blockSignals(True)
        self._table.setRowCount(len(self._regions))
        for i, r in enumerate(self._regions):
            num = QTableWidgetItem(str(i + 1))
            num.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self._table.setItem(i, 0, num)

            orig = QTableWidgetItem(r.get('original_text', ''))
            orig.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self._table.setItem(i, 1, orig)

            trans = QTableWidgetItem(r.get('text', ''))
            self._table.setItem(i, 2, trans)

            spin = QSpinBox()
            spin.setRange(6, 72)
            spin.setValue(r.get('font_size', 12))
            spin.valueChanged.connect(lambda v, idx=i: self._on_size_changed(idx, v))
            self._table.setCellWidget(i, 3, spin)
        self._table.blockSignals(False)
        self._canvas.set_regions(self._regions)
        if 0 <= prev_sel < len(self._regions):
            self._canvas.set_selected(prev_sel)
            self._table.blockSignals(True)
            self._table.selectRow(prev_sel)
            self._table.blockSignals(False)

    def _collect_from_table(self):
        for i in range(len(self._regions)):
            item_orig = self._table.item(i, 1)
            item_trans = self._table.item(i, 2)
            spin = self._table.cellWidget(i, 3)
            if item_orig is not None:
                self._regions[i]['original_text'] = item_orig.text()
            if item_trans is not None:
                self._regions[i]['text'] = item_trans.text()
            if spin is not None:
                self._regions[i]['font_size'] = spin.value()

    def _on_canvas_select(self, idx):
        if 0 <= idx < self._table.rowCount():
            self._table.selectRow(idx)

    def _on_table_select(self):
        rows = self._table.selectedItems()
        if rows:
            self._canvas.set_selected(self._table.currentRow())

    def _on_table_item_changed(self, item):
        if item.column() == 2 and 0 <= item.row() < len(self._regions):
            self._regions[item.row()]['text'] = item.text()
            self._preview_timer.start()

    def _on_size_changed(self, idx, value):
        if 0 <= idx < len(self._regions):
            self._regions[idx]['font_size'] = value
            self._preview_timer.start()

    def _on_regions_changed(self):
        self._collect_from_table()
        self._refresh_table()
        self._preview_timer.start()

    def _delete_selected(self):
        self._collect_from_table()
        self._canvas.delete_selected()
        self._refresh_table()

    def _apply_size_all(self):
        size = self._global_spin.value()
        for i in range(self._table.rowCount()):
            spin = self._table.cellWidget(i, 3)
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
class ZonePickDialog(QDialog):
    def __init__(self, parent, file_path, existing_zones=None):
        super().__init__(parent)
        self.setWindowTitle(f"Mark Zones — {Path(file_path).name}")
        self.setMinimumSize(700, 560)
        self.resize(960, 700)

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
        return [r['bbox'] for r in self._canvas.get_regions()]

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Comic Translator")
        self.setMinimumSize(900, 640)
        self.resize(1200, 800)

        self.file_list      = []
        self.translated     = {}
        self.edit_data      = {}
        self.manual_zones   = {}
        self.worker         = None
        self._cfg           = _settings.load()

        self._build_ui()
        self._apply_settings()
        self.statusBar().showMessage("Ready — add files or a folder to begin.")

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
        splitter.setStretchFactor(1, 1)
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

        self.btn_zones.setEnabled(False)
        self.btn_edit.setEnabled(False)
        self.btn_translate_one.setEnabled(False)

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

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QListWidget.SingleSelection)
        self.list_widget.currentRowChanged.connect(self._on_row_changed)
        self.list_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list_widget.customContextMenuRequested.connect(self._show_file_context_menu)
        layout.addWidget(self.list_widget)

        layout.addWidget(self._build_settings_group())

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
        g.addWidget(self.svc_combo)

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

        self.tabs = QTabWidget()

        self.viewer_orig  = ImageViewer("Select a file to preview")
        self.viewer_trans = ImageViewer("Not yet translated")

        for viewer, label in ((self.viewer_orig, "Original"),
                               (self.viewer_trans, "Translated")):
            scroll = QScrollArea()
            scroll.setWidget(viewer)
            scroll.setWidgetResizable(True)
            self.tabs.addTab(scroll, label)

        layout.addWidget(self.tabs)
        return panel

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

    def _apply_settings(self):
        cfg = self._cfg
        for combo, key in ((self.src_combo, "source_lang"),
                           (self.tgt_combo, "target_lang"),
                           (self.svc_combo, "service")):
            idx = combo.findText(cfg[key])
            if idx >= 0:
                combo.setCurrentIndex(idx)
        self.output_edit.setText(cfg["output_dir"])

    def _collect_settings(self) -> dict:
        return {
            "source_lang":     self.src_combo.currentText(),
            "target_lang":     self.tgt_combo.currentText(),
            "service":         self.svc_combo.currentText(),
            "output_dir":      self.output_edit.text(),
        }

    def closeEvent(self, event):
        _settings.save(self._collect_settings())
        super().closeEvent(event)

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
            self.edit_data[path]['regions'] = [dict(r) for r in dlg._regions]
            row = self.list_widget.currentRow()
            if 0 <= row < len(self.file_list) and self.file_list[row] == path:
                self.viewer_trans.set_pil_image(dlg.result_image)
                self.tabs.setCurrentIndex(1)

    def _start_translation(self, files=None):
        if files is None:
            files = self.file_list

        if not files:
            QMessageBox.warning(self, "No files", "Add at least one image file first.")
            return

        settings = {
            "source_lang":     SOURCE_LANGUAGES[self.src_combo.currentText()],
            "target_lang":     TARGET_LANGUAGES[self.tgt_combo.currentText()],
            "service":         self.svc_combo.currentText(),
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

    def _on_progress(self, count, message):
        self.progress_bar.setValue(count)
        self._progress_file_lbl.setText(message)
        self._progress_step_lbl.setText("")
        self.statusBar().showMessage(message)

    def _on_step(self, stage):
        self._progress_step_lbl.setText(f"  ↳  {stage}" if stage else "")

    def _on_image_done(self, path, payload):
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
                dest_dir = Path(out_dir_text)
            else:
                dest_dir = p.parent.parent / f"{p.parent.name}_translated"

            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / p.name
            dest_dirs.add(str(dest_dir))

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