import sys
import os
import csv
import numpy as np
import pandas as pd
from datetime import datetime

from PyQt6 import uic
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QLineEdit, QGraphicsDropShadowEffect,
    QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QComboBox,
    QCheckBox, QSplitter, QFrame, QToolButton, QMessageBox, QSizePolicy,
    QRadioButton, QButtonGroup, QProgressDialog, QGraphicsRectItem,
    QMenu 
)
from PyQt6.QtCore import Qt, QRectF, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QFontDatabase, QIcon, QPixmap, QAction 

import pyqtgraph as pg
from pyqtgraph import exporters
from scipy.signal import correlate, find_peaks
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean as scipy_euclidean_dist

from dtw import dtw
from fastdtw import fastdtw

# PyInstaller-compatible resource path helper
def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Set default PyQtGraph options
pg.setConfigOption('background', QColor(15, 15, 20))
pg.setConfigOption('foreground', QColor(220, 230, 255))

# Custom ViewBox class for selective zoom
class SelectiveZoomViewBox(pg.ViewBox):
    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        self.selectiveZoomEnabled = False
        self._selectionRectItem = None
        self._selectionStartPoint = None
        self._originalMouseEnabled = self.state['mouseEnabled']

    def toggleSelectiveZoom(self, enabled):
        self.selectiveZoomEnabled = enabled
        if enabled:
            self._originalMouseEnabled = self.state['mouseEnabled'] # Save current mouse state
            self.setMouseEnabled(x=False, y=False)  # Disable default pan/zoom behavior for selection
        else:
            # Restore default mouse interaction
            self.setMouseEnabled(x=True, y=True) # Re-enable general mouse interaction
            self.setMouseMode(pg.ViewBox.PanMode) # Explicitly set to PanMode (or your preferred default)
                                                  # Common modes: PanMode, RectMode
                                                  # Or restore original: self.setMouseEnabled(x=self._originalMouseEnabled[0], y=self._originalMouseEnabled[1])
                                                  # but explicitly setting mode is often more robust.

            if self._selectionRectItem:
                self.removeItem(self._selectionRectItem)
                self._selectionRectItem = None
            self._selectionStartPoint = None # Reset

    def mousePressEvent(self, ev):
        if self.selectiveZoomEnabled and ev.button() == Qt.MouseButton.LeftButton:
            ev.accept()
            self._selectionStartPoint = self.mapToView(ev.pos())

            if self._selectionRectItem is None:
                self._selectionRectItem = QGraphicsRectItem(QRectF())
                self._selectionRectItem.setPen(pg.mkPen(QColor(255, 255, 255, 150), width=1, style=Qt.PenStyle.DashLine))
                self._selectionRectItem.setBrush(QColor(255, 255, 255, 50))
                self.addItem(self._selectionRectItem, ignoreBounds=True)
            
            self._selectionRectItem.setRect(QRectF(self._selectionStartPoint, self._selectionStartPoint))
            self._selectionRectItem.setVisible(True)
        else:
            super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self.selectiveZoomEnabled and self._selectionRectItem and \
           self._selectionStartPoint and (ev.buttons() & Qt.MouseButton.LeftButton):
            ev.accept()
            currentPoint = self.mapToView(ev.pos())
            # Update selection rectangle
            selection_rect = QRectF(self._selectionStartPoint, currentPoint).normalized()
            self._selectionRectItem.setRect(selection_rect)
        else:
            super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if self.selectiveZoomEnabled and self._selectionRectItem and \
           self._selectionStartPoint and ev.button() == Qt.MouseButton.LeftButton:
            ev.accept()
            finalRect = self._selectionRectItem.rect()
            
            # Hide the selection rectangle (or remove and recreate on next press)
            self._selectionRectItem.setVisible(False) 
            # self.removeItem(self._selectionRectItem)
            # self._selectionRectItem = None
            
            self._selectionStartPoint = None # Reset start point

            if finalRect.width() > 1e-6 and finalRect.height() > 1e-6:  # Ensure valid zoom rect
                self.setRange(rect=finalRect, padding=0)  # setRange uses view coordinates
            
            # To make it a one-shot zoom and return to normal pan/zoom:
            # self.toggleSelectiveZoom(False) 
            # self.chk_selective_zoom.setChecked(False) # If you have access to the checkbox
        else:
            super().mouseReleaseEvent(ev)

class EnhancedPlotWidget(pg.PlotWidget):
    def __init__(self, parent=None, background=None, plotItem=None, **kargs):
        # Create the custom ViewBox first
        self.selectiveViewBox = SelectiveZoomViewBox()
        # If a plotItem is not provided, create one with the custom ViewBox
        if plotItem is None:
            plotItem = pg.PlotItem(viewBox=self.selectiveViewBox)
        super().__init__(parent=parent, background=background, plotItem=plotItem, **kargs)

    def toggleSelectiveZoom(self, enabled):
        if hasattr(self.plotItem, 'vb') and isinstance(self.plotItem.vb, SelectiveZoomViewBox):
             self.plotItem.vb.toggleSelectiveZoom(enabled)


class PulseAnalyzerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi(resource_path('main_window.ui'), self)

        self.reference_signal = None
        self.readings_signal = None
        self.time_readings = None
        self.reference_time = None
        self.sidebar_expanded = True
        self.results_data = None
        self.max_sample_size_for_dtw = 1000

        QFontDatabase.addApplicationFont(resource_path("fonts/Montserrat-Medium.ttf"))
        QFontDatabase.addApplicationFont(resource_path("fonts/Montserrat-Bold.ttf"))

        self._configure_ui_elements_and_layout() # Handles control creation and main layout
        self._setup_plots()
        self._apply_modern_style()
        self._connect_signals()

        self.lbl_status.setText("Status: Ready. Load data to begin.")

    def _configure_ui_elements_and_layout(self):
        # --- Main Layout is now handled by mainSplitter in the .ui file ---
        # self.ui_controls_container is self.controls_container from the .ui file
        # self.ui_plots_area_widget is self.plotsAreaWidget from the .ui file

        # Configure Splitter (from UI)
        # self.mainSplitter is the QSplitter from the UI file
        self.mainSplitter.setSizes([300, self.width() - 300 - self.mainSplitter.handleWidth()]) # Initial sizes
        self.mainSplitter.setStretchFactor(0, 0) # Controls panel doesn't stretch
        self.mainSplitter.setStretchFactor(1, 1) # Plots area stretches

        # The plotSplitter inside plotsAreaWidget can also be configured if needed:
        # self.plotSplitter.setSizes([self.plotsAreaWidget.height() // 2, self.plotsAreaWidget.height() // 2])

        # --- Populate/Configure Controls Panel (self.controls_container) ---
        # Widgets are now directly available via self.objectName from the .ui file

        # self.toggle_sidebar_btn (already in UI)
        # self.header_title (already in UI)
        # self.separator_frame (already in UI, ensure objectName matches)

        # Configure existing UI labels and inputs (names should match .ui file)
        self.label.setText("Reference Column Name (Optional):")
        self.label_2.setText("Readings Column Name (Optional):")
        # self.label_3 is txt_time_col_label in your code, ensure UI name matches
        self.txt_time_col_label = self.label_3 # If label_3 is its objectName in UI
        self.label_4.setText("Similarity Threshold (0.0-1.0):")
        self.dsp_threshold_label = self.label_4 # If label_4 is its objectName in UI

        # QComboBox dsp_threshold is already in the UI. Set current index if needed.
        self.dsp_threshold.setCurrentIndex(2) # e.g. "0.70"

        # self.chk_use_first_col (already in UI)
        # self.similarity_method_label (already in UI)
        # self.groupBox (already in UI)

        self.similarity_method_group = QButtonGroup(self)
        self.similarity_method_group.addButton(self.radio_cross_corr, 1)
        self.similarity_method_group.addButton(self.radio_cosine, 2)
        self.similarity_method_group.addButton(self.radio_dtw, 3)
        # self.radio_cross_corr.setChecked(True) (Can be set in UI or here)

        # Ensure main controls container has a maximum width (already set in UI but can be reinforced)
        self.controls_container.setMaximumWidth(350)
        self.controls_container.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self.plotsAreaWidget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Remove programmatic addition of widgets that are now in the .ui file
        # e.g., controls_panel_layout.insertWidget(...) for header, separator, etc.

    def _apply_modern_style(self):
        self.setStyleSheet("""
        QMainWindow {
            background: #0d0e12;
            font-family: 'Montserrat', sans-serif;
        }
        QLabel {
            color: #ffffff;
            font-family: 'Montserrat', sans-serif;
            font-size: 12px;
        }
        QLineEdit {
            border: 1px solid #3644a0;
            border-radius: 5px;
            padding: 5px;
            background-color: rgba(17, 18, 25, 200);
            color: #ffffff;
            font-family: 'Montserrat', sans-serif;
            font-size: 12px;
        }
        QPushButton {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #0b39c4, stop:1 #052e89);
            color: white;
            border: none;
            border-radius: 5px;
            padding: 8px 16px;
            font-family: 'Montserrat', sans-serif;
            font-weight: bold;
            font-size: 12px;
            min-height: 30px;
        }
        QPushButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #1b49d4, stop:1 #153e99);
        }
        QPushButton:pressed {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #052e89, stop:1 #052379);
        }
        QPushButton#toggle_sidebar_btn { 
            padding: 4px 8px; 
            font-size: 10px; 
        }
        QGroupBox {
            border: 1px solid #3644a0;
            border-radius: 8px;
            margin-top: 10px; /* IMPORTANT for title spacing */
            font-family: 'Montserrat', sans-serif;
            font-size: 13px; /* GroupBox title font size */
            font-weight: bold;
            color: #ffffff; /* GroupBox title color */
            background-color: rgba(17, 18, 25, 100); /* Background for GroupBox content area */
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top center;
            padding: 0 10px; /* Padding around the title text */
            /* background-color: transparent; /* Ensure title part doesn't get QGroupBox bg */
        }
        QDoubleSpinBox, QSpinBox, QComboBox {
            border: 1px solid #3644a0;
            border-radius: 5px;
            padding: 5px; /* This affects height */
            background-color: rgba(17, 18, 25, 200);
            color: #ffffff;
            font-family: 'Montserrat', sans-serif;
            min-height: 20px; /* Ensure a minimum height, adjust as needed */
        }
        QComboBox::drop-down {
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 15px;
            border-left-width: 1px;
            border-left-color: #3644a0;
            border-left-style: solid;
            border-top-right-radius: 5px;
            border-bottom-right-radius: 5px;
        }
        QStatusBar {
            background-color: rgba(10, 11, 15, 255);
            color: #ffffff;
            font-family: 'Montserrat', sans-serif;
        }
        QSlider::groove:horizontal {
            border: 1px solid #3644a0;
            height: 8px;
            background: rgba(17, 18, 25, 200);
            margin: 2px 0;
            border-radius: 4px;
        }
        QSlider::handle:horizontal {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #0b39c4, stop:1 #052e89);
            border: 1px solid #3644a0;
            width: 18px;
            margin: -2px 0;
            border-radius: 4px;
        }
        QToolButton { 
            background-color: #052e89;
            color: white;
            border: none;
            border-radius: 3px;
            font-size: 12px;
            padding: 2px;
        }
        QToolButton:hover {
            background-color: #153e99;
        }
        QSplitter::handle {
            background-color: #3644a0; 
            width: 3px; 
        }
        QSplitter::handle:horizontal {
            /* height: 10px; */ /* Not usually needed for vertical splitter handle */
        }
        QSplitter::handle:pressed {
            background-color: #4654b0;
        }
        QCheckBox, QRadioButton {
            color: #ffffff;
            font-family: 'Montserrat', sans-serif;
            font-size: 12px;
            spacing: 5px; /* Space between indicator and text */
            padding-top: 3px; /* Add some padding to align better if needed */
            padding-bottom: 3px;
        }
        QCheckBox::indicator, QRadioButton::indicator {
            width: 16px;
            height: 16px;
            border: 1px solid #3644a0;
            border-radius: 2px; /* Slightly less rounded */
            background-color: rgba(17, 18, 25, 200);
        }
        QCheckBox::indicator:checked, QRadioButton::indicator:checked {
            background-color: #0b39c4;
        }
        QProgressDialog {
            background-color: #0d0e12; 
            color: #ffffff;
            border: 1px solid #3644a0;
            border-radius: 5px;
        }
        QProgressDialog QLabel {
            color: #ffffff;
            font-family: 'Montserrat', sans-serif;
        }
        QProgressDialog QPushButton {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #0b39c4, stop:1 #052e89);
            color: white;
            border: none;
            border-radius: 5px;
            padding: 5px 10px;
             min-height: 25px; 
        }
        QProgressDialog QProgressBar {
            border: 1px solid #3644a0;
            border-radius: 3px;
            background-color: rgba(10, 11, 15, 200);
            text-align: center;
            color: white;
        }
        QProgressDialog QProgressBar::chunk {
            background-color: #0b39c4;
            width: 10px; 
            margin: 0.5px; 
        }
        """)
        for button in self.findChildren(QPushButton):
            if button.objectName() == "toggle_sidebar_btn": 
                button.setGraphicsEffect(None) 
                button.setStyleSheet("""
                    QPushButton#toggle_sidebar_btn {
                        background: #052e89; color: white; border: none; border-radius: 3px;
                        font-size: 12px; padding: 4px; min-height: 20px; max-width: 30px;
                    }
                    QPushButton#toggle_sidebar_btn:hover { background: #153e99; }
                    QPushButton#toggle_sidebar_btn:pressed { background: #031d59; }
                """)
            else:
                shadow = QGraphicsDropShadowEffect()
                shadow.setBlurRadius(20)
                shadow.setColor(QColor(11, 57, 196, 150)) 
                shadow.setOffset(0, 0)
                button.setGraphicsEffect(shadow)

        if hasattr(self, 'lbl_status') and self.lbl_status: 
            self.lbl_status.setStyleSheet("""
                color: #ffffff;
                background-color: rgba(10, 11, 15, 200);
                border-radius: 5px;
                padding: 5px;
                font-size: 11px;
            """)
        
        plot_placeholders = [self.plot_widget_placeholder_raw, self.plot_widget_placeholder_filtered]
        for widget in plot_placeholders:
            if widget: 
                widget.setStyleSheet("""
                    QWidget {
                        background-color: rgba(17, 18, 25, 0); 
                        border: 1px solid #3644a0;
                        border-radius: 8px;
                    }
                """)

    def _setup_plots(self):
        self.plot_widget_raw = EnhancedPlotWidget(name="RawPulses")
        # ... (set title, labels, legend, style as before) ...
        self._style_plot_widget(self.plot_widget_raw)
        # self.verticalLayout_7 is the layout of plot_widget_placeholder_raw from the .ui file
        self.verticalLayout_7.addWidget(self.plot_widget_raw)

        self.plot_widget_filtered = EnhancedPlotWidget(name="FilteredPulses")
        # ... (set title, labels, legend, style as before) ...
        self._style_plot_widget(self.plot_widget_filtered)
        # self.verticalLayout_8 is the layout of plot_widget_placeholder_filtered from the .ui file
        self.verticalLayout_8.addWidget(self.plot_widget_filtered)

        self.setup_plot_context_menus()

    def _style_plot_widget(self, plot_widget):
        plot_widget.setBackground(QColor(17, 18, 25, 220)) 
        plot_widget.getAxis('left').setPen(pg.mkPen(color='#3644a0', width=1))
        plot_widget.getAxis('left').setTextPen(pg.mkPen(color='#ffffff'))
        plot_widget.getAxis('left').setStyle(tickFont=QFont("Montserrat", 8))
        plot_widget.getAxis('bottom').setPen(pg.mkPen(color='#3644a0', width=1))
        plot_widget.getAxis('bottom').setTextPen(pg.mkPen(color='#ffffff'))
        plot_widget.getAxis('bottom').setStyle(tickFont=QFont("Montserrat", 8))
        plot_widget.showGrid(x=True, y=True, alpha=0.15) 
        legend = plot_widget.plotItem.legend
        if legend is not None:
            legend.setBrush(pg.mkBrush(QColor(17, 18, 25, 180))) 
            legend.setPen(pg.mkPen(color='#3644a0', width=1))
            for item in legend.items: 
                if isinstance(item, tuple) and len(item) > 1 and isinstance(item[1], pg.LabelItem):
                    item[1].setText(item[1].text, color='#dde6ff', size='9pt')

    def setup_plot_context_menus(self):
        """
        Sets up right-click context menus for the raw and filtered plot widgets.
        This allows users to export the plot images directly from the plot area.
        """
        # Action for exporting the raw plot
        export_action_raw = QAction("Export Raw Plot Image", self)
        # Connect the action to the export_plot_image method, passing the specific plot widget
        export_action_raw.triggered.connect(lambda: self.export_plot_image(self.plot_widget_raw))

        # Action for exporting the filtered plot
        export_action_filtered = QAction("Export Filtered Plot Image", self)
        # Connect the action to the export_plot_image method, passing the specific plot widget
        export_action_filtered.triggered.connect(lambda: self.export_plot_image(self.plot_widget_filtered))

        # Get the context menu for the raw plot and add the action
        # plotItem.getMenu() returns the QMenu object associated with the PlotItem
        raw_plot_menu = self.plot_widget_raw.plotItem.getMenu()
        if raw_plot_menu: # Ensure a menu exists before adding actions
            raw_plot_menu.addAction(export_action_raw)

        # Get the context menu for the filtered plot and add the action
        filtered_plot_menu = self.plot_widget_filtered.plotItem.getMenu()
        if filtered_plot_menu: # Ensure a menu exists before adding actions
            filtered_plot_menu.addAction(export_action_filtered)

        # Optional: If you want a more traditional QWidget context menu (right-click anywhere on the widget, not just the plot area)
        # self.plot_widget_raw.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        # self.plot_widget_raw.customContextMenuRequested.connect(self._show_raw_plot_context_menu)
        # self.plot_widget_filtered.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        # self.plot_widget_filtered.customContextMenuRequested.connect(self._show_filtered_plot_context_menu)

    # Optional: If using customContextMenuRequested, these helper methods would be needed:
    # def _show_raw_plot_context_menu(self, pos):
    #     menu = QMenu(self)
    #     export_action = QAction("Export Raw Plot Image", self)
    #     export_action.triggered.connect(lambda: self.export_plot_image(self.plot_widget_raw))
    #     menu.addAction(export_action)
    #     menu.exec(self.plot_widget_raw.mapToGlobal(pos))

    # def _show_filtered_plot_context_menu(self, pos):
    #     menu = QMenu(self)
    #     export_action = QAction("Export Filtered Plot Image", self)
    #     export_action.triggered.connect(lambda: self.export_plot_image(self.plot_widget_filtered))
    #     menu.addAction(export_action)
    #     menu.exec(self.plot_widget_filtered.mapToGlobal(pos))

    def _connect_signals(self):
        self.btn_load_ref.clicked.connect(self.load_reference_data)
        self.btn_load_readings.clicked.connect(self.load_readings_data)
        self.btn_analyze.clicked.connect(self.run_analysis)
        self.btn_export.clicked.connect(self.export_results)
        self.chk_selective_zoom.stateChanged.connect(self.toggle_selective_zoom_plots) 
        self.toggle_sidebar_btn.clicked.connect(self.toggle_sidebar)

    def toggle_sidebar(self):
        self.sidebar_expanded = not self.sidebar_expanded

        # These are child widgets of controls_container, managed by its layout (self.verticalLayout_3)
        # We will toggle their visibility, excluding the button itself.
        widgets_to_toggle_visibility = []
        # self.verticalLayout_3 should be the objectName of the QVBoxLayout within controls_container
        if hasattr(self, 'verticalLayout_3'): # Check if layout is loaded
            for i in range(self.verticalLayout_3.count()):
                item = self.verticalLayout_3.itemAt(i)
                if item:
                    widget = item.widget()
                    # Add to list if it's a widget and not the toggle button itself
                    if widget and widget != self.toggle_sidebar_btn:
                        widgets_to_toggle_visibility.append(widget)
        else:
            print("Error: verticalLayout_3 not found. Check .ui file and object names.")
            return


        if self.sidebar_expanded:  # Logic to EXPAND the sidebar
            # First, make all relevant child widgets visible
            for widget in widgets_to_toggle_visibility:
                widget.setVisible(True)
            
            # Explicitly show specific items like header and separator if they were hidden
            if hasattr(self, 'header_title'):
                self.header_title.setVisible(True)
            if hasattr(self, 'separator_frame'): # Name used in the provided .ui for the separator QFrame
                self.separator_frame.setVisible(True)

            # Then, set the maximum width for the expanded state
            self.controls_container.setMaximumWidth(350) # Corrected name
            self.toggle_sidebar_btn.setText("<<")

        else:  # Logic to COLLAPSE the sidebar
            # First, hide all relevant child widgets
            for widget in widgets_to_toggle_visibility:
                widget.setVisible(False)

            # Explicitly hide specific items
            if hasattr(self, 'header_title'):
                self.header_title.setVisible(False)
            if hasattr(self, 'separator_frame'):
                self.separator_frame.setVisible(False)

            # Then, set the maximum width for the collapsed state
            # Using a fixed value that matches or is slightly larger than the toggle button's width
            # The .ui file sets minimumWidth of controls_container to 50.
            collapsed_width = 50 # Or calculate dynamically if preferred:
            # button_width = self.toggle_sidebar_btn.sizeHint().width()
            # layout_margins = self.verticalLayout_3.contentsMargins()
            # collapsed_width = button_width + layout_margins.left() + layout_margins.right() + 10 # Example dynamic calculation
            
            self.controls_container.setMaximumWidth(collapsed_width) # Corrected name
            self.toggle_sidebar_btn.setText(">>")

        # Ensure the toggle button itself is always visible
        self.toggle_sidebar_btn.setVisible(True)
        
        # It might be necessary to explicitly tell the splitter to update its layout
        if hasattr(self, 'mainSplitter'):
            self.mainSplitter.refresh() # QSplitter doesn't have refresh(), sizes might need to be reset
            # A common way to force update is to slightly adjust sizes:
            # current_sizes = self.mainSplitter.sizes()
            # self.mainSplitter.setSizes(current_sizes) # This often triggers a refresh

    def toggle_selective_zoom_plots(self, state):
        is_checked = (state == Qt.CheckState.Checked.value) 
        if self.plot_widget_raw:
            self.plot_widget_raw.toggleSelectiveZoom(is_checked)
        if self.plot_widget_filtered:
            self.plot_widget_filtered.toggleSelectiveZoom(is_checked)

    def _get_file_data(self, file_path):
        try:
            file_size = os.path.getsize(file_path)
            show_progress = file_size > 5 * 1024 * 1024 
            progress = None
            if show_progress:
                progress = QProgressDialog("Loading file...", "Cancel", 0, 100, self)
                progress.setWindowTitle("Loading Data")
                progress.setWindowModality(Qt.WindowModality.WindowModal)
                progress.setValue(0)
                QApplication.processEvents() 
            df = None
            delimiters_to_try = [',', '\t', ' ', ';'] 
            def update_progress(val):
                if progress:
                    progress.setValue(val)
                    QApplication.processEvents()
                    if progress.wasCanceled():
                        raise InterruptedError("File loading cancelled by user.")
            if progress: update_progress(5)
            if file_path.endswith('.csv'):
                if progress: update_progress(30)
                df = pd.read_csv(file_path)
                if progress: update_progress(90)
            elif file_path.endswith(('.xlsx', '.xls')):
                if progress: update_progress(30)
                df = pd.read_excel(file_path)
                if progress: update_progress(90)
            elif file_path.endswith('.txt'):
                if progress: update_progress(10)
                for i, delimiter in enumerate(delimiters_to_try):
                    if progress:
                        update_progress(10 + int((i / len(delimiters_to_try)) * 60))
                    try:
                        df_try = pd.read_csv(file_path, delimiter=delimiter, on_bad_lines='warn')
                        if len(df_try.columns) > 1 or (len(df_try.columns) == 1 and df_try.shape[0] > 1) : 
                            df = df_try
                            break
                    except Exception:
                        continue
                if df is None: 
                    if progress: update_progress(70)
                    try: 
                        df = pd.read_csv(file_path, delim_whitespace=True, header=None)
                        if not (len(df.columns) > 1 or (len(df.columns) == 1 and df.shape[0] > 1)):
                            df = None 
                    except Exception:
                        pass
                    if df is None: 
                         df = pd.read_csv(file_path, header=None, names=['data'])
                if progress: update_progress(90)
            else: 
                if progress: update_progress(10)
                for i, delimiter in enumerate(delimiters_to_try):
                    if progress:
                        update_progress(10 + int((i / len(delimiters_to_try)) * 70))
                    try:
                        df_try = pd.read_csv(file_path, delimiter=delimiter, on_bad_lines='warn')
                        if len(df_try.columns) > 1 or (len(df_try.columns) == 1 and df_try.shape[0] > 1):
                            df = df_try
                            break
                    except Exception:
                        continue
                if df is None : 
                    if progress: update_progress(80)
                    try:
                        df = pd.read_csv(file_path, delim_whitespace=True, header=None)
                        if not (len(df.columns) > 1 or (len(df.columns) == 1 and df.shape[0] > 1)):
                            df = None
                    except Exception:
                        pass
                    if df is None:
                        df = pd.read_csv(file_path, header=None, names=['data'])
            if progress: update_progress(100)
            return df
        except InterruptedError:
            self.lbl_status.setText("Status: File loading cancelled.")
            if progress: progress.setValue(100) 
            return None
        except Exception as e:
            self.lbl_status.setText(f"Error loading data: {e}")
            if progress: progress.setValue(100) 
            QMessageBox.warning(self, "File Load Error", f"Could not load file {os.path.basename(file_path)}.\nError: {e}")
            return None
        finally:
            if 'progress' in locals() and progress is not None:
                progress.setValue(100)

    def _extract_signal_from_df(self, df, col_name, is_reference=False):
        try:
            if df is None: return None, None
            signal_col_found = None
            if col_name and col_name in df.columns:
                if pd.api.types.is_numeric_dtype(df[col_name]):
                    signal_col_found = col_name
                else:
                    self.lbl_status.setText(f"Warning: Column '{col_name}' is not numeric. Trying other columns.")
            if not signal_col_found and self.chk_use_first_col.isChecked():
                time_col_name_to_exclude = self.txt_time_col.text().strip()
                numeric_cols = df.select_dtypes(include=np.number).columns
                candidate_cols = [col for col in numeric_cols if not (time_col_name_to_exclude and col == time_col_name_to_exclude)]
                if candidate_cols:
                    signal_col_found = candidate_cols[0]
                    QMessageBox.information(self, "Column Selected", f"Using column '{signal_col_found}' for {'reference' if is_reference else 'readings'} signal.")
                    if is_reference: self.txt_ref_col.setText(str(signal_col_found))
                    else: self.txt_readings_col.setText(str(signal_col_found))
                else:
                    self.lbl_status.setText("Error: No suitable numeric column found.")
                    QMessageBox.critical(self, "Signal Extraction Error", "No suitable numeric column found in the data.")
                    return None, None
            elif not signal_col_found:
                self.lbl_status.setText(f"Error: Column '{col_name}' not found or not numeric, and 'Use First Column' is not checked or no suitable column found.")
                QMessageBox.critical(self, "Signal Extraction Error", f"Column '{col_name}' not found or not numeric.")
                return None, None
            if signal_col_found is None: 
                 self.lbl_status.setText(f"Error: Could not determine signal column.")
                 return None, None
            signal_data = df[signal_col_found].to_numpy(dtype=np.float64)
            if np.any(np.isnan(signal_data)) or np.any(np.isinf(signal_data)):
                num_nan_inf = np.sum(np.isnan(signal_data)) + np.sum(np.isinf(signal_data))
                self.lbl_status.setText(f"Warning: Found {num_nan_inf} NaN/inf values in '{signal_col_found}'. Replacing with 0.")
                finite_vals = signal_data[np.isfinite(signal_data)]
                fill_pos = np.max(finite_vals) if finite_vals.size > 0 else 0.0
                fill_neg = np.min(finite_vals) if finite_vals.size > 0 else 0.0
                signal_data = np.nan_to_num(signal_data, nan=0.0, posinf=fill_pos, neginf=fill_neg)
            time_data = None
            time_col_name = self.txt_time_col.text().strip()
            if time_col_name and time_col_name in df.columns:
                try:
                    time_data_series = pd.to_numeric(df[time_col_name], errors='coerce')
                    if time_data_series.isnull().any():
                        self.lbl_status.setText(f"Warning: Time column '{time_col_name}' has non-numeric values or parsing issues. Using sample indices for time.")
                        time_data = np.arange(len(signal_data))
                    else:
                        time_data = time_data_series.to_numpy(dtype=np.float64)
                        if len(time_data) != len(signal_data):
                            self.lbl_status.setText(f"Warning: Time column '{time_col_name}' length mismatch. Using sample indices.")
                            time_data = np.arange(len(signal_data))
                except Exception as e_time:
                    self.lbl_status.setText(f"Warning: Could not parse time column '{time_col_name}' ({e_time}). Using sample indices.")
                    time_data = np.arange(len(signal_data))
            else: 
                if time_col_name: 
                     self.lbl_status.setText(f"Info: Time column '{time_col_name}' not found. Using sample indices for time.")
                time_data = np.arange(len(signal_data))
            return signal_data, time_data
        except Exception as e:
            self.lbl_status.setText(f"Error extracting signal: {e}")
            QMessageBox.critical(self, "Signal Extraction Error", f"An error occurred while extracting data: {e}")
            return None, None

    def _load_data_from_file(self, data_type_label: str, specified_col_name_widget: QLineEdit):
        file_path, _ = QFileDialog.getOpenFileName(self, f"Load {data_type_label} Signal Data", "", "All Supported Files (*.csv *.txt *.xlsx *.xls *.dat);;CSV Files (*.csv);;Text Files (*.txt);;Excel Files (*.xlsx *.xls);;Data Files (*.dat);;All Files (*)")
        if not file_path: return None, None, None
        self.lbl_status.setText(f"Status: Loading {data_type_label} file: {os.path.basename(file_path)}...")
        QApplication.processEvents()
        df = self._get_file_data(file_path)
        if df is None:
            self.lbl_status.setText(f"Status: Failed to load {data_type_label} file or operation cancelled.")
            return None, None, None
        signal_col_name_input = specified_col_name_widget.text().strip()
        is_reference_file = (data_type_label == "Reference")
        signal_data, time_axis_data = self._extract_signal_from_df(df, signal_col_name_input, is_reference_file)
        if signal_data is None:
            self.lbl_status.setText(f"Status: Could not extract {data_type_label} signal from file.")
            return None, None, None
        return signal_data, file_path, time_axis_data

    def load_reference_data(self):
        signal, path, time_data = self._load_data_from_file("Reference", self.txt_ref_col)
        if signal is not None:
            self.reference_signal = signal
            self.reference_time = time_data if time_data is not None else np.arange(len(signal))
            self.lbl_ref_file.setText(f"Reference: {os.path.basename(path)} ({len(signal)} pts)")
            self.lbl_status.setText("Status: Reference signal loaded.")
            self.plot_widget_filtered.clear() 
            self.plot_widget_filtered.plot(self.reference_time, self.reference_signal, pen=pg.mkPen(color='#fc466b', width=2), name="Reference Signal")
            self.plot_widget_filtered.setTitle("Reference Signal Preview")
            self.plot_widget_filtered.autoRange()
            self.plot_widget_raw.clear()
            self.plot_widget_raw.setTitle("Load Readings Signal or Analyze")
        else:
            self.reference_signal = None; self.reference_time = None
            self.lbl_ref_file.setText("Reference: Not loaded")

    def load_readings_data(self):
        signal, path, time_data = self._load_data_from_file("Readings", self.txt_readings_col)
        if signal is not None:
            self.readings_signal = signal
            self.time_readings = time_data if time_data is not None else np.arange(len(signal))
            self.lbl_readings_file.setText(f"Readings: {os.path.basename(path)} ({len(signal)} pts)")
            self.lbl_status.setText("Status: Readings signal loaded.")
            self.plot_widget_raw.clear() 
            self.plot_widget_raw.plot(self.time_readings, self.readings_signal, pen=pg.mkPen(color='#3644a0', width=1.5), name="Full Readings Signal")
            self.plot_widget_raw.setTitle("Full Readings Signal (Awaiting Analysis)")
            self.plot_widget_raw.autoRange()
            if self.reference_signal is None:
                self.plot_widget_filtered.clear()
                self.plot_widget_filtered.setTitle("Load Reference Signal or Analyze")
        else:
            self.readings_signal = None; self.time_readings = None
            self.lbl_readings_file.setText("Readings: Not loaded")

    def _normalized_cross_correlation(self, signal, template):
        if len(template) == 0 or len(signal) == 0 or len(template) > len(signal): return np.array([])
        template_zm = template - np.mean(template)
        template_std = np.std(template)
        if np.isclose(template_std, 0): return np.zeros(len(signal) - len(template) + 1)
        signal_len = len(signal); template_len = len(template)
        ncc_scores = np.zeros(signal_len - template_len + 1)
        for i in range(signal_len - template_len + 1):
            window = signal[i: i + template_len]
            window_zm = window - np.mean(window)
            window_std = np.std(window)
            if np.isclose(window_std, 0):
                ncc_scores[i] = 0.0; continue
            corr = np.sum(window_zm * template_zm)
            norm_factor = template_len * window_std * template_std 
            if np.isclose(norm_factor, 0): ncc_scores[i] = 0.0
            else: ncc_scores[i] = corr / norm_factor
        ncc_scores = np.clip(ncc_scores, -1.0, 1.0)
        return np.nan_to_num(ncc_scores, nan=0.0, posinf=0.0, neginf=0.0)

    def _calculate_cosine_similarity(self, signal, template):
        if len(template) == 0 or len(signal) == 0 or len(template) > len(signal): return np.array([])
        signal_len = len(signal); template_len = len(template)
        similarity_scores = np.zeros(signal_len - template_len + 1)
        template_reshaped = template.reshape(1, -1)
        for i in range(signal_len - template_len + 1):
            window = signal[i:i + template_len].reshape(1, -1)
            similarity = cosine_similarity(window, template_reshaped)[0][0]
            similarity_scores[i] = similarity
        return np.nan_to_num(similarity_scores, nan=0.0, posinf=0.0, neginf=0.0)

    def _calculate_dtw_similarity(self, signal, template, progress_dialog=None):
        if len(template) == 0 or len(signal) == 0 or len(template) > len(signal): return np.array([])
        use_fastdtw = False
        if len(template) > self.max_sample_size_for_dtw or len(signal) > self.max_sample_size_for_dtw * 10:
            use_fastdtw = True
            self.lbl_status.setText("Info: Using FastDTW due to signal size.")
            QApplication.processEvents()
        signal_len = len(signal); template_len = len(template)
        similarity_scores = np.zeros(signal_len - template_len + 1)
        max_possible_dist_heuristic = template_len 
        if np.max(np.abs(template)) > 1e-9 : 
             _temp_max_abs = np.max(np.abs(template))
             max_possible_dist_heuristic = scipy_euclidean_dist(np.zeros_like(template), np.full_like(template, _temp_max_abs))
             if np.isclose(max_possible_dist_heuristic,0): max_possible_dist_heuristic = template_len 
        element_dist_func = lambda x, y: abs(x - y)
        num_windows = signal_len - template_len + 1
        for i in range(num_windows):
            if progress_dialog and i % (max(1, num_windows // 100)) == 0 : 
                progress_dialog.setValue(int((i / num_windows) * 100))
                QApplication.processEvents()
                if progress_dialog.wasCanceled(): return np.array([]) 
            window = signal[i:i + template_len]; distance = -1.0
            try:
                if use_fastdtw:
                    dist_val, path = fastdtw(template, window, dist=element_dist_func)
                    distance = float(dist_val)
                else:
                    dtw_result = dtw(template, window, dist_method=element_dist_func, keep_internals=False)
                    distance = dtw_result.distance
            except Exception as e_dtw:
                similarity_scores[i] = 0; continue 
            if np.isclose(max_possible_dist_heuristic, 0):
                similarity_scores[i] = 1.0 if np.isclose(distance, 0) else 0.0
            else:
                normalized_distance = min(distance / max_possible_dist_heuristic, 1.0)
                similarity = 1.0 - normalized_distance
                similarity_scores[i] = max(0, similarity) 
        if progress_dialog: progress_dialog.setValue(100)
        return np.nan_to_num(similarity_scores, nan=0.0, posinf=0.0, neginf=0.0)

    def run_analysis(self):
        self.lbl_status.setText("Status: Starting analysis...")
        QApplication.processEvents()
        try:
            if self.reference_signal is None or self.readings_signal is None:
                QMessageBox.warning(self, "Analysis Error", "Load both reference and readings signals first."); self.lbl_status.setText("Error: Load both signals."); return
            if len(self.reference_signal) == 0:
                QMessageBox.warning(self, "Analysis Error", "Reference signal is empty."); self.lbl_status.setText("Error: Reference signal empty."); return
            if len(self.reference_signal) > len(self.readings_signal):
                QMessageBox.warning(self, "Analysis Error", "Reference signal is longer than readings signal."); self.lbl_status.setText("Error: Reference longer than readings."); return
            self.lbl_status.setText("Status: Analyzing... Please wait."); QApplication.processEvents()
            progress = QProgressDialog("Analyzing signal...", "Cancel", 0, 100, self)
            progress.setWindowTitle("Analysis in Progress"); progress.setWindowModality(Qt.WindowModality.WindowModal); progress.setValue(0); QApplication.processEvents()
            ref_sig = self.reference_signal; read_sig = self.readings_signal; template_len = len(ref_sig)
            try:
                threshold_text = self.dsp_threshold.currentText().replace(",", ".")
                threshold = float(threshold_text)
                if not (0.0 <= threshold <= 1.0): raise ValueError("Threshold out of range [0.0, 1.0]")
            except ValueError:
                QMessageBox.warning(self, "Input Error", f"Invalid threshold value: '{self.dsp_threshold.currentText()}'. Using default 0.7."); threshold = 0.7
                idx = self.dsp_threshold.findText("0.70"); 
                if idx != -1: self.dsp_threshold.setCurrentIndex(idx)
                else: self.dsp_threshold.setCurrentText("0.70")
            if progress.wasCanceled(): self.lbl_status.setText("Status: Analysis cancelled."); progress.setValue(100); return
            progress.setValue(10); QApplication.processEvents()
            similarity_scores = np.array([]); method_name = ""
            if self.radio_cross_corr.isChecked(): method_name = "Normalized Cross Correlation"; similarity_scores = self._normalized_cross_correlation(read_sig, ref_sig)
            elif self.radio_cosine.isChecked(): method_name = "Cosine Similarity"; similarity_scores = self._calculate_cosine_similarity(read_sig, ref_sig)
            elif self.radio_dtw.isChecked():
                method_name = "Dynamic Time Warping (Similarity)"; self.lbl_status.setText(f"Status: Calculating DTW similarity..."); QApplication.processEvents()
                similarity_scores = self._calculate_dtw_similarity(read_sig, ref_sig, progress) 
            else: self.lbl_status.setText("Error: No similarity method selected."); progress.setValue(100); return
            if progress.wasCanceled(): self.lbl_status.setText("Status: Analysis cancelled."); progress.setValue(100); return
            if not self.radio_dtw.isChecked(): progress.setValue(70); QApplication.processEvents()
            if not isinstance(similarity_scores, np.ndarray) or similarity_scores.size == 0:
                self.lbl_status.setText(f"Status: Similarity calculation returned no valid scores using {method_name}.")
                QMessageBox.information(self, "Analysis Info", f"No similarity scores generated using {method_name}.")
                self.plot_widget_raw.clear(); self.plot_widget_filtered.clear(); self.plot_widget_raw.setTitle(f"No Scores ({method_name})"); self.plot_widget_filtered.setTitle(f"No Scores ({method_name})")
                progress.setValue(100); return
            peak_distance = max(1, int(template_len * 0.75)) 
            peaks_indices, properties = find_peaks(similarity_scores, height=threshold, distance=peak_distance)
            if progress.wasCanceled(): self.lbl_status.setText("Status: Analysis cancelled."); progress.setValue(100); return
            progress.setValue(80); QApplication.processEvents()
            if not peaks_indices.any(): 
                self.lbl_status.setText(f"Status: No pulses found above threshold {threshold:.2f} using {method_name}.")
                QMessageBox.information(self, "Analysis Results", f"No pulses detected with {method_name} above threshold {threshold:.2f}.")
                self.plot_widget_raw.clear(); self.plot_widget_filtered.clear()
                current_time_axis_raw = self.time_readings if self.time_readings is not None else np.arange(len(self.readings_signal))
                self.plot_widget_raw.plot(current_time_axis_raw, self.readings_signal, pen=pg.mkPen(color=QColor(180, 180, 230, 60), width=1), name="Original Signal")
                self.plot_widget_raw.setTitle(f"No Pulses Found ({method_name}, Thr: {threshold:.2f})")
                self.plot_widget_filtered.setTitle(f"No Pulses Found ({method_name}, Thr: {threshold:.2f})")
                self.btn_export.setEnabled(False); self.results_data = None; progress.setValue(100); return
            self.lbl_status.setText(f"Status: Found {len(peaks_indices)} potential pulses. Plotting..."); QApplication.processEvents()
            self.plot_widget_raw.clear(); self.plot_widget_raw.setTitle(f"Detected Pulses (Raw) - {len(peaks_indices)} ({method_name})")
            current_time_axis = self.time_readings if self.time_readings is not None else np.arange(len(self.readings_signal))
            self.plot_widget_raw.setLabel('bottom', 'Sample Index / Time' if self.time_readings is not None else 'Sample Index')
            self.plot_widget_raw.addLegend(clear=True) 
            self.plot_widget_filtered.clear(); self.plot_widget_filtered.setTitle(f"Detected Pulses (Filtered) - {len(peaks_indices)} ({method_name})")
            self.plot_widget_filtered.setLabel('bottom', 'Sample Index / Time' if self.time_readings is not None else 'Sample Index')
            self.plot_widget_filtered.addLegend(clear=True)
            self.plot_widget_raw.plot(current_time_axis, self.readings_signal, pen=pg.mkPen(color=QColor(180, 180, 230, 80), width=1), name="Original Signal")
            plot_data_noise_zero = np.zeros_like(self.readings_signal, dtype=float); detected_pulse_details = []
            pulse_colors = [QColor('#fc466b'), QColor('#0b39c4'), QColor('#00a2ff'), QColor('#50fa7b'), QColor('#f1fa8c'), QColor('#ff79c6'), QColor('#bd93f9'), QColor('#ffb86c'), QColor('#8be9fd'), QColor('#ff5555')]
            for i, peak_start_idx_in_scores in enumerate(peaks_indices):
                if progress.wasCanceled(): self.lbl_status.setText("Status: Plotting cancelled."); break
                progress.setValue(80 + int((i / len(peaks_indices)) * 19)); QApplication.processEvents()
                pulse_start_in_readings = peak_start_idx_in_scores ; pulse_end_in_readings = pulse_start_in_readings + template_len
                if pulse_end_in_readings > len(self.readings_signal): continue
                pulse_segment_data = self.readings_signal[pulse_start_in_readings:pulse_end_in_readings]
                pulse_segment_time = current_time_axis[pulse_start_in_readings:pulse_end_in_readings]
                color = pulse_colors[i % len(pulse_colors)] 
                self.plot_widget_raw.plot(pulse_segment_time, pulse_segment_data, pen=pg.mkPen(color=color, width=2.5), name=f"Pulse {i+1} (Score: {properties['peak_heights'][i]:.3f})")
                plot_data_noise_zero[pulse_start_in_readings:pulse_end_in_readings] = pulse_segment_data
                detected_pulse_details.append({"id": i + 1, "start_idx_readings": pulse_start_in_readings, "end_idx_readings": pulse_end_in_readings -1, "start_time": current_time_axis[pulse_start_in_readings], "end_time": current_time_axis[pulse_end_in_readings - 1], "similarity_score": properties['peak_heights'][i]})
            self.plot_widget_filtered.plot(current_time_axis, plot_data_noise_zero, pen=pg.mkPen(color='#0b39c4', width=2), name="Combined Filtered Pulses")
            self.plot_widget_raw.autoRange(); self.plot_widget_filtered.autoRange()
            self.btn_export.setEnabled(True)
            self.results_data = {"detected_pulses": detected_pulse_details, "method": method_name, "threshold": threshold, "reference_length": len(self.reference_signal), "total_pulses": len(peaks_indices), "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            self.lbl_status.setText(f"Status: Analysis complete. Found {len(peaks_indices)} pulses using {method_name}.")
            progress.setValue(100); QApplication.processEvents()
        except Exception as e:
            self.lbl_status.setText(f"Error during analysis: {str(e)}")
            if 'progress' in locals() and progress is not None and not progress.wasCanceled(): progress.setValue(100)
            QMessageBox.critical(self, "Analysis Error", f"An unexpected error occurred: {str(e)}")
            import traceback; traceback.print_exc()
        finally:
             if 'progress' in locals() and progress is not None: progress.setValue(100)

    def export_results(self):
        if self.results_data is None or not self.results_data.get("detected_pulses"):
            QMessageBox.information(self, "Export Error", "No analysis results to export or no pulses were detected."); self.lbl_status.setText("Error: No results to export."); return
        default_filename = f"pulse_analysis_results_{self.results_data.get('method','').replace(' ','_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        file_path, selected_filter = QFileDialog.getSaveFileName(self, "Export Analysis Results", default_filename, "CSV Files (*.csv);;All Files (*)")
        if not file_path: return 
        if selected_filter.startswith("CSV Files") and not file_path.lower().endswith('.csv'): file_path += '.csv'
        try:
            pulse_data = self.results_data["detected_pulses"]
            with open(file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Pulse Analysis Report"]); writer.writerow(["Generated on", self.results_data.get("timestamp", "")]); writer.writerow([]) 
                writer.writerow(["Analysis Parameters:"]); writer.writerow(["Method Used", self.results_data.get("method", "N/A")])
                writer.writerow(["Similarity Threshold", f"{self.results_data.get('threshold', 0.0):.3f}"])
                writer.writerow(["Reference Signal Length (samples)", self.results_data.get("reference_length", 0)])
                writer.writerow(["Total Pulses Found", self.results_data.get("total_pulses", 0)]); writer.writerow([]) 
                header = ["Pulse ID", "Start Index (Readings)", "End Index (Readings)", "Start Time", "End Time", "Similarity Score"]
                writer.writerow(header)
                for pulse in pulse_data:
                    writer.writerow([pulse["id"], pulse["start_idx_readings"], pulse["end_idx_readings"],
                                     f"{pulse['start_time']:.4f}" if isinstance(pulse['start_time'], (float, np.float64)) else pulse['start_time'], # NEW
                                     f"{pulse['end_time']:.4f}" if isinstance(pulse['end_time'], (float, np.float64)) else pulse['end_time'],       # NEW
                                     f"{pulse['similarity_score']:.4f}"])
            self.lbl_status.setText(f"Status: Results exported to {os.path.basename(file_path)}")
            QMessageBox.information(self, "Export Successful", f"Results exported to:\n{file_path}")
        except Exception as e:
            self.lbl_status.setText(f"Error exporting results: {str(e)}")
            QMessageBox.critical(self, "Export Error", f"Could not export results to CSV: {str(e)}")

    def export_plot_image(self, plot_widget):
        if not plot_widget or not plot_widget.plotItem or not plot_widget.plotItem.items:
             QMessageBox.warning(self, "Export Plot Error", "Cannot export an empty plot or plot with no data items."); return
        plot_name_for_file = plot_widget.name() if plot_widget.name() else "plot"
        default_filename = f"{plot_name_for_file}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        file_path, selected_filter = QFileDialog.getSaveFileName(self, "Export Plot Image", default_filename, "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;SVG Files (*.svg);;All Files (*)")
        if not file_path: return 
        file_ext_lower = os.path.splitext(file_path)[1].lower()
        if "(*.png)" in selected_filter and file_ext_lower != '.png': file_path += '.png'
        elif ("(*.jpg *.jpeg)" in selected_filter) and file_ext_lower not in ['.jpg', '.jpeg']: file_path += '.jpg'
        elif "(*.svg)" in selected_filter and file_ext_lower != '.svg': file_path += '.svg'
        elif not any(file_ext_lower == ext for ext in ['.png', '.jpg', '.jpeg', '.svg']): file_path += '.png'
        try:
            actual_file_ext = os.path.splitext(file_path)[1].lower()
            if actual_file_ext == '.svg': exporter = exporters.SVGExporter(plot_widget.plotItem)
            else: 
                exporter = exporters.ImageExporter(plot_widget.plotItem)
                exporter.parameters()['width'] = 1920; exporter.parameters()['height'] = 1080
            exporter.export(file_path)
            self.lbl_status.setText(f"Status: Plot exported to {os.path.basename(file_path)}")
            QMessageBox.information(self, "Export Successful", f"Plot image exported to:\n{file_path}")
        except Exception as e:
            self.lbl_status.setText(f"Error exporting plot image: {str(e)}")
            QMessageBox.critical(self, "Export Plot Error", f"Could not export plot image: {str(e)}\nCheck if required export libraries are installed (e.g., for SVG).")
            import traceback; traceback.print_exc()

# Main execution
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PulseAnalyzerApp()
    window.show()
    sys.exit(app.exec())
