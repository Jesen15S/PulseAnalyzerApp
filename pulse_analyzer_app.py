import sys
import numpy as np
import pandas as pd
from PyQt6 import uic
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QLineEdit, QGraphicsDropShadowEffect, QPushButton, QLabel
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor, QFontDatabase, QLinearGradient, QRadialGradient, QBrush, QPainter, QPen
import pyqtgraph as pg
from scipy.signal import correlate, find_peaks

# Set default PyQtGraph options
pg.setConfigOption('background', QColor(15, 15, 20))  # Almost black background
pg.setConfigOption('foreground', QColor(220, 230, 255))  # Light blue-white text

class PulseAnalyzerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi('main_window.ui', self)
        
        # Load custom font
        QFontDatabase.addApplicationFont("fonts/Montserrat-Medium.ttf")
        QFontDatabase.addApplicationFont("fonts/Montserrat-Bold.ttf")
        
        # Apply modern styling to the application
        self._apply_modern_style()
        
        self.reference_signal = None
        self.readings_signal = None
        self.time_readings = None

        self._setup_plots()
        self._connect_signals()
        self.lbl_status.setText("Status: Ready. Load data to begin.")

    def _apply_modern_style(self):
        # Main window with very dark background
        self.setStyleSheet("""
            QMainWindow {
                background: #0d0e12;  /* Very dark, almost black background */
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
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                           stop:0 #0b39c4, stop:1 #052e89);
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
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                           stop:0 #1b49d4, stop:1 #153e99);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                           stop:0 #052e89, stop:1 #052379);
            }
            QGroupBox {
                border: 1px solid #3644a0;
                border-radius: 8px;
                margin-top: 10px;
                font-family: 'Montserrat', sans-serif;
                font-size: 13px;
                font-weight: bold;
                color: #ffffff;
                background-color: rgba(17, 18, 25, 100);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 10px;
            }
            QDoubleSpinBox, QSpinBox {
                border: 1px solid #3644a0;
                border-radius: 5px;
                padding: 5px;
                background-color: rgba(17, 18, 25, 200);
                color: #ffffff;
                font-family: 'Montserrat', sans-serif;
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
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                           stop:0 #0b39c4, stop:1 #052e89);
                border: 1px solid #3644a0;
                width: 18px;
                margin: -2px 0;
                border-radius: 4px;
            }
        """)
        
        # Apply more intense shadow effects to buttons for that glow effect
        for button in self.findChildren(QPushButton):
            shadow = QGraphicsDropShadowEffect()
            shadow.setBlurRadius(20)
            shadow.setColor(QColor(11, 57, 196, 150))  # Matching blue glow
            shadow.setOffset(0, 0)
            button.setGraphicsEffect(shadow)
            
            # Force button to match reference image style
            button.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                               stop:0 #0b39c4, stop:1 #052e89);
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
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                               stop:0 #1b49d4, stop:1 #153e99);
                }
                QPushButton:pressed {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                               stop:0 #052e89, stop:1 #052379);
                }
            """)
        
        # Set title font
        title_font = QFont("Montserrat", 14, QFont.Weight.Bold)
        if hasattr(self, 'lbl_title'):
            self.lbl_title.setFont(title_font)
        
        # Enhance the status label
        if hasattr(self, 'lbl_status'):
            self.lbl_status.setStyleSheet("""
                color: #ffffff;
                background-color: rgba(10, 11, 15, 200);
                border-radius: 5px;
                padding: 5px;
                font-size: 11px;
            """)
        
        # Make input fields match the reference image style
        for line_edit in self.findChildren(QLineEdit):
            line_edit.setStyleSheet("""
                border: 1px solid #3644a0;
                border-radius: 5px;
                padding: 5px;
                background-color: rgba(17, 18, 25, 200);
                color: #ffffff;
                font-family: 'Montserrat', sans-serif;
                font-size: 12px;
            """)
        
        # Apply glass effect to plot placeholders
        for widget in [self.plot_widget_placeholder_raw, self.plot_widget_placeholder_filtered]:
            widget.setStyleSheet("""
                background-color: rgba(17, 18, 25, 200);
                border: 1px solid #3644a0;
                border-radius: 8px;
            """)
            shadow = QGraphicsDropShadowEffect()
            shadow.setBlurRadius(15)
            shadow.setColor(QColor(11, 57, 196, 100))  # Matching blue glow
            shadow.setOffset(0, 0)
            widget.setGraphicsEffect(shadow)

    def _setup_plots(self):
        # Create and style the raw plot widget
        self.plot_widget_raw = pg.PlotWidget(name="RawPulses")
        self.plot_widget_raw.setTitle("Detected Pulses (Raw)")
        self.plot_widget_raw.setLabel('left', 'Amplitude')
        self.plot_widget_raw.setLabel('bottom', 'Sample Index / Time')
        self.plot_widget_raw.addLegend(clear=True)
        
        # Style the raw plot
        self._style_plot_widget(self.plot_widget_raw)
        self.plot_widget_placeholder_raw.layout().addWidget(self.plot_widget_raw)

        # Create and style the filtered plot widget
        self.plot_widget_filtered = pg.PlotWidget(name="FilteredPulses")
        self.plot_widget_filtered.setTitle("Detected Pulses (Noise Zeroed)")
        self.plot_widget_filtered.setLabel('left', 'Amplitude')
        self.plot_widget_filtered.setLabel('bottom', 'Sample Index / Time')
        self.plot_widget_filtered.addLegend(clear=True)
        
        # Style the filtered plot
        self._style_plot_widget(self.plot_widget_filtered)
        self.plot_widget_placeholder_filtered.layout().addWidget(self.plot_widget_filtered)

    def _style_plot_widget(self, plot_widget):
        # Apply glass-like styling to match reference image
        plot_widget.setBackground(QColor(17, 18, 25, 200))
        
        # Style title
        title_html = f'<span style="color: #ffffff; font-size: 14pt; font-weight: bold; font-family: Montserrat;">{plot_widget.windowTitle()}</span>'
        plot_widget.setTitle(title_html)
        
        # Style the axes
        for axis in ['left', 'bottom']:
            plot_widget.getAxis(axis).setPen(pg.mkPen(color='#3644a0', width=1))
            plot_widget.getAxis(axis).setTextPen(pg.mkPen(color='#ffffff'))
            plot_widget.getAxis(axis).setStyle(tickFont=QFont("Montserrat", 8))
        
        # Add grid
        plot_widget.showGrid(x=True, y=True, alpha=0.2)
        
        # Style the legend
        legend = plot_widget.plotItem.legend
        if legend is not None:
            legend.setBrush(pg.mkBrush(QColor(17, 18, 25, 200)))
            legend.setPen(pg.mkPen(color='#3644a0', width=1))

    def _connect_signals(self):
        self.btn_load_ref.clicked.connect(self.load_reference_data)
        self.btn_load_readings.clicked.connect(self.load_readings_data)
        self.btn_analyze.clicked.connect(self.run_analysis)

    def _load_data_from_file(self, data_type_label: str, specified_col_name_widget: QLineEdit):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            f"Load {data_type_label} Signal Data",
            "",
            "CSV Files (*.csv);;Excel Files (*.xlsx *.xls);;All Files (*)"
        )

        if not file_path:
            return None, None, None

        signal_col_name = specified_col_name_widget.text().strip()
        if not signal_col_name:
            self.lbl_status.setText(f"Error: {data_type_label} column name cannot be empty.")
            return None, None, None

        time_col_name = self.txt_time_col.text().strip() if data_type_label == "Readings" else None

        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                self.lbl_status.setText(f"Error: Unsupported file type for {data_type_label}.")
                return None, None, None

            if signal_col_name not in df.columns:
                self.lbl_status.setText(f"Error: Column '{signal_col_name}' not found in {data_type_label} file.")
                return None, None, None

            signal_data = df[signal_col_name].to_numpy(dtype=np.float64)

            time_data_array = None
            if time_col_name and data_type_label == "Readings":
                if time_col_name in df.columns:
                    time_data_array = df[time_col_name].to_numpy(dtype=np.float64)
                    if len(time_data_array) != len(signal_data):
                        self.lbl_status.setText(f"Warning: Time and signal data length mismatch for {data_type_label}. Ignoring time data.")
                        time_data_array = np.arange(len(signal_data))
                else:
                    self.lbl_status.setText(f"Warning: Time column '{time_col_name}' not found. Using indices for x-axis.")
                    time_data_array = np.arange(len(signal_data))
            elif data_type_label == "Readings":
                time_data_array = np.arange(len(signal_data))

            if np.any(np.isnan(signal_data)) or np.any(np.isinf(signal_data)):
                self.lbl_status.setText(f"Warning: {data_type_label} signal contains NaN or Inf values. Attempting to clean by replacing with 0.")
                signal_data = np.nan_to_num(signal_data, nan=0.0, posinf=0.0, neginf=0.0)

            return signal_data, file_path, time_data_array

        except Exception as e:
            self.lbl_status.setText(f"Error loading {data_type_label} data: {e}")
            return None, None, None

    def load_reference_data(self):
        signal, path, _ = self._load_data_from_file("Reference", self.txt_ref_col)
        if signal is not None:
            self.reference_signal = signal
            self.lbl_ref_file.setText(f"Reference: {path.split('/')[-1]} ({len(signal)} pts)")
            self.lbl_status.setText("Status: Reference signal loaded.")
        else:
            self.reference_signal = None
            self.lbl_ref_file.setText("Reference: Not loaded")

    def load_readings_data(self):
        signal, path, time_data = self._load_data_from_file("Readings", self.txt_readings_col)
        if signal is not None:
            self.readings_signal = signal
            self.time_readings = time_data
            self.lbl_readings_file.setText(f"Readings: {path.split('/')[-1]} ({len(signal)} pts)")
            self.lbl_status.setText("Status: Readings signal loaded.")
            self.plot_widget_raw.clear()
            self.plot_widget_raw.plot(self.time_readings, self.readings_signal, 
                                     pen=pg.mkPen(color='#3644a0', width=1.5), 
                                     name="Full Readings Signal")
            self.plot_widget_raw.setTitle("Full Readings Signal (Awaiting Analysis)")
            self.plot_widget_filtered.clear()
        else:
            self.readings_signal = None
            self.time_readings = None
            self.lbl_readings_file.setText("Readings: Not loaded")

    def _normalized_cross_correlation(self, signal, template):
        if len(template) == 0 or len(signal) == 0 or len(template) > len(signal):
            return np.array([])

        template_zm = template - np.mean(template)
        template_std = np.std(template)

        if np.isclose(template_std, 0):
            self.lbl_status.setText("Error: Reference signal standard deviation is zero (flat signal).")
            return np.array([])

        signal_len = len(signal)
        template_len = len(template)
        ncc_scores = np.zeros(signal_len - template_len + 1)

        for i in range(signal_len - template_len + 1):
            window = signal[i : i + template_len]
            window_zm = window - np.mean(window)
            window_std = np.std(window)

            if np.isclose(window_std, 0):
                ncc_scores[i] = 0
                continue
            
            correlation = np.sum(window_zm * template_zm)
            norm_factor = template_len * window_std * template_std

            if np.isclose(norm_factor, 0):
                ncc_scores[i] = 0
            else:
                ncc_scores[i] = correlation / norm_factor

        return np.nan_to_num(ncc_scores, nan=0.0, posinf=0.0, neginf=0.0)

    def run_analysis(self):
        self.lbl_status.setText("Status: Starting analysis...")
        QApplication.processEvents()

        try:
            if self.reference_signal is None or self.readings_signal is None:
                self.lbl_status.setText("Error: Load both reference and readings signals first.")
                return
            if len(self.reference_signal) == 0:
                self.lbl_status.setText("Error: Reference signal is empty.")
                return
            if len(self.reference_signal) > len(self.readings_signal):
                self.lbl_status.setText("Error: Reference signal is longer than readings signal.")
                return

            self.lbl_status.setText("Status: Analyzing... Please wait.")
            QApplication.processEvents()

            ref_sig = self.reference_signal
            read_sig = self.readings_signal
            threshold = self.dsp_threshold.value()
            template_len = len(ref_sig)

            similarity_scores = self._normalized_cross_correlation(read_sig, ref_sig)

            if not similarity_scores.any():
                self.lbl_status.setText("Status: NCC calculation resulted in no valid scores.")
                return

            peaks_indices, properties = find_peaks(
                similarity_scores,
                height=threshold,
                distance=max(1, template_len // 2)
            )

            if not peaks_indices.any():
                self.lbl_status.setText(f"Status: No pulses found above threshold {threshold:.2f}.")
                self.plot_widget_raw.clear()
                self.plot_widget_filtered.clear()
                self.plot_widget_raw.setTitle("Detected Pulses (Raw) - None Found")
                self.plot_widget_filtered.setTitle("Detected Pulses (Noise Zeroed) - None Found")
                self.plot_widget_raw.addLegend(clear=True)
                self.plot_widget_filtered.addLegend(clear=True)
                return

            self.lbl_status.setText(f"Status: Found {len(peaks_indices)} potential pulses. Plotting...")
            QApplication.processEvents()

            self.plot_widget_raw.clear()
            self.plot_widget_raw.setTitle(f"Detected Pulses (Raw) - {len(peaks_indices)} found")
            self.plot_widget_raw.setLabel('bottom', 'Sample Index / Time')
            self.plot_widget_raw.addLegend(clear=True)

            self.plot_widget_filtered.clear()
            self.plot_widget_filtered.setTitle(f"Detected Pulses (Noise Zeroed) - {len(peaks_indices)} found")
            self.plot_widget_filtered.setLabel('bottom', 'Sample Index / Time')
            self.plot_widget_filtered.addLegend(clear=True)

            # Plot original signal in light color - more transparent than before
            self.plot_widget_raw.plot(
                self.time_readings, self.readings_signal,
                pen=pg.mkPen(color=QColor(180, 180, 230, 80), width=1), 
                name="Original Signal"
            )

            plot_data_noise_zero = np.zeros_like(self.readings_signal, dtype=float)
            detected_pulse_details = []

            # Define more vibrant color palette that matches the reference image
            modern_colors = [
                QColor('#fc466b'),  # Pink from reference image 
                QColor('#0b39c4'),  # Strong blue from reference image
                QColor('#00a2ff'),  # Cyan
                QColor('#fb5b5b'),  # Light red
                QColor('#5636f3')   # Purple
            ]

            for i, peak_start_idx_in_scores in enumerate(peaks_indices):
                pulse_start_in_readings = peak_start_idx_in_scores
                pulse_end_in_readings = pulse_start_in_readings + template_len

                if pulse_end_in_readings > len(self.readings_signal):
                    continue

                pulse_segment_data = self.readings_signal[pulse_start_in_readings:pulse_end_in_readings]
                pulse_segment_time = self.time_readings[pulse_start_in_readings:pulse_end_in_readings]

                color = modern_colors[i % len(modern_colors)]
                self.plot_widget_raw.plot(
                    pulse_segment_time, pulse_segment_data,
                    pen=pg.mkPen(color=color, width=2.5),
                    name=f"Pulse {i+1} (Score: {properties['peak_heights'][i]:.2f})"
                )

                plot_data_noise_zero[pulse_start_in_readings:pulse_end_in_readings] = pulse_segment_data

                detected_pulse_details.append({
                    "id": i+1,
                    "start_idx_readings": pulse_start_in_readings,
                    "end_idx_readings": pulse_end_in_readings,
                    "similarity_score": properties['peak_heights'][i]
                })

            # Plot filtered data with glow effect - using blue from reference image
            pulse_plot = self.plot_widget_filtered.plot(
                self.time_readings, plot_data_noise_zero, 
                pen=pg.mkPen(color=QColor('#0b39c4'), width=2.5), 
                name="Combined Filtered Pulses"
            )
            
            # Add glow effect to filtered plot line
            glow = QGraphicsDropShadowEffect()
            glow.setBlurRadius(15)
            glow.setColor(QColor('#0b39c4'))
            glow.setOffset(0, 0)
            # Note: Can't apply directly to plot line as it's not a QWidget
            
            self.lbl_status.setText(f"Status: Analysis complete. Found {len(peaks_indices)} pulses.")
            QApplication.processEvents()

        except Exception as e:
            self.lbl_status.setText(f"Error during analysis: {str(e)}")
            QApplication.processEvents()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PulseAnalyzerApp()
    window.show()
    sys.exit(app.exec())