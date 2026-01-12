from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import pyqtgraph as pg
from gui.styles import Styles
from core.preprocessor import ECGPreprocessor
from core.model_handler import ModelHandler
from core.serial_handler import SerialHandler, ShimmerReader
from core.batch_processor import RecordingBuffer, BatchProcessor
from core.physionet_loader import PhysioNetLoader
from core.shimmer_config import ShimmerConfig
from pathlib import Path
import numpy as np
import time
import os

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.preprocessor = ECGPreprocessor(fs=ShimmerConfig.MODEL_SAMPLING_RATE)
        self.model_handler = ModelHandler()
        
        # Auto-load model
        self.auto_load_model()
        
        self.recording_buffer = RecordingBuffer(
            max_duration_seconds=ShimmerConfig.MAX_RECORDING_DURATION_SEC,
            sampling_rate=ShimmerConfig.SHIMMER_SAMPLING_RATE
        )
        
        # PhysioNet mode
        self.is_physionet_mode = False
        self.physionet_data = None
        self.physionet_fs = None
        self.physionet_playback_index = 0

        self.shimmer_reader = None
        self.batch_processor = None
        
        self.is_recording = False
        self.is_processing = False
        self.recording_start_time = 0
        self.recording_duration = 300
        
        self.processing_results = None
        
        self.display_window = ShimmerConfig.WINDOW_SIZE_SECONDS
        self.processed_curve = None
        self.processed_data_buffer = []
        self.time_buffer = []
        self.current_time = 0
        self.fs_viz = ShimmerConfig.MODEL_SAMPLING_RATE

        self.mv_values_buffer = []
        
        self.init_ui()
        self.apply_styles()
        self.setup_timers()
    
    def auto_load_model(self):
        """Auto-load model from default path"""
        try:
            model_path = ShimmerConfig.DEFAULT_MODEL_PATH
            if os.path.exists(model_path):
                print(f"Auto-loading model from: {model_path}")
                success, message = self.model_handler.load_model(model_path)
                if success:
                    print("Model loaded successfully!")
                else:
                    print(f"Failed to load model: {message}")
            else:
                print(f"Warning: Model file not found at {model_path}")
        except Exception as e:
            print(f"Error auto-loading model: {e}")
        
    def init_ui(self):
        self.setWindowTitle("AF Detection System - Shimmer ECG")
        self.setGeometry(100, 100, 1600, 900)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        main_layout.addWidget(self.create_header())
        
        content = QWidget()
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(20, 20, 20, 20)
        content_layout.setSpacing(20)
        
        content_layout.addWidget(self.create_sidebar(), stretch=0)
        content_layout.addWidget(self.create_main_content(), stretch=3)
        content_layout.addWidget(self.create_results_panel(), stretch=1)
        
        main_layout.addWidget(content)
        
    def create_header(self):
        header = QWidget()
        header.setObjectName("header")
        header.setFixedHeight(80)
        
        layout = QHBoxLayout(header)
        layout.setContentsMargins(30, 0, 30, 0)
        
        title = QLabel("🫀 Atrial Fibrillation Detection System")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #1e293b;")
        
        self.connection_status = QLabel("● Not Connected")
        self.connection_status.setStyleSheet("color: #64748b; font-weight: bold; font-size: 13px;")
        
        layout.addWidget(title)
        layout.addStretch()
        layout.addWidget(self.connection_status)
        
        return header
    
    def create_sidebar(self):
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(320)
        
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Data Source Selection
        source_group = QGroupBox("Data Source")
        source_group.setObjectName("groupBox")
        source_layout = QVBoxLayout()

        self.source_combo = QComboBox()
        self.source_combo.setFixedHeight(35)
        self.source_combo.addItem("Shimmer Device", "shimmer")
        self.source_combo.addItem("PhysioNet File", "physionet")
        self.source_combo.currentIndexChanged.connect(self.on_source_changed)
        source_layout.addWidget(self.source_combo)

        source_group.setLayout(source_layout)
        layout.addWidget(source_group)

        # PhysioNet File Loader
        self.physionet_group = QGroupBox("PhysioNet File")
        self.physionet_group.setObjectName("groupBox")
        physionet_layout = QVBoxLayout()

        self.file_path_label = QLabel("No file loaded")
        self.file_path_label.setStyleSheet("color: #64748b; font-size: 11px; padding: 5px; word-wrap: break-word;")
        self.file_path_label.setWordWrap(True)
        physionet_layout.addWidget(self.file_path_label)

        fs_label = QLabel("Sampling Rate (Hz):")
        fs_label.setStyleSheet("font-weight: normal; font-size: 12px; margin-top: 5px;")
        physionet_layout.addWidget(fs_label)

        self.physionet_fs_combo = QComboBox()
        self.physionet_fs_combo.setFixedHeight(35)
        self.physionet_fs_combo.addItem("128 Hz", 128)
        self.physionet_fs_combo.addItem("200 Hz", 200)
        self.physionet_fs_combo.addItem("250 Hz", 250)
        self.physionet_fs_combo.addItem("360 Hz", 360)
        self.physionet_fs_combo.addItem("500 Hz", 500)
        self.physionet_fs_combo.addItem("1000 Hz", 1000)
        self.physionet_fs_combo.setCurrentIndex(1)
        physionet_layout.addWidget(self.physionet_fs_combo)

        self.load_file_btn = self.create_button("📁 Load .dat File", "#8b5cf6")
        self.load_file_btn.clicked.connect(self.load_physionet_file)
        physionet_layout.addWidget(self.load_file_btn)

        self.physionet_group.setLayout(physionet_layout)
        self.physionet_group.setVisible(False)
        layout.addWidget(self.physionet_group)

        
        # Serial Port
        self.port_group = QGroupBox("Serial Port")
        self.port_group.setObjectName("groupBox")
        port_layout = QVBoxLayout()
        
        port_label = QLabel("Port:")
        port_label.setStyleSheet("font-weight: normal; font-size: 12px;")
        port_layout.addWidget(port_label)
        
        self.port_combo = QComboBox()
        self.port_combo.setFixedHeight(35)
        port_layout.addWidget(self.port_combo)
        
        self.refresh_btn = self.create_button("🔄 Refresh Ports", "#8b5cf6")
        self.refresh_btn.clicked.connect(self.refresh_ports)
        port_layout.addWidget(self.refresh_btn)
        
        self.port_status = QLabel("No port selected")
        self.port_status.setStyleSheet("color: #64748b; font-size: 11px; padding: 5px;")
        port_layout.addWidget(self.port_status)
        
        self.port_group.setLayout(port_layout)
        layout.addWidget(self.port_group)
        
        # Sampling Rate
        self.sampling_group = QGroupBox("Sampling Rate")
        self.sampling_group.setObjectName("groupBox")
        sampling_layout = QVBoxLayout()
        
        rate_label = QLabel("Rate (Hz):")
        rate_label.setStyleSheet("font-weight: normal; font-size: 12px;")
        sampling_layout.addWidget(rate_label)
        
        self.sampling_rate_combo = QComboBox()
        self.sampling_rate_combo.setFixedHeight(35)
        for rate in ShimmerConfig.AVAILABLE_SAMPLING_RATES:
            self.sampling_rate_combo.addItem(str(rate), rate)
        default_index = self.sampling_rate_combo.findData(ShimmerConfig.DEFAULT_SAMPLING_RATE)
        if default_index >= 0:
            self.sampling_rate_combo.setCurrentIndex(default_index)
        self.sampling_rate_combo.currentIndexChanged.connect(self.on_sampling_rate_changed)
        sampling_layout.addWidget(self.sampling_rate_combo)
        
        self.sampling_group.setLayout(sampling_layout)
        layout.addWidget(self.sampling_group)
        
        # Recording
        recording_group = QGroupBox("Recording")
        recording_group.setObjectName("groupBox")
        recording_layout = QVBoxLayout()
        
        duration_label = QLabel("Duration:")
        duration_label.setStyleSheet("font-weight: normal; font-size: 12px;")
        recording_layout.addWidget(duration_label)
        
        self.duration_combo = QComboBox()
        self.duration_combo.addItems(["1 minute", "2 minutes", "5 minutes", "10 minutes", "15 minutes"])
        self.duration_combo.setCurrentText("5 minutes")
        self.duration_combo.setFixedHeight(35)
        self.duration_combo.currentTextChanged.connect(self.on_duration_changed)
        recording_layout.addWidget(self.duration_combo)
        
        self.timer_label = QLabel("00:00 / 05:00")
        self.timer_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #1e293b; padding: 10px; text-align: center;")
        self.timer_label.setAlignment(Qt.AlignCenter)
        recording_layout.addWidget(self.timer_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(25)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #e2e8f0;
                border-radius: 5px;
                text-align: center;
                background-color: white;
            }
            QProgressBar::chunk {
                background-color: #3b82f6;
                border-radius: 3px;
            }
        """)
        recording_layout.addWidget(self.progress_bar)
        
        self.sample_count_label = QLabel("Samples: 0")
        self.sample_count_label.setStyleSheet("color: #64748b; font-size: 11px; padding: 5px;")
        recording_layout.addWidget(self.sample_count_label)
        
        self.start_btn = self.create_button("▶ Start Recording", "#10b981")
        self.start_btn.clicked.connect(self.start_recording)
        self.start_btn.setEnabled(False)
        recording_layout.addWidget(self.start_btn)
        
        self.stop_btn = self.create_button("⏹ Stop Recording", "#ef4444")
        self.stop_btn.clicked.connect(self.stop_recording)
        self.stop_btn.setEnabled(False)
        recording_layout.addWidget(self.stop_btn)
        
        self.reset_btn = self.create_button("🔄 Reset", "#f59e0b")
        self.reset_btn.clicked.connect(self.reset_all)
        recording_layout.addWidget(self.reset_btn)
        
        recording_group.setLayout(recording_layout)
        layout.addWidget(recording_group)
        
        layout.addStretch()
        
        self.refresh_ports()
        return sidebar
    
    def create_main_content(self):
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(15)
        layout.setContentsMargins(0, 0, 0, 0)
        
        plot_container = QWidget()
        plot_container.setObjectName("plotContainer")
        plot_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        plot_layout = QVBoxLayout(plot_container)
        plot_layout.setContentsMargins(10, 35, 10, 10)
        plot_layout.setSpacing(0)
        
        self.plot_title_label = QLabel("Real-time Preprocessed ECG Signal (10 seconds window)")
        self.plot_title_label.setStyleSheet("""
            font-size: 14px;
            font-weight: bold;
            color: #1e293b;
            background-color: rgba(255, 255, 255, 200);
            padding: 5px 10px;
            border-radius: 5px;
        """)
        self.plot_title_label.setAlignment(Qt.AlignCenter)
        
        self.processed_plot = pg.PlotWidget()
        
        self.processed_plot = pg.PlotWidget()
        self.processed_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.processed_plot.setMinimumHeight(400)
        self.processed_plot.setBackground('#ffffff')
        self.processed_plot.setLabel('left', 'Amplitude (mV)', color='#1e293b')
        self.processed_plot.setLabel('bottom', 'Time (s)', color='#1e293b')
        self.processed_plot.showGrid(x=True, y=True, alpha=0.3)
        self.processed_plot.setMouseEnabled(x=False, y=False)
        self.processed_plot.setMenuEnabled(False)
        
        self.processed_curve = None
        
        plot_layout.addWidget(self.processed_plot)
        layout.addWidget(plot_container, stretch=1)
        
        self.processing_widget = self.create_processing_widget()
        self.processing_widget.setVisible(False)
        layout.addWidget(self.processing_widget)
        
        return content
    
    def create_results_panel(self):
        panel = QWidget()
        panel.setObjectName("sidebar")
        panel.setFixedWidth(350)
        
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        
        # Classification Result
        result_group = QGroupBox("Classification Result")
        result_group.setObjectName("groupBox")
        result_layout = QVBoxLayout()
        
        self.final_classification_label = QLabel("--")
        self.final_classification_label.setStyleSheet("""
            font-size: 28px;
            font-weight: bold;
            padding: 20px;
            text-align: center;
            color: #64748b;
            border: 2px solid #e2e8f0;
            border-radius: 10px;
            background-color: #f8fafc;
        """)
        self.final_classification_label.setAlignment(Qt.AlignCenter)
        self.final_classification_label.setWordWrap(True)
        result_layout.addWidget(self.final_classification_label)
        
        result_group.setLayout(result_layout)
        layout.addWidget(result_group)
        
        # Detection Statistics
        stats_group = QGroupBox("Detection Statistics")
        stats_group.setObjectName("groupBox")
        stats_layout = QVBoxLayout()
        
        af_widget = QWidget()
        af_layout = QHBoxLayout(af_widget)
        af_layout.setContentsMargins(0, 5, 0, 5)
        af_icon = QLabel("🔴")
        af_icon.setStyleSheet("font-size: 24px;")
        self.af_count_label = QLabel("AF: --")
        self.af_count_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #ef4444;")
        af_layout.addWidget(af_icon)
        af_layout.addWidget(self.af_count_label)
        af_layout.addStretch()
        stats_layout.addWidget(af_widget)
        
        normal_widget = QWidget()
        normal_layout = QHBoxLayout(normal_widget)
        normal_layout.setContentsMargins(0, 5, 0, 5)
        normal_icon = QLabel("🟢")
        normal_icon.setStyleSheet("font-size: 24px;")
        self.normal_count_label = QLabel("Normal: --")
        self.normal_count_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #10b981;")
        normal_layout.addWidget(normal_icon)
        normal_layout.addWidget(self.normal_count_label)
        normal_layout.addStretch()
        stats_layout.addWidget(normal_widget)
        
        self.total_segments_label = QLabel("Total: -- segments")
        self.total_segments_label.setStyleSheet("color: #64748b; font-size: 15px; padding-top: 10px; font-weight: bold;")
        stats_layout.addWidget(self.total_segments_label)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        # ✅ TAMBAH INI: Signal Voltage (di antara Detection Statistics dan Performance)
        voltage_group = QGroupBox("Signal Voltage")
        voltage_group.setObjectName("groupBox")
        voltage_layout = QVBoxLayout()
        
        voltage_label = QLabel("Average Voltage:")
        voltage_label.setStyleSheet("font-size: 12px; color: #64748b; font-weight: bold;")
        voltage_layout.addWidget(voltage_label)
        
        self.avg_voltage_label = QLabel("-- mV")
        self.avg_voltage_label.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            color: #3b82f6;
            background-color: #eff6ff;
            padding: 15px;
            border-radius: 8px;
            border: 2px solid #3b82f6;
        """)
        self.avg_voltage_label.setAlignment(Qt.AlignCenter)
        voltage_layout.addWidget(self.avg_voltage_label)
        
        voltage_group.setLayout(voltage_layout)
        layout.addWidget(voltage_group)
        
        # Computation Time
        comp_group = QGroupBox("Performance")
        comp_group.setObjectName("groupBox")
        comp_layout = QVBoxLayout()
        
        self.comp_time_label = QLabel("Computation Time:\n--")
        self.comp_time_label.setWordWrap(True)
        self.comp_time_label.setStyleSheet("background: #f1f5f9; padding: 15px; border-radius: 5px; font-size: 14px; color: #475569; font-weight: bold;")
        comp_layout.addWidget(self.comp_time_label)
        
        comp_group.setLayout(comp_layout)
        layout.addWidget(comp_group)
        
        layout.addStretch()
        
        return panel
    
    def create_processing_widget(self):
        widget = QWidget()
        widget.setObjectName("plotContainer")
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(20, 20, 20, 20)
        
        title = QLabel("⚙️ Processing Recording...")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #1e293b; padding: 10px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        self.processing_status_label = QLabel("Initializing...")
        self.processing_status_label.setStyleSheet("font-size: 14px; color: #64748b; padding: 5px;")
        self.processing_status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.processing_status_label)
        
        self.processing_progress_bar = QProgressBar()
        self.processing_progress_bar.setFixedHeight(30)
        self.processing_progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #e2e8f0;
                border-radius: 5px;
                text-align: center;
                background-color: white;
                font-size: 13px;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #f59e0b;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.processing_progress_bar)
        
        return widget
    
    def create_button(self, text, color):
        btn = QPushButton(text)
        btn.setFixedHeight(45)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 14px;
                font-weight: bold;
                padding: 10px;
            }}
            QPushButton:hover {{
                background-color: {self.darken_color(color)};
            }}
            QPushButton:pressed {{
                background-color: {self.darken_color(color, 0.3)};
            }}
            QPushButton:disabled {{
                background-color: #cbd5e1;
                color: #94a3b8;
            }}
        """)
        return btn
    
    def darken_color(self, color, factor=0.2):
        color = color.lstrip('#')
        r, g, b = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
        r = int(r * (1 - factor))
        g = int(g * (1 - factor))
        b = int(b * (1 - factor))
        return f'#{r:02x}{g:02x}{b:02x}'
    
    def apply_styles(self):
        self.setStyleSheet(Styles.get_main_style())
    
    def setup_timers(self):
        self.viz_timer = QTimer()
        self.viz_timer.timeout.connect(self.update_visualization)
        
        self.recording_timer = QTimer()
        self.recording_timer.timeout.connect(self.update_recording_status)
        
        self.preprocess_timer = QTimer()
        self.preprocess_timer.timeout.connect(self.preprocess_for_visualization)
    
    def on_source_changed(self, index):
        source = self.source_combo.currentData()
        if source == "physionet":
            self.is_physionet_mode = True
            self.physionet_group.setVisible(True)
            self.port_group.setVisible(False)
            self.sampling_group.setVisible(False)
            self.start_btn.setText("▶ Process File")
            self.plot_title_label.setText("Preprocessed ECG Signal (First 10 seconds)")
        else:
            self.is_physionet_mode = False
            self.physionet_group.setVisible(False)
            self.port_group.setVisible(True)
            self.sampling_group.setVisible(True)
            self.start_btn.setText("▶ Start Recording")
            self.plot_title_label.setText("Real-time Preprocessed ECG Signal (10 seconds window)")
        self.check_ready_state()

    def load_physionet_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load PhysioNet Record", "D:/skripsi_teknis/pengujian/afdb",
            "PhysioNet Files (*.dat);;All Files (*)"
        )
        if file_path:
            fs_input = self.physionet_fs_combo.currentData()
            signals, fs, success, message = PhysioNetLoader.load_physionet_record(file_path, sampling_rate=fs_input)
            if success:
                self.physionet_data = signals
                self.physionet_fs = fs
                self.physionet_data = PhysioNetLoader.convert_to_shimmer_format(self.physionet_data)
                duration_min = len(signals) / fs / 60
                self.file_path_label.setText(f"✓ Loaded: {Path(file_path).name}\n{len(signals):,} samples @ {fs} Hz\nDuration: {duration_min:.2f} min")
                self.file_path_label.setStyleSheet("color: #10b981; font-size: 11px; padding: 5px; word-wrap: break-word;")
                QMessageBox.information(self, "File Loaded", f"Successfully loaded:\n{Path(file_path).name}\n\nSamples: {len(signals):,}\nSampling Rate: {fs} Hz\nDuration: {duration_min:.2f} minutes")
            
            else:
                self.physionet_data = None
                self.physionet_fs = None
                self.file_path_label.setText(f"✗ {message}")
                self.file_path_label.setStyleSheet("color: #ef4444; font-size: 11px; padding: 5px; word-wrap: break-word;")
                QMessageBox.critical(self, "Load Error", message)
            self.check_ready_state()

    def process_physionet_file(self):
        """Process PhysioNet file with visualization"""
        print("\n=== PROCESSING PHYSIONET FILE ===")
        
        if self.physionet_data is None:
            QMessageBox.warning(self, "Error", "No file loaded")
            return
        
        self.processing_results = None
        self.final_classification_label.setText("--")
        self.af_count_label.setText("AF: --")
        self.normal_count_label.setText("Normal: --")
        self.total_segments_label.setText("Total: -- segments")
        self.comp_time_label.setText("Computation Time:\n--")
        self.avg_voltage_label.setText("-- mV")
        
        self.start_btn.setEnabled(False)
        self.load_file_btn.setEnabled(False)
        self.source_combo.setEnabled(False)
        
        self.connection_status.setText("● Processing File...")
        self.connection_status.setStyleSheet("color: #f59e0b; font-weight: bold; font-size: 13px;")
        
        # Calculate average voltage
        raw_mv = self.preprocessor.adc_to_millivolts(
            self.physionet_data, 
            gain=ShimmerConfig.ECG_GAIN,
            offset=ShimmerConfig.ADC_OFFSET
        )
        
        # ✅ TAMBAH INI: Setup untuk visualisasi
        self.is_recording = True  # Aktifkan mode "recording" untuk plot
        self.physionet_playback_index = 0
        self.processed_data_buffer.clear()
        self.time_buffer.clear()
        self.current_time = 0
        
        # Create plot curve
        self.processed_curve = self.processed_plot.plot(
            pen=pg.mkPen(color='#2C7BE5', width=1.5)
        )
        
        # ✅ Start visualization timer
        self.viz_timer.start(50)
        self.physionet_viz_timer = QTimer()
        self.physionet_viz_timer.timeout.connect(self.playback_physionet_data)
        self.physionet_viz_timer.start(50)  # Update setiap 50ms
        
        # Start batch processing (background)
        self.is_processing = True
        self.processing_widget.setVisible(True)
        self.processing_progress_bar.setValue(0)
        self.processing_status_label.setText("Initializing...")
        
        print(f"Processing {len(self.physionet_data)} samples from PhysioNet file...")
        
        self.batch_processor = BatchProcessor(
            recorded_data=self.physionet_data,
            preprocessor=self.preprocessor,
            model_handler=self.model_handler,
            original_fs=self.physionet_fs,
            target_fs=ShimmerConfig.MODEL_SAMPLING_RATE,
            window_size=ShimmerConfig.WINDOW_SIZE_SAMPLES
        )
        
        self.batch_processor.progress_update.connect(self.on_processing_progress)
        self.batch_processor.processing_complete.connect(self.on_processing_complete_physionet)
        self.batch_processor.error_occurred.connect(self.on_processing_error)
        self.batch_processor.start()

    def playback_physionet_data(self):
        """Simulate streaming from PhysioNet file"""
        if not self.is_recording or self.physionet_data is None:
            return
        
        # Ambil chunk data (simulasi streaming)
        chunk_size = ShimmerConfig.PREPROCESSING_CHUNK_SIZE  # 128 samples
        
        if self.physionet_playback_index >= len(self.physionet_data):
            # Sudah selesai semua data
            if hasattr(self, 'physionet_viz_timer'):
                self.physionet_viz_timer.stop()
            return
        
        # Get chunk
        end_index = min(self.physionet_playback_index + chunk_size, len(self.physionet_data))
        chunk = self.physionet_data[self.physionet_playback_index:end_index]
        
        if len(chunk) < chunk_size:
            # Last chunk, pad or skip
            if hasattr(self, 'physionet_viz_timer'):
                self.physionet_viz_timer.stop()
            return
        
        self.physionet_playback_index = end_index
        
        try:
            # Resample chunk
            from scipy.signal import resample
            resampled = resample(chunk, ShimmerConfig.RESAMPLED_CHUNK_SIZE)
            
            # Konversi ke mV
            raw_mv = self.preprocessor.adc_to_millivolts(
                resampled, 
                gain=ShimmerConfig.ECG_GAIN,
                offset=ShimmerConfig.ADC_OFFSET
            )
            self.mv_values_buffer.extend(raw_mv)
            
            # ✅ GANTI DENGAN INI (input sudah mV):
            processed_chunk = self.preprocessor.preprocess_for_plot(raw_mv)
            
            # Add to buffer
            for i, val in enumerate(processed_chunk):
                self.processed_data_buffer.append(val)
                self.time_buffer.append(self.current_time)
                self.current_time += 1 / ShimmerConfig.MODEL_SAMPLING_RATE
            
            # Limit buffer to 10 seconds for display
            max_samples = int(self.display_window * ShimmerConfig.MODEL_SAMPLING_RATE)
            if len(self.processed_data_buffer) > max_samples:
                excess = len(self.processed_data_buffer) - max_samples
                self.processed_data_buffer = self.processed_data_buffer[excess:]
                self.time_buffer = self.time_buffer[excess:]
        
        except Exception as e:
            print(f"Playback error: {e}")

    def on_processing_complete_physionet(self, results):
        """Handle completion for PhysioNet mode"""
        print("\n=== PROCESSING COMPLETE (PHYSIONET) ===")
        
        # Stop visualization
        self.is_recording = False
        self.viz_timer.stop()
        if hasattr(self, 'physionet_viz_timer'):
            self.physionet_viz_timer.stop()
        
        # Call original completion handler
        self.on_processing_complete(results)

    def visualize_physionet_signal(self):
        """Visualize PhysioNet signal after loading"""
        if self.physionet_data is None:
            return
        
        print("Visualizing PhysioNet signal...")
        
        # Convert to mV for visualization
        raw_mv = self.preprocessor.adc_to_millivolts(self.physionet_data)
        
        # Take first 10 seconds for display
        display_samples = int(10 * self.physionet_fs)
        if len(raw_mv) > display_samples:
            display_data = raw_mv[:display_samples]
        else:
            display_data = raw_mv
        
        # Preprocess for plot (DC removal + filter, no normalization)
        from scipy.signal import resample
        
        # Resample to model sampling rate if needed
        if self.physionet_fs != ShimmerConfig.MODEL_SAMPLING_RATE:
            target_samples = int(len(display_data) * ShimmerConfig.MODEL_SAMPLING_RATE / self.physionet_fs)
            display_data = resample(display_data, target_samples)
        
        # Apply preprocessing
        processed_data = self.preprocessor.preprocess_for_plot(display_data)
        
        # Generate time axis
        time_axis = np.arange(len(processed_data)) / ShimmerConfig.MODEL_SAMPLING_RATE
        
        # Clear previous plot
        self.processed_plot.clear()
        
        # Plot
        self.processed_curve = self.processed_plot.plot(
            time_axis,
            processed_data,
            pen=pg.mkPen(color='#2C7BE5', width=1.5)
        )
        
        # Set range
        self.processed_plot.setXRange(0, 10, padding=0)
        
        print(f"Plotted {len(processed_data)} samples (10 seconds)")
    
    def refresh_ports(self):
        self.port_combo.clear()
        ports = SerialHandler.get_available_ports()
        if ports:
            self.port_combo.addItems(ports)
            self.port_status.setText(f"Found {len(ports)} port(s)")
            self.port_status.setStyleSheet("color: #10b981; font-size: 11px; padding: 5px;")
        else:
            self.port_status.setText("No ports found")
            self.port_status.setStyleSheet("color: #ef4444; font-size: 11px; padding: 5px;")
        self.check_ready_state()
    
    def on_sampling_rate_changed(self, index):
        """Handle sampling rate change"""
        if self.is_recording:
            QMessageBox.warning(
                self,
                "Cannot Change",
                "Cannot change sampling rate during recording!"
            )
            return
        
        selected_rate = self.sampling_rate_combo.currentData()
        ShimmerConfig.SHIMMER_SAMPLING_RATE = selected_rate
        
        # Update recording buffer
        self.recording_buffer = RecordingBuffer(
            max_duration_seconds=ShimmerConfig.MAX_RECORDING_DURATION_SEC,
            sampling_rate=selected_rate
        )
    
    def check_ready_state(self):
        model_loaded = self.model_handler.model is not None
        if self.is_physionet_mode:
            file_loaded = self.physionet_data is not None
            ready = model_loaded and file_loaded and not self.is_recording
        else:
            port_available = self.port_combo.count() > 0
            ready = model_loaded and port_available and not self.is_recording
        self.start_btn.setEnabled(ready)
    
    def on_duration_changed(self, text):
        self.recording_duration = ShimmerConfig.RECORDING_DURATIONS.get(text, 300)
        minutes = self.recording_duration // 60
        seconds = self.recording_duration % 60
        self.timer_label.setText(f"00:00 / {minutes:02d}:{seconds:02d}")
        self.progress_bar.setValue(0)
    
    def reset_all(self):
        reply = QMessageBox.question(
            self,
            'Reset Confirmation',
            'This will clear all data and reset the system. Continue?',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            print("\n=== RESETTING SYSTEM ===")
            
            if self.is_recording:
                if self.shimmer_reader:
                    self.shimmer_reader.stop()
                    self.shimmer_reader = None
                self.viz_timer.stop()
                self.recording_timer.stop()
                self.preprocess_timer.stop()
            
            if self.is_processing and self.batch_processor:
                self.batch_processor.stop()
                # Wait briefly for the worker to finish to avoid race conditions, but do not block UI long
                self.batch_processor.wait(1000)
                self.batch_processor = None
            
            self.recording_buffer.clear()
            self.processed_data_buffer.clear()
            self.time_buffer.clear()
            self.current_time = 0
            self.processing_results = None

            self.mv_values_buffer = []

            self.avg_voltage_label.setText("-- mV")
            
            if self.processed_curve:
                self.processed_curve.setData([], [])
            
            self.is_recording = False
            self.is_processing = False
            
            self.final_classification_label.setText("--")
            self.final_classification_label.setStyleSheet("""
                font-size: 28px;
                font-weight: bold;
                padding: 20px;
                text-align: center;
                color: #64748b;
                border: 2px solid #e2e8f0;
                border-radius: 10px;
                background-color: #f8fafc;
            """)
            self.af_count_label.setText("AF: --")
            self.normal_count_label.setText("Normal: --")
            self.total_segments_label.setText("Total: -- segments")
            self.comp_time_label.setText("Computation Time:\n--")

            self.physionet_data = None
            self.physionet_fs = None
            self.physionet_playback_index = 0
            self.file_path_label.setText("No file loaded")
            self.file_path_label.setStyleSheet("color: #64748b; font-size: 11px; padding: 5px; word-wrap: break-word;")
            self.source_combo.setEnabled(True)
            self.load_file_btn.setEnabled(True)

            if hasattr(self, 'physionet_viz_timer'):
                self.physionet_viz_timer.stop()

            self.processed_plot.clear()
            self.processed_curve = None
            
            minutes = self.recording_duration // 60
            seconds = self.recording_duration % 60
            self.timer_label.setText(f"00:00 / {minutes:02d}:{seconds:02d}")
            self.progress_bar.setValue(0)
            self.sample_count_label.setText("Samples: 0")
            
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.port_combo.setEnabled(True)
            self.sampling_rate_combo.setEnabled(True)
            self.duration_combo.setEnabled(True)
            
            self.connection_status.setText("● Not Connected")
            self.connection_status.setStyleSheet("color: #64748b; font-weight: bold; font-size: 13px;")
            
            self.processing_widget.setVisible(False)
            
            self.check_ready_state()
            
            print("System reset complete")
            QMessageBox.information(self, "Reset Complete", "System has been reset successfully!")
    
    def start_recording(self):
        if self.is_physionet_mode:
            self.process_physionet_file()
        else:
            self.start_shimmer_recording()
    
    def start_shimmer_recording(self):
        print("\n=== STARTING RECORDING ===")
        
        self.recording_buffer.clear()
        self.processing_results = None
        
        self.processed_data_buffer.clear()
        self.time_buffer.clear()
        self.current_time = 0
        self._last_processed_count = 0

        self.mv_values_buffer = []

        self.avg_voltage_label.setText("-- mV")
        
        self.final_classification_label.setText("--")
        self.final_classification_label.setStyleSheet("""
            font-size: 28px;
            font-weight: bold;
            padding: 20px;
            text-align: center;
            color: #64748b;
            border: 2px solid #e2e8f0;
            border-radius: 10px;
            background-color: #f8fafc;
        """)
        self.af_count_label.setText("AF: --")
        self.normal_count_label.setText("Normal: --")
        self.total_segments_label.setText("Total: -- segments")
        self.comp_time_label.setText("Computation Time:\n--")
        
        port = self.port_combo.currentText()
        
        if not port:
            QMessageBox.warning(self, "Error", "Please select a COM port")
            return
        
        try:
            self.shimmer_reader = ShimmerReader(
                port=port,
                baudrate=ShimmerConfig.DEFAULT_BAUDRATE,
                channel=ShimmerConfig.DEFAULT_ECG_CHANNEL
            )
            
            self.shimmer_reader.data_received.connect(self.on_data_received)
            self.shimmer_reader.error_occurred.connect(self.on_shimmer_error)
            
            self.shimmer_reader.start()
            
            self.is_recording = True
            self.recording_start_time = time.time()
            
            self.progress_bar.setValue(0)
            self.sample_count_label.setText("Samples: 0")
            
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.port_combo.setEnabled(False)
            self.sampling_rate_combo.setEnabled(False)
            self.duration_combo.setEnabled(False)
            
            self.connection_status.setText("● Recording...")
            self.connection_status.setStyleSheet("color: #ef4444; font-weight: bold; font-size: 13px;")
            self.port_status.setText("Connected")
            self.port_status.setStyleSheet("color: #10b981; font-size: 11px; padding: 5px;")
            
            self.processed_curve = self.processed_plot.plot(
                pen=pg.mkPen(color='#2C7BE5', width=1.5)
            )
            
            self.viz_timer.start(100)
            self.recording_timer.start(100)
            self.preprocess_timer.start(100)
            
            print("Recording started successfully")
            
        except Exception as e:
            QMessageBox.critical(self, "Connection Error", f"Failed to connect: {str(e)}")
            self.is_recording = False
    
    def stop_recording(self):
        print("\n=== STOPPING RECORDING ===")

        if self.is_physionet_mode:
            return
        
        self.is_recording = False
        
        self.viz_timer.stop()
        self.recording_timer.stop()
        self.preprocess_timer.stop()
        
        if self.shimmer_reader:
            self.shimmer_reader.stop()
            self.shimmer_reader = None
        
        self.stop_btn.setEnabled(False)
        self.port_combo.setEnabled(True)
        self.sampling_rate_combo.setEnabled(True)
        self.duration_combo.setEnabled(True)
        
        self.connection_status.setText("● Recording Complete")
        self.connection_status.setStyleSheet("color: #10b981; font-weight: bold; font-size: 13px;")
        
        print(f"\n{'='*60}")
        print(f"VOLTAGE CALCULATION:")
        print(f"  mv_values_buffer length: {len(self.mv_values_buffer)}")

        if len(self.mv_values_buffer) > 0:
            mv_array = np.array(self.mv_values_buffer)
            mv_array = mv_array - np.mean(mv_array) # Remove DC offset
            avg_mv = np.mean(np.abs(mv_array))
            print(f"  Abs Mean: {avg_mv:.4f} mV")
            
            self.avg_voltage_label.setText(f"{avg_mv:.4f} mV")
            print(f"  ✅ Label set to: {avg_mv:.4f} mV")
            print(f"  ✅ Label text now: {self.avg_voltage_label.text()}")
        else:
            self.avg_voltage_label.setText("-- mV")
            print(f"  ❌ ERROR: Buffer is EMPTY!")
            print(f"  ❌ Label set to: -- mV")

        print(f"{'='*60}\n")
        
        sample_count = self.recording_buffer.get_sample_count()
        print(f"Recording stopped. Total samples: {sample_count}")
        
        if sample_count > 0:
            self.start_batch_processing()
        else:
            QMessageBox.warning(self, "No Data", "No data was recorded")
            self.check_ready_state()
    
    def start_batch_processing(self):
        print("\n=== STARTING BATCH PROCESSING ===")
        
        self.is_processing = True
        
        self.processing_widget.setVisible(True)
        self.processing_progress_bar.setValue(0)
        self.processing_status_label.setText("Initializing...")
        
        self.connection_status.setText("● Processing...")
        self.connection_status.setStyleSheet("color: #f59e0b; font-weight: bold; font-size: 13px;")
        
        recorded_data = self.recording_buffer.get_data()
        print(f"Processing {len(recorded_data)} samples...")
        
        self.batch_processor = BatchProcessor(
            recorded_data=recorded_data,
            preprocessor=self.preprocessor,
            model_handler=self.model_handler,
            original_fs=ShimmerConfig.SHIMMER_SAMPLING_RATE,
            target_fs=ShimmerConfig.MODEL_SAMPLING_RATE,
            window_size=ShimmerConfig.WINDOW_SIZE_SAMPLES
        )
        
        self.batch_processor.progress_update.connect(self.on_processing_progress)
        self.batch_processor.processing_complete.connect(self.on_processing_complete)
        self.batch_processor.error_occurred.connect(self.on_processing_error)
        self.batch_processor.start()
    
    def on_data_received(self, value):
        if not self.is_recording:
            return
        
        self.recording_buffer.add_sample(value)
        
        sample_count = self.recording_buffer.get_sample_count()
        if sample_count <= 10 or sample_count % 512 == 0:
            print(f"Received sample #{sample_count}: {value:.4f}")
        
        elapsed = time.time() - self.recording_start_time
        if elapsed >= self.recording_duration:
            print("Recording duration reached, stopping...")
            self.stop_recording()
    
    def on_shimmer_error(self, error_msg):
        print(f"Shimmer error: {error_msg}")
        QMessageBox.critical(self, "Shimmer Error", error_msg)
        if self.is_recording:
            self.stop_recording()
    
    def preprocess_for_visualization(self):
        if not self.is_recording:
            return
        
        current_sample_count = self.recording_buffer.get_sample_count()
        
        if not hasattr(self, '_last_processed_count'):
            self._last_processed_count = 0
        
        chunk_size = ShimmerConfig.PREPROCESSING_CHUNK_SIZE
        if current_sample_count - self._last_processed_count < chunk_size:
            return
        
        try:
            viz_data = self.recording_buffer.get_visualization_data()
            
            if len(viz_data) < chunk_size:
                return
            
            chunk = viz_data[-chunk_size:]
            
            from scipy.signal import resample
            resampled = resample(chunk, ShimmerConfig.RESAMPLED_CHUNK_SIZE)
            
            raw_mv = self.preprocessor.adc_to_millivolts(
                resampled, 
                gain=ShimmerConfig.ECG_GAIN,
                offset=ShimmerConfig.ADC_OFFSET  # ← TAMBAH INI
            )

            print(f"\n--- DATA DEBUG ---")
            print(f"ADC (first 10 samples): {resampled[:10].astype(int)}")
            print(f"mV  (first 10 samples): {np.round(raw_mv[:10], 4)}")
            print(f"Min/Max ADC: {np.min(resampled):.0f} / {np.max(resampled):.0f}")
            print(f"Min/Max mV : {np.min(raw_mv):.4f} / {np.max(raw_mv):.4f}")
            print(f"Average |mV|: {np.mean(np.abs(raw_mv)):.4f} mV")
            print(f"Gain used: {ShimmerConfig.ECG_GAIN}")
            print(f"-------------------\n")
            
            # ✅ GANTI DENGAN INI (input sudah mV):
            raw_mv = self.preprocessor.adc_to_millivolts(resampled)
            self.mv_values_buffer.extend(raw_mv)
            processed_chunk = self.preprocessor.preprocess_for_plot(raw_mv)

        
            for i, val in enumerate(processed_chunk):
                self.processed_data_buffer.append(val)
                self.time_buffer.append(self.current_time)
                self.current_time += 1 / ShimmerConfig.MODEL_SAMPLING_RATE
            
            # Ini tetap limit 10 detik (untuk PLOT saja, bukan untuk average)
            max_samples = int(self.display_window * ShimmerConfig.MODEL_SAMPLING_RATE)
            if len(self.processed_data_buffer) > max_samples:
                excess = len(self.processed_data_buffer) - max_samples
                self.processed_data_buffer = self.processed_data_buffer[excess:]
                self.time_buffer = self.time_buffer[excess:]
            
            self._last_processed_count = current_sample_count
                    
        except Exception as e:
            print(f"Preprocessing error: {e}")

    def update_visualization(self):
        if not self.is_recording:
            return
        
        if len(self.processed_data_buffer) > 1 and len(self.time_buffer) > 1:
            time_array = np.array(self.time_buffer)
            data_array = np.array(self.processed_data_buffer)
            
            if self.processed_curve:
                self.processed_curve.setData(time_array, data_array)
            
            max_time = time_array[-1]
            self.processed_plot.setXRange(max_time - self.display_window, max_time, padding=0)
    
    def update_recording_status(self):
        if not self.is_recording:
            return
        
        elapsed = time.time() - self.recording_start_time
        elapsed_minutes = int(elapsed // 60)
        elapsed_seconds = int(elapsed % 60)
        
        total_minutes = self.recording_duration // 60
        total_seconds = self.recording_duration % 60
        
        self.timer_label.setText(f"{elapsed_minutes:02d}:{elapsed_seconds:02d} / {total_minutes:02d}:{total_seconds:02d}")
        
        progress = int((elapsed / self.recording_duration) * 100)
        self.progress_bar.setValue(min(progress, 100))
        
        sample_count = self.recording_buffer.get_sample_count()
        self.sample_count_label.setText(f"Samples: {sample_count:,}")
    
    def on_processing_progress(self, percentage, message):
        self.processing_progress_bar.setValue(percentage)
        self.processing_status_label.setText(message)
    
    def on_processing_complete(self, results):
        print("\n=== PROCESSING COMPLETE ===")
        
        processing_time = results.get('computation_time', 0)
        
        self.is_processing = False
        self.processing_results = results
        
        self.processing_widget.setVisible(False)
        
        self.connection_status.setText("● Analysis Complete")
        self.connection_status.setStyleSheet("color: #10b981; font-weight: bold; font-size: 13px;")
        
        self.final_classification_label.setText(results['final_classification'])
        self.final_classification_label.setStyleSheet(f"""
            font-size: 24px;
            font-weight: bold;
            padding: 20px;
            text-align: center;
            color: {results['classification_color']};
            background-color: {results['classification_color']}20;
            border: 3px solid {results['classification_color']};
            border-radius: 10px;
        """)
        
        self.af_count_label.setText(f"AF: {results['af_count']} windows")
        self.normal_count_label.setText(f"Normal: {results['normal_count']} windows")
        self.total_segments_label.setText(f"Total: {results['total_windows']} segments")
        self.comp_time_label.setText(f"Computation Time:\n{processing_time:.2f} seconds")
        
        print(f"Classification: {results['final_classification']}")
        print(f"AF Windows: {results['af_count']}")
        print(f"Normal Windows: {results['normal_count']}")
        print(f"AF Percentage: {results['af_percentage']:.1f}%")
        print(f"Processing Time: {processing_time:.2f}s")
        print(f"Average voltage label text: {self.avg_voltage_label.text()}")

        self.start_btn.setEnabled(True)
        self.source_combo.setEnabled(True)
        if self.is_physionet_mode:
            raw_mv = self.preprocessor.adc_to_millivolts(self.physionet_data)
            filtered_mv = self.preprocessor.preprocess_for_plot(raw_mv)
            avg_mv = np.mean(np.abs(filtered_mv))
            self.avg_voltage_label.setText(f"{avg_mv:.4f} mV")
        
        self.check_ready_state()
        
        QMessageBox.information(
            self,
            "Analysis Complete",
            f"Classification: {results['final_classification']}\n\n"
            f"AF Windows: {results['af_count']}\n"
            f"Normal Windows: {results['normal_count']}\n"
            f"AF Percentage: {results['af_percentage']:.1f}%\n\n"
            f"Processing Time: {processing_time:.2f}s"
        )
    
    def on_processing_error(self, error_msg):
        print(f"Processing error: {error_msg}")
        self.is_processing = False
        self.processing_widget.setVisible(False)
        
        self.connection_status.setText("● Processing Failed")
        self.connection_status.setStyleSheet("color: #ef4444; font-weight: bold; font-size: 13px;")
        
        QMessageBox.critical(self, "Processing Error", f"Failed to process recording:\n{error_msg}")
        self.check_ready_state()
    
    def closeEvent(self, event):
        if self.is_recording:
            reply = QMessageBox.question(
                self,
                'Recording in Progress',
                'Recording is in progress. Are you sure you want to exit?',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                if self.shimmer_reader:
                    self.shimmer_reader.stop()
                if self.batch_processor:
                    self.batch_processor.stop()
                    # Give the batch processor a short moment to exit
                    self.batch_processor.wait(1000)
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()