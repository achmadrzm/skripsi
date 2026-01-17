from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QGroupBox, QLabel, 
                              QComboBox, QPushButton, QProgressBar)
from PyQt5.QtCore import Qt, pyqtSignal
from core.shimmer_config import config

class SidebarWidget(QWidget):
    source_changed = pyqtSignal(str)
    port_changed = pyqtSignal()
    refresh_ports_clicked = pyqtSignal()
    load_file_clicked = pyqtSignal()
    sampling_rate_changed = pyqtSignal(int)
    duration_changed = pyqtSignal(str)
    start_recording_clicked = pyqtSignal()
    stop_recording_clicked = pyqtSignal()
    reset_clicked = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.setObjectName("sidebar")
        self.setFixedWidth(320)
        self._create_ui()
    
    def _create_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        self._create_source_group(layout)
        self._create_physionet_group(layout)
        self._create_port_group(layout)
        self._create_sampling_group(layout)
        self._create_recording_group(layout)
        
        layout.addStretch()
    
    def _create_source_group(self, parent_layout):
        """Data Source Selection"""
        source_group = QGroupBox("Data Source")
        source_group.setObjectName("groupBox")
        source_layout = QVBoxLayout()
        
        self.source_combo = QComboBox()
        self.source_combo.setFixedHeight(35)
        self.source_combo.addItem("Shimmer Device", "shimmer")
        self.source_combo.addItem("PhysioNet File", "physionet")
        self.source_combo.currentIndexChanged.connect(
            lambda: self.source_changed.emit(self.source_combo.currentData())
        )
        source_layout.addWidget(self.source_combo)
        
        source_group.setLayout(source_layout)
        parent_layout.addWidget(source_group)
    
    def _create_physionet_group(self, parent_layout):
        """PhysioNet File Loader"""
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
        
        self.load_file_btn = self._create_button("📁 Load .dat File", "#8b5cf6")
        self.load_file_btn.clicked.connect(self.load_file_clicked.emit)
        physionet_layout.addWidget(self.load_file_btn)
        
        self.physionet_group.setLayout(physionet_layout)
        self.physionet_group.setVisible(False)
        parent_layout.addWidget(self.physionet_group)
    
    def _create_port_group(self, parent_layout):
        self.port_group = QGroupBox("Serial Port")
        self.port_group.setObjectName("groupBox")
        port_layout = QVBoxLayout()
        
        port_label = QLabel("Port:")
        port_label.setStyleSheet("font-weight: normal; font-size: 12px;")
        port_layout.addWidget(port_label)
        
        self.port_combo = QComboBox()
        self.port_combo.setFixedHeight(35)
        self.port_combo.currentIndexChanged.connect(self.port_changed.emit)
        port_layout.addWidget(self.port_combo)
        
        self.refresh_btn = self._create_button("🔄 Refresh Ports", "#8b5cf6")
        self.refresh_btn.clicked.connect(self.refresh_ports_clicked.emit)
        port_layout.addWidget(self.refresh_btn)
        
        self.port_status = QLabel("No port selected")
        self.port_status.setStyleSheet("color: #64748b; font-size: 11px; padding: 5px;")
        port_layout.addWidget(self.port_status)
        
        self.port_group.setLayout(port_layout)
        parent_layout.addWidget(self.port_group)
    
    def _create_sampling_group(self, parent_layout):
        self.sampling_group = QGroupBox("Sampling Rate")
        self.sampling_group.setObjectName("groupBox")
        sampling_layout = QVBoxLayout()
        
        rate_label = QLabel("Rate (Hz):")
        rate_label.setStyleSheet("font-weight: normal; font-size: 12px;")
        sampling_layout.addWidget(rate_label)
        
        self.sampling_rate_combo = QComboBox()
        self.sampling_rate_combo.setFixedHeight(35)
        
        for rate in config.hardware.AVAILABLE_SAMPLING_RATES:
            self.sampling_rate_combo.addItem(str(rate), rate)
        default_index = self.sampling_rate_combo.findData(config.hardware.DEFAULT_SAMPLING_RATE)
        
        if default_index >= 0:
            self.sampling_rate_combo.setCurrentIndex(default_index)
        self.sampling_rate_combo.currentIndexChanged.connect(
            lambda: self.sampling_rate_changed.emit(self.sampling_rate_combo.currentData())
        )
        sampling_layout.addWidget(self.sampling_rate_combo)
        
        self.sampling_group.setLayout(sampling_layout)
        parent_layout.addWidget(self.sampling_group)
    
    def _create_recording_group(self, parent_layout):
        recording_group = QGroupBox("Recording")
        recording_group.setObjectName("groupBox")
        recording_layout = QVBoxLayout()
        
        duration_label = QLabel("Duration:")
        duration_label.setStyleSheet("font-weight: normal; font-size: 12px;")
        recording_layout.addWidget(duration_label)
        
        self.duration_combo = QComboBox()
        self.duration_combo.addItems(["1 minute", "2 minutes", "5 minutes", "10 minutes"])
        self.duration_combo.setCurrentText("5 minutes")
        self.duration_combo.setFixedHeight(35)
        self.duration_combo.currentTextChanged.connect(self.duration_changed.emit)
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
        
        self.start_btn = self._create_button("▶ Start Recording", "#10b981")
        self.start_btn.clicked.connect(self.start_recording_clicked.emit)
        self.start_btn.setEnabled(False)
        recording_layout.addWidget(self.start_btn)
        
        self.stop_btn = self._create_button("⏹ Stop Recording", "#ef4444")
        self.stop_btn.clicked.connect(self.stop_recording_clicked.emit)
        self.stop_btn.setEnabled(False)
        recording_layout.addWidget(self.stop_btn)
        
        self.reset_btn = self._create_button("🔄 Reset", "#f59e0b")
        self.reset_btn.clicked.connect(self.reset_clicked.emit)
        recording_layout.addWidget(self.reset_btn)
        
        recording_group.setLayout(recording_layout)
        parent_layout.addWidget(recording_group)
    
    def _create_button(self, text, color):
        button = QPushButton(text)
        button.setFixedHeight(40)
        button.setCursor(Qt.PointingHandCursor)
        button.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border: none;
                border-radius: 6px;
                font-weight: bold;
                font-size: 13px;
            }}
            QPushButton:hover {{
                background-color: {color}dd;
            }}
            QPushButton:pressed {{
                background-color: {color}bb;
            }}
            QPushButton:disabled {{
                background-color: #cbd5e1;
                color: #94a3b8;
            }}
        """)
        return button
    
    def update_ports(self, ports):
        self.port_combo.clear()
        for port in ports:
            self.port_combo.addItem(port)
    
    def set_port_status(self, text):
        self.port_status.setText(text)
    
    def set_file_path(self, text):
        self.file_path_label.setText(text)
    
    def get_selected_port(self):
        return self.port_combo.currentText()
    
    def get_physionet_fs(self):
        return self.physionet_fs_combo.currentData()
    
    def get_sampling_rate(self):
        return self.sampling_rate_combo.currentData()