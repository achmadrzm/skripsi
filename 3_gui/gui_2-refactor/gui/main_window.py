from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QMessageBox
from PyQt5.QtCore import QTimer
from gui.widgets.header_widget import HeaderWidget
from gui.widgets.sidebar_widget import SidebarWidget
from gui.widgets.plot_panel_widget import PlotPanelWidget
from gui.widgets.results_panel_widget import ResultsPanelWidget
from gui.handlers.recording_handler import RecordingHandler
from gui.handlers.processing_handler import ProcessingHandler
from gui.styles import Styles
from core.preprocessor import ECGPreprocessor
from core.model_handler import ModelHandler
from core.serial_handler import SerialHandler, ShimmerReader
from core.physionet_loader import PhysioNetLoader
from core.shimmer_config import config
from collections import deque
from scipy.signal import resample
import numpy as np
import os

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AF Detection System")
        self.setGeometry(100, 100, 1600, 900)
        
        self._init_core_components()
        self._init_handlers()
        self._init_ui()
        self._connect_signals()
        self._setup_timers()
        self.setStyleSheet(Styles.get_main_style())
        self._refresh_ports()
    
    def _init_core_components(self):
        self.preprocessor = ECGPreprocessor(fs=config.processing.MODEL_SAMPLING_RATE)
        self.model_handler = ModelHandler()
        
        model_path = config.classification.DEFAULT_MODEL_PATH
        if os.path.exists(model_path):
            self.model_handler.load_model(model_path)
        
        self.is_shimmer_mode = True
        self.physionet_data = None
        self.physionet_fs = None
        self.recording_duration = 600
        
        self.shimmer_reader = None
        
        self.time_buffer = []
        self.data_buffer = []
        self.current_time = 0

        max_samples = int(10 * config.processing.MODEL_SAMPLING_RATE)
        self.time_buffer = deque(maxlen=max_samples)
        self.data_buffer = deque(maxlen=max_samples)
        self.current_time = 0
        self._last_processed_count = 0
    
    def _init_handlers(self):
        self.recording_handler = RecordingHandler()
        self.processing_handler = ProcessingHandler(
            self.preprocessor,
            self.model_handler
        )
    
    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        
        layout = QVBoxLayout(central)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.header = HeaderWidget()
        layout.addWidget(self.header)
        
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        
        self.sidebar = SidebarWidget()
        content_layout.addWidget(self.sidebar, stretch=0)
        
        self.plot_panel = PlotPanelWidget()
        content_layout.addWidget(self.plot_panel, stretch=3)
        
        self.results_panel = ResultsPanelWidget()
        content_layout.addWidget(self.results_panel, stretch=0)
        
        layout.addLayout(content_layout)
    
    def _connect_signals(self):
        self.sidebar.source_changed.connect(self._on_source_changed)
        self.sidebar.port_changed.connect(self._on_port_changed)
        self.sidebar.refresh_ports_clicked.connect(self._refresh_ports)
        self.sidebar.load_file_clicked.connect(self._load_physionet_file)
        self.sidebar.sampling_rate_changed.connect(self._on_sampling_rate_changed)
        self.sidebar.duration_changed.connect(self._on_duration_changed)
        self.sidebar.start_recording_clicked.connect(self._start_recording)
        self.sidebar.stop_recording_clicked.connect(self._stop_recording)
        self.sidebar.reset_clicked.connect(self._reset_all)
        
        self.recording_handler.recording_started.connect(self._on_recording_started)
        self.recording_handler.recording_stopped.connect(self._on_recording_stopped)
        self.recording_handler.timer_updated.connect(self._on_timer_update)
        self.recording_handler.sample_count_updated.connect(self._on_sample_count_update)
        self.recording_handler.duration_reached.connect(self._stop_recording)
        
        self.processing_handler.processing_started.connect(self._on_processing_started)
        self.processing_handler.progress_updated.connect(self._on_progress_update)
        self.processing_handler.processing_complete.connect(self._on_processing_complete)
        self.processing_handler.processing_error.connect(self._on_processing_error)
    
    def _setup_timers(self):
        self.recording_timer = QTimer()
        self.recording_timer.setInterval(100)
        self.recording_timer.timeout.connect(self.recording_handler.update_status)
        
        self.viz_timer = QTimer()
        self.viz_timer.setInterval(50)
        self.viz_timer.timeout.connect(self._update_visualization)
        
        self.preprocess_timer = QTimer()
        self.preprocess_timer.setInterval(200)
        self.preprocess_timer.timeout.connect(self._preprocess_chunk)
    
    def _on_source_changed(self, source):
        self.is_shimmer_mode = (source == "shimmer")
        
        if self.is_shimmer_mode:
            self.sidebar.physionet_group.setVisible(False)
            self.sidebar.port_group.setVisible(True)
            self.sidebar.sampling_group.setVisible(True)
        else:
            self.sidebar.port_group.setVisible(False)
            self.sidebar.sampling_group.setVisible(False)
            self.sidebar.physionet_group.setVisible(True)
    
    def _on_port_changed(self):
        port = self.sidebar.get_selected_port()
        if port:
            self.sidebar.set_port_status(f"Selected: {port}")
    
    def _on_sampling_rate_changed(self, rate):
        print(f"Sampling rate changed to: {rate} Hz")
        if hasattr(self, 'recording_handler'):
            self.recording_handler.buffer.sampling_rate = rate
    
    def _refresh_ports(self):
        ports = SerialHandler.get_available_ports()
        self.sidebar.update_ports(ports)
        
        if ports:
            self.sidebar.set_port_status(f"Found {len(ports)} port(s)")
            self.sidebar.start_btn.setEnabled(True)
        else:
            self.sidebar.set_port_status("No ports found")
            self.sidebar.start_btn.setEnabled(False)
    
    def _load_physionet_file(self):
        from PyQt5.QtWidgets import QFileDialog
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open PhysioNet File", "", "DAT Files (*.dat)"
        )
        
        if file_path:
            fs = self.sidebar.get_physionet_fs()
            data, actual_fs, success, msg = PhysioNetLoader.load_physionet_record(file_path, fs)
            
            if success:
                self.physionet_data = data
                self.physionet_fs = actual_fs
                self.sidebar.set_file_path(f"Loaded: {os.path.basename(file_path)}")
                self.sidebar.start_btn.setEnabled(True)
                self.header.status.set_state('connected', 'File Loaded')
            else:
                QMessageBox.critical(self, "Error", msg)
    
    def _on_duration_changed(self, duration_text):
        duration_map = config.recording.RECORDING_DURATIONS
        
        self.recording_duration = duration_map.get(duration_text, 300)
        self.recording_handler.set_duration(self.recording_duration)
        
        minutes = self.recording_duration // 60
        seconds = self.recording_duration % 60
        self.sidebar.timer_label.setText(f"00:00 / {minutes:02d}:{seconds:02d}")
    
    def _start_recording(self):
        if self.is_shimmer_mode:
            port = self.sidebar.get_selected_port()
            if not port:
                QMessageBox.warning(self, "Warning", "Please select a port")
                return
            
            try:
                self.shimmer_reader = ShimmerReader(
                    port, 
                    config.hardware.DEFAULT_BAUDRATE
                )
                self.shimmer_reader.data_received.connect(self.recording_handler.add_sample)
                self.shimmer_reader.error_occurred.connect(self._on_shimmer_error)
                self.shimmer_reader.start()
                
                self.header.status.set_state('connected', 'Shimmer Connected')
                
            except Exception as e:
                error_msg = f"Connection failed: {str(e)}"
                QMessageBox.critical(self, "Connection Error", error_msg)
                print(f"✗ {error_msg}")
                return
        
        else:
            if self.physionet_data is None:
                QMessageBox.warning(self, "Warning", "Please load a PhysioNet file first")
                return
            
            print("Using PhysioNet file for recording simulation...")
        
        self.recording_handler.start_recording()
    
    def _stop_recording(self):
        self.recording_handler.stop_recording()
        
        if self.shimmer_reader:
            self.shimmer_reader.stop()
            self.shimmer_reader = None
    
    def _reset_all(self):
        if self.recording_handler.is_recording:
            self._stop_recording()
        
        self.time_buffer.clear()
        self.data_buffer.clear()
        self.current_time = 0
        
        self.plot_panel.clear_plot()
        self.results_panel.reset()
        
        self.header.status.set_state('idle', 'Ready')
        
        self.sidebar.timer_label.setText("00:00 / 05:00")
        self.sidebar.progress_bar.setValue(0)
        self.sidebar.sample_count_label.setText("Samples: 0")
    
    # ============ Recording Events ============
    
    def _on_recording_started(self):
        self.header.status.set_state('recording', 'Recording')
        self.sidebar.start_btn.setEnabled(False)
        self.sidebar.stop_btn.setEnabled(True)
        self.results_panel.reset()
        self.time_buffer.clear()
        self.data_buffer.clear()
        self.current_time = 0
        
        self.recording_timer.start()
        self.viz_timer.start()
        self.preprocess_timer.start()
    
    def _on_recording_stopped(self):
        self.recording_timer.stop()
        self.viz_timer.stop()
        self.preprocess_timer.stop()
        
        self.header.status.set_state('processing', 'Processing')
        self.sidebar.start_btn.setEnabled(True)
        self.sidebar.stop_btn.setEnabled(False)
        
        recorded_data = self.recording_handler.get_recorded_data()
        
        if len(recorded_data) == 0:
            QMessageBox.warning(self, "Warning", "No data recorded")
            self.header.status.set_state('idle', 'Ready')
            return
        
        original_fs = self.physionet_fs if not self.is_shimmer_mode else config.hardware.DEFAULT_SAMPLING_RATE
        self.processing_handler.start_processing(recorded_data, original_fs)
    
    def _on_timer_update(self, elapsed, total):
        elapsed_min = int(elapsed // 60)
        elapsed_sec = int(elapsed % 60)
        total_min = int(total // 60)
        total_sec = int(total % 60)
        
        self.sidebar.timer_label.setText(
            f"{elapsed_min:02d}:{elapsed_sec:02d} / {total_min:02d}:{total_sec:02d}"
        )
        
        progress = min(int((elapsed / total) * 100), 100)
        self.sidebar.progress_bar.setValue(progress)
    
    def _on_sample_count_update(self, count):
        self.sidebar.sample_count_label.setText(f"Samples: {count:,}")
    
    def _preprocess_chunk(self):
        if not self.recording_handler.is_recording:
            return
        
        current_sample_count = self.recording_handler.buffer.get_sample_count()
        chunk_size = 128
        if current_sample_count - self._last_processed_count < chunk_size:
            return
        
        viz_data = self.recording_handler.get_viz_data()
        if len(viz_data) < chunk_size:
            return
        
        try:
            chunk = viz_data[-chunk_size:]
            
            resampled = resample(chunk, 250)
            
            mv_data = self.preprocessor.adc_to_millivolts(
                resampled,
                gain=config.hardware.ECG_GAIN,
                offset=config.hardware.ADC_OFFSET
            )
            processed = self.preprocessor.preprocess_for_plot(mv_data)
            
            for val in processed:
                self.data_buffer.append(val)
                self.time_buffer.append(self.current_time)
                self.current_time += 1 / config.processing.MODEL_SAMPLING_RATE
            
            self._last_processed_count = current_sample_count
            
        except Exception as e:
            print(f"Preprocessing error: {e}")
    
    def _update_visualization(self):
        if len(self.data_buffer) > 1 and len(self.time_buffer) > 1:
            time_arr = np.array(list(self.time_buffer))
            data_arr = np.array(list(self.data_buffer))
            
            max_time = time_arr[-1]
            self.plot_panel.update_plot(
                time_arr, data_arr,
                max_time - 10, max_time
            )
    
    def _on_processing_started(self):
        pass
    
    def _on_progress_update(self, percentage, message):
        print(f"Processing: {percentage}% - {message}")
    
    def _on_processing_complete(self, results):
        self.processing_widget.setVisible(False)
        self.header.status.set_state('complete', 'Analysis Complete')
        
        self.results_panel.update_classification(
            results['final_classification'],
            results['classification_color']
        )
        
        self.results_panel.update_statistics(
            results['af_count'],
            results['normal_count'],
            results['total_windows']
        )
        
        self.results_panel.update_computation_time(
            results['computation_time']
        )
        
        try:
            recorded_data = self.recording_handler.get_recorded_data()
            
            if len(recorded_data) > 0:
                if self.is_shimmer_mode:
                    original_fs = self.sidebar.get_sampling_rate()
                else:
                    original_fs = self.physionet_fs
                
                sample_size = min(1000, len(recorded_data))
                sample_data = recorded_data[:sample_size]
                
                from scipy.signal import resample
                target_length = int(len(sample_data) * config.processing.MODEL_SAMPLING_RATE / original_fs)
                resampled = resample(sample_data, target_length)
                
                mv_data = self.preprocessor.adc_to_millivolts(
                    resampled,
                    gain=config.hardware.ECG_GAIN,
                    offset=config.hardware.ADC_OFFSET
                )
                avg_voltage = np.mean(np.abs(mv_data))
                self.results_panel.update_voltage(avg_voltage)
                
                print(f"Average Voltage: {avg_voltage:.4f} mV")
            else:
                self.results_panel.update_voltage(0.0)
                
        except Exception as e:
            print(f"⚠ Error calculating voltage: {e}")
            import traceback
            traceback.print_exc()
            self.results_panel.update_voltage(0.0)
        
        print(f"Classification: {results['final_classification']}")
        print(f"AF Windows: {results['af_count']}")
        print(f"Normal Windows: {results['normal_count']}")
        print(f"AF Percentage: {results['af_percentage']:.1f}%")
        print(f"Processing Time: {results['computation_time']:.2f}s")
        
        QMessageBox.information(
            self, "Analysis Complete",
            f"Classification: {results['final_classification']}\n\n"
            f"AF Windows: {results['af_count']}\n"
            f"Normal Windows: {results['normal_count']}\n"
            f"AF Percentage: {results['af_percentage']:.1f}%\n\n"
            f"Processing Time: {results['computation_time']:.2f}s"
        )
    
    def _on_processing_error(self, error_msg):
        self.header.status.set_state('error', 'Processing Failed')
        QMessageBox.critical(self, "Error", f"Processing failed:\n{error_msg}")
    
    def _on_shimmer_error(self, error_msg):
        QMessageBox.critical(self, "Shimmer Error", error_msg)
        if self.recording_handler.is_recording:
            self._stop_recording()
    
    def closeEvent(self, event):
        if self.recording_handler.is_recording:
            reply = QMessageBox.question(
                self, 'Recording in Progress',
                'Recording is in progress. Exit anyway?',
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                if self.shimmer_reader:
                    self.shimmer_reader.stop()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()