from PyQt5.QtCore import QObject, pyqtSignal
import time
import numpy as np
from core.batch_processor import RecordingBuffer
from core.shimmer_config import config
from core.utils import SignalUtils

class RecordingHandler(QObject):
    recording_started = pyqtSignal()
    recording_stopped = pyqtSignal()
    timer_updated = pyqtSignal(float, float)
    sample_count_updated = pyqtSignal(int)
    duration_reached = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        
        self.buffer = RecordingBuffer(
            max_duration_seconds=config.recording.MAX_RECORDING_DURATION_SEC,
            sampling_rate=config.hardware.DEFAULT_SAMPLING_RATE
        )
        
        self.is_recording = False
        self.start_time = 0
        self.duration = 600
            
    def start_recording(self):
        if self.is_recording:
            return
        
        self.is_recording = True
        self.start_time = time.time()
        self.buffer.clear()
        self.recording_started.emit()
    
    def stop_recording(self):
        if not self.is_recording:
            return
        
        self.is_recording = False
        self.recording_stopped.emit()
    
    def add_sample(self, value):
        if self.is_recording:
            self.buffer.add_sample(value)
    
    def update_status(self):
        if not self.is_recording:
            return
        
        elapsed = time.time() - self.start_time
        
        self.timer_updated.emit(elapsed, self.duration)
        self.sample_count_updated.emit(self.buffer.get_sample_count())
        
        if elapsed >= self.duration:
            self.duration_reached.emit()
    
    def set_duration(self, seconds):
        self.duration = seconds
    
    def get_recorded_data(self):
        return self.buffer.get_all_data()
    
    def get_viz_data(self):
        return self.buffer.get_viz_data()