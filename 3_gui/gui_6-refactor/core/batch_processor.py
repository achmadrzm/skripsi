import numpy as np
import time
from scipy.signal import resample
from collections import deque
from PyQt5.QtCore import QThread, pyqtSignal

class RecordingBuffer:
    def __init__(self, max_duration_seconds=600, sampling_rate=128):
        self.sampling_rate = sampling_rate
        max_samples = max_duration_seconds * sampling_rate
        self.buffer = deque(maxlen=max_samples)
        self.viz_buffer = deque(maxlen=int(10 * sampling_rate))
    
    def add_sample(self, value):
        self.buffer.append(value)
        self.viz_buffer.append(value)
    
    def get_all_data(self):
        return np.array(list(self.buffer))
    
    def get_viz_data(self):
        return np.array(list(self.viz_buffer))
    
    def get_sample_count(self):
        return len(self.buffer)
    
    def clear(self):
        self.buffer.clear()
        self.viz_buffer.clear()

class BatchProcessor(QThread):
    progress_update = pyqtSignal(int, str)
    processing_complete = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, recorded_data, preprocessor, model_handler, 
                 original_fs=128, target_fs=250, window_size=2500, 
                 af_threshold=5):
        super().__init__()
        self.recorded_data = recorded_data
        self.preprocessor = preprocessor
        self.model_handler = model_handler
        self.original_fs = original_fs
        self.target_fs = target_fs
        self.window_size = window_size
        self.af_threshold = af_threshold
        self.should_stop = False
    
    def run(self):
        try:
            # Resample
            self._update_progress(10, "Resampling data...")
            resampled_data = self._resample_data()
            
            # Preprocess
            self._update_progress(20, "Preprocessing signal...")
            preprocessed_data = self.preprocessor.preprocess(resampled_data)
            
            # Split into windows
            self._update_progress(30, "Splitting into windows...")
            windows = self._split_into_windows(preprocessed_data)
            
            if len(windows) == 0:
                raise Exception("Not enough data for analysis")
            
            # Predict
            self._update_progress(40, f"Analyzing {len(windows)} windows...")
            predictions, comp_time = self._predict_windows(windows)
            
            # Calculate results
            self._update_progress(95, "Calculating results...")
            results = self._calculate_results(predictions, preprocessed_data, comp_time)
            
            self._update_progress(100, "Complete!")
            self.processing_complete.emit(results)
            
        except Exception as e:
            self.error_occurred.emit(str(e))
    
    def _resample_data(self):
        if self.should_stop:
            return None
        
        target_length = int(len(self.recorded_data) * self.target_fs / self.original_fs)
        return resample(self.recorded_data, target_length)
    
    def _split_into_windows(self, data):
        if self.should_stop:
            return []
        
        windows = []
        for i in range(0, len(data) - self.window_size + 1, self.window_size):
            windows.append(data[i:i + self.window_size])
        return windows
    
    def _predict_windows(self, windows):
        predictions = []
        start_time = time.time()
        
        total_windows = len(windows)
        for i, window in enumerate(windows):
            if self.should_stop:
                return [], 0
            
            pred = self.model_handler.predict(window)
            predictions.append(pred[0])

            progress = 40 + int((i / total_windows) * 50)
            self._update_progress(progress, f"Window {i+1}/{total_windows}")
        
        comp_time = time.time() - start_time
        return predictions, comp_time
    
    def _calculate_results(self, predictions, processed_data, comp_time):
        if self.should_stop:
            return {}
        
        predictions = np.array(predictions)
        
        af_count = int(np.sum(predictions == 1))
        normal_count = int(np.sum(predictions == 0))
        total_windows = len(predictions)
        af_percentage = (af_count / total_windows * 100) if total_windows > 0 else 0
        
        if af_percentage >= self.af_threshold:
            final_classification = "ATRIAL FIBRILLATION"
            classification_color = "#ef4444"
        else:
            final_classification = "NORMAL"
            classification_color = "#10b981"
        
        return {
            'final_classification': final_classification,
            'classification_color': classification_color,
            'af_count': af_count,
            'normal_count': normal_count,
            'total_windows': total_windows,
            'af_percentage': af_percentage,
            'predictions': predictions.tolist(),
            'processed_data': processed_data,
            'computation_time': comp_time,
            'window_size': self.window_size,
            'sampling_rate': self.target_fs
        }
    
    def _update_progress(self, percentage, message):
        if not self.should_stop:
            self.progress_update.emit(percentage, message)
    
    def stop(self):
        self.should_stop = True
        self.wait(5000)
        if self.isRunning():
            self.terminate()
            self.wait()