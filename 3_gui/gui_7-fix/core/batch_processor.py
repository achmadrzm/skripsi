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
        self.visualization_buffer = deque(maxlen=int(10 * sampling_rate))
        
    def add_sample(self, value):
        self.buffer.append(value)
        self.visualization_buffer.append(value)
    
    def get_data(self):
        return np.array(list(self.buffer))
    
    def get_visualization_data(self):
        return np.array(list(self.visualization_buffer))
    
    def get_sample_count(self):
        return len(self.buffer)
    
    def clear(self):
        self.buffer.clear()
        self.visualization_buffer.clear()


class BatchProcessor(QThread):
    progress_update = pyqtSignal(int, str)
    processing_complete = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, recorded_data, preprocessor, model_handler, 
                 original_fs=128, target_fs=250, window_size=2500):
        super().__init__()
        self.recorded_data = recorded_data
        self.preprocessor = preprocessor
        self.model_handler = model_handler
        self.original_fs = original_fs
        self.target_fs = target_fs
        self.window_size = window_size
        self.should_stop = False
        
    def run(self):
        try:
            self.progress_update.emit(10, "Resampling data...")
            if self.should_stop:
                return
            
            target_length = int(len(self.recorded_data) * self.target_fs / self.original_fs)
            resampled_data = resample(self.recorded_data, target_length)
            
            self.progress_update.emit(20, "Preprocessing signal...")
            if self.should_stop:
                return
            
            preprocessed_data = self.preprocessor.preprocess(resampled_data)
            
            self.progress_update.emit(30, "Splitting into windows...")
            if self.should_stop:
                return
            
            windows = self.split_into_windows(preprocessed_data)
            total_windows = len(windows)
            
            if total_windows == 0:
                raise Exception("Not enough data for analysis")
            
            self.progress_update.emit(40, f"Analyzing {total_windows} windows...")
            if self.should_stop:
                return

            import time
            computation_start_time = time.time()

            # Batch predict all windows at once to reduce overhead and avoid flooding the GUI with frequent updates
            windows_array = np.stack(windows)  # shape (n_windows, window_size)
            preds = self.model_handler.predict(windows_array)
            # Ensure we have a flat list of integer predictions
            predictions = list(map(int, np.array(preds).flatten()))

            computation_time = time.time() - computation_start_time

            self.progress_update.emit(90, "Finalizing results...")
            if self.should_stop:
                return

            results = self.calculate_results(predictions)
            results['computation_time'] = computation_time
            
            self.progress_update.emit(100, "Complete!")
            self.processing_complete.emit(results)
            
        except Exception as e:
            self.error_occurred.emit(str(e))
    
    def split_into_windows(self, data):
        windows = []
        for i in range(0, len(data) - self.window_size + 1, self.window_size):
            window = data[i:i + self.window_size]
            windows.append(window)
        return windows
    
    def calculate_results(self, predictions):
        from core.shimmer_config import ShimmerConfig
        
        predictions = np.array(predictions)
        
        af_count = int(np.sum(predictions == 1))
        normal_count = int(np.sum(predictions == 0))
        total_windows = len(predictions)
        
        af_percentage = (af_count / total_windows * 100) if total_windows > 0 else 0
        
        threshold = ShimmerConfig.CLASSIFICATION_THRESHOLD
        if af_percentage >= threshold:
            final_classification = "ATRIAL FIBRILLATION"
            classification_color = "#ef4444"
        else:
            final_classification = "NORMAL"
            classification_color = "#10b981"
        
        results = {
            'final_classification': final_classification,
            'classification_color': classification_color,
            'af_count': af_count,
            'normal_count': normal_count,
            'total_windows': total_windows,
            'af_percentage': af_percentage,
            'predictions': predictions.tolist(),
            'window_size': self.window_size,
            'sampling_rate': self.target_fs
        }
        
        return results
    
    def stop(self):
        # Request cooperative stop; do NOT block the caller for long periods
        # The thread will exit on its own shortly after checking `self.should_stop`.
        self.should_stop = True
        # Avoid using terminate() which can leave resources in an inconsistent state
        # Do not call wait() here to prevent UI blocking.