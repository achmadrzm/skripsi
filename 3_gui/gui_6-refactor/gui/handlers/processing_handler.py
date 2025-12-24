from PyQt5.QtCore import QObject, pyqtSignal
from core.batch_processor import BatchProcessor
from core.shimmer_config import config

class ProcessingHandler(QObject):
    processing_started = pyqtSignal()
    progress_updated = pyqtSignal(int, str)
    processing_complete = pyqtSignal(dict)
    processing_error = pyqtSignal(str)
    
    def __init__(self, preprocessor, model_handler):
        super().__init__()
        self.preprocessor = preprocessor
        self.model_handler = model_handler
        self.processor = None
    
    def start_processing(self, recorded_data, original_fs):
        if self.processor and self.processor.isRunning():
            return
        
        self.processor = BatchProcessor(
            recorded_data=recorded_data,
            preprocessor=self.preprocessor,
            model_handler=self.model_handler,
            original_fs=original_fs,
            target_fs=config.processing.MODEL_SAMPLING_RATE,
            window_size=config.processing.WINDOW_SIZE_SAMPLES,
            af_threshold=config.classification.AF_THRESHOLD_PERCENT
        )
        
        self.processor.progress_update.connect(self.progress_updated.emit)
        self.processor.processing_complete.connect(self._on_complete)
        self.processor.error_occurred.connect(self.processing_error.emit)
        
        self.processing_started.emit()
        self.processor.start()
    
    def _on_complete(self, results):
        self.processing_complete.emit(results)
    
    def stop_processing(self):
        if self.processor:
            self.processor.stop()