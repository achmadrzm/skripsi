from dataclasses import dataclass
from typing import Dict
from pyshimmer import EChannelType

@dataclass
class HardwareConfig:
    DEFAULT_BAUDRATE: int = 115200
    ECG_CHANNEL: EChannelType = EChannelType.EXG_ADS1292R_1_CH1_24BIT
    ECG_GAIN: int = 6
    ADC_OFFSET: int = 0
    
    AVAILABLE_SAMPLING_RATES: tuple = (128, 256, 360, 512)
    DEFAULT_SAMPLING_RATE: int = 128

@dataclass
class ProcessingConfig:
    MODEL_SAMPLING_RATE: int = 250
    WINDOW_DURATION_SEC: int = 10
    
    SHIMMER_CHUNK_SIZE: int = 128  # Samples per chunk dari Shimmer
    RESAMPLED_CHUNK_SIZE: int = 250  # After resampling to 250Hz
    
    @property
    def WINDOW_SIZE_SAMPLES(self) -> int:
        return self.MODEL_SAMPLING_RATE * self.WINDOW_DURATION_SEC

@dataclass
class ClassificationConfig:
    DEFAULT_MODEL_PATH: str = "best_model_full.h5"
    AF_THRESHOLD_PERCENT: float = 5.0  # 5% AF windows -> classify as AF

@dataclass  
class RecordingConfig:
    """Recording-related configurations"""
    MAX_RECORDING_DURATION_SEC: int = 600
    
    RECORDING_DURATIONS: Dict[str, int] = None
    
    def __post_init__(self):
        if self.RECORDING_DURATIONS is None:
            self.RECORDING_DURATIONS = {
                "1 minute": 60,
                "2 minutes": 120,
                "5 minutes": 300,
                "10 minutes": 600
            }

class AppConfig:
    def __init__(self):
        self.hardware = HardwareConfig()
        self.processing = ProcessingConfig()
        self.classification = ClassificationConfig()
        self.recording = RecordingConfig()
    
    @property
    def SHIMMER_SAMPLING_RATE(self):
        return self.hardware.DEFAULT_SAMPLING_RATE
    
    @property
    def MODEL_SAMPLING_RATE(self):
        return self.processing.MODEL_SAMPLING_RATE
    
    @property
    def WINDOW_SIZE_SAMPLES(self):
        return self.processing.WINDOW_SIZE_SAMPLES
    
    @property
    def WINDOW_SIZE_SECONDS(self):
        return self.processing.WINDOW_DURATION_SEC
    
    @property
    def CLASSIFICATION_THRESHOLD(self):
        return self.classification.AF_THRESHOLD_PERCENT

config = AppConfig()