from pyshimmer import EChannelType

class ShimmerConfig:
    DEFAULT_BAUDRATE = 115200
    
    DEFAULT_ECG_CHANNEL = EChannelType.EXG_ADS1292R_1_CH1_24BIT
    ECG_GAIN = 6
    ADC_OFFSET = 0
    
    AVAILABLE_SAMPLING_RATES = [128, 256, 360, 512]
    DEFAULT_SAMPLING_RATE = 128
    SHIMMER_SAMPLING_RATE = DEFAULT_SAMPLING_RATE
    
    MODEL_SAMPLING_RATE = 250
    WINDOW_DURATION = 10  # seconds
    WINDOW_SIZE_SAMPLES = MODEL_SAMPLING_RATE * WINDOW_DURATION  # 2500
    WINDOW_SIZE_SECONDS = 10
    
    DEFAULT_MODEL_PATH = "best_model_full.h5"
    
    MAX_RECORDING_DURATION_SEC = 600
    RECORDING_DURATIONS = {
        "1 minute": 60,
        "2 minutes": 120,
        "5 minutes": 300,
        "10 minutes": 600,
        "15 minutes": 900
    }
    
    PREPROCESSING_CHUNK_SIZE = 128
    RESAMPLED_CHUNK_SIZE = 250
    
    CLASSIFICATION_THRESHOLD = 5  # 5% AF windows for AF classification