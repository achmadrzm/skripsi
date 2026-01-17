import numpy as np
from scipy import signal

class ECGPreprocessor:
    def __init__(self, fs=250):
        self.fs = fs
        self._init_filters()

        self.v_ref = 2.42
        self.adc_bits = 24
    
    def _init_filters(self):
        nyquist = self.fs / 2
        
        # Bandpass filter (0.5-40 Hz)
        low = 0.5 / nyquist
        high = 40 / nyquist
        self.b_band, self.a_band = signal.butter(4, [low, high], btype='band')
        
        # Notch filter (50 Hz)
        notch_freq = 50.0 / nyquist
        quality_factor = 30.0
        self.b_notch, self.a_notch = signal.iirnotch(notch_freq, quality_factor)
    
    def adc_to_millivolts(self, adc_signal, gain=6, offset=0):
        adc_sensitivity = (self.v_ref * 1000) / (2 ** (self.adc_bits - 1) - 1)
        return ((adc_signal - offset) * adc_sensitivity) / gain
    
    def _apply_filters(self, signal_data):
        signal_data = signal_data - np.mean(signal_data) # Remove DC component
        signal_data = signal.filtfilt(self.b_band, self.a_band, signal_data) # Bandpass filter
        signal_data = signal.filtfilt(self.b_notch, self.a_notch, signal_data) # Notch filter
        
        return signal_data
    
    def preprocess_for_plot(self, mv_signal):
        return self._apply_filters(mv_signal)
    
    def preprocess(self, ecg_signal):
        filtered = self._apply_filters(ecg_signal) # Apply filters
        normalized = (filtered - np.mean(filtered)) / np.std(filtered) # Z-score normalization
        
        return normalized