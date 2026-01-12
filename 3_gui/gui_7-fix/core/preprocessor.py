import numpy as np
from scipy import signal
from core.shimmer_config import ShimmerConfig

class ECGPreprocessor:
    def __init__(self, fs=250):
        self.fs = fs

        nyquist = self.fs / 2
        low = 0.5 / nyquist
        high = 40 / nyquist
        self.b_band, self.a_band = signal.butter(4, [low, high], btype='band')
        notch_freq = 50.0 / nyquist
        quality_factor = 30.0
        self.b_notch, self.a_notch = signal.iirnotch(notch_freq, quality_factor)
        
        self.v_ref = 2.42
        self.adc_bits = 24

    def adc_to_millivolts(self, adc_signal, gain=None, offset=None):
        if gain is None:
            gain = ShimmerConfig.ECG_GAIN
        if offset is None:
            offset = ShimmerConfig.ADC_OFFSET

        adc_sensitivity = (self.v_ref * 1000) / (2 ** (self.adc_bits - 1) - 1)
        mv_signal = ((adc_signal - offset) * adc_sensitivity) / gain
        return mv_signal
    
    def preprocess_for_plot(self, mv_signal):
        mv_signal = mv_signal - np.mean(mv_signal) # DC Removal
        mv_signal = signal.filtfilt(self.b_band, self.a_band, mv_signal) # Bandpass Filter (0.5-40 Hz)
        mv_signal = signal.filtfilt(self.b_notch, self.a_notch, mv_signal) # Notch Filter (50 Hz)
        
        return mv_signal
    
    def preprocess(self, ecg_signal):
        ecg_signal = ecg_signal - np.mean(ecg_signal) # DC Removal
        ecg_signal = signal.filtfilt(self.b_band, self.a_band, ecg_signal) # Bandpass Filter (0.5-40 Hz)
        ecg_signal = signal.filtfilt(self.b_notch, self.a_notch, ecg_signal) # Notch Filter (50 Hz)
        ecg_signal = (ecg_signal - np.mean(ecg_signal)) / np.std(ecg_signal) # Z-score Normalization
        
        return ecg_signal