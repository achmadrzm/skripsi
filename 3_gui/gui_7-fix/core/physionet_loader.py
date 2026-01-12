import numpy as np
from pathlib import Path

class PhysioNetLoader:
    @staticmethod
    def load_physionet_record(file_path, sampling_rate=250):
        try:
            with open(file_path, 'rb') as f:
                raw_data = np.fromfile(f, dtype=np.int16)
            
            num_samples = len(raw_data) // 2
            signals = raw_data.reshape(num_samples, 2)
            ecg_signal = signals[:, 0].astype(np.float64)
            
            fs = sampling_rate
            duration = len(ecg_signal) / fs
            
            print(f"\n=== LOADED PHYSIONET FILE ===")
            print(f"File: {Path(file_path).name}")
            print(f"Samples: {len(ecg_signal):,}")
            print(f"Sampling Rate: {fs} Hz")
            print(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
            print(f"Value Range: [{ecg_signal.min():.0f}, {ecg_signal.max():.0f}]")
            
            return ecg_signal, fs, True, "File loaded successfully"
            
        except FileNotFoundError:
            return None, None, False, f"File not found: {file_path}"
        except Exception as e:
            return None, None, False, f"Error loading file: {str(e)}"
    
    @staticmethod
    def convert_to_shimmer_format(signals):
        scale_factor = 95.0
        offset = 195000
        scaled_signals = signals * scale_factor + offset
        
        print(f"Converted to Shimmer format:")
        print(f"  - Original range: [{signals.min():.0f}, {signals.max():.0f}]")
        print(f"  - Scaled range: [{scaled_signals.min():.0f}, {scaled_signals.max():.0f}]")
        
        return scaled_signals.astype(np.float64)