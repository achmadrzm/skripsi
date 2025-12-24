import numpy as np
import pandas as pd
from collections import Counter
from scipy import signal
import wfdb
import warnings
import os
import glob
warnings.filterwarnings('ignore')

# Configuration
WINDOW_LENGTH_SEC = 10
OVERLAP_RATIO = 0.5
NORMALIZATION_METHOD = 'zscore'
DATA_DIR = 'D:\\skripsi_teknis\\dataset\\mitbih-afdb'
OUTPUT_DIR = 'D:\\skripsi_teknis\\dataset\\mitbih-afdb\\processed'

def get_available_records():
    """
    Get list of available MIT-BIH AF records dari path lokal
    """
    print(f"Scanning for records in: {DATA_DIR}")
    
    # Cari file .dat untuk mendapatkan record IDs yang tersedia
    dat_files = glob.glob(os.path.join(DATA_DIR, "*.dat"))
    
    if not dat_files:
        print(f"No .dat files found in {DATA_DIR}")
        return []
    
    available_records = []
    for dat_file in dat_files:
        # Extract record ID dari nama file (contoh: 04015.dat -> 04015)
        record_id = os.path.splitext(os.path.basename(dat_file))[0]
        
        # Pastikan file annotation (.atr) juga tersedia
        atr_file = os.path.join(DATA_DIR, f"{record_id}.atr")
        hea_file = os.path.join(DATA_DIR, f"{record_id}.hea")
        
        if os.path.exists(atr_file) and os.path.exists(hea_file):
            available_records.append(record_id)
        else:
            print(f"Warning: Missing annotation or header file for {record_id}")
    
    print(f"Found {len(available_records)} complete records")
    return sorted(available_records)

def load_ecg_data(record_id):
    """Load ECG data dari path lokal"""
    try:
        record_path = os.path.join(DATA_DIR, record_id)
        
        # Pastikan file ada
        if not os.path.exists(f"{record_path}.dat"):
            print(f"Data file not found: {record_path}.dat")
            return None, None
            
        # Load record dari path lokal
        record = wfdb.rdrecord(record_path)
        
        # Ambil lead pertama (biasanya lead I atau II)
        if record.p_signal.shape[1] > 0:
            ecg_signal = record.p_signal[:, 0]  # Lead pertama
        else:
            print(f"No signal data found in {record_id}")
            return None, None
            
        fs = record.fs
        
        print(f"Loaded {record_id}: {len(ecg_signal)} samples, fs={fs} Hz")
        return ecg_signal, fs
        
    except Exception as e:
        print(f"Error loading record {record_id}: {e}")
        return None, None

def apply_comprehensive_filtering(ecg_signal, fs):
    
    # Step 1: DC removal
    ecg_dc_removed = ecg_signal - np.mean(ecg_signal)
    
    # Step 2: Bandpass filter (0.5-40 Hz)
    nyquist = fs / 2
    low_cutoff = 0.5 / nyquist
    high_cutoff = 40.0 / nyquist
    
    bp_b, bp_a = signal.butter(4, [low_cutoff, high_cutoff], btype='band')
    ecg_bandpass = signal.filtfilt(bp_b, bp_a, ecg_dc_removed)
    
    # Step 3: Notch filter (50 Hz)
    notch_freq = 50.0 / nyquist
    notch_b, notch_a = signal.iirnotch(notch_freq, Q=25)
    ecg_filtered = signal.filtfilt(notch_b, notch_a, ecg_bandpass)
    
    return ecg_filtered

def load_and_process_annotations(record_id):
    """Load dan process rhythm annotations dari path lokal dengan debugging"""
    try:
        annotation_path = os.path.join(DATA_DIR, record_id)
        
        # Pastikan file annotation ada
        if not os.path.exists(f"{annotation_path}.atr"):
            print(f"Annotation file not found: {annotation_path}.atr")
            return None
            
        # Load annotation dari path lokal
        annotation = wfdb.rdann(annotation_path, 'atr')
        
        if hasattr(annotation, 'aux_note') and annotation.aux_note is not None:
            rhythm_labels = [label.strip() for label in annotation.aux_note]
            annotation.rhythm_labels = rhythm_labels
            
            print(f"Record {record_id}:")
            print(f"  - Total annotations: {len(rhythm_labels)}")
            print(f"  - Unique labels: {set(rhythm_labels)}")
            
            # Debugging untuk single-annotation detection
            unique_labels = set(rhythm_labels)
            if len(unique_labels) == 1:
                print(f"  - SINGLE-ANNOTATION detected: {list(unique_labels)[0]}")
            else:
                print(f"  - Multi-annotation: {len(unique_labels)} unique labels")
            
            return annotation
        else:
            print(f"No rhythm annotations found in {record_id}")
            return None
            
    except Exception as e:
        print(f"Error loading annotations for {record_id}: {e}")
        return None

def extract_af_normal_segments_enhanced(ecg_signal, annotations, fs, record_id):
    if annotations is None:
        return None, None

    # Define categories
    af_labels = {'(AFIB', 'AFIB'}
    normal_labels = {'(N', 'N', 'NSR'}
    
    rhythm_labels = annotations.rhythm_labels
    unique_labels = set(rhythm_labels)
    
    print(f"  - Processing annotations: {unique_labels}")
    
    # Classify annotations
    af_indices = [i for i, label in enumerate(rhythm_labels) if label in af_labels]
    normal_indices = [i for i, label in enumerate(rhythm_labels) if label in normal_labels]
    
    print(f"  - AF indices: {len(af_indices)}")
    print(f"  - Normal indices: {len(normal_indices)}")
    
    if not af_indices and not normal_indices:
        print("  - No AF or Normal segments found")
        return None, None
    
    # Handle single-annotation records
    if len(unique_labels) == 1:
        single_label = list(unique_labels)[0]
        print(f"  - Single-annotation processing: {single_label}")
        
        if single_label in af_labels:
            # Entire signal is AF
            binary_label = 1
            segments = [{
                'start_sample': 0,
                'end_sample': len(ecg_signal),
                'duration_sec': len(ecg_signal) / fs,
                'label': binary_label,
                'rhythm_label': single_label
            }]
            
            print(f"  - Created single AF segment: {len(ecg_signal)} samples")
            return ecg_signal, segments
            
        elif single_label in normal_labels:
            # Entire signal is Normal
            binary_label = 0
            segments = [{
                'start_sample': 0,
                'end_sample': len(ecg_signal),
                'duration_sec': len(ecg_signal) / fs,
                'label': binary_label,
                'rhythm_label': single_label
            }]
            
            print(f"  - Created single Normal segment: {len(ecg_signal)} samples")
            return ecg_signal, segments
            
        else:
            print(f"  - Single annotation '{single_label}' is not AF or Normal")
            return None, None
    
    # Handle multi-annotation records (original logic)
    keep_indices = sorted(af_indices + normal_indices)
    print(f"  - Will create {len(keep_indices)-1} segments from multi-annotation")
    
    if len(keep_indices) < 2:
        print(f"  - Insufficient annotations for segmentation (need ≥2, got {len(keep_indices)})")
        return None, None
    
    segments = []
    clean_signal_parts = []
    
    for i in range(len(keep_indices) - 1):
        current_idx = keep_indices[i]
        next_idx = keep_indices[i + 1]
        
        start_sample = annotations.sample[current_idx]
        end_sample = annotations.sample[next_idx]
        current_rhythm = rhythm_labels[current_idx]
        
        # Binary label assignment
        binary_label = 1 if current_rhythm in af_labels else 0
        
        # Extract signal segment
        if end_sample <= len(ecg_signal):
            signal_segment = ecg_signal[start_sample:end_sample]
            clean_signal_parts.append(signal_segment)
            
            duration = (end_sample - start_sample) / fs
            segments.append({
                'start_sample': start_sample,
                'end_sample': end_sample,
                'duration_sec': duration,
                'label': binary_label,
                'rhythm_label': current_rhythm
            })
    
    # Concatenate segments for multi-annotation
    if clean_signal_parts:
        clean_ecg = np.concatenate(clean_signal_parts)
        af_count = sum(1 for seg in segments if seg['label'] == 1)
        normal_count = len(segments) - af_count
        print(f"  - Extracted segments: {af_count} AF, {normal_count} Normal")
        return clean_ecg, segments
    else:
        print(f"  - No valid signal segments extracted")
        return None, None

def create_ecg_windows_enhanced(ecg_signal, segments, fs, window_length_sec=10, overlap_ratio=0.5):
    window_samples = int(window_length_sec * fs)
    step_samples = int(window_samples * (1 - overlap_ratio))
    
    windows = []
    window_labels = []
    
    print(f"  - Window parameters: {window_samples} samples, step {step_samples}")
    
    if len(segments) == 1:
        seg = segments[0]
        print(f"  - Single segment windowing: {len(ecg_signal)} samples, label {seg['label']}")
        
        start_pos = 0
        while start_pos + window_samples <= len(ecg_signal):
            window = ecg_signal[start_pos:start_pos + window_samples]
            windows.append(window)
            window_labels.append(seg['label'])
            start_pos += step_samples
        
        print(f"  - Created {len(windows)} windows from single segment")
        return np.array(windows), np.array(window_labels)
    
    print(f"  - Multi-segment windowing: {len(segments)} segments")
    
    current_pos = 0
    
    for seg_idx, seg in enumerate(segments):
        seg_length = seg['end_sample'] - seg['start_sample']
        seg_end_pos = current_pos + seg_length
        
        print(f"    Segment {seg_idx+1}: {seg_length} samples, label {seg['label']}")
        
        seg_start = current_pos
        seg_windows = 0
        
        while seg_start + window_samples <= seg_end_pos:
            window = ecg_signal[seg_start:seg_start + window_samples]
            
            windows.append(window)
            window_labels.append(seg['label'])
            
            seg_start += step_samples
            seg_windows += 1
        
        print(f"      Created {seg_windows} windows from this segment")
        current_pos = seg_end_pos
    
    print(f"  - Total windows created: {len(windows)}")
    return np.array(windows), np.array(window_labels)

def normalize_ecg_windows(windows, method='zscore'):
    if method == 'zscore':
        normalized_windows = []
        for window in windows:
            if np.std(window) > 0:
                normalized = (window - np.mean(window)) / np.std(window)
            else:
                normalized = window - np.mean(window)
            normalized_windows.append(normalized)
        normalized_windows = np.array(normalized_windows)
        
    elif method == 'minmax':
        global_min = np.min(windows)
        global_max = np.max(windows)
        normalized_windows = (windows - global_min) / (global_max - global_min)
    
    return normalized_windows

def preprocess_single_record(record_id):
    """Complete preprocessing pipeline untuk single record - enhanced version"""
    
    print(f"\n=== Processing Record {record_id} ===")
    
    # Load ECG data
    ecg_signal, fs = load_ecg_data(record_id)
    if ecg_signal is None:
        return None
    
    print(f"Loaded: {len(ecg_signal):,} samples ({len(ecg_signal)/fs/60:.1f} min)")
    
    # Apply filtering
    filtered_ecg = apply_comprehensive_filtering(ecg_signal, fs)
    print("Filtering completed")
    
    # Load annotations
    annotations = load_and_process_annotations(record_id)
    if annotations is None:
        print("No annotations found")
        return None
    
    # Extract AF/Normal segments (enhanced)
    clean_ecg, segments = extract_af_normal_segments_enhanced(filtered_ecg, annotations, fs, record_id)
    if clean_ecg is None:
        print("No AF/Normal segments found")
        return None
    
    print(f"Clean signal: {len(clean_ecg):,} samples, {len(segments)} segments")
    
    # Create windows (enhanced)
    windows, window_labels = create_ecg_windows_enhanced(
        clean_ecg, segments, fs, 
        window_length_sec=WINDOW_LENGTH_SEC, 
        overlap_ratio=OVERLAP_RATIO
    )
    
    if len(windows) == 0:
        print("No windows created")
        return None
    
    print(f"Created {len(windows)} windows")
    af_windows = np.sum(window_labels == 1)
    normal_windows = np.sum(window_labels == 0)
    print(f"AF windows: {af_windows} ({af_windows/len(window_labels)*100:.1f}%)")
    print(f"Normal windows: {normal_windows} ({normal_windows/len(window_labels)*100:.1f}%)")
    
    # Normalize windows
    normalized_windows = normalize_ecg_windows(windows, method=NORMALIZATION_METHOD)
    print("Normalization completed")
    
    # Detect record type
    unique_labels = set(annotations.rhythm_labels)
    record_type = "single_annotation" if len(unique_labels) == 1 else "multi_annotation"
    
    # Prepare output data
    processed_data = {
        'record_id': record_id,
        'record_type': record_type,
        'sampling_frequency': fs,
        'window_length_sec': WINDOW_LENGTH_SEC,
        'overlap_ratio': OVERLAP_RATIO,
        'normalization_method': NORMALIZATION_METHOD,
        'windows': normalized_windows,
        'labels': window_labels,
        'total_windows': len(window_labels),
        'af_windows': int(np.sum(window_labels == 1)),
        'normal_windows': int(np.sum(window_labels == 0)),
        'segments_info': segments,
        'annotation_labels': list(unique_labels)
    }
    
    return processed_data

def save_processed_data(processed_data, output_dir):
    """Save processed data ke file"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    record_id = processed_data['record_id']
    output_file = os.path.join(output_dir, f'record_{record_id}_processed.npz')
    
    np.savez_compressed(output_file, **processed_data)
    print(f"Saved: {output_file}")

def main():
    """Main batch processing function with enhanced single-annotation support"""
    
    print("=== Enhanced MIT-BIH AF Dataset Preprocessing ===")
    print("Now supports both single-annotation and multi-annotation records!")
    print(f"Data directory: {DATA_DIR}")
    print(f"Window length: {WINDOW_LENGTH_SEC}s")
    print(f"Overlap ratio: {OVERLAP_RATIO}")
    print(f"Normalization: {NORMALIZATION_METHOD}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Pastikan directory data ada
    if not os.path.exists(DATA_DIR):
        print(f"ERROR: Data directory '{DATA_DIR}' not found!")
        print("Please make sure the MIT-BIH AF dataset is placed in the correct directory.")
        return
    
    # Get available records
    available_records = get_available_records()
    
    if not available_records:
        print("No records found!")
        print(f"Please check if the dataset files are in '{DATA_DIR}'")
        return
    
    # Process each record
    successful_records = []
    failed_records = []
    single_annotation_records = []
    multi_annotation_records = []
    
    total_windows = 0
    total_af_windows = 0
    
    for i, record_id in enumerate(available_records):
        try:
            print(f"\nProgress: {i+1}/{len(available_records)}")
            
            # Process single record
            processed_data = preprocess_single_record(record_id)
            
            if processed_data is not None:
                # Save processed data
                save_processed_data(processed_data, OUTPUT_DIR)
                
                successful_records.append(record_id)
                total_windows += processed_data['total_windows']
                total_af_windows += processed_data['af_windows']
                
                # Categorize by record type
                if processed_data['record_type'] == 'single_annotation':
                    single_annotation_records.append(record_id)
                else:
                    multi_annotation_records.append(record_id)
                
                print(f"✓ {record_id}: {processed_data['total_windows']} windows ({processed_data['record_type']})")
            else:
                failed_records.append(record_id)
                print(f"✗ {record_id}: Failed")
                
        except Exception as e:
            failed_records.append(record_id)
            print(f"✗ {record_id}: Error - {e}")
    
    # Enhanced summary
    print(f"\n=== Enhanced Processing Complete ===")
    print(f"Successful records: {len(successful_records)}")
    print(f"  Single-annotation: {len(single_annotation_records)}")
    print(f"  Multi-annotation: {len(multi_annotation_records)}")
    print(f"Failed records: {len(failed_records)}")
    print(f"Total windows created: {total_windows:,}")
    print(f"Total AF windows: {total_af_windows:,} ({total_af_windows/total_windows*100:.1f}%)")
    
    if single_annotation_records:
        print(f"\nSingle-annotation records processed: {single_annotation_records}")
    
    if failed_records:
        print(f"Failed records: {failed_records}")
    
    # Enhanced summary data
    summary = {
        'successful_records': successful_records,
        'failed_records': failed_records,
        'single_annotation_records': single_annotation_records,
        'multi_annotation_records': multi_annotation_records,
        'total_windows': total_windows,
        'total_af_windows': total_af_windows,
        'processing_config': {
            'window_length_sec': WINDOW_LENGTH_SEC,
            'overlap_ratio': OVERLAP_RATIO,
            'normalization_method': NORMALIZATION_METHOD,
            'supports_single_annotation': True
        }
    }
    
    summary_file = os.path.join(OUTPUT_DIR, 'preprocessing_summary.npz')
    np.savez(summary_file, **summary)
    print(f"Summary saved: {summary_file}")

if __name__ == "__main__":
    main()