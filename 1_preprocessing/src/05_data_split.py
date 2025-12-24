"""
Stratified Patient Split untuk MIT-BIH AF Dataset
Implementasi metodologi splitting yang proper untuk medical AI research
- Patient-level separation (no data leakage)
- Stratifikasi berdasarkan komposisi AF/Normal
- Balanced representation di setiap split
"""

import numpy as np
import pandas as pd
import os
import glob
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

def load_all_processed_data():
    """
    Step 1: Load semua processed data dan analyze record characteristics
    """
    print("=== Step 1: Loading dan Analyzing Processed Data ===")
    
    processed_dir = r'D:\skripsi_teknis\dataset\mitbih-afdb\processed'
    processed_files = glob.glob(os.path.join(processed_dir, 'record_*_processed.npz'))
    
    if not processed_files:
        raise FileNotFoundError(f"No processed files found in {processed_dir}")
    
    print(f"Found {len(processed_files)} processed files")
    
    record_profiles = []
    all_data = {}
    
    for file_path in processed_files:
        try:
            data = np.load(file_path, allow_pickle=True)
            
            record_id = str(data['record_id'])
            windows = data['windows']
            labels = data['labels']
            
            # Calculate record characteristics
            total_windows = len(labels)
            af_windows = np.sum(labels == 1)
            normal_windows = np.sum(labels == 0)
            af_ratio = af_windows / total_windows if total_windows > 0 else 0
            
            # Detect record type
            record_type = str(data.get('record_type', 'unknown'))
            annotation_labels = data.get('annotation_labels', [])
            
            # Store data
            all_data[record_id] = {
                'windows': windows,
                'labels': labels,
                'record_type': record_type
            }
            
            # Create record profile
            record_profile = {
                'record_id': record_id,
                'record_type': record_type,
                'total_windows': total_windows,
                'af_windows': af_windows,
                'normal_windows': normal_windows,
                'af_ratio': af_ratio,
                'annotation_labels': list(annotation_labels) if hasattr(annotation_labels, '__iter__') else [],
                'file_path': file_path
            }
            
            record_profiles.append(record_profile)
            
            print(f"  {record_id}: {total_windows:4d} windows, AF ratio: {af_ratio:.3f} ({record_type})")
            
        except Exception as e:
            print(f"  ERROR loading {file_path}: {e}")
    
    print(f"\nSuccessfully loaded {len(record_profiles)} records")
    
    # Summary statistics
    total_windows = sum(r['total_windows'] for r in record_profiles)
    total_af = sum(r['af_windows'] for r in record_profiles)
    overall_af_ratio = total_af / total_windows if total_windows > 0 else 0
    
    print(f"Dataset Summary:")
    print(f"  Total windows: {total_windows:,}")
    print(f"  AF windows: {total_af:,} ({overall_af_ratio:.1%})")
    print(f"  Normal windows: {total_windows - total_af:,} ({1-overall_af_ratio:.1%})")
    
    return record_profiles, all_data

def categorize_records(record_profiles):
    """
    Step 2: Kategorisasi records berdasarkan AF ratio untuk stratifikasi
    """
    print("\n=== Step 2: Record Categorization untuk Stratifikasi ===")
    
    # Define categorization thresholds
    AF_HEAVY_THRESHOLD = 0.7    # ≥70% AF
    NORMAL_HEAVY_THRESHOLD = 0.3  # ≤30% AF
    
    categories = {
        'af_heavy': [],      # AF-dominant records
        'normal_heavy': [],  # Normal-dominant records  
        'balanced': []       # Balanced records
    }
    
    single_annotation_count = 0
    multi_annotation_count = 0
    
    for record in record_profiles:
        af_ratio = record['af_ratio']
        record_type = record['record_type']
        
        # Count record types
        if record_type == 'single_annotation':
            single_annotation_count += 1
        else:
            multi_annotation_count += 1
        
        # Categorize by AF ratio
        if af_ratio >= AF_HEAVY_THRESHOLD:
            categories['af_heavy'].append(record)
            category = 'AF-heavy'
        elif af_ratio <= NORMAL_HEAVY_THRESHOLD:
            categories['normal_heavy'].append(record)
            category = 'Normal-heavy'
        else:
            categories['balanced'].append(record)
            category = 'Balanced'
        
        print(f"  {record['record_id']}: {af_ratio:.3f} → {category} ({record_type})")
    
    # Summary by category
    print(f"\nCategorization Summary:")
    for cat_name, records in categories.items():
        if records:
            total_windows = sum(r['total_windows'] for r in records)
            total_af = sum(r['af_windows'] for r in records)
            avg_af_ratio = total_af / total_windows if total_windows > 0 else 0
            
            single_count = sum(1 for r in records if r['record_type'] == 'single_annotation')
            multi_count = len(records) - single_count
            
            print(f"  {cat_name.replace('_', '-').title()}: {len(records)} records")
            print(f"    Windows: {total_windows:,} ({avg_af_ratio:.1%} AF)")
            print(f"    Single-annotation: {single_count}, Multi-annotation: {multi_count}")
    
    print(f"\nRecord Type Summary:")
    print(f"  Single-annotation: {single_annotation_count}")
    print(f"  Multi-annotation: {multi_annotation_count}")
    
    return categories

def stratified_patient_allocation(categories, test_size=0.2, val_size=0.2, random_seed=42):
    """
    Step 3: Stratified allocation of records ke train/val/test splits
    """
    print(f"\n=== Step 3: Stratified Patient Allocation ===")
    print(f"Target splits: Train {1-test_size-val_size:.1%}, Val {val_size:.1%}, Test {test_size:.1%}")
    
    np.random.seed(random_seed)
    
    allocated_splits = {
        'train': [],
        'val': [],
        'test': []
    }
    
    def allocate_category_records(records, category_name):
        """Allocate records dari satu kategori ke splits"""
        if not records:
            print(f"    {category_name}: No records to allocate")
            return
        
        print(f"    {category_name}: Allocating {len(records)} records")
        
        # Separate single vs multi annotation untuk better distribution
        single_ann_records = [r for r in records if r['record_type'] == 'single_annotation']
        multi_ann_records = [r for r in records if r['record_type'] != 'single_annotation']
        
        def allocate_group(group_records, group_name):
            """Allocate satu group (single/multi annotation)"""
            if not group_records:
                return
            
            # Shuffle for randomness
            shuffled = np.random.permutation(group_records).tolist()
            n_total = len(shuffled)
            
            # Calculate split sizes
            n_test = max(1, int(n_total * test_size)) if n_total > 2 else 0
            n_val = max(1, int(n_total * val_size)) if n_total > 1 else 0
            
            # Adjust if total allocation exceeds available records
            if n_test + n_val >= n_total:
                if n_total >= 3:
                    n_test, n_val = 1, 1
                elif n_total == 2:
                    n_test, n_val = 1, 0
                else:  # n_total == 1
                    n_test, n_val = 0, 0
            
            # Allocate
            test_records = shuffled[:n_test]
            val_records = shuffled[n_test:n_test + n_val]
            train_records = shuffled[n_test + n_val:]
            
            # Add to global allocation
            allocated_splits['test'].extend(test_records)
            allocated_splits['val'].extend(val_records)
            allocated_splits['train'].extend(train_records)
            
            print(f"      {group_name}: Train {len(train_records)}, Val {len(val_records)}, Test {len(test_records)}")
        
        # Allocate multi-annotation first (prioritize for val/test)
        if multi_ann_records:
            allocate_group(multi_ann_records, "Multi-annotation")
        
        # Then allocate single-annotation
        if single_ann_records:
            allocate_group(single_ann_records, "Single-annotation")
    
    # Allocate each category
    for category_name, records in categories.items():
        if records:
            allocate_category_records(records, category_name.replace('_', '-').title())
    
    # Verify allocation
    print(f"\nAllocation Results:")
    for split_name, records in allocated_splits.items():
        total_windows = sum(r['total_windows'] for r in records)
        total_af = sum(r['af_windows'] for r in records)
        af_ratio = total_af / total_windows if total_windows > 0 else 0
        
        single_count = sum(1 for r in records if r['record_type'] == 'single_annotation')
        multi_count = len(records) - single_count
        
        print(f"  {split_name.title()}: {len(records)} records, {total_windows:,} windows ({af_ratio:.1%} AF)")
        print(f"    Single-ann: {single_count}, Multi-ann: {multi_count}")
        print(f"    Records: {[r['record_id'] for r in records]}")
    
    return allocated_splits

def create_data_splits(allocated_splits, all_data):
    """
    Step 4: Create actual data splits dari allocated records
    """
    print(f"\n=== Step 4: Creating Data Splits ===")
    
    data_splits = {}
    
    for split_name, record_list in allocated_splits.items():
        if not record_list:
            print(f"  {split_name}: No records allocated")
            continue
        
        print(f"  Creating {split_name} split from {len(record_list)} records...")
        
        split_windows = []
        split_labels = []
        split_record_mapping = []
        
        for record in record_list:
            record_id = record['record_id']
            record_data = all_data[record_id]
            
            windows = record_data['windows']
            labels = record_data['labels']
            
            split_windows.append(windows)
            split_labels.append(labels)
            split_record_mapping.extend([record_id] * len(labels))
            
            af_pct = np.sum(labels == 1) / len(labels) * 100
            print(f"    {record_id}: {len(labels)} windows ({af_pct:.1f}% AF)")
        
        # Combine all windows and labels
        X_split = np.vstack(split_windows)
        y_split = np.concatenate(split_labels)
        record_mapping = np.array(split_record_mapping)
        
        # Store split data
        data_splits[split_name] = {
            'X': X_split,
            'y': y_split,
            'record_mapping': record_mapping,
            'record_ids': [r['record_id'] for r in record_list],
            'records_info': record_list
        }
        
        # Summary
        af_count = np.sum(y_split == 1)
        normal_count = np.sum(y_split == 0)
        af_ratio = af_count / len(y_split) if len(y_split) > 0 else 0
        
        print(f"    Result: {len(X_split):,} windows ({af_ratio:.1%} AF)")
    
    return data_splits

def validate_splits(data_splits):
    """
    Step 5: Validate splits untuk ensure no data leakage dan balance quality
    """
    print(f"\n=== Step 5: Split Validation ===")
    
    # Check 1: No record overlap between splits
    print("Checking for data leakage...")
    
    train_records = set(data_splits.get('train', {}).get('record_ids', []))
    val_records = set(data_splits.get('val', {}).get('record_ids', []))
    test_records = set(data_splits.get('test', {}).get('record_ids', []))
    
    leakage_checks = [
        ('Train-Val', train_records & val_records),
        ('Train-Test', train_records & test_records),
        ('Val-Test', val_records & test_records)
    ]
    
    leakage_detected = False
    for check_name, overlap in leakage_checks:
        if overlap:
            print(f"  ❌ LEAKAGE DETECTED in {check_name}: {overlap}")
            leakage_detected = True
        else:
            print(f"  ✅ {check_name}: No overlap")
    
    if not leakage_detected:
        print("  ✅ No data leakage detected!")
    
    # Check 2: Class balance analysis
    print(f"\nClass balance analysis:")
    
    af_ratios = []
    for split_name, split_data in data_splits.items():
        if 'y' in split_data:
            y = split_data['y']
            af_ratio = np.sum(y == 1) / len(y)
            af_ratios.append(af_ratio)
            
            print(f"  {split_name.title()}: {af_ratio:.3f} ({af_ratio:.1%}) AF ratio")
    
    # Check balance consistency
    if len(af_ratios) >= 2:
        max_diff = max(af_ratios) - min(af_ratios)
        print(f"  Max AF ratio difference: {max_diff:.3f}")
        
        if max_diff <= 0.15:  # 15% threshold
            print(f"  ✅ Good balance (difference ≤ 15%)")
        else:
            print(f"  ⚠️ Potential imbalance (difference > 15%)")
    
    # Check 3: Sample size adequacy
    print(f"\nSample size analysis:")
    
    min_samples_per_class = 100  # Minimum untuk reliable evaluation
    
    for split_name, split_data in data_splits.items():
        if 'y' in split_data:
            y = split_data['y']
            af_count = np.sum(y == 1)
            normal_count = np.sum(y == 0)
            
            print(f"  {split_name.title()}: AF {af_count}, Normal {normal_count}")
            
            if split_name in ['val', 'test']:
                if af_count < min_samples_per_class or normal_count < min_samples_per_class:
                    print(f"    ⚠️ Low sample count for reliable evaluation")
                else:
                    print(f"    ✅ Adequate sample sizes")
    
    return not leakage_detected

def save_stratified_splits(data_splits, allocated_splits):
    """
    Step 6: Save splits dengan comprehensive metadata
    """
    print(f"\n=== Step 6: Saving Stratified Splits ===")
    
    output_dir = r'D:\skripsi_teknis\dataset\mitbih-afdb\stratified_splits'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save data splits
    for split_name, split_data in data_splits.items():
        if 'X' in split_data and 'y' in split_data:
            X, y = split_data['X'], split_data['y']
            record_mapping = split_data['record_mapping']
            
            # Save main data
            data_file = os.path.join(output_dir, f'{split_name}_data.npz')
            np.savez_compressed(data_file, X=X, y=y, record_mapping=record_mapping)
            
            size_mb = (X.nbytes + y.nbytes) / (1024**2)
            print(f"  ✅ {split_name}_data.npz: {len(X):,} samples ({size_mb:.1f} MB)")
    
    # Save comprehensive metadata
    metadata = {
        'split_method': 'stratified_patient_split',
        'description': 'Patient-level stratified split with AF/Normal balance consideration',
        'train_records': data_splits.get('train', {}).get('record_ids', []),
        'val_records': data_splits.get('val', {}).get('record_ids', []),
        'test_records': data_splits.get('test', {}).get('record_ids', []),
        'split_stats': {}
    }
    
    # Add detailed statistics
    for split_name, split_data in data_splits.items():
        if 'y' in split_data:
            y = split_data['y']
            af_count = int(np.sum(y == 1))
            normal_count = int(np.sum(y == 0))
            
            metadata['split_stats'][split_name] = {
                'total_windows': len(y),
                'af_windows': af_count,
                'normal_windows': normal_count,
                'af_ratio': float(af_count / len(y)),
                'records': split_data.get('record_ids', [])
            }
    
    # Save metadata
    metadata_file = os.path.join(output_dir, 'stratified_split_metadata.npz')
    np.savez(metadata_file, **metadata)
    print(f"  ✅ stratified_split_metadata.npz: Comprehensive metadata saved")
    
    # Save record allocation details
    allocation_df = []
    for split_name, records in allocated_splits.items():
        for record in records:
            allocation_df.append({
                'record_id': record['record_id'],
                'split': split_name,
                'record_type': record['record_type'],
                'total_windows': record['total_windows'],
                'af_windows': record['af_windows'],
                'af_ratio': record['af_ratio']
            })
    
    if allocation_df:
        df = pd.DataFrame(allocation_df)
        csv_file = os.path.join(output_dir, 'record_allocation.csv')
        df.to_csv(csv_file, index=False)
        print(f"  ✅ record_allocation.csv: Detailed allocation table saved")
    
    print(f"\nAll files saved to: {output_dir}")
    return output_dir

def create_comprehensive_visualization(data_splits, allocated_splits):
    """
    Step 7: Create comprehensive visualizations
    """
    print(f"\n=== Step 7: Creating Visualizations ===")
    
    try:
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Split size distribution
        ax1 = plt.subplot(2, 3, 1)
        split_names = list(data_splits.keys())
        split_sizes = [len(data_splits[name]['X']) for name in split_names]
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        
        wedges, texts, autotexts = ax1.pie(split_sizes, labels=[s.capitalize() for s in split_names], 
                                          autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('Data Split Distribution\n(Total Windows)', fontweight='bold')
        
        # 2. AF percentage per split
        ax2 = plt.subplot(2, 3, 2)
        af_percentages = []
        for name in split_names:
            y = data_splits[name]['y']
            af_pct = np.sum(y == 1) / len(y) * 100
            af_percentages.append(af_pct)
        
        bars = ax2.bar([s.capitalize() for s in split_names], af_percentages, color=colors)
        ax2.set_title('AF Percentage per Split', fontweight='bold')
        ax2.set_ylabel('AF Percentage (%)')
        ax2.set_ylim(0, max(af_percentages) * 1.1)
        
        for bar, pct in zip(bars, af_percentages):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 3. Record type distribution
        ax3 = plt.subplot(2, 3, 3)
        record_type_data = defaultdict(lambda: defaultdict(int))
        
        for split_name, records in allocated_splits.items():
            for record in records:
                record_type_data[split_name][record['record_type']] += 1
        
        splits = list(record_type_data.keys())
        single_counts = [record_type_data[s].get('single_annotation', 0) for s in splits]
        multi_counts = [record_type_data[s].get('multi_annotation', 0) for s in splits]
        
        x = np.arange(len(splits))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, single_counts, width, label='Single-annotation', color='lightsteelblue')
        bars2 = ax3.bar(x + width/2, multi_counts, width, label='Multi-annotation', color='lightsalmon')
        
        ax3.set_xlabel('Split')
        ax3.set_ylabel('Number of Records')
        ax3.set_title('Record Type Distribution', fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels([s.capitalize() for s in splits])
        ax3.legend()
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{int(height)}', ha='center', va='bottom')
        
        # 4. Windows per record distribution
        ax4 = plt.subplot(2, 3, 4)
        
        all_records = []
        all_splits = []
        all_window_counts = []
        
        for split_name, records in allocated_splits.items():
            for record in records:
                all_records.append(record['record_id'])
                all_splits.append(split_name)
                all_window_counts.append(record['total_windows'])
        
        # Create box plot
        split_window_data = defaultdict(list)
        for split, count in zip(all_splits, all_window_counts):
            split_window_data[split].append(count)
        
        bp_data = [split_window_data[split] for split in split_names]
        bp = ax4.boxplot(bp_data, labels=[s.capitalize() for s in split_names], patch_artist=True)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax4.set_title('Windows per Record Distribution', fontweight='bold')
        ax4.set_ylabel('Number of Windows')
        
        # 5. AF ratio distribution
        ax5 = plt.subplot(2, 3, 5)
        
        all_af_ratios = []
        all_split_labels = []
        
        for split_name, records in allocated_splits.items():
            for record in records:
                all_af_ratios.append(record['af_ratio'])
                all_split_labels.append(split_name)
        
        # Create scatter plot
        split_colors = {'train': 'lightblue', 'val': 'lightgreen', 'test': 'lightcoral'}
        
        for split in split_names:
            split_ratios = [ratio for ratio, label in zip(all_af_ratios, all_split_labels) if label == split]
            split_indices = [i for i, label in enumerate(all_split_labels) if label == split]
            
            y_positions = [split_names.index(split)] * len(split_ratios)
            ax5.scatter(split_ratios, y_positions, alpha=0.7, 
                       c=split_colors.get(split, 'gray'), s=60, label=split.capitalize())
        
        ax5.set_xlabel('AF Ratio')
        ax5.set_ylabel('Split')
        ax5.set_title('AF Ratio Distribution per Record', fontweight='bold')
        ax5.set_yticks(range(len(split_names)))
        ax5.set_yticklabels([s.capitalize() for s in split_names])
        ax5.set_xlim(-0.1, 1.1)
        ax5.grid(True, alpha=0.3)
        
        # 6. Summary statistics table
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('tight')
        ax6.axis('off')
        
        # Create summary table
        table_data = []
        headers = ['Split', 'Records', 'Windows', 'AF %', 'Single-Ann', 'Multi-Ann']
        
        for split_name in split_names:
            split_data = data_splits[split_name]
            split_records = allocated_splits[split_name]
            
            n_records = len(split_records)
            n_windows = len(split_data['y'])
            af_pct = np.sum(split_data['y'] == 1) / len(split_data['y']) * 100
            
            single_count = sum(1 for r in split_records if r['record_type'] == 'single_annotation')
            multi_count = n_records - single_count
            
            table_data.append([
                split_name.capitalize(),
                f'{n_records}',
                f'{n_windows:,}',
                f'{af_pct:.1f}%',
                f'{single_count}',
                f'{multi_count}'
            ])
        
        table = ax6.table(cellText=table_data, colLabels=headers, 
                         cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style the table
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax6.set_title('Split Summary Statistics', fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Save visualization
        vis_file = 'stratified_splits_analysis.png'
        plt.savefig(vis_file, dpi=300, bbox_inches='tight')
        print(f"  ✅ {vis_file}: Comprehensive visualization saved")
        
        plt.show()
        
        return True
        
    except Exception as e:
        print(f"  ❌ Visualization failed: {e}")
        return False

def test_stratified_loading():
    """
    Step 8: Test loading stratified splits
    """
    print(f"\n=== Step 8: Testing Stratified Split Loading ===")
    
    splits_dir = r'D:\skripsi_teknis\dataset\mitbih-afdb\stratified_splits'
    
    try:
        # Load data splits
        train_data = np.load(os.path.join(splits_dir, 'train_data.npz'))
        val_data = np.load(os.path.join(splits_dir, 'val_data.npz'))
        test_data = np.load(os.path.join(splits_dir, 'test_data.npz'))
        
        print(f"  ✅ All stratified splits loaded successfully!")
        
        # Comprehensive data validation
        splits_info = []
        for name, data in [('Train', train_data), ('Val', val_data), ('Test', test_data)]:
            X, y = data['X'], data['y']
            record_mapping = data['record_mapping']
            
            # Basic checks
            assert len(X) == len(y) == len(record_mapping), f"{name}: Length mismatch"
            assert X.dtype in [np.float64, np.float32], f"{name}: Unexpected X dtype: {X.dtype}"
            assert set(np.unique(y)) <= {0, 1}, f"{name}: Invalid labels: {np.unique(y)}"
            
            # Statistics
            af_count = np.sum(y == 1)
            normal_count = np.sum(y == 0)
            af_pct = af_count / len(y) * 100
            unique_records = len(np.unique(record_mapping))
            
            splits_info.append({
                'split': name,
                'windows': len(X),
                'af_pct': af_pct,
                'records': unique_records
            })
            
            print(f"    {name}: {X.shape} windows, {af_pct:.1f}% AF, {unique_records} records")
        
        # Load and validate metadata
        metadata = np.load(os.path.join(splits_dir, 'stratified_split_metadata.npz'), allow_pickle=True)
        
        print(f"  ✅ Metadata loaded successfully!")
        print(f"    Split method: {metadata['split_method']}")
        
        # Verify record separation
        train_records = set(metadata['train_records'])
        val_records = set(metadata['val_records'])
        test_records = set(metadata['test_records'])
        
        # Check for overlaps
        assert len(train_records & val_records) == 0, "Train-Val record overlap detected!"
        assert len(train_records & test_records) == 0, "Train-Test record overlap detected!"
        assert len(val_records & test_records) == 0, "Val-Test record overlap detected!"
        
        print(f"  ✅ No record overlap between splits confirmed!")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Loading test failed: {e}")
        return False

def main():
    """
    Main function untuk stratified patient split
    """
    print("🔬 === Stratified Patient Split for MIT-BIH AF Dataset ===")
    print("Implementasi metodologi yang proper untuk medical AI research")
    print("- Patient-level separation (no data leakage)")
    print("- Stratified allocation berdasarkan AF/Normal composition")
    print("- Balanced representation di setiap split\n")
    
    try:
        # Step 1: Load dan analyze data
        record_profiles, all_data = load_all_processed_data()
        
        # Step 2: Categorize records
        categories = categorize_records(record_profiles)
        
        # Step 3: Stratified allocation
        allocated_splits = stratified_patient_allocation(categories)
        
        # Step 4: Create data splits
        data_splits = create_data_splits(allocated_splits, all_data)
        
        # Step 5: Validate splits
        is_valid = validate_splits(data_splits)
        
        if not is_valid:
            print("\n❌ Split validation failed! Please check the issues above.")
            return False
        
        # Step 6: Save splits
        output_dir = save_stratified_splits(data_splits, allocated_splits)
        
        # Step 7: Create visualizations
        vis_success = create_comprehensive_visualization(data_splits, allocated_splits)
        
        # Step 8: Test loading
        load_success = test_stratified_loading()
        
        if load_success:
            print(f"\n🎉 SUCCESS! Stratified Patient Split Completed!")
            print(f"\n📊 Final Summary:")
            
            total_windows = sum(len(data['X']) for data in data_splits.values())
            total_records = sum(len(records) for records in allocated_splits.values())
            
            print(f"  Total records: {total_records}")
            print(f"  Total windows: {total_windows:,}")
            
            for split_name, split_data in data_splits.items():
                y = split_data['y']
                af_pct = np.sum(y == 1) / len(y) * 100
                n_records = len(allocated_splits[split_name])
                print(f"  {split_name.title()}: {n_records} records, {len(y):,} windows ({af_pct:.1f}% AF)")
            
            print(f"\n📁 Files created in: {output_dir}")
            print(f"  - train_data.npz, val_data.npz, test_data.npz")
            print(f"  - stratified_split_metadata.npz")
            print(f"  - record_allocation.csv")
            if vis_success:
                print(f"  - stratified_splits_analysis.png")
            
            print(f"\n✅ Ready for training dengan no data leakage!")
            print(f"💡 Next step: Load splits untuk training Bi-LSTM model")
            
            # Show example loading code
            print(f"\n📝 Example loading code:")
            print(f"""
import numpy as np

# Load stratified splits
train_data = np.load('{output_dir}/train_data.npz')
val_data = np.load('{output_dir}/val_data.npz')
test_data = np.load('{output_dir}/test_data.npz')

X_train, y_train = train_data['X'], train_data['y']
X_val, y_val = val_data['X'], val_data['y']
X_test, y_test = test_data['X'], test_data['y']

print(f"Train: {{X_train.shape}}, Val: {{X_val.shape}}, Test: {{X_test.shape}}")
            """)
            
            return True
        else:
            print(f"\n❌ Loading test failed!")
            return False
            
    except Exception as e:
        print(f"\n❌ Stratified split failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print(f"\n🚀 Stratified patient split berhasil!")
        print(f"Dataset siap untuk training dengan metodologi yang proper.")
    else:
        print(f"\n💥 Ada masalah dalam proses splitting.")
        print(f"Silakan check error messages di atas dan coba lagi.")