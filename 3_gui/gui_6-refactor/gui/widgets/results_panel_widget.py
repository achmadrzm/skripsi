from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel
from PyQt5.QtCore import Qt

class ResultsPanelWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("sidebar")
        self.setFixedWidth(350)
        self._create_ui()
    
    def _create_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        
        self._create_classification_group(layout)
        self._create_statistics_group(layout)
        self._create_voltage_group(layout)
        self._create_performance_group(layout)
        
        layout.addStretch()
    
    def _create_classification_group(self, parent_layout):
        result_group = QGroupBox("Classification Result")
        result_group.setObjectName("groupBox")
        result_layout = QVBoxLayout()
        
        self.final_classification_label = QLabel("--")
        self.final_classification_label.setStyleSheet("""
            font-size: 28px;
            font-weight: bold;
            padding: 20px;
            text-align: center;
            color: #64748b;
            border: 2px solid #e2e8f0;
            border-radius: 10px;
            background-color: #f8fafc;
        """)
        self.final_classification_label.setAlignment(Qt.AlignCenter)
        self.final_classification_label.setWordWrap(True)
        result_layout.addWidget(self.final_classification_label)
        
        result_group.setLayout(result_layout)
        parent_layout.addWidget(result_group)
    
    def _create_statistics_group(self, parent_layout):
        stats_group = QGroupBox("Detection Statistics")
        stats_group.setObjectName("groupBox")
        stats_layout = QVBoxLayout()
        
        af_widget = QWidget()
        af_layout = QHBoxLayout(af_widget)
        af_layout.setContentsMargins(0, 5, 0, 5)
        af_icon = QLabel("🔴")
        af_icon.setStyleSheet("font-size: 24px;")
        self.af_count_label = QLabel("AF: --")
        self.af_count_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #ef4444;")
        af_layout.addWidget(af_icon)
        af_layout.addWidget(self.af_count_label)
        af_layout.addStretch()
        stats_layout.addWidget(af_widget)
        
        normal_widget = QWidget()
        normal_layout = QHBoxLayout(normal_widget)
        normal_layout.setContentsMargins(0, 5, 0, 5)
        normal_icon = QLabel("🟢")
        normal_icon.setStyleSheet("font-size: 24px;")
        self.normal_count_label = QLabel("Normal: --")
        self.normal_count_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #10b981;")
        normal_layout.addWidget(normal_icon)
        normal_layout.addWidget(self.normal_count_label)
        normal_layout.addStretch()
        stats_layout.addWidget(normal_widget)
        
        self.total_segments_label = QLabel("Total: -- segments")
        self.total_segments_label.setStyleSheet("color: #64748b; font-size: 15px; padding-top: 10px; font-weight: bold;")
        stats_layout.addWidget(self.total_segments_label)
        
        stats_group.setLayout(stats_layout)
        parent_layout.addWidget(stats_group)
    
    def _create_voltage_group(self, parent_layout):
        voltage_group = QGroupBox("Signal Voltage")
        voltage_group.setObjectName("groupBox")
        voltage_layout = QVBoxLayout()
        
        voltage_label = QLabel("Average Voltage:")
        voltage_label.setStyleSheet("font-size: 12px; color: #64748b; font-weight: bold;")
        voltage_layout.addWidget(voltage_label)
        
        self.avg_voltage_label = QLabel("-- mV")
        self.avg_voltage_label.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            color: #3b82f6;
            background-color: #eff6ff;
            padding: 15px;
            border-radius: 8px;
            border: 2px solid #3b82f6;
        """)
        self.avg_voltage_label.setAlignment(Qt.AlignCenter)
        voltage_layout.addWidget(self.avg_voltage_label)
        
        voltage_group.setLayout(voltage_layout)
        parent_layout.addWidget(voltage_group)
    
    def _create_performance_group(self, parent_layout):
        comp_group = QGroupBox("Performance")
        comp_group.setObjectName("groupBox")
        comp_layout = QVBoxLayout()
        
        self.comp_time_label = QLabel("Computation Time:\n--")
        self.comp_time_label.setWordWrap(True)
        self.comp_time_label.setStyleSheet("background: #f1f5f9; padding: 15px; border-radius: 5px; font-size: 14px; color: #475569; font-weight: bold;")
        comp_layout.addWidget(self.comp_time_label)
        
        comp_group.setLayout(comp_layout)
        parent_layout.addWidget(comp_group)
    
    def update_classification(self, text, color):
        self.final_classification_label.setText(text)
        self.final_classification_label.setStyleSheet(f"""
            font-size: 28px;
            font-weight: bold;
            padding: 20px;
            text-align: center;
            color: {color};
            background-color: {color}20;
            border: 3px solid {color};
            border-radius: 10px;
        """)
    
    def update_statistics(self, af_count, normal_count, total):
        self.af_count_label.setText(f"AF: {af_count} windows")
        self.normal_count_label.setText(f"Normal: {normal_count} windows")
        self.total_segments_label.setText(f"Total: {total} segments")
    
    def update_voltage(self, voltage_mv):
        self.avg_voltage_label.setText(f"{voltage_mv:.4f} mV")
    
    def update_computation_time(self, time_seconds):
        self.comp_time_label.setText(f"Computation Time:\n{time_seconds:.2f} seconds")
    
    def reset(self):
        self.final_classification_label.setText("--")
        self.final_classification_label.setStyleSheet("""
            font-size: 28px;
            font-weight: bold;
            padding: 20px;
            text-align: center;
            color: #64748b;
            border: 2px solid #e2e8f0;
            border-radius: 10px;
            background-color: #f8fafc;
        """)
        self.af_count_label.setText("AF: --")
        self.normal_count_label.setText("Normal: --")
        self.total_segments_label.setText("Total: -- segments")
        self.avg_voltage_label.setText("-- mV")
        self.comp_time_label.setText("Computation Time:\n--")