from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import pyqtgraph as pg
from typing import Callable

class StatusIndicator(QLabel):
    COLORS = {
        'idle': '#64748b',
        'connected': '#10b981',
        'recording': '#f59e0b',
        'processing': '#3b82f6',
        'complete': '#10b981',
        'error': '#ef4444'
    }
    
    def __init__(self, initial_text="Not Connected", initial_state='idle'):
        super().__init__(initial_text)
        self.set_state(initial_state)
    
    def set_state(self, state, text=None):
        color = self.COLORS.get(state, self.COLORS['idle'])
        display_text = text or self.text()
        self.setText(f"● {display_text}")
        self.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 13px;")

class StyledButton(QPushButton):
    COLORS = {
        'primary': '#3b82f6',
        'success': '#10b981',
        'warning': '#f59e0b',
        'danger': '#ef4444',
        'purple': '#8b5cf6'
    }
    
    def __init__(self, text, color_name='primary', height=40):
        super().__init__(text)
        color = self.COLORS.get(color_name, self.COLORS['primary'])
        self.setFixedHeight(height)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border: none;
                border-radius: 6px;
                font-weight: bold;
                font-size: 13px;
            }}
            QPushButton:hover {{
                background-color: {color}dd;
            }}
            QPushButton:pressed {{
                background-color: {color}bb;
            }}
            QPushButton:disabled {{
                background-color: #cbd5e1;
                color: #94a3b8;
            }}
        """)

class InfoCard(QWidget):
    def __init__(self, title, value="--", icon=""):
        super().__init__()
        self.setObjectName("infoBox")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        
        title_label = QLabel(f"{icon} {title}" if icon else title)
        title_label.setStyleSheet("color: #64748b; font-size: 12px;")
        layout.addWidget(title_label)
        
        self.value_label = QLabel(value)
        self.value_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #1e293b;")
        layout.addWidget(self.value_label)
    
    def set_value(self, value):
        self.value_label.setText(str(value))
    
    def set_style(self, color=None, bg_color=None):
        style = "font-size: 18px; font-weight: bold;"
        if color:
            style += f" color: {color};"
        if bg_color:
            style += f" background-color: {bg_color};"
        self.value_label.setStyleSheet(style)

class ECGPlotWidget(QWidget):
    def __init__(self, title="ECG Signal"):
        super().__init__()
        self.setObjectName("plotContainer")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        
        title_label = QLabel(title)
        title_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #1e293b;")
        layout.addWidget(title_label)
        
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel('bottom', 'Time', units='s')
        self.plot_widget.setLabel('left', 'Amplitude', units='mV')
        
        layout.addWidget(self.plot_widget)
        
        self.curve = self.plot_widget.plot(
            pen=pg.mkPen(color='#3b82f6', width=1.5)
        )
    
    def update_data(self, x_data, y_data):
        self.curve.setData(x_data, y_data)
    
    def set_x_range(self, x_min, x_max):
        self.plot_widget.setXRange(x_min, x_max, padding=0)

class ControlPanel(QGroupBox):
    def __init__(self, title):
        super().__init__(title)
        self.setObjectName("groupBox")
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
    
    def add_combo(self, label_text, items, callback=None):
        label = QLabel(label_text)
        label.setStyleSheet("font-weight: normal; font-size: 12px;")
        self.layout.addWidget(label)
        
        combo = QComboBox()
        combo.setFixedHeight(35)
        for item in items:
            if isinstance(item, tuple):
                combo.addItem(item[0], item[1])
            else:
                combo.addItem(str(item))
        
        if callback:
            combo.currentIndexChanged.connect(callback)
        
        self.layout.addWidget(combo)
        return combo
    
    def add_button(self, text, color_name='primary', callback=None):
        button = StyledButton(text, color_name)
        if callback:
            button.clicked.connect(callback)
        self.layout.addWidget(button)
        return button
    
    def add_label(self, text, style="color: #64748b; font-size: 11px; padding: 5px;"):
        label = QLabel(text)
        label.setStyleSheet(style)
        self.layout.addWidget(label)
        return label
    
    def add_stretch(self):
        self.layout.addStretch()


class ProgressWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("progressWidget")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        
        self.title_label = QLabel("Processing...")
        self.title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #1e293b;")
        layout.addWidget(self.title_label)
        
        self.status_label = QLabel("Initializing...")
        self.status_label.setStyleSheet("color: #64748b; font-size: 12px;")
        layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(25)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #e2e8f0;
                border-radius: 5px;
                text-align: center;
                background-color: white;
            }
            QProgressBar::chunk {
                background-color: #3b82f6;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.progress_bar)
    
    def update_progress(self, percentage, message):
        self.progress_bar.setValue(percentage)
        self.status_label.setText(message)
    
    def set_title(self, title):
        self.title_label.setText(title)


class RecordingControlWidget(QWidget):
    recording_started = pyqtSignal()
    recording_stopped = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        self.timer_label = QLabel("00:00 / 00:00")
        self.timer_label.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            color: #1e293b;
            padding: 10px;
        """)
        self.timer_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.timer_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(20)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #e2e8f0;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #f59e0b;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        self.sample_label = QLabel("Samples: 0")
        self.sample_label.setStyleSheet("color: #64748b; font-size: 12px;")
        self.sample_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.sample_label)
        
        button_layout = QHBoxLayout()
        
        self.start_btn = StyledButton("▶ Start Recording", 'success')
        self.start_btn.clicked.connect(self.recording_started.emit)
        button_layout.addWidget(self.start_btn)
        
        self.stop_btn = StyledButton("⏹ Stop", 'danger')
        self.stop_btn.clicked.connect(self.recording_stopped.emit)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.stop_btn)
        
        layout.addLayout(button_layout)
    
    def set_recording_state(self, is_recording):
        self.start_btn.setEnabled(not is_recording)
        self.stop_btn.setEnabled(is_recording)
    
    def update_timer(self, elapsed, total):
        elapsed_min = int(elapsed // 60)
        elapsed_sec = int(elapsed % 60)
        total_min = int(total // 60)
        total_sec = int(total % 60)
        
        self.timer_label.setText(
            f"{elapsed_min:02d}:{elapsed_sec:02d} / {total_min:02d}:{total_sec:02d}"
        )
        
        progress = min(int((elapsed / total) * 100), 100)
        self.progress_bar.setValue(progress)
    
    def update_sample_count(self, count):
        self.sample_label.setText(f"Samples: {count:,}")