import numpy as np
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtCore import QTimer

class SignalUtils:
    @staticmethod
    def format_duration(seconds):
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    @staticmethod
    def calculate_progress(elapsed, total):
        return min(int((elapsed / total) * 100), 100)
    
    @staticmethod
    def get_buffer_stats(data):
        if len(data) == 0:
            return None
        
        arr = np.array(data) if not isinstance(data, np.ndarray) else data
        return {
            'min': np.min(arr),
            'max': np.max(arr),
            'mean': np.mean(arr),
            'std': np.std(arr),
            'count': len(arr)
        }

class UIHelpers:
    @staticmethod
    def create_styled_button(text, color, height=40):
        button = QPushButton(text)
        button.setFixedHeight(height)
        button.setCursor(Qt.PointingHandCursor)
        button.setStyleSheet(f"""
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
        return button
    
    @staticmethod
    def set_status(label, text, color):
        label.setText(text)
        label.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 13px;")

class TimerManager:
    def __init__(self):
        self.timers = {}
    
    def create_timer(self, name, interval, callback):
        timer = QTimer()
        timer.setInterval(interval)
        timer.timeout.connect(callback)
        self.timers[name] = timer
        return timer
    
    def start_timer(self, name):
        if name in self.timers:
            self.timers[name].start()
    
    def stop_timer(self, name):
        if name in self.timers:
            self.timers[name].stop()
    
    def stop_all(self):
        for timer in self.timers.values():
            timer.stop()