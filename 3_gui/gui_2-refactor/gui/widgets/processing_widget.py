from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QProgressBar
from PyQt5.QtCore import Qt

class ProcessingWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("plotContainer")
        self._create_ui()
    
    def _create_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        
        title = QLabel("⚙️ Processing Recording...")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #1e293b; padding: 10px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        self.processing_status_label = QLabel("Initializing...")
        self.processing_status_label.setStyleSheet("font-size: 14px; color: #64748b; padding: 5px;")
        self.processing_status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.processing_status_label)
        
        self.processing_progress_bar = QProgressBar()
        self.processing_progress_bar.setFixedHeight(30)
        self.processing_progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #e2e8f0;
                border-radius: 5px;
                text-align: center;
                background-color: white;
                font-size: 13px;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #f59e0b;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.processing_progress_bar)
    
    def update_progress(self, percentage, status_text):
        self.processing_progress_bar.setValue(percentage)
        self.processing_status_label.setText(status_text)