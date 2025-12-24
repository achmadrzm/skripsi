from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel
from gui.ui_components import StatusIndicator

class HeaderWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("header")
        self.setFixedHeight(80)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(30, 0, 30, 0)
        
        title = QLabel("🫀 Atrial Fibrillation Detection System")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #1e293b;")
        
        self.status = StatusIndicator("Not Connected", 'idle')
        
        layout.addWidget(title)
        layout.addStretch()
        layout.addWidget(self.status)