from PyQt5.QtWidgets import QWidget, QVBoxLayout
from gui.ui_components import ECGPlotWidget

class PlotPanelWidget(QWidget):
    def __init__(self):
        super().__init__()
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        
        self.ecg_plot = ECGPlotWidget("")
        layout.addWidget(self.ecg_plot)
    
    def update_plot(self, time_data, ecg_data, x_min=None, x_max=None):
        self.ecg_plot.update_data(time_data, ecg_data)
        if x_min is not None and x_max is not None:
            self.ecg_plot.set_x_range(x_min, x_max)
    
    def clear_plot(self):
        self.ecg_plot.update_data([], [])