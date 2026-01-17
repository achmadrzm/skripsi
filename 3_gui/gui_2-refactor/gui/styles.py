class Styles:
    @staticmethod
    def get_main_style():
        return """
            QMainWindow {
                background-color: #f8fafc;
            }
            
            QWidget#header {
                background-color: white;
                border-bottom: 2px solid #e2e8f0;
            }
            
            QWidget#sidebar {
                background-color: white;
                border-right: 1px solid #e2e8f0;
            }
            
            QGroupBox {
                font-weight: bold;
                font-size: 13px;
                color: #1e293b;
                border: 2px solid #e2e8f0;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                background-color: white;
            }
            
            QWidget#plotContainer {
                background-color: white;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
            }
            
            QWidget#infoBox {
                background-color: white;
                border: 2px solid #e2e8f0;
                border-radius: 10px;
            }
            
            QComboBox {
                padding: 5px;
                border: 1px solid #e2e8f0;
                border-radius: 5px;
                background: white;
            }
            
            QComboBox::drop-down {
                border: none;
                padding-right: 10px;
            }
            
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #64748b;
                margin-right: 5px;
            }
            
            QComboBox QAbstractItemView {
                background-color: white;
                selection-background-color: #3b82f6;
                selection-color: white;
                border: 1px solid #e2e8f0;
            }
        """