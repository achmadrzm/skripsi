from serial import Serial
import serial.tools.list_ports
from PyQt5.QtCore import QThread, pyqtSignal
from pyshimmer import ShimmerBluetooth, DEFAULT_BAUDRATE, DataPacket, EChannelType

class ShimmerReader(QThread):
    data_received = pyqtSignal(float)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, port, baudrate=DEFAULT_BAUDRATE, channel=None):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.running = False
        self.shim_dev = None
        self.ecg_channel = channel if channel else EChannelType.EXG_ADS1292R_1_CH1_24BIT
        
    def stream_callback(self, pkt: DataPacket) -> None:
        try:
            if self.ecg_channel in pkt.channels:
                value = pkt[self.ecg_channel]
                self.data_received.emit(float(value))
        except Exception as e:
            self.error_occurred.emit(f"Callback error: {str(e)}")
    
    def run(self):
        try:
            serial_conn = Serial(self.port, self.baudrate)
            self.shim_dev = ShimmerBluetooth(serial_conn)
            self.shim_dev.initialize()
            dev_name = self.shim_dev.get_device_name()
            print(f"Connected to Shimmer: {dev_name}")
            self.shim_dev.add_stream_callback(self.stream_callback)
            self.shim_dev.start_streaming()
            self.running = True
            
            while self.running:
                self.msleep(100)
                
        except Exception as e:
            self.error_occurred.emit(f"Shimmer error: {str(e)}")
        finally:
            if self.shim_dev:
                try:
                    self.shim_dev.stop_streaming()
                    self.shim_dev.shutdown()
                except:
                    pass
    
    def stop(self):
        self.running = False
        self.wait()

class SerialHandler:
    @staticmethod
    def get_available_ports():
        ports = serial.tools.list_ports.comports()
        return [port.device for port in ports]