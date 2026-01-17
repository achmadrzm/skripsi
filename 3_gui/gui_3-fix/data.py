from __future__ import annotations
import time
import signal
import sys
import csv
from serial import Serial
from pyshimmer import ShimmerBluetooth, DEFAULT_BAUDRATE, DataPacket, EChannelType

# Global variable
shim_dev = None
csv_file = None
csv_writer = None
header_written = False

TARGET_CHANNEL = EChannelType.EXG_ADS1292R_1_CH1_24BIT

def stream_cb(pkt: DataPacket) -> None:
    global csv_writer, header_written, csv_file

    # ✅ Filter hanya channel yang diinginkan
    if TARGET_CHANNEL not in pkt.channels:
        print(f"Warning: {TARGET_CHANNEL} not found in packet")
        return

    # Print to console (hanya channel target)
    print("Received new data packet:")
    print(f"channel: {TARGET_CHANNEL}")
    print(f"value: {pkt[TARGET_CHANNEL]}")
    print("")

    # --- Write to CSV ---
    if csv_writer is None:
        return

    # Write header once (hanya channel target)
    if not header_written:
        csv_writer.writerow(["timestamp", "ECG_CH1"])
        header_written = True

    # Write data row (hanya channel target)
    row = [time.time(), pkt[TARGET_CHANNEL]]
    csv_writer.writerow(row)
    csv_file.flush()


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print('\n\nStopping acquisition...')
    global shim_dev, csv_file

    if shim_dev:
        try:
            shim_dev.stop_streaming()
            shim_dev.shutdown()
            print('Shimmer stopped successfully')
        except Exception as e:
            print(f'Error stopping Shimmer: {e}')

    if csv_file:
        csv_file.close()
        print("CSV file closed")

    sys.exit(0)


def main(args=None):
    global shim_dev, csv_file, csv_writer

    signal.signal(signal.SIGINT, signal_handler)

    try:
        csv_file = open("shimmer_data.csv", "w", newline="")
        csv_writer = csv.writer(csv_file)
        print("CSV logging to shimmer_data.csv")

        serial = Serial("COM3", DEFAULT_BAUDRATE)
        shim_dev = ShimmerBluetooth(serial)
        shim_dev.initialize()

        dev_name = shim_dev.get_device_name()
        print(f"Device Name: {dev_name}")

        info = shim_dev.get_firmware_version()
        print(f"- firmware: [{info[0]}]")
        print(f"- version: [{info[1].major}.{info[1].minor}.{info[1].rel}]")

        shim_dev.add_stream_callback(stream_cb)
        shim_dev.start_streaming()

        print(f"\nStreaming ECG CH1 only... Press Ctrl+C to stop\n")

        while True:
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt caught")
        if shim_dev:
            shim_dev.stop_streaming()
            shim_dev.shutdown()

    except Exception as e:
        print(f'\nError: {e}')
        if shim_dev:
            try:
                shim_dev.stop_streaming()
                shim_dev.shutdown()
            except:
                pass

    finally:
        if csv_file:
            csv_file.close()
            print("CSV file closed")
        print("Program terminated")

if __name__ == "__main__":
    main()