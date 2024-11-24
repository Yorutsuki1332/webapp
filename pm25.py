import serial

class PM25Sensor:
    def __init__(self, port='/dev/ttyS0', baudrate=9600):
        """Initialize PM2.5 sensor connection"""
        self.serial = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=8,
            parity='N',
            stopbits=1,
            timeout=2.0
        )
        self.serial.reset_input_buffer()
        self.serial.reset_output_buffer()

    def read_pm25(self):
        """Read PM2.5 instant value from sensor
        Returns:
            float: PM2.5 concentration in μg/m³, or None if reading fails
        """
        try:
            while True:
                # Look for header bytes (0x24, 0x4D)
                if self.serial.read() == b'\x24' and self.serial.read() == b'\x4d':
                    # Read frame length bytes (0x00, 0x08)
                    if self.serial.read() == b'\x00' and self.serial.read() == b'\x08':
                        # Read the 8 data bytes
                        data = self.serial.read(8)
                        
                        if len(data) != 8:
                            continue
                        
                        # Extract instant PM2.5 value (bytes 4 and 5)
                        pm25_instant = (data[4] << 8) | data[5]
                        
                        # Calculate checksum
                        checksum_data = b'\x24\x4d\x00\x08' + data[:6]
                        calculated_checksum = sum(checksum_data) & 0xFFFF
                        
                        # Get received checksum from data
                        received_checksum = (data[6] << 8) | data[7]
                        
                        # Verify checksum
                        if calculated_checksum == received_checksum:
                            # Apply coefficient of 0.4 as recommended in documentation
                            return pm25_instant * 0.4
                            
        except serial.SerialException as e:
            return None

    def close(self):
        """Close the serial connection"""
        if self.serial.is_open:
            self.serial.close()

def get_pm25_reading():
    """Utility function to get a single PM2.5 reading"""
    try:
        sensor = PM25Sensor()
        value = sensor.read_pm25()
        sensor.close()
        return value
    except:
        return None