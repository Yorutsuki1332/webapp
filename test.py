import json
import datetime
import random
import time
import pm25
from flask import Flask, jsonify
from flask_cors import CORS
from smbus2 import SMBus, i2c_msg

app = Flask(__name__)
CORS(app)

DEV_ADDR_SHT4X = 0x44


def gen_sht4x():
    try:
        with SMBus(1) as bus:
            bus.write_byte(DEV_ADDR_SHT4X, 0xFD)
            time.sleep(0.5)
            msg = i2c_msg.read(DEV_ADDR_SHT4X, 6)
            bus.i2c_rdwr(msg)
            data = list(msg)
            raw_t = int((data[0] << 8) | data[1])
            raw_rh = int((data[3] << 8) | data[4]) 
            rh = -6 + (125 * (raw_rh / 65535.0))
            temperature = -45 + (175 * (raw_t / 65535.0))
            return [temperature, rh]
    except Exception as e:
        print(f"Error reading SHT4X: {e}")
        return [None, None]
    
def gen_7700():
    try:
        addr = 0x10
        als_conf_0 = 0x00
        als_WH = 0x00
        als_WL = 0x00
        pow_sav = 0x00
        als = 0x04

        confValues = [0x00, 0x18] 
        interrupt_high = [0x00, 0x00] 
        interrupt_low = [0x00, 0x00]
        power_save_mode = [0x00, 0x00]

        with SMBus(1) as bus:
            bus.write_i2c_block_data(addr, als_conf_0, confValues)
            bus.write_i2c_block_data(addr, als_WH, interrupt_high)
            bus.write_i2c_block_data(addr, als_WL, interrupt_low)
            bus.write_i2c_block_data(addr, pow_sav, power_save_mode) 
            
            time.sleep(0.04)
            raw_data = bus.read_word_data(addr, als)
            
            # Convert raw data (little-endian)
            data = ((raw_data & 0xFF) << 8) | ((raw_data >> 8) & 0xFF)
            
            # Apply resolution scaling factor
            gain = 0.0036  # for ALS gain x1
            integration_time = 1  # for 100ms
            resolution = gain * integration_time
            
            lux = data * resolution
            return round(lux, 2)

    except Exception as e:
        print(f"Error reading VEML7700: {e}")
        return None

class SensorDataManager:
    

    def __init__(self):
        self.MAX_HISTORY = 100
        self.historical_data = self.load_historical_data()

    def generate_sensor_data(self):
        temp = gen_sht4x()[0]
        humidty = gen_sht4x()[1]
        light = gen_7700()
        particle = pm25.get_pm25_reading()

        return {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "temperature": temp,
            "humidity": humidty,
            "light": light,
            "particle": particle
        }

    def load_historical_data(self):
        try:
            with open('historical_data.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []

    def save_historical_data(self):
        with open('historical_data.json', 'w') as f:
            json.dump(self.historical_data, f)

    def update_data(self):
        new_data = self.generate_sensor_data()
        self.historical_data.insert(0, new_data)
        
        # Limit historical data to MAX_HISTORY entries
        if len(self.historical_data) > self.MAX_HISTORY:
            self.historical_data = self.historical_data[:self.MAX_HISTORY]
        
        self.save_historical_data()
        return new_data

sensor_manager = SensorDataManager()

@app.route('/latest-data')
def get_latest_data():
    latest_data = sensor_manager.update_data()
    return jsonify(latest_data)

@app.route('/historical-data')
def get_historical_data():
    return jsonify(sensor_manager.historical_data)

if __name__ == '__main__':
    app.run(debug=True, port=5000)