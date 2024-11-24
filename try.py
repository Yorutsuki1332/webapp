import time
import json
import board

from datetime import datetime
from flask import Flask, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)





class SensorDataManager:
    def __init__(self):
        self.MAX_HISTORY = 100
        self.DATA_FILE = 'sensor_history.json'
        self.historical_data = self.load_historical_data()

    def load_historical_data(self):
        try:
            with open(self.DATA_FILE, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def save_historical_data(self):
        with open(self.DATA_FILE, 'w') as f:
            json.dump(self.historical_data, f)

    def read_sensor_data(self):
        max_retries = 3
        for _ in range(max_retries):
            try:
                temperature = 
                humidity = 
                light = 
                
                return {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "temperature": round(temperature, 1),
                    "humidity": round(humidity, 1),
                    "pressure": round(pressure, 1)
                }
            except RuntimeError:
                # DHT22 sometimes fails to read, wait and retry
                time.sleep(2)
                continue
            except Exception as e:
                print(f"Error reading sensors: {e}")
                return None
        return None

    def update_data(self):
        new_data = self.read_sensor_data()
        if new_data:
            self.historical_data.insert(0, new_data)
            if len(self.historical_data) > self.MAX_HISTORY:
                self.historical_data = self.historical_data[:self.MAX_HISTORY]
            self.save_historical_data()
        return new_data

sensor_manager = SensorDataManager()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/latest-data')
def get_latest_data():
    latest_data = sensor_manager.update_data()
    return jsonify(latest_data if latest_data else {"error": "Failed to read sensors"})

@app.route('/historical-data')
def get_historical_data():
    return jsonify(sensor_manager.historical_data)

if __name__ == '__main__':
    # Run the server on all available network interfaces
    app.run(host='0.0.0.0', port=5000, debug=True)