from flask import Flask, render_template, jsonify
import sqlite3
import random
from datetime import datetime
import threading
import time

def init_db():
    conn = sqlite3.connect('sensor_data.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS sensor_data
                 (timestamp TEXT, temperature REAL, humidity REAL, 
                  light_level REAL, particle_level REAL)''')
    conn.commit()
    conn.close()

@app.route('/get_historical_data')
def get_historical_data():
    conn = sqlite3.connect('sensor_data.db')
    c = conn.cursor()
    c.execute('''SELECT * FROM sensor_data ORDER BY timestamp DESC LIMIT 100''')
    data = c.fetchall()
    conn.close()
    
    return jsonify([{
        'timestamp': row[0],
        'temperature': row[1],
        'humidity': row[2],
        'light_level': row[3],
        'particle_level': row[4]
    } for row in data])
    
def collect_sensor_data():
    while True:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        [temperature, humidity] = gen_sht4x()
        light_level =  gen_7700()
        particle_level = pm25.get_pm25_reading()

        conn = sqlite3.connect('sensor_data.db')
        c = conn.cursor()
        c.execute('''INSERT INTO sensor_data VALUES (?, ?, ?, ?, ?)''',
                 (timestamp, temperature, humidity, light_level, particle_level))
        conn.commit()
        conn.close()
        time.sleep(5)

@app.route('/get_current_data')
def get_current_data():
    conn = sqlite3.connect('sensor_data.db')
    c = conn.cursor()
    c.execute('''SELECT * FROM sensor_data ORDER BY timestamp DESC LIMIT 1''')
    data = c.fetchone()
    conn.close()
    
    if data:
        return jsonify({
            'timestamp': data[0],
            'temperature': data[1],
            'humidity': data[2],
            'light_level': data[3],
            'particle_level': data[4]
        })
    return jsonify({})

@app.route('/sensors')
def index():
    return render_template('index_sensors.html')

