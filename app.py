from flask import Flask, render_template, Response, send_from_directory, jsonify, g  # also , request
from smbus2 import SMBus, i2c_msg
import cv2 as cv
from datetime import datetime
from flask_socketio import SocketIO, emit
import numpy as np
import time
from tflite_runtime.interpreter import Interpreter
import sqlite3
import spidev
import pm25

app = Flask(__name__)
socketio = SocketIO(app)
DEV_ADDR_SHT4X = 0x44
VEML7700_I2C_ADDRESS = 0x10
PI25_ADDRESS = 0x48

BUS_ADDRESS = 1 
data = []


def gen_camera():
    camera = cv.VideoCapture(0)
    file_type = ".jpg"
    
    if not camera.isOpened:
        print("Error: Camera not accessible")
        return

    while True:
        ret, frame = camera.read()

        if not ret:
            print("Error: Frame not read correctly")
            break

        socketio.sleep(0.1)
        date = datetime.now()
        message_content = str(date)
        socketio.emit('datetime', {'message': message_content})
        image_resized = cv.resize(frame, down_points, interpolation= cv.INTER_LINEAR)
        socketio.sleep(0.1)
        top_k = 1
        label_id, prob = classify(interpreter, image_resized, top_k)
        labels = load_labels(labelPATH)
        MLoutput_label = labels[label_id]
        message_content = 'Class #' + str(label_id) + ', Label: ' + MLoutput_label.split(" ")[1] + ', Probability: ' + str(np.round(prob*100, 2)) + '%'
        socketio.emit('ml_label',{'message': message_content})
        
        ret, jpeg = cv.imencode(file_type, frame) 
        frame_bytes = jpeg.tobytes() 
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
        
    camera.release() 


modelDIR = "models"
modelPATH = modelDIR + "/" + "model.tflite"
labelPATH = modelDIR + "/" + "labels.txt"
interpreter = Interpreter(modelPATH)
interpreter.allocate_tensors()
_, height, width, _ = interpreter.get_input_details()[0]['shape']
down_points = (width, height) 


def load_labels(path):
    f = open(path, "r")
    return [line.strip() for line in f]

def set_input_tensor(interpreter, input_data):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0] 
    input_tensor[:, :] = input_data

def classify(interpreter, input_data, top_k):
    set_input_tensor(interpreter, input_data)
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0] 
    output_prob_q = np.squeeze(interpreter.get_tensor(output_details['index']))
    scale, zero_point = output_details['quantization']
    output_prob = scale * (output_prob_q - zero_point) 
    ordered_classes = np.argpartition(-output_prob, 1)
    return [(i, output_prob[i]) for i in ordered_classes[:top_k]][0]
### lab 3 end ###

def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect("environment_data.db", check_same_thread=False)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(error):
    db = g.pop('db', None)
    if db is not None:
        db.close()

db = sqlite3.connect("environment_data.db", check_same_thread=False)
cursor = db.cursor()
try:
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS environment_data (
            timestamp TEXT,
            temperature REAL,
            humidity REAL,
            light_level REAL,
            particle REAL
        )
    ''')
    db.commit()
except sqlite3.Error as e:
    print(f"Error creating table: {e}")

def get_latest_data():
    try:
        cursor.execute('SELECT * FROM environment_data ORDER BY timestamp DESC LIMIT 1')
        return cursor.fetchone()
    except sqlite3.Error as e:
        print(f"Error getting latest data: {e}")
        return None

def get_historical_data():
    db = get_db()
    cursor = db.cursor()
    try:
        cursor.execute('SELECT * FROM environment_data ORDER BY timestamp DESC LIMIT 100')
        return cursor.fetchall()    
    except sqlite3.Error as e:
        print(f"Error getting historical data: {e}")
        return []

def collect_data():
    [temperature, humidity] = gen_sht4x()
    light_level = gen_7700()
    particle = pm25.get_pm25_reading()
    store_data(temperature, humidity, light_level, particle)

def store_data(temperature, humidity, light_level, particle):
    db = get_db()
    cursor = db.cursor()
    try:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute('''
            INSERT INTO environment_data 
            (timestamp, temperature, humidity, light_level, particle)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (timestamp, temperature, humidity, light_level, particle))
        db.commit()
    except sqlite3.Error as e:
        print(f"Error storing data: {e}")

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


# Capture video from webcam
cap = cv.VideoCapture(0)
def face_detect():
    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

    while True:
        # Read frame
        ret, frame = cap.read()

        # Convert to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Draw rectangle around faces
        for (x, y, w, h) in faces:
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display the output
        cv.imshow('Face Detection', frame)

        # Break the loop
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

def face_recognition():
    known_image = face_recognition.load_image_file("known_person.jpg")
    known_face_encoding = face_recognition.face_encodings(known_image)[0]

    while True:
        ret, frame = cap.read()
        rgb_frame = frame[:, :, ::-1]  # Convert BGR to RGB

        # Find faces and encode them
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding in face_encodings:
            # Compare faces
            matches = face_recognition.compare_faces([known_face_encoding], face_encoding)
            if True in matches:
                print("Face Identified!")

        cv.imshow('Face Recognition', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

@app.route('/bedroom')
def readi2c():
    lux = gen_7700()
    [temp, rh] = gen_sht4x()
    particle = pm25.get_pm25_reading()
    
    store_data(
        temperature=temp,
        humidity=rh,
        light_level=lux,
        particle=particle,
    )
    
    # Get historical data for graphs
    cursor.execute('''
        SELECT timestamp, temperature, humidity, light_level, particle 
        FROM environment_data 
        WHERE timestamp >= datetime('now','local time', '-30 minutes')
        ORDER BY timestamp ASC
    ''')
    historical_data = cursor.fetchall() 


    # Prepare data for charts
    timestamps = [row[0] for row in historical_data]
    temperatures = [row[1] for row in historical_data]
    humidities = [row[2] for row in historical_data]
    light_levels = [row[3] for row in historical_data]
    particles = [row[4] for row in historical_data]
    
    sensor_data = {
        'temperature': temp,
        'rel_humidity': rh,
        'light_intensity': lux,
        'particle': particle,
        'timestamps': timestamps,
        'temperatures': temperatures,
        'humidities': humidities,
        'light_levels': light_levels,
        'particles': particles
    }
    
    return render_template("index_bedroom.html", **sensor_data)


@app.route('/video_feed')
def video_feed():                                 
    return Response(gen_camera(),
            mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return 'Hello world'

@app.route('/lab')
def lab():
    return 'Group 07'

@app.route('/sensors')
def read_i2c():
    lux = gen_7700()
    [temp, rh] = gen_sht4x()
    particle = pm25.get_pm25_reading()

    db = get_db()
    cursor = db.cursor()

    store_data(
        temperature=temp,
        humidity=rh,
        light_level=lux,
        particle=particle,
    )
    
    sensor_data = {
        'temperature': temp,
        'rel_humidity': rh,
        'light_intensity': lux,
        'particle': particle,                  
    }

    cursor.execute('SELECT * FROM environment_data ORDER BY timestamp DESC LIMIT 100')
    historical_data = cursor.fetchall()
    
    return render_template("index_sensors.html", **sensor_data, historical_data=historical_data)

@app.route('/socket')
def socket(): 
    return render_template("index_socket.html")

@app.route('/ml')
def ml():
    return render_template("index_ml.html")

@app.route('/pose')
def pose():
    return render_template('index_pose.html')

@app.route('/static/js/<path:filename>')
def send_js(filename):
    return send_from_directory('/home/pi/webapp/static/js', filename)

@app.route('/api/newdata')
def api_data():
    try:
        cursor = get_db().cursor()
        cursor.execute('SELECT * FROM environment_data ORDER BY timestamp DESC LIMIT 1')
        latest_data = cursor.fetchone()
        if latest_data is None:
            return jsonify({'error': 'No data available'}), 404
        
        return jsonify({
            'timestamp': latest_data[0],
            'temperature': latest_data[1],
            'humidity': latest_data[2],
            'light_level': latest_data[3],
            'particle': latest_data[4],
        })
    except sqlite3.Error as e:
        print(f"Error fetching data: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/h')
def farm():
    cursor.execute('SELECT * FROM environment_data ORDER BY timestamp DESC LIMIT 1')
    latest_data = cursor.fetchone()
    cursor.execute('SELECT * FROM environment_data ORDER BY timestamp DESC LIMIT 100')
    historical_data = cursor.fetchall()
    return render_template('index_home.html', latest_data=latest_data, historical_data=historical_data)

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000, use_reloader = False)
