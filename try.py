from flask import Flask, Response, render_template, jsonify, request
import cv2
import numpy as np
import threading
import base64
import json

app = Flask(__name__)

class EnhancedFaceID:
    # [Previous EnhancedFaceID class code remains exactly the same]
    # ... (keep all the existing class methods)

# Global variables
face_id = EnhancedFaceID()
camera = None
camera_lock = threading.Lock()

def get_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
    return camera

def generate_frames():
    while True:
        with camera_lock:
            camera = get_camera()
            success, frame = camera.read()
            if not success:
                continue
                
            face_rect, _ = face_id.detect_face(frame)
            if face_rect is not None:
                start_x, start_y, end_x, end_y = face_rect
                cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/register_face', methods=['POST'])
def register_face():
    user_id = request.form.get('user_id')
    if not user_id:
        return jsonify({"success": False, "message": "No user ID provided"})

    with camera_lock:
        camera = get_camera()
        success, frame = camera.read()
        if not success:
            return jsonify({"success": False, "message": "Failed to capture frame"})
        
        success, message = face_id.register_face(user_id, frame)
        return jsonify({"success": success, "message": message})

@app.route('/identify_face', methods=['POST'])
def identify_face():
    with camera_lock:
        camera = get_camera()
        success, frame = camera.read()
        if not success:
            return jsonify({"success": False, "message": "Failed to capture frame"})
        
        is_match, matched_user = face_id.identify_face(frame)
        if is_match:
            return jsonify({"success": True, "user_id": matched_user})
        return jsonify({"success": False, "message": "No match found"})

@app.route('/get_users', methods=['GET'])
def get_users():
    users = face_id.get_registered_users()
    return jsonify({"users": users})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)