import face_recognition
import cv2
import os
import numpy as np
import time

def encode_faces(image_folder):
    known_face_encodings = []
    known_face_names = []

    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(image_folder, filename)
            image = face_recognition.load_image_file(image_path)
            image.astype('uint8')
            encoding = face_recognition.face_encodings(image)[0]
            
            known_face_encodings.append(encoding)
            known_face_names.append(os.path.splitext(filename)[0])  # Use filename as name

    return known_face_encodings, known_face_names

# Example usage
image_folder = 'C:/webapp/face'
known_face_encodings, known_face_names = encode_faces(image_folder)

def recognize_faces(known_face_encodings, known_face_names, recognition_duration=10):
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print("Error: Could not open video capture.")
        return
    
    start_time = time.time()  # Record the start time

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Check if the frame is in the correct shape
        if frame is not None and frame.ndim == 3 and frame.shape[2] == 3:
            rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])

            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            print(f"Detected {len(face_locations)} face(s)")

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                if True in matches:
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            cv2.imshow('Video', frame)

        else:
            print("Invalid frame received")

        # Check if the recognition duration has been exceeded
        if time.time() - start_time > recognition_duration:
            print("Recognition duration exceeded. Exiting...")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

# Example usage
recognize_faces(known_face_encodings, known_face_names, recognition_duration=10)