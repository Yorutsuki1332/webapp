import cv2
import numpy as np

class EnhancedFaceID:
    def __init__(self):
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_database = {}
        self.recognition_threshold = 0.65
        self.max_users = 10  # Maximum number of users that can be registered
        
    def detect_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )
        
        if len(faces) == 0:
            return None, 0
            
        best_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = best_face
        face_rect = (x, y, x + w, y + h)
        confidence = min((w * h) / (frame.shape[0] * frame.shape[1]) * 4, 1.0)
        
        return face_rect, confidence

    def extract_features(self, face_img):
        if face_img is None:
            return None
            
        face_img = cv2.resize(face_img, (100, 100))
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        features = []
        
        # LBP-like features
        for i in range(0, gray.shape[0]-2, 3):
            for j in range(0, gray.shape[1]-2, 3):
                patch = gray[i:i+3, j:j+3]
                center = patch[1, 1]
                pattern = (patch >= center).astype(int)
                features.append(np.mean(pattern))
        
        # Gradient features
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag, angle = cv2.cartToPolar(gx, gy)
        features.extend(mag.flatten()[::16])
        
        # HOG-like features
        cell_size = 10
        cells_x = gray.shape[1] // cell_size
        cells_y = gray.shape[0] // cell_size
        
        for i in range(cells_y):
            for j in range(cells_x):
                cell = gray[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
                features.append(np.mean(cell))
                features.append(np.std(cell))
        
        return np.array(features)

    def verify_face_quality(self, frame, face_rect):
        if face_rect is None:
            return False
            
        start_x, start_y, end_x, end_y = face_rect
        face_img = frame[start_y:end_y, start_x:end_x]
        
        min_face_size = 60
        if (end_x - start_x) < min_face_size or (end_y - start_y) < min_face_size:
            return False
            
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        if brightness < 40 or brightness > 250:
            return False
            
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 100:
            return False
            
        return True

    def compare_features(self, features1, features2):
        if features1 is None or features2 is None:
            return 0
            
        features1_norm = features1 / (np.linalg.norm(features1) + 1e-7)
        features2_norm = features2 / (np.linalg.norm(features2) + 1e-7)
        similarity = np.dot(features1_norm, features2_norm)
        
        return similarity

    def register_face(self, user_id, frame):
        if len(self.face_database) >= self.max_users:
            return False, f"Maximum number of users ({self.max_users}) reached"
            
        if user_id in self.face_database:
            return False, "User ID already exists"
            
        face_rect, confidence = self.detect_face(frame)
        if face_rect is None:
            return False, "No face detected"
            
        if not self.verify_face_quality(frame, face_rect):
            return False, "Face quality too low"
            
        start_x, start_y, end_x, end_y = face_rect
        face_img = frame[start_y:end_y, start_x:end_x]
        
        features = self.extract_features(face_img)
        if features is None:
            return False, "Feature extraction failed"
            
        self.face_database[user_id] = {
            'features': features,
            'confidence': confidence
        }
        
        remaining_slots = self.max_users - len(self.face_database)
        return True, f"Face registered successfully (confidence: {confidence:.2f}). {remaining_slots} slots remaining"

    def identify_face(self, frame):
        if not self.face_database:
            return None, "No faces in database"
            
        face_rect, confidence = self.detect_face(frame)
        if face_rect is None:
            return None, "No face detected"
            
        if not self.verify_face_quality(frame, face_rect):
            return None, "Face quality too low"
            
        start_x, start_y, end_x, end_y = face_rect
        face_img = frame[start_y:end_y, start_x:end_x]
        
        features = self.extract_features(face_img)
        if features is None:
            return None, "Feature extraction failed"
            
        best_match = None
        best_similarity = 0
        
        for user_id, data in self.face_database.items():
            similarity = self.compare_features(features, data['features'])
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = user_id
                
        if best_similarity >= self.recognition_threshold:
            return best_match, f"Matched with confidence: {best_similarity:.2f}"
        return None, "No match found"

    def get_registered_users(self):
        """Get list of registered users"""
        return list(self.face_database.keys())

def draw_face_info(frame, face_rect, text=""):
    if face_rect is None:
        return
        
    start_x, start_y, end_x, end_y = face_rect
    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
    
    if text:
        cv2.putText(frame, text, (start_x, start_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    face_id = EnhancedFaceID()
    next_user_id = 1
    
    print("\nEnhanced Face ID System")
    print("Controls:")
    print("'r' - Register new face")
    print("'i' - Identify face")
    print("'l' - List registered users")
    print("'q' - Quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        display_frame = frame.copy()
        face_rect, _ = face_id.detect_face(frame)
        
        if face_rect is not None:
            draw_face_info(display_frame, face_rect)

        cv2.putText(display_frame, "Press 'r' to register, 'i' to identify, 'l' to list users, 'q' to quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Enhanced Face ID System', display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            user_id = f"user{next_user_id}"
            success, message = face_id.register_face(user_id, frame)
            print(f"Registration: {message}")
            if success:
                next_user_id += 1
        elif key == ord('i'):
            match_id, message = face_id.identify_face(frame)
            print(f"Identification: {message}")
            if match_id:
                draw_face_info(display_frame, face_rect, f"Matched: {match_id}")
                cv2.imshow('Enhanced Face ID System', display_frame)
                cv2.waitKey(1000)
        elif key == ord('l'):
            users = face_id.get_registered_users()
            if users:
                print("\nRegistered users:")
                for user in users:
                    print(f"- {user}")
            else:
                print("No users registered")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()