import os
import sys
import argparse
import cv2
from deepface import DeepFace

class SentioDetector:
    def __init__(self, device_id=0, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
        """
        Initializes the Sentio Real-Time Emotion Detector with custom parameters.
        """
        self.device_id = device_id
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        
        # Load the Haar Cascade Face Classifier (with fallback to local file)
        self.face_cascade = self._load_cascade_classifier()

    def _load_cascade_classifier(self):
        """
        Attempts to load the face cascade file from OpenCV's built-in data folder,
        falling back to the local workspace file if necessary.
        """
        cascade_filename = "haarcascade_frontalface_default.xml"
        
        # Method 1: Check OpenCV package data folder
        builtin_path = os.path.join(cv2.data.haarcascades, cascade_filename)
        if os.path.exists(builtin_path):
            return cv2.CascadeClassifier(builtin_path)
            
        # Method 2: Check current directory fallback
        local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), cascade_filename)
        if os.path.exists(local_path):
            return cv2.CascadeClassifier(local_path)
            
        sys.exit(f"Error: Missing face cascade XML file. Please download it and place it in the same directory.")

    def run(self):
        """
        Starts the webcam capture loop, detects faces, performs emotion recognition,
        and displays the results in real-time.
        """
        cap = cv2.VideoCapture(self.device_id)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)
            
            faces = self.face_cascade.detectMultiScale(
                gray_frame,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=self.min_size
            )
            
            for (x, y, w, h) in faces:
                face_roi = rgb_frame[y:y + h, x:x + w]
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                emotion = result[0]['dominant_emotion']
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                
            cv2.imshow('Real-time Emotion Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Sentio - Real-Time Facial Emotion Detection")
    parser.add_argument("--device", type=int, default=0, help="Webcam device ID (default: 0)")
    parser.add_argument("--scale", type=float, default=1.1, help="Face detection scale factor (default: 1.1)")
    parser.add_argument("--neighbors", type=int, default=5, help="Face detection min neighbors (default: 5)")
    args = parser.parse_args()
    
    detector = SentioDetector(
        device_id=args.device,
        scale_factor=args.scale,
        min_neighbors=args.neighbors
    )
    detector.run()

if __name__ == "__main__":
    main()
