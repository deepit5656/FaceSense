"""
FaceSense - Real-Time Facial Emotion Detection
Built with OpenCV and DeepFace.
"""

import os
import sys
import time
import argparse
import logging
import cv2
from deepface import DeepFace

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("FaceSense")


class FaceSenseDetector:
    def __init__(self, device_id=0, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
        """
        Initializes the FaceSense Real-Time Emotion Detector.
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
            logger.info(f"Loading Haar Cascade from built-in path: {builtin_path}")
            return cv2.CascadeClassifier(builtin_path)
            
        # Method 2: Check current directory fallback
        local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), cascade_filename)
        if os.path.exists(local_path):
            logger.info(f"Loading Haar Cascade from local fallback: {local_path}")
            return cv2.CascadeClassifier(local_path)
            
        # Error case
        logger.error(f"Could not find '{cascade_filename}' built-in or locally!")
        sys.exit(f"Error: Missing face cascade XML file. Please download it and place it in the same directory.")

    def run(self, show_fps=True):
        """
        Starts the webcam capture loop, detects faces, performs emotion recognition,
        and displays the results in real-time.
        """
        logger.info(f"Initializing video capture (Device ID: {self.device_id})...")
        cap = cv2.VideoCapture(self.device_id)
        
        if not cap.isOpened():
            logger.error(f"Could not open video device index {self.device_id}")
            return

        logger.info("Video capture started successfully. Press 'q' on the window to exit.")
        
        prev_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to grab frame.")
                    break
                
                # Pre-process frame: Convert to grayscale for detection
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Convert grayscale back to RGB for DeepFace analysis
                rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)
                
                # Detect faces in grayscale frame
                faces = self.face_cascade.detectMultiScale(
                    gray_frame,
                    scaleFactor=self.scale_factor,
                    minNeighbors=self.min_neighbors,
                    minSize=self.min_size
                )
                
                for (x, y, w, h) in faces:
                    # Extract face ROI (Region of Interest)
                    face_roi = rgb_frame[y:y + h, x:x + w]
                    
                    try:
                        # Analyze dominant emotion using DeepFace (non-enforcing detector inside ROI)
                        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                        dominant_emotion = result[0]['dominant_emotion']
                    except Exception as e:
                        logger.debug(f"Failed to analyze face ROI: {e}")
                        dominant_emotion = "unknown"
                    
                    # Modern drawing style: Cyan bounding box (0, 255, 255) with rounded corners
                    line_color = (255, 255, 0)  # Cyan
                    label_color = (255, 255, 0)
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), line_color, 2)
                    
                    # Draw background rectangle for text labels (for better readability)
                    label_text = f"Emotion: {dominant_emotion.upper()}"
                    (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(frame, (x, y - text_h - 15), (x + text_w + 10, y), line_color, -1)
                    
                    # Draw text label over background (dark blue text on cyan background)
                    cv2.putText(frame, label_text, (x + 5, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2)
                
                # Optionally calculate and draw FPS
                if show_fps:
                    curr_time = time.time()
                    fps = 1.0 / (curr_time - prev_time)
                    prev_time = curr_time
                    cv2.putText(
                        frame,
                        f"FPS: {int(fps)}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )
                
                # Show GUI Window
                cv2.imshow("FaceSense - Real-Time Emotion Recognition", frame)
                
                # Press 'q' to break execution loop
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Exiting on user request.")
                    break
        finally:
            # Always ensure release of video capture resource
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Resources released. Application closed.")


def main():
    parser = argparse.ArgumentParser(description="FaceSense - Real-Time Facial Emotion Detection")
    parser.add_argument("--device", type=int, default=0, help="Webcam device ID (default: 0)")
    parser.add_argument("--scale", type=float, default=1.1, help="Face detection scale factor (default: 1.1)")
    parser.add_argument("--neighbors", type=int, default=5, help="Face detection min neighbors (default: 5)")
    parser.add_argument("--hide-fps", action="store_true", help="Hide the FPS overlay on screen")
    
    args = parser.parse_args()
    
    detector = FaceSenseDetector(
        device_id=args.device,
        scale_factor=args.scale,
        min_neighbors=args.neighbors
    )
    detector.run(show_fps=not args.hide_fps)


if __name__ == "__main__":
    main()
