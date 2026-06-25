# Sentio — Real-Time Facial Emotion Detection

**Sentio** is a modular, real-time facial emotion recognition utility. Built using **OpenCV** and the **DeepFace** library, it monitors input from your webcam, detects human faces, and overlays dominant emotions (e.g., Happy, Sad, Angry, Neutral) onto live video streams.

---

## Key Features

- **Class-Based Architecture**: Highly modular, making it easy to import, extend, and integrate into other Python computer vision pipelines.
- **Robust Path Fallbacks**: Automatically resolves paths for Haar Cascade XML templates using either OpenCV package installations or local directory files.
- **Configurable Parameters**: Customize camera device index, scale factor, min neighbors, and visualization options dynamically.
- **Real-Time FPS Monitor**: Active frame rate (FPS) rendering on screen to gauge performance.
- **Modern HUD Display**: Custom cyan boundary boxes and readability backgrounds for label texts.

---

## Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone <your-repository-url>
   cd Facial-Emotion-Recognition-using-OpenCV-and-Deepface
   ```

2. **Install Dependencies**:
   It is recommended to run this inside a virtual environment (Python 3.9 - 3.12 recommended).
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

Run the detector using the command line:

```bash
python emotion.py
```

### Custom Configurations

Configure detector settings through CLI arguments:

```bash
# Run using a specific webcam index (e.g., Device ID: 1)
python emotion.py --device 1

# Adjust face detection sensitivity (Scale factor: 1.2, Min neighbors: 6)
python emotion.py --scale 1.2 --neighbors 6

# Hide the real-time FPS overlay
python emotion.py --hide-fps
```

### Windows Unicode Warning & Fix
If you encounter `UnicodeEncodeError` when logging text containing emojis in the terminal on Windows, enable UTF-8 mode before running:

* **PowerShell**:
  ```powershell
  $env:PYTHONUTF8="1"
  python emotion.py
  ```
* **Command Prompt (CMD)**:
  ```cmd
  set PYTHONUTF8=1
  python emotion.py
  ```

---

## Project Dependencies

- **[DeepFace](https://github.com/serengil/deepface)**: A deep learning facial analysis framework for predicting emotions.
- **[OpenCV (opencv-python)](https://opencv.org/)**: Open-source computer vision library used for real-time video capture and frame rendering.
- **[TF-Keras](https://github.com/tensorflow/keras)**: Provides underlying deep learning mechanics.

---

## License

This project is licensed under the [MIT License](LICENSE).
