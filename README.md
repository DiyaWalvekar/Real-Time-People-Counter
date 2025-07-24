# 👁️‍🗨️ Real-Time People Detection using YOLOv3

A Python project using **YOLOv3** and **OpenCV** to detect and count people live from webcam video feed. It draws bounding boxes around humans and displays a real-time count.

---

## 🔧 Tech Stack
- Python 3.x
- OpenCV
- NumPy
- YOLOv3 (with COCO dataset)

---

## 📁 Files Overview
- `main.py` – Webcam + detection loop
- `yolo.py` – YOLO model logic using OpenCV DNN
- `yolo-coco/` – Contains `yolov3.cfg`, `yolov3.weights`, `coco.names`
- `requirements.txt` – Required libraries

---

## 🚀 How to Run

1. **Download YOLO files** from: https://pjreddie.com/darknet/yolo/
2. Place them in a folder named `yolo-coco/`

```bash
pip install -r requirements.txt
python main.py

Press Q to quit the webcam feed

Features
Real-time person detection
Bounding boxes + live count
Non-Maximum Suppression for accuracy

Notes
Works best in good lighting
Requires yolov3.weights and config to run

