import cv2
import numpy as np
import os

class YOLODetector:
    def __init__(self):
        # Load YOLO model
        yolo_path = os.path.join(os.getcwd(), "yolo-coco")
        self.weights_path = os.path.join(yolo_path, "yolov3.weights")
        self.config_path = os.path.join(yolo_path, "yolov3.cfg")
        self.labels_path = os.path.join(yolo_path, "coco.names")

        # Load class labels
        with open(self.labels_path, "r") as f:
            self.labels = [line.strip() for line in f.readlines()]

        # Load YOLO model
        self.net = cv2.dnn.readNetFromDarknet(self.config_path, self.weights_path)
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
    
    def detect_people(self, image):
        height, width = image.shape[:2]

        # Convert image to blob
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        # Initialize lists
        boxes, confidences = [], []

        # Process outputs
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                # Check if detected object is a person (class ID = 0 in COCO dataset)
                if class_id == 0 and confidence > 0.5:
                    box = detection[:4] * np.array([width, height, width, height])
                    (center_x, center_y, w, h) = box.astype("int")
                    x = int(center_x - (w / 2))
                    y = int(center_y - (h / 2))

                    boxes.append([x, y, int(w), int(h)])
                    confidences.append(float(confidence))

        # Apply Non-Maximum Suppression (NMS)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

        if len(indices) > 0 and isinstance(indices, np.ndarray):  # Fix: Check if indices is a valid NumPy array
            return [boxes[i] for i in indices.flatten()]
        else:
            return []  # Return empty list if no detections

