import cv2
import numpy as np
import yaml
import os
from yaml.loader import SafeLoader

class YOLO_Pred:
    def __init__(self, onnx_model, data_yaml):
        self.labels = []
        self.nc = 0
        self.yolo = None

        # Load YAML file (handling missing or corrupt files)
        if not os.path.exists(data_yaml):
            print(f"⚠️ Error: Data YAML file not found at {data_yaml}")
            return
        
        try:
            with open(data_yaml, mode='r') as f:
                data = yaml.load(f, Loader=SafeLoader)
                self.labels = data.get('names', [])
                self.nc = data.get('nc', 0)
        except Exception as e:
            print(f"⚠️ Error loading YAML file: {e}")
            return

        # Load YOLO model (handling invalid model paths)
        if not os.path.exists(onnx_model):
            print(f"⚠️ Error: ONNX model not found at {onnx_model}")
            return

        try:
            self.yolo = cv2.dnn.readNetFromONNX(onnx_model)
            self.yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        except Exception as e:
            print(f"⚠️ Error loading ONNX model: {e}")
            return

    def predictions(self, image):
        if image is None or not isinstance(image, np.ndarray):
            print("⚠️ Error: Invalid image format received.")
            return None, []

        row, col, _ = image.shape

        # Convert image into square
        max_rc = max(row, col)
        input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
        input_image[0:row, 0:col] = image

        # Get prediction from square image
        INPUT_WH_YOLO = 640
        blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WH_YOLO, INPUT_WH_YOLO), swapRB=True, crop=False)
        self.yolo.setInput(blob)
        preds = self.yolo.forward()

        # Process predictions
        detections = self.process_detections(preds, input_image.shape[:2])

        return image, detections

    def process_detections(self, preds, image_shape):
        detections = preds[0]
        boxes, confidences, classes = [], [], []

        image_w, image_h = image_shape
        x_factor = image_w / 640
        y_factor = image_h / 640

        for row in detections:
            confidence = row[4]
            if confidence > 0.4:
                class_score = row[5:].max()
                class_id = row[5:].argmax()

                if class_score > 0.25:
                    cx, cy, w, h = row[:4]
                    left = int((cx - 0.5 * w) * x_factor)
                    top = int((cy - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)

                    boxes.append([left, top, width, height])
                    confidences.append(float(confidence))  # Ensure correct dtype
                    classes.append(class_id)

        return self.apply_nms(boxes, confidences, classes, image_w, image_h)

    def apply_nms(self, boxes, confidences, classes, image_w, image_h):
        detected_objects = []
        if len(boxes) == 0:
            return []

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)
        if indices is None or len(indices) == 0:
            return []

        indices = indices.flatten()

        for ind in indices:
            x, y, w, h = boxes[ind]
            class_id = classes[ind]
            confidence = int(confidences[ind] * 100)
            class_name = self.labels[class_id] if class_id < len(self.labels) else "Unknown"

            # Determine position and distance
            position = "center" if image_w / 3 < x < 2 * image_w / 3 else ("left" if x < image_w / 3 else "right")
            distance = "near" if h > image_h / 2 else "far"

            detected_objects.append({
                "label": class_name,
                "confidence": confidence,
                "position": position,
                "distance": distance,
                "x1": x, "y1": y,
                "x2": x + w, "y2": y + h
            })

        return detected_objects
