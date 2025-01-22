import base64
import json
import asyncio
import io
import numpy as np
from PIL import Image
from fastapi import WebSocket, WebSocketDisconnect
from app.yolo import YOLO_Pred

# Path to YOLO model and labels
MODEL_PATH = "app/ml_models/best.onnx"
DATA_PATH = "app/ml_models/data.yaml"

# Initialize YOLO model
yolo = YOLO_Pred(MODEL_PATH, DATA_PATH)

# WebSocket endpoint for real-time object detection
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()  # Accept WebSocket connection

    try:
        while True:
            data = await websocket.receive_text()
            text_data_json = json.loads(data)
            image_data = text_data_json.get("image")

            if not image_data:
                await websocket.send_text(json.dumps({"error": "No image provided"}))
                continue

            image = decode_base64_image(image_data)

            if image is None:
                await websocket.send_text(json.dumps({"error": "Invalid image format"}))
                continue

            # Run YOLO Prediction
            _, detect_res = yolo.predictions(image)

            # Compute Manhattan distances
            distances = compute_manhattan_distance(detect_res)

            response_data = {
                "detections": detect_res,
                "distances": distances,
            }

            await websocket.send_text(json.dumps(response_data))
            await asyncio.sleep(0.02)  # Small delay for handling real-time frames

    except WebSocketDisconnect:
        print("Client disconnected")

# Helper function to decode Base64 images
def decode_base64_image(img_str: str):
    try:
        img_data = base64.b64decode(img_str)
        image = Image.open(io.BytesIO(img_data))
        return np.array(image)
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

# Compute Manhattan Distance between detected objects
def compute_manhattan_distance(detections):
    distances = []
    num_objects = len(detections)

    for i in range(num_objects):
        for j in range(i + 1, num_objects):
            obj1, obj2 = detections[i], detections[j]

            # Ensure required coordinates exist
            if not all(k in obj1 and k in obj2 for k in ["x1", "x2", "y1", "y2"]):
                continue

            x1_center = (obj1["x1"] + obj1["x2"]) // 2
            y1_center = (obj1["y1"] + obj1["y2"]) // 2
            x2_center = (obj2["x1"] + obj2["x2"]) // 2
            y2_center = (obj2["y1"] + obj2["y2"]) // 2

            distance = abs(x1_center - x2_center) + abs(y1_center - y2_center)

            distances.append({
                "object1": obj1["label"],
                "object2": obj2["label"],
                "distance": distance,
            })

    return distances
