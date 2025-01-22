from fastapi import FastAPI, WebSocket
from app.object_detection import yolo  # Import YOLO handler
import uvicorn

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "FastAPI Object Detection WebSocket Server Running"}

@app.post("/detect/")
async def detect_endpoint(image_data: dict):
    image_base64 = image_data.get("image")
    if not image_base64:
        return {"error": "No image provided"}

    # Convert base64 to image and run YOLO
    image = yolo.decode_base64_image(image_base64)
    if image is None:
        return {"error": "Invalid image format"}

    _, detections = yolo.predictions(image)
    return {"detections": detections}

@app.websocket("/ws/detect")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_json()
        image_base64 = data.get("image")

        if not image_base64:
            await websocket.send_json({"error": "No image provided"})
            continue

        image = yolo.decode_base64_image(image_base64)
        if image is None:
            await websocket.send_json({"error": "Invalid image format"})
            continue

        _, detections = yolo.predictions(image)
        await websocket.send_json({"detections": detections})

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
