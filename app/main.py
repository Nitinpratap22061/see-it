# app/main.py
from fastapi import FastAPI
from fastapi_app import detect_image, websocket_endpoint


app = FastAPI()

# Define a route for basic testing
@app.get("/")
def read_root():
    return {"message": "FastAPI Object Detection API"}

# Add WebSocket endpoint for real-time detection
app.websocket("/ws/detect/")(websocket_endpoint)
