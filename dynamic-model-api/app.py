from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import socketio
from models import DynamicModel, Train
from generate import Generate
import asyncio

# Image datasets (anything except tabular pima) cannot exceed this epoch count
IMAGE_EPOCHS_MAX = 5


def _is_image_dataset(inp: str) -> bool:
    return inp != "pima"


def _validate_epochs(inp: str, n_epochs: int) -> None:
    if _is_image_dataset(inp) and n_epochs > IMAGE_EPOCHS_MAX:
        raise HTTPException(
            status_code=400,
            detail=f"Image datasets are limited to {IMAGE_EPOCHS_MAX} epochs (got {n_epochs})",
        )

# dumb imports that i gyatt to add
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.model_selection import train_test_split  # --> pip install scikit-learn

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://scraply-prod.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# SocketIO setup
sio = socketio.AsyncServer(
    cors_allowed_origins=["http://localhost:3000", "https://scraply-prod.vercel.app"],
    async_mode="asgi",
)
socket_app = socketio.ASGIApp(sio, app)

# Global state
active_training = {
    "is_training": False,
    "current_progress": None,
    "is_paused": False,
    "pause_confirmed": False,  # True only when training loop is actually paused
    "completed_results": None,  # Store completed training results
}

# Track connected clients and disconnect timer
connected_clients = set()  # Set of connected socket IDs
disconnect_timer_task = None  # Single timer task for checking if training should stop


@app.get("/")
async def hello_world():
    return {"data": "hello"}


@app.get("/health")
async def health_check():
    return {"status": "online", "message": "Server is running"}


@app.post("/generate")
async def generate(request: Request):
    data = await request.json()
    _validate_epochs(data.get("input", ""), data.get("epoch", 0))

    try:
        gen = Generate(data)
        notebook_json = gen.generate_notebook()
        return Response(
            content=notebook_json.encode("utf-8"),
            media_type="application/x-ipynb+json",
            headers={
                "Content-Disposition": 'attachment; filename="generated_notebook.ipynb"'
            },
        )
    except Exception as e:
        return {"status": "failed", "error": str(e)}


@sio.event
async def connect(sid, environ):
    print("Client connected")
    connected_clients.add(sid)
    # Cancel any pending disconnect timer since a client reconnected
    global disconnect_timer_task
    if disconnect_timer_task and not disconnect_timer_task.done():
        disconnect_timer_task.cancel()
        disconnect_timer_task = None
    await sio.emit("connected", {"message": "Connected to training server"}, room=sid)


async def stop_training_on_disconnect():
    """Stop training if no clients are connected after timeout"""
    await asyncio.sleep(30)  # Wait 30 seconds for reconnection
    # Check if still no clients connected and training is active
    if not connected_clients and active_training["is_training"]:
        print("No clients connected after 30 seconds - stopping training (tab likely closed)")
        active_training["is_training"] = False
        active_training["is_paused"] = False
        active_training["pause_confirmed"] = False
        active_training["current_progress"] = None
        # Note: We keep completed_results in case client reconnects later


@sio.event
async def tab_hidden(sid):
    """Client tab became hidden (switched tabs or minimized)"""
    print("Client tab hidden")
    # Don't stop training - tab might just be switched away
    # Cancel any disconnect timer since we know the tab is still open
    global disconnect_timer_task
    if disconnect_timer_task and not disconnect_timer_task.done():
        disconnect_timer_task.cancel()
        disconnect_timer_task = None


@sio.event
async def tab_visible(sid):
    """Client tab became visible again"""
    print("Client tab visible")
    # Tab is back - definitely not closed, cancel any disconnect timer
    global disconnect_timer_task
    if disconnect_timer_task and not disconnect_timer_task.done():
        disconnect_timer_task.cancel()
        disconnect_timer_task = None


@sio.event
async def disconnect(sid):
    print("Client disconnected")
    connected_clients.discard(sid)
    
    # If training is active and no clients are connected, start a timer
    # If no client reconnects within 30 seconds, stop training
    # Note: If we received tab_hidden before disconnect, the timer was already cancelled
    global disconnect_timer_task
    if active_training["is_training"] and not connected_clients:
        # Only start timer if one doesn't already exist
        if not disconnect_timer_task or disconnect_timer_task.done():
            # Start new timer task
            disconnect_timer_task = asyncio.create_task(stop_training_on_disconnect())


@sio.event
async def check_training_status(sid):
    await sio.emit(
        "training_status",
        {
            "is_training": active_training["is_training"],
            "current_progress": active_training["current_progress"],
            "is_paused": active_training["is_paused"],
            "pause_confirmed": active_training.get("pause_confirmed", False),
            "completed_results": active_training.get("completed_results"),
        },
        room=sid,
    )


@sio.event
async def pause_training(sid):
    if active_training["is_training"]:
        active_training["is_paused"] = True
        active_training["pause_confirmed"] = False
        print("Pause Requested")
        await sio.emit(
            "training_pausing",
            {"message": "Pausing training..."},
            room=sid,
        )
    else:
        print("⚠️  Warning: Attempted to pause training but no training is active")
        await sio.emit(
            "training_error", {"error": "No active training to pause"}, room=sid
        )


@sio.event
async def resume_training(sid):
    if active_training["is_training"] and active_training["is_paused"]:
        active_training["is_paused"] = False
        active_training["pause_confirmed"] = False
        print("▶️ Resume Requested (waiting for loop to resume)")
        await sio.emit(
            "training_resuming",
            {"message": "Resuming training..."},
            room=sid,
        )
    else:
        print("⚠️  Warning: Attempted to resume training but training is not paused")
        await sio.emit(
            "training_error", {"error": "No paused training to resume"}, room=sid
        )


@sio.event
async def stop_training(sid):
    if active_training["is_training"]:
        active_training["is_training"] = False
        active_training["is_paused"] = False
        active_training["pause_confirmed"] = False
        active_training["current_progress"] = None
        active_training["completed_results"] = None
        # Cancel any disconnect timer since we're explicitly stopping
        global disconnect_timer_task
        if disconnect_timer_task and not disconnect_timer_task.done():
            disconnect_timer_task.cancel()
            disconnect_timer_task = None
        print("")
        print("🛑 Training Stopped")
        await sio.emit(
            "training_stopped", {"message": "Training has been stopped"}, room=sid
        )
    else:
        print("⚠️  Warning: Attempted to stop training but no training is active")
        await sio.emit(
            "training_error", {"error": "No active training to stop"}, room=sid
        )


async def run_training_background(t, n_epochs, batch_size, active_training):
    """Run training in background task"""
    global disconnect_timer_task
    try:
        results = await t.train_test_log_stream_async(
            n_epochs, batch_size, sio, active_training
        )
        # Mark training as complete and store results
        active_training["is_training"] = False
        active_training["current_progress"] = None
        active_training["is_paused"] = False
        active_training["pause_confirmed"] = False
        active_training["completed_results"] = {
            "final_results": results,
            "message": "Training completed successfully!",
        }
        # Cancel any disconnect timer since training completed normally
        if disconnect_timer_task and not disconnect_timer_task.done():
            disconnect_timer_task.cancel()
            disconnect_timer_task = None
    except Exception as e:
        print("Background training error:", e)
        active_training["is_training"] = False
        active_training["current_progress"] = None
        active_training["is_paused"] = False
        active_training["pause_confirmed"] = False
        active_training["completed_results"] = None
        # Cancel any disconnect timer
        if disconnect_timer_task and not disconnect_timer_task.done():
            disconnect_timer_task.cancel()
            disconnect_timer_task = None
        await sio.emit("training_error", {"error": str(e)})


@app.post("/train-stream")
async def train_stream(request: Request):
    """Streaming training endpoint that emits progress via WebSocket"""
    data = await request.json()
    print("Received streaming training request:", data)

    inp = data["input"]
    layers = data["layers"]
    loss = data["loss"]
    optimizer = data["optimizer"]
    n_epochs = data["epoch"]
    batch_size = data["batch_size"]

    _validate_epochs(inp, n_epochs)

    try:
        active_training["is_training"] = True
        active_training["current_progress"] = None
        active_training["is_paused"] = False
        active_training["pause_confirmed"] = False
        active_training["completed_results"] = None  # Clear previous results

        model = DynamicModel(layers)
        t = Train(
            model=model,
            input=inp,
            loss=loss,
            optimizer=optimizer,
            batch_size=batch_size,
        )

        print("Model initialized successfully! Starting streaming training...")
        # Start background task
        import asyncio

        asyncio.create_task(
            run_training_background(t, n_epochs, batch_size, active_training)
        )

        return {
            "status": "training_started",
            "message": "Training progress will be streamed via WebSocket",
        }

    except Exception as e:
        print("Error:", e)
        # Reset training state on error
        active_training["is_training"] = False
        active_training["current_progress"] = None
        active_training["is_paused"] = False
        active_training["pause_confirmed"] = False
        active_training["completed_results"] = None
        await sio.emit("training_error", {"error": str(e)})
        return {"error": str(e)}, 500


# Export the ASGI app for uvicorn
asgi_app = socket_app
