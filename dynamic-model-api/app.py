from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import socketio
from models import DynamicModel, Train
from generate import Generate
import asyncio
import uuid

# Image datasets (anything except tabular pima) cannot exceed this epoch count
IMAGE_EPOCHS_MAX = 5
MAX_CONCURRENT_JOBS = 2
JOB_TTL_SECONDS = 15 * 60


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

# Per-session training jobs (one job per socket, isolated pause/stop/progress)
jobs = {}  # job_id -> job dict
sid_to_job = {}  # socket id -> job_id
connected_clients = set()


def _job_room(job_id: str) -> str:
    return f"job:{job_id}"


def _running_count() -> int:
    return sum(1 for job in jobs.values() if job.get("is_training"))


def _get_job_for_sid(sid: str):
    job_id = sid_to_job.get(sid)
    if not job_id:
        return None, None
    return job_id, jobs.get(job_id)


def _job_status_payload(job):
    if not job:
        return {
            "job_id": None,
            "is_training": False,
            "current_progress": None,
            "is_paused": False,
            "pause_confirmed": False,
            "completed_results": None,
        }
    return {
        "job_id": job.get("job_id"),
        "is_training": job.get("is_training", False),
        "current_progress": job.get("current_progress"),
        "is_paused": job.get("is_paused", False),
        "pause_confirmed": job.get("pause_confirmed", False),
        "completed_results": job.get("completed_results"),
    }


def _cancel_task(task):
    if task and not task.done():
        task.cancel()


def _cancel_job_timer(job):
    if not job:
        return
    _cancel_task(job.get("disconnect_timer_task"))
    job["disconnect_timer_task"] = None


def _schedule_job_expiry(job):
    """Drop a finished job after TTL so the in-memory map cannot grow forever."""
    if not job:
        return
    _cancel_task(job.get("expiry_task"))

    async def _expire():
        await asyncio.sleep(JOB_TTL_SECONDS)
        job_id = job.get("job_id")
        stored = jobs.get(job_id)
        if stored is not job or stored.get("is_training"):
            return
        jobs.pop(job_id, None)
        owner = stored.get("owner_sid")
        if owner and sid_to_job.get(owner) == job_id:
            sid_to_job.pop(owner, None)
        print(f"Expired job {job_id}")

    job["expiry_task"] = asyncio.create_task(_expire())


def _create_job(owner_sid: str) -> dict:
    job_id = str(uuid.uuid4())
    job = {
        "job_id": job_id,
        "owner_sid": owner_sid,
        "is_training": True,
        "current_progress": None,
        "is_paused": False,
        "pause_confirmed": False,
        "completed_results": None,
        "disconnect_timer_task": None,
        "expiry_task": None,
        "task": None,
        "room": _job_room(job_id),
    }
    jobs[job_id] = job
    sid_to_job[owner_sid] = job_id
    return job


async def _emit_job(job, event, data, to=None):
    payload = dict(data) if data else {}
    target = to
    if job:
        payload.setdefault("job_id", job.get("job_id"))
        target = target or job.get("room")
    if not target:
        print(f"Skipping {event}: no job room (refusing global broadcast)")
        return
    await sio.emit(event, payload, room=target)


async def _drop_sid_job(sid: str, remove_job: bool = False):
    job_id, job = _get_job_for_sid(sid)
    if not job:
        return
    try:
        await sio.leave_room(sid, job["room"])
    except Exception:
        pass
    if sid_to_job.get(sid) == job_id:
        sid_to_job.pop(sid, None)
    if remove_job:
        _cancel_job_timer(job)
        _cancel_task(job.get("expiry_task"))
        jobs.pop(job_id, None)


async def _begin_training(sid: str, data: dict) -> dict:
    if sid not in connected_clients:
        raise HTTPException(
            status_code=400,
            detail="Socket is not connected. Refresh and try again.",
        )

    _, existing = _get_job_for_sid(sid)
    if existing and existing.get("is_training"):
        raise HTTPException(
            status_code=409,
            detail="A training job is already running for this session",
        )

    if _running_count() >= MAX_CONCURRENT_JOBS:
        raise HTTPException(
            status_code=429,
            detail="Too many trainings in progress. Try again shortly.",
        )

    inp = data["input"]
    layers = data["layers"]
    loss = data["loss"]
    optimizer = data["optimizer"]
    n_epochs = data["epoch"]
    batch_size = data["batch_size"]

    _validate_epochs(inp, n_epochs)

    if existing:
        await _drop_sid_job(sid, remove_job=not existing.get("is_training"))

    job = _create_job(sid)
    await sio.enter_room(sid, job["room"])

    try:
        model = DynamicModel(layers)
        t = Train(
            model=model,
            input=inp,
            loss=loss,
            optimizer=optimizer,
            batch_size=batch_size,
        )

        print("Model initialized successfully! Starting streaming training...")
        job["task"] = asyncio.create_task(
            run_training_background(t, n_epochs, batch_size, job)
        )
        return job
    except HTTPException:
        await _drop_sid_job(sid, remove_job=True)
        raise
    except Exception as e:
        print("Error:", e)
        await _drop_sid_job(sid, remove_job=True)
        await _emit_job(None, "training_error", {"error": str(e)}, to=sid)
        raise HTTPException(status_code=500, detail=str(e))


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
    await sio.emit("connected", {"message": "Connected to training server"}, room=sid)


async def stop_job_on_disconnect(job_id: str):
    """Stop a job if its owner does not reconnect within the timeout."""
    await asyncio.sleep(30)
    job = jobs.get(job_id)
    if not job or not job.get("is_training"):
        return
    owner = job.get("owner_sid")
    if owner in connected_clients and sid_to_job.get(owner) == job_id:
        return
    print(f"No client for job {job_id} after 30 seconds - stopping training")
    job["is_training"] = False
    job["is_paused"] = False
    job["pause_confirmed"] = False
    job["current_progress"] = None
    _schedule_job_expiry(job)


@sio.event
async def join_job(sid, data):
    """Reattach a reconnecting socket to an existing job."""
    job_id = (data or {}).get("job_id")
    job = jobs.get(job_id) if job_id else None
    if not job:
        await sio.emit("job_not_found", {"job_id": job_id}, room=sid)
        return

    old_sid = job.get("owner_sid")
    if old_sid and old_sid != sid:
        if sid_to_job.get(old_sid) == job_id:
            sid_to_job.pop(old_sid, None)
        try:
            await sio.leave_room(old_sid, job["room"])
        except Exception:
            pass

    job["owner_sid"] = sid
    sid_to_job[sid] = job_id
    await sio.enter_room(sid, job["room"])
    _cancel_job_timer(job)
    if not job.get("is_training"):
        _schedule_job_expiry(job)
    await _emit_job(job, "training_status", _job_status_payload(job), to=sid)


@sio.event
async def tab_hidden(sid):
    """Client tab became hidden (switched tabs or minimized)."""
    print("Client tab hidden")
    _, job = _get_job_for_sid(sid)
    _cancel_job_timer(job)


@sio.event
async def tab_visible(sid):
    """Client tab became visible again."""
    print("Client tab visible")
    _, job = _get_job_for_sid(sid)
    _cancel_job_timer(job)


@sio.event
async def disconnect(sid):
    print("Client disconnected")
    connected_clients.discard(sid)

    job_id, job = _get_job_for_sid(sid)
    if sid_to_job.get(sid) == job_id:
        sid_to_job.pop(sid, None)

    if job and job.get("is_training"):
        timer = job.get("disconnect_timer_task")
        if not timer or timer.done():
            job["disconnect_timer_task"] = asyncio.create_task(
                stop_job_on_disconnect(job_id)
            )


@sio.event
async def check_training_status(sid):
    _, job = _get_job_for_sid(sid)
    await _emit_job(job, "training_status", _job_status_payload(job), to=sid)


@sio.event
async def pause_training(sid):
    _, job = _get_job_for_sid(sid)
    if job and job.get("is_training"):
        job["is_paused"] = True
        job["pause_confirmed"] = False
        print("Pause Requested")
        await _emit_job(job, "training_pausing", {"message": "Pausing training..."})
    else:
        print("⚠️  Warning: Attempted to pause training but no training is active")
        await sio.emit(
            "training_error", {"error": "No active training to pause"}, room=sid
        )


@sio.event
async def resume_training(sid):
    _, job = _get_job_for_sid(sid)
    if job and job.get("is_training") and job.get("is_paused"):
        job["is_paused"] = False
        job["pause_confirmed"] = False
        print("▶️ Resume Requested (waiting for loop to resume)")
        await _emit_job(job, "training_resuming", {"message": "Resuming training..."})
    else:
        print("⚠️  Warning: Attempted to resume training but training is not paused")
        await sio.emit(
            "training_error", {"error": "No paused training to resume"}, room=sid
        )


@sio.event
async def stop_training(sid):
    _, job = _get_job_for_sid(sid)
    if job and job.get("is_training"):
        job["is_training"] = False
        job["is_paused"] = False
        job["pause_confirmed"] = False
        job["current_progress"] = None
        job["completed_results"] = None
        _cancel_job_timer(job)
        _schedule_job_expiry(job)
        print("")
        print("🛑 Training Stopped")
        await _emit_job(job, "training_stopped", {"message": "Training has been stopped"})
    else:
        print("⚠️  Warning: Attempted to stop training but no training is active")
        await sio.emit(
            "training_error", {"error": "No active training to stop"}, room=sid
        )


@sio.event
async def start_training(sid, data):
    """Start a training job bound to this socket (preferred over HTTP)."""
    try:
        job = await _begin_training(sid, data or {})
        await _emit_job(
            job,
            "training_accepted",
            {"message": "Training progress will be streamed via WebSocket"},
            to=sid,
        )
    except HTTPException as e:
        detail = e.detail if isinstance(e.detail, str) else str(e.detail)
        await sio.emit("training_error", {"error": detail}, room=sid)


async def run_training_background(t, n_epochs, batch_size, job):
    """Run training in background task for a single job."""
    room = job["room"]
    try:
        results = await t.train_test_log_stream_async(
            n_epochs, batch_size, sio, job, room=room
        )
        job["is_training"] = False
        job["current_progress"] = None
        job["is_paused"] = False
        job["pause_confirmed"] = False
        job["completed_results"] = {
            "final_results": results,
            "message": "Training completed successfully!",
        }
        _cancel_job_timer(job)
        _schedule_job_expiry(job)
    except Exception as e:
        print("Background training error:", e)
        job["is_training"] = False
        job["current_progress"] = None
        job["is_paused"] = False
        job["pause_confirmed"] = False
        job["completed_results"] = None
        _cancel_job_timer(job)
        _schedule_job_expiry(job)
        await _emit_job(job, "training_error", {"error": str(e)})


@app.post("/train-stream")
async def train_stream(request: Request):
    """Streaming training endpoint that emits progress via WebSocket."""
    data = await request.json()
    print("Received streaming training request:", data)

    socket_id = data.get("socket_id")
    if not socket_id:
        raise HTTPException(status_code=400, detail="socket_id is required")

    job = await _begin_training(socket_id, data)
    return {
        "status": "training_started",
        "job_id": job["job_id"],
        "message": "Training progress will be streamed via WebSocket",
    }


# Export the ASGI app for uvicorn
asgi_app = socket_app
