from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import os
import shutil
import uuid
from typing import Optional
import pandas as pd
from sklearn.model_selection import train_test_split
from zipfile import ZipFile
from PIL import Image
import io
from datasets import load_dataset
import base64
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from io import BytesIO
from torchvision import transforms
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.models import Base, NASJob, NASResult
from backend.celery_worker import run_nas_job
from loguru import logger
import json
from datetime import datetime
import threading
from backend.nas_search import nas_search

DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///nas.db')
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

# Initialize DB tables if not present
def init_db():
    Base.metadata.create_all(bind=engine)

init_db()

app = FastAPI()

# Allow CORS for local frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  # Restrict to frontend dev origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory cache for live stats
live_stats = {
    'accuracy': None,
    'loss': None,
    'best_model': None,
    'iteration': 0,
    'history': []
}

# In-memory dataset registry (job_id -> info)
dataset_registry = {}
UPLOAD_DIR = 'uploads'
os.makedirs(UPLOAD_DIR, exist_ok=True)

# In-memory split cache: job_id -> {'train': ..., 'val': ..., 'test': ...}
dataset_splits = {}

# WebSocket manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)

manager = ConnectionManager()

@app.websocket("/ws/progress")
async def websocket_endpoint(websocket: WebSocket):
    print("[INFO] Loading page opened: WebSocket /ws/progress connection established.")
    await manager.connect(websocket)
    try:
        while True:
            await websocket.send_json(live_stats)
            await asyncio.sleep(1)  # Send updates every second
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/api/progress")
def get_progress():
    return JSONResponse(live_stats)

@app.post("/api/progress")
def update_progress(data: dict):
    # Update live stats from NAS loop
    for k, v in data.items():
        live_stats[k] = v
    # Keep a short history for frontend
    if 'history' in live_stats and 'iteration' in data:
        live_stats['history'].append({k: v for k, v in data.items() if k != 'history'})
        if len(live_stats['history']) > 100:
            live_stats['history'] = live_stats['history'][-100:]
    return {"status": "ok"}

@app.post("/api/upload_dataset")
async def upload_dataset(file: UploadFile = File(...), dataset_type: str = Form(...)):
    # dataset_type: 'csv', 'zip', 'images'
    job_id = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    dataset_registry[job_id] = {
        'filename': file.filename,
        'path': save_path,
        'type': dataset_type
    }
    return {"job_id": job_id, "filename": file.filename, "type": dataset_type}

@app.get("/api/datasets")
def list_datasets():
    return [{"job_id": k, **v} for k, v in dataset_registry.items()]

@app.post("/api/parse_dataset")
def parse_dataset(job_id: str, val_ratio: float = 0.1, test_ratio: float = 0.1):
    if job_id not in dataset_registry:
        return JSONResponse({"error": "job_id not found"}, status_code=404)
    info = dataset_registry[job_id]
    path = info['path']
    dtype = info['type']
    splits = {}
    if dtype == 'csv':
        df = pd.read_csv(path)
        train, temp = train_test_split(df, test_size=val_ratio+test_ratio, random_state=42)
        val, test = train_test_split(temp, test_size=test_ratio/(val_ratio+test_ratio), random_state=42)
        splits = {'train': train, 'val': val, 'test': test}
    elif dtype == 'zip':
        # Assume zip of images, split filenames
        with ZipFile(path) as zf:
            img_files = [f for f in zf.namelist() if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            train, temp = train_test_split(img_files, test_size=val_ratio+test_ratio, random_state=42)
            val, test = train_test_split(temp, test_size=test_ratio/(val_ratio+test_ratio), random_state=42)
            splits = {'train': train, 'val': val, 'test': test}
    elif dtype == 'images':
        # Folder of images
        img_files = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        train, temp = train_test_split(img_files, test_size=val_ratio+test_ratio, random_state=42)
        val, test = train_test_split(temp, test_size=test_ratio/(val_ratio+test_ratio), random_state=42)
        splits = {'train': train, 'val': val, 'test': test}
    else:
        return JSONResponse({"error": "Unsupported dataset type"}, status_code=400)
    dataset_splits[job_id] = splits
    return {"job_id": job_id, "splits": {k: len(v) for k, v in splits.items()}}

@app.get("/api/dataset_split")
def get_dataset_split(job_id: str):
    if job_id not in dataset_splits:
        return JSONResponse({"error": "No split found for job_id"}, status_code=404)
    # For demo, just return counts
    splits = dataset_splits[job_id]
    return {"job_id": job_id, "splits": {k: len(v) for k, v in splits.items()}}

@app.get("/api/hf_datasets")
def list_hf_datasets():
    # List a subset of popular datasets for demo
    popular = ['mnist', 'cifar10', 'fashion_mnist', 'imdb', 'ag_news', 'sst2']
    available = [d for d in popular if d in list_datasets()]
    return {"available": available}

@app.post("/api/load_hf_dataset")
def load_hf_dataset(dataset_name: str, val_ratio: float = 0.1, test_ratio: float = 0.1):
    # Load and split a HuggingFace dataset
    ds = load_dataset(dataset_name)
    # Try to use 'train' and 'test' splits if available
    if 'train' in ds and 'test' in ds:
        train = ds['train']
        test = ds['test']
        # Optionally split train into train/val
        train_indices, val_indices = train_test_split(list(range(len(train))), test_size=val_ratio, random_state=42)
        train_split = train.select(train_indices)
        val_split = train.select(val_indices)
    else:
        # If only one split, split into train/val/test
        all_data = ds[list(ds.keys())[0]]
        train_indices, temp_indices = train_test_split(list(range(len(all_data))), test_size=val_ratio+test_ratio, random_state=42)
        temp_split = all_data.select(temp_indices)
        val_indices, test_indices = train_test_split(list(range(len(temp_split))), test_size=test_ratio/(val_ratio+test_ratio), random_state=42)
        val_split = temp_split.select(val_indices)
        test_split = temp_split.select(test_indices)
        train_split = all_data.select(train_indices)
        test = test_split
    job_id = f"hf_{dataset_name}_{uuid.uuid4()}"
    dataset_registry[job_id] = {
        'filename': dataset_name,
        'path': None,
        'type': 'huggingface',
        'source': dataset_name
    }
    dataset_splits[job_id] = {
        'train': train_split,
        'val': val_split,
        'test': test
    }
    return {"job_id": job_id, "splits": {k: len(v) for k, v in dataset_splits[job_id].items()}, "source": dataset_name}

def compute_gradcam(model, input_tensor, target_layer):
    # Simple GradCAM for last conv layer
    activations = []
    gradients = []
    def forward_hook(module, inp, out):
        activations.append(out)
    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])
    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_backward_hook(backward_hook)
    output = model(input_tensor)
    pred_class = output.argmax(dim=1)
    loss = output[0, pred_class]
    model.zero_grad()
    loss.backward()
    handle_fwd.remove()
    handle_bwd.remove()
    grad = gradients[0].cpu().data.numpy()[0]
    act = activations[0].cpu().data.numpy()[0]
    weights = grad.mean(axis=(1, 2))
    cam = (weights[:, None, None] * act).sum(axis=0)
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)
    return cam

@app.post("/api/gradcam")
async def gradcam_endpoint(model_id: str = Form(...), image: UploadFile = File(...)):
    # For demo: load NASNet model from disk (assume path is model_id+'.pth')
    from nas_search import NASNet
    import numpy as np
    model_path = f"{model_id}.pth"
    if not os.path.exists(model_path):
        return JSONResponse({"error": "Model not found"}, status_code=404)
    model = NASNet({'layers': []})  # Dummy arch, will load state_dict
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    # Preprocess image
    img_bytes = await image.read()
    pil_img = Image.open(BytesIO(img_bytes)).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    input_tensor = preprocess(pil_img).unsqueeze(0)
    # Find last conv layer
    conv_layers = [m for m in model.features if isinstance(m, torch.nn.Conv2d)]
    if not conv_layers:
        return JSONResponse({"error": "No conv layers found"}, status_code=400)
    target_layer = conv_layers[-1]
    cam = compute_gradcam(model, input_tensor, target_layer)
    # Overlay CAM on image
    cam_img = np.uint8(255 * cam)
    cam_img = Image.fromarray(cam_img).resize(pil_img.size, resample=Image.BILINEAR)
    heatmap = plt.get_cmap('jet')(cam_img)[:, :, :3]
    heatmap = np.uint8(255 * heatmap)
    overlay = np.array(pil_img) * 0.5 + heatmap * 0.5
    overlay = np.uint8(overlay)
    out_img = Image.fromarray(overlay)
    buf = BytesIO()
    out_img.save(buf, format='PNG')
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return {"gradcam": img_b64}

@app.post("/api/activations")
async def activations_endpoint(model_id: str = Form(...), image: UploadFile = File(...), layer_idx: int = Form(0), n_maps: int = Form(8)):
    from nas_search import NASNet
    model_path = f"{model_id}.pth"
    if not os.path.exists(model_path):
        return JSONResponse({"error": "Model not found"}, status_code=404)
    model = NASNet({'layers': []})
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    img_bytes = await image.read()
    pil_img = Image.open(BytesIO(img_bytes)).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    input_tensor = preprocess(pil_img).unsqueeze(0)
    # Get activations from selected layer
    activations = []
    def hook_fn(module, inp, out):
        activations.append(out.detach().cpu())
    layers = list(model.features)
    if layer_idx < 0 or layer_idx >= len(layers):
        return JSONResponse({"error": "Invalid layer_idx"}, status_code=400)
    handle = layers[layer_idx].register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(input_tensor)
    handle.remove()
    if not activations:
        return JSONResponse({"error": "No activations found"}, status_code=400)
    act = activations[0][0]  # first sample
    n_maps = min(n_maps, act.shape[0])
    fig, axes = plt.subplots(1, n_maps, figsize=(n_maps*2, 2))
    for i in range(n_maps):
        axes[i].imshow(act[i].numpy(), cmap='viridis')
        axes[i].axis('off')
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='PNG')
    plt.close(fig)
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return {"activations": img_b64}

@app.get("/api/model_summary")
def model_summary(model_id: str):
    from nas_search import NASNet
    import time
    model_path = f"{model_id}.pth"
    if not os.path.exists(model_path):
        return JSONResponse({"error": "Model not found"}, status_code=404)
    model = NASNet({'layers': []})
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    # Params
    num_params = sum(p.numel() for p in model.parameters())
    # Dummy FLOPs (real calculation would use ptflops or similar)
    flops = num_params * 2  # Placeholder
    # Dummy inference time
    dummy_input = torch.randn(1, 1, 28, 28)
    start = time.time()
    with torch.no_grad():
        _ = model(dummy_input)
    inf_time = time.time() - start
    return {"model_id": model_id, "num_params": num_params, "flops": flops, "inference_time": inf_time}

ensemble_registry = {}

@app.post("/api/create_ensemble")
def create_ensemble(model_ids: list, method: str = 'voting'):
    # Store ensemble config
    ensemble_id = str(uuid.uuid4())
    ensemble_registry[ensemble_id] = {
        'model_ids': model_ids,
        'method': method
    }
    return {"ensemble_id": ensemble_id, "model_ids": model_ids, "method": method}

@app.post("/api/eval_ensemble")
def eval_ensemble(ensemble_id: str):
    from nas_search import NASNet
    import numpy as np
    # For demo: use MNIST test set
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    DEVICE = 'cpu'
    if ensemble_id not in ensemble_registry:
        return JSONResponse({"error": "Ensemble not found"}, status_code=404)
    config = ensemble_registry[ensemble_id]
    model_ids = config['model_ids']
    method = config['method']
    models = []
    for mid in model_ids:
        model_path = f"{mid}.pth"
        if not os.path.exists(model_path):
            return JSONResponse({"error": f"Model {mid} not found"}, status_code=404)
        model = NASNet({'layers': []})
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        models.append(model)
    # Load MNIST test set
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('data/MNIST', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    correct, total = 0, 0
    for xb, yb in test_loader:
        preds = []
        for model in models:
            with torch.no_grad():
                out = model(xb)
                preds.append(out)
        if method == 'averaging':
            avg_out = torch.stack(preds).mean(dim=0)
            final_pred = avg_out.argmax(dim=1)
        else:  # voting
            votes = torch.stack([p.argmax(dim=1) for p in preds])
            # Mode along models axis
            final_pred = torch.mode(votes, dim=0)[0]
        correct += (final_pred == yb).sum().item()
        total += yb.size(0)
    acc = correct / total
    return {"ensemble_id": ensemble_id, "accuracy": acc, "method": method, "num_models": len(models)}

@app.post('/api/submit_job')
def submit_job(config: dict):
    session = SessionLocal()
    try:
        job_id = str(uuid.uuid4())
        seed = config.get('seed', int.from_bytes(os.urandom(4), 'little'))
        job = NASJob(job_id=job_id, config=config, seed=seed, status='pending')
        session.add(job)
        session.commit()
        logger.info(f"Job submitted: {job_id}")
        run_nas_job.delay(job_id)
        return {"job_id": job_id, "seed": seed}
    except Exception as e:
        logger.error(f"Job submission failed: {e}")
        session.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

@app.get('/api/job/{job_id}')
def get_job(job_id: str):
    session = SessionLocal()
    try:
        job = session.query(NASJob).filter_by(job_id=job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail='Job not found')
        results = session.query(NASResult).filter_by(job_id=job_id).order_by(NASResult.step).all()
        return {
            'job_id': job.job_id,
            'status': job.status,
            'seed': job.seed,
            'config': job.config,
            'results': [
                {
                    'step': r.step,
                    'architecture': r.architecture,
                    'hparams': r.hparams,
                    'metrics': r.metrics
                } for r in results
            ]
        }
    finally:
        session.close()

@app.post('/api/replay/{job_id}')
def replay_job(job_id: str):
    session = SessionLocal()
    try:
        job = session.query(NASJob).filter_by(job_id=job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail='Job not found')
        # Create a new job with the same config and seed
        new_job_id = str(uuid.uuid4())
        new_job = NASJob(job_id=new_job_id, config=job.config, seed=job.seed, status='pending')
        session.add(new_job)
        session.commit()
        logger.info(f"Replay job submitted: {new_job_id} (original: {job_id})")
        run_nas_job.delay(new_job_id)
        return {"replay_job_id": new_job_id, "original_job_id": job_id, "seed": job.seed}
    except Exception as e:
        logger.error(f"Replay failed: {e}")
        session.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

@app.get('/api/export_config/{job_id}')
def export_config(job_id: str):
    session = SessionLocal()
    try:
        job = session.query(NASJob).filter_by(job_id=job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail='Job not found')
        config = job.config.copy() if job.config else {}
        config['seed'] = job.seed
        return JSONResponse(config)
    finally:
        session.close()

@app.get('/health')
def health():
    return {"status": "ok"}

@app.get("/strategies")
def get_strategies():
    return {"strategies": [
        "random",
        "evolutionary",
        "bayesian",
        "optuna",
        "nsga-ii",
        "one-shot",
        "reinforcement",
        "grid"
    ]}

@app.post("/nas/start")
def start_nas_job(config: dict):
    try:
        job_id = str(uuid.uuid4())
        print(f"Starting NAS job {job_id} with config: {config}")

        def progress_callback(progress):
            for k, v in progress.items():
                live_stats[k] = v

        thread = threading.Thread(
            target=nas_search,
            kwargs={
                'strategy_name': config.get('strategy', 'random'),
                'constraints': {'max_params': config.get('max_params')},
                'prior_arch_path': config.get('prior_arch'),
                'log_method': config.get('log_method', 'json'),
                'progress_callback': progress_callback
            }
        )
        thread.start()

        return {"job_id": job_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/nas/jobs")
def list_jobs():
    session = SessionLocal()
    try:
        jobs = session.query(NASJob).all()
        return [{
            "id": j.id,
            "status": j.status,
            "config": j.config,
            "created_at": j.created_at,
            "updated_at": j.updated_at
        } for j in jobs]
    finally:
        session.close()

@app.get("/nas/job/{job_id}")
def get_job(job_id: int):
    session = SessionLocal()
    try:
        job = session.query(NASJob).filter(NASJob.id == job_id).first()
        if not job:
            return JSONResponse(status_code=404, content={"error": "Job not found"})
        return {
            "id": job.id,
            "status": job.status,
            "config": job.config,
            "result": job.result,
            "error": job.error if hasattr(job, 'error') else None,
            "created_at": job.created_at,
            "updated_at": job.updated_at
        }
    finally:
        session.close() 