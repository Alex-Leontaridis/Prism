from fastapi import FastAPI, BackgroundTasks, UploadFile, File, Form, Depends, WebSocket
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import threading
import uuid
import os
import json
from nas_search import nas_search
from db import SessionLocal, Job, init_db
from sqlalchemy.orm import Session

init_db()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to ["http://localhost:8080"] for more security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to store latest progress
latest_progress = {"iteration": 0, "total_trials": 1, "status": "Waiting for job..."}

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class NASJobConfig(BaseModel):
    strategy: str = 'random'
    max_params: Optional[int] = None
    min_accuracy: Optional[float] = None
    prior_arch: Optional[str] = None
    log_method: str = 'json'

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/strategies")
def list_strategies():
    return {"strategies": [
        'random', 'evolutionary', 'bayesian', 'optuna', 'nsga2', 'oneshot', 'reinforcement', 'grid']}

@app.post("/nas/start")
def start_nas_job(config: NASJobConfig, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    job = Job(status="running", config=json.dumps(config.dict()))
    db.add(job)
    db.commit()
    db.refresh(job)
    job_id = job.id
    def run_job():
        try:
            def progress_callback(progress):
                global latest_progress
                latest_progress = progress
            nas_search(
                strategy_name=config.strategy,
                constraints={k: v for k, v in [("max_params", config.max_params), ("min_accuracy", config.min_accuracy)] if v is not None},
                prior_arch_path=config.prior_arch,
                log_method=config.log_method,
                progress_callback=progress_callback
            )
            # For demo, just load the last run from nas_history.json
            result = None
            if os.path.exists('nas_history.json'):
                with open('nas_history.json') as f:
                    result = json.load(f)
            db_job = db.query(Job).filter(Job.id == job_id).first()
            db_job.status = "completed"
            db_job.result = json.dumps(result)
            db.commit()
        except Exception as e:
            db_job = db.query(Job).filter(Job.id == job_id).first()
            db_job.status = "failed"
            db_job.error = str(e)
            db.commit()
    background_tasks.add_task(run_job)
    return {"job_id": job_id}

@app.get("/nas/jobs")
def list_jobs(db: Session = Depends(get_db)):
    jobs = db.query(Job).all()
    return [{"id": j.id, "status": j.status, "config": j.config, "created_at": j.created_at, "updated_at": j.updated_at} for j in jobs]

@app.get("/nas/job/{job_id}")
def get_job(job_id: int, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        return JSONResponse(status_code=404, content={"error": "Job not found"})
    return {"id": job.id, "status": job.status, "config": job.config, "result": job.result, "error": job.error, "created_at": job.created_at, "updated_at": job.updated_at}

@app.post("/nas/upload-dataset")
def upload_dataset(file: UploadFile = File(...)):
    # Stub: Save file to data/uploads
    os.makedirs('data/uploads', exist_ok=True)
    file_path = os.path.join('data/uploads', file.filename)
    with open(file_path, 'wb') as f:
        f.write(file.file.read())
    return {"filename": file.filename, "status": "uploaded"}

@app.post("/nas/demo")
def create_demo_job(db: Session = Depends(get_db)):
    """Create a demo job with existing experiment results for testing the Monitor"""
    try:
        # Load existing results from nas_history.json
        result = None
        if os.path.exists('nas_history.json'):
            with open('nas_history.json', 'r') as f:
                result = json.load(f)

        # Create a demo job
        demo_config = {
            "strategy": "evolutionary",
            "max_params": 500000,
            "dataset": "MNIST",
            "log_method": "json"
        }

        job = Job(
            status="completed",
            config=json.dumps(demo_config),
            result=json.dumps(result) if result else None
        )
        db.add(job)
        db.commit()
        db.refresh(job)

        return {"job_id": job.id, "message": "Demo job created successfully"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.websocket("/ws/progress")
async def websocket_progress(websocket: WebSocket):
    await websocket.accept()
    import asyncio
    try:
        while True:
            await websocket.send_json(latest_progress)
            await asyncio.sleep(0.5)
    except Exception:
        pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True) 