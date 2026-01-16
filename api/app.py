"""Production-oriented FastAPI application for the FERS project.

Features:
- Health and readiness endpoints
- Background job manager to run training tasks (dry-run or full)
- Event graph management endpoints (add node, add edge, list nodes, save/load)
- Structured logging and environment-configurable settings

This module aims for clarity and testability: heavy ML imports only occur when
full training is requested, and background work is executed in a ThreadPool.
"""
from __future__ import annotations

import os
import uuid
import json
import logging
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Any, Dict, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, BaseSettings

from services.langgraph_orchestrator import EventGraphService

logger = logging.getLogger("fers.api")
logging.basicConfig(level=logging.INFO)


class Settings(BaseSettings):
    work_dir: str = os.environ.get("WORK_DIR", "outputs")
    max_workers: int = int(os.environ.get("API_MAX_WORKERS", "2"))
    host: str = os.environ.get("API_HOST", "0.0.0.0")
    port: int = int(os.environ.get("API_PORT", "8000"))


def get_settings() -> Settings:
    return Settings()


app = FastAPI(title="FERS Orchestrator API", version="0.1.0")

# Simple in-memory job manager. For production, replace with durable queue.
class JobManager:
    def __init__(self, max_workers: int = 2):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._jobs: Dict[str, Dict[str, Any]] = {}

    def submit(self, fn, *args, **kwargs) -> str:
        job_id = uuid.uuid4().hex
        future: Future = self._executor.submit(fn, *args, **kwargs)
        self._jobs[job_id] = {"future": future, "status": "running", "result": None, "error": None}

        def _cb(f: Future) -> None:  # callback executed in thread when done
            try:
                res = f.result()
                self._jobs[job_id]["status"] = "done"
                self._jobs[job_id]["result"] = res
                logger.info("Job %s completed: %s", job_id, res)
            except Exception as exc:  # pragma: no cover - runtime exception path
                self._jobs[job_id]["status"] = "failed"
                self._jobs[job_id]["error"] = str(exc)
                logger.exception("Job %s failed", job_id)

        future.add_done_callback(_cb)
        return job_id

    def status(self, job_id: str) -> Dict[str, Any]:
        info = self._jobs.get(job_id)
        if info is None:
            raise KeyError(job_id)
        return {"status": info["status"], "result": info.get("result"), "error": info.get("error")}


# application-wide singletons
_settings = get_settings()
job_manager = JobManager(max_workers=_settings.max_workers)
graph_service = EventGraphService()


class TrainRequest(BaseModel):
    dry_run: bool = True
    config_path: Optional[str] = None
    seed: int = 42
    output_dir: Optional[str] = None


class TrainResponse(BaseModel):
    job_id: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    result: Optional[Any] = None
    error: Optional[str] = None


class AddEventRequest(BaseModel):
    text: str
    meta: Optional[Dict[str, Any]] = None


class AddEdgeRequest(BaseModel):
    src: int
    dst: int
    weight: float = 1.0


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/train/event-extraction", response_model=TrainResponse)
def train_event_extraction(req: TrainRequest, settings: Settings = Depends(get_settings)) -> TrainResponse:
    """Start a training job and return a job id. By default starts in dry-run mode.

    NOTE: For full training set `dry_run` to False and ensure the environment has
    the required heavy dependencies installed.
    """
    # import lazily to avoid heavy top-level imports
    from train_scripts import fine_tune_event_extraction as fte

    def _run(config_path: Optional[str], dry_run: bool, seed: int, output_dir: Optional[str]):
        # this runs inside the threadpool
        try:
            return fte.main(config_path=config_path, dry_run=dry_run, seed=seed, output_dir=output_dir)
        except Exception as exc:  # pragma: no cover - runtime behavior
            logger.exception("Training job raised an exception")
            raise

    out_dir = req.output_dir or os.path.join(settings.work_dir, "event_extractor")
    job_id = job_manager.submit(_run, req.config_path, req.dry_run, req.seed, out_dir)
    logger.info("Submitted training job %s (dry_run=%s)", job_id, req.dry_run)
    return TrainResponse(job_id=job_id)


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str):
    try:
        st = job_manager.status(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="job not found")
    return JobStatusResponse(job_id=job_id, status=st["status"], result=st.get("result"), error=st.get("error"))


@app.post("/graph/events")
def add_event(req: AddEventRequest):
    node_id = graph_service.add_event(req.text, meta=req.meta)
    return {"node_id": node_id}


@app.post("/graph/edges")
def add_edge(req: AddEdgeRequest):
    try:
        graph_service.add_edge(req.src, req.dst, weight=req.weight)
    except KeyError:
        raise HTTPException(status_code=404, detail="src or dst node not found")
    return {"status": "ok"}


@app.get("/graph/nodes")
def list_nodes():
    nodes = [{"id": n, **data} for n, data in graph_service.graph.nodes(data=True)]
    return {"nodes": nodes}


@app.post("/graph/save")
def save_graph(path: Optional[str] = None):
    path = Path(path or os.path.join(_settings.work_dir, "event_graph.json"))
    graph_service.save(path)
    return {"path": str(path)}


@app.post("/graph/load")
def load_graph(path: str):
    p = Path(path)
    try:
        new_svc = EventGraphService.load(p)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="file not found")
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid graph file")
    # replace singleton graph (simple strategy for this demo)
    global graph_service
    graph_service = new_svc
    return {"status": "loaded"}


def get_app() -> FastAPI:
    return app


if __name__ == "__main__":  # pragma: no cover - manual run
    import uvicorn

    uvicorn.run(get_app(), host=_settings.host, port=_settings.port)
