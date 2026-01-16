import os
import time
from pathlib import Path

from fastapi.testclient import TestClient

from api.app import get_app, graph_service

client = TestClient(get_app())


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_graph_and_train_dryrun(tmp_path: Path):
    # Add event nodes
    r = client.post("/graph/events", json={"text": "Event 1", "meta": {"source": "test"}})
    assert r.status_code == 200
    n1 = r.json()["node_id"]

    r = client.post("/graph/events", json={"text": "Event 2"})
    assert r.status_code == 200
    n2 = r.json()["node_id"]

    # Add edge
    r = client.post("/graph/edges", json={"src": n1, "dst": n2, "weight": 1.5})
    assert r.status_code == 200

    # Save graph to tmp path
    out = tmp_path / "graph.json"
    r = client.post("/graph/save", params={"path": str(out)})
    assert r.status_code == 200
    assert out.exists()

    # Start a dry-run training job
    workdir = tmp_path / "work"
    r = client.post("/train/event-extraction", json={"dry_run": True, "output_dir": str(workdir)})
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    # Poll for completion (small timeout)
    status = None
    for _ in range(20):
        r = client.get(f"/jobs/{job_id}")
        assert r.status_code == 200
        body = r.json()
        status = body["status"]
        if status in ("done", "failed"):
            break
        time.sleep(0.1)

    assert status == "done"
    # manifest should exist
    manifest = workdir / "manifest.json"
    assert manifest.exists()
