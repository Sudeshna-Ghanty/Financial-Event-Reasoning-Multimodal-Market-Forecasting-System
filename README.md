# FERS_PROJECT_FULL

This repository contains a small research & training infrastructure for
event extraction and forecasting. The codebase includes:

- `services/langgraph_orchestrator.py`: lightweight event graph service (save/load).
- `train_scripts/fine_tune_event_extraction.py`: T5-based event extraction training script with CLI and dry-run mode.
- `train_scripts/train_tft.py`: Temporal Fusion Transformer training script with CLI and dry-run mode.
- `api/app.py`: FastAPI application to orchestrate jobs and manage the event graph.

This README focuses on reproducibility and how to run quick checks (dry-run)
and full training (heavy, requires ML dependencies).

## Quick start (dry-run, fast)

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install runtime deps (un-pinned):

```powershell
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

3. Run quick smoke tests (these avoid heavy ML imports):

```powershell
# run pytest for only the fast dry-run tests
pytest -q tests/test_fine_tune_dryrun.py tests/test_train_tft_dryrun.py tests/test_api.py
```

4. Run a dry-run training from the CLI:

```powershell
python train_scripts\fine_tune_event_extraction.py --dry-run --output-dir outputs/fte_demo
python train_scripts\train_tft.py --dry-run --output-dir outputs/tft_demo
```

## Full training (heavy)

Full training requires GPU-enabled libs and sufficient RAM. Before running
full training:

- Pin dependencies in `requirements.txt` or use an environment manager (conda).
- Ensure `torch`, `transformers`, `pytorch-forecasting`, and `lightning` are installed and compatible.

Run full training (example):

```powershell
python train_scripts\fine_tune_event_extraction.py --config configs/hyperparams_event.yaml --output-dir outputs/event_extractor_run
python train_scripts\train_tft.py --config configs/hyperparams_tft.yaml --output-dir outputs/tft_run
```

## API

Start the API (requires FastAPI & uvicorn):

```powershell
uvicorn api.app:get_app --reload --host 0.0.0.0 --port 8000
```

Endpoints:
- `GET /health` - health check
- `POST /graph/events` - add event node
- `POST /graph/edges` - add directed edge
- `GET /graph/nodes` - list nodes
- `POST /graph/save` - save graph to path
- `POST /train/event-extraction` - start training job (default dry-run)
- `GET /jobs/{job_id}` - check job status

## Reproducibility notes

- Use the `--seed` argument or set `seed` in your config to make runs deterministic.
- The scripts save the resolved config used for an experiment under the experiment `output_dir` as `config_used.json`.
- For exact reproducibility, pin package versions (recommended) and record GPU/CPU environment details.

## Next steps / suggestions

- Add GitHub Actions to run linting, mypy, and the dry-run tests for CI.
- Pin dependency versions and build a `requirements.lock` for reproducibility.
- For production job handling, replace the in-memory `JobManager` with a durable queue (Redis/RabbitMQ) and use a process supervisor.


# Financial Event Reasoning System (FERS)

End-to-end research-grade project that combines:
1) Financial event extraction from news using Transformer models (T5/BART/FinBERT-ready),
2) Graph-based causal reasoning (LangGraph-style orchestration),
3) Multimodal market forecasting using numerical OHLCV + textual signals (TFT-compatible),
4) Explainable inference via event graphs and REST API.

This repository is resume- and MS-application ready. Training scripts are real and complete.
