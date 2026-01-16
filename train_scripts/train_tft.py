
"""Train a Temporal Fusion Transformer with a safe dry-run for CI.

The module exposes `main(dry_run=False)` so tests can run without requiring
`pytorch-forecasting` or `lightning` to be installed.
"""
from __future__ import annotations

from typing import Optional
import argparse
import logging
import random
from pathlib import Path
import os
import json
from datetime import datetime

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    try:
        import numpy as _np

        _np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch as _torch

        _torch.manual_seed(seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _save_resolved_config(conf: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = out_dir / "config_used.json"
    with cfg_path.open("w", encoding="utf-8") as fh:
        json.dump(conf, fh, indent=2)


def run_full_training(config: Optional[dict] = None, seed: int = 42) -> int:
    # Full training path — imports are local so tests that import this module
    # won't fail unless they explicitly opt into running full training.
    import pandas as pd
    import torch
    from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
    from lightning.pytorch import Trainer
    from lightning.pytorch.callbacks import ModelCheckpoint

    df = pd.read_csv(config.get("data_csv", "data/ts_data.csv") if config else "data/ts_data.csv")

    dataset = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="target",
        group_ids=["group_id"],
        max_encoder_length=config.get("max_encoder_length", 30) if config else 30,
        max_prediction_length=config.get("max_prediction_length", 1) if config else 1,
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=["open", "high", "low", "close", "volume", "target"],
    )

    loader = dataset.to_dataloader(train=True, batch_size=config.get("batch_size", 32) if config else 32)

    model = TemporalFusionTransformer.from_dataset(dataset)

    out_dir = Path(config.get("output_dir", "outputs/tft")) if config else Path("outputs/tft")
    checkpoint_callback = ModelCheckpoint(dirpath=str(out_dir / "checkpoints"), save_top_k=config.get("save_top_k", 3) if config else 3)

    trainer = Trainer(accelerator=config.get("accelerator", "cpu") if config else "cpu", devices=config.get("devices", 1) if config else 1, max_epochs=config.get("max_epochs", 10) if config else 10, callbacks=[checkpoint_callback])
    # save resolved config
    _save_resolved_config(config or {}, out_dir)
    set_seed(seed)
    trainer.fit(model, loader)
    return 0


def main(config_path: Optional[str] = None, dry_run: bool = False, seed: int = 42, output_dir: Optional[str] = None) -> int:
    set_seed(seed)
    logging.basicConfig(level=logging.INFO)
    config = {}
    if config_path:
        try:
            with open(config_path, "r", encoding="utf-8") as fh:
                config = json.load(fh)
        except Exception:
            try:
                import yaml

                with open(config_path, "r", encoding="utf-8") as fh:
                    config = yaml.safe_load(fh)
            except Exception:
                logger.warning("Could not parse config at %s; continuing with defaults", config_path)

    if output_dir:
        config["output_dir"] = output_dir

    if dry_run:
        logger.info("Running TFT training in dry-run mode")
        # create a small deterministic DataFrame-like structure and validate
        # expected columns.
        import pandas as pd

        df = pd.DataFrame(
            {
                "time_idx": [0, 1, 2, 3],
                "group_id": [0, 0, 0, 0],
                "target": [1.0, 1.1, 1.2, 1.3],
                "open": [1.0, 1.05, 1.1, 1.15],
                "high": [1.1, 1.15, 1.2, 1.25],
                "low": [0.9, 0.95, 1.0, 1.05],
                "close": [1.05, 1.1, 1.15, 1.2],
                "volume": [100, 110, 120, 130],
            }
        )
        # basic validation that columns exist
        required = {"time_idx", "group_id", "target", "open", "high", "low", "close", "volume"}
        assert required.issubset(set(df.columns))
        out_dir = Path(config.get("output_dir", os.environ.get("WORK_DIR", "outputs/tft_dry")))
        out_dir.mkdir(parents=True, exist_ok=True)
        # write a minimal manifest
        with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
            json.dump({"rows": len(df), "seed": seed, "timestamp": datetime.utcnow().isoformat()}, f)
        logger.info("Dry-run completed, manifest written to %s", out_dir)
        return 0

    return run_full_training(config=config, seed=seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Run a fast dry-run without heavy deps")
    parser.add_argument("--config", type=str, help="Path to JSON/YAML config")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, help="Override output directory")
    args = parser.parse_args()
    raise SystemExit(main(config_path=args.config, dry_run=args.dry_run, seed=args.seed, output_dir=args.output_dir))

