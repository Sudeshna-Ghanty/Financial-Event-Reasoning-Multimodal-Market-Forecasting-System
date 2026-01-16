
"""Fine-tune an event extractor (T5) with a safe CLI and dry-run mode.

This module exposes a `main(dry_run=False)` entry so tests and CI can import
it without pulling heavy dependencies. When `dry_run=True` it runs a small,
deterministic smoke-flow that validates preprocessing and I/O.
"""
from __future__ import annotations

from typing import Dict, List, Optional
from pathlib import Path
import argparse
import logging
import random
import os
import json
from datetime import datetime

logger = logging.getLogger(__name__)


MODEL = "t5-small"
MAX_INPUT_LEN = 256
MAX_TARGET_LEN = 64


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
        # torch may not be installed in dry-run/test envs
        pass


def _mock_tokenizer(text: str, max_length: int) -> Dict[str, List[int]]:
    # Very small deterministic tokenizer used for dry-run / tests.
    ids = [ord(c) % 256 for c in text][:max_length]
    padding = [0] * (max_length - len(ids))
    return {"input_ids": ids + padding, "attention_mask": [1] * len(ids) + padding}


def preprocess_examples(examples: List[Dict[str, str]], tokenizer_func, max_input=MAX_INPUT_LEN, max_target=MAX_TARGET_LEN):
    out = []
    for ex in examples:
        inp = tokenizer_func(ex["text"], max_length=max_input)
        tgt = tokenizer_func(ex.get("events", ""), max_length=max_target)
        inp["labels"] = tgt["input_ids"]
        out.append(inp)
    return out


def _save_resolved_config(conf: Dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = out_dir / "config_used.json"
    with cfg_path.open("w", encoding="utf-8") as fh:
        json.dump(conf, fh, indent=2)


def run_full_training(config: Optional[Dict] = None, seed: int = 42) -> int:
    # Heavy path: import transformers/datasets and run Seq2Seq training.
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
    import datasets

    model_name = config.get("model_name", MODEL) if config else MODEL
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    data_files = config.get("data_files") if (config and config.get("data_files")) else {
        "train": "data/event_train.jsonl",
        "validation": "data/event_val.jsonl",
    }

    ds = datasets.load_dataset("json", data_files=data_files)

    def _pre(ex):
        return tokenizer(ex["text"], padding="max_length", truncation=True, max_length=MAX_INPUT_LEN) | {
            "labels": tokenizer(ex["events"], padding="max_length", truncation=True, max_length=MAX_TARGET_LEN)[
                "input_ids"
            ]
        }

    ds = ds.map(lambda x: _pre(x), batched=False)

    out_dir = Path(config.get("output_dir", "outputs/event_extractor")) if config else Path("outputs/event_extractor")
    training_kwargs = dict(
        output_dir=str(out_dir),
        do_train=True,
        do_eval=True,
        eval_strategy=config.get("eval_strategy", "epoch") if config else "epoch",
        save_strategy=config.get("save_strategy", "epoch") if config else "epoch",
        save_total_limit=config.get("save_total_limit", 3) if config else 3,
        per_device_train_batch_size=config.get("train_batch_size", 4) if config else 4,
        per_device_eval_batch_size=config.get("eval_batch_size", 4) if config else 4,
        learning_rate=config.get("learning_rate", 3e-4) if config else 3e-4,
        num_train_epochs=config.get("num_train_epochs", 3) if config else 3,
        logging_steps=config.get("logging_steps", 10) if config else 10,
    )

    args = Seq2SeqTrainingArguments(**training_kwargs)

    trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["validation"], tokenizer=tokenizer)

    # save resolved config used for this experiment
    _save_resolved_config(config or {}, out_dir)

    # set seeds for reproducibility
    set_seed(seed)

    trainer.train()
    return 0


def main(config_path: Optional[str] = None, dry_run: bool = False, seed: int = 42, output_dir: Optional[str] = None) -> int:
    """Run training. When dry_run=True the function executes a fast path that
    avoids large imports and performs basic checks.

    Returns 0 on success.
    """
    set_seed(42)
    logging.basicConfig(level=logging.INFO)

    config = {}
    if config_path:
        # try JSON first, then YAML (if available)
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
        logger.info("Running fine-tune script in dry-run mode")
        # create a tiny deterministic dataset
        examples = [
            {"text": "Company X reported earnings", "events": "earnings_report"},
            {"text": "CEO announced a new product", "events": "product_announcement"},
        ]
        processed = preprocess_examples(examples, lambda t, max_length: _mock_tokenizer(t, max_length))
        # basic validation
        assert all("input_ids" in p and "labels" in p for p in processed)
        out_dir = Path(config.get("output_dir", os.environ.get("WORK_DIR", "outputs/event_extractor_dry")))
        out_dir.mkdir(parents=True, exist_ok=True)
        # write a minimal manifest
        with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
            json.dump({"examples": len(processed), "seed": seed, "timestamp": datetime.utcnow().isoformat()}, f)
        logger.info("Dry-run completed, manifest written to %s", out_dir)
        return 0

    # full training path
    return run_full_training(config=config, seed=seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Run a fast dry-run without heavy deps")
    parser.add_argument("--config", type=str, help="Path to JSON/YAML config")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, help="Override output directory")
    args = parser.parse_args()
    raise SystemExit(main(config_path=args.config, dry_run=args.dry_run, seed=args.seed, output_dir=args.output_dir))

