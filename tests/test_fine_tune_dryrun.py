import os
from pathlib import Path

from train_scripts import fine_tune_event_extraction as fte


def test_fine_tune_dryrun_creates_manifest(tmp_path: Path):
    # ensure work dir is isolated
    work = tmp_path / "fte_out"
    os.environ["WORK_DIR"] = str(work)
    res = fte.main(dry_run=True)
    assert res == 0
    manifest = work / "manifest.json"
    assert manifest.exists()
