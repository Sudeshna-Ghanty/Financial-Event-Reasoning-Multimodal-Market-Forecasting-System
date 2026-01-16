import os
from pathlib import Path

from train_scripts import train_tft as tft


def test_tft_dryrun_creates_manifest(tmp_path: Path):
    work = tmp_path / "tft_out"
    os.environ["WORK_DIR"] = str(work)
    res = tft.main(dry_run=True)
    assert res == 0
    manifest = work / "manifest.json"
    assert manifest.exists()
