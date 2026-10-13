import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from rerun_main_realdata_fixed_adaptive import run_all  # noqa: E402


def test_formal_realdata_run_rejects_nonempty_output_directory(tmp_path):
    out_dir = tmp_path / "formal"
    out_dir.mkdir()
    (out_dir / "stale.txt").write_text("stale", encoding="utf-8")
    with pytest.raises(FileExistsError, match="not empty"):
        run_all(
            selected_labels={"seattle"},
            out_dir=out_dir,
            paper_dir=tmp_path / "paper",
            fig_dir=tmp_path / "figures",
        )


def test_realdata_run_rejects_unknown_dataset_label(tmp_path):
    with pytest.raises(ValueError, match="Unknown real-data labels"):
        run_all(
            selected_labels={"seattle", "typo"},
            out_dir=tmp_path / "formal",
            paper_dir=tmp_path / "paper",
            fig_dir=tmp_path / "figures",
        )


def test_formal_run_directory_label_is_portable():
    out_dir = Path("formal-output")
    target = out_dir / "seattle_final_fixed_adaptive"
    assert target.relative_to(out_dir).as_posix() == "seattle_final_fixed_adaptive"
