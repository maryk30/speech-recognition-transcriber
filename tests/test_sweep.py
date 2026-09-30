import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))

from sweep import parse_val_log, summarise  # noqa: E402

LOG = """step 0 | val loss 2.100 (before training)
step 100/300 | loss 1.900 | lr 1.00e-05 | 2.0s/step | eta 7 min
step 100 | val loss 1.800 | val WER 0.420 (best 0.420) *
step 200 | val loss 1.700 | val WER 0.390 (best 0.390) *
step 300 | val loss 1.650 | val WER 0.405 (best 0.390)
"""


def test_parse_val_log_reads_wer_lines_only():
    assert parse_val_log(LOG) == [(100, 1.8, 0.42), (200, 1.7, 0.39), (300, 1.65, 0.405)]


def test_summarise_picks_best_not_last():
    row = summarise("x", 1e-5, 2, 300, LOG, 12.3, 0)
    assert row["best_val_wer"] == 0.39 and row["best_step"] == 200
    assert row["final_val_wer"] == 0.405


def test_summarise_failed_run_has_empty_metrics():
    row = summarise("x", 1e-5, 2, 300, "Traceback ...", 0.1, 1)
    assert row["best_val_wer"] == "" and row["returncode"] == 1
