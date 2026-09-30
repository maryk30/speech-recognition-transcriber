import random
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))

from train_utils import (BestTracker, augment_hidden, load_trainer_state,  # noqa: E402
                         save_trainer_state)


def test_augment_masks_but_does_not_mutate_input():
    h = torch.ones(1500, 768)
    out = augment_hidden(h, rng=random.Random(0))
    assert h.min() == 1.0                      # input untouched
    assert 0 < (out == 0).float().mean() < 0.3  # some, not most, masked
    assert out.shape == h.shape


def test_augment_masks_expected_span_sizes():
    h = torch.ones(1500, 768)
    out = augment_hidden(h, n_time=1, n_chan=0, rng=random.Random(1))
    assert int((out[:, 0] == 0).sum()) == 100   # exactly one 100-frame span
    out = augment_hidden(h, n_time=0, n_chan=1, rng=random.Random(1))
    assert int((out[0] == 0).sum()) == 64


def test_augment_skips_when_too_small():
    h = torch.ones(50, 32)
    assert torch.equal(augment_hidden(h, rng=random.Random(0)), h)


def test_best_tracker_only_improves():
    b = BestTracker()
    assert b.update(0.5) and b.update(0.4)
    assert not b.update(0.4) and not b.update(0.6)
    assert b.best == 0.4


def test_trainer_state_round_trip(tmp_path):
    p = torch.nn.Parameter(torch.zeros(3))
    opt = torch.optim.AdamW([p], lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0 / (s + 1))
    for _ in range(3):
        p.grad = torch.ones(3)
        opt.step(); sched.step()
    assert load_trainer_state(tmp_path) is None
    save_trainer_state(tmp_path, 3, opt, sched, 0.42)

    state = load_trainer_state(tmp_path)
    assert state["step"] == 3 and state["best_wer"] == 0.42
    p2 = torch.nn.Parameter(torch.zeros(3))
    opt2 = torch.optim.AdamW([p2], lr=1e-3, weight_decay=0.01)
    sched2 = torch.optim.lr_scheduler.LambdaLR(opt2, lambda s: 1.0 / (s + 1))
    opt2.load_state_dict(state["optimizer"]); sched2.load_state_dict(state["scheduler"])
    assert sched2.last_epoch == 3
    assert opt2.state_dict()["state"][0]["step"] == opt.state_dict()["state"][0]["step"]
