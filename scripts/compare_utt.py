"""
Paired comparison of two ASR models on the SAME utterances
(the --save files written by training/asr_eval.py).

    python scripts/compare_utt.py output/utt_stock.json output/utt_tuned.json

For each metric prints both rates, the difference (B - A) with a 95% paired
bootstrap interval, and how often B was NOT better across resamples (a
one-sided bootstrap p-value). Separate per-model intervals overlapping does
not mean "no difference"; this test is the right one for same-item scores.
Also lists the utterances where B lost the most words vs A, to read by eye.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bootstrap import paired_bootstrap_diff  # noqa: E402

METRICS = [
    ("WER (fillers ignored)", "errors", "words", True),
    ("WER (verbatim)", "v_errors", "v_words", True),
    ("filler recall", "fillers_found", "fillers_total", False),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--worst", type=int, default=8, help="show this many utterances where B is worst vs A")
    args = ap.parse_args()

    A, B = (json.loads(Path(p).read_text()) for p in (args.a, args.b))
    if len(A) != len(B) or any(x["ref"] != y["ref"] for x, y in zip(A, B)):
        sys.exit("the two files are not the same utterances in the same order (same -n / --seed / shards?)")
    if "errors" not in A[0]:
        sys.exit("old --save format without per-utterance counts; re-run asr_eval.py from this branch")

    print(f"A = {args.a}\nB = {args.b}\n{len(A)} paired utterances\n")
    for name, num, den, lower_is_better in METRICS:
        if num not in A[0]:
            continue
        r = paired_bootstrap_diff([x[num] for x in A], [y[num] for y in B], [x[den] for x in A], n_boot=args.n_boot)
        better = r["p_b_not_better"] if lower_is_better else 1 - r["p_b_not_better"]
        verdict = "B better" if r["high"] < 0 and lower_is_better or r["low"] > 0 and not lower_is_better else \
                  "B worse" if r["low"] > 0 and lower_is_better or r["high"] < 0 and not lower_is_better else "no clear difference"
        print(f"{name:24s} A {r['rate_a']:6.1%}   B {r['rate_b']:6.1%}   B-A {r['diff']:+6.1%} "
              f"[95% CI {r['low']:+.1%} .. {r['high']:+.1%}]   P(B not better) {better:.3f}   -> {verdict}")

    losses = sorted(zip(A, B), key=lambda p: p[1]["errors"] - p[0]["errors"], reverse=True)[: args.worst]
    print(f"\nutterances where B has the most extra errors (fillers ignored):")
    for x, y in losses:
        if y["errors"] <= x["errors"]:
            break
        print(f"  +{y['errors'] - x['errors']:<3d} REF {x['ref']}\n       A   {x['hyp']}\n       B   {y['hyp']}")


if __name__ == "__main__":
    main()
