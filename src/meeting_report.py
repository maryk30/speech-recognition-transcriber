"""
One meeting's full score card, shared by scripts/baseline.py (and usable from
scripts/evaluate.py): our metrics, plus MeetEval's cpWER / tcpWER as a
cross-check. Pure scoring -- takes already-produced hypothesis turns, so it
runs without any model.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import scoring
from bootstrap import block_counts, bootstrap_ratio_ci
from scoring import Turn


def score_meeting(
    ref_turns: Sequence[Turn],
    hyp_lines: Sequence[Turn],
    diar_ref: Optional[Sequence[Turn]] = None,
    hyp_diar: Optional[Sequence[Turn]] = None,
    collar: float = 0.25,
    tcp_collar: float = 5.0,
) -> Dict[str, object]:
    """Keys: cpwer, cpwer_fillers (fillers must match too), tcpwer,
    meeteval_cpwer (should equal cpwer), filler_found / filler_total,
    and -- when diarization turns are given -- der/missed/false_alarm/
    confusion plus per-reference-speaker consistency. cpwer_ci is a 95%
    block-bootstrap interval (30 s blocks, fixed speaker mapping)."""
    from meeteval_scoring import meeteval_cpwer, meeteval_tcpwer

    out: Dict[str, object] = {}
    out["cpwer"], _ = scoring.cp_wer(ref_turns, hyp_lines, drop_fillers=True)
    out["cpwer_fillers"], mapping = scoring.cp_wer(ref_turns, hyp_lines, drop_fillers=False)
    out["speaker_mapping"] = mapping
    _, cp_map = scoring.cp_wer(ref_turns, hyp_lines, drop_fillers=True)
    counts = block_counts(ref_turns, hyp_lines, cp_map, drop_fillers=True)
    _, lo, hi = bootstrap_ratio_ci([e for e, _ in counts], [n for _, n in counts])
    out["cpwer_ci"] = (lo, hi)
    out["meeteval_cpwer"] = meeteval_cpwer(ref_turns, hyp_lines)["wer"]
    out["tcpwer"] = meeteval_tcpwer(ref_turns, hyp_lines, collar=tcp_collar)["wer"]
    found, total = scoring.filler_recall(
        " ".join(t.text for t in ref_turns), " ".join(t.text for t in hyp_lines)
    )
    out["filler_found"], out["filler_total"] = found, total
    if diar_ref is not None and hyp_diar is not None:
        out.update(scoring.der(list(diar_ref), list(hyp_diar), collar=collar))
        out["consistency"] = scoring.speaker_consistency(diar_ref, hyp_diar)
    return out


def format_markdown(rows: List[Dict[str, object]]) -> str:
    """rows: dicts with 'meeting', 'asr' and score_meeting() keys."""
    def pct(v) -> str:
        return "-" if v is None else f"{v:.1%}"

    lines = [
        "| meeting | ASR | DER | cpWER | cpWER 95% CI | tcpWER | cpWER+fillers | filler recall |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        total = r.get("filler_total") or 0
        recall = f"{r['filler_found']}/{total}" if total else "-"
        lo, hi = r.get("cpwer_ci") or (None, None)
        ci = "-" if lo is None else f"{lo:.1%}-{hi:.1%}"
        lines.append(
            f"| {r['meeting']} | {r['asr']} | {pct(r.get('der'))} | {pct(r['cpwer'])} | {ci} | "
            f"{pct(r['tcpwer'])} | {pct(r['cpwer_fillers'])} | {recall} |"
        )
    return "\n".join(lines) + "\n"
