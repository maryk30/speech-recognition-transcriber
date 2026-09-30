"""
Score the full pipeline on every AMI excerpt and write a committed result.

    python scripts/baseline.py                                   # stock Whisper-small
    python scripts/baseline.py --model models/whisper-small-ami-mlx --tag tuned
    python scripts/baseline.py --meetings ES2004c_300s IS1009b_300s

Writes results/baseline_<tag>.json (all metrics + transcripts' line counts +
run metadata) and results/baseline_<tag>.md (table). Needs the models (HF
access + HF_TOKEN) -- the scoring itself is model-free, see src/meeting_report.py.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import scoring  # noqa: E402
from meeting_report import format_markdown, score_meeting  # noqa: E402
from pipeline import TranscriptionPipeline  # noqa: E402

DEFAULT_MEETINGS = ["ES2004c_300s", "IS1009b_300s", "EN2002b_300s"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meetings", nargs="*", default=DEFAULT_MEETINGS)
    ap.add_argument("--model", default="small", help="Whisper size or path")
    ap.add_argument("--tag", default="stock")
    ap.add_argument("--num-speakers", type=int, default=4)
    ap.add_argument("--no-separation", action="store_true")
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = ROOT / "data" / "ami_meetings"

    pipe = TranscriptionPipeline(asr_model_size=args.model, separate=not args.no_separation)
    rows = []
    for meeting in args.meetings:
        ref_path = data / f"{meeting}.json"
        ref = scoring.load_reference(str(ref_path))
        diar_ref = scoring.load_diarization_reference(str(ref_path))
        t0 = time.time()
        lines = pipe.run(str(data / f"{meeting}.wav"), num_speakers=args.num_speakers)
        elapsed = time.time() - t0
        hyp = [scoring.Turn(l.speaker, l.start, l.end, l.text) for l in lines]
        hyp_diar = [scoring.Turn(t.speaker_label, t.start, t.end) for t in pipe.last_turns]
        row = score_meeting(ref, hyp, diar_ref, hyp_diar)
        row.update(meeting=meeting, asr=args.tag, lines=len(lines), seconds=round(elapsed, 1))
        rows.append(row)
        print(f"{meeting}: DER {row['der']:.1%}  cpWER {row['cpwer']:.1%}  tcpWER {row['tcpwer']:.1%}", flush=True)

    meta = {
        "model": args.model, "tag": args.tag, "num_speakers": args.num_speakers,
        "separation": not args.no_separation, "device": pipe.device,
        "platform": platform.platform(), "date": time.strftime("%Y-%m-%d"),
    }
    (out_dir / f"baseline_{args.tag}.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=1, default=str))
    (out_dir / f"baseline_{args.tag}.md").write_text(format_markdown(rows))
    print(f"wrote {out_dir}/baseline_{args.tag}.json and .md")


if __name__ == "__main__":
    main()
