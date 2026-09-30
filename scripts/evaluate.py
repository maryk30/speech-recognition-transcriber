"""
Score the full pipeline against ground truth (Phases 4 and 6).

    python scripts/evaluate.py data/samples/tts_demo.wav --reference data/samples/tts_demo_truth.json
    python scripts/evaluate.py meeting.wav --reference meeting.rttm --num-speakers 4

Reference formats:
    JSON  {"turns": [{"speaker": "A", "start": 0.0, "end": 5.2, "text": "..."}]}
    RTTM  standard diarization ground truth (AMI, CHiME, ...); no text, so DER only

Reports:
    DER          diarization error rate incl. overlap (+ missed / false alarm / confusion)
    cpWER        multi-speaker WER, fillers ignored on both sides   (needs "text")
    cpWER+fill   same but fillers must match too
    filler recall  fraction of reference fillers retained in the transcript
    consistency  per true speaker: share of their speech under one label
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import scoring  # noqa: E402
from bootstrap import block_counts, bootstrap_ratio_ci  # noqa: E402
from pipeline import TranscriptionPipeline  # noqa: E402
from separation import DEFAULT_MODEL  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio")
    ap.add_argument("--reference", required=True)
    ap.add_argument("--num-speakers", type=int, default=None)
    ap.add_argument("--model", default="small", help="Whisper size")
    ap.add_argument("--sep-model", default=DEFAULT_MODEL)
    ap.add_argument("--no-separation", action="store_true")
    ap.add_argument("--verbatim-prompt", action="store_true")
    ap.add_argument("--collar", type=float, default=0.25, help="DER forgiveness collar (s)")
    args = ap.parse_args()

    ref = scoring.load_reference(args.reference)
    pipe = TranscriptionPipeline(
        asr_model_size=args.model,
        separate=not args.no_separation,
        separation_model=args.sep_model,
        verbatim_prompt=args.verbatim_prompt,
    )
    lines = pipe.run(args.audio, num_speakers=args.num_speakers)

    hyp_lines = [scoring.Turn(l.speaker, l.start, l.end, l.text) for l in lines]
    hyp_diar = [scoring.Turn(t.speaker_label, t.start, t.end) for t in pipe.last_turns]

    print("\n=== transcript")
    for l in lines:
        print(l.render())

    print("\n=== diarization")
    diar_ref = scoring.load_diarization_reference(args.reference)
    d = scoring.der(diar_ref, hyp_diar, collar=args.collar)
    print(f"DER {d['der']:.1%}   (missed {d['missed']:.1%}, false alarm {d['false_alarm']:.1%}, "
          f"confusion {d['confusion']:.1%}, collar {args.collar}s)")
    for spk, c in scoring.speaker_consistency(diar_ref, hyp_diar).items():
        print(f"consistency {spk}: {c:.1%}")

    if any(t.text for t in ref):
        print("\n=== transcription")
        w, mapping = scoring.cp_wer(ref, hyp_lines, drop_fillers=True)
        w_fill, _ = scoring.cp_wer(ref, hyp_lines, drop_fillers=False)
        counts = block_counts(ref, hyp_lines, mapping)
        _, lo, hi = bootstrap_ratio_ci([e for e, _ in counts], [n for _, n in counts])
        print(f"cpWER (fillers ignored)   {w:.1%}   [95% CI {lo:.1%}-{hi:.1%}, 30 s block bootstrap]")
        print(f"cpWER (fillers scored)    {w_fill:.1%}")
        found = total = 0
        ref_text = " ".join(t.text for t in ref)
        hyp_text = " ".join(t.text for t in hyp_lines)
        found, total = scoring.filler_recall(ref_text, hyp_text)
        if total:
            print(f"filler recall             {found}/{total}")
        print(f"speaker mapping (hyp -> ref): {mapping}")
    else:
        print("\n(reference has no text: WER skipped)")


if __name__ == "__main__":
    main()
