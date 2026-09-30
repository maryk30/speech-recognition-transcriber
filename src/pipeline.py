"""
Full offline pipeline (Phase 6).

    room audio
      -> diarization              consistent Speaker_N turns (may overlap)
      -> overlap detection / VAD  routing segments: single vs overlapping
      -> speaker centroids        ECAPA voice prints from each speaker's clean speech
      -> single segments          straight to verbatim ASR
      -> overlapping segments     SepFormer -> 2 streams -> match each stream to
                                  a speaker by voice -> verbatim ASR per stream
      -> optional enrollment      Speaker_N -> real name where confidently matched
      -> global chronological sort

Output lines look like:

    [mm:ss.s] NAME_or_Speaker_N: "verbatim text with [pause Xs] markers"

The live/streaming version (Phase 7) reuses `TranscriptionPipeline` and
just swaps "read a whole file" for a sliding-window buffer, which is why
the models are loaded once in __init__ rather than per call.
"""

from __future__ import annotations

import re
import sys
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from asr_baseline import Utterance, crop_utterances, make_transcriber
from audio_utils import load_audio, slice_audio
from config import TARGET_SR, get_hf_token, pick_device
from diarization import DiarizedSegment, Diarizer
from embeddings import SpeakerEmbedder, match_streams_to_speakers, speaker_centroids
from enrollment import EnrollmentGallery
from overlap import Segment, build_segments
from scoring import FILLERS, normalize
from separation import DEFAULT_MODEL as DEFAULT_SEPARATION_MODEL
from separation import OverlapSeparator, SeparatedStream

# Extra audio context transcribed on each side of every segment, so words at
# the segment edges aren't chopped in half. Words are then cropped back to the
# segment by midpoint (see asr_baseline.crop_utterances), so each word is
# still attributed to exactly one segment.
CONTEXT_PAD_S = 0.4
# A separated stream quieter than this fraction of the loudest one is
# treated as leakage residue, not a speaker (ASR would hallucinate on it).
MIN_STREAM_ENERGY_RATIO = 0.1


@dataclass
class AsrJob:
    """One clip to transcribe, plus how to turn its words into lines."""
    clip: np.ndarray
    clip_start: float        # where clip[0] sits in the recording (s)
    seg: Segment
    label: str
    strict: bool
    crop_lo: bool = True
    crop_hi: bool = True


@dataclass
class TranscriptLine:
    speaker: str
    start: float
    text: str
    end: float = 0.0
    from_overlap: bool = False   # produced from a separated overlap window

    def render(self) -> str:
        tenths = round(self.start * 10)   # round first so 59.96 -> 01:00.0, never 00:60.0
        mm, ss = divmod(tenths, 600)
        return f'[{mm:02d}:{ss / 10:04.1f}] {self.speaker}: "{self.text}"'


def _padded_clip(audio: np.ndarray, seg: Segment) -> tuple[np.ndarray, float]:
    """The segment plus CONTEXT_PAD_S each side; returns (clip, clip_start_s)."""
    lo = max(0.0, seg.start - CONTEXT_PAD_S)
    return slice_audio(audio, TARGET_SR, lo, seg.end + CONTEXT_PAD_S), lo


def _continues(neighbour: Optional[Segment], speaker: str, boundary: float) -> bool:
    """True if `neighbour` (within the context pad of `boundary`) also has
    `speaker` talking, i.e. that segment transcribes the same voice."""
    if neighbour is None or speaker not in neighbour.speakers:
        return False
    gap = min(abs(neighbour.end - boundary), abs(neighbour.start - boundary))
    return gap <= CONTEXT_PAD_S


def _line_span(clip_start: float, utt: Utterance, floor: float) -> tuple[float, float]:
    """Recording-time (start, end) of a transcribed utterance. The start is
    raised to `floor` (see _run_jobs); the end is raised with it, because an
    utterance whose words Whisper placed entirely before `floor` would
    otherwise end before it starts."""
    start = max(clip_start + utt.start, floor)
    return start, max(clip_start + utt.end, start)


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _usable(utt: Utterance, strict: bool) -> bool:
    """Drop empty output; on separated (artifact-prone) audio also drop what
    Whisper itself flags as probably-not-speech and low confidence."""
    if not utt.text.strip():
        return False
    # Whisper's non-speech tags ("[BLANK_AUDIO]", "(music)", partial "_AUDIO]")
    # are hallucinations on noise, never words; real speech has no brackets.
    if re.search(r"[\[\]()_]", utt.text):
        return False
    if strict and utt.no_speech_prob > 0.6 and utt.avg_logprob < -1.0:
        return False
    # A separated stream's low-level residue often makes the fine-tuned model emit a
    # confident lone "mm" / "mm-hmm". Real fillers in overlapped speech come with
    # words, so on separated audio an utterance made only of fillers is dropped.
    if strict:
        words = normalize(utt.text, drop_fillers=False)
        if words and all(w in FILLERS for w in words):
            return False
    return True


class TranscriptionPipeline:
    def __init__(
        self,
        hf_token: Optional[str] = None,
        asr_model_size: str = "small",
        gallery: Optional[EnrollmentGallery] = None,
        separate: bool = True,
        separation_model: str = DEFAULT_SEPARATION_MODEL,
        verbatim_prompt: bool = False,
        beam_size: int = 5,
        asr_backend: str = "auto",
        device: Optional[str] = None,
        log: Callable[[str], None] = _log,
    ):
        self.device = device or pick_device()
        self.log = log
        self.separate = separate
        self.separation_model = separation_model
        self.gallery = gallery
        self.log("loading diarizer + ASR models...")
        self.diarizer = Diarizer(hf_token or get_hf_token(), device=self.device)
        self.transcriber = make_transcriber(
            model_size=asr_model_size,
            backend=asr_backend,
            device="cuda" if self.device == "cuda" else "cpu",
            compute_type="float16" if self.device == "cuda" else "int8",
            verbatim_prompt=verbatim_prompt,
            beam_size=beam_size,
        )
        self._embedder = gallery.embedder if gallery else None
        self._separator: Optional[OverlapSeparator] = None
        self.last_turns: List[DiarizedSegment] = []   # raw diarizer output of the latest run (for DER)

    # -- lazily loaded heavy components ------------------------------------

    @property
    def embedder(self) -> SpeakerEmbedder:
        if self._embedder is None:
            self._embedder = SpeakerEmbedder(device=self.device)
        return self._embedder

    @property
    def separator(self) -> OverlapSeparator:
        if self._separator is None:
            self.log("loading separation model...")
            self._separator = OverlapSeparator(self.separation_model, device=self.device)
        return self._separator

    # -- main entry point -----------------------------------------------------

    def run(self, audio_path: str, num_speakers: Optional[int] = None) -> List[TranscriptLine]:
        t0 = time.time()
        audio = load_audio(audio_path)
        self.log(f"audio: {len(audio) / TARGET_SR:.1f}s")

        turns = self.diarizer.diarize(audio, TARGET_SR, num_speakers=num_speakers)
        self.last_turns = turns
        segments = build_segments(turns)
        n_overlap = sum(s.is_overlapping for s in segments)
        speakers = sorted({sp for s in segments for sp in s.speakers})
        self.log(
            f"diarization: {len(speakers)} speaker(s), {len(segments)} segments "
            f"({n_overlap} overlapping)  [{time.time() - t0:.0f}s]"
        )

        need_centroids = bool(self.gallery) or (self.separate and n_overlap > 0)
        centroids: Dict[str, np.ndarray] = {}
        if need_centroids:
            centroids = speaker_centroids(audio, TARGET_SR, segments, self.embedder)

        names: Dict[str, str] = {}
        if self.gallery:
            names = self.gallery.resolve(centroids)
            self.log(f"enrollment: matched {names or 'nobody'}")

        def display(label: str) -> str:
            return names.get(label, label)

        jobs: List[AsrJob] = []
        for i, seg in enumerate(segments):
            if seg.is_overlapping and self.separate:
                prev_seg = segments[i - 1] if i > 0 else None
                next_seg = segments[i + 1] if i + 1 < len(segments) else None
                jobs += self._overlap_jobs(audio, seg, prev_seg, next_seg, centroids, display)
            else:
                jobs += self._single_jobs(audio, seg, display)
        lines = self._run_jobs(jobs)

        # Global chronological order is what makes interruptions appear as
        # separate, correctly interleaved lines.
        lines.sort(key=lambda l: l.start)
        self.log(f"done: {len(lines)} lines in {time.time() - t0:.0f}s")
        return lines

    # -- per-segment routes ---------------------------------------------------

    def _single_jobs(
        self,
        audio: np.ndarray,
        seg: Segment,
        display: Callable[[str], str],
    ) -> List[AsrJob]:
        clip, lo = _padded_clip(audio, seg)
        # If this is an overlap we chose not to separate, attribute it to
        # everyone involved rather than silently picking one.
        label = "+".join(display(s) for s in seg.speakers)
        return [AsrJob(clip, lo, seg, label, strict=False)]

    def _overlap_jobs(
        self,
        audio: np.ndarray,
        seg: Segment,
        prev_seg: Optional[Segment],
        next_seg: Optional[Segment],
        centroids: Dict[str, np.ndarray],
        display: Callable[[str], str],
    ) -> List[AsrJob]:
        if len(seg.speakers) > 2:
            self.log(
                f"  warning: {len(seg.speakers)} speakers overlap at {seg.start:.1f}s; "
                "the 2-source separator can only recover two of them"
            )
        clip, lo = _padded_clip(audio, seg)
        streams = self.separator.separate(clip, TARGET_SR, lo, lo + len(clip) / TARGET_SR)

        loudest = max(s.rms for s in streams)
        streams = [s for s in streams if loudest > 0 and s.rms >= MIN_STREAM_ENERGY_RATIO * loudest]

        owners = self._assign_owners(streams, seg, centroids)

        jobs: List[AsrJob] = []
        for stream in streams:
            owner = owners[stream.source_index]
            # Crop a stream's edge only where a neighbouring segment also
            # transcribes this same speaker (that's what would duplicate
            # words). A speaker who *starts* inside the overlap has nothing
            # before it to duplicate, and cropping there would only lose
            # words -- Whisper stretches a first word back into leading
            # silence, so its timestamp can fall outside the window.
            jobs.append(AsrJob(
                stream.audio, lo, seg, display(owner), strict=True,
                crop_lo=_continues(prev_seg, owner, seg.start),
                crop_hi=_continues(next_seg, owner, seg.end),
            ))
        return jobs

    def _run_jobs(self, jobs: List[AsrJob]) -> List[TranscriptLine]:
        """Transcribe every job's clip (batched into as few Whisper calls as
        the backend allows) and return the lines for the words inside each
        job's segment."""
        if not jobs:
            return []
        many = getattr(self.transcriber, "transcribe_many", None)
        # Only clean single-speaker clips are packed into shared Whisper windows.
        # Separated streams (strict=True) carry separation artifacts, and packing
        # them measurably lost words in overlaps, so they are transcribed alone.
        all_utts: List[Optional[List[Utterance]]] = [None] * len(jobs)
        packable = [i for i, j in enumerate(jobs) if many and not j.strict]
        if packable:
            for i, utts in zip(packable, many([jobs[i].clip for i in packable])):
                all_utts[i] = utts
        for i, job in enumerate(jobs):
            if all_utts[i] is None:
                all_utts[i] = self.transcriber.transcribe(job.clip)

        lines: List[TranscriptLine] = []
        for job, utts in zip(jobs, all_utts):
            seg = job.seg
            utts = crop_utterances(
                utts,
                seg.start - job.clip_start if job.crop_lo else float("-inf"),
                seg.end - job.clip_start if job.crop_hi else float("inf"),
            )
            # A first word that Whisper stretched back into leading silence must
            # not be stamped earlier than the window the speaker actually entered.
            floor = -float("inf") if job.crop_lo else seg.start
            for u in utts:
                if _usable(u, job.strict):
                    start, end = _line_span(job.clip_start, u, floor)
                    lines.append(TranscriptLine(job.label, start, u.as_display_text(), end, seg.is_overlapping))
        return lines

    def _assign_owners(
        self,
        streams: List[SeparatedStream],
        seg: Segment,
        centroids: Dict[str, np.ndarray],
    ) -> Dict[int, str]:
        """source_index -> speaker label, by voice similarity to each
        speaker's clean-speech centroid (separator output order is arbitrary)."""
        embeddings = [self.embedder.embed(s.audio, s.sample_rate) for s in streams]
        by_position = match_streams_to_speakers(embeddings, list(seg.speakers), centroids)
        return {s.source_index: by_position[i] for i, s in enumerate(streams) if i in by_position}


def run_pipeline(
    audio_path: str,
    hf_token: Optional[str] = None,
    gallery: Optional[EnrollmentGallery] = None,
    asr_model_size: str = "small",
    separate: bool = True,
    num_speakers: Optional[int] = None,
) -> List[TranscriptLine]:
    """One-shot convenience wrapper around TranscriptionPipeline."""
    pipe = TranscriptionPipeline(
        hf_token=hf_token, asr_model_size=asr_model_size, gallery=gallery, separate=separate
    )
    return pipe.run(audio_path, num_speakers=num_speakers)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Multi-speaker verbatim transcription (offline).")
    ap.add_argument("audio")
    ap.add_argument("--model", default="small", help="Whisper size: tiny|base|small|medium|large-v3")
    ap.add_argument("--num-speakers", type=int, default=None, help="hint; improves diarization if known")
    ap.add_argument("--asr-backend", default="auto", choices=["auto", "mlx", "ctranslate2"],
                    help="mlx = Apple GPU (5x faster, greedy only); ctranslate2 = CPU/CUDA with beam search")
    ap.add_argument("--no-separation", action="store_true", help="skip Phase 2 (transcribe overlaps as-is)")
    ap.add_argument("--sep-model", default=DEFAULT_SEPARATION_MODEL,
                    help="SepFormer checkpoint (libri2mix | wsj02mix | whamr under speechbrain/sepformer-*)")
    ap.add_argument("--verbatim-prompt", action="store_true", help="prime Whisper to keep fillers")
    ap.add_argument("--enroll", action="append", default=[], metavar="NAME=WAV",
                    help="optional voice enrollment, repeatable: --enroll Sampath=sampath.wav")
    ap.add_argument("--out", default=None, help="also write the transcript to this file")
    args = ap.parse_args()

    gallery = None
    if args.enroll:
        gallery = EnrollmentGallery()
        for item in args.enroll:
            name, _, path = item.partition("=")
            gallery.enroll(name, path)

    pipe = TranscriptionPipeline(
        asr_model_size=args.model,
        gallery=gallery,
        separate=not args.no_separation,
        asr_backend=args.asr_backend,
        separation_model=args.sep_model,
        verbatim_prompt=args.verbatim_prompt,
    )
    result = pipe.run(args.audio, num_speakers=args.num_speakers)

    rendered = "\n".join(line.render() for line in result)
    print(rendered)
    if args.out:
        with open(args.out, "w") as f:
            f.write(rendered + "\n")
