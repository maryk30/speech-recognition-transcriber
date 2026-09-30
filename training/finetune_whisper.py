"""
Fine-tune Whisper on AMI windows (see prepare_ami.py). Goal: verbatim output
(keep um / uh / mm-hmm) and robustness to far-field, overlapped meeting
speech -- both things stock Whisper is weak at.

    python training/finetune_whisper.py --steps 1400 --out models/whisper-small-ami

Why this is not a plain full fine-tune: on an Apple M3 (MPS) the Whisper-small
encoder costs 2.8 s forward+backward for 4 windows versus 0.46 s for the
decoder, so full fine-tuning would take ~4.5 h per 1200 steps. Instead:

  1. the frozen lower encoder layers run ONCE over the dataset and their output
     (hidden states entering layer L) is cached to disk as float16;
  2. training then updates only the top `--train-layers` encoder layers, the
     encoder's final layer norm, and the whole decoder, starting from the cache.

The decoder learns the verbatim style / meeting vocabulary; the top encoder
layers adapt to far-field acoustics. Lower encoder layers keep the pretrained
weights. Saved as a normal HuggingFace checkpoint.

Labels use Whisper's native timestamp format so segment/word timing keeps working:

    <|startoftranscript|><|en|><|transcribe|>
    <|0.00|> so um i was thinking <|3.20|><|3.20|> yeah <|3.80|> ... <|endoftext|>
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import DATA_DIR, MODELS_DIR, pick_device  # noqa: E402
from scoring import wer  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_utils import (BestTracker, augment_hidden, load_trainer_state,  # noqa: E402
                         save_trainer_state)

SR = 16000
ENC_FRAMES, ENC_DIM = 1500, 768


def build_labels(tokenizer, segments) -> List[int]:
    """[en, transcribe, <|t0|> text <|t1|><|t2|> text ... <|tn|>, eot]  (no leading sot:
    the model prepends its decoder-start token when shifting labels right)."""
    ts = lambda t: tokenizer.convert_tokens_to_ids(f"<|{min(t, 30.0):.2f}|>")  # noqa: E731
    ids = [tokenizer.convert_tokens_to_ids("<|en|>"), tokenizer.convert_tokens_to_ids("<|transcribe|>")]
    for start, end, text in segments:
        ids.append(ts(start))
        ids.extend(tokenizer.encode(" " + text, add_special_tokens=False))
        ids.append(ts(end))
    ids.append(tokenizer.eos_token_id)
    return ids


def load_index(split: str):
    return json.loads((DATA_DIR / "ami" / f"windows_{split}.json").read_text())


# -- encoder cache -----------------------------------------------------------

class _Stop(Exception):
    pass


def cache_path(split: str, layer: int) -> Path:
    return DATA_DIR / "ami" / f"enc_{split}_L{layer}.f16"


def build_cache(model, fe, split: str, layer: int, device: str, batch: int = 8) -> None:
    """Run the frozen encoder up to layer `layer` once and store the hidden
    states entering that layer."""
    index = load_index(split)
    path = cache_path(split, layer)
    expected = len(index) * ENC_FRAMES * ENC_DIM * 2
    if path.exists() and path.stat().st_size == expected:
        print(f"cache ok: {path.name}", flush=True)
        return

    audio = np.memmap(DATA_DIR / "ami" / f"windows_{split}.i16", dtype=np.int16, mode="r")
    out = np.lib.format.open_memmap if False else np.memmap(path, dtype=np.float16, mode="w+",
                                                              shape=(len(index), ENC_FRAMES, ENC_DIM))
    captured = {}

    def grab(_module, _args, kwargs):
        h = _args[0] if _args else kwargs["hidden_states"]
        captured["h"] = h.detach()
        raise _Stop

    hook = model.model.encoder.layers[layer].register_forward_pre_hook(grab, with_kwargs=True)
    model.eval()
    t0 = time.time()
    try:
        for i in range(0, len(index), batch):
            rows = index[i : i + batch]
            feats = []
            for w in rows:
                clip = np.array(audio[w["offset"] : w["offset"] + w["length"]], dtype=np.float32) / 32768.0
                feats.append(fe(clip, sampling_rate=SR, return_tensors="np").input_features[0])
            x = torch.from_numpy(np.stack(feats)).to(device)
            with torch.no_grad():
                try:
                    model.model.encoder(x)
                except _Stop:
                    pass
            out[i : i + len(rows)] = captured["h"].cpu().numpy().astype(np.float16)
            if (i // batch) % 20 == 0:
                done = i + len(rows)
                print(f"  caching {split}: {done}/{len(index)} | eta {(len(index) - done) * (time.time() - t0) / done / 60:.0f} min", flush=True)
    finally:
        hook.remove()
    out.flush()
    print(f"cached {split}: {len(index)} windows -> {path.name} ({expected / 1e9:.1f} GB) in {(time.time() - t0) / 60:.1f} min", flush=True)


class CachedDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, layer: int, tokenizer, augment: bool = False):
        self.index = load_index(split)
        self.path, self.tok, self.augment = cache_path(split, layer), tokenizer, augment
        self._mm = None

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        if self._mm is None:
            self._mm = np.memmap(self.path, dtype=np.float16, mode="r", shape=(len(self.index), ENC_FRAMES, ENC_DIM))
        h = torch.from_numpy(np.array(self._mm[i]))
        if self.augment:                      # training set only; validation stays clean
            h = augment_hidden(h)
        return h, build_labels(self.tok, self.index[i]["segments"])

    def reference_text(self, i: int) -> str:
        return " ".join(text for _, _, text in self.index[i]["segments"])


def collate(batch):
    feats = torch.stack([b[0] for b in batch]).float()
    longest = max(len(b[1]) for b in batch)
    labels = torch.full((len(batch), longest), -100, dtype=torch.long)
    for i, (_, ids) in enumerate(batch):
        labels[i, : len(ids)] = torch.tensor(ids)
    return feats, labels


# -- forward from the cache --------------------------------------------------

def encode_from_cache(model, layer: int, cached):
    """Run the trainable top encoder layers over cached hidden states."""
    from transformers.modeling_outputs import BaseModelOutput

    enc = model.model.encoder
    h = cached
    for block in enc.layers[layer:]:
        out = block(h, None)
        h = out[0] if isinstance(out, tuple) else out
    return BaseModelOutput(last_hidden_state=enc.layer_norm(h))


def loss_from_cache(model, layer: int, cached, labels):
    return model(encoder_outputs=encode_from_cache(model, layer, cached), labels=labels).loss


def validation_wer(model, layer, dataset, tokenizer, device, n_windows: int, drop_fillers: bool = False) -> float:
    """Greedy-decode `n_windows` evenly spread validation windows and return
    corpus WER against their references. Fillers are scored by default:
    keeping them is the point of this fine-tune, and loss alone doesn't say
    whether the decoder still transcribes well."""
    model.eval()
    n_windows = min(n_windows, len(dataset))
    picks = np.linspace(0, len(dataset) - 1, n_windows).astype(int)
    errors = words = 0
    from scoring import edit_distance, normalize

    with torch.no_grad():
        for i in picks:
            feats, _ = dataset[int(i)]
            enc_out = encode_from_cache(model, layer, feats[None].float().to(device))
            # timestamps on, like the training labels; decode length capped by the model's positions
            ids = model.generate(encoder_outputs=enc_out, max_new_tokens=min(440, model.config.max_target_positions - 8),
                                 do_sample=False, num_beams=1, language="en", task="transcribe",
                                 return_timestamps=True)
            hyp = tokenizer.decode(ids[0], skip_special_tokens=True)
            ref_w, hyp_w = normalize(dataset.reference_text(int(i)), drop_fillers), normalize(hyp, drop_fillers)
            errors += edit_distance(ref_w, hyp_w); words += len(ref_w)
    model.train()
    return errors / max(words, 1)


def evaluate_loss(model, layer, loader, device, max_batches: int) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for b, (feats, labels) in enumerate(loader):
            if b >= max_batches:
                break
            total += loss_from_cache(model, layer, feats.to(device), labels.to(device)).item(); n += 1
    model.train()
    return total / max(n, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="openai/whisper-small")
    ap.add_argument("--out", default=str(MODELS_DIR / "whisper-small-ami"))
    ap.add_argument("--steps", type=int, default=1400, help="optimizer steps")
    ap.add_argument("--batch-size", type=int, default=2, help="windows per micro-batch (MPS driver memory grows with it: 5.0/6.2/8.4 GB at 1/2/4; 4+ swapped on a 17 GB Mac)")
    ap.add_argument("--grad-accum", type=int, default=4, help="micro-batches per optimizer step")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--train-layers", type=int, default=2, help="top encoder layers to fine-tune (cached below them)")
    ap.add_argument("--val-every", type=int, default=200)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--val-windows", type=int, default=30, help="validation windows greedy-decoded for WER at each --val-every")
    ap.add_argument("--resume", action="store_true", help="continue from --out (model + optimizer + schedule + step)")
    ap.add_argument("--cache-only", action="store_true")
    args = ap.parse_args()

    from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration, WhisperTokenizer

    device = pick_device()
    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    fe = WhisperFeatureExtractor.from_pretrained(args.base)
    tok = WhisperTokenizer.from_pretrained(args.base)
    tok.set_prefix_tokens(language="en", task="transcribe", predict_timestamps=True)
    out_dir = Path(args.out)
    last_dir = out_dir.with_name(out_dir.name + "-last")      # newest state; `out_dir` holds the best-WER weights
    resume_state = load_trainer_state(last_dir) if args.resume else None
    if args.resume and resume_state is None:
        print(f"--resume: no {last_dir}/trainer_state.pt, starting fresh", flush=True)
    model = WhisperForConditionalGeneration.from_pretrained(str(last_dir) if resume_state else args.base).to(device)
    model.config.use_cache = False
    n_layers = len(model.model.encoder.layers)
    layer = n_layers - args.train_layers

    splits = ["train"] + (["validation"] if (DATA_DIR / "ami" / "windows_validation.json").exists() else [])
    for split in splits:
        build_cache(model, fe, split, layer, device)
    if args.cache_only:
        return

    # Freeze everything, then unfreeze the top encoder layers + final norm + decoder.
    for p in model.parameters():
        p.requires_grad = False
    trainable = list(model.model.encoder.layers[layer:].parameters()) + list(model.model.encoder.layer_norm.parameters()) \
        + list(model.model.decoder.parameters())
    for p in trainable:
        p.requires_grad = True
    model.train()
    print(f"device {device} | trainable {sum(p.numel() for p in trainable) / 1e6:.0f}M of "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.0f}M | top {args.train_layers} encoder layers + decoder", flush=True)

    make = lambda ds, shuffle: torch.utils.data.DataLoader(  # noqa: E731
        ds, batch_size=args.batch_size, shuffle=shuffle, collate_fn=collate, drop_last=shuffle)
    train_dl = make(CachedDataset("train", layer, tok, augment=True), True)
    val_ds = CachedDataset("validation", layer, tok) if "validation" in splits else None
    val_dl = make(val_ds, False) if val_ds else None
    eff = args.batch_size * args.grad_accum
    print(f"train windows {len(train_dl.dataset)} | micro-batch {args.batch_size} x accum {args.grad_accum} = {eff} | "
          f"~{len(train_dl.dataset) // eff} steps/epoch, {args.steps * eff / len(train_dl.dataset):.1f} epochs planned", flush=True)

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / args.warmup, max(0.0, (args.steps - s) / max(args.steps - args.warmup, 1))))

    def batches():
        while True:
            yield from train_dl

    best, start = BestTracker(), 0
    if resume_state:
        opt.load_state_dict(resume_state["optimizer"]); sched.load_state_dict(resume_state["scheduler"])
        start, best.best = resume_state["step"], resume_state["best_wer"]
        print(f"resumed at step {start} (best val WER so far {best.best:.3f})", flush=True)

    it, running, t0 = batches(), [], time.time()
    if val_dl and not resume_state:
        print(f"step 0 | val loss {evaluate_loss(model, layer, val_dl, device, 15):.3f} (before training)", flush=True)

    def save(dir_: Path, step: int) -> None:
        model.config.use_cache = True
        model.save_pretrained(dir_); tok.save_pretrained(dir_); fe.save_pretrained(dir_)
        model.config.use_cache = False
        print(f"saved {dir_} @ step {step}", flush=True)

    for step in range(start + 1, args.steps + 1):
        for _ in range(args.grad_accum):
            feats, labels = next(it)
            loss = loss_from_cache(model, layer, feats.to(device), labels.to(device))
            (loss / args.grad_accum).backward()
            running.append(loss.item())
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)

        if device == "mps" and step % 25 == 0:
            torch.mps.empty_cache()          # return cached blocks so the driver footprint stays flat
        if step % args.log_every == 0:
            el, done = time.time() - t0, step - start
            print(f"step {step}/{args.steps} | loss {np.mean(running[-args.log_every * args.grad_accum:]):.3f} | "
                  f"lr {sched.get_last_lr()[0]:.2e} | {el / done:.1f}s/step | eta {(args.steps - step) * el / done / 60:.0f} min", flush=True)
        improved = False
        if val_dl and step % args.val_every == 0:
            v_wer = validation_wer(model, layer, val_ds, tok, device, args.val_windows)
            improved = best.update(v_wer)
            print(f"step {step} | val loss {evaluate_loss(model, layer, val_dl, device, 15):.3f} | "
                  f"val WER {v_wer:.3f} (best {best.best:.3f}){' *' if improved else ''}", flush=True)
        if step % args.save_every == 0 or step == args.steps:
            save(last_dir, step)
            save_trainer_state(last_dir, step, opt, sched, best.best)
        if improved:
            save(out_dir, step)               # `out_dir` only ever holds the best-WER weights
        elif not val_dl and (step % args.save_every == 0 or step == args.steps):
            save(out_dir, step)               # no validation split: keep the old "latest" behaviour


if __name__ == "__main__":
    main()
