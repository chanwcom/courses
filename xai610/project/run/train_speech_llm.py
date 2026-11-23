"""Trains the Speech LLM.

Stage 1 (alignment): projector only, single-speaker ASR.
  python train_speech_llm.py --stage 1 --train data/manifests/stage1_train.jsonl \
      --dev data/manifests/stage1_dev.jsonl --out_dir exp/stage1

Stage 2 (speaker-tagged ASR): projector + LoRA, 30-s multi-speaker windows.
  python train_speech_llm.py --stage 2 --init exp/stage1 \
      --train data/manifests/stage2_train.jsonl \
      --dev data/manifests/stage2_dev.jsonl --out_dir exp/stage2
"""
import argparse
import json
import math
import random
import time

import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup

from speech_llm import SpeechLLM


class SegmentDataset(Dataset):
    """Rows: {"wav", "start", "end", "text"}; reads only the needed samples."""

    def __init__(self, path, max_rows=None):
        self.rows = [json.loads(l) for l in open(path)]
        if max_rows:
            random.Random(0).shuffle(self.rows)
            self.rows = self.rows[:max_rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        sr = sf.info(r["wav"]).samplerate
        wav, _ = sf.read(r["wav"], start=int(r["start"] * sr), stop=int(r["end"] * sr),
                         dtype="float32")
        return wav[: 30 * sr], r["text"]


def collate(batch):
    return [b[0] for b in batch], [b[1] for b in batch]


@torch.no_grad()
def evaluate(model, loader, task):
    model.eval()
    tot, n = 0.0, 0
    for wavs, texts in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tot += model(wavs, texts, task).item() * len(wavs)
        n += len(wavs)
    model.train()
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=[1, 2], required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--init", default=None, help="Stage-1 checkpoint for Stage 2")
    ap.add_argument("--llm", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--encoder", default="openai/whisper-small")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--max_dev", type=int, default=300)
    args = ap.parse_args()

    stage1 = args.stage == 1
    epochs = args.epochs or (3 if stage1 else 4)
    lr = args.lr or (1e-3 if stage1 else 2e-4)
    task = "asr" if stage1 else "spk"
    torch.manual_seed(0)

    model = SpeechLLM(args.encoder, args.llm, use_lora=False)
    if args.init:                                   # Stage-1 projector
        model.load(args.init)
    if not stage1:
        model.add_lora(args.lora_r)
    model.enable_gradient_checkpointing()
    model.train()
    params = model.trainable_parameters()
    print(f"trainable parameters: {sum(p.numel() for p in params) / 1e6:.1f} M")

    train = DataLoader(SegmentDataset(args.train), batch_size=args.batch_size,
                       shuffle=True, num_workers=4, collate_fn=collate, drop_last=True)
    dev = DataLoader(SegmentDataset(args.dev, args.max_dev), batch_size=args.batch_size,
                     num_workers=4, collate_fn=collate)
    steps = epochs * len(train) // args.grad_accum
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(opt, int(0.05 * steps), steps)

    step, best, t0 = 0, math.inf, time.time()
    for ep in range(epochs):
        for i, (wavs, texts) in enumerate(train):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(wavs, texts, task) / args.grad_accum
            loss.backward()
            if (i + 1) % args.grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 50 == 0:
                print(f"ep {ep} step {step}/{steps} loss {loss.item() * args.grad_accum:.3f} "
                      f"lr {sched.get_last_lr()[0]:.2e} {(time.time() - t0) / 60:.1f} min",
                      flush=True)
        dev_loss = evaluate(model, dev, task)
        print(f"== epoch {ep}: dev loss {dev_loss:.3f}", flush=True)
        if dev_loss < best:
            best = dev_loss
            model.save(args.out_dir)
            print(f"saved to {args.out_dir}")
        wav, ref = dev.dataset[0]
        print("ref:", ref[:200], "\nhyp:", model.generate([wav], task)[0][:200], flush=True)


if __name__ == "__main__":
    main()
