"""Step 3: meeting-level speaker-attributed transcription.

System A (cascade): diarization segments -> Stage-1 model (plain ASR).
System B (joint)  : 30-s windows -> Stage-2 model (<spkN> tags);
                    the k-th local tag is mapped to the k-th speaker that
                    appears in the window according to the diarization.

  python transcribe_meeting.py --system A --ckpt exp/stage1 \
      --rttm_dir exp/diar/nemotron --out exp/hyp/A_nemotron.seglst.json
  python transcribe_meeting.py --system B --ckpt exp/stage2 \
      --rttm_dir exp/diar/nemotron --out exp/hyp/B_nemotron.seglst.json
  # oracle diarization: --rttm_dir data/ref_rttm

  meeteval-wer cpwer -r data/manifests/test/ref.seglst.json -h exp/hyp/B_nemotron.seglst.json
  meeteval-wer orcwer -r data/manifests/test/ref.seglst.json -h exp/hyp/B_nemotron.seglst.json
"""
import argparse
import os
import re

import soundfile as sf

import ami_util
from speech_llm import SpeechLLM

SR = 16000


def merge_segments(segs, max_gap=0.5, max_len=29.0, min_len=0.3):
    """Merges close segments of the same speaker; splits long ones (<= 29 s)."""
    out = []
    for s, e, spk in sorted(segs):
        if out and out[-1][2] == spk and s - out[-1][1] <= max_gap and e - out[-1][0] <= max_len:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e, spk])
    pieces = []
    for s, e, spk in out:
        while e - s > max_len:
            pieces.append((s, s + max_len, spk))
            s += max_len
        if e - s >= min_len:
            pieces.append((s, e, spk))
    return pieces


def parse_tagged(text):
    """'<spk1> a b <spk2> c' -> [(1, 'a b'), (2, 'c')]"""
    out = []
    for k, words in re.findall(r"<spk(\d+)>([^<]*)", text):
        words = ami_util.normalize(words)
        if words:
            out.append((int(k), words))
    if not out and ami_util.normalize(text):
        out.append((1, ami_util.normalize(text)))
    return out


def batched(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i: i + n]


def system_a(model, meeting, audio, segs, bs):
    turns = []
    for batch in batched(merge_segments(segs), bs):
        wavs = [audio[int(s * SR): int(e * SR)] for s, e, _ in batch]
        for (s, e, spk), hyp in zip(batch, model.generate(wavs, task="asr")):
            hyp = ami_util.normalize(hyp)
            if hyp:
                turns.append(dict(speaker=spk, start=s, end=e, text=hyp))
    return turns


def system_b(model, meeting, audio, segs, bs, win=30.0):
    dur = len(audio) / SR
    windows = [(t, min(t + win, dur)) for t in range(0, int(dur) + 1, int(win)) if t < dur - 0.5]
    # fallback for tags without a diarization speaker: the most active speaker
    talk = {}
    for s, e, spk in segs:
        talk[spk] = talk.get(spk, 0.0) + e - s
    fallback = max(talk, key=talk.get) if talk else "unknown"
    turns = []
    for batch in batched(windows, bs):
        wavs = [audio[int(s * SR): int(e * SR)] for s, e in batch]
        for (ws, we), hyp in zip(batch, model.generate(wavs, task="spk", max_new_tokens=320)):
            # speakers ordered by first appearance inside this window
            inside = sorted((max(s, ws), spk) for s, e, spk in segs if s < we and e > ws)
            order = list(dict.fromkeys(spk for _, spk in inside))
            for j, (k, words) in enumerate(parse_tagged(hyp)):
                spk = order[min(k, len(order)) - 1] if order else fallback
                turns.append(dict(speaker=spk, start=ws + j * 1e-3, end=we, text=words))
    return turns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=["A", "B"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--rttm_dir", required=True)
    ap.add_argument("--wav_dir", default="data/wav")
    ap.add_argument("--meetings", default="data/test.meetings.txt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--llm", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--encoder", default="openai/whisper-small")
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    model = SpeechLLM(args.encoder, args.llm)
    model.load(args.ckpt)
    model.eval()

    seglst = []
    for m in open(args.meetings).read().split():
        audio, sr = sf.read(os.path.join(args.wav_dir, f"{m}.Mix-Headset.wav"), dtype="float32")
        assert sr == SR
        segs = ami_util.read_rttm(f"{args.rttm_dir}/{m}.rttm")
        run = system_a if args.system == "A" else system_b
        turns = run(model, m, audio, segs, args.batch_size)
        seglst += ami_util.to_seglst(m, turns)
        print(f"{m}: {len(turns)} turns", flush=True)
    ami_util.write_json(seglst, args.out)


if __name__ == "__main__":
    main()
