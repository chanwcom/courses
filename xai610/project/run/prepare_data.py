"""Builds the training / test manifests from AMI.

Example:
  python prepare_data.py --ann_dir data/ami_annotations --wav_dir data/wav \
      --out_dir data/manifests --num_train_meetings 60

Outputs (JSON lines, one example per line):
  stage1_{train,dev}.jsonl : {"wav", "start", "end", "text"}  single speaker
  stage2_{train,dev}.jsonl : {"wav", "start", "end", "text"}  "<spk1> ... <spk2> ..."
  summary_train.jsonl      : {"meeting", "transcript", "summary"}
  test/ref.seglst.json     : reference for cpWER (MeetEval)
  test/summary_ref.jsonl   : reference transcript and summary of test meetings
"""
import argparse
import json
import os
import random

import soundfile as sf

import ami_util


def wav_path(wav_dir, m):
    return os.path.join(wav_dir, f"{m}.Mix-Headset.wav")


def dump(rows, path):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{path}: {len(rows)} examples, "
          f"{sum(r['end'] - r['start'] for r in rows) / 3600:.1f} h"
          if rows and "end" in rows[0] else f"{path}: {len(rows)} examples")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann_dir", required=True)
    ap.add_argument("--wav_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_train_meetings", type=int, default=60)
    args = ap.parse_args()
    os.makedirs(f"{args.out_dir}/test", exist_ok=True)

    splits = {s: ami_util.read_meeting_list(s) for s in ("train", "dev", "test")}
    random.seed(0)
    train = sorted(random.sample(splits["train"], args.num_train_meetings))
    with open(f"{args.out_dir}/train_meetings.txt", "w") as f:
        f.write("\n".join(train) + "\n")

    # Speech LLM data (only meetings whose audio was downloaded).
    for name, meetings in (("train", train), ("dev", splits["dev"])):
        s1, s2 = [], []
        for m in meetings:
            wav = wav_path(args.wav_dir, m)
            if not os.path.exists(wav):
                continue
            words = ami_util.read_words(args.ann_dir, m)
            dur = sf.info(wav).duration
            for t in ami_util.single_speaker_segments(words):
                s1.append(dict(wav=wav, start=t["start"], end=t["end"], text=t["text"]))
            for w in ami_util.make_windows(words, dur):
                s2.append(dict(wav=wav, start=w["start"], end=w["end"], text=w["text"]))
        dump(s1, f"{args.out_dir}/stage1_{name}.jsonl")
        dump(s2, f"{args.out_dir}/stage2_{name}.jsonl")

    # Summarization data: all train meetings with a summary (text only, no audio).
    rows = []
    for m in splits["train"]:
        summ = ami_util.read_summary(args.ann_dir, m)
        if summ:
            turns = ami_util.words_to_turns(ami_util.read_words(args.ann_dir, m))
            rows.append(dict(meeting=m, transcript=ami_util.turns_to_transcript(turns),
                             summary=summ))
    dump(rows, f"{args.out_dir}/summary_train.jsonl")

    # Test references.
    seglst, summ_rows = [], []
    for m in splits["test"]:
        turns = ami_util.words_to_turns(ami_util.read_words(args.ann_dir, m))
        seglst += ami_util.to_seglst(m, turns)
        summ = ami_util.read_summary(args.ann_dir, m)
        if summ:
            summ_rows.append(dict(meeting=m, transcript=ami_util.turns_to_transcript(turns),
                                  summary=summ))
    ami_util.write_json(seglst, f"{args.out_dir}/test/ref.seglst.json")
    dump(summ_rows, f"{args.out_dir}/test/summary_ref.jsonl")


if __name__ == "__main__":
    main()
