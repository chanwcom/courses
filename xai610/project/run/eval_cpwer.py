"""Step 3 scoring with MeetEval.

  pip install meeteval
  python eval_cpwer.py --ref data/manifests/test/ref.seglst.json --hyp exp/hyp/B_nemotron.seglst.json

cpWER : concatenated minimum-permutation WER (speaker attribution matters).
WER   : speaker-agnostic; every segment is relabelled as one speaker and the
        words are concatenated in time order. cpWER - WER ~ attribution errors.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile


def run_cpwer(ref, hyp):
    out = subprocess.run([sys.executable, "-m", "meeteval.wer", "cpwer", "-r", ref, "-h", hyp],
                         capture_output=True, text=True)
    m = re.search(r"%cpWER: ([\d.]+)%", out.stdout + out.stderr)
    if not m:
        raise RuntimeError(out.stdout + out.stderr)
    return float(m.group(1))


def single_speaker(path, tmp_dir):
    segs = json.load(open(path))
    for s in segs:
        s["speaker"] = "all"
    out = os.path.join(tmp_dir, os.path.basename(path))
    json.dump(segs, open(out, "w"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--hyp", required=True)
    args = ap.parse_args()
    cp = run_cpwer(args.ref, args.hyp)
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(f"{d}/r"); os.makedirs(f"{d}/h")
        wer = run_cpwer(single_speaker(args.ref, f"{d}/r"), single_speaker(args.hyp, f"{d}/h"))
    print(f"{os.path.basename(args.hyp)}: cpWER {cp:.2f}%  WER {wer:.2f}%  "
          f"(attribution {cp - wer:.2f}%p)")


if __name__ == "__main__":
    main()
