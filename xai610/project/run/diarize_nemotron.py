"""Step 1: speaker diarization with NVIDIA Nemotron 3 Diarization (no training).

  pip install 'nemo-toolkit[asr]'
  python diarize_nemotron.py --wav_dir data/wav --meetings data/test.meetings.txt \
      --out_dir exp/diar/nemotron
"""
import argparse
import os

from nemo.collections.asr.models import SortformerEncLabelModel

import ami_util


def parse_segment(seg):
    """NeMo returns 'begin end speaker_k' strings (or tuples)."""
    if isinstance(seg, str):
        seg = seg.split()
    s, e, spk = seg[:3]
    return float(s), float(e), str(spk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav_dir", required=True)
    ap.add_argument("--meetings", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    model = SortformerEncLabelModel.from_pretrained("nvidia/Nemotron-3-Diarization")
    model.eval()
    # "Very high latency (offline)" configuration from the model card (80 ms frames).
    m = model.sortformer_modules
    m.spkcache_len, m.fifo_len, m.chunk_len = 264, 40, 340
    m.chunk_right_context, m.spkcache_update_period = 40, 300
    model._check_streaming_parameters()

    meetings = open(args.meetings).read().split()
    wavs = [os.path.join(args.wav_dir, f"{x}.Mix-Headset.wav") for x in meetings]
    outputs = model.diarize(audio=wavs, batch_size=1)
    for meeting, segs in zip(meetings, outputs):
        segs = sorted(parse_segment(s) for s in segs)
        ami_util.write_rttm(meeting, segs, f"{args.out_dir}/{meeting}.rttm")
        print(meeting, len(segs), "segments,", len({s[2] for s in segs}), "speakers")


if __name__ == "__main__":
    main()
