"""DER with fixed conditions: collar 0, overlap included, BUT only_words reference.

  pip install pyannote.metrics
  python eval_der.py --ref_dir data/ref_rttm --uem_dir data/ref_uem \
      --hyp_dir exp/diar/nemotron --meetings data/test.meetings.txt
"""
import argparse

from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate

import ami_util


def to_annotation(segs):
    ann = Annotation()
    for i, (s, e, spk) in enumerate(segs):
        ann[Segment(s, e), i] = spk
    return ann


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_dir", required=True)
    ap.add_argument("--uem_dir", required=True)
    ap.add_argument("--hyp_dir", required=True)
    ap.add_argument("--meetings", required=True)
    args = ap.parse_args()

    metric = DiarizationErrorRate(collar=0.0, skip_overlap=False)
    for m in open(args.meetings).read().split():
        ref = to_annotation(ami_util.read_rttm(f"{args.ref_dir}/{m}.rttm"))
        hyp = to_annotation(ami_util.read_rttm(f"{args.hyp_dir}/{m}.rttm"))
        _, _, s, e = open(f"{args.uem_dir}/{m}.uem").read().split()[:4]
        der = metric(ref, hyp, uem=Timeline([Segment(float(s), float(e))]))
        print(f"{m}: DER {100 * der:.2f}%")

    total = abs(metric)
    comp = metric[:]
    t = comp["total"]
    print(f"\nTOTAL DER {100 * total:.2f}%  "
          f"(miss {100 * comp['missed detection'] / t:.2f}%, "
          f"FA {100 * comp['false alarm'] / t:.2f}%, "
          f"confusion {100 * comp['confusion'] / t:.2f}%)")


if __name__ == "__main__":
    main()
