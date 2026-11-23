"""Utilities for the AMI Meeting Corpus (manual annotations v1.6.2).

Provides
  * word-level reading with speaker labels,
  * single-speaker segments (Stage 1) and 30-s speaker-tagged windows (Stage 2),
  * reference transcripts and abstractive summaries,
  * SegLST (MeetEval) / RTTM helpers.
"""
import glob
import json
import os
import re
import urllib.request
import xml.etree.ElementTree as ET

_SETUP_URL = ("https://raw.githubusercontent.com/BUTSpeechFIT/"
              "AMI-diarization-setup/main")
_NITE_ID = "{http://nite.sourceforge.net/}id"


def normalize(text):
    """Lower-case, drop punctuation except apostrophes, squeeze spaces."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def read_meeting_list(split):
    """Meeting IDs of the Full-corpus-ASR split (train / dev / test)."""
    url = f"{_SETUP_URL}/lists/{split}.meetings.txt"
    with urllib.request.urlopen(url) as f:
        return [m.strip() for m in f.read().decode().split() if m.strip()]


def read_words(ann_dir, meeting):
    """Returns a time-sorted list of (start, end, speaker, word).

    Speaker is the AMI channel letter (A, B, C, D, ...). Punctuation tokens
    and words without time stamps are skipped.
    """
    words = []
    for path in sorted(glob.glob(f"{ann_dir}/words/{meeting}.*.words.xml")):
        spk = os.path.basename(path).split(".")[1]
        for el in ET.parse(path).getroot():
            if el.tag != "w" or el.get("punc") == "true":
                continue
            if el.get("starttime") is None or el.get("endtime") is None:
                continue
            w = normalize(el.text or "")
            if w:
                words.append((float(el.get("starttime")),
                              float(el.get("endtime")), spk, w))
    words.sort(key=lambda x: x[0])
    return words


def words_to_turns(words, max_gap=0.5):
    """Merges consecutive words of the same speaker into turns.

    Returns a list of dict(speaker, start, end, text).
    """
    turns = []
    for s, e, spk, w in words:
        if turns and turns[-1]["speaker"] == spk and s - turns[-1]["end"] <= max_gap:
            turns[-1]["end"] = max(turns[-1]["end"], e)
            turns[-1]["text"] += " " + w
        else:
            turns.append(dict(speaker=spk, start=s, end=e, text=w))
    return turns


def single_speaker_segments(words, min_dur=1.0, max_dur=20.0, max_overlap=0.1):
    """Stage-1 data: turns where other speakers talk < 10 % of the time."""
    out = []
    for t in words_to_turns(words, max_gap=1.0):
        dur = t["end"] - t["start"]
        if not min_dur <= dur <= max_dur:
            continue
        other = sum(min(e, t["end"]) - max(s, t["start"])
                    for s, e, spk, _ in words
                    if spk != t["speaker"] and s < t["end"] and e > t["start"])
        if other < max_overlap * dur:
            out.append(t)
    return out


def serialize_with_tags(words_in_window):
    """'<spk1> hi there <spk2> yes ...' with speakers numbered by arrival order.

    Returns (text, local_to_global) where local_to_global[k-1] is the AMI
    speaker letter of <spkk>.
    """
    order, parts, prev = [], [], None
    for _, _, spk, w in words_in_window:
        if spk not in order:
            order.append(spk)
        if spk != prev:
            parts.append(f"<spk{order.index(spk) + 1}>")
            prev = spk
        parts.append(w)
    return " ".join(parts), order


def make_windows(words, total_dur, win=30.0, hop=30.0):
    """Stage-2 data: fixed windows; a word belongs to the window of its midpoint."""
    out, t = [], 0.0
    while t < total_dur:
        end = min(t + win, total_dur)
        ws = [x for x in words if t <= (x[0] + x[1]) / 2 < end]
        if ws and end - t >= 5.0:
            text, order = serialize_with_tags(ws)
            out.append(dict(start=round(t, 2), end=round(end, 2), text=text,
                            speakers=order))
        t += hop
    return out


def read_summary(ann_dir, meeting):
    """The <abstract> part of the AMI abstractive summary (None if missing)."""
    path = f"{ann_dir}/abstractive/{meeting}.abssumm.xml"
    if not os.path.exists(path):
        return None
    root = ET.parse(path).getroot()
    abstract = root.find("abstract")
    sents = [s.text.strip() for s in abstract.iter("sentence") if s.text]
    return " ".join(sents)


def turns_to_transcript(turns):
    """Text the summarizer reads: one 'Speaker k: ...' line per turn.

    Speaker numbers follow the order of first appearance.
    """
    order, lines = [], []
    for t in sorted(turns, key=lambda x: x["start"]):
        if t["speaker"] not in order:
            order.append(t["speaker"])
        lines.append(f"Speaker {order.index(t['speaker']) + 1}: {t['text']}")
    return "\n".join(lines)


# ---------------- SegLST (MeetEval) and RTTM ----------------
def to_seglst(meeting, turns):
    return [dict(session_id=meeting, speaker=str(t["speaker"]),
                 start_time=float(t["start"]), end_time=float(t["end"]),
                 words=t["text"]) for t in turns]


def write_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def read_rttm(path):
    """Returns a list of (start, end, speaker)."""
    segs = []
    with open(path) as f:
        for line in f:
            p = line.split()
            if p and p[0] == "SPEAKER":
                s, d = float(p[3]), float(p[4])
                segs.append((s, s + d, p[7]))
    return sorted(segs)


def write_rttm(meeting, segs, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        for s, e, spk in segs:
            f.write(f"SPEAKER {meeting} 1 {s:.3f} {e - s:.3f} <NA> <NA> {spk} <NA> <NA>\n")
