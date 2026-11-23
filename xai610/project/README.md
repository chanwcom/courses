# XAI 610 Project Part 3: Speaker-Attributed Meeting Transcription and Summarization

Baseline code for the AMI Meeting Corpus on a single 24 GB GPU.

The Speech LLM is a **text LLM (Qwen2.5-1.5B) extended with a speech encoder**.
Whisper is not the LLM and Whisper ASR is not used: only the encoder of
Whisper-small serves as the frozen speech encoder; the Whisper decoder is discarded.

```
audio ─┬─> speech encoder (Whisper-small encoder only, frozen) ─> projector ─> Qwen2.5-1.5B + ASR LoRA ─> "<spk1> ... <spk2> ..."
       └─> Nemotron 3 Diarization (frozen) ─> who spoke when ─> merge ─> Qwen2.5-1.5B + Summary LoRA ─> summary
```

## Setup

```bash
conda create -n xai610 python=3.12 -y && conda activate xai610
pip install torch transformers peft soundfile pyannote.metrics meeteval rouge-score bert-score
pip install 'nemo-toolkit[asr]'      # Step 1 only; use a separate env if it conflicts
```

Tested with torch 2.11, transformers 5.17, peft 0.21 (transformers >= 4.56 is required).

## Run the baseline (from `run/`)

```bash
# 0. Data: annotations + Mix-Headset audio of 60 train / 18 dev / 16 test meetings (~5.6 GB)
bash download_ami.sh data 60
python prepare_data.py --ann_dir data/ami_annotations --wav_dir data/wav \
    --out_dir data/manifests --num_train_meetings 60

# 1. Diarization (inference only) and DER (collar 0, overlap included)
python diarize_nemotron.py --wav_dir data/wav --meetings data/test.meetings.txt --out_dir exp/diar/nemotron
python eval_der.py --ref_dir data/ref_rttm --uem_dir data/ref_uem \
    --hyp_dir exp/diar/nemotron --meetings data/test.meetings.txt

# 2. Speech LLM: Stage 1 (projector) -> Stage 2 (projector + LoRA, speaker tags)
python train_speech_llm.py --stage 1 --train data/manifests/stage1_train.jsonl \
    --dev data/manifests/stage1_dev.jsonl --out_dir exp/stage1
python train_speech_llm.py --stage 2 --init exp/stage1 --train data/manifests/stage2_train.jsonl \
    --dev data/manifests/stage2_dev.jsonl --out_dir exp/stage2

# 3. Meeting-level transcription: System A (cascade) and B (joint); cpWER and WER
python transcribe_meeting.py --system A --ckpt exp/stage1 --rttm_dir exp/diar/nemotron --out exp/hyp/A.seglst.json
python transcribe_meeting.py --system B --ckpt exp/stage2 --rttm_dir exp/diar/nemotron --out exp/hyp/B.seglst.json
python eval_cpwer.py --ref data/manifests/test/ref.seglst.json --hyp exp/hyp/A.seglst.json
python eval_cpwer.py --ref data/manifests/test/ref.seglst.json --hyp exp/hyp/B.seglst.json
#    oracle diarization: --rttm_dir data/ref_rttm

# 4. Summarization LoRA and evaluation on three inputs (ref, A, B)
python train_summary.py --train data/manifests/summary_train.jsonl --out_dir exp/summary_lora
python summarize_eval.py --lora exp/summary_lora --ref data/manifests/test/summary_ref.jsonl \
    --hyp A=exp/hyp/A.seglst.json B=exp/hyp/B.seglst.json --out exp/summaries.jsonl
```

## Files

| File | Purpose |
|---|---|
| `download_ami.sh` | AMI annotations, Mix-Headset audio, BUT reference RTTM/UEM |
| `ami_util.py` | Word/speaker parsing, Stage-1 segments, 30-s tagged windows, summaries, SegLST/RTTM |
| `prepare_data.py` | Manifests for Stage 1/2, summarization, test references |
| `diarize_nemotron.py`, `eval_der.py` | Step 1 |
| `speech_llm.py`, `train_speech_llm.py` | Step 2 (model and training) |
| `transcribe_meeting.py`, `eval_cpwer.py` | Step 3 |
| `train_summary.py`, `summarize_eval.py` | Step 4 |

## Measured on one RTX 4090 (shared GPU)

* Stage 2, batch 4: 8.7 GB peak; batch 8 fits in 24 GB.
* Summary LoRA, 7.3k-token input: 5.9 GB peak, about 3 s per meeting.
