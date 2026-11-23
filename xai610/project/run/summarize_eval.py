"""Step 4b: summarize test meetings from three inputs and score them.

  pip install rouge-score bert-score
  python summarize_eval.py --lora exp/summary_lora \
      --ref data/manifests/test/summary_ref.jsonl \
      --hyp A=exp/hyp/A_nemotron.seglst.json B=exp/hyp/B_nemotron.seglst.json \
      --out exp/summaries.jsonl

Input "ref" is the reference transcript with reference speakers.
Use --lora none to measure the zero-shot (not fine-tuned) LLM.
"""
import argparse
import json
from collections import defaultdict

import torch
from bert_score import score as bert_score
from peft import PeftModel
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer

import ami_util
from train_summary import build_prompt


def seglst_transcripts(path):
    by_meeting = defaultdict(list)
    for s in json.load(open(path)):
        by_meeting[s["session_id"]].append(
            dict(speaker=s["speaker"], start=s["start_time"], text=s["words"]))
    return {m: ami_util.turns_to_transcript(t) for m, t in by_meeting.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--hyp", nargs="*", default=[], help="NAME=seglst.json")
    ap.add_argument("--lora", default="none")
    ap.add_argument("--llm", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_transcript_tokens", type=int, default=7000)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.llm)
    model = AutoModelForCausalLM.from_pretrained(args.llm, dtype=torch.bfloat16).cuda().eval()
    if args.lora != "none":
        model = PeftModel.from_pretrained(model, args.lora)

    refs = [json.loads(l) for l in open(args.ref)]
    inputs = {"ref": {r["meeting"]: r["transcript"] for r in refs}}
    for h in args.hyp:
        name, path = h.split("=")
        inputs[name] = seglst_transcripts(path)

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    out = open(args.out, "w")
    for name, transcripts in inputs.items():
        cands, golds = [], []
        for r in refs:
            prompt = build_prompt(tok, transcripts.get(r["meeting"], ""), args.max_transcript_tokens)
            ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
            with torch.no_grad():
                gen = model.generate(ids, max_new_tokens=400, do_sample=False,
                                     repetition_penalty=1.1)
            summ = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True).strip()
            cands.append(summ)
            golds.append(r["summary"])
            out.write(json.dumps(dict(input=name, meeting=r["meeting"], summary=summ)) + "\n")
        rl = sum(scorer.score(g, c)["rougeL"].fmeasure for g, c in zip(golds, cands)) / len(refs)
        _, _, f1 = bert_score(cands, golds, lang="en", rescale_with_baseline=True)
        print(f"[{name}] ROUGE-L {100 * rl:.2f}  BERTScore-F1 {100 * f1.mean().item():.2f}")


if __name__ == "__main__":
    main()
