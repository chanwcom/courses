"""Step 4a: LoRA fine-tuning of the LLM for meeting summarization (text only).

  python train_summary.py --train data/manifests/summary_train.jsonl --out_dir exp/summary_lora
"""
import argparse
import json
import random
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          get_cosine_schedule_with_warmup)

INSTRUCTION = ("Summarize the meeting below in one paragraph. Say who proposed, "
               "decided or objected to what when it matters.\n\n")


def build_prompt(tok, transcript, max_transcript_tokens):
    ids = tok(transcript, add_special_tokens=False).input_ids
    if len(ids) > max_transcript_tokens:      # keep the beginning and the end
        half = max_transcript_tokens // 2
        transcript = tok.decode(ids[:half]) + "\n...\n" + tok.decode(ids[-half:])
    msgs = [{"role": "user", "content": INSTRUCTION + transcript}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--llm", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--max_transcript_tokens", type=int, default=7000)
    ap.add_argument("--lora_r", type=int, default=16)
    args = ap.parse_args()
    torch.manual_seed(0)

    tok = AutoTokenizer.from_pretrained(args.llm)
    model = AutoModelForCausalLM.from_pretrained(args.llm, dtype=torch.bfloat16).cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    rows = [json.loads(l) for l in open(args.train)]
    examples = []
    for r in rows:
        p = tok(build_prompt(tok, r["transcript"], args.max_transcript_tokens),
                add_special_tokens=False).input_ids
        t = tok(r["summary"] + "<|im_end|>", add_special_tokens=False).input_ids
        examples.append((torch.tensor(p + t), torch.tensor(t)))
    print(f"{len(examples)} meetings, max length {max(len(x) for x, _ in examples)} tokens")

    params = [p for p in model.parameters() if p.requires_grad]
    steps = args.epochs * len(examples) // args.grad_accum
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.05 * steps)), steps)
    model.train()
    t0, n = time.time(), 0
    for ep in range(args.epochs):
        random.shuffle(examples)
        for x, y in examples:                 # batch size 1, long sequences
            # Logits only for the summary positions: the full 150k-vocab logits
            # of a 7k-token transcript would need several GB.
            logits = model(input_ids=x[None].cuda(), logits_to_keep=len(y) + 1).logits[0, :-1]
            loss = torch.nn.functional.cross_entropy(logits.float(), y.cuda())
            (loss / args.grad_accum).backward()
            n += 1
            if n % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            if n % 20 == 0:
                print(f"ep {ep} ex {n} loss {loss.item():.3f} {(time.time() - t0) / 60:.1f} min",
                      flush=True)
    model.save_pretrained(args.out_dir)
    print("saved to", args.out_dir)


if __name__ == "__main__":
    main()
