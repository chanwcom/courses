"""A small Speech LLM: Whisper encoder (frozen) + projector + text LLM (LoRA).

  audio (<= 30 s, 16 kHz)
    -> Whisper-small encoder     : 1,500 frames x 768 (50 Hz), frozen
    -> frame stacking (k = 5)    : 300 x 3,840 (10 Hz)
    -> projector (2-layer MLP)   : 300 x d_llm, trained
    -> LLM (Qwen2.5-1.5B-Instruct) with LoRA

The audio embeddings are spliced into a chat prompt:
  <|im_start|>user\n [AUDIO] \n{instruction}<|im_end|>\n<|im_start|>assistant\n{target}<|im_end|>
"""
import math
import os

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          WhisperFeatureExtractor, WhisperModel)

SAMPLE_RATE = 16000
INSTRUCTIONS = {
    "asr": "Transcribe the audio.",
    "spk": "Transcribe the audio. Mark every speaker change with a tag "
           "<spk1>, <spk2>, ... numbered in order of first appearance.",
}


class Projector(nn.Module):
    def __init__(self, d_in, d_out, stack=5, d_hidden=2048):
        super().__init__()
        self.stack = stack
        self.net = nn.Sequential(nn.Linear(d_in * stack, d_hidden), nn.GELU(),
                                 nn.Linear(d_hidden, d_out))

    def forward(self, x):                      # x: (B, T, d_in)
        b, t, d = x.shape
        t = t // self.stack * self.stack
        x = x[:, :t].reshape(b, t // self.stack, d * self.stack)
        return self.net(x)


class SpeechLLM(nn.Module):
    def __init__(self, encoder_name="openai/whisper-small",
                 llm_name="Qwen/Qwen2.5-1.5B-Instruct", stack=5,
                 use_lora=False, lora_r=16, device="cuda"):
        super().__init__()
        self.device = device
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(encoder_name)
        self.encoder = WhisperModel.from_pretrained(
            encoder_name, dtype=torch.bfloat16).encoder.to(device).eval()
        self.encoder.requires_grad_(False)

        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_name, dtype=torch.bfloat16).to(device)
        self.llm.requires_grad_(False)
        if use_lora:
            self.add_lora(lora_r)
        self.projector = Projector(self.encoder.config.d_model,
                                   self.llm.config.hidden_size, stack).to(device)

    # ------------------------------------------------------------------ setup
    def add_lora(self, r=16):
        cfg = LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.05,
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                         task_type="CAUSAL_LM")
        self.llm = get_peft_model(self.llm, cfg)
        self.llm.print_trainable_parameters()

    def enable_gradient_checkpointing(self):
        self.llm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def save(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        torch.save(self.projector.state_dict(), f"{out_dir}/projector.pt")
        if isinstance(self.llm, PeftModel):
            self.llm.save_pretrained(f"{out_dir}/lora")

    def load(self, ckpt_dir, lora_trainable=False):
        self.projector.load_state_dict(
            torch.load(f"{ckpt_dir}/projector.pt", map_location=self.device))
        if os.path.isdir(f"{ckpt_dir}/lora"):
            self.llm = PeftModel.from_pretrained(self.llm, f"{ckpt_dir}/lora",
                                                 is_trainable=lora_trainable)

    # ------------------------------------------------------------------ audio
    @torch.no_grad()
    def encode(self, wavs):
        """wavs: list of 1-D float numpy arrays (16 kHz, <= 30 s).

        Returns a list of (n_i, d_llm) audio embeddings with padding removed.
        """
        feats = self.feature_extractor(wavs, sampling_rate=SAMPLE_RATE,
                                       return_tensors="pt").input_features
        h = self.encoder(feats.to(self.device, torch.bfloat16)).last_hidden_state
        n_frames = [min(h.shape[1], math.ceil(len(w) / SAMPLE_RATE * 50)) for w in wavs]
        return h.float(), n_frames

    def audio_embeddings(self, wavs):
        h, n_frames = self.encode(wavs)
        z = self.projector(h).to(torch.bfloat16)              # (B, 300, d_llm)
        s = self.projector.stack
        return [z[i, : max(1, math.ceil(n / s))] for i, n in enumerate(n_frames)]

    # ----------------------------------------------------------------- prompt
    def _ids(self, text):
        return self.tokenizer(text, add_special_tokens=False,
                              return_tensors="pt").input_ids[0].to(self.device)

    def _prompt_parts(self, instruction):
        pre = "<|im_start|>user\n"
        post = f"\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
        return self._ids(pre), self._ids(post)

    def _embed(self, ids):
        return self.llm.get_input_embeddings()(ids)

    # ---------------------------------------------------------------- forward
    def forward(self, wavs, texts, task="spk"):
        """Teacher-forced loss on the target text only."""
        audio = self.audio_embeddings(wavs)
        pre, post = self._prompt_parts(INSTRUCTIONS[task])
        seqs, labels = [], []
        for a, text in zip(audio, texts):
            tgt = self._ids(text + "<|im_end|>")
            emb = torch.cat([self._embed(pre), a, self._embed(post), self._embed(tgt)])
            lab = torch.full((emb.shape[0],), -100, device=self.device)
            lab[-len(tgt):] = tgt
            seqs.append(emb)
            labels.append(lab)
        L = max(e.shape[0] for e in seqs)                      # right padding
        d = seqs[0].shape[1]
        x = torch.zeros(len(seqs), L, d, dtype=torch.bfloat16, device=self.device)
        y = torch.full((len(seqs), L), -100, device=self.device)
        mask = torch.zeros(len(seqs), L, dtype=torch.long, device=self.device)
        for i, (e, l) in enumerate(zip(seqs, labels)):
            x[i, : len(e)], y[i, : len(l)], mask[i, : len(e)] = e, l, 1
        return self.llm(inputs_embeds=x, attention_mask=mask, labels=y).loss

    @torch.no_grad()
    def generate(self, wavs, task="spk", max_new_tokens=256, num_beams=1):
        audio = self.audio_embeddings(wavs)
        pre, post = self._prompt_parts(INSTRUCTIONS[task])
        seqs = [torch.cat([self._embed(pre), a, self._embed(post)]) for a in audio]
        L, d = max(e.shape[0] for e in seqs), seqs[0].shape[1]
        x = torch.zeros(len(seqs), L, d, dtype=torch.bfloat16, device=self.device)
        mask = torch.zeros(len(seqs), L, dtype=torch.long, device=self.device)
        for i, e in enumerate(seqs):                           # left padding
            x[i, L - len(e):], mask[i, L - len(e):] = e, 1
        out = self.llm.generate(inputs_embeds=x, attention_mask=mask,
                                max_new_tokens=max_new_tokens, num_beams=num_beams,
                                do_sample=False, repetition_penalty=1.1,
                                pad_token_id=self.tokenizer.pad_token_id)
        return [t.strip() for t in self.tokenizer.batch_decode(out, skip_special_tokens=True)]
