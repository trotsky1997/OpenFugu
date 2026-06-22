import os
"""Minimal GRPO smoke: prove the Conductor training loop actually turns on
ToolScale with our reward. Qwen3-0.6B, use_vllm=False (HF generation), 2 steps.
Not a full run — a proof the pipeline is wired end to end."""
import sys, os
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from custom_data.toolscale_data import make_datasets, make_reward_functions

MODEL = os.environ.get("FUGU_MODEL", "Qwen/Qwen3-0.6B")

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
ds = make_datasets(data_limit=64, tokenizer=tok)
rewards = make_reward_functions(include_format_reward=True)
print(f"[smoke] data ready: {len(ds['train_dataset'])} train; {len(rewards)} reward fns")

cfg = GRPOConfig(
    output_dir=os.environ.get("FUGU_OUT","grpo_smoke_out"),
    per_device_train_batch_size=2,
    num_generations=2,                 # tiny group
    max_prompt_length=512,
    max_completion_length=256,
    max_steps=2,                       # just prove it turns
    logging_steps=1,
    save_strategy="no",
    report_to=[],
    use_vllm=False,                    # HF generation — no vllm/version hell
    bf16=True,
    gradient_checkpointing=False,
    temperature=1.0,
)
trainer = GRPOTrainer(
    model=AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype="bfloat16").to("cuda"),
    processing_class=tok,
    reward_funcs=rewards,
    args=cfg,
    train_dataset=ds["train_dataset"],
)
print("[smoke] GRPOTrainer built; starting 2-step train...")
trainer.train()
print("[smoke] TRAIN LOOP COMPLETED — Conductor GRPO pipeline is live on ToolScale")
