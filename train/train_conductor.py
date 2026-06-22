import os
"""Real GRPO training run for the Conductor on ToolScale.
Llama-3.2-3B-Instruct (follows the JSON format), 8 generations/group for GRPO
variance, enough steps to see reward climb off zero. Saves the trained adapter.
Logs reward每步 so we can see learning."""
import sys, json, os
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig
from trl.trainer.callbacks import RichProgressCallback
from transformers import TrainerCallback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from custom_data.toolscale_data import make_datasets, make_reward_functions

MODEL = os.environ.get("FUGU_BASE_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
OUT = os.environ.get("FUGU_OUT", "conductor_out")

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
ds = make_datasets(data_limit=512, tokenizer=tok)
rewards = make_reward_functions(output_dir=OUT, include_format_reward=True)
print(f"[train] {len(ds['train_dataset'])} train rows; reward fns: {[f.__name__ for f in rewards]}", flush=True)

class RewardLog(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kw):
        if logs and "reward" in logs:
            print(f"[step {state.global_step}] reward={logs.get('reward'):.3f} "
                  f"fmt={logs.get('rewards/format_reward/mean',0):.3f} "
                  f"act={logs.get('rewards/action_reward/mean',0):.3f} "
                  f"loss={logs.get('loss',0):.4f} complen={logs.get('completions/mean_length',0):.0f}",
                  flush=True)

cfg = GRPOConfig(
    output_dir=OUT,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    num_generations=8,                 # GRPO group size — needs variance to learn
    max_prompt_length=640,
    max_completion_length=320,
    max_steps=40,                      # enough to see reward move
    learning_rate=1e-5,
    logging_steps=1,
    save_strategy="steps", save_steps=20,
    report_to=[],
    use_vllm=False,
    bf16=True,
    gradient_checkpointing=True,
    temperature=1.0,
    beta=0.0,                          # no KL — matches Fugu-Ultra report (beta=0)
)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype="bfloat16").to("cuda")
model.config.use_cache = False
trainer = GRPOTrainer(
    model=model, processing_class=tok, reward_funcs=rewards,
    args=cfg, train_dataset=ds["train_dataset"], callbacks=[RewardLog()],
)
print("[train] starting GRPO — watch reward climb off zero...", flush=True)
trainer.train()
trainer.save_model(OUT)
print(f"[train] DONE — Conductor trained on ToolScale, saved to {OUT}", flush=True)
