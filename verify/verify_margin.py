import os
"""Close two remaining loose ends, locally + on GPU:
  (1) the 2 missed agent cases: are they low-margin (numeric noise) or real disagreement?
  (2) cross-check our agent logits vs the recorded logits fixture (independent impl).
"""
import json, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("FUGU_MODEL","Qwen/Qwen3-0.6B")
v = np.load(os.environ.get("FUGU_VECTOR","artifacts/model_iter_60.npy")).astype(np.float64)
cases = json.load(open(os.environ.get("FUGU_FIXTURE","artifacts/qwen_router_prompt_eval_cases.json")))["cases"]

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
sd = model.state_dict()
keep = lambda k: ("model.layers." not in k) or ("model.layers.26." in k)
svf_keys = [k for k in sd if sd[k].ndim==2 and min(sd[k].shape)>1 and keep(k)]
off=0
with torch.no_grad():
    for k in svf_keys:
        W=sd[k].float(); U,S,Vh=torch.linalg.svd(W,full_matrices=False); n=S.numel()
        sc=torch.from_numpy(v[off:off+n].copy()).float()+1.0; off+=n; sS=S*sc
        model.state_dict()[k].copy_(((U@torch.diag(sS)@Vh)*(S.sum()/sS.sum())).to(sd[k].dtype))
head=torch.from_numpy(v[9216:].copy()).float().reshape(10,1024)
fmt=lambda m:"\n".join(f'{x["role"]}: {x["content"]}' for x in m)

print("=== missed agent cases: margin analysis ===")
for c in cases:
    ids=tok(fmt(c["messages"]),return_tensors="pt")
    with torch.no_grad(): h=model.model(**ids).last_hidden_state[0,-2,:]
    al=(head@h)[:7]
    top=torch.topk(al,2).values; ai=int(al.argmax()); ea=c["expected"]["agent_id"]
    if ai!=ea:
        margin=float(top[0]-top[1])
        gap_to_exp=float(al[ai]-al[ea])
        print(f"  case {c['id']}: got {ai} exp {ea} | top1-top2 margin={margin:.3f} | "
              f"logit[{ai}]-logit[{ea}]={gap_to_exp:.3f} | tag={c.get('tags')}")
        print(f"    logits={al.detach().numpy().round(2)}")
print("\n(low margin -> numeric precision; large gap -> real disagreement)")
