import os
"""37-case batch verification — statistical proof (not single-case luck).
Applies SVF in state_dict order, runs each fixture transcript through the
router, compares argmax agent/role to the fixture's expected values.
Reports hit-rate AND the naive baseline (always-guess-most-common) so we
know whether matches are real signal or just the {4:19,0:17} class prior.
Tries raw transcript ('role: content') vs chat-template formatting.
"""
import json, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter

MODEL = os.environ.get("FUGU_MODEL","Qwen/Qwen3-0.6B")
v = np.load(os.environ.get("FUGU_VECTOR","artifacts/model_iter_60.npy")).astype(np.float64)
cases = json.load(open(os.environ.get("FUGU_FIXTURE","artifacts/qwen_router_prompt_eval_cases.json")))["cases"]

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
sd = model.state_dict()

def keep(k):
    if "model.layers." not in k: return True
    return "model.layers.26." in k
svf_keys = [k for k in sd if sd[k].ndim==2 and min(sd[k].shape)>1 and keep(k)]
off=0
with torch.no_grad():
    for k in svf_keys:
        W=sd[k].float(); U,S,Vh=torch.linalg.svd(W,full_matrices=False); n=S.numel()
        scale=torch.from_numpy(v[off:off+n].copy()).float()+1.0; off+=n
        sS=S*scale
        model.state_dict()[k].copy_(((U@torch.diag(sS)@Vh)*(S.sum()/sS.sum())).to(sd[k].dtype))
assert off==9216
head=torch.from_numpy(v[9216:].copy()).float().reshape(10,1024)

def fmt_raw(msgs): return "\n".join(f'{m["role"]}: {m["content"]}' for m in msgs)
def fmt_chat(msgs): return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def run(fmt):
    a_hit=r_hit=both=0
    for c in cases:
        ids=tok(fmt(c["messages"]), return_tensors="pt")
        with torch.no_grad():
            h=model.model(**ids).last_hidden_state[0,-2,:]
        lg=head@h
        ai=int(lg[:7].argmax()); ri=int(lg[7:].argmax())
        ea=c["expected"]["agent_id"]; er=c["expected"]["role_id"]
        a_hit+=(ai==ea); r_hit+=(ri==er); both+=(ai==ea and ri==er)
    N=len(cases)
    return a_hit,r_hit,both,N

# baselines: always guess the most common class
ea_all=[c["expected"]["agent_id"] for c in cases]; er_all=[c["expected"]["role_id"] for c in cases]
base_a=Counter(ea_all).most_common(1)[0][1]; base_r=Counter(er_all).most_common(1)[0][1]
print(f"[baseline] always-guess-mode: agent {base_a}/37={base_a/37:.0%}  role {base_r}/37={base_r/37:.0%}")
print(f"[prior] agent dist={dict(Counter(ea_all))} role dist={dict(Counter(er_all))}\n")

for name,fmt in [("raw 'role: content'",fmt_raw),("chat_template",fmt_chat)]:
    a,r,b,N=run(fmt)
    print(f"[{name}] agent {a}/{N}={a/N:.0%}  role {r}/{N}={r/N:.0%}  both {b}/{N}={b/N:.0%}")
