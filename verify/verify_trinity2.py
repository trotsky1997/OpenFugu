import os
"""v2: rigorous verification — fixes two bugs in v1:
  (1) consume SVF offsets in real state_dict ORDER (embed first, lm_head last),
      matching export_sakana_trinity_safetensors.py (state_dict iteration +
      should_keep_svd_key: keep non-layer matrices + layer-26).
  (2) actually WRITE reconstructed SVF weights back into the model before forward.
Decisive test: trinity_coordinator fixture reports agent_id=4, role_id=0 for a
fixed transcript. If applying SVF in this order reproduces it, the order is proven.
"""
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("FUGU_MODEL","Qwen/Qwen3-0.6B")
VEC = os.environ.get("FUGU_VECTOR","artifacts/model_iter_60.npy")
OPT_LAYERS = {26}

v = np.load(VEC).astype(np.float64)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
sd = model.state_dict()

def keep(k):
    if "model.layers." not in k:
        return True                       # embed_tokens, lm_head, norms
    return any(f"model.layers.{i}." in k for i in OPT_LAYERS)

# iterate in STATE_DICT ORDER, keep 2-D matrices that pass the filter
svf_keys = [k for k in sd.keys() if sd[k].ndim == 2 and min(sd[k].shape) > 1 and keep(k)]
print("[order] SVF target matrices in state_dict order:")
for i,k in enumerate(svf_keys): print(f"  {i}: {k}  {tuple(sd[k].shape)} sv={min(sd[k].shape)}")

# apply SVF in that order, writing back into the model
off = 0
new_state = {}
for k in svf_keys:
    W = sd[k].float()
    U,S,Vh = torch.linalg.svd(W, full_matrices=False)
    n = S.numel()
    scale = torch.from_numpy(v[off:off+n].copy()).float() + 1.0
    off += n
    sS = S*scale
    new_state[k] = ((U@torch.diag(sS)@Vh)*(S.sum()/sS.sum())).to(sd[k].dtype)
print(f"[svf] consumed {off} offsets (expect 9216: {off==9216})")
with torch.no_grad():
    for k,w in new_state.items():
        model.state_dict()[k].copy_(w)
print("[svf] adapted weights written into model")

# head = last 10240, reshape (num_agents+num_roles, hidden) = (10,1024)
head = torch.from_numpy(v[9216:].copy()).float().reshape(10,1024)

def route(prompt, raw=False):
    if raw:
        ids = tok(prompt, return_tensors="pt")
    else:
        text = tok.apply_chat_template([{"role":"user","content":prompt}],
                                       tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt")
    with torch.no_grad():
        h = model.model(**ids, output_hidden_states=False).last_hidden_state[0,-2,:]
    lg = head @ h
    a, r = lg[:7], lg[7:]
    return int(a.argmax()), int(r.argmax()), a.detach().numpy().round(2), r.detach().numpy().round(2)

# the trinity_coordinator fixture prompt (README quoted it)
for p,raw in [("Select a TRINITY role for this reasoning task.", True),
              ("Select a TRINITY role for this reasoning task.", False)]:
    ai,ri,al,rl = route(p, raw)
    print(f"[fixture raw={raw}] agent={ai} role={ri}({['Worker','Thinker','Verifier'][ri]})  "
          f"agent_logits={al} role_logits={rl}")
print("\nfixture expects agent_id=4, role_id=0 (per trinity_coordinator README)")
