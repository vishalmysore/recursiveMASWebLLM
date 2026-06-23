#!/usr/bin/env python
"""
Train the RecursiveLink (W1/W2/W3) offline, export to recursivelink.json.

This is the part the browser CAN'T do (no autograd, no hidden states). In PyTorch
both halves the paper needs are first-class: `output_hidden_states=True` to READ the
last-layer hidden state, and `inputs_embeds=` to FEED a latent vector back in.

What the link is for: a pooled last-layer hidden vector lives in a different
distribution than the model's INPUT embeddings, so injecting it raw is out-of-
distribution and decodes to garbage. The link learns to map hidden-space -> input-
embedding-space so the injected "thought token" is readable.

Faithfulness notes vs. the paper (Sec. 3-4):
  • RecursiveLink:  R_in(h)  = h     + W2·GELU(W1·h)        (inner, within an agent)
                    R_out(h) = W3·h  + W2·GELU(W1·h)        (outer, across agents)
  • Inner loop:  cosine + magnitude objective aligning R(h) with the input-embedding
                 space (direction AND scale — scale is what derails the browser).
  • Outer loop:  unroll the looped system EXACTLY as the app injects it — ONE pooled
                 latent vector, mapped by the link, prepended to the agent prompt,
                 re-pooled each round; CE on the final logits, backprop through rounds.
                 (Matches latent-chain.js, which pools get_last_hidden to one 896-d
                 vector and injects it as a single token.)

This is a SCAFFOLD: homogeneous (one model plays every agent, so dims match and
W3 inits to identity). Adapt for heterogeneous agents by giving each ordered pair
its own R_out with the right [target_dim, source_dim] shape. The hyper-params are
placeholders — validate on real data and tune.

Usage:
  pip install torch transformers datasets
  # real data (recommended):
  python train_recursivelink.py --model Qwen/Qwen2.5-0.5B-Instruct --dataset tatsu-lab/alpaca --rounds 2
  # quick smoke test on the built-in toy set (won't generalize):
  python train_recursivelink.py --model Qwen/Qwen2.5-0.5B-Instruct
  # eval an already-trained link without retraining:
  python train_recursivelink.py --model Qwen/Qwen2.5-0.5B-Instruct --eval-only recursivelink.json
"""
import argparse, json
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class RecursiveLink(nn.Module):
    """Two-layer residual projection. target==source for the inner link."""
    def __init__(self, source_dim, target_dim, bottleneck=256):
        super().__init__()
        self.w1 = nn.Linear(source_dim, bottleneck, bias=True)
        self.w2 = nn.Linear(bottleneck, target_dim, bias=True)
        # residual branch: identity when dims match (inner), else a learned map (outer)
        self.w3 = (nn.Identity() if source_dim == target_dim
                   else nn.Linear(source_dim, target_dim, bias=False))
        # start near identity so training only learns the *correction*
        nn.init.zeros_(self.w2.weight); nn.init.zeros_(self.w2.bias)

    def forward(self, h):                      # h: [..., source_dim]
        # tanh-approx GELU to match recursive-link.js (export sets "gelu":"tanh")
        return self.w3(h) + self.w2(F.gelu(self.w1(h), approximate="tanh"))


def get_io(model):
    return model.get_input_embeddings()        # token-id -> input embedding


# ── dataset ──────────────────────────────────────────────────────────────────
TOY_WARM = ["The capital of France is Paris.", "2 + 2 = 4.",
            "Water boils at 100 degrees Celsius at sea level."]
TOY_PAIRS = [("Q: What is 12 * 12?\nA:", " 144"),
             ("Q: Capital of Japan?\nA:", " Tokyo")]


def load_pairs(name, limit):
    """(warm_texts, train_pairs) from a HF dataset. Handles a few common schemas
    (alpaca: instruction/input/output, dolly: instruction/context/response, plain
    text); else falls back to the first string columns."""
    from datasets import load_dataset
    ds = load_dataset(name, split=f"train[:{limit}]")
    cols = set(ds.column_names)
    warm, pairs = [], []
    for x in ds:
        q = a = None
        if {"instruction", "output"} <= cols:
            q = (x["instruction"] + (("\n" + x["input"]) if x.get("input") else "")).strip()
            a = (x.get("output") or "").strip()
        elif {"instruction", "response"} <= cols:
            q = (x["instruction"] + (("\n" + x.get("context", "")) if x.get("context") else "")).strip()
            a = (x.get("response") or "").strip()
        elif "text" in cols:
            a = (x["text"] or "").strip()
        else:
            strs = [v for v in x.values() if isinstance(v, str) and v.strip()]
            if strs:
                a = strs[0].strip()
            if len(strs) > 1:
                q, a = strs[0].strip(), strs[1].strip()
        if a:
            warm.append(a[:300])
            if q:
                pairs.append((q + "\nAnswer:", " " + a[:160]))
    if not warm:
        raise SystemExit(f"Could not extract text from dataset '{name}'.")
    return warm[:limit], (pairs[:limit] or TOY_PAIRS)


# ── training ─────────────────────────────────────────────────────────────────
def inner_loop(model, tok, link, texts, device, steps=300, lr=1e-3):
    """Warm-start: make R(last_hidden) match the input-embedding dist in direction AND scale."""
    emb = get_io(model)
    opt = torch.optim.AdamW(link.parameters(), lr=lr)
    model.eval()
    for step in range(steps):
        text = texts[step % len(texts)]
        ids = tok(text, return_tensors="pt", truncation=True, max_length=64).input_ids.to(device)
        with torch.no_grad():
            h = model(ids, output_hidden_states=True).hidden_states[-1][0]   # [seq, hidden]
            target = emb(ids)[0]                                             # [seq, hidden]
        pred = link(h)
        # Direction AND magnitude. Cosine alone is scale-invariant, so a link trained
        # only on it can emit a correctly-pointed but wrongly-scaled vector — and a
        # pooled last-hidden is ~30-50x larger than an input embedding. Injecting an
        # out-of-scale vector is exactly what derails generation in the browser.
        cos = (1 - F.cosine_similarity(pred, target, dim=-1)).mean()
        mag = F.l1_loss(pred.norm(dim=-1), target.norm(dim=-1))
        loss = cos + 0.1 * mag
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 25 == 0:
            print(f"  [inner] step {step:4d}  loss {loss.item():.4f} "
                  f"(cos {cos.item():.4f}  mag {mag.item():.3f})")
    return link


def _pooled_hidden(model, *, inputs_embeds=None, input_ids=None):
    out = (model(inputs_embeds=inputs_embeds, output_hidden_states=True)
           if inputs_embeds is not None
           else model(input_ids, output_hidden_states=True))
    return out.hidden_states[-1].mean(dim=1, keepdim=True)   # [1, 1, hidden]


def outer_loop(model, tok, links, data, device, agent_prompt="", rounds=2, steps=300, lr=5e-4):
    """Co-optimize the loop the SAME way the app injects (latent-chain.js): ONE pooled
    latent vector, mapped by the link, prepended to the agent prompt, re-pooled each
    round; the final round decodes the answer."""
    params = [p for lk in links for p in lk.parameters()]
    opt = torch.optim.AdamW(params, lr=lr)
    emb = get_io(model)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)                 # base model frozen; only links train
    pe = None
    if agent_prompt.strip():
        pid = tok(agent_prompt, return_tensors="pt").input_ids.to(device)
        pe = emb(pid)                           # [1, P, hidden]

    for step in range(steps):
        text, answer = data[step % len(data)]
        ids = tok(text, return_tensors="pt", truncation=True, max_length=64).input_ids.to(device)
        labels = tok(answer, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            latent = _pooled_hidden(model, input_ids=ids)       # upstream pooled latent
        inp = None
        for r in range(rounds):
            tok_emb = links[r % len(links)](latent)             # link maps latent -> 1 token
            inp = torch.cat([tok_emb, pe], dim=1) if pe is not None else tok_emb
            if r < rounds - 1:
                latent = _pooled_hidden(model, inputs_embeds=inp)
        logits = model(inputs_embeds=inp).logits[:, -labels.shape[1]:, :]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 25 == 0:
            print(f"  [outer] step {step:4d}  ce {loss.item():.4f}")
    return links


@torch.no_grad()
def evaluate(model, tok, links, device, agent_prompt="", prompts=None, max_new=40):
    """Sanity check: decode a hidden state through the link vs. raw vs. normal gen.
    If the link works, LINK should read more sensibly than RAW."""
    emb = get_io(model)
    prompts = prompts or [
        "What is the capital of Japan?",
        "Suggest a neighbourhood to stay in Tokyo.",
        "Give a one-line tip for visiting Shibuya.",
    ]
    pe = None
    if agent_prompt.strip():
        pe = emb(tok(agent_prompt, return_tensors="pt").input_ids.to(device))

    def gen_embeds(inp):
        am = torch.ones(inp.shape[:2], dtype=torch.long, device=inp.device)
        o = model.generate(inputs_embeds=inp, attention_mask=am, max_new_tokens=max_new,
                           do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(o[0], skip_special_tokens=True).strip()

    def gen_text(q):
        ids = tok(q, return_tensors="pt").input_ids.to(device)
        o = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                           pad_token_id=tok.eos_token_id)
        return tok.decode(o[0][ids.shape[1]:], skip_special_tokens=True).strip()

    print("\n==== sanity check: hidden -> (link) -> text ====")
    print("If the link helps, LINK should read better than RAW.\n")
    for q in prompts:
        ids = tok(q, return_tensors="pt").input_ids.to(device)
        latent = model(ids, output_hidden_states=True).hidden_states[-1].mean(1, keepdim=True)
        raw = torch.cat([latent, pe], 1) if pe is not None else latent
        lk = links[-1](latent)
        lk = torch.cat([lk, pe], 1) if pe is not None else lk
        print(f"PROMPT : {q}")
        print(f"  gold : {gen_text(q)!r}")
        print(f"  RAW  : {gen_embeds(raw)!r}")
        print(f"  LINK : {gen_embeds(lk)!r}\n")


def export(links, hidden, path="recursivelink.json"):
    def dump(lk):
        d = {"w1": lk.w1.weight.detach().cpu().tolist(), "b1": lk.w1.bias.detach().cpu().tolist(),
             "w2": lk.w2.weight.detach().cpu().tolist(), "b2": lk.w2.bias.detach().cpu().tolist()}
        if isinstance(lk.w3, nn.Linear):
            d["w3"] = lk.w3.weight.detach().cpu().tolist()
        return d
    obj = {"hidden": hidden, "gelu": "tanh", "links": [dump(lk) for lk in links]}
    with open(path, "w") as f:
        json.dump(obj, f)
    print(f"==> wrote {path}  ({len(links)} link(s), hidden={hidden})")


def load_links_json(path, hidden, rounds, device):
    """Rebuild RecursiveLink modules from a recursivelink.json (for --eval-only)."""
    obj = json.load(open(path))
    mods = []
    for d in obj["links"]:
        lk = RecursiveLink(obj["hidden"], obj["hidden"]).to(device)
        with torch.no_grad():
            lk.w1.weight.copy_(torch.tensor(d["w1"])); lk.w1.bias.copy_(torch.tensor(d["b1"]))
            lk.w2.weight.copy_(torch.tensor(d["w2"])); lk.w2.bias.copy_(torch.tensor(d["b2"]))
            if "w3" in d and isinstance(lk.w3, nn.Linear):
                lk.w3.weight.copy_(torch.tensor(d["w3"]))
        mods.append(lk)
    return mods


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--dataset", default=None,
                    help="HF dataset id (e.g. tatsu-lab/alpaca). Omit to use the built-in toy set.")
    ap.add_argument("--limit", type=int, default=600, help="examples to pull from --dataset")
    ap.add_argument("--inner-steps", type=int, default=300)
    ap.add_argument("--outer-steps", type=int, default=300)
    ap.add_argument("--agent-prompt", default="",
                    help="optional role framing prepended after the latent token (kept minimal)")
    ap.add_argument("--out", default="recursivelink.json")
    ap.add_argument("--eval-only", default=None,
                    help="path to an existing recursivelink.json — just run the sanity check, no training")
    ap.add_argument("--no-eval", dest="do_eval", action="store_false", default=True)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  model={args.model}  dataset={args.dataset or 'TOY'}")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(device)
    hidden = model.config.hidden_size

    if args.eval_only:
        links = load_links_json(args.eval_only, hidden, args.rounds, device)
        evaluate(model, tok, links, device, agent_prompt=args.agent_prompt)
        return

    if args.dataset:
        warm_texts, train_pairs = load_pairs(args.dataset, args.limit)
    else:
        print("WARNING: using the built-in toy dataset — pass --dataset for a usable link.")
        warm_texts, train_pairs = TOY_WARM, TOY_PAIRS
    print(f"warm_texts={len(warm_texts)}  train_pairs={len(train_pairs)}")

    link = RecursiveLink(hidden, hidden).to(device)
    print("== Stage 1: inner loop (warm-start: hidden -> embedding space) ==")
    inner_loop(model, tok, link, warm_texts, device, steps=args.inner_steps)

    links = [RecursiveLink(hidden, hidden).to(device) for _ in range(args.rounds)]
    for lk in links:
        lk.load_state_dict(link.state_dict())   # init each round from the warm-start
    print("== Stage 2: outer loop (co-optimize the looped system) ==")
    outer_loop(model, tok, links, train_pairs, device,
               agent_prompt=args.agent_prompt, rounds=args.rounds, steps=args.outer_steps)

    export(links, hidden, args.out)
    if args.do_eval:
        evaluate(model, tok, links, device, agent_prompt=args.agent_prompt)


if __name__ == "__main__":
    main()
