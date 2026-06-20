#!/usr/bin/env python
"""
Train the RecursiveLink (W1/W2/W3) offline, export to recursivelink.json.

This is the part the browser CAN'T do (no autograd, no hidden states). In PyTorch
both halves the paper needs are first-class: `output_hidden_states=True` to READ the
last-layer hidden state, and `inputs_embeds=` to FEED a latent vector back in.

Faithfulness notes vs. the paper (Sec. 3-4):
  • RecursiveLink:  R_in(h)  = h     + W2·GELU(W1·h)        (inner, within an agent)
                    R_out(h) = W3·h  + W2·GELU(W1·h)        (outer, across agents)
  • Inner loop:  cosine objective aligning R_in(h) with the input-embedding space.
  • Outer loop:  unroll the looped system over `rounds`, CE on the final logits,
                 backprop through all rounds (shared credit assignment).

This is a SCAFFOLD: homogeneous (one model plays every agent, so dims match and
W3 can init to identity). Adapt for heterogeneous agents by giving each ordered
pair its own R_out with the right [target_dim, source_dim] shape. Validate on real
data; the hyper-params are placeholders.

Usage:
  pip install torch transformers datasets
  python train_recursivelink.py --model Qwen/Qwen2.5-0.5B-Instruct --rounds 2
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
        return self.w3(h) + self.w2(F.gelu(self.w1(h)))


def get_io(model):
    emb = model.get_input_embeddings()         # token-id -> input embedding
    return emb


def inner_loop(model, tok, link, texts, device, steps=200, lr=1e-3):
    """Warm-start R_in: make R_in(last_hidden) align with the input-embedding dist."""
    emb = get_io(model)
    opt = torch.optim.AdamW(link.parameters(), lr=lr)
    model.eval()
    for step in range(steps):
        text = texts[step % len(texts)]
        ids = tok(text, return_tensors="pt", truncation=True, max_length=64).input_ids.to(device)
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
            h = out.hidden_states[-1][0]        # [seq, hidden] last-layer
            target = emb(ids)[0]                # [seq, hidden] input-embedding of same tokens
        pred = link(h)
        loss = (1 - F.cosine_similarity(pred, target, dim=-1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 25 == 0:
            print(f"  [inner] step {step:4d}  cos-loss {loss.item():.4f}")
    return link


def outer_loop(model, tok, links, data, device, rounds=2, steps=200, lr=5e-4):
    """Co-optimize the looped system: unroll `rounds`, CE on final logits."""
    params = [p for lk in links for p in lk.parameters()]
    opt = torch.optim.AdamW(params, lr=lr)
    emb = get_io(model)
    model.eval()                                # base model frozen; only links train
    for p in model.parameters(): p.requires_grad_(False)

    for step in range(steps):
        text, label_ids = data[step % len(data)]
        ids = tok(text, return_tensors="pt", truncation=True, max_length=64).input_ids.to(device)
        labels = tok(label_ids, return_tensors="pt").input_ids.to(device)

        inp = emb(ids)                          # [1, seq, hidden] start in latent space
        # Unroll the recursive loop; intermediate rounds stay latent.
        for r in range(rounds):
            out = model(inputs_embeds=inp, output_hidden_states=True)
            h = out.hidden_states[-1]           # [1, seq, hidden]
            link = links[r % len(links)]
            inp = link(h)                        # latent -> next-round input embeds
        # final round: decode through the LM head
        logits = model(inputs_embeds=inp).logits[:, -labels.shape[1]:, :]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 25 == 0:
            print(f"  [outer] step {step:4d}  ce {loss.item():.4f}")
    return links


def export(links, hidden, path="recursivelink.json"):
    def dump(lk):
        d = {"w1": lk.w1.weight.detach().cpu().tolist(), "b1": lk.w1.bias.detach().cpu().tolist(),
             "w2": lk.w2.weight.detach().cpu().tolist(), "b2": lk.w2.bias.detach().cpu().tolist()}
        if isinstance(lk.w3, nn.Linear):
            d["w3"] = lk.w3.weight.detach().cpu().tolist()
        return d
    obj = {"hidden": hidden, "gelu": "tanh", "links": [dump(lk) for lk in links]}
    with open(path, "w") as f: json.dump(obj, f)
    print(f"==> wrote {path}  ({len(links)} link(s), hidden={hidden})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--out", default="recursivelink.json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(device)
    hidden = model.config.hidden_size

    # TODO: replace with your real training set (question, answer) pairs per domain.
    warm_texts = ["The capital of France is Paris.", "2 + 2 = 4.",
                  "Water boils at 100 degrees Celsius at sea level."]
    train_pairs = [("Q: What is 12 * 12?\nA:", " 144"),
                   ("Q: Capital of Japan?\nA:", " Tokyo")]

    link = RecursiveLink(hidden, hidden).to(device)
    print("== Stage 1: inner loop (warm-start R_in) ==")
    inner_loop(model, tok, link, warm_texts, device)

    links = [RecursiveLink(hidden, hidden).to(device) for _ in range(args.rounds)]
    for lk in links: lk.load_state_dict(link.state_dict())   # init from warm-start
    print("== Stage 2: outer loop (co-optimize the loop) ==")
    outer_loop(model, tok, links, train_pairs, device, rounds=args.rounds)

    export(links, hidden, args.out)


if __name__ == "__main__":
    main()
