"""Core routines for the Phase 0 validation gate (protocol section 3.4)."""

import torch
import torch.nn.functional as F

from gamma.hooks import StreamExtractor


def tokenize_fixed_len(tokenizer, docs: list[str], seq_len: int) -> torch.Tensor:
    seqs = []
    for d in docs:
        ids = tokenizer(d, truncation=True, max_length=seq_len, return_tensors="pt").input_ids[0]
        if ids.shape[0] == seq_len:
            seqs.append(ids)
    if not seqs:
        raise RuntimeError(f"No documents long enough for seq_len={seq_len}")
    return torch.stack(seqs)


@torch.no_grad()
def collect_states(model, spec, tokenizer, docs: list[str], seq_len: int = 64, batch_size: int = 16, device: str = "cuda") -> dict:
    """Run the model over fixed-length chunks and cache per-layer states.

    Positions are shifted so that state at position t is paired with the
    model's own next-token target at t+1 -- this is what both KL-to-final
    and next-token-perplexity metrics need.
    """
    input_ids = tokenize_fixed_len(tokenizer, docs, seq_len)
    extractor = StreamExtractor(model, spec)
    x_chunks, mixer_chunks, logit_chunks = [], [], []
    try:
        for i in range(0, input_ids.shape[0], batch_size):
            batch = input_ids[i : i + batch_size].to(device)
            out = extractor.run(batch)
            x = out["x"][:, :, :-1, :]  # [L, B, T-1, H]
            x_chunks.append(x.reshape(x.shape[0], -1, x.shape[-1]).float().cpu())
            if "mixer_output" in out:
                mixer_output = out["mixer_output"][:, :, :-1, :]
                mixer_chunks.append(mixer_output.reshape(mixer_output.shape[0], -1, mixer_output.shape[-1]).float().cpu())
            logits = out["logits"][:, :-1, :]
            logit_chunks.append(logits.reshape(-1, logits.shape[-1]).float().cpu())
    finally:
        extractor.remove()

    result = {
        "x": torch.cat(x_chunks, dim=1),
        "final_logits": torch.cat(logit_chunks, dim=0),
        "target_ids": input_ids[:, 1:].reshape(-1),
    }
    if mixer_chunks:
        result["mixer_output"] = torch.cat(mixer_chunks, dim=1)
    return result


@torch.no_grad()
def collect_recurrent_states(model, spec, tokenizer, docs: list[str], seq_len: int = 48, batch_size: int = 16, device: str = "cuda") -> dict:
    """Genuine recurrent state h_t^(l), via RecurrentStateExtractor
    (step-by-step cached decoding -- see gamma/hooks.py). Slower than
    collect_states by construction: O(seq_len) forward calls per batch of
    documents instead of one batched call.

    Returns {"state": [L, N, d_inner, d_state], "final_logits": [N, V],
    "target_ids": [N]}, with the same t <-> t+1 shift as collect_states.
    """
    from gamma.hooks import RecurrentStateExtractor

    input_ids = tokenize_fixed_len(tokenizer, docs, seq_len)
    extractor = RecurrentStateExtractor(model, spec)

    state_chunks, logit_chunks, target_chunks = [], [], []
    for i in range(0, input_ids.shape[0], batch_size):
        batch = input_ids[i : i + batch_size].to(device)
        out = extractor.run(batch)  # state: [T, L, B, d_inner, d_state], logits: [B, T, V]

        state = out["state"][:-1]  # [T-1, L, B, d_inner, d_state]
        state = state.permute(1, 0, 2, 3, 4)  # [L, T-1, B, d_inner, d_state]
        L = state.shape[0]
        state = state.reshape(L, -1, state.shape[-2], state.shape[-1])  # [L, (T-1)*B, d_inner, d_state]
        state_chunks.append(state.float())

        logits = out["logits"][:, :-1, :].permute(1, 0, 2)  # [T-1, B, V]
        logit_chunks.append(logits.reshape(-1, logits.shape[-1]).float())

        target_ids = batch[:, 1:].permute(1, 0).reshape(-1)  # [(T-1)*B]
        target_chunks.append(target_ids.cpu())

    return {
        "state": torch.cat(state_chunks, dim=1),
        "final_logits": torch.cat(logit_chunks, dim=0),
        "target_ids": torch.cat(target_chunks, dim=0),
    }


def layer_metrics(lens_logits: torch.Tensor, final_logits: torch.Tensor, target_ids: torch.Tensor) -> dict:
    logp_lens = F.log_softmax(lens_logits, dim=-1)
    p_lens = logp_lens.exp()
    p_final = F.softmax(final_logits, dim=-1)

    kl = F.kl_div(logp_lens, p_final, reduction="batchmean").item()
    top1_agree = (lens_logits.argmax(-1) == final_logits.argmax(-1)).float().mean().item()
    entropy = -(p_lens * logp_lens).sum(-1).mean().item()
    nll = F.nll_loss(logp_lens, target_ids, reduction="mean").item()
    ppl = float(torch.exp(torch.clamp(torch.tensor(nll), max=30.0)))
    return {"kl": kl, "top1_agree": top1_agree, "entropy": entropy, "ppl": ppl}


def spearman(xs: list[float], ys: list[float]) -> float:
    """Dependency-free Spearman rank correlation."""
    n = len(xs)

    def ranks(vals):
        order = sorted(range(n), key=lambda i: vals[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg_rank = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg_rank
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx) ** 0.5
    vy = sum((b - my) ** 2 for b in ry) ** 0.5
    if vx == 0 or vy == 0:
        return 0.0
    return cov / (vx * vy)
