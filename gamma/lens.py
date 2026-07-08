"""Gamma-lens: the vocab probe (protocol section 3.2).

Two readout variants, both required by the protocol:
  V1 (zero-shot): direct projection through the model's own frozen final
                  norm + unembedding. No learned components.
  V2 (tuned lens): per-layer learned affine "translator" (Belrose et al.
                  2023 tuned-lens methodology), trained to minimize KL
                  against the model's own final-layer logits on held-out
                  text. V2 is the workhorse; V1 is the purist check.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_final_norm_and_head(model, spec):
    if spec.architecture == "mamba":
        return model.backbone.norm_f, model.lm_head
    elif spec.architecture == "pythia":
        return model.gpt_neox.final_layer_norm, model.embed_out
    raise ValueError(f"Unsupported architecture: {spec.architecture}")


class GammaLensV1:
    """Zero-shot direct projection. No trained components."""

    def __init__(self, model, spec):
        self.final_norm, self.unembed = get_final_norm_and_head(model, spec)

    @torch.no_grad()
    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        normed = self.final_norm(state.to(self.final_norm.weight.dtype))
        logits = normed.float() @ self.unembed.weight.float().T
        return logits


class _TunedLensLayer(nn.Module):
    """Identity-initialized residual affine translator for one layer."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.translator = nn.Linear(hidden_size, hidden_size, bias=True)
        nn.init.zeros_(self.translator.weight)
        nn.init.zeros_(self.translator.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.translator(h)


class GammaLensV2:
    """Tuned lens: per-layer learned affine map into the model's final
    norm + unembedding space, trained to minimize KL against final-layer
    logits on held-out text (standard tuned-lens methodology)."""

    def __init__(self, model, spec, num_layers: int, hidden_size: int, device: str):
        self.final_norm, self.unembed = get_final_norm_and_head(model, spec)
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [_TunedLensLayer(hidden_size) for _ in range(num_layers)]
        ).to(device=device, dtype=torch.float32)

    def logits_for_layer(self, l: int, h: torch.Tensor) -> torch.Tensor:
        translated = self.layers[l](h.float())
        normed = self.final_norm(translated.to(self.final_norm.weight.dtype))
        return normed.float() @ self.unembed.weight.float().T

    def parameters(self):
        return self.layers.parameters()

    def save(self, path: str):
        torch.save(self.layers.state_dict(), path)

    def load(self, path: str, map_location=None):
        self.layers.load_state_dict(torch.load(path, map_location=map_location))


def train_tuned_lens(
    lens: GammaLensV2,
    states: torch.Tensor,      # [L, N, H] cached per-layer states (flattened over batch/seq), float
    target_logits: torch.Tensor,  # [N, V] final-layer logits (detached, frozen)
    steps: int = 400,
    lr: float = 1e-3,
    batch_size: int = 512,
    device: str = "cuda",
) -> dict[int, list[float]]:
    """Train each layer's translator independently to minimize KL(final || lens).

    Returns per-layer loss curves for the validation report.
    """
    target_logp = F.log_softmax(target_logits.float(), dim=-1)
    target_p = target_logp.exp()
    n = states.shape[1]
    loss_history: dict[int, list[float]] = {}

    for l in range(lens.num_layers):
        opt = torch.optim.Adam(lens.layers[l].parameters(), lr=lr)
        h_l = states[l].to(device)
        losses = []
        for step in range(steps):
            idx = torch.randint(0, n, (min(batch_size, n),))
            logits = lens.logits_for_layer(l, h_l[idx.to(device)])
            logp = F.log_softmax(logits, dim=-1)
            kl = F.kl_div(logp, target_p[idx].to(device), reduction="batchmean")
            opt.zero_grad()
            kl.backward()
            opt.step()
            losses.append(kl.item())
        loss_history[l] = losses

    return loss_history
