"""State extraction hooks (protocol section 3.1).

Captures, per layer l and token position t:
  - x_t^(l): residual stream after block l (both architectures)
  - h_t^(l): SSM mixer output, post-selective-scan, pre-residual-add
             (Mamba only; not defined for the transformer control)

Design note on h_t^(l): the protocol names it "per-layer SSM hidden state
h_t^(l) (post-selective-scan)". The literal internal scan state is a
(d_inner x d_state) matrix per token that HF's Mamba implementation never
materializes in full (it's consumed inline within the scan). The
tractable, standard interpretability quantity with that name is the
mixer's output (post-scan, post-gate, post-out_proj) captured immediately
before the residual add -- it lives in hidden_size space, matching the
protocol's own formula in section 3.2 (readout(h_t^(l)) . W_vocab^T)
without requiring a dimensionality-reducing readout for V1. That is what
this module captures.
"""

import torch


class StateExtractor:
    def __init__(self, model, spec):
        self.model = model
        self.spec = spec
        self._h_store: dict[int, torch.Tensor] = {}
        self._x_store: dict[int, torch.Tensor] = {}
        self._handles = []
        self._register()

    def _register(self):
        if self.spec.architecture == "mamba":
            self._register_mamba()
        elif self.spec.architecture == "pythia":
            self._register_pythia()
        else:
            raise ValueError(f"Unsupported architecture: {self.spec.architecture}")

    def _register_mamba(self):
        layers = self.model.backbone.layers

        def make_block_hook(idx):
            def hook(module, inputs, output):
                self._x_store[idx] = output.detach()
            return hook

        def make_mixer_hook(idx):
            def hook(module, inputs, output):
                self._h_store[idx] = output.detach()
            return hook

        for i, block in enumerate(layers):
            self._handles.append(block.register_forward_hook(make_block_hook(i)))
            self._handles.append(block.mixer.register_forward_hook(make_mixer_hook(i)))

    def _register_pythia(self):
        layers = self.model.gpt_neox.layers

        def make_hook(idx):
            def hook(module, inputs, output):
                hs = output[0] if isinstance(output, tuple) else output
                self._x_store[idx] = hs.detach()
            return hook

        for i, layer in enumerate(layers):
            self._handles.append(layer.register_forward_hook(make_hook(i)))

    @torch.no_grad()
    def run(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> dict:
        self._h_store = {}
        self._x_store = {}
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)

        num_layers = len(self._x_store)
        x = torch.stack([self._x_store[i] for i in range(num_layers)], dim=0)  # [L, B, T, H]
        result = {"x": x, "logits": out.logits.detach()}
        if self._h_store:
            h = torch.stack([self._h_store[i] for i in range(num_layers)], dim=0)  # [L, B, T, H]
            result["h"] = h
        return result

    def remove(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()
