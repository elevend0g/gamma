"""State extraction (protocol section 3.1).

Two genuinely different quantities live here, and conflating them was a
Phase 0 documentation bug (see protocol/AMENDMENTS.md, Amendment 2):

  StreamExtractor -- one batched forward pass, standard hooks. Captures:
    - x_t^(l): residual stream after block l (both architectures)
    - mixer_output_t^(l): the Mamba mixer's per-token *output* (post-scan,
      post-gate, post-out_proj, pre-residual-add). This is NOT the
      recurrent state -- it does not persist across t. It's a per-token
      activation in hidden_size space, the closest Mamba analog to a
      transformer's residual stream, and correctly used for depth-axis
      mapping / cross-architecture comparison (protocol section 4.1).

  RecurrentStateExtractor -- step-by-step cached decoding. Captures:
    - h_t^(l): the genuine (d_inner, d_state) SSM recurrent state,
      read from the cache after each single-token step. This is the
      object that persists through time -- what protocol section 4.2
      (time-axis mapping: trajectories, anticipation, the SSM-native
      axis with no transformer analog), section 8.1/8.3 (persistence
      coda), and the interoceptive loop are actually about. It was NOT
      captured anywhere in the original Phase 0 pipeline.

Do not use StreamExtractor's mixer_output where the protocol means h_t.
They answer different questions.
"""

import torch
from transformers.cache_utils import DynamicCache


class StreamExtractor:
    """Batched-forward hooks: residual stream x_t^(l) (both archs) and
    Mamba's per-token mixer output (not the persistent state -- see
    module docstring)."""

    def __init__(self, model, spec):
        self.model = model
        self.spec = spec
        self._mixer_store: dict[int, torch.Tensor] = {}
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
                self._mixer_store[idx] = output.detach()
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
        self._mixer_store = {}
        self._x_store = {}
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)

        num_layers = len(self._x_store)
        x = torch.stack([self._x_store[i] for i in range(num_layers)], dim=0)  # [L, B, T, H]
        result = {"x": x, "logits": out.logits.detach()}
        if self._mixer_store:
            mixer_output = torch.stack([self._mixer_store[i] for i in range(num_layers)], dim=0)  # [L, B, T, H]
            result["mixer_output"] = mixer_output
        return result

    def remove(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()


# Backwards-compatible alias. Phase 0's saved metrics.json files were
# produced under the old name with the old ("h") key; new code should use
# StreamExtractor / "mixer_output" directly.
StateExtractor = StreamExtractor


class RecurrentStateExtractor:
    """Genuine recurrent SSM state h_t^(l), read from the cache after
    each single-token step of cached decoding. Mamba-only: there is no
    transformer analog of this quantity (protocol section 4.2).

    Slow by construction (O(T) forward calls per sequence instead of one
    batched call) -- but the time-axis experiments this feeds are
    sequential by nature anyway.
    """

    def __init__(self, model, spec):
        if spec.architecture != "mamba":
            raise ValueError(
                "Recurrent state trajectories are only defined for Mamba "
                "(no transformer analog; see protocol section 4.2)."
            )
        self.model = model
        self.spec = spec
        self.num_layers = model.config.num_hidden_layers
        self.d_inner = model.config.intermediate_size
        self.d_state = model.config.state_size

    @torch.no_grad()
    def run(self, input_ids: torch.Tensor) -> dict:
        """input_ids: [batch, T]. Returns:
          state:  [T, L, batch, d_inner, d_state] -- genuine recurrent
                  state after processing each token
          logits: [batch, T, vocab] -- the model's own next-token logits
                  at each position (for lens training/eval targets)
        """
        batch, seq_len = input_ids.shape
        cache = DynamicCache(config=self.model.config)
        states, logits_list = [], []

        for t in range(seq_len):
            step_ids = input_ids[:, t : t + 1]
            out = self.model(input_ids=step_ids, cache_params=cache, use_cache=True)
            cache = out.cache_params
            layer_states = torch.stack(
                [cache.layers[l].recurrent_states.detach().clone() for l in range(self.num_layers)],
                dim=0,
            )  # [L, batch, d_inner, d_state]
            states.append(layer_states.cpu())
            logits_list.append(out.logits[:, -1, :].detach().cpu())  # [batch, vocab]

        state = torch.stack(states, dim=0)  # [T, L, batch, d_inner, d_state]
        logits = torch.stack(logits_list, dim=1)  # [batch, T, vocab]
        return {"state": state, "logits": logits}
