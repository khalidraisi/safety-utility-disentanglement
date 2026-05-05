# Forward-hook ablation. Projects out one or more unit directions from the
# residual stream at every transformer block whose index >= start_block_idx.
# Once removed, a direction must not be allowed to re-emerge downstream.

from contextlib import contextmanager

import torch


_BLOCK_PATHS = (
    ("model", "layers"),
    ("model", "language_model", "layers"),
    ("language_model", "model", "layers"),
    ("transformer", "h"),
    ("gpt_neox", "layers"),
)


def _get_blocks(model):
    for path in _BLOCK_PATHS:
        obj = model
        ok = True
        for attr in path:
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok:
            return obj
    import torch.nn as nn
    for name, mod in model.named_modules():
        if name.endswith(".layers") and isinstance(mod, nn.ModuleList):
            return mod
    raise AttributeError("could not find transformer blocks")


def _make_hook(directions_holder):
    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            h = output[0]
            rest = output[1:]
        else:
            h = output
            rest = None

        V = directions_holder.get(h.device, h.dtype)  # (k, D)
        # h: (B, T, D); for each direction v_i: h <- h - (h . v_i) v_i
        # Doing them sequentially handles non-orthogonal directions correctly.
        for i in range(V.shape[0]):
            v = V[i]
            coeffs = (h * v).sum(dim=-1, keepdim=True)
            h = h - coeffs * v

        if rest is None:
            return h
        return (h,) + rest
    return hook


class _DirHolder:
    def __init__(self, directions_np):
        # directions_np: (k, D) numpy or list of (D,) arrays
        import numpy as np
        if isinstance(directions_np, list):
            directions_np = np.stack(directions_np, axis=0)
        V = torch.from_numpy(directions_np).to(torch.float32)
        norms = V.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        self._V_cpu = V / norms
        self._cache = {}

    def get(self, device, dtype):
        key = (device, dtype)
        if key not in self._cache:
            self._cache[key] = self._V_cpu.to(device=device, dtype=dtype)
        return self._cache[key]


@contextmanager
def projection_ablation(model, directions, block_idx):
    """
    Project the given directions out of the residual stream at a single block.

    directions: numpy array (k, D) or list of (D,) arrays. Will be unit-normalized.
    block_idx: 1-indexed against hidden_states (0 = embedding).
               Hook attaches to blocks[block_idx - 1].
    """
    if directions is None or len(directions) == 0:
        yield
        return

    blocks = _get_blocks(model)
    if block_idx < 1 or block_idx > len(blocks):
        raise ValueError(f"block_idx {block_idx} out of range [1, {len(blocks)}]")

    holder = _DirHolder(directions)
    hook_fn = _make_hook(holder)
    handle = blocks[block_idx - 1].register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()
