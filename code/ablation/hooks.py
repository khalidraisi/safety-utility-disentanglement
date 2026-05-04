# forward-hook based ablation of a single direction in the residual stream
# usage:
#   handle = install_direction_ablation(model, layer_idx=10, v=torch.tensor(v_np))
#   ... run model.generate(...) ...
#   handle.remove()
#
# the hook fires on every forward pass through the target block, so during
# autoregressive generation it ablates every token (prefill + decode).

import torch


def _get_blocks(model):
    # works for HF Llama/Gemma/Mistral-style models
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise AttributeError("could not find transformer blocks at model.model.layers")


def install_direction_ablation(model, layer_idx, v):
    """
    Project out direction v from the residual stream output of model.model.layers[layer_idx].

    layer_idx uses the SAME indexing as outputs.hidden_states from extraction:
        hidden_states[0] = embedding
        hidden_states[k] = output of model.model.layers[k-1]   for k = 1..L
    So a "best layer = k" picked from V_safety/V_utility (which were built from
    hidden_states) means we hook model.model.layers[k-1].
    """
    if layer_idx < 1:
        raise ValueError("layer_idx must be >= 1 (layer 0 is the embedding)")

    blocks = _get_blocks(model)
    block = blocks[layer_idx - 1]

    # cache v as a unit vector on the model's device/dtype, lazily on first call
    state = {"v": None}
    v_cpu = v.detach().to(torch.float32).flatten()
    v_cpu = v_cpu / (v_cpu.norm() + 1e-12)

    def hook(_module, _inputs, output):
        # block output is either a tensor or a tuple whose first element is the residual
        if isinstance(output, tuple):
            h = output[0]
            rest = output[1:]
        else:
            h = output
            rest = None

        if state["v"] is None:
            state["v"] = v_cpu.to(device=h.device, dtype=h.dtype)
        v_dev = state["v"]

        # h: (B, T, D)   v_dev: (D,)
        # project out: h' = h - (h . v) v
        coeffs = (h * v_dev).sum(dim=-1, keepdim=True)   # (B, T, 1)
        h_new = h - coeffs * v_dev

        if rest is None:
            return h_new
        return (h_new, *rest)

    return block.register_forward_hook(hook)


def install_subspace_ablation(model, layer_idx, V):
    """
    Project out a rank-k subspace V from the residual stream output of
    model.model.layers[layer_idx - 1].

    V: tensor of shape (D, k) -- columns are basis vectors (need not be
       orthonormal; we orthonormalize via QR).
    """
    if layer_idx < 1:
        raise ValueError("layer_idx must be >= 1 (layer 0 is the embedding)")

    blocks = _get_blocks(model)
    block = blocks[layer_idx - 1]

    V32 = V.detach().to(torch.float32)
    if V32.dim() == 1:
        V32 = V32.unsqueeze(-1)
    # orthonormalize columns
    Q, _ = torch.linalg.qr(V32, mode="reduced")  # (D, k)
    state = {"Q": None}

    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            h = output[0]
            rest = output[1:]
        else:
            h = output
            rest = None

        if state["Q"] is None:
            state["Q"] = Q.to(device=h.device, dtype=h.dtype)
        Qd = state["Q"]  # (D, k)

        # h: (B, T, D) -> coeffs (B, T, k) -> projection (B, T, D)
        coeffs = h @ Qd                # (B, T, k)
        proj = coeffs @ Qd.transpose(-1, -2)   # (B, T, D)
        h_new = h - proj

        if rest is None:
            return h_new
        return (h_new, *rest)

    return block.register_forward_hook(hook)
