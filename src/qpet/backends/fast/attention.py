"""Attention with a per-key multiplicative gate, fused into a single Triton kernel.

PET's attention is softmax over ``qk/sqrt(d) + log(clamp(c_k, 1e-15))``, where the
bias depends on the key index only -- it is the cutoff factor of that neighbor.

Two properties make a hand-written kernel worthwhile here. The sequence is one
atom's neighbor list, so it is *short*: sixteen to about thirty tokens, against the
thousands that stock attention kernels are tuned for. And the whole row fits in
registers, so the entire attention for one (atom, head) can be done in one pass
with no traffic to global memory for the scores.

The kernel is exact: same arithmetic as
:func:`torch.nn.functional.scaled_dot_product_attention` with the same bias, and
float32 matmuls run in IEEE mode rather than TF32 so it matches the PyTorch path
bit-for-bit in intent. :func:`gated_attention` falls back to SDPA whenever the
kernel does not apply (CPU, float64, unsupported head dimension, no Triton), so
callers never need to check.
"""

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover - exercised only without Triton
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _gated_attention_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        gate_ptr,
        out_ptr,
        scale,
        n_heads,
        n_tokens,
        stride_qn,
        stride_qh,
        stride_qt,
        stride_gn,
        stride_gt,
        BLOCK_T: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        IS_FP32: tl.constexpr,
    ):
        """One program per (atom, head); the whole neighbor row lives in registers."""
        pid = tl.program_id(0)
        node = pid // n_heads
        head = pid % n_heads

        offs_t = tl.arange(0, BLOCK_T)
        offs_d = tl.arange(0, HEAD_DIM)
        live = offs_t < n_tokens

        base = node * stride_qn + head * stride_qh
        offsets = base + offs_t[:, None] * stride_qt + offs_d[None, :]
        q = tl.load(q_ptr + offsets, mask=live[:, None], other=0.0)
        k = tl.load(k_ptr + offsets, mask=live[:, None], other=0.0)
        v = tl.load(v_ptr + offsets, mask=live[:, None], other=0.0)

        # IEEE rather than TF32: TF32 keeps about ten mantissa bits, which this
        # model compounds through six attention layers into ~1e-2 relative error.
        if IS_FP32:
            scores = tl.dot(q, tl.trans(k), input_precision="ieee")
        else:
            scores = tl.dot(q, tl.trans(k))
        scores = scores.to(tl.float32) * scale

        # The gate enters as log(clamp(c_k, 1e-15)), broadcast across queries; keys
        # with a zero gate (padding, or past the cutoff) are excluded outright.
        gate = tl.load(
            gate_ptr + node * stride_gn + offs_t * stride_gt, mask=live, other=0.0
        )
        # Every intermediate is pinned to float32. A bare Python float is a double
        # in Triton, and mixing one into an expression silently promotes the whole
        # chain -- which then fails at the second tl.dot with "both operands must be
        # same dtype". It compiled fine standalone and only broke under
        # torch.compile, so the casts are load-bearing rather than decorative.
        floor = tl.full((), 1e-15, tl.float32)
        neg_inf = tl.full((), float("-inf"), tl.float32)
        gate = gate.to(tl.float32)
        bias = tl.where(gate > 0.0, tl.log(tl.maximum(gate, floor)), neg_inf)
        scores = (scores + bias[None, :]).to(tl.float32)
        # Columns past the end of the row are not padding, they do not exist.
        scores = tl.where(live[None, :], scores, neg_inf).to(tl.float32)

        scores = (scores - tl.max(scores, axis=1)[:, None]).to(tl.float32)
        weights = tl.exp(scores).to(tl.float32)
        weights = (weights / tl.sum(weights, axis=1)[:, None]).to(tl.float32)

        if IS_FP32:
            out = tl.dot(weights, v, input_precision="ieee")
        else:
            out = tl.dot(weights.to(v.dtype), v)
        tl.store(out_ptr + offsets, out.to(v.dtype), mask=live[:, None])


def _next_power_of_two(value: int) -> int:
    return 1 << max(0, (value - 1)).bit_length()


def _kernel_applies(q: torch.Tensor, head_dim: int, n_tokens: int) -> bool:
    if not HAS_TRITON or q.device.type != "cuda":
        return False
    if q.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        return False
    # tl.dot needs both matmul dimensions to be at least 16 and a power of two.
    if head_dim < 16 or head_dim & (head_dim - 1):
        return False
    # Beyond this the register-resident design stops paying: a blocked kernel (or
    # plain SDPA, which is tuned for long sequences) is the right tool.
    return n_tokens <= 128


def gated_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    gate: torch.Tensor,
    scale: float,
    force: str | None = None,
) -> torch.Tensor:
    """Attention weighted by a per-key gate.

    :param queries: ``(n_nodes, n_heads, n_tokens, head_dim)``.
    :param keys: same shape as ``queries``.
    :param values: same shape as ``queries``.
    :param gate: ``(n_nodes, n_tokens)``, the cutoff factor of each key.
    :param scale: multiplier on the dot products, ``1 / sqrt(head_dim)``.
    :param force: ``"triton"`` or ``"sdpa"`` to pin the implementation, for tests
        and benchmarks. ``None`` picks automatically.
    :return: ``(n_nodes, n_heads, n_tokens, head_dim)``.
    """
    n_nodes, n_heads, n_tokens, head_dim = queries.shape

    if force == "triton" and not _kernel_applies(queries, head_dim, n_tokens):
        raise RuntimeError(
            "the Triton kernel does not apply here "
            f"(device={queries.device.type}, dtype={queries.dtype}, "
            f"head_dim={head_dim}, n_tokens={n_tokens})"
        )
    use_triton = force == "triton" or (
        force is None and _kernel_applies(queries, head_dim, n_tokens)
    )

    if not use_triton:
        bias = torch.log(gate.clamp_min(1e-15)).masked_fill(gate <= 0, float("-inf"))
        bias = bias[:, None, None, :]
        return torch.nn.functional.scaled_dot_product_attention(
            queries, keys, values, attn_mask=bias, scale=scale
        )

    queries = queries.contiguous()
    keys = keys.contiguous()
    values = values.contiguous()
    gate = gate.contiguous()
    out = torch.empty_like(queries)

    if n_nodes == 0:
        return out

    _gated_attention_kernel[(n_nodes * n_heads,)](
        queries,
        keys,
        values,
        gate,
        out,
        scale,
        n_heads,
        n_tokens,
        queries.stride(0),
        queries.stride(1),
        queries.stride(2),
        gate.stride(0),
        gate.stride(1),
        BLOCK_T=max(16, _next_power_of_two(n_tokens)),
        HEAD_DIM=head_dim,
        IS_FP32=queries.dtype == torch.float32,
        num_warps=4,
    )
    return out
