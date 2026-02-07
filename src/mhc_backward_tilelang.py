"""
MHC Backward Operator - TileLang Implementation

This module implements the backward pass using TileLang/TVM.
"""

import torch
import tvm
from tvm import te


def mhc_backward_tilelang(
    B: int,
    S: int,
    n: int,
    D: int,
    out_dtype="float16",
    compute_dtype="float32",
    norm_eps: float = 1e-6,
    hc_eps: float = 1e-6,
):
    """
    Compile MHC backward operator using TVM TE.

    Args:
        B: Batch size
        S: Sequence length
        n: Number of heads
        D: Head dimension
        out_dtype: Output data type
        compute_dtype: Compute data type
        norm_eps: RMSNorm epsilon
        hc_eps: Hyper connection epsilon

    Returns:
        Compiled TVM function
    """
    nD = n * D
    out_features = n * n + 2 * n

    # Input placeholders
    x = te.placeholder([B, S, n, D], name="x", dtype=out_dtype)
    phi = te.placeholder([out_features, nD], name="phi", dtype=compute_dtype)
    alpha = te.placeholder([3], name="alpha", dtype=compute_dtype)
    bias = te.placeholder([out_features], name="bias", dtype=compute_dtype)
    inv_rms = te.placeholder([B, S], name="inv_rms", dtype=compute_dtype)
    h_mix = te.placeholder([B, S, out_features], name="h_mix", dtype=compute_dtype)
    h_pre = te.placeholder([B, S, n], name="h_pre", dtype=compute_dtype)
    h_post = te.placeholder([B, S, n], name="h_post", dtype=compute_dtype)
    dh_in = te.placeholder([B, S, D], name="dh_in", dtype=out_dtype)
    dh_post = te.placeholder([B, S, n], name="dh_post", dtype=compute_dtype)
    dh_res = te.placeholder([B, S, n, n], name="dh_res", dtype=compute_dtype)
    gamma = te.placeholder([n, D], name="gamma", dtype=compute_dtype)

    # ============================================================
    # Step 1: Compute dh_pre from dh_in
    # dh_pre[b, s, i] = sum_j(dh_in[b, s, j] * x[b, s, i, j])
    # ============================================================
    dh_pre = te.compute(
        [B, S, n],
        lambda b, s, i: te.sum(
            dh_in[b, s, :].astype(compute_dtype) * x[b, s, i, :].astype(compute_dtype)
        ),
        name="dh_pre"
    )

    # ============================================================
    # Step 2: Backward through sigmoid
    # ============================================================
    s_pre2 = te.compute(
        [B, S, n],
        lambda b, s, i: h_pre[b, s, i] - hc_eps,
        name="s_pre2"
    )

    dh_pre2 = te.compute(
        [B, S, n],
        lambda b, s, i: dh_pre[b, s, i] * s_pre2[b, s, i] * (1.0 - s_pre2[b, s, i]),
        name="dh_pre2"
    )

    dh_post2 = te.compute(
        [B, S, n],
        lambda b, s, i: dh_post[b, s, i] * h_post[b, s, i] * (1.0 - h_post[b, s, i] / 2.0),
        name="dh_post2"
    )

    # ============================================================
    # Step 3: Backward through alpha scaling
    # ============================================================
    dh_pre1 = te.compute(
        [B, S, n],
        lambda b, s, i: alpha[0] * dh_pre2[b, s, i],
        name="dh_pre1"
    )

    dh_post1 = te.compute(
        [B, S, n],
        lambda b, s, i: alpha[1] * dh_post2[b, s, i],
        name="dh_post1"
    )

    dh_res1 = te.compute(
        [B, S, n, n],
        lambda b, s, i, j: alpha[2] * dh_res[b, s, i, j],
        name="dh_res1"
    )

    # ============================================================
    # Step 4: Compute dh_mix_tmp and dh_mix
    # ============================================================
    dh_mix_tmp = te.compute(
        [B, S, out_features],
        lambda b, s, k: te.if_then_else(
            k < n,
            dh_pre1[b, s, k],
            te.if_then_else(
                k < 2 * n,
                dh_post1[b, s, k - n],
                dh_res1[b, s, (k - 2 * n) // n, (k - 2 * n) % n]
            )
        ),
        name="dh_mix_tmp"
    )

    dh_mix = te.compute(
        [B, S, out_features],
        lambda b, s, k: dh_mix_tmp[b, s, k] * inv_rms[b, s],
        name="dh_mix"
    )

    # ============================================================
    # Step 5: Compute dvecX_mm = dh_mix @ phi
    # ============================================================
    dvecX_mm = te.compute(
        [B, S, nD],
        lambda b, s, i: te.sum(dh_mix[b, s, :] * phi[:, i]),
        name="dvecX_mm"
    )

    # ============================================================
    # Step 6: Compute dphi = dh_mix^T @ (x * gamma)
    # ============================================================
    x_scaled = te.compute(
        [B, S, n, D],
        lambda b, s, i, j: x[b, s, i, j].astype(compute_dtype) * gamma[i, j],
        name="x_scaled"
    )

    x_scaled_flat = te.compute(
        [B, S, nD],
        lambda b, s, idx: x_scaled[b, s, idx // D, idx % D],
        name="x_scaled_flat"
    )

    dphi = te.compute(
        [out_features, nD],
        lambda k, i: te.sum(
            dh_mix[:, :, k] * x_scaled_flat[:, :, i]
        ),
        name="dphi"
    )

    # ============================================================
    # Step 7: Compute dalpha
    # ============================================================
    h_mix_tmp = te.compute(
        [B, S, out_features],
        lambda b, s, k: h_mix[b, s, k] * inv_rms[b, s],
        name="h_mix_tmp_reuse"
    )

    h_pre1 = te.compute([B, S, n], lambda b, s, i: h_mix_tmp[b, s, i], name="h_pre1_reuse")
    h_post1 = te.compute([B, S, n], lambda b, s, i: h_mix_tmp[b, s, n + i], name="h_post1_reuse")
    h_res1 = te.compute([B, S, n, n], lambda b, s, i, j: h_mix_tmp[b, s, 2 * n + i * n + j], name="h_res1_reuse")

    dalpha_pre = te.reduce(
        lambda b, s: dh_pre2[b, s, 0] * h_pre1[b, s, 0],
        lambda x, y: x + y,
        name="dalpha_pre"
    )
    dalpha_pre = te.compute([],
        lambda: tvm.tir.Sum(dh_pre2[:, :, :] * h_pre1[:, :, :]),
        name="dalpha_pre_final"
    )

    # Use te.sum for simpler reduction
    dalpha_pre_val = te.sum(dh_pre2 * h_pre1)
    dalpha_post_val = te.sum(dh_post2 * h_post1)
    dalpha_res_val = te.sum(dh_res * h_res1)

    dalpha = te.compute([3],
        lambda k: te.if_then_else(
            k == 0,
            dalpha_pre_val,
            te.if_then_else(
                k == 1,
                dalpha_post_val,
                dalpha_res_val
            )
        ),
        name="dalpha"
    )

    # ============================================================
    # Step 8: Compute dbias
    # ============================================================
    dbias_pre = te.compute(
        [n],
        lambda i: te.sum(dh_pre2[:, :, i]),
        name="dbias_pre"
    )

    dbias_post = te.compute(
        [n],
        lambda i: te.sum(dh_post2[:, :, i]),
        name="dbias_post"
    )

    dbias_res = te.compute(
        [n * n],
        lambda idx: te.sum(dh_res[:, :, idx // n, idx % n]),
        name="dbias_res"
    )

    dbias = te.compute(
        [out_features],
        lambda k: te.if_then_else(
            k < n,
            dbias_pre[k],
            te.if_then_else(
                k < 2 * n,
                dbias_post[k - n],
                dbias_res[k - 2 * n]
            )
        ),
        name="dbias"
    )

    # ============================================================
    # Step 9: Compute dinv_rms and dvecX_inv
    # ============================================================
    dinv_rms = te.compute(
        [B, S, 1],
        lambda b, s, _: te.sum(dh_mix_tmp[b, s, :] * h_mix[b, s, :]),
        name="dinv_rms"
    )

    vecX = te.compute(
        [B, S, nD],
        lambda b, s, i: x[b, s, i // D, i % D].astype(compute_dtype),
        name="vecX"
    )

    dvecX_inv = te.compute(
        [B, S, nD],
        lambda b, s, i: -(dinv_rms[b, s, 0] * inv_rms[b, s] ** 3 / nD) * vecX[b, s, i],
        name="dvecX_inv"
    )

    # ============================================================
    # Step 10: Compute dvecX_hin
    # ============================================================
    dvecX_hin = te.compute(
        [B, S, nD],
        lambda b, s, idx: h_pre[b, s, idx // D] * dh_in[b, s, idx % D].astype(compute_dtype),
        name="dvecX_hin"
    )

    # ============================================================
    # Step 11: Compute dx
    # ============================================================
    dx = te.compute(
        [B, S, n, D],
        lambda b, s, i, j: (
            dvecX_mm[b, s, i * D + j] * gamma[i, j] +
            dvecX_inv[b, s, i * D + j] +
            dvecX_hin[b, s, i * D + j]
        ),
        name="dx"
    )

    # ============================================================
    # Step 12: Compute dgamma
    # ============================================================
    dgamma = te.compute(
        [n, D],
        lambda i, j: te.sum(
            vecX[:, :, i * D + j] * dvecX_mm[:, :, i * D + j]
        ),
        name="dgamma"
    )

    # Create schedule
    s = te.create_schedule([dx.op, dphi.op, dalpha.op, dbias.op, dgamma.op])

    # Apply optimizations
    # GPU scheduling
    if tvm.cuda().exist:
        # Parallelize over batch and sequence
        for op in [dh_pre.op, dh_pre2.op, dh_post2.op, dvecX_mm.op]:
            s[op].parallel(b)
            s[op].parallel(s)

        # GPU grid scheduling
        for op in [dx.op, dphi.op]:
            s[op].grid(
                te.thread_axis("blockIdx.x"),
                te.thread_axis("threadIdx.x")
            )

    # Build
    ctx = tvm.gpu(0) if tvm.cuda().exist else tvm.cpu(0)
    func = tvm.build(
        s,
        [x, phi, alpha, bias, inv_rms, h_mix, h_pre, h_post,
         dh_in, dh_post, dh_res, gamma,
         dx, dphi, dalpha, dbias, dgamma],
        target="cuda" if tvm.cuda().exist else "llvm",
        target_host="llvm"
    )

    return func


class MHCBackwardTileLang:
    """
    Wrapper class for MHC Backward operator using TileLang/TVM.
    """

    def __init__(
        self,
        B: int,
        S: int,
        n: int,
        D: int,
        out_dtype: str = "bfloat16",
        compute_dtype: str = "float32",
        norm_eps: float = 1e-6,
        hc_eps: float = 1e-6,
    ):
        """
        Initialize and compile the backward operator.

        Args:
            B: Batch size
            S: Sequence length
            n: Number of heads
            D: Head dimension
            out_dtype: Output data type
            compute_dtype: Computation data type
            norm_eps: RMSNorm epsilon
            hc_eps: Hyper connection epsilon
        """
        self.B = B
        self.S = S
        self.n = n
        self.D = D
        self.nD = n * D
        self.out_features = n * n + 2 * n
        self.out_dtype = out_dtype
        self.compute_dtype = compute_dtype
        self.norm_eps = norm_eps
        self.hc_eps = hc_eps

        # Compile the kernel
        self.func = mhc_backward_tilelang(
            B, S, n, D, out_dtype, compute_dtype, norm_eps, hc_eps
        )

    def __call__(
        self,
        x, phi, alpha, bias,
        inv_rms, h_mix, h_pre, h_post,
        dh_in, dh_post, dh_res, gamma,
    ):
        """
        Backward pass.

        Args:
            x: [B, S, n, D]
            phi: [out_features, nD]
            alpha: [3]
            bias: [out_features]
            inv_rms: [B, S]
            h_mix: [B, S, out_features]
            h_pre: [B, S, n]
            h_post: [B, S, n]
            dh_in: [B, S, D]
            dh_post: [B, S, n]
            dh_res: [B, S, n, n]
            gamma: [n, D]

        Returns:
            dx: [B, S, n, D]
            dphi: [out_features, nD]
            dalpha: [3]
            dbias: [out_features]
            dgamma: [n, D]
        """
        # Allocate outputs
        dx = torch.zeros(self.B, self.S, self.n, self.D, dtype=self.out_dtype, device=x.device)
        dphi = torch.zeros(self.out_features, self.nD, dtype=self.compute_dtype, device=x.device)
        dalpha = torch.zeros(3, dtype=self.compute_dtype, device=x.device)
        dbias = torch.zeros(self.out_features, dtype=self.compute_dtype, device=x.device)
        dgamma = torch.zeros(self.n, self.D, dtype=self.compute_dtype, device=x.device)

        # Run compiled kernel
        self.func(
            x, phi, alpha, bias, inv_rms, h_mix, h_pre, h_post,
            dh_in, dh_post, dh_res, gamma,
            dx, dphi, dalpha, dbias, dgamma,
        )

        return dx, dphi, dalpha, dbias, dgamma
