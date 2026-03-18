import math
from typing import Any
import torch.nn.functional as F
import torch
import os
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()
PROFILE = int(os.getenv("PROFILE", 0)) == 1

@triton.jit
def attn_fwd_pass1(
    Q, Kt,            # NOTE: Kt = pre-transposed K
    M, L,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_kd, stride_ks,   # swapped
    stride_mb, stride_mh, stride_ms,
    stride_lb, stride_lh, stride_ls,
    S, D,
    softmax_scale,
    N_H: tl.constexpr,
    G: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):

    pid_m  = tl.program_id(0)
    pid_bh = tl.program_id(1)

    batch   = pid_bh // N_H
    q_head  = pid_bh % N_H
    kv_head = q_head // G

    # -----------------------------
    # Offsets
    # -----------------------------
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    mask_m = offs_m < S
    mask_d = offs_d < D

    # -----------------------------
    # Pointers
    # -----------------------------
    q_ptr  = Q  + batch * stride_qb + q_head  * stride_qh
    kt_ptr = Kt + batch * stride_kb + kv_head * stride_kh

    # -----------------------------
    # Load Q (bf16 for MMA)
    # -----------------------------
    q = tl.load(
        q_ptr + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd,
        mask=mask_m[:, None] & mask_d[None, :],
        other=0.0,
    ).to(tl.bfloat16)

    scale = tl.full([], softmax_scale, tl.bfloat16)
    q = q * scale

    # -----------------------------
    # Init stats (fp32)
    # -----------------------------
    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)

    # -----------------------------
    # KV loop
    # -----------------------------
    for start_n in range(0, S, BLOCK_N):

        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S

        # -----------------------------
        # Load K^T (bf16)
        # shape: (BLOCK_D, BLOCK_N)
        # -----------------------------
        k = tl.load(
            kt_ptr + offs_d[:, None] * stride_kd + offs_n[None, :] * stride_ks,
            mask=mask_d[:, None] & mask_n[None, :],
            other=0.0,
        ).to(tl.bfloat16)

        # -----------------------------
        # QK^T (tensor core)
        # -----------------------------
        scores = tl.dot(q, k)
        scores = scores.to(tl.float32)

        # -----------------------------
        # Compute local max
        # -----------------------------
        m_curr = tl.max(scores, axis=1)

        # -----------------------------
        # Compute exp shifted by local max
        # -----------------------------
        scores_shifted = scores - m_curr[:, None]
        p = tl.exp(scores_shifted)

        l_curr = tl.sum(p, axis=1)

        # -----------------------------
        # Merge with running stats
        # -----------------------------
        m_new = tl.maximum(m_i, m_curr)

        alpha = tl.exp(m_i - m_new)
        beta  = tl.exp(m_curr - m_new)

        l_i = alpha * l_i + beta * l_curr
        m_i = m_new

    # -----------------------------
    # Store results
    # -----------------------------
    m_ptr = M + batch * stride_mb + q_head * stride_mh
    l_ptr = L + batch * stride_lb + q_head * stride_lh

    tl.store(m_ptr + offs_m * stride_ms, m_i, mask=mask_m)
    tl.store(l_ptr + offs_m * stride_ls, l_i, mask=mask_m)

@triton.jit
def attn_fwd_pass2(
    Q, Kt, V, O,   # NOTE: Kt = pre-transposed K
    M, L,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_kd, stride_ks,  # NOTE: swapped!
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_mb, stride_mh, stride_ms,
    stride_lb, stride_lh, stride_ls,
    S, D,
    softmax_scale,
    N_H: tl.constexpr,
    G: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):

    pid_m  = tl.program_id(0)
    pid_bh = tl.program_id(1)

    batch   = pid_bh // N_H
    q_head  = pid_bh % N_H
    kv_head = q_head // G

    # -----------------------------
    # Offsets
    # -----------------------------
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    mask_m = offs_m < S
    mask_d = offs_d < D

    # -----------------------------
    # Pointers
    # -----------------------------
    q_ptr  = Q  + batch * stride_qb + q_head  * stride_qh
    kt_ptr = Kt + batch * stride_kb + kv_head * stride_kh
    v_ptr  = V  + batch * stride_vb + kv_head * stride_vh
    o_ptr  = O  + batch * stride_ob + q_head  * stride_oh

    m_ptr = M + batch * stride_mb + q_head * stride_mh
    l_ptr = L + batch * stride_lb + q_head * stride_lh

    # -----------------------------
    # Load Q (bf16 for MMA)
    # -----------------------------
    q = tl.load(
        q_ptr + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd,
        mask=mask_m[:, None] & mask_d[None, :],
        other=0.0,
    ).to(tl.bfloat16)

    scale = tl.full([], softmax_scale, tl.bfloat16)
    q = q * scale

    # -----------------------------
    # Load softmax stats (fp32)
    # -----------------------------
    m_i = tl.load(m_ptr + offs_m * stride_ms, mask=mask_m, other=-float("inf"))
    l_i = tl.load(l_ptr + offs_m * stride_ls, mask=mask_m, other=1.0)

    inv_l = 1.0 / l_i

    # -----------------------------
    # Accumulator (fp32)
    # -----------------------------
    acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)

    # -----------------------------
    # KV loop
    # -----------------------------
    for start_n in range(0, S, BLOCK_N):

        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S

        # -----------------------------
        # Load K^T (already transposed!)
        # shape: (BLOCK_D, BLOCK_N)
        # -----------------------------
        k = tl.load(
            kt_ptr + offs_d[:, None] * stride_kd + offs_n[None, :] * stride_ks,
            mask=mask_d[:, None] & mask_n[None, :],
            other=0.0,
        ).to(tl.bfloat16)

        # -----------------------------
        # Load V
        # shape: (BLOCK_N, BLOCK_D)
        # -----------------------------
        v = tl.load(
            v_ptr + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd,
            mask=mask_n[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)   # keep V in fp32 for stability

        # -----------------------------
        # QK^T (Tensor Core path)
        # -----------------------------
        scores = tl.dot(q, k)   # bf16 × bf16 → MMA
        scores = scores.to(tl.float32)

        # -----------------------------
        # Softmax (stable)
        # -----------------------------
        scores_shifted = scores - m_i[:, None]
        p = tl.exp(scores_shifted)
        p = p * inv_l[:, None]

        # -----------------------------
        # Accumulate (fp32)
        # -----------------------------
        acc += tl.dot(p, v)

    # -----------------------------
    # Store output
    # -----------------------------
    tl.store(
        o_ptr + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od,
        acc,
        mask=mask_m[:, None] & mask_d[None, :],
    )

def compute_sram_need(Br, Bc, D_h, D_p):
    device_properties = torch.cuda.get_device_properties(0)
    q_sram = Br * D_p * 2  
    k_sram = Bc * D_p * 2  
    v_sram = Bc * D_p * 2  
    o_sram = Br * D_p * 2  
    max_sram_size = device_properties.shared_memory_per_block
    print(f"Device Name: {device_properties.name}")
    print(f"Maximum Shared Memory (SRAM) Per Block: {max_sram_size} bytes")
    print(f"Shared Memory needed: {q_sram + k_sram + v_sram + o_sram} bytes")

def compute_register_need(Br, Bc, D_h, D_p):
    q_regs = Br * D_p  
    k_regs = Bc * D_p
    v_regs = Bc * D_p 
    o_regs = Br * D_p 
    m_registers = Br
    l_registers = Br
    total_regs = q_regs + k_regs + v_regs + o_regs + m_registers + l_registers
    print(f"Registers needed per program: {total_regs} registers")
    print(f"Registers needed per thread (assuming 4 warps = 128 threads): {total_regs / 128:.2f} registers/thread")


def check_tma():
    # Print PTX to check for mma instructions
    print("\n=== Checking for Tensor Core (mma) instructions in PTX ===")
    #############################
    # Access the compiled kernel from cache
    import glob
    import os as os_module

    cache_dir = os_module.path.expanduser("~/.triton/cache")
    ptx_files = glob.glob(f"{cache_dir}/**/*.ptx", recursive=True)
    if ptx_files:
        # Get most recent PTX file
        latest_ptx = max(ptx_files, key=os_module.path.getmtime)
        with open(latest_ptx, "r") as f:
            ptx_content = f.read()
        if "mma" in ptx_content:
            print("✓ Found mma instructions - Tensor Cores ARE being used!")
            for line in ptx_content.split("\n"):
                if "mma" in line:
                    print(f"  {line.strip()}")
        else:
            print("✗ No mma instructions found - Tensor Cores NOT being used")
            # Look for what dot product instructions are used
            print("\nLooking for fma/mul instructions:")
            for line in ptx_content.split("\n"):
                if "fma" in line.lower() or (
                    "mul" in line.lower() and "bf16" in line.lower()
                ):
                    print(f"  {line.strip()}")
                    break
    else:
        print("No PTX files found in cache")
    #############################


def main():
    B = 1
    S = 1 << 17 #15,128,16
    N_h = 8
    D_h = 1 << 7
    G = 8  # GQA groups
    Gn = N_h // G

    if PROFILE:
        B_r = 64
        B_c = 64
        D_p = 32
    else:        
        B_r = 64
        B_c = 64
        D_p = 32
    N_d = D_h // D_p
    T_r = S // B_r
    T_c = S // B_c

    q = torch.randn(B, N_h, S, D_h, dtype=torch.bfloat16).cuda()
    v = torch.randn(B, N_h // G, S, D_h, dtype=torch.bfloat16).cuda()
    k = torch.randn(B, N_h // G, S, D_h, dtype=torch.bfloat16).cuda()
    o = torch.zeros_like(q)
    M = torch.empty((B, N_h, S), dtype=torch.bfloat16).cuda()
    L = torch.empty((B, N_h, S), dtype=torch.bfloat16).cuda()

    # Transpose
    k_trans = k.transpose(-1, -2).contiguous()

    compute_sram_need(B_r, B_c, D_h, D_p)
    compute_register_need(B_r, B_c, D_h, D_p)

    print("=== profiling flash attention ===")

    # Grid: (Number of Q blocks, Batch * Heads, Number of Dp blocks)
    grid = (
    triton.cdiv(S, B_r),
    B * N_h
    )


    #warmup
    for _ in range(2):

        attn_fwd_pass1[grid](
            q, k_trans,
            M, L,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            M.stride(0), M.stride(1), M.stride(2),
            L.stride(0), L.stride(1), L.stride(2),
            S,
            D_h,
            1/math.sqrt(D_h),
            N_h,   # number of query heads
            G,    # group size
            B_r,
            B_c,
            D_p,
            num_warps=8,
            num_stages=2
            )
        attn_fwd_pass2[grid](
            q, k_trans, v, o,
            M, L,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            M.stride(0), M.stride(1), M.stride(2),
            L.stride(0), L.stride(1), L.stride(2),
            S,
            D_h,
            1/math.sqrt(D_h),
            N_h,   # number of query heads
            G,    # group size
            B_r,
            B_c,
            D_p,
            num_warps=8,
            num_stages=2
        )
    torch.cuda.synchronize()

    #actual profiling
    attn_fwd_pass1[grid](
            q, k,
            M, L,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            M.stride(0), M.stride(1), M.stride(2),
            L.stride(0), L.stride(1), L.stride(2),
            S,
            D_h,
            1/math.sqrt(D_h),
            N_h,   # number of query heads
            G,    # group size
            B_r,
            B_c,
            D_p,
            num_warps=8,
            num_stages=2
            )
    attn_fwd_pass2[grid](
        q, k, v, o,
        M, L,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        M.stride(0), M.stride(1), M.stride(2),
        L.stride(0), L.stride(1), L.stride(2),
        S,
        D_h,
        1/math.sqrt(D_h),
        N_h,   # number of query heads
        G,    # group size
        B_r,
        B_c,
        D_p,
        num_warps=8,
        num_stages=2
    )
    check_tma()
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()