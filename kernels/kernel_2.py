import math
from typing import Any
import torch.nn.functional as F
import torch
import os
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()
PROFILE = int(os.getenv("PROFILE", 0)) == 1

# @triton.autotune(
# configs=[
#     triton.Config({'BLOCK_M':64,'BLOCK_N':128,'BLOCK_D':128}, num_warps=4, num_stages=2),
#     triton.Config({'BLOCK_M':128,'BLOCK_N':64,'BLOCK_D':128}, num_warps=4, num_stages=1),
#     triton.Config({'BLOCK_M':64,'BLOCK_N':64,'BLOCK_D':128}, num_warps=4, num_stages=1),
# ],
# key=['S','D']
# )
@triton.jit
def attn_kernel(
    Q, K, V, O,
    M, L,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_mb, stride_mh, stride_ms,
    stride_lb, stride_lh, stride_ls,
    S,
    D,
    softmax_scale,
    N_H: tl.constexpr,
    G: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_bh = tl.program_id(2)

    # -----------------------------
    # GQA head mapping
    # -----------------------------
    batch = pid_bh // N_H
    q_head = pid_bh % N_H
    kv_head = q_head // G

    # -----------------------------
    # offsets
    # -----------------------------
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    mask_m = offs_m < S
    mask_n = offs_n < S
    mask_d = offs_d < D

    # -----------------------------
    # pointers
    # -----------------------------
    q_ptr = Q + batch * stride_qb + q_head * stride_qh
    k_ptr = K + batch * stride_kb + kv_head * stride_kh
    v_ptr = V + batch * stride_vb + kv_head * stride_vh
    o_ptr = O + batch * stride_ob + q_head * stride_oh

    m_ptr = M + batch * stride_mb + q_head * stride_mh
    l_ptr = L + batch * stride_lb + q_head * stride_lh

    # -----------------------------
    # Load Q tile
    # -----------------------------
    q = tl.load(
        q_ptr + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd,
        mask=mask_m[:, None] & mask_d[None, :],
        other=0.0,
    ).to(tl.bfloat16)

    q = (q * softmax_scale).to(tl.bfloat16)

    # -----------------------------
    # Shared memory staging
    # -----------------------------
    k_sram = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.bfloat16)
    v_sram = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.bfloat16)

    # -----------------------------
    # Load KV tile into SRAM
    # -----------------------------
    k_sram = tl.load(
        k_ptr + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd,
        mask=mask_n[:, None] & mask_d[None, :],
        other=0.0,
    ).to(tl.bfloat16)

    v_sram = tl.load(
        v_ptr + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd,
        mask=mask_n[:, None] & mask_d[None, :],
        other=0.0,
    ).to(tl.bfloat16)

    # -----------------------------
    # compute attention scores
    # -----------------------------
    k_sram_t = tl.trans(k_sram)

    scores = tl.dot(q, k_sram_t).to(tl.float32)

    # -----------------------------
    # load softmax state
    # -----------------------------
    m_prev = tl.load(
        m_ptr + offs_m * stride_ms,
        mask=mask_m,
        other=-float("inf"),
    )

    l_prev = tl.load(
        l_ptr + offs_m * stride_ls,
        mask=mask_m,
        other=0.0,
    )

    # -----------------------------
    # softmax update
    # -----------------------------
    m_curr = tl.max(scores, axis=1)

    m_new = tl.maximum(m_prev, m_curr)

    alpha = tl.exp(m_prev - m_new)
    beta = tl.exp(m_curr - m_new)

    p = tl.exp(scores - m_curr[:, None])

    l_curr = tl.sum(p, axis=1)

    l_new = alpha * l_prev + beta * l_curr

    # -----------------------------
    # attention * V
    # -----------------------------
    acc = tl.dot(p.to(tl.float32), v_sram.to(tl.float32))

    # -----------------------------
    # previous output
    # -----------------------------
    o_prev = tl.load(
        o_ptr + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od,
        mask=mask_m[:, None] & mask_d[None, :],
        other=0.0,
    )

    o_new = alpha[:, None] * o_prev + beta[:, None] * acc

    # -----------------------------
    # store output
    # -----------------------------
    tl.store(
        o_ptr + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od,
        o_new,
        mask=mask_m[:, None] & mask_d[None, :],
    )

    # -----------------------------
    # store softmax state
    # -----------------------------
    tl.store(
        m_ptr + offs_m * stride_ms,
        m_new,
        mask=mask_m,
    )

    tl.store(
        l_ptr + offs_m * stride_ls,
        l_new,
        mask=mask_m,
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
        B_c = 128
        D_p = D_h
    else:
        B_r = 64
        B_c = 128
        D_p = D_h
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
    triton.cdiv(S, B_c),
    B * N_h
    )

    #warmup
    for _ in range(2):
        attn_kernel[grid](
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
            num_warps=4,
            num_stages=2
        )
    torch.cuda.synchronize()



    #actual profiling
    attn_kernel[grid](
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
    num_warps=4,
    num_stages=2
    )
    check_tma()

    torch.cuda.synchronize()


if __name__ == "__main__":
    main()