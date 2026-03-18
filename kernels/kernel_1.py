import math
from typing import Any
import torch.nn.functional as F
import torch
import triton
import os
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()
PROFILE = int(os.getenv("PROFILE", 0)) == 1

@triton.jit
def attn_kernel(
    Q,
    K,
    V,
    O,
    S,
    stride_q_b,
    stride_q_h,
    stride_k_b,
    stride_k_h,
    stride_k_d,
    stride_k_s,
    stride_v_b,
    stride_v_h,
    stride_v_s,
    stride_v_d,
    stride_o_b,
    stride_o_h,
    softmax_scale,
    D: tl.constexpr,
    D_p: tl.constexpr,
    T_c: tl.constexpr,
    B_c: tl.constexpr,
    B_r: tl.constexpr,
    G: tl.constexpr,
    N_h: tl.constexpr,
):

    pid_x = tl.program_id(0)  # Q block
    pid_y = tl.program_id(1)  # batch*head
    pid_d = tl.program_id(2)  # D split

    batch = pid_y // N_h
    head  = pid_y % N_h
    kv_head = head // G

    d_start = pid_d * D_p

    q_ptr = Q + batch * stride_q_b + head * stride_q_h
    k_ptr = K + batch * stride_k_b + kv_head * stride_k_h
    v_ptr = V + batch * stride_v_b + kv_head * stride_v_h
    o_ptr = O + batch * stride_o_b + head * stride_o_h

    # -----------------------------
    # Offsets
    # -----------------------------
    offs_m = pid_x * B_r + tl.arange(0, B_r)
    offs_d = d_start + tl.arange(0, D_p)

    mask_q = offs_m < S
    mask_d = offs_d < D

    # -----------------------------
    # Load Q slice
    # -----------------------------
    qi = tl.load(
        q_ptr + offs_m[:, None] * D + offs_d[None, :],
        mask=mask_q[:, None] & mask_d[None, :],
        other=0.0,
    )

    qi = (qi * softmax_scale).to(tl.bfloat16)

    # -----------------------------
    # Streaming softmax state
    # -----------------------------
    prev_m = tl.full([B_r], -float("inf"), tl.float32)
    prev_l = tl.zeros([B_r], tl.float32)

    acc = tl.zeros([B_r, D_p], tl.float32)

    #------------------------------
    # double buffering for K/V
    #------------------------------

    cols = tl.arange(0, B_c)
    mask_kv = cols < S

    kj = tl.load(
            k_ptr +
            offs_d[:, None] * stride_k_d +
            cols[None, :] * stride_k_s,
            mask=mask_d[:, None] & mask_kv[None, :],
            other=0.0,
        ).to(tl.bfloat16)
    
    vj = tl.load(
            v_ptr +
            cols[:, None] * stride_v_s +
            offs_d[None, :] * stride_v_d,
            mask=mask_kv[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.bfloat16)

    # -----------------------------
    # Loop over K/V blocks
    # -----------------------------
    for j in range(1, T_c):

        cols = j * B_c + tl.arange(0, B_c)
        mask_kv = cols < S

        # Load K slice (D_p, B_c)
        kj_next = tl.load(
            k_ptr +
            offs_d[:, None] * stride_k_d +
            cols[None, :] * stride_k_s,
            mask=mask_d[:, None] & mask_kv[None, :],
            other=0.0,
        ).to(tl.bfloat16)

        # Attention scores
        Sij = tl.dot(qi, kj)  # (B_r, B_c)

        # Load V slice (B_c, D_p)
        vj_next = tl.load(
            v_ptr +
            cols[:, None] * stride_v_s +
            offs_d[None, :] * stride_v_d,
            mask=mask_kv[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.bfloat16)

        # ---- Streaming softmax ----
        mij = tl.max(Sij, axis=1)
        pij = tl.exp(Sij - mij[:, None])
        lij = tl.sum(pij, axis=1)

        m_new = tl.maximum(prev_m, mij)
        alpha = tl.exp(prev_m - m_new)
        beta  = tl.exp(mij - m_new)

        acc = alpha[:, None] * acc + beta[:, None] * tl.dot(pij, vj.to(tl.float32))

        prev_l = prev_l * alpha + lij * beta
        prev_m = m_new

        kj = kj_next
        vj = vj_next

    # -----------------------------
    # Final normalization
    # -----------------------------
    acc = acc / prev_l[:, None]

    tl.store(
        o_ptr + offs_m[:, None] * D + offs_d[None, :],
        acc.to(tl.bfloat16),
        mask=mask_q[:, None] & mask_d[None, :],
    )


def simple_attn(q, k, v):
    # Reference needs float for precision comparison
    att = q.float() @ k.float().transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))
    att = F.softmax(att, dim=-1)
    y = att @ v.float()
    return y.to(q.dtype)


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

    if PROFILE:
        B_r = 128
        B_c = 32
        D_p = 32
    else:
        B_r = 128
        B_c = 32
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

    grid1 = (triton.cdiv(S, B_r), B * N_h, N_d)

    #warmup
    for _ in range(2):
        attn_kernel[grid1](
        q,
        k_trans,
        v,
        o,
        S,
        q.stride(0),
        q.stride(1),
        k_trans.stride(0),
        k_trans.stride(1),
        k_trans.stride(2),
        k_trans.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        o.stride(0),
        o.stride(1),
        1 / math.sqrt(D_h),
        D_h,
        D_p,
        T_c,
        B_c,
        B_r,
        G,
        N_h,
        num_warps=4,  #Critical for Tensor Core parallelization
        num_stages=2, #Double buffering for K/V, plus one for output
    )
    torch.cuda.synchronize()



    #actual profiling
    attn_kernel[grid1](
        q,
        k_trans,
        v,
        o,
        S,
        q.stride(0),
        q.stride(1),
        k_trans.stride(0),
        k_trans.stride(1),
        k_trans.stride(2),
        k_trans.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        o.stride(0),
        o.stride(1),
        1 / math.sqrt(D_h),
        D_h,
        D_p,
        T_c,
        B_c,
        B_r,
        G,
        N_h,
        num_warps=4,  #Critical for Tensor Core parallelization
        num_stages=2, #Double buffering for K/V, plus one for output
    )
    check_tma()

    torch.cuda.synchronize()


if __name__ == "__main__":
    main()