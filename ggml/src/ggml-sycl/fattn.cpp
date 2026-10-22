//
// MIT license
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//


#ਅਨ੍ਤਰ੍ਭਾਵ <sycl/sycl.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ "dpct/helper.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "common.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "fattn-common.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "fattn-tile.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "fattn-vec.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "fattn.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "fattn-onednn.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "fattn-sparse.hpp"

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ FATTN_VEC_CASE(D, type_K, type_V)                                                                        \
    {                                                                                                            \
        ਅਚਲ ਤਾਰਕਿਕ type_K_okay = K->type == (type_K) || (K->type == GGML_TYPE_F32 && (type_K) == GGML_TYPE_F16); \
        ਅਚਲ ਤਾਰਕਿਕ type_V_okay = V->type == (type_V) || (V->type == GGML_TYPE_F32 && (type_V) == GGML_TYPE_F16); \
        ਜੇ (Q->ne[0] == (D) && type_K_okay && type_V_okay) {                                                     \
            ggml_sycl_flash_attn_ext_vec_case<D, type_K, type_V>(ctx, dst);                                      \
            ਵਾਪਸ;                                                                                              \
        }                                                                                                        \
    }                                                                    \

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ FATTN_VEC_CASES_ALL_D(type_K, type_V) \
    FATTN_VEC_CASE( 64, type_K, type_V)       \
    FATTN_VEC_CASE(128, type_K, type_V)       \
    FATTN_VEC_CASE(256, type_K, type_V)       \
    FATTN_VEC_CASE(512, type_K, type_V)       \

ਸਥਿਰ ਖਾਲੀ ggml_sycl_flash_attn_ext_vec(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_FA_ALL_QUANTS
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_F16)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q4_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q4_1)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q5_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q5_1)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
#ਨਹੀਂਤਾਂ
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
#ਜੇ_ਅੰਤ // GGML_SYCL_FA_ALL_QUANTS

    GGML_ABORT("Not match KV type in vec");
}

// Best FlashAttention kernel for a specific GPU:
ਗਿਣਤੀ ਉੱਤਮ_fattn_kernel {
    ਉੱਤਮ_FATTN_KERNEL_NONE     =   0,
    ਉੱਤਮ_FATTN_KERNEL_VEC      = 100,
    ਉੱਤਮ_FATTN_KERNEL_ONEDNN   = 150, // oneDNN SDPA: native F16 (PR #25222)
    ਉੱਤਮ_FATTN_KERNEL_TILE     = 200,
    ਉੱਤਮ_FATTN_KERNEL_MKL      = 300,
};


ਸਥਿਰ ਉੱਤਮ_fattn_kernel ggml_sycl_get_ਉੱਤਮ_fattn_kernel(ਅਚਲ ਪੂਰਨ device, ਅਚਲ ggml_tensor * dst) {
#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ SYCL_FLASH_ATTN
    GGML_ਅਣਵਰਤਿਆ(dst);
    ਵਾਪਸ ਉੱਤਮ_FATTN_KERNEL_NONE;
#ਜੇ_ਅੰਤ// SYCL_FLASH_ATTN

    ਜੇ(!g_ggml_sycl_enable_flash_attention) ਵਾਪਸ ਉੱਤਮ_FATTN_KERNEL_NONE;

    ਅਚਲ ggml_tensor * KQV   = dst;
    ਅਚਲ ggml_tensor * Q     = dst->src[0];
    ਅਚਲ ggml_tensor * K     = dst->src[1];
    ਅਚਲ ggml_tensor * V     = dst->src[2];
    ਅਚਲ ggml_tensor * ਨਕਾਬ  = dst->src[3];
    ਅਚਲ ggml_tensor * sinks = dst->src[4];

    ਅਚਲ ਪੂਰਨ gqa_ਅਨੁਪਾਤ = Q->ne[2] / K->ne[2];
    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);

    ਦਸ਼ਮਲਵ max_bias = 0.0f;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&max_bias, (ਅਚਲ ਦਸ਼ਮਲਵ *) KQV->op_params + 1, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਦਸ਼ਮਲਵ logit_softcap = 0.0f;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&logit_softcap, (ਅਚਲ ਦਸ਼ਮਲਵ *) KQV->op_params + 2, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਤਾਰਕਿਕ gqa_opt_applies = gqa_ਅਨੁਪਾਤ >= 2 && ਨਕਾਬ && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;

    // XMX-accelerated path: oneDNN SDPA (native F16 and dequant+non-F16).
    // ONEDNN requires min 32 query tokens — short-circuit decode to avoid
    // calling _supported() on every decode FA call.
    ਜੇ (Q->ne[1] >= 32
        && ggml_sycl_flash_attn_ext_onednn_supported(dst)) {
        ਵਾਪਸ ਉੱਤਮ_FATTN_KERNEL_ONEDNN;
    }

    // MKL path: XMX-accelerated GEMM for prompt processing (all KV cache types).
    // The MKL kernel converts non-F16 K/V to F16 via to_fp16_sycl before GEMM,
    // so quantized, F16, BF16, and F32 caches all benefit from XMX acceleration.
    // Activates automatically when flash-attn is enabled (--flash-attn on or -fa)
    // and n_kv >= 1024. Falls through to TILE/VEC for ALiBi, logit softcap,
    // and mismatched batch dimensions (unsupported by the MKL kernel).
    // Set GGML_SYCL_ENABLE_MKL_FA=0 to force TILE/VEC path for A/B testing.
    // Example: GGML_SYCL_ENABLE_MKL_FA=0 llama-cli -m model.gguf -fa -ngl 99 ...
    // Note: MKL GEMM calls are incompatible with SYCL graph capture replay.
    ਅਚਲ ਤਾਰਕਿਕ standard_shape = Q->ne[0] >= 64 && Q->ne[0] <= 512 &&
        Q->ne[0] % 64 == 0 && Q->ne[0] == V->ne[0];
    // GLM-4.7 Flash's MLA shape is already expressible by the MKL pipeline:
    // KQ GEMM uses DKQ=576 while VKQ and output use DV=512. Keep this narrow
    // until other mismatched K/V shapes have independent correctness data.
    ਅਚਲ ਤਾਰਕਿਕ glm_mla_shape = Q->ne[0] == 576 && K->ne[0] == 576 &&
        V->ne[0] == 512 && gqa_ਅਨੁਪਾਤ == 20 &&
        K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16;

    ਜੇ (g_ggml_sycl_enable_mkl_fa == 1 && ਨਕਾਬ && !sinks && gqa_ਅਨੁਪਾਤ >= 2 &&
        (standard_shape || glm_mla_shape) &&
        Q->ne[1] >= 32 && K->ne[1] >= 1024 &&
        max_bias == 0.0f && logit_softcap == 0.0f &&
        (Q->ne[3] == K->ne[3] || K->ne[3] == 1)) {
        // F16 K/V strides must be a multiple of ne[0]*2 (the natural row size
        // in bytes). This passes both dense (nb1 == ne0*2) and interleaved
        // (nb1 == H * ne0*2). Only pathological test strides like nb1=32 or
        // nb1=75 for ne0=40 fall through to TILE.
        ਤਾਰਕਿਕ kv_strides_ok = ਸੱਚ;
        ਲਈ (ਅਚਲ ggml_tensor * t : {K, V}) {
            ਅਚਲ ਤਾਰਕਿਕ glm_v_stride = glm_mla_shape && t == V &&
                V->view_src && V->nb[1] == K->nb[1];
            ਜੇ (!glm_v_stride && t->type == GGML_TYPE_F16 &&
                    t->nb[1] % (t->ne[0] * 2) != 0) {
                kv_strides_ok = ਝੂਠ;
                ਰੋਕੋ;
            }
        }
        ਜੇ (kv_strides_ok) {
            ਵਾਪਸ ਉੱਤਮ_FATTN_KERNEL_MKL;
        }
    }
    ਲਈ (ਅਚਲ ggml_tensor * t : {Q, K, V, ਨਕਾਬ}) {
        ਜੇ (t == ਖਾਲੀ_ਸੂਚਕ || ggml_is_quantized(t->type)) {
            ਜਾਰੀ;
        }
        ਲਈ (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            ਜੇ (t->nb[i] % 16 != 0) {
                gqa_opt_applies = ਝੂਠ;
                ਰੋਕੋ;
            }
        }
    }

    ਚੋਣ (K->ne[0]) {
        ਹਾਲਤ  40:
        ਹਾਲਤ  64:
        ਹਾਲਤ  72:
        ਹਾਲਤ  80:
        ਹਾਲਤ  96:
        ਹਾਲਤ 128:
        ਹਾਲਤ 112:
        ਹਾਲਤ 256:
        ਹਾਲਤ 512:
            ਜੇ (V->ne[0] != K->ne[0]) {
                ਵਾਪਸ ਉੱਤਮ_FATTN_KERNEL_NONE;
            }
            ਰੋਕੋ;
        ਹਾਲਤ 576:
            ਜੇ (V->ne[0] != 512) {
                ਵਾਪਸ ਉੱਤਮ_FATTN_KERNEL_NONE;
            }
            ਜੇ (!gqa_opt_applies) {
                ਵਾਪਸ ਉੱਤਮ_FATTN_KERNEL_NONE;
            }
            ਰੋਕੋ;
        ਮੂਲ:
            ਵਾਪਸ ਉੱਤਮ_FATTN_KERNEL_NONE;
    }

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_FA_ALL_QUANTS
    ਜੇ (K->type != V->type) {
        ਵਾਪਸ ਉੱਤਮ_FATTN_KERNEL_NONE;
    }
#ਜੇ_ਅੰਤ // GGML_SYCL_FA_ALL_QUANTS

    ਚੋਣ (K->type) {
        ਹਾਲਤ GGML_TYPE_F32:
        ਹਾਲਤ GGML_TYPE_F16:
        ਹਾਲਤ GGML_TYPE_BF16:
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q4_1:
        ਹਾਲਤ GGML_TYPE_Q5_0:
        ਹਾਲਤ GGML_TYPE_Q5_1:
#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_FA_ALL_QUANTS
            ਵਾਪਸ ਉੱਤਮ_FATTN_KERNEL_NONE;
#ਜੇ_ਅੰਤ // GGML_SYCL_FA_ALL_QUANTS
        ਹਾਲਤ GGML_TYPE_Q4_0:
        ਹਾਲਤ GGML_TYPE_Q8_0:
            ਰੋਕੋ;
        ਮੂਲ:
            ਵਾਪਸ ਉੱਤਮ_FATTN_KERNEL_NONE;
    }

    ਜੇ (ਨਕਾਬ && ਨਕਾਬ->ne[2] != 1) {
        ਵਾਪਸ ਉੱਤਮ_FATTN_KERNEL_NONE;
    }

    // For small batch sizes the vector kernel may be preferable over the kernels optimized for large batch sizes.
    // BF16 is excluded: the VEC kernel has no BF16 template (it needs GGML_SYCL_FA_ALL_QUANTS for non-F16/Q4_0/Q8_0).
    ਅਚਲ ਤਾਰਕਿਕ has_bf16 = (K->type == GGML_TYPE_BF16 || V->type == GGML_TYPE_BF16);
    ਅਚਲ ਤਾਰਕਿਕ can_use_ਸਦਿਸ਼_kernel = Q->ne[0] <= 512 && Q->ne[0] % 64 == 0 && K->ne[1] % FATTN_KQ_STRIDE == 0
        && !has_bf16;

    // Fused-XMX path: oneDNN Graph SDPA (flash attention). Strictly
    // additive -- taken only when statically supported, otherwise falls through to VEC/TILE below.
    ਜੇ (ggml_sycl_flash_attn_ext_onednn_supported(dst)) {
        ਵਾਪਸ ਉੱਤਮ_FATTN_KERNEL_ONEDNN;
    }

    // If there are no tensor cores available, use the generic tile kernel:
    ਜੇ (can_use_ਸਦਿਸ਼_kernel) {
        ਜੇ (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
            ਜੇ (Q->ne[1] == 1) {
                ਜੇ (!gqa_opt_applies) {
                    ਵਾਪਸ ਉੱਤਮ_FATTN_KERNEL_VEC;
                }
            }
        } ਨਹੀਂਤਾਂ {
            ਜੇ (Q->ne[1] <= 2) {
                // TILE is faster for quantized KV decode on Xe2 (BMG); keep VEC on untested archs
                ਅਚਲ gpu_arch arch = ggml_sycl_info().devices[device].hw_info.arch;
                ਜੇ (arch == gpu_arch::intel_gpu_bmg_g21 || arch == gpu_arch::intel_gpu_bmg_g31) {
                    ਵਾਪਸ ਉੱਤਮ_FATTN_KERNEL_TILE;
                }
                ਵਾਪਸ ਉੱਤਮ_FATTN_KERNEL_VEC;
            }
        }
    }
    ਵਾਪਸ ਉੱਤਮ_FATTN_KERNEL_TILE;
}

ਖਾਲੀ ggml_sycl_flash_attn_ext(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_ਸੈੱਟ_device(ctx.device);

    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];
    ਜੇ (K && Q && K->ne[0] != Q->ne[0] && Q->ne[0] > 0 && K->ne[0] % Q->ne[0] == 0) {
        ਅਚਲ int64_t d = Q->ne[0];
        ਅਚਲ int64_t n_heads = (K->ne[0] * K->ne[2]) / d;
        K->ne[0] = d;
        K->ne[2] = n_heads;
    }
    ਜੇ (V && Q && V->ne[0] != Q->ne[0] && Q->ne[0] > 0 && V->ne[0] % Q->ne[0] == 0) {
        ਅਚਲ int64_t d = Q->ne[0];
        ਅਚਲ int64_t n_heads = (V->ne[0] * V->ne[2]) / d;
        V->ne[0] = d;
        V->ne[2] = n_heads;
    }

    // sparse nodes are gathered down to n_kv_max rows and re-dispatched here
    ਜੇ (ggml_sycl_flash_attn_ext_sparse(ctx, dst)) {
        ਵਾਪਸ;
    }

    // n_kv watchdog: log when n_kv differs from the last FA call with
    // the same D — helps detect cache-truncation issues.
    ਸਥਿਰ ਪੂਰਨ nkv_debug = ggml_sycl_get_env("GGML_SYCL_MKL_FA_DEBUG", 0);
    ਜੇ (nkv_debug == 1) {
        ਅਚਲ ggml_tensor * K_dbg = dst->src[1];
        ਅਚਲ ggml_tensor * V_dbg = dst->src[2];
        ਸਥਿਰ int64_t ਆਖਰੀ_nkv_d256 = 0, ਆਖਰੀ_nkv_d512 = 0;
        ਸਥਿਰ ਪੂਰਨ fa_call_seq = 0;
        fa_call_seq++;
        int64_t cur_nkv = K_dbg->ne[1];
        ਪੂਰਨ Dk = (ਪੂਰਨ)K_dbg->ne[0];
        ਅਚਲ ਅੱਖਰ * kname = "TILE";
        ਉੱਤਮ_fattn_kernel k = ggml_sycl_get_ਉੱਤਮ_fattn_kernel(ctx.device, dst);
        ਜੇ (k == ਉੱਤਮ_FATTN_KERNEL_MKL)  kname = "MKL";
        ਜੇ (k == ਉੱਤਮ_FATTN_KERNEL_ONEDNN)  kname = "ONEDNN";
        ਜੇ (k == ਉੱਤਮ_FATTN_KERNEL_VEC)  kname = "VEC";
        int64_t delta = 0;
        ਜੇ (Dk == 256) {
            delta = cur_nkv - ਆਖਰੀ_nkv_d256;
            ਆਖਰੀ_nkv_d256 = cur_nkv;
        } ਨਹੀਂਤਾਂ ਜੇ (Dk == 512) {
            delta = cur_nkv - ਆਖਰੀ_nkv_d512;
            ਆਖਰੀ_nkv_d512 = cur_nkv;
        }
        GGML_LOG_INFO("[FA-DISP] #%d %s D=%d n_kv=%lld delta=%lld "
                "V_ne1=%lld\n",
                fa_call_seq, kname, Dk,
                (ਲੰਮਾ ਲੰਮਾ)cur_nkv, (ਲੰਮਾ ਲੰਮਾ)delta,
                (ਲੰਮਾ ਲੰਮਾ)V_dbg->ne[1]);
    }

    ਅਚਲ ਉੱਤਮ_fattn_kernel fk = ggml_sycl_get_ਉੱਤਮ_fattn_kernel(ggml_sycl_get_device(), dst);
    ਚੋਣ (fk) {
        ਹਾਲਤ ਉੱਤਮ_FATTN_KERNEL_NONE:
            GGML_ABORT("Not support Flash-Attention");
        ਹਾਲਤ ਉੱਤਮ_FATTN_KERNEL_ONEDNN:
            // guarded: ggml_sycl_flash_attn_ext_onednn() is only defined under GGML_SYCL_DNNL;
            // the reference must be compiled out here or the GGML_SYCL_DNNL=0 build fails to link.
#ਜੇ GGML_SYCL_DNNL
            ggml_sycl_flash_attn_ext_onednn(ctx, dst);
#ਜੇ_ਅੰਤ
            ਰੋਕੋ;
        ਹਾਲਤ ਉੱਤਮ_FATTN_KERNEL_TILE:
            ggml_sycl_flash_attn_ext_tile(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ ਉੱਤਮ_FATTN_KERNEL_VEC:
            ggml_sycl_flash_attn_ext_vec(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ ਉੱਤਮ_FATTN_KERNEL_MKL:
            ggml_sycl_flash_attn_ext_mkl(ctx, dst);
            ਰੋਕੋ;
    }

    // --- Output fingerprint (GGML_SYCL_MKL_FA_DIAG=1) ---
    // Copy first 64 float output values to host for fingerprinting.
    // Compare MKL vs TILE (GGML_SYCL_ENABLE_MKL_FA=0) to detect divergence.
    // Only fingerprints the first 6 FA calls with n_kv >= 1024.
    ਸਥਿਰ ਪੂਰਨ fa_diag = ggml_sycl_get_env("GGML_SYCL_MKL_FA_DIAG", 0);
    ਸਥਿਰ ਪੂਰਨ fa_diag_ਗਿਣਤੀ = 0;
    ਜੇ (fa_diag == 1 && fa_diag_ਗਿਣਤੀ < 6) {
        ਅਚਲ ggml_tensor * K_diag = dst->src[1];
        ਅਚਲ ggml_tensor * V_diag = dst->src[2];
        ਅਚਲ ggml_tensor * Q_diag = dst->src[0];
        ਜੇ (K_diag->ne[1] >= 1024) {
            fa_diag_ਗਿਣਤੀ++;
            ਦਸ਼ਮਲਵ diag_buf[64];
            dpct::ਕਤਾਰ_ptr q = ctx.stream();
            q->ਸਮ੍ਰਿਤੀ_ਨਕਲ(diag_buf, dst->ਅੰਕੜੇ, 64 * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
            q->wait();
            ਅਚਲ ਅੱਖਰ * kname = "???";
            ਉੱਤਮ_fattn_kernel kb = ggml_sycl_get_ਉੱਤਮ_fattn_kernel(ctx.device, dst);
            ਜੇ (kb == ਉੱਤਮ_FATTN_KERNEL_ONEDNN) kname = "ONEDNN";
            ਜੇ (kb == ਉੱਤਮ_FATTN_KERNEL_MKL) kname = "MKL";
            ਜੇ (kb == ਉੱਤਮ_FATTN_KERNEL_TILE) kname = "TILE";
            ਜੇ (kb == ਉੱਤਮ_FATTN_KERNEL_VEC) kname = "VEC";
            GGML_LOG_INFO("[FA-DIAG] #%d %s D=%d n_kv=%lld n_q=%lld "
                    "n_qh=%lld n_kvh=%lld K=%s V=%s "
                    "nb1=%zu nb2=%zu first 64 floats:\n",
                    fa_diag_ਗਿਣਤੀ, kname,
                    (ਪੂਰਨ)K_diag->ne[0], (ਲੰਮਾ ਲੰਮਾ)K_diag->ne[1],
                    (ਲੰਮਾ ਲੰਮਾ)Q_diag->ne[1],
                    (ਲੰਮਾ ਲੰਮਾ)Q_diag->ne[2], (ਲੰਮਾ ਲੰਮਾ)K_diag->ne[2],
                    ggml_type_name(K_diag->type),
                    ggml_type_name(V_diag->type),
                    K_diag->nb[1], K_diag->nb[2]);
            ਲਈ (ਪੂਰਨ i = 0; i < 64; i += 8) {
                GGML_LOG_INFO("  [%2d] %08x %08x %08x %08x %08x %08x %08x %08x\n",
                        i,
                        *(ਬਿਨਾਨਿਸ਼ਾਨੀ *)&diag_buf[i+0], *(ਬਿਨਾਨਿਸ਼ਾਨੀ *)&diag_buf[i+1],
                        *(ਬਿਨਾਨਿਸ਼ਾਨੀ *)&diag_buf[i+2], *(ਬਿਨਾਨਿਸ਼ਾਨੀ *)&diag_buf[i+3],
                        *(ਬਿਨਾਨਿਸ਼ਾਨੀ *)&diag_buf[i+4], *(ਬਿਨਾਨਿਸ਼ਾਨੀ *)&diag_buf[i+5],
                        *(ਬਿਨਾਨਿਸ਼ਾਨੀ *)&diag_buf[i+6], *(ਬਿਨਾਨਿਸ਼ਾਨੀ *)&diag_buf[i+7]);
            }
        }
    }

}

ਤਾਰਕਿਕ ggml_sycl_flash_attn_ext_supported(ਪੂਰਨ device, ਅਚਲ ggml_tensor * dst) {
    ਵਾਪਸ ggml_sycl_get_ਉੱਤਮ_fattn_kernel(device, dst) != ਉੱਤਮ_FATTN_KERNEL_NONE;
}

ਸਥਿਰ uintptr_t ggml_sycl_fattn_reserve_halves(ggml_sycl_fattn_extra & extra, size_t n_halves) {
    ਜੇ (n_halves == 0) {
        ਵਾਪਸ 0;
    }
    extra.end = GGML_PAD(extra.end, SYCL_ਬਫਰ_ALIGNMENT);
    ਅਚਲ uintptr_t ਖੰਡ = extra.end;
    extra.end += n_halves * ਆਕਾਰ(sycl::half);
    ਵਾਪਸ ਖੰਡ;
}

ggml_sycl_fattn_extra ggml_sycl_fattn_get_extra(ਅਚਲ ggml_tensor * dst) {
    ggml_sycl_fattn_extra extra;

    extra.end = (uintptr_t) dst->ਅੰਕੜੇ + ggml_nbytes(dst);

    ਜੇ (dst->op != GGML_OP_FLASH_ATTN_EXT) {
        ਵਾਪਸ extra;
    }

    ਅਚਲ ggml_tensor * Q = dst->src[0];
    ਅਚਲ ggml_tensor * K = dst->src[1];
    ਅਚਲ ggml_tensor * V = dst->src[2];
    ਜੇ (!Q || !K || !V) {
        ਵਾਪਸ extra;
    }

    ਅਚਲ int64_t d = K->ne[0];
    ਅਚਲ int64_t H = Q->ne[2];
    ਅਚਲ int64_t q = Q->ne[1];

    // calculate the worst-case memory consumption across all kernels
    ਅਚਲ ਤਾਰਕਿਕ onednn_supported = ggml_sycl_flash_attn_ext_onednn_supported(dst, /* use_shape_limit */ ਝੂਠ);

    ਅਚਲ ਤਾਰਕਿਕ tile_needs_K = K->type != GGML_TYPE_F16;
    ਅਚਲ ਤਾਰਕਿਕ tile_needs_V = V->type != GGML_TYPE_F16;

    ਅਚਲ ਤਾਰਕਿਕ V_is_K_view = V->view_src &&
        (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));

    size_t need_K = 0, need_V = 0, need_Q = 0, need_ਬਾਹਰਲਾ = 0, need_scale = 0;
    ਜੇ (onednn_supported) {
        need_Q     = (size_t) H * q * d;
        need_ਬਾਹਰਲਾ   = (size_t) H * q * d;
        need_scale = 1;
        // an f16 cache is bound in place, so it needs no staging copy
        ਜੇ (!ggml_sycl_fattn_onednn_binds_kv(K, V)) {
            need_K = (size_t) ggml_nelements(K);
            need_V = (size_t) ggml_nelements(V);
        }
    }
    ਜੇ (tile_needs_K) {
        need_K = std::max(need_K, (size_t) ggml_nelements(K));
    }
    ਜੇ (tile_needs_V) {
        need_V = std::max(need_V, (size_t) ggml_nelements(V));
    }

    extra.Q_ਬਫਰ_ptr = ggml_sycl_fattn_reserve_halves(extra, need_Q);
    extra.K_ਬਫਰ_ptr = ggml_sycl_fattn_reserve_halves(extra, need_K);
    extra.V_ਬਫਰ_ptr = (V_is_K_view && !onednn_supported && need_V)
                       ? extra.K_ਬਫਰ_ptr
                       : ggml_sycl_fattn_reserve_halves(extra, need_V);
    extra.scale_ਬਫਰ_ptr = ggml_sycl_fattn_reserve_halves(extra, need_scale);
    extra.ਬਾਹਰਲਾ_ਬਫਰ_ptr   = ggml_sycl_fattn_reserve_halves(extra, need_ਬਾਹਰਲਾ);

    ਵਾਪਸ extra;
}

size_t ggml_sycl_flash_attn_ext_get_alloc_ਮਾਪ(ਅਚਲ ggml_tensor * dst) {
    ਅਚਲ ggml_sycl_fattn_extra extra = ggml_sycl_fattn_get_extra(dst);
    ਵਾਪਸ (size_t) (extra.end - (uintptr_t) dst->ਅੰਕੜੇ);
}
