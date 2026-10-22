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


#ಅನ್ತರ್ಭಾವ <sycl/sycl.hpp>
#ಅನ್ತರ್ಭಾವ "dpct/helper.hpp"
#ಅನ್ತರ್ಭಾವ "common.hpp"
#ಅನ್ತರ್ಭಾವ "fattn-common.hpp"
#ಅನ್ತರ್ಭಾವ "fattn-tile.hpp"
#ಅನ್ತರ್ಭಾವ "fattn-vec.hpp"
#ಅನ್ತರ್ಭಾವ "fattn.hpp"
#ಅನ್ತರ್ಭಾವ "fattn-onednn.hpp"
#ಅನ್ತರ್ಭಾವ "fattn-sparse.hpp"

#define FATTN_VEC_CASE(D, type_K, type_V)                                                                        \
    {                                                                                                            \
        ಧ್ರುವ ತಾರ್ಕಿಕ type_K_okay = K->type == (type_K) || (K->type == GGML_TYPE_F32 && (type_K) == GGML_TYPE_F16); \
        ಧ್ರುವ ತಾರ್ಕಿಕ type_V_okay = V->type == (type_V) || (V->type == GGML_TYPE_F32 && (type_V) == GGML_TYPE_F16); \
        ಯದಿ (Q->ne[0] == (D) && type_K_okay && type_V_okay) {                                                     \
            ggml_sycl_flash_attn_ext_vec_case<D, type_K, type_V>(ctx, dst);                                      \
            ಪ್ರತ್ಯಾಗಮ;                                                                                              \
        }                                                                                                        \
    }                                                                    \

#define FATTN_VEC_CASES_ALL_D(type_K, type_V) \
    FATTN_VEC_CASE( 64, type_K, type_V)       \
    FATTN_VEC_CASE(128, type_K, type_V)       \
    FATTN_VEC_CASE(256, type_K, type_V)       \
    FATTN_VEC_CASE(512, type_K, type_V)       \

ಸ್ಥಿರ ಶೂನ್ಯ ggml_sycl_flash_attn_ext_vec(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];

#ifdef GGML_SYCL_FA_ALL_QUANTS
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
#ಅನ್ಯಥಾ
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
#endif // GGML_SYCL_FA_ALL_QUANTS

    GGML_ABORT("Not match KV type in vec");
}

// Best FlashAttention kernel for a specific GPU:
ಗಣನಾ best_fattn_kernel {
    BEST_FATTN_KERNEL_NONE     =   0,
    BEST_FATTN_KERNEL_VEC      = 100,
    BEST_FATTN_KERNEL_ONEDNN   = 150, // oneDNN SDPA: native F16 (PR #25222)
    BEST_FATTN_KERNEL_TILE     = 200,
    BEST_FATTN_KERNEL_MKL      = 300,
};


ಸ್ಥಿರ best_fattn_kernel ggml_sycl_get_best_fattn_kernel(ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ device, ಧ್ರುವ ggml_tensor * dst) {
#ifndef SYCL_FLASH_ATTN
    GGML_UNUSED(dst);
    ಪ್ರತ್ಯಾಗಮ BEST_FATTN_KERNEL_NONE;
#endif// SYCL_FLASH_ATTN

    ಯದಿ(!g_ggml_sycl_enable_flash_attention) ಪ್ರತ್ಯಾಗಮ BEST_FATTN_KERNEL_NONE;

    ಧ್ರುವ ggml_tensor * KQV   = dst;
    ಧ್ರುವ ggml_tensor * Q     = dst->src[0];
    ಧ್ರುವ ggml_tensor * K     = dst->src[1];
    ಧ್ರುವ ggml_tensor * V     = dst->src[2];
    ಧ್ರುವ ggml_tensor * mask  = dst->src[3];
    ಧ್ರುವ ggml_tensor * sinks = dst->src[4];

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ gqa_ratio = Q->ne[2] / K->ne[2];
    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);

    ಭಿನ್ನ max_bias = 0.0f;
    memcpy(&max_bias, (ಧ್ರುವ ಭಿನ್ನ *) KQV->op_params + 1, ಪರಿಮಾಣ(ಭಿನ್ನ));

    ಭಿನ್ನ logit_softcap = 0.0f;
    memcpy(&logit_softcap, (ಧ್ರುವ ಭಿನ್ನ *) KQV->op_params + 2, ಪರಿಮಾಣ(ಭಿನ್ನ));

    ತಾರ್ಕಿಕ gqa_opt_applies = gqa_ratio >= 2 && mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;

    // XMX-accelerated path: oneDNN SDPA (native F16 and dequant+non-F16).
    // ONEDNN requires min 32 query tokens — short-circuit decode to avoid
    // calling _supported() on every decode FA call.
    ಯದಿ (Q->ne[1] >= 32
        && ggml_sycl_flash_attn_ext_onednn_supported(dst)) {
        ಪ್ರತ್ಯಾಗಮ BEST_FATTN_KERNEL_ONEDNN;
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
    ಧ್ರುವ ತಾರ್ಕಿಕ standard_shape = Q->ne[0] >= 64 && Q->ne[0] <= 512 &&
        Q->ne[0] % 64 == 0 && Q->ne[0] == V->ne[0];
    // GLM-4.7 Flash's MLA shape is already expressible by the MKL pipeline:
    // KQ GEMM uses DKQ=576 while VKQ and output use DV=512. Keep this narrow
    // until other mismatched K/V shapes have independent correctness data.
    ಧ್ರುವ ತಾರ್ಕಿಕ glm_mla_shape = Q->ne[0] == 576 && K->ne[0] == 576 &&
        V->ne[0] == 512 && gqa_ratio == 20 &&
        K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16;

    ಯದಿ (g_ggml_sycl_enable_mkl_fa == 1 && mask && !sinks && gqa_ratio >= 2 &&
        (standard_shape || glm_mla_shape) &&
        Q->ne[1] >= 32 && K->ne[1] >= 1024 &&
        max_bias == 0.0f && logit_softcap == 0.0f &&
        (Q->ne[3] == K->ne[3] || K->ne[3] == 1)) {
        // F16 K/V strides must be a multiple of ne[0]*2 (the natural row size
        // in bytes). This passes both dense (nb1 == ne0*2) and interleaved
        // (nb1 == H * ne0*2). Only pathological test strides like nb1=32 or
        // nb1=75 for ne0=40 fall through to TILE.
        ತಾರ್ಕಿಕ kv_strides_ok = ಸತ್ಯ;
        ಪ್ರತಿ (ಧ್ರುವ ggml_tensor * t : {K, V}) {
            ಧ್ರುವ ತಾರ್ಕಿಕ glm_v_stride = glm_mla_shape && t == V &&
                V->view_src && V->nb[1] == K->nb[1];
            ಯದಿ (!glm_v_stride && t->type == GGML_TYPE_F16 &&
                    t->nb[1] % (t->ne[0] * 2) != 0) {
                kv_strides_ok = ಅಸತ್ಯ;
                ಭಙ್ಗ;
            }
        }
        ಯದಿ (kv_strides_ok) {
            ಪ್ರತ್ಯಾಗಮ BEST_FATTN_KERNEL_MKL;
        }
    }
    ಪ್ರತಿ (ಧ್ರುವ ggml_tensor * t : {Q, K, V, mask}) {
        ಯದಿ (t == nullptr || ggml_is_quantized(t->type)) {
            ಅನುವರ್ತ;
        }
        ಪ್ರತಿ (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            ಯದಿ (t->nb[i] % 16 != 0) {
                gqa_opt_applies = ಅಸತ್ಯ;
                ಭಙ್ಗ;
            }
        }
    }

    ವಿಕಲ್ಪ (K->ne[0]) {
        ಪಕ್ಷ  40:
        ಪಕ್ಷ  64:
        ಪಕ್ಷ  72:
        ಪಕ್ಷ  80:
        ಪಕ್ಷ  96:
        ಪಕ್ಷ 128:
        ಪಕ್ಷ 112:
        ಪಕ್ಷ 256:
        ಪಕ್ಷ 512:
            ಯದಿ (V->ne[0] != K->ne[0]) {
                ಪ್ರತ್ಯಾಗಮ BEST_FATTN_KERNEL_NONE;
            }
            ಭಙ್ಗ;
        ಪಕ್ಷ 576:
            ಯದಿ (V->ne[0] != 512) {
                ಪ್ರತ್ಯಾಗಮ BEST_FATTN_KERNEL_NONE;
            }
            ಯದಿ (!gqa_opt_applies) {
                ಪ್ರತ್ಯಾಗಮ BEST_FATTN_KERNEL_NONE;
            }
            ಭಙ್ಗ;
        ಸ್ವಭಾವ:
            ಪ್ರತ್ಯಾಗಮ BEST_FATTN_KERNEL_NONE;
    }

#ifndef GGML_SYCL_FA_ALL_QUANTS
    ಯದಿ (K->type != V->type) {
        ಪ್ರತ್ಯಾಗಮ BEST_FATTN_KERNEL_NONE;
    }
#endif // GGML_SYCL_FA_ALL_QUANTS

    ವಿಕಲ್ಪ (K->type) {
        ಪಕ್ಷ GGML_TYPE_F32:
        ಪಕ್ಷ GGML_TYPE_F16:
        ಪಕ್ಷ GGML_TYPE_BF16:
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_Q4_1:
        ಪಕ್ಷ GGML_TYPE_Q5_0:
        ಪಕ್ಷ GGML_TYPE_Q5_1:
#ifndef GGML_SYCL_FA_ALL_QUANTS
            ಪ್ರತ್ಯಾಗಮ BEST_FATTN_KERNEL_NONE;
#endif // GGML_SYCL_FA_ALL_QUANTS
        ಪಕ್ಷ GGML_TYPE_Q4_0:
        ಪಕ್ಷ GGML_TYPE_Q8_0:
            ಭಙ್ಗ;
        ಸ್ವಭಾವ:
            ಪ್ರತ್ಯಾಗಮ BEST_FATTN_KERNEL_NONE;
    }

    ಯದಿ (mask && mask->ne[2] != 1) {
        ಪ್ರತ್ಯಾಗಮ BEST_FATTN_KERNEL_NONE;
    }

    // For small batch sizes the vector kernel may be preferable over the kernels optimized for large batch sizes.
    // BF16 is excluded: the VEC kernel has no BF16 template (it needs GGML_SYCL_FA_ALL_QUANTS for non-F16/Q4_0/Q8_0).
    ಧ್ರುವ ತಾರ್ಕಿಕ has_bf16 = (K->type == GGML_TYPE_BF16 || V->type == GGML_TYPE_BF16);
    ಧ್ರುವ ತಾರ್ಕಿಕ can_use_vector_kernel = Q->ne[0] <= 512 && Q->ne[0] % 64 == 0 && K->ne[1] % FATTN_KQ_STRIDE == 0
        && !has_bf16;

    // Fused-XMX path: oneDNN Graph SDPA (flash attention). Strictly
    // additive -- taken only when statically supported, otherwise falls through to VEC/TILE below.
    ಯದಿ (ggml_sycl_flash_attn_ext_onednn_supported(dst)) {
        ಪ್ರತ್ಯಾಗಮ BEST_FATTN_KERNEL_ONEDNN;
    }

    // If there are no tensor cores available, use the generic tile kernel:
    ಯದಿ (can_use_vector_kernel) {
        ಯದಿ (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
            ಯದಿ (Q->ne[1] == 1) {
                ಯದಿ (!gqa_opt_applies) {
                    ಪ್ರತ್ಯಾಗಮ BEST_FATTN_KERNEL_VEC;
                }
            }
        } ಅನ್ಯಥಾ {
            ಯದಿ (Q->ne[1] <= 2) {
                // TILE is faster for quantized KV decode on Xe2 (BMG); keep VEC on untested archs
                ಧ್ರುವ gpu_arch arch = ggml_sycl_info().devices[device].hw_info.arch;
                ಯದಿ (arch == gpu_arch::intel_gpu_bmg_g21 || arch == gpu_arch::intel_gpu_bmg_g31) {
                    ಪ್ರತ್ಯಾಗಮ BEST_FATTN_KERNEL_TILE;
                }
                ಪ್ರತ್ಯಾಗಮ BEST_FATTN_KERNEL_VEC;
            }
        }
    }
    ಪ್ರತ್ಯಾಗಮ BEST_FATTN_KERNEL_TILE;
}

ಶೂನ್ಯ ggml_sycl_flash_attn_ext(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_set_device(ctx.device);

    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];
    ಯದಿ (K && Q && K->ne[0] != Q->ne[0] && Q->ne[0] > 0 && K->ne[0] % Q->ne[0] == 0) {
        ಧ್ರುವ int64_t d = Q->ne[0];
        ಧ್ರುವ int64_t n_heads = (K->ne[0] * K->ne[2]) / d;
        K->ne[0] = d;
        K->ne[2] = n_heads;
    }
    ಯದಿ (V && Q && V->ne[0] != Q->ne[0] && Q->ne[0] > 0 && V->ne[0] % Q->ne[0] == 0) {
        ಧ್ರುವ int64_t d = Q->ne[0];
        ಧ್ರುವ int64_t n_heads = (V->ne[0] * V->ne[2]) / d;
        V->ne[0] = d;
        V->ne[2] = n_heads;
    }

    // sparse nodes are gathered down to n_kv_max rows and re-dispatched here
    ಯದಿ (ggml_sycl_flash_attn_ext_sparse(ctx, dst)) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    // n_kv watchdog: log when n_kv differs from the last FA call with
    // the same D — helps detect cache-truncation issues.
    ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ nkv_debug = ggml_sycl_get_env("GGML_SYCL_MKL_FA_DEBUG", 0);
    ಯದಿ (nkv_debug == 1) {
        ಧ್ರುವ ggml_tensor * K_dbg = dst->src[1];
        ಧ್ರುವ ggml_tensor * V_dbg = dst->src[2];
        ಸ್ಥಿರ int64_t last_nkv_d256 = 0, last_nkv_d512 = 0;
        ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ fa_call_seq = 0;
        fa_call_seq++;
        int64_t cur_nkv = K_dbg->ne[1];
        ಪೂರ್ಣಾಙ್ಕ Dk = (ಪೂರ್ಣಾಙ್ಕ)K_dbg->ne[0];
        ಧ್ರುವ ವರ್ಣ * kname = "TILE";
        best_fattn_kernel k = ggml_sycl_get_best_fattn_kernel(ctx.device, dst);
        ಯದಿ (k == BEST_FATTN_KERNEL_MKL)  kname = "MKL";
        ಯದಿ (k == BEST_FATTN_KERNEL_ONEDNN)  kname = "ONEDNN";
        ಯದಿ (k == BEST_FATTN_KERNEL_VEC)  kname = "VEC";
        int64_t delta = 0;
        ಯದಿ (Dk == 256) {
            delta = cur_nkv - last_nkv_d256;
            last_nkv_d256 = cur_nkv;
        } ಅನ್ಯಥಾ ಯದಿ (Dk == 512) {
            delta = cur_nkv - last_nkv_d512;
            last_nkv_d512 = cur_nkv;
        }
        GGML_LOG_INFO("[FA-DISP] #%d %s D=%d n_kv=%lld delta=%lld "
                "V_ne1=%lld\n",
                fa_call_seq, kname, Dk,
                (long long)cur_nkv, (long long)delta,
                (long long)V_dbg->ne[1]);
    }

    ಧ್ರುವ best_fattn_kernel fk = ggml_sycl_get_best_fattn_kernel(ggml_sycl_get_device(), dst);
    ವಿಕಲ್ಪ (fk) {
        ಪಕ್ಷ BEST_FATTN_KERNEL_NONE:
            GGML_ABORT("Not support Flash-Attention");
        ಪಕ್ಷ BEST_FATTN_KERNEL_ONEDNN:
            // guarded: ggml_sycl_flash_attn_ext_onednn() is only defined under GGML_SYCL_DNNL;
            // the reference must be compiled out here or the GGML_SYCL_DNNL=0 build fails to link.
#ಯದಿ GGML_SYCL_DNNL
            ggml_sycl_flash_attn_ext_onednn(ctx, dst);
#endif
            ಭಙ್ಗ;
        ಪಕ್ಷ BEST_FATTN_KERNEL_TILE:
            ggml_sycl_flash_attn_ext_tile(ctx, dst);
            ಭಙ್ಗ;
        ಪಕ್ಷ BEST_FATTN_KERNEL_VEC:
            ggml_sycl_flash_attn_ext_vec(ctx, dst);
            ಭಙ್ಗ;
        ಪಕ್ಷ BEST_FATTN_KERNEL_MKL:
            ggml_sycl_flash_attn_ext_mkl(ctx, dst);
            ಭಙ್ಗ;
    }

    // --- Output fingerprint (GGML_SYCL_MKL_FA_DIAG=1) ---
    // Copy first 64 float output values to host for fingerprinting.
    // Compare MKL vs TILE (GGML_SYCL_ENABLE_MKL_FA=0) to detect divergence.
    // Only fingerprints the first 6 FA calls with n_kv >= 1024.
    ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ fa_diag = ggml_sycl_get_env("GGML_SYCL_MKL_FA_DIAG", 0);
    ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ fa_diag_ಎಣಿಕೆ = 0;
    ಯದಿ (fa_diag == 1 && fa_diag_ಎಣಿಕೆ < 6) {
        ಧ್ರುವ ggml_tensor * K_diag = dst->src[1];
        ಧ್ರುವ ggml_tensor * V_diag = dst->src[2];
        ಧ್ರುವ ggml_tensor * Q_diag = dst->src[0];
        ಯದಿ (K_diag->ne[1] >= 1024) {
            fa_diag_ಎಣಿಕೆ++;
            ಭಿನ್ನ diag_buf[64];
            dpct::queue_ptr q = ctx.stream();
            q->memcpy(diag_buf, dst->data, 64 * ಪರಿಮಾಣ(ಭಿನ್ನ));
            q->wait();
            ಧ್ರುವ ವರ್ಣ * kname = "???";
            best_fattn_kernel kb = ggml_sycl_get_best_fattn_kernel(ctx.device, dst);
            ಯದಿ (kb == BEST_FATTN_KERNEL_ONEDNN) kname = "ONEDNN";
            ಯದಿ (kb == BEST_FATTN_KERNEL_MKL) kname = "MKL";
            ಯದಿ (kb == BEST_FATTN_KERNEL_TILE) kname = "TILE";
            ಯದಿ (kb == BEST_FATTN_KERNEL_VEC) kname = "VEC";
            GGML_LOG_INFO("[FA-DIAG] #%d %s D=%d n_kv=%lld n_q=%lld "
                    "n_qh=%lld n_kvh=%lld K=%s V=%s "
                    "nb1=%zu nb2=%zu first 64 floats:\n",
                    fa_diag_ಎಣಿಕೆ, kname,
                    (ಪೂರ್ಣಾಙ್ಕ)K_diag->ne[0], (long long)K_diag->ne[1],
                    (long long)Q_diag->ne[1],
                    (long long)Q_diag->ne[2], (long long)K_diag->ne[2],
                    ggml_type_name(K_diag->type),
                    ggml_type_name(V_diag->type),
                    K_diag->nb[1], K_diag->nb[2]);
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < 64; i += 8) {
                GGML_LOG_INFO("  [%2d] %08x %08x %08x %08x %08x %08x %08x %08x\n",
                        i,
                        *(unsigned *)&diag_buf[i+0], *(unsigned *)&diag_buf[i+1],
                        *(unsigned *)&diag_buf[i+2], *(unsigned *)&diag_buf[i+3],
                        *(unsigned *)&diag_buf[i+4], *(unsigned *)&diag_buf[i+5],
                        *(unsigned *)&diag_buf[i+6], *(unsigned *)&diag_buf[i+7]);
            }
        }
    }

}

ತಾರ್ಕಿಕ ggml_sycl_flash_attn_ext_supported(ಪೂರ್ಣಾಙ್ಕ device, ಧ್ರುವ ggml_tensor * dst) {
    ಪ್ರತ್ಯಾಗಮ ggml_sycl_get_best_fattn_kernel(device, dst) != BEST_FATTN_KERNEL_NONE;
}

ಸ್ಥಿರ uintptr_t ggml_sycl_fattn_reserve_halves(ggml_sycl_fattn_extra & extra, size_t n_halves) {
    ಯದಿ (n_halves == 0) {
        ಪ್ರತ್ಯಾಗಮ 0;
    }
    extra.end = GGML_PAD(extra.end, SYCL_BUFFER_ALIGNMENT);
    ಧ್ರುವ uintptr_t block = extra.end;
    extra.end += n_halves * ಪರಿಮಾಣ(sycl::half);
    ಪ್ರತ್ಯಾಗಮ block;
}

ggml_sycl_fattn_extra ggml_sycl_fattn_get_extra(ಧ್ರುವ ggml_tensor * dst) {
    ggml_sycl_fattn_extra extra;

    extra.end = (uintptr_t) dst->data + ggml_nbytes(dst);

    ಯದಿ (dst->op != GGML_OP_FLASH_ATTN_EXT) {
        ಪ್ರತ್ಯಾಗಮ extra;
    }

    ಧ್ರುವ ggml_tensor * Q = dst->src[0];
    ಧ್ರುವ ggml_tensor * K = dst->src[1];
    ಧ್ರುವ ggml_tensor * V = dst->src[2];
    ಯದಿ (!Q || !K || !V) {
        ಪ್ರತ್ಯಾಗಮ extra;
    }

    ಧ್ರುವ int64_t d = K->ne[0];
    ಧ್ರುವ int64_t H = Q->ne[2];
    ಧ್ರುವ int64_t q = Q->ne[1];

    // calculate the worst-case memory consumption across all kernels
    ಧ್ರುವ ತಾರ್ಕಿಕ onednn_supported = ggml_sycl_flash_attn_ext_onednn_supported(dst, /* use_shape_limit */ ಅಸತ್ಯ);

    ಧ್ರುವ ತಾರ್ಕಿಕ tile_needs_K = K->type != GGML_TYPE_F16;
    ಧ್ರುವ ತಾರ್ಕಿಕ tile_needs_V = V->type != GGML_TYPE_F16;

    ಧ್ರುವ ತಾರ್ಕಿಕ V_is_K_view = V->view_src &&
        (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));

    size_t need_K = 0, need_V = 0, need_Q = 0, need_out = 0, need_scale = 0;
    ಯದಿ (onednn_supported) {
        need_Q     = (size_t) H * q * d;
        need_out   = (size_t) H * q * d;
        need_scale = 1;
        // an f16 cache is bound in place, so it needs no staging copy
        ಯದಿ (!ggml_sycl_fattn_onednn_binds_kv(K, V)) {
            need_K = (size_t) ggml_nelements(K);
            need_V = (size_t) ggml_nelements(V);
        }
    }
    ಯದಿ (tile_needs_K) {
        need_K = std::max(need_K, (size_t) ggml_nelements(K));
    }
    ಯದಿ (tile_needs_V) {
        need_V = std::max(need_V, (size_t) ggml_nelements(V));
    }

    extra.Q_buffer_ptr = ggml_sycl_fattn_reserve_halves(extra, need_Q);
    extra.K_buffer_ptr = ggml_sycl_fattn_reserve_halves(extra, need_K);
    extra.V_buffer_ptr = (V_is_K_view && !onednn_supported && need_V)
                       ? extra.K_buffer_ptr
                       : ggml_sycl_fattn_reserve_halves(extra, need_V);
    extra.scale_buffer_ptr = ggml_sycl_fattn_reserve_halves(extra, need_scale);
    extra.out_buffer_ptr   = ggml_sycl_fattn_reserve_halves(extra, need_out);

    ಪ್ರತ್ಯಾಗಮ extra;
}

size_t ggml_sycl_flash_attn_ext_get_alloc_size(ಧ್ರುವ ggml_tensor * dst) {
    ಧ್ರುವ ggml_sycl_fattn_extra extra = ggml_sycl_fattn_get_extra(dst);
    ಪ್ರತ್ಯಾಗಮ (size_t) (extra.end - (uintptr_t) dst->data);
}
