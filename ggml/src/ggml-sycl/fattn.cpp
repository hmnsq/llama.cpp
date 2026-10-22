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

#ವ್ಯಾಖ್ಯಾನಿಸು FATTN_VEC_CASE(D, type_K, type_V)                                                                        \
    {                                                                                                            \
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ type_K_okay = K->type == (type_K) || (K->type == GGML_TYPE_F32 && (type_K) == GGML_TYPE_F16); \
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ type_V_okay = V->type == (type_V) || (V->type == GGML_TYPE_F32 && (type_V) == GGML_TYPE_F16); \
        ಆದರೆ (Q->ne[0] == (D) && type_K_okay && type_V_okay) {                                                     \
            ggml_sycl_flash_attn_ext_vec_case<D, type_K, type_V>(ctx, dst);                                      \
            ಹಿಂತಿರುಗಿ;                                                                                              \
        }                                                                                                        \
    }                                                                    \

#ವ್ಯಾಖ್ಯಾನಿಸು FATTN_VEC_CASES_ALL_D(type_K, type_V) \
    FATTN_VEC_CASE( 64, type_K, type_V)       \
    FATTN_VEC_CASE(128, type_K, type_V)       \
    FATTN_VEC_CASE(256, type_K, type_V)       \
    FATTN_VEC_CASE(512, type_K, type_V)       \

ಸ್ಥಿರ ಶೂನ್ಯ ggml_sycl_flash_attn_ext_vec(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_FA_ALL_QUANTS
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
#ಇಲ್ಲದಿದ್ದರೆ
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
#ಆದರೆ_ಅಂತ್ಯ // GGML_SYCL_FA_ALL_QUANTS

    GGML_ABORT("Not match KV type in vec");
}

// Best FlashAttention kernel for a specific GPU:
ಎಣಿಕೆ ಉತ್ತಮ_fattn_kernel {
    ಉತ್ತಮ_FATTN_KERNEL_NONE     =   0,
    ಉತ್ತಮ_FATTN_KERNEL_VEC      = 100,
    ಉತ್ತಮ_FATTN_KERNEL_ONEDNN   = 150, // oneDNN SDPA: native F16 (PR #25222)
    ಉತ್ತಮ_FATTN_KERNEL_TILE     = 200,
    ಉತ್ತಮ_FATTN_KERNEL_MKL      = 300,
};


ಸ್ಥಿರ ಉತ್ತಮ_fattn_kernel ggml_sycl_get_ಉತ್ತಮ_fattn_kernel(ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ device, ಸ್ಥಿರಾಂಕ ggml_tensor * dst) {
#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ SYCL_FLASH_ATTN
    GGML_ಬಳಸದ(dst);
    ಹಿಂತಿರುಗಿ ಉತ್ತಮ_FATTN_KERNEL_NONE;
#ಆದರೆ_ಅಂತ್ಯ// SYCL_FLASH_ATTN

    ಆದರೆ(!g_ggml_sycl_enable_flash_attention) ಹಿಂತಿರುಗಿ ಉತ್ತಮ_FATTN_KERNEL_NONE;

    ಸ್ಥಿರಾಂಕ ggml_tensor * KQV   = dst;
    ಸ್ಥಿರಾಂಕ ggml_tensor * Q     = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * K     = dst->src[1];
    ಸ್ಥಿರಾಂಕ ggml_tensor * V     = dst->src[2];
    ಸ್ಥಿರಾಂಕ ggml_tensor * ಮುಖವಾಡ  = dst->src[3];
    ಸ್ಥಿರಾಂಕ ggml_tensor * sinks = dst->src[4];

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ gqa_ಅನುಪಾತ = Q->ne[2] / K->ne[2];
    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);

    ದಶಮಾಂಶ max_bias = 0.0f;
    ಸ್ಮರಣೆ_ನಕಲು(&max_bias, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) KQV->op_params + 1, ಗಾತ್ರ(ದಶಮಾಂಶ));

    ದಶಮಾಂಶ logit_softcap = 0.0f;
    ಸ್ಮರಣೆ_ನಕಲು(&logit_softcap, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) KQV->op_params + 2, ಗಾತ್ರ(ದಶಮಾಂಶ));

    ತಾರ್ಕಿಕ gqa_opt_applies = gqa_ಅನುಪಾತ >= 2 && ಮುಖವಾಡ && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;

    // XMX-accelerated path: oneDNN SDPA (native F16 and dequant+non-F16).
    // ONEDNN requires min 32 query tokens — short-circuit decode to avoid
    // calling _supported() on every decode FA call.
    ಆದರೆ (Q->ne[1] >= 32
        && ggml_sycl_flash_attn_ext_onednn_supported(dst)) {
        ಹಿಂತಿರುಗಿ ಉತ್ತಮ_FATTN_KERNEL_ONEDNN;
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
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ standard_shape = Q->ne[0] >= 64 && Q->ne[0] <= 512 &&
        Q->ne[0] % 64 == 0 && Q->ne[0] == V->ne[0];
    // GLM-4.7 Flash's MLA shape is already expressible by the MKL pipeline:
    // KQ GEMM uses DKQ=576 while VKQ and output use DV=512. Keep this narrow
    // until other mismatched K/V shapes have independent correctness data.
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ glm_mla_shape = Q->ne[0] == 576 && K->ne[0] == 576 &&
        V->ne[0] == 512 && gqa_ಅನುಪಾತ == 20 &&
        K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16;

    ಆದರೆ (g_ggml_sycl_enable_mkl_fa == 1 && ಮುಖವಾಡ && !sinks && gqa_ಅನುಪಾತ >= 2 &&
        (standard_shape || glm_mla_shape) &&
        Q->ne[1] >= 32 && K->ne[1] >= 1024 &&
        max_bias == 0.0f && logit_softcap == 0.0f &&
        (Q->ne[3] == K->ne[3] || K->ne[3] == 1)) {
        // F16 K/V strides must be a multiple of ne[0]*2 (the natural row size
        // in bytes). This passes both dense (nb1 == ne0*2) and interleaved
        // (nb1 == H * ne0*2). Only pathological test strides like nb1=32 or
        // nb1=75 for ne0=40 fall through to TILE.
        ತಾರ್ಕಿಕ kv_strides_ok = ನಿಜ;
        ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ggml_tensor * t : {K, V}) {
            ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ glm_v_stride = glm_mla_shape && t == V &&
                V->view_src && V->nb[1] == K->nb[1];
            ಆದರೆ (!glm_v_stride && t->type == GGML_TYPE_F16 &&
                    t->nb[1] % (t->ne[0] * 2) != 0) {
                kv_strides_ok = ಸುಳ್ಳು;
                ನಿಲ್ಲಿಸು;
            }
        }
        ಆದರೆ (kv_strides_ok) {
            ಹಿಂತಿರುಗಿ ಉತ್ತಮ_FATTN_KERNEL_MKL;
        }
    }
    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ggml_tensor * t : {Q, K, V, ಮುಖವಾಡ}) {
        ಆದರೆ (t == ಶೂನ್ಯಸೂಚಕ || ggml_is_quantized(t->type)) {
            ಮುಂದುವರಿ;
        }
        ಗಾಗಿ (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            ಆದರೆ (t->nb[i] % 16 != 0) {
                gqa_opt_applies = ಸುಳ್ಳು;
                ನಿಲ್ಲಿಸು;
            }
        }
    }

    ಆಯ್ಕೆ (K->ne[0]) {
        ಸ್ಥಿತಿ  40:
        ಸ್ಥಿತಿ  64:
        ಸ್ಥಿತಿ  72:
        ಸ್ಥಿತಿ  80:
        ಸ್ಥಿತಿ  96:
        ಸ್ಥಿತಿ 128:
        ಸ್ಥಿತಿ 112:
        ಸ್ಥಿತಿ 256:
        ಸ್ಥಿತಿ 512:
            ಆದರೆ (V->ne[0] != K->ne[0]) {
                ಹಿಂತಿರುಗಿ ಉತ್ತಮ_FATTN_KERNEL_NONE;
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 576:
            ಆದರೆ (V->ne[0] != 512) {
                ಹಿಂತಿರುಗಿ ಉತ್ತಮ_FATTN_KERNEL_NONE;
            }
            ಆದರೆ (!gqa_opt_applies) {
                ಹಿಂತಿರುಗಿ ಉತ್ತಮ_FATTN_KERNEL_NONE;
            }
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ ಉತ್ತಮ_FATTN_KERNEL_NONE;
    }

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ GGML_SYCL_FA_ALL_QUANTS
    ಆದರೆ (K->type != V->type) {
        ಹಿಂತಿರುಗಿ ಉತ್ತಮ_FATTN_KERNEL_NONE;
    }
#ಆದರೆ_ಅಂತ್ಯ // GGML_SYCL_FA_ALL_QUANTS

    ಆಯ್ಕೆ (K->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
        ಸ್ಥಿತಿ GGML_TYPE_F16:
        ಸ್ಥಿತಿ GGML_TYPE_BF16:
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q4_1:
        ಸ್ಥಿತಿ GGML_TYPE_Q5_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q5_1:
#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ GGML_SYCL_FA_ALL_QUANTS
            ಹಿಂತಿರುಗಿ ಉತ್ತಮ_FATTN_KERNEL_NONE;
#ಆದರೆ_ಅಂತ್ಯ // GGML_SYCL_FA_ALL_QUANTS
        ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ ಉತ್ತಮ_FATTN_KERNEL_NONE;
    }

    ಆದರೆ (ಮುಖವಾಡ && ಮುಖವಾಡ->ne[2] != 1) {
        ಹಿಂತಿರುಗಿ ಉತ್ತಮ_FATTN_KERNEL_NONE;
    }

    // For small batch sizes the vector kernel may be preferable over the kernels optimized for large batch sizes.
    // BF16 is excluded: the VEC kernel has no BF16 template (it needs GGML_SYCL_FA_ALL_QUANTS for non-F16/Q4_0/Q8_0).
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ has_bf16 = (K->type == GGML_TYPE_BF16 || V->type == GGML_TYPE_BF16);
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ can_use_ಸದಿಶ_kernel = Q->ne[0] <= 512 && Q->ne[0] % 64 == 0 && K->ne[1] % FATTN_KQ_STRIDE == 0
        && !has_bf16;

    // Fused-XMX path: oneDNN Graph SDPA (flash attention). Strictly
    // additive -- taken only when statically supported, otherwise falls through to VEC/TILE below.
    ಆದರೆ (ggml_sycl_flash_attn_ext_onednn_supported(dst)) {
        ಹಿಂತಿರುಗಿ ಉತ್ತಮ_FATTN_KERNEL_ONEDNN;
    }

    // If there are no tensor cores available, use the generic tile kernel:
    ಆದರೆ (can_use_ಸದಿಶ_kernel) {
        ಆದರೆ (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
            ಆದರೆ (Q->ne[1] == 1) {
                ಆದರೆ (!gqa_opt_applies) {
                    ಹಿಂತಿರುಗಿ ಉತ್ತಮ_FATTN_KERNEL_VEC;
                }
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಆದರೆ (Q->ne[1] <= 2) {
                // TILE is faster for quantized KV decode on Xe2 (BMG); keep VEC on untested archs
                ಸ್ಥಿರಾಂಕ gpu_arch arch = ggml_sycl_info().devices[device].hw_info.arch;
                ಆದರೆ (arch == gpu_arch::intel_gpu_bmg_g21 || arch == gpu_arch::intel_gpu_bmg_g31) {
                    ಹಿಂತಿರುಗಿ ಉತ್ತಮ_FATTN_KERNEL_TILE;
                }
                ಹಿಂತಿರುಗಿ ಉತ್ತಮ_FATTN_KERNEL_VEC;
            }
        }
    }
    ಹಿಂತಿರುಗಿ ಉತ್ತಮ_FATTN_KERNEL_TILE;
}

ಶೂನ್ಯ ggml_sycl_flash_attn_ext(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_ಗಣ_device(ctx.device);

    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];
    ಆದರೆ (K && Q && K->ne[0] != Q->ne[0] && Q->ne[0] > 0 && K->ne[0] % Q->ne[0] == 0) {
        ಸ್ಥಿರಾಂಕ int64_t d = Q->ne[0];
        ಸ್ಥಿರಾಂಕ int64_t n_heads = (K->ne[0] * K->ne[2]) / d;
        K->ne[0] = d;
        K->ne[2] = n_heads;
    }
    ಆದರೆ (V && Q && V->ne[0] != Q->ne[0] && Q->ne[0] > 0 && V->ne[0] % Q->ne[0] == 0) {
        ಸ್ಥಿರಾಂಕ int64_t d = Q->ne[0];
        ಸ್ಥಿರಾಂಕ int64_t n_heads = (V->ne[0] * V->ne[2]) / d;
        V->ne[0] = d;
        V->ne[2] = n_heads;
    }

    // sparse nodes are gathered down to n_kv_max rows and re-dispatched here
    ಆದರೆ (ggml_sycl_flash_attn_ext_sparse(ctx, dst)) {
        ಹಿಂತಿರುಗಿ;
    }

    // n_kv watchdog: log when n_kv differs from the last FA call with
    // the same D — helps detect cache-truncation issues.
    ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ nkv_debug = ggml_sycl_get_env("GGML_SYCL_MKL_FA_DEBUG", 0);
    ಆದರೆ (nkv_debug == 1) {
        ಸ್ಥಿರಾಂಕ ggml_tensor * K_dbg = dst->src[1];
        ಸ್ಥಿರಾಂಕ ggml_tensor * V_dbg = dst->src[2];
        ಸ್ಥಿರ int64_t ಕೊನೆಯ_nkv_d256 = 0, ಕೊನೆಯ_nkv_d512 = 0;
        ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ fa_call_seq = 0;
        fa_call_seq++;
        int64_t cur_nkv = K_dbg->ne[1];
        ಪೂರ್ಣಾಂಕ Dk = (ಪೂರ್ಣಾಂಕ)K_dbg->ne[0];
        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * kname = "TILE";
        ಉತ್ತಮ_fattn_kernel k = ggml_sycl_get_ಉತ್ತಮ_fattn_kernel(ctx.device, dst);
        ಆದರೆ (k == ಉತ್ತಮ_FATTN_KERNEL_MKL)  kname = "MKL";
        ಆದರೆ (k == ಉತ್ತಮ_FATTN_KERNEL_ONEDNN)  kname = "ONEDNN";
        ಆದರೆ (k == ಉತ್ತಮ_FATTN_KERNEL_VEC)  kname = "VEC";
        int64_t delta = 0;
        ಆದರೆ (Dk == 256) {
            delta = cur_nkv - ಕೊನೆಯ_nkv_d256;
            ಕೊನೆಯ_nkv_d256 = cur_nkv;
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (Dk == 512) {
            delta = cur_nkv - ಕೊನೆಯ_nkv_d512;
            ಕೊನೆಯ_nkv_d512 = cur_nkv;
        }
        GGML_LOG_INFO("[FA-DISP] #%d %s D=%d n_kv=%lld delta=%lld "
                "V_ne1=%lld\n",
                fa_call_seq, kname, Dk,
                (ದೀರ್ಘ ದೀರ್ಘ)cur_nkv, (ದೀರ್ಘ ದೀರ್ಘ)delta,
                (ದೀರ್ಘ ದೀರ್ಘ)V_dbg->ne[1]);
    }

    ಸ್ಥಿರಾಂಕ ಉತ್ತಮ_fattn_kernel fk = ggml_sycl_get_ಉತ್ತಮ_fattn_kernel(ggml_sycl_get_device(), dst);
    ಆಯ್ಕೆ (fk) {
        ಸ್ಥಿತಿ ಉತ್ತಮ_FATTN_KERNEL_NONE:
            GGML_ABORT("Not support Flash-Attention");
        ಸ್ಥಿತಿ ಉತ್ತಮ_FATTN_KERNEL_ONEDNN:
            // guarded: ggml_sycl_flash_attn_ext_onednn() is only defined under GGML_SYCL_DNNL;
            // the reference must be compiled out here or the GGML_SYCL_DNNL=0 build fails to link.
#ಆದರೆ GGML_SYCL_DNNL
            ggml_sycl_flash_attn_ext_onednn(ctx, dst);
#ಆದರೆ_ಅಂತ್ಯ
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ ಉತ್ತಮ_FATTN_KERNEL_TILE:
            ggml_sycl_flash_attn_ext_tile(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ ಉತ್ತಮ_FATTN_KERNEL_VEC:
            ggml_sycl_flash_attn_ext_vec(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ ಉತ್ತಮ_FATTN_KERNEL_MKL:
            ggml_sycl_flash_attn_ext_mkl(ctx, dst);
            ನಿಲ್ಲಿಸು;
    }

    // --- Output fingerprint (GGML_SYCL_MKL_FA_DIAG=1) ---
    // Copy first 64 float output values to host for fingerprinting.
    // Compare MKL vs TILE (GGML_SYCL_ENABLE_MKL_FA=0) to detect divergence.
    // Only fingerprints the first 6 FA calls with n_kv >= 1024.
    ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ fa_diag = ggml_sycl_get_env("GGML_SYCL_MKL_FA_DIAG", 0);
    ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ fa_diag_ಎಣಿಕೆ = 0;
    ಆದರೆ (fa_diag == 1 && fa_diag_ಎಣಿಕೆ < 6) {
        ಸ್ಥಿರಾಂಕ ggml_tensor * K_diag = dst->src[1];
        ಸ್ಥಿರಾಂಕ ggml_tensor * V_diag = dst->src[2];
        ಸ್ಥಿರಾಂಕ ggml_tensor * Q_diag = dst->src[0];
        ಆದರೆ (K_diag->ne[1] >= 1024) {
            fa_diag_ಎಣಿಕೆ++;
            ದಶಮಾಂಶ diag_buf[64];
            dpct::ಸರತಿ_ptr q = ctx.stream();
            q->ಸ್ಮರಣೆ_ನಕಲು(diag_buf, dst->ದತ್ತಾಂಶ, 64 * ಗಾತ್ರ(ದಶಮಾಂಶ));
            q->wait();
            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * kname = "???";
            ಉತ್ತಮ_fattn_kernel kb = ggml_sycl_get_ಉತ್ತಮ_fattn_kernel(ctx.device, dst);
            ಆದರೆ (kb == ಉತ್ತಮ_FATTN_KERNEL_ONEDNN) kname = "ONEDNN";
            ಆದರೆ (kb == ಉತ್ತಮ_FATTN_KERNEL_MKL) kname = "MKL";
            ಆದರೆ (kb == ಉತ್ತಮ_FATTN_KERNEL_TILE) kname = "TILE";
            ಆದರೆ (kb == ಉತ್ತಮ_FATTN_KERNEL_VEC) kname = "VEC";
            GGML_LOG_INFO("[FA-DIAG] #%d %s D=%d n_kv=%lld n_q=%lld "
                    "n_qh=%lld n_kvh=%lld K=%s V=%s "
                    "nb1=%zu nb2=%zu first 64 floats:\n",
                    fa_diag_ಎಣಿಕೆ, kname,
                    (ಪೂರ್ಣಾಂಕ)K_diag->ne[0], (ದೀರ್ಘ ದೀರ್ಘ)K_diag->ne[1],
                    (ದೀರ್ಘ ದೀರ್ಘ)Q_diag->ne[1],
                    (ದೀರ್ಘ ದೀರ್ಘ)Q_diag->ne[2], (ದೀರ್ಘ ದೀರ್ಘ)K_diag->ne[2],
                    ggml_type_name(K_diag->type),
                    ggml_type_name(V_diag->type),
                    K_diag->nb[1], K_diag->nb[2]);
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 64; i += 8) {
                GGML_LOG_INFO("  [%2d] %08x %08x %08x %08x %08x %08x %08x %08x\n",
                        i,
                        *(ಚಿಹ್ನೆರಹಿತ *)&diag_buf[i+0], *(ಚಿಹ್ನೆರಹಿತ *)&diag_buf[i+1],
                        *(ಚಿಹ್ನೆರಹಿತ *)&diag_buf[i+2], *(ಚಿಹ್ನೆರಹಿತ *)&diag_buf[i+3],
                        *(ಚಿಹ್ನೆರಹಿತ *)&diag_buf[i+4], *(ಚಿಹ್ನೆರಹಿತ *)&diag_buf[i+5],
                        *(ಚಿಹ್ನೆರಹಿತ *)&diag_buf[i+6], *(ಚಿಹ್ನೆರಹಿತ *)&diag_buf[i+7]);
            }
        }
    }

}

ತಾರ್ಕಿಕ ggml_sycl_flash_attn_ext_supported(ಪೂರ್ಣಾಂಕ device, ಸ್ಥಿರಾಂಕ ggml_tensor * dst) {
    ಹಿಂತಿರುಗಿ ggml_sycl_get_ಉತ್ತಮ_fattn_kernel(device, dst) != ಉತ್ತಮ_FATTN_KERNEL_NONE;
}

ಸ್ಥಿರ uintptr_t ggml_sycl_fattn_reserve_halves(ggml_sycl_fattn_extra & extra, size_t n_halves) {
    ಆದರೆ (n_halves == 0) {
        ಹಿಂತಿರುಗಿ 0;
    }
    extra.end = GGML_PAD(extra.end, SYCL_ಬಫರ್_ALIGNMENT);
    ಸ್ಥಿರಾಂಕ uintptr_t ಖಂಡ = extra.end;
    extra.end += n_halves * ಗಾತ್ರ(sycl::half);
    ಹಿಂತಿರುಗಿ ಖಂಡ;
}

ggml_sycl_fattn_extra ggml_sycl_fattn_get_extra(ಸ್ಥಿರಾಂಕ ggml_tensor * dst) {
    ggml_sycl_fattn_extra extra;

    extra.end = (uintptr_t) dst->ದತ್ತಾಂಶ + ggml_nbytes(dst);

    ಆದರೆ (dst->op != GGML_OP_FLASH_ATTN_EXT) {
        ಹಿಂತಿರುಗಿ extra;
    }

    ಸ್ಥಿರಾಂಕ ggml_tensor * Q = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * K = dst->src[1];
    ಸ್ಥಿರಾಂಕ ggml_tensor * V = dst->src[2];
    ಆದರೆ (!Q || !K || !V) {
        ಹಿಂತಿರುಗಿ extra;
    }

    ಸ್ಥಿರಾಂಕ int64_t d = K->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t H = Q->ne[2];
    ಸ್ಥಿರಾಂಕ int64_t q = Q->ne[1];

    // calculate the worst-case memory consumption across all kernels
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ onednn_supported = ggml_sycl_flash_attn_ext_onednn_supported(dst, /* use_shape_limit */ ಸುಳ್ಳು);

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ tile_needs_K = K->type != GGML_TYPE_F16;
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ tile_needs_V = V->type != GGML_TYPE_F16;

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ V_is_K_view = V->view_src &&
        (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));

    size_t need_K = 0, need_V = 0, need_Q = 0, need_ಹೊರಹೋಗು = 0, need_scale = 0;
    ಆದರೆ (onednn_supported) {
        need_Q     = (size_t) H * q * d;
        need_ಹೊರಹೋಗು   = (size_t) H * q * d;
        need_scale = 1;
        // an f16 cache is bound in place, so it needs no staging copy
        ಆದರೆ (!ggml_sycl_fattn_onednn_binds_kv(K, V)) {
            need_K = (size_t) ggml_nelements(K);
            need_V = (size_t) ggml_nelements(V);
        }
    }
    ಆದರೆ (tile_needs_K) {
        need_K = std::max(need_K, (size_t) ggml_nelements(K));
    }
    ಆದರೆ (tile_needs_V) {
        need_V = std::max(need_V, (size_t) ggml_nelements(V));
    }

    extra.Q_ಬಫರ್_ptr = ggml_sycl_fattn_reserve_halves(extra, need_Q);
    extra.K_ಬಫರ್_ptr = ggml_sycl_fattn_reserve_halves(extra, need_K);
    extra.V_ಬಫರ್_ptr = (V_is_K_view && !onednn_supported && need_V)
                       ? extra.K_ಬಫರ್_ptr
                       : ggml_sycl_fattn_reserve_halves(extra, need_V);
    extra.scale_ಬಫರ್_ptr = ggml_sycl_fattn_reserve_halves(extra, need_scale);
    extra.ಹೊರಹೋಗು_ಬಫರ್_ptr   = ggml_sycl_fattn_reserve_halves(extra, need_ಹೊರಹೋಗು);

    ಹಿಂತಿರುಗಿ extra;
}

size_t ggml_sycl_flash_attn_ext_get_alloc_ಗಾತ್ರ(ಸ್ಥಿರಾಂಕ ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ggml_sycl_fattn_extra extra = ggml_sycl_fattn_get_extra(dst);
    ಹಿಂತಿರುಗಿ (size_t) (extra.end - (uintptr_t) dst->ದತ್ತಾಂಶ);
}
