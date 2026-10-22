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


#అన్తర్భావ <sycl/sycl.hpp>
#అన్తర్భావ "dpct/helper.hpp"
#అన్తర్భావ "common.hpp"
#అన్తర్భావ "fattn-common.hpp"
#అన్తర్భావ "fattn-tile.hpp"
#అన్తర్భావ "fattn-vec.hpp"
#అన్తర్భావ "fattn.hpp"
#అన్తర్భావ "fattn-onednn.hpp"
#అన్తర్భావ "fattn-sparse.hpp"

#నిర్వచించు FATTN_VEC_CASE(D, type_K, type_V)                                                                        \
    {                                                                                                            \
        స్థిరరాశి తార్కిక type_K_okay = K->type == (type_K) || (K->type == GGML_TYPE_F32 && (type_K) == GGML_TYPE_F16); \
        స్థిరరాశి తార్కిక type_V_okay = V->type == (type_V) || (V->type == GGML_TYPE_F32 && (type_V) == GGML_TYPE_F16); \
        ఒకవేళ (Q->ne[0] == (D) && type_K_okay && type_V_okay) {                                                     \
            ggml_sycl_flash_attn_ext_vec_case<D, type_K, type_V>(ctx, dst);                                      \
            తిరిగి;                                                                                              \
        }                                                                                                        \
    }                                                                    \

#నిర్వచించు FATTN_VEC_CASES_ALL_D(type_K, type_V) \
    FATTN_VEC_CASE( 64, type_K, type_V)       \
    FATTN_VEC_CASE(128, type_K, type_V)       \
    FATTN_VEC_CASE(256, type_K, type_V)       \
    FATTN_VEC_CASE(512, type_K, type_V)       \

స్థిర శూన్య ggml_sycl_flash_attn_ext_vec(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];

#నిర్వచితమైతే GGML_SYCL_FA_ALL_QUANTS
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
#లేకపోతే
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
#అయితే_ముగింపు // GGML_SYCL_FA_ALL_QUANTS

    GGML_ABORT("Not match KV type in vec");
}

// Best FlashAttention kernel for a specific GPU:
గణన ఉత్తమ_fattn_kernel {
    ఉత్తమ_FATTN_KERNEL_NONE     =   0,
    ఉత్తమ_FATTN_KERNEL_VEC      = 100,
    ఉత్తమ_FATTN_KERNEL_ONEDNN   = 150, // oneDNN SDPA: native F16 (PR #25222)
    ఉత్తమ_FATTN_KERNEL_TILE     = 200,
    ఉత్తమ_FATTN_KERNEL_MKL      = 300,
};


స్థిర ఉత్తమ_fattn_kernel ggml_sycl_get_ఉత్తమ_fattn_kernel(స్థిరరాశి పూర్ణాంకం device, స్థిరరాశి ggml_tensor * dst) {
#నిర్వచితం_కాకపోతే SYCL_FLASH_ATTN
    GGML_వాడని(dst);
    తిరిగి ఉత్తమ_FATTN_KERNEL_NONE;
#అయితే_ముగింపు// SYCL_FLASH_ATTN

    ఒకవేళ(!g_ggml_sycl_enable_flash_attention) తిరిగి ఉత్తమ_FATTN_KERNEL_NONE;

    స్థిరరాశి ggml_tensor * KQV   = dst;
    స్థిరరాశి ggml_tensor * Q     = dst->src[0];
    స్థిరరాశి ggml_tensor * K     = dst->src[1];
    స్థిరరాశి ggml_tensor * V     = dst->src[2];
    స్థిరరాశి ggml_tensor * ముసుగు  = dst->src[3];
    స్థిరరాశి ggml_tensor * sinks = dst->src[4];

    స్థిరరాశి పూర్ణాంకం gqa_నిష్పత్తి = Q->ne[2] / K->ne[2];
    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);

    దశాంశం max_bias = 0.0f;
    జ్ఞాపకం_నకలు(&max_bias, (స్థిరరాశి దశాంశం *) KQV->op_params + 1, పరిమాణం(దశాంశం));

    దశాంశం logit_softcap = 0.0f;
    జ్ఞాపకం_నకలు(&logit_softcap, (స్థిరరాశి దశాంశం *) KQV->op_params + 2, పరిమాణం(దశాంశం));

    తార్కిక gqa_opt_applies = gqa_నిష్పత్తి >= 2 && ముసుగు && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;

    // XMX-accelerated path: oneDNN SDPA (native F16 and dequant+non-F16).
    // ONEDNN requires min 32 query tokens — short-circuit decode to avoid
    // calling _supported() on every decode FA call.
    ఒకవేళ (Q->ne[1] >= 32
        && ggml_sycl_flash_attn_ext_onednn_supported(dst)) {
        తిరిగి ఉత్తమ_FATTN_KERNEL_ONEDNN;
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
    స్థిరరాశి తార్కిక standard_shape = Q->ne[0] >= 64 && Q->ne[0] <= 512 &&
        Q->ne[0] % 64 == 0 && Q->ne[0] == V->ne[0];
    // GLM-4.7 Flash's MLA shape is already expressible by the MKL pipeline:
    // KQ GEMM uses DKQ=576 while VKQ and output use DV=512. Keep this narrow
    // until other mismatched K/V shapes have independent correctness data.
    స్థిరరాశి తార్కిక glm_mla_shape = Q->ne[0] == 576 && K->ne[0] == 576 &&
        V->ne[0] == 512 && gqa_నిష్పత్తి == 20 &&
        K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16;

    ఒకవేళ (g_ggml_sycl_enable_mkl_fa == 1 && ముసుగు && !sinks && gqa_నిష్పత్తి >= 2 &&
        (standard_shape || glm_mla_shape) &&
        Q->ne[1] >= 32 && K->ne[1] >= 1024 &&
        max_bias == 0.0f && logit_softcap == 0.0f &&
        (Q->ne[3] == K->ne[3] || K->ne[3] == 1)) {
        // F16 K/V strides must be a multiple of ne[0]*2 (the natural row size
        // in bytes). This passes both dense (nb1 == ne0*2) and interleaved
        // (nb1 == H * ne0*2). Only pathological test strides like nb1=32 or
        // nb1=75 for ne0=40 fall through to TILE.
        తార్కిక kv_strides_ok = నిజం;
        కొరకు (స్థిరరాశి ggml_tensor * t : {K, V}) {
            స్థిరరాశి తార్కిక glm_v_stride = glm_mla_shape && t == V &&
                V->view_src && V->nb[1] == K->nb[1];
            ఒకవేళ (!glm_v_stride && t->type == GGML_TYPE_F16 &&
                    t->nb[1] % (t->ne[0] * 2) != 0) {
                kv_strides_ok = అబద్ధం;
                ఆపు;
            }
        }
        ఒకవేళ (kv_strides_ok) {
            తిరిగి ఉత్తమ_FATTN_KERNEL_MKL;
        }
    }
    కొరకు (స్థిరరాశి ggml_tensor * t : {Q, K, V, ముసుగు}) {
        ఒకవేళ (t == శూన్యసూచిక || ggml_is_quantized(t->type)) {
            కొనసాగు;
        }
        కొరకు (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            ఒకవేళ (t->nb[i] % 16 != 0) {
                gqa_opt_applies = అబద్ధం;
                ఆపు;
            }
        }
    }

    ఎంపిక (K->ne[0]) {
        స్థితి  40:
        స్థితి  64:
        స్థితి  72:
        స్థితి  80:
        స్థితి  96:
        స్థితి 128:
        స్థితి 112:
        స్థితి 256:
        స్థితి 512:
            ఒకవేళ (V->ne[0] != K->ne[0]) {
                తిరిగి ఉత్తమ_FATTN_KERNEL_NONE;
            }
            ఆపు;
        స్థితి 576:
            ఒకవేళ (V->ne[0] != 512) {
                తిరిగి ఉత్తమ_FATTN_KERNEL_NONE;
            }
            ఒకవేళ (!gqa_opt_applies) {
                తిరిగి ఉత్తమ_FATTN_KERNEL_NONE;
            }
            ఆపు;
        అప్రమేయ:
            తిరిగి ఉత్తమ_FATTN_KERNEL_NONE;
    }

#నిర్వచితం_కాకపోతే GGML_SYCL_FA_ALL_QUANTS
    ఒకవేళ (K->type != V->type) {
        తిరిగి ఉత్తమ_FATTN_KERNEL_NONE;
    }
#అయితే_ముగింపు // GGML_SYCL_FA_ALL_QUANTS

    ఎంపిక (K->type) {
        స్థితి GGML_TYPE_F32:
        స్థితి GGML_TYPE_F16:
        స్థితి GGML_TYPE_BF16:
            ఆపు;
        స్థితి GGML_TYPE_Q4_1:
        స్థితి GGML_TYPE_Q5_0:
        స్థితి GGML_TYPE_Q5_1:
#నిర్వచితం_కాకపోతే GGML_SYCL_FA_ALL_QUANTS
            తిరిగి ఉత్తమ_FATTN_KERNEL_NONE;
#అయితే_ముగింపు // GGML_SYCL_FA_ALL_QUANTS
        స్థితి GGML_TYPE_Q4_0:
        స్థితి GGML_TYPE_Q8_0:
            ఆపు;
        అప్రమేయ:
            తిరిగి ఉత్తమ_FATTN_KERNEL_NONE;
    }

    ఒకవేళ (ముసుగు && ముసుగు->ne[2] != 1) {
        తిరిగి ఉత్తమ_FATTN_KERNEL_NONE;
    }

    // For small batch sizes the vector kernel may be preferable over the kernels optimized for large batch sizes.
    // BF16 is excluded: the VEC kernel has no BF16 template (it needs GGML_SYCL_FA_ALL_QUANTS for non-F16/Q4_0/Q8_0).
    స్థిరరాశి తార్కిక has_bf16 = (K->type == GGML_TYPE_BF16 || V->type == GGML_TYPE_BF16);
    స్థిరరాశి తార్కిక can_use_సదిశ_kernel = Q->ne[0] <= 512 && Q->ne[0] % 64 == 0 && K->ne[1] % FATTN_KQ_STRIDE == 0
        && !has_bf16;

    // Fused-XMX path: oneDNN Graph SDPA (flash attention). Strictly
    // additive -- taken only when statically supported, otherwise falls through to VEC/TILE below.
    ఒకవేళ (ggml_sycl_flash_attn_ext_onednn_supported(dst)) {
        తిరిగి ఉత్తమ_FATTN_KERNEL_ONEDNN;
    }

    // If there are no tensor cores available, use the generic tile kernel:
    ఒకవేళ (can_use_సదిశ_kernel) {
        ఒకవేళ (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
            ఒకవేళ (Q->ne[1] == 1) {
                ఒకవేళ (!gqa_opt_applies) {
                    తిరిగి ఉత్తమ_FATTN_KERNEL_VEC;
                }
            }
        } లేకపోతే {
            ఒకవేళ (Q->ne[1] <= 2) {
                // TILE is faster for quantized KV decode on Xe2 (BMG); keep VEC on untested archs
                స్థిరరాశి gpu_arch arch = ggml_sycl_info().devices[device].hw_info.arch;
                ఒకవేళ (arch == gpu_arch::intel_gpu_bmg_g21 || arch == gpu_arch::intel_gpu_bmg_g31) {
                    తిరిగి ఉత్తమ_FATTN_KERNEL_TILE;
                }
                తిరిగి ఉత్తమ_FATTN_KERNEL_VEC;
            }
        }
    }
    తిరిగి ఉత్తమ_FATTN_KERNEL_TILE;
}

శూన్య ggml_sycl_flash_attn_ext(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_సమితి_device(ctx.device);

    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];
    ఒకవేళ (K && Q && K->ne[0] != Q->ne[0] && Q->ne[0] > 0 && K->ne[0] % Q->ne[0] == 0) {
        స్థిరరాశి int64_t d = Q->ne[0];
        స్థిరరాశి int64_t n_heads = (K->ne[0] * K->ne[2]) / d;
        K->ne[0] = d;
        K->ne[2] = n_heads;
    }
    ఒకవేళ (V && Q && V->ne[0] != Q->ne[0] && Q->ne[0] > 0 && V->ne[0] % Q->ne[0] == 0) {
        స్థిరరాశి int64_t d = Q->ne[0];
        స్థిరరాశి int64_t n_heads = (V->ne[0] * V->ne[2]) / d;
        V->ne[0] = d;
        V->ne[2] = n_heads;
    }

    // sparse nodes are gathered down to n_kv_max rows and re-dispatched here
    ఒకవేళ (ggml_sycl_flash_attn_ext_sparse(ctx, dst)) {
        తిరిగి;
    }

    // n_kv watchdog: log when n_kv differs from the last FA call with
    // the same D — helps detect cache-truncation issues.
    స్థిర పూర్ణాంకం nkv_debug = ggml_sycl_get_env("GGML_SYCL_MKL_FA_DEBUG", 0);
    ఒకవేళ (nkv_debug == 1) {
        స్థిరరాశి ggml_tensor * K_dbg = dst->src[1];
        స్థిరరాశి ggml_tensor * V_dbg = dst->src[2];
        స్థిర int64_t చివరి_nkv_d256 = 0, చివరి_nkv_d512 = 0;
        స్థిర పూర్ణాంకం fa_call_seq = 0;
        fa_call_seq++;
        int64_t cur_nkv = K_dbg->ne[1];
        పూర్ణాంకం Dk = (పూర్ణాంకం)K_dbg->ne[0];
        స్థిరరాశి అక్షరం * kname = "TILE";
        ఉత్తమ_fattn_kernel k = ggml_sycl_get_ఉత్తమ_fattn_kernel(ctx.device, dst);
        ఒకవేళ (k == ఉత్తమ_FATTN_KERNEL_MKL)  kname = "MKL";
        ఒకవేళ (k == ఉత్తమ_FATTN_KERNEL_ONEDNN)  kname = "ONEDNN";
        ఒకవేళ (k == ఉత్తమ_FATTN_KERNEL_VEC)  kname = "VEC";
        int64_t delta = 0;
        ఒకవేళ (Dk == 256) {
            delta = cur_nkv - చివరి_nkv_d256;
            చివరి_nkv_d256 = cur_nkv;
        } లేకపోతే ఒకవేళ (Dk == 512) {
            delta = cur_nkv - చివరి_nkv_d512;
            చివరి_nkv_d512 = cur_nkv;
        }
        GGML_LOG_INFO("[FA-DISP] #%d %s D=%d n_kv=%lld delta=%lld "
                "V_ne1=%lld\n",
                fa_call_seq, kname, Dk,
                (దీర్ఘ దీర్ఘ)cur_nkv, (దీర్ఘ దీర్ఘ)delta,
                (దీర్ఘ దీర్ఘ)V_dbg->ne[1]);
    }

    స్థిరరాశి ఉత్తమ_fattn_kernel fk = ggml_sycl_get_ఉత్తమ_fattn_kernel(ggml_sycl_get_device(), dst);
    ఎంపిక (fk) {
        స్థితి ఉత్తమ_FATTN_KERNEL_NONE:
            GGML_ABORT("Not support Flash-Attention");
        స్థితి ఉత్తమ_FATTN_KERNEL_ONEDNN:
            // guarded: ggml_sycl_flash_attn_ext_onednn() is only defined under GGML_SYCL_DNNL;
            // the reference must be compiled out here or the GGML_SYCL_DNNL=0 build fails to link.
#ఒకవేళ GGML_SYCL_DNNL
            ggml_sycl_flash_attn_ext_onednn(ctx, dst);
#అయితే_ముగింపు
            ఆపు;
        స్థితి ఉత్తమ_FATTN_KERNEL_TILE:
            ggml_sycl_flash_attn_ext_tile(ctx, dst);
            ఆపు;
        స్థితి ఉత్తమ_FATTN_KERNEL_VEC:
            ggml_sycl_flash_attn_ext_vec(ctx, dst);
            ఆపు;
        స్థితి ఉత్తమ_FATTN_KERNEL_MKL:
            ggml_sycl_flash_attn_ext_mkl(ctx, dst);
            ఆపు;
    }

    // --- Output fingerprint (GGML_SYCL_MKL_FA_DIAG=1) ---
    // Copy first 64 float output values to host for fingerprinting.
    // Compare MKL vs TILE (GGML_SYCL_ENABLE_MKL_FA=0) to detect divergence.
    // Only fingerprints the first 6 FA calls with n_kv >= 1024.
    స్థిర పూర్ణాంకం fa_diag = ggml_sycl_get_env("GGML_SYCL_MKL_FA_DIAG", 0);
    స్థిర పూర్ణాంకం fa_diag_లెక్క = 0;
    ఒకవేళ (fa_diag == 1 && fa_diag_లెక్క < 6) {
        స్థిరరాశి ggml_tensor * K_diag = dst->src[1];
        స్థిరరాశి ggml_tensor * V_diag = dst->src[2];
        స్థిరరాశి ggml_tensor * Q_diag = dst->src[0];
        ఒకవేళ (K_diag->ne[1] >= 1024) {
            fa_diag_లెక్క++;
            దశాంశం diag_buf[64];
            dpct::వరుస_ptr q = ctx.stream();
            q->జ్ఞాపకం_నకలు(diag_buf, dst->దత్తాంశం, 64 * పరిమాణం(దశాంశం));
            q->wait();
            స్థిరరాశి అక్షరం * kname = "???";
            ఉత్తమ_fattn_kernel kb = ggml_sycl_get_ఉత్తమ_fattn_kernel(ctx.device, dst);
            ఒకవేళ (kb == ఉత్తమ_FATTN_KERNEL_ONEDNN) kname = "ONEDNN";
            ఒకవేళ (kb == ఉత్తమ_FATTN_KERNEL_MKL) kname = "MKL";
            ఒకవేళ (kb == ఉత్తమ_FATTN_KERNEL_TILE) kname = "TILE";
            ఒకవేళ (kb == ఉత్తమ_FATTN_KERNEL_VEC) kname = "VEC";
            GGML_LOG_INFO("[FA-DIAG] #%d %s D=%d n_kv=%lld n_q=%lld "
                    "n_qh=%lld n_kvh=%lld K=%s V=%s "
                    "nb1=%zu nb2=%zu first 64 floats:\n",
                    fa_diag_లెక్క, kname,
                    (పూర్ణాంకం)K_diag->ne[0], (దీర్ఘ దీర్ఘ)K_diag->ne[1],
                    (దీర్ఘ దీర్ఘ)Q_diag->ne[1],
                    (దీర్ఘ దీర్ఘ)Q_diag->ne[2], (దీర్ఘ దీర్ఘ)K_diag->ne[2],
                    ggml_type_name(K_diag->type),
                    ggml_type_name(V_diag->type),
                    K_diag->nb[1], K_diag->nb[2]);
            కొరకు (పూర్ణాంకం i = 0; i < 64; i += 8) {
                GGML_LOG_INFO("  [%2d] %08x %08x %08x %08x %08x %08x %08x %08x\n",
                        i,
                        *(గుర్తులేని *)&diag_buf[i+0], *(గుర్తులేని *)&diag_buf[i+1],
                        *(గుర్తులేని *)&diag_buf[i+2], *(గుర్తులేని *)&diag_buf[i+3],
                        *(గుర్తులేని *)&diag_buf[i+4], *(గుర్తులేని *)&diag_buf[i+5],
                        *(గుర్తులేని *)&diag_buf[i+6], *(గుర్తులేని *)&diag_buf[i+7]);
            }
        }
    }

}

తార్కిక ggml_sycl_flash_attn_ext_supported(పూర్ణాంకం device, స్థిరరాశి ggml_tensor * dst) {
    తిరిగి ggml_sycl_get_ఉత్తమ_fattn_kernel(device, dst) != ఉత్తమ_FATTN_KERNEL_NONE;
}

స్థిర uintptr_t ggml_sycl_fattn_reserve_halves(ggml_sycl_fattn_extra & extra, size_t n_halves) {
    ఒకవేళ (n_halves == 0) {
        తిరిగి 0;
    }
    extra.end = GGML_PAD(extra.end, SYCL_బఫరు_ALIGNMENT);
    స్థిరరాశి uintptr_t ఖండం = extra.end;
    extra.end += n_halves * పరిమాణం(sycl::half);
    తిరిగి ఖండం;
}

ggml_sycl_fattn_extra ggml_sycl_fattn_get_extra(స్థిరరాశి ggml_tensor * dst) {
    ggml_sycl_fattn_extra extra;

    extra.end = (uintptr_t) dst->దత్తాంశం + ggml_nbytes(dst);

    ఒకవేళ (dst->op != GGML_OP_FLASH_ATTN_EXT) {
        తిరిగి extra;
    }

    స్థిరరాశి ggml_tensor * Q = dst->src[0];
    స్థిరరాశి ggml_tensor * K = dst->src[1];
    స్థిరరాశి ggml_tensor * V = dst->src[2];
    ఒకవేళ (!Q || !K || !V) {
        తిరిగి extra;
    }

    స్థిరరాశి int64_t d = K->ne[0];
    స్థిరరాశి int64_t H = Q->ne[2];
    స్థిరరాశి int64_t q = Q->ne[1];

    // calculate the worst-case memory consumption across all kernels
    స్థిరరాశి తార్కిక onednn_supported = ggml_sycl_flash_attn_ext_onednn_supported(dst, /* use_shape_limit */ అబద్ధం);

    స్థిరరాశి తార్కిక tile_needs_K = K->type != GGML_TYPE_F16;
    స్థిరరాశి తార్కిక tile_needs_V = V->type != GGML_TYPE_F16;

    స్థిరరాశి తార్కిక V_is_K_view = V->view_src &&
        (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));

    size_t need_K = 0, need_V = 0, need_Q = 0, need_బహిర్గతం = 0, need_scale = 0;
    ఒకవేళ (onednn_supported) {
        need_Q     = (size_t) H * q * d;
        need_బహిర్గతం   = (size_t) H * q * d;
        need_scale = 1;
        // an f16 cache is bound in place, so it needs no staging copy
        ఒకవేళ (!ggml_sycl_fattn_onednn_binds_kv(K, V)) {
            need_K = (size_t) ggml_nelements(K);
            need_V = (size_t) ggml_nelements(V);
        }
    }
    ఒకవేళ (tile_needs_K) {
        need_K = std::max(need_K, (size_t) ggml_nelements(K));
    }
    ఒకవేళ (tile_needs_V) {
        need_V = std::max(need_V, (size_t) ggml_nelements(V));
    }

    extra.Q_బఫరు_ptr = ggml_sycl_fattn_reserve_halves(extra, need_Q);
    extra.K_బఫరు_ptr = ggml_sycl_fattn_reserve_halves(extra, need_K);
    extra.V_బఫరు_ptr = (V_is_K_view && !onednn_supported && need_V)
                       ? extra.K_బఫరు_ptr
                       : ggml_sycl_fattn_reserve_halves(extra, need_V);
    extra.scale_బఫరు_ptr = ggml_sycl_fattn_reserve_halves(extra, need_scale);
    extra.బహిర్గతం_బఫరు_ptr   = ggml_sycl_fattn_reserve_halves(extra, need_బహిర్గతం);

    తిరిగి extra;
}

size_t ggml_sycl_flash_attn_ext_get_alloc_కొలత(స్థిరరాశి ggml_tensor * dst) {
    స్థిరరాశి ggml_sycl_fattn_extra extra = ggml_sycl_fattn_get_extra(dst);
    తిరిగి (size_t) (extra.end - (uintptr_t) dst->దత్తాంశం);
}
