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


#अन्तर्भाव <sycl/sycl.hpp>
#अन्तर्भाव "dpct/helper.hpp"
#अन्तर्भाव "common.hpp"
#अन्तर्भाव "fattn-common.hpp"
#अन्तर्भाव "fattn-tile.hpp"
#अन्तर्भाव "fattn-vec.hpp"
#अन्तर्भाव "fattn.hpp"
#अन्तर्भाव "fattn-onednn.hpp"
#अन्तर्भाव "fattn-sparse.hpp"

#define FATTN_VEC_CASE(D, type_K, type_V)                                                                        \
    {                                                                                                            \
        ध्रुव तार्किक type_K_okay = K->type == (type_K) || (K->type == GGML_TYPE_F32 && (type_K) == GGML_TYPE_F16); \
        ध्रुव तार्किक type_V_okay = V->type == (type_V) || (V->type == GGML_TYPE_F32 && (type_V) == GGML_TYPE_F16); \
        यदि (Q->ne[0] == (D) && type_K_okay && type_V_okay) {                                                     \
            ggml_sycl_flash_attn_ext_vec_case<D, type_K, type_V>(ctx, dst);                                      \
            प्रत्यागम;                                                                                              \
        }                                                                                                        \
    }                                                                    \

#define FATTN_VEC_CASES_ALL_D(type_K, type_V) \
    FATTN_VEC_CASE( 64, type_K, type_V)       \
    FATTN_VEC_CASE(128, type_K, type_V)       \
    FATTN_VEC_CASE(256, type_K, type_V)       \
    FATTN_VEC_CASE(512, type_K, type_V)       \

स्थिर शून्य ggml_sycl_flash_attn_ext_vec(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
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
#अन्यथा
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
#endif // GGML_SYCL_FA_ALL_QUANTS

    GGML_ABORT("Not match KV type in vec");
}

// Best FlashAttention kernel for a specific GPU:
गणना best_fattn_kernel {
    BEST_FATTN_KERNEL_NONE     =   0,
    BEST_FATTN_KERNEL_VEC      = 100,
    BEST_FATTN_KERNEL_ONEDNN   = 150, // oneDNN SDPA: native F16 (PR #25222)
    BEST_FATTN_KERNEL_TILE     = 200,
    BEST_FATTN_KERNEL_MKL      = 300,
};


स्थिर best_fattn_kernel ggml_sycl_get_best_fattn_kernel(ध्रुव पूर्णाङ्क device, ध्रुव ggml_tensor * dst) {
#ifndef SYCL_FLASH_ATTN
    GGML_UNUSED(dst);
    प्रत्यागम BEST_FATTN_KERNEL_NONE;
#endif// SYCL_FLASH_ATTN

    यदि(!g_ggml_sycl_enable_flash_attention) प्रत्यागम BEST_FATTN_KERNEL_NONE;

    ध्रुव ggml_tensor * KQV   = dst;
    ध्रुव ggml_tensor * Q     = dst->src[0];
    ध्रुव ggml_tensor * K     = dst->src[1];
    ध्रुव ggml_tensor * V     = dst->src[2];
    ध्रुव ggml_tensor * mask  = dst->src[3];
    ध्रुव ggml_tensor * sinks = dst->src[4];

    ध्रुव पूर्णाङ्क gqa_ratio = Q->ne[2] / K->ne[2];
    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);

    भिन्न max_bias = 0.0f;
    memcpy(&max_bias, (ध्रुव भिन्न *) KQV->op_params + 1, परिमाण(भिन्न));

    भिन्न logit_softcap = 0.0f;
    memcpy(&logit_softcap, (ध्रुव भिन्न *) KQV->op_params + 2, परिमाण(भिन्न));

    तार्किक gqa_opt_applies = gqa_ratio >= 2 && mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;

    // XMX-accelerated path: oneDNN SDPA (native F16 and dequant+non-F16).
    // ONEDNN requires min 32 query tokens — short-circuit decode to avoid
    // calling _supported() on every decode FA call.
    यदि (Q->ne[1] >= 32
        && ggml_sycl_flash_attn_ext_onednn_supported(dst)) {
        प्रत्यागम BEST_FATTN_KERNEL_ONEDNN;
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
    ध्रुव तार्किक standard_shape = Q->ne[0] >= 64 && Q->ne[0] <= 512 &&
        Q->ne[0] % 64 == 0 && Q->ne[0] == V->ne[0];
    // GLM-4.7 Flash's MLA shape is already expressible by the MKL pipeline:
    // KQ GEMM uses DKQ=576 while VKQ and output use DV=512. Keep this narrow
    // until other mismatched K/V shapes have independent correctness data.
    ध्रुव तार्किक glm_mla_shape = Q->ne[0] == 576 && K->ne[0] == 576 &&
        V->ne[0] == 512 && gqa_ratio == 20 &&
        K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16;

    यदि (g_ggml_sycl_enable_mkl_fa == 1 && mask && !sinks && gqa_ratio >= 2 &&
        (standard_shape || glm_mla_shape) &&
        Q->ne[1] >= 32 && K->ne[1] >= 1024 &&
        max_bias == 0.0f && logit_softcap == 0.0f &&
        (Q->ne[3] == K->ne[3] || K->ne[3] == 1)) {
        // F16 K/V strides must be a multiple of ne[0]*2 (the natural row size
        // in bytes). This passes both dense (nb1 == ne0*2) and interleaved
        // (nb1 == H * ne0*2). Only pathological test strides like nb1=32 or
        // nb1=75 for ne0=40 fall through to TILE.
        तार्किक kv_strides_ok = सत्य;
        प्रति (ध्रुव ggml_tensor * t : {K, V}) {
            ध्रुव तार्किक glm_v_stride = glm_mla_shape && t == V &&
                V->view_src && V->nb[1] == K->nb[1];
            यदि (!glm_v_stride && t->type == GGML_TYPE_F16 &&
                    t->nb[1] % (t->ne[0] * 2) != 0) {
                kv_strides_ok = असत्य;
                भङ्ग;
            }
        }
        यदि (kv_strides_ok) {
            प्रत्यागम BEST_FATTN_KERNEL_MKL;
        }
    }
    प्रति (ध्रुव ggml_tensor * t : {Q, K, V, mask}) {
        यदि (t == nullptr || ggml_is_quantized(t->type)) {
            अनुवर्त;
        }
        प्रति (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            यदि (t->nb[i] % 16 != 0) {
                gqa_opt_applies = असत्य;
                भङ्ग;
            }
        }
    }

    विकल्प (K->ne[0]) {
        पक्ष  40:
        पक्ष  64:
        पक्ष  72:
        पक्ष  80:
        पक्ष  96:
        पक्ष 128:
        पक्ष 112:
        पक्ष 256:
        पक्ष 512:
            यदि (V->ne[0] != K->ne[0]) {
                प्रत्यागम BEST_FATTN_KERNEL_NONE;
            }
            भङ्ग;
        पक्ष 576:
            यदि (V->ne[0] != 512) {
                प्रत्यागम BEST_FATTN_KERNEL_NONE;
            }
            यदि (!gqa_opt_applies) {
                प्रत्यागम BEST_FATTN_KERNEL_NONE;
            }
            भङ्ग;
        स्वभाव:
            प्रत्यागम BEST_FATTN_KERNEL_NONE;
    }

#ifndef GGML_SYCL_FA_ALL_QUANTS
    यदि (K->type != V->type) {
        प्रत्यागम BEST_FATTN_KERNEL_NONE;
    }
#endif // GGML_SYCL_FA_ALL_QUANTS

    विकल्प (K->type) {
        पक्ष GGML_TYPE_F32:
        पक्ष GGML_TYPE_F16:
        पक्ष GGML_TYPE_BF16:
            भङ्ग;
        पक्ष GGML_TYPE_Q4_1:
        पक्ष GGML_TYPE_Q5_0:
        पक्ष GGML_TYPE_Q5_1:
#ifndef GGML_SYCL_FA_ALL_QUANTS
            प्रत्यागम BEST_FATTN_KERNEL_NONE;
#endif // GGML_SYCL_FA_ALL_QUANTS
        पक्ष GGML_TYPE_Q4_0:
        पक्ष GGML_TYPE_Q8_0:
            भङ्ग;
        स्वभाव:
            प्रत्यागम BEST_FATTN_KERNEL_NONE;
    }

    यदि (mask && mask->ne[2] != 1) {
        प्रत्यागम BEST_FATTN_KERNEL_NONE;
    }

    // For small batch sizes the vector kernel may be preferable over the kernels optimized for large batch sizes.
    // BF16 is excluded: the VEC kernel has no BF16 template (it needs GGML_SYCL_FA_ALL_QUANTS for non-F16/Q4_0/Q8_0).
    ध्रुव तार्किक has_bf16 = (K->type == GGML_TYPE_BF16 || V->type == GGML_TYPE_BF16);
    ध्रुव तार्किक can_use_vector_kernel = Q->ne[0] <= 512 && Q->ne[0] % 64 == 0 && K->ne[1] % FATTN_KQ_STRIDE == 0
        && !has_bf16;

    // Fused-XMX path: oneDNN Graph SDPA (flash attention). Strictly
    // additive -- taken only when statically supported, otherwise falls through to VEC/TILE below.
    यदि (ggml_sycl_flash_attn_ext_onednn_supported(dst)) {
        प्रत्यागम BEST_FATTN_KERNEL_ONEDNN;
    }

    // If there are no tensor cores available, use the generic tile kernel:
    यदि (can_use_vector_kernel) {
        यदि (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
            यदि (Q->ne[1] == 1) {
                यदि (!gqa_opt_applies) {
                    प्रत्यागम BEST_FATTN_KERNEL_VEC;
                }
            }
        } अन्यथा {
            यदि (Q->ne[1] <= 2) {
                // TILE is faster for quantized KV decode on Xe2 (BMG); keep VEC on untested archs
                ध्रुव gpu_arch arch = ggml_sycl_info().devices[device].hw_info.arch;
                यदि (arch == gpu_arch::intel_gpu_bmg_g21 || arch == gpu_arch::intel_gpu_bmg_g31) {
                    प्रत्यागम BEST_FATTN_KERNEL_TILE;
                }
                प्रत्यागम BEST_FATTN_KERNEL_VEC;
            }
        }
    }
    प्रत्यागम BEST_FATTN_KERNEL_TILE;
}

शून्य ggml_sycl_flash_attn_ext(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_set_device(ctx.device);

    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];
    यदि (K && Q && K->ne[0] != Q->ne[0] && Q->ne[0] > 0 && K->ne[0] % Q->ne[0] == 0) {
        ध्रुव int64_t d = Q->ne[0];
        ध्रुव int64_t n_heads = (K->ne[0] * K->ne[2]) / d;
        K->ne[0] = d;
        K->ne[2] = n_heads;
    }
    यदि (V && Q && V->ne[0] != Q->ne[0] && Q->ne[0] > 0 && V->ne[0] % Q->ne[0] == 0) {
        ध्रुव int64_t d = Q->ne[0];
        ध्रुव int64_t n_heads = (V->ne[0] * V->ne[2]) / d;
        V->ne[0] = d;
        V->ne[2] = n_heads;
    }

    // sparse nodes are gathered down to n_kv_max rows and re-dispatched here
    यदि (ggml_sycl_flash_attn_ext_sparse(ctx, dst)) {
        प्रत्यागम;
    }

    // n_kv watchdog: log when n_kv differs from the last FA call with
    // the same D — helps detect cache-truncation issues.
    स्थिर पूर्णाङ्क nkv_debug = ggml_sycl_get_env("GGML_SYCL_MKL_FA_DEBUG", 0);
    यदि (nkv_debug == 1) {
        ध्रुव ggml_tensor * K_dbg = dst->src[1];
        ध्रुव ggml_tensor * V_dbg = dst->src[2];
        स्थिर int64_t last_nkv_d256 = 0, last_nkv_d512 = 0;
        स्थिर पूर्णाङ्क fa_call_seq = 0;
        fa_call_seq++;
        int64_t cur_nkv = K_dbg->ne[1];
        पूर्णाङ्क Dk = (पूर्णाङ्क)K_dbg->ne[0];
        ध्रुव वर्ण * kname = "TILE";
        best_fattn_kernel k = ggml_sycl_get_best_fattn_kernel(ctx.device, dst);
        यदि (k == BEST_FATTN_KERNEL_MKL)  kname = "MKL";
        यदि (k == BEST_FATTN_KERNEL_ONEDNN)  kname = "ONEDNN";
        यदि (k == BEST_FATTN_KERNEL_VEC)  kname = "VEC";
        int64_t delta = 0;
        यदि (Dk == 256) {
            delta = cur_nkv - last_nkv_d256;
            last_nkv_d256 = cur_nkv;
        } अन्यथा यदि (Dk == 512) {
            delta = cur_nkv - last_nkv_d512;
            last_nkv_d512 = cur_nkv;
        }
        GGML_LOG_INFO("[FA-DISP] #%d %s D=%d n_kv=%lld delta=%lld "
                "V_ne1=%lld\n",
                fa_call_seq, kname, Dk,
                (long long)cur_nkv, (long long)delta,
                (long long)V_dbg->ne[1]);
    }

    ध्रुव best_fattn_kernel fk = ggml_sycl_get_best_fattn_kernel(ggml_sycl_get_device(), dst);
    विकल्प (fk) {
        पक्ष BEST_FATTN_KERNEL_NONE:
            GGML_ABORT("Not support Flash-Attention");
        पक्ष BEST_FATTN_KERNEL_ONEDNN:
            // guarded: ggml_sycl_flash_attn_ext_onednn() is only defined under GGML_SYCL_DNNL;
            // the reference must be compiled out here or the GGML_SYCL_DNNL=0 build fails to link.
#यदि GGML_SYCL_DNNL
            ggml_sycl_flash_attn_ext_onednn(ctx, dst);
#endif
            भङ्ग;
        पक्ष BEST_FATTN_KERNEL_TILE:
            ggml_sycl_flash_attn_ext_tile(ctx, dst);
            भङ्ग;
        पक्ष BEST_FATTN_KERNEL_VEC:
            ggml_sycl_flash_attn_ext_vec(ctx, dst);
            भङ्ग;
        पक्ष BEST_FATTN_KERNEL_MKL:
            ggml_sycl_flash_attn_ext_mkl(ctx, dst);
            भङ्ग;
    }

    // --- Output fingerprint (GGML_SYCL_MKL_FA_DIAG=1) ---
    // Copy first 64 float output values to host for fingerprinting.
    // Compare MKL vs TILE (GGML_SYCL_ENABLE_MKL_FA=0) to detect divergence.
    // Only fingerprints the first 6 FA calls with n_kv >= 1024.
    स्थिर पूर्णाङ्क fa_diag = ggml_sycl_get_env("GGML_SYCL_MKL_FA_DIAG", 0);
    स्थिर पूर्णाङ्क fa_diag_गिनती = 0;
    यदि (fa_diag == 1 && fa_diag_गिनती < 6) {
        ध्रुव ggml_tensor * K_diag = dst->src[1];
        ध्रुव ggml_tensor * V_diag = dst->src[2];
        ध्रुव ggml_tensor * Q_diag = dst->src[0];
        यदि (K_diag->ne[1] >= 1024) {
            fa_diag_गिनती++;
            भिन्न diag_buf[64];
            dpct::queue_ptr q = ctx.stream();
            q->memcpy(diag_buf, dst->data, 64 * परिमाण(भिन्न));
            q->wait();
            ध्रुव वर्ण * kname = "???";
            best_fattn_kernel kb = ggml_sycl_get_best_fattn_kernel(ctx.device, dst);
            यदि (kb == BEST_FATTN_KERNEL_ONEDNN) kname = "ONEDNN";
            यदि (kb == BEST_FATTN_KERNEL_MKL) kname = "MKL";
            यदि (kb == BEST_FATTN_KERNEL_TILE) kname = "TILE";
            यदि (kb == BEST_FATTN_KERNEL_VEC) kname = "VEC";
            GGML_LOG_INFO("[FA-DIAG] #%d %s D=%d n_kv=%lld n_q=%lld "
                    "n_qh=%lld n_kvh=%lld K=%s V=%s "
                    "nb1=%zu nb2=%zu first 64 floats:\n",
                    fa_diag_गिनती, kname,
                    (पूर्णाङ्क)K_diag->ne[0], (long long)K_diag->ne[1],
                    (long long)Q_diag->ne[1],
                    (long long)Q_diag->ne[2], (long long)K_diag->ne[2],
                    ggml_type_name(K_diag->type),
                    ggml_type_name(V_diag->type),
                    K_diag->nb[1], K_diag->nb[2]);
            प्रति (पूर्णाङ्क i = 0; i < 64; i += 8) {
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

तार्किक ggml_sycl_flash_attn_ext_supported(पूर्णाङ्क device, ध्रुव ggml_tensor * dst) {
    प्रत्यागम ggml_sycl_get_best_fattn_kernel(device, dst) != BEST_FATTN_KERNEL_NONE;
}

स्थिर uintptr_t ggml_sycl_fattn_reserve_halves(ggml_sycl_fattn_extra & extra, size_t n_halves) {
    यदि (n_halves == 0) {
        प्रत्यागम 0;
    }
    extra.end = GGML_PAD(extra.end, SYCL_BUFFER_ALIGNMENT);
    ध्रुव uintptr_t block = extra.end;
    extra.end += n_halves * परिमाण(sycl::half);
    प्रत्यागम block;
}

ggml_sycl_fattn_extra ggml_sycl_fattn_get_extra(ध्रुव ggml_tensor * dst) {
    ggml_sycl_fattn_extra extra;

    extra.end = (uintptr_t) dst->data + ggml_nbytes(dst);

    यदि (dst->op != GGML_OP_FLASH_ATTN_EXT) {
        प्रत्यागम extra;
    }

    ध्रुव ggml_tensor * Q = dst->src[0];
    ध्रुव ggml_tensor * K = dst->src[1];
    ध्रुव ggml_tensor * V = dst->src[2];
    यदि (!Q || !K || !V) {
        प्रत्यागम extra;
    }

    ध्रुव int64_t d = K->ne[0];
    ध्रुव int64_t H = Q->ne[2];
    ध्रुव int64_t q = Q->ne[1];

    // calculate the worst-case memory consumption across all kernels
    ध्रुव तार्किक onednn_supported = ggml_sycl_flash_attn_ext_onednn_supported(dst, /* use_shape_limit */ असत्य);

    ध्रुव तार्किक tile_needs_K = K->type != GGML_TYPE_F16;
    ध्रुव तार्किक tile_needs_V = V->type != GGML_TYPE_F16;

    ध्रुव तार्किक V_is_K_view = V->view_src &&
        (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));

    size_t need_K = 0, need_V = 0, need_Q = 0, need_out = 0, need_scale = 0;
    यदि (onednn_supported) {
        need_Q     = (size_t) H * q * d;
        need_out   = (size_t) H * q * d;
        need_scale = 1;
        // an f16 cache is bound in place, so it needs no staging copy
        यदि (!ggml_sycl_fattn_onednn_binds_kv(K, V)) {
            need_K = (size_t) ggml_nelements(K);
            need_V = (size_t) ggml_nelements(V);
        }
    }
    यदि (tile_needs_K) {
        need_K = std::max(need_K, (size_t) ggml_nelements(K));
    }
    यदि (tile_needs_V) {
        need_V = std::max(need_V, (size_t) ggml_nelements(V));
    }

    extra.Q_buffer_ptr = ggml_sycl_fattn_reserve_halves(extra, need_Q);
    extra.K_buffer_ptr = ggml_sycl_fattn_reserve_halves(extra, need_K);
    extra.V_buffer_ptr = (V_is_K_view && !onednn_supported && need_V)
                       ? extra.K_buffer_ptr
                       : ggml_sycl_fattn_reserve_halves(extra, need_V);
    extra.scale_buffer_ptr = ggml_sycl_fattn_reserve_halves(extra, need_scale);
    extra.out_buffer_ptr   = ggml_sycl_fattn_reserve_halves(extra, need_out);

    प्रत्यागम extra;
}

size_t ggml_sycl_flash_attn_ext_get_alloc_size(ध्रुव ggml_tensor * dst) {
    ध्रुव ggml_sycl_fattn_extra extra = ggml_sycl_fattn_get_extra(dst);
    प्रत्यागम (size_t) (extra.end - (uintptr_t) dst->data);
}
