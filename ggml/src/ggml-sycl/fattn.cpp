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


#समावेश <sycl/sycl.hpp>
#समावेश "dpct/helper.hpp"
#समावेश "common.hpp"
#समावेश "fattn-common.hpp"
#समावेश "fattn-tile.hpp"
#समावेश "fattn-vec.hpp"
#समावेश "fattn.hpp"
#समावेश "fattn-onednn.hpp"
#समावेश "fattn-sparse.hpp"

#घोषणा FATTN_VEC_CASE(D, type_K, type_V)                                                                        \
    {                                                                                                            \
        स्थिर bool type_K_okay = K->type == (type_K) || (K->type == GGML_TYPE_F32 && (type_K) == GGML_TYPE_F16); \
        स्थिर bool type_V_okay = V->type == (type_V) || (V->type == GGML_TYPE_F32 && (type_V) == GGML_TYPE_F16); \
        अगर (Q->ne[0] == (D) && type_K_okay && type_V_okay) {                                                     \
            ggml_sycl_flash_attn_ext_vec_case<D, type_K, type_V>(ctx, dst);                                      \
            वापस;                                                                                              \
        }                                                                                                        \
    }                                                                    \

#घोषणा FATTN_VEC_CASES_ALL_D(type_K, type_V) \
    FATTN_VEC_CASE( 64, type_K, type_V)       \
    FATTN_VEC_CASE(128, type_K, type_V)       \
    FATTN_VEC_CASE(256, type_K, type_V)       \
    FATTN_VEC_CASE(512, type_K, type_V)       \

अटल व्योम ggml_sycl_flash_attn_ext_vec(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];

#अगर_घोषित GGML_SYCL_FA_ALL_QUANTS
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
#पूर्ण_अगर // GGML_SYCL_FA_ALL_QUANTS

    GGML_ABORT("Not match KV type in vec");
}

// Best FlashAttention kernel for a specific GPU:
क्रमागत best_fattn_kernel {
    BEST_FATTN_KERNEL_NONE     =   0,
    BEST_FATTN_KERNEL_VEC      = 100,
    BEST_FATTN_KERNEL_ONEDNN   = 150, // oneDNN SDPA: native F16 (PR #25222)
    BEST_FATTN_KERNEL_TILE     = 200,
    BEST_FATTN_KERNEL_MKL      = 300,
};


अटल best_fattn_kernel ggml_sycl_get_best_fattn_kernel(स्थिर पूर्णांक device, स्थिर ggml_tensor * dst) {
#अगर_अघोषित SYCL_FLASH_ATTN
    GGML_UNUSED(dst);
    वापस BEST_FATTN_KERNEL_NONE;
#पूर्ण_अगर// SYCL_FLASH_ATTN

    अगर(!g_ggml_sycl_enable_flash_attention) वापस BEST_FATTN_KERNEL_NONE;

    स्थिर ggml_tensor * KQV   = dst;
    स्थिर ggml_tensor * Q     = dst->src[0];
    स्थिर ggml_tensor * K     = dst->src[1];
    स्थिर ggml_tensor * V     = dst->src[2];
    स्थिर ggml_tensor * mask  = dst->src[3];
    स्थिर ggml_tensor * sinks = dst->src[4];

    स्थिर पूर्णांक gqa_ratio = Q->ne[2] / K->ne[2];
    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);

    भग्न max_bias = 0.0f;
    स_नकल(&max_bias, (स्थिर भग्न *) KQV->op_params + 1, माप(भग्न));

    भग्न logit_softcap = 0.0f;
    स_नकल(&logit_softcap, (स्थिर भग्न *) KQV->op_params + 2, माप(भग्न));

    bool gqa_opt_applies = gqa_ratio >= 2 && mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;

    // XMX-accelerated path: oneDNN SDPA (native F16 and dequant+non-F16).
    // ONEDNN requires min 32 query tokens — short-circuit decode to avoid
    // calling _supported() on every decode FA call.
    अगर (Q->ne[1] >= 32
        && ggml_sycl_flash_attn_ext_onednn_supported(dst)) {
        वापस BEST_FATTN_KERNEL_ONEDNN;
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
    स्थिर bool standard_shape = Q->ne[0] >= 64 && Q->ne[0] <= 512 &&
        Q->ne[0] % 64 == 0 && Q->ne[0] == V->ne[0];
    // GLM-4.7 Flash's MLA shape is already expressible by the MKL pipeline:
    // KQ GEMM uses DKQ=576 while VKQ and output use DV=512. Keep this narrow
    // until other mismatched K/V shapes have independent correctness data.
    स्थिर bool glm_mla_shape = Q->ne[0] == 576 && K->ne[0] == 576 &&
        V->ne[0] == 512 && gqa_ratio == 20 &&
        K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16;

    अगर (g_ggml_sycl_enable_mkl_fa == 1 && mask && !sinks && gqa_ratio >= 2 &&
        (standard_shape || glm_mla_shape) &&
        Q->ne[1] >= 32 && K->ne[1] >= 1024 &&
        max_bias == 0.0f && logit_softcap == 0.0f &&
        (Q->ne[3] == K->ne[3] || K->ne[3] == 1)) {
        // F16 K/V strides must be a multiple of ne[0]*2 (the natural row size
        // in bytes). This passes both dense (nb1 == ne0*2) and interleaved
        // (nb1 == H * ne0*2). Only pathological test strides like nb1=32 or
        // nb1=75 for ne0=40 fall through to TILE.
        bool kv_strides_ok = true;
        क्रम (स्थिर ggml_tensor * t : {K, V}) {
            स्थिर bool glm_v_stride = glm_mla_shape && t == V &&
                V->view_src && V->nb[1] == K->nb[1];
            अगर (!glm_v_stride && t->type == GGML_TYPE_F16 &&
                    t->nb[1] % (t->ne[0] * 2) != 0) {
                kv_strides_ok = false;
                अवरोध;
            }
        }
        अगर (kv_strides_ok) {
            वापस BEST_FATTN_KERNEL_MKL;
        }
    }
    क्रम (स्थिर ggml_tensor * t : {Q, K, V, mask}) {
        अगर (t == nullptr || ggml_is_quantized(t->type)) {
            जारी;
        }
        क्रम (माप_प्रकार i = 1; i < GGML_MAX_DIMS; ++i) {
            अगर (t->nb[i] % 16 != 0) {
                gqa_opt_applies = false;
                अवरोध;
            }
        }
    }

    चयन (K->ne[0]) {
        हाल  40:
        हाल  64:
        हाल  72:
        हाल  80:
        हाल  96:
        हाल 128:
        हाल 112:
        हाल 256:
        हाल 512:
            अगर (V->ne[0] != K->ne[0]) {
                वापस BEST_FATTN_KERNEL_NONE;
            }
            अवरोध;
        हाल 576:
            अगर (V->ne[0] != 512) {
                वापस BEST_FATTN_KERNEL_NONE;
            }
            अगर (!gqa_opt_applies) {
                वापस BEST_FATTN_KERNEL_NONE;
            }
            अवरोध;
        शेष:
            वापस BEST_FATTN_KERNEL_NONE;
    }

#अगर_अघोषित GGML_SYCL_FA_ALL_QUANTS
    अगर (K->type != V->type) {
        वापस BEST_FATTN_KERNEL_NONE;
    }
#पूर्ण_अगर // GGML_SYCL_FA_ALL_QUANTS

    चयन (K->type) {
        हाल GGML_TYPE_F32:
        हाल GGML_TYPE_F16:
        हाल GGML_TYPE_BF16:
            अवरोध;
        हाल GGML_TYPE_Q4_1:
        हाल GGML_TYPE_Q5_0:
        हाल GGML_TYPE_Q5_1:
#अगर_अघोषित GGML_SYCL_FA_ALL_QUANTS
            वापस BEST_FATTN_KERNEL_NONE;
#पूर्ण_अगर // GGML_SYCL_FA_ALL_QUANTS
        हाल GGML_TYPE_Q4_0:
        हाल GGML_TYPE_Q8_0:
            अवरोध;
        शेष:
            वापस BEST_FATTN_KERNEL_NONE;
    }

    अगर (mask && mask->ne[2] != 1) {
        वापस BEST_FATTN_KERNEL_NONE;
    }

    // For small batch sizes the vector kernel may be preferable over the kernels optimized for large batch sizes.
    // BF16 is excluded: the VEC kernel has no BF16 template (it needs GGML_SYCL_FA_ALL_QUANTS for non-F16/Q4_0/Q8_0).
    स्थिर bool has_bf16 = (K->type == GGML_TYPE_BF16 || V->type == GGML_TYPE_BF16);
    स्थिर bool can_use_vector_kernel = Q->ne[0] <= 512 && Q->ne[0] % 64 == 0 && K->ne[1] % FATTN_KQ_STRIDE == 0
        && !has_bf16;

    // Fused-XMX path: oneDNN Graph SDPA (flash attention). Strictly
    // additive -- taken only when statically supported, otherwise falls through to VEC/TILE below.
    अगर (ggml_sycl_flash_attn_ext_onednn_supported(dst)) {
        वापस BEST_FATTN_KERNEL_ONEDNN;
    }

    // If there are no tensor cores available, use the generic tile kernel:
    अगर (can_use_vector_kernel) {
        अगर (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
            अगर (Q->ne[1] == 1) {
                अगर (!gqa_opt_applies) {
                    वापस BEST_FATTN_KERNEL_VEC;
                }
            }
        } अन्यथा {
            अगर (Q->ne[1] <= 2) {
                // TILE is faster for quantized KV decode on Xe2 (BMG); keep VEC on untested archs
                स्थिर gpu_arch arch = ggml_sycl_info().devices[device].hw_info.arch;
                अगर (arch == gpu_arch::intel_gpu_bmg_g21 || arch == gpu_arch::intel_gpu_bmg_g31) {
                    वापस BEST_FATTN_KERNEL_TILE;
                }
                वापस BEST_FATTN_KERNEL_VEC;
            }
        }
    }
    वापस BEST_FATTN_KERNEL_TILE;
}

व्योम ggml_sycl_flash_attn_ext(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_set_device(ctx.device);

    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];
    अगर (K && Q && K->ne[0] != Q->ne[0] && Q->ne[0] > 0 && K->ne[0] % Q->ne[0] == 0) {
        स्थिर int64_t d = Q->ne[0];
        स्थिर int64_t n_heads = (K->ne[0] * K->ne[2]) / d;
        K->ne[0] = d;
        K->ne[2] = n_heads;
    }
    अगर (V && Q && V->ne[0] != Q->ne[0] && Q->ne[0] > 0 && V->ne[0] % Q->ne[0] == 0) {
        स्थिर int64_t d = Q->ne[0];
        स्थिर int64_t n_heads = (V->ne[0] * V->ne[2]) / d;
        V->ne[0] = d;
        V->ne[2] = n_heads;
    }

    // sparse nodes are gathered down to n_kv_max rows and re-dispatched here
    अगर (ggml_sycl_flash_attn_ext_sparse(ctx, dst)) {
        वापस;
    }

    // n_kv watchdog: log when n_kv differs from the last FA call with
    // the same D — helps detect cache-truncation issues.
    अटल पूर्णांक nkv_debug = ggml_sycl_get_env("GGML_SYCL_MKL_FA_DEBUG", 0);
    अगर (nkv_debug == 1) {
        स्थिर ggml_tensor * K_dbg = dst->src[1];
        स्थिर ggml_tensor * V_dbg = dst->src[2];
        अटल int64_t last_nkv_d256 = 0, last_nkv_d512 = 0;
        अटल पूर्णांक fa_call_seq = 0;
        fa_call_seq++;
        int64_t cur_nkv = K_dbg->ne[1];
        पूर्णांक Dk = (पूर्णांक)K_dbg->ne[0];
        स्थिर अक्षर * kname = "TILE";
        best_fattn_kernel k = ggml_sycl_get_best_fattn_kernel(ctx.device, dst);
        अगर (k == BEST_FATTN_KERNEL_MKL)  kname = "MKL";
        अगर (k == BEST_FATTN_KERNEL_ONEDNN)  kname = "ONEDNN";
        अगर (k == BEST_FATTN_KERNEL_VEC)  kname = "VEC";
        int64_t delta = 0;
        अगर (Dk == 256) {
            delta = cur_nkv - last_nkv_d256;
            last_nkv_d256 = cur_nkv;
        } अन्यथा अगर (Dk == 512) {
            delta = cur_nkv - last_nkv_d512;
            last_nkv_d512 = cur_nkv;
        }
        GGML_LOG_INFO("[FA-DISP] #%d %s D=%d n_kv=%lld delta=%lld "
                "V_ne1=%lld\n",
                fa_call_seq, kname, Dk,
                (दीर्घ दीर्घ)cur_nkv, (दीर्घ दीर्घ)delta,
                (दीर्घ दीर्घ)V_dbg->ne[1]);
    }

    स्थिर best_fattn_kernel fk = ggml_sycl_get_best_fattn_kernel(ggml_sycl_get_device(), dst);
    चयन (fk) {
        हाल BEST_FATTN_KERNEL_NONE:
            GGML_ABORT("Not support Flash-Attention");
        हाल BEST_FATTN_KERNEL_ONEDNN:
            // guarded: ggml_sycl_flash_attn_ext_onednn() is only defined under GGML_SYCL_DNNL;
            // the reference must be compiled out here or the GGML_SYCL_DNNL=0 build fails to link.
#अगर GGML_SYCL_DNNL
            ggml_sycl_flash_attn_ext_onednn(ctx, dst);
#पूर्ण_अगर
            अवरोध;
        हाल BEST_FATTN_KERNEL_TILE:
            ggml_sycl_flash_attn_ext_tile(ctx, dst);
            अवरोध;
        हाल BEST_FATTN_KERNEL_VEC:
            ggml_sycl_flash_attn_ext_vec(ctx, dst);
            अवरोध;
        हाल BEST_FATTN_KERNEL_MKL:
            ggml_sycl_flash_attn_ext_mkl(ctx, dst);
            अवरोध;
    }

    // --- Output fingerprint (GGML_SYCL_MKL_FA_DIAG=1) ---
    // Copy first 64 float output values to host for fingerprinting.
    // Compare MKL vs TILE (GGML_SYCL_ENABLE_MKL_FA=0) to detect divergence.
    // Only fingerprints the first 6 FA calls with n_kv >= 1024.
    अटल पूर्णांक fa_diag = ggml_sycl_get_env("GGML_SYCL_MKL_FA_DIAG", 0);
    अटल पूर्णांक fa_diag_गिनती = 0;
    अगर (fa_diag == 1 && fa_diag_गिनती < 6) {
        स्थिर ggml_tensor * K_diag = dst->src[1];
        स्थिर ggml_tensor * V_diag = dst->src[2];
        स्थिर ggml_tensor * Q_diag = dst->src[0];
        अगर (K_diag->ne[1] >= 1024) {
            fa_diag_गिनती++;
            भग्न diag_buf[64];
            dpct::queue_ptr q = ctx.stream();
            q->स_नकल(diag_buf, dst->तथ्य, 64 * माप(भग्न));
            q->रुको();
            स्थिर अक्षर * kname = "???";
            best_fattn_kernel kb = ggml_sycl_get_best_fattn_kernel(ctx.device, dst);
            अगर (kb == BEST_FATTN_KERNEL_ONEDNN) kname = "ONEDNN";
            अगर (kb == BEST_FATTN_KERNEL_MKL) kname = "MKL";
            अगर (kb == BEST_FATTN_KERNEL_TILE) kname = "TILE";
            अगर (kb == BEST_FATTN_KERNEL_VEC) kname = "VEC";
            GGML_LOG_INFO("[FA-DIAG] #%d %s D=%d n_kv=%lld n_q=%lld "
                    "n_qh=%lld n_kvh=%lld K=%s V=%s "
                    "nb1=%zu nb2=%zu first 64 floats:\n",
                    fa_diag_गिनती, kname,
                    (पूर्णांक)K_diag->ne[0], (दीर्घ दीर्घ)K_diag->ne[1],
                    (दीर्घ दीर्घ)Q_diag->ne[1],
                    (दीर्घ दीर्घ)Q_diag->ne[2], (दीर्घ दीर्घ)K_diag->ne[2],
                    ggml_type_name(K_diag->type),
                    ggml_type_name(V_diag->type),
                    K_diag->nb[1], K_diag->nb[2]);
            क्रम (पूर्णांक i = 0; i < 64; i += 8) {
                GGML_LOG_INFO("  [%2d] %08x %08x %08x %08x %08x %08x %08x %08x\n",
                        i,
                        *(अचिन्हित *)&diag_buf[i+0], *(अचिन्हित *)&diag_buf[i+1],
                        *(अचिन्हित *)&diag_buf[i+2], *(अचिन्हित *)&diag_buf[i+3],
                        *(अचिन्हित *)&diag_buf[i+4], *(अचिन्हित *)&diag_buf[i+5],
                        *(अचिन्हित *)&diag_buf[i+6], *(अचिन्हित *)&diag_buf[i+7]);
            }
        }
    }

}

bool ggml_sycl_flash_attn_ext_supported(पूर्णांक device, स्थिर ggml_tensor * dst) {
    वापस ggml_sycl_get_best_fattn_kernel(device, dst) != BEST_FATTN_KERNEL_NONE;
}

अटल uintptr_t ggml_sycl_fattn_reserve_halves(ggml_sycl_fattn_extra & extra, माप_प्रकार n_halves) {
    अगर (n_halves == 0) {
        वापस 0;
    }
    extra.end = GGML_PAD(extra.end, SYCL_बफर_ALIGNMENT);
    स्थिर uintptr_t block = extra.end;
    extra.end += n_halves * माप(sycl::half);
    वापस block;
}

ggml_sycl_fattn_extra ggml_sycl_fattn_get_extra(स्थिर ggml_tensor * dst) {
    ggml_sycl_fattn_extra extra;

    extra.end = (uintptr_t) dst->तथ्य + ggml_nbytes(dst);

    अगर (dst->op != GGML_OP_FLASH_ATTN_EXT) {
        वापस extra;
    }

    स्थिर ggml_tensor * Q = dst->src[0];
    स्थिर ggml_tensor * K = dst->src[1];
    स्थिर ggml_tensor * V = dst->src[2];
    अगर (!Q || !K || !V) {
        वापस extra;
    }

    स्थिर int64_t d = K->ne[0];
    स्थिर int64_t H = Q->ne[2];
    स्थिर int64_t q = Q->ne[1];

    // calculate the worst-case memory consumption across all kernels
    स्थिर bool onednn_supported = ggml_sycl_flash_attn_ext_onednn_supported(dst, /* use_shape_limit */ false);

    स्थिर bool tile_needs_K = K->type != GGML_TYPE_F16;
    स्थिर bool tile_needs_V = V->type != GGML_TYPE_F16;

    स्थिर bool V_is_K_view = V->view_src &&
        (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));

    माप_प्रकार need_K = 0, need_V = 0, need_Q = 0, need_out = 0, need_scale = 0;
    अगर (onednn_supported) {
        need_Q     = (माप_प्रकार) H * q * d;
        need_out   = (माप_प्रकार) H * q * d;
        need_scale = 1;
        // an f16 cache is bound in place, so it needs no staging copy
        अगर (!ggml_sycl_fattn_onednn_binds_kv(K, V)) {
            need_K = (माप_प्रकार) ggml_nelements(K);
            need_V = (माप_प्रकार) ggml_nelements(V);
        }
    }
    अगर (tile_needs_K) {
        need_K = std::max(need_K, (माप_प्रकार) ggml_nelements(K));
    }
    अगर (tile_needs_V) {
        need_V = std::max(need_V, (माप_प्रकार) ggml_nelements(V));
    }

    extra.Q_बफर_ptr = ggml_sycl_fattn_reserve_halves(extra, need_Q);
    extra.K_बफर_ptr = ggml_sycl_fattn_reserve_halves(extra, need_K);
    extra.V_बफर_ptr = (V_is_K_view && !onednn_supported && need_V)
                       ? extra.K_बफर_ptr
                       : ggml_sycl_fattn_reserve_halves(extra, need_V);
    extra.scale_बफर_ptr = ggml_sycl_fattn_reserve_halves(extra, need_scale);
    extra.out_बफर_ptr   = ggml_sycl_fattn_reserve_halves(extra, need_out);

    वापस extra;
}

माप_प्रकार ggml_sycl_flash_attn_ext_get_alloc_आकार(स्थिर ggml_tensor * dst) {
    स्थिर ggml_sycl_fattn_extra extra = ggml_sycl_fattn_get_extra(dst);
    वापस (माप_प्रकार) (extra.end - (uintptr_t) dst->तथ्य);
}
