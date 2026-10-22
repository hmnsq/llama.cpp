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


#شامل <sycl/sycl.hpp>
#شامل "dpct/helper.hpp"
#شامل "common.hpp"
#شامل "fattn-common.hpp"
#شامل "fattn-tile.hpp"
#شامل "fattn-vec.hpp"
#شامل "fattn.hpp"
#شامل "fattn-onednn.hpp"
#شامل "fattn-sparse.hpp"

#تعریف_کرو FATTN_VEC_CASE(D, type_K, type_V)                                                                        \
    {                                                                                                            \
        ثابت منطقی type_K_okay = K->type == (type_K) || (K->type == GGML_TYPE_F32 && (type_K) == GGML_TYPE_F16); \
        ثابت منطقی type_V_okay = V->type == (type_V) || (V->type == GGML_TYPE_F32 && (type_V) == GGML_TYPE_F16); \
        اگر (Q->ne[0] == (D) && type_K_okay && type_V_okay) {                                                     \
            ggml_sycl_flash_attn_ext_vec_case<D, type_K, type_V>(ctx, dst);                                      \
            واپس;                                                                                              \
        }                                                                                                        \
    }                                                                    \

#تعریف_کرو FATTN_VEC_CASES_ALL_D(type_K, type_V) \
    FATTN_VEC_CASE( 64, type_K, type_V)       \
    FATTN_VEC_CASE(128, type_K, type_V)       \
    FATTN_VEC_CASE(256, type_K, type_V)       \
    FATTN_VEC_CASE(512, type_K, type_V)       \

ساکن خالی ggml_sycl_flash_attn_ext_vec(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];

#اگر_تعریف_شدہ GGML_SYCL_FA_ALL_QUANTS
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
#ورنہ
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
#اگر_ختم // GGML_SYCL_FA_ALL_QUANTS

    GGML_ABORT("Not match KV type in vec");
}

// Best FlashAttention kernel for a specific GPU:
شمار بہترین_fattn_kernel {
    بہترین_FATTN_KERNEL_NONE     =   0,
    بہترین_FATTN_KERNEL_VEC      = 100,
    بہترین_FATTN_KERNEL_ONEDNN   = 150, // oneDNN SDPA: native F16 (PR #25222)
    بہترین_FATTN_KERNEL_TILE     = 200,
    بہترین_FATTN_KERNEL_MKL      = 300,
};


ساکن بہترین_fattn_kernel ggml_sycl_get_بہترین_fattn_kernel(ثابت صحیح device, ثابت ggml_tensor * dst) {
#اگر_غیر_تعریف_شدہ SYCL_FLASH_ATTN
    GGML_غیر_مستعمل(dst);
    واپس بہترین_FATTN_KERNEL_NONE;
#اگر_ختم// SYCL_FLASH_ATTN

    اگر(!g_ggml_sycl_enable_flash_attention) واپس بہترین_FATTN_KERNEL_NONE;

    ثابت ggml_tensor * KQV   = dst;
    ثابت ggml_tensor * Q     = dst->src[0];
    ثابت ggml_tensor * K     = dst->src[1];
    ثابت ggml_tensor * V     = dst->src[2];
    ثابت ggml_tensor * نقاب  = dst->src[3];
    ثابت ggml_tensor * sinks = dst->src[4];

    ثابت صحیح gqa_تناسب = Q->ne[2] / K->ne[2];
    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);

    اعشاری max_bias = 0.0f;
    حافظہ_نقل(&max_bias, (ثابت اعشاری *) KQV->op_params + 1, حجم(اعشاری));

    اعشاری logit_softcap = 0.0f;
    حافظہ_نقل(&logit_softcap, (ثابت اعشاری *) KQV->op_params + 2, حجم(اعشاری));

    منطقی gqa_opt_applies = gqa_تناسب >= 2 && نقاب && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;

    // XMX-accelerated path: oneDNN SDPA (native F16 and dequant+non-F16).
    // ONEDNN requires min 32 query tokens — short-circuit decode to avoid
    // calling _supported() on every decode FA call.
    اگر (Q->ne[1] >= 32
        && ggml_sycl_flash_attn_ext_onednn_supported(dst)) {
        واپس بہترین_FATTN_KERNEL_ONEDNN;
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
    ثابت منطقی standard_shape = Q->ne[0] >= 64 && Q->ne[0] <= 512 &&
        Q->ne[0] % 64 == 0 && Q->ne[0] == V->ne[0];
    // GLM-4.7 Flash's MLA shape is already expressible by the MKL pipeline:
    // KQ GEMM uses DKQ=576 while VKQ and output use DV=512. Keep this narrow
    // until other mismatched K/V shapes have independent correctness data.
    ثابت منطقی glm_mla_shape = Q->ne[0] == 576 && K->ne[0] == 576 &&
        V->ne[0] == 512 && gqa_تناسب == 20 &&
        K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16;

    اگر (g_ggml_sycl_enable_mkl_fa == 1 && نقاب && !sinks && gqa_تناسب >= 2 &&
        (standard_shape || glm_mla_shape) &&
        Q->ne[1] >= 32 && K->ne[1] >= 1024 &&
        max_bias == 0.0f && logit_softcap == 0.0f &&
        (Q->ne[3] == K->ne[3] || K->ne[3] == 1)) {
        // F16 K/V strides must be a multiple of ne[0]*2 (the natural row size
        // in bytes). This passes both dense (nb1 == ne0*2) and interleaved
        // (nb1 == H * ne0*2). Only pathological test strides like nb1=32 or
        // nb1=75 for ne0=40 fall through to TILE.
        منطقی kv_strides_ok = سچ;
        برائے (ثابت ggml_tensor * t : {K, V}) {
            ثابت منطقی glm_v_stride = glm_mla_shape && t == V &&
                V->view_src && V->nb[1] == K->nb[1];
            اگر (!glm_v_stride && t->type == GGML_TYPE_F16 &&
                    t->nb[1] % (t->ne[0] * 2) != 0) {
                kv_strides_ok = جھوٹ;
                توڑو;
            }
        }
        اگر (kv_strides_ok) {
            واپس بہترین_FATTN_KERNEL_MKL;
        }
    }
    برائے (ثابت ggml_tensor * t : {Q, K, V, نقاب}) {
        اگر (t == خالی_اشاریہ || ggml_is_quantized(t->type)) {
            جاری;
        }
        برائے (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            اگر (t->nb[i] % 16 != 0) {
                gqa_opt_applies = جھوٹ;
                توڑو;
            }
        }
    }

    انتخاب (K->ne[0]) {
        حالت  40:
        حالت  64:
        حالت  72:
        حالت  80:
        حالت  96:
        حالت 128:
        حالت 112:
        حالت 256:
        حالت 512:
            اگر (V->ne[0] != K->ne[0]) {
                واپس بہترین_FATTN_KERNEL_NONE;
            }
            توڑو;
        حالت 576:
            اگر (V->ne[0] != 512) {
                واپس بہترین_FATTN_KERNEL_NONE;
            }
            اگر (!gqa_opt_applies) {
                واپس بہترین_FATTN_KERNEL_NONE;
            }
            توڑو;
        طےشدہ:
            واپس بہترین_FATTN_KERNEL_NONE;
    }

#اگر_غیر_تعریف_شدہ GGML_SYCL_FA_ALL_QUANTS
    اگر (K->type != V->type) {
        واپس بہترین_FATTN_KERNEL_NONE;
    }
#اگر_ختم // GGML_SYCL_FA_ALL_QUANTS

    انتخاب (K->type) {
        حالت GGML_TYPE_F32:
        حالت GGML_TYPE_F16:
        حالت GGML_TYPE_BF16:
            توڑو;
        حالت GGML_TYPE_Q4_1:
        حالت GGML_TYPE_Q5_0:
        حالت GGML_TYPE_Q5_1:
#اگر_غیر_تعریف_شدہ GGML_SYCL_FA_ALL_QUANTS
            واپس بہترین_FATTN_KERNEL_NONE;
#اگر_ختم // GGML_SYCL_FA_ALL_QUANTS
        حالت GGML_TYPE_Q4_0:
        حالت GGML_TYPE_Q8_0:
            توڑو;
        طےشدہ:
            واپس بہترین_FATTN_KERNEL_NONE;
    }

    اگر (نقاب && نقاب->ne[2] != 1) {
        واپس بہترین_FATTN_KERNEL_NONE;
    }

    // For small batch sizes the vector kernel may be preferable over the kernels optimized for large batch sizes.
    // BF16 is excluded: the VEC kernel has no BF16 template (it needs GGML_SYCL_FA_ALL_QUANTS for non-F16/Q4_0/Q8_0).
    ثابت منطقی has_bf16 = (K->type == GGML_TYPE_BF16 || V->type == GGML_TYPE_BF16);
    ثابت منطقی can_use_سمتیہ_kernel = Q->ne[0] <= 512 && Q->ne[0] % 64 == 0 && K->ne[1] % FATTN_KQ_STRIDE == 0
        && !has_bf16;

    // Fused-XMX path: oneDNN Graph SDPA (flash attention). Strictly
    // additive -- taken only when statically supported, otherwise falls through to VEC/TILE below.
    اگر (ggml_sycl_flash_attn_ext_onednn_supported(dst)) {
        واپس بہترین_FATTN_KERNEL_ONEDNN;
    }

    // If there are no tensor cores available, use the generic tile kernel:
    اگر (can_use_سمتیہ_kernel) {
        اگر (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
            اگر (Q->ne[1] == 1) {
                اگر (!gqa_opt_applies) {
                    واپس بہترین_FATTN_KERNEL_VEC;
                }
            }
        } ورنہ {
            اگر (Q->ne[1] <= 2) {
                // TILE is faster for quantized KV decode on Xe2 (BMG); keep VEC on untested archs
                ثابت gpu_arch arch = ggml_sycl_info().devices[device].hw_info.arch;
                اگر (arch == gpu_arch::intel_gpu_bmg_g21 || arch == gpu_arch::intel_gpu_bmg_g31) {
                    واپس بہترین_FATTN_KERNEL_TILE;
                }
                واپس بہترین_FATTN_KERNEL_VEC;
            }
        }
    }
    واپس بہترین_FATTN_KERNEL_TILE;
}

خالی ggml_sycl_flash_attn_ext(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_سیٹ_device(ctx.device);

    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];
    اگر (K && Q && K->ne[0] != Q->ne[0] && Q->ne[0] > 0 && K->ne[0] % Q->ne[0] == 0) {
        ثابت int64_t d = Q->ne[0];
        ثابت int64_t n_heads = (K->ne[0] * K->ne[2]) / d;
        K->ne[0] = d;
        K->ne[2] = n_heads;
    }
    اگر (V && Q && V->ne[0] != Q->ne[0] && Q->ne[0] > 0 && V->ne[0] % Q->ne[0] == 0) {
        ثابت int64_t d = Q->ne[0];
        ثابت int64_t n_heads = (V->ne[0] * V->ne[2]) / d;
        V->ne[0] = d;
        V->ne[2] = n_heads;
    }

    // sparse nodes are gathered down to n_kv_max rows and re-dispatched here
    اگر (ggml_sycl_flash_attn_ext_sparse(ctx, dst)) {
        واپس;
    }

    // n_kv watchdog: log when n_kv differs from the last FA call with
    // the same D — helps detect cache-truncation issues.
    ساکن صحیح nkv_debug = ggml_sycl_get_env("GGML_SYCL_MKL_FA_DEBUG", 0);
    اگر (nkv_debug == 1) {
        ثابت ggml_tensor * K_dbg = dst->src[1];
        ثابت ggml_tensor * V_dbg = dst->src[2];
        ساکن int64_t آخری_nkv_d256 = 0, آخری_nkv_d512 = 0;
        ساکن صحیح fa_call_seq = 0;
        fa_call_seq++;
        int64_t cur_nkv = K_dbg->ne[1];
        صحیح Dk = (صحیح)K_dbg->ne[0];
        ثابت حرف * kname = "TILE";
        بہترین_fattn_kernel k = ggml_sycl_get_بہترین_fattn_kernel(ctx.device, dst);
        اگر (k == بہترین_FATTN_KERNEL_MKL)  kname = "MKL";
        اگر (k == بہترین_FATTN_KERNEL_ONEDNN)  kname = "ONEDNN";
        اگر (k == بہترین_FATTN_KERNEL_VEC)  kname = "VEC";
        int64_t delta = 0;
        اگر (Dk == 256) {
            delta = cur_nkv - آخری_nkv_d256;
            آخری_nkv_d256 = cur_nkv;
        } ورنہ اگر (Dk == 512) {
            delta = cur_nkv - آخری_nkv_d512;
            آخری_nkv_d512 = cur_nkv;
        }
        GGML_LOG_INFO("[FA-DISP] #%d %s D=%d n_kv=%lld delta=%lld "
                "V_ne1=%lld\n",
                fa_call_seq, kname, Dk,
                (طویل طویل)cur_nkv, (طویل طویل)delta,
                (طویل طویل)V_dbg->ne[1]);
    }

    ثابت بہترین_fattn_kernel fk = ggml_sycl_get_بہترین_fattn_kernel(ggml_sycl_get_device(), dst);
    انتخاب (fk) {
        حالت بہترین_FATTN_KERNEL_NONE:
            GGML_ABORT("Not support Flash-Attention");
        حالت بہترین_FATTN_KERNEL_ONEDNN:
            // guarded: ggml_sycl_flash_attn_ext_onednn() is only defined under GGML_SYCL_DNNL;
            // the reference must be compiled out here or the GGML_SYCL_DNNL=0 build fails to link.
#اگر GGML_SYCL_DNNL
            ggml_sycl_flash_attn_ext_onednn(ctx, dst);
#اگر_ختم
            توڑو;
        حالت بہترین_FATTN_KERNEL_TILE:
            ggml_sycl_flash_attn_ext_tile(ctx, dst);
            توڑو;
        حالت بہترین_FATTN_KERNEL_VEC:
            ggml_sycl_flash_attn_ext_vec(ctx, dst);
            توڑو;
        حالت بہترین_FATTN_KERNEL_MKL:
            ggml_sycl_flash_attn_ext_mkl(ctx, dst);
            توڑو;
    }

    // --- Output fingerprint (GGML_SYCL_MKL_FA_DIAG=1) ---
    // Copy first 64 float output values to host for fingerprinting.
    // Compare MKL vs TILE (GGML_SYCL_ENABLE_MKL_FA=0) to detect divergence.
    // Only fingerprints the first 6 FA calls with n_kv >= 1024.
    ساکن صحیح fa_diag = ggml_sycl_get_env("GGML_SYCL_MKL_FA_DIAG", 0);
    ساکن صحیح fa_diag_گنتی = 0;
    اگر (fa_diag == 1 && fa_diag_گنتی < 6) {
        ثابت ggml_tensor * K_diag = dst->src[1];
        ثابت ggml_tensor * V_diag = dst->src[2];
        ثابت ggml_tensor * Q_diag = dst->src[0];
        اگر (K_diag->ne[1] >= 1024) {
            fa_diag_گنتی++;
            اعشاری diag_buf[64];
            dpct::قطار_ptr q = ctx.stream();
            q->حافظہ_نقل(diag_buf, dst->ڈیٹا, 64 * حجم(اعشاری));
            q->wait();
            ثابت حرف * kname = "???";
            بہترین_fattn_kernel kb = ggml_sycl_get_بہترین_fattn_kernel(ctx.device, dst);
            اگر (kb == بہترین_FATTN_KERNEL_ONEDNN) kname = "ONEDNN";
            اگر (kb == بہترین_FATTN_KERNEL_MKL) kname = "MKL";
            اگر (kb == بہترین_FATTN_KERNEL_TILE) kname = "TILE";
            اگر (kb == بہترین_FATTN_KERNEL_VEC) kname = "VEC";
            GGML_LOG_INFO("[FA-DIAG] #%d %s D=%d n_kv=%lld n_q=%lld "
                    "n_qh=%lld n_kvh=%lld K=%s V=%s "
                    "nb1=%zu nb2=%zu first 64 floats:\n",
                    fa_diag_گنتی, kname,
                    (صحیح)K_diag->ne[0], (طویل طویل)K_diag->ne[1],
                    (طویل طویل)Q_diag->ne[1],
                    (طویل طویل)Q_diag->ne[2], (طویل طویل)K_diag->ne[2],
                    ggml_type_name(K_diag->type),
                    ggml_type_name(V_diag->type),
                    K_diag->nb[1], K_diag->nb[2]);
            برائے (صحیح i = 0; i < 64; i += 8) {
                GGML_LOG_INFO("  [%2d] %08x %08x %08x %08x %08x %08x %08x %08x\n",
                        i,
                        *(غیرعلامتی *)&diag_buf[i+0], *(غیرعلامتی *)&diag_buf[i+1],
                        *(غیرعلامتی *)&diag_buf[i+2], *(غیرعلامتی *)&diag_buf[i+3],
                        *(غیرعلامتی *)&diag_buf[i+4], *(غیرعلامتی *)&diag_buf[i+5],
                        *(غیرعلامتی *)&diag_buf[i+6], *(غیرعلامتی *)&diag_buf[i+7]);
            }
        }
    }

}

منطقی ggml_sycl_flash_attn_ext_supported(صحیح device, ثابت ggml_tensor * dst) {
    واپس ggml_sycl_get_بہترین_fattn_kernel(device, dst) != بہترین_FATTN_KERNEL_NONE;
}

ساکن uintptr_t ggml_sycl_fattn_reserve_halves(ggml_sycl_fattn_extra & extra, size_t n_halves) {
    اگر (n_halves == 0) {
        واپس 0;
    }
    extra.end = GGML_PAD(extra.end, SYCL_بفر_ALIGNMENT);
    ثابت uintptr_t بلاک = extra.end;
    extra.end += n_halves * حجم(sycl::half);
    واپس بلاک;
}

ggml_sycl_fattn_extra ggml_sycl_fattn_get_extra(ثابت ggml_tensor * dst) {
    ggml_sycl_fattn_extra extra;

    extra.end = (uintptr_t) dst->ڈیٹا + ggml_nbytes(dst);

    اگر (dst->op != GGML_OP_FLASH_ATTN_EXT) {
        واپس extra;
    }

    ثابت ggml_tensor * Q = dst->src[0];
    ثابت ggml_tensor * K = dst->src[1];
    ثابت ggml_tensor * V = dst->src[2];
    اگر (!Q || !K || !V) {
        واپس extra;
    }

    ثابت int64_t d = K->ne[0];
    ثابت int64_t H = Q->ne[2];
    ثابت int64_t q = Q->ne[1];

    // calculate the worst-case memory consumption across all kernels
    ثابت منطقی onednn_supported = ggml_sycl_flash_attn_ext_onednn_supported(dst, /* use_shape_limit */ جھوٹ);

    ثابت منطقی tile_needs_K = K->type != GGML_TYPE_F16;
    ثابت منطقی tile_needs_V = V->type != GGML_TYPE_F16;

    ثابت منطقی V_is_K_view = V->view_src &&
        (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));

    size_t need_K = 0, need_V = 0, need_Q = 0, need_برآمد = 0, need_scale = 0;
    اگر (onednn_supported) {
        need_Q     = (size_t) H * q * d;
        need_برآمد   = (size_t) H * q * d;
        need_scale = 1;
        // an f16 cache is bound in place, so it needs no staging copy
        اگر (!ggml_sycl_fattn_onednn_binds_kv(K, V)) {
            need_K = (size_t) ggml_nelements(K);
            need_V = (size_t) ggml_nelements(V);
        }
    }
    اگر (tile_needs_K) {
        need_K = std::max(need_K, (size_t) ggml_nelements(K));
    }
    اگر (tile_needs_V) {
        need_V = std::max(need_V, (size_t) ggml_nelements(V));
    }

    extra.Q_بفر_ptr = ggml_sycl_fattn_reserve_halves(extra, need_Q);
    extra.K_بفر_ptr = ggml_sycl_fattn_reserve_halves(extra, need_K);
    extra.V_بفر_ptr = (V_is_K_view && !onednn_supported && need_V)
                       ? extra.K_بفر_ptr
                       : ggml_sycl_fattn_reserve_halves(extra, need_V);
    extra.scale_بفر_ptr = ggml_sycl_fattn_reserve_halves(extra, need_scale);
    extra.برآمد_بفر_ptr   = ggml_sycl_fattn_reserve_halves(extra, need_برآمد);

    واپس extra;
}

size_t ggml_sycl_flash_attn_ext_get_alloc_سائز(ثابت ggml_tensor * dst) {
    ثابت ggml_sycl_fattn_extra extra = ggml_sycl_fattn_get_extra(dst);
    واپس (size_t) (extra.end - (uintptr_t) dst->ڈیٹا);
}
