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


#অন্তর্ভাব <sycl/sycl.hpp>
#অন্তর্ভাব "dpct/helper.hpp"
#অন্তর্ভাব "common.hpp"
#অন্তর্ভাব "fattn-common.hpp"
#অন্তর্ভাব "fattn-tile.hpp"
#অন্তর্ভাব "fattn-vec.hpp"
#অন্তর্ভাব "fattn.hpp"
#অন্তর্ভাব "fattn-onednn.hpp"
#অন্তর্ভাব "fattn-sparse.hpp"

#define FATTN_VEC_CASE(D, type_K, type_V)                                                                        \
    {                                                                                                            \
        ধ্রুবক যৌক্তিক type_K_okay = K->type == (type_K) || (K->type == GGML_TYPE_F32 && (type_K) == GGML_TYPE_F16); \
        ধ্রুবক যৌক্তিক type_V_okay = V->type == (type_V) || (V->type == GGML_TYPE_F32 && (type_V) == GGML_TYPE_F16); \
        যদি (Q->ne[0] == (D) && type_K_okay && type_V_okay) {                                                     \
            ggml_sycl_flash_attn_ext_vec_case<D, type_K, type_V>(ctx, dst);                                      \
            ফেরত;                                                                                              \
        }                                                                                                        \
    }                                                                    \

#define FATTN_VEC_CASES_ALL_D(type_K, type_V) \
    FATTN_VEC_CASE( 64, type_K, type_V)       \
    FATTN_VEC_CASE(128, type_K, type_V)       \
    FATTN_VEC_CASE(256, type_K, type_V)       \
    FATTN_VEC_CASE(512, type_K, type_V)       \

স্থির শূন্য ggml_sycl_flash_attn_ext_vec(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
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
#অন্যথা
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
#endif // GGML_SYCL_FA_ALL_QUANTS

    GGML_ABORT("Not match KV type in vec");
}

// Best FlashAttention kernel for a specific GPU:
গণনা best_fattn_kernel {
    BEST_FATTN_KERNEL_NONE     =   0,
    BEST_FATTN_KERNEL_VEC      = 100,
    BEST_FATTN_KERNEL_ONEDNN   = 150, // oneDNN SDPA: native F16 (PR #25222)
    BEST_FATTN_KERNEL_TILE     = 200,
    BEST_FATTN_KERNEL_MKL      = 300,
};


স্থির best_fattn_kernel ggml_sycl_get_best_fattn_kernel(ধ্রুবক পূর্ণসংখ্যা device, ধ্রুবক ggml_tensor * dst) {
#ifndef SYCL_FLASH_ATTN
    GGML_UNUSED(dst);
    ফেরত BEST_FATTN_KERNEL_NONE;
#endif// SYCL_FLASH_ATTN

    যদি(!g_ggml_sycl_enable_flash_attention) ফেরত BEST_FATTN_KERNEL_NONE;

    ধ্রুবক ggml_tensor * KQV   = dst;
    ধ্রুবক ggml_tensor * Q     = dst->src[0];
    ধ্রুবক ggml_tensor * K     = dst->src[1];
    ধ্রুবক ggml_tensor * V     = dst->src[2];
    ধ্রুবক ggml_tensor * mask  = dst->src[3];
    ধ্রুবক ggml_tensor * sinks = dst->src[4];

    ধ্রুবক পূর্ণসংখ্যা gqa_ratio = Q->ne[2] / K->ne[2];
    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);

    দশমিক max_bias = 0.0f;
    memcpy(&max_bias, (ধ্রুবক দশমিক *) KQV->op_params + 1, আকার(দশমিক));

    দশমিক logit_softcap = 0.0f;
    memcpy(&logit_softcap, (ধ্রুবক দশমিক *) KQV->op_params + 2, আকার(দশমিক));

    যৌক্তিক gqa_opt_applies = gqa_ratio >= 2 && mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;

    // XMX-accelerated path: oneDNN SDPA (native F16 and dequant+non-F16).
    // ONEDNN requires min 32 query tokens — short-circuit decode to avoid
    // calling _supported() on every decode FA call.
    যদি (Q->ne[1] >= 32
        && ggml_sycl_flash_attn_ext_onednn_supported(dst)) {
        ফেরত BEST_FATTN_KERNEL_ONEDNN;
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
    ধ্রুবক যৌক্তিক standard_shape = Q->ne[0] >= 64 && Q->ne[0] <= 512 &&
        Q->ne[0] % 64 == 0 && Q->ne[0] == V->ne[0];
    // GLM-4.7 Flash's MLA shape is already expressible by the MKL pipeline:
    // KQ GEMM uses DKQ=576 while VKQ and output use DV=512. Keep this narrow
    // until other mismatched K/V shapes have independent correctness data.
    ধ্রুবক যৌক্তিক glm_mla_shape = Q->ne[0] == 576 && K->ne[0] == 576 &&
        V->ne[0] == 512 && gqa_ratio == 20 &&
        K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16;

    যদি (g_ggml_sycl_enable_mkl_fa == 1 && mask && !sinks && gqa_ratio >= 2 &&
        (standard_shape || glm_mla_shape) &&
        Q->ne[1] >= 32 && K->ne[1] >= 1024 &&
        max_bias == 0.0f && logit_softcap == 0.0f &&
        (Q->ne[3] == K->ne[3] || K->ne[3] == 1)) {
        // F16 K/V strides must be a multiple of ne[0]*2 (the natural row size
        // in bytes). This passes both dense (nb1 == ne0*2) and interleaved
        // (nb1 == H * ne0*2). Only pathological test strides like nb1=32 or
        // nb1=75 for ne0=40 fall through to TILE.
        যৌক্তিক kv_strides_ok = সত্য;
        জন্য (ধ্রুবক ggml_tensor * t : {K, V}) {
            ধ্রুবক যৌক্তিক glm_v_stride = glm_mla_shape && t == V &&
                V->view_src && V->nb[1] == K->nb[1];
            যদি (!glm_v_stride && t->type == GGML_TYPE_F16 &&
                    t->nb[1] % (t->ne[0] * 2) != 0) {
                kv_strides_ok = মিথ্যা;
                বিরতি;
            }
        }
        যদি (kv_strides_ok) {
            ফেরত BEST_FATTN_KERNEL_MKL;
        }
    }
    জন্য (ধ্রুবক ggml_tensor * t : {Q, K, V, mask}) {
        যদি (t == nullptr || ggml_is_quantized(t->type)) {
            চালু;
        }
        জন্য (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            যদি (t->nb[i] % 16 != 0) {
                gqa_opt_applies = মিথ্যা;
                বিরতি;
            }
        }
    }

    নির্বাচন (K->ne[0]) {
        ক্ষেত্র  40:
        ক্ষেত্র  64:
        ক্ষেত্র  72:
        ক্ষেত্র  80:
        ক্ষেত্র  96:
        ক্ষেত্র 128:
        ক্ষেত্র 112:
        ক্ষেত্র 256:
        ক্ষেত্র 512:
            যদি (V->ne[0] != K->ne[0]) {
                ফেরত BEST_FATTN_KERNEL_NONE;
            }
            বিরতি;
        ক্ষেত্র 576:
            যদি (V->ne[0] != 512) {
                ফেরত BEST_FATTN_KERNEL_NONE;
            }
            যদি (!gqa_opt_applies) {
                ফেরত BEST_FATTN_KERNEL_NONE;
            }
            বিরতি;
        পূর্বনির্ধারিত:
            ফেরত BEST_FATTN_KERNEL_NONE;
    }

#ifndef GGML_SYCL_FA_ALL_QUANTS
    যদি (K->type != V->type) {
        ফেরত BEST_FATTN_KERNEL_NONE;
    }
#endif // GGML_SYCL_FA_ALL_QUANTS

    নির্বাচন (K->type) {
        ক্ষেত্র GGML_TYPE_F32:
        ক্ষেত্র GGML_TYPE_F16:
        ক্ষেত্র GGML_TYPE_BF16:
            বিরতি;
        ক্ষেত্র GGML_TYPE_Q4_1:
        ক্ষেত্র GGML_TYPE_Q5_0:
        ক্ষেত্র GGML_TYPE_Q5_1:
#ifndef GGML_SYCL_FA_ALL_QUANTS
            ফেরত BEST_FATTN_KERNEL_NONE;
#endif // GGML_SYCL_FA_ALL_QUANTS
        ক্ষেত্র GGML_TYPE_Q4_0:
        ক্ষেত্র GGML_TYPE_Q8_0:
            বিরতি;
        পূর্বনির্ধারিত:
            ফেরত BEST_FATTN_KERNEL_NONE;
    }

    যদি (mask && mask->ne[2] != 1) {
        ফেরত BEST_FATTN_KERNEL_NONE;
    }

    // For small batch sizes the vector kernel may be preferable over the kernels optimized for large batch sizes.
    // BF16 is excluded: the VEC kernel has no BF16 template (it needs GGML_SYCL_FA_ALL_QUANTS for non-F16/Q4_0/Q8_0).
    ধ্রুবক যৌক্তিক has_bf16 = (K->type == GGML_TYPE_BF16 || V->type == GGML_TYPE_BF16);
    ধ্রুবক যৌক্তিক can_use_vector_kernel = Q->ne[0] <= 512 && Q->ne[0] % 64 == 0 && K->ne[1] % FATTN_KQ_STRIDE == 0
        && !has_bf16;

    // Fused-XMX path: oneDNN Graph SDPA (flash attention). Strictly
    // additive -- taken only when statically supported, otherwise falls through to VEC/TILE below.
    যদি (ggml_sycl_flash_attn_ext_onednn_supported(dst)) {
        ফেরত BEST_FATTN_KERNEL_ONEDNN;
    }

    // If there are no tensor cores available, use the generic tile kernel:
    যদি (can_use_vector_kernel) {
        যদি (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
            যদি (Q->ne[1] == 1) {
                যদি (!gqa_opt_applies) {
                    ফেরত BEST_FATTN_KERNEL_VEC;
                }
            }
        } অন্যথা {
            যদি (Q->ne[1] <= 2) {
                // TILE is faster for quantized KV decode on Xe2 (BMG); keep VEC on untested archs
                ধ্রুবক gpu_arch arch = ggml_sycl_info().devices[device].hw_info.arch;
                যদি (arch == gpu_arch::intel_gpu_bmg_g21 || arch == gpu_arch::intel_gpu_bmg_g31) {
                    ফেরত BEST_FATTN_KERNEL_TILE;
                }
                ফেরত BEST_FATTN_KERNEL_VEC;
            }
        }
    }
    ফেরত BEST_FATTN_KERNEL_TILE;
}

শূন্য ggml_sycl_flash_attn_ext(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_set_device(ctx.device);

    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];
    যদি (K && Q && K->ne[0] != Q->ne[0] && Q->ne[0] > 0 && K->ne[0] % Q->ne[0] == 0) {
        ধ্রুবক int64_t d = Q->ne[0];
        ধ্রুবক int64_t n_heads = (K->ne[0] * K->ne[2]) / d;
        K->ne[0] = d;
        K->ne[2] = n_heads;
    }
    যদি (V && Q && V->ne[0] != Q->ne[0] && Q->ne[0] > 0 && V->ne[0] % Q->ne[0] == 0) {
        ধ্রুবক int64_t d = Q->ne[0];
        ধ্রুবক int64_t n_heads = (V->ne[0] * V->ne[2]) / d;
        V->ne[0] = d;
        V->ne[2] = n_heads;
    }

    // sparse nodes are gathered down to n_kv_max rows and re-dispatched here
    যদি (ggml_sycl_flash_attn_ext_sparse(ctx, dst)) {
        ফেরত;
    }

    // n_kv watchdog: log when n_kv differs from the last FA call with
    // the same D — helps detect cache-truncation issues.
    স্থির পূর্ণসংখ্যা nkv_debug = ggml_sycl_get_env("GGML_SYCL_MKL_FA_DEBUG", 0);
    যদি (nkv_debug == 1) {
        ধ্রুবক ggml_tensor * K_dbg = dst->src[1];
        ধ্রুবক ggml_tensor * V_dbg = dst->src[2];
        স্থির int64_t last_nkv_d256 = 0, last_nkv_d512 = 0;
        স্থির পূর্ণসংখ্যা fa_call_seq = 0;
        fa_call_seq++;
        int64_t cur_nkv = K_dbg->ne[1];
        পূর্ণসংখ্যা Dk = (পূর্ণসংখ্যা)K_dbg->ne[0];
        ধ্রুবক অক্ষর * kname = "TILE";
        best_fattn_kernel k = ggml_sycl_get_best_fattn_kernel(ctx.device, dst);
        যদি (k == BEST_FATTN_KERNEL_MKL)  kname = "MKL";
        যদি (k == BEST_FATTN_KERNEL_ONEDNN)  kname = "ONEDNN";
        যদি (k == BEST_FATTN_KERNEL_VEC)  kname = "VEC";
        int64_t delta = 0;
        যদি (Dk == 256) {
            delta = cur_nkv - last_nkv_d256;
            last_nkv_d256 = cur_nkv;
        } অন্যথা যদি (Dk == 512) {
            delta = cur_nkv - last_nkv_d512;
            last_nkv_d512 = cur_nkv;
        }
        GGML_LOG_INFO("[FA-DISP] #%d %s D=%d n_kv=%lld delta=%lld "
                "V_ne1=%lld\n",
                fa_call_seq, kname, Dk,
                (long long)cur_nkv, (long long)delta,
                (long long)V_dbg->ne[1]);
    }

    ধ্রুবক best_fattn_kernel fk = ggml_sycl_get_best_fattn_kernel(ggml_sycl_get_device(), dst);
    নির্বাচন (fk) {
        ক্ষেত্র BEST_FATTN_KERNEL_NONE:
            GGML_ABORT("Not support Flash-Attention");
        ক্ষেত্র BEST_FATTN_KERNEL_ONEDNN:
            // guarded: ggml_sycl_flash_attn_ext_onednn() is only defined under GGML_SYCL_DNNL;
            // the reference must be compiled out here or the GGML_SYCL_DNNL=0 build fails to link.
#যদি GGML_SYCL_DNNL
            ggml_sycl_flash_attn_ext_onednn(ctx, dst);
#endif
            বিরতি;
        ক্ষেত্র BEST_FATTN_KERNEL_TILE:
            ggml_sycl_flash_attn_ext_tile(ctx, dst);
            বিরতি;
        ক্ষেত্র BEST_FATTN_KERNEL_VEC:
            ggml_sycl_flash_attn_ext_vec(ctx, dst);
            বিরতি;
        ক্ষেত্র BEST_FATTN_KERNEL_MKL:
            ggml_sycl_flash_attn_ext_mkl(ctx, dst);
            বিরতি;
    }

    // --- Output fingerprint (GGML_SYCL_MKL_FA_DIAG=1) ---
    // Copy first 64 float output values to host for fingerprinting.
    // Compare MKL vs TILE (GGML_SYCL_ENABLE_MKL_FA=0) to detect divergence.
    // Only fingerprints the first 6 FA calls with n_kv >= 1024.
    স্থির পূর্ণসংখ্যা fa_diag = ggml_sycl_get_env("GGML_SYCL_MKL_FA_DIAG", 0);
    স্থির পূর্ণসংখ্যা fa_diag_গুনতি = 0;
    যদি (fa_diag == 1 && fa_diag_গুনতি < 6) {
        ধ্রুবক ggml_tensor * K_diag = dst->src[1];
        ধ্রুবক ggml_tensor * V_diag = dst->src[2];
        ধ্রুবক ggml_tensor * Q_diag = dst->src[0];
        যদি (K_diag->ne[1] >= 1024) {
            fa_diag_গুনতি++;
            দশমিক diag_buf[64];
            dpct::queue_ptr q = ctx.stream();
            q->memcpy(diag_buf, dst->তথ্য, 64 * আকার(দশমিক));
            q->wait();
            ধ্রুবক অক্ষর * kname = "???";
            best_fattn_kernel kb = ggml_sycl_get_best_fattn_kernel(ctx.device, dst);
            যদি (kb == BEST_FATTN_KERNEL_ONEDNN) kname = "ONEDNN";
            যদি (kb == BEST_FATTN_KERNEL_MKL) kname = "MKL";
            যদি (kb == BEST_FATTN_KERNEL_TILE) kname = "TILE";
            যদি (kb == BEST_FATTN_KERNEL_VEC) kname = "VEC";
            GGML_LOG_INFO("[FA-DIAG] #%d %s D=%d n_kv=%lld n_q=%lld "
                    "n_qh=%lld n_kvh=%lld K=%s V=%s "
                    "nb1=%zu nb2=%zu first 64 floats:\n",
                    fa_diag_গুনতি, kname,
                    (পূর্ণসংখ্যা)K_diag->ne[0], (long long)K_diag->ne[1],
                    (long long)Q_diag->ne[1],
                    (long long)Q_diag->ne[2], (long long)K_diag->ne[2],
                    ggml_type_name(K_diag->type),
                    ggml_type_name(V_diag->type),
                    K_diag->nb[1], K_diag->nb[2]);
            জন্য (পূর্ণসংখ্যা i = 0; i < 64; i += 8) {
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

যৌক্তিক ggml_sycl_flash_attn_ext_supported(পূর্ণসংখ্যা device, ধ্রুবক ggml_tensor * dst) {
    ফেরত ggml_sycl_get_best_fattn_kernel(device, dst) != BEST_FATTN_KERNEL_NONE;
}

স্থির uintptr_t ggml_sycl_fattn_reserve_halves(ggml_sycl_fattn_extra & extra, size_t n_halves) {
    যদি (n_halves == 0) {
        ফেরত 0;
    }
    extra.end = GGML_PAD(extra.end, SYCL_বাফার_ALIGNMENT);
    ধ্রুবক uintptr_t block = extra.end;
    extra.end += n_halves * আকার(sycl::half);
    ফেরত block;
}

ggml_sycl_fattn_extra ggml_sycl_fattn_get_extra(ধ্রুবক ggml_tensor * dst) {
    ggml_sycl_fattn_extra extra;

    extra.end = (uintptr_t) dst->তথ্য + ggml_nbytes(dst);

    যদি (dst->op != GGML_OP_FLASH_ATTN_EXT) {
        ফেরত extra;
    }

    ধ্রুবক ggml_tensor * Q = dst->src[0];
    ধ্রুবক ggml_tensor * K = dst->src[1];
    ধ্রুবক ggml_tensor * V = dst->src[2];
    যদি (!Q || !K || !V) {
        ফেরত extra;
    }

    ধ্রুবক int64_t d = K->ne[0];
    ধ্রুবক int64_t H = Q->ne[2];
    ধ্রুবক int64_t q = Q->ne[1];

    // calculate the worst-case memory consumption across all kernels
    ধ্রুবক যৌক্তিক onednn_supported = ggml_sycl_flash_attn_ext_onednn_supported(dst, /* use_shape_limit */ মিথ্যা);

    ধ্রুবক যৌক্তিক tile_needs_K = K->type != GGML_TYPE_F16;
    ধ্রুবক যৌক্তিক tile_needs_V = V->type != GGML_TYPE_F16;

    ধ্রুবক যৌক্তিক V_is_K_view = V->view_src &&
        (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));

    size_t need_K = 0, need_V = 0, need_Q = 0, need_out = 0, need_scale = 0;
    যদি (onednn_supported) {
        need_Q     = (size_t) H * q * d;
        need_out   = (size_t) H * q * d;
        need_scale = 1;
        // an f16 cache is bound in place, so it needs no staging copy
        যদি (!ggml_sycl_fattn_onednn_binds_kv(K, V)) {
            need_K = (size_t) ggml_nelements(K);
            need_V = (size_t) ggml_nelements(V);
        }
    }
    যদি (tile_needs_K) {
        need_K = std::max(need_K, (size_t) ggml_nelements(K));
    }
    যদি (tile_needs_V) {
        need_V = std::max(need_V, (size_t) ggml_nelements(V));
    }

    extra.Q_বাফার_ptr = ggml_sycl_fattn_reserve_halves(extra, need_Q);
    extra.K_বাফার_ptr = ggml_sycl_fattn_reserve_halves(extra, need_K);
    extra.V_বাফার_ptr = (V_is_K_view && !onednn_supported && need_V)
                       ? extra.K_বাফার_ptr
                       : ggml_sycl_fattn_reserve_halves(extra, need_V);
    extra.scale_বাফার_ptr = ggml_sycl_fattn_reserve_halves(extra, need_scale);
    extra.out_বাফার_ptr   = ggml_sycl_fattn_reserve_halves(extra, need_out);

    ফেরত extra;
}

size_t ggml_sycl_flash_attn_ext_get_alloc_আকার(ধ্রুবক ggml_tensor * dst) {
    ধ্রুবক ggml_sycl_fattn_extra extra = ggml_sycl_fattn_get_extra(dst);
    ফেরত (size_t) (extra.end - (uintptr_t) dst->তথ্য);
}
