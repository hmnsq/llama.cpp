#وصف_ڪريو GGML_COMMON_IMPL_CPP
#وصف_ڪريو GGML_COMMON_DECL_CPP

#ضمن "ime.h"

#ضمن "binary-ops.h"
#ضمن "common.h"
#ضمن "ggml-backend-impl.h"
#ضمن "ggml-common.h"
#ضمن "ggml-cpu.h"
#ضمن "ime_env.h"
#ضمن "ime_kernels.h"
#ضمن "ops.h"
#ضمن "repack.h"
#ضمن "rvv_kernels.h"
#ضمن "spine_mem_pool.h"
#ضمن "traits.h"
#ضمن "vec.h"

#ضمن <fcntl.h>
#ضمن <sys/mman.h>
#ضمن <unistd.h>

#ضمن <algorithm>
#ضمن <atomic>
#ضمن <cassert>
#ضمن <cerrno>
#ضمن <cmath>
#ضمن <cstdio>  // for GGML_ASSERT
#ضمن <stdexcept>
#ضمن <thread>
// clang-format off
#إذا defined(__riscv)

#إذا !defined(__riscv_v) || !defined(__riscv_v_intrinsic)
#غلطي "riscv v extension or v_intrinsic not enabled"
#وإلا
#ضمن <riscv_vector.h>
#جيڪڏهن_پورو

#إذا !defined(__riscv_zfh) || !defined(__riscv_zvfh)
#غلطي "riscv zfh extension not enabled, GGML_RV_ZFH and GGML_RV_ZVFH must be defined to 1"
#جيڪڏهن_پورو

#إذا !defined(__riscv_zba)
#غلطي "riscv zba extension not enabled, GGML_RV_ZBA must be defined to 1"
#جيڪڏهن_پورو

#إذا defined(RISCV64_SPACEMIT_IME1) || defined(RISCV64_SPACEMIT_IME2)
#وإلا
#غلطي "RISCV64_SPACEMIT_IME1 or RISCV64_SPACEMIT_IME2 not defined"
#جيڪڏهن_پورو

#وإلا

#غلطي "riscv not enabled in this build"

#جيڪڏهن_پورو

#إذا defined(__GNUC__)
#هدايت GCC diagnostic ignored "-Woverlength-strings"
#هدايت GCC diagnostic ignored "-Wcast-qual"
#هدايت GCC diagnostic ignored "-Wunused-parameter"
#جيڪڏهن_پورو

// clang-format on

خارجي "C" {
خارجي فراغ ggml_threadpool_chunk_سيٽ(بنية ggml_threadpool * tp, صحيح value);
خارجي صحيح  ggml_threadpool_chunk_جوڙيو(بنية ggml_threadpool * tp, صحيح value);
}

نطاق ggml::cpu::riscv64_spacemit {

بنية TLSContext {
    صحيح       cpu_id{ -1 };
    cpu_set_t cpuset;
    فراغ *    tcm_بفر{ خالي_اشارو };
    size_t    tcm_بفر_سائيز{ 0 };
};

thread_local TLSContext tls_context;

سانچو <قسم_نالو BLOC_TYPE, int64_t INTER_سائيز, int64_t NB_کانن_ڳڻپ> constexpr size_t get_repacked_بلاڪ_type_سائيز() {
    إذا constexpr (std::is_same_v<BLOC_TYPE, بلاڪ_q6_K> || std::is_same_v<BLOC_TYPE, بلاڪ_q8_0>) {
        أرجع حجم(بلاڪ_q8_0);
    } وإلا إذا constexpr (std::is_same_v<BLOC_TYPE, بلاڪ_q4_0>) {
        أرجع حجم(بلاڪ_q4_0) * INTER_سائيز / QK4_0;
    } وإلا إذا constexpr (std::is_same_v<BLOC_TYPE, بلاڪ_q4_1> || std::is_same_v<BLOC_TYPE, بلاڪ_q4_K>) {
        أرجع (حجم(بلاڪ_q4_0) + حجم(uint8_t)) * INTER_سائيز / QK4_1;
    } وإلا إذا constexpr (std::is_same_v<BLOC_TYPE, بلاڪ_q2_K>) {
        أرجع حجم(spacemit_kernels::nrow_بلاڪ_q2_k<1>);
    } وإلا إذا constexpr (std::is_same_v<BLOC_TYPE, بلاڪ_q3_K>) {
        أرجع حجم(spacemit_kernels::nrow_بلاڪ_q3_k<1>);
    } وإلا إذا constexpr (std::is_same_v<BLOC_TYPE, بلاڪ_mxfp4>) {
        أرجع حجم(spacemit_kernels::nrow_بلاڪ_mxfp4<1>);
    } وإلا إذا constexpr (std::is_same_v<BLOC_TYPE, بلاڪ_q5_1> || std::is_same_v<BLOC_TYPE, بلاڪ_q5_K>) {
        أرجع حجم(spacemit_kernels::nrow_بلاڪ_q5_1<1>);
    } وإلا إذا constexpr (std::is_same_v<BLOC_TYPE, بلاڪ_q5_0>) {
        أرجع حجم(spacemit_kernels::nrow_بلاڪ_q5_0<1>);
    } وإلا {
        أكد(خطأ);
        أرجع 0;
    }
}

سانچو <قسم_نالو BLOC_TYPE> constexpr منطقي بلاڪ_type_has_zp() {
    إذا constexpr (std::is_same_v<BLOC_TYPE, بلاڪ_q6_K> || std::is_same_v<BLOC_TYPE, بلاڪ_q8_0> ||
                  std::is_same_v<BLOC_TYPE, بلاڪ_q3_K> || std::is_same_v<BLOC_TYPE, بلاڪ_q4_0> ||
                  std::is_same_v<BLOC_TYPE, بلاڪ_mxfp4> || std::is_same_v<BLOC_TYPE, بلاڪ_q5_0>) {
        أرجع خطأ;
    } وإلا إذا constexpr (std::is_same_v<BLOC_TYPE, بلاڪ_q4_1> || std::is_same_v<BLOC_TYPE, بلاڪ_q4_K> ||
                         std::is_same_v<BLOC_TYPE, بلاڪ_q2_K> || std::is_same_v<BLOC_TYPE, بلاڪ_q5_1> ||
                         std::is_same_v<BLOC_TYPE, بلاڪ_q5_K>) {
        أرجع سچ;
    } وإلا {
        أكد(خطأ);
        أرجع خطأ;
    }
}

صنف tensor_traits_بنياد : عام ggml::cpu::tensor_traits {
  عام:
    مجازي صحيح repack(ggml_tensor * t, ثابت فراغ * معلومات, size_t معلومات_سائيز) = 0;
};

سانچو <قسم_نالو BLOC_TYPE, int64_t INTER_سائيز, int64_t NB_کانن_ڳڻپ> صنف tensor_traits : عام tensor_traits_بنياد {
    منطقي work_سائيز(صحيح /* n_threads */, ثابت ggml_tensor * op, size_t & size) override {
        بدل (op->op) {
            حالة GGML_OP_MUL_MAT:
                {
                    int64_t src1_nelements = ggml_nelements(op->src[1]);

                    إذا constexpr (std::is_same_v<BLOC_TYPE, بلاڪ_q2_K> || std::is_same_v<BLOC_TYPE, بلاڪ_q3_K>) {
                        size =
                            spacemit_kernels::div_ڦيرو_مٿي(src1_nelements, QK_K) * spacemit_kernels::q8k_blk_سائيز(QK_K);
                    } وإلا إذا constexpr (INTER_سائيز == QK4_0) {
                        size = spacemit_kernels::div_ڦيرو_مٿي(src1_nelements, QK4_0) *
                               spacemit_kernels::q8_blk_سائيز(QK4_0, سچ);
                    } وإلا إذا constexpr (INTER_سائيز == 256) {
                        size = spacemit_kernels::div_ڦيرو_مٿي(src1_nelements, 256) *
                               spacemit_kernels::q8_hp_blk_سائيز(256, سچ, سچ);
                    } وإلا {
                        GGML_ABORT("unsupported block type");
                    }

                    size = GGML_PAD(size, حجم(int64_t));

                    أرجع سچ;
                }
            حالة GGML_OP_MUL_MAT_ID:
                {
                    int64_t src1_nelements = ggml_nelements(op->src[1]);

                    إذا constexpr (std::is_same_v<BLOC_TYPE, بلاڪ_q2_K> || std::is_same_v<BLOC_TYPE, بلاڪ_q3_K>) {
                        size =
                            spacemit_kernels::div_ڦيرو_مٿي(src1_nelements, QK_K) * spacemit_kernels::q8k_blk_سائيز(QK_K);
                    } وإلا إذا constexpr (INTER_سائيز == QK4_0) {
                        size = spacemit_kernels::div_ڦيرو_مٿي(src1_nelements, QK4_0) *
                               spacemit_kernels::q8_blk_سائيز(QK4_0, سچ);
                    } وإلا إذا constexpr (INTER_سائيز == 256) {
                        size = spacemit_kernels::div_ڦيرو_مٿي(src1_nelements, 256) *
                               spacemit_kernels::q8_hp_blk_سائيز(256, سچ, سچ);
                    } وإلا {
                        GGML_ABORT("unsupported block type");
                    }

                    size = GGML_PAD(size, حجم(int64_t));

                    ثابت int64_t ne02 = op->src[0]->ne[2];  // n_as, n_expert
                    ثابت int64_t ne12 = op->src[1]->ne[2];  // n_tokens

                    ثابت size_t sizeof_mmid_صف_mapping = حجم(int64_t);
                    size += sizeof_mmid_صف_mapping * ne02 * (ne12 + 1) + (ne02 + 1) * حجم(int64_t);

                    size = GGML_PAD(size, حجم(int64_t));

                    أرجع سچ;
                }
            افتراضي:
                // GGML_ABORT("fatal error");
                اكسر;
        }
        أرجع خطأ;
    }

    منطقي compute_forward(ggml_compute_params * params, ggml_tensor * op) override {
        بدل (op->op) {
            حالة GGML_OP_MUL_MAT:
                بدل (op->src[0]->type) {
                    حالة GGML_TYPE_Q2_K:
                    حالة GGML_TYPE_Q3_K:
                    حالة GGML_TYPE_Q4_0:
                    حالة GGML_TYPE_Q4_1:
                    حالة GGML_TYPE_Q4_K:
                    حالة GGML_TYPE_Q6_K:
                    حالة GGML_TYPE_Q8_0:
                    حالة GGML_TYPE_Q5_0:
                    حالة GGML_TYPE_Q5_1:
                    حالة GGML_TYPE_Q5_K:
                        //case GGML_TYPE_MXFP4:
                        forward_mul_mat(params, op);
                        أرجع سچ;
                    افتراضي:
                        // GGML_ABORT("fatal error: unsupported type for src0 in MUL_MAT");
                        أرجع خطأ;
                }
                اكسر;
            حالة GGML_OP_MUL_MAT_ID:
                بدل (op->src[0]->type) {
                    حالة GGML_TYPE_Q2_K:
                    حالة GGML_TYPE_Q3_K:
                    حالة GGML_TYPE_Q4_0:
                    حالة GGML_TYPE_Q4_1:
                    حالة GGML_TYPE_Q4_K:
                    حالة GGML_TYPE_Q6_K:
                    حالة GGML_TYPE_Q8_0:
                    حالة GGML_TYPE_Q5_0:
                    حالة GGML_TYPE_Q5_1:
                    حالة GGML_TYPE_Q5_K:
                        //case GGML_TYPE_MXFP4:
                        forward_mul_mat_id(params, op);
                        أرجع سچ;
                    افتراضي:
                        // GGML_ABORT("fatal error: unsupported type for src0 in MUL_MAT_ID");
                        أرجع خطأ;
                }
                اكسر;
            افتراضي:
                // GGML_ABORT("fatal error");
                اكسر;
        }
        أرجع خطأ;
    }

    فراغ forward_mul_mat(ggml_compute_params * params, ggml_tensor * op) {
        constexpr size_t a_blk_len = INTER_سائيز;
        constexpr size_t b_blk_len = INTER_سائيز;

        ثابت ggml_tensor * src0 = op->src[0];
        ثابت ggml_tensor * src1 = op->src[1];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_ٻٽو_OP_LOCALS

        صحيح ith = params->ith;
        صحيح nth = params->nth;

        [[maybe_unused]] ثابت تعداد ggml_type type = src0->type;

        فراغ *        w_معلومات  = (فراغ *) src0->معلومات;
        ثابت عائم * feature = (ثابت عائم *) src1->معلومات;
        عائم *       output  = (عائم *) dst->معلومات;

        ثابت int64_t gemm_m = ne11 * ne12 * ne13;
        ثابت int64_t gemm_k = ne10;
        ثابت int64_t gemm_n = ne01;

        spacemit_kernels::quantize_a_صف_def       quantize_a_صف_i8;
        spacemit_kernels::quantize_a_صف_def       quantize_a_4row_i8;
        spacemit_kernels::gemm_kernel_quantize_def gemm_kernel;
        منطقي                                       سيٽ_kernel_impl = خطأ;

        int64_t بلاڪ_stride_a = spacemit_kernels::q8_blk_سائيز(a_blk_len);

#إذا defined(RISCV64_SPACEMIT_IME2)
        إذا (!سيٽ_kernel_impl && (global_spine_env_info.use_ime2)) {
            quantize_a_صف_i8  = spacemit_kernels::rvv::quantize_a_صف_i8;
            quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8;
            بلاڪ_stride_a     = spacemit_kernels::q8_blk_سائيز(a_blk_len, سچ);

            إذا constexpr (std::is_same_v<BLOC_TYPE, بلاڪ_q6_K> || std::is_same_v<BLOC_TYPE, بلاڪ_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i8;
                سيٽ_kernel_impl = سچ;
            } وإلا إذا constexpr (std::is_same_v<BLOC_TYPE, بلاڪ_q4_0> || std::is_same_v<BLOC_TYPE, بلاڪ_q4_1> ||
                                 std::is_same_v<BLOC_TYPE, بلاڪ_q4_K>) {
                إذا constexpr (INTER_سائيز == 256) {
                    gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i4_hp;
                    quantize_a_صف_i8  = spacemit_kernels::rvv::quantize_a_صف_i8_hp;
                    quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8_hp;
                    بلاڪ_stride_a     = spacemit_kernels::q8_hp_blk_سائيز(a_blk_len, سچ, سچ);
                    سيٽ_kernel_impl    = سچ;
                } وإلا {
                    gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i4;
                    quantize_a_صف_i8  = spacemit_kernels::rvv::quantize_a_صف_i8;
                    quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8;
                    بلاڪ_stride_a     = spacemit_kernels::q8_blk_سائيز(a_blk_len, سچ);
                    سيٽ_kernel_impl    = سچ;
                }
            } وإلا إذا constexpr (std::is_same_v<BLOC_TYPE, بلاڪ_q2_K>) {
                quantize_a_صف_i8  = spacemit_kernels::rvv::quantize_a_صف_i8k;
                quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8k;
                بلاڪ_stride_a     = spacemit_kernels::q8k_blk_سائيز(a_blk_len);

                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i2k;
                سيٽ_kernel_impl = سچ;
            } وإلا إذا constexpr (std::is_same_v<BLOC_TYPE, بلاڪ_q3_K>) {
                quantize_a_صف_i8  = spacemit_kernels::rvv::quantize_a_صف_i8k;
                quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8k;
                بلاڪ_stride_a     = spacemit_kernels::q8k_blk_سائيز(a_blk_len);

                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i3k;
                سيٽ_kernel_impl = سچ;
            } وإلا إذا constexpr (std::is_same_v<BLOC_TYPE, بلاڪ_mxfp4>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8mxfp4;
                سيٽ_kernel_impl = سچ;
            } وإلا إذا constexpr (std::is_same_v<BLOC_TYPE, بلاڪ_q5_1> || std::is_same_v<BLOC_TYPE, بلاڪ_q5_K> ||
                                 std::is_same_v<BLOC_TYPE, بلاڪ_q5_0>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i5;
                سيٽ_kernel_impl = سچ;
            }
        }
#جيڪڏهن_پورو

#إذا defined(RISCV64_SPACEMIT_IME1)
        إذا (!سيٽ_kernel_impl && (global_spine_env_info.use_ime1)) {
            quantize_a_صف_i8  = spacemit_kernels::ime1::quantize_a_صف_i8;
            quantize_a_4row_i8 = spacemit_kernels::ime1::quantize_a_4row_i8;

            إذا constexpr (std::is_same_v<BLOC_TYPE, بلاڪ_q4_0> || std::is_same_v<BLOC_TYPE, بلاڪ_q4_1> ||
                          std::is_same_v<BLOC_TYPE, بلاڪ_q4_K>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i4;
                سيٽ_kernel_impl = سچ;
            } وإلا إذا constexpr (std::is_same_v<BLOC_TYPE, بلاڪ_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i8;
                سيٽ_kernel_impl = سچ;
            }
        }
#جيڪڏهن_پورو
        إذا (!سيٽ_kernel_impl) {
            GGML_ABORT("no kernel implementation found for the block type");
        }

        ثابت int64_t a_k_blks = spacemit_kernels::div_ڦيرو_مٿي(gemm_k, a_blk_len);
        ثابت int64_t b_k_blks = spacemit_kernels::div_ڦيرو_مٿي(gemm_k, b_blk_len);

        ثابت int64_t صف_stride_a        = a_k_blks * بلاڪ_stride_a;
        ثابت int64_t gemm_workspace_سائيز = GGML_PAD(gemm_m * صف_stride_a, alignof(int64_t));

        إذا (ith == 0 && params->wsize < gemm_workspace_سائيز) {
            GGML_ABORT("wsize less than gemm_workspace_size");
        }

        uintptr_t ws_ptr = reinterpret_cast<uintptr_t>(params->wdata);

        فراغ *        tcm_بفر      = ggml::cpu::riscv64_spacemit::tls_context.tcm_بفر;
        ثابت int64_t tcm_بفر_سائيز = ggml::cpu::riscv64_spacemit::tls_context.tcm_بفر_سائيز;

        خودڪار * quant_a_بفر = reinterpret_cast<uint8_t *>(ws_ptr);

        constexpr int64_t صف_align = 4;
        ثابت int64_t     صف_blks  = spacemit_kernels::div_ڦيرو_مٿي(gemm_m, صف_align);

        ثابت int64_t صف_stride_b      = b_k_blks * get_repacked_بلاڪ_type_سائيز<BLOC_TYPE, INTER_سائيز, NB_کانن_ڳڻپ>();
        ثابت int64_t في_mb_صفون_wsize = صف_align * صف_stride_a;
        ثابت int64_t في_nb_کانن_ڳڻپ_wsize = NB_کانن_ڳڻپ * صف_stride_b;

        ثابت int64_t barrier_idx = static_cast<int64_t>(ith / 2);

        GGML_ASSERT(global_spine_env_info.init_barrier != خالي_اشارو);
        GGML_ASSERT(barrier_idx < spine_init_barrier_ڳڻپ);
        spine_barrier_t * cur_barrier = &global_spine_env_info.init_barrier[barrier_idx];

        إذا (gemm_m == 1) {
            صحيح task_في_thread = spacemit_kernels::div_ڦيرو_مٿي(a_k_blks, nth);
            صحيح a_blk_شروعات     = ith * task_في_thread;
            صحيح a_blk_پڇاڙي       = std::min(a_blk_شروعات + task_في_thread, (صحيح) a_k_blks);
            إذا (a_blk_شروعات < a_blk_پڇاڙي) {
                quantize_a_صف_i8(a_blk_len, feature + a_blk_شروعات * a_blk_len, (a_blk_پڇاڙي - a_blk_شروعات) * a_blk_len,
                                  quant_a_بفر + a_blk_شروعات * بلاڪ_stride_a);
            }
        } وإلا {
            صحيح task_في_thread = spacemit_kernels::div_ڦيرو_مٿي(صف_blks, nth);
            صحيح m_صف_blk_شروعات = ith * task_في_thread;
            صحيح m_صف_blk_پڇاڙي   = std::min(m_صف_blk_شروعات + task_في_thread, (صحيح) صف_blks);
            لكل (صحيح m_صف_blk = m_صف_blk_شروعات; m_صف_blk < m_صف_blk_پڇاڙي; m_صف_blk++) {
                صحيح m_idx             = m_صف_blk * صف_align;
                صحيح صفون_tobe_handled = (gemm_m - m_idx) > صف_align ? صف_align : (gemm_m - m_idx);

                إذا (صفون_tobe_handled == صف_align && quantize_a_4row_i8 != خالي_اشارو) {
                    ثابت عائم * a_صف_ptr       = feature + m_idx * gemm_k;
                    خودڪار *        quant_a_صف_ptr = quant_a_بفر + m_idx * صف_stride_a;
                    quantize_a_4row_i8(a_blk_len, a_صف_ptr, gemm_k, quant_a_صف_ptr);
                } وإلا {
                    طالما (صفون_tobe_handled) {
                        ثابت عائم * a_صف_ptr       = feature + m_idx * gemm_k;
                        خودڪار *        quant_a_صف_ptr = quant_a_بفر + m_idx * صف_stride_a;
                        quantize_a_صف_i8(a_blk_len, a_صف_ptr, gemm_k, quant_a_صف_ptr);
                        صفون_tobe_handled -= 1;
                        m_idx += 1;
                    }
                }
            }
        }

        ggml_barrier(params->threadpool);

        ثابت int64_t gemm_m_stride     = gemm_n / gemm_m > 64 ? gemm_m : 16;
        ثابت int64_t gemm_m_blocked    = spacemit_kernels::div_ڦيرو_مٿي(gemm_m, gemm_m_stride);
        ثابت int64_t max_gemm_n_stride = spacemit_kernels::div_ڦيرو_مٿي(gemm_n * gemm_m_blocked, nth);

        int64_t gemm_n_stride = gemm_n;
        إذا (max_gemm_n_stride < gemm_n) {
            gemm_n_stride =
                std::min(gemm_n_stride, spacemit_kernels::div_ڦيرو_مٿي(max_gemm_n_stride, NB_کانن_ڳڻپ) * NB_کانن_ڳڻپ);
        }

        إذا (gemm_n_stride == gemm_n && tcm_بفر != خالي_اشارو && في_mb_صفون_wsize <= tcm_بفر_سائيز) {
            لكل (int64_t m_شروعات = ith * صف_align; m_شروعات < gemm_m; m_شروعات += صف_align * nth) {
                uint8_t * b_کانو    = reinterpret_cast<uint8_t *>(w_معلومات);
                uint8_t * b_کانو_zp = بلاڪ_type_has_zp<BLOC_TYPE>() ? b_کانو : خالي_اشارو;

                int64_t m_صف_real = std::min(gemm_m - m_شروعات, صف_align);

                spacemit_kernels::rvv::memcpy1d(tcm_بفر, quant_a_بفر + m_شروعات * صف_stride_a,
                                                m_صف_real * صف_stride_a);

                int64_t n_blk_real = 0;
                لكل (int64_t ni = 0; ni < gemm_n; ni += n_blk_real, b_کانو += n_blk_real * صف_stride_b) {
                    n_blk_real = std::min(gemm_n - ni, (int64_t) NB_کانن_ڳڻپ);

                    uint8_t * a_صف_ptr = (uint8_t *) tcm_بفر;
                    عائم *   c_blk     = output + m_شروعات * gemm_n + ni;

                    int32_t صفون_بچيل = m_صف_real;

                    طالما (صفون_بچيل > 0) {
                        خودڪار صفون_handled = gemm_kernel(b_blk_len, a_صف_ptr, b_کانو, b_کانو_zp, c_blk, صفون_بچيل,
                                                        n_blk_real, b_k_blks, gemm_n);

                        c_blk += صفون_handled * gemm_n;
                        a_صف_ptr += صفون_handled * صف_stride_a;

                        صفون_بچيل -= صفون_handled;
                    }
                }
            }
        } وإلا إذا (tcm_بفر != خالي_اشارو && في_nb_کانن_ڳڻپ_wsize <= tcm_بفر_سائيز) {
            uint8_t * a_صف = quant_a_بفر;
            uint8_t * b_کانو = reinterpret_cast<uint8_t *>(tcm_بفر);
            إذا ((gemm_workspace_سائيز + في_nb_کانن_ڳڻپ_wsize) <= tcm_بفر_سائيز) {
                a_صف = (uint8_t *) tcm_بفر;
                b_کانو = reinterpret_cast<uint8_t *>(tcm_بفر) + gemm_workspace_سائيز;
            }
            uint8_t * b_کانو_zp = بلاڪ_type_has_zp<BLOC_TYPE>() ? b_کانو : خالي_اشارو;

            int64_t ni      = ith * NB_کانن_ڳڻپ;
            int64_t nb_real = std::min(gemm_n - ni, NB_کانن_ڳڻپ);

            إذا (ith % 2 == 0 && nb_real > 0) {
                spacemit_kernels::rvv::memcpy1d(b_کانو, reinterpret_cast<uint8_t *>(w_معلومات) + ni * صف_stride_b,
                                                nb_real * صف_stride_b);
                إذا (a_صف != quant_a_بفر) {
                    spacemit_kernels::rvv::memcpy1d(a_صف, quant_a_بفر, gemm_workspace_سائيز);
                }
            }

            spine_barrier_wait(cur_barrier);

            إذا (ith % 2 != 0 && nb_real > 0) {
                إذا (a_صف != quant_a_بفر) {
                    spacemit_kernels::rvv::memcpy1d(a_صف, quant_a_بفر, gemm_workspace_سائيز);
                }
                spacemit_kernels::rvv::memcpy1d(b_کانو, reinterpret_cast<uint8_t *>(w_معلومات) + ni * صف_stride_b,
                                                nb_real * صف_stride_b);
            }

            لكل (; ni < gemm_n; ni += NB_کانن_ڳڻپ * nth) {
                int64_t صفون_بچيل = gemm_m;
                عائم * c_blk          = output + ni;
                خودڪار *  a_صف_cur      = a_صف;

                إذا (ith % 2 != 0) {
                    spine_barrier_wait(cur_barrier);
                }

                طالما (صفون_بچيل > 0) {
                    خودڪار صفون_handled = gemm_kernel(b_blk_len, a_صف_cur, b_کانو, b_کانو_zp, c_blk, صفون_بچيل,
                                                    nb_real, b_k_blks, gemm_n);

                    c_blk += صفون_handled * gemm_n;
                    a_صف_cur += صفون_handled * صف_stride_a;

                    صفون_بچيل -= صفون_handled;
                }

                إذا (ith % 2 == 0) {
                    spine_barrier_wait(cur_barrier);
                }

                ثابت int64_t ايندڙ_ni = ni + NB_کانن_ڳڻپ * nth;
                إذا (ايندڙ_ni < gemm_n) {
                    nb_real = std::min(gemm_n - ايندڙ_ni, NB_کانن_ڳڻپ);
                    spacemit_kernels::rvv::memcpy1d(b_کانو, reinterpret_cast<uint8_t *>(w_معلومات) + ايندڙ_ni * صف_stride_b,
                                                    nb_real * صف_stride_b);
                }
            }
        } وإلا {
            ثابت int64_t task_ڳڻپ_m = spacemit_kernels::div_ڦيرو_مٿي(gemm_m, gemm_m_stride);
            ثابت int64_t task_ڳڻپ_n = spacemit_kernels::div_ڦيرو_مٿي(gemm_n, gemm_n_stride);

            int64_t task_ڳڻپ      = task_ڳڻپ_m * task_ڳڻپ_n;
            int64_t task_في_thread = (task_ڳڻپ + nth - 1) / nth;
            int64_t start           = ith * task_في_thread;
            int64_t end             = std::min((ith + 1) * task_في_thread, task_ڳڻپ);
            لكل (int64_t compute_idx = start; compute_idx < end; compute_idx++) {
                ثابت خودڪار tid_n = compute_idx / task_ڳڻپ_m;
                ثابت خودڪار tid_m = compute_idx % task_ڳڻپ_m;

                ثابت int64_t m_شروعات = tid_m * gemm_m_stride;
                ثابت int64_t m_ڳڻپ = std::min(gemm_m - m_شروعات, (int64_t) gemm_m_stride);

                ثابت int64_t n_شروعات = tid_n * gemm_n_stride;
                ثابت int64_t n_ڳڻپ = std::min(gemm_n - n_شروعات, (int64_t) gemm_n_stride);

                ثابت int64_t n_blk = m_ڳڻپ == 1 ? n_ڳڻپ : NB_کانن_ڳڻپ;

                uint8_t * b_کانو    = reinterpret_cast<uint8_t *>(w_معلومات) + n_شروعات * صف_stride_b;
                uint8_t * b_کانو_zp = بلاڪ_type_has_zp<BLOC_TYPE>() ? b_کانو : خالي_اشارو;

                int64_t n_blk_real = 0;
                لكل (int64_t ni = 0; ni < n_ڳڻپ; ni += n_blk_real, b_کانو += n_blk_real * صف_stride_b) {
                    n_blk_real = std::min(n_ڳڻپ - ni, n_blk);

                    uint8_t * a_صف = quant_a_بفر + m_شروعات * صف_stride_a;

                    عائم * c_blk = output + m_شروعات * gemm_n + n_شروعات + ni;

                    int64_t صفون_بچيل = m_ڳڻپ;

                    uint8_t * b_کانو_cur    = b_کانو;
                    uint8_t * b_کانو_zp_cur = b_کانو_zp;

                    طالما (صفون_بچيل > 0) {
                        خودڪار صفون_handled = gemm_kernel(b_blk_len, a_صف, b_کانو_cur, b_کانو_zp_cur, c_blk,
                                                        صفون_بچيل, n_blk_real, b_k_blks, gemm_n);

                        c_blk += صفون_handled * gemm_n;
                        a_صف += صفون_handled * صف_stride_a;

                        صفون_بچيل -= صفون_handled;
                    }
                }
            }
        }
    }

    فراغ forward_mul_mat_id(ggml_compute_params * params, ggml_tensor * op) {
        constexpr size_t a_blk_len = INTER_سائيز;
        constexpr size_t b_blk_len = INTER_سائيز;

        ثابت ggml_tensor * src0 = op->src[0];
        ثابت ggml_tensor * src1 = op->src[1];
        ثابت ggml_tensor * ids  = op->src[2];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_ٻٽو_OP_LOCALS

        صحيح ith = params->ith;
        صحيح nth = params->nth;

        // row groups
        ثابت صحيح n_ids = ids->ne[0];  // n_expert_used
        ثابت صحيح n_as  = ne02;        // n_expert

        بنية mmid_صف_mapping {
            int32_t i1;
            int32_t i2;
        };

        spacemit_kernels::quantize_a_صف_def           quantize_a_صف_i8;
        spacemit_kernels::gemm_kernel_quantize_def     gemm_kernel;
        spacemit_kernels::moe_gemm_kernel_quantize_def moe_gemm_kernel_m2;
        منطقي                                           سيٽ_kernel_impl = خطأ;
        size_t                                         بلاڪ_stride_a  = spacemit_kernels::q8_blk_سائيز(QK4_0);

#إذا defined(RISCV64_SPACEMIT_IME2)
        إذا (!سيٽ_kernel_impl && (global_spine_env_info.use_ime2)) {
            quantize_a_صف_i8 = spacemit_kernels::rvv::quantize_a_صف_i8;
            بلاڪ_stride_a    = spacemit_kernels::q8_blk_سائيز(QK4_0, سچ);

            إذا constexpr (std::is_same_v<BLOC_TYPE, بلاڪ_q6_K> || std::is_same_v<BLOC_TYPE, بلاڪ_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i8;
                سيٽ_kernel_impl = سچ;
            } وإلا إذا constexpr (std::is_same_v<BLOC_TYPE, بلاڪ_q4_0> || std::is_same_v<BLOC_TYPE, بلاڪ_q4_1> ||
                                 std::is_same_v<BLOC_TYPE, بلاڪ_q4_K>) {
                إذا constexpr (INTER_سائيز == 256) {
                    gemm_kernel       = spacemit_kernels::ime2::gemm_kernel_i8i4_hp;
                    quantize_a_صف_i8 = spacemit_kernels::rvv::quantize_a_صف_i8_hp;
                    بلاڪ_stride_a    = spacemit_kernels::q8_hp_blk_سائيز(a_blk_len, سچ, سچ);
                    سيٽ_kernel_impl   = سچ;
                } وإلا {
                    gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i4;
                    moe_gemm_kernel_m2 = spacemit_kernels::ime2::moe_m2_gemm_kernel_i8i4;
                    quantize_a_صف_i8  = spacemit_kernels::rvv::quantize_a_صف_i8;
                    بلاڪ_stride_a     = spacemit_kernels::q8_blk_سائيز(a_blk_len, سچ);
                    سيٽ_kernel_impl    = سچ;
                }
            } وإلا إذا constexpr (std::is_same_v<BLOC_TYPE, بلاڪ_q2_K>) {
                quantize_a_صف_i8 = spacemit_kernels::rvv::quantize_a_صف_i8k;
                بلاڪ_stride_a    = spacemit_kernels::q8k_blk_سائيز(a_blk_len);
                gemm_kernel       = spacemit_kernels::ime2::gemm_kernel_i8i2k;
                سيٽ_kernel_impl   = سچ;
            } وإلا إذا constexpr (std::is_same_v<BLOC_TYPE, بلاڪ_q3_K>) {
                quantize_a_صف_i8 = spacemit_kernels::rvv::quantize_a_صف_i8k;
                بلاڪ_stride_a    = spacemit_kernels::q8k_blk_سائيز(a_blk_len);
                gemm_kernel       = spacemit_kernels::ime2::gemm_kernel_i8i3k;
                سيٽ_kernel_impl   = سچ;
            } وإلا إذا constexpr (std::is_same_v<BLOC_TYPE, بلاڪ_mxfp4>) {
                gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8mxfp4;
                moe_gemm_kernel_m2 = spacemit_kernels::ime2::moe_m2_gemm_kernel_i8mxfp4;
                سيٽ_kernel_impl    = سچ;
            } وإلا إذا constexpr (std::is_same_v<BLOC_TYPE, بلاڪ_q5_1> || std::is_same_v<BLOC_TYPE, بلاڪ_q5_K> ||
                                 std::is_same_v<BLOC_TYPE, بلاڪ_q5_0>) {
                gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i5;
                moe_gemm_kernel_m2 = spacemit_kernels::ime2::moe_m2_gemm_kernel_i8i5;
                سيٽ_kernel_impl    = سچ;
            }
        }
#جيڪڏهن_پورو

#إذا defined(RISCV64_SPACEMIT_IME1)
        إذا (!سيٽ_kernel_impl && (global_spine_env_info.use_ime1)) {
            quantize_a_صف_i8 = spacemit_kernels::ime1::quantize_a_صف_i8;

            إذا constexpr (std::is_same_v<BLOC_TYPE, بلاڪ_q4_0> || std::is_same_v<BLOC_TYPE, بلاڪ_q4_1> ||
                          std::is_same_v<BLOC_TYPE, بلاڪ_q4_K>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i4;
                سيٽ_kernel_impl = سچ;
            } وإلا إذا constexpr (std::is_same_v<BLOC_TYPE, بلاڪ_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i8;
                سيٽ_kernel_impl = سچ;
            }
        }
#جيڪڏهن_پورو
        إذا (!سيٽ_kernel_impl) {
            GGML_ABORT("no kernel implementation found for the block type");
        }

        ثابت size_t a_k_blks = spacemit_kernels::div_ڦيرو_مٿي(ne10, a_blk_len);
        ثابت size_t b_k_blks = spacemit_kernels::div_ڦيرو_مٿي(ne10, b_blk_len);

        ثابت size_t nbw1                = a_k_blks * بلاڪ_stride_a;
        ثابت size_t nbw2                = ne11 * nbw1;
        ثابت size_t nbw3                = nbw2 * ne12;
        ثابت size_t gemm_workspace_سائيز = GGML_PAD(nbw3, alignof(int64_t));

        ثابت uintptr_t ws_ptr         = reinterpret_cast<uintptr_t>(params->wdata);
        خودڪار *          quant_a_بفر = reinterpret_cast<uint8_t *>(ws_ptr);

        إذا (ne11 == 1) {
            لكل (int64_t ii = ith; ii < ne12 * a_k_blks; ii += nth) {
                int64_t i12       = ii / a_k_blks;
                int64_t ak_blk_id = ii % a_k_blks;
                quantize_a_صف_i8(a_blk_len, (عائم *) ((حرف *) src1->معلومات + i12 * nb12) + ak_blk_id * a_blk_len,
                                  a_blk_len, quant_a_بفر + i12 * nbw2 + ak_blk_id * بلاڪ_stride_a);
            }
        } وإلا {
            لكل (int64_t ii = ith; ii < ne12 * ne11; ii += nth) {
                int64_t i12 = ii / ne11;
                int64_t i11 = ii % ne11;
                quantize_a_صف_i8(a_blk_len, (عائم *) ((حرف *) src1->معلومات + i12 * nb12 + i11 * nb11), ne10,
                                  quant_a_بفر + i12 * nbw2 + i11 * nbw1);
            }
        }

#وصف_ڪريو MMID_ميٽرڪس_صف(صف_id, i1) ميٽرڪس_صفون[(صف_id) *ne12 + (i1)]

        int64_t *          ميٽرڪس_صف_counts       = (int64_t *) (ws_ptr + gemm_workspace_سائيز);
        int32_t *          valid_ep_ڳڻپ          = (int32_t *) (ميٽرڪس_صف_counts + n_as);
        int32_t *          valid_act_ڳڻپ         = (int32_t *) (valid_ep_ڳڻپ + 1);
        int64_t *          valid_ميٽرڪس_صف_counts = (int64_t *) (valid_act_ڳڻپ + 1);
        mmid_صف_mapping * ميٽرڪس_صفون             = (mmid_صف_mapping *) (valid_ميٽرڪس_صف_counts + n_as);

        إذا (ith == 0) {
            // initialize matrix_row_counts
            memset(ميٽرڪس_صف_counts, 0, n_as * حجم(int64_t));

            // group rows by src0 matrix
            لكل (int32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
                لكل (int32_t id = 0; id < n_ids; ++id) {
                    ثابت int32_t i02 =
                        *(ثابت int32_t *) ((ثابت حرف *) ids->معلومات + iid1 * ids->nb[1] + id * ids->nb[0]);

                    GGML_ASSERT(i02 >= 0 && i02 < n_as);

                    MMID_ميٽرڪس_صف(i02, ميٽرڪس_صف_counts[i02]) = { id, iid1 };
                    ميٽرڪس_صف_counts[i02] += 1;
                }
            }

            int32_t valid_ep_ڳڻپ_t  = 0;
            int32_t valid_act_ڳڻپ_t = 0;
            لكل (صحيح cur_a = 0; cur_a < n_as; ++cur_a) {
                ثابت int64_t cne1 = ميٽرڪس_صف_counts[cur_a];
                إذا (cne1 == 0) {
                    استمر;
                }
                valid_ميٽرڪس_صف_counts[valid_ep_ڳڻپ_t] = cur_a;
                valid_act_ڳڻپ_t += cne1;
                valid_ep_ڳڻپ_t += 1;
            }
            valid_ep_ڳڻپ[0]  = valid_ep_ڳڻپ_t;
            valid_act_ڳڻپ[0] = valid_act_ڳڻپ_t;
        }

        ثابت int64_t barrier_idx = static_cast<int64_t>(ith / 2);

        GGML_ASSERT(global_spine_env_info.init_barrier != خالي_اشارو);
        GGML_ASSERT(barrier_idx < spine_init_barrier_ڳڻپ);
        spine_barrier_t * cur_barrier = &global_spine_env_info.init_barrier[barrier_idx];

        ggml_barrier(params->threadpool);

        ثابت size_t صف_stride_b      = b_k_blks * get_repacked_بلاڪ_type_سائيز<BLOC_TYPE, INTER_سائيز, NB_کانن_ڳڻپ>();
        ثابت size_t expert_b_stride   = ne01 * صف_stride_b;
        ثابت size_t في_nb_کانن_ڳڻپ_wsize = NB_کانن_ڳڻپ * صف_stride_b;

        std::array<ثابت uint8_t *, 2> src_workspaces;
        std::array<عائم *, 2>         dst_workspaces;

        خودڪار *     tcm_بفر      = ggml::cpu::riscv64_spacemit::tls_context.tcm_بفر;
        ثابت خودڪار tcm_بفر_سائيز = ggml::cpu::riscv64_spacemit::tls_context.tcm_بفر_سائيز;

        ثابت خودڪار valid_ep_ڳڻپ_t  = valid_ep_ڳڻپ[0];
        ثابت خودڪار valid_act_ڳڻپ_t = valid_act_ڳڻپ[0];

        صحيح nth_es = 1;
        صحيح nth_n  = nth;

        صحيح ith_es = ith % nth_es;
        صحيح ith_n  = (ith / nth_es) % nth_n;

        إذا (valid_ep_ڳڻپ_t % nth == 0 && tcm_بفر != خالي_اشارو && valid_ep_ڳڻپ_t == n_as &&
            valid_act_ڳڻپ_t == n_as && في_nb_کانن_ڳڻپ_wsize <= tcm_بفر_سائيز) {
            لكل (int64_t valid_id = ith; valid_id < valid_ep_ڳڻپ_t; valid_id += nth) {
                ثابت int64_t cur_a = valid_ميٽرڪس_صف_counts[valid_id];

                خودڪار * src0_cur = (uint8_t *) src0->معلومات + cur_a * expert_b_stride;

                mmid_صف_mapping صف_mapping = MMID_ميٽرڪس_صف(cur_a, 0);
                ثابت صحيح        id          = صف_mapping.i1;
                ثابت int64_t    i11         = id % ne11;
                ثابت int64_t    i12         = صف_mapping.i2;
                ثابت int64_t    i1          = id;
                ثابت int64_t    i2          = i12;

                خودڪار *  src1_کانو = quant_a_بفر + (i11 * nbw1 + i12 * nbw2);
                عائم * c_blk    = (عائم *) ((حرف *) dst->معلومات + (i1 * nb1 + i2 * nb2));

                uint8_t * a_صف = src1_کانو;
                uint8_t * b_کانو = reinterpret_cast<uint8_t *>(tcm_بفر);
                إذا ((nbw1 + في_nb_کانن_ڳڻپ_wsize) <= tcm_بفر_سائيز) {
                    a_صف = (uint8_t *) tcm_بفر;
                    b_کانو = reinterpret_cast<uint8_t *>(tcm_بفر) + nbw1;
                }
                uint8_t * b_کانو_zp = بلاڪ_type_has_zp<BLOC_TYPE>() ? b_کانو : خالي_اشارو;

                إذا (ith % 2 == 0) {
                    spacemit_kernels::rvv::memcpy1d(b_کانو, reinterpret_cast<uint8_t *>(src0_cur), في_nb_کانن_ڳڻپ_wsize);

                    إذا (a_صف != src1_کانو) {
                        spacemit_kernels::rvv::memcpy1d(a_صف, src1_کانو, nbw1);
                    }
                }

                spine_barrier_wait(cur_barrier);

                إذا (ith % 2 != 0) {
                    إذا (a_صف != src1_کانو) {
                        spacemit_kernels::rvv::memcpy1d(a_صف, src1_کانو, nbw1);
                    }

                    spacemit_kernels::rvv::memcpy1d(b_کانو, reinterpret_cast<uint8_t *>(src0_cur), في_nb_کانن_ڳڻپ_wsize);
                }

                int64_t nb_real = std::min(ne01, NB_کانن_ڳڻپ);
                لكل (int64_t ni = 0; ni < ne01; ni += NB_کانن_ڳڻپ) {
                    إذا (ith % 2 != 0) {
                        spine_barrier_wait(cur_barrier);
                    }

                    gemm_kernel(b_blk_len, a_صف, b_کانو, b_کانو_zp, c_blk + ni, 1, nb_real, b_k_blks, ne01);

                    إذا (ith % 2 == 0) {
                        spine_barrier_wait(cur_barrier);
                    }

                    ثابت int64_t ايندڙ_ni = ni + NB_کانن_ڳڻپ;
                    إذا (ايندڙ_ni < ne01) {
                        nb_real = std::min(ne01 - ايندڙ_ni, NB_کانن_ڳڻپ);
                        spacemit_kernels::rvv::memcpy1d(
                            b_کانو, reinterpret_cast<uint8_t *>(src0_cur) + ايندڙ_ni * صف_stride_b, في_nb_کانن_ڳڻپ_wsize);
                    }
                }
            }
        } وإلا {
            لكل (int64_t valid_id = ith_es; valid_id < valid_ep_ڳڻپ_t; valid_id += nth_es) {
                ثابت int64_t cur_a = valid_ميٽرڪس_صف_counts[valid_id];
                ثابت int64_t cne1  = ميٽرڪس_صف_counts[cur_a];

                int64_t src1_cur_شروعات = 0;
                int64_t src1_cur_پڇاڙي   = cne1;

                int64_t src0_cur_شروعات = (ith_n * ne01) / nth_n;
                int64_t src0_cur_پڇاڙي   = MIN(((ith_n + 1) * ne01) / nth_n, ne01);

                إذا (src1_cur_شروعات >= src1_cur_پڇاڙي || src0_cur_شروعات >= src0_cur_پڇاڙي) {
                    استمر;
                }

                src0_cur_شروعات =
                    (src0_cur_شروعات % NB_کانن_ڳڻپ) ? src0_cur_شروعات + NB_کانن_ڳڻپ - (src0_cur_شروعات % NB_کانن_ڳڻپ) : src0_cur_شروعات;
                src0_cur_پڇاڙي =
                    (src0_cur_پڇاڙي % NB_کانن_ڳڻپ) ? src0_cur_پڇاڙي + NB_کانن_ڳڻپ - (src0_cur_پڇاڙي % NB_کانن_ڳڻپ) : src0_cur_پڇاڙي;

                خودڪار *    src0_cur = (uint8_t *) src0->معلومات + cur_a * expert_b_stride + src0_cur_شروعات * صف_stride_b;
                uint8_t * b_کانو_zp = بلاڪ_type_has_zp<BLOC_TYPE>() ? src0_cur : خالي_اشارو;

                size_t extra_tcm_بفر_سائيز = tcm_بفر_سائيز;
                فراغ * extra_tcm_بفر      = tcm_بفر;
                إذا (tcm_بفر != خالي_اشارو && (src1_cur_پڇاڙي - src1_cur_شروعات) >= 4 &&
                    (src0_cur_پڇاڙي - src0_cur_شروعات) * صف_stride_b <= tcm_بفر_سائيز) {
                    spacemit_kernels::rvv::memcpy1d(tcm_بفر, src0_cur,
                                                    (src0_cur_پڇاڙي - src0_cur_شروعات) * صف_stride_b);
                    src0_cur = reinterpret_cast<uint8_t *>(tcm_بفر);
                    b_کانو_zp = بلاڪ_type_has_zp<BLOC_TYPE>() ? src0_cur : خالي_اشارو;
                    extra_tcm_بفر_سائيز -= (src0_cur_پڇاڙي - src0_cur_شروعات) * صف_stride_b;
                    extra_tcm_بفر = reinterpret_cast<فراغ *>(reinterpret_cast<uintptr_t>(tcm_بفر) +
                                                                (src0_cur_پڇاڙي - src0_cur_شروعات) * صف_stride_b);
                }

                صحيح ir1 = src1_cur_شروعات;

                إذا (extra_tcm_بفر_سائيز >= nbw1 && extra_tcm_بفر != خالي_اشارو) {
                    int64_t quant_a_tile_سائيز = extra_tcm_بفر_سائيز / nbw1;
                    افعل {
                        quant_a_tile_سائيز = MIN(quant_a_tile_سائيز, src1_cur_پڇاڙي - ir1);

                        uint8_t * quant_a_tile_بفر = reinterpret_cast<uint8_t *>(extra_tcm_بفر);

                        صحيح iir1 = ir1;
                        لكل (; iir1 < (ir1 + quant_a_tile_سائيز); ++iir1) {
                            mmid_صف_mapping صف_mapping = MMID_ميٽرڪس_صف(cur_a, iir1);

                            ثابت صحيح id = صف_mapping.i1;  // selected expert index

                            ثابت int64_t i11 = id % ne11;
                            ثابت int64_t i12 = صف_mapping.i2;  // row index in src1

                            خودڪار * src1_کانو = quant_a_بفر + (i11 * nbw1 + i12 * nbw2);
                            spacemit_kernels::rvv::memcpy1d(quant_a_tile_بفر, src1_کانو, nbw1);
                            quant_a_tile_بفر = quant_a_tile_بفر + nbw1;
                        }

                        quant_a_tile_بفر = reinterpret_cast<uint8_t *>(extra_tcm_بفر);
                        iir1                = ir1;

                        إذا (moe_gemm_kernel_m2 != خالي_اشارو) {
                            لكل (; iir1 < (ir1 + quant_a_tile_سائيز - 1); iir1 += 2, quant_a_tile_بفر += 2 * nbw1) {
                                mmid_صف_mapping صف_mapping_0 = MMID_ميٽرڪس_صف(cur_a, iir1);
                                mmid_صف_mapping صف_mapping_1 = MMID_ميٽرڪس_صف(cur_a, iir1 + 1);

                                src_workspaces[0] = quant_a_tile_بفر;
                                src_workspaces[1] = quant_a_tile_بفر + nbw1;

                                dst_workspaces[0] =
                                    (عائم *) ((حرف *) dst->معلومات + (صف_mapping_0.i1 * nb1 + صف_mapping_0.i2 * nb2)) +
                                    src0_cur_شروعات;
                                dst_workspaces[1] = (عائم *) ((حرف *) dst->معلومات +
                                                               ((صف_mapping_1.i1) * nb1 + (صف_mapping_1.i2) * nb2)) +
                                                    src0_cur_شروعات;
                                moe_gemm_kernel_m2(b_blk_len, src_workspaces.معلومات(), src0_cur, b_کانو_zp,
                                                   dst_workspaces.معلومات(), 1, src0_cur_پڇاڙي - src0_cur_شروعات, b_k_blks,
                                                   ne01);
                            }
                        }

                        لكل (; iir1 < (ir1 + quant_a_tile_سائيز); iir1++, quant_a_tile_بفر += nbw1) {
                            mmid_صف_mapping صف_mapping_0 = MMID_ميٽرڪس_صف(cur_a, iir1);

                            gemm_kernel(
                                b_blk_len, quant_a_tile_بفر, src0_cur, b_کانو_zp,
                                (عائم *) ((حرف *) dst->معلومات + (صف_mapping_0.i1 * nb1 + صف_mapping_0.i2 * nb2)) +
                                    src0_cur_شروعات,
                                1, src0_cur_پڇاڙي - src0_cur_شروعات, b_k_blks, ne01);
                        }

                        ir1 += quant_a_tile_سائيز;
                    } طالما (ir1 < src1_cur_پڇاڙي);
                } وإلا {
                    إذا (moe_gemm_kernel_m2 != خالي_اشارو) {
                        لكل (; ir1 < src1_cur_پڇاڙي - 1; ir1 += 2) {
                            لكل (صحيح iir1 = 0; iir1 < 2; ++iir1) {
                                mmid_صف_mapping صف_mapping = MMID_ميٽرڪس_صف(cur_a, ir1 + iir1);

                                ثابت صحيح id = صف_mapping.i1;  // selected expert index

                                ثابت int64_t i11 = id % ne11;
                                ثابت int64_t i12 = صف_mapping.i2;  // row index in src1

                                ثابت int64_t i1 = id;               // selected expert index
                                ثابت int64_t i2 = i12;              //صف

                                src_workspaces[iir1] = quant_a_بفر + (i11 * nbw1 + i12 * nbw2);

                                dst_workspaces[iir1] =
                                    (عائم *) ((حرف *) dst->معلومات + (i1 * nb1 + i2 * nb2)) + src0_cur_شروعات;
                            }

                            moe_gemm_kernel_m2(b_blk_len, src_workspaces.معلومات(), src0_cur, b_کانو_zp,
                                               dst_workspaces.معلومات(), 1, src0_cur_پڇاڙي - src0_cur_شروعات, b_k_blks, ne01);
                        }
                    }

                    لكل (; ir1 < src1_cur_پڇاڙي; ir1++) {
                        mmid_صف_mapping صف_mapping = MMID_ميٽرڪس_صف(cur_a, ir1);

                        ثابت صحيح id = صف_mapping.i1;  // selected expert index

                        ثابت int64_t i11 = id % ne11;
                        ثابت int64_t i12 = صف_mapping.i2;  // row index in src1

                        ثابت int64_t i1 = id;               // selected expert index
                        ثابت int64_t i2 = i12;              //صف

                        خودڪار * src1_کانو = quant_a_بفر + (i11 * nbw1 + i12 * nbw2);

                        gemm_kernel(b_blk_len, src1_کانو, src0_cur, b_کانو_zp,
                                    (عائم *) ((حرف *) dst->معلومات + (i1 * nb1 + i2 * nb2)) + src0_cur_شروعات, 1,
                                    src0_cur_پڇاڙي - src0_cur_شروعات, b_k_blks, ne01);
                    }
                }
            }
        }
#وصف_رد MMID_ميٽرڪس_صف
    }

    صحيح repack(ggml_tensor * t, ثابت فراغ * معلومات, size_t معلومات_سائيز) override {
        GGML_LOG_DEBUG("%s: repack tensor %s with %s_%dx%d\n", __func__, t->name, ggml_type_name(t->type),
                       (صحيح) NB_کانن_ڳڻپ, (صحيح) INTER_سائيز);
        أرجع ggml::cpu::riscv64_spacemit::repack<BLOC_TYPE, INTER_سائيز, NB_کانن_ڳڻپ>(t, معلومات, معلومات_سائيز);
    }
};

صنف tensor_traits_common : عام tensor_traits_بنياد {
    منطقي work_سائيز(صحيح n_ڌاڳا, ثابت ggml_tensor * op, size_t & size) override {
        بدل (op->op) {
            حالة GGML_OP_FLASH_ATTN_EXT:
                {
                    ثابت صحيح     n_tasks = n_ڌاڳا;
                    ثابت int64_t neq2    = op->src[0]->ne[2];  // number of query heads
                    ثابت int64_t DK      = op->src[1]->ne[0];
                    ثابت int64_t DV      = op->src[2]->ne[0];  // DV

                    // Tiled flash attention scratch (tile sizes defined in common.h)
                    // Per-thread: Q_q + KQ + mask + VKQ32 + V32 + K_f32 + padding
                    size_t prefill = حجم(عائم) *
                                     (GGML_FA_TILE_Q * DK + 2 * GGML_FA_TILE_Q * GGML_FA_TILE_KV + GGML_FA_TILE_Q * DV +
                                      GGML_FA_TILE_KV * DV + GGML_FA_TILE_KV * DK) *
                                     n_tasks;

                    // Decode path: n_kv_chunks = n_tasks (one chunk per thread)
                    // Per-thread: VKQ accmulator (DV), partial M, partial S + intra-thread scratch for V, Q and VKQ
                    size_t n_chunks = n_tasks;
                    size_t decode   = حجم(عائم) * (neq2 * n_chunks * (2 + DV) + n_tasks * (DK + 2 * DV));

                    size = MAX(prefill, decode);
                }
                أرجع سچ;
            افتراضي:
                اكسر;
        }
        أرجع خطأ;
    }

    منطقي compute_forward(ggml_compute_params * params, ggml_tensor * op) override {
        بدل (op->op) {
            حالة GGML_OP_NORM:
                بدل (op->src[0]->type) {
                    حالة GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_norm_f32(params, op);
                        أرجع سچ;
                    افتراضي:
                        GGML_ABORT("fatal error");
                }
            حالة GGML_OP_RMS_NORM:
                بدل (op->src[0]->type) {
                    حالة GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_rms_norm_f32(params, op);
                        أرجع سچ;
                    افتراضي:
                        GGML_ABORT("fatal error");
                }
            حالة GGML_OP_جوڙيو:
                بدل (op->src[0]->type) {
                    حالة GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_ٻٽو<GGML_OP_جوڙيو, عائم>(params, op);
                        أرجع سچ;
                    حالة GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_ٻٽو<GGML_OP_جوڙيو, _Float16>(params, op);
                        أرجع سچ;
                    افتراضي:
                        ggml_compute_forward_جوڙيو(params, op);
                        أرجع سچ;
                }
            حالة GGML_OP_SUB:
                بدل (op->src[0]->type) {
                    حالة GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_ٻٽو<GGML_OP_SUB, عائم>(params, op);
                        أرجع سچ;
                    حالة GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_ٻٽو<GGML_OP_SUB, _Float16>(params, op);
                        أرجع سچ;
                    افتراضي:
                        ggml_compute_forward_sub(params, op);
                        أرجع سچ;
                }
            حالة GGML_OP_MUL:
                بدل (op->src[0]->type) {
                    حالة GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_ٻٽو<GGML_OP_MUL, عائم>(params, op);
                        أرجع سچ;
                    حالة GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_ٻٽو<GGML_OP_MUL, _Float16>(params, op);
                        أرجع سچ;
                    افتراضي:
                        ggml_compute_forward_mul(params, op);
                        أرجع سچ;
                }
            حالة GGML_OP_DIV:
                بدل (op->src[0]->type) {
                    حالة GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_ٻٽو<GGML_OP_DIV, عائم>(params, op);
                        أرجع سچ;
                    حالة GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_ٻٽو<GGML_OP_DIV, _Float16>(params, op);
                        أرجع سچ;
                    افتراضي:
                        ggml_compute_forward_div(params, op);
                        أرجع سچ;
                }
            حالة GGML_OP_FLASH_ATTN_EXT:
                forward_flash_attn_ext_f16(params, op);
                أرجع سچ;
            حالة GGML_OP_CONT:
                {
                    ثابت ggml_tensor * src0 = op->src[0];
                    إذا (op->type == src0->type && op->nb[0] != src0->nb[0] && op->nb[0] == src0->nb[1] &&
                        op->ne[3] * op->ne[2] * op->nb[2] == src0->ne[3] * src0->ne[2] * src0->nb[2]) {
                        spacemit_kernels::rvv::forward_cont_with_ترتيب_بدلايو(params, op);
                    } وإلا {
                        ggml_compute_forward_cont(params, op);
                    }
                    أرجع سچ;
                }
            حالة GGML_OP_CPY:
                {
                    ثابت ggml_tensor * src0 = op->src[0];
                    إذا (op->type == src0->type && op->nb[0] == src0->nb[1] && src0->nb[0] != src0->nb[1] &&
                        ggml_nelements(src0) == ggml_nelements(op)) {
                        spacemit_kernels::rvv::forward_cpy_with_ترتيب_بدلايو(params, op);
                    } وإلا {
                        ggml_compute_forward_cpy(params, op);
                    }
                    أرجع سچ;
                }
            حالة GGML_OP_REPEAT:
                {
                    ثابت منطقي صفون_equal         = ggml_nrows(op->src[0]) == ggml_nrows(op);
                    ثابت منطقي نشر_or_equal = op->src[0]->ne[0] == 1 || op->src[0]->ne[0] == op->ne[0];

                    إذا (صفون_equal && نشر_or_equal) {
                        بدل (op->src[0]->type) {
                            حالة GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_repeat_nrows<int32_t>(params, op);
                                أرجع سچ;
                            حالة GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_repeat_nrows<int16_t>(params, op);
                                أرجع سچ;
                            افتراضي:
                                اكسر;
                        }
                    }

                    إذا (op->src[0]->ne[1] == 1 && op->src[0]->ne[0] == op->ne[0]) {
                        بدل (op->src[0]->type) {
                            حالة GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_repeat_dim1<int32_t>(params, op);
                                أرجع سچ;
                            حالة GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_repeat_dim1<int16_t>(params, op);
                                أرجع سچ;
                            افتراضي:
                                اكسر;
                        }
                    }

                    ggml_compute_forward_repeat(params, op);
                }
                أرجع سچ;
            حالة GGML_OP_جوڙ_صفون:
                {
                    إذا (op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32) {
                        spacemit_kernels::rvv::forward_جوڙ_صفون<عائم>(params, op);
                    } وإلا {
                        ggml_compute_forward_جوڙ_صفون(params, op);
                    }
                }
                أرجع سچ;
            حالة GGML_OP_GET_صفون:
                {
                    إذا (op->src[0]->type == op->type) {
                        بدل (op->src[0]->type) {
                            حالة GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_get_صفون<int32_t>(params, op);
                                أرجع سچ;
                            حالة GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_get_صفون<int16_t>(params, op);
                                أرجع سچ;
                            افتراضي:
                                اكسر;
                        }
                    }

                    ggml_compute_forward_get_صفون(params, op);
                }
                أرجع سچ;
            حالة GGML_OP_CONCAT:
                {
                    ثابت int32_t dim = ggml_get_op_params_i32(op, 0);
                    إذا (dim == 0 && op->type == op->src[0]->type) {
                        بدل (op->src[0]->type) {
                            حالة GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_concat<int32_t>(params, op);
                                أرجع سچ;
                            حالة GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_concat<int16_t>(params, op);
                                أرجع سچ;
                            افتراضي:
                                اكسر;
                        }
                    }

                    ggml_compute_forward_concat(params, op);
                }
                أرجع سچ;
            // TODO For GGML_OP_GATED_DELTA_NET
            // case GGML_OP_GATED_DELTA_NET:
            //     return true;
            افتراضي:
                اكسر;
        }
        أرجع خطأ;
    }

    فراغ forward_flash_attn_ext_f16(ثابت ggml_compute_params * params, ggml_tensor * dst) {
        ثابت ggml_tensor * q = dst->src[0];
        ثابت ggml_tensor * k = dst->src[1];
        ثابت ggml_tensor * v = dst->src[2];

        GGML_TENSOR_LOCALS(int64_t, neq, q, ne)
        GGML_TENSOR_LOCALS(size_t, nbq, q, nb)
        GGML_TENSOR_LOCALS(int64_t, nek, k, ne)
        GGML_TENSOR_LOCALS(size_t, nbk, k, nb)
        GGML_TENSOR_LOCALS(int64_t, nev, v, ne)
        GGML_TENSOR_LOCALS(size_t, nbv, v, nb)
        GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
        GGML_TENSOR_LOCALS(size_t, nb, dst, nb)

        ثابت int64_t DK = nek0;
        ثابت int64_t DV = nev0;

        ثابت منطقي supported_prec  = (dst->op_params[3] == GGML_PREC_F32 || dst->op_params[3] == GGML_PREC_DEFAULT);
        ثابت منطقي supported_types = (q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16);
        ثابت منطقي supported_shape = (DK > 0 && DK <= 128 && DV > 0 && DV <= 128);
        ثابت منطقي supported_vlen  = (__riscv_vlenb() == 128);

        إذا (!(supported_prec && supported_types && supported_shape && supported_vlen)) {
            ggml_compute_forward_flash_attn_ext(params, dst);
            أرجع;
        }

        // total rows in q
        ثابت int64_t nr = neq1 * neq2 * neq3;

        // rows per thread
        ثابت صحيح ith = params->ith;
        ثابت صحيح nth = params->nth;

        ساكن constexpr int64_t Q_TILE_SZ = ggml_fa_tile_config::Q;
        ثابت منطقي               use_tiled = !params->use_ref && (neq1 >= Q_TILE_SZ);

        // 4x chunks per thread
        // int     nth_scaled = nth * 4;
        // int64_t chunk_size = (nr + nth_scaled - 1) / nth_scaled;
        // int64_t nchunk     = (nr + chunk_size - 1) / chunk_size;

        // if (nth == 1 || nchunk < nth) {
        //     nchunk = nth;
        // }

        int64_t nchunk = nth;

        إذا (ith == 0) {
            // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
            ggml_threadpool_chunk_سيٽ(params->threadpool, nth);
        }

        ggml_barrier(params->threadpool);

        // The number of elements in each chunk
        ثابت int64_t dr = (nr + nchunk - 1) / nchunk;

        // The first chunk comes from our thread_id, the rest will get auto-assigned.
        صحيح current_chunk = ith;

        طالما (current_chunk < nchunk) {
            ثابت int64_t ir0 = dr * current_chunk;
            ثابت int64_t ir1 = MIN(ir0 + dr, nr);

            إذا (use_tiled) {
                spacemit_kernels::rvv::forward_flash_attn_ext_f16_tiled_vlen1024_vf16(
                    params, dst, ir0, ir1, ggml::cpu::riscv64_spacemit::tls_context.tcm_بفر,
                    ggml::cpu::riscv64_spacemit::tls_context.tcm_بفر_سائيز);
            } وإلا {
                spacemit_kernels::rvv::forward_flash_attn_ext_f16_one_chunk_vlen1024_vf16(
                    params, dst, ir0, ir1, ggml::cpu::riscv64_spacemit::tls_context.tcm_بفر,
                    ggml::cpu::riscv64_spacemit::tls_context.tcm_بفر_سائيز);
            }

            current_chunk = ggml_threadpool_chunk_جوڙيو(params->threadpool, 1);
        }
    }

    صحيح repack(ggml_tensor * t, ثابت فراغ * معلومات, size_t معلومات_سائيز) override {
        يادگيري_نقل(t->معلومات, معلومات, معلومات_سائيز);
        أرجع 0;
    }
};

// Impl By IME1
ساكن ثابت tensor_traits<بلاڪ_q4_0, 32, 16>  q4_0_16x32_q8_0;
ساكن ثابت tensor_traits<بلاڪ_q4_1, 32, 16>  q4_1_16x32_q8_0;
ساكن ثابت tensor_traits<بلاڪ_q4_K, 32, 16>  q4_k_16x32_q8_0;
ساكن ثابت tensor_traits<بلاڪ_q8_0, 32, 16>  q8_0_16x32_q8_0;
// Impl By IME2
ساكن ثابت tensor_traits<بلاڪ_q2_K, 256, 32> q2_k_32x256_q8_0;
ساكن ثابت tensor_traits<بلاڪ_q3_K, 256, 32> q3_k_32x256_q8_0;
ساكن ثابت tensor_traits<بلاڪ_q4_0, 32, 32>  q4_0_32x32_q8_0;
ساكن ثابت tensor_traits<بلاڪ_q4_1, 32, 32>  q4_1_32x32_q8_0;
ساكن ثابت tensor_traits<بلاڪ_q4_0, 256, 32> q4_0_32x256_q8_0;
ساكن ثابت tensor_traits<بلاڪ_q4_1, 256, 32> q4_1_32x256_q8_0;
ساكن ثابت tensor_traits<بلاڪ_q4_K, 32, 32>  q4_k_32x32_q8_0;
ساكن ثابت tensor_traits<بلاڪ_q6_K, 32, 32>  q6_k_32x32_q8_0;
ساكن ثابت tensor_traits<بلاڪ_q8_0, 32, 32>  q8_0_32x32_q8_0;
ساكن ثابت tensor_traits<بلاڪ_mxfp4, 32, 32> mxfp4_32x32_q8_0;
ساكن ثابت tensor_traits<بلاڪ_q5_K, 32, 32>  q5_k_32x32_q8_0;
ساكن ثابت tensor_traits<بلاڪ_q5_1, 32, 32>  q5_1_32x32_q8_0;
ساكن ثابت tensor_traits<بلاڪ_q5_0, 32, 32>  q5_0_32x32_q8_0;
// Impl By RVV
ساكن ثابت tensor_traits_common               rvv_impl;

}  // namespace ggml::cpu::riscv64_spacemit

ساكن ثابت ggml::cpu::tensor_traits * ggml_riscv64_spacemit_get_optimal_repack_type(ثابت ggml_tensor * cur) {
    بدل (cur->type) {
        حالة GGML_TYPE_Q2_K:
            {
#إذا defined(RISCV64_SPACEMIT_IME2)
                إذا (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    أرجع &ggml::cpu::riscv64_spacemit::q2_k_32x256_q8_0;
                }
#جيڪڏهن_پورو
            }
            اكسر;
        حالة GGML_TYPE_Q3_K:
            {
#إذا defined(RISCV64_SPACEMIT_IME2)
                إذا (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    أرجع &ggml::cpu::riscv64_spacemit::q3_k_32x256_q8_0;
                }
#جيڪڏهن_پورو
            }
            اكسر;
        حالة GGML_TYPE_Q4_0:
            {
#إذا defined(RISCV64_SPACEMIT_IME2)
                إذا (cur->ne[1] % 32 == 0 && cur->ne[0] % 256 == 0 &&
                    (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    أرجع &ggml::cpu::riscv64_spacemit::q4_0_32x256_q8_0;
                }

                إذا (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    أرجع &ggml::cpu::riscv64_spacemit::q4_0_32x32_q8_0;
                }
#جيڪڏهن_پورو

#إذا defined(RISCV64_SPACEMIT_IME1)
                إذا (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    أرجع &ggml::cpu::riscv64_spacemit::q4_0_16x32_q8_0;
                }
#جيڪڏهن_پورو
            }
            اكسر;
        حالة GGML_TYPE_Q4_1:
            {
#إذا defined(RISCV64_SPACEMIT_IME2)
                // TODO
                // if (cur->ne[1] % 32 == 0 && cur->ne[0] % 256 == 0 &&
                //     (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                //     return &ggml::cpu::riscv64_spacemit::q4_1_32x256_q8_0;
                // }

                إذا (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    أرجع &ggml::cpu::riscv64_spacemit::q4_1_32x32_q8_0;
                }
#جيڪڏهن_پورو

#إذا defined(RISCV64_SPACEMIT_IME1)
                إذا (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    أرجع &ggml::cpu::riscv64_spacemit::q4_1_16x32_q8_0;
                }
#جيڪڏهن_پورو
            }
            اكسر;
        حالة GGML_TYPE_Q4_K:
            {
#إذا defined(RISCV64_SPACEMIT_IME2)
                إذا (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    أرجع &ggml::cpu::riscv64_spacemit::q4_k_32x32_q8_0;
                }
#جيڪڏهن_پورو

#إذا defined(RISCV64_SPACEMIT_IME1)
                إذا (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    أرجع &ggml::cpu::riscv64_spacemit::q4_k_16x32_q8_0;
                }
#جيڪڏهن_پورو
            }
            اكسر;
        حالة GGML_TYPE_Q6_K:
            {
#إذا defined(RISCV64_SPACEMIT_IME2)
                إذا ((ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    أرجع &ggml::cpu::riscv64_spacemit::q6_k_32x32_q8_0;
                }
#جيڪڏهن_پورو
            }
            اكسر;
        حالة GGML_TYPE_Q8_0:
            {
#إذا defined(RISCV64_SPACEMIT_IME2)
                إذا ((ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    أرجع &ggml::cpu::riscv64_spacemit::q8_0_32x32_q8_0;
                }
#جيڪڏهن_پورو

#إذا defined(RISCV64_SPACEMIT_IME1)
                إذا (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    أرجع &ggml::cpu::riscv64_spacemit::q8_0_16x32_q8_0;
                }
#جيڪڏهن_پورو
            }
            اكسر;
        حالة GGML_TYPE_MXFP4:
            {
#إذا defined(RISCV64_SPACEMIT_IME2)
                // TODO
                // if (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                //     return &ggml::cpu::riscv64_spacemit::mxfp4_32x32_q8_0;
                // }
#جيڪڏهن_پورو
            }
            اكسر;
        حالة GGML_TYPE_Q5_K:
            {
#إذا defined(RISCV64_SPACEMIT_IME2)
                إذا (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    أرجع &ggml::cpu::riscv64_spacemit::q5_k_32x32_q8_0;
                }
#جيڪڏهن_پورو
            }
            اكسر;
        حالة GGML_TYPE_Q5_1:
            {
#إذا defined(RISCV64_SPACEMIT_IME2)
                إذا (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    أرجع &ggml::cpu::riscv64_spacemit::q5_1_32x32_q8_0;
                }
#جيڪڏهن_پورو
            }
            اكسر;
        حالة GGML_TYPE_Q5_0:
            {
#إذا defined(RISCV64_SPACEMIT_IME2)
                إذا (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    أرجع &ggml::cpu::riscv64_spacemit::q5_0_32x32_q8_0;
                }
#جيڪڏهن_پورو
            }
            اكسر;
        افتراضي:
            اكسر;
    }

    أرجع خالي_اشارو;
}

ساكن تعداد ggml_حالت ggml_backend_riscv64_spacemit_بفر_init_tensor(ggml_backend_بفر_t بفر,
                                                                         ggml_tensor *         tensor) {
    tensor->extra =
        (فراغ *) const_cast<ggml::cpu::tensor_traits *>(ggml_riscv64_spacemit_get_optimal_repack_type(tensor));

    GGML_اڻ_استعمال_ٿيل(بفر);

    أرجع GGML_حالت_SUCCESS;
}

ساكن فراغ ggml_backend_riscv64_spacemit_بفر_free_بفر(ggml_backend_بفر_t بفر) {
    GGML_ASSERT(بفر);

    فراغ * بنياد = بفر->context;
    إذا (بنياد == خالي_اشارو) {
        أرجع;
    }

    ggml::cpu::riscv64_spacemit::spine_mem_pool_free(بنياد);
}

ساكن فراغ * ggml_backend_riscv64_spacemit_بفر_get_بنياد(ggml_backend_بفر_t بفر) {
    GGML_ASSERT(بفر);

    فراغ * بنياد = بفر->context;
    GGML_ASSERT(بنياد != خالي_اشارو);
    أرجع بنياد;
}

ساكن فراغ ggml_backend_riscv64_spacemit_بفر_memset_tensor(ggml_backend_بفر_t بفر,
                                                               ggml_tensor *         tensor,
                                                               uint8_t               value,
                                                               size_t                وٿي,
                                                               size_t                size) {
    GGML_ASSERT(tensor);
    memset((حرف *) tensor->معلومات + وٿي, value, size);

    GGML_اڻ_استعمال_ٿيل(بفر);
}

ساكن فراغ ggml_backend_riscv64_spacemit_بفر_clear(ggml_backend_بفر_t بفر, uint8_t value) {
    GGML_ASSERT(بفر);

    فراغ * بنياد = بفر->context;
    GGML_ASSERT(بنياد != خالي_اشارو);
    memset(بنياد, value, بفر->size);
}

ساكن فراغ ggml_backend_riscv64_spacemit_بفر_سيٽ_tensor(ggml_backend_بفر_t بفر,
                                                            ggml_tensor *         tensor,
                                                            ثابت فراغ *          معلومات,
                                                            size_t                وٿي,
                                                            size_t                size) {
    GGML_ASSERT(وٿي == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    خودڪار tensor_traits = (ggml::cpu::riscv64_spacemit::tensor_traits_بنياد *) tensor->extra;
    إذا (tensor_traits) {
        خودڪار OK = tensor_traits->repack(tensor, معلومات, size);
        GGML_ASSERT(OK == 0);
    }

    GGML_اڻ_استعمال_ٿيل(بفر);
}

ساكن ثابت ggml_backend_بفر_i ggml_backend_riscv64_spacemit_بفر_i = {
    /* .free_buffer     = */ ggml_backend_riscv64_spacemit_بفر_free_بفر,
    /* .get_base        = */ ggml_backend_riscv64_spacemit_بفر_get_بنياد,
    /* .init_tensor     = */ ggml_backend_riscv64_spacemit_بفر_init_tensor,
    /* .memset_tensor   = */ ggml_backend_riscv64_spacemit_بفر_memset_tensor,
    /* .set_tensor      = */ ggml_backend_riscv64_spacemit_بفر_سيٽ_tensor,
    /* .get_tensor      = */ خالي_اشارو,
    /* .set_tensor_2d   = */ خالي_اشارو,
    /* .get_tensor_2d   = */ خالي_اشارو,
    /* .cpy_tensor      = */ خالي_اشارو,
    /* .clear           = */ ggml_backend_riscv64_spacemit_بفر_clear,
    /* .reset           = */ خالي_اشارو,
};

ساكن ثابت حرف * ggml_backend_cpu_riscv64_spacemit_بفر_type_get_name(ggml_backend_بفر_type_t buft) {
    أرجع "CPU_RISCV64_SPACEMIT";

    GGML_اڻ_استعمال_ٿيل(buft);
}

ساكن ggml_backend_بفر_t ggml_backend_cpu_riscv64_spacemit_بفر_type_alloc_بفر(ggml_backend_بفر_type_t buft,
                                                                                        size_t size) {
    فراغ * بنياد = ggml::cpu::riscv64_spacemit::spine_mem_pool_alloc(size, 64);
    إذا (بنياد == خالي_اشارو) {
        أرجع خالي_اشارو;
    }

    أرجع ggml_backend_بفر_init(buft, ggml_backend_riscv64_spacemit_بفر_i, بنياد, size);
}

ساكن size_t ggml_backend_cpu_riscv64_spacemit_بفر_type_get_alignment(ggml_backend_بفر_type_t buft) {
    أرجع 64;

    GGML_اڻ_استعمال_ٿيل(buft);
}

ساكن size_t ggml_backend_cpu_riscv64_spacemit_nbytes(ggml_backend_بفر_type_t buft, ثابت ggml_tensor * tensor) {
    لكل (صحيح i = 0; i < GGML_MAX_DIMS; ++i) {
        إذا (tensor->ne[i] <= 0) {
            أرجع 0;
        }
    }

    GGML_اڻ_استعمال_ٿيل(buft);

    ثابت خودڪار plain_nbytes = [&]() {
        size_t ڪل = ggml_type_سائيز(tensor->type);
        لكل (صحيح i = 0; i < GGML_MAX_DIMS; ++i) {
            ڪل += (tensor->ne[i] - 1) * tensor->nb[i];
        }
        أرجع ڪل;
    };

    ثابت size_t blck_سائيز = ggml_blck_سائيز(tensor->type);
    إذا (blck_سائيز == 1) {
        أرجع plain_nbytes();
    }

    ثابت size_t صف_nbytes = tensor->ne[0] * tensor->nb[0] / blck_سائيز;

    ثابت خودڪار جوڙيو_strided_nbytes = [&](size_t ڪل, size_t src_بلاڪ_سائيز, size_t dst_بلاڪ_سائيز) {
        لكل (صحيح i = 1; i < GGML_MAX_DIMS; ++i) {
            ڪل += (tensor->ne[i] - 1) * (tensor->nb[i] / src_بلاڪ_سائيز) * dst_بلاڪ_سائيز;
        }
        أرجع ڪل;
    };

    ثابت خودڪار remap_بلاڪ_nbytes = [&](size_t src_بلاڪ_سائيز, size_t dst_بلاڪ_سائيز, int64_t padded_صفون = 0) {
        GGML_ASSERT(صف_nbytes % src_بلاڪ_سائيز == 0);

        size_t ڪل =
            جوڙيو_strided_nbytes((صف_nbytes / src_بلاڪ_سائيز) * dst_بلاڪ_سائيز, src_بلاڪ_سائيز, dst_بلاڪ_سائيز);

        إذا (padded_صفون > 0 && tensor->ne[1] % padded_صفون != 0) {
            ڪل += (padded_صفون - tensor->ne[1] % padded_صفون) * (tensor->nb[1] / src_بلاڪ_سائيز) * dst_بلاڪ_سائيز;
        }

        أرجع ڪل;
    };

    size_t nbytes = صف_nbytes;
    بدل (tensor->type) {
        حالة GGML_TYPE_Q4_K:
            nbytes = remap_بلاڪ_nbytes(حجم(بلاڪ_q4_K), حجم(بلاڪ_q4_1) * 8);
            اكسر;
        حالة GGML_TYPE_Q6_K:
            nbytes = remap_بلاڪ_nbytes(حجم(بلاڪ_q6_K), حجم(بلاڪ_q8_0) * 8, 32);
            اكسر;
        حالة GGML_TYPE_Q8_0:
            nbytes = remap_بلاڪ_nbytes(حجم(بلاڪ_q8_0), حجم(بلاڪ_q8_0), 32);
            اكسر;
        حالة GGML_TYPE_Q2_K:
            nbytes = remap_بلاڪ_nbytes(حجم(بلاڪ_q2_K), حجم(spacemit_kernels::nrow_بلاڪ_q2_k<1>));
            اكسر;
        حالة GGML_TYPE_Q3_K:
            nbytes = remap_بلاڪ_nbytes(حجم(بلاڪ_q3_K), حجم(spacemit_kernels::nrow_بلاڪ_q3_k<1>));
            اكسر;
        حالة GGML_TYPE_MXFP4:
            nbytes = remap_بلاڪ_nbytes(حجم(بلاڪ_mxfp4), حجم(spacemit_kernels::nrow_بلاڪ_mxfp4<1>));
            اكسر;
        حالة GGML_TYPE_Q5_K:
            nbytes = remap_بلاڪ_nbytes(حجم(بلاڪ_q5_K), حجم(spacemit_kernels::nrow_بلاڪ_q5_1<1>) * 8);
            اكسر;
        حالة GGML_TYPE_Q5_1:
            nbytes = remap_بلاڪ_nbytes(حجم(بلاڪ_q5_1), حجم(spacemit_kernels::nrow_بلاڪ_q5_1<1>));
            اكسر;
        حالة GGML_TYPE_Q5_0:
            nbytes = remap_بلاڪ_nbytes(حجم(بلاڪ_q5_0), حجم(spacemit_kernels::nrow_بلاڪ_q5_0<1>));
            اكسر;
        افتراضي:
            nbytes = جوڙيو_strided_nbytes(صف_nbytes, 1, 1);
            اكسر;
    }

    أرجع nbytes;
}

نطاق ggml::cpu::riscv64_spacemit {

صنف extra_بفر_type : ggml::cpu::extra_بفر_type {
    منطقي supports_op(ggml_backend_dev_t, ثابت ggml_tensor * op) override {
        بدل (op->op) {
            حالة GGML_OP_MUL_MAT:
                إذا (op->src[0]->بفر && (ggml_n_dims(op->src[0]) == 2) &&
                    op->src[0]->بفر->buft == ggml_backend_cpu_riscv64_spacemit_بفر_type() &&
                    ggml_riscv64_spacemit_get_optimal_repack_type(op->src[0])) {
                    إذا (op->src[1]->بفر && !ggml_backend_buft_is_host(op->src[1]->بفر->buft)) {
                        أرجع خطأ;
                    }
                    إذا (op->src[1]->type == GGML_TYPE_F32) {
                        أرجع سچ;
                    }
                }
                اكسر;
            حالة GGML_OP_MUL_MAT_ID:
                إذا (op->src[0]->بفر && (ggml_n_dims(op->src[0]) == 3) &&
                    op->src[0]->بفر->buft == ggml_backend_cpu_riscv64_spacemit_بفر_type() &&
                    ggml_riscv64_spacemit_get_optimal_repack_type(op->src[0])) {
                    إذا (op->src[1]->بفر && !ggml_backend_buft_is_host(op->src[1]->بفر->buft)) {
                        أرجع خطأ;
                    }
                    إذا (op->src[1]->type == GGML_TYPE_F32) {
                        أرجع سچ;
                    }
                }
                اكسر;
            افتراضي:
                // GGML_ABORT("fatal error");
                اكسر;
        }
        أرجع خطأ;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(ثابت ggml_tensor * op) override {
        بدل (op->op) {
            حالة GGML_OP_MUL_MAT:
            حالة GGML_OP_MUL_MAT_ID:
                إذا (op->src[0]->بفر && op->src[0]->بفر->buft == ggml_backend_cpu_riscv64_spacemit_بفر_type()) {
                    أرجع (ggml::cpu::tensor_traits *) op->src[0]->extra;
                }
                اكسر;
            حالة GGML_OP_NORM:
            حالة GGML_OP_RMS_NORM:
            حالة GGML_OP_جوڙيو:
            حالة GGML_OP_SUB:
            حالة GGML_OP_MUL:
            حالة GGML_OP_DIV:
            حالة GGML_OP_FLASH_ATTN_EXT:
            حالة GGML_OP_CONT:
            حالة GGML_OP_CPY:
            حالة GGML_OP_REPEAT:
            حالة GGML_OP_جوڙ_صفون:
            حالة GGML_OP_GET_صفون:
            حالة GGML_OP_CONCAT:
                // case GGML_OP_GATED_DELTA_NET:
                أرجع (ggml::cpu::tensor_traits *) (&ggml::cpu::riscv64_spacemit::rvv_impl);
            افتراضي:
                // GGML_ABORT("fatal error");
                اكسر;
        }

        أرجع خالي_اشارو;
    }
};

}  // namespace ggml::cpu::riscv64_spacemit

ggml_backend_بفر_type_t ggml_backend_cpu_riscv64_spacemit_بفر_type(فراغ) {
    ساكن ggml_backend_بفر_type ggml_backend_cpu_بفر_type_riscv64_spacemit = {
  /* .iface    = */
        {
         /* .get_name           = */ ggml_backend_cpu_riscv64_spacemit_بفر_type_get_name,
         /* .alloc_buffer       = */ ggml_backend_cpu_riscv64_spacemit_بفر_type_alloc_بفر,
         /* .alloc_buffer_n     = */ عدم,
         /* .get_alignment      = */ ggml_backend_cpu_riscv64_spacemit_بفر_type_get_alignment,
         /* .get_max_size       = */ خالي_اشارو,
         /* .get_alloc_size     = */ ggml_backend_cpu_riscv64_spacemit_nbytes,
         /* .get_alloc_size_n   = */ عدم,
         /* .is_host            = */ خالي_اشارو,
         },
 /* .device  = */
        ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
 /* .context = */
        جديد ggml::cpu::riscv64_spacemit::extra_بفر_type(),
    };

    أرجع &ggml_backend_cpu_بفر_type_riscv64_spacemit;
}

خارجي "C" {
ساكن صحيح bind_ai_thread() {
    صحيح  fd, بائيٽون;
    حرف str[32];

    fd = open("/proc/set_ai_thread", O_WRONLY);
    إذا (fd < 0) {
        GGML_LOG_ERROR("try open /proc/set_ai_thread failed\n");
        أرجع -1;
    }

    snprintf(str, 16, "%d", 0);
    بائيٽون = اكتب(fd, str, لڙي_ڊيگهه(str));
    إذا (بائيٽون < 0) {
        GGML_LOG_ERROR("try write /proc/set_ai_thread failed\n");
        close(fd);
        أرجع -1;
    }

    close(fd);
    أرجع 0;
}

فراغ ggml_backend_cpu_riscv64_spacemit_سيٽ_numa_thread_affinity(صحيح thread_n) {
    صحيح cpu_id = sched_getcpu();
    إذا (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2 &&
        !((1 << cpu_id) & ggml::cpu::riscv64_spacemit::global_spine_env_info.cpu_نقاب)) {
        GGML_PRINT_DEBUG("bind_ai_thread for thread %d, pid %d\n", thread_n, getpid());
        bind_ai_thread();
    }

    إذا (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_tcm &&
        ggml::cpu::riscv64_spacemit::tls_context.cpu_id == -1) {
        CPU_ZERO(&(ggml::cpu::riscv64_spacemit::tls_context.cpuset));
        pthread_t    main_thread     = pthread_self();
        ثابت خودڪار & perfer_core_ids = ggml::cpu::riscv64_spacemit::global_spine_env_info.perfer_core_ids;
        إذا (thread_n < 0 || static_cast<size_t>(thread_n) >= perfer_core_ids.size()) {
            GGML_ABORT("thread_n %d exceeds perfer_core_ids size %zu\n", thread_n, perfer_core_ids.size());
        }
        خودڪار perfer_cpu_id = perfer_core_ids[static_cast<size_t>(thread_n)];
        CPU_SET(perfer_cpu_id, &(ggml::cpu::riscv64_spacemit::tls_context.cpuset));
        صحيح s =
            pthread_setaffinity_np(main_thread, حجم(cpu_set_t), &(ggml::cpu::riscv64_spacemit::tls_context.cpuset));
        إذا (s != 0) {
            GGML_ABORT("set thread affinity error for thread_n %d, cpu_id %d\n", thread_n, perfer_cpu_id);
        }

        صحيح ai_cpu_id = perfer_cpu_id - ggml::cpu::riscv64_spacemit::global_spine_env_info.aicpu_id_وٿي;
        ggml::cpu::riscv64_spacemit::tls_context.cpu_id = ai_cpu_id;
        ggml::cpu::riscv64_spacemit::tls_context.tcm_بفر =
            ggml::cpu::riscv64_spacemit::spine_mem_pool_tcm_mem_get(ai_cpu_id);
        ggml::cpu::riscv64_spacemit::tls_context.tcm_بفر_سائيز =
            ggml::cpu::riscv64_spacemit::global_spine_env_info.tcm_blk_سائيز;
    }

    إذا (ggml::cpu::riscv64_spacemit::tls_context.tcm_بفر != خالي_اشارو) {
        فراغ * rt =
            ggml::cpu::riscv64_spacemit::spine_mem_pool_tcm_mem_wait(ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        إذا (rt == خالي_اشارو) {
            GGML_ABORT("wait tcm buffer failed for cpu_id: %d", ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        }
    }
}

فراغ ggml_backend_cpu_riscv64_spacemit_clear_numa_thread_affinity_threaded(صحيح thread_n) {
    إذا (ggml::cpu::riscv64_spacemit::tls_context.tcm_بفر != خالي_اشارو) {
        خودڪار rt = ggml::cpu::riscv64_spacemit::spine_mem_pool_tcm_mem_release(
            ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        إذا (rt != 0) {
            GGML_ABORT("release tcm buffer failed for cpu_id: %d", ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        }
    }
}
}
