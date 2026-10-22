#تعریف_کرو GGML_COMMON_IMPL_CPP
#تعریف_کرو GGML_COMMON_DECL_CPP

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
#غلطی "riscv v extension or v_intrinsic not enabled"
#وإلا
#ضمن <riscv_vector.h>
#جے_انت

#إذا !defined(__riscv_zfh) || !defined(__riscv_zvfh)
#غلطی "riscv zfh extension not enabled, GGML_RV_ZFH and GGML_RV_ZVFH must be defined to 1"
#جے_انت

#إذا !defined(__riscv_zba)
#غلطی "riscv zba extension not enabled, GGML_RV_ZBA must be defined to 1"
#جے_انت

#إذا defined(RISCV64_SPACEMIT_IME1) || defined(RISCV64_SPACEMIT_IME2)
#وإلا
#غلطی "RISCV64_SPACEMIT_IME1 or RISCV64_SPACEMIT_IME2 not defined"
#جے_انت

#وإلا

#غلطی "riscv not enabled in this build"

#جے_انت

#إذا defined(__GNUC__)
#ہدایت GCC diagnostic ignored "-Woverlength-strings"
#ہدایت GCC diagnostic ignored "-Wcast-qual"
#ہدایت GCC diagnostic ignored "-Wunused-parameter"
#جے_انت

// clang-format on

خارجي "C" {
خارجي فراغ ggml_threadpool_chunk_سیٹ(بنية ggml_threadpool * tp, صحيح value);
خارجي صحيح  ggml_threadpool_chunk_جوڑو(بنية ggml_threadpool * tp, صحيح value);
}

نطاق ggml::cpu::riscv64_spacemit {

بنية TLSContext {
    صحيح       cpu_id{ -1 };
    cpu_set_t cpuset;
    فراغ *    tcm_بفر{ خالی_سوچک };
    size_t    tcm_بفر_ناپ{ 0 };
};

thread_local TLSContext tls_context;

سانچا <قسم_ناں BLOC_TYPE, int64_t INTER_ناپ, int64_t NB_خانے_گنتی> constexpr size_t get_repacked_بلاک_type_ناپ() {
    إذا constexpr (std::is_same_v<BLOC_TYPE, بلاک_q6_K> || std::is_same_v<BLOC_TYPE, بلاک_q8_0>) {
        أرجع حجم(بلاک_q8_0);
    } وإلا إذا constexpr (std::is_same_v<BLOC_TYPE, بلاک_q4_0>) {
        أرجع حجم(بلاک_q4_0) * INTER_ناپ / QK4_0;
    } وإلا إذا constexpr (std::is_same_v<BLOC_TYPE, بلاک_q4_1> || std::is_same_v<BLOC_TYPE, بلاک_q4_K>) {
        أرجع (حجم(بلاک_q4_0) + حجم(uint8_t)) * INTER_ناپ / QK4_1;
    } وإلا إذا constexpr (std::is_same_v<BLOC_TYPE, بلاک_q2_K>) {
        أرجع حجم(spacemit_kernels::nrow_بلاک_q2_k<1>);
    } وإلا إذا constexpr (std::is_same_v<BLOC_TYPE, بلاک_q3_K>) {
        أرجع حجم(spacemit_kernels::nrow_بلاک_q3_k<1>);
    } وإلا إذا constexpr (std::is_same_v<BLOC_TYPE, بلاک_mxfp4>) {
        أرجع حجم(spacemit_kernels::nrow_بلاک_mxfp4<1>);
    } وإلا إذا constexpr (std::is_same_v<BLOC_TYPE, بلاک_q5_1> || std::is_same_v<BLOC_TYPE, بلاک_q5_K>) {
        أرجع حجم(spacemit_kernels::nrow_بلاک_q5_1<1>);
    } وإلا إذا constexpr (std::is_same_v<BLOC_TYPE, بلاک_q5_0>) {
        أرجع حجم(spacemit_kernels::nrow_بلاک_q5_0<1>);
    } وإلا {
        أكد(خطأ);
        أرجع 0;
    }
}

سانچا <قسم_ناں BLOC_TYPE> constexpr منطقي بلاک_type_has_zp() {
    إذا constexpr (std::is_same_v<BLOC_TYPE, بلاک_q6_K> || std::is_same_v<BLOC_TYPE, بلاک_q8_0> ||
                  std::is_same_v<BLOC_TYPE, بلاک_q3_K> || std::is_same_v<BLOC_TYPE, بلاک_q4_0> ||
                  std::is_same_v<BLOC_TYPE, بلاک_mxfp4> || std::is_same_v<BLOC_TYPE, بلاک_q5_0>) {
        أرجع خطأ;
    } وإلا إذا constexpr (std::is_same_v<BLOC_TYPE, بلاک_q4_1> || std::is_same_v<BLOC_TYPE, بلاک_q4_K> ||
                         std::is_same_v<BLOC_TYPE, بلاک_q2_K> || std::is_same_v<BLOC_TYPE, بلاک_q5_1> ||
                         std::is_same_v<BLOC_TYPE, بلاک_q5_K>) {
        أرجع سچ;
    } وإلا {
        أكد(خطأ);
        أرجع خطأ;
    }
}

صنف tensor_traits_ادھار : عام ggml::cpu::tensor_traits {
  عام:
    مجازی صحيح repack(ggml_tensor * t, ثابت فراغ * انکڑے, size_t انکڑے_ناپ) = 0;
};

سانچا <قسم_ناں BLOC_TYPE, int64_t INTER_ناپ, int64_t NB_خانے_گنتی> صنف tensor_traits : عام tensor_traits_ادھار {
    منطقي work_ناپ(صحيح /* n_threads */, ثابت ggml_tensor * op, size_t & size) override {
        بدل (op->op) {
            حالة GGML_OP_MUL_MAT:
                {
                    int64_t src1_nelements = ggml_nelements(op->src[1]);

                    إذا constexpr (std::is_same_v<BLOC_TYPE, بلاک_q2_K> || std::is_same_v<BLOC_TYPE, بلاک_q3_K>) {
                        size =
                            spacemit_kernels::div_گیڑا_اتے(src1_nelements, QK_K) * spacemit_kernels::q8k_blk_ناپ(QK_K);
                    } وإلا إذا constexpr (INTER_ناپ == QK4_0) {
                        size = spacemit_kernels::div_گیڑا_اتے(src1_nelements, QK4_0) *
                               spacemit_kernels::q8_blk_ناپ(QK4_0, سچ);
                    } وإلا إذا constexpr (INTER_ناپ == 256) {
                        size = spacemit_kernels::div_گیڑا_اتے(src1_nelements, 256) *
                               spacemit_kernels::q8_hp_blk_ناپ(256, سچ, سچ);
                    } وإلا {
                        GGML_ABORT("unsupported block type");
                    }

                    size = GGML_PAD(size, حجم(int64_t));

                    أرجع سچ;
                }
            حالة GGML_OP_MUL_MAT_ID:
                {
                    int64_t src1_nelements = ggml_nelements(op->src[1]);

                    إذا constexpr (std::is_same_v<BLOC_TYPE, بلاک_q2_K> || std::is_same_v<BLOC_TYPE, بلاک_q3_K>) {
                        size =
                            spacemit_kernels::div_گیڑا_اتے(src1_nelements, QK_K) * spacemit_kernels::q8k_blk_ناپ(QK_K);
                    } وإلا إذا constexpr (INTER_ناپ == QK4_0) {
                        size = spacemit_kernels::div_گیڑا_اتے(src1_nelements, QK4_0) *
                               spacemit_kernels::q8_blk_ناپ(QK4_0, سچ);
                    } وإلا إذا constexpr (INTER_ناپ == 256) {
                        size = spacemit_kernels::div_گیڑا_اتے(src1_nelements, 256) *
                               spacemit_kernels::q8_hp_blk_ناپ(256, سچ, سچ);
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
        constexpr size_t a_blk_len = INTER_ناپ;
        constexpr size_t b_blk_len = INTER_ناپ;

        ثابت ggml_tensor * src0 = op->src[0];
        ثابت ggml_tensor * src1 = op->src[1];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_ثنائی_OP_LOCALS

        صحيح ith = params->ith;
        صحيح nth = params->nth;

        [[maybe_unused]] ثابت تعداد ggml_type type = src0->type;

        فراغ *        w_انکڑے  = (فراغ *) src0->انکڑے;
        ثابت عائم * feature = (ثابت عائم *) src1->انکڑے;
        عائم *       output  = (عائم *) dst->انکڑے;

        ثابت int64_t gemm_m = ne11 * ne12 * ne13;
        ثابت int64_t gemm_k = ne10;
        ثابت int64_t gemm_n = ne01;

        spacemit_kernels::quantize_a_صف_def       quantize_a_صف_i8;
        spacemit_kernels::quantize_a_صف_def       quantize_a_4row_i8;
        spacemit_kernels::gemm_kernel_quantize_def gemm_kernel;
        منطقي                                       سیٹ_kernel_impl = خطأ;

        int64_t بلاک_stride_a = spacemit_kernels::q8_blk_ناپ(a_blk_len);

#إذا defined(RISCV64_SPACEMIT_IME2)
        إذا (!سیٹ_kernel_impl && (global_spine_env_info.use_ime2)) {
            quantize_a_صف_i8  = spacemit_kernels::rvv::quantize_a_صف_i8;
            quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8;
            بلاک_stride_a     = spacemit_kernels::q8_blk_ناپ(a_blk_len, سچ);

            إذا constexpr (std::is_same_v<BLOC_TYPE, بلاک_q6_K> || std::is_same_v<BLOC_TYPE, بلاک_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i8;
                سیٹ_kernel_impl = سچ;
            } وإلا إذا constexpr (std::is_same_v<BLOC_TYPE, بلاک_q4_0> || std::is_same_v<BLOC_TYPE, بلاک_q4_1> ||
                                 std::is_same_v<BLOC_TYPE, بلاک_q4_K>) {
                إذا constexpr (INTER_ناپ == 256) {
                    gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i4_hp;
                    quantize_a_صف_i8  = spacemit_kernels::rvv::quantize_a_صف_i8_hp;
                    quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8_hp;
                    بلاک_stride_a     = spacemit_kernels::q8_hp_blk_ناپ(a_blk_len, سچ, سچ);
                    سیٹ_kernel_impl    = سچ;
                } وإلا {
                    gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i4;
                    quantize_a_صف_i8  = spacemit_kernels::rvv::quantize_a_صف_i8;
                    quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8;
                    بلاک_stride_a     = spacemit_kernels::q8_blk_ناپ(a_blk_len, سچ);
                    سیٹ_kernel_impl    = سچ;
                }
            } وإلا إذا constexpr (std::is_same_v<BLOC_TYPE, بلاک_q2_K>) {
                quantize_a_صف_i8  = spacemit_kernels::rvv::quantize_a_صف_i8k;
                quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8k;
                بلاک_stride_a     = spacemit_kernels::q8k_blk_ناپ(a_blk_len);

                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i2k;
                سیٹ_kernel_impl = سچ;
            } وإلا إذا constexpr (std::is_same_v<BLOC_TYPE, بلاک_q3_K>) {
                quantize_a_صف_i8  = spacemit_kernels::rvv::quantize_a_صف_i8k;
                quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8k;
                بلاک_stride_a     = spacemit_kernels::q8k_blk_ناپ(a_blk_len);

                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i3k;
                سیٹ_kernel_impl = سچ;
            } وإلا إذا constexpr (std::is_same_v<BLOC_TYPE, بلاک_mxfp4>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8mxfp4;
                سیٹ_kernel_impl = سچ;
            } وإلا إذا constexpr (std::is_same_v<BLOC_TYPE, بلاک_q5_1> || std::is_same_v<BLOC_TYPE, بلاک_q5_K> ||
                                 std::is_same_v<BLOC_TYPE, بلاک_q5_0>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i5;
                سیٹ_kernel_impl = سچ;
            }
        }
#جے_انت

#إذا defined(RISCV64_SPACEMIT_IME1)
        إذا (!سیٹ_kernel_impl && (global_spine_env_info.use_ime1)) {
            quantize_a_صف_i8  = spacemit_kernels::ime1::quantize_a_صف_i8;
            quantize_a_4row_i8 = spacemit_kernels::ime1::quantize_a_4row_i8;

            إذا constexpr (std::is_same_v<BLOC_TYPE, بلاک_q4_0> || std::is_same_v<BLOC_TYPE, بلاک_q4_1> ||
                          std::is_same_v<BLOC_TYPE, بلاک_q4_K>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i4;
                سیٹ_kernel_impl = سچ;
            } وإلا إذا constexpr (std::is_same_v<BLOC_TYPE, بلاک_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i8;
                سیٹ_kernel_impl = سچ;
            }
        }
#جے_انت
        إذا (!سیٹ_kernel_impl) {
            GGML_ABORT("no kernel implementation found for the block type");
        }

        ثابت int64_t a_k_blks = spacemit_kernels::div_گیڑا_اتے(gemm_k, a_blk_len);
        ثابت int64_t b_k_blks = spacemit_kernels::div_گیڑا_اتے(gemm_k, b_blk_len);

        ثابت int64_t صف_stride_a        = a_k_blks * بلاک_stride_a;
        ثابت int64_t gemm_workspace_ناپ = GGML_PAD(gemm_m * صف_stride_a, alignof(int64_t));

        إذا (ith == 0 && params->wsize < gemm_workspace_ناپ) {
            GGML_ABORT("wsize less than gemm_workspace_size");
        }

        uintptr_t ws_ptr = reinterpret_cast<uintptr_t>(params->wdata);

        فراغ *        tcm_بفر      = ggml::cpu::riscv64_spacemit::tls_context.tcm_بفر;
        ثابت int64_t tcm_بفر_ناپ = ggml::cpu::riscv64_spacemit::tls_context.tcm_بفر_ناپ;

        خودکار * quant_a_بفر = reinterpret_cast<uint8_t *>(ws_ptr);

        constexpr int64_t صف_align = 4;
        ثابت int64_t     صف_blks  = spacemit_kernels::div_گیڑا_اتے(gemm_m, صف_align);

        ثابت int64_t صف_stride_b      = b_k_blks * get_repacked_بلاک_type_ناپ<BLOC_TYPE, INTER_ناپ, NB_خانے_گنتی>();
        ثابت int64_t فی_mb_صفاں_wsize = صف_align * صف_stride_a;
        ثابت int64_t فی_nb_خانے_گنتی_wsize = NB_خانے_گنتی * صف_stride_b;

        ثابت int64_t barrier_idx = static_cast<int64_t>(ith / 2);

        GGML_ASSERT(global_spine_env_info.init_barrier != خالی_سوچک);
        GGML_ASSERT(barrier_idx < spine_init_barrier_گنتی);
        spine_barrier_t * cur_barrier = &global_spine_env_info.init_barrier[barrier_idx];

        إذا (gemm_m == 1) {
            صحيح task_فی_thread = spacemit_kernels::div_گیڑا_اتے(a_k_blks, nth);
            صحيح a_blk_شروع     = ith * task_فی_thread;
            صحيح a_blk_سرا       = std::min(a_blk_شروع + task_فی_thread, (صحيح) a_k_blks);
            إذا (a_blk_شروع < a_blk_سرا) {
                quantize_a_صف_i8(a_blk_len, feature + a_blk_شروع * a_blk_len, (a_blk_سرا - a_blk_شروع) * a_blk_len,
                                  quant_a_بفر + a_blk_شروع * بلاک_stride_a);
            }
        } وإلا {
            صحيح task_فی_thread = spacemit_kernels::div_گیڑا_اتے(صف_blks, nth);
            صحيح m_صف_blk_شروع = ith * task_فی_thread;
            صحيح m_صف_blk_سرا   = std::min(m_صف_blk_شروع + task_فی_thread, (صحيح) صف_blks);
            لكل (صحيح m_صف_blk = m_صف_blk_شروع; m_صف_blk < m_صف_blk_سرا; m_صف_blk++) {
                صحيح m_idx             = m_صف_blk * صف_align;
                صحيح صفاں_tobe_handled = (gemm_m - m_idx) > صف_align ? صف_align : (gemm_m - m_idx);

                إذا (صفاں_tobe_handled == صف_align && quantize_a_4row_i8 != خالی_سوچک) {
                    ثابت عائم * a_صف_ptr       = feature + m_idx * gemm_k;
                    خودکار *        quant_a_صف_ptr = quant_a_بفر + m_idx * صف_stride_a;
                    quantize_a_4row_i8(a_blk_len, a_صف_ptr, gemm_k, quant_a_صف_ptr);
                } وإلا {
                    طالما (صفاں_tobe_handled) {
                        ثابت عائم * a_صف_ptr       = feature + m_idx * gemm_k;
                        خودکار *        quant_a_صف_ptr = quant_a_بفر + m_idx * صف_stride_a;
                        quantize_a_صف_i8(a_blk_len, a_صف_ptr, gemm_k, quant_a_صف_ptr);
                        صفاں_tobe_handled -= 1;
                        m_idx += 1;
                    }
                }
            }
        }

        ggml_barrier(params->threadpool);

        ثابت int64_t gemm_m_stride     = gemm_n / gemm_m > 64 ? gemm_m : 16;
        ثابت int64_t gemm_m_blocked    = spacemit_kernels::div_گیڑا_اتے(gemm_m, gemm_m_stride);
        ثابت int64_t max_gemm_n_stride = spacemit_kernels::div_گیڑا_اتے(gemm_n * gemm_m_blocked, nth);

        int64_t gemm_n_stride = gemm_n;
        إذا (max_gemm_n_stride < gemm_n) {
            gemm_n_stride =
                std::min(gemm_n_stride, spacemit_kernels::div_گیڑا_اتے(max_gemm_n_stride, NB_خانے_گنتی) * NB_خانے_گنتی);
        }

        إذا (gemm_n_stride == gemm_n && tcm_بفر != خالی_سوچک && فی_mb_صفاں_wsize <= tcm_بفر_ناپ) {
            لكل (int64_t m_شروع = ith * صف_align; m_شروع < gemm_m; m_شروع += صف_align * nth) {
                uint8_t * b_خانہ    = reinterpret_cast<uint8_t *>(w_انکڑے);
                uint8_t * b_خانہ_zp = بلاک_type_has_zp<BLOC_TYPE>() ? b_خانہ : خالی_سوچک;

                int64_t m_صف_real = std::min(gemm_m - m_شروع, صف_align);

                spacemit_kernels::rvv::memcpy1d(tcm_بفر, quant_a_بفر + m_شروع * صف_stride_a,
                                                m_صف_real * صف_stride_a);

                int64_t n_blk_real = 0;
                لكل (int64_t ni = 0; ni < gemm_n; ni += n_blk_real, b_خانہ += n_blk_real * صف_stride_b) {
                    n_blk_real = std::min(gemm_n - ni, (int64_t) NB_خانے_گنتی);

                    uint8_t * a_صف_ptr = (uint8_t *) tcm_بفر;
                    عائم *   c_blk     = output + m_شروع * gemm_n + ni;

                    int32_t صفاں_بچیا = m_صف_real;

                    طالما (صفاں_بچیا > 0) {
                        خودکار صفاں_handled = gemm_kernel(b_blk_len, a_صف_ptr, b_خانہ, b_خانہ_zp, c_blk, صفاں_بچیا,
                                                        n_blk_real, b_k_blks, gemm_n);

                        c_blk += صفاں_handled * gemm_n;
                        a_صف_ptr += صفاں_handled * صف_stride_a;

                        صفاں_بچیا -= صفاں_handled;
                    }
                }
            }
        } وإلا إذا (tcm_بفر != خالی_سوچک && فی_nb_خانے_گنتی_wsize <= tcm_بفر_ناپ) {
            uint8_t * a_صف = quant_a_بفر;
            uint8_t * b_خانہ = reinterpret_cast<uint8_t *>(tcm_بفر);
            إذا ((gemm_workspace_ناپ + فی_nb_خانے_گنتی_wsize) <= tcm_بفر_ناپ) {
                a_صف = (uint8_t *) tcm_بفر;
                b_خانہ = reinterpret_cast<uint8_t *>(tcm_بفر) + gemm_workspace_ناپ;
            }
            uint8_t * b_خانہ_zp = بلاک_type_has_zp<BLOC_TYPE>() ? b_خانہ : خالی_سوچک;

            int64_t ni      = ith * NB_خانے_گنتی;
            int64_t nb_real = std::min(gemm_n - ni, NB_خانے_گنتی);

            إذا (ith % 2 == 0 && nb_real > 0) {
                spacemit_kernels::rvv::memcpy1d(b_خانہ, reinterpret_cast<uint8_t *>(w_انکڑے) + ni * صف_stride_b,
                                                nb_real * صف_stride_b);
                إذا (a_صف != quant_a_بفر) {
                    spacemit_kernels::rvv::memcpy1d(a_صف, quant_a_بفر, gemm_workspace_ناپ);
                }
            }

            spine_barrier_wait(cur_barrier);

            إذا (ith % 2 != 0 && nb_real > 0) {
                إذا (a_صف != quant_a_بفر) {
                    spacemit_kernels::rvv::memcpy1d(a_صف, quant_a_بفر, gemm_workspace_ناپ);
                }
                spacemit_kernels::rvv::memcpy1d(b_خانہ, reinterpret_cast<uint8_t *>(w_انکڑے) + ni * صف_stride_b,
                                                nb_real * صف_stride_b);
            }

            لكل (; ni < gemm_n; ni += NB_خانے_گنتی * nth) {
                int64_t صفاں_بچیا = gemm_m;
                عائم * c_blk          = output + ni;
                خودکار *  a_صف_cur      = a_صف;

                إذا (ith % 2 != 0) {
                    spine_barrier_wait(cur_barrier);
                }

                طالما (صفاں_بچیا > 0) {
                    خودکار صفاں_handled = gemm_kernel(b_blk_len, a_صف_cur, b_خانہ, b_خانہ_zp, c_blk, صفاں_بچیا,
                                                    nb_real, b_k_blks, gemm_n);

                    c_blk += صفاں_handled * gemm_n;
                    a_صف_cur += صفاں_handled * صف_stride_a;

                    صفاں_بچیا -= صفاں_handled;
                }

                إذا (ith % 2 == 0) {
                    spine_barrier_wait(cur_barrier);
                }

                ثابت int64_t اگلا_ni = ni + NB_خانے_گنتی * nth;
                إذا (اگلا_ni < gemm_n) {
                    nb_real = std::min(gemm_n - اگلا_ni, NB_خانے_گنتی);
                    spacemit_kernels::rvv::memcpy1d(b_خانہ, reinterpret_cast<uint8_t *>(w_انکڑے) + اگلا_ni * صف_stride_b,
                                                    nb_real * صف_stride_b);
                }
            }
        } وإلا {
            ثابت int64_t task_گنتی_m = spacemit_kernels::div_گیڑا_اتے(gemm_m, gemm_m_stride);
            ثابت int64_t task_گنتی_n = spacemit_kernels::div_گیڑا_اتے(gemm_n, gemm_n_stride);

            int64_t task_گنتی      = task_گنتی_m * task_گنتی_n;
            int64_t task_فی_thread = (task_گنتی + nth - 1) / nth;
            int64_t start           = ith * task_فی_thread;
            int64_t end             = std::min((ith + 1) * task_فی_thread, task_گنتی);
            لكل (int64_t compute_idx = start; compute_idx < end; compute_idx++) {
                ثابت خودکار tid_n = compute_idx / task_گنتی_m;
                ثابت خودکار tid_m = compute_idx % task_گنتی_m;

                ثابت int64_t m_شروع = tid_m * gemm_m_stride;
                ثابت int64_t m_گنتی = std::min(gemm_m - m_شروع, (int64_t) gemm_m_stride);

                ثابت int64_t n_شروع = tid_n * gemm_n_stride;
                ثابت int64_t n_گنتی = std::min(gemm_n - n_شروع, (int64_t) gemm_n_stride);

                ثابت int64_t n_blk = m_گنتی == 1 ? n_گنتی : NB_خانے_گنتی;

                uint8_t * b_خانہ    = reinterpret_cast<uint8_t *>(w_انکڑے) + n_شروع * صف_stride_b;
                uint8_t * b_خانہ_zp = بلاک_type_has_zp<BLOC_TYPE>() ? b_خانہ : خالی_سوچک;

                int64_t n_blk_real = 0;
                لكل (int64_t ni = 0; ni < n_گنتی; ni += n_blk_real, b_خانہ += n_blk_real * صف_stride_b) {
                    n_blk_real = std::min(n_گنتی - ni, n_blk);

                    uint8_t * a_صف = quant_a_بفر + m_شروع * صف_stride_a;

                    عائم * c_blk = output + m_شروع * gemm_n + n_شروع + ni;

                    int64_t صفاں_بچیا = m_گنتی;

                    uint8_t * b_خانہ_cur    = b_خانہ;
                    uint8_t * b_خانہ_zp_cur = b_خانہ_zp;

                    طالما (صفاں_بچیا > 0) {
                        خودکار صفاں_handled = gemm_kernel(b_blk_len, a_صف, b_خانہ_cur, b_خانہ_zp_cur, c_blk,
                                                        صفاں_بچیا, n_blk_real, b_k_blks, gemm_n);

                        c_blk += صفاں_handled * gemm_n;
                        a_صف += صفاں_handled * صف_stride_a;

                        صفاں_بچیا -= صفاں_handled;
                    }
                }
            }
        }
    }

    فراغ forward_mul_mat_id(ggml_compute_params * params, ggml_tensor * op) {
        constexpr size_t a_blk_len = INTER_ناپ;
        constexpr size_t b_blk_len = INTER_ناپ;

        ثابت ggml_tensor * src0 = op->src[0];
        ثابت ggml_tensor * src1 = op->src[1];
        ثابت ggml_tensor * ids  = op->src[2];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_ثنائی_OP_LOCALS

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
        منطقي                                           سیٹ_kernel_impl = خطأ;
        size_t                                         بلاک_stride_a  = spacemit_kernels::q8_blk_ناپ(QK4_0);

#إذا defined(RISCV64_SPACEMIT_IME2)
        إذا (!سیٹ_kernel_impl && (global_spine_env_info.use_ime2)) {
            quantize_a_صف_i8 = spacemit_kernels::rvv::quantize_a_صف_i8;
            بلاک_stride_a    = spacemit_kernels::q8_blk_ناپ(QK4_0, سچ);

            إذا constexpr (std::is_same_v<BLOC_TYPE, بلاک_q6_K> || std::is_same_v<BLOC_TYPE, بلاک_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i8;
                سیٹ_kernel_impl = سچ;
            } وإلا إذا constexpr (std::is_same_v<BLOC_TYPE, بلاک_q4_0> || std::is_same_v<BLOC_TYPE, بلاک_q4_1> ||
                                 std::is_same_v<BLOC_TYPE, بلاک_q4_K>) {
                إذا constexpr (INTER_ناپ == 256) {
                    gemm_kernel       = spacemit_kernels::ime2::gemm_kernel_i8i4_hp;
                    quantize_a_صف_i8 = spacemit_kernels::rvv::quantize_a_صف_i8_hp;
                    بلاک_stride_a    = spacemit_kernels::q8_hp_blk_ناپ(a_blk_len, سچ, سچ);
                    سیٹ_kernel_impl   = سچ;
                } وإلا {
                    gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i4;
                    moe_gemm_kernel_m2 = spacemit_kernels::ime2::moe_m2_gemm_kernel_i8i4;
                    quantize_a_صف_i8  = spacemit_kernels::rvv::quantize_a_صف_i8;
                    بلاک_stride_a     = spacemit_kernels::q8_blk_ناپ(a_blk_len, سچ);
                    سیٹ_kernel_impl    = سچ;
                }
            } وإلا إذا constexpr (std::is_same_v<BLOC_TYPE, بلاک_q2_K>) {
                quantize_a_صف_i8 = spacemit_kernels::rvv::quantize_a_صف_i8k;
                بلاک_stride_a    = spacemit_kernels::q8k_blk_ناپ(a_blk_len);
                gemm_kernel       = spacemit_kernels::ime2::gemm_kernel_i8i2k;
                سیٹ_kernel_impl   = سچ;
            } وإلا إذا constexpr (std::is_same_v<BLOC_TYPE, بلاک_q3_K>) {
                quantize_a_صف_i8 = spacemit_kernels::rvv::quantize_a_صف_i8k;
                بلاک_stride_a    = spacemit_kernels::q8k_blk_ناپ(a_blk_len);
                gemm_kernel       = spacemit_kernels::ime2::gemm_kernel_i8i3k;
                سیٹ_kernel_impl   = سچ;
            } وإلا إذا constexpr (std::is_same_v<BLOC_TYPE, بلاک_mxfp4>) {
                gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8mxfp4;
                moe_gemm_kernel_m2 = spacemit_kernels::ime2::moe_m2_gemm_kernel_i8mxfp4;
                سیٹ_kernel_impl    = سچ;
            } وإلا إذا constexpr (std::is_same_v<BLOC_TYPE, بلاک_q5_1> || std::is_same_v<BLOC_TYPE, بلاک_q5_K> ||
                                 std::is_same_v<BLOC_TYPE, بلاک_q5_0>) {
                gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i5;
                moe_gemm_kernel_m2 = spacemit_kernels::ime2::moe_m2_gemm_kernel_i8i5;
                سیٹ_kernel_impl    = سچ;
            }
        }
#جے_انت

#إذا defined(RISCV64_SPACEMIT_IME1)
        إذا (!سیٹ_kernel_impl && (global_spine_env_info.use_ime1)) {
            quantize_a_صف_i8 = spacemit_kernels::ime1::quantize_a_صف_i8;

            إذا constexpr (std::is_same_v<BLOC_TYPE, بلاک_q4_0> || std::is_same_v<BLOC_TYPE, بلاک_q4_1> ||
                          std::is_same_v<BLOC_TYPE, بلاک_q4_K>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i4;
                سیٹ_kernel_impl = سچ;
            } وإلا إذا constexpr (std::is_same_v<BLOC_TYPE, بلاک_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i8;
                سیٹ_kernel_impl = سچ;
            }
        }
#جے_انت
        إذا (!سیٹ_kernel_impl) {
            GGML_ABORT("no kernel implementation found for the block type");
        }

        ثابت size_t a_k_blks = spacemit_kernels::div_گیڑا_اتے(ne10, a_blk_len);
        ثابت size_t b_k_blks = spacemit_kernels::div_گیڑا_اتے(ne10, b_blk_len);

        ثابت size_t nbw1                = a_k_blks * بلاک_stride_a;
        ثابت size_t nbw2                = ne11 * nbw1;
        ثابت size_t nbw3                = nbw2 * ne12;
        ثابت size_t gemm_workspace_ناپ = GGML_PAD(nbw3, alignof(int64_t));

        ثابت uintptr_t ws_ptr         = reinterpret_cast<uintptr_t>(params->wdata);
        خودکار *          quant_a_بفر = reinterpret_cast<uint8_t *>(ws_ptr);

        إذا (ne11 == 1) {
            لكل (int64_t ii = ith; ii < ne12 * a_k_blks; ii += nth) {
                int64_t i12       = ii / a_k_blks;
                int64_t ak_blk_id = ii % a_k_blks;
                quantize_a_صف_i8(a_blk_len, (عائم *) ((حرف *) src1->انکڑے + i12 * nb12) + ak_blk_id * a_blk_len,
                                  a_blk_len, quant_a_بفر + i12 * nbw2 + ak_blk_id * بلاک_stride_a);
            }
        } وإلا {
            لكل (int64_t ii = ith; ii < ne12 * ne11; ii += nth) {
                int64_t i12 = ii / ne11;
                int64_t i11 = ii % ne11;
                quantize_a_صف_i8(a_blk_len, (عائم *) ((حرف *) src1->انکڑے + i12 * nb12 + i11 * nb11), ne10,
                                  quant_a_بفر + i12 * nbw2 + i11 * nbw1);
            }
        }

#تعریف_کرو MMID_میٹرکس_صف(صف_id, i1) میٹرکس_صفاں[(صف_id) *ne12 + (i1)]

        int64_t *          میٹرکس_صف_counts       = (int64_t *) (ws_ptr + gemm_workspace_ناپ);
        int32_t *          valid_ep_گنتی          = (int32_t *) (میٹرکس_صف_counts + n_as);
        int32_t *          valid_act_گنتی         = (int32_t *) (valid_ep_گنتی + 1);
        int64_t *          valid_میٹرکس_صف_counts = (int64_t *) (valid_act_گنتی + 1);
        mmid_صف_mapping * میٹرکس_صفاں             = (mmid_صف_mapping *) (valid_میٹرکس_صف_counts + n_as);

        إذا (ith == 0) {
            // initialize matrix_row_counts
            memset(میٹرکس_صف_counts, 0, n_as * حجم(int64_t));

            // group rows by src0 matrix
            لكل (int32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
                لكل (int32_t id = 0; id < n_ids; ++id) {
                    ثابت int32_t i02 =
                        *(ثابت int32_t *) ((ثابت حرف *) ids->انکڑے + iid1 * ids->nb[1] + id * ids->nb[0]);

                    GGML_ASSERT(i02 >= 0 && i02 < n_as);

                    MMID_میٹرکس_صف(i02, میٹرکس_صف_counts[i02]) = { id, iid1 };
                    میٹرکس_صف_counts[i02] += 1;
                }
            }

            int32_t valid_ep_گنتی_t  = 0;
            int32_t valid_act_گنتی_t = 0;
            لكل (صحيح cur_a = 0; cur_a < n_as; ++cur_a) {
                ثابت int64_t cne1 = میٹرکس_صف_counts[cur_a];
                إذا (cne1 == 0) {
                    استمر;
                }
                valid_میٹرکس_صف_counts[valid_ep_گنتی_t] = cur_a;
                valid_act_گنتی_t += cne1;
                valid_ep_گنتی_t += 1;
            }
            valid_ep_گنتی[0]  = valid_ep_گنتی_t;
            valid_act_گنتی[0] = valid_act_گنتی_t;
        }

        ثابت int64_t barrier_idx = static_cast<int64_t>(ith / 2);

        GGML_ASSERT(global_spine_env_info.init_barrier != خالی_سوچک);
        GGML_ASSERT(barrier_idx < spine_init_barrier_گنتی);
        spine_barrier_t * cur_barrier = &global_spine_env_info.init_barrier[barrier_idx];

        ggml_barrier(params->threadpool);

        ثابت size_t صف_stride_b      = b_k_blks * get_repacked_بلاک_type_ناپ<BLOC_TYPE, INTER_ناپ, NB_خانے_گنتی>();
        ثابت size_t expert_b_stride   = ne01 * صف_stride_b;
        ثابت size_t فی_nb_خانے_گنتی_wsize = NB_خانے_گنتی * صف_stride_b;

        std::array<ثابت uint8_t *, 2> src_workspaces;
        std::array<عائم *, 2>         dst_workspaces;

        خودکار *     tcm_بفر      = ggml::cpu::riscv64_spacemit::tls_context.tcm_بفر;
        ثابت خودکار tcm_بفر_ناپ = ggml::cpu::riscv64_spacemit::tls_context.tcm_بفر_ناپ;

        ثابت خودکار valid_ep_گنتی_t  = valid_ep_گنتی[0];
        ثابت خودکار valid_act_گنتی_t = valid_act_گنتی[0];

        صحيح nth_es = 1;
        صحيح nth_n  = nth;

        صحيح ith_es = ith % nth_es;
        صحيح ith_n  = (ith / nth_es) % nth_n;

        إذا (valid_ep_گنتی_t % nth == 0 && tcm_بفر != خالی_سوچک && valid_ep_گنتی_t == n_as &&
            valid_act_گنتی_t == n_as && فی_nb_خانے_گنتی_wsize <= tcm_بفر_ناپ) {
            لكل (int64_t valid_id = ith; valid_id < valid_ep_گنتی_t; valid_id += nth) {
                ثابت int64_t cur_a = valid_میٹرکس_صف_counts[valid_id];

                خودکار * src0_cur = (uint8_t *) src0->انکڑے + cur_a * expert_b_stride;

                mmid_صف_mapping صف_mapping = MMID_میٹرکس_صف(cur_a, 0);
                ثابت صحيح        id          = صف_mapping.i1;
                ثابت int64_t    i11         = id % ne11;
                ثابت int64_t    i12         = صف_mapping.i2;
                ثابت int64_t    i1          = id;
                ثابت int64_t    i2          = i12;

                خودکار *  src1_خانہ = quant_a_بفر + (i11 * nbw1 + i12 * nbw2);
                عائم * c_blk    = (عائم *) ((حرف *) dst->انکڑے + (i1 * nb1 + i2 * nb2));

                uint8_t * a_صف = src1_خانہ;
                uint8_t * b_خانہ = reinterpret_cast<uint8_t *>(tcm_بفر);
                إذا ((nbw1 + فی_nb_خانے_گنتی_wsize) <= tcm_بفر_ناپ) {
                    a_صف = (uint8_t *) tcm_بفر;
                    b_خانہ = reinterpret_cast<uint8_t *>(tcm_بفر) + nbw1;
                }
                uint8_t * b_خانہ_zp = بلاک_type_has_zp<BLOC_TYPE>() ? b_خانہ : خالی_سوچک;

                إذا (ith % 2 == 0) {
                    spacemit_kernels::rvv::memcpy1d(b_خانہ, reinterpret_cast<uint8_t *>(src0_cur), فی_nb_خانے_گنتی_wsize);

                    إذا (a_صف != src1_خانہ) {
                        spacemit_kernels::rvv::memcpy1d(a_صف, src1_خانہ, nbw1);
                    }
                }

                spine_barrier_wait(cur_barrier);

                إذا (ith % 2 != 0) {
                    إذا (a_صف != src1_خانہ) {
                        spacemit_kernels::rvv::memcpy1d(a_صف, src1_خانہ, nbw1);
                    }

                    spacemit_kernels::rvv::memcpy1d(b_خانہ, reinterpret_cast<uint8_t *>(src0_cur), فی_nb_خانے_گنتی_wsize);
                }

                int64_t nb_real = std::min(ne01, NB_خانے_گنتی);
                لكل (int64_t ni = 0; ni < ne01; ni += NB_خانے_گنتی) {
                    إذا (ith % 2 != 0) {
                        spine_barrier_wait(cur_barrier);
                    }

                    gemm_kernel(b_blk_len, a_صف, b_خانہ, b_خانہ_zp, c_blk + ni, 1, nb_real, b_k_blks, ne01);

                    إذا (ith % 2 == 0) {
                        spine_barrier_wait(cur_barrier);
                    }

                    ثابت int64_t اگلا_ni = ni + NB_خانے_گنتی;
                    إذا (اگلا_ni < ne01) {
                        nb_real = std::min(ne01 - اگلا_ni, NB_خانے_گنتی);
                        spacemit_kernels::rvv::memcpy1d(
                            b_خانہ, reinterpret_cast<uint8_t *>(src0_cur) + اگلا_ni * صف_stride_b, فی_nb_خانے_گنتی_wsize);
                    }
                }
            }
        } وإلا {
            لكل (int64_t valid_id = ith_es; valid_id < valid_ep_گنتی_t; valid_id += nth_es) {
                ثابت int64_t cur_a = valid_میٹرکس_صف_counts[valid_id];
                ثابت int64_t cne1  = میٹرکس_صف_counts[cur_a];

                int64_t src1_cur_شروع = 0;
                int64_t src1_cur_سرا   = cne1;

                int64_t src0_cur_شروع = (ith_n * ne01) / nth_n;
                int64_t src0_cur_سرا   = MIN(((ith_n + 1) * ne01) / nth_n, ne01);

                إذا (src1_cur_شروع >= src1_cur_سرا || src0_cur_شروع >= src0_cur_سرا) {
                    استمر;
                }

                src0_cur_شروع =
                    (src0_cur_شروع % NB_خانے_گنتی) ? src0_cur_شروع + NB_خانے_گنتی - (src0_cur_شروع % NB_خانے_گنتی) : src0_cur_شروع;
                src0_cur_سرا =
                    (src0_cur_سرا % NB_خانے_گنتی) ? src0_cur_سرا + NB_خانے_گنتی - (src0_cur_سرا % NB_خانے_گنتی) : src0_cur_سرا;

                خودکار *    src0_cur = (uint8_t *) src0->انکڑے + cur_a * expert_b_stride + src0_cur_شروع * صف_stride_b;
                uint8_t * b_خانہ_zp = بلاک_type_has_zp<BLOC_TYPE>() ? src0_cur : خالی_سوچک;

                size_t extra_tcm_بفر_ناپ = tcm_بفر_ناپ;
                فراغ * extra_tcm_بفر      = tcm_بفر;
                إذا (tcm_بفر != خالی_سوچک && (src1_cur_سرا - src1_cur_شروع) >= 4 &&
                    (src0_cur_سرا - src0_cur_شروع) * صف_stride_b <= tcm_بفر_ناپ) {
                    spacemit_kernels::rvv::memcpy1d(tcm_بفر, src0_cur,
                                                    (src0_cur_سرا - src0_cur_شروع) * صف_stride_b);
                    src0_cur = reinterpret_cast<uint8_t *>(tcm_بفر);
                    b_خانہ_zp = بلاک_type_has_zp<BLOC_TYPE>() ? src0_cur : خالی_سوچک;
                    extra_tcm_بفر_ناپ -= (src0_cur_سرا - src0_cur_شروع) * صف_stride_b;
                    extra_tcm_بفر = reinterpret_cast<فراغ *>(reinterpret_cast<uintptr_t>(tcm_بفر) +
                                                                (src0_cur_سرا - src0_cur_شروع) * صف_stride_b);
                }

                صحيح ir1 = src1_cur_شروع;

                إذا (extra_tcm_بفر_ناپ >= nbw1 && extra_tcm_بفر != خالی_سوچک) {
                    int64_t quant_a_tile_ناپ = extra_tcm_بفر_ناپ / nbw1;
                    افعل {
                        quant_a_tile_ناپ = MIN(quant_a_tile_ناپ, src1_cur_سرا - ir1);

                        uint8_t * quant_a_tile_بفر = reinterpret_cast<uint8_t *>(extra_tcm_بفر);

                        صحيح iir1 = ir1;
                        لكل (; iir1 < (ir1 + quant_a_tile_ناپ); ++iir1) {
                            mmid_صف_mapping صف_mapping = MMID_میٹرکس_صف(cur_a, iir1);

                            ثابت صحيح id = صف_mapping.i1;  // selected expert index

                            ثابت int64_t i11 = id % ne11;
                            ثابت int64_t i12 = صف_mapping.i2;  // row index in src1

                            خودکار * src1_خانہ = quant_a_بفر + (i11 * nbw1 + i12 * nbw2);
                            spacemit_kernels::rvv::memcpy1d(quant_a_tile_بفر, src1_خانہ, nbw1);
                            quant_a_tile_بفر = quant_a_tile_بفر + nbw1;
                        }

                        quant_a_tile_بفر = reinterpret_cast<uint8_t *>(extra_tcm_بفر);
                        iir1                = ir1;

                        إذا (moe_gemm_kernel_m2 != خالی_سوچک) {
                            لكل (; iir1 < (ir1 + quant_a_tile_ناپ - 1); iir1 += 2, quant_a_tile_بفر += 2 * nbw1) {
                                mmid_صف_mapping صف_mapping_0 = MMID_میٹرکس_صف(cur_a, iir1);
                                mmid_صف_mapping صف_mapping_1 = MMID_میٹرکس_صف(cur_a, iir1 + 1);

                                src_workspaces[0] = quant_a_tile_بفر;
                                src_workspaces[1] = quant_a_tile_بفر + nbw1;

                                dst_workspaces[0] =
                                    (عائم *) ((حرف *) dst->انکڑے + (صف_mapping_0.i1 * nb1 + صف_mapping_0.i2 * nb2)) +
                                    src0_cur_شروع;
                                dst_workspaces[1] = (عائم *) ((حرف *) dst->انکڑے +
                                                               ((صف_mapping_1.i1) * nb1 + (صف_mapping_1.i2) * nb2)) +
                                                    src0_cur_شروع;
                                moe_gemm_kernel_m2(b_blk_len, src_workspaces.انکڑے(), src0_cur, b_خانہ_zp,
                                                   dst_workspaces.انکڑے(), 1, src0_cur_سرا - src0_cur_شروع, b_k_blks,
                                                   ne01);
                            }
                        }

                        لكل (; iir1 < (ir1 + quant_a_tile_ناپ); iir1++, quant_a_tile_بفر += nbw1) {
                            mmid_صف_mapping صف_mapping_0 = MMID_میٹرکس_صف(cur_a, iir1);

                            gemm_kernel(
                                b_blk_len, quant_a_tile_بفر, src0_cur, b_خانہ_zp,
                                (عائم *) ((حرف *) dst->انکڑے + (صف_mapping_0.i1 * nb1 + صف_mapping_0.i2 * nb2)) +
                                    src0_cur_شروع,
                                1, src0_cur_سرا - src0_cur_شروع, b_k_blks, ne01);
                        }

                        ir1 += quant_a_tile_ناپ;
                    } طالما (ir1 < src1_cur_سرا);
                } وإلا {
                    إذا (moe_gemm_kernel_m2 != خالی_سوچک) {
                        لكل (; ir1 < src1_cur_سرا - 1; ir1 += 2) {
                            لكل (صحيح iir1 = 0; iir1 < 2; ++iir1) {
                                mmid_صف_mapping صف_mapping = MMID_میٹرکس_صف(cur_a, ir1 + iir1);

                                ثابت صحيح id = صف_mapping.i1;  // selected expert index

                                ثابت int64_t i11 = id % ne11;
                                ثابت int64_t i12 = صف_mapping.i2;  // row index in src1

                                ثابت int64_t i1 = id;               // selected expert index
                                ثابت int64_t i2 = i12;              //صف

                                src_workspaces[iir1] = quant_a_بفر + (i11 * nbw1 + i12 * nbw2);

                                dst_workspaces[iir1] =
                                    (عائم *) ((حرف *) dst->انکڑے + (i1 * nb1 + i2 * nb2)) + src0_cur_شروع;
                            }

                            moe_gemm_kernel_m2(b_blk_len, src_workspaces.انکڑے(), src0_cur, b_خانہ_zp,
                                               dst_workspaces.انکڑے(), 1, src0_cur_سرا - src0_cur_شروع, b_k_blks, ne01);
                        }
                    }

                    لكل (; ir1 < src1_cur_سرا; ir1++) {
                        mmid_صف_mapping صف_mapping = MMID_میٹرکس_صف(cur_a, ir1);

                        ثابت صحيح id = صف_mapping.i1;  // selected expert index

                        ثابت int64_t i11 = id % ne11;
                        ثابت int64_t i12 = صف_mapping.i2;  // row index in src1

                        ثابت int64_t i1 = id;               // selected expert index
                        ثابت int64_t i2 = i12;              //صف

                        خودکار * src1_خانہ = quant_a_بفر + (i11 * nbw1 + i12 * nbw2);

                        gemm_kernel(b_blk_len, src1_خانہ, src0_cur, b_خانہ_zp,
                                    (عائم *) ((حرف *) dst->انکڑے + (i1 * nb1 + i2 * nb2)) + src0_cur_شروع, 1,
                                    src0_cur_سرا - src0_cur_شروع, b_k_blks, ne01);
                    }
                }
            }
        }
#تعریف_رد MMID_میٹرکس_صف
    }

    صحيح repack(ggml_tensor * t, ثابت فراغ * انکڑے, size_t انکڑے_ناپ) override {
        GGML_LOG_DEBUG("%s: repack tensor %s with %s_%dx%d\n", __func__, t->name, ggml_type_name(t->type),
                       (صحيح) NB_خانے_گنتی, (صحيح) INTER_ناپ);
        أرجع ggml::cpu::riscv64_spacemit::repack<BLOC_TYPE, INTER_ناپ, NB_خانے_گنتی>(t, انکڑے, انکڑے_ناپ);
    }
};

صنف tensor_traits_common : عام tensor_traits_ادھار {
    منطقي work_ناپ(صحيح n_دھاگے, ثابت ggml_tensor * op, size_t & size) override {
        بدل (op->op) {
            حالة GGML_OP_FLASH_ATTN_EXT:
                {
                    ثابت صحيح     n_tasks = n_دھاگے;
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
            حالة GGML_OP_جوڑو:
                بدل (op->src[0]->type) {
                    حالة GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_ثنائی<GGML_OP_جوڑو, عائم>(params, op);
                        أرجع سچ;
                    حالة GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_ثنائی<GGML_OP_جوڑو, _Float16>(params, op);
                        أرجع سچ;
                    افتراضي:
                        ggml_compute_forward_جوڑو(params, op);
                        أرجع سچ;
                }
            حالة GGML_OP_SUB:
                بدل (op->src[0]->type) {
                    حالة GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_ثنائی<GGML_OP_SUB, عائم>(params, op);
                        أرجع سچ;
                    حالة GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_ثنائی<GGML_OP_SUB, _Float16>(params, op);
                        أرجع سچ;
                    افتراضي:
                        ggml_compute_forward_sub(params, op);
                        أرجع سچ;
                }
            حالة GGML_OP_MUL:
                بدل (op->src[0]->type) {
                    حالة GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_ثنائی<GGML_OP_MUL, عائم>(params, op);
                        أرجع سچ;
                    حالة GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_ثنائی<GGML_OP_MUL, _Float16>(params, op);
                        أرجع سچ;
                    افتراضي:
                        ggml_compute_forward_mul(params, op);
                        أرجع سچ;
                }
            حالة GGML_OP_DIV:
                بدل (op->src[0]->type) {
                    حالة GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_ثنائی<GGML_OP_DIV, عائم>(params, op);
                        أرجع سچ;
                    حالة GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_ثنائی<GGML_OP_DIV, _Float16>(params, op);
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
                        spacemit_kernels::rvv::forward_cont_with_ترتیب_بدلو(params, op);
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
                        spacemit_kernels::rvv::forward_cpy_with_ترتیب_بدلو(params, op);
                    } وإلا {
                        ggml_compute_forward_cpy(params, op);
                    }
                    أرجع سچ;
                }
            حالة GGML_OP_REPEAT:
                {
                    ثابت منطقي صفاں_equal         = ggml_nrows(op->src[0]) == ggml_nrows(op);
                    ثابت منطقي نشر_or_equal = op->src[0]->ne[0] == 1 || op->src[0]->ne[0] == op->ne[0];

                    إذا (صفاں_equal && نشر_or_equal) {
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
            حالة GGML_OP_جوڑ_صفاں:
                {
                    إذا (op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32) {
                        spacemit_kernels::rvv::forward_جوڑ_صفاں<عائم>(params, op);
                    } وإلا {
                        ggml_compute_forward_جوڑ_صفاں(params, op);
                    }
                }
                أرجع سچ;
            حالة GGML_OP_GET_صفاں:
                {
                    إذا (op->src[0]->type == op->type) {
                        بدل (op->src[0]->type) {
                            حالة GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_get_صفاں<int32_t>(params, op);
                                أرجع سچ;
                            حالة GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_get_صفاں<int16_t>(params, op);
                                أرجع سچ;
                            افتراضي:
                                اكسر;
                        }
                    }

                    ggml_compute_forward_get_صفاں(params, op);
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
            ggml_threadpool_chunk_سیٹ(params->threadpool, nth);
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
                    ggml::cpu::riscv64_spacemit::tls_context.tcm_بفر_ناپ);
            } وإلا {
                spacemit_kernels::rvv::forward_flash_attn_ext_f16_one_chunk_vlen1024_vf16(
                    params, dst, ir0, ir1, ggml::cpu::riscv64_spacemit::tls_context.tcm_بفر,
                    ggml::cpu::riscv64_spacemit::tls_context.tcm_بفر_ناپ);
            }

            current_chunk = ggml_threadpool_chunk_جوڑو(params->threadpool, 1);
        }
    }

    صحيح repack(ggml_tensor * t, ثابت فراغ * انکڑے, size_t انکڑے_ناپ) override {
        حافظہ_نقل(t->انکڑے, انکڑے, انکڑے_ناپ);
        أرجع 0;
    }
};

// Impl By IME1
ساكن ثابت tensor_traits<بلاک_q4_0, 32, 16>  q4_0_16x32_q8_0;
ساكن ثابت tensor_traits<بلاک_q4_1, 32, 16>  q4_1_16x32_q8_0;
ساكن ثابت tensor_traits<بلاک_q4_K, 32, 16>  q4_k_16x32_q8_0;
ساكن ثابت tensor_traits<بلاک_q8_0, 32, 16>  q8_0_16x32_q8_0;
// Impl By IME2
ساكن ثابت tensor_traits<بلاک_q2_K, 256, 32> q2_k_32x256_q8_0;
ساكن ثابت tensor_traits<بلاک_q3_K, 256, 32> q3_k_32x256_q8_0;
ساكن ثابت tensor_traits<بلاک_q4_0, 32, 32>  q4_0_32x32_q8_0;
ساكن ثابت tensor_traits<بلاک_q4_1, 32, 32>  q4_1_32x32_q8_0;
ساكن ثابت tensor_traits<بلاک_q4_0, 256, 32> q4_0_32x256_q8_0;
ساكن ثابت tensor_traits<بلاک_q4_1, 256, 32> q4_1_32x256_q8_0;
ساكن ثابت tensor_traits<بلاک_q4_K, 32, 32>  q4_k_32x32_q8_0;
ساكن ثابت tensor_traits<بلاک_q6_K, 32, 32>  q6_k_32x32_q8_0;
ساكن ثابت tensor_traits<بلاک_q8_0, 32, 32>  q8_0_32x32_q8_0;
ساكن ثابت tensor_traits<بلاک_mxfp4, 32, 32> mxfp4_32x32_q8_0;
ساكن ثابت tensor_traits<بلاک_q5_K, 32, 32>  q5_k_32x32_q8_0;
ساكن ثابت tensor_traits<بلاک_q5_1, 32, 32>  q5_1_32x32_q8_0;
ساكن ثابت tensor_traits<بلاک_q5_0, 32, 32>  q5_0_32x32_q8_0;
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
#جے_انت
            }
            اكسر;
        حالة GGML_TYPE_Q3_K:
            {
#إذا defined(RISCV64_SPACEMIT_IME2)
                إذا (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    أرجع &ggml::cpu::riscv64_spacemit::q3_k_32x256_q8_0;
                }
#جے_انت
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
#جے_انت

#إذا defined(RISCV64_SPACEMIT_IME1)
                إذا (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    أرجع &ggml::cpu::riscv64_spacemit::q4_0_16x32_q8_0;
                }
#جے_انت
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
#جے_انت

#إذا defined(RISCV64_SPACEMIT_IME1)
                إذا (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    أرجع &ggml::cpu::riscv64_spacemit::q4_1_16x32_q8_0;
                }
#جے_انت
            }
            اكسر;
        حالة GGML_TYPE_Q4_K:
            {
#إذا defined(RISCV64_SPACEMIT_IME2)
                إذا (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    أرجع &ggml::cpu::riscv64_spacemit::q4_k_32x32_q8_0;
                }
#جے_انت

#إذا defined(RISCV64_SPACEMIT_IME1)
                إذا (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    أرجع &ggml::cpu::riscv64_spacemit::q4_k_16x32_q8_0;
                }
#جے_انت
            }
            اكسر;
        حالة GGML_TYPE_Q6_K:
            {
#إذا defined(RISCV64_SPACEMIT_IME2)
                إذا ((ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    أرجع &ggml::cpu::riscv64_spacemit::q6_k_32x32_q8_0;
                }
#جے_انت
            }
            اكسر;
        حالة GGML_TYPE_Q8_0:
            {
#إذا defined(RISCV64_SPACEMIT_IME2)
                إذا ((ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    أرجع &ggml::cpu::riscv64_spacemit::q8_0_32x32_q8_0;
                }
#جے_انت

#إذا defined(RISCV64_SPACEMIT_IME1)
                إذا (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    أرجع &ggml::cpu::riscv64_spacemit::q8_0_16x32_q8_0;
                }
#جے_انت
            }
            اكسر;
        حالة GGML_TYPE_MXFP4:
            {
#إذا defined(RISCV64_SPACEMIT_IME2)
                // TODO
                // if (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                //     return &ggml::cpu::riscv64_spacemit::mxfp4_32x32_q8_0;
                // }
#جے_انت
            }
            اكسر;
        حالة GGML_TYPE_Q5_K:
            {
#إذا defined(RISCV64_SPACEMIT_IME2)
                إذا (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    أرجع &ggml::cpu::riscv64_spacemit::q5_k_32x32_q8_0;
                }
#جے_انت
            }
            اكسر;
        حالة GGML_TYPE_Q5_1:
            {
#إذا defined(RISCV64_SPACEMIT_IME2)
                إذا (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    أرجع &ggml::cpu::riscv64_spacemit::q5_1_32x32_q8_0;
                }
#جے_انت
            }
            اكسر;
        حالة GGML_TYPE_Q5_0:
            {
#إذا defined(RISCV64_SPACEMIT_IME2)
                إذا (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    أرجع &ggml::cpu::riscv64_spacemit::q5_0_32x32_q8_0;
                }
#جے_انت
            }
            اكسر;
        افتراضي:
            اكسر;
    }

    أرجع خالی_سوچک;
}

ساكن تعداد ggml_حالت ggml_backend_riscv64_spacemit_بفر_init_tensor(ggml_backend_بفر_t بفر,
                                                                         ggml_tensor *         tensor) {
    tensor->extra =
        (فراغ *) const_cast<ggml::cpu::tensor_traits *>(ggml_riscv64_spacemit_get_optimal_repack_type(tensor));

    GGML_بن_ورتیا(بفر);

    أرجع GGML_حالت_SUCCESS;
}

ساكن فراغ ggml_backend_riscv64_spacemit_بفر_free_بفر(ggml_backend_بفر_t بفر) {
    GGML_ASSERT(بفر);

    فراغ * ادھار = بفر->context;
    إذا (ادھار == خالی_سوچک) {
        أرجع;
    }

    ggml::cpu::riscv64_spacemit::spine_mem_pool_free(ادھار);
}

ساكن فراغ * ggml_backend_riscv64_spacemit_بفر_get_ادھار(ggml_backend_بفر_t بفر) {
    GGML_ASSERT(بفر);

    فراغ * ادھار = بفر->context;
    GGML_ASSERT(ادھار != خالی_سوچک);
    أرجع ادھار;
}

ساكن فراغ ggml_backend_riscv64_spacemit_بفر_memset_tensor(ggml_backend_بفر_t بفر,
                                                               ggml_tensor *         tensor,
                                                               uint8_t               value,
                                                               size_t                وتھ,
                                                               size_t                size) {
    GGML_ASSERT(tensor);
    memset((حرف *) tensor->انکڑے + وتھ, value, size);

    GGML_بن_ورتیا(بفر);
}

ساكن فراغ ggml_backend_riscv64_spacemit_بفر_clear(ggml_backend_بفر_t بفر, uint8_t value) {
    GGML_ASSERT(بفر);

    فراغ * ادھار = بفر->context;
    GGML_ASSERT(ادھار != خالی_سوچک);
    memset(ادھار, value, بفر->size);
}

ساكن فراغ ggml_backend_riscv64_spacemit_بفر_سیٹ_tensor(ggml_backend_بفر_t بفر,
                                                            ggml_tensor *         tensor,
                                                            ثابت فراغ *          انکڑے,
                                                            size_t                وتھ,
                                                            size_t                size) {
    GGML_ASSERT(وتھ == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    خودکار tensor_traits = (ggml::cpu::riscv64_spacemit::tensor_traits_ادھار *) tensor->extra;
    إذا (tensor_traits) {
        خودکار OK = tensor_traits->repack(tensor, انکڑے, size);
        GGML_ASSERT(OK == 0);
    }

    GGML_بن_ورتیا(بفر);
}

ساكن ثابت ggml_backend_بفر_i ggml_backend_riscv64_spacemit_بفر_i = {
    /* .free_buffer     = */ ggml_backend_riscv64_spacemit_بفر_free_بفر,
    /* .get_base        = */ ggml_backend_riscv64_spacemit_بفر_get_ادھار,
    /* .init_tensor     = */ ggml_backend_riscv64_spacemit_بفر_init_tensor,
    /* .memset_tensor   = */ ggml_backend_riscv64_spacemit_بفر_memset_tensor,
    /* .set_tensor      = */ ggml_backend_riscv64_spacemit_بفر_سیٹ_tensor,
    /* .get_tensor      = */ خالی_سوچک,
    /* .set_tensor_2d   = */ خالی_سوچک,
    /* .get_tensor_2d   = */ خالی_سوچک,
    /* .cpy_tensor      = */ خالی_سوچک,
    /* .clear           = */ ggml_backend_riscv64_spacemit_بفر_clear,
    /* .reset           = */ خالی_سوچک,
};

ساكن ثابت حرف * ggml_backend_cpu_riscv64_spacemit_بفر_type_get_name(ggml_backend_بفر_type_t buft) {
    أرجع "CPU_RISCV64_SPACEMIT";

    GGML_بن_ورتیا(buft);
}

ساكن ggml_backend_بفر_t ggml_backend_cpu_riscv64_spacemit_بفر_type_alloc_بفر(ggml_backend_بفر_type_t buft,
                                                                                        size_t size) {
    فراغ * ادھار = ggml::cpu::riscv64_spacemit::spine_mem_pool_alloc(size, 64);
    إذا (ادھار == خالی_سوچک) {
        أرجع خالی_سوچک;
    }

    أرجع ggml_backend_بفر_init(buft, ggml_backend_riscv64_spacemit_بفر_i, ادھار, size);
}

ساكن size_t ggml_backend_cpu_riscv64_spacemit_بفر_type_get_alignment(ggml_backend_بفر_type_t buft) {
    أرجع 64;

    GGML_بن_ورتیا(buft);
}

ساكن size_t ggml_backend_cpu_riscv64_spacemit_nbytes(ggml_backend_بفر_type_t buft, ثابت ggml_tensor * tensor) {
    لكل (صحيح i = 0; i < GGML_MAX_DIMS; ++i) {
        إذا (tensor->ne[i] <= 0) {
            أرجع 0;
        }
    }

    GGML_بن_ورتیا(buft);

    ثابت خودکار plain_nbytes = [&]() {
        size_t کل = ggml_type_ناپ(tensor->type);
        لكل (صحيح i = 0; i < GGML_MAX_DIMS; ++i) {
            کل += (tensor->ne[i] - 1) * tensor->nb[i];
        }
        أرجع کل;
    };

    ثابت size_t blck_ناپ = ggml_blck_ناپ(tensor->type);
    إذا (blck_ناپ == 1) {
        أرجع plain_nbytes();
    }

    ثابت size_t صف_nbytes = tensor->ne[0] * tensor->nb[0] / blck_ناپ;

    ثابت خودکار جوڑو_strided_nbytes = [&](size_t کل, size_t src_بلاک_ناپ, size_t dst_بلاک_ناپ) {
        لكل (صحيح i = 1; i < GGML_MAX_DIMS; ++i) {
            کل += (tensor->ne[i] - 1) * (tensor->nb[i] / src_بلاک_ناپ) * dst_بلاک_ناپ;
        }
        أرجع کل;
    };

    ثابت خودکار remap_بلاک_nbytes = [&](size_t src_بلاک_ناپ, size_t dst_بلاک_ناپ, int64_t padded_صفاں = 0) {
        GGML_ASSERT(صف_nbytes % src_بلاک_ناپ == 0);

        size_t کل =
            جوڑو_strided_nbytes((صف_nbytes / src_بلاک_ناپ) * dst_بلاک_ناپ, src_بلاک_ناپ, dst_بلاک_ناپ);

        إذا (padded_صفاں > 0 && tensor->ne[1] % padded_صفاں != 0) {
            کل += (padded_صفاں - tensor->ne[1] % padded_صفاں) * (tensor->nb[1] / src_بلاک_ناپ) * dst_بلاک_ناپ;
        }

        أرجع کل;
    };

    size_t nbytes = صف_nbytes;
    بدل (tensor->type) {
        حالة GGML_TYPE_Q4_K:
            nbytes = remap_بلاک_nbytes(حجم(بلاک_q4_K), حجم(بلاک_q4_1) * 8);
            اكسر;
        حالة GGML_TYPE_Q6_K:
            nbytes = remap_بلاک_nbytes(حجم(بلاک_q6_K), حجم(بلاک_q8_0) * 8, 32);
            اكسر;
        حالة GGML_TYPE_Q8_0:
            nbytes = remap_بلاک_nbytes(حجم(بلاک_q8_0), حجم(بلاک_q8_0), 32);
            اكسر;
        حالة GGML_TYPE_Q2_K:
            nbytes = remap_بلاک_nbytes(حجم(بلاک_q2_K), حجم(spacemit_kernels::nrow_بلاک_q2_k<1>));
            اكسر;
        حالة GGML_TYPE_Q3_K:
            nbytes = remap_بلاک_nbytes(حجم(بلاک_q3_K), حجم(spacemit_kernels::nrow_بلاک_q3_k<1>));
            اكسر;
        حالة GGML_TYPE_MXFP4:
            nbytes = remap_بلاک_nbytes(حجم(بلاک_mxfp4), حجم(spacemit_kernels::nrow_بلاک_mxfp4<1>));
            اكسر;
        حالة GGML_TYPE_Q5_K:
            nbytes = remap_بلاک_nbytes(حجم(بلاک_q5_K), حجم(spacemit_kernels::nrow_بلاک_q5_1<1>) * 8);
            اكسر;
        حالة GGML_TYPE_Q5_1:
            nbytes = remap_بلاک_nbytes(حجم(بلاک_q5_1), حجم(spacemit_kernels::nrow_بلاک_q5_1<1>));
            اكسر;
        حالة GGML_TYPE_Q5_0:
            nbytes = remap_بلاک_nbytes(حجم(بلاک_q5_0), حجم(spacemit_kernels::nrow_بلاک_q5_0<1>));
            اكسر;
        افتراضي:
            nbytes = جوڑو_strided_nbytes(صف_nbytes, 1, 1);
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
            حالة GGML_OP_جوڑو:
            حالة GGML_OP_SUB:
            حالة GGML_OP_MUL:
            حالة GGML_OP_DIV:
            حالة GGML_OP_FLASH_ATTN_EXT:
            حالة GGML_OP_CONT:
            حالة GGML_OP_CPY:
            حالة GGML_OP_REPEAT:
            حالة GGML_OP_جوڑ_صفاں:
            حالة GGML_OP_GET_صفاں:
            حالة GGML_OP_CONCAT:
                // case GGML_OP_GATED_DELTA_NET:
                أرجع (ggml::cpu::tensor_traits *) (&ggml::cpu::riscv64_spacemit::rvv_impl);
            افتراضي:
                // GGML_ABORT("fatal error");
                اكسر;
        }

        أرجع خالی_سوچک;
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
         /* .get_max_size       = */ خالی_سوچک,
         /* .get_alloc_size     = */ ggml_backend_cpu_riscv64_spacemit_nbytes,
         /* .get_alloc_size_n   = */ عدم,
         /* .is_host            = */ خالی_سوچک,
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
    صحيح  fd, بائٹاں;
    حرف str[32];

    fd = open("/proc/set_ai_thread", O_WRONLY);
    إذا (fd < 0) {
        GGML_LOG_ERROR("try open /proc/set_ai_thread failed\n");
        أرجع -1;
    }

    snprintf(str, 16, "%d", 0);
    بائٹاں = اكتب(fd, str, لڑی_لمبائی(str));
    إذا (بائٹاں < 0) {
        GGML_LOG_ERROR("try write /proc/set_ai_thread failed\n");
        close(fd);
        أرجع -1;
    }

    close(fd);
    أرجع 0;
}

فراغ ggml_backend_cpu_riscv64_spacemit_سیٹ_numa_thread_affinity(صحيح thread_n) {
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
        ثابت خودکار & perfer_core_ids = ggml::cpu::riscv64_spacemit::global_spine_env_info.perfer_core_ids;
        إذا (thread_n < 0 || static_cast<size_t>(thread_n) >= perfer_core_ids.size()) {
            GGML_ABORT("thread_n %d exceeds perfer_core_ids size %zu\n", thread_n, perfer_core_ids.size());
        }
        خودکار perfer_cpu_id = perfer_core_ids[static_cast<size_t>(thread_n)];
        CPU_SET(perfer_cpu_id, &(ggml::cpu::riscv64_spacemit::tls_context.cpuset));
        صحيح s =
            pthread_setaffinity_np(main_thread, حجم(cpu_set_t), &(ggml::cpu::riscv64_spacemit::tls_context.cpuset));
        إذا (s != 0) {
            GGML_ABORT("set thread affinity error for thread_n %d, cpu_id %d\n", thread_n, perfer_cpu_id);
        }

        صحيح ai_cpu_id = perfer_cpu_id - ggml::cpu::riscv64_spacemit::global_spine_env_info.aicpu_id_وتھ;
        ggml::cpu::riscv64_spacemit::tls_context.cpu_id = ai_cpu_id;
        ggml::cpu::riscv64_spacemit::tls_context.tcm_بفر =
            ggml::cpu::riscv64_spacemit::spine_mem_pool_tcm_mem_get(ai_cpu_id);
        ggml::cpu::riscv64_spacemit::tls_context.tcm_بفر_ناپ =
            ggml::cpu::riscv64_spacemit::global_spine_env_info.tcm_blk_ناپ;
    }

    إذا (ggml::cpu::riscv64_spacemit::tls_context.tcm_بفر != خالی_سوچک) {
        فراغ * rt =
            ggml::cpu::riscv64_spacemit::spine_mem_pool_tcm_mem_wait(ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        إذا (rt == خالی_سوچک) {
            GGML_ABORT("wait tcm buffer failed for cpu_id: %d", ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        }
    }
}

فراغ ggml_backend_cpu_riscv64_spacemit_clear_numa_thread_affinity_threaded(صحيح thread_n) {
    إذا (ggml::cpu::riscv64_spacemit::tls_context.tcm_بفر != خالی_سوچک) {
        خودکار rt = ggml::cpu::riscv64_spacemit::spine_mem_pool_tcm_mem_release(
            ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        إذا (rt != 0) {
            GGML_ABORT("release tcm buffer failed for cpu_id: %d", ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        }
    }
}
}
