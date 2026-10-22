#تعریف_کړه GGML_COMMON_IMPL_CPP
#تعریف_کړه GGML_COMMON_DECL_CPP

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
#تېروتنه "riscv v extension or v_intrinsic not enabled"
#وإلا
#ضمن <riscv_vector.h>
#د_که_پای

#إذا !defined(__riscv_zfh) || !defined(__riscv_zvfh)
#تېروتنه "riscv zfh extension not enabled, GGML_RV_ZFH and GGML_RV_ZVFH must be defined to 1"
#د_که_پای

#إذا !defined(__riscv_zba)
#تېروتنه "riscv zba extension not enabled, GGML_RV_ZBA must be defined to 1"
#د_که_پای

#إذا defined(RISCV64_SPACEMIT_IME1) || defined(RISCV64_SPACEMIT_IME2)
#وإلا
#تېروتنه "RISCV64_SPACEMIT_IME1 or RISCV64_SPACEMIT_IME2 not defined"
#د_که_پای

#وإلا

#تېروتنه "riscv not enabled in this build"

#د_که_پای

#إذا defined(__GNUC__)
#لارښوونه GCC diagnostic ignored "-Woverlength-strings"
#لارښوونه GCC diagnostic ignored "-Wcast-qual"
#لارښوونه GCC diagnostic ignored "-Wunused-parameter"
#د_که_پای

// clang-format on

خارجي "C" {
خارجي فراغ ggml_threadpool_chunk_سېټ(بنية ggml_threadpool * tp, صحيح value);
خارجي صحيح  ggml_threadpool_chunk_جمع_کړه(بنية ggml_threadpool * tp, صحيح value);
}

نطاق ggml::cpu::riscv64_spacemit {

بنية TLSContext {
    صحيح       cpu_id{ -1 };
    cpu_set_t cpuset;
    فراغ *    tcm_بفر{ تش_اشاره };
    size_t    tcm_بفر_اندازه{ 0 };
};

thread_local TLSContext tls_context;

کينډۍ <ډول_نوم BLOC_TYPE, int64_t INTER_اندازه, int64_t NB_ستون_شمېر> constexpr size_t get_repacked_بلاک_type_اندازه() {
    إذا constexpr (std::is_same_v<BLOC_TYPE, بلاک_q6_K> || std::is_same_v<BLOC_TYPE, بلاک_q8_0>) {
        أرجع حجم(بلاک_q8_0);
    } وإلا إذا constexpr (std::is_same_v<BLOC_TYPE, بلاک_q4_0>) {
        أرجع حجم(بلاک_q4_0) * INTER_اندازه / QK4_0;
    } وإلا إذا constexpr (std::is_same_v<BLOC_TYPE, بلاک_q4_1> || std::is_same_v<BLOC_TYPE, بلاک_q4_K>) {
        أرجع (حجم(بلاک_q4_0) + حجم(uint8_t)) * INTER_اندازه / QK4_1;
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

کينډۍ <ډول_نوم BLOC_TYPE> constexpr منطقي بلاک_type_has_zp() {
    إذا constexpr (std::is_same_v<BLOC_TYPE, بلاک_q6_K> || std::is_same_v<BLOC_TYPE, بلاک_q8_0> ||
                  std::is_same_v<BLOC_TYPE, بلاک_q3_K> || std::is_same_v<BLOC_TYPE, بلاک_q4_0> ||
                  std::is_same_v<BLOC_TYPE, بلاک_mxfp4> || std::is_same_v<BLOC_TYPE, بلاک_q5_0>) {
        أرجع خطأ;
    } وإلا إذا constexpr (std::is_same_v<BLOC_TYPE, بلاک_q4_1> || std::is_same_v<BLOC_TYPE, بلاک_q4_K> ||
                         std::is_same_v<BLOC_TYPE, بلاک_q2_K> || std::is_same_v<BLOC_TYPE, بلاک_q5_1> ||
                         std::is_same_v<BLOC_TYPE, بلاک_q5_K>) {
        أرجع ریښتیا;
    } وإلا {
        أكد(خطأ);
        أرجع خطأ;
    }
}

صنف tensor_traits_بنسټ : عام ggml::cpu::tensor_traits {
  عام:
    مجازي صحيح repack(ggml_tensor * t, ثابت فراغ * معلومات, size_t معلومات_اندازه) = 0;
};

کينډۍ <ډول_نوم BLOC_TYPE, int64_t INTER_اندازه, int64_t NB_ستون_شمېر> صنف tensor_traits : عام tensor_traits_بنسټ {
    منطقي work_اندازه(صحيح /* n_threads */, ثابت ggml_tensor * op, size_t & size) override {
        بدل (op->op) {
            حالة GGML_OP_MUL_MAT:
                {
                    int64_t src1_nelements = ggml_nelements(op->src[1]);

                    إذا constexpr (std::is_same_v<BLOC_TYPE, بلاک_q2_K> || std::is_same_v<BLOC_TYPE, بلاک_q3_K>) {
                        size =
                            spacemit_kernels::div_پړاو_پورته(src1_nelements, QK_K) * spacemit_kernels::q8k_blk_اندازه(QK_K);
                    } وإلا إذا constexpr (INTER_اندازه == QK4_0) {
                        size = spacemit_kernels::div_پړاو_پورته(src1_nelements, QK4_0) *
                               spacemit_kernels::q8_blk_اندازه(QK4_0, ریښتیا);
                    } وإلا إذا constexpr (INTER_اندازه == 256) {
                        size = spacemit_kernels::div_پړاو_پورته(src1_nelements, 256) *
                               spacemit_kernels::q8_hp_blk_اندازه(256, ریښتیا, ریښتیا);
                    } وإلا {
                        GGML_ABORT("unsupported block type");
                    }

                    size = GGML_PAD(size, حجم(int64_t));

                    أرجع ریښتیا;
                }
            حالة GGML_OP_MUL_MAT_ID:
                {
                    int64_t src1_nelements = ggml_nelements(op->src[1]);

                    إذا constexpr (std::is_same_v<BLOC_TYPE, بلاک_q2_K> || std::is_same_v<BLOC_TYPE, بلاک_q3_K>) {
                        size =
                            spacemit_kernels::div_پړاو_پورته(src1_nelements, QK_K) * spacemit_kernels::q8k_blk_اندازه(QK_K);
                    } وإلا إذا constexpr (INTER_اندازه == QK4_0) {
                        size = spacemit_kernels::div_پړاو_پورته(src1_nelements, QK4_0) *
                               spacemit_kernels::q8_blk_اندازه(QK4_0, ریښتیا);
                    } وإلا إذا constexpr (INTER_اندازه == 256) {
                        size = spacemit_kernels::div_پړاو_پورته(src1_nelements, 256) *
                               spacemit_kernels::q8_hp_blk_اندازه(256, ریښتیا, ریښتیا);
                    } وإلا {
                        GGML_ABORT("unsupported block type");
                    }

                    size = GGML_PAD(size, حجم(int64_t));

                    ثابت int64_t ne02 = op->src[0]->ne[2];  // n_as, n_expert
                    ثابت int64_t ne12 = op->src[1]->ne[2];  // n_tokens

                    ثابت size_t sizeof_mmid_لیکه_mapping = حجم(int64_t);
                    size += sizeof_mmid_لیکه_mapping * ne02 * (ne12 + 1) + (ne02 + 1) * حجم(int64_t);

                    size = GGML_PAD(size, حجم(int64_t));

                    أرجع ریښتیا;
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
                        أرجع ریښتیا;
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
                        أرجع ریښتیا;
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
        constexpr size_t a_blk_len = INTER_اندازه;
        constexpr size_t b_blk_len = INTER_اندازه;

        ثابت ggml_tensor * src0 = op->src[0];
        ثابت ggml_tensor * src1 = op->src[1];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_دوګونی_OP_LOCALS

        صحيح ith = params->ith;
        صحيح nth = params->nth;

        [[maybe_unused]] ثابت تعداد ggml_type type = src0->type;

        فراغ *        w_معلومات  = (فراغ *) src0->معلومات;
        ثابت عائم * feature = (ثابت عائم *) src1->معلومات;
        عائم *       output  = (عائم *) dst->معلومات;

        ثابت int64_t gemm_m = ne11 * ne12 * ne13;
        ثابت int64_t gemm_k = ne10;
        ثابت int64_t gemm_n = ne01;

        spacemit_kernels::quantize_a_لیکه_def       quantize_a_لیکه_i8;
        spacemit_kernels::quantize_a_لیکه_def       quantize_a_4row_i8;
        spacemit_kernels::gemm_kernel_quantize_def gemm_kernel;
        منطقي                                       سېټ_kernel_impl = خطأ;

        int64_t بلاک_stride_a = spacemit_kernels::q8_blk_اندازه(a_blk_len);

#إذا defined(RISCV64_SPACEMIT_IME2)
        إذا (!سېټ_kernel_impl && (global_spine_env_info.use_ime2)) {
            quantize_a_لیکه_i8  = spacemit_kernels::rvv::quantize_a_لیکه_i8;
            quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8;
            بلاک_stride_a     = spacemit_kernels::q8_blk_اندازه(a_blk_len, ریښتیا);

            إذا constexpr (std::is_same_v<BLOC_TYPE, بلاک_q6_K> || std::is_same_v<BLOC_TYPE, بلاک_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i8;
                سېټ_kernel_impl = ریښتیا;
            } وإلا إذا constexpr (std::is_same_v<BLOC_TYPE, بلاک_q4_0> || std::is_same_v<BLOC_TYPE, بلاک_q4_1> ||
                                 std::is_same_v<BLOC_TYPE, بلاک_q4_K>) {
                إذا constexpr (INTER_اندازه == 256) {
                    gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i4_hp;
                    quantize_a_لیکه_i8  = spacemit_kernels::rvv::quantize_a_لیکه_i8_hp;
                    quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8_hp;
                    بلاک_stride_a     = spacemit_kernels::q8_hp_blk_اندازه(a_blk_len, ریښتیا, ریښتیا);
                    سېټ_kernel_impl    = ریښتیا;
                } وإلا {
                    gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i4;
                    quantize_a_لیکه_i8  = spacemit_kernels::rvv::quantize_a_لیکه_i8;
                    quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8;
                    بلاک_stride_a     = spacemit_kernels::q8_blk_اندازه(a_blk_len, ریښتیا);
                    سېټ_kernel_impl    = ریښتیا;
                }
            } وإلا إذا constexpr (std::is_same_v<BLOC_TYPE, بلاک_q2_K>) {
                quantize_a_لیکه_i8  = spacemit_kernels::rvv::quantize_a_لیکه_i8k;
                quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8k;
                بلاک_stride_a     = spacemit_kernels::q8k_blk_اندازه(a_blk_len);

                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i2k;
                سېټ_kernel_impl = ریښتیا;
            } وإلا إذا constexpr (std::is_same_v<BLOC_TYPE, بلاک_q3_K>) {
                quantize_a_لیکه_i8  = spacemit_kernels::rvv::quantize_a_لیکه_i8k;
                quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8k;
                بلاک_stride_a     = spacemit_kernels::q8k_blk_اندازه(a_blk_len);

                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i3k;
                سېټ_kernel_impl = ریښتیا;
            } وإلا إذا constexpr (std::is_same_v<BLOC_TYPE, بلاک_mxfp4>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8mxfp4;
                سېټ_kernel_impl = ریښتیا;
            } وإلا إذا constexpr (std::is_same_v<BLOC_TYPE, بلاک_q5_1> || std::is_same_v<BLOC_TYPE, بلاک_q5_K> ||
                                 std::is_same_v<BLOC_TYPE, بلاک_q5_0>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i5;
                سېټ_kernel_impl = ریښتیا;
            }
        }
#د_که_پای

#إذا defined(RISCV64_SPACEMIT_IME1)
        إذا (!سېټ_kernel_impl && (global_spine_env_info.use_ime1)) {
            quantize_a_لیکه_i8  = spacemit_kernels::ime1::quantize_a_لیکه_i8;
            quantize_a_4row_i8 = spacemit_kernels::ime1::quantize_a_4row_i8;

            إذا constexpr (std::is_same_v<BLOC_TYPE, بلاک_q4_0> || std::is_same_v<BLOC_TYPE, بلاک_q4_1> ||
                          std::is_same_v<BLOC_TYPE, بلاک_q4_K>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i4;
                سېټ_kernel_impl = ریښتیا;
            } وإلا إذا constexpr (std::is_same_v<BLOC_TYPE, بلاک_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i8;
                سېټ_kernel_impl = ریښتیا;
            }
        }
#د_که_پای
        إذا (!سېټ_kernel_impl) {
            GGML_ABORT("no kernel implementation found for the block type");
        }

        ثابت int64_t a_k_blks = spacemit_kernels::div_پړاو_پورته(gemm_k, a_blk_len);
        ثابت int64_t b_k_blks = spacemit_kernels::div_پړاو_پورته(gemm_k, b_blk_len);

        ثابت int64_t لیکه_stride_a        = a_k_blks * بلاک_stride_a;
        ثابت int64_t gemm_workspace_اندازه = GGML_PAD(gemm_m * لیکه_stride_a, alignof(int64_t));

        إذا (ith == 0 && params->wsize < gemm_workspace_اندازه) {
            GGML_ABORT("wsize less than gemm_workspace_size");
        }

        uintptr_t ws_ptr = reinterpret_cast<uintptr_t>(params->wdata);

        فراغ *        tcm_بفر      = ggml::cpu::riscv64_spacemit::tls_context.tcm_بفر;
        ثابت int64_t tcm_بفر_اندازه = ggml::cpu::riscv64_spacemit::tls_context.tcm_بفر_اندازه;

        خپلکاری * quant_a_بفر = reinterpret_cast<uint8_t *>(ws_ptr);

        constexpr int64_t لیکه_align = 4;
        ثابت int64_t     لیکه_blks  = spacemit_kernels::div_پړاو_پورته(gemm_m, لیکه_align);

        ثابت int64_t لیکه_stride_b      = b_k_blks * get_repacked_بلاک_type_اندازه<BLOC_TYPE, INTER_اندازه, NB_ستون_شمېر>();
        ثابت int64_t فی_mb_لیکې_wsize = لیکه_align * لیکه_stride_a;
        ثابت int64_t فی_nb_ستون_شمېر_wsize = NB_ستون_شمېر * لیکه_stride_b;

        ثابت int64_t barrier_idx = static_cast<int64_t>(ith / 2);

        GGML_ASSERT(global_spine_env_info.init_barrier != تش_اشاره);
        GGML_ASSERT(barrier_idx < spine_init_barrier_شمېر);
        spine_barrier_t * cur_barrier = &global_spine_env_info.init_barrier[barrier_idx];

        إذا (gemm_m == 1) {
            صحيح task_فی_thread = spacemit_kernels::div_پړاو_پورته(a_k_blks, nth);
            صحيح a_blk_پیل     = ith * task_فی_thread;
            صحيح a_blk_پای_ټکی       = std::min(a_blk_پیل + task_فی_thread, (صحيح) a_k_blks);
            إذا (a_blk_پیل < a_blk_پای_ټکی) {
                quantize_a_لیکه_i8(a_blk_len, feature + a_blk_پیل * a_blk_len, (a_blk_پای_ټکی - a_blk_پیل) * a_blk_len,
                                  quant_a_بفر + a_blk_پیل * بلاک_stride_a);
            }
        } وإلا {
            صحيح task_فی_thread = spacemit_kernels::div_پړاو_پورته(لیکه_blks, nth);
            صحيح m_لیکه_blk_پیل = ith * task_فی_thread;
            صحيح m_لیکه_blk_پای_ټکی   = std::min(m_لیکه_blk_پیل + task_فی_thread, (صحيح) لیکه_blks);
            لكل (صحيح m_لیکه_blk = m_لیکه_blk_پیل; m_لیکه_blk < m_لیکه_blk_پای_ټکی; m_لیکه_blk++) {
                صحيح m_idx             = m_لیکه_blk * لیکه_align;
                صحيح لیکې_tobe_handled = (gemm_m - m_idx) > لیکه_align ? لیکه_align : (gemm_m - m_idx);

                إذا (لیکې_tobe_handled == لیکه_align && quantize_a_4row_i8 != تش_اشاره) {
                    ثابت عائم * a_لیکه_ptr       = feature + m_idx * gemm_k;
                    خپلکاری *        quant_a_لیکه_ptr = quant_a_بفر + m_idx * لیکه_stride_a;
                    quantize_a_4row_i8(a_blk_len, a_لیکه_ptr, gemm_k, quant_a_لیکه_ptr);
                } وإلا {
                    طالما (لیکې_tobe_handled) {
                        ثابت عائم * a_لیکه_ptr       = feature + m_idx * gemm_k;
                        خپلکاری *        quant_a_لیکه_ptr = quant_a_بفر + m_idx * لیکه_stride_a;
                        quantize_a_لیکه_i8(a_blk_len, a_لیکه_ptr, gemm_k, quant_a_لیکه_ptr);
                        لیکې_tobe_handled -= 1;
                        m_idx += 1;
                    }
                }
            }
        }

        ggml_barrier(params->threadpool);

        ثابت int64_t gemm_m_stride     = gemm_n / gemm_m > 64 ? gemm_m : 16;
        ثابت int64_t gemm_m_blocked    = spacemit_kernels::div_پړاو_پورته(gemm_m, gemm_m_stride);
        ثابت int64_t max_gemm_n_stride = spacemit_kernels::div_پړاو_پورته(gemm_n * gemm_m_blocked, nth);

        int64_t gemm_n_stride = gemm_n;
        إذا (max_gemm_n_stride < gemm_n) {
            gemm_n_stride =
                std::min(gemm_n_stride, spacemit_kernels::div_پړاو_پورته(max_gemm_n_stride, NB_ستون_شمېر) * NB_ستون_شمېر);
        }

        إذا (gemm_n_stride == gemm_n && tcm_بفر != تش_اشاره && فی_mb_لیکې_wsize <= tcm_بفر_اندازه) {
            لكل (int64_t m_پیل = ith * لیکه_align; m_پیل < gemm_m; m_پیل += لیکه_align * nth) {
                uint8_t * b_ستون    = reinterpret_cast<uint8_t *>(w_معلومات);
                uint8_t * b_ستون_zp = بلاک_type_has_zp<BLOC_TYPE>() ? b_ستون : تش_اشاره;

                int64_t m_لیکه_real = std::min(gemm_m - m_پیل, لیکه_align);

                spacemit_kernels::rvv::memcpy1d(tcm_بفر, quant_a_بفر + m_پیل * لیکه_stride_a,
                                                m_لیکه_real * لیکه_stride_a);

                int64_t n_blk_real = 0;
                لكل (int64_t ni = 0; ni < gemm_n; ni += n_blk_real, b_ستون += n_blk_real * لیکه_stride_b) {
                    n_blk_real = std::min(gemm_n - ni, (int64_t) NB_ستون_شمېر);

                    uint8_t * a_لیکه_ptr = (uint8_t *) tcm_بفر;
                    عائم *   c_blk     = output + m_پیل * gemm_n + ni;

                    int32_t لیکې_پاتې_شوی = m_لیکه_real;

                    طالما (لیکې_پاتې_شوی > 0) {
                        خپلکاری لیکې_handled = gemm_kernel(b_blk_len, a_لیکه_ptr, b_ستون, b_ستون_zp, c_blk, لیکې_پاتې_شوی,
                                                        n_blk_real, b_k_blks, gemm_n);

                        c_blk += لیکې_handled * gemm_n;
                        a_لیکه_ptr += لیکې_handled * لیکه_stride_a;

                        لیکې_پاتې_شوی -= لیکې_handled;
                    }
                }
            }
        } وإلا إذا (tcm_بفر != تش_اشاره && فی_nb_ستون_شمېر_wsize <= tcm_بفر_اندازه) {
            uint8_t * a_لیکه = quant_a_بفر;
            uint8_t * b_ستون = reinterpret_cast<uint8_t *>(tcm_بفر);
            إذا ((gemm_workspace_اندازه + فی_nb_ستون_شمېر_wsize) <= tcm_بفر_اندازه) {
                a_لیکه = (uint8_t *) tcm_بفر;
                b_ستون = reinterpret_cast<uint8_t *>(tcm_بفر) + gemm_workspace_اندازه;
            }
            uint8_t * b_ستون_zp = بلاک_type_has_zp<BLOC_TYPE>() ? b_ستون : تش_اشاره;

            int64_t ni      = ith * NB_ستون_شمېر;
            int64_t nb_real = std::min(gemm_n - ni, NB_ستون_شمېر);

            إذا (ith % 2 == 0 && nb_real > 0) {
                spacemit_kernels::rvv::memcpy1d(b_ستون, reinterpret_cast<uint8_t *>(w_معلومات) + ni * لیکه_stride_b,
                                                nb_real * لیکه_stride_b);
                إذا (a_لیکه != quant_a_بفر) {
                    spacemit_kernels::rvv::memcpy1d(a_لیکه, quant_a_بفر, gemm_workspace_اندازه);
                }
            }

            spine_barrier_wait(cur_barrier);

            إذا (ith % 2 != 0 && nb_real > 0) {
                إذا (a_لیکه != quant_a_بفر) {
                    spacemit_kernels::rvv::memcpy1d(a_لیکه, quant_a_بفر, gemm_workspace_اندازه);
                }
                spacemit_kernels::rvv::memcpy1d(b_ستون, reinterpret_cast<uint8_t *>(w_معلومات) + ni * لیکه_stride_b,
                                                nb_real * لیکه_stride_b);
            }

            لكل (; ni < gemm_n; ni += NB_ستون_شمېر * nth) {
                int64_t لیکې_پاتې_شوی = gemm_m;
                عائم * c_blk          = output + ni;
                خپلکاری *  a_لیکه_cur      = a_لیکه;

                إذا (ith % 2 != 0) {
                    spine_barrier_wait(cur_barrier);
                }

                طالما (لیکې_پاتې_شوی > 0) {
                    خپلکاری لیکې_handled = gemm_kernel(b_blk_len, a_لیکه_cur, b_ستون, b_ستون_zp, c_blk, لیکې_پاتې_شوی,
                                                    nb_real, b_k_blks, gemm_n);

                    c_blk += لیکې_handled * gemm_n;
                    a_لیکه_cur += لیکې_handled * لیکه_stride_a;

                    لیکې_پاتې_شوی -= لیکې_handled;
                }

                إذا (ith % 2 == 0) {
                    spine_barrier_wait(cur_barrier);
                }

                ثابت int64_t بل_ni = ni + NB_ستون_شمېر * nth;
                إذا (بل_ni < gemm_n) {
                    nb_real = std::min(gemm_n - بل_ni, NB_ستون_شمېر);
                    spacemit_kernels::rvv::memcpy1d(b_ستون, reinterpret_cast<uint8_t *>(w_معلومات) + بل_ni * لیکه_stride_b,
                                                    nb_real * لیکه_stride_b);
                }
            }
        } وإلا {
            ثابت int64_t task_شمېر_m = spacemit_kernels::div_پړاو_پورته(gemm_m, gemm_m_stride);
            ثابت int64_t task_شمېر_n = spacemit_kernels::div_پړاو_پورته(gemm_n, gemm_n_stride);

            int64_t task_شمېر      = task_شمېر_m * task_شمېر_n;
            int64_t task_فی_thread = (task_شمېر + nth - 1) / nth;
            int64_t start           = ith * task_فی_thread;
            int64_t end             = std::min((ith + 1) * task_فی_thread, task_شمېر);
            لكل (int64_t compute_idx = start; compute_idx < end; compute_idx++) {
                ثابت خپلکاری tid_n = compute_idx / task_شمېر_m;
                ثابت خپلکاری tid_m = compute_idx % task_شمېر_m;

                ثابت int64_t m_پیل = tid_m * gemm_m_stride;
                ثابت int64_t m_شمېر = std::min(gemm_m - m_پیل, (int64_t) gemm_m_stride);

                ثابت int64_t n_پیل = tid_n * gemm_n_stride;
                ثابت int64_t n_شمېر = std::min(gemm_n - n_پیل, (int64_t) gemm_n_stride);

                ثابت int64_t n_blk = m_شمېر == 1 ? n_شمېر : NB_ستون_شمېر;

                uint8_t * b_ستون    = reinterpret_cast<uint8_t *>(w_معلومات) + n_پیل * لیکه_stride_b;
                uint8_t * b_ستون_zp = بلاک_type_has_zp<BLOC_TYPE>() ? b_ستون : تش_اشاره;

                int64_t n_blk_real = 0;
                لكل (int64_t ni = 0; ni < n_شمېر; ni += n_blk_real, b_ستون += n_blk_real * لیکه_stride_b) {
                    n_blk_real = std::min(n_شمېر - ni, n_blk);

                    uint8_t * a_لیکه = quant_a_بفر + m_پیل * لیکه_stride_a;

                    عائم * c_blk = output + m_پیل * gemm_n + n_پیل + ni;

                    int64_t لیکې_پاتې_شوی = m_شمېر;

                    uint8_t * b_ستون_cur    = b_ستون;
                    uint8_t * b_ستون_zp_cur = b_ستون_zp;

                    طالما (لیکې_پاتې_شوی > 0) {
                        خپلکاری لیکې_handled = gemm_kernel(b_blk_len, a_لیکه, b_ستون_cur, b_ستون_zp_cur, c_blk,
                                                        لیکې_پاتې_شوی, n_blk_real, b_k_blks, gemm_n);

                        c_blk += لیکې_handled * gemm_n;
                        a_لیکه += لیکې_handled * لیکه_stride_a;

                        لیکې_پاتې_شوی -= لیکې_handled;
                    }
                }
            }
        }
    }

    فراغ forward_mul_mat_id(ggml_compute_params * params, ggml_tensor * op) {
        constexpr size_t a_blk_len = INTER_اندازه;
        constexpr size_t b_blk_len = INTER_اندازه;

        ثابت ggml_tensor * src0 = op->src[0];
        ثابت ggml_tensor * src1 = op->src[1];
        ثابت ggml_tensor * ids  = op->src[2];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_دوګونی_OP_LOCALS

        صحيح ith = params->ith;
        صحيح nth = params->nth;

        // row groups
        ثابت صحيح n_ids = ids->ne[0];  // n_expert_used
        ثابت صحيح n_as  = ne02;        // n_expert

        بنية mmid_لیکه_mapping {
            int32_t i1;
            int32_t i2;
        };

        spacemit_kernels::quantize_a_لیکه_def           quantize_a_لیکه_i8;
        spacemit_kernels::gemm_kernel_quantize_def     gemm_kernel;
        spacemit_kernels::moe_gemm_kernel_quantize_def moe_gemm_kernel_m2;
        منطقي                                           سېټ_kernel_impl = خطأ;
        size_t                                         بلاک_stride_a  = spacemit_kernels::q8_blk_اندازه(QK4_0);

#إذا defined(RISCV64_SPACEMIT_IME2)
        إذا (!سېټ_kernel_impl && (global_spine_env_info.use_ime2)) {
            quantize_a_لیکه_i8 = spacemit_kernels::rvv::quantize_a_لیکه_i8;
            بلاک_stride_a    = spacemit_kernels::q8_blk_اندازه(QK4_0, ریښتیا);

            إذا constexpr (std::is_same_v<BLOC_TYPE, بلاک_q6_K> || std::is_same_v<BLOC_TYPE, بلاک_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i8;
                سېټ_kernel_impl = ریښتیا;
            } وإلا إذا constexpr (std::is_same_v<BLOC_TYPE, بلاک_q4_0> || std::is_same_v<BLOC_TYPE, بلاک_q4_1> ||
                                 std::is_same_v<BLOC_TYPE, بلاک_q4_K>) {
                إذا constexpr (INTER_اندازه == 256) {
                    gemm_kernel       = spacemit_kernels::ime2::gemm_kernel_i8i4_hp;
                    quantize_a_لیکه_i8 = spacemit_kernels::rvv::quantize_a_لیکه_i8_hp;
                    بلاک_stride_a    = spacemit_kernels::q8_hp_blk_اندازه(a_blk_len, ریښتیا, ریښتیا);
                    سېټ_kernel_impl   = ریښتیا;
                } وإلا {
                    gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i4;
                    moe_gemm_kernel_m2 = spacemit_kernels::ime2::moe_m2_gemm_kernel_i8i4;
                    quantize_a_لیکه_i8  = spacemit_kernels::rvv::quantize_a_لیکه_i8;
                    بلاک_stride_a     = spacemit_kernels::q8_blk_اندازه(a_blk_len, ریښتیا);
                    سېټ_kernel_impl    = ریښتیا;
                }
            } وإلا إذا constexpr (std::is_same_v<BLOC_TYPE, بلاک_q2_K>) {
                quantize_a_لیکه_i8 = spacemit_kernels::rvv::quantize_a_لیکه_i8k;
                بلاک_stride_a    = spacemit_kernels::q8k_blk_اندازه(a_blk_len);
                gemm_kernel       = spacemit_kernels::ime2::gemm_kernel_i8i2k;
                سېټ_kernel_impl   = ریښتیا;
            } وإلا إذا constexpr (std::is_same_v<BLOC_TYPE, بلاک_q3_K>) {
                quantize_a_لیکه_i8 = spacemit_kernels::rvv::quantize_a_لیکه_i8k;
                بلاک_stride_a    = spacemit_kernels::q8k_blk_اندازه(a_blk_len);
                gemm_kernel       = spacemit_kernels::ime2::gemm_kernel_i8i3k;
                سېټ_kernel_impl   = ریښتیا;
            } وإلا إذا constexpr (std::is_same_v<BLOC_TYPE, بلاک_mxfp4>) {
                gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8mxfp4;
                moe_gemm_kernel_m2 = spacemit_kernels::ime2::moe_m2_gemm_kernel_i8mxfp4;
                سېټ_kernel_impl    = ریښتیا;
            } وإلا إذا constexpr (std::is_same_v<BLOC_TYPE, بلاک_q5_1> || std::is_same_v<BLOC_TYPE, بلاک_q5_K> ||
                                 std::is_same_v<BLOC_TYPE, بلاک_q5_0>) {
                gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i5;
                moe_gemm_kernel_m2 = spacemit_kernels::ime2::moe_m2_gemm_kernel_i8i5;
                سېټ_kernel_impl    = ریښتیا;
            }
        }
#د_که_پای

#إذا defined(RISCV64_SPACEMIT_IME1)
        إذا (!سېټ_kernel_impl && (global_spine_env_info.use_ime1)) {
            quantize_a_لیکه_i8 = spacemit_kernels::ime1::quantize_a_لیکه_i8;

            إذا constexpr (std::is_same_v<BLOC_TYPE, بلاک_q4_0> || std::is_same_v<BLOC_TYPE, بلاک_q4_1> ||
                          std::is_same_v<BLOC_TYPE, بلاک_q4_K>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i4;
                سېټ_kernel_impl = ریښتیا;
            } وإلا إذا constexpr (std::is_same_v<BLOC_TYPE, بلاک_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i8;
                سېټ_kernel_impl = ریښتیا;
            }
        }
#د_که_پای
        إذا (!سېټ_kernel_impl) {
            GGML_ABORT("no kernel implementation found for the block type");
        }

        ثابت size_t a_k_blks = spacemit_kernels::div_پړاو_پورته(ne10, a_blk_len);
        ثابت size_t b_k_blks = spacemit_kernels::div_پړاو_پورته(ne10, b_blk_len);

        ثابت size_t nbw1                = a_k_blks * بلاک_stride_a;
        ثابت size_t nbw2                = ne11 * nbw1;
        ثابت size_t nbw3                = nbw2 * ne12;
        ثابت size_t gemm_workspace_اندازه = GGML_PAD(nbw3, alignof(int64_t));

        ثابت uintptr_t ws_ptr         = reinterpret_cast<uintptr_t>(params->wdata);
        خپلکاری *          quant_a_بفر = reinterpret_cast<uint8_t *>(ws_ptr);

        إذا (ne11 == 1) {
            لكل (int64_t ii = ith; ii < ne12 * a_k_blks; ii += nth) {
                int64_t i12       = ii / a_k_blks;
                int64_t ak_blk_id = ii % a_k_blks;
                quantize_a_لیکه_i8(a_blk_len, (عائم *) ((حرف *) src1->معلومات + i12 * nb12) + ak_blk_id * a_blk_len,
                                  a_blk_len, quant_a_بفر + i12 * nbw2 + ak_blk_id * بلاک_stride_a);
            }
        } وإلا {
            لكل (int64_t ii = ith; ii < ne12 * ne11; ii += nth) {
                int64_t i12 = ii / ne11;
                int64_t i11 = ii % ne11;
                quantize_a_لیکه_i8(a_blk_len, (عائم *) ((حرف *) src1->معلومات + i12 * nb12 + i11 * nb11), ne10,
                                  quant_a_بفر + i12 * nbw2 + i11 * nbw1);
            }
        }

#تعریف_کړه MMID_ماتریکس_لیکه(لیکه_id, i1) ماتریکس_لیکې[(لیکه_id) *ne12 + (i1)]

        int64_t *          ماتریکس_لیکه_counts       = (int64_t *) (ws_ptr + gemm_workspace_اندازه);
        int32_t *          valid_ep_شمېر          = (int32_t *) (ماتریکس_لیکه_counts + n_as);
        int32_t *          valid_act_شمېر         = (int32_t *) (valid_ep_شمېر + 1);
        int64_t *          valid_ماتریکس_لیکه_counts = (int64_t *) (valid_act_شمېر + 1);
        mmid_لیکه_mapping * ماتریکس_لیکې             = (mmid_لیکه_mapping *) (valid_ماتریکس_لیکه_counts + n_as);

        إذا (ith == 0) {
            // initialize matrix_row_counts
            memset(ماتریکس_لیکه_counts, 0, n_as * حجم(int64_t));

            // group rows by src0 matrix
            لكل (int32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
                لكل (int32_t id = 0; id < n_ids; ++id) {
                    ثابت int32_t i02 =
                        *(ثابت int32_t *) ((ثابت حرف *) ids->معلومات + iid1 * ids->nb[1] + id * ids->nb[0]);

                    GGML_ASSERT(i02 >= 0 && i02 < n_as);

                    MMID_ماتریکس_لیکه(i02, ماتریکس_لیکه_counts[i02]) = { id, iid1 };
                    ماتریکس_لیکه_counts[i02] += 1;
                }
            }

            int32_t valid_ep_شمېر_t  = 0;
            int32_t valid_act_شمېر_t = 0;
            لكل (صحيح cur_a = 0; cur_a < n_as; ++cur_a) {
                ثابت int64_t cne1 = ماتریکس_لیکه_counts[cur_a];
                إذا (cne1 == 0) {
                    استمر;
                }
                valid_ماتریکس_لیکه_counts[valid_ep_شمېر_t] = cur_a;
                valid_act_شمېر_t += cne1;
                valid_ep_شمېر_t += 1;
            }
            valid_ep_شمېر[0]  = valid_ep_شمېر_t;
            valid_act_شمېر[0] = valid_act_شمېر_t;
        }

        ثابت int64_t barrier_idx = static_cast<int64_t>(ith / 2);

        GGML_ASSERT(global_spine_env_info.init_barrier != تش_اشاره);
        GGML_ASSERT(barrier_idx < spine_init_barrier_شمېر);
        spine_barrier_t * cur_barrier = &global_spine_env_info.init_barrier[barrier_idx];

        ggml_barrier(params->threadpool);

        ثابت size_t لیکه_stride_b      = b_k_blks * get_repacked_بلاک_type_اندازه<BLOC_TYPE, INTER_اندازه, NB_ستون_شمېر>();
        ثابت size_t expert_b_stride   = ne01 * لیکه_stride_b;
        ثابت size_t فی_nb_ستون_شمېر_wsize = NB_ستون_شمېر * لیکه_stride_b;

        std::array<ثابت uint8_t *, 2> src_workspaces;
        std::array<عائم *, 2>         dst_workspaces;

        خپلکاری *     tcm_بفر      = ggml::cpu::riscv64_spacemit::tls_context.tcm_بفر;
        ثابت خپلکاری tcm_بفر_اندازه = ggml::cpu::riscv64_spacemit::tls_context.tcm_بفر_اندازه;

        ثابت خپلکاری valid_ep_شمېر_t  = valid_ep_شمېر[0];
        ثابت خپلکاری valid_act_شمېر_t = valid_act_شمېر[0];

        صحيح nth_es = 1;
        صحيح nth_n  = nth;

        صحيح ith_es = ith % nth_es;
        صحيح ith_n  = (ith / nth_es) % nth_n;

        إذا (valid_ep_شمېر_t % nth == 0 && tcm_بفر != تش_اشاره && valid_ep_شمېر_t == n_as &&
            valid_act_شمېر_t == n_as && فی_nb_ستون_شمېر_wsize <= tcm_بفر_اندازه) {
            لكل (int64_t valid_id = ith; valid_id < valid_ep_شمېر_t; valid_id += nth) {
                ثابت int64_t cur_a = valid_ماتریکس_لیکه_counts[valid_id];

                خپلکاری * src0_cur = (uint8_t *) src0->معلومات + cur_a * expert_b_stride;

                mmid_لیکه_mapping لیکه_mapping = MMID_ماتریکس_لیکه(cur_a, 0);
                ثابت صحيح        id          = لیکه_mapping.i1;
                ثابت int64_t    i11         = id % ne11;
                ثابت int64_t    i12         = لیکه_mapping.i2;
                ثابت int64_t    i1          = id;
                ثابت int64_t    i2          = i12;

                خپلکاری *  src1_ستون = quant_a_بفر + (i11 * nbw1 + i12 * nbw2);
                عائم * c_blk    = (عائم *) ((حرف *) dst->معلومات + (i1 * nb1 + i2 * nb2));

                uint8_t * a_لیکه = src1_ستون;
                uint8_t * b_ستون = reinterpret_cast<uint8_t *>(tcm_بفر);
                إذا ((nbw1 + فی_nb_ستون_شمېر_wsize) <= tcm_بفر_اندازه) {
                    a_لیکه = (uint8_t *) tcm_بفر;
                    b_ستون = reinterpret_cast<uint8_t *>(tcm_بفر) + nbw1;
                }
                uint8_t * b_ستون_zp = بلاک_type_has_zp<BLOC_TYPE>() ? b_ستون : تش_اشاره;

                إذا (ith % 2 == 0) {
                    spacemit_kernels::rvv::memcpy1d(b_ستون, reinterpret_cast<uint8_t *>(src0_cur), فی_nb_ستون_شمېر_wsize);

                    إذا (a_لیکه != src1_ستون) {
                        spacemit_kernels::rvv::memcpy1d(a_لیکه, src1_ستون, nbw1);
                    }
                }

                spine_barrier_wait(cur_barrier);

                إذا (ith % 2 != 0) {
                    إذا (a_لیکه != src1_ستون) {
                        spacemit_kernels::rvv::memcpy1d(a_لیکه, src1_ستون, nbw1);
                    }

                    spacemit_kernels::rvv::memcpy1d(b_ستون, reinterpret_cast<uint8_t *>(src0_cur), فی_nb_ستون_شمېر_wsize);
                }

                int64_t nb_real = std::min(ne01, NB_ستون_شمېر);
                لكل (int64_t ni = 0; ni < ne01; ni += NB_ستون_شمېر) {
                    إذا (ith % 2 != 0) {
                        spine_barrier_wait(cur_barrier);
                    }

                    gemm_kernel(b_blk_len, a_لیکه, b_ستون, b_ستون_zp, c_blk + ni, 1, nb_real, b_k_blks, ne01);

                    إذا (ith % 2 == 0) {
                        spine_barrier_wait(cur_barrier);
                    }

                    ثابت int64_t بل_ni = ni + NB_ستون_شمېر;
                    إذا (بل_ni < ne01) {
                        nb_real = std::min(ne01 - بل_ni, NB_ستون_شمېر);
                        spacemit_kernels::rvv::memcpy1d(
                            b_ستون, reinterpret_cast<uint8_t *>(src0_cur) + بل_ni * لیکه_stride_b, فی_nb_ستون_شمېر_wsize);
                    }
                }
            }
        } وإلا {
            لكل (int64_t valid_id = ith_es; valid_id < valid_ep_شمېر_t; valid_id += nth_es) {
                ثابت int64_t cur_a = valid_ماتریکس_لیکه_counts[valid_id];
                ثابت int64_t cne1  = ماتریکس_لیکه_counts[cur_a];

                int64_t src1_cur_پیل = 0;
                int64_t src1_cur_پای_ټکی   = cne1;

                int64_t src0_cur_پیل = (ith_n * ne01) / nth_n;
                int64_t src0_cur_پای_ټکی   = MIN(((ith_n + 1) * ne01) / nth_n, ne01);

                إذا (src1_cur_پیل >= src1_cur_پای_ټکی || src0_cur_پیل >= src0_cur_پای_ټکی) {
                    استمر;
                }

                src0_cur_پیل =
                    (src0_cur_پیل % NB_ستون_شمېر) ? src0_cur_پیل + NB_ستون_شمېر - (src0_cur_پیل % NB_ستون_شمېر) : src0_cur_پیل;
                src0_cur_پای_ټکی =
                    (src0_cur_پای_ټکی % NB_ستون_شمېر) ? src0_cur_پای_ټکی + NB_ستون_شمېر - (src0_cur_پای_ټکی % NB_ستون_شمېر) : src0_cur_پای_ټکی;

                خپلکاری *    src0_cur = (uint8_t *) src0->معلومات + cur_a * expert_b_stride + src0_cur_پیل * لیکه_stride_b;
                uint8_t * b_ستون_zp = بلاک_type_has_zp<BLOC_TYPE>() ? src0_cur : تش_اشاره;

                size_t extra_tcm_بفر_اندازه = tcm_بفر_اندازه;
                فراغ * extra_tcm_بفر      = tcm_بفر;
                إذا (tcm_بفر != تش_اشاره && (src1_cur_پای_ټکی - src1_cur_پیل) >= 4 &&
                    (src0_cur_پای_ټکی - src0_cur_پیل) * لیکه_stride_b <= tcm_بفر_اندازه) {
                    spacemit_kernels::rvv::memcpy1d(tcm_بفر, src0_cur,
                                                    (src0_cur_پای_ټکی - src0_cur_پیل) * لیکه_stride_b);
                    src0_cur = reinterpret_cast<uint8_t *>(tcm_بفر);
                    b_ستون_zp = بلاک_type_has_zp<BLOC_TYPE>() ? src0_cur : تش_اشاره;
                    extra_tcm_بفر_اندازه -= (src0_cur_پای_ټکی - src0_cur_پیل) * لیکه_stride_b;
                    extra_tcm_بفر = reinterpret_cast<فراغ *>(reinterpret_cast<uintptr_t>(tcm_بفر) +
                                                                (src0_cur_پای_ټکی - src0_cur_پیل) * لیکه_stride_b);
                }

                صحيح ir1 = src1_cur_پیل;

                إذا (extra_tcm_بفر_اندازه >= nbw1 && extra_tcm_بفر != تش_اشاره) {
                    int64_t quant_a_tile_اندازه = extra_tcm_بفر_اندازه / nbw1;
                    افعل {
                        quant_a_tile_اندازه = MIN(quant_a_tile_اندازه, src1_cur_پای_ټکی - ir1);

                        uint8_t * quant_a_tile_بفر = reinterpret_cast<uint8_t *>(extra_tcm_بفر);

                        صحيح iir1 = ir1;
                        لكل (; iir1 < (ir1 + quant_a_tile_اندازه); ++iir1) {
                            mmid_لیکه_mapping لیکه_mapping = MMID_ماتریکس_لیکه(cur_a, iir1);

                            ثابت صحيح id = لیکه_mapping.i1;  // selected expert index

                            ثابت int64_t i11 = id % ne11;
                            ثابت int64_t i12 = لیکه_mapping.i2;  // row index in src1

                            خپلکاری * src1_ستون = quant_a_بفر + (i11 * nbw1 + i12 * nbw2);
                            spacemit_kernels::rvv::memcpy1d(quant_a_tile_بفر, src1_ستون, nbw1);
                            quant_a_tile_بفر = quant_a_tile_بفر + nbw1;
                        }

                        quant_a_tile_بفر = reinterpret_cast<uint8_t *>(extra_tcm_بفر);
                        iir1                = ir1;

                        إذا (moe_gemm_kernel_m2 != تش_اشاره) {
                            لكل (; iir1 < (ir1 + quant_a_tile_اندازه - 1); iir1 += 2, quant_a_tile_بفر += 2 * nbw1) {
                                mmid_لیکه_mapping لیکه_mapping_0 = MMID_ماتریکس_لیکه(cur_a, iir1);
                                mmid_لیکه_mapping لیکه_mapping_1 = MMID_ماتریکس_لیکه(cur_a, iir1 + 1);

                                src_workspaces[0] = quant_a_tile_بفر;
                                src_workspaces[1] = quant_a_tile_بفر + nbw1;

                                dst_workspaces[0] =
                                    (عائم *) ((حرف *) dst->معلومات + (لیکه_mapping_0.i1 * nb1 + لیکه_mapping_0.i2 * nb2)) +
                                    src0_cur_پیل;
                                dst_workspaces[1] = (عائم *) ((حرف *) dst->معلومات +
                                                               ((لیکه_mapping_1.i1) * nb1 + (لیکه_mapping_1.i2) * nb2)) +
                                                    src0_cur_پیل;
                                moe_gemm_kernel_m2(b_blk_len, src_workspaces.معلومات(), src0_cur, b_ستون_zp,
                                                   dst_workspaces.معلومات(), 1, src0_cur_پای_ټکی - src0_cur_پیل, b_k_blks,
                                                   ne01);
                            }
                        }

                        لكل (; iir1 < (ir1 + quant_a_tile_اندازه); iir1++, quant_a_tile_بفر += nbw1) {
                            mmid_لیکه_mapping لیکه_mapping_0 = MMID_ماتریکس_لیکه(cur_a, iir1);

                            gemm_kernel(
                                b_blk_len, quant_a_tile_بفر, src0_cur, b_ستون_zp,
                                (عائم *) ((حرف *) dst->معلومات + (لیکه_mapping_0.i1 * nb1 + لیکه_mapping_0.i2 * nb2)) +
                                    src0_cur_پیل,
                                1, src0_cur_پای_ټکی - src0_cur_پیل, b_k_blks, ne01);
                        }

                        ir1 += quant_a_tile_اندازه;
                    } طالما (ir1 < src1_cur_پای_ټکی);
                } وإلا {
                    إذا (moe_gemm_kernel_m2 != تش_اشاره) {
                        لكل (; ir1 < src1_cur_پای_ټکی - 1; ir1 += 2) {
                            لكل (صحيح iir1 = 0; iir1 < 2; ++iir1) {
                                mmid_لیکه_mapping لیکه_mapping = MMID_ماتریکس_لیکه(cur_a, ir1 + iir1);

                                ثابت صحيح id = لیکه_mapping.i1;  // selected expert index

                                ثابت int64_t i11 = id % ne11;
                                ثابت int64_t i12 = لیکه_mapping.i2;  // row index in src1

                                ثابت int64_t i1 = id;               // selected expert index
                                ثابت int64_t i2 = i12;              //لیکه

                                src_workspaces[iir1] = quant_a_بفر + (i11 * nbw1 + i12 * nbw2);

                                dst_workspaces[iir1] =
                                    (عائم *) ((حرف *) dst->معلومات + (i1 * nb1 + i2 * nb2)) + src0_cur_پیل;
                            }

                            moe_gemm_kernel_m2(b_blk_len, src_workspaces.معلومات(), src0_cur, b_ستون_zp,
                                               dst_workspaces.معلومات(), 1, src0_cur_پای_ټکی - src0_cur_پیل, b_k_blks, ne01);
                        }
                    }

                    لكل (; ir1 < src1_cur_پای_ټکی; ir1++) {
                        mmid_لیکه_mapping لیکه_mapping = MMID_ماتریکس_لیکه(cur_a, ir1);

                        ثابت صحيح id = لیکه_mapping.i1;  // selected expert index

                        ثابت int64_t i11 = id % ne11;
                        ثابت int64_t i12 = لیکه_mapping.i2;  // row index in src1

                        ثابت int64_t i1 = id;               // selected expert index
                        ثابت int64_t i2 = i12;              //لیکه

                        خپلکاری * src1_ستون = quant_a_بفر + (i11 * nbw1 + i12 * nbw2);

                        gemm_kernel(b_blk_len, src1_ستون, src0_cur, b_ستون_zp,
                                    (عائم *) ((حرف *) dst->معلومات + (i1 * nb1 + i2 * nb2)) + src0_cur_پیل, 1,
                                    src0_cur_پای_ټکی - src0_cur_پیل, b_k_blks, ne01);
                    }
                }
            }
        }
#تعریف_لغوه MMID_ماتریکس_لیکه
    }

    صحيح repack(ggml_tensor * t, ثابت فراغ * معلومات, size_t معلومات_اندازه) override {
        GGML_LOG_DEBUG("%s: repack tensor %s with %s_%dx%d\n", __func__, t->name, ggml_type_name(t->type),
                       (صحيح) NB_ستون_شمېر, (صحيح) INTER_اندازه);
        أرجع ggml::cpu::riscv64_spacemit::repack<BLOC_TYPE, INTER_اندازه, NB_ستون_شمېر>(t, معلومات, معلومات_اندازه);
    }
};

صنف tensor_traits_common : عام tensor_traits_بنسټ {
    منطقي work_اندازه(صحيح n_تارونه, ثابت ggml_tensor * op, size_t & size) override {
        بدل (op->op) {
            حالة GGML_OP_FLASH_ATTN_EXT:
                {
                    ثابت صحيح     n_tasks = n_تارونه;
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
                أرجع ریښتیا;
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
                        أرجع ریښتیا;
                    افتراضي:
                        GGML_ABORT("fatal error");
                }
            حالة GGML_OP_RMS_NORM:
                بدل (op->src[0]->type) {
                    حالة GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_rms_norm_f32(params, op);
                        أرجع ریښتیا;
                    افتراضي:
                        GGML_ABORT("fatal error");
                }
            حالة GGML_OP_جمع_کړه:
                بدل (op->src[0]->type) {
                    حالة GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_دوګونی<GGML_OP_جمع_کړه, عائم>(params, op);
                        أرجع ریښتیا;
                    حالة GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_دوګونی<GGML_OP_جمع_کړه, _Float16>(params, op);
                        أرجع ریښتیا;
                    افتراضي:
                        ggml_compute_forward_جمع_کړه(params, op);
                        أرجع ریښتیا;
                }
            حالة GGML_OP_SUB:
                بدل (op->src[0]->type) {
                    حالة GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_دوګونی<GGML_OP_SUB, عائم>(params, op);
                        أرجع ریښتیا;
                    حالة GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_دوګونی<GGML_OP_SUB, _Float16>(params, op);
                        أرجع ریښتیا;
                    افتراضي:
                        ggml_compute_forward_sub(params, op);
                        أرجع ریښتیا;
                }
            حالة GGML_OP_MUL:
                بدل (op->src[0]->type) {
                    حالة GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_دوګونی<GGML_OP_MUL, عائم>(params, op);
                        أرجع ریښتیا;
                    حالة GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_دوګونی<GGML_OP_MUL, _Float16>(params, op);
                        أرجع ریښتیا;
                    افتراضي:
                        ggml_compute_forward_mul(params, op);
                        أرجع ریښتیا;
                }
            حالة GGML_OP_DIV:
                بدل (op->src[0]->type) {
                    حالة GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_دوګونی<GGML_OP_DIV, عائم>(params, op);
                        أرجع ریښتیا;
                    حالة GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_دوګونی<GGML_OP_DIV, _Float16>(params, op);
                        أرجع ریښتیا;
                    افتراضي:
                        ggml_compute_forward_div(params, op);
                        أرجع ریښتیا;
                }
            حالة GGML_OP_FLASH_ATTN_EXT:
                forward_flash_attn_ext_f16(params, op);
                أرجع ریښتیا;
            حالة GGML_OP_CONT:
                {
                    ثابت ggml_tensor * src0 = op->src[0];
                    إذا (op->type == src0->type && op->nb[0] != src0->nb[0] && op->nb[0] == src0->nb[1] &&
                        op->ne[3] * op->ne[2] * op->nb[2] == src0->ne[3] * src0->ne[2] * src0->nb[2]) {
                        spacemit_kernels::rvv::forward_cont_with_ترتیب_بدلول(params, op);
                    } وإلا {
                        ggml_compute_forward_cont(params, op);
                    }
                    أرجع ریښتیا;
                }
            حالة GGML_OP_CPY:
                {
                    ثابت ggml_tensor * src0 = op->src[0];
                    إذا (op->type == src0->type && op->nb[0] == src0->nb[1] && src0->nb[0] != src0->nb[1] &&
                        ggml_nelements(src0) == ggml_nelements(op)) {
                        spacemit_kernels::rvv::forward_cpy_with_ترتیب_بدلول(params, op);
                    } وإلا {
                        ggml_compute_forward_cpy(params, op);
                    }
                    أرجع ریښتیا;
                }
            حالة GGML_OP_REPEAT:
                {
                    ثابت منطقي لیکې_equal         = ggml_nrows(op->src[0]) == ggml_nrows(op);
                    ثابت منطقي خپرونه_or_equal = op->src[0]->ne[0] == 1 || op->src[0]->ne[0] == op->ne[0];

                    إذا (لیکې_equal && خپرونه_or_equal) {
                        بدل (op->src[0]->type) {
                            حالة GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_repeat_nrows<int32_t>(params, op);
                                أرجع ریښتیا;
                            حالة GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_repeat_nrows<int16_t>(params, op);
                                أرجع ریښتیا;
                            افتراضي:
                                اكسر;
                        }
                    }

                    إذا (op->src[0]->ne[1] == 1 && op->src[0]->ne[0] == op->ne[0]) {
                        بدل (op->src[0]->type) {
                            حالة GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_repeat_dim1<int32_t>(params, op);
                                أرجع ریښتیا;
                            حالة GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_repeat_dim1<int16_t>(params, op);
                                أرجع ریښتیا;
                            افتراضي:
                                اكسر;
                        }
                    }

                    ggml_compute_forward_repeat(params, op);
                }
                أرجع ریښتیا;
            حالة GGML_OP_مجموعه_لیکې:
                {
                    إذا (op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32) {
                        spacemit_kernels::rvv::forward_مجموعه_لیکې<عائم>(params, op);
                    } وإلا {
                        ggml_compute_forward_مجموعه_لیکې(params, op);
                    }
                }
                أرجع ریښتیا;
            حالة GGML_OP_GET_لیکې:
                {
                    إذا (op->src[0]->type == op->type) {
                        بدل (op->src[0]->type) {
                            حالة GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_get_لیکې<int32_t>(params, op);
                                أرجع ریښتیا;
                            حالة GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_get_لیکې<int16_t>(params, op);
                                أرجع ریښتیا;
                            افتراضي:
                                اكسر;
                        }
                    }

                    ggml_compute_forward_get_لیکې(params, op);
                }
                أرجع ریښتیا;
            حالة GGML_OP_CONCAT:
                {
                    ثابت int32_t dim = ggml_get_op_params_i32(op, 0);
                    إذا (dim == 0 && op->type == op->src[0]->type) {
                        بدل (op->src[0]->type) {
                            حالة GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_concat<int32_t>(params, op);
                                أرجع ریښتیا;
                            حالة GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_concat<int16_t>(params, op);
                                أرجع ریښتیا;
                            افتراضي:
                                اكسر;
                        }
                    }

                    ggml_compute_forward_concat(params, op);
                }
                أرجع ریښتیا;
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
            ggml_threadpool_chunk_سېټ(params->threadpool, nth);
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
                    ggml::cpu::riscv64_spacemit::tls_context.tcm_بفر_اندازه);
            } وإلا {
                spacemit_kernels::rvv::forward_flash_attn_ext_f16_one_chunk_vlen1024_vf16(
                    params, dst, ir0, ir1, ggml::cpu::riscv64_spacemit::tls_context.tcm_بفر,
                    ggml::cpu::riscv64_spacemit::tls_context.tcm_بفر_اندازه);
            }

            current_chunk = ggml_threadpool_chunk_جمع_کړه(params->threadpool, 1);
        }
    }

    صحيح repack(ggml_tensor * t, ثابت فراغ * معلومات, size_t معلومات_اندازه) override {
        حافظه_نقل(t->معلومات, معلومات, معلومات_اندازه);
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
#د_که_پای
            }
            اكسر;
        حالة GGML_TYPE_Q3_K:
            {
#إذا defined(RISCV64_SPACEMIT_IME2)
                إذا (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    أرجع &ggml::cpu::riscv64_spacemit::q3_k_32x256_q8_0;
                }
#د_که_پای
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
#د_که_پای

#إذا defined(RISCV64_SPACEMIT_IME1)
                إذا (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    أرجع &ggml::cpu::riscv64_spacemit::q4_0_16x32_q8_0;
                }
#د_که_پای
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
#د_که_پای

#إذا defined(RISCV64_SPACEMIT_IME1)
                إذا (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    أرجع &ggml::cpu::riscv64_spacemit::q4_1_16x32_q8_0;
                }
#د_که_پای
            }
            اكسر;
        حالة GGML_TYPE_Q4_K:
            {
#إذا defined(RISCV64_SPACEMIT_IME2)
                إذا (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    أرجع &ggml::cpu::riscv64_spacemit::q4_k_32x32_q8_0;
                }
#د_که_پای

#إذا defined(RISCV64_SPACEMIT_IME1)
                إذا (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    أرجع &ggml::cpu::riscv64_spacemit::q4_k_16x32_q8_0;
                }
#د_که_پای
            }
            اكسر;
        حالة GGML_TYPE_Q6_K:
            {
#إذا defined(RISCV64_SPACEMIT_IME2)
                إذا ((ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    أرجع &ggml::cpu::riscv64_spacemit::q6_k_32x32_q8_0;
                }
#د_که_پای
            }
            اكسر;
        حالة GGML_TYPE_Q8_0:
            {
#إذا defined(RISCV64_SPACEMIT_IME2)
                إذا ((ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    أرجع &ggml::cpu::riscv64_spacemit::q8_0_32x32_q8_0;
                }
#د_که_پای

#إذا defined(RISCV64_SPACEMIT_IME1)
                إذا (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    أرجع &ggml::cpu::riscv64_spacemit::q8_0_16x32_q8_0;
                }
#د_که_پای
            }
            اكسر;
        حالة GGML_TYPE_MXFP4:
            {
#إذا defined(RISCV64_SPACEMIT_IME2)
                // TODO
                // if (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                //     return &ggml::cpu::riscv64_spacemit::mxfp4_32x32_q8_0;
                // }
#د_که_پای
            }
            اكسر;
        حالة GGML_TYPE_Q5_K:
            {
#إذا defined(RISCV64_SPACEMIT_IME2)
                إذا (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    أرجع &ggml::cpu::riscv64_spacemit::q5_k_32x32_q8_0;
                }
#د_که_پای
            }
            اكسر;
        حالة GGML_TYPE_Q5_1:
            {
#إذا defined(RISCV64_SPACEMIT_IME2)
                إذا (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    أرجع &ggml::cpu::riscv64_spacemit::q5_1_32x32_q8_0;
                }
#د_که_پای
            }
            اكسر;
        حالة GGML_TYPE_Q5_0:
            {
#إذا defined(RISCV64_SPACEMIT_IME2)
                إذا (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    أرجع &ggml::cpu::riscv64_spacemit::q5_0_32x32_q8_0;
                }
#د_که_پای
            }
            اكسر;
        افتراضي:
            اكسر;
    }

    أرجع تش_اشاره;
}

ساكن تعداد ggml_حالت ggml_backend_riscv64_spacemit_بفر_init_tensor(ggml_backend_بفر_t بفر,
                                                                         ggml_tensor *         tensor) {
    tensor->extra =
        (فراغ *) const_cast<ggml::cpu::tensor_traits *>(ggml_riscv64_spacemit_get_optimal_repack_type(tensor));

    GGML_ناکارول_شوی(بفر);

    أرجع GGML_حالت_SUCCESS;
}

ساكن فراغ ggml_backend_riscv64_spacemit_بفر_free_بفر(ggml_backend_بفر_t بفر) {
    GGML_ASSERT(بفر);

    فراغ * بنسټ = بفر->context;
    إذا (بنسټ == تش_اشاره) {
        أرجع;
    }

    ggml::cpu::riscv64_spacemit::spine_mem_pool_free(بنسټ);
}

ساكن فراغ * ggml_backend_riscv64_spacemit_بفر_get_بنسټ(ggml_backend_بفر_t بفر) {
    GGML_ASSERT(بفر);

    فراغ * بنسټ = بفر->context;
    GGML_ASSERT(بنسټ != تش_اشاره);
    أرجع بنسټ;
}

ساكن فراغ ggml_backend_riscv64_spacemit_بفر_memset_tensor(ggml_backend_بفر_t بفر,
                                                               ggml_tensor *         tensor,
                                                               uint8_t               value,
                                                               size_t                دننه_واټن,
                                                               size_t                size) {
    GGML_ASSERT(tensor);
    memset((حرف *) tensor->معلومات + دننه_واټن, value, size);

    GGML_ناکارول_شوی(بفر);
}

ساكن فراغ ggml_backend_riscv64_spacemit_بفر_clear(ggml_backend_بفر_t بفر, uint8_t value) {
    GGML_ASSERT(بفر);

    فراغ * بنسټ = بفر->context;
    GGML_ASSERT(بنسټ != تش_اشاره);
    memset(بنسټ, value, بفر->size);
}

ساكن فراغ ggml_backend_riscv64_spacemit_بفر_سېټ_tensor(ggml_backend_بفر_t بفر,
                                                            ggml_tensor *         tensor,
                                                            ثابت فراغ *          معلومات,
                                                            size_t                دننه_واټن,
                                                            size_t                size) {
    GGML_ASSERT(دننه_واټن == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    خپلکاری tensor_traits = (ggml::cpu::riscv64_spacemit::tensor_traits_بنسټ *) tensor->extra;
    إذا (tensor_traits) {
        خپلکاری OK = tensor_traits->repack(tensor, معلومات, size);
        GGML_ASSERT(OK == 0);
    }

    GGML_ناکارول_شوی(بفر);
}

ساكن ثابت ggml_backend_بفر_i ggml_backend_riscv64_spacemit_بفر_i = {
    /* .free_buffer     = */ ggml_backend_riscv64_spacemit_بفر_free_بفر,
    /* .get_base        = */ ggml_backend_riscv64_spacemit_بفر_get_بنسټ,
    /* .init_tensor     = */ ggml_backend_riscv64_spacemit_بفر_init_tensor,
    /* .memset_tensor   = */ ggml_backend_riscv64_spacemit_بفر_memset_tensor,
    /* .set_tensor      = */ ggml_backend_riscv64_spacemit_بفر_سېټ_tensor,
    /* .get_tensor      = */ تش_اشاره,
    /* .set_tensor_2d   = */ تش_اشاره,
    /* .get_tensor_2d   = */ تش_اشاره,
    /* .cpy_tensor      = */ تش_اشاره,
    /* .clear           = */ ggml_backend_riscv64_spacemit_بفر_clear,
    /* .reset           = */ تش_اشاره,
};

ساكن ثابت حرف * ggml_backend_cpu_riscv64_spacemit_بفر_type_get_name(ggml_backend_بفر_type_t buft) {
    أرجع "CPU_RISCV64_SPACEMIT";

    GGML_ناکارول_شوی(buft);
}

ساكن ggml_backend_بفر_t ggml_backend_cpu_riscv64_spacemit_بفر_type_alloc_بفر(ggml_backend_بفر_type_t buft,
                                                                                        size_t size) {
    فراغ * بنسټ = ggml::cpu::riscv64_spacemit::spine_mem_pool_alloc(size, 64);
    إذا (بنسټ == تش_اشاره) {
        أرجع تش_اشاره;
    }

    أرجع ggml_backend_بفر_init(buft, ggml_backend_riscv64_spacemit_بفر_i, بنسټ, size);
}

ساكن size_t ggml_backend_cpu_riscv64_spacemit_بفر_type_get_alignment(ggml_backend_بفر_type_t buft) {
    أرجع 64;

    GGML_ناکارول_شوی(buft);
}

ساكن size_t ggml_backend_cpu_riscv64_spacemit_nbytes(ggml_backend_بفر_type_t buft, ثابت ggml_tensor * tensor) {
    لكل (صحيح i = 0; i < GGML_MAX_DIMS; ++i) {
        إذا (tensor->ne[i] <= 0) {
            أرجع 0;
        }
    }

    GGML_ناکارول_شوی(buft);

    ثابت خپلکاری plain_nbytes = [&]() {
        size_t ټول = ggml_type_اندازه(tensor->type);
        لكل (صحيح i = 0; i < GGML_MAX_DIMS; ++i) {
            ټول += (tensor->ne[i] - 1) * tensor->nb[i];
        }
        أرجع ټول;
    };

    ثابت size_t blck_اندازه = ggml_blck_اندازه(tensor->type);
    إذا (blck_اندازه == 1) {
        أرجع plain_nbytes();
    }

    ثابت size_t لیکه_nbytes = tensor->ne[0] * tensor->nb[0] / blck_اندازه;

    ثابت خپلکاری جمع_کړه_strided_nbytes = [&](size_t ټول, size_t src_بلاک_اندازه, size_t dst_بلاک_اندازه) {
        لكل (صحيح i = 1; i < GGML_MAX_DIMS; ++i) {
            ټول += (tensor->ne[i] - 1) * (tensor->nb[i] / src_بلاک_اندازه) * dst_بلاک_اندازه;
        }
        أرجع ټول;
    };

    ثابت خپلکاری remap_بلاک_nbytes = [&](size_t src_بلاک_اندازه, size_t dst_بلاک_اندازه, int64_t padded_لیکې = 0) {
        GGML_ASSERT(لیکه_nbytes % src_بلاک_اندازه == 0);

        size_t ټول =
            جمع_کړه_strided_nbytes((لیکه_nbytes / src_بلاک_اندازه) * dst_بلاک_اندازه, src_بلاک_اندازه, dst_بلاک_اندازه);

        إذا (padded_لیکې > 0 && tensor->ne[1] % padded_لیکې != 0) {
            ټول += (padded_لیکې - tensor->ne[1] % padded_لیکې) * (tensor->nb[1] / src_بلاک_اندازه) * dst_بلاک_اندازه;
        }

        أرجع ټول;
    };

    size_t nbytes = لیکه_nbytes;
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
            nbytes = جمع_کړه_strided_nbytes(لیکه_nbytes, 1, 1);
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
                        أرجع ریښتیا;
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
                        أرجع ریښتیا;
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
            حالة GGML_OP_جمع_کړه:
            حالة GGML_OP_SUB:
            حالة GGML_OP_MUL:
            حالة GGML_OP_DIV:
            حالة GGML_OP_FLASH_ATTN_EXT:
            حالة GGML_OP_CONT:
            حالة GGML_OP_CPY:
            حالة GGML_OP_REPEAT:
            حالة GGML_OP_مجموعه_لیکې:
            حالة GGML_OP_GET_لیکې:
            حالة GGML_OP_CONCAT:
                // case GGML_OP_GATED_DELTA_NET:
                أرجع (ggml::cpu::tensor_traits *) (&ggml::cpu::riscv64_spacemit::rvv_impl);
            افتراضي:
                // GGML_ABORT("fatal error");
                اكسر;
        }

        أرجع تش_اشاره;
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
         /* .get_max_size       = */ تش_اشاره,
         /* .get_alloc_size     = */ ggml_backend_cpu_riscv64_spacemit_nbytes,
         /* .get_alloc_size_n   = */ عدم,
         /* .is_host            = */ تش_اشاره,
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
    صحيح  fd, بایټونه;
    حرف str[32];

    fd = open("/proc/set_ai_thread", O_WRONLY);
    إذا (fd < 0) {
        GGML_LOG_ERROR("try open /proc/set_ai_thread failed\n");
        أرجع -1;
    }

    snprintf(str, 16, "%d", 0);
    بایټونه = اكتب(fd, str, مزي_اوږدوالی(str));
    إذا (بایټونه < 0) {
        GGML_LOG_ERROR("try write /proc/set_ai_thread failed\n");
        close(fd);
        أرجع -1;
    }

    close(fd);
    أرجع 0;
}

فراغ ggml_backend_cpu_riscv64_spacemit_سېټ_numa_thread_affinity(صحيح thread_n) {
    صحيح cpu_id = sched_getcpu();
    إذا (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2 &&
        !((1 << cpu_id) & ggml::cpu::riscv64_spacemit::global_spine_env_info.cpu_ماسک)) {
        GGML_PRINT_DEBUG("bind_ai_thread for thread %d, pid %d\n", thread_n, getpid());
        bind_ai_thread();
    }

    إذا (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_tcm &&
        ggml::cpu::riscv64_spacemit::tls_context.cpu_id == -1) {
        CPU_ZERO(&(ggml::cpu::riscv64_spacemit::tls_context.cpuset));
        pthread_t    main_thread     = pthread_self();
        ثابت خپلکاری & perfer_core_ids = ggml::cpu::riscv64_spacemit::global_spine_env_info.perfer_core_ids;
        إذا (thread_n < 0 || static_cast<size_t>(thread_n) >= perfer_core_ids.size()) {
            GGML_ABORT("thread_n %d exceeds perfer_core_ids size %zu\n", thread_n, perfer_core_ids.size());
        }
        خپلکاری perfer_cpu_id = perfer_core_ids[static_cast<size_t>(thread_n)];
        CPU_SET(perfer_cpu_id, &(ggml::cpu::riscv64_spacemit::tls_context.cpuset));
        صحيح s =
            pthread_setaffinity_np(main_thread, حجم(cpu_set_t), &(ggml::cpu::riscv64_spacemit::tls_context.cpuset));
        إذا (s != 0) {
            GGML_ABORT("set thread affinity error for thread_n %d, cpu_id %d\n", thread_n, perfer_cpu_id);
        }

        صحيح ai_cpu_id = perfer_cpu_id - ggml::cpu::riscv64_spacemit::global_spine_env_info.aicpu_id_دننه_واټن;
        ggml::cpu::riscv64_spacemit::tls_context.cpu_id = ai_cpu_id;
        ggml::cpu::riscv64_spacemit::tls_context.tcm_بفر =
            ggml::cpu::riscv64_spacemit::spine_mem_pool_tcm_mem_get(ai_cpu_id);
        ggml::cpu::riscv64_spacemit::tls_context.tcm_بفر_اندازه =
            ggml::cpu::riscv64_spacemit::global_spine_env_info.tcm_blk_اندازه;
    }

    إذا (ggml::cpu::riscv64_spacemit::tls_context.tcm_بفر != تش_اشاره) {
        فراغ * rt =
            ggml::cpu::riscv64_spacemit::spine_mem_pool_tcm_mem_wait(ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        إذا (rt == تش_اشاره) {
            GGML_ABORT("wait tcm buffer failed for cpu_id: %d", ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        }
    }
}

فراغ ggml_backend_cpu_riscv64_spacemit_clear_numa_thread_affinity_threaded(صحيح thread_n) {
    إذا (ggml::cpu::riscv64_spacemit::tls_context.tcm_بفر != تش_اشاره) {
        خپلکاری rt = ggml::cpu::riscv64_spacemit::spine_mem_pool_tcm_mem_release(
            ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        إذا (rt != 0) {
            GGML_ABORT("release tcm buffer failed for cpu_id: %d", ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        }
    }
}
}
