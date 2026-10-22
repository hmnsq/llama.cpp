#تعریف_کرو GGML_COMMON_IMPL_CPP
#تعریف_کرو GGML_COMMON_DECL_CPP

#شامل "ime.h"

#شامل "binary-ops.h"
#شامل "common.h"
#شامل "ggml-backend-impl.h"
#شامل "ggml-common.h"
#شامل "ggml-cpu.h"
#شامل "ime_env.h"
#شامل "ime_kernels.h"
#شامل "ops.h"
#شامل "repack.h"
#شامل "rvv_kernels.h"
#شامل "spine_mem_pool.h"
#شامل "traits.h"
#شامل "vec.h"

#شامل <fcntl.h>
#شامل <sys/mman.h>
#شامل <unistd.h>

#شامل <algorithm>
#شامل <atomic>
#شامل <cassert>
#شامل <cerrno>
#شامل <cmath>
#شامل <cstdio>  // for GGML_ASSERT
#شامل <stdexcept>
#شامل <thread>
// clang-format off
#اگر defined(__riscv)

#اگر !defined(__riscv_v) || !defined(__riscv_v_intrinsic)
#غلطی "riscv v extension or v_intrinsic not enabled"
#ورنہ
#شامل <riscv_vector.h>
#اگر_ختم

#اگر !defined(__riscv_zfh) || !defined(__riscv_zvfh)
#غلطی "riscv zfh extension not enabled, GGML_RV_ZFH and GGML_RV_ZVFH must be defined to 1"
#اگر_ختم

#اگر !defined(__riscv_zba)
#غلطی "riscv zba extension not enabled, GGML_RV_ZBA must be defined to 1"
#اگر_ختم

#اگر defined(RISCV64_SPACEMIT_IME1) || defined(RISCV64_SPACEMIT_IME2)
#ورنہ
#غلطی "RISCV64_SPACEMIT_IME1 or RISCV64_SPACEMIT_IME2 not defined"
#اگر_ختم

#ورنہ

#غلطی "riscv not enabled in this build"

#اگر_ختم

#اگر defined(__GNUC__)
#ہدایت GCC diagnostic ignored "-Woverlength-strings"
#ہدایت GCC diagnostic ignored "-Wcast-qual"
#ہدایت GCC diagnostic ignored "-Wunused-parameter"
#اگر_ختم

// clang-format on

خارجی "C" {
خارجی خالی ggml_threadpool_chunk_سیٹ(ساخت ggml_threadpool * tp, صحیح value);
خارجی صحیح  ggml_threadpool_chunk_جوڑو(ساخت ggml_threadpool * tp, صحیح value);
}

نامخلا ggml::cpu::riscv64_spacemit {

ساخت TLSContext {
    صحیح       cpu_id{ -1 };
    cpu_set_t cpuset;
    خالی *    tcm_بفر{ خالی_اشاریہ };
    size_t    tcm_بفر_سائز{ 0 };
};

thread_local TLSContext tls_context;

سانچہ <قسم_نام BLOC_TYPE, int64_t INTER_سائز, int64_t NB_ستون_تعداد> constexpr size_t get_repacked_بلاک_type_سائز() {
    اگر constexpr (std::is_same_v<BLOC_TYPE, بلاک_q6_K> || std::is_same_v<BLOC_TYPE, بلاک_q8_0>) {
        واپس حجم(بلاک_q8_0);
    } ورنہ اگر constexpr (std::is_same_v<BLOC_TYPE, بلاک_q4_0>) {
        واپس حجم(بلاک_q4_0) * INTER_سائز / QK4_0;
    } ورنہ اگر constexpr (std::is_same_v<BLOC_TYPE, بلاک_q4_1> || std::is_same_v<BLOC_TYPE, بلاک_q4_K>) {
        واپس (حجم(بلاک_q4_0) + حجم(uint8_t)) * INTER_سائز / QK4_1;
    } ورنہ اگر constexpr (std::is_same_v<BLOC_TYPE, بلاک_q2_K>) {
        واپس حجم(spacemit_kernels::nrow_بلاک_q2_k<1>);
    } ورنہ اگر constexpr (std::is_same_v<BLOC_TYPE, بلاک_q3_K>) {
        واپس حجم(spacemit_kernels::nrow_بلاک_q3_k<1>);
    } ورنہ اگر constexpr (std::is_same_v<BLOC_TYPE, بلاک_mxfp4>) {
        واپس حجم(spacemit_kernels::nrow_بلاک_mxfp4<1>);
    } ورنہ اگر constexpr (std::is_same_v<BLOC_TYPE, بلاک_q5_1> || std::is_same_v<BLOC_TYPE, بلاک_q5_K>) {
        واپس حجم(spacemit_kernels::nrow_بلاک_q5_1<1>);
    } ورنہ اگر constexpr (std::is_same_v<BLOC_TYPE, بلاک_q5_0>) {
        واپس حجم(spacemit_kernels::nrow_بلاک_q5_0<1>);
    } ورنہ {
        أكد(جھوٹ);
        واپس 0;
    }
}

سانچہ <قسم_نام BLOC_TYPE> constexpr منطقی بلاک_type_has_zp() {
    اگر constexpr (std::is_same_v<BLOC_TYPE, بلاک_q6_K> || std::is_same_v<BLOC_TYPE, بلاک_q8_0> ||
                  std::is_same_v<BLOC_TYPE, بلاک_q3_K> || std::is_same_v<BLOC_TYPE, بلاک_q4_0> ||
                  std::is_same_v<BLOC_TYPE, بلاک_mxfp4> || std::is_same_v<BLOC_TYPE, بلاک_q5_0>) {
        واپس جھوٹ;
    } ورنہ اگر constexpr (std::is_same_v<BLOC_TYPE, بلاک_q4_1> || std::is_same_v<BLOC_TYPE, بلاک_q4_K> ||
                         std::is_same_v<BLOC_TYPE, بلاک_q2_K> || std::is_same_v<BLOC_TYPE, بلاک_q5_1> ||
                         std::is_same_v<BLOC_TYPE, بلاک_q5_K>) {
        واپس سچ;
    } ورنہ {
        أكد(جھوٹ);
        واپس جھوٹ;
    }
}

جماعت tensor_traits_بنیاد : عوامی ggml::cpu::tensor_traits {
  عوامی:
    مجازی صحیح repack(ggml_tensor * t, ثابت خالی * ڈیٹا, size_t ڈیٹا_سائز) = 0;
};

سانچہ <قسم_نام BLOC_TYPE, int64_t INTER_سائز, int64_t NB_ستون_تعداد> جماعت tensor_traits : عوامی tensor_traits_بنیاد {
    منطقی work_سائز(صحیح /* n_threads */, ثابت ggml_tensor * op, size_t & size) override {
        انتخاب (op->op) {
            حالت GGML_OP_MUL_MAT:
                {
                    int64_t src1_nelements = ggml_nelements(op->src[1]);

                    اگر constexpr (std::is_same_v<BLOC_TYPE, بلاک_q2_K> || std::is_same_v<BLOC_TYPE, بلاک_q3_K>) {
                        size =
                            spacemit_kernels::div_باری_اوپر(src1_nelements, QK_K) * spacemit_kernels::q8k_blk_سائز(QK_K);
                    } ورنہ اگر constexpr (INTER_سائز == QK4_0) {
                        size = spacemit_kernels::div_باری_اوپر(src1_nelements, QK4_0) *
                               spacemit_kernels::q8_blk_سائز(QK4_0, سچ);
                    } ورنہ اگر constexpr (INTER_سائز == 256) {
                        size = spacemit_kernels::div_باری_اوپر(src1_nelements, 256) *
                               spacemit_kernels::q8_hp_blk_سائز(256, سچ, سچ);
                    } ورنہ {
                        GGML_ABORT("unsupported block type");
                    }

                    size = GGML_PAD(size, حجم(int64_t));

                    واپس سچ;
                }
            حالت GGML_OP_MUL_MAT_ID:
                {
                    int64_t src1_nelements = ggml_nelements(op->src[1]);

                    اگر constexpr (std::is_same_v<BLOC_TYPE, بلاک_q2_K> || std::is_same_v<BLOC_TYPE, بلاک_q3_K>) {
                        size =
                            spacemit_kernels::div_باری_اوپر(src1_nelements, QK_K) * spacemit_kernels::q8k_blk_سائز(QK_K);
                    } ورنہ اگر constexpr (INTER_سائز == QK4_0) {
                        size = spacemit_kernels::div_باری_اوپر(src1_nelements, QK4_0) *
                               spacemit_kernels::q8_blk_سائز(QK4_0, سچ);
                    } ورنہ اگر constexpr (INTER_سائز == 256) {
                        size = spacemit_kernels::div_باری_اوپر(src1_nelements, 256) *
                               spacemit_kernels::q8_hp_blk_سائز(256, سچ, سچ);
                    } ورنہ {
                        GGML_ABORT("unsupported block type");
                    }

                    size = GGML_PAD(size, حجم(int64_t));

                    ثابت int64_t ne02 = op->src[0]->ne[2];  // n_as, n_expert
                    ثابت int64_t ne12 = op->src[1]->ne[2];  // n_tokens

                    ثابت size_t sizeof_mmid_صف_mapping = حجم(int64_t);
                    size += sizeof_mmid_صف_mapping * ne02 * (ne12 + 1) + (ne02 + 1) * حجم(int64_t);

                    size = GGML_PAD(size, حجم(int64_t));

                    واپس سچ;
                }
            طےشدہ:
                // GGML_ABORT("fatal error");
                توڑو;
        }
        واپس جھوٹ;
    }

    منطقی compute_forward(ggml_compute_params * params, ggml_tensor * op) override {
        انتخاب (op->op) {
            حالت GGML_OP_MUL_MAT:
                انتخاب (op->src[0]->type) {
                    حالت GGML_TYPE_Q2_K:
                    حالت GGML_TYPE_Q3_K:
                    حالت GGML_TYPE_Q4_0:
                    حالت GGML_TYPE_Q4_1:
                    حالت GGML_TYPE_Q4_K:
                    حالت GGML_TYPE_Q6_K:
                    حالت GGML_TYPE_Q8_0:
                    حالت GGML_TYPE_Q5_0:
                    حالت GGML_TYPE_Q5_1:
                    حالت GGML_TYPE_Q5_K:
                        //case GGML_TYPE_MXFP4:
                        forward_mul_mat(params, op);
                        واپس سچ;
                    طےشدہ:
                        // GGML_ABORT("fatal error: unsupported type for src0 in MUL_MAT");
                        واپس جھوٹ;
                }
                توڑو;
            حالت GGML_OP_MUL_MAT_ID:
                انتخاب (op->src[0]->type) {
                    حالت GGML_TYPE_Q2_K:
                    حالت GGML_TYPE_Q3_K:
                    حالت GGML_TYPE_Q4_0:
                    حالت GGML_TYPE_Q4_1:
                    حالت GGML_TYPE_Q4_K:
                    حالت GGML_TYPE_Q6_K:
                    حالت GGML_TYPE_Q8_0:
                    حالت GGML_TYPE_Q5_0:
                    حالت GGML_TYPE_Q5_1:
                    حالت GGML_TYPE_Q5_K:
                        //case GGML_TYPE_MXFP4:
                        forward_mul_mat_id(params, op);
                        واپس سچ;
                    طےشدہ:
                        // GGML_ABORT("fatal error: unsupported type for src0 in MUL_MAT_ID");
                        واپس جھوٹ;
                }
                توڑو;
            طےشدہ:
                // GGML_ABORT("fatal error");
                توڑو;
        }
        واپس جھوٹ;
    }

    خالی forward_mul_mat(ggml_compute_params * params, ggml_tensor * op) {
        constexpr size_t a_blk_len = INTER_سائز;
        constexpr size_t b_blk_len = INTER_سائز;

        ثابت ggml_tensor * src0 = op->src[0];
        ثابت ggml_tensor * src1 = op->src[1];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_ثنائی_OP_LOCALS

        صحیح ith = params->ith;
        صحیح nth = params->nth;

        [[maybe_unused]] ثابت شمار ggml_type type = src0->type;

        خالی *        w_ڈیٹا  = (خالی *) src0->ڈیٹا;
        ثابت اعشاری * feature = (ثابت اعشاری *) src1->ڈیٹا;
        اعشاری *       output  = (اعشاری *) dst->ڈیٹا;

        ثابت int64_t gemm_m = ne11 * ne12 * ne13;
        ثابت int64_t gemm_k = ne10;
        ثابت int64_t gemm_n = ne01;

        spacemit_kernels::quantize_a_صف_def       quantize_a_صف_i8;
        spacemit_kernels::quantize_a_صف_def       quantize_a_4row_i8;
        spacemit_kernels::gemm_kernel_quantize_def gemm_kernel;
        منطقی                                       سیٹ_kernel_impl = جھوٹ;

        int64_t بلاک_stride_a = spacemit_kernels::q8_blk_سائز(a_blk_len);

#اگر defined(RISCV64_SPACEMIT_IME2)
        اگر (!سیٹ_kernel_impl && (global_spine_env_info.use_ime2)) {
            quantize_a_صف_i8  = spacemit_kernels::rvv::quantize_a_صف_i8;
            quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8;
            بلاک_stride_a     = spacemit_kernels::q8_blk_سائز(a_blk_len, سچ);

            اگر constexpr (std::is_same_v<BLOC_TYPE, بلاک_q6_K> || std::is_same_v<BLOC_TYPE, بلاک_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i8;
                سیٹ_kernel_impl = سچ;
            } ورنہ اگر constexpr (std::is_same_v<BLOC_TYPE, بلاک_q4_0> || std::is_same_v<BLOC_TYPE, بلاک_q4_1> ||
                                 std::is_same_v<BLOC_TYPE, بلاک_q4_K>) {
                اگر constexpr (INTER_سائز == 256) {
                    gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i4_hp;
                    quantize_a_صف_i8  = spacemit_kernels::rvv::quantize_a_صف_i8_hp;
                    quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8_hp;
                    بلاک_stride_a     = spacemit_kernels::q8_hp_blk_سائز(a_blk_len, سچ, سچ);
                    سیٹ_kernel_impl    = سچ;
                } ورنہ {
                    gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i4;
                    quantize_a_صف_i8  = spacemit_kernels::rvv::quantize_a_صف_i8;
                    quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8;
                    بلاک_stride_a     = spacemit_kernels::q8_blk_سائز(a_blk_len, سچ);
                    سیٹ_kernel_impl    = سچ;
                }
            } ورنہ اگر constexpr (std::is_same_v<BLOC_TYPE, بلاک_q2_K>) {
                quantize_a_صف_i8  = spacemit_kernels::rvv::quantize_a_صف_i8k;
                quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8k;
                بلاک_stride_a     = spacemit_kernels::q8k_blk_سائز(a_blk_len);

                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i2k;
                سیٹ_kernel_impl = سچ;
            } ورنہ اگر constexpr (std::is_same_v<BLOC_TYPE, بلاک_q3_K>) {
                quantize_a_صف_i8  = spacemit_kernels::rvv::quantize_a_صف_i8k;
                quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8k;
                بلاک_stride_a     = spacemit_kernels::q8k_blk_سائز(a_blk_len);

                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i3k;
                سیٹ_kernel_impl = سچ;
            } ورنہ اگر constexpr (std::is_same_v<BLOC_TYPE, بلاک_mxfp4>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8mxfp4;
                سیٹ_kernel_impl = سچ;
            } ورنہ اگر constexpr (std::is_same_v<BLOC_TYPE, بلاک_q5_1> || std::is_same_v<BLOC_TYPE, بلاک_q5_K> ||
                                 std::is_same_v<BLOC_TYPE, بلاک_q5_0>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i5;
                سیٹ_kernel_impl = سچ;
            }
        }
#اگر_ختم

#اگر defined(RISCV64_SPACEMIT_IME1)
        اگر (!سیٹ_kernel_impl && (global_spine_env_info.use_ime1)) {
            quantize_a_صف_i8  = spacemit_kernels::ime1::quantize_a_صف_i8;
            quantize_a_4row_i8 = spacemit_kernels::ime1::quantize_a_4row_i8;

            اگر constexpr (std::is_same_v<BLOC_TYPE, بلاک_q4_0> || std::is_same_v<BLOC_TYPE, بلاک_q4_1> ||
                          std::is_same_v<BLOC_TYPE, بلاک_q4_K>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i4;
                سیٹ_kernel_impl = سچ;
            } ورنہ اگر constexpr (std::is_same_v<BLOC_TYPE, بلاک_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i8;
                سیٹ_kernel_impl = سچ;
            }
        }
#اگر_ختم
        اگر (!سیٹ_kernel_impl) {
            GGML_ABORT("no kernel implementation found for the block type");
        }

        ثابت int64_t a_k_blks = spacemit_kernels::div_باری_اوپر(gemm_k, a_blk_len);
        ثابت int64_t b_k_blks = spacemit_kernels::div_باری_اوپر(gemm_k, b_blk_len);

        ثابت int64_t صف_stride_a        = a_k_blks * بلاک_stride_a;
        ثابت int64_t gemm_workspace_سائز = GGML_PAD(gemm_m * صف_stride_a, alignof(int64_t));

        اگر (ith == 0 && params->wsize < gemm_workspace_سائز) {
            GGML_ABORT("wsize less than gemm_workspace_size");
        }

        uintptr_t ws_ptr = reinterpret_cast<uintptr_t>(params->wdata);

        خالی *        tcm_بفر      = ggml::cpu::riscv64_spacemit::tls_context.tcm_بفر;
        ثابت int64_t tcm_بفر_سائز = ggml::cpu::riscv64_spacemit::tls_context.tcm_بفر_سائز;

        خودکار * quant_a_بفر = reinterpret_cast<uint8_t *>(ws_ptr);

        constexpr int64_t صف_align = 4;
        ثابت int64_t     صف_blks  = spacemit_kernels::div_باری_اوپر(gemm_m, صف_align);

        ثابت int64_t صف_stride_b      = b_k_blks * get_repacked_بلاک_type_سائز<BLOC_TYPE, INTER_سائز, NB_ستون_تعداد>();
        ثابت int64_t فی_mb_صفیں_wsize = صف_align * صف_stride_a;
        ثابت int64_t فی_nb_ستون_تعداد_wsize = NB_ستون_تعداد * صف_stride_b;

        ثابت int64_t barrier_idx = static_cast<int64_t>(ith / 2);

        GGML_ASSERT(global_spine_env_info.init_barrier != خالی_اشاریہ);
        GGML_ASSERT(barrier_idx < spine_init_barrier_گنتی);
        spine_barrier_t * cur_barrier = &global_spine_env_info.init_barrier[barrier_idx];

        اگر (gemm_m == 1) {
            صحیح task_فی_thread = spacemit_kernels::div_باری_اوپر(a_k_blks, nth);
            صحیح a_blk_آغاز     = ith * task_فی_thread;
            صحیح a_blk_انتہا       = std::min(a_blk_آغاز + task_فی_thread, (صحیح) a_k_blks);
            اگر (a_blk_آغاز < a_blk_انتہا) {
                quantize_a_صف_i8(a_blk_len, feature + a_blk_آغاز * a_blk_len, (a_blk_انتہا - a_blk_آغاز) * a_blk_len,
                                  quant_a_بفر + a_blk_آغاز * بلاک_stride_a);
            }
        } ورنہ {
            صحیح task_فی_thread = spacemit_kernels::div_باری_اوپر(صف_blks, nth);
            صحیح m_صف_blk_آغاز = ith * task_فی_thread;
            صحیح m_صف_blk_انتہا   = std::min(m_صف_blk_آغاز + task_فی_thread, (صحیح) صف_blks);
            برائے (صحیح m_صف_blk = m_صف_blk_آغاز; m_صف_blk < m_صف_blk_انتہا; m_صف_blk++) {
                صحیح m_idx             = m_صف_blk * صف_align;
                صحیح صفیں_tobe_handled = (gemm_m - m_idx) > صف_align ? صف_align : (gemm_m - m_idx);

                اگر (صفیں_tobe_handled == صف_align && quantize_a_4row_i8 != خالی_اشاریہ) {
                    ثابت اعشاری * a_صف_ptr       = feature + m_idx * gemm_k;
                    خودکار *        quant_a_صف_ptr = quant_a_بفر + m_idx * صف_stride_a;
                    quantize_a_4row_i8(a_blk_len, a_صف_ptr, gemm_k, quant_a_صف_ptr);
                } ورنہ {
                    جب (صفیں_tobe_handled) {
                        ثابت اعشاری * a_صف_ptr       = feature + m_idx * gemm_k;
                        خودکار *        quant_a_صف_ptr = quant_a_بفر + m_idx * صف_stride_a;
                        quantize_a_صف_i8(a_blk_len, a_صف_ptr, gemm_k, quant_a_صف_ptr);
                        صفیں_tobe_handled -= 1;
                        m_idx += 1;
                    }
                }
            }
        }

        ggml_barrier(params->threadpool);

        ثابت int64_t gemm_m_stride     = gemm_n / gemm_m > 64 ? gemm_m : 16;
        ثابت int64_t gemm_m_blocked    = spacemit_kernels::div_باری_اوپر(gemm_m, gemm_m_stride);
        ثابت int64_t max_gemm_n_stride = spacemit_kernels::div_باری_اوپر(gemm_n * gemm_m_blocked, nth);

        int64_t gemm_n_stride = gemm_n;
        اگر (max_gemm_n_stride < gemm_n) {
            gemm_n_stride =
                std::min(gemm_n_stride, spacemit_kernels::div_باری_اوپر(max_gemm_n_stride, NB_ستون_تعداد) * NB_ستون_تعداد);
        }

        اگر (gemm_n_stride == gemm_n && tcm_بفر != خالی_اشاریہ && فی_mb_صفیں_wsize <= tcm_بفر_سائز) {
            برائے (int64_t m_آغاز = ith * صف_align; m_آغاز < gemm_m; m_آغاز += صف_align * nth) {
                uint8_t * b_ستون    = reinterpret_cast<uint8_t *>(w_ڈیٹا);
                uint8_t * b_ستون_zp = بلاک_type_has_zp<BLOC_TYPE>() ? b_ستون : خالی_اشاریہ;

                int64_t m_صف_real = std::min(gemm_m - m_آغاز, صف_align);

                spacemit_kernels::rvv::memcpy1d(tcm_بفر, quant_a_بفر + m_آغاز * صف_stride_a,
                                                m_صف_real * صف_stride_a);

                int64_t n_blk_real = 0;
                برائے (int64_t ni = 0; ni < gemm_n; ni += n_blk_real, b_ستون += n_blk_real * صف_stride_b) {
                    n_blk_real = std::min(gemm_n - ni, (int64_t) NB_ستون_تعداد);

                    uint8_t * a_صف_ptr = (uint8_t *) tcm_بفر;
                    اعشاری *   c_blk     = output + m_آغاز * gemm_n + ni;

                    int32_t صفیں_بقایا = m_صف_real;

                    جب (صفیں_بقایا > 0) {
                        خودکار صفیں_handled = gemm_kernel(b_blk_len, a_صف_ptr, b_ستون, b_ستون_zp, c_blk, صفیں_بقایا,
                                                        n_blk_real, b_k_blks, gemm_n);

                        c_blk += صفیں_handled * gemm_n;
                        a_صف_ptr += صفیں_handled * صف_stride_a;

                        صفیں_بقایا -= صفیں_handled;
                    }
                }
            }
        } ورنہ اگر (tcm_بفر != خالی_اشاریہ && فی_nb_ستون_تعداد_wsize <= tcm_بفر_سائز) {
            uint8_t * a_صف = quant_a_بفر;
            uint8_t * b_ستون = reinterpret_cast<uint8_t *>(tcm_بفر);
            اگر ((gemm_workspace_سائز + فی_nb_ستون_تعداد_wsize) <= tcm_بفر_سائز) {
                a_صف = (uint8_t *) tcm_بفر;
                b_ستون = reinterpret_cast<uint8_t *>(tcm_بفر) + gemm_workspace_سائز;
            }
            uint8_t * b_ستون_zp = بلاک_type_has_zp<BLOC_TYPE>() ? b_ستون : خالی_اشاریہ;

            int64_t ni      = ith * NB_ستون_تعداد;
            int64_t nb_real = std::min(gemm_n - ni, NB_ستون_تعداد);

            اگر (ith % 2 == 0 && nb_real > 0) {
                spacemit_kernels::rvv::memcpy1d(b_ستون, reinterpret_cast<uint8_t *>(w_ڈیٹا) + ni * صف_stride_b,
                                                nb_real * صف_stride_b);
                اگر (a_صف != quant_a_بفر) {
                    spacemit_kernels::rvv::memcpy1d(a_صف, quant_a_بفر, gemm_workspace_سائز);
                }
            }

            spine_barrier_wait(cur_barrier);

            اگر (ith % 2 != 0 && nb_real > 0) {
                اگر (a_صف != quant_a_بفر) {
                    spacemit_kernels::rvv::memcpy1d(a_صف, quant_a_بفر, gemm_workspace_سائز);
                }
                spacemit_kernels::rvv::memcpy1d(b_ستون, reinterpret_cast<uint8_t *>(w_ڈیٹا) + ni * صف_stride_b,
                                                nb_real * صف_stride_b);
            }

            برائے (; ni < gemm_n; ni += NB_ستون_تعداد * nth) {
                int64_t صفیں_بقایا = gemm_m;
                اعشاری * c_blk          = output + ni;
                خودکار *  a_صف_cur      = a_صف;

                اگر (ith % 2 != 0) {
                    spine_barrier_wait(cur_barrier);
                }

                جب (صفیں_بقایا > 0) {
                    خودکار صفیں_handled = gemm_kernel(b_blk_len, a_صف_cur, b_ستون, b_ستون_zp, c_blk, صفیں_بقایا,
                                                    nb_real, b_k_blks, gemm_n);

                    c_blk += صفیں_handled * gemm_n;
                    a_صف_cur += صفیں_handled * صف_stride_a;

                    صفیں_بقایا -= صفیں_handled;
                }

                اگر (ith % 2 == 0) {
                    spine_barrier_wait(cur_barrier);
                }

                ثابت int64_t اگلا_ni = ni + NB_ستون_تعداد * nth;
                اگر (اگلا_ni < gemm_n) {
                    nb_real = std::min(gemm_n - اگلا_ni, NB_ستون_تعداد);
                    spacemit_kernels::rvv::memcpy1d(b_ستون, reinterpret_cast<uint8_t *>(w_ڈیٹا) + اگلا_ni * صف_stride_b,
                                                    nb_real * صف_stride_b);
                }
            }
        } ورنہ {
            ثابت int64_t task_گنتی_m = spacemit_kernels::div_باری_اوپر(gemm_m, gemm_m_stride);
            ثابت int64_t task_گنتی_n = spacemit_kernels::div_باری_اوپر(gemm_n, gemm_n_stride);

            int64_t task_گنتی      = task_گنتی_m * task_گنتی_n;
            int64_t task_فی_thread = (task_گنتی + nth - 1) / nth;
            int64_t start           = ith * task_فی_thread;
            int64_t end             = std::min((ith + 1) * task_فی_thread, task_گنتی);
            برائے (int64_t compute_idx = start; compute_idx < end; compute_idx++) {
                ثابت خودکار tid_n = compute_idx / task_گنتی_m;
                ثابت خودکار tid_m = compute_idx % task_گنتی_m;

                ثابت int64_t m_آغاز = tid_m * gemm_m_stride;
                ثابت int64_t m_گنتی = std::min(gemm_m - m_آغاز, (int64_t) gemm_m_stride);

                ثابت int64_t n_آغاز = tid_n * gemm_n_stride;
                ثابت int64_t n_گنتی = std::min(gemm_n - n_آغاز, (int64_t) gemm_n_stride);

                ثابت int64_t n_blk = m_گنتی == 1 ? n_گنتی : NB_ستون_تعداد;

                uint8_t * b_ستون    = reinterpret_cast<uint8_t *>(w_ڈیٹا) + n_آغاز * صف_stride_b;
                uint8_t * b_ستون_zp = بلاک_type_has_zp<BLOC_TYPE>() ? b_ستون : خالی_اشاریہ;

                int64_t n_blk_real = 0;
                برائے (int64_t ni = 0; ni < n_گنتی; ni += n_blk_real, b_ستون += n_blk_real * صف_stride_b) {
                    n_blk_real = std::min(n_گنتی - ni, n_blk);

                    uint8_t * a_صف = quant_a_بفر + m_آغاز * صف_stride_a;

                    اعشاری * c_blk = output + m_آغاز * gemm_n + n_آغاز + ni;

                    int64_t صفیں_بقایا = m_گنتی;

                    uint8_t * b_ستون_cur    = b_ستون;
                    uint8_t * b_ستون_zp_cur = b_ستون_zp;

                    جب (صفیں_بقایا > 0) {
                        خودکار صفیں_handled = gemm_kernel(b_blk_len, a_صف, b_ستون_cur, b_ستون_zp_cur, c_blk,
                                                        صفیں_بقایا, n_blk_real, b_k_blks, gemm_n);

                        c_blk += صفیں_handled * gemm_n;
                        a_صف += صفیں_handled * صف_stride_a;

                        صفیں_بقایا -= صفیں_handled;
                    }
                }
            }
        }
    }

    خالی forward_mul_mat_id(ggml_compute_params * params, ggml_tensor * op) {
        constexpr size_t a_blk_len = INTER_سائز;
        constexpr size_t b_blk_len = INTER_سائز;

        ثابت ggml_tensor * src0 = op->src[0];
        ثابت ggml_tensor * src1 = op->src[1];
        ثابت ggml_tensor * ids  = op->src[2];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_ثنائی_OP_LOCALS

        صحیح ith = params->ith;
        صحیح nth = params->nth;

        // row groups
        ثابت صحیح n_ids = ids->ne[0];  // n_expert_used
        ثابت صحیح n_as  = ne02;        // n_expert

        ساخت mmid_صف_mapping {
            int32_t i1;
            int32_t i2;
        };

        spacemit_kernels::quantize_a_صف_def           quantize_a_صف_i8;
        spacemit_kernels::gemm_kernel_quantize_def     gemm_kernel;
        spacemit_kernels::moe_gemm_kernel_quantize_def moe_gemm_kernel_m2;
        منطقی                                           سیٹ_kernel_impl = جھوٹ;
        size_t                                         بلاک_stride_a  = spacemit_kernels::q8_blk_سائز(QK4_0);

#اگر defined(RISCV64_SPACEMIT_IME2)
        اگر (!سیٹ_kernel_impl && (global_spine_env_info.use_ime2)) {
            quantize_a_صف_i8 = spacemit_kernels::rvv::quantize_a_صف_i8;
            بلاک_stride_a    = spacemit_kernels::q8_blk_سائز(QK4_0, سچ);

            اگر constexpr (std::is_same_v<BLOC_TYPE, بلاک_q6_K> || std::is_same_v<BLOC_TYPE, بلاک_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i8;
                سیٹ_kernel_impl = سچ;
            } ورنہ اگر constexpr (std::is_same_v<BLOC_TYPE, بلاک_q4_0> || std::is_same_v<BLOC_TYPE, بلاک_q4_1> ||
                                 std::is_same_v<BLOC_TYPE, بلاک_q4_K>) {
                اگر constexpr (INTER_سائز == 256) {
                    gemm_kernel       = spacemit_kernels::ime2::gemm_kernel_i8i4_hp;
                    quantize_a_صف_i8 = spacemit_kernels::rvv::quantize_a_صف_i8_hp;
                    بلاک_stride_a    = spacemit_kernels::q8_hp_blk_سائز(a_blk_len, سچ, سچ);
                    سیٹ_kernel_impl   = سچ;
                } ورنہ {
                    gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i4;
                    moe_gemm_kernel_m2 = spacemit_kernels::ime2::moe_m2_gemm_kernel_i8i4;
                    quantize_a_صف_i8  = spacemit_kernels::rvv::quantize_a_صف_i8;
                    بلاک_stride_a     = spacemit_kernels::q8_blk_سائز(a_blk_len, سچ);
                    سیٹ_kernel_impl    = سچ;
                }
            } ورنہ اگر constexpr (std::is_same_v<BLOC_TYPE, بلاک_q2_K>) {
                quantize_a_صف_i8 = spacemit_kernels::rvv::quantize_a_صف_i8k;
                بلاک_stride_a    = spacemit_kernels::q8k_blk_سائز(a_blk_len);
                gemm_kernel       = spacemit_kernels::ime2::gemm_kernel_i8i2k;
                سیٹ_kernel_impl   = سچ;
            } ورنہ اگر constexpr (std::is_same_v<BLOC_TYPE, بلاک_q3_K>) {
                quantize_a_صف_i8 = spacemit_kernels::rvv::quantize_a_صف_i8k;
                بلاک_stride_a    = spacemit_kernels::q8k_blk_سائز(a_blk_len);
                gemm_kernel       = spacemit_kernels::ime2::gemm_kernel_i8i3k;
                سیٹ_kernel_impl   = سچ;
            } ورنہ اگر constexpr (std::is_same_v<BLOC_TYPE, بلاک_mxfp4>) {
                gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8mxfp4;
                moe_gemm_kernel_m2 = spacemit_kernels::ime2::moe_m2_gemm_kernel_i8mxfp4;
                سیٹ_kernel_impl    = سچ;
            } ورنہ اگر constexpr (std::is_same_v<BLOC_TYPE, بلاک_q5_1> || std::is_same_v<BLOC_TYPE, بلاک_q5_K> ||
                                 std::is_same_v<BLOC_TYPE, بلاک_q5_0>) {
                gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i5;
                moe_gemm_kernel_m2 = spacemit_kernels::ime2::moe_m2_gemm_kernel_i8i5;
                سیٹ_kernel_impl    = سچ;
            }
        }
#اگر_ختم

#اگر defined(RISCV64_SPACEMIT_IME1)
        اگر (!سیٹ_kernel_impl && (global_spine_env_info.use_ime1)) {
            quantize_a_صف_i8 = spacemit_kernels::ime1::quantize_a_صف_i8;

            اگر constexpr (std::is_same_v<BLOC_TYPE, بلاک_q4_0> || std::is_same_v<BLOC_TYPE, بلاک_q4_1> ||
                          std::is_same_v<BLOC_TYPE, بلاک_q4_K>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i4;
                سیٹ_kernel_impl = سچ;
            } ورنہ اگر constexpr (std::is_same_v<BLOC_TYPE, بلاک_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i8;
                سیٹ_kernel_impl = سچ;
            }
        }
#اگر_ختم
        اگر (!سیٹ_kernel_impl) {
            GGML_ABORT("no kernel implementation found for the block type");
        }

        ثابت size_t a_k_blks = spacemit_kernels::div_باری_اوپر(ne10, a_blk_len);
        ثابت size_t b_k_blks = spacemit_kernels::div_باری_اوپر(ne10, b_blk_len);

        ثابت size_t nbw1                = a_k_blks * بلاک_stride_a;
        ثابت size_t nbw2                = ne11 * nbw1;
        ثابت size_t nbw3                = nbw2 * ne12;
        ثابت size_t gemm_workspace_سائز = GGML_PAD(nbw3, alignof(int64_t));

        ثابت uintptr_t ws_ptr         = reinterpret_cast<uintptr_t>(params->wdata);
        خودکار *          quant_a_بفر = reinterpret_cast<uint8_t *>(ws_ptr);

        اگر (ne11 == 1) {
            برائے (int64_t ii = ith; ii < ne12 * a_k_blks; ii += nth) {
                int64_t i12       = ii / a_k_blks;
                int64_t ak_blk_id = ii % a_k_blks;
                quantize_a_صف_i8(a_blk_len, (اعشاری *) ((حرف *) src1->ڈیٹا + i12 * nb12) + ak_blk_id * a_blk_len,
                                  a_blk_len, quant_a_بفر + i12 * nbw2 + ak_blk_id * بلاک_stride_a);
            }
        } ورنہ {
            برائے (int64_t ii = ith; ii < ne12 * ne11; ii += nth) {
                int64_t i12 = ii / ne11;
                int64_t i11 = ii % ne11;
                quantize_a_صف_i8(a_blk_len, (اعشاری *) ((حرف *) src1->ڈیٹا + i12 * nb12 + i11 * nb11), ne10,
                                  quant_a_بفر + i12 * nbw2 + i11 * nbw1);
            }
        }

#تعریف_کرو MMID_میٹرکس_صف(صف_id, i1) میٹرکس_صفیں[(صف_id) *ne12 + (i1)]

        int64_t *          میٹرکس_صف_counts       = (int64_t *) (ws_ptr + gemm_workspace_سائز);
        int32_t *          valid_ep_گنتی          = (int32_t *) (میٹرکس_صف_counts + n_as);
        int32_t *          valid_act_گنتی         = (int32_t *) (valid_ep_گنتی + 1);
        int64_t *          valid_میٹرکس_صف_counts = (int64_t *) (valid_act_گنتی + 1);
        mmid_صف_mapping * میٹرکس_صفیں             = (mmid_صف_mapping *) (valid_میٹرکس_صف_counts + n_as);

        اگر (ith == 0) {
            // initialize matrix_row_counts
            memset(میٹرکس_صف_counts, 0, n_as * حجم(int64_t));

            // group rows by src0 matrix
            برائے (int32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
                برائے (int32_t id = 0; id < n_ids; ++id) {
                    ثابت int32_t i02 =
                        *(ثابت int32_t *) ((ثابت حرف *) ids->ڈیٹا + iid1 * ids->nb[1] + id * ids->nb[0]);

                    GGML_ASSERT(i02 >= 0 && i02 < n_as);

                    MMID_میٹرکس_صف(i02, میٹرکس_صف_counts[i02]) = { id, iid1 };
                    میٹرکس_صف_counts[i02] += 1;
                }
            }

            int32_t valid_ep_گنتی_t  = 0;
            int32_t valid_act_گنتی_t = 0;
            برائے (صحیح cur_a = 0; cur_a < n_as; ++cur_a) {
                ثابت int64_t cne1 = میٹرکس_صف_counts[cur_a];
                اگر (cne1 == 0) {
                    جاری;
                }
                valid_میٹرکس_صف_counts[valid_ep_گنتی_t] = cur_a;
                valid_act_گنتی_t += cne1;
                valid_ep_گنتی_t += 1;
            }
            valid_ep_گنتی[0]  = valid_ep_گنتی_t;
            valid_act_گنتی[0] = valid_act_گنتی_t;
        }

        ثابت int64_t barrier_idx = static_cast<int64_t>(ith / 2);

        GGML_ASSERT(global_spine_env_info.init_barrier != خالی_اشاریہ);
        GGML_ASSERT(barrier_idx < spine_init_barrier_گنتی);
        spine_barrier_t * cur_barrier = &global_spine_env_info.init_barrier[barrier_idx];

        ggml_barrier(params->threadpool);

        ثابت size_t صف_stride_b      = b_k_blks * get_repacked_بلاک_type_سائز<BLOC_TYPE, INTER_سائز, NB_ستون_تعداد>();
        ثابت size_t expert_b_stride   = ne01 * صف_stride_b;
        ثابت size_t فی_nb_ستون_تعداد_wsize = NB_ستون_تعداد * صف_stride_b;

        std::array<ثابت uint8_t *, 2> src_workspaces;
        std::array<اعشاری *, 2>         dst_workspaces;

        خودکار *     tcm_بفر      = ggml::cpu::riscv64_spacemit::tls_context.tcm_بفر;
        ثابت خودکار tcm_بفر_سائز = ggml::cpu::riscv64_spacemit::tls_context.tcm_بفر_سائز;

        ثابت خودکار valid_ep_گنتی_t  = valid_ep_گنتی[0];
        ثابت خودکار valid_act_گنتی_t = valid_act_گنتی[0];

        صحیح nth_es = 1;
        صحیح nth_n  = nth;

        صحیح ith_es = ith % nth_es;
        صحیح ith_n  = (ith / nth_es) % nth_n;

        اگر (valid_ep_گنتی_t % nth == 0 && tcm_بفر != خالی_اشاریہ && valid_ep_گنتی_t == n_as &&
            valid_act_گنتی_t == n_as && فی_nb_ستون_تعداد_wsize <= tcm_بفر_سائز) {
            برائے (int64_t valid_id = ith; valid_id < valid_ep_گنتی_t; valid_id += nth) {
                ثابت int64_t cur_a = valid_میٹرکس_صف_counts[valid_id];

                خودکار * src0_cur = (uint8_t *) src0->ڈیٹا + cur_a * expert_b_stride;

                mmid_صف_mapping صف_mapping = MMID_میٹرکس_صف(cur_a, 0);
                ثابت صحیح        id          = صف_mapping.i1;
                ثابت int64_t    i11         = id % ne11;
                ثابت int64_t    i12         = صف_mapping.i2;
                ثابت int64_t    i1          = id;
                ثابت int64_t    i2          = i12;

                خودکار *  src1_ستون = quant_a_بفر + (i11 * nbw1 + i12 * nbw2);
                اعشاری * c_blk    = (اعشاری *) ((حرف *) dst->ڈیٹا + (i1 * nb1 + i2 * nb2));

                uint8_t * a_صف = src1_ستون;
                uint8_t * b_ستون = reinterpret_cast<uint8_t *>(tcm_بفر);
                اگر ((nbw1 + فی_nb_ستون_تعداد_wsize) <= tcm_بفر_سائز) {
                    a_صف = (uint8_t *) tcm_بفر;
                    b_ستون = reinterpret_cast<uint8_t *>(tcm_بفر) + nbw1;
                }
                uint8_t * b_ستون_zp = بلاک_type_has_zp<BLOC_TYPE>() ? b_ستون : خالی_اشاریہ;

                اگر (ith % 2 == 0) {
                    spacemit_kernels::rvv::memcpy1d(b_ستون, reinterpret_cast<uint8_t *>(src0_cur), فی_nb_ستون_تعداد_wsize);

                    اگر (a_صف != src1_ستون) {
                        spacemit_kernels::rvv::memcpy1d(a_صف, src1_ستون, nbw1);
                    }
                }

                spine_barrier_wait(cur_barrier);

                اگر (ith % 2 != 0) {
                    اگر (a_صف != src1_ستون) {
                        spacemit_kernels::rvv::memcpy1d(a_صف, src1_ستون, nbw1);
                    }

                    spacemit_kernels::rvv::memcpy1d(b_ستون, reinterpret_cast<uint8_t *>(src0_cur), فی_nb_ستون_تعداد_wsize);
                }

                int64_t nb_real = std::min(ne01, NB_ستون_تعداد);
                برائے (int64_t ni = 0; ni < ne01; ni += NB_ستون_تعداد) {
                    اگر (ith % 2 != 0) {
                        spine_barrier_wait(cur_barrier);
                    }

                    gemm_kernel(b_blk_len, a_صف, b_ستون, b_ستون_zp, c_blk + ni, 1, nb_real, b_k_blks, ne01);

                    اگر (ith % 2 == 0) {
                        spine_barrier_wait(cur_barrier);
                    }

                    ثابت int64_t اگلا_ni = ni + NB_ستون_تعداد;
                    اگر (اگلا_ni < ne01) {
                        nb_real = std::min(ne01 - اگلا_ni, NB_ستون_تعداد);
                        spacemit_kernels::rvv::memcpy1d(
                            b_ستون, reinterpret_cast<uint8_t *>(src0_cur) + اگلا_ni * صف_stride_b, فی_nb_ستون_تعداد_wsize);
                    }
                }
            }
        } ورنہ {
            برائے (int64_t valid_id = ith_es; valid_id < valid_ep_گنتی_t; valid_id += nth_es) {
                ثابت int64_t cur_a = valid_میٹرکس_صف_counts[valid_id];
                ثابت int64_t cne1  = میٹرکس_صف_counts[cur_a];

                int64_t src1_cur_آغاز = 0;
                int64_t src1_cur_انتہا   = cne1;

                int64_t src0_cur_آغاز = (ith_n * ne01) / nth_n;
                int64_t src0_cur_انتہا   = MIN(((ith_n + 1) * ne01) / nth_n, ne01);

                اگر (src1_cur_آغاز >= src1_cur_انتہا || src0_cur_آغاز >= src0_cur_انتہا) {
                    جاری;
                }

                src0_cur_آغاز =
                    (src0_cur_آغاز % NB_ستون_تعداد) ? src0_cur_آغاز + NB_ستون_تعداد - (src0_cur_آغاز % NB_ستون_تعداد) : src0_cur_آغاز;
                src0_cur_انتہا =
                    (src0_cur_انتہا % NB_ستون_تعداد) ? src0_cur_انتہا + NB_ستون_تعداد - (src0_cur_انتہا % NB_ستون_تعداد) : src0_cur_انتہا;

                خودکار *    src0_cur = (uint8_t *) src0->ڈیٹا + cur_a * expert_b_stride + src0_cur_آغاز * صف_stride_b;
                uint8_t * b_ستون_zp = بلاک_type_has_zp<BLOC_TYPE>() ? src0_cur : خالی_اشاریہ;

                size_t extra_tcm_بفر_سائز = tcm_بفر_سائز;
                خالی * extra_tcm_بفر      = tcm_بفر;
                اگر (tcm_بفر != خالی_اشاریہ && (src1_cur_انتہا - src1_cur_آغاز) >= 4 &&
                    (src0_cur_انتہا - src0_cur_آغاز) * صف_stride_b <= tcm_بفر_سائز) {
                    spacemit_kernels::rvv::memcpy1d(tcm_بفر, src0_cur,
                                                    (src0_cur_انتہا - src0_cur_آغاز) * صف_stride_b);
                    src0_cur = reinterpret_cast<uint8_t *>(tcm_بفر);
                    b_ستون_zp = بلاک_type_has_zp<BLOC_TYPE>() ? src0_cur : خالی_اشاریہ;
                    extra_tcm_بفر_سائز -= (src0_cur_انتہا - src0_cur_آغاز) * صف_stride_b;
                    extra_tcm_بفر = reinterpret_cast<خالی *>(reinterpret_cast<uintptr_t>(tcm_بفر) +
                                                                (src0_cur_انتہا - src0_cur_آغاز) * صف_stride_b);
                }

                صحیح ir1 = src1_cur_آغاز;

                اگر (extra_tcm_بفر_سائز >= nbw1 && extra_tcm_بفر != خالی_اشاریہ) {
                    int64_t quant_a_tile_سائز = extra_tcm_بفر_سائز / nbw1;
                    کرو {
                        quant_a_tile_سائز = MIN(quant_a_tile_سائز, src1_cur_انتہا - ir1);

                        uint8_t * quant_a_tile_بفر = reinterpret_cast<uint8_t *>(extra_tcm_بفر);

                        صحیح iir1 = ir1;
                        برائے (; iir1 < (ir1 + quant_a_tile_سائز); ++iir1) {
                            mmid_صف_mapping صف_mapping = MMID_میٹرکس_صف(cur_a, iir1);

                            ثابت صحیح id = صف_mapping.i1;  // selected expert index

                            ثابت int64_t i11 = id % ne11;
                            ثابت int64_t i12 = صف_mapping.i2;  // row index in src1

                            خودکار * src1_ستون = quant_a_بفر + (i11 * nbw1 + i12 * nbw2);
                            spacemit_kernels::rvv::memcpy1d(quant_a_tile_بفر, src1_ستون, nbw1);
                            quant_a_tile_بفر = quant_a_tile_بفر + nbw1;
                        }

                        quant_a_tile_بفر = reinterpret_cast<uint8_t *>(extra_tcm_بفر);
                        iir1                = ir1;

                        اگر (moe_gemm_kernel_m2 != خالی_اشاریہ) {
                            برائے (; iir1 < (ir1 + quant_a_tile_سائز - 1); iir1 += 2, quant_a_tile_بفر += 2 * nbw1) {
                                mmid_صف_mapping صف_mapping_0 = MMID_میٹرکس_صف(cur_a, iir1);
                                mmid_صف_mapping صف_mapping_1 = MMID_میٹرکس_صف(cur_a, iir1 + 1);

                                src_workspaces[0] = quant_a_tile_بفر;
                                src_workspaces[1] = quant_a_tile_بفر + nbw1;

                                dst_workspaces[0] =
                                    (اعشاری *) ((حرف *) dst->ڈیٹا + (صف_mapping_0.i1 * nb1 + صف_mapping_0.i2 * nb2)) +
                                    src0_cur_آغاز;
                                dst_workspaces[1] = (اعشاری *) ((حرف *) dst->ڈیٹا +
                                                               ((صف_mapping_1.i1) * nb1 + (صف_mapping_1.i2) * nb2)) +
                                                    src0_cur_آغاز;
                                moe_gemm_kernel_m2(b_blk_len, src_workspaces.ڈیٹا(), src0_cur, b_ستون_zp,
                                                   dst_workspaces.ڈیٹا(), 1, src0_cur_انتہا - src0_cur_آغاز, b_k_blks,
                                                   ne01);
                            }
                        }

                        برائے (; iir1 < (ir1 + quant_a_tile_سائز); iir1++, quant_a_tile_بفر += nbw1) {
                            mmid_صف_mapping صف_mapping_0 = MMID_میٹرکس_صف(cur_a, iir1);

                            gemm_kernel(
                                b_blk_len, quant_a_tile_بفر, src0_cur, b_ستون_zp,
                                (اعشاری *) ((حرف *) dst->ڈیٹا + (صف_mapping_0.i1 * nb1 + صف_mapping_0.i2 * nb2)) +
                                    src0_cur_آغاز,
                                1, src0_cur_انتہا - src0_cur_آغاز, b_k_blks, ne01);
                        }

                        ir1 += quant_a_tile_سائز;
                    } جب (ir1 < src1_cur_انتہا);
                } ورنہ {
                    اگر (moe_gemm_kernel_m2 != خالی_اشاریہ) {
                        برائے (; ir1 < src1_cur_انتہا - 1; ir1 += 2) {
                            برائے (صحیح iir1 = 0; iir1 < 2; ++iir1) {
                                mmid_صف_mapping صف_mapping = MMID_میٹرکس_صف(cur_a, ir1 + iir1);

                                ثابت صحیح id = صف_mapping.i1;  // selected expert index

                                ثابت int64_t i11 = id % ne11;
                                ثابت int64_t i12 = صف_mapping.i2;  // row index in src1

                                ثابت int64_t i1 = id;               // selected expert index
                                ثابت int64_t i2 = i12;              //صف

                                src_workspaces[iir1] = quant_a_بفر + (i11 * nbw1 + i12 * nbw2);

                                dst_workspaces[iir1] =
                                    (اعشاری *) ((حرف *) dst->ڈیٹا + (i1 * nb1 + i2 * nb2)) + src0_cur_آغاز;
                            }

                            moe_gemm_kernel_m2(b_blk_len, src_workspaces.ڈیٹا(), src0_cur, b_ستون_zp,
                                               dst_workspaces.ڈیٹا(), 1, src0_cur_انتہا - src0_cur_آغاز, b_k_blks, ne01);
                        }
                    }

                    برائے (; ir1 < src1_cur_انتہا; ir1++) {
                        mmid_صف_mapping صف_mapping = MMID_میٹرکس_صف(cur_a, ir1);

                        ثابت صحیح id = صف_mapping.i1;  // selected expert index

                        ثابت int64_t i11 = id % ne11;
                        ثابت int64_t i12 = صف_mapping.i2;  // row index in src1

                        ثابت int64_t i1 = id;               // selected expert index
                        ثابت int64_t i2 = i12;              //صف

                        خودکار * src1_ستون = quant_a_بفر + (i11 * nbw1 + i12 * nbw2);

                        gemm_kernel(b_blk_len, src1_ستون, src0_cur, b_ستون_zp,
                                    (اعشاری *) ((حرف *) dst->ڈیٹا + (i1 * nb1 + i2 * nb2)) + src0_cur_آغاز, 1,
                                    src0_cur_انتہا - src0_cur_آغاز, b_k_blks, ne01);
                    }
                }
            }
        }
#تعریف_منسوخ MMID_میٹرکس_صف
    }

    صحیح repack(ggml_tensor * t, ثابت خالی * ڈیٹا, size_t ڈیٹا_سائز) override {
        GGML_LOG_DEBUG("%s: repack tensor %s with %s_%dx%d\n", __func__, t->name, ggml_type_name(t->type),
                       (صحیح) NB_ستون_تعداد, (صحیح) INTER_سائز);
        واپس ggml::cpu::riscv64_spacemit::repack<BLOC_TYPE, INTER_سائز, NB_ستون_تعداد>(t, ڈیٹا, ڈیٹا_سائز);
    }
};

جماعت tensor_traits_common : عوامی tensor_traits_بنیاد {
    منطقی work_سائز(صحیح n_دھاگے, ثابت ggml_tensor * op, size_t & size) override {
        انتخاب (op->op) {
            حالت GGML_OP_FLASH_ATTN_EXT:
                {
                    ثابت صحیح     n_tasks = n_دھاگے;
                    ثابت int64_t neq2    = op->src[0]->ne[2];  // number of query heads
                    ثابت int64_t DK      = op->src[1]->ne[0];
                    ثابت int64_t DV      = op->src[2]->ne[0];  // DV

                    // Tiled flash attention scratch (tile sizes defined in common.h)
                    // Per-thread: Q_q + KQ + mask + VKQ32 + V32 + K_f32 + padding
                    size_t prefill = حجم(اعشاری) *
                                     (GGML_FA_TILE_Q * DK + 2 * GGML_FA_TILE_Q * GGML_FA_TILE_KV + GGML_FA_TILE_Q * DV +
                                      GGML_FA_TILE_KV * DV + GGML_FA_TILE_KV * DK) *
                                     n_tasks;

                    // Decode path: n_kv_chunks = n_tasks (one chunk per thread)
                    // Per-thread: VKQ accmulator (DV), partial M, partial S + intra-thread scratch for V, Q and VKQ
                    size_t n_chunks = n_tasks;
                    size_t decode   = حجم(اعشاری) * (neq2 * n_chunks * (2 + DV) + n_tasks * (DK + 2 * DV));

                    size = MAX(prefill, decode);
                }
                واپس سچ;
            طےشدہ:
                توڑو;
        }
        واپس جھوٹ;
    }

    منطقی compute_forward(ggml_compute_params * params, ggml_tensor * op) override {
        انتخاب (op->op) {
            حالت GGML_OP_NORM:
                انتخاب (op->src[0]->type) {
                    حالت GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_norm_f32(params, op);
                        واپس سچ;
                    طےشدہ:
                        GGML_ABORT("fatal error");
                }
            حالت GGML_OP_RMS_NORM:
                انتخاب (op->src[0]->type) {
                    حالت GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_rms_norm_f32(params, op);
                        واپس سچ;
                    طےشدہ:
                        GGML_ABORT("fatal error");
                }
            حالت GGML_OP_جوڑو:
                انتخاب (op->src[0]->type) {
                    حالت GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_ثنائی<GGML_OP_جوڑو, اعشاری>(params, op);
                        واپس سچ;
                    حالت GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_ثنائی<GGML_OP_جوڑو, _Float16>(params, op);
                        واپس سچ;
                    طےشدہ:
                        ggml_compute_forward_جوڑو(params, op);
                        واپس سچ;
                }
            حالت GGML_OP_SUB:
                انتخاب (op->src[0]->type) {
                    حالت GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_ثنائی<GGML_OP_SUB, اعشاری>(params, op);
                        واپس سچ;
                    حالت GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_ثنائی<GGML_OP_SUB, _Float16>(params, op);
                        واپس سچ;
                    طےشدہ:
                        ggml_compute_forward_sub(params, op);
                        واپس سچ;
                }
            حالت GGML_OP_MUL:
                انتخاب (op->src[0]->type) {
                    حالت GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_ثنائی<GGML_OP_MUL, اعشاری>(params, op);
                        واپس سچ;
                    حالت GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_ثنائی<GGML_OP_MUL, _Float16>(params, op);
                        واپس سچ;
                    طےشدہ:
                        ggml_compute_forward_mul(params, op);
                        واپس سچ;
                }
            حالت GGML_OP_DIV:
                انتخاب (op->src[0]->type) {
                    حالت GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_ثنائی<GGML_OP_DIV, اعشاری>(params, op);
                        واپس سچ;
                    حالت GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_ثنائی<GGML_OP_DIV, _Float16>(params, op);
                        واپس سچ;
                    طےشدہ:
                        ggml_compute_forward_div(params, op);
                        واپس سچ;
                }
            حالت GGML_OP_FLASH_ATTN_EXT:
                forward_flash_attn_ext_f16(params, op);
                واپس سچ;
            حالت GGML_OP_CONT:
                {
                    ثابت ggml_tensor * src0 = op->src[0];
                    اگر (op->type == src0->type && op->nb[0] != src0->nb[0] && op->nb[0] == src0->nb[1] &&
                        op->ne[3] * op->ne[2] * op->nb[2] == src0->ne[3] * src0->ne[2] * src0->nb[2]) {
                        spacemit_kernels::rvv::forward_cont_with_ترتیب_بدلو(params, op);
                    } ورنہ {
                        ggml_compute_forward_cont(params, op);
                    }
                    واپس سچ;
                }
            حالت GGML_OP_CPY:
                {
                    ثابت ggml_tensor * src0 = op->src[0];
                    اگر (op->type == src0->type && op->nb[0] == src0->nb[1] && src0->nb[0] != src0->nb[1] &&
                        ggml_nelements(src0) == ggml_nelements(op)) {
                        spacemit_kernels::rvv::forward_cpy_with_ترتیب_بدلو(params, op);
                    } ورنہ {
                        ggml_compute_forward_cpy(params, op);
                    }
                    واپس سچ;
                }
            حالت GGML_OP_REPEAT:
                {
                    ثابت منطقی صفیں_equal         = ggml_nrows(op->src[0]) == ggml_nrows(op);
                    ثابت منطقی نشر_or_equal = op->src[0]->ne[0] == 1 || op->src[0]->ne[0] == op->ne[0];

                    اگر (صفیں_equal && نشر_or_equal) {
                        انتخاب (op->src[0]->type) {
                            حالت GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_repeat_nrows<int32_t>(params, op);
                                واپس سچ;
                            حالت GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_repeat_nrows<int16_t>(params, op);
                                واپس سچ;
                            طےشدہ:
                                توڑو;
                        }
                    }

                    اگر (op->src[0]->ne[1] == 1 && op->src[0]->ne[0] == op->ne[0]) {
                        انتخاب (op->src[0]->type) {
                            حالت GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_repeat_dim1<int32_t>(params, op);
                                واپس سچ;
                            حالت GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_repeat_dim1<int16_t>(params, op);
                                واپس سچ;
                            طےشدہ:
                                توڑو;
                        }
                    }

                    ggml_compute_forward_repeat(params, op);
                }
                واپس سچ;
            حالت GGML_OP_مجموع_صفیں:
                {
                    اگر (op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32) {
                        spacemit_kernels::rvv::forward_مجموع_صفیں<اعشاری>(params, op);
                    } ورنہ {
                        ggml_compute_forward_مجموع_صفیں(params, op);
                    }
                }
                واپس سچ;
            حالت GGML_OP_GET_صفیں:
                {
                    اگر (op->src[0]->type == op->type) {
                        انتخاب (op->src[0]->type) {
                            حالت GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_get_صفیں<int32_t>(params, op);
                                واپس سچ;
                            حالت GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_get_صفیں<int16_t>(params, op);
                                واپس سچ;
                            طےشدہ:
                                توڑو;
                        }
                    }

                    ggml_compute_forward_get_صفیں(params, op);
                }
                واپس سچ;
            حالت GGML_OP_CONCAT:
                {
                    ثابت int32_t dim = ggml_get_op_params_i32(op, 0);
                    اگر (dim == 0 && op->type == op->src[0]->type) {
                        انتخاب (op->src[0]->type) {
                            حالت GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_concat<int32_t>(params, op);
                                واپس سچ;
                            حالت GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_concat<int16_t>(params, op);
                                واپس سچ;
                            طےشدہ:
                                توڑو;
                        }
                    }

                    ggml_compute_forward_concat(params, op);
                }
                واپس سچ;
            // TODO For GGML_OP_GATED_DELTA_NET
            // case GGML_OP_GATED_DELTA_NET:
            //     return true;
            طےشدہ:
                توڑو;
        }
        واپس جھوٹ;
    }

    خالی forward_flash_attn_ext_f16(ثابت ggml_compute_params * params, ggml_tensor * dst) {
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

        ثابت منطقی supported_prec  = (dst->op_params[3] == GGML_PREC_F32 || dst->op_params[3] == GGML_PREC_DEFAULT);
        ثابت منطقی supported_types = (q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16);
        ثابت منطقی supported_shape = (DK > 0 && DK <= 128 && DV > 0 && DV <= 128);
        ثابت منطقی supported_vlen  = (__riscv_vlenb() == 128);

        اگر (!(supported_prec && supported_types && supported_shape && supported_vlen)) {
            ggml_compute_forward_flash_attn_ext(params, dst);
            واپس;
        }

        // total rows in q
        ثابت int64_t nr = neq1 * neq2 * neq3;

        // rows per thread
        ثابت صحیح ith = params->ith;
        ثابت صحیح nth = params->nth;

        ساکن constexpr int64_t Q_TILE_SZ = ggml_fa_tile_config::Q;
        ثابت منطقی               use_tiled = !params->use_ref && (neq1 >= Q_TILE_SZ);

        // 4x chunks per thread
        // int     nth_scaled = nth * 4;
        // int64_t chunk_size = (nr + nth_scaled - 1) / nth_scaled;
        // int64_t nchunk     = (nr + chunk_size - 1) / chunk_size;

        // if (nth == 1 || nchunk < nth) {
        //     nchunk = nth;
        // }

        int64_t nchunk = nth;

        اگر (ith == 0) {
            // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
            ggml_threadpool_chunk_سیٹ(params->threadpool, nth);
        }

        ggml_barrier(params->threadpool);

        // The number of elements in each chunk
        ثابت int64_t dr = (nr + nchunk - 1) / nchunk;

        // The first chunk comes from our thread_id, the rest will get auto-assigned.
        صحیح current_chunk = ith;

        جب (current_chunk < nchunk) {
            ثابت int64_t ir0 = dr * current_chunk;
            ثابت int64_t ir1 = MIN(ir0 + dr, nr);

            اگر (use_tiled) {
                spacemit_kernels::rvv::forward_flash_attn_ext_f16_tiled_vlen1024_vf16(
                    params, dst, ir0, ir1, ggml::cpu::riscv64_spacemit::tls_context.tcm_بفر,
                    ggml::cpu::riscv64_spacemit::tls_context.tcm_بفر_سائز);
            } ورنہ {
                spacemit_kernels::rvv::forward_flash_attn_ext_f16_one_chunk_vlen1024_vf16(
                    params, dst, ir0, ir1, ggml::cpu::riscv64_spacemit::tls_context.tcm_بفر,
                    ggml::cpu::riscv64_spacemit::tls_context.tcm_بفر_سائز);
            }

            current_chunk = ggml_threadpool_chunk_جوڑو(params->threadpool, 1);
        }
    }

    صحیح repack(ggml_tensor * t, ثابت خالی * ڈیٹا, size_t ڈیٹا_سائز) override {
        حافظہ_نقل(t->ڈیٹا, ڈیٹا, ڈیٹا_سائز);
        واپس 0;
    }
};

// Impl By IME1
ساکن ثابت tensor_traits<بلاک_q4_0, 32, 16>  q4_0_16x32_q8_0;
ساکن ثابت tensor_traits<بلاک_q4_1, 32, 16>  q4_1_16x32_q8_0;
ساکن ثابت tensor_traits<بلاک_q4_K, 32, 16>  q4_k_16x32_q8_0;
ساکن ثابت tensor_traits<بلاک_q8_0, 32, 16>  q8_0_16x32_q8_0;
// Impl By IME2
ساکن ثابت tensor_traits<بلاک_q2_K, 256, 32> q2_k_32x256_q8_0;
ساکن ثابت tensor_traits<بلاک_q3_K, 256, 32> q3_k_32x256_q8_0;
ساکن ثابت tensor_traits<بلاک_q4_0, 32, 32>  q4_0_32x32_q8_0;
ساکن ثابت tensor_traits<بلاک_q4_1, 32, 32>  q4_1_32x32_q8_0;
ساکن ثابت tensor_traits<بلاک_q4_0, 256, 32> q4_0_32x256_q8_0;
ساکن ثابت tensor_traits<بلاک_q4_1, 256, 32> q4_1_32x256_q8_0;
ساکن ثابت tensor_traits<بلاک_q4_K, 32, 32>  q4_k_32x32_q8_0;
ساکن ثابت tensor_traits<بلاک_q6_K, 32, 32>  q6_k_32x32_q8_0;
ساکن ثابت tensor_traits<بلاک_q8_0, 32, 32>  q8_0_32x32_q8_0;
ساکن ثابت tensor_traits<بلاک_mxfp4, 32, 32> mxfp4_32x32_q8_0;
ساکن ثابت tensor_traits<بلاک_q5_K, 32, 32>  q5_k_32x32_q8_0;
ساکن ثابت tensor_traits<بلاک_q5_1, 32, 32>  q5_1_32x32_q8_0;
ساکن ثابت tensor_traits<بلاک_q5_0, 32, 32>  q5_0_32x32_q8_0;
// Impl By RVV
ساکن ثابت tensor_traits_common               rvv_impl;

}  // namespace ggml::cpu::riscv64_spacemit

ساکن ثابت ggml::cpu::tensor_traits * ggml_riscv64_spacemit_get_optimal_repack_type(ثابت ggml_tensor * cur) {
    انتخاب (cur->type) {
        حالت GGML_TYPE_Q2_K:
            {
#اگر defined(RISCV64_SPACEMIT_IME2)
                اگر (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    واپس &ggml::cpu::riscv64_spacemit::q2_k_32x256_q8_0;
                }
#اگر_ختم
            }
            توڑو;
        حالت GGML_TYPE_Q3_K:
            {
#اگر defined(RISCV64_SPACEMIT_IME2)
                اگر (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    واپس &ggml::cpu::riscv64_spacemit::q3_k_32x256_q8_0;
                }
#اگر_ختم
            }
            توڑو;
        حالت GGML_TYPE_Q4_0:
            {
#اگر defined(RISCV64_SPACEMIT_IME2)
                اگر (cur->ne[1] % 32 == 0 && cur->ne[0] % 256 == 0 &&
                    (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    واپس &ggml::cpu::riscv64_spacemit::q4_0_32x256_q8_0;
                }

                اگر (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    واپس &ggml::cpu::riscv64_spacemit::q4_0_32x32_q8_0;
                }
#اگر_ختم

#اگر defined(RISCV64_SPACEMIT_IME1)
                اگر (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    واپس &ggml::cpu::riscv64_spacemit::q4_0_16x32_q8_0;
                }
#اگر_ختم
            }
            توڑو;
        حالت GGML_TYPE_Q4_1:
            {
#اگر defined(RISCV64_SPACEMIT_IME2)
                // TODO
                // if (cur->ne[1] % 32 == 0 && cur->ne[0] % 256 == 0 &&
                //     (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                //     return &ggml::cpu::riscv64_spacemit::q4_1_32x256_q8_0;
                // }

                اگر (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    واپس &ggml::cpu::riscv64_spacemit::q4_1_32x32_q8_0;
                }
#اگر_ختم

#اگر defined(RISCV64_SPACEMIT_IME1)
                اگر (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    واپس &ggml::cpu::riscv64_spacemit::q4_1_16x32_q8_0;
                }
#اگر_ختم
            }
            توڑو;
        حالت GGML_TYPE_Q4_K:
            {
#اگر defined(RISCV64_SPACEMIT_IME2)
                اگر (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    واپس &ggml::cpu::riscv64_spacemit::q4_k_32x32_q8_0;
                }
#اگر_ختم

#اگر defined(RISCV64_SPACEMIT_IME1)
                اگر (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    واپس &ggml::cpu::riscv64_spacemit::q4_k_16x32_q8_0;
                }
#اگر_ختم
            }
            توڑو;
        حالت GGML_TYPE_Q6_K:
            {
#اگر defined(RISCV64_SPACEMIT_IME2)
                اگر ((ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    واپس &ggml::cpu::riscv64_spacemit::q6_k_32x32_q8_0;
                }
#اگر_ختم
            }
            توڑو;
        حالت GGML_TYPE_Q8_0:
            {
#اگر defined(RISCV64_SPACEMIT_IME2)
                اگر ((ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    واپس &ggml::cpu::riscv64_spacemit::q8_0_32x32_q8_0;
                }
#اگر_ختم

#اگر defined(RISCV64_SPACEMIT_IME1)
                اگر (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    واپس &ggml::cpu::riscv64_spacemit::q8_0_16x32_q8_0;
                }
#اگر_ختم
            }
            توڑو;
        حالت GGML_TYPE_MXFP4:
            {
#اگر defined(RISCV64_SPACEMIT_IME2)
                // TODO
                // if (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                //     return &ggml::cpu::riscv64_spacemit::mxfp4_32x32_q8_0;
                // }
#اگر_ختم
            }
            توڑو;
        حالت GGML_TYPE_Q5_K:
            {
#اگر defined(RISCV64_SPACEMIT_IME2)
                اگر (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    واپس &ggml::cpu::riscv64_spacemit::q5_k_32x32_q8_0;
                }
#اگر_ختم
            }
            توڑو;
        حالت GGML_TYPE_Q5_1:
            {
#اگر defined(RISCV64_SPACEMIT_IME2)
                اگر (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    واپس &ggml::cpu::riscv64_spacemit::q5_1_32x32_q8_0;
                }
#اگر_ختم
            }
            توڑو;
        حالت GGML_TYPE_Q5_0:
            {
#اگر defined(RISCV64_SPACEMIT_IME2)
                اگر (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    واپس &ggml::cpu::riscv64_spacemit::q5_0_32x32_q8_0;
                }
#اگر_ختم
            }
            توڑو;
        طےشدہ:
            توڑو;
    }

    واپس خالی_اشاریہ;
}

ساکن شمار ggml_حالت ggml_backend_riscv64_spacemit_بفر_init_tensor(ggml_backend_بفر_t بفر,
                                                                         ggml_tensor *         tensor) {
    tensor->extra =
        (خالی *) const_cast<ggml::cpu::tensor_traits *>(ggml_riscv64_spacemit_get_optimal_repack_type(tensor));

    GGML_غیر_مستعمل(بفر);

    واپس GGML_حالت_SUCCESS;
}

ساکن خالی ggml_backend_riscv64_spacemit_بفر_free_بفر(ggml_backend_بفر_t بفر) {
    GGML_ASSERT(بفر);

    خالی * بنیاد = بفر->context;
    اگر (بنیاد == خالی_اشاریہ) {
        واپس;
    }

    ggml::cpu::riscv64_spacemit::spine_mem_pool_free(بنیاد);
}

ساکن خالی * ggml_backend_riscv64_spacemit_بفر_get_بنیاد(ggml_backend_بفر_t بفر) {
    GGML_ASSERT(بفر);

    خالی * بنیاد = بفر->context;
    GGML_ASSERT(بنیاد != خالی_اشاریہ);
    واپس بنیاد;
}

ساکن خالی ggml_backend_riscv64_spacemit_بفر_memset_tensor(ggml_backend_بفر_t بفر,
                                                               ggml_tensor *         tensor,
                                                               uint8_t               value,
                                                               size_t                انحراف,
                                                               size_t                size) {
    GGML_ASSERT(tensor);
    memset((حرف *) tensor->ڈیٹا + انحراف, value, size);

    GGML_غیر_مستعمل(بفر);
}

ساکن خالی ggml_backend_riscv64_spacemit_بفر_clear(ggml_backend_بفر_t بفر, uint8_t value) {
    GGML_ASSERT(بفر);

    خالی * بنیاد = بفر->context;
    GGML_ASSERT(بنیاد != خالی_اشاریہ);
    memset(بنیاد, value, بفر->size);
}

ساکن خالی ggml_backend_riscv64_spacemit_بفر_سیٹ_tensor(ggml_backend_بفر_t بفر,
                                                            ggml_tensor *         tensor,
                                                            ثابت خالی *          ڈیٹا,
                                                            size_t                انحراف,
                                                            size_t                size) {
    GGML_ASSERT(انحراف == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    خودکار tensor_traits = (ggml::cpu::riscv64_spacemit::tensor_traits_بنیاد *) tensor->extra;
    اگر (tensor_traits) {
        خودکار OK = tensor_traits->repack(tensor, ڈیٹا, size);
        GGML_ASSERT(OK == 0);
    }

    GGML_غیر_مستعمل(بفر);
}

ساکن ثابت ggml_backend_بفر_i ggml_backend_riscv64_spacemit_بفر_i = {
    /* .free_buffer     = */ ggml_backend_riscv64_spacemit_بفر_free_بفر,
    /* .get_base        = */ ggml_backend_riscv64_spacemit_بفر_get_بنیاد,
    /* .init_tensor     = */ ggml_backend_riscv64_spacemit_بفر_init_tensor,
    /* .memset_tensor   = */ ggml_backend_riscv64_spacemit_بفر_memset_tensor,
    /* .set_tensor      = */ ggml_backend_riscv64_spacemit_بفر_سیٹ_tensor,
    /* .get_tensor      = */ خالی_اشاریہ,
    /* .set_tensor_2d   = */ خالی_اشاریہ,
    /* .get_tensor_2d   = */ خالی_اشاریہ,
    /* .cpy_tensor      = */ خالی_اشاریہ,
    /* .clear           = */ ggml_backend_riscv64_spacemit_بفر_clear,
    /* .reset           = */ خالی_اشاریہ,
};

ساکن ثابت حرف * ggml_backend_cpu_riscv64_spacemit_بفر_type_get_name(ggml_backend_بفر_type_t buft) {
    واپس "CPU_RISCV64_SPACEMIT";

    GGML_غیر_مستعمل(buft);
}

ساکن ggml_backend_بفر_t ggml_backend_cpu_riscv64_spacemit_بفر_type_alloc_بفر(ggml_backend_بفر_type_t buft,
                                                                                        size_t size) {
    خالی * بنیاد = ggml::cpu::riscv64_spacemit::spine_mem_pool_alloc(size, 64);
    اگر (بنیاد == خالی_اشاریہ) {
        واپس خالی_اشاریہ;
    }

    واپس ggml_backend_بفر_init(buft, ggml_backend_riscv64_spacemit_بفر_i, بنیاد, size);
}

ساکن size_t ggml_backend_cpu_riscv64_spacemit_بفر_type_get_alignment(ggml_backend_بفر_type_t buft) {
    واپس 64;

    GGML_غیر_مستعمل(buft);
}

ساکن size_t ggml_backend_cpu_riscv64_spacemit_nbytes(ggml_backend_بفر_type_t buft, ثابت ggml_tensor * tensor) {
    برائے (صحیح i = 0; i < GGML_MAX_DIMS; ++i) {
        اگر (tensor->ne[i] <= 0) {
            واپس 0;
        }
    }

    GGML_غیر_مستعمل(buft);

    ثابت خودکار plain_nbytes = [&]() {
        size_t کل = ggml_type_سائز(tensor->type);
        برائے (صحیح i = 0; i < GGML_MAX_DIMS; ++i) {
            کل += (tensor->ne[i] - 1) * tensor->nb[i];
        }
        واپس کل;
    };

    ثابت size_t blck_سائز = ggml_blck_سائز(tensor->type);
    اگر (blck_سائز == 1) {
        واپس plain_nbytes();
    }

    ثابت size_t صف_nbytes = tensor->ne[0] * tensor->nb[0] / blck_سائز;

    ثابت خودکار جوڑو_strided_nbytes = [&](size_t کل, size_t src_بلاک_سائز, size_t dst_بلاک_سائز) {
        برائے (صحیح i = 1; i < GGML_MAX_DIMS; ++i) {
            کل += (tensor->ne[i] - 1) * (tensor->nb[i] / src_بلاک_سائز) * dst_بلاک_سائز;
        }
        واپس کل;
    };

    ثابت خودکار remap_بلاک_nbytes = [&](size_t src_بلاک_سائز, size_t dst_بلاک_سائز, int64_t padded_صفیں = 0) {
        GGML_ASSERT(صف_nbytes % src_بلاک_سائز == 0);

        size_t کل =
            جوڑو_strided_nbytes((صف_nbytes / src_بلاک_سائز) * dst_بلاک_سائز, src_بلاک_سائز, dst_بلاک_سائز);

        اگر (padded_صفیں > 0 && tensor->ne[1] % padded_صفیں != 0) {
            کل += (padded_صفیں - tensor->ne[1] % padded_صفیں) * (tensor->nb[1] / src_بلاک_سائز) * dst_بلاک_سائز;
        }

        واپس کل;
    };

    size_t nbytes = صف_nbytes;
    انتخاب (tensor->type) {
        حالت GGML_TYPE_Q4_K:
            nbytes = remap_بلاک_nbytes(حجم(بلاک_q4_K), حجم(بلاک_q4_1) * 8);
            توڑو;
        حالت GGML_TYPE_Q6_K:
            nbytes = remap_بلاک_nbytes(حجم(بلاک_q6_K), حجم(بلاک_q8_0) * 8, 32);
            توڑو;
        حالت GGML_TYPE_Q8_0:
            nbytes = remap_بلاک_nbytes(حجم(بلاک_q8_0), حجم(بلاک_q8_0), 32);
            توڑو;
        حالت GGML_TYPE_Q2_K:
            nbytes = remap_بلاک_nbytes(حجم(بلاک_q2_K), حجم(spacemit_kernels::nrow_بلاک_q2_k<1>));
            توڑو;
        حالت GGML_TYPE_Q3_K:
            nbytes = remap_بلاک_nbytes(حجم(بلاک_q3_K), حجم(spacemit_kernels::nrow_بلاک_q3_k<1>));
            توڑو;
        حالت GGML_TYPE_MXFP4:
            nbytes = remap_بلاک_nbytes(حجم(بلاک_mxfp4), حجم(spacemit_kernels::nrow_بلاک_mxfp4<1>));
            توڑو;
        حالت GGML_TYPE_Q5_K:
            nbytes = remap_بلاک_nbytes(حجم(بلاک_q5_K), حجم(spacemit_kernels::nrow_بلاک_q5_1<1>) * 8);
            توڑو;
        حالت GGML_TYPE_Q5_1:
            nbytes = remap_بلاک_nbytes(حجم(بلاک_q5_1), حجم(spacemit_kernels::nrow_بلاک_q5_1<1>));
            توڑو;
        حالت GGML_TYPE_Q5_0:
            nbytes = remap_بلاک_nbytes(حجم(بلاک_q5_0), حجم(spacemit_kernels::nrow_بلاک_q5_0<1>));
            توڑو;
        طےشدہ:
            nbytes = جوڑو_strided_nbytes(صف_nbytes, 1, 1);
            توڑو;
    }

    واپس nbytes;
}

نامخلا ggml::cpu::riscv64_spacemit {

جماعت extra_بفر_type : ggml::cpu::extra_بفر_type {
    منطقی supports_op(ggml_backend_dev_t, ثابت ggml_tensor * op) override {
        انتخاب (op->op) {
            حالت GGML_OP_MUL_MAT:
                اگر (op->src[0]->بفر && (ggml_n_dims(op->src[0]) == 2) &&
                    op->src[0]->بفر->buft == ggml_backend_cpu_riscv64_spacemit_بفر_type() &&
                    ggml_riscv64_spacemit_get_optimal_repack_type(op->src[0])) {
                    اگر (op->src[1]->بفر && !ggml_backend_buft_is_host(op->src[1]->بفر->buft)) {
                        واپس جھوٹ;
                    }
                    اگر (op->src[1]->type == GGML_TYPE_F32) {
                        واپس سچ;
                    }
                }
                توڑو;
            حالت GGML_OP_MUL_MAT_ID:
                اگر (op->src[0]->بفر && (ggml_n_dims(op->src[0]) == 3) &&
                    op->src[0]->بفر->buft == ggml_backend_cpu_riscv64_spacemit_بفر_type() &&
                    ggml_riscv64_spacemit_get_optimal_repack_type(op->src[0])) {
                    اگر (op->src[1]->بفر && !ggml_backend_buft_is_host(op->src[1]->بفر->buft)) {
                        واپس جھوٹ;
                    }
                    اگر (op->src[1]->type == GGML_TYPE_F32) {
                        واپس سچ;
                    }
                }
                توڑو;
            طےشدہ:
                // GGML_ABORT("fatal error");
                توڑو;
        }
        واپس جھوٹ;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(ثابت ggml_tensor * op) override {
        انتخاب (op->op) {
            حالت GGML_OP_MUL_MAT:
            حالت GGML_OP_MUL_MAT_ID:
                اگر (op->src[0]->بفر && op->src[0]->بفر->buft == ggml_backend_cpu_riscv64_spacemit_بفر_type()) {
                    واپس (ggml::cpu::tensor_traits *) op->src[0]->extra;
                }
                توڑو;
            حالت GGML_OP_NORM:
            حالت GGML_OP_RMS_NORM:
            حالت GGML_OP_جوڑو:
            حالت GGML_OP_SUB:
            حالت GGML_OP_MUL:
            حالت GGML_OP_DIV:
            حالت GGML_OP_FLASH_ATTN_EXT:
            حالت GGML_OP_CONT:
            حالت GGML_OP_CPY:
            حالت GGML_OP_REPEAT:
            حالت GGML_OP_مجموع_صفیں:
            حالت GGML_OP_GET_صفیں:
            حالت GGML_OP_CONCAT:
                // case GGML_OP_GATED_DELTA_NET:
                واپس (ggml::cpu::tensor_traits *) (&ggml::cpu::riscv64_spacemit::rvv_impl);
            طےشدہ:
                // GGML_ABORT("fatal error");
                توڑو;
        }

        واپس خالی_اشاریہ;
    }
};

}  // namespace ggml::cpu::riscv64_spacemit

ggml_backend_بفر_type_t ggml_backend_cpu_riscv64_spacemit_بفر_type(خالی) {
    ساکن ggml_backend_بفر_type ggml_backend_cpu_بفر_type_riscv64_spacemit = {
  /* .iface    = */
        {
         /* .get_name           = */ ggml_backend_cpu_riscv64_spacemit_بفر_type_get_name,
         /* .alloc_buffer       = */ ggml_backend_cpu_riscv64_spacemit_بفر_type_alloc_بفر,
         /* .alloc_buffer_n     = */ تہی,
         /* .get_alignment      = */ ggml_backend_cpu_riscv64_spacemit_بفر_type_get_alignment,
         /* .get_max_size       = */ خالی_اشاریہ,
         /* .get_alloc_size     = */ ggml_backend_cpu_riscv64_spacemit_nbytes,
         /* .get_alloc_size_n   = */ تہی,
         /* .is_host            = */ خالی_اشاریہ,
         },
 /* .device  = */
        ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
 /* .context = */
        نیا ggml::cpu::riscv64_spacemit::extra_بفر_type(),
    };

    واپس &ggml_backend_cpu_بفر_type_riscv64_spacemit;
}

خارجی "C" {
ساکن صحیح bind_ai_thread() {
    صحیح  fd, بائٹیں;
    حرف str[32];

    fd = open("/proc/set_ai_thread", O_WRONLY);
    اگر (fd < 0) {
        GGML_LOG_ERROR("try open /proc/set_ai_thread failed\n");
        واپس -1;
    }

    snprintf(str, 16, "%d", 0);
    بائٹیں = اكتب(fd, str, لڑی_لمبائی(str));
    اگر (بائٹیں < 0) {
        GGML_LOG_ERROR("try write /proc/set_ai_thread failed\n");
        close(fd);
        واپس -1;
    }

    close(fd);
    واپس 0;
}

خالی ggml_backend_cpu_riscv64_spacemit_سیٹ_numa_thread_affinity(صحیح thread_n) {
    صحیح cpu_id = sched_getcpu();
    اگر (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2 &&
        !((1 << cpu_id) & ggml::cpu::riscv64_spacemit::global_spine_env_info.cpu_نقاب)) {
        GGML_PRINT_DEBUG("bind_ai_thread for thread %d, pid %d\n", thread_n, getpid());
        bind_ai_thread();
    }

    اگر (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_tcm &&
        ggml::cpu::riscv64_spacemit::tls_context.cpu_id == -1) {
        CPU_ZERO(&(ggml::cpu::riscv64_spacemit::tls_context.cpuset));
        pthread_t    main_thread     = pthread_self();
        ثابت خودکار & perfer_core_ids = ggml::cpu::riscv64_spacemit::global_spine_env_info.perfer_core_ids;
        اگر (thread_n < 0 || static_cast<size_t>(thread_n) >= perfer_core_ids.size()) {
            GGML_ABORT("thread_n %d exceeds perfer_core_ids size %zu\n", thread_n, perfer_core_ids.size());
        }
        خودکار perfer_cpu_id = perfer_core_ids[static_cast<size_t>(thread_n)];
        CPU_SET(perfer_cpu_id, &(ggml::cpu::riscv64_spacemit::tls_context.cpuset));
        صحیح s =
            pthread_setaffinity_np(main_thread, حجم(cpu_set_t), &(ggml::cpu::riscv64_spacemit::tls_context.cpuset));
        اگر (s != 0) {
            GGML_ABORT("set thread affinity error for thread_n %d, cpu_id %d\n", thread_n, perfer_cpu_id);
        }

        صحیح ai_cpu_id = perfer_cpu_id - ggml::cpu::riscv64_spacemit::global_spine_env_info.aicpu_id_انحراف;
        ggml::cpu::riscv64_spacemit::tls_context.cpu_id = ai_cpu_id;
        ggml::cpu::riscv64_spacemit::tls_context.tcm_بفر =
            ggml::cpu::riscv64_spacemit::spine_mem_pool_tcm_mem_get(ai_cpu_id);
        ggml::cpu::riscv64_spacemit::tls_context.tcm_بفر_سائز =
            ggml::cpu::riscv64_spacemit::global_spine_env_info.tcm_blk_سائز;
    }

    اگر (ggml::cpu::riscv64_spacemit::tls_context.tcm_بفر != خالی_اشاریہ) {
        خالی * rt =
            ggml::cpu::riscv64_spacemit::spine_mem_pool_tcm_mem_wait(ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        اگر (rt == خالی_اشاریہ) {
            GGML_ABORT("wait tcm buffer failed for cpu_id: %d", ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        }
    }
}

خالی ggml_backend_cpu_riscv64_spacemit_clear_numa_thread_affinity_threaded(صحیح thread_n) {
    اگر (ggml::cpu::riscv64_spacemit::tls_context.tcm_بفر != خالی_اشاریہ) {
        خودکار rt = ggml::cpu::riscv64_spacemit::spine_mem_pool_tcm_mem_release(
            ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        اگر (rt != 0) {
            GGML_ABORT("release tcm buffer failed for cpu_id: %d", ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        }
    }
}
}
