#define GGML_COMMON_IMPL_CPP
#define GGML_COMMON_DECL_CPP

#অন্তর্ভাব "ime.h"

#অন্তর্ভাব "binary-ops.h"
#অন্তর্ভাব "common.h"
#অন্তর্ভাব "ggml-backend-impl.h"
#অন্তর্ভাব "ggml-common.h"
#অন্তর্ভাব "ggml-cpu.h"
#অন্তর্ভাব "ime_env.h"
#অন্তর্ভাব "ime_kernels.h"
#অন্তর্ভাব "ops.h"
#অন্তর্ভাব "repack.h"
#অন্তর্ভাব "rvv_kernels.h"
#অন্তর্ভাব "spine_mem_pool.h"
#অন্তর্ভাব "traits.h"
#অন্তর্ভাব "vec.h"

#অন্তর্ভাব <fcntl.h>
#অন্তর্ভাব <sys/mman.h>
#অন্তর্ভাব <unistd.h>

#অন্তর্ভাব <algorithm>
#অন্তর্ভাব <atomic>
#অন্তর্ভাব <cassert>
#অন্তর্ভাব <cerrno>
#অন্তর্ভাব <cmath>
#অন্তর্ভাব <cstdio>  // for GGML_ASSERT
#অন্তর্ভাব <stdexcept>
#অন্তর্ভাব <thread>
// clang-format off
#যদি defined(__riscv)

#যদি !defined(__riscv_v) || !defined(__riscv_v_intrinsic)
#error "riscv v extension or v_intrinsic not enabled"
#অন্যথা
#অন্তর্ভাব <riscv_vector.h>
#endif

#যদি !defined(__riscv_zfh) || !defined(__riscv_zvfh)
#error "riscv zfh extension not enabled, GGML_RV_ZFH and GGML_RV_ZVFH must be defined to 1"
#endif

#যদি !defined(__riscv_zba)
#error "riscv zba extension not enabled, GGML_RV_ZBA must be defined to 1"
#endif

#যদি defined(RISCV64_SPACEMIT_IME1) || defined(RISCV64_SPACEMIT_IME2)
#অন্যথা
#error "RISCV64_SPACEMIT_IME1 or RISCV64_SPACEMIT_IME2 not defined"
#endif

#অন্যথা

#error "riscv not enabled in this build"

#endif

#যদি defined(__GNUC__)
#pragma GCC diagnostic ignored "-Woverlength-strings"
#pragma GCC diagnostic ignored "-Wcast-qual"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

// clang-format on

বাহ্যিক "C" {
বাহ্যিক শূন্য ggml_threadpool_chunk_set(কাঠামো ggml_threadpool * tp, পূর্ণসংখ্যা value);
বাহ্যিক পূর্ণসংখ্যা  ggml_threadpool_chunk_add(কাঠামো ggml_threadpool * tp, পূর্ণসংখ্যা value);
}

নামাকাশ ggml::cpu::riscv64_spacemit {

কাঠামো TLSContext {
    পূর্ণসংখ্যা       cpu_id{ -1 };
    cpu_set_t cpuset;
    শূন্য *    tcm_বাফার{ nullptr };
    size_t    tcm_বাফার_আকার{ 0 };
};

thread_local TLSContext tls_context;

template <typename BLOC_TYPE, int64_t INTER_আকার, int64_t NB_COLS> constexpr size_t get_repacked_block_type_আকার() {
    যদি constexpr (std::is_same_v<BLOC_TYPE, block_q6_K> || std::is_same_v<BLOC_TYPE, block_q8_0>) {
        ফেরত আকার(block_q8_0);
    } অন্যথা যদি constexpr (std::is_same_v<BLOC_TYPE, block_q4_0>) {
        ফেরত আকার(block_q4_0) * INTER_আকার / QK4_0;
    } অন্যথা যদি constexpr (std::is_same_v<BLOC_TYPE, block_q4_1> || std::is_same_v<BLOC_TYPE, block_q4_K>) {
        ফেরত (আকার(block_q4_0) + আকার(uint8_t)) * INTER_আকার / QK4_1;
    } অন্যথা যদি constexpr (std::is_same_v<BLOC_TYPE, block_q2_K>) {
        ফেরত আকার(spacemit_kernels::nrow_block_q2_k<1>);
    } অন্যথা যদি constexpr (std::is_same_v<BLOC_TYPE, block_q3_K>) {
        ফেরত আকার(spacemit_kernels::nrow_block_q3_k<1>);
    } অন্যথা যদি constexpr (std::is_same_v<BLOC_TYPE, block_mxfp4>) {
        ফেরত আকার(spacemit_kernels::nrow_block_mxfp4<1>);
    } অন্যথা যদি constexpr (std::is_same_v<BLOC_TYPE, block_q5_1> || std::is_same_v<BLOC_TYPE, block_q5_K>) {
        ফেরত আকার(spacemit_kernels::nrow_block_q5_1<1>);
    } অন্যথা যদি constexpr (std::is_same_v<BLOC_TYPE, block_q5_0>) {
        ফেরত আকার(spacemit_kernels::nrow_block_q5_0<1>);
    } অন্যথা {
        প্রতিজ্ঞা(মিথ্যা);
        ফেরত 0;
    }
}

template <typename BLOC_TYPE> constexpr যৌক্তিক block_type_has_zp() {
    যদি constexpr (std::is_same_v<BLOC_TYPE, block_q6_K> || std::is_same_v<BLOC_TYPE, block_q8_0> ||
                  std::is_same_v<BLOC_TYPE, block_q3_K> || std::is_same_v<BLOC_TYPE, block_q4_0> ||
                  std::is_same_v<BLOC_TYPE, block_mxfp4> || std::is_same_v<BLOC_TYPE, block_q5_0>) {
        ফেরত মিথ্যা;
    } অন্যথা যদি constexpr (std::is_same_v<BLOC_TYPE, block_q4_1> || std::is_same_v<BLOC_TYPE, block_q4_K> ||
                         std::is_same_v<BLOC_TYPE, block_q2_K> || std::is_same_v<BLOC_TYPE, block_q5_1> ||
                         std::is_same_v<BLOC_TYPE, block_q5_K>) {
        ফেরত সত্য;
    } অন্যথা {
        প্রতিজ্ঞা(মিথ্যা);
        ফেরত মিথ্যা;
    }
}

শ্রেণী tensor_traits_base : সর্বজনীন ggml::cpu::tensor_traits {
  সর্বজনীন:
    আভাসী পূর্ণসংখ্যা repack(ggml_tensor * t, ধ্রুবক শূন্য * তথ্য, size_t তথ্য_আকার) = 0;
};

template <typename BLOC_TYPE, int64_t INTER_আকার, int64_t NB_COLS> শ্রেণী tensor_traits : সর্বজনীন tensor_traits_base {
    যৌক্তিক work_আকার(পূর্ণসংখ্যা /* n_threads */, ধ্রুবক ggml_tensor * op, size_t & size) override {
        নির্বাচন (op->op) {
            ক্ষেত্র GGML_OP_MUL_MAT:
                {
                    int64_t src1_nelements = ggml_nelements(op->src[1]);

                    যদি constexpr (std::is_same_v<BLOC_TYPE, block_q2_K> || std::is_same_v<BLOC_TYPE, block_q3_K>) {
                        size =
                            spacemit_kernels::div_round_up(src1_nelements, QK_K) * spacemit_kernels::q8k_blk_আকার(QK_K);
                    } অন্যথা যদি constexpr (INTER_আকার == QK4_0) {
                        size = spacemit_kernels::div_round_up(src1_nelements, QK4_0) *
                               spacemit_kernels::q8_blk_আকার(QK4_0, সত্য);
                    } অন্যথা যদি constexpr (INTER_আকার == 256) {
                        size = spacemit_kernels::div_round_up(src1_nelements, 256) *
                               spacemit_kernels::q8_hp_blk_আকার(256, সত্য, সত্য);
                    } অন্যথা {
                        GGML_ABORT("unsupported block type");
                    }

                    size = GGML_PAD(size, আকার(int64_t));

                    ফেরত সত্য;
                }
            ক্ষেত্র GGML_OP_MUL_MAT_ID:
                {
                    int64_t src1_nelements = ggml_nelements(op->src[1]);

                    যদি constexpr (std::is_same_v<BLOC_TYPE, block_q2_K> || std::is_same_v<BLOC_TYPE, block_q3_K>) {
                        size =
                            spacemit_kernels::div_round_up(src1_nelements, QK_K) * spacemit_kernels::q8k_blk_আকার(QK_K);
                    } অন্যথা যদি constexpr (INTER_আকার == QK4_0) {
                        size = spacemit_kernels::div_round_up(src1_nelements, QK4_0) *
                               spacemit_kernels::q8_blk_আকার(QK4_0, সত্য);
                    } অন্যথা যদি constexpr (INTER_আকার == 256) {
                        size = spacemit_kernels::div_round_up(src1_nelements, 256) *
                               spacemit_kernels::q8_hp_blk_আকার(256, সত্য, সত্য);
                    } অন্যথা {
                        GGML_ABORT("unsupported block type");
                    }

                    size = GGML_PAD(size, আকার(int64_t));

                    ধ্রুবক int64_t ne02 = op->src[0]->ne[2];  // n_as, n_expert
                    ধ্রুবক int64_t ne12 = op->src[1]->ne[2];  // n_tokens

                    ধ্রুবক size_t sizeof_mmid_row_mapping = আকার(int64_t);
                    size += sizeof_mmid_row_mapping * ne02 * (ne12 + 1) + (ne02 + 1) * আকার(int64_t);

                    size = GGML_PAD(size, আকার(int64_t));

                    ফেরত সত্য;
                }
            পূর্বনির্ধারিত:
                // GGML_ABORT("fatal error");
                বিরতি;
        }
        ফেরত মিথ্যা;
    }

    যৌক্তিক compute_forward(ggml_compute_params * params, ggml_tensor * op) override {
        নির্বাচন (op->op) {
            ক্ষেত্র GGML_OP_MUL_MAT:
                নির্বাচন (op->src[0]->type) {
                    ক্ষেত্র GGML_TYPE_Q2_K:
                    ক্ষেত্র GGML_TYPE_Q3_K:
                    ক্ষেত্র GGML_TYPE_Q4_0:
                    ক্ষেত্র GGML_TYPE_Q4_1:
                    ক্ষেত্র GGML_TYPE_Q4_K:
                    ক্ষেত্র GGML_TYPE_Q6_K:
                    ক্ষেত্র GGML_TYPE_Q8_0:
                    ক্ষেত্র GGML_TYPE_Q5_0:
                    ক্ষেত্র GGML_TYPE_Q5_1:
                    ক্ষেত্র GGML_TYPE_Q5_K:
                        //case GGML_TYPE_MXFP4:
                        forward_mul_mat(params, op);
                        ফেরত সত্য;
                    পূর্বনির্ধারিত:
                        // GGML_ABORT("fatal error: unsupported type for src0 in MUL_MAT");
                        ফেরত মিথ্যা;
                }
                বিরতি;
            ক্ষেত্র GGML_OP_MUL_MAT_ID:
                নির্বাচন (op->src[0]->type) {
                    ক্ষেত্র GGML_TYPE_Q2_K:
                    ক্ষেত্র GGML_TYPE_Q3_K:
                    ক্ষেত্র GGML_TYPE_Q4_0:
                    ক্ষেত্র GGML_TYPE_Q4_1:
                    ক্ষেত্র GGML_TYPE_Q4_K:
                    ক্ষেত্র GGML_TYPE_Q6_K:
                    ক্ষেত্র GGML_TYPE_Q8_0:
                    ক্ষেত্র GGML_TYPE_Q5_0:
                    ক্ষেত্র GGML_TYPE_Q5_1:
                    ক্ষেত্র GGML_TYPE_Q5_K:
                        //case GGML_TYPE_MXFP4:
                        forward_mul_mat_id(params, op);
                        ফেরত সত্য;
                    পূর্বনির্ধারিত:
                        // GGML_ABORT("fatal error: unsupported type for src0 in MUL_MAT_ID");
                        ফেরত মিথ্যা;
                }
                বিরতি;
            পূর্বনির্ধারিত:
                // GGML_ABORT("fatal error");
                বিরতি;
        }
        ফেরত মিথ্যা;
    }

    শূন্য forward_mul_mat(ggml_compute_params * params, ggml_tensor * op) {
        constexpr size_t a_blk_len = INTER_আকার;
        constexpr size_t b_blk_len = INTER_আকার;

        ধ্রুবক ggml_tensor * src0 = op->src[0];
        ধ্রুবক ggml_tensor * src1 = op->src[1];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_দ্বিমিক_OP_LOCALS

        পূর্ণসংখ্যা ith = params->ith;
        পূর্ণসংখ্যা nth = params->nth;

        [[maybe_unused]] ধ্রুবক গণনা ggml_type type = src0->type;

        শূন্য *        w_তথ্য  = (শূন্য *) src0->তথ্য;
        ধ্রুবক দশমিক * feature = (ধ্রুবক দশমিক *) src1->তথ্য;
        দশমিক *       output  = (দশমিক *) dst->তথ্য;

        ধ্রুবক int64_t gemm_m = ne11 * ne12 * ne13;
        ধ্রুবক int64_t gemm_k = ne10;
        ধ্রুবক int64_t gemm_n = ne01;

        spacemit_kernels::quantize_a_row_def       quantize_a_row_i8;
        spacemit_kernels::quantize_a_row_def       quantize_a_4row_i8;
        spacemit_kernels::gemm_kernel_quantize_def gemm_kernel;
        যৌক্তিক                                       set_kernel_impl = মিথ্যা;

        int64_t block_stride_a = spacemit_kernels::q8_blk_আকার(a_blk_len);

#যদি defined(RISCV64_SPACEMIT_IME2)
        যদি (!set_kernel_impl && (global_spine_env_info.use_ime2)) {
            quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8;
            quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8;
            block_stride_a     = spacemit_kernels::q8_blk_আকার(a_blk_len, সত্য);

            যদি constexpr (std::is_same_v<BLOC_TYPE, block_q6_K> || std::is_same_v<BLOC_TYPE, block_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i8;
                set_kernel_impl = সত্য;
            } অন্যথা যদি constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> || std::is_same_v<BLOC_TYPE, block_q4_1> ||
                                 std::is_same_v<BLOC_TYPE, block_q4_K>) {
                যদি constexpr (INTER_আকার == 256) {
                    gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i4_hp;
                    quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8_hp;
                    quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8_hp;
                    block_stride_a     = spacemit_kernels::q8_hp_blk_আকার(a_blk_len, সত্য, সত্য);
                    set_kernel_impl    = সত্য;
                } অন্যথা {
                    gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i4;
                    quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8;
                    quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8;
                    block_stride_a     = spacemit_kernels::q8_blk_আকার(a_blk_len, সত্য);
                    set_kernel_impl    = সত্য;
                }
            } অন্যথা যদি constexpr (std::is_same_v<BLOC_TYPE, block_q2_K>) {
                quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8k;
                quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8k;
                block_stride_a     = spacemit_kernels::q8k_blk_আকার(a_blk_len);

                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i2k;
                set_kernel_impl = সত্য;
            } অন্যথা যদি constexpr (std::is_same_v<BLOC_TYPE, block_q3_K>) {
                quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8k;
                quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8k;
                block_stride_a     = spacemit_kernels::q8k_blk_আকার(a_blk_len);

                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i3k;
                set_kernel_impl = সত্য;
            } অন্যথা যদি constexpr (std::is_same_v<BLOC_TYPE, block_mxfp4>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8mxfp4;
                set_kernel_impl = সত্য;
            } অন্যথা যদি constexpr (std::is_same_v<BLOC_TYPE, block_q5_1> || std::is_same_v<BLOC_TYPE, block_q5_K> ||
                                 std::is_same_v<BLOC_TYPE, block_q5_0>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i5;
                set_kernel_impl = সত্য;
            }
        }
#endif

#যদি defined(RISCV64_SPACEMIT_IME1)
        যদি (!set_kernel_impl && (global_spine_env_info.use_ime1)) {
            quantize_a_row_i8  = spacemit_kernels::ime1::quantize_a_row_i8;
            quantize_a_4row_i8 = spacemit_kernels::ime1::quantize_a_4row_i8;

            যদি constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> || std::is_same_v<BLOC_TYPE, block_q4_1> ||
                          std::is_same_v<BLOC_TYPE, block_q4_K>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i4;
                set_kernel_impl = সত্য;
            } অন্যথা যদি constexpr (std::is_same_v<BLOC_TYPE, block_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i8;
                set_kernel_impl = সত্য;
            }
        }
#endif
        যদি (!set_kernel_impl) {
            GGML_ABORT("no kernel implementation found for the block type");
        }

        ধ্রুবক int64_t a_k_blks = spacemit_kernels::div_round_up(gemm_k, a_blk_len);
        ধ্রুবক int64_t b_k_blks = spacemit_kernels::div_round_up(gemm_k, b_blk_len);

        ধ্রুবক int64_t row_stride_a        = a_k_blks * block_stride_a;
        ধ্রুবক int64_t gemm_workspace_আকার = GGML_PAD(gemm_m * row_stride_a, alignof(int64_t));

        যদি (ith == 0 && params->wsize < gemm_workspace_আকার) {
            GGML_ABORT("wsize less than gemm_workspace_size");
        }

        uintptr_t ws_ptr = reinterpret_cast<uintptr_t>(params->wdata);

        শূন্য *        tcm_বাফার      = ggml::cpu::riscv64_spacemit::tls_context.tcm_বাফার;
        ধ্রুবক int64_t tcm_বাফার_আকার = ggml::cpu::riscv64_spacemit::tls_context.tcm_বাফার_আকার;

        auto * quant_a_বাফার = reinterpret_cast<uint8_t *>(ws_ptr);

        constexpr int64_t row_align = 4;
        ধ্রুবক int64_t     row_blks  = spacemit_kernels::div_round_up(gemm_m, row_align);

        ধ্রুবক int64_t row_stride_b      = b_k_blks * get_repacked_block_type_আকার<BLOC_TYPE, INTER_আকার, NB_COLS>();
        ধ্রুবক int64_t per_mb_rows_wsize = row_align * row_stride_a;
        ধ্রুবক int64_t per_nb_cols_wsize = NB_COLS * row_stride_b;

        ধ্রুবক int64_t barrier_idx = static_cast<int64_t>(ith / 2);

        GGML_ASSERT(global_spine_env_info.init_barrier != nullptr);
        GGML_ASSERT(barrier_idx < spine_init_barrier_গুনতি);
        spine_barrier_t * cur_barrier = &global_spine_env_info.init_barrier[barrier_idx];

        যদি (gemm_m == 1) {
            পূর্ণসংখ্যা task_per_thread = spacemit_kernels::div_round_up(a_k_blks, nth);
            পূর্ণসংখ্যা a_blk_start     = ith * task_per_thread;
            পূর্ণসংখ্যা a_blk_end       = std::min(a_blk_start + task_per_thread, (পূর্ণসংখ্যা) a_k_blks);
            যদি (a_blk_start < a_blk_end) {
                quantize_a_row_i8(a_blk_len, feature + a_blk_start * a_blk_len, (a_blk_end - a_blk_start) * a_blk_len,
                                  quant_a_বাফার + a_blk_start * block_stride_a);
            }
        } অন্যথা {
            পূর্ণসংখ্যা task_per_thread = spacemit_kernels::div_round_up(row_blks, nth);
            পূর্ণসংখ্যা m_row_blk_start = ith * task_per_thread;
            পূর্ণসংখ্যা m_row_blk_end   = std::min(m_row_blk_start + task_per_thread, (পূর্ণসংখ্যা) row_blks);
            জন্য (পূর্ণসংখ্যা m_row_blk = m_row_blk_start; m_row_blk < m_row_blk_end; m_row_blk++) {
                পূর্ণসংখ্যা m_idx             = m_row_blk * row_align;
                পূর্ণসংখ্যা rows_tobe_handled = (gemm_m - m_idx) > row_align ? row_align : (gemm_m - m_idx);

                যদি (rows_tobe_handled == row_align && quantize_a_4row_i8 != nullptr) {
                    ধ্রুবক দশমিক * a_row_ptr       = feature + m_idx * gemm_k;
                    auto *        quant_a_row_ptr = quant_a_বাফার + m_idx * row_stride_a;
                    quantize_a_4row_i8(a_blk_len, a_row_ptr, gemm_k, quant_a_row_ptr);
                } অন্যথা {
                    যখন (rows_tobe_handled) {
                        ধ্রুবক দশমিক * a_row_ptr       = feature + m_idx * gemm_k;
                        auto *        quant_a_row_ptr = quant_a_বাফার + m_idx * row_stride_a;
                        quantize_a_row_i8(a_blk_len, a_row_ptr, gemm_k, quant_a_row_ptr);
                        rows_tobe_handled -= 1;
                        m_idx += 1;
                    }
                }
            }
        }

        ggml_barrier(params->threadpool);

        ধ্রুবক int64_t gemm_m_stride     = gemm_n / gemm_m > 64 ? gemm_m : 16;
        ধ্রুবক int64_t gemm_m_blocked    = spacemit_kernels::div_round_up(gemm_m, gemm_m_stride);
        ধ্রুবক int64_t max_gemm_n_stride = spacemit_kernels::div_round_up(gemm_n * gemm_m_blocked, nth);

        int64_t gemm_n_stride = gemm_n;
        যদি (max_gemm_n_stride < gemm_n) {
            gemm_n_stride =
                std::min(gemm_n_stride, spacemit_kernels::div_round_up(max_gemm_n_stride, NB_COLS) * NB_COLS);
        }

        যদি (gemm_n_stride == gemm_n && tcm_বাফার != nullptr && per_mb_rows_wsize <= tcm_বাফার_আকার) {
            জন্য (int64_t m_start = ith * row_align; m_start < gemm_m; m_start += row_align * nth) {
                uint8_t * b_col    = reinterpret_cast<uint8_t *>(w_তথ্য);
                uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? b_col : nullptr;

                int64_t m_row_real = std::min(gemm_m - m_start, row_align);

                spacemit_kernels::rvv::memcpy1d(tcm_বাফার, quant_a_বাফার + m_start * row_stride_a,
                                                m_row_real * row_stride_a);

                int64_t n_blk_real = 0;
                জন্য (int64_t ni = 0; ni < gemm_n; ni += n_blk_real, b_col += n_blk_real * row_stride_b) {
                    n_blk_real = std::min(gemm_n - ni, (int64_t) NB_COLS);

                    uint8_t * a_row_ptr = (uint8_t *) tcm_বাফার;
                    দশমিক *   c_blk     = output + m_start * gemm_n + ni;

                    int32_t rows_remaining = m_row_real;

                    যখন (rows_remaining > 0) {
                        auto rows_handled = gemm_kernel(b_blk_len, a_row_ptr, b_col, b_col_zp, c_blk, rows_remaining,
                                                        n_blk_real, b_k_blks, gemm_n);

                        c_blk += rows_handled * gemm_n;
                        a_row_ptr += rows_handled * row_stride_a;

                        rows_remaining -= rows_handled;
                    }
                }
            }
        } অন্যথা যদি (tcm_বাফার != nullptr && per_nb_cols_wsize <= tcm_বাফার_আকার) {
            uint8_t * a_row = quant_a_বাফার;
            uint8_t * b_col = reinterpret_cast<uint8_t *>(tcm_বাফার);
            যদি ((gemm_workspace_আকার + per_nb_cols_wsize) <= tcm_বাফার_আকার) {
                a_row = (uint8_t *) tcm_বাফার;
                b_col = reinterpret_cast<uint8_t *>(tcm_বাফার) + gemm_workspace_আকার;
            }
            uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? b_col : nullptr;

            int64_t ni      = ith * NB_COLS;
            int64_t nb_real = std::min(gemm_n - ni, NB_COLS);

            যদি (ith % 2 == 0 && nb_real > 0) {
                spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(w_তথ্য) + ni * row_stride_b,
                                                nb_real * row_stride_b);
                যদি (a_row != quant_a_বাফার) {
                    spacemit_kernels::rvv::memcpy1d(a_row, quant_a_বাফার, gemm_workspace_আকার);
                }
            }

            spine_barrier_wait(cur_barrier);

            যদি (ith % 2 != 0 && nb_real > 0) {
                যদি (a_row != quant_a_বাফার) {
                    spacemit_kernels::rvv::memcpy1d(a_row, quant_a_বাফার, gemm_workspace_আকার);
                }
                spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(w_তথ্য) + ni * row_stride_b,
                                                nb_real * row_stride_b);
            }

            জন্য (; ni < gemm_n; ni += NB_COLS * nth) {
                int64_t rows_remaining = gemm_m;
                দশমিক * c_blk          = output + ni;
                auto *  a_row_cur      = a_row;

                যদি (ith % 2 != 0) {
                    spine_barrier_wait(cur_barrier);
                }

                যখন (rows_remaining > 0) {
                    auto rows_handled = gemm_kernel(b_blk_len, a_row_cur, b_col, b_col_zp, c_blk, rows_remaining,
                                                    nb_real, b_k_blks, gemm_n);

                    c_blk += rows_handled * gemm_n;
                    a_row_cur += rows_handled * row_stride_a;

                    rows_remaining -= rows_handled;
                }

                যদি (ith % 2 == 0) {
                    spine_barrier_wait(cur_barrier);
                }

                ধ্রুবক int64_t পরবর্তী_ni = ni + NB_COLS * nth;
                যদি (পরবর্তী_ni < gemm_n) {
                    nb_real = std::min(gemm_n - পরবর্তী_ni, NB_COLS);
                    spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(w_তথ্য) + পরবর্তী_ni * row_stride_b,
                                                    nb_real * row_stride_b);
                }
            }
        } অন্যথা {
            ধ্রুবক int64_t task_গুনতি_m = spacemit_kernels::div_round_up(gemm_m, gemm_m_stride);
            ধ্রুবক int64_t task_গুনতি_n = spacemit_kernels::div_round_up(gemm_n, gemm_n_stride);

            int64_t task_গুনতি      = task_গুনতি_m * task_গুনতি_n;
            int64_t task_per_thread = (task_গুনতি + nth - 1) / nth;
            int64_t start           = ith * task_per_thread;
            int64_t end             = std::min((ith + 1) * task_per_thread, task_গুনতি);
            জন্য (int64_t compute_idx = start; compute_idx < end; compute_idx++) {
                ধ্রুবক auto tid_n = compute_idx / task_গুনতি_m;
                ধ্রুবক auto tid_m = compute_idx % task_গুনতি_m;

                ধ্রুবক int64_t m_start = tid_m * gemm_m_stride;
                ধ্রুবক int64_t m_গুনতি = std::min(gemm_m - m_start, (int64_t) gemm_m_stride);

                ধ্রুবক int64_t n_start = tid_n * gemm_n_stride;
                ধ্রুবক int64_t n_গুনতি = std::min(gemm_n - n_start, (int64_t) gemm_n_stride);

                ধ্রুবক int64_t n_blk = m_গুনতি == 1 ? n_গুনতি : NB_COLS;

                uint8_t * b_col    = reinterpret_cast<uint8_t *>(w_তথ্য) + n_start * row_stride_b;
                uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? b_col : nullptr;

                int64_t n_blk_real = 0;
                জন্য (int64_t ni = 0; ni < n_গুনতি; ni += n_blk_real, b_col += n_blk_real * row_stride_b) {
                    n_blk_real = std::min(n_গুনতি - ni, n_blk);

                    uint8_t * a_row = quant_a_বাফার + m_start * row_stride_a;

                    দশমিক * c_blk = output + m_start * gemm_n + n_start + ni;

                    int64_t rows_remaining = m_গুনতি;

                    uint8_t * b_col_cur    = b_col;
                    uint8_t * b_col_zp_cur = b_col_zp;

                    যখন (rows_remaining > 0) {
                        auto rows_handled = gemm_kernel(b_blk_len, a_row, b_col_cur, b_col_zp_cur, c_blk,
                                                        rows_remaining, n_blk_real, b_k_blks, gemm_n);

                        c_blk += rows_handled * gemm_n;
                        a_row += rows_handled * row_stride_a;

                        rows_remaining -= rows_handled;
                    }
                }
            }
        }
    }

    শূন্য forward_mul_mat_id(ggml_compute_params * params, ggml_tensor * op) {
        constexpr size_t a_blk_len = INTER_আকার;
        constexpr size_t b_blk_len = INTER_আকার;

        ধ্রুবক ggml_tensor * src0 = op->src[0];
        ধ্রুবক ggml_tensor * src1 = op->src[1];
        ধ্রুবক ggml_tensor * ids  = op->src[2];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_দ্বিমিক_OP_LOCALS

        পূর্ণসংখ্যা ith = params->ith;
        পূর্ণসংখ্যা nth = params->nth;

        // row groups
        ধ্রুবক পূর্ণসংখ্যা n_ids = ids->ne[0];  // n_expert_used
        ধ্রুবক পূর্ণসংখ্যা n_as  = ne02;        // n_expert

        কাঠামো mmid_row_mapping {
            int32_t i1;
            int32_t i2;
        };

        spacemit_kernels::quantize_a_row_def           quantize_a_row_i8;
        spacemit_kernels::gemm_kernel_quantize_def     gemm_kernel;
        spacemit_kernels::moe_gemm_kernel_quantize_def moe_gemm_kernel_m2;
        যৌক্তিক                                           set_kernel_impl = মিথ্যা;
        size_t                                         block_stride_a  = spacemit_kernels::q8_blk_আকার(QK4_0);

#যদি defined(RISCV64_SPACEMIT_IME2)
        যদি (!set_kernel_impl && (global_spine_env_info.use_ime2)) {
            quantize_a_row_i8 = spacemit_kernels::rvv::quantize_a_row_i8;
            block_stride_a    = spacemit_kernels::q8_blk_আকার(QK4_0, সত্য);

            যদি constexpr (std::is_same_v<BLOC_TYPE, block_q6_K> || std::is_same_v<BLOC_TYPE, block_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i8;
                set_kernel_impl = সত্য;
            } অন্যথা যদি constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> || std::is_same_v<BLOC_TYPE, block_q4_1> ||
                                 std::is_same_v<BLOC_TYPE, block_q4_K>) {
                যদি constexpr (INTER_আকার == 256) {
                    gemm_kernel       = spacemit_kernels::ime2::gemm_kernel_i8i4_hp;
                    quantize_a_row_i8 = spacemit_kernels::rvv::quantize_a_row_i8_hp;
                    block_stride_a    = spacemit_kernels::q8_hp_blk_আকার(a_blk_len, সত্য, সত্য);
                    set_kernel_impl   = সত্য;
                } অন্যথা {
                    gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i4;
                    moe_gemm_kernel_m2 = spacemit_kernels::ime2::moe_m2_gemm_kernel_i8i4;
                    quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8;
                    block_stride_a     = spacemit_kernels::q8_blk_আকার(a_blk_len, সত্য);
                    set_kernel_impl    = সত্য;
                }
            } অন্যথা যদি constexpr (std::is_same_v<BLOC_TYPE, block_q2_K>) {
                quantize_a_row_i8 = spacemit_kernels::rvv::quantize_a_row_i8k;
                block_stride_a    = spacemit_kernels::q8k_blk_আকার(a_blk_len);
                gemm_kernel       = spacemit_kernels::ime2::gemm_kernel_i8i2k;
                set_kernel_impl   = সত্য;
            } অন্যথা যদি constexpr (std::is_same_v<BLOC_TYPE, block_q3_K>) {
                quantize_a_row_i8 = spacemit_kernels::rvv::quantize_a_row_i8k;
                block_stride_a    = spacemit_kernels::q8k_blk_আকার(a_blk_len);
                gemm_kernel       = spacemit_kernels::ime2::gemm_kernel_i8i3k;
                set_kernel_impl   = সত্য;
            } অন্যথা যদি constexpr (std::is_same_v<BLOC_TYPE, block_mxfp4>) {
                gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8mxfp4;
                moe_gemm_kernel_m2 = spacemit_kernels::ime2::moe_m2_gemm_kernel_i8mxfp4;
                set_kernel_impl    = সত্য;
            } অন্যথা যদি constexpr (std::is_same_v<BLOC_TYPE, block_q5_1> || std::is_same_v<BLOC_TYPE, block_q5_K> ||
                                 std::is_same_v<BLOC_TYPE, block_q5_0>) {
                gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i5;
                moe_gemm_kernel_m2 = spacemit_kernels::ime2::moe_m2_gemm_kernel_i8i5;
                set_kernel_impl    = সত্য;
            }
        }
#endif

#যদি defined(RISCV64_SPACEMIT_IME1)
        যদি (!set_kernel_impl && (global_spine_env_info.use_ime1)) {
            quantize_a_row_i8 = spacemit_kernels::ime1::quantize_a_row_i8;

            যদি constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> || std::is_same_v<BLOC_TYPE, block_q4_1> ||
                          std::is_same_v<BLOC_TYPE, block_q4_K>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i4;
                set_kernel_impl = সত্য;
            } অন্যথা যদি constexpr (std::is_same_v<BLOC_TYPE, block_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i8;
                set_kernel_impl = সত্য;
            }
        }
#endif
        যদি (!set_kernel_impl) {
            GGML_ABORT("no kernel implementation found for the block type");
        }

        ধ্রুবক size_t a_k_blks = spacemit_kernels::div_round_up(ne10, a_blk_len);
        ধ্রুবক size_t b_k_blks = spacemit_kernels::div_round_up(ne10, b_blk_len);

        ধ্রুবক size_t nbw1                = a_k_blks * block_stride_a;
        ধ্রুবক size_t nbw2                = ne11 * nbw1;
        ধ্রুবক size_t nbw3                = nbw2 * ne12;
        ধ্রুবক size_t gemm_workspace_আকার = GGML_PAD(nbw3, alignof(int64_t));

        ধ্রুবক uintptr_t ws_ptr         = reinterpret_cast<uintptr_t>(params->wdata);
        auto *          quant_a_বাফার = reinterpret_cast<uint8_t *>(ws_ptr);

        যদি (ne11 == 1) {
            জন্য (int64_t ii = ith; ii < ne12 * a_k_blks; ii += nth) {
                int64_t i12       = ii / a_k_blks;
                int64_t ak_blk_id = ii % a_k_blks;
                quantize_a_row_i8(a_blk_len, (দশমিক *) ((অক্ষর *) src1->তথ্য + i12 * nb12) + ak_blk_id * a_blk_len,
                                  a_blk_len, quant_a_বাফার + i12 * nbw2 + ak_blk_id * block_stride_a);
            }
        } অন্যথা {
            জন্য (int64_t ii = ith; ii < ne12 * ne11; ii += nth) {
                int64_t i12 = ii / ne11;
                int64_t i11 = ii % ne11;
                quantize_a_row_i8(a_blk_len, (দশমিক *) ((অক্ষর *) src1->তথ্য + i12 * nb12 + i11 * nb11), ne10,
                                  quant_a_বাফার + i12 * nbw2 + i11 * nbw1);
            }
        }

#define MMID_MATRIX_ROW(row_id, i1) matrix_rows[(row_id) *ne12 + (i1)]

        int64_t *          matrix_row_counts       = (int64_t *) (ws_ptr + gemm_workspace_আকার);
        int32_t *          valid_ep_গুনতি          = (int32_t *) (matrix_row_counts + n_as);
        int32_t *          valid_act_গুনতি         = (int32_t *) (valid_ep_গুনতি + 1);
        int64_t *          valid_matrix_row_counts = (int64_t *) (valid_act_গুনতি + 1);
        mmid_row_mapping * matrix_rows             = (mmid_row_mapping *) (valid_matrix_row_counts + n_as);

        যদি (ith == 0) {
            // initialize matrix_row_counts
            memset(matrix_row_counts, 0, n_as * আকার(int64_t));

            // group rows by src0 matrix
            জন্য (int32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
                জন্য (int32_t id = 0; id < n_ids; ++id) {
                    ধ্রুবক int32_t i02 =
                        *(ধ্রুবক int32_t *) ((ধ্রুবক অক্ষর *) ids->তথ্য + iid1 * ids->nb[1] + id * ids->nb[0]);

                    GGML_ASSERT(i02 >= 0 && i02 < n_as);

                    MMID_MATRIX_ROW(i02, matrix_row_counts[i02]) = { id, iid1 };
                    matrix_row_counts[i02] += 1;
                }
            }

            int32_t valid_ep_গুনতি_t  = 0;
            int32_t valid_act_গুনতি_t = 0;
            জন্য (পূর্ণসংখ্যা cur_a = 0; cur_a < n_as; ++cur_a) {
                ধ্রুবক int64_t cne1 = matrix_row_counts[cur_a];
                যদি (cne1 == 0) {
                    চালু;
                }
                valid_matrix_row_counts[valid_ep_গুনতি_t] = cur_a;
                valid_act_গুনতি_t += cne1;
                valid_ep_গুনতি_t += 1;
            }
            valid_ep_গুনতি[0]  = valid_ep_গুনতি_t;
            valid_act_গুনতি[0] = valid_act_গুনতি_t;
        }

        ধ্রুবক int64_t barrier_idx = static_cast<int64_t>(ith / 2);

        GGML_ASSERT(global_spine_env_info.init_barrier != nullptr);
        GGML_ASSERT(barrier_idx < spine_init_barrier_গুনতি);
        spine_barrier_t * cur_barrier = &global_spine_env_info.init_barrier[barrier_idx];

        ggml_barrier(params->threadpool);

        ধ্রুবক size_t row_stride_b      = b_k_blks * get_repacked_block_type_আকার<BLOC_TYPE, INTER_আকার, NB_COLS>();
        ধ্রুবক size_t expert_b_stride   = ne01 * row_stride_b;
        ধ্রুবক size_t per_nb_cols_wsize = NB_COLS * row_stride_b;

        std::array<ধ্রুবক uint8_t *, 2> src_workspaces;
        std::array<দশমিক *, 2>         dst_workspaces;

        auto *     tcm_বাফার      = ggml::cpu::riscv64_spacemit::tls_context.tcm_বাফার;
        ধ্রুবক auto tcm_বাফার_আকার = ggml::cpu::riscv64_spacemit::tls_context.tcm_বাফার_আকার;

        ধ্রুবক auto valid_ep_গুনতি_t  = valid_ep_গুনতি[0];
        ধ্রুবক auto valid_act_গুনতি_t = valid_act_গুনতি[0];

        পূর্ণসংখ্যা nth_es = 1;
        পূর্ণসংখ্যা nth_n  = nth;

        পূর্ণসংখ্যা ith_es = ith % nth_es;
        পূর্ণসংখ্যা ith_n  = (ith / nth_es) % nth_n;

        যদি (valid_ep_গুনতি_t % nth == 0 && tcm_বাফার != nullptr && valid_ep_গুনতি_t == n_as &&
            valid_act_গুনতি_t == n_as && per_nb_cols_wsize <= tcm_বাফার_আকার) {
            জন্য (int64_t valid_id = ith; valid_id < valid_ep_গুনতি_t; valid_id += nth) {
                ধ্রুবক int64_t cur_a = valid_matrix_row_counts[valid_id];

                auto * src0_cur = (uint8_t *) src0->তথ্য + cur_a * expert_b_stride;

                mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, 0);
                ধ্রুবক পূর্ণসংখ্যা        id          = row_mapping.i1;
                ধ্রুবক int64_t    i11         = id % ne11;
                ধ্রুবক int64_t    i12         = row_mapping.i2;
                ধ্রুবক int64_t    i1          = id;
                ধ্রুবক int64_t    i2          = i12;

                auto *  src1_col = quant_a_বাফার + (i11 * nbw1 + i12 * nbw2);
                দশমিক * c_blk    = (দশমিক *) ((অক্ষর *) dst->তথ্য + (i1 * nb1 + i2 * nb2));

                uint8_t * a_row = src1_col;
                uint8_t * b_col = reinterpret_cast<uint8_t *>(tcm_বাফার);
                যদি ((nbw1 + per_nb_cols_wsize) <= tcm_বাফার_আকার) {
                    a_row = (uint8_t *) tcm_বাফার;
                    b_col = reinterpret_cast<uint8_t *>(tcm_বাফার) + nbw1;
                }
                uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? b_col : nullptr;

                যদি (ith % 2 == 0) {
                    spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(src0_cur), per_nb_cols_wsize);

                    যদি (a_row != src1_col) {
                        spacemit_kernels::rvv::memcpy1d(a_row, src1_col, nbw1);
                    }
                }

                spine_barrier_wait(cur_barrier);

                যদি (ith % 2 != 0) {
                    যদি (a_row != src1_col) {
                        spacemit_kernels::rvv::memcpy1d(a_row, src1_col, nbw1);
                    }

                    spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(src0_cur), per_nb_cols_wsize);
                }

                int64_t nb_real = std::min(ne01, NB_COLS);
                জন্য (int64_t ni = 0; ni < ne01; ni += NB_COLS) {
                    যদি (ith % 2 != 0) {
                        spine_barrier_wait(cur_barrier);
                    }

                    gemm_kernel(b_blk_len, a_row, b_col, b_col_zp, c_blk + ni, 1, nb_real, b_k_blks, ne01);

                    যদি (ith % 2 == 0) {
                        spine_barrier_wait(cur_barrier);
                    }

                    ধ্রুবক int64_t পরবর্তী_ni = ni + NB_COLS;
                    যদি (পরবর্তী_ni < ne01) {
                        nb_real = std::min(ne01 - পরবর্তী_ni, NB_COLS);
                        spacemit_kernels::rvv::memcpy1d(
                            b_col, reinterpret_cast<uint8_t *>(src0_cur) + পরবর্তী_ni * row_stride_b, per_nb_cols_wsize);
                    }
                }
            }
        } অন্যথা {
            জন্য (int64_t valid_id = ith_es; valid_id < valid_ep_গুনতি_t; valid_id += nth_es) {
                ধ্রুবক int64_t cur_a = valid_matrix_row_counts[valid_id];
                ধ্রুবক int64_t cne1  = matrix_row_counts[cur_a];

                int64_t src1_cur_start = 0;
                int64_t src1_cur_end   = cne1;

                int64_t src0_cur_start = (ith_n * ne01) / nth_n;
                int64_t src0_cur_end   = MIN(((ith_n + 1) * ne01) / nth_n, ne01);

                যদি (src1_cur_start >= src1_cur_end || src0_cur_start >= src0_cur_end) {
                    চালু;
                }

                src0_cur_start =
                    (src0_cur_start % NB_COLS) ? src0_cur_start + NB_COLS - (src0_cur_start % NB_COLS) : src0_cur_start;
                src0_cur_end =
                    (src0_cur_end % NB_COLS) ? src0_cur_end + NB_COLS - (src0_cur_end % NB_COLS) : src0_cur_end;

                auto *    src0_cur = (uint8_t *) src0->তথ্য + cur_a * expert_b_stride + src0_cur_start * row_stride_b;
                uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? src0_cur : nullptr;

                size_t extra_tcm_বাফার_আকার = tcm_বাফার_আকার;
                শূন্য * extra_tcm_বাফার      = tcm_বাফার;
                যদি (tcm_বাফার != nullptr && (src1_cur_end - src1_cur_start) >= 4 &&
                    (src0_cur_end - src0_cur_start) * row_stride_b <= tcm_বাফার_আকার) {
                    spacemit_kernels::rvv::memcpy1d(tcm_বাফার, src0_cur,
                                                    (src0_cur_end - src0_cur_start) * row_stride_b);
                    src0_cur = reinterpret_cast<uint8_t *>(tcm_বাফার);
                    b_col_zp = block_type_has_zp<BLOC_TYPE>() ? src0_cur : nullptr;
                    extra_tcm_বাফার_আকার -= (src0_cur_end - src0_cur_start) * row_stride_b;
                    extra_tcm_বাফার = reinterpret_cast<শূন্য *>(reinterpret_cast<uintptr_t>(tcm_বাফার) +
                                                                (src0_cur_end - src0_cur_start) * row_stride_b);
                }

                পূর্ণসংখ্যা ir1 = src1_cur_start;

                যদি (extra_tcm_বাফার_আকার >= nbw1 && extra_tcm_বাফার != nullptr) {
                    int64_t quant_a_tile_আকার = extra_tcm_বাফার_আকার / nbw1;
                    করো {
                        quant_a_tile_আকার = MIN(quant_a_tile_আকার, src1_cur_end - ir1);

                        uint8_t * quant_a_tile_বাফার = reinterpret_cast<uint8_t *>(extra_tcm_বাফার);

                        পূর্ণসংখ্যা iir1 = ir1;
                        জন্য (; iir1 < (ir1 + quant_a_tile_আকার); ++iir1) {
                            mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, iir1);

                            ধ্রুবক পূর্ণসংখ্যা id = row_mapping.i1;  // selected expert index

                            ধ্রুবক int64_t i11 = id % ne11;
                            ধ্রুবক int64_t i12 = row_mapping.i2;  // row index in src1

                            auto * src1_col = quant_a_বাফার + (i11 * nbw1 + i12 * nbw2);
                            spacemit_kernels::rvv::memcpy1d(quant_a_tile_বাফার, src1_col, nbw1);
                            quant_a_tile_বাফার = quant_a_tile_বাফার + nbw1;
                        }

                        quant_a_tile_বাফার = reinterpret_cast<uint8_t *>(extra_tcm_বাফার);
                        iir1                = ir1;

                        যদি (moe_gemm_kernel_m2 != nullptr) {
                            জন্য (; iir1 < (ir1 + quant_a_tile_আকার - 1); iir1 += 2, quant_a_tile_বাফার += 2 * nbw1) {
                                mmid_row_mapping row_mapping_0 = MMID_MATRIX_ROW(cur_a, iir1);
                                mmid_row_mapping row_mapping_1 = MMID_MATRIX_ROW(cur_a, iir1 + 1);

                                src_workspaces[0] = quant_a_tile_বাফার;
                                src_workspaces[1] = quant_a_tile_বাফার + nbw1;

                                dst_workspaces[0] =
                                    (দশমিক *) ((অক্ষর *) dst->তথ্য + (row_mapping_0.i1 * nb1 + row_mapping_0.i2 * nb2)) +
                                    src0_cur_start;
                                dst_workspaces[1] = (দশমিক *) ((অক্ষর *) dst->তথ্য +
                                                               ((row_mapping_1.i1) * nb1 + (row_mapping_1.i2) * nb2)) +
                                                    src0_cur_start;
                                moe_gemm_kernel_m2(b_blk_len, src_workspaces.তথ্য(), src0_cur, b_col_zp,
                                                   dst_workspaces.তথ্য(), 1, src0_cur_end - src0_cur_start, b_k_blks,
                                                   ne01);
                            }
                        }

                        জন্য (; iir1 < (ir1 + quant_a_tile_আকার); iir1++, quant_a_tile_বাফার += nbw1) {
                            mmid_row_mapping row_mapping_0 = MMID_MATRIX_ROW(cur_a, iir1);

                            gemm_kernel(
                                b_blk_len, quant_a_tile_বাফার, src0_cur, b_col_zp,
                                (দশমিক *) ((অক্ষর *) dst->তথ্য + (row_mapping_0.i1 * nb1 + row_mapping_0.i2 * nb2)) +
                                    src0_cur_start,
                                1, src0_cur_end - src0_cur_start, b_k_blks, ne01);
                        }

                        ir1 += quant_a_tile_আকার;
                    } যখন (ir1 < src1_cur_end);
                } অন্যথা {
                    যদি (moe_gemm_kernel_m2 != nullptr) {
                        জন্য (; ir1 < src1_cur_end - 1; ir1 += 2) {
                            জন্য (পূর্ণসংখ্যা iir1 = 0; iir1 < 2; ++iir1) {
                                mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, ir1 + iir1);

                                ধ্রুবক পূর্ণসংখ্যা id = row_mapping.i1;  // selected expert index

                                ধ্রুবক int64_t i11 = id % ne11;
                                ধ্রুবক int64_t i12 = row_mapping.i2;  // row index in src1

                                ধ্রুবক int64_t i1 = id;               // selected expert index
                                ধ্রুবক int64_t i2 = i12;              // row

                                src_workspaces[iir1] = quant_a_বাফার + (i11 * nbw1 + i12 * nbw2);

                                dst_workspaces[iir1] =
                                    (দশমিক *) ((অক্ষর *) dst->তথ্য + (i1 * nb1 + i2 * nb2)) + src0_cur_start;
                            }

                            moe_gemm_kernel_m2(b_blk_len, src_workspaces.তথ্য(), src0_cur, b_col_zp,
                                               dst_workspaces.তথ্য(), 1, src0_cur_end - src0_cur_start, b_k_blks, ne01);
                        }
                    }

                    জন্য (; ir1 < src1_cur_end; ir1++) {
                        mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, ir1);

                        ধ্রুবক পূর্ণসংখ্যা id = row_mapping.i1;  // selected expert index

                        ধ্রুবক int64_t i11 = id % ne11;
                        ধ্রুবক int64_t i12 = row_mapping.i2;  // row index in src1

                        ধ্রুবক int64_t i1 = id;               // selected expert index
                        ধ্রুবক int64_t i2 = i12;              // row

                        auto * src1_col = quant_a_বাফার + (i11 * nbw1 + i12 * nbw2);

                        gemm_kernel(b_blk_len, src1_col, src0_cur, b_col_zp,
                                    (দশমিক *) ((অক্ষর *) dst->তথ্য + (i1 * nb1 + i2 * nb2)) + src0_cur_start, 1,
                                    src0_cur_end - src0_cur_start, b_k_blks, ne01);
                    }
                }
            }
        }
#undef MMID_MATRIX_ROW
    }

    পূর্ণসংখ্যা repack(ggml_tensor * t, ধ্রুবক শূন্য * তথ্য, size_t তথ্য_আকার) override {
        GGML_LOG_DEBUG("%s: repack tensor %s with %s_%dx%d\n", __func__, t->name, ggml_type_name(t->type),
                       (পূর্ণসংখ্যা) NB_COLS, (পূর্ণসংখ্যা) INTER_আকার);
        ফেরত ggml::cpu::riscv64_spacemit::repack<BLOC_TYPE, INTER_আকার, NB_COLS>(t, তথ্য, তথ্য_আকার);
    }
};

শ্রেণী tensor_traits_common : সর্বজনীন tensor_traits_base {
    যৌক্তিক work_আকার(পূর্ণসংখ্যা n_threads, ধ্রুবক ggml_tensor * op, size_t & size) override {
        নির্বাচন (op->op) {
            ক্ষেত্র GGML_OP_FLASH_ATTN_EXT:
                {
                    ধ্রুবক পূর্ণসংখ্যা     n_tasks = n_threads;
                    ধ্রুবক int64_t neq2    = op->src[0]->ne[2];  // number of query heads
                    ধ্রুবক int64_t DK      = op->src[1]->ne[0];
                    ধ্রুবক int64_t DV      = op->src[2]->ne[0];  // DV

                    // Tiled flash attention scratch (tile sizes defined in common.h)
                    // Per-thread: Q_q + KQ + mask + VKQ32 + V32 + K_f32 + padding
                    size_t prefill = আকার(দশমিক) *
                                     (GGML_FA_TILE_Q * DK + 2 * GGML_FA_TILE_Q * GGML_FA_TILE_KV + GGML_FA_TILE_Q * DV +
                                      GGML_FA_TILE_KV * DV + GGML_FA_TILE_KV * DK) *
                                     n_tasks;

                    // Decode path: n_kv_chunks = n_tasks (one chunk per thread)
                    // Per-thread: VKQ accmulator (DV), partial M, partial S + intra-thread scratch for V, Q and VKQ
                    size_t n_chunks = n_tasks;
                    size_t decode   = আকার(দশমিক) * (neq2 * n_chunks * (2 + DV) + n_tasks * (DK + 2 * DV));

                    size = MAX(prefill, decode);
                }
                ফেরত সত্য;
            পূর্বনির্ধারিত:
                বিরতি;
        }
        ফেরত মিথ্যা;
    }

    যৌক্তিক compute_forward(ggml_compute_params * params, ggml_tensor * op) override {
        নির্বাচন (op->op) {
            ক্ষেত্র GGML_OP_NORM:
                নির্বাচন (op->src[0]->type) {
                    ক্ষেত্র GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_norm_f32(params, op);
                        ফেরত সত্য;
                    পূর্বনির্ধারিত:
                        GGML_ABORT("fatal error");
                }
            ক্ষেত্র GGML_OP_RMS_NORM:
                নির্বাচন (op->src[0]->type) {
                    ক্ষেত্র GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_rms_norm_f32(params, op);
                        ফেরত সত্য;
                    পূর্বনির্ধারিত:
                        GGML_ABORT("fatal error");
                }
            ক্ষেত্র GGML_OP_ADD:
                নির্বাচন (op->src[0]->type) {
                    ক্ষেত্র GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_দ্বিমিক<GGML_OP_ADD, দশমিক>(params, op);
                        ফেরত সত্য;
                    ক্ষেত্র GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_দ্বিমিক<GGML_OP_ADD, _Float16>(params, op);
                        ফেরত সত্য;
                    পূর্বনির্ধারিত:
                        ggml_compute_forward_add(params, op);
                        ফেরত সত্য;
                }
            ক্ষেত্র GGML_OP_SUB:
                নির্বাচন (op->src[0]->type) {
                    ক্ষেত্র GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_দ্বিমিক<GGML_OP_SUB, দশমিক>(params, op);
                        ফেরত সত্য;
                    ক্ষেত্র GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_দ্বিমিক<GGML_OP_SUB, _Float16>(params, op);
                        ফেরত সত্য;
                    পূর্বনির্ধারিত:
                        ggml_compute_forward_sub(params, op);
                        ফেরত সত্য;
                }
            ক্ষেত্র GGML_OP_MUL:
                নির্বাচন (op->src[0]->type) {
                    ক্ষেত্র GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_দ্বিমিক<GGML_OP_MUL, দশমিক>(params, op);
                        ফেরত সত্য;
                    ক্ষেত্র GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_দ্বিমিক<GGML_OP_MUL, _Float16>(params, op);
                        ফেরত সত্য;
                    পূর্বনির্ধারিত:
                        ggml_compute_forward_mul(params, op);
                        ফেরত সত্য;
                }
            ক্ষেত্র GGML_OP_DIV:
                নির্বাচন (op->src[0]->type) {
                    ক্ষেত্র GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_দ্বিমিক<GGML_OP_DIV, দশমিক>(params, op);
                        ফেরত সত্য;
                    ক্ষেত্র GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_দ্বিমিক<GGML_OP_DIV, _Float16>(params, op);
                        ফেরত সত্য;
                    পূর্বনির্ধারিত:
                        ggml_compute_forward_div(params, op);
                        ফেরত সত্য;
                }
            ক্ষেত্র GGML_OP_FLASH_ATTN_EXT:
                forward_flash_attn_ext_f16(params, op);
                ফেরত সত্য;
            ক্ষেত্র GGML_OP_CONT:
                {
                    ধ্রুবক ggml_tensor * src0 = op->src[0];
                    যদি (op->type == src0->type && op->nb[0] != src0->nb[0] && op->nb[0] == src0->nb[1] &&
                        op->ne[3] * op->ne[2] * op->nb[2] == src0->ne[3] * src0->ne[2] * src0->nb[2]) {
                        spacemit_kernels::rvv::forward_cont_with_permute(params, op);
                    } অন্যথা {
                        ggml_compute_forward_cont(params, op);
                    }
                    ফেরত সত্য;
                }
            ক্ষেত্র GGML_OP_CPY:
                {
                    ধ্রুবক ggml_tensor * src0 = op->src[0];
                    যদি (op->type == src0->type && op->nb[0] == src0->nb[1] && src0->nb[0] != src0->nb[1] &&
                        ggml_nelements(src0) == ggml_nelements(op)) {
                        spacemit_kernels::rvv::forward_cpy_with_permute(params, op);
                    } অন্যথা {
                        ggml_compute_forward_cpy(params, op);
                    }
                    ফেরত সত্য;
                }
            ক্ষেত্র GGML_OP_REPEAT:
                {
                    ধ্রুবক যৌক্তিক rows_equal         = ggml_nrows(op->src[0]) == ggml_nrows(op);
                    ধ্রুবক যৌক্তিক broadcast_or_equal = op->src[0]->ne[0] == 1 || op->src[0]->ne[0] == op->ne[0];

                    যদি (rows_equal && broadcast_or_equal) {
                        নির্বাচন (op->src[0]->type) {
                            ক্ষেত্র GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_repeat_nrows<int32_t>(params, op);
                                ফেরত সত্য;
                            ক্ষেত্র GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_repeat_nrows<int16_t>(params, op);
                                ফেরত সত্য;
                            পূর্বনির্ধারিত:
                                বিরতি;
                        }
                    }

                    যদি (op->src[0]->ne[1] == 1 && op->src[0]->ne[0] == op->ne[0]) {
                        নির্বাচন (op->src[0]->type) {
                            ক্ষেত্র GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_repeat_dim1<int32_t>(params, op);
                                ফেরত সত্য;
                            ক্ষেত্র GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_repeat_dim1<int16_t>(params, op);
                                ফেরত সত্য;
                            পূর্বনির্ধারিত:
                                বিরতি;
                        }
                    }

                    ggml_compute_forward_repeat(params, op);
                }
                ফেরত সত্য;
            ক্ষেত্র GGML_OP_যোগ_ROWS:
                {
                    যদি (op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32) {
                        spacemit_kernels::rvv::forward_যোগ_rows<দশমিক>(params, op);
                    } অন্যথা {
                        ggml_compute_forward_যোগ_rows(params, op);
                    }
                }
                ফেরত সত্য;
            ক্ষেত্র GGML_OP_GET_ROWS:
                {
                    যদি (op->src[0]->type == op->type) {
                        নির্বাচন (op->src[0]->type) {
                            ক্ষেত্র GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_get_rows<int32_t>(params, op);
                                ফেরত সত্য;
                            ক্ষেত্র GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_get_rows<int16_t>(params, op);
                                ফেরত সত্য;
                            পূর্বনির্ধারিত:
                                বিরতি;
                        }
                    }

                    ggml_compute_forward_get_rows(params, op);
                }
                ফেরত সত্য;
            ক্ষেত্র GGML_OP_CONCAT:
                {
                    ধ্রুবক int32_t dim = ggml_get_op_params_i32(op, 0);
                    যদি (dim == 0 && op->type == op->src[0]->type) {
                        নির্বাচন (op->src[0]->type) {
                            ক্ষেত্র GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_concat<int32_t>(params, op);
                                ফেরত সত্য;
                            ক্ষেত্র GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_concat<int16_t>(params, op);
                                ফেরত সত্য;
                            পূর্বনির্ধারিত:
                                বিরতি;
                        }
                    }

                    ggml_compute_forward_concat(params, op);
                }
                ফেরত সত্য;
            // TODO For GGML_OP_GATED_DELTA_NET
            // case GGML_OP_GATED_DELTA_NET:
            //     return true;
            পূর্বনির্ধারিত:
                বিরতি;
        }
        ফেরত মিথ্যা;
    }

    শূন্য forward_flash_attn_ext_f16(ধ্রুবক ggml_compute_params * params, ggml_tensor * dst) {
        ধ্রুবক ggml_tensor * q = dst->src[0];
        ধ্রুবক ggml_tensor * k = dst->src[1];
        ধ্রুবক ggml_tensor * v = dst->src[2];

        GGML_TENSOR_LOCALS(int64_t, neq, q, ne)
        GGML_TENSOR_LOCALS(size_t, nbq, q, nb)
        GGML_TENSOR_LOCALS(int64_t, nek, k, ne)
        GGML_TENSOR_LOCALS(size_t, nbk, k, nb)
        GGML_TENSOR_LOCALS(int64_t, nev, v, ne)
        GGML_TENSOR_LOCALS(size_t, nbv, v, nb)
        GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
        GGML_TENSOR_LOCALS(size_t, nb, dst, nb)

        ধ্রুবক int64_t DK = nek0;
        ধ্রুবক int64_t DV = nev0;

        ধ্রুবক যৌক্তিক supported_prec  = (dst->op_params[3] == GGML_PREC_F32 || dst->op_params[3] == GGML_PREC_DEFAULT);
        ধ্রুবক যৌক্তিক supported_types = (q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16);
        ধ্রুবক যৌক্তিক supported_shape = (DK > 0 && DK <= 128 && DV > 0 && DV <= 128);
        ধ্রুবক যৌক্তিক supported_vlen  = (__riscv_vlenb() == 128);

        যদি (!(supported_prec && supported_types && supported_shape && supported_vlen)) {
            ggml_compute_forward_flash_attn_ext(params, dst);
            ফেরত;
        }

        // total rows in q
        ধ্রুবক int64_t nr = neq1 * neq2 * neq3;

        // rows per thread
        ধ্রুবক পূর্ণসংখ্যা ith = params->ith;
        ধ্রুবক পূর্ণসংখ্যা nth = params->nth;

        স্থির constexpr int64_t Q_TILE_SZ = ggml_fa_tile_config::Q;
        ধ্রুবক যৌক্তিক               use_tiled = !params->use_ref && (neq1 >= Q_TILE_SZ);

        // 4x chunks per thread
        // int     nth_scaled = nth * 4;
        // int64_t chunk_size = (nr + nth_scaled - 1) / nth_scaled;
        // int64_t nchunk     = (nr + chunk_size - 1) / chunk_size;

        // if (nth == 1 || nchunk < nth) {
        //     nchunk = nth;
        // }

        int64_t nchunk = nth;

        যদি (ith == 0) {
            // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
            ggml_threadpool_chunk_set(params->threadpool, nth);
        }

        ggml_barrier(params->threadpool);

        // The number of elements in each chunk
        ধ্রুবক int64_t dr = (nr + nchunk - 1) / nchunk;

        // The first chunk comes from our thread_id, the rest will get auto-assigned.
        পূর্ণসংখ্যা current_chunk = ith;

        যখন (current_chunk < nchunk) {
            ধ্রুবক int64_t ir0 = dr * current_chunk;
            ধ্রুবক int64_t ir1 = MIN(ir0 + dr, nr);

            যদি (use_tiled) {
                spacemit_kernels::rvv::forward_flash_attn_ext_f16_tiled_vlen1024_vf16(
                    params, dst, ir0, ir1, ggml::cpu::riscv64_spacemit::tls_context.tcm_বাফার,
                    ggml::cpu::riscv64_spacemit::tls_context.tcm_বাফার_আকার);
            } অন্যথা {
                spacemit_kernels::rvv::forward_flash_attn_ext_f16_one_chunk_vlen1024_vf16(
                    params, dst, ir0, ir1, ggml::cpu::riscv64_spacemit::tls_context.tcm_বাফার,
                    ggml::cpu::riscv64_spacemit::tls_context.tcm_বাফার_আকার);
            }

            current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
        }
    }

    পূর্ণসংখ্যা repack(ggml_tensor * t, ধ্রুবক শূন্য * তথ্য, size_t তথ্য_আকার) override {
        memcpy(t->তথ্য, তথ্য, তথ্য_আকার);
        ফেরত 0;
    }
};

// Impl By IME1
স্থির ধ্রুবক tensor_traits<block_q4_0, 32, 16>  q4_0_16x32_q8_0;
স্থির ধ্রুবক tensor_traits<block_q4_1, 32, 16>  q4_1_16x32_q8_0;
স্থির ধ্রুবক tensor_traits<block_q4_K, 32, 16>  q4_k_16x32_q8_0;
স্থির ধ্রুবক tensor_traits<block_q8_0, 32, 16>  q8_0_16x32_q8_0;
// Impl By IME2
স্থির ধ্রুবক tensor_traits<block_q2_K, 256, 32> q2_k_32x256_q8_0;
স্থির ধ্রুবক tensor_traits<block_q3_K, 256, 32> q3_k_32x256_q8_0;
স্থির ধ্রুবক tensor_traits<block_q4_0, 32, 32>  q4_0_32x32_q8_0;
স্থির ধ্রুবক tensor_traits<block_q4_1, 32, 32>  q4_1_32x32_q8_0;
স্থির ধ্রুবক tensor_traits<block_q4_0, 256, 32> q4_0_32x256_q8_0;
স্থির ধ্রুবক tensor_traits<block_q4_1, 256, 32> q4_1_32x256_q8_0;
স্থির ধ্রুবক tensor_traits<block_q4_K, 32, 32>  q4_k_32x32_q8_0;
স্থির ধ্রুবক tensor_traits<block_q6_K, 32, 32>  q6_k_32x32_q8_0;
স্থির ধ্রুবক tensor_traits<block_q8_0, 32, 32>  q8_0_32x32_q8_0;
স্থির ধ্রুবক tensor_traits<block_mxfp4, 32, 32> mxfp4_32x32_q8_0;
স্থির ধ্রুবক tensor_traits<block_q5_K, 32, 32>  q5_k_32x32_q8_0;
স্থির ধ্রুবক tensor_traits<block_q5_1, 32, 32>  q5_1_32x32_q8_0;
স্থির ধ্রুবক tensor_traits<block_q5_0, 32, 32>  q5_0_32x32_q8_0;
// Impl By RVV
স্থির ধ্রুবক tensor_traits_common               rvv_impl;

}  // namespace ggml::cpu::riscv64_spacemit

স্থির ধ্রুবক ggml::cpu::tensor_traits * ggml_riscv64_spacemit_get_optimal_repack_type(ধ্রুবক ggml_tensor * cur) {
    নির্বাচন (cur->type) {
        ক্ষেত্র GGML_TYPE_Q2_K:
            {
#যদি defined(RISCV64_SPACEMIT_IME2)
                যদি (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    ফেরত &ggml::cpu::riscv64_spacemit::q2_k_32x256_q8_0;
                }
#endif
            }
            বিরতি;
        ক্ষেত্র GGML_TYPE_Q3_K:
            {
#যদি defined(RISCV64_SPACEMIT_IME2)
                যদি (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    ফেরত &ggml::cpu::riscv64_spacemit::q3_k_32x256_q8_0;
                }
#endif
            }
            বিরতি;
        ক্ষেত্র GGML_TYPE_Q4_0:
            {
#যদি defined(RISCV64_SPACEMIT_IME2)
                যদি (cur->ne[1] % 32 == 0 && cur->ne[0] % 256 == 0 &&
                    (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    ফেরত &ggml::cpu::riscv64_spacemit::q4_0_32x256_q8_0;
                }

                যদি (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    ফেরত &ggml::cpu::riscv64_spacemit::q4_0_32x32_q8_0;
                }
#endif

#যদি defined(RISCV64_SPACEMIT_IME1)
                যদি (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    ফেরত &ggml::cpu::riscv64_spacemit::q4_0_16x32_q8_0;
                }
#endif
            }
            বিরতি;
        ক্ষেত্র GGML_TYPE_Q4_1:
            {
#যদি defined(RISCV64_SPACEMIT_IME2)
                // TODO
                // if (cur->ne[1] % 32 == 0 && cur->ne[0] % 256 == 0 &&
                //     (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                //     return &ggml::cpu::riscv64_spacemit::q4_1_32x256_q8_0;
                // }

                যদি (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    ফেরত &ggml::cpu::riscv64_spacemit::q4_1_32x32_q8_0;
                }
#endif

#যদি defined(RISCV64_SPACEMIT_IME1)
                যদি (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    ফেরত &ggml::cpu::riscv64_spacemit::q4_1_16x32_q8_0;
                }
#endif
            }
            বিরতি;
        ক্ষেত্র GGML_TYPE_Q4_K:
            {
#যদি defined(RISCV64_SPACEMIT_IME2)
                যদি (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    ফেরত &ggml::cpu::riscv64_spacemit::q4_k_32x32_q8_0;
                }
#endif

#যদি defined(RISCV64_SPACEMIT_IME1)
                যদি (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    ফেরত &ggml::cpu::riscv64_spacemit::q4_k_16x32_q8_0;
                }
#endif
            }
            বিরতি;
        ক্ষেত্র GGML_TYPE_Q6_K:
            {
#যদি defined(RISCV64_SPACEMIT_IME2)
                যদি ((ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    ফেরত &ggml::cpu::riscv64_spacemit::q6_k_32x32_q8_0;
                }
#endif
            }
            বিরতি;
        ক্ষেত্র GGML_TYPE_Q8_0:
            {
#যদি defined(RISCV64_SPACEMIT_IME2)
                যদি ((ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    ফেরত &ggml::cpu::riscv64_spacemit::q8_0_32x32_q8_0;
                }
#endif

#যদি defined(RISCV64_SPACEMIT_IME1)
                যদি (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    ফেরত &ggml::cpu::riscv64_spacemit::q8_0_16x32_q8_0;
                }
#endif
            }
            বিরতি;
        ক্ষেত্র GGML_TYPE_MXFP4:
            {
#যদি defined(RISCV64_SPACEMIT_IME2)
                // TODO
                // if (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                //     return &ggml::cpu::riscv64_spacemit::mxfp4_32x32_q8_0;
                // }
#endif
            }
            বিরতি;
        ক্ষেত্র GGML_TYPE_Q5_K:
            {
#যদি defined(RISCV64_SPACEMIT_IME2)
                যদি (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    ফেরত &ggml::cpu::riscv64_spacemit::q5_k_32x32_q8_0;
                }
#endif
            }
            বিরতি;
        ক্ষেত্র GGML_TYPE_Q5_1:
            {
#যদি defined(RISCV64_SPACEMIT_IME2)
                যদি (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    ফেরত &ggml::cpu::riscv64_spacemit::q5_1_32x32_q8_0;
                }
#endif
            }
            বিরতি;
        ক্ষেত্র GGML_TYPE_Q5_0:
            {
#যদি defined(RISCV64_SPACEMIT_IME2)
                যদি (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    ফেরত &ggml::cpu::riscv64_spacemit::q5_0_32x32_q8_0;
                }
#endif
            }
            বিরতি;
        পূর্বনির্ধারিত:
            বিরতি;
    }

    ফেরত nullptr;
}

স্থির গণনা ggml_status ggml_backend_riscv64_spacemit_বাফার_init_tensor(ggml_backend_বাফার_t বাফার,
                                                                         ggml_tensor *         tensor) {
    tensor->extra =
        (শূন্য *) const_cast<ggml::cpu::tensor_traits *>(ggml_riscv64_spacemit_get_optimal_repack_type(tensor));

    GGML_UNUSED(বাফার);

    ফেরত GGML_STATUS_SUCCESS;
}

স্থির শূন্য ggml_backend_riscv64_spacemit_বাফার_free_বাফার(ggml_backend_বাফার_t বাফার) {
    GGML_ASSERT(বাফার);

    শূন্য * base = বাফার->context;
    যদি (base == nullptr) {
        ফেরত;
    }

    ggml::cpu::riscv64_spacemit::spine_mem_pool_free(base);
}

স্থির শূন্য * ggml_backend_riscv64_spacemit_বাফার_get_base(ggml_backend_বাফার_t বাফার) {
    GGML_ASSERT(বাফার);

    শূন্য * base = বাফার->context;
    GGML_ASSERT(base != nullptr);
    ফেরত base;
}

স্থির শূন্য ggml_backend_riscv64_spacemit_বাফার_memset_tensor(ggml_backend_বাফার_t বাফার,
                                                               ggml_tensor *         tensor,
                                                               uint8_t               value,
                                                               size_t                offset,
                                                               size_t                size) {
    GGML_ASSERT(tensor);
    memset((অক্ষর *) tensor->তথ্য + offset, value, size);

    GGML_UNUSED(বাফার);
}

স্থির শূন্য ggml_backend_riscv64_spacemit_বাফার_clear(ggml_backend_বাফার_t বাফার, uint8_t value) {
    GGML_ASSERT(বাফার);

    শূন্য * base = বাফার->context;
    GGML_ASSERT(base != nullptr);
    memset(base, value, বাফার->size);
}

স্থির শূন্য ggml_backend_riscv64_spacemit_বাফার_set_tensor(ggml_backend_বাফার_t বাফার,
                                                            ggml_tensor *         tensor,
                                                            ধ্রুবক শূন্য *          তথ্য,
                                                            size_t                offset,
                                                            size_t                size) {
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    auto tensor_traits = (ggml::cpu::riscv64_spacemit::tensor_traits_base *) tensor->extra;
    যদি (tensor_traits) {
        auto OK = tensor_traits->repack(tensor, তথ্য, size);
        GGML_ASSERT(OK == 0);
    }

    GGML_UNUSED(বাফার);
}

স্থির ধ্রুবক ggml_backend_বাফার_i ggml_backend_riscv64_spacemit_বাফার_i = {
    /* .free_buffer     = */ ggml_backend_riscv64_spacemit_বাফার_free_বাফার,
    /* .get_base        = */ ggml_backend_riscv64_spacemit_বাফার_get_base,
    /* .init_tensor     = */ ggml_backend_riscv64_spacemit_বাফার_init_tensor,
    /* .memset_tensor   = */ ggml_backend_riscv64_spacemit_বাফার_memset_tensor,
    /* .set_tensor      = */ ggml_backend_riscv64_spacemit_বাফার_set_tensor,
    /* .get_tensor      = */ nullptr,
    /* .set_tensor_2d   = */ nullptr,
    /* .get_tensor_2d   = */ nullptr,
    /* .cpy_tensor      = */ nullptr,
    /* .clear           = */ ggml_backend_riscv64_spacemit_বাফার_clear,
    /* .reset           = */ nullptr,
};

স্থির ধ্রুবক অক্ষর * ggml_backend_cpu_riscv64_spacemit_বাফার_type_get_name(ggml_backend_বাফার_type_t buft) {
    ফেরত "CPU_RISCV64_SPACEMIT";

    GGML_UNUSED(buft);
}

স্থির ggml_backend_বাফার_t ggml_backend_cpu_riscv64_spacemit_বাফার_type_alloc_বাফার(ggml_backend_বাফার_type_t buft,
                                                                                        size_t size) {
    শূন্য * base = ggml::cpu::riscv64_spacemit::spine_mem_pool_alloc(size, 64);
    যদি (base == nullptr) {
        ফেরত nullptr;
    }

    ফেরত ggml_backend_বাফার_init(buft, ggml_backend_riscv64_spacemit_বাফার_i, base, size);
}

স্থির size_t ggml_backend_cpu_riscv64_spacemit_বাফার_type_get_alignment(ggml_backend_বাফার_type_t buft) {
    ফেরত 64;

    GGML_UNUSED(buft);
}

স্থির size_t ggml_backend_cpu_riscv64_spacemit_nbytes(ggml_backend_বাফার_type_t buft, ধ্রুবক ggml_tensor * tensor) {
    জন্য (পূর্ণসংখ্যা i = 0; i < GGML_MAX_DIMS; ++i) {
        যদি (tensor->ne[i] <= 0) {
            ফেরত 0;
        }
    }

    GGML_UNUSED(buft);

    ধ্রুবক auto plain_nbytes = [&]() {
        size_t যোগ = ggml_type_আকার(tensor->type);
        জন্য (পূর্ণসংখ্যা i = 0; i < GGML_MAX_DIMS; ++i) {
            যোগ += (tensor->ne[i] - 1) * tensor->nb[i];
        }
        ফেরত যোগ;
    };

    ধ্রুবক size_t blck_আকার = ggml_blck_আকার(tensor->type);
    যদি (blck_আকার == 1) {
        ফেরত plain_nbytes();
    }

    ধ্রুবক size_t row_nbytes = tensor->ne[0] * tensor->nb[0] / blck_আকার;

    ধ্রুবক auto add_strided_nbytes = [&](size_t যোগ, size_t src_block_আকার, size_t dst_block_আকার) {
        জন্য (পূর্ণসংখ্যা i = 1; i < GGML_MAX_DIMS; ++i) {
            যোগ += (tensor->ne[i] - 1) * (tensor->nb[i] / src_block_আকার) * dst_block_আকার;
        }
        ফেরত যোগ;
    };

    ধ্রুবক auto remap_block_nbytes = [&](size_t src_block_আকার, size_t dst_block_আকার, int64_t padded_rows = 0) {
        GGML_ASSERT(row_nbytes % src_block_আকার == 0);

        size_t যোগ =
            add_strided_nbytes((row_nbytes / src_block_আকার) * dst_block_আকার, src_block_আকার, dst_block_আকার);

        যদি (padded_rows > 0 && tensor->ne[1] % padded_rows != 0) {
            যোগ += (padded_rows - tensor->ne[1] % padded_rows) * (tensor->nb[1] / src_block_আকার) * dst_block_আকার;
        }

        ফেরত যোগ;
    };

    size_t nbytes = row_nbytes;
    নির্বাচন (tensor->type) {
        ক্ষেত্র GGML_TYPE_Q4_K:
            nbytes = remap_block_nbytes(আকার(block_q4_K), আকার(block_q4_1) * 8);
            বিরতি;
        ক্ষেত্র GGML_TYPE_Q6_K:
            nbytes = remap_block_nbytes(আকার(block_q6_K), আকার(block_q8_0) * 8, 32);
            বিরতি;
        ক্ষেত্র GGML_TYPE_Q8_0:
            nbytes = remap_block_nbytes(আকার(block_q8_0), আকার(block_q8_0), 32);
            বিরতি;
        ক্ষেত্র GGML_TYPE_Q2_K:
            nbytes = remap_block_nbytes(আকার(block_q2_K), আকার(spacemit_kernels::nrow_block_q2_k<1>));
            বিরতি;
        ক্ষেত্র GGML_TYPE_Q3_K:
            nbytes = remap_block_nbytes(আকার(block_q3_K), আকার(spacemit_kernels::nrow_block_q3_k<1>));
            বিরতি;
        ক্ষেত্র GGML_TYPE_MXFP4:
            nbytes = remap_block_nbytes(আকার(block_mxfp4), আকার(spacemit_kernels::nrow_block_mxfp4<1>));
            বিরতি;
        ক্ষেত্র GGML_TYPE_Q5_K:
            nbytes = remap_block_nbytes(আকার(block_q5_K), আকার(spacemit_kernels::nrow_block_q5_1<1>) * 8);
            বিরতি;
        ক্ষেত্র GGML_TYPE_Q5_1:
            nbytes = remap_block_nbytes(আকার(block_q5_1), আকার(spacemit_kernels::nrow_block_q5_1<1>));
            বিরতি;
        ক্ষেত্র GGML_TYPE_Q5_0:
            nbytes = remap_block_nbytes(আকার(block_q5_0), আকার(spacemit_kernels::nrow_block_q5_0<1>));
            বিরতি;
        পূর্বনির্ধারিত:
            nbytes = add_strided_nbytes(row_nbytes, 1, 1);
            বিরতি;
    }

    ফেরত nbytes;
}

নামাকাশ ggml::cpu::riscv64_spacemit {

শ্রেণী extra_বাফার_type : ggml::cpu::extra_বাফার_type {
    যৌক্তিক supports_op(ggml_backend_dev_t, ধ্রুবক ggml_tensor * op) override {
        নির্বাচন (op->op) {
            ক্ষেত্র GGML_OP_MUL_MAT:
                যদি (op->src[0]->বাফার && (ggml_n_dims(op->src[0]) == 2) &&
                    op->src[0]->বাফার->buft == ggml_backend_cpu_riscv64_spacemit_বাফার_type() &&
                    ggml_riscv64_spacemit_get_optimal_repack_type(op->src[0])) {
                    যদি (op->src[1]->বাফার && !ggml_backend_buft_is_host(op->src[1]->বাফার->buft)) {
                        ফেরত মিথ্যা;
                    }
                    যদি (op->src[1]->type == GGML_TYPE_F32) {
                        ফেরত সত্য;
                    }
                }
                বিরতি;
            ক্ষেত্র GGML_OP_MUL_MAT_ID:
                যদি (op->src[0]->বাফার && (ggml_n_dims(op->src[0]) == 3) &&
                    op->src[0]->বাফার->buft == ggml_backend_cpu_riscv64_spacemit_বাফার_type() &&
                    ggml_riscv64_spacemit_get_optimal_repack_type(op->src[0])) {
                    যদি (op->src[1]->বাফার && !ggml_backend_buft_is_host(op->src[1]->বাফার->buft)) {
                        ফেরত মিথ্যা;
                    }
                    যদি (op->src[1]->type == GGML_TYPE_F32) {
                        ফেরত সত্য;
                    }
                }
                বিরতি;
            পূর্বনির্ধারিত:
                // GGML_ABORT("fatal error");
                বিরতি;
        }
        ফেরত মিথ্যা;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(ধ্রুবক ggml_tensor * op) override {
        নির্বাচন (op->op) {
            ক্ষেত্র GGML_OP_MUL_MAT:
            ক্ষেত্র GGML_OP_MUL_MAT_ID:
                যদি (op->src[0]->বাফার && op->src[0]->বাফার->buft == ggml_backend_cpu_riscv64_spacemit_বাফার_type()) {
                    ফেরত (ggml::cpu::tensor_traits *) op->src[0]->extra;
                }
                বিরতি;
            ক্ষেত্র GGML_OP_NORM:
            ক্ষেত্র GGML_OP_RMS_NORM:
            ক্ষেত্র GGML_OP_ADD:
            ক্ষেত্র GGML_OP_SUB:
            ক্ষেত্র GGML_OP_MUL:
            ক্ষেত্র GGML_OP_DIV:
            ক্ষেত্র GGML_OP_FLASH_ATTN_EXT:
            ক্ষেত্র GGML_OP_CONT:
            ক্ষেত্র GGML_OP_CPY:
            ক্ষেত্র GGML_OP_REPEAT:
            ক্ষেত্র GGML_OP_যোগ_ROWS:
            ক্ষেত্র GGML_OP_GET_ROWS:
            ক্ষেত্র GGML_OP_CONCAT:
                // case GGML_OP_GATED_DELTA_NET:
                ফেরত (ggml::cpu::tensor_traits *) (&ggml::cpu::riscv64_spacemit::rvv_impl);
            পূর্বনির্ধারিত:
                // GGML_ABORT("fatal error");
                বিরতি;
        }

        ফেরত nullptr;
    }
};

}  // namespace ggml::cpu::riscv64_spacemit

ggml_backend_বাফার_type_t ggml_backend_cpu_riscv64_spacemit_বাফার_type(শূন্য) {
    স্থির ggml_backend_বাফার_type ggml_backend_cpu_বাফার_type_riscv64_spacemit = {
  /* .iface    = */
        {
         /* .get_name           = */ ggml_backend_cpu_riscv64_spacemit_বাফার_type_get_name,
         /* .alloc_buffer       = */ ggml_backend_cpu_riscv64_spacemit_বাফার_type_alloc_বাফার,
         /* .alloc_buffer_n     = */ ফাঁকা,
         /* .get_alignment      = */ ggml_backend_cpu_riscv64_spacemit_বাফার_type_get_alignment,
         /* .get_max_size       = */ nullptr,
         /* .get_alloc_size     = */ ggml_backend_cpu_riscv64_spacemit_nbytes,
         /* .get_alloc_size_n   = */ ফাঁকা,
         /* .is_host            = */ nullptr,
         },
 /* .device  = */
        ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
 /* .context = */
        নতুন ggml::cpu::riscv64_spacemit::extra_বাফার_type(),
    };

    ফেরত &ggml_backend_cpu_বাফার_type_riscv64_spacemit;
}

বাহ্যিক "C" {
স্থির পূর্ণসংখ্যা bind_ai_thread() {
    পূর্ণসংখ্যা  fd, bytes;
    অক্ষর str[32];

    fd = open("/proc/set_ai_thread", O_WRONLY);
    যদি (fd < 0) {
        GGML_LOG_ERROR("try open /proc/set_ai_thread failed\n");
        ফেরত -1;
    }

    snprintf(str, 16, "%d", 0);
    bytes = লিখ(fd, str, strlen(str));
    যদি (bytes < 0) {
        GGML_LOG_ERROR("try write /proc/set_ai_thread failed\n");
        close(fd);
        ফেরত -1;
    }

    close(fd);
    ফেরত 0;
}

শূন্য ggml_backend_cpu_riscv64_spacemit_set_numa_thread_affinity(পূর্ণসংখ্যা thread_n) {
    পূর্ণসংখ্যা cpu_id = sched_getcpu();
    যদি (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2 &&
        !((1 << cpu_id) & ggml::cpu::riscv64_spacemit::global_spine_env_info.cpu_mask)) {
        GGML_PRINT_DEBUG("bind_ai_thread for thread %d, pid %d\n", thread_n, getpid());
        bind_ai_thread();
    }

    যদি (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_tcm &&
        ggml::cpu::riscv64_spacemit::tls_context.cpu_id == -1) {
        CPU_ZERO(&(ggml::cpu::riscv64_spacemit::tls_context.cpuset));
        pthread_t    main_thread     = pthread_self();
        ধ্রুবক auto & perfer_core_ids = ggml::cpu::riscv64_spacemit::global_spine_env_info.perfer_core_ids;
        যদি (thread_n < 0 || static_cast<size_t>(thread_n) >= perfer_core_ids.size()) {
            GGML_ABORT("thread_n %d exceeds perfer_core_ids size %zu\n", thread_n, perfer_core_ids.size());
        }
        auto perfer_cpu_id = perfer_core_ids[static_cast<size_t>(thread_n)];
        CPU_SET(perfer_cpu_id, &(ggml::cpu::riscv64_spacemit::tls_context.cpuset));
        পূর্ণসংখ্যা s =
            pthread_setaffinity_np(main_thread, আকার(cpu_set_t), &(ggml::cpu::riscv64_spacemit::tls_context.cpuset));
        যদি (s != 0) {
            GGML_ABORT("set thread affinity error for thread_n %d, cpu_id %d\n", thread_n, perfer_cpu_id);
        }

        পূর্ণসংখ্যা ai_cpu_id = perfer_cpu_id - ggml::cpu::riscv64_spacemit::global_spine_env_info.aicpu_id_offset;
        ggml::cpu::riscv64_spacemit::tls_context.cpu_id = ai_cpu_id;
        ggml::cpu::riscv64_spacemit::tls_context.tcm_বাফার =
            ggml::cpu::riscv64_spacemit::spine_mem_pool_tcm_mem_get(ai_cpu_id);
        ggml::cpu::riscv64_spacemit::tls_context.tcm_বাফার_আকার =
            ggml::cpu::riscv64_spacemit::global_spine_env_info.tcm_blk_আকার;
    }

    যদি (ggml::cpu::riscv64_spacemit::tls_context.tcm_বাফার != nullptr) {
        শূন্য * rt =
            ggml::cpu::riscv64_spacemit::spine_mem_pool_tcm_mem_wait(ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        যদি (rt == nullptr) {
            GGML_ABORT("wait tcm buffer failed for cpu_id: %d", ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        }
    }
}

শূন্য ggml_backend_cpu_riscv64_spacemit_clear_numa_thread_affinity_threaded(পূর্ণসংখ্যা thread_n) {
    যদি (ggml::cpu::riscv64_spacemit::tls_context.tcm_বাফার != nullptr) {
        auto rt = ggml::cpu::riscv64_spacemit::spine_mem_pool_tcm_mem_release(
            ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        যদি (rt != 0) {
            GGML_ABORT("release tcm buffer failed for cpu_id: %d", ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        }
    }
}
}
