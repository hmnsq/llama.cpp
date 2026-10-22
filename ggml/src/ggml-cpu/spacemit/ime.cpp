#define GGML_COMMON_IMPL_CPP
#define GGML_COMMON_DECL_CPP

#অন্তর্ভাৱ "ime.h"

#অন্তর্ভাৱ "binary-ops.h"
#অন্তর্ভাৱ "common.h"
#অন্তর্ভাৱ "ggml-backend-impl.h"
#অন্তর্ভাৱ "ggml-common.h"
#অন্তর্ভাৱ "ggml-cpu.h"
#অন্তর্ভাৱ "ime_env.h"
#অন্তর্ভাৱ "ime_kernels.h"
#অন্তর্ভাৱ "ops.h"
#অন্তর্ভাৱ "repack.h"
#অন্তর্ভাৱ "rvv_kernels.h"
#অন্তর্ভাৱ "spine_mem_pool.h"
#অন্তর্ভাৱ "traits.h"
#অন্তর্ভাৱ "vec.h"

#অন্তর্ভাৱ <fcntl.h>
#অন্তর্ভাৱ <sys/mman.h>
#অন্তর্ভাৱ <unistd.h>

#অন্তর্ভাৱ <algorithm>
#অন্তর্ভাৱ <atomic>
#অন্তর্ভাৱ <cassert>
#অন্তর্ভাৱ <cerrno>
#অন্তর্ভাৱ <cmath>
#অন্তর্ভাৱ <cstdio>  // for GGML_ASSERT
#অন্তর্ভাৱ <stdexcept>
#অন্তর্ভাৱ <thread>
// clang-format off
#যদি defined(__riscv)

#যদি !defined(__riscv_v) || !defined(__riscv_v_intrinsic)
#error "riscv v extension or v_intrinsic not enabled"
#অন্যথা
#অন্তর্ভাৱ <riscv_vector.h>
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

বাহ্য "C" {
বাহ্য শূন্য ggml_threadpool_chunk_set(সংরচনা ggml_threadpool * tp, পূর্ণাঙ্ক value);
বাহ্য পূর্ণাঙ্ক  ggml_threadpool_chunk_add(সংরচনা ggml_threadpool * tp, পূর্ণাঙ্ক value);
}

নামাকাশ ggml::cpu::riscv64_spacemit {

সংরচনা TLSContext {
    পূর্ণাঙ্ক       cpu_id{ -1 };
    cpu_set_t cpuset;
    শূন্য *    tcm_buffer{ nullptr };
    size_t    tcm_buffer_size{ 0 };
};

thread_local TLSContext tls_context;

template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS> constexpr size_t get_repacked_block_type_size() {
    যদি constexpr (std::is_same_v<BLOC_TYPE, block_q6_K> || std::is_same_v<BLOC_TYPE, block_q8_0>) {
        ঘূৰাই পরিমাণ(block_q8_0);
    } অন্যথা যদি constexpr (std::is_same_v<BLOC_TYPE, block_q4_0>) {
        ঘূৰাই পরিমাণ(block_q4_0) * INTER_SIZE / QK4_0;
    } অন্যথা যদি constexpr (std::is_same_v<BLOC_TYPE, block_q4_1> || std::is_same_v<BLOC_TYPE, block_q4_K>) {
        ঘূৰাই (পরিমাণ(block_q4_0) + পরিমাণ(uint8_t)) * INTER_SIZE / QK4_1;
    } অন্যথা যদি constexpr (std::is_same_v<BLOC_TYPE, block_q2_K>) {
        ঘূৰাই পরিমাণ(spacemit_kernels::nrow_block_q2_k<1>);
    } অন্যথা যদি constexpr (std::is_same_v<BLOC_TYPE, block_q3_K>) {
        ঘূৰাই পরিমাণ(spacemit_kernels::nrow_block_q3_k<1>);
    } অন্যথা যদি constexpr (std::is_same_v<BLOC_TYPE, block_mxfp4>) {
        ঘূৰাই পরিমাণ(spacemit_kernels::nrow_block_mxfp4<1>);
    } অন্যথা যদি constexpr (std::is_same_v<BLOC_TYPE, block_q5_1> || std::is_same_v<BLOC_TYPE, block_q5_K>) {
        ঘূৰাই পরিমাণ(spacemit_kernels::nrow_block_q5_1<1>);
    } অন্যথা যদি constexpr (std::is_same_v<BLOC_TYPE, block_q5_0>) {
        ঘূৰাই পরিমাণ(spacemit_kernels::nrow_block_q5_0<1>);
    } অন্যথা {
        প্রতিজ্ঞা(মিছা);
        ঘূৰাই 0;
    }
}

template <typename BLOC_TYPE> constexpr তার্কিক block_type_has_zp() {
    যদি constexpr (std::is_same_v<BLOC_TYPE, block_q6_K> || std::is_same_v<BLOC_TYPE, block_q8_0> ||
                  std::is_same_v<BLOC_TYPE, block_q3_K> || std::is_same_v<BLOC_TYPE, block_q4_0> ||
                  std::is_same_v<BLOC_TYPE, block_mxfp4> || std::is_same_v<BLOC_TYPE, block_q5_0>) {
        ঘূৰাই মিছা;
    } অন্যথা যদি constexpr (std::is_same_v<BLOC_TYPE, block_q4_1> || std::is_same_v<BLOC_TYPE, block_q4_K> ||
                         std::is_same_v<BLOC_TYPE, block_q2_K> || std::is_same_v<BLOC_TYPE, block_q5_1> ||
                         std::is_same_v<BLOC_TYPE, block_q5_K>) {
        ঘূৰাই সঁচা;
    } অন্যথা {
        প্রতিজ্ঞা(মিছা);
        ঘূৰাই মিছা;
    }
}

শ্ৰেণী tensor_traits_base : সার্ৱজনিক ggml::cpu::tensor_traits {
  সার্ৱজনিক:
    আভাস পূর্ণাঙ্ক repack(ggml_tensor * t, ধ্রুৱ শূন্য * data, size_t data_size) = 0;
};

template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS> শ্ৰেণী tensor_traits : সার্ৱজনিক tensor_traits_base {
    তার্কিক work_size(পূর্ণাঙ্ক /* n_threads */, ধ্রুৱ ggml_tensor * op, size_t & size) override {
        ৱিকল্প (op->op) {
            পক্ষ GGML_OP_MUL_MAT:
                {
                    int64_t src1_nelements = ggml_nelements(op->src[1]);

                    যদি constexpr (std::is_same_v<BLOC_TYPE, block_q2_K> || std::is_same_v<BLOC_TYPE, block_q3_K>) {
                        size =
                            spacemit_kernels::div_round_up(src1_nelements, QK_K) * spacemit_kernels::q8k_blk_size(QK_K);
                    } অন্যথা যদি constexpr (INTER_SIZE == QK4_0) {
                        size = spacemit_kernels::div_round_up(src1_nelements, QK4_0) *
                               spacemit_kernels::q8_blk_size(QK4_0, সঁচা);
                    } অন্যথা যদি constexpr (INTER_SIZE == 256) {
                        size = spacemit_kernels::div_round_up(src1_nelements, 256) *
                               spacemit_kernels::q8_hp_blk_size(256, সঁচা, সঁচা);
                    } অন্যথা {
                        GGML_ABORT("unsupported block type");
                    }

                    size = GGML_PAD(size, পরিমাণ(int64_t));

                    ঘূৰাই সঁচা;
                }
            পক্ষ GGML_OP_MUL_MAT_ID:
                {
                    int64_t src1_nelements = ggml_nelements(op->src[1]);

                    যদি constexpr (std::is_same_v<BLOC_TYPE, block_q2_K> || std::is_same_v<BLOC_TYPE, block_q3_K>) {
                        size =
                            spacemit_kernels::div_round_up(src1_nelements, QK_K) * spacemit_kernels::q8k_blk_size(QK_K);
                    } অন্যথা যদি constexpr (INTER_SIZE == QK4_0) {
                        size = spacemit_kernels::div_round_up(src1_nelements, QK4_0) *
                               spacemit_kernels::q8_blk_size(QK4_0, সঁচা);
                    } অন্যথা যদি constexpr (INTER_SIZE == 256) {
                        size = spacemit_kernels::div_round_up(src1_nelements, 256) *
                               spacemit_kernels::q8_hp_blk_size(256, সঁচা, সঁচা);
                    } অন্যথা {
                        GGML_ABORT("unsupported block type");
                    }

                    size = GGML_PAD(size, পরিমাণ(int64_t));

                    ধ্রুৱ int64_t ne02 = op->src[0]->ne[2];  // n_as, n_expert
                    ধ্রুৱ int64_t ne12 = op->src[1]->ne[2];  // n_tokens

                    ধ্রুৱ size_t sizeof_mmid_row_mapping = পরিমাণ(int64_t);
                    size += sizeof_mmid_row_mapping * ne02 * (ne12 + 1) + (ne02 + 1) * পরিমাণ(int64_t);

                    size = GGML_PAD(size, পরিমাণ(int64_t));

                    ঘূৰাই সঁচা;
                }
            স্ৱভাৱ:
                // GGML_ABORT("fatal error");
                ভঙ্গ;
        }
        ঘূৰাই মিছা;
    }

    তার্কিক compute_forward(ggml_compute_params * params, ggml_tensor * op) override {
        ৱিকল্প (op->op) {
            পক্ষ GGML_OP_MUL_MAT:
                ৱিকল্প (op->src[0]->type) {
                    পক্ষ GGML_TYPE_Q2_K:
                    পক্ষ GGML_TYPE_Q3_K:
                    পক্ষ GGML_TYPE_Q4_0:
                    পক্ষ GGML_TYPE_Q4_1:
                    পক্ষ GGML_TYPE_Q4_K:
                    পক্ষ GGML_TYPE_Q6_K:
                    পক্ষ GGML_TYPE_Q8_0:
                    পক্ষ GGML_TYPE_Q5_0:
                    পক্ষ GGML_TYPE_Q5_1:
                    পক্ষ GGML_TYPE_Q5_K:
                        //case GGML_TYPE_MXFP4:
                        forward_mul_mat(params, op);
                        ঘূৰাই সঁচা;
                    স্ৱভাৱ:
                        // GGML_ABORT("fatal error: unsupported type for src0 in MUL_MAT");
                        ঘূৰাই মিছা;
                }
                ভঙ্গ;
            পক্ষ GGML_OP_MUL_MAT_ID:
                ৱিকল্প (op->src[0]->type) {
                    পক্ষ GGML_TYPE_Q2_K:
                    পক্ষ GGML_TYPE_Q3_K:
                    পক্ষ GGML_TYPE_Q4_0:
                    পক্ষ GGML_TYPE_Q4_1:
                    পক্ষ GGML_TYPE_Q4_K:
                    পক্ষ GGML_TYPE_Q6_K:
                    পক্ষ GGML_TYPE_Q8_0:
                    পক্ষ GGML_TYPE_Q5_0:
                    পক্ষ GGML_TYPE_Q5_1:
                    পক্ষ GGML_TYPE_Q5_K:
                        //case GGML_TYPE_MXFP4:
                        forward_mul_mat_id(params, op);
                        ঘূৰাই সঁচা;
                    স্ৱভাৱ:
                        // GGML_ABORT("fatal error: unsupported type for src0 in MUL_MAT_ID");
                        ঘূৰাই মিছা;
                }
                ভঙ্গ;
            স্ৱভাৱ:
                // GGML_ABORT("fatal error");
                ভঙ্গ;
        }
        ঘূৰাই মিছা;
    }

    শূন্য forward_mul_mat(ggml_compute_params * params, ggml_tensor * op) {
        constexpr size_t a_blk_len = INTER_SIZE;
        constexpr size_t b_blk_len = INTER_SIZE;

        ধ্রুৱ ggml_tensor * src0 = op->src[0];
        ধ্রুৱ ggml_tensor * src1 = op->src[1];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_দ্বিমিক_OP_LOCALS

        পূর্ণাঙ্ক ith = params->ith;
        পূর্ণাঙ্ক nth = params->nth;

        [[maybe_unused]] ধ্রুৱ গণনা ggml_type type = src0->type;

        শূন্য *        w_data  = (শূন্য *) src0->data;
        ধ্রুৱ ভিন্ন * feature = (ধ্রুৱ ভিন্ন *) src1->data;
        ভিন্ন *       output  = (ভিন্ন *) dst->data;

        ধ্রুৱ int64_t gemm_m = ne11 * ne12 * ne13;
        ধ্রুৱ int64_t gemm_k = ne10;
        ধ্রুৱ int64_t gemm_n = ne01;

        spacemit_kernels::quantize_a_row_def       quantize_a_row_i8;
        spacemit_kernels::quantize_a_row_def       quantize_a_4row_i8;
        spacemit_kernels::gemm_kernel_quantize_def gemm_kernel;
        তার্কিক                                       set_kernel_impl = মিছা;

        int64_t block_stride_a = spacemit_kernels::q8_blk_size(a_blk_len);

#যদি defined(RISCV64_SPACEMIT_IME2)
        যদি (!set_kernel_impl && (global_spine_env_info.use_ime2)) {
            quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8;
            quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8;
            block_stride_a     = spacemit_kernels::q8_blk_size(a_blk_len, সঁচা);

            যদি constexpr (std::is_same_v<BLOC_TYPE, block_q6_K> || std::is_same_v<BLOC_TYPE, block_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i8;
                set_kernel_impl = সঁচা;
            } অন্যথা যদি constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> || std::is_same_v<BLOC_TYPE, block_q4_1> ||
                                 std::is_same_v<BLOC_TYPE, block_q4_K>) {
                যদি constexpr (INTER_SIZE == 256) {
                    gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i4_hp;
                    quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8_hp;
                    quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8_hp;
                    block_stride_a     = spacemit_kernels::q8_hp_blk_size(a_blk_len, সঁচা, সঁচা);
                    set_kernel_impl    = সঁচা;
                } অন্যথা {
                    gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i4;
                    quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8;
                    quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8;
                    block_stride_a     = spacemit_kernels::q8_blk_size(a_blk_len, সঁচা);
                    set_kernel_impl    = সঁচা;
                }
            } অন্যথা যদি constexpr (std::is_same_v<BLOC_TYPE, block_q2_K>) {
                quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8k;
                quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8k;
                block_stride_a     = spacemit_kernels::q8k_blk_size(a_blk_len);

                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i2k;
                set_kernel_impl = সঁচা;
            } অন্যথা যদি constexpr (std::is_same_v<BLOC_TYPE, block_q3_K>) {
                quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8k;
                quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8k;
                block_stride_a     = spacemit_kernels::q8k_blk_size(a_blk_len);

                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i3k;
                set_kernel_impl = সঁচা;
            } অন্যথা যদি constexpr (std::is_same_v<BLOC_TYPE, block_mxfp4>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8mxfp4;
                set_kernel_impl = সঁচা;
            } অন্যথা যদি constexpr (std::is_same_v<BLOC_TYPE, block_q5_1> || std::is_same_v<BLOC_TYPE, block_q5_K> ||
                                 std::is_same_v<BLOC_TYPE, block_q5_0>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i5;
                set_kernel_impl = সঁচা;
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
                set_kernel_impl = সঁচা;
            } অন্যথা যদি constexpr (std::is_same_v<BLOC_TYPE, block_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i8;
                set_kernel_impl = সঁচা;
            }
        }
#endif
        যদি (!set_kernel_impl) {
            GGML_ABORT("no kernel implementation found for the block type");
        }

        ধ্রুৱ int64_t a_k_blks = spacemit_kernels::div_round_up(gemm_k, a_blk_len);
        ধ্রুৱ int64_t b_k_blks = spacemit_kernels::div_round_up(gemm_k, b_blk_len);

        ধ্রুৱ int64_t row_stride_a        = a_k_blks * block_stride_a;
        ধ্রুৱ int64_t gemm_workspace_size = GGML_PAD(gemm_m * row_stride_a, alignof(int64_t));

        যদি (ith == 0 && params->wsize < gemm_workspace_size) {
            GGML_ABORT("wsize less than gemm_workspace_size");
        }

        uintptr_t ws_ptr = reinterpret_cast<uintptr_t>(params->wdata);

        শূন্য *        tcm_buffer      = ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer;
        ধ্রুৱ int64_t tcm_buffer_size = ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer_size;

        auto * quant_a_buffer = reinterpret_cast<uint8_t *>(ws_ptr);

        constexpr int64_t row_align = 4;
        ধ্রুৱ int64_t     row_blks  = spacemit_kernels::div_round_up(gemm_m, row_align);

        ধ্রুৱ int64_t row_stride_b      = b_k_blks * get_repacked_block_type_size<BLOC_TYPE, INTER_SIZE, NB_COLS>();
        ধ্রুৱ int64_t per_mb_rows_wsize = row_align * row_stride_a;
        ধ্রুৱ int64_t per_nb_cols_wsize = NB_COLS * row_stride_b;

        ধ্রুৱ int64_t barrier_idx = static_cast<int64_t>(ith / 2);

        GGML_ASSERT(global_spine_env_info.init_barrier != nullptr);
        GGML_ASSERT(barrier_idx < spine_init_barrier_গুনতি);
        spine_barrier_t * cur_barrier = &global_spine_env_info.init_barrier[barrier_idx];

        যদি (gemm_m == 1) {
            পূর্ণাঙ্ক task_per_thread = spacemit_kernels::div_round_up(a_k_blks, nth);
            পূর্ণাঙ্ক a_blk_start     = ith * task_per_thread;
            পূর্ণাঙ্ক a_blk_end       = std::min(a_blk_start + task_per_thread, (পূর্ণাঙ্ক) a_k_blks);
            যদি (a_blk_start < a_blk_end) {
                quantize_a_row_i8(a_blk_len, feature + a_blk_start * a_blk_len, (a_blk_end - a_blk_start) * a_blk_len,
                                  quant_a_buffer + a_blk_start * block_stride_a);
            }
        } অন্যথা {
            পূর্ণাঙ্ক task_per_thread = spacemit_kernels::div_round_up(row_blks, nth);
            পূর্ণাঙ্ক m_row_blk_start = ith * task_per_thread;
            পূর্ণাঙ্ক m_row_blk_end   = std::min(m_row_blk_start + task_per_thread, (পূর্ণাঙ্ক) row_blks);
            বাবে (পূর্ণাঙ্ক m_row_blk = m_row_blk_start; m_row_blk < m_row_blk_end; m_row_blk++) {
                পূর্ণাঙ্ক m_idx             = m_row_blk * row_align;
                পূর্ণাঙ্ক rows_tobe_handled = (gemm_m - m_idx) > row_align ? row_align : (gemm_m - m_idx);

                যদি (rows_tobe_handled == row_align && quantize_a_4row_i8 != nullptr) {
                    ধ্রুৱ ভিন্ন * a_row_ptr       = feature + m_idx * gemm_k;
                    auto *        quant_a_row_ptr = quant_a_buffer + m_idx * row_stride_a;
                    quantize_a_4row_i8(a_blk_len, a_row_ptr, gemm_k, quant_a_row_ptr);
                } অন্যথা {
                    যেতিয়ালৈ (rows_tobe_handled) {
                        ধ্রুৱ ভিন্ন * a_row_ptr       = feature + m_idx * gemm_k;
                        auto *        quant_a_row_ptr = quant_a_buffer + m_idx * row_stride_a;
                        quantize_a_row_i8(a_blk_len, a_row_ptr, gemm_k, quant_a_row_ptr);
                        rows_tobe_handled -= 1;
                        m_idx += 1;
                    }
                }
            }
        }

        ggml_barrier(params->threadpool);

        ধ্রুৱ int64_t gemm_m_stride     = gemm_n / gemm_m > 64 ? gemm_m : 16;
        ধ্রুৱ int64_t gemm_m_blocked    = spacemit_kernels::div_round_up(gemm_m, gemm_m_stride);
        ধ্রুৱ int64_t max_gemm_n_stride = spacemit_kernels::div_round_up(gemm_n * gemm_m_blocked, nth);

        int64_t gemm_n_stride = gemm_n;
        যদি (max_gemm_n_stride < gemm_n) {
            gemm_n_stride =
                std::min(gemm_n_stride, spacemit_kernels::div_round_up(max_gemm_n_stride, NB_COLS) * NB_COLS);
        }

        যদি (gemm_n_stride == gemm_n && tcm_buffer != nullptr && per_mb_rows_wsize <= tcm_buffer_size) {
            বাবে (int64_t m_start = ith * row_align; m_start < gemm_m; m_start += row_align * nth) {
                uint8_t * b_col    = reinterpret_cast<uint8_t *>(w_data);
                uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? b_col : nullptr;

                int64_t m_row_real = std::min(gemm_m - m_start, row_align);

                spacemit_kernels::rvv::memcpy1d(tcm_buffer, quant_a_buffer + m_start * row_stride_a,
                                                m_row_real * row_stride_a);

                int64_t n_blk_real = 0;
                বাবে (int64_t ni = 0; ni < gemm_n; ni += n_blk_real, b_col += n_blk_real * row_stride_b) {
                    n_blk_real = std::min(gemm_n - ni, (int64_t) NB_COLS);

                    uint8_t * a_row_ptr = (uint8_t *) tcm_buffer;
                    ভিন্ন *   c_blk     = output + m_start * gemm_n + ni;

                    int32_t rows_remaining = m_row_real;

                    যেতিয়ালৈ (rows_remaining > 0) {
                        auto rows_handled = gemm_kernel(b_blk_len, a_row_ptr, b_col, b_col_zp, c_blk, rows_remaining,
                                                        n_blk_real, b_k_blks, gemm_n);

                        c_blk += rows_handled * gemm_n;
                        a_row_ptr += rows_handled * row_stride_a;

                        rows_remaining -= rows_handled;
                    }
                }
            }
        } অন্যথা যদি (tcm_buffer != nullptr && per_nb_cols_wsize <= tcm_buffer_size) {
            uint8_t * a_row = quant_a_buffer;
            uint8_t * b_col = reinterpret_cast<uint8_t *>(tcm_buffer);
            যদি ((gemm_workspace_size + per_nb_cols_wsize) <= tcm_buffer_size) {
                a_row = (uint8_t *) tcm_buffer;
                b_col = reinterpret_cast<uint8_t *>(tcm_buffer) + gemm_workspace_size;
            }
            uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? b_col : nullptr;

            int64_t ni      = ith * NB_COLS;
            int64_t nb_real = std::min(gemm_n - ni, NB_COLS);

            যদি (ith % 2 == 0 && nb_real > 0) {
                spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(w_data) + ni * row_stride_b,
                                                nb_real * row_stride_b);
                যদি (a_row != quant_a_buffer) {
                    spacemit_kernels::rvv::memcpy1d(a_row, quant_a_buffer, gemm_workspace_size);
                }
            }

            spine_barrier_wait(cur_barrier);

            যদি (ith % 2 != 0 && nb_real > 0) {
                যদি (a_row != quant_a_buffer) {
                    spacemit_kernels::rvv::memcpy1d(a_row, quant_a_buffer, gemm_workspace_size);
                }
                spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(w_data) + ni * row_stride_b,
                                                nb_real * row_stride_b);
            }

            বাবে (; ni < gemm_n; ni += NB_COLS * nth) {
                int64_t rows_remaining = gemm_m;
                ভিন্ন * c_blk          = output + ni;
                auto *  a_row_cur      = a_row;

                যদি (ith % 2 != 0) {
                    spine_barrier_wait(cur_barrier);
                }

                যেতিয়ালৈ (rows_remaining > 0) {
                    auto rows_handled = gemm_kernel(b_blk_len, a_row_cur, b_col, b_col_zp, c_blk, rows_remaining,
                                                    nb_real, b_k_blks, gemm_n);

                    c_blk += rows_handled * gemm_n;
                    a_row_cur += rows_handled * row_stride_a;

                    rows_remaining -= rows_handled;
                }

                যদি (ith % 2 == 0) {
                    spine_barrier_wait(cur_barrier);
                }

                ধ্রুৱ int64_t next_ni = ni + NB_COLS * nth;
                যদি (next_ni < gemm_n) {
                    nb_real = std::min(gemm_n - next_ni, NB_COLS);
                    spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(w_data) + next_ni * row_stride_b,
                                                    nb_real * row_stride_b);
                }
            }
        } অন্যথা {
            ধ্রুৱ int64_t task_গুনতি_m = spacemit_kernels::div_round_up(gemm_m, gemm_m_stride);
            ধ্রুৱ int64_t task_গুনতি_n = spacemit_kernels::div_round_up(gemm_n, gemm_n_stride);

            int64_t task_গুনতি      = task_গুনতি_m * task_গুনতি_n;
            int64_t task_per_thread = (task_গুনতি + nth - 1) / nth;
            int64_t start           = ith * task_per_thread;
            int64_t end             = std::min((ith + 1) * task_per_thread, task_গুনতি);
            বাবে (int64_t compute_idx = start; compute_idx < end; compute_idx++) {
                ধ্রুৱ auto tid_n = compute_idx / task_গুনতি_m;
                ধ্রুৱ auto tid_m = compute_idx % task_গুনতি_m;

                ধ্রুৱ int64_t m_start = tid_m * gemm_m_stride;
                ধ্রুৱ int64_t m_গুনতি = std::min(gemm_m - m_start, (int64_t) gemm_m_stride);

                ধ্রুৱ int64_t n_start = tid_n * gemm_n_stride;
                ধ্রুৱ int64_t n_গুনতি = std::min(gemm_n - n_start, (int64_t) gemm_n_stride);

                ধ্রুৱ int64_t n_blk = m_গুনতি == 1 ? n_গুনতি : NB_COLS;

                uint8_t * b_col    = reinterpret_cast<uint8_t *>(w_data) + n_start * row_stride_b;
                uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? b_col : nullptr;

                int64_t n_blk_real = 0;
                বাবে (int64_t ni = 0; ni < n_গুনতি; ni += n_blk_real, b_col += n_blk_real * row_stride_b) {
                    n_blk_real = std::min(n_গুনতি - ni, n_blk);

                    uint8_t * a_row = quant_a_buffer + m_start * row_stride_a;

                    ভিন্ন * c_blk = output + m_start * gemm_n + n_start + ni;

                    int64_t rows_remaining = m_গুনতি;

                    uint8_t * b_col_cur    = b_col;
                    uint8_t * b_col_zp_cur = b_col_zp;

                    যেতিয়ালৈ (rows_remaining > 0) {
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
        constexpr size_t a_blk_len = INTER_SIZE;
        constexpr size_t b_blk_len = INTER_SIZE;

        ধ্রুৱ ggml_tensor * src0 = op->src[0];
        ধ্রুৱ ggml_tensor * src1 = op->src[1];
        ধ্রুৱ ggml_tensor * ids  = op->src[2];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_দ্বিমিক_OP_LOCALS

        পূর্ণাঙ্ক ith = params->ith;
        পূর্ণাঙ্ক nth = params->nth;

        // row groups
        ধ্রুৱ পূর্ণাঙ্ক n_ids = ids->ne[0];  // n_expert_used
        ধ্রুৱ পূর্ণাঙ্ক n_as  = ne02;        // n_expert

        সংরচনা mmid_row_mapping {
            int32_t i1;
            int32_t i2;
        };

        spacemit_kernels::quantize_a_row_def           quantize_a_row_i8;
        spacemit_kernels::gemm_kernel_quantize_def     gemm_kernel;
        spacemit_kernels::moe_gemm_kernel_quantize_def moe_gemm_kernel_m2;
        তার্কিক                                           set_kernel_impl = মিছা;
        size_t                                         block_stride_a  = spacemit_kernels::q8_blk_size(QK4_0);

#যদি defined(RISCV64_SPACEMIT_IME2)
        যদি (!set_kernel_impl && (global_spine_env_info.use_ime2)) {
            quantize_a_row_i8 = spacemit_kernels::rvv::quantize_a_row_i8;
            block_stride_a    = spacemit_kernels::q8_blk_size(QK4_0, সঁচা);

            যদি constexpr (std::is_same_v<BLOC_TYPE, block_q6_K> || std::is_same_v<BLOC_TYPE, block_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i8;
                set_kernel_impl = সঁচা;
            } অন্যথা যদি constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> || std::is_same_v<BLOC_TYPE, block_q4_1> ||
                                 std::is_same_v<BLOC_TYPE, block_q4_K>) {
                যদি constexpr (INTER_SIZE == 256) {
                    gemm_kernel       = spacemit_kernels::ime2::gemm_kernel_i8i4_hp;
                    quantize_a_row_i8 = spacemit_kernels::rvv::quantize_a_row_i8_hp;
                    block_stride_a    = spacemit_kernels::q8_hp_blk_size(a_blk_len, সঁচা, সঁচা);
                    set_kernel_impl   = সঁচা;
                } অন্যথা {
                    gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i4;
                    moe_gemm_kernel_m2 = spacemit_kernels::ime2::moe_m2_gemm_kernel_i8i4;
                    quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8;
                    block_stride_a     = spacemit_kernels::q8_blk_size(a_blk_len, সঁচা);
                    set_kernel_impl    = সঁচা;
                }
            } অন্যথা যদি constexpr (std::is_same_v<BLOC_TYPE, block_q2_K>) {
                quantize_a_row_i8 = spacemit_kernels::rvv::quantize_a_row_i8k;
                block_stride_a    = spacemit_kernels::q8k_blk_size(a_blk_len);
                gemm_kernel       = spacemit_kernels::ime2::gemm_kernel_i8i2k;
                set_kernel_impl   = সঁচা;
            } অন্যথা যদি constexpr (std::is_same_v<BLOC_TYPE, block_q3_K>) {
                quantize_a_row_i8 = spacemit_kernels::rvv::quantize_a_row_i8k;
                block_stride_a    = spacemit_kernels::q8k_blk_size(a_blk_len);
                gemm_kernel       = spacemit_kernels::ime2::gemm_kernel_i8i3k;
                set_kernel_impl   = সঁচা;
            } অন্যথা যদি constexpr (std::is_same_v<BLOC_TYPE, block_mxfp4>) {
                gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8mxfp4;
                moe_gemm_kernel_m2 = spacemit_kernels::ime2::moe_m2_gemm_kernel_i8mxfp4;
                set_kernel_impl    = সঁচা;
            } অন্যথা যদি constexpr (std::is_same_v<BLOC_TYPE, block_q5_1> || std::is_same_v<BLOC_TYPE, block_q5_K> ||
                                 std::is_same_v<BLOC_TYPE, block_q5_0>) {
                gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i5;
                moe_gemm_kernel_m2 = spacemit_kernels::ime2::moe_m2_gemm_kernel_i8i5;
                set_kernel_impl    = সঁচা;
            }
        }
#endif

#যদি defined(RISCV64_SPACEMIT_IME1)
        যদি (!set_kernel_impl && (global_spine_env_info.use_ime1)) {
            quantize_a_row_i8 = spacemit_kernels::ime1::quantize_a_row_i8;

            যদি constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> || std::is_same_v<BLOC_TYPE, block_q4_1> ||
                          std::is_same_v<BLOC_TYPE, block_q4_K>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i4;
                set_kernel_impl = সঁচা;
            } অন্যথা যদি constexpr (std::is_same_v<BLOC_TYPE, block_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i8;
                set_kernel_impl = সঁচা;
            }
        }
#endif
        যদি (!set_kernel_impl) {
            GGML_ABORT("no kernel implementation found for the block type");
        }

        ধ্রুৱ size_t a_k_blks = spacemit_kernels::div_round_up(ne10, a_blk_len);
        ধ্রুৱ size_t b_k_blks = spacemit_kernels::div_round_up(ne10, b_blk_len);

        ধ্রুৱ size_t nbw1                = a_k_blks * block_stride_a;
        ধ্রুৱ size_t nbw2                = ne11 * nbw1;
        ধ্রুৱ size_t nbw3                = nbw2 * ne12;
        ধ্রুৱ size_t gemm_workspace_size = GGML_PAD(nbw3, alignof(int64_t));

        ধ্রুৱ uintptr_t ws_ptr         = reinterpret_cast<uintptr_t>(params->wdata);
        auto *          quant_a_buffer = reinterpret_cast<uint8_t *>(ws_ptr);

        যদি (ne11 == 1) {
            বাবে (int64_t ii = ith; ii < ne12 * a_k_blks; ii += nth) {
                int64_t i12       = ii / a_k_blks;
                int64_t ak_blk_id = ii % a_k_blks;
                quantize_a_row_i8(a_blk_len, (ভিন্ন *) ((ৱর্ণ *) src1->data + i12 * nb12) + ak_blk_id * a_blk_len,
                                  a_blk_len, quant_a_buffer + i12 * nbw2 + ak_blk_id * block_stride_a);
            }
        } অন্যথা {
            বাবে (int64_t ii = ith; ii < ne12 * ne11; ii += nth) {
                int64_t i12 = ii / ne11;
                int64_t i11 = ii % ne11;
                quantize_a_row_i8(a_blk_len, (ভিন্ন *) ((ৱর্ণ *) src1->data + i12 * nb12 + i11 * nb11), ne10,
                                  quant_a_buffer + i12 * nbw2 + i11 * nbw1);
            }
        }

#define MMID_MATRIX_ROW(row_id, i1) matrix_rows[(row_id) *ne12 + (i1)]

        int64_t *          matrix_row_counts       = (int64_t *) (ws_ptr + gemm_workspace_size);
        int32_t *          valid_ep_গুনতি          = (int32_t *) (matrix_row_counts + n_as);
        int32_t *          valid_act_গুনতি         = (int32_t *) (valid_ep_গুনতি + 1);
        int64_t *          valid_matrix_row_counts = (int64_t *) (valid_act_গুনতি + 1);
        mmid_row_mapping * matrix_rows             = (mmid_row_mapping *) (valid_matrix_row_counts + n_as);

        যদি (ith == 0) {
            // initialize matrix_row_counts
            memset(matrix_row_counts, 0, n_as * পরিমাণ(int64_t));

            // group rows by src0 matrix
            বাবে (int32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
                বাবে (int32_t id = 0; id < n_ids; ++id) {
                    ধ্রুৱ int32_t i02 =
                        *(ধ্রুৱ int32_t *) ((ধ্রুৱ ৱর্ণ *) ids->data + iid1 * ids->nb[1] + id * ids->nb[0]);

                    GGML_ASSERT(i02 >= 0 && i02 < n_as);

                    MMID_MATRIX_ROW(i02, matrix_row_counts[i02]) = { id, iid1 };
                    matrix_row_counts[i02] += 1;
                }
            }

            int32_t valid_ep_গুনতি_t  = 0;
            int32_t valid_act_গুনতি_t = 0;
            বাবে (পূর্ণাঙ্ক cur_a = 0; cur_a < n_as; ++cur_a) {
                ধ্রুৱ int64_t cne1 = matrix_row_counts[cur_a];
                যদি (cne1 == 0) {
                    অনুৱর্ত;
                }
                valid_matrix_row_counts[valid_ep_গুনতি_t] = cur_a;
                valid_act_গুনতি_t += cne1;
                valid_ep_গুনতি_t += 1;
            }
            valid_ep_গুনতি[0]  = valid_ep_গুনতি_t;
            valid_act_গুনতি[0] = valid_act_গুনতি_t;
        }

        ধ্রুৱ int64_t barrier_idx = static_cast<int64_t>(ith / 2);

        GGML_ASSERT(global_spine_env_info.init_barrier != nullptr);
        GGML_ASSERT(barrier_idx < spine_init_barrier_গুনতি);
        spine_barrier_t * cur_barrier = &global_spine_env_info.init_barrier[barrier_idx];

        ggml_barrier(params->threadpool);

        ধ্রুৱ size_t row_stride_b      = b_k_blks * get_repacked_block_type_size<BLOC_TYPE, INTER_SIZE, NB_COLS>();
        ধ্রুৱ size_t expert_b_stride   = ne01 * row_stride_b;
        ধ্রুৱ size_t per_nb_cols_wsize = NB_COLS * row_stride_b;

        std::array<ধ্রুৱ uint8_t *, 2> src_workspaces;
        std::array<ভিন্ন *, 2>         dst_workspaces;

        auto *     tcm_buffer      = ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer;
        ধ্রুৱ auto tcm_buffer_size = ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer_size;

        ধ্রুৱ auto valid_ep_গুনতি_t  = valid_ep_গুনতি[0];
        ধ্রুৱ auto valid_act_গুনতি_t = valid_act_গুনতি[0];

        পূর্ণাঙ্ক nth_es = 1;
        পূর্ণাঙ্ক nth_n  = nth;

        পূর্ণাঙ্ক ith_es = ith % nth_es;
        পূর্ণাঙ্ক ith_n  = (ith / nth_es) % nth_n;

        যদি (valid_ep_গুনতি_t % nth == 0 && tcm_buffer != nullptr && valid_ep_গুনতি_t == n_as &&
            valid_act_গুনতি_t == n_as && per_nb_cols_wsize <= tcm_buffer_size) {
            বাবে (int64_t valid_id = ith; valid_id < valid_ep_গুনতি_t; valid_id += nth) {
                ধ্রুৱ int64_t cur_a = valid_matrix_row_counts[valid_id];

                auto * src0_cur = (uint8_t *) src0->data + cur_a * expert_b_stride;

                mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, 0);
                ধ্রুৱ পূর্ণাঙ্ক        id          = row_mapping.i1;
                ধ্রুৱ int64_t    i11         = id % ne11;
                ধ্রুৱ int64_t    i12         = row_mapping.i2;
                ধ্রুৱ int64_t    i1          = id;
                ধ্রুৱ int64_t    i2          = i12;

                auto *  src1_col = quant_a_buffer + (i11 * nbw1 + i12 * nbw2);
                ভিন্ন * c_blk    = (ভিন্ন *) ((ৱর্ণ *) dst->data + (i1 * nb1 + i2 * nb2));

                uint8_t * a_row = src1_col;
                uint8_t * b_col = reinterpret_cast<uint8_t *>(tcm_buffer);
                যদি ((nbw1 + per_nb_cols_wsize) <= tcm_buffer_size) {
                    a_row = (uint8_t *) tcm_buffer;
                    b_col = reinterpret_cast<uint8_t *>(tcm_buffer) + nbw1;
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
                বাবে (int64_t ni = 0; ni < ne01; ni += NB_COLS) {
                    যদি (ith % 2 != 0) {
                        spine_barrier_wait(cur_barrier);
                    }

                    gemm_kernel(b_blk_len, a_row, b_col, b_col_zp, c_blk + ni, 1, nb_real, b_k_blks, ne01);

                    যদি (ith % 2 == 0) {
                        spine_barrier_wait(cur_barrier);
                    }

                    ধ্রুৱ int64_t next_ni = ni + NB_COLS;
                    যদি (next_ni < ne01) {
                        nb_real = std::min(ne01 - next_ni, NB_COLS);
                        spacemit_kernels::rvv::memcpy1d(
                            b_col, reinterpret_cast<uint8_t *>(src0_cur) + next_ni * row_stride_b, per_nb_cols_wsize);
                    }
                }
            }
        } অন্যথা {
            বাবে (int64_t valid_id = ith_es; valid_id < valid_ep_গুনতি_t; valid_id += nth_es) {
                ধ্রুৱ int64_t cur_a = valid_matrix_row_counts[valid_id];
                ধ্রুৱ int64_t cne1  = matrix_row_counts[cur_a];

                int64_t src1_cur_start = 0;
                int64_t src1_cur_end   = cne1;

                int64_t src0_cur_start = (ith_n * ne01) / nth_n;
                int64_t src0_cur_end   = MIN(((ith_n + 1) * ne01) / nth_n, ne01);

                যদি (src1_cur_start >= src1_cur_end || src0_cur_start >= src0_cur_end) {
                    অনুৱর্ত;
                }

                src0_cur_start =
                    (src0_cur_start % NB_COLS) ? src0_cur_start + NB_COLS - (src0_cur_start % NB_COLS) : src0_cur_start;
                src0_cur_end =
                    (src0_cur_end % NB_COLS) ? src0_cur_end + NB_COLS - (src0_cur_end % NB_COLS) : src0_cur_end;

                auto *    src0_cur = (uint8_t *) src0->data + cur_a * expert_b_stride + src0_cur_start * row_stride_b;
                uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? src0_cur : nullptr;

                size_t extra_tcm_buffer_size = tcm_buffer_size;
                শূন্য * extra_tcm_buffer      = tcm_buffer;
                যদি (tcm_buffer != nullptr && (src1_cur_end - src1_cur_start) >= 4 &&
                    (src0_cur_end - src0_cur_start) * row_stride_b <= tcm_buffer_size) {
                    spacemit_kernels::rvv::memcpy1d(tcm_buffer, src0_cur,
                                                    (src0_cur_end - src0_cur_start) * row_stride_b);
                    src0_cur = reinterpret_cast<uint8_t *>(tcm_buffer);
                    b_col_zp = block_type_has_zp<BLOC_TYPE>() ? src0_cur : nullptr;
                    extra_tcm_buffer_size -= (src0_cur_end - src0_cur_start) * row_stride_b;
                    extra_tcm_buffer = reinterpret_cast<শূন্য *>(reinterpret_cast<uintptr_t>(tcm_buffer) +
                                                                (src0_cur_end - src0_cur_start) * row_stride_b);
                }

                পূর্ণাঙ্ক ir1 = src1_cur_start;

                যদি (extra_tcm_buffer_size >= nbw1 && extra_tcm_buffer != nullptr) {
                    int64_t quant_a_tile_size = extra_tcm_buffer_size / nbw1;
                    কৰা {
                        quant_a_tile_size = MIN(quant_a_tile_size, src1_cur_end - ir1);

                        uint8_t * quant_a_tile_buffer = reinterpret_cast<uint8_t *>(extra_tcm_buffer);

                        পূর্ণাঙ্ক iir1 = ir1;
                        বাবে (; iir1 < (ir1 + quant_a_tile_size); ++iir1) {
                            mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, iir1);

                            ধ্রুৱ পূর্ণাঙ্ক id = row_mapping.i1;  // selected expert index

                            ধ্রুৱ int64_t i11 = id % ne11;
                            ধ্রুৱ int64_t i12 = row_mapping.i2;  // row index in src1

                            auto * src1_col = quant_a_buffer + (i11 * nbw1 + i12 * nbw2);
                            spacemit_kernels::rvv::memcpy1d(quant_a_tile_buffer, src1_col, nbw1);
                            quant_a_tile_buffer = quant_a_tile_buffer + nbw1;
                        }

                        quant_a_tile_buffer = reinterpret_cast<uint8_t *>(extra_tcm_buffer);
                        iir1                = ir1;

                        যদি (moe_gemm_kernel_m2 != nullptr) {
                            বাবে (; iir1 < (ir1 + quant_a_tile_size - 1); iir1 += 2, quant_a_tile_buffer += 2 * nbw1) {
                                mmid_row_mapping row_mapping_0 = MMID_MATRIX_ROW(cur_a, iir1);
                                mmid_row_mapping row_mapping_1 = MMID_MATRIX_ROW(cur_a, iir1 + 1);

                                src_workspaces[0] = quant_a_tile_buffer;
                                src_workspaces[1] = quant_a_tile_buffer + nbw1;

                                dst_workspaces[0] =
                                    (ভিন্ন *) ((ৱর্ণ *) dst->data + (row_mapping_0.i1 * nb1 + row_mapping_0.i2 * nb2)) +
                                    src0_cur_start;
                                dst_workspaces[1] = (ভিন্ন *) ((ৱর্ণ *) dst->data +
                                                               ((row_mapping_1.i1) * nb1 + (row_mapping_1.i2) * nb2)) +
                                                    src0_cur_start;
                                moe_gemm_kernel_m2(b_blk_len, src_workspaces.data(), src0_cur, b_col_zp,
                                                   dst_workspaces.data(), 1, src0_cur_end - src0_cur_start, b_k_blks,
                                                   ne01);
                            }
                        }

                        বাবে (; iir1 < (ir1 + quant_a_tile_size); iir1++, quant_a_tile_buffer += nbw1) {
                            mmid_row_mapping row_mapping_0 = MMID_MATRIX_ROW(cur_a, iir1);

                            gemm_kernel(
                                b_blk_len, quant_a_tile_buffer, src0_cur, b_col_zp,
                                (ভিন্ন *) ((ৱর্ণ *) dst->data + (row_mapping_0.i1 * nb1 + row_mapping_0.i2 * nb2)) +
                                    src0_cur_start,
                                1, src0_cur_end - src0_cur_start, b_k_blks, ne01);
                        }

                        ir1 += quant_a_tile_size;
                    } যেতিয়ালৈ (ir1 < src1_cur_end);
                } অন্যথা {
                    যদি (moe_gemm_kernel_m2 != nullptr) {
                        বাবে (; ir1 < src1_cur_end - 1; ir1 += 2) {
                            বাবে (পূর্ণাঙ্ক iir1 = 0; iir1 < 2; ++iir1) {
                                mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, ir1 + iir1);

                                ধ্রুৱ পূর্ণাঙ্ক id = row_mapping.i1;  // selected expert index

                                ধ্রুৱ int64_t i11 = id % ne11;
                                ধ্রুৱ int64_t i12 = row_mapping.i2;  // row index in src1

                                ধ্রুৱ int64_t i1 = id;               // selected expert index
                                ধ্রুৱ int64_t i2 = i12;              // row

                                src_workspaces[iir1] = quant_a_buffer + (i11 * nbw1 + i12 * nbw2);

                                dst_workspaces[iir1] =
                                    (ভিন্ন *) ((ৱর্ণ *) dst->data + (i1 * nb1 + i2 * nb2)) + src0_cur_start;
                            }

                            moe_gemm_kernel_m2(b_blk_len, src_workspaces.data(), src0_cur, b_col_zp,
                                               dst_workspaces.data(), 1, src0_cur_end - src0_cur_start, b_k_blks, ne01);
                        }
                    }

                    বাবে (; ir1 < src1_cur_end; ir1++) {
                        mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, ir1);

                        ধ্রুৱ পূর্ণাঙ্ক id = row_mapping.i1;  // selected expert index

                        ধ্রুৱ int64_t i11 = id % ne11;
                        ধ্রুৱ int64_t i12 = row_mapping.i2;  // row index in src1

                        ধ্রুৱ int64_t i1 = id;               // selected expert index
                        ধ্রুৱ int64_t i2 = i12;              // row

                        auto * src1_col = quant_a_buffer + (i11 * nbw1 + i12 * nbw2);

                        gemm_kernel(b_blk_len, src1_col, src0_cur, b_col_zp,
                                    (ভিন্ন *) ((ৱর্ণ *) dst->data + (i1 * nb1 + i2 * nb2)) + src0_cur_start, 1,
                                    src0_cur_end - src0_cur_start, b_k_blks, ne01);
                    }
                }
            }
        }
#undef MMID_MATRIX_ROW
    }

    পূর্ণাঙ্ক repack(ggml_tensor * t, ধ্রুৱ শূন্য * data, size_t data_size) override {
        GGML_LOG_DEBUG("%s: repack tensor %s with %s_%dx%d\n", __func__, t->name, ggml_type_name(t->type),
                       (পূর্ণাঙ্ক) NB_COLS, (পূর্ণাঙ্ক) INTER_SIZE);
        ঘূৰাই ggml::cpu::riscv64_spacemit::repack<BLOC_TYPE, INTER_SIZE, NB_COLS>(t, data, data_size);
    }
};

শ্ৰেণী tensor_traits_common : সার্ৱজনিক tensor_traits_base {
    তার্কিক work_size(পূর্ণাঙ্ক n_threads, ধ্রুৱ ggml_tensor * op, size_t & size) override {
        ৱিকল্প (op->op) {
            পক্ষ GGML_OP_FLASH_ATTN_EXT:
                {
                    ধ্রুৱ পূর্ণাঙ্ক     n_tasks = n_threads;
                    ধ্রুৱ int64_t neq2    = op->src[0]->ne[2];  // number of query heads
                    ধ্রুৱ int64_t DK      = op->src[1]->ne[0];
                    ধ্রুৱ int64_t DV      = op->src[2]->ne[0];  // DV

                    // Tiled flash attention scratch (tile sizes defined in common.h)
                    // Per-thread: Q_q + KQ + mask + VKQ32 + V32 + K_f32 + padding
                    size_t prefill = পরিমাণ(ভিন্ন) *
                                     (GGML_FA_TILE_Q * DK + 2 * GGML_FA_TILE_Q * GGML_FA_TILE_KV + GGML_FA_TILE_Q * DV +
                                      GGML_FA_TILE_KV * DV + GGML_FA_TILE_KV * DK) *
                                     n_tasks;

                    // Decode path: n_kv_chunks = n_tasks (one chunk per thread)
                    // Per-thread: VKQ accmulator (DV), partial M, partial S + intra-thread scratch for V, Q and VKQ
                    size_t n_chunks = n_tasks;
                    size_t decode   = পরিমাণ(ভিন্ন) * (neq2 * n_chunks * (2 + DV) + n_tasks * (DK + 2 * DV));

                    size = MAX(prefill, decode);
                }
                ঘূৰাই সঁচা;
            স্ৱভাৱ:
                ভঙ্গ;
        }
        ঘূৰাই মিছা;
    }

    তার্কিক compute_forward(ggml_compute_params * params, ggml_tensor * op) override {
        ৱিকল্প (op->op) {
            পক্ষ GGML_OP_NORM:
                ৱিকল্প (op->src[0]->type) {
                    পক্ষ GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_norm_f32(params, op);
                        ঘূৰাই সঁচা;
                    স্ৱভাৱ:
                        GGML_ABORT("fatal error");
                }
            পক্ষ GGML_OP_RMS_NORM:
                ৱিকল্প (op->src[0]->type) {
                    পক্ষ GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_rms_norm_f32(params, op);
                        ঘূৰাই সঁচা;
                    স্ৱভাৱ:
                        GGML_ABORT("fatal error");
                }
            পক্ষ GGML_OP_ADD:
                ৱিকল্প (op->src[0]->type) {
                    পক্ষ GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_দ্বিমিক<GGML_OP_ADD, ভিন্ন>(params, op);
                        ঘূৰাই সঁচা;
                    পক্ষ GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_দ্বিমিক<GGML_OP_ADD, _Float16>(params, op);
                        ঘূৰাই সঁচা;
                    স্ৱভাৱ:
                        ggml_compute_forward_add(params, op);
                        ঘূৰাই সঁচা;
                }
            পক্ষ GGML_OP_SUB:
                ৱিকল্প (op->src[0]->type) {
                    পক্ষ GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_দ্বিমিক<GGML_OP_SUB, ভিন্ন>(params, op);
                        ঘূৰাই সঁচা;
                    পক্ষ GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_দ্বিমিক<GGML_OP_SUB, _Float16>(params, op);
                        ঘূৰাই সঁচা;
                    স্ৱভাৱ:
                        ggml_compute_forward_sub(params, op);
                        ঘূৰাই সঁচা;
                }
            পক্ষ GGML_OP_MUL:
                ৱিকল্প (op->src[0]->type) {
                    পক্ষ GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_দ্বিমিক<GGML_OP_MUL, ভিন্ন>(params, op);
                        ঘূৰাই সঁচা;
                    পক্ষ GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_দ্বিমিক<GGML_OP_MUL, _Float16>(params, op);
                        ঘূৰাই সঁচা;
                    স্ৱভাৱ:
                        ggml_compute_forward_mul(params, op);
                        ঘূৰাই সঁচা;
                }
            পক্ষ GGML_OP_DIV:
                ৱিকল্প (op->src[0]->type) {
                    পক্ষ GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_দ্বিমিক<GGML_OP_DIV, ভিন্ন>(params, op);
                        ঘূৰাই সঁচা;
                    পক্ষ GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_দ্বিমিক<GGML_OP_DIV, _Float16>(params, op);
                        ঘূৰাই সঁচা;
                    স্ৱভাৱ:
                        ggml_compute_forward_div(params, op);
                        ঘূৰাই সঁচা;
                }
            পক্ষ GGML_OP_FLASH_ATTN_EXT:
                forward_flash_attn_ext_f16(params, op);
                ঘূৰাই সঁচা;
            পক্ষ GGML_OP_CONT:
                {
                    ধ্রুৱ ggml_tensor * src0 = op->src[0];
                    যদি (op->type == src0->type && op->nb[0] != src0->nb[0] && op->nb[0] == src0->nb[1] &&
                        op->ne[3] * op->ne[2] * op->nb[2] == src0->ne[3] * src0->ne[2] * src0->nb[2]) {
                        spacemit_kernels::rvv::forward_cont_with_permute(params, op);
                    } অন্যথা {
                        ggml_compute_forward_cont(params, op);
                    }
                    ঘূৰাই সঁচা;
                }
            পক্ষ GGML_OP_CPY:
                {
                    ধ্রুৱ ggml_tensor * src0 = op->src[0];
                    যদি (op->type == src0->type && op->nb[0] == src0->nb[1] && src0->nb[0] != src0->nb[1] &&
                        ggml_nelements(src0) == ggml_nelements(op)) {
                        spacemit_kernels::rvv::forward_cpy_with_permute(params, op);
                    } অন্যথা {
                        ggml_compute_forward_cpy(params, op);
                    }
                    ঘূৰাই সঁচা;
                }
            পক্ষ GGML_OP_REPEAT:
                {
                    ধ্রুৱ তার্কিক rows_equal         = ggml_nrows(op->src[0]) == ggml_nrows(op);
                    ধ্রুৱ তার্কিক broadcast_or_equal = op->src[0]->ne[0] == 1 || op->src[0]->ne[0] == op->ne[0];

                    যদি (rows_equal && broadcast_or_equal) {
                        ৱিকল্প (op->src[0]->type) {
                            পক্ষ GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_repeat_nrows<int32_t>(params, op);
                                ঘূৰাই সঁচা;
                            পক্ষ GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_repeat_nrows<int16_t>(params, op);
                                ঘূৰাই সঁচা;
                            স্ৱভাৱ:
                                ভঙ্গ;
                        }
                    }

                    যদি (op->src[0]->ne[1] == 1 && op->src[0]->ne[0] == op->ne[0]) {
                        ৱিকল্প (op->src[0]->type) {
                            পক্ষ GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_repeat_dim1<int32_t>(params, op);
                                ঘূৰাই সঁচা;
                            পক্ষ GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_repeat_dim1<int16_t>(params, op);
                                ঘূৰাই সঁচা;
                            স্ৱভাৱ:
                                ভঙ্গ;
                        }
                    }

                    ggml_compute_forward_repeat(params, op);
                }
                ঘূৰাই সঁচা;
            পক্ষ GGML_OP_SUM_ROWS:
                {
                    যদি (op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32) {
                        spacemit_kernels::rvv::forward_sum_rows<ভিন্ন>(params, op);
                    } অন্যথা {
                        ggml_compute_forward_sum_rows(params, op);
                    }
                }
                ঘূৰাই সঁচা;
            পক্ষ GGML_OP_GET_ROWS:
                {
                    যদি (op->src[0]->type == op->type) {
                        ৱিকল্প (op->src[0]->type) {
                            পক্ষ GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_get_rows<int32_t>(params, op);
                                ঘূৰাই সঁচা;
                            পক্ষ GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_get_rows<int16_t>(params, op);
                                ঘূৰাই সঁচা;
                            স্ৱভাৱ:
                                ভঙ্গ;
                        }
                    }

                    ggml_compute_forward_get_rows(params, op);
                }
                ঘূৰাই সঁচা;
            পক্ষ GGML_OP_CONCAT:
                {
                    ধ্রুৱ int32_t dim = ggml_get_op_params_i32(op, 0);
                    যদি (dim == 0 && op->type == op->src[0]->type) {
                        ৱিকল্প (op->src[0]->type) {
                            পক্ষ GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_concat<int32_t>(params, op);
                                ঘূৰাই সঁচা;
                            পক্ষ GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_concat<int16_t>(params, op);
                                ঘূৰাই সঁচা;
                            স্ৱভাৱ:
                                ভঙ্গ;
                        }
                    }

                    ggml_compute_forward_concat(params, op);
                }
                ঘূৰাই সঁচা;
            // TODO For GGML_OP_GATED_DELTA_NET
            // case GGML_OP_GATED_DELTA_NET:
            //     return true;
            স্ৱভাৱ:
                ভঙ্গ;
        }
        ঘূৰাই মিছা;
    }

    শূন্য forward_flash_attn_ext_f16(ধ্রুৱ ggml_compute_params * params, ggml_tensor * dst) {
        ধ্রুৱ ggml_tensor * q = dst->src[0];
        ধ্রুৱ ggml_tensor * k = dst->src[1];
        ধ্রুৱ ggml_tensor * v = dst->src[2];

        GGML_TENSOR_LOCALS(int64_t, neq, q, ne)
        GGML_TENSOR_LOCALS(size_t, nbq, q, nb)
        GGML_TENSOR_LOCALS(int64_t, nek, k, ne)
        GGML_TENSOR_LOCALS(size_t, nbk, k, nb)
        GGML_TENSOR_LOCALS(int64_t, nev, v, ne)
        GGML_TENSOR_LOCALS(size_t, nbv, v, nb)
        GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
        GGML_TENSOR_LOCALS(size_t, nb, dst, nb)

        ধ্রুৱ int64_t DK = nek0;
        ধ্রুৱ int64_t DV = nev0;

        ধ্রুৱ তার্কিক supported_prec  = (dst->op_params[3] == GGML_PREC_F32 || dst->op_params[3] == GGML_PREC_DEFAULT);
        ধ্রুৱ তার্কিক supported_types = (q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16);
        ধ্রুৱ তার্কিক supported_shape = (DK > 0 && DK <= 128 && DV > 0 && DV <= 128);
        ধ্রুৱ তার্কিক supported_vlen  = (__riscv_vlenb() == 128);

        যদি (!(supported_prec && supported_types && supported_shape && supported_vlen)) {
            ggml_compute_forward_flash_attn_ext(params, dst);
            ঘূৰাই;
        }

        // total rows in q
        ধ্রুৱ int64_t nr = neq1 * neq2 * neq3;

        // rows per thread
        ধ্রুৱ পূর্ণাঙ্ক ith = params->ith;
        ধ্রুৱ পূর্ণাঙ্ক nth = params->nth;

        স্থির constexpr int64_t Q_TILE_SZ = ggml_fa_tile_config::Q;
        ধ্রুৱ তার্কিক               use_tiled = !params->use_ref && (neq1 >= Q_TILE_SZ);

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
        ধ্রুৱ int64_t dr = (nr + nchunk - 1) / nchunk;

        // The first chunk comes from our thread_id, the rest will get auto-assigned.
        পূর্ণাঙ্ক current_chunk = ith;

        যেতিয়ালৈ (current_chunk < nchunk) {
            ধ্রুৱ int64_t ir0 = dr * current_chunk;
            ধ্রুৱ int64_t ir1 = MIN(ir0 + dr, nr);

            যদি (use_tiled) {
                spacemit_kernels::rvv::forward_flash_attn_ext_f16_tiled_vlen1024_vf16(
                    params, dst, ir0, ir1, ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer,
                    ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer_size);
            } অন্যথা {
                spacemit_kernels::rvv::forward_flash_attn_ext_f16_one_chunk_vlen1024_vf16(
                    params, dst, ir0, ir1, ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer,
                    ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer_size);
            }

            current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
        }
    }

    পূর্ণাঙ্ক repack(ggml_tensor * t, ধ্রুৱ শূন্য * data, size_t data_size) override {
        memcpy(t->data, data, data_size);
        ঘূৰাই 0;
    }
};

// Impl By IME1
স্থির ধ্রুৱ tensor_traits<block_q4_0, 32, 16>  q4_0_16x32_q8_0;
স্থির ধ্রুৱ tensor_traits<block_q4_1, 32, 16>  q4_1_16x32_q8_0;
স্থির ধ্রুৱ tensor_traits<block_q4_K, 32, 16>  q4_k_16x32_q8_0;
স্থির ধ্রুৱ tensor_traits<block_q8_0, 32, 16>  q8_0_16x32_q8_0;
// Impl By IME2
স্থির ধ্রুৱ tensor_traits<block_q2_K, 256, 32> q2_k_32x256_q8_0;
স্থির ধ্রুৱ tensor_traits<block_q3_K, 256, 32> q3_k_32x256_q8_0;
স্থির ধ্রুৱ tensor_traits<block_q4_0, 32, 32>  q4_0_32x32_q8_0;
স্থির ধ্রুৱ tensor_traits<block_q4_1, 32, 32>  q4_1_32x32_q8_0;
স্থির ধ্রুৱ tensor_traits<block_q4_0, 256, 32> q4_0_32x256_q8_0;
স্থির ধ্রুৱ tensor_traits<block_q4_1, 256, 32> q4_1_32x256_q8_0;
স্থির ধ্রুৱ tensor_traits<block_q4_K, 32, 32>  q4_k_32x32_q8_0;
স্থির ধ্রুৱ tensor_traits<block_q6_K, 32, 32>  q6_k_32x32_q8_0;
স্থির ধ্রুৱ tensor_traits<block_q8_0, 32, 32>  q8_0_32x32_q8_0;
স্থির ধ্রুৱ tensor_traits<block_mxfp4, 32, 32> mxfp4_32x32_q8_0;
স্থির ধ্রুৱ tensor_traits<block_q5_K, 32, 32>  q5_k_32x32_q8_0;
স্থির ধ্রুৱ tensor_traits<block_q5_1, 32, 32>  q5_1_32x32_q8_0;
স্থির ধ্রুৱ tensor_traits<block_q5_0, 32, 32>  q5_0_32x32_q8_0;
// Impl By RVV
স্থির ধ্রুৱ tensor_traits_common               rvv_impl;

}  // namespace ggml::cpu::riscv64_spacemit

স্থির ধ্রুৱ ggml::cpu::tensor_traits * ggml_riscv64_spacemit_get_optimal_repack_type(ধ্রুৱ ggml_tensor * cur) {
    ৱিকল্প (cur->type) {
        পক্ষ GGML_TYPE_Q2_K:
            {
#যদি defined(RISCV64_SPACEMIT_IME2)
                যদি (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    ঘূৰাই &ggml::cpu::riscv64_spacemit::q2_k_32x256_q8_0;
                }
#endif
            }
            ভঙ্গ;
        পক্ষ GGML_TYPE_Q3_K:
            {
#যদি defined(RISCV64_SPACEMIT_IME2)
                যদি (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    ঘূৰাই &ggml::cpu::riscv64_spacemit::q3_k_32x256_q8_0;
                }
#endif
            }
            ভঙ্গ;
        পক্ষ GGML_TYPE_Q4_0:
            {
#যদি defined(RISCV64_SPACEMIT_IME2)
                যদি (cur->ne[1] % 32 == 0 && cur->ne[0] % 256 == 0 &&
                    (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    ঘূৰাই &ggml::cpu::riscv64_spacemit::q4_0_32x256_q8_0;
                }

                যদি (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    ঘূৰাই &ggml::cpu::riscv64_spacemit::q4_0_32x32_q8_0;
                }
#endif

#যদি defined(RISCV64_SPACEMIT_IME1)
                যদি (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    ঘূৰাই &ggml::cpu::riscv64_spacemit::q4_0_16x32_q8_0;
                }
#endif
            }
            ভঙ্গ;
        পক্ষ GGML_TYPE_Q4_1:
            {
#যদি defined(RISCV64_SPACEMIT_IME2)
                // TODO
                // if (cur->ne[1] % 32 == 0 && cur->ne[0] % 256 == 0 &&
                //     (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                //     return &ggml::cpu::riscv64_spacemit::q4_1_32x256_q8_0;
                // }

                যদি (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    ঘূৰাই &ggml::cpu::riscv64_spacemit::q4_1_32x32_q8_0;
                }
#endif

#যদি defined(RISCV64_SPACEMIT_IME1)
                যদি (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    ঘূৰাই &ggml::cpu::riscv64_spacemit::q4_1_16x32_q8_0;
                }
#endif
            }
            ভঙ্গ;
        পক্ষ GGML_TYPE_Q4_K:
            {
#যদি defined(RISCV64_SPACEMIT_IME2)
                যদি (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    ঘূৰাই &ggml::cpu::riscv64_spacemit::q4_k_32x32_q8_0;
                }
#endif

#যদি defined(RISCV64_SPACEMIT_IME1)
                যদি (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    ঘূৰাই &ggml::cpu::riscv64_spacemit::q4_k_16x32_q8_0;
                }
#endif
            }
            ভঙ্গ;
        পক্ষ GGML_TYPE_Q6_K:
            {
#যদি defined(RISCV64_SPACEMIT_IME2)
                যদি ((ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    ঘূৰাই &ggml::cpu::riscv64_spacemit::q6_k_32x32_q8_0;
                }
#endif
            }
            ভঙ্গ;
        পক্ষ GGML_TYPE_Q8_0:
            {
#যদি defined(RISCV64_SPACEMIT_IME2)
                যদি ((ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    ঘূৰাই &ggml::cpu::riscv64_spacemit::q8_0_32x32_q8_0;
                }
#endif

#যদি defined(RISCV64_SPACEMIT_IME1)
                যদি (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    ঘূৰাই &ggml::cpu::riscv64_spacemit::q8_0_16x32_q8_0;
                }
#endif
            }
            ভঙ্গ;
        পক্ষ GGML_TYPE_MXFP4:
            {
#যদি defined(RISCV64_SPACEMIT_IME2)
                // TODO
                // if (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                //     return &ggml::cpu::riscv64_spacemit::mxfp4_32x32_q8_0;
                // }
#endif
            }
            ভঙ্গ;
        পক্ষ GGML_TYPE_Q5_K:
            {
#যদি defined(RISCV64_SPACEMIT_IME2)
                যদি (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    ঘূৰাই &ggml::cpu::riscv64_spacemit::q5_k_32x32_q8_0;
                }
#endif
            }
            ভঙ্গ;
        পক্ষ GGML_TYPE_Q5_1:
            {
#যদি defined(RISCV64_SPACEMIT_IME2)
                যদি (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    ঘূৰাই &ggml::cpu::riscv64_spacemit::q5_1_32x32_q8_0;
                }
#endif
            }
            ভঙ্গ;
        পক্ষ GGML_TYPE_Q5_0:
            {
#যদি defined(RISCV64_SPACEMIT_IME2)
                যদি (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    ঘূৰাই &ggml::cpu::riscv64_spacemit::q5_0_32x32_q8_0;
                }
#endif
            }
            ভঙ্গ;
        স্ৱভাৱ:
            ভঙ্গ;
    }

    ঘূৰাই nullptr;
}

স্থির গণনা ggml_status ggml_backend_riscv64_spacemit_buffer_init_tensor(ggml_backend_buffer_t buffer,
                                                                         ggml_tensor *         tensor) {
    tensor->extra =
        (শূন্য *) const_cast<ggml::cpu::tensor_traits *>(ggml_riscv64_spacemit_get_optimal_repack_type(tensor));

    GGML_UNUSED(buffer);

    ঘূৰাই GGML_STATUS_SUCCESS;
}

স্থির শূন্য ggml_backend_riscv64_spacemit_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);

    শূন্য * base = buffer->context;
    যদি (base == nullptr) {
        ঘূৰাই;
    }

    ggml::cpu::riscv64_spacemit::spine_mem_pool_free(base);
}

স্থির শূন্য * ggml_backend_riscv64_spacemit_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);

    শূন্য * base = buffer->context;
    GGML_ASSERT(base != nullptr);
    ঘূৰাই base;
}

স্থির শূন্য ggml_backend_riscv64_spacemit_buffer_memset_tensor(ggml_backend_buffer_t buffer,
                                                               ggml_tensor *         tensor,
                                                               uint8_t               value,
                                                               size_t                offset,
                                                               size_t                size) {
    GGML_ASSERT(tensor);
    memset((ৱর্ণ *) tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

স্থির শূন্য ggml_backend_riscv64_spacemit_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);

    শূন্য * base = buffer->context;
    GGML_ASSERT(base != nullptr);
    memset(base, value, buffer->size);
}

স্থির শূন্য ggml_backend_riscv64_spacemit_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                            ggml_tensor *         tensor,
                                                            ধ্রুৱ শূন্য *          data,
                                                            size_t                offset,
                                                            size_t                size) {
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    auto tensor_traits = (ggml::cpu::riscv64_spacemit::tensor_traits_base *) tensor->extra;
    যদি (tensor_traits) {
        auto OK = tensor_traits->repack(tensor, data, size);
        GGML_ASSERT(OK == 0);
    }

    GGML_UNUSED(buffer);
}

স্থির ধ্রুৱ ggml_backend_buffer_i ggml_backend_riscv64_spacemit_buffer_i = {
    /* .free_buffer     = */ ggml_backend_riscv64_spacemit_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_riscv64_spacemit_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_riscv64_spacemit_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_riscv64_spacemit_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_riscv64_spacemit_buffer_set_tensor,
    /* .get_tensor      = */ nullptr,
    /* .set_tensor_2d   = */ nullptr,
    /* .get_tensor_2d   = */ nullptr,
    /* .cpy_tensor      = */ nullptr,
    /* .clear           = */ ggml_backend_riscv64_spacemit_buffer_clear,
    /* .reset           = */ nullptr,
};

স্থির ধ্রুৱ ৱর্ণ * ggml_backend_cpu_riscv64_spacemit_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ঘূৰাই "CPU_RISCV64_SPACEMIT";

    GGML_UNUSED(buft);
}

স্থির ggml_backend_buffer_t ggml_backend_cpu_riscv64_spacemit_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                                                                        size_t size) {
    শূন্য * base = ggml::cpu::riscv64_spacemit::spine_mem_pool_alloc(size, 64);
    যদি (base == nullptr) {
        ঘূৰাই nullptr;
    }

    ঘূৰাই ggml_backend_buffer_init(buft, ggml_backend_riscv64_spacemit_buffer_i, base, size);
}

স্থির size_t ggml_backend_cpu_riscv64_spacemit_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ঘূৰাই 64;

    GGML_UNUSED(buft);
}

স্থির size_t ggml_backend_cpu_riscv64_spacemit_nbytes(ggml_backend_buffer_type_t buft, ধ্রুৱ ggml_tensor * tensor) {
    বাবে (পূর্ণাঙ্ক i = 0; i < GGML_MAX_DIMS; ++i) {
        যদি (tensor->ne[i] <= 0) {
            ঘূৰাই 0;
        }
    }

    GGML_UNUSED(buft);

    ধ্রুৱ auto plain_nbytes = [&]() {
        size_t total = ggml_type_size(tensor->type);
        বাবে (পূর্ণাঙ্ক i = 0; i < GGML_MAX_DIMS; ++i) {
            total += (tensor->ne[i] - 1) * tensor->nb[i];
        }
        ঘূৰাই total;
    };

    ধ্রুৱ size_t blck_size = ggml_blck_size(tensor->type);
    যদি (blck_size == 1) {
        ঘূৰাই plain_nbytes();
    }

    ধ্রুৱ size_t row_nbytes = tensor->ne[0] * tensor->nb[0] / blck_size;

    ধ্রুৱ auto add_strided_nbytes = [&](size_t total, size_t src_block_size, size_t dst_block_size) {
        বাবে (পূর্ণাঙ্ক i = 1; i < GGML_MAX_DIMS; ++i) {
            total += (tensor->ne[i] - 1) * (tensor->nb[i] / src_block_size) * dst_block_size;
        }
        ঘূৰাই total;
    };

    ধ্রুৱ auto remap_block_nbytes = [&](size_t src_block_size, size_t dst_block_size, int64_t padded_rows = 0) {
        GGML_ASSERT(row_nbytes % src_block_size == 0);

        size_t total =
            add_strided_nbytes((row_nbytes / src_block_size) * dst_block_size, src_block_size, dst_block_size);

        যদি (padded_rows > 0 && tensor->ne[1] % padded_rows != 0) {
            total += (padded_rows - tensor->ne[1] % padded_rows) * (tensor->nb[1] / src_block_size) * dst_block_size;
        }

        ঘূৰাই total;
    };

    size_t nbytes = row_nbytes;
    ৱিকল্প (tensor->type) {
        পক্ষ GGML_TYPE_Q4_K:
            nbytes = remap_block_nbytes(পরিমাণ(block_q4_K), পরিমাণ(block_q4_1) * 8);
            ভঙ্গ;
        পক্ষ GGML_TYPE_Q6_K:
            nbytes = remap_block_nbytes(পরিমাণ(block_q6_K), পরিমাণ(block_q8_0) * 8, 32);
            ভঙ্গ;
        পক্ষ GGML_TYPE_Q8_0:
            nbytes = remap_block_nbytes(পরিমাণ(block_q8_0), পরিমাণ(block_q8_0), 32);
            ভঙ্গ;
        পক্ষ GGML_TYPE_Q2_K:
            nbytes = remap_block_nbytes(পরিমাণ(block_q2_K), পরিমাণ(spacemit_kernels::nrow_block_q2_k<1>));
            ভঙ্গ;
        পক্ষ GGML_TYPE_Q3_K:
            nbytes = remap_block_nbytes(পরিমাণ(block_q3_K), পরিমাণ(spacemit_kernels::nrow_block_q3_k<1>));
            ভঙ্গ;
        পক্ষ GGML_TYPE_MXFP4:
            nbytes = remap_block_nbytes(পরিমাণ(block_mxfp4), পরিমাণ(spacemit_kernels::nrow_block_mxfp4<1>));
            ভঙ্গ;
        পক্ষ GGML_TYPE_Q5_K:
            nbytes = remap_block_nbytes(পরিমাণ(block_q5_K), পরিমাণ(spacemit_kernels::nrow_block_q5_1<1>) * 8);
            ভঙ্গ;
        পক্ষ GGML_TYPE_Q5_1:
            nbytes = remap_block_nbytes(পরিমাণ(block_q5_1), পরিমাণ(spacemit_kernels::nrow_block_q5_1<1>));
            ভঙ্গ;
        পক্ষ GGML_TYPE_Q5_0:
            nbytes = remap_block_nbytes(পরিমাণ(block_q5_0), পরিমাণ(spacemit_kernels::nrow_block_q5_0<1>));
            ভঙ্গ;
        স্ৱভাৱ:
            nbytes = add_strided_nbytes(row_nbytes, 1, 1);
            ভঙ্গ;
    }

    ঘূৰাই nbytes;
}

নামাকাশ ggml::cpu::riscv64_spacemit {

শ্ৰেণী extra_buffer_type : ggml::cpu::extra_buffer_type {
    তার্কিক supports_op(ggml_backend_dev_t, ধ্রুৱ ggml_tensor * op) override {
        ৱিকল্প (op->op) {
            পক্ষ GGML_OP_MUL_MAT:
                যদি (op->src[0]->buffer && (ggml_n_dims(op->src[0]) == 2) &&
                    op->src[0]->buffer->buft == ggml_backend_cpu_riscv64_spacemit_buffer_type() &&
                    ggml_riscv64_spacemit_get_optimal_repack_type(op->src[0])) {
                    যদি (op->src[1]->buffer && !ggml_backend_buft_is_host(op->src[1]->buffer->buft)) {
                        ঘূৰাই মিছা;
                    }
                    যদি (op->src[1]->type == GGML_TYPE_F32) {
                        ঘূৰাই সঁচা;
                    }
                }
                ভঙ্গ;
            পক্ষ GGML_OP_MUL_MAT_ID:
                যদি (op->src[0]->buffer && (ggml_n_dims(op->src[0]) == 3) &&
                    op->src[0]->buffer->buft == ggml_backend_cpu_riscv64_spacemit_buffer_type() &&
                    ggml_riscv64_spacemit_get_optimal_repack_type(op->src[0])) {
                    যদি (op->src[1]->buffer && !ggml_backend_buft_is_host(op->src[1]->buffer->buft)) {
                        ঘূৰাই মিছা;
                    }
                    যদি (op->src[1]->type == GGML_TYPE_F32) {
                        ঘূৰাই সঁচা;
                    }
                }
                ভঙ্গ;
            স্ৱভাৱ:
                // GGML_ABORT("fatal error");
                ভঙ্গ;
        }
        ঘূৰাই মিছা;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(ধ্রুৱ ggml_tensor * op) override {
        ৱিকল্প (op->op) {
            পক্ষ GGML_OP_MUL_MAT:
            পক্ষ GGML_OP_MUL_MAT_ID:
                যদি (op->src[0]->buffer && op->src[0]->buffer->buft == ggml_backend_cpu_riscv64_spacemit_buffer_type()) {
                    ঘূৰাই (ggml::cpu::tensor_traits *) op->src[0]->extra;
                }
                ভঙ্গ;
            পক্ষ GGML_OP_NORM:
            পক্ষ GGML_OP_RMS_NORM:
            পক্ষ GGML_OP_ADD:
            পক্ষ GGML_OP_SUB:
            পক্ষ GGML_OP_MUL:
            পক্ষ GGML_OP_DIV:
            পক্ষ GGML_OP_FLASH_ATTN_EXT:
            পক্ষ GGML_OP_CONT:
            পক্ষ GGML_OP_CPY:
            পক্ষ GGML_OP_REPEAT:
            পক্ষ GGML_OP_SUM_ROWS:
            পক্ষ GGML_OP_GET_ROWS:
            পক্ষ GGML_OP_CONCAT:
                // case GGML_OP_GATED_DELTA_NET:
                ঘূৰাই (ggml::cpu::tensor_traits *) (&ggml::cpu::riscv64_spacemit::rvv_impl);
            স্ৱভাৱ:
                // GGML_ABORT("fatal error");
                ভঙ্গ;
        }

        ঘূৰাই nullptr;
    }
};

}  // namespace ggml::cpu::riscv64_spacemit

ggml_backend_buffer_type_t ggml_backend_cpu_riscv64_spacemit_buffer_type(শূন্য) {
    স্থির ggml_backend_buffer_type ggml_backend_cpu_buffer_type_riscv64_spacemit = {
  /* .iface    = */
        {
         /* .get_name           = */ ggml_backend_cpu_riscv64_spacemit_buffer_type_get_name,
         /* .alloc_buffer       = */ ggml_backend_cpu_riscv64_spacemit_buffer_type_alloc_buffer,
         /* .alloc_buffer_n     = */ রিক্ত,
         /* .get_alignment      = */ ggml_backend_cpu_riscv64_spacemit_buffer_type_get_alignment,
         /* .get_max_size       = */ nullptr,
         /* .get_alloc_size     = */ ggml_backend_cpu_riscv64_spacemit_nbytes,
         /* .get_alloc_size_n   = */ রিক্ত,
         /* .is_host            = */ nullptr,
         },
 /* .device  = */
        ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
 /* .context = */
        নূতন ggml::cpu::riscv64_spacemit::extra_buffer_type(),
    };

    ঘূৰাই &ggml_backend_cpu_buffer_type_riscv64_spacemit;
}

বাহ্য "C" {
স্থির পূর্ণাঙ্ক bind_ai_thread() {
    পূর্ণাঙ্ক  fd, bytes;
    ৱর্ণ str[32];

    fd = open("/proc/set_ai_thread", O_WRONLY);
    যদি (fd < 0) {
        GGML_LOG_ERROR("try open /proc/set_ai_thread failed\n");
        ঘূৰাই -1;
    }

    snprintf(str, 16, "%d", 0);
    bytes = লিখ(fd, str, strlen(str));
    যদি (bytes < 0) {
        GGML_LOG_ERROR("try write /proc/set_ai_thread failed\n");
        close(fd);
        ঘূৰাই -1;
    }

    close(fd);
    ঘূৰাই 0;
}

শূন্য ggml_backend_cpu_riscv64_spacemit_set_numa_thread_affinity(পূর্ণাঙ্ক thread_n) {
    পূর্ণাঙ্ক cpu_id = sched_getcpu();
    যদি (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2 &&
        !((1 << cpu_id) & ggml::cpu::riscv64_spacemit::global_spine_env_info.cpu_mask)) {
        GGML_PRINT_DEBUG("bind_ai_thread for thread %d, pid %d\n", thread_n, getpid());
        bind_ai_thread();
    }

    যদি (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_tcm &&
        ggml::cpu::riscv64_spacemit::tls_context.cpu_id == -1) {
        CPU_ZERO(&(ggml::cpu::riscv64_spacemit::tls_context.cpuset));
        pthread_t    main_thread     = pthread_self();
        ধ্রুৱ auto & perfer_core_ids = ggml::cpu::riscv64_spacemit::global_spine_env_info.perfer_core_ids;
        যদি (thread_n < 0 || static_cast<size_t>(thread_n) >= perfer_core_ids.size()) {
            GGML_ABORT("thread_n %d exceeds perfer_core_ids size %zu\n", thread_n, perfer_core_ids.size());
        }
        auto perfer_cpu_id = perfer_core_ids[static_cast<size_t>(thread_n)];
        CPU_SET(perfer_cpu_id, &(ggml::cpu::riscv64_spacemit::tls_context.cpuset));
        পূর্ণাঙ্ক s =
            pthread_setaffinity_np(main_thread, পরিমাণ(cpu_set_t), &(ggml::cpu::riscv64_spacemit::tls_context.cpuset));
        যদি (s != 0) {
            GGML_ABORT("set thread affinity error for thread_n %d, cpu_id %d\n", thread_n, perfer_cpu_id);
        }

        পূর্ণাঙ্ক ai_cpu_id = perfer_cpu_id - ggml::cpu::riscv64_spacemit::global_spine_env_info.aicpu_id_offset;
        ggml::cpu::riscv64_spacemit::tls_context.cpu_id = ai_cpu_id;
        ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer =
            ggml::cpu::riscv64_spacemit::spine_mem_pool_tcm_mem_get(ai_cpu_id);
        ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer_size =
            ggml::cpu::riscv64_spacemit::global_spine_env_info.tcm_blk_size;
    }

    যদি (ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer != nullptr) {
        শূন্য * rt =
            ggml::cpu::riscv64_spacemit::spine_mem_pool_tcm_mem_wait(ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        যদি (rt == nullptr) {
            GGML_ABORT("wait tcm buffer failed for cpu_id: %d", ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        }
    }
}

শূন্য ggml_backend_cpu_riscv64_spacemit_clear_numa_thread_affinity_threaded(পূর্ণাঙ্ক thread_n) {
    যদি (ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer != nullptr) {
        auto rt = ggml::cpu::riscv64_spacemit::spine_mem_pool_tcm_mem_release(
            ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        যদি (rt != 0) {
            GGML_ABORT("release tcm buffer failed for cpu_id: %d", ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        }
    }
}
}
