#define GGML_COMMON_IMPL_CPP
#define GGML_COMMON_DECL_CPP

#अन्तर्भाव "ime.h"

#अन्तर्भाव "binary-ops.h"
#अन्तर्भाव "common.h"
#अन्तर्भाव "ggml-backend-impl.h"
#अन्तर्भाव "ggml-common.h"
#अन्तर्भाव "ggml-cpu.h"
#अन्तर्भाव "ime_env.h"
#अन्तर्भाव "ime_kernels.h"
#अन्तर्भाव "ops.h"
#अन्तर्भाव "repack.h"
#अन्तर्भाव "rvv_kernels.h"
#अन्तर्भाव "spine_mem_pool.h"
#अन्तर्भाव "traits.h"
#अन्तर्भाव "vec.h"

#अन्तर्भाव <fcntl.h>
#अन्तर्भाव <sys/mman.h>
#अन्तर्भाव <unistd.h>

#अन्तर्भाव <algorithm>
#अन्तर्भाव <atomic>
#अन्तर्भाव <cassert>
#अन्तर्भाव <cerrno>
#अन्तर्भाव <cmath>
#अन्तर्भाव <cstdio>  // for GGML_ASSERT
#अन्तर्भाव <stdexcept>
#अन्तर्भाव <thread>
// clang-format off
#जइ defined(__riscv)

#जइ !defined(__riscv_v) || !defined(__riscv_v_intrinsic)
#error "riscv v extension or v_intrinsic not enabled"
#अण्णहा
#अन्तर्भाव <riscv_vector.h>
#endif

#जइ !defined(__riscv_zfh) || !defined(__riscv_zvfh)
#error "riscv zfh extension not enabled, GGML_RV_ZFH and GGML_RV_ZVFH must be defined to 1"
#endif

#जइ !defined(__riscv_zba)
#error "riscv zba extension not enabled, GGML_RV_ZBA must be defined to 1"
#endif

#जइ defined(RISCV64_SPACEMIT_IME1) || defined(RISCV64_SPACEMIT_IME2)
#अण्णहा
#error "RISCV64_SPACEMIT_IME1 or RISCV64_SPACEMIT_IME2 not defined"
#endif

#अण्णहा

#error "riscv not enabled in this build"

#endif

#जइ defined(__GNUC__)
#pragma GCC diagnostic ignored "-Woverlength-strings"
#pragma GCC diagnostic ignored "-Wcast-qual"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

// clang-format on

बाह्य "C" {
बाह्य शून्य ggml_threadpool_chunk_set(संरचना ggml_threadpool * tp, पूर्णाङ्क value);
बाह्य पूर्णाङ्क  ggml_threadpool_chunk_add(संरचना ggml_threadpool * tp, पूर्णाङ्क value);
}

नामाकाश ggml::cpu::riscv64_spacemit {

संरचना TLSContext {
    पूर्णाङ्क       cpu_id{ -1 };
    cpu_set_t cpuset;
    शून्य *    tcm_buffer{ nullptr };
    size_t    tcm_buffer_size{ 0 };
};

thread_local TLSContext tls_context;

template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS> constexpr size_t get_repacked_block_type_size() {
    जइ constexpr (std::is_same_v<BLOC_TYPE, block_q6_K> || std::is_same_v<BLOC_TYPE, block_q8_0>) {
        प्रत्यागम परिमाण(block_q8_0);
    } अण्णहा जइ constexpr (std::is_same_v<BLOC_TYPE, block_q4_0>) {
        प्रत्यागम परिमाण(block_q4_0) * INTER_SIZE / QK4_0;
    } अण्णहा जइ constexpr (std::is_same_v<BLOC_TYPE, block_q4_1> || std::is_same_v<BLOC_TYPE, block_q4_K>) {
        प्रत्यागम (परिमाण(block_q4_0) + परिमाण(uint8_t)) * INTER_SIZE / QK4_1;
    } अण्णहा जइ constexpr (std::is_same_v<BLOC_TYPE, block_q2_K>) {
        प्रत्यागम परिमाण(spacemit_kernels::nrow_block_q2_k<1>);
    } अण्णहा जइ constexpr (std::is_same_v<BLOC_TYPE, block_q3_K>) {
        प्रत्यागम परिमाण(spacemit_kernels::nrow_block_q3_k<1>);
    } अण्णहा जइ constexpr (std::is_same_v<BLOC_TYPE, block_mxfp4>) {
        प्रत्यागम परिमाण(spacemit_kernels::nrow_block_mxfp4<1>);
    } अण्णहा जइ constexpr (std::is_same_v<BLOC_TYPE, block_q5_1> || std::is_same_v<BLOC_TYPE, block_q5_K>) {
        प्रत्यागम परिमाण(spacemit_kernels::nrow_block_q5_1<1>);
    } अण्णहा जइ constexpr (std::is_same_v<BLOC_TYPE, block_q5_0>) {
        प्रत्यागम परिमाण(spacemit_kernels::nrow_block_q5_0<1>);
    } अण्णहा {
        प्रतिज्ञा(मिच्छ);
        प्रत्यागम 0;
    }
}

template <typename BLOC_TYPE> constexpr तार्किक block_type_has_zp() {
    जइ constexpr (std::is_same_v<BLOC_TYPE, block_q6_K> || std::is_same_v<BLOC_TYPE, block_q8_0> ||
                  std::is_same_v<BLOC_TYPE, block_q3_K> || std::is_same_v<BLOC_TYPE, block_q4_0> ||
                  std::is_same_v<BLOC_TYPE, block_mxfp4> || std::is_same_v<BLOC_TYPE, block_q5_0>) {
        प्रत्यागम मिच्छ;
    } अण्णहा जइ constexpr (std::is_same_v<BLOC_TYPE, block_q4_1> || std::is_same_v<BLOC_TYPE, block_q4_K> ||
                         std::is_same_v<BLOC_TYPE, block_q2_K> || std::is_same_v<BLOC_TYPE, block_q5_1> ||
                         std::is_same_v<BLOC_TYPE, block_q5_K>) {
        प्रत्यागम सच्च;
    } अण्णहा {
        प्रतिज्ञा(मिच्छ);
        प्रत्यागम मिच्छ;
    }
}

वर्ग tensor_traits_base : सार्वजनिक ggml::cpu::tensor_traits {
  सार्वजनिक:
    आभास पूर्णाङ्क repack(ggml_tensor * t, ध्रुव शून्य * data, size_t data_size) = 0;
};

template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS> वर्ग tensor_traits : सार्वजनिक tensor_traits_base {
    तार्किक work_size(पूर्णाङ्क /* n_threads */, ध्रुव ggml_tensor * op, size_t & size) override {
        विकल्प (op->op) {
            पक्ष GGML_OP_MUL_MAT:
                {
                    int64_t src1_nelements = ggml_nelements(op->src[1]);

                    जइ constexpr (std::is_same_v<BLOC_TYPE, block_q2_K> || std::is_same_v<BLOC_TYPE, block_q3_K>) {
                        size =
                            spacemit_kernels::div_round_up(src1_nelements, QK_K) * spacemit_kernels::q8k_blk_size(QK_K);
                    } अण्णहा जइ constexpr (INTER_SIZE == QK4_0) {
                        size = spacemit_kernels::div_round_up(src1_nelements, QK4_0) *
                               spacemit_kernels::q8_blk_size(QK4_0, सच्च);
                    } अण्णहा जइ constexpr (INTER_SIZE == 256) {
                        size = spacemit_kernels::div_round_up(src1_nelements, 256) *
                               spacemit_kernels::q8_hp_blk_size(256, सच्च, सच्च);
                    } अण्णहा {
                        GGML_ABORT("unsupported block type");
                    }

                    size = GGML_PAD(size, परिमाण(int64_t));

                    प्रत्यागम सच्च;
                }
            पक्ष GGML_OP_MUL_MAT_ID:
                {
                    int64_t src1_nelements = ggml_nelements(op->src[1]);

                    जइ constexpr (std::is_same_v<BLOC_TYPE, block_q2_K> || std::is_same_v<BLOC_TYPE, block_q3_K>) {
                        size =
                            spacemit_kernels::div_round_up(src1_nelements, QK_K) * spacemit_kernels::q8k_blk_size(QK_K);
                    } अण्णहा जइ constexpr (INTER_SIZE == QK4_0) {
                        size = spacemit_kernels::div_round_up(src1_nelements, QK4_0) *
                               spacemit_kernels::q8_blk_size(QK4_0, सच्च);
                    } अण्णहा जइ constexpr (INTER_SIZE == 256) {
                        size = spacemit_kernels::div_round_up(src1_nelements, 256) *
                               spacemit_kernels::q8_hp_blk_size(256, सच्च, सच्च);
                    } अण्णहा {
                        GGML_ABORT("unsupported block type");
                    }

                    size = GGML_PAD(size, परिमाण(int64_t));

                    ध्रुव int64_t ne02 = op->src[0]->ne[2];  // n_as, n_expert
                    ध्रुव int64_t ne12 = op->src[1]->ne[2];  // n_tokens

                    ध्रुव size_t sizeof_mmid_row_mapping = परिमाण(int64_t);
                    size += sizeof_mmid_row_mapping * ne02 * (ne12 + 1) + (ne02 + 1) * परिमाण(int64_t);

                    size = GGML_PAD(size, परिमाण(int64_t));

                    प्रत्यागम सच्च;
                }
            स्वभाव:
                // GGML_ABORT("fatal error");
                भङ्ग;
        }
        प्रत्यागम मिच्छ;
    }

    तार्किक compute_forward(ggml_compute_params * params, ggml_tensor * op) override {
        विकल्प (op->op) {
            पक्ष GGML_OP_MUL_MAT:
                विकल्प (op->src[0]->type) {
                    पक्ष GGML_TYPE_Q2_K:
                    पक्ष GGML_TYPE_Q3_K:
                    पक्ष GGML_TYPE_Q4_0:
                    पक्ष GGML_TYPE_Q4_1:
                    पक्ष GGML_TYPE_Q4_K:
                    पक्ष GGML_TYPE_Q6_K:
                    पक्ष GGML_TYPE_Q8_0:
                    पक्ष GGML_TYPE_Q5_0:
                    पक्ष GGML_TYPE_Q5_1:
                    पक्ष GGML_TYPE_Q5_K:
                        //case GGML_TYPE_MXFP4:
                        forward_mul_mat(params, op);
                        प्रत्यागम सच्च;
                    स्वभाव:
                        // GGML_ABORT("fatal error: unsupported type for src0 in MUL_MAT");
                        प्रत्यागम मिच्छ;
                }
                भङ्ग;
            पक्ष GGML_OP_MUL_MAT_ID:
                विकल्प (op->src[0]->type) {
                    पक्ष GGML_TYPE_Q2_K:
                    पक्ष GGML_TYPE_Q3_K:
                    पक्ष GGML_TYPE_Q4_0:
                    पक्ष GGML_TYPE_Q4_1:
                    पक्ष GGML_TYPE_Q4_K:
                    पक्ष GGML_TYPE_Q6_K:
                    पक्ष GGML_TYPE_Q8_0:
                    पक्ष GGML_TYPE_Q5_0:
                    पक्ष GGML_TYPE_Q5_1:
                    पक्ष GGML_TYPE_Q5_K:
                        //case GGML_TYPE_MXFP4:
                        forward_mul_mat_id(params, op);
                        प्रत्यागम सच्च;
                    स्वभाव:
                        // GGML_ABORT("fatal error: unsupported type for src0 in MUL_MAT_ID");
                        प्रत्यागम मिच्छ;
                }
                भङ्ग;
            स्वभाव:
                // GGML_ABORT("fatal error");
                भङ्ग;
        }
        प्रत्यागम मिच्छ;
    }

    शून्य forward_mul_mat(ggml_compute_params * params, ggml_tensor * op) {
        constexpr size_t a_blk_len = INTER_SIZE;
        constexpr size_t b_blk_len = INTER_SIZE;

        ध्रुव ggml_tensor * src0 = op->src[0];
        ध्रुव ggml_tensor * src1 = op->src[1];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_द्विमान_OP_LOCALS

        पूर्णाङ्क ith = params->ith;
        पूर्णाङ्क nth = params->nth;

        [[maybe_unused]] ध्रुव गणना ggml_type type = src0->type;

        शून्य *        w_data  = (शून्य *) src0->data;
        ध्रुव भिन्न * feature = (ध्रुव भिन्न *) src1->data;
        भिन्न *       output  = (भिन्न *) dst->data;

        ध्रुव int64_t gemm_m = ne11 * ne12 * ne13;
        ध्रुव int64_t gemm_k = ne10;
        ध्रुव int64_t gemm_n = ne01;

        spacemit_kernels::quantize_a_row_def       quantize_a_row_i8;
        spacemit_kernels::quantize_a_row_def       quantize_a_4row_i8;
        spacemit_kernels::gemm_kernel_quantize_def gemm_kernel;
        तार्किक                                       set_kernel_impl = मिच्छ;

        int64_t block_stride_a = spacemit_kernels::q8_blk_size(a_blk_len);

#जइ defined(RISCV64_SPACEMIT_IME2)
        जइ (!set_kernel_impl && (global_spine_env_info.use_ime2)) {
            quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8;
            quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8;
            block_stride_a     = spacemit_kernels::q8_blk_size(a_blk_len, सच्च);

            जइ constexpr (std::is_same_v<BLOC_TYPE, block_q6_K> || std::is_same_v<BLOC_TYPE, block_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i8;
                set_kernel_impl = सच्च;
            } अण्णहा जइ constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> || std::is_same_v<BLOC_TYPE, block_q4_1> ||
                                 std::is_same_v<BLOC_TYPE, block_q4_K>) {
                जइ constexpr (INTER_SIZE == 256) {
                    gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i4_hp;
                    quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8_hp;
                    quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8_hp;
                    block_stride_a     = spacemit_kernels::q8_hp_blk_size(a_blk_len, सच्च, सच्च);
                    set_kernel_impl    = सच्च;
                } अण्णहा {
                    gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i4;
                    quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8;
                    quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8;
                    block_stride_a     = spacemit_kernels::q8_blk_size(a_blk_len, सच्च);
                    set_kernel_impl    = सच्च;
                }
            } अण्णहा जइ constexpr (std::is_same_v<BLOC_TYPE, block_q2_K>) {
                quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8k;
                quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8k;
                block_stride_a     = spacemit_kernels::q8k_blk_size(a_blk_len);

                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i2k;
                set_kernel_impl = सच्च;
            } अण्णहा जइ constexpr (std::is_same_v<BLOC_TYPE, block_q3_K>) {
                quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8k;
                quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8k;
                block_stride_a     = spacemit_kernels::q8k_blk_size(a_blk_len);

                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i3k;
                set_kernel_impl = सच्च;
            } अण्णहा जइ constexpr (std::is_same_v<BLOC_TYPE, block_mxfp4>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8mxfp4;
                set_kernel_impl = सच्च;
            } अण्णहा जइ constexpr (std::is_same_v<BLOC_TYPE, block_q5_1> || std::is_same_v<BLOC_TYPE, block_q5_K> ||
                                 std::is_same_v<BLOC_TYPE, block_q5_0>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i5;
                set_kernel_impl = सच्च;
            }
        }
#endif

#जइ defined(RISCV64_SPACEMIT_IME1)
        जइ (!set_kernel_impl && (global_spine_env_info.use_ime1)) {
            quantize_a_row_i8  = spacemit_kernels::ime1::quantize_a_row_i8;
            quantize_a_4row_i8 = spacemit_kernels::ime1::quantize_a_4row_i8;

            जइ constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> || std::is_same_v<BLOC_TYPE, block_q4_1> ||
                          std::is_same_v<BLOC_TYPE, block_q4_K>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i4;
                set_kernel_impl = सच्च;
            } अण्णहा जइ constexpr (std::is_same_v<BLOC_TYPE, block_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i8;
                set_kernel_impl = सच्च;
            }
        }
#endif
        जइ (!set_kernel_impl) {
            GGML_ABORT("no kernel implementation found for the block type");
        }

        ध्रुव int64_t a_k_blks = spacemit_kernels::div_round_up(gemm_k, a_blk_len);
        ध्रुव int64_t b_k_blks = spacemit_kernels::div_round_up(gemm_k, b_blk_len);

        ध्रुव int64_t row_stride_a        = a_k_blks * block_stride_a;
        ध्रुव int64_t gemm_workspace_size = GGML_PAD(gemm_m * row_stride_a, alignof(int64_t));

        जइ (ith == 0 && params->wsize < gemm_workspace_size) {
            GGML_ABORT("wsize less than gemm_workspace_size");
        }

        uintptr_t ws_ptr = reinterpret_cast<uintptr_t>(params->wdata);

        शून्य *        tcm_buffer      = ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer;
        ध्रुव int64_t tcm_buffer_size = ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer_size;

        auto * quant_a_buffer = reinterpret_cast<uint8_t *>(ws_ptr);

        constexpr int64_t row_align = 4;
        ध्रुव int64_t     row_blks  = spacemit_kernels::div_round_up(gemm_m, row_align);

        ध्रुव int64_t row_stride_b      = b_k_blks * get_repacked_block_type_size<BLOC_TYPE, INTER_SIZE, NB_COLS>();
        ध्रुव int64_t per_mb_rows_wsize = row_align * row_stride_a;
        ध्रुव int64_t per_nb_cols_wsize = NB_COLS * row_stride_b;

        ध्रुव int64_t barrier_idx = static_cast<int64_t>(ith / 2);

        GGML_ASSERT(global_spine_env_info.init_barrier != nullptr);
        GGML_ASSERT(barrier_idx < spine_init_barrier_गणनम्);
        spine_barrier_t * cur_barrier = &global_spine_env_info.init_barrier[barrier_idx];

        जइ (gemm_m == 1) {
            पूर्णाङ्क task_per_thread = spacemit_kernels::div_round_up(a_k_blks, nth);
            पूर्णाङ्क a_blk_start     = ith * task_per_thread;
            पूर्णाङ्क a_blk_end       = std::min(a_blk_start + task_per_thread, (पूर्णाङ्क) a_k_blks);
            जइ (a_blk_start < a_blk_end) {
                quantize_a_row_i8(a_blk_len, feature + a_blk_start * a_blk_len, (a_blk_end - a_blk_start) * a_blk_len,
                                  quant_a_buffer + a_blk_start * block_stride_a);
            }
        } अण्णहा {
            पूर्णाङ्क task_per_thread = spacemit_kernels::div_round_up(row_blks, nth);
            पूर्णाङ्क m_row_blk_start = ith * task_per_thread;
            पूर्णाङ्क m_row_blk_end   = std::min(m_row_blk_start + task_per_thread, (पूर्णाङ्क) row_blks);
            प्रति (पूर्णाङ्क m_row_blk = m_row_blk_start; m_row_blk < m_row_blk_end; m_row_blk++) {
                पूर्णाङ्क m_idx             = m_row_blk * row_align;
                पूर्णाङ्क rows_tobe_handled = (gemm_m - m_idx) > row_align ? row_align : (gemm_m - m_idx);

                जइ (rows_tobe_handled == row_align && quantize_a_4row_i8 != nullptr) {
                    ध्रुव भिन्न * a_row_ptr       = feature + m_idx * gemm_k;
                    auto *        quant_a_row_ptr = quant_a_buffer + m_idx * row_stride_a;
                    quantize_a_4row_i8(a_blk_len, a_row_ptr, gemm_k, quant_a_row_ptr);
                } अण्णहा {
                    जाव (rows_tobe_handled) {
                        ध्रुव भिन्न * a_row_ptr       = feature + m_idx * gemm_k;
                        auto *        quant_a_row_ptr = quant_a_buffer + m_idx * row_stride_a;
                        quantize_a_row_i8(a_blk_len, a_row_ptr, gemm_k, quant_a_row_ptr);
                        rows_tobe_handled -= 1;
                        m_idx += 1;
                    }
                }
            }
        }

        ggml_barrier(params->threadpool);

        ध्रुव int64_t gemm_m_stride     = gemm_n / gemm_m > 64 ? gemm_m : 16;
        ध्रुव int64_t gemm_m_blocked    = spacemit_kernels::div_round_up(gemm_m, gemm_m_stride);
        ध्रुव int64_t max_gemm_n_stride = spacemit_kernels::div_round_up(gemm_n * gemm_m_blocked, nth);

        int64_t gemm_n_stride = gemm_n;
        जइ (max_gemm_n_stride < gemm_n) {
            gemm_n_stride =
                std::min(gemm_n_stride, spacemit_kernels::div_round_up(max_gemm_n_stride, NB_COLS) * NB_COLS);
        }

        जइ (gemm_n_stride == gemm_n && tcm_buffer != nullptr && per_mb_rows_wsize <= tcm_buffer_size) {
            प्रति (int64_t m_start = ith * row_align; m_start < gemm_m; m_start += row_align * nth) {
                uint8_t * b_col    = reinterpret_cast<uint8_t *>(w_data);
                uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? b_col : nullptr;

                int64_t m_row_real = std::min(gemm_m - m_start, row_align);

                spacemit_kernels::rvv::memcpy1d(tcm_buffer, quant_a_buffer + m_start * row_stride_a,
                                                m_row_real * row_stride_a);

                int64_t n_blk_real = 0;
                प्रति (int64_t ni = 0; ni < gemm_n; ni += n_blk_real, b_col += n_blk_real * row_stride_b) {
                    n_blk_real = std::min(gemm_n - ni, (int64_t) NB_COLS);

                    uint8_t * a_row_ptr = (uint8_t *) tcm_buffer;
                    भिन्न *   c_blk     = output + m_start * gemm_n + ni;

                    int32_t rows_remaining = m_row_real;

                    जाव (rows_remaining > 0) {
                        auto rows_handled = gemm_kernel(b_blk_len, a_row_ptr, b_col, b_col_zp, c_blk, rows_remaining,
                                                        n_blk_real, b_k_blks, gemm_n);

                        c_blk += rows_handled * gemm_n;
                        a_row_ptr += rows_handled * row_stride_a;

                        rows_remaining -= rows_handled;
                    }
                }
            }
        } अण्णहा जइ (tcm_buffer != nullptr && per_nb_cols_wsize <= tcm_buffer_size) {
            uint8_t * a_row = quant_a_buffer;
            uint8_t * b_col = reinterpret_cast<uint8_t *>(tcm_buffer);
            जइ ((gemm_workspace_size + per_nb_cols_wsize) <= tcm_buffer_size) {
                a_row = (uint8_t *) tcm_buffer;
                b_col = reinterpret_cast<uint8_t *>(tcm_buffer) + gemm_workspace_size;
            }
            uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? b_col : nullptr;

            int64_t ni      = ith * NB_COLS;
            int64_t nb_real = std::min(gemm_n - ni, NB_COLS);

            जइ (ith % 2 == 0 && nb_real > 0) {
                spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(w_data) + ni * row_stride_b,
                                                nb_real * row_stride_b);
                जइ (a_row != quant_a_buffer) {
                    spacemit_kernels::rvv::memcpy1d(a_row, quant_a_buffer, gemm_workspace_size);
                }
            }

            spine_barrier_wait(cur_barrier);

            जइ (ith % 2 != 0 && nb_real > 0) {
                जइ (a_row != quant_a_buffer) {
                    spacemit_kernels::rvv::memcpy1d(a_row, quant_a_buffer, gemm_workspace_size);
                }
                spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(w_data) + ni * row_stride_b,
                                                nb_real * row_stride_b);
            }

            प्रति (; ni < gemm_n; ni += NB_COLS * nth) {
                int64_t rows_remaining = gemm_m;
                भिन्न * c_blk          = output + ni;
                auto *  a_row_cur      = a_row;

                जइ (ith % 2 != 0) {
                    spine_barrier_wait(cur_barrier);
                }

                जाव (rows_remaining > 0) {
                    auto rows_handled = gemm_kernel(b_blk_len, a_row_cur, b_col, b_col_zp, c_blk, rows_remaining,
                                                    nb_real, b_k_blks, gemm_n);

                    c_blk += rows_handled * gemm_n;
                    a_row_cur += rows_handled * row_stride_a;

                    rows_remaining -= rows_handled;
                }

                जइ (ith % 2 == 0) {
                    spine_barrier_wait(cur_barrier);
                }

                ध्रुव int64_t next_ni = ni + NB_COLS * nth;
                जइ (next_ni < gemm_n) {
                    nb_real = std::min(gemm_n - next_ni, NB_COLS);
                    spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(w_data) + next_ni * row_stride_b,
                                                    nb_real * row_stride_b);
                }
            }
        } अण्णहा {
            ध्रुव int64_t task_गणनम्_m = spacemit_kernels::div_round_up(gemm_m, gemm_m_stride);
            ध्रुव int64_t task_गणनम्_n = spacemit_kernels::div_round_up(gemm_n, gemm_n_stride);

            int64_t task_गणनम्      = task_गणनम्_m * task_गणनम्_n;
            int64_t task_per_thread = (task_गणनम् + nth - 1) / nth;
            int64_t start           = ith * task_per_thread;
            int64_t end             = std::min((ith + 1) * task_per_thread, task_गणनम्);
            प्रति (int64_t compute_idx = start; compute_idx < end; compute_idx++) {
                ध्रुव auto tid_n = compute_idx / task_गणनम्_m;
                ध्रुव auto tid_m = compute_idx % task_गणनम्_m;

                ध्रुव int64_t m_start = tid_m * gemm_m_stride;
                ध्रुव int64_t m_गणनम् = std::min(gemm_m - m_start, (int64_t) gemm_m_stride);

                ध्रुव int64_t n_start = tid_n * gemm_n_stride;
                ध्रुव int64_t n_गणनम् = std::min(gemm_n - n_start, (int64_t) gemm_n_stride);

                ध्रुव int64_t n_blk = m_गणनम् == 1 ? n_गणनम् : NB_COLS;

                uint8_t * b_col    = reinterpret_cast<uint8_t *>(w_data) + n_start * row_stride_b;
                uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? b_col : nullptr;

                int64_t n_blk_real = 0;
                प्रति (int64_t ni = 0; ni < n_गणनम्; ni += n_blk_real, b_col += n_blk_real * row_stride_b) {
                    n_blk_real = std::min(n_गणनम् - ni, n_blk);

                    uint8_t * a_row = quant_a_buffer + m_start * row_stride_a;

                    भिन्न * c_blk = output + m_start * gemm_n + n_start + ni;

                    int64_t rows_remaining = m_गणनम्;

                    uint8_t * b_col_cur    = b_col;
                    uint8_t * b_col_zp_cur = b_col_zp;

                    जाव (rows_remaining > 0) {
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

    शून्य forward_mul_mat_id(ggml_compute_params * params, ggml_tensor * op) {
        constexpr size_t a_blk_len = INTER_SIZE;
        constexpr size_t b_blk_len = INTER_SIZE;

        ध्रुव ggml_tensor * src0 = op->src[0];
        ध्रुव ggml_tensor * src1 = op->src[1];
        ध्रुव ggml_tensor * ids  = op->src[2];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_द्विमान_OP_LOCALS

        पूर्णाङ्क ith = params->ith;
        पूर्णाङ्क nth = params->nth;

        // row groups
        ध्रुव पूर्णाङ्क n_ids = ids->ne[0];  // n_expert_used
        ध्रुव पूर्णाङ्क n_as  = ne02;        // n_expert

        संरचना mmid_row_mapping {
            int32_t i1;
            int32_t i2;
        };

        spacemit_kernels::quantize_a_row_def           quantize_a_row_i8;
        spacemit_kernels::gemm_kernel_quantize_def     gemm_kernel;
        spacemit_kernels::moe_gemm_kernel_quantize_def moe_gemm_kernel_m2;
        तार्किक                                           set_kernel_impl = मिच्छ;
        size_t                                         block_stride_a  = spacemit_kernels::q8_blk_size(QK4_0);

#जइ defined(RISCV64_SPACEMIT_IME2)
        जइ (!set_kernel_impl && (global_spine_env_info.use_ime2)) {
            quantize_a_row_i8 = spacemit_kernels::rvv::quantize_a_row_i8;
            block_stride_a    = spacemit_kernels::q8_blk_size(QK4_0, सच्च);

            जइ constexpr (std::is_same_v<BLOC_TYPE, block_q6_K> || std::is_same_v<BLOC_TYPE, block_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i8;
                set_kernel_impl = सच्च;
            } अण्णहा जइ constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> || std::is_same_v<BLOC_TYPE, block_q4_1> ||
                                 std::is_same_v<BLOC_TYPE, block_q4_K>) {
                जइ constexpr (INTER_SIZE == 256) {
                    gemm_kernel       = spacemit_kernels::ime2::gemm_kernel_i8i4_hp;
                    quantize_a_row_i8 = spacemit_kernels::rvv::quantize_a_row_i8_hp;
                    block_stride_a    = spacemit_kernels::q8_hp_blk_size(a_blk_len, सच्च, सच्च);
                    set_kernel_impl   = सच्च;
                } अण्णहा {
                    gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i4;
                    moe_gemm_kernel_m2 = spacemit_kernels::ime2::moe_m2_gemm_kernel_i8i4;
                    quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8;
                    block_stride_a     = spacemit_kernels::q8_blk_size(a_blk_len, सच्च);
                    set_kernel_impl    = सच्च;
                }
            } अण्णहा जइ constexpr (std::is_same_v<BLOC_TYPE, block_q2_K>) {
                quantize_a_row_i8 = spacemit_kernels::rvv::quantize_a_row_i8k;
                block_stride_a    = spacemit_kernels::q8k_blk_size(a_blk_len);
                gemm_kernel       = spacemit_kernels::ime2::gemm_kernel_i8i2k;
                set_kernel_impl   = सच्च;
            } अण्णहा जइ constexpr (std::is_same_v<BLOC_TYPE, block_q3_K>) {
                quantize_a_row_i8 = spacemit_kernels::rvv::quantize_a_row_i8k;
                block_stride_a    = spacemit_kernels::q8k_blk_size(a_blk_len);
                gemm_kernel       = spacemit_kernels::ime2::gemm_kernel_i8i3k;
                set_kernel_impl   = सच्च;
            } अण्णहा जइ constexpr (std::is_same_v<BLOC_TYPE, block_mxfp4>) {
                gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8mxfp4;
                moe_gemm_kernel_m2 = spacemit_kernels::ime2::moe_m2_gemm_kernel_i8mxfp4;
                set_kernel_impl    = सच्च;
            } अण्णहा जइ constexpr (std::is_same_v<BLOC_TYPE, block_q5_1> || std::is_same_v<BLOC_TYPE, block_q5_K> ||
                                 std::is_same_v<BLOC_TYPE, block_q5_0>) {
                gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i5;
                moe_gemm_kernel_m2 = spacemit_kernels::ime2::moe_m2_gemm_kernel_i8i5;
                set_kernel_impl    = सच्च;
            }
        }
#endif

#जइ defined(RISCV64_SPACEMIT_IME1)
        जइ (!set_kernel_impl && (global_spine_env_info.use_ime1)) {
            quantize_a_row_i8 = spacemit_kernels::ime1::quantize_a_row_i8;

            जइ constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> || std::is_same_v<BLOC_TYPE, block_q4_1> ||
                          std::is_same_v<BLOC_TYPE, block_q4_K>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i4;
                set_kernel_impl = सच्च;
            } अण्णहा जइ constexpr (std::is_same_v<BLOC_TYPE, block_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i8;
                set_kernel_impl = सच्च;
            }
        }
#endif
        जइ (!set_kernel_impl) {
            GGML_ABORT("no kernel implementation found for the block type");
        }

        ध्रुव size_t a_k_blks = spacemit_kernels::div_round_up(ne10, a_blk_len);
        ध्रुव size_t b_k_blks = spacemit_kernels::div_round_up(ne10, b_blk_len);

        ध्रुव size_t nbw1                = a_k_blks * block_stride_a;
        ध्रुव size_t nbw2                = ne11 * nbw1;
        ध्रुव size_t nbw3                = nbw2 * ne12;
        ध्रुव size_t gemm_workspace_size = GGML_PAD(nbw3, alignof(int64_t));

        ध्रुव uintptr_t ws_ptr         = reinterpret_cast<uintptr_t>(params->wdata);
        auto *          quant_a_buffer = reinterpret_cast<uint8_t *>(ws_ptr);

        जइ (ne11 == 1) {
            प्रति (int64_t ii = ith; ii < ne12 * a_k_blks; ii += nth) {
                int64_t i12       = ii / a_k_blks;
                int64_t ak_blk_id = ii % a_k_blks;
                quantize_a_row_i8(a_blk_len, (भिन्न *) ((वर्ण *) src1->data + i12 * nb12) + ak_blk_id * a_blk_len,
                                  a_blk_len, quant_a_buffer + i12 * nbw2 + ak_blk_id * block_stride_a);
            }
        } अण्णहा {
            प्रति (int64_t ii = ith; ii < ne12 * ne11; ii += nth) {
                int64_t i12 = ii / ne11;
                int64_t i11 = ii % ne11;
                quantize_a_row_i8(a_blk_len, (भिन्न *) ((वर्ण *) src1->data + i12 * nb12 + i11 * nb11), ne10,
                                  quant_a_buffer + i12 * nbw2 + i11 * nbw1);
            }
        }

#define MMID_MATRIX_ROW(row_id, i1) matrix_rows[(row_id) *ne12 + (i1)]

        int64_t *          matrix_row_counts       = (int64_t *) (ws_ptr + gemm_workspace_size);
        int32_t *          valid_ep_गणनम्          = (int32_t *) (matrix_row_counts + n_as);
        int32_t *          valid_act_गणनम्         = (int32_t *) (valid_ep_गणनम् + 1);
        int64_t *          valid_matrix_row_counts = (int64_t *) (valid_act_गणनम् + 1);
        mmid_row_mapping * matrix_rows             = (mmid_row_mapping *) (valid_matrix_row_counts + n_as);

        जइ (ith == 0) {
            // initialize matrix_row_counts
            memset(matrix_row_counts, 0, n_as * परिमाण(int64_t));

            // group rows by src0 matrix
            प्रति (int32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
                प्रति (int32_t id = 0; id < n_ids; ++id) {
                    ध्रुव int32_t i02 =
                        *(ध्रुव int32_t *) ((ध्रुव वर्ण *) ids->data + iid1 * ids->nb[1] + id * ids->nb[0]);

                    GGML_ASSERT(i02 >= 0 && i02 < n_as);

                    MMID_MATRIX_ROW(i02, matrix_row_counts[i02]) = { id, iid1 };
                    matrix_row_counts[i02] += 1;
                }
            }

            int32_t valid_ep_गणनम्_t  = 0;
            int32_t valid_act_गणनम्_t = 0;
            प्रति (पूर्णाङ्क cur_a = 0; cur_a < n_as; ++cur_a) {
                ध्रुव int64_t cne1 = matrix_row_counts[cur_a];
                जइ (cne1 == 0) {
                    अनुवर्त;
                }
                valid_matrix_row_counts[valid_ep_गणनम्_t] = cur_a;
                valid_act_गणनम्_t += cne1;
                valid_ep_गणनम्_t += 1;
            }
            valid_ep_गणनम्[0]  = valid_ep_गणनम्_t;
            valid_act_गणनम्[0] = valid_act_गणनम्_t;
        }

        ध्रुव int64_t barrier_idx = static_cast<int64_t>(ith / 2);

        GGML_ASSERT(global_spine_env_info.init_barrier != nullptr);
        GGML_ASSERT(barrier_idx < spine_init_barrier_गणनम्);
        spine_barrier_t * cur_barrier = &global_spine_env_info.init_barrier[barrier_idx];

        ggml_barrier(params->threadpool);

        ध्रुव size_t row_stride_b      = b_k_blks * get_repacked_block_type_size<BLOC_TYPE, INTER_SIZE, NB_COLS>();
        ध्रुव size_t expert_b_stride   = ne01 * row_stride_b;
        ध्रुव size_t per_nb_cols_wsize = NB_COLS * row_stride_b;

        std::array<ध्रुव uint8_t *, 2> src_workspaces;
        std::array<भिन्न *, 2>         dst_workspaces;

        auto *     tcm_buffer      = ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer;
        ध्रुव auto tcm_buffer_size = ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer_size;

        ध्रुव auto valid_ep_गणनम्_t  = valid_ep_गणनम्[0];
        ध्रुव auto valid_act_गणनम्_t = valid_act_गणनम्[0];

        पूर्णाङ्क nth_es = 1;
        पूर्णाङ्क nth_n  = nth;

        पूर्णाङ्क ith_es = ith % nth_es;
        पूर्णाङ्क ith_n  = (ith / nth_es) % nth_n;

        जइ (valid_ep_गणनम्_t % nth == 0 && tcm_buffer != nullptr && valid_ep_गणनम्_t == n_as &&
            valid_act_गणनम्_t == n_as && per_nb_cols_wsize <= tcm_buffer_size) {
            प्रति (int64_t valid_id = ith; valid_id < valid_ep_गणनम्_t; valid_id += nth) {
                ध्रुव int64_t cur_a = valid_matrix_row_counts[valid_id];

                auto * src0_cur = (uint8_t *) src0->data + cur_a * expert_b_stride;

                mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, 0);
                ध्रुव पूर्णाङ्क        id          = row_mapping.i1;
                ध्रुव int64_t    i11         = id % ne11;
                ध्रुव int64_t    i12         = row_mapping.i2;
                ध्रुव int64_t    i1          = id;
                ध्रुव int64_t    i2          = i12;

                auto *  src1_col = quant_a_buffer + (i11 * nbw1 + i12 * nbw2);
                भिन्न * c_blk    = (भिन्न *) ((वर्ण *) dst->data + (i1 * nb1 + i2 * nb2));

                uint8_t * a_row = src1_col;
                uint8_t * b_col = reinterpret_cast<uint8_t *>(tcm_buffer);
                जइ ((nbw1 + per_nb_cols_wsize) <= tcm_buffer_size) {
                    a_row = (uint8_t *) tcm_buffer;
                    b_col = reinterpret_cast<uint8_t *>(tcm_buffer) + nbw1;
                }
                uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? b_col : nullptr;

                जइ (ith % 2 == 0) {
                    spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(src0_cur), per_nb_cols_wsize);

                    जइ (a_row != src1_col) {
                        spacemit_kernels::rvv::memcpy1d(a_row, src1_col, nbw1);
                    }
                }

                spine_barrier_wait(cur_barrier);

                जइ (ith % 2 != 0) {
                    जइ (a_row != src1_col) {
                        spacemit_kernels::rvv::memcpy1d(a_row, src1_col, nbw1);
                    }

                    spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(src0_cur), per_nb_cols_wsize);
                }

                int64_t nb_real = std::min(ne01, NB_COLS);
                प्रति (int64_t ni = 0; ni < ne01; ni += NB_COLS) {
                    जइ (ith % 2 != 0) {
                        spine_barrier_wait(cur_barrier);
                    }

                    gemm_kernel(b_blk_len, a_row, b_col, b_col_zp, c_blk + ni, 1, nb_real, b_k_blks, ne01);

                    जइ (ith % 2 == 0) {
                        spine_barrier_wait(cur_barrier);
                    }

                    ध्रुव int64_t next_ni = ni + NB_COLS;
                    जइ (next_ni < ne01) {
                        nb_real = std::min(ne01 - next_ni, NB_COLS);
                        spacemit_kernels::rvv::memcpy1d(
                            b_col, reinterpret_cast<uint8_t *>(src0_cur) + next_ni * row_stride_b, per_nb_cols_wsize);
                    }
                }
            }
        } अण्णहा {
            प्रति (int64_t valid_id = ith_es; valid_id < valid_ep_गणनम्_t; valid_id += nth_es) {
                ध्रुव int64_t cur_a = valid_matrix_row_counts[valid_id];
                ध्रुव int64_t cne1  = matrix_row_counts[cur_a];

                int64_t src1_cur_start = 0;
                int64_t src1_cur_end   = cne1;

                int64_t src0_cur_start = (ith_n * ne01) / nth_n;
                int64_t src0_cur_end   = MIN(((ith_n + 1) * ne01) / nth_n, ne01);

                जइ (src1_cur_start >= src1_cur_end || src0_cur_start >= src0_cur_end) {
                    अनुवर्त;
                }

                src0_cur_start =
                    (src0_cur_start % NB_COLS) ? src0_cur_start + NB_COLS - (src0_cur_start % NB_COLS) : src0_cur_start;
                src0_cur_end =
                    (src0_cur_end % NB_COLS) ? src0_cur_end + NB_COLS - (src0_cur_end % NB_COLS) : src0_cur_end;

                auto *    src0_cur = (uint8_t *) src0->data + cur_a * expert_b_stride + src0_cur_start * row_stride_b;
                uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? src0_cur : nullptr;

                size_t extra_tcm_buffer_size = tcm_buffer_size;
                शून्य * extra_tcm_buffer      = tcm_buffer;
                जइ (tcm_buffer != nullptr && (src1_cur_end - src1_cur_start) >= 4 &&
                    (src0_cur_end - src0_cur_start) * row_stride_b <= tcm_buffer_size) {
                    spacemit_kernels::rvv::memcpy1d(tcm_buffer, src0_cur,
                                                    (src0_cur_end - src0_cur_start) * row_stride_b);
                    src0_cur = reinterpret_cast<uint8_t *>(tcm_buffer);
                    b_col_zp = block_type_has_zp<BLOC_TYPE>() ? src0_cur : nullptr;
                    extra_tcm_buffer_size -= (src0_cur_end - src0_cur_start) * row_stride_b;
                    extra_tcm_buffer = reinterpret_cast<शून्य *>(reinterpret_cast<uintptr_t>(tcm_buffer) +
                                                                (src0_cur_end - src0_cur_start) * row_stride_b);
                }

                पूर्णाङ्क ir1 = src1_cur_start;

                जइ (extra_tcm_buffer_size >= nbw1 && extra_tcm_buffer != nullptr) {
                    int64_t quant_a_tile_size = extra_tcm_buffer_size / nbw1;
                    करेहि {
                        quant_a_tile_size = MIN(quant_a_tile_size, src1_cur_end - ir1);

                        uint8_t * quant_a_tile_buffer = reinterpret_cast<uint8_t *>(extra_tcm_buffer);

                        पूर्णाङ्क iir1 = ir1;
                        प्रति (; iir1 < (ir1 + quant_a_tile_size); ++iir1) {
                            mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, iir1);

                            ध्रुव पूर्णाङ्क id = row_mapping.i1;  // selected expert index

                            ध्रुव int64_t i11 = id % ne11;
                            ध्रुव int64_t i12 = row_mapping.i2;  // row index in src1

                            auto * src1_col = quant_a_buffer + (i11 * nbw1 + i12 * nbw2);
                            spacemit_kernels::rvv::memcpy1d(quant_a_tile_buffer, src1_col, nbw1);
                            quant_a_tile_buffer = quant_a_tile_buffer + nbw1;
                        }

                        quant_a_tile_buffer = reinterpret_cast<uint8_t *>(extra_tcm_buffer);
                        iir1                = ir1;

                        जइ (moe_gemm_kernel_m2 != nullptr) {
                            प्रति (; iir1 < (ir1 + quant_a_tile_size - 1); iir1 += 2, quant_a_tile_buffer += 2 * nbw1) {
                                mmid_row_mapping row_mapping_0 = MMID_MATRIX_ROW(cur_a, iir1);
                                mmid_row_mapping row_mapping_1 = MMID_MATRIX_ROW(cur_a, iir1 + 1);

                                src_workspaces[0] = quant_a_tile_buffer;
                                src_workspaces[1] = quant_a_tile_buffer + nbw1;

                                dst_workspaces[0] =
                                    (भिन्न *) ((वर्ण *) dst->data + (row_mapping_0.i1 * nb1 + row_mapping_0.i2 * nb2)) +
                                    src0_cur_start;
                                dst_workspaces[1] = (भिन्न *) ((वर्ण *) dst->data +
                                                               ((row_mapping_1.i1) * nb1 + (row_mapping_1.i2) * nb2)) +
                                                    src0_cur_start;
                                moe_gemm_kernel_m2(b_blk_len, src_workspaces.data(), src0_cur, b_col_zp,
                                                   dst_workspaces.data(), 1, src0_cur_end - src0_cur_start, b_k_blks,
                                                   ne01);
                            }
                        }

                        प्रति (; iir1 < (ir1 + quant_a_tile_size); iir1++, quant_a_tile_buffer += nbw1) {
                            mmid_row_mapping row_mapping_0 = MMID_MATRIX_ROW(cur_a, iir1);

                            gemm_kernel(
                                b_blk_len, quant_a_tile_buffer, src0_cur, b_col_zp,
                                (भिन्न *) ((वर्ण *) dst->data + (row_mapping_0.i1 * nb1 + row_mapping_0.i2 * nb2)) +
                                    src0_cur_start,
                                1, src0_cur_end - src0_cur_start, b_k_blks, ne01);
                        }

                        ir1 += quant_a_tile_size;
                    } जाव (ir1 < src1_cur_end);
                } अण्णहा {
                    जइ (moe_gemm_kernel_m2 != nullptr) {
                        प्रति (; ir1 < src1_cur_end - 1; ir1 += 2) {
                            प्रति (पूर्णाङ्क iir1 = 0; iir1 < 2; ++iir1) {
                                mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, ir1 + iir1);

                                ध्रुव पूर्णाङ्क id = row_mapping.i1;  // selected expert index

                                ध्रुव int64_t i11 = id % ne11;
                                ध्रुव int64_t i12 = row_mapping.i2;  // row index in src1

                                ध्रुव int64_t i1 = id;               // selected expert index
                                ध्रुव int64_t i2 = i12;              // row

                                src_workspaces[iir1] = quant_a_buffer + (i11 * nbw1 + i12 * nbw2);

                                dst_workspaces[iir1] =
                                    (भिन्न *) ((वर्ण *) dst->data + (i1 * nb1 + i2 * nb2)) + src0_cur_start;
                            }

                            moe_gemm_kernel_m2(b_blk_len, src_workspaces.data(), src0_cur, b_col_zp,
                                               dst_workspaces.data(), 1, src0_cur_end - src0_cur_start, b_k_blks, ne01);
                        }
                    }

                    प्रति (; ir1 < src1_cur_end; ir1++) {
                        mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, ir1);

                        ध्रुव पूर्णाङ्क id = row_mapping.i1;  // selected expert index

                        ध्रुव int64_t i11 = id % ne11;
                        ध्रुव int64_t i12 = row_mapping.i2;  // row index in src1

                        ध्रुव int64_t i1 = id;               // selected expert index
                        ध्रुव int64_t i2 = i12;              // row

                        auto * src1_col = quant_a_buffer + (i11 * nbw1 + i12 * nbw2);

                        gemm_kernel(b_blk_len, src1_col, src0_cur, b_col_zp,
                                    (भिन्न *) ((वर्ण *) dst->data + (i1 * nb1 + i2 * nb2)) + src0_cur_start, 1,
                                    src0_cur_end - src0_cur_start, b_k_blks, ne01);
                    }
                }
            }
        }
#undef MMID_MATRIX_ROW
    }

    पूर्णाङ्क repack(ggml_tensor * t, ध्रुव शून्य * data, size_t data_size) override {
        GGML_LOG_DEBUG("%s: repack tensor %s with %s_%dx%d\n", __func__, t->name, ggml_type_name(t->type),
                       (पूर्णाङ्क) NB_COLS, (पूर्णाङ्क) INTER_SIZE);
        प्रत्यागम ggml::cpu::riscv64_spacemit::repack<BLOC_TYPE, INTER_SIZE, NB_COLS>(t, data, data_size);
    }
};

वर्ग tensor_traits_common : सार्वजनिक tensor_traits_base {
    तार्किक work_size(पूर्णाङ्क n_threads, ध्रुव ggml_tensor * op, size_t & size) override {
        विकल्प (op->op) {
            पक्ष GGML_OP_FLASH_ATTN_EXT:
                {
                    ध्रुव पूर्णाङ्क     n_tasks = n_threads;
                    ध्रुव int64_t neq2    = op->src[0]->ne[2];  // number of query heads
                    ध्रुव int64_t DK      = op->src[1]->ne[0];
                    ध्रुव int64_t DV      = op->src[2]->ne[0];  // DV

                    // Tiled flash attention scratch (tile sizes defined in common.h)
                    // Per-thread: Q_q + KQ + mask + VKQ32 + V32 + K_f32 + padding
                    size_t prefill = परिमाण(भिन्न) *
                                     (GGML_FA_TILE_Q * DK + 2 * GGML_FA_TILE_Q * GGML_FA_TILE_KV + GGML_FA_TILE_Q * DV +
                                      GGML_FA_TILE_KV * DV + GGML_FA_TILE_KV * DK) *
                                     n_tasks;

                    // Decode path: n_kv_chunks = n_tasks (one chunk per thread)
                    // Per-thread: VKQ accmulator (DV), partial M, partial S + intra-thread scratch for V, Q and VKQ
                    size_t n_chunks = n_tasks;
                    size_t decode   = परिमाण(भिन्न) * (neq2 * n_chunks * (2 + DV) + n_tasks * (DK + 2 * DV));

                    size = MAX(prefill, decode);
                }
                प्रत्यागम सच्च;
            स्वभाव:
                भङ्ग;
        }
        प्रत्यागम मिच्छ;
    }

    तार्किक compute_forward(ggml_compute_params * params, ggml_tensor * op) override {
        विकल्प (op->op) {
            पक्ष GGML_OP_NORM:
                विकल्प (op->src[0]->type) {
                    पक्ष GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_norm_f32(params, op);
                        प्रत्यागम सच्च;
                    स्वभाव:
                        GGML_ABORT("fatal error");
                }
            पक्ष GGML_OP_RMS_NORM:
                विकल्प (op->src[0]->type) {
                    पक्ष GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_rms_norm_f32(params, op);
                        प्रत्यागम सच्च;
                    स्वभाव:
                        GGML_ABORT("fatal error");
                }
            पक्ष GGML_OP_ADD:
                विकल्प (op->src[0]->type) {
                    पक्ष GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_द्विमान<GGML_OP_ADD, भिन्न>(params, op);
                        प्रत्यागम सच्च;
                    पक्ष GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_द्विमान<GGML_OP_ADD, _Float16>(params, op);
                        प्रत्यागम सच्च;
                    स्वभाव:
                        ggml_compute_forward_add(params, op);
                        प्रत्यागम सच्च;
                }
            पक्ष GGML_OP_SUB:
                विकल्प (op->src[0]->type) {
                    पक्ष GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_द्विमान<GGML_OP_SUB, भिन्न>(params, op);
                        प्रत्यागम सच्च;
                    पक्ष GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_द्विमान<GGML_OP_SUB, _Float16>(params, op);
                        प्रत्यागम सच्च;
                    स्वभाव:
                        ggml_compute_forward_sub(params, op);
                        प्रत्यागम सच्च;
                }
            पक्ष GGML_OP_MUL:
                विकल्प (op->src[0]->type) {
                    पक्ष GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_द्विमान<GGML_OP_MUL, भिन्न>(params, op);
                        प्रत्यागम सच्च;
                    पक्ष GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_द्विमान<GGML_OP_MUL, _Float16>(params, op);
                        प्रत्यागम सच्च;
                    स्वभाव:
                        ggml_compute_forward_mul(params, op);
                        प्रत्यागम सच्च;
                }
            पक्ष GGML_OP_DIV:
                विकल्प (op->src[0]->type) {
                    पक्ष GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_द्विमान<GGML_OP_DIV, भिन्न>(params, op);
                        प्रत्यागम सच्च;
                    पक्ष GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_द्विमान<GGML_OP_DIV, _Float16>(params, op);
                        प्रत्यागम सच्च;
                    स्वभाव:
                        ggml_compute_forward_div(params, op);
                        प्रत्यागम सच्च;
                }
            पक्ष GGML_OP_FLASH_ATTN_EXT:
                forward_flash_attn_ext_f16(params, op);
                प्रत्यागम सच्च;
            पक्ष GGML_OP_CONT:
                {
                    ध्रुव ggml_tensor * src0 = op->src[0];
                    जइ (op->type == src0->type && op->nb[0] != src0->nb[0] && op->nb[0] == src0->nb[1] &&
                        op->ne[3] * op->ne[2] * op->nb[2] == src0->ne[3] * src0->ne[2] * src0->nb[2]) {
                        spacemit_kernels::rvv::forward_cont_with_permute(params, op);
                    } अण्णहा {
                        ggml_compute_forward_cont(params, op);
                    }
                    प्रत्यागम सच्च;
                }
            पक्ष GGML_OP_CPY:
                {
                    ध्रुव ggml_tensor * src0 = op->src[0];
                    जइ (op->type == src0->type && op->nb[0] == src0->nb[1] && src0->nb[0] != src0->nb[1] &&
                        ggml_nelements(src0) == ggml_nelements(op)) {
                        spacemit_kernels::rvv::forward_cpy_with_permute(params, op);
                    } अण्णहा {
                        ggml_compute_forward_cpy(params, op);
                    }
                    प्रत्यागम सच्च;
                }
            पक्ष GGML_OP_REPEAT:
                {
                    ध्रुव तार्किक rows_equal         = ggml_nrows(op->src[0]) == ggml_nrows(op);
                    ध्रुव तार्किक broadcast_or_equal = op->src[0]->ne[0] == 1 || op->src[0]->ne[0] == op->ne[0];

                    जइ (rows_equal && broadcast_or_equal) {
                        विकल्प (op->src[0]->type) {
                            पक्ष GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_repeat_nrows<int32_t>(params, op);
                                प्रत्यागम सच्च;
                            पक्ष GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_repeat_nrows<int16_t>(params, op);
                                प्रत्यागम सच्च;
                            स्वभाव:
                                भङ्ग;
                        }
                    }

                    जइ (op->src[0]->ne[1] == 1 && op->src[0]->ne[0] == op->ne[0]) {
                        विकल्प (op->src[0]->type) {
                            पक्ष GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_repeat_dim1<int32_t>(params, op);
                                प्रत्यागम सच्च;
                            पक्ष GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_repeat_dim1<int16_t>(params, op);
                                प्रत्यागम सच्च;
                            स्वभाव:
                                भङ्ग;
                        }
                    }

                    ggml_compute_forward_repeat(params, op);
                }
                प्रत्यागम सच्च;
            पक्ष GGML_OP_SUM_ROWS:
                {
                    जइ (op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32) {
                        spacemit_kernels::rvv::forward_sum_rows<भिन्न>(params, op);
                    } अण्णहा {
                        ggml_compute_forward_sum_rows(params, op);
                    }
                }
                प्रत्यागम सच्च;
            पक्ष GGML_OP_GET_ROWS:
                {
                    जइ (op->src[0]->type == op->type) {
                        विकल्प (op->src[0]->type) {
                            पक्ष GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_get_rows<int32_t>(params, op);
                                प्रत्यागम सच्च;
                            पक्ष GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_get_rows<int16_t>(params, op);
                                प्रत्यागम सच्च;
                            स्वभाव:
                                भङ्ग;
                        }
                    }

                    ggml_compute_forward_get_rows(params, op);
                }
                प्रत्यागम सच्च;
            पक्ष GGML_OP_CONCAT:
                {
                    ध्रुव int32_t dim = ggml_get_op_params_i32(op, 0);
                    जइ (dim == 0 && op->type == op->src[0]->type) {
                        विकल्प (op->src[0]->type) {
                            पक्ष GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_concat<int32_t>(params, op);
                                प्रत्यागम सच्च;
                            पक्ष GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_concat<int16_t>(params, op);
                                प्रत्यागम सच्च;
                            स्वभाव:
                                भङ्ग;
                        }
                    }

                    ggml_compute_forward_concat(params, op);
                }
                प्रत्यागम सच्च;
            // TODO For GGML_OP_GATED_DELTA_NET
            // case GGML_OP_GATED_DELTA_NET:
            //     return true;
            स्वभाव:
                भङ्ग;
        }
        प्रत्यागम मिच्छ;
    }

    शून्य forward_flash_attn_ext_f16(ध्रुव ggml_compute_params * params, ggml_tensor * dst) {
        ध्रुव ggml_tensor * q = dst->src[0];
        ध्रुव ggml_tensor * k = dst->src[1];
        ध्रुव ggml_tensor * v = dst->src[2];

        GGML_TENSOR_LOCALS(int64_t, neq, q, ne)
        GGML_TENSOR_LOCALS(size_t, nbq, q, nb)
        GGML_TENSOR_LOCALS(int64_t, nek, k, ne)
        GGML_TENSOR_LOCALS(size_t, nbk, k, nb)
        GGML_TENSOR_LOCALS(int64_t, nev, v, ne)
        GGML_TENSOR_LOCALS(size_t, nbv, v, nb)
        GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
        GGML_TENSOR_LOCALS(size_t, nb, dst, nb)

        ध्रुव int64_t DK = nek0;
        ध्रुव int64_t DV = nev0;

        ध्रुव तार्किक supported_prec  = (dst->op_params[3] == GGML_PREC_F32 || dst->op_params[3] == GGML_PREC_DEFAULT);
        ध्रुव तार्किक supported_types = (q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16);
        ध्रुव तार्किक supported_shape = (DK > 0 && DK <= 128 && DV > 0 && DV <= 128);
        ध्रुव तार्किक supported_vlen  = (__riscv_vlenb() == 128);

        जइ (!(supported_prec && supported_types && supported_shape && supported_vlen)) {
            ggml_compute_forward_flash_attn_ext(params, dst);
            प्रत्यागम;
        }

        // total rows in q
        ध्रुव int64_t nr = neq1 * neq2 * neq3;

        // rows per thread
        ध्रुव पूर्णाङ्क ith = params->ith;
        ध्रुव पूर्णाङ्क nth = params->nth;

        स्थिर constexpr int64_t Q_TILE_SZ = ggml_fa_tile_config::Q;
        ध्रुव तार्किक               use_tiled = !params->use_ref && (neq1 >= Q_TILE_SZ);

        // 4x chunks per thread
        // int     nth_scaled = nth * 4;
        // int64_t chunk_size = (nr + nth_scaled - 1) / nth_scaled;
        // int64_t nchunk     = (nr + chunk_size - 1) / chunk_size;

        // if (nth == 1 || nchunk < nth) {
        //     nchunk = nth;
        // }

        int64_t nchunk = nth;

        जइ (ith == 0) {
            // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
            ggml_threadpool_chunk_set(params->threadpool, nth);
        }

        ggml_barrier(params->threadpool);

        // The number of elements in each chunk
        ध्रुव int64_t dr = (nr + nchunk - 1) / nchunk;

        // The first chunk comes from our thread_id, the rest will get auto-assigned.
        पूर्णाङ्क current_chunk = ith;

        जाव (current_chunk < nchunk) {
            ध्रुव int64_t ir0 = dr * current_chunk;
            ध्रुव int64_t ir1 = MIN(ir0 + dr, nr);

            जइ (use_tiled) {
                spacemit_kernels::rvv::forward_flash_attn_ext_f16_tiled_vlen1024_vf16(
                    params, dst, ir0, ir1, ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer,
                    ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer_size);
            } अण्णहा {
                spacemit_kernels::rvv::forward_flash_attn_ext_f16_one_chunk_vlen1024_vf16(
                    params, dst, ir0, ir1, ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer,
                    ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer_size);
            }

            current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
        }
    }

    पूर्णाङ्क repack(ggml_tensor * t, ध्रुव शून्य * data, size_t data_size) override {
        memcpy(t->data, data, data_size);
        प्रत्यागम 0;
    }
};

// Impl By IME1
स्थिर ध्रुव tensor_traits<block_q4_0, 32, 16>  q4_0_16x32_q8_0;
स्थिर ध्रुव tensor_traits<block_q4_1, 32, 16>  q4_1_16x32_q8_0;
स्थिर ध्रुव tensor_traits<block_q4_K, 32, 16>  q4_k_16x32_q8_0;
स्थिर ध्रुव tensor_traits<block_q8_0, 32, 16>  q8_0_16x32_q8_0;
// Impl By IME2
स्थिर ध्रुव tensor_traits<block_q2_K, 256, 32> q2_k_32x256_q8_0;
स्थिर ध्रुव tensor_traits<block_q3_K, 256, 32> q3_k_32x256_q8_0;
स्थिर ध्रुव tensor_traits<block_q4_0, 32, 32>  q4_0_32x32_q8_0;
स्थिर ध्रुव tensor_traits<block_q4_1, 32, 32>  q4_1_32x32_q8_0;
स्थिर ध्रुव tensor_traits<block_q4_0, 256, 32> q4_0_32x256_q8_0;
स्थिर ध्रुव tensor_traits<block_q4_1, 256, 32> q4_1_32x256_q8_0;
स्थिर ध्रुव tensor_traits<block_q4_K, 32, 32>  q4_k_32x32_q8_0;
स्थिर ध्रुव tensor_traits<block_q6_K, 32, 32>  q6_k_32x32_q8_0;
स्थिर ध्रुव tensor_traits<block_q8_0, 32, 32>  q8_0_32x32_q8_0;
स्थिर ध्रुव tensor_traits<block_mxfp4, 32, 32> mxfp4_32x32_q8_0;
स्थिर ध्रुव tensor_traits<block_q5_K, 32, 32>  q5_k_32x32_q8_0;
स्थिर ध्रुव tensor_traits<block_q5_1, 32, 32>  q5_1_32x32_q8_0;
स्थिर ध्रुव tensor_traits<block_q5_0, 32, 32>  q5_0_32x32_q8_0;
// Impl By RVV
स्थिर ध्रुव tensor_traits_common               rvv_impl;

}  // namespace ggml::cpu::riscv64_spacemit

स्थिर ध्रुव ggml::cpu::tensor_traits * ggml_riscv64_spacemit_get_optimal_repack_type(ध्रुव ggml_tensor * cur) {
    विकल्प (cur->type) {
        पक्ष GGML_TYPE_Q2_K:
            {
#जइ defined(RISCV64_SPACEMIT_IME2)
                जइ (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    प्रत्यागम &ggml::cpu::riscv64_spacemit::q2_k_32x256_q8_0;
                }
#endif
            }
            भङ्ग;
        पक्ष GGML_TYPE_Q3_K:
            {
#जइ defined(RISCV64_SPACEMIT_IME2)
                जइ (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    प्रत्यागम &ggml::cpu::riscv64_spacemit::q3_k_32x256_q8_0;
                }
#endif
            }
            भङ्ग;
        पक्ष GGML_TYPE_Q4_0:
            {
#जइ defined(RISCV64_SPACEMIT_IME2)
                जइ (cur->ne[1] % 32 == 0 && cur->ne[0] % 256 == 0 &&
                    (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    प्रत्यागम &ggml::cpu::riscv64_spacemit::q4_0_32x256_q8_0;
                }

                जइ (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    प्रत्यागम &ggml::cpu::riscv64_spacemit::q4_0_32x32_q8_0;
                }
#endif

#जइ defined(RISCV64_SPACEMIT_IME1)
                जइ (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    प्रत्यागम &ggml::cpu::riscv64_spacemit::q4_0_16x32_q8_0;
                }
#endif
            }
            भङ्ग;
        पक्ष GGML_TYPE_Q4_1:
            {
#जइ defined(RISCV64_SPACEMIT_IME2)
                // TODO
                // if (cur->ne[1] % 32 == 0 && cur->ne[0] % 256 == 0 &&
                //     (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                //     return &ggml::cpu::riscv64_spacemit::q4_1_32x256_q8_0;
                // }

                जइ (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    प्रत्यागम &ggml::cpu::riscv64_spacemit::q4_1_32x32_q8_0;
                }
#endif

#जइ defined(RISCV64_SPACEMIT_IME1)
                जइ (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    प्रत्यागम &ggml::cpu::riscv64_spacemit::q4_1_16x32_q8_0;
                }
#endif
            }
            भङ्ग;
        पक्ष GGML_TYPE_Q4_K:
            {
#जइ defined(RISCV64_SPACEMIT_IME2)
                जइ (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    प्रत्यागम &ggml::cpu::riscv64_spacemit::q4_k_32x32_q8_0;
                }
#endif

#जइ defined(RISCV64_SPACEMIT_IME1)
                जइ (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    प्रत्यागम &ggml::cpu::riscv64_spacemit::q4_k_16x32_q8_0;
                }
#endif
            }
            भङ्ग;
        पक्ष GGML_TYPE_Q6_K:
            {
#जइ defined(RISCV64_SPACEMIT_IME2)
                जइ ((ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    प्रत्यागम &ggml::cpu::riscv64_spacemit::q6_k_32x32_q8_0;
                }
#endif
            }
            भङ्ग;
        पक्ष GGML_TYPE_Q8_0:
            {
#जइ defined(RISCV64_SPACEMIT_IME2)
                जइ ((ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    प्रत्यागम &ggml::cpu::riscv64_spacemit::q8_0_32x32_q8_0;
                }
#endif

#जइ defined(RISCV64_SPACEMIT_IME1)
                जइ (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    प्रत्यागम &ggml::cpu::riscv64_spacemit::q8_0_16x32_q8_0;
                }
#endif
            }
            भङ्ग;
        पक्ष GGML_TYPE_MXFP4:
            {
#जइ defined(RISCV64_SPACEMIT_IME2)
                // TODO
                // if (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                //     return &ggml::cpu::riscv64_spacemit::mxfp4_32x32_q8_0;
                // }
#endif
            }
            भङ्ग;
        पक्ष GGML_TYPE_Q5_K:
            {
#जइ defined(RISCV64_SPACEMIT_IME2)
                जइ (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    प्रत्यागम &ggml::cpu::riscv64_spacemit::q5_k_32x32_q8_0;
                }
#endif
            }
            भङ्ग;
        पक्ष GGML_TYPE_Q5_1:
            {
#जइ defined(RISCV64_SPACEMIT_IME2)
                जइ (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    प्रत्यागम &ggml::cpu::riscv64_spacemit::q5_1_32x32_q8_0;
                }
#endif
            }
            भङ्ग;
        पक्ष GGML_TYPE_Q5_0:
            {
#जइ defined(RISCV64_SPACEMIT_IME2)
                जइ (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    प्रत्यागम &ggml::cpu::riscv64_spacemit::q5_0_32x32_q8_0;
                }
#endif
            }
            भङ्ग;
        स्वभाव:
            भङ्ग;
    }

    प्रत्यागम nullptr;
}

स्थिर गणना ggml_status ggml_backend_riscv64_spacemit_buffer_init_tensor(ggml_backend_buffer_t buffer,
                                                                         ggml_tensor *         tensor) {
    tensor->extra =
        (शून्य *) const_cast<ggml::cpu::tensor_traits *>(ggml_riscv64_spacemit_get_optimal_repack_type(tensor));

    GGML_UNUSED(buffer);

    प्रत्यागम GGML_STATUS_SUCCESS;
}

स्थिर शून्य ggml_backend_riscv64_spacemit_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);

    शून्य * base = buffer->context;
    जइ (base == nullptr) {
        प्रत्यागम;
    }

    ggml::cpu::riscv64_spacemit::spine_mem_pool_free(base);
}

स्थिर शून्य * ggml_backend_riscv64_spacemit_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);

    शून्य * base = buffer->context;
    GGML_ASSERT(base != nullptr);
    प्रत्यागम base;
}

स्थिर शून्य ggml_backend_riscv64_spacemit_buffer_memset_tensor(ggml_backend_buffer_t buffer,
                                                               ggml_tensor *         tensor,
                                                               uint8_t               value,
                                                               size_t                offset,
                                                               size_t                size) {
    GGML_ASSERT(tensor);
    memset((वर्ण *) tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

स्थिर शून्य ggml_backend_riscv64_spacemit_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);

    शून्य * base = buffer->context;
    GGML_ASSERT(base != nullptr);
    memset(base, value, buffer->size);
}

स्थिर शून्य ggml_backend_riscv64_spacemit_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                            ggml_tensor *         tensor,
                                                            ध्रुव शून्य *          data,
                                                            size_t                offset,
                                                            size_t                size) {
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    auto tensor_traits = (ggml::cpu::riscv64_spacemit::tensor_traits_base *) tensor->extra;
    जइ (tensor_traits) {
        auto OK = tensor_traits->repack(tensor, data, size);
        GGML_ASSERT(OK == 0);
    }

    GGML_UNUSED(buffer);
}

स्थिर ध्रुव ggml_backend_buffer_i ggml_backend_riscv64_spacemit_buffer_i = {
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

स्थिर ध्रुव वर्ण * ggml_backend_cpu_riscv64_spacemit_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    प्रत्यागम "CPU_RISCV64_SPACEMIT";

    GGML_UNUSED(buft);
}

स्थिर ggml_backend_buffer_t ggml_backend_cpu_riscv64_spacemit_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                                                                        size_t size) {
    शून्य * base = ggml::cpu::riscv64_spacemit::spine_mem_pool_alloc(size, 64);
    जइ (base == nullptr) {
        प्रत्यागम nullptr;
    }

    प्रत्यागम ggml_backend_buffer_init(buft, ggml_backend_riscv64_spacemit_buffer_i, base, size);
}

स्थिर size_t ggml_backend_cpu_riscv64_spacemit_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    प्रत्यागम 64;

    GGML_UNUSED(buft);
}

स्थिर size_t ggml_backend_cpu_riscv64_spacemit_nbytes(ggml_backend_buffer_type_t buft, ध्रुव ggml_tensor * tensor) {
    प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_DIMS; ++i) {
        जइ (tensor->ne[i] <= 0) {
            प्रत्यागम 0;
        }
    }

    GGML_UNUSED(buft);

    ध्रुव auto plain_nbytes = [&]() {
        size_t total = ggml_type_size(tensor->type);
        प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_DIMS; ++i) {
            total += (tensor->ne[i] - 1) * tensor->nb[i];
        }
        प्रत्यागम total;
    };

    ध्रुव size_t blck_size = ggml_blck_size(tensor->type);
    जइ (blck_size == 1) {
        प्रत्यागम plain_nbytes();
    }

    ध्रुव size_t row_nbytes = tensor->ne[0] * tensor->nb[0] / blck_size;

    ध्रुव auto add_strided_nbytes = [&](size_t total, size_t src_block_size, size_t dst_block_size) {
        प्रति (पूर्णाङ्क i = 1; i < GGML_MAX_DIMS; ++i) {
            total += (tensor->ne[i] - 1) * (tensor->nb[i] / src_block_size) * dst_block_size;
        }
        प्रत्यागम total;
    };

    ध्रुव auto remap_block_nbytes = [&](size_t src_block_size, size_t dst_block_size, int64_t padded_rows = 0) {
        GGML_ASSERT(row_nbytes % src_block_size == 0);

        size_t total =
            add_strided_nbytes((row_nbytes / src_block_size) * dst_block_size, src_block_size, dst_block_size);

        जइ (padded_rows > 0 && tensor->ne[1] % padded_rows != 0) {
            total += (padded_rows - tensor->ne[1] % padded_rows) * (tensor->nb[1] / src_block_size) * dst_block_size;
        }

        प्रत्यागम total;
    };

    size_t nbytes = row_nbytes;
    विकल्प (tensor->type) {
        पक्ष GGML_TYPE_Q4_K:
            nbytes = remap_block_nbytes(परिमाण(block_q4_K), परिमाण(block_q4_1) * 8);
            भङ्ग;
        पक्ष GGML_TYPE_Q6_K:
            nbytes = remap_block_nbytes(परिमाण(block_q6_K), परिमाण(block_q8_0) * 8, 32);
            भङ्ग;
        पक्ष GGML_TYPE_Q8_0:
            nbytes = remap_block_nbytes(परिमाण(block_q8_0), परिमाण(block_q8_0), 32);
            भङ्ग;
        पक्ष GGML_TYPE_Q2_K:
            nbytes = remap_block_nbytes(परिमाण(block_q2_K), परिमाण(spacemit_kernels::nrow_block_q2_k<1>));
            भङ्ग;
        पक्ष GGML_TYPE_Q3_K:
            nbytes = remap_block_nbytes(परिमाण(block_q3_K), परिमाण(spacemit_kernels::nrow_block_q3_k<1>));
            भङ्ग;
        पक्ष GGML_TYPE_MXFP4:
            nbytes = remap_block_nbytes(परिमाण(block_mxfp4), परिमाण(spacemit_kernels::nrow_block_mxfp4<1>));
            भङ्ग;
        पक्ष GGML_TYPE_Q5_K:
            nbytes = remap_block_nbytes(परिमाण(block_q5_K), परिमाण(spacemit_kernels::nrow_block_q5_1<1>) * 8);
            भङ्ग;
        पक्ष GGML_TYPE_Q5_1:
            nbytes = remap_block_nbytes(परिमाण(block_q5_1), परिमाण(spacemit_kernels::nrow_block_q5_1<1>));
            भङ्ग;
        पक्ष GGML_TYPE_Q5_0:
            nbytes = remap_block_nbytes(परिमाण(block_q5_0), परिमाण(spacemit_kernels::nrow_block_q5_0<1>));
            भङ्ग;
        स्वभाव:
            nbytes = add_strided_nbytes(row_nbytes, 1, 1);
            भङ्ग;
    }

    प्रत्यागम nbytes;
}

नामाकाश ggml::cpu::riscv64_spacemit {

वर्ग extra_buffer_type : ggml::cpu::extra_buffer_type {
    तार्किक supports_op(ggml_backend_dev_t, ध्रुव ggml_tensor * op) override {
        विकल्प (op->op) {
            पक्ष GGML_OP_MUL_MAT:
                जइ (op->src[0]->buffer && (ggml_n_dims(op->src[0]) == 2) &&
                    op->src[0]->buffer->buft == ggml_backend_cpu_riscv64_spacemit_buffer_type() &&
                    ggml_riscv64_spacemit_get_optimal_repack_type(op->src[0])) {
                    जइ (op->src[1]->buffer && !ggml_backend_buft_is_host(op->src[1]->buffer->buft)) {
                        प्रत्यागम मिच्छ;
                    }
                    जइ (op->src[1]->type == GGML_TYPE_F32) {
                        प्रत्यागम सच्च;
                    }
                }
                भङ्ग;
            पक्ष GGML_OP_MUL_MAT_ID:
                जइ (op->src[0]->buffer && (ggml_n_dims(op->src[0]) == 3) &&
                    op->src[0]->buffer->buft == ggml_backend_cpu_riscv64_spacemit_buffer_type() &&
                    ggml_riscv64_spacemit_get_optimal_repack_type(op->src[0])) {
                    जइ (op->src[1]->buffer && !ggml_backend_buft_is_host(op->src[1]->buffer->buft)) {
                        प्रत्यागम मिच्छ;
                    }
                    जइ (op->src[1]->type == GGML_TYPE_F32) {
                        प्रत्यागम सच्च;
                    }
                }
                भङ्ग;
            स्वभाव:
                // GGML_ABORT("fatal error");
                भङ्ग;
        }
        प्रत्यागम मिच्छ;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(ध्रुव ggml_tensor * op) override {
        विकल्प (op->op) {
            पक्ष GGML_OP_MUL_MAT:
            पक्ष GGML_OP_MUL_MAT_ID:
                जइ (op->src[0]->buffer && op->src[0]->buffer->buft == ggml_backend_cpu_riscv64_spacemit_buffer_type()) {
                    प्रत्यागम (ggml::cpu::tensor_traits *) op->src[0]->extra;
                }
                भङ्ग;
            पक्ष GGML_OP_NORM:
            पक्ष GGML_OP_RMS_NORM:
            पक्ष GGML_OP_ADD:
            पक्ष GGML_OP_SUB:
            पक्ष GGML_OP_MUL:
            पक्ष GGML_OP_DIV:
            पक्ष GGML_OP_FLASH_ATTN_EXT:
            पक्ष GGML_OP_CONT:
            पक्ष GGML_OP_CPY:
            पक्ष GGML_OP_REPEAT:
            पक्ष GGML_OP_SUM_ROWS:
            पक्ष GGML_OP_GET_ROWS:
            पक्ष GGML_OP_CONCAT:
                // case GGML_OP_GATED_DELTA_NET:
                प्रत्यागम (ggml::cpu::tensor_traits *) (&ggml::cpu::riscv64_spacemit::rvv_impl);
            स्वभाव:
                // GGML_ABORT("fatal error");
                भङ्ग;
        }

        प्रत्यागम nullptr;
    }
};

}  // namespace ggml::cpu::riscv64_spacemit

ggml_backend_buffer_type_t ggml_backend_cpu_riscv64_spacemit_buffer_type(शून्य) {
    स्थिर ggml_backend_buffer_type ggml_backend_cpu_buffer_type_riscv64_spacemit = {
  /* .iface    = */
        {
         /* .get_name           = */ ggml_backend_cpu_riscv64_spacemit_buffer_type_get_name,
         /* .alloc_buffer       = */ ggml_backend_cpu_riscv64_spacemit_buffer_type_alloc_buffer,
         /* .alloc_buffer_n     = */ रिक्त,
         /* .get_alignment      = */ ggml_backend_cpu_riscv64_spacemit_buffer_type_get_alignment,
         /* .get_max_size       = */ nullptr,
         /* .get_alloc_size     = */ ggml_backend_cpu_riscv64_spacemit_nbytes,
         /* .get_alloc_size_n   = */ रिक्त,
         /* .is_host            = */ nullptr,
         },
 /* .device  = */
        ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
 /* .context = */
        नूतन ggml::cpu::riscv64_spacemit::extra_buffer_type(),
    };

    प्रत्यागम &ggml_backend_cpu_buffer_type_riscv64_spacemit;
}

बाह्य "C" {
स्थिर पूर्णाङ्क bind_ai_thread() {
    पूर्णाङ्क  fd, bytes;
    वर्ण str[32];

    fd = open("/proc/set_ai_thread", O_WRONLY);
    जइ (fd < 0) {
        GGML_LOG_ERROR("try open /proc/set_ai_thread failed\n");
        प्रत्यागम -1;
    }

    snprintf(str, 16, "%d", 0);
    bytes = लिख(fd, str, strlen(str));
    जइ (bytes < 0) {
        GGML_LOG_ERROR("try write /proc/set_ai_thread failed\n");
        close(fd);
        प्रत्यागम -1;
    }

    close(fd);
    प्रत्यागम 0;
}

शून्य ggml_backend_cpu_riscv64_spacemit_set_numa_thread_affinity(पूर्णाङ्क thread_n) {
    पूर्णाङ्क cpu_id = sched_getcpu();
    जइ (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2 &&
        !((1 << cpu_id) & ggml::cpu::riscv64_spacemit::global_spine_env_info.cpu_mask)) {
        GGML_PRINT_DEBUG("bind_ai_thread for thread %d, pid %d\n", thread_n, getpid());
        bind_ai_thread();
    }

    जइ (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_tcm &&
        ggml::cpu::riscv64_spacemit::tls_context.cpu_id == -1) {
        CPU_ZERO(&(ggml::cpu::riscv64_spacemit::tls_context.cpuset));
        pthread_t    main_thread     = pthread_self();
        ध्रुव auto & perfer_core_ids = ggml::cpu::riscv64_spacemit::global_spine_env_info.perfer_core_ids;
        जइ (thread_n < 0 || static_cast<size_t>(thread_n) >= perfer_core_ids.size()) {
            GGML_ABORT("thread_n %d exceeds perfer_core_ids size %zu\n", thread_n, perfer_core_ids.size());
        }
        auto perfer_cpu_id = perfer_core_ids[static_cast<size_t>(thread_n)];
        CPU_SET(perfer_cpu_id, &(ggml::cpu::riscv64_spacemit::tls_context.cpuset));
        पूर्णाङ्क s =
            pthread_setaffinity_np(main_thread, परिमाण(cpu_set_t), &(ggml::cpu::riscv64_spacemit::tls_context.cpuset));
        जइ (s != 0) {
            GGML_ABORT("set thread affinity error for thread_n %d, cpu_id %d\n", thread_n, perfer_cpu_id);
        }

        पूर्णाङ्क ai_cpu_id = perfer_cpu_id - ggml::cpu::riscv64_spacemit::global_spine_env_info.aicpu_id_offset;
        ggml::cpu::riscv64_spacemit::tls_context.cpu_id = ai_cpu_id;
        ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer =
            ggml::cpu::riscv64_spacemit::spine_mem_pool_tcm_mem_get(ai_cpu_id);
        ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer_size =
            ggml::cpu::riscv64_spacemit::global_spine_env_info.tcm_blk_size;
    }

    जइ (ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer != nullptr) {
        शून्य * rt =
            ggml::cpu::riscv64_spacemit::spine_mem_pool_tcm_mem_wait(ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        जइ (rt == nullptr) {
            GGML_ABORT("wait tcm buffer failed for cpu_id: %d", ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        }
    }
}

शून्य ggml_backend_cpu_riscv64_spacemit_clear_numa_thread_affinity_threaded(पूर्णाङ्क thread_n) {
    जइ (ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer != nullptr) {
        auto rt = ggml::cpu::riscv64_spacemit::spine_mem_pool_tcm_mem_release(
            ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        जइ (rt != 0) {
            GGML_ABORT("release tcm buffer failed for cpu_id: %d", ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        }
    }
}
}
