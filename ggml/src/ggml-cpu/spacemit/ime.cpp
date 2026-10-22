#define GGML_COMMON_IMPL_CPP
#define GGML_COMMON_DECL_CPP

#અન્તર્ભાવ "ime.h"

#અન્તર્ભાવ "binary-ops.h"
#અન્તર્ભાવ "common.h"
#અન્તર્ભાવ "ggml-backend-impl.h"
#અન્તર્ભાવ "ggml-common.h"
#અન્તર્ભાવ "ggml-cpu.h"
#અન્તર્ભાવ "ime_env.h"
#અન્તર્ભાવ "ime_kernels.h"
#અન્તર્ભાવ "ops.h"
#અન્તર્ભાવ "repack.h"
#અન્તર્ભાવ "rvv_kernels.h"
#અન્તર્ભાવ "spine_mem_pool.h"
#અન્તર્ભાવ "traits.h"
#અન્તર્ભાવ "vec.h"

#અન્તર્ભાવ <fcntl.h>
#અન્તર્ભાવ <sys/mman.h>
#અન્તર્ભાવ <unistd.h>

#અન્તર્ભાવ <algorithm>
#અન્તર્ભાવ <atomic>
#અન્તર્ભાવ <cassert>
#અન્તર્ભાવ <cerrno>
#અન્તર્ભાવ <cmath>
#અન્તર્ભાવ <cstdio>  // for GGML_ASSERT
#અન્તર્ભાવ <stdexcept>
#અન્તર્ભાવ <thread>
// clang-format off
#જો defined(__riscv)

#જો !defined(__riscv_v) || !defined(__riscv_v_intrinsic)
#error "riscv v extension or v_intrinsic not enabled"
#નહીંતર
#અન્તર્ભાવ <riscv_vector.h>
#endif

#જો !defined(__riscv_zfh) || !defined(__riscv_zvfh)
#error "riscv zfh extension not enabled, GGML_RV_ZFH and GGML_RV_ZVFH must be defined to 1"
#endif

#જો !defined(__riscv_zba)
#error "riscv zba extension not enabled, GGML_RV_ZBA must be defined to 1"
#endif

#જો defined(RISCV64_SPACEMIT_IME1) || defined(RISCV64_SPACEMIT_IME2)
#નહીંતર
#error "RISCV64_SPACEMIT_IME1 or RISCV64_SPACEMIT_IME2 not defined"
#endif

#નહીંતર

#error "riscv not enabled in this build"

#endif

#જો defined(__GNUC__)
#pragma GCC diagnostic ignored "-Woverlength-strings"
#pragma GCC diagnostic ignored "-Wcast-qual"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

// clang-format on

બાહ્ય "C" {
બાહ્ય શૂન્ય ggml_threadpool_chunk_set(સંરચના ggml_threadpool * tp, પૂર્ણાંક value);
બાહ્ય પૂર્ણાંક  ggml_threadpool_chunk_add(સંરચના ggml_threadpool * tp, પૂર્ણાંક value);
}

નામાકાશ ggml::cpu::riscv64_spacemit {

સંરચના TLSContext {
    પૂર્ણાંક       cpu_id{ -1 };
    cpu_set_t cpuset;
    શૂન્ય *    tcm_buffer{ nullptr };
    size_t    tcm_buffer_size{ 0 };
};

thread_local TLSContext tls_context;

template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS> constexpr size_t get_repacked_block_type_size() {
    જો constexpr (std::is_same_v<BLOC_TYPE, block_q6_K> || std::is_same_v<BLOC_TYPE, block_q8_0>) {
        પાછા કદ(block_q8_0);
    } નહીંતર જો constexpr (std::is_same_v<BLOC_TYPE, block_q4_0>) {
        પાછા કદ(block_q4_0) * INTER_SIZE / QK4_0;
    } નહીંતર જો constexpr (std::is_same_v<BLOC_TYPE, block_q4_1> || std::is_same_v<BLOC_TYPE, block_q4_K>) {
        પાછા (કદ(block_q4_0) + કદ(uint8_t)) * INTER_SIZE / QK4_1;
    } નહીંતર જો constexpr (std::is_same_v<BLOC_TYPE, block_q2_K>) {
        પાછા કદ(spacemit_kernels::nrow_block_q2_k<1>);
    } નહીંતર જો constexpr (std::is_same_v<BLOC_TYPE, block_q3_K>) {
        પાછા કદ(spacemit_kernels::nrow_block_q3_k<1>);
    } નહીંતર જો constexpr (std::is_same_v<BLOC_TYPE, block_mxfp4>) {
        પાછા કદ(spacemit_kernels::nrow_block_mxfp4<1>);
    } નહીંતર જો constexpr (std::is_same_v<BLOC_TYPE, block_q5_1> || std::is_same_v<BLOC_TYPE, block_q5_K>) {
        પાછા કદ(spacemit_kernels::nrow_block_q5_1<1>);
    } નહીંતર જો constexpr (std::is_same_v<BLOC_TYPE, block_q5_0>) {
        પાછા કદ(spacemit_kernels::nrow_block_q5_0<1>);
    } નહીંતર {
        પ્રતિજ્ઞા(ખોટું);
        પાછા 0;
    }
}

template <typename BLOC_TYPE> constexpr તાર્કિક block_type_has_zp() {
    જો constexpr (std::is_same_v<BLOC_TYPE, block_q6_K> || std::is_same_v<BLOC_TYPE, block_q8_0> ||
                  std::is_same_v<BLOC_TYPE, block_q3_K> || std::is_same_v<BLOC_TYPE, block_q4_0> ||
                  std::is_same_v<BLOC_TYPE, block_mxfp4> || std::is_same_v<BLOC_TYPE, block_q5_0>) {
        પાછા ખોટું;
    } નહીંતર જો constexpr (std::is_same_v<BLOC_TYPE, block_q4_1> || std::is_same_v<BLOC_TYPE, block_q4_K> ||
                         std::is_same_v<BLOC_TYPE, block_q2_K> || std::is_same_v<BLOC_TYPE, block_q5_1> ||
                         std::is_same_v<BLOC_TYPE, block_q5_K>) {
        પાછા સાચું;
    } નહીંતર {
        પ્રતિજ્ઞા(ખોટું);
        પાછા ખોટું;
    }
}

વર્ગ tensor_traits_base : જાહેર ggml::cpu::tensor_traits {
  જાહેર:
    આભાસી પૂર્ણાંક repack(ggml_tensor * t, અચલ શૂન્ય * data, size_t data_size) = 0;
};

template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS> વર્ગ tensor_traits : જાહેર tensor_traits_base {
    તાર્કિક work_size(પૂર્ણાંક /* n_threads */, અચલ ggml_tensor * op, size_t & size) override {
        પસંદગી (op->op) {
            સ્થિતિ GGML_OP_MUL_MAT:
                {
                    int64_t src1_nelements = ggml_nelements(op->src[1]);

                    જો constexpr (std::is_same_v<BLOC_TYPE, block_q2_K> || std::is_same_v<BLOC_TYPE, block_q3_K>) {
                        size =
                            spacemit_kernels::div_round_up(src1_nelements, QK_K) * spacemit_kernels::q8k_blk_size(QK_K);
                    } નહીંતર જો constexpr (INTER_SIZE == QK4_0) {
                        size = spacemit_kernels::div_round_up(src1_nelements, QK4_0) *
                               spacemit_kernels::q8_blk_size(QK4_0, સાચું);
                    } નહીંતર જો constexpr (INTER_SIZE == 256) {
                        size = spacemit_kernels::div_round_up(src1_nelements, 256) *
                               spacemit_kernels::q8_hp_blk_size(256, સાચું, સાચું);
                    } નહીંતર {
                        GGML_ABORT("unsupported block type");
                    }

                    size = GGML_PAD(size, કદ(int64_t));

                    પાછા સાચું;
                }
            સ્થિતિ GGML_OP_MUL_MAT_ID:
                {
                    int64_t src1_nelements = ggml_nelements(op->src[1]);

                    જો constexpr (std::is_same_v<BLOC_TYPE, block_q2_K> || std::is_same_v<BLOC_TYPE, block_q3_K>) {
                        size =
                            spacemit_kernels::div_round_up(src1_nelements, QK_K) * spacemit_kernels::q8k_blk_size(QK_K);
                    } નહીંતર જો constexpr (INTER_SIZE == QK4_0) {
                        size = spacemit_kernels::div_round_up(src1_nelements, QK4_0) *
                               spacemit_kernels::q8_blk_size(QK4_0, સાચું);
                    } નહીંતર જો constexpr (INTER_SIZE == 256) {
                        size = spacemit_kernels::div_round_up(src1_nelements, 256) *
                               spacemit_kernels::q8_hp_blk_size(256, સાચું, સાચું);
                    } નહીંતર {
                        GGML_ABORT("unsupported block type");
                    }

                    size = GGML_PAD(size, કદ(int64_t));

                    અચલ int64_t ne02 = op->src[0]->ne[2];  // n_as, n_expert
                    અચલ int64_t ne12 = op->src[1]->ne[2];  // n_tokens

                    અચલ size_t sizeof_mmid_row_mapping = કદ(int64_t);
                    size += sizeof_mmid_row_mapping * ne02 * (ne12 + 1) + (ne02 + 1) * કદ(int64_t);

                    size = GGML_PAD(size, કદ(int64_t));

                    પાછા સાચું;
                }
            મૂળભૂત:
                // GGML_ABORT("fatal error");
                તોડો;
        }
        પાછા ખોટું;
    }

    તાર્કિક compute_forward(ggml_compute_params * params, ggml_tensor * op) override {
        પસંદગી (op->op) {
            સ્થિતિ GGML_OP_MUL_MAT:
                પસંદગી (op->src[0]->type) {
                    સ્થિતિ GGML_TYPE_Q2_K:
                    સ્થિતિ GGML_TYPE_Q3_K:
                    સ્થિતિ GGML_TYPE_Q4_0:
                    સ્થિતિ GGML_TYPE_Q4_1:
                    સ્થિતિ GGML_TYPE_Q4_K:
                    સ્થિતિ GGML_TYPE_Q6_K:
                    સ્થિતિ GGML_TYPE_Q8_0:
                    સ્થિતિ GGML_TYPE_Q5_0:
                    સ્થિતિ GGML_TYPE_Q5_1:
                    સ્થિતિ GGML_TYPE_Q5_K:
                        //case GGML_TYPE_MXFP4:
                        forward_mul_mat(params, op);
                        પાછા સાચું;
                    મૂળભૂત:
                        // GGML_ABORT("fatal error: unsupported type for src0 in MUL_MAT");
                        પાછા ખોટું;
                }
                તોડો;
            સ્થિતિ GGML_OP_MUL_MAT_ID:
                પસંદગી (op->src[0]->type) {
                    સ્થિતિ GGML_TYPE_Q2_K:
                    સ્થિતિ GGML_TYPE_Q3_K:
                    સ્થિતિ GGML_TYPE_Q4_0:
                    સ્થિતિ GGML_TYPE_Q4_1:
                    સ્થિતિ GGML_TYPE_Q4_K:
                    સ્થિતિ GGML_TYPE_Q6_K:
                    સ્થિતિ GGML_TYPE_Q8_0:
                    સ્થિતિ GGML_TYPE_Q5_0:
                    સ્થિતિ GGML_TYPE_Q5_1:
                    સ્થિતિ GGML_TYPE_Q5_K:
                        //case GGML_TYPE_MXFP4:
                        forward_mul_mat_id(params, op);
                        પાછા સાચું;
                    મૂળભૂત:
                        // GGML_ABORT("fatal error: unsupported type for src0 in MUL_MAT_ID");
                        પાછા ખોટું;
                }
                તોડો;
            મૂળભૂત:
                // GGML_ABORT("fatal error");
                તોડો;
        }
        પાછા ખોટું;
    }

    શૂન્ય forward_mul_mat(ggml_compute_params * params, ggml_tensor * op) {
        constexpr size_t a_blk_len = INTER_SIZE;
        constexpr size_t b_blk_len = INTER_SIZE;

        અચલ ggml_tensor * src0 = op->src[0];
        અચલ ggml_tensor * src1 = op->src[1];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_દ્વિઅંકી_OP_LOCALS

        પૂર્ણાંક ith = params->ith;
        પૂર્ણાંક nth = params->nth;

        [[maybe_unused]] અચલ ગણના ggml_type type = src0->type;

        શૂન્ય *        w_data  = (શૂન્ય *) src0->data;
        અચલ દશાંશ * feature = (અચલ દશાંશ *) src1->data;
        દશાંશ *       output  = (દશાંશ *) dst->data;

        અચલ int64_t gemm_m = ne11 * ne12 * ne13;
        અચલ int64_t gemm_k = ne10;
        અચલ int64_t gemm_n = ne01;

        spacemit_kernels::quantize_a_row_def       quantize_a_row_i8;
        spacemit_kernels::quantize_a_row_def       quantize_a_4row_i8;
        spacemit_kernels::gemm_kernel_quantize_def gemm_kernel;
        તાર્કિક                                       set_kernel_impl = ખોટું;

        int64_t block_stride_a = spacemit_kernels::q8_blk_size(a_blk_len);

#જો defined(RISCV64_SPACEMIT_IME2)
        જો (!set_kernel_impl && (global_spine_env_info.use_ime2)) {
            quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8;
            quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8;
            block_stride_a     = spacemit_kernels::q8_blk_size(a_blk_len, સાચું);

            જો constexpr (std::is_same_v<BLOC_TYPE, block_q6_K> || std::is_same_v<BLOC_TYPE, block_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i8;
                set_kernel_impl = સાચું;
            } નહીંતર જો constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> || std::is_same_v<BLOC_TYPE, block_q4_1> ||
                                 std::is_same_v<BLOC_TYPE, block_q4_K>) {
                જો constexpr (INTER_SIZE == 256) {
                    gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i4_hp;
                    quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8_hp;
                    quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8_hp;
                    block_stride_a     = spacemit_kernels::q8_hp_blk_size(a_blk_len, સાચું, સાચું);
                    set_kernel_impl    = સાચું;
                } નહીંતર {
                    gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i4;
                    quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8;
                    quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8;
                    block_stride_a     = spacemit_kernels::q8_blk_size(a_blk_len, સાચું);
                    set_kernel_impl    = સાચું;
                }
            } નહીંતર જો constexpr (std::is_same_v<BLOC_TYPE, block_q2_K>) {
                quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8k;
                quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8k;
                block_stride_a     = spacemit_kernels::q8k_blk_size(a_blk_len);

                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i2k;
                set_kernel_impl = સાચું;
            } નહીંતર જો constexpr (std::is_same_v<BLOC_TYPE, block_q3_K>) {
                quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8k;
                quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8k;
                block_stride_a     = spacemit_kernels::q8k_blk_size(a_blk_len);

                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i3k;
                set_kernel_impl = સાચું;
            } નહીંતર જો constexpr (std::is_same_v<BLOC_TYPE, block_mxfp4>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8mxfp4;
                set_kernel_impl = સાચું;
            } નહીંતર જો constexpr (std::is_same_v<BLOC_TYPE, block_q5_1> || std::is_same_v<BLOC_TYPE, block_q5_K> ||
                                 std::is_same_v<BLOC_TYPE, block_q5_0>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i5;
                set_kernel_impl = સાચું;
            }
        }
#endif

#જો defined(RISCV64_SPACEMIT_IME1)
        જો (!set_kernel_impl && (global_spine_env_info.use_ime1)) {
            quantize_a_row_i8  = spacemit_kernels::ime1::quantize_a_row_i8;
            quantize_a_4row_i8 = spacemit_kernels::ime1::quantize_a_4row_i8;

            જો constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> || std::is_same_v<BLOC_TYPE, block_q4_1> ||
                          std::is_same_v<BLOC_TYPE, block_q4_K>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i4;
                set_kernel_impl = સાચું;
            } નહીંતર જો constexpr (std::is_same_v<BLOC_TYPE, block_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i8;
                set_kernel_impl = સાચું;
            }
        }
#endif
        જો (!set_kernel_impl) {
            GGML_ABORT("no kernel implementation found for the block type");
        }

        અચલ int64_t a_k_blks = spacemit_kernels::div_round_up(gemm_k, a_blk_len);
        અચલ int64_t b_k_blks = spacemit_kernels::div_round_up(gemm_k, b_blk_len);

        અચલ int64_t row_stride_a        = a_k_blks * block_stride_a;
        અચલ int64_t gemm_workspace_size = GGML_PAD(gemm_m * row_stride_a, alignof(int64_t));

        જો (ith == 0 && params->wsize < gemm_workspace_size) {
            GGML_ABORT("wsize less than gemm_workspace_size");
        }

        uintptr_t ws_ptr = reinterpret_cast<uintptr_t>(params->wdata);

        શૂન્ય *        tcm_buffer      = ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer;
        અચલ int64_t tcm_buffer_size = ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer_size;

        auto * quant_a_buffer = reinterpret_cast<uint8_t *>(ws_ptr);

        constexpr int64_t row_align = 4;
        અચલ int64_t     row_blks  = spacemit_kernels::div_round_up(gemm_m, row_align);

        અચલ int64_t row_stride_b      = b_k_blks * get_repacked_block_type_size<BLOC_TYPE, INTER_SIZE, NB_COLS>();
        અચલ int64_t per_mb_rows_wsize = row_align * row_stride_a;
        અચલ int64_t per_nb_cols_wsize = NB_COLS * row_stride_b;

        અચલ int64_t barrier_idx = static_cast<int64_t>(ith / 2);

        GGML_ASSERT(global_spine_env_info.init_barrier != nullptr);
        GGML_ASSERT(barrier_idx < spine_init_barrier_ગણતરી);
        spine_barrier_t * cur_barrier = &global_spine_env_info.init_barrier[barrier_idx];

        જો (gemm_m == 1) {
            પૂર્ણાંક task_per_thread = spacemit_kernels::div_round_up(a_k_blks, nth);
            પૂર્ણાંક a_blk_start     = ith * task_per_thread;
            પૂર્ણાંક a_blk_end       = std::min(a_blk_start + task_per_thread, (પૂર્ણાંક) a_k_blks);
            જો (a_blk_start < a_blk_end) {
                quantize_a_row_i8(a_blk_len, feature + a_blk_start * a_blk_len, (a_blk_end - a_blk_start) * a_blk_len,
                                  quant_a_buffer + a_blk_start * block_stride_a);
            }
        } નહીંતર {
            પૂર્ણાંક task_per_thread = spacemit_kernels::div_round_up(row_blks, nth);
            પૂર્ણાંક m_row_blk_start = ith * task_per_thread;
            પૂર્ણાંક m_row_blk_end   = std::min(m_row_blk_start + task_per_thread, (પૂર્ણાંક) row_blks);
            માટે (પૂર્ણાંક m_row_blk = m_row_blk_start; m_row_blk < m_row_blk_end; m_row_blk++) {
                પૂર્ણાંક m_idx             = m_row_blk * row_align;
                પૂર્ણાંક rows_tobe_handled = (gemm_m - m_idx) > row_align ? row_align : (gemm_m - m_idx);

                જો (rows_tobe_handled == row_align && quantize_a_4row_i8 != nullptr) {
                    અચલ દશાંશ * a_row_ptr       = feature + m_idx * gemm_k;
                    auto *        quant_a_row_ptr = quant_a_buffer + m_idx * row_stride_a;
                    quantize_a_4row_i8(a_blk_len, a_row_ptr, gemm_k, quant_a_row_ptr);
                } નહીંતર {
                    જ્યારે (rows_tobe_handled) {
                        અચલ દશાંશ * a_row_ptr       = feature + m_idx * gemm_k;
                        auto *        quant_a_row_ptr = quant_a_buffer + m_idx * row_stride_a;
                        quantize_a_row_i8(a_blk_len, a_row_ptr, gemm_k, quant_a_row_ptr);
                        rows_tobe_handled -= 1;
                        m_idx += 1;
                    }
                }
            }
        }

        ggml_barrier(params->threadpool);

        અચલ int64_t gemm_m_stride     = gemm_n / gemm_m > 64 ? gemm_m : 16;
        અચલ int64_t gemm_m_blocked    = spacemit_kernels::div_round_up(gemm_m, gemm_m_stride);
        અચલ int64_t max_gemm_n_stride = spacemit_kernels::div_round_up(gemm_n * gemm_m_blocked, nth);

        int64_t gemm_n_stride = gemm_n;
        જો (max_gemm_n_stride < gemm_n) {
            gemm_n_stride =
                std::min(gemm_n_stride, spacemit_kernels::div_round_up(max_gemm_n_stride, NB_COLS) * NB_COLS);
        }

        જો (gemm_n_stride == gemm_n && tcm_buffer != nullptr && per_mb_rows_wsize <= tcm_buffer_size) {
            માટે (int64_t m_start = ith * row_align; m_start < gemm_m; m_start += row_align * nth) {
                uint8_t * b_col    = reinterpret_cast<uint8_t *>(w_data);
                uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? b_col : nullptr;

                int64_t m_row_real = std::min(gemm_m - m_start, row_align);

                spacemit_kernels::rvv::memcpy1d(tcm_buffer, quant_a_buffer + m_start * row_stride_a,
                                                m_row_real * row_stride_a);

                int64_t n_blk_real = 0;
                માટે (int64_t ni = 0; ni < gemm_n; ni += n_blk_real, b_col += n_blk_real * row_stride_b) {
                    n_blk_real = std::min(gemm_n - ni, (int64_t) NB_COLS);

                    uint8_t * a_row_ptr = (uint8_t *) tcm_buffer;
                    દશાંશ *   c_blk     = output + m_start * gemm_n + ni;

                    int32_t rows_remaining = m_row_real;

                    જ્યારે (rows_remaining > 0) {
                        auto rows_handled = gemm_kernel(b_blk_len, a_row_ptr, b_col, b_col_zp, c_blk, rows_remaining,
                                                        n_blk_real, b_k_blks, gemm_n);

                        c_blk += rows_handled * gemm_n;
                        a_row_ptr += rows_handled * row_stride_a;

                        rows_remaining -= rows_handled;
                    }
                }
            }
        } નહીંતર જો (tcm_buffer != nullptr && per_nb_cols_wsize <= tcm_buffer_size) {
            uint8_t * a_row = quant_a_buffer;
            uint8_t * b_col = reinterpret_cast<uint8_t *>(tcm_buffer);
            જો ((gemm_workspace_size + per_nb_cols_wsize) <= tcm_buffer_size) {
                a_row = (uint8_t *) tcm_buffer;
                b_col = reinterpret_cast<uint8_t *>(tcm_buffer) + gemm_workspace_size;
            }
            uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? b_col : nullptr;

            int64_t ni      = ith * NB_COLS;
            int64_t nb_real = std::min(gemm_n - ni, NB_COLS);

            જો (ith % 2 == 0 && nb_real > 0) {
                spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(w_data) + ni * row_stride_b,
                                                nb_real * row_stride_b);
                જો (a_row != quant_a_buffer) {
                    spacemit_kernels::rvv::memcpy1d(a_row, quant_a_buffer, gemm_workspace_size);
                }
            }

            spine_barrier_wait(cur_barrier);

            જો (ith % 2 != 0 && nb_real > 0) {
                જો (a_row != quant_a_buffer) {
                    spacemit_kernels::rvv::memcpy1d(a_row, quant_a_buffer, gemm_workspace_size);
                }
                spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(w_data) + ni * row_stride_b,
                                                nb_real * row_stride_b);
            }

            માટે (; ni < gemm_n; ni += NB_COLS * nth) {
                int64_t rows_remaining = gemm_m;
                દશાંશ * c_blk          = output + ni;
                auto *  a_row_cur      = a_row;

                જો (ith % 2 != 0) {
                    spine_barrier_wait(cur_barrier);
                }

                જ્યારે (rows_remaining > 0) {
                    auto rows_handled = gemm_kernel(b_blk_len, a_row_cur, b_col, b_col_zp, c_blk, rows_remaining,
                                                    nb_real, b_k_blks, gemm_n);

                    c_blk += rows_handled * gemm_n;
                    a_row_cur += rows_handled * row_stride_a;

                    rows_remaining -= rows_handled;
                }

                જો (ith % 2 == 0) {
                    spine_barrier_wait(cur_barrier);
                }

                અચલ int64_t next_ni = ni + NB_COLS * nth;
                જો (next_ni < gemm_n) {
                    nb_real = std::min(gemm_n - next_ni, NB_COLS);
                    spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(w_data) + next_ni * row_stride_b,
                                                    nb_real * row_stride_b);
                }
            }
        } નહીંતર {
            અચલ int64_t task_ગણતરી_m = spacemit_kernels::div_round_up(gemm_m, gemm_m_stride);
            અચલ int64_t task_ગણતરી_n = spacemit_kernels::div_round_up(gemm_n, gemm_n_stride);

            int64_t task_ગણતરી      = task_ગણતરી_m * task_ગણતરી_n;
            int64_t task_per_thread = (task_ગણતરી + nth - 1) / nth;
            int64_t start           = ith * task_per_thread;
            int64_t end             = std::min((ith + 1) * task_per_thread, task_ગણતરી);
            માટે (int64_t compute_idx = start; compute_idx < end; compute_idx++) {
                અચલ auto tid_n = compute_idx / task_ગણતરી_m;
                અચલ auto tid_m = compute_idx % task_ગણતરી_m;

                અચલ int64_t m_start = tid_m * gemm_m_stride;
                અચલ int64_t m_ગણતરી = std::min(gemm_m - m_start, (int64_t) gemm_m_stride);

                અચલ int64_t n_start = tid_n * gemm_n_stride;
                અચલ int64_t n_ગણતરી = std::min(gemm_n - n_start, (int64_t) gemm_n_stride);

                અચલ int64_t n_blk = m_ગણતરી == 1 ? n_ગણતરી : NB_COLS;

                uint8_t * b_col    = reinterpret_cast<uint8_t *>(w_data) + n_start * row_stride_b;
                uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? b_col : nullptr;

                int64_t n_blk_real = 0;
                માટે (int64_t ni = 0; ni < n_ગણતરી; ni += n_blk_real, b_col += n_blk_real * row_stride_b) {
                    n_blk_real = std::min(n_ગણતરી - ni, n_blk);

                    uint8_t * a_row = quant_a_buffer + m_start * row_stride_a;

                    દશાંશ * c_blk = output + m_start * gemm_n + n_start + ni;

                    int64_t rows_remaining = m_ગણતરી;

                    uint8_t * b_col_cur    = b_col;
                    uint8_t * b_col_zp_cur = b_col_zp;

                    જ્યારે (rows_remaining > 0) {
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

    શૂન્ય forward_mul_mat_id(ggml_compute_params * params, ggml_tensor * op) {
        constexpr size_t a_blk_len = INTER_SIZE;
        constexpr size_t b_blk_len = INTER_SIZE;

        અચલ ggml_tensor * src0 = op->src[0];
        અચલ ggml_tensor * src1 = op->src[1];
        અચલ ggml_tensor * ids  = op->src[2];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_દ્વિઅંકી_OP_LOCALS

        પૂર્ણાંક ith = params->ith;
        પૂર્ણાંક nth = params->nth;

        // row groups
        અચલ પૂર્ણાંક n_ids = ids->ne[0];  // n_expert_used
        અચલ પૂર્ણાંક n_as  = ne02;        // n_expert

        સંરચના mmid_row_mapping {
            int32_t i1;
            int32_t i2;
        };

        spacemit_kernels::quantize_a_row_def           quantize_a_row_i8;
        spacemit_kernels::gemm_kernel_quantize_def     gemm_kernel;
        spacemit_kernels::moe_gemm_kernel_quantize_def moe_gemm_kernel_m2;
        તાર્કિક                                           set_kernel_impl = ખોટું;
        size_t                                         block_stride_a  = spacemit_kernels::q8_blk_size(QK4_0);

#જો defined(RISCV64_SPACEMIT_IME2)
        જો (!set_kernel_impl && (global_spine_env_info.use_ime2)) {
            quantize_a_row_i8 = spacemit_kernels::rvv::quantize_a_row_i8;
            block_stride_a    = spacemit_kernels::q8_blk_size(QK4_0, સાચું);

            જો constexpr (std::is_same_v<BLOC_TYPE, block_q6_K> || std::is_same_v<BLOC_TYPE, block_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i8;
                set_kernel_impl = સાચું;
            } નહીંતર જો constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> || std::is_same_v<BLOC_TYPE, block_q4_1> ||
                                 std::is_same_v<BLOC_TYPE, block_q4_K>) {
                જો constexpr (INTER_SIZE == 256) {
                    gemm_kernel       = spacemit_kernels::ime2::gemm_kernel_i8i4_hp;
                    quantize_a_row_i8 = spacemit_kernels::rvv::quantize_a_row_i8_hp;
                    block_stride_a    = spacemit_kernels::q8_hp_blk_size(a_blk_len, સાચું, સાચું);
                    set_kernel_impl   = સાચું;
                } નહીંતર {
                    gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i4;
                    moe_gemm_kernel_m2 = spacemit_kernels::ime2::moe_m2_gemm_kernel_i8i4;
                    quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8;
                    block_stride_a     = spacemit_kernels::q8_blk_size(a_blk_len, સાચું);
                    set_kernel_impl    = સાચું;
                }
            } નહીંતર જો constexpr (std::is_same_v<BLOC_TYPE, block_q2_K>) {
                quantize_a_row_i8 = spacemit_kernels::rvv::quantize_a_row_i8k;
                block_stride_a    = spacemit_kernels::q8k_blk_size(a_blk_len);
                gemm_kernel       = spacemit_kernels::ime2::gemm_kernel_i8i2k;
                set_kernel_impl   = સાચું;
            } નહીંતર જો constexpr (std::is_same_v<BLOC_TYPE, block_q3_K>) {
                quantize_a_row_i8 = spacemit_kernels::rvv::quantize_a_row_i8k;
                block_stride_a    = spacemit_kernels::q8k_blk_size(a_blk_len);
                gemm_kernel       = spacemit_kernels::ime2::gemm_kernel_i8i3k;
                set_kernel_impl   = સાચું;
            } નહીંતર જો constexpr (std::is_same_v<BLOC_TYPE, block_mxfp4>) {
                gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8mxfp4;
                moe_gemm_kernel_m2 = spacemit_kernels::ime2::moe_m2_gemm_kernel_i8mxfp4;
                set_kernel_impl    = સાચું;
            } નહીંતર જો constexpr (std::is_same_v<BLOC_TYPE, block_q5_1> || std::is_same_v<BLOC_TYPE, block_q5_K> ||
                                 std::is_same_v<BLOC_TYPE, block_q5_0>) {
                gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i5;
                moe_gemm_kernel_m2 = spacemit_kernels::ime2::moe_m2_gemm_kernel_i8i5;
                set_kernel_impl    = સાચું;
            }
        }
#endif

#જો defined(RISCV64_SPACEMIT_IME1)
        જો (!set_kernel_impl && (global_spine_env_info.use_ime1)) {
            quantize_a_row_i8 = spacemit_kernels::ime1::quantize_a_row_i8;

            જો constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> || std::is_same_v<BLOC_TYPE, block_q4_1> ||
                          std::is_same_v<BLOC_TYPE, block_q4_K>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i4;
                set_kernel_impl = સાચું;
            } નહીંતર જો constexpr (std::is_same_v<BLOC_TYPE, block_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i8;
                set_kernel_impl = સાચું;
            }
        }
#endif
        જો (!set_kernel_impl) {
            GGML_ABORT("no kernel implementation found for the block type");
        }

        અચલ size_t a_k_blks = spacemit_kernels::div_round_up(ne10, a_blk_len);
        અચલ size_t b_k_blks = spacemit_kernels::div_round_up(ne10, b_blk_len);

        અચલ size_t nbw1                = a_k_blks * block_stride_a;
        અચલ size_t nbw2                = ne11 * nbw1;
        અચલ size_t nbw3                = nbw2 * ne12;
        અચલ size_t gemm_workspace_size = GGML_PAD(nbw3, alignof(int64_t));

        અચલ uintptr_t ws_ptr         = reinterpret_cast<uintptr_t>(params->wdata);
        auto *          quant_a_buffer = reinterpret_cast<uint8_t *>(ws_ptr);

        જો (ne11 == 1) {
            માટે (int64_t ii = ith; ii < ne12 * a_k_blks; ii += nth) {
                int64_t i12       = ii / a_k_blks;
                int64_t ak_blk_id = ii % a_k_blks;
                quantize_a_row_i8(a_blk_len, (દશાંશ *) ((અક્ષર *) src1->data + i12 * nb12) + ak_blk_id * a_blk_len,
                                  a_blk_len, quant_a_buffer + i12 * nbw2 + ak_blk_id * block_stride_a);
            }
        } નહીંતર {
            માટે (int64_t ii = ith; ii < ne12 * ne11; ii += nth) {
                int64_t i12 = ii / ne11;
                int64_t i11 = ii % ne11;
                quantize_a_row_i8(a_blk_len, (દશાંશ *) ((અક્ષર *) src1->data + i12 * nb12 + i11 * nb11), ne10,
                                  quant_a_buffer + i12 * nbw2 + i11 * nbw1);
            }
        }

#define MMID_MATRIX_ROW(row_id, i1) matrix_rows[(row_id) *ne12 + (i1)]

        int64_t *          matrix_row_counts       = (int64_t *) (ws_ptr + gemm_workspace_size);
        int32_t *          valid_ep_ગણતરી          = (int32_t *) (matrix_row_counts + n_as);
        int32_t *          valid_act_ગણતરી         = (int32_t *) (valid_ep_ગણતરી + 1);
        int64_t *          valid_matrix_row_counts = (int64_t *) (valid_act_ગણતરી + 1);
        mmid_row_mapping * matrix_rows             = (mmid_row_mapping *) (valid_matrix_row_counts + n_as);

        જો (ith == 0) {
            // initialize matrix_row_counts
            memset(matrix_row_counts, 0, n_as * કદ(int64_t));

            // group rows by src0 matrix
            માટે (int32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
                માટે (int32_t id = 0; id < n_ids; ++id) {
                    અચલ int32_t i02 =
                        *(અચલ int32_t *) ((અચલ અક્ષર *) ids->data + iid1 * ids->nb[1] + id * ids->nb[0]);

                    GGML_ASSERT(i02 >= 0 && i02 < n_as);

                    MMID_MATRIX_ROW(i02, matrix_row_counts[i02]) = { id, iid1 };
                    matrix_row_counts[i02] += 1;
                }
            }

            int32_t valid_ep_ગણતરી_t  = 0;
            int32_t valid_act_ગણતરી_t = 0;
            માટે (પૂર્ણાંક cur_a = 0; cur_a < n_as; ++cur_a) {
                અચલ int64_t cne1 = matrix_row_counts[cur_a];
                જો (cne1 == 0) {
                    ચાલુ;
                }
                valid_matrix_row_counts[valid_ep_ગણતરી_t] = cur_a;
                valid_act_ગણતરી_t += cne1;
                valid_ep_ગણતરી_t += 1;
            }
            valid_ep_ગણતરી[0]  = valid_ep_ગણતરી_t;
            valid_act_ગણતરી[0] = valid_act_ગણતરી_t;
        }

        અચલ int64_t barrier_idx = static_cast<int64_t>(ith / 2);

        GGML_ASSERT(global_spine_env_info.init_barrier != nullptr);
        GGML_ASSERT(barrier_idx < spine_init_barrier_ગણતરી);
        spine_barrier_t * cur_barrier = &global_spine_env_info.init_barrier[barrier_idx];

        ggml_barrier(params->threadpool);

        અચલ size_t row_stride_b      = b_k_blks * get_repacked_block_type_size<BLOC_TYPE, INTER_SIZE, NB_COLS>();
        અચલ size_t expert_b_stride   = ne01 * row_stride_b;
        અચલ size_t per_nb_cols_wsize = NB_COLS * row_stride_b;

        std::array<અચલ uint8_t *, 2> src_workspaces;
        std::array<દશાંશ *, 2>         dst_workspaces;

        auto *     tcm_buffer      = ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer;
        અચલ auto tcm_buffer_size = ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer_size;

        અચલ auto valid_ep_ગણતરી_t  = valid_ep_ગણતરી[0];
        અચલ auto valid_act_ગણતરી_t = valid_act_ગણતરી[0];

        પૂર્ણાંક nth_es = 1;
        પૂર્ણાંક nth_n  = nth;

        પૂર્ણાંક ith_es = ith % nth_es;
        પૂર્ણાંક ith_n  = (ith / nth_es) % nth_n;

        જો (valid_ep_ગણતરી_t % nth == 0 && tcm_buffer != nullptr && valid_ep_ગણતરી_t == n_as &&
            valid_act_ગણતરી_t == n_as && per_nb_cols_wsize <= tcm_buffer_size) {
            માટે (int64_t valid_id = ith; valid_id < valid_ep_ગણતરી_t; valid_id += nth) {
                અચલ int64_t cur_a = valid_matrix_row_counts[valid_id];

                auto * src0_cur = (uint8_t *) src0->data + cur_a * expert_b_stride;

                mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, 0);
                અચલ પૂર્ણાંક        id          = row_mapping.i1;
                અચલ int64_t    i11         = id % ne11;
                અચલ int64_t    i12         = row_mapping.i2;
                અચલ int64_t    i1          = id;
                અચલ int64_t    i2          = i12;

                auto *  src1_col = quant_a_buffer + (i11 * nbw1 + i12 * nbw2);
                દશાંશ * c_blk    = (દશાંશ *) ((અક્ષર *) dst->data + (i1 * nb1 + i2 * nb2));

                uint8_t * a_row = src1_col;
                uint8_t * b_col = reinterpret_cast<uint8_t *>(tcm_buffer);
                જો ((nbw1 + per_nb_cols_wsize) <= tcm_buffer_size) {
                    a_row = (uint8_t *) tcm_buffer;
                    b_col = reinterpret_cast<uint8_t *>(tcm_buffer) + nbw1;
                }
                uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? b_col : nullptr;

                જો (ith % 2 == 0) {
                    spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(src0_cur), per_nb_cols_wsize);

                    જો (a_row != src1_col) {
                        spacemit_kernels::rvv::memcpy1d(a_row, src1_col, nbw1);
                    }
                }

                spine_barrier_wait(cur_barrier);

                જો (ith % 2 != 0) {
                    જો (a_row != src1_col) {
                        spacemit_kernels::rvv::memcpy1d(a_row, src1_col, nbw1);
                    }

                    spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(src0_cur), per_nb_cols_wsize);
                }

                int64_t nb_real = std::min(ne01, NB_COLS);
                માટે (int64_t ni = 0; ni < ne01; ni += NB_COLS) {
                    જો (ith % 2 != 0) {
                        spine_barrier_wait(cur_barrier);
                    }

                    gemm_kernel(b_blk_len, a_row, b_col, b_col_zp, c_blk + ni, 1, nb_real, b_k_blks, ne01);

                    જો (ith % 2 == 0) {
                        spine_barrier_wait(cur_barrier);
                    }

                    અચલ int64_t next_ni = ni + NB_COLS;
                    જો (next_ni < ne01) {
                        nb_real = std::min(ne01 - next_ni, NB_COLS);
                        spacemit_kernels::rvv::memcpy1d(
                            b_col, reinterpret_cast<uint8_t *>(src0_cur) + next_ni * row_stride_b, per_nb_cols_wsize);
                    }
                }
            }
        } નહીંતર {
            માટે (int64_t valid_id = ith_es; valid_id < valid_ep_ગણતરી_t; valid_id += nth_es) {
                અચલ int64_t cur_a = valid_matrix_row_counts[valid_id];
                અચલ int64_t cne1  = matrix_row_counts[cur_a];

                int64_t src1_cur_start = 0;
                int64_t src1_cur_end   = cne1;

                int64_t src0_cur_start = (ith_n * ne01) / nth_n;
                int64_t src0_cur_end   = MIN(((ith_n + 1) * ne01) / nth_n, ne01);

                જો (src1_cur_start >= src1_cur_end || src0_cur_start >= src0_cur_end) {
                    ચાલુ;
                }

                src0_cur_start =
                    (src0_cur_start % NB_COLS) ? src0_cur_start + NB_COLS - (src0_cur_start % NB_COLS) : src0_cur_start;
                src0_cur_end =
                    (src0_cur_end % NB_COLS) ? src0_cur_end + NB_COLS - (src0_cur_end % NB_COLS) : src0_cur_end;

                auto *    src0_cur = (uint8_t *) src0->data + cur_a * expert_b_stride + src0_cur_start * row_stride_b;
                uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? src0_cur : nullptr;

                size_t extra_tcm_buffer_size = tcm_buffer_size;
                શૂન્ય * extra_tcm_buffer      = tcm_buffer;
                જો (tcm_buffer != nullptr && (src1_cur_end - src1_cur_start) >= 4 &&
                    (src0_cur_end - src0_cur_start) * row_stride_b <= tcm_buffer_size) {
                    spacemit_kernels::rvv::memcpy1d(tcm_buffer, src0_cur,
                                                    (src0_cur_end - src0_cur_start) * row_stride_b);
                    src0_cur = reinterpret_cast<uint8_t *>(tcm_buffer);
                    b_col_zp = block_type_has_zp<BLOC_TYPE>() ? src0_cur : nullptr;
                    extra_tcm_buffer_size -= (src0_cur_end - src0_cur_start) * row_stride_b;
                    extra_tcm_buffer = reinterpret_cast<શૂન્ય *>(reinterpret_cast<uintptr_t>(tcm_buffer) +
                                                                (src0_cur_end - src0_cur_start) * row_stride_b);
                }

                પૂર્ણાંક ir1 = src1_cur_start;

                જો (extra_tcm_buffer_size >= nbw1 && extra_tcm_buffer != nullptr) {
                    int64_t quant_a_tile_size = extra_tcm_buffer_size / nbw1;
                    કરો {
                        quant_a_tile_size = MIN(quant_a_tile_size, src1_cur_end - ir1);

                        uint8_t * quant_a_tile_buffer = reinterpret_cast<uint8_t *>(extra_tcm_buffer);

                        પૂર્ણાંક iir1 = ir1;
                        માટે (; iir1 < (ir1 + quant_a_tile_size); ++iir1) {
                            mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, iir1);

                            અચલ પૂર્ણાંક id = row_mapping.i1;  // selected expert index

                            અચલ int64_t i11 = id % ne11;
                            અચલ int64_t i12 = row_mapping.i2;  // row index in src1

                            auto * src1_col = quant_a_buffer + (i11 * nbw1 + i12 * nbw2);
                            spacemit_kernels::rvv::memcpy1d(quant_a_tile_buffer, src1_col, nbw1);
                            quant_a_tile_buffer = quant_a_tile_buffer + nbw1;
                        }

                        quant_a_tile_buffer = reinterpret_cast<uint8_t *>(extra_tcm_buffer);
                        iir1                = ir1;

                        જો (moe_gemm_kernel_m2 != nullptr) {
                            માટે (; iir1 < (ir1 + quant_a_tile_size - 1); iir1 += 2, quant_a_tile_buffer += 2 * nbw1) {
                                mmid_row_mapping row_mapping_0 = MMID_MATRIX_ROW(cur_a, iir1);
                                mmid_row_mapping row_mapping_1 = MMID_MATRIX_ROW(cur_a, iir1 + 1);

                                src_workspaces[0] = quant_a_tile_buffer;
                                src_workspaces[1] = quant_a_tile_buffer + nbw1;

                                dst_workspaces[0] =
                                    (દશાંશ *) ((અક્ષર *) dst->data + (row_mapping_0.i1 * nb1 + row_mapping_0.i2 * nb2)) +
                                    src0_cur_start;
                                dst_workspaces[1] = (દશાંશ *) ((અક્ષર *) dst->data +
                                                               ((row_mapping_1.i1) * nb1 + (row_mapping_1.i2) * nb2)) +
                                                    src0_cur_start;
                                moe_gemm_kernel_m2(b_blk_len, src_workspaces.data(), src0_cur, b_col_zp,
                                                   dst_workspaces.data(), 1, src0_cur_end - src0_cur_start, b_k_blks,
                                                   ne01);
                            }
                        }

                        માટે (; iir1 < (ir1 + quant_a_tile_size); iir1++, quant_a_tile_buffer += nbw1) {
                            mmid_row_mapping row_mapping_0 = MMID_MATRIX_ROW(cur_a, iir1);

                            gemm_kernel(
                                b_blk_len, quant_a_tile_buffer, src0_cur, b_col_zp,
                                (દશાંશ *) ((અક્ષર *) dst->data + (row_mapping_0.i1 * nb1 + row_mapping_0.i2 * nb2)) +
                                    src0_cur_start,
                                1, src0_cur_end - src0_cur_start, b_k_blks, ne01);
                        }

                        ir1 += quant_a_tile_size;
                    } જ્યારે (ir1 < src1_cur_end);
                } નહીંતર {
                    જો (moe_gemm_kernel_m2 != nullptr) {
                        માટે (; ir1 < src1_cur_end - 1; ir1 += 2) {
                            માટે (પૂર્ણાંક iir1 = 0; iir1 < 2; ++iir1) {
                                mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, ir1 + iir1);

                                અચલ પૂર્ણાંક id = row_mapping.i1;  // selected expert index

                                અચલ int64_t i11 = id % ne11;
                                અચલ int64_t i12 = row_mapping.i2;  // row index in src1

                                અચલ int64_t i1 = id;               // selected expert index
                                અચલ int64_t i2 = i12;              // row

                                src_workspaces[iir1] = quant_a_buffer + (i11 * nbw1 + i12 * nbw2);

                                dst_workspaces[iir1] =
                                    (દશાંશ *) ((અક્ષર *) dst->data + (i1 * nb1 + i2 * nb2)) + src0_cur_start;
                            }

                            moe_gemm_kernel_m2(b_blk_len, src_workspaces.data(), src0_cur, b_col_zp,
                                               dst_workspaces.data(), 1, src0_cur_end - src0_cur_start, b_k_blks, ne01);
                        }
                    }

                    માટે (; ir1 < src1_cur_end; ir1++) {
                        mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, ir1);

                        અચલ પૂર્ણાંક id = row_mapping.i1;  // selected expert index

                        અચલ int64_t i11 = id % ne11;
                        અચલ int64_t i12 = row_mapping.i2;  // row index in src1

                        અચલ int64_t i1 = id;               // selected expert index
                        અચલ int64_t i2 = i12;              // row

                        auto * src1_col = quant_a_buffer + (i11 * nbw1 + i12 * nbw2);

                        gemm_kernel(b_blk_len, src1_col, src0_cur, b_col_zp,
                                    (દશાંશ *) ((અક્ષર *) dst->data + (i1 * nb1 + i2 * nb2)) + src0_cur_start, 1,
                                    src0_cur_end - src0_cur_start, b_k_blks, ne01);
                    }
                }
            }
        }
#undef MMID_MATRIX_ROW
    }

    પૂર્ણાંક repack(ggml_tensor * t, અચલ શૂન્ય * data, size_t data_size) override {
        GGML_LOG_DEBUG("%s: repack tensor %s with %s_%dx%d\n", __func__, t->name, ggml_type_name(t->type),
                       (પૂર્ણાંક) NB_COLS, (પૂર્ણાંક) INTER_SIZE);
        પાછા ggml::cpu::riscv64_spacemit::repack<BLOC_TYPE, INTER_SIZE, NB_COLS>(t, data, data_size);
    }
};

વર્ગ tensor_traits_common : જાહેર tensor_traits_base {
    તાર્કિક work_size(પૂર્ણાંક n_threads, અચલ ggml_tensor * op, size_t & size) override {
        પસંદગી (op->op) {
            સ્થિતિ GGML_OP_FLASH_ATTN_EXT:
                {
                    અચલ પૂર્ણાંક     n_tasks = n_threads;
                    અચલ int64_t neq2    = op->src[0]->ne[2];  // number of query heads
                    અચલ int64_t DK      = op->src[1]->ne[0];
                    અચલ int64_t DV      = op->src[2]->ne[0];  // DV

                    // Tiled flash attention scratch (tile sizes defined in common.h)
                    // Per-thread: Q_q + KQ + mask + VKQ32 + V32 + K_f32 + padding
                    size_t prefill = કદ(દશાંશ) *
                                     (GGML_FA_TILE_Q * DK + 2 * GGML_FA_TILE_Q * GGML_FA_TILE_KV + GGML_FA_TILE_Q * DV +
                                      GGML_FA_TILE_KV * DV + GGML_FA_TILE_KV * DK) *
                                     n_tasks;

                    // Decode path: n_kv_chunks = n_tasks (one chunk per thread)
                    // Per-thread: VKQ accmulator (DV), partial M, partial S + intra-thread scratch for V, Q and VKQ
                    size_t n_chunks = n_tasks;
                    size_t decode   = કદ(દશાંશ) * (neq2 * n_chunks * (2 + DV) + n_tasks * (DK + 2 * DV));

                    size = MAX(prefill, decode);
                }
                પાછા સાચું;
            મૂળભૂત:
                તોડો;
        }
        પાછા ખોટું;
    }

    તાર્કિક compute_forward(ggml_compute_params * params, ggml_tensor * op) override {
        પસંદગી (op->op) {
            સ્થિતિ GGML_OP_NORM:
                પસંદગી (op->src[0]->type) {
                    સ્થિતિ GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_norm_f32(params, op);
                        પાછા સાચું;
                    મૂળભૂત:
                        GGML_ABORT("fatal error");
                }
            સ્થિતિ GGML_OP_RMS_NORM:
                પસંદગી (op->src[0]->type) {
                    સ્થિતિ GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_rms_norm_f32(params, op);
                        પાછા સાચું;
                    મૂળભૂત:
                        GGML_ABORT("fatal error");
                }
            સ્થિતિ GGML_OP_ADD:
                પસંદગી (op->src[0]->type) {
                    સ્થિતિ GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_દ્વિઅંકી<GGML_OP_ADD, દશાંશ>(params, op);
                        પાછા સાચું;
                    સ્થિતિ GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_દ્વિઅંકી<GGML_OP_ADD, _Float16>(params, op);
                        પાછા સાચું;
                    મૂળભૂત:
                        ggml_compute_forward_add(params, op);
                        પાછા સાચું;
                }
            સ્થિતિ GGML_OP_SUB:
                પસંદગી (op->src[0]->type) {
                    સ્થિતિ GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_દ્વિઅંકી<GGML_OP_SUB, દશાંશ>(params, op);
                        પાછા સાચું;
                    સ્થિતિ GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_દ્વિઅંકી<GGML_OP_SUB, _Float16>(params, op);
                        પાછા સાચું;
                    મૂળભૂત:
                        ggml_compute_forward_sub(params, op);
                        પાછા સાચું;
                }
            સ્થિતિ GGML_OP_MUL:
                પસંદગી (op->src[0]->type) {
                    સ્થિતિ GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_દ્વિઅંકી<GGML_OP_MUL, દશાંશ>(params, op);
                        પાછા સાચું;
                    સ્થિતિ GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_દ્વિઅંકી<GGML_OP_MUL, _Float16>(params, op);
                        પાછા સાચું;
                    મૂળભૂત:
                        ggml_compute_forward_mul(params, op);
                        પાછા સાચું;
                }
            સ્થિતિ GGML_OP_DIV:
                પસંદગી (op->src[0]->type) {
                    સ્થિતિ GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_દ્વિઅંકી<GGML_OP_DIV, દશાંશ>(params, op);
                        પાછા સાચું;
                    સ્થિતિ GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_દ્વિઅંકી<GGML_OP_DIV, _Float16>(params, op);
                        પાછા સાચું;
                    મૂળભૂત:
                        ggml_compute_forward_div(params, op);
                        પાછા સાચું;
                }
            સ્થિતિ GGML_OP_FLASH_ATTN_EXT:
                forward_flash_attn_ext_f16(params, op);
                પાછા સાચું;
            સ્થિતિ GGML_OP_CONT:
                {
                    અચલ ggml_tensor * src0 = op->src[0];
                    જો (op->type == src0->type && op->nb[0] != src0->nb[0] && op->nb[0] == src0->nb[1] &&
                        op->ne[3] * op->ne[2] * op->nb[2] == src0->ne[3] * src0->ne[2] * src0->nb[2]) {
                        spacemit_kernels::rvv::forward_cont_with_permute(params, op);
                    } નહીંતર {
                        ggml_compute_forward_cont(params, op);
                    }
                    પાછા સાચું;
                }
            સ્થિતિ GGML_OP_CPY:
                {
                    અચલ ggml_tensor * src0 = op->src[0];
                    જો (op->type == src0->type && op->nb[0] == src0->nb[1] && src0->nb[0] != src0->nb[1] &&
                        ggml_nelements(src0) == ggml_nelements(op)) {
                        spacemit_kernels::rvv::forward_cpy_with_permute(params, op);
                    } નહીંતર {
                        ggml_compute_forward_cpy(params, op);
                    }
                    પાછા સાચું;
                }
            સ્થિતિ GGML_OP_REPEAT:
                {
                    અચલ તાર્કિક rows_equal         = ggml_nrows(op->src[0]) == ggml_nrows(op);
                    અચલ તાર્કિક broadcast_or_equal = op->src[0]->ne[0] == 1 || op->src[0]->ne[0] == op->ne[0];

                    જો (rows_equal && broadcast_or_equal) {
                        પસંદગી (op->src[0]->type) {
                            સ્થિતિ GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_repeat_nrows<int32_t>(params, op);
                                પાછા સાચું;
                            સ્થિતિ GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_repeat_nrows<int16_t>(params, op);
                                પાછા સાચું;
                            મૂળભૂત:
                                તોડો;
                        }
                    }

                    જો (op->src[0]->ne[1] == 1 && op->src[0]->ne[0] == op->ne[0]) {
                        પસંદગી (op->src[0]->type) {
                            સ્થિતિ GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_repeat_dim1<int32_t>(params, op);
                                પાછા સાચું;
                            સ્થિતિ GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_repeat_dim1<int16_t>(params, op);
                                પાછા સાચું;
                            મૂળભૂત:
                                તોડો;
                        }
                    }

                    ggml_compute_forward_repeat(params, op);
                }
                પાછા સાચું;
            સ્થિતિ GGML_OP_SUM_ROWS:
                {
                    જો (op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32) {
                        spacemit_kernels::rvv::forward_sum_rows<દશાંશ>(params, op);
                    } નહીંતર {
                        ggml_compute_forward_sum_rows(params, op);
                    }
                }
                પાછા સાચું;
            સ્થિતિ GGML_OP_GET_ROWS:
                {
                    જો (op->src[0]->type == op->type) {
                        પસંદગી (op->src[0]->type) {
                            સ્થિતિ GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_get_rows<int32_t>(params, op);
                                પાછા સાચું;
                            સ્થિતિ GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_get_rows<int16_t>(params, op);
                                પાછા સાચું;
                            મૂળભૂત:
                                તોડો;
                        }
                    }

                    ggml_compute_forward_get_rows(params, op);
                }
                પાછા સાચું;
            સ્થિતિ GGML_OP_CONCAT:
                {
                    અચલ int32_t dim = ggml_get_op_params_i32(op, 0);
                    જો (dim == 0 && op->type == op->src[0]->type) {
                        પસંદગી (op->src[0]->type) {
                            સ્થિતિ GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_concat<int32_t>(params, op);
                                પાછા સાચું;
                            સ્થિતિ GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_concat<int16_t>(params, op);
                                પાછા સાચું;
                            મૂળભૂત:
                                તોડો;
                        }
                    }

                    ggml_compute_forward_concat(params, op);
                }
                પાછા સાચું;
            // TODO For GGML_OP_GATED_DELTA_NET
            // case GGML_OP_GATED_DELTA_NET:
            //     return true;
            મૂળભૂત:
                તોડો;
        }
        પાછા ખોટું;
    }

    શૂન્ય forward_flash_attn_ext_f16(અચલ ggml_compute_params * params, ggml_tensor * dst) {
        અચલ ggml_tensor * q = dst->src[0];
        અચલ ggml_tensor * k = dst->src[1];
        અચલ ggml_tensor * v = dst->src[2];

        GGML_TENSOR_LOCALS(int64_t, neq, q, ne)
        GGML_TENSOR_LOCALS(size_t, nbq, q, nb)
        GGML_TENSOR_LOCALS(int64_t, nek, k, ne)
        GGML_TENSOR_LOCALS(size_t, nbk, k, nb)
        GGML_TENSOR_LOCALS(int64_t, nev, v, ne)
        GGML_TENSOR_LOCALS(size_t, nbv, v, nb)
        GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
        GGML_TENSOR_LOCALS(size_t, nb, dst, nb)

        અચલ int64_t DK = nek0;
        અચલ int64_t DV = nev0;

        અચલ તાર્કિક supported_prec  = (dst->op_params[3] == GGML_PREC_F32 || dst->op_params[3] == GGML_PREC_DEFAULT);
        અચલ તાર્કિક supported_types = (q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16);
        અચલ તાર્કિક supported_shape = (DK > 0 && DK <= 128 && DV > 0 && DV <= 128);
        અચલ તાર્કિક supported_vlen  = (__riscv_vlenb() == 128);

        જો (!(supported_prec && supported_types && supported_shape && supported_vlen)) {
            ggml_compute_forward_flash_attn_ext(params, dst);
            પાછા;
        }

        // total rows in q
        અચલ int64_t nr = neq1 * neq2 * neq3;

        // rows per thread
        અચલ પૂર્ણાંક ith = params->ith;
        અચલ પૂર્ણાંક nth = params->nth;

        સ્થિર constexpr int64_t Q_TILE_SZ = ggml_fa_tile_config::Q;
        અચલ તાર્કિક               use_tiled = !params->use_ref && (neq1 >= Q_TILE_SZ);

        // 4x chunks per thread
        // int     nth_scaled = nth * 4;
        // int64_t chunk_size = (nr + nth_scaled - 1) / nth_scaled;
        // int64_t nchunk     = (nr + chunk_size - 1) / chunk_size;

        // if (nth == 1 || nchunk < nth) {
        //     nchunk = nth;
        // }

        int64_t nchunk = nth;

        જો (ith == 0) {
            // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
            ggml_threadpool_chunk_set(params->threadpool, nth);
        }

        ggml_barrier(params->threadpool);

        // The number of elements in each chunk
        અચલ int64_t dr = (nr + nchunk - 1) / nchunk;

        // The first chunk comes from our thread_id, the rest will get auto-assigned.
        પૂર્ણાંક current_chunk = ith;

        જ્યારે (current_chunk < nchunk) {
            અચલ int64_t ir0 = dr * current_chunk;
            અચલ int64_t ir1 = MIN(ir0 + dr, nr);

            જો (use_tiled) {
                spacemit_kernels::rvv::forward_flash_attn_ext_f16_tiled_vlen1024_vf16(
                    params, dst, ir0, ir1, ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer,
                    ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer_size);
            } નહીંતર {
                spacemit_kernels::rvv::forward_flash_attn_ext_f16_one_chunk_vlen1024_vf16(
                    params, dst, ir0, ir1, ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer,
                    ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer_size);
            }

            current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
        }
    }

    પૂર્ણાંક repack(ggml_tensor * t, અચલ શૂન્ય * data, size_t data_size) override {
        memcpy(t->data, data, data_size);
        પાછા 0;
    }
};

// Impl By IME1
સ્થિર અચલ tensor_traits<block_q4_0, 32, 16>  q4_0_16x32_q8_0;
સ્થિર અચલ tensor_traits<block_q4_1, 32, 16>  q4_1_16x32_q8_0;
સ્થિર અચલ tensor_traits<block_q4_K, 32, 16>  q4_k_16x32_q8_0;
સ્થિર અચલ tensor_traits<block_q8_0, 32, 16>  q8_0_16x32_q8_0;
// Impl By IME2
સ્થિર અચલ tensor_traits<block_q2_K, 256, 32> q2_k_32x256_q8_0;
સ્થિર અચલ tensor_traits<block_q3_K, 256, 32> q3_k_32x256_q8_0;
સ્થિર અચલ tensor_traits<block_q4_0, 32, 32>  q4_0_32x32_q8_0;
સ્થિર અચલ tensor_traits<block_q4_1, 32, 32>  q4_1_32x32_q8_0;
સ્થિર અચલ tensor_traits<block_q4_0, 256, 32> q4_0_32x256_q8_0;
સ્થિર અચલ tensor_traits<block_q4_1, 256, 32> q4_1_32x256_q8_0;
સ્થિર અચલ tensor_traits<block_q4_K, 32, 32>  q4_k_32x32_q8_0;
સ્થિર અચલ tensor_traits<block_q6_K, 32, 32>  q6_k_32x32_q8_0;
સ્થિર અચલ tensor_traits<block_q8_0, 32, 32>  q8_0_32x32_q8_0;
સ્થિર અચલ tensor_traits<block_mxfp4, 32, 32> mxfp4_32x32_q8_0;
સ્થિર અચલ tensor_traits<block_q5_K, 32, 32>  q5_k_32x32_q8_0;
સ્થિર અચલ tensor_traits<block_q5_1, 32, 32>  q5_1_32x32_q8_0;
સ્થિર અચલ tensor_traits<block_q5_0, 32, 32>  q5_0_32x32_q8_0;
// Impl By RVV
સ્થિર અચલ tensor_traits_common               rvv_impl;

}  // namespace ggml::cpu::riscv64_spacemit

સ્થિર અચલ ggml::cpu::tensor_traits * ggml_riscv64_spacemit_get_optimal_repack_type(અચલ ggml_tensor * cur) {
    પસંદગી (cur->type) {
        સ્થિતિ GGML_TYPE_Q2_K:
            {
#જો defined(RISCV64_SPACEMIT_IME2)
                જો (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    પાછા &ggml::cpu::riscv64_spacemit::q2_k_32x256_q8_0;
                }
#endif
            }
            તોડો;
        સ્થિતિ GGML_TYPE_Q3_K:
            {
#જો defined(RISCV64_SPACEMIT_IME2)
                જો (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    પાછા &ggml::cpu::riscv64_spacemit::q3_k_32x256_q8_0;
                }
#endif
            }
            તોડો;
        સ્થિતિ GGML_TYPE_Q4_0:
            {
#જો defined(RISCV64_SPACEMIT_IME2)
                જો (cur->ne[1] % 32 == 0 && cur->ne[0] % 256 == 0 &&
                    (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    પાછા &ggml::cpu::riscv64_spacemit::q4_0_32x256_q8_0;
                }

                જો (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    પાછા &ggml::cpu::riscv64_spacemit::q4_0_32x32_q8_0;
                }
#endif

#જો defined(RISCV64_SPACEMIT_IME1)
                જો (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    પાછા &ggml::cpu::riscv64_spacemit::q4_0_16x32_q8_0;
                }
#endif
            }
            તોડો;
        સ્થિતિ GGML_TYPE_Q4_1:
            {
#જો defined(RISCV64_SPACEMIT_IME2)
                // TODO
                // if (cur->ne[1] % 32 == 0 && cur->ne[0] % 256 == 0 &&
                //     (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                //     return &ggml::cpu::riscv64_spacemit::q4_1_32x256_q8_0;
                // }

                જો (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    પાછા &ggml::cpu::riscv64_spacemit::q4_1_32x32_q8_0;
                }
#endif

#જો defined(RISCV64_SPACEMIT_IME1)
                જો (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    પાછા &ggml::cpu::riscv64_spacemit::q4_1_16x32_q8_0;
                }
#endif
            }
            તોડો;
        સ્થિતિ GGML_TYPE_Q4_K:
            {
#જો defined(RISCV64_SPACEMIT_IME2)
                જો (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    પાછા &ggml::cpu::riscv64_spacemit::q4_k_32x32_q8_0;
                }
#endif

#જો defined(RISCV64_SPACEMIT_IME1)
                જો (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    પાછા &ggml::cpu::riscv64_spacemit::q4_k_16x32_q8_0;
                }
#endif
            }
            તોડો;
        સ્થિતિ GGML_TYPE_Q6_K:
            {
#જો defined(RISCV64_SPACEMIT_IME2)
                જો ((ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    પાછા &ggml::cpu::riscv64_spacemit::q6_k_32x32_q8_0;
                }
#endif
            }
            તોડો;
        સ્થિતિ GGML_TYPE_Q8_0:
            {
#જો defined(RISCV64_SPACEMIT_IME2)
                જો ((ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    પાછા &ggml::cpu::riscv64_spacemit::q8_0_32x32_q8_0;
                }
#endif

#જો defined(RISCV64_SPACEMIT_IME1)
                જો (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    પાછા &ggml::cpu::riscv64_spacemit::q8_0_16x32_q8_0;
                }
#endif
            }
            તોડો;
        સ્થિતિ GGML_TYPE_MXFP4:
            {
#જો defined(RISCV64_SPACEMIT_IME2)
                // TODO
                // if (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                //     return &ggml::cpu::riscv64_spacemit::mxfp4_32x32_q8_0;
                // }
#endif
            }
            તોડો;
        સ્થિતિ GGML_TYPE_Q5_K:
            {
#જો defined(RISCV64_SPACEMIT_IME2)
                જો (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    પાછા &ggml::cpu::riscv64_spacemit::q5_k_32x32_q8_0;
                }
#endif
            }
            તોડો;
        સ્થિતિ GGML_TYPE_Q5_1:
            {
#જો defined(RISCV64_SPACEMIT_IME2)
                જો (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    પાછા &ggml::cpu::riscv64_spacemit::q5_1_32x32_q8_0;
                }
#endif
            }
            તોડો;
        સ્થિતિ GGML_TYPE_Q5_0:
            {
#જો defined(RISCV64_SPACEMIT_IME2)
                જો (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    પાછા &ggml::cpu::riscv64_spacemit::q5_0_32x32_q8_0;
                }
#endif
            }
            તોડો;
        મૂળભૂત:
            તોડો;
    }

    પાછા nullptr;
}

સ્થિર ગણના ggml_status ggml_backend_riscv64_spacemit_buffer_init_tensor(ggml_backend_buffer_t buffer,
                                                                         ggml_tensor *         tensor) {
    tensor->extra =
        (શૂન્ય *) const_cast<ggml::cpu::tensor_traits *>(ggml_riscv64_spacemit_get_optimal_repack_type(tensor));

    GGML_UNUSED(buffer);

    પાછા GGML_STATUS_SUCCESS;
}

સ્થિર શૂન્ય ggml_backend_riscv64_spacemit_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);

    શૂન્ય * base = buffer->context;
    જો (base == nullptr) {
        પાછા;
    }

    ggml::cpu::riscv64_spacemit::spine_mem_pool_free(base);
}

સ્થિર શૂન્ય * ggml_backend_riscv64_spacemit_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);

    શૂન્ય * base = buffer->context;
    GGML_ASSERT(base != nullptr);
    પાછા base;
}

સ્થિર શૂન્ય ggml_backend_riscv64_spacemit_buffer_memset_tensor(ggml_backend_buffer_t buffer,
                                                               ggml_tensor *         tensor,
                                                               uint8_t               value,
                                                               size_t                offset,
                                                               size_t                size) {
    GGML_ASSERT(tensor);
    memset((અક્ષર *) tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

સ્થિર શૂન્ય ggml_backend_riscv64_spacemit_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);

    શૂન્ય * base = buffer->context;
    GGML_ASSERT(base != nullptr);
    memset(base, value, buffer->size);
}

સ્થિર શૂન્ય ggml_backend_riscv64_spacemit_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                            ggml_tensor *         tensor,
                                                            અચલ શૂન્ય *          data,
                                                            size_t                offset,
                                                            size_t                size) {
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    auto tensor_traits = (ggml::cpu::riscv64_spacemit::tensor_traits_base *) tensor->extra;
    જો (tensor_traits) {
        auto OK = tensor_traits->repack(tensor, data, size);
        GGML_ASSERT(OK == 0);
    }

    GGML_UNUSED(buffer);
}

સ્થિર અચલ ggml_backend_buffer_i ggml_backend_riscv64_spacemit_buffer_i = {
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

સ્થિર અચલ અક્ષર * ggml_backend_cpu_riscv64_spacemit_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    પાછા "CPU_RISCV64_SPACEMIT";

    GGML_UNUSED(buft);
}

સ્થિર ggml_backend_buffer_t ggml_backend_cpu_riscv64_spacemit_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                                                                        size_t size) {
    શૂન્ય * base = ggml::cpu::riscv64_spacemit::spine_mem_pool_alloc(size, 64);
    જો (base == nullptr) {
        પાછા nullptr;
    }

    પાછા ggml_backend_buffer_init(buft, ggml_backend_riscv64_spacemit_buffer_i, base, size);
}

સ્થિર size_t ggml_backend_cpu_riscv64_spacemit_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    પાછા 64;

    GGML_UNUSED(buft);
}

સ્થિર size_t ggml_backend_cpu_riscv64_spacemit_nbytes(ggml_backend_buffer_type_t buft, અચલ ggml_tensor * tensor) {
    માટે (પૂર્ણાંક i = 0; i < GGML_MAX_DIMS; ++i) {
        જો (tensor->ne[i] <= 0) {
            પાછા 0;
        }
    }

    GGML_UNUSED(buft);

    અચલ auto plain_nbytes = [&]() {
        size_t total = ggml_type_size(tensor->type);
        માટે (પૂર્ણાંક i = 0; i < GGML_MAX_DIMS; ++i) {
            total += (tensor->ne[i] - 1) * tensor->nb[i];
        }
        પાછા total;
    };

    અચલ size_t blck_size = ggml_blck_size(tensor->type);
    જો (blck_size == 1) {
        પાછા plain_nbytes();
    }

    અચલ size_t row_nbytes = tensor->ne[0] * tensor->nb[0] / blck_size;

    અચલ auto add_strided_nbytes = [&](size_t total, size_t src_block_size, size_t dst_block_size) {
        માટે (પૂર્ણાંક i = 1; i < GGML_MAX_DIMS; ++i) {
            total += (tensor->ne[i] - 1) * (tensor->nb[i] / src_block_size) * dst_block_size;
        }
        પાછા total;
    };

    અચલ auto remap_block_nbytes = [&](size_t src_block_size, size_t dst_block_size, int64_t padded_rows = 0) {
        GGML_ASSERT(row_nbytes % src_block_size == 0);

        size_t total =
            add_strided_nbytes((row_nbytes / src_block_size) * dst_block_size, src_block_size, dst_block_size);

        જો (padded_rows > 0 && tensor->ne[1] % padded_rows != 0) {
            total += (padded_rows - tensor->ne[1] % padded_rows) * (tensor->nb[1] / src_block_size) * dst_block_size;
        }

        પાછા total;
    };

    size_t nbytes = row_nbytes;
    પસંદગી (tensor->type) {
        સ્થિતિ GGML_TYPE_Q4_K:
            nbytes = remap_block_nbytes(કદ(block_q4_K), કદ(block_q4_1) * 8);
            તોડો;
        સ્થિતિ GGML_TYPE_Q6_K:
            nbytes = remap_block_nbytes(કદ(block_q6_K), કદ(block_q8_0) * 8, 32);
            તોડો;
        સ્થિતિ GGML_TYPE_Q8_0:
            nbytes = remap_block_nbytes(કદ(block_q8_0), કદ(block_q8_0), 32);
            તોડો;
        સ્થિતિ GGML_TYPE_Q2_K:
            nbytes = remap_block_nbytes(કદ(block_q2_K), કદ(spacemit_kernels::nrow_block_q2_k<1>));
            તોડો;
        સ્થિતિ GGML_TYPE_Q3_K:
            nbytes = remap_block_nbytes(કદ(block_q3_K), કદ(spacemit_kernels::nrow_block_q3_k<1>));
            તોડો;
        સ્થિતિ GGML_TYPE_MXFP4:
            nbytes = remap_block_nbytes(કદ(block_mxfp4), કદ(spacemit_kernels::nrow_block_mxfp4<1>));
            તોડો;
        સ્થિતિ GGML_TYPE_Q5_K:
            nbytes = remap_block_nbytes(કદ(block_q5_K), કદ(spacemit_kernels::nrow_block_q5_1<1>) * 8);
            તોડો;
        સ્થિતિ GGML_TYPE_Q5_1:
            nbytes = remap_block_nbytes(કદ(block_q5_1), કદ(spacemit_kernels::nrow_block_q5_1<1>));
            તોડો;
        સ્થિતિ GGML_TYPE_Q5_0:
            nbytes = remap_block_nbytes(કદ(block_q5_0), કદ(spacemit_kernels::nrow_block_q5_0<1>));
            તોડો;
        મૂળભૂત:
            nbytes = add_strided_nbytes(row_nbytes, 1, 1);
            તોડો;
    }

    પાછા nbytes;
}

નામાકાશ ggml::cpu::riscv64_spacemit {

વર્ગ extra_buffer_type : ggml::cpu::extra_buffer_type {
    તાર્કિક supports_op(ggml_backend_dev_t, અચલ ggml_tensor * op) override {
        પસંદગી (op->op) {
            સ્થિતિ GGML_OP_MUL_MAT:
                જો (op->src[0]->buffer && (ggml_n_dims(op->src[0]) == 2) &&
                    op->src[0]->buffer->buft == ggml_backend_cpu_riscv64_spacemit_buffer_type() &&
                    ggml_riscv64_spacemit_get_optimal_repack_type(op->src[0])) {
                    જો (op->src[1]->buffer && !ggml_backend_buft_is_host(op->src[1]->buffer->buft)) {
                        પાછા ખોટું;
                    }
                    જો (op->src[1]->type == GGML_TYPE_F32) {
                        પાછા સાચું;
                    }
                }
                તોડો;
            સ્થિતિ GGML_OP_MUL_MAT_ID:
                જો (op->src[0]->buffer && (ggml_n_dims(op->src[0]) == 3) &&
                    op->src[0]->buffer->buft == ggml_backend_cpu_riscv64_spacemit_buffer_type() &&
                    ggml_riscv64_spacemit_get_optimal_repack_type(op->src[0])) {
                    જો (op->src[1]->buffer && !ggml_backend_buft_is_host(op->src[1]->buffer->buft)) {
                        પાછા ખોટું;
                    }
                    જો (op->src[1]->type == GGML_TYPE_F32) {
                        પાછા સાચું;
                    }
                }
                તોડો;
            મૂળભૂત:
                // GGML_ABORT("fatal error");
                તોડો;
        }
        પાછા ખોટું;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(અચલ ggml_tensor * op) override {
        પસંદગી (op->op) {
            સ્થિતિ GGML_OP_MUL_MAT:
            સ્થિતિ GGML_OP_MUL_MAT_ID:
                જો (op->src[0]->buffer && op->src[0]->buffer->buft == ggml_backend_cpu_riscv64_spacemit_buffer_type()) {
                    પાછા (ggml::cpu::tensor_traits *) op->src[0]->extra;
                }
                તોડો;
            સ્થિતિ GGML_OP_NORM:
            સ્થિતિ GGML_OP_RMS_NORM:
            સ્થિતિ GGML_OP_ADD:
            સ્થિતિ GGML_OP_SUB:
            સ્થિતિ GGML_OP_MUL:
            સ્થિતિ GGML_OP_DIV:
            સ્થિતિ GGML_OP_FLASH_ATTN_EXT:
            સ્થિતિ GGML_OP_CONT:
            સ્થિતિ GGML_OP_CPY:
            સ્થિતિ GGML_OP_REPEAT:
            સ્થિતિ GGML_OP_SUM_ROWS:
            સ્થિતિ GGML_OP_GET_ROWS:
            સ્થિતિ GGML_OP_CONCAT:
                // case GGML_OP_GATED_DELTA_NET:
                પાછા (ggml::cpu::tensor_traits *) (&ggml::cpu::riscv64_spacemit::rvv_impl);
            મૂળભૂત:
                // GGML_ABORT("fatal error");
                તોડો;
        }

        પાછા nullptr;
    }
};

}  // namespace ggml::cpu::riscv64_spacemit

ggml_backend_buffer_type_t ggml_backend_cpu_riscv64_spacemit_buffer_type(શૂન્ય) {
    સ્થિર ggml_backend_buffer_type ggml_backend_cpu_buffer_type_riscv64_spacemit = {
  /* .iface    = */
        {
         /* .get_name           = */ ggml_backend_cpu_riscv64_spacemit_buffer_type_get_name,
         /* .alloc_buffer       = */ ggml_backend_cpu_riscv64_spacemit_buffer_type_alloc_buffer,
         /* .alloc_buffer_n     = */ ખાલી,
         /* .get_alignment      = */ ggml_backend_cpu_riscv64_spacemit_buffer_type_get_alignment,
         /* .get_max_size       = */ nullptr,
         /* .get_alloc_size     = */ ggml_backend_cpu_riscv64_spacemit_nbytes,
         /* .get_alloc_size_n   = */ ખાલી,
         /* .is_host            = */ nullptr,
         },
 /* .device  = */
        ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
 /* .context = */
        નવું ggml::cpu::riscv64_spacemit::extra_buffer_type(),
    };

    પાછા &ggml_backend_cpu_buffer_type_riscv64_spacemit;
}

બાહ્ય "C" {
સ્થિર પૂર્ણાંક bind_ai_thread() {
    પૂર્ણાંક  fd, bytes;
    અક્ષર str[32];

    fd = open("/proc/set_ai_thread", O_WRONLY);
    જો (fd < 0) {
        GGML_LOG_ERROR("try open /proc/set_ai_thread failed\n");
        પાછા -1;
    }

    snprintf(str, 16, "%d", 0);
    bytes = લિખ(fd, str, strlen(str));
    જો (bytes < 0) {
        GGML_LOG_ERROR("try write /proc/set_ai_thread failed\n");
        close(fd);
        પાછા -1;
    }

    close(fd);
    પાછા 0;
}

શૂન્ય ggml_backend_cpu_riscv64_spacemit_set_numa_thread_affinity(પૂર્ણાંક thread_n) {
    પૂર્ણાંક cpu_id = sched_getcpu();
    જો (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2 &&
        !((1 << cpu_id) & ggml::cpu::riscv64_spacemit::global_spine_env_info.cpu_mask)) {
        GGML_PRINT_DEBUG("bind_ai_thread for thread %d, pid %d\n", thread_n, getpid());
        bind_ai_thread();
    }

    જો (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_tcm &&
        ggml::cpu::riscv64_spacemit::tls_context.cpu_id == -1) {
        CPU_ZERO(&(ggml::cpu::riscv64_spacemit::tls_context.cpuset));
        pthread_t    main_thread     = pthread_self();
        અચલ auto & perfer_core_ids = ggml::cpu::riscv64_spacemit::global_spine_env_info.perfer_core_ids;
        જો (thread_n < 0 || static_cast<size_t>(thread_n) >= perfer_core_ids.size()) {
            GGML_ABORT("thread_n %d exceeds perfer_core_ids size %zu\n", thread_n, perfer_core_ids.size());
        }
        auto perfer_cpu_id = perfer_core_ids[static_cast<size_t>(thread_n)];
        CPU_SET(perfer_cpu_id, &(ggml::cpu::riscv64_spacemit::tls_context.cpuset));
        પૂર્ણાંક s =
            pthread_setaffinity_np(main_thread, કદ(cpu_set_t), &(ggml::cpu::riscv64_spacemit::tls_context.cpuset));
        જો (s != 0) {
            GGML_ABORT("set thread affinity error for thread_n %d, cpu_id %d\n", thread_n, perfer_cpu_id);
        }

        પૂર્ણાંક ai_cpu_id = perfer_cpu_id - ggml::cpu::riscv64_spacemit::global_spine_env_info.aicpu_id_offset;
        ggml::cpu::riscv64_spacemit::tls_context.cpu_id = ai_cpu_id;
        ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer =
            ggml::cpu::riscv64_spacemit::spine_mem_pool_tcm_mem_get(ai_cpu_id);
        ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer_size =
            ggml::cpu::riscv64_spacemit::global_spine_env_info.tcm_blk_size;
    }

    જો (ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer != nullptr) {
        શૂન્ય * rt =
            ggml::cpu::riscv64_spacemit::spine_mem_pool_tcm_mem_wait(ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        જો (rt == nullptr) {
            GGML_ABORT("wait tcm buffer failed for cpu_id: %d", ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        }
    }
}

શૂન્ય ggml_backend_cpu_riscv64_spacemit_clear_numa_thread_affinity_threaded(પૂર્ણાંક thread_n) {
    જો (ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer != nullptr) {
        auto rt = ggml::cpu::riscv64_spacemit::spine_mem_pool_tcm_mem_release(
            ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        જો (rt != 0) {
            GGML_ABORT("release tcm buffer failed for cpu_id: %d", ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        }
    }
}
}
