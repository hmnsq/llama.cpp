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
#जर defined(__riscv)

#जर !defined(__riscv_v) || !defined(__riscv_v_intrinsic)
#error "riscv v extension or v_intrinsic not enabled"
#अन्यथा
#अन्तर्भाव <riscv_vector.h>
#endif

#जर !defined(__riscv_zfh) || !defined(__riscv_zvfh)
#error "riscv zfh extension not enabled, GGML_RV_ZFH and GGML_RV_ZVFH must be defined to 1"
#endif

#जर !defined(__riscv_zba)
#error "riscv zba extension not enabled, GGML_RV_ZBA must be defined to 1"
#endif

#जर defined(RISCV64_SPACEMIT_IME1) || defined(RISCV64_SPACEMIT_IME2)
#अन्यथा
#error "RISCV64_SPACEMIT_IME1 or RISCV64_SPACEMIT_IME2 not defined"
#endif

#अन्यथा

#error "riscv not enabled in this build"

#endif

#जर defined(__GNUC__)
#pragma GCC diagnostic ignored "-Woverlength-strings"
#pragma GCC diagnostic ignored "-Wcast-qual"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

// clang-format on

बाह्य "C" {
बाह्य शून्य ggml_threadpool_chunk_set(संरचना ggml_threadpool * tp, पूर्णांक value);
बाह्य पूर्णांक  ggml_threadpool_chunk_add(संरचना ggml_threadpool * tp, पूर्णांक value);
}

नामाकाश ggml::cpu::riscv64_spacemit {

संरचना TLSContext {
    पूर्णांक       cpu_id{ -1 };
    cpu_set_t cpuset;
    शून्य *    tcm_buffer{ nullptr };
    माप_प्रकार    tcm_buffer_size{ 0 };
};

thread_local TLSContext tls_context;

ढाँचा <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS> constexpr माप_प्रकार get_repacked_block_type_size() {
    जर constexpr (std::is_same_v<BLOC_TYPE, block_q6_K> || std::is_same_v<BLOC_TYPE, block_q8_0>) {
        परत आकार(block_q8_0);
    } अन्यथा जर constexpr (std::is_same_v<BLOC_TYPE, block_q4_0>) {
        परत आकार(block_q4_0) * INTER_SIZE / QK4_0;
    } अन्यथा जर constexpr (std::is_same_v<BLOC_TYPE, block_q4_1> || std::is_same_v<BLOC_TYPE, block_q4_K>) {
        परत (आकार(block_q4_0) + आकार(uint8_t)) * INTER_SIZE / QK4_1;
    } अन्यथा जर constexpr (std::is_same_v<BLOC_TYPE, block_q2_K>) {
        परत आकार(spacemit_kernels::nrow_block_q2_k<1>);
    } अन्यथा जर constexpr (std::is_same_v<BLOC_TYPE, block_q3_K>) {
        परत आकार(spacemit_kernels::nrow_block_q3_k<1>);
    } अन्यथा जर constexpr (std::is_same_v<BLOC_TYPE, block_mxfp4>) {
        परत आकार(spacemit_kernels::nrow_block_mxfp4<1>);
    } अन्यथा जर constexpr (std::is_same_v<BLOC_TYPE, block_q5_1> || std::is_same_v<BLOC_TYPE, block_q5_K>) {
        परत आकार(spacemit_kernels::nrow_block_q5_1<1>);
    } अन्यथा जर constexpr (std::is_same_v<BLOC_TYPE, block_q5_0>) {
        परत आकार(spacemit_kernels::nrow_block_q5_0<1>);
    } अन्यथा {
        प्रतिज्ञा(असत्य);
        परत 0;
    }
}

ढाँचा <typename BLOC_TYPE> constexpr तार्किक block_type_has_zp() {
    जर constexpr (std::is_same_v<BLOC_TYPE, block_q6_K> || std::is_same_v<BLOC_TYPE, block_q8_0> ||
                  std::is_same_v<BLOC_TYPE, block_q3_K> || std::is_same_v<BLOC_TYPE, block_q4_0> ||
                  std::is_same_v<BLOC_TYPE, block_mxfp4> || std::is_same_v<BLOC_TYPE, block_q5_0>) {
        परत असत्य;
    } अन्यथा जर constexpr (std::is_same_v<BLOC_TYPE, block_q4_1> || std::is_same_v<BLOC_TYPE, block_q4_K> ||
                         std::is_same_v<BLOC_TYPE, block_q2_K> || std::is_same_v<BLOC_TYPE, block_q5_1> ||
                         std::is_same_v<BLOC_TYPE, block_q5_K>) {
        परत सत्य;
    } अन्यथा {
        प्रतिज्ञा(असत्य);
        परत असत्य;
    }
}

वर्ग tensor_traits_base : सार्वजनिक ggml::cpu::tensor_traits {
  सार्वजनिक:
    आभासी पूर्णांक repack(ggml_tensor * t, स्थिरांक शून्य * data, माप_प्रकार data_size) = 0;
};

ढाँचा <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS> वर्ग tensor_traits : सार्वजनिक tensor_traits_base {
    तार्किक work_size(पूर्णांक /* n_threads */, स्थिरांक ggml_tensor * op, माप_प्रकार & size) override {
        निवड (op->op) {
            स्थिती GGML_OP_MUL_MAT:
                {
                    int64_t src1_nelements = ggml_nelements(op->src[1]);

                    जर constexpr (std::is_same_v<BLOC_TYPE, block_q2_K> || std::is_same_v<BLOC_TYPE, block_q3_K>) {
                        size =
                            spacemit_kernels::div_round_up(src1_nelements, QK_K) * spacemit_kernels::q8k_blk_size(QK_K);
                    } अन्यथा जर constexpr (INTER_SIZE == QK4_0) {
                        size = spacemit_kernels::div_round_up(src1_nelements, QK4_0) *
                               spacemit_kernels::q8_blk_size(QK4_0, सत्य);
                    } अन्यथा जर constexpr (INTER_SIZE == 256) {
                        size = spacemit_kernels::div_round_up(src1_nelements, 256) *
                               spacemit_kernels::q8_hp_blk_size(256, सत्य, सत्य);
                    } अन्यथा {
                        GGML_ABORT("unsupported block type");
                    }

                    size = GGML_PAD(size, आकार(int64_t));

                    परत सत्य;
                }
            स्थिती GGML_OP_MUL_MAT_ID:
                {
                    int64_t src1_nelements = ggml_nelements(op->src[1]);

                    जर constexpr (std::is_same_v<BLOC_TYPE, block_q2_K> || std::is_same_v<BLOC_TYPE, block_q3_K>) {
                        size =
                            spacemit_kernels::div_round_up(src1_nelements, QK_K) * spacemit_kernels::q8k_blk_size(QK_K);
                    } अन्यथा जर constexpr (INTER_SIZE == QK4_0) {
                        size = spacemit_kernels::div_round_up(src1_nelements, QK4_0) *
                               spacemit_kernels::q8_blk_size(QK4_0, सत्य);
                    } अन्यथा जर constexpr (INTER_SIZE == 256) {
                        size = spacemit_kernels::div_round_up(src1_nelements, 256) *
                               spacemit_kernels::q8_hp_blk_size(256, सत्य, सत्य);
                    } अन्यथा {
                        GGML_ABORT("unsupported block type");
                    }

                    size = GGML_PAD(size, आकार(int64_t));

                    स्थिरांक int64_t ne02 = op->src[0]->ne[2];  // n_as, n_expert
                    स्थिरांक int64_t ne12 = op->src[1]->ne[2];  // n_tokens

                    स्थिरांक माप_प्रकार sizeof_mmid_row_mapping = आकार(int64_t);
                    size += sizeof_mmid_row_mapping * ne02 * (ne12 + 1) + (ne02 + 1) * आकार(int64_t);

                    size = GGML_PAD(size, आकार(int64_t));

                    परत सत्य;
                }
            पूर्वनिर्धारित:
                // GGML_ABORT("fatal error");
                थांबा;
        }
        परत असत्य;
    }

    तार्किक compute_forward(ggml_compute_params * params, ggml_tensor * op) override {
        निवड (op->op) {
            स्थिती GGML_OP_MUL_MAT:
                निवड (op->src[0]->type) {
                    स्थिती GGML_TYPE_Q2_K:
                    स्थिती GGML_TYPE_Q3_K:
                    स्थिती GGML_TYPE_Q4_0:
                    स्थिती GGML_TYPE_Q4_1:
                    स्थिती GGML_TYPE_Q4_K:
                    स्थिती GGML_TYPE_Q6_K:
                    स्थिती GGML_TYPE_Q8_0:
                    स्थिती GGML_TYPE_Q5_0:
                    स्थिती GGML_TYPE_Q5_1:
                    स्थिती GGML_TYPE_Q5_K:
                        //case GGML_TYPE_MXFP4:
                        forward_mul_mat(params, op);
                        परत सत्य;
                    पूर्वनिर्धारित:
                        // GGML_ABORT("fatal error: unsupported type for src0 in MUL_MAT");
                        परत असत्य;
                }
                थांबा;
            स्थिती GGML_OP_MUL_MAT_ID:
                निवड (op->src[0]->type) {
                    स्थिती GGML_TYPE_Q2_K:
                    स्थिती GGML_TYPE_Q3_K:
                    स्थिती GGML_TYPE_Q4_0:
                    स्थिती GGML_TYPE_Q4_1:
                    स्थिती GGML_TYPE_Q4_K:
                    स्थिती GGML_TYPE_Q6_K:
                    स्थिती GGML_TYPE_Q8_0:
                    स्थिती GGML_TYPE_Q5_0:
                    स्थिती GGML_TYPE_Q5_1:
                    स्थिती GGML_TYPE_Q5_K:
                        //case GGML_TYPE_MXFP4:
                        forward_mul_mat_id(params, op);
                        परत सत्य;
                    पूर्वनिर्धारित:
                        // GGML_ABORT("fatal error: unsupported type for src0 in MUL_MAT_ID");
                        परत असत्य;
                }
                थांबा;
            पूर्वनिर्धारित:
                // GGML_ABORT("fatal error");
                थांबा;
        }
        परत असत्य;
    }

    शून्य forward_mul_mat(ggml_compute_params * params, ggml_tensor * op) {
        constexpr माप_प्रकार a_blk_len = INTER_SIZE;
        constexpr माप_प्रकार b_blk_len = INTER_SIZE;

        स्थिरांक ggml_tensor * src0 = op->src[0];
        स्थिरांक ggml_tensor * src1 = op->src[1];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_द्विमान_OP_LOCALS

        पूर्णांक ith = params->ith;
        पूर्णांक nth = params->nth;

        [[maybe_unused]] स्थिरांक गणना ggml_type type = src0->type;

        शून्य *        w_data  = (शून्य *) src0->data;
        स्थिरांक दशांश * feature = (स्थिरांक दशांश *) src1->data;
        दशांश *       output  = (दशांश *) dst->data;

        स्थिरांक int64_t gemm_m = ne11 * ne12 * ne13;
        स्थिरांक int64_t gemm_k = ne10;
        स्थिरांक int64_t gemm_n = ne01;

        spacemit_kernels::quantize_a_row_def       quantize_a_row_i8;
        spacemit_kernels::quantize_a_row_def       quantize_a_4row_i8;
        spacemit_kernels::gemm_kernel_quantize_def gemm_kernel;
        तार्किक                                       set_kernel_impl = असत्य;

        int64_t block_stride_a = spacemit_kernels::q8_blk_size(a_blk_len);

#जर defined(RISCV64_SPACEMIT_IME2)
        जर (!set_kernel_impl && (global_spine_env_info.use_ime2)) {
            quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8;
            quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8;
            block_stride_a     = spacemit_kernels::q8_blk_size(a_blk_len, सत्य);

            जर constexpr (std::is_same_v<BLOC_TYPE, block_q6_K> || std::is_same_v<BLOC_TYPE, block_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i8;
                set_kernel_impl = सत्य;
            } अन्यथा जर constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> || std::is_same_v<BLOC_TYPE, block_q4_1> ||
                                 std::is_same_v<BLOC_TYPE, block_q4_K>) {
                जर constexpr (INTER_SIZE == 256) {
                    gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i4_hp;
                    quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8_hp;
                    quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8_hp;
                    block_stride_a     = spacemit_kernels::q8_hp_blk_size(a_blk_len, सत्य, सत्य);
                    set_kernel_impl    = सत्य;
                } अन्यथा {
                    gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i4;
                    quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8;
                    quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8;
                    block_stride_a     = spacemit_kernels::q8_blk_size(a_blk_len, सत्य);
                    set_kernel_impl    = सत्य;
                }
            } अन्यथा जर constexpr (std::is_same_v<BLOC_TYPE, block_q2_K>) {
                quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8k;
                quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8k;
                block_stride_a     = spacemit_kernels::q8k_blk_size(a_blk_len);

                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i2k;
                set_kernel_impl = सत्य;
            } अन्यथा जर constexpr (std::is_same_v<BLOC_TYPE, block_q3_K>) {
                quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8k;
                quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8k;
                block_stride_a     = spacemit_kernels::q8k_blk_size(a_blk_len);

                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i3k;
                set_kernel_impl = सत्य;
            } अन्यथा जर constexpr (std::is_same_v<BLOC_TYPE, block_mxfp4>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8mxfp4;
                set_kernel_impl = सत्य;
            } अन्यथा जर constexpr (std::is_same_v<BLOC_TYPE, block_q5_1> || std::is_same_v<BLOC_TYPE, block_q5_K> ||
                                 std::is_same_v<BLOC_TYPE, block_q5_0>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i5;
                set_kernel_impl = सत्य;
            }
        }
#endif

#जर defined(RISCV64_SPACEMIT_IME1)
        जर (!set_kernel_impl && (global_spine_env_info.use_ime1)) {
            quantize_a_row_i8  = spacemit_kernels::ime1::quantize_a_row_i8;
            quantize_a_4row_i8 = spacemit_kernels::ime1::quantize_a_4row_i8;

            जर constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> || std::is_same_v<BLOC_TYPE, block_q4_1> ||
                          std::is_same_v<BLOC_TYPE, block_q4_K>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i4;
                set_kernel_impl = सत्य;
            } अन्यथा जर constexpr (std::is_same_v<BLOC_TYPE, block_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i8;
                set_kernel_impl = सत्य;
            }
        }
#endif
        जर (!set_kernel_impl) {
            GGML_ABORT("no kernel implementation found for the block type");
        }

        स्थिरांक int64_t a_k_blks = spacemit_kernels::div_round_up(gemm_k, a_blk_len);
        स्थिरांक int64_t b_k_blks = spacemit_kernels::div_round_up(gemm_k, b_blk_len);

        स्थिरांक int64_t row_stride_a        = a_k_blks * block_stride_a;
        स्थिरांक int64_t gemm_workspace_size = GGML_PAD(gemm_m * row_stride_a, alignof(int64_t));

        जर (ith == 0 && params->wsize < gemm_workspace_size) {
            GGML_ABORT("wsize less than gemm_workspace_size");
        }

        uintptr_t ws_ptr = reinterpret_cast<uintptr_t>(params->wdata);

        शून्य *        tcm_buffer      = ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer;
        स्थिरांक int64_t tcm_buffer_size = ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer_size;

        स्वतः * quant_a_buffer = reinterpret_cast<uint8_t *>(ws_ptr);

        constexpr int64_t row_align = 4;
        स्थिरांक int64_t     row_blks  = spacemit_kernels::div_round_up(gemm_m, row_align);

        स्थिरांक int64_t row_stride_b      = b_k_blks * get_repacked_block_type_size<BLOC_TYPE, INTER_SIZE, NB_COLS>();
        स्थिरांक int64_t per_mb_rows_wsize = row_align * row_stride_a;
        स्थिरांक int64_t per_nb_cols_wsize = NB_COLS * row_stride_b;

        स्थिरांक int64_t barrier_idx = static_cast<int64_t>(ith / 2);

        GGML_ASSERT(global_spine_env_info.init_barrier != nullptr);
        GGML_ASSERT(barrier_idx < spine_init_barrier_मोजणी);
        spine_barrier_t * cur_barrier = &global_spine_env_info.init_barrier[barrier_idx];

        जर (gemm_m == 1) {
            पूर्णांक task_per_thread = spacemit_kernels::div_round_up(a_k_blks, nth);
            पूर्णांक a_blk_start     = ith * task_per_thread;
            पूर्णांक a_blk_end       = std::min(a_blk_start + task_per_thread, (पूर्णांक) a_k_blks);
            जर (a_blk_start < a_blk_end) {
                quantize_a_row_i8(a_blk_len, feature + a_blk_start * a_blk_len, (a_blk_end - a_blk_start) * a_blk_len,
                                  quant_a_buffer + a_blk_start * block_stride_a);
            }
        } अन्यथा {
            पूर्णांक task_per_thread = spacemit_kernels::div_round_up(row_blks, nth);
            पूर्णांक m_row_blk_start = ith * task_per_thread;
            पूर्णांक m_row_blk_end   = std::min(m_row_blk_start + task_per_thread, (पूर्णांक) row_blks);
            साठी (पूर्णांक m_row_blk = m_row_blk_start; m_row_blk < m_row_blk_end; m_row_blk++) {
                पूर्णांक m_idx             = m_row_blk * row_align;
                पूर्णांक rows_tobe_handled = (gemm_m - m_idx) > row_align ? row_align : (gemm_m - m_idx);

                जर (rows_tobe_handled == row_align && quantize_a_4row_i8 != nullptr) {
                    स्थिरांक दशांश * a_row_ptr       = feature + m_idx * gemm_k;
                    स्वतः *        quant_a_row_ptr = quant_a_buffer + m_idx * row_stride_a;
                    quantize_a_4row_i8(a_blk_len, a_row_ptr, gemm_k, quant_a_row_ptr);
                } अन्यथा {
                    पर्यंत (rows_tobe_handled) {
                        स्थिरांक दशांश * a_row_ptr       = feature + m_idx * gemm_k;
                        स्वतः *        quant_a_row_ptr = quant_a_buffer + m_idx * row_stride_a;
                        quantize_a_row_i8(a_blk_len, a_row_ptr, gemm_k, quant_a_row_ptr);
                        rows_tobe_handled -= 1;
                        m_idx += 1;
                    }
                }
            }
        }

        ggml_barrier(params->threadpool);

        स्थिरांक int64_t gemm_m_stride     = gemm_n / gemm_m > 64 ? gemm_m : 16;
        स्थिरांक int64_t gemm_m_blocked    = spacemit_kernels::div_round_up(gemm_m, gemm_m_stride);
        स्थिरांक int64_t max_gemm_n_stride = spacemit_kernels::div_round_up(gemm_n * gemm_m_blocked, nth);

        int64_t gemm_n_stride = gemm_n;
        जर (max_gemm_n_stride < gemm_n) {
            gemm_n_stride =
                std::min(gemm_n_stride, spacemit_kernels::div_round_up(max_gemm_n_stride, NB_COLS) * NB_COLS);
        }

        जर (gemm_n_stride == gemm_n && tcm_buffer != nullptr && per_mb_rows_wsize <= tcm_buffer_size) {
            साठी (int64_t m_start = ith * row_align; m_start < gemm_m; m_start += row_align * nth) {
                uint8_t * b_col    = reinterpret_cast<uint8_t *>(w_data);
                uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? b_col : nullptr;

                int64_t m_row_real = std::min(gemm_m - m_start, row_align);

                spacemit_kernels::rvv::memcpy1d(tcm_buffer, quant_a_buffer + m_start * row_stride_a,
                                                m_row_real * row_stride_a);

                int64_t n_blk_real = 0;
                साठी (int64_t ni = 0; ni < gemm_n; ni += n_blk_real, b_col += n_blk_real * row_stride_b) {
                    n_blk_real = std::min(gemm_n - ni, (int64_t) NB_COLS);

                    uint8_t * a_row_ptr = (uint8_t *) tcm_buffer;
                    दशांश *   c_blk     = output + m_start * gemm_n + ni;

                    int32_t rows_remaining = m_row_real;

                    पर्यंत (rows_remaining > 0) {
                        स्वतः rows_handled = gemm_kernel(b_blk_len, a_row_ptr, b_col, b_col_zp, c_blk, rows_remaining,
                                                        n_blk_real, b_k_blks, gemm_n);

                        c_blk += rows_handled * gemm_n;
                        a_row_ptr += rows_handled * row_stride_a;

                        rows_remaining -= rows_handled;
                    }
                }
            }
        } अन्यथा जर (tcm_buffer != nullptr && per_nb_cols_wsize <= tcm_buffer_size) {
            uint8_t * a_row = quant_a_buffer;
            uint8_t * b_col = reinterpret_cast<uint8_t *>(tcm_buffer);
            जर ((gemm_workspace_size + per_nb_cols_wsize) <= tcm_buffer_size) {
                a_row = (uint8_t *) tcm_buffer;
                b_col = reinterpret_cast<uint8_t *>(tcm_buffer) + gemm_workspace_size;
            }
            uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? b_col : nullptr;

            int64_t ni      = ith * NB_COLS;
            int64_t nb_real = std::min(gemm_n - ni, NB_COLS);

            जर (ith % 2 == 0 && nb_real > 0) {
                spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(w_data) + ni * row_stride_b,
                                                nb_real * row_stride_b);
                जर (a_row != quant_a_buffer) {
                    spacemit_kernels::rvv::memcpy1d(a_row, quant_a_buffer, gemm_workspace_size);
                }
            }

            spine_barrier_wait(cur_barrier);

            जर (ith % 2 != 0 && nb_real > 0) {
                जर (a_row != quant_a_buffer) {
                    spacemit_kernels::rvv::memcpy1d(a_row, quant_a_buffer, gemm_workspace_size);
                }
                spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(w_data) + ni * row_stride_b,
                                                nb_real * row_stride_b);
            }

            साठी (; ni < gemm_n; ni += NB_COLS * nth) {
                int64_t rows_remaining = gemm_m;
                दशांश * c_blk          = output + ni;
                स्वतः *  a_row_cur      = a_row;

                जर (ith % 2 != 0) {
                    spine_barrier_wait(cur_barrier);
                }

                पर्यंत (rows_remaining > 0) {
                    स्वतः rows_handled = gemm_kernel(b_blk_len, a_row_cur, b_col, b_col_zp, c_blk, rows_remaining,
                                                    nb_real, b_k_blks, gemm_n);

                    c_blk += rows_handled * gemm_n;
                    a_row_cur += rows_handled * row_stride_a;

                    rows_remaining -= rows_handled;
                }

                जर (ith % 2 == 0) {
                    spine_barrier_wait(cur_barrier);
                }

                स्थिरांक int64_t next_ni = ni + NB_COLS * nth;
                जर (next_ni < gemm_n) {
                    nb_real = std::min(gemm_n - next_ni, NB_COLS);
                    spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(w_data) + next_ni * row_stride_b,
                                                    nb_real * row_stride_b);
                }
            }
        } अन्यथा {
            स्थिरांक int64_t task_मोजणी_m = spacemit_kernels::div_round_up(gemm_m, gemm_m_stride);
            स्थिरांक int64_t task_मोजणी_n = spacemit_kernels::div_round_up(gemm_n, gemm_n_stride);

            int64_t task_मोजणी      = task_मोजणी_m * task_मोजणी_n;
            int64_t task_per_thread = (task_मोजणी + nth - 1) / nth;
            int64_t start           = ith * task_per_thread;
            int64_t end             = std::min((ith + 1) * task_per_thread, task_मोजणी);
            साठी (int64_t compute_idx = start; compute_idx < end; compute_idx++) {
                स्थिरांक स्वतः tid_n = compute_idx / task_मोजणी_m;
                स्थिरांक स्वतः tid_m = compute_idx % task_मोजणी_m;

                स्थिरांक int64_t m_start = tid_m * gemm_m_stride;
                स्थिरांक int64_t m_मोजणी = std::min(gemm_m - m_start, (int64_t) gemm_m_stride);

                स्थिरांक int64_t n_start = tid_n * gemm_n_stride;
                स्थिरांक int64_t n_मोजणी = std::min(gemm_n - n_start, (int64_t) gemm_n_stride);

                स्थिरांक int64_t n_blk = m_मोजणी == 1 ? n_मोजणी : NB_COLS;

                uint8_t * b_col    = reinterpret_cast<uint8_t *>(w_data) + n_start * row_stride_b;
                uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? b_col : nullptr;

                int64_t n_blk_real = 0;
                साठी (int64_t ni = 0; ni < n_मोजणी; ni += n_blk_real, b_col += n_blk_real * row_stride_b) {
                    n_blk_real = std::min(n_मोजणी - ni, n_blk);

                    uint8_t * a_row = quant_a_buffer + m_start * row_stride_a;

                    दशांश * c_blk = output + m_start * gemm_n + n_start + ni;

                    int64_t rows_remaining = m_मोजणी;

                    uint8_t * b_col_cur    = b_col;
                    uint8_t * b_col_zp_cur = b_col_zp;

                    पर्यंत (rows_remaining > 0) {
                        स्वतः rows_handled = gemm_kernel(b_blk_len, a_row, b_col_cur, b_col_zp_cur, c_blk,
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
        constexpr माप_प्रकार a_blk_len = INTER_SIZE;
        constexpr माप_प्रकार b_blk_len = INTER_SIZE;

        स्थिरांक ggml_tensor * src0 = op->src[0];
        स्थिरांक ggml_tensor * src1 = op->src[1];
        स्थिरांक ggml_tensor * ids  = op->src[2];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_द्विमान_OP_LOCALS

        पूर्णांक ith = params->ith;
        पूर्णांक nth = params->nth;

        // row groups
        स्थिरांक पूर्णांक n_ids = ids->ne[0];  // n_expert_used
        स्थिरांक पूर्णांक n_as  = ne02;        // n_expert

        संरचना mmid_row_mapping {
            int32_t i1;
            int32_t i2;
        };

        spacemit_kernels::quantize_a_row_def           quantize_a_row_i8;
        spacemit_kernels::gemm_kernel_quantize_def     gemm_kernel;
        spacemit_kernels::moe_gemm_kernel_quantize_def moe_gemm_kernel_m2;
        तार्किक                                           set_kernel_impl = असत्य;
        माप_प्रकार                                         block_stride_a  = spacemit_kernels::q8_blk_size(QK4_0);

#जर defined(RISCV64_SPACEMIT_IME2)
        जर (!set_kernel_impl && (global_spine_env_info.use_ime2)) {
            quantize_a_row_i8 = spacemit_kernels::rvv::quantize_a_row_i8;
            block_stride_a    = spacemit_kernels::q8_blk_size(QK4_0, सत्य);

            जर constexpr (std::is_same_v<BLOC_TYPE, block_q6_K> || std::is_same_v<BLOC_TYPE, block_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i8;
                set_kernel_impl = सत्य;
            } अन्यथा जर constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> || std::is_same_v<BLOC_TYPE, block_q4_1> ||
                                 std::is_same_v<BLOC_TYPE, block_q4_K>) {
                जर constexpr (INTER_SIZE == 256) {
                    gemm_kernel       = spacemit_kernels::ime2::gemm_kernel_i8i4_hp;
                    quantize_a_row_i8 = spacemit_kernels::rvv::quantize_a_row_i8_hp;
                    block_stride_a    = spacemit_kernels::q8_hp_blk_size(a_blk_len, सत्य, सत्य);
                    set_kernel_impl   = सत्य;
                } अन्यथा {
                    gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i4;
                    moe_gemm_kernel_m2 = spacemit_kernels::ime2::moe_m2_gemm_kernel_i8i4;
                    quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8;
                    block_stride_a     = spacemit_kernels::q8_blk_size(a_blk_len, सत्य);
                    set_kernel_impl    = सत्य;
                }
            } अन्यथा जर constexpr (std::is_same_v<BLOC_TYPE, block_q2_K>) {
                quantize_a_row_i8 = spacemit_kernels::rvv::quantize_a_row_i8k;
                block_stride_a    = spacemit_kernels::q8k_blk_size(a_blk_len);
                gemm_kernel       = spacemit_kernels::ime2::gemm_kernel_i8i2k;
                set_kernel_impl   = सत्य;
            } अन्यथा जर constexpr (std::is_same_v<BLOC_TYPE, block_q3_K>) {
                quantize_a_row_i8 = spacemit_kernels::rvv::quantize_a_row_i8k;
                block_stride_a    = spacemit_kernels::q8k_blk_size(a_blk_len);
                gemm_kernel       = spacemit_kernels::ime2::gemm_kernel_i8i3k;
                set_kernel_impl   = सत्य;
            } अन्यथा जर constexpr (std::is_same_v<BLOC_TYPE, block_mxfp4>) {
                gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8mxfp4;
                moe_gemm_kernel_m2 = spacemit_kernels::ime2::moe_m2_gemm_kernel_i8mxfp4;
                set_kernel_impl    = सत्य;
            } अन्यथा जर constexpr (std::is_same_v<BLOC_TYPE, block_q5_1> || std::is_same_v<BLOC_TYPE, block_q5_K> ||
                                 std::is_same_v<BLOC_TYPE, block_q5_0>) {
                gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i5;
                moe_gemm_kernel_m2 = spacemit_kernels::ime2::moe_m2_gemm_kernel_i8i5;
                set_kernel_impl    = सत्य;
            }
        }
#endif

#जर defined(RISCV64_SPACEMIT_IME1)
        जर (!set_kernel_impl && (global_spine_env_info.use_ime1)) {
            quantize_a_row_i8 = spacemit_kernels::ime1::quantize_a_row_i8;

            जर constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> || std::is_same_v<BLOC_TYPE, block_q4_1> ||
                          std::is_same_v<BLOC_TYPE, block_q4_K>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i4;
                set_kernel_impl = सत्य;
            } अन्यथा जर constexpr (std::is_same_v<BLOC_TYPE, block_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i8;
                set_kernel_impl = सत्य;
            }
        }
#endif
        जर (!set_kernel_impl) {
            GGML_ABORT("no kernel implementation found for the block type");
        }

        स्थिरांक माप_प्रकार a_k_blks = spacemit_kernels::div_round_up(ne10, a_blk_len);
        स्थिरांक माप_प्रकार b_k_blks = spacemit_kernels::div_round_up(ne10, b_blk_len);

        स्थिरांक माप_प्रकार nbw1                = a_k_blks * block_stride_a;
        स्थिरांक माप_प्रकार nbw2                = ne11 * nbw1;
        स्थिरांक माप_प्रकार nbw3                = nbw2 * ne12;
        स्थिरांक माप_प्रकार gemm_workspace_size = GGML_PAD(nbw3, alignof(int64_t));

        स्थिरांक uintptr_t ws_ptr         = reinterpret_cast<uintptr_t>(params->wdata);
        स्वतः *          quant_a_buffer = reinterpret_cast<uint8_t *>(ws_ptr);

        जर (ne11 == 1) {
            साठी (int64_t ii = ith; ii < ne12 * a_k_blks; ii += nth) {
                int64_t i12       = ii / a_k_blks;
                int64_t ak_blk_id = ii % a_k_blks;
                quantize_a_row_i8(a_blk_len, (दशांश *) ((अक्षर *) src1->data + i12 * nb12) + ak_blk_id * a_blk_len,
                                  a_blk_len, quant_a_buffer + i12 * nbw2 + ak_blk_id * block_stride_a);
            }
        } अन्यथा {
            साठी (int64_t ii = ith; ii < ne12 * ne11; ii += nth) {
                int64_t i12 = ii / ne11;
                int64_t i11 = ii % ne11;
                quantize_a_row_i8(a_blk_len, (दशांश *) ((अक्षर *) src1->data + i12 * nb12 + i11 * nb11), ne10,
                                  quant_a_buffer + i12 * nbw2 + i11 * nbw1);
            }
        }

#define MMID_MATRIX_ROW(row_id, i1) matrix_rows[(row_id) *ne12 + (i1)]

        int64_t *          matrix_row_counts       = (int64_t *) (ws_ptr + gemm_workspace_size);
        int32_t *          valid_ep_मोजणी          = (int32_t *) (matrix_row_counts + n_as);
        int32_t *          valid_act_मोजणी         = (int32_t *) (valid_ep_मोजणी + 1);
        int64_t *          valid_matrix_row_counts = (int64_t *) (valid_act_मोजणी + 1);
        mmid_row_mapping * matrix_rows             = (mmid_row_mapping *) (valid_matrix_row_counts + n_as);

        जर (ith == 0) {
            // initialize matrix_row_counts
            स_रखो(matrix_row_counts, 0, n_as * आकार(int64_t));

            // group rows by src0 matrix
            साठी (int32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
                साठी (int32_t id = 0; id < n_ids; ++id) {
                    स्थिरांक int32_t i02 =
                        *(स्थिरांक int32_t *) ((स्थिरांक अक्षर *) ids->data + iid1 * ids->nb[1] + id * ids->nb[0]);

                    GGML_ASSERT(i02 >= 0 && i02 < n_as);

                    MMID_MATRIX_ROW(i02, matrix_row_counts[i02]) = { id, iid1 };
                    matrix_row_counts[i02] += 1;
                }
            }

            int32_t valid_ep_मोजणी_t  = 0;
            int32_t valid_act_मोजणी_t = 0;
            साठी (पूर्णांक cur_a = 0; cur_a < n_as; ++cur_a) {
                स्थिरांक int64_t cne1 = matrix_row_counts[cur_a];
                जर (cne1 == 0) {
                    सुरूठेवा;
                }
                valid_matrix_row_counts[valid_ep_मोजणी_t] = cur_a;
                valid_act_मोजणी_t += cne1;
                valid_ep_मोजणी_t += 1;
            }
            valid_ep_मोजणी[0]  = valid_ep_मोजणी_t;
            valid_act_मोजणी[0] = valid_act_मोजणी_t;
        }

        स्थिरांक int64_t barrier_idx = static_cast<int64_t>(ith / 2);

        GGML_ASSERT(global_spine_env_info.init_barrier != nullptr);
        GGML_ASSERT(barrier_idx < spine_init_barrier_मोजणी);
        spine_barrier_t * cur_barrier = &global_spine_env_info.init_barrier[barrier_idx];

        ggml_barrier(params->threadpool);

        स्थिरांक माप_प्रकार row_stride_b      = b_k_blks * get_repacked_block_type_size<BLOC_TYPE, INTER_SIZE, NB_COLS>();
        स्थिरांक माप_प्रकार expert_b_stride   = ne01 * row_stride_b;
        स्थिरांक माप_प्रकार per_nb_cols_wsize = NB_COLS * row_stride_b;

        std::array<स्थिरांक uint8_t *, 2> src_workspaces;
        std::array<दशांश *, 2>         dst_workspaces;

        स्वतः *     tcm_buffer      = ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer;
        स्थिरांक स्वतः tcm_buffer_size = ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer_size;

        स्थिरांक स्वतः valid_ep_मोजणी_t  = valid_ep_मोजणी[0];
        स्थिरांक स्वतः valid_act_मोजणी_t = valid_act_मोजणी[0];

        पूर्णांक nth_es = 1;
        पूर्णांक nth_n  = nth;

        पूर्णांक ith_es = ith % nth_es;
        पूर्णांक ith_n  = (ith / nth_es) % nth_n;

        जर (valid_ep_मोजणी_t % nth == 0 && tcm_buffer != nullptr && valid_ep_मोजणी_t == n_as &&
            valid_act_मोजणी_t == n_as && per_nb_cols_wsize <= tcm_buffer_size) {
            साठी (int64_t valid_id = ith; valid_id < valid_ep_मोजणी_t; valid_id += nth) {
                स्थिरांक int64_t cur_a = valid_matrix_row_counts[valid_id];

                स्वतः * src0_cur = (uint8_t *) src0->data + cur_a * expert_b_stride;

                mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, 0);
                स्थिरांक पूर्णांक        id          = row_mapping.i1;
                स्थिरांक int64_t    i11         = id % ne11;
                स्थिरांक int64_t    i12         = row_mapping.i2;
                स्थिरांक int64_t    i1          = id;
                स्थिरांक int64_t    i2          = i12;

                स्वतः *  src1_col = quant_a_buffer + (i11 * nbw1 + i12 * nbw2);
                दशांश * c_blk    = (दशांश *) ((अक्षर *) dst->data + (i1 * nb1 + i2 * nb2));

                uint8_t * a_row = src1_col;
                uint8_t * b_col = reinterpret_cast<uint8_t *>(tcm_buffer);
                जर ((nbw1 + per_nb_cols_wsize) <= tcm_buffer_size) {
                    a_row = (uint8_t *) tcm_buffer;
                    b_col = reinterpret_cast<uint8_t *>(tcm_buffer) + nbw1;
                }
                uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? b_col : nullptr;

                जर (ith % 2 == 0) {
                    spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(src0_cur), per_nb_cols_wsize);

                    जर (a_row != src1_col) {
                        spacemit_kernels::rvv::memcpy1d(a_row, src1_col, nbw1);
                    }
                }

                spine_barrier_wait(cur_barrier);

                जर (ith % 2 != 0) {
                    जर (a_row != src1_col) {
                        spacemit_kernels::rvv::memcpy1d(a_row, src1_col, nbw1);
                    }

                    spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(src0_cur), per_nb_cols_wsize);
                }

                int64_t nb_real = std::min(ne01, NB_COLS);
                साठी (int64_t ni = 0; ni < ne01; ni += NB_COLS) {
                    जर (ith % 2 != 0) {
                        spine_barrier_wait(cur_barrier);
                    }

                    gemm_kernel(b_blk_len, a_row, b_col, b_col_zp, c_blk + ni, 1, nb_real, b_k_blks, ne01);

                    जर (ith % 2 == 0) {
                        spine_barrier_wait(cur_barrier);
                    }

                    स्थिरांक int64_t next_ni = ni + NB_COLS;
                    जर (next_ni < ne01) {
                        nb_real = std::min(ne01 - next_ni, NB_COLS);
                        spacemit_kernels::rvv::memcpy1d(
                            b_col, reinterpret_cast<uint8_t *>(src0_cur) + next_ni * row_stride_b, per_nb_cols_wsize);
                    }
                }
            }
        } अन्यथा {
            साठी (int64_t valid_id = ith_es; valid_id < valid_ep_मोजणी_t; valid_id += nth_es) {
                स्थिरांक int64_t cur_a = valid_matrix_row_counts[valid_id];
                स्थिरांक int64_t cne1  = matrix_row_counts[cur_a];

                int64_t src1_cur_start = 0;
                int64_t src1_cur_end   = cne1;

                int64_t src0_cur_start = (ith_n * ne01) / nth_n;
                int64_t src0_cur_end   = MIN(((ith_n + 1) * ne01) / nth_n, ne01);

                जर (src1_cur_start >= src1_cur_end || src0_cur_start >= src0_cur_end) {
                    सुरूठेवा;
                }

                src0_cur_start =
                    (src0_cur_start % NB_COLS) ? src0_cur_start + NB_COLS - (src0_cur_start % NB_COLS) : src0_cur_start;
                src0_cur_end =
                    (src0_cur_end % NB_COLS) ? src0_cur_end + NB_COLS - (src0_cur_end % NB_COLS) : src0_cur_end;

                स्वतः *    src0_cur = (uint8_t *) src0->data + cur_a * expert_b_stride + src0_cur_start * row_stride_b;
                uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? src0_cur : nullptr;

                माप_प्रकार extra_tcm_buffer_size = tcm_buffer_size;
                शून्य * extra_tcm_buffer      = tcm_buffer;
                जर (tcm_buffer != nullptr && (src1_cur_end - src1_cur_start) >= 4 &&
                    (src0_cur_end - src0_cur_start) * row_stride_b <= tcm_buffer_size) {
                    spacemit_kernels::rvv::memcpy1d(tcm_buffer, src0_cur,
                                                    (src0_cur_end - src0_cur_start) * row_stride_b);
                    src0_cur = reinterpret_cast<uint8_t *>(tcm_buffer);
                    b_col_zp = block_type_has_zp<BLOC_TYPE>() ? src0_cur : nullptr;
                    extra_tcm_buffer_size -= (src0_cur_end - src0_cur_start) * row_stride_b;
                    extra_tcm_buffer = reinterpret_cast<शून्य *>(reinterpret_cast<uintptr_t>(tcm_buffer) +
                                                                (src0_cur_end - src0_cur_start) * row_stride_b);
                }

                पूर्णांक ir1 = src1_cur_start;

                जर (extra_tcm_buffer_size >= nbw1 && extra_tcm_buffer != nullptr) {
                    int64_t quant_a_tile_size = extra_tcm_buffer_size / nbw1;
                    करा {
                        quant_a_tile_size = MIN(quant_a_tile_size, src1_cur_end - ir1);

                        uint8_t * quant_a_tile_buffer = reinterpret_cast<uint8_t *>(extra_tcm_buffer);

                        पूर्णांक iir1 = ir1;
                        साठी (; iir1 < (ir1 + quant_a_tile_size); ++iir1) {
                            mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, iir1);

                            स्थिरांक पूर्णांक id = row_mapping.i1;  // selected expert index

                            स्थिरांक int64_t i11 = id % ne11;
                            स्थिरांक int64_t i12 = row_mapping.i2;  // row index in src1

                            स्वतः * src1_col = quant_a_buffer + (i11 * nbw1 + i12 * nbw2);
                            spacemit_kernels::rvv::memcpy1d(quant_a_tile_buffer, src1_col, nbw1);
                            quant_a_tile_buffer = quant_a_tile_buffer + nbw1;
                        }

                        quant_a_tile_buffer = reinterpret_cast<uint8_t *>(extra_tcm_buffer);
                        iir1                = ir1;

                        जर (moe_gemm_kernel_m2 != nullptr) {
                            साठी (; iir1 < (ir1 + quant_a_tile_size - 1); iir1 += 2, quant_a_tile_buffer += 2 * nbw1) {
                                mmid_row_mapping row_mapping_0 = MMID_MATRIX_ROW(cur_a, iir1);
                                mmid_row_mapping row_mapping_1 = MMID_MATRIX_ROW(cur_a, iir1 + 1);

                                src_workspaces[0] = quant_a_tile_buffer;
                                src_workspaces[1] = quant_a_tile_buffer + nbw1;

                                dst_workspaces[0] =
                                    (दशांश *) ((अक्षर *) dst->data + (row_mapping_0.i1 * nb1 + row_mapping_0.i2 * nb2)) +
                                    src0_cur_start;
                                dst_workspaces[1] = (दशांश *) ((अक्षर *) dst->data +
                                                               ((row_mapping_1.i1) * nb1 + (row_mapping_1.i2) * nb2)) +
                                                    src0_cur_start;
                                moe_gemm_kernel_m2(b_blk_len, src_workspaces.data(), src0_cur, b_col_zp,
                                                   dst_workspaces.data(), 1, src0_cur_end - src0_cur_start, b_k_blks,
                                                   ne01);
                            }
                        }

                        साठी (; iir1 < (ir1 + quant_a_tile_size); iir1++, quant_a_tile_buffer += nbw1) {
                            mmid_row_mapping row_mapping_0 = MMID_MATRIX_ROW(cur_a, iir1);

                            gemm_kernel(
                                b_blk_len, quant_a_tile_buffer, src0_cur, b_col_zp,
                                (दशांश *) ((अक्षर *) dst->data + (row_mapping_0.i1 * nb1 + row_mapping_0.i2 * nb2)) +
                                    src0_cur_start,
                                1, src0_cur_end - src0_cur_start, b_k_blks, ne01);
                        }

                        ir1 += quant_a_tile_size;
                    } पर्यंत (ir1 < src1_cur_end);
                } अन्यथा {
                    जर (moe_gemm_kernel_m2 != nullptr) {
                        साठी (; ir1 < src1_cur_end - 1; ir1 += 2) {
                            साठी (पूर्णांक iir1 = 0; iir1 < 2; ++iir1) {
                                mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, ir1 + iir1);

                                स्थिरांक पूर्णांक id = row_mapping.i1;  // selected expert index

                                स्थिरांक int64_t i11 = id % ne11;
                                स्थिरांक int64_t i12 = row_mapping.i2;  // row index in src1

                                स्थिरांक int64_t i1 = id;               // selected expert index
                                स्थिरांक int64_t i2 = i12;              // row

                                src_workspaces[iir1] = quant_a_buffer + (i11 * nbw1 + i12 * nbw2);

                                dst_workspaces[iir1] =
                                    (दशांश *) ((अक्षर *) dst->data + (i1 * nb1 + i2 * nb2)) + src0_cur_start;
                            }

                            moe_gemm_kernel_m2(b_blk_len, src_workspaces.data(), src0_cur, b_col_zp,
                                               dst_workspaces.data(), 1, src0_cur_end - src0_cur_start, b_k_blks, ne01);
                        }
                    }

                    साठी (; ir1 < src1_cur_end; ir1++) {
                        mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, ir1);

                        स्थिरांक पूर्णांक id = row_mapping.i1;  // selected expert index

                        स्थिरांक int64_t i11 = id % ne11;
                        स्थिरांक int64_t i12 = row_mapping.i2;  // row index in src1

                        स्थिरांक int64_t i1 = id;               // selected expert index
                        स्थिरांक int64_t i2 = i12;              // row

                        स्वतः * src1_col = quant_a_buffer + (i11 * nbw1 + i12 * nbw2);

                        gemm_kernel(b_blk_len, src1_col, src0_cur, b_col_zp,
                                    (दशांश *) ((अक्षर *) dst->data + (i1 * nb1 + i2 * nb2)) + src0_cur_start, 1,
                                    src0_cur_end - src0_cur_start, b_k_blks, ne01);
                    }
                }
            }
        }
#undef MMID_MATRIX_ROW
    }

    पूर्णांक repack(ggml_tensor * t, स्थिरांक शून्य * data, माप_प्रकार data_size) override {
        GGML_LOG_DEBUG("%s: repack tensor %s with %s_%dx%d\n", __func__, t->name, ggml_type_name(t->type),
                       (पूर्णांक) NB_COLS, (पूर्णांक) INTER_SIZE);
        परत ggml::cpu::riscv64_spacemit::repack<BLOC_TYPE, INTER_SIZE, NB_COLS>(t, data, data_size);
    }
};

वर्ग tensor_traits_common : सार्वजनिक tensor_traits_base {
    तार्किक work_size(पूर्णांक n_threads, स्थिरांक ggml_tensor * op, माप_प्रकार & size) override {
        निवड (op->op) {
            स्थिती GGML_OP_FLASH_ATTN_EXT:
                {
                    स्थिरांक पूर्णांक     n_tasks = n_threads;
                    स्थिरांक int64_t neq2    = op->src[0]->ne[2];  // number of query heads
                    स्थिरांक int64_t DK      = op->src[1]->ne[0];
                    स्थिरांक int64_t DV      = op->src[2]->ne[0];  // DV

                    // Tiled flash attention scratch (tile sizes defined in common.h)
                    // Per-thread: Q_q + KQ + mask + VKQ32 + V32 + K_f32 + padding
                    माप_प्रकार prefill = आकार(दशांश) *
                                     (GGML_FA_TILE_Q * DK + 2 * GGML_FA_TILE_Q * GGML_FA_TILE_KV + GGML_FA_TILE_Q * DV +
                                      GGML_FA_TILE_KV * DV + GGML_FA_TILE_KV * DK) *
                                     n_tasks;

                    // Decode path: n_kv_chunks = n_tasks (one chunk per thread)
                    // Per-thread: VKQ accmulator (DV), partial M, partial S + intra-thread scratch for V, Q and VKQ
                    माप_प्रकार n_chunks = n_tasks;
                    माप_प्रकार decode   = आकार(दशांश) * (neq2 * n_chunks * (2 + DV) + n_tasks * (DK + 2 * DV));

                    size = MAX(prefill, decode);
                }
                परत सत्य;
            पूर्वनिर्धारित:
                थांबा;
        }
        परत असत्य;
    }

    तार्किक compute_forward(ggml_compute_params * params, ggml_tensor * op) override {
        निवड (op->op) {
            स्थिती GGML_OP_NORM:
                निवड (op->src[0]->type) {
                    स्थिती GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_norm_f32(params, op);
                        परत सत्य;
                    पूर्वनिर्धारित:
                        GGML_ABORT("fatal error");
                }
            स्थिती GGML_OP_RMS_NORM:
                निवड (op->src[0]->type) {
                    स्थिती GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_rms_norm_f32(params, op);
                        परत सत्य;
                    पूर्वनिर्धारित:
                        GGML_ABORT("fatal error");
                }
            स्थिती GGML_OP_ADD:
                निवड (op->src[0]->type) {
                    स्थिती GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_द्विमान<GGML_OP_ADD, दशांश>(params, op);
                        परत सत्य;
                    स्थिती GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_द्विमान<GGML_OP_ADD, _Float16>(params, op);
                        परत सत्य;
                    पूर्वनिर्धारित:
                        ggml_compute_forward_add(params, op);
                        परत सत्य;
                }
            स्थिती GGML_OP_SUB:
                निवड (op->src[0]->type) {
                    स्थिती GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_द्विमान<GGML_OP_SUB, दशांश>(params, op);
                        परत सत्य;
                    स्थिती GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_द्विमान<GGML_OP_SUB, _Float16>(params, op);
                        परत सत्य;
                    पूर्वनिर्धारित:
                        ggml_compute_forward_sub(params, op);
                        परत सत्य;
                }
            स्थिती GGML_OP_MUL:
                निवड (op->src[0]->type) {
                    स्थिती GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_द्विमान<GGML_OP_MUL, दशांश>(params, op);
                        परत सत्य;
                    स्थिती GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_द्विमान<GGML_OP_MUL, _Float16>(params, op);
                        परत सत्य;
                    पूर्वनिर्धारित:
                        ggml_compute_forward_mul(params, op);
                        परत सत्य;
                }
            स्थिती GGML_OP_DIV:
                निवड (op->src[0]->type) {
                    स्थिती GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_द्विमान<GGML_OP_DIV, दशांश>(params, op);
                        परत सत्य;
                    स्थिती GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_द्विमान<GGML_OP_DIV, _Float16>(params, op);
                        परत सत्य;
                    पूर्वनिर्धारित:
                        ggml_compute_forward_div(params, op);
                        परत सत्य;
                }
            स्थिती GGML_OP_FLASH_ATTN_EXT:
                forward_flash_attn_ext_f16(params, op);
                परत सत्य;
            स्थिती GGML_OP_CONT:
                {
                    स्थिरांक ggml_tensor * src0 = op->src[0];
                    जर (op->type == src0->type && op->nb[0] != src0->nb[0] && op->nb[0] == src0->nb[1] &&
                        op->ne[3] * op->ne[2] * op->nb[2] == src0->ne[3] * src0->ne[2] * src0->nb[2]) {
                        spacemit_kernels::rvv::forward_cont_with_permute(params, op);
                    } अन्यथा {
                        ggml_compute_forward_cont(params, op);
                    }
                    परत सत्य;
                }
            स्थिती GGML_OP_CPY:
                {
                    स्थिरांक ggml_tensor * src0 = op->src[0];
                    जर (op->type == src0->type && op->nb[0] == src0->nb[1] && src0->nb[0] != src0->nb[1] &&
                        ggml_nelements(src0) == ggml_nelements(op)) {
                        spacemit_kernels::rvv::forward_cpy_with_permute(params, op);
                    } अन्यथा {
                        ggml_compute_forward_cpy(params, op);
                    }
                    परत सत्य;
                }
            स्थिती GGML_OP_REPEAT:
                {
                    स्थिरांक तार्किक rows_equal         = ggml_nrows(op->src[0]) == ggml_nrows(op);
                    स्थिरांक तार्किक broadcast_or_equal = op->src[0]->ne[0] == 1 || op->src[0]->ne[0] == op->ne[0];

                    जर (rows_equal && broadcast_or_equal) {
                        निवड (op->src[0]->type) {
                            स्थिती GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_repeat_nrows<int32_t>(params, op);
                                परत सत्य;
                            स्थिती GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_repeat_nrows<int16_t>(params, op);
                                परत सत्य;
                            पूर्वनिर्धारित:
                                थांबा;
                        }
                    }

                    जर (op->src[0]->ne[1] == 1 && op->src[0]->ne[0] == op->ne[0]) {
                        निवड (op->src[0]->type) {
                            स्थिती GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_repeat_dim1<int32_t>(params, op);
                                परत सत्य;
                            स्थिती GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_repeat_dim1<int16_t>(params, op);
                                परत सत्य;
                            पूर्वनिर्धारित:
                                थांबा;
                        }
                    }

                    ggml_compute_forward_repeat(params, op);
                }
                परत सत्य;
            स्थिती GGML_OP_SUM_ROWS:
                {
                    जर (op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32) {
                        spacemit_kernels::rvv::forward_sum_rows<दशांश>(params, op);
                    } अन्यथा {
                        ggml_compute_forward_sum_rows(params, op);
                    }
                }
                परत सत्य;
            स्थिती GGML_OP_GET_ROWS:
                {
                    जर (op->src[0]->type == op->type) {
                        निवड (op->src[0]->type) {
                            स्थिती GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_get_rows<int32_t>(params, op);
                                परत सत्य;
                            स्थिती GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_get_rows<int16_t>(params, op);
                                परत सत्य;
                            पूर्वनिर्धारित:
                                थांबा;
                        }
                    }

                    ggml_compute_forward_get_rows(params, op);
                }
                परत सत्य;
            स्थिती GGML_OP_CONCAT:
                {
                    स्थिरांक int32_t dim = ggml_get_op_params_i32(op, 0);
                    जर (dim == 0 && op->type == op->src[0]->type) {
                        निवड (op->src[0]->type) {
                            स्थिती GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_concat<int32_t>(params, op);
                                परत सत्य;
                            स्थिती GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_concat<int16_t>(params, op);
                                परत सत्य;
                            पूर्वनिर्धारित:
                                थांबा;
                        }
                    }

                    ggml_compute_forward_concat(params, op);
                }
                परत सत्य;
            // TODO For GGML_OP_GATED_DELTA_NET
            // case GGML_OP_GATED_DELTA_NET:
            //     return true;
            पूर्वनिर्धारित:
                थांबा;
        }
        परत असत्य;
    }

    शून्य forward_flash_attn_ext_f16(स्थिरांक ggml_compute_params * params, ggml_tensor * dst) {
        स्थिरांक ggml_tensor * q = dst->src[0];
        स्थिरांक ggml_tensor * k = dst->src[1];
        स्थिरांक ggml_tensor * v = dst->src[2];

        GGML_TENSOR_LOCALS(int64_t, neq, q, ne)
        GGML_TENSOR_LOCALS(माप_प्रकार, nbq, q, nb)
        GGML_TENSOR_LOCALS(int64_t, nek, k, ne)
        GGML_TENSOR_LOCALS(माप_प्रकार, nbk, k, nb)
        GGML_TENSOR_LOCALS(int64_t, nev, v, ne)
        GGML_TENSOR_LOCALS(माप_प्रकार, nbv, v, nb)
        GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
        GGML_TENSOR_LOCALS(माप_प्रकार, nb, dst, nb)

        स्थिरांक int64_t DK = nek0;
        स्थिरांक int64_t DV = nev0;

        स्थिरांक तार्किक supported_prec  = (dst->op_params[3] == GGML_PREC_F32 || dst->op_params[3] == GGML_PREC_DEFAULT);
        स्थिरांक तार्किक supported_types = (q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16);
        स्थिरांक तार्किक supported_shape = (DK > 0 && DK <= 128 && DV > 0 && DV <= 128);
        स्थिरांक तार्किक supported_vlen  = (__riscv_vlenb() == 128);

        जर (!(supported_prec && supported_types && supported_shape && supported_vlen)) {
            ggml_compute_forward_flash_attn_ext(params, dst);
            परत;
        }

        // total rows in q
        स्थिरांक int64_t nr = neq1 * neq2 * neq3;

        // rows per thread
        स्थिरांक पूर्णांक ith = params->ith;
        स्थिरांक पूर्णांक nth = params->nth;

        स्थिर constexpr int64_t Q_TILE_SZ = ggml_fa_tile_config::Q;
        स्थिरांक तार्किक               use_tiled = !params->use_ref && (neq1 >= Q_TILE_SZ);

        // 4x chunks per thread
        // int     nth_scaled = nth * 4;
        // int64_t chunk_size = (nr + nth_scaled - 1) / nth_scaled;
        // int64_t nchunk     = (nr + chunk_size - 1) / chunk_size;

        // if (nth == 1 || nchunk < nth) {
        //     nchunk = nth;
        // }

        int64_t nchunk = nth;

        जर (ith == 0) {
            // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
            ggml_threadpool_chunk_set(params->threadpool, nth);
        }

        ggml_barrier(params->threadpool);

        // The number of elements in each chunk
        स्थिरांक int64_t dr = (nr + nchunk - 1) / nchunk;

        // The first chunk comes from our thread_id, the rest will get auto-assigned.
        पूर्णांक current_chunk = ith;

        पर्यंत (current_chunk < nchunk) {
            स्थिरांक int64_t ir0 = dr * current_chunk;
            स्थिरांक int64_t ir1 = MIN(ir0 + dr, nr);

            जर (use_tiled) {
                spacemit_kernels::rvv::forward_flash_attn_ext_f16_tiled_vlen1024_vf16(
                    params, dst, ir0, ir1, ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer,
                    ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer_size);
            } अन्यथा {
                spacemit_kernels::rvv::forward_flash_attn_ext_f16_one_chunk_vlen1024_vf16(
                    params, dst, ir0, ir1, ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer,
                    ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer_size);
            }

            current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
        }
    }

    पूर्णांक repack(ggml_tensor * t, स्थिरांक शून्य * data, माप_प्रकार data_size) override {
        स_नकल(t->data, data, data_size);
        परत 0;
    }
};

// Impl By IME1
स्थिर स्थिरांक tensor_traits<block_q4_0, 32, 16>  q4_0_16x32_q8_0;
स्थिर स्थिरांक tensor_traits<block_q4_1, 32, 16>  q4_1_16x32_q8_0;
स्थिर स्थिरांक tensor_traits<block_q4_K, 32, 16>  q4_k_16x32_q8_0;
स्थिर स्थिरांक tensor_traits<block_q8_0, 32, 16>  q8_0_16x32_q8_0;
// Impl By IME2
स्थिर स्थिरांक tensor_traits<block_q2_K, 256, 32> q2_k_32x256_q8_0;
स्थिर स्थिरांक tensor_traits<block_q3_K, 256, 32> q3_k_32x256_q8_0;
स्थिर स्थिरांक tensor_traits<block_q4_0, 32, 32>  q4_0_32x32_q8_0;
स्थिर स्थिरांक tensor_traits<block_q4_1, 32, 32>  q4_1_32x32_q8_0;
स्थिर स्थिरांक tensor_traits<block_q4_0, 256, 32> q4_0_32x256_q8_0;
स्थिर स्थिरांक tensor_traits<block_q4_1, 256, 32> q4_1_32x256_q8_0;
स्थिर स्थिरांक tensor_traits<block_q4_K, 32, 32>  q4_k_32x32_q8_0;
स्थिर स्थिरांक tensor_traits<block_q6_K, 32, 32>  q6_k_32x32_q8_0;
स्थिर स्थिरांक tensor_traits<block_q8_0, 32, 32>  q8_0_32x32_q8_0;
स्थिर स्थिरांक tensor_traits<block_mxfp4, 32, 32> mxfp4_32x32_q8_0;
स्थिर स्थिरांक tensor_traits<block_q5_K, 32, 32>  q5_k_32x32_q8_0;
स्थिर स्थिरांक tensor_traits<block_q5_1, 32, 32>  q5_1_32x32_q8_0;
स्थिर स्थिरांक tensor_traits<block_q5_0, 32, 32>  q5_0_32x32_q8_0;
// Impl By RVV
स्थिर स्थिरांक tensor_traits_common               rvv_impl;

}  // namespace ggml::cpu::riscv64_spacemit

स्थिर स्थिरांक ggml::cpu::tensor_traits * ggml_riscv64_spacemit_get_optimal_repack_type(स्थिरांक ggml_tensor * cur) {
    निवड (cur->type) {
        स्थिती GGML_TYPE_Q2_K:
            {
#जर defined(RISCV64_SPACEMIT_IME2)
                जर (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    परत &ggml::cpu::riscv64_spacemit::q2_k_32x256_q8_0;
                }
#endif
            }
            थांबा;
        स्थिती GGML_TYPE_Q3_K:
            {
#जर defined(RISCV64_SPACEMIT_IME2)
                जर (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    परत &ggml::cpu::riscv64_spacemit::q3_k_32x256_q8_0;
                }
#endif
            }
            थांबा;
        स्थिती GGML_TYPE_Q4_0:
            {
#जर defined(RISCV64_SPACEMIT_IME2)
                जर (cur->ne[1] % 32 == 0 && cur->ne[0] % 256 == 0 &&
                    (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    परत &ggml::cpu::riscv64_spacemit::q4_0_32x256_q8_0;
                }

                जर (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    परत &ggml::cpu::riscv64_spacemit::q4_0_32x32_q8_0;
                }
#endif

#जर defined(RISCV64_SPACEMIT_IME1)
                जर (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    परत &ggml::cpu::riscv64_spacemit::q4_0_16x32_q8_0;
                }
#endif
            }
            थांबा;
        स्थिती GGML_TYPE_Q4_1:
            {
#जर defined(RISCV64_SPACEMIT_IME2)
                // TODO
                // if (cur->ne[1] % 32 == 0 && cur->ne[0] % 256 == 0 &&
                //     (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                //     return &ggml::cpu::riscv64_spacemit::q4_1_32x256_q8_0;
                // }

                जर (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    परत &ggml::cpu::riscv64_spacemit::q4_1_32x32_q8_0;
                }
#endif

#जर defined(RISCV64_SPACEMIT_IME1)
                जर (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    परत &ggml::cpu::riscv64_spacemit::q4_1_16x32_q8_0;
                }
#endif
            }
            थांबा;
        स्थिती GGML_TYPE_Q4_K:
            {
#जर defined(RISCV64_SPACEMIT_IME2)
                जर (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    परत &ggml::cpu::riscv64_spacemit::q4_k_32x32_q8_0;
                }
#endif

#जर defined(RISCV64_SPACEMIT_IME1)
                जर (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    परत &ggml::cpu::riscv64_spacemit::q4_k_16x32_q8_0;
                }
#endif
            }
            थांबा;
        स्थिती GGML_TYPE_Q6_K:
            {
#जर defined(RISCV64_SPACEMIT_IME2)
                जर ((ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    परत &ggml::cpu::riscv64_spacemit::q6_k_32x32_q8_0;
                }
#endif
            }
            थांबा;
        स्थिती GGML_TYPE_Q8_0:
            {
#जर defined(RISCV64_SPACEMIT_IME2)
                जर ((ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    परत &ggml::cpu::riscv64_spacemit::q8_0_32x32_q8_0;
                }
#endif

#जर defined(RISCV64_SPACEMIT_IME1)
                जर (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    परत &ggml::cpu::riscv64_spacemit::q8_0_16x32_q8_0;
                }
#endif
            }
            थांबा;
        स्थिती GGML_TYPE_MXFP4:
            {
#जर defined(RISCV64_SPACEMIT_IME2)
                // TODO
                // if (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                //     return &ggml::cpu::riscv64_spacemit::mxfp4_32x32_q8_0;
                // }
#endif
            }
            थांबा;
        स्थिती GGML_TYPE_Q5_K:
            {
#जर defined(RISCV64_SPACEMIT_IME2)
                जर (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    परत &ggml::cpu::riscv64_spacemit::q5_k_32x32_q8_0;
                }
#endif
            }
            थांबा;
        स्थिती GGML_TYPE_Q5_1:
            {
#जर defined(RISCV64_SPACEMIT_IME2)
                जर (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    परत &ggml::cpu::riscv64_spacemit::q5_1_32x32_q8_0;
                }
#endif
            }
            थांबा;
        स्थिती GGML_TYPE_Q5_0:
            {
#जर defined(RISCV64_SPACEMIT_IME2)
                जर (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    परत &ggml::cpu::riscv64_spacemit::q5_0_32x32_q8_0;
                }
#endif
            }
            थांबा;
        पूर्वनिर्धारित:
            थांबा;
    }

    परत nullptr;
}

स्थिर गणना ggml_status ggml_backend_riscv64_spacemit_buffer_init_tensor(ggml_backend_buffer_t buffer,
                                                                         ggml_tensor *         tensor) {
    tensor->extra =
        (शून्य *) const_cast<ggml::cpu::tensor_traits *>(ggml_riscv64_spacemit_get_optimal_repack_type(tensor));

    GGML_UNUSED(buffer);

    परत GGML_STATUS_SUCCESS;
}

स्थिर शून्य ggml_backend_riscv64_spacemit_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);

    शून्य * base = buffer->context;
    जर (base == nullptr) {
        परत;
    }

    ggml::cpu::riscv64_spacemit::spine_mem_pool_free(base);
}

स्थिर शून्य * ggml_backend_riscv64_spacemit_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);

    शून्य * base = buffer->context;
    GGML_ASSERT(base != nullptr);
    परत base;
}

स्थिर शून्य ggml_backend_riscv64_spacemit_buffer_memset_tensor(ggml_backend_buffer_t buffer,
                                                               ggml_tensor *         tensor,
                                                               uint8_t               value,
                                                               माप_प्रकार                offset,
                                                               माप_प्रकार                size) {
    GGML_ASSERT(tensor);
    स_रखो((अक्षर *) tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

स्थिर शून्य ggml_backend_riscv64_spacemit_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);

    शून्य * base = buffer->context;
    GGML_ASSERT(base != nullptr);
    स_रखो(base, value, buffer->size);
}

स्थिर शून्य ggml_backend_riscv64_spacemit_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                            ggml_tensor *         tensor,
                                                            स्थिरांक शून्य *          data,
                                                            माप_प्रकार                offset,
                                                            माप_प्रकार                size) {
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    स्वतः tensor_traits = (ggml::cpu::riscv64_spacemit::tensor_traits_base *) tensor->extra;
    जर (tensor_traits) {
        स्वतः OK = tensor_traits->repack(tensor, data, size);
        GGML_ASSERT(OK == 0);
    }

    GGML_UNUSED(buffer);
}

स्थिर स्थिरांक ggml_backend_buffer_i ggml_backend_riscv64_spacemit_buffer_i = {
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

स्थिर स्थिरांक अक्षर * ggml_backend_cpu_riscv64_spacemit_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    परत "CPU_RISCV64_SPACEMIT";

    GGML_UNUSED(buft);
}

स्थिर ggml_backend_buffer_t ggml_backend_cpu_riscv64_spacemit_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                                                                        माप_प्रकार size) {
    शून्य * base = ggml::cpu::riscv64_spacemit::spine_mem_pool_alloc(size, 64);
    जर (base == nullptr) {
        परत nullptr;
    }

    परत ggml_backend_buffer_init(buft, ggml_backend_riscv64_spacemit_buffer_i, base, size);
}

स्थिर माप_प्रकार ggml_backend_cpu_riscv64_spacemit_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    परत 64;

    GGML_UNUSED(buft);
}

स्थिर माप_प्रकार ggml_backend_cpu_riscv64_spacemit_nbytes(ggml_backend_buffer_type_t buft, स्थिरांक ggml_tensor * tensor) {
    साठी (पूर्णांक i = 0; i < GGML_MAX_DIMS; ++i) {
        जर (tensor->ne[i] <= 0) {
            परत 0;
        }
    }

    GGML_UNUSED(buft);

    स्थिरांक स्वतः plain_nbytes = [&]() {
        माप_प्रकार total = ggml_type_size(tensor->type);
        साठी (पूर्णांक i = 0; i < GGML_MAX_DIMS; ++i) {
            total += (tensor->ne[i] - 1) * tensor->nb[i];
        }
        परत total;
    };

    स्थिरांक माप_प्रकार blck_size = ggml_blck_size(tensor->type);
    जर (blck_size == 1) {
        परत plain_nbytes();
    }

    स्थिरांक माप_प्रकार row_nbytes = tensor->ne[0] * tensor->nb[0] / blck_size;

    स्थिरांक स्वतः add_strided_nbytes = [&](माप_प्रकार total, माप_प्रकार src_block_size, माप_प्रकार dst_block_size) {
        साठी (पूर्णांक i = 1; i < GGML_MAX_DIMS; ++i) {
            total += (tensor->ne[i] - 1) * (tensor->nb[i] / src_block_size) * dst_block_size;
        }
        परत total;
    };

    स्थिरांक स्वतः remap_block_nbytes = [&](माप_प्रकार src_block_size, माप_प्रकार dst_block_size, int64_t padded_rows = 0) {
        GGML_ASSERT(row_nbytes % src_block_size == 0);

        माप_प्रकार total =
            add_strided_nbytes((row_nbytes / src_block_size) * dst_block_size, src_block_size, dst_block_size);

        जर (padded_rows > 0 && tensor->ne[1] % padded_rows != 0) {
            total += (padded_rows - tensor->ne[1] % padded_rows) * (tensor->nb[1] / src_block_size) * dst_block_size;
        }

        परत total;
    };

    माप_प्रकार nbytes = row_nbytes;
    निवड (tensor->type) {
        स्थिती GGML_TYPE_Q4_K:
            nbytes = remap_block_nbytes(आकार(block_q4_K), आकार(block_q4_1) * 8);
            थांबा;
        स्थिती GGML_TYPE_Q6_K:
            nbytes = remap_block_nbytes(आकार(block_q6_K), आकार(block_q8_0) * 8, 32);
            थांबा;
        स्थिती GGML_TYPE_Q8_0:
            nbytes = remap_block_nbytes(आकार(block_q8_0), आकार(block_q8_0), 32);
            थांबा;
        स्थिती GGML_TYPE_Q2_K:
            nbytes = remap_block_nbytes(आकार(block_q2_K), आकार(spacemit_kernels::nrow_block_q2_k<1>));
            थांबा;
        स्थिती GGML_TYPE_Q3_K:
            nbytes = remap_block_nbytes(आकार(block_q3_K), आकार(spacemit_kernels::nrow_block_q3_k<1>));
            थांबा;
        स्थिती GGML_TYPE_MXFP4:
            nbytes = remap_block_nbytes(आकार(block_mxfp4), आकार(spacemit_kernels::nrow_block_mxfp4<1>));
            थांबा;
        स्थिती GGML_TYPE_Q5_K:
            nbytes = remap_block_nbytes(आकार(block_q5_K), आकार(spacemit_kernels::nrow_block_q5_1<1>) * 8);
            थांबा;
        स्थिती GGML_TYPE_Q5_1:
            nbytes = remap_block_nbytes(आकार(block_q5_1), आकार(spacemit_kernels::nrow_block_q5_1<1>));
            थांबा;
        स्थिती GGML_TYPE_Q5_0:
            nbytes = remap_block_nbytes(आकार(block_q5_0), आकार(spacemit_kernels::nrow_block_q5_0<1>));
            थांबा;
        पूर्वनिर्धारित:
            nbytes = add_strided_nbytes(row_nbytes, 1, 1);
            थांबा;
    }

    परत nbytes;
}

नामाकाश ggml::cpu::riscv64_spacemit {

वर्ग extra_buffer_type : ggml::cpu::extra_buffer_type {
    तार्किक supports_op(ggml_backend_dev_t, स्थिरांक ggml_tensor * op) override {
        निवड (op->op) {
            स्थिती GGML_OP_MUL_MAT:
                जर (op->src[0]->buffer && (ggml_n_dims(op->src[0]) == 2) &&
                    op->src[0]->buffer->buft == ggml_backend_cpu_riscv64_spacemit_buffer_type() &&
                    ggml_riscv64_spacemit_get_optimal_repack_type(op->src[0])) {
                    जर (op->src[1]->buffer && !ggml_backend_buft_is_host(op->src[1]->buffer->buft)) {
                        परत असत्य;
                    }
                    जर (op->src[1]->type == GGML_TYPE_F32) {
                        परत सत्य;
                    }
                }
                थांबा;
            स्थिती GGML_OP_MUL_MAT_ID:
                जर (op->src[0]->buffer && (ggml_n_dims(op->src[0]) == 3) &&
                    op->src[0]->buffer->buft == ggml_backend_cpu_riscv64_spacemit_buffer_type() &&
                    ggml_riscv64_spacemit_get_optimal_repack_type(op->src[0])) {
                    जर (op->src[1]->buffer && !ggml_backend_buft_is_host(op->src[1]->buffer->buft)) {
                        परत असत्य;
                    }
                    जर (op->src[1]->type == GGML_TYPE_F32) {
                        परत सत्य;
                    }
                }
                थांबा;
            पूर्वनिर्धारित:
                // GGML_ABORT("fatal error");
                थांबा;
        }
        परत असत्य;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(स्थिरांक ggml_tensor * op) override {
        निवड (op->op) {
            स्थिती GGML_OP_MUL_MAT:
            स्थिती GGML_OP_MUL_MAT_ID:
                जर (op->src[0]->buffer && op->src[0]->buffer->buft == ggml_backend_cpu_riscv64_spacemit_buffer_type()) {
                    परत (ggml::cpu::tensor_traits *) op->src[0]->extra;
                }
                थांबा;
            स्थिती GGML_OP_NORM:
            स्थिती GGML_OP_RMS_NORM:
            स्थिती GGML_OP_ADD:
            स्थिती GGML_OP_SUB:
            स्थिती GGML_OP_MUL:
            स्थिती GGML_OP_DIV:
            स्थिती GGML_OP_FLASH_ATTN_EXT:
            स्थिती GGML_OP_CONT:
            स्थिती GGML_OP_CPY:
            स्थिती GGML_OP_REPEAT:
            स्थिती GGML_OP_SUM_ROWS:
            स्थिती GGML_OP_GET_ROWS:
            स्थिती GGML_OP_CONCAT:
                // case GGML_OP_GATED_DELTA_NET:
                परत (ggml::cpu::tensor_traits *) (&ggml::cpu::riscv64_spacemit::rvv_impl);
            पूर्वनिर्धारित:
                // GGML_ABORT("fatal error");
                थांबा;
        }

        परत nullptr;
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
        नवीन ggml::cpu::riscv64_spacemit::extra_buffer_type(),
    };

    परत &ggml_backend_cpu_buffer_type_riscv64_spacemit;
}

बाह्य "C" {
स्थिर पूर्णांक bind_ai_thread() {
    पूर्णांक  fd, bytes;
    अक्षर str[32];

    fd = खोलो("/proc/set_ai_thread", O_WRONLY);
    जर (fd < 0) {
        GGML_LOG_ERROR("try open /proc/set_ai_thread failed\n");
        परत -1;
    }

    snprintf(str, 16, "%d", 0);
    bytes = लिख(fd, str, म_माप(str));
    जर (bytes < 0) {
        GGML_LOG_ERROR("try write /proc/set_ai_thread failed\n");
        बंद(fd);
        परत -1;
    }

    बंद(fd);
    परत 0;
}

शून्य ggml_backend_cpu_riscv64_spacemit_set_numa_thread_affinity(पूर्णांक thread_n) {
    पूर्णांक cpu_id = sched_getcpu();
    जर (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2 &&
        !((1 << cpu_id) & ggml::cpu::riscv64_spacemit::global_spine_env_info.cpu_mask)) {
        GGML_PRINT_DEBUG("bind_ai_thread for thread %d, pid %d\n", thread_n, getpid());
        bind_ai_thread();
    }

    जर (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_tcm &&
        ggml::cpu::riscv64_spacemit::tls_context.cpu_id == -1) {
        CPU_ZERO(&(ggml::cpu::riscv64_spacemit::tls_context.cpuset));
        pthread_t    main_thread     = pthread_self();
        स्थिरांक स्वतः & perfer_core_ids = ggml::cpu::riscv64_spacemit::global_spine_env_info.perfer_core_ids;
        जर (thread_n < 0 || static_cast<माप_प्रकार>(thread_n) >= perfer_core_ids.size()) {
            GGML_ABORT("thread_n %d exceeds perfer_core_ids size %zu\n", thread_n, perfer_core_ids.size());
        }
        स्वतः perfer_cpu_id = perfer_core_ids[static_cast<माप_प्रकार>(thread_n)];
        CPU_SET(perfer_cpu_id, &(ggml::cpu::riscv64_spacemit::tls_context.cpuset));
        पूर्णांक s =
            pthread_setaffinity_np(main_thread, आकार(cpu_set_t), &(ggml::cpu::riscv64_spacemit::tls_context.cpuset));
        जर (s != 0) {
            GGML_ABORT("set thread affinity error for thread_n %d, cpu_id %d\n", thread_n, perfer_cpu_id);
        }

        पूर्णांक ai_cpu_id = perfer_cpu_id - ggml::cpu::riscv64_spacemit::global_spine_env_info.aicpu_id_offset;
        ggml::cpu::riscv64_spacemit::tls_context.cpu_id = ai_cpu_id;
        ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer =
            ggml::cpu::riscv64_spacemit::spine_mem_pool_tcm_mem_get(ai_cpu_id);
        ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer_size =
            ggml::cpu::riscv64_spacemit::global_spine_env_info.tcm_blk_size;
    }

    जर (ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer != nullptr) {
        शून्य * rt =
            ggml::cpu::riscv64_spacemit::spine_mem_pool_tcm_mem_wait(ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        जर (rt == nullptr) {
            GGML_ABORT("wait tcm buffer failed for cpu_id: %d", ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        }
    }
}

शून्य ggml_backend_cpu_riscv64_spacemit_clear_numa_thread_affinity_threaded(पूर्णांक thread_n) {
    जर (ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer != nullptr) {
        स्वतः rt = ggml::cpu::riscv64_spacemit::spine_mem_pool_tcm_mem_release(
            ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        जर (rt != 0) {
            GGML_ABORT("release tcm buffer failed for cpu_id: %d", ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        }
    }
}
}
