#घोषणा GGML_COMMON_IMPL_CPP
#घोषणा GGML_COMMON_DECL_CPP

#समावेश "ime.h"

#समावेश "binary-ops.h"
#समावेश "common.h"
#समावेश "ggml-backend-impl.h"
#समावेश "ggml-common.h"
#समावेश "ggml-cpu.h"
#समावेश "ime_env.h"
#समावेश "ime_kernels.h"
#समावेश "ops.h"
#समावेश "repack.h"
#समावेश "rvv_kernels.h"
#समावेश "spine_mem_pool.h"
#समावेश "traits.h"
#समावेश "vec.h"

#समावेश <fcntl.h>
#समावेश <sys/mman.h>
#समावेश <unistd.h>

#समावेश <algorithm>
#समावेश <atomic>
#समावेश <cassert>
#समावेश <cerrno>
#समावेश <cmath>
#समावेश <cstdio>  // for GGML_ASSERT
#समावेश <stdexcept>
#समावेश <thread>
// clang-format off
#अगर defined(__riscv)

#अगर !defined(__riscv_v) || !defined(__riscv_v_intrinsic)
#त्रुटि "riscv v extension or v_intrinsic not enabled"
#अन्यथा
#समावेश <riscv_vector.h>
#पूर्ण_अगर

#अगर !defined(__riscv_zfh) || !defined(__riscv_zvfh)
#त्रुटि "riscv zfh extension not enabled, GGML_RV_ZFH and GGML_RV_ZVFH must be defined to 1"
#पूर्ण_अगर

#अगर !defined(__riscv_zba)
#त्रुटि "riscv zba extension not enabled, GGML_RV_ZBA must be defined to 1"
#पूर्ण_अगर

#अगर defined(RISCV64_SPACEMIT_IME1) || defined(RISCV64_SPACEMIT_IME2)
#अन्यथा
#त्रुटि "RISCV64_SPACEMIT_IME1 or RISCV64_SPACEMIT_IME2 not defined"
#पूर्ण_अगर

#अन्यथा

#त्रुटि "riscv not enabled in this build"

#पूर्ण_अगर

#अगर defined(__GNUC__)
#आशय GCC diagnostic ignored "-Woverlength-strings"
#आशय GCC diagnostic ignored "-Wcast-qual"
#आशय GCC diagnostic ignored "-Wunused-parameter"
#पूर्ण_अगर

// clang-format on

बाह्य "C" {
बाह्य व्योम ggml_threadpool_chunk_set(काष्ठा ggml_threadpool * tp, पूर्णांक value);
बाह्य पूर्णांक  ggml_threadpool_chunk_add(काष्ठा ggml_threadpool * tp, पूर्णांक value);
}

namespace ggml::cpu::riscv64_spacemit {

काष्ठा TLSContext {
    पूर्णांक       cpu_id{ -1 };
    cpu_set_t cpuset;
    व्योम *    tcm_बफर{ nullptr };
    माप_प्रकार    tcm_बफर_आकार{ 0 };
};

thread_local TLSContext tls_context;

ढाँचा <typename BLOC_TYPE, int64_t INTER_आकार, int64_t NB_COLS> constexpr माप_प्रकार get_repacked_block_type_आकार() {
    अगर constexpr (std::is_same_v<BLOC_TYPE, block_q6_K> || std::is_same_v<BLOC_TYPE, block_q8_0>) {
        वापस माप(block_q8_0);
    } अन्यथा अगर constexpr (std::is_same_v<BLOC_TYPE, block_q4_0>) {
        वापस माप(block_q4_0) * INTER_आकार / QK4_0;
    } अन्यथा अगर constexpr (std::is_same_v<BLOC_TYPE, block_q4_1> || std::is_same_v<BLOC_TYPE, block_q4_K>) {
        वापस (माप(block_q4_0) + माप(uint8_t)) * INTER_आकार / QK4_1;
    } अन्यथा अगर constexpr (std::is_same_v<BLOC_TYPE, block_q2_K>) {
        वापस माप(spacemit_kernels::nrow_block_q2_k<1>);
    } अन्यथा अगर constexpr (std::is_same_v<BLOC_TYPE, block_q3_K>) {
        वापस माप(spacemit_kernels::nrow_block_q3_k<1>);
    } अन्यथा अगर constexpr (std::is_same_v<BLOC_TYPE, block_mxfp4>) {
        वापस माप(spacemit_kernels::nrow_block_mxfp4<1>);
    } अन्यथा अगर constexpr (std::is_same_v<BLOC_TYPE, block_q5_1> || std::is_same_v<BLOC_TYPE, block_q5_K>) {
        वापस माप(spacemit_kernels::nrow_block_q5_1<1>);
    } अन्यथा अगर constexpr (std::is_same_v<BLOC_TYPE, block_q5_0>) {
        वापस माप(spacemit_kernels::nrow_block_q5_0<1>);
    } अन्यथा {
        निश्चित(false);
        वापस 0;
    }
}

ढाँचा <typename BLOC_TYPE> constexpr bool block_type_has_zp() {
    अगर constexpr (std::is_same_v<BLOC_TYPE, block_q6_K> || std::is_same_v<BLOC_TYPE, block_q8_0> ||
                  std::is_same_v<BLOC_TYPE, block_q3_K> || std::is_same_v<BLOC_TYPE, block_q4_0> ||
                  std::is_same_v<BLOC_TYPE, block_mxfp4> || std::is_same_v<BLOC_TYPE, block_q5_0>) {
        वापस false;
    } अन्यथा अगर constexpr (std::is_same_v<BLOC_TYPE, block_q4_1> || std::is_same_v<BLOC_TYPE, block_q4_K> ||
                         std::is_same_v<BLOC_TYPE, block_q2_K> || std::is_same_v<BLOC_TYPE, block_q5_1> ||
                         std::is_same_v<BLOC_TYPE, block_q5_K>) {
        वापस true;
    } अन्यथा {
        निश्चित(false);
        वापस false;
    }
}

श्रेणी tensor_traits_base : खुला ggml::cpu::tensor_traits {
  खुला:
    भव पूर्णांक repack(ggml_tensor * t, स्थिर व्योम * तथ्य, माप_प्रकार तथ्य_आकार) = 0;
};

ढाँचा <typename BLOC_TYPE, int64_t INTER_आकार, int64_t NB_COLS> श्रेणी tensor_traits : खुला tensor_traits_base {
    bool work_आकार(पूर्णांक /* n_threads */, स्थिर ggml_tensor * op, माप_प्रकार & size) override {
        चयन (op->op) {
            हाल GGML_OP_MUL_MAT:
                {
                    int64_t src1_nelements = ggml_nelements(op->src[1]);

                    अगर constexpr (std::is_same_v<BLOC_TYPE, block_q2_K> || std::is_same_v<BLOC_TYPE, block_q3_K>) {
                        size =
                            spacemit_kernels::div_round_up(src1_nelements, QK_K) * spacemit_kernels::q8k_blk_आकार(QK_K);
                    } अन्यथा अगर constexpr (INTER_आकार == QK4_0) {
                        size = spacemit_kernels::div_round_up(src1_nelements, QK4_0) *
                               spacemit_kernels::q8_blk_आकार(QK4_0, true);
                    } अन्यथा अगर constexpr (INTER_आकार == 256) {
                        size = spacemit_kernels::div_round_up(src1_nelements, 256) *
                               spacemit_kernels::q8_hp_blk_आकार(256, true, true);
                    } अन्यथा {
                        GGML_ABORT("unsupported block type");
                    }

                    size = GGML_PAD(size, माप(int64_t));

                    वापस true;
                }
            हाल GGML_OP_MUL_MAT_ID:
                {
                    int64_t src1_nelements = ggml_nelements(op->src[1]);

                    अगर constexpr (std::is_same_v<BLOC_TYPE, block_q2_K> || std::is_same_v<BLOC_TYPE, block_q3_K>) {
                        size =
                            spacemit_kernels::div_round_up(src1_nelements, QK_K) * spacemit_kernels::q8k_blk_आकार(QK_K);
                    } अन्यथा अगर constexpr (INTER_आकार == QK4_0) {
                        size = spacemit_kernels::div_round_up(src1_nelements, QK4_0) *
                               spacemit_kernels::q8_blk_आकार(QK4_0, true);
                    } अन्यथा अगर constexpr (INTER_आकार == 256) {
                        size = spacemit_kernels::div_round_up(src1_nelements, 256) *
                               spacemit_kernels::q8_hp_blk_आकार(256, true, true);
                    } अन्यथा {
                        GGML_ABORT("unsupported block type");
                    }

                    size = GGML_PAD(size, माप(int64_t));

                    स्थिर int64_t ne02 = op->src[0]->ne[2];  // n_as, n_expert
                    स्थिर int64_t ne12 = op->src[1]->ne[2];  // n_tokens

                    स्थिर माप_प्रकार sizeof_mmid_row_mapping = माप(int64_t);
                    size += sizeof_mmid_row_mapping * ne02 * (ne12 + 1) + (ne02 + 1) * माप(int64_t);

                    size = GGML_PAD(size, माप(int64_t));

                    वापस true;
                }
            शेष:
                // GGML_ABORT("fatal error");
                अवरोध;
        }
        वापस false;
    }

    bool compute_forward(ggml_compute_params * params, ggml_tensor * op) override {
        चयन (op->op) {
            हाल GGML_OP_MUL_MAT:
                चयन (op->src[0]->type) {
                    हाल GGML_TYPE_Q2_K:
                    हाल GGML_TYPE_Q3_K:
                    हाल GGML_TYPE_Q4_0:
                    हाल GGML_TYPE_Q4_1:
                    हाल GGML_TYPE_Q4_K:
                    हाल GGML_TYPE_Q6_K:
                    हाल GGML_TYPE_Q8_0:
                    हाल GGML_TYPE_Q5_0:
                    हाल GGML_TYPE_Q5_1:
                    हाल GGML_TYPE_Q5_K:
                        //case GGML_TYPE_MXFP4:
                        forward_mul_mat(params, op);
                        वापस true;
                    शेष:
                        // GGML_ABORT("fatal error: unsupported type for src0 in MUL_MAT");
                        वापस false;
                }
                अवरोध;
            हाल GGML_OP_MUL_MAT_ID:
                चयन (op->src[0]->type) {
                    हाल GGML_TYPE_Q2_K:
                    हाल GGML_TYPE_Q3_K:
                    हाल GGML_TYPE_Q4_0:
                    हाल GGML_TYPE_Q4_1:
                    हाल GGML_TYPE_Q4_K:
                    हाल GGML_TYPE_Q6_K:
                    हाल GGML_TYPE_Q8_0:
                    हाल GGML_TYPE_Q5_0:
                    हाल GGML_TYPE_Q5_1:
                    हाल GGML_TYPE_Q5_K:
                        //case GGML_TYPE_MXFP4:
                        forward_mul_mat_id(params, op);
                        वापस true;
                    शेष:
                        // GGML_ABORT("fatal error: unsupported type for src0 in MUL_MAT_ID");
                        वापस false;
                }
                अवरोध;
            शेष:
                // GGML_ABORT("fatal error");
                अवरोध;
        }
        वापस false;
    }

    व्योम forward_mul_mat(ggml_compute_params * params, ggml_tensor * op) {
        constexpr माप_प्रकार a_blk_len = INTER_आकार;
        constexpr माप_प्रकार b_blk_len = INTER_आकार;

        स्थिर ggml_tensor * src0 = op->src[0];
        स्थिर ggml_tensor * src1 = op->src[1];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_द्विआधारी_OP_LOCALS

        पूर्णांक ith = params->ith;
        पूर्णांक nth = params->nth;

        [[maybe_unused]] स्थिर क्रमागत ggml_type type = src0->type;

        व्योम *        w_तथ्य  = (व्योम *) src0->तथ्य;
        स्थिर भग्न * feature = (स्थिर भग्न *) src1->तथ्य;
        भग्न *       output  = (भग्न *) dst->तथ्य;

        स्थिर int64_t gemm_m = ne11 * ne12 * ne13;
        स्थिर int64_t gemm_k = ne10;
        स्थिर int64_t gemm_n = ne01;

        spacemit_kernels::quantize_a_row_def       quantize_a_row_i8;
        spacemit_kernels::quantize_a_row_def       quantize_a_4row_i8;
        spacemit_kernels::gemm_kernel_quantize_def gemm_kernel;
        bool                                       set_kernel_impl = false;

        int64_t block_stride_a = spacemit_kernels::q8_blk_आकार(a_blk_len);

#अगर defined(RISCV64_SPACEMIT_IME2)
        अगर (!set_kernel_impl && (global_spine_env_info.use_ime2)) {
            quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8;
            quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8;
            block_stride_a     = spacemit_kernels::q8_blk_आकार(a_blk_len, true);

            अगर constexpr (std::is_same_v<BLOC_TYPE, block_q6_K> || std::is_same_v<BLOC_TYPE, block_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i8;
                set_kernel_impl = true;
            } अन्यथा अगर constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> || std::is_same_v<BLOC_TYPE, block_q4_1> ||
                                 std::is_same_v<BLOC_TYPE, block_q4_K>) {
                अगर constexpr (INTER_आकार == 256) {
                    gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i4_hp;
                    quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8_hp;
                    quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8_hp;
                    block_stride_a     = spacemit_kernels::q8_hp_blk_आकार(a_blk_len, true, true);
                    set_kernel_impl    = true;
                } अन्यथा {
                    gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i4;
                    quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8;
                    quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8;
                    block_stride_a     = spacemit_kernels::q8_blk_आकार(a_blk_len, true);
                    set_kernel_impl    = true;
                }
            } अन्यथा अगर constexpr (std::is_same_v<BLOC_TYPE, block_q2_K>) {
                quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8k;
                quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8k;
                block_stride_a     = spacemit_kernels::q8k_blk_आकार(a_blk_len);

                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i2k;
                set_kernel_impl = true;
            } अन्यथा अगर constexpr (std::is_same_v<BLOC_TYPE, block_q3_K>) {
                quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8k;
                quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8k;
                block_stride_a     = spacemit_kernels::q8k_blk_आकार(a_blk_len);

                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i3k;
                set_kernel_impl = true;
            } अन्यथा अगर constexpr (std::is_same_v<BLOC_TYPE, block_mxfp4>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8mxfp4;
                set_kernel_impl = true;
            } अन्यथा अगर constexpr (std::is_same_v<BLOC_TYPE, block_q5_1> || std::is_same_v<BLOC_TYPE, block_q5_K> ||
                                 std::is_same_v<BLOC_TYPE, block_q5_0>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i5;
                set_kernel_impl = true;
            }
        }
#पूर्ण_अगर

#अगर defined(RISCV64_SPACEMIT_IME1)
        अगर (!set_kernel_impl && (global_spine_env_info.use_ime1)) {
            quantize_a_row_i8  = spacemit_kernels::ime1::quantize_a_row_i8;
            quantize_a_4row_i8 = spacemit_kernels::ime1::quantize_a_4row_i8;

            अगर constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> || std::is_same_v<BLOC_TYPE, block_q4_1> ||
                          std::is_same_v<BLOC_TYPE, block_q4_K>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i4;
                set_kernel_impl = true;
            } अन्यथा अगर constexpr (std::is_same_v<BLOC_TYPE, block_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i8;
                set_kernel_impl = true;
            }
        }
#पूर्ण_अगर
        अगर (!set_kernel_impl) {
            GGML_ABORT("no kernel implementation found for the block type");
        }

        स्थिर int64_t a_k_blks = spacemit_kernels::div_round_up(gemm_k, a_blk_len);
        स्थिर int64_t b_k_blks = spacemit_kernels::div_round_up(gemm_k, b_blk_len);

        स्थिर int64_t row_stride_a        = a_k_blks * block_stride_a;
        स्थिर int64_t gemm_workspace_आकार = GGML_PAD(gemm_m * row_stride_a, alignof(int64_t));

        अगर (ith == 0 && params->wsize < gemm_workspace_आकार) {
            GGML_ABORT("wsize less than gemm_workspace_size");
        }

        uintptr_t ws_ptr = reinterpret_cast<uintptr_t>(params->wdata);

        व्योम *        tcm_बफर      = ggml::cpu::riscv64_spacemit::tls_context.tcm_बफर;
        स्थिर int64_t tcm_बफर_आकार = ggml::cpu::riscv64_spacemit::tls_context.tcm_बफर_आकार;

        स्वतः * quant_a_बफर = reinterpret_cast<uint8_t *>(ws_ptr);

        constexpr int64_t row_align = 4;
        स्थिर int64_t     row_blks  = spacemit_kernels::div_round_up(gemm_m, row_align);

        स्थिर int64_t row_stride_b      = b_k_blks * get_repacked_block_type_आकार<BLOC_TYPE, INTER_आकार, NB_COLS>();
        स्थिर int64_t per_mb_rows_wsize = row_align * row_stride_a;
        स्थिर int64_t per_nb_cols_wsize = NB_COLS * row_stride_b;

        स्थिर int64_t barrier_idx = static_cast<int64_t>(ith / 2);

        GGML_ASSERT(global_spine_env_info.init_barrier != nullptr);
        GGML_ASSERT(barrier_idx < spine_init_barrier_गिनती);
        spine_barrier_t * cur_barrier = &global_spine_env_info.init_barrier[barrier_idx];

        अगर (gemm_m == 1) {
            पूर्णांक task_per_thread = spacemit_kernels::div_round_up(a_k_blks, nth);
            पूर्णांक a_blk_start     = ith * task_per_thread;
            पूर्णांक a_blk_end       = std::min(a_blk_start + task_per_thread, (पूर्णांक) a_k_blks);
            अगर (a_blk_start < a_blk_end) {
                quantize_a_row_i8(a_blk_len, feature + a_blk_start * a_blk_len, (a_blk_end - a_blk_start) * a_blk_len,
                                  quant_a_बफर + a_blk_start * block_stride_a);
            }
        } अन्यथा {
            पूर्णांक task_per_thread = spacemit_kernels::div_round_up(row_blks, nth);
            पूर्णांक m_row_blk_start = ith * task_per_thread;
            पूर्णांक m_row_blk_end   = std::min(m_row_blk_start + task_per_thread, (पूर्णांक) row_blks);
            क्रम (पूर्णांक m_row_blk = m_row_blk_start; m_row_blk < m_row_blk_end; m_row_blk++) {
                पूर्णांक m_idx             = m_row_blk * row_align;
                पूर्णांक rows_tobe_handled = (gemm_m - m_idx) > row_align ? row_align : (gemm_m - m_idx);

                अगर (rows_tobe_handled == row_align && quantize_a_4row_i8 != nullptr) {
                    स्थिर भग्न * a_row_ptr       = feature + m_idx * gemm_k;
                    स्वतः *        quant_a_row_ptr = quant_a_बफर + m_idx * row_stride_a;
                    quantize_a_4row_i8(a_blk_len, a_row_ptr, gemm_k, quant_a_row_ptr);
                } अन्यथा {
                    जबतक (rows_tobe_handled) {
                        स्थिर भग्न * a_row_ptr       = feature + m_idx * gemm_k;
                        स्वतः *        quant_a_row_ptr = quant_a_बफर + m_idx * row_stride_a;
                        quantize_a_row_i8(a_blk_len, a_row_ptr, gemm_k, quant_a_row_ptr);
                        rows_tobe_handled -= 1;
                        m_idx += 1;
                    }
                }
            }
        }

        ggml_barrier(params->threadpool);

        स्थिर int64_t gemm_m_stride     = gemm_n / gemm_m > 64 ? gemm_m : 16;
        स्थिर int64_t gemm_m_blocked    = spacemit_kernels::div_round_up(gemm_m, gemm_m_stride);
        स्थिर int64_t max_gemm_n_stride = spacemit_kernels::div_round_up(gemm_n * gemm_m_blocked, nth);

        int64_t gemm_n_stride = gemm_n;
        अगर (max_gemm_n_stride < gemm_n) {
            gemm_n_stride =
                std::min(gemm_n_stride, spacemit_kernels::div_round_up(max_gemm_n_stride, NB_COLS) * NB_COLS);
        }

        अगर (gemm_n_stride == gemm_n && tcm_बफर != nullptr && per_mb_rows_wsize <= tcm_बफर_आकार) {
            क्रम (int64_t m_start = ith * row_align; m_start < gemm_m; m_start += row_align * nth) {
                uint8_t * b_col    = reinterpret_cast<uint8_t *>(w_तथ्य);
                uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? b_col : nullptr;

                int64_t m_row_real = std::min(gemm_m - m_start, row_align);

                spacemit_kernels::rvv::memcpy1d(tcm_बफर, quant_a_बफर + m_start * row_stride_a,
                                                m_row_real * row_stride_a);

                int64_t n_blk_real = 0;
                क्रम (int64_t ni = 0; ni < gemm_n; ni += n_blk_real, b_col += n_blk_real * row_stride_b) {
                    n_blk_real = std::min(gemm_n - ni, (int64_t) NB_COLS);

                    uint8_t * a_row_ptr = (uint8_t *) tcm_बफर;
                    भग्न *   c_blk     = output + m_start * gemm_n + ni;

                    int32_t rows_remaining = m_row_real;

                    जबतक (rows_remaining > 0) {
                        स्वतः rows_handled = gemm_kernel(b_blk_len, a_row_ptr, b_col, b_col_zp, c_blk, rows_remaining,
                                                        n_blk_real, b_k_blks, gemm_n);

                        c_blk += rows_handled * gemm_n;
                        a_row_ptr += rows_handled * row_stride_a;

                        rows_remaining -= rows_handled;
                    }
                }
            }
        } अन्यथा अगर (tcm_बफर != nullptr && per_nb_cols_wsize <= tcm_बफर_आकार) {
            uint8_t * a_row = quant_a_बफर;
            uint8_t * b_col = reinterpret_cast<uint8_t *>(tcm_बफर);
            अगर ((gemm_workspace_आकार + per_nb_cols_wsize) <= tcm_बफर_आकार) {
                a_row = (uint8_t *) tcm_बफर;
                b_col = reinterpret_cast<uint8_t *>(tcm_बफर) + gemm_workspace_आकार;
            }
            uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? b_col : nullptr;

            int64_t ni      = ith * NB_COLS;
            int64_t nb_real = std::min(gemm_n - ni, NB_COLS);

            अगर (ith % 2 == 0 && nb_real > 0) {
                spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(w_तथ्य) + ni * row_stride_b,
                                                nb_real * row_stride_b);
                अगर (a_row != quant_a_बफर) {
                    spacemit_kernels::rvv::memcpy1d(a_row, quant_a_बफर, gemm_workspace_आकार);
                }
            }

            spine_barrier_wait(cur_barrier);

            अगर (ith % 2 != 0 && nb_real > 0) {
                अगर (a_row != quant_a_बफर) {
                    spacemit_kernels::rvv::memcpy1d(a_row, quant_a_बफर, gemm_workspace_आकार);
                }
                spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(w_तथ्य) + ni * row_stride_b,
                                                nb_real * row_stride_b);
            }

            क्रम (; ni < gemm_n; ni += NB_COLS * nth) {
                int64_t rows_remaining = gemm_m;
                भग्न * c_blk          = output + ni;
                स्वतः *  a_row_cur      = a_row;

                अगर (ith % 2 != 0) {
                    spine_barrier_wait(cur_barrier);
                }

                जबतक (rows_remaining > 0) {
                    स्वतः rows_handled = gemm_kernel(b_blk_len, a_row_cur, b_col, b_col_zp, c_blk, rows_remaining,
                                                    nb_real, b_k_blks, gemm_n);

                    c_blk += rows_handled * gemm_n;
                    a_row_cur += rows_handled * row_stride_a;

                    rows_remaining -= rows_handled;
                }

                अगर (ith % 2 == 0) {
                    spine_barrier_wait(cur_barrier);
                }

                स्थिर int64_t अग्र_ni = ni + NB_COLS * nth;
                अगर (अग्र_ni < gemm_n) {
                    nb_real = std::min(gemm_n - अग्र_ni, NB_COLS);
                    spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(w_तथ्य) + अग्र_ni * row_stride_b,
                                                    nb_real * row_stride_b);
                }
            }
        } अन्यथा {
            स्थिर int64_t task_गिनती_m = spacemit_kernels::div_round_up(gemm_m, gemm_m_stride);
            स्थिर int64_t task_गिनती_n = spacemit_kernels::div_round_up(gemm_n, gemm_n_stride);

            int64_t task_गिनती      = task_गिनती_m * task_गिनती_n;
            int64_t task_per_thread = (task_गिनती + nth - 1) / nth;
            int64_t start           = ith * task_per_thread;
            int64_t end             = std::min((ith + 1) * task_per_thread, task_गिनती);
            क्रम (int64_t compute_idx = start; compute_idx < end; compute_idx++) {
                स्थिर स्वतः tid_n = compute_idx / task_गिनती_m;
                स्थिर स्वतः tid_m = compute_idx % task_गिनती_m;

                स्थिर int64_t m_start = tid_m * gemm_m_stride;
                स्थिर int64_t m_गिनती = std::min(gemm_m - m_start, (int64_t) gemm_m_stride);

                स्थिर int64_t n_start = tid_n * gemm_n_stride;
                स्थिर int64_t n_गिनती = std::min(gemm_n - n_start, (int64_t) gemm_n_stride);

                स्थिर int64_t n_blk = m_गिनती == 1 ? n_गिनती : NB_COLS;

                uint8_t * b_col    = reinterpret_cast<uint8_t *>(w_तथ्य) + n_start * row_stride_b;
                uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? b_col : nullptr;

                int64_t n_blk_real = 0;
                क्रम (int64_t ni = 0; ni < n_गिनती; ni += n_blk_real, b_col += n_blk_real * row_stride_b) {
                    n_blk_real = std::min(n_गिनती - ni, n_blk);

                    uint8_t * a_row = quant_a_बफर + m_start * row_stride_a;

                    भग्न * c_blk = output + m_start * gemm_n + n_start + ni;

                    int64_t rows_remaining = m_गिनती;

                    uint8_t * b_col_cur    = b_col;
                    uint8_t * b_col_zp_cur = b_col_zp;

                    जबतक (rows_remaining > 0) {
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

    व्योम forward_mul_mat_id(ggml_compute_params * params, ggml_tensor * op) {
        constexpr माप_प्रकार a_blk_len = INTER_आकार;
        constexpr माप_प्रकार b_blk_len = INTER_आकार;

        स्थिर ggml_tensor * src0 = op->src[0];
        स्थिर ggml_tensor * src1 = op->src[1];
        स्थिर ggml_tensor * ids  = op->src[2];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_द्विआधारी_OP_LOCALS

        पूर्णांक ith = params->ith;
        पूर्णांक nth = params->nth;

        // row groups
        स्थिर पूर्णांक n_ids = ids->ne[0];  // n_expert_used
        स्थिर पूर्णांक n_as  = ne02;        // n_expert

        काष्ठा mmid_row_mapping {
            int32_t i1;
            int32_t i2;
        };

        spacemit_kernels::quantize_a_row_def           quantize_a_row_i8;
        spacemit_kernels::gemm_kernel_quantize_def     gemm_kernel;
        spacemit_kernels::moe_gemm_kernel_quantize_def moe_gemm_kernel_m2;
        bool                                           set_kernel_impl = false;
        माप_प्रकार                                         block_stride_a  = spacemit_kernels::q8_blk_आकार(QK4_0);

#अगर defined(RISCV64_SPACEMIT_IME2)
        अगर (!set_kernel_impl && (global_spine_env_info.use_ime2)) {
            quantize_a_row_i8 = spacemit_kernels::rvv::quantize_a_row_i8;
            block_stride_a    = spacemit_kernels::q8_blk_आकार(QK4_0, true);

            अगर constexpr (std::is_same_v<BLOC_TYPE, block_q6_K> || std::is_same_v<BLOC_TYPE, block_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i8;
                set_kernel_impl = true;
            } अन्यथा अगर constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> || std::is_same_v<BLOC_TYPE, block_q4_1> ||
                                 std::is_same_v<BLOC_TYPE, block_q4_K>) {
                अगर constexpr (INTER_आकार == 256) {
                    gemm_kernel       = spacemit_kernels::ime2::gemm_kernel_i8i4_hp;
                    quantize_a_row_i8 = spacemit_kernels::rvv::quantize_a_row_i8_hp;
                    block_stride_a    = spacemit_kernels::q8_hp_blk_आकार(a_blk_len, true, true);
                    set_kernel_impl   = true;
                } अन्यथा {
                    gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i4;
                    moe_gemm_kernel_m2 = spacemit_kernels::ime2::moe_m2_gemm_kernel_i8i4;
                    quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8;
                    block_stride_a     = spacemit_kernels::q8_blk_आकार(a_blk_len, true);
                    set_kernel_impl    = true;
                }
            } अन्यथा अगर constexpr (std::is_same_v<BLOC_TYPE, block_q2_K>) {
                quantize_a_row_i8 = spacemit_kernels::rvv::quantize_a_row_i8k;
                block_stride_a    = spacemit_kernels::q8k_blk_आकार(a_blk_len);
                gemm_kernel       = spacemit_kernels::ime2::gemm_kernel_i8i2k;
                set_kernel_impl   = true;
            } अन्यथा अगर constexpr (std::is_same_v<BLOC_TYPE, block_q3_K>) {
                quantize_a_row_i8 = spacemit_kernels::rvv::quantize_a_row_i8k;
                block_stride_a    = spacemit_kernels::q8k_blk_आकार(a_blk_len);
                gemm_kernel       = spacemit_kernels::ime2::gemm_kernel_i8i3k;
                set_kernel_impl   = true;
            } अन्यथा अगर constexpr (std::is_same_v<BLOC_TYPE, block_mxfp4>) {
                gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8mxfp4;
                moe_gemm_kernel_m2 = spacemit_kernels::ime2::moe_m2_gemm_kernel_i8mxfp4;
                set_kernel_impl    = true;
            } अन्यथा अगर constexpr (std::is_same_v<BLOC_TYPE, block_q5_1> || std::is_same_v<BLOC_TYPE, block_q5_K> ||
                                 std::is_same_v<BLOC_TYPE, block_q5_0>) {
                gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i5;
                moe_gemm_kernel_m2 = spacemit_kernels::ime2::moe_m2_gemm_kernel_i8i5;
                set_kernel_impl    = true;
            }
        }
#पूर्ण_अगर

#अगर defined(RISCV64_SPACEMIT_IME1)
        अगर (!set_kernel_impl && (global_spine_env_info.use_ime1)) {
            quantize_a_row_i8 = spacemit_kernels::ime1::quantize_a_row_i8;

            अगर constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> || std::is_same_v<BLOC_TYPE, block_q4_1> ||
                          std::is_same_v<BLOC_TYPE, block_q4_K>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i4;
                set_kernel_impl = true;
            } अन्यथा अगर constexpr (std::is_same_v<BLOC_TYPE, block_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i8;
                set_kernel_impl = true;
            }
        }
#पूर्ण_अगर
        अगर (!set_kernel_impl) {
            GGML_ABORT("no kernel implementation found for the block type");
        }

        स्थिर माप_प्रकार a_k_blks = spacemit_kernels::div_round_up(ne10, a_blk_len);
        स्थिर माप_प्रकार b_k_blks = spacemit_kernels::div_round_up(ne10, b_blk_len);

        स्थिर माप_प्रकार nbw1                = a_k_blks * block_stride_a;
        स्थिर माप_प्रकार nbw2                = ne11 * nbw1;
        स्थिर माप_प्रकार nbw3                = nbw2 * ne12;
        स्थिर माप_प्रकार gemm_workspace_आकार = GGML_PAD(nbw3, alignof(int64_t));

        स्थिर uintptr_t ws_ptr         = reinterpret_cast<uintptr_t>(params->wdata);
        स्वतः *          quant_a_बफर = reinterpret_cast<uint8_t *>(ws_ptr);

        अगर (ne11 == 1) {
            क्रम (int64_t ii = ith; ii < ne12 * a_k_blks; ii += nth) {
                int64_t i12       = ii / a_k_blks;
                int64_t ak_blk_id = ii % a_k_blks;
                quantize_a_row_i8(a_blk_len, (भग्न *) ((अक्षर *) src1->तथ्य + i12 * nb12) + ak_blk_id * a_blk_len,
                                  a_blk_len, quant_a_बफर + i12 * nbw2 + ak_blk_id * block_stride_a);
            }
        } अन्यथा {
            क्रम (int64_t ii = ith; ii < ne12 * ne11; ii += nth) {
                int64_t i12 = ii / ne11;
                int64_t i11 = ii % ne11;
                quantize_a_row_i8(a_blk_len, (भग्न *) ((अक्षर *) src1->तथ्य + i12 * nb12 + i11 * nb11), ne10,
                                  quant_a_बफर + i12 * nbw2 + i11 * nbw1);
            }
        }

#घोषणा MMID_MATRIX_ROW(row_id, i1) matrix_rows[(row_id) *ne12 + (i1)]

        int64_t *          matrix_row_counts       = (int64_t *) (ws_ptr + gemm_workspace_आकार);
        int32_t *          valid_ep_गिनती          = (int32_t *) (matrix_row_counts + n_as);
        int32_t *          valid_act_गिनती         = (int32_t *) (valid_ep_गिनती + 1);
        int64_t *          valid_matrix_row_counts = (int64_t *) (valid_act_गिनती + 1);
        mmid_row_mapping * matrix_rows             = (mmid_row_mapping *) (valid_matrix_row_counts + n_as);

        अगर (ith == 0) {
            // initialize matrix_row_counts
            स_रखो(matrix_row_counts, 0, n_as * माप(int64_t));

            // group rows by src0 matrix
            क्रम (int32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
                क्रम (int32_t id = 0; id < n_ids; ++id) {
                    स्थिर int32_t i02 =
                        *(स्थिर int32_t *) ((स्थिर अक्षर *) ids->तथ्य + iid1 * ids->nb[1] + id * ids->nb[0]);

                    GGML_ASSERT(i02 >= 0 && i02 < n_as);

                    MMID_MATRIX_ROW(i02, matrix_row_counts[i02]) = { id, iid1 };
                    matrix_row_counts[i02] += 1;
                }
            }

            int32_t valid_ep_गिनती_t  = 0;
            int32_t valid_act_गिनती_t = 0;
            क्रम (पूर्णांक cur_a = 0; cur_a < n_as; ++cur_a) {
                स्थिर int64_t cne1 = matrix_row_counts[cur_a];
                अगर (cne1 == 0) {
                    जारी;
                }
                valid_matrix_row_counts[valid_ep_गिनती_t] = cur_a;
                valid_act_गिनती_t += cne1;
                valid_ep_गिनती_t += 1;
            }
            valid_ep_गिनती[0]  = valid_ep_गिनती_t;
            valid_act_गिनती[0] = valid_act_गिनती_t;
        }

        स्थिर int64_t barrier_idx = static_cast<int64_t>(ith / 2);

        GGML_ASSERT(global_spine_env_info.init_barrier != nullptr);
        GGML_ASSERT(barrier_idx < spine_init_barrier_गिनती);
        spine_barrier_t * cur_barrier = &global_spine_env_info.init_barrier[barrier_idx];

        ggml_barrier(params->threadpool);

        स्थिर माप_प्रकार row_stride_b      = b_k_blks * get_repacked_block_type_आकार<BLOC_TYPE, INTER_आकार, NB_COLS>();
        स्थिर माप_प्रकार expert_b_stride   = ne01 * row_stride_b;
        स्थिर माप_प्रकार per_nb_cols_wsize = NB_COLS * row_stride_b;

        std::array<स्थिर uint8_t *, 2> src_workspaces;
        std::array<भग्न *, 2>         dst_workspaces;

        स्वतः *     tcm_बफर      = ggml::cpu::riscv64_spacemit::tls_context.tcm_बफर;
        स्थिर स्वतः tcm_बफर_आकार = ggml::cpu::riscv64_spacemit::tls_context.tcm_बफर_आकार;

        स्थिर स्वतः valid_ep_गिनती_t  = valid_ep_गिनती[0];
        स्थिर स्वतः valid_act_गिनती_t = valid_act_गिनती[0];

        पूर्णांक nth_es = 1;
        पूर्णांक nth_n  = nth;

        पूर्णांक ith_es = ith % nth_es;
        पूर्णांक ith_n  = (ith / nth_es) % nth_n;

        अगर (valid_ep_गिनती_t % nth == 0 && tcm_बफर != nullptr && valid_ep_गिनती_t == n_as &&
            valid_act_गिनती_t == n_as && per_nb_cols_wsize <= tcm_बफर_आकार) {
            क्रम (int64_t valid_id = ith; valid_id < valid_ep_गिनती_t; valid_id += nth) {
                स्थिर int64_t cur_a = valid_matrix_row_counts[valid_id];

                स्वतः * src0_cur = (uint8_t *) src0->तथ्य + cur_a * expert_b_stride;

                mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, 0);
                स्थिर पूर्णांक        id          = row_mapping.i1;
                स्थिर int64_t    i11         = id % ne11;
                स्थिर int64_t    i12         = row_mapping.i2;
                स्थिर int64_t    i1          = id;
                स्थिर int64_t    i2          = i12;

                स्वतः *  src1_col = quant_a_बफर + (i11 * nbw1 + i12 * nbw2);
                भग्न * c_blk    = (भग्न *) ((अक्षर *) dst->तथ्य + (i1 * nb1 + i2 * nb2));

                uint8_t * a_row = src1_col;
                uint8_t * b_col = reinterpret_cast<uint8_t *>(tcm_बफर);
                अगर ((nbw1 + per_nb_cols_wsize) <= tcm_बफर_आकार) {
                    a_row = (uint8_t *) tcm_बफर;
                    b_col = reinterpret_cast<uint8_t *>(tcm_बफर) + nbw1;
                }
                uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? b_col : nullptr;

                अगर (ith % 2 == 0) {
                    spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(src0_cur), per_nb_cols_wsize);

                    अगर (a_row != src1_col) {
                        spacemit_kernels::rvv::memcpy1d(a_row, src1_col, nbw1);
                    }
                }

                spine_barrier_wait(cur_barrier);

                अगर (ith % 2 != 0) {
                    अगर (a_row != src1_col) {
                        spacemit_kernels::rvv::memcpy1d(a_row, src1_col, nbw1);
                    }

                    spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(src0_cur), per_nb_cols_wsize);
                }

                int64_t nb_real = std::min(ne01, NB_COLS);
                क्रम (int64_t ni = 0; ni < ne01; ni += NB_COLS) {
                    अगर (ith % 2 != 0) {
                        spine_barrier_wait(cur_barrier);
                    }

                    gemm_kernel(b_blk_len, a_row, b_col, b_col_zp, c_blk + ni, 1, nb_real, b_k_blks, ne01);

                    अगर (ith % 2 == 0) {
                        spine_barrier_wait(cur_barrier);
                    }

                    स्थिर int64_t अग्र_ni = ni + NB_COLS;
                    अगर (अग्र_ni < ne01) {
                        nb_real = std::min(ne01 - अग्र_ni, NB_COLS);
                        spacemit_kernels::rvv::memcpy1d(
                            b_col, reinterpret_cast<uint8_t *>(src0_cur) + अग्र_ni * row_stride_b, per_nb_cols_wsize);
                    }
                }
            }
        } अन्यथा {
            क्रम (int64_t valid_id = ith_es; valid_id < valid_ep_गिनती_t; valid_id += nth_es) {
                स्थिर int64_t cur_a = valid_matrix_row_counts[valid_id];
                स्थिर int64_t cne1  = matrix_row_counts[cur_a];

                int64_t src1_cur_start = 0;
                int64_t src1_cur_end   = cne1;

                int64_t src0_cur_start = (ith_n * ne01) / nth_n;
                int64_t src0_cur_end   = MIN(((ith_n + 1) * ne01) / nth_n, ne01);

                अगर (src1_cur_start >= src1_cur_end || src0_cur_start >= src0_cur_end) {
                    जारी;
                }

                src0_cur_start =
                    (src0_cur_start % NB_COLS) ? src0_cur_start + NB_COLS - (src0_cur_start % NB_COLS) : src0_cur_start;
                src0_cur_end =
                    (src0_cur_end % NB_COLS) ? src0_cur_end + NB_COLS - (src0_cur_end % NB_COLS) : src0_cur_end;

                स्वतः *    src0_cur = (uint8_t *) src0->तथ्य + cur_a * expert_b_stride + src0_cur_start * row_stride_b;
                uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? src0_cur : nullptr;

                माप_प्रकार extra_tcm_बफर_आकार = tcm_बफर_आकार;
                व्योम * extra_tcm_बफर      = tcm_बफर;
                अगर (tcm_बफर != nullptr && (src1_cur_end - src1_cur_start) >= 4 &&
                    (src0_cur_end - src0_cur_start) * row_stride_b <= tcm_बफर_आकार) {
                    spacemit_kernels::rvv::memcpy1d(tcm_बफर, src0_cur,
                                                    (src0_cur_end - src0_cur_start) * row_stride_b);
                    src0_cur = reinterpret_cast<uint8_t *>(tcm_बफर);
                    b_col_zp = block_type_has_zp<BLOC_TYPE>() ? src0_cur : nullptr;
                    extra_tcm_बफर_आकार -= (src0_cur_end - src0_cur_start) * row_stride_b;
                    extra_tcm_बफर = reinterpret_cast<व्योम *>(reinterpret_cast<uintptr_t>(tcm_बफर) +
                                                                (src0_cur_end - src0_cur_start) * row_stride_b);
                }

                पूर्णांक ir1 = src1_cur_start;

                अगर (extra_tcm_बफर_आकार >= nbw1 && extra_tcm_बफर != nullptr) {
                    int64_t quant_a_tile_आकार = extra_tcm_बफर_आकार / nbw1;
                    करो {
                        quant_a_tile_आकार = MIN(quant_a_tile_आकार, src1_cur_end - ir1);

                        uint8_t * quant_a_tile_बफर = reinterpret_cast<uint8_t *>(extra_tcm_बफर);

                        पूर्णांक iir1 = ir1;
                        क्रम (; iir1 < (ir1 + quant_a_tile_आकार); ++iir1) {
                            mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, iir1);

                            स्थिर पूर्णांक id = row_mapping.i1;  // selected expert index

                            स्थिर int64_t i11 = id % ne11;
                            स्थिर int64_t i12 = row_mapping.i2;  // row index in src1

                            स्वतः * src1_col = quant_a_बफर + (i11 * nbw1 + i12 * nbw2);
                            spacemit_kernels::rvv::memcpy1d(quant_a_tile_बफर, src1_col, nbw1);
                            quant_a_tile_बफर = quant_a_tile_बफर + nbw1;
                        }

                        quant_a_tile_बफर = reinterpret_cast<uint8_t *>(extra_tcm_बफर);
                        iir1                = ir1;

                        अगर (moe_gemm_kernel_m2 != nullptr) {
                            क्रम (; iir1 < (ir1 + quant_a_tile_आकार - 1); iir1 += 2, quant_a_tile_बफर += 2 * nbw1) {
                                mmid_row_mapping row_mapping_0 = MMID_MATRIX_ROW(cur_a, iir1);
                                mmid_row_mapping row_mapping_1 = MMID_MATRIX_ROW(cur_a, iir1 + 1);

                                src_workspaces[0] = quant_a_tile_बफर;
                                src_workspaces[1] = quant_a_tile_बफर + nbw1;

                                dst_workspaces[0] =
                                    (भग्न *) ((अक्षर *) dst->तथ्य + (row_mapping_0.i1 * nb1 + row_mapping_0.i2 * nb2)) +
                                    src0_cur_start;
                                dst_workspaces[1] = (भग्न *) ((अक्षर *) dst->तथ्य +
                                                               ((row_mapping_1.i1) * nb1 + (row_mapping_1.i2) * nb2)) +
                                                    src0_cur_start;
                                moe_gemm_kernel_m2(b_blk_len, src_workspaces.तथ्य(), src0_cur, b_col_zp,
                                                   dst_workspaces.तथ्य(), 1, src0_cur_end - src0_cur_start, b_k_blks,
                                                   ne01);
                            }
                        }

                        क्रम (; iir1 < (ir1 + quant_a_tile_आकार); iir1++, quant_a_tile_बफर += nbw1) {
                            mmid_row_mapping row_mapping_0 = MMID_MATRIX_ROW(cur_a, iir1);

                            gemm_kernel(
                                b_blk_len, quant_a_tile_बफर, src0_cur, b_col_zp,
                                (भग्न *) ((अक्षर *) dst->तथ्य + (row_mapping_0.i1 * nb1 + row_mapping_0.i2 * nb2)) +
                                    src0_cur_start,
                                1, src0_cur_end - src0_cur_start, b_k_blks, ne01);
                        }

                        ir1 += quant_a_tile_आकार;
                    } जबतक (ir1 < src1_cur_end);
                } अन्यथा {
                    अगर (moe_gemm_kernel_m2 != nullptr) {
                        क्रम (; ir1 < src1_cur_end - 1; ir1 += 2) {
                            क्रम (पूर्णांक iir1 = 0; iir1 < 2; ++iir1) {
                                mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, ir1 + iir1);

                                स्थिर पूर्णांक id = row_mapping.i1;  // selected expert index

                                स्थिर int64_t i11 = id % ne11;
                                स्थिर int64_t i12 = row_mapping.i2;  // row index in src1

                                स्थिर int64_t i1 = id;               // selected expert index
                                स्थिर int64_t i2 = i12;              // row

                                src_workspaces[iir1] = quant_a_बफर + (i11 * nbw1 + i12 * nbw2);

                                dst_workspaces[iir1] =
                                    (भग्न *) ((अक्षर *) dst->तथ्य + (i1 * nb1 + i2 * nb2)) + src0_cur_start;
                            }

                            moe_gemm_kernel_m2(b_blk_len, src_workspaces.तथ्य(), src0_cur, b_col_zp,
                                               dst_workspaces.तथ्य(), 1, src0_cur_end - src0_cur_start, b_k_blks, ne01);
                        }
                    }

                    क्रम (; ir1 < src1_cur_end; ir1++) {
                        mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, ir1);

                        स्थिर पूर्णांक id = row_mapping.i1;  // selected expert index

                        स्थिर int64_t i11 = id % ne11;
                        स्थिर int64_t i12 = row_mapping.i2;  // row index in src1

                        स्थिर int64_t i1 = id;               // selected expert index
                        स्थिर int64_t i2 = i12;              // row

                        स्वतः * src1_col = quant_a_बफर + (i11 * nbw1 + i12 * nbw2);

                        gemm_kernel(b_blk_len, src1_col, src0_cur, b_col_zp,
                                    (भग्न *) ((अक्षर *) dst->तथ्य + (i1 * nb1 + i2 * nb2)) + src0_cur_start, 1,
                                    src0_cur_end - src0_cur_start, b_k_blks, ne01);
                    }
                }
            }
        }
#अघोषित MMID_MATRIX_ROW
    }

    पूर्णांक repack(ggml_tensor * t, स्थिर व्योम * तथ्य, माप_प्रकार तथ्य_आकार) override {
        GGML_LOG_DEBUG("%s: repack tensor %s with %s_%dx%d\n", __func__, t->name, ggml_type_name(t->type),
                       (पूर्णांक) NB_COLS, (पूर्णांक) INTER_आकार);
        वापस ggml::cpu::riscv64_spacemit::repack<BLOC_TYPE, INTER_आकार, NB_COLS>(t, तथ्य, तथ्य_आकार);
    }
};

श्रेणी tensor_traits_common : खुला tensor_traits_base {
    bool work_आकार(पूर्णांक n_threads, स्थिर ggml_tensor * op, माप_प्रकार & size) override {
        चयन (op->op) {
            हाल GGML_OP_FLASH_ATTN_EXT:
                {
                    स्थिर पूर्णांक     n_tasks = n_threads;
                    स्थिर int64_t neq2    = op->src[0]->ne[2];  // number of query heads
                    स्थिर int64_t DK      = op->src[1]->ne[0];
                    स्थिर int64_t DV      = op->src[2]->ne[0];  // DV

                    // Tiled flash attention scratch (tile sizes defined in common.h)
                    // Per-thread: Q_q + KQ + mask + VKQ32 + V32 + K_f32 + padding
                    माप_प्रकार prefill = माप(भग्न) *
                                     (GGML_FA_TILE_Q * DK + 2 * GGML_FA_TILE_Q * GGML_FA_TILE_KV + GGML_FA_TILE_Q * DV +
                                      GGML_FA_TILE_KV * DV + GGML_FA_TILE_KV * DK) *
                                     n_tasks;

                    // Decode path: n_kv_chunks = n_tasks (one chunk per thread)
                    // Per-thread: VKQ accmulator (DV), partial M, partial S + intra-thread scratch for V, Q and VKQ
                    माप_प्रकार n_chunks = n_tasks;
                    माप_प्रकार decode   = माप(भग्न) * (neq2 * n_chunks * (2 + DV) + n_tasks * (DK + 2 * DV));

                    size = MAX(prefill, decode);
                }
                वापस true;
            शेष:
                अवरोध;
        }
        वापस false;
    }

    bool compute_forward(ggml_compute_params * params, ggml_tensor * op) override {
        चयन (op->op) {
            हाल GGML_OP_NORM:
                चयन (op->src[0]->type) {
                    हाल GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_norm_f32(params, op);
                        वापस true;
                    शेष:
                        GGML_ABORT("fatal error");
                }
            हाल GGML_OP_RMS_NORM:
                चयन (op->src[0]->type) {
                    हाल GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_rms_norm_f32(params, op);
                        वापस true;
                    शेष:
                        GGML_ABORT("fatal error");
                }
            हाल GGML_OP_ADD:
                चयन (op->src[0]->type) {
                    हाल GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_द्विआधारी<GGML_OP_ADD, भग्न>(params, op);
                        वापस true;
                    हाल GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_द्विआधारी<GGML_OP_ADD, _Float16>(params, op);
                        वापस true;
                    शेष:
                        ggml_compute_forward_add(params, op);
                        वापस true;
                }
            हाल GGML_OP_SUB:
                चयन (op->src[0]->type) {
                    हाल GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_द्विआधारी<GGML_OP_SUB, भग्न>(params, op);
                        वापस true;
                    हाल GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_द्विआधारी<GGML_OP_SUB, _Float16>(params, op);
                        वापस true;
                    शेष:
                        ggml_compute_forward_sub(params, op);
                        वापस true;
                }
            हाल GGML_OP_MUL:
                चयन (op->src[0]->type) {
                    हाल GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_द्विआधारी<GGML_OP_MUL, भग्न>(params, op);
                        वापस true;
                    हाल GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_द्विआधारी<GGML_OP_MUL, _Float16>(params, op);
                        वापस true;
                    शेष:
                        ggml_compute_forward_mul(params, op);
                        वापस true;
                }
            हाल GGML_OP_DIV:
                चयन (op->src[0]->type) {
                    हाल GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_द्विआधारी<GGML_OP_DIV, भग्न>(params, op);
                        वापस true;
                    हाल GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_द्विआधारी<GGML_OP_DIV, _Float16>(params, op);
                        वापस true;
                    शेष:
                        ggml_compute_forward_div(params, op);
                        वापस true;
                }
            हाल GGML_OP_FLASH_ATTN_EXT:
                forward_flash_attn_ext_f16(params, op);
                वापस true;
            हाल GGML_OP_CONT:
                {
                    स्थिर ggml_tensor * src0 = op->src[0];
                    अगर (op->type == src0->type && op->nb[0] != src0->nb[0] && op->nb[0] == src0->nb[1] &&
                        op->ne[3] * op->ne[2] * op->nb[2] == src0->ne[3] * src0->ne[2] * src0->nb[2]) {
                        spacemit_kernels::rvv::forward_cont_with_permute(params, op);
                    } अन्यथा {
                        ggml_compute_forward_cont(params, op);
                    }
                    वापस true;
                }
            हाल GGML_OP_CPY:
                {
                    स्थिर ggml_tensor * src0 = op->src[0];
                    अगर (op->type == src0->type && op->nb[0] == src0->nb[1] && src0->nb[0] != src0->nb[1] &&
                        ggml_nelements(src0) == ggml_nelements(op)) {
                        spacemit_kernels::rvv::forward_cpy_with_permute(params, op);
                    } अन्यथा {
                        ggml_compute_forward_cpy(params, op);
                    }
                    वापस true;
                }
            हाल GGML_OP_REPEAT:
                {
                    स्थिर bool rows_equal         = ggml_nrows(op->src[0]) == ggml_nrows(op);
                    स्थिर bool broadcast_or_equal = op->src[0]->ne[0] == 1 || op->src[0]->ne[0] == op->ne[0];

                    अगर (rows_equal && broadcast_or_equal) {
                        चयन (op->src[0]->type) {
                            हाल GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_repeat_nrows<int32_t>(params, op);
                                वापस true;
                            हाल GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_repeat_nrows<int16_t>(params, op);
                                वापस true;
                            शेष:
                                अवरोध;
                        }
                    }

                    अगर (op->src[0]->ne[1] == 1 && op->src[0]->ne[0] == op->ne[0]) {
                        चयन (op->src[0]->type) {
                            हाल GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_repeat_dim1<int32_t>(params, op);
                                वापस true;
                            हाल GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_repeat_dim1<int16_t>(params, op);
                                वापस true;
                            शेष:
                                अवरोध;
                        }
                    }

                    ggml_compute_forward_repeat(params, op);
                }
                वापस true;
            हाल GGML_OP_योग_ROWS:
                {
                    अगर (op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32) {
                        spacemit_kernels::rvv::forward_योग_rows<भग्न>(params, op);
                    } अन्यथा {
                        ggml_compute_forward_योग_rows(params, op);
                    }
                }
                वापस true;
            हाल GGML_OP_GET_ROWS:
                {
                    अगर (op->src[0]->type == op->type) {
                        चयन (op->src[0]->type) {
                            हाल GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_get_rows<int32_t>(params, op);
                                वापस true;
                            हाल GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_get_rows<int16_t>(params, op);
                                वापस true;
                            शेष:
                                अवरोध;
                        }
                    }

                    ggml_compute_forward_get_rows(params, op);
                }
                वापस true;
            हाल GGML_OP_CONCAT:
                {
                    स्थिर int32_t dim = ggml_get_op_params_i32(op, 0);
                    अगर (dim == 0 && op->type == op->src[0]->type) {
                        चयन (op->src[0]->type) {
                            हाल GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_concat<int32_t>(params, op);
                                वापस true;
                            हाल GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_concat<int16_t>(params, op);
                                वापस true;
                            शेष:
                                अवरोध;
                        }
                    }

                    ggml_compute_forward_concat(params, op);
                }
                वापस true;
            // TODO For GGML_OP_GATED_DELTA_NET
            // case GGML_OP_GATED_DELTA_NET:
            //     return true;
            शेष:
                अवरोध;
        }
        वापस false;
    }

    व्योम forward_flash_attn_ext_f16(स्थिर ggml_compute_params * params, ggml_tensor * dst) {
        स्थिर ggml_tensor * q = dst->src[0];
        स्थिर ggml_tensor * k = dst->src[1];
        स्थिर ggml_tensor * v = dst->src[2];

        GGML_TENSOR_LOCALS(int64_t, neq, q, ne)
        GGML_TENSOR_LOCALS(माप_प्रकार, nbq, q, nb)
        GGML_TENSOR_LOCALS(int64_t, nek, k, ne)
        GGML_TENSOR_LOCALS(माप_प्रकार, nbk, k, nb)
        GGML_TENSOR_LOCALS(int64_t, nev, v, ne)
        GGML_TENSOR_LOCALS(माप_प्रकार, nbv, v, nb)
        GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
        GGML_TENSOR_LOCALS(माप_प्रकार, nb, dst, nb)

        स्थिर int64_t DK = nek0;
        स्थिर int64_t DV = nev0;

        स्थिर bool supported_prec  = (dst->op_params[3] == GGML_PREC_F32 || dst->op_params[3] == GGML_PREC_DEFAULT);
        स्थिर bool supported_types = (q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16);
        स्थिर bool supported_shape = (DK > 0 && DK <= 128 && DV > 0 && DV <= 128);
        स्थिर bool supported_vlen  = (__riscv_vlenb() == 128);

        अगर (!(supported_prec && supported_types && supported_shape && supported_vlen)) {
            ggml_compute_forward_flash_attn_ext(params, dst);
            वापस;
        }

        // total rows in q
        स्थिर int64_t nr = neq1 * neq2 * neq3;

        // rows per thread
        स्थिर पूर्णांक ith = params->ith;
        स्थिर पूर्णांक nth = params->nth;

        अटल constexpr int64_t Q_TILE_SZ = ggml_fa_tile_config::Q;
        स्थिर bool               use_tiled = !params->use_ref && (neq1 >= Q_TILE_SZ);

        // 4x chunks per thread
        // int     nth_scaled = nth * 4;
        // int64_t chunk_size = (nr + nth_scaled - 1) / nth_scaled;
        // int64_t nchunk     = (nr + chunk_size - 1) / chunk_size;

        // if (nth == 1 || nchunk < nth) {
        //     nchunk = nth;
        // }

        int64_t nchunk = nth;

        अगर (ith == 0) {
            // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
            ggml_threadpool_chunk_set(params->threadpool, nth);
        }

        ggml_barrier(params->threadpool);

        // The number of elements in each chunk
        स्थिर int64_t dr = (nr + nchunk - 1) / nchunk;

        // The first chunk comes from our thread_id, the rest will get auto-assigned.
        पूर्णांक current_chunk = ith;

        जबतक (current_chunk < nchunk) {
            स्थिर int64_t ir0 = dr * current_chunk;
            स्थिर int64_t ir1 = MIN(ir0 + dr, nr);

            अगर (use_tiled) {
                spacemit_kernels::rvv::forward_flash_attn_ext_f16_tiled_vlen1024_vf16(
                    params, dst, ir0, ir1, ggml::cpu::riscv64_spacemit::tls_context.tcm_बफर,
                    ggml::cpu::riscv64_spacemit::tls_context.tcm_बफर_आकार);
            } अन्यथा {
                spacemit_kernels::rvv::forward_flash_attn_ext_f16_one_chunk_vlen1024_vf16(
                    params, dst, ir0, ir1, ggml::cpu::riscv64_spacemit::tls_context.tcm_बफर,
                    ggml::cpu::riscv64_spacemit::tls_context.tcm_बफर_आकार);
            }

            current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
        }
    }

    पूर्णांक repack(ggml_tensor * t, स्थिर व्योम * तथ्य, माप_प्रकार तथ्य_आकार) override {
        स_नकल(t->तथ्य, तथ्य, तथ्य_आकार);
        वापस 0;
    }
};

// Impl By IME1
अटल स्थिर tensor_traits<block_q4_0, 32, 16>  q4_0_16x32_q8_0;
अटल स्थिर tensor_traits<block_q4_1, 32, 16>  q4_1_16x32_q8_0;
अटल स्थिर tensor_traits<block_q4_K, 32, 16>  q4_k_16x32_q8_0;
अटल स्थिर tensor_traits<block_q8_0, 32, 16>  q8_0_16x32_q8_0;
// Impl By IME2
अटल स्थिर tensor_traits<block_q2_K, 256, 32> q2_k_32x256_q8_0;
अटल स्थिर tensor_traits<block_q3_K, 256, 32> q3_k_32x256_q8_0;
अटल स्थिर tensor_traits<block_q4_0, 32, 32>  q4_0_32x32_q8_0;
अटल स्थिर tensor_traits<block_q4_1, 32, 32>  q4_1_32x32_q8_0;
अटल स्थिर tensor_traits<block_q4_0, 256, 32> q4_0_32x256_q8_0;
अटल स्थिर tensor_traits<block_q4_1, 256, 32> q4_1_32x256_q8_0;
अटल स्थिर tensor_traits<block_q4_K, 32, 32>  q4_k_32x32_q8_0;
अटल स्थिर tensor_traits<block_q6_K, 32, 32>  q6_k_32x32_q8_0;
अटल स्थिर tensor_traits<block_q8_0, 32, 32>  q8_0_32x32_q8_0;
अटल स्थिर tensor_traits<block_mxfp4, 32, 32> mxfp4_32x32_q8_0;
अटल स्थिर tensor_traits<block_q5_K, 32, 32>  q5_k_32x32_q8_0;
अटल स्थिर tensor_traits<block_q5_1, 32, 32>  q5_1_32x32_q8_0;
अटल स्थिर tensor_traits<block_q5_0, 32, 32>  q5_0_32x32_q8_0;
// Impl By RVV
अटल स्थिर tensor_traits_common               rvv_impl;

}  // namespace ggml::cpu::riscv64_spacemit

अटल स्थिर ggml::cpu::tensor_traits * ggml_riscv64_spacemit_get_optimal_repack_type(स्थिर ggml_tensor * cur) {
    चयन (cur->type) {
        हाल GGML_TYPE_Q2_K:
            {
#अगर defined(RISCV64_SPACEMIT_IME2)
                अगर (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    वापस &ggml::cpu::riscv64_spacemit::q2_k_32x256_q8_0;
                }
#पूर्ण_अगर
            }
            अवरोध;
        हाल GGML_TYPE_Q3_K:
            {
#अगर defined(RISCV64_SPACEMIT_IME2)
                अगर (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    वापस &ggml::cpu::riscv64_spacemit::q3_k_32x256_q8_0;
                }
#पूर्ण_अगर
            }
            अवरोध;
        हाल GGML_TYPE_Q4_0:
            {
#अगर defined(RISCV64_SPACEMIT_IME2)
                अगर (cur->ne[1] % 32 == 0 && cur->ne[0] % 256 == 0 &&
                    (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    वापस &ggml::cpu::riscv64_spacemit::q4_0_32x256_q8_0;
                }

                अगर (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    वापस &ggml::cpu::riscv64_spacemit::q4_0_32x32_q8_0;
                }
#पूर्ण_अगर

#अगर defined(RISCV64_SPACEMIT_IME1)
                अगर (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    वापस &ggml::cpu::riscv64_spacemit::q4_0_16x32_q8_0;
                }
#पूर्ण_अगर
            }
            अवरोध;
        हाल GGML_TYPE_Q4_1:
            {
#अगर defined(RISCV64_SPACEMIT_IME2)
                // TODO
                // if (cur->ne[1] % 32 == 0 && cur->ne[0] % 256 == 0 &&
                //     (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                //     return &ggml::cpu::riscv64_spacemit::q4_1_32x256_q8_0;
                // }

                अगर (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    वापस &ggml::cpu::riscv64_spacemit::q4_1_32x32_q8_0;
                }
#पूर्ण_अगर

#अगर defined(RISCV64_SPACEMIT_IME1)
                अगर (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    वापस &ggml::cpu::riscv64_spacemit::q4_1_16x32_q8_0;
                }
#पूर्ण_अगर
            }
            अवरोध;
        हाल GGML_TYPE_Q4_K:
            {
#अगर defined(RISCV64_SPACEMIT_IME2)
                अगर (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    वापस &ggml::cpu::riscv64_spacemit::q4_k_32x32_q8_0;
                }
#पूर्ण_अगर

#अगर defined(RISCV64_SPACEMIT_IME1)
                अगर (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    वापस &ggml::cpu::riscv64_spacemit::q4_k_16x32_q8_0;
                }
#पूर्ण_अगर
            }
            अवरोध;
        हाल GGML_TYPE_Q6_K:
            {
#अगर defined(RISCV64_SPACEMIT_IME2)
                अगर ((ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    वापस &ggml::cpu::riscv64_spacemit::q6_k_32x32_q8_0;
                }
#पूर्ण_अगर
            }
            अवरोध;
        हाल GGML_TYPE_Q8_0:
            {
#अगर defined(RISCV64_SPACEMIT_IME2)
                अगर ((ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    वापस &ggml::cpu::riscv64_spacemit::q8_0_32x32_q8_0;
                }
#पूर्ण_अगर

#अगर defined(RISCV64_SPACEMIT_IME1)
                अगर (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    वापस &ggml::cpu::riscv64_spacemit::q8_0_16x32_q8_0;
                }
#पूर्ण_अगर
            }
            अवरोध;
        हाल GGML_TYPE_MXFP4:
            {
#अगर defined(RISCV64_SPACEMIT_IME2)
                // TODO
                // if (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                //     return &ggml::cpu::riscv64_spacemit::mxfp4_32x32_q8_0;
                // }
#पूर्ण_अगर
            }
            अवरोध;
        हाल GGML_TYPE_Q5_K:
            {
#अगर defined(RISCV64_SPACEMIT_IME2)
                अगर (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    वापस &ggml::cpu::riscv64_spacemit::q5_k_32x32_q8_0;
                }
#पूर्ण_अगर
            }
            अवरोध;
        हाल GGML_TYPE_Q5_1:
            {
#अगर defined(RISCV64_SPACEMIT_IME2)
                अगर (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    वापस &ggml::cpu::riscv64_spacemit::q5_1_32x32_q8_0;
                }
#पूर्ण_अगर
            }
            अवरोध;
        हाल GGML_TYPE_Q5_0:
            {
#अगर defined(RISCV64_SPACEMIT_IME2)
                अगर (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    वापस &ggml::cpu::riscv64_spacemit::q5_0_32x32_q8_0;
                }
#पूर्ण_अगर
            }
            अवरोध;
        शेष:
            अवरोध;
    }

    वापस nullptr;
}

अटल क्रमागत ggml_status ggml_backend_riscv64_spacemit_बफर_init_tensor(ggml_backend_बफर_t बफर,
                                                                         ggml_tensor *         tensor) {
    tensor->extra =
        (व्योम *) const_cast<ggml::cpu::tensor_traits *>(ggml_riscv64_spacemit_get_optimal_repack_type(tensor));

    GGML_UNUSED(बफर);

    वापस GGML_STATUS_SUCCESS;
}

अटल व्योम ggml_backend_riscv64_spacemit_बफर_free_बफर(ggml_backend_बफर_t बफर) {
    GGML_ASSERT(बफर);

    व्योम * base = बफर->context;
    अगर (base == nullptr) {
        वापस;
    }

    ggml::cpu::riscv64_spacemit::spine_mem_pool_free(base);
}

अटल व्योम * ggml_backend_riscv64_spacemit_बफर_get_base(ggml_backend_बफर_t बफर) {
    GGML_ASSERT(बफर);

    व्योम * base = बफर->context;
    GGML_ASSERT(base != nullptr);
    वापस base;
}

अटल व्योम ggml_backend_riscv64_spacemit_बफर_memset_tensor(ggml_backend_बफर_t बफर,
                                                               ggml_tensor *         tensor,
                                                               uint8_t               value,
                                                               माप_प्रकार                offset,
                                                               माप_प्रकार                size) {
    GGML_ASSERT(tensor);
    स_रखो((अक्षर *) tensor->तथ्य + offset, value, size);

    GGML_UNUSED(बफर);
}

अटल व्योम ggml_backend_riscv64_spacemit_बफर_clear(ggml_backend_बफर_t बफर, uint8_t value) {
    GGML_ASSERT(बफर);

    व्योम * base = बफर->context;
    GGML_ASSERT(base != nullptr);
    स_रखो(base, value, बफर->size);
}

अटल व्योम ggml_backend_riscv64_spacemit_बफर_set_tensor(ggml_backend_बफर_t बफर,
                                                            ggml_tensor *         tensor,
                                                            स्थिर व्योम *          तथ्य,
                                                            माप_प्रकार                offset,
                                                            माप_प्रकार                size) {
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    स्वतः tensor_traits = (ggml::cpu::riscv64_spacemit::tensor_traits_base *) tensor->extra;
    अगर (tensor_traits) {
        स्वतः OK = tensor_traits->repack(tensor, तथ्य, size);
        GGML_ASSERT(OK == 0);
    }

    GGML_UNUSED(बफर);
}

अटल स्थिर ggml_backend_बफर_i ggml_backend_riscv64_spacemit_बफर_i = {
    /* .free_buffer     = */ ggml_backend_riscv64_spacemit_बफर_free_बफर,
    /* .get_base        = */ ggml_backend_riscv64_spacemit_बफर_get_base,
    /* .init_tensor     = */ ggml_backend_riscv64_spacemit_बफर_init_tensor,
    /* .memset_tensor   = */ ggml_backend_riscv64_spacemit_बफर_memset_tensor,
    /* .set_tensor      = */ ggml_backend_riscv64_spacemit_बफर_set_tensor,
    /* .get_tensor      = */ nullptr,
    /* .set_tensor_2d   = */ nullptr,
    /* .get_tensor_2d   = */ nullptr,
    /* .cpy_tensor      = */ nullptr,
    /* .clear           = */ ggml_backend_riscv64_spacemit_बफर_clear,
    /* .reset           = */ nullptr,
};

अटल स्थिर अक्षर * ggml_backend_cpu_riscv64_spacemit_बफर_type_get_name(ggml_backend_बफर_type_t buft) {
    वापस "CPU_RISCV64_SPACEMIT";

    GGML_UNUSED(buft);
}

अटल ggml_backend_बफर_t ggml_backend_cpu_riscv64_spacemit_बफर_type_alloc_बफर(ggml_backend_बफर_type_t buft,
                                                                                        माप_प्रकार size) {
    व्योम * base = ggml::cpu::riscv64_spacemit::spine_mem_pool_alloc(size, 64);
    अगर (base == nullptr) {
        वापस nullptr;
    }

    वापस ggml_backend_बफर_init(buft, ggml_backend_riscv64_spacemit_बफर_i, base, size);
}

अटल माप_प्रकार ggml_backend_cpu_riscv64_spacemit_बफर_type_get_alignment(ggml_backend_बफर_type_t buft) {
    वापस 64;

    GGML_UNUSED(buft);
}

अटल माप_प्रकार ggml_backend_cpu_riscv64_spacemit_nbytes(ggml_backend_बफर_type_t buft, स्थिर ggml_tensor * tensor) {
    क्रम (पूर्णांक i = 0; i < GGML_MAX_DIMS; ++i) {
        अगर (tensor->ne[i] <= 0) {
            वापस 0;
        }
    }

    GGML_UNUSED(buft);

    स्थिर स्वतः plain_nbytes = [&]() {
        माप_प्रकार योग = ggml_type_आकार(tensor->type);
        क्रम (पूर्णांक i = 0; i < GGML_MAX_DIMS; ++i) {
            योग += (tensor->ne[i] - 1) * tensor->nb[i];
        }
        वापस योग;
    };

    स्थिर माप_प्रकार blck_आकार = ggml_blck_आकार(tensor->type);
    अगर (blck_आकार == 1) {
        वापस plain_nbytes();
    }

    स्थिर माप_प्रकार row_nbytes = tensor->ne[0] * tensor->nb[0] / blck_आकार;

    स्थिर स्वतः add_strided_nbytes = [&](माप_प्रकार योग, माप_प्रकार src_block_आकार, माप_प्रकार dst_block_आकार) {
        क्रम (पूर्णांक i = 1; i < GGML_MAX_DIMS; ++i) {
            योग += (tensor->ne[i] - 1) * (tensor->nb[i] / src_block_आकार) * dst_block_आकार;
        }
        वापस योग;
    };

    स्थिर स्वतः remap_block_nbytes = [&](माप_प्रकार src_block_आकार, माप_प्रकार dst_block_आकार, int64_t padded_rows = 0) {
        GGML_ASSERT(row_nbytes % src_block_आकार == 0);

        माप_प्रकार योग =
            add_strided_nbytes((row_nbytes / src_block_आकार) * dst_block_आकार, src_block_आकार, dst_block_आकार);

        अगर (padded_rows > 0 && tensor->ne[1] % padded_rows != 0) {
            योग += (padded_rows - tensor->ne[1] % padded_rows) * (tensor->nb[1] / src_block_आकार) * dst_block_आकार;
        }

        वापस योग;
    };

    माप_प्रकार nbytes = row_nbytes;
    चयन (tensor->type) {
        हाल GGML_TYPE_Q4_K:
            nbytes = remap_block_nbytes(माप(block_q4_K), माप(block_q4_1) * 8);
            अवरोध;
        हाल GGML_TYPE_Q6_K:
            nbytes = remap_block_nbytes(माप(block_q6_K), माप(block_q8_0) * 8, 32);
            अवरोध;
        हाल GGML_TYPE_Q8_0:
            nbytes = remap_block_nbytes(माप(block_q8_0), माप(block_q8_0), 32);
            अवरोध;
        हाल GGML_TYPE_Q2_K:
            nbytes = remap_block_nbytes(माप(block_q2_K), माप(spacemit_kernels::nrow_block_q2_k<1>));
            अवरोध;
        हाल GGML_TYPE_Q3_K:
            nbytes = remap_block_nbytes(माप(block_q3_K), माप(spacemit_kernels::nrow_block_q3_k<1>));
            अवरोध;
        हाल GGML_TYPE_MXFP4:
            nbytes = remap_block_nbytes(माप(block_mxfp4), माप(spacemit_kernels::nrow_block_mxfp4<1>));
            अवरोध;
        हाल GGML_TYPE_Q5_K:
            nbytes = remap_block_nbytes(माप(block_q5_K), माप(spacemit_kernels::nrow_block_q5_1<1>) * 8);
            अवरोध;
        हाल GGML_TYPE_Q5_1:
            nbytes = remap_block_nbytes(माप(block_q5_1), माप(spacemit_kernels::nrow_block_q5_1<1>));
            अवरोध;
        हाल GGML_TYPE_Q5_0:
            nbytes = remap_block_nbytes(माप(block_q5_0), माप(spacemit_kernels::nrow_block_q5_0<1>));
            अवरोध;
        शेष:
            nbytes = add_strided_nbytes(row_nbytes, 1, 1);
            अवरोध;
    }

    वापस nbytes;
}

namespace ggml::cpu::riscv64_spacemit {

श्रेणी extra_बफर_type : ggml::cpu::extra_बफर_type {
    bool supports_op(ggml_backend_dev_t, स्थिर ggml_tensor * op) override {
        चयन (op->op) {
            हाल GGML_OP_MUL_MAT:
                अगर (op->src[0]->बफर && (ggml_n_dims(op->src[0]) == 2) &&
                    op->src[0]->बफर->buft == ggml_backend_cpu_riscv64_spacemit_बफर_type() &&
                    ggml_riscv64_spacemit_get_optimal_repack_type(op->src[0])) {
                    अगर (op->src[1]->बफर && !ggml_backend_buft_is_host(op->src[1]->बफर->buft)) {
                        वापस false;
                    }
                    अगर (op->src[1]->type == GGML_TYPE_F32) {
                        वापस true;
                    }
                }
                अवरोध;
            हाल GGML_OP_MUL_MAT_ID:
                अगर (op->src[0]->बफर && (ggml_n_dims(op->src[0]) == 3) &&
                    op->src[0]->बफर->buft == ggml_backend_cpu_riscv64_spacemit_बफर_type() &&
                    ggml_riscv64_spacemit_get_optimal_repack_type(op->src[0])) {
                    अगर (op->src[1]->बफर && !ggml_backend_buft_is_host(op->src[1]->बफर->buft)) {
                        वापस false;
                    }
                    अगर (op->src[1]->type == GGML_TYPE_F32) {
                        वापस true;
                    }
                }
                अवरोध;
            शेष:
                // GGML_ABORT("fatal error");
                अवरोध;
        }
        वापस false;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(स्थिर ggml_tensor * op) override {
        चयन (op->op) {
            हाल GGML_OP_MUL_MAT:
            हाल GGML_OP_MUL_MAT_ID:
                अगर (op->src[0]->बफर && op->src[0]->बफर->buft == ggml_backend_cpu_riscv64_spacemit_बफर_type()) {
                    वापस (ggml::cpu::tensor_traits *) op->src[0]->extra;
                }
                अवरोध;
            हाल GGML_OP_NORM:
            हाल GGML_OP_RMS_NORM:
            हाल GGML_OP_ADD:
            हाल GGML_OP_SUB:
            हाल GGML_OP_MUL:
            हाल GGML_OP_DIV:
            हाल GGML_OP_FLASH_ATTN_EXT:
            हाल GGML_OP_CONT:
            हाल GGML_OP_CPY:
            हाल GGML_OP_REPEAT:
            हाल GGML_OP_योग_ROWS:
            हाल GGML_OP_GET_ROWS:
            हाल GGML_OP_CONCAT:
                // case GGML_OP_GATED_DELTA_NET:
                वापस (ggml::cpu::tensor_traits *) (&ggml::cpu::riscv64_spacemit::rvv_impl);
            शेष:
                // GGML_ABORT("fatal error");
                अवरोध;
        }

        वापस nullptr;
    }
};

}  // namespace ggml::cpu::riscv64_spacemit

ggml_backend_बफर_type_t ggml_backend_cpu_riscv64_spacemit_बफर_type(व्योम) {
    अटल ggml_backend_बफर_type ggml_backend_cpu_बफर_type_riscv64_spacemit = {
  /* .iface    = */
        {
         /* .get_name           = */ ggml_backend_cpu_riscv64_spacemit_बफर_type_get_name,
         /* .alloc_buffer       = */ ggml_backend_cpu_riscv64_spacemit_बफर_type_alloc_बफर,
         /* .alloc_buffer_n     = */ शून्य,
         /* .get_alignment      = */ ggml_backend_cpu_riscv64_spacemit_बफर_type_get_alignment,
         /* .get_max_size       = */ nullptr,
         /* .get_alloc_size     = */ ggml_backend_cpu_riscv64_spacemit_nbytes,
         /* .get_alloc_size_n   = */ शून्य,
         /* .is_host            = */ nullptr,
         },
 /* .device  = */
        ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
 /* .context = */
        नया ggml::cpu::riscv64_spacemit::extra_बफर_type(),
    };

    वापस &ggml_backend_cpu_बफर_type_riscv64_spacemit;
}

बाह्य "C" {
अटल पूर्णांक bind_ai_thread() {
    पूर्णांक  fd, bytes;
    अक्षर str[32];

    fd = खोलो("/proc/set_ai_thread", O_WRONLY);
    अगर (fd < 0) {
        GGML_LOG_ERROR("try open /proc/set_ai_thread failed\n");
        वापस -1;
    }

    snprintf(str, 16, "%d", 0);
    bytes = ग_लिखो(fd, str, म_माप(str));
    अगर (bytes < 0) {
        GGML_LOG_ERROR("try write /proc/set_ai_thread failed\n");
        बंद(fd);
        वापस -1;
    }

    बंद(fd);
    वापस 0;
}

व्योम ggml_backend_cpu_riscv64_spacemit_set_numa_thread_affinity(पूर्णांक thread_n) {
    पूर्णांक cpu_id = sched_getcpu();
    अगर (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2 &&
        !((1 << cpu_id) & ggml::cpu::riscv64_spacemit::global_spine_env_info.cpu_mask)) {
        GGML_PRINT_DEBUG("bind_ai_thread for thread %d, pid %d\n", thread_n, getpid());
        bind_ai_thread();
    }

    अगर (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_tcm &&
        ggml::cpu::riscv64_spacemit::tls_context.cpu_id == -1) {
        CPU_ZERO(&(ggml::cpu::riscv64_spacemit::tls_context.cpuset));
        pthread_t    main_thread     = pthread_self();
        स्थिर स्वतः & perfer_core_ids = ggml::cpu::riscv64_spacemit::global_spine_env_info.perfer_core_ids;
        अगर (thread_n < 0 || static_cast<माप_प्रकार>(thread_n) >= perfer_core_ids.size()) {
            GGML_ABORT("thread_n %d exceeds perfer_core_ids size %zu\n", thread_n, perfer_core_ids.size());
        }
        स्वतः perfer_cpu_id = perfer_core_ids[static_cast<माप_प्रकार>(thread_n)];
        CPU_SET(perfer_cpu_id, &(ggml::cpu::riscv64_spacemit::tls_context.cpuset));
        पूर्णांक s =
            pthread_setaffinity_np(main_thread, माप(cpu_set_t), &(ggml::cpu::riscv64_spacemit::tls_context.cpuset));
        अगर (s != 0) {
            GGML_ABORT("set thread affinity error for thread_n %d, cpu_id %d\n", thread_n, perfer_cpu_id);
        }

        पूर्णांक ai_cpu_id = perfer_cpu_id - ggml::cpu::riscv64_spacemit::global_spine_env_info.aicpu_id_offset;
        ggml::cpu::riscv64_spacemit::tls_context.cpu_id = ai_cpu_id;
        ggml::cpu::riscv64_spacemit::tls_context.tcm_बफर =
            ggml::cpu::riscv64_spacemit::spine_mem_pool_tcm_mem_get(ai_cpu_id);
        ggml::cpu::riscv64_spacemit::tls_context.tcm_बफर_आकार =
            ggml::cpu::riscv64_spacemit::global_spine_env_info.tcm_blk_आकार;
    }

    अगर (ggml::cpu::riscv64_spacemit::tls_context.tcm_बफर != nullptr) {
        व्योम * rt =
            ggml::cpu::riscv64_spacemit::spine_mem_pool_tcm_mem_wait(ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        अगर (rt == nullptr) {
            GGML_ABORT("wait tcm buffer failed for cpu_id: %d", ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        }
    }
}

व्योम ggml_backend_cpu_riscv64_spacemit_clear_numa_thread_affinity_threaded(पूर्णांक thread_n) {
    अगर (ggml::cpu::riscv64_spacemit::tls_context.tcm_बफर != nullptr) {
        स्वतः rt = ggml::cpu::riscv64_spacemit::spine_mem_pool_tcm_mem_release(
            ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        अगर (rt != 0) {
            GGML_ABORT("release tcm buffer failed for cpu_id: %d", ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        }
    }
}
}
