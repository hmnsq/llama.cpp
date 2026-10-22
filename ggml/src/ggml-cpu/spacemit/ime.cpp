#define GGML_COMMON_IMPL_CPP
#define GGML_COMMON_DECL_CPP

#ಅನ್ತರ್ಭಾವ "ime.h"

#ಅನ್ತರ್ಭಾವ "binary-ops.h"
#ಅನ್ತರ್ಭಾವ "common.h"
#ಅನ್ತರ್ಭಾವ "ggml-backend-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-common.h"
#ಅನ್ತರ್ಭಾವ "ggml-cpu.h"
#ಅನ್ತರ್ಭಾವ "ime_env.h"
#ಅನ್ತರ್ಭಾವ "ime_kernels.h"
#ಅನ್ತರ್ಭಾವ "ops.h"
#ಅನ್ತರ್ಭಾವ "repack.h"
#ಅನ್ತರ್ಭಾವ "rvv_kernels.h"
#ಅನ್ತರ್ಭಾವ "spine_mem_pool.h"
#ಅನ್ತರ್ಭಾವ "traits.h"
#ಅನ್ತರ್ಭಾವ "vec.h"

#ಅನ್ತರ್ಭಾವ <fcntl.h>
#ಅನ್ತರ್ಭಾವ <sys/mman.h>
#ಅನ್ತರ್ಭಾವ <unistd.h>

#ಅನ್ತರ್ಭಾವ <algorithm>
#ಅನ್ತರ್ಭಾವ <atomic>
#ಅನ್ತರ್ಭಾವ <cassert>
#ಅನ್ತರ್ಭಾವ <cerrno>
#ಅನ್ತರ್ಭಾವ <cmath>
#ಅನ್ತರ್ಭಾವ <cstdio>  // for GGML_ASSERT
#ಅನ್ತರ್ಭಾವ <stdexcept>
#ಅನ್ತರ್ಭಾವ <thread>
// clang-format off
#ಯದಿ defined(__riscv)

#ಯದಿ !defined(__riscv_v) || !defined(__riscv_v_intrinsic)
#error "riscv v extension or v_intrinsic not enabled"
#ಅನ್ಯಥಾ
#ಅನ್ತರ್ಭಾವ <riscv_vector.h>
#endif

#ಯದಿ !defined(__riscv_zfh) || !defined(__riscv_zvfh)
#error "riscv zfh extension not enabled, GGML_RV_ZFH and GGML_RV_ZVFH must be defined to 1"
#endif

#ಯದಿ !defined(__riscv_zba)
#error "riscv zba extension not enabled, GGML_RV_ZBA must be defined to 1"
#endif

#ಯದಿ defined(RISCV64_SPACEMIT_IME1) || defined(RISCV64_SPACEMIT_IME2)
#ಅನ್ಯಥಾ
#error "RISCV64_SPACEMIT_IME1 or RISCV64_SPACEMIT_IME2 not defined"
#endif

#ಅನ್ಯಥಾ

#error "riscv not enabled in this build"

#endif

#ಯದಿ defined(__GNUC__)
#pragma GCC diagnostic ignored "-Woverlength-strings"
#pragma GCC diagnostic ignored "-Wcast-qual"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

// clang-format on

ಬಾಹ್ಯ "C" {
ಬಾಹ್ಯ ಶೂನ್ಯ ggml_threadpool_chunk_set(ಸಂರಚನಾ ggml_threadpool * tp, ಪೂರ್ಣಾಙ್ಕ value);
ಬಾಹ್ಯ ಪೂರ್ಣಾಙ್ಕ  ggml_threadpool_chunk_add(ಸಂರಚನಾ ggml_threadpool * tp, ಪೂರ್ಣಾಙ್ಕ value);
}

ನಾಮಾಕಾಶ ggml::cpu::riscv64_spacemit {

ಸಂರಚನಾ TLSContext {
    ಪೂರ್ಣಾಙ್ಕ       cpu_id{ -1 };
    cpu_set_t cpuset;
    ಶೂನ್ಯ *    tcm_buffer{ nullptr };
    size_t    tcm_buffer_size{ 0 };
};

thread_local TLSContext tls_context;

template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS> constexpr size_t get_repacked_block_type_size() {
    ಯದಿ constexpr (std::is_same_v<BLOC_TYPE, block_q6_K> || std::is_same_v<BLOC_TYPE, block_q8_0>) {
        ಪ್ರತ್ಯಾಗಮ ಪರಿಮಾಣ(block_q8_0);
    } ಅನ್ಯಥಾ ಯದಿ constexpr (std::is_same_v<BLOC_TYPE, block_q4_0>) {
        ಪ್ರತ್ಯಾಗಮ ಪರಿಮಾಣ(block_q4_0) * INTER_SIZE / QK4_0;
    } ಅನ್ಯಥಾ ಯದಿ constexpr (std::is_same_v<BLOC_TYPE, block_q4_1> || std::is_same_v<BLOC_TYPE, block_q4_K>) {
        ಪ್ರತ್ಯಾಗಮ (ಪರಿಮಾಣ(block_q4_0) + ಪರಿಮಾಣ(uint8_t)) * INTER_SIZE / QK4_1;
    } ಅನ್ಯಥಾ ಯದಿ constexpr (std::is_same_v<BLOC_TYPE, block_q2_K>) {
        ಪ್ರತ್ಯಾಗಮ ಪರಿಮಾಣ(spacemit_kernels::nrow_block_q2_k<1>);
    } ಅನ್ಯಥಾ ಯದಿ constexpr (std::is_same_v<BLOC_TYPE, block_q3_K>) {
        ಪ್ರತ್ಯಾಗಮ ಪರಿಮಾಣ(spacemit_kernels::nrow_block_q3_k<1>);
    } ಅನ್ಯಥಾ ಯದಿ constexpr (std::is_same_v<BLOC_TYPE, block_mxfp4>) {
        ಪ್ರತ್ಯಾಗಮ ಪರಿಮಾಣ(spacemit_kernels::nrow_block_mxfp4<1>);
    } ಅನ್ಯಥಾ ಯದಿ constexpr (std::is_same_v<BLOC_TYPE, block_q5_1> || std::is_same_v<BLOC_TYPE, block_q5_K>) {
        ಪ್ರತ್ಯಾಗಮ ಪರಿಮಾಣ(spacemit_kernels::nrow_block_q5_1<1>);
    } ಅನ್ಯಥಾ ಯದಿ constexpr (std::is_same_v<BLOC_TYPE, block_q5_0>) {
        ಪ್ರತ್ಯಾಗಮ ಪರಿಮಾಣ(spacemit_kernels::nrow_block_q5_0<1>);
    } ಅನ್ಯಥಾ {
        ಪ್ರತಿಜ್ಞಾ(ಅಸತ್ಯ);
        ಪ್ರತ್ಯಾಗಮ 0;
    }
}

template <typename BLOC_TYPE> constexpr ತಾರ್ಕಿಕ block_type_has_zp() {
    ಯದಿ constexpr (std::is_same_v<BLOC_TYPE, block_q6_K> || std::is_same_v<BLOC_TYPE, block_q8_0> ||
                  std::is_same_v<BLOC_TYPE, block_q3_K> || std::is_same_v<BLOC_TYPE, block_q4_0> ||
                  std::is_same_v<BLOC_TYPE, block_mxfp4> || std::is_same_v<BLOC_TYPE, block_q5_0>) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    } ಅನ್ಯಥಾ ಯದಿ constexpr (std::is_same_v<BLOC_TYPE, block_q4_1> || std::is_same_v<BLOC_TYPE, block_q4_K> ||
                         std::is_same_v<BLOC_TYPE, block_q2_K> || std::is_same_v<BLOC_TYPE, block_q5_1> ||
                         std::is_same_v<BLOC_TYPE, block_q5_K>) {
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    } ಅನ್ಯಥಾ {
        ಪ್ರತಿಜ್ಞಾ(ಅಸತ್ಯ);
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
}

ವರ್ಗ tensor_traits_base : ಸಾರ್ವಜನಿಕ ggml::cpu::tensor_traits {
  ಸಾರ್ವಜನಿಕ:
    ಆಭಾಸ ಪೂರ್ಣಾಙ್ಕ repack(ggml_tensor * t, ಧ್ರುವ ಶೂನ್ಯ * data, size_t data_size) = 0;
};

template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS> ವರ್ಗ tensor_traits : ಸಾರ್ವಜನಿಕ tensor_traits_base {
    ತಾರ್ಕಿಕ work_size(ಪೂರ್ಣಾಙ್ಕ /* n_threads */, ಧ್ರುವ ggml_tensor * op, size_t & size) override {
        ವಿಕಲ್ಪ (op->op) {
            ಪಕ್ಷ GGML_OP_MUL_MAT:
                {
                    int64_t src1_nelements = ggml_nelements(op->src[1]);

                    ಯದಿ constexpr (std::is_same_v<BLOC_TYPE, block_q2_K> || std::is_same_v<BLOC_TYPE, block_q3_K>) {
                        size =
                            spacemit_kernels::div_round_up(src1_nelements, QK_K) * spacemit_kernels::q8k_blk_size(QK_K);
                    } ಅನ್ಯಥಾ ಯದಿ constexpr (INTER_SIZE == QK4_0) {
                        size = spacemit_kernels::div_round_up(src1_nelements, QK4_0) *
                               spacemit_kernels::q8_blk_size(QK4_0, ಸತ್ಯ);
                    } ಅನ್ಯಥಾ ಯದಿ constexpr (INTER_SIZE == 256) {
                        size = spacemit_kernels::div_round_up(src1_nelements, 256) *
                               spacemit_kernels::q8_hp_blk_size(256, ಸತ್ಯ, ಸತ್ಯ);
                    } ಅನ್ಯಥಾ {
                        GGML_ABORT("unsupported block type");
                    }

                    size = GGML_PAD(size, ಪರಿಮಾಣ(int64_t));

                    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
                }
            ಪಕ್ಷ GGML_OP_MUL_MAT_ID:
                {
                    int64_t src1_nelements = ggml_nelements(op->src[1]);

                    ಯದಿ constexpr (std::is_same_v<BLOC_TYPE, block_q2_K> || std::is_same_v<BLOC_TYPE, block_q3_K>) {
                        size =
                            spacemit_kernels::div_round_up(src1_nelements, QK_K) * spacemit_kernels::q8k_blk_size(QK_K);
                    } ಅನ್ಯಥಾ ಯದಿ constexpr (INTER_SIZE == QK4_0) {
                        size = spacemit_kernels::div_round_up(src1_nelements, QK4_0) *
                               spacemit_kernels::q8_blk_size(QK4_0, ಸತ್ಯ);
                    } ಅನ್ಯಥಾ ಯದಿ constexpr (INTER_SIZE == 256) {
                        size = spacemit_kernels::div_round_up(src1_nelements, 256) *
                               spacemit_kernels::q8_hp_blk_size(256, ಸತ್ಯ, ಸತ್ಯ);
                    } ಅನ್ಯಥಾ {
                        GGML_ABORT("unsupported block type");
                    }

                    size = GGML_PAD(size, ಪರಿಮಾಣ(int64_t));

                    ಧ್ರುವ int64_t ne02 = op->src[0]->ne[2];  // n_as, n_expert
                    ಧ್ರುವ int64_t ne12 = op->src[1]->ne[2];  // n_tokens

                    ಧ್ರುವ size_t sizeof_mmid_row_mapping = ಪರಿಮಾಣ(int64_t);
                    size += sizeof_mmid_row_mapping * ne02 * (ne12 + 1) + (ne02 + 1) * ಪರಿಮಾಣ(int64_t);

                    size = GGML_PAD(size, ಪರಿಮಾಣ(int64_t));

                    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
                }
            ಸ್ವಭಾವ:
                // GGML_ABORT("fatal error");
                ಭಙ್ಗ;
        }
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ತಾರ್ಕಿಕ compute_forward(ggml_compute_params * params, ggml_tensor * op) override {
        ವಿಕಲ್ಪ (op->op) {
            ಪಕ್ಷ GGML_OP_MUL_MAT:
                ವಿಕಲ್ಪ (op->src[0]->type) {
                    ಪಕ್ಷ GGML_TYPE_Q2_K:
                    ಪಕ್ಷ GGML_TYPE_Q3_K:
                    ಪಕ್ಷ GGML_TYPE_Q4_0:
                    ಪಕ್ಷ GGML_TYPE_Q4_1:
                    ಪಕ್ಷ GGML_TYPE_Q4_K:
                    ಪಕ್ಷ GGML_TYPE_Q6_K:
                    ಪಕ್ಷ GGML_TYPE_Q8_0:
                    ಪಕ್ಷ GGML_TYPE_Q5_0:
                    ಪಕ್ಷ GGML_TYPE_Q5_1:
                    ಪಕ್ಷ GGML_TYPE_Q5_K:
                        //case GGML_TYPE_MXFP4:
                        forward_mul_mat(params, op);
                        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
                    ಸ್ವಭಾವ:
                        // GGML_ABORT("fatal error: unsupported type for src0 in MUL_MAT");
                        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                }
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_OP_MUL_MAT_ID:
                ವಿಕಲ್ಪ (op->src[0]->type) {
                    ಪಕ್ಷ GGML_TYPE_Q2_K:
                    ಪಕ್ಷ GGML_TYPE_Q3_K:
                    ಪಕ್ಷ GGML_TYPE_Q4_0:
                    ಪಕ್ಷ GGML_TYPE_Q4_1:
                    ಪಕ್ಷ GGML_TYPE_Q4_K:
                    ಪಕ್ಷ GGML_TYPE_Q6_K:
                    ಪಕ್ಷ GGML_TYPE_Q8_0:
                    ಪಕ್ಷ GGML_TYPE_Q5_0:
                    ಪಕ್ಷ GGML_TYPE_Q5_1:
                    ಪಕ್ಷ GGML_TYPE_Q5_K:
                        //case GGML_TYPE_MXFP4:
                        forward_mul_mat_id(params, op);
                        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
                    ಸ್ವಭಾವ:
                        // GGML_ABORT("fatal error: unsupported type for src0 in MUL_MAT_ID");
                        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                }
                ಭಙ್ಗ;
            ಸ್ವಭಾವ:
                // GGML_ABORT("fatal error");
                ಭಙ್ಗ;
        }
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಶೂನ್ಯ forward_mul_mat(ggml_compute_params * params, ggml_tensor * op) {
        constexpr size_t a_blk_len = INTER_SIZE;
        constexpr size_t b_blk_len = INTER_SIZE;

        ಧ್ರುವ ggml_tensor * src0 = op->src[0];
        ಧ್ರುವ ggml_tensor * src1 = op->src[1];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS

        ಪೂರ್ಣಾಙ್ಕ ith = params->ith;
        ಪೂರ್ಣಾಙ್ಕ nth = params->nth;

        [[maybe_unused]] ಧ್ರುವ ಗಣನಾ ggml_type type = src0->type;

        ಶೂನ್ಯ *        w_data  = (ಶೂನ್ಯ *) src0->data;
        ಧ್ರುವ ಭಿನ್ನ * feature = (ಧ್ರುವ ಭಿನ್ನ *) src1->data;
        ಭಿನ್ನ *       output  = (ಭಿನ್ನ *) dst->data;

        ಧ್ರುವ int64_t gemm_m = ne11 * ne12 * ne13;
        ಧ್ರುವ int64_t gemm_k = ne10;
        ಧ್ರುವ int64_t gemm_n = ne01;

        spacemit_kernels::quantize_a_row_def       quantize_a_row_i8;
        spacemit_kernels::quantize_a_row_def       quantize_a_4row_i8;
        spacemit_kernels::gemm_kernel_quantize_def gemm_kernel;
        ತಾರ್ಕಿಕ                                       set_kernel_impl = ಅಸತ್ಯ;

        int64_t block_stride_a = spacemit_kernels::q8_blk_size(a_blk_len);

#ಯದಿ defined(RISCV64_SPACEMIT_IME2)
        ಯದಿ (!set_kernel_impl && (global_spine_env_info.use_ime2)) {
            quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8;
            quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8;
            block_stride_a     = spacemit_kernels::q8_blk_size(a_blk_len, ಸತ್ಯ);

            ಯದಿ constexpr (std::is_same_v<BLOC_TYPE, block_q6_K> || std::is_same_v<BLOC_TYPE, block_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i8;
                set_kernel_impl = ಸತ್ಯ;
            } ಅನ್ಯಥಾ ಯದಿ constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> || std::is_same_v<BLOC_TYPE, block_q4_1> ||
                                 std::is_same_v<BLOC_TYPE, block_q4_K>) {
                ಯದಿ constexpr (INTER_SIZE == 256) {
                    gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i4_hp;
                    quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8_hp;
                    quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8_hp;
                    block_stride_a     = spacemit_kernels::q8_hp_blk_size(a_blk_len, ಸತ್ಯ, ಸತ್ಯ);
                    set_kernel_impl    = ಸತ್ಯ;
                } ಅನ್ಯಥಾ {
                    gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i4;
                    quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8;
                    quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8;
                    block_stride_a     = spacemit_kernels::q8_blk_size(a_blk_len, ಸತ್ಯ);
                    set_kernel_impl    = ಸತ್ಯ;
                }
            } ಅನ್ಯಥಾ ಯದಿ constexpr (std::is_same_v<BLOC_TYPE, block_q2_K>) {
                quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8k;
                quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8k;
                block_stride_a     = spacemit_kernels::q8k_blk_size(a_blk_len);

                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i2k;
                set_kernel_impl = ಸತ್ಯ;
            } ಅನ್ಯಥಾ ಯದಿ constexpr (std::is_same_v<BLOC_TYPE, block_q3_K>) {
                quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8k;
                quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8k;
                block_stride_a     = spacemit_kernels::q8k_blk_size(a_blk_len);

                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i3k;
                set_kernel_impl = ಸತ್ಯ;
            } ಅನ್ಯಥಾ ಯದಿ constexpr (std::is_same_v<BLOC_TYPE, block_mxfp4>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8mxfp4;
                set_kernel_impl = ಸತ್ಯ;
            } ಅನ್ಯಥಾ ಯದಿ constexpr (std::is_same_v<BLOC_TYPE, block_q5_1> || std::is_same_v<BLOC_TYPE, block_q5_K> ||
                                 std::is_same_v<BLOC_TYPE, block_q5_0>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i5;
                set_kernel_impl = ಸತ್ಯ;
            }
        }
#endif

#ಯದಿ defined(RISCV64_SPACEMIT_IME1)
        ಯದಿ (!set_kernel_impl && (global_spine_env_info.use_ime1)) {
            quantize_a_row_i8  = spacemit_kernels::ime1::quantize_a_row_i8;
            quantize_a_4row_i8 = spacemit_kernels::ime1::quantize_a_4row_i8;

            ಯದಿ constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> || std::is_same_v<BLOC_TYPE, block_q4_1> ||
                          std::is_same_v<BLOC_TYPE, block_q4_K>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i4;
                set_kernel_impl = ಸತ್ಯ;
            } ಅನ್ಯಥಾ ಯದಿ constexpr (std::is_same_v<BLOC_TYPE, block_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i8;
                set_kernel_impl = ಸತ್ಯ;
            }
        }
#endif
        ಯದಿ (!set_kernel_impl) {
            GGML_ABORT("no kernel implementation found for the block type");
        }

        ಧ್ರುವ int64_t a_k_blks = spacemit_kernels::div_round_up(gemm_k, a_blk_len);
        ಧ್ರುವ int64_t b_k_blks = spacemit_kernels::div_round_up(gemm_k, b_blk_len);

        ಧ್ರುವ int64_t row_stride_a        = a_k_blks * block_stride_a;
        ಧ್ರುವ int64_t gemm_workspace_size = GGML_PAD(gemm_m * row_stride_a, alignof(int64_t));

        ಯದಿ (ith == 0 && params->wsize < gemm_workspace_size) {
            GGML_ABORT("wsize less than gemm_workspace_size");
        }

        uintptr_t ws_ptr = reinterpret_cast<uintptr_t>(params->wdata);

        ಶೂನ್ಯ *        tcm_buffer      = ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer;
        ಧ್ರುವ int64_t tcm_buffer_size = ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer_size;

        auto * quant_a_buffer = reinterpret_cast<uint8_t *>(ws_ptr);

        constexpr int64_t row_align = 4;
        ಧ್ರುವ int64_t     row_blks  = spacemit_kernels::div_round_up(gemm_m, row_align);

        ಧ್ರುವ int64_t row_stride_b      = b_k_blks * get_repacked_block_type_size<BLOC_TYPE, INTER_SIZE, NB_COLS>();
        ಧ್ರುವ int64_t per_mb_rows_wsize = row_align * row_stride_a;
        ಧ್ರುವ int64_t per_nb_cols_wsize = NB_COLS * row_stride_b;

        ಧ್ರುವ int64_t barrier_idx = static_cast<int64_t>(ith / 2);

        GGML_ASSERT(global_spine_env_info.init_barrier != nullptr);
        GGML_ASSERT(barrier_idx < spine_init_barrier_ಎಣಿಕೆ);
        spine_barrier_t * cur_barrier = &global_spine_env_info.init_barrier[barrier_idx];

        ಯದಿ (gemm_m == 1) {
            ಪೂರ್ಣಾಙ್ಕ task_per_thread = spacemit_kernels::div_round_up(a_k_blks, nth);
            ಪೂರ್ಣಾಙ್ಕ a_blk_start     = ith * task_per_thread;
            ಪೂರ್ಣಾಙ್ಕ a_blk_end       = std::min(a_blk_start + task_per_thread, (ಪೂರ್ಣಾಙ್ಕ) a_k_blks);
            ಯದಿ (a_blk_start < a_blk_end) {
                quantize_a_row_i8(a_blk_len, feature + a_blk_start * a_blk_len, (a_blk_end - a_blk_start) * a_blk_len,
                                  quant_a_buffer + a_blk_start * block_stride_a);
            }
        } ಅನ್ಯಥಾ {
            ಪೂರ್ಣಾಙ್ಕ task_per_thread = spacemit_kernels::div_round_up(row_blks, nth);
            ಪೂರ್ಣಾಙ್ಕ m_row_blk_start = ith * task_per_thread;
            ಪೂರ್ಣಾಙ್ಕ m_row_blk_end   = std::min(m_row_blk_start + task_per_thread, (ಪೂರ್ಣಾಙ್ಕ) row_blks);
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ m_row_blk = m_row_blk_start; m_row_blk < m_row_blk_end; m_row_blk++) {
                ಪೂರ್ಣಾಙ್ಕ m_idx             = m_row_blk * row_align;
                ಪೂರ್ಣಾಙ್ಕ rows_tobe_handled = (gemm_m - m_idx) > row_align ? row_align : (gemm_m - m_idx);

                ಯದಿ (rows_tobe_handled == row_align && quantize_a_4row_i8 != nullptr) {
                    ಧ್ರುವ ಭಿನ್ನ * a_row_ptr       = feature + m_idx * gemm_k;
                    auto *        quant_a_row_ptr = quant_a_buffer + m_idx * row_stride_a;
                    quantize_a_4row_i8(a_blk_len, a_row_ptr, gemm_k, quant_a_row_ptr);
                } ಅನ್ಯಥಾ {
                    ಯಾವತ್ (rows_tobe_handled) {
                        ಧ್ರುವ ಭಿನ್ನ * a_row_ptr       = feature + m_idx * gemm_k;
                        auto *        quant_a_row_ptr = quant_a_buffer + m_idx * row_stride_a;
                        quantize_a_row_i8(a_blk_len, a_row_ptr, gemm_k, quant_a_row_ptr);
                        rows_tobe_handled -= 1;
                        m_idx += 1;
                    }
                }
            }
        }

        ggml_barrier(params->threadpool);

        ಧ್ರುವ int64_t gemm_m_stride     = gemm_n / gemm_m > 64 ? gemm_m : 16;
        ಧ್ರುವ int64_t gemm_m_blocked    = spacemit_kernels::div_round_up(gemm_m, gemm_m_stride);
        ಧ್ರುವ int64_t max_gemm_n_stride = spacemit_kernels::div_round_up(gemm_n * gemm_m_blocked, nth);

        int64_t gemm_n_stride = gemm_n;
        ಯದಿ (max_gemm_n_stride < gemm_n) {
            gemm_n_stride =
                std::min(gemm_n_stride, spacemit_kernels::div_round_up(max_gemm_n_stride, NB_COLS) * NB_COLS);
        }

        ಯದಿ (gemm_n_stride == gemm_n && tcm_buffer != nullptr && per_mb_rows_wsize <= tcm_buffer_size) {
            ಪ್ರತಿ (int64_t m_start = ith * row_align; m_start < gemm_m; m_start += row_align * nth) {
                uint8_t * b_col    = reinterpret_cast<uint8_t *>(w_data);
                uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? b_col : nullptr;

                int64_t m_row_real = std::min(gemm_m - m_start, row_align);

                spacemit_kernels::rvv::memcpy1d(tcm_buffer, quant_a_buffer + m_start * row_stride_a,
                                                m_row_real * row_stride_a);

                int64_t n_blk_real = 0;
                ಪ್ರತಿ (int64_t ni = 0; ni < gemm_n; ni += n_blk_real, b_col += n_blk_real * row_stride_b) {
                    n_blk_real = std::min(gemm_n - ni, (int64_t) NB_COLS);

                    uint8_t * a_row_ptr = (uint8_t *) tcm_buffer;
                    ಭಿನ್ನ *   c_blk     = output + m_start * gemm_n + ni;

                    int32_t rows_remaining = m_row_real;

                    ಯಾವತ್ (rows_remaining > 0) {
                        auto rows_handled = gemm_kernel(b_blk_len, a_row_ptr, b_col, b_col_zp, c_blk, rows_remaining,
                                                        n_blk_real, b_k_blks, gemm_n);

                        c_blk += rows_handled * gemm_n;
                        a_row_ptr += rows_handled * row_stride_a;

                        rows_remaining -= rows_handled;
                    }
                }
            }
        } ಅನ್ಯಥಾ ಯದಿ (tcm_buffer != nullptr && per_nb_cols_wsize <= tcm_buffer_size) {
            uint8_t * a_row = quant_a_buffer;
            uint8_t * b_col = reinterpret_cast<uint8_t *>(tcm_buffer);
            ಯದಿ ((gemm_workspace_size + per_nb_cols_wsize) <= tcm_buffer_size) {
                a_row = (uint8_t *) tcm_buffer;
                b_col = reinterpret_cast<uint8_t *>(tcm_buffer) + gemm_workspace_size;
            }
            uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? b_col : nullptr;

            int64_t ni      = ith * NB_COLS;
            int64_t nb_real = std::min(gemm_n - ni, NB_COLS);

            ಯದಿ (ith % 2 == 0 && nb_real > 0) {
                spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(w_data) + ni * row_stride_b,
                                                nb_real * row_stride_b);
                ಯದಿ (a_row != quant_a_buffer) {
                    spacemit_kernels::rvv::memcpy1d(a_row, quant_a_buffer, gemm_workspace_size);
                }
            }

            spine_barrier_wait(cur_barrier);

            ಯದಿ (ith % 2 != 0 && nb_real > 0) {
                ಯದಿ (a_row != quant_a_buffer) {
                    spacemit_kernels::rvv::memcpy1d(a_row, quant_a_buffer, gemm_workspace_size);
                }
                spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(w_data) + ni * row_stride_b,
                                                nb_real * row_stride_b);
            }

            ಪ್ರತಿ (; ni < gemm_n; ni += NB_COLS * nth) {
                int64_t rows_remaining = gemm_m;
                ಭಿನ್ನ * c_blk          = output + ni;
                auto *  a_row_cur      = a_row;

                ಯದಿ (ith % 2 != 0) {
                    spine_barrier_wait(cur_barrier);
                }

                ಯಾವತ್ (rows_remaining > 0) {
                    auto rows_handled = gemm_kernel(b_blk_len, a_row_cur, b_col, b_col_zp, c_blk, rows_remaining,
                                                    nb_real, b_k_blks, gemm_n);

                    c_blk += rows_handled * gemm_n;
                    a_row_cur += rows_handled * row_stride_a;

                    rows_remaining -= rows_handled;
                }

                ಯದಿ (ith % 2 == 0) {
                    spine_barrier_wait(cur_barrier);
                }

                ಧ್ರುವ int64_t next_ni = ni + NB_COLS * nth;
                ಯದಿ (next_ni < gemm_n) {
                    nb_real = std::min(gemm_n - next_ni, NB_COLS);
                    spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(w_data) + next_ni * row_stride_b,
                                                    nb_real * row_stride_b);
                }
            }
        } ಅನ್ಯಥಾ {
            ಧ್ರುವ int64_t task_ಎಣಿಕೆ_m = spacemit_kernels::div_round_up(gemm_m, gemm_m_stride);
            ಧ್ರುವ int64_t task_ಎಣಿಕೆ_n = spacemit_kernels::div_round_up(gemm_n, gemm_n_stride);

            int64_t task_ಎಣಿಕೆ      = task_ಎಣಿಕೆ_m * task_ಎಣಿಕೆ_n;
            int64_t task_per_thread = (task_ಎಣಿಕೆ + nth - 1) / nth;
            int64_t start           = ith * task_per_thread;
            int64_t end             = std::min((ith + 1) * task_per_thread, task_ಎಣಿಕೆ);
            ಪ್ರತಿ (int64_t compute_idx = start; compute_idx < end; compute_idx++) {
                ಧ್ರುವ auto tid_n = compute_idx / task_ಎಣಿಕೆ_m;
                ಧ್ರುವ auto tid_m = compute_idx % task_ಎಣಿಕೆ_m;

                ಧ್ರುವ int64_t m_start = tid_m * gemm_m_stride;
                ಧ್ರುವ int64_t m_ಎಣಿಕೆ = std::min(gemm_m - m_start, (int64_t) gemm_m_stride);

                ಧ್ರುವ int64_t n_start = tid_n * gemm_n_stride;
                ಧ್ರುವ int64_t n_ಎಣಿಕೆ = std::min(gemm_n - n_start, (int64_t) gemm_n_stride);

                ಧ್ರುವ int64_t n_blk = m_ಎಣಿಕೆ == 1 ? n_ಎಣಿಕೆ : NB_COLS;

                uint8_t * b_col    = reinterpret_cast<uint8_t *>(w_data) + n_start * row_stride_b;
                uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? b_col : nullptr;

                int64_t n_blk_real = 0;
                ಪ್ರತಿ (int64_t ni = 0; ni < n_ಎಣಿಕೆ; ni += n_blk_real, b_col += n_blk_real * row_stride_b) {
                    n_blk_real = std::min(n_ಎಣಿಕೆ - ni, n_blk);

                    uint8_t * a_row = quant_a_buffer + m_start * row_stride_a;

                    ಭಿನ್ನ * c_blk = output + m_start * gemm_n + n_start + ni;

                    int64_t rows_remaining = m_ಎಣಿಕೆ;

                    uint8_t * b_col_cur    = b_col;
                    uint8_t * b_col_zp_cur = b_col_zp;

                    ಯಾವತ್ (rows_remaining > 0) {
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

    ಶೂನ್ಯ forward_mul_mat_id(ggml_compute_params * params, ggml_tensor * op) {
        constexpr size_t a_blk_len = INTER_SIZE;
        constexpr size_t b_blk_len = INTER_SIZE;

        ಧ್ರುವ ggml_tensor * src0 = op->src[0];
        ಧ್ರುವ ggml_tensor * src1 = op->src[1];
        ಧ್ರುವ ggml_tensor * ids  = op->src[2];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS

        ಪೂರ್ಣಾಙ್ಕ ith = params->ith;
        ಪೂರ್ಣಾಙ್ಕ nth = params->nth;

        // row groups
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_ids = ids->ne[0];  // n_expert_used
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_as  = ne02;        // n_expert

        ಸಂರಚನಾ mmid_row_mapping {
            int32_t i1;
            int32_t i2;
        };

        spacemit_kernels::quantize_a_row_def           quantize_a_row_i8;
        spacemit_kernels::gemm_kernel_quantize_def     gemm_kernel;
        spacemit_kernels::moe_gemm_kernel_quantize_def moe_gemm_kernel_m2;
        ತಾರ್ಕಿಕ                                           set_kernel_impl = ಅಸತ್ಯ;
        size_t                                         block_stride_a  = spacemit_kernels::q8_blk_size(QK4_0);

#ಯದಿ defined(RISCV64_SPACEMIT_IME2)
        ಯದಿ (!set_kernel_impl && (global_spine_env_info.use_ime2)) {
            quantize_a_row_i8 = spacemit_kernels::rvv::quantize_a_row_i8;
            block_stride_a    = spacemit_kernels::q8_blk_size(QK4_0, ಸತ್ಯ);

            ಯದಿ constexpr (std::is_same_v<BLOC_TYPE, block_q6_K> || std::is_same_v<BLOC_TYPE, block_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i8;
                set_kernel_impl = ಸತ್ಯ;
            } ಅನ್ಯಥಾ ಯದಿ constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> || std::is_same_v<BLOC_TYPE, block_q4_1> ||
                                 std::is_same_v<BLOC_TYPE, block_q4_K>) {
                ಯದಿ constexpr (INTER_SIZE == 256) {
                    gemm_kernel       = spacemit_kernels::ime2::gemm_kernel_i8i4_hp;
                    quantize_a_row_i8 = spacemit_kernels::rvv::quantize_a_row_i8_hp;
                    block_stride_a    = spacemit_kernels::q8_hp_blk_size(a_blk_len, ಸತ್ಯ, ಸತ್ಯ);
                    set_kernel_impl   = ಸತ್ಯ;
                } ಅನ್ಯಥಾ {
                    gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i4;
                    moe_gemm_kernel_m2 = spacemit_kernels::ime2::moe_m2_gemm_kernel_i8i4;
                    quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8;
                    block_stride_a     = spacemit_kernels::q8_blk_size(a_blk_len, ಸತ್ಯ);
                    set_kernel_impl    = ಸತ್ಯ;
                }
            } ಅನ್ಯಥಾ ಯದಿ constexpr (std::is_same_v<BLOC_TYPE, block_q2_K>) {
                quantize_a_row_i8 = spacemit_kernels::rvv::quantize_a_row_i8k;
                block_stride_a    = spacemit_kernels::q8k_blk_size(a_blk_len);
                gemm_kernel       = spacemit_kernels::ime2::gemm_kernel_i8i2k;
                set_kernel_impl   = ಸತ್ಯ;
            } ಅನ್ಯಥಾ ಯದಿ constexpr (std::is_same_v<BLOC_TYPE, block_q3_K>) {
                quantize_a_row_i8 = spacemit_kernels::rvv::quantize_a_row_i8k;
                block_stride_a    = spacemit_kernels::q8k_blk_size(a_blk_len);
                gemm_kernel       = spacemit_kernels::ime2::gemm_kernel_i8i3k;
                set_kernel_impl   = ಸತ್ಯ;
            } ಅನ್ಯಥಾ ಯದಿ constexpr (std::is_same_v<BLOC_TYPE, block_mxfp4>) {
                gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8mxfp4;
                moe_gemm_kernel_m2 = spacemit_kernels::ime2::moe_m2_gemm_kernel_i8mxfp4;
                set_kernel_impl    = ಸತ್ಯ;
            } ಅನ್ಯಥಾ ಯದಿ constexpr (std::is_same_v<BLOC_TYPE, block_q5_1> || std::is_same_v<BLOC_TYPE, block_q5_K> ||
                                 std::is_same_v<BLOC_TYPE, block_q5_0>) {
                gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i5;
                moe_gemm_kernel_m2 = spacemit_kernels::ime2::moe_m2_gemm_kernel_i8i5;
                set_kernel_impl    = ಸತ್ಯ;
            }
        }
#endif

#ಯದಿ defined(RISCV64_SPACEMIT_IME1)
        ಯದಿ (!set_kernel_impl && (global_spine_env_info.use_ime1)) {
            quantize_a_row_i8 = spacemit_kernels::ime1::quantize_a_row_i8;

            ಯದಿ constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> || std::is_same_v<BLOC_TYPE, block_q4_1> ||
                          std::is_same_v<BLOC_TYPE, block_q4_K>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i4;
                set_kernel_impl = ಸತ್ಯ;
            } ಅನ್ಯಥಾ ಯದಿ constexpr (std::is_same_v<BLOC_TYPE, block_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i8;
                set_kernel_impl = ಸತ್ಯ;
            }
        }
#endif
        ಯದಿ (!set_kernel_impl) {
            GGML_ABORT("no kernel implementation found for the block type");
        }

        ಧ್ರುವ size_t a_k_blks = spacemit_kernels::div_round_up(ne10, a_blk_len);
        ಧ್ರುವ size_t b_k_blks = spacemit_kernels::div_round_up(ne10, b_blk_len);

        ಧ್ರುವ size_t nbw1                = a_k_blks * block_stride_a;
        ಧ್ರುವ size_t nbw2                = ne11 * nbw1;
        ಧ್ರುವ size_t nbw3                = nbw2 * ne12;
        ಧ್ರುವ size_t gemm_workspace_size = GGML_PAD(nbw3, alignof(int64_t));

        ಧ್ರುವ uintptr_t ws_ptr         = reinterpret_cast<uintptr_t>(params->wdata);
        auto *          quant_a_buffer = reinterpret_cast<uint8_t *>(ws_ptr);

        ಯದಿ (ne11 == 1) {
            ಪ್ರತಿ (int64_t ii = ith; ii < ne12 * a_k_blks; ii += nth) {
                int64_t i12       = ii / a_k_blks;
                int64_t ak_blk_id = ii % a_k_blks;
                quantize_a_row_i8(a_blk_len, (ಭಿನ್ನ *) ((ವರ್ಣ *) src1->data + i12 * nb12) + ak_blk_id * a_blk_len,
                                  a_blk_len, quant_a_buffer + i12 * nbw2 + ak_blk_id * block_stride_a);
            }
        } ಅನ್ಯಥಾ {
            ಪ್ರತಿ (int64_t ii = ith; ii < ne12 * ne11; ii += nth) {
                int64_t i12 = ii / ne11;
                int64_t i11 = ii % ne11;
                quantize_a_row_i8(a_blk_len, (ಭಿನ್ನ *) ((ವರ್ಣ *) src1->data + i12 * nb12 + i11 * nb11), ne10,
                                  quant_a_buffer + i12 * nbw2 + i11 * nbw1);
            }
        }

#define MMID_MATRIX_ROW(row_id, i1) matrix_rows[(row_id) *ne12 + (i1)]

        int64_t *          matrix_row_counts       = (int64_t *) (ws_ptr + gemm_workspace_size);
        int32_t *          valid_ep_ಎಣಿಕೆ          = (int32_t *) (matrix_row_counts + n_as);
        int32_t *          valid_act_ಎಣಿಕೆ         = (int32_t *) (valid_ep_ಎಣಿಕೆ + 1);
        int64_t *          valid_matrix_row_counts = (int64_t *) (valid_act_ಎಣಿಕೆ + 1);
        mmid_row_mapping * matrix_rows             = (mmid_row_mapping *) (valid_matrix_row_counts + n_as);

        ಯದಿ (ith == 0) {
            // initialize matrix_row_counts
            memset(matrix_row_counts, 0, n_as * ಪರಿಮಾಣ(int64_t));

            // group rows by src0 matrix
            ಪ್ರತಿ (int32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
                ಪ್ರತಿ (int32_t id = 0; id < n_ids; ++id) {
                    ಧ್ರುವ int32_t i02 =
                        *(ಧ್ರುವ int32_t *) ((ಧ್ರುವ ವರ್ಣ *) ids->data + iid1 * ids->nb[1] + id * ids->nb[0]);

                    GGML_ASSERT(i02 >= 0 && i02 < n_as);

                    MMID_MATRIX_ROW(i02, matrix_row_counts[i02]) = { id, iid1 };
                    matrix_row_counts[i02] += 1;
                }
            }

            int32_t valid_ep_ಎಣಿಕೆ_t  = 0;
            int32_t valid_act_ಎಣಿಕೆ_t = 0;
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ cur_a = 0; cur_a < n_as; ++cur_a) {
                ಧ್ರುವ int64_t cne1 = matrix_row_counts[cur_a];
                ಯದಿ (cne1 == 0) {
                    ಅನುವರ್ತ;
                }
                valid_matrix_row_counts[valid_ep_ಎಣಿಕೆ_t] = cur_a;
                valid_act_ಎಣಿಕೆ_t += cne1;
                valid_ep_ಎಣಿಕೆ_t += 1;
            }
            valid_ep_ಎಣಿಕೆ[0]  = valid_ep_ಎಣಿಕೆ_t;
            valid_act_ಎಣಿಕೆ[0] = valid_act_ಎಣಿಕೆ_t;
        }

        ಧ್ರುವ int64_t barrier_idx = static_cast<int64_t>(ith / 2);

        GGML_ASSERT(global_spine_env_info.init_barrier != nullptr);
        GGML_ASSERT(barrier_idx < spine_init_barrier_ಎಣಿಕೆ);
        spine_barrier_t * cur_barrier = &global_spine_env_info.init_barrier[barrier_idx];

        ggml_barrier(params->threadpool);

        ಧ್ರುವ size_t row_stride_b      = b_k_blks * get_repacked_block_type_size<BLOC_TYPE, INTER_SIZE, NB_COLS>();
        ಧ್ರುವ size_t expert_b_stride   = ne01 * row_stride_b;
        ಧ್ರುವ size_t per_nb_cols_wsize = NB_COLS * row_stride_b;

        std::array<ಧ್ರುವ uint8_t *, 2> src_workspaces;
        std::array<ಭಿನ್ನ *, 2>         dst_workspaces;

        auto *     tcm_buffer      = ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer;
        ಧ್ರುವ auto tcm_buffer_size = ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer_size;

        ಧ್ರುವ auto valid_ep_ಎಣಿಕೆ_t  = valid_ep_ಎಣಿಕೆ[0];
        ಧ್ರುವ auto valid_act_ಎಣಿಕೆ_t = valid_act_ಎಣಿಕೆ[0];

        ಪೂರ್ಣಾಙ್ಕ nth_es = 1;
        ಪೂರ್ಣಾಙ್ಕ nth_n  = nth;

        ಪೂರ್ಣಾಙ್ಕ ith_es = ith % nth_es;
        ಪೂರ್ಣಾಙ್ಕ ith_n  = (ith / nth_es) % nth_n;

        ಯದಿ (valid_ep_ಎಣಿಕೆ_t % nth == 0 && tcm_buffer != nullptr && valid_ep_ಎಣಿಕೆ_t == n_as &&
            valid_act_ಎಣಿಕೆ_t == n_as && per_nb_cols_wsize <= tcm_buffer_size) {
            ಪ್ರತಿ (int64_t valid_id = ith; valid_id < valid_ep_ಎಣಿಕೆ_t; valid_id += nth) {
                ಧ್ರುವ int64_t cur_a = valid_matrix_row_counts[valid_id];

                auto * src0_cur = (uint8_t *) src0->data + cur_a * expert_b_stride;

                mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, 0);
                ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ        id          = row_mapping.i1;
                ಧ್ರುವ int64_t    i11         = id % ne11;
                ಧ್ರುವ int64_t    i12         = row_mapping.i2;
                ಧ್ರುವ int64_t    i1          = id;
                ಧ್ರುವ int64_t    i2          = i12;

                auto *  src1_col = quant_a_buffer + (i11 * nbw1 + i12 * nbw2);
                ಭಿನ್ನ * c_blk    = (ಭಿನ್ನ *) ((ವರ್ಣ *) dst->data + (i1 * nb1 + i2 * nb2));

                uint8_t * a_row = src1_col;
                uint8_t * b_col = reinterpret_cast<uint8_t *>(tcm_buffer);
                ಯದಿ ((nbw1 + per_nb_cols_wsize) <= tcm_buffer_size) {
                    a_row = (uint8_t *) tcm_buffer;
                    b_col = reinterpret_cast<uint8_t *>(tcm_buffer) + nbw1;
                }
                uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? b_col : nullptr;

                ಯದಿ (ith % 2 == 0) {
                    spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(src0_cur), per_nb_cols_wsize);

                    ಯದಿ (a_row != src1_col) {
                        spacemit_kernels::rvv::memcpy1d(a_row, src1_col, nbw1);
                    }
                }

                spine_barrier_wait(cur_barrier);

                ಯದಿ (ith % 2 != 0) {
                    ಯದಿ (a_row != src1_col) {
                        spacemit_kernels::rvv::memcpy1d(a_row, src1_col, nbw1);
                    }

                    spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(src0_cur), per_nb_cols_wsize);
                }

                int64_t nb_real = std::min(ne01, NB_COLS);
                ಪ್ರತಿ (int64_t ni = 0; ni < ne01; ni += NB_COLS) {
                    ಯದಿ (ith % 2 != 0) {
                        spine_barrier_wait(cur_barrier);
                    }

                    gemm_kernel(b_blk_len, a_row, b_col, b_col_zp, c_blk + ni, 1, nb_real, b_k_blks, ne01);

                    ಯದಿ (ith % 2 == 0) {
                        spine_barrier_wait(cur_barrier);
                    }

                    ಧ್ರುವ int64_t next_ni = ni + NB_COLS;
                    ಯದಿ (next_ni < ne01) {
                        nb_real = std::min(ne01 - next_ni, NB_COLS);
                        spacemit_kernels::rvv::memcpy1d(
                            b_col, reinterpret_cast<uint8_t *>(src0_cur) + next_ni * row_stride_b, per_nb_cols_wsize);
                    }
                }
            }
        } ಅನ್ಯಥಾ {
            ಪ್ರತಿ (int64_t valid_id = ith_es; valid_id < valid_ep_ಎಣಿಕೆ_t; valid_id += nth_es) {
                ಧ್ರುವ int64_t cur_a = valid_matrix_row_counts[valid_id];
                ಧ್ರುವ int64_t cne1  = matrix_row_counts[cur_a];

                int64_t src1_cur_start = 0;
                int64_t src1_cur_end   = cne1;

                int64_t src0_cur_start = (ith_n * ne01) / nth_n;
                int64_t src0_cur_end   = MIN(((ith_n + 1) * ne01) / nth_n, ne01);

                ಯದಿ (src1_cur_start >= src1_cur_end || src0_cur_start >= src0_cur_end) {
                    ಅನುವರ್ತ;
                }

                src0_cur_start =
                    (src0_cur_start % NB_COLS) ? src0_cur_start + NB_COLS - (src0_cur_start % NB_COLS) : src0_cur_start;
                src0_cur_end =
                    (src0_cur_end % NB_COLS) ? src0_cur_end + NB_COLS - (src0_cur_end % NB_COLS) : src0_cur_end;

                auto *    src0_cur = (uint8_t *) src0->data + cur_a * expert_b_stride + src0_cur_start * row_stride_b;
                uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? src0_cur : nullptr;

                size_t extra_tcm_buffer_size = tcm_buffer_size;
                ಶೂನ್ಯ * extra_tcm_buffer      = tcm_buffer;
                ಯದಿ (tcm_buffer != nullptr && (src1_cur_end - src1_cur_start) >= 4 &&
                    (src0_cur_end - src0_cur_start) * row_stride_b <= tcm_buffer_size) {
                    spacemit_kernels::rvv::memcpy1d(tcm_buffer, src0_cur,
                                                    (src0_cur_end - src0_cur_start) * row_stride_b);
                    src0_cur = reinterpret_cast<uint8_t *>(tcm_buffer);
                    b_col_zp = block_type_has_zp<BLOC_TYPE>() ? src0_cur : nullptr;
                    extra_tcm_buffer_size -= (src0_cur_end - src0_cur_start) * row_stride_b;
                    extra_tcm_buffer = reinterpret_cast<ಶೂನ್ಯ *>(reinterpret_cast<uintptr_t>(tcm_buffer) +
                                                                (src0_cur_end - src0_cur_start) * row_stride_b);
                }

                ಪೂರ್ಣಾಙ್ಕ ir1 = src1_cur_start;

                ಯದಿ (extra_tcm_buffer_size >= nbw1 && extra_tcm_buffer != nullptr) {
                    int64_t quant_a_tile_size = extra_tcm_buffer_size / nbw1;
                    ಕುರು {
                        quant_a_tile_size = MIN(quant_a_tile_size, src1_cur_end - ir1);

                        uint8_t * quant_a_tile_buffer = reinterpret_cast<uint8_t *>(extra_tcm_buffer);

                        ಪೂರ್ಣಾಙ್ಕ iir1 = ir1;
                        ಪ್ರತಿ (; iir1 < (ir1 + quant_a_tile_size); ++iir1) {
                            mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, iir1);

                            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ id = row_mapping.i1;  // selected expert index

                            ಧ್ರುವ int64_t i11 = id % ne11;
                            ಧ್ರುವ int64_t i12 = row_mapping.i2;  // row index in src1

                            auto * src1_col = quant_a_buffer + (i11 * nbw1 + i12 * nbw2);
                            spacemit_kernels::rvv::memcpy1d(quant_a_tile_buffer, src1_col, nbw1);
                            quant_a_tile_buffer = quant_a_tile_buffer + nbw1;
                        }

                        quant_a_tile_buffer = reinterpret_cast<uint8_t *>(extra_tcm_buffer);
                        iir1                = ir1;

                        ಯದಿ (moe_gemm_kernel_m2 != nullptr) {
                            ಪ್ರತಿ (; iir1 < (ir1 + quant_a_tile_size - 1); iir1 += 2, quant_a_tile_buffer += 2 * nbw1) {
                                mmid_row_mapping row_mapping_0 = MMID_MATRIX_ROW(cur_a, iir1);
                                mmid_row_mapping row_mapping_1 = MMID_MATRIX_ROW(cur_a, iir1 + 1);

                                src_workspaces[0] = quant_a_tile_buffer;
                                src_workspaces[1] = quant_a_tile_buffer + nbw1;

                                dst_workspaces[0] =
                                    (ಭಿನ್ನ *) ((ವರ್ಣ *) dst->data + (row_mapping_0.i1 * nb1 + row_mapping_0.i2 * nb2)) +
                                    src0_cur_start;
                                dst_workspaces[1] = (ಭಿನ್ನ *) ((ವರ್ಣ *) dst->data +
                                                               ((row_mapping_1.i1) * nb1 + (row_mapping_1.i2) * nb2)) +
                                                    src0_cur_start;
                                moe_gemm_kernel_m2(b_blk_len, src_workspaces.data(), src0_cur, b_col_zp,
                                                   dst_workspaces.data(), 1, src0_cur_end - src0_cur_start, b_k_blks,
                                                   ne01);
                            }
                        }

                        ಪ್ರತಿ (; iir1 < (ir1 + quant_a_tile_size); iir1++, quant_a_tile_buffer += nbw1) {
                            mmid_row_mapping row_mapping_0 = MMID_MATRIX_ROW(cur_a, iir1);

                            gemm_kernel(
                                b_blk_len, quant_a_tile_buffer, src0_cur, b_col_zp,
                                (ಭಿನ್ನ *) ((ವರ್ಣ *) dst->data + (row_mapping_0.i1 * nb1 + row_mapping_0.i2 * nb2)) +
                                    src0_cur_start,
                                1, src0_cur_end - src0_cur_start, b_k_blks, ne01);
                        }

                        ir1 += quant_a_tile_size;
                    } ಯಾವತ್ (ir1 < src1_cur_end);
                } ಅನ್ಯಥಾ {
                    ಯದಿ (moe_gemm_kernel_m2 != nullptr) {
                        ಪ್ರತಿ (; ir1 < src1_cur_end - 1; ir1 += 2) {
                            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ iir1 = 0; iir1 < 2; ++iir1) {
                                mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, ir1 + iir1);

                                ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ id = row_mapping.i1;  // selected expert index

                                ಧ್ರುವ int64_t i11 = id % ne11;
                                ಧ್ರುವ int64_t i12 = row_mapping.i2;  // row index in src1

                                ಧ್ರುವ int64_t i1 = id;               // selected expert index
                                ಧ್ರುವ int64_t i2 = i12;              // row

                                src_workspaces[iir1] = quant_a_buffer + (i11 * nbw1 + i12 * nbw2);

                                dst_workspaces[iir1] =
                                    (ಭಿನ್ನ *) ((ವರ್ಣ *) dst->data + (i1 * nb1 + i2 * nb2)) + src0_cur_start;
                            }

                            moe_gemm_kernel_m2(b_blk_len, src_workspaces.data(), src0_cur, b_col_zp,
                                               dst_workspaces.data(), 1, src0_cur_end - src0_cur_start, b_k_blks, ne01);
                        }
                    }

                    ಪ್ರತಿ (; ir1 < src1_cur_end; ir1++) {
                        mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, ir1);

                        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ id = row_mapping.i1;  // selected expert index

                        ಧ್ರುವ int64_t i11 = id % ne11;
                        ಧ್ರುವ int64_t i12 = row_mapping.i2;  // row index in src1

                        ಧ್ರುವ int64_t i1 = id;               // selected expert index
                        ಧ್ರುವ int64_t i2 = i12;              // row

                        auto * src1_col = quant_a_buffer + (i11 * nbw1 + i12 * nbw2);

                        gemm_kernel(b_blk_len, src1_col, src0_cur, b_col_zp,
                                    (ಭಿನ್ನ *) ((ವರ್ಣ *) dst->data + (i1 * nb1 + i2 * nb2)) + src0_cur_start, 1,
                                    src0_cur_end - src0_cur_start, b_k_blks, ne01);
                    }
                }
            }
        }
#undef MMID_MATRIX_ROW
    }

    ಪೂರ್ಣಾಙ್ಕ repack(ggml_tensor * t, ಧ್ರುವ ಶೂನ್ಯ * data, size_t data_size) override {
        GGML_LOG_DEBUG("%s: repack tensor %s with %s_%dx%d\n", __func__, t->name, ggml_type_name(t->type),
                       (ಪೂರ್ಣಾಙ್ಕ) NB_COLS, (ಪೂರ್ಣಾಙ್ಕ) INTER_SIZE);
        ಪ್ರತ್ಯಾಗಮ ggml::cpu::riscv64_spacemit::repack<BLOC_TYPE, INTER_SIZE, NB_COLS>(t, data, data_size);
    }
};

ವರ್ಗ tensor_traits_common : ಸಾರ್ವಜನಿಕ tensor_traits_base {
    ತಾರ್ಕಿಕ work_size(ಪೂರ್ಣಾಙ್ಕ n_threads, ಧ್ರುವ ggml_tensor * op, size_t & size) override {
        ವಿಕಲ್ಪ (op->op) {
            ಪಕ್ಷ GGML_OP_FLASH_ATTN_EXT:
                {
                    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ     n_tasks = n_threads;
                    ಧ್ರುವ int64_t neq2    = op->src[0]->ne[2];  // number of query heads
                    ಧ್ರುವ int64_t DK      = op->src[1]->ne[0];
                    ಧ್ರುವ int64_t DV      = op->src[2]->ne[0];  // DV

                    // Tiled flash attention scratch (tile sizes defined in common.h)
                    // Per-thread: Q_q + KQ + mask + VKQ32 + V32 + K_f32 + padding
                    size_t prefill = ಪರಿಮಾಣ(ಭಿನ್ನ) *
                                     (GGML_FA_TILE_Q * DK + 2 * GGML_FA_TILE_Q * GGML_FA_TILE_KV + GGML_FA_TILE_Q * DV +
                                      GGML_FA_TILE_KV * DV + GGML_FA_TILE_KV * DK) *
                                     n_tasks;

                    // Decode path: n_kv_chunks = n_tasks (one chunk per thread)
                    // Per-thread: VKQ accmulator (DV), partial M, partial S + intra-thread scratch for V, Q and VKQ
                    size_t n_chunks = n_tasks;
                    size_t decode   = ಪರಿಮಾಣ(ಭಿನ್ನ) * (neq2 * n_chunks * (2 + DV) + n_tasks * (DK + 2 * DV));

                    size = MAX(prefill, decode);
                }
                ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
            ಸ್ವಭಾವ:
                ಭಙ್ಗ;
        }
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ತಾರ್ಕಿಕ compute_forward(ggml_compute_params * params, ggml_tensor * op) override {
        ವಿಕಲ್ಪ (op->op) {
            ಪಕ್ಷ GGML_OP_NORM:
                ವಿಕಲ್ಪ (op->src[0]->type) {
                    ಪಕ್ಷ GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_norm_f32(params, op);
                        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
                    ಸ್ವಭಾವ:
                        GGML_ABORT("fatal error");
                }
            ಪಕ್ಷ GGML_OP_RMS_NORM:
                ವಿಕಲ್ಪ (op->src[0]->type) {
                    ಪಕ್ಷ GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_rms_norm_f32(params, op);
                        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
                    ಸ್ವಭಾವ:
                        GGML_ABORT("fatal error");
                }
            ಪಕ್ಷ GGML_OP_ADD:
                ವಿಕಲ್ಪ (op->src[0]->type) {
                    ಪಕ್ಷ GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_ದ್ವಿಮಾನ<GGML_OP_ADD, ಭಿನ್ನ>(params, op);
                        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
                    ಪಕ್ಷ GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_ದ್ವಿಮಾನ<GGML_OP_ADD, _Float16>(params, op);
                        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
                    ಸ್ವಭಾವ:
                        ggml_compute_forward_add(params, op);
                        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
                }
            ಪಕ್ಷ GGML_OP_SUB:
                ವಿಕಲ್ಪ (op->src[0]->type) {
                    ಪಕ್ಷ GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_ದ್ವಿಮಾನ<GGML_OP_SUB, ಭಿನ್ನ>(params, op);
                        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
                    ಪಕ್ಷ GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_ದ್ವಿಮಾನ<GGML_OP_SUB, _Float16>(params, op);
                        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
                    ಸ್ವಭಾವ:
                        ggml_compute_forward_sub(params, op);
                        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
                }
            ಪಕ್ಷ GGML_OP_MUL:
                ವಿಕಲ್ಪ (op->src[0]->type) {
                    ಪಕ್ಷ GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_ದ್ವಿಮಾನ<GGML_OP_MUL, ಭಿನ್ನ>(params, op);
                        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
                    ಪಕ್ಷ GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_ದ್ವಿಮಾನ<GGML_OP_MUL, _Float16>(params, op);
                        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
                    ಸ್ವಭಾವ:
                        ggml_compute_forward_mul(params, op);
                        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
                }
            ಪಕ್ಷ GGML_OP_DIV:
                ವಿಕಲ್ಪ (op->src[0]->type) {
                    ಪಕ್ಷ GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_ದ್ವಿಮಾನ<GGML_OP_DIV, ಭಿನ್ನ>(params, op);
                        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
                    ಪಕ್ಷ GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_ದ್ವಿಮಾನ<GGML_OP_DIV, _Float16>(params, op);
                        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
                    ಸ್ವಭಾವ:
                        ggml_compute_forward_div(params, op);
                        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
                }
            ಪಕ್ಷ GGML_OP_FLASH_ATTN_EXT:
                forward_flash_attn_ext_f16(params, op);
                ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
            ಪಕ್ಷ GGML_OP_CONT:
                {
                    ಧ್ರುವ ggml_tensor * src0 = op->src[0];
                    ಯದಿ (op->type == src0->type && op->nb[0] != src0->nb[0] && op->nb[0] == src0->nb[1] &&
                        op->ne[3] * op->ne[2] * op->nb[2] == src0->ne[3] * src0->ne[2] * src0->nb[2]) {
                        spacemit_kernels::rvv::forward_cont_with_permute(params, op);
                    } ಅನ್ಯಥಾ {
                        ggml_compute_forward_cont(params, op);
                    }
                    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
                }
            ಪಕ್ಷ GGML_OP_CPY:
                {
                    ಧ್ರುವ ggml_tensor * src0 = op->src[0];
                    ಯದಿ (op->type == src0->type && op->nb[0] == src0->nb[1] && src0->nb[0] != src0->nb[1] &&
                        ggml_nelements(src0) == ggml_nelements(op)) {
                        spacemit_kernels::rvv::forward_cpy_with_permute(params, op);
                    } ಅನ್ಯಥಾ {
                        ggml_compute_forward_cpy(params, op);
                    }
                    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
                }
            ಪಕ್ಷ GGML_OP_REPEAT:
                {
                    ಧ್ರುವ ತಾರ್ಕಿಕ rows_equal         = ggml_nrows(op->src[0]) == ggml_nrows(op);
                    ಧ್ರುವ ತಾರ್ಕಿಕ broadcast_or_equal = op->src[0]->ne[0] == 1 || op->src[0]->ne[0] == op->ne[0];

                    ಯದಿ (rows_equal && broadcast_or_equal) {
                        ವಿಕಲ್ಪ (op->src[0]->type) {
                            ಪಕ್ಷ GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_repeat_nrows<int32_t>(params, op);
                                ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
                            ಪಕ್ಷ GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_repeat_nrows<int16_t>(params, op);
                                ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
                            ಸ್ವಭಾವ:
                                ಭಙ್ಗ;
                        }
                    }

                    ಯದಿ (op->src[0]->ne[1] == 1 && op->src[0]->ne[0] == op->ne[0]) {
                        ವಿಕಲ್ಪ (op->src[0]->type) {
                            ಪಕ್ಷ GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_repeat_dim1<int32_t>(params, op);
                                ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
                            ಪಕ್ಷ GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_repeat_dim1<int16_t>(params, op);
                                ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
                            ಸ್ವಭಾವ:
                                ಭಙ್ಗ;
                        }
                    }

                    ggml_compute_forward_repeat(params, op);
                }
                ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
            ಪಕ್ಷ GGML_OP_SUM_ROWS:
                {
                    ಯದಿ (op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32) {
                        spacemit_kernels::rvv::forward_sum_rows<ಭಿನ್ನ>(params, op);
                    } ಅನ್ಯಥಾ {
                        ggml_compute_forward_sum_rows(params, op);
                    }
                }
                ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
            ಪಕ್ಷ GGML_OP_GET_ROWS:
                {
                    ಯದಿ (op->src[0]->type == op->type) {
                        ವಿಕಲ್ಪ (op->src[0]->type) {
                            ಪಕ್ಷ GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_get_rows<int32_t>(params, op);
                                ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
                            ಪಕ್ಷ GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_get_rows<int16_t>(params, op);
                                ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
                            ಸ್ವಭಾವ:
                                ಭಙ್ಗ;
                        }
                    }

                    ggml_compute_forward_get_rows(params, op);
                }
                ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
            ಪಕ್ಷ GGML_OP_CONCAT:
                {
                    ಧ್ರುವ int32_t dim = ggml_get_op_params_i32(op, 0);
                    ಯದಿ (dim == 0 && op->type == op->src[0]->type) {
                        ವಿಕಲ್ಪ (op->src[0]->type) {
                            ಪಕ್ಷ GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_concat<int32_t>(params, op);
                                ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
                            ಪಕ್ಷ GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_concat<int16_t>(params, op);
                                ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
                            ಸ್ವಭಾವ:
                                ಭಙ್ಗ;
                        }
                    }

                    ggml_compute_forward_concat(params, op);
                }
                ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
            // TODO For GGML_OP_GATED_DELTA_NET
            // case GGML_OP_GATED_DELTA_NET:
            //     return true;
            ಸ್ವಭಾವ:
                ಭಙ್ಗ;
        }
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಶೂನ್ಯ forward_flash_attn_ext_f16(ಧ್ರುವ ggml_compute_params * params, ggml_tensor * dst) {
        ಧ್ರುವ ggml_tensor * q = dst->src[0];
        ಧ್ರುವ ggml_tensor * k = dst->src[1];
        ಧ್ರುವ ggml_tensor * v = dst->src[2];

        GGML_TENSOR_LOCALS(int64_t, neq, q, ne)
        GGML_TENSOR_LOCALS(size_t, nbq, q, nb)
        GGML_TENSOR_LOCALS(int64_t, nek, k, ne)
        GGML_TENSOR_LOCALS(size_t, nbk, k, nb)
        GGML_TENSOR_LOCALS(int64_t, nev, v, ne)
        GGML_TENSOR_LOCALS(size_t, nbv, v, nb)
        GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
        GGML_TENSOR_LOCALS(size_t, nb, dst, nb)

        ಧ್ರುವ int64_t DK = nek0;
        ಧ್ರುವ int64_t DV = nev0;

        ಧ್ರುವ ತಾರ್ಕಿಕ supported_prec  = (dst->op_params[3] == GGML_PREC_F32 || dst->op_params[3] == GGML_PREC_DEFAULT);
        ಧ್ರುವ ತಾರ್ಕಿಕ supported_types = (q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16);
        ಧ್ರುವ ತಾರ್ಕಿಕ supported_shape = (DK > 0 && DK <= 128 && DV > 0 && DV <= 128);
        ಧ್ರುವ ತಾರ್ಕಿಕ supported_vlen  = (__riscv_vlenb() == 128);

        ಯದಿ (!(supported_prec && supported_types && supported_shape && supported_vlen)) {
            ggml_compute_forward_flash_attn_ext(params, dst);
            ಪ್ರತ್ಯಾಗಮ;
        }

        // total rows in q
        ಧ್ರುವ int64_t nr = neq1 * neq2 * neq3;

        // rows per thread
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ith = params->ith;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nth = params->nth;

        ಸ್ಥಿರ constexpr int64_t Q_TILE_SZ = ggml_fa_tile_config::Q;
        ಧ್ರುವ ತಾರ್ಕಿಕ               use_tiled = !params->use_ref && (neq1 >= Q_TILE_SZ);

        // 4x chunks per thread
        // int     nth_scaled = nth * 4;
        // int64_t chunk_size = (nr + nth_scaled - 1) / nth_scaled;
        // int64_t nchunk     = (nr + chunk_size - 1) / chunk_size;

        // if (nth == 1 || nchunk < nth) {
        //     nchunk = nth;
        // }

        int64_t nchunk = nth;

        ಯದಿ (ith == 0) {
            // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
            ggml_threadpool_chunk_set(params->threadpool, nth);
        }

        ggml_barrier(params->threadpool);

        // The number of elements in each chunk
        ಧ್ರುವ int64_t dr = (nr + nchunk - 1) / nchunk;

        // The first chunk comes from our thread_id, the rest will get auto-assigned.
        ಪೂರ್ಣಾಙ್ಕ current_chunk = ith;

        ಯಾವತ್ (current_chunk < nchunk) {
            ಧ್ರುವ int64_t ir0 = dr * current_chunk;
            ಧ್ರುವ int64_t ir1 = MIN(ir0 + dr, nr);

            ಯದಿ (use_tiled) {
                spacemit_kernels::rvv::forward_flash_attn_ext_f16_tiled_vlen1024_vf16(
                    params, dst, ir0, ir1, ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer,
                    ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer_size);
            } ಅನ್ಯಥಾ {
                spacemit_kernels::rvv::forward_flash_attn_ext_f16_one_chunk_vlen1024_vf16(
                    params, dst, ir0, ir1, ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer,
                    ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer_size);
            }

            current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
        }
    }

    ಪೂರ್ಣಾಙ್ಕ repack(ggml_tensor * t, ಧ್ರುವ ಶೂನ್ಯ * data, size_t data_size) override {
        memcpy(t->data, data, data_size);
        ಪ್ರತ್ಯಾಗಮ 0;
    }
};

// Impl By IME1
ಸ್ಥಿರ ಧ್ರುವ tensor_traits<block_q4_0, 32, 16>  q4_0_16x32_q8_0;
ಸ್ಥಿರ ಧ್ರುವ tensor_traits<block_q4_1, 32, 16>  q4_1_16x32_q8_0;
ಸ್ಥಿರ ಧ್ರುವ tensor_traits<block_q4_K, 32, 16>  q4_k_16x32_q8_0;
ಸ್ಥಿರ ಧ್ರುವ tensor_traits<block_q8_0, 32, 16>  q8_0_16x32_q8_0;
// Impl By IME2
ಸ್ಥಿರ ಧ್ರುವ tensor_traits<block_q2_K, 256, 32> q2_k_32x256_q8_0;
ಸ್ಥಿರ ಧ್ರುವ tensor_traits<block_q3_K, 256, 32> q3_k_32x256_q8_0;
ಸ್ಥಿರ ಧ್ರುವ tensor_traits<block_q4_0, 32, 32>  q4_0_32x32_q8_0;
ಸ್ಥಿರ ಧ್ರುವ tensor_traits<block_q4_1, 32, 32>  q4_1_32x32_q8_0;
ಸ್ಥಿರ ಧ್ರುವ tensor_traits<block_q4_0, 256, 32> q4_0_32x256_q8_0;
ಸ್ಥಿರ ಧ್ರುವ tensor_traits<block_q4_1, 256, 32> q4_1_32x256_q8_0;
ಸ್ಥಿರ ಧ್ರುವ tensor_traits<block_q4_K, 32, 32>  q4_k_32x32_q8_0;
ಸ್ಥಿರ ಧ್ರುವ tensor_traits<block_q6_K, 32, 32>  q6_k_32x32_q8_0;
ಸ್ಥಿರ ಧ್ರುವ tensor_traits<block_q8_0, 32, 32>  q8_0_32x32_q8_0;
ಸ್ಥಿರ ಧ್ರುವ tensor_traits<block_mxfp4, 32, 32> mxfp4_32x32_q8_0;
ಸ್ಥಿರ ಧ್ರುವ tensor_traits<block_q5_K, 32, 32>  q5_k_32x32_q8_0;
ಸ್ಥಿರ ಧ್ರುವ tensor_traits<block_q5_1, 32, 32>  q5_1_32x32_q8_0;
ಸ್ಥಿರ ಧ್ರುವ tensor_traits<block_q5_0, 32, 32>  q5_0_32x32_q8_0;
// Impl By RVV
ಸ್ಥಿರ ಧ್ರುವ tensor_traits_common               rvv_impl;

}  // namespace ggml::cpu::riscv64_spacemit

ಸ್ಥಿರ ಧ್ರುವ ggml::cpu::tensor_traits * ggml_riscv64_spacemit_get_optimal_repack_type(ಧ್ರುವ ggml_tensor * cur) {
    ವಿಕಲ್ಪ (cur->type) {
        ಪಕ್ಷ GGML_TYPE_Q2_K:
            {
#ಯದಿ defined(RISCV64_SPACEMIT_IME2)
                ಯದಿ (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    ಪ್ರತ್ಯಾಗಮ &ggml::cpu::riscv64_spacemit::q2_k_32x256_q8_0;
                }
#endif
            }
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_Q3_K:
            {
#ಯದಿ defined(RISCV64_SPACEMIT_IME2)
                ಯದಿ (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    ಪ್ರತ್ಯಾಗಮ &ggml::cpu::riscv64_spacemit::q3_k_32x256_q8_0;
                }
#endif
            }
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_Q4_0:
            {
#ಯದಿ defined(RISCV64_SPACEMIT_IME2)
                ಯದಿ (cur->ne[1] % 32 == 0 && cur->ne[0] % 256 == 0 &&
                    (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    ಪ್ರತ್ಯಾಗಮ &ggml::cpu::riscv64_spacemit::q4_0_32x256_q8_0;
                }

                ಯದಿ (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    ಪ್ರತ್ಯಾಗಮ &ggml::cpu::riscv64_spacemit::q4_0_32x32_q8_0;
                }
#endif

#ಯದಿ defined(RISCV64_SPACEMIT_IME1)
                ಯದಿ (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    ಪ್ರತ್ಯಾಗಮ &ggml::cpu::riscv64_spacemit::q4_0_16x32_q8_0;
                }
#endif
            }
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_Q4_1:
            {
#ಯದಿ defined(RISCV64_SPACEMIT_IME2)
                // TODO
                // if (cur->ne[1] % 32 == 0 && cur->ne[0] % 256 == 0 &&
                //     (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                //     return &ggml::cpu::riscv64_spacemit::q4_1_32x256_q8_0;
                // }

                ಯದಿ (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    ಪ್ರತ್ಯಾಗಮ &ggml::cpu::riscv64_spacemit::q4_1_32x32_q8_0;
                }
#endif

#ಯದಿ defined(RISCV64_SPACEMIT_IME1)
                ಯದಿ (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    ಪ್ರತ್ಯಾಗಮ &ggml::cpu::riscv64_spacemit::q4_1_16x32_q8_0;
                }
#endif
            }
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_Q4_K:
            {
#ಯದಿ defined(RISCV64_SPACEMIT_IME2)
                ಯದಿ (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    ಪ್ರತ್ಯಾಗಮ &ggml::cpu::riscv64_spacemit::q4_k_32x32_q8_0;
                }
#endif

#ಯದಿ defined(RISCV64_SPACEMIT_IME1)
                ಯದಿ (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    ಪ್ರತ್ಯಾಗಮ &ggml::cpu::riscv64_spacemit::q4_k_16x32_q8_0;
                }
#endif
            }
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_Q6_K:
            {
#ಯದಿ defined(RISCV64_SPACEMIT_IME2)
                ಯದಿ ((ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    ಪ್ರತ್ಯಾಗಮ &ggml::cpu::riscv64_spacemit::q6_k_32x32_q8_0;
                }
#endif
            }
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_Q8_0:
            {
#ಯದಿ defined(RISCV64_SPACEMIT_IME2)
                ಯದಿ ((ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    ಪ್ರತ್ಯಾಗಮ &ggml::cpu::riscv64_spacemit::q8_0_32x32_q8_0;
                }
#endif

#ಯದಿ defined(RISCV64_SPACEMIT_IME1)
                ಯದಿ (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    ಪ್ರತ್ಯಾಗಮ &ggml::cpu::riscv64_spacemit::q8_0_16x32_q8_0;
                }
#endif
            }
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_MXFP4:
            {
#ಯದಿ defined(RISCV64_SPACEMIT_IME2)
                // TODO
                // if (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                //     return &ggml::cpu::riscv64_spacemit::mxfp4_32x32_q8_0;
                // }
#endif
            }
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_Q5_K:
            {
#ಯದಿ defined(RISCV64_SPACEMIT_IME2)
                ಯದಿ (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    ಪ್ರತ್ಯಾಗಮ &ggml::cpu::riscv64_spacemit::q5_k_32x32_q8_0;
                }
#endif
            }
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_Q5_1:
            {
#ಯದಿ defined(RISCV64_SPACEMIT_IME2)
                ಯದಿ (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    ಪ್ರತ್ಯಾಗಮ &ggml::cpu::riscv64_spacemit::q5_1_32x32_q8_0;
                }
#endif
            }
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_Q5_0:
            {
#ಯದಿ defined(RISCV64_SPACEMIT_IME2)
                ಯದಿ (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    ಪ್ರತ್ಯಾಗಮ &ggml::cpu::riscv64_spacemit::q5_0_32x32_q8_0;
                }
#endif
            }
            ಭಙ್ಗ;
        ಸ್ವಭಾವ:
            ಭಙ್ಗ;
    }

    ಪ್ರತ್ಯಾಗಮ nullptr;
}

ಸ್ಥಿರ ಗಣನಾ ggml_status ggml_backend_riscv64_spacemit_buffer_init_tensor(ggml_backend_buffer_t buffer,
                                                                         ggml_tensor *         tensor) {
    tensor->extra =
        (ಶೂನ್ಯ *) const_cast<ggml::cpu::tensor_traits *>(ggml_riscv64_spacemit_get_optimal_repack_type(tensor));

    GGML_UNUSED(buffer);

    ಪ್ರತ್ಯಾಗಮ GGML_STATUS_SUCCESS;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_riscv64_spacemit_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);

    ಶೂನ್ಯ * base = buffer->context;
    ಯದಿ (base == nullptr) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    ggml::cpu::riscv64_spacemit::spine_mem_pool_free(base);
}

ಸ್ಥಿರ ಶೂನ್ಯ * ggml_backend_riscv64_spacemit_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);

    ಶೂನ್ಯ * base = buffer->context;
    GGML_ASSERT(base != nullptr);
    ಪ್ರತ್ಯಾಗಮ base;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_riscv64_spacemit_buffer_memset_tensor(ggml_backend_buffer_t buffer,
                                                               ggml_tensor *         tensor,
                                                               uint8_t               value,
                                                               size_t                offset,
                                                               size_t                size) {
    GGML_ASSERT(tensor);
    memset((ವರ್ಣ *) tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_riscv64_spacemit_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);

    ಶೂನ್ಯ * base = buffer->context;
    GGML_ASSERT(base != nullptr);
    memset(base, value, buffer->size);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_riscv64_spacemit_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                            ggml_tensor *         tensor,
                                                            ಧ್ರುವ ಶೂನ್ಯ *          data,
                                                            size_t                offset,
                                                            size_t                size) {
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    auto tensor_traits = (ggml::cpu::riscv64_spacemit::tensor_traits_base *) tensor->extra;
    ಯದಿ (tensor_traits) {
        auto OK = tensor_traits->repack(tensor, data, size);
        GGML_ASSERT(OK == 0);
    }

    GGML_UNUSED(buffer);
}

ಸ್ಥಿರ ಧ್ರುವ ggml_backend_buffer_i ggml_backend_riscv64_spacemit_buffer_i = {
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

ಸ್ಥಿರ ಧ್ರುವ ವರ್ಣ * ggml_backend_cpu_riscv64_spacemit_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ಪ್ರತ್ಯಾಗಮ "CPU_RISCV64_SPACEMIT";

    GGML_UNUSED(buft);
}

ಸ್ಥಿರ ggml_backend_buffer_t ggml_backend_cpu_riscv64_spacemit_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                                                                        size_t size) {
    ಶೂನ್ಯ * base = ggml::cpu::riscv64_spacemit::spine_mem_pool_alloc(size, 64);
    ಯದಿ (base == nullptr) {
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }

    ಪ್ರತ್ಯಾಗಮ ggml_backend_buffer_init(buft, ggml_backend_riscv64_spacemit_buffer_i, base, size);
}

ಸ್ಥಿರ size_t ggml_backend_cpu_riscv64_spacemit_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ಪ್ರತ್ಯಾಗಮ 64;

    GGML_UNUSED(buft);
}

ಸ್ಥಿರ size_t ggml_backend_cpu_riscv64_spacemit_nbytes(ggml_backend_buffer_type_t buft, ಧ್ರುವ ggml_tensor * tensor) {
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < GGML_MAX_DIMS; ++i) {
        ಯದಿ (tensor->ne[i] <= 0) {
            ಪ್ರತ್ಯಾಗಮ 0;
        }
    }

    GGML_UNUSED(buft);

    ಧ್ರುವ auto plain_nbytes = [&]() {
        size_t total = ggml_type_size(tensor->type);
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < GGML_MAX_DIMS; ++i) {
            total += (tensor->ne[i] - 1) * tensor->nb[i];
        }
        ಪ್ರತ್ಯಾಗಮ total;
    };

    ಧ್ರುವ size_t blck_size = ggml_blck_size(tensor->type);
    ಯದಿ (blck_size == 1) {
        ಪ್ರತ್ಯಾಗಮ plain_nbytes();
    }

    ಧ್ರುವ size_t row_nbytes = tensor->ne[0] * tensor->nb[0] / blck_size;

    ಧ್ರುವ auto add_strided_nbytes = [&](size_t total, size_t src_block_size, size_t dst_block_size) {
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 1; i < GGML_MAX_DIMS; ++i) {
            total += (tensor->ne[i] - 1) * (tensor->nb[i] / src_block_size) * dst_block_size;
        }
        ಪ್ರತ್ಯಾಗಮ total;
    };

    ಧ್ರುವ auto remap_block_nbytes = [&](size_t src_block_size, size_t dst_block_size, int64_t padded_rows = 0) {
        GGML_ASSERT(row_nbytes % src_block_size == 0);

        size_t total =
            add_strided_nbytes((row_nbytes / src_block_size) * dst_block_size, src_block_size, dst_block_size);

        ಯದಿ (padded_rows > 0 && tensor->ne[1] % padded_rows != 0) {
            total += (padded_rows - tensor->ne[1] % padded_rows) * (tensor->nb[1] / src_block_size) * dst_block_size;
        }

        ಪ್ರತ್ಯಾಗಮ total;
    };

    size_t nbytes = row_nbytes;
    ವಿಕಲ್ಪ (tensor->type) {
        ಪಕ್ಷ GGML_TYPE_Q4_K:
            nbytes = remap_block_nbytes(ಪರಿಮಾಣ(block_q4_K), ಪರಿಮಾಣ(block_q4_1) * 8);
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_Q6_K:
            nbytes = remap_block_nbytes(ಪರಿಮಾಣ(block_q6_K), ಪರಿಮಾಣ(block_q8_0) * 8, 32);
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_Q8_0:
            nbytes = remap_block_nbytes(ಪರಿಮಾಣ(block_q8_0), ಪರಿಮಾಣ(block_q8_0), 32);
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_Q2_K:
            nbytes = remap_block_nbytes(ಪರಿಮಾಣ(block_q2_K), ಪರಿಮಾಣ(spacemit_kernels::nrow_block_q2_k<1>));
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_Q3_K:
            nbytes = remap_block_nbytes(ಪರಿಮಾಣ(block_q3_K), ಪರಿಮಾಣ(spacemit_kernels::nrow_block_q3_k<1>));
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_MXFP4:
            nbytes = remap_block_nbytes(ಪರಿಮಾಣ(block_mxfp4), ಪರಿಮಾಣ(spacemit_kernels::nrow_block_mxfp4<1>));
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_Q5_K:
            nbytes = remap_block_nbytes(ಪರಿಮಾಣ(block_q5_K), ಪರಿಮಾಣ(spacemit_kernels::nrow_block_q5_1<1>) * 8);
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_Q5_1:
            nbytes = remap_block_nbytes(ಪರಿಮಾಣ(block_q5_1), ಪರಿಮಾಣ(spacemit_kernels::nrow_block_q5_1<1>));
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_Q5_0:
            nbytes = remap_block_nbytes(ಪರಿಮಾಣ(block_q5_0), ಪರಿಮಾಣ(spacemit_kernels::nrow_block_q5_0<1>));
            ಭಙ್ಗ;
        ಸ್ವಭಾವ:
            nbytes = add_strided_nbytes(row_nbytes, 1, 1);
            ಭಙ್ಗ;
    }

    ಪ್ರತ್ಯಾಗಮ nbytes;
}

ನಾಮಾಕಾಶ ggml::cpu::riscv64_spacemit {

ವರ್ಗ extra_buffer_type : ggml::cpu::extra_buffer_type {
    ತಾರ್ಕಿಕ supports_op(ggml_backend_dev_t, ಧ್ರುವ ggml_tensor * op) override {
        ವಿಕಲ್ಪ (op->op) {
            ಪಕ್ಷ GGML_OP_MUL_MAT:
                ಯದಿ (op->src[0]->buffer && (ggml_n_dims(op->src[0]) == 2) &&
                    op->src[0]->buffer->buft == ggml_backend_cpu_riscv64_spacemit_buffer_type() &&
                    ggml_riscv64_spacemit_get_optimal_repack_type(op->src[0])) {
                    ಯದಿ (op->src[1]->buffer && !ggml_backend_buft_is_host(op->src[1]->buffer->buft)) {
                        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                    }
                    ಯದಿ (op->src[1]->type == GGML_TYPE_F32) {
                        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
                    }
                }
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_OP_MUL_MAT_ID:
                ಯದಿ (op->src[0]->buffer && (ggml_n_dims(op->src[0]) == 3) &&
                    op->src[0]->buffer->buft == ggml_backend_cpu_riscv64_spacemit_buffer_type() &&
                    ggml_riscv64_spacemit_get_optimal_repack_type(op->src[0])) {
                    ಯದಿ (op->src[1]->buffer && !ggml_backend_buft_is_host(op->src[1]->buffer->buft)) {
                        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                    }
                    ಯದಿ (op->src[1]->type == GGML_TYPE_F32) {
                        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
                    }
                }
                ಭಙ್ಗ;
            ಸ್ವಭಾವ:
                // GGML_ABORT("fatal error");
                ಭಙ್ಗ;
        }
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(ಧ್ರುವ ggml_tensor * op) override {
        ವಿಕಲ್ಪ (op->op) {
            ಪಕ್ಷ GGML_OP_MUL_MAT:
            ಪಕ್ಷ GGML_OP_MUL_MAT_ID:
                ಯದಿ (op->src[0]->buffer && op->src[0]->buffer->buft == ggml_backend_cpu_riscv64_spacemit_buffer_type()) {
                    ಪ್ರತ್ಯಾಗಮ (ggml::cpu::tensor_traits *) op->src[0]->extra;
                }
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_OP_NORM:
            ಪಕ್ಷ GGML_OP_RMS_NORM:
            ಪಕ್ಷ GGML_OP_ADD:
            ಪಕ್ಷ GGML_OP_SUB:
            ಪಕ್ಷ GGML_OP_MUL:
            ಪಕ್ಷ GGML_OP_DIV:
            ಪಕ್ಷ GGML_OP_FLASH_ATTN_EXT:
            ಪಕ್ಷ GGML_OP_CONT:
            ಪಕ್ಷ GGML_OP_CPY:
            ಪಕ್ಷ GGML_OP_REPEAT:
            ಪಕ್ಷ GGML_OP_SUM_ROWS:
            ಪಕ್ಷ GGML_OP_GET_ROWS:
            ಪಕ್ಷ GGML_OP_CONCAT:
                // case GGML_OP_GATED_DELTA_NET:
                ಪ್ರತ್ಯಾಗಮ (ggml::cpu::tensor_traits *) (&ggml::cpu::riscv64_spacemit::rvv_impl);
            ಸ್ವಭಾವ:
                // GGML_ABORT("fatal error");
                ಭಙ್ಗ;
        }

        ಪ್ರತ್ಯಾಗಮ nullptr;
    }
};

}  // namespace ggml::cpu::riscv64_spacemit

ggml_backend_buffer_type_t ggml_backend_cpu_riscv64_spacemit_buffer_type(ಶೂನ್ಯ) {
    ಸ್ಥಿರ ggml_backend_buffer_type ggml_backend_cpu_buffer_type_riscv64_spacemit = {
  /* .iface    = */
        {
         /* .get_name           = */ ggml_backend_cpu_riscv64_spacemit_buffer_type_get_name,
         /* .alloc_buffer       = */ ggml_backend_cpu_riscv64_spacemit_buffer_type_alloc_buffer,
         /* .alloc_buffer_n     = */ ರಿಕ್ತ,
         /* .get_alignment      = */ ggml_backend_cpu_riscv64_spacemit_buffer_type_get_alignment,
         /* .get_max_size       = */ nullptr,
         /* .get_alloc_size     = */ ggml_backend_cpu_riscv64_spacemit_nbytes,
         /* .get_alloc_size_n   = */ ರಿಕ್ತ,
         /* .is_host            = */ nullptr,
         },
 /* .device  = */
        ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
 /* .context = */
        ನೂತನ ggml::cpu::riscv64_spacemit::extra_buffer_type(),
    };

    ಪ್ರತ್ಯಾಗಮ &ggml_backend_cpu_buffer_type_riscv64_spacemit;
}

ಬಾಹ್ಯ "C" {
ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ bind_ai_thread() {
    ಪೂರ್ಣಾಙ್ಕ  fd, bytes;
    ವರ್ಣ str[32];

    fd = open("/proc/set_ai_thread", O_WRONLY);
    ಯದಿ (fd < 0) {
        GGML_LOG_ERROR("try open /proc/set_ai_thread failed\n");
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    snprintf(str, 16, "%d", 0);
    bytes = ಲಿಖ(fd, str, strlen(str));
    ಯದಿ (bytes < 0) {
        GGML_LOG_ERROR("try write /proc/set_ai_thread failed\n");
        close(fd);
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    close(fd);
    ಪ್ರತ್ಯಾಗಮ 0;
}

ಶೂನ್ಯ ggml_backend_cpu_riscv64_spacemit_set_numa_thread_affinity(ಪೂರ್ಣಾಙ್ಕ thread_n) {
    ಪೂರ್ಣಾಙ್ಕ cpu_id = sched_getcpu();
    ಯದಿ (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2 &&
        !((1 << cpu_id) & ggml::cpu::riscv64_spacemit::global_spine_env_info.cpu_mask)) {
        GGML_PRINT_DEBUG("bind_ai_thread for thread %d, pid %d\n", thread_n, getpid());
        bind_ai_thread();
    }

    ಯದಿ (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_tcm &&
        ggml::cpu::riscv64_spacemit::tls_context.cpu_id == -1) {
        CPU_ZERO(&(ggml::cpu::riscv64_spacemit::tls_context.cpuset));
        pthread_t    main_thread     = pthread_self();
        ಧ್ರುವ auto & perfer_core_ids = ggml::cpu::riscv64_spacemit::global_spine_env_info.perfer_core_ids;
        ಯದಿ (thread_n < 0 || static_cast<size_t>(thread_n) >= perfer_core_ids.size()) {
            GGML_ABORT("thread_n %d exceeds perfer_core_ids size %zu\n", thread_n, perfer_core_ids.size());
        }
        auto perfer_cpu_id = perfer_core_ids[static_cast<size_t>(thread_n)];
        CPU_SET(perfer_cpu_id, &(ggml::cpu::riscv64_spacemit::tls_context.cpuset));
        ಪೂರ್ಣಾಙ್ಕ s =
            pthread_setaffinity_np(main_thread, ಪರಿಮಾಣ(cpu_set_t), &(ggml::cpu::riscv64_spacemit::tls_context.cpuset));
        ಯದಿ (s != 0) {
            GGML_ABORT("set thread affinity error for thread_n %d, cpu_id %d\n", thread_n, perfer_cpu_id);
        }

        ಪೂರ್ಣಾಙ್ಕ ai_cpu_id = perfer_cpu_id - ggml::cpu::riscv64_spacemit::global_spine_env_info.aicpu_id_offset;
        ggml::cpu::riscv64_spacemit::tls_context.cpu_id = ai_cpu_id;
        ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer =
            ggml::cpu::riscv64_spacemit::spine_mem_pool_tcm_mem_get(ai_cpu_id);
        ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer_size =
            ggml::cpu::riscv64_spacemit::global_spine_env_info.tcm_blk_size;
    }

    ಯದಿ (ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer != nullptr) {
        ಶೂನ್ಯ * rt =
            ggml::cpu::riscv64_spacemit::spine_mem_pool_tcm_mem_wait(ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        ಯದಿ (rt == nullptr) {
            GGML_ABORT("wait tcm buffer failed for cpu_id: %d", ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        }
    }
}

ಶೂನ್ಯ ggml_backend_cpu_riscv64_spacemit_clear_numa_thread_affinity_threaded(ಪೂರ್ಣಾಙ್ಕ thread_n) {
    ಯದಿ (ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer != nullptr) {
        auto rt = ggml::cpu::riscv64_spacemit::spine_mem_pool_tcm_mem_release(
            ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        ಯದಿ (rt != 0) {
            GGML_ABORT("release tcm buffer failed for cpu_id: %d", ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        }
    }
}
}
