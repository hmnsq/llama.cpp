#ವ್ಯಾಖ್ಯಾನಿಸು GGML_COMMON_IMPL_CPP
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_COMMON_DECL_CPP

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
#ಆದರೆ defined(__riscv)

#ಆದರೆ !defined(__riscv_v) || !defined(__riscv_v_intrinsic)
#ದೋಷ "riscv v extension or v_intrinsic not enabled"
#ಇಲ್ಲದಿದ್ದರೆ
#ಅನ್ತರ್ಭಾವ <riscv_vector.h>
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ !defined(__riscv_zfh) || !defined(__riscv_zvfh)
#ದೋಷ "riscv zfh extension not enabled, GGML_RV_ZFH and GGML_RV_ZVFH must be defined to 1"
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ !defined(__riscv_zba)
#ದೋಷ "riscv zba extension not enabled, GGML_RV_ZBA must be defined to 1"
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(RISCV64_SPACEMIT_IME1) || defined(RISCV64_SPACEMIT_IME2)
#ಇಲ್ಲದಿದ್ದರೆ
#ದೋಷ "RISCV64_SPACEMIT_IME1 or RISCV64_SPACEMIT_IME2 not defined"
#ಆದರೆ_ಅಂತ್ಯ

#ಇಲ್ಲದಿದ್ದರೆ

#ದೋಷ "riscv not enabled in this build"

#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(__GNUC__)
#ಸೂಚನೆ GCC diagnostic ignored "-Woverlength-strings"
#ಸೂಚನೆ GCC diagnostic ignored "-Wcast-qual"
#ಸೂಚನೆ GCC diagnostic ignored "-Wunused-parameter"
#ಆದರೆ_ಅಂತ್ಯ

// clang-format on

ಬಾಹ್ಯ "C" {
ಬಾಹ್ಯ ಶೂನ್ಯ ggml_threadpool_chunk_ಗಣ(ರಚನೆ ggml_threadpool * tp, ಪೂರ್ಣಾಂಕ value);
ಬಾಹ್ಯ ಪೂರ್ಣಾಂಕ  ggml_threadpool_chunk_ಕೂಡಿಸು(ರಚನೆ ggml_threadpool * tp, ಪೂರ್ಣಾಂಕ value);
}

ನಾಮಾಕಾಶ ggml::cpu::riscv64_spacemit {

ರಚನೆ TLSContext {
    ಪೂರ್ಣಾಂಕ       cpu_id{ -1 };
    cpu_set_t cpuset;
    ಶೂನ್ಯ *    tcm_ಬಫರ್{ ಶೂನ್ಯಸೂಚಕ };
    size_t    tcm_ಬಫರ್_ಗಾತ್ರ{ 0 };
};

thread_local TLSContext tls_context;

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು BLOC_TYPE, int64_t INTER_ಗಾತ್ರ, int64_t NB_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ> constexpr size_t get_repacked_ಖಂಡ_type_ಗಾತ್ರ() {
    ಆದರೆ constexpr (std::is_same_v<BLOC_TYPE, ಖಂಡ_q6_K> || std::is_same_v<BLOC_TYPE, ಖಂಡ_q8_0>) {
        ಹಿಂತಿರುಗಿ ಗಾತ್ರ(ಖಂಡ_q8_0);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (std::is_same_v<BLOC_TYPE, ಖಂಡ_q4_0>) {
        ಹಿಂತಿರುಗಿ ಗಾತ್ರ(ಖಂಡ_q4_0) * INTER_ಗಾತ್ರ / QK4_0;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (std::is_same_v<BLOC_TYPE, ಖಂಡ_q4_1> || std::is_same_v<BLOC_TYPE, ಖಂಡ_q4_K>) {
        ಹಿಂತಿರುಗಿ (ಗಾತ್ರ(ಖಂಡ_q4_0) + ಗಾತ್ರ(uint8_t)) * INTER_ಗಾತ್ರ / QK4_1;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (std::is_same_v<BLOC_TYPE, ಖಂಡ_q2_K>) {
        ಹಿಂತಿರುಗಿ ಗಾತ್ರ(spacemit_kernels::nrow_ಖಂಡ_q2_k<1>);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (std::is_same_v<BLOC_TYPE, ಖಂಡ_q3_K>) {
        ಹಿಂತಿರುಗಿ ಗಾತ್ರ(spacemit_kernels::nrow_ಖಂಡ_q3_k<1>);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (std::is_same_v<BLOC_TYPE, ಖಂಡ_mxfp4>) {
        ಹಿಂತಿರುಗಿ ಗಾತ್ರ(spacemit_kernels::nrow_ಖಂಡ_mxfp4<1>);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (std::is_same_v<BLOC_TYPE, ಖಂಡ_q5_1> || std::is_same_v<BLOC_TYPE, ಖಂಡ_q5_K>) {
        ಹಿಂತಿರುಗಿ ಗಾತ್ರ(spacemit_kernels::nrow_ಖಂಡ_q5_1<1>);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (std::is_same_v<BLOC_TYPE, ಖಂಡ_q5_0>) {
        ಹಿಂತಿರುಗಿ ಗಾತ್ರ(spacemit_kernels::nrow_ಖಂಡ_q5_0<1>);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಪ್ರತಿಜ್ಞಾ(ಸುಳ್ಳು);
        ಹಿಂತಿರುಗಿ 0;
    }
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು BLOC_TYPE> constexpr ತಾರ್ಕಿಕ ಖಂಡ_type_has_zp() {
    ಆದರೆ constexpr (std::is_same_v<BLOC_TYPE, ಖಂಡ_q6_K> || std::is_same_v<BLOC_TYPE, ಖಂಡ_q8_0> ||
                  std::is_same_v<BLOC_TYPE, ಖಂಡ_q3_K> || std::is_same_v<BLOC_TYPE, ಖಂಡ_q4_0> ||
                  std::is_same_v<BLOC_TYPE, ಖಂಡ_mxfp4> || std::is_same_v<BLOC_TYPE, ಖಂಡ_q5_0>) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (std::is_same_v<BLOC_TYPE, ಖಂಡ_q4_1> || std::is_same_v<BLOC_TYPE, ಖಂಡ_q4_K> ||
                         std::is_same_v<BLOC_TYPE, ಖಂಡ_q2_K> || std::is_same_v<BLOC_TYPE, ಖಂಡ_q5_1> ||
                         std::is_same_v<BLOC_TYPE, ಖಂಡ_q5_K>) {
        ಹಿಂತಿರುಗಿ ನಿಜ;
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಪ್ರತಿಜ್ಞಾ(ಸುಳ್ಳು);
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
}

ವರ್ಗ tensor_traits_ತಳ : ಸಾರ್ವಜನಿಕ ggml::cpu::tensor_traits {
  ಸಾರ್ವಜನಿಕ:
    ವಾಸ್ತವಿಕ ಪೂರ್ಣಾಂಕ repack(ggml_tensor * t, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ದತ್ತಾಂಶ_ಗಾತ್ರ) = 0;
};

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು BLOC_TYPE, int64_t INTER_ಗಾತ್ರ, int64_t NB_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ> ವರ್ಗ tensor_traits : ಸಾರ್ವಜನಿಕ tensor_traits_ತಳ {
    ತಾರ್ಕಿಕ work_ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ /* n_threads */, ಸ್ಥಿರಾಂಕ ggml_tensor * op, size_t & size) override {
        ಆಯ್ಕೆ (op->op) {
            ಸ್ಥಿತಿ GGML_OP_MUL_MAT:
                {
                    int64_t src1_nelements = ggml_nelements(op->src[1]);

                    ಆದರೆ constexpr (std::is_same_v<BLOC_TYPE, ಖಂಡ_q2_K> || std::is_same_v<BLOC_TYPE, ಖಂಡ_q3_K>) {
                        size =
                            spacemit_kernels::div_ಸುತ್ತು_ಮೇಲೆ(src1_nelements, QK_K) * spacemit_kernels::q8k_blk_ಗಾತ್ರ(QK_K);
                    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (INTER_ಗಾತ್ರ == QK4_0) {
                        size = spacemit_kernels::div_ಸುತ್ತು_ಮೇಲೆ(src1_nelements, QK4_0) *
                               spacemit_kernels::q8_blk_ಗಾತ್ರ(QK4_0, ನಿಜ);
                    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (INTER_ಗಾತ್ರ == 256) {
                        size = spacemit_kernels::div_ಸುತ್ತು_ಮೇಲೆ(src1_nelements, 256) *
                               spacemit_kernels::q8_hp_blk_ಗಾತ್ರ(256, ನಿಜ, ನಿಜ);
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        GGML_ABORT("unsupported block type");
                    }

                    size = GGML_PAD(size, ಗಾತ್ರ(int64_t));

                    ಹಿಂತಿರುಗಿ ನಿಜ;
                }
            ಸ್ಥಿತಿ GGML_OP_MUL_MAT_ID:
                {
                    int64_t src1_nelements = ggml_nelements(op->src[1]);

                    ಆದರೆ constexpr (std::is_same_v<BLOC_TYPE, ಖಂಡ_q2_K> || std::is_same_v<BLOC_TYPE, ಖಂಡ_q3_K>) {
                        size =
                            spacemit_kernels::div_ಸುತ್ತು_ಮೇಲೆ(src1_nelements, QK_K) * spacemit_kernels::q8k_blk_ಗಾತ್ರ(QK_K);
                    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (INTER_ಗಾತ್ರ == QK4_0) {
                        size = spacemit_kernels::div_ಸುತ್ತು_ಮೇಲೆ(src1_nelements, QK4_0) *
                               spacemit_kernels::q8_blk_ಗಾತ್ರ(QK4_0, ನಿಜ);
                    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (INTER_ಗಾತ್ರ == 256) {
                        size = spacemit_kernels::div_ಸುತ್ತು_ಮೇಲೆ(src1_nelements, 256) *
                               spacemit_kernels::q8_hp_blk_ಗಾತ್ರ(256, ನಿಜ, ನಿಜ);
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        GGML_ABORT("unsupported block type");
                    }

                    size = GGML_PAD(size, ಗಾತ್ರ(int64_t));

                    ಸ್ಥಿರಾಂಕ int64_t ne02 = op->src[0]->ne[2];  // n_as, n_expert
                    ಸ್ಥಿರಾಂಕ int64_t ne12 = op->src[1]->ne[2];  // n_tokens

                    ಸ್ಥಿರಾಂಕ size_t sizeof_mmid_ಅಡ್ಡಸಾಲು_mapping = ಗಾತ್ರ(int64_t);
                    size += sizeof_mmid_ಅಡ್ಡಸಾಲು_mapping * ne02 * (ne12 + 1) + (ne02 + 1) * ಗಾತ್ರ(int64_t);

                    size = GGML_PAD(size, ಗಾತ್ರ(int64_t));

                    ಹಿಂತಿರುಗಿ ನಿಜ;
                }
            ಪೂರ್ವನಿಯೋಜಿತ:
                // GGML_ABORT("fatal error");
                ನಿಲ್ಲಿಸು;
        }
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ತಾರ್ಕಿಕ compute_forward(ggml_compute_params * params, ggml_tensor * op) override {
        ಆಯ್ಕೆ (op->op) {
            ಸ್ಥಿತಿ GGML_OP_MUL_MAT:
                ಆಯ್ಕೆ (op->src[0]->type) {
                    ಸ್ಥಿತಿ GGML_TYPE_Q2_K:
                    ಸ್ಥಿತಿ GGML_TYPE_Q3_K:
                    ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
                    ಸ್ಥಿತಿ GGML_TYPE_Q4_1:
                    ಸ್ಥಿತಿ GGML_TYPE_Q4_K:
                    ಸ್ಥಿತಿ GGML_TYPE_Q6_K:
                    ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
                    ಸ್ಥಿತಿ GGML_TYPE_Q5_0:
                    ಸ್ಥಿತಿ GGML_TYPE_Q5_1:
                    ಸ್ಥಿತಿ GGML_TYPE_Q5_K:
                        //case GGML_TYPE_MXFP4:
                        forward_mul_mat(params, op);
                        ಹಿಂತಿರುಗಿ ನಿಜ;
                    ಪೂರ್ವನಿಯೋಜಿತ:
                        // GGML_ABORT("fatal error: unsupported type for src0 in MUL_MAT");
                        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_OP_MUL_MAT_ID:
                ಆಯ್ಕೆ (op->src[0]->type) {
                    ಸ್ಥಿತಿ GGML_TYPE_Q2_K:
                    ಸ್ಥಿತಿ GGML_TYPE_Q3_K:
                    ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
                    ಸ್ಥಿತಿ GGML_TYPE_Q4_1:
                    ಸ್ಥಿತಿ GGML_TYPE_Q4_K:
                    ಸ್ಥಿತಿ GGML_TYPE_Q6_K:
                    ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
                    ಸ್ಥಿತಿ GGML_TYPE_Q5_0:
                    ಸ್ಥಿತಿ GGML_TYPE_Q5_1:
                    ಸ್ಥಿತಿ GGML_TYPE_Q5_K:
                        //case GGML_TYPE_MXFP4:
                        forward_mul_mat_id(params, op);
                        ಹಿಂತಿರುಗಿ ನಿಜ;
                    ಪೂರ್ವನಿಯೋಜಿತ:
                        // GGML_ABORT("fatal error: unsupported type for src0 in MUL_MAT_ID");
                        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }
                ನಿಲ್ಲಿಸು;
            ಪೂರ್ವನಿಯೋಜಿತ:
                // GGML_ABORT("fatal error");
                ನಿಲ್ಲಿಸು;
        }
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಶೂನ್ಯ forward_mul_mat(ggml_compute_params * params, ggml_tensor * op) {
        constexpr size_t a_blk_len = INTER_ಗಾತ್ರ;
        constexpr size_t b_blk_len = INTER_ಗಾತ್ರ;

        ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = op->src[0];
        ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = op->src[1];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS

        ಪೂರ್ಣಾಂಕ ith = params->ith;
        ಪೂರ್ಣಾಂಕ nth = params->nth;

        [[maybe_unused]] ಸ್ಥಿರಾಂಕ ಎಣಿಕೆ ggml_type type = src0->type;

        ಶೂನ್ಯ *        w_ದತ್ತಾಂಶ  = (ಶೂನ್ಯ *) src0->ದತ್ತಾಂಶ;
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * feature = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src1->ದತ್ತಾಂಶ;
        ದಶಮಾಂಶ *       output  = (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ;

        ಸ್ಥಿರಾಂಕ int64_t gemm_m = ne11 * ne12 * ne13;
        ಸ್ಥಿರಾಂಕ int64_t gemm_k = ne10;
        ಸ್ಥಿರಾಂಕ int64_t gemm_n = ne01;

        spacemit_kernels::quantize_a_ಅಡ್ಡಸಾಲು_def       quantize_a_ಅಡ್ಡಸಾಲು_i8;
        spacemit_kernels::quantize_a_ಅಡ್ಡಸಾಲು_def       quantize_a_4row_i8;
        spacemit_kernels::gemm_kernel_quantize_def gemm_kernel;
        ತಾರ್ಕಿಕ                                       ಗಣ_kernel_impl = ಸುಳ್ಳು;

        int64_t ಖಂಡ_stride_a = spacemit_kernels::q8_blk_ಗಾತ್ರ(a_blk_len);

#ಆದರೆ defined(RISCV64_SPACEMIT_IME2)
        ಆದರೆ (!ಗಣ_kernel_impl && (global_spine_env_info.use_ime2)) {
            quantize_a_ಅಡ್ಡಸಾಲು_i8  = spacemit_kernels::rvv::quantize_a_ಅಡ್ಡಸಾಲು_i8;
            quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8;
            ಖಂಡ_stride_a     = spacemit_kernels::q8_blk_ಗಾತ್ರ(a_blk_len, ನಿಜ);

            ಆದರೆ constexpr (std::is_same_v<BLOC_TYPE, ಖಂಡ_q6_K> || std::is_same_v<BLOC_TYPE, ಖಂಡ_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i8;
                ಗಣ_kernel_impl = ನಿಜ;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (std::is_same_v<BLOC_TYPE, ಖಂಡ_q4_0> || std::is_same_v<BLOC_TYPE, ಖಂಡ_q4_1> ||
                                 std::is_same_v<BLOC_TYPE, ಖಂಡ_q4_K>) {
                ಆದರೆ constexpr (INTER_ಗಾತ್ರ == 256) {
                    gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i4_hp;
                    quantize_a_ಅಡ್ಡಸಾಲು_i8  = spacemit_kernels::rvv::quantize_a_ಅಡ್ಡಸಾಲು_i8_hp;
                    quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8_hp;
                    ಖಂಡ_stride_a     = spacemit_kernels::q8_hp_blk_ಗಾತ್ರ(a_blk_len, ನಿಜ, ನಿಜ);
                    ಗಣ_kernel_impl    = ನಿಜ;
                } ಇಲ್ಲದಿದ್ದರೆ {
                    gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i4;
                    quantize_a_ಅಡ್ಡಸಾಲು_i8  = spacemit_kernels::rvv::quantize_a_ಅಡ್ಡಸಾಲು_i8;
                    quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8;
                    ಖಂಡ_stride_a     = spacemit_kernels::q8_blk_ಗಾತ್ರ(a_blk_len, ನಿಜ);
                    ಗಣ_kernel_impl    = ನಿಜ;
                }
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (std::is_same_v<BLOC_TYPE, ಖಂಡ_q2_K>) {
                quantize_a_ಅಡ್ಡಸಾಲು_i8  = spacemit_kernels::rvv::quantize_a_ಅಡ್ಡಸಾಲು_i8k;
                quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8k;
                ಖಂಡ_stride_a     = spacemit_kernels::q8k_blk_ಗಾತ್ರ(a_blk_len);

                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i2k;
                ಗಣ_kernel_impl = ನಿಜ;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (std::is_same_v<BLOC_TYPE, ಖಂಡ_q3_K>) {
                quantize_a_ಅಡ್ಡಸಾಲು_i8  = spacemit_kernels::rvv::quantize_a_ಅಡ್ಡಸಾಲು_i8k;
                quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8k;
                ಖಂಡ_stride_a     = spacemit_kernels::q8k_blk_ಗಾತ್ರ(a_blk_len);

                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i3k;
                ಗಣ_kernel_impl = ನಿಜ;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (std::is_same_v<BLOC_TYPE, ಖಂಡ_mxfp4>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8mxfp4;
                ಗಣ_kernel_impl = ನಿಜ;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (std::is_same_v<BLOC_TYPE, ಖಂಡ_q5_1> || std::is_same_v<BLOC_TYPE, ಖಂಡ_q5_K> ||
                                 std::is_same_v<BLOC_TYPE, ಖಂಡ_q5_0>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i5;
                ಗಣ_kernel_impl = ನಿಜ;
            }
        }
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(RISCV64_SPACEMIT_IME1)
        ಆದರೆ (!ಗಣ_kernel_impl && (global_spine_env_info.use_ime1)) {
            quantize_a_ಅಡ್ಡಸಾಲು_i8  = spacemit_kernels::ime1::quantize_a_ಅಡ್ಡಸಾಲು_i8;
            quantize_a_4row_i8 = spacemit_kernels::ime1::quantize_a_4row_i8;

            ಆದರೆ constexpr (std::is_same_v<BLOC_TYPE, ಖಂಡ_q4_0> || std::is_same_v<BLOC_TYPE, ಖಂಡ_q4_1> ||
                          std::is_same_v<BLOC_TYPE, ಖಂಡ_q4_K>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i4;
                ಗಣ_kernel_impl = ನಿಜ;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (std::is_same_v<BLOC_TYPE, ಖಂಡ_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i8;
                ಗಣ_kernel_impl = ನಿಜ;
            }
        }
#ಆದರೆ_ಅಂತ್ಯ
        ಆದರೆ (!ಗಣ_kernel_impl) {
            GGML_ABORT("no kernel implementation found for the block type");
        }

        ಸ್ಥಿರಾಂಕ int64_t a_k_blks = spacemit_kernels::div_ಸುತ್ತು_ಮೇಲೆ(gemm_k, a_blk_len);
        ಸ್ಥಿರಾಂಕ int64_t b_k_blks = spacemit_kernels::div_ಸುತ್ತು_ಮೇಲೆ(gemm_k, b_blk_len);

        ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲು_stride_a        = a_k_blks * ಖಂಡ_stride_a;
        ಸ್ಥಿರಾಂಕ int64_t gemm_workspace_ಗಾತ್ರ = GGML_PAD(gemm_m * ಅಡ್ಡಸಾಲು_stride_a, alignof(int64_t));

        ಆದರೆ (ith == 0 && params->wsize < gemm_workspace_ಗಾತ್ರ) {
            GGML_ABORT("wsize less than gemm_workspace_size");
        }

        uintptr_t ws_ptr = reinterpret_cast<uintptr_t>(params->wdata);

        ಶೂನ್ಯ *        tcm_ಬಫರ್      = ggml::cpu::riscv64_spacemit::tls_context.tcm_ಬಫರ್;
        ಸ್ಥಿರಾಂಕ int64_t tcm_ಬಫರ್_ಗಾತ್ರ = ggml::cpu::riscv64_spacemit::tls_context.tcm_ಬಫರ್_ಗಾತ್ರ;

        ಸ್ವಯಂಚಾಲಿತ * quant_a_ಬಫರ್ = reinterpret_cast<uint8_t *>(ws_ptr);

        constexpr int64_t ಅಡ್ಡಸಾಲು_align = 4;
        ಸ್ಥಿರಾಂಕ int64_t     ಅಡ್ಡಸಾಲು_blks  = spacemit_kernels::div_ಸುತ್ತು_ಮೇಲೆ(gemm_m, ಅಡ್ಡಸಾಲು_align);

        ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲು_stride_b      = b_k_blks * get_repacked_ಖಂಡ_type_ಗಾತ್ರ<BLOC_TYPE, INTER_ಗಾತ್ರ, NB_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ>();
        ಸ್ಥಿರಾಂಕ int64_t ಪ್ರತಿ_mb_ಅಡ್ಡಸಾಲುಗಳು_wsize = ಅಡ್ಡಸಾಲು_align * ಅಡ್ಡಸಾಲು_stride_a;
        ಸ್ಥಿರಾಂಕ int64_t ಪ್ರತಿ_nb_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_wsize = NB_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ * ಅಡ್ಡಸಾಲು_stride_b;

        ಸ್ಥಿರಾಂಕ int64_t barrier_idx = static_cast<int64_t>(ith / 2);

        GGML_ASSERT(global_spine_env_info.init_barrier != ಶೂನ್ಯಸೂಚಕ);
        GGML_ASSERT(barrier_idx < spine_init_barrier_ಎಣಿಕೆ);
        spine_barrier_t * cur_barrier = &global_spine_env_info.init_barrier[barrier_idx];

        ಆದರೆ (gemm_m == 1) {
            ಪೂರ್ಣಾಂಕ task_ಪ್ರತಿ_thread = spacemit_kernels::div_ಸುತ್ತು_ಮೇಲೆ(a_k_blks, nth);
            ಪೂರ್ಣಾಂಕ a_blk_ಆರಂಭ     = ith * task_ಪ್ರತಿ_thread;
            ಪೂರ್ಣಾಂಕ a_blk_ಕೊನೆ       = std::min(a_blk_ಆರಂಭ + task_ಪ್ರತಿ_thread, (ಪೂರ್ಣಾಂಕ) a_k_blks);
            ಆದರೆ (a_blk_ಆರಂಭ < a_blk_ಕೊನೆ) {
                quantize_a_ಅಡ್ಡಸಾಲು_i8(a_blk_len, feature + a_blk_ಆರಂಭ * a_blk_len, (a_blk_ಕೊನೆ - a_blk_ಆರಂಭ) * a_blk_len,
                                  quant_a_ಬಫರ್ + a_blk_ಆರಂಭ * ಖಂಡ_stride_a);
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಪೂರ್ಣಾಂಕ task_ಪ್ರತಿ_thread = spacemit_kernels::div_ಸುತ್ತು_ಮೇಲೆ(ಅಡ್ಡಸಾಲು_blks, nth);
            ಪೂರ್ಣಾಂಕ m_ಅಡ್ಡಸಾಲು_blk_ಆರಂಭ = ith * task_ಪ್ರತಿ_thread;
            ಪೂರ್ಣಾಂಕ m_ಅಡ್ಡಸಾಲು_blk_ಕೊನೆ   = std::min(m_ಅಡ್ಡಸಾಲು_blk_ಆರಂಭ + task_ಪ್ರತಿ_thread, (ಪೂರ್ಣಾಂಕ) ಅಡ್ಡಸಾಲು_blks);
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ m_ಅಡ್ಡಸಾಲು_blk = m_ಅಡ್ಡಸಾಲು_blk_ಆರಂಭ; m_ಅಡ್ಡಸಾಲು_blk < m_ಅಡ್ಡಸಾಲು_blk_ಕೊನೆ; m_ಅಡ್ಡಸಾಲು_blk++) {
                ಪೂರ್ಣಾಂಕ m_idx             = m_ಅಡ್ಡಸಾಲು_blk * ಅಡ್ಡಸಾಲು_align;
                ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲುಗಳು_tobe_handled = (gemm_m - m_idx) > ಅಡ್ಡಸಾಲು_align ? ಅಡ್ಡಸಾಲು_align : (gemm_m - m_idx);

                ಆದರೆ (ಅಡ್ಡಸಾಲುಗಳು_tobe_handled == ಅಡ್ಡಸಾಲು_align && quantize_a_4row_i8 != ಶೂನ್ಯಸೂಚಕ) {
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * a_ಅಡ್ಡಸಾಲು_ptr       = feature + m_idx * gemm_k;
                    ಸ್ವಯಂಚಾಲಿತ *        quant_a_ಅಡ್ಡಸಾಲು_ptr = quant_a_ಬಫರ್ + m_idx * ಅಡ್ಡಸಾಲು_stride_a;
                    quantize_a_4row_i8(a_blk_len, a_ಅಡ್ಡಸಾಲು_ptr, gemm_k, quant_a_ಅಡ್ಡಸಾಲು_ptr);
                } ಇಲ್ಲದಿದ್ದರೆ {
                    ಆಗ (ಅಡ್ಡಸಾಲುಗಳು_tobe_handled) {
                        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * a_ಅಡ್ಡಸಾಲು_ptr       = feature + m_idx * gemm_k;
                        ಸ್ವಯಂಚಾಲಿತ *        quant_a_ಅಡ್ಡಸಾಲು_ptr = quant_a_ಬಫರ್ + m_idx * ಅಡ್ಡಸಾಲು_stride_a;
                        quantize_a_ಅಡ್ಡಸಾಲು_i8(a_blk_len, a_ಅಡ್ಡಸಾಲು_ptr, gemm_k, quant_a_ಅಡ್ಡಸಾಲು_ptr);
                        ಅಡ್ಡಸಾಲುಗಳು_tobe_handled -= 1;
                        m_idx += 1;
                    }
                }
            }
        }

        ggml_barrier(params->threadpool);

        ಸ್ಥಿರಾಂಕ int64_t gemm_m_stride     = gemm_n / gemm_m > 64 ? gemm_m : 16;
        ಸ್ಥಿರಾಂಕ int64_t gemm_m_blocked    = spacemit_kernels::div_ಸುತ್ತು_ಮೇಲೆ(gemm_m, gemm_m_stride);
        ಸ್ಥಿರಾಂಕ int64_t max_gemm_n_stride = spacemit_kernels::div_ಸುತ್ತು_ಮೇಲೆ(gemm_n * gemm_m_blocked, nth);

        int64_t gemm_n_stride = gemm_n;
        ಆದರೆ (max_gemm_n_stride < gemm_n) {
            gemm_n_stride =
                std::min(gemm_n_stride, spacemit_kernels::div_ಸುತ್ತು_ಮೇಲೆ(max_gemm_n_stride, NB_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ) * NB_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ);
        }

        ಆದರೆ (gemm_n_stride == gemm_n && tcm_ಬಫರ್ != ಶೂನ್ಯಸೂಚಕ && ಪ್ರತಿ_mb_ಅಡ್ಡಸಾಲುಗಳು_wsize <= tcm_ಬಫರ್_ಗಾತ್ರ) {
            ಗಾಗಿ (int64_t m_ಆರಂಭ = ith * ಅಡ್ಡಸಾಲು_align; m_ಆರಂಭ < gemm_m; m_ಆರಂಭ += ಅಡ್ಡಸಾಲು_align * nth) {
                uint8_t * b_ಲಂಬಸಾಲು    = reinterpret_cast<uint8_t *>(w_ದತ್ತಾಂಶ);
                uint8_t * b_ಲಂಬಸಾಲು_zp = ಖಂಡ_type_has_zp<BLOC_TYPE>() ? b_ಲಂಬಸಾಲು : ಶೂನ್ಯಸೂಚಕ;

                int64_t m_ಅಡ್ಡಸಾಲು_real = std::min(gemm_m - m_ಆರಂಭ, ಅಡ್ಡಸಾಲು_align);

                spacemit_kernels::rvv::memcpy1d(tcm_ಬಫರ್, quant_a_ಬಫರ್ + m_ಆರಂಭ * ಅಡ್ಡಸಾಲು_stride_a,
                                                m_ಅಡ್ಡಸಾಲು_real * ಅಡ್ಡಸಾಲು_stride_a);

                int64_t n_blk_real = 0;
                ಗಾಗಿ (int64_t ni = 0; ni < gemm_n; ni += n_blk_real, b_ಲಂಬಸಾಲು += n_blk_real * ಅಡ್ಡಸಾಲು_stride_b) {
                    n_blk_real = std::min(gemm_n - ni, (int64_t) NB_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ);

                    uint8_t * a_ಅಡ್ಡಸಾಲು_ptr = (uint8_t *) tcm_ಬಫರ್;
                    ದಶಮಾಂಶ *   c_blk     = output + m_ಆರಂಭ * gemm_n + ni;

                    int32_t ಅಡ್ಡಸಾಲುಗಳು_ಉಳಿದ = m_ಅಡ್ಡಸಾಲು_real;

                    ಆಗ (ಅಡ್ಡಸಾಲುಗಳು_ಉಳಿದ > 0) {
                        ಸ್ವಯಂಚಾಲಿತ ಅಡ್ಡಸಾಲುಗಳು_handled = gemm_kernel(b_blk_len, a_ಅಡ್ಡಸಾಲು_ptr, b_ಲಂಬಸಾಲು, b_ಲಂಬಸಾಲು_zp, c_blk, ಅಡ್ಡಸಾಲುಗಳು_ಉಳಿದ,
                                                        n_blk_real, b_k_blks, gemm_n);

                        c_blk += ಅಡ್ಡಸಾಲುಗಳು_handled * gemm_n;
                        a_ಅಡ್ಡಸಾಲು_ptr += ಅಡ್ಡಸಾಲುಗಳು_handled * ಅಡ್ಡಸಾಲು_stride_a;

                        ಅಡ್ಡಸಾಲುಗಳು_ಉಳಿದ -= ಅಡ್ಡಸಾಲುಗಳು_handled;
                    }
                }
            }
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tcm_ಬಫರ್ != ಶೂನ್ಯಸೂಚಕ && ಪ್ರತಿ_nb_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_wsize <= tcm_ಬಫರ್_ಗಾತ್ರ) {
            uint8_t * a_ಅಡ್ಡಸಾಲು = quant_a_ಬಫರ್;
            uint8_t * b_ಲಂಬಸಾಲು = reinterpret_cast<uint8_t *>(tcm_ಬಫರ್);
            ಆದರೆ ((gemm_workspace_ಗಾತ್ರ + ಪ್ರತಿ_nb_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_wsize) <= tcm_ಬಫರ್_ಗಾತ್ರ) {
                a_ಅಡ್ಡಸಾಲು = (uint8_t *) tcm_ಬಫರ್;
                b_ಲಂಬಸಾಲು = reinterpret_cast<uint8_t *>(tcm_ಬಫರ್) + gemm_workspace_ಗಾತ್ರ;
            }
            uint8_t * b_ಲಂಬಸಾಲು_zp = ಖಂಡ_type_has_zp<BLOC_TYPE>() ? b_ಲಂಬಸಾಲು : ಶೂನ್ಯಸೂಚಕ;

            int64_t ni      = ith * NB_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;
            int64_t nb_real = std::min(gemm_n - ni, NB_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ);

            ಆದರೆ (ith % 2 == 0 && nb_real > 0) {
                spacemit_kernels::rvv::memcpy1d(b_ಲಂಬಸಾಲು, reinterpret_cast<uint8_t *>(w_ದತ್ತಾಂಶ) + ni * ಅಡ್ಡಸಾಲು_stride_b,
                                                nb_real * ಅಡ್ಡಸಾಲು_stride_b);
                ಆದರೆ (a_ಅಡ್ಡಸಾಲು != quant_a_ಬಫರ್) {
                    spacemit_kernels::rvv::memcpy1d(a_ಅಡ್ಡಸಾಲು, quant_a_ಬಫರ್, gemm_workspace_ಗಾತ್ರ);
                }
            }

            spine_barrier_wait(cur_barrier);

            ಆದರೆ (ith % 2 != 0 && nb_real > 0) {
                ಆದರೆ (a_ಅಡ್ಡಸಾಲು != quant_a_ಬಫರ್) {
                    spacemit_kernels::rvv::memcpy1d(a_ಅಡ್ಡಸಾಲು, quant_a_ಬಫರ್, gemm_workspace_ಗಾತ್ರ);
                }
                spacemit_kernels::rvv::memcpy1d(b_ಲಂಬಸಾಲು, reinterpret_cast<uint8_t *>(w_ದತ್ತಾಂಶ) + ni * ಅಡ್ಡಸಾಲು_stride_b,
                                                nb_real * ಅಡ್ಡಸಾಲು_stride_b);
            }

            ಗಾಗಿ (; ni < gemm_n; ni += NB_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ * nth) {
                int64_t ಅಡ್ಡಸಾಲುಗಳು_ಉಳಿದ = gemm_m;
                ದಶಮಾಂಶ * c_blk          = output + ni;
                ಸ್ವಯಂಚಾಲಿತ *  a_ಅಡ್ಡಸಾಲು_cur      = a_ಅಡ್ಡಸಾಲು;

                ಆದರೆ (ith % 2 != 0) {
                    spine_barrier_wait(cur_barrier);
                }

                ಆಗ (ಅಡ್ಡಸಾಲುಗಳು_ಉಳಿದ > 0) {
                    ಸ್ವಯಂಚಾಲಿತ ಅಡ್ಡಸಾಲುಗಳು_handled = gemm_kernel(b_blk_len, a_ಅಡ್ಡಸಾಲು_cur, b_ಲಂಬಸಾಲು, b_ಲಂಬಸಾಲು_zp, c_blk, ಅಡ್ಡಸಾಲುಗಳು_ಉಳಿದ,
                                                    nb_real, b_k_blks, gemm_n);

                    c_blk += ಅಡ್ಡಸಾಲುಗಳು_handled * gemm_n;
                    a_ಅಡ್ಡಸಾಲು_cur += ಅಡ್ಡಸಾಲುಗಳು_handled * ಅಡ್ಡಸಾಲು_stride_a;

                    ಅಡ್ಡಸಾಲುಗಳು_ಉಳಿದ -= ಅಡ್ಡಸಾಲುಗಳು_handled;
                }

                ಆದರೆ (ith % 2 == 0) {
                    spine_barrier_wait(cur_barrier);
                }

                ಸ್ಥಿರಾಂಕ int64_t ಮುಂದಿನ_ni = ni + NB_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ * nth;
                ಆದರೆ (ಮುಂದಿನ_ni < gemm_n) {
                    nb_real = std::min(gemm_n - ಮುಂದಿನ_ni, NB_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ);
                    spacemit_kernels::rvv::memcpy1d(b_ಲಂಬಸಾಲು, reinterpret_cast<uint8_t *>(w_ದತ್ತಾಂಶ) + ಮುಂದಿನ_ni * ಅಡ್ಡಸಾಲು_stride_b,
                                                    nb_real * ಅಡ್ಡಸಾಲು_stride_b);
                }
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಸ್ಥಿರಾಂಕ int64_t task_ಎಣಿಕೆ_m = spacemit_kernels::div_ಸುತ್ತು_ಮೇಲೆ(gemm_m, gemm_m_stride);
            ಸ್ಥಿರಾಂಕ int64_t task_ಎಣಿಕೆ_n = spacemit_kernels::div_ಸುತ್ತು_ಮೇಲೆ(gemm_n, gemm_n_stride);

            int64_t task_ಎಣಿಕೆ      = task_ಎಣಿಕೆ_m * task_ಎಣಿಕೆ_n;
            int64_t task_ಪ್ರತಿ_thread = (task_ಎಣಿಕೆ + nth - 1) / nth;
            int64_t start           = ith * task_ಪ್ರತಿ_thread;
            int64_t end             = std::min((ith + 1) * task_ಪ್ರತಿ_thread, task_ಎಣಿಕೆ);
            ಗಾಗಿ (int64_t compute_idx = start; compute_idx < end; compute_idx++) {
                ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ tid_n = compute_idx / task_ಎಣಿಕೆ_m;
                ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ tid_m = compute_idx % task_ಎಣಿಕೆ_m;

                ಸ್ಥಿರಾಂಕ int64_t m_ಆರಂಭ = tid_m * gemm_m_stride;
                ಸ್ಥಿರಾಂಕ int64_t m_ಎಣಿಕೆ = std::min(gemm_m - m_ಆರಂಭ, (int64_t) gemm_m_stride);

                ಸ್ಥಿರಾಂಕ int64_t n_ಆರಂಭ = tid_n * gemm_n_stride;
                ಸ್ಥಿರಾಂಕ int64_t n_ಎಣಿಕೆ = std::min(gemm_n - n_ಆರಂಭ, (int64_t) gemm_n_stride);

                ಸ್ಥಿರಾಂಕ int64_t n_blk = m_ಎಣಿಕೆ == 1 ? n_ಎಣಿಕೆ : NB_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;

                uint8_t * b_ಲಂಬಸಾಲು    = reinterpret_cast<uint8_t *>(w_ದತ್ತಾಂಶ) + n_ಆರಂಭ * ಅಡ್ಡಸಾಲು_stride_b;
                uint8_t * b_ಲಂಬಸಾಲು_zp = ಖಂಡ_type_has_zp<BLOC_TYPE>() ? b_ಲಂಬಸಾಲು : ಶೂನ್ಯಸೂಚಕ;

                int64_t n_blk_real = 0;
                ಗಾಗಿ (int64_t ni = 0; ni < n_ಎಣಿಕೆ; ni += n_blk_real, b_ಲಂಬಸಾಲು += n_blk_real * ಅಡ್ಡಸಾಲು_stride_b) {
                    n_blk_real = std::min(n_ಎಣಿಕೆ - ni, n_blk);

                    uint8_t * a_ಅಡ್ಡಸಾಲು = quant_a_ಬಫರ್ + m_ಆರಂಭ * ಅಡ್ಡಸಾಲು_stride_a;

                    ದಶಮಾಂಶ * c_blk = output + m_ಆರಂಭ * gemm_n + n_ಆರಂಭ + ni;

                    int64_t ಅಡ್ಡಸಾಲುಗಳು_ಉಳಿದ = m_ಎಣಿಕೆ;

                    uint8_t * b_ಲಂಬಸಾಲು_cur    = b_ಲಂಬಸಾಲು;
                    uint8_t * b_ಲಂಬಸಾಲು_zp_cur = b_ಲಂಬಸಾಲು_zp;

                    ಆಗ (ಅಡ್ಡಸಾಲುಗಳು_ಉಳಿದ > 0) {
                        ಸ್ವಯಂಚಾಲಿತ ಅಡ್ಡಸಾಲುಗಳು_handled = gemm_kernel(b_blk_len, a_ಅಡ್ಡಸಾಲು, b_ಲಂಬಸಾಲು_cur, b_ಲಂಬಸಾಲು_zp_cur, c_blk,
                                                        ಅಡ್ಡಸಾಲುಗಳು_ಉಳಿದ, n_blk_real, b_k_blks, gemm_n);

                        c_blk += ಅಡ್ಡಸಾಲುಗಳು_handled * gemm_n;
                        a_ಅಡ್ಡಸಾಲು += ಅಡ್ಡಸಾಲುಗಳು_handled * ಅಡ್ಡಸಾಲು_stride_a;

                        ಅಡ್ಡಸಾಲುಗಳು_ಉಳಿದ -= ಅಡ್ಡಸಾಲುಗಳು_handled;
                    }
                }
            }
        }
    }

    ಶೂನ್ಯ forward_mul_mat_id(ggml_compute_params * params, ggml_tensor * op) {
        constexpr size_t a_blk_len = INTER_ಗಾತ್ರ;
        constexpr size_t b_blk_len = INTER_ಗಾತ್ರ;

        ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = op->src[0];
        ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = op->src[1];
        ಸ್ಥಿರಾಂಕ ggml_tensor * ids  = op->src[2];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS

        ಪೂರ್ಣಾಂಕ ith = params->ith;
        ಪೂರ್ಣಾಂಕ nth = params->nth;

        // row groups
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_ids = ids->ne[0];  // n_expert_used
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_as  = ne02;        // n_expert

        ರಚನೆ mmid_ಅಡ್ಡಸಾಲು_mapping {
            int32_t i1;
            int32_t i2;
        };

        spacemit_kernels::quantize_a_ಅಡ್ಡಸಾಲು_def           quantize_a_ಅಡ್ಡಸಾಲು_i8;
        spacemit_kernels::gemm_kernel_quantize_def     gemm_kernel;
        spacemit_kernels::moe_gemm_kernel_quantize_def moe_gemm_kernel_m2;
        ತಾರ್ಕಿಕ                                           ಗಣ_kernel_impl = ಸುಳ್ಳು;
        size_t                                         ಖಂಡ_stride_a  = spacemit_kernels::q8_blk_ಗಾತ್ರ(QK4_0);

#ಆದರೆ defined(RISCV64_SPACEMIT_IME2)
        ಆದರೆ (!ಗಣ_kernel_impl && (global_spine_env_info.use_ime2)) {
            quantize_a_ಅಡ್ಡಸಾಲು_i8 = spacemit_kernels::rvv::quantize_a_ಅಡ್ಡಸಾಲು_i8;
            ಖಂಡ_stride_a    = spacemit_kernels::q8_blk_ಗಾತ್ರ(QK4_0, ನಿಜ);

            ಆದರೆ constexpr (std::is_same_v<BLOC_TYPE, ಖಂಡ_q6_K> || std::is_same_v<BLOC_TYPE, ಖಂಡ_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i8;
                ಗಣ_kernel_impl = ನಿಜ;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (std::is_same_v<BLOC_TYPE, ಖಂಡ_q4_0> || std::is_same_v<BLOC_TYPE, ಖಂಡ_q4_1> ||
                                 std::is_same_v<BLOC_TYPE, ಖಂಡ_q4_K>) {
                ಆದರೆ constexpr (INTER_ಗಾತ್ರ == 256) {
                    gemm_kernel       = spacemit_kernels::ime2::gemm_kernel_i8i4_hp;
                    quantize_a_ಅಡ್ಡಸಾಲು_i8 = spacemit_kernels::rvv::quantize_a_ಅಡ್ಡಸಾಲು_i8_hp;
                    ಖಂಡ_stride_a    = spacemit_kernels::q8_hp_blk_ಗಾತ್ರ(a_blk_len, ನಿಜ, ನಿಜ);
                    ಗಣ_kernel_impl   = ನಿಜ;
                } ಇಲ್ಲದಿದ್ದರೆ {
                    gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i4;
                    moe_gemm_kernel_m2 = spacemit_kernels::ime2::moe_m2_gemm_kernel_i8i4;
                    quantize_a_ಅಡ್ಡಸಾಲು_i8  = spacemit_kernels::rvv::quantize_a_ಅಡ್ಡಸಾಲು_i8;
                    ಖಂಡ_stride_a     = spacemit_kernels::q8_blk_ಗಾತ್ರ(a_blk_len, ನಿಜ);
                    ಗಣ_kernel_impl    = ನಿಜ;
                }
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (std::is_same_v<BLOC_TYPE, ಖಂಡ_q2_K>) {
                quantize_a_ಅಡ್ಡಸಾಲು_i8 = spacemit_kernels::rvv::quantize_a_ಅಡ್ಡಸಾಲು_i8k;
                ಖಂಡ_stride_a    = spacemit_kernels::q8k_blk_ಗಾತ್ರ(a_blk_len);
                gemm_kernel       = spacemit_kernels::ime2::gemm_kernel_i8i2k;
                ಗಣ_kernel_impl   = ನಿಜ;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (std::is_same_v<BLOC_TYPE, ಖಂಡ_q3_K>) {
                quantize_a_ಅಡ್ಡಸಾಲು_i8 = spacemit_kernels::rvv::quantize_a_ಅಡ್ಡಸಾಲು_i8k;
                ಖಂಡ_stride_a    = spacemit_kernels::q8k_blk_ಗಾತ್ರ(a_blk_len);
                gemm_kernel       = spacemit_kernels::ime2::gemm_kernel_i8i3k;
                ಗಣ_kernel_impl   = ನಿಜ;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (std::is_same_v<BLOC_TYPE, ಖಂಡ_mxfp4>) {
                gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8mxfp4;
                moe_gemm_kernel_m2 = spacemit_kernels::ime2::moe_m2_gemm_kernel_i8mxfp4;
                ಗಣ_kernel_impl    = ನಿಜ;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (std::is_same_v<BLOC_TYPE, ಖಂಡ_q5_1> || std::is_same_v<BLOC_TYPE, ಖಂಡ_q5_K> ||
                                 std::is_same_v<BLOC_TYPE, ಖಂಡ_q5_0>) {
                gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i5;
                moe_gemm_kernel_m2 = spacemit_kernels::ime2::moe_m2_gemm_kernel_i8i5;
                ಗಣ_kernel_impl    = ನಿಜ;
            }
        }
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(RISCV64_SPACEMIT_IME1)
        ಆದರೆ (!ಗಣ_kernel_impl && (global_spine_env_info.use_ime1)) {
            quantize_a_ಅಡ್ಡಸಾಲು_i8 = spacemit_kernels::ime1::quantize_a_ಅಡ್ಡಸಾಲು_i8;

            ಆದರೆ constexpr (std::is_same_v<BLOC_TYPE, ಖಂಡ_q4_0> || std::is_same_v<BLOC_TYPE, ಖಂಡ_q4_1> ||
                          std::is_same_v<BLOC_TYPE, ಖಂಡ_q4_K>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i4;
                ಗಣ_kernel_impl = ನಿಜ;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (std::is_same_v<BLOC_TYPE, ಖಂಡ_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i8;
                ಗಣ_kernel_impl = ನಿಜ;
            }
        }
#ಆದರೆ_ಅಂತ್ಯ
        ಆದರೆ (!ಗಣ_kernel_impl) {
            GGML_ABORT("no kernel implementation found for the block type");
        }

        ಸ್ಥಿರಾಂಕ size_t a_k_blks = spacemit_kernels::div_ಸುತ್ತು_ಮೇಲೆ(ne10, a_blk_len);
        ಸ್ಥಿರಾಂಕ size_t b_k_blks = spacemit_kernels::div_ಸುತ್ತು_ಮೇಲೆ(ne10, b_blk_len);

        ಸ್ಥಿರಾಂಕ size_t nbw1                = a_k_blks * ಖಂಡ_stride_a;
        ಸ್ಥಿರಾಂಕ size_t nbw2                = ne11 * nbw1;
        ಸ್ಥಿರಾಂಕ size_t nbw3                = nbw2 * ne12;
        ಸ್ಥಿರಾಂಕ size_t gemm_workspace_ಗಾತ್ರ = GGML_PAD(nbw3, alignof(int64_t));

        ಸ್ಥಿರಾಂಕ uintptr_t ws_ptr         = reinterpret_cast<uintptr_t>(params->wdata);
        ಸ್ವಯಂಚಾಲಿತ *          quant_a_ಬಫರ್ = reinterpret_cast<uint8_t *>(ws_ptr);

        ಆದರೆ (ne11 == 1) {
            ಗಾಗಿ (int64_t ii = ith; ii < ne12 * a_k_blks; ii += nth) {
                int64_t i12       = ii / a_k_blks;
                int64_t ak_blk_id = ii % a_k_blks;
                quantize_a_ಅಡ್ಡಸಾಲು_i8(a_blk_len, (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) src1->ದತ್ತಾಂಶ + i12 * nb12) + ak_blk_id * a_blk_len,
                                  a_blk_len, quant_a_ಬಫರ್ + i12 * nbw2 + ak_blk_id * ಖಂಡ_stride_a);
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಗಾಗಿ (int64_t ii = ith; ii < ne12 * ne11; ii += nth) {
                int64_t i12 = ii / ne11;
                int64_t i11 = ii % ne11;
                quantize_a_ಅಡ್ಡಸಾಲು_i8(a_blk_len, (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) src1->ದತ್ತಾಂಶ + i12 * nb12 + i11 * nb11), ne10,
                                  quant_a_ಬಫರ್ + i12 * nbw2 + i11 * nbw1);
            }
        }

#ವ್ಯಾಖ್ಯಾನಿಸು MMID_ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು(ಅಡ್ಡಸಾಲು_id, i1) ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು[(ಅಡ್ಡಸಾಲು_id) *ne12 + (i1)]

        int64_t *          ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_counts       = (int64_t *) (ws_ptr + gemm_workspace_ಗಾತ್ರ);
        int32_t *          valid_ep_ಎಣಿಕೆ          = (int32_t *) (ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_counts + n_as);
        int32_t *          valid_act_ಎಣಿಕೆ         = (int32_t *) (valid_ep_ಎಣಿಕೆ + 1);
        int64_t *          valid_ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_counts = (int64_t *) (valid_act_ಎಣಿಕೆ + 1);
        mmid_ಅಡ್ಡಸಾಲು_mapping * ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು             = (mmid_ಅಡ್ಡಸಾಲು_mapping *) (valid_ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_counts + n_as);

        ಆದರೆ (ith == 0) {
            // initialize matrix_row_counts
            memset(ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_counts, 0, n_as * ಗಾತ್ರ(int64_t));

            // group rows by src0 matrix
            ಗಾಗಿ (int32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
                ಗಾಗಿ (int32_t id = 0; id < n_ids; ++id) {
                    ಸ್ಥಿರಾಂಕ int32_t i02 =
                        *(ಸ್ಥಿರಾಂಕ int32_t *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) ids->ದತ್ತಾಂಶ + iid1 * ids->nb[1] + id * ids->nb[0]);

                    GGML_ASSERT(i02 >= 0 && i02 < n_as);

                    MMID_ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು(i02, ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_counts[i02]) = { id, iid1 };
                    ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_counts[i02] += 1;
                }
            }

            int32_t valid_ep_ಎಣಿಕೆ_t  = 0;
            int32_t valid_act_ಎಣಿಕೆ_t = 0;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ cur_a = 0; cur_a < n_as; ++cur_a) {
                ಸ್ಥಿರಾಂಕ int64_t cne1 = ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_counts[cur_a];
                ಆದರೆ (cne1 == 0) {
                    ಮುಂದುವರಿ;
                }
                valid_ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_counts[valid_ep_ಎಣಿಕೆ_t] = cur_a;
                valid_act_ಎಣಿಕೆ_t += cne1;
                valid_ep_ಎಣಿಕೆ_t += 1;
            }
            valid_ep_ಎಣಿಕೆ[0]  = valid_ep_ಎಣಿಕೆ_t;
            valid_act_ಎಣಿಕೆ[0] = valid_act_ಎಣಿಕೆ_t;
        }

        ಸ್ಥಿರಾಂಕ int64_t barrier_idx = static_cast<int64_t>(ith / 2);

        GGML_ASSERT(global_spine_env_info.init_barrier != ಶೂನ್ಯಸೂಚಕ);
        GGML_ASSERT(barrier_idx < spine_init_barrier_ಎಣಿಕೆ);
        spine_barrier_t * cur_barrier = &global_spine_env_info.init_barrier[barrier_idx];

        ggml_barrier(params->threadpool);

        ಸ್ಥಿರಾಂಕ size_t ಅಡ್ಡಸಾಲು_stride_b      = b_k_blks * get_repacked_ಖಂಡ_type_ಗಾತ್ರ<BLOC_TYPE, INTER_ಗಾತ್ರ, NB_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ>();
        ಸ್ಥಿರಾಂಕ size_t expert_b_stride   = ne01 * ಅಡ್ಡಸಾಲು_stride_b;
        ಸ್ಥಿರಾಂಕ size_t ಪ್ರತಿ_nb_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_wsize = NB_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ * ಅಡ್ಡಸಾಲು_stride_b;

        std::array<ಸ್ಥಿರಾಂಕ uint8_t *, 2> src_workspaces;
        std::array<ದಶಮಾಂಶ *, 2>         dst_workspaces;

        ಸ್ವಯಂಚಾಲಿತ *     tcm_ಬಫರ್      = ggml::cpu::riscv64_spacemit::tls_context.tcm_ಬಫರ್;
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ tcm_ಬಫರ್_ಗಾತ್ರ = ggml::cpu::riscv64_spacemit::tls_context.tcm_ಬಫರ್_ಗಾತ್ರ;

        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ valid_ep_ಎಣಿಕೆ_t  = valid_ep_ಎಣಿಕೆ[0];
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ valid_act_ಎಣಿಕೆ_t = valid_act_ಎಣಿಕೆ[0];

        ಪೂರ್ಣಾಂಕ nth_es = 1;
        ಪೂರ್ಣಾಂಕ nth_n  = nth;

        ಪೂರ್ಣಾಂಕ ith_es = ith % nth_es;
        ಪೂರ್ಣಾಂಕ ith_n  = (ith / nth_es) % nth_n;

        ಆದರೆ (valid_ep_ಎಣಿಕೆ_t % nth == 0 && tcm_ಬಫರ್ != ಶೂನ್ಯಸೂಚಕ && valid_ep_ಎಣಿಕೆ_t == n_as &&
            valid_act_ಎಣಿಕೆ_t == n_as && ಪ್ರತಿ_nb_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_wsize <= tcm_ಬಫರ್_ಗಾತ್ರ) {
            ಗಾಗಿ (int64_t valid_id = ith; valid_id < valid_ep_ಎಣಿಕೆ_t; valid_id += nth) {
                ಸ್ಥಿರಾಂಕ int64_t cur_a = valid_ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_counts[valid_id];

                ಸ್ವಯಂಚಾಲಿತ * src0_cur = (uint8_t *) src0->ದತ್ತಾಂಶ + cur_a * expert_b_stride;

                mmid_ಅಡ್ಡಸಾಲು_mapping ಅಡ್ಡಸಾಲು_mapping = MMID_ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು(cur_a, 0);
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ        id          = ಅಡ್ಡಸಾಲು_mapping.i1;
                ಸ್ಥಿರಾಂಕ int64_t    i11         = id % ne11;
                ಸ್ಥಿರಾಂಕ int64_t    i12         = ಅಡ್ಡಸಾಲು_mapping.i2;
                ಸ್ಥಿರಾಂಕ int64_t    i1          = id;
                ಸ್ಥಿರಾಂಕ int64_t    i2          = i12;

                ಸ್ವಯಂಚಾಲಿತ *  src1_ಲಂಬಸಾಲು = quant_a_ಬಫರ್ + (i11 * nbw1 + i12 * nbw2);
                ದಶಮಾಂಶ * c_blk    = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + (i1 * nb1 + i2 * nb2));

                uint8_t * a_ಅಡ್ಡಸಾಲು = src1_ಲಂಬಸಾಲು;
                uint8_t * b_ಲಂಬಸಾಲು = reinterpret_cast<uint8_t *>(tcm_ಬಫರ್);
                ಆದರೆ ((nbw1 + ಪ್ರತಿ_nb_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_wsize) <= tcm_ಬಫರ್_ಗಾತ್ರ) {
                    a_ಅಡ್ಡಸಾಲು = (uint8_t *) tcm_ಬಫರ್;
                    b_ಲಂಬಸಾಲು = reinterpret_cast<uint8_t *>(tcm_ಬಫರ್) + nbw1;
                }
                uint8_t * b_ಲಂಬಸಾಲು_zp = ಖಂಡ_type_has_zp<BLOC_TYPE>() ? b_ಲಂಬಸಾಲು : ಶೂನ್ಯಸೂಚಕ;

                ಆದರೆ (ith % 2 == 0) {
                    spacemit_kernels::rvv::memcpy1d(b_ಲಂಬಸಾಲು, reinterpret_cast<uint8_t *>(src0_cur), ಪ್ರತಿ_nb_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_wsize);

                    ಆದರೆ (a_ಅಡ್ಡಸಾಲು != src1_ಲಂಬಸಾಲು) {
                        spacemit_kernels::rvv::memcpy1d(a_ಅಡ್ಡಸಾಲು, src1_ಲಂಬಸಾಲು, nbw1);
                    }
                }

                spine_barrier_wait(cur_barrier);

                ಆದರೆ (ith % 2 != 0) {
                    ಆದರೆ (a_ಅಡ್ಡಸಾಲು != src1_ಲಂಬಸಾಲು) {
                        spacemit_kernels::rvv::memcpy1d(a_ಅಡ್ಡಸಾಲು, src1_ಲಂಬಸಾಲು, nbw1);
                    }

                    spacemit_kernels::rvv::memcpy1d(b_ಲಂಬಸಾಲು, reinterpret_cast<uint8_t *>(src0_cur), ಪ್ರತಿ_nb_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_wsize);
                }

                int64_t nb_real = std::min(ne01, NB_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ);
                ಗಾಗಿ (int64_t ni = 0; ni < ne01; ni += NB_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ) {
                    ಆದರೆ (ith % 2 != 0) {
                        spine_barrier_wait(cur_barrier);
                    }

                    gemm_kernel(b_blk_len, a_ಅಡ್ಡಸಾಲು, b_ಲಂಬಸಾಲು, b_ಲಂಬಸಾಲು_zp, c_blk + ni, 1, nb_real, b_k_blks, ne01);

                    ಆದರೆ (ith % 2 == 0) {
                        spine_barrier_wait(cur_barrier);
                    }

                    ಸ್ಥಿರಾಂಕ int64_t ಮುಂದಿನ_ni = ni + NB_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;
                    ಆದರೆ (ಮುಂದಿನ_ni < ne01) {
                        nb_real = std::min(ne01 - ಮುಂದಿನ_ni, NB_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ);
                        spacemit_kernels::rvv::memcpy1d(
                            b_ಲಂಬಸಾಲು, reinterpret_cast<uint8_t *>(src0_cur) + ಮುಂದಿನ_ni * ಅಡ್ಡಸಾಲು_stride_b, ಪ್ರತಿ_nb_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_wsize);
                    }
                }
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಗಾಗಿ (int64_t valid_id = ith_es; valid_id < valid_ep_ಎಣಿಕೆ_t; valid_id += nth_es) {
                ಸ್ಥಿರಾಂಕ int64_t cur_a = valid_ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_counts[valid_id];
                ಸ್ಥಿರಾಂಕ int64_t cne1  = ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_counts[cur_a];

                int64_t src1_cur_ಆರಂಭ = 0;
                int64_t src1_cur_ಕೊನೆ   = cne1;

                int64_t src0_cur_ಆರಂಭ = (ith_n * ne01) / nth_n;
                int64_t src0_cur_ಕೊನೆ   = MIN(((ith_n + 1) * ne01) / nth_n, ne01);

                ಆದರೆ (src1_cur_ಆರಂಭ >= src1_cur_ಕೊನೆ || src0_cur_ಆರಂಭ >= src0_cur_ಕೊನೆ) {
                    ಮುಂದುವರಿ;
                }

                src0_cur_ಆರಂಭ =
                    (src0_cur_ಆರಂಭ % NB_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ) ? src0_cur_ಆರಂಭ + NB_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ - (src0_cur_ಆರಂಭ % NB_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ) : src0_cur_ಆರಂಭ;
                src0_cur_ಕೊನೆ =
                    (src0_cur_ಕೊನೆ % NB_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ) ? src0_cur_ಕೊನೆ + NB_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ - (src0_cur_ಕೊನೆ % NB_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ) : src0_cur_ಕೊನೆ;

                ಸ್ವಯಂಚಾಲಿತ *    src0_cur = (uint8_t *) src0->ದತ್ತಾಂಶ + cur_a * expert_b_stride + src0_cur_ಆರಂಭ * ಅಡ್ಡಸಾಲು_stride_b;
                uint8_t * b_ಲಂಬಸಾಲು_zp = ಖಂಡ_type_has_zp<BLOC_TYPE>() ? src0_cur : ಶೂನ್ಯಸೂಚಕ;

                size_t extra_tcm_ಬಫರ್_ಗಾತ್ರ = tcm_ಬಫರ್_ಗಾತ್ರ;
                ಶೂನ್ಯ * extra_tcm_ಬಫರ್      = tcm_ಬಫರ್;
                ಆದರೆ (tcm_ಬಫರ್ != ಶೂನ್ಯಸೂಚಕ && (src1_cur_ಕೊನೆ - src1_cur_ಆರಂಭ) >= 4 &&
                    (src0_cur_ಕೊನೆ - src0_cur_ಆರಂಭ) * ಅಡ್ಡಸಾಲು_stride_b <= tcm_ಬಫರ್_ಗಾತ್ರ) {
                    spacemit_kernels::rvv::memcpy1d(tcm_ಬಫರ್, src0_cur,
                                                    (src0_cur_ಕೊನೆ - src0_cur_ಆರಂಭ) * ಅಡ್ಡಸಾಲು_stride_b);
                    src0_cur = reinterpret_cast<uint8_t *>(tcm_ಬಫರ್);
                    b_ಲಂಬಸಾಲು_zp = ಖಂಡ_type_has_zp<BLOC_TYPE>() ? src0_cur : ಶೂನ್ಯಸೂಚಕ;
                    extra_tcm_ಬಫರ್_ಗಾತ್ರ -= (src0_cur_ಕೊನೆ - src0_cur_ಆರಂಭ) * ಅಡ್ಡಸಾಲು_stride_b;
                    extra_tcm_ಬಫರ್ = reinterpret_cast<ಶೂನ್ಯ *>(reinterpret_cast<uintptr_t>(tcm_ಬಫರ್) +
                                                                (src0_cur_ಕೊನೆ - src0_cur_ಆರಂಭ) * ಅಡ್ಡಸಾಲು_stride_b);
                }

                ಪೂರ್ಣಾಂಕ ir1 = src1_cur_ಆರಂಭ;

                ಆದರೆ (extra_tcm_ಬಫರ್_ಗಾತ್ರ >= nbw1 && extra_tcm_ಬಫರ್ != ಶೂನ್ಯಸೂಚಕ) {
                    int64_t quant_a_tile_ಗಾತ್ರ = extra_tcm_ಬಫರ್_ಗಾತ್ರ / nbw1;
                    ಮಾಡು {
                        quant_a_tile_ಗಾತ್ರ = MIN(quant_a_tile_ಗಾತ್ರ, src1_cur_ಕೊನೆ - ir1);

                        uint8_t * quant_a_tile_ಬಫರ್ = reinterpret_cast<uint8_t *>(extra_tcm_ಬಫರ್);

                        ಪೂರ್ಣಾಂಕ iir1 = ir1;
                        ಗಾಗಿ (; iir1 < (ir1 + quant_a_tile_ಗಾತ್ರ); ++iir1) {
                            mmid_ಅಡ್ಡಸಾಲು_mapping ಅಡ್ಡಸಾಲು_mapping = MMID_ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು(cur_a, iir1);

                            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ id = ಅಡ್ಡಸಾಲು_mapping.i1;  // selected expert index

                            ಸ್ಥಿರಾಂಕ int64_t i11 = id % ne11;
                            ಸ್ಥಿರಾಂಕ int64_t i12 = ಅಡ್ಡಸಾಲು_mapping.i2;  // row index in src1

                            ಸ್ವಯಂಚಾಲಿತ * src1_ಲಂಬಸಾಲು = quant_a_ಬಫರ್ + (i11 * nbw1 + i12 * nbw2);
                            spacemit_kernels::rvv::memcpy1d(quant_a_tile_ಬಫರ್, src1_ಲಂಬಸಾಲು, nbw1);
                            quant_a_tile_ಬಫರ್ = quant_a_tile_ಬಫರ್ + nbw1;
                        }

                        quant_a_tile_ಬಫರ್ = reinterpret_cast<uint8_t *>(extra_tcm_ಬಫರ್);
                        iir1                = ir1;

                        ಆದರೆ (moe_gemm_kernel_m2 != ಶೂನ್ಯಸೂಚಕ) {
                            ಗಾಗಿ (; iir1 < (ir1 + quant_a_tile_ಗಾತ್ರ - 1); iir1 += 2, quant_a_tile_ಬಫರ್ += 2 * nbw1) {
                                mmid_ಅಡ್ಡಸಾಲು_mapping ಅಡ್ಡಸಾಲು_mapping_0 = MMID_ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು(cur_a, iir1);
                                mmid_ಅಡ್ಡಸಾಲು_mapping ಅಡ್ಡಸಾಲು_mapping_1 = MMID_ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು(cur_a, iir1 + 1);

                                src_workspaces[0] = quant_a_tile_ಬಫರ್;
                                src_workspaces[1] = quant_a_tile_ಬಫರ್ + nbw1;

                                dst_workspaces[0] =
                                    (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + (ಅಡ್ಡಸಾಲು_mapping_0.i1 * nb1 + ಅಡ್ಡಸಾಲು_mapping_0.i2 * nb2)) +
                                    src0_cur_ಆರಂಭ;
                                dst_workspaces[1] = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ +
                                                               ((ಅಡ್ಡಸಾಲು_mapping_1.i1) * nb1 + (ಅಡ್ಡಸಾಲು_mapping_1.i2) * nb2)) +
                                                    src0_cur_ಆರಂಭ;
                                moe_gemm_kernel_m2(b_blk_len, src_workspaces.ದತ್ತಾಂಶ(), src0_cur, b_ಲಂಬಸಾಲು_zp,
                                                   dst_workspaces.ದತ್ತಾಂಶ(), 1, src0_cur_ಕೊನೆ - src0_cur_ಆರಂಭ, b_k_blks,
                                                   ne01);
                            }
                        }

                        ಗಾಗಿ (; iir1 < (ir1 + quant_a_tile_ಗಾತ್ರ); iir1++, quant_a_tile_ಬಫರ್ += nbw1) {
                            mmid_ಅಡ್ಡಸಾಲು_mapping ಅಡ್ಡಸಾಲು_mapping_0 = MMID_ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು(cur_a, iir1);

                            gemm_kernel(
                                b_blk_len, quant_a_tile_ಬಫರ್, src0_cur, b_ಲಂಬಸಾಲು_zp,
                                (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + (ಅಡ್ಡಸಾಲು_mapping_0.i1 * nb1 + ಅಡ್ಡಸಾಲು_mapping_0.i2 * nb2)) +
                                    src0_cur_ಆರಂಭ,
                                1, src0_cur_ಕೊನೆ - src0_cur_ಆರಂಭ, b_k_blks, ne01);
                        }

                        ir1 += quant_a_tile_ಗಾತ್ರ;
                    } ಆಗ (ir1 < src1_cur_ಕೊನೆ);
                } ಇಲ್ಲದಿದ್ದರೆ {
                    ಆದರೆ (moe_gemm_kernel_m2 != ಶೂನ್ಯಸೂಚಕ) {
                        ಗಾಗಿ (; ir1 < src1_cur_ಕೊನೆ - 1; ir1 += 2) {
                            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ iir1 = 0; iir1 < 2; ++iir1) {
                                mmid_ಅಡ್ಡಸಾಲು_mapping ಅಡ್ಡಸಾಲು_mapping = MMID_ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು(cur_a, ir1 + iir1);

                                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ id = ಅಡ್ಡಸಾಲು_mapping.i1;  // selected expert index

                                ಸ್ಥಿರಾಂಕ int64_t i11 = id % ne11;
                                ಸ್ಥಿರಾಂಕ int64_t i12 = ಅಡ್ಡಸಾಲು_mapping.i2;  // row index in src1

                                ಸ್ಥಿರಾಂಕ int64_t i1 = id;               // selected expert index
                                ಸ್ಥಿರಾಂಕ int64_t i2 = i12;              //ಅಡ್ಡಸಾಲು

                                src_workspaces[iir1] = quant_a_ಬಫರ್ + (i11 * nbw1 + i12 * nbw2);

                                dst_workspaces[iir1] =
                                    (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + (i1 * nb1 + i2 * nb2)) + src0_cur_ಆರಂಭ;
                            }

                            moe_gemm_kernel_m2(b_blk_len, src_workspaces.ದತ್ತಾಂಶ(), src0_cur, b_ಲಂಬಸಾಲು_zp,
                                               dst_workspaces.ದತ್ತಾಂಶ(), 1, src0_cur_ಕೊನೆ - src0_cur_ಆರಂಭ, b_k_blks, ne01);
                        }
                    }

                    ಗಾಗಿ (; ir1 < src1_cur_ಕೊನೆ; ir1++) {
                        mmid_ಅಡ್ಡಸಾಲು_mapping ಅಡ್ಡಸಾಲು_mapping = MMID_ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು(cur_a, ir1);

                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ id = ಅಡ್ಡಸಾಲು_mapping.i1;  // selected expert index

                        ಸ್ಥಿರಾಂಕ int64_t i11 = id % ne11;
                        ಸ್ಥಿರಾಂಕ int64_t i12 = ಅಡ್ಡಸಾಲು_mapping.i2;  // row index in src1

                        ಸ್ಥಿರಾಂಕ int64_t i1 = id;               // selected expert index
                        ಸ್ಥಿರಾಂಕ int64_t i2 = i12;              //ಅಡ್ಡಸಾಲು

                        ಸ್ವಯಂಚಾಲಿತ * src1_ಲಂಬಸಾಲು = quant_a_ಬಫರ್ + (i11 * nbw1 + i12 * nbw2);

                        gemm_kernel(b_blk_len, src1_ಲಂಬಸಾಲು, src0_cur, b_ಲಂಬಸಾಲು_zp,
                                    (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + (i1 * nb1 + i2 * nb2)) + src0_cur_ಆರಂಭ, 1,
                                    src0_cur_ಕೊನೆ - src0_cur_ಆರಂಭ, b_k_blks, ne01);
                    }
                }
            }
        }
#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು MMID_ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು
    }

    ಪೂರ್ಣಾಂಕ repack(ggml_tensor * t, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ದತ್ತಾಂಶ_ಗಾತ್ರ) override {
        GGML_LOG_DEBUG("%s: repack tensor %s with %s_%dx%d\n", __func__, t->name, ggml_type_name(t->type),
                       (ಪೂರ್ಣಾಂಕ) NB_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ, (ಪೂರ್ಣಾಂಕ) INTER_ಗಾತ್ರ);
        ಹಿಂತಿರುಗಿ ggml::cpu::riscv64_spacemit::repack<BLOC_TYPE, INTER_ಗಾತ್ರ, NB_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ>(t, ದತ್ತಾಂಶ, ದತ್ತಾಂಶ_ಗಾತ್ರ);
    }
};

ವರ್ಗ tensor_traits_common : ಸಾರ್ವಜನಿಕ tensor_traits_ತಳ {
    ತಾರ್ಕಿಕ work_ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ n_ಎಳೆಗಳು, ಸ್ಥಿರಾಂಕ ggml_tensor * op, size_t & size) override {
        ಆಯ್ಕೆ (op->op) {
            ಸ್ಥಿತಿ GGML_OP_FLASH_ATTN_EXT:
                {
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ     n_tasks = n_ಎಳೆಗಳು;
                    ಸ್ಥಿರಾಂಕ int64_t neq2    = op->src[0]->ne[2];  // number of query heads
                    ಸ್ಥಿರಾಂಕ int64_t DK      = op->src[1]->ne[0];
                    ಸ್ಥಿರಾಂಕ int64_t DV      = op->src[2]->ne[0];  // DV

                    // Tiled flash attention scratch (tile sizes defined in common.h)
                    // Per-thread: Q_q + KQ + mask + VKQ32 + V32 + K_f32 + padding
                    size_t prefill = ಗಾತ್ರ(ದಶಮಾಂಶ) *
                                     (GGML_FA_TILE_Q * DK + 2 * GGML_FA_TILE_Q * GGML_FA_TILE_KV + GGML_FA_TILE_Q * DV +
                                      GGML_FA_TILE_KV * DV + GGML_FA_TILE_KV * DK) *
                                     n_tasks;

                    // Decode path: n_kv_chunks = n_tasks (one chunk per thread)
                    // Per-thread: VKQ accmulator (DV), partial M, partial S + intra-thread scratch for V, Q and VKQ
                    size_t n_chunks = n_tasks;
                    size_t decode   = ಗಾತ್ರ(ದಶಮಾಂಶ) * (neq2 * n_chunks * (2 + DV) + n_tasks * (DK + 2 * DV));

                    size = MAX(prefill, decode);
                }
                ಹಿಂತಿರುಗಿ ನಿಜ;
            ಪೂರ್ವನಿಯೋಜಿತ:
                ನಿಲ್ಲಿಸು;
        }
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ತಾರ್ಕಿಕ compute_forward(ggml_compute_params * params, ggml_tensor * op) override {
        ಆಯ್ಕೆ (op->op) {
            ಸ್ಥಿತಿ GGML_OP_NORM:
                ಆಯ್ಕೆ (op->src[0]->type) {
                    ಸ್ಥಿತಿ GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_norm_f32(params, op);
                        ಹಿಂತಿರುಗಿ ನಿಜ;
                    ಪೂರ್ವನಿಯೋಜಿತ:
                        GGML_ABORT("fatal error");
                }
            ಸ್ಥಿತಿ GGML_OP_RMS_NORM:
                ಆಯ್ಕೆ (op->src[0]->type) {
                    ಸ್ಥಿತಿ GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_rms_norm_f32(params, op);
                        ಹಿಂತಿರುಗಿ ನಿಜ;
                    ಪೂರ್ವನಿಯೋಜಿತ:
                        GGML_ABORT("fatal error");
                }
            ಸ್ಥಿತಿ GGML_OP_ಕೂಡಿಸು:
                ಆಯ್ಕೆ (op->src[0]->type) {
                    ಸ್ಥಿತಿ GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_ದ್ವಿಮಾನ<GGML_OP_ಕೂಡಿಸು, ದಶಮಾಂಶ>(params, op);
                        ಹಿಂತಿರುಗಿ ನಿಜ;
                    ಸ್ಥಿತಿ GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_ದ್ವಿಮಾನ<GGML_OP_ಕೂಡಿಸು, _Float16>(params, op);
                        ಹಿಂತಿರುಗಿ ನಿಜ;
                    ಪೂರ್ವನಿಯೋಜಿತ:
                        ggml_compute_forward_ಕೂಡಿಸು(params, op);
                        ಹಿಂತಿರುಗಿ ನಿಜ;
                }
            ಸ್ಥಿತಿ GGML_OP_SUB:
                ಆಯ್ಕೆ (op->src[0]->type) {
                    ಸ್ಥಿತಿ GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_ದ್ವಿಮಾನ<GGML_OP_SUB, ದಶಮಾಂಶ>(params, op);
                        ಹಿಂತಿರುಗಿ ನಿಜ;
                    ಸ್ಥಿತಿ GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_ದ್ವಿಮಾನ<GGML_OP_SUB, _Float16>(params, op);
                        ಹಿಂತಿರುಗಿ ನಿಜ;
                    ಪೂರ್ವನಿಯೋಜಿತ:
                        ggml_compute_forward_sub(params, op);
                        ಹಿಂತಿರುಗಿ ನಿಜ;
                }
            ಸ್ಥಿತಿ GGML_OP_MUL:
                ಆಯ್ಕೆ (op->src[0]->type) {
                    ಸ್ಥಿತಿ GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_ದ್ವಿಮಾನ<GGML_OP_MUL, ದಶಮಾಂಶ>(params, op);
                        ಹಿಂತಿರುಗಿ ನಿಜ;
                    ಸ್ಥಿತಿ GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_ದ್ವಿಮಾನ<GGML_OP_MUL, _Float16>(params, op);
                        ಹಿಂತಿರುಗಿ ನಿಜ;
                    ಪೂರ್ವನಿಯೋಜಿತ:
                        ggml_compute_forward_mul(params, op);
                        ಹಿಂತಿರುಗಿ ನಿಜ;
                }
            ಸ್ಥಿತಿ GGML_OP_DIV:
                ಆಯ್ಕೆ (op->src[0]->type) {
                    ಸ್ಥಿತಿ GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_ದ್ವಿಮಾನ<GGML_OP_DIV, ದಶಮಾಂಶ>(params, op);
                        ಹಿಂತಿರುಗಿ ನಿಜ;
                    ಸ್ಥಿತಿ GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_ದ್ವಿಮಾನ<GGML_OP_DIV, _Float16>(params, op);
                        ಹಿಂತಿರುಗಿ ನಿಜ;
                    ಪೂರ್ವನಿಯೋಜಿತ:
                        ggml_compute_forward_div(params, op);
                        ಹಿಂತಿರುಗಿ ನಿಜ;
                }
            ಸ್ಥಿತಿ GGML_OP_FLASH_ATTN_EXT:
                forward_flash_attn_ext_f16(params, op);
                ಹಿಂತಿರುಗಿ ನಿಜ;
            ಸ್ಥಿತಿ GGML_OP_CONT:
                {
                    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = op->src[0];
                    ಆದರೆ (op->type == src0->type && op->nb[0] != src0->nb[0] && op->nb[0] == src0->nb[1] &&
                        op->ne[3] * op->ne[2] * op->nb[2] == src0->ne[3] * src0->ne[2] * src0->nb[2]) {
                        spacemit_kernels::rvv::forward_cont_with_ಕ್ರಮಬದಲಾವಣೆ(params, op);
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        ggml_compute_forward_cont(params, op);
                    }
                    ಹಿಂತಿರುಗಿ ನಿಜ;
                }
            ಸ್ಥಿತಿ GGML_OP_CPY:
                {
                    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = op->src[0];
                    ಆದರೆ (op->type == src0->type && op->nb[0] == src0->nb[1] && src0->nb[0] != src0->nb[1] &&
                        ggml_nelements(src0) == ggml_nelements(op)) {
                        spacemit_kernels::rvv::forward_cpy_with_ಕ್ರಮಬದಲಾವಣೆ(params, op);
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        ggml_compute_forward_cpy(params, op);
                    }
                    ಹಿಂತಿರುಗಿ ನಿಜ;
                }
            ಸ್ಥಿತಿ GGML_OP_REPEAT:
                {
                    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ ಅಡ್ಡಸಾಲುಗಳು_equal         = ggml_nrows(op->src[0]) == ggml_nrows(op);
                    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ ಪ್ರಸಾರ_or_equal = op->src[0]->ne[0] == 1 || op->src[0]->ne[0] == op->ne[0];

                    ಆದರೆ (ಅಡ್ಡಸಾಲುಗಳು_equal && ಪ್ರಸಾರ_or_equal) {
                        ಆಯ್ಕೆ (op->src[0]->type) {
                            ಸ್ಥಿತಿ GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_repeat_nrows<int32_t>(params, op);
                                ಹಿಂತಿರುಗಿ ನಿಜ;
                            ಸ್ಥಿತಿ GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_repeat_nrows<int16_t>(params, op);
                                ಹಿಂತಿರುಗಿ ನಿಜ;
                            ಪೂರ್ವನಿಯೋಜಿತ:
                                ನಿಲ್ಲಿಸು;
                        }
                    }

                    ಆದರೆ (op->src[0]->ne[1] == 1 && op->src[0]->ne[0] == op->ne[0]) {
                        ಆಯ್ಕೆ (op->src[0]->type) {
                            ಸ್ಥಿತಿ GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_repeat_dim1<int32_t>(params, op);
                                ಹಿಂತಿರುಗಿ ನಿಜ;
                            ಸ್ಥಿತಿ GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_repeat_dim1<int16_t>(params, op);
                                ಹಿಂತಿರುಗಿ ನಿಜ;
                            ಪೂರ್ವನಿಯೋಜಿತ:
                                ನಿಲ್ಲಿಸು;
                        }
                    }

                    ggml_compute_forward_repeat(params, op);
                }
                ಹಿಂತಿರುಗಿ ನಿಜ;
            ಸ್ಥಿತಿ GGML_OP_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು:
                {
                    ಆದರೆ (op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32) {
                        spacemit_kernels::rvv::forward_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು<ದಶಮಾಂಶ>(params, op);
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        ggml_compute_forward_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು(params, op);
                    }
                }
                ಹಿಂತಿರುಗಿ ನಿಜ;
            ಸ್ಥಿತಿ GGML_OP_GET_ಅಡ್ಡಸಾಲುಗಳು:
                {
                    ಆದರೆ (op->src[0]->type == op->type) {
                        ಆಯ್ಕೆ (op->src[0]->type) {
                            ಸ್ಥಿತಿ GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_get_ಅಡ್ಡಸಾಲುಗಳು<int32_t>(params, op);
                                ಹಿಂತಿರುಗಿ ನಿಜ;
                            ಸ್ಥಿತಿ GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_get_ಅಡ್ಡಸಾಲುಗಳು<int16_t>(params, op);
                                ಹಿಂತಿರುಗಿ ನಿಜ;
                            ಪೂರ್ವನಿಯೋಜಿತ:
                                ನಿಲ್ಲಿಸು;
                        }
                    }

                    ggml_compute_forward_get_ಅಡ್ಡಸಾಲುಗಳು(params, op);
                }
                ಹಿಂತಿರುಗಿ ನಿಜ;
            ಸ್ಥಿತಿ GGML_OP_CONCAT:
                {
                    ಸ್ಥಿರಾಂಕ int32_t dim = ggml_get_op_params_i32(op, 0);
                    ಆದರೆ (dim == 0 && op->type == op->src[0]->type) {
                        ಆಯ್ಕೆ (op->src[0]->type) {
                            ಸ್ಥಿತಿ GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_concat<int32_t>(params, op);
                                ಹಿಂತಿರುಗಿ ನಿಜ;
                            ಸ್ಥಿತಿ GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_concat<int16_t>(params, op);
                                ಹಿಂತಿರುಗಿ ನಿಜ;
                            ಪೂರ್ವನಿಯೋಜಿತ:
                                ನಿಲ್ಲಿಸು;
                        }
                    }

                    ggml_compute_forward_concat(params, op);
                }
                ಹಿಂತಿರುಗಿ ನಿಜ;
            // TODO For GGML_OP_GATED_DELTA_NET
            // case GGML_OP_GATED_DELTA_NET:
            //     return true;
            ಪೂರ್ವನಿಯೋಜಿತ:
                ನಿಲ್ಲಿಸು;
        }
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಶೂನ್ಯ forward_flash_attn_ext_f16(ಸ್ಥಿರಾಂಕ ggml_compute_params * params, ggml_tensor * dst) {
        ಸ್ಥಿರಾಂಕ ggml_tensor * q = dst->src[0];
        ಸ್ಥಿರಾಂಕ ggml_tensor * k = dst->src[1];
        ಸ್ಥಿರಾಂಕ ggml_tensor * v = dst->src[2];

        GGML_TENSOR_LOCALS(int64_t, neq, q, ne)
        GGML_TENSOR_LOCALS(size_t, nbq, q, nb)
        GGML_TENSOR_LOCALS(int64_t, nek, k, ne)
        GGML_TENSOR_LOCALS(size_t, nbk, k, nb)
        GGML_TENSOR_LOCALS(int64_t, nev, v, ne)
        GGML_TENSOR_LOCALS(size_t, nbv, v, nb)
        GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
        GGML_TENSOR_LOCALS(size_t, nb, dst, nb)

        ಸ್ಥಿರಾಂಕ int64_t DK = nek0;
        ಸ್ಥಿರಾಂಕ int64_t DV = nev0;

        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ supported_prec  = (dst->op_params[3] == GGML_PREC_F32 || dst->op_params[3] == GGML_PREC_DEFAULT);
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ supported_types = (q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16);
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ supported_shape = (DK > 0 && DK <= 128 && DV > 0 && DV <= 128);
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ supported_vlen  = (__riscv_vlenb() == 128);

        ಆದರೆ (!(supported_prec && supported_types && supported_shape && supported_vlen)) {
            ggml_compute_forward_flash_attn_ext(params, dst);
            ಹಿಂತಿರುಗಿ;
        }

        // total rows in q
        ಸ್ಥಿರಾಂಕ int64_t nr = neq1 * neq2 * neq3;

        // rows per thread
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

        ಸ್ಥಿರ constexpr int64_t Q_TILE_SZ = ggml_fa_tile_config::Q;
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ               use_tiled = !params->use_ref && (neq1 >= Q_TILE_SZ);

        // 4x chunks per thread
        // int     nth_scaled = nth * 4;
        // int64_t chunk_size = (nr + nth_scaled - 1) / nth_scaled;
        // int64_t nchunk     = (nr + chunk_size - 1) / chunk_size;

        // if (nth == 1 || nchunk < nth) {
        //     nchunk = nth;
        // }

        int64_t nchunk = nth;

        ಆದರೆ (ith == 0) {
            // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
            ggml_threadpool_chunk_ಗಣ(params->threadpool, nth);
        }

        ggml_barrier(params->threadpool);

        // The number of elements in each chunk
        ಸ್ಥಿರಾಂಕ int64_t dr = (nr + nchunk - 1) / nchunk;

        // The first chunk comes from our thread_id, the rest will get auto-assigned.
        ಪೂರ್ಣಾಂಕ current_chunk = ith;

        ಆಗ (current_chunk < nchunk) {
            ಸ್ಥಿರಾಂಕ int64_t ir0 = dr * current_chunk;
            ಸ್ಥಿರಾಂಕ int64_t ir1 = MIN(ir0 + dr, nr);

            ಆದರೆ (use_tiled) {
                spacemit_kernels::rvv::forward_flash_attn_ext_f16_tiled_vlen1024_vf16(
                    params, dst, ir0, ir1, ggml::cpu::riscv64_spacemit::tls_context.tcm_ಬಫರ್,
                    ggml::cpu::riscv64_spacemit::tls_context.tcm_ಬಫರ್_ಗಾತ್ರ);
            } ಇಲ್ಲದಿದ್ದರೆ {
                spacemit_kernels::rvv::forward_flash_attn_ext_f16_one_chunk_vlen1024_vf16(
                    params, dst, ir0, ir1, ggml::cpu::riscv64_spacemit::tls_context.tcm_ಬಫರ್,
                    ggml::cpu::riscv64_spacemit::tls_context.tcm_ಬಫರ್_ಗಾತ್ರ);
            }

            current_chunk = ggml_threadpool_chunk_ಕೂಡಿಸು(params->threadpool, 1);
        }
    }

    ಪೂರ್ಣಾಂಕ repack(ggml_tensor * t, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ದತ್ತಾಂಶ_ಗಾತ್ರ) override {
        ಸ್ಮರಣೆ_ನಕಲು(t->ದತ್ತಾಂಶ, ದತ್ತಾಂಶ, ದತ್ತಾಂಶ_ಗಾತ್ರ);
        ಹಿಂತಿರುಗಿ 0;
    }
};

// Impl By IME1
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ tensor_traits<ಖಂಡ_q4_0, 32, 16>  q4_0_16x32_q8_0;
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ tensor_traits<ಖಂಡ_q4_1, 32, 16>  q4_1_16x32_q8_0;
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ tensor_traits<ಖಂಡ_q4_K, 32, 16>  q4_k_16x32_q8_0;
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ tensor_traits<ಖಂಡ_q8_0, 32, 16>  q8_0_16x32_q8_0;
// Impl By IME2
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ tensor_traits<ಖಂಡ_q2_K, 256, 32> q2_k_32x256_q8_0;
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ tensor_traits<ಖಂಡ_q3_K, 256, 32> q3_k_32x256_q8_0;
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ tensor_traits<ಖಂಡ_q4_0, 32, 32>  q4_0_32x32_q8_0;
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ tensor_traits<ಖಂಡ_q4_1, 32, 32>  q4_1_32x32_q8_0;
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ tensor_traits<ಖಂಡ_q4_0, 256, 32> q4_0_32x256_q8_0;
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ tensor_traits<ಖಂಡ_q4_1, 256, 32> q4_1_32x256_q8_0;
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ tensor_traits<ಖಂಡ_q4_K, 32, 32>  q4_k_32x32_q8_0;
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ tensor_traits<ಖಂಡ_q6_K, 32, 32>  q6_k_32x32_q8_0;
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ tensor_traits<ಖಂಡ_q8_0, 32, 32>  q8_0_32x32_q8_0;
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ tensor_traits<ಖಂಡ_mxfp4, 32, 32> mxfp4_32x32_q8_0;
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ tensor_traits<ಖಂಡ_q5_K, 32, 32>  q5_k_32x32_q8_0;
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ tensor_traits<ಖಂಡ_q5_1, 32, 32>  q5_1_32x32_q8_0;
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ tensor_traits<ಖಂಡ_q5_0, 32, 32>  q5_0_32x32_q8_0;
// Impl By RVV
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ tensor_traits_common               rvv_impl;

}  // namespace ggml::cpu::riscv64_spacemit

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ggml::cpu::tensor_traits * ggml_riscv64_spacemit_get_optimal_repack_type(ಸ್ಥಿರಾಂಕ ggml_tensor * cur) {
    ಆಯ್ಕೆ (cur->type) {
        ಸ್ಥಿತಿ GGML_TYPE_Q2_K:
            {
#ಆದರೆ defined(RISCV64_SPACEMIT_IME2)
                ಆದರೆ (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    ಹಿಂತಿರುಗಿ &ggml::cpu::riscv64_spacemit::q2_k_32x256_q8_0;
                }
#ಆದರೆ_ಅಂತ್ಯ
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q3_K:
            {
#ಆದರೆ defined(RISCV64_SPACEMIT_IME2)
                ಆದರೆ (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    ಹಿಂತಿರುಗಿ &ggml::cpu::riscv64_spacemit::q3_k_32x256_q8_0;
                }
#ಆದರೆ_ಅಂತ್ಯ
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
            {
#ಆದರೆ defined(RISCV64_SPACEMIT_IME2)
                ಆದರೆ (cur->ne[1] % 32 == 0 && cur->ne[0] % 256 == 0 &&
                    (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    ಹಿಂತಿರುಗಿ &ggml::cpu::riscv64_spacemit::q4_0_32x256_q8_0;
                }

                ಆದರೆ (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    ಹಿಂತಿರುಗಿ &ggml::cpu::riscv64_spacemit::q4_0_32x32_q8_0;
                }
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(RISCV64_SPACEMIT_IME1)
                ಆದರೆ (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    ಹಿಂತಿರುಗಿ &ggml::cpu::riscv64_spacemit::q4_0_16x32_q8_0;
                }
#ಆದರೆ_ಅಂತ್ಯ
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q4_1:
            {
#ಆದರೆ defined(RISCV64_SPACEMIT_IME2)
                // TODO
                // if (cur->ne[1] % 32 == 0 && cur->ne[0] % 256 == 0 &&
                //     (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                //     return &ggml::cpu::riscv64_spacemit::q4_1_32x256_q8_0;
                // }

                ಆದರೆ (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    ಹಿಂತಿರುಗಿ &ggml::cpu::riscv64_spacemit::q4_1_32x32_q8_0;
                }
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(RISCV64_SPACEMIT_IME1)
                ಆದರೆ (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    ಹಿಂತಿರುಗಿ &ggml::cpu::riscv64_spacemit::q4_1_16x32_q8_0;
                }
#ಆದರೆ_ಅಂತ್ಯ
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q4_K:
            {
#ಆದರೆ defined(RISCV64_SPACEMIT_IME2)
                ಆದರೆ (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    ಹಿಂತಿರುಗಿ &ggml::cpu::riscv64_spacemit::q4_k_32x32_q8_0;
                }
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(RISCV64_SPACEMIT_IME1)
                ಆದರೆ (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    ಹಿಂತಿರುಗಿ &ggml::cpu::riscv64_spacemit::q4_k_16x32_q8_0;
                }
#ಆದರೆ_ಅಂತ್ಯ
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q6_K:
            {
#ಆದರೆ defined(RISCV64_SPACEMIT_IME2)
                ಆದರೆ ((ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    ಹಿಂತಿರುಗಿ &ggml::cpu::riscv64_spacemit::q6_k_32x32_q8_0;
                }
#ಆದರೆ_ಅಂತ್ಯ
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
            {
#ಆದರೆ defined(RISCV64_SPACEMIT_IME2)
                ಆದರೆ ((ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    ಹಿಂತಿರುಗಿ &ggml::cpu::riscv64_spacemit::q8_0_32x32_q8_0;
                }
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(RISCV64_SPACEMIT_IME1)
                ಆದರೆ (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    ಹಿಂತಿರುಗಿ &ggml::cpu::riscv64_spacemit::q8_0_16x32_q8_0;
                }
#ಆದರೆ_ಅಂತ್ಯ
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_MXFP4:
            {
#ಆದರೆ defined(RISCV64_SPACEMIT_IME2)
                // TODO
                // if (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                //     return &ggml::cpu::riscv64_spacemit::mxfp4_32x32_q8_0;
                // }
#ಆದರೆ_ಅಂತ್ಯ
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q5_K:
            {
#ಆದರೆ defined(RISCV64_SPACEMIT_IME2)
                ಆದರೆ (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    ಹಿಂತಿರುಗಿ &ggml::cpu::riscv64_spacemit::q5_k_32x32_q8_0;
                }
#ಆದರೆ_ಅಂತ್ಯ
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q5_1:
            {
#ಆದರೆ defined(RISCV64_SPACEMIT_IME2)
                ಆದರೆ (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    ಹಿಂತಿರುಗಿ &ggml::cpu::riscv64_spacemit::q5_1_32x32_q8_0;
                }
#ಆದರೆ_ಅಂತ್ಯ
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q5_0:
            {
#ಆದರೆ defined(RISCV64_SPACEMIT_IME2)
                ಆದರೆ (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    ಹಿಂತಿರುಗಿ &ggml::cpu::riscv64_spacemit::q5_0_32x32_q8_0;
                }
#ಆದರೆ_ಅಂತ್ಯ
            }
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ನಿಲ್ಲಿಸು;
    }

    ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
}

ಸ್ಥಿರ ಎಣಿಕೆ ggml_ಸ್ಥಿತಿ ggml_backend_riscv64_spacemit_ಬಫರ್_init_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್,
                                                                         ggml_tensor *         tensor) {
    tensor->extra =
        (ಶೂನ್ಯ *) const_cast<ggml::cpu::tensor_traits *>(ggml_riscv64_spacemit_get_optimal_repack_type(tensor));

    GGML_ಬಳಸದ(ಬಫರ್);

    ಹಿಂತಿರುಗಿ GGML_ಸ್ಥಿತಿ_SUCCESS;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_riscv64_spacemit_ಬಫರ್_free_ಬಫರ್(ggml_backend_ಬಫರ್_t ಬಫರ್) {
    GGML_ASSERT(ಬಫರ್);

    ಶೂನ್ಯ * ತಳ = ಬಫರ್->context;
    ಆದರೆ (ತಳ == ಶೂನ್ಯಸೂಚಕ) {
        ಹಿಂತಿರುಗಿ;
    }

    ggml::cpu::riscv64_spacemit::spine_mem_pool_free(ತಳ);
}

ಸ್ಥಿರ ಶೂನ್ಯ * ggml_backend_riscv64_spacemit_ಬಫರ್_get_ತಳ(ggml_backend_ಬಫರ್_t ಬಫರ್) {
    GGML_ASSERT(ಬಫರ್);

    ಶೂನ್ಯ * ತಳ = ಬಫರ್->context;
    GGML_ASSERT(ತಳ != ಶೂನ್ಯಸೂಚಕ);
    ಹಿಂತಿರುಗಿ ತಳ;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_riscv64_spacemit_ಬಫರ್_memset_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್,
                                                               ggml_tensor *         tensor,
                                                               uint8_t               value,
                                                               size_t                ಸ್ಥಾನಾಂತರ,
                                                               size_t                size) {
    GGML_ASSERT(tensor);
    memset((ಅಕ್ಷರ *) tensor->ದತ್ತಾಂಶ + ಸ್ಥಾನಾಂತರ, value, size);

    GGML_ಬಳಸದ(ಬಫರ್);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_riscv64_spacemit_ಬಫರ್_clear(ggml_backend_ಬಫರ್_t ಬಫರ್, uint8_t value) {
    GGML_ASSERT(ಬಫರ್);

    ಶೂನ್ಯ * ತಳ = ಬಫರ್->context;
    GGML_ASSERT(ತಳ != ಶೂನ್ಯಸೂಚಕ);
    memset(ತಳ, value, ಬಫರ್->size);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_riscv64_spacemit_ಬಫರ್_ಗಣ_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್,
                                                            ggml_tensor *         tensor,
                                                            ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *          ದತ್ತಾಂಶ,
                                                            size_t                ಸ್ಥಾನಾಂತರ,
                                                            size_t                size) {
    GGML_ASSERT(ಸ್ಥಾನಾಂತರ == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    ಸ್ವಯಂಚಾಲಿತ tensor_traits = (ggml::cpu::riscv64_spacemit::tensor_traits_ತಳ *) tensor->extra;
    ಆದರೆ (tensor_traits) {
        ಸ್ವಯಂಚಾಲಿತ OK = tensor_traits->repack(tensor, ದತ್ತಾಂಶ, size);
        GGML_ASSERT(OK == 0);
    }

    GGML_ಬಳಸದ(ಬಫರ್);
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ggml_backend_ಬಫರ್_i ggml_backend_riscv64_spacemit_ಬಫರ್_i = {
    /* .free_buffer     = */ ggml_backend_riscv64_spacemit_ಬಫರ್_free_ಬಫರ್,
    /* .get_base        = */ ggml_backend_riscv64_spacemit_ಬಫರ್_get_ತಳ,
    /* .init_tensor     = */ ggml_backend_riscv64_spacemit_ಬಫರ್_init_tensor,
    /* .memset_tensor   = */ ggml_backend_riscv64_spacemit_ಬಫರ್_memset_tensor,
    /* .set_tensor      = */ ggml_backend_riscv64_spacemit_ಬಫರ್_ಗಣ_tensor,
    /* .get_tensor      = */ ಶೂನ್ಯಸೂಚಕ,
    /* .set_tensor_2d   = */ ಶೂನ್ಯಸೂಚಕ,
    /* .get_tensor_2d   = */ ಶೂನ್ಯಸೂಚಕ,
    /* .cpy_tensor      = */ ಶೂನ್ಯಸೂಚಕ,
    /* .clear           = */ ggml_backend_riscv64_spacemit_ಬಫರ್_clear,
    /* .reset           = */ ಶೂನ್ಯಸೂಚಕ,
};

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_cpu_riscv64_spacemit_ಬಫರ್_type_get_name(ggml_backend_ಬಫರ್_type_t buft) {
    ಹಿಂತಿರುಗಿ "CPU_RISCV64_SPACEMIT";

    GGML_ಬಳಸದ(buft);
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_t ggml_backend_cpu_riscv64_spacemit_ಬಫರ್_type_alloc_ಬಫರ್(ggml_backend_ಬಫರ್_type_t buft,
                                                                                        size_t size) {
    ಶೂನ್ಯ * ತಳ = ggml::cpu::riscv64_spacemit::spine_mem_pool_alloc(size, 64);
    ಆದರೆ (ತಳ == ಶೂನ್ಯಸೂಚಕ) {
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }

    ಹಿಂತಿರುಗಿ ggml_backend_ಬಫರ್_init(buft, ggml_backend_riscv64_spacemit_ಬಫರ್_i, ತಳ, size);
}

ಸ್ಥಿರ size_t ggml_backend_cpu_riscv64_spacemit_ಬಫರ್_type_get_alignment(ggml_backend_ಬಫರ್_type_t buft) {
    ಹಿಂತಿರುಗಿ 64;

    GGML_ಬಳಸದ(buft);
}

ಸ್ಥಿರ size_t ggml_backend_cpu_riscv64_spacemit_nbytes(ggml_backend_ಬಫರ್_type_t buft, ಸ್ಥಿರಾಂಕ ggml_tensor * tensor) {
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_DIMS; ++i) {
        ಆದರೆ (tensor->ne[i] <= 0) {
            ಹಿಂತಿರುಗಿ 0;
        }
    }

    GGML_ಬಳಸದ(buft);

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ plain_nbytes = [&]() {
        size_t ಒಟ್ಟು = ggml_type_ಗಾತ್ರ(tensor->type);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_DIMS; ++i) {
            ಒಟ್ಟು += (tensor->ne[i] - 1) * tensor->nb[i];
        }
        ಹಿಂತಿರುಗಿ ಒಟ್ಟು;
    };

    ಸ್ಥಿರಾಂಕ size_t blck_ಗಾತ್ರ = ggml_blck_ಗಾತ್ರ(tensor->type);
    ಆದರೆ (blck_ಗಾತ್ರ == 1) {
        ಹಿಂತಿರುಗಿ plain_nbytes();
    }

    ಸ್ಥಿರಾಂಕ size_t ಅಡ್ಡಸಾಲು_nbytes = tensor->ne[0] * tensor->nb[0] / blck_ಗಾತ್ರ;

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ ಕೂಡಿಸು_strided_nbytes = [&](size_t ಒಟ್ಟು, size_t src_ಖಂಡ_ಗಾತ್ರ, size_t dst_ಖಂಡ_ಗಾತ್ರ) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i < GGML_MAX_DIMS; ++i) {
            ಒಟ್ಟು += (tensor->ne[i] - 1) * (tensor->nb[i] / src_ಖಂಡ_ಗಾತ್ರ) * dst_ಖಂಡ_ಗಾತ್ರ;
        }
        ಹಿಂತಿರುಗಿ ಒಟ್ಟು;
    };

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ remap_ಖಂಡ_nbytes = [&](size_t src_ಖಂಡ_ಗಾತ್ರ, size_t dst_ಖಂಡ_ಗಾತ್ರ, int64_t padded_ಅಡ್ಡಸಾಲುಗಳು = 0) {
        GGML_ASSERT(ಅಡ್ಡಸಾಲು_nbytes % src_ಖಂಡ_ಗಾತ್ರ == 0);

        size_t ಒಟ್ಟು =
            ಕೂಡಿಸು_strided_nbytes((ಅಡ್ಡಸಾಲು_nbytes / src_ಖಂಡ_ಗಾತ್ರ) * dst_ಖಂಡ_ಗಾತ್ರ, src_ಖಂಡ_ಗಾತ್ರ, dst_ಖಂಡ_ಗಾತ್ರ);

        ಆದರೆ (padded_ಅಡ್ಡಸಾಲುಗಳು > 0 && tensor->ne[1] % padded_ಅಡ್ಡಸಾಲುಗಳು != 0) {
            ಒಟ್ಟು += (padded_ಅಡ್ಡಸಾಲುಗಳು - tensor->ne[1] % padded_ಅಡ್ಡಸಾಲುಗಳು) * (tensor->nb[1] / src_ಖಂಡ_ಗಾತ್ರ) * dst_ಖಂಡ_ಗಾತ್ರ;
        }

        ಹಿಂತಿರುಗಿ ಒಟ್ಟು;
    };

    size_t nbytes = ಅಡ್ಡಸಾಲು_nbytes;
    ಆಯ್ಕೆ (tensor->type) {
        ಸ್ಥಿತಿ GGML_TYPE_Q4_K:
            nbytes = remap_ಖಂಡ_nbytes(ಗಾತ್ರ(ಖಂಡ_q4_K), ಗಾತ್ರ(ಖಂಡ_q4_1) * 8);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q6_K:
            nbytes = remap_ಖಂಡ_nbytes(ಗಾತ್ರ(ಖಂಡ_q6_K), ಗಾತ್ರ(ಖಂಡ_q8_0) * 8, 32);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
            nbytes = remap_ಖಂಡ_nbytes(ಗಾತ್ರ(ಖಂಡ_q8_0), ಗಾತ್ರ(ಖಂಡ_q8_0), 32);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q2_K:
            nbytes = remap_ಖಂಡ_nbytes(ಗಾತ್ರ(ಖಂಡ_q2_K), ಗಾತ್ರ(spacemit_kernels::nrow_ಖಂಡ_q2_k<1>));
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q3_K:
            nbytes = remap_ಖಂಡ_nbytes(ಗಾತ್ರ(ಖಂಡ_q3_K), ಗಾತ್ರ(spacemit_kernels::nrow_ಖಂಡ_q3_k<1>));
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_MXFP4:
            nbytes = remap_ಖಂಡ_nbytes(ಗಾತ್ರ(ಖಂಡ_mxfp4), ಗಾತ್ರ(spacemit_kernels::nrow_ಖಂಡ_mxfp4<1>));
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q5_K:
            nbytes = remap_ಖಂಡ_nbytes(ಗಾತ್ರ(ಖಂಡ_q5_K), ಗಾತ್ರ(spacemit_kernels::nrow_ಖಂಡ_q5_1<1>) * 8);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q5_1:
            nbytes = remap_ಖಂಡ_nbytes(ಗಾತ್ರ(ಖಂಡ_q5_1), ಗಾತ್ರ(spacemit_kernels::nrow_ಖಂಡ_q5_1<1>));
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q5_0:
            nbytes = remap_ಖಂಡ_nbytes(ಗಾತ್ರ(ಖಂಡ_q5_0), ಗಾತ್ರ(spacemit_kernels::nrow_ಖಂಡ_q5_0<1>));
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            nbytes = ಕೂಡಿಸು_strided_nbytes(ಅಡ್ಡಸಾಲು_nbytes, 1, 1);
            ನಿಲ್ಲಿಸು;
    }

    ಹಿಂತಿರುಗಿ nbytes;
}

ನಾಮಾಕಾಶ ggml::cpu::riscv64_spacemit {

ವರ್ಗ extra_ಬಫರ್_type : ggml::cpu::extra_ಬಫರ್_type {
    ತಾರ್ಕಿಕ supports_op(ggml_backend_dev_t, ಸ್ಥಿರಾಂಕ ggml_tensor * op) override {
        ಆಯ್ಕೆ (op->op) {
            ಸ್ಥಿತಿ GGML_OP_MUL_MAT:
                ಆದರೆ (op->src[0]->ಬಫರ್ && (ggml_n_dims(op->src[0]) == 2) &&
                    op->src[0]->ಬಫರ್->buft == ggml_backend_cpu_riscv64_spacemit_ಬಫರ್_type() &&
                    ggml_riscv64_spacemit_get_optimal_repack_type(op->src[0])) {
                    ಆದರೆ (op->src[1]->ಬಫರ್ && !ggml_backend_buft_is_host(op->src[1]->ಬಫರ್->buft)) {
                        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                    }
                    ಆದರೆ (op->src[1]->type == GGML_TYPE_F32) {
                        ಹಿಂತಿರುಗಿ ನಿಜ;
                    }
                }
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_OP_MUL_MAT_ID:
                ಆದರೆ (op->src[0]->ಬಫರ್ && (ggml_n_dims(op->src[0]) == 3) &&
                    op->src[0]->ಬಫರ್->buft == ggml_backend_cpu_riscv64_spacemit_ಬಫರ್_type() &&
                    ggml_riscv64_spacemit_get_optimal_repack_type(op->src[0])) {
                    ಆದರೆ (op->src[1]->ಬಫರ್ && !ggml_backend_buft_is_host(op->src[1]->ಬಫರ್->buft)) {
                        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                    }
                    ಆದರೆ (op->src[1]->type == GGML_TYPE_F32) {
                        ಹಿಂತಿರುಗಿ ನಿಜ;
                    }
                }
                ನಿಲ್ಲಿಸು;
            ಪೂರ್ವನಿಯೋಜಿತ:
                // GGML_ABORT("fatal error");
                ನಿಲ್ಲಿಸು;
        }
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(ಸ್ಥಿರಾಂಕ ggml_tensor * op) override {
        ಆಯ್ಕೆ (op->op) {
            ಸ್ಥಿತಿ GGML_OP_MUL_MAT:
            ಸ್ಥಿತಿ GGML_OP_MUL_MAT_ID:
                ಆದರೆ (op->src[0]->ಬಫರ್ && op->src[0]->ಬಫರ್->buft == ggml_backend_cpu_riscv64_spacemit_ಬಫರ್_type()) {
                    ಹಿಂತಿರುಗಿ (ggml::cpu::tensor_traits *) op->src[0]->extra;
                }
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_OP_NORM:
            ಸ್ಥಿತಿ GGML_OP_RMS_NORM:
            ಸ್ಥಿತಿ GGML_OP_ಕೂಡಿಸು:
            ಸ್ಥಿತಿ GGML_OP_SUB:
            ಸ್ಥಿತಿ GGML_OP_MUL:
            ಸ್ಥಿತಿ GGML_OP_DIV:
            ಸ್ಥಿತಿ GGML_OP_FLASH_ATTN_EXT:
            ಸ್ಥಿತಿ GGML_OP_CONT:
            ಸ್ಥಿತಿ GGML_OP_CPY:
            ಸ್ಥಿತಿ GGML_OP_REPEAT:
            ಸ್ಥಿತಿ GGML_OP_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು:
            ಸ್ಥಿತಿ GGML_OP_GET_ಅಡ್ಡಸಾಲುಗಳು:
            ಸ್ಥಿತಿ GGML_OP_CONCAT:
                // case GGML_OP_GATED_DELTA_NET:
                ಹಿಂತಿರುಗಿ (ggml::cpu::tensor_traits *) (&ggml::cpu::riscv64_spacemit::rvv_impl);
            ಪೂರ್ವನಿಯೋಜಿತ:
                // GGML_ABORT("fatal error");
                ನಿಲ್ಲಿಸು;
        }

        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }
};

}  // namespace ggml::cpu::riscv64_spacemit

ggml_backend_ಬಫರ್_type_t ggml_backend_cpu_riscv64_spacemit_ಬಫರ್_type(ಶೂನ್ಯ) {
    ಸ್ಥಿರ ggml_backend_ಬಫರ್_type ggml_backend_cpu_ಬಫರ್_type_riscv64_spacemit = {
  /* .iface    = */
        {
         /* .get_name           = */ ggml_backend_cpu_riscv64_spacemit_ಬಫರ್_type_get_name,
         /* .alloc_buffer       = */ ggml_backend_cpu_riscv64_spacemit_ಬಫರ್_type_alloc_ಬಫರ್,
         /* .alloc_buffer_n     = */ ಖಾಲಿ,
         /* .get_alignment      = */ ggml_backend_cpu_riscv64_spacemit_ಬಫರ್_type_get_alignment,
         /* .get_max_size       = */ ಶೂನ್ಯಸೂಚಕ,
         /* .get_alloc_size     = */ ggml_backend_cpu_riscv64_spacemit_nbytes,
         /* .get_alloc_size_n   = */ ಖಾಲಿ,
         /* .is_host            = */ ಶೂನ್ಯಸೂಚಕ,
         },
 /* .device  = */
        ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
 /* .context = */
        ಹೊಸ ggml::cpu::riscv64_spacemit::extra_ಬಫರ್_type(),
    };

    ಹಿಂತಿರುಗಿ &ggml_backend_cpu_ಬಫರ್_type_riscv64_spacemit;
}

ಬಾಹ್ಯ "C" {
ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ bind_ai_thread() {
    ಪೂರ್ಣಾಂಕ  fd, ಬೈಟುಗಳು;
    ಅಕ್ಷರ str[32];

    fd = open("/proc/set_ai_thread", O_WRONLY);
    ಆದರೆ (fd < 0) {
        GGML_LOG_ERROR("try open /proc/set_ai_thread failed\n");
        ಹಿಂತಿರುಗಿ -1;
    }

    snprintf(str, 16, "%d", 0);
    ಬೈಟುಗಳು = ಲಿಖ(fd, str, ಅಕ್ಷರಸರಣಿ_ಉದ್ದ(str));
    ಆದರೆ (ಬೈಟುಗಳು < 0) {
        GGML_LOG_ERROR("try write /proc/set_ai_thread failed\n");
        close(fd);
        ಹಿಂತಿರುಗಿ -1;
    }

    close(fd);
    ಹಿಂತಿರುಗಿ 0;
}

ಶೂನ್ಯ ggml_backend_cpu_riscv64_spacemit_ಗಣ_numa_thread_affinity(ಪೂರ್ಣಾಂಕ thread_n) {
    ಪೂರ್ಣಾಂಕ cpu_id = sched_getcpu();
    ಆದರೆ (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2 &&
        !((1 << cpu_id) & ggml::cpu::riscv64_spacemit::global_spine_env_info.cpu_ಮುಖವಾಡ)) {
        GGML_PRINT_DEBUG("bind_ai_thread for thread %d, pid %d\n", thread_n, getpid());
        bind_ai_thread();
    }

    ಆದರೆ (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_tcm &&
        ggml::cpu::riscv64_spacemit::tls_context.cpu_id == -1) {
        CPU_ZERO(&(ggml::cpu::riscv64_spacemit::tls_context.cpuset));
        pthread_t    main_thread     = pthread_self();
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & perfer_core_ids = ggml::cpu::riscv64_spacemit::global_spine_env_info.perfer_core_ids;
        ಆದರೆ (thread_n < 0 || static_cast<size_t>(thread_n) >= perfer_core_ids.size()) {
            GGML_ABORT("thread_n %d exceeds perfer_core_ids size %zu\n", thread_n, perfer_core_ids.size());
        }
        ಸ್ವಯಂಚಾಲಿತ perfer_cpu_id = perfer_core_ids[static_cast<size_t>(thread_n)];
        CPU_SET(perfer_cpu_id, &(ggml::cpu::riscv64_spacemit::tls_context.cpuset));
        ಪೂರ್ಣಾಂಕ s =
            pthread_setaffinity_np(main_thread, ಗಾತ್ರ(cpu_set_t), &(ggml::cpu::riscv64_spacemit::tls_context.cpuset));
        ಆದರೆ (s != 0) {
            GGML_ABORT("set thread affinity error for thread_n %d, cpu_id %d\n", thread_n, perfer_cpu_id);
        }

        ಪೂರ್ಣಾಂಕ ai_cpu_id = perfer_cpu_id - ggml::cpu::riscv64_spacemit::global_spine_env_info.aicpu_id_ಸ್ಥಾನಾಂತರ;
        ggml::cpu::riscv64_spacemit::tls_context.cpu_id = ai_cpu_id;
        ggml::cpu::riscv64_spacemit::tls_context.tcm_ಬಫರ್ =
            ggml::cpu::riscv64_spacemit::spine_mem_pool_tcm_mem_get(ai_cpu_id);
        ggml::cpu::riscv64_spacemit::tls_context.tcm_ಬಫರ್_ಗಾತ್ರ =
            ggml::cpu::riscv64_spacemit::global_spine_env_info.tcm_blk_ಗಾತ್ರ;
    }

    ಆದರೆ (ggml::cpu::riscv64_spacemit::tls_context.tcm_ಬಫರ್ != ಶೂನ್ಯಸೂಚಕ) {
        ಶೂನ್ಯ * rt =
            ggml::cpu::riscv64_spacemit::spine_mem_pool_tcm_mem_wait(ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        ಆದರೆ (rt == ಶೂನ್ಯಸೂಚಕ) {
            GGML_ABORT("wait tcm buffer failed for cpu_id: %d", ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        }
    }
}

ಶೂನ್ಯ ggml_backend_cpu_riscv64_spacemit_clear_numa_thread_affinity_threaded(ಪೂರ್ಣಾಂಕ thread_n) {
    ಆದರೆ (ggml::cpu::riscv64_spacemit::tls_context.tcm_ಬಫರ್ != ಶೂನ್ಯಸೂಚಕ) {
        ಸ್ವಯಂಚಾಲಿತ rt = ggml::cpu::riscv64_spacemit::spine_mem_pool_tcm_mem_release(
            ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        ಆದರೆ (rt != 0) {
            GGML_ABORT("release tcm buffer failed for cpu_id: %d", ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        }
    }
}
}
