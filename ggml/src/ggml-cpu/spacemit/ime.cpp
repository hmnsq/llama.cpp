#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_COMMON_IMPL_CPP
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_COMMON_DECL_CPP

#ਅਨ੍ਤਰ੍ਭਾਵ "ime.h"

#ਅਨ੍ਤਰ੍ਭਾਵ "binary-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-cpu.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ime_env.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ime_kernels.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "repack.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "rvv_kernels.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "spine_mem_pool.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "traits.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "vec.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <fcntl.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <sys/mman.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <unistd.h>

#ਅਨ੍ਤਰ੍ਭਾਵ <algorithm>
#ਅਨ੍ਤਰ੍ਭਾਵ <atomic>
#ਅਨ੍ਤਰ੍ਭਾਵ <cassert>
#ਅਨ੍ਤਰ੍ਭਾਵ <cerrno>
#ਅਨ੍ਤਰ੍ਭਾਵ <cmath>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdio>  // for GGML_ASSERT
#ਅਨ੍ਤਰ੍ਭਾਵ <stdexcept>
#ਅਨ੍ਤਰ੍ਭਾਵ <thread>
// clang-format off
#ਜੇ defined(__riscv)

#ਜੇ !defined(__riscv_v) || !defined(__riscv_v_intrinsic)
#ਗਲਤੀ "riscv v extension or v_intrinsic not enabled"
#ਨਹੀਂਤਾਂ
#ਅਨ੍ਤਰ੍ਭਾਵ <riscv_vector.h>
#ਜੇ_ਅੰਤ

#ਜੇ !defined(__riscv_zfh) || !defined(__riscv_zvfh)
#ਗਲਤੀ "riscv zfh extension not enabled, GGML_RV_ZFH and GGML_RV_ZVFH must be defined to 1"
#ਜੇ_ਅੰਤ

#ਜੇ !defined(__riscv_zba)
#ਗਲਤੀ "riscv zba extension not enabled, GGML_RV_ZBA must be defined to 1"
#ਜੇ_ਅੰਤ

#ਜੇ defined(RISCV64_SPACEMIT_IME1) || defined(RISCV64_SPACEMIT_IME2)
#ਨਹੀਂਤਾਂ
#ਗਲਤੀ "RISCV64_SPACEMIT_IME1 or RISCV64_SPACEMIT_IME2 not defined"
#ਜੇ_ਅੰਤ

#ਨਹੀਂਤਾਂ

#ਗਲਤੀ "riscv not enabled in this build"

#ਜੇ_ਅੰਤ

#ਜੇ defined(__GNUC__)
#ਹਦਾਇਤ GCC diagnostic ignored "-Woverlength-strings"
#ਹਦਾਇਤ GCC diagnostic ignored "-Wcast-qual"
#ਹਦਾਇਤ GCC diagnostic ignored "-Wunused-parameter"
#ਜੇ_ਅੰਤ

// clang-format on

ਬਾਹਰੀ "C" {
ਬਾਹਰੀ ਖਾਲੀ ggml_threadpool_chunk_ਸੈੱਟ(ਬਣਤਰ ggml_threadpool * tp, ਪੂਰਨ value);
ਬਾਹਰੀ ਪੂਰਨ  ggml_threadpool_chunk_ਜੋੜੋ(ਬਣਤਰ ggml_threadpool * tp, ਪੂਰਨ value);
}

ਨਾਮਾਕਾਸ਼ ggml::cpu::riscv64_spacemit {

ਬਣਤਰ TLSContext {
    ਪੂਰਨ       cpu_id{ -1 };
    cpu_set_t cpuset;
    ਖਾਲੀ *    tcm_ਬਫਰ{ ਖਾਲੀ_ਸੂਚਕ };
    size_t    tcm_ਬਫਰ_ਮਾਪ{ 0 };
};

thread_local TLSContext tls_context;

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ BLOC_TYPE, int64_t INTER_ਮਾਪ, int64_t NB_ਖਾਨੇ_ਗਿਣਤੀ> constexpr size_t get_repacked_ਖੰਡ_type_ਮਾਪ() {
    ਜੇ constexpr (std::is_same_v<BLOC_TYPE, ਖੰਡ_q6_K> || std::is_same_v<BLOC_TYPE, ਖੰਡ_q8_0>) {
        ਵਾਪਸ ਆਕਾਰ(ਖੰਡ_q8_0);
    } ਨਹੀਂਤਾਂ ਜੇ constexpr (std::is_same_v<BLOC_TYPE, ਖੰਡ_q4_0>) {
        ਵਾਪਸ ਆਕਾਰ(ਖੰਡ_q4_0) * INTER_ਮਾਪ / QK4_0;
    } ਨਹੀਂਤਾਂ ਜੇ constexpr (std::is_same_v<BLOC_TYPE, ਖੰਡ_q4_1> || std::is_same_v<BLOC_TYPE, ਖੰਡ_q4_K>) {
        ਵਾਪਸ (ਆਕਾਰ(ਖੰਡ_q4_0) + ਆਕਾਰ(uint8_t)) * INTER_ਮਾਪ / QK4_1;
    } ਨਹੀਂਤਾਂ ਜੇ constexpr (std::is_same_v<BLOC_TYPE, ਖੰਡ_q2_K>) {
        ਵਾਪਸ ਆਕਾਰ(spacemit_kernels::nrow_ਖੰਡ_q2_k<1>);
    } ਨਹੀਂਤਾਂ ਜੇ constexpr (std::is_same_v<BLOC_TYPE, ਖੰਡ_q3_K>) {
        ਵਾਪਸ ਆਕਾਰ(spacemit_kernels::nrow_ਖੰਡ_q3_k<1>);
    } ਨਹੀਂਤਾਂ ਜੇ constexpr (std::is_same_v<BLOC_TYPE, ਖੰਡ_mxfp4>) {
        ਵਾਪਸ ਆਕਾਰ(spacemit_kernels::nrow_ਖੰਡ_mxfp4<1>);
    } ਨਹੀਂਤਾਂ ਜੇ constexpr (std::is_same_v<BLOC_TYPE, ਖੰਡ_q5_1> || std::is_same_v<BLOC_TYPE, ਖੰਡ_q5_K>) {
        ਵਾਪਸ ਆਕਾਰ(spacemit_kernels::nrow_ਖੰਡ_q5_1<1>);
    } ਨਹੀਂਤਾਂ ਜੇ constexpr (std::is_same_v<BLOC_TYPE, ਖੰਡ_q5_0>) {
        ਵਾਪਸ ਆਕਾਰ(spacemit_kernels::nrow_ਖੰਡ_q5_0<1>);
    } ਨਹੀਂਤਾਂ {
        ਪ੍ਰਤਿਜ੍ਞਾ(ਝੂਠ);
        ਵਾਪਸ 0;
    }
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ BLOC_TYPE> constexpr ਤਾਰਕਿਕ ਖੰਡ_type_has_zp() {
    ਜੇ constexpr (std::is_same_v<BLOC_TYPE, ਖੰਡ_q6_K> || std::is_same_v<BLOC_TYPE, ਖੰਡ_q8_0> ||
                  std::is_same_v<BLOC_TYPE, ਖੰਡ_q3_K> || std::is_same_v<BLOC_TYPE, ਖੰਡ_q4_0> ||
                  std::is_same_v<BLOC_TYPE, ਖੰਡ_mxfp4> || std::is_same_v<BLOC_TYPE, ਖੰਡ_q5_0>) {
        ਵਾਪਸ ਝੂਠ;
    } ਨਹੀਂਤਾਂ ਜੇ constexpr (std::is_same_v<BLOC_TYPE, ਖੰਡ_q4_1> || std::is_same_v<BLOC_TYPE, ਖੰਡ_q4_K> ||
                         std::is_same_v<BLOC_TYPE, ਖੰਡ_q2_K> || std::is_same_v<BLOC_TYPE, ਖੰਡ_q5_1> ||
                         std::is_same_v<BLOC_TYPE, ਖੰਡ_q5_K>) {
        ਵਾਪਸ ਸੱਚ;
    } ਨਹੀਂਤਾਂ {
        ਪ੍ਰਤਿਜ੍ਞਾ(ਝੂਠ);
        ਵਾਪਸ ਝੂਠ;
    }
}

ਵਰਗ tensor_traits_ਅਧਾਰ : ਜਨਤਕ ggml::cpu::tensor_traits {
  ਜਨਤਕ:
    ਆਭਾਸੀ ਪੂਰਨ repack(ggml_tensor * t, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਅੰਕੜੇ_ਮਾਪ) = 0;
};

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ BLOC_TYPE, int64_t INTER_ਮਾਪ, int64_t NB_ਖਾਨੇ_ਗਿਣਤੀ> ਵਰਗ tensor_traits : ਜਨਤਕ tensor_traits_ਅਧਾਰ {
    ਤਾਰਕਿਕ work_ਮਾਪ(ਪੂਰਨ /* n_threads */, ਅਚਲ ggml_tensor * op, size_t & size) override {
        ਚੋਣ (op->op) {
            ਹਾਲਤ GGML_OP_MUL_MAT:
                {
                    int64_t src1_nelements = ggml_nelements(op->src[1]);

                    ਜੇ constexpr (std::is_same_v<BLOC_TYPE, ਖੰਡ_q2_K> || std::is_same_v<BLOC_TYPE, ਖੰਡ_q3_K>) {
                        size =
                            spacemit_kernels::div_ਗੇੜਾ_ਉੱਪਰ(src1_nelements, QK_K) * spacemit_kernels::q8k_blk_ਮਾਪ(QK_K);
                    } ਨਹੀਂਤਾਂ ਜੇ constexpr (INTER_ਮਾਪ == QK4_0) {
                        size = spacemit_kernels::div_ਗੇੜਾ_ਉੱਪਰ(src1_nelements, QK4_0) *
                               spacemit_kernels::q8_blk_ਮਾਪ(QK4_0, ਸੱਚ);
                    } ਨਹੀਂਤਾਂ ਜੇ constexpr (INTER_ਮਾਪ == 256) {
                        size = spacemit_kernels::div_ਗੇੜਾ_ਉੱਪਰ(src1_nelements, 256) *
                               spacemit_kernels::q8_hp_blk_ਮਾਪ(256, ਸੱਚ, ਸੱਚ);
                    } ਨਹੀਂਤਾਂ {
                        GGML_ABORT("unsupported block type");
                    }

                    size = GGML_PAD(size, ਆਕਾਰ(int64_t));

                    ਵਾਪਸ ਸੱਚ;
                }
            ਹਾਲਤ GGML_OP_MUL_MAT_ID:
                {
                    int64_t src1_nelements = ggml_nelements(op->src[1]);

                    ਜੇ constexpr (std::is_same_v<BLOC_TYPE, ਖੰਡ_q2_K> || std::is_same_v<BLOC_TYPE, ਖੰਡ_q3_K>) {
                        size =
                            spacemit_kernels::div_ਗੇੜਾ_ਉੱਪਰ(src1_nelements, QK_K) * spacemit_kernels::q8k_blk_ਮਾਪ(QK_K);
                    } ਨਹੀਂਤਾਂ ਜੇ constexpr (INTER_ਮਾਪ == QK4_0) {
                        size = spacemit_kernels::div_ਗੇੜਾ_ਉੱਪਰ(src1_nelements, QK4_0) *
                               spacemit_kernels::q8_blk_ਮਾਪ(QK4_0, ਸੱਚ);
                    } ਨਹੀਂਤਾਂ ਜੇ constexpr (INTER_ਮਾਪ == 256) {
                        size = spacemit_kernels::div_ਗੇੜਾ_ਉੱਪਰ(src1_nelements, 256) *
                               spacemit_kernels::q8_hp_blk_ਮਾਪ(256, ਸੱਚ, ਸੱਚ);
                    } ਨਹੀਂਤਾਂ {
                        GGML_ABORT("unsupported block type");
                    }

                    size = GGML_PAD(size, ਆਕਾਰ(int64_t));

                    ਅਚਲ int64_t ne02 = op->src[0]->ne[2];  // n_as, n_expert
                    ਅਚਲ int64_t ne12 = op->src[1]->ne[2];  // n_tokens

                    ਅਚਲ size_t sizeof_mmid_ਪੰਕਤੀ_mapping = ਆਕਾਰ(int64_t);
                    size += sizeof_mmid_ਪੰਕਤੀ_mapping * ne02 * (ne12 + 1) + (ne02 + 1) * ਆਕਾਰ(int64_t);

                    size = GGML_PAD(size, ਆਕਾਰ(int64_t));

                    ਵਾਪਸ ਸੱਚ;
                }
            ਮੂਲ:
                // GGML_ABORT("fatal error");
                ਰੋਕੋ;
        }
        ਵਾਪਸ ਝੂਠ;
    }

    ਤਾਰਕਿਕ compute_forward(ggml_compute_params * params, ggml_tensor * op) override {
        ਚੋਣ (op->op) {
            ਹਾਲਤ GGML_OP_MUL_MAT:
                ਚੋਣ (op->src[0]->type) {
                    ਹਾਲਤ GGML_TYPE_Q2_K:
                    ਹਾਲਤ GGML_TYPE_Q3_K:
                    ਹਾਲਤ GGML_TYPE_Q4_0:
                    ਹਾਲਤ GGML_TYPE_Q4_1:
                    ਹਾਲਤ GGML_TYPE_Q4_K:
                    ਹਾਲਤ GGML_TYPE_Q6_K:
                    ਹਾਲਤ GGML_TYPE_Q8_0:
                    ਹਾਲਤ GGML_TYPE_Q5_0:
                    ਹਾਲਤ GGML_TYPE_Q5_1:
                    ਹਾਲਤ GGML_TYPE_Q5_K:
                        //case GGML_TYPE_MXFP4:
                        forward_mul_mat(params, op);
                        ਵਾਪਸ ਸੱਚ;
                    ਮੂਲ:
                        // GGML_ABORT("fatal error: unsupported type for src0 in MUL_MAT");
                        ਵਾਪਸ ਝੂਠ;
                }
                ਰੋਕੋ;
            ਹਾਲਤ GGML_OP_MUL_MAT_ID:
                ਚੋਣ (op->src[0]->type) {
                    ਹਾਲਤ GGML_TYPE_Q2_K:
                    ਹਾਲਤ GGML_TYPE_Q3_K:
                    ਹਾਲਤ GGML_TYPE_Q4_0:
                    ਹਾਲਤ GGML_TYPE_Q4_1:
                    ਹਾਲਤ GGML_TYPE_Q4_K:
                    ਹਾਲਤ GGML_TYPE_Q6_K:
                    ਹਾਲਤ GGML_TYPE_Q8_0:
                    ਹਾਲਤ GGML_TYPE_Q5_0:
                    ਹਾਲਤ GGML_TYPE_Q5_1:
                    ਹਾਲਤ GGML_TYPE_Q5_K:
                        //case GGML_TYPE_MXFP4:
                        forward_mul_mat_id(params, op);
                        ਵਾਪਸ ਸੱਚ;
                    ਮੂਲ:
                        // GGML_ABORT("fatal error: unsupported type for src0 in MUL_MAT_ID");
                        ਵਾਪਸ ਝੂਠ;
                }
                ਰੋਕੋ;
            ਮੂਲ:
                // GGML_ABORT("fatal error");
                ਰੋਕੋ;
        }
        ਵਾਪਸ ਝੂਠ;
    }

    ਖਾਲੀ forward_mul_mat(ggml_compute_params * params, ggml_tensor * op) {
        constexpr size_t a_blk_len = INTER_ਮਾਪ;
        constexpr size_t b_blk_len = INTER_ਮਾਪ;

        ਅਚਲ ggml_tensor * src0 = op->src[0];
        ਅਚਲ ggml_tensor * src1 = op->src[1];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_ਦੁਆਧਾਰੀ_OP_LOCALS

        ਪੂਰਨ ith = params->ith;
        ਪੂਰਨ nth = params->nth;

        [[maybe_unused]] ਅਚਲ ਗਿਣਤੀ ggml_type type = src0->type;

        ਖਾਲੀ *        w_ਅੰਕੜੇ  = (ਖਾਲੀ *) src0->ਅੰਕੜੇ;
        ਅਚਲ ਦਸ਼ਮਲਵ * feature = (ਅਚਲ ਦਸ਼ਮਲਵ *) src1->ਅੰਕੜੇ;
        ਦਸ਼ਮਲਵ *       output  = (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ;

        ਅਚਲ int64_t gemm_m = ne11 * ne12 * ne13;
        ਅਚਲ int64_t gemm_k = ne10;
        ਅਚਲ int64_t gemm_n = ne01;

        spacemit_kernels::quantize_a_ਪੰਕਤੀ_def       quantize_a_ਪੰਕਤੀ_i8;
        spacemit_kernels::quantize_a_ਪੰਕਤੀ_def       quantize_a_4row_i8;
        spacemit_kernels::gemm_kernel_quantize_def gemm_kernel;
        ਤਾਰਕਿਕ                                       ਸੈੱਟ_kernel_impl = ਝੂਠ;

        int64_t ਖੰਡ_stride_a = spacemit_kernels::q8_blk_ਮਾਪ(a_blk_len);

#ਜੇ defined(RISCV64_SPACEMIT_IME2)
        ਜੇ (!ਸੈੱਟ_kernel_impl && (global_spine_env_info.use_ime2)) {
            quantize_a_ਪੰਕਤੀ_i8  = spacemit_kernels::rvv::quantize_a_ਪੰਕਤੀ_i8;
            quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8;
            ਖੰਡ_stride_a     = spacemit_kernels::q8_blk_ਮਾਪ(a_blk_len, ਸੱਚ);

            ਜੇ constexpr (std::is_same_v<BLOC_TYPE, ਖੰਡ_q6_K> || std::is_same_v<BLOC_TYPE, ਖੰਡ_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i8;
                ਸੈੱਟ_kernel_impl = ਸੱਚ;
            } ਨਹੀਂਤਾਂ ਜੇ constexpr (std::is_same_v<BLOC_TYPE, ਖੰਡ_q4_0> || std::is_same_v<BLOC_TYPE, ਖੰਡ_q4_1> ||
                                 std::is_same_v<BLOC_TYPE, ਖੰਡ_q4_K>) {
                ਜੇ constexpr (INTER_ਮਾਪ == 256) {
                    gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i4_hp;
                    quantize_a_ਪੰਕਤੀ_i8  = spacemit_kernels::rvv::quantize_a_ਪੰਕਤੀ_i8_hp;
                    quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8_hp;
                    ਖੰਡ_stride_a     = spacemit_kernels::q8_hp_blk_ਮਾਪ(a_blk_len, ਸੱਚ, ਸੱਚ);
                    ਸੈੱਟ_kernel_impl    = ਸੱਚ;
                } ਨਹੀਂਤਾਂ {
                    gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i4;
                    quantize_a_ਪੰਕਤੀ_i8  = spacemit_kernels::rvv::quantize_a_ਪੰਕਤੀ_i8;
                    quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8;
                    ਖੰਡ_stride_a     = spacemit_kernels::q8_blk_ਮਾਪ(a_blk_len, ਸੱਚ);
                    ਸੈੱਟ_kernel_impl    = ਸੱਚ;
                }
            } ਨਹੀਂਤਾਂ ਜੇ constexpr (std::is_same_v<BLOC_TYPE, ਖੰਡ_q2_K>) {
                quantize_a_ਪੰਕਤੀ_i8  = spacemit_kernels::rvv::quantize_a_ਪੰਕਤੀ_i8k;
                quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8k;
                ਖੰਡ_stride_a     = spacemit_kernels::q8k_blk_ਮਾਪ(a_blk_len);

                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i2k;
                ਸੈੱਟ_kernel_impl = ਸੱਚ;
            } ਨਹੀਂਤਾਂ ਜੇ constexpr (std::is_same_v<BLOC_TYPE, ਖੰਡ_q3_K>) {
                quantize_a_ਪੰਕਤੀ_i8  = spacemit_kernels::rvv::quantize_a_ਪੰਕਤੀ_i8k;
                quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8k;
                ਖੰਡ_stride_a     = spacemit_kernels::q8k_blk_ਮਾਪ(a_blk_len);

                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i3k;
                ਸੈੱਟ_kernel_impl = ਸੱਚ;
            } ਨਹੀਂਤਾਂ ਜੇ constexpr (std::is_same_v<BLOC_TYPE, ਖੰਡ_mxfp4>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8mxfp4;
                ਸੈੱਟ_kernel_impl = ਸੱਚ;
            } ਨਹੀਂਤਾਂ ਜੇ constexpr (std::is_same_v<BLOC_TYPE, ਖੰਡ_q5_1> || std::is_same_v<BLOC_TYPE, ਖੰਡ_q5_K> ||
                                 std::is_same_v<BLOC_TYPE, ਖੰਡ_q5_0>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i5;
                ਸੈੱਟ_kernel_impl = ਸੱਚ;
            }
        }
#ਜੇ_ਅੰਤ

#ਜੇ defined(RISCV64_SPACEMIT_IME1)
        ਜੇ (!ਸੈੱਟ_kernel_impl && (global_spine_env_info.use_ime1)) {
            quantize_a_ਪੰਕਤੀ_i8  = spacemit_kernels::ime1::quantize_a_ਪੰਕਤੀ_i8;
            quantize_a_4row_i8 = spacemit_kernels::ime1::quantize_a_4row_i8;

            ਜੇ constexpr (std::is_same_v<BLOC_TYPE, ਖੰਡ_q4_0> || std::is_same_v<BLOC_TYPE, ਖੰਡ_q4_1> ||
                          std::is_same_v<BLOC_TYPE, ਖੰਡ_q4_K>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i4;
                ਸੈੱਟ_kernel_impl = ਸੱਚ;
            } ਨਹੀਂਤਾਂ ਜੇ constexpr (std::is_same_v<BLOC_TYPE, ਖੰਡ_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i8;
                ਸੈੱਟ_kernel_impl = ਸੱਚ;
            }
        }
#ਜੇ_ਅੰਤ
        ਜੇ (!ਸੈੱਟ_kernel_impl) {
            GGML_ABORT("no kernel implementation found for the block type");
        }

        ਅਚਲ int64_t a_k_blks = spacemit_kernels::div_ਗੇੜਾ_ਉੱਪਰ(gemm_k, a_blk_len);
        ਅਚਲ int64_t b_k_blks = spacemit_kernels::div_ਗੇੜਾ_ਉੱਪਰ(gemm_k, b_blk_len);

        ਅਚਲ int64_t ਪੰਕਤੀ_stride_a        = a_k_blks * ਖੰਡ_stride_a;
        ਅਚਲ int64_t gemm_workspace_ਮਾਪ = GGML_PAD(gemm_m * ਪੰਕਤੀ_stride_a, alignof(int64_t));

        ਜੇ (ith == 0 && params->wsize < gemm_workspace_ਮਾਪ) {
            GGML_ABORT("wsize less than gemm_workspace_size");
        }

        uintptr_t ws_ptr = reinterpret_cast<uintptr_t>(params->wdata);

        ਖਾਲੀ *        tcm_ਬਫਰ      = ggml::cpu::riscv64_spacemit::tls_context.tcm_ਬਫਰ;
        ਅਚਲ int64_t tcm_ਬਫਰ_ਮਾਪ = ggml::cpu::riscv64_spacemit::tls_context.tcm_ਬਫਰ_ਮਾਪ;

        ਸਵੈਚਾਲਿਤ * quant_a_ਬਫਰ = reinterpret_cast<uint8_t *>(ws_ptr);

        constexpr int64_t ਪੰਕਤੀ_align = 4;
        ਅਚਲ int64_t     ਪੰਕਤੀ_blks  = spacemit_kernels::div_ਗੇੜਾ_ਉੱਪਰ(gemm_m, ਪੰਕਤੀ_align);

        ਅਚਲ int64_t ਪੰਕਤੀ_stride_b      = b_k_blks * get_repacked_ਖੰਡ_type_ਮਾਪ<BLOC_TYPE, INTER_ਮਾਪ, NB_ਖਾਨੇ_ਗਿਣਤੀ>();
        ਅਚਲ int64_t ਪ੍ਰਤੀ_mb_ਪੰਕਤੀਆਂ_wsize = ਪੰਕਤੀ_align * ਪੰਕਤੀ_stride_a;
        ਅਚਲ int64_t ਪ੍ਰਤੀ_nb_ਖਾਨੇ_ਗਿਣਤੀ_wsize = NB_ਖਾਨੇ_ਗਿਣਤੀ * ਪੰਕਤੀ_stride_b;

        ਅਚਲ int64_t barrier_idx = static_cast<int64_t>(ith / 2);

        GGML_ASSERT(global_spine_env_info.init_barrier != ਖਾਲੀ_ਸੂਚਕ);
        GGML_ASSERT(barrier_idx < spine_init_barrier_ਗਿਣਤੀ);
        spine_barrier_t * cur_barrier = &global_spine_env_info.init_barrier[barrier_idx];

        ਜੇ (gemm_m == 1) {
            ਪੂਰਨ task_ਪ੍ਰਤੀ_thread = spacemit_kernels::div_ਗੇੜਾ_ਉੱਪਰ(a_k_blks, nth);
            ਪੂਰਨ a_blk_ਸ਼ੁਰੂ     = ith * task_ਪ੍ਰਤੀ_thread;
            ਪੂਰਨ a_blk_ਸਿਰਾ       = std::min(a_blk_ਸ਼ੁਰੂ + task_ਪ੍ਰਤੀ_thread, (ਪੂਰਨ) a_k_blks);
            ਜੇ (a_blk_ਸ਼ੁਰੂ < a_blk_ਸਿਰਾ) {
                quantize_a_ਪੰਕਤੀ_i8(a_blk_len, feature + a_blk_ਸ਼ੁਰੂ * a_blk_len, (a_blk_ਸਿਰਾ - a_blk_ਸ਼ੁਰੂ) * a_blk_len,
                                  quant_a_ਬਫਰ + a_blk_ਸ਼ੁਰੂ * ਖੰਡ_stride_a);
            }
        } ਨਹੀਂਤਾਂ {
            ਪੂਰਨ task_ਪ੍ਰਤੀ_thread = spacemit_kernels::div_ਗੇੜਾ_ਉੱਪਰ(ਪੰਕਤੀ_blks, nth);
            ਪੂਰਨ m_ਪੰਕਤੀ_blk_ਸ਼ੁਰੂ = ith * task_ਪ੍ਰਤੀ_thread;
            ਪੂਰਨ m_ਪੰਕਤੀ_blk_ਸਿਰਾ   = std::min(m_ਪੰਕਤੀ_blk_ਸ਼ੁਰੂ + task_ਪ੍ਰਤੀ_thread, (ਪੂਰਨ) ਪੰਕਤੀ_blks);
            ਲਈ (ਪੂਰਨ m_ਪੰਕਤੀ_blk = m_ਪੰਕਤੀ_blk_ਸ਼ੁਰੂ; m_ਪੰਕਤੀ_blk < m_ਪੰਕਤੀ_blk_ਸਿਰਾ; m_ਪੰਕਤੀ_blk++) {
                ਪੂਰਨ m_idx             = m_ਪੰਕਤੀ_blk * ਪੰਕਤੀ_align;
                ਪੂਰਨ ਪੰਕਤੀਆਂ_tobe_handled = (gemm_m - m_idx) > ਪੰਕਤੀ_align ? ਪੰਕਤੀ_align : (gemm_m - m_idx);

                ਜੇ (ਪੰਕਤੀਆਂ_tobe_handled == ਪੰਕਤੀ_align && quantize_a_4row_i8 != ਖਾਲੀ_ਸੂਚਕ) {
                    ਅਚਲ ਦਸ਼ਮਲਵ * a_ਪੰਕਤੀ_ptr       = feature + m_idx * gemm_k;
                    ਸਵੈਚਾਲਿਤ *        quant_a_ਪੰਕਤੀ_ptr = quant_a_ਬਫਰ + m_idx * ਪੰਕਤੀ_stride_a;
                    quantize_a_4row_i8(a_blk_len, a_ਪੰਕਤੀ_ptr, gemm_k, quant_a_ਪੰਕਤੀ_ptr);
                } ਨਹੀਂਤਾਂ {
                    ਜਦੋਂ (ਪੰਕਤੀਆਂ_tobe_handled) {
                        ਅਚਲ ਦਸ਼ਮਲਵ * a_ਪੰਕਤੀ_ptr       = feature + m_idx * gemm_k;
                        ਸਵੈਚਾਲਿਤ *        quant_a_ਪੰਕਤੀ_ptr = quant_a_ਬਫਰ + m_idx * ਪੰਕਤੀ_stride_a;
                        quantize_a_ਪੰਕਤੀ_i8(a_blk_len, a_ਪੰਕਤੀ_ptr, gemm_k, quant_a_ਪੰਕਤੀ_ptr);
                        ਪੰਕਤੀਆਂ_tobe_handled -= 1;
                        m_idx += 1;
                    }
                }
            }
        }

        ggml_barrier(params->threadpool);

        ਅਚਲ int64_t gemm_m_stride     = gemm_n / gemm_m > 64 ? gemm_m : 16;
        ਅਚਲ int64_t gemm_m_blocked    = spacemit_kernels::div_ਗੇੜਾ_ਉੱਪਰ(gemm_m, gemm_m_stride);
        ਅਚਲ int64_t max_gemm_n_stride = spacemit_kernels::div_ਗੇੜਾ_ਉੱਪਰ(gemm_n * gemm_m_blocked, nth);

        int64_t gemm_n_stride = gemm_n;
        ਜੇ (max_gemm_n_stride < gemm_n) {
            gemm_n_stride =
                std::min(gemm_n_stride, spacemit_kernels::div_ਗੇੜਾ_ਉੱਪਰ(max_gemm_n_stride, NB_ਖਾਨੇ_ਗਿਣਤੀ) * NB_ਖਾਨੇ_ਗਿਣਤੀ);
        }

        ਜੇ (gemm_n_stride == gemm_n && tcm_ਬਫਰ != ਖਾਲੀ_ਸੂਚਕ && ਪ੍ਰਤੀ_mb_ਪੰਕਤੀਆਂ_wsize <= tcm_ਬਫਰ_ਮਾਪ) {
            ਲਈ (int64_t m_ਸ਼ੁਰੂ = ith * ਪੰਕਤੀ_align; m_ਸ਼ੁਰੂ < gemm_m; m_ਸ਼ੁਰੂ += ਪੰਕਤੀ_align * nth) {
                uint8_t * b_ਖਾਨਾ    = reinterpret_cast<uint8_t *>(w_ਅੰਕੜੇ);
                uint8_t * b_ਖਾਨਾ_zp = ਖੰਡ_type_has_zp<BLOC_TYPE>() ? b_ਖਾਨਾ : ਖਾਲੀ_ਸੂਚਕ;

                int64_t m_ਪੰਕਤੀ_real = std::min(gemm_m - m_ਸ਼ੁਰੂ, ਪੰਕਤੀ_align);

                spacemit_kernels::rvv::memcpy1d(tcm_ਬਫਰ, quant_a_ਬਫਰ + m_ਸ਼ੁਰੂ * ਪੰਕਤੀ_stride_a,
                                                m_ਪੰਕਤੀ_real * ਪੰਕਤੀ_stride_a);

                int64_t n_blk_real = 0;
                ਲਈ (int64_t ni = 0; ni < gemm_n; ni += n_blk_real, b_ਖਾਨਾ += n_blk_real * ਪੰਕਤੀ_stride_b) {
                    n_blk_real = std::min(gemm_n - ni, (int64_t) NB_ਖਾਨੇ_ਗਿਣਤੀ);

                    uint8_t * a_ਪੰਕਤੀ_ptr = (uint8_t *) tcm_ਬਫਰ;
                    ਦਸ਼ਮਲਵ *   c_blk     = output + m_ਸ਼ੁਰੂ * gemm_n + ni;

                    int32_t ਪੰਕਤੀਆਂ_ਬਚਿਆ = m_ਪੰਕਤੀ_real;

                    ਜਦੋਂ (ਪੰਕਤੀਆਂ_ਬਚਿਆ > 0) {
                        ਸਵੈਚਾਲਿਤ ਪੰਕਤੀਆਂ_handled = gemm_kernel(b_blk_len, a_ਪੰਕਤੀ_ptr, b_ਖਾਨਾ, b_ਖਾਨਾ_zp, c_blk, ਪੰਕਤੀਆਂ_ਬਚਿਆ,
                                                        n_blk_real, b_k_blks, gemm_n);

                        c_blk += ਪੰਕਤੀਆਂ_handled * gemm_n;
                        a_ਪੰਕਤੀ_ptr += ਪੰਕਤੀਆਂ_handled * ਪੰਕਤੀ_stride_a;

                        ਪੰਕਤੀਆਂ_ਬਚਿਆ -= ਪੰਕਤੀਆਂ_handled;
                    }
                }
            }
        } ਨਹੀਂਤਾਂ ਜੇ (tcm_ਬਫਰ != ਖਾਲੀ_ਸੂਚਕ && ਪ੍ਰਤੀ_nb_ਖਾਨੇ_ਗਿਣਤੀ_wsize <= tcm_ਬਫਰ_ਮਾਪ) {
            uint8_t * a_ਪੰਕਤੀ = quant_a_ਬਫਰ;
            uint8_t * b_ਖਾਨਾ = reinterpret_cast<uint8_t *>(tcm_ਬਫਰ);
            ਜੇ ((gemm_workspace_ਮਾਪ + ਪ੍ਰਤੀ_nb_ਖਾਨੇ_ਗਿਣਤੀ_wsize) <= tcm_ਬਫਰ_ਮਾਪ) {
                a_ਪੰਕਤੀ = (uint8_t *) tcm_ਬਫਰ;
                b_ਖਾਨਾ = reinterpret_cast<uint8_t *>(tcm_ਬਫਰ) + gemm_workspace_ਮਾਪ;
            }
            uint8_t * b_ਖਾਨਾ_zp = ਖੰਡ_type_has_zp<BLOC_TYPE>() ? b_ਖਾਨਾ : ਖਾਲੀ_ਸੂਚਕ;

            int64_t ni      = ith * NB_ਖਾਨੇ_ਗਿਣਤੀ;
            int64_t nb_real = std::min(gemm_n - ni, NB_ਖਾਨੇ_ਗਿਣਤੀ);

            ਜੇ (ith % 2 == 0 && nb_real > 0) {
                spacemit_kernels::rvv::memcpy1d(b_ਖਾਨਾ, reinterpret_cast<uint8_t *>(w_ਅੰਕੜੇ) + ni * ਪੰਕਤੀ_stride_b,
                                                nb_real * ਪੰਕਤੀ_stride_b);
                ਜੇ (a_ਪੰਕਤੀ != quant_a_ਬਫਰ) {
                    spacemit_kernels::rvv::memcpy1d(a_ਪੰਕਤੀ, quant_a_ਬਫਰ, gemm_workspace_ਮਾਪ);
                }
            }

            spine_barrier_wait(cur_barrier);

            ਜੇ (ith % 2 != 0 && nb_real > 0) {
                ਜੇ (a_ਪੰਕਤੀ != quant_a_ਬਫਰ) {
                    spacemit_kernels::rvv::memcpy1d(a_ਪੰਕਤੀ, quant_a_ਬਫਰ, gemm_workspace_ਮਾਪ);
                }
                spacemit_kernels::rvv::memcpy1d(b_ਖਾਨਾ, reinterpret_cast<uint8_t *>(w_ਅੰਕੜੇ) + ni * ਪੰਕਤੀ_stride_b,
                                                nb_real * ਪੰਕਤੀ_stride_b);
            }

            ਲਈ (; ni < gemm_n; ni += NB_ਖਾਨੇ_ਗਿਣਤੀ * nth) {
                int64_t ਪੰਕਤੀਆਂ_ਬਚਿਆ = gemm_m;
                ਦਸ਼ਮਲਵ * c_blk          = output + ni;
                ਸਵੈਚਾਲਿਤ *  a_ਪੰਕਤੀ_cur      = a_ਪੰਕਤੀ;

                ਜੇ (ith % 2 != 0) {
                    spine_barrier_wait(cur_barrier);
                }

                ਜਦੋਂ (ਪੰਕਤੀਆਂ_ਬਚਿਆ > 0) {
                    ਸਵੈਚਾਲਿਤ ਪੰਕਤੀਆਂ_handled = gemm_kernel(b_blk_len, a_ਪੰਕਤੀ_cur, b_ਖਾਨਾ, b_ਖਾਨਾ_zp, c_blk, ਪੰਕਤੀਆਂ_ਬਚਿਆ,
                                                    nb_real, b_k_blks, gemm_n);

                    c_blk += ਪੰਕਤੀਆਂ_handled * gemm_n;
                    a_ਪੰਕਤੀ_cur += ਪੰਕਤੀਆਂ_handled * ਪੰਕਤੀ_stride_a;

                    ਪੰਕਤੀਆਂ_ਬਚਿਆ -= ਪੰਕਤੀਆਂ_handled;
                }

                ਜੇ (ith % 2 == 0) {
                    spine_barrier_wait(cur_barrier);
                }

                ਅਚਲ int64_t ਅਗਲਾ_ni = ni + NB_ਖਾਨੇ_ਗਿਣਤੀ * nth;
                ਜੇ (ਅਗਲਾ_ni < gemm_n) {
                    nb_real = std::min(gemm_n - ਅਗਲਾ_ni, NB_ਖਾਨੇ_ਗਿਣਤੀ);
                    spacemit_kernels::rvv::memcpy1d(b_ਖਾਨਾ, reinterpret_cast<uint8_t *>(w_ਅੰਕੜੇ) + ਅਗਲਾ_ni * ਪੰਕਤੀ_stride_b,
                                                    nb_real * ਪੰਕਤੀ_stride_b);
                }
            }
        } ਨਹੀਂਤਾਂ {
            ਅਚਲ int64_t task_ਗਿਣਤੀ_m = spacemit_kernels::div_ਗੇੜਾ_ਉੱਪਰ(gemm_m, gemm_m_stride);
            ਅਚਲ int64_t task_ਗਿਣਤੀ_n = spacemit_kernels::div_ਗੇੜਾ_ਉੱਪਰ(gemm_n, gemm_n_stride);

            int64_t task_ਗਿਣਤੀ      = task_ਗਿਣਤੀ_m * task_ਗਿਣਤੀ_n;
            int64_t task_ਪ੍ਰਤੀ_thread = (task_ਗਿਣਤੀ + nth - 1) / nth;
            int64_t start           = ith * task_ਪ੍ਰਤੀ_thread;
            int64_t end             = std::min((ith + 1) * task_ਪ੍ਰਤੀ_thread, task_ਗਿਣਤੀ);
            ਲਈ (int64_t compute_idx = start; compute_idx < end; compute_idx++) {
                ਅਚਲ ਸਵੈਚਾਲਿਤ tid_n = compute_idx / task_ਗਿਣਤੀ_m;
                ਅਚਲ ਸਵੈਚਾਲਿਤ tid_m = compute_idx % task_ਗਿਣਤੀ_m;

                ਅਚਲ int64_t m_ਸ਼ੁਰੂ = tid_m * gemm_m_stride;
                ਅਚਲ int64_t m_ਗਿਣਤੀ = std::min(gemm_m - m_ਸ਼ੁਰੂ, (int64_t) gemm_m_stride);

                ਅਚਲ int64_t n_ਸ਼ੁਰੂ = tid_n * gemm_n_stride;
                ਅਚਲ int64_t n_ਗਿਣਤੀ = std::min(gemm_n - n_ਸ਼ੁਰੂ, (int64_t) gemm_n_stride);

                ਅਚਲ int64_t n_blk = m_ਗਿਣਤੀ == 1 ? n_ਗਿਣਤੀ : NB_ਖਾਨੇ_ਗਿਣਤੀ;

                uint8_t * b_ਖਾਨਾ    = reinterpret_cast<uint8_t *>(w_ਅੰਕੜੇ) + n_ਸ਼ੁਰੂ * ਪੰਕਤੀ_stride_b;
                uint8_t * b_ਖਾਨਾ_zp = ਖੰਡ_type_has_zp<BLOC_TYPE>() ? b_ਖਾਨਾ : ਖਾਲੀ_ਸੂਚਕ;

                int64_t n_blk_real = 0;
                ਲਈ (int64_t ni = 0; ni < n_ਗਿਣਤੀ; ni += n_blk_real, b_ਖਾਨਾ += n_blk_real * ਪੰਕਤੀ_stride_b) {
                    n_blk_real = std::min(n_ਗਿਣਤੀ - ni, n_blk);

                    uint8_t * a_ਪੰਕਤੀ = quant_a_ਬਫਰ + m_ਸ਼ੁਰੂ * ਪੰਕਤੀ_stride_a;

                    ਦਸ਼ਮਲਵ * c_blk = output + m_ਸ਼ੁਰੂ * gemm_n + n_ਸ਼ੁਰੂ + ni;

                    int64_t ਪੰਕਤੀਆਂ_ਬਚਿਆ = m_ਗਿਣਤੀ;

                    uint8_t * b_ਖਾਨਾ_cur    = b_ਖਾਨਾ;
                    uint8_t * b_ਖਾਨਾ_zp_cur = b_ਖਾਨਾ_zp;

                    ਜਦੋਂ (ਪੰਕਤੀਆਂ_ਬਚਿਆ > 0) {
                        ਸਵੈਚਾਲਿਤ ਪੰਕਤੀਆਂ_handled = gemm_kernel(b_blk_len, a_ਪੰਕਤੀ, b_ਖਾਨਾ_cur, b_ਖਾਨਾ_zp_cur, c_blk,
                                                        ਪੰਕਤੀਆਂ_ਬਚਿਆ, n_blk_real, b_k_blks, gemm_n);

                        c_blk += ਪੰਕਤੀਆਂ_handled * gemm_n;
                        a_ਪੰਕਤੀ += ਪੰਕਤੀਆਂ_handled * ਪੰਕਤੀ_stride_a;

                        ਪੰਕਤੀਆਂ_ਬਚਿਆ -= ਪੰਕਤੀਆਂ_handled;
                    }
                }
            }
        }
    }

    ਖਾਲੀ forward_mul_mat_id(ggml_compute_params * params, ggml_tensor * op) {
        constexpr size_t a_blk_len = INTER_ਮਾਪ;
        constexpr size_t b_blk_len = INTER_ਮਾਪ;

        ਅਚਲ ggml_tensor * src0 = op->src[0];
        ਅਚਲ ggml_tensor * src1 = op->src[1];
        ਅਚਲ ggml_tensor * ids  = op->src[2];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_ਦੁਆਧਾਰੀ_OP_LOCALS

        ਪੂਰਨ ith = params->ith;
        ਪੂਰਨ nth = params->nth;

        // row groups
        ਅਚਲ ਪੂਰਨ n_ids = ids->ne[0];  // n_expert_used
        ਅਚਲ ਪੂਰਨ n_as  = ne02;        // n_expert

        ਬਣਤਰ mmid_ਪੰਕਤੀ_mapping {
            int32_t i1;
            int32_t i2;
        };

        spacemit_kernels::quantize_a_ਪੰਕਤੀ_def           quantize_a_ਪੰਕਤੀ_i8;
        spacemit_kernels::gemm_kernel_quantize_def     gemm_kernel;
        spacemit_kernels::moe_gemm_kernel_quantize_def moe_gemm_kernel_m2;
        ਤਾਰਕਿਕ                                           ਸੈੱਟ_kernel_impl = ਝੂਠ;
        size_t                                         ਖੰਡ_stride_a  = spacemit_kernels::q8_blk_ਮਾਪ(QK4_0);

#ਜੇ defined(RISCV64_SPACEMIT_IME2)
        ਜੇ (!ਸੈੱਟ_kernel_impl && (global_spine_env_info.use_ime2)) {
            quantize_a_ਪੰਕਤੀ_i8 = spacemit_kernels::rvv::quantize_a_ਪੰਕਤੀ_i8;
            ਖੰਡ_stride_a    = spacemit_kernels::q8_blk_ਮਾਪ(QK4_0, ਸੱਚ);

            ਜੇ constexpr (std::is_same_v<BLOC_TYPE, ਖੰਡ_q6_K> || std::is_same_v<BLOC_TYPE, ਖੰਡ_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i8;
                ਸੈੱਟ_kernel_impl = ਸੱਚ;
            } ਨਹੀਂਤਾਂ ਜੇ constexpr (std::is_same_v<BLOC_TYPE, ਖੰਡ_q4_0> || std::is_same_v<BLOC_TYPE, ਖੰਡ_q4_1> ||
                                 std::is_same_v<BLOC_TYPE, ਖੰਡ_q4_K>) {
                ਜੇ constexpr (INTER_ਮਾਪ == 256) {
                    gemm_kernel       = spacemit_kernels::ime2::gemm_kernel_i8i4_hp;
                    quantize_a_ਪੰਕਤੀ_i8 = spacemit_kernels::rvv::quantize_a_ਪੰਕਤੀ_i8_hp;
                    ਖੰਡ_stride_a    = spacemit_kernels::q8_hp_blk_ਮਾਪ(a_blk_len, ਸੱਚ, ਸੱਚ);
                    ਸੈੱਟ_kernel_impl   = ਸੱਚ;
                } ਨਹੀਂਤਾਂ {
                    gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i4;
                    moe_gemm_kernel_m2 = spacemit_kernels::ime2::moe_m2_gemm_kernel_i8i4;
                    quantize_a_ਪੰਕਤੀ_i8  = spacemit_kernels::rvv::quantize_a_ਪੰਕਤੀ_i8;
                    ਖੰਡ_stride_a     = spacemit_kernels::q8_blk_ਮਾਪ(a_blk_len, ਸੱਚ);
                    ਸੈੱਟ_kernel_impl    = ਸੱਚ;
                }
            } ਨਹੀਂਤਾਂ ਜੇ constexpr (std::is_same_v<BLOC_TYPE, ਖੰਡ_q2_K>) {
                quantize_a_ਪੰਕਤੀ_i8 = spacemit_kernels::rvv::quantize_a_ਪੰਕਤੀ_i8k;
                ਖੰਡ_stride_a    = spacemit_kernels::q8k_blk_ਮਾਪ(a_blk_len);
                gemm_kernel       = spacemit_kernels::ime2::gemm_kernel_i8i2k;
                ਸੈੱਟ_kernel_impl   = ਸੱਚ;
            } ਨਹੀਂਤਾਂ ਜੇ constexpr (std::is_same_v<BLOC_TYPE, ਖੰਡ_q3_K>) {
                quantize_a_ਪੰਕਤੀ_i8 = spacemit_kernels::rvv::quantize_a_ਪੰਕਤੀ_i8k;
                ਖੰਡ_stride_a    = spacemit_kernels::q8k_blk_ਮਾਪ(a_blk_len);
                gemm_kernel       = spacemit_kernels::ime2::gemm_kernel_i8i3k;
                ਸੈੱਟ_kernel_impl   = ਸੱਚ;
            } ਨਹੀਂਤਾਂ ਜੇ constexpr (std::is_same_v<BLOC_TYPE, ਖੰਡ_mxfp4>) {
                gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8mxfp4;
                moe_gemm_kernel_m2 = spacemit_kernels::ime2::moe_m2_gemm_kernel_i8mxfp4;
                ਸੈੱਟ_kernel_impl    = ਸੱਚ;
            } ਨਹੀਂਤਾਂ ਜੇ constexpr (std::is_same_v<BLOC_TYPE, ਖੰਡ_q5_1> || std::is_same_v<BLOC_TYPE, ਖੰਡ_q5_K> ||
                                 std::is_same_v<BLOC_TYPE, ਖੰਡ_q5_0>) {
                gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i5;
                moe_gemm_kernel_m2 = spacemit_kernels::ime2::moe_m2_gemm_kernel_i8i5;
                ਸੈੱਟ_kernel_impl    = ਸੱਚ;
            }
        }
#ਜੇ_ਅੰਤ

#ਜੇ defined(RISCV64_SPACEMIT_IME1)
        ਜੇ (!ਸੈੱਟ_kernel_impl && (global_spine_env_info.use_ime1)) {
            quantize_a_ਪੰਕਤੀ_i8 = spacemit_kernels::ime1::quantize_a_ਪੰਕਤੀ_i8;

            ਜੇ constexpr (std::is_same_v<BLOC_TYPE, ਖੰਡ_q4_0> || std::is_same_v<BLOC_TYPE, ਖੰਡ_q4_1> ||
                          std::is_same_v<BLOC_TYPE, ਖੰਡ_q4_K>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i4;
                ਸੈੱਟ_kernel_impl = ਸੱਚ;
            } ਨਹੀਂਤਾਂ ਜੇ constexpr (std::is_same_v<BLOC_TYPE, ਖੰਡ_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i8;
                ਸੈੱਟ_kernel_impl = ਸੱਚ;
            }
        }
#ਜੇ_ਅੰਤ
        ਜੇ (!ਸੈੱਟ_kernel_impl) {
            GGML_ABORT("no kernel implementation found for the block type");
        }

        ਅਚਲ size_t a_k_blks = spacemit_kernels::div_ਗੇੜਾ_ਉੱਪਰ(ne10, a_blk_len);
        ਅਚਲ size_t b_k_blks = spacemit_kernels::div_ਗੇੜਾ_ਉੱਪਰ(ne10, b_blk_len);

        ਅਚਲ size_t nbw1                = a_k_blks * ਖੰਡ_stride_a;
        ਅਚਲ size_t nbw2                = ne11 * nbw1;
        ਅਚਲ size_t nbw3                = nbw2 * ne12;
        ਅਚਲ size_t gemm_workspace_ਮਾਪ = GGML_PAD(nbw3, alignof(int64_t));

        ਅਚਲ uintptr_t ws_ptr         = reinterpret_cast<uintptr_t>(params->wdata);
        ਸਵੈਚਾਲਿਤ *          quant_a_ਬਫਰ = reinterpret_cast<uint8_t *>(ws_ptr);

        ਜੇ (ne11 == 1) {
            ਲਈ (int64_t ii = ith; ii < ne12 * a_k_blks; ii += nth) {
                int64_t i12       = ii / a_k_blks;
                int64_t ak_blk_id = ii % a_k_blks;
                quantize_a_ਪੰਕਤੀ_i8(a_blk_len, (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) src1->ਅੰਕੜੇ + i12 * nb12) + ak_blk_id * a_blk_len,
                                  a_blk_len, quant_a_ਬਫਰ + i12 * nbw2 + ak_blk_id * ਖੰਡ_stride_a);
            }
        } ਨਹੀਂਤਾਂ {
            ਲਈ (int64_t ii = ith; ii < ne12 * ne11; ii += nth) {
                int64_t i12 = ii / ne11;
                int64_t i11 = ii % ne11;
                quantize_a_ਪੰਕਤੀ_i8(a_blk_len, (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) src1->ਅੰਕੜੇ + i12 * nb12 + i11 * nb11), ne10,
                                  quant_a_ਬਫਰ + i12 * nbw2 + i11 * nbw1);
            }
        }

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ MMID_ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ(ਪੰਕਤੀ_id, i1) ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ[(ਪੰਕਤੀ_id) *ne12 + (i1)]

        int64_t *          ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts       = (int64_t *) (ws_ptr + gemm_workspace_ਮਾਪ);
        int32_t *          valid_ep_ਗਿਣਤੀ          = (int32_t *) (ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts + n_as);
        int32_t *          valid_act_ਗਿਣਤੀ         = (int32_t *) (valid_ep_ਗਿਣਤੀ + 1);
        int64_t *          valid_ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts = (int64_t *) (valid_act_ਗਿਣਤੀ + 1);
        mmid_ਪੰਕਤੀ_mapping * ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ             = (mmid_ਪੰਕਤੀ_mapping *) (valid_ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts + n_as);

        ਜੇ (ith == 0) {
            // initialize matrix_row_counts
            memset(ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts, 0, n_as * ਆਕਾਰ(int64_t));

            // group rows by src0 matrix
            ਲਈ (int32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
                ਲਈ (int32_t id = 0; id < n_ids; ++id) {
                    ਅਚਲ int32_t i02 =
                        *(ਅਚਲ int32_t *) ((ਅਚਲ ਅੱਖਰ *) ids->ਅੰਕੜੇ + iid1 * ids->nb[1] + id * ids->nb[0]);

                    GGML_ASSERT(i02 >= 0 && i02 < n_as);

                    MMID_ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ(i02, ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts[i02]) = { id, iid1 };
                    ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts[i02] += 1;
                }
            }

            int32_t valid_ep_ਗਿਣਤੀ_t  = 0;
            int32_t valid_act_ਗਿਣਤੀ_t = 0;
            ਲਈ (ਪੂਰਨ cur_a = 0; cur_a < n_as; ++cur_a) {
                ਅਚਲ int64_t cne1 = ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts[cur_a];
                ਜੇ (cne1 == 0) {
                    ਜਾਰੀ;
                }
                valid_ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts[valid_ep_ਗਿਣਤੀ_t] = cur_a;
                valid_act_ਗਿਣਤੀ_t += cne1;
                valid_ep_ਗਿਣਤੀ_t += 1;
            }
            valid_ep_ਗਿਣਤੀ[0]  = valid_ep_ਗਿਣਤੀ_t;
            valid_act_ਗਿਣਤੀ[0] = valid_act_ਗਿਣਤੀ_t;
        }

        ਅਚਲ int64_t barrier_idx = static_cast<int64_t>(ith / 2);

        GGML_ASSERT(global_spine_env_info.init_barrier != ਖਾਲੀ_ਸੂਚਕ);
        GGML_ASSERT(barrier_idx < spine_init_barrier_ਗਿਣਤੀ);
        spine_barrier_t * cur_barrier = &global_spine_env_info.init_barrier[barrier_idx];

        ggml_barrier(params->threadpool);

        ਅਚਲ size_t ਪੰਕਤੀ_stride_b      = b_k_blks * get_repacked_ਖੰਡ_type_ਮਾਪ<BLOC_TYPE, INTER_ਮਾਪ, NB_ਖਾਨੇ_ਗਿਣਤੀ>();
        ਅਚਲ size_t expert_b_stride   = ne01 * ਪੰਕਤੀ_stride_b;
        ਅਚਲ size_t ਪ੍ਰਤੀ_nb_ਖਾਨੇ_ਗਿਣਤੀ_wsize = NB_ਖਾਨੇ_ਗਿਣਤੀ * ਪੰਕਤੀ_stride_b;

        std::array<ਅਚਲ uint8_t *, 2> src_workspaces;
        std::array<ਦਸ਼ਮਲਵ *, 2>         dst_workspaces;

        ਸਵੈਚਾਲਿਤ *     tcm_ਬਫਰ      = ggml::cpu::riscv64_spacemit::tls_context.tcm_ਬਫਰ;
        ਅਚਲ ਸਵੈਚਾਲਿਤ tcm_ਬਫਰ_ਮਾਪ = ggml::cpu::riscv64_spacemit::tls_context.tcm_ਬਫਰ_ਮਾਪ;

        ਅਚਲ ਸਵੈਚਾਲਿਤ valid_ep_ਗਿਣਤੀ_t  = valid_ep_ਗਿਣਤੀ[0];
        ਅਚਲ ਸਵੈਚਾਲਿਤ valid_act_ਗਿਣਤੀ_t = valid_act_ਗਿਣਤੀ[0];

        ਪੂਰਨ nth_es = 1;
        ਪੂਰਨ nth_n  = nth;

        ਪੂਰਨ ith_es = ith % nth_es;
        ਪੂਰਨ ith_n  = (ith / nth_es) % nth_n;

        ਜੇ (valid_ep_ਗਿਣਤੀ_t % nth == 0 && tcm_ਬਫਰ != ਖਾਲੀ_ਸੂਚਕ && valid_ep_ਗਿਣਤੀ_t == n_as &&
            valid_act_ਗਿਣਤੀ_t == n_as && ਪ੍ਰਤੀ_nb_ਖਾਨੇ_ਗਿਣਤੀ_wsize <= tcm_ਬਫਰ_ਮਾਪ) {
            ਲਈ (int64_t valid_id = ith; valid_id < valid_ep_ਗਿਣਤੀ_t; valid_id += nth) {
                ਅਚਲ int64_t cur_a = valid_ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts[valid_id];

                ਸਵੈਚਾਲਿਤ * src0_cur = (uint8_t *) src0->ਅੰਕੜੇ + cur_a * expert_b_stride;

                mmid_ਪੰਕਤੀ_mapping ਪੰਕਤੀ_mapping = MMID_ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ(cur_a, 0);
                ਅਚਲ ਪੂਰਨ        id          = ਪੰਕਤੀ_mapping.i1;
                ਅਚਲ int64_t    i11         = id % ne11;
                ਅਚਲ int64_t    i12         = ਪੰਕਤੀ_mapping.i2;
                ਅਚਲ int64_t    i1          = id;
                ਅਚਲ int64_t    i2          = i12;

                ਸਵੈਚਾਲਿਤ *  src1_ਖਾਨਾ = quant_a_ਬਫਰ + (i11 * nbw1 + i12 * nbw2);
                ਦਸ਼ਮਲਵ * c_blk    = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + (i1 * nb1 + i2 * nb2));

                uint8_t * a_ਪੰਕਤੀ = src1_ਖਾਨਾ;
                uint8_t * b_ਖਾਨਾ = reinterpret_cast<uint8_t *>(tcm_ਬਫਰ);
                ਜੇ ((nbw1 + ਪ੍ਰਤੀ_nb_ਖਾਨੇ_ਗਿਣਤੀ_wsize) <= tcm_ਬਫਰ_ਮਾਪ) {
                    a_ਪੰਕਤੀ = (uint8_t *) tcm_ਬਫਰ;
                    b_ਖਾਨਾ = reinterpret_cast<uint8_t *>(tcm_ਬਫਰ) + nbw1;
                }
                uint8_t * b_ਖਾਨਾ_zp = ਖੰਡ_type_has_zp<BLOC_TYPE>() ? b_ਖਾਨਾ : ਖਾਲੀ_ਸੂਚਕ;

                ਜੇ (ith % 2 == 0) {
                    spacemit_kernels::rvv::memcpy1d(b_ਖਾਨਾ, reinterpret_cast<uint8_t *>(src0_cur), ਪ੍ਰਤੀ_nb_ਖਾਨੇ_ਗਿਣਤੀ_wsize);

                    ਜੇ (a_ਪੰਕਤੀ != src1_ਖਾਨਾ) {
                        spacemit_kernels::rvv::memcpy1d(a_ਪੰਕਤੀ, src1_ਖਾਨਾ, nbw1);
                    }
                }

                spine_barrier_wait(cur_barrier);

                ਜੇ (ith % 2 != 0) {
                    ਜੇ (a_ਪੰਕਤੀ != src1_ਖਾਨਾ) {
                        spacemit_kernels::rvv::memcpy1d(a_ਪੰਕਤੀ, src1_ਖਾਨਾ, nbw1);
                    }

                    spacemit_kernels::rvv::memcpy1d(b_ਖਾਨਾ, reinterpret_cast<uint8_t *>(src0_cur), ਪ੍ਰਤੀ_nb_ਖਾਨੇ_ਗਿਣਤੀ_wsize);
                }

                int64_t nb_real = std::min(ne01, NB_ਖਾਨੇ_ਗਿਣਤੀ);
                ਲਈ (int64_t ni = 0; ni < ne01; ni += NB_ਖਾਨੇ_ਗਿਣਤੀ) {
                    ਜੇ (ith % 2 != 0) {
                        spine_barrier_wait(cur_barrier);
                    }

                    gemm_kernel(b_blk_len, a_ਪੰਕਤੀ, b_ਖਾਨਾ, b_ਖਾਨਾ_zp, c_blk + ni, 1, nb_real, b_k_blks, ne01);

                    ਜੇ (ith % 2 == 0) {
                        spine_barrier_wait(cur_barrier);
                    }

                    ਅਚਲ int64_t ਅਗਲਾ_ni = ni + NB_ਖਾਨੇ_ਗਿਣਤੀ;
                    ਜੇ (ਅਗਲਾ_ni < ne01) {
                        nb_real = std::min(ne01 - ਅਗਲਾ_ni, NB_ਖਾਨੇ_ਗਿਣਤੀ);
                        spacemit_kernels::rvv::memcpy1d(
                            b_ਖਾਨਾ, reinterpret_cast<uint8_t *>(src0_cur) + ਅਗਲਾ_ni * ਪੰਕਤੀ_stride_b, ਪ੍ਰਤੀ_nb_ਖਾਨੇ_ਗਿਣਤੀ_wsize);
                    }
                }
            }
        } ਨਹੀਂਤਾਂ {
            ਲਈ (int64_t valid_id = ith_es; valid_id < valid_ep_ਗਿਣਤੀ_t; valid_id += nth_es) {
                ਅਚਲ int64_t cur_a = valid_ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts[valid_id];
                ਅਚਲ int64_t cne1  = ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts[cur_a];

                int64_t src1_cur_ਸ਼ੁਰੂ = 0;
                int64_t src1_cur_ਸਿਰਾ   = cne1;

                int64_t src0_cur_ਸ਼ੁਰੂ = (ith_n * ne01) / nth_n;
                int64_t src0_cur_ਸਿਰਾ   = MIN(((ith_n + 1) * ne01) / nth_n, ne01);

                ਜੇ (src1_cur_ਸ਼ੁਰੂ >= src1_cur_ਸਿਰਾ || src0_cur_ਸ਼ੁਰੂ >= src0_cur_ਸਿਰਾ) {
                    ਜਾਰੀ;
                }

                src0_cur_ਸ਼ੁਰੂ =
                    (src0_cur_ਸ਼ੁਰੂ % NB_ਖਾਨੇ_ਗਿਣਤੀ) ? src0_cur_ਸ਼ੁਰੂ + NB_ਖਾਨੇ_ਗਿਣਤੀ - (src0_cur_ਸ਼ੁਰੂ % NB_ਖਾਨੇ_ਗਿਣਤੀ) : src0_cur_ਸ਼ੁਰੂ;
                src0_cur_ਸਿਰਾ =
                    (src0_cur_ਸਿਰਾ % NB_ਖਾਨੇ_ਗਿਣਤੀ) ? src0_cur_ਸਿਰਾ + NB_ਖਾਨੇ_ਗਿਣਤੀ - (src0_cur_ਸਿਰਾ % NB_ਖਾਨੇ_ਗਿਣਤੀ) : src0_cur_ਸਿਰਾ;

                ਸਵੈਚਾਲਿਤ *    src0_cur = (uint8_t *) src0->ਅੰਕੜੇ + cur_a * expert_b_stride + src0_cur_ਸ਼ੁਰੂ * ਪੰਕਤੀ_stride_b;
                uint8_t * b_ਖਾਨਾ_zp = ਖੰਡ_type_has_zp<BLOC_TYPE>() ? src0_cur : ਖਾਲੀ_ਸੂਚਕ;

                size_t extra_tcm_ਬਫਰ_ਮਾਪ = tcm_ਬਫਰ_ਮਾਪ;
                ਖਾਲੀ * extra_tcm_ਬਫਰ      = tcm_ਬਫਰ;
                ਜੇ (tcm_ਬਫਰ != ਖਾਲੀ_ਸੂਚਕ && (src1_cur_ਸਿਰਾ - src1_cur_ਸ਼ੁਰੂ) >= 4 &&
                    (src0_cur_ਸਿਰਾ - src0_cur_ਸ਼ੁਰੂ) * ਪੰਕਤੀ_stride_b <= tcm_ਬਫਰ_ਮਾਪ) {
                    spacemit_kernels::rvv::memcpy1d(tcm_ਬਫਰ, src0_cur,
                                                    (src0_cur_ਸਿਰਾ - src0_cur_ਸ਼ੁਰੂ) * ਪੰਕਤੀ_stride_b);
                    src0_cur = reinterpret_cast<uint8_t *>(tcm_ਬਫਰ);
                    b_ਖਾਨਾ_zp = ਖੰਡ_type_has_zp<BLOC_TYPE>() ? src0_cur : ਖਾਲੀ_ਸੂਚਕ;
                    extra_tcm_ਬਫਰ_ਮਾਪ -= (src0_cur_ਸਿਰਾ - src0_cur_ਸ਼ੁਰੂ) * ਪੰਕਤੀ_stride_b;
                    extra_tcm_ਬਫਰ = reinterpret_cast<ਖਾਲੀ *>(reinterpret_cast<uintptr_t>(tcm_ਬਫਰ) +
                                                                (src0_cur_ਸਿਰਾ - src0_cur_ਸ਼ੁਰੂ) * ਪੰਕਤੀ_stride_b);
                }

                ਪੂਰਨ ir1 = src1_cur_ਸ਼ੁਰੂ;

                ਜੇ (extra_tcm_ਬਫਰ_ਮਾਪ >= nbw1 && extra_tcm_ਬਫਰ != ਖਾਲੀ_ਸੂਚਕ) {
                    int64_t quant_a_tile_ਮਾਪ = extra_tcm_ਬਫਰ_ਮਾਪ / nbw1;
                    ਕਰੋ {
                        quant_a_tile_ਮਾਪ = MIN(quant_a_tile_ਮਾਪ, src1_cur_ਸਿਰਾ - ir1);

                        uint8_t * quant_a_tile_ਬਫਰ = reinterpret_cast<uint8_t *>(extra_tcm_ਬਫਰ);

                        ਪੂਰਨ iir1 = ir1;
                        ਲਈ (; iir1 < (ir1 + quant_a_tile_ਮਾਪ); ++iir1) {
                            mmid_ਪੰਕਤੀ_mapping ਪੰਕਤੀ_mapping = MMID_ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ(cur_a, iir1);

                            ਅਚਲ ਪੂਰਨ id = ਪੰਕਤੀ_mapping.i1;  // selected expert index

                            ਅਚਲ int64_t i11 = id % ne11;
                            ਅਚਲ int64_t i12 = ਪੰਕਤੀ_mapping.i2;  // row index in src1

                            ਸਵੈਚਾਲਿਤ * src1_ਖਾਨਾ = quant_a_ਬਫਰ + (i11 * nbw1 + i12 * nbw2);
                            spacemit_kernels::rvv::memcpy1d(quant_a_tile_ਬਫਰ, src1_ਖਾਨਾ, nbw1);
                            quant_a_tile_ਬਫਰ = quant_a_tile_ਬਫਰ + nbw1;
                        }

                        quant_a_tile_ਬਫਰ = reinterpret_cast<uint8_t *>(extra_tcm_ਬਫਰ);
                        iir1                = ir1;

                        ਜੇ (moe_gemm_kernel_m2 != ਖਾਲੀ_ਸੂਚਕ) {
                            ਲਈ (; iir1 < (ir1 + quant_a_tile_ਮਾਪ - 1); iir1 += 2, quant_a_tile_ਬਫਰ += 2 * nbw1) {
                                mmid_ਪੰਕਤੀ_mapping ਪੰਕਤੀ_mapping_0 = MMID_ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ(cur_a, iir1);
                                mmid_ਪੰਕਤੀ_mapping ਪੰਕਤੀ_mapping_1 = MMID_ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ(cur_a, iir1 + 1);

                                src_workspaces[0] = quant_a_tile_ਬਫਰ;
                                src_workspaces[1] = quant_a_tile_ਬਫਰ + nbw1;

                                dst_workspaces[0] =
                                    (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + (ਪੰਕਤੀ_mapping_0.i1 * nb1 + ਪੰਕਤੀ_mapping_0.i2 * nb2)) +
                                    src0_cur_ਸ਼ੁਰੂ;
                                dst_workspaces[1] = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ +
                                                               ((ਪੰਕਤੀ_mapping_1.i1) * nb1 + (ਪੰਕਤੀ_mapping_1.i2) * nb2)) +
                                                    src0_cur_ਸ਼ੁਰੂ;
                                moe_gemm_kernel_m2(b_blk_len, src_workspaces.ਅੰਕੜੇ(), src0_cur, b_ਖਾਨਾ_zp,
                                                   dst_workspaces.ਅੰਕੜੇ(), 1, src0_cur_ਸਿਰਾ - src0_cur_ਸ਼ੁਰੂ, b_k_blks,
                                                   ne01);
                            }
                        }

                        ਲਈ (; iir1 < (ir1 + quant_a_tile_ਮਾਪ); iir1++, quant_a_tile_ਬਫਰ += nbw1) {
                            mmid_ਪੰਕਤੀ_mapping ਪੰਕਤੀ_mapping_0 = MMID_ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ(cur_a, iir1);

                            gemm_kernel(
                                b_blk_len, quant_a_tile_ਬਫਰ, src0_cur, b_ਖਾਨਾ_zp,
                                (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + (ਪੰਕਤੀ_mapping_0.i1 * nb1 + ਪੰਕਤੀ_mapping_0.i2 * nb2)) +
                                    src0_cur_ਸ਼ੁਰੂ,
                                1, src0_cur_ਸਿਰਾ - src0_cur_ਸ਼ੁਰੂ, b_k_blks, ne01);
                        }

                        ir1 += quant_a_tile_ਮਾਪ;
                    } ਜਦੋਂ (ir1 < src1_cur_ਸਿਰਾ);
                } ਨਹੀਂਤਾਂ {
                    ਜੇ (moe_gemm_kernel_m2 != ਖਾਲੀ_ਸੂਚਕ) {
                        ਲਈ (; ir1 < src1_cur_ਸਿਰਾ - 1; ir1 += 2) {
                            ਲਈ (ਪੂਰਨ iir1 = 0; iir1 < 2; ++iir1) {
                                mmid_ਪੰਕਤੀ_mapping ਪੰਕਤੀ_mapping = MMID_ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ(cur_a, ir1 + iir1);

                                ਅਚਲ ਪੂਰਨ id = ਪੰਕਤੀ_mapping.i1;  // selected expert index

                                ਅਚਲ int64_t i11 = id % ne11;
                                ਅਚਲ int64_t i12 = ਪੰਕਤੀ_mapping.i2;  // row index in src1

                                ਅਚਲ int64_t i1 = id;               // selected expert index
                                ਅਚਲ int64_t i2 = i12;              //ਪੰਕਤੀ

                                src_workspaces[iir1] = quant_a_ਬਫਰ + (i11 * nbw1 + i12 * nbw2);

                                dst_workspaces[iir1] =
                                    (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + (i1 * nb1 + i2 * nb2)) + src0_cur_ਸ਼ੁਰੂ;
                            }

                            moe_gemm_kernel_m2(b_blk_len, src_workspaces.ਅੰਕੜੇ(), src0_cur, b_ਖਾਨਾ_zp,
                                               dst_workspaces.ਅੰਕੜੇ(), 1, src0_cur_ਸਿਰਾ - src0_cur_ਸ਼ੁਰੂ, b_k_blks, ne01);
                        }
                    }

                    ਲਈ (; ir1 < src1_cur_ਸਿਰਾ; ir1++) {
                        mmid_ਪੰਕਤੀ_mapping ਪੰਕਤੀ_mapping = MMID_ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ(cur_a, ir1);

                        ਅਚਲ ਪੂਰਨ id = ਪੰਕਤੀ_mapping.i1;  // selected expert index

                        ਅਚਲ int64_t i11 = id % ne11;
                        ਅਚਲ int64_t i12 = ਪੰਕਤੀ_mapping.i2;  // row index in src1

                        ਅਚਲ int64_t i1 = id;               // selected expert index
                        ਅਚਲ int64_t i2 = i12;              //ਪੰਕਤੀ

                        ਸਵੈਚਾਲਿਤ * src1_ਖਾਨਾ = quant_a_ਬਫਰ + (i11 * nbw1 + i12 * nbw2);

                        gemm_kernel(b_blk_len, src1_ਖਾਨਾ, src0_cur, b_ਖਾਨਾ_zp,
                                    (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + (i1 * nb1 + i2 * nb2)) + src0_cur_ਸ਼ੁਰੂ, 1,
                                    src0_cur_ਸਿਰਾ - src0_cur_ਸ਼ੁਰੂ, b_k_blks, ne01);
                    }
                }
            }
        }
#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ MMID_ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ
    }

    ਪੂਰਨ repack(ggml_tensor * t, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਅੰਕੜੇ_ਮਾਪ) override {
        GGML_LOG_DEBUG("%s: repack tensor %s with %s_%dx%d\n", __func__, t->name, ggml_type_name(t->type),
                       (ਪੂਰਨ) NB_ਖਾਨੇ_ਗਿਣਤੀ, (ਪੂਰਨ) INTER_ਮਾਪ);
        ਵਾਪਸ ggml::cpu::riscv64_spacemit::repack<BLOC_TYPE, INTER_ਮਾਪ, NB_ਖਾਨੇ_ਗਿਣਤੀ>(t, ਅੰਕੜੇ, ਅੰਕੜੇ_ਮਾਪ);
    }
};

ਵਰਗ tensor_traits_common : ਜਨਤਕ tensor_traits_ਅਧਾਰ {
    ਤਾਰਕਿਕ work_ਮਾਪ(ਪੂਰਨ n_ਧਾਗੇ, ਅਚਲ ggml_tensor * op, size_t & size) override {
        ਚੋਣ (op->op) {
            ਹਾਲਤ GGML_OP_FLASH_ATTN_EXT:
                {
                    ਅਚਲ ਪੂਰਨ     n_tasks = n_ਧਾਗੇ;
                    ਅਚਲ int64_t neq2    = op->src[0]->ne[2];  // number of query heads
                    ਅਚਲ int64_t DK      = op->src[1]->ne[0];
                    ਅਚਲ int64_t DV      = op->src[2]->ne[0];  // DV

                    // Tiled flash attention scratch (tile sizes defined in common.h)
                    // Per-thread: Q_q + KQ + mask + VKQ32 + V32 + K_f32 + padding
                    size_t prefill = ਆਕਾਰ(ਦਸ਼ਮਲਵ) *
                                     (GGML_FA_TILE_Q * DK + 2 * GGML_FA_TILE_Q * GGML_FA_TILE_KV + GGML_FA_TILE_Q * DV +
                                      GGML_FA_TILE_KV * DV + GGML_FA_TILE_KV * DK) *
                                     n_tasks;

                    // Decode path: n_kv_chunks = n_tasks (one chunk per thread)
                    // Per-thread: VKQ accmulator (DV), partial M, partial S + intra-thread scratch for V, Q and VKQ
                    size_t n_chunks = n_tasks;
                    size_t decode   = ਆਕਾਰ(ਦਸ਼ਮਲਵ) * (neq2 * n_chunks * (2 + DV) + n_tasks * (DK + 2 * DV));

                    size = MAX(prefill, decode);
                }
                ਵਾਪਸ ਸੱਚ;
            ਮੂਲ:
                ਰੋਕੋ;
        }
        ਵਾਪਸ ਝੂਠ;
    }

    ਤਾਰਕਿਕ compute_forward(ggml_compute_params * params, ggml_tensor * op) override {
        ਚੋਣ (op->op) {
            ਹਾਲਤ GGML_OP_NORM:
                ਚੋਣ (op->src[0]->type) {
                    ਹਾਲਤ GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_norm_f32(params, op);
                        ਵਾਪਸ ਸੱਚ;
                    ਮੂਲ:
                        GGML_ABORT("fatal error");
                }
            ਹਾਲਤ GGML_OP_RMS_NORM:
                ਚੋਣ (op->src[0]->type) {
                    ਹਾਲਤ GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_rms_norm_f32(params, op);
                        ਵਾਪਸ ਸੱਚ;
                    ਮੂਲ:
                        GGML_ABORT("fatal error");
                }
            ਹਾਲਤ GGML_OP_ਜੋੜੋ:
                ਚੋਣ (op->src[0]->type) {
                    ਹਾਲਤ GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_ਦੁਆਧਾਰੀ<GGML_OP_ਜੋੜੋ, ਦਸ਼ਮਲਵ>(params, op);
                        ਵਾਪਸ ਸੱਚ;
                    ਹਾਲਤ GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_ਦੁਆਧਾਰੀ<GGML_OP_ਜੋੜੋ, _Float16>(params, op);
                        ਵਾਪਸ ਸੱਚ;
                    ਮੂਲ:
                        ggml_compute_forward_ਜੋੜੋ(params, op);
                        ਵਾਪਸ ਸੱਚ;
                }
            ਹਾਲਤ GGML_OP_SUB:
                ਚੋਣ (op->src[0]->type) {
                    ਹਾਲਤ GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_ਦੁਆਧਾਰੀ<GGML_OP_SUB, ਦਸ਼ਮਲਵ>(params, op);
                        ਵਾਪਸ ਸੱਚ;
                    ਹਾਲਤ GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_ਦੁਆਧਾਰੀ<GGML_OP_SUB, _Float16>(params, op);
                        ਵਾਪਸ ਸੱਚ;
                    ਮੂਲ:
                        ggml_compute_forward_sub(params, op);
                        ਵਾਪਸ ਸੱਚ;
                }
            ਹਾਲਤ GGML_OP_MUL:
                ਚੋਣ (op->src[0]->type) {
                    ਹਾਲਤ GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_ਦੁਆਧਾਰੀ<GGML_OP_MUL, ਦਸ਼ਮਲਵ>(params, op);
                        ਵਾਪਸ ਸੱਚ;
                    ਹਾਲਤ GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_ਦੁਆਧਾਰੀ<GGML_OP_MUL, _Float16>(params, op);
                        ਵਾਪਸ ਸੱਚ;
                    ਮੂਲ:
                        ggml_compute_forward_mul(params, op);
                        ਵਾਪਸ ਸੱਚ;
                }
            ਹਾਲਤ GGML_OP_DIV:
                ਚੋਣ (op->src[0]->type) {
                    ਹਾਲਤ GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_ਦੁਆਧਾਰੀ<GGML_OP_DIV, ਦਸ਼ਮਲਵ>(params, op);
                        ਵਾਪਸ ਸੱਚ;
                    ਹਾਲਤ GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_ਦੁਆਧਾਰੀ<GGML_OP_DIV, _Float16>(params, op);
                        ਵਾਪਸ ਸੱਚ;
                    ਮੂਲ:
                        ggml_compute_forward_div(params, op);
                        ਵਾਪਸ ਸੱਚ;
                }
            ਹਾਲਤ GGML_OP_FLASH_ATTN_EXT:
                forward_flash_attn_ext_f16(params, op);
                ਵਾਪਸ ਸੱਚ;
            ਹਾਲਤ GGML_OP_CONT:
                {
                    ਅਚਲ ggml_tensor * src0 = op->src[0];
                    ਜੇ (op->type == src0->type && op->nb[0] != src0->nb[0] && op->nb[0] == src0->nb[1] &&
                        op->ne[3] * op->ne[2] * op->nb[2] == src0->ne[3] * src0->ne[2] * src0->nb[2]) {
                        spacemit_kernels::rvv::forward_cont_with_ਕ੍ਰਮ_ਬਦਲੋ(params, op);
                    } ਨਹੀਂਤਾਂ {
                        ggml_compute_forward_cont(params, op);
                    }
                    ਵਾਪਸ ਸੱਚ;
                }
            ਹਾਲਤ GGML_OP_CPY:
                {
                    ਅਚਲ ggml_tensor * src0 = op->src[0];
                    ਜੇ (op->type == src0->type && op->nb[0] == src0->nb[1] && src0->nb[0] != src0->nb[1] &&
                        ggml_nelements(src0) == ggml_nelements(op)) {
                        spacemit_kernels::rvv::forward_cpy_with_ਕ੍ਰਮ_ਬਦਲੋ(params, op);
                    } ਨਹੀਂਤਾਂ {
                        ggml_compute_forward_cpy(params, op);
                    }
                    ਵਾਪਸ ਸੱਚ;
                }
            ਹਾਲਤ GGML_OP_REPEAT:
                {
                    ਅਚਲ ਤਾਰਕਿਕ ਪੰਕਤੀਆਂ_equal         = ggml_nrows(op->src[0]) == ggml_nrows(op);
                    ਅਚਲ ਤਾਰਕਿਕ ਪ੍ਰਸਾਰਣ_or_equal = op->src[0]->ne[0] == 1 || op->src[0]->ne[0] == op->ne[0];

                    ਜੇ (ਪੰਕਤੀਆਂ_equal && ਪ੍ਰਸਾਰਣ_or_equal) {
                        ਚੋਣ (op->src[0]->type) {
                            ਹਾਲਤ GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_repeat_nrows<int32_t>(params, op);
                                ਵਾਪਸ ਸੱਚ;
                            ਹਾਲਤ GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_repeat_nrows<int16_t>(params, op);
                                ਵਾਪਸ ਸੱਚ;
                            ਮੂਲ:
                                ਰੋਕੋ;
                        }
                    }

                    ਜੇ (op->src[0]->ne[1] == 1 && op->src[0]->ne[0] == op->ne[0]) {
                        ਚੋਣ (op->src[0]->type) {
                            ਹਾਲਤ GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_repeat_dim1<int32_t>(params, op);
                                ਵਾਪਸ ਸੱਚ;
                            ਹਾਲਤ GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_repeat_dim1<int16_t>(params, op);
                                ਵਾਪਸ ਸੱਚ;
                            ਮੂਲ:
                                ਰੋਕੋ;
                        }
                    }

                    ggml_compute_forward_repeat(params, op);
                }
                ਵਾਪਸ ਸੱਚ;
            ਹਾਲਤ GGML_OP_ਜੋੜ_ਪੰਕਤੀਆਂ:
                {
                    ਜੇ (op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32) {
                        spacemit_kernels::rvv::forward_ਜੋੜ_ਪੰਕਤੀਆਂ<ਦਸ਼ਮਲਵ>(params, op);
                    } ਨਹੀਂਤਾਂ {
                        ggml_compute_forward_ਜੋੜ_ਪੰਕਤੀਆਂ(params, op);
                    }
                }
                ਵਾਪਸ ਸੱਚ;
            ਹਾਲਤ GGML_OP_GET_ਪੰਕਤੀਆਂ:
                {
                    ਜੇ (op->src[0]->type == op->type) {
                        ਚੋਣ (op->src[0]->type) {
                            ਹਾਲਤ GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_get_ਪੰਕਤੀਆਂ<int32_t>(params, op);
                                ਵਾਪਸ ਸੱਚ;
                            ਹਾਲਤ GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_get_ਪੰਕਤੀਆਂ<int16_t>(params, op);
                                ਵਾਪਸ ਸੱਚ;
                            ਮੂਲ:
                                ਰੋਕੋ;
                        }
                    }

                    ggml_compute_forward_get_ਪੰਕਤੀਆਂ(params, op);
                }
                ਵਾਪਸ ਸੱਚ;
            ਹਾਲਤ GGML_OP_CONCAT:
                {
                    ਅਚਲ int32_t dim = ggml_get_op_params_i32(op, 0);
                    ਜੇ (dim == 0 && op->type == op->src[0]->type) {
                        ਚੋਣ (op->src[0]->type) {
                            ਹਾਲਤ GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_concat<int32_t>(params, op);
                                ਵਾਪਸ ਸੱਚ;
                            ਹਾਲਤ GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_concat<int16_t>(params, op);
                                ਵਾਪਸ ਸੱਚ;
                            ਮੂਲ:
                                ਰੋਕੋ;
                        }
                    }

                    ggml_compute_forward_concat(params, op);
                }
                ਵਾਪਸ ਸੱਚ;
            // TODO For GGML_OP_GATED_DELTA_NET
            // case GGML_OP_GATED_DELTA_NET:
            //     return true;
            ਮੂਲ:
                ਰੋਕੋ;
        }
        ਵਾਪਸ ਝੂਠ;
    }

    ਖਾਲੀ forward_flash_attn_ext_f16(ਅਚਲ ggml_compute_params * params, ggml_tensor * dst) {
        ਅਚਲ ggml_tensor * q = dst->src[0];
        ਅਚਲ ggml_tensor * k = dst->src[1];
        ਅਚਲ ggml_tensor * v = dst->src[2];

        GGML_TENSOR_LOCALS(int64_t, neq, q, ne)
        GGML_TENSOR_LOCALS(size_t, nbq, q, nb)
        GGML_TENSOR_LOCALS(int64_t, nek, k, ne)
        GGML_TENSOR_LOCALS(size_t, nbk, k, nb)
        GGML_TENSOR_LOCALS(int64_t, nev, v, ne)
        GGML_TENSOR_LOCALS(size_t, nbv, v, nb)
        GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
        GGML_TENSOR_LOCALS(size_t, nb, dst, nb)

        ਅਚਲ int64_t DK = nek0;
        ਅਚਲ int64_t DV = nev0;

        ਅਚਲ ਤਾਰਕਿਕ supported_prec  = (dst->op_params[3] == GGML_PREC_F32 || dst->op_params[3] == GGML_PREC_DEFAULT);
        ਅਚਲ ਤਾਰਕਿਕ supported_types = (q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16);
        ਅਚਲ ਤਾਰਕਿਕ supported_shape = (DK > 0 && DK <= 128 && DV > 0 && DV <= 128);
        ਅਚਲ ਤਾਰਕਿਕ supported_vlen  = (__riscv_vlenb() == 128);

        ਜੇ (!(supported_prec && supported_types && supported_shape && supported_vlen)) {
            ggml_compute_forward_flash_attn_ext(params, dst);
            ਵਾਪਸ;
        }

        // total rows in q
        ਅਚਲ int64_t nr = neq1 * neq2 * neq3;

        // rows per thread
        ਅਚਲ ਪੂਰਨ ith = params->ith;
        ਅਚਲ ਪੂਰਨ nth = params->nth;

        ਸਥਿਰ constexpr int64_t Q_TILE_SZ = ggml_fa_tile_config::Q;
        ਅਚਲ ਤਾਰਕਿਕ               use_tiled = !params->use_ref && (neq1 >= Q_TILE_SZ);

        // 4x chunks per thread
        // int     nth_scaled = nth * 4;
        // int64_t chunk_size = (nr + nth_scaled - 1) / nth_scaled;
        // int64_t nchunk     = (nr + chunk_size - 1) / chunk_size;

        // if (nth == 1 || nchunk < nth) {
        //     nchunk = nth;
        // }

        int64_t nchunk = nth;

        ਜੇ (ith == 0) {
            // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
            ggml_threadpool_chunk_ਸੈੱਟ(params->threadpool, nth);
        }

        ggml_barrier(params->threadpool);

        // The number of elements in each chunk
        ਅਚਲ int64_t dr = (nr + nchunk - 1) / nchunk;

        // The first chunk comes from our thread_id, the rest will get auto-assigned.
        ਪੂਰਨ current_chunk = ith;

        ਜਦੋਂ (current_chunk < nchunk) {
            ਅਚਲ int64_t ir0 = dr * current_chunk;
            ਅਚਲ int64_t ir1 = MIN(ir0 + dr, nr);

            ਜੇ (use_tiled) {
                spacemit_kernels::rvv::forward_flash_attn_ext_f16_tiled_vlen1024_vf16(
                    params, dst, ir0, ir1, ggml::cpu::riscv64_spacemit::tls_context.tcm_ਬਫਰ,
                    ggml::cpu::riscv64_spacemit::tls_context.tcm_ਬਫਰ_ਮਾਪ);
            } ਨਹੀਂਤਾਂ {
                spacemit_kernels::rvv::forward_flash_attn_ext_f16_one_chunk_vlen1024_vf16(
                    params, dst, ir0, ir1, ggml::cpu::riscv64_spacemit::tls_context.tcm_ਬਫਰ,
                    ggml::cpu::riscv64_spacemit::tls_context.tcm_ਬਫਰ_ਮਾਪ);
            }

            current_chunk = ggml_threadpool_chunk_ਜੋੜੋ(params->threadpool, 1);
        }
    }

    ਪੂਰਨ repack(ggml_tensor * t, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਅੰਕੜੇ_ਮਾਪ) override {
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(t->ਅੰਕੜੇ, ਅੰਕੜੇ, ਅੰਕੜੇ_ਮਾਪ);
        ਵਾਪਸ 0;
    }
};

// Impl By IME1
ਸਥਿਰ ਅਚਲ tensor_traits<ਖੰਡ_q4_0, 32, 16>  q4_0_16x32_q8_0;
ਸਥਿਰ ਅਚਲ tensor_traits<ਖੰਡ_q4_1, 32, 16>  q4_1_16x32_q8_0;
ਸਥਿਰ ਅਚਲ tensor_traits<ਖੰਡ_q4_K, 32, 16>  q4_k_16x32_q8_0;
ਸਥਿਰ ਅਚਲ tensor_traits<ਖੰਡ_q8_0, 32, 16>  q8_0_16x32_q8_0;
// Impl By IME2
ਸਥਿਰ ਅਚਲ tensor_traits<ਖੰਡ_q2_K, 256, 32> q2_k_32x256_q8_0;
ਸਥਿਰ ਅਚਲ tensor_traits<ਖੰਡ_q3_K, 256, 32> q3_k_32x256_q8_0;
ਸਥਿਰ ਅਚਲ tensor_traits<ਖੰਡ_q4_0, 32, 32>  q4_0_32x32_q8_0;
ਸਥਿਰ ਅਚਲ tensor_traits<ਖੰਡ_q4_1, 32, 32>  q4_1_32x32_q8_0;
ਸਥਿਰ ਅਚਲ tensor_traits<ਖੰਡ_q4_0, 256, 32> q4_0_32x256_q8_0;
ਸਥਿਰ ਅਚਲ tensor_traits<ਖੰਡ_q4_1, 256, 32> q4_1_32x256_q8_0;
ਸਥਿਰ ਅਚਲ tensor_traits<ਖੰਡ_q4_K, 32, 32>  q4_k_32x32_q8_0;
ਸਥਿਰ ਅਚਲ tensor_traits<ਖੰਡ_q6_K, 32, 32>  q6_k_32x32_q8_0;
ਸਥਿਰ ਅਚਲ tensor_traits<ਖੰਡ_q8_0, 32, 32>  q8_0_32x32_q8_0;
ਸਥਿਰ ਅਚਲ tensor_traits<ਖੰਡ_mxfp4, 32, 32> mxfp4_32x32_q8_0;
ਸਥਿਰ ਅਚਲ tensor_traits<ਖੰਡ_q5_K, 32, 32>  q5_k_32x32_q8_0;
ਸਥਿਰ ਅਚਲ tensor_traits<ਖੰਡ_q5_1, 32, 32>  q5_1_32x32_q8_0;
ਸਥਿਰ ਅਚਲ tensor_traits<ਖੰਡ_q5_0, 32, 32>  q5_0_32x32_q8_0;
// Impl By RVV
ਸਥਿਰ ਅਚਲ tensor_traits_common               rvv_impl;

}  // namespace ggml::cpu::riscv64_spacemit

ਸਥਿਰ ਅਚਲ ggml::cpu::tensor_traits * ggml_riscv64_spacemit_get_optimal_repack_type(ਅਚਲ ggml_tensor * cur) {
    ਚੋਣ (cur->type) {
        ਹਾਲਤ GGML_TYPE_Q2_K:
            {
#ਜੇ defined(RISCV64_SPACEMIT_IME2)
                ਜੇ (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    ਵਾਪਸ &ggml::cpu::riscv64_spacemit::q2_k_32x256_q8_0;
                }
#ਜੇ_ਅੰਤ
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q3_K:
            {
#ਜੇ defined(RISCV64_SPACEMIT_IME2)
                ਜੇ (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    ਵਾਪਸ &ggml::cpu::riscv64_spacemit::q3_k_32x256_q8_0;
                }
#ਜੇ_ਅੰਤ
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q4_0:
            {
#ਜੇ defined(RISCV64_SPACEMIT_IME2)
                ਜੇ (cur->ne[1] % 32 == 0 && cur->ne[0] % 256 == 0 &&
                    (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    ਵਾਪਸ &ggml::cpu::riscv64_spacemit::q4_0_32x256_q8_0;
                }

                ਜੇ (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    ਵਾਪਸ &ggml::cpu::riscv64_spacemit::q4_0_32x32_q8_0;
                }
#ਜੇ_ਅੰਤ

#ਜੇ defined(RISCV64_SPACEMIT_IME1)
                ਜੇ (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    ਵਾਪਸ &ggml::cpu::riscv64_spacemit::q4_0_16x32_q8_0;
                }
#ਜੇ_ਅੰਤ
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q4_1:
            {
#ਜੇ defined(RISCV64_SPACEMIT_IME2)
                // TODO
                // if (cur->ne[1] % 32 == 0 && cur->ne[0] % 256 == 0 &&
                //     (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                //     return &ggml::cpu::riscv64_spacemit::q4_1_32x256_q8_0;
                // }

                ਜੇ (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    ਵਾਪਸ &ggml::cpu::riscv64_spacemit::q4_1_32x32_q8_0;
                }
#ਜੇ_ਅੰਤ

#ਜੇ defined(RISCV64_SPACEMIT_IME1)
                ਜੇ (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    ਵਾਪਸ &ggml::cpu::riscv64_spacemit::q4_1_16x32_q8_0;
                }
#ਜੇ_ਅੰਤ
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q4_K:
            {
#ਜੇ defined(RISCV64_SPACEMIT_IME2)
                ਜੇ (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    ਵਾਪਸ &ggml::cpu::riscv64_spacemit::q4_k_32x32_q8_0;
                }
#ਜੇ_ਅੰਤ

#ਜੇ defined(RISCV64_SPACEMIT_IME1)
                ਜੇ (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    ਵਾਪਸ &ggml::cpu::riscv64_spacemit::q4_k_16x32_q8_0;
                }
#ਜੇ_ਅੰਤ
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q6_K:
            {
#ਜੇ defined(RISCV64_SPACEMIT_IME2)
                ਜੇ ((ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    ਵਾਪਸ &ggml::cpu::riscv64_spacemit::q6_k_32x32_q8_0;
                }
#ਜੇ_ਅੰਤ
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q8_0:
            {
#ਜੇ defined(RISCV64_SPACEMIT_IME2)
                ਜੇ ((ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    ਵਾਪਸ &ggml::cpu::riscv64_spacemit::q8_0_32x32_q8_0;
                }
#ਜੇ_ਅੰਤ

#ਜੇ defined(RISCV64_SPACEMIT_IME1)
                ਜੇ (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    ਵਾਪਸ &ggml::cpu::riscv64_spacemit::q8_0_16x32_q8_0;
                }
#ਜੇ_ਅੰਤ
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_MXFP4:
            {
#ਜੇ defined(RISCV64_SPACEMIT_IME2)
                // TODO
                // if (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                //     return &ggml::cpu::riscv64_spacemit::mxfp4_32x32_q8_0;
                // }
#ਜੇ_ਅੰਤ
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q5_K:
            {
#ਜੇ defined(RISCV64_SPACEMIT_IME2)
                ਜੇ (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    ਵਾਪਸ &ggml::cpu::riscv64_spacemit::q5_k_32x32_q8_0;
                }
#ਜੇ_ਅੰਤ
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q5_1:
            {
#ਜੇ defined(RISCV64_SPACEMIT_IME2)
                ਜੇ (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    ਵਾਪਸ &ggml::cpu::riscv64_spacemit::q5_1_32x32_q8_0;
                }
#ਜੇ_ਅੰਤ
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q5_0:
            {
#ਜੇ defined(RISCV64_SPACEMIT_IME2)
                ਜੇ (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    ਵਾਪਸ &ggml::cpu::riscv64_spacemit::q5_0_32x32_q8_0;
                }
#ਜੇ_ਅੰਤ
            }
            ਰੋਕੋ;
        ਮੂਲ:
            ਰੋਕੋ;
    }

    ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
}

ਸਥਿਰ ਗਿਣਤੀ ggml_ਸਥਿਤੀ ggml_backend_riscv64_spacemit_ਬਫਰ_init_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ,
                                                                         ggml_tensor *         tensor) {
    tensor->extra =
        (ਖਾਲੀ *) const_cast<ggml::cpu::tensor_traits *>(ggml_riscv64_spacemit_get_optimal_repack_type(tensor));

    GGML_ਅਣਵਰਤਿਆ(ਬਫਰ);

    ਵਾਪਸ GGML_ਸਥਿਤੀ_SUCCESS;
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_riscv64_spacemit_ਬਫਰ_free_ਬਫਰ(ggml_backend_ਬਫਰ_t ਬਫਰ) {
    GGML_ASSERT(ਬਫਰ);

    ਖਾਲੀ * ਅਧਾਰ = ਬਫਰ->context;
    ਜੇ (ਅਧਾਰ == ਖਾਲੀ_ਸੂਚਕ) {
        ਵਾਪਸ;
    }

    ggml::cpu::riscv64_spacemit::spine_mem_pool_free(ਅਧਾਰ);
}

ਸਥਿਰ ਖਾਲੀ * ggml_backend_riscv64_spacemit_ਬਫਰ_get_ਅਧਾਰ(ggml_backend_ਬਫਰ_t ਬਫਰ) {
    GGML_ASSERT(ਬਫਰ);

    ਖਾਲੀ * ਅਧਾਰ = ਬਫਰ->context;
    GGML_ASSERT(ਅਧਾਰ != ਖਾਲੀ_ਸੂਚਕ);
    ਵਾਪਸ ਅਧਾਰ;
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_riscv64_spacemit_ਬਫਰ_memset_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ,
                                                               ggml_tensor *         tensor,
                                                               uint8_t               value,
                                                               size_t                ਵਿੱਥ,
                                                               size_t                size) {
    GGML_ASSERT(tensor);
    memset((ਅੱਖਰ *) tensor->ਅੰਕੜੇ + ਵਿੱਥ, value, size);

    GGML_ਅਣਵਰਤਿਆ(ਬਫਰ);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_riscv64_spacemit_ਬਫਰ_clear(ggml_backend_ਬਫਰ_t ਬਫਰ, uint8_t value) {
    GGML_ASSERT(ਬਫਰ);

    ਖਾਲੀ * ਅਧਾਰ = ਬਫਰ->context;
    GGML_ASSERT(ਅਧਾਰ != ਖਾਲੀ_ਸੂਚਕ);
    memset(ਅਧਾਰ, value, ਬਫਰ->size);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_riscv64_spacemit_ਬਫਰ_ਸੈੱਟ_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ,
                                                            ggml_tensor *         tensor,
                                                            ਅਚਲ ਖਾਲੀ *          ਅੰਕੜੇ,
                                                            size_t                ਵਿੱਥ,
                                                            size_t                size) {
    GGML_ASSERT(ਵਿੱਥ == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    ਸਵੈਚਾਲਿਤ tensor_traits = (ggml::cpu::riscv64_spacemit::tensor_traits_ਅਧਾਰ *) tensor->extra;
    ਜੇ (tensor_traits) {
        ਸਵੈਚਾਲਿਤ OK = tensor_traits->repack(tensor, ਅੰਕੜੇ, size);
        GGML_ASSERT(OK == 0);
    }

    GGML_ਅਣਵਰਤਿਆ(ਬਫਰ);
}

ਸਥਿਰ ਅਚਲ ggml_backend_ਬਫਰ_i ggml_backend_riscv64_spacemit_ਬਫਰ_i = {
    /* .free_buffer     = */ ggml_backend_riscv64_spacemit_ਬਫਰ_free_ਬਫਰ,
    /* .get_base        = */ ggml_backend_riscv64_spacemit_ਬਫਰ_get_ਅਧਾਰ,
    /* .init_tensor     = */ ggml_backend_riscv64_spacemit_ਬਫਰ_init_tensor,
    /* .memset_tensor   = */ ggml_backend_riscv64_spacemit_ਬਫਰ_memset_tensor,
    /* .set_tensor      = */ ggml_backend_riscv64_spacemit_ਬਫਰ_ਸੈੱਟ_tensor,
    /* .get_tensor      = */ ਖਾਲੀ_ਸੂਚਕ,
    /* .set_tensor_2d   = */ ਖਾਲੀ_ਸੂਚਕ,
    /* .get_tensor_2d   = */ ਖਾਲੀ_ਸੂਚਕ,
    /* .cpy_tensor      = */ ਖਾਲੀ_ਸੂਚਕ,
    /* .clear           = */ ggml_backend_riscv64_spacemit_ਬਫਰ_clear,
    /* .reset           = */ ਖਾਲੀ_ਸੂਚਕ,
};

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_cpu_riscv64_spacemit_ਬਫਰ_type_get_name(ggml_backend_ਬਫਰ_type_t buft) {
    ਵਾਪਸ "CPU_RISCV64_SPACEMIT";

    GGML_ਅਣਵਰਤਿਆ(buft);
}

ਸਥਿਰ ggml_backend_ਬਫਰ_t ggml_backend_cpu_riscv64_spacemit_ਬਫਰ_type_alloc_ਬਫਰ(ggml_backend_ਬਫਰ_type_t buft,
                                                                                        size_t size) {
    ਖਾਲੀ * ਅਧਾਰ = ggml::cpu::riscv64_spacemit::spine_mem_pool_alloc(size, 64);
    ਜੇ (ਅਧਾਰ == ਖਾਲੀ_ਸੂਚਕ) {
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }

    ਵਾਪਸ ggml_backend_ਬਫਰ_init(buft, ggml_backend_riscv64_spacemit_ਬਫਰ_i, ਅਧਾਰ, size);
}

ਸਥਿਰ size_t ggml_backend_cpu_riscv64_spacemit_ਬਫਰ_type_get_alignment(ggml_backend_ਬਫਰ_type_t buft) {
    ਵਾਪਸ 64;

    GGML_ਅਣਵਰਤਿਆ(buft);
}

ਸਥਿਰ size_t ggml_backend_cpu_riscv64_spacemit_nbytes(ggml_backend_ਬਫਰ_type_t buft, ਅਚਲ ggml_tensor * tensor) {
    ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_DIMS; ++i) {
        ਜੇ (tensor->ne[i] <= 0) {
            ਵਾਪਸ 0;
        }
    }

    GGML_ਅਣਵਰਤਿਆ(buft);

    ਅਚਲ ਸਵੈਚਾਲਿਤ plain_nbytes = [&]() {
        size_t ਕੁੱਲ = ggml_type_ਮਾਪ(tensor->type);
        ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_DIMS; ++i) {
            ਕੁੱਲ += (tensor->ne[i] - 1) * tensor->nb[i];
        }
        ਵਾਪਸ ਕੁੱਲ;
    };

    ਅਚਲ size_t blck_ਮਾਪ = ggml_blck_ਮਾਪ(tensor->type);
    ਜੇ (blck_ਮਾਪ == 1) {
        ਵਾਪਸ plain_nbytes();
    }

    ਅਚਲ size_t ਪੰਕਤੀ_nbytes = tensor->ne[0] * tensor->nb[0] / blck_ਮਾਪ;

    ਅਚਲ ਸਵੈਚਾਲਿਤ ਜੋੜੋ_strided_nbytes = [&](size_t ਕੁੱਲ, size_t src_ਖੰਡ_ਮਾਪ, size_t dst_ਖੰਡ_ਮਾਪ) {
        ਲਈ (ਪੂਰਨ i = 1; i < GGML_MAX_DIMS; ++i) {
            ਕੁੱਲ += (tensor->ne[i] - 1) * (tensor->nb[i] / src_ਖੰਡ_ਮਾਪ) * dst_ਖੰਡ_ਮਾਪ;
        }
        ਵਾਪਸ ਕੁੱਲ;
    };

    ਅਚਲ ਸਵੈਚਾਲਿਤ remap_ਖੰਡ_nbytes = [&](size_t src_ਖੰਡ_ਮਾਪ, size_t dst_ਖੰਡ_ਮਾਪ, int64_t padded_ਪੰਕਤੀਆਂ = 0) {
        GGML_ASSERT(ਪੰਕਤੀ_nbytes % src_ਖੰਡ_ਮਾਪ == 0);

        size_t ਕੁੱਲ =
            ਜੋੜੋ_strided_nbytes((ਪੰਕਤੀ_nbytes / src_ਖੰਡ_ਮਾਪ) * dst_ਖੰਡ_ਮਾਪ, src_ਖੰਡ_ਮਾਪ, dst_ਖੰਡ_ਮਾਪ);

        ਜੇ (padded_ਪੰਕਤੀਆਂ > 0 && tensor->ne[1] % padded_ਪੰਕਤੀਆਂ != 0) {
            ਕੁੱਲ += (padded_ਪੰਕਤੀਆਂ - tensor->ne[1] % padded_ਪੰਕਤੀਆਂ) * (tensor->nb[1] / src_ਖੰਡ_ਮਾਪ) * dst_ਖੰਡ_ਮਾਪ;
        }

        ਵਾਪਸ ਕੁੱਲ;
    };

    size_t nbytes = ਪੰਕਤੀ_nbytes;
    ਚੋਣ (tensor->type) {
        ਹਾਲਤ GGML_TYPE_Q4_K:
            nbytes = remap_ਖੰਡ_nbytes(ਆਕਾਰ(ਖੰਡ_q4_K), ਆਕਾਰ(ਖੰਡ_q4_1) * 8);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q6_K:
            nbytes = remap_ਖੰਡ_nbytes(ਆਕਾਰ(ਖੰਡ_q6_K), ਆਕਾਰ(ਖੰਡ_q8_0) * 8, 32);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q8_0:
            nbytes = remap_ਖੰਡ_nbytes(ਆਕਾਰ(ਖੰਡ_q8_0), ਆਕਾਰ(ਖੰਡ_q8_0), 32);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q2_K:
            nbytes = remap_ਖੰਡ_nbytes(ਆਕਾਰ(ਖੰਡ_q2_K), ਆਕਾਰ(spacemit_kernels::nrow_ਖੰਡ_q2_k<1>));
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q3_K:
            nbytes = remap_ਖੰਡ_nbytes(ਆਕਾਰ(ਖੰਡ_q3_K), ਆਕਾਰ(spacemit_kernels::nrow_ਖੰਡ_q3_k<1>));
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_MXFP4:
            nbytes = remap_ਖੰਡ_nbytes(ਆਕਾਰ(ਖੰਡ_mxfp4), ਆਕਾਰ(spacemit_kernels::nrow_ਖੰਡ_mxfp4<1>));
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q5_K:
            nbytes = remap_ਖੰਡ_nbytes(ਆਕਾਰ(ਖੰਡ_q5_K), ਆਕਾਰ(spacemit_kernels::nrow_ਖੰਡ_q5_1<1>) * 8);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q5_1:
            nbytes = remap_ਖੰਡ_nbytes(ਆਕਾਰ(ਖੰਡ_q5_1), ਆਕਾਰ(spacemit_kernels::nrow_ਖੰਡ_q5_1<1>));
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q5_0:
            nbytes = remap_ਖੰਡ_nbytes(ਆਕਾਰ(ਖੰਡ_q5_0), ਆਕਾਰ(spacemit_kernels::nrow_ਖੰਡ_q5_0<1>));
            ਰੋਕੋ;
        ਮੂਲ:
            nbytes = ਜੋੜੋ_strided_nbytes(ਪੰਕਤੀ_nbytes, 1, 1);
            ਰੋਕੋ;
    }

    ਵਾਪਸ nbytes;
}

ਨਾਮਾਕਾਸ਼ ggml::cpu::riscv64_spacemit {

ਵਰਗ extra_ਬਫਰ_type : ggml::cpu::extra_ਬਫਰ_type {
    ਤਾਰਕਿਕ supports_op(ggml_backend_dev_t, ਅਚਲ ggml_tensor * op) override {
        ਚੋਣ (op->op) {
            ਹਾਲਤ GGML_OP_MUL_MAT:
                ਜੇ (op->src[0]->ਬਫਰ && (ggml_n_dims(op->src[0]) == 2) &&
                    op->src[0]->ਬਫਰ->buft == ggml_backend_cpu_riscv64_spacemit_ਬਫਰ_type() &&
                    ggml_riscv64_spacemit_get_optimal_repack_type(op->src[0])) {
                    ਜੇ (op->src[1]->ਬਫਰ && !ggml_backend_buft_is_host(op->src[1]->ਬਫਰ->buft)) {
                        ਵਾਪਸ ਝੂਠ;
                    }
                    ਜੇ (op->src[1]->type == GGML_TYPE_F32) {
                        ਵਾਪਸ ਸੱਚ;
                    }
                }
                ਰੋਕੋ;
            ਹਾਲਤ GGML_OP_MUL_MAT_ID:
                ਜੇ (op->src[0]->ਬਫਰ && (ggml_n_dims(op->src[0]) == 3) &&
                    op->src[0]->ਬਫਰ->buft == ggml_backend_cpu_riscv64_spacemit_ਬਫਰ_type() &&
                    ggml_riscv64_spacemit_get_optimal_repack_type(op->src[0])) {
                    ਜੇ (op->src[1]->ਬਫਰ && !ggml_backend_buft_is_host(op->src[1]->ਬਫਰ->buft)) {
                        ਵਾਪਸ ਝੂਠ;
                    }
                    ਜੇ (op->src[1]->type == GGML_TYPE_F32) {
                        ਵਾਪਸ ਸੱਚ;
                    }
                }
                ਰੋਕੋ;
            ਮੂਲ:
                // GGML_ABORT("fatal error");
                ਰੋਕੋ;
        }
        ਵਾਪਸ ਝੂਠ;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(ਅਚਲ ggml_tensor * op) override {
        ਚੋਣ (op->op) {
            ਹਾਲਤ GGML_OP_MUL_MAT:
            ਹਾਲਤ GGML_OP_MUL_MAT_ID:
                ਜੇ (op->src[0]->ਬਫਰ && op->src[0]->ਬਫਰ->buft == ggml_backend_cpu_riscv64_spacemit_ਬਫਰ_type()) {
                    ਵਾਪਸ (ggml::cpu::tensor_traits *) op->src[0]->extra;
                }
                ਰੋਕੋ;
            ਹਾਲਤ GGML_OP_NORM:
            ਹਾਲਤ GGML_OP_RMS_NORM:
            ਹਾਲਤ GGML_OP_ਜੋੜੋ:
            ਹਾਲਤ GGML_OP_SUB:
            ਹਾਲਤ GGML_OP_MUL:
            ਹਾਲਤ GGML_OP_DIV:
            ਹਾਲਤ GGML_OP_FLASH_ATTN_EXT:
            ਹਾਲਤ GGML_OP_CONT:
            ਹਾਲਤ GGML_OP_CPY:
            ਹਾਲਤ GGML_OP_REPEAT:
            ਹਾਲਤ GGML_OP_ਜੋੜ_ਪੰਕਤੀਆਂ:
            ਹਾਲਤ GGML_OP_GET_ਪੰਕਤੀਆਂ:
            ਹਾਲਤ GGML_OP_CONCAT:
                // case GGML_OP_GATED_DELTA_NET:
                ਵਾਪਸ (ggml::cpu::tensor_traits *) (&ggml::cpu::riscv64_spacemit::rvv_impl);
            ਮੂਲ:
                // GGML_ABORT("fatal error");
                ਰੋਕੋ;
        }

        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }
};

}  // namespace ggml::cpu::riscv64_spacemit

ggml_backend_ਬਫਰ_type_t ggml_backend_cpu_riscv64_spacemit_ਬਫਰ_type(ਖਾਲੀ) {
    ਸਥਿਰ ggml_backend_ਬਫਰ_type ggml_backend_cpu_ਬਫਰ_type_riscv64_spacemit = {
  /* .iface    = */
        {
         /* .get_name           = */ ggml_backend_cpu_riscv64_spacemit_ਬਫਰ_type_get_name,
         /* .alloc_buffer       = */ ggml_backend_cpu_riscv64_spacemit_ਬਫਰ_type_alloc_ਬਫਰ,
         /* .alloc_buffer_n     = */ ਨਿਰਾ,
         /* .get_alignment      = */ ggml_backend_cpu_riscv64_spacemit_ਬਫਰ_type_get_alignment,
         /* .get_max_size       = */ ਖਾਲੀ_ਸੂਚਕ,
         /* .get_alloc_size     = */ ggml_backend_cpu_riscv64_spacemit_nbytes,
         /* .get_alloc_size_n   = */ ਨਿਰਾ,
         /* .is_host            = */ ਖਾਲੀ_ਸੂਚਕ,
         },
 /* .device  = */
        ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
 /* .context = */
        ਨਵਾਂ ggml::cpu::riscv64_spacemit::extra_ਬਫਰ_type(),
    };

    ਵਾਪਸ &ggml_backend_cpu_ਬਫਰ_type_riscv64_spacemit;
}

ਬਾਹਰੀ "C" {
ਸਥਿਰ ਪੂਰਨ bind_ai_thread() {
    ਪੂਰਨ  fd, ਬਾਈਟਾਂ;
    ਅੱਖਰ str[32];

    fd = open("/proc/set_ai_thread", O_WRONLY);
    ਜੇ (fd < 0) {
        GGML_LOG_ERROR("try open /proc/set_ai_thread failed\n");
        ਵਾਪਸ -1;
    }

    snprintf(str, 16, "%d", 0);
    ਬਾਈਟਾਂ = ਲਿਖ(fd, str, ਲੜੀ_ਲੰਬਾਈ(str));
    ਜੇ (ਬਾਈਟਾਂ < 0) {
        GGML_LOG_ERROR("try write /proc/set_ai_thread failed\n");
        close(fd);
        ਵਾਪਸ -1;
    }

    close(fd);
    ਵਾਪਸ 0;
}

ਖਾਲੀ ggml_backend_cpu_riscv64_spacemit_ਸੈੱਟ_numa_thread_affinity(ਪੂਰਨ thread_n) {
    ਪੂਰਨ cpu_id = sched_getcpu();
    ਜੇ (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2 &&
        !((1 << cpu_id) & ggml::cpu::riscv64_spacemit::global_spine_env_info.cpu_ਨਕਾਬ)) {
        GGML_PRINT_DEBUG("bind_ai_thread for thread %d, pid %d\n", thread_n, getpid());
        bind_ai_thread();
    }

    ਜੇ (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_tcm &&
        ggml::cpu::riscv64_spacemit::tls_context.cpu_id == -1) {
        CPU_ZERO(&(ggml::cpu::riscv64_spacemit::tls_context.cpuset));
        pthread_t    main_thread     = pthread_self();
        ਅਚਲ ਸਵੈਚਾਲਿਤ & perfer_core_ids = ggml::cpu::riscv64_spacemit::global_spine_env_info.perfer_core_ids;
        ਜੇ (thread_n < 0 || static_cast<size_t>(thread_n) >= perfer_core_ids.size()) {
            GGML_ABORT("thread_n %d exceeds perfer_core_ids size %zu\n", thread_n, perfer_core_ids.size());
        }
        ਸਵੈਚਾਲਿਤ perfer_cpu_id = perfer_core_ids[static_cast<size_t>(thread_n)];
        CPU_SET(perfer_cpu_id, &(ggml::cpu::riscv64_spacemit::tls_context.cpuset));
        ਪੂਰਨ s =
            pthread_setaffinity_np(main_thread, ਆਕਾਰ(cpu_set_t), &(ggml::cpu::riscv64_spacemit::tls_context.cpuset));
        ਜੇ (s != 0) {
            GGML_ABORT("set thread affinity error for thread_n %d, cpu_id %d\n", thread_n, perfer_cpu_id);
        }

        ਪੂਰਨ ai_cpu_id = perfer_cpu_id - ggml::cpu::riscv64_spacemit::global_spine_env_info.aicpu_id_ਵਿੱਥ;
        ggml::cpu::riscv64_spacemit::tls_context.cpu_id = ai_cpu_id;
        ggml::cpu::riscv64_spacemit::tls_context.tcm_ਬਫਰ =
            ggml::cpu::riscv64_spacemit::spine_mem_pool_tcm_mem_get(ai_cpu_id);
        ggml::cpu::riscv64_spacemit::tls_context.tcm_ਬਫਰ_ਮਾਪ =
            ggml::cpu::riscv64_spacemit::global_spine_env_info.tcm_blk_ਮਾਪ;
    }

    ਜੇ (ggml::cpu::riscv64_spacemit::tls_context.tcm_ਬਫਰ != ਖਾਲੀ_ਸੂਚਕ) {
        ਖਾਲੀ * rt =
            ggml::cpu::riscv64_spacemit::spine_mem_pool_tcm_mem_wait(ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        ਜੇ (rt == ਖਾਲੀ_ਸੂਚਕ) {
            GGML_ABORT("wait tcm buffer failed for cpu_id: %d", ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        }
    }
}

ਖਾਲੀ ggml_backend_cpu_riscv64_spacemit_clear_numa_thread_affinity_threaded(ਪੂਰਨ thread_n) {
    ਜੇ (ggml::cpu::riscv64_spacemit::tls_context.tcm_ਬਫਰ != ਖਾਲੀ_ਸੂਚਕ) {
        ਸਵੈਚਾਲਿਤ rt = ggml::cpu::riscv64_spacemit::spine_mem_pool_tcm_mem_release(
            ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        ਜੇ (rt != 0) {
            GGML_ABORT("release tcm buffer failed for cpu_id: %d", ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        }
    }
}
}
