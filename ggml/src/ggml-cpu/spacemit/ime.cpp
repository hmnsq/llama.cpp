#define GGML_COMMON_IMPL_CPP
#define GGML_COMMON_DECL_CPP

#ଅନ୍ତର୍ଭାଵ "ime.h"

#ଅନ୍ତର୍ଭାଵ "binary-ops.h"
#ଅନ୍ତର୍ଭାଵ "common.h"
#ଅନ୍ତର୍ଭାଵ "ggml-backend-impl.h"
#ଅନ୍ତର୍ଭାଵ "ggml-common.h"
#ଅନ୍ତର୍ଭାଵ "ggml-cpu.h"
#ଅନ୍ତର୍ଭାଵ "ime_env.h"
#ଅନ୍ତର୍ଭାଵ "ime_kernels.h"
#ଅନ୍ତର୍ଭାଵ "ops.h"
#ଅନ୍ତର୍ଭାଵ "repack.h"
#ଅନ୍ତର୍ଭାଵ "rvv_kernels.h"
#ଅନ୍ତର୍ଭାଵ "spine_mem_pool.h"
#ଅନ୍ତର୍ଭାଵ "traits.h"
#ଅନ୍ତର୍ଭାଵ "vec.h"

#ଅନ୍ତର୍ଭାଵ <fcntl.h>
#ଅନ୍ତର୍ଭାଵ <sys/mman.h>
#ଅନ୍ତର୍ଭାଵ <unistd.h>

#ଅନ୍ତର୍ଭାଵ <algorithm>
#ଅନ୍ତର୍ଭାଵ <atomic>
#ଅନ୍ତର୍ଭାଵ <cassert>
#ଅନ୍ତର୍ଭାଵ <cerrno>
#ଅନ୍ତର୍ଭାଵ <cmath>
#ଅନ୍ତର୍ଭାଵ <cstdio>  // for GGML_ASSERT
#ଅନ୍ତର୍ଭାଵ <stdexcept>
#ଅନ୍ତର୍ଭାଵ <thread>
// clang-format off
#ଯଦି defined(__riscv)

#ଯଦି !defined(__riscv_v) || !defined(__riscv_v_intrinsic)
#error "riscv v extension or v_intrinsic not enabled"
#ଅନ୍ୟଥା
#ଅନ୍ତର୍ଭାଵ <riscv_vector.h>
#endif

#ଯଦି !defined(__riscv_zfh) || !defined(__riscv_zvfh)
#error "riscv zfh extension not enabled, GGML_RV_ZFH and GGML_RV_ZVFH must be defined to 1"
#endif

#ଯଦି !defined(__riscv_zba)
#error "riscv zba extension not enabled, GGML_RV_ZBA must be defined to 1"
#endif

#ଯଦି defined(RISCV64_SPACEMIT_IME1) || defined(RISCV64_SPACEMIT_IME2)
#ଅନ୍ୟଥା
#error "RISCV64_SPACEMIT_IME1 or RISCV64_SPACEMIT_IME2 not defined"
#endif

#ଅନ୍ୟଥା

#error "riscv not enabled in this build"

#endif

#ଯଦି defined(__GNUC__)
#pragma GCC diagnostic ignored "-Woverlength-strings"
#pragma GCC diagnostic ignored "-Wcast-qual"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

// clang-format on

ବାହ୍ୟ "C" {
ବାହ୍ୟ ଶୂନ୍ୟ ggml_threadpool_chunk_set(ସଂରଚନା ggml_threadpool * tp, ପୂର୍ଣ୍ଣସଂଖ୍ୟା value);
ବାହ୍ୟ ପୂର୍ଣ୍ଣସଂଖ୍ୟା  ggml_threadpool_chunk_add(ସଂରଚନା ggml_threadpool * tp, ପୂର୍ଣ୍ଣସଂଖ୍ୟା value);
}

ନାମାକାଶ ggml::cpu::riscv64_spacemit {

ସଂରଚନା TLSContext {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା       cpu_id{ -1 };
    cpu_set_t cpuset;
    ଶୂନ୍ୟ *    tcm_buffer{ nullptr };
    size_t    tcm_buffer_size{ 0 };
};

thread_local TLSContext tls_context;

template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS> constexpr size_t get_repacked_block_type_size() {
    ଯଦି constexpr (std::is_same_v<BLOC_TYPE, block_q6_K> || std::is_same_v<BLOC_TYPE, block_q8_0>) {
        ଫେରନ୍ତୁ ଆକାର(block_q8_0);
    } ଅନ୍ୟଥା ଯଦି constexpr (std::is_same_v<BLOC_TYPE, block_q4_0>) {
        ଫେରନ୍ତୁ ଆକାର(block_q4_0) * INTER_SIZE / QK4_0;
    } ଅନ୍ୟଥା ଯଦି constexpr (std::is_same_v<BLOC_TYPE, block_q4_1> || std::is_same_v<BLOC_TYPE, block_q4_K>) {
        ଫେରନ୍ତୁ (ଆକାର(block_q4_0) + ଆକାର(uint8_t)) * INTER_SIZE / QK4_1;
    } ଅନ୍ୟଥା ଯଦି constexpr (std::is_same_v<BLOC_TYPE, block_q2_K>) {
        ଫେରନ୍ତୁ ଆକାର(spacemit_kernels::nrow_block_q2_k<1>);
    } ଅନ୍ୟଥା ଯଦି constexpr (std::is_same_v<BLOC_TYPE, block_q3_K>) {
        ଫେରନ୍ତୁ ଆକାର(spacemit_kernels::nrow_block_q3_k<1>);
    } ଅନ୍ୟଥା ଯଦି constexpr (std::is_same_v<BLOC_TYPE, block_mxfp4>) {
        ଫେରନ୍ତୁ ଆକାର(spacemit_kernels::nrow_block_mxfp4<1>);
    } ଅନ୍ୟଥା ଯଦି constexpr (std::is_same_v<BLOC_TYPE, block_q5_1> || std::is_same_v<BLOC_TYPE, block_q5_K>) {
        ଫେରନ୍ତୁ ଆକାର(spacemit_kernels::nrow_block_q5_1<1>);
    } ଅନ୍ୟଥା ଯଦି constexpr (std::is_same_v<BLOC_TYPE, block_q5_0>) {
        ଫେରନ୍ତୁ ଆକାର(spacemit_kernels::nrow_block_q5_0<1>);
    } ଅନ୍ୟଥା {
        ପ୍ରତିଜ୍ଞା(ମିଥ୍ୟା);
        ଫେରନ୍ତୁ 0;
    }
}

template <typename BLOC_TYPE> constexpr ତାର୍କିକ block_type_has_zp() {
    ଯଦି constexpr (std::is_same_v<BLOC_TYPE, block_q6_K> || std::is_same_v<BLOC_TYPE, block_q8_0> ||
                  std::is_same_v<BLOC_TYPE, block_q3_K> || std::is_same_v<BLOC_TYPE, block_q4_0> ||
                  std::is_same_v<BLOC_TYPE, block_mxfp4> || std::is_same_v<BLOC_TYPE, block_q5_0>) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    } ଅନ୍ୟଥା ଯଦି constexpr (std::is_same_v<BLOC_TYPE, block_q4_1> || std::is_same_v<BLOC_TYPE, block_q4_K> ||
                         std::is_same_v<BLOC_TYPE, block_q2_K> || std::is_same_v<BLOC_TYPE, block_q5_1> ||
                         std::is_same_v<BLOC_TYPE, block_q5_K>) {
        ଫେରନ୍ତୁ ସତ୍ୟ;
    } ଅନ୍ୟଥା {
        ପ୍ରତିଜ୍ଞା(ମିଥ୍ୟା);
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
}

ଶ୍ରେଣୀ tensor_traits_base : ସର୍ବସାଧାରଣ ggml::cpu::tensor_traits {
  ସର୍ବସାଧାରଣ:
    ଆଭାସୀ ପୂର୍ଣ୍ଣସଂଖ୍ୟା repack(ggml_tensor * t, ଅଚଳ ଶୂନ୍ୟ * data, size_t data_size) = 0;
};

template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS> ଶ୍ରେଣୀ tensor_traits : ସର୍ବସାଧାରଣ tensor_traits_base {
    ତାର୍କିକ work_size(ପୂର୍ଣ୍ଣସଂଖ୍ୟା /* n_threads */, ଅଚଳ ggml_tensor * op, size_t & size) override {
        ବାଛ (op->op) {
            ସ୍ଥିତି GGML_OP_MUL_MAT:
                {
                    int64_t src1_nelements = ggml_nelements(op->src[1]);

                    ଯଦି constexpr (std::is_same_v<BLOC_TYPE, block_q2_K> || std::is_same_v<BLOC_TYPE, block_q3_K>) {
                        size =
                            spacemit_kernels::div_round_up(src1_nelements, QK_K) * spacemit_kernels::q8k_blk_size(QK_K);
                    } ଅନ୍ୟଥା ଯଦି constexpr (INTER_SIZE == QK4_0) {
                        size = spacemit_kernels::div_round_up(src1_nelements, QK4_0) *
                               spacemit_kernels::q8_blk_size(QK4_0, ସତ୍ୟ);
                    } ଅନ୍ୟଥା ଯଦି constexpr (INTER_SIZE == 256) {
                        size = spacemit_kernels::div_round_up(src1_nelements, 256) *
                               spacemit_kernels::q8_hp_blk_size(256, ସତ୍ୟ, ସତ୍ୟ);
                    } ଅନ୍ୟଥା {
                        GGML_ABORT("unsupported block type");
                    }

                    size = GGML_PAD(size, ଆକାର(int64_t));

                    ଫେରନ୍ତୁ ସତ୍ୟ;
                }
            ସ୍ଥିତି GGML_OP_MUL_MAT_ID:
                {
                    int64_t src1_nelements = ggml_nelements(op->src[1]);

                    ଯଦି constexpr (std::is_same_v<BLOC_TYPE, block_q2_K> || std::is_same_v<BLOC_TYPE, block_q3_K>) {
                        size =
                            spacemit_kernels::div_round_up(src1_nelements, QK_K) * spacemit_kernels::q8k_blk_size(QK_K);
                    } ଅନ୍ୟଥା ଯଦି constexpr (INTER_SIZE == QK4_0) {
                        size = spacemit_kernels::div_round_up(src1_nelements, QK4_0) *
                               spacemit_kernels::q8_blk_size(QK4_0, ସତ୍ୟ);
                    } ଅନ୍ୟଥା ଯଦି constexpr (INTER_SIZE == 256) {
                        size = spacemit_kernels::div_round_up(src1_nelements, 256) *
                               spacemit_kernels::q8_hp_blk_size(256, ସତ୍ୟ, ସତ୍ୟ);
                    } ଅନ୍ୟଥା {
                        GGML_ABORT("unsupported block type");
                    }

                    size = GGML_PAD(size, ଆକାର(int64_t));

                    ଅଚଳ int64_t ne02 = op->src[0]->ne[2];  // n_as, n_expert
                    ଅଚଳ int64_t ne12 = op->src[1]->ne[2];  // n_tokens

                    ଅଚଳ size_t sizeof_mmid_row_mapping = ଆକାର(int64_t);
                    size += sizeof_mmid_row_mapping * ne02 * (ne12 + 1) + (ne02 + 1) * ଆକାର(int64_t);

                    size = GGML_PAD(size, ଆକାର(int64_t));

                    ଫେରନ୍ତୁ ସତ୍ୟ;
                }
            ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                // GGML_ABORT("fatal error");
                ଭାଙ୍ଗ;
        }
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ତାର୍କିକ compute_forward(ggml_compute_params * params, ggml_tensor * op) override {
        ବାଛ (op->op) {
            ସ୍ଥିତି GGML_OP_MUL_MAT:
                ବାଛ (op->src[0]->type) {
                    ସ୍ଥିତି GGML_TYPE_Q2_K:
                    ସ୍ଥିତି GGML_TYPE_Q3_K:
                    ସ୍ଥିତି GGML_TYPE_Q4_0:
                    ସ୍ଥିତି GGML_TYPE_Q4_1:
                    ସ୍ଥିତି GGML_TYPE_Q4_K:
                    ସ୍ଥିତି GGML_TYPE_Q6_K:
                    ସ୍ଥିତି GGML_TYPE_Q8_0:
                    ସ୍ଥିତି GGML_TYPE_Q5_0:
                    ସ୍ଥିତି GGML_TYPE_Q5_1:
                    ସ୍ଥିତି GGML_TYPE_Q5_K:
                        //case GGML_TYPE_MXFP4:
                        forward_mul_mat(params, op);
                        ଫେରନ୍ତୁ ସତ୍ୟ;
                    ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                        // GGML_ABORT("fatal error: unsupported type for src0 in MUL_MAT");
                        ଫେରନ୍ତୁ ମିଥ୍ୟା;
                }
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_OP_MUL_MAT_ID:
                ବାଛ (op->src[0]->type) {
                    ସ୍ଥିତି GGML_TYPE_Q2_K:
                    ସ୍ଥିତି GGML_TYPE_Q3_K:
                    ସ୍ଥିତି GGML_TYPE_Q4_0:
                    ସ୍ଥିତି GGML_TYPE_Q4_1:
                    ସ୍ଥିତି GGML_TYPE_Q4_K:
                    ସ୍ଥିତି GGML_TYPE_Q6_K:
                    ସ୍ଥିତି GGML_TYPE_Q8_0:
                    ସ୍ଥିତି GGML_TYPE_Q5_0:
                    ସ୍ଥିତି GGML_TYPE_Q5_1:
                    ସ୍ଥିତି GGML_TYPE_Q5_K:
                        //case GGML_TYPE_MXFP4:
                        forward_mul_mat_id(params, op);
                        ଫେରନ୍ତୁ ସତ୍ୟ;
                    ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                        // GGML_ABORT("fatal error: unsupported type for src0 in MUL_MAT_ID");
                        ଫେରନ୍ତୁ ମିଥ୍ୟା;
                }
                ଭାଙ୍ଗ;
            ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                // GGML_ABORT("fatal error");
                ଭାଙ୍ଗ;
        }
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଶୂନ୍ୟ forward_mul_mat(ggml_compute_params * params, ggml_tensor * op) {
        constexpr size_t a_blk_len = INTER_SIZE;
        constexpr size_t b_blk_len = INTER_SIZE;

        ଅଚଳ ggml_tensor * src0 = op->src[0];
        ଅଚଳ ggml_tensor * src1 = op->src[1];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_ଦ୍ୱିମିକ_OP_LOCALS

        ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

        [[maybe_unused]] ଅଚଳ ଗଣନା ggml_type type = src0->type;

        ଶୂନ୍ୟ *        w_data  = (ଶୂନ୍ୟ *) src0->data;
        ଅଚଳ ଦଶମିକ * feature = (ଅଚଳ ଦଶମିକ *) src1->data;
        ଦଶମିକ *       output  = (ଦଶମିକ *) dst->data;

        ଅଚଳ int64_t gemm_m = ne11 * ne12 * ne13;
        ଅଚଳ int64_t gemm_k = ne10;
        ଅଚଳ int64_t gemm_n = ne01;

        spacemit_kernels::quantize_a_row_def       quantize_a_row_i8;
        spacemit_kernels::quantize_a_row_def       quantize_a_4row_i8;
        spacemit_kernels::gemm_kernel_quantize_def gemm_kernel;
        ତାର୍କିକ                                       set_kernel_impl = ମିଥ୍ୟା;

        int64_t block_stride_a = spacemit_kernels::q8_blk_size(a_blk_len);

#ଯଦି defined(RISCV64_SPACEMIT_IME2)
        ଯଦି (!set_kernel_impl && (global_spine_env_info.use_ime2)) {
            quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8;
            quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8;
            block_stride_a     = spacemit_kernels::q8_blk_size(a_blk_len, ସତ୍ୟ);

            ଯଦି constexpr (std::is_same_v<BLOC_TYPE, block_q6_K> || std::is_same_v<BLOC_TYPE, block_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i8;
                set_kernel_impl = ସତ୍ୟ;
            } ଅନ୍ୟଥା ଯଦି constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> || std::is_same_v<BLOC_TYPE, block_q4_1> ||
                                 std::is_same_v<BLOC_TYPE, block_q4_K>) {
                ଯଦି constexpr (INTER_SIZE == 256) {
                    gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i4_hp;
                    quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8_hp;
                    quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8_hp;
                    block_stride_a     = spacemit_kernels::q8_hp_blk_size(a_blk_len, ସତ୍ୟ, ସତ୍ୟ);
                    set_kernel_impl    = ସତ୍ୟ;
                } ଅନ୍ୟଥା {
                    gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i4;
                    quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8;
                    quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8;
                    block_stride_a     = spacemit_kernels::q8_blk_size(a_blk_len, ସତ୍ୟ);
                    set_kernel_impl    = ସତ୍ୟ;
                }
            } ଅନ୍ୟଥା ଯଦି constexpr (std::is_same_v<BLOC_TYPE, block_q2_K>) {
                quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8k;
                quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8k;
                block_stride_a     = spacemit_kernels::q8k_blk_size(a_blk_len);

                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i2k;
                set_kernel_impl = ସତ୍ୟ;
            } ଅନ୍ୟଥା ଯଦି constexpr (std::is_same_v<BLOC_TYPE, block_q3_K>) {
                quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8k;
                quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8k;
                block_stride_a     = spacemit_kernels::q8k_blk_size(a_blk_len);

                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i3k;
                set_kernel_impl = ସତ୍ୟ;
            } ଅନ୍ୟଥା ଯଦି constexpr (std::is_same_v<BLOC_TYPE, block_mxfp4>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8mxfp4;
                set_kernel_impl = ସତ୍ୟ;
            } ଅନ୍ୟଥା ଯଦି constexpr (std::is_same_v<BLOC_TYPE, block_q5_1> || std::is_same_v<BLOC_TYPE, block_q5_K> ||
                                 std::is_same_v<BLOC_TYPE, block_q5_0>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i5;
                set_kernel_impl = ସତ୍ୟ;
            }
        }
#endif

#ଯଦି defined(RISCV64_SPACEMIT_IME1)
        ଯଦି (!set_kernel_impl && (global_spine_env_info.use_ime1)) {
            quantize_a_row_i8  = spacemit_kernels::ime1::quantize_a_row_i8;
            quantize_a_4row_i8 = spacemit_kernels::ime1::quantize_a_4row_i8;

            ଯଦି constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> || std::is_same_v<BLOC_TYPE, block_q4_1> ||
                          std::is_same_v<BLOC_TYPE, block_q4_K>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i4;
                set_kernel_impl = ସତ୍ୟ;
            } ଅନ୍ୟଥା ଯଦି constexpr (std::is_same_v<BLOC_TYPE, block_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i8;
                set_kernel_impl = ସତ୍ୟ;
            }
        }
#endif
        ଯଦି (!set_kernel_impl) {
            GGML_ABORT("no kernel implementation found for the block type");
        }

        ଅଚଳ int64_t a_k_blks = spacemit_kernels::div_round_up(gemm_k, a_blk_len);
        ଅଚଳ int64_t b_k_blks = spacemit_kernels::div_round_up(gemm_k, b_blk_len);

        ଅଚଳ int64_t row_stride_a        = a_k_blks * block_stride_a;
        ଅଚଳ int64_t gemm_workspace_size = GGML_PAD(gemm_m * row_stride_a, alignof(int64_t));

        ଯଦି (ith == 0 && params->wsize < gemm_workspace_size) {
            GGML_ABORT("wsize less than gemm_workspace_size");
        }

        uintptr_t ws_ptr = reinterpret_cast<uintptr_t>(params->wdata);

        ଶୂନ୍ୟ *        tcm_buffer      = ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer;
        ଅଚଳ int64_t tcm_buffer_size = ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer_size;

        auto * quant_a_buffer = reinterpret_cast<uint8_t *>(ws_ptr);

        constexpr int64_t row_align = 4;
        ଅଚଳ int64_t     row_blks  = spacemit_kernels::div_round_up(gemm_m, row_align);

        ଅଚଳ int64_t row_stride_b      = b_k_blks * get_repacked_block_type_size<BLOC_TYPE, INTER_SIZE, NB_COLS>();
        ଅଚଳ int64_t per_mb_rows_wsize = row_align * row_stride_a;
        ଅଚଳ int64_t per_nb_cols_wsize = NB_COLS * row_stride_b;

        ଅଚଳ int64_t barrier_idx = static_cast<int64_t>(ith / 2);

        GGML_ASSERT(global_spine_env_info.init_barrier != nullptr);
        GGML_ASSERT(barrier_idx < spine_init_barrier_ଗଣନା);
        spine_barrier_t * cur_barrier = &global_spine_env_info.init_barrier[barrier_idx];

        ଯଦି (gemm_m == 1) {
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା task_per_thread = spacemit_kernels::div_round_up(a_k_blks, nth);
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା a_blk_start     = ith * task_per_thread;
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା a_blk_end       = std::min(a_blk_start + task_per_thread, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) a_k_blks);
            ଯଦି (a_blk_start < a_blk_end) {
                quantize_a_row_i8(a_blk_len, feature + a_blk_start * a_blk_len, (a_blk_end - a_blk_start) * a_blk_len,
                                  quant_a_buffer + a_blk_start * block_stride_a);
            }
        } ଅନ୍ୟଥା {
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା task_per_thread = spacemit_kernels::div_round_up(row_blks, nth);
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା m_row_blk_start = ith * task_per_thread;
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା m_row_blk_end   = std::min(m_row_blk_start + task_per_thread, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) row_blks);
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା m_row_blk = m_row_blk_start; m_row_blk < m_row_blk_end; m_row_blk++) {
                ପୂର୍ଣ୍ଣସଂଖ୍ୟା m_idx             = m_row_blk * row_align;
                ପୂର୍ଣ୍ଣସଂଖ୍ୟା rows_tobe_handled = (gemm_m - m_idx) > row_align ? row_align : (gemm_m - m_idx);

                ଯଦି (rows_tobe_handled == row_align && quantize_a_4row_i8 != nullptr) {
                    ଅଚଳ ଦଶମିକ * a_row_ptr       = feature + m_idx * gemm_k;
                    auto *        quant_a_row_ptr = quant_a_buffer + m_idx * row_stride_a;
                    quantize_a_4row_i8(a_blk_len, a_row_ptr, gemm_k, quant_a_row_ptr);
                } ଅନ୍ୟଥା {
                    ଯେତେବେଳେ (rows_tobe_handled) {
                        ଅଚଳ ଦଶମିକ * a_row_ptr       = feature + m_idx * gemm_k;
                        auto *        quant_a_row_ptr = quant_a_buffer + m_idx * row_stride_a;
                        quantize_a_row_i8(a_blk_len, a_row_ptr, gemm_k, quant_a_row_ptr);
                        rows_tobe_handled -= 1;
                        m_idx += 1;
                    }
                }
            }
        }

        ggml_barrier(params->threadpool);

        ଅଚଳ int64_t gemm_m_stride     = gemm_n / gemm_m > 64 ? gemm_m : 16;
        ଅଚଳ int64_t gemm_m_blocked    = spacemit_kernels::div_round_up(gemm_m, gemm_m_stride);
        ଅଚଳ int64_t max_gemm_n_stride = spacemit_kernels::div_round_up(gemm_n * gemm_m_blocked, nth);

        int64_t gemm_n_stride = gemm_n;
        ଯଦି (max_gemm_n_stride < gemm_n) {
            gemm_n_stride =
                std::min(gemm_n_stride, spacemit_kernels::div_round_up(max_gemm_n_stride, NB_COLS) * NB_COLS);
        }

        ଯଦି (gemm_n_stride == gemm_n && tcm_buffer != nullptr && per_mb_rows_wsize <= tcm_buffer_size) {
            ପାଇଁ (int64_t m_start = ith * row_align; m_start < gemm_m; m_start += row_align * nth) {
                uint8_t * b_col    = reinterpret_cast<uint8_t *>(w_data);
                uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? b_col : nullptr;

                int64_t m_row_real = std::min(gemm_m - m_start, row_align);

                spacemit_kernels::rvv::memcpy1d(tcm_buffer, quant_a_buffer + m_start * row_stride_a,
                                                m_row_real * row_stride_a);

                int64_t n_blk_real = 0;
                ପାଇଁ (int64_t ni = 0; ni < gemm_n; ni += n_blk_real, b_col += n_blk_real * row_stride_b) {
                    n_blk_real = std::min(gemm_n - ni, (int64_t) NB_COLS);

                    uint8_t * a_row_ptr = (uint8_t *) tcm_buffer;
                    ଦଶମିକ *   c_blk     = output + m_start * gemm_n + ni;

                    int32_t rows_remaining = m_row_real;

                    ଯେତେବେଳେ (rows_remaining > 0) {
                        auto rows_handled = gemm_kernel(b_blk_len, a_row_ptr, b_col, b_col_zp, c_blk, rows_remaining,
                                                        n_blk_real, b_k_blks, gemm_n);

                        c_blk += rows_handled * gemm_n;
                        a_row_ptr += rows_handled * row_stride_a;

                        rows_remaining -= rows_handled;
                    }
                }
            }
        } ଅନ୍ୟଥା ଯଦି (tcm_buffer != nullptr && per_nb_cols_wsize <= tcm_buffer_size) {
            uint8_t * a_row = quant_a_buffer;
            uint8_t * b_col = reinterpret_cast<uint8_t *>(tcm_buffer);
            ଯଦି ((gemm_workspace_size + per_nb_cols_wsize) <= tcm_buffer_size) {
                a_row = (uint8_t *) tcm_buffer;
                b_col = reinterpret_cast<uint8_t *>(tcm_buffer) + gemm_workspace_size;
            }
            uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? b_col : nullptr;

            int64_t ni      = ith * NB_COLS;
            int64_t nb_real = std::min(gemm_n - ni, NB_COLS);

            ଯଦି (ith % 2 == 0 && nb_real > 0) {
                spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(w_data) + ni * row_stride_b,
                                                nb_real * row_stride_b);
                ଯଦି (a_row != quant_a_buffer) {
                    spacemit_kernels::rvv::memcpy1d(a_row, quant_a_buffer, gemm_workspace_size);
                }
            }

            spine_barrier_wait(cur_barrier);

            ଯଦି (ith % 2 != 0 && nb_real > 0) {
                ଯଦି (a_row != quant_a_buffer) {
                    spacemit_kernels::rvv::memcpy1d(a_row, quant_a_buffer, gemm_workspace_size);
                }
                spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(w_data) + ni * row_stride_b,
                                                nb_real * row_stride_b);
            }

            ପାଇଁ (; ni < gemm_n; ni += NB_COLS * nth) {
                int64_t rows_remaining = gemm_m;
                ଦଶମିକ * c_blk          = output + ni;
                auto *  a_row_cur      = a_row;

                ଯଦି (ith % 2 != 0) {
                    spine_barrier_wait(cur_barrier);
                }

                ଯେତେବେଳେ (rows_remaining > 0) {
                    auto rows_handled = gemm_kernel(b_blk_len, a_row_cur, b_col, b_col_zp, c_blk, rows_remaining,
                                                    nb_real, b_k_blks, gemm_n);

                    c_blk += rows_handled * gemm_n;
                    a_row_cur += rows_handled * row_stride_a;

                    rows_remaining -= rows_handled;
                }

                ଯଦି (ith % 2 == 0) {
                    spine_barrier_wait(cur_barrier);
                }

                ଅଚଳ int64_t next_ni = ni + NB_COLS * nth;
                ଯଦି (next_ni < gemm_n) {
                    nb_real = std::min(gemm_n - next_ni, NB_COLS);
                    spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(w_data) + next_ni * row_stride_b,
                                                    nb_real * row_stride_b);
                }
            }
        } ଅନ୍ୟଥା {
            ଅଚଳ int64_t task_ଗଣନା_m = spacemit_kernels::div_round_up(gemm_m, gemm_m_stride);
            ଅଚଳ int64_t task_ଗଣନା_n = spacemit_kernels::div_round_up(gemm_n, gemm_n_stride);

            int64_t task_ଗଣନା      = task_ଗଣନା_m * task_ଗଣନା_n;
            int64_t task_per_thread = (task_ଗଣନା + nth - 1) / nth;
            int64_t start           = ith * task_per_thread;
            int64_t end             = std::min((ith + 1) * task_per_thread, task_ଗଣନା);
            ପାଇଁ (int64_t compute_idx = start; compute_idx < end; compute_idx++) {
                ଅଚଳ auto tid_n = compute_idx / task_ଗଣନା_m;
                ଅଚଳ auto tid_m = compute_idx % task_ଗଣନା_m;

                ଅଚଳ int64_t m_start = tid_m * gemm_m_stride;
                ଅଚଳ int64_t m_ଗଣନା = std::min(gemm_m - m_start, (int64_t) gemm_m_stride);

                ଅଚଳ int64_t n_start = tid_n * gemm_n_stride;
                ଅଚଳ int64_t n_ଗଣନା = std::min(gemm_n - n_start, (int64_t) gemm_n_stride);

                ଅଚଳ int64_t n_blk = m_ଗଣନା == 1 ? n_ଗଣନା : NB_COLS;

                uint8_t * b_col    = reinterpret_cast<uint8_t *>(w_data) + n_start * row_stride_b;
                uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? b_col : nullptr;

                int64_t n_blk_real = 0;
                ପାଇଁ (int64_t ni = 0; ni < n_ଗଣନା; ni += n_blk_real, b_col += n_blk_real * row_stride_b) {
                    n_blk_real = std::min(n_ଗଣନା - ni, n_blk);

                    uint8_t * a_row = quant_a_buffer + m_start * row_stride_a;

                    ଦଶମିକ * c_blk = output + m_start * gemm_n + n_start + ni;

                    int64_t rows_remaining = m_ଗଣନା;

                    uint8_t * b_col_cur    = b_col;
                    uint8_t * b_col_zp_cur = b_col_zp;

                    ଯେତେବେଳେ (rows_remaining > 0) {
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

    ଶୂନ୍ୟ forward_mul_mat_id(ggml_compute_params * params, ggml_tensor * op) {
        constexpr size_t a_blk_len = INTER_SIZE;
        constexpr size_t b_blk_len = INTER_SIZE;

        ଅଚଳ ggml_tensor * src0 = op->src[0];
        ଅଚଳ ggml_tensor * src1 = op->src[1];
        ଅଚଳ ggml_tensor * ids  = op->src[2];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_ଦ୍ୱିମିକ_OP_LOCALS

        ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

        // row groups
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_ids = ids->ne[0];  // n_expert_used
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_as  = ne02;        // n_expert

        ସଂରଚନା mmid_row_mapping {
            int32_t i1;
            int32_t i2;
        };

        spacemit_kernels::quantize_a_row_def           quantize_a_row_i8;
        spacemit_kernels::gemm_kernel_quantize_def     gemm_kernel;
        spacemit_kernels::moe_gemm_kernel_quantize_def moe_gemm_kernel_m2;
        ତାର୍କିକ                                           set_kernel_impl = ମିଥ୍ୟା;
        size_t                                         block_stride_a  = spacemit_kernels::q8_blk_size(QK4_0);

#ଯଦି defined(RISCV64_SPACEMIT_IME2)
        ଯଦି (!set_kernel_impl && (global_spine_env_info.use_ime2)) {
            quantize_a_row_i8 = spacemit_kernels::rvv::quantize_a_row_i8;
            block_stride_a    = spacemit_kernels::q8_blk_size(QK4_0, ସତ୍ୟ);

            ଯଦି constexpr (std::is_same_v<BLOC_TYPE, block_q6_K> || std::is_same_v<BLOC_TYPE, block_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i8;
                set_kernel_impl = ସତ୍ୟ;
            } ଅନ୍ୟଥା ଯଦି constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> || std::is_same_v<BLOC_TYPE, block_q4_1> ||
                                 std::is_same_v<BLOC_TYPE, block_q4_K>) {
                ଯଦି constexpr (INTER_SIZE == 256) {
                    gemm_kernel       = spacemit_kernels::ime2::gemm_kernel_i8i4_hp;
                    quantize_a_row_i8 = spacemit_kernels::rvv::quantize_a_row_i8_hp;
                    block_stride_a    = spacemit_kernels::q8_hp_blk_size(a_blk_len, ସତ୍ୟ, ସତ୍ୟ);
                    set_kernel_impl   = ସତ୍ୟ;
                } ଅନ୍ୟଥା {
                    gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i4;
                    moe_gemm_kernel_m2 = spacemit_kernels::ime2::moe_m2_gemm_kernel_i8i4;
                    quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8;
                    block_stride_a     = spacemit_kernels::q8_blk_size(a_blk_len, ସତ୍ୟ);
                    set_kernel_impl    = ସତ୍ୟ;
                }
            } ଅନ୍ୟଥା ଯଦି constexpr (std::is_same_v<BLOC_TYPE, block_q2_K>) {
                quantize_a_row_i8 = spacemit_kernels::rvv::quantize_a_row_i8k;
                block_stride_a    = spacemit_kernels::q8k_blk_size(a_blk_len);
                gemm_kernel       = spacemit_kernels::ime2::gemm_kernel_i8i2k;
                set_kernel_impl   = ସତ୍ୟ;
            } ଅନ୍ୟଥା ଯଦି constexpr (std::is_same_v<BLOC_TYPE, block_q3_K>) {
                quantize_a_row_i8 = spacemit_kernels::rvv::quantize_a_row_i8k;
                block_stride_a    = spacemit_kernels::q8k_blk_size(a_blk_len);
                gemm_kernel       = spacemit_kernels::ime2::gemm_kernel_i8i3k;
                set_kernel_impl   = ସତ୍ୟ;
            } ଅନ୍ୟଥା ଯଦି constexpr (std::is_same_v<BLOC_TYPE, block_mxfp4>) {
                gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8mxfp4;
                moe_gemm_kernel_m2 = spacemit_kernels::ime2::moe_m2_gemm_kernel_i8mxfp4;
                set_kernel_impl    = ସତ୍ୟ;
            } ଅନ୍ୟଥା ଯଦି constexpr (std::is_same_v<BLOC_TYPE, block_q5_1> || std::is_same_v<BLOC_TYPE, block_q5_K> ||
                                 std::is_same_v<BLOC_TYPE, block_q5_0>) {
                gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i5;
                moe_gemm_kernel_m2 = spacemit_kernels::ime2::moe_m2_gemm_kernel_i8i5;
                set_kernel_impl    = ସତ୍ୟ;
            }
        }
#endif

#ଯଦି defined(RISCV64_SPACEMIT_IME1)
        ଯଦି (!set_kernel_impl && (global_spine_env_info.use_ime1)) {
            quantize_a_row_i8 = spacemit_kernels::ime1::quantize_a_row_i8;

            ଯଦି constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> || std::is_same_v<BLOC_TYPE, block_q4_1> ||
                          std::is_same_v<BLOC_TYPE, block_q4_K>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i4;
                set_kernel_impl = ସତ୍ୟ;
            } ଅନ୍ୟଥା ଯଦି constexpr (std::is_same_v<BLOC_TYPE, block_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i8;
                set_kernel_impl = ସତ୍ୟ;
            }
        }
#endif
        ଯଦି (!set_kernel_impl) {
            GGML_ABORT("no kernel implementation found for the block type");
        }

        ଅଚଳ size_t a_k_blks = spacemit_kernels::div_round_up(ne10, a_blk_len);
        ଅଚଳ size_t b_k_blks = spacemit_kernels::div_round_up(ne10, b_blk_len);

        ଅଚଳ size_t nbw1                = a_k_blks * block_stride_a;
        ଅଚଳ size_t nbw2                = ne11 * nbw1;
        ଅଚଳ size_t nbw3                = nbw2 * ne12;
        ଅଚଳ size_t gemm_workspace_size = GGML_PAD(nbw3, alignof(int64_t));

        ଅଚଳ uintptr_t ws_ptr         = reinterpret_cast<uintptr_t>(params->wdata);
        auto *          quant_a_buffer = reinterpret_cast<uint8_t *>(ws_ptr);

        ଯଦି (ne11 == 1) {
            ପାଇଁ (int64_t ii = ith; ii < ne12 * a_k_blks; ii += nth) {
                int64_t i12       = ii / a_k_blks;
                int64_t ak_blk_id = ii % a_k_blks;
                quantize_a_row_i8(a_blk_len, (ଦଶମିକ *) ((ଅକ୍ଷର *) src1->data + i12 * nb12) + ak_blk_id * a_blk_len,
                                  a_blk_len, quant_a_buffer + i12 * nbw2 + ak_blk_id * block_stride_a);
            }
        } ଅନ୍ୟଥା {
            ପାଇଁ (int64_t ii = ith; ii < ne12 * ne11; ii += nth) {
                int64_t i12 = ii / ne11;
                int64_t i11 = ii % ne11;
                quantize_a_row_i8(a_blk_len, (ଦଶମିକ *) ((ଅକ୍ଷର *) src1->data + i12 * nb12 + i11 * nb11), ne10,
                                  quant_a_buffer + i12 * nbw2 + i11 * nbw1);
            }
        }

#define MMID_MATRIX_ROW(row_id, i1) matrix_rows[(row_id) *ne12 + (i1)]

        int64_t *          matrix_row_counts       = (int64_t *) (ws_ptr + gemm_workspace_size);
        int32_t *          valid_ep_ଗଣନା          = (int32_t *) (matrix_row_counts + n_as);
        int32_t *          valid_act_ଗଣନା         = (int32_t *) (valid_ep_ଗଣନା + 1);
        int64_t *          valid_matrix_row_counts = (int64_t *) (valid_act_ଗଣନା + 1);
        mmid_row_mapping * matrix_rows             = (mmid_row_mapping *) (valid_matrix_row_counts + n_as);

        ଯଦି (ith == 0) {
            // initialize matrix_row_counts
            memset(matrix_row_counts, 0, n_as * ଆକାର(int64_t));

            // group rows by src0 matrix
            ପାଇଁ (int32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
                ପାଇଁ (int32_t id = 0; id < n_ids; ++id) {
                    ଅଚଳ int32_t i02 =
                        *(ଅଚଳ int32_t *) ((ଅଚଳ ଅକ୍ଷର *) ids->data + iid1 * ids->nb[1] + id * ids->nb[0]);

                    GGML_ASSERT(i02 >= 0 && i02 < n_as);

                    MMID_MATRIX_ROW(i02, matrix_row_counts[i02]) = { id, iid1 };
                    matrix_row_counts[i02] += 1;
                }
            }

            int32_t valid_ep_ଗଣନା_t  = 0;
            int32_t valid_act_ଗଣନା_t = 0;
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା cur_a = 0; cur_a < n_as; ++cur_a) {
                ଅଚଳ int64_t cne1 = matrix_row_counts[cur_a];
                ଯଦି (cne1 == 0) {
                    ଜାରି;
                }
                valid_matrix_row_counts[valid_ep_ଗଣନା_t] = cur_a;
                valid_act_ଗଣନା_t += cne1;
                valid_ep_ଗଣନା_t += 1;
            }
            valid_ep_ଗଣନା[0]  = valid_ep_ଗଣନା_t;
            valid_act_ଗଣନା[0] = valid_act_ଗଣନା_t;
        }

        ଅଚଳ int64_t barrier_idx = static_cast<int64_t>(ith / 2);

        GGML_ASSERT(global_spine_env_info.init_barrier != nullptr);
        GGML_ASSERT(barrier_idx < spine_init_barrier_ଗଣନା);
        spine_barrier_t * cur_barrier = &global_spine_env_info.init_barrier[barrier_idx];

        ggml_barrier(params->threadpool);

        ଅଚଳ size_t row_stride_b      = b_k_blks * get_repacked_block_type_size<BLOC_TYPE, INTER_SIZE, NB_COLS>();
        ଅଚଳ size_t expert_b_stride   = ne01 * row_stride_b;
        ଅଚଳ size_t per_nb_cols_wsize = NB_COLS * row_stride_b;

        std::array<ଅଚଳ uint8_t *, 2> src_workspaces;
        std::array<ଦଶମିକ *, 2>         dst_workspaces;

        auto *     tcm_buffer      = ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer;
        ଅଚଳ auto tcm_buffer_size = ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer_size;

        ଅଚଳ auto valid_ep_ଗଣନା_t  = valid_ep_ଗଣନା[0];
        ଅଚଳ auto valid_act_ଗଣନା_t = valid_act_ଗଣନା[0];

        ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth_es = 1;
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth_n  = nth;

        ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith_es = ith % nth_es;
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith_n  = (ith / nth_es) % nth_n;

        ଯଦି (valid_ep_ଗଣନା_t % nth == 0 && tcm_buffer != nullptr && valid_ep_ଗଣନା_t == n_as &&
            valid_act_ଗଣନା_t == n_as && per_nb_cols_wsize <= tcm_buffer_size) {
            ପାଇଁ (int64_t valid_id = ith; valid_id < valid_ep_ଗଣନା_t; valid_id += nth) {
                ଅଚଳ int64_t cur_a = valid_matrix_row_counts[valid_id];

                auto * src0_cur = (uint8_t *) src0->data + cur_a * expert_b_stride;

                mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, 0);
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା        id          = row_mapping.i1;
                ଅଚଳ int64_t    i11         = id % ne11;
                ଅଚଳ int64_t    i12         = row_mapping.i2;
                ଅଚଳ int64_t    i1          = id;
                ଅଚଳ int64_t    i2          = i12;

                auto *  src1_col = quant_a_buffer + (i11 * nbw1 + i12 * nbw2);
                ଦଶମିକ * c_blk    = (ଦଶମିକ *) ((ଅକ୍ଷର *) dst->data + (i1 * nb1 + i2 * nb2));

                uint8_t * a_row = src1_col;
                uint8_t * b_col = reinterpret_cast<uint8_t *>(tcm_buffer);
                ଯଦି ((nbw1 + per_nb_cols_wsize) <= tcm_buffer_size) {
                    a_row = (uint8_t *) tcm_buffer;
                    b_col = reinterpret_cast<uint8_t *>(tcm_buffer) + nbw1;
                }
                uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? b_col : nullptr;

                ଯଦି (ith % 2 == 0) {
                    spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(src0_cur), per_nb_cols_wsize);

                    ଯଦି (a_row != src1_col) {
                        spacemit_kernels::rvv::memcpy1d(a_row, src1_col, nbw1);
                    }
                }

                spine_barrier_wait(cur_barrier);

                ଯଦି (ith % 2 != 0) {
                    ଯଦି (a_row != src1_col) {
                        spacemit_kernels::rvv::memcpy1d(a_row, src1_col, nbw1);
                    }

                    spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(src0_cur), per_nb_cols_wsize);
                }

                int64_t nb_real = std::min(ne01, NB_COLS);
                ପାଇଁ (int64_t ni = 0; ni < ne01; ni += NB_COLS) {
                    ଯଦି (ith % 2 != 0) {
                        spine_barrier_wait(cur_barrier);
                    }

                    gemm_kernel(b_blk_len, a_row, b_col, b_col_zp, c_blk + ni, 1, nb_real, b_k_blks, ne01);

                    ଯଦି (ith % 2 == 0) {
                        spine_barrier_wait(cur_barrier);
                    }

                    ଅଚଳ int64_t next_ni = ni + NB_COLS;
                    ଯଦି (next_ni < ne01) {
                        nb_real = std::min(ne01 - next_ni, NB_COLS);
                        spacemit_kernels::rvv::memcpy1d(
                            b_col, reinterpret_cast<uint8_t *>(src0_cur) + next_ni * row_stride_b, per_nb_cols_wsize);
                    }
                }
            }
        } ଅନ୍ୟଥା {
            ପାଇଁ (int64_t valid_id = ith_es; valid_id < valid_ep_ଗଣନା_t; valid_id += nth_es) {
                ଅଚଳ int64_t cur_a = valid_matrix_row_counts[valid_id];
                ଅଚଳ int64_t cne1  = matrix_row_counts[cur_a];

                int64_t src1_cur_start = 0;
                int64_t src1_cur_end   = cne1;

                int64_t src0_cur_start = (ith_n * ne01) / nth_n;
                int64_t src0_cur_end   = MIN(((ith_n + 1) * ne01) / nth_n, ne01);

                ଯଦି (src1_cur_start >= src1_cur_end || src0_cur_start >= src0_cur_end) {
                    ଜାରି;
                }

                src0_cur_start =
                    (src0_cur_start % NB_COLS) ? src0_cur_start + NB_COLS - (src0_cur_start % NB_COLS) : src0_cur_start;
                src0_cur_end =
                    (src0_cur_end % NB_COLS) ? src0_cur_end + NB_COLS - (src0_cur_end % NB_COLS) : src0_cur_end;

                auto *    src0_cur = (uint8_t *) src0->data + cur_a * expert_b_stride + src0_cur_start * row_stride_b;
                uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? src0_cur : nullptr;

                size_t extra_tcm_buffer_size = tcm_buffer_size;
                ଶୂନ୍ୟ * extra_tcm_buffer      = tcm_buffer;
                ଯଦି (tcm_buffer != nullptr && (src1_cur_end - src1_cur_start) >= 4 &&
                    (src0_cur_end - src0_cur_start) * row_stride_b <= tcm_buffer_size) {
                    spacemit_kernels::rvv::memcpy1d(tcm_buffer, src0_cur,
                                                    (src0_cur_end - src0_cur_start) * row_stride_b);
                    src0_cur = reinterpret_cast<uint8_t *>(tcm_buffer);
                    b_col_zp = block_type_has_zp<BLOC_TYPE>() ? src0_cur : nullptr;
                    extra_tcm_buffer_size -= (src0_cur_end - src0_cur_start) * row_stride_b;
                    extra_tcm_buffer = reinterpret_cast<ଶୂନ୍ୟ *>(reinterpret_cast<uintptr_t>(tcm_buffer) +
                                                                (src0_cur_end - src0_cur_start) * row_stride_b);
                }

                ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir1 = src1_cur_start;

                ଯଦି (extra_tcm_buffer_size >= nbw1 && extra_tcm_buffer != nullptr) {
                    int64_t quant_a_tile_size = extra_tcm_buffer_size / nbw1;
                    କର {
                        quant_a_tile_size = MIN(quant_a_tile_size, src1_cur_end - ir1);

                        uint8_t * quant_a_tile_buffer = reinterpret_cast<uint8_t *>(extra_tcm_buffer);

                        ପୂର୍ଣ୍ଣସଂଖ୍ୟା iir1 = ir1;
                        ପାଇଁ (; iir1 < (ir1 + quant_a_tile_size); ++iir1) {
                            mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, iir1);

                            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା id = row_mapping.i1;  // selected expert index

                            ଅଚଳ int64_t i11 = id % ne11;
                            ଅଚଳ int64_t i12 = row_mapping.i2;  // row index in src1

                            auto * src1_col = quant_a_buffer + (i11 * nbw1 + i12 * nbw2);
                            spacemit_kernels::rvv::memcpy1d(quant_a_tile_buffer, src1_col, nbw1);
                            quant_a_tile_buffer = quant_a_tile_buffer + nbw1;
                        }

                        quant_a_tile_buffer = reinterpret_cast<uint8_t *>(extra_tcm_buffer);
                        iir1                = ir1;

                        ଯଦି (moe_gemm_kernel_m2 != nullptr) {
                            ପାଇଁ (; iir1 < (ir1 + quant_a_tile_size - 1); iir1 += 2, quant_a_tile_buffer += 2 * nbw1) {
                                mmid_row_mapping row_mapping_0 = MMID_MATRIX_ROW(cur_a, iir1);
                                mmid_row_mapping row_mapping_1 = MMID_MATRIX_ROW(cur_a, iir1 + 1);

                                src_workspaces[0] = quant_a_tile_buffer;
                                src_workspaces[1] = quant_a_tile_buffer + nbw1;

                                dst_workspaces[0] =
                                    (ଦଶମିକ *) ((ଅକ୍ଷର *) dst->data + (row_mapping_0.i1 * nb1 + row_mapping_0.i2 * nb2)) +
                                    src0_cur_start;
                                dst_workspaces[1] = (ଦଶମିକ *) ((ଅକ୍ଷର *) dst->data +
                                                               ((row_mapping_1.i1) * nb1 + (row_mapping_1.i2) * nb2)) +
                                                    src0_cur_start;
                                moe_gemm_kernel_m2(b_blk_len, src_workspaces.data(), src0_cur, b_col_zp,
                                                   dst_workspaces.data(), 1, src0_cur_end - src0_cur_start, b_k_blks,
                                                   ne01);
                            }
                        }

                        ପାଇଁ (; iir1 < (ir1 + quant_a_tile_size); iir1++, quant_a_tile_buffer += nbw1) {
                            mmid_row_mapping row_mapping_0 = MMID_MATRIX_ROW(cur_a, iir1);

                            gemm_kernel(
                                b_blk_len, quant_a_tile_buffer, src0_cur, b_col_zp,
                                (ଦଶମିକ *) ((ଅକ୍ଷର *) dst->data + (row_mapping_0.i1 * nb1 + row_mapping_0.i2 * nb2)) +
                                    src0_cur_start,
                                1, src0_cur_end - src0_cur_start, b_k_blks, ne01);
                        }

                        ir1 += quant_a_tile_size;
                    } ଯେତେବେଳେ (ir1 < src1_cur_end);
                } ଅନ୍ୟଥା {
                    ଯଦି (moe_gemm_kernel_m2 != nullptr) {
                        ପାଇଁ (; ir1 < src1_cur_end - 1; ir1 += 2) {
                            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା iir1 = 0; iir1 < 2; ++iir1) {
                                mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, ir1 + iir1);

                                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା id = row_mapping.i1;  // selected expert index

                                ଅଚଳ int64_t i11 = id % ne11;
                                ଅଚଳ int64_t i12 = row_mapping.i2;  // row index in src1

                                ଅଚଳ int64_t i1 = id;               // selected expert index
                                ଅଚଳ int64_t i2 = i12;              // row

                                src_workspaces[iir1] = quant_a_buffer + (i11 * nbw1 + i12 * nbw2);

                                dst_workspaces[iir1] =
                                    (ଦଶମିକ *) ((ଅକ୍ଷର *) dst->data + (i1 * nb1 + i2 * nb2)) + src0_cur_start;
                            }

                            moe_gemm_kernel_m2(b_blk_len, src_workspaces.data(), src0_cur, b_col_zp,
                                               dst_workspaces.data(), 1, src0_cur_end - src0_cur_start, b_k_blks, ne01);
                        }
                    }

                    ପାଇଁ (; ir1 < src1_cur_end; ir1++) {
                        mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, ir1);

                        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା id = row_mapping.i1;  // selected expert index

                        ଅଚଳ int64_t i11 = id % ne11;
                        ଅଚଳ int64_t i12 = row_mapping.i2;  // row index in src1

                        ଅଚଳ int64_t i1 = id;               // selected expert index
                        ଅଚଳ int64_t i2 = i12;              // row

                        auto * src1_col = quant_a_buffer + (i11 * nbw1 + i12 * nbw2);

                        gemm_kernel(b_blk_len, src1_col, src0_cur, b_col_zp,
                                    (ଦଶମିକ *) ((ଅକ୍ଷର *) dst->data + (i1 * nb1 + i2 * nb2)) + src0_cur_start, 1,
                                    src0_cur_end - src0_cur_start, b_k_blks, ne01);
                    }
                }
            }
        }
#undef MMID_MATRIX_ROW
    }

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା repack(ggml_tensor * t, ଅଚଳ ଶୂନ୍ୟ * data, size_t data_size) override {
        GGML_LOG_DEBUG("%s: repack tensor %s with %s_%dx%d\n", __func__, t->name, ggml_type_name(t->type),
                       (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) NB_COLS, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) INTER_SIZE);
        ଫେରନ୍ତୁ ggml::cpu::riscv64_spacemit::repack<BLOC_TYPE, INTER_SIZE, NB_COLS>(t, data, data_size);
    }
};

ଶ୍ରେଣୀ tensor_traits_common : ସର୍ବସାଧାରଣ tensor_traits_base {
    ତାର୍କିକ work_size(ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_threads, ଅଚଳ ggml_tensor * op, size_t & size) override {
        ବାଛ (op->op) {
            ସ୍ଥିତି GGML_OP_FLASH_ATTN_EXT:
                {
                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା     n_tasks = n_threads;
                    ଅଚଳ int64_t neq2    = op->src[0]->ne[2];  // number of query heads
                    ଅଚଳ int64_t DK      = op->src[1]->ne[0];
                    ଅଚଳ int64_t DV      = op->src[2]->ne[0];  // DV

                    // Tiled flash attention scratch (tile sizes defined in common.h)
                    // Per-thread: Q_q + KQ + mask + VKQ32 + V32 + K_f32 + padding
                    size_t prefill = ଆକାର(ଦଶମିକ) *
                                     (GGML_FA_TILE_Q * DK + 2 * GGML_FA_TILE_Q * GGML_FA_TILE_KV + GGML_FA_TILE_Q * DV +
                                      GGML_FA_TILE_KV * DV + GGML_FA_TILE_KV * DK) *
                                     n_tasks;

                    // Decode path: n_kv_chunks = n_tasks (one chunk per thread)
                    // Per-thread: VKQ accmulator (DV), partial M, partial S + intra-thread scratch for V, Q and VKQ
                    size_t n_chunks = n_tasks;
                    size_t decode   = ଆକାର(ଦଶମିକ) * (neq2 * n_chunks * (2 + DV) + n_tasks * (DK + 2 * DV));

                    size = MAX(prefill, decode);
                }
                ଫେରନ୍ତୁ ସତ୍ୟ;
            ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                ଭାଙ୍ଗ;
        }
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ତାର୍କିକ compute_forward(ggml_compute_params * params, ggml_tensor * op) override {
        ବାଛ (op->op) {
            ସ୍ଥିତି GGML_OP_NORM:
                ବାଛ (op->src[0]->type) {
                    ସ୍ଥିତି GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_norm_f32(params, op);
                        ଫେରନ୍ତୁ ସତ୍ୟ;
                    ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                        GGML_ABORT("fatal error");
                }
            ସ୍ଥିତି GGML_OP_RMS_NORM:
                ବାଛ (op->src[0]->type) {
                    ସ୍ଥିତି GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_rms_norm_f32(params, op);
                        ଫେରନ୍ତୁ ସତ୍ୟ;
                    ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                        GGML_ABORT("fatal error");
                }
            ସ୍ଥିତି GGML_OP_ADD:
                ବାଛ (op->src[0]->type) {
                    ସ୍ଥିତି GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_ଦ୍ୱିମିକ<GGML_OP_ADD, ଦଶମିକ>(params, op);
                        ଫେରନ୍ତୁ ସତ୍ୟ;
                    ସ୍ଥିତି GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_ଦ୍ୱିମିକ<GGML_OP_ADD, _Float16>(params, op);
                        ଫେରନ୍ତୁ ସତ୍ୟ;
                    ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                        ggml_compute_forward_add(params, op);
                        ଫେରନ୍ତୁ ସତ୍ୟ;
                }
            ସ୍ଥିତି GGML_OP_SUB:
                ବାଛ (op->src[0]->type) {
                    ସ୍ଥିତି GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_ଦ୍ୱିମିକ<GGML_OP_SUB, ଦଶମିକ>(params, op);
                        ଫେରନ୍ତୁ ସତ୍ୟ;
                    ସ୍ଥିତି GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_ଦ୍ୱିମିକ<GGML_OP_SUB, _Float16>(params, op);
                        ଫେରନ୍ତୁ ସତ୍ୟ;
                    ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                        ggml_compute_forward_sub(params, op);
                        ଫେରନ୍ତୁ ସତ୍ୟ;
                }
            ସ୍ଥିତି GGML_OP_MUL:
                ବାଛ (op->src[0]->type) {
                    ସ୍ଥିତି GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_ଦ୍ୱିମିକ<GGML_OP_MUL, ଦଶମିକ>(params, op);
                        ଫେରନ୍ତୁ ସତ୍ୟ;
                    ସ୍ଥିତି GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_ଦ୍ୱିମିକ<GGML_OP_MUL, _Float16>(params, op);
                        ଫେରନ୍ତୁ ସତ୍ୟ;
                    ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                        ggml_compute_forward_mul(params, op);
                        ଫେରନ୍ତୁ ସତ୍ୟ;
                }
            ସ୍ଥିତି GGML_OP_DIV:
                ବାଛ (op->src[0]->type) {
                    ସ୍ଥିତି GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_ଦ୍ୱିମିକ<GGML_OP_DIV, ଦଶମିକ>(params, op);
                        ଫେରନ୍ତୁ ସତ୍ୟ;
                    ସ୍ଥିତି GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_ଦ୍ୱିମିକ<GGML_OP_DIV, _Float16>(params, op);
                        ଫେରନ୍ତୁ ସତ୍ୟ;
                    ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                        ggml_compute_forward_div(params, op);
                        ଫେରନ୍ତୁ ସତ୍ୟ;
                }
            ସ୍ଥିତି GGML_OP_FLASH_ATTN_EXT:
                forward_flash_attn_ext_f16(params, op);
                ଫେରନ୍ତୁ ସତ୍ୟ;
            ସ୍ଥିତି GGML_OP_CONT:
                {
                    ଅଚଳ ggml_tensor * src0 = op->src[0];
                    ଯଦି (op->type == src0->type && op->nb[0] != src0->nb[0] && op->nb[0] == src0->nb[1] &&
                        op->ne[3] * op->ne[2] * op->nb[2] == src0->ne[3] * src0->ne[2] * src0->nb[2]) {
                        spacemit_kernels::rvv::forward_cont_with_permute(params, op);
                    } ଅନ୍ୟଥା {
                        ggml_compute_forward_cont(params, op);
                    }
                    ଫେରନ୍ତୁ ସତ୍ୟ;
                }
            ସ୍ଥିତି GGML_OP_CPY:
                {
                    ଅଚଳ ggml_tensor * src0 = op->src[0];
                    ଯଦି (op->type == src0->type && op->nb[0] == src0->nb[1] && src0->nb[0] != src0->nb[1] &&
                        ggml_nelements(src0) == ggml_nelements(op)) {
                        spacemit_kernels::rvv::forward_cpy_with_permute(params, op);
                    } ଅନ୍ୟଥା {
                        ggml_compute_forward_cpy(params, op);
                    }
                    ଫେରନ୍ତୁ ସତ୍ୟ;
                }
            ସ୍ଥିତି GGML_OP_REPEAT:
                {
                    ଅଚଳ ତାର୍କିକ rows_equal         = ggml_nrows(op->src[0]) == ggml_nrows(op);
                    ଅଚଳ ତାର୍କିକ broadcast_or_equal = op->src[0]->ne[0] == 1 || op->src[0]->ne[0] == op->ne[0];

                    ଯଦି (rows_equal && broadcast_or_equal) {
                        ବାଛ (op->src[0]->type) {
                            ସ୍ଥିତି GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_repeat_nrows<int32_t>(params, op);
                                ଫେରନ୍ତୁ ସତ୍ୟ;
                            ସ୍ଥିତି GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_repeat_nrows<int16_t>(params, op);
                                ଫେରନ୍ତୁ ସତ୍ୟ;
                            ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                                ଭାଙ୍ଗ;
                        }
                    }

                    ଯଦି (op->src[0]->ne[1] == 1 && op->src[0]->ne[0] == op->ne[0]) {
                        ବାଛ (op->src[0]->type) {
                            ସ୍ଥିତି GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_repeat_dim1<int32_t>(params, op);
                                ଫେରନ୍ତୁ ସତ୍ୟ;
                            ସ୍ଥିତି GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_repeat_dim1<int16_t>(params, op);
                                ଫେରନ୍ତୁ ସତ୍ୟ;
                            ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                                ଭାଙ୍ଗ;
                        }
                    }

                    ggml_compute_forward_repeat(params, op);
                }
                ଫେରନ୍ତୁ ସତ୍ୟ;
            ସ୍ଥିତି GGML_OP_SUM_ROWS:
                {
                    ଯଦି (op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32) {
                        spacemit_kernels::rvv::forward_sum_rows<ଦଶମିକ>(params, op);
                    } ଅନ୍ୟଥା {
                        ggml_compute_forward_sum_rows(params, op);
                    }
                }
                ଫେରନ୍ତୁ ସତ୍ୟ;
            ସ୍ଥିତି GGML_OP_GET_ROWS:
                {
                    ଯଦି (op->src[0]->type == op->type) {
                        ବାଛ (op->src[0]->type) {
                            ସ୍ଥିତି GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_get_rows<int32_t>(params, op);
                                ଫେରନ୍ତୁ ସତ୍ୟ;
                            ସ୍ଥିତି GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_get_rows<int16_t>(params, op);
                                ଫେରନ୍ତୁ ସତ୍ୟ;
                            ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                                ଭାଙ୍ଗ;
                        }
                    }

                    ggml_compute_forward_get_rows(params, op);
                }
                ଫେରନ୍ତୁ ସତ୍ୟ;
            ସ୍ଥିତି GGML_OP_CONCAT:
                {
                    ଅଚଳ int32_t dim = ggml_get_op_params_i32(op, 0);
                    ଯଦି (dim == 0 && op->type == op->src[0]->type) {
                        ବାଛ (op->src[0]->type) {
                            ସ୍ଥିତି GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_concat<int32_t>(params, op);
                                ଫେରନ୍ତୁ ସତ୍ୟ;
                            ସ୍ଥିତି GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_concat<int16_t>(params, op);
                                ଫେରନ୍ତୁ ସତ୍ୟ;
                            ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                                ଭାଙ୍ଗ;
                        }
                    }

                    ggml_compute_forward_concat(params, op);
                }
                ଫେରନ୍ତୁ ସତ୍ୟ;
            // TODO For GGML_OP_GATED_DELTA_NET
            // case GGML_OP_GATED_DELTA_NET:
            //     return true;
            ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                ଭାଙ୍ଗ;
        }
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଶୂନ୍ୟ forward_flash_attn_ext_f16(ଅଚଳ ggml_compute_params * params, ggml_tensor * dst) {
        ଅଚଳ ggml_tensor * q = dst->src[0];
        ଅଚଳ ggml_tensor * k = dst->src[1];
        ଅଚଳ ggml_tensor * v = dst->src[2];

        GGML_TENSOR_LOCALS(int64_t, neq, q, ne)
        GGML_TENSOR_LOCALS(size_t, nbq, q, nb)
        GGML_TENSOR_LOCALS(int64_t, nek, k, ne)
        GGML_TENSOR_LOCALS(size_t, nbk, k, nb)
        GGML_TENSOR_LOCALS(int64_t, nev, v, ne)
        GGML_TENSOR_LOCALS(size_t, nbv, v, nb)
        GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
        GGML_TENSOR_LOCALS(size_t, nb, dst, nb)

        ଅଚଳ int64_t DK = nek0;
        ଅଚଳ int64_t DV = nev0;

        ଅଚଳ ତାର୍କିକ supported_prec  = (dst->op_params[3] == GGML_PREC_F32 || dst->op_params[3] == GGML_PREC_DEFAULT);
        ଅଚଳ ତାର୍କିକ supported_types = (q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16);
        ଅଚଳ ତାର୍କିକ supported_shape = (DK > 0 && DK <= 128 && DV > 0 && DV <= 128);
        ଅଚଳ ତାର୍କିକ supported_vlen  = (__riscv_vlenb() == 128);

        ଯଦି (!(supported_prec && supported_types && supported_shape && supported_vlen)) {
            ggml_compute_forward_flash_attn_ext(params, dst);
            ଫେରନ୍ତୁ;
        }

        // total rows in q
        ଅଚଳ int64_t nr = neq1 * neq2 * neq3;

        // rows per thread
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

        ସ୍ଥିର constexpr int64_t Q_TILE_SZ = ggml_fa_tile_config::Q;
        ଅଚଳ ତାର୍କିକ               use_tiled = !params->use_ref && (neq1 >= Q_TILE_SZ);

        // 4x chunks per thread
        // int     nth_scaled = nth * 4;
        // int64_t chunk_size = (nr + nth_scaled - 1) / nth_scaled;
        // int64_t nchunk     = (nr + chunk_size - 1) / chunk_size;

        // if (nth == 1 || nchunk < nth) {
        //     nchunk = nth;
        // }

        int64_t nchunk = nth;

        ଯଦି (ith == 0) {
            // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
            ggml_threadpool_chunk_set(params->threadpool, nth);
        }

        ggml_barrier(params->threadpool);

        // The number of elements in each chunk
        ଅଚଳ int64_t dr = (nr + nchunk - 1) / nchunk;

        // The first chunk comes from our thread_id, the rest will get auto-assigned.
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା current_chunk = ith;

        ଯେତେବେଳେ (current_chunk < nchunk) {
            ଅଚଳ int64_t ir0 = dr * current_chunk;
            ଅଚଳ int64_t ir1 = MIN(ir0 + dr, nr);

            ଯଦି (use_tiled) {
                spacemit_kernels::rvv::forward_flash_attn_ext_f16_tiled_vlen1024_vf16(
                    params, dst, ir0, ir1, ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer,
                    ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer_size);
            } ଅନ୍ୟଥା {
                spacemit_kernels::rvv::forward_flash_attn_ext_f16_one_chunk_vlen1024_vf16(
                    params, dst, ir0, ir1, ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer,
                    ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer_size);
            }

            current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
        }
    }

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା repack(ggml_tensor * t, ଅଚଳ ଶୂନ୍ୟ * data, size_t data_size) override {
        memcpy(t->data, data, data_size);
        ଫେରନ୍ତୁ 0;
    }
};

// Impl By IME1
ସ୍ଥିର ଅଚଳ tensor_traits<block_q4_0, 32, 16>  q4_0_16x32_q8_0;
ସ୍ଥିର ଅଚଳ tensor_traits<block_q4_1, 32, 16>  q4_1_16x32_q8_0;
ସ୍ଥିର ଅଚଳ tensor_traits<block_q4_K, 32, 16>  q4_k_16x32_q8_0;
ସ୍ଥିର ଅଚଳ tensor_traits<block_q8_0, 32, 16>  q8_0_16x32_q8_0;
// Impl By IME2
ସ୍ଥିର ଅଚଳ tensor_traits<block_q2_K, 256, 32> q2_k_32x256_q8_0;
ସ୍ଥିର ଅଚଳ tensor_traits<block_q3_K, 256, 32> q3_k_32x256_q8_0;
ସ୍ଥିର ଅଚଳ tensor_traits<block_q4_0, 32, 32>  q4_0_32x32_q8_0;
ସ୍ଥିର ଅଚଳ tensor_traits<block_q4_1, 32, 32>  q4_1_32x32_q8_0;
ସ୍ଥିର ଅଚଳ tensor_traits<block_q4_0, 256, 32> q4_0_32x256_q8_0;
ସ୍ଥିର ଅଚଳ tensor_traits<block_q4_1, 256, 32> q4_1_32x256_q8_0;
ସ୍ଥିର ଅଚଳ tensor_traits<block_q4_K, 32, 32>  q4_k_32x32_q8_0;
ସ୍ଥିର ଅଚଳ tensor_traits<block_q6_K, 32, 32>  q6_k_32x32_q8_0;
ସ୍ଥିର ଅଚଳ tensor_traits<block_q8_0, 32, 32>  q8_0_32x32_q8_0;
ସ୍ଥିର ଅଚଳ tensor_traits<block_mxfp4, 32, 32> mxfp4_32x32_q8_0;
ସ୍ଥିର ଅଚଳ tensor_traits<block_q5_K, 32, 32>  q5_k_32x32_q8_0;
ସ୍ଥିର ଅଚଳ tensor_traits<block_q5_1, 32, 32>  q5_1_32x32_q8_0;
ସ୍ଥିର ଅଚଳ tensor_traits<block_q5_0, 32, 32>  q5_0_32x32_q8_0;
// Impl By RVV
ସ୍ଥିର ଅଚଳ tensor_traits_common               rvv_impl;

}  // namespace ggml::cpu::riscv64_spacemit

ସ୍ଥିର ଅଚଳ ggml::cpu::tensor_traits * ggml_riscv64_spacemit_get_optimal_repack_type(ଅଚଳ ggml_tensor * cur) {
    ବାଛ (cur->type) {
        ସ୍ଥିତି GGML_TYPE_Q2_K:
            {
#ଯଦି defined(RISCV64_SPACEMIT_IME2)
                ଯଦି (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    ଫେରନ୍ତୁ &ggml::cpu::riscv64_spacemit::q2_k_32x256_q8_0;
                }
#endif
            }
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q3_K:
            {
#ଯଦି defined(RISCV64_SPACEMIT_IME2)
                ଯଦି (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    ଫେରନ୍ତୁ &ggml::cpu::riscv64_spacemit::q3_k_32x256_q8_0;
                }
#endif
            }
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q4_0:
            {
#ଯଦି defined(RISCV64_SPACEMIT_IME2)
                ଯଦି (cur->ne[1] % 32 == 0 && cur->ne[0] % 256 == 0 &&
                    (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    ଫେରନ୍ତୁ &ggml::cpu::riscv64_spacemit::q4_0_32x256_q8_0;
                }

                ଯଦି (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    ଫେରନ୍ତୁ &ggml::cpu::riscv64_spacemit::q4_0_32x32_q8_0;
                }
#endif

#ଯଦି defined(RISCV64_SPACEMIT_IME1)
                ଯଦି (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    ଫେରନ୍ତୁ &ggml::cpu::riscv64_spacemit::q4_0_16x32_q8_0;
                }
#endif
            }
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q4_1:
            {
#ଯଦି defined(RISCV64_SPACEMIT_IME2)
                // TODO
                // if (cur->ne[1] % 32 == 0 && cur->ne[0] % 256 == 0 &&
                //     (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                //     return &ggml::cpu::riscv64_spacemit::q4_1_32x256_q8_0;
                // }

                ଯଦି (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    ଫେରନ୍ତୁ &ggml::cpu::riscv64_spacemit::q4_1_32x32_q8_0;
                }
#endif

#ଯଦି defined(RISCV64_SPACEMIT_IME1)
                ଯଦି (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    ଫେରନ୍ତୁ &ggml::cpu::riscv64_spacemit::q4_1_16x32_q8_0;
                }
#endif
            }
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q4_K:
            {
#ଯଦି defined(RISCV64_SPACEMIT_IME2)
                ଯଦି (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    ଫେରନ୍ତୁ &ggml::cpu::riscv64_spacemit::q4_k_32x32_q8_0;
                }
#endif

#ଯଦି defined(RISCV64_SPACEMIT_IME1)
                ଯଦି (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    ଫେରନ୍ତୁ &ggml::cpu::riscv64_spacemit::q4_k_16x32_q8_0;
                }
#endif
            }
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q6_K:
            {
#ଯଦି defined(RISCV64_SPACEMIT_IME2)
                ଯଦି ((ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    ଫେରନ୍ତୁ &ggml::cpu::riscv64_spacemit::q6_k_32x32_q8_0;
                }
#endif
            }
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q8_0:
            {
#ଯଦି defined(RISCV64_SPACEMIT_IME2)
                ଯଦି ((ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    ଫେରନ୍ତୁ &ggml::cpu::riscv64_spacemit::q8_0_32x32_q8_0;
                }
#endif

#ଯଦି defined(RISCV64_SPACEMIT_IME1)
                ଯଦି (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    ଫେରନ୍ତୁ &ggml::cpu::riscv64_spacemit::q8_0_16x32_q8_0;
                }
#endif
            }
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_MXFP4:
            {
#ଯଦି defined(RISCV64_SPACEMIT_IME2)
                // TODO
                // if (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                //     return &ggml::cpu::riscv64_spacemit::mxfp4_32x32_q8_0;
                // }
#endif
            }
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q5_K:
            {
#ଯଦି defined(RISCV64_SPACEMIT_IME2)
                ଯଦି (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    ଫେରନ୍ତୁ &ggml::cpu::riscv64_spacemit::q5_k_32x32_q8_0;
                }
#endif
            }
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q5_1:
            {
#ଯଦି defined(RISCV64_SPACEMIT_IME2)
                ଯଦି (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    ଫେରନ୍ତୁ &ggml::cpu::riscv64_spacemit::q5_1_32x32_q8_0;
                }
#endif
            }
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q5_0:
            {
#ଯଦି defined(RISCV64_SPACEMIT_IME2)
                ଯଦି (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    ଫେରନ୍ତୁ &ggml::cpu::riscv64_spacemit::q5_0_32x32_q8_0;
                }
#endif
            }
            ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            ଭାଙ୍ଗ;
    }

    ଫେରନ୍ତୁ nullptr;
}

ସ୍ଥିର ଗଣନା ggml_status ggml_backend_riscv64_spacemit_buffer_init_tensor(ggml_backend_buffer_t buffer,
                                                                         ggml_tensor *         tensor) {
    tensor->extra =
        (ଶୂନ୍ୟ *) const_cast<ggml::cpu::tensor_traits *>(ggml_riscv64_spacemit_get_optimal_repack_type(tensor));

    GGML_UNUSED(buffer);

    ଫେରନ୍ତୁ GGML_STATUS_SUCCESS;
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_riscv64_spacemit_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);

    ଶୂନ୍ୟ * base = buffer->context;
    ଯଦି (base == nullptr) {
        ଫେରନ୍ତୁ;
    }

    ggml::cpu::riscv64_spacemit::spine_mem_pool_free(base);
}

ସ୍ଥିର ଶୂନ୍ୟ * ggml_backend_riscv64_spacemit_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);

    ଶୂନ୍ୟ * base = buffer->context;
    GGML_ASSERT(base != nullptr);
    ଫେରନ୍ତୁ base;
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_riscv64_spacemit_buffer_memset_tensor(ggml_backend_buffer_t buffer,
                                                               ggml_tensor *         tensor,
                                                               uint8_t               value,
                                                               size_t                offset,
                                                               size_t                size) {
    GGML_ASSERT(tensor);
    memset((ଅକ୍ଷର *) tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_riscv64_spacemit_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);

    ଶୂନ୍ୟ * base = buffer->context;
    GGML_ASSERT(base != nullptr);
    memset(base, value, buffer->size);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_riscv64_spacemit_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                            ggml_tensor *         tensor,
                                                            ଅଚଳ ଶୂନ୍ୟ *          data,
                                                            size_t                offset,
                                                            size_t                size) {
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    auto tensor_traits = (ggml::cpu::riscv64_spacemit::tensor_traits_base *) tensor->extra;
    ଯଦି (tensor_traits) {
        auto OK = tensor_traits->repack(tensor, data, size);
        GGML_ASSERT(OK == 0);
    }

    GGML_UNUSED(buffer);
}

ସ୍ଥିର ଅଚଳ ggml_backend_buffer_i ggml_backend_riscv64_spacemit_buffer_i = {
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

ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * ggml_backend_cpu_riscv64_spacemit_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ଫେରନ୍ତୁ "CPU_RISCV64_SPACEMIT";

    GGML_UNUSED(buft);
}

ସ୍ଥିର ggml_backend_buffer_t ggml_backend_cpu_riscv64_spacemit_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                                                                        size_t size) {
    ଶୂନ୍ୟ * base = ggml::cpu::riscv64_spacemit::spine_mem_pool_alloc(size, 64);
    ଯଦି (base == nullptr) {
        ଫେରନ୍ତୁ nullptr;
    }

    ଫେରନ୍ତୁ ggml_backend_buffer_init(buft, ggml_backend_riscv64_spacemit_buffer_i, base, size);
}

ସ୍ଥିର size_t ggml_backend_cpu_riscv64_spacemit_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ଫେରନ୍ତୁ 64;

    GGML_UNUSED(buft);
}

ସ୍ଥିର size_t ggml_backend_cpu_riscv64_spacemit_nbytes(ggml_backend_buffer_type_t buft, ଅଚଳ ggml_tensor * tensor) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < GGML_MAX_DIMS; ++i) {
        ଯଦି (tensor->ne[i] <= 0) {
            ଫେରନ୍ତୁ 0;
        }
    }

    GGML_UNUSED(buft);

    ଅଚଳ auto plain_nbytes = [&]() {
        size_t total = ggml_type_size(tensor->type);
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < GGML_MAX_DIMS; ++i) {
            total += (tensor->ne[i] - 1) * tensor->nb[i];
        }
        ଫେରନ୍ତୁ total;
    };

    ଅଚଳ size_t blck_size = ggml_blck_size(tensor->type);
    ଯଦି (blck_size == 1) {
        ଫେରନ୍ତୁ plain_nbytes();
    }

    ଅଚଳ size_t row_nbytes = tensor->ne[0] * tensor->nb[0] / blck_size;

    ଅଚଳ auto add_strided_nbytes = [&](size_t total, size_t src_block_size, size_t dst_block_size) {
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 1; i < GGML_MAX_DIMS; ++i) {
            total += (tensor->ne[i] - 1) * (tensor->nb[i] / src_block_size) * dst_block_size;
        }
        ଫେରନ୍ତୁ total;
    };

    ଅଚଳ auto remap_block_nbytes = [&](size_t src_block_size, size_t dst_block_size, int64_t padded_rows = 0) {
        GGML_ASSERT(row_nbytes % src_block_size == 0);

        size_t total =
            add_strided_nbytes((row_nbytes / src_block_size) * dst_block_size, src_block_size, dst_block_size);

        ଯଦି (padded_rows > 0 && tensor->ne[1] % padded_rows != 0) {
            total += (padded_rows - tensor->ne[1] % padded_rows) * (tensor->nb[1] / src_block_size) * dst_block_size;
        }

        ଫେରନ୍ତୁ total;
    };

    size_t nbytes = row_nbytes;
    ବାଛ (tensor->type) {
        ସ୍ଥିତି GGML_TYPE_Q4_K:
            nbytes = remap_block_nbytes(ଆକାର(block_q4_K), ଆକାର(block_q4_1) * 8);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q6_K:
            nbytes = remap_block_nbytes(ଆକାର(block_q6_K), ଆକାର(block_q8_0) * 8, 32);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q8_0:
            nbytes = remap_block_nbytes(ଆକାର(block_q8_0), ଆକାର(block_q8_0), 32);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q2_K:
            nbytes = remap_block_nbytes(ଆକାର(block_q2_K), ଆକାର(spacemit_kernels::nrow_block_q2_k<1>));
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q3_K:
            nbytes = remap_block_nbytes(ଆକାର(block_q3_K), ଆକାର(spacemit_kernels::nrow_block_q3_k<1>));
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_MXFP4:
            nbytes = remap_block_nbytes(ଆକାର(block_mxfp4), ଆକାର(spacemit_kernels::nrow_block_mxfp4<1>));
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q5_K:
            nbytes = remap_block_nbytes(ଆକାର(block_q5_K), ଆକାର(spacemit_kernels::nrow_block_q5_1<1>) * 8);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q5_1:
            nbytes = remap_block_nbytes(ଆକାର(block_q5_1), ଆକାର(spacemit_kernels::nrow_block_q5_1<1>));
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q5_0:
            nbytes = remap_block_nbytes(ଆକାର(block_q5_0), ଆକାର(spacemit_kernels::nrow_block_q5_0<1>));
            ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            nbytes = add_strided_nbytes(row_nbytes, 1, 1);
            ଭାଙ୍ଗ;
    }

    ଫେରନ୍ତୁ nbytes;
}

ନାମାକାଶ ggml::cpu::riscv64_spacemit {

ଶ୍ରେଣୀ extra_buffer_type : ggml::cpu::extra_buffer_type {
    ତାର୍କିକ supports_op(ggml_backend_dev_t, ଅଚଳ ggml_tensor * op) override {
        ବାଛ (op->op) {
            ସ୍ଥିତି GGML_OP_MUL_MAT:
                ଯଦି (op->src[0]->buffer && (ggml_n_dims(op->src[0]) == 2) &&
                    op->src[0]->buffer->buft == ggml_backend_cpu_riscv64_spacemit_buffer_type() &&
                    ggml_riscv64_spacemit_get_optimal_repack_type(op->src[0])) {
                    ଯଦି (op->src[1]->buffer && !ggml_backend_buft_is_host(op->src[1]->buffer->buft)) {
                        ଫେରନ୍ତୁ ମିଥ୍ୟା;
                    }
                    ଯଦି (op->src[1]->type == GGML_TYPE_F32) {
                        ଫେରନ୍ତୁ ସତ୍ୟ;
                    }
                }
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_OP_MUL_MAT_ID:
                ଯଦି (op->src[0]->buffer && (ggml_n_dims(op->src[0]) == 3) &&
                    op->src[0]->buffer->buft == ggml_backend_cpu_riscv64_spacemit_buffer_type() &&
                    ggml_riscv64_spacemit_get_optimal_repack_type(op->src[0])) {
                    ଯଦି (op->src[1]->buffer && !ggml_backend_buft_is_host(op->src[1]->buffer->buft)) {
                        ଫେରନ୍ତୁ ମିଥ୍ୟା;
                    }
                    ଯଦି (op->src[1]->type == GGML_TYPE_F32) {
                        ଫେରନ୍ତୁ ସତ୍ୟ;
                    }
                }
                ଭାଙ୍ଗ;
            ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                // GGML_ABORT("fatal error");
                ଭାଙ୍ଗ;
        }
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(ଅଚଳ ggml_tensor * op) override {
        ବାଛ (op->op) {
            ସ୍ଥିତି GGML_OP_MUL_MAT:
            ସ୍ଥିତି GGML_OP_MUL_MAT_ID:
                ଯଦି (op->src[0]->buffer && op->src[0]->buffer->buft == ggml_backend_cpu_riscv64_spacemit_buffer_type()) {
                    ଫେରନ୍ତୁ (ggml::cpu::tensor_traits *) op->src[0]->extra;
                }
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_OP_NORM:
            ସ୍ଥିତି GGML_OP_RMS_NORM:
            ସ୍ଥିତି GGML_OP_ADD:
            ସ୍ଥିତି GGML_OP_SUB:
            ସ୍ଥିତି GGML_OP_MUL:
            ସ୍ଥିତି GGML_OP_DIV:
            ସ୍ଥିତି GGML_OP_FLASH_ATTN_EXT:
            ସ୍ଥିତି GGML_OP_CONT:
            ସ୍ଥିତି GGML_OP_CPY:
            ସ୍ଥିତି GGML_OP_REPEAT:
            ସ୍ଥିତି GGML_OP_SUM_ROWS:
            ସ୍ଥିତି GGML_OP_GET_ROWS:
            ସ୍ଥିତି GGML_OP_CONCAT:
                // case GGML_OP_GATED_DELTA_NET:
                ଫେରନ୍ତୁ (ggml::cpu::tensor_traits *) (&ggml::cpu::riscv64_spacemit::rvv_impl);
            ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                // GGML_ABORT("fatal error");
                ଭାଙ୍ଗ;
        }

        ଫେରନ୍ତୁ nullptr;
    }
};

}  // namespace ggml::cpu::riscv64_spacemit

ggml_backend_buffer_type_t ggml_backend_cpu_riscv64_spacemit_buffer_type(ଶୂନ୍ୟ) {
    ସ୍ଥିର ggml_backend_buffer_type ggml_backend_cpu_buffer_type_riscv64_spacemit = {
  /* .iface    = */
        {
         /* .get_name           = */ ggml_backend_cpu_riscv64_spacemit_buffer_type_get_name,
         /* .alloc_buffer       = */ ggml_backend_cpu_riscv64_spacemit_buffer_type_alloc_buffer,
         /* .alloc_buffer_n     = */ ଖାଲି,
         /* .get_alignment      = */ ggml_backend_cpu_riscv64_spacemit_buffer_type_get_alignment,
         /* .get_max_size       = */ nullptr,
         /* .get_alloc_size     = */ ggml_backend_cpu_riscv64_spacemit_nbytes,
         /* .get_alloc_size_n   = */ ଖାଲି,
         /* .is_host            = */ nullptr,
         },
 /* .device  = */
        ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
 /* .context = */
        ନୂତନ ggml::cpu::riscv64_spacemit::extra_buffer_type(),
    };

    ଫେରନ୍ତୁ &ggml_backend_cpu_buffer_type_riscv64_spacemit;
}

ବାହ୍ୟ "C" {
ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା bind_ai_thread() {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା  fd, bytes;
    ଅକ୍ଷର str[32];

    fd = open("/proc/set_ai_thread", O_WRONLY);
    ଯଦି (fd < 0) {
        GGML_LOG_ERROR("try open /proc/set_ai_thread failed\n");
        ଫେରନ୍ତୁ -1;
    }

    snprintf(str, 16, "%d", 0);
    bytes = ଲିଖ(fd, str, strlen(str));
    ଯଦି (bytes < 0) {
        GGML_LOG_ERROR("try write /proc/set_ai_thread failed\n");
        close(fd);
        ଫେରନ୍ତୁ -1;
    }

    close(fd);
    ଫେରନ୍ତୁ 0;
}

ଶୂନ୍ୟ ggml_backend_cpu_riscv64_spacemit_set_numa_thread_affinity(ପୂର୍ଣ୍ଣସଂଖ୍ୟା thread_n) {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା cpu_id = sched_getcpu();
    ଯଦି (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2 &&
        !((1 << cpu_id) & ggml::cpu::riscv64_spacemit::global_spine_env_info.cpu_mask)) {
        GGML_PRINT_DEBUG("bind_ai_thread for thread %d, pid %d\n", thread_n, getpid());
        bind_ai_thread();
    }

    ଯଦି (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_tcm &&
        ggml::cpu::riscv64_spacemit::tls_context.cpu_id == -1) {
        CPU_ZERO(&(ggml::cpu::riscv64_spacemit::tls_context.cpuset));
        pthread_t    main_thread     = pthread_self();
        ଅଚଳ auto & perfer_core_ids = ggml::cpu::riscv64_spacemit::global_spine_env_info.perfer_core_ids;
        ଯଦି (thread_n < 0 || static_cast<size_t>(thread_n) >= perfer_core_ids.size()) {
            GGML_ABORT("thread_n %d exceeds perfer_core_ids size %zu\n", thread_n, perfer_core_ids.size());
        }
        auto perfer_cpu_id = perfer_core_ids[static_cast<size_t>(thread_n)];
        CPU_SET(perfer_cpu_id, &(ggml::cpu::riscv64_spacemit::tls_context.cpuset));
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା s =
            pthread_setaffinity_np(main_thread, ଆକାର(cpu_set_t), &(ggml::cpu::riscv64_spacemit::tls_context.cpuset));
        ଯଦି (s != 0) {
            GGML_ABORT("set thread affinity error for thread_n %d, cpu_id %d\n", thread_n, perfer_cpu_id);
        }

        ପୂର୍ଣ୍ଣସଂଖ୍ୟା ai_cpu_id = perfer_cpu_id - ggml::cpu::riscv64_spacemit::global_spine_env_info.aicpu_id_offset;
        ggml::cpu::riscv64_spacemit::tls_context.cpu_id = ai_cpu_id;
        ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer =
            ggml::cpu::riscv64_spacemit::spine_mem_pool_tcm_mem_get(ai_cpu_id);
        ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer_size =
            ggml::cpu::riscv64_spacemit::global_spine_env_info.tcm_blk_size;
    }

    ଯଦି (ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer != nullptr) {
        ଶୂନ୍ୟ * rt =
            ggml::cpu::riscv64_spacemit::spine_mem_pool_tcm_mem_wait(ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        ଯଦି (rt == nullptr) {
            GGML_ABORT("wait tcm buffer failed for cpu_id: %d", ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        }
    }
}

ଶୂନ୍ୟ ggml_backend_cpu_riscv64_spacemit_clear_numa_thread_affinity_threaded(ପୂର୍ଣ୍ଣସଂଖ୍ୟା thread_n) {
    ଯଦି (ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer != nullptr) {
        auto rt = ggml::cpu::riscv64_spacemit::spine_mem_pool_tcm_mem_release(
            ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        ଯଦି (rt != 0) {
            GGML_ABORT("release tcm buffer failed for cpu_id: %d", ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        }
    }
}
}
