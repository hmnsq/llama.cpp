#define GGML_COMMON_IMPL_CPP
#define GGML_COMMON_DECL_CPP

#അന്തര്ഭാവ "ime.h"

#അന്തര്ഭാവ "binary-ops.h"
#അന്തര്ഭാവ "common.h"
#അന്തര്ഭാവ "ggml-backend-impl.h"
#അന്തര്ഭാവ "ggml-common.h"
#അന്തര്ഭാവ "ggml-cpu.h"
#അന്തര്ഭാവ "ime_env.h"
#അന്തര്ഭാവ "ime_kernels.h"
#അന്തര്ഭാവ "ops.h"
#അന്തര്ഭാവ "repack.h"
#അന്തര്ഭാവ "rvv_kernels.h"
#അന്തര്ഭാവ "spine_mem_pool.h"
#അന്തര്ഭാവ "traits.h"
#അന്തര്ഭാവ "vec.h"

#അന്തര്ഭാവ <fcntl.h>
#അന്തര്ഭാവ <sys/mman.h>
#അന്തര്ഭാവ <unistd.h>

#അന്തര്ഭാവ <algorithm>
#അന്തര്ഭാവ <atomic>
#അന്തര്ഭാവ <cassert>
#അന്തര്ഭാവ <cerrno>
#അന്തര്ഭാവ <cmath>
#അന്തര്ഭാവ <cstdio>  // for GGML_ASSERT
#അന്തര്ഭാവ <stdexcept>
#അന്തര്ഭാവ <thread>
// clang-format off
#എങ്കിൽ defined(__riscv)

#എങ്കിൽ !defined(__riscv_v) || !defined(__riscv_v_intrinsic)
#error "riscv v extension or v_intrinsic not enabled"
#അല്ലെങ്കിൽ
#അന്തര്ഭാവ <riscv_vector.h>
#endif

#എങ്കിൽ !defined(__riscv_zfh) || !defined(__riscv_zvfh)
#error "riscv zfh extension not enabled, GGML_RV_ZFH and GGML_RV_ZVFH must be defined to 1"
#endif

#എങ്കിൽ !defined(__riscv_zba)
#error "riscv zba extension not enabled, GGML_RV_ZBA must be defined to 1"
#endif

#എങ്കിൽ defined(RISCV64_SPACEMIT_IME1) || defined(RISCV64_SPACEMIT_IME2)
#അല്ലെങ്കിൽ
#error "RISCV64_SPACEMIT_IME1 or RISCV64_SPACEMIT_IME2 not defined"
#endif

#അല്ലെങ്കിൽ

#error "riscv not enabled in this build"

#endif

#എങ്കിൽ defined(__GNUC__)
#pragma GCC diagnostic ignored "-Woverlength-strings"
#pragma GCC diagnostic ignored "-Wcast-qual"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

// clang-format on

ബാഹ്യ "C" {
ബാഹ്യ ശൂന്യം ggml_threadpool_chunk_set(ഘടന ggml_threadpool * tp, പൂർണ്ണസംഖ്യ value);
ബാഹ്യ പൂർണ്ണസംഖ്യ  ggml_threadpool_chunk_add(ഘടന ggml_threadpool * tp, പൂർണ്ണസംഖ്യ value);
}

നാമാകാശ ggml::cpu::riscv64_spacemit {

ഘടന TLSContext {
    പൂർണ്ണസംഖ്യ       cpu_id{ -1 };
    cpu_set_t cpuset;
    ശൂന്യം *    tcm_buffer{ nullptr };
    size_t    tcm_buffer_size{ 0 };
};

thread_local TLSContext tls_context;

template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS> constexpr size_t get_repacked_block_type_size() {
    എങ്കിൽ constexpr (std::is_same_v<BLOC_TYPE, block_q6_K> || std::is_same_v<BLOC_TYPE, block_q8_0>) {
        തിരികെ വലുപ്പം(block_q8_0);
    } അല്ലെങ്കിൽ എങ്കിൽ constexpr (std::is_same_v<BLOC_TYPE, block_q4_0>) {
        തിരികെ വലുപ്പം(block_q4_0) * INTER_SIZE / QK4_0;
    } അല്ലെങ്കിൽ എങ്കിൽ constexpr (std::is_same_v<BLOC_TYPE, block_q4_1> || std::is_same_v<BLOC_TYPE, block_q4_K>) {
        തിരികെ (വലുപ്പം(block_q4_0) + വലുപ്പം(uint8_t)) * INTER_SIZE / QK4_1;
    } അല്ലെങ്കിൽ എങ്കിൽ constexpr (std::is_same_v<BLOC_TYPE, block_q2_K>) {
        തിരികെ വലുപ്പം(spacemit_kernels::nrow_block_q2_k<1>);
    } അല്ലെങ്കിൽ എങ്കിൽ constexpr (std::is_same_v<BLOC_TYPE, block_q3_K>) {
        തിരികെ വലുപ്പം(spacemit_kernels::nrow_block_q3_k<1>);
    } അല്ലെങ്കിൽ എങ്കിൽ constexpr (std::is_same_v<BLOC_TYPE, block_mxfp4>) {
        തിരികെ വലുപ്പം(spacemit_kernels::nrow_block_mxfp4<1>);
    } അല്ലെങ്കിൽ എങ്കിൽ constexpr (std::is_same_v<BLOC_TYPE, block_q5_1> || std::is_same_v<BLOC_TYPE, block_q5_K>) {
        തിരികെ വലുപ്പം(spacemit_kernels::nrow_block_q5_1<1>);
    } അല്ലെങ്കിൽ എങ്കിൽ constexpr (std::is_same_v<BLOC_TYPE, block_q5_0>) {
        തിരികെ വലുപ്പം(spacemit_kernels::nrow_block_q5_0<1>);
    } അല്ലെങ്കിൽ {
        പ്രതിജ്ഞാ(അസത്യം);
        തിരികെ 0;
    }
}

template <typename BLOC_TYPE> constexpr തർക്കം block_type_has_zp() {
    എങ്കിൽ constexpr (std::is_same_v<BLOC_TYPE, block_q6_K> || std::is_same_v<BLOC_TYPE, block_q8_0> ||
                  std::is_same_v<BLOC_TYPE, block_q3_K> || std::is_same_v<BLOC_TYPE, block_q4_0> ||
                  std::is_same_v<BLOC_TYPE, block_mxfp4> || std::is_same_v<BLOC_TYPE, block_q5_0>) {
        തിരികെ അസത്യം;
    } അല്ലെങ്കിൽ എങ്കിൽ constexpr (std::is_same_v<BLOC_TYPE, block_q4_1> || std::is_same_v<BLOC_TYPE, block_q4_K> ||
                         std::is_same_v<BLOC_TYPE, block_q2_K> || std::is_same_v<BLOC_TYPE, block_q5_1> ||
                         std::is_same_v<BLOC_TYPE, block_q5_K>) {
        തിരികെ സത്യം;
    } അല്ലെങ്കിൽ {
        പ്രതിജ്ഞാ(അസത്യം);
        തിരികെ അസത്യം;
    }
}

വർഗ്ഗം tensor_traits_base : പൊതു ggml::cpu::tensor_traits {
  പൊതു:
    മായാ പൂർണ്ണസംഖ്യ repack(ggml_tensor * t, സ്ഥിരം ശൂന്യം * data, size_t data_size) = 0;
};

template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS> വർഗ്ഗം tensor_traits : പൊതു tensor_traits_base {
    തർക്കം work_size(പൂർണ്ണസംഖ്യ /* n_threads */, സ്ഥിരം ggml_tensor * op, size_t & size) override {
        തിരഞ്ഞെടുപ്പ് (op->op) {
            അവസ്ഥ GGML_OP_MUL_MAT:
                {
                    int64_t src1_nelements = ggml_nelements(op->src[1]);

                    എങ്കിൽ constexpr (std::is_same_v<BLOC_TYPE, block_q2_K> || std::is_same_v<BLOC_TYPE, block_q3_K>) {
                        size =
                            spacemit_kernels::div_round_up(src1_nelements, QK_K) * spacemit_kernels::q8k_blk_size(QK_K);
                    } അല്ലെങ്കിൽ എങ്കിൽ constexpr (INTER_SIZE == QK4_0) {
                        size = spacemit_kernels::div_round_up(src1_nelements, QK4_0) *
                               spacemit_kernels::q8_blk_size(QK4_0, സത്യം);
                    } അല്ലെങ്കിൽ എങ്കിൽ constexpr (INTER_SIZE == 256) {
                        size = spacemit_kernels::div_round_up(src1_nelements, 256) *
                               spacemit_kernels::q8_hp_blk_size(256, സത്യം, സത്യം);
                    } അല്ലെങ്കിൽ {
                        GGML_ABORT("unsupported block type");
                    }

                    size = GGML_PAD(size, വലുപ്പം(int64_t));

                    തിരികെ സത്യം;
                }
            അവസ്ഥ GGML_OP_MUL_MAT_ID:
                {
                    int64_t src1_nelements = ggml_nelements(op->src[1]);

                    എങ്കിൽ constexpr (std::is_same_v<BLOC_TYPE, block_q2_K> || std::is_same_v<BLOC_TYPE, block_q3_K>) {
                        size =
                            spacemit_kernels::div_round_up(src1_nelements, QK_K) * spacemit_kernels::q8k_blk_size(QK_K);
                    } അല്ലെങ്കിൽ എങ്കിൽ constexpr (INTER_SIZE == QK4_0) {
                        size = spacemit_kernels::div_round_up(src1_nelements, QK4_0) *
                               spacemit_kernels::q8_blk_size(QK4_0, സത്യം);
                    } അല്ലെങ്കിൽ എങ്കിൽ constexpr (INTER_SIZE == 256) {
                        size = spacemit_kernels::div_round_up(src1_nelements, 256) *
                               spacemit_kernels::q8_hp_blk_size(256, സത്യം, സത്യം);
                    } അല്ലെങ്കിൽ {
                        GGML_ABORT("unsupported block type");
                    }

                    size = GGML_PAD(size, വലുപ്പം(int64_t));

                    സ്ഥിരം int64_t ne02 = op->src[0]->ne[2];  // n_as, n_expert
                    സ്ഥിരം int64_t ne12 = op->src[1]->ne[2];  // n_tokens

                    സ്ഥിരം size_t sizeof_mmid_row_mapping = വലുപ്പം(int64_t);
                    size += sizeof_mmid_row_mapping * ne02 * (ne12 + 1) + (ne02 + 1) * വലുപ്പം(int64_t);

                    size = GGML_PAD(size, വലുപ്പം(int64_t));

                    തിരികെ സത്യം;
                }
            സ്ഥിരസ്ഥിതി:
                // GGML_ABORT("fatal error");
                നിർത്തുക;
        }
        തിരികെ അസത്യം;
    }

    തർക്കം compute_forward(ggml_compute_params * params, ggml_tensor * op) override {
        തിരഞ്ഞെടുപ്പ് (op->op) {
            അവസ്ഥ GGML_OP_MUL_MAT:
                തിരഞ്ഞെടുപ്പ് (op->src[0]->type) {
                    അവസ്ഥ GGML_TYPE_Q2_K:
                    അവസ്ഥ GGML_TYPE_Q3_K:
                    അവസ്ഥ GGML_TYPE_Q4_0:
                    അവസ്ഥ GGML_TYPE_Q4_1:
                    അവസ്ഥ GGML_TYPE_Q4_K:
                    അവസ്ഥ GGML_TYPE_Q6_K:
                    അവസ്ഥ GGML_TYPE_Q8_0:
                    അവസ്ഥ GGML_TYPE_Q5_0:
                    അവസ്ഥ GGML_TYPE_Q5_1:
                    അവസ്ഥ GGML_TYPE_Q5_K:
                        //case GGML_TYPE_MXFP4:
                        forward_mul_mat(params, op);
                        തിരികെ സത്യം;
                    സ്ഥിരസ്ഥിതി:
                        // GGML_ABORT("fatal error: unsupported type for src0 in MUL_MAT");
                        തിരികെ അസത്യം;
                }
                നിർത്തുക;
            അവസ്ഥ GGML_OP_MUL_MAT_ID:
                തിരഞ്ഞെടുപ്പ് (op->src[0]->type) {
                    അവസ്ഥ GGML_TYPE_Q2_K:
                    അവസ്ഥ GGML_TYPE_Q3_K:
                    അവസ്ഥ GGML_TYPE_Q4_0:
                    അവസ്ഥ GGML_TYPE_Q4_1:
                    അവസ്ഥ GGML_TYPE_Q4_K:
                    അവസ്ഥ GGML_TYPE_Q6_K:
                    അവസ്ഥ GGML_TYPE_Q8_0:
                    അവസ്ഥ GGML_TYPE_Q5_0:
                    അവസ്ഥ GGML_TYPE_Q5_1:
                    അവസ്ഥ GGML_TYPE_Q5_K:
                        //case GGML_TYPE_MXFP4:
                        forward_mul_mat_id(params, op);
                        തിരികെ സത്യം;
                    സ്ഥിരസ്ഥിതി:
                        // GGML_ABORT("fatal error: unsupported type for src0 in MUL_MAT_ID");
                        തിരികെ അസത്യം;
                }
                നിർത്തുക;
            സ്ഥിരസ്ഥിതി:
                // GGML_ABORT("fatal error");
                നിർത്തുക;
        }
        തിരികെ അസത്യം;
    }

    ശൂന്യം forward_mul_mat(ggml_compute_params * params, ggml_tensor * op) {
        constexpr size_t a_blk_len = INTER_SIZE;
        constexpr size_t b_blk_len = INTER_SIZE;

        സ്ഥിരം ggml_tensor * src0 = op->src[0];
        സ്ഥിരം ggml_tensor * src1 = op->src[1];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_ദ്വിമാന_OP_LOCALS

        പൂർണ്ണസംഖ്യ ith = params->ith;
        പൂർണ്ണസംഖ്യ nth = params->nth;

        [[maybe_unused]] സ്ഥിരം എണ്ണൽ ggml_type type = src0->type;

        ശൂന്യം *        w_data  = (ശൂന്യം *) src0->data;
        സ്ഥിരം ദശാംശം * feature = (സ്ഥിരം ദശാംശം *) src1->data;
        ദശാംശം *       output  = (ദശാംശം *) dst->data;

        സ്ഥിരം int64_t gemm_m = ne11 * ne12 * ne13;
        സ്ഥിരം int64_t gemm_k = ne10;
        സ്ഥിരം int64_t gemm_n = ne01;

        spacemit_kernels::quantize_a_row_def       quantize_a_row_i8;
        spacemit_kernels::quantize_a_row_def       quantize_a_4row_i8;
        spacemit_kernels::gemm_kernel_quantize_def gemm_kernel;
        തർക്കം                                       set_kernel_impl = അസത്യം;

        int64_t block_stride_a = spacemit_kernels::q8_blk_size(a_blk_len);

#എങ്കിൽ defined(RISCV64_SPACEMIT_IME2)
        എങ്കിൽ (!set_kernel_impl && (global_spine_env_info.use_ime2)) {
            quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8;
            quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8;
            block_stride_a     = spacemit_kernels::q8_blk_size(a_blk_len, സത്യം);

            എങ്കിൽ constexpr (std::is_same_v<BLOC_TYPE, block_q6_K> || std::is_same_v<BLOC_TYPE, block_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i8;
                set_kernel_impl = സത്യം;
            } അല്ലെങ്കിൽ എങ്കിൽ constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> || std::is_same_v<BLOC_TYPE, block_q4_1> ||
                                 std::is_same_v<BLOC_TYPE, block_q4_K>) {
                എങ്കിൽ constexpr (INTER_SIZE == 256) {
                    gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i4_hp;
                    quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8_hp;
                    quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8_hp;
                    block_stride_a     = spacemit_kernels::q8_hp_blk_size(a_blk_len, സത്യം, സത്യം);
                    set_kernel_impl    = സത്യം;
                } അല്ലെങ്കിൽ {
                    gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i4;
                    quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8;
                    quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8;
                    block_stride_a     = spacemit_kernels::q8_blk_size(a_blk_len, സത്യം);
                    set_kernel_impl    = സത്യം;
                }
            } അല്ലെങ്കിൽ എങ്കിൽ constexpr (std::is_same_v<BLOC_TYPE, block_q2_K>) {
                quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8k;
                quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8k;
                block_stride_a     = spacemit_kernels::q8k_blk_size(a_blk_len);

                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i2k;
                set_kernel_impl = സത്യം;
            } അല്ലെങ്കിൽ എങ്കിൽ constexpr (std::is_same_v<BLOC_TYPE, block_q3_K>) {
                quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8k;
                quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8k;
                block_stride_a     = spacemit_kernels::q8k_blk_size(a_blk_len);

                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i3k;
                set_kernel_impl = സത്യം;
            } അല്ലെങ്കിൽ എങ്കിൽ constexpr (std::is_same_v<BLOC_TYPE, block_mxfp4>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8mxfp4;
                set_kernel_impl = സത്യം;
            } അല്ലെങ്കിൽ എങ്കിൽ constexpr (std::is_same_v<BLOC_TYPE, block_q5_1> || std::is_same_v<BLOC_TYPE, block_q5_K> ||
                                 std::is_same_v<BLOC_TYPE, block_q5_0>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i5;
                set_kernel_impl = സത്യം;
            }
        }
#endif

#എങ്കിൽ defined(RISCV64_SPACEMIT_IME1)
        എങ്കിൽ (!set_kernel_impl && (global_spine_env_info.use_ime1)) {
            quantize_a_row_i8  = spacemit_kernels::ime1::quantize_a_row_i8;
            quantize_a_4row_i8 = spacemit_kernels::ime1::quantize_a_4row_i8;

            എങ്കിൽ constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> || std::is_same_v<BLOC_TYPE, block_q4_1> ||
                          std::is_same_v<BLOC_TYPE, block_q4_K>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i4;
                set_kernel_impl = സത്യം;
            } അല്ലെങ്കിൽ എങ്കിൽ constexpr (std::is_same_v<BLOC_TYPE, block_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i8;
                set_kernel_impl = സത്യം;
            }
        }
#endif
        എങ്കിൽ (!set_kernel_impl) {
            GGML_ABORT("no kernel implementation found for the block type");
        }

        സ്ഥിരം int64_t a_k_blks = spacemit_kernels::div_round_up(gemm_k, a_blk_len);
        സ്ഥിരം int64_t b_k_blks = spacemit_kernels::div_round_up(gemm_k, b_blk_len);

        സ്ഥിരം int64_t row_stride_a        = a_k_blks * block_stride_a;
        സ്ഥിരം int64_t gemm_workspace_size = GGML_PAD(gemm_m * row_stride_a, alignof(int64_t));

        എങ്കിൽ (ith == 0 && params->wsize < gemm_workspace_size) {
            GGML_ABORT("wsize less than gemm_workspace_size");
        }

        uintptr_t ws_ptr = reinterpret_cast<uintptr_t>(params->wdata);

        ശൂന്യം *        tcm_buffer      = ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer;
        സ്ഥിരം int64_t tcm_buffer_size = ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer_size;

        auto * quant_a_buffer = reinterpret_cast<uint8_t *>(ws_ptr);

        constexpr int64_t row_align = 4;
        സ്ഥിരം int64_t     row_blks  = spacemit_kernels::div_round_up(gemm_m, row_align);

        സ്ഥിരം int64_t row_stride_b      = b_k_blks * get_repacked_block_type_size<BLOC_TYPE, INTER_SIZE, NB_COLS>();
        സ്ഥിരം int64_t per_mb_rows_wsize = row_align * row_stride_a;
        സ്ഥിരം int64_t per_nb_cols_wsize = NB_COLS * row_stride_b;

        സ്ഥിരം int64_t barrier_idx = static_cast<int64_t>(ith / 2);

        GGML_ASSERT(global_spine_env_info.init_barrier != nullptr);
        GGML_ASSERT(barrier_idx < spine_init_barrier_എണ്ണം);
        spine_barrier_t * cur_barrier = &global_spine_env_info.init_barrier[barrier_idx];

        എങ്കിൽ (gemm_m == 1) {
            പൂർണ്ണസംഖ്യ task_per_thread = spacemit_kernels::div_round_up(a_k_blks, nth);
            പൂർണ്ണസംഖ്യ a_blk_start     = ith * task_per_thread;
            പൂർണ്ണസംഖ്യ a_blk_end       = std::min(a_blk_start + task_per_thread, (പൂർണ്ണസംഖ്യ) a_k_blks);
            എങ്കിൽ (a_blk_start < a_blk_end) {
                quantize_a_row_i8(a_blk_len, feature + a_blk_start * a_blk_len, (a_blk_end - a_blk_start) * a_blk_len,
                                  quant_a_buffer + a_blk_start * block_stride_a);
            }
        } അല്ലെങ്കിൽ {
            പൂർണ്ണസംഖ്യ task_per_thread = spacemit_kernels::div_round_up(row_blks, nth);
            പൂർണ്ണസംഖ്യ m_row_blk_start = ith * task_per_thread;
            പൂർണ്ണസംഖ്യ m_row_blk_end   = std::min(m_row_blk_start + task_per_thread, (പൂർണ്ണസംഖ്യ) row_blks);
            വേണ്ടി (പൂർണ്ണസംഖ്യ m_row_blk = m_row_blk_start; m_row_blk < m_row_blk_end; m_row_blk++) {
                പൂർണ്ണസംഖ്യ m_idx             = m_row_blk * row_align;
                പൂർണ്ണസംഖ്യ rows_tobe_handled = (gemm_m - m_idx) > row_align ? row_align : (gemm_m - m_idx);

                എങ്കിൽ (rows_tobe_handled == row_align && quantize_a_4row_i8 != nullptr) {
                    സ്ഥിരം ദശാംശം * a_row_ptr       = feature + m_idx * gemm_k;
                    auto *        quant_a_row_ptr = quant_a_buffer + m_idx * row_stride_a;
                    quantize_a_4row_i8(a_blk_len, a_row_ptr, gemm_k, quant_a_row_ptr);
                } അല്ലെങ്കിൽ {
                    ആകുമ്പോൾ (rows_tobe_handled) {
                        സ്ഥിരം ദശാംശം * a_row_ptr       = feature + m_idx * gemm_k;
                        auto *        quant_a_row_ptr = quant_a_buffer + m_idx * row_stride_a;
                        quantize_a_row_i8(a_blk_len, a_row_ptr, gemm_k, quant_a_row_ptr);
                        rows_tobe_handled -= 1;
                        m_idx += 1;
                    }
                }
            }
        }

        ggml_barrier(params->threadpool);

        സ്ഥിരം int64_t gemm_m_stride     = gemm_n / gemm_m > 64 ? gemm_m : 16;
        സ്ഥിരം int64_t gemm_m_blocked    = spacemit_kernels::div_round_up(gemm_m, gemm_m_stride);
        സ്ഥിരം int64_t max_gemm_n_stride = spacemit_kernels::div_round_up(gemm_n * gemm_m_blocked, nth);

        int64_t gemm_n_stride = gemm_n;
        എങ്കിൽ (max_gemm_n_stride < gemm_n) {
            gemm_n_stride =
                std::min(gemm_n_stride, spacemit_kernels::div_round_up(max_gemm_n_stride, NB_COLS) * NB_COLS);
        }

        എങ്കിൽ (gemm_n_stride == gemm_n && tcm_buffer != nullptr && per_mb_rows_wsize <= tcm_buffer_size) {
            വേണ്ടി (int64_t m_start = ith * row_align; m_start < gemm_m; m_start += row_align * nth) {
                uint8_t * b_col    = reinterpret_cast<uint8_t *>(w_data);
                uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? b_col : nullptr;

                int64_t m_row_real = std::min(gemm_m - m_start, row_align);

                spacemit_kernels::rvv::memcpy1d(tcm_buffer, quant_a_buffer + m_start * row_stride_a,
                                                m_row_real * row_stride_a);

                int64_t n_blk_real = 0;
                വേണ്ടി (int64_t ni = 0; ni < gemm_n; ni += n_blk_real, b_col += n_blk_real * row_stride_b) {
                    n_blk_real = std::min(gemm_n - ni, (int64_t) NB_COLS);

                    uint8_t * a_row_ptr = (uint8_t *) tcm_buffer;
                    ദശാംശം *   c_blk     = output + m_start * gemm_n + ni;

                    int32_t rows_remaining = m_row_real;

                    ആകുമ്പോൾ (rows_remaining > 0) {
                        auto rows_handled = gemm_kernel(b_blk_len, a_row_ptr, b_col, b_col_zp, c_blk, rows_remaining,
                                                        n_blk_real, b_k_blks, gemm_n);

                        c_blk += rows_handled * gemm_n;
                        a_row_ptr += rows_handled * row_stride_a;

                        rows_remaining -= rows_handled;
                    }
                }
            }
        } അല്ലെങ്കിൽ എങ്കിൽ (tcm_buffer != nullptr && per_nb_cols_wsize <= tcm_buffer_size) {
            uint8_t * a_row = quant_a_buffer;
            uint8_t * b_col = reinterpret_cast<uint8_t *>(tcm_buffer);
            എങ്കിൽ ((gemm_workspace_size + per_nb_cols_wsize) <= tcm_buffer_size) {
                a_row = (uint8_t *) tcm_buffer;
                b_col = reinterpret_cast<uint8_t *>(tcm_buffer) + gemm_workspace_size;
            }
            uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? b_col : nullptr;

            int64_t ni      = ith * NB_COLS;
            int64_t nb_real = std::min(gemm_n - ni, NB_COLS);

            എങ്കിൽ (ith % 2 == 0 && nb_real > 0) {
                spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(w_data) + ni * row_stride_b,
                                                nb_real * row_stride_b);
                എങ്കിൽ (a_row != quant_a_buffer) {
                    spacemit_kernels::rvv::memcpy1d(a_row, quant_a_buffer, gemm_workspace_size);
                }
            }

            spine_barrier_wait(cur_barrier);

            എങ്കിൽ (ith % 2 != 0 && nb_real > 0) {
                എങ്കിൽ (a_row != quant_a_buffer) {
                    spacemit_kernels::rvv::memcpy1d(a_row, quant_a_buffer, gemm_workspace_size);
                }
                spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(w_data) + ni * row_stride_b,
                                                nb_real * row_stride_b);
            }

            വേണ്ടി (; ni < gemm_n; ni += NB_COLS * nth) {
                int64_t rows_remaining = gemm_m;
                ദശാംശം * c_blk          = output + ni;
                auto *  a_row_cur      = a_row;

                എങ്കിൽ (ith % 2 != 0) {
                    spine_barrier_wait(cur_barrier);
                }

                ആകുമ്പോൾ (rows_remaining > 0) {
                    auto rows_handled = gemm_kernel(b_blk_len, a_row_cur, b_col, b_col_zp, c_blk, rows_remaining,
                                                    nb_real, b_k_blks, gemm_n);

                    c_blk += rows_handled * gemm_n;
                    a_row_cur += rows_handled * row_stride_a;

                    rows_remaining -= rows_handled;
                }

                എങ്കിൽ (ith % 2 == 0) {
                    spine_barrier_wait(cur_barrier);
                }

                സ്ഥിരം int64_t next_ni = ni + NB_COLS * nth;
                എങ്കിൽ (next_ni < gemm_n) {
                    nb_real = std::min(gemm_n - next_ni, NB_COLS);
                    spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(w_data) + next_ni * row_stride_b,
                                                    nb_real * row_stride_b);
                }
            }
        } അല്ലെങ്കിൽ {
            സ്ഥിരം int64_t task_എണ്ണം_m = spacemit_kernels::div_round_up(gemm_m, gemm_m_stride);
            സ്ഥിരം int64_t task_എണ്ണം_n = spacemit_kernels::div_round_up(gemm_n, gemm_n_stride);

            int64_t task_എണ്ണം      = task_എണ്ണം_m * task_എണ്ണം_n;
            int64_t task_per_thread = (task_എണ്ണം + nth - 1) / nth;
            int64_t start           = ith * task_per_thread;
            int64_t end             = std::min((ith + 1) * task_per_thread, task_എണ്ണം);
            വേണ്ടി (int64_t compute_idx = start; compute_idx < end; compute_idx++) {
                സ്ഥിരം auto tid_n = compute_idx / task_എണ്ണം_m;
                സ്ഥിരം auto tid_m = compute_idx % task_എണ്ണം_m;

                സ്ഥിരം int64_t m_start = tid_m * gemm_m_stride;
                സ്ഥിരം int64_t m_എണ്ണം = std::min(gemm_m - m_start, (int64_t) gemm_m_stride);

                സ്ഥിരം int64_t n_start = tid_n * gemm_n_stride;
                സ്ഥിരം int64_t n_എണ്ണം = std::min(gemm_n - n_start, (int64_t) gemm_n_stride);

                സ്ഥിരം int64_t n_blk = m_എണ്ണം == 1 ? n_എണ്ണം : NB_COLS;

                uint8_t * b_col    = reinterpret_cast<uint8_t *>(w_data) + n_start * row_stride_b;
                uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? b_col : nullptr;

                int64_t n_blk_real = 0;
                വേണ്ടി (int64_t ni = 0; ni < n_എണ്ണം; ni += n_blk_real, b_col += n_blk_real * row_stride_b) {
                    n_blk_real = std::min(n_എണ്ണം - ni, n_blk);

                    uint8_t * a_row = quant_a_buffer + m_start * row_stride_a;

                    ദശാംശം * c_blk = output + m_start * gemm_n + n_start + ni;

                    int64_t rows_remaining = m_എണ്ണം;

                    uint8_t * b_col_cur    = b_col;
                    uint8_t * b_col_zp_cur = b_col_zp;

                    ആകുമ്പോൾ (rows_remaining > 0) {
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

    ശൂന്യം forward_mul_mat_id(ggml_compute_params * params, ggml_tensor * op) {
        constexpr size_t a_blk_len = INTER_SIZE;
        constexpr size_t b_blk_len = INTER_SIZE;

        സ്ഥിരം ggml_tensor * src0 = op->src[0];
        സ്ഥിരം ggml_tensor * src1 = op->src[1];
        സ്ഥിരം ggml_tensor * ids  = op->src[2];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_ദ്വിമാന_OP_LOCALS

        പൂർണ്ണസംഖ്യ ith = params->ith;
        പൂർണ്ണസംഖ്യ nth = params->nth;

        // row groups
        സ്ഥിരം പൂർണ്ണസംഖ്യ n_ids = ids->ne[0];  // n_expert_used
        സ്ഥിരം പൂർണ്ണസംഖ്യ n_as  = ne02;        // n_expert

        ഘടന mmid_row_mapping {
            int32_t i1;
            int32_t i2;
        };

        spacemit_kernels::quantize_a_row_def           quantize_a_row_i8;
        spacemit_kernels::gemm_kernel_quantize_def     gemm_kernel;
        spacemit_kernels::moe_gemm_kernel_quantize_def moe_gemm_kernel_m2;
        തർക്കം                                           set_kernel_impl = അസത്യം;
        size_t                                         block_stride_a  = spacemit_kernels::q8_blk_size(QK4_0);

#എങ്കിൽ defined(RISCV64_SPACEMIT_IME2)
        എങ്കിൽ (!set_kernel_impl && (global_spine_env_info.use_ime2)) {
            quantize_a_row_i8 = spacemit_kernels::rvv::quantize_a_row_i8;
            block_stride_a    = spacemit_kernels::q8_blk_size(QK4_0, സത്യം);

            എങ്കിൽ constexpr (std::is_same_v<BLOC_TYPE, block_q6_K> || std::is_same_v<BLOC_TYPE, block_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i8;
                set_kernel_impl = സത്യം;
            } അല്ലെങ്കിൽ എങ്കിൽ constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> || std::is_same_v<BLOC_TYPE, block_q4_1> ||
                                 std::is_same_v<BLOC_TYPE, block_q4_K>) {
                എങ്കിൽ constexpr (INTER_SIZE == 256) {
                    gemm_kernel       = spacemit_kernels::ime2::gemm_kernel_i8i4_hp;
                    quantize_a_row_i8 = spacemit_kernels::rvv::quantize_a_row_i8_hp;
                    block_stride_a    = spacemit_kernels::q8_hp_blk_size(a_blk_len, സത്യം, സത്യം);
                    set_kernel_impl   = സത്യം;
                } അല്ലെങ്കിൽ {
                    gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i4;
                    moe_gemm_kernel_m2 = spacemit_kernels::ime2::moe_m2_gemm_kernel_i8i4;
                    quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8;
                    block_stride_a     = spacemit_kernels::q8_blk_size(a_blk_len, സത്യം);
                    set_kernel_impl    = സത്യം;
                }
            } അല്ലെങ്കിൽ എങ്കിൽ constexpr (std::is_same_v<BLOC_TYPE, block_q2_K>) {
                quantize_a_row_i8 = spacemit_kernels::rvv::quantize_a_row_i8k;
                block_stride_a    = spacemit_kernels::q8k_blk_size(a_blk_len);
                gemm_kernel       = spacemit_kernels::ime2::gemm_kernel_i8i2k;
                set_kernel_impl   = സത്യം;
            } അല്ലെങ്കിൽ എങ്കിൽ constexpr (std::is_same_v<BLOC_TYPE, block_q3_K>) {
                quantize_a_row_i8 = spacemit_kernels::rvv::quantize_a_row_i8k;
                block_stride_a    = spacemit_kernels::q8k_blk_size(a_blk_len);
                gemm_kernel       = spacemit_kernels::ime2::gemm_kernel_i8i3k;
                set_kernel_impl   = സത്യം;
            } അല്ലെങ്കിൽ എങ്കിൽ constexpr (std::is_same_v<BLOC_TYPE, block_mxfp4>) {
                gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8mxfp4;
                moe_gemm_kernel_m2 = spacemit_kernels::ime2::moe_m2_gemm_kernel_i8mxfp4;
                set_kernel_impl    = സത്യം;
            } അല്ലെങ്കിൽ എങ്കിൽ constexpr (std::is_same_v<BLOC_TYPE, block_q5_1> || std::is_same_v<BLOC_TYPE, block_q5_K> ||
                                 std::is_same_v<BLOC_TYPE, block_q5_0>) {
                gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i5;
                moe_gemm_kernel_m2 = spacemit_kernels::ime2::moe_m2_gemm_kernel_i8i5;
                set_kernel_impl    = സത്യം;
            }
        }
#endif

#എങ്കിൽ defined(RISCV64_SPACEMIT_IME1)
        എങ്കിൽ (!set_kernel_impl && (global_spine_env_info.use_ime1)) {
            quantize_a_row_i8 = spacemit_kernels::ime1::quantize_a_row_i8;

            എങ്കിൽ constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> || std::is_same_v<BLOC_TYPE, block_q4_1> ||
                          std::is_same_v<BLOC_TYPE, block_q4_K>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i4;
                set_kernel_impl = സത്യം;
            } അല്ലെങ്കിൽ എങ്കിൽ constexpr (std::is_same_v<BLOC_TYPE, block_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i8;
                set_kernel_impl = സത്യം;
            }
        }
#endif
        എങ്കിൽ (!set_kernel_impl) {
            GGML_ABORT("no kernel implementation found for the block type");
        }

        സ്ഥിരം size_t a_k_blks = spacemit_kernels::div_round_up(ne10, a_blk_len);
        സ്ഥിരം size_t b_k_blks = spacemit_kernels::div_round_up(ne10, b_blk_len);

        സ്ഥിരം size_t nbw1                = a_k_blks * block_stride_a;
        സ്ഥിരം size_t nbw2                = ne11 * nbw1;
        സ്ഥിരം size_t nbw3                = nbw2 * ne12;
        സ്ഥിരം size_t gemm_workspace_size = GGML_PAD(nbw3, alignof(int64_t));

        സ്ഥിരം uintptr_t ws_ptr         = reinterpret_cast<uintptr_t>(params->wdata);
        auto *          quant_a_buffer = reinterpret_cast<uint8_t *>(ws_ptr);

        എങ്കിൽ (ne11 == 1) {
            വേണ്ടി (int64_t ii = ith; ii < ne12 * a_k_blks; ii += nth) {
                int64_t i12       = ii / a_k_blks;
                int64_t ak_blk_id = ii % a_k_blks;
                quantize_a_row_i8(a_blk_len, (ദശാംശം *) ((അക്ഷരം *) src1->data + i12 * nb12) + ak_blk_id * a_blk_len,
                                  a_blk_len, quant_a_buffer + i12 * nbw2 + ak_blk_id * block_stride_a);
            }
        } അല്ലെങ്കിൽ {
            വേണ്ടി (int64_t ii = ith; ii < ne12 * ne11; ii += nth) {
                int64_t i12 = ii / ne11;
                int64_t i11 = ii % ne11;
                quantize_a_row_i8(a_blk_len, (ദശാംശം *) ((അക്ഷരം *) src1->data + i12 * nb12 + i11 * nb11), ne10,
                                  quant_a_buffer + i12 * nbw2 + i11 * nbw1);
            }
        }

#define MMID_MATRIX_ROW(row_id, i1) matrix_rows[(row_id) *ne12 + (i1)]

        int64_t *          matrix_row_counts       = (int64_t *) (ws_ptr + gemm_workspace_size);
        int32_t *          valid_ep_എണ്ണം          = (int32_t *) (matrix_row_counts + n_as);
        int32_t *          valid_act_എണ്ണം         = (int32_t *) (valid_ep_എണ്ണം + 1);
        int64_t *          valid_matrix_row_counts = (int64_t *) (valid_act_എണ്ണം + 1);
        mmid_row_mapping * matrix_rows             = (mmid_row_mapping *) (valid_matrix_row_counts + n_as);

        എങ്കിൽ (ith == 0) {
            // initialize matrix_row_counts
            memset(matrix_row_counts, 0, n_as * വലുപ്പം(int64_t));

            // group rows by src0 matrix
            വേണ്ടി (int32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
                വേണ്ടി (int32_t id = 0; id < n_ids; ++id) {
                    സ്ഥിരം int32_t i02 =
                        *(സ്ഥിരം int32_t *) ((സ്ഥിരം അക്ഷരം *) ids->data + iid1 * ids->nb[1] + id * ids->nb[0]);

                    GGML_ASSERT(i02 >= 0 && i02 < n_as);

                    MMID_MATRIX_ROW(i02, matrix_row_counts[i02]) = { id, iid1 };
                    matrix_row_counts[i02] += 1;
                }
            }

            int32_t valid_ep_എണ്ണം_t  = 0;
            int32_t valid_act_എണ്ണം_t = 0;
            വേണ്ടി (പൂർണ്ണസംഖ്യ cur_a = 0; cur_a < n_as; ++cur_a) {
                സ്ഥിരം int64_t cne1 = matrix_row_counts[cur_a];
                എങ്കിൽ (cne1 == 0) {
                    തുടരുക;
                }
                valid_matrix_row_counts[valid_ep_എണ്ണം_t] = cur_a;
                valid_act_എണ്ണം_t += cne1;
                valid_ep_എണ്ണം_t += 1;
            }
            valid_ep_എണ്ണം[0]  = valid_ep_എണ്ണം_t;
            valid_act_എണ്ണം[0] = valid_act_എണ്ണം_t;
        }

        സ്ഥിരം int64_t barrier_idx = static_cast<int64_t>(ith / 2);

        GGML_ASSERT(global_spine_env_info.init_barrier != nullptr);
        GGML_ASSERT(barrier_idx < spine_init_barrier_എണ്ണം);
        spine_barrier_t * cur_barrier = &global_spine_env_info.init_barrier[barrier_idx];

        ggml_barrier(params->threadpool);

        സ്ഥിരം size_t row_stride_b      = b_k_blks * get_repacked_block_type_size<BLOC_TYPE, INTER_SIZE, NB_COLS>();
        സ്ഥിരം size_t expert_b_stride   = ne01 * row_stride_b;
        സ്ഥിരം size_t per_nb_cols_wsize = NB_COLS * row_stride_b;

        std::array<സ്ഥിരം uint8_t *, 2> src_workspaces;
        std::array<ദശാംശം *, 2>         dst_workspaces;

        auto *     tcm_buffer      = ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer;
        സ്ഥിരം auto tcm_buffer_size = ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer_size;

        സ്ഥിരം auto valid_ep_എണ്ണം_t  = valid_ep_എണ്ണം[0];
        സ്ഥിരം auto valid_act_എണ്ണം_t = valid_act_എണ്ണം[0];

        പൂർണ്ണസംഖ്യ nth_es = 1;
        പൂർണ്ണസംഖ്യ nth_n  = nth;

        പൂർണ്ണസംഖ്യ ith_es = ith % nth_es;
        പൂർണ്ണസംഖ്യ ith_n  = (ith / nth_es) % nth_n;

        എങ്കിൽ (valid_ep_എണ്ണം_t % nth == 0 && tcm_buffer != nullptr && valid_ep_എണ്ണം_t == n_as &&
            valid_act_എണ്ണം_t == n_as && per_nb_cols_wsize <= tcm_buffer_size) {
            വേണ്ടി (int64_t valid_id = ith; valid_id < valid_ep_എണ്ണം_t; valid_id += nth) {
                സ്ഥിരം int64_t cur_a = valid_matrix_row_counts[valid_id];

                auto * src0_cur = (uint8_t *) src0->data + cur_a * expert_b_stride;

                mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, 0);
                സ്ഥിരം പൂർണ്ണസംഖ്യ        id          = row_mapping.i1;
                സ്ഥിരം int64_t    i11         = id % ne11;
                സ്ഥിരം int64_t    i12         = row_mapping.i2;
                സ്ഥിരം int64_t    i1          = id;
                സ്ഥിരം int64_t    i2          = i12;

                auto *  src1_col = quant_a_buffer + (i11 * nbw1 + i12 * nbw2);
                ദശാംശം * c_blk    = (ദശാംശം *) ((അക്ഷരം *) dst->data + (i1 * nb1 + i2 * nb2));

                uint8_t * a_row = src1_col;
                uint8_t * b_col = reinterpret_cast<uint8_t *>(tcm_buffer);
                എങ്കിൽ ((nbw1 + per_nb_cols_wsize) <= tcm_buffer_size) {
                    a_row = (uint8_t *) tcm_buffer;
                    b_col = reinterpret_cast<uint8_t *>(tcm_buffer) + nbw1;
                }
                uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? b_col : nullptr;

                എങ്കിൽ (ith % 2 == 0) {
                    spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(src0_cur), per_nb_cols_wsize);

                    എങ്കിൽ (a_row != src1_col) {
                        spacemit_kernels::rvv::memcpy1d(a_row, src1_col, nbw1);
                    }
                }

                spine_barrier_wait(cur_barrier);

                എങ്കിൽ (ith % 2 != 0) {
                    എങ്കിൽ (a_row != src1_col) {
                        spacemit_kernels::rvv::memcpy1d(a_row, src1_col, nbw1);
                    }

                    spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(src0_cur), per_nb_cols_wsize);
                }

                int64_t nb_real = std::min(ne01, NB_COLS);
                വേണ്ടി (int64_t ni = 0; ni < ne01; ni += NB_COLS) {
                    എങ്കിൽ (ith % 2 != 0) {
                        spine_barrier_wait(cur_barrier);
                    }

                    gemm_kernel(b_blk_len, a_row, b_col, b_col_zp, c_blk + ni, 1, nb_real, b_k_blks, ne01);

                    എങ്കിൽ (ith % 2 == 0) {
                        spine_barrier_wait(cur_barrier);
                    }

                    സ്ഥിരം int64_t next_ni = ni + NB_COLS;
                    എങ്കിൽ (next_ni < ne01) {
                        nb_real = std::min(ne01 - next_ni, NB_COLS);
                        spacemit_kernels::rvv::memcpy1d(
                            b_col, reinterpret_cast<uint8_t *>(src0_cur) + next_ni * row_stride_b, per_nb_cols_wsize);
                    }
                }
            }
        } അല്ലെങ്കിൽ {
            വേണ്ടി (int64_t valid_id = ith_es; valid_id < valid_ep_എണ്ണം_t; valid_id += nth_es) {
                സ്ഥിരം int64_t cur_a = valid_matrix_row_counts[valid_id];
                സ്ഥിരം int64_t cne1  = matrix_row_counts[cur_a];

                int64_t src1_cur_start = 0;
                int64_t src1_cur_end   = cne1;

                int64_t src0_cur_start = (ith_n * ne01) / nth_n;
                int64_t src0_cur_end   = MIN(((ith_n + 1) * ne01) / nth_n, ne01);

                എങ്കിൽ (src1_cur_start >= src1_cur_end || src0_cur_start >= src0_cur_end) {
                    തുടരുക;
                }

                src0_cur_start =
                    (src0_cur_start % NB_COLS) ? src0_cur_start + NB_COLS - (src0_cur_start % NB_COLS) : src0_cur_start;
                src0_cur_end =
                    (src0_cur_end % NB_COLS) ? src0_cur_end + NB_COLS - (src0_cur_end % NB_COLS) : src0_cur_end;

                auto *    src0_cur = (uint8_t *) src0->data + cur_a * expert_b_stride + src0_cur_start * row_stride_b;
                uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? src0_cur : nullptr;

                size_t extra_tcm_buffer_size = tcm_buffer_size;
                ശൂന്യം * extra_tcm_buffer      = tcm_buffer;
                എങ്കിൽ (tcm_buffer != nullptr && (src1_cur_end - src1_cur_start) >= 4 &&
                    (src0_cur_end - src0_cur_start) * row_stride_b <= tcm_buffer_size) {
                    spacemit_kernels::rvv::memcpy1d(tcm_buffer, src0_cur,
                                                    (src0_cur_end - src0_cur_start) * row_stride_b);
                    src0_cur = reinterpret_cast<uint8_t *>(tcm_buffer);
                    b_col_zp = block_type_has_zp<BLOC_TYPE>() ? src0_cur : nullptr;
                    extra_tcm_buffer_size -= (src0_cur_end - src0_cur_start) * row_stride_b;
                    extra_tcm_buffer = reinterpret_cast<ശൂന്യം *>(reinterpret_cast<uintptr_t>(tcm_buffer) +
                                                                (src0_cur_end - src0_cur_start) * row_stride_b);
                }

                പൂർണ്ണസംഖ്യ ir1 = src1_cur_start;

                എങ്കിൽ (extra_tcm_buffer_size >= nbw1 && extra_tcm_buffer != nullptr) {
                    int64_t quant_a_tile_size = extra_tcm_buffer_size / nbw1;
                    ചെയ്യുക {
                        quant_a_tile_size = MIN(quant_a_tile_size, src1_cur_end - ir1);

                        uint8_t * quant_a_tile_buffer = reinterpret_cast<uint8_t *>(extra_tcm_buffer);

                        പൂർണ്ണസംഖ്യ iir1 = ir1;
                        വേണ്ടി (; iir1 < (ir1 + quant_a_tile_size); ++iir1) {
                            mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, iir1);

                            സ്ഥിരം പൂർണ്ണസംഖ്യ id = row_mapping.i1;  // selected expert index

                            സ്ഥിരം int64_t i11 = id % ne11;
                            സ്ഥിരം int64_t i12 = row_mapping.i2;  // row index in src1

                            auto * src1_col = quant_a_buffer + (i11 * nbw1 + i12 * nbw2);
                            spacemit_kernels::rvv::memcpy1d(quant_a_tile_buffer, src1_col, nbw1);
                            quant_a_tile_buffer = quant_a_tile_buffer + nbw1;
                        }

                        quant_a_tile_buffer = reinterpret_cast<uint8_t *>(extra_tcm_buffer);
                        iir1                = ir1;

                        എങ്കിൽ (moe_gemm_kernel_m2 != nullptr) {
                            വേണ്ടി (; iir1 < (ir1 + quant_a_tile_size - 1); iir1 += 2, quant_a_tile_buffer += 2 * nbw1) {
                                mmid_row_mapping row_mapping_0 = MMID_MATRIX_ROW(cur_a, iir1);
                                mmid_row_mapping row_mapping_1 = MMID_MATRIX_ROW(cur_a, iir1 + 1);

                                src_workspaces[0] = quant_a_tile_buffer;
                                src_workspaces[1] = quant_a_tile_buffer + nbw1;

                                dst_workspaces[0] =
                                    (ദശാംശം *) ((അക്ഷരം *) dst->data + (row_mapping_0.i1 * nb1 + row_mapping_0.i2 * nb2)) +
                                    src0_cur_start;
                                dst_workspaces[1] = (ദശാംശം *) ((അക്ഷരം *) dst->data +
                                                               ((row_mapping_1.i1) * nb1 + (row_mapping_1.i2) * nb2)) +
                                                    src0_cur_start;
                                moe_gemm_kernel_m2(b_blk_len, src_workspaces.data(), src0_cur, b_col_zp,
                                                   dst_workspaces.data(), 1, src0_cur_end - src0_cur_start, b_k_blks,
                                                   ne01);
                            }
                        }

                        വേണ്ടി (; iir1 < (ir1 + quant_a_tile_size); iir1++, quant_a_tile_buffer += nbw1) {
                            mmid_row_mapping row_mapping_0 = MMID_MATRIX_ROW(cur_a, iir1);

                            gemm_kernel(
                                b_blk_len, quant_a_tile_buffer, src0_cur, b_col_zp,
                                (ദശാംശം *) ((അക്ഷരം *) dst->data + (row_mapping_0.i1 * nb1 + row_mapping_0.i2 * nb2)) +
                                    src0_cur_start,
                                1, src0_cur_end - src0_cur_start, b_k_blks, ne01);
                        }

                        ir1 += quant_a_tile_size;
                    } ആകുമ്പോൾ (ir1 < src1_cur_end);
                } അല്ലെങ്കിൽ {
                    എങ്കിൽ (moe_gemm_kernel_m2 != nullptr) {
                        വേണ്ടി (; ir1 < src1_cur_end - 1; ir1 += 2) {
                            വേണ്ടി (പൂർണ്ണസംഖ്യ iir1 = 0; iir1 < 2; ++iir1) {
                                mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, ir1 + iir1);

                                സ്ഥിരം പൂർണ്ണസംഖ്യ id = row_mapping.i1;  // selected expert index

                                സ്ഥിരം int64_t i11 = id % ne11;
                                സ്ഥിരം int64_t i12 = row_mapping.i2;  // row index in src1

                                സ്ഥിരം int64_t i1 = id;               // selected expert index
                                സ്ഥിരം int64_t i2 = i12;              // row

                                src_workspaces[iir1] = quant_a_buffer + (i11 * nbw1 + i12 * nbw2);

                                dst_workspaces[iir1] =
                                    (ദശാംശം *) ((അക്ഷരം *) dst->data + (i1 * nb1 + i2 * nb2)) + src0_cur_start;
                            }

                            moe_gemm_kernel_m2(b_blk_len, src_workspaces.data(), src0_cur, b_col_zp,
                                               dst_workspaces.data(), 1, src0_cur_end - src0_cur_start, b_k_blks, ne01);
                        }
                    }

                    വേണ്ടി (; ir1 < src1_cur_end; ir1++) {
                        mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, ir1);

                        സ്ഥിരം പൂർണ്ണസംഖ്യ id = row_mapping.i1;  // selected expert index

                        സ്ഥിരം int64_t i11 = id % ne11;
                        സ്ഥിരം int64_t i12 = row_mapping.i2;  // row index in src1

                        സ്ഥിരം int64_t i1 = id;               // selected expert index
                        സ്ഥിരം int64_t i2 = i12;              // row

                        auto * src1_col = quant_a_buffer + (i11 * nbw1 + i12 * nbw2);

                        gemm_kernel(b_blk_len, src1_col, src0_cur, b_col_zp,
                                    (ദശാംശം *) ((അക്ഷരം *) dst->data + (i1 * nb1 + i2 * nb2)) + src0_cur_start, 1,
                                    src0_cur_end - src0_cur_start, b_k_blks, ne01);
                    }
                }
            }
        }
#undef MMID_MATRIX_ROW
    }

    പൂർണ്ണസംഖ്യ repack(ggml_tensor * t, സ്ഥിരം ശൂന്യം * data, size_t data_size) override {
        GGML_LOG_DEBUG("%s: repack tensor %s with %s_%dx%d\n", __func__, t->name, ggml_type_name(t->type),
                       (പൂർണ്ണസംഖ്യ) NB_COLS, (പൂർണ്ണസംഖ്യ) INTER_SIZE);
        തിരികെ ggml::cpu::riscv64_spacemit::repack<BLOC_TYPE, INTER_SIZE, NB_COLS>(t, data, data_size);
    }
};

വർഗ്ഗം tensor_traits_common : പൊതു tensor_traits_base {
    തർക്കം work_size(പൂർണ്ണസംഖ്യ n_threads, സ്ഥിരം ggml_tensor * op, size_t & size) override {
        തിരഞ്ഞെടുപ്പ് (op->op) {
            അവസ്ഥ GGML_OP_FLASH_ATTN_EXT:
                {
                    സ്ഥിരം പൂർണ്ണസംഖ്യ     n_tasks = n_threads;
                    സ്ഥിരം int64_t neq2    = op->src[0]->ne[2];  // number of query heads
                    സ്ഥിരം int64_t DK      = op->src[1]->ne[0];
                    സ്ഥിരം int64_t DV      = op->src[2]->ne[0];  // DV

                    // Tiled flash attention scratch (tile sizes defined in common.h)
                    // Per-thread: Q_q + KQ + mask + VKQ32 + V32 + K_f32 + padding
                    size_t prefill = വലുപ്പം(ദശാംശം) *
                                     (GGML_FA_TILE_Q * DK + 2 * GGML_FA_TILE_Q * GGML_FA_TILE_KV + GGML_FA_TILE_Q * DV +
                                      GGML_FA_TILE_KV * DV + GGML_FA_TILE_KV * DK) *
                                     n_tasks;

                    // Decode path: n_kv_chunks = n_tasks (one chunk per thread)
                    // Per-thread: VKQ accmulator (DV), partial M, partial S + intra-thread scratch for V, Q and VKQ
                    size_t n_chunks = n_tasks;
                    size_t decode   = വലുപ്പം(ദശാംശം) * (neq2 * n_chunks * (2 + DV) + n_tasks * (DK + 2 * DV));

                    size = MAX(prefill, decode);
                }
                തിരികെ സത്യം;
            സ്ഥിരസ്ഥിതി:
                നിർത്തുക;
        }
        തിരികെ അസത്യം;
    }

    തർക്കം compute_forward(ggml_compute_params * params, ggml_tensor * op) override {
        തിരഞ്ഞെടുപ്പ് (op->op) {
            അവസ്ഥ GGML_OP_NORM:
                തിരഞ്ഞെടുപ്പ് (op->src[0]->type) {
                    അവസ്ഥ GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_norm_f32(params, op);
                        തിരികെ സത്യം;
                    സ്ഥിരസ്ഥിതി:
                        GGML_ABORT("fatal error");
                }
            അവസ്ഥ GGML_OP_RMS_NORM:
                തിരഞ്ഞെടുപ്പ് (op->src[0]->type) {
                    അവസ്ഥ GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_rms_norm_f32(params, op);
                        തിരികെ സത്യം;
                    സ്ഥിരസ്ഥിതി:
                        GGML_ABORT("fatal error");
                }
            അവസ്ഥ GGML_OP_ADD:
                തിരഞ്ഞെടുപ്പ് (op->src[0]->type) {
                    അവസ്ഥ GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_ദ്വിമാന<GGML_OP_ADD, ദശാംശം>(params, op);
                        തിരികെ സത്യം;
                    അവസ്ഥ GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_ദ്വിമാന<GGML_OP_ADD, _Float16>(params, op);
                        തിരികെ സത്യം;
                    സ്ഥിരസ്ഥിതി:
                        ggml_compute_forward_add(params, op);
                        തിരികെ സത്യം;
                }
            അവസ്ഥ GGML_OP_SUB:
                തിരഞ്ഞെടുപ്പ് (op->src[0]->type) {
                    അവസ്ഥ GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_ദ്വിമാന<GGML_OP_SUB, ദശാംശം>(params, op);
                        തിരികെ സത്യം;
                    അവസ്ഥ GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_ദ്വിമാന<GGML_OP_SUB, _Float16>(params, op);
                        തിരികെ സത്യം;
                    സ്ഥിരസ്ഥിതി:
                        ggml_compute_forward_sub(params, op);
                        തിരികെ സത്യം;
                }
            അവസ്ഥ GGML_OP_MUL:
                തിരഞ്ഞെടുപ്പ് (op->src[0]->type) {
                    അവസ്ഥ GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_ദ്വിമാന<GGML_OP_MUL, ദശാംശം>(params, op);
                        തിരികെ സത്യം;
                    അവസ്ഥ GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_ദ്വിമാന<GGML_OP_MUL, _Float16>(params, op);
                        തിരികെ സത്യം;
                    സ്ഥിരസ്ഥിതി:
                        ggml_compute_forward_mul(params, op);
                        തിരികെ സത്യം;
                }
            അവസ്ഥ GGML_OP_DIV:
                തിരഞ്ഞെടുപ്പ് (op->src[0]->type) {
                    അവസ്ഥ GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_ദ്വിമാന<GGML_OP_DIV, ദശാംശം>(params, op);
                        തിരികെ സത്യം;
                    അവസ്ഥ GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_ദ്വിമാന<GGML_OP_DIV, _Float16>(params, op);
                        തിരികെ സത്യം;
                    സ്ഥിരസ്ഥിതി:
                        ggml_compute_forward_div(params, op);
                        തിരികെ സത്യം;
                }
            അവസ്ഥ GGML_OP_FLASH_ATTN_EXT:
                forward_flash_attn_ext_f16(params, op);
                തിരികെ സത്യം;
            അവസ്ഥ GGML_OP_CONT:
                {
                    സ്ഥിരം ggml_tensor * src0 = op->src[0];
                    എങ്കിൽ (op->type == src0->type && op->nb[0] != src0->nb[0] && op->nb[0] == src0->nb[1] &&
                        op->ne[3] * op->ne[2] * op->nb[2] == src0->ne[3] * src0->ne[2] * src0->nb[2]) {
                        spacemit_kernels::rvv::forward_cont_with_permute(params, op);
                    } അല്ലെങ്കിൽ {
                        ggml_compute_forward_cont(params, op);
                    }
                    തിരികെ സത്യം;
                }
            അവസ്ഥ GGML_OP_CPY:
                {
                    സ്ഥിരം ggml_tensor * src0 = op->src[0];
                    എങ്കിൽ (op->type == src0->type && op->nb[0] == src0->nb[1] && src0->nb[0] != src0->nb[1] &&
                        ggml_nelements(src0) == ggml_nelements(op)) {
                        spacemit_kernels::rvv::forward_cpy_with_permute(params, op);
                    } അല്ലെങ്കിൽ {
                        ggml_compute_forward_cpy(params, op);
                    }
                    തിരികെ സത്യം;
                }
            അവസ്ഥ GGML_OP_REPEAT:
                {
                    സ്ഥിരം തർക്കം rows_equal         = ggml_nrows(op->src[0]) == ggml_nrows(op);
                    സ്ഥിരം തർക്കം broadcast_or_equal = op->src[0]->ne[0] == 1 || op->src[0]->ne[0] == op->ne[0];

                    എങ്കിൽ (rows_equal && broadcast_or_equal) {
                        തിരഞ്ഞെടുപ്പ് (op->src[0]->type) {
                            അവസ്ഥ GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_repeat_nrows<int32_t>(params, op);
                                തിരികെ സത്യം;
                            അവസ്ഥ GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_repeat_nrows<int16_t>(params, op);
                                തിരികെ സത്യം;
                            സ്ഥിരസ്ഥിതി:
                                നിർത്തുക;
                        }
                    }

                    എങ്കിൽ (op->src[0]->ne[1] == 1 && op->src[0]->ne[0] == op->ne[0]) {
                        തിരഞ്ഞെടുപ്പ് (op->src[0]->type) {
                            അവസ്ഥ GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_repeat_dim1<int32_t>(params, op);
                                തിരികെ സത്യം;
                            അവസ്ഥ GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_repeat_dim1<int16_t>(params, op);
                                തിരികെ സത്യം;
                            സ്ഥിരസ്ഥിതി:
                                നിർത്തുക;
                        }
                    }

                    ggml_compute_forward_repeat(params, op);
                }
                തിരികെ സത്യം;
            അവസ്ഥ GGML_OP_SUM_ROWS:
                {
                    എങ്കിൽ (op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32) {
                        spacemit_kernels::rvv::forward_sum_rows<ദശാംശം>(params, op);
                    } അല്ലെങ്കിൽ {
                        ggml_compute_forward_sum_rows(params, op);
                    }
                }
                തിരികെ സത്യം;
            അവസ്ഥ GGML_OP_GET_ROWS:
                {
                    എങ്കിൽ (op->src[0]->type == op->type) {
                        തിരഞ്ഞെടുപ്പ് (op->src[0]->type) {
                            അവസ്ഥ GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_get_rows<int32_t>(params, op);
                                തിരികെ സത്യം;
                            അവസ്ഥ GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_get_rows<int16_t>(params, op);
                                തിരികെ സത്യം;
                            സ്ഥിരസ്ഥിതി:
                                നിർത്തുക;
                        }
                    }

                    ggml_compute_forward_get_rows(params, op);
                }
                തിരികെ സത്യം;
            അവസ്ഥ GGML_OP_CONCAT:
                {
                    സ്ഥിരം int32_t dim = ggml_get_op_params_i32(op, 0);
                    എങ്കിൽ (dim == 0 && op->type == op->src[0]->type) {
                        തിരഞ്ഞെടുപ്പ് (op->src[0]->type) {
                            അവസ്ഥ GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_concat<int32_t>(params, op);
                                തിരികെ സത്യം;
                            അവസ്ഥ GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_concat<int16_t>(params, op);
                                തിരികെ സത്യം;
                            സ്ഥിരസ്ഥിതി:
                                നിർത്തുക;
                        }
                    }

                    ggml_compute_forward_concat(params, op);
                }
                തിരികെ സത്യം;
            // TODO For GGML_OP_GATED_DELTA_NET
            // case GGML_OP_GATED_DELTA_NET:
            //     return true;
            സ്ഥിരസ്ഥിതി:
                നിർത്തുക;
        }
        തിരികെ അസത്യം;
    }

    ശൂന്യം forward_flash_attn_ext_f16(സ്ഥിരം ggml_compute_params * params, ggml_tensor * dst) {
        സ്ഥിരം ggml_tensor * q = dst->src[0];
        സ്ഥിരം ggml_tensor * k = dst->src[1];
        സ്ഥിരം ggml_tensor * v = dst->src[2];

        GGML_TENSOR_LOCALS(int64_t, neq, q, ne)
        GGML_TENSOR_LOCALS(size_t, nbq, q, nb)
        GGML_TENSOR_LOCALS(int64_t, nek, k, ne)
        GGML_TENSOR_LOCALS(size_t, nbk, k, nb)
        GGML_TENSOR_LOCALS(int64_t, nev, v, ne)
        GGML_TENSOR_LOCALS(size_t, nbv, v, nb)
        GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
        GGML_TENSOR_LOCALS(size_t, nb, dst, nb)

        സ്ഥിരം int64_t DK = nek0;
        സ്ഥിരം int64_t DV = nev0;

        സ്ഥിരം തർക്കം supported_prec  = (dst->op_params[3] == GGML_PREC_F32 || dst->op_params[3] == GGML_PREC_DEFAULT);
        സ്ഥിരം തർക്കം supported_types = (q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16);
        സ്ഥിരം തർക്കം supported_shape = (DK > 0 && DK <= 128 && DV > 0 && DV <= 128);
        സ്ഥിരം തർക്കം supported_vlen  = (__riscv_vlenb() == 128);

        എങ്കിൽ (!(supported_prec && supported_types && supported_shape && supported_vlen)) {
            ggml_compute_forward_flash_attn_ext(params, dst);
            തിരികെ;
        }

        // total rows in q
        സ്ഥിരം int64_t nr = neq1 * neq2 * neq3;

        // rows per thread
        സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
        സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

        സ്ഥിതം constexpr int64_t Q_TILE_SZ = ggml_fa_tile_config::Q;
        സ്ഥിരം തർക്കം               use_tiled = !params->use_ref && (neq1 >= Q_TILE_SZ);

        // 4x chunks per thread
        // int     nth_scaled = nth * 4;
        // int64_t chunk_size = (nr + nth_scaled - 1) / nth_scaled;
        // int64_t nchunk     = (nr + chunk_size - 1) / chunk_size;

        // if (nth == 1 || nchunk < nth) {
        //     nchunk = nth;
        // }

        int64_t nchunk = nth;

        എങ്കിൽ (ith == 0) {
            // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
            ggml_threadpool_chunk_set(params->threadpool, nth);
        }

        ggml_barrier(params->threadpool);

        // The number of elements in each chunk
        സ്ഥിരം int64_t dr = (nr + nchunk - 1) / nchunk;

        // The first chunk comes from our thread_id, the rest will get auto-assigned.
        പൂർണ്ണസംഖ്യ current_chunk = ith;

        ആകുമ്പോൾ (current_chunk < nchunk) {
            സ്ഥിരം int64_t ir0 = dr * current_chunk;
            സ്ഥിരം int64_t ir1 = MIN(ir0 + dr, nr);

            എങ്കിൽ (use_tiled) {
                spacemit_kernels::rvv::forward_flash_attn_ext_f16_tiled_vlen1024_vf16(
                    params, dst, ir0, ir1, ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer,
                    ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer_size);
            } അല്ലെങ്കിൽ {
                spacemit_kernels::rvv::forward_flash_attn_ext_f16_one_chunk_vlen1024_vf16(
                    params, dst, ir0, ir1, ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer,
                    ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer_size);
            }

            current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
        }
    }

    പൂർണ്ണസംഖ്യ repack(ggml_tensor * t, സ്ഥിരം ശൂന്യം * data, size_t data_size) override {
        memcpy(t->data, data, data_size);
        തിരികെ 0;
    }
};

// Impl By IME1
സ്ഥിതം സ്ഥിരം tensor_traits<block_q4_0, 32, 16>  q4_0_16x32_q8_0;
സ്ഥിതം സ്ഥിരം tensor_traits<block_q4_1, 32, 16>  q4_1_16x32_q8_0;
സ്ഥിതം സ്ഥിരം tensor_traits<block_q4_K, 32, 16>  q4_k_16x32_q8_0;
സ്ഥിതം സ്ഥിരം tensor_traits<block_q8_0, 32, 16>  q8_0_16x32_q8_0;
// Impl By IME2
സ്ഥിതം സ്ഥിരം tensor_traits<block_q2_K, 256, 32> q2_k_32x256_q8_0;
സ്ഥിതം സ്ഥിരം tensor_traits<block_q3_K, 256, 32> q3_k_32x256_q8_0;
സ്ഥിതം സ്ഥിരം tensor_traits<block_q4_0, 32, 32>  q4_0_32x32_q8_0;
സ്ഥിതം സ്ഥിരം tensor_traits<block_q4_1, 32, 32>  q4_1_32x32_q8_0;
സ്ഥിതം സ്ഥിരം tensor_traits<block_q4_0, 256, 32> q4_0_32x256_q8_0;
സ്ഥിതം സ്ഥിരം tensor_traits<block_q4_1, 256, 32> q4_1_32x256_q8_0;
സ്ഥിതം സ്ഥിരം tensor_traits<block_q4_K, 32, 32>  q4_k_32x32_q8_0;
സ്ഥിതം സ്ഥിരം tensor_traits<block_q6_K, 32, 32>  q6_k_32x32_q8_0;
സ്ഥിതം സ്ഥിരം tensor_traits<block_q8_0, 32, 32>  q8_0_32x32_q8_0;
സ്ഥിതം സ്ഥിരം tensor_traits<block_mxfp4, 32, 32> mxfp4_32x32_q8_0;
സ്ഥിതം സ്ഥിരം tensor_traits<block_q5_K, 32, 32>  q5_k_32x32_q8_0;
സ്ഥിതം സ്ഥിരം tensor_traits<block_q5_1, 32, 32>  q5_1_32x32_q8_0;
സ്ഥിതം സ്ഥിരം tensor_traits<block_q5_0, 32, 32>  q5_0_32x32_q8_0;
// Impl By RVV
സ്ഥിതം സ്ഥിരം tensor_traits_common               rvv_impl;

}  // namespace ggml::cpu::riscv64_spacemit

സ്ഥിതം സ്ഥിരം ggml::cpu::tensor_traits * ggml_riscv64_spacemit_get_optimal_repack_type(സ്ഥിരം ggml_tensor * cur) {
    തിരഞ്ഞെടുപ്പ് (cur->type) {
        അവസ്ഥ GGML_TYPE_Q2_K:
            {
#എങ്കിൽ defined(RISCV64_SPACEMIT_IME2)
                എങ്കിൽ (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    തിരികെ &ggml::cpu::riscv64_spacemit::q2_k_32x256_q8_0;
                }
#endif
            }
            നിർത്തുക;
        അവസ്ഥ GGML_TYPE_Q3_K:
            {
#എങ്കിൽ defined(RISCV64_SPACEMIT_IME2)
                എങ്കിൽ (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    തിരികെ &ggml::cpu::riscv64_spacemit::q3_k_32x256_q8_0;
                }
#endif
            }
            നിർത്തുക;
        അവസ്ഥ GGML_TYPE_Q4_0:
            {
#എങ്കിൽ defined(RISCV64_SPACEMIT_IME2)
                എങ്കിൽ (cur->ne[1] % 32 == 0 && cur->ne[0] % 256 == 0 &&
                    (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    തിരികെ &ggml::cpu::riscv64_spacemit::q4_0_32x256_q8_0;
                }

                എങ്കിൽ (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    തിരികെ &ggml::cpu::riscv64_spacemit::q4_0_32x32_q8_0;
                }
#endif

#എങ്കിൽ defined(RISCV64_SPACEMIT_IME1)
                എങ്കിൽ (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    തിരികെ &ggml::cpu::riscv64_spacemit::q4_0_16x32_q8_0;
                }
#endif
            }
            നിർത്തുക;
        അവസ്ഥ GGML_TYPE_Q4_1:
            {
#എങ്കിൽ defined(RISCV64_SPACEMIT_IME2)
                // TODO
                // if (cur->ne[1] % 32 == 0 && cur->ne[0] % 256 == 0 &&
                //     (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                //     return &ggml::cpu::riscv64_spacemit::q4_1_32x256_q8_0;
                // }

                എങ്കിൽ (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    തിരികെ &ggml::cpu::riscv64_spacemit::q4_1_32x32_q8_0;
                }
#endif

#എങ്കിൽ defined(RISCV64_SPACEMIT_IME1)
                എങ്കിൽ (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    തിരികെ &ggml::cpu::riscv64_spacemit::q4_1_16x32_q8_0;
                }
#endif
            }
            നിർത്തുക;
        അവസ്ഥ GGML_TYPE_Q4_K:
            {
#എങ്കിൽ defined(RISCV64_SPACEMIT_IME2)
                എങ്കിൽ (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    തിരികെ &ggml::cpu::riscv64_spacemit::q4_k_32x32_q8_0;
                }
#endif

#എങ്കിൽ defined(RISCV64_SPACEMIT_IME1)
                എങ്കിൽ (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    തിരികെ &ggml::cpu::riscv64_spacemit::q4_k_16x32_q8_0;
                }
#endif
            }
            നിർത്തുക;
        അവസ്ഥ GGML_TYPE_Q6_K:
            {
#എങ്കിൽ defined(RISCV64_SPACEMIT_IME2)
                എങ്കിൽ ((ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    തിരികെ &ggml::cpu::riscv64_spacemit::q6_k_32x32_q8_0;
                }
#endif
            }
            നിർത്തുക;
        അവസ്ഥ GGML_TYPE_Q8_0:
            {
#എങ്കിൽ defined(RISCV64_SPACEMIT_IME2)
                എങ്കിൽ ((ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    തിരികെ &ggml::cpu::riscv64_spacemit::q8_0_32x32_q8_0;
                }
#endif

#എങ്കിൽ defined(RISCV64_SPACEMIT_IME1)
                എങ്കിൽ (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    തിരികെ &ggml::cpu::riscv64_spacemit::q8_0_16x32_q8_0;
                }
#endif
            }
            നിർത്തുക;
        അവസ്ഥ GGML_TYPE_MXFP4:
            {
#എങ്കിൽ defined(RISCV64_SPACEMIT_IME2)
                // TODO
                // if (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                //     return &ggml::cpu::riscv64_spacemit::mxfp4_32x32_q8_0;
                // }
#endif
            }
            നിർത്തുക;
        അവസ്ഥ GGML_TYPE_Q5_K:
            {
#എങ്കിൽ defined(RISCV64_SPACEMIT_IME2)
                എങ്കിൽ (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    തിരികെ &ggml::cpu::riscv64_spacemit::q5_k_32x32_q8_0;
                }
#endif
            }
            നിർത്തുക;
        അവസ്ഥ GGML_TYPE_Q5_1:
            {
#എങ്കിൽ defined(RISCV64_SPACEMIT_IME2)
                എങ്കിൽ (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    തിരികെ &ggml::cpu::riscv64_spacemit::q5_1_32x32_q8_0;
                }
#endif
            }
            നിർത്തുക;
        അവസ്ഥ GGML_TYPE_Q5_0:
            {
#എങ്കിൽ defined(RISCV64_SPACEMIT_IME2)
                എങ്കിൽ (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    തിരികെ &ggml::cpu::riscv64_spacemit::q5_0_32x32_q8_0;
                }
#endif
            }
            നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            നിർത്തുക;
    }

    തിരികെ nullptr;
}

സ്ഥിതം എണ്ണൽ ggml_status ggml_backend_riscv64_spacemit_buffer_init_tensor(ggml_backend_buffer_t buffer,
                                                                         ggml_tensor *         tensor) {
    tensor->extra =
        (ശൂന്യം *) const_cast<ggml::cpu::tensor_traits *>(ggml_riscv64_spacemit_get_optimal_repack_type(tensor));

    GGML_UNUSED(buffer);

    തിരികെ GGML_STATUS_SUCCESS;
}

സ്ഥിതം ശൂന്യം ggml_backend_riscv64_spacemit_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);

    ശൂന്യം * base = buffer->context;
    എങ്കിൽ (base == nullptr) {
        തിരികെ;
    }

    ggml::cpu::riscv64_spacemit::spine_mem_pool_free(base);
}

സ്ഥിതം ശൂന്യം * ggml_backend_riscv64_spacemit_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);

    ശൂന്യം * base = buffer->context;
    GGML_ASSERT(base != nullptr);
    തിരികെ base;
}

സ്ഥിതം ശൂന്യം ggml_backend_riscv64_spacemit_buffer_memset_tensor(ggml_backend_buffer_t buffer,
                                                               ggml_tensor *         tensor,
                                                               uint8_t               value,
                                                               size_t                offset,
                                                               size_t                size) {
    GGML_ASSERT(tensor);
    memset((അക്ഷരം *) tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

സ്ഥിതം ശൂന്യം ggml_backend_riscv64_spacemit_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);

    ശൂന്യം * base = buffer->context;
    GGML_ASSERT(base != nullptr);
    memset(base, value, buffer->size);
}

സ്ഥിതം ശൂന്യം ggml_backend_riscv64_spacemit_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                            ggml_tensor *         tensor,
                                                            സ്ഥിരം ശൂന്യം *          data,
                                                            size_t                offset,
                                                            size_t                size) {
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    auto tensor_traits = (ggml::cpu::riscv64_spacemit::tensor_traits_base *) tensor->extra;
    എങ്കിൽ (tensor_traits) {
        auto OK = tensor_traits->repack(tensor, data, size);
        GGML_ASSERT(OK == 0);
    }

    GGML_UNUSED(buffer);
}

സ്ഥിതം സ്ഥിരം ggml_backend_buffer_i ggml_backend_riscv64_spacemit_buffer_i = {
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

സ്ഥിതം സ്ഥിരം അക്ഷരം * ggml_backend_cpu_riscv64_spacemit_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    തിരികെ "CPU_RISCV64_SPACEMIT";

    GGML_UNUSED(buft);
}

സ്ഥിതം ggml_backend_buffer_t ggml_backend_cpu_riscv64_spacemit_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                                                                        size_t size) {
    ശൂന്യം * base = ggml::cpu::riscv64_spacemit::spine_mem_pool_alloc(size, 64);
    എങ്കിൽ (base == nullptr) {
        തിരികെ nullptr;
    }

    തിരികെ ggml_backend_buffer_init(buft, ggml_backend_riscv64_spacemit_buffer_i, base, size);
}

സ്ഥിതം size_t ggml_backend_cpu_riscv64_spacemit_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    തിരികെ 64;

    GGML_UNUSED(buft);
}

സ്ഥിതം size_t ggml_backend_cpu_riscv64_spacemit_nbytes(ggml_backend_buffer_type_t buft, സ്ഥിരം ggml_tensor * tensor) {
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < GGML_MAX_DIMS; ++i) {
        എങ്കിൽ (tensor->ne[i] <= 0) {
            തിരികെ 0;
        }
    }

    GGML_UNUSED(buft);

    സ്ഥിരം auto plain_nbytes = [&]() {
        size_t total = ggml_type_size(tensor->type);
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < GGML_MAX_DIMS; ++i) {
            total += (tensor->ne[i] - 1) * tensor->nb[i];
        }
        തിരികെ total;
    };

    സ്ഥിരം size_t blck_size = ggml_blck_size(tensor->type);
    എങ്കിൽ (blck_size == 1) {
        തിരികെ plain_nbytes();
    }

    സ്ഥിരം size_t row_nbytes = tensor->ne[0] * tensor->nb[0] / blck_size;

    സ്ഥിരം auto add_strided_nbytes = [&](size_t total, size_t src_block_size, size_t dst_block_size) {
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 1; i < GGML_MAX_DIMS; ++i) {
            total += (tensor->ne[i] - 1) * (tensor->nb[i] / src_block_size) * dst_block_size;
        }
        തിരികെ total;
    };

    സ്ഥിരം auto remap_block_nbytes = [&](size_t src_block_size, size_t dst_block_size, int64_t padded_rows = 0) {
        GGML_ASSERT(row_nbytes % src_block_size == 0);

        size_t total =
            add_strided_nbytes((row_nbytes / src_block_size) * dst_block_size, src_block_size, dst_block_size);

        എങ്കിൽ (padded_rows > 0 && tensor->ne[1] % padded_rows != 0) {
            total += (padded_rows - tensor->ne[1] % padded_rows) * (tensor->nb[1] / src_block_size) * dst_block_size;
        }

        തിരികെ total;
    };

    size_t nbytes = row_nbytes;
    തിരഞ്ഞെടുപ്പ് (tensor->type) {
        അവസ്ഥ GGML_TYPE_Q4_K:
            nbytes = remap_block_nbytes(വലുപ്പം(block_q4_K), വലുപ്പം(block_q4_1) * 8);
            നിർത്തുക;
        അവസ്ഥ GGML_TYPE_Q6_K:
            nbytes = remap_block_nbytes(വലുപ്പം(block_q6_K), വലുപ്പം(block_q8_0) * 8, 32);
            നിർത്തുക;
        അവസ്ഥ GGML_TYPE_Q8_0:
            nbytes = remap_block_nbytes(വലുപ്പം(block_q8_0), വലുപ്പം(block_q8_0), 32);
            നിർത്തുക;
        അവസ്ഥ GGML_TYPE_Q2_K:
            nbytes = remap_block_nbytes(വലുപ്പം(block_q2_K), വലുപ്പം(spacemit_kernels::nrow_block_q2_k<1>));
            നിർത്തുക;
        അവസ്ഥ GGML_TYPE_Q3_K:
            nbytes = remap_block_nbytes(വലുപ്പം(block_q3_K), വലുപ്പം(spacemit_kernels::nrow_block_q3_k<1>));
            നിർത്തുക;
        അവസ്ഥ GGML_TYPE_MXFP4:
            nbytes = remap_block_nbytes(വലുപ്പം(block_mxfp4), വലുപ്പം(spacemit_kernels::nrow_block_mxfp4<1>));
            നിർത്തുക;
        അവസ്ഥ GGML_TYPE_Q5_K:
            nbytes = remap_block_nbytes(വലുപ്പം(block_q5_K), വലുപ്പം(spacemit_kernels::nrow_block_q5_1<1>) * 8);
            നിർത്തുക;
        അവസ്ഥ GGML_TYPE_Q5_1:
            nbytes = remap_block_nbytes(വലുപ്പം(block_q5_1), വലുപ്പം(spacemit_kernels::nrow_block_q5_1<1>));
            നിർത്തുക;
        അവസ്ഥ GGML_TYPE_Q5_0:
            nbytes = remap_block_nbytes(വലുപ്പം(block_q5_0), വലുപ്പം(spacemit_kernels::nrow_block_q5_0<1>));
            നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            nbytes = add_strided_nbytes(row_nbytes, 1, 1);
            നിർത്തുക;
    }

    തിരികെ nbytes;
}

നാമാകാശ ggml::cpu::riscv64_spacemit {

വർഗ്ഗം extra_buffer_type : ggml::cpu::extra_buffer_type {
    തർക്കം supports_op(ggml_backend_dev_t, സ്ഥിരം ggml_tensor * op) override {
        തിരഞ്ഞെടുപ്പ് (op->op) {
            അവസ്ഥ GGML_OP_MUL_MAT:
                എങ്കിൽ (op->src[0]->buffer && (ggml_n_dims(op->src[0]) == 2) &&
                    op->src[0]->buffer->buft == ggml_backend_cpu_riscv64_spacemit_buffer_type() &&
                    ggml_riscv64_spacemit_get_optimal_repack_type(op->src[0])) {
                    എങ്കിൽ (op->src[1]->buffer && !ggml_backend_buft_is_host(op->src[1]->buffer->buft)) {
                        തിരികെ അസത്യം;
                    }
                    എങ്കിൽ (op->src[1]->type == GGML_TYPE_F32) {
                        തിരികെ സത്യം;
                    }
                }
                നിർത്തുക;
            അവസ്ഥ GGML_OP_MUL_MAT_ID:
                എങ്കിൽ (op->src[0]->buffer && (ggml_n_dims(op->src[0]) == 3) &&
                    op->src[0]->buffer->buft == ggml_backend_cpu_riscv64_spacemit_buffer_type() &&
                    ggml_riscv64_spacemit_get_optimal_repack_type(op->src[0])) {
                    എങ്കിൽ (op->src[1]->buffer && !ggml_backend_buft_is_host(op->src[1]->buffer->buft)) {
                        തിരികെ അസത്യം;
                    }
                    എങ്കിൽ (op->src[1]->type == GGML_TYPE_F32) {
                        തിരികെ സത്യം;
                    }
                }
                നിർത്തുക;
            സ്ഥിരസ്ഥിതി:
                // GGML_ABORT("fatal error");
                നിർത്തുക;
        }
        തിരികെ അസത്യം;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(സ്ഥിരം ggml_tensor * op) override {
        തിരഞ്ഞെടുപ്പ് (op->op) {
            അവസ്ഥ GGML_OP_MUL_MAT:
            അവസ്ഥ GGML_OP_MUL_MAT_ID:
                എങ്കിൽ (op->src[0]->buffer && op->src[0]->buffer->buft == ggml_backend_cpu_riscv64_spacemit_buffer_type()) {
                    തിരികെ (ggml::cpu::tensor_traits *) op->src[0]->extra;
                }
                നിർത്തുക;
            അവസ്ഥ GGML_OP_NORM:
            അവസ്ഥ GGML_OP_RMS_NORM:
            അവസ്ഥ GGML_OP_ADD:
            അവസ്ഥ GGML_OP_SUB:
            അവസ്ഥ GGML_OP_MUL:
            അവസ്ഥ GGML_OP_DIV:
            അവസ്ഥ GGML_OP_FLASH_ATTN_EXT:
            അവസ്ഥ GGML_OP_CONT:
            അവസ്ഥ GGML_OP_CPY:
            അവസ്ഥ GGML_OP_REPEAT:
            അവസ്ഥ GGML_OP_SUM_ROWS:
            അവസ്ഥ GGML_OP_GET_ROWS:
            അവസ്ഥ GGML_OP_CONCAT:
                // case GGML_OP_GATED_DELTA_NET:
                തിരികെ (ggml::cpu::tensor_traits *) (&ggml::cpu::riscv64_spacemit::rvv_impl);
            സ്ഥിരസ്ഥിതി:
                // GGML_ABORT("fatal error");
                നിർത്തുക;
        }

        തിരികെ nullptr;
    }
};

}  // namespace ggml::cpu::riscv64_spacemit

ggml_backend_buffer_type_t ggml_backend_cpu_riscv64_spacemit_buffer_type(ശൂന്യം) {
    സ്ഥിതം ggml_backend_buffer_type ggml_backend_cpu_buffer_type_riscv64_spacemit = {
  /* .iface    = */
        {
         /* .get_name           = */ ggml_backend_cpu_riscv64_spacemit_buffer_type_get_name,
         /* .alloc_buffer       = */ ggml_backend_cpu_riscv64_spacemit_buffer_type_alloc_buffer,
         /* .alloc_buffer_n     = */ ശൂന്യ,
         /* .get_alignment      = */ ggml_backend_cpu_riscv64_spacemit_buffer_type_get_alignment,
         /* .get_max_size       = */ nullptr,
         /* .get_alloc_size     = */ ggml_backend_cpu_riscv64_spacemit_nbytes,
         /* .get_alloc_size_n   = */ ശൂന്യ,
         /* .is_host            = */ nullptr,
         },
 /* .device  = */
        ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
 /* .context = */
        പുതിയ ggml::cpu::riscv64_spacemit::extra_buffer_type(),
    };

    തിരികെ &ggml_backend_cpu_buffer_type_riscv64_spacemit;
}

ബാഹ്യ "C" {
സ്ഥിതം പൂർണ്ണസംഖ്യ bind_ai_thread() {
    പൂർണ്ണസംഖ്യ  fd, bytes;
    അക്ഷരം str[32];

    fd = open("/proc/set_ai_thread", O_WRONLY);
    എങ്കിൽ (fd < 0) {
        GGML_LOG_ERROR("try open /proc/set_ai_thread failed\n");
        തിരികെ -1;
    }

    snprintf(str, 16, "%d", 0);
    bytes = ലിഖ(fd, str, strlen(str));
    എങ്കിൽ (bytes < 0) {
        GGML_LOG_ERROR("try write /proc/set_ai_thread failed\n");
        close(fd);
        തിരികെ -1;
    }

    close(fd);
    തിരികെ 0;
}

ശൂന്യം ggml_backend_cpu_riscv64_spacemit_set_numa_thread_affinity(പൂർണ്ണസംഖ്യ thread_n) {
    പൂർണ്ണസംഖ്യ cpu_id = sched_getcpu();
    എങ്കിൽ (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2 &&
        !((1 << cpu_id) & ggml::cpu::riscv64_spacemit::global_spine_env_info.cpu_mask)) {
        GGML_PRINT_DEBUG("bind_ai_thread for thread %d, pid %d\n", thread_n, getpid());
        bind_ai_thread();
    }

    എങ്കിൽ (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_tcm &&
        ggml::cpu::riscv64_spacemit::tls_context.cpu_id == -1) {
        CPU_ZERO(&(ggml::cpu::riscv64_spacemit::tls_context.cpuset));
        pthread_t    main_thread     = pthread_self();
        സ്ഥിരം auto & perfer_core_ids = ggml::cpu::riscv64_spacemit::global_spine_env_info.perfer_core_ids;
        എങ്കിൽ (thread_n < 0 || static_cast<size_t>(thread_n) >= perfer_core_ids.size()) {
            GGML_ABORT("thread_n %d exceeds perfer_core_ids size %zu\n", thread_n, perfer_core_ids.size());
        }
        auto perfer_cpu_id = perfer_core_ids[static_cast<size_t>(thread_n)];
        CPU_SET(perfer_cpu_id, &(ggml::cpu::riscv64_spacemit::tls_context.cpuset));
        പൂർണ്ണസംഖ്യ s =
            pthread_setaffinity_np(main_thread, വലുപ്പം(cpu_set_t), &(ggml::cpu::riscv64_spacemit::tls_context.cpuset));
        എങ്കിൽ (s != 0) {
            GGML_ABORT("set thread affinity error for thread_n %d, cpu_id %d\n", thread_n, perfer_cpu_id);
        }

        പൂർണ്ണസംഖ്യ ai_cpu_id = perfer_cpu_id - ggml::cpu::riscv64_spacemit::global_spine_env_info.aicpu_id_offset;
        ggml::cpu::riscv64_spacemit::tls_context.cpu_id = ai_cpu_id;
        ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer =
            ggml::cpu::riscv64_spacemit::spine_mem_pool_tcm_mem_get(ai_cpu_id);
        ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer_size =
            ggml::cpu::riscv64_spacemit::global_spine_env_info.tcm_blk_size;
    }

    എങ്കിൽ (ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer != nullptr) {
        ശൂന്യം * rt =
            ggml::cpu::riscv64_spacemit::spine_mem_pool_tcm_mem_wait(ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        എങ്കിൽ (rt == nullptr) {
            GGML_ABORT("wait tcm buffer failed for cpu_id: %d", ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        }
    }
}

ശൂന്യം ggml_backend_cpu_riscv64_spacemit_clear_numa_thread_affinity_threaded(പൂർണ്ണസംഖ്യ thread_n) {
    എങ്കിൽ (ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer != nullptr) {
        auto rt = ggml::cpu::riscv64_spacemit::spine_mem_pool_tcm_mem_release(
            ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        എങ്കിൽ (rt != 0) {
            GGML_ABORT("release tcm buffer failed for cpu_id: %d", ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        }
    }
}
}
