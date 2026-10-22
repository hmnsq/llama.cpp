#నిర్వచించు GGML_COMMON_IMPL_CPP
#నిర్వచించు GGML_COMMON_DECL_CPP

#అన్తర్భావ "ime.h"

#అన్తర్భావ "binary-ops.h"
#అన్తర్భావ "common.h"
#అన్తర్భావ "ggml-backend-impl.h"
#అన్తర్భావ "ggml-common.h"
#అన్తర్భావ "ggml-cpu.h"
#అన్తర్భావ "ime_env.h"
#అన్తర్భావ "ime_kernels.h"
#అన్తర్భావ "ops.h"
#అన్తర్భావ "repack.h"
#అన్తర్భావ "rvv_kernels.h"
#అన్తర్భావ "spine_mem_pool.h"
#అన్తర్భావ "traits.h"
#అన్తర్భావ "vec.h"

#అన్తర్భావ <fcntl.h>
#అన్తర్భావ <sys/mman.h>
#అన్తర్భావ <unistd.h>

#అన్తర్భావ <algorithm>
#అన్తర్భావ <atomic>
#అన్తర్భావ <cassert>
#అన్తర్భావ <cerrno>
#అన్తర్భావ <cmath>
#అన్తర్భావ <cstdio>  // for GGML_ASSERT
#అన్తర్భావ <stdexcept>
#అన్తర్భావ <thread>
// clang-format off
#ఒకవేళ defined(__riscv)

#ఒకవేళ !defined(__riscv_v) || !defined(__riscv_v_intrinsic)
#దోషం "riscv v extension or v_intrinsic not enabled"
#లేకపోతే
#అన్తర్భావ <riscv_vector.h>
#అయితే_ముగింపు

#ఒకవేళ !defined(__riscv_zfh) || !defined(__riscv_zvfh)
#దోషం "riscv zfh extension not enabled, GGML_RV_ZFH and GGML_RV_ZVFH must be defined to 1"
#అయితే_ముగింపు

#ఒకవేళ !defined(__riscv_zba)
#దోషం "riscv zba extension not enabled, GGML_RV_ZBA must be defined to 1"
#అయితే_ముగింపు

#ఒకవేళ defined(RISCV64_SPACEMIT_IME1) || defined(RISCV64_SPACEMIT_IME2)
#లేకపోతే
#దోషం "RISCV64_SPACEMIT_IME1 or RISCV64_SPACEMIT_IME2 not defined"
#అయితే_ముగింపు

#లేకపోతే

#దోషం "riscv not enabled in this build"

#అయితే_ముగింపు

#ఒకవేళ defined(__GNUC__)
#సూచన GCC diagnostic ignored "-Woverlength-strings"
#సూచన GCC diagnostic ignored "-Wcast-qual"
#సూచన GCC diagnostic ignored "-Wunused-parameter"
#అయితే_ముగింపు

// clang-format on

బాహ్య "C" {
బాహ్య శూన్య ggml_threadpool_chunk_సమితి(నిర్మాణం ggml_threadpool * tp, పూర్ణాంకం value);
బాహ్య పూర్ణాంకం  ggml_threadpool_chunk_కూడు(నిర్మాణం ggml_threadpool * tp, పూర్ణాంకం value);
}

నామాకాశ ggml::cpu::riscv64_spacemit {

నిర్మాణం TLSContext {
    పూర్ణాంకం       cpu_id{ -1 };
    cpu_set_t cpuset;
    శూన్య *    tcm_బఫరు{ శూన్యసూచిక };
    size_t    tcm_బఫరు_కొలత{ 0 };
};

thread_local TLSContext tls_context;

మూస <రకం_పేరు BLOC_TYPE, int64_t INTER_కొలత, int64_t NB_నిలువుల_సంఖ్య> constexpr size_t get_repacked_ఖండం_type_కొలత() {
    ఒకవేళ constexpr (std::is_same_v<BLOC_TYPE, ఖండం_q6_K> || std::is_same_v<BLOC_TYPE, ఖండం_q8_0>) {
        తిరిగి పరిమాణం(ఖండం_q8_0);
    } లేకపోతే ఒకవేళ constexpr (std::is_same_v<BLOC_TYPE, ఖండం_q4_0>) {
        తిరిగి పరిమాణం(ఖండం_q4_0) * INTER_కొలత / QK4_0;
    } లేకపోతే ఒకవేళ constexpr (std::is_same_v<BLOC_TYPE, ఖండం_q4_1> || std::is_same_v<BLOC_TYPE, ఖండం_q4_K>) {
        తిరిగి (పరిమాణం(ఖండం_q4_0) + పరిమాణం(uint8_t)) * INTER_కొలత / QK4_1;
    } లేకపోతే ఒకవేళ constexpr (std::is_same_v<BLOC_TYPE, ఖండం_q2_K>) {
        తిరిగి పరిమాణం(spacemit_kernels::nrow_ఖండం_q2_k<1>);
    } లేకపోతే ఒకవేళ constexpr (std::is_same_v<BLOC_TYPE, ఖండం_q3_K>) {
        తిరిగి పరిమాణం(spacemit_kernels::nrow_ఖండం_q3_k<1>);
    } లేకపోతే ఒకవేళ constexpr (std::is_same_v<BLOC_TYPE, ఖండం_mxfp4>) {
        తిరిగి పరిమాణం(spacemit_kernels::nrow_ఖండం_mxfp4<1>);
    } లేకపోతే ఒకవేళ constexpr (std::is_same_v<BLOC_TYPE, ఖండం_q5_1> || std::is_same_v<BLOC_TYPE, ఖండం_q5_K>) {
        తిరిగి పరిమాణం(spacemit_kernels::nrow_ఖండం_q5_1<1>);
    } లేకపోతే ఒకవేళ constexpr (std::is_same_v<BLOC_TYPE, ఖండం_q5_0>) {
        తిరిగి పరిమాణం(spacemit_kernels::nrow_ఖండం_q5_0<1>);
    } లేకపోతే {
        ప్రతిజ్ఞా(అబద్ధం);
        తిరిగి 0;
    }
}

మూస <రకం_పేరు BLOC_TYPE> constexpr తార్కిక ఖండం_type_has_zp() {
    ఒకవేళ constexpr (std::is_same_v<BLOC_TYPE, ఖండం_q6_K> || std::is_same_v<BLOC_TYPE, ఖండం_q8_0> ||
                  std::is_same_v<BLOC_TYPE, ఖండం_q3_K> || std::is_same_v<BLOC_TYPE, ఖండం_q4_0> ||
                  std::is_same_v<BLOC_TYPE, ఖండం_mxfp4> || std::is_same_v<BLOC_TYPE, ఖండం_q5_0>) {
        తిరిగి అబద్ధం;
    } లేకపోతే ఒకవేళ constexpr (std::is_same_v<BLOC_TYPE, ఖండం_q4_1> || std::is_same_v<BLOC_TYPE, ఖండం_q4_K> ||
                         std::is_same_v<BLOC_TYPE, ఖండం_q2_K> || std::is_same_v<BLOC_TYPE, ఖండం_q5_1> ||
                         std::is_same_v<BLOC_TYPE, ఖండం_q5_K>) {
        తిరిగి నిజం;
    } లేకపోతే {
        ప్రతిజ్ఞా(అబద్ధం);
        తిరిగి అబద్ధం;
    }
}

తరగతి tensor_traits_పునాది : బహిరంగ ggml::cpu::tensor_traits {
  బహిరంగ:
    మాయా పూర్ణాంకం repack(ggml_tensor * t, స్థిరరాశి శూన్య * దత్తాంశం, size_t దత్తాంశం_కొలత) = 0;
};

మూస <రకం_పేరు BLOC_TYPE, int64_t INTER_కొలత, int64_t NB_నిలువుల_సంఖ్య> తరగతి tensor_traits : బహిరంగ tensor_traits_పునాది {
    తార్కిక work_కొలత(పూర్ణాంకం /* n_threads */, స్థిరరాశి ggml_tensor * op, size_t & size) override {
        ఎంపిక (op->op) {
            స్థితి GGML_OP_MUL_MAT:
                {
                    int64_t src1_nelements = ggml_nelements(op->src[1]);

                    ఒకవేళ constexpr (std::is_same_v<BLOC_TYPE, ఖండం_q2_K> || std::is_same_v<BLOC_TYPE, ఖండం_q3_K>) {
                        size =
                            spacemit_kernels::div_విడత_పైకి(src1_nelements, QK_K) * spacemit_kernels::q8k_blk_కొలత(QK_K);
                    } లేకపోతే ఒకవేళ constexpr (INTER_కొలత == QK4_0) {
                        size = spacemit_kernels::div_విడత_పైకి(src1_nelements, QK4_0) *
                               spacemit_kernels::q8_blk_కొలత(QK4_0, నిజం);
                    } లేకపోతే ఒకవేళ constexpr (INTER_కొలత == 256) {
                        size = spacemit_kernels::div_విడత_పైకి(src1_nelements, 256) *
                               spacemit_kernels::q8_hp_blk_కొలత(256, నిజం, నిజం);
                    } లేకపోతే {
                        GGML_ABORT("unsupported block type");
                    }

                    size = GGML_PAD(size, పరిమాణం(int64_t));

                    తిరిగి నిజం;
                }
            స్థితి GGML_OP_MUL_MAT_ID:
                {
                    int64_t src1_nelements = ggml_nelements(op->src[1]);

                    ఒకవేళ constexpr (std::is_same_v<BLOC_TYPE, ఖండం_q2_K> || std::is_same_v<BLOC_TYPE, ఖండం_q3_K>) {
                        size =
                            spacemit_kernels::div_విడత_పైకి(src1_nelements, QK_K) * spacemit_kernels::q8k_blk_కొలత(QK_K);
                    } లేకపోతే ఒకవేళ constexpr (INTER_కొలత == QK4_0) {
                        size = spacemit_kernels::div_విడత_పైకి(src1_nelements, QK4_0) *
                               spacemit_kernels::q8_blk_కొలత(QK4_0, నిజం);
                    } లేకపోతే ఒకవేళ constexpr (INTER_కొలత == 256) {
                        size = spacemit_kernels::div_విడత_పైకి(src1_nelements, 256) *
                               spacemit_kernels::q8_hp_blk_కొలత(256, నిజం, నిజం);
                    } లేకపోతే {
                        GGML_ABORT("unsupported block type");
                    }

                    size = GGML_PAD(size, పరిమాణం(int64_t));

                    స్థిరరాశి int64_t ne02 = op->src[0]->ne[2];  // n_as, n_expert
                    స్థిరరాశి int64_t ne12 = op->src[1]->ne[2];  // n_tokens

                    స్థిరరాశి size_t sizeof_mmid_అడ్డువరుస_mapping = పరిమాణం(int64_t);
                    size += sizeof_mmid_అడ్డువరుస_mapping * ne02 * (ne12 + 1) + (ne02 + 1) * పరిమాణం(int64_t);

                    size = GGML_PAD(size, పరిమాణం(int64_t));

                    తిరిగి నిజం;
                }
            అప్రమేయ:
                // GGML_ABORT("fatal error");
                ఆపు;
        }
        తిరిగి అబద్ధం;
    }

    తార్కిక compute_forward(ggml_compute_params * params, ggml_tensor * op) override {
        ఎంపిక (op->op) {
            స్థితి GGML_OP_MUL_MAT:
                ఎంపిక (op->src[0]->type) {
                    స్థితి GGML_TYPE_Q2_K:
                    స్థితి GGML_TYPE_Q3_K:
                    స్థితి GGML_TYPE_Q4_0:
                    స్థితి GGML_TYPE_Q4_1:
                    స్థితి GGML_TYPE_Q4_K:
                    స్థితి GGML_TYPE_Q6_K:
                    స్థితి GGML_TYPE_Q8_0:
                    స్థితి GGML_TYPE_Q5_0:
                    స్థితి GGML_TYPE_Q5_1:
                    స్థితి GGML_TYPE_Q5_K:
                        //case GGML_TYPE_MXFP4:
                        forward_mul_mat(params, op);
                        తిరిగి నిజం;
                    అప్రమేయ:
                        // GGML_ABORT("fatal error: unsupported type for src0 in MUL_MAT");
                        తిరిగి అబద్ధం;
                }
                ఆపు;
            స్థితి GGML_OP_MUL_MAT_ID:
                ఎంపిక (op->src[0]->type) {
                    స్థితి GGML_TYPE_Q2_K:
                    స్థితి GGML_TYPE_Q3_K:
                    స్థితి GGML_TYPE_Q4_0:
                    స్థితి GGML_TYPE_Q4_1:
                    స్థితి GGML_TYPE_Q4_K:
                    స్థితి GGML_TYPE_Q6_K:
                    స్థితి GGML_TYPE_Q8_0:
                    స్థితి GGML_TYPE_Q5_0:
                    స్థితి GGML_TYPE_Q5_1:
                    స్థితి GGML_TYPE_Q5_K:
                        //case GGML_TYPE_MXFP4:
                        forward_mul_mat_id(params, op);
                        తిరిగి నిజం;
                    అప్రమేయ:
                        // GGML_ABORT("fatal error: unsupported type for src0 in MUL_MAT_ID");
                        తిరిగి అబద్ధం;
                }
                ఆపు;
            అప్రమేయ:
                // GGML_ABORT("fatal error");
                ఆపు;
        }
        తిరిగి అబద్ధం;
    }

    శూన్య forward_mul_mat(ggml_compute_params * params, ggml_tensor * op) {
        constexpr size_t a_blk_len = INTER_కొలత;
        constexpr size_t b_blk_len = INTER_కొలత;

        స్థిరరాశి ggml_tensor * src0 = op->src[0];
        స్థిరరాశి ggml_tensor * src1 = op->src[1];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_ద్విమాన_OP_LOCALS

        పూర్ణాంకం ith = params->ith;
        పూర్ణాంకం nth = params->nth;

        [[maybe_unused]] స్థిరరాశి గణన ggml_type type = src0->type;

        శూన్య *        w_దత్తాంశం  = (శూన్య *) src0->దత్తాంశం;
        స్థిరరాశి దశాంశం * feature = (స్థిరరాశి దశాంశం *) src1->దత్తాంశం;
        దశాంశం *       output  = (దశాంశం *) dst->దత్తాంశం;

        స్థిరరాశి int64_t gemm_m = ne11 * ne12 * ne13;
        స్థిరరాశి int64_t gemm_k = ne10;
        స్థిరరాశి int64_t gemm_n = ne01;

        spacemit_kernels::quantize_a_అడ్డువరుస_def       quantize_a_అడ్డువరుస_i8;
        spacemit_kernels::quantize_a_అడ్డువరుస_def       quantize_a_4row_i8;
        spacemit_kernels::gemm_kernel_quantize_def gemm_kernel;
        తార్కిక                                       సమితి_kernel_impl = అబద్ధం;

        int64_t ఖండం_stride_a = spacemit_kernels::q8_blk_కొలత(a_blk_len);

#ఒకవేళ defined(RISCV64_SPACEMIT_IME2)
        ఒకవేళ (!సమితి_kernel_impl && (global_spine_env_info.use_ime2)) {
            quantize_a_అడ్డువరుస_i8  = spacemit_kernels::rvv::quantize_a_అడ్డువరుస_i8;
            quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8;
            ఖండం_stride_a     = spacemit_kernels::q8_blk_కొలత(a_blk_len, నిజం);

            ఒకవేళ constexpr (std::is_same_v<BLOC_TYPE, ఖండం_q6_K> || std::is_same_v<BLOC_TYPE, ఖండం_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i8;
                సమితి_kernel_impl = నిజం;
            } లేకపోతే ఒకవేళ constexpr (std::is_same_v<BLOC_TYPE, ఖండం_q4_0> || std::is_same_v<BLOC_TYPE, ఖండం_q4_1> ||
                                 std::is_same_v<BLOC_TYPE, ఖండం_q4_K>) {
                ఒకవేళ constexpr (INTER_కొలత == 256) {
                    gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i4_hp;
                    quantize_a_అడ్డువరుస_i8  = spacemit_kernels::rvv::quantize_a_అడ్డువరుస_i8_hp;
                    quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8_hp;
                    ఖండం_stride_a     = spacemit_kernels::q8_hp_blk_కొలత(a_blk_len, నిజం, నిజం);
                    సమితి_kernel_impl    = నిజం;
                } లేకపోతే {
                    gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i4;
                    quantize_a_అడ్డువరుస_i8  = spacemit_kernels::rvv::quantize_a_అడ్డువరుస_i8;
                    quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8;
                    ఖండం_stride_a     = spacemit_kernels::q8_blk_కొలత(a_blk_len, నిజం);
                    సమితి_kernel_impl    = నిజం;
                }
            } లేకపోతే ఒకవేళ constexpr (std::is_same_v<BLOC_TYPE, ఖండం_q2_K>) {
                quantize_a_అడ్డువరుస_i8  = spacemit_kernels::rvv::quantize_a_అడ్డువరుస_i8k;
                quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8k;
                ఖండం_stride_a     = spacemit_kernels::q8k_blk_కొలత(a_blk_len);

                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i2k;
                సమితి_kernel_impl = నిజం;
            } లేకపోతే ఒకవేళ constexpr (std::is_same_v<BLOC_TYPE, ఖండం_q3_K>) {
                quantize_a_అడ్డువరుస_i8  = spacemit_kernels::rvv::quantize_a_అడ్డువరుస_i8k;
                quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8k;
                ఖండం_stride_a     = spacemit_kernels::q8k_blk_కొలత(a_blk_len);

                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i3k;
                సమితి_kernel_impl = నిజం;
            } లేకపోతే ఒకవేళ constexpr (std::is_same_v<BLOC_TYPE, ఖండం_mxfp4>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8mxfp4;
                సమితి_kernel_impl = నిజం;
            } లేకపోతే ఒకవేళ constexpr (std::is_same_v<BLOC_TYPE, ఖండం_q5_1> || std::is_same_v<BLOC_TYPE, ఖండం_q5_K> ||
                                 std::is_same_v<BLOC_TYPE, ఖండం_q5_0>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i5;
                సమితి_kernel_impl = నిజం;
            }
        }
#అయితే_ముగింపు

#ఒకవేళ defined(RISCV64_SPACEMIT_IME1)
        ఒకవేళ (!సమితి_kernel_impl && (global_spine_env_info.use_ime1)) {
            quantize_a_అడ్డువరుస_i8  = spacemit_kernels::ime1::quantize_a_అడ్డువరుస_i8;
            quantize_a_4row_i8 = spacemit_kernels::ime1::quantize_a_4row_i8;

            ఒకవేళ constexpr (std::is_same_v<BLOC_TYPE, ఖండం_q4_0> || std::is_same_v<BLOC_TYPE, ఖండం_q4_1> ||
                          std::is_same_v<BLOC_TYPE, ఖండం_q4_K>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i4;
                సమితి_kernel_impl = నిజం;
            } లేకపోతే ఒకవేళ constexpr (std::is_same_v<BLOC_TYPE, ఖండం_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i8;
                సమితి_kernel_impl = నిజం;
            }
        }
#అయితే_ముగింపు
        ఒకవేళ (!సమితి_kernel_impl) {
            GGML_ABORT("no kernel implementation found for the block type");
        }

        స్థిరరాశి int64_t a_k_blks = spacemit_kernels::div_విడత_పైకి(gemm_k, a_blk_len);
        స్థిరరాశి int64_t b_k_blks = spacemit_kernels::div_విడత_పైకి(gemm_k, b_blk_len);

        స్థిరరాశి int64_t అడ్డువరుస_stride_a        = a_k_blks * ఖండం_stride_a;
        స్థిరరాశి int64_t gemm_workspace_కొలత = GGML_PAD(gemm_m * అడ్డువరుస_stride_a, alignof(int64_t));

        ఒకవేళ (ith == 0 && params->wsize < gemm_workspace_కొలత) {
            GGML_ABORT("wsize less than gemm_workspace_size");
        }

        uintptr_t ws_ptr = reinterpret_cast<uintptr_t>(params->wdata);

        శూన్య *        tcm_బఫరు      = ggml::cpu::riscv64_spacemit::tls_context.tcm_బఫరు;
        స్థిరరాశి int64_t tcm_బఫరు_కొలత = ggml::cpu::riscv64_spacemit::tls_context.tcm_బఫరు_కొలత;

        స్వయంచాలక * quant_a_బఫరు = reinterpret_cast<uint8_t *>(ws_ptr);

        constexpr int64_t అడ్డువరుస_align = 4;
        స్థిరరాశి int64_t     అడ్డువరుస_blks  = spacemit_kernels::div_విడత_పైకి(gemm_m, అడ్డువరుస_align);

        స్థిరరాశి int64_t అడ్డువరుస_stride_b      = b_k_blks * get_repacked_ఖండం_type_కొలత<BLOC_TYPE, INTER_కొలత, NB_నిలువుల_సంఖ్య>();
        స్థిరరాశి int64_t చొప్పున_mb_అడ్డువరుసలు_wsize = అడ్డువరుస_align * అడ్డువరుస_stride_a;
        స్థిరరాశి int64_t చొప్పున_nb_నిలువుల_సంఖ్య_wsize = NB_నిలువుల_సంఖ్య * అడ్డువరుస_stride_b;

        స్థిరరాశి int64_t barrier_idx = static_cast<int64_t>(ith / 2);

        GGML_ASSERT(global_spine_env_info.init_barrier != శూన్యసూచిక);
        GGML_ASSERT(barrier_idx < spine_init_barrier_లెక్క);
        spine_barrier_t * cur_barrier = &global_spine_env_info.init_barrier[barrier_idx];

        ఒకవేళ (gemm_m == 1) {
            పూర్ణాంకం task_చొప్పున_thread = spacemit_kernels::div_విడత_పైకి(a_k_blks, nth);
            పూర్ణాంకం a_blk_ప్రారంభం     = ith * task_చొప్పున_thread;
            పూర్ణాంకం a_blk_చివర       = std::min(a_blk_ప్రారంభం + task_చొప్పున_thread, (పూర్ణాంకం) a_k_blks);
            ఒకవేళ (a_blk_ప్రారంభం < a_blk_చివర) {
                quantize_a_అడ్డువరుస_i8(a_blk_len, feature + a_blk_ప్రారంభం * a_blk_len, (a_blk_చివర - a_blk_ప్రారంభం) * a_blk_len,
                                  quant_a_బఫరు + a_blk_ప్రారంభం * ఖండం_stride_a);
            }
        } లేకపోతే {
            పూర్ణాంకం task_చొప్పున_thread = spacemit_kernels::div_విడత_పైకి(అడ్డువరుస_blks, nth);
            పూర్ణాంకం m_అడ్డువరుస_blk_ప్రారంభం = ith * task_చొప్పున_thread;
            పూర్ణాంకం m_అడ్డువరుస_blk_చివర   = std::min(m_అడ్డువరుస_blk_ప్రారంభం + task_చొప్పున_thread, (పూర్ణాంకం) అడ్డువరుస_blks);
            కొరకు (పూర్ణాంకం m_అడ్డువరుస_blk = m_అడ్డువరుస_blk_ప్రారంభం; m_అడ్డువరుస_blk < m_అడ్డువరుస_blk_చివర; m_అడ్డువరుస_blk++) {
                పూర్ణాంకం m_idx             = m_అడ్డువరుస_blk * అడ్డువరుస_align;
                పూర్ణాంకం అడ్డువరుసలు_tobe_handled = (gemm_m - m_idx) > అడ్డువరుస_align ? అడ్డువరుస_align : (gemm_m - m_idx);

                ఒకవేళ (అడ్డువరుసలు_tobe_handled == అడ్డువరుస_align && quantize_a_4row_i8 != శూన్యసూచిక) {
                    స్థిరరాశి దశాంశం * a_అడ్డువరుస_ptr       = feature + m_idx * gemm_k;
                    స్వయంచాలక *        quant_a_అడ్డువరుస_ptr = quant_a_బఫరు + m_idx * అడ్డువరుస_stride_a;
                    quantize_a_4row_i8(a_blk_len, a_అడ్డువరుస_ptr, gemm_k, quant_a_అడ్డువరుస_ptr);
                } లేకపోతే {
                    అయితే (అడ్డువరుసలు_tobe_handled) {
                        స్థిరరాశి దశాంశం * a_అడ్డువరుస_ptr       = feature + m_idx * gemm_k;
                        స్వయంచాలక *        quant_a_అడ్డువరుస_ptr = quant_a_బఫరు + m_idx * అడ్డువరుస_stride_a;
                        quantize_a_అడ్డువరుస_i8(a_blk_len, a_అడ్డువరుస_ptr, gemm_k, quant_a_అడ్డువరుస_ptr);
                        అడ్డువరుసలు_tobe_handled -= 1;
                        m_idx += 1;
                    }
                }
            }
        }

        ggml_barrier(params->threadpool);

        స్థిరరాశి int64_t gemm_m_stride     = gemm_n / gemm_m > 64 ? gemm_m : 16;
        స్థిరరాశి int64_t gemm_m_blocked    = spacemit_kernels::div_విడత_పైకి(gemm_m, gemm_m_stride);
        స్థిరరాశి int64_t max_gemm_n_stride = spacemit_kernels::div_విడత_పైకి(gemm_n * gemm_m_blocked, nth);

        int64_t gemm_n_stride = gemm_n;
        ఒకవేళ (max_gemm_n_stride < gemm_n) {
            gemm_n_stride =
                std::min(gemm_n_stride, spacemit_kernels::div_విడత_పైకి(max_gemm_n_stride, NB_నిలువుల_సంఖ్య) * NB_నిలువుల_సంఖ్య);
        }

        ఒకవేళ (gemm_n_stride == gemm_n && tcm_బఫరు != శూన్యసూచిక && చొప్పున_mb_అడ్డువరుసలు_wsize <= tcm_బఫరు_కొలత) {
            కొరకు (int64_t m_ప్రారంభం = ith * అడ్డువరుస_align; m_ప్రారంభం < gemm_m; m_ప్రారంభం += అడ్డువరుస_align * nth) {
                uint8_t * b_నిలువు    = reinterpret_cast<uint8_t *>(w_దత్తాంశం);
                uint8_t * b_నిలువు_zp = ఖండం_type_has_zp<BLOC_TYPE>() ? b_నిలువు : శూన్యసూచిక;

                int64_t m_అడ్డువరుస_real = std::min(gemm_m - m_ప్రారంభం, అడ్డువరుస_align);

                spacemit_kernels::rvv::memcpy1d(tcm_బఫరు, quant_a_బఫరు + m_ప్రారంభం * అడ్డువరుస_stride_a,
                                                m_అడ్డువరుస_real * అడ్డువరుస_stride_a);

                int64_t n_blk_real = 0;
                కొరకు (int64_t ni = 0; ni < gemm_n; ni += n_blk_real, b_నిలువు += n_blk_real * అడ్డువరుస_stride_b) {
                    n_blk_real = std::min(gemm_n - ni, (int64_t) NB_నిలువుల_సంఖ్య);

                    uint8_t * a_అడ్డువరుస_ptr = (uint8_t *) tcm_బఫరు;
                    దశాంశం *   c_blk     = output + m_ప్రారంభం * gemm_n + ni;

                    int32_t అడ్డువరుసలు_మిగిలిన = m_అడ్డువరుస_real;

                    అయితే (అడ్డువరుసలు_మిగిలిన > 0) {
                        స్వయంచాలక అడ్డువరుసలు_handled = gemm_kernel(b_blk_len, a_అడ్డువరుస_ptr, b_నిలువు, b_నిలువు_zp, c_blk, అడ్డువరుసలు_మిగిలిన,
                                                        n_blk_real, b_k_blks, gemm_n);

                        c_blk += అడ్డువరుసలు_handled * gemm_n;
                        a_అడ్డువరుస_ptr += అడ్డువరుసలు_handled * అడ్డువరుస_stride_a;

                        అడ్డువరుసలు_మిగిలిన -= అడ్డువరుసలు_handled;
                    }
                }
            }
        } లేకపోతే ఒకవేళ (tcm_బఫరు != శూన్యసూచిక && చొప్పున_nb_నిలువుల_సంఖ్య_wsize <= tcm_బఫరు_కొలత) {
            uint8_t * a_అడ్డువరుస = quant_a_బఫరు;
            uint8_t * b_నిలువు = reinterpret_cast<uint8_t *>(tcm_బఫరు);
            ఒకవేళ ((gemm_workspace_కొలత + చొప్పున_nb_నిలువుల_సంఖ్య_wsize) <= tcm_బఫరు_కొలత) {
                a_అడ్డువరుస = (uint8_t *) tcm_బఫరు;
                b_నిలువు = reinterpret_cast<uint8_t *>(tcm_బఫరు) + gemm_workspace_కొలత;
            }
            uint8_t * b_నిలువు_zp = ఖండం_type_has_zp<BLOC_TYPE>() ? b_నిలువు : శూన్యసూచిక;

            int64_t ni      = ith * NB_నిలువుల_సంఖ్య;
            int64_t nb_real = std::min(gemm_n - ni, NB_నిలువుల_సంఖ్య);

            ఒకవేళ (ith % 2 == 0 && nb_real > 0) {
                spacemit_kernels::rvv::memcpy1d(b_నిలువు, reinterpret_cast<uint8_t *>(w_దత్తాంశం) + ni * అడ్డువరుస_stride_b,
                                                nb_real * అడ్డువరుస_stride_b);
                ఒకవేళ (a_అడ్డువరుస != quant_a_బఫరు) {
                    spacemit_kernels::rvv::memcpy1d(a_అడ్డువరుస, quant_a_బఫరు, gemm_workspace_కొలత);
                }
            }

            spine_barrier_wait(cur_barrier);

            ఒకవేళ (ith % 2 != 0 && nb_real > 0) {
                ఒకవేళ (a_అడ్డువరుస != quant_a_బఫరు) {
                    spacemit_kernels::rvv::memcpy1d(a_అడ్డువరుస, quant_a_బఫరు, gemm_workspace_కొలత);
                }
                spacemit_kernels::rvv::memcpy1d(b_నిలువు, reinterpret_cast<uint8_t *>(w_దత్తాంశం) + ni * అడ్డువరుస_stride_b,
                                                nb_real * అడ్డువరుస_stride_b);
            }

            కొరకు (; ni < gemm_n; ni += NB_నిలువుల_సంఖ్య * nth) {
                int64_t అడ్డువరుసలు_మిగిలిన = gemm_m;
                దశాంశం * c_blk          = output + ni;
                స్వయంచాలక *  a_అడ్డువరుస_cur      = a_అడ్డువరుస;

                ఒకవేళ (ith % 2 != 0) {
                    spine_barrier_wait(cur_barrier);
                }

                అయితే (అడ్డువరుసలు_మిగిలిన > 0) {
                    స్వయంచాలక అడ్డువరుసలు_handled = gemm_kernel(b_blk_len, a_అడ్డువరుస_cur, b_నిలువు, b_నిలువు_zp, c_blk, అడ్డువరుసలు_మిగిలిన,
                                                    nb_real, b_k_blks, gemm_n);

                    c_blk += అడ్డువరుసలు_handled * gemm_n;
                    a_అడ్డువరుస_cur += అడ్డువరుసలు_handled * అడ్డువరుస_stride_a;

                    అడ్డువరుసలు_మిగిలిన -= అడ్డువరుసలు_handled;
                }

                ఒకవేళ (ith % 2 == 0) {
                    spine_barrier_wait(cur_barrier);
                }

                స్థిరరాశి int64_t తదుపరి_ni = ni + NB_నిలువుల_సంఖ్య * nth;
                ఒకవేళ (తదుపరి_ni < gemm_n) {
                    nb_real = std::min(gemm_n - తదుపరి_ni, NB_నిలువుల_సంఖ్య);
                    spacemit_kernels::rvv::memcpy1d(b_నిలువు, reinterpret_cast<uint8_t *>(w_దత్తాంశం) + తదుపరి_ni * అడ్డువరుస_stride_b,
                                                    nb_real * అడ్డువరుస_stride_b);
                }
            }
        } లేకపోతే {
            స్థిరరాశి int64_t task_లెక్క_m = spacemit_kernels::div_విడత_పైకి(gemm_m, gemm_m_stride);
            స్థిరరాశి int64_t task_లెక్క_n = spacemit_kernels::div_విడత_పైకి(gemm_n, gemm_n_stride);

            int64_t task_లెక్క      = task_లెక్క_m * task_లెక్క_n;
            int64_t task_చొప్పున_thread = (task_లెక్క + nth - 1) / nth;
            int64_t start           = ith * task_చొప్పున_thread;
            int64_t end             = std::min((ith + 1) * task_చొప్పున_thread, task_లెక్క);
            కొరకు (int64_t compute_idx = start; compute_idx < end; compute_idx++) {
                స్థిరరాశి స్వయంచాలక tid_n = compute_idx / task_లెక్క_m;
                స్థిరరాశి స్వయంచాలక tid_m = compute_idx % task_లెక్క_m;

                స్థిరరాశి int64_t m_ప్రారంభం = tid_m * gemm_m_stride;
                స్థిరరాశి int64_t m_లెక్క = std::min(gemm_m - m_ప్రారంభం, (int64_t) gemm_m_stride);

                స్థిరరాశి int64_t n_ప్రారంభం = tid_n * gemm_n_stride;
                స్థిరరాశి int64_t n_లెక్క = std::min(gemm_n - n_ప్రారంభం, (int64_t) gemm_n_stride);

                స్థిరరాశి int64_t n_blk = m_లెక్క == 1 ? n_లెక్క : NB_నిలువుల_సంఖ్య;

                uint8_t * b_నిలువు    = reinterpret_cast<uint8_t *>(w_దత్తాంశం) + n_ప్రారంభం * అడ్డువరుస_stride_b;
                uint8_t * b_నిలువు_zp = ఖండం_type_has_zp<BLOC_TYPE>() ? b_నిలువు : శూన్యసూచిక;

                int64_t n_blk_real = 0;
                కొరకు (int64_t ni = 0; ni < n_లెక్క; ni += n_blk_real, b_నిలువు += n_blk_real * అడ్డువరుస_stride_b) {
                    n_blk_real = std::min(n_లెక్క - ni, n_blk);

                    uint8_t * a_అడ్డువరుస = quant_a_బఫరు + m_ప్రారంభం * అడ్డువరుస_stride_a;

                    దశాంశం * c_blk = output + m_ప్రారంభం * gemm_n + n_ప్రారంభం + ni;

                    int64_t అడ్డువరుసలు_మిగిలిన = m_లెక్క;

                    uint8_t * b_నిలువు_cur    = b_నిలువు;
                    uint8_t * b_నిలువు_zp_cur = b_నిలువు_zp;

                    అయితే (అడ్డువరుసలు_మిగిలిన > 0) {
                        స్వయంచాలక అడ్డువరుసలు_handled = gemm_kernel(b_blk_len, a_అడ్డువరుస, b_నిలువు_cur, b_నిలువు_zp_cur, c_blk,
                                                        అడ్డువరుసలు_మిగిలిన, n_blk_real, b_k_blks, gemm_n);

                        c_blk += అడ్డువరుసలు_handled * gemm_n;
                        a_అడ్డువరుస += అడ్డువరుసలు_handled * అడ్డువరుస_stride_a;

                        అడ్డువరుసలు_మిగిలిన -= అడ్డువరుసలు_handled;
                    }
                }
            }
        }
    }

    శూన్య forward_mul_mat_id(ggml_compute_params * params, ggml_tensor * op) {
        constexpr size_t a_blk_len = INTER_కొలత;
        constexpr size_t b_blk_len = INTER_కొలత;

        స్థిరరాశి ggml_tensor * src0 = op->src[0];
        స్థిరరాశి ggml_tensor * src1 = op->src[1];
        స్థిరరాశి ggml_tensor * ids  = op->src[2];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_ద్విమాన_OP_LOCALS

        పూర్ణాంకం ith = params->ith;
        పూర్ణాంకం nth = params->nth;

        // row groups
        స్థిరరాశి పూర్ణాంకం n_ids = ids->ne[0];  // n_expert_used
        స్థిరరాశి పూర్ణాంకం n_as  = ne02;        // n_expert

        నిర్మాణం mmid_అడ్డువరుస_mapping {
            int32_t i1;
            int32_t i2;
        };

        spacemit_kernels::quantize_a_అడ్డువరుస_def           quantize_a_అడ్డువరుస_i8;
        spacemit_kernels::gemm_kernel_quantize_def     gemm_kernel;
        spacemit_kernels::moe_gemm_kernel_quantize_def moe_gemm_kernel_m2;
        తార్కిక                                           సమితి_kernel_impl = అబద్ధం;
        size_t                                         ఖండం_stride_a  = spacemit_kernels::q8_blk_కొలత(QK4_0);

#ఒకవేళ defined(RISCV64_SPACEMIT_IME2)
        ఒకవేళ (!సమితి_kernel_impl && (global_spine_env_info.use_ime2)) {
            quantize_a_అడ్డువరుస_i8 = spacemit_kernels::rvv::quantize_a_అడ్డువరుస_i8;
            ఖండం_stride_a    = spacemit_kernels::q8_blk_కొలత(QK4_0, నిజం);

            ఒకవేళ constexpr (std::is_same_v<BLOC_TYPE, ఖండం_q6_K> || std::is_same_v<BLOC_TYPE, ఖండం_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i8;
                సమితి_kernel_impl = నిజం;
            } లేకపోతే ఒకవేళ constexpr (std::is_same_v<BLOC_TYPE, ఖండం_q4_0> || std::is_same_v<BLOC_TYPE, ఖండం_q4_1> ||
                                 std::is_same_v<BLOC_TYPE, ఖండం_q4_K>) {
                ఒకవేళ constexpr (INTER_కొలత == 256) {
                    gemm_kernel       = spacemit_kernels::ime2::gemm_kernel_i8i4_hp;
                    quantize_a_అడ్డువరుస_i8 = spacemit_kernels::rvv::quantize_a_అడ్డువరుస_i8_hp;
                    ఖండం_stride_a    = spacemit_kernels::q8_hp_blk_కొలత(a_blk_len, నిజం, నిజం);
                    సమితి_kernel_impl   = నిజం;
                } లేకపోతే {
                    gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i4;
                    moe_gemm_kernel_m2 = spacemit_kernels::ime2::moe_m2_gemm_kernel_i8i4;
                    quantize_a_అడ్డువరుస_i8  = spacemit_kernels::rvv::quantize_a_అడ్డువరుస_i8;
                    ఖండం_stride_a     = spacemit_kernels::q8_blk_కొలత(a_blk_len, నిజం);
                    సమితి_kernel_impl    = నిజం;
                }
            } లేకపోతే ఒకవేళ constexpr (std::is_same_v<BLOC_TYPE, ఖండం_q2_K>) {
                quantize_a_అడ్డువరుస_i8 = spacemit_kernels::rvv::quantize_a_అడ్డువరుస_i8k;
                ఖండం_stride_a    = spacemit_kernels::q8k_blk_కొలత(a_blk_len);
                gemm_kernel       = spacemit_kernels::ime2::gemm_kernel_i8i2k;
                సమితి_kernel_impl   = నిజం;
            } లేకపోతే ఒకవేళ constexpr (std::is_same_v<BLOC_TYPE, ఖండం_q3_K>) {
                quantize_a_అడ్డువరుస_i8 = spacemit_kernels::rvv::quantize_a_అడ్డువరుస_i8k;
                ఖండం_stride_a    = spacemit_kernels::q8k_blk_కొలత(a_blk_len);
                gemm_kernel       = spacemit_kernels::ime2::gemm_kernel_i8i3k;
                సమితి_kernel_impl   = నిజం;
            } లేకపోతే ఒకవేళ constexpr (std::is_same_v<BLOC_TYPE, ఖండం_mxfp4>) {
                gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8mxfp4;
                moe_gemm_kernel_m2 = spacemit_kernels::ime2::moe_m2_gemm_kernel_i8mxfp4;
                సమితి_kernel_impl    = నిజం;
            } లేకపోతే ఒకవేళ constexpr (std::is_same_v<BLOC_TYPE, ఖండం_q5_1> || std::is_same_v<BLOC_TYPE, ఖండం_q5_K> ||
                                 std::is_same_v<BLOC_TYPE, ఖండం_q5_0>) {
                gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i5;
                moe_gemm_kernel_m2 = spacemit_kernels::ime2::moe_m2_gemm_kernel_i8i5;
                సమితి_kernel_impl    = నిజం;
            }
        }
#అయితే_ముగింపు

#ఒకవేళ defined(RISCV64_SPACEMIT_IME1)
        ఒకవేళ (!సమితి_kernel_impl && (global_spine_env_info.use_ime1)) {
            quantize_a_అడ్డువరుస_i8 = spacemit_kernels::ime1::quantize_a_అడ్డువరుస_i8;

            ఒకవేళ constexpr (std::is_same_v<BLOC_TYPE, ఖండం_q4_0> || std::is_same_v<BLOC_TYPE, ఖండం_q4_1> ||
                          std::is_same_v<BLOC_TYPE, ఖండం_q4_K>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i4;
                సమితి_kernel_impl = నిజం;
            } లేకపోతే ఒకవేళ constexpr (std::is_same_v<BLOC_TYPE, ఖండం_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i8;
                సమితి_kernel_impl = నిజం;
            }
        }
#అయితే_ముగింపు
        ఒకవేళ (!సమితి_kernel_impl) {
            GGML_ABORT("no kernel implementation found for the block type");
        }

        స్థిరరాశి size_t a_k_blks = spacemit_kernels::div_విడత_పైకి(ne10, a_blk_len);
        స్థిరరాశి size_t b_k_blks = spacemit_kernels::div_విడత_పైకి(ne10, b_blk_len);

        స్థిరరాశి size_t nbw1                = a_k_blks * ఖండం_stride_a;
        స్థిరరాశి size_t nbw2                = ne11 * nbw1;
        స్థిరరాశి size_t nbw3                = nbw2 * ne12;
        స్థిరరాశి size_t gemm_workspace_కొలత = GGML_PAD(nbw3, alignof(int64_t));

        స్థిరరాశి uintptr_t ws_ptr         = reinterpret_cast<uintptr_t>(params->wdata);
        స్వయంచాలక *          quant_a_బఫరు = reinterpret_cast<uint8_t *>(ws_ptr);

        ఒకవేళ (ne11 == 1) {
            కొరకు (int64_t ii = ith; ii < ne12 * a_k_blks; ii += nth) {
                int64_t i12       = ii / a_k_blks;
                int64_t ak_blk_id = ii % a_k_blks;
                quantize_a_అడ్డువరుస_i8(a_blk_len, (దశాంశం *) ((అక్షరం *) src1->దత్తాంశం + i12 * nb12) + ak_blk_id * a_blk_len,
                                  a_blk_len, quant_a_బఫరు + i12 * nbw2 + ak_blk_id * ఖండం_stride_a);
            }
        } లేకపోతే {
            కొరకు (int64_t ii = ith; ii < ne12 * ne11; ii += nth) {
                int64_t i12 = ii / ne11;
                int64_t i11 = ii % ne11;
                quantize_a_అడ్డువరుస_i8(a_blk_len, (దశాంశం *) ((అక్షరం *) src1->దత్తాంశం + i12 * nb12 + i11 * nb11), ne10,
                                  quant_a_బఫరు + i12 * nbw2 + i11 * nbw1);
            }
        }

#నిర్వచించు MMID_మాతృక_అడ్డువరుస(అడ్డువరుస_id, i1) మాతృక_అడ్డువరుసలు[(అడ్డువరుస_id) *ne12 + (i1)]

        int64_t *          మాతృక_అడ్డువరుస_counts       = (int64_t *) (ws_ptr + gemm_workspace_కొలత);
        int32_t *          valid_ep_లెక్క          = (int32_t *) (మాతృక_అడ్డువరుస_counts + n_as);
        int32_t *          valid_act_లెక్క         = (int32_t *) (valid_ep_లెక్క + 1);
        int64_t *          valid_మాతృక_అడ్డువరుస_counts = (int64_t *) (valid_act_లెక్క + 1);
        mmid_అడ్డువరుస_mapping * మాతృక_అడ్డువరుసలు             = (mmid_అడ్డువరుస_mapping *) (valid_మాతృక_అడ్డువరుస_counts + n_as);

        ఒకవేళ (ith == 0) {
            // initialize matrix_row_counts
            memset(మాతృక_అడ్డువరుస_counts, 0, n_as * పరిమాణం(int64_t));

            // group rows by src0 matrix
            కొరకు (int32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
                కొరకు (int32_t id = 0; id < n_ids; ++id) {
                    స్థిరరాశి int32_t i02 =
                        *(స్థిరరాశి int32_t *) ((స్థిరరాశి అక్షరం *) ids->దత్తాంశం + iid1 * ids->nb[1] + id * ids->nb[0]);

                    GGML_ASSERT(i02 >= 0 && i02 < n_as);

                    MMID_మాతృక_అడ్డువరుస(i02, మాతృక_అడ్డువరుస_counts[i02]) = { id, iid1 };
                    మాతృక_అడ్డువరుస_counts[i02] += 1;
                }
            }

            int32_t valid_ep_లెక్క_t  = 0;
            int32_t valid_act_లెక్క_t = 0;
            కొరకు (పూర్ణాంకం cur_a = 0; cur_a < n_as; ++cur_a) {
                స్థిరరాశి int64_t cne1 = మాతృక_అడ్డువరుస_counts[cur_a];
                ఒకవేళ (cne1 == 0) {
                    కొనసాగు;
                }
                valid_మాతృక_అడ్డువరుస_counts[valid_ep_లెక్క_t] = cur_a;
                valid_act_లెక్క_t += cne1;
                valid_ep_లెక్క_t += 1;
            }
            valid_ep_లెక్క[0]  = valid_ep_లెక్క_t;
            valid_act_లెక్క[0] = valid_act_లెక్క_t;
        }

        స్థిరరాశి int64_t barrier_idx = static_cast<int64_t>(ith / 2);

        GGML_ASSERT(global_spine_env_info.init_barrier != శూన్యసూచిక);
        GGML_ASSERT(barrier_idx < spine_init_barrier_లెక్క);
        spine_barrier_t * cur_barrier = &global_spine_env_info.init_barrier[barrier_idx];

        ggml_barrier(params->threadpool);

        స్థిరరాశి size_t అడ్డువరుస_stride_b      = b_k_blks * get_repacked_ఖండం_type_కొలత<BLOC_TYPE, INTER_కొలత, NB_నిలువుల_సంఖ్య>();
        స్థిరరాశి size_t expert_b_stride   = ne01 * అడ్డువరుస_stride_b;
        స్థిరరాశి size_t చొప్పున_nb_నిలువుల_సంఖ్య_wsize = NB_నిలువుల_సంఖ్య * అడ్డువరుస_stride_b;

        std::array<స్థిరరాశి uint8_t *, 2> src_workspaces;
        std::array<దశాంశం *, 2>         dst_workspaces;

        స్వయంచాలక *     tcm_బఫరు      = ggml::cpu::riscv64_spacemit::tls_context.tcm_బఫరు;
        స్థిరరాశి స్వయంచాలక tcm_బఫరు_కొలత = ggml::cpu::riscv64_spacemit::tls_context.tcm_బఫరు_కొలత;

        స్థిరరాశి స్వయంచాలక valid_ep_లెక్క_t  = valid_ep_లెక్క[0];
        స్థిరరాశి స్వయంచాలక valid_act_లెక్క_t = valid_act_లెక్క[0];

        పూర్ణాంకం nth_es = 1;
        పూర్ణాంకం nth_n  = nth;

        పూర్ణాంకం ith_es = ith % nth_es;
        పూర్ణాంకం ith_n  = (ith / nth_es) % nth_n;

        ఒకవేళ (valid_ep_లెక్క_t % nth == 0 && tcm_బఫరు != శూన్యసూచిక && valid_ep_లెక్క_t == n_as &&
            valid_act_లెక్క_t == n_as && చొప్పున_nb_నిలువుల_సంఖ్య_wsize <= tcm_బఫరు_కొలత) {
            కొరకు (int64_t valid_id = ith; valid_id < valid_ep_లెక్క_t; valid_id += nth) {
                స్థిరరాశి int64_t cur_a = valid_మాతృక_అడ్డువరుస_counts[valid_id];

                స్వయంచాలక * src0_cur = (uint8_t *) src0->దత్తాంశం + cur_a * expert_b_stride;

                mmid_అడ్డువరుస_mapping అడ్డువరుస_mapping = MMID_మాతృక_అడ్డువరుస(cur_a, 0);
                స్థిరరాశి పూర్ణాంకం        id          = అడ్డువరుస_mapping.i1;
                స్థిరరాశి int64_t    i11         = id % ne11;
                స్థిరరాశి int64_t    i12         = అడ్డువరుస_mapping.i2;
                స్థిరరాశి int64_t    i1          = id;
                స్థిరరాశి int64_t    i2          = i12;

                స్వయంచాలక *  src1_నిలువు = quant_a_బఫరు + (i11 * nbw1 + i12 * nbw2);
                దశాంశం * c_blk    = (దశాంశం *) ((అక్షరం *) dst->దత్తాంశం + (i1 * nb1 + i2 * nb2));

                uint8_t * a_అడ్డువరుస = src1_నిలువు;
                uint8_t * b_నిలువు = reinterpret_cast<uint8_t *>(tcm_బఫరు);
                ఒకవేళ ((nbw1 + చొప్పున_nb_నిలువుల_సంఖ్య_wsize) <= tcm_బఫరు_కొలత) {
                    a_అడ్డువరుస = (uint8_t *) tcm_బఫరు;
                    b_నిలువు = reinterpret_cast<uint8_t *>(tcm_బఫరు) + nbw1;
                }
                uint8_t * b_నిలువు_zp = ఖండం_type_has_zp<BLOC_TYPE>() ? b_నిలువు : శూన్యసూచిక;

                ఒకవేళ (ith % 2 == 0) {
                    spacemit_kernels::rvv::memcpy1d(b_నిలువు, reinterpret_cast<uint8_t *>(src0_cur), చొప్పున_nb_నిలువుల_సంఖ్య_wsize);

                    ఒకవేళ (a_అడ్డువరుస != src1_నిలువు) {
                        spacemit_kernels::rvv::memcpy1d(a_అడ్డువరుస, src1_నిలువు, nbw1);
                    }
                }

                spine_barrier_wait(cur_barrier);

                ఒకవేళ (ith % 2 != 0) {
                    ఒకవేళ (a_అడ్డువరుస != src1_నిలువు) {
                        spacemit_kernels::rvv::memcpy1d(a_అడ్డువరుస, src1_నిలువు, nbw1);
                    }

                    spacemit_kernels::rvv::memcpy1d(b_నిలువు, reinterpret_cast<uint8_t *>(src0_cur), చొప్పున_nb_నిలువుల_సంఖ్య_wsize);
                }

                int64_t nb_real = std::min(ne01, NB_నిలువుల_సంఖ్య);
                కొరకు (int64_t ni = 0; ni < ne01; ni += NB_నిలువుల_సంఖ్య) {
                    ఒకవేళ (ith % 2 != 0) {
                        spine_barrier_wait(cur_barrier);
                    }

                    gemm_kernel(b_blk_len, a_అడ్డువరుస, b_నిలువు, b_నిలువు_zp, c_blk + ni, 1, nb_real, b_k_blks, ne01);

                    ఒకవేళ (ith % 2 == 0) {
                        spine_barrier_wait(cur_barrier);
                    }

                    స్థిరరాశి int64_t తదుపరి_ni = ni + NB_నిలువుల_సంఖ్య;
                    ఒకవేళ (తదుపరి_ni < ne01) {
                        nb_real = std::min(ne01 - తదుపరి_ni, NB_నిలువుల_సంఖ్య);
                        spacemit_kernels::rvv::memcpy1d(
                            b_నిలువు, reinterpret_cast<uint8_t *>(src0_cur) + తదుపరి_ni * అడ్డువరుస_stride_b, చొప్పున_nb_నిలువుల_సంఖ్య_wsize);
                    }
                }
            }
        } లేకపోతే {
            కొరకు (int64_t valid_id = ith_es; valid_id < valid_ep_లెక్క_t; valid_id += nth_es) {
                స్థిరరాశి int64_t cur_a = valid_మాతృక_అడ్డువరుస_counts[valid_id];
                స్థిరరాశి int64_t cne1  = మాతృక_అడ్డువరుస_counts[cur_a];

                int64_t src1_cur_ప్రారంభం = 0;
                int64_t src1_cur_చివర   = cne1;

                int64_t src0_cur_ప్రారంభం = (ith_n * ne01) / nth_n;
                int64_t src0_cur_చివర   = MIN(((ith_n + 1) * ne01) / nth_n, ne01);

                ఒకవేళ (src1_cur_ప్రారంభం >= src1_cur_చివర || src0_cur_ప్రారంభం >= src0_cur_చివర) {
                    కొనసాగు;
                }

                src0_cur_ప్రారంభం =
                    (src0_cur_ప్రారంభం % NB_నిలువుల_సంఖ్య) ? src0_cur_ప్రారంభం + NB_నిలువుల_సంఖ్య - (src0_cur_ప్రారంభం % NB_నిలువుల_సంఖ్య) : src0_cur_ప్రారంభం;
                src0_cur_చివర =
                    (src0_cur_చివర % NB_నిలువుల_సంఖ్య) ? src0_cur_చివర + NB_నిలువుల_సంఖ్య - (src0_cur_చివర % NB_నిలువుల_సంఖ్య) : src0_cur_చివర;

                స్వయంచాలక *    src0_cur = (uint8_t *) src0->దత్తాంశం + cur_a * expert_b_stride + src0_cur_ప్రారంభం * అడ్డువరుస_stride_b;
                uint8_t * b_నిలువు_zp = ఖండం_type_has_zp<BLOC_TYPE>() ? src0_cur : శూన్యసూచిక;

                size_t extra_tcm_బఫరు_కొలత = tcm_బఫరు_కొలత;
                శూన్య * extra_tcm_బఫరు      = tcm_బఫరు;
                ఒకవేళ (tcm_బఫరు != శూన్యసూచిక && (src1_cur_చివర - src1_cur_ప్రారంభం) >= 4 &&
                    (src0_cur_చివర - src0_cur_ప్రారంభం) * అడ్డువరుస_stride_b <= tcm_బఫరు_కొలత) {
                    spacemit_kernels::rvv::memcpy1d(tcm_బఫరు, src0_cur,
                                                    (src0_cur_చివర - src0_cur_ప్రారంభం) * అడ్డువరుస_stride_b);
                    src0_cur = reinterpret_cast<uint8_t *>(tcm_బఫరు);
                    b_నిలువు_zp = ఖండం_type_has_zp<BLOC_TYPE>() ? src0_cur : శూన్యసూచిక;
                    extra_tcm_బఫరు_కొలత -= (src0_cur_చివర - src0_cur_ప్రారంభం) * అడ్డువరుస_stride_b;
                    extra_tcm_బఫరు = reinterpret_cast<శూన్య *>(reinterpret_cast<uintptr_t>(tcm_బఫరు) +
                                                                (src0_cur_చివర - src0_cur_ప్రారంభం) * అడ్డువరుస_stride_b);
                }

                పూర్ణాంకం ir1 = src1_cur_ప్రారంభం;

                ఒకవేళ (extra_tcm_బఫరు_కొలత >= nbw1 && extra_tcm_బఫరు != శూన్యసూచిక) {
                    int64_t quant_a_tile_కొలత = extra_tcm_బఫరు_కొలత / nbw1;
                    చేయి {
                        quant_a_tile_కొలత = MIN(quant_a_tile_కొలత, src1_cur_చివర - ir1);

                        uint8_t * quant_a_tile_బఫరు = reinterpret_cast<uint8_t *>(extra_tcm_బఫరు);

                        పూర్ణాంకం iir1 = ir1;
                        కొరకు (; iir1 < (ir1 + quant_a_tile_కొలత); ++iir1) {
                            mmid_అడ్డువరుస_mapping అడ్డువరుస_mapping = MMID_మాతృక_అడ్డువరుస(cur_a, iir1);

                            స్థిరరాశి పూర్ణాంకం id = అడ్డువరుస_mapping.i1;  // selected expert index

                            స్థిరరాశి int64_t i11 = id % ne11;
                            స్థిరరాశి int64_t i12 = అడ్డువరుస_mapping.i2;  // row index in src1

                            స్వయంచాలక * src1_నిలువు = quant_a_బఫరు + (i11 * nbw1 + i12 * nbw2);
                            spacemit_kernels::rvv::memcpy1d(quant_a_tile_బఫరు, src1_నిలువు, nbw1);
                            quant_a_tile_బఫరు = quant_a_tile_బఫరు + nbw1;
                        }

                        quant_a_tile_బఫరు = reinterpret_cast<uint8_t *>(extra_tcm_బఫరు);
                        iir1                = ir1;

                        ఒకవేళ (moe_gemm_kernel_m2 != శూన్యసూచిక) {
                            కొరకు (; iir1 < (ir1 + quant_a_tile_కొలత - 1); iir1 += 2, quant_a_tile_బఫరు += 2 * nbw1) {
                                mmid_అడ్డువరుస_mapping అడ్డువరుస_mapping_0 = MMID_మాతృక_అడ్డువరుస(cur_a, iir1);
                                mmid_అడ్డువరుస_mapping అడ్డువరుస_mapping_1 = MMID_మాతృక_అడ్డువరుస(cur_a, iir1 + 1);

                                src_workspaces[0] = quant_a_tile_బఫరు;
                                src_workspaces[1] = quant_a_tile_బఫరు + nbw1;

                                dst_workspaces[0] =
                                    (దశాంశం *) ((అక్షరం *) dst->దత్తాంశం + (అడ్డువరుస_mapping_0.i1 * nb1 + అడ్డువరుస_mapping_0.i2 * nb2)) +
                                    src0_cur_ప్రారంభం;
                                dst_workspaces[1] = (దశాంశం *) ((అక్షరం *) dst->దత్తాంశం +
                                                               ((అడ్డువరుస_mapping_1.i1) * nb1 + (అడ్డువరుస_mapping_1.i2) * nb2)) +
                                                    src0_cur_ప్రారంభం;
                                moe_gemm_kernel_m2(b_blk_len, src_workspaces.దత్తాంశం(), src0_cur, b_నిలువు_zp,
                                                   dst_workspaces.దత్తాంశం(), 1, src0_cur_చివర - src0_cur_ప్రారంభం, b_k_blks,
                                                   ne01);
                            }
                        }

                        కొరకు (; iir1 < (ir1 + quant_a_tile_కొలత); iir1++, quant_a_tile_బఫరు += nbw1) {
                            mmid_అడ్డువరుస_mapping అడ్డువరుస_mapping_0 = MMID_మాతృక_అడ్డువరుస(cur_a, iir1);

                            gemm_kernel(
                                b_blk_len, quant_a_tile_బఫరు, src0_cur, b_నిలువు_zp,
                                (దశాంశం *) ((అక్షరం *) dst->దత్తాంశం + (అడ్డువరుస_mapping_0.i1 * nb1 + అడ్డువరుస_mapping_0.i2 * nb2)) +
                                    src0_cur_ప్రారంభం,
                                1, src0_cur_చివర - src0_cur_ప్రారంభం, b_k_blks, ne01);
                        }

                        ir1 += quant_a_tile_కొలత;
                    } అయితే (ir1 < src1_cur_చివర);
                } లేకపోతే {
                    ఒకవేళ (moe_gemm_kernel_m2 != శూన్యసూచిక) {
                        కొరకు (; ir1 < src1_cur_చివర - 1; ir1 += 2) {
                            కొరకు (పూర్ణాంకం iir1 = 0; iir1 < 2; ++iir1) {
                                mmid_అడ్డువరుస_mapping అడ్డువరుస_mapping = MMID_మాతృక_అడ్డువరుస(cur_a, ir1 + iir1);

                                స్థిరరాశి పూర్ణాంకం id = అడ్డువరుస_mapping.i1;  // selected expert index

                                స్థిరరాశి int64_t i11 = id % ne11;
                                స్థిరరాశి int64_t i12 = అడ్డువరుస_mapping.i2;  // row index in src1

                                స్థిరరాశి int64_t i1 = id;               // selected expert index
                                స్థిరరాశి int64_t i2 = i12;              //అడ్డువరుస

                                src_workspaces[iir1] = quant_a_బఫరు + (i11 * nbw1 + i12 * nbw2);

                                dst_workspaces[iir1] =
                                    (దశాంశం *) ((అక్షరం *) dst->దత్తాంశం + (i1 * nb1 + i2 * nb2)) + src0_cur_ప్రారంభం;
                            }

                            moe_gemm_kernel_m2(b_blk_len, src_workspaces.దత్తాంశం(), src0_cur, b_నిలువు_zp,
                                               dst_workspaces.దత్తాంశం(), 1, src0_cur_చివర - src0_cur_ప్రారంభం, b_k_blks, ne01);
                        }
                    }

                    కొరకు (; ir1 < src1_cur_చివర; ir1++) {
                        mmid_అడ్డువరుస_mapping అడ్డువరుస_mapping = MMID_మాతృక_అడ్డువరుస(cur_a, ir1);

                        స్థిరరాశి పూర్ణాంకం id = అడ్డువరుస_mapping.i1;  // selected expert index

                        స్థిరరాశి int64_t i11 = id % ne11;
                        స్థిరరాశి int64_t i12 = అడ్డువరుస_mapping.i2;  // row index in src1

                        స్థిరరాశి int64_t i1 = id;               // selected expert index
                        స్థిరరాశి int64_t i2 = i12;              //అడ్డువరుస

                        స్వయంచాలక * src1_నిలువు = quant_a_బఫరు + (i11 * nbw1 + i12 * nbw2);

                        gemm_kernel(b_blk_len, src1_నిలువు, src0_cur, b_నిలువు_zp,
                                    (దశాంశం *) ((అక్షరం *) dst->దత్తాంశం + (i1 * nb1 + i2 * nb2)) + src0_cur_ప్రారంభం, 1,
                                    src0_cur_చివర - src0_cur_ప్రారంభం, b_k_blks, ne01);
                    }
                }
            }
        }
#నిర్వచనం_రద్దు MMID_మాతృక_అడ్డువరుస
    }

    పూర్ణాంకం repack(ggml_tensor * t, స్థిరరాశి శూన్య * దత్తాంశం, size_t దత్తాంశం_కొలత) override {
        GGML_LOG_DEBUG("%s: repack tensor %s with %s_%dx%d\n", __func__, t->name, ggml_type_name(t->type),
                       (పూర్ణాంకం) NB_నిలువుల_సంఖ్య, (పూర్ణాంకం) INTER_కొలత);
        తిరిగి ggml::cpu::riscv64_spacemit::repack<BLOC_TYPE, INTER_కొలత, NB_నిలువుల_సంఖ్య>(t, దత్తాంశం, దత్తాంశం_కొలత);
    }
};

తరగతి tensor_traits_common : బహిరంగ tensor_traits_పునాది {
    తార్కిక work_కొలత(పూర్ణాంకం n_దారాలు, స్థిరరాశి ggml_tensor * op, size_t & size) override {
        ఎంపిక (op->op) {
            స్థితి GGML_OP_FLASH_ATTN_EXT:
                {
                    స్థిరరాశి పూర్ణాంకం     n_tasks = n_దారాలు;
                    స్థిరరాశి int64_t neq2    = op->src[0]->ne[2];  // number of query heads
                    స్థిరరాశి int64_t DK      = op->src[1]->ne[0];
                    స్థిరరాశి int64_t DV      = op->src[2]->ne[0];  // DV

                    // Tiled flash attention scratch (tile sizes defined in common.h)
                    // Per-thread: Q_q + KQ + mask + VKQ32 + V32 + K_f32 + padding
                    size_t prefill = పరిమాణం(దశాంశం) *
                                     (GGML_FA_TILE_Q * DK + 2 * GGML_FA_TILE_Q * GGML_FA_TILE_KV + GGML_FA_TILE_Q * DV +
                                      GGML_FA_TILE_KV * DV + GGML_FA_TILE_KV * DK) *
                                     n_tasks;

                    // Decode path: n_kv_chunks = n_tasks (one chunk per thread)
                    // Per-thread: VKQ accmulator (DV), partial M, partial S + intra-thread scratch for V, Q and VKQ
                    size_t n_chunks = n_tasks;
                    size_t decode   = పరిమాణం(దశాంశం) * (neq2 * n_chunks * (2 + DV) + n_tasks * (DK + 2 * DV));

                    size = MAX(prefill, decode);
                }
                తిరిగి నిజం;
            అప్రమేయ:
                ఆపు;
        }
        తిరిగి అబద్ధం;
    }

    తార్కిక compute_forward(ggml_compute_params * params, ggml_tensor * op) override {
        ఎంపిక (op->op) {
            స్థితి GGML_OP_NORM:
                ఎంపిక (op->src[0]->type) {
                    స్థితి GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_norm_f32(params, op);
                        తిరిగి నిజం;
                    అప్రమేయ:
                        GGML_ABORT("fatal error");
                }
            స్థితి GGML_OP_RMS_NORM:
                ఎంపిక (op->src[0]->type) {
                    స్థితి GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_rms_norm_f32(params, op);
                        తిరిగి నిజం;
                    అప్రమేయ:
                        GGML_ABORT("fatal error");
                }
            స్థితి GGML_OP_కూడు:
                ఎంపిక (op->src[0]->type) {
                    స్థితి GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_ద్విమాన<GGML_OP_కూడు, దశాంశం>(params, op);
                        తిరిగి నిజం;
                    స్థితి GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_ద్విమాన<GGML_OP_కూడు, _Float16>(params, op);
                        తిరిగి నిజం;
                    అప్రమేయ:
                        ggml_compute_forward_కూడు(params, op);
                        తిరిగి నిజం;
                }
            స్థితి GGML_OP_SUB:
                ఎంపిక (op->src[0]->type) {
                    స్థితి GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_ద్విమాన<GGML_OP_SUB, దశాంశం>(params, op);
                        తిరిగి నిజం;
                    స్థితి GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_ద్విమాన<GGML_OP_SUB, _Float16>(params, op);
                        తిరిగి నిజం;
                    అప్రమేయ:
                        ggml_compute_forward_sub(params, op);
                        తిరిగి నిజం;
                }
            స్థితి GGML_OP_MUL:
                ఎంపిక (op->src[0]->type) {
                    స్థితి GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_ద్విమాన<GGML_OP_MUL, దశాంశం>(params, op);
                        తిరిగి నిజం;
                    స్థితి GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_ద్విమాన<GGML_OP_MUL, _Float16>(params, op);
                        తిరిగి నిజం;
                    అప్రమేయ:
                        ggml_compute_forward_mul(params, op);
                        తిరిగి నిజం;
                }
            స్థితి GGML_OP_DIV:
                ఎంపిక (op->src[0]->type) {
                    స్థితి GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_ద్విమాన<GGML_OP_DIV, దశాంశం>(params, op);
                        తిరిగి నిజం;
                    స్థితి GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_ద్విమాన<GGML_OP_DIV, _Float16>(params, op);
                        తిరిగి నిజం;
                    అప్రమేయ:
                        ggml_compute_forward_div(params, op);
                        తిరిగి నిజం;
                }
            స్థితి GGML_OP_FLASH_ATTN_EXT:
                forward_flash_attn_ext_f16(params, op);
                తిరిగి నిజం;
            స్థితి GGML_OP_CONT:
                {
                    స్థిరరాశి ggml_tensor * src0 = op->src[0];
                    ఒకవేళ (op->type == src0->type && op->nb[0] != src0->nb[0] && op->nb[0] == src0->nb[1] &&
                        op->ne[3] * op->ne[2] * op->nb[2] == src0->ne[3] * src0->ne[2] * src0->nb[2]) {
                        spacemit_kernels::rvv::forward_cont_with_క్రమమార్పు(params, op);
                    } లేకపోతే {
                        ggml_compute_forward_cont(params, op);
                    }
                    తిరిగి నిజం;
                }
            స్థితి GGML_OP_CPY:
                {
                    స్థిరరాశి ggml_tensor * src0 = op->src[0];
                    ఒకవేళ (op->type == src0->type && op->nb[0] == src0->nb[1] && src0->nb[0] != src0->nb[1] &&
                        ggml_nelements(src0) == ggml_nelements(op)) {
                        spacemit_kernels::rvv::forward_cpy_with_క్రమమార్పు(params, op);
                    } లేకపోతే {
                        ggml_compute_forward_cpy(params, op);
                    }
                    తిరిగి నిజం;
                }
            స్థితి GGML_OP_REPEAT:
                {
                    స్థిరరాశి తార్కిక అడ్డువరుసలు_equal         = ggml_nrows(op->src[0]) == ggml_nrows(op);
                    స్థిరరాశి తార్కిక ప్రసారం_or_equal = op->src[0]->ne[0] == 1 || op->src[0]->ne[0] == op->ne[0];

                    ఒకవేళ (అడ్డువరుసలు_equal && ప్రసారం_or_equal) {
                        ఎంపిక (op->src[0]->type) {
                            స్థితి GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_repeat_nrows<int32_t>(params, op);
                                తిరిగి నిజం;
                            స్థితి GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_repeat_nrows<int16_t>(params, op);
                                తిరిగి నిజం;
                            అప్రమేయ:
                                ఆపు;
                        }
                    }

                    ఒకవేళ (op->src[0]->ne[1] == 1 && op->src[0]->ne[0] == op->ne[0]) {
                        ఎంపిక (op->src[0]->type) {
                            స్థితి GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_repeat_dim1<int32_t>(params, op);
                                తిరిగి నిజం;
                            స్థితి GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_repeat_dim1<int16_t>(params, op);
                                తిరిగి నిజం;
                            అప్రమేయ:
                                ఆపు;
                        }
                    }

                    ggml_compute_forward_repeat(params, op);
                }
                తిరిగి నిజం;
            స్థితి GGML_OP_కూడిక_అడ్డువరుసలు:
                {
                    ఒకవేళ (op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32) {
                        spacemit_kernels::rvv::forward_కూడిక_అడ్డువరుసలు<దశాంశం>(params, op);
                    } లేకపోతే {
                        ggml_compute_forward_కూడిక_అడ్డువరుసలు(params, op);
                    }
                }
                తిరిగి నిజం;
            స్థితి GGML_OP_GET_అడ్డువరుసలు:
                {
                    ఒకవేళ (op->src[0]->type == op->type) {
                        ఎంపిక (op->src[0]->type) {
                            స్థితి GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_get_అడ్డువరుసలు<int32_t>(params, op);
                                తిరిగి నిజం;
                            స్థితి GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_get_అడ్డువరుసలు<int16_t>(params, op);
                                తిరిగి నిజం;
                            అప్రమేయ:
                                ఆపు;
                        }
                    }

                    ggml_compute_forward_get_అడ్డువరుసలు(params, op);
                }
                తిరిగి నిజం;
            స్థితి GGML_OP_CONCAT:
                {
                    స్థిరరాశి int32_t dim = ggml_get_op_params_i32(op, 0);
                    ఒకవేళ (dim == 0 && op->type == op->src[0]->type) {
                        ఎంపిక (op->src[0]->type) {
                            స్థితి GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_concat<int32_t>(params, op);
                                తిరిగి నిజం;
                            స్థితి GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_concat<int16_t>(params, op);
                                తిరిగి నిజం;
                            అప్రమేయ:
                                ఆపు;
                        }
                    }

                    ggml_compute_forward_concat(params, op);
                }
                తిరిగి నిజం;
            // TODO For GGML_OP_GATED_DELTA_NET
            // case GGML_OP_GATED_DELTA_NET:
            //     return true;
            అప్రమేయ:
                ఆపు;
        }
        తిరిగి అబద్ధం;
    }

    శూన్య forward_flash_attn_ext_f16(స్థిరరాశి ggml_compute_params * params, ggml_tensor * dst) {
        స్థిరరాశి ggml_tensor * q = dst->src[0];
        స్థిరరాశి ggml_tensor * k = dst->src[1];
        స్థిరరాశి ggml_tensor * v = dst->src[2];

        GGML_TENSOR_LOCALS(int64_t, neq, q, ne)
        GGML_TENSOR_LOCALS(size_t, nbq, q, nb)
        GGML_TENSOR_LOCALS(int64_t, nek, k, ne)
        GGML_TENSOR_LOCALS(size_t, nbk, k, nb)
        GGML_TENSOR_LOCALS(int64_t, nev, v, ne)
        GGML_TENSOR_LOCALS(size_t, nbv, v, nb)
        GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
        GGML_TENSOR_LOCALS(size_t, nb, dst, nb)

        స్థిరరాశి int64_t DK = nek0;
        స్థిరరాశి int64_t DV = nev0;

        స్థిరరాశి తార్కిక supported_prec  = (dst->op_params[3] == GGML_PREC_F32 || dst->op_params[3] == GGML_PREC_DEFAULT);
        స్థిరరాశి తార్కిక supported_types = (q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16);
        స్థిరరాశి తార్కిక supported_shape = (DK > 0 && DK <= 128 && DV > 0 && DV <= 128);
        స్థిరరాశి తార్కిక supported_vlen  = (__riscv_vlenb() == 128);

        ఒకవేళ (!(supported_prec && supported_types && supported_shape && supported_vlen)) {
            ggml_compute_forward_flash_attn_ext(params, dst);
            తిరిగి;
        }

        // total rows in q
        స్థిరరాశి int64_t nr = neq1 * neq2 * neq3;

        // rows per thread
        స్థిరరాశి పూర్ణాంకం ith = params->ith;
        స్థిరరాశి పూర్ణాంకం nth = params->nth;

        స్థిర constexpr int64_t Q_TILE_SZ = ggml_fa_tile_config::Q;
        స్థిరరాశి తార్కిక               use_tiled = !params->use_ref && (neq1 >= Q_TILE_SZ);

        // 4x chunks per thread
        // int     nth_scaled = nth * 4;
        // int64_t chunk_size = (nr + nth_scaled - 1) / nth_scaled;
        // int64_t nchunk     = (nr + chunk_size - 1) / chunk_size;

        // if (nth == 1 || nchunk < nth) {
        //     nchunk = nth;
        // }

        int64_t nchunk = nth;

        ఒకవేళ (ith == 0) {
            // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
            ggml_threadpool_chunk_సమితి(params->threadpool, nth);
        }

        ggml_barrier(params->threadpool);

        // The number of elements in each chunk
        స్థిరరాశి int64_t dr = (nr + nchunk - 1) / nchunk;

        // The first chunk comes from our thread_id, the rest will get auto-assigned.
        పూర్ణాంకం current_chunk = ith;

        అయితే (current_chunk < nchunk) {
            స్థిరరాశి int64_t ir0 = dr * current_chunk;
            స్థిరరాశి int64_t ir1 = MIN(ir0 + dr, nr);

            ఒకవేళ (use_tiled) {
                spacemit_kernels::rvv::forward_flash_attn_ext_f16_tiled_vlen1024_vf16(
                    params, dst, ir0, ir1, ggml::cpu::riscv64_spacemit::tls_context.tcm_బఫరు,
                    ggml::cpu::riscv64_spacemit::tls_context.tcm_బఫరు_కొలత);
            } లేకపోతే {
                spacemit_kernels::rvv::forward_flash_attn_ext_f16_one_chunk_vlen1024_vf16(
                    params, dst, ir0, ir1, ggml::cpu::riscv64_spacemit::tls_context.tcm_బఫరు,
                    ggml::cpu::riscv64_spacemit::tls_context.tcm_బఫరు_కొలత);
            }

            current_chunk = ggml_threadpool_chunk_కూడు(params->threadpool, 1);
        }
    }

    పూర్ణాంకం repack(ggml_tensor * t, స్థిరరాశి శూన్య * దత్తాంశం, size_t దత్తాంశం_కొలత) override {
        జ్ఞాపకం_నకలు(t->దత్తాంశం, దత్తాంశం, దత్తాంశం_కొలత);
        తిరిగి 0;
    }
};

// Impl By IME1
స్థిర స్థిరరాశి tensor_traits<ఖండం_q4_0, 32, 16>  q4_0_16x32_q8_0;
స్థిర స్థిరరాశి tensor_traits<ఖండం_q4_1, 32, 16>  q4_1_16x32_q8_0;
స్థిర స్థిరరాశి tensor_traits<ఖండం_q4_K, 32, 16>  q4_k_16x32_q8_0;
స్థిర స్థిరరాశి tensor_traits<ఖండం_q8_0, 32, 16>  q8_0_16x32_q8_0;
// Impl By IME2
స్థిర స్థిరరాశి tensor_traits<ఖండం_q2_K, 256, 32> q2_k_32x256_q8_0;
స్థిర స్థిరరాశి tensor_traits<ఖండం_q3_K, 256, 32> q3_k_32x256_q8_0;
స్థిర స్థిరరాశి tensor_traits<ఖండం_q4_0, 32, 32>  q4_0_32x32_q8_0;
స్థిర స్థిరరాశి tensor_traits<ఖండం_q4_1, 32, 32>  q4_1_32x32_q8_0;
స్థిర స్థిరరాశి tensor_traits<ఖండం_q4_0, 256, 32> q4_0_32x256_q8_0;
స్థిర స్థిరరాశి tensor_traits<ఖండం_q4_1, 256, 32> q4_1_32x256_q8_0;
స్థిర స్థిరరాశి tensor_traits<ఖండం_q4_K, 32, 32>  q4_k_32x32_q8_0;
స్థిర స్థిరరాశి tensor_traits<ఖండం_q6_K, 32, 32>  q6_k_32x32_q8_0;
స్థిర స్థిరరాశి tensor_traits<ఖండం_q8_0, 32, 32>  q8_0_32x32_q8_0;
స్థిర స్థిరరాశి tensor_traits<ఖండం_mxfp4, 32, 32> mxfp4_32x32_q8_0;
స్థిర స్థిరరాశి tensor_traits<ఖండం_q5_K, 32, 32>  q5_k_32x32_q8_0;
స్థిర స్థిరరాశి tensor_traits<ఖండం_q5_1, 32, 32>  q5_1_32x32_q8_0;
స్థిర స్థిరరాశి tensor_traits<ఖండం_q5_0, 32, 32>  q5_0_32x32_q8_0;
// Impl By RVV
స్థిర స్థిరరాశి tensor_traits_common               rvv_impl;

}  // namespace ggml::cpu::riscv64_spacemit

స్థిర స్థిరరాశి ggml::cpu::tensor_traits * ggml_riscv64_spacemit_get_optimal_repack_type(స్థిరరాశి ggml_tensor * cur) {
    ఎంపిక (cur->type) {
        స్థితి GGML_TYPE_Q2_K:
            {
#ఒకవేళ defined(RISCV64_SPACEMIT_IME2)
                ఒకవేళ (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    తిరిగి &ggml::cpu::riscv64_spacemit::q2_k_32x256_q8_0;
                }
#అయితే_ముగింపు
            }
            ఆపు;
        స్థితి GGML_TYPE_Q3_K:
            {
#ఒకవేళ defined(RISCV64_SPACEMIT_IME2)
                ఒకవేళ (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    తిరిగి &ggml::cpu::riscv64_spacemit::q3_k_32x256_q8_0;
                }
#అయితే_ముగింపు
            }
            ఆపు;
        స్థితి GGML_TYPE_Q4_0:
            {
#ఒకవేళ defined(RISCV64_SPACEMIT_IME2)
                ఒకవేళ (cur->ne[1] % 32 == 0 && cur->ne[0] % 256 == 0 &&
                    (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    తిరిగి &ggml::cpu::riscv64_spacemit::q4_0_32x256_q8_0;
                }

                ఒకవేళ (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    తిరిగి &ggml::cpu::riscv64_spacemit::q4_0_32x32_q8_0;
                }
#అయితే_ముగింపు

#ఒకవేళ defined(RISCV64_SPACEMIT_IME1)
                ఒకవేళ (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    తిరిగి &ggml::cpu::riscv64_spacemit::q4_0_16x32_q8_0;
                }
#అయితే_ముగింపు
            }
            ఆపు;
        స్థితి GGML_TYPE_Q4_1:
            {
#ఒకవేళ defined(RISCV64_SPACEMIT_IME2)
                // TODO
                // if (cur->ne[1] % 32 == 0 && cur->ne[0] % 256 == 0 &&
                //     (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                //     return &ggml::cpu::riscv64_spacemit::q4_1_32x256_q8_0;
                // }

                ఒకవేళ (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    తిరిగి &ggml::cpu::riscv64_spacemit::q4_1_32x32_q8_0;
                }
#అయితే_ముగింపు

#ఒకవేళ defined(RISCV64_SPACEMIT_IME1)
                ఒకవేళ (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    తిరిగి &ggml::cpu::riscv64_spacemit::q4_1_16x32_q8_0;
                }
#అయితే_ముగింపు
            }
            ఆపు;
        స్థితి GGML_TYPE_Q4_K:
            {
#ఒకవేళ defined(RISCV64_SPACEMIT_IME2)
                ఒకవేళ (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    తిరిగి &ggml::cpu::riscv64_spacemit::q4_k_32x32_q8_0;
                }
#అయితే_ముగింపు

#ఒకవేళ defined(RISCV64_SPACEMIT_IME1)
                ఒకవేళ (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    తిరిగి &ggml::cpu::riscv64_spacemit::q4_k_16x32_q8_0;
                }
#అయితే_ముగింపు
            }
            ఆపు;
        స్థితి GGML_TYPE_Q6_K:
            {
#ఒకవేళ defined(RISCV64_SPACEMIT_IME2)
                ఒకవేళ ((ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    తిరిగి &ggml::cpu::riscv64_spacemit::q6_k_32x32_q8_0;
                }
#అయితే_ముగింపు
            }
            ఆపు;
        స్థితి GGML_TYPE_Q8_0:
            {
#ఒకవేళ defined(RISCV64_SPACEMIT_IME2)
                ఒకవేళ ((ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    తిరిగి &ggml::cpu::riscv64_spacemit::q8_0_32x32_q8_0;
                }
#అయితే_ముగింపు

#ఒకవేళ defined(RISCV64_SPACEMIT_IME1)
                ఒకవేళ (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    తిరిగి &ggml::cpu::riscv64_spacemit::q8_0_16x32_q8_0;
                }
#అయితే_ముగింపు
            }
            ఆపు;
        స్థితి GGML_TYPE_MXFP4:
            {
#ఒకవేళ defined(RISCV64_SPACEMIT_IME2)
                // TODO
                // if (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                //     return &ggml::cpu::riscv64_spacemit::mxfp4_32x32_q8_0;
                // }
#అయితే_ముగింపు
            }
            ఆపు;
        స్థితి GGML_TYPE_Q5_K:
            {
#ఒకవేళ defined(RISCV64_SPACEMIT_IME2)
                ఒకవేళ (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    తిరిగి &ggml::cpu::riscv64_spacemit::q5_k_32x32_q8_0;
                }
#అయితే_ముగింపు
            }
            ఆపు;
        స్థితి GGML_TYPE_Q5_1:
            {
#ఒకవేళ defined(RISCV64_SPACEMIT_IME2)
                ఒకవేళ (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    తిరిగి &ggml::cpu::riscv64_spacemit::q5_1_32x32_q8_0;
                }
#అయితే_ముగింపు
            }
            ఆపు;
        స్థితి GGML_TYPE_Q5_0:
            {
#ఒకవేళ defined(RISCV64_SPACEMIT_IME2)
                ఒకవేళ (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    తిరిగి &ggml::cpu::riscv64_spacemit::q5_0_32x32_q8_0;
                }
#అయితే_ముగింపు
            }
            ఆపు;
        అప్రమేయ:
            ఆపు;
    }

    తిరిగి శూన్యసూచిక;
}

స్థిర గణన ggml_స్థితి ggml_backend_riscv64_spacemit_బఫరు_init_tensor(ggml_backend_బఫరు_t బఫరు,
                                                                         ggml_tensor *         tensor) {
    tensor->extra =
        (శూన్య *) const_cast<ggml::cpu::tensor_traits *>(ggml_riscv64_spacemit_get_optimal_repack_type(tensor));

    GGML_వాడని(బఫరు);

    తిరిగి GGML_స్థితి_SUCCESS;
}

స్థిర శూన్య ggml_backend_riscv64_spacemit_బఫరు_free_బఫరు(ggml_backend_బఫరు_t బఫరు) {
    GGML_ASSERT(బఫరు);

    శూన్య * పునాది = బఫరు->context;
    ఒకవేళ (పునాది == శూన్యసూచిక) {
        తిరిగి;
    }

    ggml::cpu::riscv64_spacemit::spine_mem_pool_free(పునాది);
}

స్థిర శూన్య * ggml_backend_riscv64_spacemit_బఫరు_get_పునాది(ggml_backend_బఫరు_t బఫరు) {
    GGML_ASSERT(బఫరు);

    శూన్య * పునాది = బఫరు->context;
    GGML_ASSERT(పునాది != శూన్యసూచిక);
    తిరిగి పునాది;
}

స్థిర శూన్య ggml_backend_riscv64_spacemit_బఫరు_memset_tensor(ggml_backend_బఫరు_t బఫరు,
                                                               ggml_tensor *         tensor,
                                                               uint8_t               value,
                                                               size_t                స్థానభ్రంశం,
                                                               size_t                size) {
    GGML_ASSERT(tensor);
    memset((అక్షరం *) tensor->దత్తాంశం + స్థానభ్రంశం, value, size);

    GGML_వాడని(బఫరు);
}

స్థిర శూన్య ggml_backend_riscv64_spacemit_బఫరు_clear(ggml_backend_బఫరు_t బఫరు, uint8_t value) {
    GGML_ASSERT(బఫరు);

    శూన్య * పునాది = బఫరు->context;
    GGML_ASSERT(పునాది != శూన్యసూచిక);
    memset(పునాది, value, బఫరు->size);
}

స్థిర శూన్య ggml_backend_riscv64_spacemit_బఫరు_సమితి_tensor(ggml_backend_బఫరు_t బఫరు,
                                                            ggml_tensor *         tensor,
                                                            స్థిరరాశి శూన్య *          దత్తాంశం,
                                                            size_t                స్థానభ్రంశం,
                                                            size_t                size) {
    GGML_ASSERT(స్థానభ్రంశం == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    స్వయంచాలక tensor_traits = (ggml::cpu::riscv64_spacemit::tensor_traits_పునాది *) tensor->extra;
    ఒకవేళ (tensor_traits) {
        స్వయంచాలక OK = tensor_traits->repack(tensor, దత్తాంశం, size);
        GGML_ASSERT(OK == 0);
    }

    GGML_వాడని(బఫరు);
}

స్థిర స్థిరరాశి ggml_backend_బఫరు_i ggml_backend_riscv64_spacemit_బఫరు_i = {
    /* .free_buffer     = */ ggml_backend_riscv64_spacemit_బఫరు_free_బఫరు,
    /* .get_base        = */ ggml_backend_riscv64_spacemit_బఫరు_get_పునాది,
    /* .init_tensor     = */ ggml_backend_riscv64_spacemit_బఫరు_init_tensor,
    /* .memset_tensor   = */ ggml_backend_riscv64_spacemit_బఫరు_memset_tensor,
    /* .set_tensor      = */ ggml_backend_riscv64_spacemit_బఫరు_సమితి_tensor,
    /* .get_tensor      = */ శూన్యసూచిక,
    /* .set_tensor_2d   = */ శూన్యసూచిక,
    /* .get_tensor_2d   = */ శూన్యసూచిక,
    /* .cpy_tensor      = */ శూన్యసూచిక,
    /* .clear           = */ ggml_backend_riscv64_spacemit_బఫరు_clear,
    /* .reset           = */ శూన్యసూచిక,
};

స్థిర స్థిరరాశి అక్షరం * ggml_backend_cpu_riscv64_spacemit_బఫరు_type_get_name(ggml_backend_బఫరు_type_t buft) {
    తిరిగి "CPU_RISCV64_SPACEMIT";

    GGML_వాడని(buft);
}

స్థిర ggml_backend_బఫరు_t ggml_backend_cpu_riscv64_spacemit_బఫరు_type_alloc_బఫరు(ggml_backend_బఫరు_type_t buft,
                                                                                        size_t size) {
    శూన్య * పునాది = ggml::cpu::riscv64_spacemit::spine_mem_pool_alloc(size, 64);
    ఒకవేళ (పునాది == శూన్యసూచిక) {
        తిరిగి శూన్యసూచిక;
    }

    తిరిగి ggml_backend_బఫరు_init(buft, ggml_backend_riscv64_spacemit_బఫరు_i, పునాది, size);
}

స్థిర size_t ggml_backend_cpu_riscv64_spacemit_బఫరు_type_get_alignment(ggml_backend_బఫరు_type_t buft) {
    తిరిగి 64;

    GGML_వాడని(buft);
}

స్థిర size_t ggml_backend_cpu_riscv64_spacemit_nbytes(ggml_backend_బఫరు_type_t buft, స్థిరరాశి ggml_tensor * tensor) {
    కొరకు (పూర్ణాంకం i = 0; i < GGML_MAX_DIMS; ++i) {
        ఒకవేళ (tensor->ne[i] <= 0) {
            తిరిగి 0;
        }
    }

    GGML_వాడని(buft);

    స్థిరరాశి స్వయంచాలక plain_nbytes = [&]() {
        size_t మొత్తం = ggml_type_కొలత(tensor->type);
        కొరకు (పూర్ణాంకం i = 0; i < GGML_MAX_DIMS; ++i) {
            మొత్తం += (tensor->ne[i] - 1) * tensor->nb[i];
        }
        తిరిగి మొత్తం;
    };

    స్థిరరాశి size_t blck_కొలత = ggml_blck_కొలత(tensor->type);
    ఒకవేళ (blck_కొలత == 1) {
        తిరిగి plain_nbytes();
    }

    స్థిరరాశి size_t అడ్డువరుస_nbytes = tensor->ne[0] * tensor->nb[0] / blck_కొలత;

    స్థిరరాశి స్వయంచాలక కూడు_strided_nbytes = [&](size_t మొత్తం, size_t src_ఖండం_కొలత, size_t dst_ఖండం_కొలత) {
        కొరకు (పూర్ణాంకం i = 1; i < GGML_MAX_DIMS; ++i) {
            మొత్తం += (tensor->ne[i] - 1) * (tensor->nb[i] / src_ఖండం_కొలత) * dst_ఖండం_కొలత;
        }
        తిరిగి మొత్తం;
    };

    స్థిరరాశి స్వయంచాలక remap_ఖండం_nbytes = [&](size_t src_ఖండం_కొలత, size_t dst_ఖండం_కొలత, int64_t padded_అడ్డువరుసలు = 0) {
        GGML_ASSERT(అడ్డువరుస_nbytes % src_ఖండం_కొలత == 0);

        size_t మొత్తం =
            కూడు_strided_nbytes((అడ్డువరుస_nbytes / src_ఖండం_కొలత) * dst_ఖండం_కొలత, src_ఖండం_కొలత, dst_ఖండం_కొలత);

        ఒకవేళ (padded_అడ్డువరుసలు > 0 && tensor->ne[1] % padded_అడ్డువరుసలు != 0) {
            మొత్తం += (padded_అడ్డువరుసలు - tensor->ne[1] % padded_అడ్డువరుసలు) * (tensor->nb[1] / src_ఖండం_కొలత) * dst_ఖండం_కొలత;
        }

        తిరిగి మొత్తం;
    };

    size_t nbytes = అడ్డువరుస_nbytes;
    ఎంపిక (tensor->type) {
        స్థితి GGML_TYPE_Q4_K:
            nbytes = remap_ఖండం_nbytes(పరిమాణం(ఖండం_q4_K), పరిమాణం(ఖండం_q4_1) * 8);
            ఆపు;
        స్థితి GGML_TYPE_Q6_K:
            nbytes = remap_ఖండం_nbytes(పరిమాణం(ఖండం_q6_K), పరిమాణం(ఖండం_q8_0) * 8, 32);
            ఆపు;
        స్థితి GGML_TYPE_Q8_0:
            nbytes = remap_ఖండం_nbytes(పరిమాణం(ఖండం_q8_0), పరిమాణం(ఖండం_q8_0), 32);
            ఆపు;
        స్థితి GGML_TYPE_Q2_K:
            nbytes = remap_ఖండం_nbytes(పరిమాణం(ఖండం_q2_K), పరిమాణం(spacemit_kernels::nrow_ఖండం_q2_k<1>));
            ఆపు;
        స్థితి GGML_TYPE_Q3_K:
            nbytes = remap_ఖండం_nbytes(పరిమాణం(ఖండం_q3_K), పరిమాణం(spacemit_kernels::nrow_ఖండం_q3_k<1>));
            ఆపు;
        స్థితి GGML_TYPE_MXFP4:
            nbytes = remap_ఖండం_nbytes(పరిమాణం(ఖండం_mxfp4), పరిమాణం(spacemit_kernels::nrow_ఖండం_mxfp4<1>));
            ఆపు;
        స్థితి GGML_TYPE_Q5_K:
            nbytes = remap_ఖండం_nbytes(పరిమాణం(ఖండం_q5_K), పరిమాణం(spacemit_kernels::nrow_ఖండం_q5_1<1>) * 8);
            ఆపు;
        స్థితి GGML_TYPE_Q5_1:
            nbytes = remap_ఖండం_nbytes(పరిమాణం(ఖండం_q5_1), పరిమాణం(spacemit_kernels::nrow_ఖండం_q5_1<1>));
            ఆపు;
        స్థితి GGML_TYPE_Q5_0:
            nbytes = remap_ఖండం_nbytes(పరిమాణం(ఖండం_q5_0), పరిమాణం(spacemit_kernels::nrow_ఖండం_q5_0<1>));
            ఆపు;
        అప్రమేయ:
            nbytes = కూడు_strided_nbytes(అడ్డువరుస_nbytes, 1, 1);
            ఆపు;
    }

    తిరిగి nbytes;
}

నామాకాశ ggml::cpu::riscv64_spacemit {

తరగతి extra_బఫరు_type : ggml::cpu::extra_బఫరు_type {
    తార్కిక supports_op(ggml_backend_dev_t, స్థిరరాశి ggml_tensor * op) override {
        ఎంపిక (op->op) {
            స్థితి GGML_OP_MUL_MAT:
                ఒకవేళ (op->src[0]->బఫరు && (ggml_n_dims(op->src[0]) == 2) &&
                    op->src[0]->బఫరు->buft == ggml_backend_cpu_riscv64_spacemit_బఫరు_type() &&
                    ggml_riscv64_spacemit_get_optimal_repack_type(op->src[0])) {
                    ఒకవేళ (op->src[1]->బఫరు && !ggml_backend_buft_is_host(op->src[1]->బఫరు->buft)) {
                        తిరిగి అబద్ధం;
                    }
                    ఒకవేళ (op->src[1]->type == GGML_TYPE_F32) {
                        తిరిగి నిజం;
                    }
                }
                ఆపు;
            స్థితి GGML_OP_MUL_MAT_ID:
                ఒకవేళ (op->src[0]->బఫరు && (ggml_n_dims(op->src[0]) == 3) &&
                    op->src[0]->బఫరు->buft == ggml_backend_cpu_riscv64_spacemit_బఫరు_type() &&
                    ggml_riscv64_spacemit_get_optimal_repack_type(op->src[0])) {
                    ఒకవేళ (op->src[1]->బఫరు && !ggml_backend_buft_is_host(op->src[1]->బఫరు->buft)) {
                        తిరిగి అబద్ధం;
                    }
                    ఒకవేళ (op->src[1]->type == GGML_TYPE_F32) {
                        తిరిగి నిజం;
                    }
                }
                ఆపు;
            అప్రమేయ:
                // GGML_ABORT("fatal error");
                ఆపు;
        }
        తిరిగి అబద్ధం;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(స్థిరరాశి ggml_tensor * op) override {
        ఎంపిక (op->op) {
            స్థితి GGML_OP_MUL_MAT:
            స్థితి GGML_OP_MUL_MAT_ID:
                ఒకవేళ (op->src[0]->బఫరు && op->src[0]->బఫరు->buft == ggml_backend_cpu_riscv64_spacemit_బఫరు_type()) {
                    తిరిగి (ggml::cpu::tensor_traits *) op->src[0]->extra;
                }
                ఆపు;
            స్థితి GGML_OP_NORM:
            స్థితి GGML_OP_RMS_NORM:
            స్థితి GGML_OP_కూడు:
            స్థితి GGML_OP_SUB:
            స్థితి GGML_OP_MUL:
            స్థితి GGML_OP_DIV:
            స్థితి GGML_OP_FLASH_ATTN_EXT:
            స్థితి GGML_OP_CONT:
            స్థితి GGML_OP_CPY:
            స్థితి GGML_OP_REPEAT:
            స్థితి GGML_OP_కూడిక_అడ్డువరుసలు:
            స్థితి GGML_OP_GET_అడ్డువరుసలు:
            స్థితి GGML_OP_CONCAT:
                // case GGML_OP_GATED_DELTA_NET:
                తిరిగి (ggml::cpu::tensor_traits *) (&ggml::cpu::riscv64_spacemit::rvv_impl);
            అప్రమేయ:
                // GGML_ABORT("fatal error");
                ఆపు;
        }

        తిరిగి శూన్యసూచిక;
    }
};

}  // namespace ggml::cpu::riscv64_spacemit

ggml_backend_బఫరు_type_t ggml_backend_cpu_riscv64_spacemit_బఫరు_type(శూన్య) {
    స్థిర ggml_backend_బఫరు_type ggml_backend_cpu_బఫరు_type_riscv64_spacemit = {
  /* .iface    = */
        {
         /* .get_name           = */ ggml_backend_cpu_riscv64_spacemit_బఫరు_type_get_name,
         /* .alloc_buffer       = */ ggml_backend_cpu_riscv64_spacemit_బఫరు_type_alloc_బఫరు,
         /* .alloc_buffer_n     = */ ఖాళీ,
         /* .get_alignment      = */ ggml_backend_cpu_riscv64_spacemit_బఫరు_type_get_alignment,
         /* .get_max_size       = */ శూన్యసూచిక,
         /* .get_alloc_size     = */ ggml_backend_cpu_riscv64_spacemit_nbytes,
         /* .get_alloc_size_n   = */ ఖాళీ,
         /* .is_host            = */ శూన్యసూచిక,
         },
 /* .device  = */
        ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
 /* .context = */
        కొత్త ggml::cpu::riscv64_spacemit::extra_బఫరు_type(),
    };

    తిరిగి &ggml_backend_cpu_బఫరు_type_riscv64_spacemit;
}

బాహ్య "C" {
స్థిర పూర్ణాంకం bind_ai_thread() {
    పూర్ణాంకం  fd, బైట్లు;
    అక్షరం str[32];

    fd = open("/proc/set_ai_thread", O_WRONLY);
    ఒకవేళ (fd < 0) {
        GGML_LOG_ERROR("try open /proc/set_ai_thread failed\n");
        తిరిగి -1;
    }

    snprintf(str, 16, "%d", 0);
    బైట్లు = లిఖ(fd, str, అక్షరశ్రేణి_పొడవు(str));
    ఒకవేళ (బైట్లు < 0) {
        GGML_LOG_ERROR("try write /proc/set_ai_thread failed\n");
        close(fd);
        తిరిగి -1;
    }

    close(fd);
    తిరిగి 0;
}

శూన్య ggml_backend_cpu_riscv64_spacemit_సమితి_numa_thread_affinity(పూర్ణాంకం thread_n) {
    పూర్ణాంకం cpu_id = sched_getcpu();
    ఒకవేళ (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2 &&
        !((1 << cpu_id) & ggml::cpu::riscv64_spacemit::global_spine_env_info.cpu_ముసుగు)) {
        GGML_PRINT_DEBUG("bind_ai_thread for thread %d, pid %d\n", thread_n, getpid());
        bind_ai_thread();
    }

    ఒకవేళ (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_tcm &&
        ggml::cpu::riscv64_spacemit::tls_context.cpu_id == -1) {
        CPU_ZERO(&(ggml::cpu::riscv64_spacemit::tls_context.cpuset));
        pthread_t    main_thread     = pthread_self();
        స్థిరరాశి స్వయంచాలక & perfer_core_ids = ggml::cpu::riscv64_spacemit::global_spine_env_info.perfer_core_ids;
        ఒకవేళ (thread_n < 0 || static_cast<size_t>(thread_n) >= perfer_core_ids.size()) {
            GGML_ABORT("thread_n %d exceeds perfer_core_ids size %zu\n", thread_n, perfer_core_ids.size());
        }
        స్వయంచాలక perfer_cpu_id = perfer_core_ids[static_cast<size_t>(thread_n)];
        CPU_SET(perfer_cpu_id, &(ggml::cpu::riscv64_spacemit::tls_context.cpuset));
        పూర్ణాంకం s =
            pthread_setaffinity_np(main_thread, పరిమాణం(cpu_set_t), &(ggml::cpu::riscv64_spacemit::tls_context.cpuset));
        ఒకవేళ (s != 0) {
            GGML_ABORT("set thread affinity error for thread_n %d, cpu_id %d\n", thread_n, perfer_cpu_id);
        }

        పూర్ణాంకం ai_cpu_id = perfer_cpu_id - ggml::cpu::riscv64_spacemit::global_spine_env_info.aicpu_id_స్థానభ్రంశం;
        ggml::cpu::riscv64_spacemit::tls_context.cpu_id = ai_cpu_id;
        ggml::cpu::riscv64_spacemit::tls_context.tcm_బఫరు =
            ggml::cpu::riscv64_spacemit::spine_mem_pool_tcm_mem_get(ai_cpu_id);
        ggml::cpu::riscv64_spacemit::tls_context.tcm_బఫరు_కొలత =
            ggml::cpu::riscv64_spacemit::global_spine_env_info.tcm_blk_కొలత;
    }

    ఒకవేళ (ggml::cpu::riscv64_spacemit::tls_context.tcm_బఫరు != శూన్యసూచిక) {
        శూన్య * rt =
            ggml::cpu::riscv64_spacemit::spine_mem_pool_tcm_mem_wait(ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        ఒకవేళ (rt == శూన్యసూచిక) {
            GGML_ABORT("wait tcm buffer failed for cpu_id: %d", ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        }
    }
}

శూన్య ggml_backend_cpu_riscv64_spacemit_clear_numa_thread_affinity_threaded(పూర్ణాంకం thread_n) {
    ఒకవేళ (ggml::cpu::riscv64_spacemit::tls_context.tcm_బఫరు != శూన్యసూచిక) {
        స్వయంచాలక rt = ggml::cpu::riscv64_spacemit::spine_mem_pool_tcm_mem_release(
            ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        ఒకవేళ (rt != 0) {
            GGML_ABORT("release tcm buffer failed for cpu_id: %d", ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        }
    }
}
}
