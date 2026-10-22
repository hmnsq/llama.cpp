#ifndef GGML_SYCL_FUSED_GEMM_HPP
#define GGML_SYCL_FUSED_GEMM_HPP

#অন্তর্ভাব "common.hpp"


// Shape and type gates for the kernels below. Device capability is separate: it needs a queue to ask.
স্থির constexpr পূর্ণসংখ্যা GGML_SYCL_FG_MAX_N = 64; // widest N taken; each shape covers it in BN-wide tiles

// weight formats the fused A stage decodes; K must cover whole stored blocks
constexpr যৌক্তিক ggml_sycl_fused_dequant_gemm_type_ok(ggml_type src0_type, int64_t K) {
    // iq4_nl and q8_0 store 32 values per block; every other format here is a 256-value superblock
    // that the A stage walks in steps of 32, so K must cover whole superblocks.
    যদি (src0_type == GGML_TYPE_IQ4_NL || src0_type == GGML_TYPE_Q8_0) {
        ফেরত K % 32 == 0;
    }
    ধ্রুবক যৌক্তিক superblock =
           src0_type == GGML_TYPE_Q4_K ||
           src0_type == GGML_TYPE_Q5_K ||
           src0_type == GGML_TYPE_Q6_K ||
           src0_type == GGML_TYPE_IQ3_S ||
           src0_type == GGML_TYPE_IQ4_XS ||
           src0_type == GGML_TYPE_IQ3_XXS ||
           src0_type == GGML_TYPE_IQ2_XXS ||
           src0_type == GGML_TYPE_IQ2_XS ||
           src0_type == GGML_TYPE_IQ2_S ||
           src0_type == GGML_TYPE_IQ1_S ||
           src0_type == GGML_TYPE_IQ1_M;
    ফেরত superblock && QK_K == 256 && K % QK_K == 0;
}

constexpr যৌক্তিক ggml_sycl_fused_dequant_gemm_shape_ok(ggml_type src0_type, int64_t M, int64_t N, int64_t K,
                                                     int64_t ldd) {
    ফেরত ggml_sycl_fused_dequant_gemm_type_ok(src0_type, K) && M > 0 && N > 0 && K > 0 &&
           N <= GGML_SYCL_FG_MAX_N &&
           M <= INT32_MAX && N <= INT32_MAX && K <= INT32_MAX && ldd <= INT32_MAX;
}

// grouped variant: the per-expert fused kernel is only worth it while each expert is narrow,
// so wider average slices are left to the per-expert library GEMM loop
constexpr যৌক্তিক ggml_sycl_grouped_dequant_gemm_shape_ok(ggml_type src0_type, int64_t M, int64_t K,
                                                       int64_t যোগ_rows, int64_t n_active) {
    ফেরত ggml_sycl_fused_dequant_gemm_shape_ok(src0_type, M, 1, K, M) && যোগ_rows > 0 &&
           যোগ_rows <= INT32_MAX && যোগ_rows <= n_active * GGML_SYCL_FG_MAX_N;
}

// Runtime type gate, kept out of the constexpr predicates above so those stay pure.
অন্তঃরেখ যৌক্তিক ggml_sycl_xmx_gather_type_enabled(ggml_type src0_type) {
    নির্বাচন (src0_type) {
        ক্ষেত্র GGML_TYPE_IQ4_NL:  ফেরত (g_ggml_sycl_xmx_gather_types & GGML_SYCL_XMX_GATHER_IQ4_NL  ) != 0;
        ক্ষেত্র GGML_TYPE_IQ3_S:   ফেরত (g_ggml_sycl_xmx_gather_types & GGML_SYCL_XMX_GATHER_IQ3_S   ) != 0;
        ক্ষেত্র GGML_TYPE_IQ4_XS:  ফেরত (g_ggml_sycl_xmx_gather_types & GGML_SYCL_XMX_GATHER_IQ4_XS  ) != 0;
        ক্ষেত্র GGML_TYPE_IQ3_XXS: ফেরত (g_ggml_sycl_xmx_gather_types & GGML_SYCL_XMX_GATHER_IQ3_XXS ) != 0;
        ক্ষেত্র GGML_TYPE_IQ2_XXS: ফেরত (g_ggml_sycl_xmx_gather_types & GGML_SYCL_XMX_GATHER_IQ2_XXS ) != 0;
        ক্ষেত্র GGML_TYPE_IQ2_XS:  ফেরত (g_ggml_sycl_xmx_gather_types & GGML_SYCL_XMX_GATHER_IQ2_XS  ) != 0;
        ক্ষেত্র GGML_TYPE_IQ2_S:   ফেরত (g_ggml_sycl_xmx_gather_types & GGML_SYCL_XMX_GATHER_IQ2_S   ) != 0;
        ক্ষেত্র GGML_TYPE_IQ1_S:   ফেরত (g_ggml_sycl_xmx_gather_types & GGML_SYCL_XMX_GATHER_IQ1_S   ) != 0;
        ক্ষেত্র GGML_TYPE_IQ1_M:   ফেরত (g_ggml_sycl_xmx_gather_types & GGML_SYCL_XMX_GATHER_IQ1_M   ) != 0;
        ক্ষেত্র GGML_TYPE_Q8_0:    ফেরত (g_ggml_sycl_xmx_gather_types & GGML_SYCL_XMX_GATHER_Q8_0    ) != 0;
        ক্ষেত্র GGML_TYPE_Q4_K:    ফেরত (g_ggml_sycl_xmx_gather_types & GGML_SYCL_XMX_GATHER_Q4_K    ) != 0;
        ক্ষেত্র GGML_TYPE_Q5_K:    ফেরত (g_ggml_sycl_xmx_gather_types & GGML_SYCL_XMX_GATHER_Q5_K    ) != 0;
        ক্ষেত্র GGML_TYPE_Q6_K:    ফেরত (g_ggml_sycl_xmx_gather_types & GGML_SYCL_XMX_GATHER_Q6_K    ) != 0;
        পূর্বনির্ধারিত:          ফেরত মিথ্যা;
    }
}

// dst[n*ldd + m] = sum_k dequant(src0)[m*K + k] * src1[n*K + k], src1 is F32, F16 or BF16.
// The XMX combination is picked per call from the src1 type and its precision request src1_prec
// (op_params[3], [TAG_GGML_PREC]); the accumulator is f32, which meets any request.
// q8_0 and the k-quants are not handled here, only in the grouped path below.
// Returns false when the case is not handled (type, device, precision, or shape).
যৌক্তিক ggml_sycl_fused_dequant_gemm(ggml_type src0_type, যৌক্তিক src0_reordered, ধ্রুবক শূন্য * src0, ধ্রুবক শূন্য * src1, ggml_type src1_type,
                                  int32_t src1_prec, দশমিক * dst, int64_t M, int64_t N, int64_t K, int64_t ldd,
                                  ggml_sycl_pool & pool, dpct::queue_ptr stream);

// One launch for every expert of a MUL_MAT_ID: rows of src1/dst are grouped by expert, expert e
// owns rows [expert_row_offsets[e], expert_row_offsets[e+1]) and reads its weights at
// src0_base + e*expert_stride. tiles is host scratch that must stay alive until the queue drains.
// reordered: each expert slice is in the reorder (SoA) layout of reorder_qw().
// dst[n*M + m] = sum_k dequant(src0_e)[m*K + k] * src1[n*K + k]
// Returns false when the case is not handled (type, layout, device, precision, or shape).
যৌক্তিক ggml_sycl_grouped_dequant_gemm(ggml_type src0_type, যৌক্তিক reordered, ধ্রুবক শূন্য * src0_base, size_t expert_stride,
                                    ধ্রুবক দশমিক * src1, int32_t src1_prec, দশমিক * dst,
                                    ধ্রুবক int64_t * expert_row_offsets, int64_t n_as, int64_t M, int64_t K,
                                    int64_t যোগ_rows, std::vector<ggml_sycl_gg_tile> & tiles,
                                    ggml_sycl_pool & pool, dpct::queue_ptr stream);

#endif // GGML_SYCL_FUSED_GEMM_HPP
