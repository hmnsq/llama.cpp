#ifndef GGML_SYCL_FUSED_GEMM_HPP
#define GGML_SYCL_FUSED_GEMM_HPP

#ଅନ୍ତର୍ଭାଵ "common.hpp"


// Shape and type gates for the kernels below. Device capability is separate: it needs a queue to ask.
ସ୍ଥିର constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା GGML_SYCL_FG_MAX_N = 64; // widest N taken; each shape covers it in BN-wide tiles

// weight formats the fused A stage decodes; K must cover whole stored blocks
constexpr ତାର୍କିକ ggml_sycl_fused_dequant_gemm_type_ok(ggml_type src0_type, int64_t K) {
    // iq4_nl and q8_0 store 32 values per block; every other format here is a 256-value superblock
    // that the A stage walks in steps of 32, so K must cover whole superblocks.
    ଯଦି (src0_type == GGML_TYPE_IQ4_NL || src0_type == GGML_TYPE_Q8_0) {
        ଫେରନ୍ତୁ K % 32 == 0;
    }
    ଅଚଳ ତାର୍କିକ superblock =
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
    ଫେରନ୍ତୁ superblock && QK_K == 256 && K % QK_K == 0;
}

constexpr ତାର୍କିକ ggml_sycl_fused_dequant_gemm_shape_ok(ggml_type src0_type, int64_t M, int64_t N, int64_t K,
                                                     int64_t ldd) {
    ଫେରନ୍ତୁ ggml_sycl_fused_dequant_gemm_type_ok(src0_type, K) && M > 0 && N > 0 && K > 0 &&
           N <= GGML_SYCL_FG_MAX_N &&
           M <= INT32_MAX && N <= INT32_MAX && K <= INT32_MAX && ldd <= INT32_MAX;
}

// grouped variant: the per-expert fused kernel is only worth it while each expert is narrow,
// so wider average slices are left to the per-expert library GEMM loop
constexpr ତାର୍କିକ ggml_sycl_grouped_dequant_gemm_shape_ok(ggml_type src0_type, int64_t M, int64_t K,
                                                       int64_t total_rows, int64_t n_active) {
    ଫେରନ୍ତୁ ggml_sycl_fused_dequant_gemm_shape_ok(src0_type, M, 1, K, M) && total_rows > 0 &&
           total_rows <= INT32_MAX && total_rows <= n_active * GGML_SYCL_FG_MAX_N;
}

// Runtime type gate, kept out of the constexpr predicates above so those stay pure.
ଧାଡ଼ିଭିତରେ ତାର୍କିକ ggml_sycl_xmx_gather_type_enabled(ggml_type src0_type) {
    ବାଛ (src0_type) {
        ସ୍ଥିତି GGML_TYPE_IQ4_NL:  ଫେରନ୍ତୁ (g_ggml_sycl_xmx_gather_types & GGML_SYCL_XMX_GATHER_IQ4_NL  ) != 0;
        ସ୍ଥିତି GGML_TYPE_IQ3_S:   ଫେରନ୍ତୁ (g_ggml_sycl_xmx_gather_types & GGML_SYCL_XMX_GATHER_IQ3_S   ) != 0;
        ସ୍ଥିତି GGML_TYPE_IQ4_XS:  ଫେରନ୍ତୁ (g_ggml_sycl_xmx_gather_types & GGML_SYCL_XMX_GATHER_IQ4_XS  ) != 0;
        ସ୍ଥିତି GGML_TYPE_IQ3_XXS: ଫେରନ୍ତୁ (g_ggml_sycl_xmx_gather_types & GGML_SYCL_XMX_GATHER_IQ3_XXS ) != 0;
        ସ୍ଥିତି GGML_TYPE_IQ2_XXS: ଫେରନ୍ତୁ (g_ggml_sycl_xmx_gather_types & GGML_SYCL_XMX_GATHER_IQ2_XXS ) != 0;
        ସ୍ଥିତି GGML_TYPE_IQ2_XS:  ଫେରନ୍ତୁ (g_ggml_sycl_xmx_gather_types & GGML_SYCL_XMX_GATHER_IQ2_XS  ) != 0;
        ସ୍ଥିତି GGML_TYPE_IQ2_S:   ଫେରନ୍ତୁ (g_ggml_sycl_xmx_gather_types & GGML_SYCL_XMX_GATHER_IQ2_S   ) != 0;
        ସ୍ଥିତି GGML_TYPE_IQ1_S:   ଫେରନ୍ତୁ (g_ggml_sycl_xmx_gather_types & GGML_SYCL_XMX_GATHER_IQ1_S   ) != 0;
        ସ୍ଥିତି GGML_TYPE_IQ1_M:   ଫେରନ୍ତୁ (g_ggml_sycl_xmx_gather_types & GGML_SYCL_XMX_GATHER_IQ1_M   ) != 0;
        ସ୍ଥିତି GGML_TYPE_Q8_0:    ଫେରନ୍ତୁ (g_ggml_sycl_xmx_gather_types & GGML_SYCL_XMX_GATHER_Q8_0    ) != 0;
        ସ୍ଥିତି GGML_TYPE_Q4_K:    ଫେରନ୍ତୁ (g_ggml_sycl_xmx_gather_types & GGML_SYCL_XMX_GATHER_Q4_K    ) != 0;
        ସ୍ଥିତି GGML_TYPE_Q5_K:    ଫେରନ୍ତୁ (g_ggml_sycl_xmx_gather_types & GGML_SYCL_XMX_GATHER_Q5_K    ) != 0;
        ସ୍ଥିତି GGML_TYPE_Q6_K:    ଫେରନ୍ତୁ (g_ggml_sycl_xmx_gather_types & GGML_SYCL_XMX_GATHER_Q6_K    ) != 0;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:          ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
}

// dst[n*ldd + m] = sum_k dequant(src0)[m*K + k] * src1[n*K + k], src1 is F32, F16 or BF16.
// The XMX combination is picked per call from the src1 type and its precision request src1_prec
// (op_params[3], [TAG_GGML_PREC]); the accumulator is f32, which meets any request.
// q8_0 and the k-quants are not handled here, only in the grouped path below.
// Returns false when the case is not handled (type, device, precision, or shape).
ତାର୍କିକ ggml_sycl_fused_dequant_gemm(ggml_type src0_type, ତାର୍କିକ src0_reordered, ଅଚଳ ଶୂନ୍ୟ * src0, ଅଚଳ ଶୂନ୍ୟ * src1, ggml_type src1_type,
                                  int32_t src1_prec, ଦଶମିକ * dst, int64_t M, int64_t N, int64_t K, int64_t ldd,
                                  ggml_sycl_pool & pool, dpct::queue_ptr stream);

// One launch for every expert of a MUL_MAT_ID: rows of src1/dst are grouped by expert, expert e
// owns rows [expert_row_offsets[e], expert_row_offsets[e+1]) and reads its weights at
// src0_base + e*expert_stride. tiles is host scratch that must stay alive until the queue drains.
// reordered: each expert slice is in the reorder (SoA) layout of reorder_qw().
// dst[n*M + m] = sum_k dequant(src0_e)[m*K + k] * src1[n*K + k]
// Returns false when the case is not handled (type, layout, device, precision, or shape).
ତାର୍କିକ ggml_sycl_grouped_dequant_gemm(ggml_type src0_type, ତାର୍କିକ reordered, ଅଚଳ ଶୂନ୍ୟ * src0_base, size_t expert_stride,
                                    ଅଚଳ ଦଶମିକ * src1, int32_t src1_prec, ଦଶମିକ * dst,
                                    ଅଚଳ int64_t * expert_row_offsets, int64_t n_as, int64_t M, int64_t K,
                                    int64_t total_rows, std::vector<ggml_sycl_gg_tile> & tiles,
                                    ggml_sycl_pool & pool, dpct::queue_ptr stream);

#endif // GGML_SYCL_FUSED_GEMM_HPP
