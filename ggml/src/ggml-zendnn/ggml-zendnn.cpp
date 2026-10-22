#ଅନ୍ତର୍ଭାଵ "ggml-zendnn.h"

#ଅନ୍ତର୍ଭାଵ "ggml-backend-impl.h"
#ଅନ୍ତର୍ଭାଵ "ggml-impl.h"

#define GGML_COMMON_DECL_CPP
#ଅନ୍ତର୍ଭାଵ "ggml-common.h"

#ଅନ୍ତର୍ଭାଵ "zendnnl.hpp"

#ଅନ୍ତର୍ଭାଵ <cstring>


ସଂରଚନା ggml_backend_zendnn_context {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_threads = GGML_DEFAULT_N_THREADS;
    std::unique_ptr<ଅକ୍ଷର[]> work_data;
    size_t work_size = 0;
};

template<typename T>
zendnnl::common::data_type_t ggml_to_zendnn_type() {
    ଯଦି constexpr (std::is_same_v<T, ଦଶମିକ>) {
        ଫେରନ୍ତୁ zendnnl::common::data_type_t::f32;
    } ଅନ୍ୟଥା ଯଦି constexpr (std::is_same_v<T, ggml_bf16_t>) {
        ଫେରନ୍ତୁ zendnnl::common::data_type_t::bf16;
    } ଅନ୍ୟଥା ଯଦି constexpr (std::is_same_v<T, block_q8_0>) {
        ଫେରନ୍ତୁ zendnnl::common::data_type_t::s8;
    } ଅନ୍ୟଥା {
        ଫେରନ୍ତୁ zendnnl::common::data_type_t::none;
    }
}

/**
 * Builds the matmul_params shared by ggml_zendnn_matmul() and ggml_zendnn_group_matmul():
 * dtype selection plus, for Q8_0 weights, dynamic-quant setup. Callers still need to set
 * quant_params.src_scale.dims themselves, since that depends on the batch size(s) in use.
 */
template <typename TA, typename TB, typename TC>
ସ୍ଥିର zendnnl::lowoha::matmul::matmul_params ggml_zendnn_make_matmul_params(ggml_backend_zendnn_context * ctx) {
    zendnnl::lowoha::matmul::matmul_params params;
    params.dtypes.src = ggml_to_zendnn_type<TB>();
    params.dtypes.wei = ggml_to_zendnn_type<TA>();
    params.dtypes.dst = ggml_to_zendnn_type<TC>();
    params.ଅଙ୍କ_threads = ctx->n_threads;

    ଯଦି constexpr (std::is_same_v<TA, block_q8_0>) {
        params.dtypes.compute = zendnnl::common::data_type_t::s8;
        params.dynamic_quant = ସତ୍ୟ;
        params.quant_params.src_scale.buff = nullptr;
        params.quant_params.src_scale.dt   = zendnnl::common::data_type_t::bf16;
        params.packing.pack_format_b = 1;
    }
    ଫେରନ୍ତୁ params;
}

/**
 * ZenDNN matmul: computes C = B * A.
 *
 * - A: weights, shape (k, m), column-major (each column is a weight vector for one output).
 * - B: input, shape (n, k), row-major (each row is an input sample).
 * - C: output, shape (n, m), row-major.
 *
 * Dimensions:
 *   m = output features (columns of C, columns of A)
 *   n = batch size      (rows of C, rows of B)
 *   k = inner dimension (columns of B, rows of A)
 */
template <typename TA, typename TB, typename TC>
ସ୍ଥିର ତାର୍କିକ ggml_zendnn_matmul(ggml_backend_zendnn_context * ctx, int64_t m, int64_t n, int64_t k,
                               ଅଚଳ TA * A, int64_t lda, ଅଚଳ TB * B, int64_t ldb, TC * C,
                               int64_t ldc) {

    zendnnl::lowoha::matmul::matmul_params params = ggml_zendnn_make_matmul_params<TA, TB, TC>(ctx);

    zendnnl::lowoha::matmul::matmul_batch_params_t batch_params;

    ଯଦି constexpr (std::is_same_v<TA, block_q8_0>) {
        params.quant_params.src_scale.dims = {n, k / QK8_0};
    }

    zendnnl::error_handling::status_t status = zendnnl::lowoha::matmul::matmul_direct(
        'r', ମିଥ୍ୟା, ସତ୍ୟ,   // row-major, don't transpose B, transpose A (because it's column-major)
        n,                  // M: rows of B and C
        m,                  // N: cols of A^T and C
        k,                  // K: cols of B, rows of A
        1.0f,               // alpha
        B, ldb,             // src: B[n,k]
        A, lda,             // weight: A[k,m] column-major (transposed)
        nullptr,            // bias
        0.0f,               // beta
        C, ldc,             // output C[n,m]
        ସତ୍ୟ,               // is_weights_const
        batch_params,       // batch_params
        params              // params
    );

    ଯଦି (status != zendnnl::error_handling::status_t::success) {
        GGML_LOG_ERROR("%s, ZenDNN matmul failed: status=%d\n", __func__, static_cast<ପୂର୍ଣ୍ଣସଂଖ୍ୟା>(status));
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ସ୍ଥିର ତାର୍କିକ ggml_zendnn_gemm(ggml_backend_zendnn_context * ctx, int64_t m, int64_t n, int64_t k,
                              ଅଚଳ ଶୂନ୍ୟ * A, int64_t lda, ଅଚଳ ଶୂନ୍ୟ * B, int64_t ldb, ଶୂନ୍ୟ * C,
                              int64_t ldc, ପୂର୍ଣ୍ଣସଂଖ୍ୟା Atype, ପୂର୍ଣ୍ଣସଂଖ୍ୟା Btype, ପୂର୍ଣ୍ଣସଂଖ୍ୟା Ctype) {

    ପ୍ରତିଜ୍ଞା(m >= 0);
    ପ୍ରତିଜ୍ଞା(n >= 0);
    ପ୍ରତିଜ୍ଞା(k >= 0);
    ପ୍ରତିଜ୍ଞା(lda >= k);
    ପ୍ରତିଜ୍ଞା(ldb >= k);
    ପ୍ରତିଜ୍ଞା(ldc >= m);

    // categorize types
    ବାଛ (Atype) {
        ସ୍ଥିତି GGML_TYPE_F32:
            ଯଦି (Btype != GGML_TYPE_F32 || Ctype != GGML_TYPE_F32)
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            ଫେରନ୍ତୁ ggml_zendnn_matmul<ଦଶମିକ, ଦଶମିକ, ଦଶମିକ>(
                ctx, m, n, k,
                (ଅଚଳ ଦଶମିକ *)A, lda,
                (ଅଚଳ ଦଶମିକ *)B, ldb,
                (ଦଶମିକ *)C, ldc);
        ସ୍ଥିତି GGML_TYPE_BF16:
            ଯଦି (Btype != GGML_TYPE_BF16)
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            ଯଦି (Ctype == GGML_TYPE_BF16)
                ଫେରନ୍ତୁ ggml_zendnn_matmul<ggml_bf16_t, ggml_bf16_t, ggml_bf16_t>(
                    ctx, m, n, k,
                    (ଅଚଳ ggml_bf16_t *)A, lda,
                    (ଅଚଳ ggml_bf16_t *)B, ldb,
                    (ggml_bf16_t *)C, ldc);
            ଯଦି (Ctype == GGML_TYPE_F32)
                ଫେରନ୍ତୁ ggml_zendnn_matmul<ggml_bf16_t, ggml_bf16_t, ଦଶମିକ>(
                    ctx, m, n, k,
                    (ଅଚଳ ggml_bf16_t *)A, lda,
                    (ଅଚଳ ggml_bf16_t *)B, ldb,
                    (ଦଶମିକ *)C, ldc);
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        ସ୍ଥିତି GGML_TYPE_Q8_0:
            ଯଦି (Btype != GGML_TYPE_F32 || Ctype != GGML_TYPE_F32)
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            ଫେରନ୍ତୁ ggml_zendnn_matmul<block_q8_0, ଦଶମିକ, ଦଶମିକ>(
                ctx, m, n, k,
                (ଅଚଳ block_q8_0 *)A, lda,
                (ଅଚଳ ଦଶମିକ *)B, ldb,
                (ଦଶମିକ *)C, ldc);
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            ଫେରନ୍ତୁ ମିଥ୍ୟା; // unsupported type
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_zendnn_compute_forward_mul_mat(
    ggml_backend_zendnn_context * ctx,
    ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];  // weights
    ଅଚଳ ggml_tensor * src1 = dst->src[1];  // inputs

    GGML_TENSOR_ଦ୍ୱିମିକ_OP_LOCALS

    ggml_type         ଅଚଳ vec_dot_type = src0->type;
    ggml_from_float_t ଅଚଳ from_float = ggml_get_type_traits(vec_dot_type)->from_float_ref;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == ଆକାର(ଦଶମିକ));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // broadcast factors
    ଅଚଳ int64_t r2 = ne12/ne02;
    ଅଚଳ int64_t r3 = ne13/ne03;

    ଶୂନ୍ୟ * work_data = ctx->work_data.get();

    // ZenDNN requires FP32 for dynamic quantization, so conversion is skipped
    ଯଦି (src1->type != vec_dot_type && src0->type != GGML_TYPE_Q8_0) {
        ଅଚଳ size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
        ଅଚଳ size_t nbw2 = nbw1 * ne11;
        ଅଚଳ size_t nbw3 = nbw2 * ne12;
        ଅଚଳ size_t desired_wsize = ne13 * nbw3;
        ଯଦି (ctx->work_size < desired_wsize) {
            ctx->work_data.reset(ନୂତନ ଅକ୍ଷର[desired_wsize]);
            ctx->work_size = desired_wsize;
        }
        work_data = ctx->work_data.get();

        // #pragma omp parallel for num_threads(ctx->n_threads)
        #pragma omp parallel ପାଇଁ collapse(3) ଅଙ୍କ_threads(ctx->n_threads) schedule(ସ୍ଥିର)
        ପାଇଁ (int64_t i13 = 0; i13 < ne13; ++i13) {
            ପାଇଁ (int64_t i12 = 0; i12 < ne12; ++i12) {
                ପାଇଁ (int64_t i11 = 0; i11 < ne11; ++i11) {
                    ଅଚଳ ଦଶମିକ * src1_f32 = (ଦଶମିକ *)((ଅକ୍ଷର *)src1->data + i11*nb11 + i12*nb12 + i13*nb13);
                    ଶୂନ୍ୟ * src1_conv = (ଅକ୍ଷର *)work_data + i11*nbw1 + i12*nbw2 + i13*nbw3;
                    from_float(src1_f32, src1_conv, ne10);
                }
            }
        }
    }

    ପାଇଁ (int64_t i13 = 0; i13 < ne13; i13++) {
        ପାଇଁ (int64_t i12 = 0; i12 < ne12; i12++) {
            ଅଚଳ ଶୂନ୍ୟ* wdata = (src1->type == vec_dot_type || src0->type == GGML_TYPE_Q8_0) ? src1->data : work_data;
            ଅଚଳ size_t row_size = ggml_row_size(vec_dot_type, ne10);
            ଯଦି (!ggml_zendnn_gemm(ctx,
                                  ne01,     // m
                                  ne11,     // n
                                  ne10,     // k
                                  static_cast<ଅଚଳ ଅକ୍ଷର *>(src0->data) + (i12/r2)*nb02 + (i13/r3)*nb03,
                                  ne00,     // lda
                                  static_cast<ଅଚଳ ଅକ୍ଷର *>(wdata) + (i12*ne11 + i13*ne12*ne11)*row_size,
                                  ne10,     // ldb
                                  static_cast<ଅକ୍ଷର *>(dst->data) + i12*nb2 + i13*nb3,
                                  ne01,     // ldc
                                  src0->type,
                                  src0->type == GGML_TYPE_Q8_0 ? GGML_TYPE_F32 : vec_dot_type,
                                  dst->type))
                GGML_ABORT("%s: ZenDNN gemm failed\n", __func__);
        }
    }
}

ସଂରଚନା mmid_row_mapping {
    int32_t i1;
    int32_t i2;
};

/**
 * ZenDNN batched matmul: computes C[i] = B[i] * A[i] for every active expert i via a single
 * group_matmul_direct() call. Batched analogue of ggml_zendnn_matmul() - see its docs for the
 * per-expert A/B/C shape convention. m and k are shared by every expert; n (batch size) varies
 * per expert, hence the vector.
 */
template <typename TA, typename TB, typename TC>
ସ୍ଥିର ତାର୍କିକ ggml_zendnn_group_matmul(ggml_backend_zendnn_context * ctx, int64_t m, int64_t k,
                                     ଅଚଳ std::vector<int64_t> & n,
                                     ଅଚଳ std::vector<ଅଚଳ ଶୂନ୍ୟ *> & A, int64_t lda,
                                     ଅଚଳ std::vector<ଅଚଳ ଶୂନ୍ୟ *> & B, int64_t ldb,
                                     ଅଚଳ std::vector<ଶୂନ୍ୟ *> & C, int64_t ldc) {

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_experts = n.size();

    zendnnl::lowoha::matmul::matmul_params base_params = ggml_zendnn_make_matmul_params<TA, TB, TC>(ctx);

    std::vector<ଅକ୍ଷର>         layout(n_experts, 'r');
    std::vector<ତାର୍କିକ>         trans_a(n_experts, ମିଥ୍ୟା);
    std::vector<ତାର୍କିକ>         trans_b(n_experts, ସତ୍ୟ);
    std::vector<ପୂର୍ଣ୍ଣସଂଖ୍ୟା>          batch_m(n_experts);
    std::vector<ପୂର୍ଣ୍ଣସଂଖ୍ୟା>          batch_n(n_experts, m);
    std::vector<ପୂର୍ଣ୍ଣସଂଖ୍ୟା>          batch_k(n_experts, k);
    std::vector<ଦଶମିକ>        alpha(n_experts, 1.0f);
    std::vector<ଦଶମିକ>        beta(n_experts, 0.0f);
    std::vector<ଅଚଳ ଶୂନ୍ୟ *> bias(n_experts, nullptr);
    std::vector<ପୂର୍ଣ୍ଣସଂଖ୍ୟା>          lda_v(n_experts, lda);
    std::vector<ପୂର୍ଣ୍ଣସଂଖ୍ୟା>          ldb_v(n_experts, ldb);
    std::vector<ପୂର୍ଣ୍ଣସଂଖ୍ୟା>          ldc_v(n_experts, ldc);
    std::vector<ତାର୍କିକ>         is_wei_const(n_experts, ସତ୍ୟ);
    std::vector<zendnnl::lowoha::matmul::matmul_params> params(n_experts, base_params);

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n_experts; i++) {
        batch_m[i] = n[i];

        // src_scale.dims depends on this expert's row count, unlike the rest of base_params
        ଯଦି constexpr (std::is_same_v<TA, block_q8_0>) {
            params[i].quant_params.src_scale.dims = {n[i], k / QK8_0};
        }
    }

    zendnnl::error_handling::status_t status = zendnnl::lowoha::matmul::group_matmul_direct(
        layout, trans_a, trans_b, batch_m, batch_n, batch_k, alpha,
        B, ldb_v, A, lda_v, bias, beta,
        C, ldc_v, is_wei_const, params);

    ଯଦି (status != zendnnl::error_handling::status_t::success) {
        GGML_LOG_ERROR("%s, ZenDNN group matmul failed: status=%d\n", __func__, static_cast<ପୂର୍ଣ୍ଣସଂଖ୍ୟା>(status));
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ସ୍ଥିର ତାର୍କିକ ggml_zendnn_group_gemm(ggml_backend_zendnn_context * ctx, int64_t m, int64_t k,
                                    ଅଚଳ std::vector<int64_t> & n,
                                    ଅଚଳ std::vector<ଅଚଳ ଶୂନ୍ୟ *> & A, int64_t lda,
                                    ଅଚଳ std::vector<ଅଚଳ ଶୂନ୍ୟ *> & B, int64_t ldb,
                                    ଅଚଳ std::vector<ଶୂନ୍ୟ *> & C, int64_t ldc,
                                    ପୂର୍ଣ୍ଣସଂଖ୍ୟା Atype, ପୂର୍ଣ୍ଣସଂଖ୍ୟା Btype, ପୂର୍ଣ୍ଣସଂଖ୍ୟା Ctype) {

    ପ୍ରତିଜ୍ଞା(m >= 0);
    ପାଇଁ (size_t i = 0; i < n.size(); i++) {
        ପ୍ରତିଜ୍ଞା(n[i] >= 0);
    }
    ପ୍ରତିଜ୍ଞା(k >= 0);
    ପ୍ରତିଜ୍ଞା(lda >= k);
    ପ୍ରତିଜ୍ଞା(ldb >= k);
    ପ୍ରତିଜ୍ଞା(ldc >= m);

    // categorize types
    ବାଛ (Atype) {
        ସ୍ଥିତି GGML_TYPE_F32:
            ଯଦି (Btype != GGML_TYPE_F32 || Ctype != GGML_TYPE_F32)
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            ଫେରନ୍ତୁ ggml_zendnn_group_matmul<ଦଶମିକ, ଦଶମିକ, ଦଶମିକ>(ctx, m, k, n, A, lda, B, ldb, C, ldc);
        ସ୍ଥିତି GGML_TYPE_BF16:
            ଯଦି (Btype != GGML_TYPE_BF16)
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            ଯଦି (Ctype == GGML_TYPE_BF16)
                ଫେରନ୍ତୁ ggml_zendnn_group_matmul<ggml_bf16_t, ggml_bf16_t, ggml_bf16_t>(
                    ctx, m, k, n, A, lda, B, ldb, C, ldc);
            ଯଦି (Ctype == GGML_TYPE_F32)
                ଫେରନ୍ତୁ ggml_zendnn_group_matmul<ggml_bf16_t, ggml_bf16_t, ଦଶମିକ>(ctx, m, k, n, A, lda, B, ldb, C, ldc);
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        ସ୍ଥିତି GGML_TYPE_Q8_0:
            ଯଦି (Btype != GGML_TYPE_F32 || Ctype != GGML_TYPE_F32)
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            ଫେରନ୍ତୁ ggml_zendnn_group_matmul<block_q8_0, ଦଶମିକ, ଦଶମିକ>(ctx, m, k, n, A, lda, B, ldb, C, ldc);
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            ଫେରନ୍ତୁ ମିଥ୍ୟା; // unsupported type
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_zendnn_compute_forward_mul_mat_id(
    ggml_backend_zendnn_context * ctx,
    ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];  // expert weights
    ଅଚଳ ggml_tensor * src1 = dst->src[1];  // inputs
    ଅଚଳ ggml_tensor * ids  = dst->src[2];  // expert ids

    GGML_TENSOR_ଦ୍ୱିମିକ_OP_LOCALS

    // exit for no tokens to process
    ଯଦି (ne2 == 0 || ne11 == 0) {
        ଫେରନ୍ତୁ;
    }

    ggml_type         ଅଚଳ vec_dot_type = src0->type;
    ggml_from_float_t ଅଚଳ from_float = ggml_get_type_traits(vec_dot_type)->from_float_ref;

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == ଆକାର(ଦଶମିକ));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(ne03 == 1);
    GGML_ASSERT(ne13 == 1);
    GGML_ASSERT(ne3  == 1);

    // row groups
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_ids = ids->ne[0]; // n_expert_used
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_as  = ne02;       // n_experts

    std::vector<int64_t> matrix_row_counts(n_as, 0);
    std::vector<std::vector<mmid_row_mapping>> matrix_rows(n_as);

    int64_t total_rows = 0;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା     n_active_experts = 0;
    // group rows by expert (preprocessing step)
    ପାଇଁ (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା id = 0; id < n_ids; ++id) {
            ଅଚଳ int32_t i02 = *(ଅଚଳ int32_t *)((ଅଚଳ ଅକ୍ଷର *)ids->data + iid1*ids->nb[1] + id*ids->nb[0]);

            GGML_ASSERT(i02 >= 0 && i02 < n_as);

            ଯଦି (matrix_row_counts[i02] == 0) {
                n_active_experts++;
            }
            matrix_rows[i02].push_back({id, iid1});
            matrix_row_counts[i02]++;
            total_rows++;
        }
    }

    ଯଦି (total_rows == 0) {
        ଫେରନ୍ତୁ; // no rows to process
    }

    ଅଚଳ size_t row_size = ggml_row_size(vec_dot_type, ne10);

    // For Q8_0, src1 is always F32; the gather buffer must hold F32 rows (ne10*4 bytes),
    // not Q8_0-encoded rows (row_size ≈ ne10/32*34 bytes) — they differ by ~4x.
    ଅଚଳ size_t f32_row_size = (size_t)ne10 * ଆକାର(ଦଶମିକ);
    ଅଚଳ size_t gather_row_size = (src0->type == GGML_TYPE_Q8_0) ? f32_row_size : row_size;

    ଯଦି (src1->type != vec_dot_type && src0->type != GGML_TYPE_Q8_0) {
        GGML_ASSERT(src1->type == GGML_TYPE_F32);
    }

    // size for MoE gather/scatter buffers
    ଅଚଳ size_t wdata_cur_size = total_rows * gather_row_size;
    ଅଚଳ size_t dst_cur_size = total_rows * ggml_row_size(dst->type, ne01);

    // allocate single buffer for all needs
    ଅଚଳ size_t total_size = wdata_cur_size + dst_cur_size;
    ଯଦି (ctx->work_size < total_size) {
        ctx->work_data.reset(ନୂତନ ଅକ୍ଷର[total_size]);
        ctx->work_size = total_size;
    }

    // partition the buffer
    ଅକ୍ଷର * wdata_cur = ctx->work_data.get();
    ଅକ୍ଷର * dst_cur = wdata_cur + wdata_cur_size;

    // per-expert data collected during gather, handed to ggml_zendnn_group_gemm() as one batch
    std::vector<int64_t>      expert_row_ଗଣନା(n_active_experts);
    std::vector<ଅଚଳ ଶୂନ୍ୟ *> batch_src(n_active_experts);
    std::vector<ଅଚଳ ଶୂନ୍ୟ *> batch_wei(n_active_experts);
    std::vector<ଶୂନ୍ୟ *>       batch_dst(n_active_experts);

    // precompute per-expert buffer offsets and batch indices for the parallel loop below
    std::vector<int64_t> expert_wdata_off(n_as, 0);
    std::vector<int64_t> expert_dst_off(n_as, 0);
    std::vector<ପୂର୍ଣ୍ଣସଂଖ୍ୟା>     expert_batch_idx(n_as, -1);
    {
        int64_t w_off = 0;
        int64_t d_off = 0;
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା     batch_idx = 0;
        ପାଇଁ (int64_t cur_a = 0; cur_a < n_as; ++cur_a) {
            ଯଦି (matrix_row_counts[cur_a] == 0) {
                ଜାରି;
            }
            expert_wdata_off[cur_a] = w_off;
            expert_dst_off[cur_a]   = d_off;
            expert_batch_idx[cur_a] = batch_idx;
            w_off += matrix_row_counts[cur_a] * gather_row_size;
            d_off += matrix_row_counts[cur_a] * ggml_row_size(dst->type, ne01);
            batch_idx++;
        }
    }

    // gather + inline-convert input rows into each expert's batch slot
    #pragma omp parallel ପାଇଁ ଅଙ୍କ_threads(ctx->n_threads) schedule(ସ୍ଥିର)
    ପାଇଁ (int64_t cur_a = 0; cur_a < n_as; ++cur_a) {
        ଅଚଳ int64_t cne1 = matrix_row_counts[cur_a];

        ଯଦି (cne1 == 0) {
            ଜାରି;
        }

        ଅଚଳ int64_t w_off     = expert_wdata_off[cur_a];
        ଅଚଳ int64_t d_off     = expert_dst_off[cur_a];
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା     batch_idx = expert_batch_idx[cur_a];

        ପାଇଁ (int64_t ir1 = 0; ir1 < cne1; ++ir1) {
            ଅଚଳ mmid_row_mapping & row_mapping = matrix_rows[cur_a][ir1];
            ଅଚଳ int64_t id  = row_mapping.i1;
            ଅଚଳ int64_t i11 = id % ne11;
            ଅଚଳ int64_t i12 = row_mapping.i2;

            ଅଚଳ ଅକ୍ଷର * src_row = (ଅଚଳ ଅକ୍ଷର *) src1->data + i11*nb11 + i12*nb12;
            ଶୂନ୍ୟ *       dst_row = wdata_cur + w_off + ir1 * gather_row_size;

            ଯଦି (src1->type != vec_dot_type && src0->type != GGML_TYPE_Q8_0) {
                from_float((ଅଚଳ ଦଶମିକ *) src_row, dst_row, ne10);
            } ଅନ୍ୟଥା {
                // no conversion: src1 already matches vec_dot_type, or src0 is Q8_0, whose
                // ZenDNN dynamic quantization requires the row to stay in F32
                std::memcpy(dst_row, src_row, gather_row_size);
            }
        }

        expert_row_ଗଣନା[batch_idx] = cne1;
        batch_src[batch_idx]        = wdata_cur + w_off;
        batch_wei[batch_idx]        = (ଅଚଳ ଅକ୍ଷର *) src0->data + cur_a * nb02;
        batch_dst[batch_idx]        = dst_cur + d_off;
    }

    ଯଦି (!ggml_zendnn_group_gemm(ctx,
                                ne01,             // m
                                ne10,             // k
                                expert_row_ଗଣନା, // n (per expert)
                                batch_wei, ne00,  // A: weights, lda
                                batch_src, ne10,  // B: input, ldb
                                batch_dst, ne01,  // C: output, ldc
                                src0->type,
                                src0->type == GGML_TYPE_Q8_0 ? GGML_TYPE_F32 : vec_dot_type,
                                dst->type))
        GGML_ABORT("%s: ZenDNN group gemm failed\n", __func__);

    // scatter output rows to destination
    #pragma omp parallel ପାଇଁ ଅଙ୍କ_threads(ctx->n_threads) schedule(ସ୍ଥିର)
    ପାଇଁ (int64_t cur_a = 0; cur_a < n_as; ++cur_a) {
        ଅଚଳ int64_t cne1 = matrix_row_counts[cur_a];

        ଯଦି (cne1 == 0) {
            ଜାରି;
        }

        ଅଚଳ int64_t d_off = expert_dst_off[cur_a];

        ପାଇଁ (int64_t ir1 = 0; ir1 < cne1; ++ir1) {
            ଅଚଳ mmid_row_mapping & row_mapping = matrix_rows[cur_a][ir1];
            ଅଚଳ int64_t id = row_mapping.i1;
            ଅଚଳ int64_t i1 = id;
            ଅଚଳ int64_t i2 = row_mapping.i2;

            std::memcpy(
                (ଅକ୍ଷର *) dst->data + i1*nb1 + i2*nb2,
                dst_cur + d_off + ir1 * ggml_row_size(dst->type, ne01),
                ggml_row_size(dst->type, ne01)
            );
        }
    }
}

// backend interface

ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * ggml_backend_zendnn_get_name(ggml_backend_t backend) {
    ଫେରନ୍ତୁ "ZenDNN";

    GGML_UNUSED(backend);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_zendnn_free(ggml_backend_t backend) {
    ggml_backend_zendnn_context * ctx = (ggml_backend_zendnn_context *)backend->context;
    delete ctx;
    delete backend;
}

ସ୍ଥିର ggml_status ggml_backend_zendnn_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_zendnn_context * ctx = (ggml_backend_zendnn_context *)backend->context;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < cgraph->n_nodes; i++) {
        ସଂରଚନା ggml_tensor * ଗଣ୍ଠି = cgraph->nodes[i];

        ଯଦି ((ଗଣ୍ଠି->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            ଜାରି;
        }

        ବାଛ (ଗଣ୍ଠି->op) {
            ସ୍ଥିତି GGML_OP_MUL_MAT:
                ggml_zendnn_compute_forward_mul_mat(ctx, ଗଣ୍ଠି);
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_OP_MUL_MAT_ID:
                ggml_zendnn_compute_forward_mul_mat_id(ctx, ଗଣ୍ଠି);
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_OP_NONE:
            ସ୍ଥିତି GGML_OP_RESHAPE:
            ସ୍ଥିତି GGML_OP_VIEW:
            ସ୍ଥିତି GGML_OP_PERMUTE:
            ସ୍ଥିତି GGML_OP_TRANSPOSE:
                ଭାଙ୍ଗ;

            ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(ଗଣ୍ଠି));
        }
    }

    ଫେରନ୍ତୁ GGML_STATUS_SUCCESS;

    GGML_UNUSED(backend);
}

ସ୍ଥିର ସଂରଚନା ggml_backend_i ggml_backend_zendnn_i = {
    /* .get_name                = */ ggml_backend_zendnn_get_name,
    /* .free                    = */ ggml_backend_zendnn_free,
    /* .set_tensor_async        = */ ଖାଲି,
    /* .get_tensor_async        = */ ଖାଲି,
    /* .set_tensor_2d_async     = */ ଖାଲି,
    /* .get_tensor_2d_async     = */ ଖାଲି,
    /* .cpy_tensor_async        = */ ଖାଲି,
    /* .synchronize             = */ ଖାଲି,
    /* .graph_plan_create       = */ ଖାଲି,
    /* .graph_plan_free         = */ ଖାଲି,
    /* .graph_plan_update       = */ ଖାଲି,
    /* .graph_plan_compute      = */ ଖାଲି,
    /* .graph_compute           = */ ggml_backend_zendnn_graph_compute,
    /* .event_record            = */ ଖାଲି,
    /* .event_wait              = */ ଖାଲି,
    /* .graph_optimize          = */ ଖାଲି,
};

ସ୍ଥିର ggml_guid_t ggml_backend_zendnn_guid(ଶୂନ୍ୟ) {
    ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * guid_str = "AMD-ZENDNN-ACCEL";
    ଫେରନ୍ତୁ reinterpret_cast<ggml_guid_t>(const_cast<ଅକ୍ଷର*>(guid_str));
}

ggml_backend_t ggml_backend_zendnn_init(ଶୂନ୍ୟ) {
    ggml_backend_zendnn_context * ctx = ନୂତନ ggml_backend_zendnn_context;

    ggml_backend_t backend = ନୂତନ ggml_backend {
        /* .guid    = */ ggml_backend_zendnn_guid(),
        /* .iface   = */ ggml_backend_zendnn_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_zendnn_reg(), 0),
        /* .context = */ ctx,
    };

    ଫେରନ୍ତୁ backend;
}

ତାର୍କିକ ggml_backend_is_zendnn(ggml_backend_t backend) {
    ଫେରନ୍ତୁ backend != ଖାଲି && ggml_guid_matches(backend->guid, ggml_backend_zendnn_guid());
}

ଶୂନ୍ୟ ggml_backend_zendnn_set_n_threads(ggml_backend_t backend_zendnn, ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_threads) {
    GGML_ASSERT(ggml_backend_is_zendnn(backend_zendnn));

    ggml_backend_zendnn_context * ctx = (ggml_backend_zendnn_context *)backend_zendnn->context;
    ctx->n_threads = n_threads;
}

// device interface
ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * ggml_backend_zendnn_device_get_name(ggml_backend_dev_t dev) {
    ଫେରନ୍ତୁ "ZenDNN";

    GGML_UNUSED(dev);
}
/**
 * ZenDNN is AMD's performance library providing optimized primitives and implementations
 * for deep learning workloads on AMD CPUs. It targets improved performance for common
 * neural network operations on AMD architectures. For more information, see:
 * https://www.amd.com/en/developer/zendnn.html
 */
ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * ggml_backend_zendnn_device_get_description(ggml_backend_dev_t dev) {
    ଫେରନ୍ତୁ "ZenDNN: AMD optimized primitives backend for GGML (optimized for AMD CPUs)";

    GGML_UNUSED(dev);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_zendnn_device_get_memory(ggml_backend_dev_t dev, size_t * ମୁକ୍ତ, size_t * total) {
    *ମୁକ୍ତ  = 0;
    *total = 0;

    GGML_UNUSED(dev);
}

ସ୍ଥିର ଗଣନା ggml_backend_dev_type ggml_backend_zendnn_device_get_type(ggml_backend_dev_t dev) {
    ଫେରନ୍ତୁ GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_UNUSED(dev);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_zendnn_device_get_props(ggml_backend_dev_t dev, ସଂରଚନା ggml_backend_dev_props * props) {
    props->name        = ggml_backend_zendnn_device_get_name(dev);
    props->description = ggml_backend_zendnn_device_get_description(dev);
    props->type        = ggml_backend_zendnn_device_get_type(dev);
    ggml_backend_zendnn_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                = */ ମିଥ୍ୟା,
        /* .host_buffer          = */ ମିଥ୍ୟା,
        /* .buffer_from_host_ptr = */ ସତ୍ୟ,
        /* .events               = */ ମିଥ୍ୟା,
        /* .mmap_support         = */ ସତ୍ୟ,
    };
}

ସ୍ଥିର ggml_backend_t ggml_backend_zendnn_device_init_backend(ggml_backend_dev_t dev, ଅଚଳ ଅକ୍ଷର * params) {
    ggml_backend_t backend = ggml_backend_zendnn_init();
    ଯଦି (backend == ଖାଲି) {
        GGML_LOG_ERROR("%s: error: failed to initialize ZenDNN backend\n", __func__);
        ଫେରନ୍ତୁ ଖାଲି;
    }

    ଫେରନ୍ତୁ backend;

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

ସ୍ଥିର ggml_backend_buffer_type_t ggml_backend_zendnn_device_get_buffer_type(ggml_backend_dev_t dev) {
    ଫେରନ୍ତୁ ggml_backend_cpu_buffer_type();

    GGML_UNUSED(dev);
}

ସ୍ଥିର ggml_backend_buffer_t ggml_backend_zendnn_device_buffer_from_host_ptr(ggml_backend_dev_t dev, ଶୂନ୍ୟ * ptr, size_t size, size_t max_tensor_size) {
    ଫେରନ୍ତୁ ggml_backend_cpu_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

ସ୍ଥିର ତାର୍କିକ ggml_zendnn_adaptive_fallback_enabled() {
    ସ୍ଥିର ଅଚଳ ତାର୍କିକ enabled = std::getenv("GGML_ZENDNN_ADAPTIVE_FALLBACK") == nullptr ||
                                std::atoi(std::getenv("GGML_ZENDNN_ADAPTIVE_FALLBACK")) != 0;
    ଫେରନ୍ତୁ enabled;
}

ସ୍ଥିର ତାର୍କିକ ggml_backend_zendnn_device_supports_op(ggml_backend_dev_t dev, ଅଚଳ ସଂରଚନା ggml_tensor * op) {
    ବାଛ (op->op) {
        ସ୍ଥିତି GGML_OP_NONE:
        ସ୍ଥିତି GGML_OP_RESHAPE:
        ସ୍ଥିତି GGML_OP_VIEW:
        ସ୍ଥିତି GGML_OP_PERMUTE:
        ସ୍ଥିତି GGML_OP_TRANSPOSE:
            ଫେରନ୍ତୁ ସତ୍ୟ;

        ସ୍ଥିତି GGML_OP_MUL_MAT:
        ସ୍ଥିତି GGML_OP_MUL_MAT_ID:
        {
            ଅଚଳ ggml_tensor * weights = op->src[0];
            ଅଚଳ ggml_tensor * inputs = op->src[1];

            ଅଚଳ int64_t ne10 = inputs->ne[0];
            ଅଚଳ int64_t ne0 = op->ne[0];
            ଅଚଳ int64_t ne1 = op->ne[1];
            ଅଚଳ int64_t min_batch = 1;

            ଯଦି(!ggml_is_contiguous(weights) || !ggml_is_contiguous(inputs)) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }

            ଯଦି (ggml_zendnn_adaptive_fallback_enabled()) {
                ଅଚଳ int64_t K = inputs->ne[0];
                ଅଚଳ int64_t N = (inputs->ne[1]*inputs->ne[2]*inputs->ne[3]);
                ଅଚଳ int64_t M = weights->ne[1];
                ଯଦି(K <= 256 || N <= 128 || M <= 96) {
                    ଫେରନ୍ତୁ ମିଥ୍ୟା;
                }

                // MUL_MAT_ID's gather+matmul+scatter approach favors a moderate expert count
                ଯଦି (op->op == GGML_OP_MUL_MAT_ID) {
                    ଅଚଳ int64_t n_experts = weights->ne[2];
                    ଅଚଳ int64_t max_experts = 32;
                    ଯଦି (n_experts > max_experts) {
                        ଫେରନ୍ତୁ ମିଥ୍ୟା;
                    }

                    // fall back once the average rows per expert (N / n_experts) is too thin
                    // to amortize each per-expert GEMM's overhead
                    ଯଦି (N / n_experts <= 32) {
                        ଫେରନ୍ତୁ ମିଥ୍ୟା;
                    }
                }
            }
            ଅନ୍ୟଥା ଯଦି (ne0 < min_batch || ne1 < min_batch || ne10 < min_batch) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }

            ବାଛ (weights->type) {
                ସ୍ଥିତି GGML_TYPE_F32:
                ସ୍ଥିତି GGML_TYPE_BF16:
                ସ୍ଥିତି GGML_TYPE_Q8_0:
                    ଫେରନ୍ତୁ ସତ୍ୟ;
                ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                    ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }
        } ଭାଙ୍ଗ;

        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    GGML_UNUSED(dev);
}

ସ୍ଥିର ତାର୍କିକ ggml_backend_zendnn_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    ଫେରନ୍ତୁ ggml_backend_buft_is_host(buft);

    GGML_UNUSED(dev);
}

ସ୍ଥିର ଅଚଳ ସଂରଚନା ggml_backend_device_i ggml_backend_zendnn_device_i = {
    /* .get_name               = */ ggml_backend_zendnn_device_get_name,
    /* .get_description        = */ ggml_backend_zendnn_device_get_description,
    /* .get_memory             = */ ggml_backend_zendnn_device_get_memory,
    /* .get_type               = */ ggml_backend_zendnn_device_get_type,
    /* .get_props              = */ ggml_backend_zendnn_device_get_props,
    /* .init_backend           = */ ggml_backend_zendnn_device_init_backend,
    /* .get_buffer_type        = */ ggml_backend_zendnn_device_get_buffer_type,
    /* .get_host_buffer_type   = */ ଖାଲି,
    /* .buffer_from_host_ptr   = */ ggml_backend_zendnn_device_buffer_from_host_ptr,
    /* .supports_op            = */ ggml_backend_zendnn_device_supports_op,
    /* .supports_buft          = */ ggml_backend_zendnn_device_supports_buft,
    /* .offload_op             = */ ଖାଲି,
    /* .event_new              = */ ଖାଲି,
    /* .event_free             = */ ଖାଲି,
    /* .event_synchronize      = */ ଖାଲି,
};

// backend reg interface
ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * ggml_backend_zendnn_reg_get_name(ggml_backend_reg_t reg) {
    ଫେରନ୍ତୁ "ZenDNN";

    GGML_UNUSED(reg);
}

ସ୍ଥିର size_t ggml_backend_zendnn_reg_get_device_ଗଣନା(ggml_backend_reg_t reg) {
    ଫେରନ୍ତୁ 1;

    GGML_UNUSED(reg);
}

ସ୍ଥିର ggml_backend_dev_t ggml_backend_zendnn_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    ସ୍ଥିର ggml_backend_device ggml_backend_zendnn_device = {
        /* .iface   = */ ggml_backend_zendnn_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };

    ଫେରନ୍ତୁ &ggml_backend_zendnn_device;
}

ସ୍ଥିର ଶୂନ୍ୟ * ggml_backend_zendnn_get_proc_address(ggml_backend_reg_t reg, ଅଚଳ ଅକ୍ଷର * name) {
    ଯଦି (std::strcmp(name, "ggml_backend_set_n_threads") == 0) {
        ଫେରନ୍ତୁ (ଶୂନ୍ୟ *) ggml_backend_zendnn_set_n_threads;
    }
    ଫେରନ୍ତୁ ଖାଲି;

    GGML_UNUSED(reg);
    GGML_UNUSED(name);
}

ସ୍ଥିର ଅଚଳ ସଂରଚନା ggml_backend_reg_i ggml_backend_zendnn_reg_i = {
    /* .get_name         = */ ggml_backend_zendnn_reg_get_name,
    /* .get_device_count = */ ggml_backend_zendnn_reg_get_device_ଗଣନା,
    /* .get_device       = */ ggml_backend_zendnn_reg_get_device,
    /* .get_proc_address = */ ggml_backend_zendnn_get_proc_address,
};

ggml_backend_reg_t ggml_backend_zendnn_reg(ଶୂନ୍ୟ) {
    ସ୍ଥିର ସଂରଚନା ggml_backend_reg ggml_backend_zendnn_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_zendnn_reg_i,
        /* .context     = */ ଖାଲି,
    };

    ଫେରନ୍ତୁ &ggml_backend_zendnn_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_zendnn_reg)
