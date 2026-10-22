#ಅನ್ತರ್ಭಾವ "ggml-zendnn.h"

#ಅನ್ತರ್ಭಾವ "ggml-backend-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_COMMON_DECL_CPP
#ಅನ್ತರ್ಭಾವ "ggml-common.h"

#ಅನ್ತರ್ಭಾವ "zendnnl.hpp"

#ಅನ್ತರ್ಭಾವ <cstring>


ರಚನೆ ggml_backend_zendnn_context {
    ಪೂರ್ಣಾಂಕ n_ಎಳೆಗಳು = GGML_DEFAULT_N_ಎಳೆಗಳು;
    std::unique_ptr<ಅಕ್ಷರ[]> work_ದತ್ತಾಂಶ;
    size_t work_ಗಾತ್ರ = 0;
};

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು T>
zendnnl::common::ದತ್ತಾಂಶ_type_t ggml_ಗಮ್ಯ_zendnn_type() {
    ಆದರೆ constexpr (std::is_same_v<T, ದಶಮಾಂಶ>) {
        ಹಿಂತಿರುಗಿ zendnnl::common::ದತ್ತಾಂಶ_type_t::f32;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (std::is_same_v<T, ggml_bf16_t>) {
        ಹಿಂತಿರುಗಿ zendnnl::common::ದತ್ತಾಂಶ_type_t::bf16;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (std::is_same_v<T, ಖಂಡ_q8_0>) {
        ಹಿಂತಿರುಗಿ zendnnl::common::ದತ್ತಾಂಶ_type_t::s8;
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಹಿಂತಿರುಗಿ zendnnl::common::ದತ್ತಾಂಶ_type_t::none;
    }
}

/**
 * Builds the matmul_params shared by ggml_zendnn_matmul() and ggml_zendnn_group_matmul():
 * dtype selection plus, for Q8_0 weights, dynamic-quant setup. Callers still need to set
 * quant_params.src_scale.dims themselves, since that depends on the batch size(s) in use.
 */
ಮಾದರಿ <ಬಗೆ_ಹೆಸರು TA, ಬಗೆ_ಹೆಸರು TB, ಬಗೆ_ಹೆಸರು TC>
ಸ್ಥಿರ zendnnl::lowoha::matmul::matmul_params ggml_zendnn_make_matmul_params(ggml_backend_zendnn_context * ctx) {
    zendnnl::lowoha::matmul::matmul_params params;
    params.dtypes.src = ggml_ಗಮ್ಯ_zendnn_type<TB>();
    params.dtypes.wei = ggml_ಗಮ್ಯ_zendnn_type<TA>();
    params.dtypes.dst = ggml_ಗಮ್ಯ_zendnn_type<TC>();
    params.ಅಂಕೆ_ಎಳೆಗಳು = ctx->n_ಎಳೆಗಳು;

    ಆದರೆ constexpr (std::is_same_v<TA, ಖಂಡ_q8_0>) {
        params.dtypes.compute = zendnnl::common::ದತ್ತಾಂಶ_type_t::s8;
        params.dynamic_quant = ನಿಜ;
        params.quant_params.src_scale.buff = ಶೂನ್ಯಸೂಚಕ;
        params.quant_params.src_scale.dt   = zendnnl::common::ದತ್ತಾಂಶ_type_t::bf16;
        params.packing.pack_format_b = 1;
    }
    ಹಿಂತಿರುಗಿ params;
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
ಮಾದರಿ <ಬಗೆ_ಹೆಸರು TA, ಬಗೆ_ಹೆಸರು TB, ಬಗೆ_ಹೆಸರು TC>
ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_zendnn_matmul(ggml_backend_zendnn_context * ctx, int64_t m, int64_t n, int64_t k,
                               ಸ್ಥಿರಾಂಕ TA * A, int64_t lda, ಸ್ಥಿರಾಂಕ TB * B, int64_t ldb, TC * C,
                               int64_t ldc) {

    zendnnl::lowoha::matmul::matmul_params params = ggml_zendnn_make_matmul_params<TA, TB, TC>(ctx);

    zendnnl::lowoha::matmul::matmul_batch_params_t batch_params;

    ಆದರೆ constexpr (std::is_same_v<TA, ಖಂಡ_q8_0>) {
        params.quant_params.src_scale.dims = {n, k / QK8_0};
    }

    zendnnl::error_handling::ಸ್ಥಿತಿ_t ಸ್ಥಿತಿ_2 = zendnnl::lowoha::matmul::matmul_ನೇರ(
        'r', ಸುಳ್ಳು, ನಿಜ,   // row-major, don't transpose B, transpose A (because it's column-major)
        n,                  // M: rows of B and C
        m,                  // N: cols of A^T and C
        k,                  // K: cols of B, rows of A
        1.0f,               // alpha
        B, ldb,             // src: B[n,k]
        A, lda,             // weight: A[k,m] column-major (transposed)
        ಶೂನ್ಯಸೂಚಕ,            // bias
        0.0f,               // beta
        C, ldc,             // output C[n,m]
        ನಿಜ,               // is_weights_const
        batch_params,       // batch_params
        params              // params
    );

    ಆದರೆ (ಸ್ಥಿತಿ_2 != zendnnl::error_handling::ಸ್ಥಿತಿ_t::success) {
        GGML_LOG_ERROR("%s, ZenDNN matmul failed: status=%d\n", __func__, static_cast<ಪೂರ್ಣಾಂಕ>(ಸ್ಥಿತಿ_2));
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_zendnn_gemm(ggml_backend_zendnn_context * ctx, int64_t m, int64_t n, int64_t k,
                              ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * A, int64_t lda, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * B, int64_t ldb, ಶೂನ್ಯ * C,
                              int64_t ldc, ಪೂರ್ಣಾಂಕ Atype, ಪೂರ್ಣಾಂಕ Btype, ಪೂರ್ಣಾಂಕ Ctype) {

    ಪ್ರತಿಜ್ಞಾ(m >= 0);
    ಪ್ರತಿಜ್ಞಾ(n >= 0);
    ಪ್ರತಿಜ್ಞಾ(k >= 0);
    ಪ್ರತಿಜ್ಞಾ(lda >= k);
    ಪ್ರತಿಜ್ಞಾ(ldb >= k);
    ಪ್ರತಿಜ್ಞಾ(ldc >= m);

    // categorize types
    ಆಯ್ಕೆ (Atype) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            ಆದರೆ (Btype != GGML_TYPE_F32 || Ctype != GGML_TYPE_F32)
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            ಹಿಂತಿರುಗಿ ggml_zendnn_matmul<ದಶಮಾಂಶ, ದಶಮಾಂಶ, ದಶಮಾಂಶ>(
                ctx, m, n, k,
                (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)A, lda,
                (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)B, ldb,
                (ದಶಮಾಂಶ *)C, ldc);
        ಸ್ಥಿತಿ GGML_TYPE_BF16:
            ಆದರೆ (Btype != GGML_TYPE_BF16)
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            ಆದರೆ (Ctype == GGML_TYPE_BF16)
                ಹಿಂತಿರುಗಿ ggml_zendnn_matmul<ggml_bf16_t, ggml_bf16_t, ggml_bf16_t>(
                    ctx, m, n, k,
                    (ಸ್ಥಿರಾಂಕ ggml_bf16_t *)A, lda,
                    (ಸ್ಥಿರಾಂಕ ggml_bf16_t *)B, ldb,
                    (ggml_bf16_t *)C, ldc);
            ಆದರೆ (Ctype == GGML_TYPE_F32)
                ಹಿಂತಿರುಗಿ ggml_zendnn_matmul<ggml_bf16_t, ggml_bf16_t, ದಶಮಾಂಶ>(
                    ctx, m, n, k,
                    (ಸ್ಥಿರಾಂಕ ggml_bf16_t *)A, lda,
                    (ಸ್ಥಿರಾಂಕ ggml_bf16_t *)B, ldb,
                    (ದಶಮಾಂಶ *)C, ldc);
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
            ಆದರೆ (Btype != GGML_TYPE_F32 || Ctype != GGML_TYPE_F32)
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            ಹಿಂತಿರುಗಿ ggml_zendnn_matmul<ಖಂಡ_q8_0, ದಶಮಾಂಶ, ದಶಮಾಂಶ>(
                ctx, m, n, k,
                (ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 *)A, lda,
                (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)B, ldb,
                (ದಶಮಾಂಶ *)C, ldc);
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು; // unsupported type
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_zendnn_compute_forward_mul_mat(
    ggml_backend_zendnn_context * ctx,
    ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];  // weights
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];  // inputs

    GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS

    ggml_type         ಸ್ಥಿರಾಂಕ vec_dot_type = src0->type;
    ggml_from_float_t ಸ್ಥಿರಾಂಕ from_float = ggml_get_type_traits(vec_dot_type)->from_float_ref;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_ಗಾತ್ರ(src0->type));
    GGML_ASSERT(nb10 == ggml_type_ಗಾತ್ರ(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == ಗಾತ್ರ(ದಶಮಾಂಶ));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // broadcast factors
    ಸ್ಥಿರಾಂಕ int64_t r2 = ne12/ne02;
    ಸ್ಥಿರಾಂಕ int64_t r3 = ne13/ne03;

    ಶೂನ್ಯ * work_ದತ್ತಾಂಶ = ctx->work_ದತ್ತಾಂಶ.get();

    // ZenDNN requires FP32 for dynamic quantization, so conversion is skipped
    ಆದರೆ (src1->type != vec_dot_type && src0->type != GGML_TYPE_Q8_0) {
        ಸ್ಥಿರಾಂಕ size_t nbw1 = ggml_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(vec_dot_type, ne10);
        ಸ್ಥಿರಾಂಕ size_t nbw2 = nbw1 * ne11;
        ಸ್ಥಿರಾಂಕ size_t nbw3 = nbw2 * ne12;
        ಸ್ಥಿರಾಂಕ size_t desired_wsize = ne13 * nbw3;
        ಆದರೆ (ctx->work_ಗಾತ್ರ < desired_wsize) {
            ctx->work_ದತ್ತಾಂಶ.reset(ಹೊಸ ಅಕ್ಷರ[desired_wsize]);
            ctx->work_ಗಾತ್ರ = desired_wsize;
        }
        work_ದತ್ತಾಂಶ = ctx->work_ದತ್ತಾಂಶ.get();

        // #pragma omp parallel for num_threads(ctx->n_threads)
        #ಸೂಚನೆ omp parallel ಗಾಗಿ collapse(3) ಅಂಕೆ_ಎಳೆಗಳು(ctx->n_ಎಳೆಗಳು) ವೇಳಾಪಟ್ಟಿ(ಸ್ಥಿರ)
        ಗಾಗಿ (int64_t i13 = 0; i13 < ne13; ++i13) {
            ಗಾಗಿ (int64_t i12 = 0; i12 < ne12; ++i12) {
                ಗಾಗಿ (int64_t i11 = 0; i11 < ne11; ++i11) {
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src1_f32 = (ದಶಮಾಂಶ *)((ಅಕ್ಷರ *)src1->ದತ್ತಾಂಶ + i11*nb11 + i12*nb12 + i13*nb13);
                    ಶೂನ್ಯ * src1_conv = (ಅಕ್ಷರ *)work_ದತ್ತಾಂಶ + i11*nbw1 + i12*nbw2 + i13*nbw3;
                    from_float(src1_f32, src1_conv, ne10);
                }
            }
        }
    }

    ಗಾಗಿ (int64_t i13 = 0; i13 < ne13; i13++) {
        ಗಾಗಿ (int64_t i12 = 0; i12 < ne12; i12++) {
            ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ* wdata = (src1->type == vec_dot_type || src0->type == GGML_TYPE_Q8_0) ? src1->ದತ್ತಾಂಶ : work_ದತ್ತಾಂಶ;
            ಸ್ಥಿರಾಂಕ size_t ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = ggml_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(vec_dot_type, ne10);
            ಆದರೆ (!ggml_zendnn_gemm(ctx,
                                  ne01,     // m
                                  ne11,     // n
                                  ne10,     // k
                                  static_cast<ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *>(src0->ದತ್ತಾಂಶ) + (i12/r2)*nb02 + (i13/r3)*nb03,
                                  ne00,     // lda
                                  static_cast<ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *>(wdata) + (i12*ne11 + i13*ne12*ne11)*ಅಡ್ಡಸಾಲು_ಗಾತ್ರ,
                                  ne10,     // ldb
                                  static_cast<ಅಕ್ಷರ *>(dst->ದತ್ತಾಂಶ) + i12*nb2 + i13*nb3,
                                  ne01,     // ldc
                                  src0->type,
                                  src0->type == GGML_TYPE_Q8_0 ? GGML_TYPE_F32 : vec_dot_type,
                                  dst->type))
                GGML_ABORT("%s: ZenDNN gemm failed\n", __func__);
        }
    }
}

ರಚನೆ mmid_ಅಡ್ಡಸಾಲು_mapping {
    int32_t i1;
    int32_t i2;
};

/**
 * ZenDNN batched matmul: computes C[i] = B[i] * A[i] for every active expert i via a single
 * group_matmul_direct() call. Batched analogue of ggml_zendnn_matmul() - see its docs for the
 * per-expert A/B/C shape convention. m and k are shared by every expert; n (batch size) varies
 * per expert, hence the vector.
 */
ಮಾದರಿ <ಬಗೆ_ಹೆಸರು TA, ಬಗೆ_ಹೆಸರು TB, ಬಗೆ_ಹೆಸರು TC>
ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_zendnn_ಗುಂಪು_matmul(ggml_backend_zendnn_context * ctx, int64_t m, int64_t k,
                                     ಸ್ಥಿರಾಂಕ std::vector<int64_t> & n,
                                     ಸ್ಥಿರಾಂಕ std::vector<ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *> & A, int64_t lda,
                                     ಸ್ಥಿರಾಂಕ std::vector<ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *> & B, int64_t ldb,
                                     ಸ್ಥಿರಾಂಕ std::vector<ಶೂನ್ಯ *> & C, int64_t ldc) {

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_experts = n.size();

    zendnnl::lowoha::matmul::matmul_params ತಳ_params = ggml_zendnn_make_matmul_params<TA, TB, TC>(ctx);

    std::vector<ಅಕ್ಷರ>         layout(n_experts, 'r');
    std::vector<ತಾರ್ಕಿಕ>         trans_a(n_experts, ಸುಳ್ಳು);
    std::vector<ತಾರ್ಕಿಕ>         trans_b(n_experts, ನಿಜ);
    std::vector<ಪೂರ್ಣಾಂಕ>          batch_m(n_experts);
    std::vector<ಪೂರ್ಣಾಂಕ>          batch_n(n_experts, m);
    std::vector<ಪೂರ್ಣಾಂಕ>          batch_k(n_experts, k);
    std::vector<ದಶಮಾಂಶ>        alpha(n_experts, 1.0f);
    std::vector<ದಶಮಾಂಶ>        beta(n_experts, 0.0f);
    std::vector<ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *> bias(n_experts, ಶೂನ್ಯಸೂಚಕ);
    std::vector<ಪೂರ್ಣಾಂಕ>          lda_v(n_experts, lda);
    std::vector<ಪೂರ್ಣಾಂಕ>          ldb_v(n_experts, ldb);
    std::vector<ಪೂರ್ಣಾಂಕ>          ldc_v(n_experts, ldc);
    std::vector<ತಾರ್ಕಿಕ>         is_wei_const(n_experts, ನಿಜ);
    std::vector<zendnnl::lowoha::matmul::matmul_params> params(n_experts, ತಳ_params);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n_experts; i++) {
        batch_m[i] = n[i];

        // src_scale.dims depends on this expert's row count, unlike the rest of base_params
        ಆದರೆ constexpr (std::is_same_v<TA, ಖಂಡ_q8_0>) {
            params[i].quant_params.src_scale.dims = {n[i], k / QK8_0};
        }
    }

    zendnnl::error_handling::ಸ್ಥಿತಿ_t ಸ್ಥಿತಿ_2 = zendnnl::lowoha::matmul::ಗುಂಪು_matmul_ನೇರ(
        layout, trans_a, trans_b, batch_m, batch_n, batch_k, alpha,
        B, ldb_v, A, lda_v, bias, beta,
        C, ldc_v, is_wei_const, params);

    ಆದರೆ (ಸ್ಥಿತಿ_2 != zendnnl::error_handling::ಸ್ಥಿತಿ_t::success) {
        GGML_LOG_ERROR("%s, ZenDNN group matmul failed: status=%d\n", __func__, static_cast<ಪೂರ್ಣಾಂಕ>(ಸ್ಥಿತಿ_2));
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_zendnn_ಗುಂಪು_gemm(ggml_backend_zendnn_context * ctx, int64_t m, int64_t k,
                                    ಸ್ಥಿರಾಂಕ std::vector<int64_t> & n,
                                    ಸ್ಥಿರಾಂಕ std::vector<ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *> & A, int64_t lda,
                                    ಸ್ಥಿರಾಂಕ std::vector<ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *> & B, int64_t ldb,
                                    ಸ್ಥಿರಾಂಕ std::vector<ಶೂನ್ಯ *> & C, int64_t ldc,
                                    ಪೂರ್ಣಾಂಕ Atype, ಪೂರ್ಣಾಂಕ Btype, ಪೂರ್ಣಾಂಕ Ctype) {

    ಪ್ರತಿಜ್ಞಾ(m >= 0);
    ಗಾಗಿ (size_t i = 0; i < n.size(); i++) {
        ಪ್ರತಿಜ್ಞಾ(n[i] >= 0);
    }
    ಪ್ರತಿಜ್ಞಾ(k >= 0);
    ಪ್ರತಿಜ್ಞಾ(lda >= k);
    ಪ್ರತಿಜ್ಞಾ(ldb >= k);
    ಪ್ರತಿಜ್ಞಾ(ldc >= m);

    // categorize types
    ಆಯ್ಕೆ (Atype) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            ಆದರೆ (Btype != GGML_TYPE_F32 || Ctype != GGML_TYPE_F32)
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            ಹಿಂತಿರುಗಿ ggml_zendnn_ಗುಂಪು_matmul<ದಶಮಾಂಶ, ದಶಮಾಂಶ, ದಶಮಾಂಶ>(ctx, m, k, n, A, lda, B, ldb, C, ldc);
        ಸ್ಥಿತಿ GGML_TYPE_BF16:
            ಆದರೆ (Btype != GGML_TYPE_BF16)
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            ಆದರೆ (Ctype == GGML_TYPE_BF16)
                ಹಿಂತಿರುಗಿ ggml_zendnn_ಗುಂಪು_matmul<ggml_bf16_t, ggml_bf16_t, ggml_bf16_t>(
                    ctx, m, k, n, A, lda, B, ldb, C, ldc);
            ಆದರೆ (Ctype == GGML_TYPE_F32)
                ಹಿಂತಿರುಗಿ ggml_zendnn_ಗುಂಪು_matmul<ggml_bf16_t, ggml_bf16_t, ದಶಮಾಂಶ>(ctx, m, k, n, A, lda, B, ldb, C, ldc);
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
            ಆದರೆ (Btype != GGML_TYPE_F32 || Ctype != GGML_TYPE_F32)
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            ಹಿಂತಿರುಗಿ ggml_zendnn_ಗುಂಪು_matmul<ಖಂಡ_q8_0, ದಶಮಾಂಶ, ದಶಮಾಂಶ>(ctx, m, k, n, A, lda, B, ldb, C, ldc);
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು; // unsupported type
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_zendnn_compute_forward_mul_mat_id(
    ggml_backend_zendnn_context * ctx,
    ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];  // expert weights
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];  // inputs
    ಸ್ಥಿರಾಂಕ ggml_tensor * ids  = dst->src[2];  // expert ids

    GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS

    // exit for no tokens to process
    ಆದರೆ (ne2 == 0 || ne11 == 0) {
        ಹಿಂತಿರುಗಿ;
    }

    ggml_type         ಸ್ಥಿರಾಂಕ vec_dot_type = src0->type;
    ggml_from_float_t ಸ್ಥಿರಾಂಕ from_float = ggml_get_type_traits(vec_dot_type)->from_float_ref;

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_ಗಾತ್ರ(src0->type));
    GGML_ASSERT(nb10 == ggml_type_ಗಾತ್ರ(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == ಗಾತ್ರ(ದಶಮಾಂಶ));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(ne03 == 1);
    GGML_ASSERT(ne13 == 1);
    GGML_ASSERT(ne3  == 1);

    // row groups
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_ids = ids->ne[0]; // n_expert_used
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_as  = ne02;       // n_experts

    std::vector<int64_t> ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_counts(n_as, 0);
    std::vector<std::vector<mmid_ಅಡ್ಡಸಾಲು_mapping>> ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು(n_as);

    int64_t ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು = 0;
    ಪೂರ್ಣಾಂಕ     n_ಸಕ್ರಿಯ_experts = 0;
    // group rows by expert (preprocessing step)
    ಗಾಗಿ (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ id = 0; id < n_ids; ++id) {
            ಸ್ಥಿರಾಂಕ int32_t i02 = *(ಸ್ಥಿರಾಂಕ int32_t *)((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)ids->ದತ್ತಾಂಶ + iid1*ids->nb[1] + id*ids->nb[0]);

            GGML_ASSERT(i02 >= 0 && i02 < n_as);

            ಆದರೆ (ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_counts[i02] == 0) {
                n_ಸಕ್ರಿಯ_experts++;
            }
            ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು[i02].push_back({id, iid1});
            ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_counts[i02]++;
            ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು++;
        }
    }

    ಆದರೆ (ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು == 0) {
        ಹಿಂತಿರುಗಿ; // no rows to process
    }

    ಸ್ಥಿರಾಂಕ size_t ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = ggml_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(vec_dot_type, ne10);

    // For Q8_0, src1 is always F32; the gather buffer must hold F32 rows (ne10*4 bytes),
    // not Q8_0-encoded rows (row_size ≈ ne10/32*34 bytes) — they differ by ~4x.
    ಸ್ಥಿರಾಂಕ size_t f32_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = (size_t)ne10 * ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ size_t gather_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = (src0->type == GGML_TYPE_Q8_0) ? f32_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ : ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;

    ಆದರೆ (src1->type != vec_dot_type && src0->type != GGML_TYPE_Q8_0) {
        GGML_ASSERT(src1->type == GGML_TYPE_F32);
    }

    // size for MoE gather/scatter buffers
    ಸ್ಥಿರಾಂಕ size_t wdata_cur_ಗಾತ್ರ = ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು * gather_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ size_t dst_cur_ಗಾತ್ರ = ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು * ggml_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(dst->type, ne01);

    // allocate single buffer for all needs
    ಸ್ಥಿರಾಂಕ size_t ಒಟ್ಟು_ಗಾತ್ರ = wdata_cur_ಗಾತ್ರ + dst_cur_ಗಾತ್ರ;
    ಆದರೆ (ctx->work_ಗಾತ್ರ < ಒಟ್ಟು_ಗಾತ್ರ) {
        ctx->work_ದತ್ತಾಂಶ.reset(ಹೊಸ ಅಕ್ಷರ[ಒಟ್ಟು_ಗಾತ್ರ]);
        ctx->work_ಗಾತ್ರ = ಒಟ್ಟು_ಗಾತ್ರ;
    }

    // partition the buffer
    ಅಕ್ಷರ * wdata_cur = ctx->work_ದತ್ತಾಂಶ.get();
    ಅಕ್ಷರ * dst_cur = wdata_cur + wdata_cur_ಗಾತ್ರ;

    // per-expert data collected during gather, handed to ggml_zendnn_group_gemm() as one batch
    std::vector<int64_t>      expert_ಅಡ್ಡಸಾಲು_ಎಣಿಕೆ(n_ಸಕ್ರಿಯ_experts);
    std::vector<ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *> batch_src(n_ಸಕ್ರಿಯ_experts);
    std::vector<ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *> batch_wei(n_ಸಕ್ರಿಯ_experts);
    std::vector<ಶೂನ್ಯ *>       batch_dst(n_ಸಕ್ರಿಯ_experts);

    // precompute per-expert buffer offsets and batch indices for the parallel loop below
    std::vector<int64_t> expert_wdata_off(n_as, 0);
    std::vector<int64_t> expert_dst_off(n_as, 0);
    std::vector<ಪೂರ್ಣಾಂಕ>     expert_batch_idx(n_as, -1);
    {
        int64_t w_off = 0;
        int64_t d_off = 0;
        ಪೂರ್ಣಾಂಕ     batch_idx = 0;
        ಗಾಗಿ (int64_t cur_a = 0; cur_a < n_as; ++cur_a) {
            ಆದರೆ (ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_counts[cur_a] == 0) {
                ಮುಂದುವರಿ;
            }
            expert_wdata_off[cur_a] = w_off;
            expert_dst_off[cur_a]   = d_off;
            expert_batch_idx[cur_a] = batch_idx;
            w_off += ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_counts[cur_a] * gather_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;
            d_off += ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_counts[cur_a] * ggml_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(dst->type, ne01);
            batch_idx++;
        }
    }

    // gather + inline-convert input rows into each expert's batch slot
    #ಸೂಚನೆ omp parallel ಗಾಗಿ ಅಂಕೆ_ಎಳೆಗಳು(ctx->n_ಎಳೆಗಳು) ವೇಳಾಪಟ್ಟಿ(ಸ್ಥಿರ)
    ಗಾಗಿ (int64_t cur_a = 0; cur_a < n_as; ++cur_a) {
        ಸ್ಥಿರಾಂಕ int64_t cne1 = ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_counts[cur_a];

        ಆದರೆ (cne1 == 0) {
            ಮುಂದುವರಿ;
        }

        ಸ್ಥಿರಾಂಕ int64_t w_off     = expert_wdata_off[cur_a];
        ಸ್ಥಿರಾಂಕ int64_t d_off     = expert_dst_off[cur_a];
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ     batch_idx = expert_batch_idx[cur_a];

        ಗಾಗಿ (int64_t ir1 = 0; ir1 < cne1; ++ir1) {
            ಸ್ಥಿರಾಂಕ mmid_ಅಡ್ಡಸಾಲು_mapping & ಅಡ್ಡಸಾಲು_mapping = ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು[cur_a][ir1];
            ಸ್ಥಿರಾಂಕ int64_t id  = ಅಡ್ಡಸಾಲು_mapping.i1;
            ಸ್ಥಿರಾಂಕ int64_t i11 = id % ne11;
            ಸ್ಥಿರಾಂಕ int64_t i12 = ಅಡ್ಡಸಾಲು_mapping.i2;

            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * src_ಅಡ್ಡಸಾಲು = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src1->ದತ್ತಾಂಶ + i11*nb11 + i12*nb12;
            ಶೂನ್ಯ *       dst_ಅಡ್ಡಸಾಲು = wdata_cur + w_off + ir1 * gather_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;

            ಆದರೆ (src1->type != vec_dot_type && src0->type != GGML_TYPE_Q8_0) {
                from_float((ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src_ಅಡ್ಡಸಾಲು, dst_ಅಡ್ಡಸಾಲು, ne10);
            } ಇಲ್ಲದಿದ್ದರೆ {
                // no conversion: src1 already matches vec_dot_type, or src0 is Q8_0, whose
                // ZenDNN dynamic quantization requires the row to stay in F32
                std::ಸ್ಮರಣೆ_ನಕಲು(dst_ಅಡ್ಡಸಾಲು, src_ಅಡ್ಡಸಾಲು, gather_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ);
            }
        }

        expert_ಅಡ್ಡಸಾಲು_ಎಣಿಕೆ[batch_idx] = cne1;
        batch_src[batch_idx]        = wdata_cur + w_off;
        batch_wei[batch_idx]        = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + cur_a * nb02;
        batch_dst[batch_idx]        = dst_cur + d_off;
    }

    ಆದರೆ (!ggml_zendnn_ಗುಂಪು_gemm(ctx,
                                ne01,             // m
                                ne10,             // k
                                expert_ಅಡ್ಡಸಾಲು_ಎಣಿಕೆ, // n (per expert)
                                batch_wei, ne00,  // A: weights, lda
                                batch_src, ne10,  // B: input, ldb
                                batch_dst, ne01,  // C: output, ldc
                                src0->type,
                                src0->type == GGML_TYPE_Q8_0 ? GGML_TYPE_F32 : vec_dot_type,
                                dst->type))
        GGML_ABORT("%s: ZenDNN group gemm failed\n", __func__);

    // scatter output rows to destination
    #ಸೂಚನೆ omp parallel ಗಾಗಿ ಅಂಕೆ_ಎಳೆಗಳು(ctx->n_ಎಳೆಗಳು) ವೇಳಾಪಟ್ಟಿ(ಸ್ಥಿರ)
    ಗಾಗಿ (int64_t cur_a = 0; cur_a < n_as; ++cur_a) {
        ಸ್ಥಿರಾಂಕ int64_t cne1 = ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_counts[cur_a];

        ಆದರೆ (cne1 == 0) {
            ಮುಂದುವರಿ;
        }

        ಸ್ಥಿರಾಂಕ int64_t d_off = expert_dst_off[cur_a];

        ಗಾಗಿ (int64_t ir1 = 0; ir1 < cne1; ++ir1) {
            ಸ್ಥಿರಾಂಕ mmid_ಅಡ್ಡಸಾಲು_mapping & ಅಡ್ಡಸಾಲು_mapping = ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲುಗಳು[cur_a][ir1];
            ಸ್ಥಿರಾಂಕ int64_t id = ಅಡ್ಡಸಾಲು_mapping.i1;
            ಸ್ಥಿರಾಂಕ int64_t i1 = id;
            ಸ್ಥಿರಾಂಕ int64_t i2 = ಅಡ್ಡಸಾಲು_mapping.i2;

            std::ಸ್ಮರಣೆ_ನಕಲು(
                (ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + i1*nb1 + i2*nb2,
                dst_cur + d_off + ir1 * ggml_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(dst->type, ne01),
                ggml_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(dst->type, ne01)
            );
        }
    }
}

// backend interface

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_zendnn_get_name(ggml_backend_t backend) {
    ಹಿಂತಿರುಗಿ "ZenDNN";

    GGML_ಬಳಸದ(backend);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_zendnn_free(ggml_backend_t backend) {
    ggml_backend_zendnn_context * ctx = (ggml_backend_zendnn_context *)backend->context;
    ಅಳಿಸು ctx;
    ಅಳಿಸು backend;
}

ಸ್ಥಿರ ggml_ಸ್ಥಿತಿ ggml_backend_zendnn_ನಕ್ಷೆ_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_zendnn_context * ctx = (ggml_backend_zendnn_context *)backend->context;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < cgraph->n_nodes; i++) {
        ರಚನೆ ggml_tensor * ಗಂಟು = cgraph->nodes[i];

        ಆದರೆ ((ಗಂಟು->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            ಮುಂದುವರಿ;
        }

        ಆಯ್ಕೆ (ಗಂಟು->op) {
            ಸ್ಥಿತಿ GGML_OP_MUL_MAT:
                ggml_zendnn_compute_forward_mul_mat(ctx, ಗಂಟು);
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_OP_MUL_MAT_ID:
                ggml_zendnn_compute_forward_mul_mat_id(ctx, ಗಂಟು);
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_OP_NONE:
            ಸ್ಥಿತಿ GGML_OP_RESHAPE:
            ಸ್ಥಿತಿ GGML_OP_VIEW:
            ಸ್ಥಿತಿ GGML_OP_ಕ್ರಮಬದಲಾವಣೆ:
            ಸ್ಥಿತಿ GGML_OP_TRANSPOSE:
                ನಿಲ್ಲಿಸು;

            ಪೂರ್ವನಿಯೋಜಿತ:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(ಗಂಟು));
        }
    }

    ಹಿಂತಿರುಗಿ GGML_ಸ್ಥಿತಿ_SUCCESS;

    GGML_ಬಳಸದ(backend);
}

ಸ್ಥಿರ ರಚನೆ ggml_backend_i ggml_backend_zendnn_i = {
    /* .get_name                = */ ggml_backend_zendnn_get_name,
    /* .free                    = */ ggml_backend_zendnn_free,
    /* .set_tensor_async        = */ ಖಾಲಿ,
    /* .get_tensor_async        = */ ಖಾಲಿ,
    /* .set_tensor_2d_async     = */ ಖಾಲಿ,
    /* .get_tensor_2d_async     = */ ಖಾಲಿ,
    /* .cpy_tensor_async        = */ ಖಾಲಿ,
    /* .synchronize             = */ ಖಾಲಿ,
    /* .graph_plan_create       = */ ಖಾಲಿ,
    /* .graph_plan_free         = */ ಖಾಲಿ,
    /* .graph_plan_update       = */ ಖಾಲಿ,
    /* .graph_plan_compute      = */ ಖಾಲಿ,
    /* .graph_compute           = */ ggml_backend_zendnn_ನಕ್ಷೆ_compute,
    /* .event_record            = */ ಖಾಲಿ,
    /* .event_wait              = */ ಖಾಲಿ,
    /* .graph_optimize          = */ ಖಾಲಿ,
};

ಸ್ಥಿರ ggml_guid_t ggml_backend_zendnn_guid(ಶೂನ್ಯ) {
    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * guid_str = "AMD-ZENDNN-ACCEL";
    ಹಿಂತಿರುಗಿ reinterpret_cast<ggml_guid_t>(const_cast<ಅಕ್ಷರ*>(guid_str));
}

ggml_backend_t ggml_backend_zendnn_init(ಶೂನ್ಯ) {
    ggml_backend_zendnn_context * ctx = ಹೊಸ ggml_backend_zendnn_context;

    ggml_backend_t backend = ಹೊಸ ggml_backend {
        /* .guid    = */ ggml_backend_zendnn_guid(),
        /* .iface   = */ ggml_backend_zendnn_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_zendnn_reg(), 0),
        /* .context = */ ctx,
    };

    ಹಿಂತಿರುಗಿ backend;
}

ತಾರ್ಕಿಕ ggml_backend_is_zendnn(ggml_backend_t backend) {
    ಹಿಂತಿರುಗಿ backend != ಖಾಲಿ && ggml_guid_matches(backend->guid, ggml_backend_zendnn_guid());
}

ಶೂನ್ಯ ggml_backend_zendnn_ಗಣ_n_ಎಳೆಗಳು(ggml_backend_t backend_zendnn, ಪೂರ್ಣಾಂಕ n_ಎಳೆಗಳು) {
    GGML_ASSERT(ggml_backend_is_zendnn(backend_zendnn));

    ggml_backend_zendnn_context * ctx = (ggml_backend_zendnn_context *)backend_zendnn->context;
    ctx->n_ಎಳೆಗಳು = n_ಎಳೆಗಳು;
}

// device interface
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_zendnn_device_get_name(ggml_backend_dev_t dev) {
    ಹಿಂತಿರುಗಿ "ZenDNN";

    GGML_ಬಳಸದ(dev);
}
/**
 * ZenDNN is AMD's performance library providing optimized primitives and implementations
 * for deep learning workloads on AMD CPUs. It targets improved performance for common
 * neural network operations on AMD architectures. For more information, see:
 * https://www.amd.com/en/developer/zendnn.html
 */
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_zendnn_device_get_description(ggml_backend_dev_t dev) {
    ಹಿಂತಿರುಗಿ "ZenDNN: AMD optimized primitives backend for GGML (optimized for AMD CPUs)";

    GGML_ಬಳಸದ(dev);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_zendnn_device_get_ಸ್ಮರಣೆ(ggml_backend_dev_t dev, size_t * ಬಿಡುಗಡೆ, size_t * ಒಟ್ಟು) {
    *ಬಿಡುಗಡೆ  = 0;
    *ಒಟ್ಟು = 0;

    GGML_ಬಳಸದ(dev);
}

ಸ್ಥಿರ ಎಣಿಕೆ ggml_backend_dev_type ggml_backend_zendnn_device_get_type(ggml_backend_dev_t dev) {
    ಹಿಂತಿರುಗಿ GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_ಬಳಸದ(dev);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_zendnn_device_get_props(ggml_backend_dev_t dev, ರಚನೆ ggml_backend_dev_props * props) {
    props->name        = ggml_backend_zendnn_device_get_name(dev);
    props->description = ggml_backend_zendnn_device_get_description(dev);
    props->type        = ggml_backend_zendnn_device_get_type(dev);
    ggml_backend_zendnn_device_get_ಸ್ಮರಣೆ(dev, &props->ಸ್ಮರಣೆ_free, &props->ಸ್ಮರಣೆ_ಒಟ್ಟು);
    props->caps = {
        /* .async                = */ ಸುಳ್ಳು,
        /* .host_buffer          = */ ಸುಳ್ಳು,
        /* .buffer_from_host_ptr = */ ನಿಜ,
        /* .events               = */ ಸುಳ್ಳು,
        /* .mmap_support         = */ ನಿಜ,
    };
}

ಸ್ಥಿರ ggml_backend_t ggml_backend_zendnn_device_init_backend(ggml_backend_dev_t dev, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * params) {
    ggml_backend_t backend = ggml_backend_zendnn_init();
    ಆದರೆ (backend == ಖಾಲಿ) {
        GGML_LOG_ERROR("%s: error: failed to initialize ZenDNN backend\n", __func__);
        ಹಿಂತಿರುಗಿ ಖಾಲಿ;
    }

    ಹಿಂತಿರುಗಿ backend;

    GGML_ಬಳಸದ(dev);
    GGML_ಬಳಸದ(params);
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_type_t ggml_backend_zendnn_device_get_ಬಫರ್_type(ggml_backend_dev_t dev) {
    ಹಿಂತಿರುಗಿ ggml_backend_cpu_ಬಫರ್_type();

    GGML_ಬಳಸದ(dev);
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_t ggml_backend_zendnn_device_ಬಫರ್_from_host_ptr(ggml_backend_dev_t dev, ಶೂನ್ಯ * ptr, size_t size, size_t max_tensor_ಗಾತ್ರ) {
    ಹಿಂತಿರುಗಿ ggml_backend_cpu_ಬಫರ್_from_ptr(ptr, size);

    GGML_ಬಳಸದ(dev);
    GGML_ಬಳಸದ(max_tensor_ಗಾತ್ರ);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_zendnn_adaptive_fallback_enabled() {
    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ enabled = std::getenv("GGML_ZENDNN_ADAPTIVE_FALLBACK") == ಶೂನ್ಯಸೂಚಕ ||
                                std::atoi(std::getenv("GGML_ZENDNN_ADAPTIVE_FALLBACK")) != 0;
    ಹಿಂತಿರುಗಿ enabled;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_zendnn_device_supports_op(ggml_backend_dev_t dev, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op) {
    ಆಯ್ಕೆ (op->op) {
        ಸ್ಥಿತಿ GGML_OP_NONE:
        ಸ್ಥಿತಿ GGML_OP_RESHAPE:
        ಸ್ಥಿತಿ GGML_OP_VIEW:
        ಸ್ಥಿತಿ GGML_OP_ಕ್ರಮಬದಲಾವಣೆ:
        ಸ್ಥಿತಿ GGML_OP_TRANSPOSE:
            ಹಿಂತಿರುಗಿ ನಿಜ;

        ಸ್ಥಿತಿ GGML_OP_MUL_MAT:
        ಸ್ಥಿತಿ GGML_OP_MUL_MAT_ID:
        {
            ಸ್ಥಿರಾಂಕ ggml_tensor * weights = op->src[0];
            ಸ್ಥಿರಾಂಕ ggml_tensor * inputs = op->src[1];

            ಸ್ಥಿರಾಂಕ int64_t ne10 = inputs->ne[0];
            ಸ್ಥಿರಾಂಕ int64_t ne0 = op->ne[0];
            ಸ್ಥಿರಾಂಕ int64_t ne1 = op->ne[1];
            ಸ್ಥಿರಾಂಕ int64_t min_batch = 1;

            ಆದರೆ(!ggml_is_contiguous(weights) || !ggml_is_contiguous(inputs)) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }

            ಆದರೆ (ggml_zendnn_adaptive_fallback_enabled()) {
                ಸ್ಥಿರಾಂಕ int64_t K = inputs->ne[0];
                ಸ್ಥಿರಾಂಕ int64_t N = (inputs->ne[1]*inputs->ne[2]*inputs->ne[3]);
                ಸ್ಥಿರಾಂಕ int64_t M = weights->ne[1];
                ಆದರೆ(K <= 256 || N <= 128 || M <= 96) {
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }

                // MUL_MAT_ID's gather+matmul+scatter approach favors a moderate expert count
                ಆದರೆ (op->op == GGML_OP_MUL_MAT_ID) {
                    ಸ್ಥಿರಾಂಕ int64_t n_experts = weights->ne[2];
                    ಸ್ಥಿರಾಂಕ int64_t max_experts = 32;
                    ಆದರೆ (n_experts > max_experts) {
                        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                    }

                    // fall back once the average rows per expert (N / n_experts) is too thin
                    // to amortize each per-expert GEMM's overhead
                    ಆದರೆ (N / n_experts <= 32) {
                        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                    }
                }
            }
            ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ne0 < min_batch || ne1 < min_batch || ne10 < min_batch) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }

            ಆಯ್ಕೆ (weights->type) {
                ಸ್ಥಿತಿ GGML_TYPE_F32:
                ಸ್ಥಿತಿ GGML_TYPE_BF16:
                ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
                    ಹಿಂತಿರುಗಿ ನಿಜ;
                ಪೂರ್ವನಿಯೋಜಿತ:
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
        } ನಿಲ್ಲಿಸು;

        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    GGML_ಬಳಸದ(dev);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_zendnn_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_ಬಫರ್_type_t buft) {
    ಹಿಂತಿರುಗಿ ggml_backend_buft_is_host(buft);

    GGML_ಬಳಸದ(dev);
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_backend_device_i ggml_backend_zendnn_device_i = {
    /* .get_name               = */ ggml_backend_zendnn_device_get_name,
    /* .get_description        = */ ggml_backend_zendnn_device_get_description,
    /* .get_memory             = */ ggml_backend_zendnn_device_get_ಸ್ಮರಣೆ,
    /* .get_type               = */ ggml_backend_zendnn_device_get_type,
    /* .get_props              = */ ggml_backend_zendnn_device_get_props,
    /* .init_backend           = */ ggml_backend_zendnn_device_init_backend,
    /* .get_buffer_type        = */ ggml_backend_zendnn_device_get_ಬಫರ್_type,
    /* .get_host_buffer_type   = */ ಖಾಲಿ,
    /* .buffer_from_host_ptr   = */ ggml_backend_zendnn_device_ಬಫರ್_from_host_ptr,
    /* .supports_op            = */ ggml_backend_zendnn_device_supports_op,
    /* .supports_buft          = */ ggml_backend_zendnn_device_supports_buft,
    /* .offload_op             = */ ಖಾಲಿ,
    /* .event_new              = */ ಖಾಲಿ,
    /* .event_free             = */ ಖಾಲಿ,
    /* .event_synchronize      = */ ಖಾಲಿ,
};

// backend reg interface
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_zendnn_reg_get_name(ggml_backend_reg_t reg) {
    ಹಿಂತಿರುಗಿ "ZenDNN";

    GGML_ಬಳಸದ(reg);
}

ಸ್ಥಿರ size_t ggml_backend_zendnn_reg_get_device_ಎಣಿಕೆ(ggml_backend_reg_t reg) {
    ಹಿಂತಿರುಗಿ 1;

    GGML_ಬಳಸದ(reg);
}

ಸ್ಥಿರ ggml_backend_dev_t ggml_backend_zendnn_reg_get_device(ggml_backend_reg_t reg, size_t ಸೂಚ್ಯಂಕ) {
    GGML_ASSERT(ಸೂಚ್ಯಂಕ == 0);

    ಸ್ಥಿರ ggml_backend_device ggml_backend_zendnn_device = {
        /* .iface   = */ ggml_backend_zendnn_device_i,
        /* .reg     = */ reg,
        /* .context = */ ಶೂನ್ಯಸೂಚಕ,
    };

    ಹಿಂತಿರುಗಿ &ggml_backend_zendnn_device;
}

ಸ್ಥಿರ ಶೂನ್ಯ * ggml_backend_zendnn_get_proc_ವಿಳಾಸ(ggml_backend_reg_t reg, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * name) {
    ಆದರೆ (std::strcmp(name, "ggml_backend_set_n_threads") == 0) {
        ಹಿಂತಿರುಗಿ (ಶೂನ್ಯ *) ggml_backend_zendnn_ಗಣ_n_ಎಳೆಗಳು;
    }
    ಹಿಂತಿರುಗಿ ಖಾಲಿ;

    GGML_ಬಳಸದ(reg);
    GGML_ಬಳಸದ(name);
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_backend_reg_i ggml_backend_zendnn_reg_i = {
    /* .get_name         = */ ggml_backend_zendnn_reg_get_name,
    /* .get_device_count = */ ggml_backend_zendnn_reg_get_device_ಎಣಿಕೆ,
    /* .get_device       = */ ggml_backend_zendnn_reg_get_device,
    /* .get_proc_address = */ ggml_backend_zendnn_get_proc_ವಿಳಾಸ,
};

ggml_backend_reg_t ggml_backend_zendnn_reg(ಶೂನ್ಯ) {
    ಸ್ಥಿರ ರಚನೆ ggml_backend_reg ggml_backend_zendnn_reg = {
        /* .api_version = */ GGML_BACKEND_API_ಆವೃತ್ತಿ,
        /* .iface       = */ ggml_backend_zendnn_reg_i,
        /* .context     = */ ಖಾಲಿ,
    };

    ಹಿಂತಿರುಗಿ &ggml_backend_zendnn_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_zendnn_reg)
