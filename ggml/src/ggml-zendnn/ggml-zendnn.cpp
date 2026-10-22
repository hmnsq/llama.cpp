#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-zendnn.h"

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_COMMON_DECL_CPP
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-common.h"

#ਅਨ੍ਤਰ੍ਭਾਵ "zendnnl.hpp"

#ਅਨ੍ਤਰ੍ਭਾਵ <cstring>


ਬਣਤਰ ggml_backend_zendnn_context {
    ਪੂਰਨ n_ਧਾਗੇ = GGML_DEFAULT_N_ਧਾਗੇ;
    std::unique_ptr<ਅੱਖਰ[]> work_ਅੰਕੜੇ;
    size_t work_ਮਾਪ = 0;
};

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ T>
zendnnl::common::ਅੰਕੜੇ_type_t ggml_ਮੰਜ਼ਿਲ_zendnn_type() {
    ਜੇ constexpr (std::is_same_v<T, ਦਸ਼ਮਲਵ>) {
        ਵਾਪਸ zendnnl::common::ਅੰਕੜੇ_type_t::f32;
    } ਨਹੀਂਤਾਂ ਜੇ constexpr (std::is_same_v<T, ggml_bf16_t>) {
        ਵਾਪਸ zendnnl::common::ਅੰਕੜੇ_type_t::bf16;
    } ਨਹੀਂਤਾਂ ਜੇ constexpr (std::is_same_v<T, ਖੰਡ_q8_0>) {
        ਵਾਪਸ zendnnl::common::ਅੰਕੜੇ_type_t::s8;
    } ਨਹੀਂਤਾਂ {
        ਵਾਪਸ zendnnl::common::ਅੰਕੜੇ_type_t::none;
    }
}

/**
 * Builds the matmul_params shared by ggml_zendnn_matmul() and ggml_zendnn_group_matmul():
 * dtype selection plus, for Q8_0 weights, dynamic-quant setup. Callers still need to set
 * quant_params.src_scale.dims themselves, since that depends on the batch size(s) in use.
 */
ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ TA, ਕਿਸਮ_ਨਾਂ TB, ਕਿਸਮ_ਨਾਂ TC>
ਸਥਿਰ zendnnl::lowoha::matmul::matmul_params ggml_zendnn_make_matmul_params(ggml_backend_zendnn_context * ctx) {
    zendnnl::lowoha::matmul::matmul_params params;
    params.dtypes.src = ggml_ਮੰਜ਼ਿਲ_zendnn_type<TB>();
    params.dtypes.wei = ggml_ਮੰਜ਼ਿਲ_zendnn_type<TA>();
    params.dtypes.dst = ggml_ਮੰਜ਼ਿਲ_zendnn_type<TC>();
    params.ਅੰਕ_ਧਾਗੇ = ctx->n_ਧਾਗੇ;

    ਜੇ constexpr (std::is_same_v<TA, ਖੰਡ_q8_0>) {
        params.dtypes.compute = zendnnl::common::ਅੰਕੜੇ_type_t::s8;
        params.dynamic_quant = ਸੱਚ;
        params.quant_params.src_scale.buff = ਖਾਲੀ_ਸੂਚਕ;
        params.quant_params.src_scale.dt   = zendnnl::common::ਅੰਕੜੇ_type_t::bf16;
        params.packing.pack_format_b = 1;
    }
    ਵਾਪਸ params;
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
ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ TA, ਕਿਸਮ_ਨਾਂ TB, ਕਿਸਮ_ਨਾਂ TC>
ਸਥਿਰ ਤਾਰਕਿਕ ggml_zendnn_matmul(ggml_backend_zendnn_context * ctx, int64_t m, int64_t n, int64_t k,
                               ਅਚਲ TA * A, int64_t lda, ਅਚਲ TB * B, int64_t ldb, TC * C,
                               int64_t ldc) {

    zendnnl::lowoha::matmul::matmul_params params = ggml_zendnn_make_matmul_params<TA, TB, TC>(ctx);

    zendnnl::lowoha::matmul::matmul_batch_params_t batch_params;

    ਜੇ constexpr (std::is_same_v<TA, ਖੰਡ_q8_0>) {
        params.quant_params.src_scale.dims = {n, k / QK8_0};
    }

    zendnnl::error_handling::ਸਥਿਤੀ_t ਸਥਿਤੀ = zendnnl::lowoha::matmul::matmul_ਸਿੱਧਾ(
        'r', ਝੂਠ, ਸੱਚ,   // row-major, don't transpose B, transpose A (because it's column-major)
        n,                  // M: rows of B and C
        m,                  // N: cols of A^T and C
        k,                  // K: cols of B, rows of A
        1.0f,               // alpha
        B, ldb,             // src: B[n,k]
        A, lda,             // weight: A[k,m] column-major (transposed)
        ਖਾਲੀ_ਸੂਚਕ,            // bias
        0.0f,               // beta
        C, ldc,             // output C[n,m]
        ਸੱਚ,               // is_weights_const
        batch_params,       // batch_params
        params              // params
    );

    ਜੇ (ਸਥਿਤੀ != zendnnl::error_handling::ਸਥਿਤੀ_t::success) {
        GGML_LOG_ERROR("%s, ZenDNN matmul failed: status=%d\n", __func__, static_cast<ਪੂਰਨ>(ਸਥਿਤੀ));
        ਵਾਪਸ ਝੂਠ;
    }
    ਵਾਪਸ ਸੱਚ;
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_zendnn_gemm(ggml_backend_zendnn_context * ctx, int64_t m, int64_t n, int64_t k,
                              ਅਚਲ ਖਾਲੀ * A, int64_t lda, ਅਚਲ ਖਾਲੀ * B, int64_t ldb, ਖਾਲੀ * C,
                              int64_t ldc, ਪੂਰਨ Atype, ਪੂਰਨ Btype, ਪੂਰਨ Ctype) {

    ਪ੍ਰਤਿਜ੍ਞਾ(m >= 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(n >= 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(k >= 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(lda >= k);
    ਪ੍ਰਤਿਜ੍ਞਾ(ldb >= k);
    ਪ੍ਰਤਿਜ੍ਞਾ(ldc >= m);

    // categorize types
    ਚੋਣ (Atype) {
        ਹਾਲਤ GGML_TYPE_F32:
            ਜੇ (Btype != GGML_TYPE_F32 || Ctype != GGML_TYPE_F32)
                ਵਾਪਸ ਝੂਠ;
            ਵਾਪਸ ggml_zendnn_matmul<ਦਸ਼ਮਲਵ, ਦਸ਼ਮਲਵ, ਦਸ਼ਮਲਵ>(
                ctx, m, n, k,
                (ਅਚਲ ਦਸ਼ਮਲਵ *)A, lda,
                (ਅਚਲ ਦਸ਼ਮਲਵ *)B, ldb,
                (ਦਸ਼ਮਲਵ *)C, ldc);
        ਹਾਲਤ GGML_TYPE_BF16:
            ਜੇ (Btype != GGML_TYPE_BF16)
                ਵਾਪਸ ਝੂਠ;
            ਜੇ (Ctype == GGML_TYPE_BF16)
                ਵਾਪਸ ggml_zendnn_matmul<ggml_bf16_t, ggml_bf16_t, ggml_bf16_t>(
                    ctx, m, n, k,
                    (ਅਚਲ ggml_bf16_t *)A, lda,
                    (ਅਚਲ ggml_bf16_t *)B, ldb,
                    (ggml_bf16_t *)C, ldc);
            ਜੇ (Ctype == GGML_TYPE_F32)
                ਵਾਪਸ ggml_zendnn_matmul<ggml_bf16_t, ggml_bf16_t, ਦਸ਼ਮਲਵ>(
                    ctx, m, n, k,
                    (ਅਚਲ ggml_bf16_t *)A, lda,
                    (ਅਚਲ ggml_bf16_t *)B, ldb,
                    (ਦਸ਼ਮਲਵ *)C, ldc);
            ਵਾਪਸ ਝੂਠ;
        ਹਾਲਤ GGML_TYPE_Q8_0:
            ਜੇ (Btype != GGML_TYPE_F32 || Ctype != GGML_TYPE_F32)
                ਵਾਪਸ ਝੂਠ;
            ਵਾਪਸ ggml_zendnn_matmul<ਖੰਡ_q8_0, ਦਸ਼ਮਲਵ, ਦਸ਼ਮਲਵ>(
                ctx, m, n, k,
                (ਅਚਲ ਖੰਡ_q8_0 *)A, lda,
                (ਅਚਲ ਦਸ਼ਮਲਵ *)B, ldb,
                (ਦਸ਼ਮਲਵ *)C, ldc);
        ਮੂਲ:
            ਵਾਪਸ ਝੂਠ; // unsupported type
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_zendnn_compute_forward_mul_mat(
    ggml_backend_zendnn_context * ctx,
    ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];  // weights
    ਅਚਲ ggml_tensor * src1 = dst->src[1];  // inputs

    GGML_TENSOR_ਦੁਆਧਾਰੀ_OP_LOCALS

    ggml_type         ਅਚਲ vec_dot_type = src0->type;
    ggml_from_float_t ਅਚਲ from_float = ggml_get_type_traits(vec_dot_type)->from_float_ref;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_ਮਾਪ(src0->type));
    GGML_ASSERT(nb10 == ggml_type_ਮਾਪ(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // broadcast factors
    ਅਚਲ int64_t r2 = ne12/ne02;
    ਅਚਲ int64_t r3 = ne13/ne03;

    ਖਾਲੀ * work_ਅੰਕੜੇ = ctx->work_ਅੰਕੜੇ.get();

    // ZenDNN requires FP32 for dynamic quantization, so conversion is skipped
    ਜੇ (src1->type != vec_dot_type && src0->type != GGML_TYPE_Q8_0) {
        ਅਚਲ size_t nbw1 = ggml_ਪੰਕਤੀ_ਮਾਪ(vec_dot_type, ne10);
        ਅਚਲ size_t nbw2 = nbw1 * ne11;
        ਅਚਲ size_t nbw3 = nbw2 * ne12;
        ਅਚਲ size_t desired_wsize = ne13 * nbw3;
        ਜੇ (ctx->work_ਮਾਪ < desired_wsize) {
            ctx->work_ਅੰਕੜੇ.reset(ਨਵਾਂ ਅੱਖਰ[desired_wsize]);
            ctx->work_ਮਾਪ = desired_wsize;
        }
        work_ਅੰਕੜੇ = ctx->work_ਅੰਕੜੇ.get();

        // #pragma omp parallel for num_threads(ctx->n_threads)
        #ਹਦਾਇਤ omp parallel ਲਈ collapse(3) ਅੰਕ_ਧਾਗੇ(ctx->n_ਧਾਗੇ) ਸਮਾਂ_ਸਾਰਣੀ(ਸਥਿਰ)
        ਲਈ (int64_t i13 = 0; i13 < ne13; ++i13) {
            ਲਈ (int64_t i12 = 0; i12 < ne12; ++i12) {
                ਲਈ (int64_t i11 = 0; i11 < ne11; ++i11) {
                    ਅਚਲ ਦਸ਼ਮਲਵ * src1_f32 = (ਦਸ਼ਮਲਵ *)((ਅੱਖਰ *)src1->ਅੰਕੜੇ + i11*nb11 + i12*nb12 + i13*nb13);
                    ਖਾਲੀ * src1_conv = (ਅੱਖਰ *)work_ਅੰਕੜੇ + i11*nbw1 + i12*nbw2 + i13*nbw3;
                    from_float(src1_f32, src1_conv, ne10);
                }
            }
        }
    }

    ਲਈ (int64_t i13 = 0; i13 < ne13; i13++) {
        ਲਈ (int64_t i12 = 0; i12 < ne12; i12++) {
            ਅਚਲ ਖਾਲੀ* wdata = (src1->type == vec_dot_type || src0->type == GGML_TYPE_Q8_0) ? src1->ਅੰਕੜੇ : work_ਅੰਕੜੇ;
            ਅਚਲ size_t ਪੰਕਤੀ_ਮਾਪ = ggml_ਪੰਕਤੀ_ਮਾਪ(vec_dot_type, ne10);
            ਜੇ (!ggml_zendnn_gemm(ctx,
                                  ne01,     // m
                                  ne11,     // n
                                  ne10,     // k
                                  static_cast<ਅਚਲ ਅੱਖਰ *>(src0->ਅੰਕੜੇ) + (i12/r2)*nb02 + (i13/r3)*nb03,
                                  ne00,     // lda
                                  static_cast<ਅਚਲ ਅੱਖਰ *>(wdata) + (i12*ne11 + i13*ne12*ne11)*ਪੰਕਤੀ_ਮਾਪ,
                                  ne10,     // ldb
                                  static_cast<ਅੱਖਰ *>(dst->ਅੰਕੜੇ) + i12*nb2 + i13*nb3,
                                  ne01,     // ldc
                                  src0->type,
                                  src0->type == GGML_TYPE_Q8_0 ? GGML_TYPE_F32 : vec_dot_type,
                                  dst->type))
                GGML_ABORT("%s: ZenDNN gemm failed\n", __func__);
        }
    }
}

ਬਣਤਰ mmid_ਪੰਕਤੀ_mapping {
    int32_t i1;
    int32_t i2;
};

/**
 * ZenDNN batched matmul: computes C[i] = B[i] * A[i] for every active expert i via a single
 * group_matmul_direct() call. Batched analogue of ggml_zendnn_matmul() - see its docs for the
 * per-expert A/B/C shape convention. m and k are shared by every expert; n (batch size) varies
 * per expert, hence the vector.
 */
ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ TA, ਕਿਸਮ_ਨਾਂ TB, ਕਿਸਮ_ਨਾਂ TC>
ਸਥਿਰ ਤਾਰਕਿਕ ggml_zendnn_ਸਮੂਹ_matmul(ggml_backend_zendnn_context * ctx, int64_t m, int64_t k,
                                     ਅਚਲ std::vector<int64_t> & n,
                                     ਅਚਲ std::vector<ਅਚਲ ਖਾਲੀ *> & A, int64_t lda,
                                     ਅਚਲ std::vector<ਅਚਲ ਖਾਲੀ *> & B, int64_t ldb,
                                     ਅਚਲ std::vector<ਖਾਲੀ *> & C, int64_t ldc) {

    ਅਚਲ ਪੂਰਨ n_experts = n.size();

    zendnnl::lowoha::matmul::matmul_params ਅਧਾਰ_params = ggml_zendnn_make_matmul_params<TA, TB, TC>(ctx);

    std::vector<ਅੱਖਰ>         layout(n_experts, 'r');
    std::vector<ਤਾਰਕਿਕ>         trans_a(n_experts, ਝੂਠ);
    std::vector<ਤਾਰਕਿਕ>         trans_b(n_experts, ਸੱਚ);
    std::vector<ਪੂਰਨ>          batch_m(n_experts);
    std::vector<ਪੂਰਨ>          batch_n(n_experts, m);
    std::vector<ਪੂਰਨ>          batch_k(n_experts, k);
    std::vector<ਦਸ਼ਮਲਵ>        alpha(n_experts, 1.0f);
    std::vector<ਦਸ਼ਮਲਵ>        beta(n_experts, 0.0f);
    std::vector<ਅਚਲ ਖਾਲੀ *> bias(n_experts, ਖਾਲੀ_ਸੂਚਕ);
    std::vector<ਪੂਰਨ>          lda_v(n_experts, lda);
    std::vector<ਪੂਰਨ>          ldb_v(n_experts, ldb);
    std::vector<ਪੂਰਨ>          ldc_v(n_experts, ldc);
    std::vector<ਤਾਰਕਿਕ>         is_wei_const(n_experts, ਸੱਚ);
    std::vector<zendnnl::lowoha::matmul::matmul_params> params(n_experts, ਅਧਾਰ_params);

    ਲਈ (ਪੂਰਨ i = 0; i < n_experts; i++) {
        batch_m[i] = n[i];

        // src_scale.dims depends on this expert's row count, unlike the rest of base_params
        ਜੇ constexpr (std::is_same_v<TA, ਖੰਡ_q8_0>) {
            params[i].quant_params.src_scale.dims = {n[i], k / QK8_0};
        }
    }

    zendnnl::error_handling::ਸਥਿਤੀ_t ਸਥਿਤੀ = zendnnl::lowoha::matmul::ਸਮੂਹ_matmul_ਸਿੱਧਾ(
        layout, trans_a, trans_b, batch_m, batch_n, batch_k, alpha,
        B, ldb_v, A, lda_v, bias, beta,
        C, ldc_v, is_wei_const, params);

    ਜੇ (ਸਥਿਤੀ != zendnnl::error_handling::ਸਥਿਤੀ_t::success) {
        GGML_LOG_ERROR("%s, ZenDNN group matmul failed: status=%d\n", __func__, static_cast<ਪੂਰਨ>(ਸਥਿਤੀ));
        ਵਾਪਸ ਝੂਠ;
    }
    ਵਾਪਸ ਸੱਚ;
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_zendnn_ਸਮੂਹ_gemm(ggml_backend_zendnn_context * ctx, int64_t m, int64_t k,
                                    ਅਚਲ std::vector<int64_t> & n,
                                    ਅਚਲ std::vector<ਅਚਲ ਖਾਲੀ *> & A, int64_t lda,
                                    ਅਚਲ std::vector<ਅਚਲ ਖਾਲੀ *> & B, int64_t ldb,
                                    ਅਚਲ std::vector<ਖਾਲੀ *> & C, int64_t ldc,
                                    ਪੂਰਨ Atype, ਪੂਰਨ Btype, ਪੂਰਨ Ctype) {

    ਪ੍ਰਤਿਜ੍ਞਾ(m >= 0);
    ਲਈ (size_t i = 0; i < n.size(); i++) {
        ਪ੍ਰਤਿਜ੍ਞਾ(n[i] >= 0);
    }
    ਪ੍ਰਤਿਜ੍ਞਾ(k >= 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(lda >= k);
    ਪ੍ਰਤਿਜ੍ਞਾ(ldb >= k);
    ਪ੍ਰਤਿਜ੍ਞਾ(ldc >= m);

    // categorize types
    ਚੋਣ (Atype) {
        ਹਾਲਤ GGML_TYPE_F32:
            ਜੇ (Btype != GGML_TYPE_F32 || Ctype != GGML_TYPE_F32)
                ਵਾਪਸ ਝੂਠ;
            ਵਾਪਸ ggml_zendnn_ਸਮੂਹ_matmul<ਦਸ਼ਮਲਵ, ਦਸ਼ਮਲਵ, ਦਸ਼ਮਲਵ>(ctx, m, k, n, A, lda, B, ldb, C, ldc);
        ਹਾਲਤ GGML_TYPE_BF16:
            ਜੇ (Btype != GGML_TYPE_BF16)
                ਵਾਪਸ ਝੂਠ;
            ਜੇ (Ctype == GGML_TYPE_BF16)
                ਵਾਪਸ ggml_zendnn_ਸਮੂਹ_matmul<ggml_bf16_t, ggml_bf16_t, ggml_bf16_t>(
                    ctx, m, k, n, A, lda, B, ldb, C, ldc);
            ਜੇ (Ctype == GGML_TYPE_F32)
                ਵਾਪਸ ggml_zendnn_ਸਮੂਹ_matmul<ggml_bf16_t, ggml_bf16_t, ਦਸ਼ਮਲਵ>(ctx, m, k, n, A, lda, B, ldb, C, ldc);
            ਵਾਪਸ ਝੂਠ;
        ਹਾਲਤ GGML_TYPE_Q8_0:
            ਜੇ (Btype != GGML_TYPE_F32 || Ctype != GGML_TYPE_F32)
                ਵਾਪਸ ਝੂਠ;
            ਵਾਪਸ ggml_zendnn_ਸਮੂਹ_matmul<ਖੰਡ_q8_0, ਦਸ਼ਮਲਵ, ਦਸ਼ਮਲਵ>(ctx, m, k, n, A, lda, B, ldb, C, ldc);
        ਮੂਲ:
            ਵਾਪਸ ਝੂਠ; // unsupported type
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_zendnn_compute_forward_mul_mat_id(
    ggml_backend_zendnn_context * ctx,
    ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];  // expert weights
    ਅਚਲ ggml_tensor * src1 = dst->src[1];  // inputs
    ਅਚਲ ggml_tensor * ids  = dst->src[2];  // expert ids

    GGML_TENSOR_ਦੁਆਧਾਰੀ_OP_LOCALS

    // exit for no tokens to process
    ਜੇ (ne2 == 0 || ne11 == 0) {
        ਵਾਪਸ;
    }

    ggml_type         ਅਚਲ vec_dot_type = src0->type;
    ggml_from_float_t ਅਚਲ from_float = ggml_get_type_traits(vec_dot_type)->from_float_ref;

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_ਮਾਪ(src0->type));
    GGML_ASSERT(nb10 == ggml_type_ਮਾਪ(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(ne03 == 1);
    GGML_ASSERT(ne13 == 1);
    GGML_ASSERT(ne3  == 1);

    // row groups
    ਅਚਲ ਪੂਰਨ n_ids = ids->ne[0]; // n_expert_used
    ਅਚਲ ਪੂਰਨ n_as  = ne02;       // n_experts

    std::vector<int64_t> ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts(n_as, 0);
    std::vector<std::vector<mmid_ਪੰਕਤੀ_mapping>> ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ(n_as);

    int64_t ਕੁੱਲ_ਪੰਕਤੀਆਂ = 0;
    ਪੂਰਨ     n_ਸਰਗਰਮ_experts = 0;
    // group rows by expert (preprocessing step)
    ਲਈ (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
        ਲਈ (ਪੂਰਨ id = 0; id < n_ids; ++id) {
            ਅਚਲ int32_t i02 = *(ਅਚਲ int32_t *)((ਅਚਲ ਅੱਖਰ *)ids->ਅੰਕੜੇ + iid1*ids->nb[1] + id*ids->nb[0]);

            GGML_ASSERT(i02 >= 0 && i02 < n_as);

            ਜੇ (ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts[i02] == 0) {
                n_ਸਰਗਰਮ_experts++;
            }
            ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ[i02].push_back({id, iid1});
            ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts[i02]++;
            ਕੁੱਲ_ਪੰਕਤੀਆਂ++;
        }
    }

    ਜੇ (ਕੁੱਲ_ਪੰਕਤੀਆਂ == 0) {
        ਵਾਪਸ; // no rows to process
    }

    ਅਚਲ size_t ਪੰਕਤੀ_ਮਾਪ = ggml_ਪੰਕਤੀ_ਮਾਪ(vec_dot_type, ne10);

    // For Q8_0, src1 is always F32; the gather buffer must hold F32 rows (ne10*4 bytes),
    // not Q8_0-encoded rows (row_size ≈ ne10/32*34 bytes) — they differ by ~4x.
    ਅਚਲ size_t f32_ਪੰਕਤੀ_ਮਾਪ = (size_t)ne10 * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ size_t gather_ਪੰਕਤੀ_ਮਾਪ = (src0->type == GGML_TYPE_Q8_0) ? f32_ਪੰਕਤੀ_ਮਾਪ : ਪੰਕਤੀ_ਮਾਪ;

    ਜੇ (src1->type != vec_dot_type && src0->type != GGML_TYPE_Q8_0) {
        GGML_ASSERT(src1->type == GGML_TYPE_F32);
    }

    // size for MoE gather/scatter buffers
    ਅਚਲ size_t wdata_cur_ਮਾਪ = ਕੁੱਲ_ਪੰਕਤੀਆਂ * gather_ਪੰਕਤੀ_ਮਾਪ;
    ਅਚਲ size_t dst_cur_ਮਾਪ = ਕੁੱਲ_ਪੰਕਤੀਆਂ * ggml_ਪੰਕਤੀ_ਮਾਪ(dst->type, ne01);

    // allocate single buffer for all needs
    ਅਚਲ size_t ਕੁੱਲ_ਮਾਪ = wdata_cur_ਮਾਪ + dst_cur_ਮਾਪ;
    ਜੇ (ctx->work_ਮਾਪ < ਕੁੱਲ_ਮਾਪ) {
        ctx->work_ਅੰਕੜੇ.reset(ਨਵਾਂ ਅੱਖਰ[ਕੁੱਲ_ਮਾਪ]);
        ctx->work_ਮਾਪ = ਕੁੱਲ_ਮਾਪ;
    }

    // partition the buffer
    ਅੱਖਰ * wdata_cur = ctx->work_ਅੰਕੜੇ.get();
    ਅੱਖਰ * dst_cur = wdata_cur + wdata_cur_ਮਾਪ;

    // per-expert data collected during gather, handed to ggml_zendnn_group_gemm() as one batch
    std::vector<int64_t>      expert_ਪੰਕਤੀ_ਗਿਣਤੀ(n_ਸਰਗਰਮ_experts);
    std::vector<ਅਚਲ ਖਾਲੀ *> batch_src(n_ਸਰਗਰਮ_experts);
    std::vector<ਅਚਲ ਖਾਲੀ *> batch_wei(n_ਸਰਗਰਮ_experts);
    std::vector<ਖਾਲੀ *>       batch_dst(n_ਸਰਗਰਮ_experts);

    // precompute per-expert buffer offsets and batch indices for the parallel loop below
    std::vector<int64_t> expert_wdata_off(n_as, 0);
    std::vector<int64_t> expert_dst_off(n_as, 0);
    std::vector<ਪੂਰਨ>     expert_batch_idx(n_as, -1);
    {
        int64_t w_off = 0;
        int64_t d_off = 0;
        ਪੂਰਨ     batch_idx = 0;
        ਲਈ (int64_t cur_a = 0; cur_a < n_as; ++cur_a) {
            ਜੇ (ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts[cur_a] == 0) {
                ਜਾਰੀ;
            }
            expert_wdata_off[cur_a] = w_off;
            expert_dst_off[cur_a]   = d_off;
            expert_batch_idx[cur_a] = batch_idx;
            w_off += ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts[cur_a] * gather_ਪੰਕਤੀ_ਮਾਪ;
            d_off += ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts[cur_a] * ggml_ਪੰਕਤੀ_ਮਾਪ(dst->type, ne01);
            batch_idx++;
        }
    }

    // gather + inline-convert input rows into each expert's batch slot
    #ਹਦਾਇਤ omp parallel ਲਈ ਅੰਕ_ਧਾਗੇ(ctx->n_ਧਾਗੇ) ਸਮਾਂ_ਸਾਰਣੀ(ਸਥਿਰ)
    ਲਈ (int64_t cur_a = 0; cur_a < n_as; ++cur_a) {
        ਅਚਲ int64_t cne1 = ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts[cur_a];

        ਜੇ (cne1 == 0) {
            ਜਾਰੀ;
        }

        ਅਚਲ int64_t w_off     = expert_wdata_off[cur_a];
        ਅਚਲ int64_t d_off     = expert_dst_off[cur_a];
        ਅਚਲ ਪੂਰਨ     batch_idx = expert_batch_idx[cur_a];

        ਲਈ (int64_t ir1 = 0; ir1 < cne1; ++ir1) {
            ਅਚਲ mmid_ਪੰਕਤੀ_mapping & ਪੰਕਤੀ_mapping = ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ[cur_a][ir1];
            ਅਚਲ int64_t id  = ਪੰਕਤੀ_mapping.i1;
            ਅਚਲ int64_t i11 = id % ne11;
            ਅਚਲ int64_t i12 = ਪੰਕਤੀ_mapping.i2;

            ਅਚਲ ਅੱਖਰ * src_ਪੰਕਤੀ = (ਅਚਲ ਅੱਖਰ *) src1->ਅੰਕੜੇ + i11*nb11 + i12*nb12;
            ਖਾਲੀ *       dst_ਪੰਕਤੀ = wdata_cur + w_off + ir1 * gather_ਪੰਕਤੀ_ਮਾਪ;

            ਜੇ (src1->type != vec_dot_type && src0->type != GGML_TYPE_Q8_0) {
                from_float((ਅਚਲ ਦਸ਼ਮਲਵ *) src_ਪੰਕਤੀ, dst_ਪੰਕਤੀ, ne10);
            } ਨਹੀਂਤਾਂ {
                // no conversion: src1 already matches vec_dot_type, or src0 is Q8_0, whose
                // ZenDNN dynamic quantization requires the row to stay in F32
                std::ਸਮ੍ਰਿਤੀ_ਨਕਲ(dst_ਪੰਕਤੀ, src_ਪੰਕਤੀ, gather_ਪੰਕਤੀ_ਮਾਪ);
            }
        }

        expert_ਪੰਕਤੀ_ਗਿਣਤੀ[batch_idx] = cne1;
        batch_src[batch_idx]        = wdata_cur + w_off;
        batch_wei[batch_idx]        = (ਅਚਲ ਅੱਖਰ *) src0->ਅੰਕੜੇ + cur_a * nb02;
        batch_dst[batch_idx]        = dst_cur + d_off;
    }

    ਜੇ (!ggml_zendnn_ਸਮੂਹ_gemm(ctx,
                                ne01,             // m
                                ne10,             // k
                                expert_ਪੰਕਤੀ_ਗਿਣਤੀ, // n (per expert)
                                batch_wei, ne00,  // A: weights, lda
                                batch_src, ne10,  // B: input, ldb
                                batch_dst, ne01,  // C: output, ldc
                                src0->type,
                                src0->type == GGML_TYPE_Q8_0 ? GGML_TYPE_F32 : vec_dot_type,
                                dst->type))
        GGML_ABORT("%s: ZenDNN group gemm failed\n", __func__);

    // scatter output rows to destination
    #ਹਦਾਇਤ omp parallel ਲਈ ਅੰਕ_ਧਾਗੇ(ctx->n_ਧਾਗੇ) ਸਮਾਂ_ਸਾਰਣੀ(ਸਥਿਰ)
    ਲਈ (int64_t cur_a = 0; cur_a < n_as; ++cur_a) {
        ਅਚਲ int64_t cne1 = ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_counts[cur_a];

        ਜੇ (cne1 == 0) {
            ਜਾਰੀ;
        }

        ਅਚਲ int64_t d_off = expert_dst_off[cur_a];

        ਲਈ (int64_t ir1 = 0; ir1 < cne1; ++ir1) {
            ਅਚਲ mmid_ਪੰਕਤੀ_mapping & ਪੰਕਤੀ_mapping = ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀਆਂ[cur_a][ir1];
            ਅਚਲ int64_t id = ਪੰਕਤੀ_mapping.i1;
            ਅਚਲ int64_t i1 = id;
            ਅਚਲ int64_t i2 = ਪੰਕਤੀ_mapping.i2;

            std::ਸਮ੍ਰਿਤੀ_ਨਕਲ(
                (ਅੱਖਰ *) dst->ਅੰਕੜੇ + i1*nb1 + i2*nb2,
                dst_cur + d_off + ir1 * ggml_ਪੰਕਤੀ_ਮਾਪ(dst->type, ne01),
                ggml_ਪੰਕਤੀ_ਮਾਪ(dst->type, ne01)
            );
        }
    }
}

// backend interface

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_zendnn_get_name(ggml_backend_t backend) {
    ਵਾਪਸ "ZenDNN";

    GGML_ਅਣਵਰਤਿਆ(backend);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_zendnn_free(ggml_backend_t backend) {
    ggml_backend_zendnn_context * ctx = (ggml_backend_zendnn_context *)backend->context;
    ਮਿਟਾਓ ctx;
    ਮਿਟਾਓ backend;
}

ਸਥਿਰ ggml_ਸਥਿਤੀ ggml_backend_zendnn_ਗ੍ਰਾਫ_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_zendnn_context * ctx = (ggml_backend_zendnn_context *)backend->context;

    ਲਈ (ਪੂਰਨ i = 0; i < cgraph->n_nodes; i++) {
        ਬਣਤਰ ggml_tensor * ਗੰਢ = cgraph->nodes[i];

        ਜੇ ((ਗੰਢ->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            ਜਾਰੀ;
        }

        ਚੋਣ (ਗੰਢ->op) {
            ਹਾਲਤ GGML_OP_MUL_MAT:
                ggml_zendnn_compute_forward_mul_mat(ctx, ਗੰਢ);
                ਰੋਕੋ;
            ਹਾਲਤ GGML_OP_MUL_MAT_ID:
                ggml_zendnn_compute_forward_mul_mat_id(ctx, ਗੰਢ);
                ਰੋਕੋ;
            ਹਾਲਤ GGML_OP_NONE:
            ਹਾਲਤ GGML_OP_RESHAPE:
            ਹਾਲਤ GGML_OP_VIEW:
            ਹਾਲਤ GGML_OP_ਕ੍ਰਮ_ਬਦਲੋ:
            ਹਾਲਤ GGML_OP_TRANSPOSE:
                ਰੋਕੋ;

            ਮੂਲ:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(ਗੰਢ));
        }
    }

    ਵਾਪਸ GGML_ਸਥਿਤੀ_SUCCESS;

    GGML_ਅਣਵਰਤਿਆ(backend);
}

ਸਥਿਰ ਬਣਤਰ ggml_backend_i ggml_backend_zendnn_i = {
    /* .get_name                = */ ggml_backend_zendnn_get_name,
    /* .free                    = */ ggml_backend_zendnn_free,
    /* .set_tensor_async        = */ ਨਿਰਾ,
    /* .get_tensor_async        = */ ਨਿਰਾ,
    /* .set_tensor_2d_async     = */ ਨਿਰਾ,
    /* .get_tensor_2d_async     = */ ਨਿਰਾ,
    /* .cpy_tensor_async        = */ ਨਿਰਾ,
    /* .synchronize             = */ ਨਿਰਾ,
    /* .graph_plan_create       = */ ਨਿਰਾ,
    /* .graph_plan_free         = */ ਨਿਰਾ,
    /* .graph_plan_update       = */ ਨਿਰਾ,
    /* .graph_plan_compute      = */ ਨਿਰਾ,
    /* .graph_compute           = */ ggml_backend_zendnn_ਗ੍ਰਾਫ_compute,
    /* .event_record            = */ ਨਿਰਾ,
    /* .event_wait              = */ ਨਿਰਾ,
    /* .graph_optimize          = */ ਨਿਰਾ,
};

ਸਥਿਰ ggml_guid_t ggml_backend_zendnn_guid(ਖਾਲੀ) {
    ਸਥਿਰ ਅਚਲ ਅੱਖਰ * guid_str = "AMD-ZENDNN-ACCEL";
    ਵਾਪਸ reinterpret_cast<ggml_guid_t>(const_cast<ਅੱਖਰ*>(guid_str));
}

ggml_backend_t ggml_backend_zendnn_init(ਖਾਲੀ) {
    ggml_backend_zendnn_context * ctx = ਨਵਾਂ ggml_backend_zendnn_context;

    ggml_backend_t backend = ਨਵਾਂ ggml_backend {
        /* .guid    = */ ggml_backend_zendnn_guid(),
        /* .iface   = */ ggml_backend_zendnn_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_zendnn_reg(), 0),
        /* .context = */ ctx,
    };

    ਵਾਪਸ backend;
}

ਤਾਰਕਿਕ ggml_backend_is_zendnn(ggml_backend_t backend) {
    ਵਾਪਸ backend != ਨਿਰਾ && ggml_guid_matches(backend->guid, ggml_backend_zendnn_guid());
}

ਖਾਲੀ ggml_backend_zendnn_ਸੈੱਟ_n_ਧਾਗੇ(ggml_backend_t backend_zendnn, ਪੂਰਨ n_ਧਾਗੇ) {
    GGML_ASSERT(ggml_backend_is_zendnn(backend_zendnn));

    ggml_backend_zendnn_context * ctx = (ggml_backend_zendnn_context *)backend_zendnn->context;
    ctx->n_ਧਾਗੇ = n_ਧਾਗੇ;
}

// device interface
ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_zendnn_device_get_name(ggml_backend_dev_t dev) {
    ਵਾਪਸ "ZenDNN";

    GGML_ਅਣਵਰਤਿਆ(dev);
}
/**
 * ZenDNN is AMD's performance library providing optimized primitives and implementations
 * for deep learning workloads on AMD CPUs. It targets improved performance for common
 * neural network operations on AMD architectures. For more information, see:
 * https://www.amd.com/en/developer/zendnn.html
 */
ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_zendnn_device_get_description(ggml_backend_dev_t dev) {
    ਵਾਪਸ "ZenDNN: AMD optimized primitives backend for GGML (optimized for AMD CPUs)";

    GGML_ਅਣਵਰਤਿਆ(dev);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_zendnn_device_get_ਸਮ੍ਰਿਤੀ(ggml_backend_dev_t dev, size_t * ਮੁਕਤ, size_t * ਕੁੱਲ) {
    *ਮੁਕਤ  = 0;
    *ਕੁੱਲ = 0;

    GGML_ਅਣਵਰਤਿਆ(dev);
}

ਸਥਿਰ ਗਿਣਤੀ ggml_backend_dev_type ggml_backend_zendnn_device_get_type(ggml_backend_dev_t dev) {
    ਵਾਪਸ GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_ਅਣਵਰਤਿਆ(dev);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_zendnn_device_get_props(ggml_backend_dev_t dev, ਬਣਤਰ ggml_backend_dev_props * props) {
    props->name        = ggml_backend_zendnn_device_get_name(dev);
    props->description = ggml_backend_zendnn_device_get_description(dev);
    props->type        = ggml_backend_zendnn_device_get_type(dev);
    ggml_backend_zendnn_device_get_ਸਮ੍ਰਿਤੀ(dev, &props->ਸਮ੍ਰਿਤੀ_free, &props->ਸਮ੍ਰਿਤੀ_ਕੁੱਲ);
    props->caps = {
        /* .async                = */ ਝੂਠ,
        /* .host_buffer          = */ ਝੂਠ,
        /* .buffer_from_host_ptr = */ ਸੱਚ,
        /* .events               = */ ਝੂਠ,
        /* .mmap_support         = */ ਸੱਚ,
    };
}

ਸਥਿਰ ggml_backend_t ggml_backend_zendnn_device_init_backend(ggml_backend_dev_t dev, ਅਚਲ ਅੱਖਰ * params) {
    ggml_backend_t backend = ggml_backend_zendnn_init();
    ਜੇ (backend == ਨਿਰਾ) {
        GGML_LOG_ERROR("%s: error: failed to initialize ZenDNN backend\n", __func__);
        ਵਾਪਸ ਨਿਰਾ;
    }

    ਵਾਪਸ backend;

    GGML_ਅਣਵਰਤਿਆ(dev);
    GGML_ਅਣਵਰਤਿਆ(params);
}

ਸਥਿਰ ggml_backend_ਬਫਰ_type_t ggml_backend_zendnn_device_get_ਬਫਰ_type(ggml_backend_dev_t dev) {
    ਵਾਪਸ ggml_backend_cpu_ਬਫਰ_type();

    GGML_ਅਣਵਰਤਿਆ(dev);
}

ਸਥਿਰ ggml_backend_ਬਫਰ_t ggml_backend_zendnn_device_ਬਫਰ_from_host_ptr(ggml_backend_dev_t dev, ਖਾਲੀ * ptr, size_t size, size_t max_tensor_ਮਾਪ) {
    ਵਾਪਸ ggml_backend_cpu_ਬਫਰ_from_ptr(ptr, size);

    GGML_ਅਣਵਰਤਿਆ(dev);
    GGML_ਅਣਵਰਤਿਆ(max_tensor_ਮਾਪ);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_zendnn_adaptive_fallback_enabled() {
    ਸਥਿਰ ਅਚਲ ਤਾਰਕਿਕ enabled = std::getenv("GGML_ZENDNN_ADAPTIVE_FALLBACK") == ਖਾਲੀ_ਸੂਚਕ ||
                                std::atoi(std::getenv("GGML_ZENDNN_ADAPTIVE_FALLBACK")) != 0;
    ਵਾਪਸ enabled;
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_zendnn_device_supports_op(ggml_backend_dev_t dev, ਅਚਲ ਬਣਤਰ ggml_tensor * op) {
    ਚੋਣ (op->op) {
        ਹਾਲਤ GGML_OP_NONE:
        ਹਾਲਤ GGML_OP_RESHAPE:
        ਹਾਲਤ GGML_OP_VIEW:
        ਹਾਲਤ GGML_OP_ਕ੍ਰਮ_ਬਦਲੋ:
        ਹਾਲਤ GGML_OP_TRANSPOSE:
            ਵਾਪਸ ਸੱਚ;

        ਹਾਲਤ GGML_OP_MUL_MAT:
        ਹਾਲਤ GGML_OP_MUL_MAT_ID:
        {
            ਅਚਲ ggml_tensor * weights = op->src[0];
            ਅਚਲ ggml_tensor * inputs = op->src[1];

            ਅਚਲ int64_t ne10 = inputs->ne[0];
            ਅਚਲ int64_t ne0 = op->ne[0];
            ਅਚਲ int64_t ne1 = op->ne[1];
            ਅਚਲ int64_t min_batch = 1;

            ਜੇ(!ggml_is_contiguous(weights) || !ggml_is_contiguous(inputs)) {
                ਵਾਪਸ ਝੂਠ;
            }

            ਜੇ (ggml_zendnn_adaptive_fallback_enabled()) {
                ਅਚਲ int64_t K = inputs->ne[0];
                ਅਚਲ int64_t N = (inputs->ne[1]*inputs->ne[2]*inputs->ne[3]);
                ਅਚਲ int64_t M = weights->ne[1];
                ਜੇ(K <= 256 || N <= 128 || M <= 96) {
                    ਵਾਪਸ ਝੂਠ;
                }

                // MUL_MAT_ID's gather+matmul+scatter approach favors a moderate expert count
                ਜੇ (op->op == GGML_OP_MUL_MAT_ID) {
                    ਅਚਲ int64_t n_experts = weights->ne[2];
                    ਅਚਲ int64_t max_experts = 32;
                    ਜੇ (n_experts > max_experts) {
                        ਵਾਪਸ ਝੂਠ;
                    }

                    // fall back once the average rows per expert (N / n_experts) is too thin
                    // to amortize each per-expert GEMM's overhead
                    ਜੇ (N / n_experts <= 32) {
                        ਵਾਪਸ ਝੂਠ;
                    }
                }
            }
            ਨਹੀਂਤਾਂ ਜੇ (ne0 < min_batch || ne1 < min_batch || ne10 < min_batch) {
                ਵਾਪਸ ਝੂਠ;
            }

            ਚੋਣ (weights->type) {
                ਹਾਲਤ GGML_TYPE_F32:
                ਹਾਲਤ GGML_TYPE_BF16:
                ਹਾਲਤ GGML_TYPE_Q8_0:
                    ਵਾਪਸ ਸੱਚ;
                ਮੂਲ:
                    ਵਾਪਸ ਝੂਠ;
            }
        } ਰੋਕੋ;

        ਮੂਲ:
            ਵਾਪਸ ਝੂਠ;
    }

    GGML_ਅਣਵਰਤਿਆ(dev);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_zendnn_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_ਬਫਰ_type_t buft) {
    ਵਾਪਸ ggml_backend_buft_is_host(buft);

    GGML_ਅਣਵਰਤਿਆ(dev);
}

ਸਥਿਰ ਅਚਲ ਬਣਤਰ ggml_backend_device_i ggml_backend_zendnn_device_i = {
    /* .get_name               = */ ggml_backend_zendnn_device_get_name,
    /* .get_description        = */ ggml_backend_zendnn_device_get_description,
    /* .get_memory             = */ ggml_backend_zendnn_device_get_ਸਮ੍ਰਿਤੀ,
    /* .get_type               = */ ggml_backend_zendnn_device_get_type,
    /* .get_props              = */ ggml_backend_zendnn_device_get_props,
    /* .init_backend           = */ ggml_backend_zendnn_device_init_backend,
    /* .get_buffer_type        = */ ggml_backend_zendnn_device_get_ਬਫਰ_type,
    /* .get_host_buffer_type   = */ ਨਿਰਾ,
    /* .buffer_from_host_ptr   = */ ggml_backend_zendnn_device_ਬਫਰ_from_host_ptr,
    /* .supports_op            = */ ggml_backend_zendnn_device_supports_op,
    /* .supports_buft          = */ ggml_backend_zendnn_device_supports_buft,
    /* .offload_op             = */ ਨਿਰਾ,
    /* .event_new              = */ ਨਿਰਾ,
    /* .event_free             = */ ਨਿਰਾ,
    /* .event_synchronize      = */ ਨਿਰਾ,
};

// backend reg interface
ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_zendnn_reg_get_name(ggml_backend_reg_t reg) {
    ਵਾਪਸ "ZenDNN";

    GGML_ਅਣਵਰਤਿਆ(reg);
}

ਸਥਿਰ size_t ggml_backend_zendnn_reg_get_device_ਗਿਣਤੀ(ggml_backend_reg_t reg) {
    ਵਾਪਸ 1;

    GGML_ਅਣਵਰਤਿਆ(reg);
}

ਸਥਿਰ ggml_backend_dev_t ggml_backend_zendnn_reg_get_device(ggml_backend_reg_t reg, size_t ਸੂਚਕਾਂਕ) {
    GGML_ASSERT(ਸੂਚਕਾਂਕ == 0);

    ਸਥਿਰ ggml_backend_device ggml_backend_zendnn_device = {
        /* .iface   = */ ggml_backend_zendnn_device_i,
        /* .reg     = */ reg,
        /* .context = */ ਖਾਲੀ_ਸੂਚਕ,
    };

    ਵਾਪਸ &ggml_backend_zendnn_device;
}

ਸਥਿਰ ਖਾਲੀ * ggml_backend_zendnn_get_proc_ਪਤਾ(ggml_backend_reg_t reg, ਅਚਲ ਅੱਖਰ * name) {
    ਜੇ (std::strcmp(name, "ggml_backend_set_n_threads") == 0) {
        ਵਾਪਸ (ਖਾਲੀ *) ggml_backend_zendnn_ਸੈੱਟ_n_ਧਾਗੇ;
    }
    ਵਾਪਸ ਨਿਰਾ;

    GGML_ਅਣਵਰਤਿਆ(reg);
    GGML_ਅਣਵਰਤਿਆ(name);
}

ਸਥਿਰ ਅਚਲ ਬਣਤਰ ggml_backend_reg_i ggml_backend_zendnn_reg_i = {
    /* .get_name         = */ ggml_backend_zendnn_reg_get_name,
    /* .get_device_count = */ ggml_backend_zendnn_reg_get_device_ਗਿਣਤੀ,
    /* .get_device       = */ ggml_backend_zendnn_reg_get_device,
    /* .get_proc_address = */ ggml_backend_zendnn_get_proc_ਪਤਾ,
};

ggml_backend_reg_t ggml_backend_zendnn_reg(ਖਾਲੀ) {
    ਸਥਿਰ ਬਣਤਰ ggml_backend_reg ggml_backend_zendnn_reg = {
        /* .api_version = */ GGML_BACKEND_API_ਸੰਸਕਰਣ,
        /* .iface       = */ ggml_backend_zendnn_reg_i,
        /* .context     = */ ਨਿਰਾ,
    };

    ਵਾਪਸ &ggml_backend_zendnn_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_zendnn_reg)
