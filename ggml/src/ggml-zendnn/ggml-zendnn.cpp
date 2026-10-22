#అన్తర్భావ "ggml-zendnn.h"

#అన్తర్భావ "ggml-backend-impl.h"
#అన్తర్భావ "ggml-impl.h"

#నిర్వచించు GGML_COMMON_DECL_CPP
#అన్తర్భావ "ggml-common.h"

#అన్తర్భావ "zendnnl.hpp"

#అన్తర్భావ <cstring>


నిర్మాణం ggml_backend_zendnn_context {
    పూర్ణాంకం n_దారాలు = GGML_DEFAULT_N_దారాలు;
    std::unique_ptr<అక్షరం[]> work_దత్తాంశం;
    size_t work_కొలత = 0;
};

మూస<రకం_పేరు T>
zendnnl::common::దత్తాంశం_type_t ggml_గమ్యం_zendnn_type() {
    ఒకవేళ constexpr (std::is_same_v<T, దశాంశం>) {
        తిరిగి zendnnl::common::దత్తాంశం_type_t::f32;
    } లేకపోతే ఒకవేళ constexpr (std::is_same_v<T, ggml_bf16_t>) {
        తిరిగి zendnnl::common::దత్తాంశం_type_t::bf16;
    } లేకపోతే ఒకవేళ constexpr (std::is_same_v<T, ఖండం_q8_0>) {
        తిరిగి zendnnl::common::దత్తాంశం_type_t::s8;
    } లేకపోతే {
        తిరిగి zendnnl::common::దత్తాంశం_type_t::none;
    }
}

/**
 * Builds the matmul_params shared by ggml_zendnn_matmul() and ggml_zendnn_group_matmul():
 * dtype selection plus, for Q8_0 weights, dynamic-quant setup. Callers still need to set
 * quant_params.src_scale.dims themselves, since that depends on the batch size(s) in use.
 */
మూస <రకం_పేరు TA, రకం_పేరు TB, రకం_పేరు TC>
స్థిర zendnnl::lowoha::matmul::matmul_params ggml_zendnn_make_matmul_params(ggml_backend_zendnn_context * ctx) {
    zendnnl::lowoha::matmul::matmul_params params;
    params.dtypes.src = ggml_గమ్యం_zendnn_type<TB>();
    params.dtypes.wei = ggml_గమ్యం_zendnn_type<TA>();
    params.dtypes.dst = ggml_గమ్యం_zendnn_type<TC>();
    params.అంకె_దారాలు = ctx->n_దారాలు;

    ఒకవేళ constexpr (std::is_same_v<TA, ఖండం_q8_0>) {
        params.dtypes.compute = zendnnl::common::దత్తాంశం_type_t::s8;
        params.dynamic_quant = నిజం;
        params.quant_params.src_scale.buff = శూన్యసూచిక;
        params.quant_params.src_scale.dt   = zendnnl::common::దత్తాంశం_type_t::bf16;
        params.packing.pack_format_b = 1;
    }
    తిరిగి params;
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
మూస <రకం_పేరు TA, రకం_పేరు TB, రకం_పేరు TC>
స్థిర తార్కిక ggml_zendnn_matmul(ggml_backend_zendnn_context * ctx, int64_t m, int64_t n, int64_t k,
                               స్థిరరాశి TA * A, int64_t lda, స్థిరరాశి TB * B, int64_t ldb, TC * C,
                               int64_t ldc) {

    zendnnl::lowoha::matmul::matmul_params params = ggml_zendnn_make_matmul_params<TA, TB, TC>(ctx);

    zendnnl::lowoha::matmul::matmul_batch_params_t batch_params;

    ఒకవేళ constexpr (std::is_same_v<TA, ఖండం_q8_0>) {
        params.quant_params.src_scale.dims = {n, k / QK8_0};
    }

    zendnnl::error_handling::స్థితి_t స్థితి_2 = zendnnl::lowoha::matmul::matmul_ప్రత్యక్ష(
        'r', అబద్ధం, నిజం,   // row-major, don't transpose B, transpose A (because it's column-major)
        n,                  // M: rows of B and C
        m,                  // N: cols of A^T and C
        k,                  // K: cols of B, rows of A
        1.0f,               // alpha
        B, ldb,             // src: B[n,k]
        A, lda,             // weight: A[k,m] column-major (transposed)
        శూన్యసూచిక,            // bias
        0.0f,               // beta
        C, ldc,             // output C[n,m]
        నిజం,               // is_weights_const
        batch_params,       // batch_params
        params              // params
    );

    ఒకవేళ (స్థితి_2 != zendnnl::error_handling::స్థితి_t::success) {
        GGML_LOG_ERROR("%s, ZenDNN matmul failed: status=%d\n", __func__, static_cast<పూర్ణాంకం>(స్థితి_2));
        తిరిగి అబద్ధం;
    }
    తిరిగి నిజం;
}

స్థిర తార్కిక ggml_zendnn_gemm(ggml_backend_zendnn_context * ctx, int64_t m, int64_t n, int64_t k,
                              స్థిరరాశి శూన్య * A, int64_t lda, స్థిరరాశి శూన్య * B, int64_t ldb, శూన్య * C,
                              int64_t ldc, పూర్ణాంకం Atype, పూర్ణాంకం Btype, పూర్ణాంకం Ctype) {

    ప్రతిజ్ఞా(m >= 0);
    ప్రతిజ్ఞా(n >= 0);
    ప్రతిజ్ఞా(k >= 0);
    ప్రతిజ్ఞా(lda >= k);
    ప్రతిజ్ఞా(ldb >= k);
    ప్రతిజ్ఞా(ldc >= m);

    // categorize types
    ఎంపిక (Atype) {
        స్థితి GGML_TYPE_F32:
            ఒకవేళ (Btype != GGML_TYPE_F32 || Ctype != GGML_TYPE_F32)
                తిరిగి అబద్ధం;
            తిరిగి ggml_zendnn_matmul<దశాంశం, దశాంశం, దశాంశం>(
                ctx, m, n, k,
                (స్థిరరాశి దశాంశం *)A, lda,
                (స్థిరరాశి దశాంశం *)B, ldb,
                (దశాంశం *)C, ldc);
        స్థితి GGML_TYPE_BF16:
            ఒకవేళ (Btype != GGML_TYPE_BF16)
                తిరిగి అబద్ధం;
            ఒకవేళ (Ctype == GGML_TYPE_BF16)
                తిరిగి ggml_zendnn_matmul<ggml_bf16_t, ggml_bf16_t, ggml_bf16_t>(
                    ctx, m, n, k,
                    (స్థిరరాశి ggml_bf16_t *)A, lda,
                    (స్థిరరాశి ggml_bf16_t *)B, ldb,
                    (ggml_bf16_t *)C, ldc);
            ఒకవేళ (Ctype == GGML_TYPE_F32)
                తిరిగి ggml_zendnn_matmul<ggml_bf16_t, ggml_bf16_t, దశాంశం>(
                    ctx, m, n, k,
                    (స్థిరరాశి ggml_bf16_t *)A, lda,
                    (స్థిరరాశి ggml_bf16_t *)B, ldb,
                    (దశాంశం *)C, ldc);
            తిరిగి అబద్ధం;
        స్థితి GGML_TYPE_Q8_0:
            ఒకవేళ (Btype != GGML_TYPE_F32 || Ctype != GGML_TYPE_F32)
                తిరిగి అబద్ధం;
            తిరిగి ggml_zendnn_matmul<ఖండం_q8_0, దశాంశం, దశాంశం>(
                ctx, m, n, k,
                (స్థిరరాశి ఖండం_q8_0 *)A, lda,
                (స్థిరరాశి దశాంశం *)B, ldb,
                (దశాంశం *)C, ldc);
        అప్రమేయ:
            తిరిగి అబద్ధం; // unsupported type
    }
}

స్థిర శూన్య ggml_zendnn_compute_forward_mul_mat(
    ggml_backend_zendnn_context * ctx,
    ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];  // weights
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];  // inputs

    GGML_TENSOR_ద్విమాన_OP_LOCALS

    ggml_type         స్థిరరాశి vec_dot_type = src0->type;
    ggml_from_float_t స్థిరరాశి from_float = ggml_get_type_traits(vec_dot_type)->from_float_ref;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_కొలత(src0->type));
    GGML_ASSERT(nb10 == ggml_type_కొలత(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == పరిమాణం(దశాంశం));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // broadcast factors
    స్థిరరాశి int64_t r2 = ne12/ne02;
    స్థిరరాశి int64_t r3 = ne13/ne03;

    శూన్య * work_దత్తాంశం = ctx->work_దత్తాంశం.get();

    // ZenDNN requires FP32 for dynamic quantization, so conversion is skipped
    ఒకవేళ (src1->type != vec_dot_type && src0->type != GGML_TYPE_Q8_0) {
        స్థిరరాశి size_t nbw1 = ggml_అడ్డువరుస_కొలత(vec_dot_type, ne10);
        స్థిరరాశి size_t nbw2 = nbw1 * ne11;
        స్థిరరాశి size_t nbw3 = nbw2 * ne12;
        స్థిరరాశి size_t desired_wsize = ne13 * nbw3;
        ఒకవేళ (ctx->work_కొలత < desired_wsize) {
            ctx->work_దత్తాంశం.reset(కొత్త అక్షరం[desired_wsize]);
            ctx->work_కొలత = desired_wsize;
        }
        work_దత్తాంశం = ctx->work_దత్తాంశం.get();

        // #pragma omp parallel for num_threads(ctx->n_threads)
        #సూచన omp parallel కొరకు collapse(3) అంకె_దారాలు(ctx->n_దారాలు) కాలపట్టిక(స్థిర)
        కొరకు (int64_t i13 = 0; i13 < ne13; ++i13) {
            కొరకు (int64_t i12 = 0; i12 < ne12; ++i12) {
                కొరకు (int64_t i11 = 0; i11 < ne11; ++i11) {
                    స్థిరరాశి దశాంశం * src1_f32 = (దశాంశం *)((అక్షరం *)src1->దత్తాంశం + i11*nb11 + i12*nb12 + i13*nb13);
                    శూన్య * src1_conv = (అక్షరం *)work_దత్తాంశం + i11*nbw1 + i12*nbw2 + i13*nbw3;
                    from_float(src1_f32, src1_conv, ne10);
                }
            }
        }
    }

    కొరకు (int64_t i13 = 0; i13 < ne13; i13++) {
        కొరకు (int64_t i12 = 0; i12 < ne12; i12++) {
            స్థిరరాశి శూన్య* wdata = (src1->type == vec_dot_type || src0->type == GGML_TYPE_Q8_0) ? src1->దత్తాంశం : work_దత్తాంశం;
            స్థిరరాశి size_t అడ్డువరుస_కొలత = ggml_అడ్డువరుస_కొలత(vec_dot_type, ne10);
            ఒకవేళ (!ggml_zendnn_gemm(ctx,
                                  ne01,     // m
                                  ne11,     // n
                                  ne10,     // k
                                  static_cast<స్థిరరాశి అక్షరం *>(src0->దత్తాంశం) + (i12/r2)*nb02 + (i13/r3)*nb03,
                                  ne00,     // lda
                                  static_cast<స్థిరరాశి అక్షరం *>(wdata) + (i12*ne11 + i13*ne12*ne11)*అడ్డువరుస_కొలత,
                                  ne10,     // ldb
                                  static_cast<అక్షరం *>(dst->దత్తాంశం) + i12*nb2 + i13*nb3,
                                  ne01,     // ldc
                                  src0->type,
                                  src0->type == GGML_TYPE_Q8_0 ? GGML_TYPE_F32 : vec_dot_type,
                                  dst->type))
                GGML_ABORT("%s: ZenDNN gemm failed\n", __func__);
        }
    }
}

నిర్మాణం mmid_అడ్డువరుస_mapping {
    int32_t i1;
    int32_t i2;
};

/**
 * ZenDNN batched matmul: computes C[i] = B[i] * A[i] for every active expert i via a single
 * group_matmul_direct() call. Batched analogue of ggml_zendnn_matmul() - see its docs for the
 * per-expert A/B/C shape convention. m and k are shared by every expert; n (batch size) varies
 * per expert, hence the vector.
 */
మూస <రకం_పేరు TA, రకం_పేరు TB, రకం_పేరు TC>
స్థిర తార్కిక ggml_zendnn_సమూహం_matmul(ggml_backend_zendnn_context * ctx, int64_t m, int64_t k,
                                     స్థిరరాశి std::vector<int64_t> & n,
                                     స్థిరరాశి std::vector<స్థిరరాశి శూన్య *> & A, int64_t lda,
                                     స్థిరరాశి std::vector<స్థిరరాశి శూన్య *> & B, int64_t ldb,
                                     స్థిరరాశి std::vector<శూన్య *> & C, int64_t ldc) {

    స్థిరరాశి పూర్ణాంకం n_experts = n.size();

    zendnnl::lowoha::matmul::matmul_params పునాది_params = ggml_zendnn_make_matmul_params<TA, TB, TC>(ctx);

    std::vector<అక్షరం>         layout(n_experts, 'r');
    std::vector<తార్కిక>         trans_a(n_experts, అబద్ధం);
    std::vector<తార్కిక>         trans_b(n_experts, నిజం);
    std::vector<పూర్ణాంకం>          batch_m(n_experts);
    std::vector<పూర్ణాంకం>          batch_n(n_experts, m);
    std::vector<పూర్ణాంకం>          batch_k(n_experts, k);
    std::vector<దశాంశం>        alpha(n_experts, 1.0f);
    std::vector<దశాంశం>        beta(n_experts, 0.0f);
    std::vector<స్థిరరాశి శూన్య *> bias(n_experts, శూన్యసూచిక);
    std::vector<పూర్ణాంకం>          lda_v(n_experts, lda);
    std::vector<పూర్ణాంకం>          ldb_v(n_experts, ldb);
    std::vector<పూర్ణాంకం>          ldc_v(n_experts, ldc);
    std::vector<తార్కిక>         is_wei_const(n_experts, నిజం);
    std::vector<zendnnl::lowoha::matmul::matmul_params> params(n_experts, పునాది_params);

    కొరకు (పూర్ణాంకం i = 0; i < n_experts; i++) {
        batch_m[i] = n[i];

        // src_scale.dims depends on this expert's row count, unlike the rest of base_params
        ఒకవేళ constexpr (std::is_same_v<TA, ఖండం_q8_0>) {
            params[i].quant_params.src_scale.dims = {n[i], k / QK8_0};
        }
    }

    zendnnl::error_handling::స్థితి_t స్థితి_2 = zendnnl::lowoha::matmul::సమూహం_matmul_ప్రత్యక్ష(
        layout, trans_a, trans_b, batch_m, batch_n, batch_k, alpha,
        B, ldb_v, A, lda_v, bias, beta,
        C, ldc_v, is_wei_const, params);

    ఒకవేళ (స్థితి_2 != zendnnl::error_handling::స్థితి_t::success) {
        GGML_LOG_ERROR("%s, ZenDNN group matmul failed: status=%d\n", __func__, static_cast<పూర్ణాంకం>(స్థితి_2));
        తిరిగి అబద్ధం;
    }
    తిరిగి నిజం;
}

స్థిర తార్కిక ggml_zendnn_సమూహం_gemm(ggml_backend_zendnn_context * ctx, int64_t m, int64_t k,
                                    స్థిరరాశి std::vector<int64_t> & n,
                                    స్థిరరాశి std::vector<స్థిరరాశి శూన్య *> & A, int64_t lda,
                                    స్థిరరాశి std::vector<స్థిరరాశి శూన్య *> & B, int64_t ldb,
                                    స్థిరరాశి std::vector<శూన్య *> & C, int64_t ldc,
                                    పూర్ణాంకం Atype, పూర్ణాంకం Btype, పూర్ణాంకం Ctype) {

    ప్రతిజ్ఞా(m >= 0);
    కొరకు (size_t i = 0; i < n.size(); i++) {
        ప్రతిజ్ఞా(n[i] >= 0);
    }
    ప్రతిజ్ఞా(k >= 0);
    ప్రతిజ్ఞా(lda >= k);
    ప్రతిజ్ఞా(ldb >= k);
    ప్రతిజ్ఞా(ldc >= m);

    // categorize types
    ఎంపిక (Atype) {
        స్థితి GGML_TYPE_F32:
            ఒకవేళ (Btype != GGML_TYPE_F32 || Ctype != GGML_TYPE_F32)
                తిరిగి అబద్ధం;
            తిరిగి ggml_zendnn_సమూహం_matmul<దశాంశం, దశాంశం, దశాంశం>(ctx, m, k, n, A, lda, B, ldb, C, ldc);
        స్థితి GGML_TYPE_BF16:
            ఒకవేళ (Btype != GGML_TYPE_BF16)
                తిరిగి అబద్ధం;
            ఒకవేళ (Ctype == GGML_TYPE_BF16)
                తిరిగి ggml_zendnn_సమూహం_matmul<ggml_bf16_t, ggml_bf16_t, ggml_bf16_t>(
                    ctx, m, k, n, A, lda, B, ldb, C, ldc);
            ఒకవేళ (Ctype == GGML_TYPE_F32)
                తిరిగి ggml_zendnn_సమూహం_matmul<ggml_bf16_t, ggml_bf16_t, దశాంశం>(ctx, m, k, n, A, lda, B, ldb, C, ldc);
            తిరిగి అబద్ధం;
        స్థితి GGML_TYPE_Q8_0:
            ఒకవేళ (Btype != GGML_TYPE_F32 || Ctype != GGML_TYPE_F32)
                తిరిగి అబద్ధం;
            తిరిగి ggml_zendnn_సమూహం_matmul<ఖండం_q8_0, దశాంశం, దశాంశం>(ctx, m, k, n, A, lda, B, ldb, C, ldc);
        అప్రమేయ:
            తిరిగి అబద్ధం; // unsupported type
    }
}

స్థిర శూన్య ggml_zendnn_compute_forward_mul_mat_id(
    ggml_backend_zendnn_context * ctx,
    ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];  // expert weights
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];  // inputs
    స్థిరరాశి ggml_tensor * ids  = dst->src[2];  // expert ids

    GGML_TENSOR_ద్విమాన_OP_LOCALS

    // exit for no tokens to process
    ఒకవేళ (ne2 == 0 || ne11 == 0) {
        తిరిగి;
    }

    ggml_type         స్థిరరాశి vec_dot_type = src0->type;
    ggml_from_float_t స్థిరరాశి from_float = ggml_get_type_traits(vec_dot_type)->from_float_ref;

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_కొలత(src0->type));
    GGML_ASSERT(nb10 == ggml_type_కొలత(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == పరిమాణం(దశాంశం));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(ne03 == 1);
    GGML_ASSERT(ne13 == 1);
    GGML_ASSERT(ne3  == 1);

    // row groups
    స్థిరరాశి పూర్ణాంకం n_ids = ids->ne[0]; // n_expert_used
    స్థిరరాశి పూర్ణాంకం n_as  = ne02;       // n_experts

    std::vector<int64_t> మాతృక_అడ్డువరుస_counts(n_as, 0);
    std::vector<std::vector<mmid_అడ్డువరుస_mapping>> మాతృక_అడ్డువరుసలు(n_as);

    int64_t మొత్తం_అడ్డువరుసలు = 0;
    పూర్ణాంకం     n_చురుకైన_experts = 0;
    // group rows by expert (preprocessing step)
    కొరకు (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
        కొరకు (పూర్ణాంకం id = 0; id < n_ids; ++id) {
            స్థిరరాశి int32_t i02 = *(స్థిరరాశి int32_t *)((స్థిరరాశి అక్షరం *)ids->దత్తాంశం + iid1*ids->nb[1] + id*ids->nb[0]);

            GGML_ASSERT(i02 >= 0 && i02 < n_as);

            ఒకవేళ (మాతృక_అడ్డువరుస_counts[i02] == 0) {
                n_చురుకైన_experts++;
            }
            మాతృక_అడ్డువరుసలు[i02].push_back({id, iid1});
            మాతృక_అడ్డువరుస_counts[i02]++;
            మొత్తం_అడ్డువరుసలు++;
        }
    }

    ఒకవేళ (మొత్తం_అడ్డువరుసలు == 0) {
        తిరిగి; // no rows to process
    }

    స్థిరరాశి size_t అడ్డువరుస_కొలత = ggml_అడ్డువరుస_కొలత(vec_dot_type, ne10);

    // For Q8_0, src1 is always F32; the gather buffer must hold F32 rows (ne10*4 bytes),
    // not Q8_0-encoded rows (row_size ≈ ne10/32*34 bytes) — they differ by ~4x.
    స్థిరరాశి size_t f32_అడ్డువరుస_కొలత = (size_t)ne10 * పరిమాణం(దశాంశం);
    స్థిరరాశి size_t gather_అడ్డువరుస_కొలత = (src0->type == GGML_TYPE_Q8_0) ? f32_అడ్డువరుస_కొలత : అడ్డువరుస_కొలత;

    ఒకవేళ (src1->type != vec_dot_type && src0->type != GGML_TYPE_Q8_0) {
        GGML_ASSERT(src1->type == GGML_TYPE_F32);
    }

    // size for MoE gather/scatter buffers
    స్థిరరాశి size_t wdata_cur_కొలత = మొత్తం_అడ్డువరుసలు * gather_అడ్డువరుస_కొలత;
    స్థిరరాశి size_t dst_cur_కొలత = మొత్తం_అడ్డువరుసలు * ggml_అడ్డువరుస_కొలత(dst->type, ne01);

    // allocate single buffer for all needs
    స్థిరరాశి size_t మొత్తం_కొలత = wdata_cur_కొలత + dst_cur_కొలత;
    ఒకవేళ (ctx->work_కొలత < మొత్తం_కొలత) {
        ctx->work_దత్తాంశం.reset(కొత్త అక్షరం[మొత్తం_కొలత]);
        ctx->work_కొలత = మొత్తం_కొలత;
    }

    // partition the buffer
    అక్షరం * wdata_cur = ctx->work_దత్తాంశం.get();
    అక్షరం * dst_cur = wdata_cur + wdata_cur_కొలత;

    // per-expert data collected during gather, handed to ggml_zendnn_group_gemm() as one batch
    std::vector<int64_t>      expert_అడ్డువరుస_లెక్క(n_చురుకైన_experts);
    std::vector<స్థిరరాశి శూన్య *> batch_src(n_చురుకైన_experts);
    std::vector<స్థిరరాశి శూన్య *> batch_wei(n_చురుకైన_experts);
    std::vector<శూన్య *>       batch_dst(n_చురుకైన_experts);

    // precompute per-expert buffer offsets and batch indices for the parallel loop below
    std::vector<int64_t> expert_wdata_off(n_as, 0);
    std::vector<int64_t> expert_dst_off(n_as, 0);
    std::vector<పూర్ణాంకం>     expert_batch_idx(n_as, -1);
    {
        int64_t w_off = 0;
        int64_t d_off = 0;
        పూర్ణాంకం     batch_idx = 0;
        కొరకు (int64_t cur_a = 0; cur_a < n_as; ++cur_a) {
            ఒకవేళ (మాతృక_అడ్డువరుస_counts[cur_a] == 0) {
                కొనసాగు;
            }
            expert_wdata_off[cur_a] = w_off;
            expert_dst_off[cur_a]   = d_off;
            expert_batch_idx[cur_a] = batch_idx;
            w_off += మాతృక_అడ్డువరుస_counts[cur_a] * gather_అడ్డువరుస_కొలత;
            d_off += మాతృక_అడ్డువరుస_counts[cur_a] * ggml_అడ్డువరుస_కొలత(dst->type, ne01);
            batch_idx++;
        }
    }

    // gather + inline-convert input rows into each expert's batch slot
    #సూచన omp parallel కొరకు అంకె_దారాలు(ctx->n_దారాలు) కాలపట్టిక(స్థిర)
    కొరకు (int64_t cur_a = 0; cur_a < n_as; ++cur_a) {
        స్థిరరాశి int64_t cne1 = మాతృక_అడ్డువరుస_counts[cur_a];

        ఒకవేళ (cne1 == 0) {
            కొనసాగు;
        }

        స్థిరరాశి int64_t w_off     = expert_wdata_off[cur_a];
        స్థిరరాశి int64_t d_off     = expert_dst_off[cur_a];
        స్థిరరాశి పూర్ణాంకం     batch_idx = expert_batch_idx[cur_a];

        కొరకు (int64_t ir1 = 0; ir1 < cne1; ++ir1) {
            స్థిరరాశి mmid_అడ్డువరుస_mapping & అడ్డువరుస_mapping = మాతృక_అడ్డువరుసలు[cur_a][ir1];
            స్థిరరాశి int64_t id  = అడ్డువరుస_mapping.i1;
            స్థిరరాశి int64_t i11 = id % ne11;
            స్థిరరాశి int64_t i12 = అడ్డువరుస_mapping.i2;

            స్థిరరాశి అక్షరం * src_అడ్డువరుస = (స్థిరరాశి అక్షరం *) src1->దత్తాంశం + i11*nb11 + i12*nb12;
            శూన్య *       dst_అడ్డువరుస = wdata_cur + w_off + ir1 * gather_అడ్డువరుస_కొలత;

            ఒకవేళ (src1->type != vec_dot_type && src0->type != GGML_TYPE_Q8_0) {
                from_float((స్థిరరాశి దశాంశం *) src_అడ్డువరుస, dst_అడ్డువరుస, ne10);
            } లేకపోతే {
                // no conversion: src1 already matches vec_dot_type, or src0 is Q8_0, whose
                // ZenDNN dynamic quantization requires the row to stay in F32
                std::జ్ఞాపకం_నకలు(dst_అడ్డువరుస, src_అడ్డువరుస, gather_అడ్డువరుస_కొలత);
            }
        }

        expert_అడ్డువరుస_లెక్క[batch_idx] = cne1;
        batch_src[batch_idx]        = wdata_cur + w_off;
        batch_wei[batch_idx]        = (స్థిరరాశి అక్షరం *) src0->దత్తాంశం + cur_a * nb02;
        batch_dst[batch_idx]        = dst_cur + d_off;
    }

    ఒకవేళ (!ggml_zendnn_సమూహం_gemm(ctx,
                                ne01,             // m
                                ne10,             // k
                                expert_అడ్డువరుస_లెక్క, // n (per expert)
                                batch_wei, ne00,  // A: weights, lda
                                batch_src, ne10,  // B: input, ldb
                                batch_dst, ne01,  // C: output, ldc
                                src0->type,
                                src0->type == GGML_TYPE_Q8_0 ? GGML_TYPE_F32 : vec_dot_type,
                                dst->type))
        GGML_ABORT("%s: ZenDNN group gemm failed\n", __func__);

    // scatter output rows to destination
    #సూచన omp parallel కొరకు అంకె_దారాలు(ctx->n_దారాలు) కాలపట్టిక(స్థిర)
    కొరకు (int64_t cur_a = 0; cur_a < n_as; ++cur_a) {
        స్థిరరాశి int64_t cne1 = మాతృక_అడ్డువరుస_counts[cur_a];

        ఒకవేళ (cne1 == 0) {
            కొనసాగు;
        }

        స్థిరరాశి int64_t d_off = expert_dst_off[cur_a];

        కొరకు (int64_t ir1 = 0; ir1 < cne1; ++ir1) {
            స్థిరరాశి mmid_అడ్డువరుస_mapping & అడ్డువరుస_mapping = మాతృక_అడ్డువరుసలు[cur_a][ir1];
            స్థిరరాశి int64_t id = అడ్డువరుస_mapping.i1;
            స్థిరరాశి int64_t i1 = id;
            స్థిరరాశి int64_t i2 = అడ్డువరుస_mapping.i2;

            std::జ్ఞాపకం_నకలు(
                (అక్షరం *) dst->దత్తాంశం + i1*nb1 + i2*nb2,
                dst_cur + d_off + ir1 * ggml_అడ్డువరుస_కొలత(dst->type, ne01),
                ggml_అడ్డువరుస_కొలత(dst->type, ne01)
            );
        }
    }
}

// backend interface

స్థిర స్థిరరాశి అక్షరం * ggml_backend_zendnn_get_name(ggml_backend_t backend) {
    తిరిగి "ZenDNN";

    GGML_వాడని(backend);
}

స్థిర శూన్య ggml_backend_zendnn_free(ggml_backend_t backend) {
    ggml_backend_zendnn_context * ctx = (ggml_backend_zendnn_context *)backend->context;
    తొలగించు ctx;
    తొలగించు backend;
}

స్థిర ggml_స్థితి ggml_backend_zendnn_లేఖాచిత్రం_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_zendnn_context * ctx = (ggml_backend_zendnn_context *)backend->context;

    కొరకు (పూర్ణాంకం i = 0; i < cgraph->n_nodes; i++) {
        నిర్మాణం ggml_tensor * కణుపు = cgraph->nodes[i];

        ఒకవేళ ((కణుపు->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            కొనసాగు;
        }

        ఎంపిక (కణుపు->op) {
            స్థితి GGML_OP_MUL_MAT:
                ggml_zendnn_compute_forward_mul_mat(ctx, కణుపు);
                ఆపు;
            స్థితి GGML_OP_MUL_MAT_ID:
                ggml_zendnn_compute_forward_mul_mat_id(ctx, కణుపు);
                ఆపు;
            స్థితి GGML_OP_NONE:
            స్థితి GGML_OP_RESHAPE:
            స్థితి GGML_OP_VIEW:
            స్థితి GGML_OP_క్రమమార్పు:
            స్థితి GGML_OP_TRANSPOSE:
                ఆపు;

            అప్రమేయ:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(కణుపు));
        }
    }

    తిరిగి GGML_స్థితి_SUCCESS;

    GGML_వాడని(backend);
}

స్థిర నిర్మాణం ggml_backend_i ggml_backend_zendnn_i = {
    /* .get_name                = */ ggml_backend_zendnn_get_name,
    /* .free                    = */ ggml_backend_zendnn_free,
    /* .set_tensor_async        = */ ఖాళీ,
    /* .get_tensor_async        = */ ఖాళీ,
    /* .set_tensor_2d_async     = */ ఖాళీ,
    /* .get_tensor_2d_async     = */ ఖాళీ,
    /* .cpy_tensor_async        = */ ఖాళీ,
    /* .synchronize             = */ ఖాళీ,
    /* .graph_plan_create       = */ ఖాళీ,
    /* .graph_plan_free         = */ ఖాళీ,
    /* .graph_plan_update       = */ ఖాళీ,
    /* .graph_plan_compute      = */ ఖాళీ,
    /* .graph_compute           = */ ggml_backend_zendnn_లేఖాచిత్రం_compute,
    /* .event_record            = */ ఖాళీ,
    /* .event_wait              = */ ఖాళీ,
    /* .graph_optimize          = */ ఖాళీ,
};

స్థిర ggml_guid_t ggml_backend_zendnn_guid(శూన్య) {
    స్థిర స్థిరరాశి అక్షరం * guid_str = "AMD-ZENDNN-ACCEL";
    తిరిగి reinterpret_cast<ggml_guid_t>(const_cast<అక్షరం*>(guid_str));
}

ggml_backend_t ggml_backend_zendnn_init(శూన్య) {
    ggml_backend_zendnn_context * ctx = కొత్త ggml_backend_zendnn_context;

    ggml_backend_t backend = కొత్త ggml_backend {
        /* .guid    = */ ggml_backend_zendnn_guid(),
        /* .iface   = */ ggml_backend_zendnn_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_zendnn_reg(), 0),
        /* .context = */ ctx,
    };

    తిరిగి backend;
}

తార్కిక ggml_backend_is_zendnn(ggml_backend_t backend) {
    తిరిగి backend != ఖాళీ && ggml_guid_matches(backend->guid, ggml_backend_zendnn_guid());
}

శూన్య ggml_backend_zendnn_సమితి_n_దారాలు(ggml_backend_t backend_zendnn, పూర్ణాంకం n_దారాలు) {
    GGML_ASSERT(ggml_backend_is_zendnn(backend_zendnn));

    ggml_backend_zendnn_context * ctx = (ggml_backend_zendnn_context *)backend_zendnn->context;
    ctx->n_దారాలు = n_దారాలు;
}

// device interface
స్థిర స్థిరరాశి అక్షరం * ggml_backend_zendnn_device_get_name(ggml_backend_dev_t dev) {
    తిరిగి "ZenDNN";

    GGML_వాడని(dev);
}
/**
 * ZenDNN is AMD's performance library providing optimized primitives and implementations
 * for deep learning workloads on AMD CPUs. It targets improved performance for common
 * neural network operations on AMD architectures. For more information, see:
 * https://www.amd.com/en/developer/zendnn.html
 */
స్థిర స్థిరరాశి అక్షరం * ggml_backend_zendnn_device_get_description(ggml_backend_dev_t dev) {
    తిరిగి "ZenDNN: AMD optimized primitives backend for GGML (optimized for AMD CPUs)";

    GGML_వాడని(dev);
}

స్థిర శూన్య ggml_backend_zendnn_device_get_జ్ఞాపకం(ggml_backend_dev_t dev, size_t * విడుదల, size_t * మొత్తం) {
    *విడుదల  = 0;
    *మొత్తం = 0;

    GGML_వాడని(dev);
}

స్థిర గణన ggml_backend_dev_type ggml_backend_zendnn_device_get_type(ggml_backend_dev_t dev) {
    తిరిగి GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_వాడని(dev);
}

స్థిర శూన్య ggml_backend_zendnn_device_get_props(ggml_backend_dev_t dev, నిర్మాణం ggml_backend_dev_props * props) {
    props->name        = ggml_backend_zendnn_device_get_name(dev);
    props->description = ggml_backend_zendnn_device_get_description(dev);
    props->type        = ggml_backend_zendnn_device_get_type(dev);
    ggml_backend_zendnn_device_get_జ్ఞాపకం(dev, &props->జ్ఞాపకం_free, &props->జ్ఞాపకం_మొత్తం);
    props->caps = {
        /* .async                = */ అబద్ధం,
        /* .host_buffer          = */ అబద్ధం,
        /* .buffer_from_host_ptr = */ నిజం,
        /* .events               = */ అబద్ధం,
        /* .mmap_support         = */ నిజం,
    };
}

స్థిర ggml_backend_t ggml_backend_zendnn_device_init_backend(ggml_backend_dev_t dev, స్థిరరాశి అక్షరం * params) {
    ggml_backend_t backend = ggml_backend_zendnn_init();
    ఒకవేళ (backend == ఖాళీ) {
        GGML_LOG_ERROR("%s: error: failed to initialize ZenDNN backend\n", __func__);
        తిరిగి ఖాళీ;
    }

    తిరిగి backend;

    GGML_వాడని(dev);
    GGML_వాడని(params);
}

స్థిర ggml_backend_బఫరు_type_t ggml_backend_zendnn_device_get_బఫరు_type(ggml_backend_dev_t dev) {
    తిరిగి ggml_backend_cpu_బఫరు_type();

    GGML_వాడని(dev);
}

స్థిర ggml_backend_బఫరు_t ggml_backend_zendnn_device_బఫరు_from_host_ptr(ggml_backend_dev_t dev, శూన్య * ptr, size_t size, size_t max_tensor_కొలత) {
    తిరిగి ggml_backend_cpu_బఫరు_from_ptr(ptr, size);

    GGML_వాడని(dev);
    GGML_వాడని(max_tensor_కొలత);
}

స్థిర తార్కిక ggml_zendnn_adaptive_fallback_enabled() {
    స్థిర స్థిరరాశి తార్కిక enabled = std::getenv("GGML_ZENDNN_ADAPTIVE_FALLBACK") == శూన్యసూచిక ||
                                std::atoi(std::getenv("GGML_ZENDNN_ADAPTIVE_FALLBACK")) != 0;
    తిరిగి enabled;
}

స్థిర తార్కిక ggml_backend_zendnn_device_supports_op(ggml_backend_dev_t dev, స్థిరరాశి నిర్మాణం ggml_tensor * op) {
    ఎంపిక (op->op) {
        స్థితి GGML_OP_NONE:
        స్థితి GGML_OP_RESHAPE:
        స్థితి GGML_OP_VIEW:
        స్థితి GGML_OP_క్రమమార్పు:
        స్థితి GGML_OP_TRANSPOSE:
            తిరిగి నిజం;

        స్థితి GGML_OP_MUL_MAT:
        స్థితి GGML_OP_MUL_MAT_ID:
        {
            స్థిరరాశి ggml_tensor * weights = op->src[0];
            స్థిరరాశి ggml_tensor * inputs = op->src[1];

            స్థిరరాశి int64_t ne10 = inputs->ne[0];
            స్థిరరాశి int64_t ne0 = op->ne[0];
            స్థిరరాశి int64_t ne1 = op->ne[1];
            స్థిరరాశి int64_t min_batch = 1;

            ఒకవేళ(!ggml_is_contiguous(weights) || !ggml_is_contiguous(inputs)) {
                తిరిగి అబద్ధం;
            }

            ఒకవేళ (ggml_zendnn_adaptive_fallback_enabled()) {
                స్థిరరాశి int64_t K = inputs->ne[0];
                స్థిరరాశి int64_t N = (inputs->ne[1]*inputs->ne[2]*inputs->ne[3]);
                స్థిరరాశి int64_t M = weights->ne[1];
                ఒకవేళ(K <= 256 || N <= 128 || M <= 96) {
                    తిరిగి అబద్ధం;
                }

                // MUL_MAT_ID's gather+matmul+scatter approach favors a moderate expert count
                ఒకవేళ (op->op == GGML_OP_MUL_MAT_ID) {
                    స్థిరరాశి int64_t n_experts = weights->ne[2];
                    స్థిరరాశి int64_t max_experts = 32;
                    ఒకవేళ (n_experts > max_experts) {
                        తిరిగి అబద్ధం;
                    }

                    // fall back once the average rows per expert (N / n_experts) is too thin
                    // to amortize each per-expert GEMM's overhead
                    ఒకవేళ (N / n_experts <= 32) {
                        తిరిగి అబద్ధం;
                    }
                }
            }
            లేకపోతే ఒకవేళ (ne0 < min_batch || ne1 < min_batch || ne10 < min_batch) {
                తిరిగి అబద్ధం;
            }

            ఎంపిక (weights->type) {
                స్థితి GGML_TYPE_F32:
                స్థితి GGML_TYPE_BF16:
                స్థితి GGML_TYPE_Q8_0:
                    తిరిగి నిజం;
                అప్రమేయ:
                    తిరిగి అబద్ధం;
            }
        } ఆపు;

        అప్రమేయ:
            తిరిగి అబద్ధం;
    }

    GGML_వాడని(dev);
}

స్థిర తార్కిక ggml_backend_zendnn_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_బఫరు_type_t buft) {
    తిరిగి ggml_backend_buft_is_host(buft);

    GGML_వాడని(dev);
}

స్థిర స్థిరరాశి నిర్మాణం ggml_backend_device_i ggml_backend_zendnn_device_i = {
    /* .get_name               = */ ggml_backend_zendnn_device_get_name,
    /* .get_description        = */ ggml_backend_zendnn_device_get_description,
    /* .get_memory             = */ ggml_backend_zendnn_device_get_జ్ఞాపకం,
    /* .get_type               = */ ggml_backend_zendnn_device_get_type,
    /* .get_props              = */ ggml_backend_zendnn_device_get_props,
    /* .init_backend           = */ ggml_backend_zendnn_device_init_backend,
    /* .get_buffer_type        = */ ggml_backend_zendnn_device_get_బఫరు_type,
    /* .get_host_buffer_type   = */ ఖాళీ,
    /* .buffer_from_host_ptr   = */ ggml_backend_zendnn_device_బఫరు_from_host_ptr,
    /* .supports_op            = */ ggml_backend_zendnn_device_supports_op,
    /* .supports_buft          = */ ggml_backend_zendnn_device_supports_buft,
    /* .offload_op             = */ ఖాళీ,
    /* .event_new              = */ ఖాళీ,
    /* .event_free             = */ ఖాళీ,
    /* .event_synchronize      = */ ఖాళీ,
};

// backend reg interface
స్థిర స్థిరరాశి అక్షరం * ggml_backend_zendnn_reg_get_name(ggml_backend_reg_t reg) {
    తిరిగి "ZenDNN";

    GGML_వాడని(reg);
}

స్థిర size_t ggml_backend_zendnn_reg_get_device_లెక్క(ggml_backend_reg_t reg) {
    తిరిగి 1;

    GGML_వాడని(reg);
}

స్థిర ggml_backend_dev_t ggml_backend_zendnn_reg_get_device(ggml_backend_reg_t reg, size_t సూచీ) {
    GGML_ASSERT(సూచీ == 0);

    స్థిర ggml_backend_device ggml_backend_zendnn_device = {
        /* .iface   = */ ggml_backend_zendnn_device_i,
        /* .reg     = */ reg,
        /* .context = */ శూన్యసూచిక,
    };

    తిరిగి &ggml_backend_zendnn_device;
}

స్థిర శూన్య * ggml_backend_zendnn_get_proc_చిరునామా(ggml_backend_reg_t reg, స్థిరరాశి అక్షరం * name) {
    ఒకవేళ (std::strcmp(name, "ggml_backend_set_n_threads") == 0) {
        తిరిగి (శూన్య *) ggml_backend_zendnn_సమితి_n_దారాలు;
    }
    తిరిగి ఖాళీ;

    GGML_వాడని(reg);
    GGML_వాడని(name);
}

స్థిర స్థిరరాశి నిర్మాణం ggml_backend_reg_i ggml_backend_zendnn_reg_i = {
    /* .get_name         = */ ggml_backend_zendnn_reg_get_name,
    /* .get_device_count = */ ggml_backend_zendnn_reg_get_device_లెక్క,
    /* .get_device       = */ ggml_backend_zendnn_reg_get_device,
    /* .get_proc_address = */ ggml_backend_zendnn_get_proc_చిరునామా,
};

ggml_backend_reg_t ggml_backend_zendnn_reg(శూన్య) {
    స్థిర నిర్మాణం ggml_backend_reg ggml_backend_zendnn_reg = {
        /* .api_version = */ GGML_BACKEND_API_సంస్కరణ,
        /* .iface       = */ ggml_backend_zendnn_reg_i,
        /* .context     = */ ఖాళీ,
    };

    తిరిగి &ggml_backend_zendnn_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_zendnn_reg)
