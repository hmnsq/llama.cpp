#அந்தர்பாவ "ggml-zendnn.h"

#அந்தர்பாவ "ggml-backend-impl.h"
#அந்தர்பாவ "ggml-impl.h"

#define GGML_COMMON_DECL_CPP
#அந்தர்பாவ "ggml-common.h"

#அந்தர்பாவ "zendnnl.hpp"

#அந்தர்பாவ <cstring>


கட்டமைப்பு ggml_backend_zendnn_context {
    முழு n_threads = GGML_DEFAULT_N_THREADS;
    std::unique_ptr<எழுத்து[]> work_data;
    size_t work_size = 0;
};

template<typename T>
zendnnl::common::data_type_t ggml_to_zendnn_type() {
    எனில் constexpr (std::is_same_v<T, தசம>) {
        திரும்பு zendnnl::common::data_type_t::f32;
    } இல்லையெனில் எனில் constexpr (std::is_same_v<T, ggml_bf16_t>) {
        திரும்பு zendnnl::common::data_type_t::bf16;
    } இல்லையெனில் எனில் constexpr (std::is_same_v<T, block_q8_0>) {
        திரும்பு zendnnl::common::data_type_t::s8;
    } இல்லையெனில் {
        திரும்பு zendnnl::common::data_type_t::none;
    }
}

/**
 * Builds the matmul_params shared by ggml_zendnn_matmul() and ggml_zendnn_group_matmul():
 * dtype selection plus, for Q8_0 weights, dynamic-quant setup. Callers still need to set
 * quant_params.src_scale.dims themselves, since that depends on the batch size(s) in use.
 */
template <typename TA, typename TB, typename TC>
நிலையான zendnnl::lowoha::matmul::matmul_params ggml_zendnn_make_matmul_params(ggml_backend_zendnn_context * ctx) {
    zendnnl::lowoha::matmul::matmul_params params;
    params.dtypes.src = ggml_to_zendnn_type<TB>();
    params.dtypes.wei = ggml_to_zendnn_type<TA>();
    params.dtypes.dst = ggml_to_zendnn_type<TC>();
    params.இலக்கம்_threads = ctx->n_threads;

    எனில் constexpr (std::is_same_v<TA, block_q8_0>) {
        params.dtypes.compute = zendnnl::common::data_type_t::s8;
        params.dynamic_quant = உண்மை;
        params.quant_params.src_scale.buff = nullptr;
        params.quant_params.src_scale.dt   = zendnnl::common::data_type_t::bf16;
        params.packing.pack_format_b = 1;
    }
    திரும்பு params;
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
நிலையான தருக்க ggml_zendnn_matmul(ggml_backend_zendnn_context * ctx, int64_t m, int64_t n, int64_t k,
                               மாறிலி TA * A, int64_t lda, மாறிலி TB * B, int64_t ldb, TC * C,
                               int64_t ldc) {

    zendnnl::lowoha::matmul::matmul_params params = ggml_zendnn_make_matmul_params<TA, TB, TC>(ctx);

    zendnnl::lowoha::matmul::matmul_batch_params_t batch_params;

    எனில் constexpr (std::is_same_v<TA, block_q8_0>) {
        params.quant_params.src_scale.dims = {n, k / QK8_0};
    }

    zendnnl::error_handling::status_t status = zendnnl::lowoha::matmul::matmul_direct(
        'r', பொய், உண்மை,   // row-major, don't transpose B, transpose A (because it's column-major)
        n,                  // M: rows of B and C
        m,                  // N: cols of A^T and C
        k,                  // K: cols of B, rows of A
        1.0f,               // alpha
        B, ldb,             // src: B[n,k]
        A, lda,             // weight: A[k,m] column-major (transposed)
        nullptr,            // bias
        0.0f,               // beta
        C, ldc,             // output C[n,m]
        உண்மை,               // is_weights_const
        batch_params,       // batch_params
        params              // params
    );

    எனில் (status != zendnnl::error_handling::status_t::success) {
        GGML_LOG_ERROR("%s, ZenDNN matmul failed: status=%d\n", __func__, static_cast<முழு>(status));
        திரும்பு பொய்;
    }
    திரும்பு உண்மை;
}

நிலையான தருக்க ggml_zendnn_gemm(ggml_backend_zendnn_context * ctx, int64_t m, int64_t n, int64_t k,
                              மாறிலி வெறுமை * A, int64_t lda, மாறிலி வெறுமை * B, int64_t ldb, வெறுமை * C,
                              int64_t ldc, முழு Atype, முழு Btype, முழு Ctype) {

    ப்ரதிஜ்ஞா(m >= 0);
    ப்ரதிஜ்ஞா(n >= 0);
    ப்ரதிஜ்ஞா(k >= 0);
    ப்ரதிஜ்ஞா(lda >= k);
    ப்ரதிஜ்ஞா(ldb >= k);
    ப்ரதிஜ்ஞா(ldc >= m);

    // categorize types
    தேர்வு (Atype) {
        நிலை GGML_TYPE_F32:
            எனில் (Btype != GGML_TYPE_F32 || Ctype != GGML_TYPE_F32)
                திரும்பு பொய்;
            திரும்பு ggml_zendnn_matmul<தசம, தசம, தசம>(
                ctx, m, n, k,
                (மாறிலி தசம *)A, lda,
                (மாறிலி தசம *)B, ldb,
                (தசம *)C, ldc);
        நிலை GGML_TYPE_BF16:
            எனில் (Btype != GGML_TYPE_BF16)
                திரும்பு பொய்;
            எனில் (Ctype == GGML_TYPE_BF16)
                திரும்பு ggml_zendnn_matmul<ggml_bf16_t, ggml_bf16_t, ggml_bf16_t>(
                    ctx, m, n, k,
                    (மாறிலி ggml_bf16_t *)A, lda,
                    (மாறிலி ggml_bf16_t *)B, ldb,
                    (ggml_bf16_t *)C, ldc);
            எனில் (Ctype == GGML_TYPE_F32)
                திரும்பு ggml_zendnn_matmul<ggml_bf16_t, ggml_bf16_t, தசம>(
                    ctx, m, n, k,
                    (மாறிலி ggml_bf16_t *)A, lda,
                    (மாறிலி ggml_bf16_t *)B, ldb,
                    (தசம *)C, ldc);
            திரும்பு பொய்;
        நிலை GGML_TYPE_Q8_0:
            எனில் (Btype != GGML_TYPE_F32 || Ctype != GGML_TYPE_F32)
                திரும்பு பொய்;
            திரும்பு ggml_zendnn_matmul<block_q8_0, தசம, தசம>(
                ctx, m, n, k,
                (மாறிலி block_q8_0 *)A, lda,
                (மாறிலி தசம *)B, ldb,
                (தசம *)C, ldc);
        இயல்பு:
            திரும்பு பொய்; // unsupported type
    }
}

நிலையான வெறுமை ggml_zendnn_compute_forward_mul_mat(
    ggml_backend_zendnn_context * ctx,
    ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];  // weights
    மாறிலி ggml_tensor * src1 = dst->src[1];  // inputs

    GGML_TENSOR_இருமம்_OP_LOCALS

    ggml_type         மாறிலி vec_dot_type = src0->type;
    ggml_from_float_t மாறிலி from_float = ggml_get_type_traits(vec_dot_type)->from_float_ref;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == அளவு(தசம));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // broadcast factors
    மாறிலி int64_t r2 = ne12/ne02;
    மாறிலி int64_t r3 = ne13/ne03;

    வெறுமை * work_data = ctx->work_data.get();

    // ZenDNN requires FP32 for dynamic quantization, so conversion is skipped
    எனில் (src1->type != vec_dot_type && src0->type != GGML_TYPE_Q8_0) {
        மாறிலி size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
        மாறிலி size_t nbw2 = nbw1 * ne11;
        மாறிலி size_t nbw3 = nbw2 * ne12;
        மாறிலி size_t desired_wsize = ne13 * nbw3;
        எனில் (ctx->work_size < desired_wsize) {
            ctx->work_data.reset(புதிய எழுத்து[desired_wsize]);
            ctx->work_size = desired_wsize;
        }
        work_data = ctx->work_data.get();

        // #pragma omp parallel for num_threads(ctx->n_threads)
        #pragma omp parallel க்கு collapse(3) இலக்கம்_threads(ctx->n_threads) schedule(நிலையான)
        க்கு (int64_t i13 = 0; i13 < ne13; ++i13) {
            க்கு (int64_t i12 = 0; i12 < ne12; ++i12) {
                க்கு (int64_t i11 = 0; i11 < ne11; ++i11) {
                    மாறிலி தசம * src1_f32 = (தசம *)((எழுத்து *)src1->data + i11*nb11 + i12*nb12 + i13*nb13);
                    வெறுமை * src1_conv = (எழுத்து *)work_data + i11*nbw1 + i12*nbw2 + i13*nbw3;
                    from_float(src1_f32, src1_conv, ne10);
                }
            }
        }
    }

    க்கு (int64_t i13 = 0; i13 < ne13; i13++) {
        க்கு (int64_t i12 = 0; i12 < ne12; i12++) {
            மாறிலி வெறுமை* wdata = (src1->type == vec_dot_type || src0->type == GGML_TYPE_Q8_0) ? src1->data : work_data;
            மாறிலி size_t row_size = ggml_row_size(vec_dot_type, ne10);
            எனில் (!ggml_zendnn_gemm(ctx,
                                  ne01,     // m
                                  ne11,     // n
                                  ne10,     // k
                                  static_cast<மாறிலி எழுத்து *>(src0->data) + (i12/r2)*nb02 + (i13/r3)*nb03,
                                  ne00,     // lda
                                  static_cast<மாறிலி எழுத்து *>(wdata) + (i12*ne11 + i13*ne12*ne11)*row_size,
                                  ne10,     // ldb
                                  static_cast<எழுத்து *>(dst->data) + i12*nb2 + i13*nb3,
                                  ne01,     // ldc
                                  src0->type,
                                  src0->type == GGML_TYPE_Q8_0 ? GGML_TYPE_F32 : vec_dot_type,
                                  dst->type))
                GGML_ABORT("%s: ZenDNN gemm failed\n", __func__);
        }
    }
}

கட்டமைப்பு mmid_row_mapping {
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
நிலையான தருக்க ggml_zendnn_group_matmul(ggml_backend_zendnn_context * ctx, int64_t m, int64_t k,
                                     மாறிலி std::vector<int64_t> & n,
                                     மாறிலி std::vector<மாறிலி வெறுமை *> & A, int64_t lda,
                                     மாறிலி std::vector<மாறிலி வெறுமை *> & B, int64_t ldb,
                                     மாறிலி std::vector<வெறுமை *> & C, int64_t ldc) {

    மாறிலி முழு n_experts = n.size();

    zendnnl::lowoha::matmul::matmul_params base_params = ggml_zendnn_make_matmul_params<TA, TB, TC>(ctx);

    std::vector<எழுத்து>         layout(n_experts, 'r');
    std::vector<தருக்க>         trans_a(n_experts, பொய்);
    std::vector<தருக்க>         trans_b(n_experts, உண்மை);
    std::vector<முழு>          batch_m(n_experts);
    std::vector<முழு>          batch_n(n_experts, m);
    std::vector<முழு>          batch_k(n_experts, k);
    std::vector<தசம>        alpha(n_experts, 1.0f);
    std::vector<தசம>        beta(n_experts, 0.0f);
    std::vector<மாறிலி வெறுமை *> bias(n_experts, nullptr);
    std::vector<முழு>          lda_v(n_experts, lda);
    std::vector<முழு>          ldb_v(n_experts, ldb);
    std::vector<முழு>          ldc_v(n_experts, ldc);
    std::vector<தருக்க>         is_wei_const(n_experts, உண்மை);
    std::vector<zendnnl::lowoha::matmul::matmul_params> params(n_experts, base_params);

    க்கு (முழு i = 0; i < n_experts; i++) {
        batch_m[i] = n[i];

        // src_scale.dims depends on this expert's row count, unlike the rest of base_params
        எனில் constexpr (std::is_same_v<TA, block_q8_0>) {
            params[i].quant_params.src_scale.dims = {n[i], k / QK8_0};
        }
    }

    zendnnl::error_handling::status_t status = zendnnl::lowoha::matmul::group_matmul_direct(
        layout, trans_a, trans_b, batch_m, batch_n, batch_k, alpha,
        B, ldb_v, A, lda_v, bias, beta,
        C, ldc_v, is_wei_const, params);

    எனில் (status != zendnnl::error_handling::status_t::success) {
        GGML_LOG_ERROR("%s, ZenDNN group matmul failed: status=%d\n", __func__, static_cast<முழு>(status));
        திரும்பு பொய்;
    }
    திரும்பு உண்மை;
}

நிலையான தருக்க ggml_zendnn_group_gemm(ggml_backend_zendnn_context * ctx, int64_t m, int64_t k,
                                    மாறிலி std::vector<int64_t> & n,
                                    மாறிலி std::vector<மாறிலி வெறுமை *> & A, int64_t lda,
                                    மாறிலி std::vector<மாறிலி வெறுமை *> & B, int64_t ldb,
                                    மாறிலி std::vector<வெறுமை *> & C, int64_t ldc,
                                    முழு Atype, முழு Btype, முழு Ctype) {

    ப்ரதிஜ்ஞா(m >= 0);
    க்கு (size_t i = 0; i < n.size(); i++) {
        ப்ரதிஜ்ஞா(n[i] >= 0);
    }
    ப்ரதிஜ்ஞா(k >= 0);
    ப்ரதிஜ்ஞா(lda >= k);
    ப்ரதிஜ்ஞா(ldb >= k);
    ப்ரதிஜ்ஞா(ldc >= m);

    // categorize types
    தேர்வு (Atype) {
        நிலை GGML_TYPE_F32:
            எனில் (Btype != GGML_TYPE_F32 || Ctype != GGML_TYPE_F32)
                திரும்பு பொய்;
            திரும்பு ggml_zendnn_group_matmul<தசம, தசம, தசம>(ctx, m, k, n, A, lda, B, ldb, C, ldc);
        நிலை GGML_TYPE_BF16:
            எனில் (Btype != GGML_TYPE_BF16)
                திரும்பு பொய்;
            எனில் (Ctype == GGML_TYPE_BF16)
                திரும்பு ggml_zendnn_group_matmul<ggml_bf16_t, ggml_bf16_t, ggml_bf16_t>(
                    ctx, m, k, n, A, lda, B, ldb, C, ldc);
            எனில் (Ctype == GGML_TYPE_F32)
                திரும்பு ggml_zendnn_group_matmul<ggml_bf16_t, ggml_bf16_t, தசம>(ctx, m, k, n, A, lda, B, ldb, C, ldc);
            திரும்பு பொய்;
        நிலை GGML_TYPE_Q8_0:
            எனில் (Btype != GGML_TYPE_F32 || Ctype != GGML_TYPE_F32)
                திரும்பு பொய்;
            திரும்பு ggml_zendnn_group_matmul<block_q8_0, தசம, தசம>(ctx, m, k, n, A, lda, B, ldb, C, ldc);
        இயல்பு:
            திரும்பு பொய்; // unsupported type
    }
}

நிலையான வெறுமை ggml_zendnn_compute_forward_mul_mat_id(
    ggml_backend_zendnn_context * ctx,
    ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];  // expert weights
    மாறிலி ggml_tensor * src1 = dst->src[1];  // inputs
    மாறிலி ggml_tensor * ids  = dst->src[2];  // expert ids

    GGML_TENSOR_இருமம்_OP_LOCALS

    // exit for no tokens to process
    எனில் (ne2 == 0 || ne11 == 0) {
        திரும்பு;
    }

    ggml_type         மாறிலி vec_dot_type = src0->type;
    ggml_from_float_t மாறிலி from_float = ggml_get_type_traits(vec_dot_type)->from_float_ref;

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == அளவு(தசம));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(ne03 == 1);
    GGML_ASSERT(ne13 == 1);
    GGML_ASSERT(ne3  == 1);

    // row groups
    மாறிலி முழு n_ids = ids->ne[0]; // n_expert_used
    மாறிலி முழு n_as  = ne02;       // n_experts

    std::vector<int64_t> matrix_row_counts(n_as, 0);
    std::vector<std::vector<mmid_row_mapping>> matrix_rows(n_as);

    int64_t total_rows = 0;
    முழு     n_active_experts = 0;
    // group rows by expert (preprocessing step)
    க்கு (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
        க்கு (முழு id = 0; id < n_ids; ++id) {
            மாறிலி int32_t i02 = *(மாறிலி int32_t *)((மாறிலி எழுத்து *)ids->data + iid1*ids->nb[1] + id*ids->nb[0]);

            GGML_ASSERT(i02 >= 0 && i02 < n_as);

            எனில் (matrix_row_counts[i02] == 0) {
                n_active_experts++;
            }
            matrix_rows[i02].push_back({id, iid1});
            matrix_row_counts[i02]++;
            total_rows++;
        }
    }

    எனில் (total_rows == 0) {
        திரும்பு; // no rows to process
    }

    மாறிலி size_t row_size = ggml_row_size(vec_dot_type, ne10);

    // For Q8_0, src1 is always F32; the gather buffer must hold F32 rows (ne10*4 bytes),
    // not Q8_0-encoded rows (row_size ≈ ne10/32*34 bytes) — they differ by ~4x.
    மாறிலி size_t f32_row_size = (size_t)ne10 * அளவு(தசம);
    மாறிலி size_t gather_row_size = (src0->type == GGML_TYPE_Q8_0) ? f32_row_size : row_size;

    எனில் (src1->type != vec_dot_type && src0->type != GGML_TYPE_Q8_0) {
        GGML_ASSERT(src1->type == GGML_TYPE_F32);
    }

    // size for MoE gather/scatter buffers
    மாறிலி size_t wdata_cur_size = total_rows * gather_row_size;
    மாறிலி size_t dst_cur_size = total_rows * ggml_row_size(dst->type, ne01);

    // allocate single buffer for all needs
    மாறிலி size_t total_size = wdata_cur_size + dst_cur_size;
    எனில் (ctx->work_size < total_size) {
        ctx->work_data.reset(புதிய எழுத்து[total_size]);
        ctx->work_size = total_size;
    }

    // partition the buffer
    எழுத்து * wdata_cur = ctx->work_data.get();
    எழுத்து * dst_cur = wdata_cur + wdata_cur_size;

    // per-expert data collected during gather, handed to ggml_zendnn_group_gemm() as one batch
    std::vector<int64_t>      expert_row_மொத்தம்(n_active_experts);
    std::vector<மாறிலி வெறுமை *> batch_src(n_active_experts);
    std::vector<மாறிலி வெறுமை *> batch_wei(n_active_experts);
    std::vector<வெறுமை *>       batch_dst(n_active_experts);

    // precompute per-expert buffer offsets and batch indices for the parallel loop below
    std::vector<int64_t> expert_wdata_off(n_as, 0);
    std::vector<int64_t> expert_dst_off(n_as, 0);
    std::vector<முழு>     expert_batch_idx(n_as, -1);
    {
        int64_t w_off = 0;
        int64_t d_off = 0;
        முழு     batch_idx = 0;
        க்கு (int64_t cur_a = 0; cur_a < n_as; ++cur_a) {
            எனில் (matrix_row_counts[cur_a] == 0) {
                தொடர்;
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
    #pragma omp parallel க்கு இலக்கம்_threads(ctx->n_threads) schedule(நிலையான)
    க்கு (int64_t cur_a = 0; cur_a < n_as; ++cur_a) {
        மாறிலி int64_t cne1 = matrix_row_counts[cur_a];

        எனில் (cne1 == 0) {
            தொடர்;
        }

        மாறிலி int64_t w_off     = expert_wdata_off[cur_a];
        மாறிலி int64_t d_off     = expert_dst_off[cur_a];
        மாறிலி முழு     batch_idx = expert_batch_idx[cur_a];

        க்கு (int64_t ir1 = 0; ir1 < cne1; ++ir1) {
            மாறிலி mmid_row_mapping & row_mapping = matrix_rows[cur_a][ir1];
            மாறிலி int64_t id  = row_mapping.i1;
            மாறிலி int64_t i11 = id % ne11;
            மாறிலி int64_t i12 = row_mapping.i2;

            மாறிலி எழுத்து * src_row = (மாறிலி எழுத்து *) src1->data + i11*nb11 + i12*nb12;
            வெறுமை *       dst_row = wdata_cur + w_off + ir1 * gather_row_size;

            எனில் (src1->type != vec_dot_type && src0->type != GGML_TYPE_Q8_0) {
                from_float((மாறிலி தசம *) src_row, dst_row, ne10);
            } இல்லையெனில் {
                // no conversion: src1 already matches vec_dot_type, or src0 is Q8_0, whose
                // ZenDNN dynamic quantization requires the row to stay in F32
                std::memcpy(dst_row, src_row, gather_row_size);
            }
        }

        expert_row_மொத்தம்[batch_idx] = cne1;
        batch_src[batch_idx]        = wdata_cur + w_off;
        batch_wei[batch_idx]        = (மாறிலி எழுத்து *) src0->data + cur_a * nb02;
        batch_dst[batch_idx]        = dst_cur + d_off;
    }

    எனில் (!ggml_zendnn_group_gemm(ctx,
                                ne01,             // m
                                ne10,             // k
                                expert_row_மொத்தம், // n (per expert)
                                batch_wei, ne00,  // A: weights, lda
                                batch_src, ne10,  // B: input, ldb
                                batch_dst, ne01,  // C: output, ldc
                                src0->type,
                                src0->type == GGML_TYPE_Q8_0 ? GGML_TYPE_F32 : vec_dot_type,
                                dst->type))
        GGML_ABORT("%s: ZenDNN group gemm failed\n", __func__);

    // scatter output rows to destination
    #pragma omp parallel க்கு இலக்கம்_threads(ctx->n_threads) schedule(நிலையான)
    க்கு (int64_t cur_a = 0; cur_a < n_as; ++cur_a) {
        மாறிலி int64_t cne1 = matrix_row_counts[cur_a];

        எனில் (cne1 == 0) {
            தொடர்;
        }

        மாறிலி int64_t d_off = expert_dst_off[cur_a];

        க்கு (int64_t ir1 = 0; ir1 < cne1; ++ir1) {
            மாறிலி mmid_row_mapping & row_mapping = matrix_rows[cur_a][ir1];
            மாறிலி int64_t id = row_mapping.i1;
            மாறிலி int64_t i1 = id;
            மாறிலி int64_t i2 = row_mapping.i2;

            std::memcpy(
                (எழுத்து *) dst->data + i1*nb1 + i2*nb2,
                dst_cur + d_off + ir1 * ggml_row_size(dst->type, ne01),
                ggml_row_size(dst->type, ne01)
            );
        }
    }
}

// backend interface

நிலையான மாறிலி எழுத்து * ggml_backend_zendnn_get_name(ggml_backend_t backend) {
    திரும்பு "ZenDNN";

    GGML_UNUSED(backend);
}

நிலையான வெறுமை ggml_backend_zendnn_free(ggml_backend_t backend) {
    ggml_backend_zendnn_context * ctx = (ggml_backend_zendnn_context *)backend->context;
    delete ctx;
    delete backend;
}

நிலையான ggml_status ggml_backend_zendnn_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_zendnn_context * ctx = (ggml_backend_zendnn_context *)backend->context;

    க்கு (முழு i = 0; i < cgraph->n_nodes; i++) {
        கட்டமைப்பு ggml_tensor * முனை = cgraph->nodes[i];

        எனில் ((முனை->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            தொடர்;
        }

        தேர்வு (முனை->op) {
            நிலை GGML_OP_MUL_MAT:
                ggml_zendnn_compute_forward_mul_mat(ctx, முனை);
                நிறுத்து;
            நிலை GGML_OP_MUL_MAT_ID:
                ggml_zendnn_compute_forward_mul_mat_id(ctx, முனை);
                நிறுத்து;
            நிலை GGML_OP_NONE:
            நிலை GGML_OP_RESHAPE:
            நிலை GGML_OP_VIEW:
            நிலை GGML_OP_PERMUTE:
            நிலை GGML_OP_TRANSPOSE:
                நிறுத்து;

            இயல்பு:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(முனை));
        }
    }

    திரும்பு GGML_STATUS_SUCCESS;

    GGML_UNUSED(backend);
}

நிலையான கட்டமைப்பு ggml_backend_i ggml_backend_zendnn_i = {
    /* .get_name                = */ ggml_backend_zendnn_get_name,
    /* .free                    = */ ggml_backend_zendnn_free,
    /* .set_tensor_async        = */ வெற்று,
    /* .get_tensor_async        = */ வெற்று,
    /* .set_tensor_2d_async     = */ வெற்று,
    /* .get_tensor_2d_async     = */ வெற்று,
    /* .cpy_tensor_async        = */ வெற்று,
    /* .synchronize             = */ வெற்று,
    /* .graph_plan_create       = */ வெற்று,
    /* .graph_plan_free         = */ வெற்று,
    /* .graph_plan_update       = */ வெற்று,
    /* .graph_plan_compute      = */ வெற்று,
    /* .graph_compute           = */ ggml_backend_zendnn_graph_compute,
    /* .event_record            = */ வெற்று,
    /* .event_wait              = */ வெற்று,
    /* .graph_optimize          = */ வெற்று,
};

நிலையான ggml_guid_t ggml_backend_zendnn_guid(வெறுமை) {
    நிலையான மாறிலி எழுத்து * guid_str = "AMD-ZENDNN-ACCEL";
    திரும்பு reinterpret_cast<ggml_guid_t>(const_cast<எழுத்து*>(guid_str));
}

ggml_backend_t ggml_backend_zendnn_init(வெறுமை) {
    ggml_backend_zendnn_context * ctx = புதிய ggml_backend_zendnn_context;

    ggml_backend_t backend = புதிய ggml_backend {
        /* .guid    = */ ggml_backend_zendnn_guid(),
        /* .iface   = */ ggml_backend_zendnn_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_zendnn_reg(), 0),
        /* .context = */ ctx,
    };

    திரும்பு backend;
}

தருக்க ggml_backend_is_zendnn(ggml_backend_t backend) {
    திரும்பு backend != வெற்று && ggml_guid_matches(backend->guid, ggml_backend_zendnn_guid());
}

வெறுமை ggml_backend_zendnn_set_n_threads(ggml_backend_t backend_zendnn, முழு n_threads) {
    GGML_ASSERT(ggml_backend_is_zendnn(backend_zendnn));

    ggml_backend_zendnn_context * ctx = (ggml_backend_zendnn_context *)backend_zendnn->context;
    ctx->n_threads = n_threads;
}

// device interface
நிலையான மாறிலி எழுத்து * ggml_backend_zendnn_device_get_name(ggml_backend_dev_t dev) {
    திரும்பு "ZenDNN";

    GGML_UNUSED(dev);
}
/**
 * ZenDNN is AMD's performance library providing optimized primitives and implementations
 * for deep learning workloads on AMD CPUs. It targets improved performance for common
 * neural network operations on AMD architectures. For more information, see:
 * https://www.amd.com/en/developer/zendnn.html
 */
நிலையான மாறிலி எழுத்து * ggml_backend_zendnn_device_get_description(ggml_backend_dev_t dev) {
    திரும்பு "ZenDNN: AMD optimized primitives backend for GGML (optimized for AMD CPUs)";

    GGML_UNUSED(dev);
}

நிலையான வெறுமை ggml_backend_zendnn_device_get_memory(ggml_backend_dev_t dev, size_t * விடுவி, size_t * total) {
    *விடுவி  = 0;
    *total = 0;

    GGML_UNUSED(dev);
}

நிலையான எண்ணிக்கை ggml_backend_dev_type ggml_backend_zendnn_device_get_type(ggml_backend_dev_t dev) {
    திரும்பு GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_UNUSED(dev);
}

நிலையான வெறுமை ggml_backend_zendnn_device_get_props(ggml_backend_dev_t dev, கட்டமைப்பு ggml_backend_dev_props * props) {
    props->name        = ggml_backend_zendnn_device_get_name(dev);
    props->description = ggml_backend_zendnn_device_get_description(dev);
    props->type        = ggml_backend_zendnn_device_get_type(dev);
    ggml_backend_zendnn_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                = */ பொய்,
        /* .host_buffer          = */ பொய்,
        /* .buffer_from_host_ptr = */ உண்மை,
        /* .events               = */ பொய்,
        /* .mmap_support         = */ உண்மை,
    };
}

நிலையான ggml_backend_t ggml_backend_zendnn_device_init_backend(ggml_backend_dev_t dev, மாறிலி எழுத்து * params) {
    ggml_backend_t backend = ggml_backend_zendnn_init();
    எனில் (backend == வெற்று) {
        GGML_LOG_ERROR("%s: error: failed to initialize ZenDNN backend\n", __func__);
        திரும்பு வெற்று;
    }

    திரும்பு backend;

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

நிலையான ggml_backend_buffer_type_t ggml_backend_zendnn_device_get_buffer_type(ggml_backend_dev_t dev) {
    திரும்பு ggml_backend_cpu_buffer_type();

    GGML_UNUSED(dev);
}

நிலையான ggml_backend_buffer_t ggml_backend_zendnn_device_buffer_from_host_ptr(ggml_backend_dev_t dev, வெறுமை * ptr, size_t size, size_t max_tensor_size) {
    திரும்பு ggml_backend_cpu_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

நிலையான தருக்க ggml_zendnn_adaptive_fallback_enabled() {
    நிலையான மாறிலி தருக்க enabled = std::getenv("GGML_ZENDNN_ADAPTIVE_FALLBACK") == nullptr ||
                                std::atoi(std::getenv("GGML_ZENDNN_ADAPTIVE_FALLBACK")) != 0;
    திரும்பு enabled;
}

நிலையான தருக்க ggml_backend_zendnn_device_supports_op(ggml_backend_dev_t dev, மாறிலி கட்டமைப்பு ggml_tensor * op) {
    தேர்வு (op->op) {
        நிலை GGML_OP_NONE:
        நிலை GGML_OP_RESHAPE:
        நிலை GGML_OP_VIEW:
        நிலை GGML_OP_PERMUTE:
        நிலை GGML_OP_TRANSPOSE:
            திரும்பு உண்மை;

        நிலை GGML_OP_MUL_MAT:
        நிலை GGML_OP_MUL_MAT_ID:
        {
            மாறிலி ggml_tensor * weights = op->src[0];
            மாறிலி ggml_tensor * inputs = op->src[1];

            மாறிலி int64_t ne10 = inputs->ne[0];
            மாறிலி int64_t ne0 = op->ne[0];
            மாறிலி int64_t ne1 = op->ne[1];
            மாறிலி int64_t min_batch = 1;

            எனில்(!ggml_is_contiguous(weights) || !ggml_is_contiguous(inputs)) {
                திரும்பு பொய்;
            }

            எனில் (ggml_zendnn_adaptive_fallback_enabled()) {
                மாறிலி int64_t K = inputs->ne[0];
                மாறிலி int64_t N = (inputs->ne[1]*inputs->ne[2]*inputs->ne[3]);
                மாறிலி int64_t M = weights->ne[1];
                எனில்(K <= 256 || N <= 128 || M <= 96) {
                    திரும்பு பொய்;
                }

                // MUL_MAT_ID's gather+matmul+scatter approach favors a moderate expert count
                எனில் (op->op == GGML_OP_MUL_MAT_ID) {
                    மாறிலி int64_t n_experts = weights->ne[2];
                    மாறிலி int64_t max_experts = 32;
                    எனில் (n_experts > max_experts) {
                        திரும்பு பொய்;
                    }

                    // fall back once the average rows per expert (N / n_experts) is too thin
                    // to amortize each per-expert GEMM's overhead
                    எனில் (N / n_experts <= 32) {
                        திரும்பு பொய்;
                    }
                }
            }
            இல்லையெனில் எனில் (ne0 < min_batch || ne1 < min_batch || ne10 < min_batch) {
                திரும்பு பொய்;
            }

            தேர்வு (weights->type) {
                நிலை GGML_TYPE_F32:
                நிலை GGML_TYPE_BF16:
                நிலை GGML_TYPE_Q8_0:
                    திரும்பு உண்மை;
                இயல்பு:
                    திரும்பு பொய்;
            }
        } நிறுத்து;

        இயல்பு:
            திரும்பு பொய்;
    }

    GGML_UNUSED(dev);
}

நிலையான தருக்க ggml_backend_zendnn_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    திரும்பு ggml_backend_buft_is_host(buft);

    GGML_UNUSED(dev);
}

நிலையான மாறிலி கட்டமைப்பு ggml_backend_device_i ggml_backend_zendnn_device_i = {
    /* .get_name               = */ ggml_backend_zendnn_device_get_name,
    /* .get_description        = */ ggml_backend_zendnn_device_get_description,
    /* .get_memory             = */ ggml_backend_zendnn_device_get_memory,
    /* .get_type               = */ ggml_backend_zendnn_device_get_type,
    /* .get_props              = */ ggml_backend_zendnn_device_get_props,
    /* .init_backend           = */ ggml_backend_zendnn_device_init_backend,
    /* .get_buffer_type        = */ ggml_backend_zendnn_device_get_buffer_type,
    /* .get_host_buffer_type   = */ வெற்று,
    /* .buffer_from_host_ptr   = */ ggml_backend_zendnn_device_buffer_from_host_ptr,
    /* .supports_op            = */ ggml_backend_zendnn_device_supports_op,
    /* .supports_buft          = */ ggml_backend_zendnn_device_supports_buft,
    /* .offload_op             = */ வெற்று,
    /* .event_new              = */ வெற்று,
    /* .event_free             = */ வெற்று,
    /* .event_synchronize      = */ வெற்று,
};

// backend reg interface
நிலையான மாறிலி எழுத்து * ggml_backend_zendnn_reg_get_name(ggml_backend_reg_t reg) {
    திரும்பு "ZenDNN";

    GGML_UNUSED(reg);
}

நிலையான size_t ggml_backend_zendnn_reg_get_device_மொத்தம்(ggml_backend_reg_t reg) {
    திரும்பு 1;

    GGML_UNUSED(reg);
}

நிலையான ggml_backend_dev_t ggml_backend_zendnn_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    நிலையான ggml_backend_device ggml_backend_zendnn_device = {
        /* .iface   = */ ggml_backend_zendnn_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };

    திரும்பு &ggml_backend_zendnn_device;
}

நிலையான வெறுமை * ggml_backend_zendnn_get_proc_address(ggml_backend_reg_t reg, மாறிலி எழுத்து * name) {
    எனில் (std::strcmp(name, "ggml_backend_set_n_threads") == 0) {
        திரும்பு (வெறுமை *) ggml_backend_zendnn_set_n_threads;
    }
    திரும்பு வெற்று;

    GGML_UNUSED(reg);
    GGML_UNUSED(name);
}

நிலையான மாறிலி கட்டமைப்பு ggml_backend_reg_i ggml_backend_zendnn_reg_i = {
    /* .get_name         = */ ggml_backend_zendnn_reg_get_name,
    /* .get_device_count = */ ggml_backend_zendnn_reg_get_device_மொத்தம்,
    /* .get_device       = */ ggml_backend_zendnn_reg_get_device,
    /* .get_proc_address = */ ggml_backend_zendnn_get_proc_address,
};

ggml_backend_reg_t ggml_backend_zendnn_reg(வெறுமை) {
    நிலையான கட்டமைப்பு ggml_backend_reg ggml_backend_zendnn_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_zendnn_reg_i,
        /* .context     = */ வெற்று,
    };

    திரும்பு &ggml_backend_zendnn_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_zendnn_reg)
