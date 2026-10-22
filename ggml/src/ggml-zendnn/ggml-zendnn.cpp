#અન્તર્ભાવ "ggml-zendnn.h"

#અન્તર્ભાવ "ggml-backend-impl.h"
#અન્તર્ભાવ "ggml-impl.h"

#define GGML_COMMON_DECL_CPP
#અન્તર્ભાવ "ggml-common.h"

#અન્તર્ભાવ "zendnnl.hpp"

#અન્તર્ભાવ <cstring>


સંરચના ggml_backend_zendnn_context {
    પૂર્ણાંક n_threads = GGML_DEFAULT_N_THREADS;
    std::unique_ptr<અક્ષર[]> work_data;
    size_t work_size = 0;
};

template<typename T>
zendnnl::common::data_type_t ggml_to_zendnn_type() {
    જો constexpr (std::is_same_v<T, દશાંશ>) {
        પાછા zendnnl::common::data_type_t::f32;
    } નહીંતર જો constexpr (std::is_same_v<T, ggml_bf16_t>) {
        પાછા zendnnl::common::data_type_t::bf16;
    } નહીંતર જો constexpr (std::is_same_v<T, block_q8_0>) {
        પાછા zendnnl::common::data_type_t::s8;
    } નહીંતર {
        પાછા zendnnl::common::data_type_t::none;
    }
}

/**
 * Builds the matmul_params shared by ggml_zendnn_matmul() and ggml_zendnn_group_matmul():
 * dtype selection plus, for Q8_0 weights, dynamic-quant setup. Callers still need to set
 * quant_params.src_scale.dims themselves, since that depends on the batch size(s) in use.
 */
template <typename TA, typename TB, typename TC>
સ્થિર zendnnl::lowoha::matmul::matmul_params ggml_zendnn_make_matmul_params(ggml_backend_zendnn_context * ctx) {
    zendnnl::lowoha::matmul::matmul_params params;
    params.dtypes.src = ggml_to_zendnn_type<TB>();
    params.dtypes.wei = ggml_to_zendnn_type<TA>();
    params.dtypes.dst = ggml_to_zendnn_type<TC>();
    params.અંક_threads = ctx->n_threads;

    જો constexpr (std::is_same_v<TA, block_q8_0>) {
        params.dtypes.compute = zendnnl::common::data_type_t::s8;
        params.dynamic_quant = સાચું;
        params.quant_params.src_scale.buff = nullptr;
        params.quant_params.src_scale.dt   = zendnnl::common::data_type_t::bf16;
        params.packing.pack_format_b = 1;
    }
    પાછા params;
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
સ્થિર તાર્કિક ggml_zendnn_matmul(ggml_backend_zendnn_context * ctx, int64_t m, int64_t n, int64_t k,
                               અચલ TA * A, int64_t lda, અચલ TB * B, int64_t ldb, TC * C,
                               int64_t ldc) {

    zendnnl::lowoha::matmul::matmul_params params = ggml_zendnn_make_matmul_params<TA, TB, TC>(ctx);

    zendnnl::lowoha::matmul::matmul_batch_params_t batch_params;

    જો constexpr (std::is_same_v<TA, block_q8_0>) {
        params.quant_params.src_scale.dims = {n, k / QK8_0};
    }

    zendnnl::error_handling::status_t status = zendnnl::lowoha::matmul::matmul_direct(
        'r', ખોટું, સાચું,   // row-major, don't transpose B, transpose A (because it's column-major)
        n,                  // M: rows of B and C
        m,                  // N: cols of A^T and C
        k,                  // K: cols of B, rows of A
        1.0f,               // alpha
        B, ldb,             // src: B[n,k]
        A, lda,             // weight: A[k,m] column-major (transposed)
        nullptr,            // bias
        0.0f,               // beta
        C, ldc,             // output C[n,m]
        સાચું,               // is_weights_const
        batch_params,       // batch_params
        params              // params
    );

    જો (status != zendnnl::error_handling::status_t::success) {
        GGML_LOG_ERROR("%s, ZenDNN matmul failed: status=%d\n", __func__, static_cast<પૂર્ણાંક>(status));
        પાછા ખોટું;
    }
    પાછા સાચું;
}

સ્થિર તાર્કિક ggml_zendnn_gemm(ggml_backend_zendnn_context * ctx, int64_t m, int64_t n, int64_t k,
                              અચલ શૂન્ય * A, int64_t lda, અચલ શૂન્ય * B, int64_t ldb, શૂન્ય * C,
                              int64_t ldc, પૂર્ણાંક Atype, પૂર્ણાંક Btype, પૂર્ણાંક Ctype) {

    પ્રતિજ્ઞા(m >= 0);
    પ્રતિજ્ઞા(n >= 0);
    પ્રતિજ્ઞા(k >= 0);
    પ્રતિજ્ઞા(lda >= k);
    પ્રતિજ્ઞા(ldb >= k);
    પ્રતિજ્ઞા(ldc >= m);

    // categorize types
    પસંદગી (Atype) {
        સ્થિતિ GGML_TYPE_F32:
            જો (Btype != GGML_TYPE_F32 || Ctype != GGML_TYPE_F32)
                પાછા ખોટું;
            પાછા ggml_zendnn_matmul<દશાંશ, દશાંશ, દશાંશ>(
                ctx, m, n, k,
                (અચલ દશાંશ *)A, lda,
                (અચલ દશાંશ *)B, ldb,
                (દશાંશ *)C, ldc);
        સ્થિતિ GGML_TYPE_BF16:
            જો (Btype != GGML_TYPE_BF16)
                પાછા ખોટું;
            જો (Ctype == GGML_TYPE_BF16)
                પાછા ggml_zendnn_matmul<ggml_bf16_t, ggml_bf16_t, ggml_bf16_t>(
                    ctx, m, n, k,
                    (અચલ ggml_bf16_t *)A, lda,
                    (અચલ ggml_bf16_t *)B, ldb,
                    (ggml_bf16_t *)C, ldc);
            જો (Ctype == GGML_TYPE_F32)
                પાછા ggml_zendnn_matmul<ggml_bf16_t, ggml_bf16_t, દશાંશ>(
                    ctx, m, n, k,
                    (અચલ ggml_bf16_t *)A, lda,
                    (અચલ ggml_bf16_t *)B, ldb,
                    (દશાંશ *)C, ldc);
            પાછા ખોટું;
        સ્થિતિ GGML_TYPE_Q8_0:
            જો (Btype != GGML_TYPE_F32 || Ctype != GGML_TYPE_F32)
                પાછા ખોટું;
            પાછા ggml_zendnn_matmul<block_q8_0, દશાંશ, દશાંશ>(
                ctx, m, n, k,
                (અચલ block_q8_0 *)A, lda,
                (અચલ દશાંશ *)B, ldb,
                (દશાંશ *)C, ldc);
        મૂળભૂત:
            પાછા ખોટું; // unsupported type
    }
}

સ્થિર શૂન્ય ggml_zendnn_compute_forward_mul_mat(
    ggml_backend_zendnn_context * ctx,
    ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];  // weights
    અચલ ggml_tensor * src1 = dst->src[1];  // inputs

    GGML_TENSOR_દ્વિઅંકી_OP_LOCALS

    ggml_type         અચલ vec_dot_type = src0->type;
    ggml_from_float_t અચલ from_float = ggml_get_type_traits(vec_dot_type)->from_float_ref;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == કદ(દશાંશ));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // broadcast factors
    અચલ int64_t r2 = ne12/ne02;
    અચલ int64_t r3 = ne13/ne03;

    શૂન્ય * work_data = ctx->work_data.get();

    // ZenDNN requires FP32 for dynamic quantization, so conversion is skipped
    જો (src1->type != vec_dot_type && src0->type != GGML_TYPE_Q8_0) {
        અચલ size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
        અચલ size_t nbw2 = nbw1 * ne11;
        અચલ size_t nbw3 = nbw2 * ne12;
        અચલ size_t desired_wsize = ne13 * nbw3;
        જો (ctx->work_size < desired_wsize) {
            ctx->work_data.reset(નવું અક્ષર[desired_wsize]);
            ctx->work_size = desired_wsize;
        }
        work_data = ctx->work_data.get();

        // #pragma omp parallel for num_threads(ctx->n_threads)
        #pragma omp parallel માટે collapse(3) અંક_threads(ctx->n_threads) schedule(સ્થિર)
        માટે (int64_t i13 = 0; i13 < ne13; ++i13) {
            માટે (int64_t i12 = 0; i12 < ne12; ++i12) {
                માટે (int64_t i11 = 0; i11 < ne11; ++i11) {
                    અચલ દશાંશ * src1_f32 = (દશાંશ *)((અક્ષર *)src1->data + i11*nb11 + i12*nb12 + i13*nb13);
                    શૂન્ય * src1_conv = (અક્ષર *)work_data + i11*nbw1 + i12*nbw2 + i13*nbw3;
                    from_float(src1_f32, src1_conv, ne10);
                }
            }
        }
    }

    માટે (int64_t i13 = 0; i13 < ne13; i13++) {
        માટે (int64_t i12 = 0; i12 < ne12; i12++) {
            અચલ શૂન્ય* wdata = (src1->type == vec_dot_type || src0->type == GGML_TYPE_Q8_0) ? src1->data : work_data;
            અચલ size_t row_size = ggml_row_size(vec_dot_type, ne10);
            જો (!ggml_zendnn_gemm(ctx,
                                  ne01,     // m
                                  ne11,     // n
                                  ne10,     // k
                                  static_cast<અચલ અક્ષર *>(src0->data) + (i12/r2)*nb02 + (i13/r3)*nb03,
                                  ne00,     // lda
                                  static_cast<અચલ અક્ષર *>(wdata) + (i12*ne11 + i13*ne12*ne11)*row_size,
                                  ne10,     // ldb
                                  static_cast<અક્ષર *>(dst->data) + i12*nb2 + i13*nb3,
                                  ne01,     // ldc
                                  src0->type,
                                  src0->type == GGML_TYPE_Q8_0 ? GGML_TYPE_F32 : vec_dot_type,
                                  dst->type))
                GGML_ABORT("%s: ZenDNN gemm failed\n", __func__);
        }
    }
}

સંરચના mmid_row_mapping {
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
સ્થિર તાર્કિક ggml_zendnn_group_matmul(ggml_backend_zendnn_context * ctx, int64_t m, int64_t k,
                                     અચલ std::vector<int64_t> & n,
                                     અચલ std::vector<અચલ શૂન્ય *> & A, int64_t lda,
                                     અચલ std::vector<અચલ શૂન્ય *> & B, int64_t ldb,
                                     અચલ std::vector<શૂન્ય *> & C, int64_t ldc) {

    અચલ પૂર્ણાંક n_experts = n.size();

    zendnnl::lowoha::matmul::matmul_params base_params = ggml_zendnn_make_matmul_params<TA, TB, TC>(ctx);

    std::vector<અક્ષર>         layout(n_experts, 'r');
    std::vector<તાર્કિક>         trans_a(n_experts, ખોટું);
    std::vector<તાર્કિક>         trans_b(n_experts, સાચું);
    std::vector<પૂર્ણાંક>          batch_m(n_experts);
    std::vector<પૂર્ણાંક>          batch_n(n_experts, m);
    std::vector<પૂર્ણાંક>          batch_k(n_experts, k);
    std::vector<દશાંશ>        alpha(n_experts, 1.0f);
    std::vector<દશાંશ>        beta(n_experts, 0.0f);
    std::vector<અચલ શૂન્ય *> bias(n_experts, nullptr);
    std::vector<પૂર્ણાંક>          lda_v(n_experts, lda);
    std::vector<પૂર્ણાંક>          ldb_v(n_experts, ldb);
    std::vector<પૂર્ણાંક>          ldc_v(n_experts, ldc);
    std::vector<તાર્કિક>         is_wei_const(n_experts, સાચું);
    std::vector<zendnnl::lowoha::matmul::matmul_params> params(n_experts, base_params);

    માટે (પૂર્ણાંક i = 0; i < n_experts; i++) {
        batch_m[i] = n[i];

        // src_scale.dims depends on this expert's row count, unlike the rest of base_params
        જો constexpr (std::is_same_v<TA, block_q8_0>) {
            params[i].quant_params.src_scale.dims = {n[i], k / QK8_0};
        }
    }

    zendnnl::error_handling::status_t status = zendnnl::lowoha::matmul::group_matmul_direct(
        layout, trans_a, trans_b, batch_m, batch_n, batch_k, alpha,
        B, ldb_v, A, lda_v, bias, beta,
        C, ldc_v, is_wei_const, params);

    જો (status != zendnnl::error_handling::status_t::success) {
        GGML_LOG_ERROR("%s, ZenDNN group matmul failed: status=%d\n", __func__, static_cast<પૂર્ણાંક>(status));
        પાછા ખોટું;
    }
    પાછા સાચું;
}

સ્થિર તાર્કિક ggml_zendnn_group_gemm(ggml_backend_zendnn_context * ctx, int64_t m, int64_t k,
                                    અચલ std::vector<int64_t> & n,
                                    અચલ std::vector<અચલ શૂન્ય *> & A, int64_t lda,
                                    અચલ std::vector<અચલ શૂન્ય *> & B, int64_t ldb,
                                    અચલ std::vector<શૂન્ય *> & C, int64_t ldc,
                                    પૂર્ણાંક Atype, પૂર્ણાંક Btype, પૂર્ણાંક Ctype) {

    પ્રતિજ્ઞા(m >= 0);
    માટે (size_t i = 0; i < n.size(); i++) {
        પ્રતિજ્ઞા(n[i] >= 0);
    }
    પ્રતિજ્ઞા(k >= 0);
    પ્રતિજ્ઞા(lda >= k);
    પ્રતિજ્ઞા(ldb >= k);
    પ્રતિજ્ઞા(ldc >= m);

    // categorize types
    પસંદગી (Atype) {
        સ્થિતિ GGML_TYPE_F32:
            જો (Btype != GGML_TYPE_F32 || Ctype != GGML_TYPE_F32)
                પાછા ખોટું;
            પાછા ggml_zendnn_group_matmul<દશાંશ, દશાંશ, દશાંશ>(ctx, m, k, n, A, lda, B, ldb, C, ldc);
        સ્થિતિ GGML_TYPE_BF16:
            જો (Btype != GGML_TYPE_BF16)
                પાછા ખોટું;
            જો (Ctype == GGML_TYPE_BF16)
                પાછા ggml_zendnn_group_matmul<ggml_bf16_t, ggml_bf16_t, ggml_bf16_t>(
                    ctx, m, k, n, A, lda, B, ldb, C, ldc);
            જો (Ctype == GGML_TYPE_F32)
                પાછા ggml_zendnn_group_matmul<ggml_bf16_t, ggml_bf16_t, દશાંશ>(ctx, m, k, n, A, lda, B, ldb, C, ldc);
            પાછા ખોટું;
        સ્થિતિ GGML_TYPE_Q8_0:
            જો (Btype != GGML_TYPE_F32 || Ctype != GGML_TYPE_F32)
                પાછા ખોટું;
            પાછા ggml_zendnn_group_matmul<block_q8_0, દશાંશ, દશાંશ>(ctx, m, k, n, A, lda, B, ldb, C, ldc);
        મૂળભૂત:
            પાછા ખોટું; // unsupported type
    }
}

સ્થિર શૂન્ય ggml_zendnn_compute_forward_mul_mat_id(
    ggml_backend_zendnn_context * ctx,
    ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];  // expert weights
    અચલ ggml_tensor * src1 = dst->src[1];  // inputs
    અચલ ggml_tensor * ids  = dst->src[2];  // expert ids

    GGML_TENSOR_દ્વિઅંકી_OP_LOCALS

    // exit for no tokens to process
    જો (ne2 == 0 || ne11 == 0) {
        પાછા;
    }

    ggml_type         અચલ vec_dot_type = src0->type;
    ggml_from_float_t અચલ from_float = ggml_get_type_traits(vec_dot_type)->from_float_ref;

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == કદ(દશાંશ));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(ne03 == 1);
    GGML_ASSERT(ne13 == 1);
    GGML_ASSERT(ne3  == 1);

    // row groups
    અચલ પૂર્ણાંક n_ids = ids->ne[0]; // n_expert_used
    અચલ પૂર્ણાંક n_as  = ne02;       // n_experts

    std::vector<int64_t> matrix_row_counts(n_as, 0);
    std::vector<std::vector<mmid_row_mapping>> matrix_rows(n_as);

    int64_t total_rows = 0;
    પૂર્ણાંક     n_active_experts = 0;
    // group rows by expert (preprocessing step)
    માટે (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
        માટે (પૂર્ણાંક id = 0; id < n_ids; ++id) {
            અચલ int32_t i02 = *(અચલ int32_t *)((અચલ અક્ષર *)ids->data + iid1*ids->nb[1] + id*ids->nb[0]);

            GGML_ASSERT(i02 >= 0 && i02 < n_as);

            જો (matrix_row_counts[i02] == 0) {
                n_active_experts++;
            }
            matrix_rows[i02].push_back({id, iid1});
            matrix_row_counts[i02]++;
            total_rows++;
        }
    }

    જો (total_rows == 0) {
        પાછા; // no rows to process
    }

    અચલ size_t row_size = ggml_row_size(vec_dot_type, ne10);

    // For Q8_0, src1 is always F32; the gather buffer must hold F32 rows (ne10*4 bytes),
    // not Q8_0-encoded rows (row_size ≈ ne10/32*34 bytes) — they differ by ~4x.
    અચલ size_t f32_row_size = (size_t)ne10 * કદ(દશાંશ);
    અચલ size_t gather_row_size = (src0->type == GGML_TYPE_Q8_0) ? f32_row_size : row_size;

    જો (src1->type != vec_dot_type && src0->type != GGML_TYPE_Q8_0) {
        GGML_ASSERT(src1->type == GGML_TYPE_F32);
    }

    // size for MoE gather/scatter buffers
    અચલ size_t wdata_cur_size = total_rows * gather_row_size;
    અચલ size_t dst_cur_size = total_rows * ggml_row_size(dst->type, ne01);

    // allocate single buffer for all needs
    અચલ size_t total_size = wdata_cur_size + dst_cur_size;
    જો (ctx->work_size < total_size) {
        ctx->work_data.reset(નવું અક્ષર[total_size]);
        ctx->work_size = total_size;
    }

    // partition the buffer
    અક્ષર * wdata_cur = ctx->work_data.get();
    અક્ષર * dst_cur = wdata_cur + wdata_cur_size;

    // per-expert data collected during gather, handed to ggml_zendnn_group_gemm() as one batch
    std::vector<int64_t>      expert_row_ગણતરી(n_active_experts);
    std::vector<અચલ શૂન્ય *> batch_src(n_active_experts);
    std::vector<અચલ શૂન્ય *> batch_wei(n_active_experts);
    std::vector<શૂન્ય *>       batch_dst(n_active_experts);

    // precompute per-expert buffer offsets and batch indices for the parallel loop below
    std::vector<int64_t> expert_wdata_off(n_as, 0);
    std::vector<int64_t> expert_dst_off(n_as, 0);
    std::vector<પૂર્ણાંક>     expert_batch_idx(n_as, -1);
    {
        int64_t w_off = 0;
        int64_t d_off = 0;
        પૂર્ણાંક     batch_idx = 0;
        માટે (int64_t cur_a = 0; cur_a < n_as; ++cur_a) {
            જો (matrix_row_counts[cur_a] == 0) {
                ચાલુ;
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
    #pragma omp parallel માટે અંક_threads(ctx->n_threads) schedule(સ્થિર)
    માટે (int64_t cur_a = 0; cur_a < n_as; ++cur_a) {
        અચલ int64_t cne1 = matrix_row_counts[cur_a];

        જો (cne1 == 0) {
            ચાલુ;
        }

        અચલ int64_t w_off     = expert_wdata_off[cur_a];
        અચલ int64_t d_off     = expert_dst_off[cur_a];
        અચલ પૂર્ણાંક     batch_idx = expert_batch_idx[cur_a];

        માટે (int64_t ir1 = 0; ir1 < cne1; ++ir1) {
            અચલ mmid_row_mapping & row_mapping = matrix_rows[cur_a][ir1];
            અચલ int64_t id  = row_mapping.i1;
            અચલ int64_t i11 = id % ne11;
            અચલ int64_t i12 = row_mapping.i2;

            અચલ અક્ષર * src_row = (અચલ અક્ષર *) src1->data + i11*nb11 + i12*nb12;
            શૂન્ય *       dst_row = wdata_cur + w_off + ir1 * gather_row_size;

            જો (src1->type != vec_dot_type && src0->type != GGML_TYPE_Q8_0) {
                from_float((અચલ દશાંશ *) src_row, dst_row, ne10);
            } નહીંતર {
                // no conversion: src1 already matches vec_dot_type, or src0 is Q8_0, whose
                // ZenDNN dynamic quantization requires the row to stay in F32
                std::memcpy(dst_row, src_row, gather_row_size);
            }
        }

        expert_row_ગણતરી[batch_idx] = cne1;
        batch_src[batch_idx]        = wdata_cur + w_off;
        batch_wei[batch_idx]        = (અચલ અક્ષર *) src0->data + cur_a * nb02;
        batch_dst[batch_idx]        = dst_cur + d_off;
    }

    જો (!ggml_zendnn_group_gemm(ctx,
                                ne01,             // m
                                ne10,             // k
                                expert_row_ગણતરી, // n (per expert)
                                batch_wei, ne00,  // A: weights, lda
                                batch_src, ne10,  // B: input, ldb
                                batch_dst, ne01,  // C: output, ldc
                                src0->type,
                                src0->type == GGML_TYPE_Q8_0 ? GGML_TYPE_F32 : vec_dot_type,
                                dst->type))
        GGML_ABORT("%s: ZenDNN group gemm failed\n", __func__);

    // scatter output rows to destination
    #pragma omp parallel માટે અંક_threads(ctx->n_threads) schedule(સ્થિર)
    માટે (int64_t cur_a = 0; cur_a < n_as; ++cur_a) {
        અચલ int64_t cne1 = matrix_row_counts[cur_a];

        જો (cne1 == 0) {
            ચાલુ;
        }

        અચલ int64_t d_off = expert_dst_off[cur_a];

        માટે (int64_t ir1 = 0; ir1 < cne1; ++ir1) {
            અચલ mmid_row_mapping & row_mapping = matrix_rows[cur_a][ir1];
            અચલ int64_t id = row_mapping.i1;
            અચલ int64_t i1 = id;
            અચલ int64_t i2 = row_mapping.i2;

            std::memcpy(
                (અક્ષર *) dst->data + i1*nb1 + i2*nb2,
                dst_cur + d_off + ir1 * ggml_row_size(dst->type, ne01),
                ggml_row_size(dst->type, ne01)
            );
        }
    }
}

// backend interface

સ્થિર અચલ અક્ષર * ggml_backend_zendnn_get_name(ggml_backend_t backend) {
    પાછા "ZenDNN";

    GGML_UNUSED(backend);
}

સ્થિર શૂન્ય ggml_backend_zendnn_free(ggml_backend_t backend) {
    ggml_backend_zendnn_context * ctx = (ggml_backend_zendnn_context *)backend->context;
    delete ctx;
    delete backend;
}

સ્થિર ggml_status ggml_backend_zendnn_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_zendnn_context * ctx = (ggml_backend_zendnn_context *)backend->context;

    માટે (પૂર્ણાંક i = 0; i < cgraph->n_nodes; i++) {
        સંરચના ggml_tensor * ગાંઠ = cgraph->nodes[i];

        જો ((ગાંઠ->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            ચાલુ;
        }

        પસંદગી (ગાંઠ->op) {
            સ્થિતિ GGML_OP_MUL_MAT:
                ggml_zendnn_compute_forward_mul_mat(ctx, ગાંઠ);
                તોડો;
            સ્થિતિ GGML_OP_MUL_MAT_ID:
                ggml_zendnn_compute_forward_mul_mat_id(ctx, ગાંઠ);
                તોડો;
            સ્થિતિ GGML_OP_NONE:
            સ્થિતિ GGML_OP_RESHAPE:
            સ્થિતિ GGML_OP_VIEW:
            સ્થિતિ GGML_OP_PERMUTE:
            સ્થિતિ GGML_OP_TRANSPOSE:
                તોડો;

            મૂળભૂત:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(ગાંઠ));
        }
    }

    પાછા GGML_STATUS_SUCCESS;

    GGML_UNUSED(backend);
}

સ્થિર સંરચના ggml_backend_i ggml_backend_zendnn_i = {
    /* .get_name                = */ ggml_backend_zendnn_get_name,
    /* .free                    = */ ggml_backend_zendnn_free,
    /* .set_tensor_async        = */ ખાલી,
    /* .get_tensor_async        = */ ખાલી,
    /* .set_tensor_2d_async     = */ ખાલી,
    /* .get_tensor_2d_async     = */ ખાલી,
    /* .cpy_tensor_async        = */ ખાલી,
    /* .synchronize             = */ ખાલી,
    /* .graph_plan_create       = */ ખાલી,
    /* .graph_plan_free         = */ ખાલી,
    /* .graph_plan_update       = */ ખાલી,
    /* .graph_plan_compute      = */ ખાલી,
    /* .graph_compute           = */ ggml_backend_zendnn_graph_compute,
    /* .event_record            = */ ખાલી,
    /* .event_wait              = */ ખાલી,
    /* .graph_optimize          = */ ખાલી,
};

સ્થિર ggml_guid_t ggml_backend_zendnn_guid(શૂન્ય) {
    સ્થિર અચલ અક્ષર * guid_str = "AMD-ZENDNN-ACCEL";
    પાછા reinterpret_cast<ggml_guid_t>(const_cast<અક્ષર*>(guid_str));
}

ggml_backend_t ggml_backend_zendnn_init(શૂન્ય) {
    ggml_backend_zendnn_context * ctx = નવું ggml_backend_zendnn_context;

    ggml_backend_t backend = નવું ggml_backend {
        /* .guid    = */ ggml_backend_zendnn_guid(),
        /* .iface   = */ ggml_backend_zendnn_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_zendnn_reg(), 0),
        /* .context = */ ctx,
    };

    પાછા backend;
}

તાર્કિક ggml_backend_is_zendnn(ggml_backend_t backend) {
    પાછા backend != ખાલી && ggml_guid_matches(backend->guid, ggml_backend_zendnn_guid());
}

શૂન્ય ggml_backend_zendnn_set_n_threads(ggml_backend_t backend_zendnn, પૂર્ણાંક n_threads) {
    GGML_ASSERT(ggml_backend_is_zendnn(backend_zendnn));

    ggml_backend_zendnn_context * ctx = (ggml_backend_zendnn_context *)backend_zendnn->context;
    ctx->n_threads = n_threads;
}

// device interface
સ્થિર અચલ અક્ષર * ggml_backend_zendnn_device_get_name(ggml_backend_dev_t dev) {
    પાછા "ZenDNN";

    GGML_UNUSED(dev);
}
/**
 * ZenDNN is AMD's performance library providing optimized primitives and implementations
 * for deep learning workloads on AMD CPUs. It targets improved performance for common
 * neural network operations on AMD architectures. For more information, see:
 * https://www.amd.com/en/developer/zendnn.html
 */
સ્થિર અચલ અક્ષર * ggml_backend_zendnn_device_get_description(ggml_backend_dev_t dev) {
    પાછા "ZenDNN: AMD optimized primitives backend for GGML (optimized for AMD CPUs)";

    GGML_UNUSED(dev);
}

સ્થિર શૂન્ય ggml_backend_zendnn_device_get_memory(ggml_backend_dev_t dev, size_t * મુક્ત, size_t * total) {
    *મુક્ત  = 0;
    *total = 0;

    GGML_UNUSED(dev);
}

સ્થિર ગણના ggml_backend_dev_type ggml_backend_zendnn_device_get_type(ggml_backend_dev_t dev) {
    પાછા GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_UNUSED(dev);
}

સ્થિર શૂન્ય ggml_backend_zendnn_device_get_props(ggml_backend_dev_t dev, સંરચના ggml_backend_dev_props * props) {
    props->name        = ggml_backend_zendnn_device_get_name(dev);
    props->description = ggml_backend_zendnn_device_get_description(dev);
    props->type        = ggml_backend_zendnn_device_get_type(dev);
    ggml_backend_zendnn_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                = */ ખોટું,
        /* .host_buffer          = */ ખોટું,
        /* .buffer_from_host_ptr = */ સાચું,
        /* .events               = */ ખોટું,
        /* .mmap_support         = */ સાચું,
    };
}

સ્થિર ggml_backend_t ggml_backend_zendnn_device_init_backend(ggml_backend_dev_t dev, અચલ અક્ષર * params) {
    ggml_backend_t backend = ggml_backend_zendnn_init();
    જો (backend == ખાલી) {
        GGML_LOG_ERROR("%s: error: failed to initialize ZenDNN backend\n", __func__);
        પાછા ખાલી;
    }

    પાછા backend;

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

સ્થિર ggml_backend_buffer_type_t ggml_backend_zendnn_device_get_buffer_type(ggml_backend_dev_t dev) {
    પાછા ggml_backend_cpu_buffer_type();

    GGML_UNUSED(dev);
}

સ્થિર ggml_backend_buffer_t ggml_backend_zendnn_device_buffer_from_host_ptr(ggml_backend_dev_t dev, શૂન્ય * ptr, size_t size, size_t max_tensor_size) {
    પાછા ggml_backend_cpu_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

સ્થિર તાર્કિક ggml_zendnn_adaptive_fallback_enabled() {
    સ્થિર અચલ તાર્કિક enabled = std::getenv("GGML_ZENDNN_ADAPTIVE_FALLBACK") == nullptr ||
                                std::atoi(std::getenv("GGML_ZENDNN_ADAPTIVE_FALLBACK")) != 0;
    પાછા enabled;
}

સ્થિર તાર્કિક ggml_backend_zendnn_device_supports_op(ggml_backend_dev_t dev, અચલ સંરચના ggml_tensor * op) {
    પસંદગી (op->op) {
        સ્થિતિ GGML_OP_NONE:
        સ્થિતિ GGML_OP_RESHAPE:
        સ્થિતિ GGML_OP_VIEW:
        સ્થિતિ GGML_OP_PERMUTE:
        સ્થિતિ GGML_OP_TRANSPOSE:
            પાછા સાચું;

        સ્થિતિ GGML_OP_MUL_MAT:
        સ્થિતિ GGML_OP_MUL_MAT_ID:
        {
            અચલ ggml_tensor * weights = op->src[0];
            અચલ ggml_tensor * inputs = op->src[1];

            અચલ int64_t ne10 = inputs->ne[0];
            અચલ int64_t ne0 = op->ne[0];
            અચલ int64_t ne1 = op->ne[1];
            અચલ int64_t min_batch = 1;

            જો(!ggml_is_contiguous(weights) || !ggml_is_contiguous(inputs)) {
                પાછા ખોટું;
            }

            જો (ggml_zendnn_adaptive_fallback_enabled()) {
                અચલ int64_t K = inputs->ne[0];
                અચલ int64_t N = (inputs->ne[1]*inputs->ne[2]*inputs->ne[3]);
                અચલ int64_t M = weights->ne[1];
                જો(K <= 256 || N <= 128 || M <= 96) {
                    પાછા ખોટું;
                }

                // MUL_MAT_ID's gather+matmul+scatter approach favors a moderate expert count
                જો (op->op == GGML_OP_MUL_MAT_ID) {
                    અચલ int64_t n_experts = weights->ne[2];
                    અચલ int64_t max_experts = 32;
                    જો (n_experts > max_experts) {
                        પાછા ખોટું;
                    }

                    // fall back once the average rows per expert (N / n_experts) is too thin
                    // to amortize each per-expert GEMM's overhead
                    જો (N / n_experts <= 32) {
                        પાછા ખોટું;
                    }
                }
            }
            નહીંતર જો (ne0 < min_batch || ne1 < min_batch || ne10 < min_batch) {
                પાછા ખોટું;
            }

            પસંદગી (weights->type) {
                સ્થિતિ GGML_TYPE_F32:
                સ્થિતિ GGML_TYPE_BF16:
                સ્થિતિ GGML_TYPE_Q8_0:
                    પાછા સાચું;
                મૂળભૂત:
                    પાછા ખોટું;
            }
        } તોડો;

        મૂળભૂત:
            પાછા ખોટું;
    }

    GGML_UNUSED(dev);
}

સ્થિર તાર્કિક ggml_backend_zendnn_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    પાછા ggml_backend_buft_is_host(buft);

    GGML_UNUSED(dev);
}

સ્થિર અચલ સંરચના ggml_backend_device_i ggml_backend_zendnn_device_i = {
    /* .get_name               = */ ggml_backend_zendnn_device_get_name,
    /* .get_description        = */ ggml_backend_zendnn_device_get_description,
    /* .get_memory             = */ ggml_backend_zendnn_device_get_memory,
    /* .get_type               = */ ggml_backend_zendnn_device_get_type,
    /* .get_props              = */ ggml_backend_zendnn_device_get_props,
    /* .init_backend           = */ ggml_backend_zendnn_device_init_backend,
    /* .get_buffer_type        = */ ggml_backend_zendnn_device_get_buffer_type,
    /* .get_host_buffer_type   = */ ખાલી,
    /* .buffer_from_host_ptr   = */ ggml_backend_zendnn_device_buffer_from_host_ptr,
    /* .supports_op            = */ ggml_backend_zendnn_device_supports_op,
    /* .supports_buft          = */ ggml_backend_zendnn_device_supports_buft,
    /* .offload_op             = */ ખાલી,
    /* .event_new              = */ ખાલી,
    /* .event_free             = */ ખાલી,
    /* .event_synchronize      = */ ખાલી,
};

// backend reg interface
સ્થિર અચલ અક્ષર * ggml_backend_zendnn_reg_get_name(ggml_backend_reg_t reg) {
    પાછા "ZenDNN";

    GGML_UNUSED(reg);
}

સ્થિર size_t ggml_backend_zendnn_reg_get_device_ગણતરી(ggml_backend_reg_t reg) {
    પાછા 1;

    GGML_UNUSED(reg);
}

સ્થિર ggml_backend_dev_t ggml_backend_zendnn_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    સ્થિર ggml_backend_device ggml_backend_zendnn_device = {
        /* .iface   = */ ggml_backend_zendnn_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };

    પાછા &ggml_backend_zendnn_device;
}

સ્થિર શૂન્ય * ggml_backend_zendnn_get_proc_address(ggml_backend_reg_t reg, અચલ અક્ષર * name) {
    જો (std::strcmp(name, "ggml_backend_set_n_threads") == 0) {
        પાછા (શૂન્ય *) ggml_backend_zendnn_set_n_threads;
    }
    પાછા ખાલી;

    GGML_UNUSED(reg);
    GGML_UNUSED(name);
}

સ્થિર અચલ સંરચના ggml_backend_reg_i ggml_backend_zendnn_reg_i = {
    /* .get_name         = */ ggml_backend_zendnn_reg_get_name,
    /* .get_device_count = */ ggml_backend_zendnn_reg_get_device_ગણતરી,
    /* .get_device       = */ ggml_backend_zendnn_reg_get_device,
    /* .get_proc_address = */ ggml_backend_zendnn_get_proc_address,
};

ggml_backend_reg_t ggml_backend_zendnn_reg(શૂન્ય) {
    સ્થિર સંરચના ggml_backend_reg ggml_backend_zendnn_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_zendnn_reg_i,
        /* .context     = */ ખાલી,
    };

    પાછા &ggml_backend_zendnn_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_zendnn_reg)
