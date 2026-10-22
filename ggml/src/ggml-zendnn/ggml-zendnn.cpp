#समावेश "ggml-zendnn.h"

#समावेश "ggml-backend-impl.h"
#समावेश "ggml-impl.h"

#घोषणा GGML_COMMON_DECL_CPP
#समावेश "ggml-common.h"

#समावेश "zendnnl.hpp"

#समावेश <cstring>


काष्ठा ggml_backend_zendnn_context {
    पूर्णांक n_threads = GGML_DEFAULT_N_THREADS;
    std::unique_ptr<अक्षर[]> work_तथ्य;
    माप_प्रकार work_आकार = 0;
};

ढाँचा<typename T>
zendnnl::common::तथ्य_type_t ggml_to_zendnn_type() {
    अगर constexpr (std::is_same_v<T, भग्न>) {
        वापस zendnnl::common::तथ्य_type_t::f32;
    } अन्यथा अगर constexpr (std::is_same_v<T, ggml_bf16_t>) {
        वापस zendnnl::common::तथ्य_type_t::bf16;
    } अन्यथा अगर constexpr (std::is_same_v<T, block_q8_0>) {
        वापस zendnnl::common::तथ्य_type_t::s8;
    } अन्यथा {
        वापस zendnnl::common::तथ्य_type_t::none;
    }
}

/**
 * Builds the matmul_params shared by ggml_zendnn_matmul() and ggml_zendnn_group_matmul():
 * dtype selection plus, for Q8_0 weights, dynamic-quant setup. Callers still need to set
 * quant_params.src_scale.dims themselves, since that depends on the batch size(s) in use.
 */
ढाँचा <typename TA, typename TB, typename TC>
अटल zendnnl::lowoha::matmul::matmul_params ggml_zendnn_make_matmul_params(ggml_backend_zendnn_context * ctx) {
    zendnnl::lowoha::matmul::matmul_params params;
    params.dtypes.src = ggml_to_zendnn_type<TB>();
    params.dtypes.wei = ggml_to_zendnn_type<TA>();
    params.dtypes.dst = ggml_to_zendnn_type<TC>();
    params.अंक_threads = ctx->n_threads;

    अगर constexpr (std::is_same_v<TA, block_q8_0>) {
        params.dtypes.compute = zendnnl::common::तथ्य_type_t::s8;
        params.dynamic_quant = true;
        params.quant_params.src_scale.buff = nullptr;
        params.quant_params.src_scale.dt   = zendnnl::common::तथ्य_type_t::bf16;
        params.packing.pack_format_b = 1;
    }
    वापस params;
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
ढाँचा <typename TA, typename TB, typename TC>
अटल bool ggml_zendnn_matmul(ggml_backend_zendnn_context * ctx, int64_t m, int64_t n, int64_t k,
                               स्थिर TA * A, int64_t lda, स्थिर TB * B, int64_t ldb, TC * C,
                               int64_t ldc) {

    zendnnl::lowoha::matmul::matmul_params params = ggml_zendnn_make_matmul_params<TA, TB, TC>(ctx);

    zendnnl::lowoha::matmul::matmul_batch_params_t batch_params;

    अगर constexpr (std::is_same_v<TA, block_q8_0>) {
        params.quant_params.src_scale.dims = {n, k / QK8_0};
    }

    zendnnl::error_handling::status_t status = zendnnl::lowoha::matmul::matmul_direct(
        'r', false, true,   // row-major, don't transpose B, transpose A (because it's column-major)
        n,                  // M: rows of B and C
        m,                  // N: cols of A^T and C
        k,                  // K: cols of B, rows of A
        1.0f,               // alpha
        B, ldb,             // src: B[n,k]
        A, lda,             // weight: A[k,m] column-major (transposed)
        nullptr,            // bias
        0.0f,               // beta
        C, ldc,             // output C[n,m]
        true,               // is_weights_const
        batch_params,       // batch_params
        params              // params
    );

    अगर (status != zendnnl::error_handling::status_t::success) {
        GGML_LOG_ERROR("%s, ZenDNN matmul failed: status=%d\n", __func__, static_cast<पूर्णांक>(status));
        वापस false;
    }
    वापस true;
}

अटल bool ggml_zendnn_gemm(ggml_backend_zendnn_context * ctx, int64_t m, int64_t n, int64_t k,
                              स्थिर व्योम * A, int64_t lda, स्थिर व्योम * B, int64_t ldb, व्योम * C,
                              int64_t ldc, पूर्णांक Atype, पूर्णांक Btype, पूर्णांक Ctype) {

    निश्चित(m >= 0);
    निश्चित(n >= 0);
    निश्चित(k >= 0);
    निश्चित(lda >= k);
    निश्चित(ldb >= k);
    निश्चित(ldc >= m);

    // categorize types
    चयन (Atype) {
        हाल GGML_TYPE_F32:
            अगर (Btype != GGML_TYPE_F32 || Ctype != GGML_TYPE_F32)
                वापस false;
            वापस ggml_zendnn_matmul<भग्न, भग्न, भग्न>(
                ctx, m, n, k,
                (स्थिर भग्न *)A, lda,
                (स्थिर भग्न *)B, ldb,
                (भग्न *)C, ldc);
        हाल GGML_TYPE_BF16:
            अगर (Btype != GGML_TYPE_BF16)
                वापस false;
            अगर (Ctype == GGML_TYPE_BF16)
                वापस ggml_zendnn_matmul<ggml_bf16_t, ggml_bf16_t, ggml_bf16_t>(
                    ctx, m, n, k,
                    (स्थिर ggml_bf16_t *)A, lda,
                    (स्थिर ggml_bf16_t *)B, ldb,
                    (ggml_bf16_t *)C, ldc);
            अगर (Ctype == GGML_TYPE_F32)
                वापस ggml_zendnn_matmul<ggml_bf16_t, ggml_bf16_t, भग्न>(
                    ctx, m, n, k,
                    (स्थिर ggml_bf16_t *)A, lda,
                    (स्थिर ggml_bf16_t *)B, ldb,
                    (भग्न *)C, ldc);
            वापस false;
        हाल GGML_TYPE_Q8_0:
            अगर (Btype != GGML_TYPE_F32 || Ctype != GGML_TYPE_F32)
                वापस false;
            वापस ggml_zendnn_matmul<block_q8_0, भग्न, भग्न>(
                ctx, m, n, k,
                (स्थिर block_q8_0 *)A, lda,
                (स्थिर भग्न *)B, ldb,
                (भग्न *)C, ldc);
        शेष:
            वापस false; // unsupported type
    }
}

अटल व्योम ggml_zendnn_compute_forward_mul_mat(
    ggml_backend_zendnn_context * ctx,
    ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];  // weights
    स्थिर ggml_tensor * src1 = dst->src[1];  // inputs

    GGML_TENSOR_द्विआधारी_OP_LOCALS

    ggml_type         स्थिर vec_dot_type = src0->type;
    ggml_from_float_t स्थिर from_float = ggml_get_type_traits(vec_dot_type)->from_float_ref;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_आकार(src0->type));
    GGML_ASSERT(nb10 == ggml_type_आकार(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == माप(भग्न));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // broadcast factors
    स्थिर int64_t r2 = ne12/ne02;
    स्थिर int64_t r3 = ne13/ne03;

    व्योम * work_तथ्य = ctx->work_तथ्य.लो();

    // ZenDNN requires FP32 for dynamic quantization, so conversion is skipped
    अगर (src1->type != vec_dot_type && src0->type != GGML_TYPE_Q8_0) {
        स्थिर माप_प्रकार nbw1 = ggml_row_आकार(vec_dot_type, ne10);
        स्थिर माप_प्रकार nbw2 = nbw1 * ne11;
        स्थिर माप_प्रकार nbw3 = nbw2 * ne12;
        स्थिर माप_प्रकार desired_wsize = ne13 * nbw3;
        अगर (ctx->work_आकार < desired_wsize) {
            ctx->work_तथ्य.reset(नया अक्षर[desired_wsize]);
            ctx->work_आकार = desired_wsize;
        }
        work_तथ्य = ctx->work_तथ्य.लो();

        // #pragma omp parallel for num_threads(ctx->n_threads)
        #आशय omp parallel क्रम collapse(3) अंक_threads(ctx->n_threads) schedule(अटल)
        क्रम (int64_t i13 = 0; i13 < ne13; ++i13) {
            क्रम (int64_t i12 = 0; i12 < ne12; ++i12) {
                क्रम (int64_t i11 = 0; i11 < ne11; ++i11) {
                    स्थिर भग्न * src1_f32 = (भग्न *)((अक्षर *)src1->तथ्य + i11*nb11 + i12*nb12 + i13*nb13);
                    व्योम * src1_conv = (अक्षर *)work_तथ्य + i11*nbw1 + i12*nbw2 + i13*nbw3;
                    from_float(src1_f32, src1_conv, ne10);
                }
            }
        }
    }

    क्रम (int64_t i13 = 0; i13 < ne13; i13++) {
        क्रम (int64_t i12 = 0; i12 < ne12; i12++) {
            स्थिर व्योम* wdata = (src1->type == vec_dot_type || src0->type == GGML_TYPE_Q8_0) ? src1->तथ्य : work_तथ्य;
            स्थिर माप_प्रकार row_आकार = ggml_row_आकार(vec_dot_type, ne10);
            अगर (!ggml_zendnn_gemm(ctx,
                                  ne01,     // m
                                  ne11,     // n
                                  ne10,     // k
                                  static_cast<स्थिर अक्षर *>(src0->तथ्य) + (i12/r2)*nb02 + (i13/r3)*nb03,
                                  ne00,     // lda
                                  static_cast<स्थिर अक्षर *>(wdata) + (i12*ne11 + i13*ne12*ne11)*row_आकार,
                                  ne10,     // ldb
                                  static_cast<अक्षर *>(dst->तथ्य) + i12*nb2 + i13*nb3,
                                  ne01,     // ldc
                                  src0->type,
                                  src0->type == GGML_TYPE_Q8_0 ? GGML_TYPE_F32 : vec_dot_type,
                                  dst->type))
                GGML_ABORT("%s: ZenDNN gemm failed\n", __func__);
        }
    }
}

काष्ठा mmid_row_mapping {
    int32_t i1;
    int32_t i2;
};

/**
 * ZenDNN batched matmul: computes C[i] = B[i] * A[i] for every active expert i via a single
 * group_matmul_direct() call. Batched analogue of ggml_zendnn_matmul() - see its docs for the
 * per-expert A/B/C shape convention. m and k are shared by every expert; n (batch size) varies
 * per expert, hence the vector.
 */
ढाँचा <typename TA, typename TB, typename TC>
अटल bool ggml_zendnn_group_matmul(ggml_backend_zendnn_context * ctx, int64_t m, int64_t k,
                                     स्थिर std::vector<int64_t> & n,
                                     स्थिर std::vector<स्थिर व्योम *> & A, int64_t lda,
                                     स्थिर std::vector<स्थिर व्योम *> & B, int64_t ldb,
                                     स्थिर std::vector<व्योम *> & C, int64_t ldc) {

    स्थिर पूर्णांक n_experts = n.size();

    zendnnl::lowoha::matmul::matmul_params base_params = ggml_zendnn_make_matmul_params<TA, TB, TC>(ctx);

    std::vector<अक्षर>         layout(n_experts, 'r');
    std::vector<bool>         trans_a(n_experts, false);
    std::vector<bool>         trans_b(n_experts, true);
    std::vector<पूर्णांक>          batch_m(n_experts);
    std::vector<पूर्णांक>          batch_n(n_experts, m);
    std::vector<पूर्णांक>          batch_k(n_experts, k);
    std::vector<भग्न>        alpha(n_experts, 1.0f);
    std::vector<भग्न>        beta(n_experts, 0.0f);
    std::vector<स्थिर व्योम *> bias(n_experts, nullptr);
    std::vector<पूर्णांक>          lda_v(n_experts, lda);
    std::vector<पूर्णांक>          ldb_v(n_experts, ldb);
    std::vector<पूर्णांक>          ldc_v(n_experts, ldc);
    std::vector<bool>         is_wei_const(n_experts, true);
    std::vector<zendnnl::lowoha::matmul::matmul_params> params(n_experts, base_params);

    क्रम (पूर्णांक i = 0; i < n_experts; i++) {
        batch_m[i] = n[i];

        // src_scale.dims depends on this expert's row count, unlike the rest of base_params
        अगर constexpr (std::is_same_v<TA, block_q8_0>) {
            params[i].quant_params.src_scale.dims = {n[i], k / QK8_0};
        }
    }

    zendnnl::error_handling::status_t status = zendnnl::lowoha::matmul::group_matmul_direct(
        layout, trans_a, trans_b, batch_m, batch_n, batch_k, alpha,
        B, ldb_v, A, lda_v, bias, beta,
        C, ldc_v, is_wei_const, params);

    अगर (status != zendnnl::error_handling::status_t::success) {
        GGML_LOG_ERROR("%s, ZenDNN group matmul failed: status=%d\n", __func__, static_cast<पूर्णांक>(status));
        वापस false;
    }
    वापस true;
}

अटल bool ggml_zendnn_group_gemm(ggml_backend_zendnn_context * ctx, int64_t m, int64_t k,
                                    स्थिर std::vector<int64_t> & n,
                                    स्थिर std::vector<स्थिर व्योम *> & A, int64_t lda,
                                    स्थिर std::vector<स्थिर व्योम *> & B, int64_t ldb,
                                    स्थिर std::vector<व्योम *> & C, int64_t ldc,
                                    पूर्णांक Atype, पूर्णांक Btype, पूर्णांक Ctype) {

    निश्चित(m >= 0);
    क्रम (माप_प्रकार i = 0; i < n.size(); i++) {
        निश्चित(n[i] >= 0);
    }
    निश्चित(k >= 0);
    निश्चित(lda >= k);
    निश्चित(ldb >= k);
    निश्चित(ldc >= m);

    // categorize types
    चयन (Atype) {
        हाल GGML_TYPE_F32:
            अगर (Btype != GGML_TYPE_F32 || Ctype != GGML_TYPE_F32)
                वापस false;
            वापस ggml_zendnn_group_matmul<भग्न, भग्न, भग्न>(ctx, m, k, n, A, lda, B, ldb, C, ldc);
        हाल GGML_TYPE_BF16:
            अगर (Btype != GGML_TYPE_BF16)
                वापस false;
            अगर (Ctype == GGML_TYPE_BF16)
                वापस ggml_zendnn_group_matmul<ggml_bf16_t, ggml_bf16_t, ggml_bf16_t>(
                    ctx, m, k, n, A, lda, B, ldb, C, ldc);
            अगर (Ctype == GGML_TYPE_F32)
                वापस ggml_zendnn_group_matmul<ggml_bf16_t, ggml_bf16_t, भग्न>(ctx, m, k, n, A, lda, B, ldb, C, ldc);
            वापस false;
        हाल GGML_TYPE_Q8_0:
            अगर (Btype != GGML_TYPE_F32 || Ctype != GGML_TYPE_F32)
                वापस false;
            वापस ggml_zendnn_group_matmul<block_q8_0, भग्न, भग्न>(ctx, m, k, n, A, lda, B, ldb, C, ldc);
        शेष:
            वापस false; // unsupported type
    }
}

अटल व्योम ggml_zendnn_compute_forward_mul_mat_id(
    ggml_backend_zendnn_context * ctx,
    ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];  // expert weights
    स्थिर ggml_tensor * src1 = dst->src[1];  // inputs
    स्थिर ggml_tensor * ids  = dst->src[2];  // expert ids

    GGML_TENSOR_द्विआधारी_OP_LOCALS

    // exit for no tokens to process
    अगर (ne2 == 0 || ne11 == 0) {
        वापस;
    }

    ggml_type         स्थिर vec_dot_type = src0->type;
    ggml_from_float_t स्थिर from_float = ggml_get_type_traits(vec_dot_type)->from_float_ref;

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_आकार(src0->type));
    GGML_ASSERT(nb10 == ggml_type_आकार(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == माप(भग्न));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(ne03 == 1);
    GGML_ASSERT(ne13 == 1);
    GGML_ASSERT(ne3  == 1);

    // row groups
    स्थिर पूर्णांक n_ids = ids->ne[0]; // n_expert_used
    स्थिर पूर्णांक n_as  = ne02;       // n_experts

    std::vector<int64_t> matrix_row_counts(n_as, 0);
    std::vector<std::vector<mmid_row_mapping>> matrix_rows(n_as);

    int64_t योग_rows = 0;
    पूर्णांक     n_active_experts = 0;
    // group rows by expert (preprocessing step)
    क्रम (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
        क्रम (पूर्णांक id = 0; id < n_ids; ++id) {
            स्थिर int32_t i02 = *(स्थिर int32_t *)((स्थिर अक्षर *)ids->तथ्य + iid1*ids->nb[1] + id*ids->nb[0]);

            GGML_ASSERT(i02 >= 0 && i02 < n_as);

            अगर (matrix_row_counts[i02] == 0) {
                n_active_experts++;
            }
            matrix_rows[i02].push_back({id, iid1});
            matrix_row_counts[i02]++;
            योग_rows++;
        }
    }

    अगर (योग_rows == 0) {
        वापस; // no rows to process
    }

    स्थिर माप_प्रकार row_आकार = ggml_row_आकार(vec_dot_type, ne10);

    // For Q8_0, src1 is always F32; the gather buffer must hold F32 rows (ne10*4 bytes),
    // not Q8_0-encoded rows (row_size ≈ ne10/32*34 bytes) — they differ by ~4x.
    स्थिर माप_प्रकार f32_row_आकार = (माप_प्रकार)ne10 * माप(भग्न);
    स्थिर माप_प्रकार gather_row_आकार = (src0->type == GGML_TYPE_Q8_0) ? f32_row_आकार : row_आकार;

    अगर (src1->type != vec_dot_type && src0->type != GGML_TYPE_Q8_0) {
        GGML_ASSERT(src1->type == GGML_TYPE_F32);
    }

    // size for MoE gather/scatter buffers
    स्थिर माप_प्रकार wdata_cur_आकार = योग_rows * gather_row_आकार;
    स्थिर माप_प्रकार dst_cur_आकार = योग_rows * ggml_row_आकार(dst->type, ne01);

    // allocate single buffer for all needs
    स्थिर माप_प्रकार योग_आकार = wdata_cur_आकार + dst_cur_आकार;
    अगर (ctx->work_आकार < योग_आकार) {
        ctx->work_तथ्य.reset(नया अक्षर[योग_आकार]);
        ctx->work_आकार = योग_आकार;
    }

    // partition the buffer
    अक्षर * wdata_cur = ctx->work_तथ्य.लो();
    अक्षर * dst_cur = wdata_cur + wdata_cur_आकार;

    // per-expert data collected during gather, handed to ggml_zendnn_group_gemm() as one batch
    std::vector<int64_t>      expert_row_गिनती(n_active_experts);
    std::vector<स्थिर व्योम *> batch_src(n_active_experts);
    std::vector<स्थिर व्योम *> batch_wei(n_active_experts);
    std::vector<व्योम *>       batch_dst(n_active_experts);

    // precompute per-expert buffer offsets and batch indices for the parallel loop below
    std::vector<int64_t> expert_wdata_off(n_as, 0);
    std::vector<int64_t> expert_dst_off(n_as, 0);
    std::vector<पूर्णांक>     expert_batch_idx(n_as, -1);
    {
        int64_t w_off = 0;
        int64_t d_off = 0;
        पूर्णांक     batch_idx = 0;
        क्रम (int64_t cur_a = 0; cur_a < n_as; ++cur_a) {
            अगर (matrix_row_counts[cur_a] == 0) {
                जारी;
            }
            expert_wdata_off[cur_a] = w_off;
            expert_dst_off[cur_a]   = d_off;
            expert_batch_idx[cur_a] = batch_idx;
            w_off += matrix_row_counts[cur_a] * gather_row_आकार;
            d_off += matrix_row_counts[cur_a] * ggml_row_आकार(dst->type, ne01);
            batch_idx++;
        }
    }

    // gather + inline-convert input rows into each expert's batch slot
    #आशय omp parallel क्रम अंक_threads(ctx->n_threads) schedule(अटल)
    क्रम (int64_t cur_a = 0; cur_a < n_as; ++cur_a) {
        स्थिर int64_t cne1 = matrix_row_counts[cur_a];

        अगर (cne1 == 0) {
            जारी;
        }

        स्थिर int64_t w_off     = expert_wdata_off[cur_a];
        स्थिर int64_t d_off     = expert_dst_off[cur_a];
        स्थिर पूर्णांक     batch_idx = expert_batch_idx[cur_a];

        क्रम (int64_t ir1 = 0; ir1 < cne1; ++ir1) {
            स्थिर mmid_row_mapping & row_mapping = matrix_rows[cur_a][ir1];
            स्थिर int64_t id  = row_mapping.i1;
            स्थिर int64_t i11 = id % ne11;
            स्थिर int64_t i12 = row_mapping.i2;

            स्थिर अक्षर * src_row = (स्थिर अक्षर *) src1->तथ्य + i11*nb11 + i12*nb12;
            व्योम *       dst_row = wdata_cur + w_off + ir1 * gather_row_आकार;

            अगर (src1->type != vec_dot_type && src0->type != GGML_TYPE_Q8_0) {
                from_float((स्थिर भग्न *) src_row, dst_row, ne10);
            } अन्यथा {
                // no conversion: src1 already matches vec_dot_type, or src0 is Q8_0, whose
                // ZenDNN dynamic quantization requires the row to stay in F32
                std::स_नकल(dst_row, src_row, gather_row_आकार);
            }
        }

        expert_row_गिनती[batch_idx] = cne1;
        batch_src[batch_idx]        = wdata_cur + w_off;
        batch_wei[batch_idx]        = (स्थिर अक्षर *) src0->तथ्य + cur_a * nb02;
        batch_dst[batch_idx]        = dst_cur + d_off;
    }

    अगर (!ggml_zendnn_group_gemm(ctx,
                                ne01,             // m
                                ne10,             // k
                                expert_row_गिनती, // n (per expert)
                                batch_wei, ne00,  // A: weights, lda
                                batch_src, ne10,  // B: input, ldb
                                batch_dst, ne01,  // C: output, ldc
                                src0->type,
                                src0->type == GGML_TYPE_Q8_0 ? GGML_TYPE_F32 : vec_dot_type,
                                dst->type))
        GGML_ABORT("%s: ZenDNN group gemm failed\n", __func__);

    // scatter output rows to destination
    #आशय omp parallel क्रम अंक_threads(ctx->n_threads) schedule(अटल)
    क्रम (int64_t cur_a = 0; cur_a < n_as; ++cur_a) {
        स्थिर int64_t cne1 = matrix_row_counts[cur_a];

        अगर (cne1 == 0) {
            जारी;
        }

        स्थिर int64_t d_off = expert_dst_off[cur_a];

        क्रम (int64_t ir1 = 0; ir1 < cne1; ++ir1) {
            स्थिर mmid_row_mapping & row_mapping = matrix_rows[cur_a][ir1];
            स्थिर int64_t id = row_mapping.i1;
            स्थिर int64_t i1 = id;
            स्थिर int64_t i2 = row_mapping.i2;

            std::स_नकल(
                (अक्षर *) dst->तथ्य + i1*nb1 + i2*nb2,
                dst_cur + d_off + ir1 * ggml_row_आकार(dst->type, ne01),
                ggml_row_आकार(dst->type, ne01)
            );
        }
    }
}

// backend interface

अटल स्थिर अक्षर * ggml_backend_zendnn_get_name(ggml_backend_t backend) {
    वापस "ZenDNN";

    GGML_UNUSED(backend);
}

अटल व्योम ggml_backend_zendnn_free(ggml_backend_t backend) {
    ggml_backend_zendnn_context * ctx = (ggml_backend_zendnn_context *)backend->context;
    मिटाओ ctx;
    मिटाओ backend;
}

अटल ggml_status ggml_backend_zendnn_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_zendnn_context * ctx = (ggml_backend_zendnn_context *)backend->context;

    क्रम (पूर्णांक i = 0; i < cgraph->n_nodes; i++) {
        काष्ठा ggml_tensor * गाँठ = cgraph->nodes[i];

        अगर ((गाँठ->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            जारी;
        }

        चयन (गाँठ->op) {
            हाल GGML_OP_MUL_MAT:
                ggml_zendnn_compute_forward_mul_mat(ctx, गाँठ);
                अवरोध;
            हाल GGML_OP_MUL_MAT_ID:
                ggml_zendnn_compute_forward_mul_mat_id(ctx, गाँठ);
                अवरोध;
            हाल GGML_OP_NONE:
            हाल GGML_OP_RESHAPE:
            हाल GGML_OP_VIEW:
            हाल GGML_OP_PERMUTE:
            हाल GGML_OP_TRANSPOSE:
                अवरोध;

            शेष:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(गाँठ));
        }
    }

    वापस GGML_STATUS_SUCCESS;

    GGML_UNUSED(backend);
}

अटल काष्ठा ggml_backend_i ggml_backend_zendnn_i = {
    /* .get_name                = */ ggml_backend_zendnn_get_name,
    /* .free                    = */ ggml_backend_zendnn_free,
    /* .set_tensor_async        = */ शून्य,
    /* .get_tensor_async        = */ शून्य,
    /* .set_tensor_2d_async     = */ शून्य,
    /* .get_tensor_2d_async     = */ शून्य,
    /* .cpy_tensor_async        = */ शून्य,
    /* .synchronize             = */ शून्य,
    /* .graph_plan_create       = */ शून्य,
    /* .graph_plan_free         = */ शून्य,
    /* .graph_plan_update       = */ शून्य,
    /* .graph_plan_compute      = */ शून्य,
    /* .graph_compute           = */ ggml_backend_zendnn_graph_compute,
    /* .event_record            = */ शून्य,
    /* .event_wait              = */ शून्य,
    /* .graph_optimize          = */ शून्य,
};

अटल ggml_guid_t ggml_backend_zendnn_guid(व्योम) {
    अटल स्थिर अक्षर * guid_str = "AMD-ZENDNN-ACCEL";
    वापस reinterpret_cast<ggml_guid_t>(const_cast<अक्षर*>(guid_str));
}

ggml_backend_t ggml_backend_zendnn_init(व्योम) {
    ggml_backend_zendnn_context * ctx = नया ggml_backend_zendnn_context;

    ggml_backend_t backend = नया ggml_backend {
        /* .guid    = */ ggml_backend_zendnn_guid(),
        /* .iface   = */ ggml_backend_zendnn_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_zendnn_reg(), 0),
        /* .context = */ ctx,
    };

    वापस backend;
}

bool ggml_backend_is_zendnn(ggml_backend_t backend) {
    वापस backend != शून्य && ggml_guid_matches(backend->guid, ggml_backend_zendnn_guid());
}

व्योम ggml_backend_zendnn_set_n_threads(ggml_backend_t backend_zendnn, पूर्णांक n_threads) {
    GGML_ASSERT(ggml_backend_is_zendnn(backend_zendnn));

    ggml_backend_zendnn_context * ctx = (ggml_backend_zendnn_context *)backend_zendnn->context;
    ctx->n_threads = n_threads;
}

// device interface
अटल स्थिर अक्षर * ggml_backend_zendnn_device_get_name(ggml_backend_dev_t dev) {
    वापस "ZenDNN";

    GGML_UNUSED(dev);
}
/**
 * ZenDNN is AMD's performance library providing optimized primitives and implementations
 * for deep learning workloads on AMD CPUs. It targets improved performance for common
 * neural network operations on AMD architectures. For more information, see:
 * https://www.amd.com/en/developer/zendnn.html
 */
अटल स्थिर अक्षर * ggml_backend_zendnn_device_get_description(ggml_backend_dev_t dev) {
    वापस "ZenDNN: AMD optimized primitives backend for GGML (optimized for AMD CPUs)";

    GGML_UNUSED(dev);
}

अटल व्योम ggml_backend_zendnn_device_get_memory(ggml_backend_dev_t dev, माप_प्रकार * मुक्त, माप_प्रकार * योग) {
    *मुक्त  = 0;
    *योग = 0;

    GGML_UNUSED(dev);
}

अटल क्रमागत ggml_backend_dev_type ggml_backend_zendnn_device_get_type(ggml_backend_dev_t dev) {
    वापस GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_UNUSED(dev);
}

अटल व्योम ggml_backend_zendnn_device_get_props(ggml_backend_dev_t dev, काष्ठा ggml_backend_dev_props * props) {
    props->name        = ggml_backend_zendnn_device_get_name(dev);
    props->description = ggml_backend_zendnn_device_get_description(dev);
    props->type        = ggml_backend_zendnn_device_get_type(dev);
    ggml_backend_zendnn_device_get_memory(dev, &props->memory_free, &props->memory_योग);
    props->caps = {
        /* .async                = */ false,
        /* .host_buffer          = */ false,
        /* .buffer_from_host_ptr = */ true,
        /* .events               = */ false,
        /* .mmap_support         = */ true,
    };
}

अटल ggml_backend_t ggml_backend_zendnn_device_init_backend(ggml_backend_dev_t dev, स्थिर अक्षर * params) {
    ggml_backend_t backend = ggml_backend_zendnn_init();
    अगर (backend == शून्य) {
        GGML_LOG_ERROR("%s: error: failed to initialize ZenDNN backend\n", __func__);
        वापस शून्य;
    }

    वापस backend;

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

अटल ggml_backend_बफर_type_t ggml_backend_zendnn_device_get_बफर_type(ggml_backend_dev_t dev) {
    वापस ggml_backend_cpu_बफर_type();

    GGML_UNUSED(dev);
}

अटल ggml_backend_बफर_t ggml_backend_zendnn_device_बफर_from_host_ptr(ggml_backend_dev_t dev, व्योम * ptr, माप_प्रकार size, माप_प्रकार max_tensor_आकार) {
    वापस ggml_backend_cpu_बफर_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_आकार);
}

अटल bool ggml_zendnn_adaptive_fallback_enabled() {
    अटल स्थिर bool enabled = std::दो_पर्या("GGML_ZENDNN_ADAPTIVE_FALLBACK") == nullptr ||
                                std::म_से_प(std::दो_पर्या("GGML_ZENDNN_ADAPTIVE_FALLBACK")) != 0;
    वापस enabled;
}

अटल bool ggml_backend_zendnn_device_supports_op(ggml_backend_dev_t dev, स्थिर काष्ठा ggml_tensor * op) {
    चयन (op->op) {
        हाल GGML_OP_NONE:
        हाल GGML_OP_RESHAPE:
        हाल GGML_OP_VIEW:
        हाल GGML_OP_PERMUTE:
        हाल GGML_OP_TRANSPOSE:
            वापस true;

        हाल GGML_OP_MUL_MAT:
        हाल GGML_OP_MUL_MAT_ID:
        {
            स्थिर ggml_tensor * weights = op->src[0];
            स्थिर ggml_tensor * inputs = op->src[1];

            स्थिर int64_t ne10 = inputs->ne[0];
            स्थिर int64_t ne0 = op->ne[0];
            स्थिर int64_t ne1 = op->ne[1];
            स्थिर int64_t min_batch = 1;

            अगर(!ggml_is_contiguous(weights) || !ggml_is_contiguous(inputs)) {
                वापस false;
            }

            अगर (ggml_zendnn_adaptive_fallback_enabled()) {
                स्थिर int64_t K = inputs->ne[0];
                स्थिर int64_t N = (inputs->ne[1]*inputs->ne[2]*inputs->ne[3]);
                स्थिर int64_t M = weights->ne[1];
                अगर(K <= 256 || N <= 128 || M <= 96) {
                    वापस false;
                }

                // MUL_MAT_ID's gather+matmul+scatter approach favors a moderate expert count
                अगर (op->op == GGML_OP_MUL_MAT_ID) {
                    स्थिर int64_t n_experts = weights->ne[2];
                    स्थिर int64_t max_experts = 32;
                    अगर (n_experts > max_experts) {
                        वापस false;
                    }

                    // fall back once the average rows per expert (N / n_experts) is too thin
                    // to amortize each per-expert GEMM's overhead
                    अगर (N / n_experts <= 32) {
                        वापस false;
                    }
                }
            }
            अन्यथा अगर (ne0 < min_batch || ne1 < min_batch || ne10 < min_batch) {
                वापस false;
            }

            चयन (weights->type) {
                हाल GGML_TYPE_F32:
                हाल GGML_TYPE_BF16:
                हाल GGML_TYPE_Q8_0:
                    वापस true;
                शेष:
                    वापस false;
            }
        } अवरोध;

        शेष:
            वापस false;
    }

    GGML_UNUSED(dev);
}

अटल bool ggml_backend_zendnn_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_बफर_type_t buft) {
    वापस ggml_backend_buft_is_host(buft);

    GGML_UNUSED(dev);
}

अटल स्थिर काष्ठा ggml_backend_device_i ggml_backend_zendnn_device_i = {
    /* .get_name               = */ ggml_backend_zendnn_device_get_name,
    /* .get_description        = */ ggml_backend_zendnn_device_get_description,
    /* .get_memory             = */ ggml_backend_zendnn_device_get_memory,
    /* .get_type               = */ ggml_backend_zendnn_device_get_type,
    /* .get_props              = */ ggml_backend_zendnn_device_get_props,
    /* .init_backend           = */ ggml_backend_zendnn_device_init_backend,
    /* .get_buffer_type        = */ ggml_backend_zendnn_device_get_बफर_type,
    /* .get_host_buffer_type   = */ शून्य,
    /* .buffer_from_host_ptr   = */ ggml_backend_zendnn_device_बफर_from_host_ptr,
    /* .supports_op            = */ ggml_backend_zendnn_device_supports_op,
    /* .supports_buft          = */ ggml_backend_zendnn_device_supports_buft,
    /* .offload_op             = */ शून्य,
    /* .event_new              = */ शून्य,
    /* .event_free             = */ शून्य,
    /* .event_synchronize      = */ शून्य,
};

// backend reg interface
अटल स्थिर अक्षर * ggml_backend_zendnn_reg_get_name(ggml_backend_reg_t reg) {
    वापस "ZenDNN";

    GGML_UNUSED(reg);
}

अटल माप_प्रकार ggml_backend_zendnn_reg_get_device_गिनती(ggml_backend_reg_t reg) {
    वापस 1;

    GGML_UNUSED(reg);
}

अटल ggml_backend_dev_t ggml_backend_zendnn_reg_get_device(ggml_backend_reg_t reg, माप_प्रकार सूचकांक) {
    GGML_ASSERT(सूचकांक == 0);

    अटल ggml_backend_device ggml_backend_zendnn_device = {
        /* .iface   = */ ggml_backend_zendnn_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };

    वापस &ggml_backend_zendnn_device;
}

अटल व्योम * ggml_backend_zendnn_get_proc_address(ggml_backend_reg_t reg, स्थिर अक्षर * name) {
    अगर (std::म_भेद(name, "ggml_backend_set_n_threads") == 0) {
        वापस (व्योम *) ggml_backend_zendnn_set_n_threads;
    }
    वापस शून्य;

    GGML_UNUSED(reg);
    GGML_UNUSED(name);
}

अटल स्थिर काष्ठा ggml_backend_reg_i ggml_backend_zendnn_reg_i = {
    /* .get_name         = */ ggml_backend_zendnn_reg_get_name,
    /* .get_device_count = */ ggml_backend_zendnn_reg_get_device_गिनती,
    /* .get_device       = */ ggml_backend_zendnn_reg_get_device,
    /* .get_proc_address = */ ggml_backend_zendnn_get_proc_address,
};

ggml_backend_reg_t ggml_backend_zendnn_reg(व्योम) {
    अटल काष्ठा ggml_backend_reg ggml_backend_zendnn_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_zendnn_reg_i,
        /* .context     = */ शून्य,
    };

    वापस &ggml_backend_zendnn_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_zendnn_reg)
