#अन्तर्भाव "ggml-zendnn.h"

#अन्तर्भाव "ggml-backend-impl.h"
#अन्तर्भाव "ggml-impl.h"

#define GGML_COMMON_DECL_CPP
#अन्तर्भाव "ggml-common.h"

#अन्तर्भाव "zendnnl.hpp"

#अन्तर्भाव <cstring>


संरचना ggml_backend_zendnn_context {
    पूर्णाङ्क n_threads = GGML_DEFAULT_N_THREADS;
    std::unique_ptr<वर्ण[]> work_data;
    size_t work_size = 0;
};

template<typename T>
zendnnl::common::data_type_t ggml_to_zendnn_type() {
    यदि constexpr (std::is_same_v<T, भिन्न>) {
        प्रत्यागम zendnnl::common::data_type_t::f32;
    } अन्यथा यदि constexpr (std::is_same_v<T, ggml_bf16_t>) {
        प्रत्यागम zendnnl::common::data_type_t::bf16;
    } अन्यथा यदि constexpr (std::is_same_v<T, block_q8_0>) {
        प्रत्यागम zendnnl::common::data_type_t::s8;
    } अन्यथा {
        प्रत्यागम zendnnl::common::data_type_t::none;
    }
}

/**
 * Builds the matmul_params shared by ggml_zendnn_matmul() and ggml_zendnn_group_matmul():
 * dtype selection plus, for Q8_0 weights, dynamic-quant setup. Callers still need to set
 * quant_params.src_scale.dims themselves, since that depends on the batch size(s) in use.
 */
template <typename TA, typename TB, typename TC>
स्थिर zendnnl::lowoha::matmul::matmul_params ggml_zendnn_make_matmul_params(ggml_backend_zendnn_context * ctx) {
    zendnnl::lowoha::matmul::matmul_params params;
    params.dtypes.src = ggml_to_zendnn_type<TB>();
    params.dtypes.wei = ggml_to_zendnn_type<TA>();
    params.dtypes.dst = ggml_to_zendnn_type<TC>();
    params.अंक_threads = ctx->n_threads;

    यदि constexpr (std::is_same_v<TA, block_q8_0>) {
        params.dtypes.compute = zendnnl::common::data_type_t::s8;
        params.dynamic_quant = सत्य;
        params.quant_params.src_scale.buff = nullptr;
        params.quant_params.src_scale.dt   = zendnnl::common::data_type_t::bf16;
        params.packing.pack_format_b = 1;
    }
    प्रत्यागम params;
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
स्थिर तार्किक ggml_zendnn_matmul(ggml_backend_zendnn_context * ctx, int64_t m, int64_t n, int64_t k,
                               ध्रुव TA * A, int64_t lda, ध्रुव TB * B, int64_t ldb, TC * C,
                               int64_t ldc) {

    zendnnl::lowoha::matmul::matmul_params params = ggml_zendnn_make_matmul_params<TA, TB, TC>(ctx);

    zendnnl::lowoha::matmul::matmul_batch_params_t batch_params;

    यदि constexpr (std::is_same_v<TA, block_q8_0>) {
        params.quant_params.src_scale.dims = {n, k / QK8_0};
    }

    zendnnl::error_handling::status_t status = zendnnl::lowoha::matmul::matmul_direct(
        'r', असत्य, सत्य,   // row-major, don't transpose B, transpose A (because it's column-major)
        n,                  // M: rows of B and C
        m,                  // N: cols of A^T and C
        k,                  // K: cols of B, rows of A
        1.0f,               // alpha
        B, ldb,             // src: B[n,k]
        A, lda,             // weight: A[k,m] column-major (transposed)
        nullptr,            // bias
        0.0f,               // beta
        C, ldc,             // output C[n,m]
        सत्य,               // is_weights_const
        batch_params,       // batch_params
        params              // params
    );

    यदि (status != zendnnl::error_handling::status_t::success) {
        GGML_LOG_ERROR("%s, ZenDNN matmul failed: status=%d\n", __func__, static_cast<पूर्णाङ्क>(status));
        प्रत्यागम असत्य;
    }
    प्रत्यागम सत्य;
}

स्थिर तार्किक ggml_zendnn_gemm(ggml_backend_zendnn_context * ctx, int64_t m, int64_t n, int64_t k,
                              ध्रुव शून्य * A, int64_t lda, ध्रुव शून्य * B, int64_t ldb, शून्य * C,
                              int64_t ldc, पूर्णाङ्क Atype, पूर्णाङ्क Btype, पूर्णाङ्क Ctype) {

    प्रतिज्ञा(m >= 0);
    प्रतिज्ञा(n >= 0);
    प्रतिज्ञा(k >= 0);
    प्रतिज्ञा(lda >= k);
    प्रतिज्ञा(ldb >= k);
    प्रतिज्ञा(ldc >= m);

    // categorize types
    विकल्प (Atype) {
        पक्ष GGML_TYPE_F32:
            यदि (Btype != GGML_TYPE_F32 || Ctype != GGML_TYPE_F32)
                प्रत्यागम असत्य;
            प्रत्यागम ggml_zendnn_matmul<भिन्न, भिन्न, भिन्न>(
                ctx, m, n, k,
                (ध्रुव भिन्न *)A, lda,
                (ध्रुव भिन्न *)B, ldb,
                (भिन्न *)C, ldc);
        पक्ष GGML_TYPE_BF16:
            यदि (Btype != GGML_TYPE_BF16)
                प्रत्यागम असत्य;
            यदि (Ctype == GGML_TYPE_BF16)
                प्रत्यागम ggml_zendnn_matmul<ggml_bf16_t, ggml_bf16_t, ggml_bf16_t>(
                    ctx, m, n, k,
                    (ध्रुव ggml_bf16_t *)A, lda,
                    (ध्रुव ggml_bf16_t *)B, ldb,
                    (ggml_bf16_t *)C, ldc);
            यदि (Ctype == GGML_TYPE_F32)
                प्रत्यागम ggml_zendnn_matmul<ggml_bf16_t, ggml_bf16_t, भिन्न>(
                    ctx, m, n, k,
                    (ध्रुव ggml_bf16_t *)A, lda,
                    (ध्रुव ggml_bf16_t *)B, ldb,
                    (भिन्न *)C, ldc);
            प्रत्यागम असत्य;
        पक्ष GGML_TYPE_Q8_0:
            यदि (Btype != GGML_TYPE_F32 || Ctype != GGML_TYPE_F32)
                प्रत्यागम असत्य;
            प्रत्यागम ggml_zendnn_matmul<block_q8_0, भिन्न, भिन्न>(
                ctx, m, n, k,
                (ध्रुव block_q8_0 *)A, lda,
                (ध्रुव भिन्न *)B, ldb,
                (भिन्न *)C, ldc);
        स्वभाव:
            प्रत्यागम असत्य; // unsupported type
    }
}

स्थिर शून्य ggml_zendnn_compute_forward_mul_mat(
    ggml_backend_zendnn_context * ctx,
    ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];  // weights
    ध्रुव ggml_tensor * src1 = dst->src[1];  // inputs

    GGML_TENSOR_द्विमान_OP_LOCALS

    ggml_type         ध्रुव vec_dot_type = src0->type;
    ggml_from_float_t ध्रुव from_float = ggml_get_type_traits(vec_dot_type)->from_float_ref;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == परिमाण(भिन्न));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // broadcast factors
    ध्रुव int64_t r2 = ne12/ne02;
    ध्रुव int64_t r3 = ne13/ne03;

    शून्य * work_data = ctx->work_data.get();

    // ZenDNN requires FP32 for dynamic quantization, so conversion is skipped
    यदि (src1->type != vec_dot_type && src0->type != GGML_TYPE_Q8_0) {
        ध्रुव size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
        ध्रुव size_t nbw2 = nbw1 * ne11;
        ध्रुव size_t nbw3 = nbw2 * ne12;
        ध्रुव size_t desired_wsize = ne13 * nbw3;
        यदि (ctx->work_size < desired_wsize) {
            ctx->work_data.reset(नूतन वर्ण[desired_wsize]);
            ctx->work_size = desired_wsize;
        }
        work_data = ctx->work_data.get();

        // #pragma omp parallel for num_threads(ctx->n_threads)
        #pragma omp parallel प्रति collapse(3) अंक_threads(ctx->n_threads) schedule(स्थिर)
        प्रति (int64_t i13 = 0; i13 < ne13; ++i13) {
            प्रति (int64_t i12 = 0; i12 < ne12; ++i12) {
                प्रति (int64_t i11 = 0; i11 < ne11; ++i11) {
                    ध्रुव भिन्न * src1_f32 = (भिन्न *)((वर्ण *)src1->data + i11*nb11 + i12*nb12 + i13*nb13);
                    शून्य * src1_conv = (वर्ण *)work_data + i11*nbw1 + i12*nbw2 + i13*nbw3;
                    from_float(src1_f32, src1_conv, ne10);
                }
            }
        }
    }

    प्रति (int64_t i13 = 0; i13 < ne13; i13++) {
        प्रति (int64_t i12 = 0; i12 < ne12; i12++) {
            ध्रुव शून्य* wdata = (src1->type == vec_dot_type || src0->type == GGML_TYPE_Q8_0) ? src1->data : work_data;
            ध्रुव size_t row_size = ggml_row_size(vec_dot_type, ne10);
            यदि (!ggml_zendnn_gemm(ctx,
                                  ne01,     // m
                                  ne11,     // n
                                  ne10,     // k
                                  static_cast<ध्रुव वर्ण *>(src0->data) + (i12/r2)*nb02 + (i13/r3)*nb03,
                                  ne00,     // lda
                                  static_cast<ध्रुव वर्ण *>(wdata) + (i12*ne11 + i13*ne12*ne11)*row_size,
                                  ne10,     // ldb
                                  static_cast<वर्ण *>(dst->data) + i12*nb2 + i13*nb3,
                                  ne01,     // ldc
                                  src0->type,
                                  src0->type == GGML_TYPE_Q8_0 ? GGML_TYPE_F32 : vec_dot_type,
                                  dst->type))
                GGML_ABORT("%s: ZenDNN gemm failed\n", __func__);
        }
    }
}

संरचना mmid_row_mapping {
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
स्थिर तार्किक ggml_zendnn_group_matmul(ggml_backend_zendnn_context * ctx, int64_t m, int64_t k,
                                     ध्रुव std::vector<int64_t> & n,
                                     ध्रुव std::vector<ध्रुव शून्य *> & A, int64_t lda,
                                     ध्रुव std::vector<ध्रुव शून्य *> & B, int64_t ldb,
                                     ध्रुव std::vector<शून्य *> & C, int64_t ldc) {

    ध्रुव पूर्णाङ्क n_experts = n.size();

    zendnnl::lowoha::matmul::matmul_params base_params = ggml_zendnn_make_matmul_params<TA, TB, TC>(ctx);

    std::vector<वर्ण>         layout(n_experts, 'r');
    std::vector<तार्किक>         trans_a(n_experts, असत्य);
    std::vector<तार्किक>         trans_b(n_experts, सत्य);
    std::vector<पूर्णाङ्क>          batch_m(n_experts);
    std::vector<पूर्णाङ्क>          batch_n(n_experts, m);
    std::vector<पूर्णाङ्क>          batch_k(n_experts, k);
    std::vector<भिन्न>        alpha(n_experts, 1.0f);
    std::vector<भिन्न>        beta(n_experts, 0.0f);
    std::vector<ध्रुव शून्य *> bias(n_experts, nullptr);
    std::vector<पूर्णाङ्क>          lda_v(n_experts, lda);
    std::vector<पूर्णाङ्क>          ldb_v(n_experts, ldb);
    std::vector<पूर्णाङ्क>          ldc_v(n_experts, ldc);
    std::vector<तार्किक>         is_wei_const(n_experts, सत्य);
    std::vector<zendnnl::lowoha::matmul::matmul_params> params(n_experts, base_params);

    प्रति (पूर्णाङ्क i = 0; i < n_experts; i++) {
        batch_m[i] = n[i];

        // src_scale.dims depends on this expert's row count, unlike the rest of base_params
        यदि constexpr (std::is_same_v<TA, block_q8_0>) {
            params[i].quant_params.src_scale.dims = {n[i], k / QK8_0};
        }
    }

    zendnnl::error_handling::status_t status = zendnnl::lowoha::matmul::group_matmul_direct(
        layout, trans_a, trans_b, batch_m, batch_n, batch_k, alpha,
        B, ldb_v, A, lda_v, bias, beta,
        C, ldc_v, is_wei_const, params);

    यदि (status != zendnnl::error_handling::status_t::success) {
        GGML_LOG_ERROR("%s, ZenDNN group matmul failed: status=%d\n", __func__, static_cast<पूर्णाङ्क>(status));
        प्रत्यागम असत्य;
    }
    प्रत्यागम सत्य;
}

स्थिर तार्किक ggml_zendnn_group_gemm(ggml_backend_zendnn_context * ctx, int64_t m, int64_t k,
                                    ध्रुव std::vector<int64_t> & n,
                                    ध्रुव std::vector<ध्रुव शून्य *> & A, int64_t lda,
                                    ध्रुव std::vector<ध्रुव शून्य *> & B, int64_t ldb,
                                    ध्रुव std::vector<शून्य *> & C, int64_t ldc,
                                    पूर्णाङ्क Atype, पूर्णाङ्क Btype, पूर्णाङ्क Ctype) {

    प्रतिज्ञा(m >= 0);
    प्रति (size_t i = 0; i < n.size(); i++) {
        प्रतिज्ञा(n[i] >= 0);
    }
    प्रतिज्ञा(k >= 0);
    प्रतिज्ञा(lda >= k);
    प्रतिज्ञा(ldb >= k);
    प्रतिज्ञा(ldc >= m);

    // categorize types
    विकल्प (Atype) {
        पक्ष GGML_TYPE_F32:
            यदि (Btype != GGML_TYPE_F32 || Ctype != GGML_TYPE_F32)
                प्रत्यागम असत्य;
            प्रत्यागम ggml_zendnn_group_matmul<भिन्न, भिन्न, भिन्न>(ctx, m, k, n, A, lda, B, ldb, C, ldc);
        पक्ष GGML_TYPE_BF16:
            यदि (Btype != GGML_TYPE_BF16)
                प्रत्यागम असत्य;
            यदि (Ctype == GGML_TYPE_BF16)
                प्रत्यागम ggml_zendnn_group_matmul<ggml_bf16_t, ggml_bf16_t, ggml_bf16_t>(
                    ctx, m, k, n, A, lda, B, ldb, C, ldc);
            यदि (Ctype == GGML_TYPE_F32)
                प्रत्यागम ggml_zendnn_group_matmul<ggml_bf16_t, ggml_bf16_t, भिन्न>(ctx, m, k, n, A, lda, B, ldb, C, ldc);
            प्रत्यागम असत्य;
        पक्ष GGML_TYPE_Q8_0:
            यदि (Btype != GGML_TYPE_F32 || Ctype != GGML_TYPE_F32)
                प्रत्यागम असत्य;
            प्रत्यागम ggml_zendnn_group_matmul<block_q8_0, भिन्न, भिन्न>(ctx, m, k, n, A, lda, B, ldb, C, ldc);
        स्वभाव:
            प्रत्यागम असत्य; // unsupported type
    }
}

स्थिर शून्य ggml_zendnn_compute_forward_mul_mat_id(
    ggml_backend_zendnn_context * ctx,
    ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];  // expert weights
    ध्रुव ggml_tensor * src1 = dst->src[1];  // inputs
    ध्रुव ggml_tensor * ids  = dst->src[2];  // expert ids

    GGML_TENSOR_द्विमान_OP_LOCALS

    // exit for no tokens to process
    यदि (ne2 == 0 || ne11 == 0) {
        प्रत्यागम;
    }

    ggml_type         ध्रुव vec_dot_type = src0->type;
    ggml_from_float_t ध्रुव from_float = ggml_get_type_traits(vec_dot_type)->from_float_ref;

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == परिमाण(भिन्न));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(ne03 == 1);
    GGML_ASSERT(ne13 == 1);
    GGML_ASSERT(ne3  == 1);

    // row groups
    ध्रुव पूर्णाङ्क n_ids = ids->ne[0]; // n_expert_used
    ध्रुव पूर्णाङ्क n_as  = ne02;       // n_experts

    std::vector<int64_t> matrix_row_counts(n_as, 0);
    std::vector<std::vector<mmid_row_mapping>> matrix_rows(n_as);

    int64_t total_rows = 0;
    पूर्णाङ्क     n_active_experts = 0;
    // group rows by expert (preprocessing step)
    प्रति (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
        प्रति (पूर्णाङ्क id = 0; id < n_ids; ++id) {
            ध्रुव int32_t i02 = *(ध्रुव int32_t *)((ध्रुव वर्ण *)ids->data + iid1*ids->nb[1] + id*ids->nb[0]);

            GGML_ASSERT(i02 >= 0 && i02 < n_as);

            यदि (matrix_row_counts[i02] == 0) {
                n_active_experts++;
            }
            matrix_rows[i02].push_back({id, iid1});
            matrix_row_counts[i02]++;
            total_rows++;
        }
    }

    यदि (total_rows == 0) {
        प्रत्यागम; // no rows to process
    }

    ध्रुव size_t row_size = ggml_row_size(vec_dot_type, ne10);

    // For Q8_0, src1 is always F32; the gather buffer must hold F32 rows (ne10*4 bytes),
    // not Q8_0-encoded rows (row_size ≈ ne10/32*34 bytes) — they differ by ~4x.
    ध्रुव size_t f32_row_size = (size_t)ne10 * परिमाण(भिन्न);
    ध्रुव size_t gather_row_size = (src0->type == GGML_TYPE_Q8_0) ? f32_row_size : row_size;

    यदि (src1->type != vec_dot_type && src0->type != GGML_TYPE_Q8_0) {
        GGML_ASSERT(src1->type == GGML_TYPE_F32);
    }

    // size for MoE gather/scatter buffers
    ध्रुव size_t wdata_cur_size = total_rows * gather_row_size;
    ध्रुव size_t dst_cur_size = total_rows * ggml_row_size(dst->type, ne01);

    // allocate single buffer for all needs
    ध्रुव size_t total_size = wdata_cur_size + dst_cur_size;
    यदि (ctx->work_size < total_size) {
        ctx->work_data.reset(नूतन वर्ण[total_size]);
        ctx->work_size = total_size;
    }

    // partition the buffer
    वर्ण * wdata_cur = ctx->work_data.get();
    वर्ण * dst_cur = wdata_cur + wdata_cur_size;

    // per-expert data collected during gather, handed to ggml_zendnn_group_gemm() as one batch
    std::vector<int64_t>      expert_row_मोजणी(n_active_experts);
    std::vector<ध्रुव शून्य *> batch_src(n_active_experts);
    std::vector<ध्रुव शून्य *> batch_wei(n_active_experts);
    std::vector<शून्य *>       batch_dst(n_active_experts);

    // precompute per-expert buffer offsets and batch indices for the parallel loop below
    std::vector<int64_t> expert_wdata_off(n_as, 0);
    std::vector<int64_t> expert_dst_off(n_as, 0);
    std::vector<पूर्णाङ्क>     expert_batch_idx(n_as, -1);
    {
        int64_t w_off = 0;
        int64_t d_off = 0;
        पूर्णाङ्क     batch_idx = 0;
        प्रति (int64_t cur_a = 0; cur_a < n_as; ++cur_a) {
            यदि (matrix_row_counts[cur_a] == 0) {
                अनुवर्त;
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
    #pragma omp parallel प्रति अंक_threads(ctx->n_threads) schedule(स्थिर)
    प्रति (int64_t cur_a = 0; cur_a < n_as; ++cur_a) {
        ध्रुव int64_t cne1 = matrix_row_counts[cur_a];

        यदि (cne1 == 0) {
            अनुवर्त;
        }

        ध्रुव int64_t w_off     = expert_wdata_off[cur_a];
        ध्रुव int64_t d_off     = expert_dst_off[cur_a];
        ध्रुव पूर्णाङ्क     batch_idx = expert_batch_idx[cur_a];

        प्रति (int64_t ir1 = 0; ir1 < cne1; ++ir1) {
            ध्रुव mmid_row_mapping & row_mapping = matrix_rows[cur_a][ir1];
            ध्रुव int64_t id  = row_mapping.i1;
            ध्रुव int64_t i11 = id % ne11;
            ध्रुव int64_t i12 = row_mapping.i2;

            ध्रुव वर्ण * src_row = (ध्रुव वर्ण *) src1->data + i11*nb11 + i12*nb12;
            शून्य *       dst_row = wdata_cur + w_off + ir1 * gather_row_size;

            यदि (src1->type != vec_dot_type && src0->type != GGML_TYPE_Q8_0) {
                from_float((ध्रुव भिन्न *) src_row, dst_row, ne10);
            } अन्यथा {
                // no conversion: src1 already matches vec_dot_type, or src0 is Q8_0, whose
                // ZenDNN dynamic quantization requires the row to stay in F32
                std::memcpy(dst_row, src_row, gather_row_size);
            }
        }

        expert_row_मोजणी[batch_idx] = cne1;
        batch_src[batch_idx]        = wdata_cur + w_off;
        batch_wei[batch_idx]        = (ध्रुव वर्ण *) src0->data + cur_a * nb02;
        batch_dst[batch_idx]        = dst_cur + d_off;
    }

    यदि (!ggml_zendnn_group_gemm(ctx,
                                ne01,             // m
                                ne10,             // k
                                expert_row_मोजणी, // n (per expert)
                                batch_wei, ne00,  // A: weights, lda
                                batch_src, ne10,  // B: input, ldb
                                batch_dst, ne01,  // C: output, ldc
                                src0->type,
                                src0->type == GGML_TYPE_Q8_0 ? GGML_TYPE_F32 : vec_dot_type,
                                dst->type))
        GGML_ABORT("%s: ZenDNN group gemm failed\n", __func__);

    // scatter output rows to destination
    #pragma omp parallel प्रति अंक_threads(ctx->n_threads) schedule(स्थिर)
    प्रति (int64_t cur_a = 0; cur_a < n_as; ++cur_a) {
        ध्रुव int64_t cne1 = matrix_row_counts[cur_a];

        यदि (cne1 == 0) {
            अनुवर्त;
        }

        ध्रुव int64_t d_off = expert_dst_off[cur_a];

        प्रति (int64_t ir1 = 0; ir1 < cne1; ++ir1) {
            ध्रुव mmid_row_mapping & row_mapping = matrix_rows[cur_a][ir1];
            ध्रुव int64_t id = row_mapping.i1;
            ध्रुव int64_t i1 = id;
            ध्रुव int64_t i2 = row_mapping.i2;

            std::memcpy(
                (वर्ण *) dst->data + i1*nb1 + i2*nb2,
                dst_cur + d_off + ir1 * ggml_row_size(dst->type, ne01),
                ggml_row_size(dst->type, ne01)
            );
        }
    }
}

// backend interface

स्थिर ध्रुव वर्ण * ggml_backend_zendnn_get_name(ggml_backend_t backend) {
    प्रत्यागम "ZenDNN";

    GGML_UNUSED(backend);
}

स्थिर शून्य ggml_backend_zendnn_free(ggml_backend_t backend) {
    ggml_backend_zendnn_context * ctx = (ggml_backend_zendnn_context *)backend->context;
    delete ctx;
    delete backend;
}

स्थिर ggml_status ggml_backend_zendnn_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_zendnn_context * ctx = (ggml_backend_zendnn_context *)backend->context;

    प्रति (पूर्णाङ्क i = 0; i < cgraph->n_nodes; i++) {
        संरचना ggml_tensor * गाठ = cgraph->nodes[i];

        यदि ((गाठ->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            अनुवर्त;
        }

        विकल्प (गाठ->op) {
            पक्ष GGML_OP_MUL_MAT:
                ggml_zendnn_compute_forward_mul_mat(ctx, गाठ);
                भङ्ग;
            पक्ष GGML_OP_MUL_MAT_ID:
                ggml_zendnn_compute_forward_mul_mat_id(ctx, गाठ);
                भङ्ग;
            पक्ष GGML_OP_NONE:
            पक्ष GGML_OP_RESHAPE:
            पक्ष GGML_OP_VIEW:
            पक्ष GGML_OP_PERMUTE:
            पक्ष GGML_OP_TRANSPOSE:
                भङ्ग;

            स्वभाव:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(गाठ));
        }
    }

    प्रत्यागम GGML_STATUS_SUCCESS;

    GGML_UNUSED(backend);
}

स्थिर संरचना ggml_backend_i ggml_backend_zendnn_i = {
    /* .get_name                = */ ggml_backend_zendnn_get_name,
    /* .free                    = */ ggml_backend_zendnn_free,
    /* .set_tensor_async        = */ रिक्त,
    /* .get_tensor_async        = */ रिक्त,
    /* .set_tensor_2d_async     = */ रिक्त,
    /* .get_tensor_2d_async     = */ रिक्त,
    /* .cpy_tensor_async        = */ रिक्त,
    /* .synchronize             = */ रिक्त,
    /* .graph_plan_create       = */ रिक्त,
    /* .graph_plan_free         = */ रिक्त,
    /* .graph_plan_update       = */ रिक्त,
    /* .graph_plan_compute      = */ रिक्त,
    /* .graph_compute           = */ ggml_backend_zendnn_graph_compute,
    /* .event_record            = */ रिक्त,
    /* .event_wait              = */ रिक्त,
    /* .graph_optimize          = */ रिक्त,
};

स्थिर ggml_guid_t ggml_backend_zendnn_guid(शून्य) {
    स्थिर ध्रुव वर्ण * guid_str = "AMD-ZENDNN-ACCEL";
    प्रत्यागम reinterpret_cast<ggml_guid_t>(const_cast<वर्ण*>(guid_str));
}

ggml_backend_t ggml_backend_zendnn_init(शून्य) {
    ggml_backend_zendnn_context * ctx = नूतन ggml_backend_zendnn_context;

    ggml_backend_t backend = नूतन ggml_backend {
        /* .guid    = */ ggml_backend_zendnn_guid(),
        /* .iface   = */ ggml_backend_zendnn_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_zendnn_reg(), 0),
        /* .context = */ ctx,
    };

    प्रत्यागम backend;
}

तार्किक ggml_backend_is_zendnn(ggml_backend_t backend) {
    प्रत्यागम backend != रिक्त && ggml_guid_matches(backend->guid, ggml_backend_zendnn_guid());
}

शून्य ggml_backend_zendnn_set_n_threads(ggml_backend_t backend_zendnn, पूर्णाङ्क n_threads) {
    GGML_ASSERT(ggml_backend_is_zendnn(backend_zendnn));

    ggml_backend_zendnn_context * ctx = (ggml_backend_zendnn_context *)backend_zendnn->context;
    ctx->n_threads = n_threads;
}

// device interface
स्थिर ध्रुव वर्ण * ggml_backend_zendnn_device_get_name(ggml_backend_dev_t dev) {
    प्रत्यागम "ZenDNN";

    GGML_UNUSED(dev);
}
/**
 * ZenDNN is AMD's performance library providing optimized primitives and implementations
 * for deep learning workloads on AMD CPUs. It targets improved performance for common
 * neural network operations on AMD architectures. For more information, see:
 * https://www.amd.com/en/developer/zendnn.html
 */
स्थिर ध्रुव वर्ण * ggml_backend_zendnn_device_get_description(ggml_backend_dev_t dev) {
    प्रत्यागम "ZenDNN: AMD optimized primitives backend for GGML (optimized for AMD CPUs)";

    GGML_UNUSED(dev);
}

स्थिर शून्य ggml_backend_zendnn_device_get_memory(ggml_backend_dev_t dev, size_t * मुक्त, size_t * total) {
    *मुक्त  = 0;
    *total = 0;

    GGML_UNUSED(dev);
}

स्थिर गणना ggml_backend_dev_type ggml_backend_zendnn_device_get_type(ggml_backend_dev_t dev) {
    प्रत्यागम GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_UNUSED(dev);
}

स्थिर शून्य ggml_backend_zendnn_device_get_props(ggml_backend_dev_t dev, संरचना ggml_backend_dev_props * props) {
    props->name        = ggml_backend_zendnn_device_get_name(dev);
    props->description = ggml_backend_zendnn_device_get_description(dev);
    props->type        = ggml_backend_zendnn_device_get_type(dev);
    ggml_backend_zendnn_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                = */ असत्य,
        /* .host_buffer          = */ असत्य,
        /* .buffer_from_host_ptr = */ सत्य,
        /* .events               = */ असत्य,
        /* .mmap_support         = */ सत्य,
    };
}

स्थिर ggml_backend_t ggml_backend_zendnn_device_init_backend(ggml_backend_dev_t dev, ध्रुव वर्ण * params) {
    ggml_backend_t backend = ggml_backend_zendnn_init();
    यदि (backend == रिक्त) {
        GGML_LOG_ERROR("%s: error: failed to initialize ZenDNN backend\n", __func__);
        प्रत्यागम रिक्त;
    }

    प्रत्यागम backend;

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

स्थिर ggml_backend_buffer_type_t ggml_backend_zendnn_device_get_buffer_type(ggml_backend_dev_t dev) {
    प्रत्यागम ggml_backend_cpu_buffer_type();

    GGML_UNUSED(dev);
}

स्थिर ggml_backend_buffer_t ggml_backend_zendnn_device_buffer_from_host_ptr(ggml_backend_dev_t dev, शून्य * ptr, size_t size, size_t max_tensor_size) {
    प्रत्यागम ggml_backend_cpu_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

स्थिर तार्किक ggml_zendnn_adaptive_fallback_enabled() {
    स्थिर ध्रुव तार्किक enabled = std::getenv("GGML_ZENDNN_ADAPTIVE_FALLBACK") == nullptr ||
                                std::atoi(std::getenv("GGML_ZENDNN_ADAPTIVE_FALLBACK")) != 0;
    प्रत्यागम enabled;
}

स्थिर तार्किक ggml_backend_zendnn_device_supports_op(ggml_backend_dev_t dev, ध्रुव संरचना ggml_tensor * op) {
    विकल्प (op->op) {
        पक्ष GGML_OP_NONE:
        पक्ष GGML_OP_RESHAPE:
        पक्ष GGML_OP_VIEW:
        पक्ष GGML_OP_PERMUTE:
        पक्ष GGML_OP_TRANSPOSE:
            प्रत्यागम सत्य;

        पक्ष GGML_OP_MUL_MAT:
        पक्ष GGML_OP_MUL_MAT_ID:
        {
            ध्रुव ggml_tensor * weights = op->src[0];
            ध्रुव ggml_tensor * inputs = op->src[1];

            ध्रुव int64_t ne10 = inputs->ne[0];
            ध्रुव int64_t ne0 = op->ne[0];
            ध्रुव int64_t ne1 = op->ne[1];
            ध्रुव int64_t min_batch = 1;

            यदि(!ggml_is_contiguous(weights) || !ggml_is_contiguous(inputs)) {
                प्रत्यागम असत्य;
            }

            यदि (ggml_zendnn_adaptive_fallback_enabled()) {
                ध्रुव int64_t K = inputs->ne[0];
                ध्रुव int64_t N = (inputs->ne[1]*inputs->ne[2]*inputs->ne[3]);
                ध्रुव int64_t M = weights->ne[1];
                यदि(K <= 256 || N <= 128 || M <= 96) {
                    प्रत्यागम असत्य;
                }

                // MUL_MAT_ID's gather+matmul+scatter approach favors a moderate expert count
                यदि (op->op == GGML_OP_MUL_MAT_ID) {
                    ध्रुव int64_t n_experts = weights->ne[2];
                    ध्रुव int64_t max_experts = 32;
                    यदि (n_experts > max_experts) {
                        प्रत्यागम असत्य;
                    }

                    // fall back once the average rows per expert (N / n_experts) is too thin
                    // to amortize each per-expert GEMM's overhead
                    यदि (N / n_experts <= 32) {
                        प्रत्यागम असत्य;
                    }
                }
            }
            अन्यथा यदि (ne0 < min_batch || ne1 < min_batch || ne10 < min_batch) {
                प्रत्यागम असत्य;
            }

            विकल्प (weights->type) {
                पक्ष GGML_TYPE_F32:
                पक्ष GGML_TYPE_BF16:
                पक्ष GGML_TYPE_Q8_0:
                    प्रत्यागम सत्य;
                स्वभाव:
                    प्रत्यागम असत्य;
            }
        } भङ्ग;

        स्वभाव:
            प्रत्यागम असत्य;
    }

    GGML_UNUSED(dev);
}

स्थिर तार्किक ggml_backend_zendnn_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    प्रत्यागम ggml_backend_buft_is_host(buft);

    GGML_UNUSED(dev);
}

स्थिर ध्रुव संरचना ggml_backend_device_i ggml_backend_zendnn_device_i = {
    /* .get_name               = */ ggml_backend_zendnn_device_get_name,
    /* .get_description        = */ ggml_backend_zendnn_device_get_description,
    /* .get_memory             = */ ggml_backend_zendnn_device_get_memory,
    /* .get_type               = */ ggml_backend_zendnn_device_get_type,
    /* .get_props              = */ ggml_backend_zendnn_device_get_props,
    /* .init_backend           = */ ggml_backend_zendnn_device_init_backend,
    /* .get_buffer_type        = */ ggml_backend_zendnn_device_get_buffer_type,
    /* .get_host_buffer_type   = */ रिक्त,
    /* .buffer_from_host_ptr   = */ ggml_backend_zendnn_device_buffer_from_host_ptr,
    /* .supports_op            = */ ggml_backend_zendnn_device_supports_op,
    /* .supports_buft          = */ ggml_backend_zendnn_device_supports_buft,
    /* .offload_op             = */ रिक्त,
    /* .event_new              = */ रिक्त,
    /* .event_free             = */ रिक्त,
    /* .event_synchronize      = */ रिक्त,
};

// backend reg interface
स्थिर ध्रुव वर्ण * ggml_backend_zendnn_reg_get_name(ggml_backend_reg_t reg) {
    प्रत्यागम "ZenDNN";

    GGML_UNUSED(reg);
}

स्थिर size_t ggml_backend_zendnn_reg_get_device_मोजणी(ggml_backend_reg_t reg) {
    प्रत्यागम 1;

    GGML_UNUSED(reg);
}

स्थिर ggml_backend_dev_t ggml_backend_zendnn_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    स्थिर ggml_backend_device ggml_backend_zendnn_device = {
        /* .iface   = */ ggml_backend_zendnn_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };

    प्रत्यागम &ggml_backend_zendnn_device;
}

स्थिर शून्य * ggml_backend_zendnn_get_proc_address(ggml_backend_reg_t reg, ध्रुव वर्ण * name) {
    यदि (std::strcmp(name, "ggml_backend_set_n_threads") == 0) {
        प्रत्यागम (शून्य *) ggml_backend_zendnn_set_n_threads;
    }
    प्रत्यागम रिक्त;

    GGML_UNUSED(reg);
    GGML_UNUSED(name);
}

स्थिर ध्रुव संरचना ggml_backend_reg_i ggml_backend_zendnn_reg_i = {
    /* .get_name         = */ ggml_backend_zendnn_reg_get_name,
    /* .get_device_count = */ ggml_backend_zendnn_reg_get_device_मोजणी,
    /* .get_device       = */ ggml_backend_zendnn_reg_get_device,
    /* .get_proc_address = */ ggml_backend_zendnn_get_proc_address,
};

ggml_backend_reg_t ggml_backend_zendnn_reg(शून्य) {
    स्थिर संरचना ggml_backend_reg ggml_backend_zendnn_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_zendnn_reg_i,
        /* .context     = */ रिक्त,
    };

    प्रत्यागम &ggml_backend_zendnn_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_zendnn_reg)
