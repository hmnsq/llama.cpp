#অন্তর্ভাৱ "ggml-zendnn.h"

#অন্তর্ভাৱ "ggml-backend-impl.h"
#অন্তর্ভাৱ "ggml-impl.h"

#define GGML_COMMON_DECL_CPP
#অন্তর্ভাৱ "ggml-common.h"

#অন্তর্ভাৱ "zendnnl.hpp"

#অন্তর্ভাৱ <cstring>


সংরচনা ggml_backend_zendnn_context {
    পূর্ণাঙ্ক n_threads = GGML_DEFAULT_N_THREADS;
    std::unique_ptr<ৱর্ণ[]> work_data;
    size_t work_size = 0;
};

template<typename T>
zendnnl::common::data_type_t ggml_to_zendnn_type() {
    যদি constexpr (std::is_same_v<T, ভিন্ন>) {
        ঘূৰাই zendnnl::common::data_type_t::f32;
    } অন্যথা যদি constexpr (std::is_same_v<T, ggml_bf16_t>) {
        ঘূৰাই zendnnl::common::data_type_t::bf16;
    } অন্যথা যদি constexpr (std::is_same_v<T, block_q8_0>) {
        ঘূৰাই zendnnl::common::data_type_t::s8;
    } অন্যথা {
        ঘূৰাই zendnnl::common::data_type_t::none;
    }
}

/**
 * Builds the matmul_params shared by ggml_zendnn_matmul() and ggml_zendnn_group_matmul():
 * dtype selection plus, for Q8_0 weights, dynamic-quant setup. Callers still need to set
 * quant_params.src_scale.dims themselves, since that depends on the batch size(s) in use.
 */
template <typename TA, typename TB, typename TC>
স্থির zendnnl::lowoha::matmul::matmul_params ggml_zendnn_make_matmul_params(ggml_backend_zendnn_context * ctx) {
    zendnnl::lowoha::matmul::matmul_params params;
    params.dtypes.src = ggml_to_zendnn_type<TB>();
    params.dtypes.wei = ggml_to_zendnn_type<TA>();
    params.dtypes.dst = ggml_to_zendnn_type<TC>();
    params.অঙ্ক_threads = ctx->n_threads;

    যদি constexpr (std::is_same_v<TA, block_q8_0>) {
        params.dtypes.compute = zendnnl::common::data_type_t::s8;
        params.dynamic_quant = সঁচা;
        params.quant_params.src_scale.buff = nullptr;
        params.quant_params.src_scale.dt   = zendnnl::common::data_type_t::bf16;
        params.packing.pack_format_b = 1;
    }
    ঘূৰাই params;
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
স্থির তার্কিক ggml_zendnn_matmul(ggml_backend_zendnn_context * ctx, int64_t m, int64_t n, int64_t k,
                               ধ্রুৱ TA * A, int64_t lda, ধ্রুৱ TB * B, int64_t ldb, TC * C,
                               int64_t ldc) {

    zendnnl::lowoha::matmul::matmul_params params = ggml_zendnn_make_matmul_params<TA, TB, TC>(ctx);

    zendnnl::lowoha::matmul::matmul_batch_params_t batch_params;

    যদি constexpr (std::is_same_v<TA, block_q8_0>) {
        params.quant_params.src_scale.dims = {n, k / QK8_0};
    }

    zendnnl::error_handling::status_t status = zendnnl::lowoha::matmul::matmul_direct(
        'r', মিছা, সঁচা,   // row-major, don't transpose B, transpose A (because it's column-major)
        n,                  // M: rows of B and C
        m,                  // N: cols of A^T and C
        k,                  // K: cols of B, rows of A
        1.0f,               // alpha
        B, ldb,             // src: B[n,k]
        A, lda,             // weight: A[k,m] column-major (transposed)
        nullptr,            // bias
        0.0f,               // beta
        C, ldc,             // output C[n,m]
        সঁচা,               // is_weights_const
        batch_params,       // batch_params
        params              // params
    );

    যদি (status != zendnnl::error_handling::status_t::success) {
        GGML_LOG_ERROR("%s, ZenDNN matmul failed: status=%d\n", __func__, static_cast<পূর্ণাঙ্ক>(status));
        ঘূৰাই মিছা;
    }
    ঘূৰাই সঁচা;
}

স্থির তার্কিক ggml_zendnn_gemm(ggml_backend_zendnn_context * ctx, int64_t m, int64_t n, int64_t k,
                              ধ্রুৱ শূন্য * A, int64_t lda, ধ্রুৱ শূন্য * B, int64_t ldb, শূন্য * C,
                              int64_t ldc, পূর্ণাঙ্ক Atype, পূর্ণাঙ্ক Btype, পূর্ণাঙ্ক Ctype) {

    প্রতিজ্ঞা(m >= 0);
    প্রতিজ্ঞা(n >= 0);
    প্রতিজ্ঞা(k >= 0);
    প্রতিজ্ঞা(lda >= k);
    প্রতিজ্ঞা(ldb >= k);
    প্রতিজ্ঞা(ldc >= m);

    // categorize types
    ৱিকল্প (Atype) {
        পক্ষ GGML_TYPE_F32:
            যদি (Btype != GGML_TYPE_F32 || Ctype != GGML_TYPE_F32)
                ঘূৰাই মিছা;
            ঘূৰাই ggml_zendnn_matmul<ভিন্ন, ভিন্ন, ভিন্ন>(
                ctx, m, n, k,
                (ধ্রুৱ ভিন্ন *)A, lda,
                (ধ্রুৱ ভিন্ন *)B, ldb,
                (ভিন্ন *)C, ldc);
        পক্ষ GGML_TYPE_BF16:
            যদি (Btype != GGML_TYPE_BF16)
                ঘূৰাই মিছা;
            যদি (Ctype == GGML_TYPE_BF16)
                ঘূৰাই ggml_zendnn_matmul<ggml_bf16_t, ggml_bf16_t, ggml_bf16_t>(
                    ctx, m, n, k,
                    (ধ্রুৱ ggml_bf16_t *)A, lda,
                    (ধ্রুৱ ggml_bf16_t *)B, ldb,
                    (ggml_bf16_t *)C, ldc);
            যদি (Ctype == GGML_TYPE_F32)
                ঘূৰাই ggml_zendnn_matmul<ggml_bf16_t, ggml_bf16_t, ভিন্ন>(
                    ctx, m, n, k,
                    (ধ্রুৱ ggml_bf16_t *)A, lda,
                    (ধ্রুৱ ggml_bf16_t *)B, ldb,
                    (ভিন্ন *)C, ldc);
            ঘূৰাই মিছা;
        পক্ষ GGML_TYPE_Q8_0:
            যদি (Btype != GGML_TYPE_F32 || Ctype != GGML_TYPE_F32)
                ঘূৰাই মিছা;
            ঘূৰাই ggml_zendnn_matmul<block_q8_0, ভিন্ন, ভিন্ন>(
                ctx, m, n, k,
                (ধ্রুৱ block_q8_0 *)A, lda,
                (ধ্রুৱ ভিন্ন *)B, ldb,
                (ভিন্ন *)C, ldc);
        স্ৱভাৱ:
            ঘূৰাই মিছা; // unsupported type
    }
}

স্থির শূন্য ggml_zendnn_compute_forward_mul_mat(
    ggml_backend_zendnn_context * ctx,
    ggml_tensor * dst) {

    ধ্রুৱ ggml_tensor * src0 = dst->src[0];  // weights
    ধ্রুৱ ggml_tensor * src1 = dst->src[1];  // inputs

    GGML_TENSOR_দ্বিমিক_OP_LOCALS

    ggml_type         ধ্রুৱ vec_dot_type = src0->type;
    ggml_from_float_t ধ্রুৱ from_float = ggml_get_type_traits(vec_dot_type)->from_float_ref;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == পরিমাণ(ভিন্ন));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // broadcast factors
    ধ্রুৱ int64_t r2 = ne12/ne02;
    ধ্রুৱ int64_t r3 = ne13/ne03;

    শূন্য * work_data = ctx->work_data.get();

    // ZenDNN requires FP32 for dynamic quantization, so conversion is skipped
    যদি (src1->type != vec_dot_type && src0->type != GGML_TYPE_Q8_0) {
        ধ্রুৱ size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
        ধ্রুৱ size_t nbw2 = nbw1 * ne11;
        ধ্রুৱ size_t nbw3 = nbw2 * ne12;
        ধ্রুৱ size_t desired_wsize = ne13 * nbw3;
        যদি (ctx->work_size < desired_wsize) {
            ctx->work_data.reset(নূতন ৱর্ণ[desired_wsize]);
            ctx->work_size = desired_wsize;
        }
        work_data = ctx->work_data.get();

        // #pragma omp parallel for num_threads(ctx->n_threads)
        #pragma omp parallel বাবে collapse(3) অঙ্ক_threads(ctx->n_threads) schedule(স্থির)
        বাবে (int64_t i13 = 0; i13 < ne13; ++i13) {
            বাবে (int64_t i12 = 0; i12 < ne12; ++i12) {
                বাবে (int64_t i11 = 0; i11 < ne11; ++i11) {
                    ধ্রুৱ ভিন্ন * src1_f32 = (ভিন্ন *)((ৱর্ণ *)src1->data + i11*nb11 + i12*nb12 + i13*nb13);
                    শূন্য * src1_conv = (ৱর্ণ *)work_data + i11*nbw1 + i12*nbw2 + i13*nbw3;
                    from_float(src1_f32, src1_conv, ne10);
                }
            }
        }
    }

    বাবে (int64_t i13 = 0; i13 < ne13; i13++) {
        বাবে (int64_t i12 = 0; i12 < ne12; i12++) {
            ধ্রুৱ শূন্য* wdata = (src1->type == vec_dot_type || src0->type == GGML_TYPE_Q8_0) ? src1->data : work_data;
            ধ্রুৱ size_t row_size = ggml_row_size(vec_dot_type, ne10);
            যদি (!ggml_zendnn_gemm(ctx,
                                  ne01,     // m
                                  ne11,     // n
                                  ne10,     // k
                                  static_cast<ধ্রুৱ ৱর্ণ *>(src0->data) + (i12/r2)*nb02 + (i13/r3)*nb03,
                                  ne00,     // lda
                                  static_cast<ধ্রুৱ ৱর্ণ *>(wdata) + (i12*ne11 + i13*ne12*ne11)*row_size,
                                  ne10,     // ldb
                                  static_cast<ৱর্ণ *>(dst->data) + i12*nb2 + i13*nb3,
                                  ne01,     // ldc
                                  src0->type,
                                  src0->type == GGML_TYPE_Q8_0 ? GGML_TYPE_F32 : vec_dot_type,
                                  dst->type))
                GGML_ABORT("%s: ZenDNN gemm failed\n", __func__);
        }
    }
}

সংরচনা mmid_row_mapping {
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
স্থির তার্কিক ggml_zendnn_group_matmul(ggml_backend_zendnn_context * ctx, int64_t m, int64_t k,
                                     ধ্রুৱ std::vector<int64_t> & n,
                                     ধ্রুৱ std::vector<ধ্রুৱ শূন্য *> & A, int64_t lda,
                                     ধ্রুৱ std::vector<ধ্রুৱ শূন্য *> & B, int64_t ldb,
                                     ধ্রুৱ std::vector<শূন্য *> & C, int64_t ldc) {

    ধ্রুৱ পূর্ণাঙ্ক n_experts = n.size();

    zendnnl::lowoha::matmul::matmul_params base_params = ggml_zendnn_make_matmul_params<TA, TB, TC>(ctx);

    std::vector<ৱর্ণ>         layout(n_experts, 'r');
    std::vector<তার্কিক>         trans_a(n_experts, মিছা);
    std::vector<তার্কিক>         trans_b(n_experts, সঁচা);
    std::vector<পূর্ণাঙ্ক>          batch_m(n_experts);
    std::vector<পূর্ণাঙ্ক>          batch_n(n_experts, m);
    std::vector<পূর্ণাঙ্ক>          batch_k(n_experts, k);
    std::vector<ভিন্ন>        alpha(n_experts, 1.0f);
    std::vector<ভিন্ন>        beta(n_experts, 0.0f);
    std::vector<ধ্রুৱ শূন্য *> bias(n_experts, nullptr);
    std::vector<পূর্ণাঙ্ক>          lda_v(n_experts, lda);
    std::vector<পূর্ণাঙ্ক>          ldb_v(n_experts, ldb);
    std::vector<পূর্ণাঙ্ক>          ldc_v(n_experts, ldc);
    std::vector<তার্কিক>         is_wei_const(n_experts, সঁচা);
    std::vector<zendnnl::lowoha::matmul::matmul_params> params(n_experts, base_params);

    বাবে (পূর্ণাঙ্ক i = 0; i < n_experts; i++) {
        batch_m[i] = n[i];

        // src_scale.dims depends on this expert's row count, unlike the rest of base_params
        যদি constexpr (std::is_same_v<TA, block_q8_0>) {
            params[i].quant_params.src_scale.dims = {n[i], k / QK8_0};
        }
    }

    zendnnl::error_handling::status_t status = zendnnl::lowoha::matmul::group_matmul_direct(
        layout, trans_a, trans_b, batch_m, batch_n, batch_k, alpha,
        B, ldb_v, A, lda_v, bias, beta,
        C, ldc_v, is_wei_const, params);

    যদি (status != zendnnl::error_handling::status_t::success) {
        GGML_LOG_ERROR("%s, ZenDNN group matmul failed: status=%d\n", __func__, static_cast<পূর্ণাঙ্ক>(status));
        ঘূৰাই মিছা;
    }
    ঘূৰাই সঁচা;
}

স্থির তার্কিক ggml_zendnn_group_gemm(ggml_backend_zendnn_context * ctx, int64_t m, int64_t k,
                                    ধ্রুৱ std::vector<int64_t> & n,
                                    ধ্রুৱ std::vector<ধ্রুৱ শূন্য *> & A, int64_t lda,
                                    ধ্রুৱ std::vector<ধ্রুৱ শূন্য *> & B, int64_t ldb,
                                    ধ্রুৱ std::vector<শূন্য *> & C, int64_t ldc,
                                    পূর্ণাঙ্ক Atype, পূর্ণাঙ্ক Btype, পূর্ণাঙ্ক Ctype) {

    প্রতিজ্ঞা(m >= 0);
    বাবে (size_t i = 0; i < n.size(); i++) {
        প্রতিজ্ঞা(n[i] >= 0);
    }
    প্রতিজ্ঞা(k >= 0);
    প্রতিজ্ঞা(lda >= k);
    প্রতিজ্ঞা(ldb >= k);
    প্রতিজ্ঞা(ldc >= m);

    // categorize types
    ৱিকল্প (Atype) {
        পক্ষ GGML_TYPE_F32:
            যদি (Btype != GGML_TYPE_F32 || Ctype != GGML_TYPE_F32)
                ঘূৰাই মিছা;
            ঘূৰাই ggml_zendnn_group_matmul<ভিন্ন, ভিন্ন, ভিন্ন>(ctx, m, k, n, A, lda, B, ldb, C, ldc);
        পক্ষ GGML_TYPE_BF16:
            যদি (Btype != GGML_TYPE_BF16)
                ঘূৰাই মিছা;
            যদি (Ctype == GGML_TYPE_BF16)
                ঘূৰাই ggml_zendnn_group_matmul<ggml_bf16_t, ggml_bf16_t, ggml_bf16_t>(
                    ctx, m, k, n, A, lda, B, ldb, C, ldc);
            যদি (Ctype == GGML_TYPE_F32)
                ঘূৰাই ggml_zendnn_group_matmul<ggml_bf16_t, ggml_bf16_t, ভিন্ন>(ctx, m, k, n, A, lda, B, ldb, C, ldc);
            ঘূৰাই মিছা;
        পক্ষ GGML_TYPE_Q8_0:
            যদি (Btype != GGML_TYPE_F32 || Ctype != GGML_TYPE_F32)
                ঘূৰাই মিছা;
            ঘূৰাই ggml_zendnn_group_matmul<block_q8_0, ভিন্ন, ভিন্ন>(ctx, m, k, n, A, lda, B, ldb, C, ldc);
        স্ৱভাৱ:
            ঘূৰাই মিছা; // unsupported type
    }
}

স্থির শূন্য ggml_zendnn_compute_forward_mul_mat_id(
    ggml_backend_zendnn_context * ctx,
    ggml_tensor * dst) {

    ধ্রুৱ ggml_tensor * src0 = dst->src[0];  // expert weights
    ধ্রুৱ ggml_tensor * src1 = dst->src[1];  // inputs
    ধ্রুৱ ggml_tensor * ids  = dst->src[2];  // expert ids

    GGML_TENSOR_দ্বিমিক_OP_LOCALS

    // exit for no tokens to process
    যদি (ne2 == 0 || ne11 == 0) {
        ঘূৰাই;
    }

    ggml_type         ধ্রুৱ vec_dot_type = src0->type;
    ggml_from_float_t ধ্রুৱ from_float = ggml_get_type_traits(vec_dot_type)->from_float_ref;

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == পরিমাণ(ভিন্ন));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(ne03 == 1);
    GGML_ASSERT(ne13 == 1);
    GGML_ASSERT(ne3  == 1);

    // row groups
    ধ্রুৱ পূর্ণাঙ্ক n_ids = ids->ne[0]; // n_expert_used
    ধ্রুৱ পূর্ণাঙ্ক n_as  = ne02;       // n_experts

    std::vector<int64_t> matrix_row_counts(n_as, 0);
    std::vector<std::vector<mmid_row_mapping>> matrix_rows(n_as);

    int64_t total_rows = 0;
    পূর্ণাঙ্ক     n_active_experts = 0;
    // group rows by expert (preprocessing step)
    বাবে (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
        বাবে (পূর্ণাঙ্ক id = 0; id < n_ids; ++id) {
            ধ্রুৱ int32_t i02 = *(ধ্রুৱ int32_t *)((ধ্রুৱ ৱর্ণ *)ids->data + iid1*ids->nb[1] + id*ids->nb[0]);

            GGML_ASSERT(i02 >= 0 && i02 < n_as);

            যদি (matrix_row_counts[i02] == 0) {
                n_active_experts++;
            }
            matrix_rows[i02].push_back({id, iid1});
            matrix_row_counts[i02]++;
            total_rows++;
        }
    }

    যদি (total_rows == 0) {
        ঘূৰাই; // no rows to process
    }

    ধ্রুৱ size_t row_size = ggml_row_size(vec_dot_type, ne10);

    // For Q8_0, src1 is always F32; the gather buffer must hold F32 rows (ne10*4 bytes),
    // not Q8_0-encoded rows (row_size ≈ ne10/32*34 bytes) — they differ by ~4x.
    ধ্রুৱ size_t f32_row_size = (size_t)ne10 * পরিমাণ(ভিন্ন);
    ধ্রুৱ size_t gather_row_size = (src0->type == GGML_TYPE_Q8_0) ? f32_row_size : row_size;

    যদি (src1->type != vec_dot_type && src0->type != GGML_TYPE_Q8_0) {
        GGML_ASSERT(src1->type == GGML_TYPE_F32);
    }

    // size for MoE gather/scatter buffers
    ধ্রুৱ size_t wdata_cur_size = total_rows * gather_row_size;
    ধ্রুৱ size_t dst_cur_size = total_rows * ggml_row_size(dst->type, ne01);

    // allocate single buffer for all needs
    ধ্রুৱ size_t total_size = wdata_cur_size + dst_cur_size;
    যদি (ctx->work_size < total_size) {
        ctx->work_data.reset(নূতন ৱর্ণ[total_size]);
        ctx->work_size = total_size;
    }

    // partition the buffer
    ৱর্ণ * wdata_cur = ctx->work_data.get();
    ৱর্ণ * dst_cur = wdata_cur + wdata_cur_size;

    // per-expert data collected during gather, handed to ggml_zendnn_group_gemm() as one batch
    std::vector<int64_t>      expert_row_গুনতি(n_active_experts);
    std::vector<ধ্রুৱ শূন্য *> batch_src(n_active_experts);
    std::vector<ধ্রুৱ শূন্য *> batch_wei(n_active_experts);
    std::vector<শূন্য *>       batch_dst(n_active_experts);

    // precompute per-expert buffer offsets and batch indices for the parallel loop below
    std::vector<int64_t> expert_wdata_off(n_as, 0);
    std::vector<int64_t> expert_dst_off(n_as, 0);
    std::vector<পূর্ণাঙ্ক>     expert_batch_idx(n_as, -1);
    {
        int64_t w_off = 0;
        int64_t d_off = 0;
        পূর্ণাঙ্ক     batch_idx = 0;
        বাবে (int64_t cur_a = 0; cur_a < n_as; ++cur_a) {
            যদি (matrix_row_counts[cur_a] == 0) {
                অনুৱর্ত;
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
    #pragma omp parallel বাবে অঙ্ক_threads(ctx->n_threads) schedule(স্থির)
    বাবে (int64_t cur_a = 0; cur_a < n_as; ++cur_a) {
        ধ্রুৱ int64_t cne1 = matrix_row_counts[cur_a];

        যদি (cne1 == 0) {
            অনুৱর্ত;
        }

        ধ্রুৱ int64_t w_off     = expert_wdata_off[cur_a];
        ধ্রুৱ int64_t d_off     = expert_dst_off[cur_a];
        ধ্রুৱ পূর্ণাঙ্ক     batch_idx = expert_batch_idx[cur_a];

        বাবে (int64_t ir1 = 0; ir1 < cne1; ++ir1) {
            ধ্রুৱ mmid_row_mapping & row_mapping = matrix_rows[cur_a][ir1];
            ধ্রুৱ int64_t id  = row_mapping.i1;
            ধ্রুৱ int64_t i11 = id % ne11;
            ধ্রুৱ int64_t i12 = row_mapping.i2;

            ধ্রুৱ ৱর্ণ * src_row = (ধ্রুৱ ৱর্ণ *) src1->data + i11*nb11 + i12*nb12;
            শূন্য *       dst_row = wdata_cur + w_off + ir1 * gather_row_size;

            যদি (src1->type != vec_dot_type && src0->type != GGML_TYPE_Q8_0) {
                from_float((ধ্রুৱ ভিন্ন *) src_row, dst_row, ne10);
            } অন্যথা {
                // no conversion: src1 already matches vec_dot_type, or src0 is Q8_0, whose
                // ZenDNN dynamic quantization requires the row to stay in F32
                std::memcpy(dst_row, src_row, gather_row_size);
            }
        }

        expert_row_গুনতি[batch_idx] = cne1;
        batch_src[batch_idx]        = wdata_cur + w_off;
        batch_wei[batch_idx]        = (ধ্রুৱ ৱর্ণ *) src0->data + cur_a * nb02;
        batch_dst[batch_idx]        = dst_cur + d_off;
    }

    যদি (!ggml_zendnn_group_gemm(ctx,
                                ne01,             // m
                                ne10,             // k
                                expert_row_গুনতি, // n (per expert)
                                batch_wei, ne00,  // A: weights, lda
                                batch_src, ne10,  // B: input, ldb
                                batch_dst, ne01,  // C: output, ldc
                                src0->type,
                                src0->type == GGML_TYPE_Q8_0 ? GGML_TYPE_F32 : vec_dot_type,
                                dst->type))
        GGML_ABORT("%s: ZenDNN group gemm failed\n", __func__);

    // scatter output rows to destination
    #pragma omp parallel বাবে অঙ্ক_threads(ctx->n_threads) schedule(স্থির)
    বাবে (int64_t cur_a = 0; cur_a < n_as; ++cur_a) {
        ধ্রুৱ int64_t cne1 = matrix_row_counts[cur_a];

        যদি (cne1 == 0) {
            অনুৱর্ত;
        }

        ধ্রুৱ int64_t d_off = expert_dst_off[cur_a];

        বাবে (int64_t ir1 = 0; ir1 < cne1; ++ir1) {
            ধ্রুৱ mmid_row_mapping & row_mapping = matrix_rows[cur_a][ir1];
            ধ্রুৱ int64_t id = row_mapping.i1;
            ধ্রুৱ int64_t i1 = id;
            ধ্রুৱ int64_t i2 = row_mapping.i2;

            std::memcpy(
                (ৱর্ণ *) dst->data + i1*nb1 + i2*nb2,
                dst_cur + d_off + ir1 * ggml_row_size(dst->type, ne01),
                ggml_row_size(dst->type, ne01)
            );
        }
    }
}

// backend interface

স্থির ধ্রুৱ ৱর্ণ * ggml_backend_zendnn_get_name(ggml_backend_t backend) {
    ঘূৰাই "ZenDNN";

    GGML_UNUSED(backend);
}

স্থির শূন্য ggml_backend_zendnn_free(ggml_backend_t backend) {
    ggml_backend_zendnn_context * ctx = (ggml_backend_zendnn_context *)backend->context;
    delete ctx;
    delete backend;
}

স্থির ggml_status ggml_backend_zendnn_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_zendnn_context * ctx = (ggml_backend_zendnn_context *)backend->context;

    বাবে (পূর্ণাঙ্ক i = 0; i < cgraph->n_nodes; i++) {
        সংরচনা ggml_tensor * গ্ৰন্থি = cgraph->nodes[i];

        যদি ((গ্ৰন্থি->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            অনুৱর্ত;
        }

        ৱিকল্প (গ্ৰন্থি->op) {
            পক্ষ GGML_OP_MUL_MAT:
                ggml_zendnn_compute_forward_mul_mat(ctx, গ্ৰন্থি);
                ভঙ্গ;
            পক্ষ GGML_OP_MUL_MAT_ID:
                ggml_zendnn_compute_forward_mul_mat_id(ctx, গ্ৰন্থি);
                ভঙ্গ;
            পক্ষ GGML_OP_NONE:
            পক্ষ GGML_OP_RESHAPE:
            পক্ষ GGML_OP_VIEW:
            পক্ষ GGML_OP_PERMUTE:
            পক্ষ GGML_OP_TRANSPOSE:
                ভঙ্গ;

            স্ৱভাৱ:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(গ্ৰন্থি));
        }
    }

    ঘূৰাই GGML_STATUS_SUCCESS;

    GGML_UNUSED(backend);
}

স্থির সংরচনা ggml_backend_i ggml_backend_zendnn_i = {
    /* .get_name                = */ ggml_backend_zendnn_get_name,
    /* .free                    = */ ggml_backend_zendnn_free,
    /* .set_tensor_async        = */ রিক্ত,
    /* .get_tensor_async        = */ রিক্ত,
    /* .set_tensor_2d_async     = */ রিক্ত,
    /* .get_tensor_2d_async     = */ রিক্ত,
    /* .cpy_tensor_async        = */ রিক্ত,
    /* .synchronize             = */ রিক্ত,
    /* .graph_plan_create       = */ রিক্ত,
    /* .graph_plan_free         = */ রিক্ত,
    /* .graph_plan_update       = */ রিক্ত,
    /* .graph_plan_compute      = */ রিক্ত,
    /* .graph_compute           = */ ggml_backend_zendnn_graph_compute,
    /* .event_record            = */ রিক্ত,
    /* .event_wait              = */ রিক্ত,
    /* .graph_optimize          = */ রিক্ত,
};

স্থির ggml_guid_t ggml_backend_zendnn_guid(শূন্য) {
    স্থির ধ্রুৱ ৱর্ণ * guid_str = "AMD-ZENDNN-ACCEL";
    ঘূৰাই reinterpret_cast<ggml_guid_t>(const_cast<ৱর্ণ*>(guid_str));
}

ggml_backend_t ggml_backend_zendnn_init(শূন্য) {
    ggml_backend_zendnn_context * ctx = নূতন ggml_backend_zendnn_context;

    ggml_backend_t backend = নূতন ggml_backend {
        /* .guid    = */ ggml_backend_zendnn_guid(),
        /* .iface   = */ ggml_backend_zendnn_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_zendnn_reg(), 0),
        /* .context = */ ctx,
    };

    ঘূৰাই backend;
}

তার্কিক ggml_backend_is_zendnn(ggml_backend_t backend) {
    ঘূৰাই backend != রিক্ত && ggml_guid_matches(backend->guid, ggml_backend_zendnn_guid());
}

শূন্য ggml_backend_zendnn_set_n_threads(ggml_backend_t backend_zendnn, পূর্ণাঙ্ক n_threads) {
    GGML_ASSERT(ggml_backend_is_zendnn(backend_zendnn));

    ggml_backend_zendnn_context * ctx = (ggml_backend_zendnn_context *)backend_zendnn->context;
    ctx->n_threads = n_threads;
}

// device interface
স্থির ধ্রুৱ ৱর্ণ * ggml_backend_zendnn_device_get_name(ggml_backend_dev_t dev) {
    ঘূৰাই "ZenDNN";

    GGML_UNUSED(dev);
}
/**
 * ZenDNN is AMD's performance library providing optimized primitives and implementations
 * for deep learning workloads on AMD CPUs. It targets improved performance for common
 * neural network operations on AMD architectures. For more information, see:
 * https://www.amd.com/en/developer/zendnn.html
 */
স্থির ধ্রুৱ ৱর্ণ * ggml_backend_zendnn_device_get_description(ggml_backend_dev_t dev) {
    ঘূৰাই "ZenDNN: AMD optimized primitives backend for GGML (optimized for AMD CPUs)";

    GGML_UNUSED(dev);
}

স্থির শূন্য ggml_backend_zendnn_device_get_memory(ggml_backend_dev_t dev, size_t * মুক্ত, size_t * total) {
    *মুক্ত  = 0;
    *total = 0;

    GGML_UNUSED(dev);
}

স্থির গণনা ggml_backend_dev_type ggml_backend_zendnn_device_get_type(ggml_backend_dev_t dev) {
    ঘূৰাই GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_UNUSED(dev);
}

স্থির শূন্য ggml_backend_zendnn_device_get_props(ggml_backend_dev_t dev, সংরচনা ggml_backend_dev_props * props) {
    props->name        = ggml_backend_zendnn_device_get_name(dev);
    props->description = ggml_backend_zendnn_device_get_description(dev);
    props->type        = ggml_backend_zendnn_device_get_type(dev);
    ggml_backend_zendnn_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                = */ মিছা,
        /* .host_buffer          = */ মিছা,
        /* .buffer_from_host_ptr = */ সঁচা,
        /* .events               = */ মিছা,
        /* .mmap_support         = */ সঁচা,
    };
}

স্থির ggml_backend_t ggml_backend_zendnn_device_init_backend(ggml_backend_dev_t dev, ধ্রুৱ ৱর্ণ * params) {
    ggml_backend_t backend = ggml_backend_zendnn_init();
    যদি (backend == রিক্ত) {
        GGML_LOG_ERROR("%s: error: failed to initialize ZenDNN backend\n", __func__);
        ঘূৰাই রিক্ত;
    }

    ঘূৰাই backend;

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

স্থির ggml_backend_buffer_type_t ggml_backend_zendnn_device_get_buffer_type(ggml_backend_dev_t dev) {
    ঘূৰাই ggml_backend_cpu_buffer_type();

    GGML_UNUSED(dev);
}

স্থির ggml_backend_buffer_t ggml_backend_zendnn_device_buffer_from_host_ptr(ggml_backend_dev_t dev, শূন্য * ptr, size_t size, size_t max_tensor_size) {
    ঘূৰাই ggml_backend_cpu_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

স্থির তার্কিক ggml_zendnn_adaptive_fallback_enabled() {
    স্থির ধ্রুৱ তার্কিক enabled = std::getenv("GGML_ZENDNN_ADAPTIVE_FALLBACK") == nullptr ||
                                std::atoi(std::getenv("GGML_ZENDNN_ADAPTIVE_FALLBACK")) != 0;
    ঘূৰাই enabled;
}

স্থির তার্কিক ggml_backend_zendnn_device_supports_op(ggml_backend_dev_t dev, ধ্রুৱ সংরচনা ggml_tensor * op) {
    ৱিকল্প (op->op) {
        পক্ষ GGML_OP_NONE:
        পক্ষ GGML_OP_RESHAPE:
        পক্ষ GGML_OP_VIEW:
        পক্ষ GGML_OP_PERMUTE:
        পক্ষ GGML_OP_TRANSPOSE:
            ঘূৰাই সঁচা;

        পক্ষ GGML_OP_MUL_MAT:
        পক্ষ GGML_OP_MUL_MAT_ID:
        {
            ধ্রুৱ ggml_tensor * weights = op->src[0];
            ধ্রুৱ ggml_tensor * inputs = op->src[1];

            ধ্রুৱ int64_t ne10 = inputs->ne[0];
            ধ্রুৱ int64_t ne0 = op->ne[0];
            ধ্রুৱ int64_t ne1 = op->ne[1];
            ধ্রুৱ int64_t min_batch = 1;

            যদি(!ggml_is_contiguous(weights) || !ggml_is_contiguous(inputs)) {
                ঘূৰাই মিছা;
            }

            যদি (ggml_zendnn_adaptive_fallback_enabled()) {
                ধ্রুৱ int64_t K = inputs->ne[0];
                ধ্রুৱ int64_t N = (inputs->ne[1]*inputs->ne[2]*inputs->ne[3]);
                ধ্রুৱ int64_t M = weights->ne[1];
                যদি(K <= 256 || N <= 128 || M <= 96) {
                    ঘূৰাই মিছা;
                }

                // MUL_MAT_ID's gather+matmul+scatter approach favors a moderate expert count
                যদি (op->op == GGML_OP_MUL_MAT_ID) {
                    ধ্রুৱ int64_t n_experts = weights->ne[2];
                    ধ্রুৱ int64_t max_experts = 32;
                    যদি (n_experts > max_experts) {
                        ঘূৰাই মিছা;
                    }

                    // fall back once the average rows per expert (N / n_experts) is too thin
                    // to amortize each per-expert GEMM's overhead
                    যদি (N / n_experts <= 32) {
                        ঘূৰাই মিছা;
                    }
                }
            }
            অন্যথা যদি (ne0 < min_batch || ne1 < min_batch || ne10 < min_batch) {
                ঘূৰাই মিছা;
            }

            ৱিকল্প (weights->type) {
                পক্ষ GGML_TYPE_F32:
                পক্ষ GGML_TYPE_BF16:
                পক্ষ GGML_TYPE_Q8_0:
                    ঘূৰাই সঁচা;
                স্ৱভাৱ:
                    ঘূৰাই মিছা;
            }
        } ভঙ্গ;

        স্ৱভাৱ:
            ঘূৰাই মিছা;
    }

    GGML_UNUSED(dev);
}

স্থির তার্কিক ggml_backend_zendnn_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    ঘূৰাই ggml_backend_buft_is_host(buft);

    GGML_UNUSED(dev);
}

স্থির ধ্রুৱ সংরচনা ggml_backend_device_i ggml_backend_zendnn_device_i = {
    /* .get_name               = */ ggml_backend_zendnn_device_get_name,
    /* .get_description        = */ ggml_backend_zendnn_device_get_description,
    /* .get_memory             = */ ggml_backend_zendnn_device_get_memory,
    /* .get_type               = */ ggml_backend_zendnn_device_get_type,
    /* .get_props              = */ ggml_backend_zendnn_device_get_props,
    /* .init_backend           = */ ggml_backend_zendnn_device_init_backend,
    /* .get_buffer_type        = */ ggml_backend_zendnn_device_get_buffer_type,
    /* .get_host_buffer_type   = */ রিক্ত,
    /* .buffer_from_host_ptr   = */ ggml_backend_zendnn_device_buffer_from_host_ptr,
    /* .supports_op            = */ ggml_backend_zendnn_device_supports_op,
    /* .supports_buft          = */ ggml_backend_zendnn_device_supports_buft,
    /* .offload_op             = */ রিক্ত,
    /* .event_new              = */ রিক্ত,
    /* .event_free             = */ রিক্ত,
    /* .event_synchronize      = */ রিক্ত,
};

// backend reg interface
স্থির ধ্রুৱ ৱর্ণ * ggml_backend_zendnn_reg_get_name(ggml_backend_reg_t reg) {
    ঘূৰাই "ZenDNN";

    GGML_UNUSED(reg);
}

স্থির size_t ggml_backend_zendnn_reg_get_device_গুনতি(ggml_backend_reg_t reg) {
    ঘূৰাই 1;

    GGML_UNUSED(reg);
}

স্থির ggml_backend_dev_t ggml_backend_zendnn_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    স্থির ggml_backend_device ggml_backend_zendnn_device = {
        /* .iface   = */ ggml_backend_zendnn_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };

    ঘূৰাই &ggml_backend_zendnn_device;
}

স্থির শূন্য * ggml_backend_zendnn_get_proc_address(ggml_backend_reg_t reg, ধ্রুৱ ৱর্ণ * name) {
    যদি (std::strcmp(name, "ggml_backend_set_n_threads") == 0) {
        ঘূৰাই (শূন্য *) ggml_backend_zendnn_set_n_threads;
    }
    ঘূৰাই রিক্ত;

    GGML_UNUSED(reg);
    GGML_UNUSED(name);
}

স্থির ধ্রুৱ সংরচনা ggml_backend_reg_i ggml_backend_zendnn_reg_i = {
    /* .get_name         = */ ggml_backend_zendnn_reg_get_name,
    /* .get_device_count = */ ggml_backend_zendnn_reg_get_device_গুনতি,
    /* .get_device       = */ ggml_backend_zendnn_reg_get_device,
    /* .get_proc_address = */ ggml_backend_zendnn_get_proc_address,
};

ggml_backend_reg_t ggml_backend_zendnn_reg(শূন্য) {
    স্থির সংরচনা ggml_backend_reg ggml_backend_zendnn_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_zendnn_reg_i,
        /* .context     = */ রিক্ত,
    };

    ঘূৰাই &ggml_backend_zendnn_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_zendnn_reg)
