#অন্তর্ভাব "ggml-zendnn.h"

#অন্তর্ভাব "ggml-backend-impl.h"
#অন্তর্ভাব "ggml-impl.h"

#define GGML_COMMON_DECL_CPP
#অন্তর্ভাব "ggml-common.h"

#অন্তর্ভাব "zendnnl.hpp"

#অন্তর্ভাব <cstring>


কাঠামো ggml_backend_zendnn_context {
    পূর্ণসংখ্যা n_threads = GGML_DEFAULT_N_THREADS;
    std::unique_ptr<অক্ষর[]> work_তথ্য;
    size_t work_আকার = 0;
};

template<typename T>
zendnnl::common::তথ্য_type_t ggml_to_zendnn_type() {
    যদি constexpr (std::is_same_v<T, দশমিক>) {
        ফেরত zendnnl::common::তথ্য_type_t::f32;
    } অন্যথা যদি constexpr (std::is_same_v<T, ggml_bf16_t>) {
        ফেরত zendnnl::common::তথ্য_type_t::bf16;
    } অন্যথা যদি constexpr (std::is_same_v<T, block_q8_0>) {
        ফেরত zendnnl::common::তথ্য_type_t::s8;
    } অন্যথা {
        ফেরত zendnnl::common::তথ্য_type_t::none;
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
        params.dtypes.compute = zendnnl::common::তথ্য_type_t::s8;
        params.dynamic_quant = সত্য;
        params.quant_params.src_scale.buff = nullptr;
        params.quant_params.src_scale.dt   = zendnnl::common::তথ্য_type_t::bf16;
        params.packing.pack_format_b = 1;
    }
    ফেরত params;
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
স্থির যৌক্তিক ggml_zendnn_matmul(ggml_backend_zendnn_context * ctx, int64_t m, int64_t n, int64_t k,
                               ধ্রুবক TA * A, int64_t lda, ধ্রুবক TB * B, int64_t ldb, TC * C,
                               int64_t ldc) {

    zendnnl::lowoha::matmul::matmul_params params = ggml_zendnn_make_matmul_params<TA, TB, TC>(ctx);

    zendnnl::lowoha::matmul::matmul_batch_params_t batch_params;

    যদি constexpr (std::is_same_v<TA, block_q8_0>) {
        params.quant_params.src_scale.dims = {n, k / QK8_0};
    }

    zendnnl::error_handling::status_t status = zendnnl::lowoha::matmul::matmul_direct(
        'r', মিথ্যা, সত্য,   // row-major, don't transpose B, transpose A (because it's column-major)
        n,                  // M: rows of B and C
        m,                  // N: cols of A^T and C
        k,                  // K: cols of B, rows of A
        1.0f,               // alpha
        B, ldb,             // src: B[n,k]
        A, lda,             // weight: A[k,m] column-major (transposed)
        nullptr,            // bias
        0.0f,               // beta
        C, ldc,             // output C[n,m]
        সত্য,               // is_weights_const
        batch_params,       // batch_params
        params              // params
    );

    যদি (status != zendnnl::error_handling::status_t::success) {
        GGML_LOG_ERROR("%s, ZenDNN matmul failed: status=%d\n", __func__, static_cast<পূর্ণসংখ্যা>(status));
        ফেরত মিথ্যা;
    }
    ফেরত সত্য;
}

স্থির যৌক্তিক ggml_zendnn_gemm(ggml_backend_zendnn_context * ctx, int64_t m, int64_t n, int64_t k,
                              ধ্রুবক শূন্য * A, int64_t lda, ধ্রুবক শূন্য * B, int64_t ldb, শূন্য * C,
                              int64_t ldc, পূর্ণসংখ্যা Atype, পূর্ণসংখ্যা Btype, পূর্ণসংখ্যা Ctype) {

    প্রতিজ্ঞা(m >= 0);
    প্রতিজ্ঞা(n >= 0);
    প্রতিজ্ঞা(k >= 0);
    প্রতিজ্ঞা(lda >= k);
    প্রতিজ্ঞা(ldb >= k);
    প্রতিজ্ঞা(ldc >= m);

    // categorize types
    নির্বাচন (Atype) {
        ক্ষেত্র GGML_TYPE_F32:
            যদি (Btype != GGML_TYPE_F32 || Ctype != GGML_TYPE_F32)
                ফেরত মিথ্যা;
            ফেরত ggml_zendnn_matmul<দশমিক, দশমিক, দশমিক>(
                ctx, m, n, k,
                (ধ্রুবক দশমিক *)A, lda,
                (ধ্রুবক দশমিক *)B, ldb,
                (দশমিক *)C, ldc);
        ক্ষেত্র GGML_TYPE_BF16:
            যদি (Btype != GGML_TYPE_BF16)
                ফেরত মিথ্যা;
            যদি (Ctype == GGML_TYPE_BF16)
                ফেরত ggml_zendnn_matmul<ggml_bf16_t, ggml_bf16_t, ggml_bf16_t>(
                    ctx, m, n, k,
                    (ধ্রুবক ggml_bf16_t *)A, lda,
                    (ধ্রুবক ggml_bf16_t *)B, ldb,
                    (ggml_bf16_t *)C, ldc);
            যদি (Ctype == GGML_TYPE_F32)
                ফেরত ggml_zendnn_matmul<ggml_bf16_t, ggml_bf16_t, দশমিক>(
                    ctx, m, n, k,
                    (ধ্রুবক ggml_bf16_t *)A, lda,
                    (ধ্রুবক ggml_bf16_t *)B, ldb,
                    (দশমিক *)C, ldc);
            ফেরত মিথ্যা;
        ক্ষেত্র GGML_TYPE_Q8_0:
            যদি (Btype != GGML_TYPE_F32 || Ctype != GGML_TYPE_F32)
                ফেরত মিথ্যা;
            ফেরত ggml_zendnn_matmul<block_q8_0, দশমিক, দশমিক>(
                ctx, m, n, k,
                (ধ্রুবক block_q8_0 *)A, lda,
                (ধ্রুবক দশমিক *)B, ldb,
                (দশমিক *)C, ldc);
        পূর্বনির্ধারিত:
            ফেরত মিথ্যা; // unsupported type
    }
}

স্থির শূন্য ggml_zendnn_compute_forward_mul_mat(
    ggml_backend_zendnn_context * ctx,
    ggml_tensor * dst) {

    ধ্রুবক ggml_tensor * src0 = dst->src[0];  // weights
    ধ্রুবক ggml_tensor * src1 = dst->src[1];  // inputs

    GGML_TENSOR_দ্বিমিক_OP_LOCALS

    ggml_type         ধ্রুবক vec_dot_type = src0->type;
    ggml_from_float_t ধ্রুবক from_float = ggml_get_type_traits(vec_dot_type)->from_float_ref;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_আকার(src0->type));
    GGML_ASSERT(nb10 == ggml_type_আকার(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == আকার(দশমিক));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // broadcast factors
    ধ্রুবক int64_t r2 = ne12/ne02;
    ধ্রুবক int64_t r3 = ne13/ne03;

    শূন্য * work_তথ্য = ctx->work_তথ্য.get();

    // ZenDNN requires FP32 for dynamic quantization, so conversion is skipped
    যদি (src1->type != vec_dot_type && src0->type != GGML_TYPE_Q8_0) {
        ধ্রুবক size_t nbw1 = ggml_row_আকার(vec_dot_type, ne10);
        ধ্রুবক size_t nbw2 = nbw1 * ne11;
        ধ্রুবক size_t nbw3 = nbw2 * ne12;
        ধ্রুবক size_t desired_wsize = ne13 * nbw3;
        যদি (ctx->work_আকার < desired_wsize) {
            ctx->work_তথ্য.reset(নতুন অক্ষর[desired_wsize]);
            ctx->work_আকার = desired_wsize;
        }
        work_তথ্য = ctx->work_তথ্য.get();

        // #pragma omp parallel for num_threads(ctx->n_threads)
        #pragma omp parallel জন্য collapse(3) অঙ্ক_threads(ctx->n_threads) schedule(স্থির)
        জন্য (int64_t i13 = 0; i13 < ne13; ++i13) {
            জন্য (int64_t i12 = 0; i12 < ne12; ++i12) {
                জন্য (int64_t i11 = 0; i11 < ne11; ++i11) {
                    ধ্রুবক দশমিক * src1_f32 = (দশমিক *)((অক্ষর *)src1->তথ্য + i11*nb11 + i12*nb12 + i13*nb13);
                    শূন্য * src1_conv = (অক্ষর *)work_তথ্য + i11*nbw1 + i12*nbw2 + i13*nbw3;
                    from_float(src1_f32, src1_conv, ne10);
                }
            }
        }
    }

    জন্য (int64_t i13 = 0; i13 < ne13; i13++) {
        জন্য (int64_t i12 = 0; i12 < ne12; i12++) {
            ধ্রুবক শূন্য* wdata = (src1->type == vec_dot_type || src0->type == GGML_TYPE_Q8_0) ? src1->তথ্য : work_তথ্য;
            ধ্রুবক size_t row_আকার = ggml_row_আকার(vec_dot_type, ne10);
            যদি (!ggml_zendnn_gemm(ctx,
                                  ne01,     // m
                                  ne11,     // n
                                  ne10,     // k
                                  static_cast<ধ্রুবক অক্ষর *>(src0->তথ্য) + (i12/r2)*nb02 + (i13/r3)*nb03,
                                  ne00,     // lda
                                  static_cast<ধ্রুবক অক্ষর *>(wdata) + (i12*ne11 + i13*ne12*ne11)*row_আকার,
                                  ne10,     // ldb
                                  static_cast<অক্ষর *>(dst->তথ্য) + i12*nb2 + i13*nb3,
                                  ne01,     // ldc
                                  src0->type,
                                  src0->type == GGML_TYPE_Q8_0 ? GGML_TYPE_F32 : vec_dot_type,
                                  dst->type))
                GGML_ABORT("%s: ZenDNN gemm failed\n", __func__);
        }
    }
}

কাঠামো mmid_row_mapping {
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
স্থির যৌক্তিক ggml_zendnn_group_matmul(ggml_backend_zendnn_context * ctx, int64_t m, int64_t k,
                                     ধ্রুবক std::vector<int64_t> & n,
                                     ধ্রুবক std::vector<ধ্রুবক শূন্য *> & A, int64_t lda,
                                     ধ্রুবক std::vector<ধ্রুবক শূন্য *> & B, int64_t ldb,
                                     ধ্রুবক std::vector<শূন্য *> & C, int64_t ldc) {

    ধ্রুবক পূর্ণসংখ্যা n_experts = n.size();

    zendnnl::lowoha::matmul::matmul_params base_params = ggml_zendnn_make_matmul_params<TA, TB, TC>(ctx);

    std::vector<অক্ষর>         layout(n_experts, 'r');
    std::vector<যৌক্তিক>         trans_a(n_experts, মিথ্যা);
    std::vector<যৌক্তিক>         trans_b(n_experts, সত্য);
    std::vector<পূর্ণসংখ্যা>          batch_m(n_experts);
    std::vector<পূর্ণসংখ্যা>          batch_n(n_experts, m);
    std::vector<পূর্ণসংখ্যা>          batch_k(n_experts, k);
    std::vector<দশমিক>        alpha(n_experts, 1.0f);
    std::vector<দশমিক>        beta(n_experts, 0.0f);
    std::vector<ধ্রুবক শূন্য *> bias(n_experts, nullptr);
    std::vector<পূর্ণসংখ্যা>          lda_v(n_experts, lda);
    std::vector<পূর্ণসংখ্যা>          ldb_v(n_experts, ldb);
    std::vector<পূর্ণসংখ্যা>          ldc_v(n_experts, ldc);
    std::vector<যৌক্তিক>         is_wei_const(n_experts, সত্য);
    std::vector<zendnnl::lowoha::matmul::matmul_params> params(n_experts, base_params);

    জন্য (পূর্ণসংখ্যা i = 0; i < n_experts; i++) {
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
        GGML_LOG_ERROR("%s, ZenDNN group matmul failed: status=%d\n", __func__, static_cast<পূর্ণসংখ্যা>(status));
        ফেরত মিথ্যা;
    }
    ফেরত সত্য;
}

স্থির যৌক্তিক ggml_zendnn_group_gemm(ggml_backend_zendnn_context * ctx, int64_t m, int64_t k,
                                    ধ্রুবক std::vector<int64_t> & n,
                                    ধ্রুবক std::vector<ধ্রুবক শূন্য *> & A, int64_t lda,
                                    ধ্রুবক std::vector<ধ্রুবক শূন্য *> & B, int64_t ldb,
                                    ধ্রুবক std::vector<শূন্য *> & C, int64_t ldc,
                                    পূর্ণসংখ্যা Atype, পূর্ণসংখ্যা Btype, পূর্ণসংখ্যা Ctype) {

    প্রতিজ্ঞা(m >= 0);
    জন্য (size_t i = 0; i < n.size(); i++) {
        প্রতিজ্ঞা(n[i] >= 0);
    }
    প্রতিজ্ঞা(k >= 0);
    প্রতিজ্ঞা(lda >= k);
    প্রতিজ্ঞা(ldb >= k);
    প্রতিজ্ঞা(ldc >= m);

    // categorize types
    নির্বাচন (Atype) {
        ক্ষেত্র GGML_TYPE_F32:
            যদি (Btype != GGML_TYPE_F32 || Ctype != GGML_TYPE_F32)
                ফেরত মিথ্যা;
            ফেরত ggml_zendnn_group_matmul<দশমিক, দশমিক, দশমিক>(ctx, m, k, n, A, lda, B, ldb, C, ldc);
        ক্ষেত্র GGML_TYPE_BF16:
            যদি (Btype != GGML_TYPE_BF16)
                ফেরত মিথ্যা;
            যদি (Ctype == GGML_TYPE_BF16)
                ফেরত ggml_zendnn_group_matmul<ggml_bf16_t, ggml_bf16_t, ggml_bf16_t>(
                    ctx, m, k, n, A, lda, B, ldb, C, ldc);
            যদি (Ctype == GGML_TYPE_F32)
                ফেরত ggml_zendnn_group_matmul<ggml_bf16_t, ggml_bf16_t, দশমিক>(ctx, m, k, n, A, lda, B, ldb, C, ldc);
            ফেরত মিথ্যা;
        ক্ষেত্র GGML_TYPE_Q8_0:
            যদি (Btype != GGML_TYPE_F32 || Ctype != GGML_TYPE_F32)
                ফেরত মিথ্যা;
            ফেরত ggml_zendnn_group_matmul<block_q8_0, দশমিক, দশমিক>(ctx, m, k, n, A, lda, B, ldb, C, ldc);
        পূর্বনির্ধারিত:
            ফেরত মিথ্যা; // unsupported type
    }
}

স্থির শূন্য ggml_zendnn_compute_forward_mul_mat_id(
    ggml_backend_zendnn_context * ctx,
    ggml_tensor * dst) {

    ধ্রুবক ggml_tensor * src0 = dst->src[0];  // expert weights
    ধ্রুবক ggml_tensor * src1 = dst->src[1];  // inputs
    ধ্রুবক ggml_tensor * ids  = dst->src[2];  // expert ids

    GGML_TENSOR_দ্বিমিক_OP_LOCALS

    // exit for no tokens to process
    যদি (ne2 == 0 || ne11 == 0) {
        ফেরত;
    }

    ggml_type         ধ্রুবক vec_dot_type = src0->type;
    ggml_from_float_t ধ্রুবক from_float = ggml_get_type_traits(vec_dot_type)->from_float_ref;

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_আকার(src0->type));
    GGML_ASSERT(nb10 == ggml_type_আকার(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == আকার(দশমিক));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(ne03 == 1);
    GGML_ASSERT(ne13 == 1);
    GGML_ASSERT(ne3  == 1);

    // row groups
    ধ্রুবক পূর্ণসংখ্যা n_ids = ids->ne[0]; // n_expert_used
    ধ্রুবক পূর্ণসংখ্যা n_as  = ne02;       // n_experts

    std::vector<int64_t> matrix_row_counts(n_as, 0);
    std::vector<std::vector<mmid_row_mapping>> matrix_rows(n_as);

    int64_t যোগ_rows = 0;
    পূর্ণসংখ্যা     n_active_experts = 0;
    // group rows by expert (preprocessing step)
    জন্য (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
        জন্য (পূর্ণসংখ্যা id = 0; id < n_ids; ++id) {
            ধ্রুবক int32_t i02 = *(ধ্রুবক int32_t *)((ধ্রুবক অক্ষর *)ids->তথ্য + iid1*ids->nb[1] + id*ids->nb[0]);

            GGML_ASSERT(i02 >= 0 && i02 < n_as);

            যদি (matrix_row_counts[i02] == 0) {
                n_active_experts++;
            }
            matrix_rows[i02].push_back({id, iid1});
            matrix_row_counts[i02]++;
            যোগ_rows++;
        }
    }

    যদি (যোগ_rows == 0) {
        ফেরত; // no rows to process
    }

    ধ্রুবক size_t row_আকার = ggml_row_আকার(vec_dot_type, ne10);

    // For Q8_0, src1 is always F32; the gather buffer must hold F32 rows (ne10*4 bytes),
    // not Q8_0-encoded rows (row_size ≈ ne10/32*34 bytes) — they differ by ~4x.
    ধ্রুবক size_t f32_row_আকার = (size_t)ne10 * আকার(দশমিক);
    ধ্রুবক size_t gather_row_আকার = (src0->type == GGML_TYPE_Q8_0) ? f32_row_আকার : row_আকার;

    যদি (src1->type != vec_dot_type && src0->type != GGML_TYPE_Q8_0) {
        GGML_ASSERT(src1->type == GGML_TYPE_F32);
    }

    // size for MoE gather/scatter buffers
    ধ্রুবক size_t wdata_cur_আকার = যোগ_rows * gather_row_আকার;
    ধ্রুবক size_t dst_cur_আকার = যোগ_rows * ggml_row_আকার(dst->type, ne01);

    // allocate single buffer for all needs
    ধ্রুবক size_t যোগ_আকার = wdata_cur_আকার + dst_cur_আকার;
    যদি (ctx->work_আকার < যোগ_আকার) {
        ctx->work_তথ্য.reset(নতুন অক্ষর[যোগ_আকার]);
        ctx->work_আকার = যোগ_আকার;
    }

    // partition the buffer
    অক্ষর * wdata_cur = ctx->work_তথ্য.get();
    অক্ষর * dst_cur = wdata_cur + wdata_cur_আকার;

    // per-expert data collected during gather, handed to ggml_zendnn_group_gemm() as one batch
    std::vector<int64_t>      expert_row_গুনতি(n_active_experts);
    std::vector<ধ্রুবক শূন্য *> batch_src(n_active_experts);
    std::vector<ধ্রুবক শূন্য *> batch_wei(n_active_experts);
    std::vector<শূন্য *>       batch_dst(n_active_experts);

    // precompute per-expert buffer offsets and batch indices for the parallel loop below
    std::vector<int64_t> expert_wdata_off(n_as, 0);
    std::vector<int64_t> expert_dst_off(n_as, 0);
    std::vector<পূর্ণসংখ্যা>     expert_batch_idx(n_as, -1);
    {
        int64_t w_off = 0;
        int64_t d_off = 0;
        পূর্ণসংখ্যা     batch_idx = 0;
        জন্য (int64_t cur_a = 0; cur_a < n_as; ++cur_a) {
            যদি (matrix_row_counts[cur_a] == 0) {
                চালু;
            }
            expert_wdata_off[cur_a] = w_off;
            expert_dst_off[cur_a]   = d_off;
            expert_batch_idx[cur_a] = batch_idx;
            w_off += matrix_row_counts[cur_a] * gather_row_আকার;
            d_off += matrix_row_counts[cur_a] * ggml_row_আকার(dst->type, ne01);
            batch_idx++;
        }
    }

    // gather + inline-convert input rows into each expert's batch slot
    #pragma omp parallel জন্য অঙ্ক_threads(ctx->n_threads) schedule(স্থির)
    জন্য (int64_t cur_a = 0; cur_a < n_as; ++cur_a) {
        ধ্রুবক int64_t cne1 = matrix_row_counts[cur_a];

        যদি (cne1 == 0) {
            চালু;
        }

        ধ্রুবক int64_t w_off     = expert_wdata_off[cur_a];
        ধ্রুবক int64_t d_off     = expert_dst_off[cur_a];
        ধ্রুবক পূর্ণসংখ্যা     batch_idx = expert_batch_idx[cur_a];

        জন্য (int64_t ir1 = 0; ir1 < cne1; ++ir1) {
            ধ্রুবক mmid_row_mapping & row_mapping = matrix_rows[cur_a][ir1];
            ধ্রুবক int64_t id  = row_mapping.i1;
            ধ্রুবক int64_t i11 = id % ne11;
            ধ্রুবক int64_t i12 = row_mapping.i2;

            ধ্রুবক অক্ষর * src_row = (ধ্রুবক অক্ষর *) src1->তথ্য + i11*nb11 + i12*nb12;
            শূন্য *       dst_row = wdata_cur + w_off + ir1 * gather_row_আকার;

            যদি (src1->type != vec_dot_type && src0->type != GGML_TYPE_Q8_0) {
                from_float((ধ্রুবক দশমিক *) src_row, dst_row, ne10);
            } অন্যথা {
                // no conversion: src1 already matches vec_dot_type, or src0 is Q8_0, whose
                // ZenDNN dynamic quantization requires the row to stay in F32
                std::memcpy(dst_row, src_row, gather_row_আকার);
            }
        }

        expert_row_গুনতি[batch_idx] = cne1;
        batch_src[batch_idx]        = wdata_cur + w_off;
        batch_wei[batch_idx]        = (ধ্রুবক অক্ষর *) src0->তথ্য + cur_a * nb02;
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
    #pragma omp parallel জন্য অঙ্ক_threads(ctx->n_threads) schedule(স্থির)
    জন্য (int64_t cur_a = 0; cur_a < n_as; ++cur_a) {
        ধ্রুবক int64_t cne1 = matrix_row_counts[cur_a];

        যদি (cne1 == 0) {
            চালু;
        }

        ধ্রুবক int64_t d_off = expert_dst_off[cur_a];

        জন্য (int64_t ir1 = 0; ir1 < cne1; ++ir1) {
            ধ্রুবক mmid_row_mapping & row_mapping = matrix_rows[cur_a][ir1];
            ধ্রুবক int64_t id = row_mapping.i1;
            ধ্রুবক int64_t i1 = id;
            ধ্রুবক int64_t i2 = row_mapping.i2;

            std::memcpy(
                (অক্ষর *) dst->তথ্য + i1*nb1 + i2*nb2,
                dst_cur + d_off + ir1 * ggml_row_আকার(dst->type, ne01),
                ggml_row_আকার(dst->type, ne01)
            );
        }
    }
}

// backend interface

স্থির ধ্রুবক অক্ষর * ggml_backend_zendnn_get_name(ggml_backend_t backend) {
    ফেরত "ZenDNN";

    GGML_UNUSED(backend);
}

স্থির শূন্য ggml_backend_zendnn_free(ggml_backend_t backend) {
    ggml_backend_zendnn_context * ctx = (ggml_backend_zendnn_context *)backend->context;
    delete ctx;
    delete backend;
}

স্থির ggml_status ggml_backend_zendnn_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_zendnn_context * ctx = (ggml_backend_zendnn_context *)backend->context;

    জন্য (পূর্ণসংখ্যা i = 0; i < cgraph->n_nodes; i++) {
        কাঠামো ggml_tensor * গ্রন্থি = cgraph->nodes[i];

        যদি ((গ্রন্থি->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            চালু;
        }

        নির্বাচন (গ্রন্থি->op) {
            ক্ষেত্র GGML_OP_MUL_MAT:
                ggml_zendnn_compute_forward_mul_mat(ctx, গ্রন্থি);
                বিরতি;
            ক্ষেত্র GGML_OP_MUL_MAT_ID:
                ggml_zendnn_compute_forward_mul_mat_id(ctx, গ্রন্থি);
                বিরতি;
            ক্ষেত্র GGML_OP_NONE:
            ক্ষেত্র GGML_OP_RESHAPE:
            ক্ষেত্র GGML_OP_VIEW:
            ক্ষেত্র GGML_OP_PERMUTE:
            ক্ষেত্র GGML_OP_TRANSPOSE:
                বিরতি;

            পূর্বনির্ধারিত:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(গ্রন্থি));
        }
    }

    ফেরত GGML_STATUS_SUCCESS;

    GGML_UNUSED(backend);
}

স্থির কাঠামো ggml_backend_i ggml_backend_zendnn_i = {
    /* .get_name                = */ ggml_backend_zendnn_get_name,
    /* .free                    = */ ggml_backend_zendnn_free,
    /* .set_tensor_async        = */ ফাঁকা,
    /* .get_tensor_async        = */ ফাঁকা,
    /* .set_tensor_2d_async     = */ ফাঁকা,
    /* .get_tensor_2d_async     = */ ফাঁকা,
    /* .cpy_tensor_async        = */ ফাঁকা,
    /* .synchronize             = */ ফাঁকা,
    /* .graph_plan_create       = */ ফাঁকা,
    /* .graph_plan_free         = */ ফাঁকা,
    /* .graph_plan_update       = */ ফাঁকা,
    /* .graph_plan_compute      = */ ফাঁকা,
    /* .graph_compute           = */ ggml_backend_zendnn_graph_compute,
    /* .event_record            = */ ফাঁকা,
    /* .event_wait              = */ ফাঁকা,
    /* .graph_optimize          = */ ফাঁকা,
};

স্থির ggml_guid_t ggml_backend_zendnn_guid(শূন্য) {
    স্থির ধ্রুবক অক্ষর * guid_str = "AMD-ZENDNN-ACCEL";
    ফেরত reinterpret_cast<ggml_guid_t>(const_cast<অক্ষর*>(guid_str));
}

ggml_backend_t ggml_backend_zendnn_init(শূন্য) {
    ggml_backend_zendnn_context * ctx = নতুন ggml_backend_zendnn_context;

    ggml_backend_t backend = নতুন ggml_backend {
        /* .guid    = */ ggml_backend_zendnn_guid(),
        /* .iface   = */ ggml_backend_zendnn_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_zendnn_reg(), 0),
        /* .context = */ ctx,
    };

    ফেরত backend;
}

যৌক্তিক ggml_backend_is_zendnn(ggml_backend_t backend) {
    ফেরত backend != ফাঁকা && ggml_guid_matches(backend->guid, ggml_backend_zendnn_guid());
}

শূন্য ggml_backend_zendnn_set_n_threads(ggml_backend_t backend_zendnn, পূর্ণসংখ্যা n_threads) {
    GGML_ASSERT(ggml_backend_is_zendnn(backend_zendnn));

    ggml_backend_zendnn_context * ctx = (ggml_backend_zendnn_context *)backend_zendnn->context;
    ctx->n_threads = n_threads;
}

// device interface
স্থির ধ্রুবক অক্ষর * ggml_backend_zendnn_device_get_name(ggml_backend_dev_t dev) {
    ফেরত "ZenDNN";

    GGML_UNUSED(dev);
}
/**
 * ZenDNN is AMD's performance library providing optimized primitives and implementations
 * for deep learning workloads on AMD CPUs. It targets improved performance for common
 * neural network operations on AMD architectures. For more information, see:
 * https://www.amd.com/en/developer/zendnn.html
 */
স্থির ধ্রুবক অক্ষর * ggml_backend_zendnn_device_get_description(ggml_backend_dev_t dev) {
    ফেরত "ZenDNN: AMD optimized primitives backend for GGML (optimized for AMD CPUs)";

    GGML_UNUSED(dev);
}

স্থির শূন্য ggml_backend_zendnn_device_get_memory(ggml_backend_dev_t dev, size_t * মুক্ত, size_t * যোগ) {
    *মুক্ত  = 0;
    *যোগ = 0;

    GGML_UNUSED(dev);
}

স্থির গণনা ggml_backend_dev_type ggml_backend_zendnn_device_get_type(ggml_backend_dev_t dev) {
    ফেরত GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_UNUSED(dev);
}

স্থির শূন্য ggml_backend_zendnn_device_get_props(ggml_backend_dev_t dev, কাঠামো ggml_backend_dev_props * props) {
    props->name        = ggml_backend_zendnn_device_get_name(dev);
    props->description = ggml_backend_zendnn_device_get_description(dev);
    props->type        = ggml_backend_zendnn_device_get_type(dev);
    ggml_backend_zendnn_device_get_memory(dev, &props->memory_free, &props->memory_যোগ);
    props->caps = {
        /* .async                = */ মিথ্যা,
        /* .host_buffer          = */ মিথ্যা,
        /* .buffer_from_host_ptr = */ সত্য,
        /* .events               = */ মিথ্যা,
        /* .mmap_support         = */ সত্য,
    };
}

স্থির ggml_backend_t ggml_backend_zendnn_device_init_backend(ggml_backend_dev_t dev, ধ্রুবক অক্ষর * params) {
    ggml_backend_t backend = ggml_backend_zendnn_init();
    যদি (backend == ফাঁকা) {
        GGML_LOG_ERROR("%s: error: failed to initialize ZenDNN backend\n", __func__);
        ফেরত ফাঁকা;
    }

    ফেরত backend;

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

স্থির ggml_backend_বাফার_type_t ggml_backend_zendnn_device_get_বাফার_type(ggml_backend_dev_t dev) {
    ফেরত ggml_backend_cpu_বাফার_type();

    GGML_UNUSED(dev);
}

স্থির ggml_backend_বাফার_t ggml_backend_zendnn_device_বাফার_from_host_ptr(ggml_backend_dev_t dev, শূন্য * ptr, size_t size, size_t max_tensor_আকার) {
    ফেরত ggml_backend_cpu_বাফার_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_আকার);
}

স্থির যৌক্তিক ggml_zendnn_adaptive_fallback_enabled() {
    স্থির ধ্রুবক যৌক্তিক enabled = std::getenv("GGML_ZENDNN_ADAPTIVE_FALLBACK") == nullptr ||
                                std::atoi(std::getenv("GGML_ZENDNN_ADAPTIVE_FALLBACK")) != 0;
    ফেরত enabled;
}

স্থির যৌক্তিক ggml_backend_zendnn_device_supports_op(ggml_backend_dev_t dev, ধ্রুবক কাঠামো ggml_tensor * op) {
    নির্বাচন (op->op) {
        ক্ষেত্র GGML_OP_NONE:
        ক্ষেত্র GGML_OP_RESHAPE:
        ক্ষেত্র GGML_OP_VIEW:
        ক্ষেত্র GGML_OP_PERMUTE:
        ক্ষেত্র GGML_OP_TRANSPOSE:
            ফেরত সত্য;

        ক্ষেত্র GGML_OP_MUL_MAT:
        ক্ষেত্র GGML_OP_MUL_MAT_ID:
        {
            ধ্রুবক ggml_tensor * weights = op->src[0];
            ধ্রুবক ggml_tensor * inputs = op->src[1];

            ধ্রুবক int64_t ne10 = inputs->ne[0];
            ধ্রুবক int64_t ne0 = op->ne[0];
            ধ্রুবক int64_t ne1 = op->ne[1];
            ধ্রুবক int64_t min_batch = 1;

            যদি(!ggml_is_contiguous(weights) || !ggml_is_contiguous(inputs)) {
                ফেরত মিথ্যা;
            }

            যদি (ggml_zendnn_adaptive_fallback_enabled()) {
                ধ্রুবক int64_t K = inputs->ne[0];
                ধ্রুবক int64_t N = (inputs->ne[1]*inputs->ne[2]*inputs->ne[3]);
                ধ্রুবক int64_t M = weights->ne[1];
                যদি(K <= 256 || N <= 128 || M <= 96) {
                    ফেরত মিথ্যা;
                }

                // MUL_MAT_ID's gather+matmul+scatter approach favors a moderate expert count
                যদি (op->op == GGML_OP_MUL_MAT_ID) {
                    ধ্রুবক int64_t n_experts = weights->ne[2];
                    ধ্রুবক int64_t max_experts = 32;
                    যদি (n_experts > max_experts) {
                        ফেরত মিথ্যা;
                    }

                    // fall back once the average rows per expert (N / n_experts) is too thin
                    // to amortize each per-expert GEMM's overhead
                    যদি (N / n_experts <= 32) {
                        ফেরত মিথ্যা;
                    }
                }
            }
            অন্যথা যদি (ne0 < min_batch || ne1 < min_batch || ne10 < min_batch) {
                ফেরত মিথ্যা;
            }

            নির্বাচন (weights->type) {
                ক্ষেত্র GGML_TYPE_F32:
                ক্ষেত্র GGML_TYPE_BF16:
                ক্ষেত্র GGML_TYPE_Q8_0:
                    ফেরত সত্য;
                পূর্বনির্ধারিত:
                    ফেরত মিথ্যা;
            }
        } বিরতি;

        পূর্বনির্ধারিত:
            ফেরত মিথ্যা;
    }

    GGML_UNUSED(dev);
}

স্থির যৌক্তিক ggml_backend_zendnn_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_বাফার_type_t buft) {
    ফেরত ggml_backend_buft_is_host(buft);

    GGML_UNUSED(dev);
}

স্থির ধ্রুবক কাঠামো ggml_backend_device_i ggml_backend_zendnn_device_i = {
    /* .get_name               = */ ggml_backend_zendnn_device_get_name,
    /* .get_description        = */ ggml_backend_zendnn_device_get_description,
    /* .get_memory             = */ ggml_backend_zendnn_device_get_memory,
    /* .get_type               = */ ggml_backend_zendnn_device_get_type,
    /* .get_props              = */ ggml_backend_zendnn_device_get_props,
    /* .init_backend           = */ ggml_backend_zendnn_device_init_backend,
    /* .get_buffer_type        = */ ggml_backend_zendnn_device_get_বাফার_type,
    /* .get_host_buffer_type   = */ ফাঁকা,
    /* .buffer_from_host_ptr   = */ ggml_backend_zendnn_device_বাফার_from_host_ptr,
    /* .supports_op            = */ ggml_backend_zendnn_device_supports_op,
    /* .supports_buft          = */ ggml_backend_zendnn_device_supports_buft,
    /* .offload_op             = */ ফাঁকা,
    /* .event_new              = */ ফাঁকা,
    /* .event_free             = */ ফাঁকা,
    /* .event_synchronize      = */ ফাঁকা,
};

// backend reg interface
স্থির ধ্রুবক অক্ষর * ggml_backend_zendnn_reg_get_name(ggml_backend_reg_t reg) {
    ফেরত "ZenDNN";

    GGML_UNUSED(reg);
}

স্থির size_t ggml_backend_zendnn_reg_get_device_গুনতি(ggml_backend_reg_t reg) {
    ফেরত 1;

    GGML_UNUSED(reg);
}

স্থির ggml_backend_dev_t ggml_backend_zendnn_reg_get_device(ggml_backend_reg_t reg, size_t সূচক) {
    GGML_ASSERT(সূচক == 0);

    স্থির ggml_backend_device ggml_backend_zendnn_device = {
        /* .iface   = */ ggml_backend_zendnn_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };

    ফেরত &ggml_backend_zendnn_device;
}

স্থির শূন্য * ggml_backend_zendnn_get_proc_address(ggml_backend_reg_t reg, ধ্রুবক অক্ষর * name) {
    যদি (std::strcmp(name, "ggml_backend_set_n_threads") == 0) {
        ফেরত (শূন্য *) ggml_backend_zendnn_set_n_threads;
    }
    ফেরত ফাঁকা;

    GGML_UNUSED(reg);
    GGML_UNUSED(name);
}

স্থির ধ্রুবক কাঠামো ggml_backend_reg_i ggml_backend_zendnn_reg_i = {
    /* .get_name         = */ ggml_backend_zendnn_reg_get_name,
    /* .get_device_count = */ ggml_backend_zendnn_reg_get_device_গুনতি,
    /* .get_device       = */ ggml_backend_zendnn_reg_get_device,
    /* .get_proc_address = */ ggml_backend_zendnn_get_proc_address,
};

ggml_backend_reg_t ggml_backend_zendnn_reg(শূন্য) {
    স্থির কাঠামো ggml_backend_reg ggml_backend_zendnn_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_zendnn_reg_i,
        /* .context     = */ ফাঁকা,
    };

    ফেরত &ggml_backend_zendnn_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_zendnn_reg)
