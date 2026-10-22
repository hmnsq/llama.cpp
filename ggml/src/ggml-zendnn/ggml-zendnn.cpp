#شامل "ggml-zendnn.h"

#شامل "ggml-backend-impl.h"
#شامل "ggml-impl.h"

#define GGML_COMMON_DECL_CPP
#شامل "ggml-common.h"

#شامل "zendnnl.hpp"

#شامل <cstring>


ساختار ggml_backend_zendnn_context {
    صحیح n_threads = GGML_DEFAULT_N_THREADS;
    std::unique_ptr<نویسه[]> work_data;
    size_t work_size = 0;
};

template<typename T>
zendnnl::common::data_type_t ggml_to_zendnn_type() {
    اگر constexpr (std::is_same_v<T, اعشاری>) {
        بازگشت zendnnl::common::data_type_t::f32;
    } وگرنه اگر constexpr (std::is_same_v<T, ggml_bf16_t>) {
        بازگشت zendnnl::common::data_type_t::bf16;
    } وگرنه اگر constexpr (std::is_same_v<T, block_q8_0>) {
        بازگشت zendnnl::common::data_type_t::s8;
    } وگرنه {
        بازگشت zendnnl::common::data_type_t::none;
    }
}

/**
 * Builds the matmul_params shared by ggml_zendnn_matmul() and ggml_zendnn_group_matmul():
 * dtype selection plus, for Q8_0 weights, dynamic-quant setup. Callers still need to set
 * quant_params.src_scale.dims themselves, since that depends on the batch size(s) in use.
 */
template <typename TA, typename TB, typename TC>
ایستا zendnnl::lowoha::matmul::matmul_params ggml_zendnn_make_matmul_params(ggml_backend_zendnn_context * ctx) {
    zendnnl::lowoha::matmul::matmul_params params;
    params.dtypes.src = ggml_to_zendnn_type<TB>();
    params.dtypes.wei = ggml_to_zendnn_type<TA>();
    params.dtypes.dst = ggml_to_zendnn_type<TC>();
    params.رقم_threads = ctx->n_threads;

    اگر constexpr (std::is_same_v<TA, block_q8_0>) {
        params.dtypes.compute = zendnnl::common::data_type_t::s8;
        params.dynamic_quant = درست;
        params.quant_params.src_scale.buff = nullptr;
        params.quant_params.src_scale.dt   = zendnnl::common::data_type_t::bf16;
        params.packing.pack_format_b = 1;
    }
    بازگشت params;
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
ایستا منطقی ggml_zendnn_matmul(ggml_backend_zendnn_context * ctx, int64_t m, int64_t n, int64_t k,
                               ثابت TA * A, int64_t lda, ثابت TB * B, int64_t ldb, TC * C,
                               int64_t ldc) {

    zendnnl::lowoha::matmul::matmul_params params = ggml_zendnn_make_matmul_params<TA, TB, TC>(ctx);

    zendnnl::lowoha::matmul::matmul_batch_params_t batch_params;

    اگر constexpr (std::is_same_v<TA, block_q8_0>) {
        params.quant_params.src_scale.dims = {n, k / QK8_0};
    }

    zendnnl::error_handling::status_t status = zendnnl::lowoha::matmul::matmul_direct(
        'r', نادرست, درست,   // row-major, don't transpose B, transpose A (because it's column-major)
        n,                  // M: rows of B and C
        m,                  // N: cols of A^T and C
        k,                  // K: cols of B, rows of A
        1.0f,               // alpha
        B, ldb,             // src: B[n,k]
        A, lda,             // weight: A[k,m] column-major (transposed)
        nullptr,            // bias
        0.0f,               // beta
        C, ldc,             // output C[n,m]
        درست,               // is_weights_const
        batch_params,       // batch_params
        params              // params
    );

    اگر (status != zendnnl::error_handling::status_t::success) {
        GGML_LOG_ERROR("%s, ZenDNN matmul failed: status=%d\n", __func__, static_cast<صحیح>(status));
        بازگشت نادرست;
    }
    بازگشت درست;
}

ایستا منطقی ggml_zendnn_gemm(ggml_backend_zendnn_context * ctx, int64_t m, int64_t n, int64_t k,
                              ثابت تهی * A, int64_t lda, ثابت تهی * B, int64_t ldb, تهی * C,
                              int64_t ldc, صحیح Atype, صحیح Btype, صحیح Ctype) {

    أكد(m >= 0);
    أكد(n >= 0);
    أكد(k >= 0);
    أكد(lda >= k);
    أكد(ldb >= k);
    أكد(ldc >= m);

    // categorize types
    انتخاب (Atype) {
        حالت GGML_TYPE_F32:
            اگر (Btype != GGML_TYPE_F32 || Ctype != GGML_TYPE_F32)
                بازگشت نادرست;
            بازگشت ggml_zendnn_matmul<اعشاری, اعشاری, اعشاری>(
                ctx, m, n, k,
                (ثابت اعشاری *)A, lda,
                (ثابت اعشاری *)B, ldb,
                (اعشاری *)C, ldc);
        حالت GGML_TYPE_BF16:
            اگر (Btype != GGML_TYPE_BF16)
                بازگشت نادرست;
            اگر (Ctype == GGML_TYPE_BF16)
                بازگشت ggml_zendnn_matmul<ggml_bf16_t, ggml_bf16_t, ggml_bf16_t>(
                    ctx, m, n, k,
                    (ثابت ggml_bf16_t *)A, lda,
                    (ثابت ggml_bf16_t *)B, ldb,
                    (ggml_bf16_t *)C, ldc);
            اگر (Ctype == GGML_TYPE_F32)
                بازگشت ggml_zendnn_matmul<ggml_bf16_t, ggml_bf16_t, اعشاری>(
                    ctx, m, n, k,
                    (ثابت ggml_bf16_t *)A, lda,
                    (ثابت ggml_bf16_t *)B, ldb,
                    (اعشاری *)C, ldc);
            بازگشت نادرست;
        حالت GGML_TYPE_Q8_0:
            اگر (Btype != GGML_TYPE_F32 || Ctype != GGML_TYPE_F32)
                بازگشت نادرست;
            بازگشت ggml_zendnn_matmul<block_q8_0, اعشاری, اعشاری>(
                ctx, m, n, k,
                (ثابت block_q8_0 *)A, lda,
                (ثابت اعشاری *)B, ldb,
                (اعشاری *)C, ldc);
        پیشفرض:
            بازگشت نادرست; // unsupported type
    }
}

ایستا تهی ggml_zendnn_compute_forward_mul_mat(
    ggml_backend_zendnn_context * ctx,
    ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];  // weights
    ثابت ggml_tensor * src1 = dst->src[1];  // inputs

    GGML_TENSOR_دودویی_OP_LOCALS

    ggml_type         ثابت vec_dot_type = src0->type;
    ggml_from_float_t ثابت from_float = ggml_get_type_traits(vec_dot_type)->from_float_ref;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == اندازه(اعشاری));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // broadcast factors
    ثابت int64_t r2 = ne12/ne02;
    ثابت int64_t r3 = ne13/ne03;

    تهی * work_data = ctx->work_data.get();

    // ZenDNN requires FP32 for dynamic quantization, so conversion is skipped
    اگر (src1->type != vec_dot_type && src0->type != GGML_TYPE_Q8_0) {
        ثابت size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
        ثابت size_t nbw2 = nbw1 * ne11;
        ثابت size_t nbw3 = nbw2 * ne12;
        ثابت size_t desired_wsize = ne13 * nbw3;
        اگر (ctx->work_size < desired_wsize) {
            ctx->work_data.reset(جدید نویسه[desired_wsize]);
            ctx->work_size = desired_wsize;
        }
        work_data = ctx->work_data.get();

        // #pragma omp parallel for num_threads(ctx->n_threads)
        #pragma omp parallel برای collapse(3) رقم_threads(ctx->n_threads) schedule(ایستا)
        برای (int64_t i13 = 0; i13 < ne13; ++i13) {
            برای (int64_t i12 = 0; i12 < ne12; ++i12) {
                برای (int64_t i11 = 0; i11 < ne11; ++i11) {
                    ثابت اعشاری * src1_f32 = (اعشاری *)((نویسه *)src1->data + i11*nb11 + i12*nb12 + i13*nb13);
                    تهی * src1_conv = (نویسه *)work_data + i11*nbw1 + i12*nbw2 + i13*nbw3;
                    from_float(src1_f32, src1_conv, ne10);
                }
            }
        }
    }

    برای (int64_t i13 = 0; i13 < ne13; i13++) {
        برای (int64_t i12 = 0; i12 < ne12; i12++) {
            ثابت تهی* wdata = (src1->type == vec_dot_type || src0->type == GGML_TYPE_Q8_0) ? src1->data : work_data;
            ثابت size_t row_size = ggml_row_size(vec_dot_type, ne10);
            اگر (!ggml_zendnn_gemm(ctx,
                                  ne01,     // m
                                  ne11,     // n
                                  ne10,     // k
                                  static_cast<ثابت نویسه *>(src0->data) + (i12/r2)*nb02 + (i13/r3)*nb03,
                                  ne00,     // lda
                                  static_cast<ثابت نویسه *>(wdata) + (i12*ne11 + i13*ne12*ne11)*row_size,
                                  ne10,     // ldb
                                  static_cast<نویسه *>(dst->data) + i12*nb2 + i13*nb3,
                                  ne01,     // ldc
                                  src0->type,
                                  src0->type == GGML_TYPE_Q8_0 ? GGML_TYPE_F32 : vec_dot_type,
                                  dst->type))
                GGML_ABORT("%s: ZenDNN gemm failed\n", __func__);
        }
    }
}

ساختار mmid_row_mapping {
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
ایستا منطقی ggml_zendnn_group_matmul(ggml_backend_zendnn_context * ctx, int64_t m, int64_t k,
                                     ثابت std::vector<int64_t> & n,
                                     ثابت std::vector<ثابت تهی *> & A, int64_t lda,
                                     ثابت std::vector<ثابت تهی *> & B, int64_t ldb,
                                     ثابت std::vector<تهی *> & C, int64_t ldc) {

    ثابت صحیح n_experts = n.size();

    zendnnl::lowoha::matmul::matmul_params base_params = ggml_zendnn_make_matmul_params<TA, TB, TC>(ctx);

    std::vector<نویسه>         layout(n_experts, 'r');
    std::vector<منطقی>         trans_a(n_experts, نادرست);
    std::vector<منطقی>         trans_b(n_experts, درست);
    std::vector<صحیح>          batch_m(n_experts);
    std::vector<صحیح>          batch_n(n_experts, m);
    std::vector<صحیح>          batch_k(n_experts, k);
    std::vector<اعشاری>        alpha(n_experts, 1.0f);
    std::vector<اعشاری>        beta(n_experts, 0.0f);
    std::vector<ثابت تهی *> bias(n_experts, nullptr);
    std::vector<صحیح>          lda_v(n_experts, lda);
    std::vector<صحیح>          ldb_v(n_experts, ldb);
    std::vector<صحیح>          ldc_v(n_experts, ldc);
    std::vector<منطقی>         is_wei_const(n_experts, درست);
    std::vector<zendnnl::lowoha::matmul::matmul_params> params(n_experts, base_params);

    برای (صحیح i = 0; i < n_experts; i++) {
        batch_m[i] = n[i];

        // src_scale.dims depends on this expert's row count, unlike the rest of base_params
        اگر constexpr (std::is_same_v<TA, block_q8_0>) {
            params[i].quant_params.src_scale.dims = {n[i], k / QK8_0};
        }
    }

    zendnnl::error_handling::status_t status = zendnnl::lowoha::matmul::group_matmul_direct(
        layout, trans_a, trans_b, batch_m, batch_n, batch_k, alpha,
        B, ldb_v, A, lda_v, bias, beta,
        C, ldc_v, is_wei_const, params);

    اگر (status != zendnnl::error_handling::status_t::success) {
        GGML_LOG_ERROR("%s, ZenDNN group matmul failed: status=%d\n", __func__, static_cast<صحیح>(status));
        بازگشت نادرست;
    }
    بازگشت درست;
}

ایستا منطقی ggml_zendnn_group_gemm(ggml_backend_zendnn_context * ctx, int64_t m, int64_t k,
                                    ثابت std::vector<int64_t> & n,
                                    ثابت std::vector<ثابت تهی *> & A, int64_t lda,
                                    ثابت std::vector<ثابت تهی *> & B, int64_t ldb,
                                    ثابت std::vector<تهی *> & C, int64_t ldc,
                                    صحیح Atype, صحیح Btype, صحیح Ctype) {

    أكد(m >= 0);
    برای (size_t i = 0; i < n.size(); i++) {
        أكد(n[i] >= 0);
    }
    أكد(k >= 0);
    أكد(lda >= k);
    أكد(ldb >= k);
    أكد(ldc >= m);

    // categorize types
    انتخاب (Atype) {
        حالت GGML_TYPE_F32:
            اگر (Btype != GGML_TYPE_F32 || Ctype != GGML_TYPE_F32)
                بازگشت نادرست;
            بازگشت ggml_zendnn_group_matmul<اعشاری, اعشاری, اعشاری>(ctx, m, k, n, A, lda, B, ldb, C, ldc);
        حالت GGML_TYPE_BF16:
            اگر (Btype != GGML_TYPE_BF16)
                بازگشت نادرست;
            اگر (Ctype == GGML_TYPE_BF16)
                بازگشت ggml_zendnn_group_matmul<ggml_bf16_t, ggml_bf16_t, ggml_bf16_t>(
                    ctx, m, k, n, A, lda, B, ldb, C, ldc);
            اگر (Ctype == GGML_TYPE_F32)
                بازگشت ggml_zendnn_group_matmul<ggml_bf16_t, ggml_bf16_t, اعشاری>(ctx, m, k, n, A, lda, B, ldb, C, ldc);
            بازگشت نادرست;
        حالت GGML_TYPE_Q8_0:
            اگر (Btype != GGML_TYPE_F32 || Ctype != GGML_TYPE_F32)
                بازگشت نادرست;
            بازگشت ggml_zendnn_group_matmul<block_q8_0, اعشاری, اعشاری>(ctx, m, k, n, A, lda, B, ldb, C, ldc);
        پیشفرض:
            بازگشت نادرست; // unsupported type
    }
}

ایستا تهی ggml_zendnn_compute_forward_mul_mat_id(
    ggml_backend_zendnn_context * ctx,
    ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];  // expert weights
    ثابت ggml_tensor * src1 = dst->src[1];  // inputs
    ثابت ggml_tensor * ids  = dst->src[2];  // expert ids

    GGML_TENSOR_دودویی_OP_LOCALS

    // exit for no tokens to process
    اگر (ne2 == 0 || ne11 == 0) {
        بازگشت;
    }

    ggml_type         ثابت vec_dot_type = src0->type;
    ggml_from_float_t ثابت from_float = ggml_get_type_traits(vec_dot_type)->from_float_ref;

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == اندازه(اعشاری));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(ne03 == 1);
    GGML_ASSERT(ne13 == 1);
    GGML_ASSERT(ne3  == 1);

    // row groups
    ثابت صحیح n_ids = ids->ne[0]; // n_expert_used
    ثابت صحیح n_as  = ne02;       // n_experts

    std::vector<int64_t> matrix_row_counts(n_as, 0);
    std::vector<std::vector<mmid_row_mapping>> matrix_rows(n_as);

    int64_t total_rows = 0;
    صحیح     n_active_experts = 0;
    // group rows by expert (preprocessing step)
    برای (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
        برای (صحیح id = 0; id < n_ids; ++id) {
            ثابت int32_t i02 = *(ثابت int32_t *)((ثابت نویسه *)ids->data + iid1*ids->nb[1] + id*ids->nb[0]);

            GGML_ASSERT(i02 >= 0 && i02 < n_as);

            اگر (matrix_row_counts[i02] == 0) {
                n_active_experts++;
            }
            matrix_rows[i02].push_back({id, iid1});
            matrix_row_counts[i02]++;
            total_rows++;
        }
    }

    اگر (total_rows == 0) {
        بازگشت; // no rows to process
    }

    ثابت size_t row_size = ggml_row_size(vec_dot_type, ne10);

    // For Q8_0, src1 is always F32; the gather buffer must hold F32 rows (ne10*4 bytes),
    // not Q8_0-encoded rows (row_size ≈ ne10/32*34 bytes) — they differ by ~4x.
    ثابت size_t f32_row_size = (size_t)ne10 * اندازه(اعشاری);
    ثابت size_t gather_row_size = (src0->type == GGML_TYPE_Q8_0) ? f32_row_size : row_size;

    اگر (src1->type != vec_dot_type && src0->type != GGML_TYPE_Q8_0) {
        GGML_ASSERT(src1->type == GGML_TYPE_F32);
    }

    // size for MoE gather/scatter buffers
    ثابت size_t wdata_cur_size = total_rows * gather_row_size;
    ثابت size_t dst_cur_size = total_rows * ggml_row_size(dst->type, ne01);

    // allocate single buffer for all needs
    ثابت size_t total_size = wdata_cur_size + dst_cur_size;
    اگر (ctx->work_size < total_size) {
        ctx->work_data.reset(جدید نویسه[total_size]);
        ctx->work_size = total_size;
    }

    // partition the buffer
    نویسه * wdata_cur = ctx->work_data.get();
    نویسه * dst_cur = wdata_cur + wdata_cur_size;

    // per-expert data collected during gather, handed to ggml_zendnn_group_gemm() as one batch
    std::vector<int64_t>      expert_row_شمار(n_active_experts);
    std::vector<ثابت تهی *> batch_src(n_active_experts);
    std::vector<ثابت تهی *> batch_wei(n_active_experts);
    std::vector<تهی *>       batch_dst(n_active_experts);

    // precompute per-expert buffer offsets and batch indices for the parallel loop below
    std::vector<int64_t> expert_wdata_off(n_as, 0);
    std::vector<int64_t> expert_dst_off(n_as, 0);
    std::vector<صحیح>     expert_batch_idx(n_as, -1);
    {
        int64_t w_off = 0;
        int64_t d_off = 0;
        صحیح     batch_idx = 0;
        برای (int64_t cur_a = 0; cur_a < n_as; ++cur_a) {
            اگر (matrix_row_counts[cur_a] == 0) {
                ادامه;
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
    #pragma omp parallel برای رقم_threads(ctx->n_threads) schedule(ایستا)
    برای (int64_t cur_a = 0; cur_a < n_as; ++cur_a) {
        ثابت int64_t cne1 = matrix_row_counts[cur_a];

        اگر (cne1 == 0) {
            ادامه;
        }

        ثابت int64_t w_off     = expert_wdata_off[cur_a];
        ثابت int64_t d_off     = expert_dst_off[cur_a];
        ثابت صحیح     batch_idx = expert_batch_idx[cur_a];

        برای (int64_t ir1 = 0; ir1 < cne1; ++ir1) {
            ثابت mmid_row_mapping & row_mapping = matrix_rows[cur_a][ir1];
            ثابت int64_t id  = row_mapping.i1;
            ثابت int64_t i11 = id % ne11;
            ثابت int64_t i12 = row_mapping.i2;

            ثابت نویسه * src_row = (ثابت نویسه *) src1->data + i11*nb11 + i12*nb12;
            تهی *       dst_row = wdata_cur + w_off + ir1 * gather_row_size;

            اگر (src1->type != vec_dot_type && src0->type != GGML_TYPE_Q8_0) {
                from_float((ثابت اعشاری *) src_row, dst_row, ne10);
            } وگرنه {
                // no conversion: src1 already matches vec_dot_type, or src0 is Q8_0, whose
                // ZenDNN dynamic quantization requires the row to stay in F32
                std::memcpy(dst_row, src_row, gather_row_size);
            }
        }

        expert_row_شمار[batch_idx] = cne1;
        batch_src[batch_idx]        = wdata_cur + w_off;
        batch_wei[batch_idx]        = (ثابت نویسه *) src0->data + cur_a * nb02;
        batch_dst[batch_idx]        = dst_cur + d_off;
    }

    اگر (!ggml_zendnn_group_gemm(ctx,
                                ne01,             // m
                                ne10,             // k
                                expert_row_شمار, // n (per expert)
                                batch_wei, ne00,  // A: weights, lda
                                batch_src, ne10,  // B: input, ldb
                                batch_dst, ne01,  // C: output, ldc
                                src0->type,
                                src0->type == GGML_TYPE_Q8_0 ? GGML_TYPE_F32 : vec_dot_type,
                                dst->type))
        GGML_ABORT("%s: ZenDNN group gemm failed\n", __func__);

    // scatter output rows to destination
    #pragma omp parallel برای رقم_threads(ctx->n_threads) schedule(ایستا)
    برای (int64_t cur_a = 0; cur_a < n_as; ++cur_a) {
        ثابت int64_t cne1 = matrix_row_counts[cur_a];

        اگر (cne1 == 0) {
            ادامه;
        }

        ثابت int64_t d_off = expert_dst_off[cur_a];

        برای (int64_t ir1 = 0; ir1 < cne1; ++ir1) {
            ثابت mmid_row_mapping & row_mapping = matrix_rows[cur_a][ir1];
            ثابت int64_t id = row_mapping.i1;
            ثابت int64_t i1 = id;
            ثابت int64_t i2 = row_mapping.i2;

            std::memcpy(
                (نویسه *) dst->data + i1*nb1 + i2*nb2,
                dst_cur + d_off + ir1 * ggml_row_size(dst->type, ne01),
                ggml_row_size(dst->type, ne01)
            );
        }
    }
}

// backend interface

ایستا ثابت نویسه * ggml_backend_zendnn_get_name(ggml_backend_t backend) {
    بازگشت "ZenDNN";

    GGML_UNUSED(backend);
}

ایستا تهی ggml_backend_zendnn_free(ggml_backend_t backend) {
    ggml_backend_zendnn_context * ctx = (ggml_backend_zendnn_context *)backend->context;
    delete ctx;
    delete backend;
}

ایستا ggml_status ggml_backend_zendnn_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_zendnn_context * ctx = (ggml_backend_zendnn_context *)backend->context;

    برای (صحیح i = 0; i < cgraph->n_nodes; i++) {
        ساختار ggml_tensor * گره = cgraph->nodes[i];

        اگر ((گره->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            ادامه;
        }

        انتخاب (گره->op) {
            حالت GGML_OP_MUL_MAT:
                ggml_zendnn_compute_forward_mul_mat(ctx, گره);
                بشکن;
            حالت GGML_OP_MUL_MAT_ID:
                ggml_zendnn_compute_forward_mul_mat_id(ctx, گره);
                بشکن;
            حالت GGML_OP_NONE:
            حالت GGML_OP_RESHAPE:
            حالت GGML_OP_VIEW:
            حالت GGML_OP_PERMUTE:
            حالت GGML_OP_TRANSPOSE:
                بشکن;

            پیشفرض:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(گره));
        }
    }

    بازگشت GGML_STATUS_SUCCESS;

    GGML_UNUSED(backend);
}

ایستا ساختار ggml_backend_i ggml_backend_zendnn_i = {
    /* .get_name                = */ ggml_backend_zendnn_get_name,
    /* .free                    = */ ggml_backend_zendnn_free,
    /* .set_tensor_async        = */ پوچ,
    /* .get_tensor_async        = */ پوچ,
    /* .set_tensor_2d_async     = */ پوچ,
    /* .get_tensor_2d_async     = */ پوچ,
    /* .cpy_tensor_async        = */ پوچ,
    /* .synchronize             = */ پوچ,
    /* .graph_plan_create       = */ پوچ,
    /* .graph_plan_free         = */ پوچ,
    /* .graph_plan_update       = */ پوچ,
    /* .graph_plan_compute      = */ پوچ,
    /* .graph_compute           = */ ggml_backend_zendnn_graph_compute,
    /* .event_record            = */ پوچ,
    /* .event_wait              = */ پوچ,
    /* .graph_optimize          = */ پوچ,
};

ایستا ggml_guid_t ggml_backend_zendnn_guid(تهی) {
    ایستا ثابت نویسه * guid_str = "AMD-ZENDNN-ACCEL";
    بازگشت reinterpret_cast<ggml_guid_t>(const_cast<نویسه*>(guid_str));
}

ggml_backend_t ggml_backend_zendnn_init(تهی) {
    ggml_backend_zendnn_context * ctx = جدید ggml_backend_zendnn_context;

    ggml_backend_t backend = جدید ggml_backend {
        /* .guid    = */ ggml_backend_zendnn_guid(),
        /* .iface   = */ ggml_backend_zendnn_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_zendnn_reg(), 0),
        /* .context = */ ctx,
    };

    بازگشت backend;
}

منطقی ggml_backend_is_zendnn(ggml_backend_t backend) {
    بازگشت backend != پوچ && ggml_guid_matches(backend->guid, ggml_backend_zendnn_guid());
}

تهی ggml_backend_zendnn_set_n_threads(ggml_backend_t backend_zendnn, صحیح n_threads) {
    GGML_ASSERT(ggml_backend_is_zendnn(backend_zendnn));

    ggml_backend_zendnn_context * ctx = (ggml_backend_zendnn_context *)backend_zendnn->context;
    ctx->n_threads = n_threads;
}

// device interface
ایستا ثابت نویسه * ggml_backend_zendnn_device_get_name(ggml_backend_dev_t dev) {
    بازگشت "ZenDNN";

    GGML_UNUSED(dev);
}
/**
 * ZenDNN is AMD's performance library providing optimized primitives and implementations
 * for deep learning workloads on AMD CPUs. It targets improved performance for common
 * neural network operations on AMD architectures. For more information, see:
 * https://www.amd.com/en/developer/zendnn.html
 */
ایستا ثابت نویسه * ggml_backend_zendnn_device_get_description(ggml_backend_dev_t dev) {
    بازگشت "ZenDNN: AMD optimized primitives backend for GGML (optimized for AMD CPUs)";

    GGML_UNUSED(dev);
}

ایستا تهی ggml_backend_zendnn_device_get_memory(ggml_backend_dev_t dev, size_t * آزاد, size_t * total) {
    *آزاد  = 0;
    *total = 0;

    GGML_UNUSED(dev);
}

ایستا شمارش ggml_backend_dev_type ggml_backend_zendnn_device_get_type(ggml_backend_dev_t dev) {
    بازگشت GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_UNUSED(dev);
}

ایستا تهی ggml_backend_zendnn_device_get_props(ggml_backend_dev_t dev, ساختار ggml_backend_dev_props * props) {
    props->name        = ggml_backend_zendnn_device_get_name(dev);
    props->description = ggml_backend_zendnn_device_get_description(dev);
    props->type        = ggml_backend_zendnn_device_get_type(dev);
    ggml_backend_zendnn_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                = */ نادرست,
        /* .host_buffer          = */ نادرست,
        /* .buffer_from_host_ptr = */ درست,
        /* .events               = */ نادرست,
        /* .mmap_support         = */ درست,
    };
}

ایستا ggml_backend_t ggml_backend_zendnn_device_init_backend(ggml_backend_dev_t dev, ثابت نویسه * params) {
    ggml_backend_t backend = ggml_backend_zendnn_init();
    اگر (backend == پوچ) {
        GGML_LOG_ERROR("%s: error: failed to initialize ZenDNN backend\n", __func__);
        بازگشت پوچ;
    }

    بازگشت backend;

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

ایستا ggml_backend_buffer_type_t ggml_backend_zendnn_device_get_buffer_type(ggml_backend_dev_t dev) {
    بازگشت ggml_backend_cpu_buffer_type();

    GGML_UNUSED(dev);
}

ایستا ggml_backend_buffer_t ggml_backend_zendnn_device_buffer_from_host_ptr(ggml_backend_dev_t dev, تهی * ptr, size_t size, size_t max_tensor_size) {
    بازگشت ggml_backend_cpu_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

ایستا منطقی ggml_zendnn_adaptive_fallback_enabled() {
    ایستا ثابت منطقی enabled = std::getenv("GGML_ZENDNN_ADAPTIVE_FALLBACK") == nullptr ||
                                std::atoi(std::getenv("GGML_ZENDNN_ADAPTIVE_FALLBACK")) != 0;
    بازگشت enabled;
}

ایستا منطقی ggml_backend_zendnn_device_supports_op(ggml_backend_dev_t dev, ثابت ساختار ggml_tensor * op) {
    انتخاب (op->op) {
        حالت GGML_OP_NONE:
        حالت GGML_OP_RESHAPE:
        حالت GGML_OP_VIEW:
        حالت GGML_OP_PERMUTE:
        حالت GGML_OP_TRANSPOSE:
            بازگشت درست;

        حالت GGML_OP_MUL_MAT:
        حالت GGML_OP_MUL_MAT_ID:
        {
            ثابت ggml_tensor * weights = op->src[0];
            ثابت ggml_tensor * inputs = op->src[1];

            ثابت int64_t ne10 = inputs->ne[0];
            ثابت int64_t ne0 = op->ne[0];
            ثابت int64_t ne1 = op->ne[1];
            ثابت int64_t min_batch = 1;

            اگر(!ggml_is_contiguous(weights) || !ggml_is_contiguous(inputs)) {
                بازگشت نادرست;
            }

            اگر (ggml_zendnn_adaptive_fallback_enabled()) {
                ثابت int64_t K = inputs->ne[0];
                ثابت int64_t N = (inputs->ne[1]*inputs->ne[2]*inputs->ne[3]);
                ثابت int64_t M = weights->ne[1];
                اگر(K <= 256 || N <= 128 || M <= 96) {
                    بازگشت نادرست;
                }

                // MUL_MAT_ID's gather+matmul+scatter approach favors a moderate expert count
                اگر (op->op == GGML_OP_MUL_MAT_ID) {
                    ثابت int64_t n_experts = weights->ne[2];
                    ثابت int64_t max_experts = 32;
                    اگر (n_experts > max_experts) {
                        بازگشت نادرست;
                    }

                    // fall back once the average rows per expert (N / n_experts) is too thin
                    // to amortize each per-expert GEMM's overhead
                    اگر (N / n_experts <= 32) {
                        بازگشت نادرست;
                    }
                }
            }
            وگرنه اگر (ne0 < min_batch || ne1 < min_batch || ne10 < min_batch) {
                بازگشت نادرست;
            }

            انتخاب (weights->type) {
                حالت GGML_TYPE_F32:
                حالت GGML_TYPE_BF16:
                حالت GGML_TYPE_Q8_0:
                    بازگشت درست;
                پیشفرض:
                    بازگشت نادرست;
            }
        } بشکن;

        پیشفرض:
            بازگشت نادرست;
    }

    GGML_UNUSED(dev);
}

ایستا منطقی ggml_backend_zendnn_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    بازگشت ggml_backend_buft_is_host(buft);

    GGML_UNUSED(dev);
}

ایستا ثابت ساختار ggml_backend_device_i ggml_backend_zendnn_device_i = {
    /* .get_name               = */ ggml_backend_zendnn_device_get_name,
    /* .get_description        = */ ggml_backend_zendnn_device_get_description,
    /* .get_memory             = */ ggml_backend_zendnn_device_get_memory,
    /* .get_type               = */ ggml_backend_zendnn_device_get_type,
    /* .get_props              = */ ggml_backend_zendnn_device_get_props,
    /* .init_backend           = */ ggml_backend_zendnn_device_init_backend,
    /* .get_buffer_type        = */ ggml_backend_zendnn_device_get_buffer_type,
    /* .get_host_buffer_type   = */ پوچ,
    /* .buffer_from_host_ptr   = */ ggml_backend_zendnn_device_buffer_from_host_ptr,
    /* .supports_op            = */ ggml_backend_zendnn_device_supports_op,
    /* .supports_buft          = */ ggml_backend_zendnn_device_supports_buft,
    /* .offload_op             = */ پوچ,
    /* .event_new              = */ پوچ,
    /* .event_free             = */ پوچ,
    /* .event_synchronize      = */ پوچ,
};

// backend reg interface
ایستا ثابت نویسه * ggml_backend_zendnn_reg_get_name(ggml_backend_reg_t reg) {
    بازگشت "ZenDNN";

    GGML_UNUSED(reg);
}

ایستا size_t ggml_backend_zendnn_reg_get_device_شمار(ggml_backend_reg_t reg) {
    بازگشت 1;

    GGML_UNUSED(reg);
}

ایستا ggml_backend_dev_t ggml_backend_zendnn_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    ایستا ggml_backend_device ggml_backend_zendnn_device = {
        /* .iface   = */ ggml_backend_zendnn_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };

    بازگشت &ggml_backend_zendnn_device;
}

ایستا تهی * ggml_backend_zendnn_get_proc_address(ggml_backend_reg_t reg, ثابت نویسه * name) {
    اگر (std::strcmp(name, "ggml_backend_set_n_threads") == 0) {
        بازگشت (تهی *) ggml_backend_zendnn_set_n_threads;
    }
    بازگشت پوچ;

    GGML_UNUSED(reg);
    GGML_UNUSED(name);
}

ایستا ثابت ساختار ggml_backend_reg_i ggml_backend_zendnn_reg_i = {
    /* .get_name         = */ ggml_backend_zendnn_reg_get_name,
    /* .get_device_count = */ ggml_backend_zendnn_reg_get_device_شمار,
    /* .get_device       = */ ggml_backend_zendnn_reg_get_device,
    /* .get_proc_address = */ ggml_backend_zendnn_get_proc_address,
};

ggml_backend_reg_t ggml_backend_zendnn_reg(تهی) {
    ایستا ساختار ggml_backend_reg ggml_backend_zendnn_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_zendnn_reg_i,
        /* .context     = */ پوچ,
    };

    بازگشت &ggml_backend_zendnn_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_zendnn_reg)
