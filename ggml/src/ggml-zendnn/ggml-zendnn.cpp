#ضمن "ggml-zendnn.h"

#ضمن "ggml-backend-impl.h"
#ضمن "ggml-impl.h"

#تعریف_کرو GGML_COMMON_DECL_CPP
#ضمن "ggml-common.h"

#ضمن "zendnnl.hpp"

#ضمن <cstring>


بنية ggml_backend_zendnn_context {
    صحيح n_دھاگے = GGML_DEFAULT_N_دھاگے;
    std::unique_ptr<حرف[]> work_انکڑے;
    size_t work_ناپ = 0;
};

سانچا<قسم_ناں T>
zendnnl::common::انکڑے_type_t ggml_منزل_zendnn_type() {
    إذا constexpr (std::is_same_v<T, عائم>) {
        أرجع zendnnl::common::انکڑے_type_t::f32;
    } وإلا إذا constexpr (std::is_same_v<T, ggml_bf16_t>) {
        أرجع zendnnl::common::انکڑے_type_t::bf16;
    } وإلا إذا constexpr (std::is_same_v<T, بلاک_q8_0>) {
        أرجع zendnnl::common::انکڑے_type_t::s8;
    } وإلا {
        أرجع zendnnl::common::انکڑے_type_t::none;
    }
}

/**
 * Builds the matmul_params shared by ggml_zendnn_matmul() and ggml_zendnn_group_matmul():
 * dtype selection plus, for Q8_0 weights, dynamic-quant setup. Callers still need to set
 * quant_params.src_scale.dims themselves, since that depends on the batch size(s) in use.
 */
سانچا <قسم_ناں TA, قسم_ناں TB, قسم_ناں TC>
ساكن zendnnl::lowoha::matmul::matmul_params ggml_zendnn_make_matmul_params(ggml_backend_zendnn_context * ctx) {
    zendnnl::lowoha::matmul::matmul_params params;
    params.dtypes.src = ggml_منزل_zendnn_type<TB>();
    params.dtypes.wei = ggml_منزل_zendnn_type<TA>();
    params.dtypes.dst = ggml_منزل_zendnn_type<TC>();
    params.عدد_دھاگے = ctx->n_دھاگے;

    إذا constexpr (std::is_same_v<TA, بلاک_q8_0>) {
        params.dtypes.compute = zendnnl::common::انکڑے_type_t::s8;
        params.dynamic_quant = سچ;
        params.quant_params.src_scale.buff = خالی_سوچک;
        params.quant_params.src_scale.dt   = zendnnl::common::انکڑے_type_t::bf16;
        params.packing.pack_format_b = 1;
    }
    أرجع params;
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
سانچا <قسم_ناں TA, قسم_ناں TB, قسم_ناں TC>
ساكن منطقي ggml_zendnn_matmul(ggml_backend_zendnn_context * ctx, int64_t m, int64_t n, int64_t k,
                               ثابت TA * A, int64_t lda, ثابت TB * B, int64_t ldb, TC * C,
                               int64_t ldc) {

    zendnnl::lowoha::matmul::matmul_params params = ggml_zendnn_make_matmul_params<TA, TB, TC>(ctx);

    zendnnl::lowoha::matmul::matmul_batch_params_t batch_params;

    إذا constexpr (std::is_same_v<TA, بلاک_q8_0>) {
        params.quant_params.src_scale.dims = {n, k / QK8_0};
    }

    zendnnl::error_handling::حالت_t حالت = zendnnl::lowoha::matmul::matmul_سدھا(
        'r', خطأ, سچ,   // row-major, don't transpose B, transpose A (because it's column-major)
        n,                  // M: rows of B and C
        m,                  // N: cols of A^T and C
        k,                  // K: cols of B, rows of A
        1.0f,               // alpha
        B, ldb,             // src: B[n,k]
        A, lda,             // weight: A[k,m] column-major (transposed)
        خالی_سوچک,            // bias
        0.0f,               // beta
        C, ldc,             // output C[n,m]
        سچ,               // is_weights_const
        batch_params,       // batch_params
        params              // params
    );

    إذا (حالت != zendnnl::error_handling::حالت_t::success) {
        GGML_LOG_ERROR("%s, ZenDNN matmul failed: status=%d\n", __func__, static_cast<صحيح>(حالت));
        أرجع خطأ;
    }
    أرجع سچ;
}

ساكن منطقي ggml_zendnn_gemm(ggml_backend_zendnn_context * ctx, int64_t m, int64_t n, int64_t k,
                              ثابت فراغ * A, int64_t lda, ثابت فراغ * B, int64_t ldb, فراغ * C,
                              int64_t ldc, صحيح Atype, صحيح Btype, صحيح Ctype) {

    أكد(m >= 0);
    أكد(n >= 0);
    أكد(k >= 0);
    أكد(lda >= k);
    أكد(ldb >= k);
    أكد(ldc >= m);

    // categorize types
    بدل (Atype) {
        حالة GGML_TYPE_F32:
            إذا (Btype != GGML_TYPE_F32 || Ctype != GGML_TYPE_F32)
                أرجع خطأ;
            أرجع ggml_zendnn_matmul<عائم, عائم, عائم>(
                ctx, m, n, k,
                (ثابت عائم *)A, lda,
                (ثابت عائم *)B, ldb,
                (عائم *)C, ldc);
        حالة GGML_TYPE_BF16:
            إذا (Btype != GGML_TYPE_BF16)
                أرجع خطأ;
            إذا (Ctype == GGML_TYPE_BF16)
                أرجع ggml_zendnn_matmul<ggml_bf16_t, ggml_bf16_t, ggml_bf16_t>(
                    ctx, m, n, k,
                    (ثابت ggml_bf16_t *)A, lda,
                    (ثابت ggml_bf16_t *)B, ldb,
                    (ggml_bf16_t *)C, ldc);
            إذا (Ctype == GGML_TYPE_F32)
                أرجع ggml_zendnn_matmul<ggml_bf16_t, ggml_bf16_t, عائم>(
                    ctx, m, n, k,
                    (ثابت ggml_bf16_t *)A, lda,
                    (ثابت ggml_bf16_t *)B, ldb,
                    (عائم *)C, ldc);
            أرجع خطأ;
        حالة GGML_TYPE_Q8_0:
            إذا (Btype != GGML_TYPE_F32 || Ctype != GGML_TYPE_F32)
                أرجع خطأ;
            أرجع ggml_zendnn_matmul<بلاک_q8_0, عائم, عائم>(
                ctx, m, n, k,
                (ثابت بلاک_q8_0 *)A, lda,
                (ثابت عائم *)B, ldb,
                (عائم *)C, ldc);
        افتراضي:
            أرجع خطأ; // unsupported type
    }
}

ساكن فراغ ggml_zendnn_compute_forward_mul_mat(
    ggml_backend_zendnn_context * ctx,
    ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];  // weights
    ثابت ggml_tensor * src1 = dst->src[1];  // inputs

    GGML_TENSOR_ثنائی_OP_LOCALS

    ggml_type         ثابت vec_dot_type = src0->type;
    ggml_from_float_t ثابت from_float = ggml_get_type_traits(vec_dot_type)->from_float_ref;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_ناپ(src0->type));
    GGML_ASSERT(nb10 == ggml_type_ناپ(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == حجم(عائم));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // broadcast factors
    ثابت int64_t r2 = ne12/ne02;
    ثابت int64_t r3 = ne13/ne03;

    فراغ * work_انکڑے = ctx->work_انکڑے.get();

    // ZenDNN requires FP32 for dynamic quantization, so conversion is skipped
    إذا (src1->type != vec_dot_type && src0->type != GGML_TYPE_Q8_0) {
        ثابت size_t nbw1 = ggml_صف_ناپ(vec_dot_type, ne10);
        ثابت size_t nbw2 = nbw1 * ne11;
        ثابت size_t nbw3 = nbw2 * ne12;
        ثابت size_t desired_wsize = ne13 * nbw3;
        إذا (ctx->work_ناپ < desired_wsize) {
            ctx->work_انکڑے.reset(جديد حرف[desired_wsize]);
            ctx->work_ناپ = desired_wsize;
        }
        work_انکڑے = ctx->work_انکڑے.get();

        // #pragma omp parallel for num_threads(ctx->n_threads)
        #ہدایت omp parallel لكل collapse(3) عدد_دھاگے(ctx->n_دھاگے) ویلا_سارنی(ساكن)
        لكل (int64_t i13 = 0; i13 < ne13; ++i13) {
            لكل (int64_t i12 = 0; i12 < ne12; ++i12) {
                لكل (int64_t i11 = 0; i11 < ne11; ++i11) {
                    ثابت عائم * src1_f32 = (عائم *)((حرف *)src1->انکڑے + i11*nb11 + i12*nb12 + i13*nb13);
                    فراغ * src1_conv = (حرف *)work_انکڑے + i11*nbw1 + i12*nbw2 + i13*nbw3;
                    from_float(src1_f32, src1_conv, ne10);
                }
            }
        }
    }

    لكل (int64_t i13 = 0; i13 < ne13; i13++) {
        لكل (int64_t i12 = 0; i12 < ne12; i12++) {
            ثابت فراغ* wdata = (src1->type == vec_dot_type || src0->type == GGML_TYPE_Q8_0) ? src1->انکڑے : work_انکڑے;
            ثابت size_t صف_ناپ = ggml_صف_ناپ(vec_dot_type, ne10);
            إذا (!ggml_zendnn_gemm(ctx,
                                  ne01,     // m
                                  ne11,     // n
                                  ne10,     // k
                                  static_cast<ثابت حرف *>(src0->انکڑے) + (i12/r2)*nb02 + (i13/r3)*nb03,
                                  ne00,     // lda
                                  static_cast<ثابت حرف *>(wdata) + (i12*ne11 + i13*ne12*ne11)*صف_ناپ,
                                  ne10,     // ldb
                                  static_cast<حرف *>(dst->انکڑے) + i12*nb2 + i13*nb3,
                                  ne01,     // ldc
                                  src0->type,
                                  src0->type == GGML_TYPE_Q8_0 ? GGML_TYPE_F32 : vec_dot_type,
                                  dst->type))
                GGML_ABORT("%s: ZenDNN gemm failed\n", __func__);
        }
    }
}

بنية mmid_صف_mapping {
    int32_t i1;
    int32_t i2;
};

/**
 * ZenDNN batched matmul: computes C[i] = B[i] * A[i] for every active expert i via a single
 * group_matmul_direct() call. Batched analogue of ggml_zendnn_matmul() - see its docs for the
 * per-expert A/B/C shape convention. m and k are shared by every expert; n (batch size) varies
 * per expert, hence the vector.
 */
سانچا <قسم_ناں TA, قسم_ناں TB, قسم_ناں TC>
ساكن منطقي ggml_zendnn_گروہ_matmul(ggml_backend_zendnn_context * ctx, int64_t m, int64_t k,
                                     ثابت std::vector<int64_t> & n,
                                     ثابت std::vector<ثابت فراغ *> & A, int64_t lda,
                                     ثابت std::vector<ثابت فراغ *> & B, int64_t ldb,
                                     ثابت std::vector<فراغ *> & C, int64_t ldc) {

    ثابت صحيح n_experts = n.size();

    zendnnl::lowoha::matmul::matmul_params ادھار_params = ggml_zendnn_make_matmul_params<TA, TB, TC>(ctx);

    std::vector<حرف>         layout(n_experts, 'r');
    std::vector<منطقي>         trans_a(n_experts, خطأ);
    std::vector<منطقي>         trans_b(n_experts, سچ);
    std::vector<صحيح>          batch_m(n_experts);
    std::vector<صحيح>          batch_n(n_experts, m);
    std::vector<صحيح>          batch_k(n_experts, k);
    std::vector<عائم>        alpha(n_experts, 1.0f);
    std::vector<عائم>        beta(n_experts, 0.0f);
    std::vector<ثابت فراغ *> bias(n_experts, خالی_سوچک);
    std::vector<صحيح>          lda_v(n_experts, lda);
    std::vector<صحيح>          ldb_v(n_experts, ldb);
    std::vector<صحيح>          ldc_v(n_experts, ldc);
    std::vector<منطقي>         is_wei_const(n_experts, سچ);
    std::vector<zendnnl::lowoha::matmul::matmul_params> params(n_experts, ادھار_params);

    لكل (صحيح i = 0; i < n_experts; i++) {
        batch_m[i] = n[i];

        // src_scale.dims depends on this expert's row count, unlike the rest of base_params
        إذا constexpr (std::is_same_v<TA, بلاک_q8_0>) {
            params[i].quant_params.src_scale.dims = {n[i], k / QK8_0};
        }
    }

    zendnnl::error_handling::حالت_t حالت = zendnnl::lowoha::matmul::گروہ_matmul_سدھا(
        layout, trans_a, trans_b, batch_m, batch_n, batch_k, alpha,
        B, ldb_v, A, lda_v, bias, beta,
        C, ldc_v, is_wei_const, params);

    إذا (حالت != zendnnl::error_handling::حالت_t::success) {
        GGML_LOG_ERROR("%s, ZenDNN group matmul failed: status=%d\n", __func__, static_cast<صحيح>(حالت));
        أرجع خطأ;
    }
    أرجع سچ;
}

ساكن منطقي ggml_zendnn_گروہ_gemm(ggml_backend_zendnn_context * ctx, int64_t m, int64_t k,
                                    ثابت std::vector<int64_t> & n,
                                    ثابت std::vector<ثابت فراغ *> & A, int64_t lda,
                                    ثابت std::vector<ثابت فراغ *> & B, int64_t ldb,
                                    ثابت std::vector<فراغ *> & C, int64_t ldc,
                                    صحيح Atype, صحيح Btype, صحيح Ctype) {

    أكد(m >= 0);
    لكل (size_t i = 0; i < n.size(); i++) {
        أكد(n[i] >= 0);
    }
    أكد(k >= 0);
    أكد(lda >= k);
    أكد(ldb >= k);
    أكد(ldc >= m);

    // categorize types
    بدل (Atype) {
        حالة GGML_TYPE_F32:
            إذا (Btype != GGML_TYPE_F32 || Ctype != GGML_TYPE_F32)
                أرجع خطأ;
            أرجع ggml_zendnn_گروہ_matmul<عائم, عائم, عائم>(ctx, m, k, n, A, lda, B, ldb, C, ldc);
        حالة GGML_TYPE_BF16:
            إذا (Btype != GGML_TYPE_BF16)
                أرجع خطأ;
            إذا (Ctype == GGML_TYPE_BF16)
                أرجع ggml_zendnn_گروہ_matmul<ggml_bf16_t, ggml_bf16_t, ggml_bf16_t>(
                    ctx, m, k, n, A, lda, B, ldb, C, ldc);
            إذا (Ctype == GGML_TYPE_F32)
                أرجع ggml_zendnn_گروہ_matmul<ggml_bf16_t, ggml_bf16_t, عائم>(ctx, m, k, n, A, lda, B, ldb, C, ldc);
            أرجع خطأ;
        حالة GGML_TYPE_Q8_0:
            إذا (Btype != GGML_TYPE_F32 || Ctype != GGML_TYPE_F32)
                أرجع خطأ;
            أرجع ggml_zendnn_گروہ_matmul<بلاک_q8_0, عائم, عائم>(ctx, m, k, n, A, lda, B, ldb, C, ldc);
        افتراضي:
            أرجع خطأ; // unsupported type
    }
}

ساكن فراغ ggml_zendnn_compute_forward_mul_mat_id(
    ggml_backend_zendnn_context * ctx,
    ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];  // expert weights
    ثابت ggml_tensor * src1 = dst->src[1];  // inputs
    ثابت ggml_tensor * ids  = dst->src[2];  // expert ids

    GGML_TENSOR_ثنائی_OP_LOCALS

    // exit for no tokens to process
    إذا (ne2 == 0 || ne11 == 0) {
        أرجع;
    }

    ggml_type         ثابت vec_dot_type = src0->type;
    ggml_from_float_t ثابت from_float = ggml_get_type_traits(vec_dot_type)->from_float_ref;

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_ناپ(src0->type));
    GGML_ASSERT(nb10 == ggml_type_ناپ(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == حجم(عائم));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(ne03 == 1);
    GGML_ASSERT(ne13 == 1);
    GGML_ASSERT(ne3  == 1);

    // row groups
    ثابت صحيح n_ids = ids->ne[0]; // n_expert_used
    ثابت صحيح n_as  = ne02;       // n_experts

    std::vector<int64_t> میٹرکس_صف_counts(n_as, 0);
    std::vector<std::vector<mmid_صف_mapping>> میٹرکس_صفاں(n_as);

    int64_t کل_صفاں = 0;
    صحيح     n_سرگرم_experts = 0;
    // group rows by expert (preprocessing step)
    لكل (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
        لكل (صحيح id = 0; id < n_ids; ++id) {
            ثابت int32_t i02 = *(ثابت int32_t *)((ثابت حرف *)ids->انکڑے + iid1*ids->nb[1] + id*ids->nb[0]);

            GGML_ASSERT(i02 >= 0 && i02 < n_as);

            إذا (میٹرکس_صف_counts[i02] == 0) {
                n_سرگرم_experts++;
            }
            میٹرکس_صفاں[i02].push_back({id, iid1});
            میٹرکس_صف_counts[i02]++;
            کل_صفاں++;
        }
    }

    إذا (کل_صفاں == 0) {
        أرجع; // no rows to process
    }

    ثابت size_t صف_ناپ = ggml_صف_ناپ(vec_dot_type, ne10);

    // For Q8_0, src1 is always F32; the gather buffer must hold F32 rows (ne10*4 bytes),
    // not Q8_0-encoded rows (row_size ≈ ne10/32*34 bytes) — they differ by ~4x.
    ثابت size_t f32_صف_ناپ = (size_t)ne10 * حجم(عائم);
    ثابت size_t gather_صف_ناپ = (src0->type == GGML_TYPE_Q8_0) ? f32_صف_ناپ : صف_ناپ;

    إذا (src1->type != vec_dot_type && src0->type != GGML_TYPE_Q8_0) {
        GGML_ASSERT(src1->type == GGML_TYPE_F32);
    }

    // size for MoE gather/scatter buffers
    ثابت size_t wdata_cur_ناپ = کل_صفاں * gather_صف_ناپ;
    ثابت size_t dst_cur_ناپ = کل_صفاں * ggml_صف_ناپ(dst->type, ne01);

    // allocate single buffer for all needs
    ثابت size_t کل_ناپ = wdata_cur_ناپ + dst_cur_ناپ;
    إذا (ctx->work_ناپ < کل_ناپ) {
        ctx->work_انکڑے.reset(جديد حرف[کل_ناپ]);
        ctx->work_ناپ = کل_ناپ;
    }

    // partition the buffer
    حرف * wdata_cur = ctx->work_انکڑے.get();
    حرف * dst_cur = wdata_cur + wdata_cur_ناپ;

    // per-expert data collected during gather, handed to ggml_zendnn_group_gemm() as one batch
    std::vector<int64_t>      expert_صف_گنتی(n_سرگرم_experts);
    std::vector<ثابت فراغ *> batch_src(n_سرگرم_experts);
    std::vector<ثابت فراغ *> batch_wei(n_سرگرم_experts);
    std::vector<فراغ *>       batch_dst(n_سرگرم_experts);

    // precompute per-expert buffer offsets and batch indices for the parallel loop below
    std::vector<int64_t> expert_wdata_off(n_as, 0);
    std::vector<int64_t> expert_dst_off(n_as, 0);
    std::vector<صحيح>     expert_batch_idx(n_as, -1);
    {
        int64_t w_off = 0;
        int64_t d_off = 0;
        صحيح     batch_idx = 0;
        لكل (int64_t cur_a = 0; cur_a < n_as; ++cur_a) {
            إذا (میٹرکس_صف_counts[cur_a] == 0) {
                استمر;
            }
            expert_wdata_off[cur_a] = w_off;
            expert_dst_off[cur_a]   = d_off;
            expert_batch_idx[cur_a] = batch_idx;
            w_off += میٹرکس_صف_counts[cur_a] * gather_صف_ناپ;
            d_off += میٹرکس_صف_counts[cur_a] * ggml_صف_ناپ(dst->type, ne01);
            batch_idx++;
        }
    }

    // gather + inline-convert input rows into each expert's batch slot
    #ہدایت omp parallel لكل عدد_دھاگے(ctx->n_دھاگے) ویلا_سارنی(ساكن)
    لكل (int64_t cur_a = 0; cur_a < n_as; ++cur_a) {
        ثابت int64_t cne1 = میٹرکس_صف_counts[cur_a];

        إذا (cne1 == 0) {
            استمر;
        }

        ثابت int64_t w_off     = expert_wdata_off[cur_a];
        ثابت int64_t d_off     = expert_dst_off[cur_a];
        ثابت صحيح     batch_idx = expert_batch_idx[cur_a];

        لكل (int64_t ir1 = 0; ir1 < cne1; ++ir1) {
            ثابت mmid_صف_mapping & صف_mapping = میٹرکس_صفاں[cur_a][ir1];
            ثابت int64_t id  = صف_mapping.i1;
            ثابت int64_t i11 = id % ne11;
            ثابت int64_t i12 = صف_mapping.i2;

            ثابت حرف * src_صف = (ثابت حرف *) src1->انکڑے + i11*nb11 + i12*nb12;
            فراغ *       dst_صف = wdata_cur + w_off + ir1 * gather_صف_ناپ;

            إذا (src1->type != vec_dot_type && src0->type != GGML_TYPE_Q8_0) {
                from_float((ثابت عائم *) src_صف, dst_صف, ne10);
            } وإلا {
                // no conversion: src1 already matches vec_dot_type, or src0 is Q8_0, whose
                // ZenDNN dynamic quantization requires the row to stay in F32
                std::حافظہ_نقل(dst_صف, src_صف, gather_صف_ناپ);
            }
        }

        expert_صف_گنتی[batch_idx] = cne1;
        batch_src[batch_idx]        = wdata_cur + w_off;
        batch_wei[batch_idx]        = (ثابت حرف *) src0->انکڑے + cur_a * nb02;
        batch_dst[batch_idx]        = dst_cur + d_off;
    }

    إذا (!ggml_zendnn_گروہ_gemm(ctx,
                                ne01,             // m
                                ne10,             // k
                                expert_صف_گنتی, // n (per expert)
                                batch_wei, ne00,  // A: weights, lda
                                batch_src, ne10,  // B: input, ldb
                                batch_dst, ne01,  // C: output, ldc
                                src0->type,
                                src0->type == GGML_TYPE_Q8_0 ? GGML_TYPE_F32 : vec_dot_type,
                                dst->type))
        GGML_ABORT("%s: ZenDNN group gemm failed\n", __func__);

    // scatter output rows to destination
    #ہدایت omp parallel لكل عدد_دھاگے(ctx->n_دھاگے) ویلا_سارنی(ساكن)
    لكل (int64_t cur_a = 0; cur_a < n_as; ++cur_a) {
        ثابت int64_t cne1 = میٹرکس_صف_counts[cur_a];

        إذا (cne1 == 0) {
            استمر;
        }

        ثابت int64_t d_off = expert_dst_off[cur_a];

        لكل (int64_t ir1 = 0; ir1 < cne1; ++ir1) {
            ثابت mmid_صف_mapping & صف_mapping = میٹرکس_صفاں[cur_a][ir1];
            ثابت int64_t id = صف_mapping.i1;
            ثابت int64_t i1 = id;
            ثابت int64_t i2 = صف_mapping.i2;

            std::حافظہ_نقل(
                (حرف *) dst->انکڑے + i1*nb1 + i2*nb2,
                dst_cur + d_off + ir1 * ggml_صف_ناپ(dst->type, ne01),
                ggml_صف_ناپ(dst->type, ne01)
            );
        }
    }
}

// backend interface

ساكن ثابت حرف * ggml_backend_zendnn_get_name(ggml_backend_t backend) {
    أرجع "ZenDNN";

    GGML_بن_ورتیا(backend);
}

ساكن فراغ ggml_backend_zendnn_free(ggml_backend_t backend) {
    ggml_backend_zendnn_context * ctx = (ggml_backend_zendnn_context *)backend->context;
    مٹاؤ ctx;
    مٹاؤ backend;
}

ساكن ggml_حالت ggml_backend_zendnn_گراف_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_zendnn_context * ctx = (ggml_backend_zendnn_context *)backend->context;

    لكل (صحيح i = 0; i < cgraph->n_nodes; i++) {
        بنية ggml_tensor * گرہ = cgraph->nodes[i];

        إذا ((گرہ->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            استمر;
        }

        بدل (گرہ->op) {
            حالة GGML_OP_MUL_MAT:
                ggml_zendnn_compute_forward_mul_mat(ctx, گرہ);
                اكسر;
            حالة GGML_OP_MUL_MAT_ID:
                ggml_zendnn_compute_forward_mul_mat_id(ctx, گرہ);
                اكسر;
            حالة GGML_OP_NONE:
            حالة GGML_OP_RESHAPE:
            حالة GGML_OP_VIEW:
            حالة GGML_OP_ترتیب_بدلو:
            حالة GGML_OP_TRANSPOSE:
                اكسر;

            افتراضي:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(گرہ));
        }
    }

    أرجع GGML_حالت_SUCCESS;

    GGML_بن_ورتیا(backend);
}

ساكن بنية ggml_backend_i ggml_backend_zendnn_i = {
    /* .get_name                = */ ggml_backend_zendnn_get_name,
    /* .free                    = */ ggml_backend_zendnn_free,
    /* .set_tensor_async        = */ عدم,
    /* .get_tensor_async        = */ عدم,
    /* .set_tensor_2d_async     = */ عدم,
    /* .get_tensor_2d_async     = */ عدم,
    /* .cpy_tensor_async        = */ عدم,
    /* .synchronize             = */ عدم,
    /* .graph_plan_create       = */ عدم,
    /* .graph_plan_free         = */ عدم,
    /* .graph_plan_update       = */ عدم,
    /* .graph_plan_compute      = */ عدم,
    /* .graph_compute           = */ ggml_backend_zendnn_گراف_compute,
    /* .event_record            = */ عدم,
    /* .event_wait              = */ عدم,
    /* .graph_optimize          = */ عدم,
};

ساكن ggml_guid_t ggml_backend_zendnn_guid(فراغ) {
    ساكن ثابت حرف * guid_str = "AMD-ZENDNN-ACCEL";
    أرجع reinterpret_cast<ggml_guid_t>(const_cast<حرف*>(guid_str));
}

ggml_backend_t ggml_backend_zendnn_init(فراغ) {
    ggml_backend_zendnn_context * ctx = جديد ggml_backend_zendnn_context;

    ggml_backend_t backend = جديد ggml_backend {
        /* .guid    = */ ggml_backend_zendnn_guid(),
        /* .iface   = */ ggml_backend_zendnn_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_zendnn_reg(), 0),
        /* .context = */ ctx,
    };

    أرجع backend;
}

منطقي ggml_backend_is_zendnn(ggml_backend_t backend) {
    أرجع backend != عدم && ggml_guid_matches(backend->guid, ggml_backend_zendnn_guid());
}

فراغ ggml_backend_zendnn_سیٹ_n_دھاگے(ggml_backend_t backend_zendnn, صحيح n_دھاگے) {
    GGML_ASSERT(ggml_backend_is_zendnn(backend_zendnn));

    ggml_backend_zendnn_context * ctx = (ggml_backend_zendnn_context *)backend_zendnn->context;
    ctx->n_دھاگے = n_دھاگے;
}

// device interface
ساكن ثابت حرف * ggml_backend_zendnn_device_get_name(ggml_backend_dev_t dev) {
    أرجع "ZenDNN";

    GGML_بن_ورتیا(dev);
}
/**
 * ZenDNN is AMD's performance library providing optimized primitives and implementations
 * for deep learning workloads on AMD CPUs. It targets improved performance for common
 * neural network operations on AMD architectures. For more information, see:
 * https://www.amd.com/en/developer/zendnn.html
 */
ساكن ثابت حرف * ggml_backend_zendnn_device_get_description(ggml_backend_dev_t dev) {
    أرجع "ZenDNN: AMD optimized primitives backend for GGML (optimized for AMD CPUs)";

    GGML_بن_ورتیا(dev);
}

ساكن فراغ ggml_backend_zendnn_device_get_حافظہ(ggml_backend_dev_t dev, size_t * حرر, size_t * کل) {
    *حرر  = 0;
    *کل = 0;

    GGML_بن_ورتیا(dev);
}

ساكن تعداد ggml_backend_dev_type ggml_backend_zendnn_device_get_type(ggml_backend_dev_t dev) {
    أرجع GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_بن_ورتیا(dev);
}

ساكن فراغ ggml_backend_zendnn_device_get_props(ggml_backend_dev_t dev, بنية ggml_backend_dev_props * props) {
    props->name        = ggml_backend_zendnn_device_get_name(dev);
    props->description = ggml_backend_zendnn_device_get_description(dev);
    props->type        = ggml_backend_zendnn_device_get_type(dev);
    ggml_backend_zendnn_device_get_حافظہ(dev, &props->حافظہ_free, &props->حافظہ_کل);
    props->caps = {
        /* .async                = */ خطأ,
        /* .host_buffer          = */ خطأ,
        /* .buffer_from_host_ptr = */ سچ,
        /* .events               = */ خطأ,
        /* .mmap_support         = */ سچ,
    };
}

ساكن ggml_backend_t ggml_backend_zendnn_device_init_backend(ggml_backend_dev_t dev, ثابت حرف * params) {
    ggml_backend_t backend = ggml_backend_zendnn_init();
    إذا (backend == عدم) {
        GGML_LOG_ERROR("%s: error: failed to initialize ZenDNN backend\n", __func__);
        أرجع عدم;
    }

    أرجع backend;

    GGML_بن_ورتیا(dev);
    GGML_بن_ورتیا(params);
}

ساكن ggml_backend_بفر_type_t ggml_backend_zendnn_device_get_بفر_type(ggml_backend_dev_t dev) {
    أرجع ggml_backend_cpu_بفر_type();

    GGML_بن_ورتیا(dev);
}

ساكن ggml_backend_بفر_t ggml_backend_zendnn_device_بفر_from_host_ptr(ggml_backend_dev_t dev, فراغ * ptr, size_t size, size_t max_tensor_ناپ) {
    أرجع ggml_backend_cpu_بفر_from_ptr(ptr, size);

    GGML_بن_ورتیا(dev);
    GGML_بن_ورتیا(max_tensor_ناپ);
}

ساكن منطقي ggml_zendnn_adaptive_fallback_enabled() {
    ساكن ثابت منطقي enabled = std::getenv("GGML_ZENDNN_ADAPTIVE_FALLBACK") == خالی_سوچک ||
                                std::atoi(std::getenv("GGML_ZENDNN_ADAPTIVE_FALLBACK")) != 0;
    أرجع enabled;
}

ساكن منطقي ggml_backend_zendnn_device_supports_op(ggml_backend_dev_t dev, ثابت بنية ggml_tensor * op) {
    بدل (op->op) {
        حالة GGML_OP_NONE:
        حالة GGML_OP_RESHAPE:
        حالة GGML_OP_VIEW:
        حالة GGML_OP_ترتیب_بدلو:
        حالة GGML_OP_TRANSPOSE:
            أرجع سچ;

        حالة GGML_OP_MUL_MAT:
        حالة GGML_OP_MUL_MAT_ID:
        {
            ثابت ggml_tensor * weights = op->src[0];
            ثابت ggml_tensor * inputs = op->src[1];

            ثابت int64_t ne10 = inputs->ne[0];
            ثابت int64_t ne0 = op->ne[0];
            ثابت int64_t ne1 = op->ne[1];
            ثابت int64_t min_batch = 1;

            إذا(!ggml_is_contiguous(weights) || !ggml_is_contiguous(inputs)) {
                أرجع خطأ;
            }

            إذا (ggml_zendnn_adaptive_fallback_enabled()) {
                ثابت int64_t K = inputs->ne[0];
                ثابت int64_t N = (inputs->ne[1]*inputs->ne[2]*inputs->ne[3]);
                ثابت int64_t M = weights->ne[1];
                إذا(K <= 256 || N <= 128 || M <= 96) {
                    أرجع خطأ;
                }

                // MUL_MAT_ID's gather+matmul+scatter approach favors a moderate expert count
                إذا (op->op == GGML_OP_MUL_MAT_ID) {
                    ثابت int64_t n_experts = weights->ne[2];
                    ثابت int64_t max_experts = 32;
                    إذا (n_experts > max_experts) {
                        أرجع خطأ;
                    }

                    // fall back once the average rows per expert (N / n_experts) is too thin
                    // to amortize each per-expert GEMM's overhead
                    إذا (N / n_experts <= 32) {
                        أرجع خطأ;
                    }
                }
            }
            وإلا إذا (ne0 < min_batch || ne1 < min_batch || ne10 < min_batch) {
                أرجع خطأ;
            }

            بدل (weights->type) {
                حالة GGML_TYPE_F32:
                حالة GGML_TYPE_BF16:
                حالة GGML_TYPE_Q8_0:
                    أرجع سچ;
                افتراضي:
                    أرجع خطأ;
            }
        } اكسر;

        افتراضي:
            أرجع خطأ;
    }

    GGML_بن_ورتیا(dev);
}

ساكن منطقي ggml_backend_zendnn_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_بفر_type_t buft) {
    أرجع ggml_backend_buft_is_host(buft);

    GGML_بن_ورتیا(dev);
}

ساكن ثابت بنية ggml_backend_device_i ggml_backend_zendnn_device_i = {
    /* .get_name               = */ ggml_backend_zendnn_device_get_name,
    /* .get_description        = */ ggml_backend_zendnn_device_get_description,
    /* .get_memory             = */ ggml_backend_zendnn_device_get_حافظہ,
    /* .get_type               = */ ggml_backend_zendnn_device_get_type,
    /* .get_props              = */ ggml_backend_zendnn_device_get_props,
    /* .init_backend           = */ ggml_backend_zendnn_device_init_backend,
    /* .get_buffer_type        = */ ggml_backend_zendnn_device_get_بفر_type,
    /* .get_host_buffer_type   = */ عدم,
    /* .buffer_from_host_ptr   = */ ggml_backend_zendnn_device_بفر_from_host_ptr,
    /* .supports_op            = */ ggml_backend_zendnn_device_supports_op,
    /* .supports_buft          = */ ggml_backend_zendnn_device_supports_buft,
    /* .offload_op             = */ عدم,
    /* .event_new              = */ عدم,
    /* .event_free             = */ عدم,
    /* .event_synchronize      = */ عدم,
};

// backend reg interface
ساكن ثابت حرف * ggml_backend_zendnn_reg_get_name(ggml_backend_reg_t reg) {
    أرجع "ZenDNN";

    GGML_بن_ورتیا(reg);
}

ساكن size_t ggml_backend_zendnn_reg_get_device_گنتی(ggml_backend_reg_t reg) {
    أرجع 1;

    GGML_بن_ورتیا(reg);
}

ساكن ggml_backend_dev_t ggml_backend_zendnn_reg_get_device(ggml_backend_reg_t reg, size_t اشاریہ) {
    GGML_ASSERT(اشاریہ == 0);

    ساكن ggml_backend_device ggml_backend_zendnn_device = {
        /* .iface   = */ ggml_backend_zendnn_device_i,
        /* .reg     = */ reg,
        /* .context = */ خالی_سوچک,
    };

    أرجع &ggml_backend_zendnn_device;
}

ساكن فراغ * ggml_backend_zendnn_get_proc_پتہ(ggml_backend_reg_t reg, ثابت حرف * name) {
    إذا (std::strcmp(name, "ggml_backend_set_n_threads") == 0) {
        أرجع (فراغ *) ggml_backend_zendnn_سیٹ_n_دھاگے;
    }
    أرجع عدم;

    GGML_بن_ورتیا(reg);
    GGML_بن_ورتیا(name);
}

ساكن ثابت بنية ggml_backend_reg_i ggml_backend_zendnn_reg_i = {
    /* .get_name         = */ ggml_backend_zendnn_reg_get_name,
    /* .get_device_count = */ ggml_backend_zendnn_reg_get_device_گنتی,
    /* .get_device       = */ ggml_backend_zendnn_reg_get_device,
    /* .get_proc_address = */ ggml_backend_zendnn_get_proc_پتہ,
};

ggml_backend_reg_t ggml_backend_zendnn_reg(فراغ) {
    ساكن بنية ggml_backend_reg ggml_backend_zendnn_reg = {
        /* .api_version = */ GGML_BACKEND_API_نسخہ,
        /* .iface       = */ ggml_backend_zendnn_reg_i,
        /* .context     = */ عدم,
    };

    أرجع &ggml_backend_zendnn_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_zendnn_reg)
