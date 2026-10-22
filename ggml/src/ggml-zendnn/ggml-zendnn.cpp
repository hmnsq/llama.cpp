#അന്തര്ഭാവ "ggml-zendnn.h"

#അന്തര്ഭാവ "ggml-backend-impl.h"
#അന്തര്ഭാവ "ggml-impl.h"

#define GGML_COMMON_DECL_CPP
#അന്തര്ഭാവ "ggml-common.h"

#അന്തര്ഭാവ "zendnnl.hpp"

#അന്തര്ഭാവ <cstring>


ഘടന ggml_backend_zendnn_context {
    പൂർണ്ണസംഖ്യ n_threads = GGML_DEFAULT_N_THREADS;
    std::unique_ptr<അക്ഷരം[]> work_data;
    size_t work_size = 0;
};

template<typename T>
zendnnl::common::data_type_t ggml_to_zendnn_type() {
    എങ്കിൽ constexpr (std::is_same_v<T, ദശാംശം>) {
        തിരികെ zendnnl::common::data_type_t::f32;
    } അല്ലെങ്കിൽ എങ്കിൽ constexpr (std::is_same_v<T, ggml_bf16_t>) {
        തിരികെ zendnnl::common::data_type_t::bf16;
    } അല്ലെങ്കിൽ എങ്കിൽ constexpr (std::is_same_v<T, block_q8_0>) {
        തിരികെ zendnnl::common::data_type_t::s8;
    } അല്ലെങ്കിൽ {
        തിരികെ zendnnl::common::data_type_t::none;
    }
}

/**
 * Builds the matmul_params shared by ggml_zendnn_matmul() and ggml_zendnn_group_matmul():
 * dtype selection plus, for Q8_0 weights, dynamic-quant setup. Callers still need to set
 * quant_params.src_scale.dims themselves, since that depends on the batch size(s) in use.
 */
template <typename TA, typename TB, typename TC>
സ്ഥിതം zendnnl::lowoha::matmul::matmul_params ggml_zendnn_make_matmul_params(ggml_backend_zendnn_context * ctx) {
    zendnnl::lowoha::matmul::matmul_params params;
    params.dtypes.src = ggml_to_zendnn_type<TB>();
    params.dtypes.wei = ggml_to_zendnn_type<TA>();
    params.dtypes.dst = ggml_to_zendnn_type<TC>();
    params.അക്കം_threads = ctx->n_threads;

    എങ്കിൽ constexpr (std::is_same_v<TA, block_q8_0>) {
        params.dtypes.compute = zendnnl::common::data_type_t::s8;
        params.dynamic_quant = സത്യം;
        params.quant_params.src_scale.buff = nullptr;
        params.quant_params.src_scale.dt   = zendnnl::common::data_type_t::bf16;
        params.packing.pack_format_b = 1;
    }
    തിരികെ params;
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
സ്ഥിതം തർക്കം ggml_zendnn_matmul(ggml_backend_zendnn_context * ctx, int64_t m, int64_t n, int64_t k,
                               സ്ഥിരം TA * A, int64_t lda, സ്ഥിരം TB * B, int64_t ldb, TC * C,
                               int64_t ldc) {

    zendnnl::lowoha::matmul::matmul_params params = ggml_zendnn_make_matmul_params<TA, TB, TC>(ctx);

    zendnnl::lowoha::matmul::matmul_batch_params_t batch_params;

    എങ്കിൽ constexpr (std::is_same_v<TA, block_q8_0>) {
        params.quant_params.src_scale.dims = {n, k / QK8_0};
    }

    zendnnl::error_handling::status_t status = zendnnl::lowoha::matmul::matmul_direct(
        'r', അസത്യം, സത്യം,   // row-major, don't transpose B, transpose A (because it's column-major)
        n,                  // M: rows of B and C
        m,                  // N: cols of A^T and C
        k,                  // K: cols of B, rows of A
        1.0f,               // alpha
        B, ldb,             // src: B[n,k]
        A, lda,             // weight: A[k,m] column-major (transposed)
        nullptr,            // bias
        0.0f,               // beta
        C, ldc,             // output C[n,m]
        സത്യം,               // is_weights_const
        batch_params,       // batch_params
        params              // params
    );

    എങ്കിൽ (status != zendnnl::error_handling::status_t::success) {
        GGML_LOG_ERROR("%s, ZenDNN matmul failed: status=%d\n", __func__, static_cast<പൂർണ്ണസംഖ്യ>(status));
        തിരികെ അസത്യം;
    }
    തിരികെ സത്യം;
}

സ്ഥിതം തർക്കം ggml_zendnn_gemm(ggml_backend_zendnn_context * ctx, int64_t m, int64_t n, int64_t k,
                              സ്ഥിരം ശൂന്യം * A, int64_t lda, സ്ഥിരം ശൂന്യം * B, int64_t ldb, ശൂന്യം * C,
                              int64_t ldc, പൂർണ്ണസംഖ്യ Atype, പൂർണ്ണസംഖ്യ Btype, പൂർണ്ണസംഖ്യ Ctype) {

    പ്രതിജ്ഞാ(m >= 0);
    പ്രതിജ്ഞാ(n >= 0);
    പ്രതിജ്ഞാ(k >= 0);
    പ്രതിജ്ഞാ(lda >= k);
    പ്രതിജ്ഞാ(ldb >= k);
    പ്രതിജ്ഞാ(ldc >= m);

    // categorize types
    തിരഞ്ഞെടുപ്പ് (Atype) {
        അവസ്ഥ GGML_TYPE_F32:
            എങ്കിൽ (Btype != GGML_TYPE_F32 || Ctype != GGML_TYPE_F32)
                തിരികെ അസത്യം;
            തിരികെ ggml_zendnn_matmul<ദശാംശം, ദശാംശം, ദശാംശം>(
                ctx, m, n, k,
                (സ്ഥിരം ദശാംശം *)A, lda,
                (സ്ഥിരം ദശാംശം *)B, ldb,
                (ദശാംശം *)C, ldc);
        അവസ്ഥ GGML_TYPE_BF16:
            എങ്കിൽ (Btype != GGML_TYPE_BF16)
                തിരികെ അസത്യം;
            എങ്കിൽ (Ctype == GGML_TYPE_BF16)
                തിരികെ ggml_zendnn_matmul<ggml_bf16_t, ggml_bf16_t, ggml_bf16_t>(
                    ctx, m, n, k,
                    (സ്ഥിരം ggml_bf16_t *)A, lda,
                    (സ്ഥിരം ggml_bf16_t *)B, ldb,
                    (ggml_bf16_t *)C, ldc);
            എങ്കിൽ (Ctype == GGML_TYPE_F32)
                തിരികെ ggml_zendnn_matmul<ggml_bf16_t, ggml_bf16_t, ദശാംശം>(
                    ctx, m, n, k,
                    (സ്ഥിരം ggml_bf16_t *)A, lda,
                    (സ്ഥിരം ggml_bf16_t *)B, ldb,
                    (ദശാംശം *)C, ldc);
            തിരികെ അസത്യം;
        അവസ്ഥ GGML_TYPE_Q8_0:
            എങ്കിൽ (Btype != GGML_TYPE_F32 || Ctype != GGML_TYPE_F32)
                തിരികെ അസത്യം;
            തിരികെ ggml_zendnn_matmul<block_q8_0, ദശാംശം, ദശാംശം>(
                ctx, m, n, k,
                (സ്ഥിരം block_q8_0 *)A, lda,
                (സ്ഥിരം ദശാംശം *)B, ldb,
                (ദശാംശം *)C, ldc);
        സ്ഥിരസ്ഥിതി:
            തിരികെ അസത്യം; // unsupported type
    }
}

സ്ഥിതം ശൂന്യം ggml_zendnn_compute_forward_mul_mat(
    ggml_backend_zendnn_context * ctx,
    ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];  // weights
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];  // inputs

    GGML_TENSOR_ദ്വിമാന_OP_LOCALS

    ggml_type         സ്ഥിരം vec_dot_type = src0->type;
    ggml_from_float_t സ്ഥിരം from_float = ggml_get_type_traits(vec_dot_type)->from_float_ref;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == വലുപ്പം(ദശാംശം));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // broadcast factors
    സ്ഥിരം int64_t r2 = ne12/ne02;
    സ്ഥിരം int64_t r3 = ne13/ne03;

    ശൂന്യം * work_data = ctx->work_data.get();

    // ZenDNN requires FP32 for dynamic quantization, so conversion is skipped
    എങ്കിൽ (src1->type != vec_dot_type && src0->type != GGML_TYPE_Q8_0) {
        സ്ഥിരം size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
        സ്ഥിരം size_t nbw2 = nbw1 * ne11;
        സ്ഥിരം size_t nbw3 = nbw2 * ne12;
        സ്ഥിരം size_t desired_wsize = ne13 * nbw3;
        എങ്കിൽ (ctx->work_size < desired_wsize) {
            ctx->work_data.reset(പുതിയ അക്ഷരം[desired_wsize]);
            ctx->work_size = desired_wsize;
        }
        work_data = ctx->work_data.get();

        // #pragma omp parallel for num_threads(ctx->n_threads)
        #pragma omp parallel വേണ്ടി collapse(3) അക്കം_threads(ctx->n_threads) schedule(സ്ഥിതം)
        വേണ്ടി (int64_t i13 = 0; i13 < ne13; ++i13) {
            വേണ്ടി (int64_t i12 = 0; i12 < ne12; ++i12) {
                വേണ്ടി (int64_t i11 = 0; i11 < ne11; ++i11) {
                    സ്ഥിരം ദശാംശം * src1_f32 = (ദശാംശം *)((അക്ഷരം *)src1->data + i11*nb11 + i12*nb12 + i13*nb13);
                    ശൂന്യം * src1_conv = (അക്ഷരം *)work_data + i11*nbw1 + i12*nbw2 + i13*nbw3;
                    from_float(src1_f32, src1_conv, ne10);
                }
            }
        }
    }

    വേണ്ടി (int64_t i13 = 0; i13 < ne13; i13++) {
        വേണ്ടി (int64_t i12 = 0; i12 < ne12; i12++) {
            സ്ഥിരം ശൂന്യം* wdata = (src1->type == vec_dot_type || src0->type == GGML_TYPE_Q8_0) ? src1->data : work_data;
            സ്ഥിരം size_t row_size = ggml_row_size(vec_dot_type, ne10);
            എങ്കിൽ (!ggml_zendnn_gemm(ctx,
                                  ne01,     // m
                                  ne11,     // n
                                  ne10,     // k
                                  static_cast<സ്ഥിരം അക്ഷരം *>(src0->data) + (i12/r2)*nb02 + (i13/r3)*nb03,
                                  ne00,     // lda
                                  static_cast<സ്ഥിരം അക്ഷരം *>(wdata) + (i12*ne11 + i13*ne12*ne11)*row_size,
                                  ne10,     // ldb
                                  static_cast<അക്ഷരം *>(dst->data) + i12*nb2 + i13*nb3,
                                  ne01,     // ldc
                                  src0->type,
                                  src0->type == GGML_TYPE_Q8_0 ? GGML_TYPE_F32 : vec_dot_type,
                                  dst->type))
                GGML_ABORT("%s: ZenDNN gemm failed\n", __func__);
        }
    }
}

ഘടന mmid_row_mapping {
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
സ്ഥിതം തർക്കം ggml_zendnn_group_matmul(ggml_backend_zendnn_context * ctx, int64_t m, int64_t k,
                                     സ്ഥിരം std::vector<int64_t> & n,
                                     സ്ഥിരം std::vector<സ്ഥിരം ശൂന്യം *> & A, int64_t lda,
                                     സ്ഥിരം std::vector<സ്ഥിരം ശൂന്യം *> & B, int64_t ldb,
                                     സ്ഥിരം std::vector<ശൂന്യം *> & C, int64_t ldc) {

    സ്ഥിരം പൂർണ്ണസംഖ്യ n_experts = n.size();

    zendnnl::lowoha::matmul::matmul_params base_params = ggml_zendnn_make_matmul_params<TA, TB, TC>(ctx);

    std::vector<അക്ഷരം>         layout(n_experts, 'r');
    std::vector<തർക്കം>         trans_a(n_experts, അസത്യം);
    std::vector<തർക്കം>         trans_b(n_experts, സത്യം);
    std::vector<പൂർണ്ണസംഖ്യ>          batch_m(n_experts);
    std::vector<പൂർണ്ണസംഖ്യ>          batch_n(n_experts, m);
    std::vector<പൂർണ്ണസംഖ്യ>          batch_k(n_experts, k);
    std::vector<ദശാംശം>        alpha(n_experts, 1.0f);
    std::vector<ദശാംശം>        beta(n_experts, 0.0f);
    std::vector<സ്ഥിരം ശൂന്യം *> bias(n_experts, nullptr);
    std::vector<പൂർണ്ണസംഖ്യ>          lda_v(n_experts, lda);
    std::vector<പൂർണ്ണസംഖ്യ>          ldb_v(n_experts, ldb);
    std::vector<പൂർണ്ണസംഖ്യ>          ldc_v(n_experts, ldc);
    std::vector<തർക്കം>         is_wei_const(n_experts, സത്യം);
    std::vector<zendnnl::lowoha::matmul::matmul_params> params(n_experts, base_params);

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < n_experts; i++) {
        batch_m[i] = n[i];

        // src_scale.dims depends on this expert's row count, unlike the rest of base_params
        എങ്കിൽ constexpr (std::is_same_v<TA, block_q8_0>) {
            params[i].quant_params.src_scale.dims = {n[i], k / QK8_0};
        }
    }

    zendnnl::error_handling::status_t status = zendnnl::lowoha::matmul::group_matmul_direct(
        layout, trans_a, trans_b, batch_m, batch_n, batch_k, alpha,
        B, ldb_v, A, lda_v, bias, beta,
        C, ldc_v, is_wei_const, params);

    എങ്കിൽ (status != zendnnl::error_handling::status_t::success) {
        GGML_LOG_ERROR("%s, ZenDNN group matmul failed: status=%d\n", __func__, static_cast<പൂർണ്ണസംഖ്യ>(status));
        തിരികെ അസത്യം;
    }
    തിരികെ സത്യം;
}

സ്ഥിതം തർക്കം ggml_zendnn_group_gemm(ggml_backend_zendnn_context * ctx, int64_t m, int64_t k,
                                    സ്ഥിരം std::vector<int64_t> & n,
                                    സ്ഥിരം std::vector<സ്ഥിരം ശൂന്യം *> & A, int64_t lda,
                                    സ്ഥിരം std::vector<സ്ഥിരം ശൂന്യം *> & B, int64_t ldb,
                                    സ്ഥിരം std::vector<ശൂന്യം *> & C, int64_t ldc,
                                    പൂർണ്ണസംഖ്യ Atype, പൂർണ്ണസംഖ്യ Btype, പൂർണ്ണസംഖ്യ Ctype) {

    പ്രതിജ്ഞാ(m >= 0);
    വേണ്ടി (size_t i = 0; i < n.size(); i++) {
        പ്രതിജ്ഞാ(n[i] >= 0);
    }
    പ്രതിജ്ഞാ(k >= 0);
    പ്രതിജ്ഞാ(lda >= k);
    പ്രതിജ്ഞാ(ldb >= k);
    പ്രതിജ്ഞാ(ldc >= m);

    // categorize types
    തിരഞ്ഞെടുപ്പ് (Atype) {
        അവസ്ഥ GGML_TYPE_F32:
            എങ്കിൽ (Btype != GGML_TYPE_F32 || Ctype != GGML_TYPE_F32)
                തിരികെ അസത്യം;
            തിരികെ ggml_zendnn_group_matmul<ദശാംശം, ദശാംശം, ദശാംശം>(ctx, m, k, n, A, lda, B, ldb, C, ldc);
        അവസ്ഥ GGML_TYPE_BF16:
            എങ്കിൽ (Btype != GGML_TYPE_BF16)
                തിരികെ അസത്യം;
            എങ്കിൽ (Ctype == GGML_TYPE_BF16)
                തിരികെ ggml_zendnn_group_matmul<ggml_bf16_t, ggml_bf16_t, ggml_bf16_t>(
                    ctx, m, k, n, A, lda, B, ldb, C, ldc);
            എങ്കിൽ (Ctype == GGML_TYPE_F32)
                തിരികെ ggml_zendnn_group_matmul<ggml_bf16_t, ggml_bf16_t, ദശാംശം>(ctx, m, k, n, A, lda, B, ldb, C, ldc);
            തിരികെ അസത്യം;
        അവസ്ഥ GGML_TYPE_Q8_0:
            എങ്കിൽ (Btype != GGML_TYPE_F32 || Ctype != GGML_TYPE_F32)
                തിരികെ അസത്യം;
            തിരികെ ggml_zendnn_group_matmul<block_q8_0, ദശാംശം, ദശാംശം>(ctx, m, k, n, A, lda, B, ldb, C, ldc);
        സ്ഥിരസ്ഥിതി:
            തിരികെ അസത്യം; // unsupported type
    }
}

സ്ഥിതം ശൂന്യം ggml_zendnn_compute_forward_mul_mat_id(
    ggml_backend_zendnn_context * ctx,
    ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];  // expert weights
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];  // inputs
    സ്ഥിരം ggml_tensor * ids  = dst->src[2];  // expert ids

    GGML_TENSOR_ദ്വിമാന_OP_LOCALS

    // exit for no tokens to process
    എങ്കിൽ (ne2 == 0 || ne11 == 0) {
        തിരികെ;
    }

    ggml_type         സ്ഥിരം vec_dot_type = src0->type;
    ggml_from_float_t സ്ഥിരം from_float = ggml_get_type_traits(vec_dot_type)->from_float_ref;

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == വലുപ്പം(ദശാംശം));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(ne03 == 1);
    GGML_ASSERT(ne13 == 1);
    GGML_ASSERT(ne3  == 1);

    // row groups
    സ്ഥിരം പൂർണ്ണസംഖ്യ n_ids = ids->ne[0]; // n_expert_used
    സ്ഥിരം പൂർണ്ണസംഖ്യ n_as  = ne02;       // n_experts

    std::vector<int64_t> matrix_row_counts(n_as, 0);
    std::vector<std::vector<mmid_row_mapping>> matrix_rows(n_as);

    int64_t total_rows = 0;
    പൂർണ്ണസംഖ്യ     n_active_experts = 0;
    // group rows by expert (preprocessing step)
    വേണ്ടി (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
        വേണ്ടി (പൂർണ്ണസംഖ്യ id = 0; id < n_ids; ++id) {
            സ്ഥിരം int32_t i02 = *(സ്ഥിരം int32_t *)((സ്ഥിരം അക്ഷരം *)ids->data + iid1*ids->nb[1] + id*ids->nb[0]);

            GGML_ASSERT(i02 >= 0 && i02 < n_as);

            എങ്കിൽ (matrix_row_counts[i02] == 0) {
                n_active_experts++;
            }
            matrix_rows[i02].push_back({id, iid1});
            matrix_row_counts[i02]++;
            total_rows++;
        }
    }

    എങ്കിൽ (total_rows == 0) {
        തിരികെ; // no rows to process
    }

    സ്ഥിരം size_t row_size = ggml_row_size(vec_dot_type, ne10);

    // For Q8_0, src1 is always F32; the gather buffer must hold F32 rows (ne10*4 bytes),
    // not Q8_0-encoded rows (row_size ≈ ne10/32*34 bytes) — they differ by ~4x.
    സ്ഥിരം size_t f32_row_size = (size_t)ne10 * വലുപ്പം(ദശാംശം);
    സ്ഥിരം size_t gather_row_size = (src0->type == GGML_TYPE_Q8_0) ? f32_row_size : row_size;

    എങ്കിൽ (src1->type != vec_dot_type && src0->type != GGML_TYPE_Q8_0) {
        GGML_ASSERT(src1->type == GGML_TYPE_F32);
    }

    // size for MoE gather/scatter buffers
    സ്ഥിരം size_t wdata_cur_size = total_rows * gather_row_size;
    സ്ഥിരം size_t dst_cur_size = total_rows * ggml_row_size(dst->type, ne01);

    // allocate single buffer for all needs
    സ്ഥിരം size_t total_size = wdata_cur_size + dst_cur_size;
    എങ്കിൽ (ctx->work_size < total_size) {
        ctx->work_data.reset(പുതിയ അക്ഷരം[total_size]);
        ctx->work_size = total_size;
    }

    // partition the buffer
    അക്ഷരം * wdata_cur = ctx->work_data.get();
    അക്ഷരം * dst_cur = wdata_cur + wdata_cur_size;

    // per-expert data collected during gather, handed to ggml_zendnn_group_gemm() as one batch
    std::vector<int64_t>      expert_row_എണ്ണം(n_active_experts);
    std::vector<സ്ഥിരം ശൂന്യം *> batch_src(n_active_experts);
    std::vector<സ്ഥിരം ശൂന്യം *> batch_wei(n_active_experts);
    std::vector<ശൂന്യം *>       batch_dst(n_active_experts);

    // precompute per-expert buffer offsets and batch indices for the parallel loop below
    std::vector<int64_t> expert_wdata_off(n_as, 0);
    std::vector<int64_t> expert_dst_off(n_as, 0);
    std::vector<പൂർണ്ണസംഖ്യ>     expert_batch_idx(n_as, -1);
    {
        int64_t w_off = 0;
        int64_t d_off = 0;
        പൂർണ്ണസംഖ്യ     batch_idx = 0;
        വേണ്ടി (int64_t cur_a = 0; cur_a < n_as; ++cur_a) {
            എങ്കിൽ (matrix_row_counts[cur_a] == 0) {
                തുടരുക;
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
    #pragma omp parallel വേണ്ടി അക്കം_threads(ctx->n_threads) schedule(സ്ഥിതം)
    വേണ്ടി (int64_t cur_a = 0; cur_a < n_as; ++cur_a) {
        സ്ഥിരം int64_t cne1 = matrix_row_counts[cur_a];

        എങ്കിൽ (cne1 == 0) {
            തുടരുക;
        }

        സ്ഥിരം int64_t w_off     = expert_wdata_off[cur_a];
        സ്ഥിരം int64_t d_off     = expert_dst_off[cur_a];
        സ്ഥിരം പൂർണ്ണസംഖ്യ     batch_idx = expert_batch_idx[cur_a];

        വേണ്ടി (int64_t ir1 = 0; ir1 < cne1; ++ir1) {
            സ്ഥിരം mmid_row_mapping & row_mapping = matrix_rows[cur_a][ir1];
            സ്ഥിരം int64_t id  = row_mapping.i1;
            സ്ഥിരം int64_t i11 = id % ne11;
            സ്ഥിരം int64_t i12 = row_mapping.i2;

            സ്ഥിരം അക്ഷരം * src_row = (സ്ഥിരം അക്ഷരം *) src1->data + i11*nb11 + i12*nb12;
            ശൂന്യം *       dst_row = wdata_cur + w_off + ir1 * gather_row_size;

            എങ്കിൽ (src1->type != vec_dot_type && src0->type != GGML_TYPE_Q8_0) {
                from_float((സ്ഥിരം ദശാംശം *) src_row, dst_row, ne10);
            } അല്ലെങ്കിൽ {
                // no conversion: src1 already matches vec_dot_type, or src0 is Q8_0, whose
                // ZenDNN dynamic quantization requires the row to stay in F32
                std::memcpy(dst_row, src_row, gather_row_size);
            }
        }

        expert_row_എണ്ണം[batch_idx] = cne1;
        batch_src[batch_idx]        = wdata_cur + w_off;
        batch_wei[batch_idx]        = (സ്ഥിരം അക്ഷരം *) src0->data + cur_a * nb02;
        batch_dst[batch_idx]        = dst_cur + d_off;
    }

    എങ്കിൽ (!ggml_zendnn_group_gemm(ctx,
                                ne01,             // m
                                ne10,             // k
                                expert_row_എണ്ണം, // n (per expert)
                                batch_wei, ne00,  // A: weights, lda
                                batch_src, ne10,  // B: input, ldb
                                batch_dst, ne01,  // C: output, ldc
                                src0->type,
                                src0->type == GGML_TYPE_Q8_0 ? GGML_TYPE_F32 : vec_dot_type,
                                dst->type))
        GGML_ABORT("%s: ZenDNN group gemm failed\n", __func__);

    // scatter output rows to destination
    #pragma omp parallel വേണ്ടി അക്കം_threads(ctx->n_threads) schedule(സ്ഥിതം)
    വേണ്ടി (int64_t cur_a = 0; cur_a < n_as; ++cur_a) {
        സ്ഥിരം int64_t cne1 = matrix_row_counts[cur_a];

        എങ്കിൽ (cne1 == 0) {
            തുടരുക;
        }

        സ്ഥിരം int64_t d_off = expert_dst_off[cur_a];

        വേണ്ടി (int64_t ir1 = 0; ir1 < cne1; ++ir1) {
            സ്ഥിരം mmid_row_mapping & row_mapping = matrix_rows[cur_a][ir1];
            സ്ഥിരം int64_t id = row_mapping.i1;
            സ്ഥിരം int64_t i1 = id;
            സ്ഥിരം int64_t i2 = row_mapping.i2;

            std::memcpy(
                (അക്ഷരം *) dst->data + i1*nb1 + i2*nb2,
                dst_cur + d_off + ir1 * ggml_row_size(dst->type, ne01),
                ggml_row_size(dst->type, ne01)
            );
        }
    }
}

// backend interface

സ്ഥിതം സ്ഥിരം അക്ഷരം * ggml_backend_zendnn_get_name(ggml_backend_t backend) {
    തിരികെ "ZenDNN";

    GGML_UNUSED(backend);
}

സ്ഥിതം ശൂന്യം ggml_backend_zendnn_free(ggml_backend_t backend) {
    ggml_backend_zendnn_context * ctx = (ggml_backend_zendnn_context *)backend->context;
    delete ctx;
    delete backend;
}

സ്ഥിതം ggml_status ggml_backend_zendnn_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_zendnn_context * ctx = (ggml_backend_zendnn_context *)backend->context;

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < cgraph->n_nodes; i++) {
        ഘടന ggml_tensor * കണ്ണി = cgraph->nodes[i];

        എങ്കിൽ ((കണ്ണി->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            തുടരുക;
        }

        തിരഞ്ഞെടുപ്പ് (കണ്ണി->op) {
            അവസ്ഥ GGML_OP_MUL_MAT:
                ggml_zendnn_compute_forward_mul_mat(ctx, കണ്ണി);
                നിർത്തുക;
            അവസ്ഥ GGML_OP_MUL_MAT_ID:
                ggml_zendnn_compute_forward_mul_mat_id(ctx, കണ്ണി);
                നിർത്തുക;
            അവസ്ഥ GGML_OP_NONE:
            അവസ്ഥ GGML_OP_RESHAPE:
            അവസ്ഥ GGML_OP_VIEW:
            അവസ്ഥ GGML_OP_PERMUTE:
            അവസ്ഥ GGML_OP_TRANSPOSE:
                നിർത്തുക;

            സ്ഥിരസ്ഥിതി:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(കണ്ണി));
        }
    }

    തിരികെ GGML_STATUS_SUCCESS;

    GGML_UNUSED(backend);
}

സ്ഥിതം ഘടന ggml_backend_i ggml_backend_zendnn_i = {
    /* .get_name                = */ ggml_backend_zendnn_get_name,
    /* .free                    = */ ggml_backend_zendnn_free,
    /* .set_tensor_async        = */ ശൂന്യ,
    /* .get_tensor_async        = */ ശൂന്യ,
    /* .set_tensor_2d_async     = */ ശൂന്യ,
    /* .get_tensor_2d_async     = */ ശൂന്യ,
    /* .cpy_tensor_async        = */ ശൂന്യ,
    /* .synchronize             = */ ശൂന്യ,
    /* .graph_plan_create       = */ ശൂന്യ,
    /* .graph_plan_free         = */ ശൂന്യ,
    /* .graph_plan_update       = */ ശൂന്യ,
    /* .graph_plan_compute      = */ ശൂന്യ,
    /* .graph_compute           = */ ggml_backend_zendnn_graph_compute,
    /* .event_record            = */ ശൂന്യ,
    /* .event_wait              = */ ശൂന്യ,
    /* .graph_optimize          = */ ശൂന്യ,
};

സ്ഥിതം ggml_guid_t ggml_backend_zendnn_guid(ശൂന്യം) {
    സ്ഥിതം സ്ഥിരം അക്ഷരം * guid_str = "AMD-ZENDNN-ACCEL";
    തിരികെ reinterpret_cast<ggml_guid_t>(const_cast<അക്ഷരം*>(guid_str));
}

ggml_backend_t ggml_backend_zendnn_init(ശൂന്യം) {
    ggml_backend_zendnn_context * ctx = പുതിയ ggml_backend_zendnn_context;

    ggml_backend_t backend = പുതിയ ggml_backend {
        /* .guid    = */ ggml_backend_zendnn_guid(),
        /* .iface   = */ ggml_backend_zendnn_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_zendnn_reg(), 0),
        /* .context = */ ctx,
    };

    തിരികെ backend;
}

തർക്കം ggml_backend_is_zendnn(ggml_backend_t backend) {
    തിരികെ backend != ശൂന്യ && ggml_guid_matches(backend->guid, ggml_backend_zendnn_guid());
}

ശൂന്യം ggml_backend_zendnn_set_n_threads(ggml_backend_t backend_zendnn, പൂർണ്ണസംഖ്യ n_threads) {
    GGML_ASSERT(ggml_backend_is_zendnn(backend_zendnn));

    ggml_backend_zendnn_context * ctx = (ggml_backend_zendnn_context *)backend_zendnn->context;
    ctx->n_threads = n_threads;
}

// device interface
സ്ഥിതം സ്ഥിരം അക്ഷരം * ggml_backend_zendnn_device_get_name(ggml_backend_dev_t dev) {
    തിരികെ "ZenDNN";

    GGML_UNUSED(dev);
}
/**
 * ZenDNN is AMD's performance library providing optimized primitives and implementations
 * for deep learning workloads on AMD CPUs. It targets improved performance for common
 * neural network operations on AMD architectures. For more information, see:
 * https://www.amd.com/en/developer/zendnn.html
 */
സ്ഥിതം സ്ഥിരം അക്ഷരം * ggml_backend_zendnn_device_get_description(ggml_backend_dev_t dev) {
    തിരികെ "ZenDNN: AMD optimized primitives backend for GGML (optimized for AMD CPUs)";

    GGML_UNUSED(dev);
}

സ്ഥിതം ശൂന്യം ggml_backend_zendnn_device_get_memory(ggml_backend_dev_t dev, size_t * സ്വതന്ത്രം, size_t * total) {
    *സ്വതന്ത്രം  = 0;
    *total = 0;

    GGML_UNUSED(dev);
}

സ്ഥിതം എണ്ണൽ ggml_backend_dev_type ggml_backend_zendnn_device_get_type(ggml_backend_dev_t dev) {
    തിരികെ GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_UNUSED(dev);
}

സ്ഥിതം ശൂന്യം ggml_backend_zendnn_device_get_props(ggml_backend_dev_t dev, ഘടന ggml_backend_dev_props * props) {
    props->name        = ggml_backend_zendnn_device_get_name(dev);
    props->description = ggml_backend_zendnn_device_get_description(dev);
    props->type        = ggml_backend_zendnn_device_get_type(dev);
    ggml_backend_zendnn_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                = */ അസത്യം,
        /* .host_buffer          = */ അസത്യം,
        /* .buffer_from_host_ptr = */ സത്യം,
        /* .events               = */ അസത്യം,
        /* .mmap_support         = */ സത്യം,
    };
}

സ്ഥിതം ggml_backend_t ggml_backend_zendnn_device_init_backend(ggml_backend_dev_t dev, സ്ഥിരം അക്ഷരം * params) {
    ggml_backend_t backend = ggml_backend_zendnn_init();
    എങ്കിൽ (backend == ശൂന്യ) {
        GGML_LOG_ERROR("%s: error: failed to initialize ZenDNN backend\n", __func__);
        തിരികെ ശൂന്യ;
    }

    തിരികെ backend;

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

സ്ഥിതം ggml_backend_buffer_type_t ggml_backend_zendnn_device_get_buffer_type(ggml_backend_dev_t dev) {
    തിരികെ ggml_backend_cpu_buffer_type();

    GGML_UNUSED(dev);
}

സ്ഥിതം ggml_backend_buffer_t ggml_backend_zendnn_device_buffer_from_host_ptr(ggml_backend_dev_t dev, ശൂന്യം * ptr, size_t size, size_t max_tensor_size) {
    തിരികെ ggml_backend_cpu_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

സ്ഥിതം തർക്കം ggml_zendnn_adaptive_fallback_enabled() {
    സ്ഥിതം സ്ഥിരം തർക്കം enabled = std::getenv("GGML_ZENDNN_ADAPTIVE_FALLBACK") == nullptr ||
                                std::atoi(std::getenv("GGML_ZENDNN_ADAPTIVE_FALLBACK")) != 0;
    തിരികെ enabled;
}

സ്ഥിതം തർക്കം ggml_backend_zendnn_device_supports_op(ggml_backend_dev_t dev, സ്ഥിരം ഘടന ggml_tensor * op) {
    തിരഞ്ഞെടുപ്പ് (op->op) {
        അവസ്ഥ GGML_OP_NONE:
        അവസ്ഥ GGML_OP_RESHAPE:
        അവസ്ഥ GGML_OP_VIEW:
        അവസ്ഥ GGML_OP_PERMUTE:
        അവസ്ഥ GGML_OP_TRANSPOSE:
            തിരികെ സത്യം;

        അവസ്ഥ GGML_OP_MUL_MAT:
        അവസ്ഥ GGML_OP_MUL_MAT_ID:
        {
            സ്ഥിരം ggml_tensor * weights = op->src[0];
            സ്ഥിരം ggml_tensor * inputs = op->src[1];

            സ്ഥിരം int64_t ne10 = inputs->ne[0];
            സ്ഥിരം int64_t ne0 = op->ne[0];
            സ്ഥിരം int64_t ne1 = op->ne[1];
            സ്ഥിരം int64_t min_batch = 1;

            എങ്കിൽ(!ggml_is_contiguous(weights) || !ggml_is_contiguous(inputs)) {
                തിരികെ അസത്യം;
            }

            എങ്കിൽ (ggml_zendnn_adaptive_fallback_enabled()) {
                സ്ഥിരം int64_t K = inputs->ne[0];
                സ്ഥിരം int64_t N = (inputs->ne[1]*inputs->ne[2]*inputs->ne[3]);
                സ്ഥിരം int64_t M = weights->ne[1];
                എങ്കിൽ(K <= 256 || N <= 128 || M <= 96) {
                    തിരികെ അസത്യം;
                }

                // MUL_MAT_ID's gather+matmul+scatter approach favors a moderate expert count
                എങ്കിൽ (op->op == GGML_OP_MUL_MAT_ID) {
                    സ്ഥിരം int64_t n_experts = weights->ne[2];
                    സ്ഥിരം int64_t max_experts = 32;
                    എങ്കിൽ (n_experts > max_experts) {
                        തിരികെ അസത്യം;
                    }

                    // fall back once the average rows per expert (N / n_experts) is too thin
                    // to amortize each per-expert GEMM's overhead
                    എങ്കിൽ (N / n_experts <= 32) {
                        തിരികെ അസത്യം;
                    }
                }
            }
            അല്ലെങ്കിൽ എങ്കിൽ (ne0 < min_batch || ne1 < min_batch || ne10 < min_batch) {
                തിരികെ അസത്യം;
            }

            തിരഞ്ഞെടുപ്പ് (weights->type) {
                അവസ്ഥ GGML_TYPE_F32:
                അവസ്ഥ GGML_TYPE_BF16:
                അവസ്ഥ GGML_TYPE_Q8_0:
                    തിരികെ സത്യം;
                സ്ഥിരസ്ഥിതി:
                    തിരികെ അസത്യം;
            }
        } നിർത്തുക;

        സ്ഥിരസ്ഥിതി:
            തിരികെ അസത്യം;
    }

    GGML_UNUSED(dev);
}

സ്ഥിതം തർക്കം ggml_backend_zendnn_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    തിരികെ ggml_backend_buft_is_host(buft);

    GGML_UNUSED(dev);
}

സ്ഥിതം സ്ഥിരം ഘടന ggml_backend_device_i ggml_backend_zendnn_device_i = {
    /* .get_name               = */ ggml_backend_zendnn_device_get_name,
    /* .get_description        = */ ggml_backend_zendnn_device_get_description,
    /* .get_memory             = */ ggml_backend_zendnn_device_get_memory,
    /* .get_type               = */ ggml_backend_zendnn_device_get_type,
    /* .get_props              = */ ggml_backend_zendnn_device_get_props,
    /* .init_backend           = */ ggml_backend_zendnn_device_init_backend,
    /* .get_buffer_type        = */ ggml_backend_zendnn_device_get_buffer_type,
    /* .get_host_buffer_type   = */ ശൂന്യ,
    /* .buffer_from_host_ptr   = */ ggml_backend_zendnn_device_buffer_from_host_ptr,
    /* .supports_op            = */ ggml_backend_zendnn_device_supports_op,
    /* .supports_buft          = */ ggml_backend_zendnn_device_supports_buft,
    /* .offload_op             = */ ശൂന്യ,
    /* .event_new              = */ ശൂന്യ,
    /* .event_free             = */ ശൂന്യ,
    /* .event_synchronize      = */ ശൂന്യ,
};

// backend reg interface
സ്ഥിതം സ്ഥിരം അക്ഷരം * ggml_backend_zendnn_reg_get_name(ggml_backend_reg_t reg) {
    തിരികെ "ZenDNN";

    GGML_UNUSED(reg);
}

സ്ഥിതം size_t ggml_backend_zendnn_reg_get_device_എണ്ണം(ggml_backend_reg_t reg) {
    തിരികെ 1;

    GGML_UNUSED(reg);
}

സ്ഥിതം ggml_backend_dev_t ggml_backend_zendnn_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    സ്ഥിതം ggml_backend_device ggml_backend_zendnn_device = {
        /* .iface   = */ ggml_backend_zendnn_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };

    തിരികെ &ggml_backend_zendnn_device;
}

സ്ഥിതം ശൂന്യം * ggml_backend_zendnn_get_proc_address(ggml_backend_reg_t reg, സ്ഥിരം അക്ഷരം * name) {
    എങ്കിൽ (std::strcmp(name, "ggml_backend_set_n_threads") == 0) {
        തിരികെ (ശൂന്യം *) ggml_backend_zendnn_set_n_threads;
    }
    തിരികെ ശൂന്യ;

    GGML_UNUSED(reg);
    GGML_UNUSED(name);
}

സ്ഥിതം സ്ഥിരം ഘടന ggml_backend_reg_i ggml_backend_zendnn_reg_i = {
    /* .get_name         = */ ggml_backend_zendnn_reg_get_name,
    /* .get_device_count = */ ggml_backend_zendnn_reg_get_device_എണ്ണം,
    /* .get_device       = */ ggml_backend_zendnn_reg_get_device,
    /* .get_proc_address = */ ggml_backend_zendnn_get_proc_address,
};

ggml_backend_reg_t ggml_backend_zendnn_reg(ശൂന്യം) {
    സ്ഥിതം ഘടന ggml_backend_reg ggml_backend_zendnn_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_zendnn_reg_i,
        /* .context     = */ ശൂന്യ,
    };

    തിരികെ &ggml_backend_zendnn_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_zendnn_reg)
