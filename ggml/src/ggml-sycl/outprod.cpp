#ਅਨ੍ਤਰ੍ਭਾਵ "outprod.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "convert.hpp"

ਖਾਲੀ ggml_sycl_op_ਬਾਹਰਲਾ_prod(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ਅਚਲ ggml_tensor *src0 = dst->src[0];
    ਅਚਲ ggml_tensor *src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_Q1_0);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));

    GGML_TENSOR_ਦੁਆਧਾਰੀ_OP_LOCALS

    // Get SYCL queue
    dpct::ਕਤਾਰ_ptr stream = ctx.stream();

    // Dimension checks
    GGML_ASSERT(ne01 == ne11);  // Inner dimensions must match
    GGML_ASSERT(ne0 == ne00);   // Output rows match src0 rows
    GGML_ASSERT(ne1 == ne10);   // Output cols match src1 cols
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);
    GGML_ASSERT(ne2 % ne02 == 0);
    GGML_ASSERT(ne3 % ne03 == 0);

    // Get data pointers
    ਅਚਲ ਦਸ਼ਮਲਵ * src0_d = (ਅਚਲ ਦਸ਼ਮਲਵ *) src0->ਅੰਕੜੇ;
    ਅਚਲ ਦਸ਼ਮਲਵ * src1_d = (ਅਚਲ ਦਸ਼ਮਲਵ *) src1->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ * dst_d = (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ;

    ggml_sycl_pool_alloc<ਦਸ਼ਮਲਵ> src0_as_f32(ctx.pool());
    int64_t src0_nb02 = nb02;
    int64_t src0_nb03 = nb03;
    ਜੇ (src0->type == GGML_TYPE_Q1_0) {
        scope_op_debug_print scope_dbg_print(__func__, "/to_fp32_sycl", dst, /*num_src=*/2,
                                             " : converting src0 Q1_0 to fp32");
        src0_d = src0_as_f32.alloc(ne00 * ne01 * ne02 * ne03);
        ਅਚਲ ਮੰਜ਼ਿਲ_fp32_sycl_t ਮੰਜ਼ਿਲ_fp32_sycl = ggml_get_ਮੰਜ਼ਿਲ_fp32_sycl(src0->type, dst);
        GGML_ASSERT(ਮੰਜ਼ਿਲ_fp32_sycl != ਖਾਲੀ_ਸੂਚਕ);
        ਮੰਜ਼ਿਲ_fp32_sycl(src0->ਅੰਕੜੇ, const_cast<ਦਸ਼ਮਲਵ *>(src0_d), ne00 * ne01 * ne02 * ne03, stream);

        // Dequantized src0 buffer is contiguous fp32 [ne00, ne01, ne02, ne03].
        src0_nb02 = ne00 * ne01 * (int64_t) ਆਕਾਰ(ਦਸ਼ਮਲਵ);
        src0_nb03 = ne00 * ne01 * ne02 * (int64_t) ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    }

    // GEMM parameters
    ਅਚਲ ਦਸ਼ਮਲਵ alpha = 1.0f;
    ਅਚਲ ਦਸ਼ਮਲਵ beta = 0.0f;

    // Handle transposition of src1
    ਅਚਲ ਤਾਰਕਿਕ src1_T = ggml_is_transposed(src1);
    ਅਚਲ oneapi::mkl::transpose src1_op = src1_T ? oneapi::mkl::transpose::nontrans : oneapi::mkl::transpose::trans;
    ਅਚਲ int64_t ldb = (src1_T ? nb10 : nb11) / ਆਕਾਰ(ਦਸ਼ਮਲਵ);

    ਅਚਲ int64_t r2 = ne2 / ne02;
    ਅਚਲ int64_t r3 = ne3 / ne03;

    ਕੋਸ਼ਿਸ਼ {
        // OUT_PROD applies independently to each (i2, i3) destination plane.
        ਲਈ (int64_t i3 = 0; i3 < ne3; ++i3) {
            ਲਈ (int64_t i2 = 0; i2 < ne2; ++i2) {
                ਅਚਲ int64_t i03 = i3 / r3;
                ਅਚਲ int64_t i02 = i2 / r2;

                ਅਚਲ ਦਸ਼ਮਲਵ * src0_plane = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) src0_d + i02 * src0_nb02 + i03 * src0_nb03);
                ਅਚਲ ਦਸ਼ਮਲਵ * src1_plane = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) src1_d + i2 * nb12 + i3 * nb13);
                ਦਸ਼ਮਲਵ * dst_plane = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst_d + i2 * nb2 + i3 * nb3);

                // Perform matrix multiplication using oneMKL GEMM
                oneapi::mkl::blas::ਕਾਲਮ_major::gemm(*stream, oneapi::mkl::transpose::nontrans, src1_op,
                                                      ne0, ne1, ne01, alpha, src0_plane, ne00,
                                                      src1_plane, ldb, beta, dst_plane, ne0);
            }
        }
    } ਫੜੋ (sycl::exception ਅਚਲ& exc) {
        std::cerr << exc.what() << std::ਸਤਰ_ਅੰਤ;
        GGML_ASSERT(ਝੂਠ);
    }
}
