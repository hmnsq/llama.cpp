#అన్తర్భావ "outprod.hpp"
#అన్తర్భావ "convert.hpp"

శూన్య ggml_sycl_op_బహిర్గతం_prod(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    స్థిరరాశి ggml_tensor *src0 = dst->src[0];
    స్థిరరాశి ggml_tensor *src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_Q1_0);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));

    GGML_TENSOR_ద్విమాన_OP_LOCALS

    // Get SYCL queue
    dpct::వరుస_ptr stream = ctx.stream();

    // Dimension checks
    GGML_ASSERT(ne01 == ne11);  // Inner dimensions must match
    GGML_ASSERT(ne0 == ne00);   // Output rows match src0 rows
    GGML_ASSERT(ne1 == ne10);   // Output cols match src1 cols
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);
    GGML_ASSERT(ne2 % ne02 == 0);
    GGML_ASSERT(ne3 % ne03 == 0);

    // Get data pointers
    స్థిరరాశి దశాంశం * src0_d = (స్థిరరాశి దశాంశం *) src0->దత్తాంశం;
    స్థిరరాశి దశాంశం * src1_d = (స్థిరరాశి దశాంశం *) src1->దత్తాంశం;
    దశాంశం * dst_d = (దశాంశం *) dst->దత్తాంశం;

    ggml_sycl_pool_alloc<దశాంశం> src0_as_f32(ctx.pool());
    int64_t src0_nb02 = nb02;
    int64_t src0_nb03 = nb03;
    ఒకవేళ (src0->type == GGML_TYPE_Q1_0) {
        scope_op_debug_print scope_dbg_print(__func__, "/to_fp32_sycl", dst, /*num_src=*/2,
                                             " : converting src0 Q1_0 to fp32");
        src0_d = src0_as_f32.alloc(ne00 * ne01 * ne02 * ne03);
        స్థిరరాశి గమ్యం_fp32_sycl_t గమ్యం_fp32_sycl = ggml_get_గమ్యం_fp32_sycl(src0->type, dst);
        GGML_ASSERT(గమ్యం_fp32_sycl != శూన్యసూచిక);
        గమ్యం_fp32_sycl(src0->దత్తాంశం, const_cast<దశాంశం *>(src0_d), ne00 * ne01 * ne02 * ne03, stream);

        // Dequantized src0 buffer is contiguous fp32 [ne00, ne01, ne02, ne03].
        src0_nb02 = ne00 * ne01 * (int64_t) పరిమాణం(దశాంశం);
        src0_nb03 = ne00 * ne01 * ne02 * (int64_t) పరిమాణం(దశాంశం);
    }

    // GEMM parameters
    స్థిరరాశి దశాంశం alpha = 1.0f;
    స్థిరరాశి దశాంశం beta = 0.0f;

    // Handle transposition of src1
    స్థిరరాశి తార్కిక src1_T = ggml_is_transposed(src1);
    స్థిరరాశి oneapi::mkl::transpose src1_op = src1_T ? oneapi::mkl::transpose::nontrans : oneapi::mkl::transpose::trans;
    స్థిరరాశి int64_t ldb = (src1_T ? nb10 : nb11) / పరిమాణం(దశాంశం);

    స్థిరరాశి int64_t r2 = ne2 / ne02;
    స్థిరరాశి int64_t r3 = ne3 / ne03;

    ప్రయత్నించు {
        // OUT_PROD applies independently to each (i2, i3) destination plane.
        కొరకు (int64_t i3 = 0; i3 < ne3; ++i3) {
            కొరకు (int64_t i2 = 0; i2 < ne2; ++i2) {
                స్థిరరాశి int64_t i03 = i3 / r3;
                స్థిరరాశి int64_t i02 = i2 / r2;

                స్థిరరాశి దశాంశం * src0_plane = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) src0_d + i02 * src0_nb02 + i03 * src0_nb03);
                స్థిరరాశి దశాంశం * src1_plane = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) src1_d + i2 * nb12 + i3 * nb13);
                దశాంశం * dst_plane = (దశాంశం *) ((అక్షరం *) dst_d + i2 * nb2 + i3 * nb3);

                // Perform matrix multiplication using oneMKL GEMM
                oneapi::mkl::blas::స్తంభం_major::gemm(*stream, oneapi::mkl::transpose::nontrans, src1_op,
                                                      ne0, ne1, ne01, alpha, src0_plane, ne00,
                                                      src1_plane, ldb, beta, dst_plane, ne0);
            }
        }
    } పట్టుకో (sycl::exception స్థిరరాశి& exc) {
        std::cerr << exc.what() << std::పంక్తి_ముగింపు;
        GGML_ASSERT(అబద్ధం);
    }
}
