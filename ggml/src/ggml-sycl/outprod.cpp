#अन्तर्भाव "outprod.hpp"
#अन्तर्भाव "convert.hpp"

शून्य ggml_sycl_op_out_prod(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ध्रुव ggml_tensor *src0 = dst->src[0];
    ध्रुव ggml_tensor *src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_Q1_0);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));

    GGML_TENSOR_द्विमान_OP_LOCALS

    // Get SYCL queue
    dpct::queue_ptr stream = ctx.stream();

    // Dimension checks
    GGML_ASSERT(ne01 == ne11);  // Inner dimensions must match
    GGML_ASSERT(ne0 == ne00);   // Output rows match src0 rows
    GGML_ASSERT(ne1 == ne10);   // Output cols match src1 cols
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);
    GGML_ASSERT(ne2 % ne02 == 0);
    GGML_ASSERT(ne3 % ne03 == 0);

    // Get data pointers
    ध्रुव भिन्न * src0_d = (ध्रुव भिन्न *) src0->data;
    ध्रुव भिन्न * src1_d = (ध्रुव भिन्न *) src1->data;
    भिन्न * dst_d = (भिन्न *) dst->data;

    ggml_sycl_pool_alloc<भिन्न> src0_as_f32(ctx.pool());
    int64_t src0_nb02 = nb02;
    int64_t src0_nb03 = nb03;
    यदि (src0->type == GGML_TYPE_Q1_0) {
        scope_op_debug_print scope_dbg_print(__func__, "/to_fp32_sycl", dst, /*num_src=*/2,
                                             " : converting src0 Q1_0 to fp32");
        src0_d = src0_as_f32.alloc(ne00 * ne01 * ne02 * ne03);
        ध्रुव to_fp32_sycl_t to_fp32_sycl = ggml_get_to_fp32_sycl(src0->type, dst);
        GGML_ASSERT(to_fp32_sycl != nullptr);
        to_fp32_sycl(src0->data, const_cast<भिन्न *>(src0_d), ne00 * ne01 * ne02 * ne03, stream);

        // Dequantized src0 buffer is contiguous fp32 [ne00, ne01, ne02, ne03].
        src0_nb02 = ne00 * ne01 * (int64_t) परिमाण(भिन्न);
        src0_nb03 = ne00 * ne01 * ne02 * (int64_t) परिमाण(भिन्न);
    }

    // GEMM parameters
    ध्रुव भिन्न alpha = 1.0f;
    ध्रुव भिन्न beta = 0.0f;

    // Handle transposition of src1
    ध्रुव तार्किक src1_T = ggml_is_transposed(src1);
    ध्रुव oneapi::mkl::transpose src1_op = src1_T ? oneapi::mkl::transpose::nontrans : oneapi::mkl::transpose::trans;
    ध्रुव int64_t ldb = (src1_T ? nb10 : nb11) / परिमाण(भिन्न);

    ध्रुव int64_t r2 = ne2 / ne02;
    ध्रुव int64_t r3 = ne3 / ne03;

    यत्न {
        // OUT_PROD applies independently to each (i2, i3) destination plane.
        प्रति (int64_t i3 = 0; i3 < ne3; ++i3) {
            प्रति (int64_t i2 = 0; i2 < ne2; ++i2) {
                ध्रुव int64_t i03 = i3 / r3;
                ध्रुव int64_t i02 = i2 / r2;

                ध्रुव भिन्न * src0_plane = (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src0_d + i02 * src0_nb02 + i03 * src0_nb03);
                ध्रुव भिन्न * src1_plane = (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src1_d + i2 * nb12 + i3 * nb13);
                भिन्न * dst_plane = (भिन्न *) ((वर्ण *) dst_d + i2 * nb2 + i3 * nb3);

                // Perform matrix multiplication using oneMKL GEMM
                oneapi::mkl::blas::column_major::gemm(*stream, oneapi::mkl::transpose::nontrans, src1_op,
                                                      ne0, ne1, ne01, alpha, src0_plane, ne00,
                                                      src1_plane, ldb, beta, dst_plane, ne0);
            }
        }
    } गृहाण (sycl::exception ध्रुव& exc) {
        std::cerr << exc.what() << std::endl;
        GGML_ASSERT(असत्य);
    }
}
