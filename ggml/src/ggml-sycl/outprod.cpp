#অন্তর্ভাব "outprod.hpp"
#অন্তর্ভাব "convert.hpp"

শূন্য ggml_sycl_op_out_prod(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ধ্রুবক ggml_tensor *src0 = dst->src[0];
    ধ্রুবক ggml_tensor *src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_Q1_0);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));

    GGML_TENSOR_দ্বিমিক_OP_LOCALS

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
    ধ্রুবক দশমিক * src0_d = (ধ্রুবক দশমিক *) src0->তথ্য;
    ধ্রুবক দশমিক * src1_d = (ধ্রুবক দশমিক *) src1->তথ্য;
    দশমিক * dst_d = (দশমিক *) dst->তথ্য;

    ggml_sycl_pool_alloc<দশমিক> src0_as_f32(ctx.pool());
    int64_t src0_nb02 = nb02;
    int64_t src0_nb03 = nb03;
    যদি (src0->type == GGML_TYPE_Q1_0) {
        scope_op_debug_print scope_dbg_print(__func__, "/to_fp32_sycl", dst, /*num_src=*/2,
                                             " : converting src0 Q1_0 to fp32");
        src0_d = src0_as_f32.alloc(ne00 * ne01 * ne02 * ne03);
        ধ্রুবক to_fp32_sycl_t to_fp32_sycl = ggml_get_to_fp32_sycl(src0->type, dst);
        GGML_ASSERT(to_fp32_sycl != nullptr);
        to_fp32_sycl(src0->তথ্য, const_cast<দশমিক *>(src0_d), ne00 * ne01 * ne02 * ne03, stream);

        // Dequantized src0 buffer is contiguous fp32 [ne00, ne01, ne02, ne03].
        src0_nb02 = ne00 * ne01 * (int64_t) আকার(দশমিক);
        src0_nb03 = ne00 * ne01 * ne02 * (int64_t) আকার(দশমিক);
    }

    // GEMM parameters
    ধ্রুবক দশমিক alpha = 1.0f;
    ধ্রুবক দশমিক beta = 0.0f;

    // Handle transposition of src1
    ধ্রুবক যৌক্তিক src1_T = ggml_is_transposed(src1);
    ধ্রুবক oneapi::mkl::transpose src1_op = src1_T ? oneapi::mkl::transpose::nontrans : oneapi::mkl::transpose::trans;
    ধ্রুবক int64_t ldb = (src1_T ? nb10 : nb11) / আকার(দশমিক);

    ধ্রুবক int64_t r2 = ne2 / ne02;
    ধ্রুবক int64_t r3 = ne3 / ne03;

    চেষ্টা {
        // OUT_PROD applies independently to each (i2, i3) destination plane.
        জন্য (int64_t i3 = 0; i3 < ne3; ++i3) {
            জন্য (int64_t i2 = 0; i2 < ne2; ++i2) {
                ধ্রুবক int64_t i03 = i3 / r3;
                ধ্রুবক int64_t i02 = i2 / r2;

                ধ্রুবক দশমিক * src0_plane = (ধ্রুবক দশমিক *) ((ধ্রুবক অক্ষর *) src0_d + i02 * src0_nb02 + i03 * src0_nb03);
                ধ্রুবক দশমিক * src1_plane = (ধ্রুবক দশমিক *) ((ধ্রুবক অক্ষর *) src1_d + i2 * nb12 + i3 * nb13);
                দশমিক * dst_plane = (দশমিক *) ((অক্ষর *) dst_d + i2 * nb2 + i3 * nb3);

                // Perform matrix multiplication using oneMKL GEMM
                oneapi::mkl::blas::column_major::gemm(*stream, oneapi::mkl::transpose::nontrans, src1_op,
                                                      ne0, ne1, ne01, alpha, src0_plane, ne00,
                                                      src1_plane, ldb, beta, dst_plane, ne0);
            }
        }
    } ধরো (sycl::exception ধ্রুবক& exc) {
        std::cerr << exc.what() << std::endl;
        GGML_ASSERT(মিথ্যা);
    }
}
