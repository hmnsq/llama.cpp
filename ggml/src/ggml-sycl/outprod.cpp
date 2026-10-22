#ಅನ್ತರ್ಭಾವ "outprod.hpp"
#ಅನ್ತರ್ಭಾವ "convert.hpp"

ಶೂನ್ಯ ggml_sycl_op_ಹೊರಹೋಗು_prod(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ಸ್ಥಿರಾಂಕ ggml_tensor *src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor *src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_Q1_0);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));

    GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS

    // Get SYCL queue
    dpct::ಸರತಿ_ptr stream = ctx.stream();

    // Dimension checks
    GGML_ASSERT(ne01 == ne11);  // Inner dimensions must match
    GGML_ASSERT(ne0 == ne00);   // Output rows match src0 rows
    GGML_ASSERT(ne1 == ne10);   // Output cols match src1 cols
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);
    GGML_ASSERT(ne2 % ne02 == 0);
    GGML_ASSERT(ne3 % ne03 == 0);

    // Get data pointers
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src0_d = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src0->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src1_d = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src1->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ * dst_d = (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ;

    ggml_sycl_pool_alloc<ದಶಮಾಂಶ> src0_as_f32(ctx.pool());
    int64_t src0_nb02 = nb02;
    int64_t src0_nb03 = nb03;
    ಆದರೆ (src0->type == GGML_TYPE_Q1_0) {
        scope_op_debug_print scope_dbg_print(__func__, "/to_fp32_sycl", dst, /*num_src=*/2,
                                             " : converting src0 Q1_0 to fp32");
        src0_d = src0_as_f32.alloc(ne00 * ne01 * ne02 * ne03);
        ಸ್ಥಿರಾಂಕ ಗಮ್ಯ_fp32_sycl_t ಗಮ್ಯ_fp32_sycl = ggml_get_ಗಮ್ಯ_fp32_sycl(src0->type, dst);
        GGML_ASSERT(ಗಮ್ಯ_fp32_sycl != ಶೂನ್ಯಸೂಚಕ);
        ಗಮ್ಯ_fp32_sycl(src0->ದತ್ತಾಂಶ, const_cast<ದಶಮಾಂಶ *>(src0_d), ne00 * ne01 * ne02 * ne03, stream);

        // Dequantized src0 buffer is contiguous fp32 [ne00, ne01, ne02, ne03].
        src0_nb02 = ne00 * ne01 * (int64_t) ಗಾತ್ರ(ದಶಮಾಂಶ);
        src0_nb03 = ne00 * ne01 * ne02 * (int64_t) ಗಾತ್ರ(ದಶಮಾಂಶ);
    }

    // GEMM parameters
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ alpha = 1.0f;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ beta = 0.0f;

    // Handle transposition of src1
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ src1_T = ggml_is_transposed(src1);
    ಸ್ಥಿರಾಂಕ oneapi::mkl::transpose src1_op = src1_T ? oneapi::mkl::transpose::nontrans : oneapi::mkl::transpose::trans;
    ಸ್ಥಿರಾಂಕ int64_t ldb = (src1_T ? nb10 : nb11) / ಗಾತ್ರ(ದಶಮಾಂಶ);

    ಸ್ಥಿರಾಂಕ int64_t r2 = ne2 / ne02;
    ಸ್ಥಿರಾಂಕ int64_t r3 = ne3 / ne03;

    ಪ್ರಯತ್ನಿಸು {
        // OUT_PROD applies independently to each (i2, i3) destination plane.
        ಗಾಗಿ (int64_t i3 = 0; i3 < ne3; ++i3) {
            ಗಾಗಿ (int64_t i2 = 0; i2 < ne2; ++i2) {
                ಸ್ಥಿರಾಂಕ int64_t i03 = i3 / r3;
                ಸ್ಥಿರಾಂಕ int64_t i02 = i2 / r2;

                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src0_plane = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src0_d + i02 * src0_nb02 + i03 * src0_nb03);
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src1_plane = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src1_d + i2 * nb12 + i3 * nb13);
                ದಶಮಾಂಶ * dst_plane = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst_d + i2 * nb2 + i3 * nb3);

                // Perform matrix multiplication using oneMKL GEMM
                oneapi::mkl::blas::ಸ್ತಂಭ_major::gemm(*stream, oneapi::mkl::transpose::nontrans, src1_op,
                                                      ne0, ne1, ne01, alpha, src0_plane, ne00,
                                                      src1_plane, ldb, beta, dst_plane, ne0);
            }
        }
    } ಹಿಡಿ (sycl::exception ಸ್ಥಿರಾಂಕ& exc) {
        std::cerr << exc.what() << std::ಸಾಲು_ಅಂತ್ಯ;
        GGML_ASSERT(ಸುಳ್ಳು);
    }
}
