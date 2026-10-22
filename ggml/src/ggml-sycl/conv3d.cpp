#ಅನ್ತರ್ಭಾವ "conv3d.hpp"

ಸ್ಥಿರ ಸಾಲೊಳಗೆ int64_t ggml_sycl_conv3d_calc_patch_ಒಟ್ಟು(ಸ್ಥಿರಾಂಕ ggml_tensor * dst, int32_t n) {
    ಹಿಂತಿರುಗಿ (int64_t) n * dst->ne[0] * dst->ne[1] * dst->ne[2];
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ int64_t ggml_sycl_conv3d_calc_knl_n_ಒಟ್ಟು(ಸ್ಥಿರಾಂಕ ggml_tensor * src0, int32_t c) {
    ಹಿಂತಿರುಗಿ (int64_t) src0->ne[0] * src0->ne[1] * src0->ne[2] * c;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_sycl_conv3d_write_output(
        ಸ್ಥಿರಾಂಕ ggml_tensor * dst,
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src, ದಶಮಾಂಶ * dst_ದತ್ತಾಂಶ,
        int64_t patch_ಒಟ್ಟು, int64_t oc,
        int64_t dst_w, int64_t dst_h, int64_t dst_d,
        dpct::ಸರತಿ_ptr stream) {
    ಸ್ಥಿರಾಂಕ int64_t dst_nb0 = dst->nb[0];
    ಸ್ಥಿರಾಂಕ int64_t dst_nb1 = dst->nb[1];
    ಸ್ಥಿರಾಂಕ int64_t dst_nb2 = dst->nb[2];
    ಸ್ಥಿರಾಂಕ int64_t dst_nb3 = dst->nb[3];
    ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು = patch_ಒಟ್ಟು * oc;
    ಸ್ಥಿರಾಂಕ int64_t ಖಂಡ_ಗಾತ್ರ_3 = 256;
    ಸ್ಥಿರಾಂಕ int64_t ಅಂಕೆ_work_ವಸ್ತುಗಳು = ((ಒಟ್ಟು + ಖಂಡ_ಗಾತ್ರ_3 - 1) / ಖಂಡ_ಗಾತ್ರ_3) * ಖಂಡ_ಗಾತ್ರ_3;

    stream->parallel_for(sycl::range<1>(ಅಂಕೆ_work_ವಸ್ತುಗಳು), [=](sycl::id<1> id) {
        ಸ್ಥಿರಾಂಕ int64_t i = id[0];
        ಆದರೆ (i >= ಒಟ್ಟು) {
            ಹಿಂತಿರುಗಿ;
        }

        ಸ್ಥಿರಾಂಕ int64_t patch_idx = i / oc;
        ಸ್ಥಿರಾಂಕ int64_t ಹೊರಹೋಗು_ch = i % oc;
        ಸ್ಥಿರಾಂಕ int64_t p_ಒಳ_batch = patch_idx % (dst_w * dst_h * dst_d);
        ಸ್ಥಿರಾಂಕ int64_t batch_idx = patch_idx / (dst_w * dst_h * dst_d);
        ಸ್ಥಿರಾಂಕ int64_t dst_z = p_ಒಳ_batch / (dst_w * dst_h);
        ಸ್ಥಿರಾಂಕ int64_t dst_y = (p_ಒಳ_batch % (dst_w * dst_h)) / dst_w;
        ಸ್ಥಿರಾಂಕ int64_t dst_x = p_ಒಳ_batch % dst_w;
        ಸ್ಥಿರಾಂಕ int64_t ocn_idx = batch_idx * oc + ಹೊರಹೋಗು_ch;

        ಸ್ಥಿರಾಂಕ int64_t dst_ಸ್ಥಾನಾಂತರ = dst_x * dst_nb0 + dst_y * dst_nb1 + dst_z * dst_nb2 + ocn_idx * dst_nb3;
        // `src` is a column-major (m x n) GEMM output where m == patch_total, n == oc.
        // GEMM stores element (row, col) at index `row + col*m`, so compute index accordingly.
        ಸ್ಥಿರಾಂಕ int64_t src_ಸೂಚ್ಯಂಕ = patch_idx + ಹೊರಹೋಗು_ch * patch_ಒಟ್ಟು;
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ value = src[src_ಸೂಚ್ಯಂಕ];
        *(ದಶಮಾಂಶ *)((ಅಕ್ಷರ *)dst_ದತ್ತಾಂಶ + dst_ಸ್ಥಾನಾಂತರ) = value;
    });
}

ಶೂನ್ಯ ggml_sycl_op_conv_3d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));

    ಸ್ಥಿರಾಂಕ int32_t * opts = (ಸ್ಥಿರಾಂಕ int32_t *) dst->op_params;
    ಸ್ಥಿರಾಂಕ int32_t s0 = opts[0];
    ಸ್ಥಿರಾಂಕ int32_t s1 = opts[1];
    ಸ್ಥಿರಾಂಕ int32_t s2 = opts[2];
    ಸ್ಥಿರಾಂಕ int32_t p0 = opts[3];
    ಸ್ಥಿರಾಂಕ int32_t p1 = opts[4];
    ಸ್ಥಿರಾಂಕ int32_t p2 = opts[5];
    ಸ್ಥಿರಾಂಕ int32_t d0 = opts[6];
    ಸ್ಥಿರಾಂಕ int32_t d1 = opts[7];
    ಸ್ಥಿರಾಂಕ int32_t d2 = opts[8];
    ಸ್ಥಿರಾಂಕ int32_t c  = opts[9];
    ಸ್ಥಿರಾಂಕ int32_t n  = opts[10];
    ಸ್ಥಿರಾಂಕ int32_t oc = opts[11];

    ಸ್ಥಿರಾಂಕ int64_t knl_w = src0->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t knl_h = src0->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t knl_d = src0->ne[2];

    ಸ್ಥಿರಾಂಕ int64_t patch_ಒಟ್ಟು = ggml_sycl_conv3d_calc_patch_ಒಟ್ಟು(dst, n);
    ಸ್ಥಿರಾಂಕ int64_t knl_n_ಒಟ್ಟು = ggml_sycl_conv3d_calc_knl_n_ಒಟ್ಟು(src0, c);

    ಸ್ಥಿರಾಂಕ size_t kernel_type_ಗಾತ್ರ = ggml_element_ಗಾತ್ರ(src0);

    ggml_sycl_pool_alloc<ದಶಮಾಂಶ> gemm_output(ctx.pool());
    gemm_output.alloc((size_t) patch_ಒಟ್ಟು * oc);

    ggml_tensor dst_mat = {};
    dst_mat.type = GGML_TYPE_F32;
    dst_mat.ne[0] = patch_ಒಟ್ಟು;
    dst_mat.ne[1] = oc;
    dst_mat.ne[2] = 1;
    dst_mat.ne[3] = 1;
    dst_mat.nb[0] = ಗಾತ್ರ(ದಶಮಾಂಶ);
    dst_mat.nb[1] = dst_mat.nb[0] * dst_mat.ne[0];
    dst_mat.nb[2] = dst_mat.nb[1];
    dst_mat.nb[3] = dst_mat.nb[2];
    dst_mat.ದತ್ತಾಂಶ = gemm_output.get();
    dst_mat.ಬಫರ್ = dst->ಬಫರ್;
    dst_mat.extra = dst->extra;

    dpct::ಸರತಿ_ptr stream = ctx.stream();

    // allocate packed arrays: A_packed (k x m), B_packed (k x n)
    ggml_sycl_pool_alloc<ದಶಮಾಂಶ> A_packed_alloc(ctx.pool());
    ggml_sycl_pool_alloc<ದಶಮಾಂಶ> B_packed_alloc(ctx.pool());
    A_packed_alloc.alloc((size_t) knl_n_ಒಟ್ಟು * patch_ಒಟ್ಟು);
    B_packed_alloc.alloc((size_t) knl_n_ಒಟ್ಟು * oc);

    ದಶಮಾಂಶ * A_packed = A_packed_alloc.get();
    ದಶಮಾಂಶ * B_packed = B_packed_alloc.get();

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ m = (ಪೂರ್ಣಾಂಕ) patch_ಒಟ್ಟು;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_gemm = (ಪೂರ್ಣಾಂಕ) oc;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ k = (ಪೂರ್ಣಾಂಕ) knl_n_ಒಟ್ಟು;

    // Combined kernel: im2col -> pack A, and pack B simultaneously
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * src1_ತಳ = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src1->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * src0_ತಳ = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ int64_t src1_nb0 = src1->nb[0];
    ಸ್ಥಿರಾಂಕ int64_t src1_nb1 = src1->nb[1];
    ಸ್ಥಿರಾಂಕ int64_t src1_nb2 = src1->nb[2];
    ಸ್ಥಿರಾಂಕ int64_t src1_nb3 = src1->nb[3];
    ಸ್ಥಿರಾಂಕ int64_t src1_w = src1->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t src1_h = src1->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t src1_d = src1->ne[2];

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ src0_is_f32 = (src0->type == GGML_TYPE_F32);

    // Compute correct strides for src0 as (knl_n_total, oc) matrix
    ಸ್ಥಿರಾಂಕ int64_t src0_packed_nb0 = kernel_type_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ int64_t src0_packed_nb1 = kernel_type_ಗಾತ್ರ * knl_n_ಒಟ್ಟು;

    ಸ್ಥಿರಾಂಕ int64_t KW = knl_w;
    ಸ್ಥಿರಾಂಕ int64_t KH = knl_h;
    ಸ್ಥಿರಾಂಕ int64_t KD = knl_d;
    ಸ್ಥಿರಾಂಕ int64_t PW = dst->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t PH = dst->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t PD = dst->ne[2];

    // Pack A (with inline im2col): for each (row, col) in k x m matrix
    ಸ್ಥಿರಾಂಕ int64_t A_ಒಟ್ಟು = (int64_t)k * m;
    ಸ್ಥಿರಾಂಕ int64_t A_ಖಂಡ_ಗಾತ್ರ = 256;
    ಸ್ಥಿರಾಂಕ int64_t A_ಅಂಕೆ_work = ((A_ಒಟ್ಟು + A_ಖಂಡ_ಗಾತ್ರ - 1) / A_ಖಂಡ_ಗಾತ್ರ) * A_ಖಂಡ_ಗಾತ್ರ;

    stream->parallel_for(sycl::range<1>(A_ಅಂಕೆ_work), [=](sycl::id<1> id) {
        ಸ್ಥಿರಾಂಕ int64_t t = id[0];
        ಆದರೆ (t >= A_ಒಟ್ಟು) ಹಿಂತಿರುಗಿ;

        ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲು = t % k;
        ಸ್ಥಿರಾಂಕ int64_t ಲಂಬಸಾಲು = t / k;

        // Inline im2col for this element
        ಸ್ಥಿರಾಂಕ int64_t k_ಸೂಚ್ಯಂಕ = ಅಡ್ಡಸಾಲು;
        ಸ್ಥಿರಾಂಕ int64_t patch_idx = ಲಂಬಸಾಲು;

        ಸ್ಥಿರಾಂಕ int64_t ic = k_ಸೂಚ್ಯಂಕ / (KD * KH * KW);
        ಸ್ಥಿರಾಂಕ int64_t rem = k_ಸೂಚ್ಯಂಕ - ic * (KD * KH * KW);
        ಸ್ಥಿರಾಂಕ int64_t kz = rem / (KH * KW);
        ಸ್ಥಿರಾಂಕ int64_t rem2 = rem - kz * (KH * KW);
        ಸ್ಥಿರಾಂಕ int64_t ky = rem2 / KW;
        ಸ್ಥಿರಾಂಕ int64_t kx = rem2 % KW;

        ಸ್ಥಿರಾಂಕ int64_t p_ಒಳ_batch = patch_idx % (PW * PH * PD);
        ಸ್ಥಿರಾಂಕ int64_t batch_idx = patch_idx / (PW * PH * PD);
        ಸ್ಥಿರಾಂಕ int64_t dst_z = p_ಒಳ_batch / (PW * PH);
        ಸ್ಥಿರಾಂಕ int64_t dst_y = (p_ಒಳ_batch % (PW * PH)) / PW;
        ಸ್ಥಿರಾಂಕ int64_t dst_x = p_ಒಳ_batch % PW;

        ಸ್ಥಿರಾಂಕ int64_t sx = dst_x * s0 + kx * d0 - p0;
        ಸ್ಥಿರಾಂಕ int64_t sy = dst_y * s1 + ky * d1 - p1;
        ಸ್ಥಿರಾಂಕ int64_t sz = dst_z * s2 + kz * d2 - p2;

        ದಶಮಾಂಶ val = 0.0f;
        ಆದರೆ (sx >= 0 && sx < src1_w && sy >= 0 && sy < src1_h && sz >= 0 && sz < src1_d) {
            ಸ್ಥಿರಾಂಕ int64_t channel_idx = batch_idx * c + ic;
            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ptr = src1_ತಳ + sx * src1_nb0 + sy * src1_nb1 + sz * src1_nb2 + channel_idx * src1_nb3;
            val = *(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ptr;
        }
        A_packed[ಅಡ್ಡಸಾಲು + ಲಂಬಸಾಲು * (int64_t)k] = val;
    });

    // Pack B: for each (row, col) in k x n_gemm matrix
    ಸ್ಥಿರಾಂಕ int64_t B_ಒಟ್ಟು = (int64_t)k * n_gemm;
    ಸ್ಥಿರಾಂಕ int64_t B_ಖಂಡ_ಗಾತ್ರ = 256;
    ಸ್ಥಿರಾಂಕ int64_t B_ಅಂಕೆ_work = ((B_ಒಟ್ಟು + B_ಖಂಡ_ಗಾತ್ರ - 1) / B_ಖಂಡ_ಗಾತ್ರ) * B_ಖಂಡ_ಗಾತ್ರ;

    stream->parallel_for(sycl::range<1>(B_ಅಂಕೆ_work), [=](sycl::id<1> id) {
        ಸ್ಥಿರಾಂಕ int64_t t = id[0];
        ಆದರೆ (t >= B_ಒಟ್ಟು) ಹಿಂತಿರುಗಿ;

        ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲು = t % k;
        ಸ್ಥಿರಾಂಕ int64_t ಲಂಬಸಾಲು = t / k;
        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * src_ptr = src0_ತಳ + ಅಡ್ಡಸಾಲು * src0_packed_nb0 + ಲಂಬಸಾಲು * src0_packed_nb1;
        ದಶಮಾಂಶ v;
        ಆದರೆ (src0_is_f32) {
            v = *(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src_ptr;
        } ಇಲ್ಲದಿದ್ದರೆ {
            v = sycl::vec<sycl::half, 1>(*(ಸ್ಥಿರಾಂಕ sycl::half *) src_ptr).convert<ದಶಮಾಂಶ, sycl::rounding_ಬಹುಲಕ::automatic>()[0];
        }
        B_packed[ಅಡ್ಡಸಾಲು + ಲಂಬಸಾಲು * (int64_t)k] = v;
    });

    // GEMM: C = A^T * B where A is (k x m), B is (k x n), C is (m x n)
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ alpha = 1.0f;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ beta  = 0.0f;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ lda = k;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ldb = k;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ldc = m;

    SYCL_CHECK(CHECK_TRY_ERROR(oneapi::mkl::blas::ಸ್ತಂಭ_major::gemm(
        *stream, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans,
        m, n_gemm, k,
        dpct::get_ಮೌಲ್ಯ(&alpha, *stream),
        (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) A_packed, lda,
        (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) B_packed, ldb,
        dpct::get_ಮೌಲ್ಯ(&beta, *stream),
        (ದಶಮಾಂಶ *) dst_mat.ದತ್ತಾಂಶ, ldc)));

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * gemm_ದತ್ತಾಂಶ = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) dst_mat.ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ * dst_ದತ್ತಾಂಶ = (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ;

    ggml_sycl_conv3d_write_output(dst, gemm_ದತ್ತಾಂಶ, dst_ದತ್ತಾಂಶ, patch_ಒಟ್ಟು, oc,
                                  dst->ne[0], dst->ne[1], dst->ne[2], stream);
}
