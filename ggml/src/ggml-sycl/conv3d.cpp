#অন্তর্ভাব "conv3d.hpp"

স্থির অন্তঃরেখ int64_t ggml_sycl_conv3d_calc_patch_যোগ(ধ্রুবক ggml_tensor * dst, int32_t n) {
    ফেরত (int64_t) n * dst->ne[0] * dst->ne[1] * dst->ne[2];
}

স্থির অন্তঃরেখ int64_t ggml_sycl_conv3d_calc_knl_n_যোগ(ধ্রুবক ggml_tensor * src0, int32_t c) {
    ফেরত (int64_t) src0->ne[0] * src0->ne[1] * src0->ne[2] * c;
}

স্থির অন্তঃরেখ শূন্য ggml_sycl_conv3d_write_output(
        ধ্রুবক ggml_tensor * dst,
        ধ্রুবক দশমিক * src, দশমিক * dst_তথ্য,
        int64_t patch_যোগ, int64_t oc,
        int64_t dst_w, int64_t dst_h, int64_t dst_d,
        dpct::queue_ptr stream) {
    ধ্রুবক int64_t dst_nb0 = dst->nb[0];
    ধ্রুবক int64_t dst_nb1 = dst->nb[1];
    ধ্রুবক int64_t dst_nb2 = dst->nb[2];
    ধ্রুবক int64_t dst_nb3 = dst->nb[3];
    ধ্রুবক int64_t যোগ = patch_যোগ * oc;
    ধ্রুবক int64_t block_আকার_2 = 256;
    ধ্রুবক int64_t অঙ্ক_work_বস্তুসমূহ = ((যোগ + block_আকার_2 - 1) / block_আকার_2) * block_আকার_2;

    stream->parallel_for(sycl::range<1>(অঙ্ক_work_বস্তুসমূহ), [=](sycl::id<1> id) {
        ধ্রুবক int64_t i = id[0];
        যদি (i >= যোগ) {
            ফেরত;
        }

        ধ্রুবক int64_t patch_idx = i / oc;
        ধ্রুবক int64_t out_ch = i % oc;
        ধ্রুবক int64_t p_in_batch = patch_idx % (dst_w * dst_h * dst_d);
        ধ্রুবক int64_t batch_idx = patch_idx / (dst_w * dst_h * dst_d);
        ধ্রুবক int64_t dst_z = p_in_batch / (dst_w * dst_h);
        ধ্রুবক int64_t dst_y = (p_in_batch % (dst_w * dst_h)) / dst_w;
        ধ্রুবক int64_t dst_x = p_in_batch % dst_w;
        ধ্রুবক int64_t ocn_idx = batch_idx * oc + out_ch;

        ধ্রুবক int64_t dst_offset = dst_x * dst_nb0 + dst_y * dst_nb1 + dst_z * dst_nb2 + ocn_idx * dst_nb3;
        // `src` is a column-major (m x n) GEMM output where m == patch_total, n == oc.
        // GEMM stores element (row, col) at index `row + col*m`, so compute index accordingly.
        ধ্রুবক int64_t src_সূচক = patch_idx + out_ch * patch_যোগ;
        ধ্রুবক দশমিক value = src[src_সূচক];
        *(দশমিক *)((অক্ষর *)dst_তথ্য + dst_offset) = value;
    });
}

শূন্য ggml_sycl_op_conv_3d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ধ্রুবক ggml_tensor * src0 = dst->src[0];
    ধ্রুবক ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));

    ধ্রুবক int32_t * opts = (ধ্রুবক int32_t *) dst->op_params;
    ধ্রুবক int32_t s0 = opts[0];
    ধ্রুবক int32_t s1 = opts[1];
    ধ্রুবক int32_t s2 = opts[2];
    ধ্রুবক int32_t p0 = opts[3];
    ধ্রুবক int32_t p1 = opts[4];
    ধ্রুবক int32_t p2 = opts[5];
    ধ্রুবক int32_t d0 = opts[6];
    ধ্রুবক int32_t d1 = opts[7];
    ধ্রুবক int32_t d2 = opts[8];
    ধ্রুবক int32_t c  = opts[9];
    ধ্রুবক int32_t n  = opts[10];
    ধ্রুবক int32_t oc = opts[11];

    ধ্রুবক int64_t knl_w = src0->ne[0];
    ধ্রুবক int64_t knl_h = src0->ne[1];
    ধ্রুবক int64_t knl_d = src0->ne[2];

    ধ্রুবক int64_t patch_যোগ = ggml_sycl_conv3d_calc_patch_যোগ(dst, n);
    ধ্রুবক int64_t knl_n_যোগ = ggml_sycl_conv3d_calc_knl_n_যোগ(src0, c);

    ধ্রুবক size_t kernel_type_আকার = ggml_element_আকার(src0);

    ggml_sycl_pool_alloc<দশমিক> gemm_output(ctx.pool());
    gemm_output.alloc((size_t) patch_যোগ * oc);

    ggml_tensor dst_mat = {};
    dst_mat.type = GGML_TYPE_F32;
    dst_mat.ne[0] = patch_যোগ;
    dst_mat.ne[1] = oc;
    dst_mat.ne[2] = 1;
    dst_mat.ne[3] = 1;
    dst_mat.nb[0] = আকার(দশমিক);
    dst_mat.nb[1] = dst_mat.nb[0] * dst_mat.ne[0];
    dst_mat.nb[2] = dst_mat.nb[1];
    dst_mat.nb[3] = dst_mat.nb[2];
    dst_mat.তথ্য = gemm_output.get();
    dst_mat.বাফার = dst->বাফার;
    dst_mat.extra = dst->extra;

    dpct::queue_ptr stream = ctx.stream();

    // allocate packed arrays: A_packed (k x m), B_packed (k x n)
    ggml_sycl_pool_alloc<দশমিক> A_packed_alloc(ctx.pool());
    ggml_sycl_pool_alloc<দশমিক> B_packed_alloc(ctx.pool());
    A_packed_alloc.alloc((size_t) knl_n_যোগ * patch_যোগ);
    B_packed_alloc.alloc((size_t) knl_n_যোগ * oc);

    দশমিক * A_packed = A_packed_alloc.get();
    দশমিক * B_packed = B_packed_alloc.get();

    ধ্রুবক পূর্ণসংখ্যা m = (পূর্ণসংখ্যা) patch_যোগ;
    ধ্রুবক পূর্ণসংখ্যা n_gemm = (পূর্ণসংখ্যা) oc;
    ধ্রুবক পূর্ণসংখ্যা k = (পূর্ণসংখ্যা) knl_n_যোগ;

    // Combined kernel: im2col -> pack A, and pack B simultaneously
    ধ্রুবক অক্ষর * src1_base = (ধ্রুবক অক্ষর *) src1->তথ্য;
    ধ্রুবক অক্ষর * src0_base = (ধ্রুবক অক্ষর *) src0->তথ্য;
    ধ্রুবক int64_t src1_nb0 = src1->nb[0];
    ধ্রুবক int64_t src1_nb1 = src1->nb[1];
    ধ্রুবক int64_t src1_nb2 = src1->nb[2];
    ধ্রুবক int64_t src1_nb3 = src1->nb[3];
    ধ্রুবক int64_t src1_w = src1->ne[0];
    ধ্রুবক int64_t src1_h = src1->ne[1];
    ধ্রুবক int64_t src1_d = src1->ne[2];

    ধ্রুবক যৌক্তিক src0_is_f32 = (src0->type == GGML_TYPE_F32);

    // Compute correct strides for src0 as (knl_n_total, oc) matrix
    ধ্রুবক int64_t src0_packed_nb0 = kernel_type_আকার;
    ধ্রুবক int64_t src0_packed_nb1 = kernel_type_আকার * knl_n_যোগ;

    ধ্রুবক int64_t KW = knl_w;
    ধ্রুবক int64_t KH = knl_h;
    ধ্রুবক int64_t KD = knl_d;
    ধ্রুবক int64_t PW = dst->ne[0];
    ধ্রুবক int64_t PH = dst->ne[1];
    ধ্রুবক int64_t PD = dst->ne[2];

    // Pack A (with inline im2col): for each (row, col) in k x m matrix
    ধ্রুবক int64_t A_যোগ = (int64_t)k * m;
    ধ্রুবক int64_t A_block_আকার = 256;
    ধ্রুবক int64_t A_অঙ্ক_work = ((A_যোগ + A_block_আকার - 1) / A_block_আকার) * A_block_আকার;

    stream->parallel_for(sycl::range<1>(A_অঙ্ক_work), [=](sycl::id<1> id) {
        ধ্রুবক int64_t t = id[0];
        যদি (t >= A_যোগ) ফেরত;

        ধ্রুবক int64_t row = t % k;
        ধ্রুবক int64_t col = t / k;

        // Inline im2col for this element
        ধ্রুবক int64_t k_সূচক = row;
        ধ্রুবক int64_t patch_idx = col;

        ধ্রুবক int64_t ic = k_সূচক / (KD * KH * KW);
        ধ্রুবক int64_t rem = k_সূচক - ic * (KD * KH * KW);
        ধ্রুবক int64_t kz = rem / (KH * KW);
        ধ্রুবক int64_t rem2 = rem - kz * (KH * KW);
        ধ্রুবক int64_t ky = rem2 / KW;
        ধ্রুবক int64_t kx = rem2 % KW;

        ধ্রুবক int64_t p_in_batch = patch_idx % (PW * PH * PD);
        ধ্রুবক int64_t batch_idx = patch_idx / (PW * PH * PD);
        ধ্রুবক int64_t dst_z = p_in_batch / (PW * PH);
        ধ্রুবক int64_t dst_y = (p_in_batch % (PW * PH)) / PW;
        ধ্রুবক int64_t dst_x = p_in_batch % PW;

        ধ্রুবক int64_t sx = dst_x * s0 + kx * d0 - p0;
        ধ্রুবক int64_t sy = dst_y * s1 + ky * d1 - p1;
        ধ্রুবক int64_t sz = dst_z * s2 + kz * d2 - p2;

        দশমিক val = 0.0f;
        যদি (sx >= 0 && sx < src1_w && sy >= 0 && sy < src1_h && sz >= 0 && sz < src1_d) {
            ধ্রুবক int64_t channel_idx = batch_idx * c + ic;
            ধ্রুবক অক্ষর * ptr = src1_base + sx * src1_nb0 + sy * src1_nb1 + sz * src1_nb2 + channel_idx * src1_nb3;
            val = *(ধ্রুবক দশমিক *) ptr;
        }
        A_packed[row + col * (int64_t)k] = val;
    });

    // Pack B: for each (row, col) in k x n_gemm matrix
    ধ্রুবক int64_t B_যোগ = (int64_t)k * n_gemm;
    ধ্রুবক int64_t B_block_আকার = 256;
    ধ্রুবক int64_t B_অঙ্ক_work = ((B_যোগ + B_block_আকার - 1) / B_block_আকার) * B_block_আকার;

    stream->parallel_for(sycl::range<1>(B_অঙ্ক_work), [=](sycl::id<1> id) {
        ধ্রুবক int64_t t = id[0];
        যদি (t >= B_যোগ) ফেরত;

        ধ্রুবক int64_t row = t % k;
        ধ্রুবক int64_t col = t / k;
        ধ্রুবক অক্ষর * src_ptr = src0_base + row * src0_packed_nb0 + col * src0_packed_nb1;
        দশমিক v;
        যদি (src0_is_f32) {
            v = *(ধ্রুবক দশমিক *) src_ptr;
        } অন্যথা {
            v = sycl::vec<sycl::half, 1>(*(ধ্রুবক sycl::half *) src_ptr).convert<দশমিক, sycl::rounding_mode::automatic>()[0];
        }
        B_packed[row + col * (int64_t)k] = v;
    });

    // GEMM: C = A^T * B where A is (k x m), B is (k x n), C is (m x n)
    ধ্রুবক দশমিক alpha = 1.0f;
    ধ্রুবক দশমিক beta  = 0.0f;
    ধ্রুবক পূর্ণসংখ্যা lda = k;
    ধ্রুবক পূর্ণসংখ্যা ldb = k;
    ধ্রুবক পূর্ণসংখ্যা ldc = m;

    SYCL_CHECK(CHECK_TRY_ERROR(oneapi::mkl::blas::column_major::gemm(
        *stream, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans,
        m, n_gemm, k,
        dpct::get_মান(&alpha, *stream),
        (ধ্রুবক দশমিক *) A_packed, lda,
        (ধ্রুবক দশমিক *) B_packed, ldb,
        dpct::get_মান(&beta, *stream),
        (দশমিক *) dst_mat.তথ্য, ldc)));

    ধ্রুবক দশমিক * gemm_তথ্য = (ধ্রুবক দশমিক *) dst_mat.তথ্য;
    দশমিক * dst_তথ্য = (দশমিক *) dst->তথ্য;

    ggml_sycl_conv3d_write_output(dst, gemm_তথ্য, dst_তথ্য, patch_যোগ, oc,
                                  dst->ne[0], dst->ne[1], dst->ne[2], stream);
}
