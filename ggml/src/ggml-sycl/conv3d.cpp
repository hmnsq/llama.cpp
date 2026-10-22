#ਅਨ੍ਤਰ੍ਭਾਵ "conv3d.hpp"

ਸਥਿਰ ਸਤਰਅੰਦਰ int64_t ggml_sycl_conv3d_calc_patch_ਕੁੱਲ(ਅਚਲ ggml_tensor * dst, int32_t n) {
    ਵਾਪਸ (int64_t) n * dst->ne[0] * dst->ne[1] * dst->ne[2];
}

ਸਥਿਰ ਸਤਰਅੰਦਰ int64_t ggml_sycl_conv3d_calc_knl_n_ਕੁੱਲ(ਅਚਲ ggml_tensor * src0, int32_t c) {
    ਵਾਪਸ (int64_t) src0->ne[0] * src0->ne[1] * src0->ne[2] * c;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_sycl_conv3d_write_output(
        ਅਚਲ ggml_tensor * dst,
        ਅਚਲ ਦਸ਼ਮਲਵ * src, ਦਸ਼ਮਲਵ * dst_ਅੰਕੜੇ,
        int64_t patch_ਕੁੱਲ, int64_t oc,
        int64_t dst_w, int64_t dst_h, int64_t dst_d,
        dpct::ਕਤਾਰ_ptr stream) {
    ਅਚਲ int64_t dst_nb0 = dst->nb[0];
    ਅਚਲ int64_t dst_nb1 = dst->nb[1];
    ਅਚਲ int64_t dst_nb2 = dst->nb[2];
    ਅਚਲ int64_t dst_nb3 = dst->nb[3];
    ਅਚਲ int64_t ਕੁੱਲ = patch_ਕੁੱਲ * oc;
    ਅਚਲ int64_t ਖੰਡ_ਮਾਪ_3 = 256;
    ਅਚਲ int64_t ਅੰਕ_work_ਚੀਜ਼ਾਂ = ((ਕੁੱਲ + ਖੰਡ_ਮਾਪ_3 - 1) / ਖੰਡ_ਮਾਪ_3) * ਖੰਡ_ਮਾਪ_3;

    stream->parallel_for(sycl::range<1>(ਅੰਕ_work_ਚੀਜ਼ਾਂ), [=](sycl::id<1> id) {
        ਅਚਲ int64_t i = id[0];
        ਜੇ (i >= ਕੁੱਲ) {
            ਵਾਪਸ;
        }

        ਅਚਲ int64_t patch_idx = i / oc;
        ਅਚਲ int64_t ਬਾਹਰਲਾ_ch = i % oc;
        ਅਚਲ int64_t p_ਵਿੱਚ_batch = patch_idx % (dst_w * dst_h * dst_d);
        ਅਚਲ int64_t batch_idx = patch_idx / (dst_w * dst_h * dst_d);
        ਅਚਲ int64_t dst_z = p_ਵਿੱਚ_batch / (dst_w * dst_h);
        ਅਚਲ int64_t dst_y = (p_ਵਿੱਚ_batch % (dst_w * dst_h)) / dst_w;
        ਅਚਲ int64_t dst_x = p_ਵਿੱਚ_batch % dst_w;
        ਅਚਲ int64_t ocn_idx = batch_idx * oc + ਬਾਹਰਲਾ_ch;

        ਅਚਲ int64_t dst_ਵਿੱਥ = dst_x * dst_nb0 + dst_y * dst_nb1 + dst_z * dst_nb2 + ocn_idx * dst_nb3;
        // `src` is a column-major (m x n) GEMM output where m == patch_total, n == oc.
        // GEMM stores element (row, col) at index `row + col*m`, so compute index accordingly.
        ਅਚਲ int64_t src_ਸੂਚਕਾਂਕ = patch_idx + ਬਾਹਰਲਾ_ch * patch_ਕੁੱਲ;
        ਅਚਲ ਦਸ਼ਮਲਵ value = src[src_ਸੂਚਕਾਂਕ];
        *(ਦਸ਼ਮਲਵ *)((ਅੱਖਰ *)dst_ਅੰਕੜੇ + dst_ਵਿੱਥ) = value;
    });
}

ਖਾਲੀ ggml_sycl_op_conv_3d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));

    ਅਚਲ int32_t * opts = (ਅਚਲ int32_t *) dst->op_params;
    ਅਚਲ int32_t s0 = opts[0];
    ਅਚਲ int32_t s1 = opts[1];
    ਅਚਲ int32_t s2 = opts[2];
    ਅਚਲ int32_t p0 = opts[3];
    ਅਚਲ int32_t p1 = opts[4];
    ਅਚਲ int32_t p2 = opts[5];
    ਅਚਲ int32_t d0 = opts[6];
    ਅਚਲ int32_t d1 = opts[7];
    ਅਚਲ int32_t d2 = opts[8];
    ਅਚਲ int32_t c  = opts[9];
    ਅਚਲ int32_t n  = opts[10];
    ਅਚਲ int32_t oc = opts[11];

    ਅਚਲ int64_t knl_w = src0->ne[0];
    ਅਚਲ int64_t knl_h = src0->ne[1];
    ਅਚਲ int64_t knl_d = src0->ne[2];

    ਅਚਲ int64_t patch_ਕੁੱਲ = ggml_sycl_conv3d_calc_patch_ਕੁੱਲ(dst, n);
    ਅਚਲ int64_t knl_n_ਕੁੱਲ = ggml_sycl_conv3d_calc_knl_n_ਕੁੱਲ(src0, c);

    ਅਚਲ size_t kernel_type_ਮਾਪ = ggml_element_ਮਾਪ(src0);

    ggml_sycl_pool_alloc<ਦਸ਼ਮਲਵ> gemm_output(ctx.pool());
    gemm_output.alloc((size_t) patch_ਕੁੱਲ * oc);

    ggml_tensor dst_mat = {};
    dst_mat.type = GGML_TYPE_F32;
    dst_mat.ne[0] = patch_ਕੁੱਲ;
    dst_mat.ne[1] = oc;
    dst_mat.ne[2] = 1;
    dst_mat.ne[3] = 1;
    dst_mat.nb[0] = ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    dst_mat.nb[1] = dst_mat.nb[0] * dst_mat.ne[0];
    dst_mat.nb[2] = dst_mat.nb[1];
    dst_mat.nb[3] = dst_mat.nb[2];
    dst_mat.ਅੰਕੜੇ = gemm_output.get();
    dst_mat.ਬਫਰ = dst->ਬਫਰ;
    dst_mat.extra = dst->extra;

    dpct::ਕਤਾਰ_ptr stream = ctx.stream();

    // allocate packed arrays: A_packed (k x m), B_packed (k x n)
    ggml_sycl_pool_alloc<ਦਸ਼ਮਲਵ> A_packed_alloc(ctx.pool());
    ggml_sycl_pool_alloc<ਦਸ਼ਮਲਵ> B_packed_alloc(ctx.pool());
    A_packed_alloc.alloc((size_t) knl_n_ਕੁੱਲ * patch_ਕੁੱਲ);
    B_packed_alloc.alloc((size_t) knl_n_ਕੁੱਲ * oc);

    ਦਸ਼ਮਲਵ * A_packed = A_packed_alloc.get();
    ਦਸ਼ਮਲਵ * B_packed = B_packed_alloc.get();

    ਅਚਲ ਪੂਰਨ m = (ਪੂਰਨ) patch_ਕੁੱਲ;
    ਅਚਲ ਪੂਰਨ n_gemm = (ਪੂਰਨ) oc;
    ਅਚਲ ਪੂਰਨ k = (ਪੂਰਨ) knl_n_ਕੁੱਲ;

    // Combined kernel: im2col -> pack A, and pack B simultaneously
    ਅਚਲ ਅੱਖਰ * src1_ਅਧਾਰ = (ਅਚਲ ਅੱਖਰ *) src1->ਅੰਕੜੇ;
    ਅਚਲ ਅੱਖਰ * src0_ਅਧਾਰ = (ਅਚਲ ਅੱਖਰ *) src0->ਅੰਕੜੇ;
    ਅਚਲ int64_t src1_nb0 = src1->nb[0];
    ਅਚਲ int64_t src1_nb1 = src1->nb[1];
    ਅਚਲ int64_t src1_nb2 = src1->nb[2];
    ਅਚਲ int64_t src1_nb3 = src1->nb[3];
    ਅਚਲ int64_t src1_w = src1->ne[0];
    ਅਚਲ int64_t src1_h = src1->ne[1];
    ਅਚਲ int64_t src1_d = src1->ne[2];

    ਅਚਲ ਤਾਰਕਿਕ src0_is_f32 = (src0->type == GGML_TYPE_F32);

    // Compute correct strides for src0 as (knl_n_total, oc) matrix
    ਅਚਲ int64_t src0_packed_nb0 = kernel_type_ਮਾਪ;
    ਅਚਲ int64_t src0_packed_nb1 = kernel_type_ਮਾਪ * knl_n_ਕੁੱਲ;

    ਅਚਲ int64_t KW = knl_w;
    ਅਚਲ int64_t KH = knl_h;
    ਅਚਲ int64_t KD = knl_d;
    ਅਚਲ int64_t PW = dst->ne[0];
    ਅਚਲ int64_t PH = dst->ne[1];
    ਅਚਲ int64_t PD = dst->ne[2];

    // Pack A (with inline im2col): for each (row, col) in k x m matrix
    ਅਚਲ int64_t A_ਕੁੱਲ = (int64_t)k * m;
    ਅਚਲ int64_t A_ਖੰਡ_ਮਾਪ = 256;
    ਅਚਲ int64_t A_ਅੰਕ_work = ((A_ਕੁੱਲ + A_ਖੰਡ_ਮਾਪ - 1) / A_ਖੰਡ_ਮਾਪ) * A_ਖੰਡ_ਮਾਪ;

    stream->parallel_for(sycl::range<1>(A_ਅੰਕ_work), [=](sycl::id<1> id) {
        ਅਚਲ int64_t t = id[0];
        ਜੇ (t >= A_ਕੁੱਲ) ਵਾਪਸ;

        ਅਚਲ int64_t ਪੰਕਤੀ = t % k;
        ਅਚਲ int64_t ਖਾਨਾ = t / k;

        // Inline im2col for this element
        ਅਚਲ int64_t k_ਸੂਚਕਾਂਕ = ਪੰਕਤੀ;
        ਅਚਲ int64_t patch_idx = ਖਾਨਾ;

        ਅਚਲ int64_t ic = k_ਸੂਚਕਾਂਕ / (KD * KH * KW);
        ਅਚਲ int64_t rem = k_ਸੂਚਕਾਂਕ - ic * (KD * KH * KW);
        ਅਚਲ int64_t kz = rem / (KH * KW);
        ਅਚਲ int64_t rem2 = rem - kz * (KH * KW);
        ਅਚਲ int64_t ky = rem2 / KW;
        ਅਚਲ int64_t kx = rem2 % KW;

        ਅਚਲ int64_t p_ਵਿੱਚ_batch = patch_idx % (PW * PH * PD);
        ਅਚਲ int64_t batch_idx = patch_idx / (PW * PH * PD);
        ਅਚਲ int64_t dst_z = p_ਵਿੱਚ_batch / (PW * PH);
        ਅਚਲ int64_t dst_y = (p_ਵਿੱਚ_batch % (PW * PH)) / PW;
        ਅਚਲ int64_t dst_x = p_ਵਿੱਚ_batch % PW;

        ਅਚਲ int64_t sx = dst_x * s0 + kx * d0 - p0;
        ਅਚਲ int64_t sy = dst_y * s1 + ky * d1 - p1;
        ਅਚਲ int64_t sz = dst_z * s2 + kz * d2 - p2;

        ਦਸ਼ਮਲਵ val = 0.0f;
        ਜੇ (sx >= 0 && sx < src1_w && sy >= 0 && sy < src1_h && sz >= 0 && sz < src1_d) {
            ਅਚਲ int64_t channel_idx = batch_idx * c + ic;
            ਅਚਲ ਅੱਖਰ * ptr = src1_ਅਧਾਰ + sx * src1_nb0 + sy * src1_nb1 + sz * src1_nb2 + channel_idx * src1_nb3;
            val = *(ਅਚਲ ਦਸ਼ਮਲਵ *) ptr;
        }
        A_packed[ਪੰਕਤੀ + ਖਾਨਾ * (int64_t)k] = val;
    });

    // Pack B: for each (row, col) in k x n_gemm matrix
    ਅਚਲ int64_t B_ਕੁੱਲ = (int64_t)k * n_gemm;
    ਅਚਲ int64_t B_ਖੰਡ_ਮਾਪ = 256;
    ਅਚਲ int64_t B_ਅੰਕ_work = ((B_ਕੁੱਲ + B_ਖੰਡ_ਮਾਪ - 1) / B_ਖੰਡ_ਮਾਪ) * B_ਖੰਡ_ਮਾਪ;

    stream->parallel_for(sycl::range<1>(B_ਅੰਕ_work), [=](sycl::id<1> id) {
        ਅਚਲ int64_t t = id[0];
        ਜੇ (t >= B_ਕੁੱਲ) ਵਾਪਸ;

        ਅਚਲ int64_t ਪੰਕਤੀ = t % k;
        ਅਚਲ int64_t ਖਾਨਾ = t / k;
        ਅਚਲ ਅੱਖਰ * src_ptr = src0_ਅਧਾਰ + ਪੰਕਤੀ * src0_packed_nb0 + ਖਾਨਾ * src0_packed_nb1;
        ਦਸ਼ਮਲਵ v;
        ਜੇ (src0_is_f32) {
            v = *(ਅਚਲ ਦਸ਼ਮਲਵ *) src_ptr;
        } ਨਹੀਂਤਾਂ {
            v = sycl::vec<sycl::half, 1>(*(ਅਚਲ sycl::half *) src_ptr).convert<ਦਸ਼ਮਲਵ, sycl::rounding_ਬਹੁਲਕ::automatic>()[0];
        }
        B_packed[ਪੰਕਤੀ + ਖਾਨਾ * (int64_t)k] = v;
    });

    // GEMM: C = A^T * B where A is (k x m), B is (k x n), C is (m x n)
    ਅਚਲ ਦਸ਼ਮਲਵ alpha = 1.0f;
    ਅਚਲ ਦਸ਼ਮਲਵ beta  = 0.0f;
    ਅਚਲ ਪੂਰਨ lda = k;
    ਅਚਲ ਪੂਰਨ ldb = k;
    ਅਚਲ ਪੂਰਨ ldc = m;

    SYCL_CHECK(CHECK_TRY_ERROR(oneapi::mkl::blas::ਕਾਲਮ_major::gemm(
        *stream, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans,
        m, n_gemm, k,
        dpct::get_ਮੁੱਲ(&alpha, *stream),
        (ਅਚਲ ਦਸ਼ਮਲਵ *) A_packed, lda,
        (ਅਚਲ ਦਸ਼ਮਲਵ *) B_packed, ldb,
        dpct::get_ਮੁੱਲ(&beta, *stream),
        (ਦਸ਼ਮਲਵ *) dst_mat.ਅੰਕੜੇ, ldc)));

    ਅਚਲ ਦਸ਼ਮਲਵ * gemm_ਅੰਕੜੇ = (ਅਚਲ ਦਸ਼ਮਲਵ *) dst_mat.ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ * dst_ਅੰਕੜੇ = (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ;

    ggml_sycl_conv3d_write_output(dst, gemm_ਅੰਕੜੇ, dst_ਅੰਕੜੇ, patch_ਕੁੱਲ, oc,
                                  dst->ne[0], dst->ne[1], dst->ne[2], stream);
}
