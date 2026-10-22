#ಅನ್ತರ್ಭಾವ "conv3d.hpp"

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ int64_t ggml_sycl_conv3d_calc_patch_total(ಧ್ರುವ ggml_tensor * dst, int32_t n) {
    ಪ್ರತ್ಯಾಗಮ (int64_t) n * dst->ne[0] * dst->ne[1] * dst->ne[2];
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ int64_t ggml_sycl_conv3d_calc_knl_n_total(ಧ್ರುವ ggml_tensor * src0, int32_t c) {
    ಪ್ರತ್ಯಾಗಮ (int64_t) src0->ne[0] * src0->ne[1] * src0->ne[2] * c;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ ggml_sycl_conv3d_write_output(
        ಧ್ರುವ ggml_tensor * dst,
        ಧ್ರುವ ಭಿನ್ನ * src, ಭಿನ್ನ * dst_data,
        int64_t patch_total, int64_t oc,
        int64_t dst_w, int64_t dst_h, int64_t dst_d,
        dpct::queue_ptr stream) {
    ಧ್ರುವ int64_t dst_nb0 = dst->nb[0];
    ಧ್ರುವ int64_t dst_nb1 = dst->nb[1];
    ಧ್ರುವ int64_t dst_nb2 = dst->nb[2];
    ಧ್ರುವ int64_t dst_nb3 = dst->nb[3];
    ಧ್ರುವ int64_t total = patch_total * oc;
    ಧ್ರುವ int64_t block_size = 256;
    ಧ್ರುವ int64_t ಅಂಕೆ_work_ವಸ್ತುಗಳು = ((total + block_size - 1) / block_size) * block_size;

    stream->parallel_for(sycl::range<1>(ಅಂಕೆ_work_ವಸ್ತುಗಳು), [=](sycl::id<1> id) {
        ಧ್ರುವ int64_t i = id[0];
        ಯದಿ (i >= total) {
            ಪ್ರತ್ಯಾಗಮ;
        }

        ಧ್ರುವ int64_t patch_idx = i / oc;
        ಧ್ರುವ int64_t out_ch = i % oc;
        ಧ್ರುವ int64_t p_in_batch = patch_idx % (dst_w * dst_h * dst_d);
        ಧ್ರುವ int64_t batch_idx = patch_idx / (dst_w * dst_h * dst_d);
        ಧ್ರುವ int64_t dst_z = p_in_batch / (dst_w * dst_h);
        ಧ್ರುವ int64_t dst_y = (p_in_batch % (dst_w * dst_h)) / dst_w;
        ಧ್ರುವ int64_t dst_x = p_in_batch % dst_w;
        ಧ್ರುವ int64_t ocn_idx = batch_idx * oc + out_ch;

        ಧ್ರುವ int64_t dst_offset = dst_x * dst_nb0 + dst_y * dst_nb1 + dst_z * dst_nb2 + ocn_idx * dst_nb3;
        // `src` is a column-major (m x n) GEMM output where m == patch_total, n == oc.
        // GEMM stores element (row, col) at index `row + col*m`, so compute index accordingly.
        ಧ್ರುವ int64_t src_index = patch_idx + out_ch * patch_total;
        ಧ್ರುವ ಭಿನ್ನ value = src[src_index];
        *(ಭಿನ್ನ *)((ವರ್ಣ *)dst_data + dst_offset) = value;
    });
}

ಶೂನ್ಯ ggml_sycl_op_conv_3d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ಧ್ರುವ ggml_tensor * src0 = dst->src[0];
    ಧ್ರುವ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));

    ಧ್ರುವ int32_t * opts = (ಧ್ರುವ int32_t *) dst->op_params;
    ಧ್ರುವ int32_t s0 = opts[0];
    ಧ್ರುವ int32_t s1 = opts[1];
    ಧ್ರುವ int32_t s2 = opts[2];
    ಧ್ರುವ int32_t p0 = opts[3];
    ಧ್ರುವ int32_t p1 = opts[4];
    ಧ್ರುವ int32_t p2 = opts[5];
    ಧ್ರುವ int32_t d0 = opts[6];
    ಧ್ರುವ int32_t d1 = opts[7];
    ಧ್ರುವ int32_t d2 = opts[8];
    ಧ್ರುವ int32_t c  = opts[9];
    ಧ್ರುವ int32_t n  = opts[10];
    ಧ್ರುವ int32_t oc = opts[11];

    ಧ್ರುವ int64_t knl_w = src0->ne[0];
    ಧ್ರುವ int64_t knl_h = src0->ne[1];
    ಧ್ರುವ int64_t knl_d = src0->ne[2];

    ಧ್ರುವ int64_t patch_total = ggml_sycl_conv3d_calc_patch_total(dst, n);
    ಧ್ರುವ int64_t knl_n_total = ggml_sycl_conv3d_calc_knl_n_total(src0, c);

    ಧ್ರುವ size_t kernel_type_size = ggml_element_size(src0);

    ggml_sycl_pool_alloc<ಭಿನ್ನ> gemm_output(ctx.pool());
    gemm_output.alloc((size_t) patch_total * oc);

    ggml_tensor dst_mat = {};
    dst_mat.type = GGML_TYPE_F32;
    dst_mat.ne[0] = patch_total;
    dst_mat.ne[1] = oc;
    dst_mat.ne[2] = 1;
    dst_mat.ne[3] = 1;
    dst_mat.nb[0] = ಪರಿಮಾಣ(ಭಿನ್ನ);
    dst_mat.nb[1] = dst_mat.nb[0] * dst_mat.ne[0];
    dst_mat.nb[2] = dst_mat.nb[1];
    dst_mat.nb[3] = dst_mat.nb[2];
    dst_mat.data = gemm_output.get();
    dst_mat.buffer = dst->buffer;
    dst_mat.extra = dst->extra;

    dpct::queue_ptr stream = ctx.stream();

    // allocate packed arrays: A_packed (k x m), B_packed (k x n)
    ggml_sycl_pool_alloc<ಭಿನ್ನ> A_packed_alloc(ctx.pool());
    ggml_sycl_pool_alloc<ಭಿನ್ನ> B_packed_alloc(ctx.pool());
    A_packed_alloc.alloc((size_t) knl_n_total * patch_total);
    B_packed_alloc.alloc((size_t) knl_n_total * oc);

    ಭಿನ್ನ * A_packed = A_packed_alloc.get();
    ಭಿನ್ನ * B_packed = B_packed_alloc.get();

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ m = (ಪೂರ್ಣಾಙ್ಕ) patch_total;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_gemm = (ಪೂರ್ಣಾಙ್ಕ) oc;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ k = (ಪೂರ್ಣಾಙ್ಕ) knl_n_total;

    // Combined kernel: im2col -> pack A, and pack B simultaneously
    ಧ್ರುವ ವರ್ಣ * src1_base = (ಧ್ರುವ ವರ್ಣ *) src1->data;
    ಧ್ರುವ ವರ್ಣ * src0_base = (ಧ್ರುವ ವರ್ಣ *) src0->data;
    ಧ್ರುವ int64_t src1_nb0 = src1->nb[0];
    ಧ್ರುವ int64_t src1_nb1 = src1->nb[1];
    ಧ್ರುವ int64_t src1_nb2 = src1->nb[2];
    ಧ್ರುವ int64_t src1_nb3 = src1->nb[3];
    ಧ್ರುವ int64_t src1_w = src1->ne[0];
    ಧ್ರುವ int64_t src1_h = src1->ne[1];
    ಧ್ರುವ int64_t src1_d = src1->ne[2];

    ಧ್ರುವ ತಾರ್ಕಿಕ src0_is_f32 = (src0->type == GGML_TYPE_F32);

    // Compute correct strides for src0 as (knl_n_total, oc) matrix
    ಧ್ರುವ int64_t src0_packed_nb0 = kernel_type_size;
    ಧ್ರುವ int64_t src0_packed_nb1 = kernel_type_size * knl_n_total;

    ಧ್ರುವ int64_t KW = knl_w;
    ಧ್ರುವ int64_t KH = knl_h;
    ಧ್ರುವ int64_t KD = knl_d;
    ಧ್ರುವ int64_t PW = dst->ne[0];
    ಧ್ರುವ int64_t PH = dst->ne[1];
    ಧ್ರುವ int64_t PD = dst->ne[2];

    // Pack A (with inline im2col): for each (row, col) in k x m matrix
    ಧ್ರುವ int64_t A_total = (int64_t)k * m;
    ಧ್ರುವ int64_t A_block_size = 256;
    ಧ್ರುವ int64_t A_ಅಂಕೆ_work = ((A_total + A_block_size - 1) / A_block_size) * A_block_size;

    stream->parallel_for(sycl::range<1>(A_ಅಂಕೆ_work), [=](sycl::id<1> id) {
        ಧ್ರುವ int64_t t = id[0];
        ಯದಿ (t >= A_total) ಪ್ರತ್ಯಾಗಮ;

        ಧ್ರುವ int64_t row = t % k;
        ಧ್ರುವ int64_t col = t / k;

        // Inline im2col for this element
        ಧ್ರುವ int64_t k_index = row;
        ಧ್ರುವ int64_t patch_idx = col;

        ಧ್ರುವ int64_t ic = k_index / (KD * KH * KW);
        ಧ್ರುವ int64_t rem = k_index - ic * (KD * KH * KW);
        ಧ್ರುವ int64_t kz = rem / (KH * KW);
        ಧ್ರುವ int64_t rem2 = rem - kz * (KH * KW);
        ಧ್ರುವ int64_t ky = rem2 / KW;
        ಧ್ರುವ int64_t kx = rem2 % KW;

        ಧ್ರುವ int64_t p_in_batch = patch_idx % (PW * PH * PD);
        ಧ್ರುವ int64_t batch_idx = patch_idx / (PW * PH * PD);
        ಧ್ರುವ int64_t dst_z = p_in_batch / (PW * PH);
        ಧ್ರುವ int64_t dst_y = (p_in_batch % (PW * PH)) / PW;
        ಧ್ರುವ int64_t dst_x = p_in_batch % PW;

        ಧ್ರುವ int64_t sx = dst_x * s0 + kx * d0 - p0;
        ಧ್ರುವ int64_t sy = dst_y * s1 + ky * d1 - p1;
        ಧ್ರುವ int64_t sz = dst_z * s2 + kz * d2 - p2;

        ಭಿನ್ನ val = 0.0f;
        ಯದಿ (sx >= 0 && sx < src1_w && sy >= 0 && sy < src1_h && sz >= 0 && sz < src1_d) {
            ಧ್ರುವ int64_t channel_idx = batch_idx * c + ic;
            ಧ್ರುವ ವರ್ಣ * ptr = src1_base + sx * src1_nb0 + sy * src1_nb1 + sz * src1_nb2 + channel_idx * src1_nb3;
            val = *(ಧ್ರುವ ಭಿನ್ನ *) ptr;
        }
        A_packed[row + col * (int64_t)k] = val;
    });

    // Pack B: for each (row, col) in k x n_gemm matrix
    ಧ್ರುವ int64_t B_total = (int64_t)k * n_gemm;
    ಧ್ರುವ int64_t B_block_size = 256;
    ಧ್ರುವ int64_t B_ಅಂಕೆ_work = ((B_total + B_block_size - 1) / B_block_size) * B_block_size;

    stream->parallel_for(sycl::range<1>(B_ಅಂಕೆ_work), [=](sycl::id<1> id) {
        ಧ್ರುವ int64_t t = id[0];
        ಯದಿ (t >= B_total) ಪ್ರತ್ಯಾಗಮ;

        ಧ್ರುವ int64_t row = t % k;
        ಧ್ರುವ int64_t col = t / k;
        ಧ್ರುವ ವರ್ಣ * src_ptr = src0_base + row * src0_packed_nb0 + col * src0_packed_nb1;
        ಭಿನ್ನ v;
        ಯದಿ (src0_is_f32) {
            v = *(ಧ್ರುವ ಭಿನ್ನ *) src_ptr;
        } ಅನ್ಯಥಾ {
            v = sycl::vec<sycl::half, 1>(*(ಧ್ರುವ sycl::half *) src_ptr).convert<ಭಿನ್ನ, sycl::rounding_mode::automatic>()[0];
        }
        B_packed[row + col * (int64_t)k] = v;
    });

    // GEMM: C = A^T * B where A is (k x m), B is (k x n), C is (m x n)
    ಧ್ರುವ ಭಿನ್ನ alpha = 1.0f;
    ಧ್ರುವ ಭಿನ್ನ beta  = 0.0f;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ lda = k;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ldb = k;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ldc = m;

    SYCL_CHECK(CHECK_TRY_ERROR(oneapi::mkl::blas::column_major::gemm(
        *stream, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans,
        m, n_gemm, k,
        dpct::get_ಮೌಲ್ಯ(&alpha, *stream),
        (ಧ್ರುವ ಭಿನ್ನ *) A_packed, lda,
        (ಧ್ರುವ ಭಿನ್ನ *) B_packed, ldb,
        dpct::get_ಮೌಲ್ಯ(&beta, *stream),
        (ಭಿನ್ನ *) dst_mat.data, ldc)));

    ಧ್ರುವ ಭಿನ್ನ * gemm_data = (ಧ್ರುವ ಭಿನ್ನ *) dst_mat.data;
    ಭಿನ್ನ * dst_data = (ಭಿನ್ನ *) dst->data;

    ggml_sycl_conv3d_write_output(dst, gemm_data, dst_data, patch_total, oc,
                                  dst->ne[0], dst->ne[1], dst->ne[2], stream);
}
