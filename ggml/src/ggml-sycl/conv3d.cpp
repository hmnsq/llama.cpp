#समावेश "conv3d.hpp"

अटल अंतरभूत int64_t ggml_sycl_conv3d_calc_patch_योग(स्थिर ggml_tensor * dst, int32_t n) {
    वापस (int64_t) n * dst->ne[0] * dst->ne[1] * dst->ne[2];
}

अटल अंतरभूत int64_t ggml_sycl_conv3d_calc_knl_n_योग(स्थिर ggml_tensor * src0, int32_t c) {
    वापस (int64_t) src0->ne[0] * src0->ne[1] * src0->ne[2] * c;
}

अटल अंतरभूत व्योम ggml_sycl_conv3d_write_output(
        स्थिर ggml_tensor * dst,
        स्थिर भग्न * src, भग्न * dst_तथ्य,
        int64_t patch_योग, int64_t oc,
        int64_t dst_w, int64_t dst_h, int64_t dst_d,
        dpct::queue_ptr stream) {
    स्थिर int64_t dst_nb0 = dst->nb[0];
    स्थिर int64_t dst_nb1 = dst->nb[1];
    स्थिर int64_t dst_nb2 = dst->nb[2];
    स्थिर int64_t dst_nb3 = dst->nb[3];
    स्थिर int64_t योग = patch_योग * oc;
    स्थिर int64_t block_आकार_2 = 256;
    स्थिर int64_t अंक_work_वस्तुएँ = ((योग + block_आकार_2 - 1) / block_आकार_2) * block_आकार_2;

    stream->parallel_for(sycl::range<1>(अंक_work_वस्तुएँ), [=](sycl::id<1> id) {
        स्थिर int64_t i = id[0];
        अगर (i >= योग) {
            वापस;
        }

        स्थिर int64_t patch_idx = i / oc;
        स्थिर int64_t out_ch = i % oc;
        स्थिर int64_t p_in_batch = patch_idx % (dst_w * dst_h * dst_d);
        स्थिर int64_t batch_idx = patch_idx / (dst_w * dst_h * dst_d);
        स्थिर int64_t dst_z = p_in_batch / (dst_w * dst_h);
        स्थिर int64_t dst_y = (p_in_batch % (dst_w * dst_h)) / dst_w;
        स्थिर int64_t dst_x = p_in_batch % dst_w;
        स्थिर int64_t ocn_idx = batch_idx * oc + out_ch;

        स्थिर int64_t dst_offset = dst_x * dst_nb0 + dst_y * dst_nb1 + dst_z * dst_nb2 + ocn_idx * dst_nb3;
        // `src` is a column-major (m x n) GEMM output where m == patch_total, n == oc.
        // GEMM stores element (row, col) at index `row + col*m`, so compute index accordingly.
        स्थिर int64_t src_सूचकांक = patch_idx + out_ch * patch_योग;
        स्थिर भग्न value = src[src_सूचकांक];
        *(भग्न *)((अक्षर *)dst_तथ्य + dst_offset) = value;
    });
}

व्योम ggml_sycl_op_conv_3d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));

    स्थिर int32_t * opts = (स्थिर int32_t *) dst->op_params;
    स्थिर int32_t s0 = opts[0];
    स्थिर int32_t s1 = opts[1];
    स्थिर int32_t s2 = opts[2];
    स्थिर int32_t p0 = opts[3];
    स्थिर int32_t p1 = opts[4];
    स्थिर int32_t p2 = opts[5];
    स्थिर int32_t d0 = opts[6];
    स्थिर int32_t d1 = opts[7];
    स्थिर int32_t d2 = opts[8];
    स्थिर int32_t c  = opts[9];
    स्थिर int32_t n  = opts[10];
    स्थिर int32_t oc = opts[11];

    स्थिर int64_t knl_w = src0->ne[0];
    स्थिर int64_t knl_h = src0->ne[1];
    स्थिर int64_t knl_d = src0->ne[2];

    स्थिर int64_t patch_योग = ggml_sycl_conv3d_calc_patch_योग(dst, n);
    स्थिर int64_t knl_n_योग = ggml_sycl_conv3d_calc_knl_n_योग(src0, c);

    स्थिर माप_प्रकार kernel_type_आकार = ggml_element_आकार(src0);

    ggml_sycl_pool_alloc<भग्न> gemm_output(ctx.pool());
    gemm_output.alloc((माप_प्रकार) patch_योग * oc);

    ggml_tensor dst_mat = {};
    dst_mat.type = GGML_TYPE_F32;
    dst_mat.ne[0] = patch_योग;
    dst_mat.ne[1] = oc;
    dst_mat.ne[2] = 1;
    dst_mat.ne[3] = 1;
    dst_mat.nb[0] = माप(भग्न);
    dst_mat.nb[1] = dst_mat.nb[0] * dst_mat.ne[0];
    dst_mat.nb[2] = dst_mat.nb[1];
    dst_mat.nb[3] = dst_mat.nb[2];
    dst_mat.तथ्य = gemm_output.लो();
    dst_mat.बफर = dst->बफर;
    dst_mat.extra = dst->extra;

    dpct::queue_ptr stream = ctx.stream();

    // allocate packed arrays: A_packed (k x m), B_packed (k x n)
    ggml_sycl_pool_alloc<भग्न> A_packed_alloc(ctx.pool());
    ggml_sycl_pool_alloc<भग्न> B_packed_alloc(ctx.pool());
    A_packed_alloc.alloc((माप_प्रकार) knl_n_योग * patch_योग);
    B_packed_alloc.alloc((माप_प्रकार) knl_n_योग * oc);

    भग्न * A_packed = A_packed_alloc.लो();
    भग्न * B_packed = B_packed_alloc.लो();

    स्थिर पूर्णांक m = (पूर्णांक) patch_योग;
    स्थिर पूर्णांक n_gemm = (पूर्णांक) oc;
    स्थिर पूर्णांक k = (पूर्णांक) knl_n_योग;

    // Combined kernel: im2col -> pack A, and pack B simultaneously
    स्थिर अक्षर * src1_base = (स्थिर अक्षर *) src1->तथ्य;
    स्थिर अक्षर * src0_base = (स्थिर अक्षर *) src0->तथ्य;
    स्थिर int64_t src1_nb0 = src1->nb[0];
    स्थिर int64_t src1_nb1 = src1->nb[1];
    स्थिर int64_t src1_nb2 = src1->nb[2];
    स्थिर int64_t src1_nb3 = src1->nb[3];
    स्थिर int64_t src1_w = src1->ne[0];
    स्थिर int64_t src1_h = src1->ne[1];
    स्थिर int64_t src1_d = src1->ne[2];

    स्थिर bool src0_is_f32 = (src0->type == GGML_TYPE_F32);

    // Compute correct strides for src0 as (knl_n_total, oc) matrix
    स्थिर int64_t src0_packed_nb0 = kernel_type_आकार;
    स्थिर int64_t src0_packed_nb1 = kernel_type_आकार * knl_n_योग;

    स्थिर int64_t KW = knl_w;
    स्थिर int64_t KH = knl_h;
    स्थिर int64_t KD = knl_d;
    स्थिर int64_t PW = dst->ne[0];
    स्थिर int64_t PH = dst->ne[1];
    स्थिर int64_t PD = dst->ne[2];

    // Pack A (with inline im2col): for each (row, col) in k x m matrix
    स्थिर int64_t A_योग = (int64_t)k * m;
    स्थिर int64_t A_block_आकार = 256;
    स्थिर int64_t A_अंक_work = ((A_योग + A_block_आकार - 1) / A_block_आकार) * A_block_आकार;

    stream->parallel_for(sycl::range<1>(A_अंक_work), [=](sycl::id<1> id) {
        स्थिर int64_t t = id[0];
        अगर (t >= A_योग) वापस;

        स्थिर int64_t row = t % k;
        स्थिर int64_t col = t / k;

        // Inline im2col for this element
        स्थिर int64_t k_सूचकांक = row;
        स्थिर int64_t patch_idx = col;

        स्थिर int64_t ic = k_सूचकांक / (KD * KH * KW);
        स्थिर int64_t rem = k_सूचकांक - ic * (KD * KH * KW);
        स्थिर int64_t kz = rem / (KH * KW);
        स्थिर int64_t rem2 = rem - kz * (KH * KW);
        स्थिर int64_t ky = rem2 / KW;
        स्थिर int64_t kx = rem2 % KW;

        स्थिर int64_t p_in_batch = patch_idx % (PW * PH * PD);
        स्थिर int64_t batch_idx = patch_idx / (PW * PH * PD);
        स्थिर int64_t dst_z = p_in_batch / (PW * PH);
        स्थिर int64_t dst_y = (p_in_batch % (PW * PH)) / PW;
        स्थिर int64_t dst_x = p_in_batch % PW;

        स्थिर int64_t sx = dst_x * s0 + kx * d0 - p0;
        स्थिर int64_t sy = dst_y * s1 + ky * d1 - p1;
        स्थिर int64_t sz = dst_z * s2 + kz * d2 - p2;

        भग्न val = 0.0f;
        अगर (sx >= 0 && sx < src1_w && sy >= 0 && sy < src1_h && sz >= 0 && sz < src1_d) {
            स्थिर int64_t channel_idx = batch_idx * c + ic;
            स्थिर अक्षर * ptr = src1_base + sx * src1_nb0 + sy * src1_nb1 + sz * src1_nb2 + channel_idx * src1_nb3;
            val = *(स्थिर भग्न *) ptr;
        }
        A_packed[row + col * (int64_t)k] = val;
    });

    // Pack B: for each (row, col) in k x n_gemm matrix
    स्थिर int64_t B_योग = (int64_t)k * n_gemm;
    स्थिर int64_t B_block_आकार = 256;
    स्थिर int64_t B_अंक_work = ((B_योग + B_block_आकार - 1) / B_block_आकार) * B_block_आकार;

    stream->parallel_for(sycl::range<1>(B_अंक_work), [=](sycl::id<1> id) {
        स्थिर int64_t t = id[0];
        अगर (t >= B_योग) वापस;

        स्थिर int64_t row = t % k;
        स्थिर int64_t col = t / k;
        स्थिर अक्षर * src_ptr = src0_base + row * src0_packed_nb0 + col * src0_packed_nb1;
        भग्न v;
        अगर (src0_is_f32) {
            v = *(स्थिर भग्न *) src_ptr;
        } अन्यथा {
            v = sycl::vec<sycl::half, 1>(*(स्थिर sycl::half *) src_ptr).convert<भग्न, sycl::rounding_mode::automatic>()[0];
        }
        B_packed[row + col * (int64_t)k] = v;
    });

    // GEMM: C = A^T * B where A is (k x m), B is (k x n), C is (m x n)
    स्थिर भग्न alpha = 1.0f;
    स्थिर भग्न beta  = 0.0f;
    स्थिर पूर्णांक lda = k;
    स्थिर पूर्णांक ldb = k;
    स्थिर पूर्णांक ldc = m;

    SYCL_CHECK(CHECK_TRY_ERROR(oneapi::mkl::blas::column_major::gemm(
        *stream, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans,
        m, n_gemm, k,
        dpct::get_मूल्य(&alpha, *stream),
        (स्थिर भग्न *) A_packed, lda,
        (स्थिर भग्न *) B_packed, ldb,
        dpct::get_मूल्य(&beta, *stream),
        (भग्न *) dst_mat.तथ्य, ldc)));

    स्थिर भग्न * gemm_तथ्य = (स्थिर भग्न *) dst_mat.तथ्य;
    भग्न * dst_तथ्य = (भग्न *) dst->तथ्य;

    ggml_sycl_conv3d_write_output(dst, gemm_तथ्य, dst_तथ्य, patch_योग, oc,
                                  dst->ne[0], dst->ne[1], dst->ne[2], stream);
}
