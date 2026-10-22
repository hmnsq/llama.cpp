#अन्तर्भाव "conv3d.hpp"

स्थिर अन्तःस्थ int64_t ggml_sycl_conv3d_calc_patch_योग(ध्रुव ggml_tensor * dst, int32_t n) {
    प्रत्यागम (int64_t) n * dst->ne[0] * dst->ne[1] * dst->ne[2];
}

स्थिर अन्तःस्थ int64_t ggml_sycl_conv3d_calc_knl_n_योग(ध्रुव ggml_tensor * src0, int32_t c) {
    प्रत्यागम (int64_t) src0->ne[0] * src0->ne[1] * src0->ne[2] * c;
}

स्थिर अन्तःस्थ शून्य ggml_sycl_conv3d_write_output(
        ध्रुव ggml_tensor * dst,
        ध्रुव भिन्न * src, भिन्न * dst_दत्त,
        int64_t patch_योग, int64_t oc,
        int64_t dst_w, int64_t dst_h, int64_t dst_d,
        dpct::queue_ptr stream) {
    ध्रुव int64_t dst_nb0 = dst->nb[0];
    ध्रुव int64_t dst_nb1 = dst->nb[1];
    ध्रुव int64_t dst_nb2 = dst->nb[2];
    ध्रुव int64_t dst_nb3 = dst->nb[3];
    ध्रुव int64_t योग = patch_योग * oc;
    ध्रुव int64_t खण्डम्_आकार_3 = 256;
    ध्रुव int64_t अङ्कः_work_वस्तूनि = ((योग + खण्डम्_आकार_3 - 1) / खण्डम्_आकार_3) * खण्डम्_आकार_3;

    stream->parallel_for(sycl::range<1>(अङ्कः_work_वस्तूनि), [=](sycl::id<1> id) {
        ध्रुव int64_t i = id[0];
        यदि (i >= योग) {
            प्रत्यागम;
        }

        ध्रुव int64_t patch_idx = i / oc;
        ध्रुव int64_t बहिर्गमनम्_ch = i % oc;
        ध्रुव int64_t p_अन्तर्_batch = patch_idx % (dst_w * dst_h * dst_d);
        ध्रुव int64_t batch_idx = patch_idx / (dst_w * dst_h * dst_d);
        ध्रुव int64_t dst_z = p_अन्तर्_batch / (dst_w * dst_h);
        ध्रुव int64_t dst_y = (p_अन्तर्_batch % (dst_w * dst_h)) / dst_w;
        ध्रुव int64_t dst_x = p_अन्तर्_batch % dst_w;
        ध्रुव int64_t ocn_idx = batch_idx * oc + बहिर्गमनम्_ch;

        ध्रुव int64_t dst_विस्थापनम् = dst_x * dst_nb0 + dst_y * dst_nb1 + dst_z * dst_nb2 + ocn_idx * dst_nb3;
        // `src` is a column-major (m x n) GEMM output where m == patch_total, n == oc.
        // GEMM stores element (row, col) at index `row + col*m`, so compute index accordingly.
        ध्रुव int64_t src_सूचकांक = patch_idx + बहिर्गमनम्_ch * patch_योग;
        ध्रुव भिन्न value = src[src_सूचकांक];
        *(भिन्न *)((वर्ण *)dst_दत्त + dst_विस्थापनम्) = value;
    });
}

शून्य ggml_sycl_op_conv_3d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));

    ध्रुव int32_t * opts = (ध्रुव int32_t *) dst->op_params;
    ध्रुव int32_t s0 = opts[0];
    ध्रुव int32_t s1 = opts[1];
    ध्रुव int32_t s2 = opts[2];
    ध्रुव int32_t p0 = opts[3];
    ध्रुव int32_t p1 = opts[4];
    ध्रुव int32_t p2 = opts[5];
    ध्रुव int32_t d0 = opts[6];
    ध्रुव int32_t d1 = opts[7];
    ध्रुव int32_t d2 = opts[8];
    ध्रुव int32_t c  = opts[9];
    ध्रुव int32_t n  = opts[10];
    ध्रुव int32_t oc = opts[11];

    ध्रुव int64_t knl_w = src0->ne[0];
    ध्रुव int64_t knl_h = src0->ne[1];
    ध्रुव int64_t knl_d = src0->ne[2];

    ध्रुव int64_t patch_योग = ggml_sycl_conv3d_calc_patch_योग(dst, n);
    ध्रुव int64_t knl_n_योग = ggml_sycl_conv3d_calc_knl_n_योग(src0, c);

    ध्रुव size_t kernel_type_आकार = ggml_element_आकार(src0);

    ggml_sycl_pool_alloc<भिन्न> gemm_output(ctx.pool());
    gemm_output.alloc((size_t) patch_योग * oc);

    ggml_tensor dst_mat = {};
    dst_mat.type = GGML_TYPE_F32;
    dst_mat.ne[0] = patch_योग;
    dst_mat.ne[1] = oc;
    dst_mat.ne[2] = 1;
    dst_mat.ne[3] = 1;
    dst_mat.nb[0] = परिमाण(भिन्न);
    dst_mat.nb[1] = dst_mat.nb[0] * dst_mat.ne[0];
    dst_mat.nb[2] = dst_mat.nb[1];
    dst_mat.nb[3] = dst_mat.nb[2];
    dst_mat.दत्त = gemm_output.get();
    dst_mat.बफर = dst->बफर;
    dst_mat.extra = dst->extra;

    dpct::queue_ptr stream = ctx.stream();

    // allocate packed arrays: A_packed (k x m), B_packed (k x n)
    ggml_sycl_pool_alloc<भिन्न> A_packed_alloc(ctx.pool());
    ggml_sycl_pool_alloc<भिन्न> B_packed_alloc(ctx.pool());
    A_packed_alloc.alloc((size_t) knl_n_योग * patch_योग);
    B_packed_alloc.alloc((size_t) knl_n_योग * oc);

    भिन्न * A_packed = A_packed_alloc.get();
    भिन्न * B_packed = B_packed_alloc.get();

    ध्रुव पूर्णाङ्क m = (पूर्णाङ्क) patch_योग;
    ध्रुव पूर्णाङ्क n_gemm = (पूर्णाङ्क) oc;
    ध्रुव पूर्णाङ्क k = (पूर्णाङ्क) knl_n_योग;

    // Combined kernel: im2col -> pack A, and pack B simultaneously
    ध्रुव वर्ण * src1_आधारः = (ध्रुव वर्ण *) src1->दत्त;
    ध्रुव वर्ण * src0_आधारः = (ध्रुव वर्ण *) src0->दत्त;
    ध्रुव int64_t src1_nb0 = src1->nb[0];
    ध्रुव int64_t src1_nb1 = src1->nb[1];
    ध्रुव int64_t src1_nb2 = src1->nb[2];
    ध्रुव int64_t src1_nb3 = src1->nb[3];
    ध्रुव int64_t src1_w = src1->ne[0];
    ध्रुव int64_t src1_h = src1->ne[1];
    ध्रुव int64_t src1_d = src1->ne[2];

    ध्रुव तार्किक src0_is_f32 = (src0->type == GGML_TYPE_F32);

    // Compute correct strides for src0 as (knl_n_total, oc) matrix
    ध्रुव int64_t src0_packed_nb0 = kernel_type_आकार;
    ध्रुव int64_t src0_packed_nb1 = kernel_type_आकार * knl_n_योग;

    ध्रुव int64_t KW = knl_w;
    ध्रुव int64_t KH = knl_h;
    ध्रुव int64_t KD = knl_d;
    ध्रुव int64_t PW = dst->ne[0];
    ध्रुव int64_t PH = dst->ne[1];
    ध्रुव int64_t PD = dst->ne[2];

    // Pack A (with inline im2col): for each (row, col) in k x m matrix
    ध्रुव int64_t A_योग = (int64_t)k * m;
    ध्रुव int64_t A_खण्डम्_आकार = 256;
    ध्रुव int64_t A_अङ्कः_work = ((A_योग + A_खण्डम्_आकार - 1) / A_खण्डम्_आकार) * A_खण्डम्_आकार;

    stream->parallel_for(sycl::range<1>(A_अङ्कः_work), [=](sycl::id<1> id) {
        ध्रुव int64_t t = id[0];
        यदि (t >= A_योग) प्रत्यागम;

        ध्रुव int64_t row = t % k;
        ध्रुव int64_t col = t / k;

        // Inline im2col for this element
        ध्रुव int64_t k_सूचकांक = row;
        ध्रुव int64_t patch_idx = col;

        ध्रुव int64_t ic = k_सूचकांक / (KD * KH * KW);
        ध्रुव int64_t rem = k_सूचकांक - ic * (KD * KH * KW);
        ध्रुव int64_t kz = rem / (KH * KW);
        ध्रुव int64_t rem2 = rem - kz * (KH * KW);
        ध्रुव int64_t ky = rem2 / KW;
        ध्रुव int64_t kx = rem2 % KW;

        ध्रुव int64_t p_अन्तर्_batch = patch_idx % (PW * PH * PD);
        ध्रुव int64_t batch_idx = patch_idx / (PW * PH * PD);
        ध्रुव int64_t dst_z = p_अन्तर्_batch / (PW * PH);
        ध्रुव int64_t dst_y = (p_अन्तर्_batch % (PW * PH)) / PW;
        ध्रुव int64_t dst_x = p_अन्तर्_batch % PW;

        ध्रुव int64_t sx = dst_x * s0 + kx * d0 - p0;
        ध्रुव int64_t sy = dst_y * s1 + ky * d1 - p1;
        ध्रुव int64_t sz = dst_z * s2 + kz * d2 - p2;

        भिन्न val = 0.0f;
        यदि (sx >= 0 && sx < src1_w && sy >= 0 && sy < src1_h && sz >= 0 && sz < src1_d) {
            ध्रुव int64_t channel_idx = batch_idx * c + ic;
            ध्रुव वर्ण * ptr = src1_आधारः + sx * src1_nb0 + sy * src1_nb1 + sz * src1_nb2 + channel_idx * src1_nb3;
            val = *(ध्रुव भिन्न *) ptr;
        }
        A_packed[row + col * (int64_t)k] = val;
    });

    // Pack B: for each (row, col) in k x n_gemm matrix
    ध्रुव int64_t B_योग = (int64_t)k * n_gemm;
    ध्रुव int64_t B_खण्डम्_आकार = 256;
    ध्रुव int64_t B_अङ्कः_work = ((B_योग + B_खण्डम्_आकार - 1) / B_खण्डम्_आकार) * B_खण्डम्_आकार;

    stream->parallel_for(sycl::range<1>(B_अङ्कः_work), [=](sycl::id<1> id) {
        ध्रुव int64_t t = id[0];
        यदि (t >= B_योग) प्रत्यागम;

        ध्रुव int64_t row = t % k;
        ध्रुव int64_t col = t / k;
        ध्रुव वर्ण * src_ptr = src0_आधारः + row * src0_packed_nb0 + col * src0_packed_nb1;
        भिन्न v;
        यदि (src0_is_f32) {
            v = *(ध्रुव भिन्न *) src_ptr;
        } अन्यथा {
            v = sycl::vec<sycl::half, 1>(*(ध्रुव sycl::half *) src_ptr).convert<भिन्न, sycl::rounding_बहुलकम्::automatic>()[0];
        }
        B_packed[row + col * (int64_t)k] = v;
    });

    // GEMM: C = A^T * B where A is (k x m), B is (k x n), C is (m x n)
    ध्रुव भिन्न alpha = 1.0f;
    ध्रुव भिन्न beta  = 0.0f;
    ध्रुव पूर्णाङ्क lda = k;
    ध्रुव पूर्णाङ्क ldb = k;
    ध्रुव पूर्णाङ्क ldc = m;

    SYCL_CHECK(CHECK_TRY_ERROR(oneapi::mkl::blas::column_major::gemm(
        *stream, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans,
        m, n_gemm, k,
        dpct::get_मानम्(&alpha, *stream),
        (ध्रुव भिन्न *) A_packed, lda,
        (ध्रुव भिन्न *) B_packed, ldb,
        dpct::get_मानम्(&beta, *stream),
        (भिन्न *) dst_mat.दत्त, ldc)));

    ध्रुव भिन्न * gemm_दत्त = (ध्रुव भिन्न *) dst_mat.दत्त;
    भिन्न * dst_दत्त = (भिन्न *) dst->दत्त;

    ggml_sycl_conv3d_write_output(dst, gemm_दत्त, dst_दत्त, patch_योग, oc,
                                  dst->ne[0], dst->ne[1], dst->ne[2], stream);
}
