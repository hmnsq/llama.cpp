#అన్తర్భావ "conv3d.hpp"

స్థిర అంతఃవరుస int64_t ggml_sycl_conv3d_calc_patch_మొత్తం(స్థిరరాశి ggml_tensor * dst, int32_t n) {
    తిరిగి (int64_t) n * dst->ne[0] * dst->ne[1] * dst->ne[2];
}

స్థిర అంతఃవరుస int64_t ggml_sycl_conv3d_calc_knl_n_మొత్తం(స్థిరరాశి ggml_tensor * src0, int32_t c) {
    తిరిగి (int64_t) src0->ne[0] * src0->ne[1] * src0->ne[2] * c;
}

స్థిర అంతఃవరుస శూన్య ggml_sycl_conv3d_write_output(
        స్థిరరాశి ggml_tensor * dst,
        స్థిరరాశి దశాంశం * src, దశాంశం * dst_దత్తాంశం,
        int64_t patch_మొత్తం, int64_t oc,
        int64_t dst_w, int64_t dst_h, int64_t dst_d,
        dpct::వరుస_ptr stream) {
    స్థిరరాశి int64_t dst_nb0 = dst->nb[0];
    స్థిరరాశి int64_t dst_nb1 = dst->nb[1];
    స్థిరరాశి int64_t dst_nb2 = dst->nb[2];
    స్థిరరాశి int64_t dst_nb3 = dst->nb[3];
    స్థిరరాశి int64_t మొత్తం = patch_మొత్తం * oc;
    స్థిరరాశి int64_t ఖండం_కొలత_3 = 256;
    స్థిరరాశి int64_t అంకె_work_వస్తువులు = ((మొత్తం + ఖండం_కొలత_3 - 1) / ఖండం_కొలత_3) * ఖండం_కొలత_3;

    stream->parallel_for(sycl::range<1>(అంకె_work_వస్తువులు), [=](sycl::id<1> id) {
        స్థిరరాశి int64_t i = id[0];
        ఒకవేళ (i >= మొత్తం) {
            తిరిగి;
        }

        స్థిరరాశి int64_t patch_idx = i / oc;
        స్థిరరాశి int64_t బహిర్గతం_ch = i % oc;
        స్థిరరాశి int64_t p_లోపల_batch = patch_idx % (dst_w * dst_h * dst_d);
        స్థిరరాశి int64_t batch_idx = patch_idx / (dst_w * dst_h * dst_d);
        స్థిరరాశి int64_t dst_z = p_లోపల_batch / (dst_w * dst_h);
        స్థిరరాశి int64_t dst_y = (p_లోపల_batch % (dst_w * dst_h)) / dst_w;
        స్థిరరాశి int64_t dst_x = p_లోపల_batch % dst_w;
        స్థిరరాశి int64_t ocn_idx = batch_idx * oc + బహిర్గతం_ch;

        స్థిరరాశి int64_t dst_స్థానభ్రంశం = dst_x * dst_nb0 + dst_y * dst_nb1 + dst_z * dst_nb2 + ocn_idx * dst_nb3;
        // `src` is a column-major (m x n) GEMM output where m == patch_total, n == oc.
        // GEMM stores element (row, col) at index `row + col*m`, so compute index accordingly.
        స్థిరరాశి int64_t src_సూచీ = patch_idx + బహిర్గతం_ch * patch_మొత్తం;
        స్థిరరాశి దశాంశం value = src[src_సూచీ];
        *(దశాంశం *)((అక్షరం *)dst_దత్తాంశం + dst_స్థానభ్రంశం) = value;
    });
}

శూన్య ggml_sycl_op_conv_3d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));

    స్థిరరాశి int32_t * opts = (స్థిరరాశి int32_t *) dst->op_params;
    స్థిరరాశి int32_t s0 = opts[0];
    స్థిరరాశి int32_t s1 = opts[1];
    స్థిరరాశి int32_t s2 = opts[2];
    స్థిరరాశి int32_t p0 = opts[3];
    స్థిరరాశి int32_t p1 = opts[4];
    స్థిరరాశి int32_t p2 = opts[5];
    స్థిరరాశి int32_t d0 = opts[6];
    స్థిరరాశి int32_t d1 = opts[7];
    స్థిరరాశి int32_t d2 = opts[8];
    స్థిరరాశి int32_t c  = opts[9];
    స్థిరరాశి int32_t n  = opts[10];
    స్థిరరాశి int32_t oc = opts[11];

    స్థిరరాశి int64_t knl_w = src0->ne[0];
    స్థిరరాశి int64_t knl_h = src0->ne[1];
    స్థిరరాశి int64_t knl_d = src0->ne[2];

    స్థిరరాశి int64_t patch_మొత్తం = ggml_sycl_conv3d_calc_patch_మొత్తం(dst, n);
    స్థిరరాశి int64_t knl_n_మొత్తం = ggml_sycl_conv3d_calc_knl_n_మొత్తం(src0, c);

    స్థిరరాశి size_t kernel_type_కొలత = ggml_element_కొలత(src0);

    ggml_sycl_pool_alloc<దశాంశం> gemm_output(ctx.pool());
    gemm_output.alloc((size_t) patch_మొత్తం * oc);

    ggml_tensor dst_mat = {};
    dst_mat.type = GGML_TYPE_F32;
    dst_mat.ne[0] = patch_మొత్తం;
    dst_mat.ne[1] = oc;
    dst_mat.ne[2] = 1;
    dst_mat.ne[3] = 1;
    dst_mat.nb[0] = పరిమాణం(దశాంశం);
    dst_mat.nb[1] = dst_mat.nb[0] * dst_mat.ne[0];
    dst_mat.nb[2] = dst_mat.nb[1];
    dst_mat.nb[3] = dst_mat.nb[2];
    dst_mat.దత్తాంశం = gemm_output.get();
    dst_mat.బఫరు = dst->బఫరు;
    dst_mat.extra = dst->extra;

    dpct::వరుస_ptr stream = ctx.stream();

    // allocate packed arrays: A_packed (k x m), B_packed (k x n)
    ggml_sycl_pool_alloc<దశాంశం> A_packed_alloc(ctx.pool());
    ggml_sycl_pool_alloc<దశాంశం> B_packed_alloc(ctx.pool());
    A_packed_alloc.alloc((size_t) knl_n_మొత్తం * patch_మొత్తం);
    B_packed_alloc.alloc((size_t) knl_n_మొత్తం * oc);

    దశాంశం * A_packed = A_packed_alloc.get();
    దశాంశం * B_packed = B_packed_alloc.get();

    స్థిరరాశి పూర్ణాంకం m = (పూర్ణాంకం) patch_మొత్తం;
    స్థిరరాశి పూర్ణాంకం n_gemm = (పూర్ణాంకం) oc;
    స్థిరరాశి పూర్ణాంకం k = (పూర్ణాంకం) knl_n_మొత్తం;

    // Combined kernel: im2col -> pack A, and pack B simultaneously
    స్థిరరాశి అక్షరం * src1_పునాది = (స్థిరరాశి అక్షరం *) src1->దత్తాంశం;
    స్థిరరాశి అక్షరం * src0_పునాది = (స్థిరరాశి అక్షరం *) src0->దత్తాంశం;
    స్థిరరాశి int64_t src1_nb0 = src1->nb[0];
    స్థిరరాశి int64_t src1_nb1 = src1->nb[1];
    స్థిరరాశి int64_t src1_nb2 = src1->nb[2];
    స్థిరరాశి int64_t src1_nb3 = src1->nb[3];
    స్థిరరాశి int64_t src1_w = src1->ne[0];
    స్థిరరాశి int64_t src1_h = src1->ne[1];
    స్థిరరాశి int64_t src1_d = src1->ne[2];

    స్థిరరాశి తార్కిక src0_is_f32 = (src0->type == GGML_TYPE_F32);

    // Compute correct strides for src0 as (knl_n_total, oc) matrix
    స్థిరరాశి int64_t src0_packed_nb0 = kernel_type_కొలత;
    స్థిరరాశి int64_t src0_packed_nb1 = kernel_type_కొలత * knl_n_మొత్తం;

    స్థిరరాశి int64_t KW = knl_w;
    స్థిరరాశి int64_t KH = knl_h;
    స్థిరరాశి int64_t KD = knl_d;
    స్థిరరాశి int64_t PW = dst->ne[0];
    స్థిరరాశి int64_t PH = dst->ne[1];
    స్థిరరాశి int64_t PD = dst->ne[2];

    // Pack A (with inline im2col): for each (row, col) in k x m matrix
    స్థిరరాశి int64_t A_మొత్తం = (int64_t)k * m;
    స్థిరరాశి int64_t A_ఖండం_కొలత = 256;
    స్థిరరాశి int64_t A_అంకె_work = ((A_మొత్తం + A_ఖండం_కొలత - 1) / A_ఖండం_కొలత) * A_ఖండం_కొలత;

    stream->parallel_for(sycl::range<1>(A_అంకె_work), [=](sycl::id<1> id) {
        స్థిరరాశి int64_t t = id[0];
        ఒకవేళ (t >= A_మొత్తం) తిరిగి;

        స్థిరరాశి int64_t అడ్డువరుస = t % k;
        స్థిరరాశి int64_t నిలువు = t / k;

        // Inline im2col for this element
        స్థిరరాశి int64_t k_సూచీ = అడ్డువరుస;
        స్థిరరాశి int64_t patch_idx = నిలువు;

        స్థిరరాశి int64_t ic = k_సూచీ / (KD * KH * KW);
        స్థిరరాశి int64_t rem = k_సూచీ - ic * (KD * KH * KW);
        స్థిరరాశి int64_t kz = rem / (KH * KW);
        స్థిరరాశి int64_t rem2 = rem - kz * (KH * KW);
        స్థిరరాశి int64_t ky = rem2 / KW;
        స్థిరరాశి int64_t kx = rem2 % KW;

        స్థిరరాశి int64_t p_లోపల_batch = patch_idx % (PW * PH * PD);
        స్థిరరాశి int64_t batch_idx = patch_idx / (PW * PH * PD);
        స్థిరరాశి int64_t dst_z = p_లోపల_batch / (PW * PH);
        స్థిరరాశి int64_t dst_y = (p_లోపల_batch % (PW * PH)) / PW;
        స్థిరరాశి int64_t dst_x = p_లోపల_batch % PW;

        స్థిరరాశి int64_t sx = dst_x * s0 + kx * d0 - p0;
        స్థిరరాశి int64_t sy = dst_y * s1 + ky * d1 - p1;
        స్థిరరాశి int64_t sz = dst_z * s2 + kz * d2 - p2;

        దశాంశం val = 0.0f;
        ఒకవేళ (sx >= 0 && sx < src1_w && sy >= 0 && sy < src1_h && sz >= 0 && sz < src1_d) {
            స్థిరరాశి int64_t channel_idx = batch_idx * c + ic;
            స్థిరరాశి అక్షరం * ptr = src1_పునాది + sx * src1_nb0 + sy * src1_nb1 + sz * src1_nb2 + channel_idx * src1_nb3;
            val = *(స్థిరరాశి దశాంశం *) ptr;
        }
        A_packed[అడ్డువరుస + నిలువు * (int64_t)k] = val;
    });

    // Pack B: for each (row, col) in k x n_gemm matrix
    స్థిరరాశి int64_t B_మొత్తం = (int64_t)k * n_gemm;
    స్థిరరాశి int64_t B_ఖండం_కొలత = 256;
    స్థిరరాశి int64_t B_అంకె_work = ((B_మొత్తం + B_ఖండం_కొలత - 1) / B_ఖండం_కొలత) * B_ఖండం_కొలత;

    stream->parallel_for(sycl::range<1>(B_అంకె_work), [=](sycl::id<1> id) {
        స్థిరరాశి int64_t t = id[0];
        ఒకవేళ (t >= B_మొత్తం) తిరిగి;

        స్థిరరాశి int64_t అడ్డువరుస = t % k;
        స్థిరరాశి int64_t నిలువు = t / k;
        స్థిరరాశి అక్షరం * src_ptr = src0_పునాది + అడ్డువరుస * src0_packed_nb0 + నిలువు * src0_packed_nb1;
        దశాంశం v;
        ఒకవేళ (src0_is_f32) {
            v = *(స్థిరరాశి దశాంశం *) src_ptr;
        } లేకపోతే {
            v = sycl::vec<sycl::half, 1>(*(స్థిరరాశి sycl::half *) src_ptr).convert<దశాంశం, sycl::rounding_బహుళకం::automatic>()[0];
        }
        B_packed[అడ్డువరుస + నిలువు * (int64_t)k] = v;
    });

    // GEMM: C = A^T * B where A is (k x m), B is (k x n), C is (m x n)
    స్థిరరాశి దశాంశం alpha = 1.0f;
    స్థిరరాశి దశాంశం beta  = 0.0f;
    స్థిరరాశి పూర్ణాంకం lda = k;
    స్థిరరాశి పూర్ణాంకం ldb = k;
    స్థిరరాశి పూర్ణాంకం ldc = m;

    SYCL_CHECK(CHECK_TRY_ERROR(oneapi::mkl::blas::స్తంభం_major::gemm(
        *stream, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans,
        m, n_gemm, k,
        dpct::get_విలువ(&alpha, *stream),
        (స్థిరరాశి దశాంశం *) A_packed, lda,
        (స్థిరరాశి దశాంశం *) B_packed, ldb,
        dpct::get_విలువ(&beta, *stream),
        (దశాంశం *) dst_mat.దత్తాంశం, ldc)));

    స్థిరరాశి దశాంశం * gemm_దత్తాంశం = (స్థిరరాశి దశాంశం *) dst_mat.దత్తాంశం;
    దశాంశం * dst_దత్తాంశం = (దశాంశం *) dst->దత్తాంశం;

    ggml_sycl_conv3d_write_output(dst, gemm_దత్తాంశం, dst_దత్తాంశం, patch_మొత్తం, oc,
                                  dst->ne[0], dst->ne[1], dst->ne[2], stream);
}
