#ضمن "conv3d.hpp"

ساكن مضمن int64_t ggml_sycl_conv3d_calc_patch_total(ثابت ggml_tensor * dst, int32_t n) {
    ܦܢܐ (int64_t) n * dst->ne[0] * dst->ne[1] * dst->ne[2];
}

ساكن مضمن int64_t ggml_sycl_conv3d_calc_knl_n_total(ثابت ggml_tensor * src0, int32_t c) {
    ܦܢܐ (int64_t) src0->ne[0] * src0->ne[1] * src0->ne[2] * c;
}

ساكن مضمن فراغ ggml_sycl_conv3d_write_output(
        ثابت ggml_tensor * dst,
        ثابت عائم * src, عائم * dst_data,
        int64_t patch_total, int64_t oc,
        int64_t dst_w, int64_t dst_h, int64_t dst_d,
        dpct::queue_ptr stream) {
    ثابت int64_t dst_nb0 = dst->nb[0];
    ثابت int64_t dst_nb1 = dst->nb[1];
    ثابت int64_t dst_nb2 = dst->nb[2];
    ثابت int64_t dst_nb3 = dst->nb[3];
    ثابت int64_t total = patch_total * oc;
    ثابت int64_t block_size = 256;
    ثابت int64_t num_work_items = ((total + block_size - 1) / block_size) * block_size;

    stream->parallel_for(sycl::range<1>(num_work_items), [=](sycl::id<1> id) {
        ثابت int64_t i = id[0];
        ܐܢ (i >= total) {
            ܦܢܐ;
        }

        ثابت int64_t patch_idx = i / oc;
        ثابت int64_t out_ch = i % oc;
        ثابت int64_t p_in_batch = patch_idx % (dst_w * dst_h * dst_d);
        ثابت int64_t batch_idx = patch_idx / (dst_w * dst_h * dst_d);
        ثابت int64_t dst_z = p_in_batch / (dst_w * dst_h);
        ثابت int64_t dst_y = (p_in_batch % (dst_w * dst_h)) / dst_w;
        ثابت int64_t dst_x = p_in_batch % dst_w;
        ثابت int64_t ocn_idx = batch_idx * oc + out_ch;

        ثابت int64_t dst_offset = dst_x * dst_nb0 + dst_y * dst_nb1 + dst_z * dst_nb2 + ocn_idx * dst_nb3;
        // `src` is a column-major (m x n) GEMM output where m == patch_total, n == oc.
        // GEMM stores element (row, col) at index `row + col*m`, so compute index accordingly.
        ثابت int64_t src_index = patch_idx + out_ch * patch_total;
        ثابت عائم value = src[src_index];
        *(عائم *)((حرف *)dst_data + dst_offset) = value;
    });
}

فراغ ggml_sycl_op_conv_3d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));

    ثابت int32_t * opts = (ثابت int32_t *) dst->op_params;
    ثابت int32_t s0 = opts[0];
    ثابت int32_t s1 = opts[1];
    ثابت int32_t s2 = opts[2];
    ثابت int32_t p0 = opts[3];
    ثابت int32_t p1 = opts[4];
    ثابت int32_t p2 = opts[5];
    ثابت int32_t d0 = opts[6];
    ثابت int32_t d1 = opts[7];
    ثابت int32_t d2 = opts[8];
    ثابت int32_t c  = opts[9];
    ثابت int32_t n  = opts[10];
    ثابت int32_t oc = opts[11];

    ثابت int64_t knl_w = src0->ne[0];
    ثابت int64_t knl_h = src0->ne[1];
    ثابت int64_t knl_d = src0->ne[2];

    ثابت int64_t patch_total = ggml_sycl_conv3d_calc_patch_total(dst, n);
    ثابت int64_t knl_n_total = ggml_sycl_conv3d_calc_knl_n_total(src0, c);

    ثابت size_t kernel_type_size = ggml_element_size(src0);

    ggml_sycl_pool_alloc<عائم> gemm_output(ctx.pool());
    gemm_output.alloc((size_t) patch_total * oc);

    ggml_tensor dst_mat = {};
    dst_mat.type = GGML_TYPE_F32;
    dst_mat.ne[0] = patch_total;
    dst_mat.ne[1] = oc;
    dst_mat.ne[2] = 1;
    dst_mat.ne[3] = 1;
    dst_mat.nb[0] = حجم(عائم);
    dst_mat.nb[1] = dst_mat.nb[0] * dst_mat.ne[0];
    dst_mat.nb[2] = dst_mat.nb[1];
    dst_mat.nb[3] = dst_mat.nb[2];
    dst_mat.data = gemm_output.get();
    dst_mat.buffer = dst->buffer;
    dst_mat.extra = dst->extra;

    dpct::queue_ptr stream = ctx.stream();

    // allocate packed arrays: A_packed (k x m), B_packed (k x n)
    ggml_sycl_pool_alloc<عائم> A_packed_alloc(ctx.pool());
    ggml_sycl_pool_alloc<عائم> B_packed_alloc(ctx.pool());
    A_packed_alloc.alloc((size_t) knl_n_total * patch_total);
    B_packed_alloc.alloc((size_t) knl_n_total * oc);

    عائم * A_packed = A_packed_alloc.get();
    عائم * B_packed = B_packed_alloc.get();

    ثابت صحيح m = (صحيح) patch_total;
    ثابت صحيح n_gemm = (صحيح) oc;
    ثابت صحيح k = (صحيح) knl_n_total;

    // Combined kernel: im2col -> pack A, and pack B simultaneously
    ثابت حرف * src1_base = (ثابت حرف *) src1->data;
    ثابت حرف * src0_base = (ثابت حرف *) src0->data;
    ثابت int64_t src1_nb0 = src1->nb[0];
    ثابت int64_t src1_nb1 = src1->nb[1];
    ثابت int64_t src1_nb2 = src1->nb[2];
    ثابت int64_t src1_nb3 = src1->nb[3];
    ثابت int64_t src1_w = src1->ne[0];
    ثابت int64_t src1_h = src1->ne[1];
    ثابت int64_t src1_d = src1->ne[2];

    ثابت منطقي src0_is_f32 = (src0->type == GGML_TYPE_F32);

    // Compute correct strides for src0 as (knl_n_total, oc) matrix
    ثابت int64_t src0_packed_nb0 = kernel_type_size;
    ثابت int64_t src0_packed_nb1 = kernel_type_size * knl_n_total;

    ثابت int64_t KW = knl_w;
    ثابت int64_t KH = knl_h;
    ثابت int64_t KD = knl_d;
    ثابت int64_t PW = dst->ne[0];
    ثابت int64_t PH = dst->ne[1];
    ثابت int64_t PD = dst->ne[2];

    // Pack A (with inline im2col): for each (row, col) in k x m matrix
    ثابت int64_t A_total = (int64_t)k * m;
    ثابت int64_t A_block_size = 256;
    ثابت int64_t A_num_work = ((A_total + A_block_size - 1) / A_block_size) * A_block_size;

    stream->parallel_for(sycl::range<1>(A_num_work), [=](sycl::id<1> id) {
        ثابت int64_t t = id[0];
        ܐܢ (t >= A_total) ܦܢܐ;

        ثابت int64_t row = t % k;
        ثابت int64_t col = t / k;

        // Inline im2col for this element
        ثابت int64_t k_index = row;
        ثابت int64_t patch_idx = col;

        ثابت int64_t ic = k_index / (KD * KH * KW);
        ثابت int64_t rem = k_index - ic * (KD * KH * KW);
        ثابت int64_t kz = rem / (KH * KW);
        ثابت int64_t rem2 = rem - kz * (KH * KW);
        ثابت int64_t ky = rem2 / KW;
        ثابت int64_t kx = rem2 % KW;

        ثابت int64_t p_in_batch = patch_idx % (PW * PH * PD);
        ثابت int64_t batch_idx = patch_idx / (PW * PH * PD);
        ثابت int64_t dst_z = p_in_batch / (PW * PH);
        ثابت int64_t dst_y = (p_in_batch % (PW * PH)) / PW;
        ثابت int64_t dst_x = p_in_batch % PW;

        ثابت int64_t sx = dst_x * s0 + kx * d0 - p0;
        ثابت int64_t sy = dst_y * s1 + ky * d1 - p1;
        ثابت int64_t sz = dst_z * s2 + kz * d2 - p2;

        عائم val = 0.0f;
        ܐܢ (sx >= 0 && sx < src1_w && sy >= 0 && sy < src1_h && sz >= 0 && sz < src1_d) {
            ثابت int64_t channel_idx = batch_idx * c + ic;
            ثابت حرف * ptr = src1_base + sx * src1_nb0 + sy * src1_nb1 + sz * src1_nb2 + channel_idx * src1_nb3;
            val = *(ثابت عائم *) ptr;
        }
        A_packed[row + col * (int64_t)k] = val;
    });

    // Pack B: for each (row, col) in k x n_gemm matrix
    ثابت int64_t B_total = (int64_t)k * n_gemm;
    ثابت int64_t B_block_size = 256;
    ثابت int64_t B_num_work = ((B_total + B_block_size - 1) / B_block_size) * B_block_size;

    stream->parallel_for(sycl::range<1>(B_num_work), [=](sycl::id<1> id) {
        ثابت int64_t t = id[0];
        ܐܢ (t >= B_total) ܦܢܐ;

        ثابت int64_t row = t % k;
        ثابت int64_t col = t / k;
        ثابت حرف * src_ptr = src0_base + row * src0_packed_nb0 + col * src0_packed_nb1;
        عائم v;
        ܐܢ (src0_is_f32) {
            v = *(ثابت عائم *) src_ptr;
        } ܐܚܪܢܐ {
            v = sycl::vec<sycl::half, 1>(*(ثابت sycl::half *) src_ptr).convert<عائم, sycl::rounding_mode::automatic>()[0];
        }
        B_packed[row + col * (int64_t)k] = v;
    });

    // GEMM: C = A^T * B where A is (k x m), B is (k x n), C is (m x n)
    ثابت عائم alpha = 1.0f;
    ثابت عائم beta  = 0.0f;
    ثابت صحيح lda = k;
    ثابت صحيح ldb = k;
    ثابت صحيح ldc = m;

    SYCL_CHECK(CHECK_TRY_ERROR(oneapi::mkl::blas::column_major::gemm(
        *stream, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans,
        m, n_gemm, k,
        dpct::get_value(&alpha, *stream),
        (ثابت عائم *) A_packed, lda,
        (ثابت عائم *) B_packed, ldb,
        dpct::get_value(&beta, *stream),
        (عائم *) dst_mat.data, ldc)));

    ثابت عائم * gemm_data = (ثابت عائم *) dst_mat.data;
    عائم * dst_data = (عائم *) dst->data;

    ggml_sycl_conv3d_write_output(dst, gemm_data, dst_data, patch_total, oc,
                                  dst->ne[0], dst->ne[1], dst->ne[2], stream);
}
