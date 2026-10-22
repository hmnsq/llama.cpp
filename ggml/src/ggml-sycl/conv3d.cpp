#অন্তর্ভাব "conv3d.hpp"

স্থির অন্তঃস্থ int64_t ggml_sycl_conv3d_calc_patch_total(ধ্রুব ggml_tensor * dst, int32_t n) {
    প্রত্যাগম (int64_t) n * dst->ne[0] * dst->ne[1] * dst->ne[2];
}

স্থির অন্তঃস্থ int64_t ggml_sycl_conv3d_calc_knl_n_total(ধ্রুব ggml_tensor * src0, int32_t c) {
    প্রত্যাগম (int64_t) src0->ne[0] * src0->ne[1] * src0->ne[2] * c;
}

স্থির অন্তঃস্থ শূন্য ggml_sycl_conv3d_write_output(
        ধ্রুব ggml_tensor * dst,
        ধ্রুব ভিন্ন * src, ভিন্ন * dst_data,
        int64_t patch_total, int64_t oc,
        int64_t dst_w, int64_t dst_h, int64_t dst_d,
        dpct::queue_ptr stream) {
    ধ্রুব int64_t dst_nb0 = dst->nb[0];
    ধ্রুব int64_t dst_nb1 = dst->nb[1];
    ধ্রুব int64_t dst_nb2 = dst->nb[2];
    ধ্রুব int64_t dst_nb3 = dst->nb[3];
    ধ্রুব int64_t total = patch_total * oc;
    ধ্রুব int64_t block_size = 256;
    ধ্রুব int64_t অঙ্ক_work_বস্তুসমূহ = ((total + block_size - 1) / block_size) * block_size;

    stream->parallel_for(sycl::range<1>(অঙ্ক_work_বস্তুসমূহ), [=](sycl::id<1> id) {
        ধ্রুব int64_t i = id[0];
        যদি (i >= total) {
            প্রত্যাগম;
        }

        ধ্রুব int64_t patch_idx = i / oc;
        ধ্রুব int64_t out_ch = i % oc;
        ধ্রুব int64_t p_in_batch = patch_idx % (dst_w * dst_h * dst_d);
        ধ্রুব int64_t batch_idx = patch_idx / (dst_w * dst_h * dst_d);
        ধ্রুব int64_t dst_z = p_in_batch / (dst_w * dst_h);
        ধ্রুব int64_t dst_y = (p_in_batch % (dst_w * dst_h)) / dst_w;
        ধ্রুব int64_t dst_x = p_in_batch % dst_w;
        ধ্রুব int64_t ocn_idx = batch_idx * oc + out_ch;

        ধ্রুব int64_t dst_offset = dst_x * dst_nb0 + dst_y * dst_nb1 + dst_z * dst_nb2 + ocn_idx * dst_nb3;
        // `src` is a column-major (m x n) GEMM output where m == patch_total, n == oc.
        // GEMM stores element (row, col) at index `row + col*m`, so compute index accordingly.
        ধ্রুব int64_t src_index = patch_idx + out_ch * patch_total;
        ধ্রুব ভিন্ন value = src[src_index];
        *(ভিন্ন *)((বর্ণ *)dst_data + dst_offset) = value;
    });
}

শূন্য ggml_sycl_op_conv_3d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));

    ধ্রুব int32_t * opts = (ধ্রুব int32_t *) dst->op_params;
    ধ্রুব int32_t s0 = opts[0];
    ধ্রুব int32_t s1 = opts[1];
    ধ্রুব int32_t s2 = opts[2];
    ধ্রুব int32_t p0 = opts[3];
    ধ্রুব int32_t p1 = opts[4];
    ধ্রুব int32_t p2 = opts[5];
    ধ্রুব int32_t d0 = opts[6];
    ধ্রুব int32_t d1 = opts[7];
    ধ্রুব int32_t d2 = opts[8];
    ধ্রুব int32_t c  = opts[9];
    ধ্রুব int32_t n  = opts[10];
    ধ্রুব int32_t oc = opts[11];

    ধ্রুব int64_t knl_w = src0->ne[0];
    ধ্রুব int64_t knl_h = src0->ne[1];
    ধ্রুব int64_t knl_d = src0->ne[2];

    ধ্রুব int64_t patch_total = ggml_sycl_conv3d_calc_patch_total(dst, n);
    ধ্রুব int64_t knl_n_total = ggml_sycl_conv3d_calc_knl_n_total(src0, c);

    ধ্রুব size_t kernel_type_size = ggml_element_size(src0);

    ggml_sycl_pool_alloc<ভিন্ন> gemm_output(ctx.pool());
    gemm_output.alloc((size_t) patch_total * oc);

    ggml_tensor dst_mat = {};
    dst_mat.type = GGML_TYPE_F32;
    dst_mat.ne[0] = patch_total;
    dst_mat.ne[1] = oc;
    dst_mat.ne[2] = 1;
    dst_mat.ne[3] = 1;
    dst_mat.nb[0] = পরিমাণ(ভিন্ন);
    dst_mat.nb[1] = dst_mat.nb[0] * dst_mat.ne[0];
    dst_mat.nb[2] = dst_mat.nb[1];
    dst_mat.nb[3] = dst_mat.nb[2];
    dst_mat.data = gemm_output.get();
    dst_mat.buffer = dst->buffer;
    dst_mat.extra = dst->extra;

    dpct::queue_ptr stream = ctx.stream();

    // allocate packed arrays: A_packed (k x m), B_packed (k x n)
    ggml_sycl_pool_alloc<ভিন্ন> A_packed_alloc(ctx.pool());
    ggml_sycl_pool_alloc<ভিন্ন> B_packed_alloc(ctx.pool());
    A_packed_alloc.alloc((size_t) knl_n_total * patch_total);
    B_packed_alloc.alloc((size_t) knl_n_total * oc);

    ভিন্ন * A_packed = A_packed_alloc.get();
    ভিন্ন * B_packed = B_packed_alloc.get();

    ধ্রুব পূর্ণাঙ্ক m = (পূর্ণাঙ্ক) patch_total;
    ধ্রুব পূর্ণাঙ্ক n_gemm = (পূর্ণাঙ্ক) oc;
    ধ্রুব পূর্ণাঙ্ক k = (পূর্ণাঙ্ক) knl_n_total;

    // Combined kernel: im2col -> pack A, and pack B simultaneously
    ধ্রুব বর্ণ * src1_base = (ধ্রুব বর্ণ *) src1->data;
    ধ্রুব বর্ণ * src0_base = (ধ্রুব বর্ণ *) src0->data;
    ধ্রুব int64_t src1_nb0 = src1->nb[0];
    ধ্রুব int64_t src1_nb1 = src1->nb[1];
    ধ্রুব int64_t src1_nb2 = src1->nb[2];
    ধ্রুব int64_t src1_nb3 = src1->nb[3];
    ধ্রুব int64_t src1_w = src1->ne[0];
    ধ্রুব int64_t src1_h = src1->ne[1];
    ধ্রুব int64_t src1_d = src1->ne[2];

    ধ্রুব তার্কিক src0_is_f32 = (src0->type == GGML_TYPE_F32);

    // Compute correct strides for src0 as (knl_n_total, oc) matrix
    ধ্রুব int64_t src0_packed_nb0 = kernel_type_size;
    ধ্রুব int64_t src0_packed_nb1 = kernel_type_size * knl_n_total;

    ধ্রুব int64_t KW = knl_w;
    ধ্রুব int64_t KH = knl_h;
    ধ্রুব int64_t KD = knl_d;
    ধ্রুব int64_t PW = dst->ne[0];
    ধ্রুব int64_t PH = dst->ne[1];
    ধ্রুব int64_t PD = dst->ne[2];

    // Pack A (with inline im2col): for each (row, col) in k x m matrix
    ধ্রুব int64_t A_total = (int64_t)k * m;
    ধ্রুব int64_t A_block_size = 256;
    ধ্রুব int64_t A_অঙ্ক_work = ((A_total + A_block_size - 1) / A_block_size) * A_block_size;

    stream->parallel_for(sycl::range<1>(A_অঙ্ক_work), [=](sycl::id<1> id) {
        ধ্রুব int64_t t = id[0];
        যদি (t >= A_total) প্রত্যাগম;

        ধ্রুব int64_t row = t % k;
        ধ্রুব int64_t col = t / k;

        // Inline im2col for this element
        ধ্রুব int64_t k_index = row;
        ধ্রুব int64_t patch_idx = col;

        ধ্রুব int64_t ic = k_index / (KD * KH * KW);
        ধ্রুব int64_t rem = k_index - ic * (KD * KH * KW);
        ধ্রুব int64_t kz = rem / (KH * KW);
        ধ্রুব int64_t rem2 = rem - kz * (KH * KW);
        ধ্রুব int64_t ky = rem2 / KW;
        ধ্রুব int64_t kx = rem2 % KW;

        ধ্রুব int64_t p_in_batch = patch_idx % (PW * PH * PD);
        ধ্রুব int64_t batch_idx = patch_idx / (PW * PH * PD);
        ধ্রুব int64_t dst_z = p_in_batch / (PW * PH);
        ধ্রুব int64_t dst_y = (p_in_batch % (PW * PH)) / PW;
        ধ্রুব int64_t dst_x = p_in_batch % PW;

        ধ্রুব int64_t sx = dst_x * s0 + kx * d0 - p0;
        ধ্রুব int64_t sy = dst_y * s1 + ky * d1 - p1;
        ধ্রুব int64_t sz = dst_z * s2 + kz * d2 - p2;

        ভিন্ন val = 0.0f;
        যদি (sx >= 0 && sx < src1_w && sy >= 0 && sy < src1_h && sz >= 0 && sz < src1_d) {
            ধ্রুব int64_t channel_idx = batch_idx * c + ic;
            ধ্রুব বর্ণ * ptr = src1_base + sx * src1_nb0 + sy * src1_nb1 + sz * src1_nb2 + channel_idx * src1_nb3;
            val = *(ধ্রুব ভিন্ন *) ptr;
        }
        A_packed[row + col * (int64_t)k] = val;
    });

    // Pack B: for each (row, col) in k x n_gemm matrix
    ধ্রুব int64_t B_total = (int64_t)k * n_gemm;
    ধ্রুব int64_t B_block_size = 256;
    ধ্রুব int64_t B_অঙ্ক_work = ((B_total + B_block_size - 1) / B_block_size) * B_block_size;

    stream->parallel_for(sycl::range<1>(B_অঙ্ক_work), [=](sycl::id<1> id) {
        ধ্রুব int64_t t = id[0];
        যদি (t >= B_total) প্রত্যাগম;

        ধ্রুব int64_t row = t % k;
        ধ্রুব int64_t col = t / k;
        ধ্রুব বর্ণ * src_ptr = src0_base + row * src0_packed_nb0 + col * src0_packed_nb1;
        ভিন্ন v;
        যদি (src0_is_f32) {
            v = *(ধ্রুব ভিন্ন *) src_ptr;
        } অন্যথা {
            v = sycl::vec<sycl::half, 1>(*(ধ্রুব sycl::half *) src_ptr).convert<ভিন্ন, sycl::rounding_mode::automatic>()[0];
        }
        B_packed[row + col * (int64_t)k] = v;
    });

    // GEMM: C = A^T * B where A is (k x m), B is (k x n), C is (m x n)
    ধ্রুব ভিন্ন alpha = 1.0f;
    ধ্রুব ভিন্ন beta  = 0.0f;
    ধ্রুব পূর্ণাঙ্ক lda = k;
    ধ্রুব পূর্ণাঙ্ক ldb = k;
    ধ্রুব পূর্ণাঙ্ক ldc = m;

    SYCL_CHECK(CHECK_TRY_ERROR(oneapi::mkl::blas::column_major::gemm(
        *stream, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans,
        m, n_gemm, k,
        dpct::get_মান(&alpha, *stream),
        (ধ্রুব ভিন্ন *) A_packed, lda,
        (ধ্রুব ভিন্ন *) B_packed, ldb,
        dpct::get_মান(&beta, *stream),
        (ভিন্ন *) dst_mat.data, ldc)));

    ধ্রুব ভিন্ন * gemm_data = (ধ্রুব ভিন্ন *) dst_mat.data;
    ভিন্ন * dst_data = (ভিন্ন *) dst->data;

    ggml_sycl_conv3d_write_output(dst, gemm_data, dst_data, patch_total, oc,
                                  dst->ne[0], dst->ne[1], dst->ne[2], stream);
}
