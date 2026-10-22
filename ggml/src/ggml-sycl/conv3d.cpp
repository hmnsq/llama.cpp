#അന്തര്ഭാവ "conv3d.hpp"

സ്ഥിതം വരിയിൽ int64_t ggml_sycl_conv3d_calc_patch_total(സ്ഥിരം ggml_tensor * dst, int32_t n) {
    തിരികെ (int64_t) n * dst->ne[0] * dst->ne[1] * dst->ne[2];
}

സ്ഥിതം വരിയിൽ int64_t ggml_sycl_conv3d_calc_knl_n_total(സ്ഥിരം ggml_tensor * src0, int32_t c) {
    തിരികെ (int64_t) src0->ne[0] * src0->ne[1] * src0->ne[2] * c;
}

സ്ഥിതം വരിയിൽ ശൂന്യം ggml_sycl_conv3d_write_output(
        സ്ഥിരം ggml_tensor * dst,
        സ്ഥിരം ദശാംശം * src, ദശാംശം * dst_data,
        int64_t patch_total, int64_t oc,
        int64_t dst_w, int64_t dst_h, int64_t dst_d,
        dpct::queue_ptr stream) {
    സ്ഥിരം int64_t dst_nb0 = dst->nb[0];
    സ്ഥിരം int64_t dst_nb1 = dst->nb[1];
    സ്ഥിരം int64_t dst_nb2 = dst->nb[2];
    സ്ഥിരം int64_t dst_nb3 = dst->nb[3];
    സ്ഥിരം int64_t total = patch_total * oc;
    സ്ഥിരം int64_t block_size = 256;
    സ്ഥിരം int64_t അക്കം_work_ഇനങ്ങൾ = ((total + block_size - 1) / block_size) * block_size;

    stream->parallel_for(sycl::range<1>(അക്കം_work_ഇനങ്ങൾ), [=](sycl::id<1> id) {
        സ്ഥിരം int64_t i = id[0];
        എങ്കിൽ (i >= total) {
            തിരികെ;
        }

        സ്ഥിരം int64_t patch_idx = i / oc;
        സ്ഥിരം int64_t out_ch = i % oc;
        സ്ഥിരം int64_t p_in_batch = patch_idx % (dst_w * dst_h * dst_d);
        സ്ഥിരം int64_t batch_idx = patch_idx / (dst_w * dst_h * dst_d);
        സ്ഥിരം int64_t dst_z = p_in_batch / (dst_w * dst_h);
        സ്ഥിരം int64_t dst_y = (p_in_batch % (dst_w * dst_h)) / dst_w;
        സ്ഥിരം int64_t dst_x = p_in_batch % dst_w;
        സ്ഥിരം int64_t ocn_idx = batch_idx * oc + out_ch;

        സ്ഥിരം int64_t dst_offset = dst_x * dst_nb0 + dst_y * dst_nb1 + dst_z * dst_nb2 + ocn_idx * dst_nb3;
        // `src` is a column-major (m x n) GEMM output where m == patch_total, n == oc.
        // GEMM stores element (row, col) at index `row + col*m`, so compute index accordingly.
        സ്ഥിരം int64_t src_index = patch_idx + out_ch * patch_total;
        സ്ഥിരം ദശാംശം value = src[src_index];
        *(ദശാംശം *)((അക്ഷരം *)dst_data + dst_offset) = value;
    });
}

ശൂന്യം ggml_sycl_op_conv_3d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));

    സ്ഥിരം int32_t * opts = (സ്ഥിരം int32_t *) dst->op_params;
    സ്ഥിരം int32_t s0 = opts[0];
    സ്ഥിരം int32_t s1 = opts[1];
    സ്ഥിരം int32_t s2 = opts[2];
    സ്ഥിരം int32_t p0 = opts[3];
    സ്ഥിരം int32_t p1 = opts[4];
    സ്ഥിരം int32_t p2 = opts[5];
    സ്ഥിരം int32_t d0 = opts[6];
    സ്ഥിരം int32_t d1 = opts[7];
    സ്ഥിരം int32_t d2 = opts[8];
    സ്ഥിരം int32_t c  = opts[9];
    സ്ഥിരം int32_t n  = opts[10];
    സ്ഥിരം int32_t oc = opts[11];

    സ്ഥിരം int64_t knl_w = src0->ne[0];
    സ്ഥിരം int64_t knl_h = src0->ne[1];
    സ്ഥിരം int64_t knl_d = src0->ne[2];

    സ്ഥിരം int64_t patch_total = ggml_sycl_conv3d_calc_patch_total(dst, n);
    സ്ഥിരം int64_t knl_n_total = ggml_sycl_conv3d_calc_knl_n_total(src0, c);

    സ്ഥിരം size_t kernel_type_size = ggml_element_size(src0);

    ggml_sycl_pool_alloc<ദശാംശം> gemm_output(ctx.pool());
    gemm_output.alloc((size_t) patch_total * oc);

    ggml_tensor dst_mat = {};
    dst_mat.type = GGML_TYPE_F32;
    dst_mat.ne[0] = patch_total;
    dst_mat.ne[1] = oc;
    dst_mat.ne[2] = 1;
    dst_mat.ne[3] = 1;
    dst_mat.nb[0] = വലുപ്പം(ദശാംശം);
    dst_mat.nb[1] = dst_mat.nb[0] * dst_mat.ne[0];
    dst_mat.nb[2] = dst_mat.nb[1];
    dst_mat.nb[3] = dst_mat.nb[2];
    dst_mat.data = gemm_output.get();
    dst_mat.buffer = dst->buffer;
    dst_mat.extra = dst->extra;

    dpct::queue_ptr stream = ctx.stream();

    // allocate packed arrays: A_packed (k x m), B_packed (k x n)
    ggml_sycl_pool_alloc<ദശാംശം> A_packed_alloc(ctx.pool());
    ggml_sycl_pool_alloc<ദശാംശം> B_packed_alloc(ctx.pool());
    A_packed_alloc.alloc((size_t) knl_n_total * patch_total);
    B_packed_alloc.alloc((size_t) knl_n_total * oc);

    ദശാംശം * A_packed = A_packed_alloc.get();
    ദശാംശം * B_packed = B_packed_alloc.get();

    സ്ഥിരം പൂർണ്ണസംഖ്യ m = (പൂർണ്ണസംഖ്യ) patch_total;
    സ്ഥിരം പൂർണ്ണസംഖ്യ n_gemm = (പൂർണ്ണസംഖ്യ) oc;
    സ്ഥിരം പൂർണ്ണസംഖ്യ k = (പൂർണ്ണസംഖ്യ) knl_n_total;

    // Combined kernel: im2col -> pack A, and pack B simultaneously
    സ്ഥിരം അക്ഷരം * src1_base = (സ്ഥിരം അക്ഷരം *) src1->data;
    സ്ഥിരം അക്ഷരം * src0_base = (സ്ഥിരം അക്ഷരം *) src0->data;
    സ്ഥിരം int64_t src1_nb0 = src1->nb[0];
    സ്ഥിരം int64_t src1_nb1 = src1->nb[1];
    സ്ഥിരം int64_t src1_nb2 = src1->nb[2];
    സ്ഥിരം int64_t src1_nb3 = src1->nb[3];
    സ്ഥിരം int64_t src1_w = src1->ne[0];
    സ്ഥിരം int64_t src1_h = src1->ne[1];
    സ്ഥിരം int64_t src1_d = src1->ne[2];

    സ്ഥിരം തർക്കം src0_is_f32 = (src0->type == GGML_TYPE_F32);

    // Compute correct strides for src0 as (knl_n_total, oc) matrix
    സ്ഥിരം int64_t src0_packed_nb0 = kernel_type_size;
    സ്ഥിരം int64_t src0_packed_nb1 = kernel_type_size * knl_n_total;

    സ്ഥിരം int64_t KW = knl_w;
    സ്ഥിരം int64_t KH = knl_h;
    സ്ഥിരം int64_t KD = knl_d;
    സ്ഥിരം int64_t PW = dst->ne[0];
    സ്ഥിരം int64_t PH = dst->ne[1];
    സ്ഥിരം int64_t PD = dst->ne[2];

    // Pack A (with inline im2col): for each (row, col) in k x m matrix
    സ്ഥിരം int64_t A_total = (int64_t)k * m;
    സ്ഥിരം int64_t A_block_size = 256;
    സ്ഥിരം int64_t A_അക്കം_work = ((A_total + A_block_size - 1) / A_block_size) * A_block_size;

    stream->parallel_for(sycl::range<1>(A_അക്കം_work), [=](sycl::id<1> id) {
        സ്ഥിരം int64_t t = id[0];
        എങ്കിൽ (t >= A_total) തിരികെ;

        സ്ഥിരം int64_t row = t % k;
        സ്ഥിരം int64_t col = t / k;

        // Inline im2col for this element
        സ്ഥിരം int64_t k_index = row;
        സ്ഥിരം int64_t patch_idx = col;

        സ്ഥിരം int64_t ic = k_index / (KD * KH * KW);
        സ്ഥിരം int64_t rem = k_index - ic * (KD * KH * KW);
        സ്ഥിരം int64_t kz = rem / (KH * KW);
        സ്ഥിരം int64_t rem2 = rem - kz * (KH * KW);
        സ്ഥിരം int64_t ky = rem2 / KW;
        സ്ഥിരം int64_t kx = rem2 % KW;

        സ്ഥിരം int64_t p_in_batch = patch_idx % (PW * PH * PD);
        സ്ഥിരം int64_t batch_idx = patch_idx / (PW * PH * PD);
        സ്ഥിരം int64_t dst_z = p_in_batch / (PW * PH);
        സ്ഥിരം int64_t dst_y = (p_in_batch % (PW * PH)) / PW;
        സ്ഥിരം int64_t dst_x = p_in_batch % PW;

        സ്ഥിരം int64_t sx = dst_x * s0 + kx * d0 - p0;
        സ്ഥിരം int64_t sy = dst_y * s1 + ky * d1 - p1;
        സ്ഥിരം int64_t sz = dst_z * s2 + kz * d2 - p2;

        ദശാംശം val = 0.0f;
        എങ്കിൽ (sx >= 0 && sx < src1_w && sy >= 0 && sy < src1_h && sz >= 0 && sz < src1_d) {
            സ്ഥിരം int64_t channel_idx = batch_idx * c + ic;
            സ്ഥിരം അക്ഷരം * ptr = src1_base + sx * src1_nb0 + sy * src1_nb1 + sz * src1_nb2 + channel_idx * src1_nb3;
            val = *(സ്ഥിരം ദശാംശം *) ptr;
        }
        A_packed[row + col * (int64_t)k] = val;
    });

    // Pack B: for each (row, col) in k x n_gemm matrix
    സ്ഥിരം int64_t B_total = (int64_t)k * n_gemm;
    സ്ഥിരം int64_t B_block_size = 256;
    സ്ഥിരം int64_t B_അക്കം_work = ((B_total + B_block_size - 1) / B_block_size) * B_block_size;

    stream->parallel_for(sycl::range<1>(B_അക്കം_work), [=](sycl::id<1> id) {
        സ്ഥിരം int64_t t = id[0];
        എങ്കിൽ (t >= B_total) തിരികെ;

        സ്ഥിരം int64_t row = t % k;
        സ്ഥിരം int64_t col = t / k;
        സ്ഥിരം അക്ഷരം * src_ptr = src0_base + row * src0_packed_nb0 + col * src0_packed_nb1;
        ദശാംശം v;
        എങ്കിൽ (src0_is_f32) {
            v = *(സ്ഥിരം ദശാംശം *) src_ptr;
        } അല്ലെങ്കിൽ {
            v = sycl::vec<sycl::half, 1>(*(സ്ഥിരം sycl::half *) src_ptr).convert<ദശാംശം, sycl::rounding_mode::automatic>()[0];
        }
        B_packed[row + col * (int64_t)k] = v;
    });

    // GEMM: C = A^T * B where A is (k x m), B is (k x n), C is (m x n)
    സ്ഥിരം ദശാംശം alpha = 1.0f;
    സ്ഥിരം ദശാംശം beta  = 0.0f;
    സ്ഥിരം പൂർണ്ണസംഖ്യ lda = k;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ldb = k;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ldc = m;

    SYCL_CHECK(CHECK_TRY_ERROR(oneapi::mkl::blas::column_major::gemm(
        *stream, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans,
        m, n_gemm, k,
        dpct::get_മൂല്യം(&alpha, *stream),
        (സ്ഥിരം ദശാംശം *) A_packed, lda,
        (സ്ഥിരം ദശാംശം *) B_packed, ldb,
        dpct::get_മൂല്യം(&beta, *stream),
        (ദശാംശം *) dst_mat.data, ldc)));

    സ്ഥിരം ദശാംശം * gemm_data = (സ്ഥിരം ദശാംശം *) dst_mat.data;
    ദശാംശം * dst_data = (ദശാംശം *) dst->data;

    ggml_sycl_conv3d_write_output(dst, gemm_data, dst_data, patch_total, oc,
                                  dst->ne[0], dst->ne[1], dst->ne[2], stream);
}
