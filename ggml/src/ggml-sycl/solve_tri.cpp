#अन्तर्भाव "solve_tri.hpp"
#अन्तर्भाव "common.hpp"
#अन्तर्भाव <oneapi/mkl/blas.hpp>

ढाँचा <पूर्णांक n_template, पूर्णांक k_template>
स्थिर शून्य solve_tri_f32_fast(स्थिरांक दशांश * __restrict__ A,
                               स्थिरांक दशांश * __restrict__ B,
                               दशांश * __restrict__ X,
                               स्थिरांक int64_t ne02, [[maybe_unused]] स्थिरांक int64_t ne03,
                               स्थिरांक int64_t nb02, स्थिरांक int64_t nb03,
                               स्थिरांक int64_t nb12, स्थिरांक int64_t nb13,
                               स्थिरांक int64_t nb2,  स्थिरांक int64_t nb3,
                               स्थिरांक पूर्णांक n_arg, स्थिरांक पूर्णांक k_arg,
                               स्थिरांक sycl::nd_item<2> & item, दशांश * sA) {

    स्थिरांक पूर्णांक n = n_template == 0 ? n_arg : n_template;
    स्थिरांक पूर्णांक k = k_template == 0 ? k_arg : k_template;

    स्थिरांक पूर्णांक batch_idx = item.get_group(1);
    स्थिरांक पूर्णांक lane      = item.get_local_id(1) % WARP_SIZE;
    स्थिरांक पूर्णांक col_idx   = item.get_local_id(0);

    जर (col_idx >= k) {
        परत;
    }

    स्थिरांक int64_t i03 = batch_idx / ne02;
    स्थिरांक int64_t i02 = batch_idx % ne02;

    स्थिरांक दशांश * A_batch = (स्थिरांक दशांश *) ((स्थिरांक अक्षर *) A + i02 * nb02 + i03 * nb03);
    स्थिरांक दशांश * B_batch = (स्थिरांक दशांश *) ((स्थिरांक अक्षर *) B + i02 * nb12 + i03 * nb13);
    दशांश *       X_batch = (दशांश *)       ((अक्षर *)       X + i02 * nb2  + i03 * nb3);

    स्थिरांक पूर्णांक offset = item.get_local_id(1) + item.get_local_id(0) * item.get_local_range(1);

#pragma unroll
    साठी (पूर्णांक i = 0; i < n * n; i += k * WARP_SIZE) {
        स्थिरांक पूर्णांक i0 = i + offset;
        जर (i0 < n * n) {
            sA[i0] = A_batch[i0];
        }
    }

    item.barrier(sycl::access::fence_space::local_space);

    दशांश x_खालचा  = (lane < n) ? B_batch[lane * k + col_idx] : 0.0f;
    दशांश x_वरचा = (WARP_SIZE + lane < n) ? B_batch[(WARP_SIZE + lane) * k + col_idx] : 0.0f;

    स्थिरांक पूर्णांक half      = WARP_SIZE;
    स्थिरांक पूर्णांक nrows_खालचा = (n < half) ? n : half;

#pragma unroll
    साठी (पूर्णांक row = 0; row < nrows_खालचा; ++row) {
        दशांश sum = 0.0f;
        जर (lane < row) {
            sum += sA[row * n + lane] * x_खालचा;
        }
        sum = warp_reduce_sum<WARP_SIZE>(sum);
        जर (lane == row) {
            x_खालचा = (x_खालचा - sum) / sA[row * n + row];
        }
    }

#pragma unroll
    साठी (पूर्णांक row = half; row < n; ++row) {
        दशांश     sum = sA[row * n + lane] * x_खालचा;
        स्थिरांक पूर्णांक j   = half + lane;
        जर (j < row) {
            sum += sA[row * n + j] * x_वरचा;
        }
        sum = warp_reduce_sum<WARP_SIZE>(sum);
        जर (lane == row - half) {
            x_वरचा = (x_वरचा - sum) / sA[row * n + row];
        }
    }

#pragma unroll
    साठी (पूर्णांक rr = 0; rr < 2; ++rr) {
        स्थिरांक पूर्णांक row = rr * WARP_SIZE + lane;
        जर (row < n) {
            स्थिरांक दशांश val            = (row < half) ? x_खालचा : x_वरचा;
            X_batch[row * k + col_idx] = val;
        }
    }
}

स्थिर शून्य solve_tri_f32_mkl(dpct::queue_ptr stream,
                               स्थिरांक दशांश * A, दशांश * X,
                               पूर्णांक n, पूर्णांक k,
                               int64_t ne02, [[maybe_unused]] int64_t ne03,
                               int64_t nb02, [[maybe_unused]] int64_t nb03,
                               int64_t nb2,  [[maybe_unused]] int64_t nb3) {
    स्थिरांक दशांश alpha = 1.0f;
    स्थिरांक int64_t total_batches = ne02 * ne03;
    जर (total_batches == 0) {
        परत;
    }

    स्थिरांक int64_t stride_a = nb02 / आकार(दशांश);
    स्थिरांक int64_t stride_x = nb2 / आकार(दशांश);

    oneapi::mkl::blas::trsm_batch(
        *stream,
        oneapi::mkl::side::right,
        oneapi::mkl::uplo::upper,
        oneapi::mkl::transpose::nontrans,
        oneapi::mkl::diag::nonunit,
        k, n, alpha,
        A, n, stride_a,
        X, k, stride_x,
        total_batches);
}

ओळीत शून्य ggml_sycl_op_solve_tri(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    स्थिरांक ggml_tensor * src0 = dst->src[0];
    स्थिरांक ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(src0->type == GGML_TYPE_F32);

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    स्थिरांक पूर्णांक n    = src0->ne[0];
    स्थिरांक पूर्णांक k    = src1->ne[0];
    स्थिरांक int64_t ne02 = src0->ne[2];
    स्थिरांक int64_t ne03 = src0->ne[3];

    GGML_ASSERT(n <= SYCL_SOLVE_TRI_MAX_N && k <= SYCL_SOLVE_TRI_MAX_K);

    स्थिरांक दशांश * A_d = static_cast<स्थिरांक दशांश *>(src0->data);
    स्थिरांक दशांश * B_d = static_cast<स्थिरांक दशांश *>(src1->data);
    दशांश * X_d       = static_cast<दशांश *>(dst->data);

    जर (X_d != B_d) {
        स्थिरांक int64_t total_elements = (int64_t)n * k * ne02 * ne03;
        stream->स_नकल(X_d, B_d, total_elements * आकार(दशांश));
    }

    स्थिरांक int64_t nb02 = src0->nb[2];
    स्थिरांक int64_t nb03 = src0->nb[3];
    स्थिरांक int64_t nb12 = src1->nb[2];
    स्थिरांक int64_t nb13 = src1->nb[3];
    स्थिरांक int64_t nb2  = dst->nb[2];
    स्थिरांक int64_t nb3  = dst->nb[3];

    स्थिरांक int64_t total_batches = ne02 * ne03;

    जर (n <= 2 * WARP_SIZE && k <= 32) {
        स्थिरांक पूर्णांक smem_size = 2 * WARP_SIZE * 2 * WARP_SIZE;
        स्थिरांक sycl::range<2> grid(1, total_batches);
        स्थिरांक sycl::range<2> block(k, WARP_SIZE);
        stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<दशांश, 1> smem_acc(sycl::range<1>(smem_size), cgh);
            cgh.parallel_for(
                sycl::nd_range<2>(grid * block, block),
                [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    solve_tri_f32_fast<0, 0>(A_d, B_d, X_d, ne02, ne03,
                                              nb02, nb03, nb12, nb13, nb2, nb3,
                                              n, k, item, get_pointer(smem_acc));
                });
        });
    } अन्यथा {
        solve_tri_f32_mkl(stream, A_d, X_d, n, k, ne02, ne03, nb02, nb03, nb2, nb3);
    }
}

शून्य ggml_sycl_solve_tri(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_solve_tri(ctx, dst);
}
