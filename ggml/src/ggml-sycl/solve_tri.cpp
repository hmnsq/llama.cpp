#समावेश "solve_tri.hpp"
#समावेश "common.hpp"
#समावेश <oneapi/mkl/blas.hpp>

ढाँचा <पूर्णांक n_template, पूर्णांक k_template>
अटल व्योम solve_tri_f32_fast(स्थिर भग्न * __restrict__ A,
                               स्थिर भग्न * __restrict__ B,
                               भग्न * __restrict__ X,
                               स्थिर int64_t ne02, [[maybe_unused]] स्थिर int64_t ne03,
                               स्थिर int64_t nb02, स्थिर int64_t nb03,
                               स्थिर int64_t nb12, स्थिर int64_t nb13,
                               स्थिर int64_t nb2,  स्थिर int64_t nb3,
                               स्थिर पूर्णांक n_arg, स्थिर पूर्णांक k_arg,
                               स्थिर sycl::nd_item<2> & item, भग्न * sA) {

    स्थिर पूर्णांक n = n_template == 0 ? n_arg : n_template;
    स्थिर पूर्णांक k = k_template == 0 ? k_arg : k_template;

    स्थिर पूर्णांक batch_idx = item.get_group(1);
    स्थिर पूर्णांक lane      = item.get_local_id(1) % WARP_आकार;
    स्थिर पूर्णांक col_idx   = item.get_local_id(0);

    अगर (col_idx >= k) {
        वापस;
    }

    स्थिर int64_t i03 = batch_idx / ne02;
    स्थिर int64_t i02 = batch_idx % ne02;

    स्थिर भग्न * A_batch = (स्थिर भग्न *) ((स्थिर अक्षर *) A + i02 * nb02 + i03 * nb03);
    स्थिर भग्न * B_batch = (स्थिर भग्न *) ((स्थिर अक्षर *) B + i02 * nb12 + i03 * nb13);
    भग्न *       X_batch = (भग्न *)       ((अक्षर *)       X + i02 * nb2  + i03 * nb3);

    स्थिर पूर्णांक offset = item.get_local_id(1) + item.get_local_id(0) * item.get_local_range(1);

#आशय unroll
    क्रम (पूर्णांक i = 0; i < n * n; i += k * WARP_आकार) {
        स्थिर पूर्णांक i0 = i + offset;
        अगर (i0 < n * n) {
            sA[i0] = A_batch[i0];
        }
    }

    item.barrier(sycl::access::fence_space::local_space);

    भग्न x_निचला  = (lane < n) ? B_batch[lane * k + col_idx] : 0.0f;
    भग्न x_ऊपरी = (WARP_आकार + lane < n) ? B_batch[(WARP_आकार + lane) * k + col_idx] : 0.0f;

    स्थिर पूर्णांक half      = WARP_आकार;
    स्थिर पूर्णांक nrows_निचला = (n < half) ? n : half;

#आशय unroll
    क्रम (पूर्णांक row = 0; row < nrows_निचला; ++row) {
        भग्न योग_2 = 0.0f;
        अगर (lane < row) {
            योग_2 += sA[row * n + lane] * x_निचला;
        }
        योग_2 = warp_reduce_योग<WARP_आकार>(योग_2);
        अगर (lane == row) {
            x_निचला = (x_निचला - योग_2) / sA[row * n + row];
        }
    }

#आशय unroll
    क्रम (पूर्णांक row = half; row < n; ++row) {
        भग्न     योग_2 = sA[row * n + lane] * x_निचला;
        स्थिर पूर्णांक j   = half + lane;
        अगर (j < row) {
            योग_2 += sA[row * n + j] * x_ऊपरी;
        }
        योग_2 = warp_reduce_योग<WARP_आकार>(योग_2);
        अगर (lane == row - half) {
            x_ऊपरी = (x_ऊपरी - योग_2) / sA[row * n + row];
        }
    }

#आशय unroll
    क्रम (पूर्णांक rr = 0; rr < 2; ++rr) {
        स्थिर पूर्णांक row = rr * WARP_आकार + lane;
        अगर (row < n) {
            स्थिर भग्न val            = (row < half) ? x_निचला : x_ऊपरी;
            X_batch[row * k + col_idx] = val;
        }
    }
}

अटल व्योम solve_tri_f32_mkl(dpct::queue_ptr stream,
                               स्थिर भग्न * A, भग्न * X,
                               पूर्णांक n, पूर्णांक k,
                               int64_t ne02, [[maybe_unused]] int64_t ne03,
                               int64_t nb02, [[maybe_unused]] int64_t nb03,
                               int64_t nb2,  [[maybe_unused]] int64_t nb3) {
    स्थिर भग्न alpha = 1.0f;
    स्थिर int64_t योग_batches = ne02 * ne03;
    अगर (योग_batches == 0) {
        वापस;
    }

    स्थिर int64_t stride_a = nb02 / माप(भग्न);
    स्थिर int64_t stride_x = nb2 / माप(भग्न);

    oneapi::mkl::blas::trsm_batch(
        *stream,
        oneapi::mkl::side::दाँए,
        oneapi::mkl::uplo::upper,
        oneapi::mkl::transpose::nontrans,
        oneapi::mkl::diag::nonunit,
        k, n, alpha,
        A, n, stride_a,
        X, k, stride_x,
        योग_batches);
}

अंतरभूत व्योम ggml_sycl_op_solve_tri(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(src0->type == GGML_TYPE_F32);

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    स्थिर पूर्णांक n    = src0->ne[0];
    स्थिर पूर्णांक k    = src1->ne[0];
    स्थिर int64_t ne02 = src0->ne[2];
    स्थिर int64_t ne03 = src0->ne[3];

    GGML_ASSERT(n <= SYCL_SOLVE_TRI_MAX_N && k <= SYCL_SOLVE_TRI_MAX_K);

    स्थिर भग्न * A_d = static_cast<स्थिर भग्न *>(src0->तथ्य);
    स्थिर भग्न * B_d = static_cast<स्थिर भग्न *>(src1->तथ्य);
    भग्न * X_d       = static_cast<भग्न *>(dst->तथ्य);

    अगर (X_d != B_d) {
        स्थिर int64_t योग_elements = (int64_t)n * k * ne02 * ne03;
        stream->स_नकल(X_d, B_d, योग_elements * माप(भग्न));
    }

    स्थिर int64_t nb02 = src0->nb[2];
    स्थिर int64_t nb03 = src0->nb[3];
    स्थिर int64_t nb12 = src1->nb[2];
    स्थिर int64_t nb13 = src1->nb[3];
    स्थिर int64_t nb2  = dst->nb[2];
    स्थिर int64_t nb3  = dst->nb[3];

    स्थिर int64_t योग_batches = ne02 * ne03;

    अगर (n <= 2 * WARP_आकार && k <= 32) {
        स्थिर पूर्णांक smem_आकार = 2 * WARP_आकार * 2 * WARP_आकार;
        स्थिर sycl::range<2> grid(1, योग_batches);
        स्थिर sycl::range<2> block(k, WARP_आकार);
        stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<भग्न, 1> smem_acc(sycl::range<1>(smem_आकार), cgh);
            cgh.parallel_for(
                sycl::nd_range<2>(grid * block, block),
                [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_आकार(WARP_आकार)]] {
                    solve_tri_f32_fast<0, 0>(A_d, B_d, X_d, ne02, ne03,
                                              nb02, nb03, nb12, nb13, nb2, nb3,
                                              n, k, item, get_pointer(smem_acc));
                });
        });
    } अन्यथा {
        solve_tri_f32_mkl(stream, A_d, X_d, n, k, ne02, ne03, nb02, nb03, nb2, nb3);
    }
}

व्योम ggml_sycl_solve_tri(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_solve_tri(ctx, dst);
}
