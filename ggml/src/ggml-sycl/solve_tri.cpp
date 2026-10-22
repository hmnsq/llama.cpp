#अन्तर्भाव "solve_tri.hpp"
#अन्तर्भाव "common.hpp"
#अन्तर्भाव <oneapi/mkl/blas.hpp>

ढाँचा <पूर्णाङ्क n_template, पूर्णाङ्क k_template>
स्थिर शून्य solve_tri_f32_fast(अचर भिन्न * __restrict__ A,
                               अचर भिन्न * __restrict__ B,
                               भिन्न * __restrict__ X,
                               अचर int64_t ne02, [[maybe_unused]] अचर int64_t ne03,
                               अचर int64_t nb02, अचर int64_t nb03,
                               अचर int64_t nb12, अचर int64_t nb13,
                               अचर int64_t nb2,  अचर int64_t nb3,
                               अचर पूर्णाङ्क n_arg, अचर पूर्णाङ्क k_arg,
                               अचर sycl::nd_item<2> & item, भिन्न * sA) {

    अचर पूर्णाङ्क n = n_template == 0 ? n_arg : n_template;
    अचर पूर्णाङ्क k = k_template == 0 ? k_arg : k_template;

    अचर पूर्णाङ्क batch_idx = item.get_group(1);
    अचर पूर्णाङ्क lane      = item.get_local_id(1) % WARP_SIZE;
    अचर पूर्णाङ्क col_idx   = item.get_local_id(0);

    यदि (col_idx >= k) {
        फर्काऊ;
    }

    अचर int64_t i03 = batch_idx / ne02;
    अचर int64_t i02 = batch_idx % ne02;

    अचर भिन्न * A_batch = (अचर भिन्न *) ((अचर वर्ण *) A + i02 * nb02 + i03 * nb03);
    अचर भिन्न * B_batch = (अचर भिन्न *) ((अचर वर्ण *) B + i02 * nb12 + i03 * nb13);
    भिन्न *       X_batch = (भिन्न *)       ((वर्ण *)       X + i02 * nb2  + i03 * nb3);

    अचर पूर्णाङ्क offset = item.get_local_id(1) + item.get_local_id(0) * item.get_local_range(1);

#pragma unroll
    लागि (पूर्णाङ्क i = 0; i < n * n; i += k * WARP_SIZE) {
        अचर पूर्णाङ्क i0 = i + offset;
        यदि (i0 < n * n) {
            sA[i0] = A_batch[i0];
        }
    }

    item.barrier(sycl::access::fence_space::local_space);

    भिन्न x_तल्लो  = (lane < n) ? B_batch[lane * k + col_idx] : 0.0f;
    भिन्न x_माथिल्लो = (WARP_SIZE + lane < n) ? B_batch[(WARP_SIZE + lane) * k + col_idx] : 0.0f;

    अचर पूर्णाङ्क half      = WARP_SIZE;
    अचर पूर्णाङ्क nrows_तल्लो = (n < half) ? n : half;

#pragma unroll
    लागि (पूर्णाङ्क row = 0; row < nrows_तल्लो; ++row) {
        भिन्न sum = 0.0f;
        यदि (lane < row) {
            sum += sA[row * n + lane] * x_तल्लो;
        }
        sum = warp_reduce_sum<WARP_SIZE>(sum);
        यदि (lane == row) {
            x_तल्लो = (x_तल्लो - sum) / sA[row * n + row];
        }
    }

#pragma unroll
    लागि (पूर्णाङ्क row = half; row < n; ++row) {
        भिन्न     sum = sA[row * n + lane] * x_तल्लो;
        अचर पूर्णाङ्क j   = half + lane;
        यदि (j < row) {
            sum += sA[row * n + j] * x_माथिल्लो;
        }
        sum = warp_reduce_sum<WARP_SIZE>(sum);
        यदि (lane == row - half) {
            x_माथिल्लो = (x_माथिल्लो - sum) / sA[row * n + row];
        }
    }

#pragma unroll
    लागि (पूर्णाङ्क rr = 0; rr < 2; ++rr) {
        अचर पूर्णाङ्क row = rr * WARP_SIZE + lane;
        यदि (row < n) {
            अचर भिन्न val            = (row < half) ? x_तल्लो : x_माथिल्लो;
            X_batch[row * k + col_idx] = val;
        }
    }
}

स्थिर शून्य solve_tri_f32_mkl(dpct::queue_ptr stream,
                               अचर भिन्न * A, भिन्न * X,
                               पूर्णाङ्क n, पूर्णाङ्क k,
                               int64_t ne02, [[maybe_unused]] int64_t ne03,
                               int64_t nb02, [[maybe_unused]] int64_t nb03,
                               int64_t nb2,  [[maybe_unused]] int64_t nb3) {
    अचर भिन्न alpha = 1.0f;
    अचर int64_t total_batches = ne02 * ne03;
    यदि (total_batches == 0) {
        फर्काऊ;
    }

    अचर int64_t stride_a = nb02 / परिमाण(भिन्न);
    अचर int64_t stride_x = nb2 / परिमाण(भिन्न);

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

अन्तःस्थ शून्य ggml_sycl_op_solve_tri(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    अचर ggml_tensor * src0 = dst->src[0];
    अचर ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(src0->type == GGML_TYPE_F32);

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    अचर पूर्णाङ्क n    = src0->ne[0];
    अचर पूर्णाङ्क k    = src1->ne[0];
    अचर int64_t ne02 = src0->ne[2];
    अचर int64_t ne03 = src0->ne[3];

    GGML_ASSERT(n <= SYCL_SOLVE_TRI_MAX_N && k <= SYCL_SOLVE_TRI_MAX_K);

    अचर भिन्न * A_d = static_cast<अचर भिन्न *>(src0->data);
    अचर भिन्न * B_d = static_cast<अचर भिन्न *>(src1->data);
    भिन्न * X_d       = static_cast<भिन्न *>(dst->data);

    यदि (X_d != B_d) {
        अचर int64_t total_elements = (int64_t)n * k * ne02 * ne03;
        stream->स_नकल(X_d, B_d, total_elements * परिमाण(भिन्न));
    }

    अचर int64_t nb02 = src0->nb[2];
    अचर int64_t nb03 = src0->nb[3];
    अचर int64_t nb12 = src1->nb[2];
    अचर int64_t nb13 = src1->nb[3];
    अचर int64_t nb2  = dst->nb[2];
    अचर int64_t nb3  = dst->nb[3];

    अचर int64_t total_batches = ne02 * ne03;

    यदि (n <= 2 * WARP_SIZE && k <= 32) {
        अचर पूर्णाङ्क smem_size = 2 * WARP_SIZE * 2 * WARP_SIZE;
        अचर sycl::range<2> grid(1, total_batches);
        अचर sycl::range<2> block(k, WARP_SIZE);
        stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<भिन्न, 1> smem_acc(sycl::range<1>(smem_size), cgh);
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
