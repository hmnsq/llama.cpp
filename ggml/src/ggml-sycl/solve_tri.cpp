#अन्तर्भाव "solve_tri.hpp"
#अन्तर्भाव "common.hpp"
#अन्तर्भाव <oneapi/mkl/blas.hpp>

template <पूर्णाङ्क n_template, पूर्णाङ्क k_template>
स्थिर शून्य solve_tri_f32_fast(ध्रुव भिन्न * __restrict__ A,
                               ध्रुव भिन्न * __restrict__ B,
                               भिन्न * __restrict__ X,
                               ध्रुव int64_t ne02, [[maybe_unused]] ध्रुव int64_t ne03,
                               ध्रुव int64_t nb02, ध्रुव int64_t nb03,
                               ध्रुव int64_t nb12, ध्रुव int64_t nb13,
                               ध्रुव int64_t nb2,  ध्रुव int64_t nb3,
                               ध्रुव पूर्णाङ्क n_arg, ध्रुव पूर्णाङ्क k_arg,
                               ध्रुव sycl::nd_item<2> & item, भिन्न * sA) {

    ध्रुव पूर्णाङ्क n = n_template == 0 ? n_arg : n_template;
    ध्रुव पूर्णाङ्क k = k_template == 0 ? k_arg : k_template;

    ध्रुव पूर्णाङ्क batch_idx = item.get_group(1);
    ध्रुव पूर्णाङ्क lane      = item.get_local_id(1) % WARP_SIZE;
    ध्रुव पूर्णाङ्क col_idx   = item.get_local_id(0);

    सचे (col_idx >= k) {
        पटिनिवत्त;
    }

    ध्रुव int64_t i03 = batch_idx / ne02;
    ध्रुव int64_t i02 = batch_idx % ne02;

    ध्रुव भिन्न * A_batch = (ध्रुव भिन्न *) ((ध्रुव वर्ण *) A + i02 * nb02 + i03 * nb03);
    ध्रुव भिन्न * B_batch = (ध्रुव भिन्न *) ((ध्रुव वर्ण *) B + i02 * nb12 + i03 * nb13);
    भिन्न *       X_batch = (भिन्न *)       ((वर्ण *)       X + i02 * nb2  + i03 * nb3);

    ध्रुव पूर्णाङ्क offset = item.get_local_id(1) + item.get_local_id(0) * item.get_local_range(1);

#pragma unroll
    अत्थाय (पूर्णाङ्क i = 0; i < n * n; i += k * WARP_SIZE) {
        ध्रुव पूर्णाङ्क i0 = i + offset;
        सचे (i0 < n * n) {
            sA[i0] = A_batch[i0];
        }
    }

    item.barrier(sycl::access::fence_space::local_space);

    भिन्न x_नीचः  = (lane < n) ? B_batch[lane * k + col_idx] : 0.0f;
    भिन्न x_उच्चः = (WARP_SIZE + lane < n) ? B_batch[(WARP_SIZE + lane) * k + col_idx] : 0.0f;

    ध्रुव पूर्णाङ्क half      = WARP_SIZE;
    ध्रुव पूर्णाङ्क nrows_नीचः = (n < half) ? n : half;

#pragma unroll
    अत्थाय (पूर्णाङ्क row = 0; row < nrows_नीचः; ++row) {
        भिन्न sum = 0.0f;
        सचे (lane < row) {
            sum += sA[row * n + lane] * x_नीचः;
        }
        sum = warp_reduce_sum<WARP_SIZE>(sum);
        सचे (lane == row) {
            x_नीचः = (x_नीचः - sum) / sA[row * n + row];
        }
    }

#pragma unroll
    अत्थाय (पूर्णाङ्क row = half; row < n; ++row) {
        भिन्न     sum = sA[row * n + lane] * x_नीचः;
        ध्रुव पूर्णाङ्क j   = half + lane;
        सचे (j < row) {
            sum += sA[row * n + j] * x_उच्चः;
        }
        sum = warp_reduce_sum<WARP_SIZE>(sum);
        सचे (lane == row - half) {
            x_उच्चः = (x_उच्चः - sum) / sA[row * n + row];
        }
    }

#pragma unroll
    अत्थाय (पूर्णाङ्क rr = 0; rr < 2; ++rr) {
        ध्रुव पूर्णाङ्क row = rr * WARP_SIZE + lane;
        सचे (row < n) {
            ध्रुव भिन्न val            = (row < half) ? x_नीचः : x_उच्चः;
            X_batch[row * k + col_idx] = val;
        }
    }
}

स्थिर शून्य solve_tri_f32_mkl(dpct::queue_ptr stream,
                               ध्रुव भिन्न * A, भिन्न * X,
                               पूर्णाङ्क n, पूर्णाङ्क k,
                               int64_t ne02, [[maybe_unused]] int64_t ne03,
                               int64_t nb02, [[maybe_unused]] int64_t nb03,
                               int64_t nb2,  [[maybe_unused]] int64_t nb3) {
    ध्रुव भिन्न alpha = 1.0f;
    ध्रुव int64_t total_batches = ne02 * ne03;
    सचे (total_batches == 0) {
        पटिनिवत्त;
    }

    ध्रुव int64_t stride_a = nb02 / परिमाण(भिन्न);
    ध्रुव int64_t stride_x = nb2 / परिमाण(भिन्न);

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
    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(src0->type == GGML_TYPE_F32);

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ध्रुव पूर्णाङ्क n    = src0->ne[0];
    ध्रुव पूर्णाङ्क k    = src1->ne[0];
    ध्रुव int64_t ne02 = src0->ne[2];
    ध्रुव int64_t ne03 = src0->ne[3];

    GGML_ASSERT(n <= SYCL_SOLVE_TRI_MAX_N && k <= SYCL_SOLVE_TRI_MAX_K);

    ध्रुव भिन्न * A_d = static_cast<ध्रुव भिन्न *>(src0->data);
    ध्रुव भिन्न * B_d = static_cast<ध्रुव भिन्न *>(src1->data);
    भिन्न * X_d       = static_cast<भिन्न *>(dst->data);

    सचे (X_d != B_d) {
        ध्रुव int64_t total_elements = (int64_t)n * k * ne02 * ne03;
        stream->memcpy(X_d, B_d, total_elements * परिमाण(भिन्न));
    }

    ध्रुव int64_t nb02 = src0->nb[2];
    ध्रुव int64_t nb03 = src0->nb[3];
    ध्रुव int64_t nb12 = src1->nb[2];
    ध्रुव int64_t nb13 = src1->nb[3];
    ध्रुव int64_t nb2  = dst->nb[2];
    ध्रुव int64_t nb3  = dst->nb[3];

    ध्रुव int64_t total_batches = ne02 * ne03;

    सचे (n <= 2 * WARP_SIZE && k <= 32) {
        ध्रुव पूर्णाङ्क smem_size = 2 * WARP_SIZE * 2 * WARP_SIZE;
        ध्रुव sycl::range<2> grid(1, total_batches);
        ध्रुव sycl::range<2> block(k, WARP_SIZE);
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
    } अञ्ञथा {
        solve_tri_f32_mkl(stream, A_d, X_d, n, k, ne02, ne03, nb02, nb03, nb2, nb3);
    }
}

शून्य ggml_sycl_solve_tri(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_solve_tri(ctx, dst);
}
