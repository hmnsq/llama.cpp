#ଅନ୍ତର୍ଭାଵ "solve_tri.hpp"
#ଅନ୍ତର୍ଭାଵ "common.hpp"
#ଅନ୍ତର୍ଭାଵ <oneapi/mkl/blas.hpp>

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_template, ପୂର୍ଣ୍ଣସଂଖ୍ୟା k_template>
ସ୍ଥିର ଶୂନ୍ୟ solve_tri_f32_fast(ଅଚଳ ଦଶମିକ * __restrict__ A,
                               ଅଚଳ ଦଶମିକ * __restrict__ B,
                               ଦଶମିକ * __restrict__ X,
                               ଅଚଳ int64_t ne02, [[maybe_unused]] ଅଚଳ int64_t ne03,
                               ଅଚଳ int64_t nb02, ଅଚଳ int64_t nb03,
                               ଅଚଳ int64_t nb12, ଅଚଳ int64_t nb13,
                               ଅଚଳ int64_t nb2,  ଅଚଳ int64_t nb3,
                               ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_arg, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା k_arg,
                               ଅଚଳ sycl::nd_item<2> & item, ଦଶମିକ * sA) {

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n = n_template == 0 ? n_arg : n_template;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = k_template == 0 ? k_arg : k_template;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା batch_idx = item.get_group(1);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା lane      = item.get_local_id(1) % WARP_SIZE;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା col_idx   = item.get_local_id(0);

    ଯଦି (col_idx >= k) {
        ଫେରନ୍ତୁ;
    }

    ଅଚଳ int64_t i03 = batch_idx / ne02;
    ଅଚଳ int64_t i02 = batch_idx % ne02;

    ଅଚଳ ଦଶମିକ * A_batch = (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) A + i02 * nb02 + i03 * nb03);
    ଅଚଳ ଦଶମିକ * B_batch = (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) B + i02 * nb12 + i03 * nb13);
    ଦଶମିକ *       X_batch = (ଦଶମିକ *)       ((ଅକ୍ଷର *)       X + i02 * nb2  + i03 * nb3);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା offset = item.get_local_id(1) + item.get_local_id(0) * item.get_local_range(1);

#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n * n; i += k * WARP_SIZE) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i0 = i + offset;
        ଯଦି (i0 < n * n) {
            sA[i0] = A_batch[i0];
        }
    }

    item.barrier(sycl::access::fence_space::local_space);

    ଦଶମିକ x_ତଳ  = (lane < n) ? B_batch[lane * k + col_idx] : 0.0f;
    ଦଶମିକ x_ଉପର = (WARP_SIZE + lane < n) ? B_batch[(WARP_SIZE + lane) * k + col_idx] : 0.0f;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା half      = WARP_SIZE;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows_ତଳ = (n < half) ? n : half;

#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା row = 0; row < nrows_ତଳ; ++row) {
        ଦଶମିକ sum = 0.0f;
        ଯଦି (lane < row) {
            sum += sA[row * n + lane] * x_ତଳ;
        }
        sum = warp_reduce_sum<WARP_SIZE>(sum);
        ଯଦି (lane == row) {
            x_ତଳ = (x_ତଳ - sum) / sA[row * n + row];
        }
    }

#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା row = half; row < n; ++row) {
        ଦଶମିକ     sum = sA[row * n + lane] * x_ତଳ;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା j   = half + lane;
        ଯଦି (j < row) {
            sum += sA[row * n + j] * x_ଉପର;
        }
        sum = warp_reduce_sum<WARP_SIZE>(sum);
        ଯଦି (lane == row - half) {
            x_ଉପର = (x_ଉପର - sum) / sA[row * n + row];
        }
    }

#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା rr = 0; rr < 2; ++rr) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା row = rr * WARP_SIZE + lane;
        ଯଦି (row < n) {
            ଅଚଳ ଦଶମିକ val            = (row < half) ? x_ତଳ : x_ଉପର;
            X_batch[row * k + col_idx] = val;
        }
    }
}

ସ୍ଥିର ଶୂନ୍ୟ solve_tri_f32_mkl(dpct::queue_ptr stream,
                               ଅଚଳ ଦଶମିକ * A, ଦଶମିକ * X,
                               ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ପୂର୍ଣ୍ଣସଂଖ୍ୟା k,
                               int64_t ne02, [[maybe_unused]] int64_t ne03,
                               int64_t nb02, [[maybe_unused]] int64_t nb03,
                               int64_t nb2,  [[maybe_unused]] int64_t nb3) {
    ଅଚଳ ଦଶମିକ alpha = 1.0f;
    ଅଚଳ int64_t total_batches = ne02 * ne03;
    ଯଦି (total_batches == 0) {
        ଫେରନ୍ତୁ;
    }

    ଅଚଳ int64_t stride_a = nb02 / ଆକାର(ଦଶମିକ);
    ଅଚଳ int64_t stride_x = nb2 / ଆକାର(ଦଶମିକ);

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

ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ ggml_sycl_op_solve_tri(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(src0->type == GGML_TYPE_F32);

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n    = src0->ne[0];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା k    = src1->ne[0];
    ଅଚଳ int64_t ne02 = src0->ne[2];
    ଅଚଳ int64_t ne03 = src0->ne[3];

    GGML_ASSERT(n <= SYCL_SOLVE_TRI_MAX_N && k <= SYCL_SOLVE_TRI_MAX_K);

    ଅଚଳ ଦଶମିକ * A_d = static_cast<ଅଚଳ ଦଶମିକ *>(src0->data);
    ଅଚଳ ଦଶମିକ * B_d = static_cast<ଅଚଳ ଦଶମିକ *>(src1->data);
    ଦଶମିକ * X_d       = static_cast<ଦଶମିକ *>(dst->data);

    ଯଦି (X_d != B_d) {
        ଅଚଳ int64_t total_elements = (int64_t)n * k * ne02 * ne03;
        stream->memcpy(X_d, B_d, total_elements * ଆକାର(ଦଶମିକ));
    }

    ଅଚଳ int64_t nb02 = src0->nb[2];
    ଅଚଳ int64_t nb03 = src0->nb[3];
    ଅଚଳ int64_t nb12 = src1->nb[2];
    ଅଚଳ int64_t nb13 = src1->nb[3];
    ଅଚଳ int64_t nb2  = dst->nb[2];
    ଅଚଳ int64_t nb3  = dst->nb[3];

    ଅଚଳ int64_t total_batches = ne02 * ne03;

    ଯଦି (n <= 2 * WARP_SIZE && k <= 32) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା smem_size = 2 * WARP_SIZE * 2 * WARP_SIZE;
        ଅଚଳ sycl::range<2> grid(1, total_batches);
        ଅଚଳ sycl::range<2> block(k, WARP_SIZE);
        stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<ଦଶମିକ, 1> smem_acc(sycl::range<1>(smem_size), cgh);
            cgh.parallel_for(
                sycl::nd_range<2>(grid * block, block),
                [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    solve_tri_f32_fast<0, 0>(A_d, B_d, X_d, ne02, ne03,
                                              nb02, nb03, nb12, nb13, nb2, nb3,
                                              n, k, item, get_pointer(smem_acc));
                });
        });
    } ଅନ୍ୟଥା {
        solve_tri_f32_mkl(stream, A_d, X_d, n, k, ne02, ne03, nb02, nb03, nb2, nb3);
    }
}

ଶୂନ୍ୟ ggml_sycl_solve_tri(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_solve_tri(ctx, dst);
}
