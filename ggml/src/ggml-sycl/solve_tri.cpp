#অন্তর্ভাৱ "solve_tri.hpp"
#অন্তর্ভাৱ "common.hpp"
#অন্তর্ভাৱ <oneapi/mkl/blas.hpp>

template <পূর্ণাঙ্ক n_template, পূর্ণাঙ্ক k_template>
স্থির শূন্য solve_tri_f32_fast(ধ্রুৱ ভিন্ন * __restrict__ A,
                               ধ্রুৱ ভিন্ন * __restrict__ B,
                               ভিন্ন * __restrict__ X,
                               ধ্রুৱ int64_t ne02, [[maybe_unused]] ধ্রুৱ int64_t ne03,
                               ধ্রুৱ int64_t nb02, ধ্রুৱ int64_t nb03,
                               ধ্রুৱ int64_t nb12, ধ্রুৱ int64_t nb13,
                               ধ্রুৱ int64_t nb2,  ধ্রুৱ int64_t nb3,
                               ধ্রুৱ পূর্ণাঙ্ক n_arg, ধ্রুৱ পূর্ণাঙ্ক k_arg,
                               ধ্রুৱ sycl::nd_item<2> & item, ভিন্ন * sA) {

    ধ্রুৱ পূর্ণাঙ্ক n = n_template == 0 ? n_arg : n_template;
    ধ্রুৱ পূর্ণাঙ্ক k = k_template == 0 ? k_arg : k_template;

    ধ্রুৱ পূর্ণাঙ্ক batch_idx = item.get_group(1);
    ধ্রুৱ পূর্ণাঙ্ক lane      = item.get_local_id(1) % WARP_SIZE;
    ধ্রুৱ পূর্ণাঙ্ক col_idx   = item.get_local_id(0);

    যদি (col_idx >= k) {
        ঘূৰাই;
    }

    ধ্রুৱ int64_t i03 = batch_idx / ne02;
    ধ্রুৱ int64_t i02 = batch_idx % ne02;

    ধ্রুৱ ভিন্ন * A_batch = (ধ্রুৱ ভিন্ন *) ((ধ্রুৱ ৱর্ণ *) A + i02 * nb02 + i03 * nb03);
    ধ্রুৱ ভিন্ন * B_batch = (ধ্রুৱ ভিন্ন *) ((ধ্রুৱ ৱর্ণ *) B + i02 * nb12 + i03 * nb13);
    ভিন্ন *       X_batch = (ভিন্ন *)       ((ৱর্ণ *)       X + i02 * nb2  + i03 * nb3);

    ধ্রুৱ পূর্ণাঙ্ক offset = item.get_local_id(1) + item.get_local_id(0) * item.get_local_range(1);

#pragma unroll
    বাবে (পূর্ণাঙ্ক i = 0; i < n * n; i += k * WARP_SIZE) {
        ধ্রুৱ পূর্ণাঙ্ক i0 = i + offset;
        যদি (i0 < n * n) {
            sA[i0] = A_batch[i0];
        }
    }

    item.barrier(sycl::access::fence_space::local_space);

    ভিন্ন x_নিচেৰ  = (lane < n) ? B_batch[lane * k + col_idx] : 0.0f;
    ভিন্ন x_উপৰেৰ = (WARP_SIZE + lane < n) ? B_batch[(WARP_SIZE + lane) * k + col_idx] : 0.0f;

    ধ্রুৱ পূর্ণাঙ্ক half      = WARP_SIZE;
    ধ্রুৱ পূর্ণাঙ্ক nrows_নিচেৰ = (n < half) ? n : half;

#pragma unroll
    বাবে (পূর্ণাঙ্ক row = 0; row < nrows_নিচেৰ; ++row) {
        ভিন্ন sum = 0.0f;
        যদি (lane < row) {
            sum += sA[row * n + lane] * x_নিচেৰ;
        }
        sum = warp_reduce_sum<WARP_SIZE>(sum);
        যদি (lane == row) {
            x_নিচেৰ = (x_নিচেৰ - sum) / sA[row * n + row];
        }
    }

#pragma unroll
    বাবে (পূর্ণাঙ্ক row = half; row < n; ++row) {
        ভিন্ন     sum = sA[row * n + lane] * x_নিচেৰ;
        ধ্রুৱ পূর্ণাঙ্ক j   = half + lane;
        যদি (j < row) {
            sum += sA[row * n + j] * x_উপৰেৰ;
        }
        sum = warp_reduce_sum<WARP_SIZE>(sum);
        যদি (lane == row - half) {
            x_উপৰেৰ = (x_উপৰেৰ - sum) / sA[row * n + row];
        }
    }

#pragma unroll
    বাবে (পূর্ণাঙ্ক rr = 0; rr < 2; ++rr) {
        ধ্রুৱ পূর্ণাঙ্ক row = rr * WARP_SIZE + lane;
        যদি (row < n) {
            ধ্রুৱ ভিন্ন val            = (row < half) ? x_নিচেৰ : x_উপৰেৰ;
            X_batch[row * k + col_idx] = val;
        }
    }
}

স্থির শূন্য solve_tri_f32_mkl(dpct::queue_ptr stream,
                               ধ্রুৱ ভিন্ন * A, ভিন্ন * X,
                               পূর্ণাঙ্ক n, পূর্ণাঙ্ক k,
                               int64_t ne02, [[maybe_unused]] int64_t ne03,
                               int64_t nb02, [[maybe_unused]] int64_t nb03,
                               int64_t nb2,  [[maybe_unused]] int64_t nb3) {
    ধ্রুৱ ভিন্ন alpha = 1.0f;
    ধ্রুৱ int64_t total_batches = ne02 * ne03;
    যদি (total_batches == 0) {
        ঘূৰাই;
    }

    ধ্রুৱ int64_t stride_a = nb02 / পরিমাণ(ভিন্ন);
    ধ্রুৱ int64_t stride_x = nb2 / পরিমাণ(ভিন্ন);

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

অন্তঃস্থ শূন্য ggml_sycl_op_solve_tri(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ধ্রুৱ ggml_tensor * src0 = dst->src[0];
    ধ্রুৱ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(src0->type == GGML_TYPE_F32);

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ধ্রুৱ পূর্ণাঙ্ক n    = src0->ne[0];
    ধ্রুৱ পূর্ণাঙ্ক k    = src1->ne[0];
    ধ্রুৱ int64_t ne02 = src0->ne[2];
    ধ্রুৱ int64_t ne03 = src0->ne[3];

    GGML_ASSERT(n <= SYCL_SOLVE_TRI_MAX_N && k <= SYCL_SOLVE_TRI_MAX_K);

    ধ্রুৱ ভিন্ন * A_d = static_cast<ধ্রুৱ ভিন্ন *>(src0->data);
    ধ্রুৱ ভিন্ন * B_d = static_cast<ধ্রুৱ ভিন্ন *>(src1->data);
    ভিন্ন * X_d       = static_cast<ভিন্ন *>(dst->data);

    যদি (X_d != B_d) {
        ধ্রুৱ int64_t total_elements = (int64_t)n * k * ne02 * ne03;
        stream->memcpy(X_d, B_d, total_elements * পরিমাণ(ভিন্ন));
    }

    ধ্রুৱ int64_t nb02 = src0->nb[2];
    ধ্রুৱ int64_t nb03 = src0->nb[3];
    ধ্রুৱ int64_t nb12 = src1->nb[2];
    ধ্রুৱ int64_t nb13 = src1->nb[3];
    ধ্রুৱ int64_t nb2  = dst->nb[2];
    ধ্রুৱ int64_t nb3  = dst->nb[3];

    ধ্রুৱ int64_t total_batches = ne02 * ne03;

    যদি (n <= 2 * WARP_SIZE && k <= 32) {
        ধ্রুৱ পূর্ণাঙ্ক smem_size = 2 * WARP_SIZE * 2 * WARP_SIZE;
        ধ্রুৱ sycl::range<2> grid(1, total_batches);
        ধ্রুৱ sycl::range<2> block(k, WARP_SIZE);
        stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<ভিন্ন, 1> smem_acc(sycl::range<1>(smem_size), cgh);
            cgh.parallel_for(
                sycl::nd_range<2>(grid * block, block),
                [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    solve_tri_f32_fast<0, 0>(A_d, B_d, X_d, ne02, ne03,
                                              nb02, nb03, nb12, nb13, nb2, nb3,
                                              n, k, item, get_pointer(smem_acc));
                });
        });
    } অন্যথা {
        solve_tri_f32_mkl(stream, A_d, X_d, n, k, ne02, ne03, nb02, nb03, nb2, nb3);
    }
}

শূন্য ggml_sycl_solve_tri(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_solve_tri(ctx, dst);
}
