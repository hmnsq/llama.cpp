#অন্তর্ভাব "solve_tri.hpp"
#অন্তর্ভাব "common.hpp"
#অন্তর্ভাব <oneapi/mkl/blas.hpp>

template <পূর্ণসংখ্যা n_template, পূর্ণসংখ্যা k_template>
স্থির শূন্য solve_tri_f32_fast(ধ্রুবক দশমিক * __restrict__ A,
                               ধ্রুবক দশমিক * __restrict__ B,
                               দশমিক * __restrict__ X,
                               ধ্রুবক int64_t ne02, [[maybe_unused]] ধ্রুবক int64_t ne03,
                               ধ্রুবক int64_t nb02, ধ্রুবক int64_t nb03,
                               ধ্রুবক int64_t nb12, ধ্রুবক int64_t nb13,
                               ধ্রুবক int64_t nb2,  ধ্রুবক int64_t nb3,
                               ধ্রুবক পূর্ণসংখ্যা n_arg, ধ্রুবক পূর্ণসংখ্যা k_arg,
                               ধ্রুবক sycl::nd_item<2> & item, দশমিক * sA) {

    ধ্রুবক পূর্ণসংখ্যা n = n_template == 0 ? n_arg : n_template;
    ধ্রুবক পূর্ণসংখ্যা k = k_template == 0 ? k_arg : k_template;

    ধ্রুবক পূর্ণসংখ্যা batch_idx = item.get_group(1);
    ধ্রুবক পূর্ণসংখ্যা lane      = item.get_local_id(1) % WARP_আকার;
    ধ্রুবক পূর্ণসংখ্যা col_idx   = item.get_local_id(0);

    যদি (col_idx >= k) {
        ফেরত;
    }

    ধ্রুবক int64_t i03 = batch_idx / ne02;
    ধ্রুবক int64_t i02 = batch_idx % ne02;

    ধ্রুবক দশমিক * A_batch = (ধ্রুবক দশমিক *) ((ধ্রুবক অক্ষর *) A + i02 * nb02 + i03 * nb03);
    ধ্রুবক দশমিক * B_batch = (ধ্রুবক দশমিক *) ((ধ্রুবক অক্ষর *) B + i02 * nb12 + i03 * nb13);
    দশমিক *       X_batch = (দশমিক *)       ((অক্ষর *)       X + i02 * nb2  + i03 * nb3);

    ধ্রুবক পূর্ণসংখ্যা offset = item.get_local_id(1) + item.get_local_id(0) * item.get_local_range(1);

#pragma unroll
    জন্য (পূর্ণসংখ্যা i = 0; i < n * n; i += k * WARP_আকার) {
        ধ্রুবক পূর্ণসংখ্যা i0 = i + offset;
        যদি (i0 < n * n) {
            sA[i0] = A_batch[i0];
        }
    }

    item.barrier(sycl::access::fence_space::local_space);

    দশমিক x_নিচের  = (lane < n) ? B_batch[lane * k + col_idx] : 0.0f;
    দশমিক x_উপরের = (WARP_আকার + lane < n) ? B_batch[(WARP_আকার + lane) * k + col_idx] : 0.0f;

    ধ্রুবক পূর্ণসংখ্যা half      = WARP_আকার;
    ধ্রুবক পূর্ণসংখ্যা nrows_নিচের = (n < half) ? n : half;

#pragma unroll
    জন্য (পূর্ণসংখ্যা row = 0; row < nrows_নিচের; ++row) {
        দশমিক যোগ_2 = 0.0f;
        যদি (lane < row) {
            যোগ_2 += sA[row * n + lane] * x_নিচের;
        }
        যোগ_2 = warp_reduce_যোগ<WARP_আকার>(যোগ_2);
        যদি (lane == row) {
            x_নিচের = (x_নিচের - যোগ_2) / sA[row * n + row];
        }
    }

#pragma unroll
    জন্য (পূর্ণসংখ্যা row = half; row < n; ++row) {
        দশমিক     যোগ_2 = sA[row * n + lane] * x_নিচের;
        ধ্রুবক পূর্ণসংখ্যা j   = half + lane;
        যদি (j < row) {
            যোগ_2 += sA[row * n + j] * x_উপরের;
        }
        যোগ_2 = warp_reduce_যোগ<WARP_আকার>(যোগ_2);
        যদি (lane == row - half) {
            x_উপরের = (x_উপরের - যোগ_2) / sA[row * n + row];
        }
    }

#pragma unroll
    জন্য (পূর্ণসংখ্যা rr = 0; rr < 2; ++rr) {
        ধ্রুবক পূর্ণসংখ্যা row = rr * WARP_আকার + lane;
        যদি (row < n) {
            ধ্রুবক দশমিক val            = (row < half) ? x_নিচের : x_উপরের;
            X_batch[row * k + col_idx] = val;
        }
    }
}

স্থির শূন্য solve_tri_f32_mkl(dpct::queue_ptr stream,
                               ধ্রুবক দশমিক * A, দশমিক * X,
                               পূর্ণসংখ্যা n, পূর্ণসংখ্যা k,
                               int64_t ne02, [[maybe_unused]] int64_t ne03,
                               int64_t nb02, [[maybe_unused]] int64_t nb03,
                               int64_t nb2,  [[maybe_unused]] int64_t nb3) {
    ধ্রুবক দশমিক alpha = 1.0f;
    ধ্রুবক int64_t যোগ_batches = ne02 * ne03;
    যদি (যোগ_batches == 0) {
        ফেরত;
    }

    ধ্রুবক int64_t stride_a = nb02 / আকার(দশমিক);
    ধ্রুবক int64_t stride_x = nb2 / আকার(দশমিক);

    oneapi::mkl::blas::trsm_batch(
        *stream,
        oneapi::mkl::side::right,
        oneapi::mkl::uplo::upper,
        oneapi::mkl::transpose::nontrans,
        oneapi::mkl::diag::nonunit,
        k, n, alpha,
        A, n, stride_a,
        X, k, stride_x,
        যোগ_batches);
}

অন্তঃরেখ শূন্য ggml_sycl_op_solve_tri(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ধ্রুবক ggml_tensor * src0 = dst->src[0];
    ধ্রুবক ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(src0->type == GGML_TYPE_F32);

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ধ্রুবক পূর্ণসংখ্যা n    = src0->ne[0];
    ধ্রুবক পূর্ণসংখ্যা k    = src1->ne[0];
    ধ্রুবক int64_t ne02 = src0->ne[2];
    ধ্রুবক int64_t ne03 = src0->ne[3];

    GGML_ASSERT(n <= SYCL_SOLVE_TRI_MAX_N && k <= SYCL_SOLVE_TRI_MAX_K);

    ধ্রুবক দশমিক * A_d = static_cast<ধ্রুবক দশমিক *>(src0->তথ্য);
    ধ্রুবক দশমিক * B_d = static_cast<ধ্রুবক দশমিক *>(src1->তথ্য);
    দশমিক * X_d       = static_cast<দশমিক *>(dst->তথ্য);

    যদি (X_d != B_d) {
        ধ্রুবক int64_t যোগ_elements = (int64_t)n * k * ne02 * ne03;
        stream->memcpy(X_d, B_d, যোগ_elements * আকার(দশমিক));
    }

    ধ্রুবক int64_t nb02 = src0->nb[2];
    ধ্রুবক int64_t nb03 = src0->nb[3];
    ধ্রুবক int64_t nb12 = src1->nb[2];
    ধ্রুবক int64_t nb13 = src1->nb[3];
    ধ্রুবক int64_t nb2  = dst->nb[2];
    ধ্রুবক int64_t nb3  = dst->nb[3];

    ধ্রুবক int64_t যোগ_batches = ne02 * ne03;

    যদি (n <= 2 * WARP_আকার && k <= 32) {
        ধ্রুবক পূর্ণসংখ্যা smem_আকার = 2 * WARP_আকার * 2 * WARP_আকার;
        ধ্রুবক sycl::range<2> grid(1, যোগ_batches);
        ধ্রুবক sycl::range<2> block(k, WARP_আকার);
        stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<দশমিক, 1> smem_acc(sycl::range<1>(smem_আকার), cgh);
            cgh.parallel_for(
                sycl::nd_range<2>(grid * block, block),
                [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_আকার(WARP_আকার)]] {
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
