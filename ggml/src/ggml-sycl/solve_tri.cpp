#ضمن "solve_tri.hpp"
#ضمن "common.hpp"
#ضمن <oneapi/mkl/blas.hpp>

template <صحيح n_template, صحيح k_template>
ساكن فراغ solve_tri_f32_fast(ثابت عائم * __restrict__ A,
                               ثابت عائم * __restrict__ B,
                               عائم * __restrict__ X,
                               ثابت int64_t ne02, [[maybe_unused]] ثابت int64_t ne03,
                               ثابت int64_t nb02, ثابت int64_t nb03,
                               ثابت int64_t nb12, ثابت int64_t nb13,
                               ثابت int64_t nb2,  ثابت int64_t nb3,
                               ثابت صحيح n_arg, ثابت صحيح k_arg,
                               ثابت sycl::nd_item<2> & item, عائم * sA) {

    ثابت صحيح n = n_template == 0 ? n_arg : n_template;
    ثابت صحيح k = k_template == 0 ? k_arg : k_template;

    ثابت صحيح batch_idx = item.get_group(1);
    ثابت صحيح lane      = item.get_local_id(1) % WARP_SIZE;
    ثابت صحيح col_idx   = item.get_local_id(0);

    إذا (col_idx >= k) {
        أرجع;
    }

    ثابت int64_t i03 = batch_idx / ne02;
    ثابت int64_t i02 = batch_idx % ne02;

    ثابت عائم * A_batch = (ثابت عائم *) ((ثابت حرف *) A + i02 * nb02 + i03 * nb03);
    ثابت عائم * B_batch = (ثابت عائم *) ((ثابت حرف *) B + i02 * nb12 + i03 * nb13);
    عائم *       X_batch = (عائم *)       ((حرف *)       X + i02 * nb2  + i03 * nb3);

    ثابت صحيح offset = item.get_local_id(1) + item.get_local_id(0) * item.get_local_range(1);

#pragma unroll
    لكل (صحيح i = 0; i < n * n; i += k * WARP_SIZE) {
        ثابت صحيح i0 = i + offset;
        إذا (i0 < n * n) {
            sA[i0] = A_batch[i0];
        }
    }

    item.barrier(sycl::access::fence_space::local_space);

    عائم x_كس  = (lane < n) ? B_batch[lane * k + col_idx] : 0.0f;
    عائم x_سم = (WARP_SIZE + lane < n) ? B_batch[(WARP_SIZE + lane) * k + col_idx] : 0.0f;

    ثابت صحيح half      = WARP_SIZE;
    ثابت صحيح nrows_كس = (n < half) ? n : half;

#pragma unroll
    لكل (صحيح row = 0; row < nrows_كس; ++row) {
        عائم sum = 0.0f;
        إذا (lane < row) {
            sum += sA[row * n + lane] * x_كس;
        }
        sum = warp_reduce_sum<WARP_SIZE>(sum);
        إذا (lane == row) {
            x_كس = (x_كس - sum) / sA[row * n + row];
        }
    }

#pragma unroll
    لكل (صحيح row = half; row < n; ++row) {
        عائم     sum = sA[row * n + lane] * x_كس;
        ثابت صحيح j   = half + lane;
        إذا (j < row) {
            sum += sA[row * n + j] * x_سم;
        }
        sum = warp_reduce_sum<WARP_SIZE>(sum);
        إذا (lane == row - half) {
            x_سم = (x_سم - sum) / sA[row * n + row];
        }
    }

#pragma unroll
    لكل (صحيح rr = 0; rr < 2; ++rr) {
        ثابت صحيح row = rr * WARP_SIZE + lane;
        إذا (row < n) {
            ثابت عائم val            = (row < half) ? x_كس : x_سم;
            X_batch[row * k + col_idx] = val;
        }
    }
}

ساكن فراغ solve_tri_f32_mkl(dpct::queue_ptr stream,
                               ثابت عائم * A, عائم * X,
                               صحيح n, صحيح k,
                               int64_t ne02, [[maybe_unused]] int64_t ne03,
                               int64_t nb02, [[maybe_unused]] int64_t nb03,
                               int64_t nb2,  [[maybe_unused]] int64_t nb3) {
    ثابت عائم alpha = 1.0f;
    ثابت int64_t total_batches = ne02 * ne03;
    إذا (total_batches == 0) {
        أرجع;
    }

    ثابت int64_t stride_a = nb02 / حجم(عائم);
    ثابت int64_t stride_x = nb2 / حجم(عائم);

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

مضمن فراغ ggml_sycl_op_solve_tri(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(src0->type == GGML_TYPE_F32);

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ثابت صحيح n    = src0->ne[0];
    ثابت صحيح k    = src1->ne[0];
    ثابت int64_t ne02 = src0->ne[2];
    ثابت int64_t ne03 = src0->ne[3];

    GGML_ASSERT(n <= SYCL_SOLVE_TRI_MAX_N && k <= SYCL_SOLVE_TRI_MAX_K);

    ثابت عائم * A_d = static_cast<ثابت عائم *>(src0->data);
    ثابت عائم * B_d = static_cast<ثابت عائم *>(src1->data);
    عائم * X_d       = static_cast<عائم *>(dst->data);

    إذا (X_d != B_d) {
        ثابت int64_t total_elements = (int64_t)n * k * ne02 * ne03;
        stream->memcpy(X_d, B_d, total_elements * حجم(عائم));
    }

    ثابت int64_t nb02 = src0->nb[2];
    ثابت int64_t nb03 = src0->nb[3];
    ثابت int64_t nb12 = src1->nb[2];
    ثابت int64_t nb13 = src1->nb[3];
    ثابت int64_t nb2  = dst->nb[2];
    ثابت int64_t nb3  = dst->nb[3];

    ثابت int64_t total_batches = ne02 * ne03;

    إذا (n <= 2 * WARP_SIZE && k <= 32) {
        ثابت صحيح smem_size = 2 * WARP_SIZE * 2 * WARP_SIZE;
        ثابت sycl::range<2> grid(1, total_batches);
        ثابت sycl::range<2> block(k, WARP_SIZE);
        stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<عائم, 1> smem_acc(sycl::range<1>(smem_size), cgh);
            cgh.parallel_for(
                sycl::nd_range<2>(grid * block, block),
                [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    solve_tri_f32_fast<0, 0>(A_d, B_d, X_d, ne02, ne03,
                                              nb02, nb03, nb12, nb13, nb2, nb3,
                                              n, k, item, get_pointer(smem_acc));
                });
        });
    } وإلا {
        solve_tri_f32_mkl(stream, A_d, X_d, n, k, ne02, ne03, nb02, nb03, nb2, nb3);
    }
}

فراغ ggml_sycl_solve_tri(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_بغ scope_dbg_بغ(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_solve_tri(ctx, dst);
}
