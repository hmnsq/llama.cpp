#شامل "solve_tri.hpp"
#شامل "common.hpp"
#شامل <oneapi/mkl/blas.hpp>

template <صحیح n_template, صحیح k_template>
ایستا تهی solve_tri_f32_fast(ثابت اعشاری * __restrict__ A,
                               ثابت اعشاری * __restrict__ B,
                               اعشاری * __restrict__ X,
                               ثابت int64_t ne02, [[maybe_unused]] ثابت int64_t ne03,
                               ثابت int64_t nb02, ثابت int64_t nb03,
                               ثابت int64_t nb12, ثابت int64_t nb13,
                               ثابت int64_t nb2,  ثابت int64_t nb3,
                               ثابت صحیح n_arg, ثابت صحیح k_arg,
                               ثابت sycl::nd_item<2> & item, اعشاری * sA) {

    ثابت صحیح n = n_template == 0 ? n_arg : n_template;
    ثابت صحیح k = k_template == 0 ? k_arg : k_template;

    ثابت صحیح batch_idx = item.get_group(1);
    ثابت صحیح lane      = item.get_local_id(1) % WARP_SIZE;
    ثابت صحیح col_idx   = item.get_local_id(0);

    اگر (col_idx >= k) {
        بازگشت;
    }

    ثابت int64_t i03 = batch_idx / ne02;
    ثابت int64_t i02 = batch_idx % ne02;

    ثابت اعشاری * A_batch = (ثابت اعشاری *) ((ثابت نویسه *) A + i02 * nb02 + i03 * nb03);
    ثابت اعشاری * B_batch = (ثابت اعشاری *) ((ثابت نویسه *) B + i02 * nb12 + i03 * nb13);
    اعشاری *       X_batch = (اعشاری *)       ((نویسه *)       X + i02 * nb2  + i03 * nb3);

    ثابت صحیح offset = item.get_local_id(1) + item.get_local_id(0) * item.get_local_range(1);

#pragma unroll
    برای (صحیح i = 0; i < n * n; i += k * WARP_SIZE) {
        ثابت صحیح i0 = i + offset;
        اگر (i0 < n * n) {
            sA[i0] = A_batch[i0];
        }
    }

    item.barrier(sycl::access::fence_space::local_space);

    اعشاری x_کم  = (lane < n) ? B_batch[lane * k + col_idx] : 0.0f;
    اعشاری x_بالا = (WARP_SIZE + lane < n) ? B_batch[(WARP_SIZE + lane) * k + col_idx] : 0.0f;

    ثابت صحیح half      = WARP_SIZE;
    ثابت صحیح nrows_کم = (n < half) ? n : half;

#pragma unroll
    برای (صحیح row = 0; row < nrows_کم; ++row) {
        اعشاری sum = 0.0f;
        اگر (lane < row) {
            sum += sA[row * n + lane] * x_کم;
        }
        sum = warp_reduce_sum<WARP_SIZE>(sum);
        اگر (lane == row) {
            x_کم = (x_کم - sum) / sA[row * n + row];
        }
    }

#pragma unroll
    برای (صحیح row = half; row < n; ++row) {
        اعشاری     sum = sA[row * n + lane] * x_کم;
        ثابت صحیح j   = half + lane;
        اگر (j < row) {
            sum += sA[row * n + j] * x_بالا;
        }
        sum = warp_reduce_sum<WARP_SIZE>(sum);
        اگر (lane == row - half) {
            x_بالا = (x_بالا - sum) / sA[row * n + row];
        }
    }

#pragma unroll
    برای (صحیح rr = 0; rr < 2; ++rr) {
        ثابت صحیح row = rr * WARP_SIZE + lane;
        اگر (row < n) {
            ثابت اعشاری val            = (row < half) ? x_کم : x_بالا;
            X_batch[row * k + col_idx] = val;
        }
    }
}

ایستا تهی solve_tri_f32_mkl(dpct::queue_ptr stream,
                               ثابت اعشاری * A, اعشاری * X,
                               صحیح n, صحیح k,
                               int64_t ne02, [[maybe_unused]] int64_t ne03,
                               int64_t nb02, [[maybe_unused]] int64_t nb03,
                               int64_t nb2,  [[maybe_unused]] int64_t nb3) {
    ثابت اعشاری alpha = 1.0f;
    ثابت int64_t total_batches = ne02 * ne03;
    اگر (total_batches == 0) {
        بازگشت;
    }

    ثابت int64_t stride_a = nb02 / اندازه(اعشاری);
    ثابت int64_t stride_x = nb2 / اندازه(اعشاری);

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

درونخطی تهی ggml_sycl_op_solve_tri(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(src0->type == GGML_TYPE_F32);

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ثابت صحیح n    = src0->ne[0];
    ثابت صحیح k    = src1->ne[0];
    ثابت int64_t ne02 = src0->ne[2];
    ثابت int64_t ne03 = src0->ne[3];

    GGML_ASSERT(n <= SYCL_SOLVE_TRI_MAX_N && k <= SYCL_SOLVE_TRI_MAX_K);

    ثابت اعشاری * A_d = static_cast<ثابت اعشاری *>(src0->data);
    ثابت اعشاری * B_d = static_cast<ثابت اعشاری *>(src1->data);
    اعشاری * X_d       = static_cast<اعشاری *>(dst->data);

    اگر (X_d != B_d) {
        ثابت int64_t total_elements = (int64_t)n * k * ne02 * ne03;
        stream->memcpy(X_d, B_d, total_elements * اندازه(اعشاری));
    }

    ثابت int64_t nb02 = src0->nb[2];
    ثابت int64_t nb03 = src0->nb[3];
    ثابت int64_t nb12 = src1->nb[2];
    ثابت int64_t nb13 = src1->nb[3];
    ثابت int64_t nb2  = dst->nb[2];
    ثابت int64_t nb3  = dst->nb[3];

    ثابت int64_t total_batches = ne02 * ne03;

    اگر (n <= 2 * WARP_SIZE && k <= 32) {
        ثابت صحیح smem_size = 2 * WARP_SIZE * 2 * WARP_SIZE;
        ثابت sycl::range<2> grid(1, total_batches);
        ثابت sycl::range<2> block(k, WARP_SIZE);
        stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<اعشاری, 1> smem_acc(sycl::range<1>(smem_size), cgh);
            cgh.parallel_for(
                sycl::nd_range<2>(grid * block, block),
                [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    solve_tri_f32_fast<0, 0>(A_d, B_d, X_d, ne02, ne03,
                                              nb02, nb03, nb12, nb13, nb2, nb3,
                                              n, k, item, get_pointer(smem_acc));
                });
        });
    } وگرنه {
        solve_tri_f32_mkl(stream, A_d, X_d, n, k, ne02, ne03, nb02, nb03, nb2, nb3);
    }
}

تهی ggml_sycl_solve_tri(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_solve_tri(ctx, dst);
}
