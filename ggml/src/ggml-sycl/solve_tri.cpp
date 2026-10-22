#شامل "solve_tri.hpp"
#شامل "common.hpp"
#شامل <oneapi/mkl/blas.hpp>

سانچہ <صحیح n_template, صحیح k_template>
ساکن خالی solve_tri_f32_fast(ثابت اعشاری * __restrict__ A,
                               ثابت اعشاری * __restrict__ B,
                               اعشاری * __restrict__ X,
                               ثابت int64_t ne02, [[maybe_unused]] ثابت int64_t ne03,
                               ثابت int64_t nb02, ثابت int64_t nb03,
                               ثابت int64_t nb12, ثابت int64_t nb13,
                               ثابت int64_t nb2,  ثابت int64_t nb3,
                               ثابت صحیح n_arg, ثابت صحیح k_arg,
                               ثابت sycl::nd_شے<2> & شے, اعشاری * sA) {

    ثابت صحیح n = n_template == 0 ? n_arg : n_template;
    ثابت صحیح k = k_template == 0 ? k_arg : k_template;

    ثابت صحیح batch_idx = شے.get_گروہ(1);
    ثابت صحیح lane      = شے.get_local_id(1) % WARP_سائز;
    ثابت صحیح ستون_idx   = شے.get_local_id(0);

    اگر (ستون_idx >= k) {
        واپس;
    }

    ثابت int64_t i03 = batch_idx / ne02;
    ثابت int64_t i02 = batch_idx % ne02;

    ثابت اعشاری * A_batch = (ثابت اعشاری *) ((ثابت حرف *) A + i02 * nb02 + i03 * nb03);
    ثابت اعشاری * B_batch = (ثابت اعشاری *) ((ثابت حرف *) B + i02 * nb12 + i03 * nb13);
    اعشاری *       X_batch = (اعشاری *)       ((حرف *)       X + i02 * nb2  + i03 * nb3);

    ثابت صحیح انحراف = شے.get_local_id(1) + شے.get_local_id(0) * شے.get_local_range(1);

#ہدایت unroll
    برائے (صحیح i = 0; i < n * n; i += k * WARP_سائز) {
        ثابت صحیح i0 = i + انحراف;
        اگر (i0 < n * n) {
            sA[i0] = A_batch[i0];
        }
    }

    شے.barrier(sycl::access::fence_space::local_space);

    اعشاری x_کم  = (lane < n) ? B_batch[lane * k + ستون_idx] : 0.0f;
    اعشاری x_بلند = (WARP_سائز + lane < n) ? B_batch[(WARP_سائز + lane) * k + ستون_idx] : 0.0f;

    ثابت صحیح half      = WARP_سائز;
    ثابت صحیح nrows_کم = (n < half) ? n : half;

#ہدایت unroll
    برائے (صحیح صف = 0; صف < nrows_کم; ++صف) {
        اعشاری مجموع = 0.0f;
        اگر (lane < صف) {
            مجموع += sA[صف * n + lane] * x_کم;
        }
        مجموع = warp_reduce_مجموع<WARP_سائز>(مجموع);
        اگر (lane == صف) {
            x_کم = (x_کم - مجموع) / sA[صف * n + صف];
        }
    }

#ہدایت unroll
    برائے (صحیح صف = half; صف < n; ++صف) {
        اعشاری     مجموع = sA[صف * n + lane] * x_کم;
        ثابت صحیح j   = half + lane;
        اگر (j < صف) {
            مجموع += sA[صف * n + j] * x_بلند;
        }
        مجموع = warp_reduce_مجموع<WARP_سائز>(مجموع);
        اگر (lane == صف - half) {
            x_بلند = (x_بلند - مجموع) / sA[صف * n + صف];
        }
    }

#ہدایت unroll
    برائے (صحیح rr = 0; rr < 2; ++rr) {
        ثابت صحیح صف = rr * WARP_سائز + lane;
        اگر (صف < n) {
            ثابت اعشاری val            = (صف < half) ? x_کم : x_بلند;
            X_batch[صف * k + ستون_idx] = val;
        }
    }
}

ساکن خالی solve_tri_f32_mkl(dpct::قطار_ptr stream,
                               ثابت اعشاری * A, اعشاری * X,
                               صحیح n, صحیح k,
                               int64_t ne02, [[maybe_unused]] int64_t ne03,
                               int64_t nb02, [[maybe_unused]] int64_t nb03,
                               int64_t nb2,  [[maybe_unused]] int64_t nb3) {
    ثابت اعشاری alpha = 1.0f;
    ثابت int64_t کل_batches = ne02 * ne03;
    اگر (کل_batches == 0) {
        واپس;
    }

    ثابت int64_t stride_a = nb02 / حجم(اعشاری);
    ثابت int64_t stride_x = nb2 / حجم(اعشاری);

    oneapi::mkl::blas::trsm_batch(
        *stream,
        oneapi::mkl::side::دائیں,
        oneapi::mkl::uplo::upper,
        oneapi::mkl::transpose::nontrans,
        oneapi::mkl::diag::nonunit,
        k, n, alpha,
        A, n, stride_a,
        X, k, stride_x,
        کل_batches);
}

اندرونی خالی ggml_sycl_op_solve_tri(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(src0->type == GGML_TYPE_F32);

    dpct::قطار_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_سیٹ_device(ctx.device));

    ثابت صحیح n    = src0->ne[0];
    ثابت صحیح k    = src1->ne[0];
    ثابت int64_t ne02 = src0->ne[2];
    ثابت int64_t ne03 = src0->ne[3];

    GGML_ASSERT(n <= SYCL_SOLVE_TRI_MAX_N && k <= SYCL_SOLVE_TRI_MAX_K);

    ثابت اعشاری * A_d = static_cast<ثابت اعشاری *>(src0->ڈیٹا);
    ثابت اعشاری * B_d = static_cast<ثابت اعشاری *>(src1->ڈیٹا);
    اعشاری * X_d       = static_cast<اعشاری *>(dst->ڈیٹا);

    اگر (X_d != B_d) {
        ثابت int64_t کل_elements = (int64_t)n * k * ne02 * ne03;
        stream->حافظہ_نقل(X_d, B_d, کل_elements * حجم(اعشاری));
    }

    ثابت int64_t nb02 = src0->nb[2];
    ثابت int64_t nb03 = src0->nb[3];
    ثابت int64_t nb12 = src1->nb[2];
    ثابت int64_t nb13 = src1->nb[3];
    ثابت int64_t nb2  = dst->nb[2];
    ثابت int64_t nb3  = dst->nb[3];

    ثابت int64_t کل_batches = ne02 * ne03;

    اگر (n <= 2 * WARP_سائز && k <= 32) {
        ثابت صحیح smem_سائز = 2 * WARP_سائز * 2 * WARP_سائز;
        ثابت sycl::range<2> grid(1, کل_batches);
        ثابت sycl::range<2> بلاک(k, WARP_سائز);
        stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<اعشاری, 1> smem_acc(sycl::range<1>(smem_سائز), cgh);
            cgh.parallel_for(
                sycl::nd_range<2>(grid * بلاک, بلاک),
                [=](sycl::nd_شے<2> شے) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                    solve_tri_f32_fast<0, 0>(A_d, B_d, X_d, ne02, ne03,
                                              nb02, nb03, nb12, nb13, nb2, nb3,
                                              n, k, شے, get_اشارہ_گر(smem_acc));
                });
        });
    } ورنہ {
        solve_tri_f32_mkl(stream, A_d, X_d, n, k, ne02, ne03, nb02, nb03, nb2, nb3);
    }
}

خالی ggml_sycl_solve_tri(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_solve_tri(ctx, dst);
}
