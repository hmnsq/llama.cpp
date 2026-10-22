#ضمن "solve_tri.hpp"
#ضمن "common.hpp"
#ضمن <oneapi/mkl/blas.hpp>

سانچا <صحيح n_template, صحيح k_template>
ساكن فراغ solve_tri_f32_fast(ثابت عائم * __restrict__ A,
                               ثابت عائم * __restrict__ B,
                               عائم * __restrict__ X,
                               ثابت int64_t ne02, [[maybe_unused]] ثابت int64_t ne03,
                               ثابت int64_t nb02, ثابت int64_t nb03,
                               ثابت int64_t nb12, ثابت int64_t nb13,
                               ثابت int64_t nb2,  ثابت int64_t nb3,
                               ثابت صحيح n_arg, ثابت صحيح k_arg,
                               ثابت sycl::nd_شے<2> & شے, عائم * sA) {

    ثابت صحيح n = n_template == 0 ? n_arg : n_template;
    ثابت صحيح k = k_template == 0 ? k_arg : k_template;

    ثابت صحيح batch_idx = شے.get_گروہ(1);
    ثابت صحيح lane      = شے.get_local_id(1) % WARP_ناپ;
    ثابت صحيح خانہ_idx   = شے.get_local_id(0);

    إذا (خانہ_idx >= k) {
        أرجع;
    }

    ثابت int64_t i03 = batch_idx / ne02;
    ثابت int64_t i02 = batch_idx % ne02;

    ثابت عائم * A_batch = (ثابت عائم *) ((ثابت حرف *) A + i02 * nb02 + i03 * nb03);
    ثابت عائم * B_batch = (ثابت عائم *) ((ثابت حرف *) B + i02 * nb12 + i03 * nb13);
    عائم *       X_batch = (عائم *)       ((حرف *)       X + i02 * nb2  + i03 * nb3);

    ثابت صحيح وتھ = شے.get_local_id(1) + شے.get_local_id(0) * شے.get_local_range(1);

#ہدایت unroll
    لكل (صحيح i = 0; i < n * n; i += k * WARP_ناپ) {
        ثابت صحيح i0 = i + وتھ;
        إذا (i0 < n * n) {
            sA[i0] = A_batch[i0];
        }
    }

    شے.barrier(sycl::access::fence_space::local_space);

    عائم x_کم  = (lane < n) ? B_batch[lane * k + خانہ_idx] : 0.0f;
    عائم x_بلند = (WARP_ناپ + lane < n) ? B_batch[(WARP_ناپ + lane) * k + خانہ_idx] : 0.0f;

    ثابت صحيح half      = WARP_ناپ;
    ثابت صحيح nrows_کم = (n < half) ? n : half;

#ہدایت unroll
    لكل (صحيح صف = 0; صف < nrows_کم; ++صف) {
        عائم جوڑ = 0.0f;
        إذا (lane < صف) {
            جوڑ += sA[صف * n + lane] * x_کم;
        }
        جوڑ = warp_reduce_جوڑ<WARP_ناپ>(جوڑ);
        إذا (lane == صف) {
            x_کم = (x_کم - جوڑ) / sA[صف * n + صف];
        }
    }

#ہدایت unroll
    لكل (صحيح صف = half; صف < n; ++صف) {
        عائم     جوڑ = sA[صف * n + lane] * x_کم;
        ثابت صحيح j   = half + lane;
        إذا (j < صف) {
            جوڑ += sA[صف * n + j] * x_بلند;
        }
        جوڑ = warp_reduce_جوڑ<WARP_ناپ>(جوڑ);
        إذا (lane == صف - half) {
            x_بلند = (x_بلند - جوڑ) / sA[صف * n + صف];
        }
    }

#ہدایت unroll
    لكل (صحيح rr = 0; rr < 2; ++rr) {
        ثابت صحيح صف = rr * WARP_ناپ + lane;
        إذا (صف < n) {
            ثابت عائم val            = (صف < half) ? x_کم : x_بلند;
            X_batch[صف * k + خانہ_idx] = val;
        }
    }
}

ساكن فراغ solve_tri_f32_mkl(dpct::قطار_ptr stream,
                               ثابت عائم * A, عائم * X,
                               صحيح n, صحيح k,
                               int64_t ne02, [[maybe_unused]] int64_t ne03,
                               int64_t nb02, [[maybe_unused]] int64_t nb03,
                               int64_t nb2,  [[maybe_unused]] int64_t nb3) {
    ثابت عائم alpha = 1.0f;
    ثابت int64_t کل_batches = ne02 * ne03;
    إذا (کل_batches == 0) {
        أرجع;
    }

    ثابت int64_t stride_a = nb02 / حجم(عائم);
    ثابت int64_t stride_x = nb2 / حجم(عائم);

    oneapi::mkl::blas::trsm_batch(
        *stream,
        oneapi::mkl::side::سجے,
        oneapi::mkl::uplo::upper,
        oneapi::mkl::transpose::nontrans,
        oneapi::mkl::diag::nonunit,
        k, n, alpha,
        A, n, stride_a,
        X, k, stride_x,
        کل_batches);
}

مضمن فراغ ggml_sycl_op_solve_tri(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(src0->type == GGML_TYPE_F32);

    dpct::قطار_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_سیٹ_device(ctx.device));

    ثابت صحيح n    = src0->ne[0];
    ثابت صحيح k    = src1->ne[0];
    ثابت int64_t ne02 = src0->ne[2];
    ثابت int64_t ne03 = src0->ne[3];

    GGML_ASSERT(n <= SYCL_SOLVE_TRI_MAX_N && k <= SYCL_SOLVE_TRI_MAX_K);

    ثابت عائم * A_d = static_cast<ثابت عائم *>(src0->انکڑے);
    ثابت عائم * B_d = static_cast<ثابت عائم *>(src1->انکڑے);
    عائم * X_d       = static_cast<عائم *>(dst->انکڑے);

    إذا (X_d != B_d) {
        ثابت int64_t کل_elements = (int64_t)n * k * ne02 * ne03;
        stream->حافظہ_نقل(X_d, B_d, کل_elements * حجم(عائم));
    }

    ثابت int64_t nb02 = src0->nb[2];
    ثابت int64_t nb03 = src0->nb[3];
    ثابت int64_t nb12 = src1->nb[2];
    ثابت int64_t nb13 = src1->nb[3];
    ثابت int64_t nb2  = dst->nb[2];
    ثابت int64_t nb3  = dst->nb[3];

    ثابت int64_t کل_batches = ne02 * ne03;

    إذا (n <= 2 * WARP_ناپ && k <= 32) {
        ثابت صحيح smem_ناپ = 2 * WARP_ناپ * 2 * WARP_ناپ;
        ثابت sycl::range<2> grid(1, کل_batches);
        ثابت sycl::range<2> بلاک(k, WARP_ناپ);
        stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<عائم, 1> smem_acc(sycl::range<1>(smem_ناپ), cgh);
            cgh.parallel_for(
                sycl::nd_range<2>(grid * بلاک, بلاک),
                [=](sycl::nd_شے<2> شے) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                    solve_tri_f32_fast<0, 0>(A_d, B_d, X_d, ne02, ne03,
                                              nb02, nb03, nb12, nb13, nb2, nb3,
                                              n, k, شے, get_اشارہ_گر(smem_acc));
                });
        });
    } وإلا {
        solve_tri_f32_mkl(stream, A_d, X_d, n, k, ne02, ne03, nb02, nb03, nb2, nb3);
    }
}

فراغ ggml_sycl_solve_tri(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_solve_tri(ctx, dst);
}
