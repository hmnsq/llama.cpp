#ضمن "solve_tri.hpp"
#ضمن "common.hpp"
#ضمن <oneapi/mkl/blas.hpp>

قالب <صحيح n_template, صحيح k_template>
ساكن فراغ solve_tri_f32_fast(ثابت عائم * __restrict__ A,
                               ثابت عائم * __restrict__ B,
                               عائم * __restrict__ X,
                               ثابت int64_t ne02, [[maybe_unused]] ثابت int64_t ne03,
                               ثابت int64_t nb02, ثابت int64_t nb03,
                               ثابت int64_t nb12, ثابت int64_t nb13,
                               ثابت int64_t nb2,  ثابت int64_t nb3,
                               ثابت صحيح n_arg, ثابت صحيح k_arg,
                               ثابت sycl::nd_قلم<2> & قلم, عائم * sA) {

    ثابت صحيح n = n_template == 0 ? n_arg : n_template;
    ثابت صحيح k = k_template == 0 ? k_arg : k_template;

    ثابت صحيح batch_idx = قلم.get_گروه(1);
    ثابت صحيح lane      = قلم.get_local_id(1) % WARP_اندازه;
    ثابت صحيح ستون_idx   = قلم.get_local_id(0);

    إذا (ستون_idx >= k) {
        أرجع;
    }

    ثابت int64_t i03 = batch_idx / ne02;
    ثابت int64_t i02 = batch_idx % ne02;

    ثابت عائم * A_batch = (ثابت عائم *) ((ثابت حرف *) A + i02 * nb02 + i03 * nb03);
    ثابت عائم * B_batch = (ثابت عائم *) ((ثابت حرف *) B + i02 * nb12 + i03 * nb13);
    عائم *       X_batch = (عائم *)       ((حرف *)       X + i02 * nb2  + i03 * nb3);

    ثابت صحيح جابه_جایی = قلم.get_local_id(1) + قلم.get_local_id(0) * قلم.get_local_range(1);

#رهنمود unroll
    لكل (صحيح i = 0; i < n * n; i += k * WARP_اندازه) {
        ثابت صحيح i0 = i + جابه_جایی;
        إذا (i0 < n * n) {
            sA[i0] = A_batch[i0];
        }
    }

    قلم.barrier(sycl::access::fence_space::local_space);

    عائم x_کم  = (lane < n) ? B_batch[lane * k + ستون_idx] : 0.0f;
    عائم x_بالا = (WARP_اندازه + lane < n) ? B_batch[(WARP_اندازه + lane) * k + ستون_idx] : 0.0f;

    ثابت صحيح half      = WARP_اندازه;
    ثابت صحيح nrows_کم = (n < half) ? n : half;

#رهنمود unroll
    لكل (صحيح ردیف = 0; ردیف < nrows_کم; ++ردیف) {
        عائم جمع = 0.0f;
        إذا (lane < ردیف) {
            جمع += sA[ردیف * n + lane] * x_کم;
        }
        جمع = warp_reduce_جمع<WARP_اندازه>(جمع);
        إذا (lane == ردیف) {
            x_کم = (x_کم - جمع) / sA[ردیف * n + ردیف];
        }
    }

#رهنمود unroll
    لكل (صحيح ردیف = half; ردیف < n; ++ردیف) {
        عائم     جمع = sA[ردیف * n + lane] * x_کم;
        ثابت صحيح j   = half + lane;
        إذا (j < ردیف) {
            جمع += sA[ردیف * n + j] * x_بالا;
        }
        جمع = warp_reduce_جمع<WARP_اندازه>(جمع);
        إذا (lane == ردیف - half) {
            x_بالا = (x_بالا - جمع) / sA[ردیف * n + ردیف];
        }
    }

#رهنمود unroll
    لكل (صحيح rr = 0; rr < 2; ++rr) {
        ثابت صحيح ردیف = rr * WARP_اندازه + lane;
        إذا (ردیف < n) {
            ثابت عائم val            = (ردیف < half) ? x_کم : x_بالا;
            X_batch[ردیف * k + ستون_idx] = val;
        }
    }
}

ساكن فراغ solve_tri_f32_mkl(dpct::صف_ptr stream,
                               ثابت عائم * A, عائم * X,
                               صحيح n, صحيح k,
                               int64_t ne02, [[maybe_unused]] int64_t ne03,
                               int64_t nb02, [[maybe_unused]] int64_t nb03,
                               int64_t nb2,  [[maybe_unused]] int64_t nb3) {
    ثابت عائم alpha = 1.0f;
    ثابت int64_t مجموع_batches = ne02 * ne03;
    إذا (مجموع_batches == 0) {
        أرجع;
    }

    ثابت int64_t stride_a = nb02 / حجم(عائم);
    ثابت int64_t stride_x = nb2 / حجم(عائم);

    oneapi::mkl::blas::trsm_batch(
        *stream,
        oneapi::mkl::side::راست,
        oneapi::mkl::uplo::upper,
        oneapi::mkl::transpose::nontrans,
        oneapi::mkl::diag::nonunit,
        k, n, alpha,
        A, n, stride_a,
        X, k, stride_x,
        مجموع_batches);
}

مضمن فراغ ggml_sycl_op_solve_tri(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(src0->type == GGML_TYPE_F32);

    dpct::صف_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_مجموعه_device(ctx.device));

    ثابت صحيح n    = src0->ne[0];
    ثابت صحيح k    = src1->ne[0];
    ثابت int64_t ne02 = src0->ne[2];
    ثابت int64_t ne03 = src0->ne[3];

    GGML_ASSERT(n <= SYCL_SOLVE_TRI_MAX_N && k <= SYCL_SOLVE_TRI_MAX_K);

    ثابت عائم * A_d = static_cast<ثابت عائم *>(src0->داده);
    ثابت عائم * B_d = static_cast<ثابت عائم *>(src1->داده);
    عائم * X_d       = static_cast<عائم *>(dst->داده);

    إذا (X_d != B_d) {
        ثابت int64_t مجموع_elements = (int64_t)n * k * ne02 * ne03;
        stream->کپی_حافظه(X_d, B_d, مجموع_elements * حجم(عائم));
    }

    ثابت int64_t nb02 = src0->nb[2];
    ثابت int64_t nb03 = src0->nb[3];
    ثابت int64_t nb12 = src1->nb[2];
    ثابت int64_t nb13 = src1->nb[3];
    ثابت int64_t nb2  = dst->nb[2];
    ثابت int64_t nb3  = dst->nb[3];

    ثابت int64_t مجموع_batches = ne02 * ne03;

    إذا (n <= 2 * WARP_اندازه && k <= 32) {
        ثابت صحيح smem_اندازه = 2 * WARP_اندازه * 2 * WARP_اندازه;
        ثابت sycl::range<2> grid(1, مجموع_batches);
        ثابت sycl::range<2> بلوک(k, WARP_اندازه);
        stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<عائم, 1> smem_acc(sycl::range<1>(smem_اندازه), cgh);
            cgh.parallel_for(
                sycl::nd_range<2>(grid * بلوک, بلوک),
                [=](sycl::nd_قلم<2> قلم) [[sycl::reqd_sub_گروه_اندازه(WARP_اندازه)]] {
                    solve_tri_f32_fast<0, 0>(A_d, B_d, X_d, ne02, ne03,
                                              nb02, nb03, nb12, nb13, nb2, nb3,
                                              n, k, قلم, get_اشاره_گر(smem_acc));
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
