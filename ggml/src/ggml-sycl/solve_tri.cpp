#ضمن "solve_tri.hpp"
#ضمن "common.hpp"
#ضمن <oneapi/mkl/blas.hpp>

کينډۍ <صحيح n_template, صحيح k_template>
ساكن فراغ solve_tri_f32_fast(ثابت عائم * __restrict__ A,
                               ثابت عائم * __restrict__ B,
                               عائم * __restrict__ X,
                               ثابت int64_t ne02, [[maybe_unused]] ثابت int64_t ne03,
                               ثابت int64_t nb02, ثابت int64_t nb03,
                               ثابت int64_t nb12, ثابت int64_t nb13,
                               ثابت int64_t nb2,  ثابت int64_t nb3,
                               ثابت صحيح n_arg, ثابت صحيح k_arg,
                               ثابت sycl::nd_توکی<2> & توکی, عائم * sA) {

    ثابت صحيح n = n_template == 0 ? n_arg : n_template;
    ثابت صحيح k = k_template == 0 ? k_arg : k_template;

    ثابت صحيح batch_idx = توکی.get_ډله(1);
    ثابت صحيح lane      = توکی.get_local_id(1) % WARP_اندازه;
    ثابت صحيح ستون_idx   = توکی.get_local_id(0);

    إذا (ستون_idx >= k) {
        أرجع;
    }

    ثابت int64_t i03 = batch_idx / ne02;
    ثابت int64_t i02 = batch_idx % ne02;

    ثابت عائم * A_batch = (ثابت عائم *) ((ثابت حرف *) A + i02 * nb02 + i03 * nb03);
    ثابت عائم * B_batch = (ثابت عائم *) ((ثابت حرف *) B + i02 * nb12 + i03 * nb13);
    عائم *       X_batch = (عائم *)       ((حرف *)       X + i02 * nb2  + i03 * nb3);

    ثابت صحيح دننه_واټن = توکی.get_local_id(1) + توکی.get_local_id(0) * توکی.get_local_range(1);

#لارښوونه unroll
    لكل (صحيح i = 0; i < n * n; i += k * WARP_اندازه) {
        ثابت صحيح i0 = i + دننه_واټن;
        إذا (i0 < n * n) {
            sA[i0] = A_batch[i0];
        }
    }

    توکی.barrier(sycl::access::fence_space::local_space);

    عائم x_ټیټ  = (lane < n) ? B_batch[lane * k + ستون_idx] : 0.0f;
    عائم x_لوړ = (WARP_اندازه + lane < n) ? B_batch[(WARP_اندازه + lane) * k + ستون_idx] : 0.0f;

    ثابت صحيح half      = WARP_اندازه;
    ثابت صحيح nrows_ټیټ = (n < half) ? n : half;

#لارښوونه unroll
    لكل (صحيح لیکه = 0; لیکه < nrows_ټیټ; ++لیکه) {
        عائم مجموعه = 0.0f;
        إذا (lane < لیکه) {
            مجموعه += sA[لیکه * n + lane] * x_ټیټ;
        }
        مجموعه = warp_reduce_مجموعه<WARP_اندازه>(مجموعه);
        إذا (lane == لیکه) {
            x_ټیټ = (x_ټیټ - مجموعه) / sA[لیکه * n + لیکه];
        }
    }

#لارښوونه unroll
    لكل (صحيح لیکه = half; لیکه < n; ++لیکه) {
        عائم     مجموعه = sA[لیکه * n + lane] * x_ټیټ;
        ثابت صحيح j   = half + lane;
        إذا (j < لیکه) {
            مجموعه += sA[لیکه * n + j] * x_لوړ;
        }
        مجموعه = warp_reduce_مجموعه<WARP_اندازه>(مجموعه);
        إذا (lane == لیکه - half) {
            x_لوړ = (x_لوړ - مجموعه) / sA[لیکه * n + لیکه];
        }
    }

#لارښوونه unroll
    لكل (صحيح rr = 0; rr < 2; ++rr) {
        ثابت صحيح لیکه = rr * WARP_اندازه + lane;
        إذا (لیکه < n) {
            ثابت عائم val            = (لیکه < half) ? x_ټیټ : x_لوړ;
            X_batch[لیکه * k + ستون_idx] = val;
        }
    }
}

ساكن فراغ solve_tri_f32_mkl(dpct::کتار_ptr stream,
                               ثابت عائم * A, عائم * X,
                               صحيح n, صحيح k,
                               int64_t ne02, [[maybe_unused]] int64_t ne03,
                               int64_t nb02, [[maybe_unused]] int64_t nb03,
                               int64_t nb2,  [[maybe_unused]] int64_t nb3) {
    ثابت عائم alpha = 1.0f;
    ثابت int64_t ټول_batches = ne02 * ne03;
    إذا (ټول_batches == 0) {
        أرجع;
    }

    ثابت int64_t stride_a = nb02 / حجم(عائم);
    ثابت int64_t stride_x = nb2 / حجم(عائم);

    oneapi::mkl::blas::trsm_batch(
        *stream,
        oneapi::mkl::side::ښی,
        oneapi::mkl::uplo::upper,
        oneapi::mkl::transpose::nontrans,
        oneapi::mkl::diag::nonunit,
        k, n, alpha,
        A, n, stride_a,
        X, k, stride_x,
        ټول_batches);
}

مضمن فراغ ggml_sycl_op_solve_tri(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(src0->type == GGML_TYPE_F32);

    dpct::کتار_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_سېټ_device(ctx.device));

    ثابت صحيح n    = src0->ne[0];
    ثابت صحيح k    = src1->ne[0];
    ثابت int64_t ne02 = src0->ne[2];
    ثابت int64_t ne03 = src0->ne[3];

    GGML_ASSERT(n <= SYCL_SOLVE_TRI_MAX_N && k <= SYCL_SOLVE_TRI_MAX_K);

    ثابت عائم * A_d = static_cast<ثابت عائم *>(src0->معلومات);
    ثابت عائم * B_d = static_cast<ثابت عائم *>(src1->معلومات);
    عائم * X_d       = static_cast<عائم *>(dst->معلومات);

    إذا (X_d != B_d) {
        ثابت int64_t ټول_elements = (int64_t)n * k * ne02 * ne03;
        stream->حافظه_نقل(X_d, B_d, ټول_elements * حجم(عائم));
    }

    ثابت int64_t nb02 = src0->nb[2];
    ثابت int64_t nb03 = src0->nb[3];
    ثابت int64_t nb12 = src1->nb[2];
    ثابت int64_t nb13 = src1->nb[3];
    ثابت int64_t nb2  = dst->nb[2];
    ثابت int64_t nb3  = dst->nb[3];

    ثابت int64_t ټول_batches = ne02 * ne03;

    إذا (n <= 2 * WARP_اندازه && k <= 32) {
        ثابت صحيح smem_اندازه = 2 * WARP_اندازه * 2 * WARP_اندازه;
        ثابت sycl::range<2> grid(1, ټول_batches);
        ثابت sycl::range<2> بلاک(k, WARP_اندازه);
        stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<عائم, 1> smem_acc(sycl::range<1>(smem_اندازه), cgh);
            cgh.parallel_for(
                sycl::nd_range<2>(grid * بلاک, بلاک),
                [=](sycl::nd_توکی<2> توکی) [[sycl::reqd_sub_ډله_اندازه(WARP_اندازه)]] {
                    solve_tri_f32_fast<0, 0>(A_d, B_d, X_d, ne02, ne03,
                                              nb02, nb03, nb12, nb13, nb2, nb3,
                                              n, k, توکی, get_اشاره_ګر(smem_acc));
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
