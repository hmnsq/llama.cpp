#అన్తర్భావ "solve_tri.hpp"
#అన్తర్భావ "common.hpp"
#అన్తర్భావ <oneapi/mkl/blas.hpp>

మూస <పూర్ణాంకం n_template, పూర్ణాంకం k_template>
స్థిర శూన్య solve_tri_f32_fast(స్థిరరాశి దశాంశం * __restrict__ A,
                               స్థిరరాశి దశాంశం * __restrict__ B,
                               దశాంశం * __restrict__ X,
                               స్థిరరాశి int64_t ne02, [[maybe_unused]] స్థిరరాశి int64_t ne03,
                               స్థిరరాశి int64_t nb02, స్థిరరాశి int64_t nb03,
                               స్థిరరాశి int64_t nb12, స్థిరరాశి int64_t nb13,
                               స్థిరరాశి int64_t nb2,  స్థిరరాశి int64_t nb3,
                               స్థిరరాశి పూర్ణాంకం n_arg, స్థిరరాశి పూర్ణాంకం k_arg,
                               స్థిరరాశి sycl::nd_అంశం<2> & అంశం, దశాంశం * sA) {

    స్థిరరాశి పూర్ణాంకం n = n_template == 0 ? n_arg : n_template;
    స్థిరరాశి పూర్ణాంకం k = k_template == 0 ? k_arg : k_template;

    స్థిరరాశి పూర్ణాంకం batch_idx = అంశం.get_సమూహం(1);
    స్థిరరాశి పూర్ణాంకం lane      = అంశం.get_local_id(1) % WARP_కొలత;
    స్థిరరాశి పూర్ణాంకం నిలువు_idx   = అంశం.get_local_id(0);

    ఒకవేళ (నిలువు_idx >= k) {
        తిరిగి;
    }

    స్థిరరాశి int64_t i03 = batch_idx / ne02;
    స్థిరరాశి int64_t i02 = batch_idx % ne02;

    స్థిరరాశి దశాంశం * A_batch = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) A + i02 * nb02 + i03 * nb03);
    స్థిరరాశి దశాంశం * B_batch = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) B + i02 * nb12 + i03 * nb13);
    దశాంశం *       X_batch = (దశాంశం *)       ((అక్షరం *)       X + i02 * nb2  + i03 * nb3);

    స్థిరరాశి పూర్ణాంకం స్థానభ్రంశం = అంశం.get_local_id(1) + అంశం.get_local_id(0) * అంశం.get_local_range(1);

#సూచన unroll
    కొరకు (పూర్ణాంకం i = 0; i < n * n; i += k * WARP_కొలత) {
        స్థిరరాశి పూర్ణాంకం i0 = i + స్థానభ్రంశం;
        ఒకవేళ (i0 < n * n) {
            sA[i0] = A_batch[i0];
        }
    }

    అంశం.barrier(sycl::access::fence_space::local_space);

    దశాంశం x_దిగువ  = (lane < n) ? B_batch[lane * k + నిలువు_idx] : 0.0f;
    దశాంశం x_ఎగువ = (WARP_కొలత + lane < n) ? B_batch[(WARP_కొలత + lane) * k + నిలువు_idx] : 0.0f;

    స్థిరరాశి పూర్ణాంకం half      = WARP_కొలత;
    స్థిరరాశి పూర్ణాంకం nrows_దిగువ = (n < half) ? n : half;

#సూచన unroll
    కొరకు (పూర్ణాంకం అడ్డువరుస = 0; అడ్డువరుస < nrows_దిగువ; ++అడ్డువరుస) {
        దశాంశం కూడిక = 0.0f;
        ఒకవేళ (lane < అడ్డువరుస) {
            కూడిక += sA[అడ్డువరుస * n + lane] * x_దిగువ;
        }
        కూడిక = warp_reduce_కూడిక<WARP_కొలత>(కూడిక);
        ఒకవేళ (lane == అడ్డువరుస) {
            x_దిగువ = (x_దిగువ - కూడిక) / sA[అడ్డువరుస * n + అడ్డువరుస];
        }
    }

#సూచన unroll
    కొరకు (పూర్ణాంకం అడ్డువరుస = half; అడ్డువరుస < n; ++అడ్డువరుస) {
        దశాంశం     కూడిక = sA[అడ్డువరుస * n + lane] * x_దిగువ;
        స్థిరరాశి పూర్ణాంకం j   = half + lane;
        ఒకవేళ (j < అడ్డువరుస) {
            కూడిక += sA[అడ్డువరుస * n + j] * x_ఎగువ;
        }
        కూడిక = warp_reduce_కూడిక<WARP_కొలత>(కూడిక);
        ఒకవేళ (lane == అడ్డువరుస - half) {
            x_ఎగువ = (x_ఎగువ - కూడిక) / sA[అడ్డువరుస * n + అడ్డువరుస];
        }
    }

#సూచన unroll
    కొరకు (పూర్ణాంకం rr = 0; rr < 2; ++rr) {
        స్థిరరాశి పూర్ణాంకం అడ్డువరుస = rr * WARP_కొలత + lane;
        ఒకవేళ (అడ్డువరుస < n) {
            స్థిరరాశి దశాంశం val            = (అడ్డువరుస < half) ? x_దిగువ : x_ఎగువ;
            X_batch[అడ్డువరుస * k + నిలువు_idx] = val;
        }
    }
}

స్థిర శూన్య solve_tri_f32_mkl(dpct::వరుస_ptr stream,
                               స్థిరరాశి దశాంశం * A, దశాంశం * X,
                               పూర్ణాంకం n, పూర్ణాంకం k,
                               int64_t ne02, [[maybe_unused]] int64_t ne03,
                               int64_t nb02, [[maybe_unused]] int64_t nb03,
                               int64_t nb2,  [[maybe_unused]] int64_t nb3) {
    స్థిరరాశి దశాంశం alpha = 1.0f;
    స్థిరరాశి int64_t మొత్తం_batches = ne02 * ne03;
    ఒకవేళ (మొత్తం_batches == 0) {
        తిరిగి;
    }

    స్థిరరాశి int64_t stride_a = nb02 / పరిమాణం(దశాంశం);
    స్థిరరాశి int64_t stride_x = nb2 / పరిమాణం(దశాంశం);

    oneapi::mkl::blas::trsm_batch(
        *stream,
        oneapi::mkl::side::కుడి,
        oneapi::mkl::uplo::upper,
        oneapi::mkl::transpose::nontrans,
        oneapi::mkl::diag::nonunit,
        k, n, alpha,
        A, n, stride_a,
        X, k, stride_x,
        మొత్తం_batches);
}

అంతఃవరుస శూన్య ggml_sycl_op_solve_tri(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(src0->type == GGML_TYPE_F32);

    dpct::వరుస_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_సమితి_device(ctx.device));

    స్థిరరాశి పూర్ణాంకం n    = src0->ne[0];
    స్థిరరాశి పూర్ణాంకం k    = src1->ne[0];
    స్థిరరాశి int64_t ne02 = src0->ne[2];
    స్థిరరాశి int64_t ne03 = src0->ne[3];

    GGML_ASSERT(n <= SYCL_SOLVE_TRI_MAX_N && k <= SYCL_SOLVE_TRI_MAX_K);

    స్థిరరాశి దశాంశం * A_d = static_cast<స్థిరరాశి దశాంశం *>(src0->దత్తాంశం);
    స్థిరరాశి దశాంశం * B_d = static_cast<స్థిరరాశి దశాంశం *>(src1->దత్తాంశం);
    దశాంశం * X_d       = static_cast<దశాంశం *>(dst->దత్తాంశం);

    ఒకవేళ (X_d != B_d) {
        స్థిరరాశి int64_t మొత్తం_elements = (int64_t)n * k * ne02 * ne03;
        stream->జ్ఞాపకం_నకలు(X_d, B_d, మొత్తం_elements * పరిమాణం(దశాంశం));
    }

    స్థిరరాశి int64_t nb02 = src0->nb[2];
    స్థిరరాశి int64_t nb03 = src0->nb[3];
    స్థిరరాశి int64_t nb12 = src1->nb[2];
    స్థిరరాశి int64_t nb13 = src1->nb[3];
    స్థిరరాశి int64_t nb2  = dst->nb[2];
    స్థిరరాశి int64_t nb3  = dst->nb[3];

    స్థిరరాశి int64_t మొత్తం_batches = ne02 * ne03;

    ఒకవేళ (n <= 2 * WARP_కొలత && k <= 32) {
        స్థిరరాశి పూర్ణాంకం smem_కొలత = 2 * WARP_కొలత * 2 * WARP_కొలత;
        స్థిరరాశి sycl::range<2> grid(1, మొత్తం_batches);
        స్థిరరాశి sycl::range<2> ఖండం(k, WARP_కొలత);
        stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<దశాంశం, 1> smem_acc(sycl::range<1>(smem_కొలత), cgh);
            cgh.parallel_for(
                sycl::nd_range<2>(grid * ఖండం, ఖండం),
                [=](sycl::nd_అంశం<2> అంశం) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                    solve_tri_f32_fast<0, 0>(A_d, B_d, X_d, ne02, ne03,
                                              nb02, nb03, nb12, nb13, nb2, nb3,
                                              n, k, అంశం, get_సూచిక(smem_acc));
                });
        });
    } లేకపోతే {
        solve_tri_f32_mkl(stream, A_d, X_d, n, k, ne02, ne03, nb02, nb03, nb2, nb3);
    }
}

శూన్య ggml_sycl_solve_tri(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_solve_tri(ctx, dst);
}
