#ਅਨ੍ਤਰ੍ਭਾਵ "solve_tri.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "common.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ <oneapi/mkl/blas.hpp>

ਸਾਂਚਾ <ਪੂਰਨ n_template, ਪੂਰਨ k_template>
ਸਥਿਰ ਖਾਲੀ solve_tri_f32_fast(ਅਚਲ ਦਸ਼ਮਲਵ * __restrict__ A,
                               ਅਚਲ ਦਸ਼ਮਲਵ * __restrict__ B,
                               ਦਸ਼ਮਲਵ * __restrict__ X,
                               ਅਚਲ int64_t ne02, [[maybe_unused]] ਅਚਲ int64_t ne03,
                               ਅਚਲ int64_t nb02, ਅਚਲ int64_t nb03,
                               ਅਚਲ int64_t nb12, ਅਚਲ int64_t nb13,
                               ਅਚਲ int64_t nb2,  ਅਚਲ int64_t nb3,
                               ਅਚਲ ਪੂਰਨ n_arg, ਅਚਲ ਪੂਰਨ k_arg,
                               ਅਚਲ sycl::nd_ਵਸਤੂ<2> & ਵਸਤੂ, ਦਸ਼ਮਲਵ * sA) {

    ਅਚਲ ਪੂਰਨ n = n_template == 0 ? n_arg : n_template;
    ਅਚਲ ਪੂਰਨ k = k_template == 0 ? k_arg : k_template;

    ਅਚਲ ਪੂਰਨ batch_idx = ਵਸਤੂ.get_ਸਮੂਹ(1);
    ਅਚਲ ਪੂਰਨ lane      = ਵਸਤੂ.get_local_id(1) % WARP_ਮਾਪ;
    ਅਚਲ ਪੂਰਨ ਖਾਨਾ_idx   = ਵਸਤੂ.get_local_id(0);

    ਜੇ (ਖਾਨਾ_idx >= k) {
        ਵਾਪਸ;
    }

    ਅਚਲ int64_t i03 = batch_idx / ne02;
    ਅਚਲ int64_t i02 = batch_idx % ne02;

    ਅਚਲ ਦਸ਼ਮਲਵ * A_batch = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) A + i02 * nb02 + i03 * nb03);
    ਅਚਲ ਦਸ਼ਮਲਵ * B_batch = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) B + i02 * nb12 + i03 * nb13);
    ਦਸ਼ਮਲਵ *       X_batch = (ਦਸ਼ਮਲਵ *)       ((ਅੱਖਰ *)       X + i02 * nb2  + i03 * nb3);

    ਅਚਲ ਪੂਰਨ ਵਿੱਥ = ਵਸਤੂ.get_local_id(1) + ਵਸਤੂ.get_local_id(0) * ਵਸਤੂ.get_local_range(1);

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i = 0; i < n * n; i += k * WARP_ਮਾਪ) {
        ਅਚਲ ਪੂਰਨ i0 = i + ਵਿੱਥ;
        ਜੇ (i0 < n * n) {
            sA[i0] = A_batch[i0];
        }
    }

    ਵਸਤੂ.barrier(sycl::access::fence_space::local_space);

    ਦਸ਼ਮਲਵ x_ਨੀਵਾਂ  = (lane < n) ? B_batch[lane * k + ਖਾਨਾ_idx] : 0.0f;
    ਦਸ਼ਮਲਵ x_ਉੱਚਾ = (WARP_ਮਾਪ + lane < n) ? B_batch[(WARP_ਮਾਪ + lane) * k + ਖਾਨਾ_idx] : 0.0f;

    ਅਚਲ ਪੂਰਨ half      = WARP_ਮਾਪ;
    ਅਚਲ ਪੂਰਨ nrows_ਨੀਵਾਂ = (n < half) ? n : half;

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ ਪੰਕਤੀ = 0; ਪੰਕਤੀ < nrows_ਨੀਵਾਂ; ++ਪੰਕਤੀ) {
        ਦਸ਼ਮਲਵ ਜੋੜ = 0.0f;
        ਜੇ (lane < ਪੰਕਤੀ) {
            ਜੋੜ += sA[ਪੰਕਤੀ * n + lane] * x_ਨੀਵਾਂ;
        }
        ਜੋੜ = warp_reduce_ਜੋੜ<WARP_ਮਾਪ>(ਜੋੜ);
        ਜੇ (lane == ਪੰਕਤੀ) {
            x_ਨੀਵਾਂ = (x_ਨੀਵਾਂ - ਜੋੜ) / sA[ਪੰਕਤੀ * n + ਪੰਕਤੀ];
        }
    }

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ ਪੰਕਤੀ = half; ਪੰਕਤੀ < n; ++ਪੰਕਤੀ) {
        ਦਸ਼ਮਲਵ     ਜੋੜ = sA[ਪੰਕਤੀ * n + lane] * x_ਨੀਵਾਂ;
        ਅਚਲ ਪੂਰਨ j   = half + lane;
        ਜੇ (j < ਪੰਕਤੀ) {
            ਜੋੜ += sA[ਪੰਕਤੀ * n + j] * x_ਉੱਚਾ;
        }
        ਜੋੜ = warp_reduce_ਜੋੜ<WARP_ਮਾਪ>(ਜੋੜ);
        ਜੇ (lane == ਪੰਕਤੀ - half) {
            x_ਉੱਚਾ = (x_ਉੱਚਾ - ਜੋੜ) / sA[ਪੰਕਤੀ * n + ਪੰਕਤੀ];
        }
    }

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ rr = 0; rr < 2; ++rr) {
        ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ = rr * WARP_ਮਾਪ + lane;
        ਜੇ (ਪੰਕਤੀ < n) {
            ਅਚਲ ਦਸ਼ਮਲਵ val            = (ਪੰਕਤੀ < half) ? x_ਨੀਵਾਂ : x_ਉੱਚਾ;
            X_batch[ਪੰਕਤੀ * k + ਖਾਨਾ_idx] = val;
        }
    }
}

ਸਥਿਰ ਖਾਲੀ solve_tri_f32_mkl(dpct::ਕਤਾਰ_ptr stream,
                               ਅਚਲ ਦਸ਼ਮਲਵ * A, ਦਸ਼ਮਲਵ * X,
                               ਪੂਰਨ n, ਪੂਰਨ k,
                               int64_t ne02, [[maybe_unused]] int64_t ne03,
                               int64_t nb02, [[maybe_unused]] int64_t nb03,
                               int64_t nb2,  [[maybe_unused]] int64_t nb3) {
    ਅਚਲ ਦਸ਼ਮਲਵ alpha = 1.0f;
    ਅਚਲ int64_t ਕੁੱਲ_batches = ne02 * ne03;
    ਜੇ (ਕੁੱਲ_batches == 0) {
        ਵਾਪਸ;
    }

    ਅਚਲ int64_t stride_a = nb02 / ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ int64_t stride_x = nb2 / ਆਕਾਰ(ਦਸ਼ਮਲਵ);

    oneapi::mkl::blas::trsm_batch(
        *stream,
        oneapi::mkl::side::ਸੱਜੇ,
        oneapi::mkl::uplo::upper,
        oneapi::mkl::transpose::nontrans,
        oneapi::mkl::diag::nonunit,
        k, n, alpha,
        A, n, stride_a,
        X, k, stride_x,
        ਕੁੱਲ_batches);
}

ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_sycl_op_solve_tri(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(src0->type == GGML_TYPE_F32);

    dpct::ਕਤਾਰ_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_ਸੈੱਟ_device(ctx.device));

    ਅਚਲ ਪੂਰਨ n    = src0->ne[0];
    ਅਚਲ ਪੂਰਨ k    = src1->ne[0];
    ਅਚਲ int64_t ne02 = src0->ne[2];
    ਅਚਲ int64_t ne03 = src0->ne[3];

    GGML_ASSERT(n <= SYCL_SOLVE_TRI_MAX_N && k <= SYCL_SOLVE_TRI_MAX_K);

    ਅਚਲ ਦਸ਼ਮਲਵ * A_d = static_cast<ਅਚਲ ਦਸ਼ਮਲਵ *>(src0->ਅੰਕੜੇ);
    ਅਚਲ ਦਸ਼ਮਲਵ * B_d = static_cast<ਅਚਲ ਦਸ਼ਮਲਵ *>(src1->ਅੰਕੜੇ);
    ਦਸ਼ਮਲਵ * X_d       = static_cast<ਦਸ਼ਮਲਵ *>(dst->ਅੰਕੜੇ);

    ਜੇ (X_d != B_d) {
        ਅਚਲ int64_t ਕੁੱਲ_elements = (int64_t)n * k * ne02 * ne03;
        stream->ਸਮ੍ਰਿਤੀ_ਨਕਲ(X_d, B_d, ਕੁੱਲ_elements * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    }

    ਅਚਲ int64_t nb02 = src0->nb[2];
    ਅਚਲ int64_t nb03 = src0->nb[3];
    ਅਚਲ int64_t nb12 = src1->nb[2];
    ਅਚਲ int64_t nb13 = src1->nb[3];
    ਅਚਲ int64_t nb2  = dst->nb[2];
    ਅਚਲ int64_t nb3  = dst->nb[3];

    ਅਚਲ int64_t ਕੁੱਲ_batches = ne02 * ne03;

    ਜੇ (n <= 2 * WARP_ਮਾਪ && k <= 32) {
        ਅਚਲ ਪੂਰਨ smem_ਮਾਪ = 2 * WARP_ਮਾਪ * 2 * WARP_ਮਾਪ;
        ਅਚਲ sycl::range<2> grid(1, ਕੁੱਲ_batches);
        ਅਚਲ sycl::range<2> ਖੰਡ(k, WARP_ਮਾਪ);
        stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<ਦਸ਼ਮਲਵ, 1> smem_acc(sycl::range<1>(smem_ਮਾਪ), cgh);
            cgh.parallel_for(
                sycl::nd_range<2>(grid * ਖੰਡ, ਖੰਡ),
                [=](sycl::nd_ਵਸਤੂ<2> ਵਸਤੂ) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                    solve_tri_f32_fast<0, 0>(A_d, B_d, X_d, ne02, ne03,
                                              nb02, nb03, nb12, nb13, nb2, nb3,
                                              n, k, ਵਸਤੂ, get_ਸੂਚਕ(smem_acc));
                });
        });
    } ਨਹੀਂਤਾਂ {
        solve_tri_f32_mkl(stream, A_d, X_d, n, k, ne02, ne03, nb02, nb03, nb2, nb3);
    }
}

ਖਾਲੀ ggml_sycl_solve_tri(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_solve_tri(ctx, dst);
}
