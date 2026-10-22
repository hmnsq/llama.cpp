#ಅನ್ತರ್ಭಾವ "solve_tri.hpp"
#ಅನ್ತರ್ಭಾವ "common.hpp"
#ಅನ್ತರ್ಭಾವ <oneapi/mkl/blas.hpp>

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ n_template, ಪೂರ್ಣಾಂಕ k_template>
ಸ್ಥಿರ ಶೂನ್ಯ solve_tri_f32_fast(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * __restrict__ A,
                               ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * __restrict__ B,
                               ದಶಮಾಂಶ * __restrict__ X,
                               ಸ್ಥಿರಾಂಕ int64_t ne02, [[maybe_unused]] ಸ್ಥಿರಾಂಕ int64_t ne03,
                               ಸ್ಥಿರಾಂಕ int64_t nb02, ಸ್ಥಿರಾಂಕ int64_t nb03,
                               ಸ್ಥಿರಾಂಕ int64_t nb12, ಸ್ಥಿರಾಂಕ int64_t nb13,
                               ಸ್ಥಿರಾಂಕ int64_t nb2,  ಸ್ಥಿರಾಂಕ int64_t nb3,
                               ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_arg, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ k_arg,
                               ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<2> & ಅಂಶ, ದಶಮಾಂಶ * sA) {

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n = n_template == 0 ? n_arg : n_template;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ k = k_template == 0 ? k_arg : k_template;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ batch_idx = ಅಂಶ.get_ಗುಂಪು(1);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ lane      = ಅಂಶ.get_local_id(1) % WARP_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು_idx   = ಅಂಶ.get_local_id(0);

    ಆದರೆ (ಲಂಬಸಾಲು_idx >= k) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ int64_t i03 = batch_idx / ne02;
    ಸ್ಥಿರಾಂಕ int64_t i02 = batch_idx % ne02;

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * A_batch = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) A + i02 * nb02 + i03 * nb03);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * B_batch = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) B + i02 * nb12 + i03 * nb13);
    ದಶಮಾಂಶ *       X_batch = (ದಶಮಾಂಶ *)       ((ಅಕ್ಷರ *)       X + i02 * nb2  + i03 * nb3);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಸ್ಥಾನಾಂತರ = ಅಂಶ.get_local_id(1) + ಅಂಶ.get_local_id(0) * ಅಂಶ.get_local_range(1);

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n * n; i += k * WARP_ಗಾತ್ರ) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i0 = i + ಸ್ಥಾನಾಂತರ;
        ಆದರೆ (i0 < n * n) {
            sA[i0] = A_batch[i0];
        }
    }

    ಅಂಶ.barrier(sycl::access::fence_space::local_space);

    ದಶಮಾಂಶ x_ಕೆಳಗಿನ  = (lane < n) ? B_batch[lane * k + ಲಂಬಸಾಲು_idx] : 0.0f;
    ದಶಮಾಂಶ x_ಮೇಲಿನ = (WARP_ಗಾತ್ರ + lane < n) ? B_batch[(WARP_ಗಾತ್ರ + lane) * k + ಲಂಬಸಾಲು_idx] : 0.0f;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ half      = WARP_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_ಕೆಳಗಿನ = (n < half) ? n : half;

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = 0; ಅಡ್ಡಸಾಲು < nrows_ಕೆಳಗಿನ; ++ಅಡ್ಡಸಾಲು) {
        ದಶಮಾಂಶ ಮೊತ್ತ = 0.0f;
        ಆದರೆ (lane < ಅಡ್ಡಸಾಲು) {
            ಮೊತ್ತ += sA[ಅಡ್ಡಸಾಲು * n + lane] * x_ಕೆಳಗಿನ;
        }
        ಮೊತ್ತ = warp_reduce_ಮೊತ್ತ<WARP_ಗಾತ್ರ>(ಮೊತ್ತ);
        ಆದರೆ (lane == ಅಡ್ಡಸಾಲು) {
            x_ಕೆಳಗಿನ = (x_ಕೆಳಗಿನ - ಮೊತ್ತ) / sA[ಅಡ್ಡಸಾಲು * n + ಅಡ್ಡಸಾಲು];
        }
    }

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = half; ಅಡ್ಡಸಾಲು < n; ++ಅಡ್ಡಸಾಲು) {
        ದಶಮಾಂಶ     ಮೊತ್ತ = sA[ಅಡ್ಡಸಾಲು * n + lane] * x_ಕೆಳಗಿನ;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ j   = half + lane;
        ಆದರೆ (j < ಅಡ್ಡಸಾಲು) {
            ಮೊತ್ತ += sA[ಅಡ್ಡಸಾಲು * n + j] * x_ಮೇಲಿನ;
        }
        ಮೊತ್ತ = warp_reduce_ಮೊತ್ತ<WARP_ಗಾತ್ರ>(ಮೊತ್ತ);
        ಆದರೆ (lane == ಅಡ್ಡಸಾಲು - half) {
            x_ಮೇಲಿನ = (x_ಮೇಲಿನ - ಮೊತ್ತ) / sA[ಅಡ್ಡಸಾಲು * n + ಅಡ್ಡಸಾಲು];
        }
    }

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ rr = 0; rr < 2; ++rr) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = rr * WARP_ಗಾತ್ರ + lane;
        ಆದರೆ (ಅಡ್ಡಸಾಲು < n) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ val            = (ಅಡ್ಡಸಾಲು < half) ? x_ಕೆಳಗಿನ : x_ಮೇಲಿನ;
            X_batch[ಅಡ್ಡಸಾಲು * k + ಲಂಬಸಾಲು_idx] = val;
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ solve_tri_f32_mkl(dpct::ಸರತಿ_ptr stream,
                               ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * A, ದಶಮಾಂಶ * X,
                               ಪೂರ್ಣಾಂಕ n, ಪೂರ್ಣಾಂಕ k,
                               int64_t ne02, [[maybe_unused]] int64_t ne03,
                               int64_t nb02, [[maybe_unused]] int64_t nb03,
                               int64_t nb2,  [[maybe_unused]] int64_t nb3) {
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ alpha = 1.0f;
    ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_batches = ne02 * ne03;
    ಆದರೆ (ಒಟ್ಟು_batches == 0) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ int64_t stride_a = nb02 / ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ int64_t stride_x = nb2 / ಗಾತ್ರ(ದಶಮಾಂಶ);

    oneapi::mkl::blas::trsm_batch(
        *stream,
        oneapi::mkl::side::ಬಲ,
        oneapi::mkl::uplo::upper,
        oneapi::mkl::transpose::nontrans,
        oneapi::mkl::diag::nonunit,
        k, n, alpha,
        A, n, stride_a,
        X, k, stride_x,
        ಒಟ್ಟು_batches);
}

ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_sycl_op_solve_tri(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(src0->type == GGML_TYPE_F32);

    dpct::ಸರತಿ_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_ಗಣ_device(ctx.device));

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n    = src0->ne[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ k    = src1->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t ne02 = src0->ne[2];
    ಸ್ಥಿರಾಂಕ int64_t ne03 = src0->ne[3];

    GGML_ASSERT(n <= SYCL_SOLVE_TRI_MAX_N && k <= SYCL_SOLVE_TRI_MAX_K);

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * A_d = static_cast<ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *>(src0->ದತ್ತಾಂಶ);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * B_d = static_cast<ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *>(src1->ದತ್ತಾಂಶ);
    ದಶಮಾಂಶ * X_d       = static_cast<ದಶಮಾಂಶ *>(dst->ದತ್ತಾಂಶ);

    ಆದರೆ (X_d != B_d) {
        ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_elements = (int64_t)n * k * ne02 * ne03;
        stream->ಸ್ಮರಣೆ_ನಕಲು(X_d, B_d, ಒಟ್ಟು_elements * ಗಾತ್ರ(ದಶಮಾಂಶ));
    }

    ಸ್ಥಿರಾಂಕ int64_t nb02 = src0->nb[2];
    ಸ್ಥಿರಾಂಕ int64_t nb03 = src0->nb[3];
    ಸ್ಥಿರಾಂಕ int64_t nb12 = src1->nb[2];
    ಸ್ಥಿರಾಂಕ int64_t nb13 = src1->nb[3];
    ಸ್ಥಿರಾಂಕ int64_t nb2  = dst->nb[2];
    ಸ್ಥಿರಾಂಕ int64_t nb3  = dst->nb[3];

    ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_batches = ne02 * ne03;

    ಆದರೆ (n <= 2 * WARP_ಗಾತ್ರ && k <= 32) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ smem_ಗಾತ್ರ = 2 * WARP_ಗಾತ್ರ * 2 * WARP_ಗಾತ್ರ;
        ಸ್ಥಿರಾಂಕ sycl::range<2> grid(1, ಒಟ್ಟು_batches);
        ಸ್ಥಿರಾಂಕ sycl::range<2> ಖಂಡ(k, WARP_ಗಾತ್ರ);
        stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<ದಶಮಾಂಶ, 1> smem_acc(sycl::range<1>(smem_ಗಾತ್ರ), cgh);
            cgh.parallel_for(
                sycl::nd_range<2>(grid * ಖಂಡ, ಖಂಡ),
                [=](sycl::nd_ಅಂಶ<2> ಅಂಶ) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                    solve_tri_f32_fast<0, 0>(A_d, B_d, X_d, ne02, ne03,
                                              nb02, nb03, nb12, nb13, nb2, nb3,
                                              n, k, ಅಂಶ, get_ಸೂಚಕ(smem_acc));
                });
        });
    } ಇಲ್ಲದಿದ್ದರೆ {
        solve_tri_f32_mkl(stream, A_d, X_d, n, k, ne02, ne03, nb02, nb03, nb2, nb3);
    }
}

ಶೂನ್ಯ ggml_sycl_solve_tri(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_solve_tri(ctx, dst);
}
