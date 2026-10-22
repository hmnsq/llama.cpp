#ಅನ್ತರ್ಭಾವ "solve_tri.hpp"
#ಅನ್ತರ್ಭಾವ "common.hpp"
#ಅನ್ತರ್ಭಾವ <oneapi/mkl/blas.hpp>

template <ಪೂರ್ಣಾಙ್ಕ n_template, ಪೂರ್ಣಾಙ್ಕ k_template>
ಸ್ಥಿರ ಶೂನ್ಯ solve_tri_f32_fast(ಧ್ರುವ ಭಿನ್ನ * __restrict__ A,
                               ಧ್ರುವ ಭಿನ್ನ * __restrict__ B,
                               ಭಿನ್ನ * __restrict__ X,
                               ಧ್ರುವ int64_t ne02, [[maybe_unused]] ಧ್ರುವ int64_t ne03,
                               ಧ್ರುವ int64_t nb02, ಧ್ರುವ int64_t nb03,
                               ಧ್ರುವ int64_t nb12, ಧ್ರುವ int64_t nb13,
                               ಧ್ರುವ int64_t nb2,  ಧ್ರುವ int64_t nb3,
                               ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_arg, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ k_arg,
                               ಧ್ರುವ sycl::nd_item<2> & item, ಭಿನ್ನ * sA) {

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n = n_template == 0 ? n_arg : n_template;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ k = k_template == 0 ? k_arg : k_template;

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ batch_idx = item.get_group(1);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ lane      = item.get_local_id(1) % WARP_SIZE;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ col_idx   = item.get_local_id(0);

    ಯದಿ (col_idx >= k) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಧ್ರುವ int64_t i03 = batch_idx / ne02;
    ಧ್ರುವ int64_t i02 = batch_idx % ne02;

    ಧ್ರುವ ಭಿನ್ನ * A_batch = (ಧ್ರುವ ಭಿನ್ನ *) ((ಧ್ರುವ ವರ್ಣ *) A + i02 * nb02 + i03 * nb03);
    ಧ್ರುವ ಭಿನ್ನ * B_batch = (ಧ್ರುವ ಭಿನ್ನ *) ((ಧ್ರುವ ವರ್ಣ *) B + i02 * nb12 + i03 * nb13);
    ಭಿನ್ನ *       X_batch = (ಭಿನ್ನ *)       ((ವರ್ಣ *)       X + i02 * nb2  + i03 * nb3);

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ offset = item.get_local_id(1) + item.get_local_id(0) * item.get_local_range(1);

#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < n * n; i += k * WARP_SIZE) {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i0 = i + offset;
        ಯದಿ (i0 < n * n) {
            sA[i0] = A_batch[i0];
        }
    }

    item.barrier(sycl::access::fence_space::local_space);

    ಭಿನ್ನ x_ಕೆಳಗಿನ  = (lane < n) ? B_batch[lane * k + col_idx] : 0.0f;
    ಭಿನ್ನ x_ಮೇಲಿನ = (WARP_SIZE + lane < n) ? B_batch[(WARP_SIZE + lane) * k + col_idx] : 0.0f;

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ half      = WARP_SIZE;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nrows_ಕೆಳಗಿನ = (n < half) ? n : half;

#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ row = 0; row < nrows_ಕೆಳಗಿನ; ++row) {
        ಭಿನ್ನ sum = 0.0f;
        ಯದಿ (lane < row) {
            sum += sA[row * n + lane] * x_ಕೆಳಗಿನ;
        }
        sum = warp_reduce_sum<WARP_SIZE>(sum);
        ಯದಿ (lane == row) {
            x_ಕೆಳಗಿನ = (x_ಕೆಳಗಿನ - sum) / sA[row * n + row];
        }
    }

#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ row = half; row < n; ++row) {
        ಭಿನ್ನ     sum = sA[row * n + lane] * x_ಕೆಳಗಿನ;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ j   = half + lane;
        ಯದಿ (j < row) {
            sum += sA[row * n + j] * x_ಮೇಲಿನ;
        }
        sum = warp_reduce_sum<WARP_SIZE>(sum);
        ಯದಿ (lane == row - half) {
            x_ಮೇಲಿನ = (x_ಮೇಲಿನ - sum) / sA[row * n + row];
        }
    }

#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ rr = 0; rr < 2; ++rr) {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ row = rr * WARP_SIZE + lane;
        ಯದಿ (row < n) {
            ಧ್ರುವ ಭಿನ್ನ val            = (row < half) ? x_ಕೆಳಗಿನ : x_ಮೇಲಿನ;
            X_batch[row * k + col_idx] = val;
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ solve_tri_f32_mkl(dpct::queue_ptr stream,
                               ಧ್ರುವ ಭಿನ್ನ * A, ಭಿನ್ನ * X,
                               ಪೂರ್ಣಾಙ್ಕ n, ಪೂರ್ಣಾಙ್ಕ k,
                               int64_t ne02, [[maybe_unused]] int64_t ne03,
                               int64_t nb02, [[maybe_unused]] int64_t nb03,
                               int64_t nb2,  [[maybe_unused]] int64_t nb3) {
    ಧ್ರುವ ಭಿನ್ನ alpha = 1.0f;
    ಧ್ರುವ int64_t total_batches = ne02 * ne03;
    ಯದಿ (total_batches == 0) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಧ್ರುವ int64_t stride_a = nb02 / ಪರಿಮಾಣ(ಭಿನ್ನ);
    ಧ್ರುವ int64_t stride_x = nb2 / ಪರಿಮಾಣ(ಭಿನ್ನ);

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

ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ ggml_sycl_op_solve_tri(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ಧ್ರುವ ggml_tensor * src0 = dst->src[0];
    ಧ್ರುವ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(src0->type == GGML_TYPE_F32);

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n    = src0->ne[0];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ k    = src1->ne[0];
    ಧ್ರುವ int64_t ne02 = src0->ne[2];
    ಧ್ರುವ int64_t ne03 = src0->ne[3];

    GGML_ASSERT(n <= SYCL_SOLVE_TRI_MAX_N && k <= SYCL_SOLVE_TRI_MAX_K);

    ಧ್ರುವ ಭಿನ್ನ * A_d = static_cast<ಧ್ರುವ ಭಿನ್ನ *>(src0->data);
    ಧ್ರುವ ಭಿನ್ನ * B_d = static_cast<ಧ್ರುವ ಭಿನ್ನ *>(src1->data);
    ಭಿನ್ನ * X_d       = static_cast<ಭಿನ್ನ *>(dst->data);

    ಯದಿ (X_d != B_d) {
        ಧ್ರುವ int64_t total_elements = (int64_t)n * k * ne02 * ne03;
        stream->memcpy(X_d, B_d, total_elements * ಪರಿಮಾಣ(ಭಿನ್ನ));
    }

    ಧ್ರುವ int64_t nb02 = src0->nb[2];
    ಧ್ರುವ int64_t nb03 = src0->nb[3];
    ಧ್ರುವ int64_t nb12 = src1->nb[2];
    ಧ್ರುವ int64_t nb13 = src1->nb[3];
    ಧ್ರುವ int64_t nb2  = dst->nb[2];
    ಧ್ರುವ int64_t nb3  = dst->nb[3];

    ಧ್ರುವ int64_t total_batches = ne02 * ne03;

    ಯದಿ (n <= 2 * WARP_SIZE && k <= 32) {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ smem_size = 2 * WARP_SIZE * 2 * WARP_SIZE;
        ಧ್ರುವ sycl::range<2> grid(1, total_batches);
        ಧ್ರುವ sycl::range<2> block(k, WARP_SIZE);
        stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<ಭಿನ್ನ, 1> smem_acc(sycl::range<1>(smem_size), cgh);
            cgh.parallel_for(
                sycl::nd_range<2>(grid * block, block),
                [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    solve_tri_f32_fast<0, 0>(A_d, B_d, X_d, ne02, ne03,
                                              nb02, nb03, nb12, nb13, nb2, nb3,
                                              n, k, item, get_pointer(smem_acc));
                });
        });
    } ಅನ್ಯಥಾ {
        solve_tri_f32_mkl(stream, A_d, X_d, n, k, ne02, ne03, nb02, nb03, nb2, nb3);
    }
}

ಶೂನ್ಯ ggml_sycl_solve_tri(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_solve_tri(ctx, dst);
}
