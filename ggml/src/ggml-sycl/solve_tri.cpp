#அந்தர்பாவ "solve_tri.hpp"
#அந்தர்பாவ "common.hpp"
#அந்தர்பாவ <oneapi/mkl/blas.hpp>

template <முழு n_template, முழு k_template>
நிலையான வெறுமை solve_tri_f32_fast(மாறிலி தசம * __restrict__ A,
                               மாறிலி தசம * __restrict__ B,
                               தசம * __restrict__ X,
                               மாறிலி int64_t ne02, [[maybe_unused]] மாறிலி int64_t ne03,
                               மாறிலி int64_t nb02, மாறிலி int64_t nb03,
                               மாறிலி int64_t nb12, மாறிலி int64_t nb13,
                               மாறிலி int64_t nb2,  மாறிலி int64_t nb3,
                               மாறிலி முழு n_arg, மாறிலி முழு k_arg,
                               மாறிலி sycl::nd_item<2> & item, தசம * sA) {

    மாறிலி முழு n = n_template == 0 ? n_arg : n_template;
    மாறிலி முழு k = k_template == 0 ? k_arg : k_template;

    மாறிலி முழு batch_idx = item.get_group(1);
    மாறிலி முழு lane      = item.get_local_id(1) % WARP_SIZE;
    மாறிலி முழு col_idx   = item.get_local_id(0);

    எனில் (col_idx >= k) {
        திரும்பு;
    }

    மாறிலி int64_t i03 = batch_idx / ne02;
    மாறிலி int64_t i02 = batch_idx % ne02;

    மாறிலி தசம * A_batch = (மாறிலி தசம *) ((மாறிலி எழுத்து *) A + i02 * nb02 + i03 * nb03);
    மாறிலி தசம * B_batch = (மாறிலி தசம *) ((மாறிலி எழுத்து *) B + i02 * nb12 + i03 * nb13);
    தசம *       X_batch = (தசம *)       ((எழுத்து *)       X + i02 * nb2  + i03 * nb3);

    மாறிலி முழு offset = item.get_local_id(1) + item.get_local_id(0) * item.get_local_range(1);

#pragma unroll
    க்கு (முழு i = 0; i < n * n; i += k * WARP_SIZE) {
        மாறிலி முழு i0 = i + offset;
        எனில் (i0 < n * n) {
            sA[i0] = A_batch[i0];
        }
    }

    item.barrier(sycl::access::fence_space::local_space);

    தசம x_கீழ்  = (lane < n) ? B_batch[lane * k + col_idx] : 0.0f;
    தசம x_உயர்வு = (WARP_SIZE + lane < n) ? B_batch[(WARP_SIZE + lane) * k + col_idx] : 0.0f;

    மாறிலி முழு half      = WARP_SIZE;
    மாறிலி முழு nrows_கீழ் = (n < half) ? n : half;

#pragma unroll
    க்கு (முழு row = 0; row < nrows_கீழ்; ++row) {
        தசம sum = 0.0f;
        எனில் (lane < row) {
            sum += sA[row * n + lane] * x_கீழ்;
        }
        sum = warp_reduce_sum<WARP_SIZE>(sum);
        எனில் (lane == row) {
            x_கீழ் = (x_கீழ் - sum) / sA[row * n + row];
        }
    }

#pragma unroll
    க்கு (முழு row = half; row < n; ++row) {
        தசம     sum = sA[row * n + lane] * x_கீழ்;
        மாறிலி முழு j   = half + lane;
        எனில் (j < row) {
            sum += sA[row * n + j] * x_உயர்வு;
        }
        sum = warp_reduce_sum<WARP_SIZE>(sum);
        எனில் (lane == row - half) {
            x_உயர்வு = (x_உயர்வு - sum) / sA[row * n + row];
        }
    }

#pragma unroll
    க்கு (முழு rr = 0; rr < 2; ++rr) {
        மாறிலி முழு row = rr * WARP_SIZE + lane;
        எனில் (row < n) {
            மாறிலி தசம val            = (row < half) ? x_கீழ் : x_உயர்வு;
            X_batch[row * k + col_idx] = val;
        }
    }
}

நிலையான வெறுமை solve_tri_f32_mkl(dpct::queue_ptr stream,
                               மாறிலி தசம * A, தசம * X,
                               முழு n, முழு k,
                               int64_t ne02, [[maybe_unused]] int64_t ne03,
                               int64_t nb02, [[maybe_unused]] int64_t nb03,
                               int64_t nb2,  [[maybe_unused]] int64_t nb3) {
    மாறிலி தசம alpha = 1.0f;
    மாறிலி int64_t total_batches = ne02 * ne03;
    எனில் (total_batches == 0) {
        திரும்பு;
    }

    மாறிலி int64_t stride_a = nb02 / அளவு(தசம);
    மாறிலி int64_t stride_x = nb2 / அளவு(தசம);

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

உள்வரி வெறுமை ggml_sycl_op_solve_tri(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(src0->type == GGML_TYPE_F32);

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    மாறிலி முழு n    = src0->ne[0];
    மாறிலி முழு k    = src1->ne[0];
    மாறிலி int64_t ne02 = src0->ne[2];
    மாறிலி int64_t ne03 = src0->ne[3];

    GGML_ASSERT(n <= SYCL_SOLVE_TRI_MAX_N && k <= SYCL_SOLVE_TRI_MAX_K);

    மாறிலி தசம * A_d = static_cast<மாறிலி தசம *>(src0->data);
    மாறிலி தசம * B_d = static_cast<மாறிலி தசம *>(src1->data);
    தசம * X_d       = static_cast<தசம *>(dst->data);

    எனில் (X_d != B_d) {
        மாறிலி int64_t total_elements = (int64_t)n * k * ne02 * ne03;
        stream->memcpy(X_d, B_d, total_elements * அளவு(தசம));
    }

    மாறிலி int64_t nb02 = src0->nb[2];
    மாறிலி int64_t nb03 = src0->nb[3];
    மாறிலி int64_t nb12 = src1->nb[2];
    மாறிலி int64_t nb13 = src1->nb[3];
    மாறிலி int64_t nb2  = dst->nb[2];
    மாறிலி int64_t nb3  = dst->nb[3];

    மாறிலி int64_t total_batches = ne02 * ne03;

    எனில் (n <= 2 * WARP_SIZE && k <= 32) {
        மாறிலி முழு smem_size = 2 * WARP_SIZE * 2 * WARP_SIZE;
        மாறிலி sycl::range<2> grid(1, total_batches);
        மாறிலி sycl::range<2> block(k, WARP_SIZE);
        stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<தசம, 1> smem_acc(sycl::range<1>(smem_size), cgh);
            cgh.parallel_for(
                sycl::nd_range<2>(grid * block, block),
                [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    solve_tri_f32_fast<0, 0>(A_d, B_d, X_d, ne02, ne03,
                                              nb02, nb03, nb12, nb13, nb2, nb3,
                                              n, k, item, get_pointer(smem_acc));
                });
        });
    } இல்லையெனில் {
        solve_tri_f32_mkl(stream, A_d, X_d, n, k, ne02, ne03, nb02, nb03, nb2, nb3);
    }
}

வெறுமை ggml_sycl_solve_tri(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_solve_tri(ctx, dst);
}
