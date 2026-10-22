#અન્તર્ભાવ "solve_tri.hpp"
#અન્તર્ભાવ "common.hpp"
#અન્તર્ભાવ <oneapi/mkl/blas.hpp>

template <પૂર્ણાંક n_template, પૂર્ણાંક k_template>
સ્થિર શૂન્ય solve_tri_f32_fast(અચલ દશાંશ * __restrict__ A,
                               અચલ દશાંશ * __restrict__ B,
                               દશાંશ * __restrict__ X,
                               અચલ int64_t ne02, [[maybe_unused]] અચલ int64_t ne03,
                               અચલ int64_t nb02, અચલ int64_t nb03,
                               અચલ int64_t nb12, અચલ int64_t nb13,
                               અચલ int64_t nb2,  અચલ int64_t nb3,
                               અચલ પૂર્ણાંક n_arg, અચલ પૂર્ણાંક k_arg,
                               અચલ sycl::nd_item<2> & item, દશાંશ * sA) {

    અચલ પૂર્ણાંક n = n_template == 0 ? n_arg : n_template;
    અચલ પૂર્ણાંક k = k_template == 0 ? k_arg : k_template;

    અચલ પૂર્ણાંક batch_idx = item.get_group(1);
    અચલ પૂર્ણાંક lane      = item.get_local_id(1) % WARP_SIZE;
    અચલ પૂર્ણાંક col_idx   = item.get_local_id(0);

    જો (col_idx >= k) {
        પાછા;
    }

    અચલ int64_t i03 = batch_idx / ne02;
    અચલ int64_t i02 = batch_idx % ne02;

    અચલ દશાંશ * A_batch = (અચલ દશાંશ *) ((અચલ અક્ષર *) A + i02 * nb02 + i03 * nb03);
    અચલ દશાંશ * B_batch = (અચલ દશાંશ *) ((અચલ અક્ષર *) B + i02 * nb12 + i03 * nb13);
    દશાંશ *       X_batch = (દશાંશ *)       ((અક્ષર *)       X + i02 * nb2  + i03 * nb3);

    અચલ પૂર્ણાંક offset = item.get_local_id(1) + item.get_local_id(0) * item.get_local_range(1);

#pragma unroll
    માટે (પૂર્ણાંક i = 0; i < n * n; i += k * WARP_SIZE) {
        અચલ પૂર્ણાંક i0 = i + offset;
        જો (i0 < n * n) {
            sA[i0] = A_batch[i0];
        }
    }

    item.barrier(sycl::access::fence_space::local_space);

    દશાંશ x_નીચલું  = (lane < n) ? B_batch[lane * k + col_idx] : 0.0f;
    દશાંશ x_ઉપલું = (WARP_SIZE + lane < n) ? B_batch[(WARP_SIZE + lane) * k + col_idx] : 0.0f;

    અચલ પૂર્ણાંક half      = WARP_SIZE;
    અચલ પૂર્ણાંક nrows_નીચલું = (n < half) ? n : half;

#pragma unroll
    માટે (પૂર્ણાંક row = 0; row < nrows_નીચલું; ++row) {
        દશાંશ sum = 0.0f;
        જો (lane < row) {
            sum += sA[row * n + lane] * x_નીચલું;
        }
        sum = warp_reduce_sum<WARP_SIZE>(sum);
        જો (lane == row) {
            x_નીચલું = (x_નીચલું - sum) / sA[row * n + row];
        }
    }

#pragma unroll
    માટે (પૂર્ણાંક row = half; row < n; ++row) {
        દશાંશ     sum = sA[row * n + lane] * x_નીચલું;
        અચલ પૂર્ણાંક j   = half + lane;
        જો (j < row) {
            sum += sA[row * n + j] * x_ઉપલું;
        }
        sum = warp_reduce_sum<WARP_SIZE>(sum);
        જો (lane == row - half) {
            x_ઉપલું = (x_ઉપલું - sum) / sA[row * n + row];
        }
    }

#pragma unroll
    માટે (પૂર્ણાંક rr = 0; rr < 2; ++rr) {
        અચલ પૂર્ણાંક row = rr * WARP_SIZE + lane;
        જો (row < n) {
            અચલ દશાંશ val            = (row < half) ? x_નીચલું : x_ઉપલું;
            X_batch[row * k + col_idx] = val;
        }
    }
}

સ્થિર શૂન્ય solve_tri_f32_mkl(dpct::queue_ptr stream,
                               અચલ દશાંશ * A, દશાંશ * X,
                               પૂર્ણાંક n, પૂર્ણાંક k,
                               int64_t ne02, [[maybe_unused]] int64_t ne03,
                               int64_t nb02, [[maybe_unused]] int64_t nb03,
                               int64_t nb2,  [[maybe_unused]] int64_t nb3) {
    અચલ દશાંશ alpha = 1.0f;
    અચલ int64_t total_batches = ne02 * ne03;
    જો (total_batches == 0) {
        પાછા;
    }

    અચલ int64_t stride_a = nb02 / કદ(દશાંશ);
    અચલ int64_t stride_x = nb2 / કદ(દશાંશ);

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

અંતઃરેખ શૂન્ય ggml_sycl_op_solve_tri(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(src0->type == GGML_TYPE_F32);

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    અચલ પૂર્ણાંક n    = src0->ne[0];
    અચલ પૂર્ણાંક k    = src1->ne[0];
    અચલ int64_t ne02 = src0->ne[2];
    અચલ int64_t ne03 = src0->ne[3];

    GGML_ASSERT(n <= SYCL_SOLVE_TRI_MAX_N && k <= SYCL_SOLVE_TRI_MAX_K);

    અચલ દશાંશ * A_d = static_cast<અચલ દશાંશ *>(src0->data);
    અચલ દશાંશ * B_d = static_cast<અચલ દશાંશ *>(src1->data);
    દશાંશ * X_d       = static_cast<દશાંશ *>(dst->data);

    જો (X_d != B_d) {
        અચલ int64_t total_elements = (int64_t)n * k * ne02 * ne03;
        stream->memcpy(X_d, B_d, total_elements * કદ(દશાંશ));
    }

    અચલ int64_t nb02 = src0->nb[2];
    અચલ int64_t nb03 = src0->nb[3];
    અચલ int64_t nb12 = src1->nb[2];
    અચલ int64_t nb13 = src1->nb[3];
    અચલ int64_t nb2  = dst->nb[2];
    અચલ int64_t nb3  = dst->nb[3];

    અચલ int64_t total_batches = ne02 * ne03;

    જો (n <= 2 * WARP_SIZE && k <= 32) {
        અચલ પૂર્ણાંક smem_size = 2 * WARP_SIZE * 2 * WARP_SIZE;
        અચલ sycl::range<2> grid(1, total_batches);
        અચલ sycl::range<2> block(k, WARP_SIZE);
        stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<દશાંશ, 1> smem_acc(sycl::range<1>(smem_size), cgh);
            cgh.parallel_for(
                sycl::nd_range<2>(grid * block, block),
                [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    solve_tri_f32_fast<0, 0>(A_d, B_d, X_d, ne02, ne03,
                                              nb02, nb03, nb12, nb13, nb2, nb3,
                                              n, k, item, get_pointer(smem_acc));
                });
        });
    } નહીંતર {
        solve_tri_f32_mkl(stream, A_d, X_d, n, k, ne02, ne03, nb02, nb03, nb2, nb3);
    }
}

શૂન્ય ggml_sycl_solve_tri(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_solve_tri(ctx, dst);
}
