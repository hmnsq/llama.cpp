#അന്തര്ഭാവ "solve_tri.hpp"
#അന്തര്ഭാവ "common.hpp"
#അന്തര്ഭാവ <oneapi/mkl/blas.hpp>

template <പൂർണ്ണസംഖ്യ n_template, പൂർണ്ണസംഖ്യ k_template>
സ്ഥിതം ശൂന്യം solve_tri_f32_fast(സ്ഥിരം ദശാംശം * __restrict__ A,
                               സ്ഥിരം ദശാംശം * __restrict__ B,
                               ദശാംശം * __restrict__ X,
                               സ്ഥിരം int64_t ne02, [[maybe_unused]] സ്ഥിരം int64_t ne03,
                               സ്ഥിരം int64_t nb02, സ്ഥിരം int64_t nb03,
                               സ്ഥിരം int64_t nb12, സ്ഥിരം int64_t nb13,
                               സ്ഥിരം int64_t nb2,  സ്ഥിരം int64_t nb3,
                               സ്ഥിരം പൂർണ്ണസംഖ്യ n_arg, സ്ഥിരം പൂർണ്ണസംഖ്യ k_arg,
                               സ്ഥിരം sycl::nd_item<2> & item, ദശാംശം * sA) {

    സ്ഥിരം പൂർണ്ണസംഖ്യ n = n_template == 0 ? n_arg : n_template;
    സ്ഥിരം പൂർണ്ണസംഖ്യ k = k_template == 0 ? k_arg : k_template;

    സ്ഥിരം പൂർണ്ണസംഖ്യ batch_idx = item.get_group(1);
    സ്ഥിരം പൂർണ്ണസംഖ്യ lane      = item.get_local_id(1) % WARP_SIZE;
    സ്ഥിരം പൂർണ്ണസംഖ്യ col_idx   = item.get_local_id(0);

    എങ്കിൽ (col_idx >= k) {
        തിരികെ;
    }

    സ്ഥിരം int64_t i03 = batch_idx / ne02;
    സ്ഥിരം int64_t i02 = batch_idx % ne02;

    സ്ഥിരം ദശാംശം * A_batch = (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) A + i02 * nb02 + i03 * nb03);
    സ്ഥിരം ദശാംശം * B_batch = (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) B + i02 * nb12 + i03 * nb13);
    ദശാംശം *       X_batch = (ദശാംശം *)       ((അക്ഷരം *)       X + i02 * nb2  + i03 * nb3);

    സ്ഥിരം പൂർണ്ണസംഖ്യ offset = item.get_local_id(1) + item.get_local_id(0) * item.get_local_range(1);

#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < n * n; i += k * WARP_SIZE) {
        സ്ഥിരം പൂർണ്ണസംഖ്യ i0 = i + offset;
        എങ്കിൽ (i0 < n * n) {
            sA[i0] = A_batch[i0];
        }
    }

    item.barrier(sycl::access::fence_space::local_space);

    ദശാംശം x_താഴെയുള്ള  = (lane < n) ? B_batch[lane * k + col_idx] : 0.0f;
    ദശാംശം x_മുകളിലെ = (WARP_SIZE + lane < n) ? B_batch[(WARP_SIZE + lane) * k + col_idx] : 0.0f;

    സ്ഥിരം പൂർണ്ണസംഖ്യ half      = WARP_SIZE;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_താഴെയുള്ള = (n < half) ? n : half;

#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ row = 0; row < nrows_താഴെയുള്ള; ++row) {
        ദശാംശം sum = 0.0f;
        എങ്കിൽ (lane < row) {
            sum += sA[row * n + lane] * x_താഴെയുള്ള;
        }
        sum = warp_reduce_sum<WARP_SIZE>(sum);
        എങ്കിൽ (lane == row) {
            x_താഴെയുള്ള = (x_താഴെയുള്ള - sum) / sA[row * n + row];
        }
    }

#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ row = half; row < n; ++row) {
        ദശാംശം     sum = sA[row * n + lane] * x_താഴെയുള്ള;
        സ്ഥിരം പൂർണ്ണസംഖ്യ j   = half + lane;
        എങ്കിൽ (j < row) {
            sum += sA[row * n + j] * x_മുകളിലെ;
        }
        sum = warp_reduce_sum<WARP_SIZE>(sum);
        എങ്കിൽ (lane == row - half) {
            x_മുകളിലെ = (x_മുകളിലെ - sum) / sA[row * n + row];
        }
    }

#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ rr = 0; rr < 2; ++rr) {
        സ്ഥിരം പൂർണ്ണസംഖ്യ row = rr * WARP_SIZE + lane;
        എങ്കിൽ (row < n) {
            സ്ഥിരം ദശാംശം val            = (row < half) ? x_താഴെയുള്ള : x_മുകളിലെ;
            X_batch[row * k + col_idx] = val;
        }
    }
}

സ്ഥിതം ശൂന്യം solve_tri_f32_mkl(dpct::queue_ptr stream,
                               സ്ഥിരം ദശാംശം * A, ദശാംശം * X,
                               പൂർണ്ണസംഖ്യ n, പൂർണ്ണസംഖ്യ k,
                               int64_t ne02, [[maybe_unused]] int64_t ne03,
                               int64_t nb02, [[maybe_unused]] int64_t nb03,
                               int64_t nb2,  [[maybe_unused]] int64_t nb3) {
    സ്ഥിരം ദശാംശം alpha = 1.0f;
    സ്ഥിരം int64_t total_batches = ne02 * ne03;
    എങ്കിൽ (total_batches == 0) {
        തിരികെ;
    }

    സ്ഥിരം int64_t stride_a = nb02 / വലുപ്പം(ദശാംശം);
    സ്ഥിരം int64_t stride_x = nb2 / വലുപ്പം(ദശാംശം);

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

വരിയിൽ ശൂന്യം ggml_sycl_op_solve_tri(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(src0->type == GGML_TYPE_F32);

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    സ്ഥിരം പൂർണ്ണസംഖ്യ n    = src0->ne[0];
    സ്ഥിരം പൂർണ്ണസംഖ്യ k    = src1->ne[0];
    സ്ഥിരം int64_t ne02 = src0->ne[2];
    സ്ഥിരം int64_t ne03 = src0->ne[3];

    GGML_ASSERT(n <= SYCL_SOLVE_TRI_MAX_N && k <= SYCL_SOLVE_TRI_MAX_K);

    സ്ഥിരം ദശാംശം * A_d = static_cast<സ്ഥിരം ദശാംശം *>(src0->data);
    സ്ഥിരം ദശാംശം * B_d = static_cast<സ്ഥിരം ദശാംശം *>(src1->data);
    ദശാംശം * X_d       = static_cast<ദശാംശം *>(dst->data);

    എങ്കിൽ (X_d != B_d) {
        സ്ഥിരം int64_t total_elements = (int64_t)n * k * ne02 * ne03;
        stream->memcpy(X_d, B_d, total_elements * വലുപ്പം(ദശാംശം));
    }

    സ്ഥിരം int64_t nb02 = src0->nb[2];
    സ്ഥിരം int64_t nb03 = src0->nb[3];
    സ്ഥിരം int64_t nb12 = src1->nb[2];
    സ്ഥിരം int64_t nb13 = src1->nb[3];
    സ്ഥിരം int64_t nb2  = dst->nb[2];
    സ്ഥിരം int64_t nb3  = dst->nb[3];

    സ്ഥിരം int64_t total_batches = ne02 * ne03;

    എങ്കിൽ (n <= 2 * WARP_SIZE && k <= 32) {
        സ്ഥിരം പൂർണ്ണസംഖ്യ smem_size = 2 * WARP_SIZE * 2 * WARP_SIZE;
        സ്ഥിരം sycl::range<2> grid(1, total_batches);
        സ്ഥിരം sycl::range<2> block(k, WARP_SIZE);
        stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<ദശാംശം, 1> smem_acc(sycl::range<1>(smem_size), cgh);
            cgh.parallel_for(
                sycl::nd_range<2>(grid * block, block),
                [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    solve_tri_f32_fast<0, 0>(A_d, B_d, X_d, ne02, ne03,
                                              nb02, nb03, nb12, nb13, nb2, nb3,
                                              n, k, item, get_pointer(smem_acc));
                });
        });
    } അല്ലെങ്കിൽ {
        solve_tri_f32_mkl(stream, A_d, X_d, n, k, ne02, ne03, nb02, nb03, nb2, nb3);
    }
}

ശൂന്യം ggml_sycl_solve_tri(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_solve_tri(ctx, dst);
}
