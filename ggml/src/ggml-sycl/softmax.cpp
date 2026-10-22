#ಅನ್ತರ್ಭಾವ "softmax.hpp"
#ಅನ್ತರ್ಭಾವ <cstdint>
#ಅನ್ತರ್ಭಾವ <utility>
#ಅನ್ತರ್ಭಾವ <cmath>


template <typename T> ಸ್ಥಿರ __dpct_inline__ ಭಿನ್ನ t2f32(T val) {
    ಪ್ರತ್ಯಾಗಮ (ಭಿನ್ನ) val;
}

template <> ಭಿನ್ನ __dpct_inline__ t2f32<sycl::half>(sycl::half val) {
  ಪ್ರತ್ಯಾಗಮ sycl::vec<sycl::half, 1>(val)
      .convert<ಭಿನ್ನ, sycl::rounding_mode::automatic>()[0];
}

ಸಂರಚನಾ soft_max_params {

    int64_t nheads;
    uint32_t n_head_log2;
    int64_t ncols;
    int64_t nrows_x;
    int64_t nrows_y;
    int64_t ne00;
    int64_t ne01;
    int64_t ne02;
    int64_t ne03;
    int64_t nb11;
    int64_t nb12;
    int64_t nb13;

    int64_t ne12;
    int64_t ne13;
    ಭಿನ್ನ scale;
    ಭಿನ್ನ max_bias;
    ಭಿನ್ನ m0;
    ಭಿನ್ನ m1;
};

// When ncols_template == 0 the bounds for the loops in this function are not known and can't be unrolled.
// As we want to keep pragma unroll for all other cases we suppress the clang transformation warning here.
#ifdef __clang__
#pragma clang diagnostic ತಳ್ಳು
#pragma clang diagnostic ignored "-Wpass-failed"
#endif // __clang__
template <ತಾರ್ಕಿಕ use_shared, ಪೂರ್ಣಾಙ್ಕ ncols_template, ಪೂರ್ಣಾಙ್ಕ block_size_template, typename T>
ಸ್ಥಿರ ಶೂನ್ಯ soft_max_f32(ಧ್ರುವ ಭಿನ್ನ *         x,
                         ಧ್ರುವ T *             mask,
                         ಧ್ರುವ ಭಿನ್ನ *         sinks,
                         ಭಿನ್ನ *               dst,
                         ಧ್ರುವ soft_max_params p,
                         uint8_t *             dpct_local) {
    auto      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ncols    = ncols_template == 0 ? p.ncols : ncols_template;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ block_size = block_size_template == 0
                               ? item_ct1.get_local_range(2)
                               : block_size_template;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nthreads = block_size;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nwarps = nthreads / WARP_SIZE;
    ಧ್ರುವ size_t nreduce = nwarps / WARP_SIZE;

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ tid = item_ct1.get_local_id(2);

    ಧ್ರುವ int64_t i03 = item_ct1.get_group(0);
    ಧ್ರುವ int64_t i02 = item_ct1.get_group(1);
    ಧ್ರುವ int64_t i01 = item_ct1.get_group(2);

    //TODO: noncontigous inputs/outputs
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ rowx = item_ct1.get_group(2) +
                     item_ct1.get_group(1) * item_ct1.get_group_range(2) +
                     item_ct1.get_group(0) * item_ct1.get_group_range(2) *
                         item_ct1.get_group_range(1);

    ಧ್ರುವ int64_t i11 = i01;
    ಧ್ರುವ int64_t i12 = i02 % p.ne12;
    ಧ್ರುವ int64_t i13 = i03 % p.ne13;

    x    += int64_t(rowx)*ncols;
    mask += (i11*p.nb11 + i12*p.nb12 + i13*p.nb13) / ಪರಿಮಾಣ(T) * (mask != nullptr);
    dst  += int64_t(rowx)*ncols;

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ warp_id = item_ct1.get_local_id(2) / WARP_SIZE;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ lane_id = item_ct1.get_local_id(2) % WARP_SIZE;

    ಧ್ರುವ ಭಿನ್ನ slope = get_alibi_slope(p.max_bias, i02, p.n_head_log2, p.m0, p.m1);

    ಭಿನ್ನ * buf_iw = (ಭಿನ್ನ *) dpct_local;

    // shared memory buffer to cache values between iterations:
    ಭಿನ್ನ *vals = use_shared ? buf_iw + sycl::max(nwarps, WARP_SIZE) : dst;
    ಭಿನ್ನ max_val = sinks ? sinks[i02] : -INFINITY;
#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ col0 = 0; col0 < ncols; col0 += block_size) {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ col = col0 + tid;

        ಯದಿ (ncols_template == 0 && col >= ncols) {
            ಭಙ್ಗ;
        }

        ಧ್ರುವ ಭಿನ್ನ val = x[col]*p.scale + (mask ? slope*t2f32(mask[col]) : 0.0f);

        vals[col] = val;
        max_val   = sycl::max(max_val, val);
    }
    // find the max value in the block
    max_val = warp_reduce_max<WARP_SIZE>(max_val);

    ಯದಿ (block_size > WARP_SIZE) {
        ಯದಿ (lane_id == 0) {
            buf_iw[warp_id] = max_val;
        }
        item_ct1.barrier();

        max_val = -INFINITY;
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = lane_id; i < nwarps; i += WARP_SIZE) {
            max_val = sycl::max(max_val, buf_iw[i]);
        }
        max_val = warp_reduce_max<WARP_SIZE>(max_val);
    }
    ಭಿನ್ನ tmp = 0.0f; // partial sum

#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ col0 = 0; col0 < ncols; col0 += block_size) {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ col = col0 + tid;

        ಯದಿ (ncols_template == 0 && col >= ncols) {
            ಭಙ್ಗ;
        }

        ಧ್ರುವ ಭಿನ್ನ val = sycl::native::exp(sycl::max(vals[col] - max_val, -80.0f));
        tmp += val;
        vals[col] = val;
    }
    // find the sum of exps in the block
    tmp = warp_reduce_sum<WARP_SIZE>(tmp);
    ಯದಿ (block_size > WARP_SIZE) {
        item_ct1.barrier();
        ಯದಿ (warp_id == 0) {
            buf_iw[lane_id] = 0.0f;
            ಪ್ರತಿ (size_t i = 1; i < nreduce; i += 1) {
                buf_iw[lane_id + i * WARP_SIZE] = 0.f;
            }
        }
        item_ct1.barrier();

        ಯದಿ (lane_id == 0) {
            buf_iw[warp_id] = tmp;
        }
        item_ct1.barrier();

        tmp = buf_iw[lane_id];
        ಪ್ರತಿ (size_t i = 1; i < nreduce; i += 1) {
            tmp += buf_iw[lane_id + i * WARP_SIZE];
        }
        tmp = warp_reduce_sum<WARP_SIZE>(tmp);
    }
    ಯದಿ (sinks) {
        tmp += sycl::native::exp(sycl::max(sinks[i02] - max_val, -80.0f));
    }
    ಧ್ರುವ ಭಿನ್ನ inv_sum = 1.0f / tmp;

#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ col0 = 0; col0 < ncols; col0 += block_size) {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ col = col0 + tid;

        ಯದಿ (ncols_template == 0 && col >= ncols) {
            ಪ್ರತ್ಯಾಗಮ;
        }

        dst[col] = vals[col] * inv_sum;
    }
}
#ifdef __clang__
#pragma clang diagnostic ತೆಗೆ
#endif // __clang__

ಸ್ಥಿರ ಶೂನ್ಯ soft_max_back_f32(ಧ್ರುವ ಭಿನ್ನ *grad, ಧ್ರುವ ಭಿನ್ನ *dstf, ಭಿನ್ನ *dst,
                              ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ncols, ಧ್ರುವ ಭಿನ್ನ scale) {
    auto      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ tid      = item_ct1.get_local_id(2);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ rowx     = item_ct1.get_group(2);

    grad += int64_t(rowx)*ncols;
    dstf += int64_t(rowx)*ncols;
    dst  += int64_t(rowx)*ncols;

    ಭಿನ್ನ dgf_dot = 0.0f; // dot product of dst from forward pass and gradients

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ col = tid; col < ncols; col += WARP_SIZE) {
        dgf_dot += dstf[col]*grad[col];
    }

    dgf_dot = warp_reduce_sum<WARP_SIZE>(dgf_dot);

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ col = tid; col < ncols; col += WARP_SIZE) {
        dst[col] = scale * (grad[col] - dgf_dot) * dstf[col];
    }
}

template <ಪೂರ್ಣಾಙ್ಕ... Ns, typename T>
ಸ್ಥಿರ ಶೂನ್ಯ launch_soft_max_kernels(ಧ್ರುವ ಭಿನ್ನ *           x,
                                    ಧ್ರುವ T *               mask,
                                    ಧ್ರುವ ಭಿನ್ನ *           sinks,
                                    ಭಿನ್ನ *                 dst,
                                    ಧ್ರುವ soft_max_params & p,
                                    dpct::queue_ptr         stream,
                                    dpct::dim3              block_dims,
                                    dpct::dim3              block_nums,
                                    size_t                  nbytes_shared)
{
    auto launch_kernel = [=](auto I) -> ತಾರ್ಕಿಕ {
        constexpr ಪೂರ್ಣಾಙ್ಕ ncols = decltype(I)::value;
        constexpr ಪೂರ್ಣಾಙ್ಕ block = (ncols > 1024 ? 1024 : ncols);
        ಯದಿ (p.ncols == ncols) {
            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                    sycl::range<1>(nbytes_shared), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(
                        WARP_SIZE)]] {
                        soft_max_f32<ಸತ್ಯ, ncols, block>(
                            x, mask, sinks, dst, p,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                        GGML_UNUSED(item_ct1);
                    });
            });
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        }
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    };

    // unary fold over launch_kernel
    ಯದಿ ((launch_kernel(std::integral_constant<ಪೂರ್ಣಾಙ್ಕ, Ns>{}) || ...)) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range<1>(nbytes_shared), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    soft_max_f32<ಸತ್ಯ, 0, 0>(
                        x, mask, sinks, dst, p,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                    GGML_UNUSED(item_ct1);
                });
    });
}

template <typename T>
ಸ್ಥಿರ ಶೂನ್ಯ soft_max_f32_sycl(ಧ್ರುವ ಭಿನ್ನ *x, ಧ್ರುವ T *mask,
                              ಧ್ರುವ ಭಿನ್ನ *sinks, ಭಿನ್ನ *dst,
                              ಧ್ರುವ soft_max_params &params,
                              dpct::queue_ptr stream, ಪೂರ್ಣಾಙ್ಕ device) {
    ಪೂರ್ಣಾಙ್ಕ nth = WARP_SIZE;
    ಪೂರ್ಣಾಙ್ಕ max_block_size = ggml_sycl_info().max_work_group_sizes[device];
    ಧ್ರುವ int64_t ncols_x = params.ncols;

    ಯಾವತ್ (nth < ncols_x && nth < max_block_size) nth *= 2;
    ಯದಿ (nth>max_block_size) nth = max_block_size;

    ಧ್ರುವ dpct::dim3 block_dims(nth, 1, 1);
    ಧ್ರುವ dpct::dim3 block_nums(params.ne01, params.ne02, params.ne03);
    ಧ್ರುವ size_t nbytes_shared =
        (GGML_PAD(ncols_x, WARP_SIZE) + WARP_SIZE) * ಪರಿಮಾಣ(ಭಿನ್ನ);

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ id       = get_current_device_id();
    ಧ್ರುವ size_t smpbo = ggml_sycl_info().devices[id].smpbo;

    ಯದಿ (nbytes_shared <= smpbo && ncols_x <= max_block_size) {
        launch_soft_max_kernels<32, 64, 128, 256, 512, 1024, 2048, 4096>(
            x, mask, sinks, dst, params, stream, block_dims, block_nums,
            nbytes_shared);
    } ಅನ್ಯಥಾ {
        ಧ್ರುವ size_t nbytes_shared_ಕೆಳಗಿನ = WARP_SIZE * ಪರಿಮಾಣ(ಭಿನ್ನ);

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(nbytes_shared_ಕೆಳಗಿನ), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    soft_max_f32<ಅಸತ್ಯ, 0, 0>(
                        x, mask, sinks, dst, params,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                    GGML_UNUSED(item_ct1);
                });
        });
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ soft_max_back_f32_sycl(ಧ್ರುವ ಭಿನ್ನ *   grad,
                                   ಧ್ರುವ ಭಿನ್ನ *   dstf,
                                   ಭಿನ್ನ *         dst,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ       ncols,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ       nrows,
                                   ಧ್ರುವ ಭಿನ್ನ     scale,
                                   dpct::queue_ptr stream) {
    ಧ್ರುವ dpct::dim3 block_dims(WARP_SIZE, 1, 1);
    ಧ್ರುವ dpct::dim3 block_nums(nrows, 1, 1);

    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) {
                             soft_max_back_f32(grad, dstf, dst, ncols, scale);
                             GGML_UNUSED(item_ct1);
                         });
}

ಶೂನ್ಯ ggml_sycl_op_soft_max(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ಧ್ರುವ ggml_tensor * src0 = dst->src[0];
    ಧ್ರುವ ggml_tensor * src1 = dst->src[1];
    ಧ್ರುವ ggml_tensor * src2 = dst->src[2];

    ಧ್ರುವ ಭಿನ್ನ * src0_d = (ಧ್ರುವ ಭಿನ್ನ *) src0->data;
    ಧ್ರುವ ಶೂನ್ಯ  * src1_d = src1 ? (ಧ್ರುವ ಶೂನ್ಯ *) src1->data : nullptr;
    ಧ್ರುವ ಶೂನ್ಯ  * src2_d = src2 ? (ಧ್ರುವ ಶೂನ್ಯ *) src2->data : nullptr;
    ಭಿನ್ನ       *  dst_d = (ಭಿನ್ನ *) dst->data;

    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    // src1 contains mask and it is optional
    GGML_ASSERT(!src1 || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32);

    ಧ್ರುವ int64_t nrows_x = ggml_nrows(src0);
    ಧ್ರುವ int64_t nrows_y = src0->ne[1];

    ಧ್ರುವ int64_t ne00 = src0->ne[0];

    ಭಿನ್ನ scale    = 1.0f;
    ಭಿನ್ನ max_bias = 0.0f;

    memcpy(&scale,    (ಧ್ರುವ ಭಿನ್ನ *) dst->op_params + 0, ಪರಿಮಾಣ(ಭಿನ್ನ));
    memcpy(&max_bias, (ಧ್ರುವ ಭಿನ್ನ *) dst->op_params + 1, ಪರಿಮಾಣ(ಭಿನ್ನ));

    ಧ್ರುವ ತಾರ್ಕಿಕ use_f16 = (src1 && src1->type == GGML_TYPE_F16);

    ಧ್ರುವ int64_t nb11 = src1 ? src1->nb[1] : 1;
    ಧ್ರುವ int64_t nb12 = src1 ? src1->nb[2] : 1;
    ಧ್ರುವ int64_t nb13 = src1 ? src1->nb[3] : 1;

    ಧ್ರುವ int64_t ne12 = src1 ? src1->ne[2] : 1;
    ಧ್ರುವ int64_t ne13 = src1 ? src1->ne[3] : 1;

    ಧ್ರುವ uint32_t n_head      = src0->ne[2];
    ಧ್ರುವ uint32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((ಭಿನ್ನ) n_head));

    ಧ್ರುವ ಭಿನ್ನ m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    ಧ್ರುವ ಭಿನ್ನ m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);


    soft_max_params params = {};
    params.nheads = src0->ne[2];
    params.n_head_log2 = n_head_log2;
    params.ncols = ne00;
    params.nrows_x = nrows_x;
    params.nrows_y = nrows_y;
    params.ne00 = src0->ne[0];
    params.ne01 = src0->ne[1];
    params.ne02 = src0->ne[2];
    params.ne03 = src0->ne[3];
    params.nb11 = nb11;
    params.nb12 = nb12;
    params.nb13 = nb13;
    params.ne12 = ne12;
    params.ne13 = ne13;
    params.scale = scale;
    params.max_bias = max_bias;
    params.m0 = m0;
    params.m1 = m1;

    ಯದಿ (use_f16) {
        soft_max_f32_sycl(src0_d, (ಧ್ರುವ sycl::half *)src1_d,
                          (ಧ್ರುವ ಭಿನ್ನ *)src2_d, dst_d, params, stream,
                          ctx.device);
    } ಅನ್ಯಥಾ {
        soft_max_f32_sycl(src0_d, (ಧ್ರುವ ಭಿನ್ನ *)src1_d, (ಧ್ರುವ ಭಿನ್ನ *)src2_d,
                          dst_d, params, stream, ctx.device);
    }
}

ಶೂನ್ಯ ggml_sycl_op_soft_max_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ಧ್ರುವ ggml_tensor * src0 = dst->src[0]; // grad
    ಧ್ರುವ ggml_tensor * src1 = dst->src[1]; // forward pass output

    ಧ್ರುವ ಭಿನ್ನ * src0_d = (ಧ್ರುವ ಭಿನ್ನ *) src0->data;
    ಧ್ರುವ ಭಿನ್ನ * src1_d = (ಧ್ರುವ ಭಿನ್ನ *) src1->data;
    ಭಿನ್ನ       * dst_d  = (ಭಿನ್ನ       *) dst->data;

    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    ಧ್ರುವ int64_t ncols = src0->ne[0];
    ಧ್ರುವ int64_t nrows = ggml_nrows(src0);

    ಭಿನ್ನ scale    = 1.0f;
    ಭಿನ್ನ max_bias = 0.0f;

    memcpy(&scale,    (ಧ್ರುವ ಭಿನ್ನ *) dst->op_params + 0, ಪರಿಮಾಣ(ಭಿನ್ನ));
    memcpy(&max_bias, (ಧ್ರುವ ಭಿನ್ನ *) dst->op_params + 1, ಪರಿಮಾಣ(ಭಿನ್ನ));

    GGML_ASSERT(max_bias == 0.0f);

    soft_max_back_f32_sycl(src0_d, src1_d, dst_d, ncols, nrows, scale, stream);
}
