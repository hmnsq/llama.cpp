#ଅନ୍ତର୍ଭାଵ "softmax.hpp"
#ଅନ୍ତର୍ଭାଵ <cstdint>
#ଅନ୍ତର୍ଭାଵ <utility>
#ଅନ୍ତର୍ଭାଵ <cmath>


template <typename T> ସ୍ଥିର __dpct_inline__ ଦଶମିକ t2f32(T val) {
    ଫେରନ୍ତୁ (ଦଶମିକ) val;
}

template <> ଦଶମିକ __dpct_inline__ t2f32<sycl::half>(sycl::half val) {
  ଫେରନ୍ତୁ sycl::vec<sycl::half, 1>(val)
      .convert<ଦଶମିକ, sycl::rounding_mode::automatic>()[0];
}

ସଂରଚନା soft_max_params {

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
    ଦଶମିକ scale;
    ଦଶମିକ max_bias;
    ଦଶମିକ m0;
    ଦଶମିକ m1;
};

// When ncols_template == 0 the bounds for the loops in this function are not known and can't be unrolled.
// As we want to keep pragma unroll for all other cases we suppress the clang transformation warning here.
#ifdef __clang__
#pragma clang diagnostic ଠେଲ
#pragma clang diagnostic ignored "-Wpass-failed"
#endif // __clang__
template <ତାର୍କିକ use_shared, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_template, ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_size_template, typename T>
ସ୍ଥିର ଶୂନ୍ୟ soft_max_f32(ଅଚଳ ଦଶମିକ *         x,
                         ଅଚଳ T *             mask,
                         ଅଚଳ ଦଶମିକ *         sinks,
                         ଦଶମିକ *               dst,
                         ଅଚଳ soft_max_params p,
                         uint8_t *             dpct_local) {
    auto      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols    = ncols_template == 0 ? p.ncols : ncols_template;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_size = block_size_template == 0
                               ? item_ct1.get_local_range(2)
                               : block_size_template;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nthreads = block_size;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nwarps = nthreads / WARP_SIZE;
    ଅଚଳ size_t nreduce = nwarps / WARP_SIZE;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା tid = item_ct1.get_local_id(2);

    ଅଚଳ int64_t i03 = item_ct1.get_group(0);
    ଅଚଳ int64_t i02 = item_ct1.get_group(1);
    ଅଚଳ int64_t i01 = item_ct1.get_group(2);

    //TODO: noncontigous inputs/outputs
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା rowx = item_ct1.get_group(2) +
                     item_ct1.get_group(1) * item_ct1.get_group_range(2) +
                     item_ct1.get_group(0) * item_ct1.get_group_range(2) *
                         item_ct1.get_group_range(1);

    ଅଚଳ int64_t i11 = i01;
    ଅଚଳ int64_t i12 = i02 % p.ne12;
    ଅଚଳ int64_t i13 = i03 % p.ne13;

    x    += int64_t(rowx)*ncols;
    mask += (i11*p.nb11 + i12*p.nb12 + i13*p.nb13) / ଆକାର(T) * (mask != nullptr);
    dst  += int64_t(rowx)*ncols;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା warp_id = item_ct1.get_local_id(2) / WARP_SIZE;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା lane_id = item_ct1.get_local_id(2) % WARP_SIZE;

    ଅଚଳ ଦଶମିକ slope = get_alibi_slope(p.max_bias, i02, p.n_head_log2, p.m0, p.m1);

    ଦଶମିକ * buf_iw = (ଦଶମିକ *) dpct_local;

    // shared memory buffer to cache values between iterations:
    ଦଶମିକ *vals = use_shared ? buf_iw + sycl::max(nwarps, WARP_SIZE) : dst;
    ଦଶମିକ max_val = sinks ? sinks[i02] : -INFINITY;
#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା col0 = 0; col0 < ncols; col0 += block_size) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା col = col0 + tid;

        ଯଦି (ncols_template == 0 && col >= ncols) {
            ଭାଙ୍ଗ;
        }

        ଅଚଳ ଦଶମିକ val = x[col]*p.scale + (mask ? slope*t2f32(mask[col]) : 0.0f);

        vals[col] = val;
        max_val   = sycl::max(max_val, val);
    }
    // find the max value in the block
    max_val = warp_reduce_max<WARP_SIZE>(max_val);

    ଯଦି (block_size > WARP_SIZE) {
        ଯଦି (lane_id == 0) {
            buf_iw[warp_id] = max_val;
        }
        item_ct1.barrier();

        max_val = -INFINITY;
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = lane_id; i < nwarps; i += WARP_SIZE) {
            max_val = sycl::max(max_val, buf_iw[i]);
        }
        max_val = warp_reduce_max<WARP_SIZE>(max_val);
    }
    ଦଶମିକ tmp = 0.0f; // partial sum

#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା col0 = 0; col0 < ncols; col0 += block_size) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା col = col0 + tid;

        ଯଦି (ncols_template == 0 && col >= ncols) {
            ଭାଙ୍ଗ;
        }

        ଅଚଳ ଦଶମିକ val = sycl::native::exp(sycl::max(vals[col] - max_val, -80.0f));
        tmp += val;
        vals[col] = val;
    }
    // find the sum of exps in the block
    tmp = warp_reduce_sum<WARP_SIZE>(tmp);
    ଯଦି (block_size > WARP_SIZE) {
        item_ct1.barrier();
        ଯଦି (warp_id == 0) {
            buf_iw[lane_id] = 0.0f;
            ପାଇଁ (size_t i = 1; i < nreduce; i += 1) {
                buf_iw[lane_id + i * WARP_SIZE] = 0.f;
            }
        }
        item_ct1.barrier();

        ଯଦି (lane_id == 0) {
            buf_iw[warp_id] = tmp;
        }
        item_ct1.barrier();

        tmp = buf_iw[lane_id];
        ପାଇଁ (size_t i = 1; i < nreduce; i += 1) {
            tmp += buf_iw[lane_id + i * WARP_SIZE];
        }
        tmp = warp_reduce_sum<WARP_SIZE>(tmp);
    }
    ଯଦି (sinks) {
        tmp += sycl::native::exp(sycl::max(sinks[i02] - max_val, -80.0f));
    }
    ଅଚଳ ଦଶମିକ inv_sum = 1.0f / tmp;

#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା col0 = 0; col0 < ncols; col0 += block_size) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା col = col0 + tid;

        ଯଦି (ncols_template == 0 && col >= ncols) {
            ଫେରନ୍ତୁ;
        }

        dst[col] = vals[col] * inv_sum;
    }
}
#ifdef __clang__
#pragma clang diagnostic କାଢ଼
#endif // __clang__

ସ୍ଥିର ଶୂନ୍ୟ soft_max_back_f32(ଅଚଳ ଦଶମିକ *grad, ଅଚଳ ଦଶମିକ *dstf, ଦଶମିକ *dst,
                              ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ଦଶମିକ scale) {
    auto      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା tid      = item_ct1.get_local_id(2);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା rowx     = item_ct1.get_group(2);

    grad += int64_t(rowx)*ncols;
    dstf += int64_t(rowx)*ncols;
    dst  += int64_t(rowx)*ncols;

    ଦଶମିକ dgf_dot = 0.0f; // dot product of dst from forward pass and gradients

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା col = tid; col < ncols; col += WARP_SIZE) {
        dgf_dot += dstf[col]*grad[col];
    }

    dgf_dot = warp_reduce_sum<WARP_SIZE>(dgf_dot);

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା col = tid; col < ncols; col += WARP_SIZE) {
        dst[col] = scale * (grad[col] - dgf_dot) * dstf[col];
    }
}

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା... Ns, typename T>
ସ୍ଥିର ଶୂନ୍ୟ launch_soft_max_kernels(ଅଚଳ ଦଶମିକ *           x,
                                    ଅଚଳ T *               mask,
                                    ଅଚଳ ଦଶମିକ *           sinks,
                                    ଦଶମିକ *                 dst,
                                    ଅଚଳ soft_max_params & p,
                                    dpct::queue_ptr         stream,
                                    dpct::dim3              block_dims,
                                    dpct::dim3              block_nums,
                                    size_t                  nbytes_shared)
{
    auto launch_kernel = [=](auto I) -> ତାର୍କିକ {
        constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols = decltype(I)::value;
        constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା block = (ncols > 1024 ? 1024 : ncols);
        ଯଦି (p.ncols == ncols) {
            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                    sycl::range<1>(nbytes_shared), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(
                        WARP_SIZE)]] {
                        soft_max_f32<ସତ୍ୟ, ncols, block>(
                            x, mask, sinks, dst, p,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                        GGML_UNUSED(item_ct1);
                    });
            });
            ଫେରନ୍ତୁ ସତ୍ୟ;
        }
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    };

    // unary fold over launch_kernel
    ଯଦି ((launch_kernel(std::integral_constant<ପୂର୍ଣ୍ଣସଂଖ୍ୟା, Ns>{}) || ...)) {
        ଫେରନ୍ତୁ;
    }

    stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range<1>(nbytes_shared), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    soft_max_f32<ସତ୍ୟ, 0, 0>(
                        x, mask, sinks, dst, p,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                    GGML_UNUSED(item_ct1);
                });
    });
}

template <typename T>
ସ୍ଥିର ଶୂନ୍ୟ soft_max_f32_sycl(ଅଚଳ ଦଶମିକ *x, ଅଚଳ T *mask,
                              ଅଚଳ ଦଶମିକ *sinks, ଦଶମିକ *dst,
                              ଅଚଳ soft_max_params &params,
                              dpct::queue_ptr stream, ପୂର୍ଣ୍ଣସଂଖ୍ୟା device) {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = WARP_SIZE;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା max_block_size = ggml_sycl_info().max_work_group_sizes[device];
    ଅଚଳ int64_t ncols_x = params.ncols;

    ଯେତେବେଳେ (nth < ncols_x && nth < max_block_size) nth *= 2;
    ଯଦି (nth>max_block_size) nth = max_block_size;

    ଅଚଳ dpct::dim3 block_dims(nth, 1, 1);
    ଅଚଳ dpct::dim3 block_nums(params.ne01, params.ne02, params.ne03);
    ଅଚଳ size_t nbytes_shared =
        (GGML_PAD(ncols_x, WARP_SIZE) + WARP_SIZE) * ଆକାର(ଦଶମିକ);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା id       = get_current_device_id();
    ଅଚଳ size_t smpbo = ggml_sycl_info().devices[id].smpbo;

    ଯଦି (nbytes_shared <= smpbo && ncols_x <= max_block_size) {
        launch_soft_max_kernels<32, 64, 128, 256, 512, 1024, 2048, 4096>(
            x, mask, sinks, dst, params, stream, block_dims, block_nums,
            nbytes_shared);
    } ଅନ୍ୟଥା {
        ଅଚଳ size_t nbytes_shared_ତଳ = WARP_SIZE * ଆକାର(ଦଶମିକ);

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(nbytes_shared_ତଳ), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    soft_max_f32<ମିଥ୍ୟା, 0, 0>(
                        x, mask, sinks, dst, params,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                    GGML_UNUSED(item_ct1);
                });
        });
    }
}

ସ୍ଥିର ଶୂନ୍ୟ soft_max_back_f32_sycl(ଅଚଳ ଦଶମିକ *   grad,
                                   ଅଚଳ ଦଶମିକ *   dstf,
                                   ଦଶମିକ *         dst,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା       ncols,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା       nrows,
                                   ଅଚଳ ଦଶମିକ     scale,
                                   dpct::queue_ptr stream) {
    ଅଚଳ dpct::dim3 block_dims(WARP_SIZE, 1, 1);
    ଅଚଳ dpct::dim3 block_nums(nrows, 1, 1);

    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) {
                             soft_max_back_f32(grad, dstf, dst, ncols, scale);
                             GGML_UNUSED(item_ct1);
                         });
}

ଶୂନ୍ୟ ggml_sycl_op_soft_max(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];
    ଅଚଳ ggml_tensor * src2 = dst->src[2];

    ଅଚଳ ଦଶମିକ * src0_d = (ଅଚଳ ଦଶମିକ *) src0->data;
    ଅଚଳ ଶୂନ୍ୟ  * src1_d = src1 ? (ଅଚଳ ଶୂନ୍ୟ *) src1->data : nullptr;
    ଅଚଳ ଶୂନ୍ୟ  * src2_d = src2 ? (ଅଚଳ ଶୂନ୍ୟ *) src2->data : nullptr;
    ଦଶମିକ       *  dst_d = (ଦଶମିକ *) dst->data;

    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    // src1 contains mask and it is optional
    GGML_ASSERT(!src1 || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32);

    ଅଚଳ int64_t nrows_x = ggml_nrows(src0);
    ଅଚଳ int64_t nrows_y = src0->ne[1];

    ଅଚଳ int64_t ne00 = src0->ne[0];

    ଦଶମିକ scale    = 1.0f;
    ଦଶମିକ max_bias = 0.0f;

    memcpy(&scale,    (ଅଚଳ ଦଶମିକ *) dst->op_params + 0, ଆକାର(ଦଶମିକ));
    memcpy(&max_bias, (ଅଚଳ ଦଶମିକ *) dst->op_params + 1, ଆକାର(ଦଶମିକ));

    ଅଚଳ ତାର୍କିକ use_f16 = (src1 && src1->type == GGML_TYPE_F16);

    ଅଚଳ int64_t nb11 = src1 ? src1->nb[1] : 1;
    ଅଚଳ int64_t nb12 = src1 ? src1->nb[2] : 1;
    ଅଚଳ int64_t nb13 = src1 ? src1->nb[3] : 1;

    ଅଚଳ int64_t ne12 = src1 ? src1->ne[2] : 1;
    ଅଚଳ int64_t ne13 = src1 ? src1->ne[3] : 1;

    ଅଚଳ uint32_t n_head      = src0->ne[2];
    ଅଚଳ uint32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((ଦଶମିକ) n_head));

    ଅଚଳ ଦଶମିକ m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    ଅଚଳ ଦଶମିକ m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);


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

    ଯଦି (use_f16) {
        soft_max_f32_sycl(src0_d, (ଅଚଳ sycl::half *)src1_d,
                          (ଅଚଳ ଦଶମିକ *)src2_d, dst_d, params, stream,
                          ctx.device);
    } ଅନ୍ୟଥା {
        soft_max_f32_sycl(src0_d, (ଅଚଳ ଦଶମିକ *)src1_d, (ଅଚଳ ଦଶମିକ *)src2_d,
                          dst_d, params, stream, ctx.device);
    }
}

ଶୂନ୍ୟ ggml_sycl_op_soft_max_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ଅଚଳ ggml_tensor * src0 = dst->src[0]; // grad
    ଅଚଳ ggml_tensor * src1 = dst->src[1]; // forward pass output

    ଅଚଳ ଦଶମିକ * src0_d = (ଅଚଳ ଦଶମିକ *) src0->data;
    ଅଚଳ ଦଶମିକ * src1_d = (ଅଚଳ ଦଶମିକ *) src1->data;
    ଦଶମିକ       * dst_d  = (ଦଶମିକ       *) dst->data;

    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    ଅଚଳ int64_t ncols = src0->ne[0];
    ଅଚଳ int64_t nrows = ggml_nrows(src0);

    ଦଶମିକ scale    = 1.0f;
    ଦଶମିକ max_bias = 0.0f;

    memcpy(&scale,    (ଅଚଳ ଦଶମିକ *) dst->op_params + 0, ଆକାର(ଦଶମିକ));
    memcpy(&max_bias, (ଅଚଳ ଦଶମିକ *) dst->op_params + 1, ଆକାର(ଦଶମିକ));

    GGML_ASSERT(max_bias == 0.0f);

    soft_max_back_f32_sycl(src0_d, src1_d, dst_d, ncols, nrows, scale, stream);
}
