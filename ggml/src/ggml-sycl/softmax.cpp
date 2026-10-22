#அந்தர்பாவ "softmax.hpp"
#அந்தர்பாவ <cstdint>
#அந்தர்பாவ <utility>
#அந்தர்பாவ <cmath>


template <typename T> நிலையான __dpct_inline__ தசம t2f32(T val) {
    திரும்பு (தசம) val;
}

template <> தசம __dpct_inline__ t2f32<sycl::half>(sycl::half val) {
  திரும்பு sycl::vec<sycl::half, 1>(val)
      .convert<தசம, sycl::rounding_mode::automatic>()[0];
}

கட்டமைப்பு soft_max_params {

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
    தசம scale;
    தசம max_bias;
    தசம m0;
    தசம m1;
};

// When ncols_template == 0 the bounds for the loops in this function are not known and can't be unrolled.
// As we want to keep pragma unroll for all other cases we suppress the clang transformation warning here.
#ifdef __clang__
#pragma clang diagnostic தள்ளு
#pragma clang diagnostic ignored "-Wpass-failed"
#endif // __clang__
template <தருக்க use_shared, முழு ncols_template, முழு block_size_template, typename T>
நிலையான வெறுமை soft_max_f32(மாறிலி தசம *         x,
                         மாறிலி T *             mask,
                         மாறிலி தசம *         sinks,
                         தசம *               dst,
                         மாறிலி soft_max_params p,
                         uint8_t *             dpct_local) {
    auto      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    மாறிலி முழு ncols    = ncols_template == 0 ? p.ncols : ncols_template;
    மாறிலி முழு block_size = block_size_template == 0
                               ? item_ct1.get_local_range(2)
                               : block_size_template;
    மாறிலி முழு nthreads = block_size;
    மாறிலி முழு nwarps = nthreads / WARP_SIZE;
    மாறிலி size_t nreduce = nwarps / WARP_SIZE;

    மாறிலி முழு tid = item_ct1.get_local_id(2);

    மாறிலி int64_t i03 = item_ct1.get_group(0);
    மாறிலி int64_t i02 = item_ct1.get_group(1);
    மாறிலி int64_t i01 = item_ct1.get_group(2);

    //TODO: noncontigous inputs/outputs
    மாறிலி முழு rowx = item_ct1.get_group(2) +
                     item_ct1.get_group(1) * item_ct1.get_group_range(2) +
                     item_ct1.get_group(0) * item_ct1.get_group_range(2) *
                         item_ct1.get_group_range(1);

    மாறிலி int64_t i11 = i01;
    மாறிலி int64_t i12 = i02 % p.ne12;
    மாறிலி int64_t i13 = i03 % p.ne13;

    x    += int64_t(rowx)*ncols;
    mask += (i11*p.nb11 + i12*p.nb12 + i13*p.nb13) / அளவு(T) * (mask != nullptr);
    dst  += int64_t(rowx)*ncols;

    மாறிலி முழு warp_id = item_ct1.get_local_id(2) / WARP_SIZE;
    மாறிலி முழு lane_id = item_ct1.get_local_id(2) % WARP_SIZE;

    மாறிலி தசம slope = get_alibi_slope(p.max_bias, i02, p.n_head_log2, p.m0, p.m1);

    தசம * buf_iw = (தசம *) dpct_local;

    // shared memory buffer to cache values between iterations:
    தசம *vals = use_shared ? buf_iw + sycl::max(nwarps, WARP_SIZE) : dst;
    தசம max_val = sinks ? sinks[i02] : -INFINITY;
#pragma unroll
    க்கு (முழு col0 = 0; col0 < ncols; col0 += block_size) {
        மாறிலி முழு col = col0 + tid;

        எனில் (ncols_template == 0 && col >= ncols) {
            நிறுத்து;
        }

        மாறிலி தசம val = x[col]*p.scale + (mask ? slope*t2f32(mask[col]) : 0.0f);

        vals[col] = val;
        max_val   = sycl::max(max_val, val);
    }
    // find the max value in the block
    max_val = warp_reduce_max<WARP_SIZE>(max_val);

    எனில் (block_size > WARP_SIZE) {
        எனில் (lane_id == 0) {
            buf_iw[warp_id] = max_val;
        }
        item_ct1.barrier();

        max_val = -INFINITY;
        க்கு (முழு i = lane_id; i < nwarps; i += WARP_SIZE) {
            max_val = sycl::max(max_val, buf_iw[i]);
        }
        max_val = warp_reduce_max<WARP_SIZE>(max_val);
    }
    தசம tmp = 0.0f; // partial sum

#pragma unroll
    க்கு (முழு col0 = 0; col0 < ncols; col0 += block_size) {
        மாறிலி முழு col = col0 + tid;

        எனில் (ncols_template == 0 && col >= ncols) {
            நிறுத்து;
        }

        மாறிலி தசம val = sycl::native::exp(sycl::max(vals[col] - max_val, -80.0f));
        tmp += val;
        vals[col] = val;
    }
    // find the sum of exps in the block
    tmp = warp_reduce_sum<WARP_SIZE>(tmp);
    எனில் (block_size > WARP_SIZE) {
        item_ct1.barrier();
        எனில் (warp_id == 0) {
            buf_iw[lane_id] = 0.0f;
            க்கு (size_t i = 1; i < nreduce; i += 1) {
                buf_iw[lane_id + i * WARP_SIZE] = 0.f;
            }
        }
        item_ct1.barrier();

        எனில் (lane_id == 0) {
            buf_iw[warp_id] = tmp;
        }
        item_ct1.barrier();

        tmp = buf_iw[lane_id];
        க்கு (size_t i = 1; i < nreduce; i += 1) {
            tmp += buf_iw[lane_id + i * WARP_SIZE];
        }
        tmp = warp_reduce_sum<WARP_SIZE>(tmp);
    }
    எனில் (sinks) {
        tmp += sycl::native::exp(sycl::max(sinks[i02] - max_val, -80.0f));
    }
    மாறிலி தசம inv_sum = 1.0f / tmp;

#pragma unroll
    க்கு (முழு col0 = 0; col0 < ncols; col0 += block_size) {
        மாறிலி முழு col = col0 + tid;

        எனில் (ncols_template == 0 && col >= ncols) {
            திரும்பு;
        }

        dst[col] = vals[col] * inv_sum;
    }
}
#ifdef __clang__
#pragma clang diagnostic எடு
#endif // __clang__

நிலையான வெறுமை soft_max_back_f32(மாறிலி தசம *grad, மாறிலி தசம *dstf, தசம *dst,
                              மாறிலி முழு ncols, மாறிலி தசம scale) {
    auto      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    மாறிலி முழு tid      = item_ct1.get_local_id(2);
    மாறிலி முழு rowx     = item_ct1.get_group(2);

    grad += int64_t(rowx)*ncols;
    dstf += int64_t(rowx)*ncols;
    dst  += int64_t(rowx)*ncols;

    தசம dgf_dot = 0.0f; // dot product of dst from forward pass and gradients

    க்கு (முழு col = tid; col < ncols; col += WARP_SIZE) {
        dgf_dot += dstf[col]*grad[col];
    }

    dgf_dot = warp_reduce_sum<WARP_SIZE>(dgf_dot);

    க்கு (முழு col = tid; col < ncols; col += WARP_SIZE) {
        dst[col] = scale * (grad[col] - dgf_dot) * dstf[col];
    }
}

template <முழு... Ns, typename T>
நிலையான வெறுமை launch_soft_max_kernels(மாறிலி தசம *           x,
                                    மாறிலி T *               mask,
                                    மாறிலி தசம *           sinks,
                                    தசம *                 dst,
                                    மாறிலி soft_max_params & p,
                                    dpct::queue_ptr         stream,
                                    dpct::dim3              block_dims,
                                    dpct::dim3              block_nums,
                                    size_t                  nbytes_shared)
{
    auto launch_kernel = [=](auto I) -> தருக்க {
        constexpr முழு ncols = decltype(I)::value;
        constexpr முழு block = (ncols > 1024 ? 1024 : ncols);
        எனில் (p.ncols == ncols) {
            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                    sycl::range<1>(nbytes_shared), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(
                        WARP_SIZE)]] {
                        soft_max_f32<உண்மை, ncols, block>(
                            x, mask, sinks, dst, p,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                        GGML_UNUSED(item_ct1);
                    });
            });
            திரும்பு உண்மை;
        }
        திரும்பு பொய்;
    };

    // unary fold over launch_kernel
    எனில் ((launch_kernel(std::integral_constant<முழு, Ns>{}) || ...)) {
        திரும்பு;
    }

    stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range<1>(nbytes_shared), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    soft_max_f32<உண்மை, 0, 0>(
                        x, mask, sinks, dst, p,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                    GGML_UNUSED(item_ct1);
                });
    });
}

template <typename T>
நிலையான வெறுமை soft_max_f32_sycl(மாறிலி தசம *x, மாறிலி T *mask,
                              மாறிலி தசம *sinks, தசம *dst,
                              மாறிலி soft_max_params &params,
                              dpct::queue_ptr stream, முழு device) {
    முழு nth = WARP_SIZE;
    முழு max_block_size = ggml_sycl_info().max_work_group_sizes[device];
    மாறிலி int64_t ncols_x = params.ncols;

    போது (nth < ncols_x && nth < max_block_size) nth *= 2;
    எனில் (nth>max_block_size) nth = max_block_size;

    மாறிலி dpct::dim3 block_dims(nth, 1, 1);
    மாறிலி dpct::dim3 block_nums(params.ne01, params.ne02, params.ne03);
    மாறிலி size_t nbytes_shared =
        (GGML_PAD(ncols_x, WARP_SIZE) + WARP_SIZE) * அளவு(தசம);

    மாறிலி முழு id       = get_current_device_id();
    மாறிலி size_t smpbo = ggml_sycl_info().devices[id].smpbo;

    எனில் (nbytes_shared <= smpbo && ncols_x <= max_block_size) {
        launch_soft_max_kernels<32, 64, 128, 256, 512, 1024, 2048, 4096>(
            x, mask, sinks, dst, params, stream, block_dims, block_nums,
            nbytes_shared);
    } இல்லையெனில் {
        மாறிலி size_t nbytes_shared_கீழ் = WARP_SIZE * அளவு(தசம);

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(nbytes_shared_கீழ்), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    soft_max_f32<பொய், 0, 0>(
                        x, mask, sinks, dst, params,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                    GGML_UNUSED(item_ct1);
                });
        });
    }
}

நிலையான வெறுமை soft_max_back_f32_sycl(மாறிலி தசம *   grad,
                                   மாறிலி தசம *   dstf,
                                   தசம *         dst,
                                   மாறிலி முழு       ncols,
                                   மாறிலி முழு       nrows,
                                   மாறிலி தசம     scale,
                                   dpct::queue_ptr stream) {
    மாறிலி dpct::dim3 block_dims(WARP_SIZE, 1, 1);
    மாறிலி dpct::dim3 block_nums(nrows, 1, 1);

    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) {
                             soft_max_back_f32(grad, dstf, dst, ncols, scale);
                             GGML_UNUSED(item_ct1);
                         });
}

வெறுமை ggml_sycl_op_soft_max(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];
    மாறிலி ggml_tensor * src2 = dst->src[2];

    மாறிலி தசம * src0_d = (மாறிலி தசம *) src0->data;
    மாறிலி வெறுமை  * src1_d = src1 ? (மாறிலி வெறுமை *) src1->data : nullptr;
    மாறிலி வெறுமை  * src2_d = src2 ? (மாறிலி வெறுமை *) src2->data : nullptr;
    தசம       *  dst_d = (தசம *) dst->data;

    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    // src1 contains mask and it is optional
    GGML_ASSERT(!src1 || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32);

    மாறிலி int64_t nrows_x = ggml_nrows(src0);
    மாறிலி int64_t nrows_y = src0->ne[1];

    மாறிலி int64_t ne00 = src0->ne[0];

    தசம scale    = 1.0f;
    தசம max_bias = 0.0f;

    memcpy(&scale,    (மாறிலி தசம *) dst->op_params + 0, அளவு(தசம));
    memcpy(&max_bias, (மாறிலி தசம *) dst->op_params + 1, அளவு(தசம));

    மாறிலி தருக்க use_f16 = (src1 && src1->type == GGML_TYPE_F16);

    மாறிலி int64_t nb11 = src1 ? src1->nb[1] : 1;
    மாறிலி int64_t nb12 = src1 ? src1->nb[2] : 1;
    மாறிலி int64_t nb13 = src1 ? src1->nb[3] : 1;

    மாறிலி int64_t ne12 = src1 ? src1->ne[2] : 1;
    மாறிலி int64_t ne13 = src1 ? src1->ne[3] : 1;

    மாறிலி uint32_t n_head      = src0->ne[2];
    மாறிலி uint32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((தசம) n_head));

    மாறிலி தசம m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    மாறிலி தசம m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);


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

    எனில் (use_f16) {
        soft_max_f32_sycl(src0_d, (மாறிலி sycl::half *)src1_d,
                          (மாறிலி தசம *)src2_d, dst_d, params, stream,
                          ctx.device);
    } இல்லையெனில் {
        soft_max_f32_sycl(src0_d, (மாறிலி தசம *)src1_d, (மாறிலி தசம *)src2_d,
                          dst_d, params, stream, ctx.device);
    }
}

வெறுமை ggml_sycl_op_soft_max_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    மாறிலி ggml_tensor * src0 = dst->src[0]; // grad
    மாறிலி ggml_tensor * src1 = dst->src[1]; // forward pass output

    மாறிலி தசம * src0_d = (மாறிலி தசம *) src0->data;
    மாறிலி தசம * src1_d = (மாறிலி தசம *) src1->data;
    தசம       * dst_d  = (தசம       *) dst->data;

    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    மாறிலி int64_t ncols = src0->ne[0];
    மாறிலி int64_t nrows = ggml_nrows(src0);

    தசம scale    = 1.0f;
    தசம max_bias = 0.0f;

    memcpy(&scale,    (மாறிலி தசம *) dst->op_params + 0, அளவு(தசம));
    memcpy(&max_bias, (மாறிலி தசம *) dst->op_params + 1, அளவு(தசம));

    GGML_ASSERT(max_bias == 0.0f);

    soft_max_back_f32_sycl(src0_d, src1_d, dst_d, ncols, nrows, scale, stream);
}
