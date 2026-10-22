#অন্তর্ভাব "softmax.hpp"
#অন্তর্ভাব <cstdint>
#অন্তর্ভাব <utility>
#অন্তর্ভাব <cmath>


template <typename T> স্থির __dpct_inline__ ভিন্ন t2f32(T val) {
    প্রত্যাগম (ভিন্ন) val;
}

template <> ভিন্ন __dpct_inline__ t2f32<sycl::half>(sycl::half val) {
  প্রত্যাগম sycl::vec<sycl::half, 1>(val)
      .convert<ভিন্ন, sycl::rounding_mode::automatic>()[0];
}

সংরচনা soft_max_params {

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
    ভিন্ন scale;
    ভিন্ন max_bias;
    ভিন্ন m0;
    ভিন্ন m1;
};

// When ncols_template == 0 the bounds for the loops in this function are not known and can't be unrolled.
// As we want to keep pragma unroll for all other cases we suppress the clang transformation warning here.
#ifdef __clang__
#pragma clang diagnostic ঠেলো
#pragma clang diagnostic ignored "-Wpass-failed"
#endif // __clang__
template <তার্কিক use_shared, পূর্ণাঙ্ক ncols_template, পূর্ণাঙ্ক block_size_template, typename T>
স্থির শূন্য soft_max_f32(ধ্রুব ভিন্ন *         x,
                         ধ্রুব T *             mask,
                         ধ্রুব ভিন্ন *         sinks,
                         ভিন্ন *               dst,
                         ধ্রুব soft_max_params p,
                         uint8_t *             dpct_local) {
    auto      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ধ্রুব পূর্ণাঙ্ক ncols    = ncols_template == 0 ? p.ncols : ncols_template;
    ধ্রুব পূর্ণাঙ্ক block_size = block_size_template == 0
                               ? item_ct1.get_local_range(2)
                               : block_size_template;
    ধ্রুব পূর্ণাঙ্ক nthreads = block_size;
    ধ্রুব পূর্ণাঙ্ক nwarps = nthreads / WARP_SIZE;
    ধ্রুব size_t nreduce = nwarps / WARP_SIZE;

    ধ্রুব পূর্ণাঙ্ক tid = item_ct1.get_local_id(2);

    ধ্রুব int64_t i03 = item_ct1.get_group(0);
    ধ্রুব int64_t i02 = item_ct1.get_group(1);
    ধ্রুব int64_t i01 = item_ct1.get_group(2);

    //TODO: noncontigous inputs/outputs
    ধ্রুব পূর্ণাঙ্ক rowx = item_ct1.get_group(2) +
                     item_ct1.get_group(1) * item_ct1.get_group_range(2) +
                     item_ct1.get_group(0) * item_ct1.get_group_range(2) *
                         item_ct1.get_group_range(1);

    ধ্রুব int64_t i11 = i01;
    ধ্রুব int64_t i12 = i02 % p.ne12;
    ধ্রুব int64_t i13 = i03 % p.ne13;

    x    += int64_t(rowx)*ncols;
    mask += (i11*p.nb11 + i12*p.nb12 + i13*p.nb13) / পরিমাণ(T) * (mask != nullptr);
    dst  += int64_t(rowx)*ncols;

    ধ্রুব পূর্ণাঙ্ক warp_id = item_ct1.get_local_id(2) / WARP_SIZE;
    ধ্রুব পূর্ণাঙ্ক lane_id = item_ct1.get_local_id(2) % WARP_SIZE;

    ধ্রুব ভিন্ন slope = get_alibi_slope(p.max_bias, i02, p.n_head_log2, p.m0, p.m1);

    ভিন্ন * buf_iw = (ভিন্ন *) dpct_local;

    // shared memory buffer to cache values between iterations:
    ভিন্ন *vals = use_shared ? buf_iw + sycl::max(nwarps, WARP_SIZE) : dst;
    ভিন্ন max_val = sinks ? sinks[i02] : -INFINITY;
#pragma unroll
    প্রতি (পূর্ণাঙ্ক col0 = 0; col0 < ncols; col0 += block_size) {
        ধ্রুব পূর্ণাঙ্ক col = col0 + tid;

        যদি (ncols_template == 0 && col >= ncols) {
            ভঙ্গ;
        }

        ধ্রুব ভিন্ন val = x[col]*p.scale + (mask ? slope*t2f32(mask[col]) : 0.0f);

        vals[col] = val;
        max_val   = sycl::max(max_val, val);
    }
    // find the max value in the block
    max_val = warp_reduce_max<WARP_SIZE>(max_val);

    যদি (block_size > WARP_SIZE) {
        যদি (lane_id == 0) {
            buf_iw[warp_id] = max_val;
        }
        item_ct1.barrier();

        max_val = -INFINITY;
        প্রতি (পূর্ণাঙ্ক i = lane_id; i < nwarps; i += WARP_SIZE) {
            max_val = sycl::max(max_val, buf_iw[i]);
        }
        max_val = warp_reduce_max<WARP_SIZE>(max_val);
    }
    ভিন্ন tmp = 0.0f; // partial sum

#pragma unroll
    প্রতি (পূর্ণাঙ্ক col0 = 0; col0 < ncols; col0 += block_size) {
        ধ্রুব পূর্ণাঙ্ক col = col0 + tid;

        যদি (ncols_template == 0 && col >= ncols) {
            ভঙ্গ;
        }

        ধ্রুব ভিন্ন val = sycl::native::exp(sycl::max(vals[col] - max_val, -80.0f));
        tmp += val;
        vals[col] = val;
    }
    // find the sum of exps in the block
    tmp = warp_reduce_sum<WARP_SIZE>(tmp);
    যদি (block_size > WARP_SIZE) {
        item_ct1.barrier();
        যদি (warp_id == 0) {
            buf_iw[lane_id] = 0.0f;
            প্রতি (size_t i = 1; i < nreduce; i += 1) {
                buf_iw[lane_id + i * WARP_SIZE] = 0.f;
            }
        }
        item_ct1.barrier();

        যদি (lane_id == 0) {
            buf_iw[warp_id] = tmp;
        }
        item_ct1.barrier();

        tmp = buf_iw[lane_id];
        প্রতি (size_t i = 1; i < nreduce; i += 1) {
            tmp += buf_iw[lane_id + i * WARP_SIZE];
        }
        tmp = warp_reduce_sum<WARP_SIZE>(tmp);
    }
    যদি (sinks) {
        tmp += sycl::native::exp(sycl::max(sinks[i02] - max_val, -80.0f));
    }
    ধ্রুব ভিন্ন inv_sum = 1.0f / tmp;

#pragma unroll
    প্রতি (পূর্ণাঙ্ক col0 = 0; col0 < ncols; col0 += block_size) {
        ধ্রুব পূর্ণাঙ্ক col = col0 + tid;

        যদি (ncols_template == 0 && col >= ncols) {
            প্রত্যাগম;
        }

        dst[col] = vals[col] * inv_sum;
    }
}
#ifdef __clang__
#pragma clang diagnostic বের_করো
#endif // __clang__

স্থির শূন্য soft_max_back_f32(ধ্রুব ভিন্ন *grad, ধ্রুব ভিন্ন *dstf, ভিন্ন *dst,
                              ধ্রুব পূর্ণাঙ্ক ncols, ধ্রুব ভিন্ন scale) {
    auto      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ধ্রুব পূর্ণাঙ্ক tid      = item_ct1.get_local_id(2);
    ধ্রুব পূর্ণাঙ্ক rowx     = item_ct1.get_group(2);

    grad += int64_t(rowx)*ncols;
    dstf += int64_t(rowx)*ncols;
    dst  += int64_t(rowx)*ncols;

    ভিন্ন dgf_dot = 0.0f; // dot product of dst from forward pass and gradients

    প্রতি (পূর্ণাঙ্ক col = tid; col < ncols; col += WARP_SIZE) {
        dgf_dot += dstf[col]*grad[col];
    }

    dgf_dot = warp_reduce_sum<WARP_SIZE>(dgf_dot);

    প্রতি (পূর্ণাঙ্ক col = tid; col < ncols; col += WARP_SIZE) {
        dst[col] = scale * (grad[col] - dgf_dot) * dstf[col];
    }
}

template <পূর্ণাঙ্ক... Ns, typename T>
স্থির শূন্য launch_soft_max_kernels(ধ্রুব ভিন্ন *           x,
                                    ধ্রুব T *               mask,
                                    ধ্রুব ভিন্ন *           sinks,
                                    ভিন্ন *                 dst,
                                    ধ্রুব soft_max_params & p,
                                    dpct::queue_ptr         stream,
                                    dpct::dim3              block_dims,
                                    dpct::dim3              block_nums,
                                    size_t                  nbytes_shared)
{
    auto launch_kernel = [=](auto I) -> তার্কিক {
        constexpr পূর্ণাঙ্ক ncols = decltype(I)::value;
        constexpr পূর্ণাঙ্ক block = (ncols > 1024 ? 1024 : ncols);
        যদি (p.ncols == ncols) {
            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                    sycl::range<1>(nbytes_shared), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(
                        WARP_SIZE)]] {
                        soft_max_f32<সত্য, ncols, block>(
                            x, mask, sinks, dst, p,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                        GGML_UNUSED(item_ct1);
                    });
            });
            প্রত্যাগম সত্য;
        }
        প্রত্যাগম অসত্য;
    };

    // unary fold over launch_kernel
    যদি ((launch_kernel(std::integral_constant<পূর্ণাঙ্ক, Ns>{}) || ...)) {
        প্রত্যাগম;
    }

    stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range<1>(nbytes_shared), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    soft_max_f32<সত্য, 0, 0>(
                        x, mask, sinks, dst, p,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                    GGML_UNUSED(item_ct1);
                });
    });
}

template <typename T>
স্থির শূন্য soft_max_f32_sycl(ধ্রুব ভিন্ন *x, ধ্রুব T *mask,
                              ধ্রুব ভিন্ন *sinks, ভিন্ন *dst,
                              ধ্রুব soft_max_params &params,
                              dpct::queue_ptr stream, পূর্ণাঙ্ক device) {
    পূর্ণাঙ্ক nth = WARP_SIZE;
    পূর্ণাঙ্ক max_block_size = ggml_sycl_info().max_work_group_sizes[device];
    ধ্রুব int64_t ncols_x = params.ncols;

    যাবত্ (nth < ncols_x && nth < max_block_size) nth *= 2;
    যদি (nth>max_block_size) nth = max_block_size;

    ধ্রুব dpct::dim3 block_dims(nth, 1, 1);
    ধ্রুব dpct::dim3 block_nums(params.ne01, params.ne02, params.ne03);
    ধ্রুব size_t nbytes_shared =
        (GGML_PAD(ncols_x, WARP_SIZE) + WARP_SIZE) * পরিমাণ(ভিন্ন);

    ধ্রুব পূর্ণাঙ্ক id       = get_current_device_id();
    ধ্রুব size_t smpbo = ggml_sycl_info().devices[id].smpbo;

    যদি (nbytes_shared <= smpbo && ncols_x <= max_block_size) {
        launch_soft_max_kernels<32, 64, 128, 256, 512, 1024, 2048, 4096>(
            x, mask, sinks, dst, params, stream, block_dims, block_nums,
            nbytes_shared);
    } অন্যথা {
        ধ্রুব size_t nbytes_shared_নিচের = WARP_SIZE * পরিমাণ(ভিন্ন);

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(nbytes_shared_নিচের), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    soft_max_f32<অসত্য, 0, 0>(
                        x, mask, sinks, dst, params,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                    GGML_UNUSED(item_ct1);
                });
        });
    }
}

স্থির শূন্য soft_max_back_f32_sycl(ধ্রুব ভিন্ন *   grad,
                                   ধ্রুব ভিন্ন *   dstf,
                                   ভিন্ন *         dst,
                                   ধ্রুব পূর্ণাঙ্ক       ncols,
                                   ধ্রুব পূর্ণাঙ্ক       nrows,
                                   ধ্রুব ভিন্ন     scale,
                                   dpct::queue_ptr stream) {
    ধ্রুব dpct::dim3 block_dims(WARP_SIZE, 1, 1);
    ধ্রুব dpct::dim3 block_nums(nrows, 1, 1);

    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) {
                             soft_max_back_f32(grad, dstf, dst, ncols, scale);
                             GGML_UNUSED(item_ct1);
                         });
}

শূন্য ggml_sycl_op_soft_max(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব ggml_tensor * src1 = dst->src[1];
    ধ্রুব ggml_tensor * src2 = dst->src[2];

    ধ্রুব ভিন্ন * src0_d = (ধ্রুব ভিন্ন *) src0->data;
    ধ্রুব শূন্য  * src1_d = src1 ? (ধ্রুব শূন্য *) src1->data : nullptr;
    ধ্রুব শূন্য  * src2_d = src2 ? (ধ্রুব শূন্য *) src2->data : nullptr;
    ভিন্ন       *  dst_d = (ভিন্ন *) dst->data;

    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    // src1 contains mask and it is optional
    GGML_ASSERT(!src1 || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32);

    ধ্রুব int64_t nrows_x = ggml_nrows(src0);
    ধ্রুব int64_t nrows_y = src0->ne[1];

    ধ্রুব int64_t ne00 = src0->ne[0];

    ভিন্ন scale    = 1.0f;
    ভিন্ন max_bias = 0.0f;

    memcpy(&scale,    (ধ্রুব ভিন্ন *) dst->op_params + 0, পরিমাণ(ভিন্ন));
    memcpy(&max_bias, (ধ্রুব ভিন্ন *) dst->op_params + 1, পরিমাণ(ভিন্ন));

    ধ্রুব তার্কিক use_f16 = (src1 && src1->type == GGML_TYPE_F16);

    ধ্রুব int64_t nb11 = src1 ? src1->nb[1] : 1;
    ধ্রুব int64_t nb12 = src1 ? src1->nb[2] : 1;
    ধ্রুব int64_t nb13 = src1 ? src1->nb[3] : 1;

    ধ্রুব int64_t ne12 = src1 ? src1->ne[2] : 1;
    ধ্রুব int64_t ne13 = src1 ? src1->ne[3] : 1;

    ধ্রুব uint32_t n_head      = src0->ne[2];
    ধ্রুব uint32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((ভিন্ন) n_head));

    ধ্রুব ভিন্ন m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    ধ্রুব ভিন্ন m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);


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

    যদি (use_f16) {
        soft_max_f32_sycl(src0_d, (ধ্রুব sycl::half *)src1_d,
                          (ধ্রুব ভিন্ন *)src2_d, dst_d, params, stream,
                          ctx.device);
    } অন্যথা {
        soft_max_f32_sycl(src0_d, (ধ্রুব ভিন্ন *)src1_d, (ধ্রুব ভিন্ন *)src2_d,
                          dst_d, params, stream, ctx.device);
    }
}

শূন্য ggml_sycl_op_soft_max_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ধ্রুব ggml_tensor * src0 = dst->src[0]; // grad
    ধ্রুব ggml_tensor * src1 = dst->src[1]; // forward pass output

    ধ্রুব ভিন্ন * src0_d = (ধ্রুব ভিন্ন *) src0->data;
    ধ্রুব ভিন্ন * src1_d = (ধ্রুব ভিন্ন *) src1->data;
    ভিন্ন       * dst_d  = (ভিন্ন       *) dst->data;

    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    ধ্রুব int64_t ncols = src0->ne[0];
    ধ্রুব int64_t nrows = ggml_nrows(src0);

    ভিন্ন scale    = 1.0f;
    ভিন্ন max_bias = 0.0f;

    memcpy(&scale,    (ধ্রুব ভিন্ন *) dst->op_params + 0, পরিমাণ(ভিন্ন));
    memcpy(&max_bias, (ধ্রুব ভিন্ন *) dst->op_params + 1, পরিমাণ(ভিন্ন));

    GGML_ASSERT(max_bias == 0.0f);

    soft_max_back_f32_sycl(src0_d, src1_d, dst_d, ncols, nrows, scale, stream);
}
