#شامل "softmax.hpp"
#شامل <cstdint>
#شامل <utility>
#شامل <cmath>


template <typename T> ایستا __dpct_inline__ اعشاری t2f32(T val) {
    بازگشت (اعشاری) val;
}

template <> اعشاری __dpct_inline__ t2f32<sycl::half>(sycl::half val) {
  بازگشت sycl::vec<sycl::half, 1>(val)
      .convert<اعشاری, sycl::rounding_mode::automatic>()[0];
}

ساختار soft_max_params {

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
    اعشاری scale;
    اعشاری max_bias;
    اعشاری m0;
    اعشاری m1;
};

// When ncols_template == 0 the bounds for the loops in this function are not known and can't be unrolled.
// As we want to keep pragma unroll for all other cases we suppress the clang transformation warning here.
#ifdef __clang__
#pragma clang diagnostic افزودن
#pragma clang diagnostic ignored "-Wpass-failed"
#endif // __clang__
template <منطقی use_shared, صحیح ncols_template, صحیح block_size_template, typename T>
ایستا تهی soft_max_f32(ثابت اعشاری *         x,
                         ثابت T *             mask,
                         ثابت اعشاری *         sinks,
                         اعشاری *               dst,
                         ثابت soft_max_params p,
                         uint8_t *             dpct_local) {
    auto      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ثابت صحیح ncols    = ncols_template == 0 ? p.ncols : ncols_template;
    ثابت صحیح block_size = block_size_template == 0
                               ? item_ct1.get_local_range(2)
                               : block_size_template;
    ثابت صحیح nthreads = block_size;
    ثابت صحیح nwarps = nthreads / WARP_SIZE;
    ثابت size_t nreduce = nwarps / WARP_SIZE;

    ثابت صحیح tid = item_ct1.get_local_id(2);

    ثابت int64_t i03 = item_ct1.get_group(0);
    ثابت int64_t i02 = item_ct1.get_group(1);
    ثابت int64_t i01 = item_ct1.get_group(2);

    //TODO: noncontigous inputs/outputs
    ثابت صحیح rowx = item_ct1.get_group(2) +
                     item_ct1.get_group(1) * item_ct1.get_group_range(2) +
                     item_ct1.get_group(0) * item_ct1.get_group_range(2) *
                         item_ct1.get_group_range(1);

    ثابت int64_t i11 = i01;
    ثابت int64_t i12 = i02 % p.ne12;
    ثابت int64_t i13 = i03 % p.ne13;

    x    += int64_t(rowx)*ncols;
    mask += (i11*p.nb11 + i12*p.nb12 + i13*p.nb13) / اندازه(T) * (mask != nullptr);
    dst  += int64_t(rowx)*ncols;

    ثابت صحیح warp_id = item_ct1.get_local_id(2) / WARP_SIZE;
    ثابت صحیح lane_id = item_ct1.get_local_id(2) % WARP_SIZE;

    ثابت اعشاری slope = get_alibi_slope(p.max_bias, i02, p.n_head_log2, p.m0, p.m1);

    اعشاری * buf_iw = (اعشاری *) dpct_local;

    // shared memory buffer to cache values between iterations:
    اعشاری *vals = use_shared ? buf_iw + sycl::max(nwarps, WARP_SIZE) : dst;
    اعشاری max_val = sinks ? sinks[i02] : -INFINITY;
#pragma unroll
    برای (صحیح col0 = 0; col0 < ncols; col0 += block_size) {
        ثابت صحیح col = col0 + tid;

        اگر (ncols_template == 0 && col >= ncols) {
            بشکن;
        }

        ثابت اعشاری val = x[col]*p.scale + (mask ? slope*t2f32(mask[col]) : 0.0f);

        vals[col] = val;
        max_val   = sycl::max(max_val, val);
    }
    // find the max value in the block
    max_val = warp_reduce_max<WARP_SIZE>(max_val);

    اگر (block_size > WARP_SIZE) {
        اگر (lane_id == 0) {
            buf_iw[warp_id] = max_val;
        }
        item_ct1.barrier();

        max_val = -INFINITY;
        برای (صحیح i = lane_id; i < nwarps; i += WARP_SIZE) {
            max_val = sycl::max(max_val, buf_iw[i]);
        }
        max_val = warp_reduce_max<WARP_SIZE>(max_val);
    }
    اعشاری tmp = 0.0f; // partial sum

#pragma unroll
    برای (صحیح col0 = 0; col0 < ncols; col0 += block_size) {
        ثابت صحیح col = col0 + tid;

        اگر (ncols_template == 0 && col >= ncols) {
            بشکن;
        }

        ثابت اعشاری val = sycl::native::exp(sycl::max(vals[col] - max_val, -80.0f));
        tmp += val;
        vals[col] = val;
    }
    // find the sum of exps in the block
    tmp = warp_reduce_sum<WARP_SIZE>(tmp);
    اگر (block_size > WARP_SIZE) {
        item_ct1.barrier();
        اگر (warp_id == 0) {
            buf_iw[lane_id] = 0.0f;
            برای (size_t i = 1; i < nreduce; i += 1) {
                buf_iw[lane_id + i * WARP_SIZE] = 0.f;
            }
        }
        item_ct1.barrier();

        اگر (lane_id == 0) {
            buf_iw[warp_id] = tmp;
        }
        item_ct1.barrier();

        tmp = buf_iw[lane_id];
        برای (size_t i = 1; i < nreduce; i += 1) {
            tmp += buf_iw[lane_id + i * WARP_SIZE];
        }
        tmp = warp_reduce_sum<WARP_SIZE>(tmp);
    }
    اگر (sinks) {
        tmp += sycl::native::exp(sycl::max(sinks[i02] - max_val, -80.0f));
    }
    ثابت اعشاری inv_sum = 1.0f / tmp;

#pragma unroll
    برای (صحیح col0 = 0; col0 < ncols; col0 += block_size) {
        ثابت صحیح col = col0 + tid;

        اگر (ncols_template == 0 && col >= ncols) {
            بازگشت;
        }

        dst[col] = vals[col] * inv_sum;
    }
}
#ifdef __clang__
#pragma clang diagnostic برداشت
#endif // __clang__

ایستا تهی soft_max_back_f32(ثابت اعشاری *grad, ثابت اعشاری *dstf, اعشاری *dst,
                              ثابت صحیح ncols, ثابت اعشاری scale) {
    auto      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ثابت صحیح tid      = item_ct1.get_local_id(2);
    ثابت صحیح rowx     = item_ct1.get_group(2);

    grad += int64_t(rowx)*ncols;
    dstf += int64_t(rowx)*ncols;
    dst  += int64_t(rowx)*ncols;

    اعشاری dgf_dot = 0.0f; // dot product of dst from forward pass and gradients

    برای (صحیح col = tid; col < ncols; col += WARP_SIZE) {
        dgf_dot += dstf[col]*grad[col];
    }

    dgf_dot = warp_reduce_sum<WARP_SIZE>(dgf_dot);

    برای (صحیح col = tid; col < ncols; col += WARP_SIZE) {
        dst[col] = scale * (grad[col] - dgf_dot) * dstf[col];
    }
}

template <صحیح... Ns, typename T>
ایستا تهی launch_soft_max_kernels(ثابت اعشاری *           x,
                                    ثابت T *               mask,
                                    ثابت اعشاری *           sinks,
                                    اعشاری *                 dst,
                                    ثابت soft_max_params & p,
                                    dpct::queue_ptr         stream,
                                    dpct::dim3              block_dims,
                                    dpct::dim3              block_nums,
                                    size_t                  nbytes_shared)
{
    auto launch_kernel = [=](auto I) -> منطقی {
        constexpr صحیح ncols = decltype(I)::value;
        constexpr صحیح block = (ncols > 1024 ? 1024 : ncols);
        اگر (p.ncols == ncols) {
            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                    sycl::range<1>(nbytes_shared), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(
                        WARP_SIZE)]] {
                        soft_max_f32<درست, ncols, block>(
                            x, mask, sinks, dst, p,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                        GGML_UNUSED(item_ct1);
                    });
            });
            بازگشت درست;
        }
        بازگشت نادرست;
    };

    // unary fold over launch_kernel
    اگر ((launch_kernel(std::integral_constant<صحیح, Ns>{}) || ...)) {
        بازگشت;
    }

    stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range<1>(nbytes_shared), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    soft_max_f32<درست, 0, 0>(
                        x, mask, sinks, dst, p,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                    GGML_UNUSED(item_ct1);
                });
    });
}

template <typename T>
ایستا تهی soft_max_f32_sycl(ثابت اعشاری *x, ثابت T *mask,
                              ثابت اعشاری *sinks, اعشاری *dst,
                              ثابت soft_max_params &params,
                              dpct::queue_ptr stream, صحیح device) {
    صحیح nth = WARP_SIZE;
    صحیح max_block_size = ggml_sycl_info().max_work_group_sizes[device];
    ثابت int64_t ncols_x = params.ncols;

    وقتی (nth < ncols_x && nth < max_block_size) nth *= 2;
    اگر (nth>max_block_size) nth = max_block_size;

    ثابت dpct::dim3 block_dims(nth, 1, 1);
    ثابت dpct::dim3 block_nums(params.ne01, params.ne02, params.ne03);
    ثابت size_t nbytes_shared =
        (GGML_PAD(ncols_x, WARP_SIZE) + WARP_SIZE) * اندازه(اعشاری);

    ثابت صحیح id       = get_current_device_id();
    ثابت size_t smpbo = ggml_sycl_info().devices[id].smpbo;

    اگر (nbytes_shared <= smpbo && ncols_x <= max_block_size) {
        launch_soft_max_kernels<32, 64, 128, 256, 512, 1024, 2048, 4096>(
            x, mask, sinks, dst, params, stream, block_dims, block_nums,
            nbytes_shared);
    } وگرنه {
        ثابت size_t nbytes_shared_کم = WARP_SIZE * اندازه(اعشاری);

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(nbytes_shared_کم), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    soft_max_f32<نادرست, 0, 0>(
                        x, mask, sinks, dst, params,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                    GGML_UNUSED(item_ct1);
                });
        });
    }
}

ایستا تهی soft_max_back_f32_sycl(ثابت اعشاری *   grad,
                                   ثابت اعشاری *   dstf,
                                   اعشاری *         dst,
                                   ثابت صحیح       ncols,
                                   ثابت صحیح       nrows,
                                   ثابت اعشاری     scale,
                                   dpct::queue_ptr stream) {
    ثابت dpct::dim3 block_dims(WARP_SIZE, 1, 1);
    ثابت dpct::dim3 block_nums(nrows, 1, 1);

    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) {
                             soft_max_back_f32(grad, dstf, dst, ncols, scale);
                             GGML_UNUSED(item_ct1);
                         });
}

تهی ggml_sycl_op_soft_max(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];
    ثابت ggml_tensor * src2 = dst->src[2];

    ثابت اعشاری * src0_d = (ثابت اعشاری *) src0->data;
    ثابت تهی  * src1_d = src1 ? (ثابت تهی *) src1->data : nullptr;
    ثابت تهی  * src2_d = src2 ? (ثابت تهی *) src2->data : nullptr;
    اعشاری       *  dst_d = (اعشاری *) dst->data;

    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    // src1 contains mask and it is optional
    GGML_ASSERT(!src1 || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32);

    ثابت int64_t nrows_x = ggml_nrows(src0);
    ثابت int64_t nrows_y = src0->ne[1];

    ثابت int64_t ne00 = src0->ne[0];

    اعشاری scale    = 1.0f;
    اعشاری max_bias = 0.0f;

    memcpy(&scale,    (ثابت اعشاری *) dst->op_params + 0, اندازه(اعشاری));
    memcpy(&max_bias, (ثابت اعشاری *) dst->op_params + 1, اندازه(اعشاری));

    ثابت منطقی use_f16 = (src1 && src1->type == GGML_TYPE_F16);

    ثابت int64_t nb11 = src1 ? src1->nb[1] : 1;
    ثابت int64_t nb12 = src1 ? src1->nb[2] : 1;
    ثابت int64_t nb13 = src1 ? src1->nb[3] : 1;

    ثابت int64_t ne12 = src1 ? src1->ne[2] : 1;
    ثابت int64_t ne13 = src1 ? src1->ne[3] : 1;

    ثابت uint32_t n_head      = src0->ne[2];
    ثابت uint32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((اعشاری) n_head));

    ثابت اعشاری m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    ثابت اعشاری m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);


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

    اگر (use_f16) {
        soft_max_f32_sycl(src0_d, (ثابت sycl::half *)src1_d,
                          (ثابت اعشاری *)src2_d, dst_d, params, stream,
                          ctx.device);
    } وگرنه {
        soft_max_f32_sycl(src0_d, (ثابت اعشاری *)src1_d, (ثابت اعشاری *)src2_d,
                          dst_d, params, stream, ctx.device);
    }
}

تهی ggml_sycl_op_soft_max_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ثابت ggml_tensor * src0 = dst->src[0]; // grad
    ثابت ggml_tensor * src1 = dst->src[1]; // forward pass output

    ثابت اعشاری * src0_d = (ثابت اعشاری *) src0->data;
    ثابت اعشاری * src1_d = (ثابت اعشاری *) src1->data;
    اعشاری       * dst_d  = (اعشاری       *) dst->data;

    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    ثابت int64_t ncols = src0->ne[0];
    ثابت int64_t nrows = ggml_nrows(src0);

    اعشاری scale    = 1.0f;
    اعشاری max_bias = 0.0f;

    memcpy(&scale,    (ثابت اعشاری *) dst->op_params + 0, اندازه(اعشاری));
    memcpy(&max_bias, (ثابت اعشاری *) dst->op_params + 1, اندازه(اعشاری));

    GGML_ASSERT(max_bias == 0.0f);

    soft_max_back_f32_sycl(src0_d, src1_d, dst_d, ncols, nrows, scale, stream);
}
