#અન્તર્ભાવ "softmax.hpp"
#અન્તર્ભાવ <cstdint>
#અન્તર્ભાવ <utility>
#અન્તર્ભાવ <cmath>


template <typename T> સ્થિર __dpct_inline__ દશાંશ t2f32(T val) {
    પાછા (દશાંશ) val;
}

template <> દશાંશ __dpct_inline__ t2f32<sycl::half>(sycl::half val) {
  પાછા sycl::vec<sycl::half, 1>(val)
      .convert<દશાંશ, sycl::rounding_mode::automatic>()[0];
}

સંરચના soft_max_params {

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
    દશાંશ scale;
    દશાંશ max_bias;
    દશાંશ m0;
    દશાંશ m1;
};

// When ncols_template == 0 the bounds for the loops in this function are not known and can't be unrolled.
// As we want to keep pragma unroll for all other cases we suppress the clang transformation warning here.
#ifdef __clang__
#pragma clang diagnostic ધકેલો
#pragma clang diagnostic ignored "-Wpass-failed"
#endif // __clang__
template <તાર્કિક use_shared, પૂર્ણાંક ncols_template, પૂર્ણાંક block_size_template, typename T>
સ્થિર શૂન્ય soft_max_f32(અચલ દશાંશ *         x,
                         અચલ T *             mask,
                         અચલ દશાંશ *         sinks,
                         દશાંશ *               dst,
                         અચલ soft_max_params p,
                         uint8_t *             dpct_local) {
    auto      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    અચલ પૂર્ણાંક ncols    = ncols_template == 0 ? p.ncols : ncols_template;
    અચલ પૂર્ણાંક block_size = block_size_template == 0
                               ? item_ct1.get_local_range(2)
                               : block_size_template;
    અચલ પૂર્ણાંક nthreads = block_size;
    અચલ પૂર્ણાંક nwarps = nthreads / WARP_SIZE;
    અચલ size_t nreduce = nwarps / WARP_SIZE;

    અચલ પૂર્ણાંક tid = item_ct1.get_local_id(2);

    અચલ int64_t i03 = item_ct1.get_group(0);
    અચલ int64_t i02 = item_ct1.get_group(1);
    અચલ int64_t i01 = item_ct1.get_group(2);

    //TODO: noncontigous inputs/outputs
    અચલ પૂર્ણાંક rowx = item_ct1.get_group(2) +
                     item_ct1.get_group(1) * item_ct1.get_group_range(2) +
                     item_ct1.get_group(0) * item_ct1.get_group_range(2) *
                         item_ct1.get_group_range(1);

    અચલ int64_t i11 = i01;
    અચલ int64_t i12 = i02 % p.ne12;
    અચલ int64_t i13 = i03 % p.ne13;

    x    += int64_t(rowx)*ncols;
    mask += (i11*p.nb11 + i12*p.nb12 + i13*p.nb13) / કદ(T) * (mask != nullptr);
    dst  += int64_t(rowx)*ncols;

    અચલ પૂર્ણાંક warp_id = item_ct1.get_local_id(2) / WARP_SIZE;
    અચલ પૂર્ણાંક lane_id = item_ct1.get_local_id(2) % WARP_SIZE;

    અચલ દશાંશ slope = get_alibi_slope(p.max_bias, i02, p.n_head_log2, p.m0, p.m1);

    દશાંશ * buf_iw = (દશાંશ *) dpct_local;

    // shared memory buffer to cache values between iterations:
    દશાંશ *vals = use_shared ? buf_iw + sycl::max(nwarps, WARP_SIZE) : dst;
    દશાંશ max_val = sinks ? sinks[i02] : -INFINITY;
#pragma unroll
    માટે (પૂર્ણાંક col0 = 0; col0 < ncols; col0 += block_size) {
        અચલ પૂર્ણાંક col = col0 + tid;

        જો (ncols_template == 0 && col >= ncols) {
            તોડો;
        }

        અચલ દશાંશ val = x[col]*p.scale + (mask ? slope*t2f32(mask[col]) : 0.0f);

        vals[col] = val;
        max_val   = sycl::max(max_val, val);
    }
    // find the max value in the block
    max_val = warp_reduce_max<WARP_SIZE>(max_val);

    જો (block_size > WARP_SIZE) {
        જો (lane_id == 0) {
            buf_iw[warp_id] = max_val;
        }
        item_ct1.barrier();

        max_val = -INFINITY;
        માટે (પૂર્ણાંક i = lane_id; i < nwarps; i += WARP_SIZE) {
            max_val = sycl::max(max_val, buf_iw[i]);
        }
        max_val = warp_reduce_max<WARP_SIZE>(max_val);
    }
    દશાંશ tmp = 0.0f; // partial sum

#pragma unroll
    માટે (પૂર્ણાંક col0 = 0; col0 < ncols; col0 += block_size) {
        અચલ પૂર્ણાંક col = col0 + tid;

        જો (ncols_template == 0 && col >= ncols) {
            તોડો;
        }

        અચલ દશાંશ val = sycl::native::exp(sycl::max(vals[col] - max_val, -80.0f));
        tmp += val;
        vals[col] = val;
    }
    // find the sum of exps in the block
    tmp = warp_reduce_sum<WARP_SIZE>(tmp);
    જો (block_size > WARP_SIZE) {
        item_ct1.barrier();
        જો (warp_id == 0) {
            buf_iw[lane_id] = 0.0f;
            માટે (size_t i = 1; i < nreduce; i += 1) {
                buf_iw[lane_id + i * WARP_SIZE] = 0.f;
            }
        }
        item_ct1.barrier();

        જો (lane_id == 0) {
            buf_iw[warp_id] = tmp;
        }
        item_ct1.barrier();

        tmp = buf_iw[lane_id];
        માટે (size_t i = 1; i < nreduce; i += 1) {
            tmp += buf_iw[lane_id + i * WARP_SIZE];
        }
        tmp = warp_reduce_sum<WARP_SIZE>(tmp);
    }
    જો (sinks) {
        tmp += sycl::native::exp(sycl::max(sinks[i02] - max_val, -80.0f));
    }
    અચલ દશાંશ inv_sum = 1.0f / tmp;

#pragma unroll
    માટે (પૂર્ણાંક col0 = 0; col0 < ncols; col0 += block_size) {
        અચલ પૂર્ણાંક col = col0 + tid;

        જો (ncols_template == 0 && col >= ncols) {
            પાછા;
        }

        dst[col] = vals[col] * inv_sum;
    }
}
#ifdef __clang__
#pragma clang diagnostic કાઢો
#endif // __clang__

સ્થિર શૂન્ય soft_max_back_f32(અચલ દશાંશ *grad, અચલ દશાંશ *dstf, દશાંશ *dst,
                              અચલ પૂર્ણાંક ncols, અચલ દશાંશ scale) {
    auto      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    અચલ પૂર્ણાંક tid      = item_ct1.get_local_id(2);
    અચલ પૂર્ણાંક rowx     = item_ct1.get_group(2);

    grad += int64_t(rowx)*ncols;
    dstf += int64_t(rowx)*ncols;
    dst  += int64_t(rowx)*ncols;

    દશાંશ dgf_dot = 0.0f; // dot product of dst from forward pass and gradients

    માટે (પૂર્ણાંક col = tid; col < ncols; col += WARP_SIZE) {
        dgf_dot += dstf[col]*grad[col];
    }

    dgf_dot = warp_reduce_sum<WARP_SIZE>(dgf_dot);

    માટે (પૂર્ણાંક col = tid; col < ncols; col += WARP_SIZE) {
        dst[col] = scale * (grad[col] - dgf_dot) * dstf[col];
    }
}

template <પૂર્ણાંક... Ns, typename T>
સ્થિર શૂન્ય launch_soft_max_kernels(અચલ દશાંશ *           x,
                                    અચલ T *               mask,
                                    અચલ દશાંશ *           sinks,
                                    દશાંશ *                 dst,
                                    અચલ soft_max_params & p,
                                    dpct::queue_ptr         stream,
                                    dpct::dim3              block_dims,
                                    dpct::dim3              block_nums,
                                    size_t                  nbytes_shared)
{
    auto launch_kernel = [=](auto I) -> તાર્કિક {
        constexpr પૂર્ણાંક ncols = decltype(I)::value;
        constexpr પૂર્ણાંક block = (ncols > 1024 ? 1024 : ncols);
        જો (p.ncols == ncols) {
            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                    sycl::range<1>(nbytes_shared), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(
                        WARP_SIZE)]] {
                        soft_max_f32<સાચું, ncols, block>(
                            x, mask, sinks, dst, p,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                        GGML_UNUSED(item_ct1);
                    });
            });
            પાછા સાચું;
        }
        પાછા ખોટું;
    };

    // unary fold over launch_kernel
    જો ((launch_kernel(std::integral_constant<પૂર્ણાંક, Ns>{}) || ...)) {
        પાછા;
    }

    stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range<1>(nbytes_shared), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    soft_max_f32<સાચું, 0, 0>(
                        x, mask, sinks, dst, p,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                    GGML_UNUSED(item_ct1);
                });
    });
}

template <typename T>
સ્થિર શૂન્ય soft_max_f32_sycl(અચલ દશાંશ *x, અચલ T *mask,
                              અચલ દશાંશ *sinks, દશાંશ *dst,
                              અચલ soft_max_params &params,
                              dpct::queue_ptr stream, પૂર્ણાંક device) {
    પૂર્ણાંક nth = WARP_SIZE;
    પૂર્ણાંક max_block_size = ggml_sycl_info().max_work_group_sizes[device];
    અચલ int64_t ncols_x = params.ncols;

    જ્યારે (nth < ncols_x && nth < max_block_size) nth *= 2;
    જો (nth>max_block_size) nth = max_block_size;

    અચલ dpct::dim3 block_dims(nth, 1, 1);
    અચલ dpct::dim3 block_nums(params.ne01, params.ne02, params.ne03);
    અચલ size_t nbytes_shared =
        (GGML_PAD(ncols_x, WARP_SIZE) + WARP_SIZE) * કદ(દશાંશ);

    અચલ પૂર્ણાંક id       = get_current_device_id();
    અચલ size_t smpbo = ggml_sycl_info().devices[id].smpbo;

    જો (nbytes_shared <= smpbo && ncols_x <= max_block_size) {
        launch_soft_max_kernels<32, 64, 128, 256, 512, 1024, 2048, 4096>(
            x, mask, sinks, dst, params, stream, block_dims, block_nums,
            nbytes_shared);
    } નહીંતર {
        અચલ size_t nbytes_shared_નીચલું = WARP_SIZE * કદ(દશાંશ);

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(nbytes_shared_નીચલું), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    soft_max_f32<ખોટું, 0, 0>(
                        x, mask, sinks, dst, params,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                    GGML_UNUSED(item_ct1);
                });
        });
    }
}

સ્થિર શૂન્ય soft_max_back_f32_sycl(અચલ દશાંશ *   grad,
                                   અચલ દશાંશ *   dstf,
                                   દશાંશ *         dst,
                                   અચલ પૂર્ણાંક       ncols,
                                   અચલ પૂર્ણાંક       nrows,
                                   અચલ દશાંશ     scale,
                                   dpct::queue_ptr stream) {
    અચલ dpct::dim3 block_dims(WARP_SIZE, 1, 1);
    અચલ dpct::dim3 block_nums(nrows, 1, 1);

    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) {
                             soft_max_back_f32(grad, dstf, dst, ncols, scale);
                             GGML_UNUSED(item_ct1);
                         });
}

શૂન્ય ggml_sycl_op_soft_max(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ ggml_tensor * src1 = dst->src[1];
    અચલ ggml_tensor * src2 = dst->src[2];

    અચલ દશાંશ * src0_d = (અચલ દશાંશ *) src0->data;
    અચલ શૂન્ય  * src1_d = src1 ? (અચલ શૂન્ય *) src1->data : nullptr;
    અચલ શૂન્ય  * src2_d = src2 ? (અચલ શૂન્ય *) src2->data : nullptr;
    દશાંશ       *  dst_d = (દશાંશ *) dst->data;

    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    // src1 contains mask and it is optional
    GGML_ASSERT(!src1 || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32);

    અચલ int64_t nrows_x = ggml_nrows(src0);
    અચલ int64_t nrows_y = src0->ne[1];

    અચલ int64_t ne00 = src0->ne[0];

    દશાંશ scale    = 1.0f;
    દશાંશ max_bias = 0.0f;

    memcpy(&scale,    (અચલ દશાંશ *) dst->op_params + 0, કદ(દશાંશ));
    memcpy(&max_bias, (અચલ દશાંશ *) dst->op_params + 1, કદ(દશાંશ));

    અચલ તાર્કિક use_f16 = (src1 && src1->type == GGML_TYPE_F16);

    અચલ int64_t nb11 = src1 ? src1->nb[1] : 1;
    અચલ int64_t nb12 = src1 ? src1->nb[2] : 1;
    અચલ int64_t nb13 = src1 ? src1->nb[3] : 1;

    અચલ int64_t ne12 = src1 ? src1->ne[2] : 1;
    અચલ int64_t ne13 = src1 ? src1->ne[3] : 1;

    અચલ uint32_t n_head      = src0->ne[2];
    અચલ uint32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((દશાંશ) n_head));

    અચલ દશાંશ m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    અચલ દશાંશ m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);


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

    જો (use_f16) {
        soft_max_f32_sycl(src0_d, (અચલ sycl::half *)src1_d,
                          (અચલ દશાંશ *)src2_d, dst_d, params, stream,
                          ctx.device);
    } નહીંતર {
        soft_max_f32_sycl(src0_d, (અચલ દશાંશ *)src1_d, (અચલ દશાંશ *)src2_d,
                          dst_d, params, stream, ctx.device);
    }
}

શૂન્ય ggml_sycl_op_soft_max_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    અચલ ggml_tensor * src0 = dst->src[0]; // grad
    અચલ ggml_tensor * src1 = dst->src[1]; // forward pass output

    અચલ દશાંશ * src0_d = (અચલ દશાંશ *) src0->data;
    અચલ દશાંશ * src1_d = (અચલ દશાંશ *) src1->data;
    દશાંશ       * dst_d  = (દશાંશ       *) dst->data;

    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    અચલ int64_t ncols = src0->ne[0];
    અચલ int64_t nrows = ggml_nrows(src0);

    દશાંશ scale    = 1.0f;
    દશાંશ max_bias = 0.0f;

    memcpy(&scale,    (અચલ દશાંશ *) dst->op_params + 0, કદ(દશાંશ));
    memcpy(&max_bias, (અચલ દશાંશ *) dst->op_params + 1, કદ(દશાંશ));

    GGML_ASSERT(max_bias == 0.0f);

    soft_max_back_f32_sycl(src0_d, src1_d, dst_d, ncols, nrows, scale, stream);
}
