#അന്തര്ഭാവ "softmax.hpp"
#അന്തര്ഭാവ <cstdint>
#അന്തര്ഭാവ <utility>
#അന്തര്ഭാവ <cmath>


template <typename T> സ്ഥിതം __dpct_inline__ ദശാംശം t2f32(T val) {
    തിരികെ (ദശാംശം) val;
}

template <> ദശാംശം __dpct_inline__ t2f32<sycl::half>(sycl::half val) {
  തിരികെ sycl::vec<sycl::half, 1>(val)
      .convert<ദശാംശം, sycl::rounding_mode::automatic>()[0];
}

ഘടന soft_max_params {

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
    ദശാംശം scale;
    ദശാംശം max_bias;
    ദശാംശം m0;
    ദശാംശം m1;
};

// When ncols_template == 0 the bounds for the loops in this function are not known and can't be unrolled.
// As we want to keep pragma unroll for all other cases we suppress the clang transformation warning here.
#ifdef __clang__
#pragma clang diagnostic തള്ളുക
#pragma clang diagnostic ignored "-Wpass-failed"
#endif // __clang__
template <തർക്കം use_shared, പൂർണ്ണസംഖ്യ ncols_template, പൂർണ്ണസംഖ്യ block_size_template, typename T>
സ്ഥിതം ശൂന്യം soft_max_f32(സ്ഥിരം ദശാംശം *         x,
                         സ്ഥിരം T *             mask,
                         സ്ഥിരം ദശാംശം *         sinks,
                         ദശാംശം *               dst,
                         സ്ഥിരം soft_max_params p,
                         uint8_t *             dpct_local) {
    auto      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    സ്ഥിരം പൂർണ്ണസംഖ്യ ncols    = ncols_template == 0 ? p.ncols : ncols_template;
    സ്ഥിരം പൂർണ്ണസംഖ്യ block_size = block_size_template == 0
                               ? item_ct1.get_local_range(2)
                               : block_size_template;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nthreads = block_size;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nwarps = nthreads / WARP_SIZE;
    സ്ഥിരം size_t nreduce = nwarps / WARP_SIZE;

    സ്ഥിരം പൂർണ്ണസംഖ്യ tid = item_ct1.get_local_id(2);

    സ്ഥിരം int64_t i03 = item_ct1.get_group(0);
    സ്ഥിരം int64_t i02 = item_ct1.get_group(1);
    സ്ഥിരം int64_t i01 = item_ct1.get_group(2);

    //TODO: noncontigous inputs/outputs
    സ്ഥിരം പൂർണ്ണസംഖ്യ rowx = item_ct1.get_group(2) +
                     item_ct1.get_group(1) * item_ct1.get_group_range(2) +
                     item_ct1.get_group(0) * item_ct1.get_group_range(2) *
                         item_ct1.get_group_range(1);

    സ്ഥിരം int64_t i11 = i01;
    സ്ഥിരം int64_t i12 = i02 % p.ne12;
    സ്ഥിരം int64_t i13 = i03 % p.ne13;

    x    += int64_t(rowx)*ncols;
    mask += (i11*p.nb11 + i12*p.nb12 + i13*p.nb13) / വലുപ്പം(T) * (mask != nullptr);
    dst  += int64_t(rowx)*ncols;

    സ്ഥിരം പൂർണ്ണസംഖ്യ warp_id = item_ct1.get_local_id(2) / WARP_SIZE;
    സ്ഥിരം പൂർണ്ണസംഖ്യ lane_id = item_ct1.get_local_id(2) % WARP_SIZE;

    സ്ഥിരം ദശാംശം slope = get_alibi_slope(p.max_bias, i02, p.n_head_log2, p.m0, p.m1);

    ദശാംശം * buf_iw = (ദശാംശം *) dpct_local;

    // shared memory buffer to cache values between iterations:
    ദശാംശം *vals = use_shared ? buf_iw + sycl::max(nwarps, WARP_SIZE) : dst;
    ദശാംശം max_val = sinks ? sinks[i02] : -INFINITY;
#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ col0 = 0; col0 < ncols; col0 += block_size) {
        സ്ഥിരം പൂർണ്ണസംഖ്യ col = col0 + tid;

        എങ്കിൽ (ncols_template == 0 && col >= ncols) {
            നിർത്തുക;
        }

        സ്ഥിരം ദശാംശം val = x[col]*p.scale + (mask ? slope*t2f32(mask[col]) : 0.0f);

        vals[col] = val;
        max_val   = sycl::max(max_val, val);
    }
    // find the max value in the block
    max_val = warp_reduce_max<WARP_SIZE>(max_val);

    എങ്കിൽ (block_size > WARP_SIZE) {
        എങ്കിൽ (lane_id == 0) {
            buf_iw[warp_id] = max_val;
        }
        item_ct1.barrier();

        max_val = -INFINITY;
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = lane_id; i < nwarps; i += WARP_SIZE) {
            max_val = sycl::max(max_val, buf_iw[i]);
        }
        max_val = warp_reduce_max<WARP_SIZE>(max_val);
    }
    ദശാംശം tmp = 0.0f; // partial sum

#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ col0 = 0; col0 < ncols; col0 += block_size) {
        സ്ഥിരം പൂർണ്ണസംഖ്യ col = col0 + tid;

        എങ്കിൽ (ncols_template == 0 && col >= ncols) {
            നിർത്തുക;
        }

        സ്ഥിരം ദശാംശം val = sycl::native::exp(sycl::max(vals[col] - max_val, -80.0f));
        tmp += val;
        vals[col] = val;
    }
    // find the sum of exps in the block
    tmp = warp_reduce_sum<WARP_SIZE>(tmp);
    എങ്കിൽ (block_size > WARP_SIZE) {
        item_ct1.barrier();
        എങ്കിൽ (warp_id == 0) {
            buf_iw[lane_id] = 0.0f;
            വേണ്ടി (size_t i = 1; i < nreduce; i += 1) {
                buf_iw[lane_id + i * WARP_SIZE] = 0.f;
            }
        }
        item_ct1.barrier();

        എങ്കിൽ (lane_id == 0) {
            buf_iw[warp_id] = tmp;
        }
        item_ct1.barrier();

        tmp = buf_iw[lane_id];
        വേണ്ടി (size_t i = 1; i < nreduce; i += 1) {
            tmp += buf_iw[lane_id + i * WARP_SIZE];
        }
        tmp = warp_reduce_sum<WARP_SIZE>(tmp);
    }
    എങ്കിൽ (sinks) {
        tmp += sycl::native::exp(sycl::max(sinks[i02] - max_val, -80.0f));
    }
    സ്ഥിരം ദശാംശം inv_sum = 1.0f / tmp;

#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ col0 = 0; col0 < ncols; col0 += block_size) {
        സ്ഥിരം പൂർണ്ണസംഖ്യ col = col0 + tid;

        എങ്കിൽ (ncols_template == 0 && col >= ncols) {
            തിരികെ;
        }

        dst[col] = vals[col] * inv_sum;
    }
}
#ifdef __clang__
#pragma clang diagnostic എടുക്കുക
#endif // __clang__

സ്ഥിതം ശൂന്യം soft_max_back_f32(സ്ഥിരം ദശാംശം *grad, സ്ഥിരം ദശാംശം *dstf, ദശാംശം *dst,
                              സ്ഥിരം പൂർണ്ണസംഖ്യ ncols, സ്ഥിരം ദശാംശം scale) {
    auto      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    സ്ഥിരം പൂർണ്ണസംഖ്യ tid      = item_ct1.get_local_id(2);
    സ്ഥിരം പൂർണ്ണസംഖ്യ rowx     = item_ct1.get_group(2);

    grad += int64_t(rowx)*ncols;
    dstf += int64_t(rowx)*ncols;
    dst  += int64_t(rowx)*ncols;

    ദശാംശം dgf_dot = 0.0f; // dot product of dst from forward pass and gradients

    വേണ്ടി (പൂർണ്ണസംഖ്യ col = tid; col < ncols; col += WARP_SIZE) {
        dgf_dot += dstf[col]*grad[col];
    }

    dgf_dot = warp_reduce_sum<WARP_SIZE>(dgf_dot);

    വേണ്ടി (പൂർണ്ണസംഖ്യ col = tid; col < ncols; col += WARP_SIZE) {
        dst[col] = scale * (grad[col] - dgf_dot) * dstf[col];
    }
}

template <പൂർണ്ണസംഖ്യ... Ns, typename T>
സ്ഥിതം ശൂന്യം launch_soft_max_kernels(സ്ഥിരം ദശാംശം *           x,
                                    സ്ഥിരം T *               mask,
                                    സ്ഥിരം ദശാംശം *           sinks,
                                    ദശാംശം *                 dst,
                                    സ്ഥിരം soft_max_params & p,
                                    dpct::queue_ptr         stream,
                                    dpct::dim3              block_dims,
                                    dpct::dim3              block_nums,
                                    size_t                  nbytes_shared)
{
    auto launch_kernel = [=](auto I) -> തർക്കം {
        constexpr പൂർണ്ണസംഖ്യ ncols = decltype(I)::value;
        constexpr പൂർണ്ണസംഖ്യ block = (ncols > 1024 ? 1024 : ncols);
        എങ്കിൽ (p.ncols == ncols) {
            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                    sycl::range<1>(nbytes_shared), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(
                        WARP_SIZE)]] {
                        soft_max_f32<സത്യം, ncols, block>(
                            x, mask, sinks, dst, p,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                        GGML_UNUSED(item_ct1);
                    });
            });
            തിരികെ സത്യം;
        }
        തിരികെ അസത്യം;
    };

    // unary fold over launch_kernel
    എങ്കിൽ ((launch_kernel(std::integral_constant<പൂർണ്ണസംഖ്യ, Ns>{}) || ...)) {
        തിരികെ;
    }

    stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range<1>(nbytes_shared), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    soft_max_f32<സത്യം, 0, 0>(
                        x, mask, sinks, dst, p,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                    GGML_UNUSED(item_ct1);
                });
    });
}

template <typename T>
സ്ഥിതം ശൂന്യം soft_max_f32_sycl(സ്ഥിരം ദശാംശം *x, സ്ഥിരം T *mask,
                              സ്ഥിരം ദശാംശം *sinks, ദശാംശം *dst,
                              സ്ഥിരം soft_max_params &params,
                              dpct::queue_ptr stream, പൂർണ്ണസംഖ്യ device) {
    പൂർണ്ണസംഖ്യ nth = WARP_SIZE;
    പൂർണ്ണസംഖ്യ max_block_size = ggml_sycl_info().max_work_group_sizes[device];
    സ്ഥിരം int64_t ncols_x = params.ncols;

    ആകുമ്പോൾ (nth < ncols_x && nth < max_block_size) nth *= 2;
    എങ്കിൽ (nth>max_block_size) nth = max_block_size;

    സ്ഥിരം dpct::dim3 block_dims(nth, 1, 1);
    സ്ഥിരം dpct::dim3 block_nums(params.ne01, params.ne02, params.ne03);
    സ്ഥിരം size_t nbytes_shared =
        (GGML_PAD(ncols_x, WARP_SIZE) + WARP_SIZE) * വലുപ്പം(ദശാംശം);

    സ്ഥിരം പൂർണ്ണസംഖ്യ id       = get_current_device_id();
    സ്ഥിരം size_t smpbo = ggml_sycl_info().devices[id].smpbo;

    എങ്കിൽ (nbytes_shared <= smpbo && ncols_x <= max_block_size) {
        launch_soft_max_kernels<32, 64, 128, 256, 512, 1024, 2048, 4096>(
            x, mask, sinks, dst, params, stream, block_dims, block_nums,
            nbytes_shared);
    } അല്ലെങ്കിൽ {
        സ്ഥിരം size_t nbytes_shared_താഴെയുള്ള = WARP_SIZE * വലുപ്പം(ദശാംശം);

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(nbytes_shared_താഴെയുള്ള), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    soft_max_f32<അസത്യം, 0, 0>(
                        x, mask, sinks, dst, params,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                    GGML_UNUSED(item_ct1);
                });
        });
    }
}

സ്ഥിതം ശൂന്യം soft_max_back_f32_sycl(സ്ഥിരം ദശാംശം *   grad,
                                   സ്ഥിരം ദശാംശം *   dstf,
                                   ദശാംശം *         dst,
                                   സ്ഥിരം പൂർണ്ണസംഖ്യ       ncols,
                                   സ്ഥിരം പൂർണ്ണസംഖ്യ       nrows,
                                   സ്ഥിരം ദശാംശം     scale,
                                   dpct::queue_ptr stream) {
    സ്ഥിരം dpct::dim3 block_dims(WARP_SIZE, 1, 1);
    സ്ഥിരം dpct::dim3 block_nums(nrows, 1, 1);

    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) {
                             soft_max_back_f32(grad, dstf, dst, ncols, scale);
                             GGML_UNUSED(item_ct1);
                         });
}

ശൂന്യം ggml_sycl_op_soft_max(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];
    സ്ഥിരം ggml_tensor * src2 = dst->src[2];

    സ്ഥിരം ദശാംശം * src0_d = (സ്ഥിരം ദശാംശം *) src0->data;
    സ്ഥിരം ശൂന്യം  * src1_d = src1 ? (സ്ഥിരം ശൂന്യം *) src1->data : nullptr;
    സ്ഥിരം ശൂന്യം  * src2_d = src2 ? (സ്ഥിരം ശൂന്യം *) src2->data : nullptr;
    ദശാംശം       *  dst_d = (ദശാംശം *) dst->data;

    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    // src1 contains mask and it is optional
    GGML_ASSERT(!src1 || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32);

    സ്ഥിരം int64_t nrows_x = ggml_nrows(src0);
    സ്ഥിരം int64_t nrows_y = src0->ne[1];

    സ്ഥിരം int64_t ne00 = src0->ne[0];

    ദശാംശം scale    = 1.0f;
    ദശാംശം max_bias = 0.0f;

    memcpy(&scale,    (സ്ഥിരം ദശാംശം *) dst->op_params + 0, വലുപ്പം(ദശാംശം));
    memcpy(&max_bias, (സ്ഥിരം ദശാംശം *) dst->op_params + 1, വലുപ്പം(ദശാംശം));

    സ്ഥിരം തർക്കം use_f16 = (src1 && src1->type == GGML_TYPE_F16);

    സ്ഥിരം int64_t nb11 = src1 ? src1->nb[1] : 1;
    സ്ഥിരം int64_t nb12 = src1 ? src1->nb[2] : 1;
    സ്ഥിരം int64_t nb13 = src1 ? src1->nb[3] : 1;

    സ്ഥിരം int64_t ne12 = src1 ? src1->ne[2] : 1;
    സ്ഥിരം int64_t ne13 = src1 ? src1->ne[3] : 1;

    സ്ഥിരം uint32_t n_head      = src0->ne[2];
    സ്ഥിരം uint32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((ദശാംശം) n_head));

    സ്ഥിരം ദശാംശം m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    സ്ഥിരം ദശാംശം m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);


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

    എങ്കിൽ (use_f16) {
        soft_max_f32_sycl(src0_d, (സ്ഥിരം sycl::half *)src1_d,
                          (സ്ഥിരം ദശാംശം *)src2_d, dst_d, params, stream,
                          ctx.device);
    } അല്ലെങ്കിൽ {
        soft_max_f32_sycl(src0_d, (സ്ഥിരം ദശാംശം *)src1_d, (സ്ഥിരം ദശാംശം *)src2_d,
                          dst_d, params, stream, ctx.device);
    }
}

ശൂന്യം ggml_sycl_op_soft_max_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    സ്ഥിരം ggml_tensor * src0 = dst->src[0]; // grad
    സ്ഥിരം ggml_tensor * src1 = dst->src[1]; // forward pass output

    സ്ഥിരം ദശാംശം * src0_d = (സ്ഥിരം ദശാംശം *) src0->data;
    സ്ഥിരം ദശാംശം * src1_d = (സ്ഥിരം ദശാംശം *) src1->data;
    ദശാംശം       * dst_d  = (ദശാംശം       *) dst->data;

    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    സ്ഥിരം int64_t ncols = src0->ne[0];
    സ്ഥിരം int64_t nrows = ggml_nrows(src0);

    ദശാംശം scale    = 1.0f;
    ദശാംശം max_bias = 0.0f;

    memcpy(&scale,    (സ്ഥിരം ദശാംശം *) dst->op_params + 0, വലുപ്പം(ദശാംശം));
    memcpy(&max_bias, (സ്ഥിരം ദശാംശം *) dst->op_params + 1, വലുപ്പം(ദശാംശം));

    GGML_ASSERT(max_bias == 0.0f);

    soft_max_back_f32_sycl(src0_d, src1_d, dst_d, ncols, nrows, scale, stream);
}
