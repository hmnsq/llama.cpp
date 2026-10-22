#अन्तर्भाव "softmax.hpp"
#अन्तर्भाव <cstdint>
#अन्तर्भाव <utility>
#अन्तर्भाव <cmath>


ढाँचा <typename T> स्थिर __dpct_inline__ भिन्न t2f32(T val) {
    प्रत्यागम (भिन्न) val;
}

ढाँचा <> भिन्न __dpct_inline__ t2f32<sycl::half>(sycl::half val) {
  प्रत्यागम sycl::vec<sycl::half, 1>(val)
      .convert<भिन्न, sycl::rounding_mode::automatic>()[0];
}

संरचना soft_max_params {

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
    भिन्न scale;
    भिन्न max_bias;
    भिन्न m0;
    भिन्न m1;
};

// When ncols_template == 0 the bounds for the loops in this function are not known and can't be unrolled.
// As we want to keep pragma unroll for all other cases we suppress the clang transformation warning here.
#ifdef __clang__
#pragma clang diagnostic धकेलो
#pragma clang diagnostic ignored "-Wpass-failed"
#endif // __clang__
ढाँचा <तार्किक use_shared, पूर्णाङ्क ncols_template, पूर्णाङ्क block_size_template, typename T>
स्थिर शून्य soft_max_f32(ध्रुव भिन्न *         x,
                         ध्रुव T *             mask,
                         ध्रुव भिन्न *         sinks,
                         भिन्न *               dst,
                         ध्रुव soft_max_params p,
                         uint8_t *             dpct_local) {
    स्वतः      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ध्रुव पूर्णाङ्क ncols    = ncols_template == 0 ? p.ncols : ncols_template;
    ध्रुव पूर्णाङ्क block_size = block_size_template == 0
                               ? item_ct1.get_local_range(2)
                               : block_size_template;
    ध्रुव पूर्णाङ्क nthreads = block_size;
    ध्रुव पूर्णाङ्क nwarps = nthreads / WARP_SIZE;
    ध्रुव माप_प्रकार nreduce = nwarps / WARP_SIZE;

    ध्रुव पूर्णाङ्क tid = item_ct1.get_local_id(2);

    ध्रुव int64_t i03 = item_ct1.get_group(0);
    ध्रुव int64_t i02 = item_ct1.get_group(1);
    ध्रुव int64_t i01 = item_ct1.get_group(2);

    //TODO: noncontigous inputs/outputs
    ध्रुव पूर्णाङ्क rowx = item_ct1.get_group(2) +
                     item_ct1.get_group(1) * item_ct1.get_group_range(2) +
                     item_ct1.get_group(0) * item_ct1.get_group_range(2) *
                         item_ct1.get_group_range(1);

    ध्रुव int64_t i11 = i01;
    ध्रुव int64_t i12 = i02 % p.ne12;
    ध्रुव int64_t i13 = i03 % p.ne13;

    x    += int64_t(rowx)*ncols;
    mask += (i11*p.nb11 + i12*p.nb12 + i13*p.nb13) / परिमाण(T) * (mask != nullptr);
    dst  += int64_t(rowx)*ncols;

    ध्रुव पूर्णाङ्क warp_id = item_ct1.get_local_id(2) / WARP_SIZE;
    ध्रुव पूर्णाङ्क lane_id = item_ct1.get_local_id(2) % WARP_SIZE;

    ध्रुव भिन्न slope = get_alibi_slope(p.max_bias, i02, p.n_head_log2, p.m0, p.m1);

    भिन्न * buf_iw = (भिन्न *) dpct_local;

    // shared memory buffer to cache values between iterations:
    भिन्न *vals = use_shared ? buf_iw + sycl::max(nwarps, WARP_SIZE) : dst;
    भिन्न max_val = sinks ? sinks[i02] : -अनन्त;
#pragma unroll
    प्रति (पूर्णाङ्क col0 = 0; col0 < ncols; col0 += block_size) {
        ध्रुव पूर्णाङ्क col = col0 + tid;

        यदि (ncols_template == 0 && col >= ncols) {
            भङ्ग;
        }

        ध्रुव भिन्न val = x[col]*p.scale + (mask ? slope*t2f32(mask[col]) : 0.0f);

        vals[col] = val;
        max_val   = sycl::max(max_val, val);
    }
    // find the max value in the block
    max_val = warp_reduce_max<WARP_SIZE>(max_val);

    यदि (block_size > WARP_SIZE) {
        यदि (lane_id == 0) {
            buf_iw[warp_id] = max_val;
        }
        item_ct1.barrier();

        max_val = -अनन्त;
        प्रति (पूर्णाङ्क i = lane_id; i < nwarps; i += WARP_SIZE) {
            max_val = sycl::max(max_val, buf_iw[i]);
        }
        max_val = warp_reduce_max<WARP_SIZE>(max_val);
    }
    भिन्न tmp = 0.0f; // partial sum

#pragma unroll
    प्रति (पूर्णाङ्क col0 = 0; col0 < ncols; col0 += block_size) {
        ध्रुव पूर्णाङ्क col = col0 + tid;

        यदि (ncols_template == 0 && col >= ncols) {
            भङ्ग;
        }

        ध्रुव भिन्न val = sycl::native::exp(sycl::max(vals[col] - max_val, -80.0f));
        tmp += val;
        vals[col] = val;
    }
    // find the sum of exps in the block
    tmp = warp_reduce_sum<WARP_SIZE>(tmp);
    यदि (block_size > WARP_SIZE) {
        item_ct1.barrier();
        यदि (warp_id == 0) {
            buf_iw[lane_id] = 0.0f;
            प्रति (माप_प्रकार i = 1; i < nreduce; i += 1) {
                buf_iw[lane_id + i * WARP_SIZE] = 0.f;
            }
        }
        item_ct1.barrier();

        यदि (lane_id == 0) {
            buf_iw[warp_id] = tmp;
        }
        item_ct1.barrier();

        tmp = buf_iw[lane_id];
        प्रति (माप_प्रकार i = 1; i < nreduce; i += 1) {
            tmp += buf_iw[lane_id + i * WARP_SIZE];
        }
        tmp = warp_reduce_sum<WARP_SIZE>(tmp);
    }
    यदि (sinks) {
        tmp += sycl::native::exp(sycl::max(sinks[i02] - max_val, -80.0f));
    }
    ध्रुव भिन्न inv_sum = 1.0f / tmp;

#pragma unroll
    प्रति (पूर्णाङ्क col0 = 0; col0 < ncols; col0 += block_size) {
        ध्रुव पूर्णाङ्क col = col0 + tid;

        यदि (ncols_template == 0 && col >= ncols) {
            प्रत्यागम;
        }

        dst[col] = vals[col] * inv_sum;
    }
}
#ifdef __clang__
#pragma clang diagnostic निकालो
#endif // __clang__

स्थिर शून्य soft_max_back_f32(ध्रुव भिन्न *grad, ध्रुव भिन्न *dstf, भिन्न *dst,
                              ध्रुव पूर्णाङ्क ncols, ध्रुव भिन्न scale) {
    स्वतः      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ध्रुव पूर्णाङ्क tid      = item_ct1.get_local_id(2);
    ध्रुव पूर्णाङ्क rowx     = item_ct1.get_group(2);

    grad += int64_t(rowx)*ncols;
    dstf += int64_t(rowx)*ncols;
    dst  += int64_t(rowx)*ncols;

    भिन्न dgf_dot = 0.0f; // dot product of dst from forward pass and gradients

    प्रति (पूर्णाङ्क col = tid; col < ncols; col += WARP_SIZE) {
        dgf_dot += dstf[col]*grad[col];
    }

    dgf_dot = warp_reduce_sum<WARP_SIZE>(dgf_dot);

    प्रति (पूर्णाङ्क col = tid; col < ncols; col += WARP_SIZE) {
        dst[col] = scale * (grad[col] - dgf_dot) * dstf[col];
    }
}

ढाँचा <पूर्णाङ्क... Ns, typename T>
स्थिर शून्य launch_soft_max_kernels(ध्रुव भिन्न *           x,
                                    ध्रुव T *               mask,
                                    ध्रुव भिन्न *           sinks,
                                    भिन्न *                 dst,
                                    ध्रुव soft_max_params & p,
                                    dpct::queue_ptr         stream,
                                    dpct::dim3              block_dims,
                                    dpct::dim3              block_nums,
                                    माप_प्रकार                  nbytes_shared)
{
    स्वतः launch_kernel = [=](स्वतः I) -> तार्किक {
        constexpr पूर्णाङ्क ncols = decltype(I)::value;
        constexpr पूर्णाङ्क block = (ncols > 1024 ? 1024 : ncols);
        यदि (p.ncols == ncols) {
            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                    sycl::range<1>(nbytes_shared), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(
                        WARP_SIZE)]] {
                        soft_max_f32<सत्य, ncols, block>(
                            x, mask, sinks, dst, p,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                        GGML_UNUSED(item_ct1);
                    });
            });
            प्रत्यागम सत्य;
        }
        प्रत्यागम असत्य;
    };

    // unary fold over launch_kernel
    यदि ((launch_kernel(std::integral_constant<पूर्णाङ्क, Ns>{}) || ...)) {
        प्रत्यागम;
    }

    stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range<1>(nbytes_shared), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    soft_max_f32<सत्य, 0, 0>(
                        x, mask, sinks, dst, p,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                    GGML_UNUSED(item_ct1);
                });
    });
}

ढाँचा <typename T>
स्थिर शून्य soft_max_f32_sycl(ध्रुव भिन्न *x, ध्रुव T *mask,
                              ध्रुव भिन्न *sinks, भिन्न *dst,
                              ध्रुव soft_max_params &params,
                              dpct::queue_ptr stream, पूर्णाङ्क device) {
    पूर्णाङ्क nth = WARP_SIZE;
    पूर्णाङ्क max_block_size = ggml_sycl_info().max_work_group_sizes[device];
    ध्रुव int64_t ncols_x = params.ncols;

    यावत् (nth < ncols_x && nth < max_block_size) nth *= 2;
    यदि (nth>max_block_size) nth = max_block_size;

    ध्रुव dpct::dim3 block_dims(nth, 1, 1);
    ध्रुव dpct::dim3 block_nums(params.ne01, params.ne02, params.ne03);
    ध्रुव माप_प्रकार nbytes_shared =
        (GGML_PAD(ncols_x, WARP_SIZE) + WARP_SIZE) * परिमाण(भिन्न);

    ध्रुव पूर्णाङ्क id       = get_current_device_id();
    ध्रुव माप_प्रकार smpbo = ggml_sycl_info().devices[id].smpbo;

    यदि (nbytes_shared <= smpbo && ncols_x <= max_block_size) {
        launch_soft_max_kernels<32, 64, 128, 256, 512, 1024, 2048, 4096>(
            x, mask, sinks, dst, params, stream, block_dims, block_nums,
            nbytes_shared);
    } अन्यथा {
        ध्रुव माप_प्रकार nbytes_shared_निचला = WARP_SIZE * परिमाण(भिन्न);

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(nbytes_shared_निचला), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    soft_max_f32<असत्य, 0, 0>(
                        x, mask, sinks, dst, params,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                    GGML_UNUSED(item_ct1);
                });
        });
    }
}

स्थिर शून्य soft_max_back_f32_sycl(ध्रुव भिन्न *   grad,
                                   ध्रुव भिन्न *   dstf,
                                   भिन्न *         dst,
                                   ध्रुव पूर्णाङ्क       ncols,
                                   ध्रुव पूर्णाङ्क       nrows,
                                   ध्रुव भिन्न     scale,
                                   dpct::queue_ptr stream) {
    ध्रुव dpct::dim3 block_dims(WARP_SIZE, 1, 1);
    ध्रुव dpct::dim3 block_nums(nrows, 1, 1);

    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) {
                             soft_max_back_f32(grad, dstf, dst, ncols, scale);
                             GGML_UNUSED(item_ct1);
                         });
}

शून्य ggml_sycl_op_soft_max(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];
    ध्रुव ggml_tensor * src2 = dst->src[2];

    ध्रुव भिन्न * src0_d = (ध्रुव भिन्न *) src0->data;
    ध्रुव शून्य  * src1_d = src1 ? (ध्रुव शून्य *) src1->data : nullptr;
    ध्रुव शून्य  * src2_d = src2 ? (ध्रुव शून्य *) src2->data : nullptr;
    भिन्न       *  dst_d = (भिन्न *) dst->data;

    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    // src1 contains mask and it is optional
    GGML_ASSERT(!src1 || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32);

    ध्रुव int64_t nrows_x = ggml_nrows(src0);
    ध्रुव int64_t nrows_y = src0->ne[1];

    ध्रुव int64_t ne00 = src0->ne[0];

    भिन्न scale    = 1.0f;
    भिन्न max_bias = 0.0f;

    स_नकल(&scale,    (ध्रुव भिन्न *) dst->op_params + 0, परिमाण(भिन्न));
    स_नकल(&max_bias, (ध्रुव भिन्न *) dst->op_params + 1, परिमाण(भिन्न));

    ध्रुव तार्किक use_f16 = (src1 && src1->type == GGML_TYPE_F16);

    ध्रुव int64_t nb11 = src1 ? src1->nb[1] : 1;
    ध्रुव int64_t nb12 = src1 ? src1->nb[2] : 1;
    ध्रुव int64_t nb13 = src1 ? src1->nb[3] : 1;

    ध्रुव int64_t ne12 = src1 ? src1->ne[2] : 1;
    ध्रुव int64_t ne13 = src1 ? src1->ne[3] : 1;

    ध्रुव uint32_t n_head      = src0->ne[2];
    ध्रुव uint32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((भिन्न) n_head));

    ध्रुव भिन्न m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    ध्रुव भिन्न m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);


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

    यदि (use_f16) {
        soft_max_f32_sycl(src0_d, (ध्रुव sycl::half *)src1_d,
                          (ध्रुव भिन्न *)src2_d, dst_d, params, stream,
                          ctx.device);
    } अन्यथा {
        soft_max_f32_sycl(src0_d, (ध्रुव भिन्न *)src1_d, (ध्रुव भिन्न *)src2_d,
                          dst_d, params, stream, ctx.device);
    }
}

शून्य ggml_sycl_op_soft_max_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ध्रुव ggml_tensor * src0 = dst->src[0]; // grad
    ध्रुव ggml_tensor * src1 = dst->src[1]; // forward pass output

    ध्रुव भिन्न * src0_d = (ध्रुव भिन्न *) src0->data;
    ध्रुव भिन्न * src1_d = (ध्रुव भिन्न *) src1->data;
    भिन्न       * dst_d  = (भिन्न       *) dst->data;

    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    ध्रुव int64_t ncols = src0->ne[0];
    ध्रुव int64_t nrows = ggml_nrows(src0);

    भिन्न scale    = 1.0f;
    भिन्न max_bias = 0.0f;

    स_नकल(&scale,    (ध्रुव भिन्न *) dst->op_params + 0, परिमाण(भिन्न));
    स_नकल(&max_bias, (ध्रुव भिन्न *) dst->op_params + 1, परिमाण(भिन्न));

    GGML_ASSERT(max_bias == 0.0f);

    soft_max_back_f32_sycl(src0_d, src1_d, dst_d, ncols, nrows, scale, stream);
}
