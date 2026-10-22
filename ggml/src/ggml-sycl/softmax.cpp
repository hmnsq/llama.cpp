#समावेश "softmax.hpp"
#समावेश <cstdint>
#समावेश <utility>
#समावेश <cmath>


ढाँचा <typename T> अटल __dpct_inline__ भग्न t2f32(T val) {
    वापस (भग्न) val;
}

ढाँचा <> भग्न __dpct_inline__ t2f32<sycl::half>(sycl::half val) {
  वापस sycl::vec<sycl::half, 1>(val)
      .convert<भग्न, sycl::rounding_mode::automatic>()[0];
}

काष्ठा soft_max_params {

    int64_t nheads;
    uint32_t n_शीर्ष_log2;
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
    भग्न scale;
    भग्न max_bias;
    भग्न m0;
    भग्न m1;
};

// When ncols_template == 0 the bounds for the loops in this function are not known and can't be unrolled.
// As we want to keep pragma unroll for all other cases we suppress the clang transformation warning here.
#अगर_घोषित __clang__
#आशय clang diagnostic धकेलो
#आशय clang diagnostic ignored "-Wpass-failed"
#पूर्ण_अगर // __clang__
ढाँचा <bool use_shared, पूर्णांक ncols_template, पूर्णांक block_आकार_template, typename T>
अटल व्योम soft_max_f32(स्थिर भग्न *         x,
                         स्थिर T *             mask,
                         स्थिर भग्न *         sinks,
                         भग्न *               dst,
                         स्थिर soft_max_params p,
                         uint8_t *             dpct_local) {
    स्वतः      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    स्थिर पूर्णांक ncols    = ncols_template == 0 ? p.ncols : ncols_template;
    स्थिर पूर्णांक block_आकार_2 = block_आकार_template == 0
                               ? item_ct1.get_local_range(2)
                               : block_आकार_template;
    स्थिर पूर्णांक nthreads = block_आकार_2;
    स्थिर पूर्णांक nwarps = nthreads / WARP_आकार;
    स्थिर माप_प्रकार nreduce = nwarps / WARP_आकार;

    स्थिर पूर्णांक tid = item_ct1.get_local_id(2);

    स्थिर int64_t i03 = item_ct1.get_group(0);
    स्थिर int64_t i02 = item_ct1.get_group(1);
    स्थिर int64_t i01 = item_ct1.get_group(2);

    //TODO: noncontigous inputs/outputs
    स्थिर पूर्णांक rowx = item_ct1.get_group(2) +
                     item_ct1.get_group(1) * item_ct1.get_group_range(2) +
                     item_ct1.get_group(0) * item_ct1.get_group_range(2) *
                         item_ct1.get_group_range(1);

    स्थिर int64_t i11 = i01;
    स्थिर int64_t i12 = i02 % p.ne12;
    स्थिर int64_t i13 = i03 % p.ne13;

    x    += int64_t(rowx)*ncols;
    mask += (i11*p.nb11 + i12*p.nb12 + i13*p.nb13) / माप(T) * (mask != nullptr);
    dst  += int64_t(rowx)*ncols;

    स्थिर पूर्णांक warp_id = item_ct1.get_local_id(2) / WARP_आकार;
    स्थिर पूर्णांक lane_id = item_ct1.get_local_id(2) % WARP_आकार;

    स्थिर भग्न slope = get_alibi_slope(p.max_bias, i02, p.n_शीर्ष_log2, p.m0, p.m1);

    भग्न * buf_iw = (भग्न *) dpct_local;

    // shared memory buffer to cache values between iterations:
    भग्न *vals = use_shared ? buf_iw + sycl::max(nwarps, WARP_आकार) : dst;
    भग्न max_val = sinks ? sinks[i02] : -अनन्त;
#आशय unroll
    क्रम (पूर्णांक col0 = 0; col0 < ncols; col0 += block_आकार_2) {
        स्थिर पूर्णांक col = col0 + tid;

        अगर (ncols_template == 0 && col >= ncols) {
            अवरोध;
        }

        स्थिर भग्न val = x[col]*p.scale + (mask ? slope*t2f32(mask[col]) : 0.0f);

        vals[col] = val;
        max_val   = sycl::max(max_val, val);
    }
    // find the max value in the block
    max_val = warp_reduce_max<WARP_आकार>(max_val);

    अगर (block_आकार_2 > WARP_आकार) {
        अगर (lane_id == 0) {
            buf_iw[warp_id] = max_val;
        }
        item_ct1.barrier();

        max_val = -अनन्त;
        क्रम (पूर्णांक i = lane_id; i < nwarps; i += WARP_आकार) {
            max_val = sycl::max(max_val, buf_iw[i]);
        }
        max_val = warp_reduce_max<WARP_आकार>(max_val);
    }
    भग्न tmp = 0.0f; // partial sum

#आशय unroll
    क्रम (पूर्णांक col0 = 0; col0 < ncols; col0 += block_आकार_2) {
        स्थिर पूर्णांक col = col0 + tid;

        अगर (ncols_template == 0 && col >= ncols) {
            अवरोध;
        }

        स्थिर भग्न val = sycl::native::exp(sycl::max(vals[col] - max_val, -80.0f));
        tmp += val;
        vals[col] = val;
    }
    // find the sum of exps in the block
    tmp = warp_reduce_योग<WARP_आकार>(tmp);
    अगर (block_आकार_2 > WARP_आकार) {
        item_ct1.barrier();
        अगर (warp_id == 0) {
            buf_iw[lane_id] = 0.0f;
            क्रम (माप_प्रकार i = 1; i < nreduce; i += 1) {
                buf_iw[lane_id + i * WARP_आकार] = 0.f;
            }
        }
        item_ct1.barrier();

        अगर (lane_id == 0) {
            buf_iw[warp_id] = tmp;
        }
        item_ct1.barrier();

        tmp = buf_iw[lane_id];
        क्रम (माप_प्रकार i = 1; i < nreduce; i += 1) {
            tmp += buf_iw[lane_id + i * WARP_आकार];
        }
        tmp = warp_reduce_योग<WARP_आकार>(tmp);
    }
    अगर (sinks) {
        tmp += sycl::native::exp(sycl::max(sinks[i02] - max_val, -80.0f));
    }
    स्थिर भग्न inv_योग = 1.0f / tmp;

#आशय unroll
    क्रम (पूर्णांक col0 = 0; col0 < ncols; col0 += block_आकार_2) {
        स्थिर पूर्णांक col = col0 + tid;

        अगर (ncols_template == 0 && col >= ncols) {
            वापस;
        }

        dst[col] = vals[col] * inv_योग;
    }
}
#अगर_घोषित __clang__
#आशय clang diagnostic निकालो
#पूर्ण_अगर // __clang__

अटल व्योम soft_max_back_f32(स्थिर भग्न *grad, स्थिर भग्न *dstf, भग्न *dst,
                              स्थिर पूर्णांक ncols, स्थिर भग्न scale) {
    स्वतः      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    स्थिर पूर्णांक tid      = item_ct1.get_local_id(2);
    स्थिर पूर्णांक rowx     = item_ct1.get_group(2);

    grad += int64_t(rowx)*ncols;
    dstf += int64_t(rowx)*ncols;
    dst  += int64_t(rowx)*ncols;

    भग्न dgf_dot = 0.0f; // dot product of dst from forward pass and gradients

    क्रम (पूर्णांक col = tid; col < ncols; col += WARP_आकार) {
        dgf_dot += dstf[col]*grad[col];
    }

    dgf_dot = warp_reduce_योग<WARP_आकार>(dgf_dot);

    क्रम (पूर्णांक col = tid; col < ncols; col += WARP_आकार) {
        dst[col] = scale * (grad[col] - dgf_dot) * dstf[col];
    }
}

ढाँचा <पूर्णांक... Ns, typename T>
अटल व्योम launch_soft_max_kernels(स्थिर भग्न *           x,
                                    स्थिर T *               mask,
                                    स्थिर भग्न *           sinks,
                                    भग्न *                 dst,
                                    स्थिर soft_max_params & p,
                                    dpct::queue_ptr         stream,
                                    dpct::dim3              block_dims,
                                    dpct::dim3              block_nums,
                                    माप_प्रकार                  nbytes_shared)
{
    स्वतः launch_kernel = [=](स्वतः I) -> bool {
        constexpr पूर्णांक ncols = decltype(I)::value;
        constexpr पूर्णांक block = (ncols > 1024 ? 1024 : ncols);
        अगर (p.ncols == ncols) {
            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                    sycl::range<1>(nbytes_shared), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_आकार(
                        WARP_आकार)]] {
                        soft_max_f32<true, ncols, block>(
                            x, mask, sinks, dst, p,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .लो());
                        GGML_UNUSED(item_ct1);
                    });
            });
            वापस true;
        }
        वापस false;
    };

    // unary fold over launch_kernel
    अगर ((launch_kernel(std::integral_constant<पूर्णांक, Ns>{}) || ...)) {
        वापस;
    }

    stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range<1>(nbytes_shared), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_आकार(WARP_आकार)]] {
                    soft_max_f32<true, 0, 0>(
                        x, mask, sinks, dst, p,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .लो());
                    GGML_UNUSED(item_ct1);
                });
    });
}

ढाँचा <typename T>
अटल व्योम soft_max_f32_sycl(स्थिर भग्न *x, स्थिर T *mask,
                              स्थिर भग्न *sinks, भग्न *dst,
                              स्थिर soft_max_params &params,
                              dpct::queue_ptr stream, पूर्णांक device) {
    पूर्णांक nth = WARP_आकार;
    पूर्णांक max_block_आकार = ggml_sycl_info().max_work_group_sizes[device];
    स्थिर int64_t ncols_x = params.ncols;

    जबतक (nth < ncols_x && nth < max_block_आकार) nth *= 2;
    अगर (nth>max_block_आकार) nth = max_block_आकार;

    स्थिर dpct::dim3 block_dims(nth, 1, 1);
    स्थिर dpct::dim3 block_nums(params.ne01, params.ne02, params.ne03);
    स्थिर माप_प्रकार nbytes_shared =
        (GGML_PAD(ncols_x, WARP_आकार) + WARP_आकार) * माप(भग्न);

    स्थिर पूर्णांक id       = get_current_device_id();
    स्थिर माप_प्रकार smpbo = ggml_sycl_info().devices[id].smpbo;

    अगर (nbytes_shared <= smpbo && ncols_x <= max_block_आकार) {
        launch_soft_max_kernels<32, 64, 128, 256, 512, 1024, 2048, 4096>(
            x, mask, sinks, dst, params, stream, block_dims, block_nums,
            nbytes_shared);
    } अन्यथा {
        स्थिर माप_प्रकार nbytes_shared_निचला = WARP_आकार * माप(भग्न);

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(nbytes_shared_निचला), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_आकार(WARP_आकार)]] {
                    soft_max_f32<false, 0, 0>(
                        x, mask, sinks, dst, params,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .लो());
                    GGML_UNUSED(item_ct1);
                });
        });
    }
}

अटल व्योम soft_max_back_f32_sycl(स्थिर भग्न *   grad,
                                   स्थिर भग्न *   dstf,
                                   भग्न *         dst,
                                   स्थिर पूर्णांक       ncols,
                                   स्थिर पूर्णांक       nrows,
                                   स्थिर भग्न     scale,
                                   dpct::queue_ptr stream) {
    स्थिर dpct::dim3 block_dims(WARP_आकार, 1, 1);
    स्थिर dpct::dim3 block_nums(nrows, 1, 1);

    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) {
                             soft_max_back_f32(grad, dstf, dst, ncols, scale);
                             GGML_UNUSED(item_ct1);
                         });
}

व्योम ggml_sycl_op_soft_max(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];
    स्थिर ggml_tensor * src2 = dst->src[2];

    स्थिर भग्न * src0_d = (स्थिर भग्न *) src0->तथ्य;
    स्थिर व्योम  * src1_d = src1 ? (स्थिर व्योम *) src1->तथ्य : nullptr;
    स्थिर व्योम  * src2_d = src2 ? (स्थिर व्योम *) src2->तथ्य : nullptr;
    भग्न       *  dst_d = (भग्न *) dst->तथ्य;

    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    // src1 contains mask and it is optional
    GGML_ASSERT(!src1 || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32);

    स्थिर int64_t nrows_x = ggml_nrows(src0);
    स्थिर int64_t nrows_y = src0->ne[1];

    स्थिर int64_t ne00 = src0->ne[0];

    भग्न scale    = 1.0f;
    भग्न max_bias = 0.0f;

    स_नकल(&scale,    (स्थिर भग्न *) dst->op_params + 0, माप(भग्न));
    स_नकल(&max_bias, (स्थिर भग्न *) dst->op_params + 1, माप(भग्न));

    स्थिर bool use_f16 = (src1 && src1->type == GGML_TYPE_F16);

    स्थिर int64_t nb11 = src1 ? src1->nb[1] : 1;
    स्थिर int64_t nb12 = src1 ? src1->nb[2] : 1;
    स्थिर int64_t nb13 = src1 ? src1->nb[3] : 1;

    स्थिर int64_t ne12 = src1 ? src1->ne[2] : 1;
    स्थिर int64_t ne13 = src1 ? src1->ne[3] : 1;

    स्थिर uint32_t n_शीर्ष      = src0->ne[2];
    स्थिर uint32_t n_शीर्ष_log2 = 1u << (uint32_t) floorf(log2f((भग्न) n_शीर्ष));

    स्थिर भग्न m0 = powf(2.0f, -(max_bias       ) / n_शीर्ष_log2);
    स्थिर भग्न m1 = powf(2.0f, -(max_bias / 2.0f) / n_शीर्ष_log2);


    soft_max_params params = {};
    params.nheads = src0->ne[2];
    params.n_शीर्ष_log2 = n_शीर्ष_log2;
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

    अगर (use_f16) {
        soft_max_f32_sycl(src0_d, (स्थिर sycl::half *)src1_d,
                          (स्थिर भग्न *)src2_d, dst_d, params, stream,
                          ctx.device);
    } अन्यथा {
        soft_max_f32_sycl(src0_d, (स्थिर भग्न *)src1_d, (स्थिर भग्न *)src2_d,
                          dst_d, params, stream, ctx.device);
    }
}

व्योम ggml_sycl_op_soft_max_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    स्थिर ggml_tensor * src0 = dst->src[0]; // grad
    स्थिर ggml_tensor * src1 = dst->src[1]; // forward pass output

    स्थिर भग्न * src0_d = (स्थिर भग्न *) src0->तथ्य;
    स्थिर भग्न * src1_d = (स्थिर भग्न *) src1->तथ्य;
    भग्न       * dst_d  = (भग्न       *) dst->तथ्य;

    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    स्थिर int64_t ncols = src0->ne[0];
    स्थिर int64_t nrows = ggml_nrows(src0);

    भग्न scale    = 1.0f;
    भग्न max_bias = 0.0f;

    स_नकल(&scale,    (स्थिर भग्न *) dst->op_params + 0, माप(भग्न));
    स_नकल(&max_bias, (स्थिर भग्न *) dst->op_params + 1, माप(भग्न));

    GGML_ASSERT(max_bias == 0.0f);

    soft_max_back_f32_sycl(src0_d, src1_d, dst_d, ncols, nrows, scale, stream);
}
