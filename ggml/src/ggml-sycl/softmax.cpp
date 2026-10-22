#अन्तर्भाव "softmax.hpp"
#अन्तर्भाव <cstdint>
#अन्तर्भाव <utility>
#अन्तर्भाव <cmath>


ढाँचा <typename T> स्थिर __dpct_inline__ दशांश t2f32(T val) {
    परत (दशांश) val;
}

ढाँचा <> दशांश __dpct_inline__ t2f32<sycl::half>(sycl::half val) {
  परत sycl::vec<sycl::half, 1>(val)
      .convert<दशांश, sycl::rounding_mode::automatic>()[0];
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
    दशांश scale;
    दशांश max_bias;
    दशांश m0;
    दशांश m1;
};

// When ncols_template == 0 the bounds for the loops in this function are not known and can't be unrolled.
// As we want to keep pragma unroll for all other cases we suppress the clang transformation warning here.
#ifdef __clang__
#pragma clang diagnostic ढकला
#pragma clang diagnostic ignored "-Wpass-failed"
#endif // __clang__
ढाँचा <तार्किक use_shared, पूर्णांक ncols_template, पूर्णांक block_size_template, typename T>
स्थिर शून्य soft_max_f32(स्थिरांक दशांश *         x,
                         स्थिरांक T *             mask,
                         स्थिरांक दशांश *         sinks,
                         दशांश *               dst,
                         स्थिरांक soft_max_params p,
                         uint8_t *             dpct_local) {
    स्वतः      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    स्थिरांक पूर्णांक ncols    = ncols_template == 0 ? p.ncols : ncols_template;
    स्थिरांक पूर्णांक block_size = block_size_template == 0
                               ? item_ct1.get_local_range(2)
                               : block_size_template;
    स्थिरांक पूर्णांक nthreads = block_size;
    स्थिरांक पूर्णांक nwarps = nthreads / WARP_SIZE;
    स्थिरांक माप_प्रकार nreduce = nwarps / WARP_SIZE;

    स्थिरांक पूर्णांक tid = item_ct1.get_local_id(2);

    स्थिरांक int64_t i03 = item_ct1.get_group(0);
    स्थिरांक int64_t i02 = item_ct1.get_group(1);
    स्थिरांक int64_t i01 = item_ct1.get_group(2);

    //TODO: noncontigous inputs/outputs
    स्थिरांक पूर्णांक rowx = item_ct1.get_group(2) +
                     item_ct1.get_group(1) * item_ct1.get_group_range(2) +
                     item_ct1.get_group(0) * item_ct1.get_group_range(2) *
                         item_ct1.get_group_range(1);

    स्थिरांक int64_t i11 = i01;
    स्थिरांक int64_t i12 = i02 % p.ne12;
    स्थिरांक int64_t i13 = i03 % p.ne13;

    x    += int64_t(rowx)*ncols;
    mask += (i11*p.nb11 + i12*p.nb12 + i13*p.nb13) / आकार(T) * (mask != nullptr);
    dst  += int64_t(rowx)*ncols;

    स्थिरांक पूर्णांक warp_id = item_ct1.get_local_id(2) / WARP_SIZE;
    स्थिरांक पूर्णांक lane_id = item_ct1.get_local_id(2) % WARP_SIZE;

    स्थिरांक दशांश slope = get_alibi_slope(p.max_bias, i02, p.n_head_log2, p.m0, p.m1);

    दशांश * buf_iw = (दशांश *) dpct_local;

    // shared memory buffer to cache values between iterations:
    दशांश *vals = use_shared ? buf_iw + sycl::max(nwarps, WARP_SIZE) : dst;
    दशांश max_val = sinks ? sinks[i02] : -अनन्त;
#pragma unroll
    साठी (पूर्णांक col0 = 0; col0 < ncols; col0 += block_size) {
        स्थिरांक पूर्णांक col = col0 + tid;

        जर (ncols_template == 0 && col >= ncols) {
            थांबा;
        }

        स्थिरांक दशांश val = x[col]*p.scale + (mask ? slope*t2f32(mask[col]) : 0.0f);

        vals[col] = val;
        max_val   = sycl::max(max_val, val);
    }
    // find the max value in the block
    max_val = warp_reduce_max<WARP_SIZE>(max_val);

    जर (block_size > WARP_SIZE) {
        जर (lane_id == 0) {
            buf_iw[warp_id] = max_val;
        }
        item_ct1.barrier();

        max_val = -अनन्त;
        साठी (पूर्णांक i = lane_id; i < nwarps; i += WARP_SIZE) {
            max_val = sycl::max(max_val, buf_iw[i]);
        }
        max_val = warp_reduce_max<WARP_SIZE>(max_val);
    }
    दशांश tmp = 0.0f; // partial sum

#pragma unroll
    साठी (पूर्णांक col0 = 0; col0 < ncols; col0 += block_size) {
        स्थिरांक पूर्णांक col = col0 + tid;

        जर (ncols_template == 0 && col >= ncols) {
            थांबा;
        }

        स्थिरांक दशांश val = sycl::native::exp(sycl::max(vals[col] - max_val, -80.0f));
        tmp += val;
        vals[col] = val;
    }
    // find the sum of exps in the block
    tmp = warp_reduce_sum<WARP_SIZE>(tmp);
    जर (block_size > WARP_SIZE) {
        item_ct1.barrier();
        जर (warp_id == 0) {
            buf_iw[lane_id] = 0.0f;
            साठी (माप_प्रकार i = 1; i < nreduce; i += 1) {
                buf_iw[lane_id + i * WARP_SIZE] = 0.f;
            }
        }
        item_ct1.barrier();

        जर (lane_id == 0) {
            buf_iw[warp_id] = tmp;
        }
        item_ct1.barrier();

        tmp = buf_iw[lane_id];
        साठी (माप_प्रकार i = 1; i < nreduce; i += 1) {
            tmp += buf_iw[lane_id + i * WARP_SIZE];
        }
        tmp = warp_reduce_sum<WARP_SIZE>(tmp);
    }
    जर (sinks) {
        tmp += sycl::native::exp(sycl::max(sinks[i02] - max_val, -80.0f));
    }
    स्थिरांक दशांश inv_sum = 1.0f / tmp;

#pragma unroll
    साठी (पूर्णांक col0 = 0; col0 < ncols; col0 += block_size) {
        स्थिरांक पूर्णांक col = col0 + tid;

        जर (ncols_template == 0 && col >= ncols) {
            परत;
        }

        dst[col] = vals[col] * inv_sum;
    }
}
#ifdef __clang__
#pragma clang diagnostic काढा
#endif // __clang__

स्थिर शून्य soft_max_back_f32(स्थिरांक दशांश *grad, स्थिरांक दशांश *dstf, दशांश *dst,
                              स्थिरांक पूर्णांक ncols, स्थिरांक दशांश scale) {
    स्वतः      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    स्थिरांक पूर्णांक tid      = item_ct1.get_local_id(2);
    स्थिरांक पूर्णांक rowx     = item_ct1.get_group(2);

    grad += int64_t(rowx)*ncols;
    dstf += int64_t(rowx)*ncols;
    dst  += int64_t(rowx)*ncols;

    दशांश dgf_dot = 0.0f; // dot product of dst from forward pass and gradients

    साठी (पूर्णांक col = tid; col < ncols; col += WARP_SIZE) {
        dgf_dot += dstf[col]*grad[col];
    }

    dgf_dot = warp_reduce_sum<WARP_SIZE>(dgf_dot);

    साठी (पूर्णांक col = tid; col < ncols; col += WARP_SIZE) {
        dst[col] = scale * (grad[col] - dgf_dot) * dstf[col];
    }
}

ढाँचा <पूर्णांक... Ns, typename T>
स्थिर शून्य launch_soft_max_kernels(स्थिरांक दशांश *           x,
                                    स्थिरांक T *               mask,
                                    स्थिरांक दशांश *           sinks,
                                    दशांश *                 dst,
                                    स्थिरांक soft_max_params & p,
                                    dpct::queue_ptr         stream,
                                    dpct::dim3              block_dims,
                                    dpct::dim3              block_nums,
                                    माप_प्रकार                  nbytes_shared)
{
    स्वतः launch_kernel = [=](स्वतः I) -> तार्किक {
        constexpr पूर्णांक ncols = decltype(I)::value;
        constexpr पूर्णांक block = (ncols > 1024 ? 1024 : ncols);
        जर (p.ncols == ncols) {
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
            परत सत्य;
        }
        परत असत्य;
    };

    // unary fold over launch_kernel
    जर ((launch_kernel(std::integral_constant<पूर्णांक, Ns>{}) || ...)) {
        परत;
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
स्थिर शून्य soft_max_f32_sycl(स्थिरांक दशांश *x, स्थिरांक T *mask,
                              स्थिरांक दशांश *sinks, दशांश *dst,
                              स्थिरांक soft_max_params &params,
                              dpct::queue_ptr stream, पूर्णांक device) {
    पूर्णांक nth = WARP_SIZE;
    पूर्णांक max_block_size = ggml_sycl_info().max_work_group_sizes[device];
    स्थिरांक int64_t ncols_x = params.ncols;

    पर्यंत (nth < ncols_x && nth < max_block_size) nth *= 2;
    जर (nth>max_block_size) nth = max_block_size;

    स्थिरांक dpct::dim3 block_dims(nth, 1, 1);
    स्थिरांक dpct::dim3 block_nums(params.ne01, params.ne02, params.ne03);
    स्थिरांक माप_प्रकार nbytes_shared =
        (GGML_PAD(ncols_x, WARP_SIZE) + WARP_SIZE) * आकार(दशांश);

    स्थिरांक पूर्णांक id       = get_current_device_id();
    स्थिरांक माप_प्रकार smpbo = ggml_sycl_info().devices[id].smpbo;

    जर (nbytes_shared <= smpbo && ncols_x <= max_block_size) {
        launch_soft_max_kernels<32, 64, 128, 256, 512, 1024, 2048, 4096>(
            x, mask, sinks, dst, params, stream, block_dims, block_nums,
            nbytes_shared);
    } अन्यथा {
        स्थिरांक माप_प्रकार nbytes_shared_खालचा = WARP_SIZE * आकार(दशांश);

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(nbytes_shared_खालचा), cgh);

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

स्थिर शून्य soft_max_back_f32_sycl(स्थिरांक दशांश *   grad,
                                   स्थिरांक दशांश *   dstf,
                                   दशांश *         dst,
                                   स्थिरांक पूर्णांक       ncols,
                                   स्थिरांक पूर्णांक       nrows,
                                   स्थिरांक दशांश     scale,
                                   dpct::queue_ptr stream) {
    स्थिरांक dpct::dim3 block_dims(WARP_SIZE, 1, 1);
    स्थिरांक dpct::dim3 block_nums(nrows, 1, 1);

    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) {
                             soft_max_back_f32(grad, dstf, dst, ncols, scale);
                             GGML_UNUSED(item_ct1);
                         });
}

शून्य ggml_sycl_op_soft_max(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    स्थिरांक ggml_tensor * src0 = dst->src[0];
    स्थिरांक ggml_tensor * src1 = dst->src[1];
    स्थिरांक ggml_tensor * src2 = dst->src[2];

    स्थिरांक दशांश * src0_d = (स्थिरांक दशांश *) src0->data;
    स्थिरांक शून्य  * src1_d = src1 ? (स्थिरांक शून्य *) src1->data : nullptr;
    स्थिरांक शून्य  * src2_d = src2 ? (स्थिरांक शून्य *) src2->data : nullptr;
    दशांश       *  dst_d = (दशांश *) dst->data;

    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    // src1 contains mask and it is optional
    GGML_ASSERT(!src1 || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32);

    स्थिरांक int64_t nrows_x = ggml_nrows(src0);
    स्थिरांक int64_t nrows_y = src0->ne[1];

    स्थिरांक int64_t ne00 = src0->ne[0];

    दशांश scale    = 1.0f;
    दशांश max_bias = 0.0f;

    स_नकल(&scale,    (स्थिरांक दशांश *) dst->op_params + 0, आकार(दशांश));
    स_नकल(&max_bias, (स्थिरांक दशांश *) dst->op_params + 1, आकार(दशांश));

    स्थिरांक तार्किक use_f16 = (src1 && src1->type == GGML_TYPE_F16);

    स्थिरांक int64_t nb11 = src1 ? src1->nb[1] : 1;
    स्थिरांक int64_t nb12 = src1 ? src1->nb[2] : 1;
    स्थिरांक int64_t nb13 = src1 ? src1->nb[3] : 1;

    स्थिरांक int64_t ne12 = src1 ? src1->ne[2] : 1;
    स्थिरांक int64_t ne13 = src1 ? src1->ne[3] : 1;

    स्थिरांक uint32_t n_head      = src0->ne[2];
    स्थिरांक uint32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((दशांश) n_head));

    स्थिरांक दशांश m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    स्थिरांक दशांश m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);


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

    जर (use_f16) {
        soft_max_f32_sycl(src0_d, (स्थिरांक sycl::half *)src1_d,
                          (स्थिरांक दशांश *)src2_d, dst_d, params, stream,
                          ctx.device);
    } अन्यथा {
        soft_max_f32_sycl(src0_d, (स्थिरांक दशांश *)src1_d, (स्थिरांक दशांश *)src2_d,
                          dst_d, params, stream, ctx.device);
    }
}

शून्य ggml_sycl_op_soft_max_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    स्थिरांक ggml_tensor * src0 = dst->src[0]; // grad
    स्थिरांक ggml_tensor * src1 = dst->src[1]; // forward pass output

    स्थिरांक दशांश * src0_d = (स्थिरांक दशांश *) src0->data;
    स्थिरांक दशांश * src1_d = (स्थिरांक दशांश *) src1->data;
    दशांश       * dst_d  = (दशांश       *) dst->data;

    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    स्थिरांक int64_t ncols = src0->ne[0];
    स्थिरांक int64_t nrows = ggml_nrows(src0);

    दशांश scale    = 1.0f;
    दशांश max_bias = 0.0f;

    स_नकल(&scale,    (स्थिरांक दशांश *) dst->op_params + 0, आकार(दशांश));
    स_नकल(&max_bias, (स्थिरांक दशांश *) dst->op_params + 1, आकार(दशांश));

    GGML_ASSERT(max_bias == 0.0f);

    soft_max_back_f32_sycl(src0_d, src1_d, dst_d, ncols, nrows, scale, stream);
}
