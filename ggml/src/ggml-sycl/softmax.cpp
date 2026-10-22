#ضمن "softmax.hpp"
#ضمن <cstdint>
#ضمن <utility>
#ضمن <cmath>


سانچو <قسم_نالو T> ساكن __dpct_inline__ عائم t2f32(T val) {
    أرجع (عائم) val;
}

سانچو <> عائم __dpct_inline__ t2f32<sycl::half>(sycl::half val) {
  أرجع sycl::vec<sycl::half, 1>(val)
      .convert<عائم, sycl::rounding_گهڻائي::automatic>()[0];
}

بنية soft_max_params {

    int64_t nheads;
    uint32_t n_سر_log2;
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
    عائم scale;
    عائم max_bias;
    عائم m0;
    عائم m1;
};

// When ncols_template == 0 the bounds for the loops in this function are not known and can't be unrolled.
// As we want to keep pragma unroll for all other cases we suppress the clang transformation warning here.
#جيڪڏهن_وصف_ٿيل __clang__
#هدايت clang diagnostic ڌڪو
#هدايت clang diagnostic ignored "-Wpass-failed"
#جيڪڏهن_پورو // __clang__
سانچو <منطقي use_shared, صحيح ncols_template, صحيح بلاڪ_سائيز_template, قسم_نالو T>
ساكن فراغ soft_max_f32(ثابت عائم *         x,
                         ثابت T *             نقاب,
                         ثابت عائم *         sinks,
                         عائم *               dst,
                         ثابت soft_max_params p,
                         uint8_t *             dpct_local) {
    خودڪار      شي_ct1 = sycl::ext::oneapi::this_work_شي::get_nd_شي<3>();
    ثابت صحيح ncols    = ncols_template == 0 ? p.ncols : ncols_template;
    ثابت صحيح بلاڪ_سائيز_3 = بلاڪ_سائيز_template == 0
                               ? شي_ct1.get_local_range(2)
                               : بلاڪ_سائيز_template;
    ثابت صحيح nthreads = بلاڪ_سائيز_3;
    ثابت صحيح nwarps = nthreads / WARP_سائيز;
    ثابت size_t nreduce = nwarps / WARP_سائيز;

    ثابت صحيح tid = شي_ct1.get_local_id(2);

    ثابت int64_t i03 = شي_ct1.get_گروهه(0);
    ثابت int64_t i02 = شي_ct1.get_گروهه(1);
    ثابت int64_t i01 = شي_ct1.get_گروهه(2);

    //TODO: noncontigous inputs/outputs
    ثابت صحيح rowx = شي_ct1.get_گروهه(2) +
                     شي_ct1.get_گروهه(1) * شي_ct1.get_گروهه_range(2) +
                     شي_ct1.get_گروهه(0) * شي_ct1.get_گروهه_range(2) *
                         شي_ct1.get_گروهه_range(1);

    ثابت int64_t i11 = i01;
    ثابت int64_t i12 = i02 % p.ne12;
    ثابت int64_t i13 = i03 % p.ne13;

    x    += int64_t(rowx)*ncols;
    نقاب += (i11*p.nb11 + i12*p.nb12 + i13*p.nb13) / حجم(T) * (نقاب != خالي_اشارو);
    dst  += int64_t(rowx)*ncols;

    ثابت صحيح warp_id = شي_ct1.get_local_id(2) / WARP_سائيز;
    ثابت صحيح lane_id = شي_ct1.get_local_id(2) % WARP_سائيز;

    ثابت عائم لاهي = get_alibi_لاهي(p.max_bias, i02, p.n_سر_log2, p.m0, p.m1);

    عائم * buf_iw = (عائم *) dpct_local;

    // shared memory buffer to cache values between iterations:
    عائم *vals = use_shared ? buf_iw + sycl::max(nwarps, WARP_سائيز) : dst;
    عائم max_val = sinks ? sinks[i02] : -INFINITY;
#هدايت unroll
    لكل (صحيح col0 = 0; col0 < ncols; col0 += بلاڪ_سائيز_3) {
        ثابت صحيح کانو = col0 + tid;

        إذا (ncols_template == 0 && کانو >= ncols) {
            اكسر;
        }

        ثابت عائم val = x[کانو]*p.scale + (نقاب ? لاهي*t2f32(نقاب[کانو]) : 0.0f);

        vals[کانو] = val;
        max_val   = sycl::max(max_val, val);
    }
    // find the max value in the block
    max_val = warp_reduce_max<WARP_سائيز>(max_val);

    إذا (بلاڪ_سائيز_3 > WARP_سائيز) {
        إذا (lane_id == 0) {
            buf_iw[warp_id] = max_val;
        }
        شي_ct1.barrier();

        max_val = -INFINITY;
        لكل (صحيح i = lane_id; i < nwarps; i += WARP_سائيز) {
            max_val = sycl::max(max_val, buf_iw[i]);
        }
        max_val = warp_reduce_max<WARP_سائيز>(max_val);
    }
    عائم tmp = 0.0f; // partial sum

#هدايت unroll
    لكل (صحيح col0 = 0; col0 < ncols; col0 += بلاڪ_سائيز_3) {
        ثابت صحيح کانو = col0 + tid;

        إذا (ncols_template == 0 && کانو >= ncols) {
            اكسر;
        }

        ثابت عائم val = sycl::native::قوت_نما(sycl::max(vals[کانو] - max_val, -80.0f));
        tmp += val;
        vals[کانو] = val;
    }
    // find the sum of exps in the block
    tmp = warp_reduce_جوڙ<WARP_سائيز>(tmp);
    إذا (بلاڪ_سائيز_3 > WARP_سائيز) {
        شي_ct1.barrier();
        إذا (warp_id == 0) {
            buf_iw[lane_id] = 0.0f;
            لكل (size_t i = 1; i < nreduce; i += 1) {
                buf_iw[lane_id + i * WARP_سائيز] = 0.f;
            }
        }
        شي_ct1.barrier();

        إذا (lane_id == 0) {
            buf_iw[warp_id] = tmp;
        }
        شي_ct1.barrier();

        tmp = buf_iw[lane_id];
        لكل (size_t i = 1; i < nreduce; i += 1) {
            tmp += buf_iw[lane_id + i * WARP_سائيز];
        }
        tmp = warp_reduce_جوڙ<WARP_سائيز>(tmp);
    }
    إذا (sinks) {
        tmp += sycl::native::قوت_نما(sycl::max(sinks[i02] - max_val, -80.0f));
    }
    ثابت عائم inv_جوڙ = 1.0f / tmp;

#هدايت unroll
    لكل (صحيح col0 = 0; col0 < ncols; col0 += بلاڪ_سائيز_3) {
        ثابت صحيح کانو = col0 + tid;

        إذا (ncols_template == 0 && کانو >= ncols) {
            أرجع;
        }

        dst[کانو] = vals[کانو] * inv_جوڙ;
    }
}
#جيڪڏهن_وصف_ٿيل __clang__
#هدايت clang diagnostic ڪڍو
#جيڪڏهن_پورو // __clang__

ساكن فراغ soft_max_پٺ_f32(ثابت عائم *grad, ثابت عائم *dstf, عائم *dst,
                              ثابت صحيح ncols, ثابت عائم scale) {
    خودڪار      شي_ct1 = sycl::ext::oneapi::this_work_شي::get_nd_شي<3>();
    ثابت صحيح tid      = شي_ct1.get_local_id(2);
    ثابت صحيح rowx     = شي_ct1.get_گروهه(2);

    grad += int64_t(rowx)*ncols;
    dstf += int64_t(rowx)*ncols;
    dst  += int64_t(rowx)*ncols;

    عائم dgf_dot = 0.0f; // dot product of dst from forward pass and gradients

    لكل (صحيح کانو = tid; کانو < ncols; کانو += WARP_سائيز) {
        dgf_dot += dstf[کانو]*grad[کانو];
    }

    dgf_dot = warp_reduce_جوڙ<WARP_سائيز>(dgf_dot);

    لكل (صحيح کانو = tid; کانو < ncols; کانو += WARP_سائيز) {
        dst[کانو] = scale * (grad[کانو] - dgf_dot) * dstf[کانو];
    }
}

سانچو <صحيح... Ns, قسم_نالو T>
ساكن فراغ launch_soft_max_kernels(ثابت عائم *           x,
                                    ثابت T *               نقاب,
                                    ثابت عائم *           sinks,
                                    عائم *                 dst,
                                    ثابت soft_max_params & p,
                                    dpct::قطار_ptr         stream,
                                    dpct::dim3              بلاڪ_dims,
                                    dpct::dim3              بلاڪ_nums,
                                    size_t                  nbytes_shared)
{
    خودڪار launch_kernel = [=](خودڪار I) -> منطقي {
        constexpr صحيح ncols = decltype(I)::value;
        constexpr صحيح بلاڪ = (ncols > 1024 ? 1024 : ncols);
        إذا (p.ncols == ncols) {
            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                    sycl::range<1>(nbytes_shared), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(بلاڪ_nums * بلاڪ_dims, بلاڪ_dims),
                    [=](sycl::nd_شي<3> شي_ct1) [[sycl::reqd_sub_گروهه_سائيز(
                        WARP_سائيز)]] {
                        soft_max_f32<سچ, ncols, بلاڪ>(
                            x, نقاب, sinks, dst, p,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                        GGML_اڻ_استعمال_ٿيل(شي_ct1);
                    });
            });
            أرجع سچ;
        }
        أرجع خطأ;
    };

    // unary fold over launch_kernel
    إذا ((launch_kernel(std::integral_constant<صحيح, Ns>{}) || ...)) {
        أرجع;
    }

    stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range<1>(nbytes_shared), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(بلاڪ_nums * بلاڪ_dims, بلاڪ_dims),
            [=](sycl::nd_شي<3> شي_ct1)
                [[sycl::reqd_sub_گروهه_سائيز(WARP_سائيز)]] {
                    soft_max_f32<سچ, 0, 0>(
                        x, نقاب, sinks, dst, p,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                    GGML_اڻ_استعمال_ٿيل(شي_ct1);
                });
    });
}

سانچو <قسم_نالو T>
ساكن فراغ soft_max_f32_sycl(ثابت عائم *x, ثابت T *نقاب,
                              ثابت عائم *sinks, عائم *dst,
                              ثابت soft_max_params &params,
                              dpct::قطار_ptr stream, صحيح device) {
    صحيح nth = WARP_سائيز;
    صحيح max_بلاڪ_سائيز = ggml_sycl_info().max_work_گروهه_sizes[device];
    ثابت int64_t ncols_x = params.ncols;

    طالما (nth < ncols_x && nth < max_بلاڪ_سائيز) nth *= 2;
    إذا (nth>max_بلاڪ_سائيز) nth = max_بلاڪ_سائيز;

    ثابت dpct::dim3 بلاڪ_dims(nth, 1, 1);
    ثابت dpct::dim3 بلاڪ_nums(params.ne01, params.ne02, params.ne03);
    ثابت size_t nbytes_shared =
        (GGML_PAD(ncols_x, WARP_سائيز) + WARP_سائيز) * حجم(عائم);

    ثابت صحيح id       = get_current_device_id();
    ثابت size_t smpbo = ggml_sycl_info().devices[id].smpbo;

    إذا (nbytes_shared <= smpbo && ncols_x <= max_بلاڪ_سائيز) {
        launch_soft_max_kernels<32, 64, 128, 256, 512, 1024, 2048, 4096>(
            x, نقاب, sinks, dst, params, stream, بلاڪ_dims, بلاڪ_nums,
            nbytes_shared);
    } وإلا {
        ثابت size_t nbytes_shared_هيٺيون = WARP_سائيز * حجم(عائم);

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(nbytes_shared_هيٺيون), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(بلاڪ_nums * بلاڪ_dims, بلاڪ_dims),
                [=](sycl::nd_شي<3> شي_ct1)
                    [[sycl::reqd_sub_گروهه_سائيز(WARP_سائيز)]] {
                    soft_max_f32<خطأ, 0, 0>(
                        x, نقاب, sinks, dst, params,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                    GGML_اڻ_استعمال_ٿيل(شي_ct1);
                });
        });
    }
}

ساكن فراغ soft_max_پٺ_f32_sycl(ثابت عائم *   grad,
                                   ثابت عائم *   dstf,
                                   عائم *         dst,
                                   ثابت صحيح       ncols,
                                   ثابت صحيح       nrows,
                                   ثابت عائم     scale,
                                   dpct::قطار_ptr stream) {
    ثابت dpct::dim3 بلاڪ_dims(WARP_سائيز, 1, 1);
    ثابت dpct::dim3 بلاڪ_nums(nrows, 1, 1);

    stream->parallel_for(sycl::nd_range<3>(بلاڪ_nums * بلاڪ_dims, بلاڪ_dims),
                         [=](sycl::nd_شي<3> شي_ct1) {
                             soft_max_پٺ_f32(grad, dstf, dst, ncols, scale);
                             GGML_اڻ_استعمال_ٿيل(شي_ct1);
                         });
}

فراغ ggml_sycl_op_soft_max(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];
    ثابت ggml_tensor * src2 = dst->src[2];

    ثابت عائم * src0_d = (ثابت عائم *) src0->معلومات;
    ثابت فراغ  * src1_d = src1 ? (ثابت فراغ *) src1->معلومات : خالي_اشارو;
    ثابت فراغ  * src2_d = src2 ? (ثابت فراغ *) src2->معلومات : خالي_اشارو;
    عائم       *  dst_d = (عائم *) dst->معلومات;

    dpct::قطار_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    // src1 contains mask and it is optional
    GGML_ASSERT(!src1 || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32);

    ثابت int64_t nrows_x = ggml_nrows(src0);
    ثابت int64_t nrows_y = src0->ne[1];

    ثابت int64_t ne00 = src0->ne[0];

    عائم scale    = 1.0f;
    عائم max_bias = 0.0f;

    يادگيري_نقل(&scale,    (ثابت عائم *) dst->op_params + 0, حجم(عائم));
    يادگيري_نقل(&max_bias, (ثابت عائم *) dst->op_params + 1, حجم(عائم));

    ثابت منطقي use_f16 = (src1 && src1->type == GGML_TYPE_F16);

    ثابت int64_t nb11 = src1 ? src1->nb[1] : 1;
    ثابت int64_t nb12 = src1 ? src1->nb[2] : 1;
    ثابت int64_t nb13 = src1 ? src1->nb[3] : 1;

    ثابت int64_t ne12 = src1 ? src1->ne[2] : 1;
    ثابت int64_t ne13 = src1 ? src1->ne[3] : 1;

    ثابت uint32_t n_سر      = src0->ne[2];
    ثابت uint32_t n_سر_log2 = 1u << (uint32_t) floorf(log2f((عائم) n_سر));

    ثابت عائم m0 = powf(2.0f, -(max_bias       ) / n_سر_log2);
    ثابت عائم m1 = powf(2.0f, -(max_bias / 2.0f) / n_سر_log2);


    soft_max_params params = {};
    params.nheads = src0->ne[2];
    params.n_سر_log2 = n_سر_log2;
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

    إذا (use_f16) {
        soft_max_f32_sycl(src0_d, (ثابت sycl::half *)src1_d,
                          (ثابت عائم *)src2_d, dst_d, params, stream,
                          ctx.device);
    } وإلا {
        soft_max_f32_sycl(src0_d, (ثابت عائم *)src1_d, (ثابت عائم *)src2_d,
                          dst_d, params, stream, ctx.device);
    }
}

فراغ ggml_sycl_op_soft_max_پٺ(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ثابت ggml_tensor * src0 = dst->src[0]; // grad
    ثابت ggml_tensor * src1 = dst->src[1]; // forward pass output

    ثابت عائم * src0_d = (ثابت عائم *) src0->معلومات;
    ثابت عائم * src1_d = (ثابت عائم *) src1->معلومات;
    عائم       * dst_d  = (عائم       *) dst->معلومات;

    dpct::قطار_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    ثابت int64_t ncols = src0->ne[0];
    ثابت int64_t nrows = ggml_nrows(src0);

    عائم scale    = 1.0f;
    عائم max_bias = 0.0f;

    يادگيري_نقل(&scale,    (ثابت عائم *) dst->op_params + 0, حجم(عائم));
    يادگيري_نقل(&max_bias, (ثابت عائم *) dst->op_params + 1, حجم(عائم));

    GGML_ASSERT(max_bias == 0.0f);

    soft_max_پٺ_f32_sycl(src0_d, src1_d, dst_d, ncols, nrows, scale, stream);
}
