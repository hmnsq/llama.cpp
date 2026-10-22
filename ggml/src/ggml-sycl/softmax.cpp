#شامل "softmax.hpp"
#شامل <cstdint>
#شامل <utility>
#شامل <cmath>


سانچہ <قسم_نام T> ساکن __dpct_inline__ اعشاری t2f32(T val) {
    واپس (اعشاری) val;
}

سانچہ <> اعشاری __dpct_inline__ t2f32<sycl::half>(sycl::half val) {
  واپس sycl::vec<sycl::half, 1>(val)
      .convert<اعشاری, sycl::rounding_منوال::automatic>()[0];
}

ساخت soft_max_params {

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
    اعشاری scale;
    اعشاری max_bias;
    اعشاری m0;
    اعشاری m1;
};

// When ncols_template == 0 the bounds for the loops in this function are not known and can't be unrolled.
// As we want to keep pragma unroll for all other cases we suppress the clang transformation warning here.
#اگر_تعریف_شدہ __clang__
#ہدایت clang diagnostic ڈالو
#ہدایت clang diagnostic ignored "-Wpass-failed"
#اگر_ختم // __clang__
سانچہ <منطقی use_shared, صحیح ncols_template, صحیح بلاک_سائز_template, قسم_نام T>
ساکن خالی soft_max_f32(ثابت اعشاری *         x,
                         ثابت T *             نقاب,
                         ثابت اعشاری *         sinks,
                         اعشاری *               dst,
                         ثابت soft_max_params p,
                         uint8_t *             dpct_local) {
    خودکار      شے_ct1 = sycl::ext::oneapi::this_work_شے::get_nd_شے<3>();
    ثابت صحیح ncols    = ncols_template == 0 ? p.ncols : ncols_template;
    ثابت صحیح بلاک_سائز_3 = بلاک_سائز_template == 0
                               ? شے_ct1.get_local_range(2)
                               : بلاک_سائز_template;
    ثابت صحیح nthreads = بلاک_سائز_3;
    ثابت صحیح nwarps = nthreads / WARP_سائز;
    ثابت size_t nreduce = nwarps / WARP_سائز;

    ثابت صحیح tid = شے_ct1.get_local_id(2);

    ثابت int64_t i03 = شے_ct1.get_گروہ(0);
    ثابت int64_t i02 = شے_ct1.get_گروہ(1);
    ثابت int64_t i01 = شے_ct1.get_گروہ(2);

    //TODO: noncontigous inputs/outputs
    ثابت صحیح rowx = شے_ct1.get_گروہ(2) +
                     شے_ct1.get_گروہ(1) * شے_ct1.get_گروہ_range(2) +
                     شے_ct1.get_گروہ(0) * شے_ct1.get_گروہ_range(2) *
                         شے_ct1.get_گروہ_range(1);

    ثابت int64_t i11 = i01;
    ثابت int64_t i12 = i02 % p.ne12;
    ثابت int64_t i13 = i03 % p.ne13;

    x    += int64_t(rowx)*ncols;
    نقاب += (i11*p.nb11 + i12*p.nb12 + i13*p.nb13) / حجم(T) * (نقاب != خالی_اشاریہ);
    dst  += int64_t(rowx)*ncols;

    ثابت صحیح warp_id = شے_ct1.get_local_id(2) / WARP_سائز;
    ثابت صحیح lane_id = شے_ct1.get_local_id(2) % WARP_سائز;

    ثابت اعشاری ڈھلان = get_alibi_ڈھلان(p.max_bias, i02, p.n_سر_log2, p.m0, p.m1);

    اعشاری * buf_iw = (اعشاری *) dpct_local;

    // shared memory buffer to cache values between iterations:
    اعشاری *vals = use_shared ? buf_iw + sycl::max(nwarps, WARP_سائز) : dst;
    اعشاری max_val = sinks ? sinks[i02] : -INFINITY;
#ہدایت unroll
    برائے (صحیح col0 = 0; col0 < ncols; col0 += بلاک_سائز_3) {
        ثابت صحیح ستون = col0 + tid;

        اگر (ncols_template == 0 && ستون >= ncols) {
            توڑو;
        }

        ثابت اعشاری val = x[ستون]*p.scale + (نقاب ? ڈھلان*t2f32(نقاب[ستون]) : 0.0f);

        vals[ستون] = val;
        max_val   = sycl::max(max_val, val);
    }
    // find the max value in the block
    max_val = warp_reduce_max<WARP_سائز>(max_val);

    اگر (بلاک_سائز_3 > WARP_سائز) {
        اگر (lane_id == 0) {
            buf_iw[warp_id] = max_val;
        }
        شے_ct1.barrier();

        max_val = -INFINITY;
        برائے (صحیح i = lane_id; i < nwarps; i += WARP_سائز) {
            max_val = sycl::max(max_val, buf_iw[i]);
        }
        max_val = warp_reduce_max<WARP_سائز>(max_val);
    }
    اعشاری tmp = 0.0f; // partial sum

#ہدایت unroll
    برائے (صحیح col0 = 0; col0 < ncols; col0 += بلاک_سائز_3) {
        ثابت صحیح ستون = col0 + tid;

        اگر (ncols_template == 0 && ستون >= ncols) {
            توڑو;
        }

        ثابت اعشاری val = sycl::native::قوت_نما(sycl::max(vals[ستون] - max_val, -80.0f));
        tmp += val;
        vals[ستون] = val;
    }
    // find the sum of exps in the block
    tmp = warp_reduce_مجموع<WARP_سائز>(tmp);
    اگر (بلاک_سائز_3 > WARP_سائز) {
        شے_ct1.barrier();
        اگر (warp_id == 0) {
            buf_iw[lane_id] = 0.0f;
            برائے (size_t i = 1; i < nreduce; i += 1) {
                buf_iw[lane_id + i * WARP_سائز] = 0.f;
            }
        }
        شے_ct1.barrier();

        اگر (lane_id == 0) {
            buf_iw[warp_id] = tmp;
        }
        شے_ct1.barrier();

        tmp = buf_iw[lane_id];
        برائے (size_t i = 1; i < nreduce; i += 1) {
            tmp += buf_iw[lane_id + i * WARP_سائز];
        }
        tmp = warp_reduce_مجموع<WARP_سائز>(tmp);
    }
    اگر (sinks) {
        tmp += sycl::native::قوت_نما(sycl::max(sinks[i02] - max_val, -80.0f));
    }
    ثابت اعشاری inv_مجموع = 1.0f / tmp;

#ہدایت unroll
    برائے (صحیح col0 = 0; col0 < ncols; col0 += بلاک_سائز_3) {
        ثابت صحیح ستون = col0 + tid;

        اگر (ncols_template == 0 && ستون >= ncols) {
            واپس;
        }

        dst[ستون] = vals[ستون] * inv_مجموع;
    }
}
#اگر_تعریف_شدہ __clang__
#ہدایت clang diagnostic نکالو
#اگر_ختم // __clang__

ساکن خالی soft_max_عقب_f32(ثابت اعشاری *grad, ثابت اعشاری *dstf, اعشاری *dst,
                              ثابت صحیح ncols, ثابت اعشاری scale) {
    خودکار      شے_ct1 = sycl::ext::oneapi::this_work_شے::get_nd_شے<3>();
    ثابت صحیح tid      = شے_ct1.get_local_id(2);
    ثابت صحیح rowx     = شے_ct1.get_گروہ(2);

    grad += int64_t(rowx)*ncols;
    dstf += int64_t(rowx)*ncols;
    dst  += int64_t(rowx)*ncols;

    اعشاری dgf_dot = 0.0f; // dot product of dst from forward pass and gradients

    برائے (صحیح ستون = tid; ستون < ncols; ستون += WARP_سائز) {
        dgf_dot += dstf[ستون]*grad[ستون];
    }

    dgf_dot = warp_reduce_مجموع<WARP_سائز>(dgf_dot);

    برائے (صحیح ستون = tid; ستون < ncols; ستون += WARP_سائز) {
        dst[ستون] = scale * (grad[ستون] - dgf_dot) * dstf[ستون];
    }
}

سانچہ <صحیح... Ns, قسم_نام T>
ساکن خالی launch_soft_max_kernels(ثابت اعشاری *           x,
                                    ثابت T *               نقاب,
                                    ثابت اعشاری *           sinks,
                                    اعشاری *                 dst,
                                    ثابت soft_max_params & p,
                                    dpct::قطار_ptr         stream,
                                    dpct::dim3              بلاک_dims,
                                    dpct::dim3              بلاک_nums,
                                    size_t                  nbytes_shared)
{
    خودکار launch_kernel = [=](خودکار I) -> منطقی {
        constexpr صحیح ncols = decltype(I)::value;
        constexpr صحیح بلاک = (ncols > 1024 ? 1024 : ncols);
        اگر (p.ncols == ncols) {
            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                    sycl::range<1>(nbytes_shared), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                    [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(
                        WARP_سائز)]] {
                        soft_max_f32<سچ, ncols, بلاک>(
                            x, نقاب, sinks, dst, p,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                        GGML_غیر_مستعمل(شے_ct1);
                    });
            });
            واپس سچ;
        }
        واپس جھوٹ;
    };

    // unary fold over launch_kernel
    اگر ((launch_kernel(std::integral_constant<صحیح, Ns>{}) || ...)) {
        واپس;
    }

    stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range<1>(nbytes_shared), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<3> شے_ct1)
                [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                    soft_max_f32<سچ, 0, 0>(
                        x, نقاب, sinks, dst, p,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                    GGML_غیر_مستعمل(شے_ct1);
                });
    });
}

سانچہ <قسم_نام T>
ساکن خالی soft_max_f32_sycl(ثابت اعشاری *x, ثابت T *نقاب,
                              ثابت اعشاری *sinks, اعشاری *dst,
                              ثابت soft_max_params &params,
                              dpct::قطار_ptr stream, صحیح device) {
    صحیح nth = WARP_سائز;
    صحیح max_بلاک_سائز = ggml_sycl_info().max_work_گروہ_sizes[device];
    ثابت int64_t ncols_x = params.ncols;

    جب (nth < ncols_x && nth < max_بلاک_سائز) nth *= 2;
    اگر (nth>max_بلاک_سائز) nth = max_بلاک_سائز;

    ثابت dpct::dim3 بلاک_dims(nth, 1, 1);
    ثابت dpct::dim3 بلاک_nums(params.ne01, params.ne02, params.ne03);
    ثابت size_t nbytes_shared =
        (GGML_PAD(ncols_x, WARP_سائز) + WARP_سائز) * حجم(اعشاری);

    ثابت صحیح id       = get_current_device_id();
    ثابت size_t smpbo = ggml_sycl_info().devices[id].smpbo;

    اگر (nbytes_shared <= smpbo && ncols_x <= max_بلاک_سائز) {
        launch_soft_max_kernels<32, 64, 128, 256, 512, 1024, 2048, 4096>(
            x, نقاب, sinks, dst, params, stream, بلاک_dims, بلاک_nums,
            nbytes_shared);
    } ورنہ {
        ثابت size_t nbytes_shared_کم = WARP_سائز * حجم(اعشاری);

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(nbytes_shared_کم), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1)
                    [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                    soft_max_f32<جھوٹ, 0, 0>(
                        x, نقاب, sinks, dst, params,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                    GGML_غیر_مستعمل(شے_ct1);
                });
        });
    }
}

ساکن خالی soft_max_عقب_f32_sycl(ثابت اعشاری *   grad,
                                   ثابت اعشاری *   dstf,
                                   اعشاری *         dst,
                                   ثابت صحیح       ncols,
                                   ثابت صحیح       nrows,
                                   ثابت اعشاری     scale,
                                   dpct::قطار_ptr stream) {
    ثابت dpct::dim3 بلاک_dims(WARP_سائز, 1, 1);
    ثابت dpct::dim3 بلاک_nums(nrows, 1, 1);

    stream->parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                         [=](sycl::nd_شے<3> شے_ct1) {
                             soft_max_عقب_f32(grad, dstf, dst, ncols, scale);
                             GGML_غیر_مستعمل(شے_ct1);
                         });
}

خالی ggml_sycl_op_soft_max(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];
    ثابت ggml_tensor * src2 = dst->src[2];

    ثابت اعشاری * src0_d = (ثابت اعشاری *) src0->ڈیٹا;
    ثابت خالی  * src1_d = src1 ? (ثابت خالی *) src1->ڈیٹا : خالی_اشاریہ;
    ثابت خالی  * src2_d = src2 ? (ثابت خالی *) src2->ڈیٹا : خالی_اشاریہ;
    اعشاری       *  dst_d = (اعشاری *) dst->ڈیٹا;

    dpct::قطار_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    // src1 contains mask and it is optional
    GGML_ASSERT(!src1 || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32);

    ثابت int64_t nrows_x = ggml_nrows(src0);
    ثابت int64_t nrows_y = src0->ne[1];

    ثابت int64_t ne00 = src0->ne[0];

    اعشاری scale    = 1.0f;
    اعشاری max_bias = 0.0f;

    حافظہ_نقل(&scale,    (ثابت اعشاری *) dst->op_params + 0, حجم(اعشاری));
    حافظہ_نقل(&max_bias, (ثابت اعشاری *) dst->op_params + 1, حجم(اعشاری));

    ثابت منطقی use_f16 = (src1 && src1->type == GGML_TYPE_F16);

    ثابت int64_t nb11 = src1 ? src1->nb[1] : 1;
    ثابت int64_t nb12 = src1 ? src1->nb[2] : 1;
    ثابت int64_t nb13 = src1 ? src1->nb[3] : 1;

    ثابت int64_t ne12 = src1 ? src1->ne[2] : 1;
    ثابت int64_t ne13 = src1 ? src1->ne[3] : 1;

    ثابت uint32_t n_سر      = src0->ne[2];
    ثابت uint32_t n_سر_log2 = 1u << (uint32_t) floorf(log2f((اعشاری) n_سر));

    ثابت اعشاری m0 = powf(2.0f, -(max_bias       ) / n_سر_log2);
    ثابت اعشاری m1 = powf(2.0f, -(max_bias / 2.0f) / n_سر_log2);


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

    اگر (use_f16) {
        soft_max_f32_sycl(src0_d, (ثابت sycl::half *)src1_d,
                          (ثابت اعشاری *)src2_d, dst_d, params, stream,
                          ctx.device);
    } ورنہ {
        soft_max_f32_sycl(src0_d, (ثابت اعشاری *)src1_d, (ثابت اعشاری *)src2_d,
                          dst_d, params, stream, ctx.device);
    }
}

خالی ggml_sycl_op_soft_max_عقب(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ثابت ggml_tensor * src0 = dst->src[0]; // grad
    ثابت ggml_tensor * src1 = dst->src[1]; // forward pass output

    ثابت اعشاری * src0_d = (ثابت اعشاری *) src0->ڈیٹا;
    ثابت اعشاری * src1_d = (ثابت اعشاری *) src1->ڈیٹا;
    اعشاری       * dst_d  = (اعشاری       *) dst->ڈیٹا;

    dpct::قطار_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    ثابت int64_t ncols = src0->ne[0];
    ثابت int64_t nrows = ggml_nrows(src0);

    اعشاری scale    = 1.0f;
    اعشاری max_bias = 0.0f;

    حافظہ_نقل(&scale,    (ثابت اعشاری *) dst->op_params + 0, حجم(اعشاری));
    حافظہ_نقل(&max_bias, (ثابت اعشاری *) dst->op_params + 1, حجم(اعشاری));

    GGML_ASSERT(max_bias == 0.0f);

    soft_max_عقب_f32_sycl(src0_d, src1_d, dst_d, ncols, nrows, scale, stream);
}
