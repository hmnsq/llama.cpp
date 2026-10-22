#ضمن "softmax.hpp"
#ضمن <cstdint>
#ضمن <utility>
#ضمن <cmath>


سانچا <قسم_ناں T> ساكن __dpct_inline__ عائم t2f32(T val) {
    أرجع (عائم) val;
}

سانچا <> عائم __dpct_inline__ t2f32<sycl::half>(sycl::half val) {
  أرجع sycl::vec<sycl::half, 1>(val)
      .convert<عائم, sycl::rounding_منوال::automatic>()[0];
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
#جے_تعریف_شدہ __clang__
#ہدایت clang diagnostic ڈالو
#ہدایت clang diagnostic ignored "-Wpass-failed"
#جے_انت // __clang__
سانچا <منطقي use_shared, صحيح ncols_template, صحيح بلاک_ناپ_template, قسم_ناں T>
ساكن فراغ soft_max_f32(ثابت عائم *         x,
                         ثابت T *             نقاب,
                         ثابت عائم *         sinks,
                         عائم *               dst,
                         ثابت soft_max_params p,
                         uint8_t *             dpct_local) {
    خودکار      شے_ct1 = sycl::ext::oneapi::this_work_شے::get_nd_شے<3>();
    ثابت صحيح ncols    = ncols_template == 0 ? p.ncols : ncols_template;
    ثابت صحيح بلاک_ناپ_3 = بلاک_ناپ_template == 0
                               ? شے_ct1.get_local_range(2)
                               : بلاک_ناپ_template;
    ثابت صحيح nthreads = بلاک_ناپ_3;
    ثابت صحيح nwarps = nthreads / WARP_ناپ;
    ثابت size_t nreduce = nwarps / WARP_ناپ;

    ثابت صحيح tid = شے_ct1.get_local_id(2);

    ثابت int64_t i03 = شے_ct1.get_گروہ(0);
    ثابت int64_t i02 = شے_ct1.get_گروہ(1);
    ثابت int64_t i01 = شے_ct1.get_گروہ(2);

    //TODO: noncontigous inputs/outputs
    ثابت صحيح rowx = شے_ct1.get_گروہ(2) +
                     شے_ct1.get_گروہ(1) * شے_ct1.get_گروہ_range(2) +
                     شے_ct1.get_گروہ(0) * شے_ct1.get_گروہ_range(2) *
                         شے_ct1.get_گروہ_range(1);

    ثابت int64_t i11 = i01;
    ثابت int64_t i12 = i02 % p.ne12;
    ثابت int64_t i13 = i03 % p.ne13;

    x    += int64_t(rowx)*ncols;
    نقاب += (i11*p.nb11 + i12*p.nb12 + i13*p.nb13) / حجم(T) * (نقاب != خالی_سوچک);
    dst  += int64_t(rowx)*ncols;

    ثابت صحيح warp_id = شے_ct1.get_local_id(2) / WARP_ناپ;
    ثابت صحيح lane_id = شے_ct1.get_local_id(2) % WARP_ناپ;

    ثابت عائم ڈھلوان = get_alibi_ڈھلوان(p.max_bias, i02, p.n_سر_log2, p.m0, p.m1);

    عائم * buf_iw = (عائم *) dpct_local;

    // shared memory buffer to cache values between iterations:
    عائم *vals = use_shared ? buf_iw + sycl::max(nwarps, WARP_ناپ) : dst;
    عائم max_val = sinks ? sinks[i02] : -INFINITY;
#ہدایت unroll
    لكل (صحيح col0 = 0; col0 < ncols; col0 += بلاک_ناپ_3) {
        ثابت صحيح خانہ = col0 + tid;

        إذا (ncols_template == 0 && خانہ >= ncols) {
            اكسر;
        }

        ثابت عائم val = x[خانہ]*p.scale + (نقاب ? ڈھلوان*t2f32(نقاب[خانہ]) : 0.0f);

        vals[خانہ] = val;
        max_val   = sycl::max(max_val, val);
    }
    // find the max value in the block
    max_val = warp_reduce_max<WARP_ناپ>(max_val);

    إذا (بلاک_ناپ_3 > WARP_ناپ) {
        إذا (lane_id == 0) {
            buf_iw[warp_id] = max_val;
        }
        شے_ct1.barrier();

        max_val = -INFINITY;
        لكل (صحيح i = lane_id; i < nwarps; i += WARP_ناپ) {
            max_val = sycl::max(max_val, buf_iw[i]);
        }
        max_val = warp_reduce_max<WARP_ناپ>(max_val);
    }
    عائم tmp = 0.0f; // partial sum

#ہدایت unroll
    لكل (صحيح col0 = 0; col0 < ncols; col0 += بلاک_ناپ_3) {
        ثابت صحيح خانہ = col0 + tid;

        إذا (ncols_template == 0 && خانہ >= ncols) {
            اكسر;
        }

        ثابت عائم val = sycl::native::قوت_نما(sycl::max(vals[خانہ] - max_val, -80.0f));
        tmp += val;
        vals[خانہ] = val;
    }
    // find the sum of exps in the block
    tmp = warp_reduce_جوڑ<WARP_ناپ>(tmp);
    إذا (بلاک_ناپ_3 > WARP_ناپ) {
        شے_ct1.barrier();
        إذا (warp_id == 0) {
            buf_iw[lane_id] = 0.0f;
            لكل (size_t i = 1; i < nreduce; i += 1) {
                buf_iw[lane_id + i * WARP_ناپ] = 0.f;
            }
        }
        شے_ct1.barrier();

        إذا (lane_id == 0) {
            buf_iw[warp_id] = tmp;
        }
        شے_ct1.barrier();

        tmp = buf_iw[lane_id];
        لكل (size_t i = 1; i < nreduce; i += 1) {
            tmp += buf_iw[lane_id + i * WARP_ناپ];
        }
        tmp = warp_reduce_جوڑ<WARP_ناپ>(tmp);
    }
    إذا (sinks) {
        tmp += sycl::native::قوت_نما(sycl::max(sinks[i02] - max_val, -80.0f));
    }
    ثابت عائم inv_جوڑ = 1.0f / tmp;

#ہدایت unroll
    لكل (صحيح col0 = 0; col0 < ncols; col0 += بلاک_ناپ_3) {
        ثابت صحيح خانہ = col0 + tid;

        إذا (ncols_template == 0 && خانہ >= ncols) {
            أرجع;
        }

        dst[خانہ] = vals[خانہ] * inv_جوڑ;
    }
}
#جے_تعریف_شدہ __clang__
#ہدایت clang diagnostic نکالو
#جے_انت // __clang__

ساكن فراغ soft_max_پٹھ_f32(ثابت عائم *grad, ثابت عائم *dstf, عائم *dst,
                              ثابت صحيح ncols, ثابت عائم scale) {
    خودکار      شے_ct1 = sycl::ext::oneapi::this_work_شے::get_nd_شے<3>();
    ثابت صحيح tid      = شے_ct1.get_local_id(2);
    ثابت صحيح rowx     = شے_ct1.get_گروہ(2);

    grad += int64_t(rowx)*ncols;
    dstf += int64_t(rowx)*ncols;
    dst  += int64_t(rowx)*ncols;

    عائم dgf_dot = 0.0f; // dot product of dst from forward pass and gradients

    لكل (صحيح خانہ = tid; خانہ < ncols; خانہ += WARP_ناپ) {
        dgf_dot += dstf[خانہ]*grad[خانہ];
    }

    dgf_dot = warp_reduce_جوڑ<WARP_ناپ>(dgf_dot);

    لكل (صحيح خانہ = tid; خانہ < ncols; خانہ += WARP_ناپ) {
        dst[خانہ] = scale * (grad[خانہ] - dgf_dot) * dstf[خانہ];
    }
}

سانچا <صحيح... Ns, قسم_ناں T>
ساكن فراغ launch_soft_max_kernels(ثابت عائم *           x,
                                    ثابت T *               نقاب,
                                    ثابت عائم *           sinks,
                                    عائم *                 dst,
                                    ثابت soft_max_params & p,
                                    dpct::قطار_ptr         stream,
                                    dpct::dim3              بلاک_dims,
                                    dpct::dim3              بلاک_nums,
                                    size_t                  nbytes_shared)
{
    خودکار launch_kernel = [=](خودکار I) -> منطقي {
        constexpr صحيح ncols = decltype(I)::value;
        constexpr صحيح بلاک = (ncols > 1024 ? 1024 : ncols);
        إذا (p.ncols == ncols) {
            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                    sycl::range<1>(nbytes_shared), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                    [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_ناپ(
                        WARP_ناپ)]] {
                        soft_max_f32<سچ, ncols, بلاک>(
                            x, نقاب, sinks, dst, p,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                        GGML_بن_ورتیا(شے_ct1);
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
            sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<3> شے_ct1)
                [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                    soft_max_f32<سچ, 0, 0>(
                        x, نقاب, sinks, dst, p,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                    GGML_بن_ورتیا(شے_ct1);
                });
    });
}

سانچا <قسم_ناں T>
ساكن فراغ soft_max_f32_sycl(ثابت عائم *x, ثابت T *نقاب,
                              ثابت عائم *sinks, عائم *dst,
                              ثابت soft_max_params &params,
                              dpct::قطار_ptr stream, صحيح device) {
    صحيح nth = WARP_ناپ;
    صحيح max_بلاک_ناپ = ggml_sycl_info().max_work_گروہ_sizes[device];
    ثابت int64_t ncols_x = params.ncols;

    طالما (nth < ncols_x && nth < max_بلاک_ناپ) nth *= 2;
    إذا (nth>max_بلاک_ناپ) nth = max_بلاک_ناپ;

    ثابت dpct::dim3 بلاک_dims(nth, 1, 1);
    ثابت dpct::dim3 بلاک_nums(params.ne01, params.ne02, params.ne03);
    ثابت size_t nbytes_shared =
        (GGML_PAD(ncols_x, WARP_ناپ) + WARP_ناپ) * حجم(عائم);

    ثابت صحيح id       = get_current_device_id();
    ثابت size_t smpbo = ggml_sycl_info().devices[id].smpbo;

    إذا (nbytes_shared <= smpbo && ncols_x <= max_بلاک_ناپ) {
        launch_soft_max_kernels<32, 64, 128, 256, 512, 1024, 2048, 4096>(
            x, نقاب, sinks, dst, params, stream, بلاک_dims, بلاک_nums,
            nbytes_shared);
    } وإلا {
        ثابت size_t nbytes_shared_کم = WARP_ناپ * حجم(عائم);

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(nbytes_shared_کم), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1)
                    [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                    soft_max_f32<خطأ, 0, 0>(
                        x, نقاب, sinks, dst, params,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                    GGML_بن_ورتیا(شے_ct1);
                });
        });
    }
}

ساكن فراغ soft_max_پٹھ_f32_sycl(ثابت عائم *   grad,
                                   ثابت عائم *   dstf,
                                   عائم *         dst,
                                   ثابت صحيح       ncols,
                                   ثابت صحيح       nrows,
                                   ثابت عائم     scale,
                                   dpct::قطار_ptr stream) {
    ثابت dpct::dim3 بلاک_dims(WARP_ناپ, 1, 1);
    ثابت dpct::dim3 بلاک_nums(nrows, 1, 1);

    stream->parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                         [=](sycl::nd_شے<3> شے_ct1) {
                             soft_max_پٹھ_f32(grad, dstf, dst, ncols, scale);
                             GGML_بن_ورتیا(شے_ct1);
                         });
}

فراغ ggml_sycl_op_soft_max(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];
    ثابت ggml_tensor * src2 = dst->src[2];

    ثابت عائم * src0_d = (ثابت عائم *) src0->انکڑے;
    ثابت فراغ  * src1_d = src1 ? (ثابت فراغ *) src1->انکڑے : خالی_سوچک;
    ثابت فراغ  * src2_d = src2 ? (ثابت فراغ *) src2->انکڑے : خالی_سوچک;
    عائم       *  dst_d = (عائم *) dst->انکڑے;

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

    حافظہ_نقل(&scale,    (ثابت عائم *) dst->op_params + 0, حجم(عائم));
    حافظہ_نقل(&max_bias, (ثابت عائم *) dst->op_params + 1, حجم(عائم));

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

فراغ ggml_sycl_op_soft_max_پٹھ(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ثابت ggml_tensor * src0 = dst->src[0]; // grad
    ثابت ggml_tensor * src1 = dst->src[1]; // forward pass output

    ثابت عائم * src0_d = (ثابت عائم *) src0->انکڑے;
    ثابت عائم * src1_d = (ثابت عائم *) src1->انکڑے;
    عائم       * dst_d  = (عائم       *) dst->انکڑے;

    dpct::قطار_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    ثابت int64_t ncols = src0->ne[0];
    ثابت int64_t nrows = ggml_nrows(src0);

    عائم scale    = 1.0f;
    عائم max_bias = 0.0f;

    حافظہ_نقل(&scale,    (ثابت عائم *) dst->op_params + 0, حجم(عائم));
    حافظہ_نقل(&max_bias, (ثابت عائم *) dst->op_params + 1, حجم(عائم));

    GGML_ASSERT(max_bias == 0.0f);

    soft_max_پٹھ_f32_sycl(src0_d, src1_d, dst_d, ncols, nrows, scale, stream);
}
