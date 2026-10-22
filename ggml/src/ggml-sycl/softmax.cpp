#ضمن "softmax.hpp"
#ضمن <cstdint>
#ضمن <utility>
#ضمن <cmath>


قالب <نام_نوع T> ساكن __dpct_inline__ عائم t2f32(T val) {
    أرجع (عائم) val;
}

قالب <> عائم __dpct_inline__ t2f32<sycl::half>(sycl::half val) {
  أرجع sycl::vec<sycl::half, 1>(val)
      .convert<عائم, sycl::rounding_نما::automatic>()[0];
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
#اگر_تعریف_شده __clang__
#رهنمود clang diagnostic افزودن
#رهنمود clang diagnostic ignored "-Wpass-failed"
#پایان_اگر // __clang__
قالب <منطقي use_shared, صحيح ncols_template, صحيح بلوک_اندازه_template, نام_نوع T>
ساكن فراغ soft_max_f32(ثابت عائم *         x,
                         ثابت T *             ماسک,
                         ثابت عائم *         sinks,
                         عائم *               dst,
                         ثابت soft_max_params p,
                         uint8_t *             dpct_local) {
    خودکار      قلم_ct1 = sycl::ext::oneapi::this_work_قلم::get_nd_قلم<3>();
    ثابت صحيح ncols    = ncols_template == 0 ? p.ncols : ncols_template;
    ثابت صحيح بلوک_اندازه_3 = بلوک_اندازه_template == 0
                               ? قلم_ct1.get_local_range(2)
                               : بلوک_اندازه_template;
    ثابت صحيح nthreads = بلوک_اندازه_3;
    ثابت صحيح nwarps = nthreads / WARP_اندازه;
    ثابت size_t nreduce = nwarps / WARP_اندازه;

    ثابت صحيح tid = قلم_ct1.get_local_id(2);

    ثابت int64_t i03 = قلم_ct1.get_گروه(0);
    ثابت int64_t i02 = قلم_ct1.get_گروه(1);
    ثابت int64_t i01 = قلم_ct1.get_گروه(2);

    //TODO: noncontigous inputs/outputs
    ثابت صحيح rowx = قلم_ct1.get_گروه(2) +
                     قلم_ct1.get_گروه(1) * قلم_ct1.get_گروه_range(2) +
                     قلم_ct1.get_گروه(0) * قلم_ct1.get_گروه_range(2) *
                         قلم_ct1.get_گروه_range(1);

    ثابت int64_t i11 = i01;
    ثابت int64_t i12 = i02 % p.ne12;
    ثابت int64_t i13 = i03 % p.ne13;

    x    += int64_t(rowx)*ncols;
    ماسک += (i11*p.nb11 + i12*p.nb12 + i13*p.nb13) / حجم(T) * (ماسک != اشاره_تهی);
    dst  += int64_t(rowx)*ncols;

    ثابت صحيح warp_id = قلم_ct1.get_local_id(2) / WARP_اندازه;
    ثابت صحيح lane_id = قلم_ct1.get_local_id(2) % WARP_اندازه;

    ثابت عائم شیب = get_alibi_شیب(p.max_bias, i02, p.n_سر_log2, p.m0, p.m1);

    عائم * buf_iw = (عائم *) dpct_local;

    // shared memory buffer to cache values between iterations:
    عائم *vals = use_shared ? buf_iw + sycl::max(nwarps, WARP_اندازه) : dst;
    عائم max_val = sinks ? sinks[i02] : -INFINITY;
#رهنمود unroll
    لكل (صحيح col0 = 0; col0 < ncols; col0 += بلوک_اندازه_3) {
        ثابت صحيح ستون = col0 + tid;

        إذا (ncols_template == 0 && ستون >= ncols) {
            اكسر;
        }

        ثابت عائم val = x[ستون]*p.scale + (ماسک ? شیب*t2f32(ماسک[ستون]) : 0.0f);

        vals[ستون] = val;
        max_val   = sycl::max(max_val, val);
    }
    // find the max value in the block
    max_val = warp_reduce_max<WARP_اندازه>(max_val);

    إذا (بلوک_اندازه_3 > WARP_اندازه) {
        إذا (lane_id == 0) {
            buf_iw[warp_id] = max_val;
        }
        قلم_ct1.barrier();

        max_val = -INFINITY;
        لكل (صحيح i = lane_id; i < nwarps; i += WARP_اندازه) {
            max_val = sycl::max(max_val, buf_iw[i]);
        }
        max_val = warp_reduce_max<WARP_اندازه>(max_val);
    }
    عائم tmp = 0.0f; // partial sum

#رهنمود unroll
    لكل (صحيح col0 = 0; col0 < ncols; col0 += بلوک_اندازه_3) {
        ثابت صحيح ستون = col0 + tid;

        إذا (ncols_template == 0 && ستون >= ncols) {
            اكسر;
        }

        ثابت عائم val = sycl::native::نمایی_تابع(sycl::max(vals[ستون] - max_val, -80.0f));
        tmp += val;
        vals[ستون] = val;
    }
    // find the sum of exps in the block
    tmp = warp_reduce_جمع<WARP_اندازه>(tmp);
    إذا (بلوک_اندازه_3 > WARP_اندازه) {
        قلم_ct1.barrier();
        إذا (warp_id == 0) {
            buf_iw[lane_id] = 0.0f;
            لكل (size_t i = 1; i < nreduce; i += 1) {
                buf_iw[lane_id + i * WARP_اندازه] = 0.f;
            }
        }
        قلم_ct1.barrier();

        إذا (lane_id == 0) {
            buf_iw[warp_id] = tmp;
        }
        قلم_ct1.barrier();

        tmp = buf_iw[lane_id];
        لكل (size_t i = 1; i < nreduce; i += 1) {
            tmp += buf_iw[lane_id + i * WARP_اندازه];
        }
        tmp = warp_reduce_جمع<WARP_اندازه>(tmp);
    }
    إذا (sinks) {
        tmp += sycl::native::نمایی_تابع(sycl::max(sinks[i02] - max_val, -80.0f));
    }
    ثابت عائم inv_جمع = 1.0f / tmp;

#رهنمود unroll
    لكل (صحيح col0 = 0; col0 < ncols; col0 += بلوک_اندازه_3) {
        ثابت صحيح ستون = col0 + tid;

        إذا (ncols_template == 0 && ستون >= ncols) {
            أرجع;
        }

        dst[ستون] = vals[ستون] * inv_جمع;
    }
}
#اگر_تعریف_شده __clang__
#رهنمود clang diagnostic برداشت
#پایان_اگر // __clang__

ساكن فراغ soft_max_پشت_f32(ثابت عائم *grad, ثابت عائم *dstf, عائم *dst,
                              ثابت صحيح ncols, ثابت عائم scale) {
    خودکار      قلم_ct1 = sycl::ext::oneapi::this_work_قلم::get_nd_قلم<3>();
    ثابت صحيح tid      = قلم_ct1.get_local_id(2);
    ثابت صحيح rowx     = قلم_ct1.get_گروه(2);

    grad += int64_t(rowx)*ncols;
    dstf += int64_t(rowx)*ncols;
    dst  += int64_t(rowx)*ncols;

    عائم dgf_dot = 0.0f; // dot product of dst from forward pass and gradients

    لكل (صحيح ستون = tid; ستون < ncols; ستون += WARP_اندازه) {
        dgf_dot += dstf[ستون]*grad[ستون];
    }

    dgf_dot = warp_reduce_جمع<WARP_اندازه>(dgf_dot);

    لكل (صحيح ستون = tid; ستون < ncols; ستون += WARP_اندازه) {
        dst[ستون] = scale * (grad[ستون] - dgf_dot) * dstf[ستون];
    }
}

قالب <صحيح... Ns, نام_نوع T>
ساكن فراغ launch_soft_max_kernels(ثابت عائم *           x,
                                    ثابت T *               ماسک,
                                    ثابت عائم *           sinks,
                                    عائم *                 dst,
                                    ثابت soft_max_params & p,
                                    dpct::صف_ptr         stream,
                                    dpct::dim3              بلوک_dims,
                                    dpct::dim3              بلوک_nums,
                                    size_t                  nbytes_shared)
{
    خودکار launch_kernel = [=](خودکار I) -> منطقي {
        constexpr صحيح ncols = decltype(I)::value;
        constexpr صحيح بلوک = (ncols > 1024 ? 1024 : ncols);
        إذا (p.ncols == ncols) {
            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                    sycl::range<1>(nbytes_shared), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(بلوک_nums * بلوک_dims, بلوک_dims),
                    [=](sycl::nd_قلم<3> قلم_ct1) [[sycl::reqd_sub_گروه_اندازه(
                        WARP_اندازه)]] {
                        soft_max_f32<درست, ncols, بلوک>(
                            x, ماسک, sinks, dst, p,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                        GGML_استفادهنشده(قلم_ct1);
                    });
            });
            أرجع درست;
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
            sycl::nd_range<3>(بلوک_nums * بلوک_dims, بلوک_dims),
            [=](sycl::nd_قلم<3> قلم_ct1)
                [[sycl::reqd_sub_گروه_اندازه(WARP_اندازه)]] {
                    soft_max_f32<درست, 0, 0>(
                        x, ماسک, sinks, dst, p,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                    GGML_استفادهنشده(قلم_ct1);
                });
    });
}

قالب <نام_نوع T>
ساكن فراغ soft_max_f32_sycl(ثابت عائم *x, ثابت T *ماسک,
                              ثابت عائم *sinks, عائم *dst,
                              ثابت soft_max_params &params,
                              dpct::صف_ptr stream, صحيح device) {
    صحيح nth = WARP_اندازه;
    صحيح max_بلوک_اندازه = ggml_sycl_info().max_work_گروه_sizes[device];
    ثابت int64_t ncols_x = params.ncols;

    طالما (nth < ncols_x && nth < max_بلوک_اندازه) nth *= 2;
    إذا (nth>max_بلوک_اندازه) nth = max_بلوک_اندازه;

    ثابت dpct::dim3 بلوک_dims(nth, 1, 1);
    ثابت dpct::dim3 بلوک_nums(params.ne01, params.ne02, params.ne03);
    ثابت size_t nbytes_shared =
        (GGML_PAD(ncols_x, WARP_اندازه) + WARP_اندازه) * حجم(عائم);

    ثابت صحيح id       = get_current_device_id();
    ثابت size_t smpbo = ggml_sycl_info().devices[id].smpbo;

    إذا (nbytes_shared <= smpbo && ncols_x <= max_بلوک_اندازه) {
        launch_soft_max_kernels<32, 64, 128, 256, 512, 1024, 2048, 4096>(
            x, ماسک, sinks, dst, params, stream, بلوک_dims, بلوک_nums,
            nbytes_shared);
    } وإلا {
        ثابت size_t nbytes_shared_کم = WARP_اندازه * حجم(عائم);

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(nbytes_shared_کم), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(بلوک_nums * بلوک_dims, بلوک_dims),
                [=](sycl::nd_قلم<3> قلم_ct1)
                    [[sycl::reqd_sub_گروه_اندازه(WARP_اندازه)]] {
                    soft_max_f32<خطأ, 0, 0>(
                        x, ماسک, sinks, dst, params,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                    GGML_استفادهنشده(قلم_ct1);
                });
        });
    }
}

ساكن فراغ soft_max_پشت_f32_sycl(ثابت عائم *   grad,
                                   ثابت عائم *   dstf,
                                   عائم *         dst,
                                   ثابت صحيح       ncols,
                                   ثابت صحيح       nrows,
                                   ثابت عائم     scale,
                                   dpct::صف_ptr stream) {
    ثابت dpct::dim3 بلوک_dims(WARP_اندازه, 1, 1);
    ثابت dpct::dim3 بلوک_nums(nrows, 1, 1);

    stream->parallel_for(sycl::nd_range<3>(بلوک_nums * بلوک_dims, بلوک_dims),
                         [=](sycl::nd_قلم<3> قلم_ct1) {
                             soft_max_پشت_f32(grad, dstf, dst, ncols, scale);
                             GGML_استفادهنشده(قلم_ct1);
                         });
}

فراغ ggml_sycl_op_soft_max(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];
    ثابت ggml_tensor * src2 = dst->src[2];

    ثابت عائم * src0_d = (ثابت عائم *) src0->داده;
    ثابت فراغ  * src1_d = src1 ? (ثابت فراغ *) src1->داده : اشاره_تهی;
    ثابت فراغ  * src2_d = src2 ? (ثابت فراغ *) src2->داده : اشاره_تهی;
    عائم       *  dst_d = (عائم *) dst->داده;

    dpct::صف_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    // src1 contains mask and it is optional
    GGML_ASSERT(!src1 || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32);

    ثابت int64_t nrows_x = ggml_nrows(src0);
    ثابت int64_t nrows_y = src0->ne[1];

    ثابت int64_t ne00 = src0->ne[0];

    عائم scale    = 1.0f;
    عائم max_bias = 0.0f;

    کپی_حافظه(&scale,    (ثابت عائم *) dst->op_params + 0, حجم(عائم));
    کپی_حافظه(&max_bias, (ثابت عائم *) dst->op_params + 1, حجم(عائم));

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

فراغ ggml_sycl_op_soft_max_پشت(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ثابت ggml_tensor * src0 = dst->src[0]; // grad
    ثابت ggml_tensor * src1 = dst->src[1]; // forward pass output

    ثابت عائم * src0_d = (ثابت عائم *) src0->داده;
    ثابت عائم * src1_d = (ثابت عائم *) src1->داده;
    عائم       * dst_d  = (عائم       *) dst->داده;

    dpct::صف_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    ثابت int64_t ncols = src0->ne[0];
    ثابت int64_t nrows = ggml_nrows(src0);

    عائم scale    = 1.0f;
    عائم max_bias = 0.0f;

    کپی_حافظه(&scale,    (ثابت عائم *) dst->op_params + 0, حجم(عائم));
    کپی_حافظه(&max_bias, (ثابت عائم *) dst->op_params + 1, حجم(عائم));

    GGML_ASSERT(max_bias == 0.0f);

    soft_max_پشت_f32_sycl(src0_d, src1_d, dst_d, ncols, nrows, scale, stream);
}
