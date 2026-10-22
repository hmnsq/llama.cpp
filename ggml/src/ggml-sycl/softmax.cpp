#ضمن "softmax.hpp"
#ضمن <cstdint>
#ضمن <utility>
#ضمن <cmath>


کينډۍ <ډول_نوم T> ساكن __dpct_inline__ عائم t2f32(T val) {
    أرجع (عائم) val;
}

کينډۍ <> عائم __dpct_inline__ t2f32<sycl::half>(sycl::half val) {
  أرجع sycl::vec<sycl::half, 1>(val)
      .convert<عائم, sycl::rounding_ډېرپېښ::automatic>()[0];
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
#که_تعریف_شوی __clang__
#لارښوونه clang diagnostic ټېل
#لارښوونه clang diagnostic ignored "-Wpass-failed"
#د_که_پای // __clang__
کينډۍ <منطقي use_shared, صحيح ncols_template, صحيح بلاک_اندازه_template, ډول_نوم T>
ساكن فراغ soft_max_f32(ثابت عائم *         x,
                         ثابت T *             ماسک,
                         ثابت عائم *         sinks,
                         عائم *               dst,
                         ثابت soft_max_params p,
                         uint8_t *             dpct_local) {
    خپلکاری      توکی_ct1 = sycl::ext::oneapi::this_work_توکی::get_nd_توکی<3>();
    ثابت صحيح ncols    = ncols_template == 0 ? p.ncols : ncols_template;
    ثابت صحيح بلاک_اندازه_3 = بلاک_اندازه_template == 0
                               ? توکی_ct1.get_local_range(2)
                               : بلاک_اندازه_template;
    ثابت صحيح nthreads = بلاک_اندازه_3;
    ثابت صحيح nwarps = nthreads / WARP_اندازه;
    ثابت size_t nreduce = nwarps / WARP_اندازه;

    ثابت صحيح tid = توکی_ct1.get_local_id(2);

    ثابت int64_t i03 = توکی_ct1.get_ډله(0);
    ثابت int64_t i02 = توکی_ct1.get_ډله(1);
    ثابت int64_t i01 = توکی_ct1.get_ډله(2);

    //TODO: noncontigous inputs/outputs
    ثابت صحيح rowx = توکی_ct1.get_ډله(2) +
                     توکی_ct1.get_ډله(1) * توکی_ct1.get_ډله_range(2) +
                     توکی_ct1.get_ډله(0) * توکی_ct1.get_ډله_range(2) *
                         توکی_ct1.get_ډله_range(1);

    ثابت int64_t i11 = i01;
    ثابت int64_t i12 = i02 % p.ne12;
    ثابت int64_t i13 = i03 % p.ne13;

    x    += int64_t(rowx)*ncols;
    ماسک += (i11*p.nb11 + i12*p.nb12 + i13*p.nb13) / حجم(T) * (ماسک != تش_اشاره);
    dst  += int64_t(rowx)*ncols;

    ثابت صحيح warp_id = توکی_ct1.get_local_id(2) / WARP_اندازه;
    ثابت صحيح lane_id = توکی_ct1.get_local_id(2) % WARP_اندازه;

    ثابت عائم میلان = get_alibi_میلان(p.max_bias, i02, p.n_سر_log2, p.m0, p.m1);

    عائم * buf_iw = (عائم *) dpct_local;

    // shared memory buffer to cache values between iterations:
    عائم *vals = use_shared ? buf_iw + sycl::max(nwarps, WARP_اندازه) : dst;
    عائم max_val = sinks ? sinks[i02] : -INFINITY;
#لارښوونه unroll
    لكل (صحيح col0 = 0; col0 < ncols; col0 += بلاک_اندازه_3) {
        ثابت صحيح ستون = col0 + tid;

        إذا (ncols_template == 0 && ستون >= ncols) {
            اكسر;
        }

        ثابت عائم val = x[ستون]*p.scale + (ماسک ? میلان*t2f32(ماسک[ستون]) : 0.0f);

        vals[ستون] = val;
        max_val   = sycl::max(max_val, val);
    }
    // find the max value in the block
    max_val = warp_reduce_max<WARP_اندازه>(max_val);

    إذا (بلاک_اندازه_3 > WARP_اندازه) {
        إذا (lane_id == 0) {
            buf_iw[warp_id] = max_val;
        }
        توکی_ct1.barrier();

        max_val = -INFINITY;
        لكل (صحيح i = lane_id; i < nwarps; i += WARP_اندازه) {
            max_val = sycl::max(max_val, buf_iw[i]);
        }
        max_val = warp_reduce_max<WARP_اندازه>(max_val);
    }
    عائم tmp = 0.0f; // partial sum

#لارښوونه unroll
    لكل (صحيح col0 = 0; col0 < ncols; col0 += بلاک_اندازه_3) {
        ثابت صحيح ستون = col0 + tid;

        إذا (ncols_template == 0 && ستون >= ncols) {
            اكسر;
        }

        ثابت عائم val = sycl::native::توانیز_تابع(sycl::max(vals[ستون] - max_val, -80.0f));
        tmp += val;
        vals[ستون] = val;
    }
    // find the sum of exps in the block
    tmp = warp_reduce_مجموعه<WARP_اندازه>(tmp);
    إذا (بلاک_اندازه_3 > WARP_اندازه) {
        توکی_ct1.barrier();
        إذا (warp_id == 0) {
            buf_iw[lane_id] = 0.0f;
            لكل (size_t i = 1; i < nreduce; i += 1) {
                buf_iw[lane_id + i * WARP_اندازه] = 0.f;
            }
        }
        توکی_ct1.barrier();

        إذا (lane_id == 0) {
            buf_iw[warp_id] = tmp;
        }
        توکی_ct1.barrier();

        tmp = buf_iw[lane_id];
        لكل (size_t i = 1; i < nreduce; i += 1) {
            tmp += buf_iw[lane_id + i * WARP_اندازه];
        }
        tmp = warp_reduce_مجموعه<WARP_اندازه>(tmp);
    }
    إذا (sinks) {
        tmp += sycl::native::توانیز_تابع(sycl::max(sinks[i02] - max_val, -80.0f));
    }
    ثابت عائم inv_مجموعه = 1.0f / tmp;

#لارښوونه unroll
    لكل (صحيح col0 = 0; col0 < ncols; col0 += بلاک_اندازه_3) {
        ثابت صحيح ستون = col0 + tid;

        إذا (ncols_template == 0 && ستون >= ncols) {
            أرجع;
        }

        dst[ستون] = vals[ستون] * inv_مجموعه;
    }
}
#که_تعریف_شوی __clang__
#لارښوونه clang diagnostic راوباسه
#د_که_پای // __clang__

ساكن فراغ soft_max_شا_f32(ثابت عائم *grad, ثابت عائم *dstf, عائم *dst,
                              ثابت صحيح ncols, ثابت عائم scale) {
    خپلکاری      توکی_ct1 = sycl::ext::oneapi::this_work_توکی::get_nd_توکی<3>();
    ثابت صحيح tid      = توکی_ct1.get_local_id(2);
    ثابت صحيح rowx     = توکی_ct1.get_ډله(2);

    grad += int64_t(rowx)*ncols;
    dstf += int64_t(rowx)*ncols;
    dst  += int64_t(rowx)*ncols;

    عائم dgf_dot = 0.0f; // dot product of dst from forward pass and gradients

    لكل (صحيح ستون = tid; ستون < ncols; ستون += WARP_اندازه) {
        dgf_dot += dstf[ستون]*grad[ستون];
    }

    dgf_dot = warp_reduce_مجموعه<WARP_اندازه>(dgf_dot);

    لكل (صحيح ستون = tid; ستون < ncols; ستون += WARP_اندازه) {
        dst[ستون] = scale * (grad[ستون] - dgf_dot) * dstf[ستون];
    }
}

کينډۍ <صحيح... Ns, ډول_نوم T>
ساكن فراغ launch_soft_max_kernels(ثابت عائم *           x,
                                    ثابت T *               ماسک,
                                    ثابت عائم *           sinks,
                                    عائم *                 dst,
                                    ثابت soft_max_params & p,
                                    dpct::کتار_ptr         stream,
                                    dpct::dim3              بلاک_dims,
                                    dpct::dim3              بلاک_nums,
                                    size_t                  nbytes_shared)
{
    خپلکاری launch_kernel = [=](خپلکاری I) -> منطقي {
        constexpr صحيح ncols = decltype(I)::value;
        constexpr صحيح بلاک = (ncols > 1024 ? 1024 : ncols);
        إذا (p.ncols == ncols) {
            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                    sycl::range<1>(nbytes_shared), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                    [=](sycl::nd_توکی<3> توکی_ct1) [[sycl::reqd_sub_ډله_اندازه(
                        WARP_اندازه)]] {
                        soft_max_f32<ریښتیا, ncols, بلاک>(
                            x, ماسک, sinks, dst, p,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                        GGML_ناکارول_شوی(توکی_ct1);
                    });
            });
            أرجع ریښتیا;
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
            [=](sycl::nd_توکی<3> توکی_ct1)
                [[sycl::reqd_sub_ډله_اندازه(WARP_اندازه)]] {
                    soft_max_f32<ریښتیا, 0, 0>(
                        x, ماسک, sinks, dst, p,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                    GGML_ناکارول_شوی(توکی_ct1);
                });
    });
}

کينډۍ <ډول_نوم T>
ساكن فراغ soft_max_f32_sycl(ثابت عائم *x, ثابت T *ماسک,
                              ثابت عائم *sinks, عائم *dst,
                              ثابت soft_max_params &params,
                              dpct::کتار_ptr stream, صحيح device) {
    صحيح nth = WARP_اندازه;
    صحيح max_بلاک_اندازه = ggml_sycl_info().max_work_ډله_sizes[device];
    ثابت int64_t ncols_x = params.ncols;

    طالما (nth < ncols_x && nth < max_بلاک_اندازه) nth *= 2;
    إذا (nth>max_بلاک_اندازه) nth = max_بلاک_اندازه;

    ثابت dpct::dim3 بلاک_dims(nth, 1, 1);
    ثابت dpct::dim3 بلاک_nums(params.ne01, params.ne02, params.ne03);
    ثابت size_t nbytes_shared =
        (GGML_PAD(ncols_x, WARP_اندازه) + WARP_اندازه) * حجم(عائم);

    ثابت صحيح id       = get_current_device_id();
    ثابت size_t smpbo = ggml_sycl_info().devices[id].smpbo;

    إذا (nbytes_shared <= smpbo && ncols_x <= max_بلاک_اندازه) {
        launch_soft_max_kernels<32, 64, 128, 256, 512, 1024, 2048, 4096>(
            x, ماسک, sinks, dst, params, stream, بلاک_dims, بلاک_nums,
            nbytes_shared);
    } وإلا {
        ثابت size_t nbytes_shared_ټیټ = WARP_اندازه * حجم(عائم);

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(nbytes_shared_ټیټ), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                [=](sycl::nd_توکی<3> توکی_ct1)
                    [[sycl::reqd_sub_ډله_اندازه(WARP_اندازه)]] {
                    soft_max_f32<خطأ, 0, 0>(
                        x, ماسک, sinks, dst, params,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                    GGML_ناکارول_شوی(توکی_ct1);
                });
        });
    }
}

ساكن فراغ soft_max_شا_f32_sycl(ثابت عائم *   grad,
                                   ثابت عائم *   dstf,
                                   عائم *         dst,
                                   ثابت صحيح       ncols,
                                   ثابت صحيح       nrows,
                                   ثابت عائم     scale,
                                   dpct::کتار_ptr stream) {
    ثابت dpct::dim3 بلاک_dims(WARP_اندازه, 1, 1);
    ثابت dpct::dim3 بلاک_nums(nrows, 1, 1);

    stream->parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                         [=](sycl::nd_توکی<3> توکی_ct1) {
                             soft_max_شا_f32(grad, dstf, dst, ncols, scale);
                             GGML_ناکارول_شوی(توکی_ct1);
                         });
}

فراغ ggml_sycl_op_soft_max(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];
    ثابت ggml_tensor * src2 = dst->src[2];

    ثابت عائم * src0_d = (ثابت عائم *) src0->معلومات;
    ثابت فراغ  * src1_d = src1 ? (ثابت فراغ *) src1->معلومات : تش_اشاره;
    ثابت فراغ  * src2_d = src2 ? (ثابت فراغ *) src2->معلومات : تش_اشاره;
    عائم       *  dst_d = (عائم *) dst->معلومات;

    dpct::کتار_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    // src1 contains mask and it is optional
    GGML_ASSERT(!src1 || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32);

    ثابت int64_t nrows_x = ggml_nrows(src0);
    ثابت int64_t nrows_y = src0->ne[1];

    ثابت int64_t ne00 = src0->ne[0];

    عائم scale    = 1.0f;
    عائم max_bias = 0.0f;

    حافظه_نقل(&scale,    (ثابت عائم *) dst->op_params + 0, حجم(عائم));
    حافظه_نقل(&max_bias, (ثابت عائم *) dst->op_params + 1, حجم(عائم));

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

فراغ ggml_sycl_op_soft_max_شا(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ثابت ggml_tensor * src0 = dst->src[0]; // grad
    ثابت ggml_tensor * src1 = dst->src[1]; // forward pass output

    ثابت عائم * src0_d = (ثابت عائم *) src0->معلومات;
    ثابت عائم * src1_d = (ثابت عائم *) src1->معلومات;
    عائم       * dst_d  = (عائم       *) dst->معلومات;

    dpct::کتار_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    ثابت int64_t ncols = src0->ne[0];
    ثابت int64_t nrows = ggml_nrows(src0);

    عائم scale    = 1.0f;
    عائم max_bias = 0.0f;

    حافظه_نقل(&scale,    (ثابت عائم *) dst->op_params + 0, حجم(عائم));
    حافظه_نقل(&max_bias, (ثابت عائم *) dst->op_params + 1, حجم(عائم));

    GGML_ASSERT(max_bias == 0.0f);

    soft_max_شا_f32_sycl(src0_d, src1_d, dst_d, ncols, nrows, scale, stream);
}
