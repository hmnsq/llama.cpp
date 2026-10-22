#ضمن "norm.hpp"
#ضمن "ggml-sycl/common.hpp"
#ضمن "ggml-sycl/presets.hpp"

ساكن فراغ norm_f32(ثابت عائم* x, عائم* dst, ثابت صحيح ncols,
    ثابت int64_t src_stride_ستون, ثابت int64_t src_stride_لیکه, ثابت int64_t src_stride_channel, ثابت int64_t src_stride_sample,
    ثابت int64_t dst_stride_ستون, ثابت int64_t dst_stride_لیکه, ثابت int64_t dst_stride_channel, ثابت int64_t dst_stride_sample,
    ثابت عائم eps, ثابت sycl::nd_توکی<3>& توکی_ct1, sycl::float2* s_مجموعه, صحيح بلاک_اندازه_3) {

    ثابت صحيح nthreads = توکی_ct1.get_local_range(2);
    ثابت صحيح sample  = توکی_ct1.get_ډله(0);
    ثابت صحيح channel = توکی_ct1.get_ډله(1);
    ثابت صحيح لیکه     = توکی_ct1.get_ډله(2);

    ثابت صحيح tid = توکی_ct1.get_local_id(2);
    ثابت صحيح nwarps = nthreads / WARP_اندازه;

    ثابت خپلکاری src_دننه_واټن = calculate_دننه_واټن<3>({src_stride_sample, src_stride_channel, src_stride_لیکه}, {sample, channel, لیکه});
    ثابت خپلکاری dst_دننه_واټن = calculate_دننه_واټن<3>({dst_stride_sample, dst_stride_channel, dst_stride_لیکه}, {sample, channel, لیکه});

    x += src_دننه_واټن;
    dst += dst_دننه_واټن;

    sycl::float2 منځنۍ_var = sycl::float2(0.f, 0.f);

    لكل (صحيح ستون = tid; ستون < ncols; ستون += بلاک_اندازه_3) {
        ثابت عائم xi = x[ستون * src_stride_ستون];
        منځنۍ_var.x() += xi;
        منځنۍ_var.y() += xi * xi;
    }

    // sum up partial sums
    منځنۍ_var = warp_reduce_مجموعه(منځنۍ_var, توکی_ct1);
    إذا  (بلاک_اندازه_3 > WARP_اندازه) {
        ثابت خپلکاری sub_ډله = توکی_ct1.get_sub_ډله();
        ثابت خپلکاری sg_id = sub_ډله.get_ډله_خطي_id();
        ثابت خپلکاری wi_دننه_sg = sub_ډله.get_local_خطي_id();
        إذا (wi_دننه_sg == 0) {
            s_مجموعه[sg_id] = منځنۍ_var;
        }
        توکی_ct1.barrier(sycl::access::fence_space::local_space);
        منځنۍ_var = 0.f;
        ثابت size_t nreduce = ceil_div(nwarps, WARP_اندازه);
        لكل (size_t i = 0; i < nreduce; i += 1)
        {
            منځنۍ_var += s_مجموعه[wi_دننه_sg + i * WARP_اندازه];
        }
        منځنۍ_var = warp_reduce_مجموعه(منځنۍ_var, توکی_ct1);
    }

    ثابت عائم منځنۍ_2 = منځنۍ_var.x() / ncols;
    ثابت عائم var = منځنۍ_var.y() / ncols - منځنۍ_2 * منځنۍ_2;
    ثابت عائم inv_std = sycl::rsqrt(var + eps);

    لكل (صحيح ستون = tid; ستون < ncols; ستون += بلاک_اندازه_3) {
        dst[ستون * dst_stride_ستون] = (x[ستون * src_stride_ستون] - منځنۍ_2) * inv_std;
    }
}

ساكن فراغ ډله_norm_f32(ثابت عائم* x, عائم* dst, ثابت صحيح ډله_اندازه, ثابت صحيح ne_elements, ثابت عائم eps,
    ثابت sycl::nd_توکی<3>& توکی_ct1, عائم* s_مجموعه, صحيح بلاک_اندازه_3) {
    صحيح start = توکی_ct1.get_ډله(2) * ډله_اندازه;
    صحيح end = start + ډله_اندازه;
    ثابت صحيح nthreads = توکی_ct1.get_local_range(2);
    ثابت صحيح nwarps = nthreads / WARP_اندازه;
    start += توکی_ct1.get_local_id(2);
    size_t nreduce = nwarps / WARP_اندازه;

    إذا (end >= ne_elements) {
        end = ne_elements;
    }

    عائم tmp = 0.0f; // partial sum for thread in warp

    لكل (صحيح j = start; j < end; j += بلاک_اندازه_3) {
        tmp += x[j];
    }

    tmp = warp_reduce_مجموعه(tmp, توکی_ct1);
    إذا (بلاک_اندازه_3 > WARP_اندازه) {

        صحيح warp_id = توکی_ct1.get_local_id(2) / WARP_اندازه;
        صحيح lane_id = توکی_ct1.get_local_id(2) % WARP_اندازه;
        إذا (lane_id == 0) {
            s_مجموعه[warp_id] = tmp;
        }
        /*
        DPCT1118:1: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:54: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        توکی_ct1.barrier();
        tmp = 0.f;
        لكل (size_t i = 0; i < nreduce; i += 1)
        {
            tmp += s_مجموعه[lane_id + i * WARP_اندازه];
        }
        tmp = warp_reduce_مجموعه(tmp, توکی_ct1);
    }

    عائم منځنۍ_2 = tmp / ډله_اندازه;
    tmp = 0.0f;

    لكل (صحيح j = start; j < end; j += بلاک_اندازه_3) {
        عائم xi = x[j] - منځنۍ_2;
        dst[j] = xi;
        tmp += xi * xi;
    }

    tmp = warp_reduce_مجموعه(tmp, توکی_ct1);
    إذا (بلاک_اندازه_3 > WARP_اندازه) {

        صحيح warp_id = توکی_ct1.get_local_id(2) / WARP_اندازه;
        صحيح lane_id = توکی_ct1.get_local_id(2) % WARP_اندازه;
        إذا (lane_id == 0) {
            s_مجموعه[warp_id] = tmp;
        }
        /*
        DPCT1118:2: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:55: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        توکی_ct1.barrier();
        tmp = 0.f;
        لكل (size_t i = 0; i < nreduce; i += 1)
        {
            tmp += s_مجموعه[lane_id + i * WARP_اندازه];
        }
        tmp = warp_reduce_مجموعه(tmp, توکی_ct1);
    }

    عائم variance = tmp / ډله_اندازه;
    عائم scale = sycl::rsqrt(variance + eps);
    لكل (صحيح j = start; j < end; j += بلاک_اندازه_3) {
        dst[j] *= scale;
    }
}

کينډۍ <منطقي do_multiply = خطأ, منطقي do_جمع_کړه = خطأ>
ساكن فراغ rms_norm_f32(ثابت عائم* x, عائم* dst, ثابت صحيح ncols,
    ثابت int64_t src_stride_ستون, ثابت int64_t src_stride_لیکه, ثابت int64_t src_stride_channel, ثابت int64_t src_stride_sample,
    ثابت int64_t dst_stride_ستون, ثابت int64_t dst_stride_لیکه, ثابت int64_t dst_stride_channel, ثابت int64_t dst_stride_sample,
    ثابت عائم eps, ثابت sycl::nd_توکی<3>& توکی_ct1, عائم* s_مجموعه, صحيح بلاک_اندازه_3,
    ثابت عائم* mul = تش_اشاره, ثابت int64_t mul_stride_لیکه = 0, ثابت int64_t mul_stride_channel = 0,
    ثابت int64_t mul_stride_sample = 0, ثابت صحيح mul_nrows = 0, ثابت صحيح mul_nchannels = 0, ثابت صحيح mul_nsamples = 0,
    ثابت عائم* جمع_کړه_2 = تش_اشاره, ثابت int64_t جمع_کړه_stride_لیکه = 0, ثابت int64_t جمع_کړه_stride_channel = 0,
    ثابت int64_t جمع_کړه_stride_sample = 0, ثابت صحيح جمع_کړه_nrows = 0, ثابت صحيح جمع_کړه_nchannels = 0, ثابت صحيح جمع_کړه_nsamples = 0,
    ثابت عائم scale_mul = 1.0f) {

    static_assert(!do_جمع_کړه || do_multiply, "fusing add is not supported without multiplying");

    ثابت صحيح sample  = توکی_ct1.get_ډله(0);
    ثابت صحيح channel = توکی_ct1.get_ډله(1);
    ثابت صحيح لیکه     = توکی_ct1.get_ډله(2);

    ثابت صحيح nthreads = توکی_ct1.get_local_range(2);

    ثابت صحيح tid = توکی_ct1.get_local_id(2);
    ثابت صحيح nwarps = nthreads / WARP_اندازه;

    ثابت خپلکاری src_دننه_واټن = calculate_دننه_واټن<3>({src_stride_sample, src_stride_channel, src_stride_لیکه}, {sample, channel, لیکه});
    ثابت خپلکاری dst_دننه_واټن = calculate_دننه_واټن<3>({dst_stride_sample, dst_stride_channel, dst_stride_لیکه}, {sample, channel, لیکه});

    x   += src_دننه_واټن;
    dst += dst_دننه_واټن;

    إذا constexpr (do_multiply) {
        ثابت صحيح mul_لیکه     = لیکه     % mul_nrows;
        ثابت صحيح mul_channel = channel % mul_nchannels;
        ثابت صحيح mul_sample  = sample  % mul_nsamples;
        mul += mul_sample * mul_stride_sample + mul_channel * mul_stride_channel + mul_لیکه * mul_stride_لیکه;
    }

    إذا constexpr (do_جمع_کړه) {
        ثابت صحيح جمع_کړه_لیکه     = لیکه     % جمع_کړه_nrows;
        ثابت صحيح جمع_کړه_channel = channel % جمع_کړه_nchannels;
        ثابت صحيح جمع_کړه_sample  = sample  % جمع_کړه_nsamples;
        جمع_کړه_2 += جمع_کړه_sample * جمع_کړه_stride_sample + جمع_کړه_channel * جمع_کړه_stride_channel + جمع_کړه_لیکه * جمع_کړه_stride_لیکه;
    }

    عائم tmp = 0.0f; // partial sum for thread in warp

    لكل (صحيح ستون = tid; ستون < ncols; ستون += بلاک_اندازه_3) {
        ثابت عائم xi = x[ستون * src_stride_ستون];
        tmp += xi * xi;
    }

    // sum up partial sums
    tmp = warp_reduce_مجموعه(tmp, توکی_ct1);
    إذا (بلاک_اندازه_3 > WARP_اندازه) {
        ثابت خپلکاری sub_ډله = توکی_ct1.get_sub_ډله();
        ثابت خپلکاری sg_id = sub_ډله.get_ډله_خطي_id();
        ثابت خپلکاری wi_دننه_sg = sub_ډله.get_local_خطي_id();
        إذا (wi_دننه_sg == 0) {
            s_مجموعه[sg_id] = tmp;
        }

        توکی_ct1.barrier(sycl::access::fence_space::local_space);
        ثابت size_t nreduce = ceil_div(nwarps, WARP_اندازه);
        tmp = 0.f;
        لكل (size_t i = 0; i < nreduce; i += 1)
        {
            tmp += s_مجموعه[wi_دننه_sg + i * WARP_اندازه];
        }
        tmp = warp_reduce_مجموعه(tmp, توکی_ct1);
    }

    ثابت عائم منځنۍ_2 = tmp / ncols;
    ثابت عائم scale = sycl::rsqrt(منځنۍ_2 + eps);

    لكل (صحيح ستون = tid; ستون < ncols; ستون += بلاک_اندازه_3) {
        إذا constexpr (do_multiply && do_جمع_کړه) {
            dst[ستون * dst_stride_ستون] = scale * x[ستون * src_stride_ستون] * mul[ستون] + جمع_کړه_2[ستون];
        } وإلا إذا constexpr (do_multiply) {
            dst[ستون * dst_stride_ستون] = scale * x[ستون * src_stride_ستون] * mul[ستون];
        } وإلا {
            // folded epilogue of a fused GGML_OP_SCALE consumer (qwen35 GDN l2 norms);
            // the explicit temporary keeps the float evaluation order identical to
            // running rms_norm and scale as two separate kernels
            ثابت عائم v = scale * x[ستون * src_stride_ستون];
            dst[ستون * dst_stride_ستون] = v * scale_mul;
        }
    }
}

کينډۍ<صحيح warp_اندازه>
ساكن فراغ l2_norm_f32(ثابت عائم * x, عائم * dst, ثابت صحيح ncols,
    ثابت int64_t src_stride_ستون, ثابت int64_t src_stride_لیکه, ثابت int64_t src_stride_channel,
    ثابت int64_t src_stride_sample, ثابت int64_t dst_stride_ستون, ثابت int64_t dst_stride_لیکه,
    ثابت int64_t dst_stride_channel, ثابت int64_t dst_stride_sample, ثابت عائم eps,
    ثابت sycl::nd_توکی<3>& توکی_ct1, عائم* s_مجموعه, ثابت صحيح بلاک_اندازه_3) {

    ثابت صحيح لیکه     = توکی_ct1.get_ډله(2);
    ثابت صحيح channel = توکی_ct1.get_ډله(1);
    ثابت صحيح sample  = توکی_ct1.get_ډله(0);
    ثابت صحيح tid     = توکی_ct1.get_local_id(2);

    x   += sample*src_stride_sample + channel*src_stride_channel + لیکه*src_stride_لیکه;
    dst += sample*dst_stride_sample + channel*dst_stride_channel + لیکه*dst_stride_لیکه;

    عائم tmp = 0.0f; // partial sum for thread in warp

    لكل (صحيح ستون = tid; ستون < ncols; ستون += بلاک_اندازه_3) {
        ثابت عائم xi = x[ستون * src_stride_ستون];
        tmp += xi * xi;
    }

    tmp = بلاک_reduce<بلاک_reduce_لاره::مجموعه_3, warp_اندازه>(tmp, s_مجموعه, بلاک_اندازه_3);
    ثابت عائم scale = sycl::rsqrt(sycl::fmax(tmp, eps * eps));

    لكل (صحيح ستون = tid; ستون < ncols; ستون += بلاک_اندازه_3) {
        dst[ستون * dst_stride_ستون] = scale * x[ستون * src_stride_ستون];
    }
}

ساكن فراغ norm_f32_sycl(ثابت عائم * x, عائم * dst, ثابت صحيح ncols, ثابت صحيح nrows, ثابت صحيح nchannels, ثابت صحيح nsamples,
    ثابت int64_t src_stride_ستون, ثابت int64_t src_stride_لیکه, ثابت int64_t src_stride_channel, ثابت int64_t src_stride_sample,
    ثابت int64_t dst_stride_ستون, ثابت int64_t dst_stride_لیکه, ثابت int64_t dst_stride_channel, ثابت int64_t dst_stride_sample,
        ثابت عائم eps, کتار_ptr stream, صحيح device) {

    ثابت sycl::range<3> global_dims(nsamples, nchannels, nrows);
    إذا (ncols < 1024) {
        ثابت sycl::range<3> بلاک_dims(1, 1, WARP_اندازه);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * بلاک_dims, بلاک_dims),
                [=](sycl::nd_توکی<3> توکی_ct1)
                [[sycl::reqd_sub_ډله_اندازه(WARP_اندازه)]] {
                    norm_f32(x, dst, ncols,
                        src_stride_ستون, src_stride_لیکه, src_stride_channel, src_stride_sample,
                        dst_stride_ستون, dst_stride_لیکه, dst_stride_channel, dst_stride_sample,
                        eps, توکی_ct1, تش_اشاره, WARP_اندازه);
                });
            });
    }
    وإلا {
        ثابت صحيح work_ډله_اندازه = ggml_sycl_info().max_work_ډله_sizes[device];
        أكد(work_ډله_اندازه % (WARP_اندازه * WARP_اندازه) == 0);
        ثابت sycl::range<3> بلاک_dims(1, 1, work_ډله_اندازه);
        /*
        DPCT1049:17: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<sycl::float2, 1> s_مجموعه_acc_ct1(
                            sycl::range<1>(work_ډله_اندازه / WARP_اندازه), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * بلاک_dims, بلاک_dims),
                [=](sycl::nd_توکی<3> توکی_ct1)
                [[sycl::reqd_sub_ډله_اندازه(WARP_اندازه)]] {
                    norm_f32(x, dst, ncols,
                        src_stride_ستون, src_stride_لیکه, src_stride_channel, src_stride_sample,
                        dst_stride_ستون, dst_stride_لیکه, dst_stride_channel, dst_stride_sample,
                        eps, توکی_ct1, get_اشاره_ګر(s_مجموعه_acc_ct1), work_ډله_اندازه);
                });
            });
    }
}

ساكن فراغ ډله_norm_f32_sycl(ثابت عائم* x, عائم* dst,
    ثابت صحيح عدد_groups, ثابت عائم eps, ثابت صحيح ډله_اندازه,
    ثابت صحيح ne_elements, کتار_ptr stream, صحيح device) {
    إذا (ډله_اندازه < 1024) {
        ثابت sycl::range<3> بلاک_dims(1, 1, WARP_اندازه);
        stream->submit([&](sycl::handler& cgh) {
            ثابت عائم eps_ct4 = eps;
            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, عدد_groups) * بلاک_dims,
                    بلاک_dims),
                [=](sycl::nd_توکی<3> توکی_ct1)
                [[sycl::reqd_sub_ډله_اندازه(WARP_اندازه)]] {
                    ډله_norm_f32(
                        x, dst, ډله_اندازه, ne_elements, eps_ct4, توکی_ct1,
                        تش_اشاره, WARP_اندازه);
                });
            });
    }
    وإلا {
        ثابت صحيح work_ډله_اندازه = ggml_sycl_info().max_work_ډله_sizes[device];
        أكد(work_ډله_اندازه % (WARP_اندازه * WARP_اندازه) == 0);
        ثابت sycl::range<3> بلاک_dims(1, 1, work_ډله_اندازه);
        /*
        DPCT1049:18: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */

        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<عائم, 1> s_مجموعه_acc_ct1(sycl::range<1>(work_ډله_اندازه / WARP_اندازه),
                cgh);

            ثابت عائم eps_ct4 = eps;

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, عدد_groups) * بلاک_dims,
                    بلاک_dims),
                [=](sycl::nd_توکی<3> توکی_ct1)
                [[sycl::reqd_sub_ډله_اندازه(WARP_اندازه)]] {
                    ډله_norm_f32(x, dst, ډله_اندازه, ne_elements,
                        eps_ct4, توکی_ct1,
                        get_اشاره_ګر(s_مجموعه_acc_ct1), work_ډله_اندازه);
                });
            });
    }
}

ساكن فراغ rms_norm_f32_sycl(ثابت عائم* x, عائم* dst, ثابت صحيح ncols, ثابت صحيح nrows, ثابت صحيح nchannels, ثابت صحيح nsamples,
    ثابت int64_t src_stride_ستون, ثابت int64_t src_stride_لیکه, ثابت int64_t src_stride_channel, ثابت int64_t src_stride_sample,
    ثابت int64_t dst_stride_ستون, ثابت int64_t dst_stride_لیکه, ثابت int64_t dst_stride_channel, ثابت int64_t dst_stride_sample,
    ثابت عائم eps, کتار_ptr stream, صحيح device) {
    // printf("%s ncols=%d, nrows=%d, WARP_SIZE=%d\n", __func__, ncols, nrows, WARP_SIZE);

    ثابت sycl::range<3> global_dims(nsamples, nchannels, nrows);
    إذا (ncols < 1024) {
        ثابت sycl::range<3> بلاک_dims(1, 1, WARP_اندازه);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * بلاک_dims, بلاک_dims),
                [=](sycl::nd_توکی<3> توکی_ct1)
                [[sycl::reqd_sub_ډله_اندازه(WARP_اندازه)]] {
                    rms_norm_f32(x, dst, ncols,
                        src_stride_ستون, src_stride_لیکه, src_stride_channel, src_stride_sample,
                        dst_stride_ستون, dst_stride_لیکه, dst_stride_channel, dst_stride_sample,
                        eps, توکی_ct1, تش_اشاره, WARP_اندازه);
                });
            });
    }
    وإلا {
        ثابت صحيح work_ډله_اندازه = ggml_sycl_info().max_work_ډله_sizes[device];
        أكد(work_ډله_اندازه % (WARP_اندازه * WARP_اندازه) == 0);
        ثابت sycl::range<3> بلاک_dims(1, 1, work_ډله_اندازه);
        /*
        DPCT1049:19: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<عائم, 1> s_مجموعه_acc_ct1(sycl::range<1>(work_ډله_اندازه / WARP_اندازه),
                cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * بلاک_dims, بلاک_dims),
                [=](sycl::nd_توکی<3> توکی_ct1)
                [[sycl::reqd_sub_ډله_اندازه(WARP_اندازه)]] {
                    rms_norm_f32(x, dst, ncols,
                        src_stride_ستون, src_stride_لیکه, src_stride_channel, src_stride_sample,
                        dst_stride_ستون, dst_stride_لیکه, dst_stride_channel, dst_stride_sample,
                        eps, توکی_ct1, get_اشاره_ګر(s_مجموعه_acc_ct1), work_ډله_اندازه);
                });
            });
    }
}

ساكن فراغ rms_norm_scale_f32_sycl(ثابت عائم* x, عائم* dst, ثابت صحيح ncols, ثابت صحيح nrows,
    ثابت صحيح nchannels, ثابت صحيح nsamples,
    ثابت int64_t src_stride_ستون, ثابت int64_t src_stride_لیکه, ثابت int64_t src_stride_channel, ثابت int64_t src_stride_sample,
    ثابت int64_t dst_stride_ستون, ثابت int64_t dst_stride_لیکه, ثابت int64_t dst_stride_channel, ثابت int64_t dst_stride_sample,
    ثابت عائم eps, ثابت عائم scale_mul, کتار_ptr stream, صحيح device) {
    ثابت sycl::range<3> global_dims(nsamples, nchannels, nrows);
    إذا (ncols < 1024) {
        ثابت sycl::range<3> بلاک_dims(1, 1, WARP_اندازه);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * بلاک_dims, بلاک_dims),
                [=](sycl::nd_توکی<3> توکی_ct1)
                [[sycl::reqd_sub_ډله_اندازه(WARP_اندازه)]] {
                    rms_norm_f32(x, dst, ncols,
                        src_stride_ستون, src_stride_لیکه, src_stride_channel, src_stride_sample,
                        dst_stride_ستون, dst_stride_لیکه, dst_stride_channel, dst_stride_sample,
                        eps, توکی_ct1, تش_اشاره, WARP_اندازه,
                        تش_اشاره, 0, 0, 0, 0, 0, 0,
                        تش_اشاره, 0, 0, 0, 0, 0, 0,
                        scale_mul);
                });
            });
    }
    وإلا {
        ثابت صحيح work_ډله_اندازه = ggml_sycl_info().max_work_ډله_sizes[device];
        أكد(work_ډله_اندازه % (WARP_اندازه * WARP_اندازه) == 0);
        ثابت sycl::range<3> بلاک_dims(1, 1, work_ډله_اندازه);
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<عائم, 1> s_مجموعه_acc_ct1(sycl::range<1>(work_ډله_اندازه / WARP_اندازه),
                cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * بلاک_dims, بلاک_dims),
                [=](sycl::nd_توکی<3> توکی_ct1)
                [[sycl::reqd_sub_ډله_اندازه(WARP_اندازه)]] {
                    rms_norm_f32(x, dst, ncols,
                        src_stride_ستون, src_stride_لیکه, src_stride_channel, src_stride_sample,
                        dst_stride_ستون, dst_stride_لیکه, dst_stride_channel, dst_stride_sample,
                        eps, توکی_ct1, get_اشاره_ګر(s_مجموعه_acc_ct1), work_ډله_اندازه,
                        تش_اشاره, 0, 0, 0, 0, 0, 0,
                        تش_اشاره, 0, 0, 0, 0, 0, 0,
                        scale_mul);
                });
            });
    }
}

ساكن فراغ rms_norm_mul_f32_sycl(ثابت عائم* x, ثابت عائم* mul, عائم* dst, ثابت صحيح ncols, ثابت صحيح nrows,
        ثابت صحيح nchannels, ثابت صحيح nsamples,
        ثابت int64_t src_stride_ستون, ثابت int64_t src_stride_لیکه, ثابت int64_t src_stride_channel, ثابت int64_t src_stride_sample,
        ثابت int64_t dst_stride_ستون, ثابت int64_t dst_stride_لیکه, ثابت int64_t dst_stride_channel, ثابت int64_t dst_stride_sample,
        ثابت int64_t mul_stride_لیکه, ثابت int64_t mul_stride_channel, ثابت int64_t mul_stride_sample,
        ثابت صحيح mul_nrows, ثابت صحيح mul_nchannels, ثابت صحيح mul_nsamples,
        ثابت عائم eps, کتار_ptr stream, صحيح device) {
    ثابت sycl::range<3> global_dims(nsamples, nchannels, nrows);
    إذا (ncols < 1024) {
        ثابت sycl::range<3> بلاک_dims(1, 1, WARP_اندازه);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * بلاک_dims, بلاک_dims),
                [=](sycl::nd_توکی<3> توکی_ct1)
                [[sycl::reqd_sub_ډله_اندازه(WARP_اندازه)]] {
                    rms_norm_f32<ریښتیا>(x, dst, ncols,
                        src_stride_ستون, src_stride_لیکه, src_stride_channel, src_stride_sample,
                        dst_stride_ستون, dst_stride_لیکه, dst_stride_channel, dst_stride_sample,
                        eps, توکی_ct1, تش_اشاره, WARP_اندازه,
                        mul, mul_stride_لیکه, mul_stride_channel, mul_stride_sample, mul_nrows, mul_nchannels, mul_nsamples);
                });
            });
    }
    وإلا {
        ثابت صحيح work_ډله_اندازه = ggml_sycl_info().max_work_ډله_sizes[device];
        أكد(work_ډله_اندازه % (WARP_اندازه * WARP_اندازه) == 0);
        ثابت sycl::range<3> بلاک_dims(1, 1, work_ډله_اندازه);
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<عائم, 1> s_مجموعه_acc_ct1(sycl::range<1>(work_ډله_اندازه / WARP_اندازه), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * بلاک_dims, بلاک_dims),
                [=](sycl::nd_توکی<3> توکی_ct1)
                [[sycl::reqd_sub_ډله_اندازه(WARP_اندازه)]] {
                    rms_norm_f32<ریښتیا>(x, dst, ncols,
                        src_stride_ستون, src_stride_لیکه, src_stride_channel, src_stride_sample,
                        dst_stride_ستون, dst_stride_لیکه, dst_stride_channel, dst_stride_sample,
                        eps, توکی_ct1, get_اشاره_ګر(s_مجموعه_acc_ct1), work_ډله_اندازه,
                        mul, mul_stride_لیکه, mul_stride_channel, mul_stride_sample, mul_nrows, mul_nchannels, mul_nsamples);
                });
            });
    }
}

ساكن فراغ rms_norm_mul_جمع_کړه_f32_sycl(ثابت عائم* x, ثابت عائم* mul, ثابت عائم* جمع_کړه_2, عائم* dst,
        ثابت صحيح ncols, ثابت صحيح nrows, ثابت صحيح nchannels, ثابت صحيح nsamples,
        ثابت int64_t src_stride_ستون, ثابت int64_t src_stride_لیکه, ثابت int64_t src_stride_channel, ثابت int64_t src_stride_sample,
        ثابت int64_t dst_stride_ستون, ثابت int64_t dst_stride_لیکه, ثابت int64_t dst_stride_channel, ثابت int64_t dst_stride_sample,
        ثابت int64_t mul_stride_لیکه, ثابت int64_t mul_stride_channel, ثابت int64_t mul_stride_sample,
        ثابت صحيح mul_nrows, ثابت صحيح mul_nchannels, ثابت صحيح mul_nsamples,
        ثابت int64_t جمع_کړه_stride_لیکه, ثابت int64_t جمع_کړه_stride_channel, ثابت int64_t جمع_کړه_stride_sample,
        ثابت صحيح جمع_کړه_nrows, ثابت صحيح جمع_کړه_nchannels, ثابت صحيح جمع_کړه_nsamples,
        ثابت عائم eps, کتار_ptr stream, صحيح device) {
    ثابت sycl::range<3> global_dims(nsamples, nchannels, nrows);
    إذا (ncols < 1024) {
        ثابت sycl::range<3> بلاک_dims(1, 1, WARP_اندازه);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * بلاک_dims, بلاک_dims),
                [=](sycl::nd_توکی<3> توکی_ct1)
                [[sycl::reqd_sub_ډله_اندازه(WARP_اندازه)]] {
                    rms_norm_f32<ریښتیا, ریښتیا>(x, dst, ncols,
                        src_stride_ستون, src_stride_لیکه, src_stride_channel, src_stride_sample,
                        dst_stride_ستون, dst_stride_لیکه, dst_stride_channel, dst_stride_sample,
                        eps, توکی_ct1, تش_اشاره, WARP_اندازه,
                        mul, mul_stride_لیکه, mul_stride_channel, mul_stride_sample, mul_nrows, mul_nchannels, mul_nsamples,
                        جمع_کړه_2, جمع_کړه_stride_لیکه, جمع_کړه_stride_channel, جمع_کړه_stride_sample, جمع_کړه_nrows, جمع_کړه_nchannels, جمع_کړه_nsamples);
                });
            });
    }
    وإلا {
        ثابت صحيح work_ډله_اندازه = ggml_sycl_info().max_work_ډله_sizes[device];
        أكد(work_ډله_اندازه % (WARP_اندازه * WARP_اندازه) == 0);
        ثابت sycl::range<3> بلاک_dims(1, 1, work_ډله_اندازه);
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<عائم, 1> s_مجموعه_acc_ct1(sycl::range<1>(work_ډله_اندازه / WARP_اندازه), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * بلاک_dims, بلاک_dims),
                [=](sycl::nd_توکی<3> توکی_ct1)
                [[sycl::reqd_sub_ډله_اندازه(WARP_اندازه)]] {
                    rms_norm_f32<ریښتیا, ریښتیا>(x, dst, ncols,
                        src_stride_ستون, src_stride_لیکه, src_stride_channel, src_stride_sample,
                        dst_stride_ستون, dst_stride_لیکه, dst_stride_channel, dst_stride_sample,
                        eps, توکی_ct1, get_اشاره_ګر(s_مجموعه_acc_ct1), work_ډله_اندازه,
                        mul, mul_stride_لیکه, mul_stride_channel, mul_stride_sample, mul_nrows, mul_nchannels, mul_nsamples,
                        جمع_کړه_2, جمع_کړه_stride_لیکه, جمع_کړه_stride_channel, جمع_کړه_stride_sample, جمع_کړه_nrows, جمع_کړه_nchannels, جمع_کړه_nsamples);
                });
            });
    }
}

کينډۍ<صحيح warp_اندازه>
ساكن فراغ l2_norm_f32_sycl(ثابت عائم *   x,
                             عائم *         dst,
                             ثابت صحيح       ncols,
                             ثابت صحيح       nrows,
                             ثابت صحيح       nchannels,
                             ثابت صحيح       nsamples,
                             ثابت int64_t   src_stride_ستون,
                             ثابت int64_t   src_stride_لیکه,
                             ثابت int64_t   src_stride_channel,
                             ثابت int64_t   src_stride_sample,
                             ثابت int64_t   dst_stride_ستون,
                             ثابت int64_t   dst_stride_لیکه,
                             ثابت int64_t   dst_stride_channel,
                             ثابت int64_t   dst_stride_sample,
                             ثابت عائم     eps,
                             کتار_ptr       stream,
                             صحيح             device) {
    ثابت dpct::dim3 بلاکونه_عدد(nrows, nchannels, nsamples);

    إذا (ncols < 1024) {
        ثابت dpct::dim3 بلاک_dims(warp_اندازه, 1, 1);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(بلاکونه_عدد * بلاک_dims,
                    بلاک_dims),
                [=](sycl::nd_توکی<3> توکی_ct1)
                [[sycl::reqd_sub_ډله_اندازه(warp_اندازه)]] {
                    l2_norm_f32<warp_اندازه>(x, dst, ncols,
                        src_stride_ستون, src_stride_لیکه, src_stride_channel, src_stride_sample,
                        dst_stride_ستون, dst_stride_لیکه, dst_stride_channel, dst_stride_sample,
                        eps, توکی_ct1,
                        تش_اشاره, warp_اندازه);
                });
            });
    }
    وإلا {
        ثابت صحيح work_ډله_اندازه = ggml_sycl_info().max_work_ډله_sizes[device];
        أكد(work_ډله_اندازه % (warp_اندازه * warp_اندازه) == 0);
        ثابت sycl::range<3> بلاک_dims(1, 1, work_ډله_اندازه);
        صحيح lsm_اندازه =  بلاک_dims[2] > warp_اندازه ? work_ډله_اندازه / warp_اندازه * حجم(عائم): 0;
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<عائم, 1> s_مجموعه_acc_ct1(sycl::range<1>(lsm_اندازه),
                cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(بلاکونه_عدد * بلاک_dims,
                    بلاک_dims),
                [=](sycl::nd_توکی<3> توکی_ct1)
                [[sycl::reqd_sub_ډله_اندازه(warp_اندازه)]] {
                    l2_norm_f32<warp_اندازه>(x, dst, ncols,
                        src_stride_ستون, src_stride_لیکه, src_stride_channel, src_stride_sample,
                        dst_stride_ستون, dst_stride_لیکه, dst_stride_channel, dst_stride_sample,
                        eps, توکی_ct1, get_اشاره_ګر(s_مجموعه_acc_ct1), work_ډله_اندازه);
                });
            });
    }
}

// Batched L2 norm: N independent same-shape F32 tensors in one launch; the tensor
// index is folded into grid dim0 and each row's reduction is identical to the
// single-tensor kernel, so the result is bit-exact.
بنية l2_batch_ptrs {
    ثابت عائم * src[GGML_SYCL_L2_BATCH_MAX];
    عائم *       dst[GGML_SYCL_L2_BATCH_MAX];
};

// One stride set shared by the whole batch: the caller only groups tensors whose nb[]
// all match, so per-tensor state stays two pointers.
بنية l2_batch_strides {
    صحيح     ne1, ne2;
    int64_t ss0, ss1, ss2, ss3;
    int64_t ds0, ds1, ds2, ds3;
};

کينډۍ <صحيح warp_اندازه>
ساكن فراغ l2_norm_f32_batch(l2_batch_ptrs p, l2_batch_strides st, ثابت صحيح ncols, ثابت عائم eps,
                              ثابت sycl::nd_توکی<3> & توکی_ct1) {
    ثابت صحيح t   = توکی_ct1.get_ډله(0);  // tensor index
    ثابت صحيح r   = توکی_ct1.get_ډله(2);  // flattened row over ne1*ne2*ne3
    ثابت صحيح tid = توکی_ct1.get_local_id(2);

    ثابت صحيح i1 = r % st.ne1;
    ثابت صحيح i2 = (r / st.ne1) % st.ne2;
    ثابت صحيح i3 = r / (st.ne1 * st.ne2);

    ثابت عائم * x   = p.src[t] + i3 * st.ss3 + i2 * st.ss2 + i1 * st.ss1;
    عائم *       dst = p.dst[t] + i3 * st.ds3 + i2 * st.ds2 + i1 * st.ds1;

    عائم tmp = 0.0f;
    لكل (صحيح ستون = tid; ستون < ncols; ستون += warp_اندازه) {
        ثابت عائم xi = x[ستون * st.ss0];
        tmp += xi * xi;
    }
    tmp = بلاک_reduce<بلاک_reduce_لاره::مجموعه_3, warp_اندازه>(tmp, (عائم *) تش_اشاره, warp_اندازه);
    ثابت عائم scale = sycl::rsqrt(sycl::fmax(tmp, eps * eps));
    لكل (صحيح ستون = tid; ستون < ncols; ستون += warp_اندازه) {
        dst[ستون * st.ds0] = scale * x[ستون * st.ss0];
    }
}

کينډۍ <صحيح warp_اندازه>
ساكن فراغ l2_norm_f32_batch_sycl(l2_batch_ptrs p, l2_batch_strides st, ثابت صحيح n_tensors,
                                   ثابت صحيح ncols, ثابت صحيح nrows_ټول, ثابت عائم eps,
                                   کتار_ptr stream) {
    ثابت dpct::dim3 بلاکونه_عدد(nrows_ټول, 1, n_tensors);
    ثابت dpct::dim3 بلاک_dims(warp_اندازه, 1, 1);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(بلاکونه_عدد * بلاک_dims, بلاک_dims),
            [=](sycl::nd_توکی<3> توکی_ct1) [[sycl::reqd_sub_ډله_اندازه(warp_اندازه)]] {
                l2_norm_f32_batch<warp_اندازه>(p, st, ncols, eps, توکی_ct1);
            });
    });
}

فراغ ggml_sycl_op_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    ثابت ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS
    dpct::کتار_ptr main_بهیر = ctx.stream();
    SYCL_CHECK(ggml_sycl_سېټ_device(ctx.device));
    ثابت عائم * src0_dd = static_cast<ثابت عائم *>(dst->src[0]->معلومات);
    عائم *       dst_dd  = static_cast<عائم *>(dst->معلومات);

    عائم eps;
    حافظه_نقل(&eps, dst->op_params, حجم(عائم));
    GGML_ASSERT(eps >= 0.0f);
    ثابت size_t ts0 = ggml_type_اندازه(src0->type);
    ثابت size_t tdst = ggml_type_اندازه(dst->type);
    GGML_ASSERT(nb00 % ts0 == 0 && nb01 % ts0 == 0 && nb02 % ts0 == 0 && nb03 % ts0 == 0);
    GGML_ASSERT(nb0 % tdst == 0 && nb1 % tdst == 0 && nb2 % tdst == 0 && nb3 % tdst == 0);
    ثابت int64_t ss0 = nb00 / ts0;
    ثابت int64_t ss1 = nb01 / ts0;
    ثابت int64_t ss2 = nb02 / ts0;
    ثابت int64_t ss3 = nb03 / ts0;
    ثابت int64_t ds0 = nb0 / tdst;
    ثابت int64_t ds1 = nb1 / tdst;
    ثابت int64_t ds2 = nb2 / tdst;
    ثابت int64_t ds3 = nb3 / tdst;

    norm_f32_sycl(src0_dd, dst_dd, ne00, ne01, ne02, ne03,
        ss0, ss1, ss2, ss3, ds0, ds1, ds2, ds3, eps, main_بهیر, ctx.device);
}

فراغ ggml_sycl_op_ډله_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    صحيح عدد_groups = dst->op_params[0];
    dpct::کتار_ptr main_بهیر = ctx.stream();
    SYCL_CHECK(ggml_sycl_سېټ_device(ctx.device));

    ثابت عائم * src0_dd = static_cast<ثابت عائم *>(dst->src[0]->معلومات);
    عائم *       dst_dd  = static_cast<عائم *>(dst->معلومات);

    عائم eps;
    حافظه_نقل(&eps, dst->op_params + 1, حجم(عائم));

    صحيح ډله_اندازه = dst->src[0]->ne[0] * dst->src[0]->ne[1] * ((dst->src[0]->ne[2] + عدد_groups - 1) / عدد_groups);
    ډله_norm_f32_sycl(src0_dd, dst_dd, عدد_groups, eps, ډله_اندازه, dst->src[0]->ne[0] * dst->src[0]->ne[1] * dst->src[0]->ne[2], main_بهیر, ctx.device);
}

فراغ ggml_sycl_op_rms_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    dpct::کتار_ptr main_بهیر = ctx.stream();
    SYCL_CHECK(ggml_sycl_سېټ_device(ctx.device));

    ثابت عائم * src0_dd = static_cast<ثابت عائم *>(dst->src[0]->معلومات);
    عائم *       dst_dd  = static_cast<عائم *>(dst->معلومات);

    عائم eps;
    حافظه_نقل(&eps, dst->op_params, حجم(عائم));

    GGML_TENSOR_UNARY_OP_LOCALS
    ثابت size_t ts0 = ggml_type_اندازه(src0->type);
    ثابت size_t tdst = ggml_type_اندازه(dst->type);
    GGML_ASSERT(nb00 % ts0 == 0 && nb01 % ts0 == 0 && nb02 % ts0 == 0 && nb03 % ts0 == 0);
    GGML_ASSERT(nb0 % tdst == 0 && nb1 % tdst == 0 && nb2 % tdst == 0 && nb3 % tdst == 0);
    ثابت int64_t ss0 = nb00 / ts0;
    ثابت int64_t ss1 = nb01 / ts0;
    ثابت int64_t ss2 = nb02 / ts0;
    ثابت int64_t ss3 = nb03 / ts0;
    ثابت int64_t ds0 = nb0 / tdst;
    ثابت int64_t ds1 = nb1 / tdst;
    ثابت int64_t ds2 = nb2 / tdst;
    ثابت int64_t ds3 = nb3 / tdst;
    rms_norm_f32_sycl(src0_dd, dst_dd, ne00, ne01, ne02, ne03,
        ss0, ss1, ss2, ss3, ds0, ds1, ds2, ds3, eps, main_بهیر, ctx.device);
}

// Fused rms_norm + scale (the qwen35 GDN l2-norm pair build_gdn_l2_norm emits):
// the scale factor is a host scalar in the GGML_OP_SCALE node's op_params, so
// unlike the mul variants there is no second device tensor to wire up.
فراغ ggml_sycl_op_rms_norm_scale_fused(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                       ggml_tensor * scale_tensor) {
    ثابت ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(scale_tensor->type == GGML_TYPE_F32);

    dpct::کتار_ptr main_بهیر = ctx.stream();
    SYCL_CHECK(ggml_sycl_سېټ_device(ctx.device));

    ثابت عائم * src0_dd = static_cast<ثابت عائم *>(src0->معلومات);
    عائم *       dst_dd  = static_cast<عائم *>(scale_tensor->معلومات);

    عائم eps;
    حافظه_نقل(&eps, dst->op_params, حجم(عائم));
    عائم scale_mul;
    حافظه_نقل(&scale_mul, scale_tensor->op_params, حجم(عائم));
    GGML_ASSERT(scale_mul >= 0.0f);

    GGML_TENSOR_UNARY_OP_LOCALS
    ثابت size_t ts0 = ggml_type_اندازه(src0->type);
    ثابت size_t tdst = ggml_type_اندازه(scale_tensor->type);
    GGML_ASSERT(nb00 % ts0 == 0 && nb01 % ts0 == 0 && nb02 % ts0 == 0 && nb03 % ts0 == 0);
    GGML_ASSERT(scale_tensor->nb[0] % tdst == 0 && scale_tensor->nb[1] % tdst == 0 &&
                scale_tensor->nb[2] % tdst == 0 && scale_tensor->nb[3] % tdst == 0);
    ثابت int64_t ss0 = nb00 / ts0;
    ثابت int64_t ss1 = nb01 / ts0;
    ثابت int64_t ss2 = nb02 / ts0;
    ثابت int64_t ss3 = nb03 / ts0;
    ثابت int64_t ds0 = scale_tensor->nb[0] / tdst;
    ثابت int64_t ds1 = scale_tensor->nb[1] / tdst;
    ثابت int64_t ds2 = scale_tensor->nb[2] / tdst;
    ثابت int64_t ds3 = scale_tensor->nb[3] / tdst;
    rms_norm_scale_f32_sycl(src0_dd, dst_dd, ne00, ne01, ne02, ne03,
        ss0, ss1, ss2, ss3, ds0, ds1, ds2, ds3, eps, scale_mul, main_بهیر, ctx.device);
}

فراغ ggml_sycl_op_rms_norm_fused(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor) {
    ثابت ggml_tensor * rms_norm_src = dst->src[0];
    عائم eps = 0.0f;
    حافظه_نقل(&eps, dst->op_params, حجم(عائم));

    ثابت عائم *       src0_dd = static_cast<ثابت عائم *>(rms_norm_src->معلومات);
    ثابت عائم *       mul_dd  = تش_اشاره;
    ثابت ggml_tensor * mul_src = تش_اشاره;
    إذا (mul_tensor->src[0] == dst) {
        mul_dd  = static_cast<ثابت عائم *>(mul_tensor->src[1]->معلومات);
        mul_src = mul_tensor->src[1];
    } وإلا إذا (mul_tensor->src[1] == dst) {
        mul_dd  = static_cast<ثابت عائم *>(mul_tensor->src[0]->معلومات);
        mul_src = mul_tensor->src[0];
    } وإلا {
        GGML_ASSERT(خطأ);
    }
    عائم * dst_dd = static_cast<عائم *>(mul_tensor->معلومات);

    dpct::کتار_ptr main_بهیر = ctx.stream();
    SYCL_CHECK(ggml_sycl_سېټ_device(ctx.device));

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    ثابت int64_t ne00 = rms_norm_src->ne[0];
    ثابت int64_t ne01 = rms_norm_src->ne[1];
    ثابت int64_t ne02 = rms_norm_src->ne[2];
    ثابت int64_t ne03 = rms_norm_src->ne[3];

    ثابت size_t ts0 = ggml_type_اندازه(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    ثابت int64_t s00 = rms_norm_src->nb[0] / ts0;
    ثابت int64_t s01 = rms_norm_src->nb[1] / ts0;
    ثابت int64_t s02 = rms_norm_src->nb[2] / ts0;
    ثابت int64_t s03 = rms_norm_src->nb[3] / ts0;

    ثابت size_t tdst = ggml_type_اندازه(mul_tensor->type);
    GGML_ASSERT(mul_tensor->nb[0] == tdst);
    ثابت int64_t d00 = mul_tensor->nb[0] / tdst;
    ثابت int64_t d01 = mul_tensor->nb[1] / tdst;
    ثابت int64_t d02 = mul_tensor->nb[2] / tdst;
    ثابت int64_t d03 = mul_tensor->nb[3] / tdst;

    ثابت size_t ts_mul = ggml_type_اندازه(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    ثابت int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    ثابت int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    ثابت int64_t mul_s03 = mul_src->nb[3] / ts_mul;
    ثابت صحيح mul_nrows     = mul_src->ne[1];
    ثابت صحيح mul_nchannels = mul_src->ne[2];
    ثابت صحيح mul_nsamples  = mul_src->ne[3];

    rms_norm_mul_f32_sycl(src0_dd, mul_dd, dst_dd, ne00, ne01, ne02, ne03,
        s00, s01, s02, s03, d00, d01, d02, d03,
        mul_s01, mul_s02, mul_s03, mul_nrows, mul_nchannels, mul_nsamples, eps, main_بهیر, ctx.device);
}

فراغ ggml_sycl_op_rms_norm_fused_جمع_کړه(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                     ggml_tensor * mul_tensor, ggml_tensor * جمع_کړه_tensor) {
    ثابت ggml_tensor * rms_norm_src = dst->src[0];
    عائم eps = 0.0f;
    حافظه_نقل(&eps, dst->op_params, حجم(عائم));

    ثابت عائم *       src0_dd = static_cast<ثابت عائم *>(rms_norm_src->معلومات);
    ثابت عائم *       mul_dd  = تش_اشاره;
    ثابت ggml_tensor * mul_src = تش_اشاره;
    إذا (mul_tensor->src[0] == dst) {
        mul_dd  = static_cast<ثابت عائم *>(mul_tensor->src[1]->معلومات);
        mul_src = mul_tensor->src[1];
    } وإلا إذا (mul_tensor->src[1] == dst) {
        mul_dd  = static_cast<ثابت عائم *>(mul_tensor->src[0]->معلومات);
        mul_src = mul_tensor->src[0];
    } وإلا {
        GGML_ASSERT(خطأ);
    }

    ثابت عائم *       جمع_کړه_dd  = تش_اشاره;
    ثابت ggml_tensor * جمع_کړه_src = تش_اشاره;
    إذا (جمع_کړه_tensor->src[0] == mul_tensor) {
        جمع_کړه_dd  = static_cast<ثابت عائم *>(جمع_کړه_tensor->src[1]->معلومات);
        جمع_کړه_src = جمع_کړه_tensor->src[1];
    } وإلا إذا (جمع_کړه_tensor->src[1] == mul_tensor) {
        جمع_کړه_dd  = static_cast<ثابت عائم *>(جمع_کړه_tensor->src[0]->معلومات);
        جمع_کړه_src = جمع_کړه_tensor->src[0];
    } وإلا {
        GGML_ASSERT(خطأ);
    }

    عائم * dst_dd = static_cast<عائم *>(جمع_کړه_tensor->معلومات);

    dpct::کتار_ptr main_بهیر = ctx.stream();
    SYCL_CHECK(ggml_sycl_سېټ_device(ctx.device));

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(جمع_کړه_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    ثابت int64_t ne00 = rms_norm_src->ne[0];
    ثابت int64_t ne01 = rms_norm_src->ne[1];
    ثابت int64_t ne02 = rms_norm_src->ne[2];
    ثابت int64_t ne03 = rms_norm_src->ne[3];

    ثابت size_t ts0 = ggml_type_اندازه(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    ثابت int64_t s00 = rms_norm_src->nb[0] / ts0;
    ثابت int64_t s01 = rms_norm_src->nb[1] / ts0;
    ثابت int64_t s02 = rms_norm_src->nb[2] / ts0;
    ثابت int64_t s03 = rms_norm_src->nb[3] / ts0;

    ثابت size_t tdst = ggml_type_اندازه(جمع_کړه_tensor->type);
    GGML_ASSERT(جمع_کړه_tensor->nb[0] == tdst);
    ثابت int64_t d00 = جمع_کړه_tensor->nb[0] / tdst;
    ثابت int64_t d01 = جمع_کړه_tensor->nb[1] / tdst;
    ثابت int64_t d02 = جمع_کړه_tensor->nb[2] / tdst;
    ثابت int64_t d03 = جمع_کړه_tensor->nb[3] / tdst;

    ثابت size_t ts_mul = ggml_type_اندازه(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    ثابت int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    ثابت int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    ثابت int64_t mul_s03 = mul_src->nb[3] / ts_mul;
    ثابت صحيح mul_nrows     = mul_src->ne[1];
    ثابت صحيح mul_nchannels = mul_src->ne[2];
    ثابت صحيح mul_nsamples  = mul_src->ne[3];

    ثابت size_t ts_جمع_کړه = ggml_type_اندازه(جمع_کړه_src->type);
    GGML_ASSERT(جمع_کړه_src->nb[0] == ts_جمع_کړه);
    ثابت int64_t جمع_کړه_s01 = جمع_کړه_src->nb[1] / ts_جمع_کړه;
    ثابت int64_t جمع_کړه_s02 = جمع_کړه_src->nb[2] / ts_جمع_کړه;
    ثابت int64_t جمع_کړه_s03 = جمع_کړه_src->nb[3] / ts_جمع_کړه;
    ثابت صحيح جمع_کړه_nrows     = جمع_کړه_src->ne[1];
    ثابت صحيح جمع_کړه_nchannels = جمع_کړه_src->ne[2];
    ثابت صحيح جمع_کړه_nsamples  = جمع_کړه_src->ne[3];

    rms_norm_mul_جمع_کړه_f32_sycl(src0_dd, mul_dd, جمع_کړه_dd, dst_dd, ne00, ne01, ne02, ne03,
        s00, s01, s02, s03, d00, d01, d02, d03,
        mul_s01, mul_s02, mul_s03, mul_nrows, mul_nchannels, mul_nsamples,
        جمع_کړه_s01, جمع_کړه_s02, جمع_کړه_s03, جمع_کړه_nrows, جمع_کړه_nchannels, جمع_کړه_nsamples, eps, main_بهیر, ctx.device);
}

فراغ ggml_sycl_op_rms_norm_شا(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32); // dz
    GGML_ASSERT(dst->src[1]->type == GGML_TYPE_F32); // x
    GGML_ASSERT(dst->type         == GGML_TYPE_F32);

    عائم eps = 1e-5f;
    std::حافظه_نقل(&eps, dst->op_params, حجم(عائم));
    إذا (!(eps > 0.0f) || !std::isfinite(eps)) eps = 1e-5f;

    ثابت عائم * g_بنسټ  = static_cast<ثابت عائم *>(dst->src[0]->معلومات); // dz
    ثابت عائم * x_بنسټ  = static_cast<ثابت عائم *>(dst->src[1]->معلومات); // x
          عائم * dx_بنسټ = static_cast<      عائم *>(dst->معلومات);

    ثابت int64_t D  = dst->ne[0];
    ثابت int64_t n1 = dst->ne[1], n2 = dst->ne[2], n3 = dst->ne[3]; (فراغ) n3;
    ثابت int64_t N  = ggml_nrows(dst);
    إذا (D == 0 || N == 0) أرجع;

    ثابت ggml_tensor *G = dst->src[0];
    ثابت ggml_tensor *X = dst->src[1];
    ثابت صحيح ts = (صحيح) ggml_type_اندازه(X->type);
    GGML_ASSERT((size_t) X->nb[0]   == (size_t) ts);
    GGML_ASSERT((size_t) G->nb[0]   == (size_t) ts);
    GGML_ASSERT((size_t) dst->nb[0] == (size_t) ts);

    ثابت int64_t xs1 = X->nb[1] / ts, xs2 = X->nb[2] / ts, xs3 = X->nb[3] / ts;
    ثابت int64_t gs1 = G->nb[1] / ts, gs2 = G->nb[2] / ts, gs3 = G->nb[3] / ts;
    ثابت int64_t ds1 = dst->nb[1] / ts, ds2 = dst->nb[2] / ts, ds3 = dst->nb[3] / ts;

    dpct::کتار_ptr q = ctx.stream();

    // work-group size: multiple of WARP_SIZE, capped by device and 256, and not larger than D
    ثابت صحيح device_max_wg = ggml_sycl_info().max_work_ډله_sizes[ctx.device];
    خپلکاری roundup = [](صحيح v, صحيح m) { أرجع ((v + m - 1) / m) * m; };
    صحيح wg_cap = 256;
    إذا (device_max_wg > 0) wg_cap = std::min(wg_cap, device_max_wg);
    صحيح WG = std::max(WARP_اندازه, std::min(roundup((صحيح)std::min<int64_t>(D, wg_cap), WARP_اندازه), wg_cap));

    // FP32 path: per-thread compensated accumulation + hierarchical reduction
    q->submit([&](sycl::handler &cgh) {
        ثابت صحيح nwarps_loc = std::max(1, WG / WARP_اندازه);
        // store one partial value per warp (xx and xg) for cross-warp reduction
        خپلکاری l_xx   = sycl::local_accessor<sycl::float2, 1>(sycl::range<1>(nwarps_loc), cgh);
        خپلکاری l_xg   = sycl::local_accessor<sycl::float2, 1>(sycl::range<1>(nwarps_loc), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, N) * sycl::range<3>(1, 1, WG),
                              sycl::range<3>(1, 1, WG)),
            [=](sycl::nd_توکی<3> توکی_ct1) [[sycl::reqd_sub_ډله_اندازه(WARP_اندازه)]] {
                ثابت صحيح لیکه = توکی_ct1.get_ډله(2);
                ثابت صحيح tid = توکی_ct1.get_local_id(2);

                ثابت int64_t i1 = لیکه % n1;
                ثابت int64_t i2 = (لیکه / n1) % n2;
                ثابت int64_t i3 = لیکه / (n1 * n2);

                ثابت عائم *__restrict x_لیکه = x_بنسټ + i3 * xs3 + i2 * xs2 + i1 * xs1;
                ثابت عائم *__restrict g_لیکه = g_بنسټ + i3 * gs3 + i2 * gs2 + i1 * gs1;
                عائم *__restrict d_لیکه       = dx_بنسټ + i3 * ds3 + i2 * ds2 + i1 * ds1;

                // per-thread accumulation (compensated by default)
                عائم مجموعه_xx = 0.f, مجموعه_xg = 0.f;
#که_نه_تعریف_شوی GGML_SYCL_RMS_شا_FAST
                عائم c_xx = 0.f, c_xg = 0.f;
#د_که_پای
                لكل (int64_t ستون = tid; ستون < D; ستون += WG) {
                    ثابت عائم xv = x_لیکه[ستون];
                    ثابت عائم gv = g_لیکه[ستون];
#که_تعریف_شوی GGML_SYCL_RMS_شا_FAST
                    مجموعه_xx += xv * xv;
                    مجموعه_xg += xv * gv;
#وإلا
                    عائم y1 = xv * xv - c_xx;
                    عائم t1 = مجموعه_xx + y1;
                    c_xx = (t1 - مجموعه_xx) - y1;
                    مجموعه_xx = t1;

                    عائم y2 = xv * gv - c_xg;
                    عائم t2 = مجموعه_xg + y2;
                    c_xg = (t2 - مجموعه_xg) - y2;
                    مجموعه_xg = t2;
#د_که_پای
                }

                // warp-level reduction
                sycl::float2 xx = sycl::float2(مجموعه_xx,
#که_نه_تعریف_شوی GGML_SYCL_RMS_شا_FAST
                    c_xx
#وإلا
                    0.f
#د_که_پای
                );
                sycl::float2 xg = sycl::float2(مجموعه_xg,
#که_نه_تعریف_شوی GGML_SYCL_RMS_شا_FAST
                    c_xg
#وإلا
                    0.f
#د_که_پای
                );
                xx = warp_reduce_مجموعه(xx, توکی_ct1);
                xg = warp_reduce_مجموعه(xg, توکی_ct1);

                // cross-warp reduction using local memory (single barrier)
                ثابت خپلکاری sub_ډله = توکی_ct1.get_sub_ډله();
                ثابت خپلکاری sg_id     = sub_ډله.get_ډله_خطي_id();
                ثابت خپلکاری wi_دننه_sg  = sub_ډله.get_local_خطي_id();
                ثابت صحيح nthreads   = توکی_ct1.get_local_range(2);
                ثابت صحيح nwarps     = nthreads / WARP_اندازه;

                sycl::float2 xx_ټول = xx;
                sycl::float2 xg_ټول = xg;
                إذا (nwarps > 1) {
                    إذا (wi_دننه_sg == 0) {
                        l_xx[sg_id] = xx;
                        l_xg[sg_id] = xg;
                    }
                    توکی_ct1.barrier(sycl::access::fence_space::local_space);

                    إذا (sg_id == 0) {
                        ثابت بینښه wi_u = wi_دننه_sg;
                        sycl::float2 xx_لومړی = (wi_u < static_cast<بینښه>(nwarps)) ? l_xx[wi_u] : sycl::float2(0.f, 0.f);
                        sycl::float2 xg_لومړی = (wi_u < static_cast<بینښه>(nwarps)) ? l_xg[wi_u] : sycl::float2(0.f, 0.f);
                        xx_ټول = warp_reduce_مجموعه(xx_لومړی, توکی_ct1);
                        xg_ټول = warp_reduce_مجموعه(xg_لومړی, توکی_ct1);
                    } وإلا {
                        // other subgroups keep their local totals; they'll be ignored
                        xx_ټول = xx;
                        xg_ټول = xg;
                    }
                    // ensure all threads see the first-subgroup result via broadcast below
                }

                // compute inv_r and coeff once per row and broadcast to the whole work-group
                عائم inv_r = 0.f;
                عائم coeff = 0.f;
                إذا (tid == 0) {
                    ثابت عائم مجموعه_xx_f  = xx_ټول.x() + xx_ټول.y();
                    ثابت عائم مجموعه_xdz_f = xg_ټول.x() + xg_ټول.y();
                    ثابت عائم منځنۍ_eps  = مجموعه_xx_f / (عائم) D + eps;
                    ثابت عائم مجموعه_eps   = مجموعه_xx_f + eps * (عائم) D;
                    inv_r = sycl::rsqrt(منځنۍ_eps);
                    coeff = -مجموعه_xdz_f / مجموعه_eps;
                }
                inv_r = sycl::ډله_خپرونه(توکی_ct1.get_ډله(), inv_r);
                coeff = sycl::ډله_خپرونه(توکی_ct1.get_ډله(), coeff);

                لكل (int64_t ستون = tid; ستون < D; ستون += WG) {
                    d_لیکه[ستون] = (g_لیکه[ستون] + coeff * x_لیکه[ستون]) * inv_r;
                }
            });
    });

}

فراغ ggml_sycl_op_l2_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت عائم * src0_d = (ثابت عائم *) src0->معلومات;
    عائم * dst_d = (عائم *) dst->معلومات;
    dpct::کتار_ptr     stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    عائم eps;
    حافظه_نقل(&eps, dst->op_params, حجم(عائم));
    GGML_ASSERT(eps >= 0.0f);

    ثابت size_t ts0 = ggml_type_اندازه(src0->type);
    ثابت size_t tdst = ggml_type_اندازه(dst->type);
    GGML_ASSERT(nb00 % ts0 == 0 && nb01 % ts0 == 0 && nb02 % ts0 == 0 && nb03 % ts0 == 0);
    GGML_ASSERT(nb0 % tdst == 0 && nb1 % tdst == 0 && nb2 % tdst == 0 && nb3 % tdst == 0);
    ثابت int64_t ss0 = nb00 / ts0;
    ثابت int64_t ss1 = nb01 / ts0;
    ثابت int64_t ss2 = nb02 / ts0;
    ثابت int64_t ss3 = nb03 / ts0;
    ثابت int64_t ds0 = nb0 / tdst;
    ثابت int64_t ds1 = nb1 / tdst;
    ثابت int64_t ds2 = nb2 / tdst;
    ثابت int64_t ds3 = nb3 / tdst;

    /*support both WARP_SIZE or WARP_32_SIZE in code
      choose by hardware for better performance
    */
    l2_norm_f32_sycl<WARP_اندازه>(src0_d, dst_d, ne00, ne01, ne02, ne03,
            ss0, ss1, ss2, ss3, ds0, ds1, ds2, ds3, eps, stream, ctx.device);
}

// nodes[0..count) are independent, same-shape, same-eps, same-nb L2_NORM ops validated
// by the caller; requires ncols < 1024 (the warp reduction path).
فراغ ggml_sycl_l2_norm_batch(ggml_backend_sycl_context & ctx, ggml_tensor ** nodes, صحيح count) {
    ثابت ggml_tensor * s0 = nodes[0]->src[0];
    ثابت صحيح ncols       = (صحيح) s0->ne[0];
    ثابت صحيح nrows_ټول = (صحيح) ggml_nrows(s0);
    عائم eps;
    حافظه_نقل(&eps, nodes[0]->op_params, حجم(عائم));
    GGML_ASSERT(eps >= 0.0f);

    l2_batch_ptrs p{};
    لكل (صحيح t = 0; t < count; ++t) {
        p.src[t] = (ثابت عائم *) nodes[t]->src[0]->معلومات;
        p.dst[t] = (عائم *) nodes[t]->معلومات;
    }

    ثابت ggml_tensor * d0 = nodes[0];
    ثابت size_t        ts = ggml_type_اندازه(GGML_TYPE_F32);
    l2_batch_strides    st{};
    st.ne1 = (صحيح) s0->ne[1];
    st.ne2 = (صحيح) s0->ne[2];
    st.ss0 = s0->nb[0] / ts; st.ss1 = s0->nb[1] / ts; st.ss2 = s0->nb[2] / ts; st.ss3 = s0->nb[3] / ts;
    st.ds0 = d0->nb[0] / ts; st.ds1 = d0->nb[1] / ts; st.ds2 = d0->nb[2] / ts; st.ds3 = d0->nb[3] / ts;

    l2_norm_f32_batch_sycl<WARP_اندازه>(p, st, count, ncols, nrows_ټول, eps, ctx.stream());
}
