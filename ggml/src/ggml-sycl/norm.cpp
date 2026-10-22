#ضمن "norm.hpp"
#ضمن "ggml-sycl/common.hpp"
#ضمن "ggml-sycl/presets.hpp"

ساكن فراغ norm_f32(ثابت عائم* x, عائم* dst, ثابت صحيح ncols,
    ثابت int64_t src_stride_ستون, ثابت int64_t src_stride_ردیف, ثابت int64_t src_stride_channel, ثابت int64_t src_stride_sample,
    ثابت int64_t dst_stride_ستون, ثابت int64_t dst_stride_ردیف, ثابت int64_t dst_stride_channel, ثابت int64_t dst_stride_sample,
    ثابت عائم eps, ثابت sycl::nd_قلم<3>& قلم_ct1, sycl::float2* s_جمع, صحيح بلوک_اندازه_3) {

    ثابت صحيح nthreads = قلم_ct1.get_local_range(2);
    ثابت صحيح sample  = قلم_ct1.get_گروه(0);
    ثابت صحيح channel = قلم_ct1.get_گروه(1);
    ثابت صحيح ردیف     = قلم_ct1.get_گروه(2);

    ثابت صحيح tid = قلم_ct1.get_local_id(2);
    ثابت صحيح nwarps = nthreads / WARP_اندازه;

    ثابت خودکار src_جابه_جایی = calculate_جابه_جایی<3>({src_stride_sample, src_stride_channel, src_stride_ردیف}, {sample, channel, ردیف});
    ثابت خودکار dst_جابه_جایی = calculate_جابه_جایی<3>({dst_stride_sample, dst_stride_channel, dst_stride_ردیف}, {sample, channel, ردیف});

    x += src_جابه_جایی;
    dst += dst_جابه_جایی;

    sycl::float2 میانگین_var = sycl::float2(0.f, 0.f);

    لكل (صحيح ستون = tid; ستون < ncols; ستون += بلوک_اندازه_3) {
        ثابت عائم xi = x[ستون * src_stride_ستون];
        میانگین_var.x() += xi;
        میانگین_var.y() += xi * xi;
    }

    // sum up partial sums
    میانگین_var = warp_reduce_جمع(میانگین_var, قلم_ct1);
    إذا  (بلوک_اندازه_3 > WARP_اندازه) {
        ثابت خودکار sub_گروه = قلم_ct1.get_sub_گروه();
        ثابت خودکار sg_id = sub_گروه.get_گروه_خطی_id();
        ثابت خودکار wi_درون_sg = sub_گروه.get_local_خطی_id();
        إذا (wi_درون_sg == 0) {
            s_جمع[sg_id] = میانگین_var;
        }
        قلم_ct1.barrier(sycl::access::fence_space::local_space);
        میانگین_var = 0.f;
        ثابت size_t nreduce = ceil_div(nwarps, WARP_اندازه);
        لكل (size_t i = 0; i < nreduce; i += 1)
        {
            میانگین_var += s_جمع[wi_درون_sg + i * WARP_اندازه];
        }
        میانگین_var = warp_reduce_جمع(میانگین_var, قلم_ct1);
    }

    ثابت عائم میانگین_2 = میانگین_var.x() / ncols;
    ثابت عائم var = میانگین_var.y() / ncols - میانگین_2 * میانگین_2;
    ثابت عائم inv_std = sycl::rsqrt(var + eps);

    لكل (صحيح ستون = tid; ستون < ncols; ستون += بلوک_اندازه_3) {
        dst[ستون * dst_stride_ستون] = (x[ستون * src_stride_ستون] - میانگین_2) * inv_std;
    }
}

ساكن فراغ گروه_norm_f32(ثابت عائم* x, عائم* dst, ثابت صحيح گروه_اندازه, ثابت صحيح ne_elements, ثابت عائم eps,
    ثابت sycl::nd_قلم<3>& قلم_ct1, عائم* s_جمع, صحيح بلوک_اندازه_3) {
    صحيح start = قلم_ct1.get_گروه(2) * گروه_اندازه;
    صحيح end = start + گروه_اندازه;
    ثابت صحيح nthreads = قلم_ct1.get_local_range(2);
    ثابت صحيح nwarps = nthreads / WARP_اندازه;
    start += قلم_ct1.get_local_id(2);
    size_t nreduce = nwarps / WARP_اندازه;

    إذا (end >= ne_elements) {
        end = ne_elements;
    }

    عائم tmp = 0.0f; // partial sum for thread in warp

    لكل (صحيح j = start; j < end; j += بلوک_اندازه_3) {
        tmp += x[j];
    }

    tmp = warp_reduce_جمع(tmp, قلم_ct1);
    إذا (بلوک_اندازه_3 > WARP_اندازه) {

        صحيح warp_id = قلم_ct1.get_local_id(2) / WARP_اندازه;
        صحيح lane_id = قلم_ct1.get_local_id(2) % WARP_اندازه;
        إذا (lane_id == 0) {
            s_جمع[warp_id] = tmp;
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
        قلم_ct1.barrier();
        tmp = 0.f;
        لكل (size_t i = 0; i < nreduce; i += 1)
        {
            tmp += s_جمع[lane_id + i * WARP_اندازه];
        }
        tmp = warp_reduce_جمع(tmp, قلم_ct1);
    }

    عائم میانگین_2 = tmp / گروه_اندازه;
    tmp = 0.0f;

    لكل (صحيح j = start; j < end; j += بلوک_اندازه_3) {
        عائم xi = x[j] - میانگین_2;
        dst[j] = xi;
        tmp += xi * xi;
    }

    tmp = warp_reduce_جمع(tmp, قلم_ct1);
    إذا (بلوک_اندازه_3 > WARP_اندازه) {

        صحيح warp_id = قلم_ct1.get_local_id(2) / WARP_اندازه;
        صحيح lane_id = قلم_ct1.get_local_id(2) % WARP_اندازه;
        إذا (lane_id == 0) {
            s_جمع[warp_id] = tmp;
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
        قلم_ct1.barrier();
        tmp = 0.f;
        لكل (size_t i = 0; i < nreduce; i += 1)
        {
            tmp += s_جمع[lane_id + i * WARP_اندازه];
        }
        tmp = warp_reduce_جمع(tmp, قلم_ct1);
    }

    عائم variance = tmp / گروه_اندازه;
    عائم scale = sycl::rsqrt(variance + eps);
    لكل (صحيح j = start; j < end; j += بلوک_اندازه_3) {
        dst[j] *= scale;
    }
}

قالب <منطقي do_multiply = خطأ, منطقي do_جمع_کن = خطأ>
ساكن فراغ rms_norm_f32(ثابت عائم* x, عائم* dst, ثابت صحيح ncols,
    ثابت int64_t src_stride_ستون, ثابت int64_t src_stride_ردیف, ثابت int64_t src_stride_channel, ثابت int64_t src_stride_sample,
    ثابت int64_t dst_stride_ستون, ثابت int64_t dst_stride_ردیف, ثابت int64_t dst_stride_channel, ثابت int64_t dst_stride_sample,
    ثابت عائم eps, ثابت sycl::nd_قلم<3>& قلم_ct1, عائم* s_جمع, صحيح بلوک_اندازه_3,
    ثابت عائم* mul = اشاره_تهی, ثابت int64_t mul_stride_ردیف = 0, ثابت int64_t mul_stride_channel = 0,
    ثابت int64_t mul_stride_sample = 0, ثابت صحيح mul_nrows = 0, ثابت صحيح mul_nchannels = 0, ثابت صحيح mul_nsamples = 0,
    ثابت عائم* جمع_کن_2 = اشاره_تهی, ثابت int64_t جمع_کن_stride_ردیف = 0, ثابت int64_t جمع_کن_stride_channel = 0,
    ثابت int64_t جمع_کن_stride_sample = 0, ثابت صحيح جمع_کن_nrows = 0, ثابت صحيح جمع_کن_nchannels = 0, ثابت صحيح جمع_کن_nsamples = 0,
    ثابت عائم scale_mul = 1.0f) {

    static_assert(!do_جمع_کن || do_multiply, "fusing add is not supported without multiplying");

    ثابت صحيح sample  = قلم_ct1.get_گروه(0);
    ثابت صحيح channel = قلم_ct1.get_گروه(1);
    ثابت صحيح ردیف     = قلم_ct1.get_گروه(2);

    ثابت صحيح nthreads = قلم_ct1.get_local_range(2);

    ثابت صحيح tid = قلم_ct1.get_local_id(2);
    ثابت صحيح nwarps = nthreads / WARP_اندازه;

    ثابت خودکار src_جابه_جایی = calculate_جابه_جایی<3>({src_stride_sample, src_stride_channel, src_stride_ردیف}, {sample, channel, ردیف});
    ثابت خودکار dst_جابه_جایی = calculate_جابه_جایی<3>({dst_stride_sample, dst_stride_channel, dst_stride_ردیف}, {sample, channel, ردیف});

    x   += src_جابه_جایی;
    dst += dst_جابه_جایی;

    إذا constexpr (do_multiply) {
        ثابت صحيح mul_ردیف     = ردیف     % mul_nrows;
        ثابت صحيح mul_channel = channel % mul_nchannels;
        ثابت صحيح mul_sample  = sample  % mul_nsamples;
        mul += mul_sample * mul_stride_sample + mul_channel * mul_stride_channel + mul_ردیف * mul_stride_ردیف;
    }

    إذا constexpr (do_جمع_کن) {
        ثابت صحيح جمع_کن_ردیف     = ردیف     % جمع_کن_nrows;
        ثابت صحيح جمع_کن_channel = channel % جمع_کن_nchannels;
        ثابت صحيح جمع_کن_sample  = sample  % جمع_کن_nsamples;
        جمع_کن_2 += جمع_کن_sample * جمع_کن_stride_sample + جمع_کن_channel * جمع_کن_stride_channel + جمع_کن_ردیف * جمع_کن_stride_ردیف;
    }

    عائم tmp = 0.0f; // partial sum for thread in warp

    لكل (صحيح ستون = tid; ستون < ncols; ستون += بلوک_اندازه_3) {
        ثابت عائم xi = x[ستون * src_stride_ستون];
        tmp += xi * xi;
    }

    // sum up partial sums
    tmp = warp_reduce_جمع(tmp, قلم_ct1);
    إذا (بلوک_اندازه_3 > WARP_اندازه) {
        ثابت خودکار sub_گروه = قلم_ct1.get_sub_گروه();
        ثابت خودکار sg_id = sub_گروه.get_گروه_خطی_id();
        ثابت خودکار wi_درون_sg = sub_گروه.get_local_خطی_id();
        إذا (wi_درون_sg == 0) {
            s_جمع[sg_id] = tmp;
        }

        قلم_ct1.barrier(sycl::access::fence_space::local_space);
        ثابت size_t nreduce = ceil_div(nwarps, WARP_اندازه);
        tmp = 0.f;
        لكل (size_t i = 0; i < nreduce; i += 1)
        {
            tmp += s_جمع[wi_درون_sg + i * WARP_اندازه];
        }
        tmp = warp_reduce_جمع(tmp, قلم_ct1);
    }

    ثابت عائم میانگین_2 = tmp / ncols;
    ثابت عائم scale = sycl::rsqrt(میانگین_2 + eps);

    لكل (صحيح ستون = tid; ستون < ncols; ستون += بلوک_اندازه_3) {
        إذا constexpr (do_multiply && do_جمع_کن) {
            dst[ستون * dst_stride_ستون] = scale * x[ستون * src_stride_ستون] * mul[ستون] + جمع_کن_2[ستون];
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

قالب<صحيح warp_اندازه>
ساكن فراغ l2_norm_f32(ثابت عائم * x, عائم * dst, ثابت صحيح ncols,
    ثابت int64_t src_stride_ستون, ثابت int64_t src_stride_ردیف, ثابت int64_t src_stride_channel,
    ثابت int64_t src_stride_sample, ثابت int64_t dst_stride_ستون, ثابت int64_t dst_stride_ردیف,
    ثابت int64_t dst_stride_channel, ثابت int64_t dst_stride_sample, ثابت عائم eps,
    ثابت sycl::nd_قلم<3>& قلم_ct1, عائم* s_جمع, ثابت صحيح بلوک_اندازه_3) {

    ثابت صحيح ردیف     = قلم_ct1.get_گروه(2);
    ثابت صحيح channel = قلم_ct1.get_گروه(1);
    ثابت صحيح sample  = قلم_ct1.get_گروه(0);
    ثابت صحيح tid     = قلم_ct1.get_local_id(2);

    x   += sample*src_stride_sample + channel*src_stride_channel + ردیف*src_stride_ردیف;
    dst += sample*dst_stride_sample + channel*dst_stride_channel + ردیف*dst_stride_ردیف;

    عائم tmp = 0.0f; // partial sum for thread in warp

    لكل (صحيح ستون = tid; ستون < ncols; ستون += بلوک_اندازه_3) {
        ثابت عائم xi = x[ستون * src_stride_ستون];
        tmp += xi * xi;
    }

    tmp = بلوک_reduce<بلوک_reduce_روش::جمع_3, warp_اندازه>(tmp, s_جمع, بلوک_اندازه_3);
    ثابت عائم scale = sycl::rsqrt(sycl::fmax(tmp, eps * eps));

    لكل (صحيح ستون = tid; ستون < ncols; ستون += بلوک_اندازه_3) {
        dst[ستون * dst_stride_ستون] = scale * x[ستون * src_stride_ستون];
    }
}

ساكن فراغ norm_f32_sycl(ثابت عائم * x, عائم * dst, ثابت صحيح ncols, ثابت صحيح nrows, ثابت صحيح nchannels, ثابت صحيح nsamples,
    ثابت int64_t src_stride_ستون, ثابت int64_t src_stride_ردیف, ثابت int64_t src_stride_channel, ثابت int64_t src_stride_sample,
    ثابت int64_t dst_stride_ستون, ثابت int64_t dst_stride_ردیف, ثابت int64_t dst_stride_channel, ثابت int64_t dst_stride_sample,
        ثابت عائم eps, صف_ptr stream, صحيح device) {

    ثابت sycl::range<3> global_dims(nsamples, nchannels, nrows);
    إذا (ncols < 1024) {
        ثابت sycl::range<3> بلوک_dims(1, 1, WARP_اندازه);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * بلوک_dims, بلوک_dims),
                [=](sycl::nd_قلم<3> قلم_ct1)
                [[sycl::reqd_sub_گروه_اندازه(WARP_اندازه)]] {
                    norm_f32(x, dst, ncols,
                        src_stride_ستون, src_stride_ردیف, src_stride_channel, src_stride_sample,
                        dst_stride_ستون, dst_stride_ردیف, dst_stride_channel, dst_stride_sample,
                        eps, قلم_ct1, اشاره_تهی, WARP_اندازه);
                });
            });
    }
    وإلا {
        ثابت صحيح work_گروه_اندازه = ggml_sycl_info().max_work_گروه_sizes[device];
        أكد(work_گروه_اندازه % (WARP_اندازه * WARP_اندازه) == 0);
        ثابت sycl::range<3> بلوک_dims(1, 1, work_گروه_اندازه);
        /*
        DPCT1049:17: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<sycl::float2, 1> s_جمع_acc_ct1(
                            sycl::range<1>(work_گروه_اندازه / WARP_اندازه), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * بلوک_dims, بلوک_dims),
                [=](sycl::nd_قلم<3> قلم_ct1)
                [[sycl::reqd_sub_گروه_اندازه(WARP_اندازه)]] {
                    norm_f32(x, dst, ncols,
                        src_stride_ستون, src_stride_ردیف, src_stride_channel, src_stride_sample,
                        dst_stride_ستون, dst_stride_ردیف, dst_stride_channel, dst_stride_sample,
                        eps, قلم_ct1, get_اشاره_گر(s_جمع_acc_ct1), work_گروه_اندازه);
                });
            });
    }
}

ساكن فراغ گروه_norm_f32_sycl(ثابت عائم* x, عائم* dst,
    ثابت صحيح رقم_groups, ثابت عائم eps, ثابت صحيح گروه_اندازه,
    ثابت صحيح ne_elements, صف_ptr stream, صحيح device) {
    إذا (گروه_اندازه < 1024) {
        ثابت sycl::range<3> بلوک_dims(1, 1, WARP_اندازه);
        stream->submit([&](sycl::handler& cgh) {
            ثابت عائم eps_ct4 = eps;
            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, رقم_groups) * بلوک_dims,
                    بلوک_dims),
                [=](sycl::nd_قلم<3> قلم_ct1)
                [[sycl::reqd_sub_گروه_اندازه(WARP_اندازه)]] {
                    گروه_norm_f32(
                        x, dst, گروه_اندازه, ne_elements, eps_ct4, قلم_ct1,
                        اشاره_تهی, WARP_اندازه);
                });
            });
    }
    وإلا {
        ثابت صحيح work_گروه_اندازه = ggml_sycl_info().max_work_گروه_sizes[device];
        أكد(work_گروه_اندازه % (WARP_اندازه * WARP_اندازه) == 0);
        ثابت sycl::range<3> بلوک_dims(1, 1, work_گروه_اندازه);
        /*
        DPCT1049:18: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */

        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<عائم, 1> s_جمع_acc_ct1(sycl::range<1>(work_گروه_اندازه / WARP_اندازه),
                cgh);

            ثابت عائم eps_ct4 = eps;

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, رقم_groups) * بلوک_dims,
                    بلوک_dims),
                [=](sycl::nd_قلم<3> قلم_ct1)
                [[sycl::reqd_sub_گروه_اندازه(WARP_اندازه)]] {
                    گروه_norm_f32(x, dst, گروه_اندازه, ne_elements,
                        eps_ct4, قلم_ct1,
                        get_اشاره_گر(s_جمع_acc_ct1), work_گروه_اندازه);
                });
            });
    }
}

ساكن فراغ rms_norm_f32_sycl(ثابت عائم* x, عائم* dst, ثابت صحيح ncols, ثابت صحيح nrows, ثابت صحيح nchannels, ثابت صحيح nsamples,
    ثابت int64_t src_stride_ستون, ثابت int64_t src_stride_ردیف, ثابت int64_t src_stride_channel, ثابت int64_t src_stride_sample,
    ثابت int64_t dst_stride_ستون, ثابت int64_t dst_stride_ردیف, ثابت int64_t dst_stride_channel, ثابت int64_t dst_stride_sample,
    ثابت عائم eps, صف_ptr stream, صحيح device) {
    // printf("%s ncols=%d, nrows=%d, WARP_SIZE=%d\n", __func__, ncols, nrows, WARP_SIZE);

    ثابت sycl::range<3> global_dims(nsamples, nchannels, nrows);
    إذا (ncols < 1024) {
        ثابت sycl::range<3> بلوک_dims(1, 1, WARP_اندازه);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * بلوک_dims, بلوک_dims),
                [=](sycl::nd_قلم<3> قلم_ct1)
                [[sycl::reqd_sub_گروه_اندازه(WARP_اندازه)]] {
                    rms_norm_f32(x, dst, ncols,
                        src_stride_ستون, src_stride_ردیف, src_stride_channel, src_stride_sample,
                        dst_stride_ستون, dst_stride_ردیف, dst_stride_channel, dst_stride_sample,
                        eps, قلم_ct1, اشاره_تهی, WARP_اندازه);
                });
            });
    }
    وإلا {
        ثابت صحيح work_گروه_اندازه = ggml_sycl_info().max_work_گروه_sizes[device];
        أكد(work_گروه_اندازه % (WARP_اندازه * WARP_اندازه) == 0);
        ثابت sycl::range<3> بلوک_dims(1, 1, work_گروه_اندازه);
        /*
        DPCT1049:19: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<عائم, 1> s_جمع_acc_ct1(sycl::range<1>(work_گروه_اندازه / WARP_اندازه),
                cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * بلوک_dims, بلوک_dims),
                [=](sycl::nd_قلم<3> قلم_ct1)
                [[sycl::reqd_sub_گروه_اندازه(WARP_اندازه)]] {
                    rms_norm_f32(x, dst, ncols,
                        src_stride_ستون, src_stride_ردیف, src_stride_channel, src_stride_sample,
                        dst_stride_ستون, dst_stride_ردیف, dst_stride_channel, dst_stride_sample,
                        eps, قلم_ct1, get_اشاره_گر(s_جمع_acc_ct1), work_گروه_اندازه);
                });
            });
    }
}

ساكن فراغ rms_norm_scale_f32_sycl(ثابت عائم* x, عائم* dst, ثابت صحيح ncols, ثابت صحيح nrows,
    ثابت صحيح nchannels, ثابت صحيح nsamples,
    ثابت int64_t src_stride_ستون, ثابت int64_t src_stride_ردیف, ثابت int64_t src_stride_channel, ثابت int64_t src_stride_sample,
    ثابت int64_t dst_stride_ستون, ثابت int64_t dst_stride_ردیف, ثابت int64_t dst_stride_channel, ثابت int64_t dst_stride_sample,
    ثابت عائم eps, ثابت عائم scale_mul, صف_ptr stream, صحيح device) {
    ثابت sycl::range<3> global_dims(nsamples, nchannels, nrows);
    إذا (ncols < 1024) {
        ثابت sycl::range<3> بلوک_dims(1, 1, WARP_اندازه);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * بلوک_dims, بلوک_dims),
                [=](sycl::nd_قلم<3> قلم_ct1)
                [[sycl::reqd_sub_گروه_اندازه(WARP_اندازه)]] {
                    rms_norm_f32(x, dst, ncols,
                        src_stride_ستون, src_stride_ردیف, src_stride_channel, src_stride_sample,
                        dst_stride_ستون, dst_stride_ردیف, dst_stride_channel, dst_stride_sample,
                        eps, قلم_ct1, اشاره_تهی, WARP_اندازه,
                        اشاره_تهی, 0, 0, 0, 0, 0, 0,
                        اشاره_تهی, 0, 0, 0, 0, 0, 0,
                        scale_mul);
                });
            });
    }
    وإلا {
        ثابت صحيح work_گروه_اندازه = ggml_sycl_info().max_work_گروه_sizes[device];
        أكد(work_گروه_اندازه % (WARP_اندازه * WARP_اندازه) == 0);
        ثابت sycl::range<3> بلوک_dims(1, 1, work_گروه_اندازه);
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<عائم, 1> s_جمع_acc_ct1(sycl::range<1>(work_گروه_اندازه / WARP_اندازه),
                cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * بلوک_dims, بلوک_dims),
                [=](sycl::nd_قلم<3> قلم_ct1)
                [[sycl::reqd_sub_گروه_اندازه(WARP_اندازه)]] {
                    rms_norm_f32(x, dst, ncols,
                        src_stride_ستون, src_stride_ردیف, src_stride_channel, src_stride_sample,
                        dst_stride_ستون, dst_stride_ردیف, dst_stride_channel, dst_stride_sample,
                        eps, قلم_ct1, get_اشاره_گر(s_جمع_acc_ct1), work_گروه_اندازه,
                        اشاره_تهی, 0, 0, 0, 0, 0, 0,
                        اشاره_تهی, 0, 0, 0, 0, 0, 0,
                        scale_mul);
                });
            });
    }
}

ساكن فراغ rms_norm_mul_f32_sycl(ثابت عائم* x, ثابت عائم* mul, عائم* dst, ثابت صحيح ncols, ثابت صحيح nrows,
        ثابت صحيح nchannels, ثابت صحيح nsamples,
        ثابت int64_t src_stride_ستون, ثابت int64_t src_stride_ردیف, ثابت int64_t src_stride_channel, ثابت int64_t src_stride_sample,
        ثابت int64_t dst_stride_ستون, ثابت int64_t dst_stride_ردیف, ثابت int64_t dst_stride_channel, ثابت int64_t dst_stride_sample,
        ثابت int64_t mul_stride_ردیف, ثابت int64_t mul_stride_channel, ثابت int64_t mul_stride_sample,
        ثابت صحيح mul_nrows, ثابت صحيح mul_nchannels, ثابت صحيح mul_nsamples,
        ثابت عائم eps, صف_ptr stream, صحيح device) {
    ثابت sycl::range<3> global_dims(nsamples, nchannels, nrows);
    إذا (ncols < 1024) {
        ثابت sycl::range<3> بلوک_dims(1, 1, WARP_اندازه);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * بلوک_dims, بلوک_dims),
                [=](sycl::nd_قلم<3> قلم_ct1)
                [[sycl::reqd_sub_گروه_اندازه(WARP_اندازه)]] {
                    rms_norm_f32<درست>(x, dst, ncols,
                        src_stride_ستون, src_stride_ردیف, src_stride_channel, src_stride_sample,
                        dst_stride_ستون, dst_stride_ردیف, dst_stride_channel, dst_stride_sample,
                        eps, قلم_ct1, اشاره_تهی, WARP_اندازه,
                        mul, mul_stride_ردیف, mul_stride_channel, mul_stride_sample, mul_nrows, mul_nchannels, mul_nsamples);
                });
            });
    }
    وإلا {
        ثابت صحيح work_گروه_اندازه = ggml_sycl_info().max_work_گروه_sizes[device];
        أكد(work_گروه_اندازه % (WARP_اندازه * WARP_اندازه) == 0);
        ثابت sycl::range<3> بلوک_dims(1, 1, work_گروه_اندازه);
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<عائم, 1> s_جمع_acc_ct1(sycl::range<1>(work_گروه_اندازه / WARP_اندازه), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * بلوک_dims, بلوک_dims),
                [=](sycl::nd_قلم<3> قلم_ct1)
                [[sycl::reqd_sub_گروه_اندازه(WARP_اندازه)]] {
                    rms_norm_f32<درست>(x, dst, ncols,
                        src_stride_ستون, src_stride_ردیف, src_stride_channel, src_stride_sample,
                        dst_stride_ستون, dst_stride_ردیف, dst_stride_channel, dst_stride_sample,
                        eps, قلم_ct1, get_اشاره_گر(s_جمع_acc_ct1), work_گروه_اندازه,
                        mul, mul_stride_ردیف, mul_stride_channel, mul_stride_sample, mul_nrows, mul_nchannels, mul_nsamples);
                });
            });
    }
}

ساكن فراغ rms_norm_mul_جمع_کن_f32_sycl(ثابت عائم* x, ثابت عائم* mul, ثابت عائم* جمع_کن_2, عائم* dst,
        ثابت صحيح ncols, ثابت صحيح nrows, ثابت صحيح nchannels, ثابت صحيح nsamples,
        ثابت int64_t src_stride_ستون, ثابت int64_t src_stride_ردیف, ثابت int64_t src_stride_channel, ثابت int64_t src_stride_sample,
        ثابت int64_t dst_stride_ستون, ثابت int64_t dst_stride_ردیف, ثابت int64_t dst_stride_channel, ثابت int64_t dst_stride_sample,
        ثابت int64_t mul_stride_ردیف, ثابت int64_t mul_stride_channel, ثابت int64_t mul_stride_sample,
        ثابت صحيح mul_nrows, ثابت صحيح mul_nchannels, ثابت صحيح mul_nsamples,
        ثابت int64_t جمع_کن_stride_ردیف, ثابت int64_t جمع_کن_stride_channel, ثابت int64_t جمع_کن_stride_sample,
        ثابت صحيح جمع_کن_nrows, ثابت صحيح جمع_کن_nchannels, ثابت صحيح جمع_کن_nsamples,
        ثابت عائم eps, صف_ptr stream, صحيح device) {
    ثابت sycl::range<3> global_dims(nsamples, nchannels, nrows);
    إذا (ncols < 1024) {
        ثابت sycl::range<3> بلوک_dims(1, 1, WARP_اندازه);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * بلوک_dims, بلوک_dims),
                [=](sycl::nd_قلم<3> قلم_ct1)
                [[sycl::reqd_sub_گروه_اندازه(WARP_اندازه)]] {
                    rms_norm_f32<درست, درست>(x, dst, ncols,
                        src_stride_ستون, src_stride_ردیف, src_stride_channel, src_stride_sample,
                        dst_stride_ستون, dst_stride_ردیف, dst_stride_channel, dst_stride_sample,
                        eps, قلم_ct1, اشاره_تهی, WARP_اندازه,
                        mul, mul_stride_ردیف, mul_stride_channel, mul_stride_sample, mul_nrows, mul_nchannels, mul_nsamples,
                        جمع_کن_2, جمع_کن_stride_ردیف, جمع_کن_stride_channel, جمع_کن_stride_sample, جمع_کن_nrows, جمع_کن_nchannels, جمع_کن_nsamples);
                });
            });
    }
    وإلا {
        ثابت صحيح work_گروه_اندازه = ggml_sycl_info().max_work_گروه_sizes[device];
        أكد(work_گروه_اندازه % (WARP_اندازه * WARP_اندازه) == 0);
        ثابت sycl::range<3> بلوک_dims(1, 1, work_گروه_اندازه);
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<عائم, 1> s_جمع_acc_ct1(sycl::range<1>(work_گروه_اندازه / WARP_اندازه), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * بلوک_dims, بلوک_dims),
                [=](sycl::nd_قلم<3> قلم_ct1)
                [[sycl::reqd_sub_گروه_اندازه(WARP_اندازه)]] {
                    rms_norm_f32<درست, درست>(x, dst, ncols,
                        src_stride_ستون, src_stride_ردیف, src_stride_channel, src_stride_sample,
                        dst_stride_ستون, dst_stride_ردیف, dst_stride_channel, dst_stride_sample,
                        eps, قلم_ct1, get_اشاره_گر(s_جمع_acc_ct1), work_گروه_اندازه,
                        mul, mul_stride_ردیف, mul_stride_channel, mul_stride_sample, mul_nrows, mul_nchannels, mul_nsamples,
                        جمع_کن_2, جمع_کن_stride_ردیف, جمع_کن_stride_channel, جمع_کن_stride_sample, جمع_کن_nrows, جمع_کن_nchannels, جمع_کن_nsamples);
                });
            });
    }
}

قالب<صحيح warp_اندازه>
ساكن فراغ l2_norm_f32_sycl(ثابت عائم *   x,
                             عائم *         dst,
                             ثابت صحيح       ncols,
                             ثابت صحيح       nrows,
                             ثابت صحيح       nchannels,
                             ثابت صحيح       nsamples,
                             ثابت int64_t   src_stride_ستون,
                             ثابت int64_t   src_stride_ردیف,
                             ثابت int64_t   src_stride_channel,
                             ثابت int64_t   src_stride_sample,
                             ثابت int64_t   dst_stride_ستون,
                             ثابت int64_t   dst_stride_ردیف,
                             ثابت int64_t   dst_stride_channel,
                             ثابت int64_t   dst_stride_sample,
                             ثابت عائم     eps,
                             صف_ptr       stream,
                             صحيح             device) {
    ثابت dpct::dim3 بلوکها_رقم(nrows, nchannels, nsamples);

    إذا (ncols < 1024) {
        ثابت dpct::dim3 بلوک_dims(warp_اندازه, 1, 1);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(بلوکها_رقم * بلوک_dims,
                    بلوک_dims),
                [=](sycl::nd_قلم<3> قلم_ct1)
                [[sycl::reqd_sub_گروه_اندازه(warp_اندازه)]] {
                    l2_norm_f32<warp_اندازه>(x, dst, ncols,
                        src_stride_ستون, src_stride_ردیف, src_stride_channel, src_stride_sample,
                        dst_stride_ستون, dst_stride_ردیف, dst_stride_channel, dst_stride_sample,
                        eps, قلم_ct1,
                        اشاره_تهی, warp_اندازه);
                });
            });
    }
    وإلا {
        ثابت صحيح work_گروه_اندازه = ggml_sycl_info().max_work_گروه_sizes[device];
        أكد(work_گروه_اندازه % (warp_اندازه * warp_اندازه) == 0);
        ثابت sycl::range<3> بلوک_dims(1, 1, work_گروه_اندازه);
        صحيح lsm_اندازه =  بلوک_dims[2] > warp_اندازه ? work_گروه_اندازه / warp_اندازه * حجم(عائم): 0;
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<عائم, 1> s_جمع_acc_ct1(sycl::range<1>(lsm_اندازه),
                cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(بلوکها_رقم * بلوک_dims,
                    بلوک_dims),
                [=](sycl::nd_قلم<3> قلم_ct1)
                [[sycl::reqd_sub_گروه_اندازه(warp_اندازه)]] {
                    l2_norm_f32<warp_اندازه>(x, dst, ncols,
                        src_stride_ستون, src_stride_ردیف, src_stride_channel, src_stride_sample,
                        dst_stride_ستون, dst_stride_ردیف, dst_stride_channel, dst_stride_sample,
                        eps, قلم_ct1, get_اشاره_گر(s_جمع_acc_ct1), work_گروه_اندازه);
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

قالب <صحيح warp_اندازه>
ساكن فراغ l2_norm_f32_batch(l2_batch_ptrs p, l2_batch_strides st, ثابت صحيح ncols, ثابت عائم eps,
                              ثابت sycl::nd_قلم<3> & قلم_ct1) {
    ثابت صحيح t   = قلم_ct1.get_گروه(0);  // tensor index
    ثابت صحيح r   = قلم_ct1.get_گروه(2);  // flattened row over ne1*ne2*ne3
    ثابت صحيح tid = قلم_ct1.get_local_id(2);

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
    tmp = بلوک_reduce<بلوک_reduce_روش::جمع_3, warp_اندازه>(tmp, (عائم *) اشاره_تهی, warp_اندازه);
    ثابت عائم scale = sycl::rsqrt(sycl::fmax(tmp, eps * eps));
    لكل (صحيح ستون = tid; ستون < ncols; ستون += warp_اندازه) {
        dst[ستون * st.ds0] = scale * x[ستون * st.ss0];
    }
}

قالب <صحيح warp_اندازه>
ساكن فراغ l2_norm_f32_batch_sycl(l2_batch_ptrs p, l2_batch_strides st, ثابت صحيح n_tensors,
                                   ثابت صحيح ncols, ثابت صحيح nrows_مجموع, ثابت عائم eps,
                                   صف_ptr stream) {
    ثابت dpct::dim3 بلوکها_رقم(nrows_مجموع, 1, n_tensors);
    ثابت dpct::dim3 بلوک_dims(warp_اندازه, 1, 1);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(بلوکها_رقم * بلوک_dims, بلوک_dims),
            [=](sycl::nd_قلم<3> قلم_ct1) [[sycl::reqd_sub_گروه_اندازه(warp_اندازه)]] {
                l2_norm_f32_batch<warp_اندازه>(p, st, ncols, eps, قلم_ct1);
            });
    });
}

فراغ ggml_sycl_op_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    ثابت ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS
    dpct::صف_ptr main_جریان = ctx.stream();
    SYCL_CHECK(ggml_sycl_مجموعه_device(ctx.device));
    ثابت عائم * src0_dd = static_cast<ثابت عائم *>(dst->src[0]->داده);
    عائم *       dst_dd  = static_cast<عائم *>(dst->داده);

    عائم eps;
    کپی_حافظه(&eps, dst->op_params, حجم(عائم));
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
        ss0, ss1, ss2, ss3, ds0, ds1, ds2, ds3, eps, main_جریان, ctx.device);
}

فراغ ggml_sycl_op_گروه_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    صحيح رقم_groups = dst->op_params[0];
    dpct::صف_ptr main_جریان = ctx.stream();
    SYCL_CHECK(ggml_sycl_مجموعه_device(ctx.device));

    ثابت عائم * src0_dd = static_cast<ثابت عائم *>(dst->src[0]->داده);
    عائم *       dst_dd  = static_cast<عائم *>(dst->داده);

    عائم eps;
    کپی_حافظه(&eps, dst->op_params + 1, حجم(عائم));

    صحيح گروه_اندازه = dst->src[0]->ne[0] * dst->src[0]->ne[1] * ((dst->src[0]->ne[2] + رقم_groups - 1) / رقم_groups);
    گروه_norm_f32_sycl(src0_dd, dst_dd, رقم_groups, eps, گروه_اندازه, dst->src[0]->ne[0] * dst->src[0]->ne[1] * dst->src[0]->ne[2], main_جریان, ctx.device);
}

فراغ ggml_sycl_op_rms_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    dpct::صف_ptr main_جریان = ctx.stream();
    SYCL_CHECK(ggml_sycl_مجموعه_device(ctx.device));

    ثابت عائم * src0_dd = static_cast<ثابت عائم *>(dst->src[0]->داده);
    عائم *       dst_dd  = static_cast<عائم *>(dst->داده);

    عائم eps;
    کپی_حافظه(&eps, dst->op_params, حجم(عائم));

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
        ss0, ss1, ss2, ss3, ds0, ds1, ds2, ds3, eps, main_جریان, ctx.device);
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

    dpct::صف_ptr main_جریان = ctx.stream();
    SYCL_CHECK(ggml_sycl_مجموعه_device(ctx.device));

    ثابت عائم * src0_dd = static_cast<ثابت عائم *>(src0->داده);
    عائم *       dst_dd  = static_cast<عائم *>(scale_tensor->داده);

    عائم eps;
    کپی_حافظه(&eps, dst->op_params, حجم(عائم));
    عائم scale_mul;
    کپی_حافظه(&scale_mul, scale_tensor->op_params, حجم(عائم));
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
        ss0, ss1, ss2, ss3, ds0, ds1, ds2, ds3, eps, scale_mul, main_جریان, ctx.device);
}

فراغ ggml_sycl_op_rms_norm_fused(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor) {
    ثابت ggml_tensor * rms_norm_src = dst->src[0];
    عائم eps = 0.0f;
    کپی_حافظه(&eps, dst->op_params, حجم(عائم));

    ثابت عائم *       src0_dd = static_cast<ثابت عائم *>(rms_norm_src->داده);
    ثابت عائم *       mul_dd  = اشاره_تهی;
    ثابت ggml_tensor * mul_src = اشاره_تهی;
    إذا (mul_tensor->src[0] == dst) {
        mul_dd  = static_cast<ثابت عائم *>(mul_tensor->src[1]->داده);
        mul_src = mul_tensor->src[1];
    } وإلا إذا (mul_tensor->src[1] == dst) {
        mul_dd  = static_cast<ثابت عائم *>(mul_tensor->src[0]->داده);
        mul_src = mul_tensor->src[0];
    } وإلا {
        GGML_ASSERT(خطأ);
    }
    عائم * dst_dd = static_cast<عائم *>(mul_tensor->داده);

    dpct::صف_ptr main_جریان = ctx.stream();
    SYCL_CHECK(ggml_sycl_مجموعه_device(ctx.device));

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
        mul_s01, mul_s02, mul_s03, mul_nrows, mul_nchannels, mul_nsamples, eps, main_جریان, ctx.device);
}

فراغ ggml_sycl_op_rms_norm_fused_جمع_کن(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                     ggml_tensor * mul_tensor, ggml_tensor * جمع_کن_tensor) {
    ثابت ggml_tensor * rms_norm_src = dst->src[0];
    عائم eps = 0.0f;
    کپی_حافظه(&eps, dst->op_params, حجم(عائم));

    ثابت عائم *       src0_dd = static_cast<ثابت عائم *>(rms_norm_src->داده);
    ثابت عائم *       mul_dd  = اشاره_تهی;
    ثابت ggml_tensor * mul_src = اشاره_تهی;
    إذا (mul_tensor->src[0] == dst) {
        mul_dd  = static_cast<ثابت عائم *>(mul_tensor->src[1]->داده);
        mul_src = mul_tensor->src[1];
    } وإلا إذا (mul_tensor->src[1] == dst) {
        mul_dd  = static_cast<ثابت عائم *>(mul_tensor->src[0]->داده);
        mul_src = mul_tensor->src[0];
    } وإلا {
        GGML_ASSERT(خطأ);
    }

    ثابت عائم *       جمع_کن_dd  = اشاره_تهی;
    ثابت ggml_tensor * جمع_کن_src = اشاره_تهی;
    إذا (جمع_کن_tensor->src[0] == mul_tensor) {
        جمع_کن_dd  = static_cast<ثابت عائم *>(جمع_کن_tensor->src[1]->داده);
        جمع_کن_src = جمع_کن_tensor->src[1];
    } وإلا إذا (جمع_کن_tensor->src[1] == mul_tensor) {
        جمع_کن_dd  = static_cast<ثابت عائم *>(جمع_کن_tensor->src[0]->داده);
        جمع_کن_src = جمع_کن_tensor->src[0];
    } وإلا {
        GGML_ASSERT(خطأ);
    }

    عائم * dst_dd = static_cast<عائم *>(جمع_کن_tensor->داده);

    dpct::صف_ptr main_جریان = ctx.stream();
    SYCL_CHECK(ggml_sycl_مجموعه_device(ctx.device));

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(جمع_کن_tensor->type == GGML_TYPE_F32);
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

    ثابت size_t tdst = ggml_type_اندازه(جمع_کن_tensor->type);
    GGML_ASSERT(جمع_کن_tensor->nb[0] == tdst);
    ثابت int64_t d00 = جمع_کن_tensor->nb[0] / tdst;
    ثابت int64_t d01 = جمع_کن_tensor->nb[1] / tdst;
    ثابت int64_t d02 = جمع_کن_tensor->nb[2] / tdst;
    ثابت int64_t d03 = جمع_کن_tensor->nb[3] / tdst;

    ثابت size_t ts_mul = ggml_type_اندازه(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    ثابت int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    ثابت int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    ثابت int64_t mul_s03 = mul_src->nb[3] / ts_mul;
    ثابت صحيح mul_nrows     = mul_src->ne[1];
    ثابت صحيح mul_nchannels = mul_src->ne[2];
    ثابت صحيح mul_nsamples  = mul_src->ne[3];

    ثابت size_t ts_جمع_کن = ggml_type_اندازه(جمع_کن_src->type);
    GGML_ASSERT(جمع_کن_src->nb[0] == ts_جمع_کن);
    ثابت int64_t جمع_کن_s01 = جمع_کن_src->nb[1] / ts_جمع_کن;
    ثابت int64_t جمع_کن_s02 = جمع_کن_src->nb[2] / ts_جمع_کن;
    ثابت int64_t جمع_کن_s03 = جمع_کن_src->nb[3] / ts_جمع_کن;
    ثابت صحيح جمع_کن_nrows     = جمع_کن_src->ne[1];
    ثابت صحيح جمع_کن_nchannels = جمع_کن_src->ne[2];
    ثابت صحيح جمع_کن_nsamples  = جمع_کن_src->ne[3];

    rms_norm_mul_جمع_کن_f32_sycl(src0_dd, mul_dd, جمع_کن_dd, dst_dd, ne00, ne01, ne02, ne03,
        s00, s01, s02, s03, d00, d01, d02, d03,
        mul_s01, mul_s02, mul_s03, mul_nrows, mul_nchannels, mul_nsamples,
        جمع_کن_s01, جمع_کن_s02, جمع_کن_s03, جمع_کن_nrows, جمع_کن_nchannels, جمع_کن_nsamples, eps, main_جریان, ctx.device);
}

فراغ ggml_sycl_op_rms_norm_پشت(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32); // dz
    GGML_ASSERT(dst->src[1]->type == GGML_TYPE_F32); // x
    GGML_ASSERT(dst->type         == GGML_TYPE_F32);

    عائم eps = 1e-5f;
    std::کپی_حافظه(&eps, dst->op_params, حجم(عائم));
    إذا (!(eps > 0.0f) || !std::isfinite(eps)) eps = 1e-5f;

    ثابت عائم * g_پایه  = static_cast<ثابت عائم *>(dst->src[0]->داده); // dz
    ثابت عائم * x_پایه  = static_cast<ثابت عائم *>(dst->src[1]->داده); // x
          عائم * dx_پایه = static_cast<      عائم *>(dst->داده);

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

    dpct::صف_ptr q = ctx.stream();

    // work-group size: multiple of WARP_SIZE, capped by device and 256, and not larger than D
    ثابت صحيح device_max_wg = ggml_sycl_info().max_work_گروه_sizes[ctx.device];
    خودکار roundup = [](صحيح v, صحيح m) { أرجع ((v + m - 1) / m) * m; };
    صحيح wg_cap = 256;
    إذا (device_max_wg > 0) wg_cap = std::min(wg_cap, device_max_wg);
    صحيح WG = std::max(WARP_اندازه, std::min(roundup((صحيح)std::min<int64_t>(D, wg_cap), WARP_اندازه), wg_cap));

    // FP32 path: per-thread compensated accumulation + hierarchical reduction
    q->submit([&](sycl::handler &cgh) {
        ثابت صحيح nwarps_loc = std::max(1, WG / WARP_اندازه);
        // store one partial value per warp (xx and xg) for cross-warp reduction
        خودکار l_xx   = sycl::local_accessor<sycl::float2, 1>(sycl::range<1>(nwarps_loc), cgh);
        خودکار l_xg   = sycl::local_accessor<sycl::float2, 1>(sycl::range<1>(nwarps_loc), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, N) * sycl::range<3>(1, 1, WG),
                              sycl::range<3>(1, 1, WG)),
            [=](sycl::nd_قلم<3> قلم_ct1) [[sycl::reqd_sub_گروه_اندازه(WARP_اندازه)]] {
                ثابت صحيح ردیف = قلم_ct1.get_گروه(2);
                ثابت صحيح tid = قلم_ct1.get_local_id(2);

                ثابت int64_t i1 = ردیف % n1;
                ثابت int64_t i2 = (ردیف / n1) % n2;
                ثابت int64_t i3 = ردیف / (n1 * n2);

                ثابت عائم *__restrict x_ردیف = x_پایه + i3 * xs3 + i2 * xs2 + i1 * xs1;
                ثابت عائم *__restrict g_ردیف = g_پایه + i3 * gs3 + i2 * gs2 + i1 * gs1;
                عائم *__restrict d_ردیف       = dx_پایه + i3 * ds3 + i2 * ds2 + i1 * ds1;

                // per-thread accumulation (compensated by default)
                عائم جمع_xx = 0.f, جمع_xg = 0.f;
#اگر_تعریف_نشده GGML_SYCL_RMS_پشت_FAST
                عائم c_xx = 0.f, c_xg = 0.f;
#پایان_اگر
                لكل (int64_t ستون = tid; ستون < D; ستون += WG) {
                    ثابت عائم xv = x_ردیف[ستون];
                    ثابت عائم gv = g_ردیف[ستون];
#اگر_تعریف_شده GGML_SYCL_RMS_پشت_FAST
                    جمع_xx += xv * xv;
                    جمع_xg += xv * gv;
#وإلا
                    عائم y1 = xv * xv - c_xx;
                    عائم t1 = جمع_xx + y1;
                    c_xx = (t1 - جمع_xx) - y1;
                    جمع_xx = t1;

                    عائم y2 = xv * gv - c_xg;
                    عائم t2 = جمع_xg + y2;
                    c_xg = (t2 - جمع_xg) - y2;
                    جمع_xg = t2;
#پایان_اگر
                }

                // warp-level reduction
                sycl::float2 xx = sycl::float2(جمع_xx,
#اگر_تعریف_نشده GGML_SYCL_RMS_پشت_FAST
                    c_xx
#وإلا
                    0.f
#پایان_اگر
                );
                sycl::float2 xg = sycl::float2(جمع_xg,
#اگر_تعریف_نشده GGML_SYCL_RMS_پشت_FAST
                    c_xg
#وإلا
                    0.f
#پایان_اگر
                );
                xx = warp_reduce_جمع(xx, قلم_ct1);
                xg = warp_reduce_جمع(xg, قلم_ct1);

                // cross-warp reduction using local memory (single barrier)
                ثابت خودکار sub_گروه = قلم_ct1.get_sub_گروه();
                ثابت خودکار sg_id     = sub_گروه.get_گروه_خطی_id();
                ثابت خودکار wi_درون_sg  = sub_گروه.get_local_خطی_id();
                ثابت صحيح nthreads   = قلم_ct1.get_local_range(2);
                ثابت صحيح nwarps     = nthreads / WARP_اندازه;

                sycl::float2 xx_مجموع = xx;
                sycl::float2 xg_مجموع = xg;
                إذا (nwarps > 1) {
                    إذا (wi_درون_sg == 0) {
                        l_xx[sg_id] = xx;
                        l_xg[sg_id] = xg;
                    }
                    قلم_ct1.barrier(sycl::access::fence_space::local_space);

                    إذا (sg_id == 0) {
                        ثابت بیعلامت wi_u = wi_درون_sg;
                        sycl::float2 xx_اولی = (wi_u < static_cast<بیعلامت>(nwarps)) ? l_xx[wi_u] : sycl::float2(0.f, 0.f);
                        sycl::float2 xg_اولی = (wi_u < static_cast<بیعلامت>(nwarps)) ? l_xg[wi_u] : sycl::float2(0.f, 0.f);
                        xx_مجموع = warp_reduce_جمع(xx_اولی, قلم_ct1);
                        xg_مجموع = warp_reduce_جمع(xg_اولی, قلم_ct1);
                    } وإلا {
                        // other subgroups keep their local totals; they'll be ignored
                        xx_مجموع = xx;
                        xg_مجموع = xg;
                    }
                    // ensure all threads see the first-subgroup result via broadcast below
                }

                // compute inv_r and coeff once per row and broadcast to the whole work-group
                عائم inv_r = 0.f;
                عائم coeff = 0.f;
                إذا (tid == 0) {
                    ثابت عائم جمع_xx_f  = xx_مجموع.x() + xx_مجموع.y();
                    ثابت عائم جمع_xdz_f = xg_مجموع.x() + xg_مجموع.y();
                    ثابت عائم میانگین_eps  = جمع_xx_f / (عائم) D + eps;
                    ثابت عائم جمع_eps   = جمع_xx_f + eps * (عائم) D;
                    inv_r = sycl::rsqrt(میانگین_eps);
                    coeff = -جمع_xdz_f / جمع_eps;
                }
                inv_r = sycl::گروه_پخش(قلم_ct1.get_گروه(), inv_r);
                coeff = sycl::گروه_پخش(قلم_ct1.get_گروه(), coeff);

                لكل (int64_t ستون = tid; ستون < D; ستون += WG) {
                    d_ردیف[ستون] = (g_ردیف[ستون] + coeff * x_ردیف[ستون]) * inv_r;
                }
            });
    });

}

فراغ ggml_sycl_op_l2_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت عائم * src0_d = (ثابت عائم *) src0->داده;
    عائم * dst_d = (عائم *) dst->داده;
    dpct::صف_ptr     stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    عائم eps;
    کپی_حافظه(&eps, dst->op_params, حجم(عائم));
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
    ثابت صحيح nrows_مجموع = (صحيح) ggml_nrows(s0);
    عائم eps;
    کپی_حافظه(&eps, nodes[0]->op_params, حجم(عائم));
    GGML_ASSERT(eps >= 0.0f);

    l2_batch_ptrs p{};
    لكل (صحيح t = 0; t < count; ++t) {
        p.src[t] = (ثابت عائم *) nodes[t]->src[0]->داده;
        p.dst[t] = (عائم *) nodes[t]->داده;
    }

    ثابت ggml_tensor * d0 = nodes[0];
    ثابت size_t        ts = ggml_type_اندازه(GGML_TYPE_F32);
    l2_batch_strides    st{};
    st.ne1 = (صحيح) s0->ne[1];
    st.ne2 = (صحيح) s0->ne[2];
    st.ss0 = s0->nb[0] / ts; st.ss1 = s0->nb[1] / ts; st.ss2 = s0->nb[2] / ts; st.ss3 = s0->nb[3] / ts;
    st.ds0 = d0->nb[0] / ts; st.ds1 = d0->nb[1] / ts; st.ds2 = d0->nb[2] / ts; st.ds3 = d0->nb[3] / ts;

    l2_norm_f32_batch_sycl<WARP_اندازه>(p, st, count, ncols, nrows_مجموع, eps, ctx.stream());
}
