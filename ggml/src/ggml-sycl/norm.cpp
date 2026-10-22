#ضمن "norm.hpp"
#ضمن "ggml-sycl/common.hpp"
#ضمن "ggml-sycl/presets.hpp"

ساكن فراغ norm_f32(ثابت عائم* x, عائم* dst, ثابت صحيح ncols,
    ثابت int64_t src_stride_کانو, ثابت int64_t src_stride_صف, ثابت int64_t src_stride_channel, ثابت int64_t src_stride_sample,
    ثابت int64_t dst_stride_کانو, ثابت int64_t dst_stride_صف, ثابت int64_t dst_stride_channel, ثابت int64_t dst_stride_sample,
    ثابت عائم eps, ثابت sycl::nd_شي<3>& شي_ct1, sycl::float2* s_جوڙ, صحيح بلاڪ_سائيز_3) {

    ثابت صحيح nthreads = شي_ct1.get_local_range(2);
    ثابت صحيح sample  = شي_ct1.get_گروهه(0);
    ثابت صحيح channel = شي_ct1.get_گروهه(1);
    ثابت صحيح صف     = شي_ct1.get_گروهه(2);

    ثابت صحيح tid = شي_ct1.get_local_id(2);
    ثابت صحيح nwarps = nthreads / WARP_سائيز;

    ثابت خودڪار src_وٿي = calculate_وٿي<3>({src_stride_sample, src_stride_channel, src_stride_صف}, {sample, channel, صف});
    ثابت خودڪار dst_وٿي = calculate_وٿي<3>({dst_stride_sample, dst_stride_channel, dst_stride_صف}, {sample, channel, صف});

    x += src_وٿي;
    dst += dst_وٿي;

    sycl::float2 سراسري_var = sycl::float2(0.f, 0.f);

    لكل (صحيح کانو = tid; کانو < ncols; کانو += بلاڪ_سائيز_3) {
        ثابت عائم xi = x[کانو * src_stride_کانو];
        سراسري_var.x() += xi;
        سراسري_var.y() += xi * xi;
    }

    // sum up partial sums
    سراسري_var = warp_reduce_جوڙ(سراسري_var, شي_ct1);
    إذا  (بلاڪ_سائيز_3 > WARP_سائيز) {
        ثابت خودڪار sub_گروهه = شي_ct1.get_sub_گروهه();
        ثابت خودڪار sg_id = sub_گروهه.get_گروهه_لڪيري_id();
        ثابت خودڪار wi_اندريون_sg = sub_گروهه.get_local_لڪيري_id();
        إذا (wi_اندريون_sg == 0) {
            s_جوڙ[sg_id] = سراسري_var;
        }
        شي_ct1.barrier(sycl::access::fence_space::local_space);
        سراسري_var = 0.f;
        ثابت size_t nreduce = ceil_div(nwarps, WARP_سائيز);
        لكل (size_t i = 0; i < nreduce; i += 1)
        {
            سراسري_var += s_جوڙ[wi_اندريون_sg + i * WARP_سائيز];
        }
        سراسري_var = warp_reduce_جوڙ(سراسري_var, شي_ct1);
    }

    ثابت عائم سراسري_2 = سراسري_var.x() / ncols;
    ثابت عائم var = سراسري_var.y() / ncols - سراسري_2 * سراسري_2;
    ثابت عائم inv_std = sycl::rsqrt(var + eps);

    لكل (صحيح کانو = tid; کانو < ncols; کانو += بلاڪ_سائيز_3) {
        dst[کانو * dst_stride_کانو] = (x[کانو * src_stride_کانو] - سراسري_2) * inv_std;
    }
}

ساكن فراغ گروهه_norm_f32(ثابت عائم* x, عائم* dst, ثابت صحيح گروهه_سائيز, ثابت صحيح ne_elements, ثابت عائم eps,
    ثابت sycl::nd_شي<3>& شي_ct1, عائم* s_جوڙ, صحيح بلاڪ_سائيز_3) {
    صحيح start = شي_ct1.get_گروهه(2) * گروهه_سائيز;
    صحيح end = start + گروهه_سائيز;
    ثابت صحيح nthreads = شي_ct1.get_local_range(2);
    ثابت صحيح nwarps = nthreads / WARP_سائيز;
    start += شي_ct1.get_local_id(2);
    size_t nreduce = nwarps / WARP_سائيز;

    إذا (end >= ne_elements) {
        end = ne_elements;
    }

    عائم tmp = 0.0f; // partial sum for thread in warp

    لكل (صحيح j = start; j < end; j += بلاڪ_سائيز_3) {
        tmp += x[j];
    }

    tmp = warp_reduce_جوڙ(tmp, شي_ct1);
    إذا (بلاڪ_سائيز_3 > WARP_سائيز) {

        صحيح warp_id = شي_ct1.get_local_id(2) / WARP_سائيز;
        صحيح lane_id = شي_ct1.get_local_id(2) % WARP_سائيز;
        إذا (lane_id == 0) {
            s_جوڙ[warp_id] = tmp;
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
        شي_ct1.barrier();
        tmp = 0.f;
        لكل (size_t i = 0; i < nreduce; i += 1)
        {
            tmp += s_جوڙ[lane_id + i * WARP_سائيز];
        }
        tmp = warp_reduce_جوڙ(tmp, شي_ct1);
    }

    عائم سراسري_2 = tmp / گروهه_سائيز;
    tmp = 0.0f;

    لكل (صحيح j = start; j < end; j += بلاڪ_سائيز_3) {
        عائم xi = x[j] - سراسري_2;
        dst[j] = xi;
        tmp += xi * xi;
    }

    tmp = warp_reduce_جوڙ(tmp, شي_ct1);
    إذا (بلاڪ_سائيز_3 > WARP_سائيز) {

        صحيح warp_id = شي_ct1.get_local_id(2) / WARP_سائيز;
        صحيح lane_id = شي_ct1.get_local_id(2) % WARP_سائيز;
        إذا (lane_id == 0) {
            s_جوڙ[warp_id] = tmp;
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
        شي_ct1.barrier();
        tmp = 0.f;
        لكل (size_t i = 0; i < nreduce; i += 1)
        {
            tmp += s_جوڙ[lane_id + i * WARP_سائيز];
        }
        tmp = warp_reduce_جوڙ(tmp, شي_ct1);
    }

    عائم variance = tmp / گروهه_سائيز;
    عائم scale = sycl::rsqrt(variance + eps);
    لكل (صحيح j = start; j < end; j += بلاڪ_سائيز_3) {
        dst[j] *= scale;
    }
}

سانچو <منطقي do_multiply = خطأ, منطقي do_جوڙيو = خطأ>
ساكن فراغ rms_norm_f32(ثابت عائم* x, عائم* dst, ثابت صحيح ncols,
    ثابت int64_t src_stride_کانو, ثابت int64_t src_stride_صف, ثابت int64_t src_stride_channel, ثابت int64_t src_stride_sample,
    ثابت int64_t dst_stride_کانو, ثابت int64_t dst_stride_صف, ثابت int64_t dst_stride_channel, ثابت int64_t dst_stride_sample,
    ثابت عائم eps, ثابت sycl::nd_شي<3>& شي_ct1, عائم* s_جوڙ, صحيح بلاڪ_سائيز_3,
    ثابت عائم* mul = خالي_اشارو, ثابت int64_t mul_stride_صف = 0, ثابت int64_t mul_stride_channel = 0,
    ثابت int64_t mul_stride_sample = 0, ثابت صحيح mul_nrows = 0, ثابت صحيح mul_nchannels = 0, ثابت صحيح mul_nsamples = 0,
    ثابت عائم* جوڙيو_2 = خالي_اشارو, ثابت int64_t جوڙيو_stride_صف = 0, ثابت int64_t جوڙيو_stride_channel = 0,
    ثابت int64_t جوڙيو_stride_sample = 0, ثابت صحيح جوڙيو_nrows = 0, ثابت صحيح جوڙيو_nchannels = 0, ثابت صحيح جوڙيو_nsamples = 0,
    ثابت عائم scale_mul = 1.0f) {

    static_assert(!do_جوڙيو || do_multiply, "fusing add is not supported without multiplying");

    ثابت صحيح sample  = شي_ct1.get_گروهه(0);
    ثابت صحيح channel = شي_ct1.get_گروهه(1);
    ثابت صحيح صف     = شي_ct1.get_گروهه(2);

    ثابت صحيح nthreads = شي_ct1.get_local_range(2);

    ثابت صحيح tid = شي_ct1.get_local_id(2);
    ثابت صحيح nwarps = nthreads / WARP_سائيز;

    ثابت خودڪار src_وٿي = calculate_وٿي<3>({src_stride_sample, src_stride_channel, src_stride_صف}, {sample, channel, صف});
    ثابت خودڪار dst_وٿي = calculate_وٿي<3>({dst_stride_sample, dst_stride_channel, dst_stride_صف}, {sample, channel, صف});

    x   += src_وٿي;
    dst += dst_وٿي;

    إذا constexpr (do_multiply) {
        ثابت صحيح mul_صف     = صف     % mul_nrows;
        ثابت صحيح mul_channel = channel % mul_nchannels;
        ثابت صحيح mul_sample  = sample  % mul_nsamples;
        mul += mul_sample * mul_stride_sample + mul_channel * mul_stride_channel + mul_صف * mul_stride_صف;
    }

    إذا constexpr (do_جوڙيو) {
        ثابت صحيح جوڙيو_صف     = صف     % جوڙيو_nrows;
        ثابت صحيح جوڙيو_channel = channel % جوڙيو_nchannels;
        ثابت صحيح جوڙيو_sample  = sample  % جوڙيو_nsamples;
        جوڙيو_2 += جوڙيو_sample * جوڙيو_stride_sample + جوڙيو_channel * جوڙيو_stride_channel + جوڙيو_صف * جوڙيو_stride_صف;
    }

    عائم tmp = 0.0f; // partial sum for thread in warp

    لكل (صحيح کانو = tid; کانو < ncols; کانو += بلاڪ_سائيز_3) {
        ثابت عائم xi = x[کانو * src_stride_کانو];
        tmp += xi * xi;
    }

    // sum up partial sums
    tmp = warp_reduce_جوڙ(tmp, شي_ct1);
    إذا (بلاڪ_سائيز_3 > WARP_سائيز) {
        ثابت خودڪار sub_گروهه = شي_ct1.get_sub_گروهه();
        ثابت خودڪار sg_id = sub_گروهه.get_گروهه_لڪيري_id();
        ثابت خودڪار wi_اندريون_sg = sub_گروهه.get_local_لڪيري_id();
        إذا (wi_اندريون_sg == 0) {
            s_جوڙ[sg_id] = tmp;
        }

        شي_ct1.barrier(sycl::access::fence_space::local_space);
        ثابت size_t nreduce = ceil_div(nwarps, WARP_سائيز);
        tmp = 0.f;
        لكل (size_t i = 0; i < nreduce; i += 1)
        {
            tmp += s_جوڙ[wi_اندريون_sg + i * WARP_سائيز];
        }
        tmp = warp_reduce_جوڙ(tmp, شي_ct1);
    }

    ثابت عائم سراسري_2 = tmp / ncols;
    ثابت عائم scale = sycl::rsqrt(سراسري_2 + eps);

    لكل (صحيح کانو = tid; کانو < ncols; کانو += بلاڪ_سائيز_3) {
        إذا constexpr (do_multiply && do_جوڙيو) {
            dst[کانو * dst_stride_کانو] = scale * x[کانو * src_stride_کانو] * mul[کانو] + جوڙيو_2[کانو];
        } وإلا إذا constexpr (do_multiply) {
            dst[کانو * dst_stride_کانو] = scale * x[کانو * src_stride_کانو] * mul[کانو];
        } وإلا {
            // folded epilogue of a fused GGML_OP_SCALE consumer (qwen35 GDN l2 norms);
            // the explicit temporary keeps the float evaluation order identical to
            // running rms_norm and scale as two separate kernels
            ثابت عائم v = scale * x[کانو * src_stride_کانو];
            dst[کانو * dst_stride_کانو] = v * scale_mul;
        }
    }
}

سانچو<صحيح warp_سائيز>
ساكن فراغ l2_norm_f32(ثابت عائم * x, عائم * dst, ثابت صحيح ncols,
    ثابت int64_t src_stride_کانو, ثابت int64_t src_stride_صف, ثابت int64_t src_stride_channel,
    ثابت int64_t src_stride_sample, ثابت int64_t dst_stride_کانو, ثابت int64_t dst_stride_صف,
    ثابت int64_t dst_stride_channel, ثابت int64_t dst_stride_sample, ثابت عائم eps,
    ثابت sycl::nd_شي<3>& شي_ct1, عائم* s_جوڙ, ثابت صحيح بلاڪ_سائيز_3) {

    ثابت صحيح صف     = شي_ct1.get_گروهه(2);
    ثابت صحيح channel = شي_ct1.get_گروهه(1);
    ثابت صحيح sample  = شي_ct1.get_گروهه(0);
    ثابت صحيح tid     = شي_ct1.get_local_id(2);

    x   += sample*src_stride_sample + channel*src_stride_channel + صف*src_stride_صف;
    dst += sample*dst_stride_sample + channel*dst_stride_channel + صف*dst_stride_صف;

    عائم tmp = 0.0f; // partial sum for thread in warp

    لكل (صحيح کانو = tid; کانو < ncols; کانو += بلاڪ_سائيز_3) {
        ثابت عائم xi = x[کانو * src_stride_کانو];
        tmp += xi * xi;
    }

    tmp = بلاڪ_reduce<بلاڪ_reduce_طريقو::جوڙ_3, warp_سائيز>(tmp, s_جوڙ, بلاڪ_سائيز_3);
    ثابت عائم scale = sycl::rsqrt(sycl::fmax(tmp, eps * eps));

    لكل (صحيح کانو = tid; کانو < ncols; کانو += بلاڪ_سائيز_3) {
        dst[کانو * dst_stride_کانو] = scale * x[کانو * src_stride_کانو];
    }
}

ساكن فراغ norm_f32_sycl(ثابت عائم * x, عائم * dst, ثابت صحيح ncols, ثابت صحيح nrows, ثابت صحيح nchannels, ثابت صحيح nsamples,
    ثابت int64_t src_stride_کانو, ثابت int64_t src_stride_صف, ثابت int64_t src_stride_channel, ثابت int64_t src_stride_sample,
    ثابت int64_t dst_stride_کانو, ثابت int64_t dst_stride_صف, ثابت int64_t dst_stride_channel, ثابت int64_t dst_stride_sample,
        ثابت عائم eps, قطار_ptr stream, صحيح device) {

    ثابت sycl::range<3> global_dims(nsamples, nchannels, nrows);
    إذا (ncols < 1024) {
        ثابت sycl::range<3> بلاڪ_dims(1, 1, WARP_سائيز);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * بلاڪ_dims, بلاڪ_dims),
                [=](sycl::nd_شي<3> شي_ct1)
                [[sycl::reqd_sub_گروهه_سائيز(WARP_سائيز)]] {
                    norm_f32(x, dst, ncols,
                        src_stride_کانو, src_stride_صف, src_stride_channel, src_stride_sample,
                        dst_stride_کانو, dst_stride_صف, dst_stride_channel, dst_stride_sample,
                        eps, شي_ct1, خالي_اشارو, WARP_سائيز);
                });
            });
    }
    وإلا {
        ثابت صحيح work_گروهه_سائيز = ggml_sycl_info().max_work_گروهه_sizes[device];
        أكد(work_گروهه_سائيز % (WARP_سائيز * WARP_سائيز) == 0);
        ثابت sycl::range<3> بلاڪ_dims(1, 1, work_گروهه_سائيز);
        /*
        DPCT1049:17: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<sycl::float2, 1> s_جوڙ_acc_ct1(
                            sycl::range<1>(work_گروهه_سائيز / WARP_سائيز), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * بلاڪ_dims, بلاڪ_dims),
                [=](sycl::nd_شي<3> شي_ct1)
                [[sycl::reqd_sub_گروهه_سائيز(WARP_سائيز)]] {
                    norm_f32(x, dst, ncols,
                        src_stride_کانو, src_stride_صف, src_stride_channel, src_stride_sample,
                        dst_stride_کانو, dst_stride_صف, dst_stride_channel, dst_stride_sample,
                        eps, شي_ct1, get_اشارو(s_جوڙ_acc_ct1), work_گروهه_سائيز);
                });
            });
    }
}

ساكن فراغ گروهه_norm_f32_sycl(ثابت عائم* x, عائم* dst,
    ثابت صحيح انگ_groups, ثابت عائم eps, ثابت صحيح گروهه_سائيز,
    ثابت صحيح ne_elements, قطار_ptr stream, صحيح device) {
    إذا (گروهه_سائيز < 1024) {
        ثابت sycl::range<3> بلاڪ_dims(1, 1, WARP_سائيز);
        stream->submit([&](sycl::handler& cgh) {
            ثابت عائم eps_ct4 = eps;
            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, انگ_groups) * بلاڪ_dims,
                    بلاڪ_dims),
                [=](sycl::nd_شي<3> شي_ct1)
                [[sycl::reqd_sub_گروهه_سائيز(WARP_سائيز)]] {
                    گروهه_norm_f32(
                        x, dst, گروهه_سائيز, ne_elements, eps_ct4, شي_ct1,
                        خالي_اشارو, WARP_سائيز);
                });
            });
    }
    وإلا {
        ثابت صحيح work_گروهه_سائيز = ggml_sycl_info().max_work_گروهه_sizes[device];
        أكد(work_گروهه_سائيز % (WARP_سائيز * WARP_سائيز) == 0);
        ثابت sycl::range<3> بلاڪ_dims(1, 1, work_گروهه_سائيز);
        /*
        DPCT1049:18: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */

        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<عائم, 1> s_جوڙ_acc_ct1(sycl::range<1>(work_گروهه_سائيز / WARP_سائيز),
                cgh);

            ثابت عائم eps_ct4 = eps;

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, انگ_groups) * بلاڪ_dims,
                    بلاڪ_dims),
                [=](sycl::nd_شي<3> شي_ct1)
                [[sycl::reqd_sub_گروهه_سائيز(WARP_سائيز)]] {
                    گروهه_norm_f32(x, dst, گروهه_سائيز, ne_elements,
                        eps_ct4, شي_ct1,
                        get_اشارو(s_جوڙ_acc_ct1), work_گروهه_سائيز);
                });
            });
    }
}

ساكن فراغ rms_norm_f32_sycl(ثابت عائم* x, عائم* dst, ثابت صحيح ncols, ثابت صحيح nrows, ثابت صحيح nchannels, ثابت صحيح nsamples,
    ثابت int64_t src_stride_کانو, ثابت int64_t src_stride_صف, ثابت int64_t src_stride_channel, ثابت int64_t src_stride_sample,
    ثابت int64_t dst_stride_کانو, ثابت int64_t dst_stride_صف, ثابت int64_t dst_stride_channel, ثابت int64_t dst_stride_sample,
    ثابت عائم eps, قطار_ptr stream, صحيح device) {
    // printf("%s ncols=%d, nrows=%d, WARP_SIZE=%d\n", __func__, ncols, nrows, WARP_SIZE);

    ثابت sycl::range<3> global_dims(nsamples, nchannels, nrows);
    إذا (ncols < 1024) {
        ثابت sycl::range<3> بلاڪ_dims(1, 1, WARP_سائيز);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * بلاڪ_dims, بلاڪ_dims),
                [=](sycl::nd_شي<3> شي_ct1)
                [[sycl::reqd_sub_گروهه_سائيز(WARP_سائيز)]] {
                    rms_norm_f32(x, dst, ncols,
                        src_stride_کانو, src_stride_صف, src_stride_channel, src_stride_sample,
                        dst_stride_کانو, dst_stride_صف, dst_stride_channel, dst_stride_sample,
                        eps, شي_ct1, خالي_اشارو, WARP_سائيز);
                });
            });
    }
    وإلا {
        ثابت صحيح work_گروهه_سائيز = ggml_sycl_info().max_work_گروهه_sizes[device];
        أكد(work_گروهه_سائيز % (WARP_سائيز * WARP_سائيز) == 0);
        ثابت sycl::range<3> بلاڪ_dims(1, 1, work_گروهه_سائيز);
        /*
        DPCT1049:19: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<عائم, 1> s_جوڙ_acc_ct1(sycl::range<1>(work_گروهه_سائيز / WARP_سائيز),
                cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * بلاڪ_dims, بلاڪ_dims),
                [=](sycl::nd_شي<3> شي_ct1)
                [[sycl::reqd_sub_گروهه_سائيز(WARP_سائيز)]] {
                    rms_norm_f32(x, dst, ncols,
                        src_stride_کانو, src_stride_صف, src_stride_channel, src_stride_sample,
                        dst_stride_کانو, dst_stride_صف, dst_stride_channel, dst_stride_sample,
                        eps, شي_ct1, get_اشارو(s_جوڙ_acc_ct1), work_گروهه_سائيز);
                });
            });
    }
}

ساكن فراغ rms_norm_scale_f32_sycl(ثابت عائم* x, عائم* dst, ثابت صحيح ncols, ثابت صحيح nrows,
    ثابت صحيح nchannels, ثابت صحيح nsamples,
    ثابت int64_t src_stride_کانو, ثابت int64_t src_stride_صف, ثابت int64_t src_stride_channel, ثابت int64_t src_stride_sample,
    ثابت int64_t dst_stride_کانو, ثابت int64_t dst_stride_صف, ثابت int64_t dst_stride_channel, ثابت int64_t dst_stride_sample,
    ثابت عائم eps, ثابت عائم scale_mul, قطار_ptr stream, صحيح device) {
    ثابت sycl::range<3> global_dims(nsamples, nchannels, nrows);
    إذا (ncols < 1024) {
        ثابت sycl::range<3> بلاڪ_dims(1, 1, WARP_سائيز);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * بلاڪ_dims, بلاڪ_dims),
                [=](sycl::nd_شي<3> شي_ct1)
                [[sycl::reqd_sub_گروهه_سائيز(WARP_سائيز)]] {
                    rms_norm_f32(x, dst, ncols,
                        src_stride_کانو, src_stride_صف, src_stride_channel, src_stride_sample,
                        dst_stride_کانو, dst_stride_صف, dst_stride_channel, dst_stride_sample,
                        eps, شي_ct1, خالي_اشارو, WARP_سائيز,
                        خالي_اشارو, 0, 0, 0, 0, 0, 0,
                        خالي_اشارو, 0, 0, 0, 0, 0, 0,
                        scale_mul);
                });
            });
    }
    وإلا {
        ثابت صحيح work_گروهه_سائيز = ggml_sycl_info().max_work_گروهه_sizes[device];
        أكد(work_گروهه_سائيز % (WARP_سائيز * WARP_سائيز) == 0);
        ثابت sycl::range<3> بلاڪ_dims(1, 1, work_گروهه_سائيز);
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<عائم, 1> s_جوڙ_acc_ct1(sycl::range<1>(work_گروهه_سائيز / WARP_سائيز),
                cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * بلاڪ_dims, بلاڪ_dims),
                [=](sycl::nd_شي<3> شي_ct1)
                [[sycl::reqd_sub_گروهه_سائيز(WARP_سائيز)]] {
                    rms_norm_f32(x, dst, ncols,
                        src_stride_کانو, src_stride_صف, src_stride_channel, src_stride_sample,
                        dst_stride_کانو, dst_stride_صف, dst_stride_channel, dst_stride_sample,
                        eps, شي_ct1, get_اشارو(s_جوڙ_acc_ct1), work_گروهه_سائيز,
                        خالي_اشارو, 0, 0, 0, 0, 0, 0,
                        خالي_اشارو, 0, 0, 0, 0, 0, 0,
                        scale_mul);
                });
            });
    }
}

ساكن فراغ rms_norm_mul_f32_sycl(ثابت عائم* x, ثابت عائم* mul, عائم* dst, ثابت صحيح ncols, ثابت صحيح nrows,
        ثابت صحيح nchannels, ثابت صحيح nsamples,
        ثابت int64_t src_stride_کانو, ثابت int64_t src_stride_صف, ثابت int64_t src_stride_channel, ثابت int64_t src_stride_sample,
        ثابت int64_t dst_stride_کانو, ثابت int64_t dst_stride_صف, ثابت int64_t dst_stride_channel, ثابت int64_t dst_stride_sample,
        ثابت int64_t mul_stride_صف, ثابت int64_t mul_stride_channel, ثابت int64_t mul_stride_sample,
        ثابت صحيح mul_nrows, ثابت صحيح mul_nchannels, ثابت صحيح mul_nsamples,
        ثابت عائم eps, قطار_ptr stream, صحيح device) {
    ثابت sycl::range<3> global_dims(nsamples, nchannels, nrows);
    إذا (ncols < 1024) {
        ثابت sycl::range<3> بلاڪ_dims(1, 1, WARP_سائيز);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * بلاڪ_dims, بلاڪ_dims),
                [=](sycl::nd_شي<3> شي_ct1)
                [[sycl::reqd_sub_گروهه_سائيز(WARP_سائيز)]] {
                    rms_norm_f32<سچ>(x, dst, ncols,
                        src_stride_کانو, src_stride_صف, src_stride_channel, src_stride_sample,
                        dst_stride_کانو, dst_stride_صف, dst_stride_channel, dst_stride_sample,
                        eps, شي_ct1, خالي_اشارو, WARP_سائيز,
                        mul, mul_stride_صف, mul_stride_channel, mul_stride_sample, mul_nrows, mul_nchannels, mul_nsamples);
                });
            });
    }
    وإلا {
        ثابت صحيح work_گروهه_سائيز = ggml_sycl_info().max_work_گروهه_sizes[device];
        أكد(work_گروهه_سائيز % (WARP_سائيز * WARP_سائيز) == 0);
        ثابت sycl::range<3> بلاڪ_dims(1, 1, work_گروهه_سائيز);
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<عائم, 1> s_جوڙ_acc_ct1(sycl::range<1>(work_گروهه_سائيز / WARP_سائيز), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * بلاڪ_dims, بلاڪ_dims),
                [=](sycl::nd_شي<3> شي_ct1)
                [[sycl::reqd_sub_گروهه_سائيز(WARP_سائيز)]] {
                    rms_norm_f32<سچ>(x, dst, ncols,
                        src_stride_کانو, src_stride_صف, src_stride_channel, src_stride_sample,
                        dst_stride_کانو, dst_stride_صف, dst_stride_channel, dst_stride_sample,
                        eps, شي_ct1, get_اشارو(s_جوڙ_acc_ct1), work_گروهه_سائيز,
                        mul, mul_stride_صف, mul_stride_channel, mul_stride_sample, mul_nrows, mul_nchannels, mul_nsamples);
                });
            });
    }
}

ساكن فراغ rms_norm_mul_جوڙيو_f32_sycl(ثابت عائم* x, ثابت عائم* mul, ثابت عائم* جوڙيو_2, عائم* dst,
        ثابت صحيح ncols, ثابت صحيح nrows, ثابت صحيح nchannels, ثابت صحيح nsamples,
        ثابت int64_t src_stride_کانو, ثابت int64_t src_stride_صف, ثابت int64_t src_stride_channel, ثابت int64_t src_stride_sample,
        ثابت int64_t dst_stride_کانو, ثابت int64_t dst_stride_صف, ثابت int64_t dst_stride_channel, ثابت int64_t dst_stride_sample,
        ثابت int64_t mul_stride_صف, ثابت int64_t mul_stride_channel, ثابت int64_t mul_stride_sample,
        ثابت صحيح mul_nrows, ثابت صحيح mul_nchannels, ثابت صحيح mul_nsamples,
        ثابت int64_t جوڙيو_stride_صف, ثابت int64_t جوڙيو_stride_channel, ثابت int64_t جوڙيو_stride_sample,
        ثابت صحيح جوڙيو_nrows, ثابت صحيح جوڙيو_nchannels, ثابت صحيح جوڙيو_nsamples,
        ثابت عائم eps, قطار_ptr stream, صحيح device) {
    ثابت sycl::range<3> global_dims(nsamples, nchannels, nrows);
    إذا (ncols < 1024) {
        ثابت sycl::range<3> بلاڪ_dims(1, 1, WARP_سائيز);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * بلاڪ_dims, بلاڪ_dims),
                [=](sycl::nd_شي<3> شي_ct1)
                [[sycl::reqd_sub_گروهه_سائيز(WARP_سائيز)]] {
                    rms_norm_f32<سچ, سچ>(x, dst, ncols,
                        src_stride_کانو, src_stride_صف, src_stride_channel, src_stride_sample,
                        dst_stride_کانو, dst_stride_صف, dst_stride_channel, dst_stride_sample,
                        eps, شي_ct1, خالي_اشارو, WARP_سائيز,
                        mul, mul_stride_صف, mul_stride_channel, mul_stride_sample, mul_nrows, mul_nchannels, mul_nsamples,
                        جوڙيو_2, جوڙيو_stride_صف, جوڙيو_stride_channel, جوڙيو_stride_sample, جوڙيو_nrows, جوڙيو_nchannels, جوڙيو_nsamples);
                });
            });
    }
    وإلا {
        ثابت صحيح work_گروهه_سائيز = ggml_sycl_info().max_work_گروهه_sizes[device];
        أكد(work_گروهه_سائيز % (WARP_سائيز * WARP_سائيز) == 0);
        ثابت sycl::range<3> بلاڪ_dims(1, 1, work_گروهه_سائيز);
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<عائم, 1> s_جوڙ_acc_ct1(sycl::range<1>(work_گروهه_سائيز / WARP_سائيز), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * بلاڪ_dims, بلاڪ_dims),
                [=](sycl::nd_شي<3> شي_ct1)
                [[sycl::reqd_sub_گروهه_سائيز(WARP_سائيز)]] {
                    rms_norm_f32<سچ, سچ>(x, dst, ncols,
                        src_stride_کانو, src_stride_صف, src_stride_channel, src_stride_sample,
                        dst_stride_کانو, dst_stride_صف, dst_stride_channel, dst_stride_sample,
                        eps, شي_ct1, get_اشارو(s_جوڙ_acc_ct1), work_گروهه_سائيز,
                        mul, mul_stride_صف, mul_stride_channel, mul_stride_sample, mul_nrows, mul_nchannels, mul_nsamples,
                        جوڙيو_2, جوڙيو_stride_صف, جوڙيو_stride_channel, جوڙيو_stride_sample, جوڙيو_nrows, جوڙيو_nchannels, جوڙيو_nsamples);
                });
            });
    }
}

سانچو<صحيح warp_سائيز>
ساكن فراغ l2_norm_f32_sycl(ثابت عائم *   x,
                             عائم *         dst,
                             ثابت صحيح       ncols,
                             ثابت صحيح       nrows,
                             ثابت صحيح       nchannels,
                             ثابت صحيح       nsamples,
                             ثابت int64_t   src_stride_کانو,
                             ثابت int64_t   src_stride_صف,
                             ثابت int64_t   src_stride_channel,
                             ثابت int64_t   src_stride_sample,
                             ثابت int64_t   dst_stride_کانو,
                             ثابت int64_t   dst_stride_صف,
                             ثابت int64_t   dst_stride_channel,
                             ثابت int64_t   dst_stride_sample,
                             ثابت عائم     eps,
                             قطار_ptr       stream,
                             صحيح             device) {
    ثابت dpct::dim3 بلاڪون_انگ(nrows, nchannels, nsamples);

    إذا (ncols < 1024) {
        ثابت dpct::dim3 بلاڪ_dims(warp_سائيز, 1, 1);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(بلاڪون_انگ * بلاڪ_dims,
                    بلاڪ_dims),
                [=](sycl::nd_شي<3> شي_ct1)
                [[sycl::reqd_sub_گروهه_سائيز(warp_سائيز)]] {
                    l2_norm_f32<warp_سائيز>(x, dst, ncols,
                        src_stride_کانو, src_stride_صف, src_stride_channel, src_stride_sample,
                        dst_stride_کانو, dst_stride_صف, dst_stride_channel, dst_stride_sample,
                        eps, شي_ct1,
                        خالي_اشارو, warp_سائيز);
                });
            });
    }
    وإلا {
        ثابت صحيح work_گروهه_سائيز = ggml_sycl_info().max_work_گروهه_sizes[device];
        أكد(work_گروهه_سائيز % (warp_سائيز * warp_سائيز) == 0);
        ثابت sycl::range<3> بلاڪ_dims(1, 1, work_گروهه_سائيز);
        صحيح lsm_سائيز =  بلاڪ_dims[2] > warp_سائيز ? work_گروهه_سائيز / warp_سائيز * حجم(عائم): 0;
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<عائم, 1> s_جوڙ_acc_ct1(sycl::range<1>(lsm_سائيز),
                cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(بلاڪون_انگ * بلاڪ_dims,
                    بلاڪ_dims),
                [=](sycl::nd_شي<3> شي_ct1)
                [[sycl::reqd_sub_گروهه_سائيز(warp_سائيز)]] {
                    l2_norm_f32<warp_سائيز>(x, dst, ncols,
                        src_stride_کانو, src_stride_صف, src_stride_channel, src_stride_sample,
                        dst_stride_کانو, dst_stride_صف, dst_stride_channel, dst_stride_sample,
                        eps, شي_ct1, get_اشارو(s_جوڙ_acc_ct1), work_گروهه_سائيز);
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

سانچو <صحيح warp_سائيز>
ساكن فراغ l2_norm_f32_batch(l2_batch_ptrs p, l2_batch_strides st, ثابت صحيح ncols, ثابت عائم eps,
                              ثابت sycl::nd_شي<3> & شي_ct1) {
    ثابت صحيح t   = شي_ct1.get_گروهه(0);  // tensor index
    ثابت صحيح r   = شي_ct1.get_گروهه(2);  // flattened row over ne1*ne2*ne3
    ثابت صحيح tid = شي_ct1.get_local_id(2);

    ثابت صحيح i1 = r % st.ne1;
    ثابت صحيح i2 = (r / st.ne1) % st.ne2;
    ثابت صحيح i3 = r / (st.ne1 * st.ne2);

    ثابت عائم * x   = p.src[t] + i3 * st.ss3 + i2 * st.ss2 + i1 * st.ss1;
    عائم *       dst = p.dst[t] + i3 * st.ds3 + i2 * st.ds2 + i1 * st.ds1;

    عائم tmp = 0.0f;
    لكل (صحيح کانو = tid; کانو < ncols; کانو += warp_سائيز) {
        ثابت عائم xi = x[کانو * st.ss0];
        tmp += xi * xi;
    }
    tmp = بلاڪ_reduce<بلاڪ_reduce_طريقو::جوڙ_3, warp_سائيز>(tmp, (عائم *) خالي_اشارو, warp_سائيز);
    ثابت عائم scale = sycl::rsqrt(sycl::fmax(tmp, eps * eps));
    لكل (صحيح کانو = tid; کانو < ncols; کانو += warp_سائيز) {
        dst[کانو * st.ds0] = scale * x[کانو * st.ss0];
    }
}

سانچو <صحيح warp_سائيز>
ساكن فراغ l2_norm_f32_batch_sycl(l2_batch_ptrs p, l2_batch_strides st, ثابت صحيح n_tensors,
                                   ثابت صحيح ncols, ثابت صحيح nrows_ڪل, ثابت عائم eps,
                                   قطار_ptr stream) {
    ثابت dpct::dim3 بلاڪون_انگ(nrows_ڪل, 1, n_tensors);
    ثابت dpct::dim3 بلاڪ_dims(warp_سائيز, 1, 1);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(بلاڪون_انگ * بلاڪ_dims, بلاڪ_dims),
            [=](sycl::nd_شي<3> شي_ct1) [[sycl::reqd_sub_گروهه_سائيز(warp_سائيز)]] {
                l2_norm_f32_batch<warp_سائيز>(p, st, ncols, eps, شي_ct1);
            });
    });
}

فراغ ggml_sycl_op_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    ثابت ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS
    dpct::قطار_ptr main_وهڪرو = ctx.stream();
    SYCL_CHECK(ggml_sycl_سيٽ_device(ctx.device));
    ثابت عائم * src0_dd = static_cast<ثابت عائم *>(dst->src[0]->معلومات);
    عائم *       dst_dd  = static_cast<عائم *>(dst->معلومات);

    عائم eps;
    يادگيري_نقل(&eps, dst->op_params, حجم(عائم));
    GGML_ASSERT(eps >= 0.0f);
    ثابت size_t ts0 = ggml_type_سائيز(src0->type);
    ثابت size_t tdst = ggml_type_سائيز(dst->type);
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
        ss0, ss1, ss2, ss3, ds0, ds1, ds2, ds3, eps, main_وهڪرو, ctx.device);
}

فراغ ggml_sycl_op_گروهه_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    صحيح انگ_groups = dst->op_params[0];
    dpct::قطار_ptr main_وهڪرو = ctx.stream();
    SYCL_CHECK(ggml_sycl_سيٽ_device(ctx.device));

    ثابت عائم * src0_dd = static_cast<ثابت عائم *>(dst->src[0]->معلومات);
    عائم *       dst_dd  = static_cast<عائم *>(dst->معلومات);

    عائم eps;
    يادگيري_نقل(&eps, dst->op_params + 1, حجم(عائم));

    صحيح گروهه_سائيز = dst->src[0]->ne[0] * dst->src[0]->ne[1] * ((dst->src[0]->ne[2] + انگ_groups - 1) / انگ_groups);
    گروهه_norm_f32_sycl(src0_dd, dst_dd, انگ_groups, eps, گروهه_سائيز, dst->src[0]->ne[0] * dst->src[0]->ne[1] * dst->src[0]->ne[2], main_وهڪرو, ctx.device);
}

فراغ ggml_sycl_op_rms_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    dpct::قطار_ptr main_وهڪرو = ctx.stream();
    SYCL_CHECK(ggml_sycl_سيٽ_device(ctx.device));

    ثابت عائم * src0_dd = static_cast<ثابت عائم *>(dst->src[0]->معلومات);
    عائم *       dst_dd  = static_cast<عائم *>(dst->معلومات);

    عائم eps;
    يادگيري_نقل(&eps, dst->op_params, حجم(عائم));

    GGML_TENSOR_UNARY_OP_LOCALS
    ثابت size_t ts0 = ggml_type_سائيز(src0->type);
    ثابت size_t tdst = ggml_type_سائيز(dst->type);
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
        ss0, ss1, ss2, ss3, ds0, ds1, ds2, ds3, eps, main_وهڪرو, ctx.device);
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

    dpct::قطار_ptr main_وهڪرو = ctx.stream();
    SYCL_CHECK(ggml_sycl_سيٽ_device(ctx.device));

    ثابت عائم * src0_dd = static_cast<ثابت عائم *>(src0->معلومات);
    عائم *       dst_dd  = static_cast<عائم *>(scale_tensor->معلومات);

    عائم eps;
    يادگيري_نقل(&eps, dst->op_params, حجم(عائم));
    عائم scale_mul;
    يادگيري_نقل(&scale_mul, scale_tensor->op_params, حجم(عائم));
    GGML_ASSERT(scale_mul >= 0.0f);

    GGML_TENSOR_UNARY_OP_LOCALS
    ثابت size_t ts0 = ggml_type_سائيز(src0->type);
    ثابت size_t tdst = ggml_type_سائيز(scale_tensor->type);
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
        ss0, ss1, ss2, ss3, ds0, ds1, ds2, ds3, eps, scale_mul, main_وهڪرو, ctx.device);
}

فراغ ggml_sycl_op_rms_norm_fused(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor) {
    ثابت ggml_tensor * rms_norm_src = dst->src[0];
    عائم eps = 0.0f;
    يادگيري_نقل(&eps, dst->op_params, حجم(عائم));

    ثابت عائم *       src0_dd = static_cast<ثابت عائم *>(rms_norm_src->معلومات);
    ثابت عائم *       mul_dd  = خالي_اشارو;
    ثابت ggml_tensor * mul_src = خالي_اشارو;
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

    dpct::قطار_ptr main_وهڪرو = ctx.stream();
    SYCL_CHECK(ggml_sycl_سيٽ_device(ctx.device));

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    ثابت int64_t ne00 = rms_norm_src->ne[0];
    ثابت int64_t ne01 = rms_norm_src->ne[1];
    ثابت int64_t ne02 = rms_norm_src->ne[2];
    ثابت int64_t ne03 = rms_norm_src->ne[3];

    ثابت size_t ts0 = ggml_type_سائيز(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    ثابت int64_t s00 = rms_norm_src->nb[0] / ts0;
    ثابت int64_t s01 = rms_norm_src->nb[1] / ts0;
    ثابت int64_t s02 = rms_norm_src->nb[2] / ts0;
    ثابت int64_t s03 = rms_norm_src->nb[3] / ts0;

    ثابت size_t tdst = ggml_type_سائيز(mul_tensor->type);
    GGML_ASSERT(mul_tensor->nb[0] == tdst);
    ثابت int64_t d00 = mul_tensor->nb[0] / tdst;
    ثابت int64_t d01 = mul_tensor->nb[1] / tdst;
    ثابت int64_t d02 = mul_tensor->nb[2] / tdst;
    ثابت int64_t d03 = mul_tensor->nb[3] / tdst;

    ثابت size_t ts_mul = ggml_type_سائيز(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    ثابت int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    ثابت int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    ثابت int64_t mul_s03 = mul_src->nb[3] / ts_mul;
    ثابت صحيح mul_nrows     = mul_src->ne[1];
    ثابت صحيح mul_nchannels = mul_src->ne[2];
    ثابت صحيح mul_nsamples  = mul_src->ne[3];

    rms_norm_mul_f32_sycl(src0_dd, mul_dd, dst_dd, ne00, ne01, ne02, ne03,
        s00, s01, s02, s03, d00, d01, d02, d03,
        mul_s01, mul_s02, mul_s03, mul_nrows, mul_nchannels, mul_nsamples, eps, main_وهڪرو, ctx.device);
}

فراغ ggml_sycl_op_rms_norm_fused_جوڙيو(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                     ggml_tensor * mul_tensor, ggml_tensor * جوڙيو_tensor) {
    ثابت ggml_tensor * rms_norm_src = dst->src[0];
    عائم eps = 0.0f;
    يادگيري_نقل(&eps, dst->op_params, حجم(عائم));

    ثابت عائم *       src0_dd = static_cast<ثابت عائم *>(rms_norm_src->معلومات);
    ثابت عائم *       mul_dd  = خالي_اشارو;
    ثابت ggml_tensor * mul_src = خالي_اشارو;
    إذا (mul_tensor->src[0] == dst) {
        mul_dd  = static_cast<ثابت عائم *>(mul_tensor->src[1]->معلومات);
        mul_src = mul_tensor->src[1];
    } وإلا إذا (mul_tensor->src[1] == dst) {
        mul_dd  = static_cast<ثابت عائم *>(mul_tensor->src[0]->معلومات);
        mul_src = mul_tensor->src[0];
    } وإلا {
        GGML_ASSERT(خطأ);
    }

    ثابت عائم *       جوڙيو_dd  = خالي_اشارو;
    ثابت ggml_tensor * جوڙيو_src = خالي_اشارو;
    إذا (جوڙيو_tensor->src[0] == mul_tensor) {
        جوڙيو_dd  = static_cast<ثابت عائم *>(جوڙيو_tensor->src[1]->معلومات);
        جوڙيو_src = جوڙيو_tensor->src[1];
    } وإلا إذا (جوڙيو_tensor->src[1] == mul_tensor) {
        جوڙيو_dd  = static_cast<ثابت عائم *>(جوڙيو_tensor->src[0]->معلومات);
        جوڙيو_src = جوڙيو_tensor->src[0];
    } وإلا {
        GGML_ASSERT(خطأ);
    }

    عائم * dst_dd = static_cast<عائم *>(جوڙيو_tensor->معلومات);

    dpct::قطار_ptr main_وهڪرو = ctx.stream();
    SYCL_CHECK(ggml_sycl_سيٽ_device(ctx.device));

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(جوڙيو_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    ثابت int64_t ne00 = rms_norm_src->ne[0];
    ثابت int64_t ne01 = rms_norm_src->ne[1];
    ثابت int64_t ne02 = rms_norm_src->ne[2];
    ثابت int64_t ne03 = rms_norm_src->ne[3];

    ثابت size_t ts0 = ggml_type_سائيز(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    ثابت int64_t s00 = rms_norm_src->nb[0] / ts0;
    ثابت int64_t s01 = rms_norm_src->nb[1] / ts0;
    ثابت int64_t s02 = rms_norm_src->nb[2] / ts0;
    ثابت int64_t s03 = rms_norm_src->nb[3] / ts0;

    ثابت size_t tdst = ggml_type_سائيز(جوڙيو_tensor->type);
    GGML_ASSERT(جوڙيو_tensor->nb[0] == tdst);
    ثابت int64_t d00 = جوڙيو_tensor->nb[0] / tdst;
    ثابت int64_t d01 = جوڙيو_tensor->nb[1] / tdst;
    ثابت int64_t d02 = جوڙيو_tensor->nb[2] / tdst;
    ثابت int64_t d03 = جوڙيو_tensor->nb[3] / tdst;

    ثابت size_t ts_mul = ggml_type_سائيز(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    ثابت int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    ثابت int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    ثابت int64_t mul_s03 = mul_src->nb[3] / ts_mul;
    ثابت صحيح mul_nrows     = mul_src->ne[1];
    ثابت صحيح mul_nchannels = mul_src->ne[2];
    ثابت صحيح mul_nsamples  = mul_src->ne[3];

    ثابت size_t ts_جوڙيو = ggml_type_سائيز(جوڙيو_src->type);
    GGML_ASSERT(جوڙيو_src->nb[0] == ts_جوڙيو);
    ثابت int64_t جوڙيو_s01 = جوڙيو_src->nb[1] / ts_جوڙيو;
    ثابت int64_t جوڙيو_s02 = جوڙيو_src->nb[2] / ts_جوڙيو;
    ثابت int64_t جوڙيو_s03 = جوڙيو_src->nb[3] / ts_جوڙيو;
    ثابت صحيح جوڙيو_nrows     = جوڙيو_src->ne[1];
    ثابت صحيح جوڙيو_nchannels = جوڙيو_src->ne[2];
    ثابت صحيح جوڙيو_nsamples  = جوڙيو_src->ne[3];

    rms_norm_mul_جوڙيو_f32_sycl(src0_dd, mul_dd, جوڙيو_dd, dst_dd, ne00, ne01, ne02, ne03,
        s00, s01, s02, s03, d00, d01, d02, d03,
        mul_s01, mul_s02, mul_s03, mul_nrows, mul_nchannels, mul_nsamples,
        جوڙيو_s01, جوڙيو_s02, جوڙيو_s03, جوڙيو_nrows, جوڙيو_nchannels, جوڙيو_nsamples, eps, main_وهڪرو, ctx.device);
}

فراغ ggml_sycl_op_rms_norm_پٺ(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32); // dz
    GGML_ASSERT(dst->src[1]->type == GGML_TYPE_F32); // x
    GGML_ASSERT(dst->type         == GGML_TYPE_F32);

    عائم eps = 1e-5f;
    std::يادگيري_نقل(&eps, dst->op_params, حجم(عائم));
    إذا (!(eps > 0.0f) || !std::isfinite(eps)) eps = 1e-5f;

    ثابت عائم * g_بنياد  = static_cast<ثابت عائم *>(dst->src[0]->معلومات); // dz
    ثابت عائم * x_بنياد  = static_cast<ثابت عائم *>(dst->src[1]->معلومات); // x
          عائم * dx_بنياد = static_cast<      عائم *>(dst->معلومات);

    ثابت int64_t D  = dst->ne[0];
    ثابت int64_t n1 = dst->ne[1], n2 = dst->ne[2], n3 = dst->ne[3]; (فراغ) n3;
    ثابت int64_t N  = ggml_nrows(dst);
    إذا (D == 0 || N == 0) أرجع;

    ثابت ggml_tensor *G = dst->src[0];
    ثابت ggml_tensor *X = dst->src[1];
    ثابت صحيح ts = (صحيح) ggml_type_سائيز(X->type);
    GGML_ASSERT((size_t) X->nb[0]   == (size_t) ts);
    GGML_ASSERT((size_t) G->nb[0]   == (size_t) ts);
    GGML_ASSERT((size_t) dst->nb[0] == (size_t) ts);

    ثابت int64_t xs1 = X->nb[1] / ts, xs2 = X->nb[2] / ts, xs3 = X->nb[3] / ts;
    ثابت int64_t gs1 = G->nb[1] / ts, gs2 = G->nb[2] / ts, gs3 = G->nb[3] / ts;
    ثابت int64_t ds1 = dst->nb[1] / ts, ds2 = dst->nb[2] / ts, ds3 = dst->nb[3] / ts;

    dpct::قطار_ptr q = ctx.stream();

    // work-group size: multiple of WARP_SIZE, capped by device and 256, and not larger than D
    ثابت صحيح device_max_wg = ggml_sycl_info().max_work_گروهه_sizes[ctx.device];
    خودڪار roundup = [](صحيح v, صحيح m) { أرجع ((v + m - 1) / m) * m; };
    صحيح wg_cap = 256;
    إذا (device_max_wg > 0) wg_cap = std::min(wg_cap, device_max_wg);
    صحيح WG = std::max(WARP_سائيز, std::min(roundup((صحيح)std::min<int64_t>(D, wg_cap), WARP_سائيز), wg_cap));

    // FP32 path: per-thread compensated accumulation + hierarchical reduction
    q->submit([&](sycl::handler &cgh) {
        ثابت صحيح nwarps_loc = std::max(1, WG / WARP_سائيز);
        // store one partial value per warp (xx and xg) for cross-warp reduction
        خودڪار l_xx   = sycl::local_accessor<sycl::float2, 1>(sycl::range<1>(nwarps_loc), cgh);
        خودڪار l_xg   = sycl::local_accessor<sycl::float2, 1>(sycl::range<1>(nwarps_loc), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, N) * sycl::range<3>(1, 1, WG),
                              sycl::range<3>(1, 1, WG)),
            [=](sycl::nd_شي<3> شي_ct1) [[sycl::reqd_sub_گروهه_سائيز(WARP_سائيز)]] {
                ثابت صحيح صف = شي_ct1.get_گروهه(2);
                ثابت صحيح tid = شي_ct1.get_local_id(2);

                ثابت int64_t i1 = صف % n1;
                ثابت int64_t i2 = (صف / n1) % n2;
                ثابت int64_t i3 = صف / (n1 * n2);

                ثابت عائم *__restrict x_صف = x_بنياد + i3 * xs3 + i2 * xs2 + i1 * xs1;
                ثابت عائم *__restrict g_صف = g_بنياد + i3 * gs3 + i2 * gs2 + i1 * gs1;
                عائم *__restrict d_صف       = dx_بنياد + i3 * ds3 + i2 * ds2 + i1 * ds1;

                // per-thread accumulation (compensated by default)
                عائم جوڙ_xx = 0.f, جوڙ_xg = 0.f;
#جيڪڏهن_وصف_نه_ٿيل GGML_SYCL_RMS_پٺ_FAST
                عائم c_xx = 0.f, c_xg = 0.f;
#جيڪڏهن_پورو
                لكل (int64_t کانو = tid; کانو < D; کانو += WG) {
                    ثابت عائم xv = x_صف[کانو];
                    ثابت عائم gv = g_صف[کانو];
#جيڪڏهن_وصف_ٿيل GGML_SYCL_RMS_پٺ_FAST
                    جوڙ_xx += xv * xv;
                    جوڙ_xg += xv * gv;
#وإلا
                    عائم y1 = xv * xv - c_xx;
                    عائم t1 = جوڙ_xx + y1;
                    c_xx = (t1 - جوڙ_xx) - y1;
                    جوڙ_xx = t1;

                    عائم y2 = xv * gv - c_xg;
                    عائم t2 = جوڙ_xg + y2;
                    c_xg = (t2 - جوڙ_xg) - y2;
                    جوڙ_xg = t2;
#جيڪڏهن_پورو
                }

                // warp-level reduction
                sycl::float2 xx = sycl::float2(جوڙ_xx,
#جيڪڏهن_وصف_نه_ٿيل GGML_SYCL_RMS_پٺ_FAST
                    c_xx
#وإلا
                    0.f
#جيڪڏهن_پورو
                );
                sycl::float2 xg = sycl::float2(جوڙ_xg,
#جيڪڏهن_وصف_نه_ٿيل GGML_SYCL_RMS_پٺ_FAST
                    c_xg
#وإلا
                    0.f
#جيڪڏهن_پورو
                );
                xx = warp_reduce_جوڙ(xx, شي_ct1);
                xg = warp_reduce_جوڙ(xg, شي_ct1);

                // cross-warp reduction using local memory (single barrier)
                ثابت خودڪار sub_گروهه = شي_ct1.get_sub_گروهه();
                ثابت خودڪار sg_id     = sub_گروهه.get_گروهه_لڪيري_id();
                ثابت خودڪار wi_اندريون_sg  = sub_گروهه.get_local_لڪيري_id();
                ثابت صحيح nthreads   = شي_ct1.get_local_range(2);
                ثابت صحيح nwarps     = nthreads / WARP_سائيز;

                sycl::float2 xx_ڪل = xx;
                sycl::float2 xg_ڪل = xg;
                إذا (nwarps > 1) {
                    إذا (wi_اندريون_sg == 0) {
                        l_xx[sg_id] = xx;
                        l_xg[sg_id] = xg;
                    }
                    شي_ct1.barrier(sycl::access::fence_space::local_space);

                    إذا (sg_id == 0) {
                        ثابت بنانشان wi_u = wi_اندريون_sg;
                        sycl::float2 xx_پهريون = (wi_u < static_cast<بنانشان>(nwarps)) ? l_xx[wi_u] : sycl::float2(0.f, 0.f);
                        sycl::float2 xg_پهريون = (wi_u < static_cast<بنانشان>(nwarps)) ? l_xg[wi_u] : sycl::float2(0.f, 0.f);
                        xx_ڪل = warp_reduce_جوڙ(xx_پهريون, شي_ct1);
                        xg_ڪل = warp_reduce_جوڙ(xg_پهريون, شي_ct1);
                    } وإلا {
                        // other subgroups keep their local totals; they'll be ignored
                        xx_ڪل = xx;
                        xg_ڪل = xg;
                    }
                    // ensure all threads see the first-subgroup result via broadcast below
                }

                // compute inv_r and coeff once per row and broadcast to the whole work-group
                عائم inv_r = 0.f;
                عائم coeff = 0.f;
                إذا (tid == 0) {
                    ثابت عائم جوڙ_xx_f  = xx_ڪل.x() + xx_ڪل.y();
                    ثابت عائم جوڙ_xdz_f = xg_ڪل.x() + xg_ڪل.y();
                    ثابت عائم سراسري_eps  = جوڙ_xx_f / (عائم) D + eps;
                    ثابت عائم جوڙ_eps   = جوڙ_xx_f + eps * (عائم) D;
                    inv_r = sycl::rsqrt(سراسري_eps);
                    coeff = -جوڙ_xdz_f / جوڙ_eps;
                }
                inv_r = sycl::گروهه_نشر(شي_ct1.get_گروهه(), inv_r);
                coeff = sycl::گروهه_نشر(شي_ct1.get_گروهه(), coeff);

                لكل (int64_t کانو = tid; کانو < D; کانو += WG) {
                    d_صف[کانو] = (g_صف[کانو] + coeff * x_صف[کانو]) * inv_r;
                }
            });
    });

}

فراغ ggml_sycl_op_l2_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت عائم * src0_d = (ثابت عائم *) src0->معلومات;
    عائم * dst_d = (عائم *) dst->معلومات;
    dpct::قطار_ptr     stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    عائم eps;
    يادگيري_نقل(&eps, dst->op_params, حجم(عائم));
    GGML_ASSERT(eps >= 0.0f);

    ثابت size_t ts0 = ggml_type_سائيز(src0->type);
    ثابت size_t tdst = ggml_type_سائيز(dst->type);
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
    l2_norm_f32_sycl<WARP_سائيز>(src0_d, dst_d, ne00, ne01, ne02, ne03,
            ss0, ss1, ss2, ss3, ds0, ds1, ds2, ds3, eps, stream, ctx.device);
}

// nodes[0..count) are independent, same-shape, same-eps, same-nb L2_NORM ops validated
// by the caller; requires ncols < 1024 (the warp reduction path).
فراغ ggml_sycl_l2_norm_batch(ggml_backend_sycl_context & ctx, ggml_tensor ** nodes, صحيح count) {
    ثابت ggml_tensor * s0 = nodes[0]->src[0];
    ثابت صحيح ncols       = (صحيح) s0->ne[0];
    ثابت صحيح nrows_ڪل = (صحيح) ggml_nrows(s0);
    عائم eps;
    يادگيري_نقل(&eps, nodes[0]->op_params, حجم(عائم));
    GGML_ASSERT(eps >= 0.0f);

    l2_batch_ptrs p{};
    لكل (صحيح t = 0; t < count; ++t) {
        p.src[t] = (ثابت عائم *) nodes[t]->src[0]->معلومات;
        p.dst[t] = (عائم *) nodes[t]->معلومات;
    }

    ثابت ggml_tensor * d0 = nodes[0];
    ثابت size_t        ts = ggml_type_سائيز(GGML_TYPE_F32);
    l2_batch_strides    st{};
    st.ne1 = (صحيح) s0->ne[1];
    st.ne2 = (صحيح) s0->ne[2];
    st.ss0 = s0->nb[0] / ts; st.ss1 = s0->nb[1] / ts; st.ss2 = s0->nb[2] / ts; st.ss3 = s0->nb[3] / ts;
    st.ds0 = d0->nb[0] / ts; st.ds1 = d0->nb[1] / ts; st.ds2 = d0->nb[2] / ts; st.ds3 = d0->nb[3] / ts;

    l2_norm_f32_batch_sycl<WARP_سائيز>(p, st, count, ncols, nrows_ڪل, eps, ctx.stream());
}
