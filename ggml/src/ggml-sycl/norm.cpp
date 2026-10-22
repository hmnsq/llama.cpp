#شامل "norm.hpp"
#شامل "ggml-sycl/common.hpp"
#شامل "ggml-sycl/presets.hpp"

ساکن خالی norm_f32(ثابت اعشاری* x, اعشاری* dst, ثابت صحیح ncols,
    ثابت int64_t src_stride_ستون, ثابت int64_t src_stride_صف, ثابت int64_t src_stride_channel, ثابت int64_t src_stride_sample,
    ثابت int64_t dst_stride_ستون, ثابت int64_t dst_stride_صف, ثابت int64_t dst_stride_channel, ثابت int64_t dst_stride_sample,
    ثابت اعشاری eps, ثابت sycl::nd_شے<3>& شے_ct1, sycl::float2* s_مجموع, صحیح بلاک_سائز_3) {

    ثابت صحیح nthreads = شے_ct1.get_local_range(2);
    ثابت صحیح sample  = شے_ct1.get_گروہ(0);
    ثابت صحیح channel = شے_ct1.get_گروہ(1);
    ثابت صحیح صف     = شے_ct1.get_گروہ(2);

    ثابت صحیح tid = شے_ct1.get_local_id(2);
    ثابت صحیح nwarps = nthreads / WARP_سائز;

    ثابت خودکار src_انحراف = calculate_انحراف<3>({src_stride_sample, src_stride_channel, src_stride_صف}, {sample, channel, صف});
    ثابت خودکار dst_انحراف = calculate_انحراف<3>({dst_stride_sample, dst_stride_channel, dst_stride_صف}, {sample, channel, صف});

    x += src_انحراف;
    dst += dst_انحراف;

    sycl::float2 اوسط_var = sycl::float2(0.f, 0.f);

    برائے (صحیح ستون = tid; ستون < ncols; ستون += بلاک_سائز_3) {
        ثابت اعشاری xi = x[ستون * src_stride_ستون];
        اوسط_var.x() += xi;
        اوسط_var.y() += xi * xi;
    }

    // sum up partial sums
    اوسط_var = warp_reduce_مجموع(اوسط_var, شے_ct1);
    اگر  (بلاک_سائز_3 > WARP_سائز) {
        ثابت خودکار sub_گروہ = شے_ct1.get_sub_گروہ();
        ثابت خودکار sg_id = sub_گروہ.get_گروہ_خطی_id();
        ثابت خودکار wi_اندر_sg = sub_گروہ.get_local_خطی_id();
        اگر (wi_اندر_sg == 0) {
            s_مجموع[sg_id] = اوسط_var;
        }
        شے_ct1.barrier(sycl::access::fence_space::local_space);
        اوسط_var = 0.f;
        ثابت size_t nreduce = ceil_div(nwarps, WARP_سائز);
        برائے (size_t i = 0; i < nreduce; i += 1)
        {
            اوسط_var += s_مجموع[wi_اندر_sg + i * WARP_سائز];
        }
        اوسط_var = warp_reduce_مجموع(اوسط_var, شے_ct1);
    }

    ثابت اعشاری اوسط_2 = اوسط_var.x() / ncols;
    ثابت اعشاری var = اوسط_var.y() / ncols - اوسط_2 * اوسط_2;
    ثابت اعشاری inv_std = sycl::rsqrt(var + eps);

    برائے (صحیح ستون = tid; ستون < ncols; ستون += بلاک_سائز_3) {
        dst[ستون * dst_stride_ستون] = (x[ستون * src_stride_ستون] - اوسط_2) * inv_std;
    }
}

ساکن خالی گروہ_norm_f32(ثابت اعشاری* x, اعشاری* dst, ثابت صحیح گروہ_سائز, ثابت صحیح ne_elements, ثابت اعشاری eps,
    ثابت sycl::nd_شے<3>& شے_ct1, اعشاری* s_مجموع, صحیح بلاک_سائز_3) {
    صحیح start = شے_ct1.get_گروہ(2) * گروہ_سائز;
    صحیح end = start + گروہ_سائز;
    ثابت صحیح nthreads = شے_ct1.get_local_range(2);
    ثابت صحیح nwarps = nthreads / WARP_سائز;
    start += شے_ct1.get_local_id(2);
    size_t nreduce = nwarps / WARP_سائز;

    اگر (end >= ne_elements) {
        end = ne_elements;
    }

    اعشاری tmp = 0.0f; // partial sum for thread in warp

    برائے (صحیح j = start; j < end; j += بلاک_سائز_3) {
        tmp += x[j];
    }

    tmp = warp_reduce_مجموع(tmp, شے_ct1);
    اگر (بلاک_سائز_3 > WARP_سائز) {

        صحیح warp_id = شے_ct1.get_local_id(2) / WARP_سائز;
        صحیح lane_id = شے_ct1.get_local_id(2) % WARP_سائز;
        اگر (lane_id == 0) {
            s_مجموع[warp_id] = tmp;
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
        شے_ct1.barrier();
        tmp = 0.f;
        برائے (size_t i = 0; i < nreduce; i += 1)
        {
            tmp += s_مجموع[lane_id + i * WARP_سائز];
        }
        tmp = warp_reduce_مجموع(tmp, شے_ct1);
    }

    اعشاری اوسط_2 = tmp / گروہ_سائز;
    tmp = 0.0f;

    برائے (صحیح j = start; j < end; j += بلاک_سائز_3) {
        اعشاری xi = x[j] - اوسط_2;
        dst[j] = xi;
        tmp += xi * xi;
    }

    tmp = warp_reduce_مجموع(tmp, شے_ct1);
    اگر (بلاک_سائز_3 > WARP_سائز) {

        صحیح warp_id = شے_ct1.get_local_id(2) / WARP_سائز;
        صحیح lane_id = شے_ct1.get_local_id(2) % WARP_سائز;
        اگر (lane_id == 0) {
            s_مجموع[warp_id] = tmp;
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
        شے_ct1.barrier();
        tmp = 0.f;
        برائے (size_t i = 0; i < nreduce; i += 1)
        {
            tmp += s_مجموع[lane_id + i * WARP_سائز];
        }
        tmp = warp_reduce_مجموع(tmp, شے_ct1);
    }

    اعشاری variance = tmp / گروہ_سائز;
    اعشاری scale = sycl::rsqrt(variance + eps);
    برائے (صحیح j = start; j < end; j += بلاک_سائز_3) {
        dst[j] *= scale;
    }
}

سانچہ <منطقی do_multiply = جھوٹ, منطقی do_جوڑو = جھوٹ>
ساکن خالی rms_norm_f32(ثابت اعشاری* x, اعشاری* dst, ثابت صحیح ncols,
    ثابت int64_t src_stride_ستون, ثابت int64_t src_stride_صف, ثابت int64_t src_stride_channel, ثابت int64_t src_stride_sample,
    ثابت int64_t dst_stride_ستون, ثابت int64_t dst_stride_صف, ثابت int64_t dst_stride_channel, ثابت int64_t dst_stride_sample,
    ثابت اعشاری eps, ثابت sycl::nd_شے<3>& شے_ct1, اعشاری* s_مجموع, صحیح بلاک_سائز_3,
    ثابت اعشاری* mul = خالی_اشاریہ, ثابت int64_t mul_stride_صف = 0, ثابت int64_t mul_stride_channel = 0,
    ثابت int64_t mul_stride_sample = 0, ثابت صحیح mul_nrows = 0, ثابت صحیح mul_nchannels = 0, ثابت صحیح mul_nsamples = 0,
    ثابت اعشاری* جوڑو_2 = خالی_اشاریہ, ثابت int64_t جوڑو_stride_صف = 0, ثابت int64_t جوڑو_stride_channel = 0,
    ثابت int64_t جوڑو_stride_sample = 0, ثابت صحیح جوڑو_nrows = 0, ثابت صحیح جوڑو_nchannels = 0, ثابت صحیح جوڑو_nsamples = 0,
    ثابت اعشاری scale_mul = 1.0f) {

    static_assert(!do_جوڑو || do_multiply, "fusing add is not supported without multiplying");

    ثابت صحیح sample  = شے_ct1.get_گروہ(0);
    ثابت صحیح channel = شے_ct1.get_گروہ(1);
    ثابت صحیح صف     = شے_ct1.get_گروہ(2);

    ثابت صحیح nthreads = شے_ct1.get_local_range(2);

    ثابت صحیح tid = شے_ct1.get_local_id(2);
    ثابت صحیح nwarps = nthreads / WARP_سائز;

    ثابت خودکار src_انحراف = calculate_انحراف<3>({src_stride_sample, src_stride_channel, src_stride_صف}, {sample, channel, صف});
    ثابت خودکار dst_انحراف = calculate_انحراف<3>({dst_stride_sample, dst_stride_channel, dst_stride_صف}, {sample, channel, صف});

    x   += src_انحراف;
    dst += dst_انحراف;

    اگر constexpr (do_multiply) {
        ثابت صحیح mul_صف     = صف     % mul_nrows;
        ثابت صحیح mul_channel = channel % mul_nchannels;
        ثابت صحیح mul_sample  = sample  % mul_nsamples;
        mul += mul_sample * mul_stride_sample + mul_channel * mul_stride_channel + mul_صف * mul_stride_صف;
    }

    اگر constexpr (do_جوڑو) {
        ثابت صحیح جوڑو_صف     = صف     % جوڑو_nrows;
        ثابت صحیح جوڑو_channel = channel % جوڑو_nchannels;
        ثابت صحیح جوڑو_sample  = sample  % جوڑو_nsamples;
        جوڑو_2 += جوڑو_sample * جوڑو_stride_sample + جوڑو_channel * جوڑو_stride_channel + جوڑو_صف * جوڑو_stride_صف;
    }

    اعشاری tmp = 0.0f; // partial sum for thread in warp

    برائے (صحیح ستون = tid; ستون < ncols; ستون += بلاک_سائز_3) {
        ثابت اعشاری xi = x[ستون * src_stride_ستون];
        tmp += xi * xi;
    }

    // sum up partial sums
    tmp = warp_reduce_مجموع(tmp, شے_ct1);
    اگر (بلاک_سائز_3 > WARP_سائز) {
        ثابت خودکار sub_گروہ = شے_ct1.get_sub_گروہ();
        ثابت خودکار sg_id = sub_گروہ.get_گروہ_خطی_id();
        ثابت خودکار wi_اندر_sg = sub_گروہ.get_local_خطی_id();
        اگر (wi_اندر_sg == 0) {
            s_مجموع[sg_id] = tmp;
        }

        شے_ct1.barrier(sycl::access::fence_space::local_space);
        ثابت size_t nreduce = ceil_div(nwarps, WARP_سائز);
        tmp = 0.f;
        برائے (size_t i = 0; i < nreduce; i += 1)
        {
            tmp += s_مجموع[wi_اندر_sg + i * WARP_سائز];
        }
        tmp = warp_reduce_مجموع(tmp, شے_ct1);
    }

    ثابت اعشاری اوسط_2 = tmp / ncols;
    ثابت اعشاری scale = sycl::rsqrt(اوسط_2 + eps);

    برائے (صحیح ستون = tid; ستون < ncols; ستون += بلاک_سائز_3) {
        اگر constexpr (do_multiply && do_جوڑو) {
            dst[ستون * dst_stride_ستون] = scale * x[ستون * src_stride_ستون] * mul[ستون] + جوڑو_2[ستون];
        } ورنہ اگر constexpr (do_multiply) {
            dst[ستون * dst_stride_ستون] = scale * x[ستون * src_stride_ستون] * mul[ستون];
        } ورنہ {
            // folded epilogue of a fused GGML_OP_SCALE consumer (qwen35 GDN l2 norms);
            // the explicit temporary keeps the float evaluation order identical to
            // running rms_norm and scale as two separate kernels
            ثابت اعشاری v = scale * x[ستون * src_stride_ستون];
            dst[ستون * dst_stride_ستون] = v * scale_mul;
        }
    }
}

سانچہ<صحیح warp_سائز>
ساکن خالی l2_norm_f32(ثابت اعشاری * x, اعشاری * dst, ثابت صحیح ncols,
    ثابت int64_t src_stride_ستون, ثابت int64_t src_stride_صف, ثابت int64_t src_stride_channel,
    ثابت int64_t src_stride_sample, ثابت int64_t dst_stride_ستون, ثابت int64_t dst_stride_صف,
    ثابت int64_t dst_stride_channel, ثابت int64_t dst_stride_sample, ثابت اعشاری eps,
    ثابت sycl::nd_شے<3>& شے_ct1, اعشاری* s_مجموع, ثابت صحیح بلاک_سائز_3) {

    ثابت صحیح صف     = شے_ct1.get_گروہ(2);
    ثابت صحیح channel = شے_ct1.get_گروہ(1);
    ثابت صحیح sample  = شے_ct1.get_گروہ(0);
    ثابت صحیح tid     = شے_ct1.get_local_id(2);

    x   += sample*src_stride_sample + channel*src_stride_channel + صف*src_stride_صف;
    dst += sample*dst_stride_sample + channel*dst_stride_channel + صف*dst_stride_صف;

    اعشاری tmp = 0.0f; // partial sum for thread in warp

    برائے (صحیح ستون = tid; ستون < ncols; ستون += بلاک_سائز_3) {
        ثابت اعشاری xi = x[ستون * src_stride_ستون];
        tmp += xi * xi;
    }

    tmp = بلاک_reduce<بلاک_reduce_طریقہ::مجموع_3, warp_سائز>(tmp, s_مجموع, بلاک_سائز_3);
    ثابت اعشاری scale = sycl::rsqrt(sycl::fmax(tmp, eps * eps));

    برائے (صحیح ستون = tid; ستون < ncols; ستون += بلاک_سائز_3) {
        dst[ستون * dst_stride_ستون] = scale * x[ستون * src_stride_ستون];
    }
}

ساکن خالی norm_f32_sycl(ثابت اعشاری * x, اعشاری * dst, ثابت صحیح ncols, ثابت صحیح nrows, ثابت صحیح nchannels, ثابت صحیح nsamples,
    ثابت int64_t src_stride_ستون, ثابت int64_t src_stride_صف, ثابت int64_t src_stride_channel, ثابت int64_t src_stride_sample,
    ثابت int64_t dst_stride_ستون, ثابت int64_t dst_stride_صف, ثابت int64_t dst_stride_channel, ثابت int64_t dst_stride_sample,
        ثابت اعشاری eps, قطار_ptr stream, صحیح device) {

    ثابت sycl::range<3> global_dims(nsamples, nchannels, nrows);
    اگر (ncols < 1024) {
        ثابت sycl::range<3> بلاک_dims(1, 1, WARP_سائز);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * بلاک_dims, بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1)
                [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                    norm_f32(x, dst, ncols,
                        src_stride_ستون, src_stride_صف, src_stride_channel, src_stride_sample,
                        dst_stride_ستون, dst_stride_صف, dst_stride_channel, dst_stride_sample,
                        eps, شے_ct1, خالی_اشاریہ, WARP_سائز);
                });
            });
    }
    ورنہ {
        ثابت صحیح work_گروہ_سائز = ggml_sycl_info().max_work_گروہ_sizes[device];
        أكد(work_گروہ_سائز % (WARP_سائز * WARP_سائز) == 0);
        ثابت sycl::range<3> بلاک_dims(1, 1, work_گروہ_سائز);
        /*
        DPCT1049:17: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<sycl::float2, 1> s_مجموع_acc_ct1(
                            sycl::range<1>(work_گروہ_سائز / WARP_سائز), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * بلاک_dims, بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1)
                [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                    norm_f32(x, dst, ncols,
                        src_stride_ستون, src_stride_صف, src_stride_channel, src_stride_sample,
                        dst_stride_ستون, dst_stride_صف, dst_stride_channel, dst_stride_sample,
                        eps, شے_ct1, get_اشارہ_گر(s_مجموع_acc_ct1), work_گروہ_سائز);
                });
            });
    }
}

ساکن خالی گروہ_norm_f32_sycl(ثابت اعشاری* x, اعشاری* dst,
    ثابت صحیح عدد_groups, ثابت اعشاری eps, ثابت صحیح گروہ_سائز,
    ثابت صحیح ne_elements, قطار_ptr stream, صحیح device) {
    اگر (گروہ_سائز < 1024) {
        ثابت sycl::range<3> بلاک_dims(1, 1, WARP_سائز);
        stream->submit([&](sycl::handler& cgh) {
            ثابت اعشاری eps_ct4 = eps;
            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, عدد_groups) * بلاک_dims,
                    بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1)
                [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                    گروہ_norm_f32(
                        x, dst, گروہ_سائز, ne_elements, eps_ct4, شے_ct1,
                        خالی_اشاریہ, WARP_سائز);
                });
            });
    }
    ورنہ {
        ثابت صحیح work_گروہ_سائز = ggml_sycl_info().max_work_گروہ_sizes[device];
        أكد(work_گروہ_سائز % (WARP_سائز * WARP_سائز) == 0);
        ثابت sycl::range<3> بلاک_dims(1, 1, work_گروہ_سائز);
        /*
        DPCT1049:18: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */

        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<اعشاری, 1> s_مجموع_acc_ct1(sycl::range<1>(work_گروہ_سائز / WARP_سائز),
                cgh);

            ثابت اعشاری eps_ct4 = eps;

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, عدد_groups) * بلاک_dims,
                    بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1)
                [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                    گروہ_norm_f32(x, dst, گروہ_سائز, ne_elements,
                        eps_ct4, شے_ct1,
                        get_اشارہ_گر(s_مجموع_acc_ct1), work_گروہ_سائز);
                });
            });
    }
}

ساکن خالی rms_norm_f32_sycl(ثابت اعشاری* x, اعشاری* dst, ثابت صحیح ncols, ثابت صحیح nrows, ثابت صحیح nchannels, ثابت صحیح nsamples,
    ثابت int64_t src_stride_ستون, ثابت int64_t src_stride_صف, ثابت int64_t src_stride_channel, ثابت int64_t src_stride_sample,
    ثابت int64_t dst_stride_ستون, ثابت int64_t dst_stride_صف, ثابت int64_t dst_stride_channel, ثابت int64_t dst_stride_sample,
    ثابت اعشاری eps, قطار_ptr stream, صحیح device) {
    // printf("%s ncols=%d, nrows=%d, WARP_SIZE=%d\n", __func__, ncols, nrows, WARP_SIZE);

    ثابت sycl::range<3> global_dims(nsamples, nchannels, nrows);
    اگر (ncols < 1024) {
        ثابت sycl::range<3> بلاک_dims(1, 1, WARP_سائز);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * بلاک_dims, بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1)
                [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                    rms_norm_f32(x, dst, ncols,
                        src_stride_ستون, src_stride_صف, src_stride_channel, src_stride_sample,
                        dst_stride_ستون, dst_stride_صف, dst_stride_channel, dst_stride_sample,
                        eps, شے_ct1, خالی_اشاریہ, WARP_سائز);
                });
            });
    }
    ورنہ {
        ثابت صحیح work_گروہ_سائز = ggml_sycl_info().max_work_گروہ_sizes[device];
        أكد(work_گروہ_سائز % (WARP_سائز * WARP_سائز) == 0);
        ثابت sycl::range<3> بلاک_dims(1, 1, work_گروہ_سائز);
        /*
        DPCT1049:19: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<اعشاری, 1> s_مجموع_acc_ct1(sycl::range<1>(work_گروہ_سائز / WARP_سائز),
                cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * بلاک_dims, بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1)
                [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                    rms_norm_f32(x, dst, ncols,
                        src_stride_ستون, src_stride_صف, src_stride_channel, src_stride_sample,
                        dst_stride_ستون, dst_stride_صف, dst_stride_channel, dst_stride_sample,
                        eps, شے_ct1, get_اشارہ_گر(s_مجموع_acc_ct1), work_گروہ_سائز);
                });
            });
    }
}

ساکن خالی rms_norm_scale_f32_sycl(ثابت اعشاری* x, اعشاری* dst, ثابت صحیح ncols, ثابت صحیح nrows,
    ثابت صحیح nchannels, ثابت صحیح nsamples,
    ثابت int64_t src_stride_ستون, ثابت int64_t src_stride_صف, ثابت int64_t src_stride_channel, ثابت int64_t src_stride_sample,
    ثابت int64_t dst_stride_ستون, ثابت int64_t dst_stride_صف, ثابت int64_t dst_stride_channel, ثابت int64_t dst_stride_sample,
    ثابت اعشاری eps, ثابت اعشاری scale_mul, قطار_ptr stream, صحیح device) {
    ثابت sycl::range<3> global_dims(nsamples, nchannels, nrows);
    اگر (ncols < 1024) {
        ثابت sycl::range<3> بلاک_dims(1, 1, WARP_سائز);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * بلاک_dims, بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1)
                [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                    rms_norm_f32(x, dst, ncols,
                        src_stride_ستون, src_stride_صف, src_stride_channel, src_stride_sample,
                        dst_stride_ستون, dst_stride_صف, dst_stride_channel, dst_stride_sample,
                        eps, شے_ct1, خالی_اشاریہ, WARP_سائز,
                        خالی_اشاریہ, 0, 0, 0, 0, 0, 0,
                        خالی_اشاریہ, 0, 0, 0, 0, 0, 0,
                        scale_mul);
                });
            });
    }
    ورنہ {
        ثابت صحیح work_گروہ_سائز = ggml_sycl_info().max_work_گروہ_sizes[device];
        أكد(work_گروہ_سائز % (WARP_سائز * WARP_سائز) == 0);
        ثابت sycl::range<3> بلاک_dims(1, 1, work_گروہ_سائز);
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<اعشاری, 1> s_مجموع_acc_ct1(sycl::range<1>(work_گروہ_سائز / WARP_سائز),
                cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * بلاک_dims, بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1)
                [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                    rms_norm_f32(x, dst, ncols,
                        src_stride_ستون, src_stride_صف, src_stride_channel, src_stride_sample,
                        dst_stride_ستون, dst_stride_صف, dst_stride_channel, dst_stride_sample,
                        eps, شے_ct1, get_اشارہ_گر(s_مجموع_acc_ct1), work_گروہ_سائز,
                        خالی_اشاریہ, 0, 0, 0, 0, 0, 0,
                        خالی_اشاریہ, 0, 0, 0, 0, 0, 0,
                        scale_mul);
                });
            });
    }
}

ساکن خالی rms_norm_mul_f32_sycl(ثابت اعشاری* x, ثابت اعشاری* mul, اعشاری* dst, ثابت صحیح ncols, ثابت صحیح nrows,
        ثابت صحیح nchannels, ثابت صحیح nsamples,
        ثابت int64_t src_stride_ستون, ثابت int64_t src_stride_صف, ثابت int64_t src_stride_channel, ثابت int64_t src_stride_sample,
        ثابت int64_t dst_stride_ستون, ثابت int64_t dst_stride_صف, ثابت int64_t dst_stride_channel, ثابت int64_t dst_stride_sample,
        ثابت int64_t mul_stride_صف, ثابت int64_t mul_stride_channel, ثابت int64_t mul_stride_sample,
        ثابت صحیح mul_nrows, ثابت صحیح mul_nchannels, ثابت صحیح mul_nsamples,
        ثابت اعشاری eps, قطار_ptr stream, صحیح device) {
    ثابت sycl::range<3> global_dims(nsamples, nchannels, nrows);
    اگر (ncols < 1024) {
        ثابت sycl::range<3> بلاک_dims(1, 1, WARP_سائز);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * بلاک_dims, بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1)
                [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                    rms_norm_f32<سچ>(x, dst, ncols,
                        src_stride_ستون, src_stride_صف, src_stride_channel, src_stride_sample,
                        dst_stride_ستون, dst_stride_صف, dst_stride_channel, dst_stride_sample,
                        eps, شے_ct1, خالی_اشاریہ, WARP_سائز,
                        mul, mul_stride_صف, mul_stride_channel, mul_stride_sample, mul_nrows, mul_nchannels, mul_nsamples);
                });
            });
    }
    ورنہ {
        ثابت صحیح work_گروہ_سائز = ggml_sycl_info().max_work_گروہ_sizes[device];
        أكد(work_گروہ_سائز % (WARP_سائز * WARP_سائز) == 0);
        ثابت sycl::range<3> بلاک_dims(1, 1, work_گروہ_سائز);
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<اعشاری, 1> s_مجموع_acc_ct1(sycl::range<1>(work_گروہ_سائز / WARP_سائز), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * بلاک_dims, بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1)
                [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                    rms_norm_f32<سچ>(x, dst, ncols,
                        src_stride_ستون, src_stride_صف, src_stride_channel, src_stride_sample,
                        dst_stride_ستون, dst_stride_صف, dst_stride_channel, dst_stride_sample,
                        eps, شے_ct1, get_اشارہ_گر(s_مجموع_acc_ct1), work_گروہ_سائز,
                        mul, mul_stride_صف, mul_stride_channel, mul_stride_sample, mul_nrows, mul_nchannels, mul_nsamples);
                });
            });
    }
}

ساکن خالی rms_norm_mul_جوڑو_f32_sycl(ثابت اعشاری* x, ثابت اعشاری* mul, ثابت اعشاری* جوڑو_2, اعشاری* dst,
        ثابت صحیح ncols, ثابت صحیح nrows, ثابت صحیح nchannels, ثابت صحیح nsamples,
        ثابت int64_t src_stride_ستون, ثابت int64_t src_stride_صف, ثابت int64_t src_stride_channel, ثابت int64_t src_stride_sample,
        ثابت int64_t dst_stride_ستون, ثابت int64_t dst_stride_صف, ثابت int64_t dst_stride_channel, ثابت int64_t dst_stride_sample,
        ثابت int64_t mul_stride_صف, ثابت int64_t mul_stride_channel, ثابت int64_t mul_stride_sample,
        ثابت صحیح mul_nrows, ثابت صحیح mul_nchannels, ثابت صحیح mul_nsamples,
        ثابت int64_t جوڑو_stride_صف, ثابت int64_t جوڑو_stride_channel, ثابت int64_t جوڑو_stride_sample,
        ثابت صحیح جوڑو_nrows, ثابت صحیح جوڑو_nchannels, ثابت صحیح جوڑو_nsamples,
        ثابت اعشاری eps, قطار_ptr stream, صحیح device) {
    ثابت sycl::range<3> global_dims(nsamples, nchannels, nrows);
    اگر (ncols < 1024) {
        ثابت sycl::range<3> بلاک_dims(1, 1, WARP_سائز);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * بلاک_dims, بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1)
                [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                    rms_norm_f32<سچ, سچ>(x, dst, ncols,
                        src_stride_ستون, src_stride_صف, src_stride_channel, src_stride_sample,
                        dst_stride_ستون, dst_stride_صف, dst_stride_channel, dst_stride_sample,
                        eps, شے_ct1, خالی_اشاریہ, WARP_سائز,
                        mul, mul_stride_صف, mul_stride_channel, mul_stride_sample, mul_nrows, mul_nchannels, mul_nsamples,
                        جوڑو_2, جوڑو_stride_صف, جوڑو_stride_channel, جوڑو_stride_sample, جوڑو_nrows, جوڑو_nchannels, جوڑو_nsamples);
                });
            });
    }
    ورنہ {
        ثابت صحیح work_گروہ_سائز = ggml_sycl_info().max_work_گروہ_sizes[device];
        أكد(work_گروہ_سائز % (WARP_سائز * WARP_سائز) == 0);
        ثابت sycl::range<3> بلاک_dims(1, 1, work_گروہ_سائز);
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<اعشاری, 1> s_مجموع_acc_ct1(sycl::range<1>(work_گروہ_سائز / WARP_سائز), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * بلاک_dims, بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1)
                [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                    rms_norm_f32<سچ, سچ>(x, dst, ncols,
                        src_stride_ستون, src_stride_صف, src_stride_channel, src_stride_sample,
                        dst_stride_ستون, dst_stride_صف, dst_stride_channel, dst_stride_sample,
                        eps, شے_ct1, get_اشارہ_گر(s_مجموع_acc_ct1), work_گروہ_سائز,
                        mul, mul_stride_صف, mul_stride_channel, mul_stride_sample, mul_nrows, mul_nchannels, mul_nsamples,
                        جوڑو_2, جوڑو_stride_صف, جوڑو_stride_channel, جوڑو_stride_sample, جوڑو_nrows, جوڑو_nchannels, جوڑو_nsamples);
                });
            });
    }
}

سانچہ<صحیح warp_سائز>
ساکن خالی l2_norm_f32_sycl(ثابت اعشاری *   x,
                             اعشاری *         dst,
                             ثابت صحیح       ncols,
                             ثابت صحیح       nrows,
                             ثابت صحیح       nchannels,
                             ثابت صحیح       nsamples,
                             ثابت int64_t   src_stride_ستون,
                             ثابت int64_t   src_stride_صف,
                             ثابت int64_t   src_stride_channel,
                             ثابت int64_t   src_stride_sample,
                             ثابت int64_t   dst_stride_ستون,
                             ثابت int64_t   dst_stride_صف,
                             ثابت int64_t   dst_stride_channel,
                             ثابت int64_t   dst_stride_sample,
                             ثابت اعشاری     eps,
                             قطار_ptr       stream,
                             صحیح             device) {
    ثابت dpct::dim3 بلاکیں_عدد(nrows, nchannels, nsamples);

    اگر (ncols < 1024) {
        ثابت dpct::dim3 بلاک_dims(warp_سائز, 1, 1);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(بلاکیں_عدد * بلاک_dims,
                    بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1)
                [[sycl::reqd_sub_گروہ_سائز(warp_سائز)]] {
                    l2_norm_f32<warp_سائز>(x, dst, ncols,
                        src_stride_ستون, src_stride_صف, src_stride_channel, src_stride_sample,
                        dst_stride_ستون, dst_stride_صف, dst_stride_channel, dst_stride_sample,
                        eps, شے_ct1,
                        خالی_اشاریہ, warp_سائز);
                });
            });
    }
    ورنہ {
        ثابت صحیح work_گروہ_سائز = ggml_sycl_info().max_work_گروہ_sizes[device];
        أكد(work_گروہ_سائز % (warp_سائز * warp_سائز) == 0);
        ثابت sycl::range<3> بلاک_dims(1, 1, work_گروہ_سائز);
        صحیح lsm_سائز =  بلاک_dims[2] > warp_سائز ? work_گروہ_سائز / warp_سائز * حجم(اعشاری): 0;
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<اعشاری, 1> s_مجموع_acc_ct1(sycl::range<1>(lsm_سائز),
                cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(بلاکیں_عدد * بلاک_dims,
                    بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1)
                [[sycl::reqd_sub_گروہ_سائز(warp_سائز)]] {
                    l2_norm_f32<warp_سائز>(x, dst, ncols,
                        src_stride_ستون, src_stride_صف, src_stride_channel, src_stride_sample,
                        dst_stride_ستون, dst_stride_صف, dst_stride_channel, dst_stride_sample,
                        eps, شے_ct1, get_اشارہ_گر(s_مجموع_acc_ct1), work_گروہ_سائز);
                });
            });
    }
}

// Batched L2 norm: N independent same-shape F32 tensors in one launch; the tensor
// index is folded into grid dim0 and each row's reduction is identical to the
// single-tensor kernel, so the result is bit-exact.
ساخت l2_batch_ptrs {
    ثابت اعشاری * src[GGML_SYCL_L2_BATCH_MAX];
    اعشاری *       dst[GGML_SYCL_L2_BATCH_MAX];
};

// One stride set shared by the whole batch: the caller only groups tensors whose nb[]
// all match, so per-tensor state stays two pointers.
ساخت l2_batch_strides {
    صحیح     ne1, ne2;
    int64_t ss0, ss1, ss2, ss3;
    int64_t ds0, ds1, ds2, ds3;
};

سانچہ <صحیح warp_سائز>
ساکن خالی l2_norm_f32_batch(l2_batch_ptrs p, l2_batch_strides st, ثابت صحیح ncols, ثابت اعشاری eps,
                              ثابت sycl::nd_شے<3> & شے_ct1) {
    ثابت صحیح t   = شے_ct1.get_گروہ(0);  // tensor index
    ثابت صحیح r   = شے_ct1.get_گروہ(2);  // flattened row over ne1*ne2*ne3
    ثابت صحیح tid = شے_ct1.get_local_id(2);

    ثابت صحیح i1 = r % st.ne1;
    ثابت صحیح i2 = (r / st.ne1) % st.ne2;
    ثابت صحیح i3 = r / (st.ne1 * st.ne2);

    ثابت اعشاری * x   = p.src[t] + i3 * st.ss3 + i2 * st.ss2 + i1 * st.ss1;
    اعشاری *       dst = p.dst[t] + i3 * st.ds3 + i2 * st.ds2 + i1 * st.ds1;

    اعشاری tmp = 0.0f;
    برائے (صحیح ستون = tid; ستون < ncols; ستون += warp_سائز) {
        ثابت اعشاری xi = x[ستون * st.ss0];
        tmp += xi * xi;
    }
    tmp = بلاک_reduce<بلاک_reduce_طریقہ::مجموع_3, warp_سائز>(tmp, (اعشاری *) خالی_اشاریہ, warp_سائز);
    ثابت اعشاری scale = sycl::rsqrt(sycl::fmax(tmp, eps * eps));
    برائے (صحیح ستون = tid; ستون < ncols; ستون += warp_سائز) {
        dst[ستون * st.ds0] = scale * x[ستون * st.ss0];
    }
}

سانچہ <صحیح warp_سائز>
ساکن خالی l2_norm_f32_batch_sycl(l2_batch_ptrs p, l2_batch_strides st, ثابت صحیح n_tensors,
                                   ثابت صحیح ncols, ثابت صحیح nrows_کل, ثابت اعشاری eps,
                                   قطار_ptr stream) {
    ثابت dpct::dim3 بلاکیں_عدد(nrows_کل, 1, n_tensors);
    ثابت dpct::dim3 بلاک_dims(warp_سائز, 1, 1);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(بلاکیں_عدد * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(warp_سائز)]] {
                l2_norm_f32_batch<warp_سائز>(p, st, ncols, eps, شے_ct1);
            });
    });
}

خالی ggml_sycl_op_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    ثابت ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS
    dpct::قطار_ptr main_دھارا = ctx.stream();
    SYCL_CHECK(ggml_sycl_سیٹ_device(ctx.device));
    ثابت اعشاری * src0_dd = static_cast<ثابت اعشاری *>(dst->src[0]->ڈیٹا);
    اعشاری *       dst_dd  = static_cast<اعشاری *>(dst->ڈیٹا);

    اعشاری eps;
    حافظہ_نقل(&eps, dst->op_params, حجم(اعشاری));
    GGML_ASSERT(eps >= 0.0f);
    ثابت size_t ts0 = ggml_type_سائز(src0->type);
    ثابت size_t tdst = ggml_type_سائز(dst->type);
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
        ss0, ss1, ss2, ss3, ds0, ds1, ds2, ds3, eps, main_دھارا, ctx.device);
}

خالی ggml_sycl_op_گروہ_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    صحیح عدد_groups = dst->op_params[0];
    dpct::قطار_ptr main_دھارا = ctx.stream();
    SYCL_CHECK(ggml_sycl_سیٹ_device(ctx.device));

    ثابت اعشاری * src0_dd = static_cast<ثابت اعشاری *>(dst->src[0]->ڈیٹا);
    اعشاری *       dst_dd  = static_cast<اعشاری *>(dst->ڈیٹا);

    اعشاری eps;
    حافظہ_نقل(&eps, dst->op_params + 1, حجم(اعشاری));

    صحیح گروہ_سائز = dst->src[0]->ne[0] * dst->src[0]->ne[1] * ((dst->src[0]->ne[2] + عدد_groups - 1) / عدد_groups);
    گروہ_norm_f32_sycl(src0_dd, dst_dd, عدد_groups, eps, گروہ_سائز, dst->src[0]->ne[0] * dst->src[0]->ne[1] * dst->src[0]->ne[2], main_دھارا, ctx.device);
}

خالی ggml_sycl_op_rms_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    dpct::قطار_ptr main_دھارا = ctx.stream();
    SYCL_CHECK(ggml_sycl_سیٹ_device(ctx.device));

    ثابت اعشاری * src0_dd = static_cast<ثابت اعشاری *>(dst->src[0]->ڈیٹا);
    اعشاری *       dst_dd  = static_cast<اعشاری *>(dst->ڈیٹا);

    اعشاری eps;
    حافظہ_نقل(&eps, dst->op_params, حجم(اعشاری));

    GGML_TENSOR_UNARY_OP_LOCALS
    ثابت size_t ts0 = ggml_type_سائز(src0->type);
    ثابت size_t tdst = ggml_type_سائز(dst->type);
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
        ss0, ss1, ss2, ss3, ds0, ds1, ds2, ds3, eps, main_دھارا, ctx.device);
}

// Fused rms_norm + scale (the qwen35 GDN l2-norm pair build_gdn_l2_norm emits):
// the scale factor is a host scalar in the GGML_OP_SCALE node's op_params, so
// unlike the mul variants there is no second device tensor to wire up.
خالی ggml_sycl_op_rms_norm_scale_fused(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                       ggml_tensor * scale_tensor) {
    ثابت ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(scale_tensor->type == GGML_TYPE_F32);

    dpct::قطار_ptr main_دھارا = ctx.stream();
    SYCL_CHECK(ggml_sycl_سیٹ_device(ctx.device));

    ثابت اعشاری * src0_dd = static_cast<ثابت اعشاری *>(src0->ڈیٹا);
    اعشاری *       dst_dd  = static_cast<اعشاری *>(scale_tensor->ڈیٹا);

    اعشاری eps;
    حافظہ_نقل(&eps, dst->op_params, حجم(اعشاری));
    اعشاری scale_mul;
    حافظہ_نقل(&scale_mul, scale_tensor->op_params, حجم(اعشاری));
    GGML_ASSERT(scale_mul >= 0.0f);

    GGML_TENSOR_UNARY_OP_LOCALS
    ثابت size_t ts0 = ggml_type_سائز(src0->type);
    ثابت size_t tdst = ggml_type_سائز(scale_tensor->type);
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
        ss0, ss1, ss2, ss3, ds0, ds1, ds2, ds3, eps, scale_mul, main_دھارا, ctx.device);
}

خالی ggml_sycl_op_rms_norm_fused(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor) {
    ثابت ggml_tensor * rms_norm_src = dst->src[0];
    اعشاری eps = 0.0f;
    حافظہ_نقل(&eps, dst->op_params, حجم(اعشاری));

    ثابت اعشاری *       src0_dd = static_cast<ثابت اعشاری *>(rms_norm_src->ڈیٹا);
    ثابت اعشاری *       mul_dd  = خالی_اشاریہ;
    ثابت ggml_tensor * mul_src = خالی_اشاریہ;
    اگر (mul_tensor->src[0] == dst) {
        mul_dd  = static_cast<ثابت اعشاری *>(mul_tensor->src[1]->ڈیٹا);
        mul_src = mul_tensor->src[1];
    } ورنہ اگر (mul_tensor->src[1] == dst) {
        mul_dd  = static_cast<ثابت اعشاری *>(mul_tensor->src[0]->ڈیٹا);
        mul_src = mul_tensor->src[0];
    } ورنہ {
        GGML_ASSERT(جھوٹ);
    }
    اعشاری * dst_dd = static_cast<اعشاری *>(mul_tensor->ڈیٹا);

    dpct::قطار_ptr main_دھارا = ctx.stream();
    SYCL_CHECK(ggml_sycl_سیٹ_device(ctx.device));

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    ثابت int64_t ne00 = rms_norm_src->ne[0];
    ثابت int64_t ne01 = rms_norm_src->ne[1];
    ثابت int64_t ne02 = rms_norm_src->ne[2];
    ثابت int64_t ne03 = rms_norm_src->ne[3];

    ثابت size_t ts0 = ggml_type_سائز(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    ثابت int64_t s00 = rms_norm_src->nb[0] / ts0;
    ثابت int64_t s01 = rms_norm_src->nb[1] / ts0;
    ثابت int64_t s02 = rms_norm_src->nb[2] / ts0;
    ثابت int64_t s03 = rms_norm_src->nb[3] / ts0;

    ثابت size_t tdst = ggml_type_سائز(mul_tensor->type);
    GGML_ASSERT(mul_tensor->nb[0] == tdst);
    ثابت int64_t d00 = mul_tensor->nb[0] / tdst;
    ثابت int64_t d01 = mul_tensor->nb[1] / tdst;
    ثابت int64_t d02 = mul_tensor->nb[2] / tdst;
    ثابت int64_t d03 = mul_tensor->nb[3] / tdst;

    ثابت size_t ts_mul = ggml_type_سائز(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    ثابت int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    ثابت int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    ثابت int64_t mul_s03 = mul_src->nb[3] / ts_mul;
    ثابت صحیح mul_nrows     = mul_src->ne[1];
    ثابت صحیح mul_nchannels = mul_src->ne[2];
    ثابت صحیح mul_nsamples  = mul_src->ne[3];

    rms_norm_mul_f32_sycl(src0_dd, mul_dd, dst_dd, ne00, ne01, ne02, ne03,
        s00, s01, s02, s03, d00, d01, d02, d03,
        mul_s01, mul_s02, mul_s03, mul_nrows, mul_nchannels, mul_nsamples, eps, main_دھارا, ctx.device);
}

خالی ggml_sycl_op_rms_norm_fused_جوڑو(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                     ggml_tensor * mul_tensor, ggml_tensor * جوڑو_tensor) {
    ثابت ggml_tensor * rms_norm_src = dst->src[0];
    اعشاری eps = 0.0f;
    حافظہ_نقل(&eps, dst->op_params, حجم(اعشاری));

    ثابت اعشاری *       src0_dd = static_cast<ثابت اعشاری *>(rms_norm_src->ڈیٹا);
    ثابت اعشاری *       mul_dd  = خالی_اشاریہ;
    ثابت ggml_tensor * mul_src = خالی_اشاریہ;
    اگر (mul_tensor->src[0] == dst) {
        mul_dd  = static_cast<ثابت اعشاری *>(mul_tensor->src[1]->ڈیٹا);
        mul_src = mul_tensor->src[1];
    } ورنہ اگر (mul_tensor->src[1] == dst) {
        mul_dd  = static_cast<ثابت اعشاری *>(mul_tensor->src[0]->ڈیٹا);
        mul_src = mul_tensor->src[0];
    } ورنہ {
        GGML_ASSERT(جھوٹ);
    }

    ثابت اعشاری *       جوڑو_dd  = خالی_اشاریہ;
    ثابت ggml_tensor * جوڑو_src = خالی_اشاریہ;
    اگر (جوڑو_tensor->src[0] == mul_tensor) {
        جوڑو_dd  = static_cast<ثابت اعشاری *>(جوڑو_tensor->src[1]->ڈیٹا);
        جوڑو_src = جوڑو_tensor->src[1];
    } ورنہ اگر (جوڑو_tensor->src[1] == mul_tensor) {
        جوڑو_dd  = static_cast<ثابت اعشاری *>(جوڑو_tensor->src[0]->ڈیٹا);
        جوڑو_src = جوڑو_tensor->src[0];
    } ورنہ {
        GGML_ASSERT(جھوٹ);
    }

    اعشاری * dst_dd = static_cast<اعشاری *>(جوڑو_tensor->ڈیٹا);

    dpct::قطار_ptr main_دھارا = ctx.stream();
    SYCL_CHECK(ggml_sycl_سیٹ_device(ctx.device));

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(جوڑو_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    ثابت int64_t ne00 = rms_norm_src->ne[0];
    ثابت int64_t ne01 = rms_norm_src->ne[1];
    ثابت int64_t ne02 = rms_norm_src->ne[2];
    ثابت int64_t ne03 = rms_norm_src->ne[3];

    ثابت size_t ts0 = ggml_type_سائز(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    ثابت int64_t s00 = rms_norm_src->nb[0] / ts0;
    ثابت int64_t s01 = rms_norm_src->nb[1] / ts0;
    ثابت int64_t s02 = rms_norm_src->nb[2] / ts0;
    ثابت int64_t s03 = rms_norm_src->nb[3] / ts0;

    ثابت size_t tdst = ggml_type_سائز(جوڑو_tensor->type);
    GGML_ASSERT(جوڑو_tensor->nb[0] == tdst);
    ثابت int64_t d00 = جوڑو_tensor->nb[0] / tdst;
    ثابت int64_t d01 = جوڑو_tensor->nb[1] / tdst;
    ثابت int64_t d02 = جوڑو_tensor->nb[2] / tdst;
    ثابت int64_t d03 = جوڑو_tensor->nb[3] / tdst;

    ثابت size_t ts_mul = ggml_type_سائز(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    ثابت int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    ثابت int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    ثابت int64_t mul_s03 = mul_src->nb[3] / ts_mul;
    ثابت صحیح mul_nrows     = mul_src->ne[1];
    ثابت صحیح mul_nchannels = mul_src->ne[2];
    ثابت صحیح mul_nsamples  = mul_src->ne[3];

    ثابت size_t ts_جوڑو = ggml_type_سائز(جوڑو_src->type);
    GGML_ASSERT(جوڑو_src->nb[0] == ts_جوڑو);
    ثابت int64_t جوڑو_s01 = جوڑو_src->nb[1] / ts_جوڑو;
    ثابت int64_t جوڑو_s02 = جوڑو_src->nb[2] / ts_جوڑو;
    ثابت int64_t جوڑو_s03 = جوڑو_src->nb[3] / ts_جوڑو;
    ثابت صحیح جوڑو_nrows     = جوڑو_src->ne[1];
    ثابت صحیح جوڑو_nchannels = جوڑو_src->ne[2];
    ثابت صحیح جوڑو_nsamples  = جوڑو_src->ne[3];

    rms_norm_mul_جوڑو_f32_sycl(src0_dd, mul_dd, جوڑو_dd, dst_dd, ne00, ne01, ne02, ne03,
        s00, s01, s02, s03, d00, d01, d02, d03,
        mul_s01, mul_s02, mul_s03, mul_nrows, mul_nchannels, mul_nsamples,
        جوڑو_s01, جوڑو_s02, جوڑو_s03, جوڑو_nrows, جوڑو_nchannels, جوڑو_nsamples, eps, main_دھارا, ctx.device);
}

خالی ggml_sycl_op_rms_norm_عقب(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32); // dz
    GGML_ASSERT(dst->src[1]->type == GGML_TYPE_F32); // x
    GGML_ASSERT(dst->type         == GGML_TYPE_F32);

    اعشاری eps = 1e-5f;
    std::حافظہ_نقل(&eps, dst->op_params, حجم(اعشاری));
    اگر (!(eps > 0.0f) || !std::isfinite(eps)) eps = 1e-5f;

    ثابت اعشاری * g_بنیاد  = static_cast<ثابت اعشاری *>(dst->src[0]->ڈیٹا); // dz
    ثابت اعشاری * x_بنیاد  = static_cast<ثابت اعشاری *>(dst->src[1]->ڈیٹا); // x
          اعشاری * dx_بنیاد = static_cast<      اعشاری *>(dst->ڈیٹا);

    ثابت int64_t D  = dst->ne[0];
    ثابت int64_t n1 = dst->ne[1], n2 = dst->ne[2], n3 = dst->ne[3]; (خالی) n3;
    ثابت int64_t N  = ggml_nrows(dst);
    اگر (D == 0 || N == 0) واپس;

    ثابت ggml_tensor *G = dst->src[0];
    ثابت ggml_tensor *X = dst->src[1];
    ثابت صحیح ts = (صحیح) ggml_type_سائز(X->type);
    GGML_ASSERT((size_t) X->nb[0]   == (size_t) ts);
    GGML_ASSERT((size_t) G->nb[0]   == (size_t) ts);
    GGML_ASSERT((size_t) dst->nb[0] == (size_t) ts);

    ثابت int64_t xs1 = X->nb[1] / ts, xs2 = X->nb[2] / ts, xs3 = X->nb[3] / ts;
    ثابت int64_t gs1 = G->nb[1] / ts, gs2 = G->nb[2] / ts, gs3 = G->nb[3] / ts;
    ثابت int64_t ds1 = dst->nb[1] / ts, ds2 = dst->nb[2] / ts, ds3 = dst->nb[3] / ts;

    dpct::قطار_ptr q = ctx.stream();

    // work-group size: multiple of WARP_SIZE, capped by device and 256, and not larger than D
    ثابت صحیح device_max_wg = ggml_sycl_info().max_work_گروہ_sizes[ctx.device];
    خودکار roundup = [](صحیح v, صحیح m) { واپس ((v + m - 1) / m) * m; };
    صحیح wg_cap = 256;
    اگر (device_max_wg > 0) wg_cap = std::min(wg_cap, device_max_wg);
    صحیح WG = std::max(WARP_سائز, std::min(roundup((صحیح)std::min<int64_t>(D, wg_cap), WARP_سائز), wg_cap));

    // FP32 path: per-thread compensated accumulation + hierarchical reduction
    q->submit([&](sycl::handler &cgh) {
        ثابت صحیح nwarps_loc = std::max(1, WG / WARP_سائز);
        // store one partial value per warp (xx and xg) for cross-warp reduction
        خودکار l_xx   = sycl::local_accessor<sycl::float2, 1>(sycl::range<1>(nwarps_loc), cgh);
        خودکار l_xg   = sycl::local_accessor<sycl::float2, 1>(sycl::range<1>(nwarps_loc), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, N) * sycl::range<3>(1, 1, WG),
                              sycl::range<3>(1, 1, WG)),
            [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                ثابت صحیح صف = شے_ct1.get_گروہ(2);
                ثابت صحیح tid = شے_ct1.get_local_id(2);

                ثابت int64_t i1 = صف % n1;
                ثابت int64_t i2 = (صف / n1) % n2;
                ثابت int64_t i3 = صف / (n1 * n2);

                ثابت اعشاری *__restrict x_صف = x_بنیاد + i3 * xs3 + i2 * xs2 + i1 * xs1;
                ثابت اعشاری *__restrict g_صف = g_بنیاد + i3 * gs3 + i2 * gs2 + i1 * gs1;
                اعشاری *__restrict d_صف       = dx_بنیاد + i3 * ds3 + i2 * ds2 + i1 * ds1;

                // per-thread accumulation (compensated by default)
                اعشاری مجموع_xx = 0.f, مجموع_xg = 0.f;
#اگر_غیر_تعریف_شدہ GGML_SYCL_RMS_عقب_FAST
                اعشاری c_xx = 0.f, c_xg = 0.f;
#اگر_ختم
                برائے (int64_t ستون = tid; ستون < D; ستون += WG) {
                    ثابت اعشاری xv = x_صف[ستون];
                    ثابت اعشاری gv = g_صف[ستون];
#اگر_تعریف_شدہ GGML_SYCL_RMS_عقب_FAST
                    مجموع_xx += xv * xv;
                    مجموع_xg += xv * gv;
#ورنہ
                    اعشاری y1 = xv * xv - c_xx;
                    اعشاری t1 = مجموع_xx + y1;
                    c_xx = (t1 - مجموع_xx) - y1;
                    مجموع_xx = t1;

                    اعشاری y2 = xv * gv - c_xg;
                    اعشاری t2 = مجموع_xg + y2;
                    c_xg = (t2 - مجموع_xg) - y2;
                    مجموع_xg = t2;
#اگر_ختم
                }

                // warp-level reduction
                sycl::float2 xx = sycl::float2(مجموع_xx,
#اگر_غیر_تعریف_شدہ GGML_SYCL_RMS_عقب_FAST
                    c_xx
#ورنہ
                    0.f
#اگر_ختم
                );
                sycl::float2 xg = sycl::float2(مجموع_xg,
#اگر_غیر_تعریف_شدہ GGML_SYCL_RMS_عقب_FAST
                    c_xg
#ورنہ
                    0.f
#اگر_ختم
                );
                xx = warp_reduce_مجموع(xx, شے_ct1);
                xg = warp_reduce_مجموع(xg, شے_ct1);

                // cross-warp reduction using local memory (single barrier)
                ثابت خودکار sub_گروہ = شے_ct1.get_sub_گروہ();
                ثابت خودکار sg_id     = sub_گروہ.get_گروہ_خطی_id();
                ثابت خودکار wi_اندر_sg  = sub_گروہ.get_local_خطی_id();
                ثابت صحیح nthreads   = شے_ct1.get_local_range(2);
                ثابت صحیح nwarps     = nthreads / WARP_سائز;

                sycl::float2 xx_کل = xx;
                sycl::float2 xg_کل = xg;
                اگر (nwarps > 1) {
                    اگر (wi_اندر_sg == 0) {
                        l_xx[sg_id] = xx;
                        l_xg[sg_id] = xg;
                    }
                    شے_ct1.barrier(sycl::access::fence_space::local_space);

                    اگر (sg_id == 0) {
                        ثابت غیرعلامتی wi_u = wi_اندر_sg;
                        sycl::float2 xx_پہلا = (wi_u < static_cast<غیرعلامتی>(nwarps)) ? l_xx[wi_u] : sycl::float2(0.f, 0.f);
                        sycl::float2 xg_پہلا = (wi_u < static_cast<غیرعلامتی>(nwarps)) ? l_xg[wi_u] : sycl::float2(0.f, 0.f);
                        xx_کل = warp_reduce_مجموع(xx_پہلا, شے_ct1);
                        xg_کل = warp_reduce_مجموع(xg_پہلا, شے_ct1);
                    } ورنہ {
                        // other subgroups keep their local totals; they'll be ignored
                        xx_کل = xx;
                        xg_کل = xg;
                    }
                    // ensure all threads see the first-subgroup result via broadcast below
                }

                // compute inv_r and coeff once per row and broadcast to the whole work-group
                اعشاری inv_r = 0.f;
                اعشاری coeff = 0.f;
                اگر (tid == 0) {
                    ثابت اعشاری مجموع_xx_f  = xx_کل.x() + xx_کل.y();
                    ثابت اعشاری مجموع_xdz_f = xg_کل.x() + xg_کل.y();
                    ثابت اعشاری اوسط_eps  = مجموع_xx_f / (اعشاری) D + eps;
                    ثابت اعشاری مجموع_eps   = مجموع_xx_f + eps * (اعشاری) D;
                    inv_r = sycl::rsqrt(اوسط_eps);
                    coeff = -مجموع_xdz_f / مجموع_eps;
                }
                inv_r = sycl::گروہ_نشر(شے_ct1.get_گروہ(), inv_r);
                coeff = sycl::گروہ_نشر(شے_ct1.get_گروہ(), coeff);

                برائے (int64_t ستون = tid; ستون < D; ستون += WG) {
                    d_صف[ستون] = (g_صف[ستون] + coeff * x_صف[ستون]) * inv_r;
                }
            });
    });

}

خالی ggml_sycl_op_l2_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت اعشاری * src0_d = (ثابت اعشاری *) src0->ڈیٹا;
    اعشاری * dst_d = (اعشاری *) dst->ڈیٹا;
    dpct::قطار_ptr     stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    اعشاری eps;
    حافظہ_نقل(&eps, dst->op_params, حجم(اعشاری));
    GGML_ASSERT(eps >= 0.0f);

    ثابت size_t ts0 = ggml_type_سائز(src0->type);
    ثابت size_t tdst = ggml_type_سائز(dst->type);
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
    l2_norm_f32_sycl<WARP_سائز>(src0_d, dst_d, ne00, ne01, ne02, ne03,
            ss0, ss1, ss2, ss3, ds0, ds1, ds2, ds3, eps, stream, ctx.device);
}

// nodes[0..count) are independent, same-shape, same-eps, same-nb L2_NORM ops validated
// by the caller; requires ncols < 1024 (the warp reduction path).
خالی ggml_sycl_l2_norm_batch(ggml_backend_sycl_context & ctx, ggml_tensor ** nodes, صحیح count) {
    ثابت ggml_tensor * s0 = nodes[0]->src[0];
    ثابت صحیح ncols       = (صحیح) s0->ne[0];
    ثابت صحیح nrows_کل = (صحیح) ggml_nrows(s0);
    اعشاری eps;
    حافظہ_نقل(&eps, nodes[0]->op_params, حجم(اعشاری));
    GGML_ASSERT(eps >= 0.0f);

    l2_batch_ptrs p{};
    برائے (صحیح t = 0; t < count; ++t) {
        p.src[t] = (ثابت اعشاری *) nodes[t]->src[0]->ڈیٹا;
        p.dst[t] = (اعشاری *) nodes[t]->ڈیٹا;
    }

    ثابت ggml_tensor * d0 = nodes[0];
    ثابت size_t        ts = ggml_type_سائز(GGML_TYPE_F32);
    l2_batch_strides    st{};
    st.ne1 = (صحیح) s0->ne[1];
    st.ne2 = (صحیح) s0->ne[2];
    st.ss0 = s0->nb[0] / ts; st.ss1 = s0->nb[1] / ts; st.ss2 = s0->nb[2] / ts; st.ss3 = s0->nb[3] / ts;
    st.ds0 = d0->nb[0] / ts; st.ds1 = d0->nb[1] / ts; st.ds2 = d0->nb[2] / ts; st.ds3 = d0->nb[3] / ts;

    l2_norm_f32_batch_sycl<WARP_سائز>(p, st, count, ncols, nrows_کل, eps, ctx.stream());
}
