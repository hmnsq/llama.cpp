#ضمن "norm.hpp"
#ضمن "ggml-sycl/common.hpp"
#ضمن "ggml-sycl/presets.hpp"

ساكن فراغ norm_f32(ثابت عائم* x, عائم* dst, ثابت صحيح ncols,
    ثابت int64_t src_stride_خانہ, ثابت int64_t src_stride_صف, ثابت int64_t src_stride_channel, ثابت int64_t src_stride_sample,
    ثابت int64_t dst_stride_خانہ, ثابت int64_t dst_stride_صف, ثابت int64_t dst_stride_channel, ثابت int64_t dst_stride_sample,
    ثابت عائم eps, ثابت sycl::nd_شے<3>& شے_ct1, sycl::float2* s_جوڑ, صحيح بلاک_ناپ_3) {

    ثابت صحيح nthreads = شے_ct1.get_local_range(2);
    ثابت صحيح sample  = شے_ct1.get_گروہ(0);
    ثابت صحيح channel = شے_ct1.get_گروہ(1);
    ثابت صحيح صف     = شے_ct1.get_گروہ(2);

    ثابت صحيح tid = شے_ct1.get_local_id(2);
    ثابت صحيح nwarps = nthreads / WARP_ناپ;

    ثابت خودکار src_وتھ = calculate_وتھ<3>({src_stride_sample, src_stride_channel, src_stride_صف}, {sample, channel, صف});
    ثابت خودکار dst_وتھ = calculate_وتھ<3>({dst_stride_sample, dst_stride_channel, dst_stride_صف}, {sample, channel, صف});

    x += src_وتھ;
    dst += dst_وتھ;

    sycl::float2 اوسط_var = sycl::float2(0.f, 0.f);

    لكل (صحيح خانہ = tid; خانہ < ncols; خانہ += بلاک_ناپ_3) {
        ثابت عائم xi = x[خانہ * src_stride_خانہ];
        اوسط_var.x() += xi;
        اوسط_var.y() += xi * xi;
    }

    // sum up partial sums
    اوسط_var = warp_reduce_جوڑ(اوسط_var, شے_ct1);
    إذا  (بلاک_ناپ_3 > WARP_ناپ) {
        ثابت خودکار sub_گروہ = شے_ct1.get_sub_گروہ();
        ثابت خودکار sg_id = sub_گروہ.get_گروہ_خطی_id();
        ثابت خودکار wi_وچ_sg = sub_گروہ.get_local_خطی_id();
        إذا (wi_وچ_sg == 0) {
            s_جوڑ[sg_id] = اوسط_var;
        }
        شے_ct1.barrier(sycl::access::fence_space::local_space);
        اوسط_var = 0.f;
        ثابت size_t nreduce = ceil_div(nwarps, WARP_ناپ);
        لكل (size_t i = 0; i < nreduce; i += 1)
        {
            اوسط_var += s_جوڑ[wi_وچ_sg + i * WARP_ناپ];
        }
        اوسط_var = warp_reduce_جوڑ(اوسط_var, شے_ct1);
    }

    ثابت عائم اوسط_2 = اوسط_var.x() / ncols;
    ثابت عائم var = اوسط_var.y() / ncols - اوسط_2 * اوسط_2;
    ثابت عائم inv_std = sycl::rsqrt(var + eps);

    لكل (صحيح خانہ = tid; خانہ < ncols; خانہ += بلاک_ناپ_3) {
        dst[خانہ * dst_stride_خانہ] = (x[خانہ * src_stride_خانہ] - اوسط_2) * inv_std;
    }
}

ساكن فراغ گروہ_norm_f32(ثابت عائم* x, عائم* dst, ثابت صحيح گروہ_ناپ, ثابت صحيح ne_elements, ثابت عائم eps,
    ثابت sycl::nd_شے<3>& شے_ct1, عائم* s_جوڑ, صحيح بلاک_ناپ_3) {
    صحيح start = شے_ct1.get_گروہ(2) * گروہ_ناپ;
    صحيح end = start + گروہ_ناپ;
    ثابت صحيح nthreads = شے_ct1.get_local_range(2);
    ثابت صحيح nwarps = nthreads / WARP_ناپ;
    start += شے_ct1.get_local_id(2);
    size_t nreduce = nwarps / WARP_ناپ;

    إذا (end >= ne_elements) {
        end = ne_elements;
    }

    عائم tmp = 0.0f; // partial sum for thread in warp

    لكل (صحيح j = start; j < end; j += بلاک_ناپ_3) {
        tmp += x[j];
    }

    tmp = warp_reduce_جوڑ(tmp, شے_ct1);
    إذا (بلاک_ناپ_3 > WARP_ناپ) {

        صحيح warp_id = شے_ct1.get_local_id(2) / WARP_ناپ;
        صحيح lane_id = شے_ct1.get_local_id(2) % WARP_ناپ;
        إذا (lane_id == 0) {
            s_جوڑ[warp_id] = tmp;
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
        لكل (size_t i = 0; i < nreduce; i += 1)
        {
            tmp += s_جوڑ[lane_id + i * WARP_ناپ];
        }
        tmp = warp_reduce_جوڑ(tmp, شے_ct1);
    }

    عائم اوسط_2 = tmp / گروہ_ناپ;
    tmp = 0.0f;

    لكل (صحيح j = start; j < end; j += بلاک_ناپ_3) {
        عائم xi = x[j] - اوسط_2;
        dst[j] = xi;
        tmp += xi * xi;
    }

    tmp = warp_reduce_جوڑ(tmp, شے_ct1);
    إذا (بلاک_ناپ_3 > WARP_ناپ) {

        صحيح warp_id = شے_ct1.get_local_id(2) / WARP_ناپ;
        صحيح lane_id = شے_ct1.get_local_id(2) % WARP_ناپ;
        إذا (lane_id == 0) {
            s_جوڑ[warp_id] = tmp;
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
        لكل (size_t i = 0; i < nreduce; i += 1)
        {
            tmp += s_جوڑ[lane_id + i * WARP_ناپ];
        }
        tmp = warp_reduce_جوڑ(tmp, شے_ct1);
    }

    عائم variance = tmp / گروہ_ناپ;
    عائم scale = sycl::rsqrt(variance + eps);
    لكل (صحيح j = start; j < end; j += بلاک_ناپ_3) {
        dst[j] *= scale;
    }
}

سانچا <منطقي do_multiply = خطأ, منطقي do_جوڑو = خطأ>
ساكن فراغ rms_norm_f32(ثابت عائم* x, عائم* dst, ثابت صحيح ncols,
    ثابت int64_t src_stride_خانہ, ثابت int64_t src_stride_صف, ثابت int64_t src_stride_channel, ثابت int64_t src_stride_sample,
    ثابت int64_t dst_stride_خانہ, ثابت int64_t dst_stride_صف, ثابت int64_t dst_stride_channel, ثابت int64_t dst_stride_sample,
    ثابت عائم eps, ثابت sycl::nd_شے<3>& شے_ct1, عائم* s_جوڑ, صحيح بلاک_ناپ_3,
    ثابت عائم* mul = خالی_سوچک, ثابت int64_t mul_stride_صف = 0, ثابت int64_t mul_stride_channel = 0,
    ثابت int64_t mul_stride_sample = 0, ثابت صحيح mul_nrows = 0, ثابت صحيح mul_nchannels = 0, ثابت صحيح mul_nsamples = 0,
    ثابت عائم* جوڑو_2 = خالی_سوچک, ثابت int64_t جوڑو_stride_صف = 0, ثابت int64_t جوڑو_stride_channel = 0,
    ثابت int64_t جوڑو_stride_sample = 0, ثابت صحيح جوڑو_nrows = 0, ثابت صحيح جوڑو_nchannels = 0, ثابت صحيح جوڑو_nsamples = 0,
    ثابت عائم scale_mul = 1.0f) {

    static_assert(!do_جوڑو || do_multiply, "fusing add is not supported without multiplying");

    ثابت صحيح sample  = شے_ct1.get_گروہ(0);
    ثابت صحيح channel = شے_ct1.get_گروہ(1);
    ثابت صحيح صف     = شے_ct1.get_گروہ(2);

    ثابت صحيح nthreads = شے_ct1.get_local_range(2);

    ثابت صحيح tid = شے_ct1.get_local_id(2);
    ثابت صحيح nwarps = nthreads / WARP_ناپ;

    ثابت خودکار src_وتھ = calculate_وتھ<3>({src_stride_sample, src_stride_channel, src_stride_صف}, {sample, channel, صف});
    ثابت خودکار dst_وتھ = calculate_وتھ<3>({dst_stride_sample, dst_stride_channel, dst_stride_صف}, {sample, channel, صف});

    x   += src_وتھ;
    dst += dst_وتھ;

    إذا constexpr (do_multiply) {
        ثابت صحيح mul_صف     = صف     % mul_nrows;
        ثابت صحيح mul_channel = channel % mul_nchannels;
        ثابت صحيح mul_sample  = sample  % mul_nsamples;
        mul += mul_sample * mul_stride_sample + mul_channel * mul_stride_channel + mul_صف * mul_stride_صف;
    }

    إذا constexpr (do_جوڑو) {
        ثابت صحيح جوڑو_صف     = صف     % جوڑو_nrows;
        ثابت صحيح جوڑو_channel = channel % جوڑو_nchannels;
        ثابت صحيح جوڑو_sample  = sample  % جوڑو_nsamples;
        جوڑو_2 += جوڑو_sample * جوڑو_stride_sample + جوڑو_channel * جوڑو_stride_channel + جوڑو_صف * جوڑو_stride_صف;
    }

    عائم tmp = 0.0f; // partial sum for thread in warp

    لكل (صحيح خانہ = tid; خانہ < ncols; خانہ += بلاک_ناپ_3) {
        ثابت عائم xi = x[خانہ * src_stride_خانہ];
        tmp += xi * xi;
    }

    // sum up partial sums
    tmp = warp_reduce_جوڑ(tmp, شے_ct1);
    إذا (بلاک_ناپ_3 > WARP_ناپ) {
        ثابت خودکار sub_گروہ = شے_ct1.get_sub_گروہ();
        ثابت خودکار sg_id = sub_گروہ.get_گروہ_خطی_id();
        ثابت خودکار wi_وچ_sg = sub_گروہ.get_local_خطی_id();
        إذا (wi_وچ_sg == 0) {
            s_جوڑ[sg_id] = tmp;
        }

        شے_ct1.barrier(sycl::access::fence_space::local_space);
        ثابت size_t nreduce = ceil_div(nwarps, WARP_ناپ);
        tmp = 0.f;
        لكل (size_t i = 0; i < nreduce; i += 1)
        {
            tmp += s_جوڑ[wi_وچ_sg + i * WARP_ناپ];
        }
        tmp = warp_reduce_جوڑ(tmp, شے_ct1);
    }

    ثابت عائم اوسط_2 = tmp / ncols;
    ثابت عائم scale = sycl::rsqrt(اوسط_2 + eps);

    لكل (صحيح خانہ = tid; خانہ < ncols; خانہ += بلاک_ناپ_3) {
        إذا constexpr (do_multiply && do_جوڑو) {
            dst[خانہ * dst_stride_خانہ] = scale * x[خانہ * src_stride_خانہ] * mul[خانہ] + جوڑو_2[خانہ];
        } وإلا إذا constexpr (do_multiply) {
            dst[خانہ * dst_stride_خانہ] = scale * x[خانہ * src_stride_خانہ] * mul[خانہ];
        } وإلا {
            // folded epilogue of a fused GGML_OP_SCALE consumer (qwen35 GDN l2 norms);
            // the explicit temporary keeps the float evaluation order identical to
            // running rms_norm and scale as two separate kernels
            ثابت عائم v = scale * x[خانہ * src_stride_خانہ];
            dst[خانہ * dst_stride_خانہ] = v * scale_mul;
        }
    }
}

سانچا<صحيح warp_ناپ>
ساكن فراغ l2_norm_f32(ثابت عائم * x, عائم * dst, ثابت صحيح ncols,
    ثابت int64_t src_stride_خانہ, ثابت int64_t src_stride_صف, ثابت int64_t src_stride_channel,
    ثابت int64_t src_stride_sample, ثابت int64_t dst_stride_خانہ, ثابت int64_t dst_stride_صف,
    ثابت int64_t dst_stride_channel, ثابت int64_t dst_stride_sample, ثابت عائم eps,
    ثابت sycl::nd_شے<3>& شے_ct1, عائم* s_جوڑ, ثابت صحيح بلاک_ناپ_3) {

    ثابت صحيح صف     = شے_ct1.get_گروہ(2);
    ثابت صحيح channel = شے_ct1.get_گروہ(1);
    ثابت صحيح sample  = شے_ct1.get_گروہ(0);
    ثابت صحيح tid     = شے_ct1.get_local_id(2);

    x   += sample*src_stride_sample + channel*src_stride_channel + صف*src_stride_صف;
    dst += sample*dst_stride_sample + channel*dst_stride_channel + صف*dst_stride_صف;

    عائم tmp = 0.0f; // partial sum for thread in warp

    لكل (صحيح خانہ = tid; خانہ < ncols; خانہ += بلاک_ناپ_3) {
        ثابت عائم xi = x[خانہ * src_stride_خانہ];
        tmp += xi * xi;
    }

    tmp = بلاک_reduce<بلاک_reduce_طریقہ::جوڑ_3, warp_ناپ>(tmp, s_جوڑ, بلاک_ناپ_3);
    ثابت عائم scale = sycl::rsqrt(sycl::fmax(tmp, eps * eps));

    لكل (صحيح خانہ = tid; خانہ < ncols; خانہ += بلاک_ناپ_3) {
        dst[خانہ * dst_stride_خانہ] = scale * x[خانہ * src_stride_خانہ];
    }
}

ساكن فراغ norm_f32_sycl(ثابت عائم * x, عائم * dst, ثابت صحيح ncols, ثابت صحيح nrows, ثابت صحيح nchannels, ثابت صحيح nsamples,
    ثابت int64_t src_stride_خانہ, ثابت int64_t src_stride_صف, ثابت int64_t src_stride_channel, ثابت int64_t src_stride_sample,
    ثابت int64_t dst_stride_خانہ, ثابت int64_t dst_stride_صف, ثابت int64_t dst_stride_channel, ثابت int64_t dst_stride_sample,
        ثابت عائم eps, قطار_ptr stream, صحيح device) {

    ثابت sycl::range<3> global_dims(nsamples, nchannels, nrows);
    إذا (ncols < 1024) {
        ثابت sycl::range<3> بلاک_dims(1, 1, WARP_ناپ);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * بلاک_dims, بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1)
                [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                    norm_f32(x, dst, ncols,
                        src_stride_خانہ, src_stride_صف, src_stride_channel, src_stride_sample,
                        dst_stride_خانہ, dst_stride_صف, dst_stride_channel, dst_stride_sample,
                        eps, شے_ct1, خالی_سوچک, WARP_ناپ);
                });
            });
    }
    وإلا {
        ثابت صحيح work_گروہ_ناپ = ggml_sycl_info().max_work_گروہ_sizes[device];
        أكد(work_گروہ_ناپ % (WARP_ناپ * WARP_ناپ) == 0);
        ثابت sycl::range<3> بلاک_dims(1, 1, work_گروہ_ناپ);
        /*
        DPCT1049:17: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<sycl::float2, 1> s_جوڑ_acc_ct1(
                            sycl::range<1>(work_گروہ_ناپ / WARP_ناپ), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * بلاک_dims, بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1)
                [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                    norm_f32(x, dst, ncols,
                        src_stride_خانہ, src_stride_صف, src_stride_channel, src_stride_sample,
                        dst_stride_خانہ, dst_stride_صف, dst_stride_channel, dst_stride_sample,
                        eps, شے_ct1, get_اشارہ_گر(s_جوڑ_acc_ct1), work_گروہ_ناپ);
                });
            });
    }
}

ساكن فراغ گروہ_norm_f32_sycl(ثابت عائم* x, عائم* dst,
    ثابت صحيح عدد_groups, ثابت عائم eps, ثابت صحيح گروہ_ناپ,
    ثابت صحيح ne_elements, قطار_ptr stream, صحيح device) {
    إذا (گروہ_ناپ < 1024) {
        ثابت sycl::range<3> بلاک_dims(1, 1, WARP_ناپ);
        stream->submit([&](sycl::handler& cgh) {
            ثابت عائم eps_ct4 = eps;
            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, عدد_groups) * بلاک_dims,
                    بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1)
                [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                    گروہ_norm_f32(
                        x, dst, گروہ_ناپ, ne_elements, eps_ct4, شے_ct1,
                        خالی_سوچک, WARP_ناپ);
                });
            });
    }
    وإلا {
        ثابت صحيح work_گروہ_ناپ = ggml_sycl_info().max_work_گروہ_sizes[device];
        أكد(work_گروہ_ناپ % (WARP_ناپ * WARP_ناپ) == 0);
        ثابت sycl::range<3> بلاک_dims(1, 1, work_گروہ_ناپ);
        /*
        DPCT1049:18: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */

        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<عائم, 1> s_جوڑ_acc_ct1(sycl::range<1>(work_گروہ_ناپ / WARP_ناپ),
                cgh);

            ثابت عائم eps_ct4 = eps;

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, عدد_groups) * بلاک_dims,
                    بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1)
                [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                    گروہ_norm_f32(x, dst, گروہ_ناپ, ne_elements,
                        eps_ct4, شے_ct1,
                        get_اشارہ_گر(s_جوڑ_acc_ct1), work_گروہ_ناپ);
                });
            });
    }
}

ساكن فراغ rms_norm_f32_sycl(ثابت عائم* x, عائم* dst, ثابت صحيح ncols, ثابت صحيح nrows, ثابت صحيح nchannels, ثابت صحيح nsamples,
    ثابت int64_t src_stride_خانہ, ثابت int64_t src_stride_صف, ثابت int64_t src_stride_channel, ثابت int64_t src_stride_sample,
    ثابت int64_t dst_stride_خانہ, ثابت int64_t dst_stride_صف, ثابت int64_t dst_stride_channel, ثابت int64_t dst_stride_sample,
    ثابت عائم eps, قطار_ptr stream, صحيح device) {
    // printf("%s ncols=%d, nrows=%d, WARP_SIZE=%d\n", __func__, ncols, nrows, WARP_SIZE);

    ثابت sycl::range<3> global_dims(nsamples, nchannels, nrows);
    إذا (ncols < 1024) {
        ثابت sycl::range<3> بلاک_dims(1, 1, WARP_ناپ);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * بلاک_dims, بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1)
                [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                    rms_norm_f32(x, dst, ncols,
                        src_stride_خانہ, src_stride_صف, src_stride_channel, src_stride_sample,
                        dst_stride_خانہ, dst_stride_صف, dst_stride_channel, dst_stride_sample,
                        eps, شے_ct1, خالی_سوچک, WARP_ناپ);
                });
            });
    }
    وإلا {
        ثابت صحيح work_گروہ_ناپ = ggml_sycl_info().max_work_گروہ_sizes[device];
        أكد(work_گروہ_ناپ % (WARP_ناپ * WARP_ناپ) == 0);
        ثابت sycl::range<3> بلاک_dims(1, 1, work_گروہ_ناپ);
        /*
        DPCT1049:19: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<عائم, 1> s_جوڑ_acc_ct1(sycl::range<1>(work_گروہ_ناپ / WARP_ناپ),
                cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * بلاک_dims, بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1)
                [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                    rms_norm_f32(x, dst, ncols,
                        src_stride_خانہ, src_stride_صف, src_stride_channel, src_stride_sample,
                        dst_stride_خانہ, dst_stride_صف, dst_stride_channel, dst_stride_sample,
                        eps, شے_ct1, get_اشارہ_گر(s_جوڑ_acc_ct1), work_گروہ_ناپ);
                });
            });
    }
}

ساكن فراغ rms_norm_scale_f32_sycl(ثابت عائم* x, عائم* dst, ثابت صحيح ncols, ثابت صحيح nrows,
    ثابت صحيح nchannels, ثابت صحيح nsamples,
    ثابت int64_t src_stride_خانہ, ثابت int64_t src_stride_صف, ثابت int64_t src_stride_channel, ثابت int64_t src_stride_sample,
    ثابت int64_t dst_stride_خانہ, ثابت int64_t dst_stride_صف, ثابت int64_t dst_stride_channel, ثابت int64_t dst_stride_sample,
    ثابت عائم eps, ثابت عائم scale_mul, قطار_ptr stream, صحيح device) {
    ثابت sycl::range<3> global_dims(nsamples, nchannels, nrows);
    إذا (ncols < 1024) {
        ثابت sycl::range<3> بلاک_dims(1, 1, WARP_ناپ);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * بلاک_dims, بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1)
                [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                    rms_norm_f32(x, dst, ncols,
                        src_stride_خانہ, src_stride_صف, src_stride_channel, src_stride_sample,
                        dst_stride_خانہ, dst_stride_صف, dst_stride_channel, dst_stride_sample,
                        eps, شے_ct1, خالی_سوچک, WARP_ناپ,
                        خالی_سوچک, 0, 0, 0, 0, 0, 0,
                        خالی_سوچک, 0, 0, 0, 0, 0, 0,
                        scale_mul);
                });
            });
    }
    وإلا {
        ثابت صحيح work_گروہ_ناپ = ggml_sycl_info().max_work_گروہ_sizes[device];
        أكد(work_گروہ_ناپ % (WARP_ناپ * WARP_ناپ) == 0);
        ثابت sycl::range<3> بلاک_dims(1, 1, work_گروہ_ناپ);
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<عائم, 1> s_جوڑ_acc_ct1(sycl::range<1>(work_گروہ_ناپ / WARP_ناپ),
                cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * بلاک_dims, بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1)
                [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                    rms_norm_f32(x, dst, ncols,
                        src_stride_خانہ, src_stride_صف, src_stride_channel, src_stride_sample,
                        dst_stride_خانہ, dst_stride_صف, dst_stride_channel, dst_stride_sample,
                        eps, شے_ct1, get_اشارہ_گر(s_جوڑ_acc_ct1), work_گروہ_ناپ,
                        خالی_سوچک, 0, 0, 0, 0, 0, 0,
                        خالی_سوچک, 0, 0, 0, 0, 0, 0,
                        scale_mul);
                });
            });
    }
}

ساكن فراغ rms_norm_mul_f32_sycl(ثابت عائم* x, ثابت عائم* mul, عائم* dst, ثابت صحيح ncols, ثابت صحيح nrows,
        ثابت صحيح nchannels, ثابت صحيح nsamples,
        ثابت int64_t src_stride_خانہ, ثابت int64_t src_stride_صف, ثابت int64_t src_stride_channel, ثابت int64_t src_stride_sample,
        ثابت int64_t dst_stride_خانہ, ثابت int64_t dst_stride_صف, ثابت int64_t dst_stride_channel, ثابت int64_t dst_stride_sample,
        ثابت int64_t mul_stride_صف, ثابت int64_t mul_stride_channel, ثابت int64_t mul_stride_sample,
        ثابت صحيح mul_nrows, ثابت صحيح mul_nchannels, ثابت صحيح mul_nsamples,
        ثابت عائم eps, قطار_ptr stream, صحيح device) {
    ثابت sycl::range<3> global_dims(nsamples, nchannels, nrows);
    إذا (ncols < 1024) {
        ثابت sycl::range<3> بلاک_dims(1, 1, WARP_ناپ);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * بلاک_dims, بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1)
                [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                    rms_norm_f32<سچ>(x, dst, ncols,
                        src_stride_خانہ, src_stride_صف, src_stride_channel, src_stride_sample,
                        dst_stride_خانہ, dst_stride_صف, dst_stride_channel, dst_stride_sample,
                        eps, شے_ct1, خالی_سوچک, WARP_ناپ,
                        mul, mul_stride_صف, mul_stride_channel, mul_stride_sample, mul_nrows, mul_nchannels, mul_nsamples);
                });
            });
    }
    وإلا {
        ثابت صحيح work_گروہ_ناپ = ggml_sycl_info().max_work_گروہ_sizes[device];
        أكد(work_گروہ_ناپ % (WARP_ناپ * WARP_ناپ) == 0);
        ثابت sycl::range<3> بلاک_dims(1, 1, work_گروہ_ناپ);
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<عائم, 1> s_جوڑ_acc_ct1(sycl::range<1>(work_گروہ_ناپ / WARP_ناپ), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * بلاک_dims, بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1)
                [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                    rms_norm_f32<سچ>(x, dst, ncols,
                        src_stride_خانہ, src_stride_صف, src_stride_channel, src_stride_sample,
                        dst_stride_خانہ, dst_stride_صف, dst_stride_channel, dst_stride_sample,
                        eps, شے_ct1, get_اشارہ_گر(s_جوڑ_acc_ct1), work_گروہ_ناپ,
                        mul, mul_stride_صف, mul_stride_channel, mul_stride_sample, mul_nrows, mul_nchannels, mul_nsamples);
                });
            });
    }
}

ساكن فراغ rms_norm_mul_جوڑو_f32_sycl(ثابت عائم* x, ثابت عائم* mul, ثابت عائم* جوڑو_2, عائم* dst,
        ثابت صحيح ncols, ثابت صحيح nrows, ثابت صحيح nchannels, ثابت صحيح nsamples,
        ثابت int64_t src_stride_خانہ, ثابت int64_t src_stride_صف, ثابت int64_t src_stride_channel, ثابت int64_t src_stride_sample,
        ثابت int64_t dst_stride_خانہ, ثابت int64_t dst_stride_صف, ثابت int64_t dst_stride_channel, ثابت int64_t dst_stride_sample,
        ثابت int64_t mul_stride_صف, ثابت int64_t mul_stride_channel, ثابت int64_t mul_stride_sample,
        ثابت صحيح mul_nrows, ثابت صحيح mul_nchannels, ثابت صحيح mul_nsamples,
        ثابت int64_t جوڑو_stride_صف, ثابت int64_t جوڑو_stride_channel, ثابت int64_t جوڑو_stride_sample,
        ثابت صحيح جوڑو_nrows, ثابت صحيح جوڑو_nchannels, ثابت صحيح جوڑو_nsamples,
        ثابت عائم eps, قطار_ptr stream, صحيح device) {
    ثابت sycl::range<3> global_dims(nsamples, nchannels, nrows);
    إذا (ncols < 1024) {
        ثابت sycl::range<3> بلاک_dims(1, 1, WARP_ناپ);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * بلاک_dims, بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1)
                [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                    rms_norm_f32<سچ, سچ>(x, dst, ncols,
                        src_stride_خانہ, src_stride_صف, src_stride_channel, src_stride_sample,
                        dst_stride_خانہ, dst_stride_صف, dst_stride_channel, dst_stride_sample,
                        eps, شے_ct1, خالی_سوچک, WARP_ناپ,
                        mul, mul_stride_صف, mul_stride_channel, mul_stride_sample, mul_nrows, mul_nchannels, mul_nsamples,
                        جوڑو_2, جوڑو_stride_صف, جوڑو_stride_channel, جوڑو_stride_sample, جوڑو_nrows, جوڑو_nchannels, جوڑو_nsamples);
                });
            });
    }
    وإلا {
        ثابت صحيح work_گروہ_ناپ = ggml_sycl_info().max_work_گروہ_sizes[device];
        أكد(work_گروہ_ناپ % (WARP_ناپ * WARP_ناپ) == 0);
        ثابت sycl::range<3> بلاک_dims(1, 1, work_گروہ_ناپ);
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<عائم, 1> s_جوڑ_acc_ct1(sycl::range<1>(work_گروہ_ناپ / WARP_ناپ), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * بلاک_dims, بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1)
                [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                    rms_norm_f32<سچ, سچ>(x, dst, ncols,
                        src_stride_خانہ, src_stride_صف, src_stride_channel, src_stride_sample,
                        dst_stride_خانہ, dst_stride_صف, dst_stride_channel, dst_stride_sample,
                        eps, شے_ct1, get_اشارہ_گر(s_جوڑ_acc_ct1), work_گروہ_ناپ,
                        mul, mul_stride_صف, mul_stride_channel, mul_stride_sample, mul_nrows, mul_nchannels, mul_nsamples,
                        جوڑو_2, جوڑو_stride_صف, جوڑو_stride_channel, جوڑو_stride_sample, جوڑو_nrows, جوڑو_nchannels, جوڑو_nsamples);
                });
            });
    }
}

سانچا<صحيح warp_ناپ>
ساكن فراغ l2_norm_f32_sycl(ثابت عائم *   x,
                             عائم *         dst,
                             ثابت صحيح       ncols,
                             ثابت صحيح       nrows,
                             ثابت صحيح       nchannels,
                             ثابت صحيح       nsamples,
                             ثابت int64_t   src_stride_خانہ,
                             ثابت int64_t   src_stride_صف,
                             ثابت int64_t   src_stride_channel,
                             ثابت int64_t   src_stride_sample,
                             ثابت int64_t   dst_stride_خانہ,
                             ثابت int64_t   dst_stride_صف,
                             ثابت int64_t   dst_stride_channel,
                             ثابت int64_t   dst_stride_sample,
                             ثابت عائم     eps,
                             قطار_ptr       stream,
                             صحيح             device) {
    ثابت dpct::dim3 بلاکاں_عدد(nrows, nchannels, nsamples);

    إذا (ncols < 1024) {
        ثابت dpct::dim3 بلاک_dims(warp_ناپ, 1, 1);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(بلاکاں_عدد * بلاک_dims,
                    بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1)
                [[sycl::reqd_sub_گروہ_ناپ(warp_ناپ)]] {
                    l2_norm_f32<warp_ناپ>(x, dst, ncols,
                        src_stride_خانہ, src_stride_صف, src_stride_channel, src_stride_sample,
                        dst_stride_خانہ, dst_stride_صف, dst_stride_channel, dst_stride_sample,
                        eps, شے_ct1,
                        خالی_سوچک, warp_ناپ);
                });
            });
    }
    وإلا {
        ثابت صحيح work_گروہ_ناپ = ggml_sycl_info().max_work_گروہ_sizes[device];
        أكد(work_گروہ_ناپ % (warp_ناپ * warp_ناپ) == 0);
        ثابت sycl::range<3> بلاک_dims(1, 1, work_گروہ_ناپ);
        صحيح lsm_ناپ =  بلاک_dims[2] > warp_ناپ ? work_گروہ_ناپ / warp_ناپ * حجم(عائم): 0;
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<عائم, 1> s_جوڑ_acc_ct1(sycl::range<1>(lsm_ناپ),
                cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(بلاکاں_عدد * بلاک_dims,
                    بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1)
                [[sycl::reqd_sub_گروہ_ناپ(warp_ناپ)]] {
                    l2_norm_f32<warp_ناپ>(x, dst, ncols,
                        src_stride_خانہ, src_stride_صف, src_stride_channel, src_stride_sample,
                        dst_stride_خانہ, dst_stride_صف, dst_stride_channel, dst_stride_sample,
                        eps, شے_ct1, get_اشارہ_گر(s_جوڑ_acc_ct1), work_گروہ_ناپ);
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

سانچا <صحيح warp_ناپ>
ساكن فراغ l2_norm_f32_batch(l2_batch_ptrs p, l2_batch_strides st, ثابت صحيح ncols, ثابت عائم eps,
                              ثابت sycl::nd_شے<3> & شے_ct1) {
    ثابت صحيح t   = شے_ct1.get_گروہ(0);  // tensor index
    ثابت صحيح r   = شے_ct1.get_گروہ(2);  // flattened row over ne1*ne2*ne3
    ثابت صحيح tid = شے_ct1.get_local_id(2);

    ثابت صحيح i1 = r % st.ne1;
    ثابت صحيح i2 = (r / st.ne1) % st.ne2;
    ثابت صحيح i3 = r / (st.ne1 * st.ne2);

    ثابت عائم * x   = p.src[t] + i3 * st.ss3 + i2 * st.ss2 + i1 * st.ss1;
    عائم *       dst = p.dst[t] + i3 * st.ds3 + i2 * st.ds2 + i1 * st.ds1;

    عائم tmp = 0.0f;
    لكل (صحيح خانہ = tid; خانہ < ncols; خانہ += warp_ناپ) {
        ثابت عائم xi = x[خانہ * st.ss0];
        tmp += xi * xi;
    }
    tmp = بلاک_reduce<بلاک_reduce_طریقہ::جوڑ_3, warp_ناپ>(tmp, (عائم *) خالی_سوچک, warp_ناپ);
    ثابت عائم scale = sycl::rsqrt(sycl::fmax(tmp, eps * eps));
    لكل (صحيح خانہ = tid; خانہ < ncols; خانہ += warp_ناپ) {
        dst[خانہ * st.ds0] = scale * x[خانہ * st.ss0];
    }
}

سانچا <صحيح warp_ناپ>
ساكن فراغ l2_norm_f32_batch_sycl(l2_batch_ptrs p, l2_batch_strides st, ثابت صحيح n_tensors,
                                   ثابت صحيح ncols, ثابت صحيح nrows_کل, ثابت عائم eps,
                                   قطار_ptr stream) {
    ثابت dpct::dim3 بلاکاں_عدد(nrows_کل, 1, n_tensors);
    ثابت dpct::dim3 بلاک_dims(warp_ناپ, 1, 1);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(بلاکاں_عدد * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_ناپ(warp_ناپ)]] {
                l2_norm_f32_batch<warp_ناپ>(p, st, ncols, eps, شے_ct1);
            });
    });
}

فراغ ggml_sycl_op_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    ثابت ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS
    dpct::قطار_ptr main_دھارا = ctx.stream();
    SYCL_CHECK(ggml_sycl_سیٹ_device(ctx.device));
    ثابت عائم * src0_dd = static_cast<ثابت عائم *>(dst->src[0]->انکڑے);
    عائم *       dst_dd  = static_cast<عائم *>(dst->انکڑے);

    عائم eps;
    حافظہ_نقل(&eps, dst->op_params, حجم(عائم));
    GGML_ASSERT(eps >= 0.0f);
    ثابت size_t ts0 = ggml_type_ناپ(src0->type);
    ثابت size_t tdst = ggml_type_ناپ(dst->type);
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

فراغ ggml_sycl_op_گروہ_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    صحيح عدد_groups = dst->op_params[0];
    dpct::قطار_ptr main_دھارا = ctx.stream();
    SYCL_CHECK(ggml_sycl_سیٹ_device(ctx.device));

    ثابت عائم * src0_dd = static_cast<ثابت عائم *>(dst->src[0]->انکڑے);
    عائم *       dst_dd  = static_cast<عائم *>(dst->انکڑے);

    عائم eps;
    حافظہ_نقل(&eps, dst->op_params + 1, حجم(عائم));

    صحيح گروہ_ناپ = dst->src[0]->ne[0] * dst->src[0]->ne[1] * ((dst->src[0]->ne[2] + عدد_groups - 1) / عدد_groups);
    گروہ_norm_f32_sycl(src0_dd, dst_dd, عدد_groups, eps, گروہ_ناپ, dst->src[0]->ne[0] * dst->src[0]->ne[1] * dst->src[0]->ne[2], main_دھارا, ctx.device);
}

فراغ ggml_sycl_op_rms_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    dpct::قطار_ptr main_دھارا = ctx.stream();
    SYCL_CHECK(ggml_sycl_سیٹ_device(ctx.device));

    ثابت عائم * src0_dd = static_cast<ثابت عائم *>(dst->src[0]->انکڑے);
    عائم *       dst_dd  = static_cast<عائم *>(dst->انکڑے);

    عائم eps;
    حافظہ_نقل(&eps, dst->op_params, حجم(عائم));

    GGML_TENSOR_UNARY_OP_LOCALS
    ثابت size_t ts0 = ggml_type_ناپ(src0->type);
    ثابت size_t tdst = ggml_type_ناپ(dst->type);
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
فراغ ggml_sycl_op_rms_norm_scale_fused(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                       ggml_tensor * scale_tensor) {
    ثابت ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(scale_tensor->type == GGML_TYPE_F32);

    dpct::قطار_ptr main_دھارا = ctx.stream();
    SYCL_CHECK(ggml_sycl_سیٹ_device(ctx.device));

    ثابت عائم * src0_dd = static_cast<ثابت عائم *>(src0->انکڑے);
    عائم *       dst_dd  = static_cast<عائم *>(scale_tensor->انکڑے);

    عائم eps;
    حافظہ_نقل(&eps, dst->op_params, حجم(عائم));
    عائم scale_mul;
    حافظہ_نقل(&scale_mul, scale_tensor->op_params, حجم(عائم));
    GGML_ASSERT(scale_mul >= 0.0f);

    GGML_TENSOR_UNARY_OP_LOCALS
    ثابت size_t ts0 = ggml_type_ناپ(src0->type);
    ثابت size_t tdst = ggml_type_ناپ(scale_tensor->type);
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

فراغ ggml_sycl_op_rms_norm_fused(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor) {
    ثابت ggml_tensor * rms_norm_src = dst->src[0];
    عائم eps = 0.0f;
    حافظہ_نقل(&eps, dst->op_params, حجم(عائم));

    ثابت عائم *       src0_dd = static_cast<ثابت عائم *>(rms_norm_src->انکڑے);
    ثابت عائم *       mul_dd  = خالی_سوچک;
    ثابت ggml_tensor * mul_src = خالی_سوچک;
    إذا (mul_tensor->src[0] == dst) {
        mul_dd  = static_cast<ثابت عائم *>(mul_tensor->src[1]->انکڑے);
        mul_src = mul_tensor->src[1];
    } وإلا إذا (mul_tensor->src[1] == dst) {
        mul_dd  = static_cast<ثابت عائم *>(mul_tensor->src[0]->انکڑے);
        mul_src = mul_tensor->src[0];
    } وإلا {
        GGML_ASSERT(خطأ);
    }
    عائم * dst_dd = static_cast<عائم *>(mul_tensor->انکڑے);

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

    ثابت size_t ts0 = ggml_type_ناپ(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    ثابت int64_t s00 = rms_norm_src->nb[0] / ts0;
    ثابت int64_t s01 = rms_norm_src->nb[1] / ts0;
    ثابت int64_t s02 = rms_norm_src->nb[2] / ts0;
    ثابت int64_t s03 = rms_norm_src->nb[3] / ts0;

    ثابت size_t tdst = ggml_type_ناپ(mul_tensor->type);
    GGML_ASSERT(mul_tensor->nb[0] == tdst);
    ثابت int64_t d00 = mul_tensor->nb[0] / tdst;
    ثابت int64_t d01 = mul_tensor->nb[1] / tdst;
    ثابت int64_t d02 = mul_tensor->nb[2] / tdst;
    ثابت int64_t d03 = mul_tensor->nb[3] / tdst;

    ثابت size_t ts_mul = ggml_type_ناپ(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    ثابت int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    ثابت int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    ثابت int64_t mul_s03 = mul_src->nb[3] / ts_mul;
    ثابت صحيح mul_nrows     = mul_src->ne[1];
    ثابت صحيح mul_nchannels = mul_src->ne[2];
    ثابت صحيح mul_nsamples  = mul_src->ne[3];

    rms_norm_mul_f32_sycl(src0_dd, mul_dd, dst_dd, ne00, ne01, ne02, ne03,
        s00, s01, s02, s03, d00, d01, d02, d03,
        mul_s01, mul_s02, mul_s03, mul_nrows, mul_nchannels, mul_nsamples, eps, main_دھارا, ctx.device);
}

فراغ ggml_sycl_op_rms_norm_fused_جوڑو(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                     ggml_tensor * mul_tensor, ggml_tensor * جوڑو_tensor) {
    ثابت ggml_tensor * rms_norm_src = dst->src[0];
    عائم eps = 0.0f;
    حافظہ_نقل(&eps, dst->op_params, حجم(عائم));

    ثابت عائم *       src0_dd = static_cast<ثابت عائم *>(rms_norm_src->انکڑے);
    ثابت عائم *       mul_dd  = خالی_سوچک;
    ثابت ggml_tensor * mul_src = خالی_سوچک;
    إذا (mul_tensor->src[0] == dst) {
        mul_dd  = static_cast<ثابت عائم *>(mul_tensor->src[1]->انکڑے);
        mul_src = mul_tensor->src[1];
    } وإلا إذا (mul_tensor->src[1] == dst) {
        mul_dd  = static_cast<ثابت عائم *>(mul_tensor->src[0]->انکڑے);
        mul_src = mul_tensor->src[0];
    } وإلا {
        GGML_ASSERT(خطأ);
    }

    ثابت عائم *       جوڑو_dd  = خالی_سوچک;
    ثابت ggml_tensor * جوڑو_src = خالی_سوچک;
    إذا (جوڑو_tensor->src[0] == mul_tensor) {
        جوڑو_dd  = static_cast<ثابت عائم *>(جوڑو_tensor->src[1]->انکڑے);
        جوڑو_src = جوڑو_tensor->src[1];
    } وإلا إذا (جوڑو_tensor->src[1] == mul_tensor) {
        جوڑو_dd  = static_cast<ثابت عائم *>(جوڑو_tensor->src[0]->انکڑے);
        جوڑو_src = جوڑو_tensor->src[0];
    } وإلا {
        GGML_ASSERT(خطأ);
    }

    عائم * dst_dd = static_cast<عائم *>(جوڑو_tensor->انکڑے);

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

    ثابت size_t ts0 = ggml_type_ناپ(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    ثابت int64_t s00 = rms_norm_src->nb[0] / ts0;
    ثابت int64_t s01 = rms_norm_src->nb[1] / ts0;
    ثابت int64_t s02 = rms_norm_src->nb[2] / ts0;
    ثابت int64_t s03 = rms_norm_src->nb[3] / ts0;

    ثابت size_t tdst = ggml_type_ناپ(جوڑو_tensor->type);
    GGML_ASSERT(جوڑو_tensor->nb[0] == tdst);
    ثابت int64_t d00 = جوڑو_tensor->nb[0] / tdst;
    ثابت int64_t d01 = جوڑو_tensor->nb[1] / tdst;
    ثابت int64_t d02 = جوڑو_tensor->nb[2] / tdst;
    ثابت int64_t d03 = جوڑو_tensor->nb[3] / tdst;

    ثابت size_t ts_mul = ggml_type_ناپ(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    ثابت int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    ثابت int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    ثابت int64_t mul_s03 = mul_src->nb[3] / ts_mul;
    ثابت صحيح mul_nrows     = mul_src->ne[1];
    ثابت صحيح mul_nchannels = mul_src->ne[2];
    ثابت صحيح mul_nsamples  = mul_src->ne[3];

    ثابت size_t ts_جوڑو = ggml_type_ناپ(جوڑو_src->type);
    GGML_ASSERT(جوڑو_src->nb[0] == ts_جوڑو);
    ثابت int64_t جوڑو_s01 = جوڑو_src->nb[1] / ts_جوڑو;
    ثابت int64_t جوڑو_s02 = جوڑو_src->nb[2] / ts_جوڑو;
    ثابت int64_t جوڑو_s03 = جوڑو_src->nb[3] / ts_جوڑو;
    ثابت صحيح جوڑو_nrows     = جوڑو_src->ne[1];
    ثابت صحيح جوڑو_nchannels = جوڑو_src->ne[2];
    ثابت صحيح جوڑو_nsamples  = جوڑو_src->ne[3];

    rms_norm_mul_جوڑو_f32_sycl(src0_dd, mul_dd, جوڑو_dd, dst_dd, ne00, ne01, ne02, ne03,
        s00, s01, s02, s03, d00, d01, d02, d03,
        mul_s01, mul_s02, mul_s03, mul_nrows, mul_nchannels, mul_nsamples,
        جوڑو_s01, جوڑو_s02, جوڑو_s03, جوڑو_nrows, جوڑو_nchannels, جوڑو_nsamples, eps, main_دھارا, ctx.device);
}

فراغ ggml_sycl_op_rms_norm_پٹھ(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32); // dz
    GGML_ASSERT(dst->src[1]->type == GGML_TYPE_F32); // x
    GGML_ASSERT(dst->type         == GGML_TYPE_F32);

    عائم eps = 1e-5f;
    std::حافظہ_نقل(&eps, dst->op_params, حجم(عائم));
    إذا (!(eps > 0.0f) || !std::isfinite(eps)) eps = 1e-5f;

    ثابت عائم * g_ادھار  = static_cast<ثابت عائم *>(dst->src[0]->انکڑے); // dz
    ثابت عائم * x_ادھار  = static_cast<ثابت عائم *>(dst->src[1]->انکڑے); // x
          عائم * dx_ادھار = static_cast<      عائم *>(dst->انکڑے);

    ثابت int64_t D  = dst->ne[0];
    ثابت int64_t n1 = dst->ne[1], n2 = dst->ne[2], n3 = dst->ne[3]; (فراغ) n3;
    ثابت int64_t N  = ggml_nrows(dst);
    إذا (D == 0 || N == 0) أرجع;

    ثابت ggml_tensor *G = dst->src[0];
    ثابت ggml_tensor *X = dst->src[1];
    ثابت صحيح ts = (صحيح) ggml_type_ناپ(X->type);
    GGML_ASSERT((size_t) X->nb[0]   == (size_t) ts);
    GGML_ASSERT((size_t) G->nb[0]   == (size_t) ts);
    GGML_ASSERT((size_t) dst->nb[0] == (size_t) ts);

    ثابت int64_t xs1 = X->nb[1] / ts, xs2 = X->nb[2] / ts, xs3 = X->nb[3] / ts;
    ثابت int64_t gs1 = G->nb[1] / ts, gs2 = G->nb[2] / ts, gs3 = G->nb[3] / ts;
    ثابت int64_t ds1 = dst->nb[1] / ts, ds2 = dst->nb[2] / ts, ds3 = dst->nb[3] / ts;

    dpct::قطار_ptr q = ctx.stream();

    // work-group size: multiple of WARP_SIZE, capped by device and 256, and not larger than D
    ثابت صحيح device_max_wg = ggml_sycl_info().max_work_گروہ_sizes[ctx.device];
    خودکار roundup = [](صحيح v, صحيح m) { أرجع ((v + m - 1) / m) * m; };
    صحيح wg_cap = 256;
    إذا (device_max_wg > 0) wg_cap = std::min(wg_cap, device_max_wg);
    صحيح WG = std::max(WARP_ناپ, std::min(roundup((صحيح)std::min<int64_t>(D, wg_cap), WARP_ناپ), wg_cap));

    // FP32 path: per-thread compensated accumulation + hierarchical reduction
    q->submit([&](sycl::handler &cgh) {
        ثابت صحيح nwarps_loc = std::max(1, WG / WARP_ناپ);
        // store one partial value per warp (xx and xg) for cross-warp reduction
        خودکار l_xx   = sycl::local_accessor<sycl::float2, 1>(sycl::range<1>(nwarps_loc), cgh);
        خودکار l_xg   = sycl::local_accessor<sycl::float2, 1>(sycl::range<1>(nwarps_loc), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, N) * sycl::range<3>(1, 1, WG),
                              sycl::range<3>(1, 1, WG)),
            [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                ثابت صحيح صف = شے_ct1.get_گروہ(2);
                ثابت صحيح tid = شے_ct1.get_local_id(2);

                ثابت int64_t i1 = صف % n1;
                ثابت int64_t i2 = (صف / n1) % n2;
                ثابت int64_t i3 = صف / (n1 * n2);

                ثابت عائم *__restrict x_صف = x_ادھار + i3 * xs3 + i2 * xs2 + i1 * xs1;
                ثابت عائم *__restrict g_صف = g_ادھار + i3 * gs3 + i2 * gs2 + i1 * gs1;
                عائم *__restrict d_صف       = dx_ادھار + i3 * ds3 + i2 * ds2 + i1 * ds1;

                // per-thread accumulation (compensated by default)
                عائم جوڑ_xx = 0.f, جوڑ_xg = 0.f;
#جے_بنا_تعریف GGML_SYCL_RMS_پٹھ_FAST
                عائم c_xx = 0.f, c_xg = 0.f;
#جے_انت
                لكل (int64_t خانہ = tid; خانہ < D; خانہ += WG) {
                    ثابت عائم xv = x_صف[خانہ];
                    ثابت عائم gv = g_صف[خانہ];
#جے_تعریف_شدہ GGML_SYCL_RMS_پٹھ_FAST
                    جوڑ_xx += xv * xv;
                    جوڑ_xg += xv * gv;
#وإلا
                    عائم y1 = xv * xv - c_xx;
                    عائم t1 = جوڑ_xx + y1;
                    c_xx = (t1 - جوڑ_xx) - y1;
                    جوڑ_xx = t1;

                    عائم y2 = xv * gv - c_xg;
                    عائم t2 = جوڑ_xg + y2;
                    c_xg = (t2 - جوڑ_xg) - y2;
                    جوڑ_xg = t2;
#جے_انت
                }

                // warp-level reduction
                sycl::float2 xx = sycl::float2(جوڑ_xx,
#جے_بنا_تعریف GGML_SYCL_RMS_پٹھ_FAST
                    c_xx
#وإلا
                    0.f
#جے_انت
                );
                sycl::float2 xg = sycl::float2(جوڑ_xg,
#جے_بنا_تعریف GGML_SYCL_RMS_پٹھ_FAST
                    c_xg
#وإلا
                    0.f
#جے_انت
                );
                xx = warp_reduce_جوڑ(xx, شے_ct1);
                xg = warp_reduce_جوڑ(xg, شے_ct1);

                // cross-warp reduction using local memory (single barrier)
                ثابت خودکار sub_گروہ = شے_ct1.get_sub_گروہ();
                ثابت خودکار sg_id     = sub_گروہ.get_گروہ_خطی_id();
                ثابت خودکار wi_وچ_sg  = sub_گروہ.get_local_خطی_id();
                ثابت صحيح nthreads   = شے_ct1.get_local_range(2);
                ثابت صحيح nwarps     = nthreads / WARP_ناپ;

                sycl::float2 xx_کل = xx;
                sycl::float2 xg_کل = xg;
                إذا (nwarps > 1) {
                    إذا (wi_وچ_sg == 0) {
                        l_xx[sg_id] = xx;
                        l_xg[sg_id] = xg;
                    }
                    شے_ct1.barrier(sycl::access::fence_space::local_space);

                    إذا (sg_id == 0) {
                        ثابت بنانشانی wi_u = wi_وچ_sg;
                        sycl::float2 xx_پہلا = (wi_u < static_cast<بنانشانی>(nwarps)) ? l_xx[wi_u] : sycl::float2(0.f, 0.f);
                        sycl::float2 xg_پہلا = (wi_u < static_cast<بنانشانی>(nwarps)) ? l_xg[wi_u] : sycl::float2(0.f, 0.f);
                        xx_کل = warp_reduce_جوڑ(xx_پہلا, شے_ct1);
                        xg_کل = warp_reduce_جوڑ(xg_پہلا, شے_ct1);
                    } وإلا {
                        // other subgroups keep their local totals; they'll be ignored
                        xx_کل = xx;
                        xg_کل = xg;
                    }
                    // ensure all threads see the first-subgroup result via broadcast below
                }

                // compute inv_r and coeff once per row and broadcast to the whole work-group
                عائم inv_r = 0.f;
                عائم coeff = 0.f;
                إذا (tid == 0) {
                    ثابت عائم جوڑ_xx_f  = xx_کل.x() + xx_کل.y();
                    ثابت عائم جوڑ_xdz_f = xg_کل.x() + xg_کل.y();
                    ثابت عائم اوسط_eps  = جوڑ_xx_f / (عائم) D + eps;
                    ثابت عائم جوڑ_eps   = جوڑ_xx_f + eps * (عائم) D;
                    inv_r = sycl::rsqrt(اوسط_eps);
                    coeff = -جوڑ_xdz_f / جوڑ_eps;
                }
                inv_r = sycl::گروہ_نشر(شے_ct1.get_گروہ(), inv_r);
                coeff = sycl::گروہ_نشر(شے_ct1.get_گروہ(), coeff);

                لكل (int64_t خانہ = tid; خانہ < D; خانہ += WG) {
                    d_صف[خانہ] = (g_صف[خانہ] + coeff * x_صف[خانہ]) * inv_r;
                }
            });
    });

}

فراغ ggml_sycl_op_l2_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت عائم * src0_d = (ثابت عائم *) src0->انکڑے;
    عائم * dst_d = (عائم *) dst->انکڑے;
    dpct::قطار_ptr     stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    عائم eps;
    حافظہ_نقل(&eps, dst->op_params, حجم(عائم));
    GGML_ASSERT(eps >= 0.0f);

    ثابت size_t ts0 = ggml_type_ناپ(src0->type);
    ثابت size_t tdst = ggml_type_ناپ(dst->type);
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
    l2_norm_f32_sycl<WARP_ناپ>(src0_d, dst_d, ne00, ne01, ne02, ne03,
            ss0, ss1, ss2, ss3, ds0, ds1, ds2, ds3, eps, stream, ctx.device);
}

// nodes[0..count) are independent, same-shape, same-eps, same-nb L2_NORM ops validated
// by the caller; requires ncols < 1024 (the warp reduction path).
فراغ ggml_sycl_l2_norm_batch(ggml_backend_sycl_context & ctx, ggml_tensor ** nodes, صحيح count) {
    ثابت ggml_tensor * s0 = nodes[0]->src[0];
    ثابت صحيح ncols       = (صحيح) s0->ne[0];
    ثابت صحيح nrows_کل = (صحيح) ggml_nrows(s0);
    عائم eps;
    حافظہ_نقل(&eps, nodes[0]->op_params, حجم(عائم));
    GGML_ASSERT(eps >= 0.0f);

    l2_batch_ptrs p{};
    لكل (صحيح t = 0; t < count; ++t) {
        p.src[t] = (ثابت عائم *) nodes[t]->src[0]->انکڑے;
        p.dst[t] = (عائم *) nodes[t]->انکڑے;
    }

    ثابت ggml_tensor * d0 = nodes[0];
    ثابت size_t        ts = ggml_type_ناپ(GGML_TYPE_F32);
    l2_batch_strides    st{};
    st.ne1 = (صحيح) s0->ne[1];
    st.ne2 = (صحيح) s0->ne[2];
    st.ss0 = s0->nb[0] / ts; st.ss1 = s0->nb[1] / ts; st.ss2 = s0->nb[2] / ts; st.ss3 = s0->nb[3] / ts;
    st.ds0 = d0->nb[0] / ts; st.ds1 = d0->nb[1] / ts; st.ds2 = d0->nb[2] / ts; st.ds3 = d0->nb[3] / ts;

    l2_norm_f32_batch_sycl<WARP_ناپ>(p, st, count, ncols, nrows_کل, eps, ctx.stream());
}
