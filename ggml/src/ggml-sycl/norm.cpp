#ਅਨ੍ਤਰ੍ਭਾਵ "norm.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-sycl/common.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-sycl/presets.hpp"

ਸਥਿਰ ਖਾਲੀ norm_f32(ਅਚਲ ਦਸ਼ਮਲਵ* x, ਦਸ਼ਮਲਵ* dst, ਅਚਲ ਪੂਰਨ ncols,
    ਅਚਲ int64_t src_stride_ਖਾਨਾ, ਅਚਲ int64_t src_stride_ਪੰਕਤੀ, ਅਚਲ int64_t src_stride_channel, ਅਚਲ int64_t src_stride_sample,
    ਅਚਲ int64_t dst_stride_ਖਾਨਾ, ਅਚਲ int64_t dst_stride_ਪੰਕਤੀ, ਅਚਲ int64_t dst_stride_channel, ਅਚਲ int64_t dst_stride_sample,
    ਅਚਲ ਦਸ਼ਮਲਵ eps, ਅਚਲ sycl::nd_ਵਸਤੂ<3>& ਵਸਤੂ_ct1, sycl::float2* s_ਜੋੜ, ਪੂਰਨ ਖੰਡ_ਮਾਪ_3) {

    ਅਚਲ ਪੂਰਨ nthreads = ਵਸਤੂ_ct1.get_local_range(2);
    ਅਚਲ ਪੂਰਨ sample  = ਵਸਤੂ_ct1.get_ਸਮੂਹ(0);
    ਅਚਲ ਪੂਰਨ channel = ਵਸਤੂ_ct1.get_ਸਮੂਹ(1);
    ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ     = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2);

    ਅਚਲ ਪੂਰਨ tid = ਵਸਤੂ_ct1.get_local_id(2);
    ਅਚਲ ਪੂਰਨ nwarps = nthreads / WARP_ਮਾਪ;

    ਅਚਲ ਸਵੈਚਾਲਿਤ src_ਵਿੱਥ = calculate_ਵਿੱਥ<3>({src_stride_sample, src_stride_channel, src_stride_ਪੰਕਤੀ}, {sample, channel, ਪੰਕਤੀ});
    ਅਚਲ ਸਵੈਚਾਲਿਤ dst_ਵਿੱਥ = calculate_ਵਿੱਥ<3>({dst_stride_sample, dst_stride_channel, dst_stride_ਪੰਕਤੀ}, {sample, channel, ਪੰਕਤੀ});

    x += src_ਵਿੱਥ;
    dst += dst_ਵਿੱਥ;

    sycl::float2 ਔਸਤ_var = sycl::float2(0.f, 0.f);

    ਲਈ (ਪੂਰਨ ਖਾਨਾ = tid; ਖਾਨਾ < ncols; ਖਾਨਾ += ਖੰਡ_ਮਾਪ_3) {
        ਅਚਲ ਦਸ਼ਮਲਵ xi = x[ਖਾਨਾ * src_stride_ਖਾਨਾ];
        ਔਸਤ_var.x() += xi;
        ਔਸਤ_var.y() += xi * xi;
    }

    // sum up partial sums
    ਔਸਤ_var = warp_reduce_ਜੋੜ(ਔਸਤ_var, ਵਸਤੂ_ct1);
    ਜੇ  (ਖੰਡ_ਮਾਪ_3 > WARP_ਮਾਪ) {
        ਅਚਲ ਸਵੈਚਾਲਿਤ sub_ਸਮੂਹ = ਵਸਤੂ_ct1.get_sub_ਸਮੂਹ();
        ਅਚਲ ਸਵੈਚਾਲਿਤ sg_id = sub_ਸਮੂਹ.get_ਸਮੂਹ_ਰੇਖਿਕ_id();
        ਅਚਲ ਸਵੈਚਾਲਿਤ wi_ਵਿੱਚ_sg = sub_ਸਮੂਹ.get_local_ਰੇਖਿਕ_id();
        ਜੇ (wi_ਵਿੱਚ_sg == 0) {
            s_ਜੋੜ[sg_id] = ਔਸਤ_var;
        }
        ਵਸਤੂ_ct1.barrier(sycl::access::fence_space::local_space);
        ਔਸਤ_var = 0.f;
        ਅਚਲ size_t nreduce = ceil_div(nwarps, WARP_ਮਾਪ);
        ਲਈ (size_t i = 0; i < nreduce; i += 1)
        {
            ਔਸਤ_var += s_ਜੋੜ[wi_ਵਿੱਚ_sg + i * WARP_ਮਾਪ];
        }
        ਔਸਤ_var = warp_reduce_ਜੋੜ(ਔਸਤ_var, ਵਸਤੂ_ct1);
    }

    ਅਚਲ ਦਸ਼ਮਲਵ ਔਸਤ_2 = ਔਸਤ_var.x() / ncols;
    ਅਚਲ ਦਸ਼ਮਲਵ var = ਔਸਤ_var.y() / ncols - ਔਸਤ_2 * ਔਸਤ_2;
    ਅਚਲ ਦਸ਼ਮਲਵ inv_std = sycl::rsqrt(var + eps);

    ਲਈ (ਪੂਰਨ ਖਾਨਾ = tid; ਖਾਨਾ < ncols; ਖਾਨਾ += ਖੰਡ_ਮਾਪ_3) {
        dst[ਖਾਨਾ * dst_stride_ਖਾਨਾ] = (x[ਖਾਨਾ * src_stride_ਖਾਨਾ] - ਔਸਤ_2) * inv_std;
    }
}

ਸਥਿਰ ਖਾਲੀ ਸਮੂਹ_norm_f32(ਅਚਲ ਦਸ਼ਮਲਵ* x, ਦਸ਼ਮਲਵ* dst, ਅਚਲ ਪੂਰਨ ਸਮੂਹ_ਮਾਪ, ਅਚਲ ਪੂਰਨ ne_elements, ਅਚਲ ਦਸ਼ਮਲਵ eps,
    ਅਚਲ sycl::nd_ਵਸਤੂ<3>& ਵਸਤੂ_ct1, ਦਸ਼ਮਲਵ* s_ਜੋੜ, ਪੂਰਨ ਖੰਡ_ਮਾਪ_3) {
    ਪੂਰਨ start = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) * ਸਮੂਹ_ਮਾਪ;
    ਪੂਰਨ end = start + ਸਮੂਹ_ਮਾਪ;
    ਅਚਲ ਪੂਰਨ nthreads = ਵਸਤੂ_ct1.get_local_range(2);
    ਅਚਲ ਪੂਰਨ nwarps = nthreads / WARP_ਮਾਪ;
    start += ਵਸਤੂ_ct1.get_local_id(2);
    size_t nreduce = nwarps / WARP_ਮਾਪ;

    ਜੇ (end >= ne_elements) {
        end = ne_elements;
    }

    ਦਸ਼ਮਲਵ tmp = 0.0f; // partial sum for thread in warp

    ਲਈ (ਪੂਰਨ j = start; j < end; j += ਖੰਡ_ਮਾਪ_3) {
        tmp += x[j];
    }

    tmp = warp_reduce_ਜੋੜ(tmp, ਵਸਤੂ_ct1);
    ਜੇ (ਖੰਡ_ਮਾਪ_3 > WARP_ਮਾਪ) {

        ਪੂਰਨ warp_id = ਵਸਤੂ_ct1.get_local_id(2) / WARP_ਮਾਪ;
        ਪੂਰਨ lane_id = ਵਸਤੂ_ct1.get_local_id(2) % WARP_ਮਾਪ;
        ਜੇ (lane_id == 0) {
            s_ਜੋੜ[warp_id] = tmp;
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
        ਵਸਤੂ_ct1.barrier();
        tmp = 0.f;
        ਲਈ (size_t i = 0; i < nreduce; i += 1)
        {
            tmp += s_ਜੋੜ[lane_id + i * WARP_ਮਾਪ];
        }
        tmp = warp_reduce_ਜੋੜ(tmp, ਵਸਤੂ_ct1);
    }

    ਦਸ਼ਮਲਵ ਔਸਤ_2 = tmp / ਸਮੂਹ_ਮਾਪ;
    tmp = 0.0f;

    ਲਈ (ਪੂਰਨ j = start; j < end; j += ਖੰਡ_ਮਾਪ_3) {
        ਦਸ਼ਮਲਵ xi = x[j] - ਔਸਤ_2;
        dst[j] = xi;
        tmp += xi * xi;
    }

    tmp = warp_reduce_ਜੋੜ(tmp, ਵਸਤੂ_ct1);
    ਜੇ (ਖੰਡ_ਮਾਪ_3 > WARP_ਮਾਪ) {

        ਪੂਰਨ warp_id = ਵਸਤੂ_ct1.get_local_id(2) / WARP_ਮਾਪ;
        ਪੂਰਨ lane_id = ਵਸਤੂ_ct1.get_local_id(2) % WARP_ਮਾਪ;
        ਜੇ (lane_id == 0) {
            s_ਜੋੜ[warp_id] = tmp;
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
        ਵਸਤੂ_ct1.barrier();
        tmp = 0.f;
        ਲਈ (size_t i = 0; i < nreduce; i += 1)
        {
            tmp += s_ਜੋੜ[lane_id + i * WARP_ਮਾਪ];
        }
        tmp = warp_reduce_ਜੋੜ(tmp, ਵਸਤੂ_ct1);
    }

    ਦਸ਼ਮਲਵ variance = tmp / ਸਮੂਹ_ਮਾਪ;
    ਦਸ਼ਮਲਵ scale = sycl::rsqrt(variance + eps);
    ਲਈ (ਪੂਰਨ j = start; j < end; j += ਖੰਡ_ਮਾਪ_3) {
        dst[j] *= scale;
    }
}

ਸਾਂਚਾ <ਤਾਰਕਿਕ do_multiply = ਝੂਠ, ਤਾਰਕਿਕ do_ਜੋੜੋ = ਝੂਠ>
ਸਥਿਰ ਖਾਲੀ rms_norm_f32(ਅਚਲ ਦਸ਼ਮਲਵ* x, ਦਸ਼ਮਲਵ* dst, ਅਚਲ ਪੂਰਨ ncols,
    ਅਚਲ int64_t src_stride_ਖਾਨਾ, ਅਚਲ int64_t src_stride_ਪੰਕਤੀ, ਅਚਲ int64_t src_stride_channel, ਅਚਲ int64_t src_stride_sample,
    ਅਚਲ int64_t dst_stride_ਖਾਨਾ, ਅਚਲ int64_t dst_stride_ਪੰਕਤੀ, ਅਚਲ int64_t dst_stride_channel, ਅਚਲ int64_t dst_stride_sample,
    ਅਚਲ ਦਸ਼ਮਲਵ eps, ਅਚਲ sycl::nd_ਵਸਤੂ<3>& ਵਸਤੂ_ct1, ਦਸ਼ਮਲਵ* s_ਜੋੜ, ਪੂਰਨ ਖੰਡ_ਮਾਪ_3,
    ਅਚਲ ਦਸ਼ਮਲਵ* mul = ਖਾਲੀ_ਸੂਚਕ, ਅਚਲ int64_t mul_stride_ਪੰਕਤੀ = 0, ਅਚਲ int64_t mul_stride_channel = 0,
    ਅਚਲ int64_t mul_stride_sample = 0, ਅਚਲ ਪੂਰਨ mul_nrows = 0, ਅਚਲ ਪੂਰਨ mul_nchannels = 0, ਅਚਲ ਪੂਰਨ mul_nsamples = 0,
    ਅਚਲ ਦਸ਼ਮਲਵ* ਜੋੜੋ_2 = ਖਾਲੀ_ਸੂਚਕ, ਅਚਲ int64_t ਜੋੜੋ_stride_ਪੰਕਤੀ = 0, ਅਚਲ int64_t ਜੋੜੋ_stride_channel = 0,
    ਅਚਲ int64_t ਜੋੜੋ_stride_sample = 0, ਅਚਲ ਪੂਰਨ ਜੋੜੋ_nrows = 0, ਅਚਲ ਪੂਰਨ ਜੋੜੋ_nchannels = 0, ਅਚਲ ਪੂਰਨ ਜੋੜੋ_nsamples = 0,
    ਅਚਲ ਦਸ਼ਮਲਵ scale_mul = 1.0f) {

    static_assert(!do_ਜੋੜੋ || do_multiply, "fusing add is not supported without multiplying");

    ਅਚਲ ਪੂਰਨ sample  = ਵਸਤੂ_ct1.get_ਸਮੂਹ(0);
    ਅਚਲ ਪੂਰਨ channel = ਵਸਤੂ_ct1.get_ਸਮੂਹ(1);
    ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ     = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2);

    ਅਚਲ ਪੂਰਨ nthreads = ਵਸਤੂ_ct1.get_local_range(2);

    ਅਚਲ ਪੂਰਨ tid = ਵਸਤੂ_ct1.get_local_id(2);
    ਅਚਲ ਪੂਰਨ nwarps = nthreads / WARP_ਮਾਪ;

    ਅਚਲ ਸਵੈਚਾਲਿਤ src_ਵਿੱਥ = calculate_ਵਿੱਥ<3>({src_stride_sample, src_stride_channel, src_stride_ਪੰਕਤੀ}, {sample, channel, ਪੰਕਤੀ});
    ਅਚਲ ਸਵੈਚਾਲਿਤ dst_ਵਿੱਥ = calculate_ਵਿੱਥ<3>({dst_stride_sample, dst_stride_channel, dst_stride_ਪੰਕਤੀ}, {sample, channel, ਪੰਕਤੀ});

    x   += src_ਵਿੱਥ;
    dst += dst_ਵਿੱਥ;

    ਜੇ constexpr (do_multiply) {
        ਅਚਲ ਪੂਰਨ mul_ਪੰਕਤੀ     = ਪੰਕਤੀ     % mul_nrows;
        ਅਚਲ ਪੂਰਨ mul_channel = channel % mul_nchannels;
        ਅਚਲ ਪੂਰਨ mul_sample  = sample  % mul_nsamples;
        mul += mul_sample * mul_stride_sample + mul_channel * mul_stride_channel + mul_ਪੰਕਤੀ * mul_stride_ਪੰਕਤੀ;
    }

    ਜੇ constexpr (do_ਜੋੜੋ) {
        ਅਚਲ ਪੂਰਨ ਜੋੜੋ_ਪੰਕਤੀ     = ਪੰਕਤੀ     % ਜੋੜੋ_nrows;
        ਅਚਲ ਪੂਰਨ ਜੋੜੋ_channel = channel % ਜੋੜੋ_nchannels;
        ਅਚਲ ਪੂਰਨ ਜੋੜੋ_sample  = sample  % ਜੋੜੋ_nsamples;
        ਜੋੜੋ_2 += ਜੋੜੋ_sample * ਜੋੜੋ_stride_sample + ਜੋੜੋ_channel * ਜੋੜੋ_stride_channel + ਜੋੜੋ_ਪੰਕਤੀ * ਜੋੜੋ_stride_ਪੰਕਤੀ;
    }

    ਦਸ਼ਮਲਵ tmp = 0.0f; // partial sum for thread in warp

    ਲਈ (ਪੂਰਨ ਖਾਨਾ = tid; ਖਾਨਾ < ncols; ਖਾਨਾ += ਖੰਡ_ਮਾਪ_3) {
        ਅਚਲ ਦਸ਼ਮਲਵ xi = x[ਖਾਨਾ * src_stride_ਖਾਨਾ];
        tmp += xi * xi;
    }

    // sum up partial sums
    tmp = warp_reduce_ਜੋੜ(tmp, ਵਸਤੂ_ct1);
    ਜੇ (ਖੰਡ_ਮਾਪ_3 > WARP_ਮਾਪ) {
        ਅਚਲ ਸਵੈਚਾਲਿਤ sub_ਸਮੂਹ = ਵਸਤੂ_ct1.get_sub_ਸਮੂਹ();
        ਅਚਲ ਸਵੈਚਾਲਿਤ sg_id = sub_ਸਮੂਹ.get_ਸਮੂਹ_ਰੇਖਿਕ_id();
        ਅਚਲ ਸਵੈਚਾਲਿਤ wi_ਵਿੱਚ_sg = sub_ਸਮੂਹ.get_local_ਰੇਖਿਕ_id();
        ਜੇ (wi_ਵਿੱਚ_sg == 0) {
            s_ਜੋੜ[sg_id] = tmp;
        }

        ਵਸਤੂ_ct1.barrier(sycl::access::fence_space::local_space);
        ਅਚਲ size_t nreduce = ceil_div(nwarps, WARP_ਮਾਪ);
        tmp = 0.f;
        ਲਈ (size_t i = 0; i < nreduce; i += 1)
        {
            tmp += s_ਜੋੜ[wi_ਵਿੱਚ_sg + i * WARP_ਮਾਪ];
        }
        tmp = warp_reduce_ਜੋੜ(tmp, ਵਸਤੂ_ct1);
    }

    ਅਚਲ ਦਸ਼ਮਲਵ ਔਸਤ_2 = tmp / ncols;
    ਅਚਲ ਦਸ਼ਮਲਵ scale = sycl::rsqrt(ਔਸਤ_2 + eps);

    ਲਈ (ਪੂਰਨ ਖਾਨਾ = tid; ਖਾਨਾ < ncols; ਖਾਨਾ += ਖੰਡ_ਮਾਪ_3) {
        ਜੇ constexpr (do_multiply && do_ਜੋੜੋ) {
            dst[ਖਾਨਾ * dst_stride_ਖਾਨਾ] = scale * x[ਖਾਨਾ * src_stride_ਖਾਨਾ] * mul[ਖਾਨਾ] + ਜੋੜੋ_2[ਖਾਨਾ];
        } ਨਹੀਂਤਾਂ ਜੇ constexpr (do_multiply) {
            dst[ਖਾਨਾ * dst_stride_ਖਾਨਾ] = scale * x[ਖਾਨਾ * src_stride_ਖਾਨਾ] * mul[ਖਾਨਾ];
        } ਨਹੀਂਤਾਂ {
            // folded epilogue of a fused GGML_OP_SCALE consumer (qwen35 GDN l2 norms);
            // the explicit temporary keeps the float evaluation order identical to
            // running rms_norm and scale as two separate kernels
            ਅਚਲ ਦਸ਼ਮਲਵ v = scale * x[ਖਾਨਾ * src_stride_ਖਾਨਾ];
            dst[ਖਾਨਾ * dst_stride_ਖਾਨਾ] = v * scale_mul;
        }
    }
}

ਸਾਂਚਾ<ਪੂਰਨ warp_ਮਾਪ>
ਸਥਿਰ ਖਾਲੀ l2_norm_f32(ਅਚਲ ਦਸ਼ਮਲਵ * x, ਦਸ਼ਮਲਵ * dst, ਅਚਲ ਪੂਰਨ ncols,
    ਅਚਲ int64_t src_stride_ਖਾਨਾ, ਅਚਲ int64_t src_stride_ਪੰਕਤੀ, ਅਚਲ int64_t src_stride_channel,
    ਅਚਲ int64_t src_stride_sample, ਅਚਲ int64_t dst_stride_ਖਾਨਾ, ਅਚਲ int64_t dst_stride_ਪੰਕਤੀ,
    ਅਚਲ int64_t dst_stride_channel, ਅਚਲ int64_t dst_stride_sample, ਅਚਲ ਦਸ਼ਮਲਵ eps,
    ਅਚਲ sycl::nd_ਵਸਤੂ<3>& ਵਸਤੂ_ct1, ਦਸ਼ਮਲਵ* s_ਜੋੜ, ਅਚਲ ਪੂਰਨ ਖੰਡ_ਮਾਪ_3) {

    ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ     = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2);
    ਅਚਲ ਪੂਰਨ channel = ਵਸਤੂ_ct1.get_ਸਮੂਹ(1);
    ਅਚਲ ਪੂਰਨ sample  = ਵਸਤੂ_ct1.get_ਸਮੂਹ(0);
    ਅਚਲ ਪੂਰਨ tid     = ਵਸਤੂ_ct1.get_local_id(2);

    x   += sample*src_stride_sample + channel*src_stride_channel + ਪੰਕਤੀ*src_stride_ਪੰਕਤੀ;
    dst += sample*dst_stride_sample + channel*dst_stride_channel + ਪੰਕਤੀ*dst_stride_ਪੰਕਤੀ;

    ਦਸ਼ਮਲਵ tmp = 0.0f; // partial sum for thread in warp

    ਲਈ (ਪੂਰਨ ਖਾਨਾ = tid; ਖਾਨਾ < ncols; ਖਾਨਾ += ਖੰਡ_ਮਾਪ_3) {
        ਅਚਲ ਦਸ਼ਮਲਵ xi = x[ਖਾਨਾ * src_stride_ਖਾਨਾ];
        tmp += xi * xi;
    }

    tmp = ਖੰਡ_reduce<ਖੰਡ_reduce_ਵਿਧੀ::ਜੋੜ_3, warp_ਮਾਪ>(tmp, s_ਜੋੜ, ਖੰਡ_ਮਾਪ_3);
    ਅਚਲ ਦਸ਼ਮਲਵ scale = sycl::rsqrt(sycl::fmax(tmp, eps * eps));

    ਲਈ (ਪੂਰਨ ਖਾਨਾ = tid; ਖਾਨਾ < ncols; ਖਾਨਾ += ਖੰਡ_ਮਾਪ_3) {
        dst[ਖਾਨਾ * dst_stride_ਖਾਨਾ] = scale * x[ਖਾਨਾ * src_stride_ਖਾਨਾ];
    }
}

ਸਥਿਰ ਖਾਲੀ norm_f32_sycl(ਅਚਲ ਦਸ਼ਮਲਵ * x, ਦਸ਼ਮਲਵ * dst, ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows, ਅਚਲ ਪੂਰਨ nchannels, ਅਚਲ ਪੂਰਨ nsamples,
    ਅਚਲ int64_t src_stride_ਖਾਨਾ, ਅਚਲ int64_t src_stride_ਪੰਕਤੀ, ਅਚਲ int64_t src_stride_channel, ਅਚਲ int64_t src_stride_sample,
    ਅਚਲ int64_t dst_stride_ਖਾਨਾ, ਅਚਲ int64_t dst_stride_ਪੰਕਤੀ, ਅਚਲ int64_t dst_stride_channel, ਅਚਲ int64_t dst_stride_sample,
        ਅਚਲ ਦਸ਼ਮਲਵ eps, ਕਤਾਰ_ptr stream, ਪੂਰਨ device) {

    ਅਚਲ sycl::range<3> global_dims(nsamples, nchannels, nrows);
    ਜੇ (ncols < 1024) {
        ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, 1, WARP_ਮਾਪ);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * ਖੰਡ_dims, ਖੰਡ_dims),
                [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1)
                [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                    norm_f32(x, dst, ncols,
                        src_stride_ਖਾਨਾ, src_stride_ਪੰਕਤੀ, src_stride_channel, src_stride_sample,
                        dst_stride_ਖਾਨਾ, dst_stride_ਪੰਕਤੀ, dst_stride_channel, dst_stride_sample,
                        eps, ਵਸਤੂ_ct1, ਖਾਲੀ_ਸੂਚਕ, WARP_ਮਾਪ);
                });
            });
    }
    ਨਹੀਂਤਾਂ {
        ਅਚਲ ਪੂਰਨ work_ਸਮੂਹ_ਮਾਪ = ggml_sycl_info().max_work_ਸਮੂਹ_sizes[device];
        ਪ੍ਰਤਿਜ੍ਞਾ(work_ਸਮੂਹ_ਮਾਪ % (WARP_ਮਾਪ * WARP_ਮਾਪ) == 0);
        ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, 1, work_ਸਮੂਹ_ਮਾਪ);
        /*
        DPCT1049:17: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<sycl::float2, 1> s_ਜੋੜ_acc_ct1(
                            sycl::range<1>(work_ਸਮੂਹ_ਮਾਪ / WARP_ਮਾਪ), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * ਖੰਡ_dims, ਖੰਡ_dims),
                [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1)
                [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                    norm_f32(x, dst, ncols,
                        src_stride_ਖਾਨਾ, src_stride_ਪੰਕਤੀ, src_stride_channel, src_stride_sample,
                        dst_stride_ਖਾਨਾ, dst_stride_ਪੰਕਤੀ, dst_stride_channel, dst_stride_sample,
                        eps, ਵਸਤੂ_ct1, get_ਸੂਚਕ(s_ਜੋੜ_acc_ct1), work_ਸਮੂਹ_ਮਾਪ);
                });
            });
    }
}

ਸਥਿਰ ਖਾਲੀ ਸਮੂਹ_norm_f32_sycl(ਅਚਲ ਦਸ਼ਮਲਵ* x, ਦਸ਼ਮਲਵ* dst,
    ਅਚਲ ਪੂਰਨ ਅੰਕ_groups, ਅਚਲ ਦਸ਼ਮਲਵ eps, ਅਚਲ ਪੂਰਨ ਸਮੂਹ_ਮਾਪ,
    ਅਚਲ ਪੂਰਨ ne_elements, ਕਤਾਰ_ptr stream, ਪੂਰਨ device) {
    ਜੇ (ਸਮੂਹ_ਮਾਪ < 1024) {
        ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, 1, WARP_ਮਾਪ);
        stream->submit([&](sycl::handler& cgh) {
            ਅਚਲ ਦਸ਼ਮਲਵ eps_ct4 = eps;
            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, ਅੰਕ_groups) * ਖੰਡ_dims,
                    ਖੰਡ_dims),
                [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1)
                [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                    ਸਮੂਹ_norm_f32(
                        x, dst, ਸਮੂਹ_ਮਾਪ, ne_elements, eps_ct4, ਵਸਤੂ_ct1,
                        ਖਾਲੀ_ਸੂਚਕ, WARP_ਮਾਪ);
                });
            });
    }
    ਨਹੀਂਤਾਂ {
        ਅਚਲ ਪੂਰਨ work_ਸਮੂਹ_ਮਾਪ = ggml_sycl_info().max_work_ਸਮੂਹ_sizes[device];
        ਪ੍ਰਤਿਜ੍ਞਾ(work_ਸਮੂਹ_ਮਾਪ % (WARP_ਮਾਪ * WARP_ਮਾਪ) == 0);
        ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, 1, work_ਸਮੂਹ_ਮਾਪ);
        /*
        DPCT1049:18: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */

        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<ਦਸ਼ਮਲਵ, 1> s_ਜੋੜ_acc_ct1(sycl::range<1>(work_ਸਮੂਹ_ਮਾਪ / WARP_ਮਾਪ),
                cgh);

            ਅਚਲ ਦਸ਼ਮਲਵ eps_ct4 = eps;

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, ਅੰਕ_groups) * ਖੰਡ_dims,
                    ਖੰਡ_dims),
                [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1)
                [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                    ਸਮੂਹ_norm_f32(x, dst, ਸਮੂਹ_ਮਾਪ, ne_elements,
                        eps_ct4, ਵਸਤੂ_ct1,
                        get_ਸੂਚਕ(s_ਜੋੜ_acc_ct1), work_ਸਮੂਹ_ਮਾਪ);
                });
            });
    }
}

ਸਥਿਰ ਖਾਲੀ rms_norm_f32_sycl(ਅਚਲ ਦਸ਼ਮਲਵ* x, ਦਸ਼ਮਲਵ* dst, ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows, ਅਚਲ ਪੂਰਨ nchannels, ਅਚਲ ਪੂਰਨ nsamples,
    ਅਚਲ int64_t src_stride_ਖਾਨਾ, ਅਚਲ int64_t src_stride_ਪੰਕਤੀ, ਅਚਲ int64_t src_stride_channel, ਅਚਲ int64_t src_stride_sample,
    ਅਚਲ int64_t dst_stride_ਖਾਨਾ, ਅਚਲ int64_t dst_stride_ਪੰਕਤੀ, ਅਚਲ int64_t dst_stride_channel, ਅਚਲ int64_t dst_stride_sample,
    ਅਚਲ ਦਸ਼ਮਲਵ eps, ਕਤਾਰ_ptr stream, ਪੂਰਨ device) {
    // printf("%s ncols=%d, nrows=%d, WARP_SIZE=%d\n", __func__, ncols, nrows, WARP_SIZE);

    ਅਚਲ sycl::range<3> global_dims(nsamples, nchannels, nrows);
    ਜੇ (ncols < 1024) {
        ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, 1, WARP_ਮਾਪ);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * ਖੰਡ_dims, ਖੰਡ_dims),
                [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1)
                [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                    rms_norm_f32(x, dst, ncols,
                        src_stride_ਖਾਨਾ, src_stride_ਪੰਕਤੀ, src_stride_channel, src_stride_sample,
                        dst_stride_ਖਾਨਾ, dst_stride_ਪੰਕਤੀ, dst_stride_channel, dst_stride_sample,
                        eps, ਵਸਤੂ_ct1, ਖਾਲੀ_ਸੂਚਕ, WARP_ਮਾਪ);
                });
            });
    }
    ਨਹੀਂਤਾਂ {
        ਅਚਲ ਪੂਰਨ work_ਸਮੂਹ_ਮਾਪ = ggml_sycl_info().max_work_ਸਮੂਹ_sizes[device];
        ਪ੍ਰਤਿਜ੍ਞਾ(work_ਸਮੂਹ_ਮਾਪ % (WARP_ਮਾਪ * WARP_ਮਾਪ) == 0);
        ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, 1, work_ਸਮੂਹ_ਮਾਪ);
        /*
        DPCT1049:19: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<ਦਸ਼ਮਲਵ, 1> s_ਜੋੜ_acc_ct1(sycl::range<1>(work_ਸਮੂਹ_ਮਾਪ / WARP_ਮਾਪ),
                cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * ਖੰਡ_dims, ਖੰਡ_dims),
                [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1)
                [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                    rms_norm_f32(x, dst, ncols,
                        src_stride_ਖਾਨਾ, src_stride_ਪੰਕਤੀ, src_stride_channel, src_stride_sample,
                        dst_stride_ਖਾਨਾ, dst_stride_ਪੰਕਤੀ, dst_stride_channel, dst_stride_sample,
                        eps, ਵਸਤੂ_ct1, get_ਸੂਚਕ(s_ਜੋੜ_acc_ct1), work_ਸਮੂਹ_ਮਾਪ);
                });
            });
    }
}

ਸਥਿਰ ਖਾਲੀ rms_norm_scale_f32_sycl(ਅਚਲ ਦਸ਼ਮਲਵ* x, ਦਸ਼ਮਲਵ* dst, ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows,
    ਅਚਲ ਪੂਰਨ nchannels, ਅਚਲ ਪੂਰਨ nsamples,
    ਅਚਲ int64_t src_stride_ਖਾਨਾ, ਅਚਲ int64_t src_stride_ਪੰਕਤੀ, ਅਚਲ int64_t src_stride_channel, ਅਚਲ int64_t src_stride_sample,
    ਅਚਲ int64_t dst_stride_ਖਾਨਾ, ਅਚਲ int64_t dst_stride_ਪੰਕਤੀ, ਅਚਲ int64_t dst_stride_channel, ਅਚਲ int64_t dst_stride_sample,
    ਅਚਲ ਦਸ਼ਮਲਵ eps, ਅਚਲ ਦਸ਼ਮਲਵ scale_mul, ਕਤਾਰ_ptr stream, ਪੂਰਨ device) {
    ਅਚਲ sycl::range<3> global_dims(nsamples, nchannels, nrows);
    ਜੇ (ncols < 1024) {
        ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, 1, WARP_ਮਾਪ);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * ਖੰਡ_dims, ਖੰਡ_dims),
                [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1)
                [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                    rms_norm_f32(x, dst, ncols,
                        src_stride_ਖਾਨਾ, src_stride_ਪੰਕਤੀ, src_stride_channel, src_stride_sample,
                        dst_stride_ਖਾਨਾ, dst_stride_ਪੰਕਤੀ, dst_stride_channel, dst_stride_sample,
                        eps, ਵਸਤੂ_ct1, ਖਾਲੀ_ਸੂਚਕ, WARP_ਮਾਪ,
                        ਖਾਲੀ_ਸੂਚਕ, 0, 0, 0, 0, 0, 0,
                        ਖਾਲੀ_ਸੂਚਕ, 0, 0, 0, 0, 0, 0,
                        scale_mul);
                });
            });
    }
    ਨਹੀਂਤਾਂ {
        ਅਚਲ ਪੂਰਨ work_ਸਮੂਹ_ਮਾਪ = ggml_sycl_info().max_work_ਸਮੂਹ_sizes[device];
        ਪ੍ਰਤਿਜ੍ਞਾ(work_ਸਮੂਹ_ਮਾਪ % (WARP_ਮਾਪ * WARP_ਮਾਪ) == 0);
        ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, 1, work_ਸਮੂਹ_ਮਾਪ);
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<ਦਸ਼ਮਲਵ, 1> s_ਜੋੜ_acc_ct1(sycl::range<1>(work_ਸਮੂਹ_ਮਾਪ / WARP_ਮਾਪ),
                cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * ਖੰਡ_dims, ਖੰਡ_dims),
                [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1)
                [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                    rms_norm_f32(x, dst, ncols,
                        src_stride_ਖਾਨਾ, src_stride_ਪੰਕਤੀ, src_stride_channel, src_stride_sample,
                        dst_stride_ਖਾਨਾ, dst_stride_ਪੰਕਤੀ, dst_stride_channel, dst_stride_sample,
                        eps, ਵਸਤੂ_ct1, get_ਸੂਚਕ(s_ਜੋੜ_acc_ct1), work_ਸਮੂਹ_ਮਾਪ,
                        ਖਾਲੀ_ਸੂਚਕ, 0, 0, 0, 0, 0, 0,
                        ਖਾਲੀ_ਸੂਚਕ, 0, 0, 0, 0, 0, 0,
                        scale_mul);
                });
            });
    }
}

ਸਥਿਰ ਖਾਲੀ rms_norm_mul_f32_sycl(ਅਚਲ ਦਸ਼ਮਲਵ* x, ਅਚਲ ਦਸ਼ਮਲਵ* mul, ਦਸ਼ਮਲਵ* dst, ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows,
        ਅਚਲ ਪੂਰਨ nchannels, ਅਚਲ ਪੂਰਨ nsamples,
        ਅਚਲ int64_t src_stride_ਖਾਨਾ, ਅਚਲ int64_t src_stride_ਪੰਕਤੀ, ਅਚਲ int64_t src_stride_channel, ਅਚਲ int64_t src_stride_sample,
        ਅਚਲ int64_t dst_stride_ਖਾਨਾ, ਅਚਲ int64_t dst_stride_ਪੰਕਤੀ, ਅਚਲ int64_t dst_stride_channel, ਅਚਲ int64_t dst_stride_sample,
        ਅਚਲ int64_t mul_stride_ਪੰਕਤੀ, ਅਚਲ int64_t mul_stride_channel, ਅਚਲ int64_t mul_stride_sample,
        ਅਚਲ ਪੂਰਨ mul_nrows, ਅਚਲ ਪੂਰਨ mul_nchannels, ਅਚਲ ਪੂਰਨ mul_nsamples,
        ਅਚਲ ਦਸ਼ਮਲਵ eps, ਕਤਾਰ_ptr stream, ਪੂਰਨ device) {
    ਅਚਲ sycl::range<3> global_dims(nsamples, nchannels, nrows);
    ਜੇ (ncols < 1024) {
        ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, 1, WARP_ਮਾਪ);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * ਖੰਡ_dims, ਖੰਡ_dims),
                [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1)
                [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                    rms_norm_f32<ਸੱਚ>(x, dst, ncols,
                        src_stride_ਖਾਨਾ, src_stride_ਪੰਕਤੀ, src_stride_channel, src_stride_sample,
                        dst_stride_ਖਾਨਾ, dst_stride_ਪੰਕਤੀ, dst_stride_channel, dst_stride_sample,
                        eps, ਵਸਤੂ_ct1, ਖਾਲੀ_ਸੂਚਕ, WARP_ਮਾਪ,
                        mul, mul_stride_ਪੰਕਤੀ, mul_stride_channel, mul_stride_sample, mul_nrows, mul_nchannels, mul_nsamples);
                });
            });
    }
    ਨਹੀਂਤਾਂ {
        ਅਚਲ ਪੂਰਨ work_ਸਮੂਹ_ਮਾਪ = ggml_sycl_info().max_work_ਸਮੂਹ_sizes[device];
        ਪ੍ਰਤਿਜ੍ਞਾ(work_ਸਮੂਹ_ਮਾਪ % (WARP_ਮਾਪ * WARP_ਮਾਪ) == 0);
        ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, 1, work_ਸਮੂਹ_ਮਾਪ);
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<ਦਸ਼ਮਲਵ, 1> s_ਜੋੜ_acc_ct1(sycl::range<1>(work_ਸਮੂਹ_ਮਾਪ / WARP_ਮਾਪ), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * ਖੰਡ_dims, ਖੰਡ_dims),
                [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1)
                [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                    rms_norm_f32<ਸੱਚ>(x, dst, ncols,
                        src_stride_ਖਾਨਾ, src_stride_ਪੰਕਤੀ, src_stride_channel, src_stride_sample,
                        dst_stride_ਖਾਨਾ, dst_stride_ਪੰਕਤੀ, dst_stride_channel, dst_stride_sample,
                        eps, ਵਸਤੂ_ct1, get_ਸੂਚਕ(s_ਜੋੜ_acc_ct1), work_ਸਮੂਹ_ਮਾਪ,
                        mul, mul_stride_ਪੰਕਤੀ, mul_stride_channel, mul_stride_sample, mul_nrows, mul_nchannels, mul_nsamples);
                });
            });
    }
}

ਸਥਿਰ ਖਾਲੀ rms_norm_mul_ਜੋੜੋ_f32_sycl(ਅਚਲ ਦਸ਼ਮਲਵ* x, ਅਚਲ ਦਸ਼ਮਲਵ* mul, ਅਚਲ ਦਸ਼ਮਲਵ* ਜੋੜੋ_2, ਦਸ਼ਮਲਵ* dst,
        ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows, ਅਚਲ ਪੂਰਨ nchannels, ਅਚਲ ਪੂਰਨ nsamples,
        ਅਚਲ int64_t src_stride_ਖਾਨਾ, ਅਚਲ int64_t src_stride_ਪੰਕਤੀ, ਅਚਲ int64_t src_stride_channel, ਅਚਲ int64_t src_stride_sample,
        ਅਚਲ int64_t dst_stride_ਖਾਨਾ, ਅਚਲ int64_t dst_stride_ਪੰਕਤੀ, ਅਚਲ int64_t dst_stride_channel, ਅਚਲ int64_t dst_stride_sample,
        ਅਚਲ int64_t mul_stride_ਪੰਕਤੀ, ਅਚਲ int64_t mul_stride_channel, ਅਚਲ int64_t mul_stride_sample,
        ਅਚਲ ਪੂਰਨ mul_nrows, ਅਚਲ ਪੂਰਨ mul_nchannels, ਅਚਲ ਪੂਰਨ mul_nsamples,
        ਅਚਲ int64_t ਜੋੜੋ_stride_ਪੰਕਤੀ, ਅਚਲ int64_t ਜੋੜੋ_stride_channel, ਅਚਲ int64_t ਜੋੜੋ_stride_sample,
        ਅਚਲ ਪੂਰਨ ਜੋੜੋ_nrows, ਅਚਲ ਪੂਰਨ ਜੋੜੋ_nchannels, ਅਚਲ ਪੂਰਨ ਜੋੜੋ_nsamples,
        ਅਚਲ ਦਸ਼ਮਲਵ eps, ਕਤਾਰ_ptr stream, ਪੂਰਨ device) {
    ਅਚਲ sycl::range<3> global_dims(nsamples, nchannels, nrows);
    ਜੇ (ncols < 1024) {
        ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, 1, WARP_ਮਾਪ);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * ਖੰਡ_dims, ਖੰਡ_dims),
                [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1)
                [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                    rms_norm_f32<ਸੱਚ, ਸੱਚ>(x, dst, ncols,
                        src_stride_ਖਾਨਾ, src_stride_ਪੰਕਤੀ, src_stride_channel, src_stride_sample,
                        dst_stride_ਖਾਨਾ, dst_stride_ਪੰਕਤੀ, dst_stride_channel, dst_stride_sample,
                        eps, ਵਸਤੂ_ct1, ਖਾਲੀ_ਸੂਚਕ, WARP_ਮਾਪ,
                        mul, mul_stride_ਪੰਕਤੀ, mul_stride_channel, mul_stride_sample, mul_nrows, mul_nchannels, mul_nsamples,
                        ਜੋੜੋ_2, ਜੋੜੋ_stride_ਪੰਕਤੀ, ਜੋੜੋ_stride_channel, ਜੋੜੋ_stride_sample, ਜੋੜੋ_nrows, ਜੋੜੋ_nchannels, ਜੋੜੋ_nsamples);
                });
            });
    }
    ਨਹੀਂਤਾਂ {
        ਅਚਲ ਪੂਰਨ work_ਸਮੂਹ_ਮਾਪ = ggml_sycl_info().max_work_ਸਮੂਹ_sizes[device];
        ਪ੍ਰਤਿਜ੍ਞਾ(work_ਸਮੂਹ_ਮਾਪ % (WARP_ਮਾਪ * WARP_ਮਾਪ) == 0);
        ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, 1, work_ਸਮੂਹ_ਮਾਪ);
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<ਦਸ਼ਮਲਵ, 1> s_ਜੋੜ_acc_ct1(sycl::range<1>(work_ਸਮੂਹ_ਮਾਪ / WARP_ਮਾਪ), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * ਖੰਡ_dims, ਖੰਡ_dims),
                [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1)
                [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                    rms_norm_f32<ਸੱਚ, ਸੱਚ>(x, dst, ncols,
                        src_stride_ਖਾਨਾ, src_stride_ਪੰਕਤੀ, src_stride_channel, src_stride_sample,
                        dst_stride_ਖਾਨਾ, dst_stride_ਪੰਕਤੀ, dst_stride_channel, dst_stride_sample,
                        eps, ਵਸਤੂ_ct1, get_ਸੂਚਕ(s_ਜੋੜ_acc_ct1), work_ਸਮੂਹ_ਮਾਪ,
                        mul, mul_stride_ਪੰਕਤੀ, mul_stride_channel, mul_stride_sample, mul_nrows, mul_nchannels, mul_nsamples,
                        ਜੋੜੋ_2, ਜੋੜੋ_stride_ਪੰਕਤੀ, ਜੋੜੋ_stride_channel, ਜੋੜੋ_stride_sample, ਜੋੜੋ_nrows, ਜੋੜੋ_nchannels, ਜੋੜੋ_nsamples);
                });
            });
    }
}

ਸਾਂਚਾ<ਪੂਰਨ warp_ਮਾਪ>
ਸਥਿਰ ਖਾਲੀ l2_norm_f32_sycl(ਅਚਲ ਦਸ਼ਮਲਵ *   x,
                             ਦਸ਼ਮਲਵ *         dst,
                             ਅਚਲ ਪੂਰਨ       ncols,
                             ਅਚਲ ਪੂਰਨ       nrows,
                             ਅਚਲ ਪੂਰਨ       nchannels,
                             ਅਚਲ ਪੂਰਨ       nsamples,
                             ਅਚਲ int64_t   src_stride_ਖਾਨਾ,
                             ਅਚਲ int64_t   src_stride_ਪੰਕਤੀ,
                             ਅਚਲ int64_t   src_stride_channel,
                             ਅਚਲ int64_t   src_stride_sample,
                             ਅਚਲ int64_t   dst_stride_ਖਾਨਾ,
                             ਅਚਲ int64_t   dst_stride_ਪੰਕਤੀ,
                             ਅਚਲ int64_t   dst_stride_channel,
                             ਅਚਲ int64_t   dst_stride_sample,
                             ਅਚਲ ਦਸ਼ਮਲਵ     eps,
                             ਕਤਾਰ_ptr       stream,
                             ਪੂਰਨ             device) {
    ਅਚਲ dpct::dim3 ਖੰਡਾਂ_ਅੰਕ(nrows, nchannels, nsamples);

    ਜੇ (ncols < 1024) {
        ਅਚਲ dpct::dim3 ਖੰਡ_dims(warp_ਮਾਪ, 1, 1);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(ਖੰਡਾਂ_ਅੰਕ * ਖੰਡ_dims,
                    ਖੰਡ_dims),
                [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1)
                [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(warp_ਮਾਪ)]] {
                    l2_norm_f32<warp_ਮਾਪ>(x, dst, ncols,
                        src_stride_ਖਾਨਾ, src_stride_ਪੰਕਤੀ, src_stride_channel, src_stride_sample,
                        dst_stride_ਖਾਨਾ, dst_stride_ਪੰਕਤੀ, dst_stride_channel, dst_stride_sample,
                        eps, ਵਸਤੂ_ct1,
                        ਖਾਲੀ_ਸੂਚਕ, warp_ਮਾਪ);
                });
            });
    }
    ਨਹੀਂਤਾਂ {
        ਅਚਲ ਪੂਰਨ work_ਸਮੂਹ_ਮਾਪ = ggml_sycl_info().max_work_ਸਮੂਹ_sizes[device];
        ਪ੍ਰਤਿਜ੍ਞਾ(work_ਸਮੂਹ_ਮਾਪ % (warp_ਮਾਪ * warp_ਮਾਪ) == 0);
        ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, 1, work_ਸਮੂਹ_ਮਾਪ);
        ਪੂਰਨ lsm_ਮਾਪ =  ਖੰਡ_dims[2] > warp_ਮਾਪ ? work_ਸਮੂਹ_ਮਾਪ / warp_ਮਾਪ * ਆਕਾਰ(ਦਸ਼ਮਲਵ): 0;
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<ਦਸ਼ਮਲਵ, 1> s_ਜੋੜ_acc_ct1(sycl::range<1>(lsm_ਮਾਪ),
                cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(ਖੰਡਾਂ_ਅੰਕ * ਖੰਡ_dims,
                    ਖੰਡ_dims),
                [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1)
                [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(warp_ਮਾਪ)]] {
                    l2_norm_f32<warp_ਮਾਪ>(x, dst, ncols,
                        src_stride_ਖਾਨਾ, src_stride_ਪੰਕਤੀ, src_stride_channel, src_stride_sample,
                        dst_stride_ਖਾਨਾ, dst_stride_ਪੰਕਤੀ, dst_stride_channel, dst_stride_sample,
                        eps, ਵਸਤੂ_ct1, get_ਸੂਚਕ(s_ਜੋੜ_acc_ct1), work_ਸਮੂਹ_ਮਾਪ);
                });
            });
    }
}

// Batched L2 norm: N independent same-shape F32 tensors in one launch; the tensor
// index is folded into grid dim0 and each row's reduction is identical to the
// single-tensor kernel, so the result is bit-exact.
ਬਣਤਰ l2_batch_ptrs {
    ਅਚਲ ਦਸ਼ਮਲਵ * src[GGML_SYCL_L2_BATCH_MAX];
    ਦਸ਼ਮਲਵ *       dst[GGML_SYCL_L2_BATCH_MAX];
};

// One stride set shared by the whole batch: the caller only groups tensors whose nb[]
// all match, so per-tensor state stays two pointers.
ਬਣਤਰ l2_batch_strides {
    ਪੂਰਨ     ne1, ne2;
    int64_t ss0, ss1, ss2, ss3;
    int64_t ds0, ds1, ds2, ds3;
};

ਸਾਂਚਾ <ਪੂਰਨ warp_ਮਾਪ>
ਸਥਿਰ ਖਾਲੀ l2_norm_f32_batch(l2_batch_ptrs p, l2_batch_strides st, ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਦਸ਼ਮਲਵ eps,
                              ਅਚਲ sycl::nd_ਵਸਤੂ<3> & ਵਸਤੂ_ct1) {
    ਅਚਲ ਪੂਰਨ t   = ਵਸਤੂ_ct1.get_ਸਮੂਹ(0);  // tensor index
    ਅਚਲ ਪੂਰਨ r   = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2);  // flattened row over ne1*ne2*ne3
    ਅਚਲ ਪੂਰਨ tid = ਵਸਤੂ_ct1.get_local_id(2);

    ਅਚਲ ਪੂਰਨ i1 = r % st.ne1;
    ਅਚਲ ਪੂਰਨ i2 = (r / st.ne1) % st.ne2;
    ਅਚਲ ਪੂਰਨ i3 = r / (st.ne1 * st.ne2);

    ਅਚਲ ਦਸ਼ਮਲਵ * x   = p.src[t] + i3 * st.ss3 + i2 * st.ss2 + i1 * st.ss1;
    ਦਸ਼ਮਲਵ *       dst = p.dst[t] + i3 * st.ds3 + i2 * st.ds2 + i1 * st.ds1;

    ਦਸ਼ਮਲਵ tmp = 0.0f;
    ਲਈ (ਪੂਰਨ ਖਾਨਾ = tid; ਖਾਨਾ < ncols; ਖਾਨਾ += warp_ਮਾਪ) {
        ਅਚਲ ਦਸ਼ਮਲਵ xi = x[ਖਾਨਾ * st.ss0];
        tmp += xi * xi;
    }
    tmp = ਖੰਡ_reduce<ਖੰਡ_reduce_ਵਿਧੀ::ਜੋੜ_3, warp_ਮਾਪ>(tmp, (ਦਸ਼ਮਲਵ *) ਖਾਲੀ_ਸੂਚਕ, warp_ਮਾਪ);
    ਅਚਲ ਦਸ਼ਮਲਵ scale = sycl::rsqrt(sycl::fmax(tmp, eps * eps));
    ਲਈ (ਪੂਰਨ ਖਾਨਾ = tid; ਖਾਨਾ < ncols; ਖਾਨਾ += warp_ਮਾਪ) {
        dst[ਖਾਨਾ * st.ds0] = scale * x[ਖਾਨਾ * st.ss0];
    }
}

ਸਾਂਚਾ <ਪੂਰਨ warp_ਮਾਪ>
ਸਥਿਰ ਖਾਲੀ l2_norm_f32_batch_sycl(l2_batch_ptrs p, l2_batch_strides st, ਅਚਲ ਪੂਰਨ n_tensors,
                                   ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows_ਕੁੱਲ, ਅਚਲ ਦਸ਼ਮਲਵ eps,
                                   ਕਤਾਰ_ptr stream) {
    ਅਚਲ dpct::dim3 ਖੰਡਾਂ_ਅੰਕ(nrows_ਕੁੱਲ, 1, n_tensors);
    ਅਚਲ dpct::dim3 ਖੰਡ_dims(warp_ਮਾਪ, 1, 1);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(ਖੰਡਾਂ_ਅੰਕ * ਖੰਡ_dims, ਖੰਡ_dims),
            [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(warp_ਮਾਪ)]] {
                l2_norm_f32_batch<warp_ਮਾਪ>(p, st, ncols, eps, ਵਸਤੂ_ct1);
            });
    });
}

ਖਾਲੀ ggml_sycl_op_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS
    dpct::ਕਤਾਰ_ptr main_ਧਾਰਾ = ctx.stream();
    SYCL_CHECK(ggml_sycl_ਸੈੱਟ_device(ctx.device));
    ਅਚਲ ਦਸ਼ਮਲਵ * src0_dd = static_cast<ਅਚਲ ਦਸ਼ਮਲਵ *>(dst->src[0]->ਅੰਕੜੇ);
    ਦਸ਼ਮਲਵ *       dst_dd  = static_cast<ਦਸ਼ਮਲਵ *>(dst->ਅੰਕੜੇ);

    ਦਸ਼ਮਲਵ eps;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&eps, dst->op_params, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    GGML_ASSERT(eps >= 0.0f);
    ਅਚਲ size_t ts0 = ggml_type_ਮਾਪ(src0->type);
    ਅਚਲ size_t tdst = ggml_type_ਮਾਪ(dst->type);
    GGML_ASSERT(nb00 % ts0 == 0 && nb01 % ts0 == 0 && nb02 % ts0 == 0 && nb03 % ts0 == 0);
    GGML_ASSERT(nb0 % tdst == 0 && nb1 % tdst == 0 && nb2 % tdst == 0 && nb3 % tdst == 0);
    ਅਚਲ int64_t ss0 = nb00 / ts0;
    ਅਚਲ int64_t ss1 = nb01 / ts0;
    ਅਚਲ int64_t ss2 = nb02 / ts0;
    ਅਚਲ int64_t ss3 = nb03 / ts0;
    ਅਚਲ int64_t ds0 = nb0 / tdst;
    ਅਚਲ int64_t ds1 = nb1 / tdst;
    ਅਚਲ int64_t ds2 = nb2 / tdst;
    ਅਚਲ int64_t ds3 = nb3 / tdst;

    norm_f32_sycl(src0_dd, dst_dd, ne00, ne01, ne02, ne03,
        ss0, ss1, ss2, ss3, ds0, ds1, ds2, ds3, eps, main_ਧਾਰਾ, ctx.device);
}

ਖਾਲੀ ggml_sycl_op_ਸਮੂਹ_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    ਪੂਰਨ ਅੰਕ_groups = dst->op_params[0];
    dpct::ਕਤਾਰ_ptr main_ਧਾਰਾ = ctx.stream();
    SYCL_CHECK(ggml_sycl_ਸੈੱਟ_device(ctx.device));

    ਅਚਲ ਦਸ਼ਮਲਵ * src0_dd = static_cast<ਅਚਲ ਦਸ਼ਮਲਵ *>(dst->src[0]->ਅੰਕੜੇ);
    ਦਸ਼ਮਲਵ *       dst_dd  = static_cast<ਦਸ਼ਮਲਵ *>(dst->ਅੰਕੜੇ);

    ਦਸ਼ਮਲਵ eps;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&eps, dst->op_params + 1, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਪੂਰਨ ਸਮੂਹ_ਮਾਪ = dst->src[0]->ne[0] * dst->src[0]->ne[1] * ((dst->src[0]->ne[2] + ਅੰਕ_groups - 1) / ਅੰਕ_groups);
    ਸਮੂਹ_norm_f32_sycl(src0_dd, dst_dd, ਅੰਕ_groups, eps, ਸਮੂਹ_ਮਾਪ, dst->src[0]->ne[0] * dst->src[0]->ne[1] * dst->src[0]->ne[2], main_ਧਾਰਾ, ctx.device);
}

ਖਾਲੀ ggml_sycl_op_rms_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    dpct::ਕਤਾਰ_ptr main_ਧਾਰਾ = ctx.stream();
    SYCL_CHECK(ggml_sycl_ਸੈੱਟ_device(ctx.device));

    ਅਚਲ ਦਸ਼ਮਲਵ * src0_dd = static_cast<ਅਚਲ ਦਸ਼ਮਲਵ *>(dst->src[0]->ਅੰਕੜੇ);
    ਦਸ਼ਮਲਵ *       dst_dd  = static_cast<ਦਸ਼ਮਲਵ *>(dst->ਅੰਕੜੇ);

    ਦਸ਼ਮਲਵ eps;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&eps, dst->op_params, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    GGML_TENSOR_UNARY_OP_LOCALS
    ਅਚਲ size_t ts0 = ggml_type_ਮਾਪ(src0->type);
    ਅਚਲ size_t tdst = ggml_type_ਮਾਪ(dst->type);
    GGML_ASSERT(nb00 % ts0 == 0 && nb01 % ts0 == 0 && nb02 % ts0 == 0 && nb03 % ts0 == 0);
    GGML_ASSERT(nb0 % tdst == 0 && nb1 % tdst == 0 && nb2 % tdst == 0 && nb3 % tdst == 0);
    ਅਚਲ int64_t ss0 = nb00 / ts0;
    ਅਚਲ int64_t ss1 = nb01 / ts0;
    ਅਚਲ int64_t ss2 = nb02 / ts0;
    ਅਚਲ int64_t ss3 = nb03 / ts0;
    ਅਚਲ int64_t ds0 = nb0 / tdst;
    ਅਚਲ int64_t ds1 = nb1 / tdst;
    ਅਚਲ int64_t ds2 = nb2 / tdst;
    ਅਚਲ int64_t ds3 = nb3 / tdst;
    rms_norm_f32_sycl(src0_dd, dst_dd, ne00, ne01, ne02, ne03,
        ss0, ss1, ss2, ss3, ds0, ds1, ds2, ds3, eps, main_ਧਾਰਾ, ctx.device);
}

// Fused rms_norm + scale (the qwen35 GDN l2-norm pair build_gdn_l2_norm emits):
// the scale factor is a host scalar in the GGML_OP_SCALE node's op_params, so
// unlike the mul variants there is no second device tensor to wire up.
ਖਾਲੀ ggml_sycl_op_rms_norm_scale_fused(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                       ggml_tensor * scale_tensor) {
    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(scale_tensor->type == GGML_TYPE_F32);

    dpct::ਕਤਾਰ_ptr main_ਧਾਰਾ = ctx.stream();
    SYCL_CHECK(ggml_sycl_ਸੈੱਟ_device(ctx.device));

    ਅਚਲ ਦਸ਼ਮਲਵ * src0_dd = static_cast<ਅਚਲ ਦਸ਼ਮਲਵ *>(src0->ਅੰਕੜੇ);
    ਦਸ਼ਮਲਵ *       dst_dd  = static_cast<ਦਸ਼ਮਲਵ *>(scale_tensor->ਅੰਕੜੇ);

    ਦਸ਼ਮਲਵ eps;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&eps, dst->op_params, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਦਸ਼ਮਲਵ scale_mul;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&scale_mul, scale_tensor->op_params, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    GGML_ASSERT(scale_mul >= 0.0f);

    GGML_TENSOR_UNARY_OP_LOCALS
    ਅਚਲ size_t ts0 = ggml_type_ਮਾਪ(src0->type);
    ਅਚਲ size_t tdst = ggml_type_ਮਾਪ(scale_tensor->type);
    GGML_ASSERT(nb00 % ts0 == 0 && nb01 % ts0 == 0 && nb02 % ts0 == 0 && nb03 % ts0 == 0);
    GGML_ASSERT(scale_tensor->nb[0] % tdst == 0 && scale_tensor->nb[1] % tdst == 0 &&
                scale_tensor->nb[2] % tdst == 0 && scale_tensor->nb[3] % tdst == 0);
    ਅਚਲ int64_t ss0 = nb00 / ts0;
    ਅਚਲ int64_t ss1 = nb01 / ts0;
    ਅਚਲ int64_t ss2 = nb02 / ts0;
    ਅਚਲ int64_t ss3 = nb03 / ts0;
    ਅਚਲ int64_t ds0 = scale_tensor->nb[0] / tdst;
    ਅਚਲ int64_t ds1 = scale_tensor->nb[1] / tdst;
    ਅਚਲ int64_t ds2 = scale_tensor->nb[2] / tdst;
    ਅਚਲ int64_t ds3 = scale_tensor->nb[3] / tdst;
    rms_norm_scale_f32_sycl(src0_dd, dst_dd, ne00, ne01, ne02, ne03,
        ss0, ss1, ss2, ss3, ds0, ds1, ds2, ds3, eps, scale_mul, main_ਧਾਰਾ, ctx.device);
}

ਖਾਲੀ ggml_sycl_op_rms_norm_fused(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor) {
    ਅਚਲ ggml_tensor * rms_norm_src = dst->src[0];
    ਦਸ਼ਮਲਵ eps = 0.0f;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&eps, dst->op_params, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਅਚਲ ਦਸ਼ਮਲਵ *       src0_dd = static_cast<ਅਚਲ ਦਸ਼ਮਲਵ *>(rms_norm_src->ਅੰਕੜੇ);
    ਅਚਲ ਦਸ਼ਮਲਵ *       mul_dd  = ਖਾਲੀ_ਸੂਚਕ;
    ਅਚਲ ggml_tensor * mul_src = ਖਾਲੀ_ਸੂਚਕ;
    ਜੇ (mul_tensor->src[0] == dst) {
        mul_dd  = static_cast<ਅਚਲ ਦਸ਼ਮਲਵ *>(mul_tensor->src[1]->ਅੰਕੜੇ);
        mul_src = mul_tensor->src[1];
    } ਨਹੀਂਤਾਂ ਜੇ (mul_tensor->src[1] == dst) {
        mul_dd  = static_cast<ਅਚਲ ਦਸ਼ਮਲਵ *>(mul_tensor->src[0]->ਅੰਕੜੇ);
        mul_src = mul_tensor->src[0];
    } ਨਹੀਂਤਾਂ {
        GGML_ASSERT(ਝੂਠ);
    }
    ਦਸ਼ਮਲਵ * dst_dd = static_cast<ਦਸ਼ਮਲਵ *>(mul_tensor->ਅੰਕੜੇ);

    dpct::ਕਤਾਰ_ptr main_ਧਾਰਾ = ctx.stream();
    SYCL_CHECK(ggml_sycl_ਸੈੱਟ_device(ctx.device));

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    ਅਚਲ int64_t ne00 = rms_norm_src->ne[0];
    ਅਚਲ int64_t ne01 = rms_norm_src->ne[1];
    ਅਚਲ int64_t ne02 = rms_norm_src->ne[2];
    ਅਚਲ int64_t ne03 = rms_norm_src->ne[3];

    ਅਚਲ size_t ts0 = ggml_type_ਮਾਪ(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    ਅਚਲ int64_t s00 = rms_norm_src->nb[0] / ts0;
    ਅਚਲ int64_t s01 = rms_norm_src->nb[1] / ts0;
    ਅਚਲ int64_t s02 = rms_norm_src->nb[2] / ts0;
    ਅਚਲ int64_t s03 = rms_norm_src->nb[3] / ts0;

    ਅਚਲ size_t tdst = ggml_type_ਮਾਪ(mul_tensor->type);
    GGML_ASSERT(mul_tensor->nb[0] == tdst);
    ਅਚਲ int64_t d00 = mul_tensor->nb[0] / tdst;
    ਅਚਲ int64_t d01 = mul_tensor->nb[1] / tdst;
    ਅਚਲ int64_t d02 = mul_tensor->nb[2] / tdst;
    ਅਚਲ int64_t d03 = mul_tensor->nb[3] / tdst;

    ਅਚਲ size_t ts_mul = ggml_type_ਮਾਪ(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    ਅਚਲ int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    ਅਚਲ int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    ਅਚਲ int64_t mul_s03 = mul_src->nb[3] / ts_mul;
    ਅਚਲ ਪੂਰਨ mul_nrows     = mul_src->ne[1];
    ਅਚਲ ਪੂਰਨ mul_nchannels = mul_src->ne[2];
    ਅਚਲ ਪੂਰਨ mul_nsamples  = mul_src->ne[3];

    rms_norm_mul_f32_sycl(src0_dd, mul_dd, dst_dd, ne00, ne01, ne02, ne03,
        s00, s01, s02, s03, d00, d01, d02, d03,
        mul_s01, mul_s02, mul_s03, mul_nrows, mul_nchannels, mul_nsamples, eps, main_ਧਾਰਾ, ctx.device);
}

ਖਾਲੀ ggml_sycl_op_rms_norm_fused_ਜੋੜੋ(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                     ggml_tensor * mul_tensor, ggml_tensor * ਜੋੜੋ_tensor) {
    ਅਚਲ ggml_tensor * rms_norm_src = dst->src[0];
    ਦਸ਼ਮਲਵ eps = 0.0f;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&eps, dst->op_params, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਅਚਲ ਦਸ਼ਮਲਵ *       src0_dd = static_cast<ਅਚਲ ਦਸ਼ਮਲਵ *>(rms_norm_src->ਅੰਕੜੇ);
    ਅਚਲ ਦਸ਼ਮਲਵ *       mul_dd  = ਖਾਲੀ_ਸੂਚਕ;
    ਅਚਲ ggml_tensor * mul_src = ਖਾਲੀ_ਸੂਚਕ;
    ਜੇ (mul_tensor->src[0] == dst) {
        mul_dd  = static_cast<ਅਚਲ ਦਸ਼ਮਲਵ *>(mul_tensor->src[1]->ਅੰਕੜੇ);
        mul_src = mul_tensor->src[1];
    } ਨਹੀਂਤਾਂ ਜੇ (mul_tensor->src[1] == dst) {
        mul_dd  = static_cast<ਅਚਲ ਦਸ਼ਮਲਵ *>(mul_tensor->src[0]->ਅੰਕੜੇ);
        mul_src = mul_tensor->src[0];
    } ਨਹੀਂਤਾਂ {
        GGML_ASSERT(ਝੂਠ);
    }

    ਅਚਲ ਦਸ਼ਮਲਵ *       ਜੋੜੋ_dd  = ਖਾਲੀ_ਸੂਚਕ;
    ਅਚਲ ggml_tensor * ਜੋੜੋ_src = ਖਾਲੀ_ਸੂਚਕ;
    ਜੇ (ਜੋੜੋ_tensor->src[0] == mul_tensor) {
        ਜੋੜੋ_dd  = static_cast<ਅਚਲ ਦਸ਼ਮਲਵ *>(ਜੋੜੋ_tensor->src[1]->ਅੰਕੜੇ);
        ਜੋੜੋ_src = ਜੋੜੋ_tensor->src[1];
    } ਨਹੀਂਤਾਂ ਜੇ (ਜੋੜੋ_tensor->src[1] == mul_tensor) {
        ਜੋੜੋ_dd  = static_cast<ਅਚਲ ਦਸ਼ਮਲਵ *>(ਜੋੜੋ_tensor->src[0]->ਅੰਕੜੇ);
        ਜੋੜੋ_src = ਜੋੜੋ_tensor->src[0];
    } ਨਹੀਂਤਾਂ {
        GGML_ASSERT(ਝੂਠ);
    }

    ਦਸ਼ਮਲਵ * dst_dd = static_cast<ਦਸ਼ਮਲਵ *>(ਜੋੜੋ_tensor->ਅੰਕੜੇ);

    dpct::ਕਤਾਰ_ptr main_ਧਾਰਾ = ctx.stream();
    SYCL_CHECK(ggml_sycl_ਸੈੱਟ_device(ctx.device));

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(ਜੋੜੋ_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    ਅਚਲ int64_t ne00 = rms_norm_src->ne[0];
    ਅਚਲ int64_t ne01 = rms_norm_src->ne[1];
    ਅਚਲ int64_t ne02 = rms_norm_src->ne[2];
    ਅਚਲ int64_t ne03 = rms_norm_src->ne[3];

    ਅਚਲ size_t ts0 = ggml_type_ਮਾਪ(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    ਅਚਲ int64_t s00 = rms_norm_src->nb[0] / ts0;
    ਅਚਲ int64_t s01 = rms_norm_src->nb[1] / ts0;
    ਅਚਲ int64_t s02 = rms_norm_src->nb[2] / ts0;
    ਅਚਲ int64_t s03 = rms_norm_src->nb[3] / ts0;

    ਅਚਲ size_t tdst = ggml_type_ਮਾਪ(ਜੋੜੋ_tensor->type);
    GGML_ASSERT(ਜੋੜੋ_tensor->nb[0] == tdst);
    ਅਚਲ int64_t d00 = ਜੋੜੋ_tensor->nb[0] / tdst;
    ਅਚਲ int64_t d01 = ਜੋੜੋ_tensor->nb[1] / tdst;
    ਅਚਲ int64_t d02 = ਜੋੜੋ_tensor->nb[2] / tdst;
    ਅਚਲ int64_t d03 = ਜੋੜੋ_tensor->nb[3] / tdst;

    ਅਚਲ size_t ts_mul = ggml_type_ਮਾਪ(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    ਅਚਲ int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    ਅਚਲ int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    ਅਚਲ int64_t mul_s03 = mul_src->nb[3] / ts_mul;
    ਅਚਲ ਪੂਰਨ mul_nrows     = mul_src->ne[1];
    ਅਚਲ ਪੂਰਨ mul_nchannels = mul_src->ne[2];
    ਅਚਲ ਪੂਰਨ mul_nsamples  = mul_src->ne[3];

    ਅਚਲ size_t ts_ਜੋੜੋ = ggml_type_ਮਾਪ(ਜੋੜੋ_src->type);
    GGML_ASSERT(ਜੋੜੋ_src->nb[0] == ts_ਜੋੜੋ);
    ਅਚਲ int64_t ਜੋੜੋ_s01 = ਜੋੜੋ_src->nb[1] / ts_ਜੋੜੋ;
    ਅਚਲ int64_t ਜੋੜੋ_s02 = ਜੋੜੋ_src->nb[2] / ts_ਜੋੜੋ;
    ਅਚਲ int64_t ਜੋੜੋ_s03 = ਜੋੜੋ_src->nb[3] / ts_ਜੋੜੋ;
    ਅਚਲ ਪੂਰਨ ਜੋੜੋ_nrows     = ਜੋੜੋ_src->ne[1];
    ਅਚਲ ਪੂਰਨ ਜੋੜੋ_nchannels = ਜੋੜੋ_src->ne[2];
    ਅਚਲ ਪੂਰਨ ਜੋੜੋ_nsamples  = ਜੋੜੋ_src->ne[3];

    rms_norm_mul_ਜੋੜੋ_f32_sycl(src0_dd, mul_dd, ਜੋੜੋ_dd, dst_dd, ne00, ne01, ne02, ne03,
        s00, s01, s02, s03, d00, d01, d02, d03,
        mul_s01, mul_s02, mul_s03, mul_nrows, mul_nchannels, mul_nsamples,
        ਜੋੜੋ_s01, ਜੋੜੋ_s02, ਜੋੜੋ_s03, ਜੋੜੋ_nrows, ਜੋੜੋ_nchannels, ਜੋੜੋ_nsamples, eps, main_ਧਾਰਾ, ctx.device);
}

ਖਾਲੀ ggml_sycl_op_rms_norm_ਪਿੱਠ(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32); // dz
    GGML_ASSERT(dst->src[1]->type == GGML_TYPE_F32); // x
    GGML_ASSERT(dst->type         == GGML_TYPE_F32);

    ਦਸ਼ਮਲਵ eps = 1e-5f;
    std::ਸਮ੍ਰਿਤੀ_ਨਕਲ(&eps, dst->op_params, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਜੇ (!(eps > 0.0f) || !std::isfinite(eps)) eps = 1e-5f;

    ਅਚਲ ਦਸ਼ਮਲਵ * g_ਅਧਾਰ  = static_cast<ਅਚਲ ਦਸ਼ਮਲਵ *>(dst->src[0]->ਅੰਕੜੇ); // dz
    ਅਚਲ ਦਸ਼ਮਲਵ * x_ਅਧਾਰ  = static_cast<ਅਚਲ ਦਸ਼ਮਲਵ *>(dst->src[1]->ਅੰਕੜੇ); // x
          ਦਸ਼ਮਲਵ * dx_ਅਧਾਰ = static_cast<      ਦਸ਼ਮਲਵ *>(dst->ਅੰਕੜੇ);

    ਅਚਲ int64_t D  = dst->ne[0];
    ਅਚਲ int64_t n1 = dst->ne[1], n2 = dst->ne[2], n3 = dst->ne[3]; (ਖਾਲੀ) n3;
    ਅਚਲ int64_t N  = ggml_nrows(dst);
    ਜੇ (D == 0 || N == 0) ਵਾਪਸ;

    ਅਚਲ ggml_tensor *G = dst->src[0];
    ਅਚਲ ggml_tensor *X = dst->src[1];
    ਅਚਲ ਪੂਰਨ ts = (ਪੂਰਨ) ggml_type_ਮਾਪ(X->type);
    GGML_ASSERT((size_t) X->nb[0]   == (size_t) ts);
    GGML_ASSERT((size_t) G->nb[0]   == (size_t) ts);
    GGML_ASSERT((size_t) dst->nb[0] == (size_t) ts);

    ਅਚਲ int64_t xs1 = X->nb[1] / ts, xs2 = X->nb[2] / ts, xs3 = X->nb[3] / ts;
    ਅਚਲ int64_t gs1 = G->nb[1] / ts, gs2 = G->nb[2] / ts, gs3 = G->nb[3] / ts;
    ਅਚਲ int64_t ds1 = dst->nb[1] / ts, ds2 = dst->nb[2] / ts, ds3 = dst->nb[3] / ts;

    dpct::ਕਤਾਰ_ptr q = ctx.stream();

    // work-group size: multiple of WARP_SIZE, capped by device and 256, and not larger than D
    ਅਚਲ ਪੂਰਨ device_max_wg = ggml_sycl_info().max_work_ਸਮੂਹ_sizes[ctx.device];
    ਸਵੈਚਾਲਿਤ roundup = [](ਪੂਰਨ v, ਪੂਰਨ m) { ਵਾਪਸ ((v + m - 1) / m) * m; };
    ਪੂਰਨ wg_cap = 256;
    ਜੇ (device_max_wg > 0) wg_cap = std::min(wg_cap, device_max_wg);
    ਪੂਰਨ WG = std::max(WARP_ਮਾਪ, std::min(roundup((ਪੂਰਨ)std::min<int64_t>(D, wg_cap), WARP_ਮਾਪ), wg_cap));

    // FP32 path: per-thread compensated accumulation + hierarchical reduction
    q->submit([&](sycl::handler &cgh) {
        ਅਚਲ ਪੂਰਨ nwarps_loc = std::max(1, WG / WARP_ਮਾਪ);
        // store one partial value per warp (xx and xg) for cross-warp reduction
        ਸਵੈਚਾਲਿਤ l_xx   = sycl::local_accessor<sycl::float2, 1>(sycl::range<1>(nwarps_loc), cgh);
        ਸਵੈਚਾਲਿਤ l_xg   = sycl::local_accessor<sycl::float2, 1>(sycl::range<1>(nwarps_loc), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, N) * sycl::range<3>(1, 1, WG),
                              sycl::range<3>(1, 1, WG)),
            [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2);
                ਅਚਲ ਪੂਰਨ tid = ਵਸਤੂ_ct1.get_local_id(2);

                ਅਚਲ int64_t i1 = ਪੰਕਤੀ % n1;
                ਅਚਲ int64_t i2 = (ਪੰਕਤੀ / n1) % n2;
                ਅਚਲ int64_t i3 = ਪੰਕਤੀ / (n1 * n2);

                ਅਚਲ ਦਸ਼ਮਲਵ *__restrict x_ਪੰਕਤੀ = x_ਅਧਾਰ + i3 * xs3 + i2 * xs2 + i1 * xs1;
                ਅਚਲ ਦਸ਼ਮਲਵ *__restrict g_ਪੰਕਤੀ = g_ਅਧਾਰ + i3 * gs3 + i2 * gs2 + i1 * gs1;
                ਦਸ਼ਮਲਵ *__restrict d_ਪੰਕਤੀ       = dx_ਅਧਾਰ + i3 * ds3 + i2 * ds2 + i1 * ds1;

                // per-thread accumulation (compensated by default)
                ਦਸ਼ਮਲਵ ਜੋੜ_xx = 0.f, ਜੋੜ_xg = 0.f;
#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_RMS_ਪਿੱਠ_FAST
                ਦਸ਼ਮਲਵ c_xx = 0.f, c_xg = 0.f;
#ਜੇ_ਅੰਤ
                ਲਈ (int64_t ਖਾਨਾ = tid; ਖਾਨਾ < D; ਖਾਨਾ += WG) {
                    ਅਚਲ ਦਸ਼ਮਲਵ xv = x_ਪੰਕਤੀ[ਖਾਨਾ];
                    ਅਚਲ ਦਸ਼ਮਲਵ gv = g_ਪੰਕਤੀ[ਖਾਨਾ];
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_RMS_ਪਿੱਠ_FAST
                    ਜੋੜ_xx += xv * xv;
                    ਜੋੜ_xg += xv * gv;
#ਨਹੀਂਤਾਂ
                    ਦਸ਼ਮਲਵ y1 = xv * xv - c_xx;
                    ਦਸ਼ਮਲਵ t1 = ਜੋੜ_xx + y1;
                    c_xx = (t1 - ਜੋੜ_xx) - y1;
                    ਜੋੜ_xx = t1;

                    ਦਸ਼ਮਲਵ y2 = xv * gv - c_xg;
                    ਦਸ਼ਮਲਵ t2 = ਜੋੜ_xg + y2;
                    c_xg = (t2 - ਜੋੜ_xg) - y2;
                    ਜੋੜ_xg = t2;
#ਜੇ_ਅੰਤ
                }

                // warp-level reduction
                sycl::float2 xx = sycl::float2(ਜੋੜ_xx,
#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_RMS_ਪਿੱਠ_FAST
                    c_xx
#ਨਹੀਂਤਾਂ
                    0.f
#ਜੇ_ਅੰਤ
                );
                sycl::float2 xg = sycl::float2(ਜੋੜ_xg,
#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_RMS_ਪਿੱਠ_FAST
                    c_xg
#ਨਹੀਂਤਾਂ
                    0.f
#ਜੇ_ਅੰਤ
                );
                xx = warp_reduce_ਜੋੜ(xx, ਵਸਤੂ_ct1);
                xg = warp_reduce_ਜੋੜ(xg, ਵਸਤੂ_ct1);

                // cross-warp reduction using local memory (single barrier)
                ਅਚਲ ਸਵੈਚਾਲਿਤ sub_ਸਮੂਹ = ਵਸਤੂ_ct1.get_sub_ਸਮੂਹ();
                ਅਚਲ ਸਵੈਚਾਲਿਤ sg_id     = sub_ਸਮੂਹ.get_ਸਮੂਹ_ਰੇਖਿਕ_id();
                ਅਚਲ ਸਵੈਚਾਲਿਤ wi_ਵਿੱਚ_sg  = sub_ਸਮੂਹ.get_local_ਰੇਖਿਕ_id();
                ਅਚਲ ਪੂਰਨ nthreads   = ਵਸਤੂ_ct1.get_local_range(2);
                ਅਚਲ ਪੂਰਨ nwarps     = nthreads / WARP_ਮਾਪ;

                sycl::float2 xx_ਕੁੱਲ = xx;
                sycl::float2 xg_ਕੁੱਲ = xg;
                ਜੇ (nwarps > 1) {
                    ਜੇ (wi_ਵਿੱਚ_sg == 0) {
                        l_xx[sg_id] = xx;
                        l_xg[sg_id] = xg;
                    }
                    ਵਸਤੂ_ct1.barrier(sycl::access::fence_space::local_space);

                    ਜੇ (sg_id == 0) {
                        ਅਚਲ ਬਿਨਾਨਿਸ਼ਾਨੀ wi_u = wi_ਵਿੱਚ_sg;
                        sycl::float2 xx_ਪਹਿਲਾ = (wi_u < static_cast<ਬਿਨਾਨਿਸ਼ਾਨੀ>(nwarps)) ? l_xx[wi_u] : sycl::float2(0.f, 0.f);
                        sycl::float2 xg_ਪਹਿਲਾ = (wi_u < static_cast<ਬਿਨਾਨਿਸ਼ਾਨੀ>(nwarps)) ? l_xg[wi_u] : sycl::float2(0.f, 0.f);
                        xx_ਕੁੱਲ = warp_reduce_ਜੋੜ(xx_ਪਹਿਲਾ, ਵਸਤੂ_ct1);
                        xg_ਕੁੱਲ = warp_reduce_ਜੋੜ(xg_ਪਹਿਲਾ, ਵਸਤੂ_ct1);
                    } ਨਹੀਂਤਾਂ {
                        // other subgroups keep their local totals; they'll be ignored
                        xx_ਕੁੱਲ = xx;
                        xg_ਕੁੱਲ = xg;
                    }
                    // ensure all threads see the first-subgroup result via broadcast below
                }

                // compute inv_r and coeff once per row and broadcast to the whole work-group
                ਦਸ਼ਮਲਵ inv_r = 0.f;
                ਦਸ਼ਮਲਵ coeff = 0.f;
                ਜੇ (tid == 0) {
                    ਅਚਲ ਦਸ਼ਮਲਵ ਜੋੜ_xx_f  = xx_ਕੁੱਲ.x() + xx_ਕੁੱਲ.y();
                    ਅਚਲ ਦਸ਼ਮਲਵ ਜੋੜ_xdz_f = xg_ਕੁੱਲ.x() + xg_ਕੁੱਲ.y();
                    ਅਚਲ ਦਸ਼ਮਲਵ ਔਸਤ_eps  = ਜੋੜ_xx_f / (ਦਸ਼ਮਲਵ) D + eps;
                    ਅਚਲ ਦਸ਼ਮਲਵ ਜੋੜ_eps   = ਜੋੜ_xx_f + eps * (ਦਸ਼ਮਲਵ) D;
                    inv_r = sycl::rsqrt(ਔਸਤ_eps);
                    coeff = -ਜੋੜ_xdz_f / ਜੋੜ_eps;
                }
                inv_r = sycl::ਸਮੂਹ_ਪ੍ਰਸਾਰਣ(ਵਸਤੂ_ct1.get_ਸਮੂਹ(), inv_r);
                coeff = sycl::ਸਮੂਹ_ਪ੍ਰਸਾਰਣ(ਵਸਤੂ_ct1.get_ਸਮੂਹ(), coeff);

                ਲਈ (int64_t ਖਾਨਾ = tid; ਖਾਨਾ < D; ਖਾਨਾ += WG) {
                    d_ਪੰਕਤੀ[ਖਾਨਾ] = (g_ਪੰਕਤੀ[ਖਾਨਾ] + coeff * x_ਪੰਕਤੀ[ਖਾਨਾ]) * inv_r;
                }
            });
    });

}

ਖਾਲੀ ggml_sycl_op_l2_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ਦਸ਼ਮਲਵ * src0_d = (ਅਚਲ ਦਸ਼ਮਲਵ *) src0->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ * dst_d = (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ;
    dpct::ਕਤਾਰ_ptr     stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    ਦਸ਼ਮਲਵ eps;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&eps, dst->op_params, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    GGML_ASSERT(eps >= 0.0f);

    ਅਚਲ size_t ts0 = ggml_type_ਮਾਪ(src0->type);
    ਅਚਲ size_t tdst = ggml_type_ਮਾਪ(dst->type);
    GGML_ASSERT(nb00 % ts0 == 0 && nb01 % ts0 == 0 && nb02 % ts0 == 0 && nb03 % ts0 == 0);
    GGML_ASSERT(nb0 % tdst == 0 && nb1 % tdst == 0 && nb2 % tdst == 0 && nb3 % tdst == 0);
    ਅਚਲ int64_t ss0 = nb00 / ts0;
    ਅਚਲ int64_t ss1 = nb01 / ts0;
    ਅਚਲ int64_t ss2 = nb02 / ts0;
    ਅਚਲ int64_t ss3 = nb03 / ts0;
    ਅਚਲ int64_t ds0 = nb0 / tdst;
    ਅਚਲ int64_t ds1 = nb1 / tdst;
    ਅਚਲ int64_t ds2 = nb2 / tdst;
    ਅਚਲ int64_t ds3 = nb3 / tdst;

    /*support both WARP_SIZE or WARP_32_SIZE in code
      choose by hardware for better performance
    */
    l2_norm_f32_sycl<WARP_ਮਾਪ>(src0_d, dst_d, ne00, ne01, ne02, ne03,
            ss0, ss1, ss2, ss3, ds0, ds1, ds2, ds3, eps, stream, ctx.device);
}

// nodes[0..count) are independent, same-shape, same-eps, same-nb L2_NORM ops validated
// by the caller; requires ncols < 1024 (the warp reduction path).
ਖਾਲੀ ggml_sycl_l2_norm_batch(ggml_backend_sycl_context & ctx, ggml_tensor ** nodes, ਪੂਰਨ count) {
    ਅਚਲ ggml_tensor * s0 = nodes[0]->src[0];
    ਅਚਲ ਪੂਰਨ ncols       = (ਪੂਰਨ) s0->ne[0];
    ਅਚਲ ਪੂਰਨ nrows_ਕੁੱਲ = (ਪੂਰਨ) ggml_nrows(s0);
    ਦਸ਼ਮਲਵ eps;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&eps, nodes[0]->op_params, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    GGML_ASSERT(eps >= 0.0f);

    l2_batch_ptrs p{};
    ਲਈ (ਪੂਰਨ t = 0; t < count; ++t) {
        p.src[t] = (ਅਚਲ ਦਸ਼ਮਲਵ *) nodes[t]->src[0]->ਅੰਕੜੇ;
        p.dst[t] = (ਦਸ਼ਮਲਵ *) nodes[t]->ਅੰਕੜੇ;
    }

    ਅਚਲ ggml_tensor * d0 = nodes[0];
    ਅਚਲ size_t        ts = ggml_type_ਮਾਪ(GGML_TYPE_F32);
    l2_batch_strides    st{};
    st.ne1 = (ਪੂਰਨ) s0->ne[1];
    st.ne2 = (ਪੂਰਨ) s0->ne[2];
    st.ss0 = s0->nb[0] / ts; st.ss1 = s0->nb[1] / ts; st.ss2 = s0->nb[2] / ts; st.ss3 = s0->nb[3] / ts;
    st.ds0 = d0->nb[0] / ts; st.ds1 = d0->nb[1] / ts; st.ds2 = d0->nb[2] / ts; st.ds3 = d0->nb[3] / ts;

    l2_norm_f32_batch_sycl<WARP_ਮਾਪ>(p, st, count, ncols, nrows_ਕੁੱਲ, eps, ctx.stream());
}
