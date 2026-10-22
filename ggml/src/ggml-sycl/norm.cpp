#ಅನ್ತರ್ಭಾವ "norm.hpp"
#ಅನ್ತರ್ಭಾವ "ggml-sycl/common.hpp"
#ಅನ್ತರ್ಭಾವ "ggml-sycl/presets.hpp"

ಸ್ಥಿರ ಶೂನ್ಯ norm_f32(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* x, ದಶಮಾಂಶ* dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
    ಸ್ಥಿರಾಂಕ int64_t src_stride_ಲಂಬಸಾಲು, ಸ್ಥಿರಾಂಕ int64_t src_stride_ಅಡ್ಡಸಾಲು, ಸ್ಥಿರಾಂಕ int64_t src_stride_channel, ಸ್ಥಿರಾಂಕ int64_t src_stride_sample,
    ಸ್ಥಿರಾಂಕ int64_t dst_stride_ಲಂಬಸಾಲು, ಸ್ಥಿರಾಂಕ int64_t dst_stride_ಅಡ್ಡಸಾಲು, ಸ್ಥಿರಾಂಕ int64_t dst_stride_channel, ಸ್ಥಿರಾಂಕ int64_t dst_stride_sample,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ eps, ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3>& ಅಂಶ_ct1, sycl::float2* s_ಮೊತ್ತ, ಪೂರ್ಣಾಂಕ ಖಂಡ_ಗಾತ್ರ_3) {

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nthreads = ಅಂಶ_ct1.get_local_range(2);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ sample  = ಅಂಶ_ct1.get_ಗುಂಪು(0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ channel = ಅಂಶ_ct1.get_ಗುಂಪು(1);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು     = ಅಂಶ_ct1.get_ಗುಂಪು(2);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ tid = ಅಂಶ_ct1.get_local_id(2);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nwarps = nthreads / WARP_ಗಾತ್ರ;

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ src_ಸ್ಥಾನಾಂತರ = calculate_ಸ್ಥಾನಾಂತರ<3>({src_stride_sample, src_stride_channel, src_stride_ಅಡ್ಡಸಾಲು}, {sample, channel, ಅಡ್ಡಸಾಲು});
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ dst_ಸ್ಥಾನಾಂತರ = calculate_ಸ್ಥಾನಾಂತರ<3>({dst_stride_sample, dst_stride_channel, dst_stride_ಅಡ್ಡಸಾಲು}, {sample, channel, ಅಡ್ಡಸಾಲು});

    x += src_ಸ್ಥಾನಾಂತರ;
    dst += dst_ಸ್ಥಾನಾಂತರ;

    sycl::float2 ಸರಾಸರಿ_var = sycl::float2(0.f, 0.f);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು = tid; ಲಂಬಸಾಲು < ncols; ಲಂಬಸಾಲು += ಖಂಡ_ಗಾತ್ರ_3) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ xi = x[ಲಂಬಸಾಲು * src_stride_ಲಂಬಸಾಲು];
        ಸರಾಸರಿ_var.x() += xi;
        ಸರಾಸರಿ_var.y() += xi * xi;
    }

    // sum up partial sums
    ಸರಾಸರಿ_var = warp_reduce_ಮೊತ್ತ(ಸರಾಸರಿ_var, ಅಂಶ_ct1);
    ಆದರೆ  (ಖಂಡ_ಗಾತ್ರ_3 > WARP_ಗಾತ್ರ) {
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ sub_ಗುಂಪು = ಅಂಶ_ct1.get_sub_ಗುಂಪು();
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ sg_id = sub_ಗುಂಪು.get_ಗುಂಪು_ರೇಖೀಯ_id();
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ wi_ಒಳ_sg = sub_ಗುಂಪು.get_local_ರೇಖೀಯ_id();
        ಆದರೆ (wi_ಒಳ_sg == 0) {
            s_ಮೊತ್ತ[sg_id] = ಸರಾಸರಿ_var;
        }
        ಅಂಶ_ct1.barrier(sycl::access::fence_space::local_space);
        ಸರಾಸರಿ_var = 0.f;
        ಸ್ಥಿರಾಂಕ size_t nreduce = ceil_div(nwarps, WARP_ಗಾತ್ರ);
        ಗಾಗಿ (size_t i = 0; i < nreduce; i += 1)
        {
            ಸರಾಸರಿ_var += s_ಮೊತ್ತ[wi_ಒಳ_sg + i * WARP_ಗಾತ್ರ];
        }
        ಸರಾಸರಿ_var = warp_reduce_ಮೊತ್ತ(ಸರಾಸರಿ_var, ಅಂಶ_ct1);
    }

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ಸರಾಸರಿ_2 = ಸರಾಸರಿ_var.x() / ncols;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ var = ಸರಾಸರಿ_var.y() / ncols - ಸರಾಸರಿ_2 * ಸರಾಸರಿ_2;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ inv_std = sycl::rsqrt(var + eps);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು = tid; ಲಂಬಸಾಲು < ncols; ಲಂಬಸಾಲು += ಖಂಡ_ಗಾತ್ರ_3) {
        dst[ಲಂಬಸಾಲು * dst_stride_ಲಂಬಸಾಲು] = (x[ಲಂಬಸಾಲು * src_stride_ಲಂಬಸಾಲು] - ಸರಾಸರಿ_2) * inv_std;
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ಗುಂಪು_norm_f32(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* x, ದಶಮಾಂಶ* dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಗುಂಪು_ಗಾತ್ರ, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne_elements, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ eps,
    ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3>& ಅಂಶ_ct1, ದಶಮಾಂಶ* s_ಮೊತ್ತ, ಪೂರ್ಣಾಂಕ ಖಂಡ_ಗಾತ್ರ_3) {
    ಪೂರ್ಣಾಂಕ start = ಅಂಶ_ct1.get_ಗುಂಪು(2) * ಗುಂಪು_ಗಾತ್ರ;
    ಪೂರ್ಣಾಂಕ end = start + ಗುಂಪು_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nthreads = ಅಂಶ_ct1.get_local_range(2);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nwarps = nthreads / WARP_ಗಾತ್ರ;
    start += ಅಂಶ_ct1.get_local_id(2);
    size_t nreduce = nwarps / WARP_ಗಾತ್ರ;

    ಆದರೆ (end >= ne_elements) {
        end = ne_elements;
    }

    ದಶಮಾಂಶ tmp = 0.0f; // partial sum for thread in warp

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = start; j < end; j += ಖಂಡ_ಗಾತ್ರ_3) {
        tmp += x[j];
    }

    tmp = warp_reduce_ಮೊತ್ತ(tmp, ಅಂಶ_ct1);
    ಆದರೆ (ಖಂಡ_ಗಾತ್ರ_3 > WARP_ಗಾತ್ರ) {

        ಪೂರ್ಣಾಂಕ warp_id = ಅಂಶ_ct1.get_local_id(2) / WARP_ಗಾತ್ರ;
        ಪೂರ್ಣಾಂಕ lane_id = ಅಂಶ_ct1.get_local_id(2) % WARP_ಗಾತ್ರ;
        ಆದರೆ (lane_id == 0) {
            s_ಮೊತ್ತ[warp_id] = tmp;
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
        ಅಂಶ_ct1.barrier();
        tmp = 0.f;
        ಗಾಗಿ (size_t i = 0; i < nreduce; i += 1)
        {
            tmp += s_ಮೊತ್ತ[lane_id + i * WARP_ಗಾತ್ರ];
        }
        tmp = warp_reduce_ಮೊತ್ತ(tmp, ಅಂಶ_ct1);
    }

    ದಶಮಾಂಶ ಸರಾಸರಿ_2 = tmp / ಗುಂಪು_ಗಾತ್ರ;
    tmp = 0.0f;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = start; j < end; j += ಖಂಡ_ಗಾತ್ರ_3) {
        ದಶಮಾಂಶ xi = x[j] - ಸರಾಸರಿ_2;
        dst[j] = xi;
        tmp += xi * xi;
    }

    tmp = warp_reduce_ಮೊತ್ತ(tmp, ಅಂಶ_ct1);
    ಆದರೆ (ಖಂಡ_ಗಾತ್ರ_3 > WARP_ಗಾತ್ರ) {

        ಪೂರ್ಣಾಂಕ warp_id = ಅಂಶ_ct1.get_local_id(2) / WARP_ಗಾತ್ರ;
        ಪೂರ್ಣಾಂಕ lane_id = ಅಂಶ_ct1.get_local_id(2) % WARP_ಗಾತ್ರ;
        ಆದರೆ (lane_id == 0) {
            s_ಮೊತ್ತ[warp_id] = tmp;
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
        ಅಂಶ_ct1.barrier();
        tmp = 0.f;
        ಗಾಗಿ (size_t i = 0; i < nreduce; i += 1)
        {
            tmp += s_ಮೊತ್ತ[lane_id + i * WARP_ಗಾತ್ರ];
        }
        tmp = warp_reduce_ಮೊತ್ತ(tmp, ಅಂಶ_ct1);
    }

    ದಶಮಾಂಶ variance = tmp / ಗುಂಪು_ಗಾತ್ರ;
    ದಶಮಾಂಶ scale = sycl::rsqrt(variance + eps);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = start; j < end; j += ಖಂಡ_ಗಾತ್ರ_3) {
        dst[j] *= scale;
    }
}

ಮಾದರಿ <ತಾರ್ಕಿಕ do_multiply = ಸುಳ್ಳು, ತಾರ್ಕಿಕ do_ಕೂಡಿಸು = ಸುಳ್ಳು>
ಸ್ಥಿರ ಶೂನ್ಯ rms_norm_f32(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* x, ದಶಮಾಂಶ* dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
    ಸ್ಥಿರಾಂಕ int64_t src_stride_ಲಂಬಸಾಲು, ಸ್ಥಿರಾಂಕ int64_t src_stride_ಅಡ್ಡಸಾಲು, ಸ್ಥಿರಾಂಕ int64_t src_stride_channel, ಸ್ಥಿರಾಂಕ int64_t src_stride_sample,
    ಸ್ಥಿರಾಂಕ int64_t dst_stride_ಲಂಬಸಾಲು, ಸ್ಥಿರಾಂಕ int64_t dst_stride_ಅಡ್ಡಸಾಲು, ಸ್ಥಿರಾಂಕ int64_t dst_stride_channel, ಸ್ಥಿರಾಂಕ int64_t dst_stride_sample,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ eps, ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3>& ಅಂಶ_ct1, ದಶಮಾಂಶ* s_ಮೊತ್ತ, ಪೂರ್ಣಾಂಕ ಖಂಡ_ಗಾತ್ರ_3,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* mul = ಶೂನ್ಯಸೂಚಕ, ಸ್ಥಿರಾಂಕ int64_t mul_stride_ಅಡ್ಡಸಾಲು = 0, ಸ್ಥಿರಾಂಕ int64_t mul_stride_channel = 0,
    ಸ್ಥಿರಾಂಕ int64_t mul_stride_sample = 0, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ mul_nrows = 0, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ mul_nchannels = 0, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ mul_nsamples = 0,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* ಕೂಡಿಸು_2 = ಶೂನ್ಯಸೂಚಕ, ಸ್ಥಿರಾಂಕ int64_t ಕೂಡಿಸು_stride_ಅಡ್ಡಸಾಲು = 0, ಸ್ಥಿರಾಂಕ int64_t ಕೂಡಿಸು_stride_channel = 0,
    ಸ್ಥಿರಾಂಕ int64_t ಕೂಡಿಸು_stride_sample = 0, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಕೂಡಿಸು_nrows = 0, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಕೂಡಿಸು_nchannels = 0, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಕೂಡಿಸು_nsamples = 0,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale_mul = 1.0f) {

    static_assert(!do_ಕೂಡಿಸು || do_multiply, "fusing add is not supported without multiplying");

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ sample  = ಅಂಶ_ct1.get_ಗುಂಪು(0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ channel = ಅಂಶ_ct1.get_ಗುಂಪು(1);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು     = ಅಂಶ_ct1.get_ಗುಂಪು(2);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nthreads = ಅಂಶ_ct1.get_local_range(2);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ tid = ಅಂಶ_ct1.get_local_id(2);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nwarps = nthreads / WARP_ಗಾತ್ರ;

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ src_ಸ್ಥಾನಾಂತರ = calculate_ಸ್ಥಾನಾಂತರ<3>({src_stride_sample, src_stride_channel, src_stride_ಅಡ್ಡಸಾಲು}, {sample, channel, ಅಡ್ಡಸಾಲು});
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ dst_ಸ್ಥಾನಾಂತರ = calculate_ಸ್ಥಾನಾಂತರ<3>({dst_stride_sample, dst_stride_channel, dst_stride_ಅಡ್ಡಸಾಲು}, {sample, channel, ಅಡ್ಡಸಾಲು});

    x   += src_ಸ್ಥಾನಾಂತರ;
    dst += dst_ಸ್ಥಾನಾಂತರ;

    ಆದರೆ constexpr (do_multiply) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ mul_ಅಡ್ಡಸಾಲು     = ಅಡ್ಡಸಾಲು     % mul_nrows;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ mul_channel = channel % mul_nchannels;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ mul_sample  = sample  % mul_nsamples;
        mul += mul_sample * mul_stride_sample + mul_channel * mul_stride_channel + mul_ಅಡ್ಡಸಾಲು * mul_stride_ಅಡ್ಡಸಾಲು;
    }

    ಆದರೆ constexpr (do_ಕೂಡಿಸು) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಕೂಡಿಸು_ಅಡ್ಡಸಾಲು     = ಅಡ್ಡಸಾಲು     % ಕೂಡಿಸು_nrows;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಕೂಡಿಸು_channel = channel % ಕೂಡಿಸು_nchannels;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಕೂಡಿಸು_sample  = sample  % ಕೂಡಿಸು_nsamples;
        ಕೂಡಿಸು_2 += ಕೂಡಿಸು_sample * ಕೂಡಿಸು_stride_sample + ಕೂಡಿಸು_channel * ಕೂಡಿಸು_stride_channel + ಕೂಡಿಸು_ಅಡ್ಡಸಾಲು * ಕೂಡಿಸು_stride_ಅಡ್ಡಸಾಲು;
    }

    ದಶಮಾಂಶ tmp = 0.0f; // partial sum for thread in warp

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು = tid; ಲಂಬಸಾಲು < ncols; ಲಂಬಸಾಲು += ಖಂಡ_ಗಾತ್ರ_3) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ xi = x[ಲಂಬಸಾಲು * src_stride_ಲಂಬಸಾಲು];
        tmp += xi * xi;
    }

    // sum up partial sums
    tmp = warp_reduce_ಮೊತ್ತ(tmp, ಅಂಶ_ct1);
    ಆದರೆ (ಖಂಡ_ಗಾತ್ರ_3 > WARP_ಗಾತ್ರ) {
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ sub_ಗುಂಪು = ಅಂಶ_ct1.get_sub_ಗುಂಪು();
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ sg_id = sub_ಗುಂಪು.get_ಗುಂಪು_ರೇಖೀಯ_id();
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ wi_ಒಳ_sg = sub_ಗುಂಪು.get_local_ರೇಖೀಯ_id();
        ಆದರೆ (wi_ಒಳ_sg == 0) {
            s_ಮೊತ್ತ[sg_id] = tmp;
        }

        ಅಂಶ_ct1.barrier(sycl::access::fence_space::local_space);
        ಸ್ಥಿರಾಂಕ size_t nreduce = ceil_div(nwarps, WARP_ಗಾತ್ರ);
        tmp = 0.f;
        ಗಾಗಿ (size_t i = 0; i < nreduce; i += 1)
        {
            tmp += s_ಮೊತ್ತ[wi_ಒಳ_sg + i * WARP_ಗಾತ್ರ];
        }
        tmp = warp_reduce_ಮೊತ್ತ(tmp, ಅಂಶ_ct1);
    }

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ಸರಾಸರಿ_2 = tmp / ncols;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale = sycl::rsqrt(ಸರಾಸರಿ_2 + eps);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು = tid; ಲಂಬಸಾಲು < ncols; ಲಂಬಸಾಲು += ಖಂಡ_ಗಾತ್ರ_3) {
        ಆದರೆ constexpr (do_multiply && do_ಕೂಡಿಸು) {
            dst[ಲಂಬಸಾಲು * dst_stride_ಲಂಬಸಾಲು] = scale * x[ಲಂಬಸಾಲು * src_stride_ಲಂಬಸಾಲು] * mul[ಲಂಬಸಾಲು] + ಕೂಡಿಸು_2[ಲಂಬಸಾಲು];
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (do_multiply) {
            dst[ಲಂಬಸಾಲು * dst_stride_ಲಂಬಸಾಲು] = scale * x[ಲಂಬಸಾಲು * src_stride_ಲಂಬಸಾಲು] * mul[ಲಂಬಸಾಲು];
        } ಇಲ್ಲದಿದ್ದರೆ {
            // folded epilogue of a fused GGML_OP_SCALE consumer (qwen35 GDN l2 norms);
            // the explicit temporary keeps the float evaluation order identical to
            // running rms_norm and scale as two separate kernels
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ v = scale * x[ಲಂಬಸಾಲು * src_stride_ಲಂಬಸಾಲು];
            dst[ಲಂಬಸಾಲು * dst_stride_ಲಂಬಸಾಲು] = v * scale_mul;
        }
    }
}

ಮಾದರಿ<ಪೂರ್ಣಾಂಕ warp_ಗಾತ್ರ>
ಸ್ಥಿರ ಶೂನ್ಯ l2_norm_f32(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * x, ದಶಮಾಂಶ * dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
    ಸ್ಥಿರಾಂಕ int64_t src_stride_ಲಂಬಸಾಲು, ಸ್ಥಿರಾಂಕ int64_t src_stride_ಅಡ್ಡಸಾಲು, ಸ್ಥಿರಾಂಕ int64_t src_stride_channel,
    ಸ್ಥಿರಾಂಕ int64_t src_stride_sample, ಸ್ಥಿರಾಂಕ int64_t dst_stride_ಲಂಬಸಾಲು, ಸ್ಥಿರಾಂಕ int64_t dst_stride_ಅಡ್ಡಸಾಲು,
    ಸ್ಥಿರಾಂಕ int64_t dst_stride_channel, ಸ್ಥಿರಾಂಕ int64_t dst_stride_sample, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ eps,
    ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3>& ಅಂಶ_ct1, ದಶಮಾಂಶ* s_ಮೊತ್ತ, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಗಾತ್ರ_3) {

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು     = ಅಂಶ_ct1.get_ಗುಂಪು(2);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ channel = ಅಂಶ_ct1.get_ಗುಂಪು(1);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ sample  = ಅಂಶ_ct1.get_ಗುಂಪು(0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ tid     = ಅಂಶ_ct1.get_local_id(2);

    x   += sample*src_stride_sample + channel*src_stride_channel + ಅಡ್ಡಸಾಲು*src_stride_ಅಡ್ಡಸಾಲು;
    dst += sample*dst_stride_sample + channel*dst_stride_channel + ಅಡ್ಡಸಾಲು*dst_stride_ಅಡ್ಡಸಾಲು;

    ದಶಮಾಂಶ tmp = 0.0f; // partial sum for thread in warp

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು = tid; ಲಂಬಸಾಲು < ncols; ಲಂಬಸಾಲು += ಖಂಡ_ಗಾತ್ರ_3) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ xi = x[ಲಂಬಸಾಲು * src_stride_ಲಂಬಸಾಲು];
        tmp += xi * xi;
    }

    tmp = ಖಂಡ_reduce<ಖಂಡ_reduce_ವಿಧಾನ::ಮೊತ್ತ_3, warp_ಗಾತ್ರ>(tmp, s_ಮೊತ್ತ, ಖಂಡ_ಗಾತ್ರ_3);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale = sycl::rsqrt(sycl::fmax(tmp, eps * eps));

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು = tid; ಲಂಬಸಾಲು < ncols; ಲಂಬಸಾಲು += ಖಂಡ_ಗಾತ್ರ_3) {
        dst[ಲಂಬಸಾಲು * dst_stride_ಲಂಬಸಾಲು] = scale * x[ಲಂಬಸಾಲು * src_stride_ಲಂಬಸಾಲು];
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ norm_f32_sycl(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * x, ದಶಮಾಂಶ * dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nchannels, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nsamples,
    ಸ್ಥಿರಾಂಕ int64_t src_stride_ಲಂಬಸಾಲು, ಸ್ಥಿರಾಂಕ int64_t src_stride_ಅಡ್ಡಸಾಲು, ಸ್ಥಿರಾಂಕ int64_t src_stride_channel, ಸ್ಥಿರಾಂಕ int64_t src_stride_sample,
    ಸ್ಥಿರಾಂಕ int64_t dst_stride_ಲಂಬಸಾಲು, ಸ್ಥಿರಾಂಕ int64_t dst_stride_ಅಡ್ಡಸಾಲು, ಸ್ಥಿರಾಂಕ int64_t dst_stride_channel, ಸ್ಥಿರಾಂಕ int64_t dst_stride_sample,
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ eps, ಸರತಿ_ptr stream, ಪೂರ್ಣಾಂಕ device) {

    ಸ್ಥಿರಾಂಕ sycl::range<3> global_dims(nsamples, nchannels, nrows);
    ಆದರೆ (ncols < 1024) {
        ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, 1, WARP_ಗಾತ್ರ);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * ಖಂಡ_dims, ಖಂಡ_dims),
                [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1)
                [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                    norm_f32(x, dst, ncols,
                        src_stride_ಲಂಬಸಾಲು, src_stride_ಅಡ್ಡಸಾಲು, src_stride_channel, src_stride_sample,
                        dst_stride_ಲಂಬಸಾಲು, dst_stride_ಅಡ್ಡಸಾಲು, dst_stride_channel, dst_stride_sample,
                        eps, ಅಂಶ_ct1, ಶೂನ್ಯಸೂಚಕ, WARP_ಗಾತ್ರ);
                });
            });
    }
    ಇಲ್ಲದಿದ್ದರೆ {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ work_ಗುಂಪು_ಗಾತ್ರ = ggml_sycl_info().max_work_ಗುಂಪು_sizes[device];
        ಪ್ರತಿಜ್ಞಾ(work_ಗುಂಪು_ಗಾತ್ರ % (WARP_ಗಾತ್ರ * WARP_ಗಾತ್ರ) == 0);
        ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, 1, work_ಗುಂಪು_ಗಾತ್ರ);
        /*
        DPCT1049:17: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<sycl::float2, 1> s_ಮೊತ್ತ_acc_ct1(
                            sycl::range<1>(work_ಗುಂಪು_ಗಾತ್ರ / WARP_ಗಾತ್ರ), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * ಖಂಡ_dims, ಖಂಡ_dims),
                [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1)
                [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                    norm_f32(x, dst, ncols,
                        src_stride_ಲಂಬಸಾಲು, src_stride_ಅಡ್ಡಸಾಲು, src_stride_channel, src_stride_sample,
                        dst_stride_ಲಂಬಸಾಲು, dst_stride_ಅಡ್ಡಸಾಲು, dst_stride_channel, dst_stride_sample,
                        eps, ಅಂಶ_ct1, get_ಸೂಚಕ(s_ಮೊತ್ತ_acc_ct1), work_ಗುಂಪು_ಗಾತ್ರ);
                });
            });
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ಗುಂಪು_norm_f32_sycl(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* x, ದಶಮಾಂಶ* dst,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_groups, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ eps, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಗುಂಪು_ಗಾತ್ರ,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne_elements, ಸರತಿ_ptr stream, ಪೂರ್ಣಾಂಕ device) {
    ಆದರೆ (ಗುಂಪು_ಗಾತ್ರ < 1024) {
        ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, 1, WARP_ಗಾತ್ರ);
        stream->submit([&](sycl::handler& cgh) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ eps_ct4 = eps;
            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, ಅಂಕೆ_groups) * ಖಂಡ_dims,
                    ಖಂಡ_dims),
                [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1)
                [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                    ಗುಂಪು_norm_f32(
                        x, dst, ಗುಂಪು_ಗಾತ್ರ, ne_elements, eps_ct4, ಅಂಶ_ct1,
                        ಶೂನ್ಯಸೂಚಕ, WARP_ಗಾತ್ರ);
                });
            });
    }
    ಇಲ್ಲದಿದ್ದರೆ {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ work_ಗುಂಪು_ಗಾತ್ರ = ggml_sycl_info().max_work_ಗುಂಪು_sizes[device];
        ಪ್ರತಿಜ್ಞಾ(work_ಗುಂಪು_ಗಾತ್ರ % (WARP_ಗಾತ್ರ * WARP_ಗಾತ್ರ) == 0);
        ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, 1, work_ಗುಂಪು_ಗಾತ್ರ);
        /*
        DPCT1049:18: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */

        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<ದಶಮಾಂಶ, 1> s_ಮೊತ್ತ_acc_ct1(sycl::range<1>(work_ಗುಂಪು_ಗಾತ್ರ / WARP_ಗಾತ್ರ),
                cgh);

            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ eps_ct4 = eps;

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, ಅಂಕೆ_groups) * ಖಂಡ_dims,
                    ಖಂಡ_dims),
                [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1)
                [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                    ಗುಂಪು_norm_f32(x, dst, ಗುಂಪು_ಗಾತ್ರ, ne_elements,
                        eps_ct4, ಅಂಶ_ct1,
                        get_ಸೂಚಕ(s_ಮೊತ್ತ_acc_ct1), work_ಗುಂಪು_ಗಾತ್ರ);
                });
            });
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ rms_norm_f32_sycl(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* x, ದಶಮಾಂಶ* dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nchannels, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nsamples,
    ಸ್ಥಿರಾಂಕ int64_t src_stride_ಲಂಬಸಾಲು, ಸ್ಥಿರಾಂಕ int64_t src_stride_ಅಡ್ಡಸಾಲು, ಸ್ಥಿರಾಂಕ int64_t src_stride_channel, ಸ್ಥಿರಾಂಕ int64_t src_stride_sample,
    ಸ್ಥಿರಾಂಕ int64_t dst_stride_ಲಂಬಸಾಲು, ಸ್ಥಿರಾಂಕ int64_t dst_stride_ಅಡ್ಡಸಾಲು, ಸ್ಥಿರಾಂಕ int64_t dst_stride_channel, ಸ್ಥಿರಾಂಕ int64_t dst_stride_sample,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ eps, ಸರತಿ_ptr stream, ಪೂರ್ಣಾಂಕ device) {
    // printf("%s ncols=%d, nrows=%d, WARP_SIZE=%d\n", __func__, ncols, nrows, WARP_SIZE);

    ಸ್ಥಿರಾಂಕ sycl::range<3> global_dims(nsamples, nchannels, nrows);
    ಆದರೆ (ncols < 1024) {
        ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, 1, WARP_ಗಾತ್ರ);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * ಖಂಡ_dims, ಖಂಡ_dims),
                [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1)
                [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                    rms_norm_f32(x, dst, ncols,
                        src_stride_ಲಂಬಸಾಲು, src_stride_ಅಡ್ಡಸಾಲು, src_stride_channel, src_stride_sample,
                        dst_stride_ಲಂಬಸಾಲು, dst_stride_ಅಡ್ಡಸಾಲು, dst_stride_channel, dst_stride_sample,
                        eps, ಅಂಶ_ct1, ಶೂನ್ಯಸೂಚಕ, WARP_ಗಾತ್ರ);
                });
            });
    }
    ಇಲ್ಲದಿದ್ದರೆ {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ work_ಗುಂಪು_ಗಾತ್ರ = ggml_sycl_info().max_work_ಗುಂಪು_sizes[device];
        ಪ್ರತಿಜ್ಞಾ(work_ಗುಂಪು_ಗಾತ್ರ % (WARP_ಗಾತ್ರ * WARP_ಗಾತ್ರ) == 0);
        ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, 1, work_ಗುಂಪು_ಗಾತ್ರ);
        /*
        DPCT1049:19: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<ದಶಮಾಂಶ, 1> s_ಮೊತ್ತ_acc_ct1(sycl::range<1>(work_ಗುಂಪು_ಗಾತ್ರ / WARP_ಗಾತ್ರ),
                cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * ಖಂಡ_dims, ಖಂಡ_dims),
                [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1)
                [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                    rms_norm_f32(x, dst, ncols,
                        src_stride_ಲಂಬಸಾಲು, src_stride_ಅಡ್ಡಸಾಲು, src_stride_channel, src_stride_sample,
                        dst_stride_ಲಂಬಸಾಲು, dst_stride_ಅಡ್ಡಸಾಲು, dst_stride_channel, dst_stride_sample,
                        eps, ಅಂಶ_ct1, get_ಸೂಚಕ(s_ಮೊತ್ತ_acc_ct1), work_ಗುಂಪು_ಗಾತ್ರ);
                });
            });
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ rms_norm_scale_f32_sycl(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* x, ದಶಮಾಂಶ* dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nchannels, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nsamples,
    ಸ್ಥಿರಾಂಕ int64_t src_stride_ಲಂಬಸಾಲು, ಸ್ಥಿರಾಂಕ int64_t src_stride_ಅಡ್ಡಸಾಲು, ಸ್ಥಿರಾಂಕ int64_t src_stride_channel, ಸ್ಥಿರಾಂಕ int64_t src_stride_sample,
    ಸ್ಥಿರಾಂಕ int64_t dst_stride_ಲಂಬಸಾಲು, ಸ್ಥಿರಾಂಕ int64_t dst_stride_ಅಡ್ಡಸಾಲು, ಸ್ಥಿರಾಂಕ int64_t dst_stride_channel, ಸ್ಥಿರಾಂಕ int64_t dst_stride_sample,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ eps, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale_mul, ಸರತಿ_ptr stream, ಪೂರ್ಣಾಂಕ device) {
    ಸ್ಥಿರಾಂಕ sycl::range<3> global_dims(nsamples, nchannels, nrows);
    ಆದರೆ (ncols < 1024) {
        ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, 1, WARP_ಗಾತ್ರ);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * ಖಂಡ_dims, ಖಂಡ_dims),
                [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1)
                [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                    rms_norm_f32(x, dst, ncols,
                        src_stride_ಲಂಬಸಾಲು, src_stride_ಅಡ್ಡಸಾಲು, src_stride_channel, src_stride_sample,
                        dst_stride_ಲಂಬಸಾಲು, dst_stride_ಅಡ್ಡಸಾಲು, dst_stride_channel, dst_stride_sample,
                        eps, ಅಂಶ_ct1, ಶೂನ್ಯಸೂಚಕ, WARP_ಗಾತ್ರ,
                        ಶೂನ್ಯಸೂಚಕ, 0, 0, 0, 0, 0, 0,
                        ಶೂನ್ಯಸೂಚಕ, 0, 0, 0, 0, 0, 0,
                        scale_mul);
                });
            });
    }
    ಇಲ್ಲದಿದ್ದರೆ {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ work_ಗುಂಪು_ಗಾತ್ರ = ggml_sycl_info().max_work_ಗುಂಪು_sizes[device];
        ಪ್ರತಿಜ್ಞಾ(work_ಗುಂಪು_ಗಾತ್ರ % (WARP_ಗಾತ್ರ * WARP_ಗಾತ್ರ) == 0);
        ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, 1, work_ಗುಂಪು_ಗಾತ್ರ);
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<ದಶಮಾಂಶ, 1> s_ಮೊತ್ತ_acc_ct1(sycl::range<1>(work_ಗುಂಪು_ಗಾತ್ರ / WARP_ಗಾತ್ರ),
                cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * ಖಂಡ_dims, ಖಂಡ_dims),
                [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1)
                [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                    rms_norm_f32(x, dst, ncols,
                        src_stride_ಲಂಬಸಾಲು, src_stride_ಅಡ್ಡಸಾಲು, src_stride_channel, src_stride_sample,
                        dst_stride_ಲಂಬಸಾಲು, dst_stride_ಅಡ್ಡಸಾಲು, dst_stride_channel, dst_stride_sample,
                        eps, ಅಂಶ_ct1, get_ಸೂಚಕ(s_ಮೊತ್ತ_acc_ct1), work_ಗುಂಪು_ಗಾತ್ರ,
                        ಶೂನ್ಯಸೂಚಕ, 0, 0, 0, 0, 0, 0,
                        ಶೂನ್ಯಸೂಚಕ, 0, 0, 0, 0, 0, 0,
                        scale_mul);
                });
            });
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ rms_norm_mul_f32_sycl(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* x, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* mul, ದಶಮಾಂಶ* dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nchannels, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nsamples,
        ಸ್ಥಿರಾಂಕ int64_t src_stride_ಲಂಬಸಾಲು, ಸ್ಥಿರಾಂಕ int64_t src_stride_ಅಡ್ಡಸಾಲು, ಸ್ಥಿರಾಂಕ int64_t src_stride_channel, ಸ್ಥಿರಾಂಕ int64_t src_stride_sample,
        ಸ್ಥಿರಾಂಕ int64_t dst_stride_ಲಂಬಸಾಲು, ಸ್ಥಿರಾಂಕ int64_t dst_stride_ಅಡ್ಡಸಾಲು, ಸ್ಥಿರಾಂಕ int64_t dst_stride_channel, ಸ್ಥಿರಾಂಕ int64_t dst_stride_sample,
        ಸ್ಥಿರಾಂಕ int64_t mul_stride_ಅಡ್ಡಸಾಲು, ಸ್ಥಿರಾಂಕ int64_t mul_stride_channel, ಸ್ಥಿರಾಂಕ int64_t mul_stride_sample,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ mul_nrows, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ mul_nchannels, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ mul_nsamples,
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ eps, ಸರತಿ_ptr stream, ಪೂರ್ಣಾಂಕ device) {
    ಸ್ಥಿರಾಂಕ sycl::range<3> global_dims(nsamples, nchannels, nrows);
    ಆದರೆ (ncols < 1024) {
        ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, 1, WARP_ಗಾತ್ರ);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * ಖಂಡ_dims, ಖಂಡ_dims),
                [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1)
                [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                    rms_norm_f32<ನಿಜ>(x, dst, ncols,
                        src_stride_ಲಂಬಸಾಲು, src_stride_ಅಡ್ಡಸಾಲು, src_stride_channel, src_stride_sample,
                        dst_stride_ಲಂಬಸಾಲು, dst_stride_ಅಡ್ಡಸಾಲು, dst_stride_channel, dst_stride_sample,
                        eps, ಅಂಶ_ct1, ಶೂನ್ಯಸೂಚಕ, WARP_ಗಾತ್ರ,
                        mul, mul_stride_ಅಡ್ಡಸಾಲು, mul_stride_channel, mul_stride_sample, mul_nrows, mul_nchannels, mul_nsamples);
                });
            });
    }
    ಇಲ್ಲದಿದ್ದರೆ {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ work_ಗುಂಪು_ಗಾತ್ರ = ggml_sycl_info().max_work_ಗುಂಪು_sizes[device];
        ಪ್ರತಿಜ್ಞಾ(work_ಗುಂಪು_ಗಾತ್ರ % (WARP_ಗಾತ್ರ * WARP_ಗಾತ್ರ) == 0);
        ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, 1, work_ಗುಂಪು_ಗಾತ್ರ);
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<ದಶಮಾಂಶ, 1> s_ಮೊತ್ತ_acc_ct1(sycl::range<1>(work_ಗುಂಪು_ಗಾತ್ರ / WARP_ಗಾತ್ರ), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * ಖಂಡ_dims, ಖಂಡ_dims),
                [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1)
                [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                    rms_norm_f32<ನಿಜ>(x, dst, ncols,
                        src_stride_ಲಂಬಸಾಲು, src_stride_ಅಡ್ಡಸಾಲು, src_stride_channel, src_stride_sample,
                        dst_stride_ಲಂಬಸಾಲು, dst_stride_ಅಡ್ಡಸಾಲು, dst_stride_channel, dst_stride_sample,
                        eps, ಅಂಶ_ct1, get_ಸೂಚಕ(s_ಮೊತ್ತ_acc_ct1), work_ಗುಂಪು_ಗಾತ್ರ,
                        mul, mul_stride_ಅಡ್ಡಸಾಲು, mul_stride_channel, mul_stride_sample, mul_nrows, mul_nchannels, mul_nsamples);
                });
            });
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ rms_norm_mul_ಕೂಡಿಸು_f32_sycl(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* x, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* mul, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* ಕೂಡಿಸು_2, ದಶಮಾಂಶ* dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nchannels, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nsamples,
        ಸ್ಥಿರಾಂಕ int64_t src_stride_ಲಂಬಸಾಲು, ಸ್ಥಿರಾಂಕ int64_t src_stride_ಅಡ್ಡಸಾಲು, ಸ್ಥಿರಾಂಕ int64_t src_stride_channel, ಸ್ಥಿರಾಂಕ int64_t src_stride_sample,
        ಸ್ಥಿರಾಂಕ int64_t dst_stride_ಲಂಬಸಾಲು, ಸ್ಥಿರಾಂಕ int64_t dst_stride_ಅಡ್ಡಸಾಲು, ಸ್ಥಿರಾಂಕ int64_t dst_stride_channel, ಸ್ಥಿರಾಂಕ int64_t dst_stride_sample,
        ಸ್ಥಿರಾಂಕ int64_t mul_stride_ಅಡ್ಡಸಾಲು, ಸ್ಥಿರಾಂಕ int64_t mul_stride_channel, ಸ್ಥಿರಾಂಕ int64_t mul_stride_sample,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ mul_nrows, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ mul_nchannels, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ mul_nsamples,
        ಸ್ಥಿರಾಂಕ int64_t ಕೂಡಿಸು_stride_ಅಡ್ಡಸಾಲು, ಸ್ಥಿರಾಂಕ int64_t ಕೂಡಿಸು_stride_channel, ಸ್ಥಿರಾಂಕ int64_t ಕೂಡಿಸು_stride_sample,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಕೂಡಿಸು_nrows, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಕೂಡಿಸು_nchannels, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಕೂಡಿಸು_nsamples,
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ eps, ಸರತಿ_ptr stream, ಪೂರ್ಣಾಂಕ device) {
    ಸ್ಥಿರಾಂಕ sycl::range<3> global_dims(nsamples, nchannels, nrows);
    ಆದರೆ (ncols < 1024) {
        ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, 1, WARP_ಗಾತ್ರ);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * ಖಂಡ_dims, ಖಂಡ_dims),
                [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1)
                [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                    rms_norm_f32<ನಿಜ, ನಿಜ>(x, dst, ncols,
                        src_stride_ಲಂಬಸಾಲು, src_stride_ಅಡ್ಡಸಾಲು, src_stride_channel, src_stride_sample,
                        dst_stride_ಲಂಬಸಾಲು, dst_stride_ಅಡ್ಡಸಾಲು, dst_stride_channel, dst_stride_sample,
                        eps, ಅಂಶ_ct1, ಶೂನ್ಯಸೂಚಕ, WARP_ಗಾತ್ರ,
                        mul, mul_stride_ಅಡ್ಡಸಾಲು, mul_stride_channel, mul_stride_sample, mul_nrows, mul_nchannels, mul_nsamples,
                        ಕೂಡಿಸು_2, ಕೂಡಿಸು_stride_ಅಡ್ಡಸಾಲು, ಕೂಡಿಸು_stride_channel, ಕೂಡಿಸು_stride_sample, ಕೂಡಿಸು_nrows, ಕೂಡಿಸು_nchannels, ಕೂಡಿಸು_nsamples);
                });
            });
    }
    ಇಲ್ಲದಿದ್ದರೆ {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ work_ಗುಂಪು_ಗಾತ್ರ = ggml_sycl_info().max_work_ಗುಂಪು_sizes[device];
        ಪ್ರತಿಜ್ಞಾ(work_ಗುಂಪು_ಗಾತ್ರ % (WARP_ಗಾತ್ರ * WARP_ಗಾತ್ರ) == 0);
        ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, 1, work_ಗುಂಪು_ಗಾತ್ರ);
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<ದಶಮಾಂಶ, 1> s_ಮೊತ್ತ_acc_ct1(sycl::range<1>(work_ಗುಂಪು_ಗಾತ್ರ / WARP_ಗಾತ್ರ), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * ಖಂಡ_dims, ಖಂಡ_dims),
                [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1)
                [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                    rms_norm_f32<ನಿಜ, ನಿಜ>(x, dst, ncols,
                        src_stride_ಲಂಬಸಾಲು, src_stride_ಅಡ್ಡಸಾಲು, src_stride_channel, src_stride_sample,
                        dst_stride_ಲಂಬಸಾಲು, dst_stride_ಅಡ್ಡಸಾಲು, dst_stride_channel, dst_stride_sample,
                        eps, ಅಂಶ_ct1, get_ಸೂಚಕ(s_ಮೊತ್ತ_acc_ct1), work_ಗುಂಪು_ಗಾತ್ರ,
                        mul, mul_stride_ಅಡ್ಡಸಾಲು, mul_stride_channel, mul_stride_sample, mul_nrows, mul_nchannels, mul_nsamples,
                        ಕೂಡಿಸು_2, ಕೂಡಿಸು_stride_ಅಡ್ಡಸಾಲು, ಕೂಡಿಸು_stride_channel, ಕೂಡಿಸು_stride_sample, ಕೂಡಿಸು_nrows, ಕೂಡಿಸು_nchannels, ಕೂಡಿಸು_nsamples);
                });
            });
    }
}

ಮಾದರಿ<ಪೂರ್ಣಾಂಕ warp_ಗಾತ್ರ>
ಸ್ಥಿರ ಶೂನ್ಯ l2_norm_f32_sycl(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *   x,
                             ದಶಮಾಂಶ *         dst,
                             ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       ncols,
                             ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       nrows,
                             ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       nchannels,
                             ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       nsamples,
                             ಸ್ಥಿರಾಂಕ int64_t   src_stride_ಲಂಬಸಾಲು,
                             ಸ್ಥಿರಾಂಕ int64_t   src_stride_ಅಡ್ಡಸಾಲು,
                             ಸ್ಥಿರಾಂಕ int64_t   src_stride_channel,
                             ಸ್ಥಿರಾಂಕ int64_t   src_stride_sample,
                             ಸ್ಥಿರಾಂಕ int64_t   dst_stride_ಲಂಬಸಾಲು,
                             ಸ್ಥಿರಾಂಕ int64_t   dst_stride_ಅಡ್ಡಸಾಲು,
                             ಸ್ಥಿರಾಂಕ int64_t   dst_stride_channel,
                             ಸ್ಥಿರಾಂಕ int64_t   dst_stride_sample,
                             ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ     eps,
                             ಸರತಿ_ptr       stream,
                             ಪೂರ್ಣಾಂಕ             device) {
    ಸ್ಥಿರಾಂಕ dpct::dim3 ಖಂಡಗಳು_ಅಂಕೆ(nrows, nchannels, nsamples);

    ಆದರೆ (ncols < 1024) {
        ಸ್ಥಿರಾಂಕ dpct::dim3 ಖಂಡ_dims(warp_ಗಾತ್ರ, 1, 1);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(ಖಂಡಗಳು_ಅಂಕೆ * ಖಂಡ_dims,
                    ಖಂಡ_dims),
                [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1)
                [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(warp_ಗಾತ್ರ)]] {
                    l2_norm_f32<warp_ಗಾತ್ರ>(x, dst, ncols,
                        src_stride_ಲಂಬಸಾಲು, src_stride_ಅಡ್ಡಸಾಲು, src_stride_channel, src_stride_sample,
                        dst_stride_ಲಂಬಸಾಲು, dst_stride_ಅಡ್ಡಸಾಲು, dst_stride_channel, dst_stride_sample,
                        eps, ಅಂಶ_ct1,
                        ಶೂನ್ಯಸೂಚಕ, warp_ಗಾತ್ರ);
                });
            });
    }
    ಇಲ್ಲದಿದ್ದರೆ {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ work_ಗುಂಪು_ಗಾತ್ರ = ggml_sycl_info().max_work_ಗುಂಪು_sizes[device];
        ಪ್ರತಿಜ್ಞಾ(work_ಗುಂಪು_ಗಾತ್ರ % (warp_ಗಾತ್ರ * warp_ಗಾತ್ರ) == 0);
        ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, 1, work_ಗುಂಪು_ಗಾತ್ರ);
        ಪೂರ್ಣಾಂಕ lsm_ಗಾತ್ರ =  ಖಂಡ_dims[2] > warp_ಗಾತ್ರ ? work_ಗುಂಪು_ಗಾತ್ರ / warp_ಗಾತ್ರ * ಗಾತ್ರ(ದಶಮಾಂಶ): 0;
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<ದಶಮಾಂಶ, 1> s_ಮೊತ್ತ_acc_ct1(sycl::range<1>(lsm_ಗಾತ್ರ),
                cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(ಖಂಡಗಳು_ಅಂಕೆ * ಖಂಡ_dims,
                    ಖಂಡ_dims),
                [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1)
                [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(warp_ಗಾತ್ರ)]] {
                    l2_norm_f32<warp_ಗಾತ್ರ>(x, dst, ncols,
                        src_stride_ಲಂಬಸಾಲು, src_stride_ಅಡ್ಡಸಾಲು, src_stride_channel, src_stride_sample,
                        dst_stride_ಲಂಬಸಾಲು, dst_stride_ಅಡ್ಡಸಾಲು, dst_stride_channel, dst_stride_sample,
                        eps, ಅಂಶ_ct1, get_ಸೂಚಕ(s_ಮೊತ್ತ_acc_ct1), work_ಗುಂಪು_ಗಾತ್ರ);
                });
            });
    }
}

// Batched L2 norm: N independent same-shape F32 tensors in one launch; the tensor
// index is folded into grid dim0 and each row's reduction is identical to the
// single-tensor kernel, so the result is bit-exact.
ರಚನೆ l2_batch_ptrs {
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src[GGML_SYCL_L2_BATCH_MAX];
    ದಶಮಾಂಶ *       dst[GGML_SYCL_L2_BATCH_MAX];
};

// One stride set shared by the whole batch: the caller only groups tensors whose nb[]
// all match, so per-tensor state stays two pointers.
ರಚನೆ l2_batch_strides {
    ಪೂರ್ಣಾಂಕ     ne1, ne2;
    int64_t ss0, ss1, ss2, ss3;
    int64_t ds0, ds1, ds2, ds3;
};

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ warp_ಗಾತ್ರ>
ಸ್ಥಿರ ಶೂನ್ಯ l2_norm_f32_batch(l2_batch_ptrs p, l2_batch_strides st, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ eps,
                              ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> & ಅಂಶ_ct1) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ t   = ಅಂಶ_ct1.get_ಗುಂಪು(0);  // tensor index
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ r   = ಅಂಶ_ct1.get_ಗುಂಪು(2);  // flattened row over ne1*ne2*ne3
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ tid = ಅಂಶ_ct1.get_local_id(2);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i1 = r % st.ne1;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i2 = (r / st.ne1) % st.ne2;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i3 = r / (st.ne1 * st.ne2);

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * x   = p.src[t] + i3 * st.ss3 + i2 * st.ss2 + i1 * st.ss1;
    ದಶಮಾಂಶ *       dst = p.dst[t] + i3 * st.ds3 + i2 * st.ds2 + i1 * st.ds1;

    ದಶಮಾಂಶ tmp = 0.0f;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು = tid; ಲಂಬಸಾಲು < ncols; ಲಂಬಸಾಲು += warp_ಗಾತ್ರ) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ xi = x[ಲಂಬಸಾಲು * st.ss0];
        tmp += xi * xi;
    }
    tmp = ಖಂಡ_reduce<ಖಂಡ_reduce_ವಿಧಾನ::ಮೊತ್ತ_3, warp_ಗಾತ್ರ>(tmp, (ದಶಮಾಂಶ *) ಶೂನ್ಯಸೂಚಕ, warp_ಗಾತ್ರ);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale = sycl::rsqrt(sycl::fmax(tmp, eps * eps));
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು = tid; ಲಂಬಸಾಲು < ncols; ಲಂಬಸಾಲು += warp_ಗಾತ್ರ) {
        dst[ಲಂಬಸಾಲು * st.ds0] = scale * x[ಲಂಬಸಾಲು * st.ss0];
    }
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ warp_ಗಾತ್ರ>
ಸ್ಥಿರ ಶೂನ್ಯ l2_norm_f32_batch_sycl(l2_batch_ptrs p, l2_batch_strides st, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_tensors,
                                   ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_ಒಟ್ಟು, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ eps,
                                   ಸರತಿ_ptr stream) {
    ಸ್ಥಿರಾಂಕ dpct::dim3 ಖಂಡಗಳು_ಅಂಕೆ(nrows_ಒಟ್ಟು, 1, n_tensors);
    ಸ್ಥಿರಾಂಕ dpct::dim3 ಖಂಡ_dims(warp_ಗಾತ್ರ, 1, 1);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(ಖಂಡಗಳು_ಅಂಕೆ * ಖಂಡ_dims, ಖಂಡ_dims),
            [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(warp_ಗಾತ್ರ)]] {
                l2_norm_f32_batch<warp_ಗಾತ್ರ>(p, st, ncols, eps, ಅಂಶ_ct1);
            });
    });
}

ಶೂನ್ಯ ggml_sycl_op_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS
    dpct::ಸರತಿ_ptr main_ಹರಿವು = ctx.stream();
    SYCL_CHECK(ggml_sycl_ಗಣ_device(ctx.device));
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src0_dd = static_cast<ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *>(dst->src[0]->ದತ್ತಾಂಶ);
    ದಶಮಾಂಶ *       dst_dd  = static_cast<ದಶಮಾಂಶ *>(dst->ದತ್ತಾಂಶ);

    ದಶಮಾಂಶ eps;
    ಸ್ಮರಣೆ_ನಕಲು(&eps, dst->op_params, ಗಾತ್ರ(ದಶಮಾಂಶ));
    GGML_ASSERT(eps >= 0.0f);
    ಸ್ಥಿರಾಂಕ size_t ts0 = ggml_type_ಗಾತ್ರ(src0->type);
    ಸ್ಥಿರಾಂಕ size_t tdst = ggml_type_ಗಾತ್ರ(dst->type);
    GGML_ASSERT(nb00 % ts0 == 0 && nb01 % ts0 == 0 && nb02 % ts0 == 0 && nb03 % ts0 == 0);
    GGML_ASSERT(nb0 % tdst == 0 && nb1 % tdst == 0 && nb2 % tdst == 0 && nb3 % tdst == 0);
    ಸ್ಥಿರಾಂಕ int64_t ss0 = nb00 / ts0;
    ಸ್ಥಿರಾಂಕ int64_t ss1 = nb01 / ts0;
    ಸ್ಥಿರಾಂಕ int64_t ss2 = nb02 / ts0;
    ಸ್ಥಿರಾಂಕ int64_t ss3 = nb03 / ts0;
    ಸ್ಥಿರಾಂಕ int64_t ds0 = nb0 / tdst;
    ಸ್ಥಿರಾಂಕ int64_t ds1 = nb1 / tdst;
    ಸ್ಥಿರಾಂಕ int64_t ds2 = nb2 / tdst;
    ಸ್ಥಿರಾಂಕ int64_t ds3 = nb3 / tdst;

    norm_f32_sycl(src0_dd, dst_dd, ne00, ne01, ne02, ne03,
        ss0, ss1, ss2, ss3, ds0, ds1, ds2, ds3, eps, main_ಹರಿವು, ctx.device);
}

ಶೂನ್ಯ ggml_sycl_op_ಗುಂಪು_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    ಪೂರ್ಣಾಂಕ ಅಂಕೆ_groups = dst->op_params[0];
    dpct::ಸರತಿ_ptr main_ಹರಿವು = ctx.stream();
    SYCL_CHECK(ggml_sycl_ಗಣ_device(ctx.device));

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src0_dd = static_cast<ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *>(dst->src[0]->ದತ್ತಾಂಶ);
    ದಶಮಾಂಶ *       dst_dd  = static_cast<ದಶಮಾಂಶ *>(dst->ದತ್ತಾಂಶ);

    ದಶಮಾಂಶ eps;
    ಸ್ಮರಣೆ_ನಕಲು(&eps, dst->op_params + 1, ಗಾತ್ರ(ದಶಮಾಂಶ));

    ಪೂರ್ಣಾಂಕ ಗುಂಪು_ಗಾತ್ರ = dst->src[0]->ne[0] * dst->src[0]->ne[1] * ((dst->src[0]->ne[2] + ಅಂಕೆ_groups - 1) / ಅಂಕೆ_groups);
    ಗುಂಪು_norm_f32_sycl(src0_dd, dst_dd, ಅಂಕೆ_groups, eps, ಗುಂಪು_ಗಾತ್ರ, dst->src[0]->ne[0] * dst->src[0]->ne[1] * dst->src[0]->ne[2], main_ಹರಿವು, ctx.device);
}

ಶೂನ್ಯ ggml_sycl_op_rms_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    dpct::ಸರತಿ_ptr main_ಹರಿವು = ctx.stream();
    SYCL_CHECK(ggml_sycl_ಗಣ_device(ctx.device));

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src0_dd = static_cast<ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *>(dst->src[0]->ದತ್ತಾಂಶ);
    ದಶಮಾಂಶ *       dst_dd  = static_cast<ದಶಮಾಂಶ *>(dst->ದತ್ತಾಂಶ);

    ದಶಮಾಂಶ eps;
    ಸ್ಮರಣೆ_ನಕಲು(&eps, dst->op_params, ಗಾತ್ರ(ದಶಮಾಂಶ));

    GGML_TENSOR_UNARY_OP_LOCALS
    ಸ್ಥಿರಾಂಕ size_t ts0 = ggml_type_ಗಾತ್ರ(src0->type);
    ಸ್ಥಿರಾಂಕ size_t tdst = ggml_type_ಗಾತ್ರ(dst->type);
    GGML_ASSERT(nb00 % ts0 == 0 && nb01 % ts0 == 0 && nb02 % ts0 == 0 && nb03 % ts0 == 0);
    GGML_ASSERT(nb0 % tdst == 0 && nb1 % tdst == 0 && nb2 % tdst == 0 && nb3 % tdst == 0);
    ಸ್ಥಿರಾಂಕ int64_t ss0 = nb00 / ts0;
    ಸ್ಥಿರಾಂಕ int64_t ss1 = nb01 / ts0;
    ಸ್ಥಿರಾಂಕ int64_t ss2 = nb02 / ts0;
    ಸ್ಥಿರಾಂಕ int64_t ss3 = nb03 / ts0;
    ಸ್ಥಿರಾಂಕ int64_t ds0 = nb0 / tdst;
    ಸ್ಥಿರಾಂಕ int64_t ds1 = nb1 / tdst;
    ಸ್ಥಿರಾಂಕ int64_t ds2 = nb2 / tdst;
    ಸ್ಥಿರಾಂಕ int64_t ds3 = nb3 / tdst;
    rms_norm_f32_sycl(src0_dd, dst_dd, ne00, ne01, ne02, ne03,
        ss0, ss1, ss2, ss3, ds0, ds1, ds2, ds3, eps, main_ಹರಿವು, ctx.device);
}

// Fused rms_norm + scale (the qwen35 GDN l2-norm pair build_gdn_l2_norm emits):
// the scale factor is a host scalar in the GGML_OP_SCALE node's op_params, so
// unlike the mul variants there is no second device tensor to wire up.
ಶೂನ್ಯ ggml_sycl_op_rms_norm_scale_fused(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                       ggml_tensor * scale_tensor) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(scale_tensor->type == GGML_TYPE_F32);

    dpct::ಸರತಿ_ptr main_ಹರಿವು = ctx.stream();
    SYCL_CHECK(ggml_sycl_ಗಣ_device(ctx.device));

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src0_dd = static_cast<ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *>(src0->ದತ್ತಾಂಶ);
    ದಶಮಾಂಶ *       dst_dd  = static_cast<ದಶಮಾಂಶ *>(scale_tensor->ದತ್ತಾಂಶ);

    ದಶಮಾಂಶ eps;
    ಸ್ಮರಣೆ_ನಕಲು(&eps, dst->op_params, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ದಶಮಾಂಶ scale_mul;
    ಸ್ಮರಣೆ_ನಕಲು(&scale_mul, scale_tensor->op_params, ಗಾತ್ರ(ದಶಮಾಂಶ));
    GGML_ASSERT(scale_mul >= 0.0f);

    GGML_TENSOR_UNARY_OP_LOCALS
    ಸ್ಥಿರಾಂಕ size_t ts0 = ggml_type_ಗಾತ್ರ(src0->type);
    ಸ್ಥಿರಾಂಕ size_t tdst = ggml_type_ಗಾತ್ರ(scale_tensor->type);
    GGML_ASSERT(nb00 % ts0 == 0 && nb01 % ts0 == 0 && nb02 % ts0 == 0 && nb03 % ts0 == 0);
    GGML_ASSERT(scale_tensor->nb[0] % tdst == 0 && scale_tensor->nb[1] % tdst == 0 &&
                scale_tensor->nb[2] % tdst == 0 && scale_tensor->nb[3] % tdst == 0);
    ಸ್ಥಿರಾಂಕ int64_t ss0 = nb00 / ts0;
    ಸ್ಥಿರಾಂಕ int64_t ss1 = nb01 / ts0;
    ಸ್ಥಿರಾಂಕ int64_t ss2 = nb02 / ts0;
    ಸ್ಥಿರಾಂಕ int64_t ss3 = nb03 / ts0;
    ಸ್ಥಿರಾಂಕ int64_t ds0 = scale_tensor->nb[0] / tdst;
    ಸ್ಥಿರಾಂಕ int64_t ds1 = scale_tensor->nb[1] / tdst;
    ಸ್ಥಿರಾಂಕ int64_t ds2 = scale_tensor->nb[2] / tdst;
    ಸ್ಥಿರಾಂಕ int64_t ds3 = scale_tensor->nb[3] / tdst;
    rms_norm_scale_f32_sycl(src0_dd, dst_dd, ne00, ne01, ne02, ne03,
        ss0, ss1, ss2, ss3, ds0, ds1, ds2, ds3, eps, scale_mul, main_ಹರಿವು, ctx.device);
}

ಶೂನ್ಯ ggml_sycl_op_rms_norm_fused(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * rms_norm_src = dst->src[0];
    ದಶಮಾಂಶ eps = 0.0f;
    ಸ್ಮರಣೆ_ನಕಲು(&eps, dst->op_params, ಗಾತ್ರ(ದಶಮಾಂಶ));

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *       src0_dd = static_cast<ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *>(rms_norm_src->ದತ್ತಾಂಶ);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *       mul_dd  = ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿರಾಂಕ ggml_tensor * mul_src = ಶೂನ್ಯಸೂಚಕ;
    ಆದರೆ (mul_tensor->src[0] == dst) {
        mul_dd  = static_cast<ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *>(mul_tensor->src[1]->ದತ್ತಾಂಶ);
        mul_src = mul_tensor->src[1];
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (mul_tensor->src[1] == dst) {
        mul_dd  = static_cast<ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *>(mul_tensor->src[0]->ದತ್ತಾಂಶ);
        mul_src = mul_tensor->src[0];
    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_ASSERT(ಸುಳ್ಳು);
    }
    ದಶಮಾಂಶ * dst_dd = static_cast<ದಶಮಾಂಶ *>(mul_tensor->ದತ್ತಾಂಶ);

    dpct::ಸರತಿ_ptr main_ಹರಿವು = ctx.stream();
    SYCL_CHECK(ggml_sycl_ಗಣ_device(ctx.device));

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    ಸ್ಥಿರಾಂಕ int64_t ne00 = rms_norm_src->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t ne01 = rms_norm_src->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t ne02 = rms_norm_src->ne[2];
    ಸ್ಥಿರಾಂಕ int64_t ne03 = rms_norm_src->ne[3];

    ಸ್ಥಿರಾಂಕ size_t ts0 = ggml_type_ಗಾತ್ರ(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    ಸ್ಥಿರಾಂಕ int64_t s00 = rms_norm_src->nb[0] / ts0;
    ಸ್ಥಿರಾಂಕ int64_t s01 = rms_norm_src->nb[1] / ts0;
    ಸ್ಥಿರಾಂಕ int64_t s02 = rms_norm_src->nb[2] / ts0;
    ಸ್ಥಿರಾಂಕ int64_t s03 = rms_norm_src->nb[3] / ts0;

    ಸ್ಥಿರಾಂಕ size_t tdst = ggml_type_ಗಾತ್ರ(mul_tensor->type);
    GGML_ASSERT(mul_tensor->nb[0] == tdst);
    ಸ್ಥಿರಾಂಕ int64_t d00 = mul_tensor->nb[0] / tdst;
    ಸ್ಥಿರಾಂಕ int64_t d01 = mul_tensor->nb[1] / tdst;
    ಸ್ಥಿರಾಂಕ int64_t d02 = mul_tensor->nb[2] / tdst;
    ಸ್ಥಿರಾಂಕ int64_t d03 = mul_tensor->nb[3] / tdst;

    ಸ್ಥಿರಾಂಕ size_t ts_mul = ggml_type_ಗಾತ್ರ(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    ಸ್ಥಿರಾಂಕ int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    ಸ್ಥಿರಾಂಕ int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    ಸ್ಥಿರಾಂಕ int64_t mul_s03 = mul_src->nb[3] / ts_mul;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ mul_nrows     = mul_src->ne[1];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ mul_nchannels = mul_src->ne[2];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ mul_nsamples  = mul_src->ne[3];

    rms_norm_mul_f32_sycl(src0_dd, mul_dd, dst_dd, ne00, ne01, ne02, ne03,
        s00, s01, s02, s03, d00, d01, d02, d03,
        mul_s01, mul_s02, mul_s03, mul_nrows, mul_nchannels, mul_nsamples, eps, main_ಹರಿವು, ctx.device);
}

ಶೂನ್ಯ ggml_sycl_op_rms_norm_fused_ಕೂಡಿಸು(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                     ggml_tensor * mul_tensor, ggml_tensor * ಕೂಡಿಸು_tensor) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * rms_norm_src = dst->src[0];
    ದಶಮಾಂಶ eps = 0.0f;
    ಸ್ಮರಣೆ_ನಕಲು(&eps, dst->op_params, ಗಾತ್ರ(ದಶಮಾಂಶ));

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *       src0_dd = static_cast<ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *>(rms_norm_src->ದತ್ತಾಂಶ);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *       mul_dd  = ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿರಾಂಕ ggml_tensor * mul_src = ಶೂನ್ಯಸೂಚಕ;
    ಆದರೆ (mul_tensor->src[0] == dst) {
        mul_dd  = static_cast<ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *>(mul_tensor->src[1]->ದತ್ತಾಂಶ);
        mul_src = mul_tensor->src[1];
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (mul_tensor->src[1] == dst) {
        mul_dd  = static_cast<ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *>(mul_tensor->src[0]->ದತ್ತಾಂಶ);
        mul_src = mul_tensor->src[0];
    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_ASSERT(ಸುಳ್ಳು);
    }

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *       ಕೂಡಿಸು_dd  = ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿರಾಂಕ ggml_tensor * ಕೂಡಿಸು_src = ಶೂನ್ಯಸೂಚಕ;
    ಆದರೆ (ಕೂಡಿಸು_tensor->src[0] == mul_tensor) {
        ಕೂಡಿಸು_dd  = static_cast<ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *>(ಕೂಡಿಸು_tensor->src[1]->ದತ್ತಾಂಶ);
        ಕೂಡಿಸು_src = ಕೂಡಿಸು_tensor->src[1];
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಕೂಡಿಸು_tensor->src[1] == mul_tensor) {
        ಕೂಡಿಸು_dd  = static_cast<ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *>(ಕೂಡಿಸು_tensor->src[0]->ದತ್ತಾಂಶ);
        ಕೂಡಿಸು_src = ಕೂಡಿಸು_tensor->src[0];
    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_ASSERT(ಸುಳ್ಳು);
    }

    ದಶಮಾಂಶ * dst_dd = static_cast<ದಶಮಾಂಶ *>(ಕೂಡಿಸು_tensor->ದತ್ತಾಂಶ);

    dpct::ಸರತಿ_ptr main_ಹರಿವು = ctx.stream();
    SYCL_CHECK(ggml_sycl_ಗಣ_device(ctx.device));

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(ಕೂಡಿಸು_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    ಸ್ಥಿರಾಂಕ int64_t ne00 = rms_norm_src->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t ne01 = rms_norm_src->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t ne02 = rms_norm_src->ne[2];
    ಸ್ಥಿರಾಂಕ int64_t ne03 = rms_norm_src->ne[3];

    ಸ್ಥಿರಾಂಕ size_t ts0 = ggml_type_ಗಾತ್ರ(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    ಸ್ಥಿರಾಂಕ int64_t s00 = rms_norm_src->nb[0] / ts0;
    ಸ್ಥಿರಾಂಕ int64_t s01 = rms_norm_src->nb[1] / ts0;
    ಸ್ಥಿರಾಂಕ int64_t s02 = rms_norm_src->nb[2] / ts0;
    ಸ್ಥಿರಾಂಕ int64_t s03 = rms_norm_src->nb[3] / ts0;

    ಸ್ಥಿರಾಂಕ size_t tdst = ggml_type_ಗಾತ್ರ(ಕೂಡಿಸು_tensor->type);
    GGML_ASSERT(ಕೂಡಿಸು_tensor->nb[0] == tdst);
    ಸ್ಥಿರಾಂಕ int64_t d00 = ಕೂಡಿಸು_tensor->nb[0] / tdst;
    ಸ್ಥಿರಾಂಕ int64_t d01 = ಕೂಡಿಸು_tensor->nb[1] / tdst;
    ಸ್ಥಿರಾಂಕ int64_t d02 = ಕೂಡಿಸು_tensor->nb[2] / tdst;
    ಸ್ಥಿರಾಂಕ int64_t d03 = ಕೂಡಿಸು_tensor->nb[3] / tdst;

    ಸ್ಥಿರಾಂಕ size_t ts_mul = ggml_type_ಗಾತ್ರ(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    ಸ್ಥಿರಾಂಕ int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    ಸ್ಥಿರಾಂಕ int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    ಸ್ಥಿರಾಂಕ int64_t mul_s03 = mul_src->nb[3] / ts_mul;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ mul_nrows     = mul_src->ne[1];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ mul_nchannels = mul_src->ne[2];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ mul_nsamples  = mul_src->ne[3];

    ಸ್ಥಿರಾಂಕ size_t ts_ಕೂಡಿಸು = ggml_type_ಗಾತ್ರ(ಕೂಡಿಸು_src->type);
    GGML_ASSERT(ಕೂಡಿಸು_src->nb[0] == ts_ಕೂಡಿಸು);
    ಸ್ಥಿರಾಂಕ int64_t ಕೂಡಿಸು_s01 = ಕೂಡಿಸು_src->nb[1] / ts_ಕೂಡಿಸು;
    ಸ್ಥಿರಾಂಕ int64_t ಕೂಡಿಸು_s02 = ಕೂಡಿಸು_src->nb[2] / ts_ಕೂಡಿಸು;
    ಸ್ಥಿರಾಂಕ int64_t ಕೂಡಿಸು_s03 = ಕೂಡಿಸು_src->nb[3] / ts_ಕೂಡಿಸು;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಕೂಡಿಸು_nrows     = ಕೂಡಿಸು_src->ne[1];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಕೂಡಿಸು_nchannels = ಕೂಡಿಸು_src->ne[2];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಕೂಡಿಸು_nsamples  = ಕೂಡಿಸು_src->ne[3];

    rms_norm_mul_ಕೂಡಿಸು_f32_sycl(src0_dd, mul_dd, ಕೂಡಿಸು_dd, dst_dd, ne00, ne01, ne02, ne03,
        s00, s01, s02, s03, d00, d01, d02, d03,
        mul_s01, mul_s02, mul_s03, mul_nrows, mul_nchannels, mul_nsamples,
        ಕೂಡಿಸು_s01, ಕೂಡಿಸು_s02, ಕೂಡಿಸು_s03, ಕೂಡಿಸು_nrows, ಕೂಡಿಸು_nchannels, ಕೂಡಿಸು_nsamples, eps, main_ಹರಿವು, ctx.device);
}

ಶೂನ್ಯ ggml_sycl_op_rms_norm_ಹಿಂಭಾಗ(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32); // dz
    GGML_ASSERT(dst->src[1]->type == GGML_TYPE_F32); // x
    GGML_ASSERT(dst->type         == GGML_TYPE_F32);

    ದಶಮಾಂಶ eps = 1e-5f;
    std::ಸ್ಮರಣೆ_ನಕಲು(&eps, dst->op_params, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಆದರೆ (!(eps > 0.0f) || !std::isfinite(eps)) eps = 1e-5f;

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * g_ತಳ  = static_cast<ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *>(dst->src[0]->ದತ್ತಾಂಶ); // dz
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * x_ತಳ  = static_cast<ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *>(dst->src[1]->ದತ್ತಾಂಶ); // x
          ದಶಮಾಂಶ * dx_ತಳ = static_cast<      ದಶಮಾಂಶ *>(dst->ದತ್ತಾಂಶ);

    ಸ್ಥಿರಾಂಕ int64_t D  = dst->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t n1 = dst->ne[1], n2 = dst->ne[2], n3 = dst->ne[3]; (ಶೂನ್ಯ) n3;
    ಸ್ಥಿರಾಂಕ int64_t N  = ggml_nrows(dst);
    ಆದರೆ (D == 0 || N == 0) ಹಿಂತಿರುಗಿ;

    ಸ್ಥಿರಾಂಕ ggml_tensor *G = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor *X = dst->src[1];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ts = (ಪೂರ್ಣಾಂಕ) ggml_type_ಗಾತ್ರ(X->type);
    GGML_ASSERT((size_t) X->nb[0]   == (size_t) ts);
    GGML_ASSERT((size_t) G->nb[0]   == (size_t) ts);
    GGML_ASSERT((size_t) dst->nb[0] == (size_t) ts);

    ಸ್ಥಿರಾಂಕ int64_t xs1 = X->nb[1] / ts, xs2 = X->nb[2] / ts, xs3 = X->nb[3] / ts;
    ಸ್ಥಿರಾಂಕ int64_t gs1 = G->nb[1] / ts, gs2 = G->nb[2] / ts, gs3 = G->nb[3] / ts;
    ಸ್ಥಿರಾಂಕ int64_t ds1 = dst->nb[1] / ts, ds2 = dst->nb[2] / ts, ds3 = dst->nb[3] / ts;

    dpct::ಸರತಿ_ptr q = ctx.stream();

    // work-group size: multiple of WARP_SIZE, capped by device and 256, and not larger than D
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ device_max_wg = ggml_sycl_info().max_work_ಗುಂಪು_sizes[ctx.device];
    ಸ್ವಯಂಚಾಲಿತ roundup = [](ಪೂರ್ಣಾಂಕ v, ಪೂರ್ಣಾಂಕ m) { ಹಿಂತಿರುಗಿ ((v + m - 1) / m) * m; };
    ಪೂರ್ಣಾಂಕ wg_cap = 256;
    ಆದರೆ (device_max_wg > 0) wg_cap = std::min(wg_cap, device_max_wg);
    ಪೂರ್ಣಾಂಕ WG = std::max(WARP_ಗಾತ್ರ, std::min(roundup((ಪೂರ್ಣಾಂಕ)std::min<int64_t>(D, wg_cap), WARP_ಗಾತ್ರ), wg_cap));

    // FP32 path: per-thread compensated accumulation + hierarchical reduction
    q->submit([&](sycl::handler &cgh) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nwarps_loc = std::max(1, WG / WARP_ಗಾತ್ರ);
        // store one partial value per warp (xx and xg) for cross-warp reduction
        ಸ್ವಯಂಚಾಲಿತ l_xx   = sycl::local_accessor<sycl::float2, 1>(sycl::range<1>(nwarps_loc), cgh);
        ಸ್ವಯಂಚಾಲಿತ l_xg   = sycl::local_accessor<sycl::float2, 1>(sycl::range<1>(nwarps_loc), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, N) * sycl::range<3>(1, 1, WG),
                              sycl::range<3>(1, 1, WG)),
            [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = ಅಂಶ_ct1.get_ಗುಂಪು(2);
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ tid = ಅಂಶ_ct1.get_local_id(2);

                ಸ್ಥಿರಾಂಕ int64_t i1 = ಅಡ್ಡಸಾಲು % n1;
                ಸ್ಥಿರಾಂಕ int64_t i2 = (ಅಡ್ಡಸಾಲು / n1) % n2;
                ಸ್ಥಿರಾಂಕ int64_t i3 = ಅಡ್ಡಸಾಲು / (n1 * n2);

                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *__restrict x_ಅಡ್ಡಸಾಲು = x_ತಳ + i3 * xs3 + i2 * xs2 + i1 * xs1;
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *__restrict g_ಅಡ್ಡಸಾಲು = g_ತಳ + i3 * gs3 + i2 * gs2 + i1 * gs1;
                ದಶಮಾಂಶ *__restrict d_ಅಡ್ಡಸಾಲು       = dx_ತಳ + i3 * ds3 + i2 * ds2 + i1 * ds1;

                // per-thread accumulation (compensated by default)
                ದಶಮಾಂಶ ಮೊತ್ತ_xx = 0.f, ಮೊತ್ತ_xg = 0.f;
#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ GGML_SYCL_RMS_ಹಿಂಭಾಗ_FAST
                ದಶಮಾಂಶ c_xx = 0.f, c_xg = 0.f;
#ಆದರೆ_ಅಂತ್ಯ
                ಗಾಗಿ (int64_t ಲಂಬಸಾಲು = tid; ಲಂಬಸಾಲು < D; ಲಂಬಸಾಲು += WG) {
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ xv = x_ಅಡ್ಡಸಾಲು[ಲಂಬಸಾಲು];
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ gv = g_ಅಡ್ಡಸಾಲು[ಲಂಬಸಾಲು];
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_RMS_ಹಿಂಭಾಗ_FAST
                    ಮೊತ್ತ_xx += xv * xv;
                    ಮೊತ್ತ_xg += xv * gv;
#ಇಲ್ಲದಿದ್ದರೆ
                    ದಶಮಾಂಶ y1 = xv * xv - c_xx;
                    ದಶಮಾಂಶ t1 = ಮೊತ್ತ_xx + y1;
                    c_xx = (t1 - ಮೊತ್ತ_xx) - y1;
                    ಮೊತ್ತ_xx = t1;

                    ದಶಮಾಂಶ y2 = xv * gv - c_xg;
                    ದಶಮಾಂಶ t2 = ಮೊತ್ತ_xg + y2;
                    c_xg = (t2 - ಮೊತ್ತ_xg) - y2;
                    ಮೊತ್ತ_xg = t2;
#ಆದರೆ_ಅಂತ್ಯ
                }

                // warp-level reduction
                sycl::float2 xx = sycl::float2(ಮೊತ್ತ_xx,
#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ GGML_SYCL_RMS_ಹಿಂಭಾಗ_FAST
                    c_xx
#ಇಲ್ಲದಿದ್ದರೆ
                    0.f
#ಆದರೆ_ಅಂತ್ಯ
                );
                sycl::float2 xg = sycl::float2(ಮೊತ್ತ_xg,
#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ GGML_SYCL_RMS_ಹಿಂಭಾಗ_FAST
                    c_xg
#ಇಲ್ಲದಿದ್ದರೆ
                    0.f
#ಆದರೆ_ಅಂತ್ಯ
                );
                xx = warp_reduce_ಮೊತ್ತ(xx, ಅಂಶ_ct1);
                xg = warp_reduce_ಮೊತ್ತ(xg, ಅಂಶ_ct1);

                // cross-warp reduction using local memory (single barrier)
                ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ sub_ಗುಂಪು = ಅಂಶ_ct1.get_sub_ಗುಂಪು();
                ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ sg_id     = sub_ಗುಂಪು.get_ಗುಂಪು_ರೇಖೀಯ_id();
                ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ wi_ಒಳ_sg  = sub_ಗುಂಪು.get_local_ರೇಖೀಯ_id();
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nthreads   = ಅಂಶ_ct1.get_local_range(2);
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nwarps     = nthreads / WARP_ಗಾತ್ರ;

                sycl::float2 xx_ಒಟ್ಟು = xx;
                sycl::float2 xg_ಒಟ್ಟು = xg;
                ಆದರೆ (nwarps > 1) {
                    ಆದರೆ (wi_ಒಳ_sg == 0) {
                        l_xx[sg_id] = xx;
                        l_xg[sg_id] = xg;
                    }
                    ಅಂಶ_ct1.barrier(sycl::access::fence_space::local_space);

                    ಆದರೆ (sg_id == 0) {
                        ಸ್ಥಿರಾಂಕ ಚಿಹ್ನೆರಹಿತ wi_u = wi_ಒಳ_sg;
                        sycl::float2 xx_ಮೊದಲ = (wi_u < static_cast<ಚಿಹ್ನೆರಹಿತ>(nwarps)) ? l_xx[wi_u] : sycl::float2(0.f, 0.f);
                        sycl::float2 xg_ಮೊದಲ = (wi_u < static_cast<ಚಿಹ್ನೆರಹಿತ>(nwarps)) ? l_xg[wi_u] : sycl::float2(0.f, 0.f);
                        xx_ಒಟ್ಟು = warp_reduce_ಮೊತ್ತ(xx_ಮೊದಲ, ಅಂಶ_ct1);
                        xg_ಒಟ್ಟು = warp_reduce_ಮೊತ್ತ(xg_ಮೊದಲ, ಅಂಶ_ct1);
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        // other subgroups keep their local totals; they'll be ignored
                        xx_ಒಟ್ಟು = xx;
                        xg_ಒಟ್ಟು = xg;
                    }
                    // ensure all threads see the first-subgroup result via broadcast below
                }

                // compute inv_r and coeff once per row and broadcast to the whole work-group
                ದಶಮಾಂಶ inv_r = 0.f;
                ದಶಮಾಂಶ coeff = 0.f;
                ಆದರೆ (tid == 0) {
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ಮೊತ್ತ_xx_f  = xx_ಒಟ್ಟು.x() + xx_ಒಟ್ಟು.y();
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ಮೊತ್ತ_xdz_f = xg_ಒಟ್ಟು.x() + xg_ಒಟ್ಟು.y();
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ಸರಾಸರಿ_eps  = ಮೊತ್ತ_xx_f / (ದಶಮಾಂಶ) D + eps;
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ಮೊತ್ತ_eps   = ಮೊತ್ತ_xx_f + eps * (ದಶಮಾಂಶ) D;
                    inv_r = sycl::rsqrt(ಸರಾಸರಿ_eps);
                    coeff = -ಮೊತ್ತ_xdz_f / ಮೊತ್ತ_eps;
                }
                inv_r = sycl::ಗುಂಪು_ಪ್ರಸಾರ(ಅಂಶ_ct1.get_ಗುಂಪು(), inv_r);
                coeff = sycl::ಗುಂಪು_ಪ್ರಸಾರ(ಅಂಶ_ct1.get_ಗುಂಪು(), coeff);

                ಗಾಗಿ (int64_t ಲಂಬಸಾಲು = tid; ಲಂಬಸಾಲು < D; ಲಂಬಸಾಲು += WG) {
                    d_ಅಡ್ಡಸಾಲು[ಲಂಬಸಾಲು] = (g_ಅಡ್ಡಸಾಲು[ಲಂಬಸಾಲು] + coeff * x_ಅಡ್ಡಸಾಲು[ಲಂಬಸಾಲು]) * inv_r;
                }
            });
    });

}

ಶೂನ್ಯ ggml_sycl_op_l2_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src0_d = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src0->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ * dst_d = (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ;
    dpct::ಸರತಿ_ptr     stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    ದಶಮಾಂಶ eps;
    ಸ್ಮರಣೆ_ನಕಲು(&eps, dst->op_params, ಗಾತ್ರ(ದಶಮಾಂಶ));
    GGML_ASSERT(eps >= 0.0f);

    ಸ್ಥಿರಾಂಕ size_t ts0 = ggml_type_ಗಾತ್ರ(src0->type);
    ಸ್ಥಿರಾಂಕ size_t tdst = ggml_type_ಗಾತ್ರ(dst->type);
    GGML_ASSERT(nb00 % ts0 == 0 && nb01 % ts0 == 0 && nb02 % ts0 == 0 && nb03 % ts0 == 0);
    GGML_ASSERT(nb0 % tdst == 0 && nb1 % tdst == 0 && nb2 % tdst == 0 && nb3 % tdst == 0);
    ಸ್ಥಿರಾಂಕ int64_t ss0 = nb00 / ts0;
    ಸ್ಥಿರಾಂಕ int64_t ss1 = nb01 / ts0;
    ಸ್ಥಿರಾಂಕ int64_t ss2 = nb02 / ts0;
    ಸ್ಥಿರಾಂಕ int64_t ss3 = nb03 / ts0;
    ಸ್ಥಿರಾಂಕ int64_t ds0 = nb0 / tdst;
    ಸ್ಥಿರಾಂಕ int64_t ds1 = nb1 / tdst;
    ಸ್ಥಿರಾಂಕ int64_t ds2 = nb2 / tdst;
    ಸ್ಥಿರಾಂಕ int64_t ds3 = nb3 / tdst;

    /*support both WARP_SIZE or WARP_32_SIZE in code
      choose by hardware for better performance
    */
    l2_norm_f32_sycl<WARP_ಗಾತ್ರ>(src0_d, dst_d, ne00, ne01, ne02, ne03,
            ss0, ss1, ss2, ss3, ds0, ds1, ds2, ds3, eps, stream, ctx.device);
}

// nodes[0..count) are independent, same-shape, same-eps, same-nb L2_NORM ops validated
// by the caller; requires ncols < 1024 (the warp reduction path).
ಶೂನ್ಯ ggml_sycl_l2_norm_batch(ggml_backend_sycl_context & ctx, ggml_tensor ** nodes, ಪೂರ್ಣಾಂಕ count) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * s0 = nodes[0]->src[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols       = (ಪೂರ್ಣಾಂಕ) s0->ne[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_ಒಟ್ಟು = (ಪೂರ್ಣಾಂಕ) ggml_nrows(s0);
    ದಶಮಾಂಶ eps;
    ಸ್ಮರಣೆ_ನಕಲು(&eps, nodes[0]->op_params, ಗಾತ್ರ(ದಶಮಾಂಶ));
    GGML_ASSERT(eps >= 0.0f);

    l2_batch_ptrs p{};
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ t = 0; t < count; ++t) {
        p.src[t] = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) nodes[t]->src[0]->ದತ್ತಾಂಶ;
        p.dst[t] = (ದಶಮಾಂಶ *) nodes[t]->ದತ್ತಾಂಶ;
    }

    ಸ್ಥಿರಾಂಕ ggml_tensor * d0 = nodes[0];
    ಸ್ಥಿರಾಂಕ size_t        ts = ggml_type_ಗಾತ್ರ(GGML_TYPE_F32);
    l2_batch_strides    st{};
    st.ne1 = (ಪೂರ್ಣಾಂಕ) s0->ne[1];
    st.ne2 = (ಪೂರ್ಣಾಂಕ) s0->ne[2];
    st.ss0 = s0->nb[0] / ts; st.ss1 = s0->nb[1] / ts; st.ss2 = s0->nb[2] / ts; st.ss3 = s0->nb[3] / ts;
    st.ds0 = d0->nb[0] / ts; st.ds1 = d0->nb[1] / ts; st.ds2 = d0->nb[2] / ts; st.ds3 = d0->nb[3] / ts;

    l2_norm_f32_batch_sycl<WARP_ಗಾತ್ರ>(p, st, count, ncols, nrows_ಒಟ್ಟು, eps, ctx.stream());
}
