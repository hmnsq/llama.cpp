#ಅನ್ತರ್ಭಾವ "softmax.hpp"
#ಅನ್ತರ್ಭಾವ <cstdint>
#ಅನ್ತರ್ಭಾವ <utility>
#ಅನ್ತರ್ಭಾವ <cmath>


ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T> ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ t2f32(T val) {
    ಹಿಂತಿರುಗಿ (ದಶಮಾಂಶ) val;
}

ಮಾದರಿ <> ದಶಮಾಂಶ __dpct_inline__ t2f32<sycl::half>(sycl::half val) {
  ಹಿಂತಿರುಗಿ sycl::vec<sycl::half, 1>(val)
      .convert<ದಶಮಾಂಶ, sycl::rounding_ಬಹುಲಕ::automatic>()[0];
}

ರಚನೆ soft_max_params {

    int64_t nheads;
    uint32_t n_ತಲೆ_log2;
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
    ದಶಮಾಂಶ scale;
    ದಶಮಾಂಶ max_bias;
    ದಶಮಾಂಶ m0;
    ದಶಮಾಂಶ m1;
};

// When ncols_template == 0 the bounds for the loops in this function are not known and can't be unrolled.
// As we want to keep pragma unroll for all other cases we suppress the clang transformation warning here.
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ __clang__
#ಸೂಚನೆ clang diagnostic ತಳ್ಳು
#ಸೂಚನೆ clang diagnostic ignored "-Wpass-failed"
#ಆದರೆ_ಅಂತ್ಯ // __clang__
ಮಾದರಿ <ತಾರ್ಕಿಕ use_shared, ಪೂರ್ಣಾಂಕ ncols_template, ಪೂರ್ಣಾಂಕ ಖಂಡ_ಗಾತ್ರ_template, ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ ಶೂನ್ಯ soft_max_f32(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *         x,
                         ಸ್ಥಿರಾಂಕ T *             ಮುಖವಾಡ,
                         ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *         sinks,
                         ದಶಮಾಂಶ *               dst,
                         ಸ್ಥಿರಾಂಕ soft_max_params p,
                         uint8_t *             dpct_local) {
    ಸ್ವಯಂಚಾಲಿತ      ಅಂಶ_ct1 = sycl::ext::oneapi::this_work_ಅಂಶ::get_nd_ಅಂಶ<3>();
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols    = ncols_template == 0 ? p.ncols : ncols_template;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಗಾತ್ರ_3 = ಖಂಡ_ಗಾತ್ರ_template == 0
                               ? ಅಂಶ_ct1.get_local_range(2)
                               : ಖಂಡ_ಗಾತ್ರ_template;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nthreads = ಖಂಡ_ಗಾತ್ರ_3;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nwarps = nthreads / WARP_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ size_t nreduce = nwarps / WARP_ಗಾತ್ರ;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ tid = ಅಂಶ_ct1.get_local_id(2);

    ಸ್ಥಿರಾಂಕ int64_t i03 = ಅಂಶ_ct1.get_ಗುಂಪು(0);
    ಸ್ಥಿರಾಂಕ int64_t i02 = ಅಂಶ_ct1.get_ಗುಂಪು(1);
    ಸ್ಥಿರಾಂಕ int64_t i01 = ಅಂಶ_ct1.get_ಗುಂಪು(2);

    //TODO: noncontigous inputs/outputs
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ rowx = ಅಂಶ_ct1.get_ಗುಂಪು(2) +
                     ಅಂಶ_ct1.get_ಗುಂಪು(1) * ಅಂಶ_ct1.get_ಗುಂಪು_range(2) +
                     ಅಂಶ_ct1.get_ಗುಂಪು(0) * ಅಂಶ_ct1.get_ಗುಂಪು_range(2) *
                         ಅಂಶ_ct1.get_ಗುಂಪು_range(1);

    ಸ್ಥಿರಾಂಕ int64_t i11 = i01;
    ಸ್ಥಿರಾಂಕ int64_t i12 = i02 % p.ne12;
    ಸ್ಥಿರಾಂಕ int64_t i13 = i03 % p.ne13;

    x    += int64_t(rowx)*ncols;
    ಮುಖವಾಡ += (i11*p.nb11 + i12*p.nb12 + i13*p.nb13) / ಗಾತ್ರ(T) * (ಮುಖವಾಡ != ಶೂನ್ಯಸೂಚಕ);
    dst  += int64_t(rowx)*ncols;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ warp_id = ಅಂಶ_ct1.get_local_id(2) / WARP_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ lane_id = ಅಂಶ_ct1.get_local_id(2) % WARP_ಗಾತ್ರ;

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ಇಳಿಜಾರು = get_alibi_ಇಳಿಜಾರು(p.max_bias, i02, p.n_ತಲೆ_log2, p.m0, p.m1);

    ದಶಮಾಂಶ * buf_iw = (ದಶಮಾಂಶ *) dpct_local;

    // shared memory buffer to cache values between iterations:
    ದಶಮಾಂಶ *vals = use_shared ? buf_iw + sycl::max(nwarps, WARP_ಗಾತ್ರ) : dst;
    ದಶಮಾಂಶ max_val = sinks ? sinks[i02] : -INFINITY;
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ col0 = 0; col0 < ncols; col0 += ಖಂಡ_ಗಾತ್ರ_3) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು = col0 + tid;

        ಆದರೆ (ncols_template == 0 && ಲಂಬಸಾಲು >= ncols) {
            ನಿಲ್ಲಿಸು;
        }

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ val = x[ಲಂಬಸಾಲು]*p.scale + (ಮುಖವಾಡ ? ಇಳಿಜಾರು*t2f32(ಮುಖವಾಡ[ಲಂಬಸಾಲು]) : 0.0f);

        vals[ಲಂಬಸಾಲು] = val;
        max_val   = sycl::max(max_val, val);
    }
    // find the max value in the block
    max_val = warp_reduce_max<WARP_ಗಾತ್ರ>(max_val);

    ಆದರೆ (ಖಂಡ_ಗಾತ್ರ_3 > WARP_ಗಾತ್ರ) {
        ಆದರೆ (lane_id == 0) {
            buf_iw[warp_id] = max_val;
        }
        ಅಂಶ_ct1.barrier();

        max_val = -INFINITY;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = lane_id; i < nwarps; i += WARP_ಗಾತ್ರ) {
            max_val = sycl::max(max_val, buf_iw[i]);
        }
        max_val = warp_reduce_max<WARP_ಗಾತ್ರ>(max_val);
    }
    ದಶಮಾಂಶ tmp = 0.0f; // partial sum

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ col0 = 0; col0 < ncols; col0 += ಖಂಡ_ಗಾತ್ರ_3) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು = col0 + tid;

        ಆದರೆ (ncols_template == 0 && ಲಂಬಸಾಲು >= ncols) {
            ನಿಲ್ಲಿಸು;
        }

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ val = sycl::native::ಘಾತಾಂಕ(sycl::max(vals[ಲಂಬಸಾಲು] - max_val, -80.0f));
        tmp += val;
        vals[ಲಂಬಸಾಲು] = val;
    }
    // find the sum of exps in the block
    tmp = warp_reduce_ಮೊತ್ತ<WARP_ಗಾತ್ರ>(tmp);
    ಆದರೆ (ಖಂಡ_ಗಾತ್ರ_3 > WARP_ಗಾತ್ರ) {
        ಅಂಶ_ct1.barrier();
        ಆದರೆ (warp_id == 0) {
            buf_iw[lane_id] = 0.0f;
            ಗಾಗಿ (size_t i = 1; i < nreduce; i += 1) {
                buf_iw[lane_id + i * WARP_ಗಾತ್ರ] = 0.f;
            }
        }
        ಅಂಶ_ct1.barrier();

        ಆದರೆ (lane_id == 0) {
            buf_iw[warp_id] = tmp;
        }
        ಅಂಶ_ct1.barrier();

        tmp = buf_iw[lane_id];
        ಗಾಗಿ (size_t i = 1; i < nreduce; i += 1) {
            tmp += buf_iw[lane_id + i * WARP_ಗಾತ್ರ];
        }
        tmp = warp_reduce_ಮೊತ್ತ<WARP_ಗಾತ್ರ>(tmp);
    }
    ಆದರೆ (sinks) {
        tmp += sycl::native::ಘಾತಾಂಕ(sycl::max(sinks[i02] - max_val, -80.0f));
    }
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ inv_ಮೊತ್ತ = 1.0f / tmp;

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ col0 = 0; col0 < ncols; col0 += ಖಂಡ_ಗಾತ್ರ_3) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು = col0 + tid;

        ಆದರೆ (ncols_template == 0 && ಲಂಬಸಾಲು >= ncols) {
            ಹಿಂತಿರುಗಿ;
        }

        dst[ಲಂಬಸಾಲು] = vals[ಲಂಬಸಾಲು] * inv_ಮೊತ್ತ;
    }
}
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ __clang__
#ಸೂಚನೆ clang diagnostic ತೆಗೆ
#ಆದರೆ_ಅಂತ್ಯ // __clang__

ಸ್ಥಿರ ಶೂನ್ಯ soft_max_ಹಿಂಭಾಗ_f32(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *grad, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *dstf, ದಶಮಾಂಶ *dst,
                              ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale) {
    ಸ್ವಯಂಚಾಲಿತ      ಅಂಶ_ct1 = sycl::ext::oneapi::this_work_ಅಂಶ::get_nd_ಅಂಶ<3>();
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ tid      = ಅಂಶ_ct1.get_local_id(2);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ rowx     = ಅಂಶ_ct1.get_ಗುಂಪು(2);

    grad += int64_t(rowx)*ncols;
    dstf += int64_t(rowx)*ncols;
    dst  += int64_t(rowx)*ncols;

    ದಶಮಾಂಶ dgf_dot = 0.0f; // dot product of dst from forward pass and gradients

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು = tid; ಲಂಬಸಾಲು < ncols; ಲಂಬಸಾಲು += WARP_ಗಾತ್ರ) {
        dgf_dot += dstf[ಲಂಬಸಾಲು]*grad[ಲಂಬಸಾಲು];
    }

    dgf_dot = warp_reduce_ಮೊತ್ತ<WARP_ಗಾತ್ರ>(dgf_dot);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು = tid; ಲಂಬಸಾಲು < ncols; ಲಂಬಸಾಲು += WARP_ಗಾತ್ರ) {
        dst[ಲಂಬಸಾಲು] = scale * (grad[ಲಂಬಸಾಲು] - dgf_dot) * dstf[ಲಂಬಸಾಲು];
    }
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ... Ns, ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ ಶೂನ್ಯ launch_soft_max_kernels(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *           x,
                                    ಸ್ಥಿರಾಂಕ T *               ಮುಖವಾಡ,
                                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *           sinks,
                                    ದಶಮಾಂಶ *                 dst,
                                    ಸ್ಥಿರಾಂಕ soft_max_params & p,
                                    dpct::ಸರತಿ_ptr         stream,
                                    dpct::dim3              ಖಂಡ_dims,
                                    dpct::dim3              ಖಂಡ_nums,
                                    size_t                  nbytes_shared)
{
    ಸ್ವಯಂಚಾಲಿತ launch_kernel = [=](ಸ್ವಯಂಚಾಲಿತ I) -> ತಾರ್ಕಿಕ {
        constexpr ಪೂರ್ಣಾಂಕ ncols = decltype(I)::value;
        constexpr ಪೂರ್ಣಾಂಕ ಖಂಡ = (ncols > 1024 ? 1024 : ncols);
        ಆದರೆ (p.ncols == ncols) {
            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                    sycl::range<1>(nbytes_shared), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
                    [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(
                        WARP_ಗಾತ್ರ)]] {
                        soft_max_f32<ನಿಜ, ncols, ಖಂಡ>(
                            x, ಮುಖವಾಡ, sinks, dst, p,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                        GGML_ಬಳಸದ(ಅಂಶ_ct1);
                    });
            });
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    };

    // unary fold over launch_kernel
    ಆದರೆ ((launch_kernel(std::integral_constant<ಪೂರ್ಣಾಂಕ, Ns>{}) || ...)) {
        ಹಿಂತಿರುಗಿ;
    }

    stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range<1>(nbytes_shared), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
            [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1)
                [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                    soft_max_f32<ನಿಜ, 0, 0>(
                        x, ಮುಖವಾಡ, sinks, dst, p,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                    GGML_ಬಳಸದ(ಅಂಶ_ct1);
                });
    });
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ ಶೂನ್ಯ soft_max_f32_sycl(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *x, ಸ್ಥಿರಾಂಕ T *ಮುಖವಾಡ,
                              ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *sinks, ದಶಮಾಂಶ *dst,
                              ಸ್ಥಿರಾಂಕ soft_max_params &params,
                              dpct::ಸರತಿ_ptr stream, ಪೂರ್ಣಾಂಕ device) {
    ಪೂರ್ಣಾಂಕ nth = WARP_ಗಾತ್ರ;
    ಪೂರ್ಣಾಂಕ max_ಖಂಡ_ಗಾತ್ರ = ggml_sycl_info().max_work_ಗುಂಪು_sizes[device];
    ಸ್ಥಿರಾಂಕ int64_t ncols_x = params.ncols;

    ಆಗ (nth < ncols_x && nth < max_ಖಂಡ_ಗಾತ್ರ) nth *= 2;
    ಆದರೆ (nth>max_ಖಂಡ_ಗಾತ್ರ) nth = max_ಖಂಡ_ಗಾತ್ರ;

    ಸ್ಥಿರಾಂಕ dpct::dim3 ಖಂಡ_dims(nth, 1, 1);
    ಸ್ಥಿರಾಂಕ dpct::dim3 ಖಂಡ_nums(params.ne01, params.ne02, params.ne03);
    ಸ್ಥಿರಾಂಕ size_t nbytes_shared =
        (GGML_PAD(ncols_x, WARP_ಗಾತ್ರ) + WARP_ಗಾತ್ರ) * ಗಾತ್ರ(ದಶಮಾಂಶ);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ id       = get_current_device_id();
    ಸ್ಥಿರಾಂಕ size_t smpbo = ggml_sycl_info().devices[id].smpbo;

    ಆದರೆ (nbytes_shared <= smpbo && ncols_x <= max_ಖಂಡ_ಗಾತ್ರ) {
        launch_soft_max_kernels<32, 64, 128, 256, 512, 1024, 2048, 4096>(
            x, ಮುಖವಾಡ, sinks, dst, params, stream, ಖಂಡ_dims, ಖಂಡ_nums,
            nbytes_shared);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಸ್ಥಿರಾಂಕ size_t nbytes_shared_ಕೆಳಗಿನ = WARP_ಗಾತ್ರ * ಗಾತ್ರ(ದಶಮಾಂಶ);

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(nbytes_shared_ಕೆಳಗಿನ), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
                [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1)
                    [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                    soft_max_f32<ಸುಳ್ಳು, 0, 0>(
                        x, ಮುಖವಾಡ, sinks, dst, params,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                    GGML_ಬಳಸದ(ಅಂಶ_ct1);
                });
        });
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ soft_max_ಹಿಂಭಾಗ_f32_sycl(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *   grad,
                                   ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *   dstf,
                                   ದಶಮಾಂಶ *         dst,
                                   ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       ncols,
                                   ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       nrows,
                                   ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ     scale,
                                   dpct::ಸರತಿ_ptr stream) {
    ಸ್ಥಿರಾಂಕ dpct::dim3 ಖಂಡ_dims(WARP_ಗಾತ್ರ, 1, 1);
    ಸ್ಥಿರಾಂಕ dpct::dim3 ಖಂಡ_nums(nrows, 1, 1);

    stream->parallel_for(sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
                         [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
                             soft_max_ಹಿಂಭಾಗ_f32(grad, dstf, dst, ncols, scale);
                             GGML_ಬಳಸದ(ಅಂಶ_ct1);
                         });
}

ಶೂನ್ಯ ggml_sycl_op_soft_max(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src2 = dst->src[2];

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src0_d = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src0->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ  * src1_d = src1 ? (ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *) src1->ದತ್ತಾಂಶ : ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ  * src2_d = src2 ? (ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *) src2->ದತ್ತಾಂಶ : ಶೂನ್ಯಸೂಚಕ;
    ದಶಮಾಂಶ       *  dst_d = (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ;

    dpct::ಸರತಿ_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    // src1 contains mask and it is optional
    GGML_ASSERT(!src1 || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32);

    ಸ್ಥಿರಾಂಕ int64_t nrows_x = ggml_nrows(src0);
    ಸ್ಥಿರಾಂಕ int64_t nrows_y = src0->ne[1];

    ಸ್ಥಿರಾಂಕ int64_t ne00 = src0->ne[0];

    ದಶಮಾಂಶ scale    = 1.0f;
    ದಶಮಾಂಶ max_bias = 0.0f;

    ಸ್ಮರಣೆ_ನಕಲು(&scale,    (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) dst->op_params + 0, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&max_bias, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) dst->op_params + 1, ಗಾತ್ರ(ದಶಮಾಂಶ));

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ use_f16 = (src1 && src1->type == GGML_TYPE_F16);

    ಸ್ಥಿರಾಂಕ int64_t nb11 = src1 ? src1->nb[1] : 1;
    ಸ್ಥಿರಾಂಕ int64_t nb12 = src1 ? src1->nb[2] : 1;
    ಸ್ಥಿರಾಂಕ int64_t nb13 = src1 ? src1->nb[3] : 1;

    ಸ್ಥಿರಾಂಕ int64_t ne12 = src1 ? src1->ne[2] : 1;
    ಸ್ಥಿರಾಂಕ int64_t ne13 = src1 ? src1->ne[3] : 1;

    ಸ್ಥಿರಾಂಕ uint32_t n_ತಲೆ      = src0->ne[2];
    ಸ್ಥಿರಾಂಕ uint32_t n_ತಲೆ_log2 = 1u << (uint32_t) floorf(log2f((ದಶಮಾಂಶ) n_ತಲೆ));

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ m0 = powf(2.0f, -(max_bias       ) / n_ತಲೆ_log2);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ m1 = powf(2.0f, -(max_bias / 2.0f) / n_ತಲೆ_log2);


    soft_max_params params = {};
    params.nheads = src0->ne[2];
    params.n_ತಲೆ_log2 = n_ತಲೆ_log2;
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

    ಆದರೆ (use_f16) {
        soft_max_f32_sycl(src0_d, (ಸ್ಥಿರಾಂಕ sycl::half *)src1_d,
                          (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)src2_d, dst_d, params, stream,
                          ctx.device);
    } ಇಲ್ಲದಿದ್ದರೆ {
        soft_max_f32_sycl(src0_d, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)src1_d, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)src2_d,
                          dst_d, params, stream, ctx.device);
    }
}

ಶೂನ್ಯ ggml_sycl_op_soft_max_ಹಿಂಭಾಗ(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0]; // grad
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1]; // forward pass output

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src0_d = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src0->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src1_d = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src1->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ       * dst_d  = (ದಶಮಾಂಶ       *) dst->ದತ್ತಾಂಶ;

    dpct::ಸರತಿ_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    ಸ್ಥಿರಾಂಕ int64_t ncols = src0->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t nrows = ggml_nrows(src0);

    ದಶಮಾಂಶ scale    = 1.0f;
    ದಶಮಾಂಶ max_bias = 0.0f;

    ಸ್ಮರಣೆ_ನಕಲು(&scale,    (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) dst->op_params + 0, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&max_bias, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) dst->op_params + 1, ಗಾತ್ರ(ದಶಮಾಂಶ));

    GGML_ASSERT(max_bias == 0.0f);

    soft_max_ಹಿಂಭಾಗ_f32_sycl(src0_d, src1_d, dst_d, ncols, nrows, scale, stream);
}
