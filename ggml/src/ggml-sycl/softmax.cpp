#ਅਨ੍ਤਰ੍ਭਾਵ "softmax.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdint>
#ਅਨ੍ਤਰ੍ਭਾਵ <utility>
#ਅਨ੍ਤਰ੍ਭਾਵ <cmath>


ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T> ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ t2f32(T val) {
    ਵਾਪਸ (ਦਸ਼ਮਲਵ) val;
}

ਸਾਂਚਾ <> ਦਸ਼ਮਲਵ __dpct_inline__ t2f32<sycl::half>(sycl::half val) {
  ਵਾਪਸ sycl::vec<sycl::half, 1>(val)
      .convert<ਦਸ਼ਮਲਵ, sycl::rounding_ਬਹੁਲਕ::automatic>()[0];
}

ਬਣਤਰ soft_max_params {

    int64_t nheads;
    uint32_t n_ਸਿਰ_log2;
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
    ਦਸ਼ਮਲਵ scale;
    ਦਸ਼ਮਲਵ max_bias;
    ਦਸ਼ਮਲਵ m0;
    ਦਸ਼ਮਲਵ m1;
};

// When ncols_template == 0 the bounds for the loops in this function are not known and can't be unrolled.
// As we want to keep pragma unroll for all other cases we suppress the clang transformation warning here.
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ __clang__
#ਹਦਾਇਤ clang diagnostic ਧੱਕੋ
#ਹਦਾਇਤ clang diagnostic ignored "-Wpass-failed"
#ਜੇ_ਅੰਤ // __clang__
ਸਾਂਚਾ <ਤਾਰਕਿਕ use_shared, ਪੂਰਨ ncols_template, ਪੂਰਨ ਖੰਡ_ਮਾਪ_template, ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ ਖਾਲੀ soft_max_f32(ਅਚਲ ਦਸ਼ਮਲਵ *         x,
                         ਅਚਲ T *             ਨਕਾਬ,
                         ਅਚਲ ਦਸ਼ਮਲਵ *         sinks,
                         ਦਸ਼ਮਲਵ *               dst,
                         ਅਚਲ soft_max_params p,
                         uint8_t *             dpct_local) {
    ਸਵੈਚਾਲਿਤ      ਵਸਤੂ_ct1 = sycl::ext::oneapi::this_work_ਵਸਤੂ::get_nd_ਵਸਤੂ<3>();
    ਅਚਲ ਪੂਰਨ ncols    = ncols_template == 0 ? p.ncols : ncols_template;
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਮਾਪ_3 = ਖੰਡ_ਮਾਪ_template == 0
                               ? ਵਸਤੂ_ct1.get_local_range(2)
                               : ਖੰਡ_ਮਾਪ_template;
    ਅਚਲ ਪੂਰਨ nthreads = ਖੰਡ_ਮਾਪ_3;
    ਅਚਲ ਪੂਰਨ nwarps = nthreads / WARP_ਮਾਪ;
    ਅਚਲ size_t nreduce = nwarps / WARP_ਮਾਪ;

    ਅਚਲ ਪੂਰਨ tid = ਵਸਤੂ_ct1.get_local_id(2);

    ਅਚਲ int64_t i03 = ਵਸਤੂ_ct1.get_ਸਮੂਹ(0);
    ਅਚਲ int64_t i02 = ਵਸਤੂ_ct1.get_ਸਮੂਹ(1);
    ਅਚਲ int64_t i01 = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2);

    //TODO: noncontigous inputs/outputs
    ਅਚਲ ਪੂਰਨ rowx = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) +
                     ਵਸਤੂ_ct1.get_ਸਮੂਹ(1) * ਵਸਤੂ_ct1.get_ਸਮੂਹ_range(2) +
                     ਵਸਤੂ_ct1.get_ਸਮੂਹ(0) * ਵਸਤੂ_ct1.get_ਸਮੂਹ_range(2) *
                         ਵਸਤੂ_ct1.get_ਸਮੂਹ_range(1);

    ਅਚਲ int64_t i11 = i01;
    ਅਚਲ int64_t i12 = i02 % p.ne12;
    ਅਚਲ int64_t i13 = i03 % p.ne13;

    x    += int64_t(rowx)*ncols;
    ਨਕਾਬ += (i11*p.nb11 + i12*p.nb12 + i13*p.nb13) / ਆਕਾਰ(T) * (ਨਕਾਬ != ਖਾਲੀ_ਸੂਚਕ);
    dst  += int64_t(rowx)*ncols;

    ਅਚਲ ਪੂਰਨ warp_id = ਵਸਤੂ_ct1.get_local_id(2) / WARP_ਮਾਪ;
    ਅਚਲ ਪੂਰਨ lane_id = ਵਸਤੂ_ct1.get_local_id(2) % WARP_ਮਾਪ;

    ਅਚਲ ਦਸ਼ਮਲਵ ਢਲਾਨ = get_alibi_ਢਲਾਨ(p.max_bias, i02, p.n_ਸਿਰ_log2, p.m0, p.m1);

    ਦਸ਼ਮਲਵ * buf_iw = (ਦਸ਼ਮਲਵ *) dpct_local;

    // shared memory buffer to cache values between iterations:
    ਦਸ਼ਮਲਵ *vals = use_shared ? buf_iw + sycl::max(nwarps, WARP_ਮਾਪ) : dst;
    ਦਸ਼ਮਲਵ max_val = sinks ? sinks[i02] : -INFINITY;
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ col0 = 0; col0 < ncols; col0 += ਖੰਡ_ਮਾਪ_3) {
        ਅਚਲ ਪੂਰਨ ਖਾਨਾ = col0 + tid;

        ਜੇ (ncols_template == 0 && ਖਾਨਾ >= ncols) {
            ਰੋਕੋ;
        }

        ਅਚਲ ਦਸ਼ਮਲਵ val = x[ਖਾਨਾ]*p.scale + (ਨਕਾਬ ? ਢਲਾਨ*t2f32(ਨਕਾਬ[ਖਾਨਾ]) : 0.0f);

        vals[ਖਾਨਾ] = val;
        max_val   = sycl::max(max_val, val);
    }
    // find the max value in the block
    max_val = warp_reduce_max<WARP_ਮਾਪ>(max_val);

    ਜੇ (ਖੰਡ_ਮਾਪ_3 > WARP_ਮਾਪ) {
        ਜੇ (lane_id == 0) {
            buf_iw[warp_id] = max_val;
        }
        ਵਸਤੂ_ct1.barrier();

        max_val = -INFINITY;
        ਲਈ (ਪੂਰਨ i = lane_id; i < nwarps; i += WARP_ਮਾਪ) {
            max_val = sycl::max(max_val, buf_iw[i]);
        }
        max_val = warp_reduce_max<WARP_ਮਾਪ>(max_val);
    }
    ਦਸ਼ਮਲਵ tmp = 0.0f; // partial sum

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ col0 = 0; col0 < ncols; col0 += ਖੰਡ_ਮਾਪ_3) {
        ਅਚਲ ਪੂਰਨ ਖਾਨਾ = col0 + tid;

        ਜੇ (ncols_template == 0 && ਖਾਨਾ >= ncols) {
            ਰੋਕੋ;
        }

        ਅਚਲ ਦਸ਼ਮਲਵ val = sycl::native::ਘਾਤਾਂਕ(sycl::max(vals[ਖਾਨਾ] - max_val, -80.0f));
        tmp += val;
        vals[ਖਾਨਾ] = val;
    }
    // find the sum of exps in the block
    tmp = warp_reduce_ਜੋੜ<WARP_ਮਾਪ>(tmp);
    ਜੇ (ਖੰਡ_ਮਾਪ_3 > WARP_ਮਾਪ) {
        ਵਸਤੂ_ct1.barrier();
        ਜੇ (warp_id == 0) {
            buf_iw[lane_id] = 0.0f;
            ਲਈ (size_t i = 1; i < nreduce; i += 1) {
                buf_iw[lane_id + i * WARP_ਮਾਪ] = 0.f;
            }
        }
        ਵਸਤੂ_ct1.barrier();

        ਜੇ (lane_id == 0) {
            buf_iw[warp_id] = tmp;
        }
        ਵਸਤੂ_ct1.barrier();

        tmp = buf_iw[lane_id];
        ਲਈ (size_t i = 1; i < nreduce; i += 1) {
            tmp += buf_iw[lane_id + i * WARP_ਮਾਪ];
        }
        tmp = warp_reduce_ਜੋੜ<WARP_ਮਾਪ>(tmp);
    }
    ਜੇ (sinks) {
        tmp += sycl::native::ਘਾਤਾਂਕ(sycl::max(sinks[i02] - max_val, -80.0f));
    }
    ਅਚਲ ਦਸ਼ਮਲਵ inv_ਜੋੜ = 1.0f / tmp;

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ col0 = 0; col0 < ncols; col0 += ਖੰਡ_ਮਾਪ_3) {
        ਅਚਲ ਪੂਰਨ ਖਾਨਾ = col0 + tid;

        ਜੇ (ncols_template == 0 && ਖਾਨਾ >= ncols) {
            ਵਾਪਸ;
        }

        dst[ਖਾਨਾ] = vals[ਖਾਨਾ] * inv_ਜੋੜ;
    }
}
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ __clang__
#ਹਦਾਇਤ clang diagnostic ਕੱਢੋ
#ਜੇ_ਅੰਤ // __clang__

ਸਥਿਰ ਖਾਲੀ soft_max_ਪਿੱਠ_f32(ਅਚਲ ਦਸ਼ਮਲਵ *grad, ਅਚਲ ਦਸ਼ਮਲਵ *dstf, ਦਸ਼ਮਲਵ *dst,
                              ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਦਸ਼ਮਲਵ scale) {
    ਸਵੈਚਾਲਿਤ      ਵਸਤੂ_ct1 = sycl::ext::oneapi::this_work_ਵਸਤੂ::get_nd_ਵਸਤੂ<3>();
    ਅਚਲ ਪੂਰਨ tid      = ਵਸਤੂ_ct1.get_local_id(2);
    ਅਚਲ ਪੂਰਨ rowx     = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2);

    grad += int64_t(rowx)*ncols;
    dstf += int64_t(rowx)*ncols;
    dst  += int64_t(rowx)*ncols;

    ਦਸ਼ਮਲਵ dgf_dot = 0.0f; // dot product of dst from forward pass and gradients

    ਲਈ (ਪੂਰਨ ਖਾਨਾ = tid; ਖਾਨਾ < ncols; ਖਾਨਾ += WARP_ਮਾਪ) {
        dgf_dot += dstf[ਖਾਨਾ]*grad[ਖਾਨਾ];
    }

    dgf_dot = warp_reduce_ਜੋੜ<WARP_ਮਾਪ>(dgf_dot);

    ਲਈ (ਪੂਰਨ ਖਾਨਾ = tid; ਖਾਨਾ < ncols; ਖਾਨਾ += WARP_ਮਾਪ) {
        dst[ਖਾਨਾ] = scale * (grad[ਖਾਨਾ] - dgf_dot) * dstf[ਖਾਨਾ];
    }
}

ਸਾਂਚਾ <ਪੂਰਨ... Ns, ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ ਖਾਲੀ launch_soft_max_kernels(ਅਚਲ ਦਸ਼ਮਲਵ *           x,
                                    ਅਚਲ T *               ਨਕਾਬ,
                                    ਅਚਲ ਦਸ਼ਮਲਵ *           sinks,
                                    ਦਸ਼ਮਲਵ *                 dst,
                                    ਅਚਲ soft_max_params & p,
                                    dpct::ਕਤਾਰ_ptr         stream,
                                    dpct::dim3              ਖੰਡ_dims,
                                    dpct::dim3              ਖੰਡ_nums,
                                    size_t                  nbytes_shared)
{
    ਸਵੈਚਾਲਿਤ launch_kernel = [=](ਸਵੈਚਾਲਿਤ I) -> ਤਾਰਕਿਕ {
        constexpr ਪੂਰਨ ncols = decltype(I)::value;
        constexpr ਪੂਰਨ ਖੰਡ = (ncols > 1024 ? 1024 : ncols);
        ਜੇ (p.ncols == ncols) {
            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                    sycl::range<1>(nbytes_shared), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
                    [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(
                        WARP_ਮਾਪ)]] {
                        soft_max_f32<ਸੱਚ, ncols, ਖੰਡ>(
                            x, ਨਕਾਬ, sinks, dst, p,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                        GGML_ਅਣਵਰਤਿਆ(ਵਸਤੂ_ct1);
                    });
            });
            ਵਾਪਸ ਸੱਚ;
        }
        ਵਾਪਸ ਝੂਠ;
    };

    // unary fold over launch_kernel
    ਜੇ ((launch_kernel(std::integral_constant<ਪੂਰਨ, Ns>{}) || ...)) {
        ਵਾਪਸ;
    }

    stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range<1>(nbytes_shared), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
            [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1)
                [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                    soft_max_f32<ਸੱਚ, 0, 0>(
                        x, ਨਕਾਬ, sinks, dst, p,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                    GGML_ਅਣਵਰਤਿਆ(ਵਸਤੂ_ct1);
                });
    });
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ ਖਾਲੀ soft_max_f32_sycl(ਅਚਲ ਦਸ਼ਮਲਵ *x, ਅਚਲ T *ਨਕਾਬ,
                              ਅਚਲ ਦਸ਼ਮਲਵ *sinks, ਦਸ਼ਮਲਵ *dst,
                              ਅਚਲ soft_max_params &params,
                              dpct::ਕਤਾਰ_ptr stream, ਪੂਰਨ device) {
    ਪੂਰਨ nth = WARP_ਮਾਪ;
    ਪੂਰਨ max_ਖੰਡ_ਮਾਪ = ggml_sycl_info().max_work_ਸਮੂਹ_sizes[device];
    ਅਚਲ int64_t ncols_x = params.ncols;

    ਜਦੋਂ (nth < ncols_x && nth < max_ਖੰਡ_ਮਾਪ) nth *= 2;
    ਜੇ (nth>max_ਖੰਡ_ਮਾਪ) nth = max_ਖੰਡ_ਮਾਪ;

    ਅਚਲ dpct::dim3 ਖੰਡ_dims(nth, 1, 1);
    ਅਚਲ dpct::dim3 ਖੰਡ_nums(params.ne01, params.ne02, params.ne03);
    ਅਚਲ size_t nbytes_shared =
        (GGML_PAD(ncols_x, WARP_ਮਾਪ) + WARP_ਮਾਪ) * ਆਕਾਰ(ਦਸ਼ਮਲਵ);

    ਅਚਲ ਪੂਰਨ id       = get_current_device_id();
    ਅਚਲ size_t smpbo = ggml_sycl_info().devices[id].smpbo;

    ਜੇ (nbytes_shared <= smpbo && ncols_x <= max_ਖੰਡ_ਮਾਪ) {
        launch_soft_max_kernels<32, 64, 128, 256, 512, 1024, 2048, 4096>(
            x, ਨਕਾਬ, sinks, dst, params, stream, ਖੰਡ_dims, ਖੰਡ_nums,
            nbytes_shared);
    } ਨਹੀਂਤਾਂ {
        ਅਚਲ size_t nbytes_shared_ਨੀਵਾਂ = WARP_ਮਾਪ * ਆਕਾਰ(ਦਸ਼ਮਲਵ);

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(nbytes_shared_ਨੀਵਾਂ), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
                [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1)
                    [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                    soft_max_f32<ਝੂਠ, 0, 0>(
                        x, ਨਕਾਬ, sinks, dst, params,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                    GGML_ਅਣਵਰਤਿਆ(ਵਸਤੂ_ct1);
                });
        });
    }
}

ਸਥਿਰ ਖਾਲੀ soft_max_ਪਿੱਠ_f32_sycl(ਅਚਲ ਦਸ਼ਮਲਵ *   grad,
                                   ਅਚਲ ਦਸ਼ਮਲਵ *   dstf,
                                   ਦਸ਼ਮਲਵ *         dst,
                                   ਅਚਲ ਪੂਰਨ       ncols,
                                   ਅਚਲ ਪੂਰਨ       nrows,
                                   ਅਚਲ ਦਸ਼ਮਲਵ     scale,
                                   dpct::ਕਤਾਰ_ptr stream) {
    ਅਚਲ dpct::dim3 ਖੰਡ_dims(WARP_ਮਾਪ, 1, 1);
    ਅਚਲ dpct::dim3 ਖੰਡ_nums(nrows, 1, 1);

    stream->parallel_for(sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
                         [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
                             soft_max_ਪਿੱਠ_f32(grad, dstf, dst, ncols, scale);
                             GGML_ਅਣਵਰਤਿਆ(ਵਸਤੂ_ct1);
                         });
}

ਖਾਲੀ ggml_sycl_op_soft_max(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];
    ਅਚਲ ggml_tensor * src2 = dst->src[2];

    ਅਚਲ ਦਸ਼ਮਲਵ * src0_d = (ਅਚਲ ਦਸ਼ਮਲਵ *) src0->ਅੰਕੜੇ;
    ਅਚਲ ਖਾਲੀ  * src1_d = src1 ? (ਅਚਲ ਖਾਲੀ *) src1->ਅੰਕੜੇ : ਖਾਲੀ_ਸੂਚਕ;
    ਅਚਲ ਖਾਲੀ  * src2_d = src2 ? (ਅਚਲ ਖਾਲੀ *) src2->ਅੰਕੜੇ : ਖਾਲੀ_ਸੂਚਕ;
    ਦਸ਼ਮਲਵ       *  dst_d = (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ;

    dpct::ਕਤਾਰ_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    // src1 contains mask and it is optional
    GGML_ASSERT(!src1 || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32);

    ਅਚਲ int64_t nrows_x = ggml_nrows(src0);
    ਅਚਲ int64_t nrows_y = src0->ne[1];

    ਅਚਲ int64_t ne00 = src0->ne[0];

    ਦਸ਼ਮਲਵ scale    = 1.0f;
    ਦਸ਼ਮਲਵ max_bias = 0.0f;

    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&scale,    (ਅਚਲ ਦਸ਼ਮਲਵ *) dst->op_params + 0, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&max_bias, (ਅਚਲ ਦਸ਼ਮਲਵ *) dst->op_params + 1, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਅਚਲ ਤਾਰਕਿਕ use_f16 = (src1 && src1->type == GGML_TYPE_F16);

    ਅਚਲ int64_t nb11 = src1 ? src1->nb[1] : 1;
    ਅਚਲ int64_t nb12 = src1 ? src1->nb[2] : 1;
    ਅਚਲ int64_t nb13 = src1 ? src1->nb[3] : 1;

    ਅਚਲ int64_t ne12 = src1 ? src1->ne[2] : 1;
    ਅਚਲ int64_t ne13 = src1 ? src1->ne[3] : 1;

    ਅਚਲ uint32_t n_ਸਿਰ      = src0->ne[2];
    ਅਚਲ uint32_t n_ਸਿਰ_log2 = 1u << (uint32_t) floorf(log2f((ਦਸ਼ਮਲਵ) n_ਸਿਰ));

    ਅਚਲ ਦਸ਼ਮਲਵ m0 = powf(2.0f, -(max_bias       ) / n_ਸਿਰ_log2);
    ਅਚਲ ਦਸ਼ਮਲਵ m1 = powf(2.0f, -(max_bias / 2.0f) / n_ਸਿਰ_log2);


    soft_max_params params = {};
    params.nheads = src0->ne[2];
    params.n_ਸਿਰ_log2 = n_ਸਿਰ_log2;
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

    ਜੇ (use_f16) {
        soft_max_f32_sycl(src0_d, (ਅਚਲ sycl::half *)src1_d,
                          (ਅਚਲ ਦਸ਼ਮਲਵ *)src2_d, dst_d, params, stream,
                          ctx.device);
    } ਨਹੀਂਤਾਂ {
        soft_max_f32_sycl(src0_d, (ਅਚਲ ਦਸ਼ਮਲਵ *)src1_d, (ਅਚਲ ਦਸ਼ਮਲਵ *)src2_d,
                          dst_d, params, stream, ctx.device);
    }
}

ਖਾਲੀ ggml_sycl_op_soft_max_ਪਿੱਠ(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ਅਚਲ ggml_tensor * src0 = dst->src[0]; // grad
    ਅਚਲ ggml_tensor * src1 = dst->src[1]; // forward pass output

    ਅਚਲ ਦਸ਼ਮਲਵ * src0_d = (ਅਚਲ ਦਸ਼ਮਲਵ *) src0->ਅੰਕੜੇ;
    ਅਚਲ ਦਸ਼ਮਲਵ * src1_d = (ਅਚਲ ਦਸ਼ਮਲਵ *) src1->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ       * dst_d  = (ਦਸ਼ਮਲਵ       *) dst->ਅੰਕੜੇ;

    dpct::ਕਤਾਰ_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    ਅਚਲ int64_t ncols = src0->ne[0];
    ਅਚਲ int64_t nrows = ggml_nrows(src0);

    ਦਸ਼ਮਲਵ scale    = 1.0f;
    ਦਸ਼ਮਲਵ max_bias = 0.0f;

    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&scale,    (ਅਚਲ ਦਸ਼ਮਲਵ *) dst->op_params + 0, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&max_bias, (ਅਚਲ ਦਸ਼ਮਲਵ *) dst->op_params + 1, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    GGML_ASSERT(max_bias == 0.0f);

    soft_max_ਪਿੱਠ_f32_sycl(src0_d, src1_d, dst_d, ncols, nrows, scale, stream);
}
