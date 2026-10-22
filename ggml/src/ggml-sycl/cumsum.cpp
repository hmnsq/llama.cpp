#ਅਨ੍ਤਰ੍ਭਾਵ "cumsum.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "common.hpp"

#ਅਨ੍ਤਰ੍ਭਾਵ <algorithm>

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SYCL_CUMSUM_ਖੰਡ_ਮਾਪ 256

ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ warp_ਅਗੇਤਰ_inclusive_ਜੋੜ_f32(ਦਸ਼ਮਲਵ x, ਅਚਲ sycl::nd_ਵਸਤੂ<3> & ਵਸਤੂ) {
    ਵਾਪਸ sycl::inclusive_ਜਾਂਚ_over_ਸਮੂਹ(ਵਸਤੂ.get_sub_ਸਮੂਹ(), x, sycl::plus<ਦਸ਼ਮਲਵ>());
}

ਸਥਿਰ ਖਾਲੀ cumsum_f32_kernel(
        ਅਚਲ ਦਸ਼ਮਲਵ * __restrict__ src, ਦਸ਼ਮਲਵ * __restrict__ dst,
        ਅਚਲ int64_t ne00, ਅਚਲ int64_t ne01, ਅਚਲ int64_t ne02, ਅਚਲ int64_t ne03,
        ਅਚਲ int64_t s01, ਅਚਲ int64_t s02, ਅਚਲ int64_t s03,
        ਅਚਲ int64_t  d1, ਅਚਲ int64_t  d2, ਅਚਲ int64_t  d3,
        ਅਚਲ sycl::nd_ਵਸਤੂ<3> & ਵਸਤੂ, ਦਸ਼ਮਲਵ * smem) {

    ਅਚਲ ਪੂਰਨ tid = ਵਸਤੂ.get_local_id(2);
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਮਾਪ_3 = ਵਸਤੂ.get_local_range(2);
    ਅਚਲ ਪੂਰਨ lane = tid % WARP_ਮਾਪ;
    ਅਚਲ ਪੂਰਨ warp = tid / WARP_ਮਾਪ;
    ਅਚਲ ਪੂਰਨ warps_ਪ੍ਰਤੀ_ਖੰਡ = ਖੰਡ_ਮਾਪ_3 / WARP_ਮਾਪ;

    ਦਸ਼ਮਲਵ * s_vals      = smem;
    ਦਸ਼ਮਲਵ * s_warp_sums = smem + ਖੰਡ_ਮਾਪ_3;
    ਦਸ਼ਮਲਵ * s_ਹਾਸਲ     = smem + ਖੰਡ_ਮਾਪ_3 + warps_ਪ੍ਰਤੀ_ਖੰਡ;

    ਜੇ (tid == 0) {
        s_ਹਾਸਲ[0] = 0.0f;
    }
    ਵਸਤੂ.barrier(sycl::access::fence_space::local_space);

    ਅਚਲ int64_t i3 = ਵਸਤੂ.get_ਸਮੂਹ(0);
    ਅਚਲ int64_t i2 = ਵਸਤੂ.get_ਸਮੂਹ(1);
    ਅਚਲ int64_t i1 = ਵਸਤੂ.get_ਸਮੂਹ(2);
    ਜੇ (i3 >= ne03 || i2 >= ne02 || i1 >= ne01) {
        ਵਾਪਸ;
    }

    ਅਚਲ ਦਸ਼ਮਲਵ * src_ਪੰਕਤੀ = src + i1 * s01 + i2 * s02 + i3 * s03;
    ਦਸ਼ਮਲਵ       * dst_ਪੰਕਤੀ = dst + i1 * d1  + i2 * d2  + i3 * d3;

    constexpr ਪੂਰਨ ਅੰਕ_unroll = 4;
    ਦਸ਼ਮਲਵ ਆਰਜ਼ੀ[ਅੰਕ_unroll];

    ਲਈ (int64_t i = 0; i < ne00; i += ਅੰਕ_unroll * ਖੰਡ_ਮਾਪ_3) {
        int64_t idx = i + tid * ਅੰਕ_unroll;

        ਆਰਜ਼ੀ[0] = (idx < ne00 ? src_ਪੰਕਤੀ[idx] : 0.0f);
#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ j = 1; j < ਅੰਕ_unroll; j++) {
            ਆਰਜ਼ੀ[j] = ਆਰਜ਼ੀ[j - 1];
            ਜੇ (idx + j < ne00) {
                ਆਰਜ਼ੀ[j] += src_ਪੰਕਤੀ[idx + j];
            }
        }

        ਦਸ਼ਮਲਵ val = (idx < ne00) ? ਆਰਜ਼ੀ[ਅੰਕ_unroll - 1] : 0.0f;

        val = warp_ਅਗੇਤਰ_inclusive_ਜੋੜ_f32(val, ਵਸਤੂ);
        s_vals[tid] = val;

        ਜੇ (lane == WARP_ਮਾਪ - 1) {
            s_warp_sums[warp] = val;
        }
        ਵਸਤੂ.barrier(sycl::access::fence_space::local_space);

        ਜੇ (warp == 0) {
            ਦਸ਼ਮਲਵ w = (tid < warps_ਪ੍ਰਤੀ_ਖੰਡ) ? s_warp_sums[tid] : 0.0f;
            ਦਸ਼ਮਲਵ inc = warp_ਅਗੇਤਰ_inclusive_ਜੋੜ_f32(w, ਵਸਤੂ);
            ਜੇ (tid < warps_ਪ੍ਰਤੀ_ਖੰਡ) {
                s_warp_sums[tid] = inc - w;
            }
            ਜੇ (tid == warps_ਪ੍ਰਤੀ_ਖੰਡ - 1) {
                s_ਹਾਸਲ[1] = inc;
            }
        }
        ਵਸਤੂ.barrier(sycl::access::fence_space::local_space);

        ਦਸ਼ਮਲਵ ਹਾਸਲ = s_ਹਾਸਲ[0];
        ਦਸ਼ਮਲਵ final_ਵਿੱਥ = s_vals[tid] + s_warp_sums[warp] + ਹਾਸਲ - ਆਰਜ਼ੀ[ਅੰਕ_unroll - 1];

#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ j = 0; j < ਅੰਕ_unroll; j++) {
            ਜੇ (idx + j < ne00) {
                dst_ਪੰਕਤੀ[idx + j] = ਆਰਜ਼ੀ[j] + final_ਵਿੱਥ;
            }
        }

        ਵਸਤੂ.barrier(sycl::access::fence_space::local_space);

        ਜੇ (tid == 0) {
            s_ਹਾਸਲ[0] += s_ਹਾਸਲ[1];
        }
    }
}

ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_sycl_op_cumsum(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    dpct::ਕਤਾਰ_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_ਸੈੱਟ_device(ctx.device));

    ਅਚਲ ਦਸ਼ਮਲਵ * src_d = static_cast<ਅਚਲ ਦਸ਼ਮਲਵ *>(src0->ਅੰਕੜੇ);
    ਦਸ਼ਮਲਵ       * dst_d = static_cast<ਦਸ਼ਮਲਵ *>(dst->ਅੰਕੜੇ);

    ਅਚਲ int64_t ne00 = src0->ne[0];
    ਅਚਲ int64_t ne01 = src0->ne[1];
    ਅਚਲ int64_t ne02 = src0->ne[2];
    ਅਚਲ int64_t ne03 = src0->ne[3];

    ਅਚਲ size_t ts = ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ int64_t s01 = src0->nb[1] / ts;
    ਅਚਲ int64_t s02 = src0->nb[2] / ts;
    ਅਚਲ int64_t s03 = src0->nb[3] / ts;
    ਅਚਲ int64_t d1  = dst->nb[1] / ts;
    ਅਚਲ int64_t d2  = dst->nb[2] / ts;
    ਅਚਲ int64_t d3  = dst->nb[3] / ts;

    ਅਚਲ ਪੂਰਨ ਅੰਕ_warps = (ne00 + WARP_ਮਾਪ - 1) / WARP_ਮਾਪ;
    ਪੂਰਨ ਖੰਡ_ਮਾਪ_3 = ਅੰਕ_warps * WARP_ਮਾਪ;
    ਖੰਡ_ਮਾਪ_3 = std::min(ਖੰਡ_ਮਾਪ_3, SYCL_CUMSUM_ਖੰਡ_ਮਾਪ);
    ਅਚਲ ਪੂਰਨ warps_ਪ੍ਰਤੀ_ਖੰਡ = ਖੰਡ_ਮਾਪ_3 / WARP_ਮਾਪ;
    ਅਚਲ ਪੂਰਨ smem_ਮਾਪ = ਖੰਡ_ਮਾਪ_3 + warps_ਪ੍ਰਤੀ_ਖੰਡ + 2;

    ਅਚਲ sycl::range<3> grid(ne03, ne02, ne01);
    ਅਚਲ sycl::range<3> ਖੰਡ(1, 1, ਖੰਡ_ਮਾਪ_3);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<ਦਸ਼ਮਲਵ, 1> smem_acc(sycl::range<1>(smem_ਮਾਪ), cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(grid * ਖੰਡ, ਖੰਡ),
            [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                cumsum_f32_kernel(src_d, dst_d, ne00, ne01, ne02, ne03,
                                  s01, s02, s03, d1, d2, d3,
                                  ਵਸਤੂ, get_ਸੂਚਕ(smem_acc));
            });
    });
}

ਖਾਲੀ ggml_sycl_cumsum(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_cumsum(ctx, dst);
}
