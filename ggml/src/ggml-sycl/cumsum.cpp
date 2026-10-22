#ಅನ್ತರ್ಭಾವ "cumsum.hpp"
#ಅನ್ತರ್ಭಾವ "common.hpp"

#ಅನ್ತರ್ಭಾವ <algorithm>

#ವ್ಯಾಖ್ಯಾನಿಸು SYCL_CUMSUM_ಖಂಡ_ಗಾತ್ರ 256

ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ warp_ಪೂರ್ವಪ್ರತ್ಯಯ_inclusive_ಮೊತ್ತ_f32(ದಶಮಾಂಶ x, ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> & ಅಂಶ) {
    ಹಿಂತಿರುಗಿ sycl::inclusive_ಪರಿಶೀಲನೆ_over_ಗುಂಪು(ಅಂಶ.get_sub_ಗುಂಪು(), x, sycl::plus<ದಶಮಾಂಶ>());
}

ಸ್ಥಿರ ಶೂನ್ಯ cumsum_f32_kernel(
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * __restrict__ src, ದಶಮಾಂಶ * __restrict__ dst,
        ಸ್ಥಿರಾಂಕ int64_t ne00, ಸ್ಥಿರಾಂಕ int64_t ne01, ಸ್ಥಿರಾಂಕ int64_t ne02, ಸ್ಥಿರಾಂಕ int64_t ne03,
        ಸ್ಥಿರಾಂಕ int64_t s01, ಸ್ಥಿರಾಂಕ int64_t s02, ಸ್ಥಿರಾಂಕ int64_t s03,
        ಸ್ಥಿರಾಂಕ int64_t  d1, ಸ್ಥಿರಾಂಕ int64_t  d2, ಸ್ಥಿರಾಂಕ int64_t  d3,
        ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> & ಅಂಶ, ದಶಮಾಂಶ * smem) {

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ tid = ಅಂಶ.get_local_id(2);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಗಾತ್ರ_3 = ಅಂಶ.get_local_range(2);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ lane = tid % WARP_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ warp = tid / WARP_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ warps_ಪ್ರತಿ_ಖಂಡ = ಖಂಡ_ಗಾತ್ರ_3 / WARP_ಗಾತ್ರ;

    ದಶಮಾಂಶ * s_vals      = smem;
    ದಶಮಾಂಶ * s_warp_sums = smem + ಖಂಡ_ಗಾತ್ರ_3;
    ದಶಮಾಂಶ * s_ವಾಹಕ     = smem + ಖಂಡ_ಗಾತ್ರ_3 + warps_ಪ್ರತಿ_ಖಂಡ;

    ಆದರೆ (tid == 0) {
        s_ವಾಹಕ[0] = 0.0f;
    }
    ಅಂಶ.barrier(sycl::access::fence_space::local_space);

    ಸ್ಥಿರಾಂಕ int64_t i3 = ಅಂಶ.get_ಗುಂಪು(0);
    ಸ್ಥಿರಾಂಕ int64_t i2 = ಅಂಶ.get_ಗುಂಪು(1);
    ಸ್ಥಿರಾಂಕ int64_t i1 = ಅಂಶ.get_ಗುಂಪು(2);
    ಆದರೆ (i3 >= ne03 || i2 >= ne02 || i1 >= ne01) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src_ಅಡ್ಡಸಾಲು = src + i1 * s01 + i2 * s02 + i3 * s03;
    ದಶಮಾಂಶ       * dst_ಅಡ್ಡಸಾಲು = dst + i1 * d1  + i2 * d2  + i3 * d3;

    constexpr ಪೂರ್ಣಾಂಕ ಅಂಕೆ_unroll = 4;
    ದಶಮಾಂಶ ತಾತ್ಕಾಲಿಕ[ಅಂಕೆ_unroll];

    ಗಾಗಿ (int64_t i = 0; i < ne00; i += ಅಂಕೆ_unroll * ಖಂಡ_ಗಾತ್ರ_3) {
        int64_t idx = i + tid * ಅಂಕೆ_unroll;

        ತಾತ್ಕಾಲಿಕ[0] = (idx < ne00 ? src_ಅಡ್ಡಸಾಲು[idx] : 0.0f);
#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 1; j < ಅಂಕೆ_unroll; j++) {
            ತಾತ್ಕಾಲಿಕ[j] = ತಾತ್ಕಾಲಿಕ[j - 1];
            ಆದರೆ (idx + j < ne00) {
                ತಾತ್ಕಾಲಿಕ[j] += src_ಅಡ್ಡಸಾಲು[idx + j];
            }
        }

        ದಶಮಾಂಶ val = (idx < ne00) ? ತಾತ್ಕಾಲಿಕ[ಅಂಕೆ_unroll - 1] : 0.0f;

        val = warp_ಪೂರ್ವಪ್ರತ್ಯಯ_inclusive_ಮೊತ್ತ_f32(val, ಅಂಶ);
        s_vals[tid] = val;

        ಆದರೆ (lane == WARP_ಗಾತ್ರ - 1) {
            s_warp_sums[warp] = val;
        }
        ಅಂಶ.barrier(sycl::access::fence_space::local_space);

        ಆದರೆ (warp == 0) {
            ದಶಮಾಂಶ w = (tid < warps_ಪ್ರತಿ_ಖಂಡ) ? s_warp_sums[tid] : 0.0f;
            ದಶಮಾಂಶ inc = warp_ಪೂರ್ವಪ್ರತ್ಯಯ_inclusive_ಮೊತ್ತ_f32(w, ಅಂಶ);
            ಆದರೆ (tid < warps_ಪ್ರತಿ_ಖಂಡ) {
                s_warp_sums[tid] = inc - w;
            }
            ಆದರೆ (tid == warps_ಪ್ರತಿ_ಖಂಡ - 1) {
                s_ವಾಹಕ[1] = inc;
            }
        }
        ಅಂಶ.barrier(sycl::access::fence_space::local_space);

        ದಶಮಾಂಶ ವಾಹಕ = s_ವಾಹಕ[0];
        ದಶಮಾಂಶ final_ಸ್ಥಾನಾಂತರ = s_vals[tid] + s_warp_sums[warp] + ವಾಹಕ - ತಾತ್ಕಾಲಿಕ[ಅಂಕೆ_unroll - 1];

#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < ಅಂಕೆ_unroll; j++) {
            ಆದರೆ (idx + j < ne00) {
                dst_ಅಡ್ಡಸಾಲು[idx + j] = ತಾತ್ಕಾಲಿಕ[j] + final_ಸ್ಥಾನಾಂತರ;
            }
        }

        ಅಂಶ.barrier(sycl::access::fence_space::local_space);

        ಆದರೆ (tid == 0) {
            s_ವಾಹಕ[0] += s_ವಾಹಕ[1];
        }
    }
}

ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_sycl_op_cumsum(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    dpct::ಸರತಿ_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_ಗಣ_device(ctx.device));

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src_d = static_cast<ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *>(src0->ದತ್ತಾಂಶ);
    ದಶಮಾಂಶ       * dst_d = static_cast<ದಶಮಾಂಶ *>(dst->ದತ್ತಾಂಶ);

    ಸ್ಥಿರಾಂಕ int64_t ne00 = src0->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t ne01 = src0->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t ne02 = src0->ne[2];
    ಸ್ಥಿರಾಂಕ int64_t ne03 = src0->ne[3];

    ಸ್ಥಿರಾಂಕ size_t ts = ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ int64_t s01 = src0->nb[1] / ts;
    ಸ್ಥಿರಾಂಕ int64_t s02 = src0->nb[2] / ts;
    ಸ್ಥಿರಾಂಕ int64_t s03 = src0->nb[3] / ts;
    ಸ್ಥಿರಾಂಕ int64_t d1  = dst->nb[1] / ts;
    ಸ್ಥಿರಾಂಕ int64_t d2  = dst->nb[2] / ts;
    ಸ್ಥಿರಾಂಕ int64_t d3  = dst->nb[3] / ts;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_warps = (ne00 + WARP_ಗಾತ್ರ - 1) / WARP_ಗಾತ್ರ;
    ಪೂರ್ಣಾಂಕ ಖಂಡ_ಗಾತ್ರ_3 = ಅಂಕೆ_warps * WARP_ಗಾತ್ರ;
    ಖಂಡ_ಗಾತ್ರ_3 = std::min(ಖಂಡ_ಗಾತ್ರ_3, SYCL_CUMSUM_ಖಂಡ_ಗಾತ್ರ);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ warps_ಪ್ರತಿ_ಖಂಡ = ಖಂಡ_ಗಾತ್ರ_3 / WARP_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ smem_ಗಾತ್ರ = ಖಂಡ_ಗಾತ್ರ_3 + warps_ಪ್ರತಿ_ಖಂಡ + 2;

    ಸ್ಥಿರಾಂಕ sycl::range<3> grid(ne03, ne02, ne01);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ(1, 1, ಖಂಡ_ಗಾತ್ರ_3);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<ದಶಮಾಂಶ, 1> smem_acc(sycl::range<1>(smem_ಗಾತ್ರ), cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(grid * ಖಂಡ, ಖಂಡ),
            [=](sycl::nd_ಅಂಶ<3> ಅಂಶ) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                cumsum_f32_kernel(src_d, dst_d, ne00, ne01, ne02, ne03,
                                  s01, s02, s03, d1, d2, d3,
                                  ಅಂಶ, get_ಸೂಚಕ(smem_acc));
            });
    });
}

ಶೂನ್ಯ ggml_sycl_cumsum(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_cumsum(ctx, dst);
}
