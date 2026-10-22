#ضمن "cumsum.hpp"
#ضمن "common.hpp"

#ضمن <algorithm>

#تعریف_کرو SYCL_CUMSUM_بلاک_ناپ 256

ساكن __dpct_inline__ عائم warp_اگیتر_inclusive_جوڑ_f32(عائم x, ثابت sycl::nd_شے<3> & شے) {
    أرجع sycl::inclusive_جانچ_over_گروہ(شے.get_sub_گروہ(), x, sycl::plus<عائم>());
}

ساكن فراغ cumsum_f32_kernel(
        ثابت عائم * __restrict__ src, عائم * __restrict__ dst,
        ثابت int64_t ne00, ثابت int64_t ne01, ثابت int64_t ne02, ثابت int64_t ne03,
        ثابت int64_t s01, ثابت int64_t s02, ثابت int64_t s03,
        ثابت int64_t  d1, ثابت int64_t  d2, ثابت int64_t  d3,
        ثابت sycl::nd_شے<3> & شے, عائم * smem) {

    ثابت صحيح tid = شے.get_local_id(2);
    ثابت صحيح بلاک_ناپ_3 = شے.get_local_range(2);
    ثابت صحيح lane = tid % WARP_ناپ;
    ثابت صحيح warp = tid / WARP_ناپ;
    ثابت صحيح warps_فی_بلاک = بلاک_ناپ_3 / WARP_ناپ;

    عائم * s_vals      = smem;
    عائم * s_warp_sums = smem + بلاک_ناپ_3;
    عائم * s_حاصل     = smem + بلاک_ناپ_3 + warps_فی_بلاک;

    إذا (tid == 0) {
        s_حاصل[0] = 0.0f;
    }
    شے.barrier(sycl::access::fence_space::local_space);

    ثابت int64_t i3 = شے.get_گروہ(0);
    ثابت int64_t i2 = شے.get_گروہ(1);
    ثابت int64_t i1 = شے.get_گروہ(2);
    إذا (i3 >= ne03 || i2 >= ne02 || i1 >= ne01) {
        أرجع;
    }

    ثابت عائم * src_صف = src + i1 * s01 + i2 * s02 + i3 * s03;
    عائم       * dst_صف = dst + i1 * d1  + i2 * d2  + i3 * d3;

    constexpr صحيح عدد_unroll = 4;
    عائم عارضی[عدد_unroll];

    لكل (int64_t i = 0; i < ne00; i += عدد_unroll * بلاک_ناپ_3) {
        int64_t idx = i + tid * عدد_unroll;

        عارضی[0] = (idx < ne00 ? src_صف[idx] : 0.0f);
#ہدایت unroll
        لكل (صحيح j = 1; j < عدد_unroll; j++) {
            عارضی[j] = عارضی[j - 1];
            إذا (idx + j < ne00) {
                عارضی[j] += src_صف[idx + j];
            }
        }

        عائم val = (idx < ne00) ? عارضی[عدد_unroll - 1] : 0.0f;

        val = warp_اگیتر_inclusive_جوڑ_f32(val, شے);
        s_vals[tid] = val;

        إذا (lane == WARP_ناپ - 1) {
            s_warp_sums[warp] = val;
        }
        شے.barrier(sycl::access::fence_space::local_space);

        إذا (warp == 0) {
            عائم w = (tid < warps_فی_بلاک) ? s_warp_sums[tid] : 0.0f;
            عائم inc = warp_اگیتر_inclusive_جوڑ_f32(w, شے);
            إذا (tid < warps_فی_بلاک) {
                s_warp_sums[tid] = inc - w;
            }
            إذا (tid == warps_فی_بلاک - 1) {
                s_حاصل[1] = inc;
            }
        }
        شے.barrier(sycl::access::fence_space::local_space);

        عائم حاصل = s_حاصل[0];
        عائم final_وتھ = s_vals[tid] + s_warp_sums[warp] + حاصل - عارضی[عدد_unroll - 1];

#ہدایت unroll
        لكل (صحيح j = 0; j < عدد_unroll; j++) {
            إذا (idx + j < ne00) {
                dst_صف[idx + j] = عارضی[j] + final_وتھ;
            }
        }

        شے.barrier(sycl::access::fence_space::local_space);

        إذا (tid == 0) {
            s_حاصل[0] += s_حاصل[1];
        }
    }
}

مضمن فراغ ggml_sycl_op_cumsum(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    dpct::قطار_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_سیٹ_device(ctx.device));

    ثابت عائم * src_d = static_cast<ثابت عائم *>(src0->انکڑے);
    عائم       * dst_d = static_cast<عائم *>(dst->انکڑے);

    ثابت int64_t ne00 = src0->ne[0];
    ثابت int64_t ne01 = src0->ne[1];
    ثابت int64_t ne02 = src0->ne[2];
    ثابت int64_t ne03 = src0->ne[3];

    ثابت size_t ts = حجم(عائم);
    ثابت int64_t s01 = src0->nb[1] / ts;
    ثابت int64_t s02 = src0->nb[2] / ts;
    ثابت int64_t s03 = src0->nb[3] / ts;
    ثابت int64_t d1  = dst->nb[1] / ts;
    ثابت int64_t d2  = dst->nb[2] / ts;
    ثابت int64_t d3  = dst->nb[3] / ts;

    ثابت صحيح عدد_warps = (ne00 + WARP_ناپ - 1) / WARP_ناپ;
    صحيح بلاک_ناپ_3 = عدد_warps * WARP_ناپ;
    بلاک_ناپ_3 = std::min(بلاک_ناپ_3, SYCL_CUMSUM_بلاک_ناپ);
    ثابت صحيح warps_فی_بلاک = بلاک_ناپ_3 / WARP_ناپ;
    ثابت صحيح smem_ناپ = بلاک_ناپ_3 + warps_فی_بلاک + 2;

    ثابت sycl::range<3> grid(ne03, ne02, ne01);
    ثابت sycl::range<3> بلاک(1, 1, بلاک_ناپ_3);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<عائم, 1> smem_acc(sycl::range<1>(smem_ناپ), cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(grid * بلاک, بلاک),
            [=](sycl::nd_شے<3> شے) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                cumsum_f32_kernel(src_d, dst_d, ne00, ne01, ne02, ne03,
                                  s01, s02, s03, d1, d2, d3,
                                  شے, get_اشارہ_گر(smem_acc));
            });
    });
}

فراغ ggml_sycl_cumsum(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_cumsum(ctx, dst);
}
