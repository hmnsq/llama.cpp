#ضمن "cumsum.hpp"
#ضمن "common.hpp"

#ضمن <algorithm>

#وصف_ڪريو SYCL_CUMSUM_بلاڪ_سائيز 256

ساكن __dpct_inline__ عائم warp_اڳياڙي_inclusive_جوڙ_f32(عائم x, ثابت sycl::nd_شي<3> & شي) {
    أرجع sycl::inclusive_جاچ_over_گروهه(شي.get_sub_گروهه(), x, sycl::plus<عائم>());
}

ساكن فراغ cumsum_f32_kernel(
        ثابت عائم * __restrict__ src, عائم * __restrict__ dst,
        ثابت int64_t ne00, ثابت int64_t ne01, ثابت int64_t ne02, ثابت int64_t ne03,
        ثابت int64_t s01, ثابت int64_t s02, ثابت int64_t s03,
        ثابت int64_t  d1, ثابت int64_t  d2, ثابت int64_t  d3,
        ثابت sycl::nd_شي<3> & شي, عائم * smem) {

    ثابت صحيح tid = شي.get_local_id(2);
    ثابت صحيح بلاڪ_سائيز_3 = شي.get_local_range(2);
    ثابت صحيح lane = tid % WARP_سائيز;
    ثابت صحيح warp = tid / WARP_سائيز;
    ثابت صحيح warps_في_بلاڪ = بلاڪ_سائيز_3 / WARP_سائيز;

    عائم * s_vals      = smem;
    عائم * s_warp_sums = smem + بلاڪ_سائيز_3;
    عائم * s_حاصل     = smem + بلاڪ_سائيز_3 + warps_في_بلاڪ;

    إذا (tid == 0) {
        s_حاصل[0] = 0.0f;
    }
    شي.barrier(sycl::access::fence_space::local_space);

    ثابت int64_t i3 = شي.get_گروهه(0);
    ثابت int64_t i2 = شي.get_گروهه(1);
    ثابت int64_t i1 = شي.get_گروهه(2);
    إذا (i3 >= ne03 || i2 >= ne02 || i1 >= ne01) {
        أرجع;
    }

    ثابت عائم * src_صف = src + i1 * s01 + i2 * s02 + i3 * s03;
    عائم       * dst_صف = dst + i1 * d1  + i2 * d2  + i3 * d3;

    constexpr صحيح انگ_unroll = 4;
    عائم عارضي[انگ_unroll];

    لكل (int64_t i = 0; i < ne00; i += انگ_unroll * بلاڪ_سائيز_3) {
        int64_t idx = i + tid * انگ_unroll;

        عارضي[0] = (idx < ne00 ? src_صف[idx] : 0.0f);
#هدايت unroll
        لكل (صحيح j = 1; j < انگ_unroll; j++) {
            عارضي[j] = عارضي[j - 1];
            إذا (idx + j < ne00) {
                عارضي[j] += src_صف[idx + j];
            }
        }

        عائم val = (idx < ne00) ? عارضي[انگ_unroll - 1] : 0.0f;

        val = warp_اڳياڙي_inclusive_جوڙ_f32(val, شي);
        s_vals[tid] = val;

        إذا (lane == WARP_سائيز - 1) {
            s_warp_sums[warp] = val;
        }
        شي.barrier(sycl::access::fence_space::local_space);

        إذا (warp == 0) {
            عائم w = (tid < warps_في_بلاڪ) ? s_warp_sums[tid] : 0.0f;
            عائم inc = warp_اڳياڙي_inclusive_جوڙ_f32(w, شي);
            إذا (tid < warps_في_بلاڪ) {
                s_warp_sums[tid] = inc - w;
            }
            إذا (tid == warps_في_بلاڪ - 1) {
                s_حاصل[1] = inc;
            }
        }
        شي.barrier(sycl::access::fence_space::local_space);

        عائم حاصل = s_حاصل[0];
        عائم final_وٿي = s_vals[tid] + s_warp_sums[warp] + حاصل - عارضي[انگ_unroll - 1];

#هدايت unroll
        لكل (صحيح j = 0; j < انگ_unroll; j++) {
            إذا (idx + j < ne00) {
                dst_صف[idx + j] = عارضي[j] + final_وٿي;
            }
        }

        شي.barrier(sycl::access::fence_space::local_space);

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
    SYCL_CHECK(ggml_sycl_سيٽ_device(ctx.device));

    ثابت عائم * src_d = static_cast<ثابت عائم *>(src0->معلومات);
    عائم       * dst_d = static_cast<عائم *>(dst->معلومات);

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

    ثابت صحيح انگ_warps = (ne00 + WARP_سائيز - 1) / WARP_سائيز;
    صحيح بلاڪ_سائيز_3 = انگ_warps * WARP_سائيز;
    بلاڪ_سائيز_3 = std::min(بلاڪ_سائيز_3, SYCL_CUMSUM_بلاڪ_سائيز);
    ثابت صحيح warps_في_بلاڪ = بلاڪ_سائيز_3 / WARP_سائيز;
    ثابت صحيح smem_سائيز = بلاڪ_سائيز_3 + warps_في_بلاڪ + 2;

    ثابت sycl::range<3> grid(ne03, ne02, ne01);
    ثابت sycl::range<3> بلاڪ(1, 1, بلاڪ_سائيز_3);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<عائم, 1> smem_acc(sycl::range<1>(smem_سائيز), cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(grid * بلاڪ, بلاڪ),
            [=](sycl::nd_شي<3> شي) [[sycl::reqd_sub_گروهه_سائيز(WARP_سائيز)]] {
                cumsum_f32_kernel(src_d, dst_d, ne00, ne01, ne02, ne03,
                                  s01, s02, s03, d1, d2, d3,
                                  شي, get_اشارو(smem_acc));
            });
    });
}

فراغ ggml_sycl_cumsum(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_cumsum(ctx, dst);
}
