#ضمن "cumsum.hpp"
#ضمن "common.hpp"

#ضمن <algorithm>

#تعریف_کن SYCL_CUMSUM_بلوک_اندازه 256

ساكن __dpct_inline__ عائم warp_پیشوند_inclusive_جمع_f32(عائم x, ثابت sycl::nd_قلم<3> & قلم) {
    أرجع sycl::inclusive_پیمایش_over_گروه(قلم.get_sub_گروه(), x, sycl::plus<عائم>());
}

ساكن فراغ cumsum_f32_kernel(
        ثابت عائم * __restrict__ src, عائم * __restrict__ dst,
        ثابت int64_t ne00, ثابت int64_t ne01, ثابت int64_t ne02, ثابت int64_t ne03,
        ثابت int64_t s01, ثابت int64_t s02, ثابت int64_t s03,
        ثابت int64_t  d1, ثابت int64_t  d2, ثابت int64_t  d3,
        ثابت sycl::nd_قلم<3> & قلم, عائم * smem) {

    ثابت صحيح tid = قلم.get_local_id(2);
    ثابت صحيح بلوک_اندازه_3 = قلم.get_local_range(2);
    ثابت صحيح lane = tid % WARP_اندازه;
    ثابت صحيح warp = tid / WARP_اندازه;
    ثابت صحيح warps_در_هر_بلوک = بلوک_اندازه_3 / WARP_اندازه;

    عائم * s_vals      = smem;
    عائم * s_warp_sums = smem + بلوک_اندازه_3;
    عائم * s_انتقال     = smem + بلوک_اندازه_3 + warps_در_هر_بلوک;

    إذا (tid == 0) {
        s_انتقال[0] = 0.0f;
    }
    قلم.barrier(sycl::access::fence_space::local_space);

    ثابت int64_t i3 = قلم.get_گروه(0);
    ثابت int64_t i2 = قلم.get_گروه(1);
    ثابت int64_t i1 = قلم.get_گروه(2);
    إذا (i3 >= ne03 || i2 >= ne02 || i1 >= ne01) {
        أرجع;
    }

    ثابت عائم * src_ردیف = src + i1 * s01 + i2 * s02 + i3 * s03;
    عائم       * dst_ردیف = dst + i1 * d1  + i2 * d2  + i3 * d3;

    constexpr صحيح رقم_unroll = 4;
    عائم موقت[رقم_unroll];

    لكل (int64_t i = 0; i < ne00; i += رقم_unroll * بلوک_اندازه_3) {
        int64_t idx = i + tid * رقم_unroll;

        موقت[0] = (idx < ne00 ? src_ردیف[idx] : 0.0f);
#رهنمود unroll
        لكل (صحيح j = 1; j < رقم_unroll; j++) {
            موقت[j] = موقت[j - 1];
            إذا (idx + j < ne00) {
                موقت[j] += src_ردیف[idx + j];
            }
        }

        عائم val = (idx < ne00) ? موقت[رقم_unroll - 1] : 0.0f;

        val = warp_پیشوند_inclusive_جمع_f32(val, قلم);
        s_vals[tid] = val;

        إذا (lane == WARP_اندازه - 1) {
            s_warp_sums[warp] = val;
        }
        قلم.barrier(sycl::access::fence_space::local_space);

        إذا (warp == 0) {
            عائم w = (tid < warps_در_هر_بلوک) ? s_warp_sums[tid] : 0.0f;
            عائم inc = warp_پیشوند_inclusive_جمع_f32(w, قلم);
            إذا (tid < warps_در_هر_بلوک) {
                s_warp_sums[tid] = inc - w;
            }
            إذا (tid == warps_در_هر_بلوک - 1) {
                s_انتقال[1] = inc;
            }
        }
        قلم.barrier(sycl::access::fence_space::local_space);

        عائم انتقال = s_انتقال[0];
        عائم final_جابه_جایی = s_vals[tid] + s_warp_sums[warp] + انتقال - موقت[رقم_unroll - 1];

#رهنمود unroll
        لكل (صحيح j = 0; j < رقم_unroll; j++) {
            إذا (idx + j < ne00) {
                dst_ردیف[idx + j] = موقت[j] + final_جابه_جایی;
            }
        }

        قلم.barrier(sycl::access::fence_space::local_space);

        إذا (tid == 0) {
            s_انتقال[0] += s_انتقال[1];
        }
    }
}

مضمن فراغ ggml_sycl_op_cumsum(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    dpct::صف_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_مجموعه_device(ctx.device));

    ثابت عائم * src_d = static_cast<ثابت عائم *>(src0->داده);
    عائم       * dst_d = static_cast<عائم *>(dst->داده);

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

    ثابت صحيح رقم_warps = (ne00 + WARP_اندازه - 1) / WARP_اندازه;
    صحيح بلوک_اندازه_3 = رقم_warps * WARP_اندازه;
    بلوک_اندازه_3 = std::min(بلوک_اندازه_3, SYCL_CUMSUM_بلوک_اندازه);
    ثابت صحيح warps_در_هر_بلوک = بلوک_اندازه_3 / WARP_اندازه;
    ثابت صحيح smem_اندازه = بلوک_اندازه_3 + warps_در_هر_بلوک + 2;

    ثابت sycl::range<3> grid(ne03, ne02, ne01);
    ثابت sycl::range<3> بلوک(1, 1, بلوک_اندازه_3);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<عائم, 1> smem_acc(sycl::range<1>(smem_اندازه), cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(grid * بلوک, بلوک),
            [=](sycl::nd_قلم<3> قلم) [[sycl::reqd_sub_گروه_اندازه(WARP_اندازه)]] {
                cumsum_f32_kernel(src_d, dst_d, ne00, ne01, ne02, ne03,
                                  s01, s02, s03, d1, d2, d3,
                                  قلم, get_اشاره_گر(smem_acc));
            });
    });
}

فراغ ggml_sycl_cumsum(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_cumsum(ctx, dst);
}
