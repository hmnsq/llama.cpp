#شامل "cumsum.hpp"
#شامل "common.hpp"

#شامل <algorithm>

#تعریف_کرو SYCL_CUMSUM_بلاک_سائز 256

ساکن __dpct_inline__ اعشاری warp_سابقہ_inclusive_مجموع_f32(اعشاری x, ثابت sycl::nd_شے<3> & شے) {
    واپس sycl::inclusive_جائزہ_over_گروہ(شے.get_sub_گروہ(), x, sycl::plus<اعشاری>());
}

ساکن خالی cumsum_f32_kernel(
        ثابت اعشاری * __restrict__ src, اعشاری * __restrict__ dst,
        ثابت int64_t ne00, ثابت int64_t ne01, ثابت int64_t ne02, ثابت int64_t ne03,
        ثابت int64_t s01, ثابت int64_t s02, ثابت int64_t s03,
        ثابت int64_t  d1, ثابت int64_t  d2, ثابت int64_t  d3,
        ثابت sycl::nd_شے<3> & شے, اعشاری * smem) {

    ثابت صحیح tid = شے.get_local_id(2);
    ثابت صحیح بلاک_سائز_3 = شے.get_local_range(2);
    ثابت صحیح lane = tid % WARP_سائز;
    ثابت صحیح warp = tid / WARP_سائز;
    ثابت صحیح warps_فی_بلاک = بلاک_سائز_3 / WARP_سائز;

    اعشاری * s_vals      = smem;
    اعشاری * s_warp_sums = smem + بلاک_سائز_3;
    اعشاری * s_حاصل     = smem + بلاک_سائز_3 + warps_فی_بلاک;

    اگر (tid == 0) {
        s_حاصل[0] = 0.0f;
    }
    شے.barrier(sycl::access::fence_space::local_space);

    ثابت int64_t i3 = شے.get_گروہ(0);
    ثابت int64_t i2 = شے.get_گروہ(1);
    ثابت int64_t i1 = شے.get_گروہ(2);
    اگر (i3 >= ne03 || i2 >= ne02 || i1 >= ne01) {
        واپس;
    }

    ثابت اعشاری * src_صف = src + i1 * s01 + i2 * s02 + i3 * s03;
    اعشاری       * dst_صف = dst + i1 * d1  + i2 * d2  + i3 * d3;

    constexpr صحیح عدد_unroll = 4;
    اعشاری عارضی[عدد_unroll];

    برائے (int64_t i = 0; i < ne00; i += عدد_unroll * بلاک_سائز_3) {
        int64_t idx = i + tid * عدد_unroll;

        عارضی[0] = (idx < ne00 ? src_صف[idx] : 0.0f);
#ہدایت unroll
        برائے (صحیح j = 1; j < عدد_unroll; j++) {
            عارضی[j] = عارضی[j - 1];
            اگر (idx + j < ne00) {
                عارضی[j] += src_صف[idx + j];
            }
        }

        اعشاری val = (idx < ne00) ? عارضی[عدد_unroll - 1] : 0.0f;

        val = warp_سابقہ_inclusive_مجموع_f32(val, شے);
        s_vals[tid] = val;

        اگر (lane == WARP_سائز - 1) {
            s_warp_sums[warp] = val;
        }
        شے.barrier(sycl::access::fence_space::local_space);

        اگر (warp == 0) {
            اعشاری w = (tid < warps_فی_بلاک) ? s_warp_sums[tid] : 0.0f;
            اعشاری inc = warp_سابقہ_inclusive_مجموع_f32(w, شے);
            اگر (tid < warps_فی_بلاک) {
                s_warp_sums[tid] = inc - w;
            }
            اگر (tid == warps_فی_بلاک - 1) {
                s_حاصل[1] = inc;
            }
        }
        شے.barrier(sycl::access::fence_space::local_space);

        اعشاری حاصل = s_حاصل[0];
        اعشاری final_انحراف = s_vals[tid] + s_warp_sums[warp] + حاصل - عارضی[عدد_unroll - 1];

#ہدایت unroll
        برائے (صحیح j = 0; j < عدد_unroll; j++) {
            اگر (idx + j < ne00) {
                dst_صف[idx + j] = عارضی[j] + final_انحراف;
            }
        }

        شے.barrier(sycl::access::fence_space::local_space);

        اگر (tid == 0) {
            s_حاصل[0] += s_حاصل[1];
        }
    }
}

اندرونی خالی ggml_sycl_op_cumsum(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    dpct::قطار_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_سیٹ_device(ctx.device));

    ثابت اعشاری * src_d = static_cast<ثابت اعشاری *>(src0->ڈیٹا);
    اعشاری       * dst_d = static_cast<اعشاری *>(dst->ڈیٹا);

    ثابت int64_t ne00 = src0->ne[0];
    ثابت int64_t ne01 = src0->ne[1];
    ثابت int64_t ne02 = src0->ne[2];
    ثابت int64_t ne03 = src0->ne[3];

    ثابت size_t ts = حجم(اعشاری);
    ثابت int64_t s01 = src0->nb[1] / ts;
    ثابت int64_t s02 = src0->nb[2] / ts;
    ثابت int64_t s03 = src0->nb[3] / ts;
    ثابت int64_t d1  = dst->nb[1] / ts;
    ثابت int64_t d2  = dst->nb[2] / ts;
    ثابت int64_t d3  = dst->nb[3] / ts;

    ثابت صحیح عدد_warps = (ne00 + WARP_سائز - 1) / WARP_سائز;
    صحیح بلاک_سائز_3 = عدد_warps * WARP_سائز;
    بلاک_سائز_3 = std::min(بلاک_سائز_3, SYCL_CUMSUM_بلاک_سائز);
    ثابت صحیح warps_فی_بلاک = بلاک_سائز_3 / WARP_سائز;
    ثابت صحیح smem_سائز = بلاک_سائز_3 + warps_فی_بلاک + 2;

    ثابت sycl::range<3> grid(ne03, ne02, ne01);
    ثابت sycl::range<3> بلاک(1, 1, بلاک_سائز_3);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<اعشاری, 1> smem_acc(sycl::range<1>(smem_سائز), cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(grid * بلاک, بلاک),
            [=](sycl::nd_شے<3> شے) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                cumsum_f32_kernel(src_d, dst_d, ne00, ne01, ne02, ne03,
                                  s01, s02, s03, d1, d2, d3,
                                  شے, get_اشارہ_گر(smem_acc));
            });
    });
}

خالی ggml_sycl_cumsum(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_cumsum(ctx, dst);
}
