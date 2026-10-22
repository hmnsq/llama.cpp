#pragma once

#ضمن <sycl/sycl.hpp>
#ضمن <sycl/ext/intel/experimental/grf_size_properties.hpp>
#ضمن "dpct/helper.hpp"
#ضمن "common.hpp"
#ضمن "convert.hpp"
#ضمن "vecdotq.hpp"
#ضمن "fattn.hpp"

#ضمن "ggml.h"

#ضمن <cstdint>
#ضمن <cmath>
#ضمن <float.h>


#define FATTN_KQ_STRIDE       256
#define HALF_MAX_HALF         sycl::half(65504.0f/2) // Use neg. of this instead of -INFINITY to initialize KQ max vals to avoid NaN upon subtraction.
#define SOFTMAX_FTZ_THRESHOLD -20.0f                   // Softmax exp. of values smaller than this are flushed to zero to avoid NaNs.
#define FATTN_KQ_MAX_OFFSET (3.0f*0.6931f)

تعريفنوع فراغ (*fattn_kernel_t)(
    ثابت حرف* Q,
    ثابت حرف* K,
    ثابت حرف* V,
    ثابت حرف* mask,
    ثابت حرف* sinks,
    ثابت صحيح* KV_max,
    عائم* dst,
    sycl::float2* dst_meta,
    ثابت عائم scale,
    ثابت عائم max_bias,
    ثابت عائم m0,
    ثابت عائم m1,
    ثابت uint32_t n_head_log2,
    ثابت عائم logit_softcap,
    ثابت int32_t ne00,
    ثابت sycl::uint3 ne01,
    ثابت int32_t ne02,
    ثابت int32_t ne03,
    ثابت int32_t nb01,
    ثابت int32_t nb02,
    ثابت int32_t nb03,
    ثابت int32_t ne10,
    ثابت int32_t ne11,
    ثابت int32_t ne12,
    ثابت int32_t ne13,
    ثابت int32_t nb11,
    ثابت int32_t nb12,
    ثابت int64_t nb13,
    ثابت int32_t nb21,
    ثابت int32_t nb22,
    ثابت int64_t nb23,
    ثابت int32_t ne31,
    ثابت int32_t ne32,
    ثابت int32_t ne33,
    ثابت int32_t nb31,
    ثابت int32_t nb32,
    ثابت int64_t nb33);

تعريفنوع عائم (*vec_dot_KQ_t)(
    ثابت حرف * __restrict__ K_c, ثابت فراغ * __restrict__ Q_v, ثابت صحيح * __restrict__ Q_q8 , ثابت فراغ * __restrict__ Q_ds);

template <صحيح D, صحيح nthreads>
ساكن __dpct_inline__ عائم vec_dot_fattn_vec_KQ_f16(ثابت حرف * __restrict__ K_c,
                                                      ثابت فراغ * __restrict__ Q_v,
                                                      ثابت صحيح * __restrict__ Q_q8,
                                                      ثابت فراغ * __restrict__ Q_ds_v) {
    ثابت sycl::half2 * K_h2 = (ثابت sycl::half2 *) K_c;
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v);

    constexpr صحيح cpy_nb = ggml_sycl_get_max_cpy_bytes();
    constexpr صحيح cpy_ne = cpy_nb / 4;

    عائم sum = 0.0f;

#pragma unroll
    لكل (صحيح k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads*cpy_ne) {
        sycl::half2 tmp[cpy_ne];
        ggml_sycl_memcpy_1<حجم(tmp)>(
            tmp,
            K_h2 + k_KQ_0 + (sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2) % nthreads) * cpy_ne);
#pragma unroll
        لكل (صحيح k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
#ifdef GGML_SYCL_F16
            ggml_sycl_mad(sum,                tmp[k_KQ_1] , ((ثابت sycl::half2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#وإلا
            ggml_sycl_mad(sum, __half22float2(tmp[k_KQ_1]), ((ثابت sycl::float2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#endif // GGML_SYCL_F16
        }
    }

    أرجع sum;
}

template <صحيح D, صحيح nthreads, صحيح warp_size>
ساكن __dpct_inline__ عائم vec_dot_fattn_vec_KQ_q4_0(ثابت حرف * __restrict__ K_c,
                                                       ثابت فراغ * __restrict__ Q_v,
                                                       ثابت صحيح * __restrict__ Q_q8,
                                                       ثابت فراغ * __restrict__ Q_ds_v) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    ثابت block_q4_0 * K_q4_0   = (ثابت block_q4_0 *) K_c;
    GGML_UNUSED(Q_v);

    عائم sum = 0.0f;

#pragma unroll
    لكل (صحيح k_KQ_0 = 0; k_KQ_0 < صحيح(D/حجم(صحيح)); k_KQ_0 += nthreads) {
        ثابت صحيح k_KQ =
            k_KQ_0 + (nthreads == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads);

        ثابت صحيح ib    = k_KQ /  QI8_1;
        ثابت صحيح iqs4  = k_KQ %  QI4_0;
        ثابت صحيح shift = k_KQ & (QI8_1/2);

        صحيح v;
        ggml_sycl_memcpy_1<حجم(صحيح), 2>(&v, K_q4_0[ib].qs + حجم(صحيح)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;
        ثابت صحيح u = Q_q8[k_KQ_0/nthreads];

        ثابت صحيح sumi = ggml_sycl_dp4a(v, u, 0);

        ثابت sycl::float2 Q_ds = ((ثابت sycl::float2 *) Q_ds_v)[k_KQ_0 / nthreads];
        sum += __half2float(K_q4_0[ib].d) * (sumi*Q_ds.x() - (8/QI8_1)*Q_ds.y());
    }

    أرجع sum;
}

template <صحيح D, صحيح nthreads , صحيح warp_size>
ساكن __dpct_inline__ عائم vec_dot_fattn_vec_KQ_q4_1(ثابت حرف * __restrict__ K_c,
                                                       ثابت فراغ * __restrict__ Q_v,
                                                       ثابت صحيح * __restrict__ Q_q8,
                                                       ثابت فراغ * __restrict__ Q_ds_v) {
    auto               item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ثابت block_q4_1 * K_q4_1   = (ثابت block_q4_1 *) K_c;
    GGML_UNUSED(Q_v);

    عائم sum = 0.0f;

#pragma unroll
    لكل (صحيح k_KQ_0 = 0; k_KQ_0 < صحيح(D/حجم(صحيح)); k_KQ_0 += nthreads) {
        ثابت صحيح k_KQ =
            k_KQ_0 + (nthreads == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads);

        ثابت صحيح ib    = k_KQ /  QI8_1;
        ثابت صحيح iqs4  = k_KQ %  QI4_1;
        ثابت صحيح shift = k_KQ & (QI8_1/2);

        صحيح v;
        ggml_sycl_memcpy_1<حجم(صحيح)>(&v, K_q4_1[ib].qs + حجم(صحيح)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;
        ثابت صحيح u = Q_q8[k_KQ_0/nthreads];

        ثابت صحيح sumi = ggml_sycl_dp4a(v, u, 0);

        ثابت sycl::float2 K_dm = (K_q4_1[ib].dm).template convert<عائم, sycl::rounding_mode::automatic>();
        ثابت sycl::float2 Q_ds = ((ثابت sycl::float2 *) Q_ds_v)[k_KQ_0 / nthreads];

        sum += K_dm.x()*Q_ds.x()*sumi + K_dm.y()*Q_ds.y()/QI8_1;
    }

    أرجع sum;
}

template <صحيح D, صحيح nthreads, صحيح warp_size>
ساكن __dpct_inline__ عائم vec_dot_fattn_vec_KQ_q5_0(ثابت حرف * __restrict__ K_c,
                                                       ثابت فراغ * __restrict__ Q_v,
                                                       ثابت صحيح * __restrict__ Q_q8,
                                                       ثابت فراغ * __restrict__ Q_ds_v) {
    auto               item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ثابت block_q5_0 * K_q5_0   = (ثابت block_q5_0 *) K_c;
    GGML_UNUSED(Q_v);

    عائم sum = 0.0f;

#pragma unroll
    لكل (صحيح k_KQ_0 = 0; k_KQ_0 < صحيح(D/حجم(صحيح)); k_KQ_0 += nthreads) {
        ثابت صحيح k_KQ =
            k_KQ_0 + (nthreads == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads);

        ثابت صحيح ib    = k_KQ /  QI8_1;
        ثابت صحيح iqs4  = k_KQ %  QI5_0;
        ثابت صحيح iqs8  = k_KQ %  QI8_1;
        ثابت صحيح shift = k_KQ & (QI8_1/2);

        صحيح v;
        ggml_sycl_memcpy_1<حجم(صحيح), 2>(&v, K_q5_0[ib].qs + حجم(صحيح)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;

        {
            صحيح vh;
            ggml_sycl_memcpy_1<حجم(صحيح), 2>(&vh, K_q5_0[ib].qh);
            vh >>= iqs8 * QI5_0;

            v |= (vh <<  4) & 0x00000010; // 0 ->  4
            v |= (vh << 11) & 0x00001000; // 1 -> 12
            v |= (vh << 18) & 0x00100000; // 2 -> 20
            v |= (vh << 25) & 0x10000000; // 3 -> 28
        }

        ثابت صحيح u = Q_q8[k_KQ_0/nthreads];

        ثابت صحيح sumi = ggml_sycl_dp4a(v, u, 0);

        ثابت sycl::float2 Q_ds = ((ثابت sycl::float2 *) Q_ds_v)[k_KQ_0 / nthreads];

        sum += __half2float(K_q5_0[ib].d) * (sumi*Q_ds.x() - (16/QI8_1)*Q_ds.y());
    }

    أرجع sum;
}

template <صحيح D, صحيح nthreads, صحيح warp_size>
ساكن __dpct_inline__ عائم vec_dot_fattn_vec_KQ_q5_1(ثابت حرف * __restrict__ K_c,
                                                       ثابت فراغ * __restrict__ Q_v,
                                                       ثابت صحيح * __restrict__ Q_q8,
                                                       ثابت فراغ * __restrict__ Q_ds_v) {
    auto               item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ثابت block_q5_1 * K_q5_1   = (ثابت block_q5_1 *) K_c;
    GGML_UNUSED(Q_v);

    عائم sum = 0.0f;

#pragma unroll
    لكل (صحيح k_KQ_0 = 0; k_KQ_0 < صحيح(D/حجم(صحيح)); k_KQ_0 += nthreads) {
        ثابت صحيح k_KQ =
            k_KQ_0 + (nthreads == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads);

        ثابت صحيح ib    = k_KQ /  QI8_1;
        ثابت صحيح iqs4  = k_KQ %  QI5_1;
        ثابت صحيح iqs8  = k_KQ %  QI8_1;
        ثابت صحيح shift = k_KQ & (QI8_1/2);

        صحيح v;
        ggml_sycl_memcpy_1<حجم(صحيح)>(&v, K_q5_1[ib].qs + حجم(صحيح)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;

        {
            صحيح vh;
            ggml_sycl_memcpy_1<حجم(صحيح)>(&vh, K_q5_1[ib].qh);
            vh >>= iqs8 * QI5_0;

            v |= (vh <<  4) & 0x00000010; // 0 ->  4
            v |= (vh << 11) & 0x00001000; // 1 -> 12
            v |= (vh << 18) & 0x00100000; // 2 -> 20
            v |= (vh << 25) & 0x10000000; // 3 -> 28
        }

        ثابت صحيح u = Q_q8[k_KQ_0/nthreads];

        ثابت صحيح sumi = ggml_sycl_dp4a(v, u, 0);

        ثابت sycl::float2 K_dm = (K_q5_1[ib].dm).template convert<عائم, sycl::rounding_mode::automatic>();
        ثابت sycl::float2 Q_ds = ((ثابت sycl::float2 *) Q_ds_v)[k_KQ_0 / nthreads];

        sum += K_dm.x()*Q_ds.x()*sumi + K_dm.y()*Q_ds.y()/QI8_1;
    }

    أرجع sum;
}

template <صحيح D, صحيح nthreads, صحيح warp_size>
ساكن __dpct_inline__ عائم vec_dot_fattn_vec_KQ_q8_0(ثابت حرف * __restrict__ K_c,
                                                       ثابت فراغ * __restrict__ Q_v,
                                                       ثابت صحيح * __restrict__ Q_q8,
                                                       ثابت فراغ * __restrict__ Q_ds_v) {
    auto               item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ثابت block_q8_0 * K_q8_0   = (ثابت block_q8_0 *) K_c;
    GGML_UNUSED(Q_v);

    عائم sum = 0.0f;

#pragma unroll
    لكل (صحيح k_KQ_0 = 0; k_KQ_0 < صحيح(D/حجم(صحيح)); k_KQ_0 += nthreads) {
        ثابت صحيح k_KQ =
            k_KQ_0 + (nthreads == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads);

        ثابت صحيح ib  = k_KQ / QI8_0;
        ثابت صحيح iqs = k_KQ % QI8_0;

        صحيح v;
        ggml_sycl_memcpy_1<حجم(v), 2>(&v, K_q8_0[ib].qs + 4*iqs);

        ثابت sycl::float2 * Q_ds = (ثابت sycl::float2 *) Q_ds_v;
        ثابت عائم          Q_d  = Q_ds[k_KQ_0 / nthreads].x();

        sum += vec_dot_q8_0_q8_1_impl<عائم, 1>(&v, &Q_q8[k_KQ_0/nthreads], K_q8_0[ib].d, Q_d);
    }

    أرجع sum;
}

template <typename Tds, صحيح ni, صحيح warp_size>
ساكن __dpct_inline__ فراغ quantize_q8_1_to_shared(ثابت عائم * __restrict__ x,
                                                    ثابت عائم scale,
                                                    صحيح * __restrict__ yq32,
                                                    فراغ * __restrict__ yds) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    عائم vals[حجم(صحيح)] = { 0.0f };
#pragma unroll
    لكل (صحيح l = 0; l < صحيح(حجم(صحيح)); ++l) {
        vals[l] =
            (ni == warp_size || item_ct1.get_local_id(2) < ni) ? scale * x[4 * item_ct1.get_local_id(2) + l] : 0.0f;
    }

    عائم amax = sycl::fabs(vals[0]);
    عائم sum  = vals[0];
#pragma unroll
    لكل (صحيح l = 1; l < صحيح(حجم(صحيح)); ++l) {
        amax = sycl::fmax(amax, sycl::fabs(vals[l]));
        sum += vals[l];
    }
#pragma unroll
    لكل (صحيح mask = QI8_1/2; mask > 0; mask >>= 1) {
        amax = sycl::fmax(
            amax, dpct::permute_sub_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(), amax, mask));
        sum += dpct::permute_sub_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(), sum, mask);
    }

    ثابت عائم d = amax / 127;
    صحيح q32 = 0;
    int8_t * q8 = (int8_t *) &q32;

    إذا (d != 0.0f) {
#pragma unroll
        لكل (صحيح l = 0; l < صحيح(حجم(صحيح)); ++l) {
            q8[l] = sycl::round(vals[l] / d);
        }
    }

    yq32[item_ct1.get_local_id(2)] = q32;
    إذا (item_ct1.get_local_id(2) % QI8_1 == 0 && (ni == warp_size || item_ct1.get_local_id(2) < ni)) {
        إذا (std::is_same<Tds, sycl::half2>::value) {
            ((sycl::half2  *) yds)[item_ct1.get_local_id(2)/QI8_1] =  make_half2(d, sum);
        } وإلا {
            ((sycl::float2 *) yds)[item_ct1.get_local_id(2)/QI8_1] = make_float2(d, sum);
        }
    }
}

تعريفنوع فراغ (*dequantize_V_t)(ثابت فراغ *, فراغ *, ثابت int64_t);

template <typename T, صحيح ne>
ساكن __dpct_inline__ فراغ dequantize_V_f16(ثابت فراغ * __restrict__ vx, فراغ * __restrict__ dst, ثابت int64_t i0) {
    إذا constexpr (std::is_same_v<T, sycl::half>) {
        ggml_sycl_memcpy_1<ne * حجم(sycl::half)>(dst, (ثابت sycl::half *) vx + i0);
    } وإلا إذا constexpr (std::is_same_v<T, عائم>) {
        static_assert(ne % 2 == 0, "bad ne");
        sycl::half2 tmp[ne / 2];
        ggml_sycl_memcpy_1<ne * حجم(sycl::half)>(tmp, (ثابت sycl::half *) vx + i0);
        sycl::float2 * dst_f2 = (sycl::float2 *) dst;
#pragma unroll
        لكل (صحيح l = 0; l < ne/2; ++l) {
            dst_f2[l] = tmp[l].template convert<عائم, sycl::rounding_mode::automatic>();
        }
    } وإلا {
        static_assert(std::is_same_v<T, فراغ>, "unsupported type");
    }
}

template <typename T, صحيح ne>
ساكن __dpct_inline__ فراغ dequantize_V_q4_0(ثابت فراغ * __restrict__ vx, فراغ * __restrict__ dst, ثابت int64_t i0) {
    ثابت block_q4_0 * x = (ثابت block_q4_0 *) vx;

    ثابت int64_t ib    =  i0          /  QK4_0;
    ثابت صحيح     iqs   =  i0          % (QK4_0/2);
    ثابت صحيح     shift = (i0 % QK4_0) / (QK4_0/2);

    صحيح q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_sycl_memcpy_1<ne, 2>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;
    q = dpct::vectorized_دووانەیی<sycl::char4>(q, 0x08080808, dpct::sub_sat());

    ثابت int8_t * q8 = (ثابت int8_t *) &q;

#ifdef GGML_SYCL_F16
    إذا constexpr (std::is_same_v<T, sycl::half>) {
        ثابت sycl::half2 d = sycl::half2(x[ib].d);

#pragma unroll
        لكل (صحيح l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * sycl::half2(q8[l0 + 0], q8[l0 + 1]);
        }
    } وإلا
#endif // GGML_SYCL_F16
    إذا constexpr (std::is_same_v<T, عائم>) {
        ثابت عائم d = x[ib].d;

#pragma unroll
        لكل (صحيح l = 0; l < ne; ++l) {
            ((عائم *) dst)[l] = d * q8[l];
        }
    } وإلا {
        static_assert(std::is_same_v<T, فراغ>, "bad type");
    }
}

template <typename T, صحيح ne>
ساكن __dpct_inline__ فراغ dequantize_V_q4_1(ثابت فراغ * __restrict__ vx, فراغ * __restrict__ dst, ثابت int64_t i0) {
    ثابت block_q4_1 * x = (ثابت block_q4_1 *) vx;

    ثابت int64_t ib    =  i0          /  QK4_1;
    ثابت صحيح     iqs   =  i0          % (QK4_1/2);
    ثابت صحيح     shift = (i0 % QK4_1) / (QK4_1/2);

    صحيح q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_sycl_memcpy_1<ne>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    ثابت int8_t * q8 = (ثابت int8_t *) &q;

#ifdef GGML_SYCL_F16
    إذا constexpr (std::is_same_v<T, sycl::half>) {
        ثابت sycl::half2 dm = x[ib].dm;
        ثابت sycl::half2 d  = sycl::half2(dm[0]);
        ثابت sycl::half2 m  = sycl::half2(dm[1]);

#pragma unroll
        لكل (صحيح l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * sycl::half2(q8[l0 + 0], q8[l0 + 1]) + m;
        }
    } وإلا
#endif // GGML_SYCL_F16
    إذا constexpr (std::is_same_v<T, عائم>) {
        ثابت sycl::float2 dm = (x[ib].dm).template convert<عائم, sycl::rounding_mode::automatic>();

#pragma unroll
        لكل (صحيح l = 0; l < ne; ++l) {
            ((عائم *) dst)[l] = dm.x() * q8[l] + dm.y();
        }
    } وإلا {
        static_assert(std::is_same_v<T, فراغ>, "bad type");
    }
}

template <typename T, صحيح ne>
ساكن __dpct_inline__ فراغ dequantize_V_q5_0(ثابت فراغ * __restrict__ vx, فراغ * __restrict__ dst, ثابت int64_t i0) {
    ثابت block_q5_0 * x = (ثابت block_q5_0 *) vx;

    ثابت int64_t ib    =  i0          /  QK5_0;
    ثابت صحيح     idq   =  i0          %  QK5_0;
    ثابت صحيح     iqs   =  i0          % (QK5_0/2);
    ثابت صحيح     shift = (i0 % QK5_0) / (QK5_0/2);

    صحيح q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_sycl_memcpy_1<ne, 2>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    {
        صحيح qh;
        ggml_sycl_memcpy_1<ne, 2>(&qh, x[ib].qh);
#pragma unroll
        لكل (صحيح l = 0; l < ne; ++l) {
            q |= ((qh >> (idq + l)) & 0x00000001) << (8*l + 4);
        }
    }

    q = dpct::vectorized_دووانەیی<sycl::char4>(q, 0x10101010, dpct::sub_sat());

    ثابت int8_t * q8 = (ثابت int8_t *) &q;

#ifdef GGML_SYCL_F16
    إذا constexpr (std::is_same_v<T, sycl::half>) {
        ثابت sycl::half2 d = sycl::half2(x[ib].d);

#pragma unroll
        لكل (صحيح l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * sycl::half2(q8[l0 + 0], q8[l0 + 1]);
        }
    } وإلا
#endif // GGML_SYCL_F16
    إذا constexpr (std::is_same_v<T, عائم>) {
        ثابت عائم d = x[ib].d;

#pragma unroll
        لكل (صحيح l = 0; l < ne; ++l) {
            ((عائم *) dst)[l] = d * q8[l];
        }
    } وإلا {
        static_assert(std::is_same_v<T, فراغ>, "bad type");
    }
}

template <typename T, صحيح ne>
ساكن __dpct_inline__ فراغ dequantize_V_q5_1(ثابت فراغ * __restrict__ vx, فراغ * __restrict__ dst, ثابت int64_t i0) {
    ثابت block_q5_1 * x = (ثابت block_q5_1 *) vx;

    ثابت int64_t ib    =  i0          /  QK5_1;
    ثابت صحيح     idq   =  i0          %  QK5_1;
    ثابت صحيح     iqs   =  i0          % (QK5_1/2);
    ثابت صحيح     shift = (i0 % QK5_1) / (QK5_1/2);

    صحيح q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_sycl_memcpy_1<ne>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    {
        صحيح qh;
        ggml_sycl_memcpy_1<ne>(&qh, x[ib].qh);
#pragma unroll
        لكل (صحيح l = 0; l < ne; ++l) {
            q |= ((qh >> (idq + l)) & 0x00000001) << (8*l + 4);
        }
    }

    ثابت int8_t * q8 = (ثابت int8_t *) &q;

#ifdef GGML_SYCL_F16
    إذا constexpr (std::is_same_v<T, sycl::half>) {
        ثابت sycl::half2 dm = x[ib].dm;
        ثابت sycl::half2 d  = sycl::half2(dm[0]);
        ثابت sycl::half2 m  = sycl::half2(dm[1]);

#pragma unroll
        لكل (صحيح l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * sycl::half2(q8[l0 + 0], q8[l0 + 1]) + m;
        }
    } وإلا
#endif // GGML_SYCL_F16
    إذا constexpr (std::is_same_v<T, عائم>) {
        ثابت sycl::float2 dm = (x[ib].dm).template convert<عائم, sycl::rounding_mode::automatic>();

#pragma unroll
        لكل (صحيح l = 0; l < ne; ++l) {
            ((عائم *) dst)[l] = dm.x() * q8[l] + dm.y();
        }
    } وإلا {
        static_assert(std::is_same_v<T, فراغ>, "bad type");
    }
}

template <typename T, صحيح ne>
ساكن __dpct_inline__ فراغ dequantize_V_q8_0(ثابت فراغ * __restrict__ vx, فراغ * __restrict__ dst, ثابت int64_t i0) {
    ثابت block_q8_0 * x = (ثابت block_q8_0 *) vx;

    ثابت int64_t ib  = i0 / QK8_0;
    ثابت صحيح     iqs = i0 % QK8_0;

    static_assert(ne % 2 == 0, "bad ne");
    int8_t qs[ne];
    ggml_sycl_memcpy_1<ne, 2>(qs, x[ib].qs + iqs);

#ifdef GGML_SYCL_F16
    إذا constexpr (std::is_same<T, sycl::half>::value) {
        ثابت sycl::half2 d = sycl::half2(x[ib].d);

#pragma unroll
        لكل (صحيح l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * make_half2(qs[l0 + 0], qs[l0 + 1]);
        }
    } وإلا
#endif // GGML_SYCL_F16
    إذا constexpr (std::is_same<T, عائم>::value) {
        ثابت عائم d = x[ib].d;

#pragma unroll
        لكل (صحيح l = 0; l < ne; ++l) {
            ((عائم *) dst)[l] = d * qs[l];
        }
    } وإلا {
        static_assert(std::is_same_v<T, فراغ>, "unsupported type");
    }
}

template <صحيح type_K, صحيح D, صحيح nthreads, صحيح warp_size>
constexpr vec_dot_KQ_t get_vec_dot_KQ() {
    إذا constexpr (type_K == GGML_TYPE_F16) {
        أرجع vec_dot_fattn_vec_KQ_f16<D, nthreads>;
    } وإلا إذا constexpr (type_K == GGML_TYPE_Q4_0) {
        أرجع vec_dot_fattn_vec_KQ_q4_0<D, nthreads, warp_size>;
    } وإلا إذا constexpr (type_K == GGML_TYPE_Q4_1) {
        أرجع vec_dot_fattn_vec_KQ_q4_1<D, nthreads, warp_size>;
    } وإلا إذا constexpr (type_K == GGML_TYPE_Q5_0) {
        أرجع vec_dot_fattn_vec_KQ_q5_0<D, nthreads, warp_size>;
    } وإلا إذا constexpr (type_K == GGML_TYPE_Q5_1) {
        أرجع vec_dot_fattn_vec_KQ_q5_1<D, nthreads, warp_size>;
    } وإلا إذا constexpr (type_K == GGML_TYPE_Q8_0) {
        أرجع vec_dot_fattn_vec_KQ_q8_0<D, nthreads, warp_size>;
    } وإلا {
        static_assert(type_K == -1, "bad type");
        أرجع nullptr;
    }
}

template <صحيح type_V, typename T, صحيح ne>
constexpr dequantize_V_t get_dequantize_V() {
    إذا constexpr (type_V == GGML_TYPE_F16) {
        أرجع dequantize_V_f16<T, ne>;
    } وإلا إذا constexpr (type_V == GGML_TYPE_Q4_0) {
        أرجع dequantize_V_q4_0<T, ne>;
    } وإلا إذا constexpr (type_V == GGML_TYPE_Q4_1) {
        أرجع dequantize_V_q4_1<T, ne>;
    } وإلا إذا constexpr (type_V == GGML_TYPE_Q5_0) {
        أرجع dequantize_V_q5_0<T, ne>;
    } وإلا إذا constexpr (type_V == GGML_TYPE_Q5_1) {
        أرجع dequantize_V_q5_1<T, ne>;
    } وإلا إذا constexpr (type_V == GGML_TYPE_Q8_0) {
        أرجع dequantize_V_q8_0<T, ne>;
    } وإلا {
        static_assert(type_V == -1, "bad type");
        أرجع nullptr;
    }
}

template <صحيح ncols1, صحيح warp_size>
ساكن فراغ flash_attn_mask_to_KV_max(ثابت sycl::half2 * __restrict__ mask,
                                      صحيح * __restrict__ KV_max,
                                      ثابت صحيح ne30,
                                      ثابت صحيح s31,
                                      ثابت صحيح s33,
                                      صحيح *     buf_iw) {
    auto      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ثابت صحيح ne31     = item_ct1.get_group_range(2);
    ثابت صحيح tid      = item_ct1.get_local_id(2);
    ثابت صحيح sequence = item_ct1.get_group(1);
    ثابت صحيح jt       = item_ct1.get_group(2);

    mask += sequence*s33 + jt*ncols1*s31;

    إذا (tid < warp_size) {
        buf_iw[tid] = 1;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);

    صحيح KV_max_sj = (ne30 - 1) * FATTN_KQ_STRIDE;
    لكل (; KV_max_sj >= 0; KV_max_sj -= FATTN_KQ_STRIDE) {
        صحيح all_inf = 1;

#pragma unroll
        لكل (صحيح j = 0; j < ncols1; ++j) {
            ثابت sycl::float2 tmp =
                mask[j * s31 + KV_max_sj / 2 + tid].template convert<عائم, sycl::rounding_mode::automatic>();
            all_inf = all_inf && صحيح(sycl::isinf((عائم) (tmp.x()))) && صحيح(sycl::isinf((عائم) (tmp.y())));
        }

        all_inf = warp_reduce_all<warp_size>(all_inf);
        إذا (tid % warp_size == 0) {
            buf_iw[tid / warp_size] = all_inf;
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);
        all_inf = buf_iw[tid % warp_size];
        item_ct1.barrier(sycl::access::fence_space::local_space);
        all_inf = warp_reduce_all<warp_size>(all_inf);

        إذا (!all_inf) {
            اكسر;
        }
    }

    // If the break in the loop was not triggered, KV_max_sj is now -FATTN_KQ_STRIDE.
    // If the break was triggered it's the lower edge of the tile with the first non-masked values.
    // In either case, walk back the decrementation by FATTN_KQ_STRIDE.
    KV_max_sj += FATTN_KQ_STRIDE;

    إذا (item_ct1.get_local_id(2) != 0) {
        أرجع;
    }

    KV_max[sequence*ne31 + jt] = KV_max_sj;
}

template <صحيح D, صحيح ncols1, صحيح ncols2>  // D == head size

ساكن فراغ flash_attn_stream_k_fixup(عائم * __restrict__ dst,
                                      ثابت sycl::float2 * __restrict__ dst_fixup,
                                      ثابت صحيح ne01,
                                      ثابت صحيح ne02,
                                      ثابت صحيح ne03,
                                      ثابت صحيح ne11,
                                      ثابت صحيح ne12,
                                      ثابت صحيح nbatch_fa) {
    auto          item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr صحيح ncols    = ncols1 * ncols2;

    ثابت صحيح bidx0 = item_ct1.get_group(2);
    ثابت صحيح j     = item_ct1.get_group(1);
    ثابت صحيح c     = item_ct1.get_group(0);
    ثابت صحيح jc    = j*ncols2 + c;
    ثابت صحيح tid   = item_ct1.get_local_id(2);

    ثابت عائم * dst_fixup_data = ((ثابت عائم *) dst_fixup) + item_ct1.get_group_range(2) * (2 * 2 * ncols);

    ثابت صحيح gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.

    ثابت صحيح iter_k     = (ne11      + (nbatch_fa - 1)) / nbatch_fa;
    ثابت صحيح iter_j     = (ne01      + (ncols1    - 1)) / ncols1;
    ثابت صحيح iter_z_gqa = (gqa_ratio + (ncols2    - 1)) / ncols2;

    ثابت صحيح kbc0 = int64_t(bidx0 + 0) * (iter_k * iter_j * iter_z_gqa * ne12 * ne03) / item_ct1.get_group_range(2);
    ثابت صحيح kbc0_stop =
        int64_t(bidx0 + 1) * (iter_k * iter_j * iter_z_gqa * ne12 * ne03) / item_ct1.get_group_range(2);

    ثابت منطقي did_not_have_any_data   = kbc0 == kbc0_stop;
    ثابت منطقي wrote_beginning_of_tile = kbc0 % iter_k == 0;
    ثابت منطقي did_not_write_last      = kbc0/iter_k == kbc0_stop/iter_k && kbc0_stop % iter_k != 0;
    إذا (did_not_have_any_data || wrote_beginning_of_tile || did_not_write_last) {
        أرجع;
    }

    // z_KV == K/V head index, zt_gqa = Q head start index per K/V head, jt = token position start index
    ثابت صحيح sequence =  kbc0 /(iter_k*iter_j*iter_z_gqa*ne12);
    ثابت صحيح z_KV     = (kbc0 - iter_k*iter_j*iter_z_gqa*ne12 * sequence)/(iter_k*iter_j*iter_z_gqa);
    ثابت صحيح zt_gqa   = (kbc0 - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV)/(iter_k*iter_j);
    ثابت صحيح jt       = (kbc0 - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV - iter_k*iter_j * zt_gqa) / iter_k;

    ثابت صحيح zt_Q = z_KV*gqa_ratio + zt_gqa*ncols2; // Global Q head start index.

    إذا (jt*ncols1 + j >= ne01 || zt_gqa*ncols2 + c >= gqa_ratio) {
        أرجع;
    }

    dst += sequence*ne02*ne01*D + jt*ne02*(ncols1*D) + zt_Q*D + (j*ne02 + c)*D + tid;

    // Load the partial result that needs a fixup:
    عائم dst_val = 0.0f;
    عائم max_val = 0.0f;
    عائم rowsum  = 0.0f;
    {
        dst_val = *dst;

        ثابت sycl::float2 tmp = dst_fixup[bidx0 * ncols + jc];
        max_val                = tmp.x();
        rowsum                 = tmp.y();
    }

    // Iterate over previous blocks and compute the combined results.
    // All SYCL blocks that get here must have a previous block that needs a fixup.
    صحيح bidx = bidx0 - 1;
    صحيح kbc_stop = kbc0;
    طالما(true) {
        ثابت صحيح kbc = int64_t(bidx) * (iter_k * iter_j * iter_z_gqa * ne12 * ne03) / item_ct1.get_group_range(2);
        إذا (kbc == kbc_stop) { // Did not have any data.
            bidx--;
            kbc_stop = kbc;
            استمر;
        }

        ثابت عائم dst_add = dst_fixup_data[bidx*ncols*D + jc*D + tid];

        ثابت sycl::float2 tmp = dst_fixup[(item_ct1.get_group_range(2) + bidx) * ncols + jc];

        // Scale the current and new value accumulators depending on the max. values.
        ثابت عائم max_val_new = sycl::fmax(max_val, tmp.x());

        ثابت عائم diff_val = max_val - max_val_new;
        ثابت عائم diff_add = tmp.x() - max_val_new;

        ثابت عائم scale_val = diff_val >= SOFTMAX_FTZ_THRESHOLD ? sycl::native::exp(diff_val) : 0.0f;
        ثابت عائم scale_add = diff_add >= SOFTMAX_FTZ_THRESHOLD ? sycl::native::exp(diff_add) : 0.0f;

        dst_val = scale_val*dst_val + scale_add*dst_add;
        rowsum  = scale_val * rowsum + scale_add * tmp.y();

        max_val = max_val_new;

        // If this block started in a previous tile we are done and don't need to combine additional partial results.
        إذا (kbc % iter_k == 0 || kbc/iter_k < kbc0/iter_k) {
            اكسر;
        }
        bidx--;
        kbc_stop = kbc;
    }

    // Write back final result:
    *dst = dst_val / rowsum;
}

template <صحيح D>  // D == head size

ساكن فراغ flash_attn_combine_results(ثابت عائم * __restrict__ VKQ_parts,
                                       ثابت sycl::float2 * __restrict__ VKQ_meta,
                                       عائم * __restrict__ dst,
                                       ثابت صحيح parallel_blocks,
                                       uint8_t * dpct_local) {
    // Dimension 0: threadIdx.x
    // Dimension 1: blockIdx.x
    // Dimension 2: blockIdx.y
    // Dimension 3: blockIdx.z
    // Memory layout is permuted with [0, 2, 1, 3]

    auto      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ثابت صحيح ne01     = item_ct1.get_group_range(2);
    ثابت صحيح ne02     = item_ct1.get_group_range(1);

    ثابت صحيح col      = item_ct1.get_group(2);
    ثابت صحيح head     = item_ct1.get_group(1);
    ثابت صحيح sequence = item_ct1.get_group(0);

    ثابت صحيح j_dst_unrolled = (sequence*ne01 + col)*ne02 + head;

    VKQ_parts += j_dst_unrolled * parallel_blocks*D;
    VKQ_meta  += j_dst_unrolled * parallel_blocks;
    dst       += j_dst_unrolled *                 D;

    ثابت صحيح tid = item_ct1.get_local_id(2);
    __builtin_assume(tid < D);

    auto meta = (sycl::float2 *) dpct_local;
    لكل (صحيح i = tid; i < 2*parallel_blocks; i += D) {
        ((عائم *) meta)[i] = ((ثابت عائم *)VKQ_meta) [i];
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    عائم kqmax = meta[0].x();
    لكل (صحيح l = 1; l < parallel_blocks; ++l) {
        kqmax = sycl::max(kqmax, meta[l].x());
    }

    عائم VKQ_numerator   = 0.0f;
    عائم VKQ_denominator = 0.0f;
    لكل (صحيح l = 0; l < parallel_blocks; ++l) {
        ثابت عائم KQ_max_scale = sycl::native::exp(meta[l].x() - kqmax);

        VKQ_numerator   += KQ_max_scale * VKQ_parts[l*D + tid];
        VKQ_denominator += KQ_max_scale * meta[l].y();
    }

    dst[tid] = VKQ_numerator / VKQ_denominator;
}

// Carries the sub-group size (and grf_size_automatic when use_large_grf) as
// kernel properties via get(properties_tag), in place of the
// reqd_sub_group_size attribute and the deprecated parallel_for overload.
template <typename KernelFunc, uint32_t Size, منطقي UseGrf>
بنية props_kernel {
    KernelFunc kernel;

    فراغ operator()(sycl::nd_item<3> item) ثابت {
        kernel(item);
    }

    auto get(sycl::ext::oneapi::experimental::properties_tag) ثابت {
        إذا constexpr (UseGrf) {
            أرجع sycl::ext::oneapi::experimental::properties{
                sycl::ext::intel::experimental::grf_size_automatic,
                sycl::ext::oneapi::experimental::sub_group_size<Size> };
        } وإلا {
            أرجع sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::sub_group_size<Size> };
        }
    }
};

template <fattn_kernel_t fattn_kernel, صحيح warp_size, منطقي use_large_grf = خطأ>
ساكن فراغ lauch_kernel(
    dpct::dim3 group_range,
    dpct::dim3 local_range,
    queue_ptr q,
    unsigned صحيح local_mem_size,
    ثابت حرف* __restrict__ Q,
    ثابت حرف* __restrict__ K,
    ثابت حرف* __restrict__ V,
    ثابت حرف* __restrict__ mask,
    ثابت حرف* __restrict__ sinks,
    ثابت صحيح* __restrict__ KV_max,
    عائم* __restrict__ dst,
    sycl::float2* __restrict__ dst_meta,
    ثابت عائم scale,
    ثابت عائم max_bias,
    ثابت عائم m0,
    ثابت عائم m1,
    ثابت uint32_t n_head_log2,
    ثابت عائم logit_softcap,
    ثابت int32_t ne00,
    ثابت sycl::uint3 ne01,
    ثابت int32_t ne02,
    ثابت int32_t ne03,
    ثابت int32_t nb01,
    ثابت int32_t nb02,
    ثابت int32_t nb03,
    ثابت int32_t ne10,
    ثابت int32_t ne11,
    ثابت int32_t ne12,
    ثابت int32_t ne13,
    ثابت int32_t nb11,
    ثابت int32_t nb12,
    ثابت int64_t nb13,
    ثابت int32_t nb21,
    ثابت int32_t nb22,
    ثابت int64_t nb23,
    ثابت int32_t ne31,
    ثابت int32_t ne32,
    ثابت int32_t ne33,
    ثابت int32_t nb31,
    ثابت int32_t nb32,
    ثابت int64_t nb33) {
    GGML_UNUSED(local_mem_size);

    ثابت auto rng = sycl::nd_range<3>(
        static_cast<sycl::range<3>>(group_range * local_range),
        static_cast<sycl::range<3>>(local_range));

    ثابت auto kernel = [=](sycl::nd_item<3> item_ct1) {
        GGML_UNUSED(item_ct1);
        fattn_kernel(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
                     max_bias, m0, m1, n_head_log2, logit_softcap, ne00,
                     ne01, ne02, ne03, nb01, nb02, nb03, ne10, ne11,
                     ne12, ne13, nb11, nb12, nb13, nb21, nb22, nb23,
                     ne31, ne32, ne33, nb31, nb32, nb33);
    };

    q->submit([&](sycl::handler &cgh) {
        // grf_size_automatic lets the compiler use the large register file when the kernel needs it. grf_size<256> would say the same but is undefined on devices that do not have it.
        cgh.parallel_for(rng, props_kernel<decltype(kernel), warp_size, use_large_grf>{kernel});
    });
}

// use_large_grf lets the compiler choose the large register file. Opt in where spilling limits the kernel, not thread occupancy.
template <صحيح DV, صحيح ncols1, صحيح ncols2, fattn_kernel_t fattn_kernel, صحيح warp_size, منطقي use_large_grf = خطأ>
فراغ launch_fattn(
    ggml_backend_sycl_context & ctx, ggml_tensor * dst, ثابت صحيح nwarps, ثابت size_t nbytes_shared,
    ثابت صحيح nbatch_fa, ثابت منطقي need_f16_K, ثابت منطقي need_f16_V, ثابت منطقي stream_k) {

    constexpr صحيح ncols = ncols1 * ncols2;

    ثابت ggml_tensor * Q = dst->src[0];
    ثابت ggml_tensor * K = dst->src[1];
    ثابت ggml_tensor * V = dst->src[2];

    ثابت منطقي V_is_K_view = V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));

    ثابت ggml_tensor * mask  = dst->src[3];
    ثابت ggml_tensor * sinks = dst->src[4];

    ggml_tensor * KQV = dst;

    GGML_ASSERT(Q->type == GGML_TYPE_F32);
    GGML_ASSERT(KQV->type == GGML_TYPE_F32);

    GGML_ASSERT(Q->nb[0] == ggml_element_size(Q));
    GGML_ASSERT(K->nb[0] == ggml_element_size(K));
    GGML_ASSERT(V->nb[0] == ggml_element_size(V));

    GGML_ASSERT(!mask || mask->type == GGML_TYPE_F16);

    ggml_sycl_pool & pool = ctx.pool();
    dpct::queue_ptr  main_stream = ctx.stream();
    ثابت صحيح id  = ggml_sycl_get_device();
    ثابت صحيح nsm = ggml_sycl_info().devices[id].nsm;

    ثابت ggml_sycl_fattn_extra  extra = ggml_sycl_fattn_get_extra(dst);
    ggml_sycl_pool_alloc<صحيح>    KV_max(pool);
    ggml_sycl_pool_alloc<عائم>  dst_tmp(pool);
    ggml_sycl_pool_alloc<sycl::float2> dst_tmp_meta(pool);

    ثابت حرف * K_data = (ثابت حرف *) K->data;
    size_t nb11 = K->nb[1];
    size_t nb12 = K->nb[2];
    size_t nb13 = K->nb[3];

    ثابت حرف * V_data = (ثابت حرف *) V->data;
    size_t nb21 = V->nb[1];
    size_t nb22 = V->nb[2];
    size_t nb23 = V->nb[3];

    إذا (need_f16_K && K->type != GGML_TYPE_F16) {
        ثابت size_t bs = ggml_blck_size(K->type);
        ثابت size_t ts = ggml_type_size(K->type);

        GGML_ASSERT(extra.K_buffer_ptr);
        sycl::half * K_f16_ptr = (sycl::half *) extra.K_buffer_ptr;
        إذا (ggml_is_contiguously_allocated(K)) {
            to_fp16_sycl_t to_fp16 = ggml_get_to_fp16_sycl(K->type, dst);
            to_fp16(K_data, K_f16_ptr, ggml_nelements(K), main_stream);

            nb11 = nb11 * bs * حجم(sycl::half) / ts;
            nb12 = nb12 * bs * حجم(sycl::half) / ts;
            nb13 = nb13 * bs * حجم(sycl::half) / ts;
        } وإلا {
            GGML_ASSERT(K->nb[0] == ts);
            to_fp16_nc_sycl_t to_fp16 = ggml_get_to_fp16_nc_sycl(K->type);
            ثابت int64_t s01 = nb11 / ts;
            ثابت int64_t s02 = nb12 / ts;
            ثابت int64_t s03 = nb13 / ts;
            to_fp16(K_data, K_f16_ptr, K->ne[0], K->ne[1], K->ne[2], K->ne[3], s01, s02, s03, main_stream);

            nb11 = K->ne[0] * حجم(sycl::half);
            nb12 = K->ne[1] * nb11;
            nb13 = K->ne[2] * nb12;
        }
        K_data = (حرف *) K_f16_ptr;
    }

    إذا (need_f16_V && V->type != GGML_TYPE_F16) {
        إذا (V_is_K_view) {
            V_data = K_data;
            nb21   = nb11;
            nb22   = nb12;
            nb23   = nb13;
        } وإلا {
            ثابت size_t bs = ggml_blck_size(V->type);
            ثابت size_t ts = ggml_type_size(V->type);

            GGML_ASSERT(extra.V_buffer_ptr);
            sycl::half * V_f16_ptr = (sycl::half *) extra.V_buffer_ptr;
            إذا (ggml_is_contiguously_allocated(V)) {
                to_fp16_sycl_t to_fp16 = ggml_get_to_fp16_sycl(V->type, dst);
                to_fp16(V_data, V_f16_ptr, ggml_nelements(V), main_stream);
                V_data = (حرف *) V_f16_ptr;

                nb21 = nb21 * bs * حجم(sycl::half) / ts;
                nb22 = nb22 * bs * حجم(sycl::half) / ts;
                nb23 = nb23 * bs * حجم(sycl::half) / ts;
            } وإلا {
                GGML_ASSERT(V->nb[0] == ts);
                to_fp16_nc_sycl_t to_fp16 = ggml_get_to_fp16_nc_sycl(V->type);
                ثابت int64_t s01 = nb21 / ts;
                ثابت int64_t s02 = nb22 / ts;
                ثابت int64_t s03 = nb23 / ts;
                to_fp16(V_data, V_f16_ptr, V->ne[0], V->ne[1], V->ne[2], V->ne[3], s01, s02, s03, main_stream);

                nb21 = V->ne[0] * حجم(sycl::half);
                nb22 = V->ne[1] * nb21;
                nb23 = V->ne[2] * nb22;
            }
            V_data = (حرف *) V_f16_ptr;
        }
    }

    ثابت صحيح ntiles_x     = ((Q->ne[1] + ncols1 - 1) / ncols1);
    ثابت صحيح gqa_ratio    = Q->ne[2] / K->ne[2];
    ثابت صحيح ntiles_z_gqa = ((gqa_ratio + ncols2 - 1) / ncols2);
    ثابت صحيح ntiles_total = ntiles_x * ntiles_z_gqa * K->ne[2] * Q->ne[3];

    // Optional optimization where the mask is scanned to determine whether part of the calculation can be skipped.
    // Only worth the overhead if there is at lease one FATTN_KQ_STRIDE x FATTN_KQ_STRIDE square to be skipped or
    //     multiple sequences of possibly different lengths.
    إذا (mask && K->ne[1] % FATTN_KQ_STRIDE == 0 && (Q->ne[1] >= 1024 || Q->ne[3] > 1)) {
        ثابت صحيح s31 = mask->nb[1] / حجم(sycl::half2);
        ثابت صحيح s33 = mask->nb[3] / حجم(sycl::half2);

        ثابت dpct::dim3 blocks_ژمارە_KV_max(ntiles_x, Q->ne[3], 1);
        ثابت dpct::dim3 block_dim_KV_max(FATTN_KQ_STRIDE / 2, 1, 1);

        ثابت صحيح ne_KV_max = blocks_ژمارە_KV_max.x*blocks_ژمارە_KV_max.y;
        ثابت صحيح iter_k = K->ne[1] / FATTN_KQ_STRIDE;

        KV_max.alloc(ne_KV_max);
        {
            dpct::has_capability_or_fail(main_stream->get_device(), { sycl::aspect::fp16 });

            main_stream->submit([&](sycl::handler & cgh) {
                sycl::local_accessor<صحيح, 1> buf_iw_acc_ct1(sycl::range<1>(warp_size), cgh);

                auto mask_data_ct0  = (ثابت sycl::half2 *) mask->data;
                auto KV_max_ptr_ct1 = KV_max.ptr;

                cgh.parallel_for(sycl::nd_range<3>(blocks_ژمارە_KV_max * block_dim_KV_max, block_dim_KV_max),
                                 [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(warp_size)]] {
                                     GGML_UNUSED(item_ct1);
                                     flash_attn_mask_to_KV_max<ncols1, warp_size>(
                                         mask_data_ct0, KV_max_ptr_ct1, iter_k, s31, s33,
                                         buf_iw_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                                 });
            });
        }
        SYCL_CHECK(0);
    }

    ثابت dpct::dim3 block_dim(warp_size, nwarps, 1);

    // Max. number of active blocks limited by occupancy.
    صحيح max_blocks_per_sm = ggml_sycl_info().devices[id].max_wg_per_cu;
    صحيح parallel_blocks = max_blocks_per_sm;
    dpct::dim3 blocks_ژمارە;
    إذا (stream_k) {
        // For short contexts it can be faster to have the SMs work on whole tiles because this lets us skip the fixup.
        ثابت صحيح max_blocks = max_blocks_per_sm*nsm;
        ثابت صحيح nblocks_stream_k = max_blocks;
        ثابت منطقي use_stream_k = true;

        blocks_ژمارە.x = use_stream_k ? nblocks_stream_k : ntiles_total;
        blocks_ژمارە.y = 1;
        blocks_ژمارە.z = 1;

        إذا (ntiles_total % blocks_ژمارە.x != 0) { // Fixup is only needed if the SMs work on fractional tiles.
            dst_tmp_meta.alloc((size_t(blocks_ژمارە.x) * ncols * (2 + DV/2)));
        }
    } وإلا {
        ثابت صحيح ntiles_KQ = (K->ne[1] + nbatch_fa - 1) / nbatch_fa; // Max. number of parallel blocks limited by tensor size.

        // parallel_blocks must not be larger than what the tensor size allows:
        parallel_blocks = std::min(parallel_blocks, ntiles_KQ);
        // todo fix the hard code change
        // parallel_blocks = ntiles_KQ;

        // If ntiles_total % blocks_per_wave != 0 then some efficiency is lost due to tail effects.
        // Test whether parallel_blocks can be set to a higher value for better efficiency.
        ثابت صحيح blocks_per_wave = nsm * max_blocks_per_sm;
        صحيح nwaves_best = 0;
        صحيح efficiency_percent_best = 0;
        لكل (صحيح parallel_blocks_test = parallel_blocks; parallel_blocks_test <= ntiles_KQ; ++parallel_blocks_test) {
            ثابت صحيح nblocks_total = ntiles_total * parallel_blocks_test;
            ثابت صحيح nwaves = (nblocks_total + blocks_per_wave - 1) / blocks_per_wave;
            ثابت صحيح efficiency_percent = 100 * nblocks_total / (nwaves*blocks_per_wave);

            // Stop trying configurations with more waves if we already have good efficiency to avoid excessive overhead.
            إذا (efficiency_percent_best >= 95 && nwaves > nwaves_best) {
                اكسر;
            }

            إذا (efficiency_percent > efficiency_percent_best) {
                nwaves_best = nwaves;
                efficiency_percent_best = efficiency_percent;
                parallel_blocks = parallel_blocks_test;
            }
        }

        blocks_ژمارە.x = ntiles_x;
        blocks_ژمارە.y = parallel_blocks;
        blocks_ژمارە.z = ntiles_z_gqa*K->ne[2]*Q->ne[3];

        إذا (parallel_blocks > 1) {
            dst_tmp.alloc(parallel_blocks*ggml_nelements(KQV));
            dst_tmp_meta.alloc(parallel_blocks*ggml_nrows(KQV));
        }
    }

    عائم scale         = 1.0f;
    عائم max_bias      = 0.0f;
    عائم logit_softcap = 0.0f;

    memcpy(&scale,         (ثابت عائم *) KQV->op_params + 0, حجم(عائم));
    memcpy(&max_bias,      (ثابت عائم *) KQV->op_params + 1, حجم(عائم));
    memcpy(&logit_softcap, (ثابت عائم *) KQV->op_params + 2, حجم(عائم));

    إذا (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    ثابت uint32_t n_head      = Q->ne[2];
    ثابت uint32_t n_head_log2 = 1u << uint32_t(floorf(log2f(عائم(n_head))));

    ثابت عائم m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    ثابت عائم m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    // TODO other tensor dimensions after removal of WMMA kernel:
    ثابت sycl::uint3 ne01 = init_fastdiv_بەهاکان(Q->ne[1]);

    GGML_ASSERT(block_dim.x % warp_size == 0);

    lauch_kernel<fattn_kernel, warp_size, use_large_grf>(
        blocks_ژمارە, block_dim, main_stream, (unsigned صحيح) nbytes_shared, (ثابت حرف *) Q->data, K_data, V_data,
        mask ? ((ثابت حرف *) mask->data) : nullptr, sinks ? ((ثابت حرف *) sinks->data) : nullptr, KV_max.ptr,
        !stream_k && parallel_blocks > 1 ? dst_tmp.ptr : (عائم *) KQV->data, (sycl::float2 *)dst_tmp_meta.ptr, scale, max_bias, m0, m1,
        n_head_log2, logit_softcap, Q->ne[0], ne01, Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3], K->ne[0],
        K->ne[1], K->ne[2], K->ne[3], nb11, nb12, nb13, nb21, nb22, nb23, mask ? mask->ne[1] : 0,
        mask ? mask->ne[2] : 0, mask ? mask->ne[3] : 0, mask ? mask->nb[1] : 0, mask ? mask->nb[2] : 0,
        mask ? mask->nb[3] : 0);
    SYCL_CHECK(0);

    إذا (stream_k) {
        إذا (ntiles_total % blocks_ژمارە.x != 0) { // Fixup is only needed if the SMs work on fractional tiles.
            ثابت dpct::dim3 block_dim_combine(DV, 1, 1);
            ثابت dpct::dim3 blocks_ژمارە_combine = { blocks_ژمارە.x, ncols1, ncols2 };

            main_stream->submit([&](sycl::handler & cgh) {
                auto KQV_data_ct0         = (عائم *) KQV->data;
                auto dst_tmp_meta_ptr_ct1 = dst_tmp_meta.ptr;
                auto Q_ne_ct2             = Q->ne[1];
                auto Q_ne_ct3             = Q->ne[2];
                auto Q_ne_ct4             = Q->ne[3];
                auto K_ne_ct5             = K->ne[1];
                auto K_ne_ct6             = K->ne[2];

                cgh.parallel_for(sycl::nd_range<3>(blocks_ژمارە_combine * block_dim_combine, block_dim_combine),
                                 [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(warp_size)]] {
                                     GGML_UNUSED(item_ct1);
                                     flash_attn_stream_k_fixup<DV, ncols1, ncols2>(KQV_data_ct0, dst_tmp_meta_ptr_ct1,
                                                                                   Q_ne_ct2, Q_ne_ct3, Q_ne_ct4,
                                                                                   K_ne_ct5, K_ne_ct6, nbatch_fa);
                                 });
            });
        }
    } وإلا إذا (parallel_blocks > 1) {
        ثابت dpct::dim3 block_dim_combine(DV, 1, 1);
        ثابت dpct::dim3 blocks_ژمارە_combine(Q->ne[1], Q->ne[2], Q->ne[3]);
        ثابت size_t     nbytes_shared_combine = parallel_blocks * حجم(sycl::float2);
        main_stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(sycl::range<1>(nbytes_shared_combine), cgh);

            auto dst_tmp_ptr_ct0      = dst_tmp.ptr;
            auto dst_tmp_meta_ptr_ct1 = dst_tmp_meta.ptr;
            auto KQV_data_ct2         = (عائم *) KQV->data;

            cgh.parallel_for(sycl::nd_range<3>(blocks_ژمارە_combine * block_dim_combine, block_dim_combine),
                             [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(warp_size)]] {
                                 GGML_UNUSED(item_ct1);
                                 flash_attn_combine_results<DV>(
                                     dst_tmp_ptr_ct0, dst_tmp_meta_ptr_ct1, KQV_data_ct2, parallel_blocks,
                                     dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                             });
        });
    }
    SYCL_CHECK(0);
}
