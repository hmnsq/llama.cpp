#pragma once

#அந்தர்பாவ <sycl/sycl.hpp>
#அந்தர்பாவ <sycl/ext/intel/experimental/grf_size_properties.hpp>
#அந்தர்பாவ "dpct/helper.hpp"
#அந்தர்பாவ "common.hpp"
#அந்தர்பாவ "convert.hpp"
#அந்தர்பாவ "vecdotq.hpp"
#அந்தர்பாவ "fattn.hpp"

#அந்தர்பாவ "ggml.h"

#அந்தர்பாவ <cstdint>
#அந்தர்பாவ <cmath>
#அந்தர்பாவ <float.h>


#define FATTN_KQ_STRIDE       256
#define HALF_MAX_HALF         sycl::half(65504.0f/2) // Use neg. of this instead of -INFINITY to initialize KQ max vals to avoid NaN upon subtraction.
#define SOFTMAX_FTZ_THRESHOLD -20.0f                   // Softmax exp. of values smaller than this are flushed to zero to avoid NaNs.
#define FATTN_KQ_MAX_OFFSET (3.0f*0.6931f)

வகைவரை வெறுமை (*fattn_kernel_t)(
    மாறிலி எழுத்து* Q,
    மாறிலி எழுத்து* K,
    மாறிலி எழுத்து* V,
    மாறிலி எழுத்து* mask,
    மாறிலி எழுத்து* sinks,
    மாறிலி முழு* KV_max,
    தசம* dst,
    sycl::float2* dst_meta,
    மாறிலி தசம scale,
    மாறிலி தசம max_bias,
    மாறிலி தசம m0,
    மாறிலி தசம m1,
    மாறிலி uint32_t n_head_log2,
    மாறிலி தசம logit_softcap,
    மாறிலி int32_t ne00,
    மாறிலி sycl::uint3 ne01,
    மாறிலி int32_t ne02,
    மாறிலி int32_t ne03,
    மாறிலி int32_t nb01,
    மாறிலி int32_t nb02,
    மாறிலி int32_t nb03,
    மாறிலி int32_t ne10,
    மாறிலி int32_t ne11,
    மாறிலி int32_t ne12,
    மாறிலி int32_t ne13,
    மாறிலி int32_t nb11,
    மாறிலி int32_t nb12,
    மாறிலி int64_t nb13,
    மாறிலி int32_t nb21,
    மாறிலி int32_t nb22,
    மாறிலி int64_t nb23,
    மாறிலி int32_t ne31,
    மாறிலி int32_t ne32,
    மாறிலி int32_t ne33,
    மாறிலி int32_t nb31,
    மாறிலி int32_t nb32,
    மாறிலி int64_t nb33);

வகைவரை தசம (*vec_dot_KQ_t)(
    மாறிலி எழுத்து * __restrict__ K_c, மாறிலி வெறுமை * __restrict__ Q_v, மாறிலி முழு * __restrict__ Q_q8 , மாறிலி வெறுமை * __restrict__ Q_ds);

template <முழு D, முழு nthreads>
நிலையான __dpct_inline__ தசம vec_dot_fattn_vec_KQ_f16(மாறிலி எழுத்து * __restrict__ K_c,
                                                      மாறிலி வெறுமை * __restrict__ Q_v,
                                                      மாறிலி முழு * __restrict__ Q_q8,
                                                      மாறிலி வெறுமை * __restrict__ Q_ds_v) {
    மாறிலி sycl::half2 * K_h2 = (மாறிலி sycl::half2 *) K_c;
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v);

    constexpr முழு cpy_nb = ggml_sycl_get_max_cpy_bytes();
    constexpr முழு cpy_ne = cpy_nb / 4;

    தசம sum = 0.0f;

#pragma unroll
    க்கு (முழு k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads*cpy_ne) {
        sycl::half2 tmp[cpy_ne];
        ggml_sycl_memcpy_1<அளவு(tmp)>(
            tmp,
            K_h2 + k_KQ_0 + (sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2) % nthreads) * cpy_ne);
#pragma unroll
        க்கு (முழு k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
#ifdef GGML_SYCL_F16
            ggml_sycl_mad(sum,                tmp[k_KQ_1] , ((மாறிலி sycl::half2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#இல்லையெனில்
            ggml_sycl_mad(sum, __half22float2(tmp[k_KQ_1]), ((மாறிலி sycl::float2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#endif // GGML_SYCL_F16
        }
    }

    திரும்பு sum;
}

template <முழு D, முழு nthreads, முழு warp_size>
நிலையான __dpct_inline__ தசம vec_dot_fattn_vec_KQ_q4_0(மாறிலி எழுத்து * __restrict__ K_c,
                                                       மாறிலி வெறுமை * __restrict__ Q_v,
                                                       மாறிலி முழு * __restrict__ Q_q8,
                                                       மாறிலி வெறுமை * __restrict__ Q_ds_v) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    மாறிலி block_q4_0 * K_q4_0   = (மாறிலி block_q4_0 *) K_c;
    GGML_UNUSED(Q_v);

    தசம sum = 0.0f;

#pragma unroll
    க்கு (முழு k_KQ_0 = 0; k_KQ_0 < முழு(D/அளவு(முழு)); k_KQ_0 += nthreads) {
        மாறிலி முழு k_KQ =
            k_KQ_0 + (nthreads == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads);

        மாறிலி முழு ib    = k_KQ /  QI8_1;
        மாறிலி முழு iqs4  = k_KQ %  QI4_0;
        மாறிலி முழு shift = k_KQ & (QI8_1/2);

        முழு v;
        ggml_sycl_memcpy_1<அளவு(முழு), 2>(&v, K_q4_0[ib].qs + அளவு(முழு)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;
        மாறிலி முழு u = Q_q8[k_KQ_0/nthreads];

        மாறிலி முழு sumi = ggml_sycl_dp4a(v, u, 0);

        மாறிலி sycl::float2 Q_ds = ((மாறிலி sycl::float2 *) Q_ds_v)[k_KQ_0 / nthreads];
        sum += __half2float(K_q4_0[ib].d) * (sumi*Q_ds.x() - (8/QI8_1)*Q_ds.y());
    }

    திரும்பு sum;
}

template <முழு D, முழு nthreads , முழு warp_size>
நிலையான __dpct_inline__ தசம vec_dot_fattn_vec_KQ_q4_1(மாறிலி எழுத்து * __restrict__ K_c,
                                                       மாறிலி வெறுமை * __restrict__ Q_v,
                                                       மாறிலி முழு * __restrict__ Q_q8,
                                                       மாறிலி வெறுமை * __restrict__ Q_ds_v) {
    auto               item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    மாறிலி block_q4_1 * K_q4_1   = (மாறிலி block_q4_1 *) K_c;
    GGML_UNUSED(Q_v);

    தசம sum = 0.0f;

#pragma unroll
    க்கு (முழு k_KQ_0 = 0; k_KQ_0 < முழு(D/அளவு(முழு)); k_KQ_0 += nthreads) {
        மாறிலி முழு k_KQ =
            k_KQ_0 + (nthreads == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads);

        மாறிலி முழு ib    = k_KQ /  QI8_1;
        மாறிலி முழு iqs4  = k_KQ %  QI4_1;
        மாறிலி முழு shift = k_KQ & (QI8_1/2);

        முழு v;
        ggml_sycl_memcpy_1<அளவு(முழு)>(&v, K_q4_1[ib].qs + அளவு(முழு)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;
        மாறிலி முழு u = Q_q8[k_KQ_0/nthreads];

        மாறிலி முழு sumi = ggml_sycl_dp4a(v, u, 0);

        மாறிலி sycl::float2 K_dm = (K_q4_1[ib].dm).template convert<தசம, sycl::rounding_mode::automatic>();
        மாறிலி sycl::float2 Q_ds = ((மாறிலி sycl::float2 *) Q_ds_v)[k_KQ_0 / nthreads];

        sum += K_dm.x()*Q_ds.x()*sumi + K_dm.y()*Q_ds.y()/QI8_1;
    }

    திரும்பு sum;
}

template <முழு D, முழு nthreads, முழு warp_size>
நிலையான __dpct_inline__ தசம vec_dot_fattn_vec_KQ_q5_0(மாறிலி எழுத்து * __restrict__ K_c,
                                                       மாறிலி வெறுமை * __restrict__ Q_v,
                                                       மாறிலி முழு * __restrict__ Q_q8,
                                                       மாறிலி வெறுமை * __restrict__ Q_ds_v) {
    auto               item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    மாறிலி block_q5_0 * K_q5_0   = (மாறிலி block_q5_0 *) K_c;
    GGML_UNUSED(Q_v);

    தசம sum = 0.0f;

#pragma unroll
    க்கு (முழு k_KQ_0 = 0; k_KQ_0 < முழு(D/அளவு(முழு)); k_KQ_0 += nthreads) {
        மாறிலி முழு k_KQ =
            k_KQ_0 + (nthreads == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads);

        மாறிலி முழு ib    = k_KQ /  QI8_1;
        மாறிலி முழு iqs4  = k_KQ %  QI5_0;
        மாறிலி முழு iqs8  = k_KQ %  QI8_1;
        மாறிலி முழு shift = k_KQ & (QI8_1/2);

        முழு v;
        ggml_sycl_memcpy_1<அளவு(முழு), 2>(&v, K_q5_0[ib].qs + அளவு(முழு)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;

        {
            முழு vh;
            ggml_sycl_memcpy_1<அளவு(முழு), 2>(&vh, K_q5_0[ib].qh);
            vh >>= iqs8 * QI5_0;

            v |= (vh <<  4) & 0x00000010; // 0 ->  4
            v |= (vh << 11) & 0x00001000; // 1 -> 12
            v |= (vh << 18) & 0x00100000; // 2 -> 20
            v |= (vh << 25) & 0x10000000; // 3 -> 28
        }

        மாறிலி முழு u = Q_q8[k_KQ_0/nthreads];

        மாறிலி முழு sumi = ggml_sycl_dp4a(v, u, 0);

        மாறிலி sycl::float2 Q_ds = ((மாறிலி sycl::float2 *) Q_ds_v)[k_KQ_0 / nthreads];

        sum += __half2float(K_q5_0[ib].d) * (sumi*Q_ds.x() - (16/QI8_1)*Q_ds.y());
    }

    திரும்பு sum;
}

template <முழு D, முழு nthreads, முழு warp_size>
நிலையான __dpct_inline__ தசம vec_dot_fattn_vec_KQ_q5_1(மாறிலி எழுத்து * __restrict__ K_c,
                                                       மாறிலி வெறுமை * __restrict__ Q_v,
                                                       மாறிலி முழு * __restrict__ Q_q8,
                                                       மாறிலி வெறுமை * __restrict__ Q_ds_v) {
    auto               item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    மாறிலி block_q5_1 * K_q5_1   = (மாறிலி block_q5_1 *) K_c;
    GGML_UNUSED(Q_v);

    தசம sum = 0.0f;

#pragma unroll
    க்கு (முழு k_KQ_0 = 0; k_KQ_0 < முழு(D/அளவு(முழு)); k_KQ_0 += nthreads) {
        மாறிலி முழு k_KQ =
            k_KQ_0 + (nthreads == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads);

        மாறிலி முழு ib    = k_KQ /  QI8_1;
        மாறிலி முழு iqs4  = k_KQ %  QI5_1;
        மாறிலி முழு iqs8  = k_KQ %  QI8_1;
        மாறிலி முழு shift = k_KQ & (QI8_1/2);

        முழு v;
        ggml_sycl_memcpy_1<அளவு(முழு)>(&v, K_q5_1[ib].qs + அளவு(முழு)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;

        {
            முழு vh;
            ggml_sycl_memcpy_1<அளவு(முழு)>(&vh, K_q5_1[ib].qh);
            vh >>= iqs8 * QI5_0;

            v |= (vh <<  4) & 0x00000010; // 0 ->  4
            v |= (vh << 11) & 0x00001000; // 1 -> 12
            v |= (vh << 18) & 0x00100000; // 2 -> 20
            v |= (vh << 25) & 0x10000000; // 3 -> 28
        }

        மாறிலி முழு u = Q_q8[k_KQ_0/nthreads];

        மாறிலி முழு sumi = ggml_sycl_dp4a(v, u, 0);

        மாறிலி sycl::float2 K_dm = (K_q5_1[ib].dm).template convert<தசம, sycl::rounding_mode::automatic>();
        மாறிலி sycl::float2 Q_ds = ((மாறிலி sycl::float2 *) Q_ds_v)[k_KQ_0 / nthreads];

        sum += K_dm.x()*Q_ds.x()*sumi + K_dm.y()*Q_ds.y()/QI8_1;
    }

    திரும்பு sum;
}

template <முழு D, முழு nthreads, முழு warp_size>
நிலையான __dpct_inline__ தசம vec_dot_fattn_vec_KQ_q8_0(மாறிலி எழுத்து * __restrict__ K_c,
                                                       மாறிலி வெறுமை * __restrict__ Q_v,
                                                       மாறிலி முழு * __restrict__ Q_q8,
                                                       மாறிலி வெறுமை * __restrict__ Q_ds_v) {
    auto               item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    மாறிலி block_q8_0 * K_q8_0   = (மாறிலி block_q8_0 *) K_c;
    GGML_UNUSED(Q_v);

    தசம sum = 0.0f;

#pragma unroll
    க்கு (முழு k_KQ_0 = 0; k_KQ_0 < முழு(D/அளவு(முழு)); k_KQ_0 += nthreads) {
        மாறிலி முழு k_KQ =
            k_KQ_0 + (nthreads == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads);

        மாறிலி முழு ib  = k_KQ / QI8_0;
        மாறிலி முழு iqs = k_KQ % QI8_0;

        முழு v;
        ggml_sycl_memcpy_1<அளவு(v), 2>(&v, K_q8_0[ib].qs + 4*iqs);

        மாறிலி sycl::float2 * Q_ds = (மாறிலி sycl::float2 *) Q_ds_v;
        மாறிலி தசம          Q_d  = Q_ds[k_KQ_0 / nthreads].x();

        sum += vec_dot_q8_0_q8_1_impl<தசம, 1>(&v, &Q_q8[k_KQ_0/nthreads], K_q8_0[ib].d, Q_d);
    }

    திரும்பு sum;
}

template <typename Tds, முழு ni, முழு warp_size>
நிலையான __dpct_inline__ வெறுமை quantize_q8_1_to_shared(மாறிலி தசம * __restrict__ x,
                                                    மாறிலி தசம scale,
                                                    முழு * __restrict__ yq32,
                                                    வெறுமை * __restrict__ yds) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    தசம vals[அளவு(முழு)] = { 0.0f };
#pragma unroll
    க்கு (முழு l = 0; l < முழு(அளவு(முழு)); ++l) {
        vals[l] =
            (ni == warp_size || item_ct1.get_local_id(2) < ni) ? scale * x[4 * item_ct1.get_local_id(2) + l] : 0.0f;
    }

    தசம amax = sycl::fabs(vals[0]);
    தசம sum  = vals[0];
#pragma unroll
    க்கு (முழு l = 1; l < முழு(அளவு(முழு)); ++l) {
        amax = sycl::fmax(amax, sycl::fabs(vals[l]));
        sum += vals[l];
    }
#pragma unroll
    க்கு (முழு mask = QI8_1/2; mask > 0; mask >>= 1) {
        amax = sycl::fmax(
            amax, dpct::permute_sub_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(), amax, mask));
        sum += dpct::permute_sub_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(), sum, mask);
    }

    மாறிலி தசம d = amax / 127;
    முழு q32 = 0;
    int8_t * q8 = (int8_t *) &q32;

    எனில் (d != 0.0f) {
#pragma unroll
        க்கு (முழு l = 0; l < முழு(அளவு(முழு)); ++l) {
            q8[l] = sycl::round(vals[l] / d);
        }
    }

    yq32[item_ct1.get_local_id(2)] = q32;
    எனில் (item_ct1.get_local_id(2) % QI8_1 == 0 && (ni == warp_size || item_ct1.get_local_id(2) < ni)) {
        எனில் (std::is_same<Tds, sycl::half2>::value) {
            ((sycl::half2  *) yds)[item_ct1.get_local_id(2)/QI8_1] =  make_half2(d, sum);
        } இல்லையெனில் {
            ((sycl::float2 *) yds)[item_ct1.get_local_id(2)/QI8_1] = make_float2(d, sum);
        }
    }
}

வகைவரை வெறுமை (*dequantize_V_t)(மாறிலி வெறுமை *, வெறுமை *, மாறிலி int64_t);

template <typename T, முழு ne>
நிலையான __dpct_inline__ வெறுமை dequantize_V_f16(மாறிலி வெறுமை * __restrict__ vx, வெறுமை * __restrict__ dst, மாறிலி int64_t i0) {
    எனில் constexpr (std::is_same_v<T, sycl::half>) {
        ggml_sycl_memcpy_1<ne * அளவு(sycl::half)>(dst, (மாறிலி sycl::half *) vx + i0);
    } இல்லையெனில் எனில் constexpr (std::is_same_v<T, தசம>) {
        static_assert(ne % 2 == 0, "bad ne");
        sycl::half2 tmp[ne / 2];
        ggml_sycl_memcpy_1<ne * அளவு(sycl::half)>(tmp, (மாறிலி sycl::half *) vx + i0);
        sycl::float2 * dst_f2 = (sycl::float2 *) dst;
#pragma unroll
        க்கு (முழு l = 0; l < ne/2; ++l) {
            dst_f2[l] = tmp[l].template convert<தசம, sycl::rounding_mode::automatic>();
        }
    } இல்லையெனில் {
        static_assert(std::is_same_v<T, வெறுமை>, "unsupported type");
    }
}

template <typename T, முழு ne>
நிலையான __dpct_inline__ வெறுமை dequantize_V_q4_0(மாறிலி வெறுமை * __restrict__ vx, வெறுமை * __restrict__ dst, மாறிலி int64_t i0) {
    மாறிலி block_q4_0 * x = (மாறிலி block_q4_0 *) vx;

    மாறிலி int64_t ib    =  i0          /  QK4_0;
    மாறிலி முழு     iqs   =  i0          % (QK4_0/2);
    மாறிலி முழு     shift = (i0 % QK4_0) / (QK4_0/2);

    முழு q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_sycl_memcpy_1<ne, 2>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;
    q = dpct::vectorized_இருமம்<sycl::char4>(q, 0x08080808, dpct::sub_sat());

    மாறிலி int8_t * q8 = (மாறிலி int8_t *) &q;

#ifdef GGML_SYCL_F16
    எனில் constexpr (std::is_same_v<T, sycl::half>) {
        மாறிலி sycl::half2 d = sycl::half2(x[ib].d);

#pragma unroll
        க்கு (முழு l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * sycl::half2(q8[l0 + 0], q8[l0 + 1]);
        }
    } இல்லையெனில்
#endif // GGML_SYCL_F16
    எனில் constexpr (std::is_same_v<T, தசம>) {
        மாறிலி தசம d = x[ib].d;

#pragma unroll
        க்கு (முழு l = 0; l < ne; ++l) {
            ((தசம *) dst)[l] = d * q8[l];
        }
    } இல்லையெனில் {
        static_assert(std::is_same_v<T, வெறுமை>, "bad type");
    }
}

template <typename T, முழு ne>
நிலையான __dpct_inline__ வெறுமை dequantize_V_q4_1(மாறிலி வெறுமை * __restrict__ vx, வெறுமை * __restrict__ dst, மாறிலி int64_t i0) {
    மாறிலி block_q4_1 * x = (மாறிலி block_q4_1 *) vx;

    மாறிலி int64_t ib    =  i0          /  QK4_1;
    மாறிலி முழு     iqs   =  i0          % (QK4_1/2);
    மாறிலி முழு     shift = (i0 % QK4_1) / (QK4_1/2);

    முழு q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_sycl_memcpy_1<ne>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    மாறிலி int8_t * q8 = (மாறிலி int8_t *) &q;

#ifdef GGML_SYCL_F16
    எனில் constexpr (std::is_same_v<T, sycl::half>) {
        மாறிலி sycl::half2 dm = x[ib].dm;
        மாறிலி sycl::half2 d  = sycl::half2(dm[0]);
        மாறிலி sycl::half2 m  = sycl::half2(dm[1]);

#pragma unroll
        க்கு (முழு l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * sycl::half2(q8[l0 + 0], q8[l0 + 1]) + m;
        }
    } இல்லையெனில்
#endif // GGML_SYCL_F16
    எனில் constexpr (std::is_same_v<T, தசம>) {
        மாறிலி sycl::float2 dm = (x[ib].dm).template convert<தசம, sycl::rounding_mode::automatic>();

#pragma unroll
        க்கு (முழு l = 0; l < ne; ++l) {
            ((தசம *) dst)[l] = dm.x() * q8[l] + dm.y();
        }
    } இல்லையெனில் {
        static_assert(std::is_same_v<T, வெறுமை>, "bad type");
    }
}

template <typename T, முழு ne>
நிலையான __dpct_inline__ வெறுமை dequantize_V_q5_0(மாறிலி வெறுமை * __restrict__ vx, வெறுமை * __restrict__ dst, மாறிலி int64_t i0) {
    மாறிலி block_q5_0 * x = (மாறிலி block_q5_0 *) vx;

    மாறிலி int64_t ib    =  i0          /  QK5_0;
    மாறிலி முழு     idq   =  i0          %  QK5_0;
    மாறிலி முழு     iqs   =  i0          % (QK5_0/2);
    மாறிலி முழு     shift = (i0 % QK5_0) / (QK5_0/2);

    முழு q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_sycl_memcpy_1<ne, 2>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    {
        முழு qh;
        ggml_sycl_memcpy_1<ne, 2>(&qh, x[ib].qh);
#pragma unroll
        க்கு (முழு l = 0; l < ne; ++l) {
            q |= ((qh >> (idq + l)) & 0x00000001) << (8*l + 4);
        }
    }

    q = dpct::vectorized_இருமம்<sycl::char4>(q, 0x10101010, dpct::sub_sat());

    மாறிலி int8_t * q8 = (மாறிலி int8_t *) &q;

#ifdef GGML_SYCL_F16
    எனில் constexpr (std::is_same_v<T, sycl::half>) {
        மாறிலி sycl::half2 d = sycl::half2(x[ib].d);

#pragma unroll
        க்கு (முழு l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * sycl::half2(q8[l0 + 0], q8[l0 + 1]);
        }
    } இல்லையெனில்
#endif // GGML_SYCL_F16
    எனில் constexpr (std::is_same_v<T, தசம>) {
        மாறிலி தசம d = x[ib].d;

#pragma unroll
        க்கு (முழு l = 0; l < ne; ++l) {
            ((தசம *) dst)[l] = d * q8[l];
        }
    } இல்லையெனில் {
        static_assert(std::is_same_v<T, வெறுமை>, "bad type");
    }
}

template <typename T, முழு ne>
நிலையான __dpct_inline__ வெறுமை dequantize_V_q5_1(மாறிலி வெறுமை * __restrict__ vx, வெறுமை * __restrict__ dst, மாறிலி int64_t i0) {
    மாறிலி block_q5_1 * x = (மாறிலி block_q5_1 *) vx;

    மாறிலி int64_t ib    =  i0          /  QK5_1;
    மாறிலி முழு     idq   =  i0          %  QK5_1;
    மாறிலி முழு     iqs   =  i0          % (QK5_1/2);
    மாறிலி முழு     shift = (i0 % QK5_1) / (QK5_1/2);

    முழு q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_sycl_memcpy_1<ne>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    {
        முழு qh;
        ggml_sycl_memcpy_1<ne>(&qh, x[ib].qh);
#pragma unroll
        க்கு (முழு l = 0; l < ne; ++l) {
            q |= ((qh >> (idq + l)) & 0x00000001) << (8*l + 4);
        }
    }

    மாறிலி int8_t * q8 = (மாறிலி int8_t *) &q;

#ifdef GGML_SYCL_F16
    எனில் constexpr (std::is_same_v<T, sycl::half>) {
        மாறிலி sycl::half2 dm = x[ib].dm;
        மாறிலி sycl::half2 d  = sycl::half2(dm[0]);
        மாறிலி sycl::half2 m  = sycl::half2(dm[1]);

#pragma unroll
        க்கு (முழு l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * sycl::half2(q8[l0 + 0], q8[l0 + 1]) + m;
        }
    } இல்லையெனில்
#endif // GGML_SYCL_F16
    எனில் constexpr (std::is_same_v<T, தசம>) {
        மாறிலி sycl::float2 dm = (x[ib].dm).template convert<தசம, sycl::rounding_mode::automatic>();

#pragma unroll
        க்கு (முழு l = 0; l < ne; ++l) {
            ((தசம *) dst)[l] = dm.x() * q8[l] + dm.y();
        }
    } இல்லையெனில் {
        static_assert(std::is_same_v<T, வெறுமை>, "bad type");
    }
}

template <typename T, முழு ne>
நிலையான __dpct_inline__ வெறுமை dequantize_V_q8_0(மாறிலி வெறுமை * __restrict__ vx, வெறுமை * __restrict__ dst, மாறிலி int64_t i0) {
    மாறிலி block_q8_0 * x = (மாறிலி block_q8_0 *) vx;

    மாறிலி int64_t ib  = i0 / QK8_0;
    மாறிலி முழு     iqs = i0 % QK8_0;

    static_assert(ne % 2 == 0, "bad ne");
    int8_t qs[ne];
    ggml_sycl_memcpy_1<ne, 2>(qs, x[ib].qs + iqs);

#ifdef GGML_SYCL_F16
    எனில் constexpr (std::is_same<T, sycl::half>::value) {
        மாறிலி sycl::half2 d = sycl::half2(x[ib].d);

#pragma unroll
        க்கு (முழு l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * make_half2(qs[l0 + 0], qs[l0 + 1]);
        }
    } இல்லையெனில்
#endif // GGML_SYCL_F16
    எனில் constexpr (std::is_same<T, தசம>::value) {
        மாறிலி தசம d = x[ib].d;

#pragma unroll
        க்கு (முழு l = 0; l < ne; ++l) {
            ((தசம *) dst)[l] = d * qs[l];
        }
    } இல்லையெனில் {
        static_assert(std::is_same_v<T, வெறுமை>, "unsupported type");
    }
}

template <முழு type_K, முழு D, முழு nthreads, முழு warp_size>
constexpr vec_dot_KQ_t get_vec_dot_KQ() {
    எனில் constexpr (type_K == GGML_TYPE_F16) {
        திரும்பு vec_dot_fattn_vec_KQ_f16<D, nthreads>;
    } இல்லையெனில் எனில் constexpr (type_K == GGML_TYPE_Q4_0) {
        திரும்பு vec_dot_fattn_vec_KQ_q4_0<D, nthreads, warp_size>;
    } இல்லையெனில் எனில் constexpr (type_K == GGML_TYPE_Q4_1) {
        திரும்பு vec_dot_fattn_vec_KQ_q4_1<D, nthreads, warp_size>;
    } இல்லையெனில் எனில் constexpr (type_K == GGML_TYPE_Q5_0) {
        திரும்பு vec_dot_fattn_vec_KQ_q5_0<D, nthreads, warp_size>;
    } இல்லையெனில் எனில் constexpr (type_K == GGML_TYPE_Q5_1) {
        திரும்பு vec_dot_fattn_vec_KQ_q5_1<D, nthreads, warp_size>;
    } இல்லையெனில் எனில் constexpr (type_K == GGML_TYPE_Q8_0) {
        திரும்பு vec_dot_fattn_vec_KQ_q8_0<D, nthreads, warp_size>;
    } இல்லையெனில் {
        static_assert(type_K == -1, "bad type");
        திரும்பு nullptr;
    }
}

template <முழு type_V, typename T, முழு ne>
constexpr dequantize_V_t get_dequantize_V() {
    எனில் constexpr (type_V == GGML_TYPE_F16) {
        திரும்பு dequantize_V_f16<T, ne>;
    } இல்லையெனில் எனில் constexpr (type_V == GGML_TYPE_Q4_0) {
        திரும்பு dequantize_V_q4_0<T, ne>;
    } இல்லையெனில் எனில் constexpr (type_V == GGML_TYPE_Q4_1) {
        திரும்பு dequantize_V_q4_1<T, ne>;
    } இல்லையெனில் எனில் constexpr (type_V == GGML_TYPE_Q5_0) {
        திரும்பு dequantize_V_q5_0<T, ne>;
    } இல்லையெனில் எனில் constexpr (type_V == GGML_TYPE_Q5_1) {
        திரும்பு dequantize_V_q5_1<T, ne>;
    } இல்லையெனில் எனில் constexpr (type_V == GGML_TYPE_Q8_0) {
        திரும்பு dequantize_V_q8_0<T, ne>;
    } இல்லையெனில் {
        static_assert(type_V == -1, "bad type");
        திரும்பு nullptr;
    }
}

template <முழு ncols1, முழு warp_size>
நிலையான வெறுமை flash_attn_mask_to_KV_max(மாறிலி sycl::half2 * __restrict__ mask,
                                      முழு * __restrict__ KV_max,
                                      மாறிலி முழு ne30,
                                      மாறிலி முழு s31,
                                      மாறிலி முழு s33,
                                      முழு *     buf_iw) {
    auto      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    மாறிலி முழு ne31     = item_ct1.get_group_range(2);
    மாறிலி முழு tid      = item_ct1.get_local_id(2);
    மாறிலி முழு sequence = item_ct1.get_group(1);
    மாறிலி முழு jt       = item_ct1.get_group(2);

    mask += sequence*s33 + jt*ncols1*s31;

    எனில் (tid < warp_size) {
        buf_iw[tid] = 1;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);

    முழு KV_max_sj = (ne30 - 1) * FATTN_KQ_STRIDE;
    க்கு (; KV_max_sj >= 0; KV_max_sj -= FATTN_KQ_STRIDE) {
        முழு all_inf = 1;

#pragma unroll
        க்கு (முழு j = 0; j < ncols1; ++j) {
            மாறிலி sycl::float2 tmp =
                mask[j * s31 + KV_max_sj / 2 + tid].template convert<தசம, sycl::rounding_mode::automatic>();
            all_inf = all_inf && முழு(sycl::isinf((தசம) (tmp.x()))) && முழு(sycl::isinf((தசம) (tmp.y())));
        }

        all_inf = warp_reduce_all<warp_size>(all_inf);
        எனில் (tid % warp_size == 0) {
            buf_iw[tid / warp_size] = all_inf;
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);
        all_inf = buf_iw[tid % warp_size];
        item_ct1.barrier(sycl::access::fence_space::local_space);
        all_inf = warp_reduce_all<warp_size>(all_inf);

        எனில் (!all_inf) {
            நிறுத்து;
        }
    }

    // If the break in the loop was not triggered, KV_max_sj is now -FATTN_KQ_STRIDE.
    // If the break was triggered it's the lower edge of the tile with the first non-masked values.
    // In either case, walk back the decrementation by FATTN_KQ_STRIDE.
    KV_max_sj += FATTN_KQ_STRIDE;

    எனில் (item_ct1.get_local_id(2) != 0) {
        திரும்பு;
    }

    KV_max[sequence*ne31 + jt] = KV_max_sj;
}

template <முழு D, முழு ncols1, முழு ncols2>  // D == head size

நிலையான வெறுமை flash_attn_stream_k_fixup(தசம * __restrict__ dst,
                                      மாறிலி sycl::float2 * __restrict__ dst_fixup,
                                      மாறிலி முழு ne01,
                                      மாறிலி முழு ne02,
                                      மாறிலி முழு ne03,
                                      மாறிலி முழு ne11,
                                      மாறிலி முழு ne12,
                                      மாறிலி முழு nbatch_fa) {
    auto          item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr முழு ncols    = ncols1 * ncols2;

    மாறிலி முழு bidx0 = item_ct1.get_group(2);
    மாறிலி முழு j     = item_ct1.get_group(1);
    மாறிலி முழு c     = item_ct1.get_group(0);
    மாறிலி முழு jc    = j*ncols2 + c;
    மாறிலி முழு tid   = item_ct1.get_local_id(2);

    மாறிலி தசம * dst_fixup_data = ((மாறிலி தசம *) dst_fixup) + item_ct1.get_group_range(2) * (2 * 2 * ncols);

    மாறிலி முழு gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.

    மாறிலி முழு iter_k     = (ne11      + (nbatch_fa - 1)) / nbatch_fa;
    மாறிலி முழு iter_j     = (ne01      + (ncols1    - 1)) / ncols1;
    மாறிலி முழு iter_z_gqa = (gqa_ratio + (ncols2    - 1)) / ncols2;

    மாறிலி முழு kbc0 = int64_t(bidx0 + 0) * (iter_k * iter_j * iter_z_gqa * ne12 * ne03) / item_ct1.get_group_range(2);
    மாறிலி முழு kbc0_stop =
        int64_t(bidx0 + 1) * (iter_k * iter_j * iter_z_gqa * ne12 * ne03) / item_ct1.get_group_range(2);

    மாறிலி தருக்க did_not_have_any_data   = kbc0 == kbc0_stop;
    மாறிலி தருக்க wrote_beginning_of_tile = kbc0 % iter_k == 0;
    மாறிலி தருக்க did_not_write_last      = kbc0/iter_k == kbc0_stop/iter_k && kbc0_stop % iter_k != 0;
    எனில் (did_not_have_any_data || wrote_beginning_of_tile || did_not_write_last) {
        திரும்பு;
    }

    // z_KV == K/V head index, zt_gqa = Q head start index per K/V head, jt = token position start index
    மாறிலி முழு sequence =  kbc0 /(iter_k*iter_j*iter_z_gqa*ne12);
    மாறிலி முழு z_KV     = (kbc0 - iter_k*iter_j*iter_z_gqa*ne12 * sequence)/(iter_k*iter_j*iter_z_gqa);
    மாறிலி முழு zt_gqa   = (kbc0 - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV)/(iter_k*iter_j);
    மாறிலி முழு jt       = (kbc0 - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV - iter_k*iter_j * zt_gqa) / iter_k;

    மாறிலி முழு zt_Q = z_KV*gqa_ratio + zt_gqa*ncols2; // Global Q head start index.

    எனில் (jt*ncols1 + j >= ne01 || zt_gqa*ncols2 + c >= gqa_ratio) {
        திரும்பு;
    }

    dst += sequence*ne02*ne01*D + jt*ne02*(ncols1*D) + zt_Q*D + (j*ne02 + c)*D + tid;

    // Load the partial result that needs a fixup:
    தசம dst_val = 0.0f;
    தசம max_val = 0.0f;
    தசம rowsum  = 0.0f;
    {
        dst_val = *dst;

        மாறிலி sycl::float2 tmp = dst_fixup[bidx0 * ncols + jc];
        max_val                = tmp.x();
        rowsum                 = tmp.y();
    }

    // Iterate over previous blocks and compute the combined results.
    // All SYCL blocks that get here must have a previous block that needs a fixup.
    முழு bidx = bidx0 - 1;
    முழு kbc_stop = kbc0;
    போது(உண்மை) {
        மாறிலி முழு kbc = int64_t(bidx) * (iter_k * iter_j * iter_z_gqa * ne12 * ne03) / item_ct1.get_group_range(2);
        எனில் (kbc == kbc_stop) { // Did not have any data.
            bidx--;
            kbc_stop = kbc;
            தொடர்;
        }

        மாறிலி தசம dst_add = dst_fixup_data[bidx*ncols*D + jc*D + tid];

        மாறிலி sycl::float2 tmp = dst_fixup[(item_ct1.get_group_range(2) + bidx) * ncols + jc];

        // Scale the current and new value accumulators depending on the max. values.
        மாறிலி தசம max_val_new = sycl::fmax(max_val, tmp.x());

        மாறிலி தசம diff_val = max_val - max_val_new;
        மாறிலி தசம diff_add = tmp.x() - max_val_new;

        மாறிலி தசம scale_val = diff_val >= SOFTMAX_FTZ_THRESHOLD ? sycl::native::exp(diff_val) : 0.0f;
        மாறிலி தசம scale_add = diff_add >= SOFTMAX_FTZ_THRESHOLD ? sycl::native::exp(diff_add) : 0.0f;

        dst_val = scale_val*dst_val + scale_add*dst_add;
        rowsum  = scale_val * rowsum + scale_add * tmp.y();

        max_val = max_val_new;

        // If this block started in a previous tile we are done and don't need to combine additional partial results.
        எனில் (kbc % iter_k == 0 || kbc/iter_k < kbc0/iter_k) {
            நிறுத்து;
        }
        bidx--;
        kbc_stop = kbc;
    }

    // Write back final result:
    *dst = dst_val / rowsum;
}

template <முழு D>  // D == head size

நிலையான வெறுமை flash_attn_combine_results(மாறிலி தசம * __restrict__ VKQ_parts,
                                       மாறிலி sycl::float2 * __restrict__ VKQ_meta,
                                       தசம * __restrict__ dst,
                                       மாறிலி முழு parallel_blocks,
                                       uint8_t * dpct_local) {
    // Dimension 0: threadIdx.x
    // Dimension 1: blockIdx.x
    // Dimension 2: blockIdx.y
    // Dimension 3: blockIdx.z
    // Memory layout is permuted with [0, 2, 1, 3]

    auto      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    மாறிலி முழு ne01     = item_ct1.get_group_range(2);
    மாறிலி முழு ne02     = item_ct1.get_group_range(1);

    மாறிலி முழு col      = item_ct1.get_group(2);
    மாறிலி முழு head     = item_ct1.get_group(1);
    மாறிலி முழு sequence = item_ct1.get_group(0);

    மாறிலி முழு j_dst_unrolled = (sequence*ne01 + col)*ne02 + head;

    VKQ_parts += j_dst_unrolled * parallel_blocks*D;
    VKQ_meta  += j_dst_unrolled * parallel_blocks;
    dst       += j_dst_unrolled *                 D;

    மாறிலி முழு tid = item_ct1.get_local_id(2);
    __builtin_assume(tid < D);

    auto meta = (sycl::float2 *) dpct_local;
    க்கு (முழு i = tid; i < 2*parallel_blocks; i += D) {
        ((தசம *) meta)[i] = ((மாறிலி தசம *)VKQ_meta) [i];
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    தசம kqmax = meta[0].x();
    க்கு (முழு l = 1; l < parallel_blocks; ++l) {
        kqmax = sycl::max(kqmax, meta[l].x());
    }

    தசம VKQ_numerator   = 0.0f;
    தசம VKQ_denominator = 0.0f;
    க்கு (முழு l = 0; l < parallel_blocks; ++l) {
        மாறிலி தசம KQ_max_scale = sycl::native::exp(meta[l].x() - kqmax);

        VKQ_numerator   += KQ_max_scale * VKQ_parts[l*D + tid];
        VKQ_denominator += KQ_max_scale * meta[l].y();
    }

    dst[tid] = VKQ_numerator / VKQ_denominator;
}

// Carries the sub-group size (and grf_size_automatic when use_large_grf) as
// kernel properties via get(properties_tag), in place of the
// reqd_sub_group_size attribute and the deprecated parallel_for overload.
template <typename KernelFunc, uint32_t Size, தருக்க UseGrf>
கட்டமைப்பு props_kernel {
    KernelFunc kernel;

    வெறுமை operator()(sycl::nd_item<3> item) மாறிலி {
        kernel(item);
    }

    auto get(sycl::ext::oneapi::experimental::properties_tag) மாறிலி {
        எனில் constexpr (UseGrf) {
            திரும்பு sycl::ext::oneapi::experimental::properties{
                sycl::ext::intel::experimental::grf_size_automatic,
                sycl::ext::oneapi::experimental::sub_group_size<Size> };
        } இல்லையெனில் {
            திரும்பு sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::sub_group_size<Size> };
        }
    }
};

template <fattn_kernel_t fattn_kernel, முழு warp_size, தருக்க use_large_grf = பொய்>
நிலையான வெறுமை lauch_kernel(
    dpct::dim3 group_range,
    dpct::dim3 local_range,
    queue_ptr q,
    unsigned முழு local_mem_size,
    மாறிலி எழுத்து* __restrict__ Q,
    மாறிலி எழுத்து* __restrict__ K,
    மாறிலி எழுத்து* __restrict__ V,
    மாறிலி எழுத்து* __restrict__ mask,
    மாறிலி எழுத்து* __restrict__ sinks,
    மாறிலி முழு* __restrict__ KV_max,
    தசம* __restrict__ dst,
    sycl::float2* __restrict__ dst_meta,
    மாறிலி தசம scale,
    மாறிலி தசம max_bias,
    மாறிலி தசம m0,
    மாறிலி தசம m1,
    மாறிலி uint32_t n_head_log2,
    மாறிலி தசம logit_softcap,
    மாறிலி int32_t ne00,
    மாறிலி sycl::uint3 ne01,
    மாறிலி int32_t ne02,
    மாறிலி int32_t ne03,
    மாறிலி int32_t nb01,
    மாறிலி int32_t nb02,
    மாறிலி int32_t nb03,
    மாறிலி int32_t ne10,
    மாறிலி int32_t ne11,
    மாறிலி int32_t ne12,
    மாறிலி int32_t ne13,
    மாறிலி int32_t nb11,
    மாறிலி int32_t nb12,
    மாறிலி int64_t nb13,
    மாறிலி int32_t nb21,
    மாறிலி int32_t nb22,
    மாறிலி int64_t nb23,
    மாறிலி int32_t ne31,
    மாறிலி int32_t ne32,
    மாறிலி int32_t ne33,
    மாறிலி int32_t nb31,
    மாறிலி int32_t nb32,
    மாறிலி int64_t nb33) {
    GGML_UNUSED(local_mem_size);

    மாறிலி auto rng = sycl::nd_range<3>(
        static_cast<sycl::range<3>>(group_range * local_range),
        static_cast<sycl::range<3>>(local_range));

    மாறிலி auto kernel = [=](sycl::nd_item<3> item_ct1) {
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
template <முழு DV, முழு ncols1, முழு ncols2, fattn_kernel_t fattn_kernel, முழு warp_size, தருக்க use_large_grf = பொய்>
வெறுமை launch_fattn(
    ggml_backend_sycl_context & ctx, ggml_tensor * dst, மாறிலி முழு nwarps, மாறிலி size_t nbytes_shared,
    மாறிலி முழு nbatch_fa, மாறிலி தருக்க need_f16_K, மாறிலி தருக்க need_f16_V, மாறிலி தருக்க stream_k) {

    constexpr முழு ncols = ncols1 * ncols2;

    மாறிலி ggml_tensor * Q = dst->src[0];
    மாறிலி ggml_tensor * K = dst->src[1];
    மாறிலி ggml_tensor * V = dst->src[2];

    மாறிலி தருக்க V_is_K_view = V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));

    மாறிலி ggml_tensor * mask  = dst->src[3];
    மாறிலி ggml_tensor * sinks = dst->src[4];

    ggml_tensor * KQV = dst;

    GGML_ASSERT(Q->type == GGML_TYPE_F32);
    GGML_ASSERT(KQV->type == GGML_TYPE_F32);

    GGML_ASSERT(Q->nb[0] == ggml_element_size(Q));
    GGML_ASSERT(K->nb[0] == ggml_element_size(K));
    GGML_ASSERT(V->nb[0] == ggml_element_size(V));

    GGML_ASSERT(!mask || mask->type == GGML_TYPE_F16);

    ggml_sycl_pool & pool = ctx.pool();
    dpct::queue_ptr  main_stream = ctx.stream();
    மாறிலி முழு id  = ggml_sycl_get_device();
    மாறிலி முழு nsm = ggml_sycl_info().devices[id].nsm;

    மாறிலி ggml_sycl_fattn_extra  extra = ggml_sycl_fattn_get_extra(dst);
    ggml_sycl_pool_alloc<முழு>    KV_max(pool);
    ggml_sycl_pool_alloc<தசம>  dst_tmp(pool);
    ggml_sycl_pool_alloc<sycl::float2> dst_tmp_meta(pool);

    மாறிலி எழுத்து * K_data = (மாறிலி எழுத்து *) K->data;
    size_t nb11 = K->nb[1];
    size_t nb12 = K->nb[2];
    size_t nb13 = K->nb[3];

    மாறிலி எழுத்து * V_data = (மாறிலி எழுத்து *) V->data;
    size_t nb21 = V->nb[1];
    size_t nb22 = V->nb[2];
    size_t nb23 = V->nb[3];

    எனில் (need_f16_K && K->type != GGML_TYPE_F16) {
        மாறிலி size_t bs = ggml_blck_size(K->type);
        மாறிலி size_t ts = ggml_type_size(K->type);

        GGML_ASSERT(extra.K_buffer_ptr);
        sycl::half * K_f16_ptr = (sycl::half *) extra.K_buffer_ptr;
        எனில் (ggml_is_contiguously_allocated(K)) {
            to_fp16_sycl_t to_fp16 = ggml_get_to_fp16_sycl(K->type, dst);
            to_fp16(K_data, K_f16_ptr, ggml_nelements(K), main_stream);

            nb11 = nb11 * bs * அளவு(sycl::half) / ts;
            nb12 = nb12 * bs * அளவு(sycl::half) / ts;
            nb13 = nb13 * bs * அளவு(sycl::half) / ts;
        } இல்லையெனில் {
            GGML_ASSERT(K->nb[0] == ts);
            to_fp16_nc_sycl_t to_fp16 = ggml_get_to_fp16_nc_sycl(K->type);
            மாறிலி int64_t s01 = nb11 / ts;
            மாறிலி int64_t s02 = nb12 / ts;
            மாறிலி int64_t s03 = nb13 / ts;
            to_fp16(K_data, K_f16_ptr, K->ne[0], K->ne[1], K->ne[2], K->ne[3], s01, s02, s03, main_stream);

            nb11 = K->ne[0] * அளவு(sycl::half);
            nb12 = K->ne[1] * nb11;
            nb13 = K->ne[2] * nb12;
        }
        K_data = (எழுத்து *) K_f16_ptr;
    }

    எனில் (need_f16_V && V->type != GGML_TYPE_F16) {
        எனில் (V_is_K_view) {
            V_data = K_data;
            nb21   = nb11;
            nb22   = nb12;
            nb23   = nb13;
        } இல்லையெனில் {
            மாறிலி size_t bs = ggml_blck_size(V->type);
            மாறிலி size_t ts = ggml_type_size(V->type);

            GGML_ASSERT(extra.V_buffer_ptr);
            sycl::half * V_f16_ptr = (sycl::half *) extra.V_buffer_ptr;
            எனில் (ggml_is_contiguously_allocated(V)) {
                to_fp16_sycl_t to_fp16 = ggml_get_to_fp16_sycl(V->type, dst);
                to_fp16(V_data, V_f16_ptr, ggml_nelements(V), main_stream);
                V_data = (எழுத்து *) V_f16_ptr;

                nb21 = nb21 * bs * அளவு(sycl::half) / ts;
                nb22 = nb22 * bs * அளவு(sycl::half) / ts;
                nb23 = nb23 * bs * அளவு(sycl::half) / ts;
            } இல்லையெனில் {
                GGML_ASSERT(V->nb[0] == ts);
                to_fp16_nc_sycl_t to_fp16 = ggml_get_to_fp16_nc_sycl(V->type);
                மாறிலி int64_t s01 = nb21 / ts;
                மாறிலி int64_t s02 = nb22 / ts;
                மாறிலி int64_t s03 = nb23 / ts;
                to_fp16(V_data, V_f16_ptr, V->ne[0], V->ne[1], V->ne[2], V->ne[3], s01, s02, s03, main_stream);

                nb21 = V->ne[0] * அளவு(sycl::half);
                nb22 = V->ne[1] * nb21;
                nb23 = V->ne[2] * nb22;
            }
            V_data = (எழுத்து *) V_f16_ptr;
        }
    }

    மாறிலி முழு ntiles_x     = ((Q->ne[1] + ncols1 - 1) / ncols1);
    மாறிலி முழு gqa_ratio    = Q->ne[2] / K->ne[2];
    மாறிலி முழு ntiles_z_gqa = ((gqa_ratio + ncols2 - 1) / ncols2);
    மாறிலி முழு ntiles_total = ntiles_x * ntiles_z_gqa * K->ne[2] * Q->ne[3];

    // Optional optimization where the mask is scanned to determine whether part of the calculation can be skipped.
    // Only worth the overhead if there is at lease one FATTN_KQ_STRIDE x FATTN_KQ_STRIDE square to be skipped or
    //     multiple sequences of possibly different lengths.
    எனில் (mask && K->ne[1] % FATTN_KQ_STRIDE == 0 && (Q->ne[1] >= 1024 || Q->ne[3] > 1)) {
        மாறிலி முழு s31 = mask->nb[1] / அளவு(sycl::half2);
        மாறிலி முழு s33 = mask->nb[3] / அளவு(sycl::half2);

        மாறிலி dpct::dim3 blocks_இலக்கம்_KV_max(ntiles_x, Q->ne[3], 1);
        மாறிலி dpct::dim3 block_dim_KV_max(FATTN_KQ_STRIDE / 2, 1, 1);

        மாறிலி முழு ne_KV_max = blocks_இலக்கம்_KV_max.x*blocks_இலக்கம்_KV_max.y;
        மாறிலி முழு iter_k = K->ne[1] / FATTN_KQ_STRIDE;

        KV_max.alloc(ne_KV_max);
        {
            dpct::has_capability_or_fail(main_stream->get_device(), { sycl::aspect::fp16 });

            main_stream->submit([&](sycl::handler & cgh) {
                sycl::local_accessor<முழு, 1> buf_iw_acc_ct1(sycl::range<1>(warp_size), cgh);

                auto mask_data_ct0  = (மாறிலி sycl::half2 *) mask->data;
                auto KV_max_ptr_ct1 = KV_max.ptr;

                cgh.parallel_for(sycl::nd_range<3>(blocks_இலக்கம்_KV_max * block_dim_KV_max, block_dim_KV_max),
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

    மாறிலி dpct::dim3 block_dim(warp_size, nwarps, 1);

    // Max. number of active blocks limited by occupancy.
    முழு max_blocks_per_sm = ggml_sycl_info().devices[id].max_wg_per_cu;
    முழு parallel_blocks = max_blocks_per_sm;
    dpct::dim3 blocks_இலக்கம்;
    எனில் (stream_k) {
        // For short contexts it can be faster to have the SMs work on whole tiles because this lets us skip the fixup.
        மாறிலி முழு max_blocks = max_blocks_per_sm*nsm;
        மாறிலி முழு nblocks_stream_k = max_blocks;
        மாறிலி தருக்க use_stream_k = உண்மை;

        blocks_இலக்கம்.x = use_stream_k ? nblocks_stream_k : ntiles_total;
        blocks_இலக்கம்.y = 1;
        blocks_இலக்கம்.z = 1;

        எனில் (ntiles_total % blocks_இலக்கம்.x != 0) { // Fixup is only needed if the SMs work on fractional tiles.
            dst_tmp_meta.alloc((size_t(blocks_இலக்கம்.x) * ncols * (2 + DV/2)));
        }
    } இல்லையெனில் {
        மாறிலி முழு ntiles_KQ = (K->ne[1] + nbatch_fa - 1) / nbatch_fa; // Max. number of parallel blocks limited by tensor size.

        // parallel_blocks must not be larger than what the tensor size allows:
        parallel_blocks = std::min(parallel_blocks, ntiles_KQ);
        // todo fix the hard code change
        // parallel_blocks = ntiles_KQ;

        // If ntiles_total % blocks_per_wave != 0 then some efficiency is lost due to tail effects.
        // Test whether parallel_blocks can be set to a higher value for better efficiency.
        மாறிலி முழு blocks_per_wave = nsm * max_blocks_per_sm;
        முழு nwaves_best = 0;
        முழு efficiency_percent_best = 0;
        க்கு (முழு parallel_blocks_test = parallel_blocks; parallel_blocks_test <= ntiles_KQ; ++parallel_blocks_test) {
            மாறிலி முழு nblocks_total = ntiles_total * parallel_blocks_test;
            மாறிலி முழு nwaves = (nblocks_total + blocks_per_wave - 1) / blocks_per_wave;
            மாறிலி முழு efficiency_percent = 100 * nblocks_total / (nwaves*blocks_per_wave);

            // Stop trying configurations with more waves if we already have good efficiency to avoid excessive overhead.
            எனில் (efficiency_percent_best >= 95 && nwaves > nwaves_best) {
                நிறுத்து;
            }

            எனில் (efficiency_percent > efficiency_percent_best) {
                nwaves_best = nwaves;
                efficiency_percent_best = efficiency_percent;
                parallel_blocks = parallel_blocks_test;
            }
        }

        blocks_இலக்கம்.x = ntiles_x;
        blocks_இலக்கம்.y = parallel_blocks;
        blocks_இலக்கம்.z = ntiles_z_gqa*K->ne[2]*Q->ne[3];

        எனில் (parallel_blocks > 1) {
            dst_tmp.alloc(parallel_blocks*ggml_nelements(KQV));
            dst_tmp_meta.alloc(parallel_blocks*ggml_nrows(KQV));
        }
    }

    தசம scale         = 1.0f;
    தசம max_bias      = 0.0f;
    தசம logit_softcap = 0.0f;

    memcpy(&scale,         (மாறிலி தசம *) KQV->op_params + 0, அளவு(தசம));
    memcpy(&max_bias,      (மாறிலி தசம *) KQV->op_params + 1, அளவு(தசம));
    memcpy(&logit_softcap, (மாறிலி தசம *) KQV->op_params + 2, அளவு(தசம));

    எனில் (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    மாறிலி uint32_t n_head      = Q->ne[2];
    மாறிலி uint32_t n_head_log2 = 1u << uint32_t(floorf(log2f(தசம(n_head))));

    மாறிலி தசம m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    மாறிலி தசம m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    // TODO other tensor dimensions after removal of WMMA kernel:
    மாறிலி sycl::uint3 ne01 = init_fastdiv_மதிப்புகள்(Q->ne[1]);

    GGML_ASSERT(block_dim.x % warp_size == 0);

    lauch_kernel<fattn_kernel, warp_size, use_large_grf>(
        blocks_இலக்கம், block_dim, main_stream, (unsigned முழு) nbytes_shared, (மாறிலி எழுத்து *) Q->data, K_data, V_data,
        mask ? ((மாறிலி எழுத்து *) mask->data) : nullptr, sinks ? ((மாறிலி எழுத்து *) sinks->data) : nullptr, KV_max.ptr,
        !stream_k && parallel_blocks > 1 ? dst_tmp.ptr : (தசம *) KQV->data, (sycl::float2 *)dst_tmp_meta.ptr, scale, max_bias, m0, m1,
        n_head_log2, logit_softcap, Q->ne[0], ne01, Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3], K->ne[0],
        K->ne[1], K->ne[2], K->ne[3], nb11, nb12, nb13, nb21, nb22, nb23, mask ? mask->ne[1] : 0,
        mask ? mask->ne[2] : 0, mask ? mask->ne[3] : 0, mask ? mask->nb[1] : 0, mask ? mask->nb[2] : 0,
        mask ? mask->nb[3] : 0);
    SYCL_CHECK(0);

    எனில் (stream_k) {
        எனில் (ntiles_total % blocks_இலக்கம்.x != 0) { // Fixup is only needed if the SMs work on fractional tiles.
            மாறிலி dpct::dim3 block_dim_combine(DV, 1, 1);
            மாறிலி dpct::dim3 blocks_இலக்கம்_combine = { blocks_இலக்கம்.x, ncols1, ncols2 };

            main_stream->submit([&](sycl::handler & cgh) {
                auto KQV_data_ct0         = (தசம *) KQV->data;
                auto dst_tmp_meta_ptr_ct1 = dst_tmp_meta.ptr;
                auto Q_ne_ct2             = Q->ne[1];
                auto Q_ne_ct3             = Q->ne[2];
                auto Q_ne_ct4             = Q->ne[3];
                auto K_ne_ct5             = K->ne[1];
                auto K_ne_ct6             = K->ne[2];

                cgh.parallel_for(sycl::nd_range<3>(blocks_இலக்கம்_combine * block_dim_combine, block_dim_combine),
                                 [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(warp_size)]] {
                                     GGML_UNUSED(item_ct1);
                                     flash_attn_stream_k_fixup<DV, ncols1, ncols2>(KQV_data_ct0, dst_tmp_meta_ptr_ct1,
                                                                                   Q_ne_ct2, Q_ne_ct3, Q_ne_ct4,
                                                                                   K_ne_ct5, K_ne_ct6, nbatch_fa);
                                 });
            });
        }
    } இல்லையெனில் எனில் (parallel_blocks > 1) {
        மாறிலி dpct::dim3 block_dim_combine(DV, 1, 1);
        மாறிலி dpct::dim3 blocks_இலக்கம்_combine(Q->ne[1], Q->ne[2], Q->ne[3]);
        மாறிலி size_t     nbytes_shared_combine = parallel_blocks * அளவு(sycl::float2);
        main_stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(sycl::range<1>(nbytes_shared_combine), cgh);

            auto dst_tmp_ptr_ct0      = dst_tmp.ptr;
            auto dst_tmp_meta_ptr_ct1 = dst_tmp_meta.ptr;
            auto KQV_data_ct2         = (தசம *) KQV->data;

            cgh.parallel_for(sycl::nd_range<3>(blocks_இலக்கம்_combine * block_dim_combine, block_dim_combine),
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
