#pragma once

#अन्तर्भाव <sycl/sycl.hpp>
#अन्तर्भाव <sycl/ext/intel/experimental/grf_size_properties.hpp>
#अन्तर्भाव "dpct/helper.hpp"
#अन्तर्भाव "common.hpp"
#अन्तर्भाव "convert.hpp"
#अन्तर्भाव "vecdotq.hpp"
#अन्तर्भाव "fattn.hpp"

#अन्तर्भाव "ggml.h"

#अन्तर्भाव <cstdint>
#अन्तर्भाव <cmath>
#अन्तर्भाव <float.h>


#define FATTN_KQ_STRIDE       256
#define HALF_MAX_HALF         sycl::half(65504.0f/2) // Use neg. of this instead of -INFINITY to initialize KQ max vals to avoid NaN upon subtraction.
#define SOFTMAX_FTZ_THRESHOLD -20.0f                   // Softmax exp. of values smaller than this are flushed to zero to avoid NaNs.
#define FATTN_KQ_MAX_OFFSET (3.0f*0.6931f)

प्रकारनिर्देश शून्य (*fattn_kernel_t)(
    अचर वर्ण* Q,
    अचर वर्ण* K,
    अचर वर्ण* V,
    अचर वर्ण* mask,
    अचर वर्ण* sinks,
    अचर पूर्णाङ्क* KV_max,
    भिन्न* dst,
    sycl::float2* dst_meta,
    अचर भिन्न scale,
    अचर भिन्न max_bias,
    अचर भिन्न m0,
    अचर भिन्न m1,
    अचर uint32_t n_head_log2,
    अचर भिन्न logit_softcap,
    अचर int32_t ne00,
    अचर sycl::uint3 ne01,
    अचर int32_t ne02,
    अचर int32_t ne03,
    अचर int32_t nb01,
    अचर int32_t nb02,
    अचर int32_t nb03,
    अचर int32_t ne10,
    अचर int32_t ne11,
    अचर int32_t ne12,
    अचर int32_t ne13,
    अचर int32_t nb11,
    अचर int32_t nb12,
    अचर int64_t nb13,
    अचर int32_t nb21,
    अचर int32_t nb22,
    अचर int64_t nb23,
    अचर int32_t ne31,
    अचर int32_t ne32,
    अचर int32_t ne33,
    अचर int32_t nb31,
    अचर int32_t nb32,
    अचर int64_t nb33);

प्रकारनिर्देश भिन्न (*vec_dot_KQ_t)(
    अचर वर्ण * __restrict__ K_c, अचर शून्य * __restrict__ Q_v, अचर पूर्णाङ्क * __restrict__ Q_q8 , अचर शून्य * __restrict__ Q_ds);

ढाँचा <पूर्णाङ्क D, पूर्णाङ्क nthreads>
स्थिर __dpct_inline__ भिन्न vec_dot_fattn_vec_KQ_f16(अचर वर्ण * __restrict__ K_c,
                                                      अचर शून्य * __restrict__ Q_v,
                                                      अचर पूर्णाङ्क * __restrict__ Q_q8,
                                                      अचर शून्य * __restrict__ Q_ds_v) {
    अचर sycl::half2 * K_h2 = (अचर sycl::half2 *) K_c;
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v);

    constexpr पूर्णाङ्क cpy_nb = ggml_sycl_get_max_cpy_bytes();
    constexpr पूर्णाङ्क cpy_ne = cpy_nb / 4;

    भिन्न sum = 0.0f;

#pragma unroll
    लागि (पूर्णाङ्क k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads*cpy_ne) {
        sycl::half2 tmp[cpy_ne];
        ggml_sycl_memcpy_1<परिमाण(tmp)>(
            tmp,
            K_h2 + k_KQ_0 + (sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2) % nthreads) * cpy_ne);
#pragma unroll
        लागि (पूर्णाङ्क k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
#ifdef GGML_SYCL_F16
            ggml_sycl_mad(sum,                tmp[k_KQ_1] , ((अचर sycl::half2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#अन्यथा
            ggml_sycl_mad(sum, __half22float2(tmp[k_KQ_1]), ((अचर sycl::float2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#endif // GGML_SYCL_F16
        }
    }

    फर्काऊ sum;
}

ढाँचा <पूर्णाङ्क D, पूर्णाङ्क nthreads, पूर्णाङ्क warp_size>
स्थिर __dpct_inline__ भिन्न vec_dot_fattn_vec_KQ_q4_0(अचर वर्ण * __restrict__ K_c,
                                                       अचर शून्य * __restrict__ Q_v,
                                                       अचर पूर्णाङ्क * __restrict__ Q_q8,
                                                       अचर शून्य * __restrict__ Q_ds_v) {
    स्वतः item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    अचर block_q4_0 * K_q4_0   = (अचर block_q4_0 *) K_c;
    GGML_UNUSED(Q_v);

    भिन्न sum = 0.0f;

#pragma unroll
    लागि (पूर्णाङ्क k_KQ_0 = 0; k_KQ_0 < पूर्णाङ्क(D/परिमाण(पूर्णाङ्क)); k_KQ_0 += nthreads) {
        अचर पूर्णाङ्क k_KQ =
            k_KQ_0 + (nthreads == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads);

        अचर पूर्णाङ्क ib    = k_KQ /  QI8_1;
        अचर पूर्णाङ्क iqs4  = k_KQ %  QI4_0;
        अचर पूर्णाङ्क shift = k_KQ & (QI8_1/2);

        पूर्णाङ्क v;
        ggml_sycl_memcpy_1<परिमाण(पूर्णाङ्क), 2>(&v, K_q4_0[ib].qs + परिमाण(पूर्णाङ्क)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;
        अचर पूर्णाङ्क u = Q_q8[k_KQ_0/nthreads];

        अचर पूर्णाङ्क sumi = ggml_sycl_dp4a(v, u, 0);

        अचर sycl::float2 Q_ds = ((अचर sycl::float2 *) Q_ds_v)[k_KQ_0 / nthreads];
        sum += __half2float(K_q4_0[ib].d) * (sumi*Q_ds.x() - (8/QI8_1)*Q_ds.y());
    }

    फर्काऊ sum;
}

ढाँचा <पूर्णाङ्क D, पूर्णाङ्क nthreads , पूर्णाङ्क warp_size>
स्थिर __dpct_inline__ भिन्न vec_dot_fattn_vec_KQ_q4_1(अचर वर्ण * __restrict__ K_c,
                                                       अचर शून्य * __restrict__ Q_v,
                                                       अचर पूर्णाङ्क * __restrict__ Q_q8,
                                                       अचर शून्य * __restrict__ Q_ds_v) {
    स्वतः               item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    अचर block_q4_1 * K_q4_1   = (अचर block_q4_1 *) K_c;
    GGML_UNUSED(Q_v);

    भिन्न sum = 0.0f;

#pragma unroll
    लागि (पूर्णाङ्क k_KQ_0 = 0; k_KQ_0 < पूर्णाङ्क(D/परिमाण(पूर्णाङ्क)); k_KQ_0 += nthreads) {
        अचर पूर्णाङ्क k_KQ =
            k_KQ_0 + (nthreads == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads);

        अचर पूर्णाङ्क ib    = k_KQ /  QI8_1;
        अचर पूर्णाङ्क iqs4  = k_KQ %  QI4_1;
        अचर पूर्णाङ्क shift = k_KQ & (QI8_1/2);

        पूर्णाङ्क v;
        ggml_sycl_memcpy_1<परिमाण(पूर्णाङ्क)>(&v, K_q4_1[ib].qs + परिमाण(पूर्णाङ्क)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;
        अचर पूर्णाङ्क u = Q_q8[k_KQ_0/nthreads];

        अचर पूर्णाङ्क sumi = ggml_sycl_dp4a(v, u, 0);

        अचर sycl::float2 K_dm = (K_q4_1[ib].dm).ढाँचा convert<भिन्न, sycl::rounding_mode::automatic>();
        अचर sycl::float2 Q_ds = ((अचर sycl::float2 *) Q_ds_v)[k_KQ_0 / nthreads];

        sum += K_dm.x()*Q_ds.x()*sumi + K_dm.y()*Q_ds.y()/QI8_1;
    }

    फर्काऊ sum;
}

ढाँचा <पूर्णाङ्क D, पूर्णाङ्क nthreads, पूर्णाङ्क warp_size>
स्थिर __dpct_inline__ भिन्न vec_dot_fattn_vec_KQ_q5_0(अचर वर्ण * __restrict__ K_c,
                                                       अचर शून्य * __restrict__ Q_v,
                                                       अचर पूर्णाङ्क * __restrict__ Q_q8,
                                                       अचर शून्य * __restrict__ Q_ds_v) {
    स्वतः               item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    अचर block_q5_0 * K_q5_0   = (अचर block_q5_0 *) K_c;
    GGML_UNUSED(Q_v);

    भिन्न sum = 0.0f;

#pragma unroll
    लागि (पूर्णाङ्क k_KQ_0 = 0; k_KQ_0 < पूर्णाङ्क(D/परिमाण(पूर्णाङ्क)); k_KQ_0 += nthreads) {
        अचर पूर्णाङ्क k_KQ =
            k_KQ_0 + (nthreads == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads);

        अचर पूर्णाङ्क ib    = k_KQ /  QI8_1;
        अचर पूर्णाङ्क iqs4  = k_KQ %  QI5_0;
        अचर पूर्णाङ्क iqs8  = k_KQ %  QI8_1;
        अचर पूर्णाङ्क shift = k_KQ & (QI8_1/2);

        पूर्णाङ्क v;
        ggml_sycl_memcpy_1<परिमाण(पूर्णाङ्क), 2>(&v, K_q5_0[ib].qs + परिमाण(पूर्णाङ्क)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;

        {
            पूर्णाङ्क vh;
            ggml_sycl_memcpy_1<परिमाण(पूर्णाङ्क), 2>(&vh, K_q5_0[ib].qh);
            vh >>= iqs8 * QI5_0;

            v |= (vh <<  4) & 0x00000010; // 0 ->  4
            v |= (vh << 11) & 0x00001000; // 1 -> 12
            v |= (vh << 18) & 0x00100000; // 2 -> 20
            v |= (vh << 25) & 0x10000000; // 3 -> 28
        }

        अचर पूर्णाङ्क u = Q_q8[k_KQ_0/nthreads];

        अचर पूर्णाङ्क sumi = ggml_sycl_dp4a(v, u, 0);

        अचर sycl::float2 Q_ds = ((अचर sycl::float2 *) Q_ds_v)[k_KQ_0 / nthreads];

        sum += __half2float(K_q5_0[ib].d) * (sumi*Q_ds.x() - (16/QI8_1)*Q_ds.y());
    }

    फर्काऊ sum;
}

ढाँचा <पूर्णाङ्क D, पूर्णाङ्क nthreads, पूर्णाङ्क warp_size>
स्थिर __dpct_inline__ भिन्न vec_dot_fattn_vec_KQ_q5_1(अचर वर्ण * __restrict__ K_c,
                                                       अचर शून्य * __restrict__ Q_v,
                                                       अचर पूर्णाङ्क * __restrict__ Q_q8,
                                                       अचर शून्य * __restrict__ Q_ds_v) {
    स्वतः               item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    अचर block_q5_1 * K_q5_1   = (अचर block_q5_1 *) K_c;
    GGML_UNUSED(Q_v);

    भिन्न sum = 0.0f;

#pragma unroll
    लागि (पूर्णाङ्क k_KQ_0 = 0; k_KQ_0 < पूर्णाङ्क(D/परिमाण(पूर्णाङ्क)); k_KQ_0 += nthreads) {
        अचर पूर्णाङ्क k_KQ =
            k_KQ_0 + (nthreads == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads);

        अचर पूर्णाङ्क ib    = k_KQ /  QI8_1;
        अचर पूर्णाङ्क iqs4  = k_KQ %  QI5_1;
        अचर पूर्णाङ्क iqs8  = k_KQ %  QI8_1;
        अचर पूर्णाङ्क shift = k_KQ & (QI8_1/2);

        पूर्णाङ्क v;
        ggml_sycl_memcpy_1<परिमाण(पूर्णाङ्क)>(&v, K_q5_1[ib].qs + परिमाण(पूर्णाङ्क)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;

        {
            पूर्णाङ्क vh;
            ggml_sycl_memcpy_1<परिमाण(पूर्णाङ्क)>(&vh, K_q5_1[ib].qh);
            vh >>= iqs8 * QI5_0;

            v |= (vh <<  4) & 0x00000010; // 0 ->  4
            v |= (vh << 11) & 0x00001000; // 1 -> 12
            v |= (vh << 18) & 0x00100000; // 2 -> 20
            v |= (vh << 25) & 0x10000000; // 3 -> 28
        }

        अचर पूर्णाङ्क u = Q_q8[k_KQ_0/nthreads];

        अचर पूर्णाङ्क sumi = ggml_sycl_dp4a(v, u, 0);

        अचर sycl::float2 K_dm = (K_q5_1[ib].dm).ढाँचा convert<भिन्न, sycl::rounding_mode::automatic>();
        अचर sycl::float2 Q_ds = ((अचर sycl::float2 *) Q_ds_v)[k_KQ_0 / nthreads];

        sum += K_dm.x()*Q_ds.x()*sumi + K_dm.y()*Q_ds.y()/QI8_1;
    }

    फर्काऊ sum;
}

ढाँचा <पूर्णाङ्क D, पूर्णाङ्क nthreads, पूर्णाङ्क warp_size>
स्थिर __dpct_inline__ भिन्न vec_dot_fattn_vec_KQ_q8_0(अचर वर्ण * __restrict__ K_c,
                                                       अचर शून्य * __restrict__ Q_v,
                                                       अचर पूर्णाङ्क * __restrict__ Q_q8,
                                                       अचर शून्य * __restrict__ Q_ds_v) {
    स्वतः               item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    अचर block_q8_0 * K_q8_0   = (अचर block_q8_0 *) K_c;
    GGML_UNUSED(Q_v);

    भिन्न sum = 0.0f;

#pragma unroll
    लागि (पूर्णाङ्क k_KQ_0 = 0; k_KQ_0 < पूर्णाङ्क(D/परिमाण(पूर्णाङ्क)); k_KQ_0 += nthreads) {
        अचर पूर्णाङ्क k_KQ =
            k_KQ_0 + (nthreads == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads);

        अचर पूर्णाङ्क ib  = k_KQ / QI8_0;
        अचर पूर्णाङ्क iqs = k_KQ % QI8_0;

        पूर्णाङ्क v;
        ggml_sycl_memcpy_1<परिमाण(v), 2>(&v, K_q8_0[ib].qs + 4*iqs);

        अचर sycl::float2 * Q_ds = (अचर sycl::float2 *) Q_ds_v;
        अचर भिन्न          Q_d  = Q_ds[k_KQ_0 / nthreads].x();

        sum += vec_dot_q8_0_q8_1_impl<भिन्न, 1>(&v, &Q_q8[k_KQ_0/nthreads], K_q8_0[ib].d, Q_d);
    }

    फर्काऊ sum;
}

ढाँचा <typename Tds, पूर्णाङ्क ni, पूर्णाङ्क warp_size>
स्थिर __dpct_inline__ शून्य quantize_q8_1_to_shared(अचर भिन्न * __restrict__ x,
                                                    अचर भिन्न scale,
                                                    पूर्णाङ्क * __restrict__ yq32,
                                                    शून्य * __restrict__ yds) {
    स्वतः item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    भिन्न vals[परिमाण(पूर्णाङ्क)] = { 0.0f };
#pragma unroll
    लागि (पूर्णाङ्क l = 0; l < पूर्णाङ्क(परिमाण(पूर्णाङ्क)); ++l) {
        vals[l] =
            (ni == warp_size || item_ct1.get_local_id(2) < ni) ? scale * x[4 * item_ct1.get_local_id(2) + l] : 0.0f;
    }

    भिन्न amax = sycl::भ_असल(vals[0]);
    भिन्न sum  = vals[0];
#pragma unroll
    लागि (पूर्णाङ्क l = 1; l < पूर्णाङ्क(परिमाण(पूर्णाङ्क)); ++l) {
        amax = sycl::fmax(amax, sycl::भ_असल(vals[l]));
        sum += vals[l];
    }
#pragma unroll
    लागि (पूर्णाङ्क mask = QI8_1/2; mask > 0; mask >>= 1) {
        amax = sycl::fmax(
            amax, dpct::permute_sub_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(), amax, mask));
        sum += dpct::permute_sub_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(), sum, mask);
    }

    अचर भिन्न d = amax / 127;
    पूर्णाङ्क q32 = 0;
    int8_t * q8 = (int8_t *) &q32;

    यदि (d != 0.0f) {
#pragma unroll
        लागि (पूर्णाङ्क l = 0; l < पूर्णाङ्क(परिमाण(पूर्णाङ्क)); ++l) {
            q8[l] = sycl::round(vals[l] / d);
        }
    }

    yq32[item_ct1.get_local_id(2)] = q32;
    यदि (item_ct1.get_local_id(2) % QI8_1 == 0 && (ni == warp_size || item_ct1.get_local_id(2) < ni)) {
        यदि (std::is_same<Tds, sycl::half2>::value) {
            ((sycl::half2  *) yds)[item_ct1.get_local_id(2)/QI8_1] =  make_half2(d, sum);
        } अन्यथा {
            ((sycl::float2 *) yds)[item_ct1.get_local_id(2)/QI8_1] = make_float2(d, sum);
        }
    }
}

प्रकारनिर्देश शून्य (*dequantize_V_t)(अचर शून्य *, शून्य *, अचर int64_t);

ढाँचा <typename T, पूर्णाङ्क ne>
स्थिर __dpct_inline__ शून्य dequantize_V_f16(अचर शून्य * __restrict__ vx, शून्य * __restrict__ dst, अचर int64_t i0) {
    यदि constexpr (std::is_same_v<T, sycl::half>) {
        ggml_sycl_memcpy_1<ne * परिमाण(sycl::half)>(dst, (अचर sycl::half *) vx + i0);
    } अन्यथा यदि constexpr (std::is_same_v<T, भिन्न>) {
        static_assert(ne % 2 == 0, "bad ne");
        sycl::half2 tmp[ne / 2];
        ggml_sycl_memcpy_1<ne * परिमाण(sycl::half)>(tmp, (अचर sycl::half *) vx + i0);
        sycl::float2 * dst_f2 = (sycl::float2 *) dst;
#pragma unroll
        लागि (पूर्णाङ्क l = 0; l < ne/2; ++l) {
            dst_f2[l] = tmp[l].ढाँचा convert<भिन्न, sycl::rounding_mode::automatic>();
        }
    } अन्यथा {
        static_assert(std::is_same_v<T, शून्य>, "unsupported type");
    }
}

ढाँचा <typename T, पूर्णाङ्क ne>
स्थिर __dpct_inline__ शून्य dequantize_V_q4_0(अचर शून्य * __restrict__ vx, शून्य * __restrict__ dst, अचर int64_t i0) {
    अचर block_q4_0 * x = (अचर block_q4_0 *) vx;

    अचर int64_t ib    =  i0          /  QK4_0;
    अचर पूर्णाङ्क     iqs   =  i0          % (QK4_0/2);
    अचर पूर्णाङ्क     shift = (i0 % QK4_0) / (QK4_0/2);

    पूर्णाङ्क q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_sycl_memcpy_1<ne, 2>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;
    q = dpct::vectorized_द्विआधारी<sycl::char4>(q, 0x08080808, dpct::sub_sat());

    अचर int8_t * q8 = (अचर int8_t *) &q;

#ifdef GGML_SYCL_F16
    यदि constexpr (std::is_same_v<T, sycl::half>) {
        अचर sycl::half2 d = sycl::half2(x[ib].d);

#pragma unroll
        लागि (पूर्णाङ्क l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * sycl::half2(q8[l0 + 0], q8[l0 + 1]);
        }
    } अन्यथा
#endif // GGML_SYCL_F16
    यदि constexpr (std::is_same_v<T, भिन्न>) {
        अचर भिन्न d = x[ib].d;

#pragma unroll
        लागि (पूर्णाङ्क l = 0; l < ne; ++l) {
            ((भिन्न *) dst)[l] = d * q8[l];
        }
    } अन्यथा {
        static_assert(std::is_same_v<T, शून्य>, "bad type");
    }
}

ढाँचा <typename T, पूर्णाङ्क ne>
स्थिर __dpct_inline__ शून्य dequantize_V_q4_1(अचर शून्य * __restrict__ vx, शून्य * __restrict__ dst, अचर int64_t i0) {
    अचर block_q4_1 * x = (अचर block_q4_1 *) vx;

    अचर int64_t ib    =  i0          /  QK4_1;
    अचर पूर्णाङ्क     iqs   =  i0          % (QK4_1/2);
    अचर पूर्णाङ्क     shift = (i0 % QK4_1) / (QK4_1/2);

    पूर्णाङ्क q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_sycl_memcpy_1<ne>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    अचर int8_t * q8 = (अचर int8_t *) &q;

#ifdef GGML_SYCL_F16
    यदि constexpr (std::is_same_v<T, sycl::half>) {
        अचर sycl::half2 dm = x[ib].dm;
        अचर sycl::half2 d  = sycl::half2(dm[0]);
        अचर sycl::half2 m  = sycl::half2(dm[1]);

#pragma unroll
        लागि (पूर्णाङ्क l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * sycl::half2(q8[l0 + 0], q8[l0 + 1]) + m;
        }
    } अन्यथा
#endif // GGML_SYCL_F16
    यदि constexpr (std::is_same_v<T, भिन्न>) {
        अचर sycl::float2 dm = (x[ib].dm).ढाँचा convert<भिन्न, sycl::rounding_mode::automatic>();

#pragma unroll
        लागि (पूर्णाङ्क l = 0; l < ne; ++l) {
            ((भिन्न *) dst)[l] = dm.x() * q8[l] + dm.y();
        }
    } अन्यथा {
        static_assert(std::is_same_v<T, शून्य>, "bad type");
    }
}

ढाँचा <typename T, पूर्णाङ्क ne>
स्थिर __dpct_inline__ शून्य dequantize_V_q5_0(अचर शून्य * __restrict__ vx, शून्य * __restrict__ dst, अचर int64_t i0) {
    अचर block_q5_0 * x = (अचर block_q5_0 *) vx;

    अचर int64_t ib    =  i0          /  QK5_0;
    अचर पूर्णाङ्क     idq   =  i0          %  QK5_0;
    अचर पूर्णाङ्क     iqs   =  i0          % (QK5_0/2);
    अचर पूर्णाङ्क     shift = (i0 % QK5_0) / (QK5_0/2);

    पूर्णाङ्क q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_sycl_memcpy_1<ne, 2>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    {
        पूर्णाङ्क qh;
        ggml_sycl_memcpy_1<ne, 2>(&qh, x[ib].qh);
#pragma unroll
        लागि (पूर्णाङ्क l = 0; l < ne; ++l) {
            q |= ((qh >> (idq + l)) & 0x00000001) << (8*l + 4);
        }
    }

    q = dpct::vectorized_द्विआधारी<sycl::char4>(q, 0x10101010, dpct::sub_sat());

    अचर int8_t * q8 = (अचर int8_t *) &q;

#ifdef GGML_SYCL_F16
    यदि constexpr (std::is_same_v<T, sycl::half>) {
        अचर sycl::half2 d = sycl::half2(x[ib].d);

#pragma unroll
        लागि (पूर्णाङ्क l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * sycl::half2(q8[l0 + 0], q8[l0 + 1]);
        }
    } अन्यथा
#endif // GGML_SYCL_F16
    यदि constexpr (std::is_same_v<T, भिन्न>) {
        अचर भिन्न d = x[ib].d;

#pragma unroll
        लागि (पूर्णाङ्क l = 0; l < ne; ++l) {
            ((भिन्न *) dst)[l] = d * q8[l];
        }
    } अन्यथा {
        static_assert(std::is_same_v<T, शून्य>, "bad type");
    }
}

ढाँचा <typename T, पूर्णाङ्क ne>
स्थिर __dpct_inline__ शून्य dequantize_V_q5_1(अचर शून्य * __restrict__ vx, शून्य * __restrict__ dst, अचर int64_t i0) {
    अचर block_q5_1 * x = (अचर block_q5_1 *) vx;

    अचर int64_t ib    =  i0          /  QK5_1;
    अचर पूर्णाङ्क     idq   =  i0          %  QK5_1;
    अचर पूर्णाङ्क     iqs   =  i0          % (QK5_1/2);
    अचर पूर्णाङ्क     shift = (i0 % QK5_1) / (QK5_1/2);

    पूर्णाङ्क q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_sycl_memcpy_1<ne>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    {
        पूर्णाङ्क qh;
        ggml_sycl_memcpy_1<ne>(&qh, x[ib].qh);
#pragma unroll
        लागि (पूर्णाङ्क l = 0; l < ne; ++l) {
            q |= ((qh >> (idq + l)) & 0x00000001) << (8*l + 4);
        }
    }

    अचर int8_t * q8 = (अचर int8_t *) &q;

#ifdef GGML_SYCL_F16
    यदि constexpr (std::is_same_v<T, sycl::half>) {
        अचर sycl::half2 dm = x[ib].dm;
        अचर sycl::half2 d  = sycl::half2(dm[0]);
        अचर sycl::half2 m  = sycl::half2(dm[1]);

#pragma unroll
        लागि (पूर्णाङ्क l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * sycl::half2(q8[l0 + 0], q8[l0 + 1]) + m;
        }
    } अन्यथा
#endif // GGML_SYCL_F16
    यदि constexpr (std::is_same_v<T, भिन्न>) {
        अचर sycl::float2 dm = (x[ib].dm).ढाँचा convert<भिन्न, sycl::rounding_mode::automatic>();

#pragma unroll
        लागि (पूर्णाङ्क l = 0; l < ne; ++l) {
            ((भिन्न *) dst)[l] = dm.x() * q8[l] + dm.y();
        }
    } अन्यथा {
        static_assert(std::is_same_v<T, शून्य>, "bad type");
    }
}

ढाँचा <typename T, पूर्णाङ्क ne>
स्थिर __dpct_inline__ शून्य dequantize_V_q8_0(अचर शून्य * __restrict__ vx, शून्य * __restrict__ dst, अचर int64_t i0) {
    अचर block_q8_0 * x = (अचर block_q8_0 *) vx;

    अचर int64_t ib  = i0 / QK8_0;
    अचर पूर्णाङ्क     iqs = i0 % QK8_0;

    static_assert(ne % 2 == 0, "bad ne");
    int8_t qs[ne];
    ggml_sycl_memcpy_1<ne, 2>(qs, x[ib].qs + iqs);

#ifdef GGML_SYCL_F16
    यदि constexpr (std::is_same<T, sycl::half>::value) {
        अचर sycl::half2 d = sycl::half2(x[ib].d);

#pragma unroll
        लागि (पूर्णाङ्क l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * make_half2(qs[l0 + 0], qs[l0 + 1]);
        }
    } अन्यथा
#endif // GGML_SYCL_F16
    यदि constexpr (std::is_same<T, भिन्न>::value) {
        अचर भिन्न d = x[ib].d;

#pragma unroll
        लागि (पूर्णाङ्क l = 0; l < ne; ++l) {
            ((भिन्न *) dst)[l] = d * qs[l];
        }
    } अन्यथा {
        static_assert(std::is_same_v<T, शून्य>, "unsupported type");
    }
}

ढाँचा <पूर्णाङ्क type_K, पूर्णाङ्क D, पूर्णाङ्क nthreads, पूर्णाङ्क warp_size>
constexpr vec_dot_KQ_t get_vec_dot_KQ() {
    यदि constexpr (type_K == GGML_TYPE_F16) {
        फर्काऊ vec_dot_fattn_vec_KQ_f16<D, nthreads>;
    } अन्यथा यदि constexpr (type_K == GGML_TYPE_Q4_0) {
        फर्काऊ vec_dot_fattn_vec_KQ_q4_0<D, nthreads, warp_size>;
    } अन्यथा यदि constexpr (type_K == GGML_TYPE_Q4_1) {
        फर्काऊ vec_dot_fattn_vec_KQ_q4_1<D, nthreads, warp_size>;
    } अन्यथा यदि constexpr (type_K == GGML_TYPE_Q5_0) {
        फर्काऊ vec_dot_fattn_vec_KQ_q5_0<D, nthreads, warp_size>;
    } अन्यथा यदि constexpr (type_K == GGML_TYPE_Q5_1) {
        फर्काऊ vec_dot_fattn_vec_KQ_q5_1<D, nthreads, warp_size>;
    } अन्यथा यदि constexpr (type_K == GGML_TYPE_Q8_0) {
        फर्काऊ vec_dot_fattn_vec_KQ_q8_0<D, nthreads, warp_size>;
    } अन्यथा {
        static_assert(type_K == -1, "bad type");
        फर्काऊ nullptr;
    }
}

ढाँचा <पूर्णाङ्क type_V, typename T, पूर्णाङ्क ne>
constexpr dequantize_V_t get_dequantize_V() {
    यदि constexpr (type_V == GGML_TYPE_F16) {
        फर्काऊ dequantize_V_f16<T, ne>;
    } अन्यथा यदि constexpr (type_V == GGML_TYPE_Q4_0) {
        फर्काऊ dequantize_V_q4_0<T, ne>;
    } अन्यथा यदि constexpr (type_V == GGML_TYPE_Q4_1) {
        फर्काऊ dequantize_V_q4_1<T, ne>;
    } अन्यथा यदि constexpr (type_V == GGML_TYPE_Q5_0) {
        फर्काऊ dequantize_V_q5_0<T, ne>;
    } अन्यथा यदि constexpr (type_V == GGML_TYPE_Q5_1) {
        फर्काऊ dequantize_V_q5_1<T, ne>;
    } अन्यथा यदि constexpr (type_V == GGML_TYPE_Q8_0) {
        फर्काऊ dequantize_V_q8_0<T, ne>;
    } अन्यथा {
        static_assert(type_V == -1, "bad type");
        फर्काऊ nullptr;
    }
}

ढाँचा <पूर्णाङ्क ncols1, पूर्णाङ्क warp_size>
स्थिर शून्य flash_attn_mask_to_KV_max(अचर sycl::half2 * __restrict__ mask,
                                      पूर्णाङ्क * __restrict__ KV_max,
                                      अचर पूर्णाङ्क ne30,
                                      अचर पूर्णाङ्क s31,
                                      अचर पूर्णाङ्क s33,
                                      पूर्णाङ्क *     buf_iw) {
    स्वतः      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    अचर पूर्णाङ्क ne31     = item_ct1.get_group_range(2);
    अचर पूर्णाङ्क tid      = item_ct1.get_local_id(2);
    अचर पूर्णाङ्क sequence = item_ct1.get_group(1);
    अचर पूर्णाङ्क jt       = item_ct1.get_group(2);

    mask += sequence*s33 + jt*ncols1*s31;

    यदि (tid < warp_size) {
        buf_iw[tid] = 1;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);

    पूर्णाङ्क KV_max_sj = (ne30 - 1) * FATTN_KQ_STRIDE;
    लागि (; KV_max_sj >= 0; KV_max_sj -= FATTN_KQ_STRIDE) {
        पूर्णाङ्क all_inf = 1;

#pragma unroll
        लागि (पूर्णाङ्क j = 0; j < ncols1; ++j) {
            अचर sycl::float2 tmp =
                mask[j * s31 + KV_max_sj / 2 + tid].ढाँचा convert<भिन्न, sycl::rounding_mode::automatic>();
            all_inf = all_inf && पूर्णाङ्क(sycl::isinf((भिन्न) (tmp.x()))) && पूर्णाङ्क(sycl::isinf((भिन्न) (tmp.y())));
        }

        all_inf = warp_reduce_all<warp_size>(all_inf);
        यदि (tid % warp_size == 0) {
            buf_iw[tid / warp_size] = all_inf;
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);
        all_inf = buf_iw[tid % warp_size];
        item_ct1.barrier(sycl::access::fence_space::local_space);
        all_inf = warp_reduce_all<warp_size>(all_inf);

        यदि (!all_inf) {
            रोक;
        }
    }

    // If the break in the loop was not triggered, KV_max_sj is now -FATTN_KQ_STRIDE.
    // If the break was triggered it's the lower edge of the tile with the first non-masked values.
    // In either case, walk back the decrementation by FATTN_KQ_STRIDE.
    KV_max_sj += FATTN_KQ_STRIDE;

    यदि (item_ct1.get_local_id(2) != 0) {
        फर्काऊ;
    }

    KV_max[sequence*ne31 + jt] = KV_max_sj;
}

ढाँचा <पूर्णाङ्क D, पूर्णाङ्क ncols1, पूर्णाङ्क ncols2>  // D == head size

स्थिर शून्य flash_attn_stream_k_fixup(भिन्न * __restrict__ dst,
                                      अचर sycl::float2 * __restrict__ dst_fixup,
                                      अचर पूर्णाङ्क ne01,
                                      अचर पूर्णाङ्क ne02,
                                      अचर पूर्णाङ्क ne03,
                                      अचर पूर्णाङ्क ne11,
                                      अचर पूर्णाङ्क ne12,
                                      अचर पूर्णाङ्क nbatch_fa) {
    स्वतः          item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr पूर्णाङ्क ncols    = ncols1 * ncols2;

    अचर पूर्णाङ्क bidx0 = item_ct1.get_group(2);
    अचर पूर्णाङ्क j     = item_ct1.get_group(1);
    अचर पूर्णाङ्क c     = item_ct1.get_group(0);
    अचर पूर्णाङ्क jc    = j*ncols2 + c;
    अचर पूर्णाङ्क tid   = item_ct1.get_local_id(2);

    अचर भिन्न * dst_fixup_data = ((अचर भिन्न *) dst_fixup) + item_ct1.get_group_range(2) * (2 * 2 * ncols);

    अचर पूर्णाङ्क gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.

    अचर पूर्णाङ्क iter_k     = (ne11      + (nbatch_fa - 1)) / nbatch_fa;
    अचर पूर्णाङ्क iter_j     = (ne01      + (ncols1    - 1)) / ncols1;
    अचर पूर्णाङ्क iter_z_gqa = (gqa_ratio + (ncols2    - 1)) / ncols2;

    अचर पूर्णाङ्क kbc0 = int64_t(bidx0 + 0) * (iter_k * iter_j * iter_z_gqa * ne12 * ne03) / item_ct1.get_group_range(2);
    अचर पूर्णाङ्क kbc0_stop =
        int64_t(bidx0 + 1) * (iter_k * iter_j * iter_z_gqa * ne12 * ne03) / item_ct1.get_group_range(2);

    अचर तार्किक did_not_have_any_data   = kbc0 == kbc0_stop;
    अचर तार्किक wrote_beginning_of_tile = kbc0 % iter_k == 0;
    अचर तार्किक did_not_write_last      = kbc0/iter_k == kbc0_stop/iter_k && kbc0_stop % iter_k != 0;
    यदि (did_not_have_any_data || wrote_beginning_of_tile || did_not_write_last) {
        फर्काऊ;
    }

    // z_KV == K/V head index, zt_gqa = Q head start index per K/V head, jt = token position start index
    अचर पूर्णाङ्क sequence =  kbc0 /(iter_k*iter_j*iter_z_gqa*ne12);
    अचर पूर्णाङ्क z_KV     = (kbc0 - iter_k*iter_j*iter_z_gqa*ne12 * sequence)/(iter_k*iter_j*iter_z_gqa);
    अचर पूर्णाङ्क zt_gqa   = (kbc0 - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV)/(iter_k*iter_j);
    अचर पूर्णाङ्क jt       = (kbc0 - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV - iter_k*iter_j * zt_gqa) / iter_k;

    अचर पूर्णाङ्क zt_Q = z_KV*gqa_ratio + zt_gqa*ncols2; // Global Q head start index.

    यदि (jt*ncols1 + j >= ne01 || zt_gqa*ncols2 + c >= gqa_ratio) {
        फर्काऊ;
    }

    dst += sequence*ne02*ne01*D + jt*ne02*(ncols1*D) + zt_Q*D + (j*ne02 + c)*D + tid;

    // Load the partial result that needs a fixup:
    भिन्न dst_val = 0.0f;
    भिन्न max_val = 0.0f;
    भिन्न rowsum  = 0.0f;
    {
        dst_val = *dst;

        अचर sycl::float2 tmp = dst_fixup[bidx0 * ncols + jc];
        max_val                = tmp.x();
        rowsum                 = tmp.y();
    }

    // Iterate over previous blocks and compute the combined results.
    // All SYCL blocks that get here must have a previous block that needs a fixup.
    पूर्णाङ्क bidx = bidx0 - 1;
    पूर्णाङ्क kbc_stop = kbc0;
    जबसम्म(सत्य) {
        अचर पूर्णाङ्क kbc = int64_t(bidx) * (iter_k * iter_j * iter_z_gqa * ne12 * ne03) / item_ct1.get_group_range(2);
        यदि (kbc == kbc_stop) { // Did not have any data.
            bidx--;
            kbc_stop = kbc;
            जारी;
        }

        अचर भिन्न dst_add = dst_fixup_data[bidx*ncols*D + jc*D + tid];

        अचर sycl::float2 tmp = dst_fixup[(item_ct1.get_group_range(2) + bidx) * ncols + jc];

        // Scale the current and new value accumulators depending on the max. values.
        अचर भिन्न max_val_new = sycl::fmax(max_val, tmp.x());

        अचर भिन्न diff_val = max_val - max_val_new;
        अचर भिन्न diff_add = tmp.x() - max_val_new;

        अचर भिन्न scale_val = diff_val >= SOFTMAX_FTZ_THRESHOLD ? sycl::native::exp(diff_val) : 0.0f;
        अचर भिन्न scale_add = diff_add >= SOFTMAX_FTZ_THRESHOLD ? sycl::native::exp(diff_add) : 0.0f;

        dst_val = scale_val*dst_val + scale_add*dst_add;
        rowsum  = scale_val * rowsum + scale_add * tmp.y();

        max_val = max_val_new;

        // If this block started in a previous tile we are done and don't need to combine additional partial results.
        यदि (kbc % iter_k == 0 || kbc/iter_k < kbc0/iter_k) {
            रोक;
        }
        bidx--;
        kbc_stop = kbc;
    }

    // Write back final result:
    *dst = dst_val / rowsum;
}

ढाँचा <पूर्णाङ्क D>  // D == head size

स्थिर शून्य flash_attn_combine_results(अचर भिन्न * __restrict__ VKQ_parts,
                                       अचर sycl::float2 * __restrict__ VKQ_meta,
                                       भिन्न * __restrict__ dst,
                                       अचर पूर्णाङ्क parallel_blocks,
                                       uint8_t * dpct_local) {
    // Dimension 0: threadIdx.x
    // Dimension 1: blockIdx.x
    // Dimension 2: blockIdx.y
    // Dimension 3: blockIdx.z
    // Memory layout is permuted with [0, 2, 1, 3]

    स्वतः      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    अचर पूर्णाङ्क ne01     = item_ct1.get_group_range(2);
    अचर पूर्णाङ्क ne02     = item_ct1.get_group_range(1);

    अचर पूर्णाङ्क col      = item_ct1.get_group(2);
    अचर पूर्णाङ्क head     = item_ct1.get_group(1);
    अचर पूर्णाङ्क sequence = item_ct1.get_group(0);

    अचर पूर्णाङ्क j_dst_unrolled = (sequence*ne01 + col)*ne02 + head;

    VKQ_parts += j_dst_unrolled * parallel_blocks*D;
    VKQ_meta  += j_dst_unrolled * parallel_blocks;
    dst       += j_dst_unrolled *                 D;

    अचर पूर्णाङ्क tid = item_ct1.get_local_id(2);
    __builtin_assume(tid < D);

    स्वतः meta = (sycl::float2 *) dpct_local;
    लागि (पूर्णाङ्क i = tid; i < 2*parallel_blocks; i += D) {
        ((भिन्न *) meta)[i] = ((अचर भिन्न *)VKQ_meta) [i];
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    भिन्न kqmax = meta[0].x();
    लागि (पूर्णाङ्क l = 1; l < parallel_blocks; ++l) {
        kqmax = sycl::max(kqmax, meta[l].x());
    }

    भिन्न VKQ_numerator   = 0.0f;
    भिन्न VKQ_denominator = 0.0f;
    लागि (पूर्णाङ्क l = 0; l < parallel_blocks; ++l) {
        अचर भिन्न KQ_max_scale = sycl::native::exp(meta[l].x() - kqmax);

        VKQ_numerator   += KQ_max_scale * VKQ_parts[l*D + tid];
        VKQ_denominator += KQ_max_scale * meta[l].y();
    }

    dst[tid] = VKQ_numerator / VKQ_denominator;
}

// Carries the sub-group size (and grf_size_automatic when use_large_grf) as
// kernel properties via get(properties_tag), in place of the
// reqd_sub_group_size attribute and the deprecated parallel_for overload.
ढाँचा <typename KernelFunc, uint32_t Size, तार्किक UseGrf>
संरचना props_kernel {
    KernelFunc kernel;

    शून्य चालक()(sycl::nd_item<3> item) अचर {
        kernel(item);
    }

    स्वतः get(sycl::ext::oneapi::experimental::properties_tag) अचर {
        यदि constexpr (UseGrf) {
            फर्काऊ sycl::ext::oneapi::experimental::properties{
                sycl::ext::intel::experimental::grf_size_automatic,
                sycl::ext::oneapi::experimental::sub_group_size<Size> };
        } अन्यथा {
            फर्काऊ sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::sub_group_size<Size> };
        }
    }
};

ढाँचा <fattn_kernel_t fattn_kernel, पूर्णाङ्क warp_size, तार्किक use_large_grf = असत्य>
स्थिर शून्य lauch_kernel(
    dpct::dim3 group_range,
    dpct::dim3 local_range,
    queue_ptr q,
    अचिन्हित पूर्णाङ्क local_mem_size,
    अचर वर्ण* __restrict__ Q,
    अचर वर्ण* __restrict__ K,
    अचर वर्ण* __restrict__ V,
    अचर वर्ण* __restrict__ mask,
    अचर वर्ण* __restrict__ sinks,
    अचर पूर्णाङ्क* __restrict__ KV_max,
    भिन्न* __restrict__ dst,
    sycl::float2* __restrict__ dst_meta,
    अचर भिन्न scale,
    अचर भिन्न max_bias,
    अचर भिन्न m0,
    अचर भिन्न m1,
    अचर uint32_t n_head_log2,
    अचर भिन्न logit_softcap,
    अचर int32_t ne00,
    अचर sycl::uint3 ne01,
    अचर int32_t ne02,
    अचर int32_t ne03,
    अचर int32_t nb01,
    अचर int32_t nb02,
    अचर int32_t nb03,
    अचर int32_t ne10,
    अचर int32_t ne11,
    अचर int32_t ne12,
    अचर int32_t ne13,
    अचर int32_t nb11,
    अचर int32_t nb12,
    अचर int64_t nb13,
    अचर int32_t nb21,
    अचर int32_t nb22,
    अचर int64_t nb23,
    अचर int32_t ne31,
    अचर int32_t ne32,
    अचर int32_t ne33,
    अचर int32_t nb31,
    अचर int32_t nb32,
    अचर int64_t nb33) {
    GGML_UNUSED(local_mem_size);

    अचर स्वतः rng = sycl::nd_range<3>(
        static_cast<sycl::range<3>>(group_range * local_range),
        static_cast<sycl::range<3>>(local_range));

    अचर स्वतः kernel = [=](sycl::nd_item<3> item_ct1) {
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
ढाँचा <पूर्णाङ्क DV, पूर्णाङ्क ncols1, पूर्णाङ्क ncols2, fattn_kernel_t fattn_kernel, पूर्णाङ्क warp_size, तार्किक use_large_grf = असत्य>
शून्य launch_fattn(
    ggml_backend_sycl_context & ctx, ggml_tensor * dst, अचर पूर्णाङ्क nwarps, अचर माप_प्रकार nbytes_shared,
    अचर पूर्णाङ्क nbatch_fa, अचर तार्किक need_f16_K, अचर तार्किक need_f16_V, अचर तार्किक stream_k) {

    constexpr पूर्णाङ्क ncols = ncols1 * ncols2;

    अचर ggml_tensor * Q = dst->src[0];
    अचर ggml_tensor * K = dst->src[1];
    अचर ggml_tensor * V = dst->src[2];

    अचर तार्किक V_is_K_view = V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));

    अचर ggml_tensor * mask  = dst->src[3];
    अचर ggml_tensor * sinks = dst->src[4];

    ggml_tensor * KQV = dst;

    GGML_ASSERT(Q->type == GGML_TYPE_F32);
    GGML_ASSERT(KQV->type == GGML_TYPE_F32);

    GGML_ASSERT(Q->nb[0] == ggml_element_size(Q));
    GGML_ASSERT(K->nb[0] == ggml_element_size(K));
    GGML_ASSERT(V->nb[0] == ggml_element_size(V));

    GGML_ASSERT(!mask || mask->type == GGML_TYPE_F16);

    ggml_sycl_pool & pool = ctx.pool();
    dpct::queue_ptr  main_stream = ctx.stream();
    अचर पूर्णाङ्क id  = ggml_sycl_get_device();
    अचर पूर्णाङ्क nsm = ggml_sycl_info().devices[id].nsm;

    अचर ggml_sycl_fattn_extra  extra = ggml_sycl_fattn_get_extra(dst);
    ggml_sycl_pool_alloc<पूर्णाङ्क>    KV_max(pool);
    ggml_sycl_pool_alloc<भिन्न>  dst_tmp(pool);
    ggml_sycl_pool_alloc<sycl::float2> dst_tmp_meta(pool);

    अचर वर्ण * K_data = (अचर वर्ण *) K->data;
    माप_प्रकार nb11 = K->nb[1];
    माप_प्रकार nb12 = K->nb[2];
    माप_प्रकार nb13 = K->nb[3];

    अचर वर्ण * V_data = (अचर वर्ण *) V->data;
    माप_प्रकार nb21 = V->nb[1];
    माप_प्रकार nb22 = V->nb[2];
    माप_प्रकार nb23 = V->nb[3];

    यदि (need_f16_K && K->type != GGML_TYPE_F16) {
        अचर माप_प्रकार bs = ggml_blck_size(K->type);
        अचर माप_प्रकार ts = ggml_type_size(K->type);

        GGML_ASSERT(extra.K_buffer_ptr);
        sycl::half * K_f16_ptr = (sycl::half *) extra.K_buffer_ptr;
        यदि (ggml_is_contiguously_allocated(K)) {
            to_fp16_sycl_t to_fp16 = ggml_get_to_fp16_sycl(K->type, dst);
            to_fp16(K_data, K_f16_ptr, ggml_nelements(K), main_stream);

            nb11 = nb11 * bs * परिमाण(sycl::half) / ts;
            nb12 = nb12 * bs * परिमाण(sycl::half) / ts;
            nb13 = nb13 * bs * परिमाण(sycl::half) / ts;
        } अन्यथा {
            GGML_ASSERT(K->nb[0] == ts);
            to_fp16_nc_sycl_t to_fp16 = ggml_get_to_fp16_nc_sycl(K->type);
            अचर int64_t s01 = nb11 / ts;
            अचर int64_t s02 = nb12 / ts;
            अचर int64_t s03 = nb13 / ts;
            to_fp16(K_data, K_f16_ptr, K->ne[0], K->ne[1], K->ne[2], K->ne[3], s01, s02, s03, main_stream);

            nb11 = K->ne[0] * परिमाण(sycl::half);
            nb12 = K->ne[1] * nb11;
            nb13 = K->ne[2] * nb12;
        }
        K_data = (वर्ण *) K_f16_ptr;
    }

    यदि (need_f16_V && V->type != GGML_TYPE_F16) {
        यदि (V_is_K_view) {
            V_data = K_data;
            nb21   = nb11;
            nb22   = nb12;
            nb23   = nb13;
        } अन्यथा {
            अचर माप_प्रकार bs = ggml_blck_size(V->type);
            अचर माप_प्रकार ts = ggml_type_size(V->type);

            GGML_ASSERT(extra.V_buffer_ptr);
            sycl::half * V_f16_ptr = (sycl::half *) extra.V_buffer_ptr;
            यदि (ggml_is_contiguously_allocated(V)) {
                to_fp16_sycl_t to_fp16 = ggml_get_to_fp16_sycl(V->type, dst);
                to_fp16(V_data, V_f16_ptr, ggml_nelements(V), main_stream);
                V_data = (वर्ण *) V_f16_ptr;

                nb21 = nb21 * bs * परिमाण(sycl::half) / ts;
                nb22 = nb22 * bs * परिमाण(sycl::half) / ts;
                nb23 = nb23 * bs * परिमाण(sycl::half) / ts;
            } अन्यथा {
                GGML_ASSERT(V->nb[0] == ts);
                to_fp16_nc_sycl_t to_fp16 = ggml_get_to_fp16_nc_sycl(V->type);
                अचर int64_t s01 = nb21 / ts;
                अचर int64_t s02 = nb22 / ts;
                अचर int64_t s03 = nb23 / ts;
                to_fp16(V_data, V_f16_ptr, V->ne[0], V->ne[1], V->ne[2], V->ne[3], s01, s02, s03, main_stream);

                nb21 = V->ne[0] * परिमाण(sycl::half);
                nb22 = V->ne[1] * nb21;
                nb23 = V->ne[2] * nb22;
            }
            V_data = (वर्ण *) V_f16_ptr;
        }
    }

    अचर पूर्णाङ्क ntiles_x     = ((Q->ne[1] + ncols1 - 1) / ncols1);
    अचर पूर्णाङ्क gqa_ratio    = Q->ne[2] / K->ne[2];
    अचर पूर्णाङ्क ntiles_z_gqa = ((gqa_ratio + ncols2 - 1) / ncols2);
    अचर पूर्णाङ्क ntiles_total = ntiles_x * ntiles_z_gqa * K->ne[2] * Q->ne[3];

    // Optional optimization where the mask is scanned to determine whether part of the calculation can be skipped.
    // Only worth the overhead if there is at lease one FATTN_KQ_STRIDE x FATTN_KQ_STRIDE square to be skipped or
    //     multiple sequences of possibly different lengths.
    यदि (mask && K->ne[1] % FATTN_KQ_STRIDE == 0 && (Q->ne[1] >= 1024 || Q->ne[3] > 1)) {
        अचर पूर्णाङ्क s31 = mask->nb[1] / परिमाण(sycl::half2);
        अचर पूर्णाङ्क s33 = mask->nb[3] / परिमाण(sycl::half2);

        अचर dpct::dim3 blocks_अङ्क_KV_max(ntiles_x, Q->ne[3], 1);
        अचर dpct::dim3 block_dim_KV_max(FATTN_KQ_STRIDE / 2, 1, 1);

        अचर पूर्णाङ्क ne_KV_max = blocks_अङ्क_KV_max.x*blocks_अङ्क_KV_max.y;
        अचर पूर्णाङ्क iter_k = K->ne[1] / FATTN_KQ_STRIDE;

        KV_max.alloc(ne_KV_max);
        {
            dpct::has_capability_or_fail(main_stream->get_device(), { sycl::aspect::fp16 });

            main_stream->submit([&](sycl::handler & cgh) {
                sycl::local_accessor<पूर्णाङ्क, 1> buf_iw_acc_ct1(sycl::range<1>(warp_size), cgh);

                स्वतः mask_data_ct0  = (अचर sycl::half2 *) mask->data;
                स्वतः KV_max_ptr_ct1 = KV_max.ptr;

                cgh.parallel_for(sycl::nd_range<3>(blocks_अङ्क_KV_max * block_dim_KV_max, block_dim_KV_max),
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

    अचर dpct::dim3 block_dim(warp_size, nwarps, 1);

    // Max. number of active blocks limited by occupancy.
    पूर्णाङ्क max_blocks_per_sm = ggml_sycl_info().devices[id].max_wg_per_cu;
    पूर्णाङ्क parallel_blocks = max_blocks_per_sm;
    dpct::dim3 blocks_अङ्क;
    यदि (stream_k) {
        // For short contexts it can be faster to have the SMs work on whole tiles because this lets us skip the fixup.
        अचर पूर्णाङ्क max_blocks = max_blocks_per_sm*nsm;
        अचर पूर्णाङ्क nblocks_stream_k = max_blocks;
        अचर तार्किक use_stream_k = सत्य;

        blocks_अङ्क.x = use_stream_k ? nblocks_stream_k : ntiles_total;
        blocks_अङ्क.y = 1;
        blocks_अङ्क.z = 1;

        यदि (ntiles_total % blocks_अङ्क.x != 0) { // Fixup is only needed if the SMs work on fractional tiles.
            dst_tmp_meta.alloc((माप_प्रकार(blocks_अङ्क.x) * ncols * (2 + DV/2)));
        }
    } अन्यथा {
        अचर पूर्णाङ्क ntiles_KQ = (K->ne[1] + nbatch_fa - 1) / nbatch_fa; // Max. number of parallel blocks limited by tensor size.

        // parallel_blocks must not be larger than what the tensor size allows:
        parallel_blocks = std::min(parallel_blocks, ntiles_KQ);
        // todo fix the hard code change
        // parallel_blocks = ntiles_KQ;

        // If ntiles_total % blocks_per_wave != 0 then some efficiency is lost due to tail effects.
        // Test whether parallel_blocks can be set to a higher value for better efficiency.
        अचर पूर्णाङ्क blocks_per_wave = nsm * max_blocks_per_sm;
        पूर्णाङ्क nwaves_best = 0;
        पूर्णाङ्क efficiency_percent_best = 0;
        लागि (पूर्णाङ्क parallel_blocks_test = parallel_blocks; parallel_blocks_test <= ntiles_KQ; ++parallel_blocks_test) {
            अचर पूर्णाङ्क nblocks_total = ntiles_total * parallel_blocks_test;
            अचर पूर्णाङ्क nwaves = (nblocks_total + blocks_per_wave - 1) / blocks_per_wave;
            अचर पूर्णाङ्क efficiency_percent = 100 * nblocks_total / (nwaves*blocks_per_wave);

            // Stop trying configurations with more waves if we already have good efficiency to avoid excessive overhead.
            यदि (efficiency_percent_best >= 95 && nwaves > nwaves_best) {
                रोक;
            }

            यदि (efficiency_percent > efficiency_percent_best) {
                nwaves_best = nwaves;
                efficiency_percent_best = efficiency_percent;
                parallel_blocks = parallel_blocks_test;
            }
        }

        blocks_अङ्क.x = ntiles_x;
        blocks_अङ्क.y = parallel_blocks;
        blocks_अङ्क.z = ntiles_z_gqa*K->ne[2]*Q->ne[3];

        यदि (parallel_blocks > 1) {
            dst_tmp.alloc(parallel_blocks*ggml_nelements(KQV));
            dst_tmp_meta.alloc(parallel_blocks*ggml_nrows(KQV));
        }
    }

    भिन्न scale         = 1.0f;
    भिन्न max_bias      = 0.0f;
    भिन्न logit_softcap = 0.0f;

    स_नकल(&scale,         (अचर भिन्न *) KQV->op_params + 0, परिमाण(भिन्न));
    स_नकल(&max_bias,      (अचर भिन्न *) KQV->op_params + 1, परिमाण(भिन्न));
    स_नकल(&logit_softcap, (अचर भिन्न *) KQV->op_params + 2, परिमाण(भिन्न));

    यदि (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    अचर uint32_t n_head      = Q->ne[2];
    अचर uint32_t n_head_log2 = 1u << uint32_t(floorf(log2f(भिन्न(n_head))));

    अचर भिन्न m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    अचर भिन्न m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    // TODO other tensor dimensions after removal of WMMA kernel:
    अचर sycl::uint3 ne01 = init_fastdiv_मानहरू(Q->ne[1]);

    GGML_ASSERT(block_dim.x % warp_size == 0);

    lauch_kernel<fattn_kernel, warp_size, use_large_grf>(
        blocks_अङ्क, block_dim, main_stream, (अचिन्हित पूर्णाङ्क) nbytes_shared, (अचर वर्ण *) Q->data, K_data, V_data,
        mask ? ((अचर वर्ण *) mask->data) : nullptr, sinks ? ((अचर वर्ण *) sinks->data) : nullptr, KV_max.ptr,
        !stream_k && parallel_blocks > 1 ? dst_tmp.ptr : (भिन्न *) KQV->data, (sycl::float2 *)dst_tmp_meta.ptr, scale, max_bias, m0, m1,
        n_head_log2, logit_softcap, Q->ne[0], ne01, Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3], K->ne[0],
        K->ne[1], K->ne[2], K->ne[3], nb11, nb12, nb13, nb21, nb22, nb23, mask ? mask->ne[1] : 0,
        mask ? mask->ne[2] : 0, mask ? mask->ne[3] : 0, mask ? mask->nb[1] : 0, mask ? mask->nb[2] : 0,
        mask ? mask->nb[3] : 0);
    SYCL_CHECK(0);

    यदि (stream_k) {
        यदि (ntiles_total % blocks_अङ्क.x != 0) { // Fixup is only needed if the SMs work on fractional tiles.
            अचर dpct::dim3 block_dim_combine(DV, 1, 1);
            अचर dpct::dim3 blocks_अङ्क_combine = { blocks_अङ्क.x, ncols1, ncols2 };

            main_stream->submit([&](sycl::handler & cgh) {
                स्वतः KQV_data_ct0         = (भिन्न *) KQV->data;
                स्वतः dst_tmp_meta_ptr_ct1 = dst_tmp_meta.ptr;
                स्वतः Q_ne_ct2             = Q->ne[1];
                स्वतः Q_ne_ct3             = Q->ne[2];
                स्वतः Q_ne_ct4             = Q->ne[3];
                स्वतः K_ne_ct5             = K->ne[1];
                स्वतः K_ne_ct6             = K->ne[2];

                cgh.parallel_for(sycl::nd_range<3>(blocks_अङ्क_combine * block_dim_combine, block_dim_combine),
                                 [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(warp_size)]] {
                                     GGML_UNUSED(item_ct1);
                                     flash_attn_stream_k_fixup<DV, ncols1, ncols2>(KQV_data_ct0, dst_tmp_meta_ptr_ct1,
                                                                                   Q_ne_ct2, Q_ne_ct3, Q_ne_ct4,
                                                                                   K_ne_ct5, K_ne_ct6, nbatch_fa);
                                 });
            });
        }
    } अन्यथा यदि (parallel_blocks > 1) {
        अचर dpct::dim3 block_dim_combine(DV, 1, 1);
        अचर dpct::dim3 blocks_अङ्क_combine(Q->ne[1], Q->ne[2], Q->ne[3]);
        अचर माप_प्रकार     nbytes_shared_combine = parallel_blocks * परिमाण(sycl::float2);
        main_stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(sycl::range<1>(nbytes_shared_combine), cgh);

            स्वतः dst_tmp_ptr_ct0      = dst_tmp.ptr;
            स्वतः dst_tmp_meta_ptr_ct1 = dst_tmp_meta.ptr;
            स्वतः KQV_data_ct2         = (भिन्न *) KQV->data;

            cgh.parallel_for(sycl::nd_range<3>(blocks_अङ्क_combine * block_dim_combine, block_dim_combine),
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
