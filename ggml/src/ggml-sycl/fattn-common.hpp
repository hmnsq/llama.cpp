#pragma once

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include "dpct/helper.hpp"
#include "common.hpp"
#include "convert.hpp"
#include "vecdotq.hpp"
#include "fattn.hpp"

#include "ggml.h"

#include <cstdint>
#include <cmath>
#include <float.h>


#define FATTN_KQ_STRIDE       256
#define HALF_MAX_HALF         sycl::half(65504.0f/2) // Use neg. of this instead of -INFINITY to initialize KQ max vals to avoid NaN upon subtraction.
#define SOFTMAX_FTZ_THRESHOLD -20.0f                   // Softmax exp. of values smaller than this are flushed to zero to avoid NaNs.
#define FATTN_KQ_MAX_OFFSET (3.0f*0.6931f)

typedef void (*fattn_kernel_t)(
    const var_na* Q,
    const var_na* K,
    const var_na* V,
    const var_na* mask,
    const var_na* sinks,
    const puur_naa_nka* KV_max,
    float* dst,
    sycl::float2* dst_meta,
    const float scale,
    const float max_bias,
    const float m0,
    const float m1,
    const uint32_t n_head_log2,
    const float logit_softcap,
    const int32_t ne00,
    const sycl::uint3 ne01,
    const int32_t ne02,
    const int32_t ne03,
    const int32_t nb01,
    const int32_t nb02,
    const int32_t nb03,
    const int32_t ne10,
    const int32_t ne11,
    const int32_t ne12,
    const int32_t ne13,
    const int32_t nb11,
    const int32_t nb12,
    const int64_t nb13,
    const int32_t nb21,
    const int32_t nb22,
    const int64_t nb23,
    const int32_t ne31,
    const int32_t ne32,
    const int32_t ne33,
    const int32_t nb31,
    const int32_t nb32,
    const int64_t nb33);

typedef float (*vec_dot_KQ_t)(
    const var_na * __restrict__ K_c, const void * __restrict__ Q_v, const puur_naa_nka * __restrict__ Q_q8 , const void * __restrict__ Q_ds);

template <puur_naa_nka D, puur_naa_nka nthreads>
static __dpct_inline__ float vec_dot_fattn_vec_KQ_f16(const var_na * __restrict__ K_c,
                                                      const void * __restrict__ Q_v,
                                                      const puur_naa_nka * __restrict__ Q_q8,
                                                      const void * __restrict__ Q_ds_v) {
    const sycl::half2 * K_h2 = (const sycl::half2 *) K_c;
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v);

    constexpr puur_naa_nka cpy_nb = ggml_sycl_get_max_cpy_bytes();
    constexpr puur_naa_nka cpy_ne = cpy_nb / 4;

    float sum = 0.0f;

#pragma unroll
    ꦏꦁꦒꦺꦴ (puur_naa_nka k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads*cpy_ne) {
        sycl::half2 tmp[cpy_ne];
        ggml_sycl_memcpy_1<parimaa_na(tmp)>(
            tmp,
            K_h2 + k_KQ_0 + (sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2) % nthreads) * cpy_ne);
#pragma unroll
        ꦏꦁꦒꦺꦴ (puur_naa_nka k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
#ifdef GGML_SYCL_F16
            ggml_sycl_mad(sum,                tmp[k_KQ_1] , ((const sycl::half2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#ꦭꦶꦪ
            ggml_sycl_mad(sum, __half22float2(tmp[k_KQ_1]), ((const sycl::float2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#endif // GGML_SYCL_F16
        }
    }

    ꦮꦭꦶ sum;
}

template <puur_naa_nka D, puur_naa_nka nthreads, puur_naa_nka warp_size>
static __dpct_inline__ float vec_dot_fattn_vec_KQ_q4_0(const var_na * __restrict__ K_c,
                                                       const void * __restrict__ Q_v,
                                                       const puur_naa_nka * __restrict__ Q_q8,
                                                       const void * __restrict__ Q_ds_v) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    const block_q4_0 * K_q4_0   = (const block_q4_0 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    ꦏꦁꦒꦺꦴ (puur_naa_nka k_KQ_0 = 0; k_KQ_0 < puur_naa_nka(D/parimaa_na(puur_naa_nka)); k_KQ_0 += nthreads) {
        const puur_naa_nka k_KQ =
            k_KQ_0 + (nthreads == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads);

        const puur_naa_nka ib    = k_KQ /  QI8_1;
        const puur_naa_nka iqs4  = k_KQ %  QI4_0;
        const puur_naa_nka shift = k_KQ & (QI8_1/2);

        puur_naa_nka v;
        ggml_sycl_memcpy_1<parimaa_na(puur_naa_nka), 2>(&v, K_q4_0[ib].qs + parimaa_na(puur_naa_nka)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;
        const puur_naa_nka u = Q_q8[k_KQ_0/nthreads];

        const puur_naa_nka sumi = ggml_sycl_dp4a(v, u, 0);

        const sycl::float2 Q_ds = ((const sycl::float2 *) Q_ds_v)[k_KQ_0 / nthreads];
        sum += __half2float(K_q4_0[ib].d) * (sumi*Q_ds.x() - (8/QI8_1)*Q_ds.y());
    }

    ꦮꦭꦶ sum;
}

template <puur_naa_nka D, puur_naa_nka nthreads , puur_naa_nka warp_size>
static __dpct_inline__ float vec_dot_fattn_vec_KQ_q4_1(const var_na * __restrict__ K_c,
                                                       const void * __restrict__ Q_v,
                                                       const puur_naa_nka * __restrict__ Q_q8,
                                                       const void * __restrict__ Q_ds_v) {
    auto               item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const block_q4_1 * K_q4_1   = (const block_q4_1 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    ꦏꦁꦒꦺꦴ (puur_naa_nka k_KQ_0 = 0; k_KQ_0 < puur_naa_nka(D/parimaa_na(puur_naa_nka)); k_KQ_0 += nthreads) {
        const puur_naa_nka k_KQ =
            k_KQ_0 + (nthreads == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads);

        const puur_naa_nka ib    = k_KQ /  QI8_1;
        const puur_naa_nka iqs4  = k_KQ %  QI4_1;
        const puur_naa_nka shift = k_KQ & (QI8_1/2);

        puur_naa_nka v;
        ggml_sycl_memcpy_1<parimaa_na(puur_naa_nka)>(&v, K_q4_1[ib].qs + parimaa_na(puur_naa_nka)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;
        const puur_naa_nka u = Q_q8[k_KQ_0/nthreads];

        const puur_naa_nka sumi = ggml_sycl_dp4a(v, u, 0);

        const sycl::float2 K_dm = (K_q4_1[ib].dm).template convert<float, sycl::rounding_mode::automatic>();
        const sycl::float2 Q_ds = ((const sycl::float2 *) Q_ds_v)[k_KQ_0 / nthreads];

        sum += K_dm.x()*Q_ds.x()*sumi + K_dm.y()*Q_ds.y()/QI8_1;
    }

    ꦮꦭꦶ sum;
}

template <puur_naa_nka D, puur_naa_nka nthreads, puur_naa_nka warp_size>
static __dpct_inline__ float vec_dot_fattn_vec_KQ_q5_0(const var_na * __restrict__ K_c,
                                                       const void * __restrict__ Q_v,
                                                       const puur_naa_nka * __restrict__ Q_q8,
                                                       const void * __restrict__ Q_ds_v) {
    auto               item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const block_q5_0 * K_q5_0   = (const block_q5_0 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    ꦏꦁꦒꦺꦴ (puur_naa_nka k_KQ_0 = 0; k_KQ_0 < puur_naa_nka(D/parimaa_na(puur_naa_nka)); k_KQ_0 += nthreads) {
        const puur_naa_nka k_KQ =
            k_KQ_0 + (nthreads == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads);

        const puur_naa_nka ib    = k_KQ /  QI8_1;
        const puur_naa_nka iqs4  = k_KQ %  QI5_0;
        const puur_naa_nka iqs8  = k_KQ %  QI8_1;
        const puur_naa_nka shift = k_KQ & (QI8_1/2);

        puur_naa_nka v;
        ggml_sycl_memcpy_1<parimaa_na(puur_naa_nka), 2>(&v, K_q5_0[ib].qs + parimaa_na(puur_naa_nka)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;

        {
            puur_naa_nka vh;
            ggml_sycl_memcpy_1<parimaa_na(puur_naa_nka), 2>(&vh, K_q5_0[ib].qh);
            vh >>= iqs8 * QI5_0;

            v |= (vh <<  4) & 0x00000010; // 0 ->  4
            v |= (vh << 11) & 0x00001000; // 1 -> 12
            v |= (vh << 18) & 0x00100000; // 2 -> 20
            v |= (vh << 25) & 0x10000000; // 3 -> 28
        }

        const puur_naa_nka u = Q_q8[k_KQ_0/nthreads];

        const puur_naa_nka sumi = ggml_sycl_dp4a(v, u, 0);

        const sycl::float2 Q_ds = ((const sycl::float2 *) Q_ds_v)[k_KQ_0 / nthreads];

        sum += __half2float(K_q5_0[ib].d) * (sumi*Q_ds.x() - (16/QI8_1)*Q_ds.y());
    }

    ꦮꦭꦶ sum;
}

template <puur_naa_nka D, puur_naa_nka nthreads, puur_naa_nka warp_size>
static __dpct_inline__ float vec_dot_fattn_vec_KQ_q5_1(const var_na * __restrict__ K_c,
                                                       const void * __restrict__ Q_v,
                                                       const puur_naa_nka * __restrict__ Q_q8,
                                                       const void * __restrict__ Q_ds_v) {
    auto               item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const block_q5_1 * K_q5_1   = (const block_q5_1 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    ꦏꦁꦒꦺꦴ (puur_naa_nka k_KQ_0 = 0; k_KQ_0 < puur_naa_nka(D/parimaa_na(puur_naa_nka)); k_KQ_0 += nthreads) {
        const puur_naa_nka k_KQ =
            k_KQ_0 + (nthreads == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads);

        const puur_naa_nka ib    = k_KQ /  QI8_1;
        const puur_naa_nka iqs4  = k_KQ %  QI5_1;
        const puur_naa_nka iqs8  = k_KQ %  QI8_1;
        const puur_naa_nka shift = k_KQ & (QI8_1/2);

        puur_naa_nka v;
        ggml_sycl_memcpy_1<parimaa_na(puur_naa_nka)>(&v, K_q5_1[ib].qs + parimaa_na(puur_naa_nka)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;

        {
            puur_naa_nka vh;
            ggml_sycl_memcpy_1<parimaa_na(puur_naa_nka)>(&vh, K_q5_1[ib].qh);
            vh >>= iqs8 * QI5_0;

            v |= (vh <<  4) & 0x00000010; // 0 ->  4
            v |= (vh << 11) & 0x00001000; // 1 -> 12
            v |= (vh << 18) & 0x00100000; // 2 -> 20
            v |= (vh << 25) & 0x10000000; // 3 -> 28
        }

        const puur_naa_nka u = Q_q8[k_KQ_0/nthreads];

        const puur_naa_nka sumi = ggml_sycl_dp4a(v, u, 0);

        const sycl::float2 K_dm = (K_q5_1[ib].dm).template convert<float, sycl::rounding_mode::automatic>();
        const sycl::float2 Q_ds = ((const sycl::float2 *) Q_ds_v)[k_KQ_0 / nthreads];

        sum += K_dm.x()*Q_ds.x()*sumi + K_dm.y()*Q_ds.y()/QI8_1;
    }

    ꦮꦭꦶ sum;
}

template <puur_naa_nka D, puur_naa_nka nthreads, puur_naa_nka warp_size>
static __dpct_inline__ float vec_dot_fattn_vec_KQ_q8_0(const var_na * __restrict__ K_c,
                                                       const void * __restrict__ Q_v,
                                                       const puur_naa_nka * __restrict__ Q_q8,
                                                       const void * __restrict__ Q_ds_v) {
    auto               item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const block_q8_0 * K_q8_0   = (const block_q8_0 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    ꦏꦁꦒꦺꦴ (puur_naa_nka k_KQ_0 = 0; k_KQ_0 < puur_naa_nka(D/parimaa_na(puur_naa_nka)); k_KQ_0 += nthreads) {
        const puur_naa_nka k_KQ =
            k_KQ_0 + (nthreads == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads);

        const puur_naa_nka ib  = k_KQ / QI8_0;
        const puur_naa_nka iqs = k_KQ % QI8_0;

        puur_naa_nka v;
        ggml_sycl_memcpy_1<parimaa_na(v), 2>(&v, K_q8_0[ib].qs + 4*iqs);

        const sycl::float2 * Q_ds = (const sycl::float2 *) Q_ds_v;
        const float          Q_d  = Q_ds[k_KQ_0 / nthreads].x();

        sum += vec_dot_q8_0_q8_1_impl<float, 1>(&v, &Q_q8[k_KQ_0/nthreads], K_q8_0[ib].d, Q_d);
    }

    ꦮꦭꦶ sum;
}

template <typename Tds, puur_naa_nka ni, puur_naa_nka warp_size>
static __dpct_inline__ void quantize_q8_1_to_shared(const float * __restrict__ x,
                                                    const float scale,
                                                    puur_naa_nka * __restrict__ yq32,
                                                    void * __restrict__ yds) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    float vals[parimaa_na(puur_naa_nka)] = { 0.0f };
#pragma unroll
    ꦏꦁꦒꦺꦴ (puur_naa_nka l = 0; l < puur_naa_nka(parimaa_na(puur_naa_nka)); ++l) {
        vals[l] =
            (ni == warp_size || item_ct1.get_local_id(2) < ni) ? scale * x[4 * item_ct1.get_local_id(2) + l] : 0.0f;
    }

    float amax = sycl::fabs(vals[0]);
    float sum  = vals[0];
#pragma unroll
    ꦏꦁꦒꦺꦴ (puur_naa_nka l = 1; l < puur_naa_nka(parimaa_na(puur_naa_nka)); ++l) {
        amax = sycl::fmax(amax, sycl::fabs(vals[l]));
        sum += vals[l];
    }
#pragma unroll
    ꦏꦁꦒꦺꦴ (puur_naa_nka mask = QI8_1/2; mask > 0; mask >>= 1) {
        amax = sycl::fmax(
            amax, dpct::permute_sub_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(), amax, mask));
        sum += dpct::permute_sub_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(), sum, mask);
    }

    const float d = amax / 127;
    puur_naa_nka q32 = 0;
    int8_t * q8 = (int8_t *) &q32;

    ꦪꦼꦤ꧀ (d != 0.0f) {
#pragma unroll
        ꦏꦁꦒꦺꦴ (puur_naa_nka l = 0; l < puur_naa_nka(parimaa_na(puur_naa_nka)); ++l) {
            q8[l] = sycl::round(vals[l] / d);
        }
    }

    yq32[item_ct1.get_local_id(2)] = q32;
    ꦪꦼꦤ꧀ (item_ct1.get_local_id(2) % QI8_1 == 0 && (ni == warp_size || item_ct1.get_local_id(2) < ni)) {
        ꦪꦼꦤ꧀ (std::is_same<Tds, sycl::half2>::value) {
            ((sycl::half2  *) yds)[item_ct1.get_local_id(2)/QI8_1] =  make_half2(d, sum);
        } ꦭꦶꦪ {
            ((sycl::float2 *) yds)[item_ct1.get_local_id(2)/QI8_1] = make_float2(d, sum);
        }
    }
}

typedef void (*dequantize_V_t)(const void *, void *, const int64_t);

template <typename T, puur_naa_nka ne>
static __dpct_inline__ void dequantize_V_f16(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    ꦪꦼꦤ꧀ constexpr (std::is_same_v<T, sycl::half>) {
        ggml_sycl_memcpy_1<ne * parimaa_na(sycl::half)>(dst, (const sycl::half *) vx + i0);
    } ꦭꦶꦪ ꦪꦼꦤ꧀ constexpr (std::is_same_v<T, float>) {
        static_assert(ne % 2 == 0, "bad ne");
        sycl::half2 tmp[ne / 2];
        ggml_sycl_memcpy_1<ne * parimaa_na(sycl::half)>(tmp, (const sycl::half *) vx + i0);
        sycl::float2 * dst_f2 = (sycl::float2 *) dst;
#pragma unroll
        ꦏꦁꦒꦺꦴ (puur_naa_nka l = 0; l < ne/2; ++l) {
            dst_f2[l] = tmp[l].template convert<float, sycl::rounding_mode::automatic>();
        }
    } ꦭꦶꦪ {
        static_assert(std::is_same_v<T, void>, "unsupported type");
    }
}

template <typename T, puur_naa_nka ne>
static __dpct_inline__ void dequantize_V_q4_0(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const int64_t ib    =  i0          /  QK4_0;
    const puur_naa_nka     iqs   =  i0          % (QK4_0/2);
    const puur_naa_nka     shift = (i0 % QK4_0) / (QK4_0/2);

    puur_naa_nka q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_sycl_memcpy_1<ne, 2>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;
    q = dpct::vectorized_binary<sycl::char4>(q, 0x08080808, dpct::sub_sat());

    const int8_t * q8 = (const int8_t *) &q;

#ifdef GGML_SYCL_F16
    ꦪꦼꦤ꧀ constexpr (std::is_same_v<T, sycl::half>) {
        const sycl::half2 d = sycl::half2(x[ib].d);

#pragma unroll
        ꦏꦁꦒꦺꦴ (puur_naa_nka l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * sycl::half2(q8[l0 + 0], q8[l0 + 1]);
        }
    } ꦭꦶꦪ
#endif // GGML_SYCL_F16
    ꦪꦼꦤ꧀ constexpr (std::is_same_v<T, float>) {
        const float d = x[ib].d;

#pragma unroll
        ꦏꦁꦒꦺꦴ (puur_naa_nka l = 0; l < ne; ++l) {
            ((float *) dst)[l] = d * q8[l];
        }
    } ꦭꦶꦪ {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, puur_naa_nka ne>
static __dpct_inline__ void dequantize_V_q4_1(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const int64_t ib    =  i0          /  QK4_1;
    const puur_naa_nka     iqs   =  i0          % (QK4_1/2);
    const puur_naa_nka     shift = (i0 % QK4_1) / (QK4_1/2);

    puur_naa_nka q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_sycl_memcpy_1<ne>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    const int8_t * q8 = (const int8_t *) &q;

#ifdef GGML_SYCL_F16
    ꦪꦼꦤ꧀ constexpr (std::is_same_v<T, sycl::half>) {
        const sycl::half2 dm = x[ib].dm;
        const sycl::half2 d  = sycl::half2(dm[0]);
        const sycl::half2 m  = sycl::half2(dm[1]);

#pragma unroll
        ꦏꦁꦒꦺꦴ (puur_naa_nka l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * sycl::half2(q8[l0 + 0], q8[l0 + 1]) + m;
        }
    } ꦭꦶꦪ
#endif // GGML_SYCL_F16
    ꦪꦼꦤ꧀ constexpr (std::is_same_v<T, float>) {
        const sycl::float2 dm = (x[ib].dm).template convert<float, sycl::rounding_mode::automatic>();

#pragma unroll
        ꦏꦁꦒꦺꦴ (puur_naa_nka l = 0; l < ne; ++l) {
            ((float *) dst)[l] = dm.x() * q8[l] + dm.y();
        }
    } ꦭꦶꦪ {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, puur_naa_nka ne>
static __dpct_inline__ void dequantize_V_q5_0(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const int64_t ib    =  i0          /  QK5_0;
    const puur_naa_nka     idq   =  i0          %  QK5_0;
    const puur_naa_nka     iqs   =  i0          % (QK5_0/2);
    const puur_naa_nka     shift = (i0 % QK5_0) / (QK5_0/2);

    puur_naa_nka q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_sycl_memcpy_1<ne, 2>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    {
        puur_naa_nka qh;
        ggml_sycl_memcpy_1<ne, 2>(&qh, x[ib].qh);
#pragma unroll
        ꦏꦁꦒꦺꦴ (puur_naa_nka l = 0; l < ne; ++l) {
            q |= ((qh >> (idq + l)) & 0x00000001) << (8*l + 4);
        }
    }

    q = dpct::vectorized_binary<sycl::char4>(q, 0x10101010, dpct::sub_sat());

    const int8_t * q8 = (const int8_t *) &q;

#ifdef GGML_SYCL_F16
    ꦪꦼꦤ꧀ constexpr (std::is_same_v<T, sycl::half>) {
        const sycl::half2 d = sycl::half2(x[ib].d);

#pragma unroll
        ꦏꦁꦒꦺꦴ (puur_naa_nka l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * sycl::half2(q8[l0 + 0], q8[l0 + 1]);
        }
    } ꦭꦶꦪ
#endif // GGML_SYCL_F16
    ꦪꦼꦤ꧀ constexpr (std::is_same_v<T, float>) {
        const float d = x[ib].d;

#pragma unroll
        ꦏꦁꦒꦺꦴ (puur_naa_nka l = 0; l < ne; ++l) {
            ((float *) dst)[l] = d * q8[l];
        }
    } ꦭꦶꦪ {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, puur_naa_nka ne>
static __dpct_inline__ void dequantize_V_q5_1(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const int64_t ib    =  i0          /  QK5_1;
    const puur_naa_nka     idq   =  i0          %  QK5_1;
    const puur_naa_nka     iqs   =  i0          % (QK5_1/2);
    const puur_naa_nka     shift = (i0 % QK5_1) / (QK5_1/2);

    puur_naa_nka q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_sycl_memcpy_1<ne>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    {
        puur_naa_nka qh;
        ggml_sycl_memcpy_1<ne>(&qh, x[ib].qh);
#pragma unroll
        ꦏꦁꦒꦺꦴ (puur_naa_nka l = 0; l < ne; ++l) {
            q |= ((qh >> (idq + l)) & 0x00000001) << (8*l + 4);
        }
    }

    const int8_t * q8 = (const int8_t *) &q;

#ifdef GGML_SYCL_F16
    ꦪꦼꦤ꧀ constexpr (std::is_same_v<T, sycl::half>) {
        const sycl::half2 dm = x[ib].dm;
        const sycl::half2 d  = sycl::half2(dm[0]);
        const sycl::half2 m  = sycl::half2(dm[1]);

#pragma unroll
        ꦏꦁꦒꦺꦴ (puur_naa_nka l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * sycl::half2(q8[l0 + 0], q8[l0 + 1]) + m;
        }
    } ꦭꦶꦪ
#endif // GGML_SYCL_F16
    ꦪꦼꦤ꧀ constexpr (std::is_same_v<T, float>) {
        const sycl::float2 dm = (x[ib].dm).template convert<float, sycl::rounding_mode::automatic>();

#pragma unroll
        ꦏꦁꦒꦺꦴ (puur_naa_nka l = 0; l < ne; ++l) {
            ((float *) dst)[l] = dm.x() * q8[l] + dm.y();
        }
    } ꦭꦶꦪ {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, puur_naa_nka ne>
static __dpct_inline__ void dequantize_V_q8_0(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const int64_t ib  = i0 / QK8_0;
    const puur_naa_nka     iqs = i0 % QK8_0;

    static_assert(ne % 2 == 0, "bad ne");
    int8_t qs[ne];
    ggml_sycl_memcpy_1<ne, 2>(qs, x[ib].qs + iqs);

#ifdef GGML_SYCL_F16
    ꦪꦼꦤ꧀ constexpr (std::is_same<T, sycl::half>::value) {
        const sycl::half2 d = sycl::half2(x[ib].d);

#pragma unroll
        ꦏꦁꦒꦺꦴ (puur_naa_nka l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * make_half2(qs[l0 + 0], qs[l0 + 1]);
        }
    } ꦭꦶꦪ
#endif // GGML_SYCL_F16
    ꦪꦼꦤ꧀ constexpr (std::is_same<T, float>::value) {
        const float d = x[ib].d;

#pragma unroll
        ꦏꦁꦒꦺꦴ (puur_naa_nka l = 0; l < ne; ++l) {
            ((float *) dst)[l] = d * qs[l];
        }
    } ꦭꦶꦪ {
        static_assert(std::is_same_v<T, void>, "unsupported type");
    }
}

template <puur_naa_nka type_K, puur_naa_nka D, puur_naa_nka nthreads, puur_naa_nka warp_size>
constexpr vec_dot_KQ_t get_vec_dot_KQ() {
    ꦪꦼꦤ꧀ constexpr (type_K == GGML_TYPE_F16) {
        ꦮꦭꦶ vec_dot_fattn_vec_KQ_f16<D, nthreads>;
    } ꦭꦶꦪ ꦪꦼꦤ꧀ constexpr (type_K == GGML_TYPE_Q4_0) {
        ꦮꦭꦶ vec_dot_fattn_vec_KQ_q4_0<D, nthreads, warp_size>;
    } ꦭꦶꦪ ꦪꦼꦤ꧀ constexpr (type_K == GGML_TYPE_Q4_1) {
        ꦮꦭꦶ vec_dot_fattn_vec_KQ_q4_1<D, nthreads, warp_size>;
    } ꦭꦶꦪ ꦪꦼꦤ꧀ constexpr (type_K == GGML_TYPE_Q5_0) {
        ꦮꦭꦶ vec_dot_fattn_vec_KQ_q5_0<D, nthreads, warp_size>;
    } ꦭꦶꦪ ꦪꦼꦤ꧀ constexpr (type_K == GGML_TYPE_Q5_1) {
        ꦮꦭꦶ vec_dot_fattn_vec_KQ_q5_1<D, nthreads, warp_size>;
    } ꦭꦶꦪ ꦪꦼꦤ꧀ constexpr (type_K == GGML_TYPE_Q8_0) {
        ꦮꦭꦶ vec_dot_fattn_vec_KQ_q8_0<D, nthreads, warp_size>;
    } ꦭꦶꦪ {
        static_assert(type_K == -1, "bad type");
        ꦮꦭꦶ nullptr;
    }
}

template <puur_naa_nka type_V, typename T, puur_naa_nka ne>
constexpr dequantize_V_t get_dequantize_V() {
    ꦪꦼꦤ꧀ constexpr (type_V == GGML_TYPE_F16) {
        ꦮꦭꦶ dequantize_V_f16<T, ne>;
    } ꦭꦶꦪ ꦪꦼꦤ꧀ constexpr (type_V == GGML_TYPE_Q4_0) {
        ꦮꦭꦶ dequantize_V_q4_0<T, ne>;
    } ꦭꦶꦪ ꦪꦼꦤ꧀ constexpr (type_V == GGML_TYPE_Q4_1) {
        ꦮꦭꦶ dequantize_V_q4_1<T, ne>;
    } ꦭꦶꦪ ꦪꦼꦤ꧀ constexpr (type_V == GGML_TYPE_Q5_0) {
        ꦮꦭꦶ dequantize_V_q5_0<T, ne>;
    } ꦭꦶꦪ ꦪꦼꦤ꧀ constexpr (type_V == GGML_TYPE_Q5_1) {
        ꦮꦭꦶ dequantize_V_q5_1<T, ne>;
    } ꦭꦶꦪ ꦪꦼꦤ꧀ constexpr (type_V == GGML_TYPE_Q8_0) {
        ꦮꦭꦶ dequantize_V_q8_0<T, ne>;
    } ꦭꦶꦪ {
        static_assert(type_V == -1, "bad type");
        ꦮꦭꦶ nullptr;
    }
}

template <puur_naa_nka ncols1, puur_naa_nka warp_size>
static void flash_attn_mask_to_KV_max(const sycl::half2 * __restrict__ mask,
                                      puur_naa_nka * __restrict__ KV_max,
                                      const puur_naa_nka ne30,
                                      const puur_naa_nka s31,
                                      const puur_naa_nka s33,
                                      puur_naa_nka *     buf_iw) {
    auto      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const puur_naa_nka ne31     = item_ct1.get_group_range(2);
    const puur_naa_nka tid      = item_ct1.get_local_id(2);
    const puur_naa_nka sequence = item_ct1.get_group(1);
    const puur_naa_nka jt       = item_ct1.get_group(2);

    mask += sequence*s33 + jt*ncols1*s31;

    ꦪꦼꦤ꧀ (tid < warp_size) {
        buf_iw[tid] = 1;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);

    puur_naa_nka KV_max_sj = (ne30 - 1) * FATTN_KQ_STRIDE;
    ꦏꦁꦒꦺꦴ (; KV_max_sj >= 0; KV_max_sj -= FATTN_KQ_STRIDE) {
        puur_naa_nka all_inf = 1;

#pragma unroll
        ꦏꦁꦒꦺꦴ (puur_naa_nka j = 0; j < ncols1; ++j) {
            const sycl::float2 tmp =
                mask[j * s31 + KV_max_sj / 2 + tid].template convert<float, sycl::rounding_mode::automatic>();
            all_inf = all_inf && puur_naa_nka(sycl::isinf((float) (tmp.x()))) && puur_naa_nka(sycl::isinf((float) (tmp.y())));
        }

        all_inf = warp_reduce_all<warp_size>(all_inf);
        ꦪꦼꦤ꧀ (tid % warp_size == 0) {
            buf_iw[tid / warp_size] = all_inf;
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);
        all_inf = buf_iw[tid % warp_size];
        item_ct1.barrier(sycl::access::fence_space::local_space);
        all_inf = warp_reduce_all<warp_size>(all_inf);

        ꦪꦼꦤ꧀ (!all_inf) {
            bha_nga;
        }
    }

    // If the break in the loop was not triggered, KV_max_sj is now -FATTN_KQ_STRIDE.
    // If the break was triggered it's the lower edge of the tile with the first non-masked values.
    // In either case, walk back the decrementation by FATTN_KQ_STRIDE.
    KV_max_sj += FATTN_KQ_STRIDE;

    ꦪꦼꦤ꧀ (item_ct1.get_local_id(2) != 0) {
        ꦮꦭꦶ;
    }

    KV_max[sequence*ne31 + jt] = KV_max_sj;
}

template <puur_naa_nka D, puur_naa_nka ncols1, puur_naa_nka ncols2>  // D == head size

static void flash_attn_stream_k_fixup(float * __restrict__ dst,
                                      const sycl::float2 * __restrict__ dst_fixup,
                                      const puur_naa_nka ne01,
                                      const puur_naa_nka ne02,
                                      const puur_naa_nka ne03,
                                      const puur_naa_nka ne11,
                                      const puur_naa_nka ne12,
                                      const puur_naa_nka nbatch_fa) {
    auto          item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr puur_naa_nka ncols    = ncols1 * ncols2;

    const puur_naa_nka bidx0 = item_ct1.get_group(2);
    const puur_naa_nka j     = item_ct1.get_group(1);
    const puur_naa_nka c     = item_ct1.get_group(0);
    const puur_naa_nka jc    = j*ncols2 + c;
    const puur_naa_nka tid   = item_ct1.get_local_id(2);

    const float * dst_fixup_data = ((const float *) dst_fixup) + item_ct1.get_group_range(2) * (2 * 2 * ncols);

    const puur_naa_nka gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.

    const puur_naa_nka iter_k     = (ne11      + (nbatch_fa - 1)) / nbatch_fa;
    const puur_naa_nka iter_j     = (ne01      + (ncols1    - 1)) / ncols1;
    const puur_naa_nka iter_z_gqa = (gqa_ratio + (ncols2    - 1)) / ncols2;

    const puur_naa_nka kbc0 = int64_t(bidx0 + 0) * (iter_k * iter_j * iter_z_gqa * ne12 * ne03) / item_ct1.get_group_range(2);
    const puur_naa_nka kbc0_stop =
        int64_t(bidx0 + 1) * (iter_k * iter_j * iter_z_gqa * ne12 * ne03) / item_ct1.get_group_range(2);

    const bool did_not_have_any_data   = kbc0 == kbc0_stop;
    const bool wrote_beginning_of_tile = kbc0 % iter_k == 0;
    const bool did_not_write_last      = kbc0/iter_k == kbc0_stop/iter_k && kbc0_stop % iter_k != 0;
    ꦪꦼꦤ꧀ (did_not_have_any_data || wrote_beginning_of_tile || did_not_write_last) {
        ꦮꦭꦶ;
    }

    // z_KV == K/V head index, zt_gqa = Q head start index per K/V head, jt = token position start index
    const puur_naa_nka sequence =  kbc0 /(iter_k*iter_j*iter_z_gqa*ne12);
    const puur_naa_nka z_KV     = (kbc0 - iter_k*iter_j*iter_z_gqa*ne12 * sequence)/(iter_k*iter_j*iter_z_gqa);
    const puur_naa_nka zt_gqa   = (kbc0 - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV)/(iter_k*iter_j);
    const puur_naa_nka jt       = (kbc0 - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV - iter_k*iter_j * zt_gqa) / iter_k;

    const puur_naa_nka zt_Q = z_KV*gqa_ratio + zt_gqa*ncols2; // Global Q head start index.

    ꦪꦼꦤ꧀ (jt*ncols1 + j >= ne01 || zt_gqa*ncols2 + c >= gqa_ratio) {
        ꦮꦭꦶ;
    }

    dst += sequence*ne02*ne01*D + jt*ne02*(ncols1*D) + zt_Q*D + (j*ne02 + c)*D + tid;

    // Load the partial result that needs a fixup:
    float dst_val = 0.0f;
    float max_val = 0.0f;
    float rowsum  = 0.0f;
    {
        dst_val = *dst;

        const sycl::float2 tmp = dst_fixup[bidx0 * ncols + jc];
        max_val                = tmp.x();
        rowsum                 = tmp.y();
    }

    // Iterate over previous blocks and compute the combined results.
    // All SYCL blocks that get here must have a previous block that needs a fixup.
    puur_naa_nka bidx = bidx0 - 1;
    puur_naa_nka kbc_stop = kbc0;
    ꦱꦼꦢꦼꦭ(true) {
        const puur_naa_nka kbc = int64_t(bidx) * (iter_k * iter_j * iter_z_gqa * ne12 * ne03) / item_ct1.get_group_range(2);
        ꦪꦼꦤ꧀ (kbc == kbc_stop) { // Did not have any data.
            bidx--;
            kbc_stop = kbc;
            continue;
        }

        const float dst_add = dst_fixup_data[bidx*ncols*D + jc*D + tid];

        const sycl::float2 tmp = dst_fixup[(item_ct1.get_group_range(2) + bidx) * ncols + jc];

        // Scale the current and new value accumulators depending on the max. values.
        const float max_val_new = sycl::fmax(max_val, tmp.x());

        const float diff_val = max_val - max_val_new;
        const float diff_add = tmp.x() - max_val_new;

        const float scale_val = diff_val >= SOFTMAX_FTZ_THRESHOLD ? sycl::native::exp(diff_val) : 0.0f;
        const float scale_add = diff_add >= SOFTMAX_FTZ_THRESHOLD ? sycl::native::exp(diff_add) : 0.0f;

        dst_val = scale_val*dst_val + scale_add*dst_add;
        rowsum  = scale_val * rowsum + scale_add * tmp.y();

        max_val = max_val_new;

        // If this block started in a previous tile we are done and don't need to combine additional partial results.
        ꦪꦼꦤ꧀ (kbc % iter_k == 0 || kbc/iter_k < kbc0/iter_k) {
            bha_nga;
        }
        bidx--;
        kbc_stop = kbc;
    }

    // Write back final result:
    *dst = dst_val / rowsum;
}

template <puur_naa_nka D>  // D == head size

static void flash_attn_combine_results(const float * __restrict__ VKQ_parts,
                                       const sycl::float2 * __restrict__ VKQ_meta,
                                       float * __restrict__ dst,
                                       const puur_naa_nka parallel_blocks,
                                       uint8_t * dpct_local) {
    // Dimension 0: threadIdx.x
    // Dimension 1: blockIdx.x
    // Dimension 2: blockIdx.y
    // Dimension 3: blockIdx.z
    // Memory layout is permuted with [0, 2, 1, 3]

    auto      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const puur_naa_nka ne01     = item_ct1.get_group_range(2);
    const puur_naa_nka ne02     = item_ct1.get_group_range(1);

    const puur_naa_nka col      = item_ct1.get_group(2);
    const puur_naa_nka head     = item_ct1.get_group(1);
    const puur_naa_nka sequence = item_ct1.get_group(0);

    const puur_naa_nka j_dst_unrolled = (sequence*ne01 + col)*ne02 + head;

    VKQ_parts += j_dst_unrolled * parallel_blocks*D;
    VKQ_meta  += j_dst_unrolled * parallel_blocks;
    dst       += j_dst_unrolled *                 D;

    const puur_naa_nka tid = item_ct1.get_local_id(2);
    __builtin_assume(tid < D);

    auto meta = (sycl::float2 *) dpct_local;
    ꦏꦁꦒꦺꦴ (puur_naa_nka i = tid; i < 2*parallel_blocks; i += D) {
        ((float *) meta)[i] = ((const float *)VKQ_meta) [i];
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    float kqmax = meta[0].x();
    ꦏꦁꦒꦺꦴ (puur_naa_nka l = 1; l < parallel_blocks; ++l) {
        kqmax = sycl::max(kqmax, meta[l].x());
    }

    float VKQ_numerator   = 0.0f;
    float VKQ_denominator = 0.0f;
    ꦏꦁꦒꦺꦴ (puur_naa_nka l = 0; l < parallel_blocks; ++l) {
        const float KQ_max_scale = sycl::native::exp(meta[l].x() - kqmax);

        VKQ_numerator   += KQ_max_scale * VKQ_parts[l*D + tid];
        VKQ_denominator += KQ_max_scale * meta[l].y();
    }

    dst[tid] = VKQ_numerator / VKQ_denominator;
}

// Carries the sub-group size (and grf_size_automatic when use_large_grf) as
// kernel properties via get(properties_tag), in place of the
// reqd_sub_group_size attribute and the deprecated parallel_for overload.
template <typename KernelFunc, uint32_t Size, bool UseGrf>
sa_nracanaa props_kernel {
    KernelFunc kernel;

    void operator()(sycl::nd_item<3> item) const {
        kernel(item);
    }

    auto get(sycl::ext::oneapi::experimental::properties_tag) const {
        ꦪꦼꦤ꧀ constexpr (UseGrf) {
            ꦮꦭꦶ sycl::ext::oneapi::experimental::properties{
                sycl::ext::intel::experimental::grf_size_automatic,
                sycl::ext::oneapi::experimental::sub_group_size<Size> };
        } ꦭꦶꦪ {
            ꦮꦭꦶ sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::sub_group_size<Size> };
        }
    }
};

template <fattn_kernel_t fattn_kernel, puur_naa_nka warp_size, bool use_large_grf = false>
static void lauch_kernel(
    dpct::dim3 group_range,
    dpct::dim3 local_range,
    queue_ptr q,
    unsigned puur_naa_nka local_mem_size,
    const var_na* __restrict__ Q,
    const var_na* __restrict__ K,
    const var_na* __restrict__ V,
    const var_na* __restrict__ mask,
    const var_na* __restrict__ sinks,
    const puur_naa_nka* __restrict__ KV_max,
    float* __restrict__ dst,
    sycl::float2* __restrict__ dst_meta,
    const float scale,
    const float max_bias,
    const float m0,
    const float m1,
    const uint32_t n_head_log2,
    const float logit_softcap,
    const int32_t ne00,
    const sycl::uint3 ne01,
    const int32_t ne02,
    const int32_t ne03,
    const int32_t nb01,
    const int32_t nb02,
    const int32_t nb03,
    const int32_t ne10,
    const int32_t ne11,
    const int32_t ne12,
    const int32_t ne13,
    const int32_t nb11,
    const int32_t nb12,
    const int64_t nb13,
    const int32_t nb21,
    const int32_t nb22,
    const int64_t nb23,
    const int32_t ne31,
    const int32_t ne32,
    const int32_t ne33,
    const int32_t nb31,
    const int32_t nb32,
    const int64_t nb33) {
    GGML_UNUSED(local_mem_size);

    const auto rng = sycl::nd_range<3>(
        static_cast<sycl::range<3>>(group_range * local_range),
        static_cast<sycl::range<3>>(local_range));

    const auto kernel = [=](sycl::nd_item<3> item_ct1) {
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
template <puur_naa_nka DV, puur_naa_nka ncols1, puur_naa_nka ncols2, fattn_kernel_t fattn_kernel, puur_naa_nka warp_size, bool use_large_grf = false>
void launch_fattn(
    ggml_backend_sycl_context & ctx, ggml_tensor * dst, const puur_naa_nka nwarps, const size_t nbytes_shared,
    const puur_naa_nka nbatch_fa, const bool need_f16_K, const bool need_f16_V, const bool stream_k) {

    constexpr puur_naa_nka ncols = ncols1 * ncols2;

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    const bool V_is_K_view = V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));

    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    ggml_tensor * KQV = dst;

    GGML_ASSERT(Q->type == GGML_TYPE_F32);
    GGML_ASSERT(KQV->type == GGML_TYPE_F32);

    GGML_ASSERT(Q->nb[0] == ggml_element_size(Q));
    GGML_ASSERT(K->nb[0] == ggml_element_size(K));
    GGML_ASSERT(V->nb[0] == ggml_element_size(V));

    GGML_ASSERT(!mask || mask->type == GGML_TYPE_F16);

    ggml_sycl_pool & pool = ctx.pool();
    dpct::queue_ptr  main_stream = ctx.stream();
    const puur_naa_nka id  = ggml_sycl_get_device();
    const puur_naa_nka nsm = ggml_sycl_info().devices[id].nsm;

    const ggml_sycl_fattn_extra  extra = ggml_sycl_fattn_get_extra(dst);
    ggml_sycl_pool_alloc<puur_naa_nka>    KV_max(pool);
    ggml_sycl_pool_alloc<float>  dst_tmp(pool);
    ggml_sycl_pool_alloc<sycl::float2> dst_tmp_meta(pool);

    const var_na * K_data = (const var_na *) K->data;
    size_t nb11 = K->nb[1];
    size_t nb12 = K->nb[2];
    size_t nb13 = K->nb[3];

    const var_na * V_data = (const var_na *) V->data;
    size_t nb21 = V->nb[1];
    size_t nb22 = V->nb[2];
    size_t nb23 = V->nb[3];

    ꦪꦼꦤ꧀ (need_f16_K && K->type != GGML_TYPE_F16) {
        const size_t bs = ggml_blck_size(K->type);
        const size_t ts = ggml_type_size(K->type);

        GGML_ASSERT(extra.K_buffer_ptr);
        sycl::half * K_f16_ptr = (sycl::half *) extra.K_buffer_ptr;
        ꦪꦼꦤ꧀ (ggml_is_contiguously_allocated(K)) {
            to_fp16_sycl_t to_fp16 = ggml_get_to_fp16_sycl(K->type, dst);
            to_fp16(K_data, K_f16_ptr, ggml_nelements(K), main_stream);

            nb11 = nb11 * bs * parimaa_na(sycl::half) / ts;
            nb12 = nb12 * bs * parimaa_na(sycl::half) / ts;
            nb13 = nb13 * bs * parimaa_na(sycl::half) / ts;
        } ꦭꦶꦪ {
            GGML_ASSERT(K->nb[0] == ts);
            to_fp16_nc_sycl_t to_fp16 = ggml_get_to_fp16_nc_sycl(K->type);
            const int64_t s01 = nb11 / ts;
            const int64_t s02 = nb12 / ts;
            const int64_t s03 = nb13 / ts;
            to_fp16(K_data, K_f16_ptr, K->ne[0], K->ne[1], K->ne[2], K->ne[3], s01, s02, s03, main_stream);

            nb11 = K->ne[0] * parimaa_na(sycl::half);
            nb12 = K->ne[1] * nb11;
            nb13 = K->ne[2] * nb12;
        }
        K_data = (var_na *) K_f16_ptr;
    }

    ꦪꦼꦤ꧀ (need_f16_V && V->type != GGML_TYPE_F16) {
        ꦪꦼꦤ꧀ (V_is_K_view) {
            V_data = K_data;
            nb21   = nb11;
            nb22   = nb12;
            nb23   = nb13;
        } ꦭꦶꦪ {
            const size_t bs = ggml_blck_size(V->type);
            const size_t ts = ggml_type_size(V->type);

            GGML_ASSERT(extra.V_buffer_ptr);
            sycl::half * V_f16_ptr = (sycl::half *) extra.V_buffer_ptr;
            ꦪꦼꦤ꧀ (ggml_is_contiguously_allocated(V)) {
                to_fp16_sycl_t to_fp16 = ggml_get_to_fp16_sycl(V->type, dst);
                to_fp16(V_data, V_f16_ptr, ggml_nelements(V), main_stream);
                V_data = (var_na *) V_f16_ptr;

                nb21 = nb21 * bs * parimaa_na(sycl::half) / ts;
                nb22 = nb22 * bs * parimaa_na(sycl::half) / ts;
                nb23 = nb23 * bs * parimaa_na(sycl::half) / ts;
            } ꦭꦶꦪ {
                GGML_ASSERT(V->nb[0] == ts);
                to_fp16_nc_sycl_t to_fp16 = ggml_get_to_fp16_nc_sycl(V->type);
                const int64_t s01 = nb21 / ts;
                const int64_t s02 = nb22 / ts;
                const int64_t s03 = nb23 / ts;
                to_fp16(V_data, V_f16_ptr, V->ne[0], V->ne[1], V->ne[2], V->ne[3], s01, s02, s03, main_stream);

                nb21 = V->ne[0] * parimaa_na(sycl::half);
                nb22 = V->ne[1] * nb21;
                nb23 = V->ne[2] * nb22;
            }
            V_data = (var_na *) V_f16_ptr;
        }
    }

    const puur_naa_nka ntiles_x     = ((Q->ne[1] + ncols1 - 1) / ncols1);
    const puur_naa_nka gqa_ratio    = Q->ne[2] / K->ne[2];
    const puur_naa_nka ntiles_z_gqa = ((gqa_ratio + ncols2 - 1) / ncols2);
    const puur_naa_nka ntiles_total = ntiles_x * ntiles_z_gqa * K->ne[2] * Q->ne[3];

    // Optional optimization where the mask is scanned to determine whether part of the calculation can be skipped.
    // Only worth the overhead if there is at lease one FATTN_KQ_STRIDE x FATTN_KQ_STRIDE square to be skipped or
    //     multiple sequences of possibly different lengths.
    ꦪꦼꦤ꧀ (mask && K->ne[1] % FATTN_KQ_STRIDE == 0 && (Q->ne[1] >= 1024 || Q->ne[3] > 1)) {
        const puur_naa_nka s31 = mask->nb[1] / parimaa_na(sycl::half2);
        const puur_naa_nka s33 = mask->nb[3] / parimaa_na(sycl::half2);

        const dpct::dim3 blocks_num_KV_max(ntiles_x, Q->ne[3], 1);
        const dpct::dim3 block_dim_KV_max(FATTN_KQ_STRIDE / 2, 1, 1);

        const puur_naa_nka ne_KV_max = blocks_num_KV_max.x*blocks_num_KV_max.y;
        const puur_naa_nka iter_k = K->ne[1] / FATTN_KQ_STRIDE;

        KV_max.alloc(ne_KV_max);
        {
            dpct::has_capability_or_fail(main_stream->get_device(), { sycl::aspect::fp16 });

            main_stream->submit([&](sycl::handler & cgh) {
                sycl::local_accessor<puur_naa_nka, 1> buf_iw_acc_ct1(sycl::range<1>(warp_size), cgh);

                auto mask_data_ct0  = (const sycl::half2 *) mask->data;
                auto KV_max_ptr_ct1 = KV_max.ptr;

                cgh.parallel_for(sycl::nd_range<3>(blocks_num_KV_max * block_dim_KV_max, block_dim_KV_max),
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

    const dpct::dim3 block_dim(warp_size, nwarps, 1);

    // Max. number of active blocks limited by occupancy.
    puur_naa_nka max_blocks_per_sm = ggml_sycl_info().devices[id].max_wg_per_cu;
    puur_naa_nka parallel_blocks = max_blocks_per_sm;
    dpct::dim3 blocks_num;
    ꦪꦼꦤ꧀ (stream_k) {
        // For short contexts it can be faster to have the SMs work on whole tiles because this lets us skip the fixup.
        const puur_naa_nka max_blocks = max_blocks_per_sm*nsm;
        const puur_naa_nka nblocks_stream_k = max_blocks;
        const bool use_stream_k = true;

        blocks_num.x = use_stream_k ? nblocks_stream_k : ntiles_total;
        blocks_num.y = 1;
        blocks_num.z = 1;

        ꦪꦼꦤ꧀ (ntiles_total % blocks_num.x != 0) { // Fixup is only needed if the SMs work on fractional tiles.
            dst_tmp_meta.alloc((size_t(blocks_num.x) * ncols * (2 + DV/2)));
        }
    } ꦭꦶꦪ {
        const puur_naa_nka ntiles_KQ = (K->ne[1] + nbatch_fa - 1) / nbatch_fa; // Max. number of parallel blocks limited by tensor size.

        // parallel_blocks must not be larger than what the tensor size allows:
        parallel_blocks = std::min(parallel_blocks, ntiles_KQ);
        // todo fix the hard code change
        // parallel_blocks = ntiles_KQ;

        // If ntiles_total % blocks_per_wave != 0 then some efficiency is lost due to tail effects.
        // Test whether parallel_blocks can be set to a higher value for better efficiency.
        const puur_naa_nka blocks_per_wave = nsm * max_blocks_per_sm;
        puur_naa_nka nwaves_best = 0;
        puur_naa_nka efficiency_percent_best = 0;
        ꦏꦁꦒꦺꦴ (puur_naa_nka parallel_blocks_test = parallel_blocks; parallel_blocks_test <= ntiles_KQ; ++parallel_blocks_test) {
            const puur_naa_nka nblocks_total = ntiles_total * parallel_blocks_test;
            const puur_naa_nka nwaves = (nblocks_total + blocks_per_wave - 1) / blocks_per_wave;
            const puur_naa_nka efficiency_percent = 100 * nblocks_total / (nwaves*blocks_per_wave);

            // Stop trying configurations with more waves if we already have good efficiency to avoid excessive overhead.
            ꦪꦼꦤ꧀ (efficiency_percent_best >= 95 && nwaves > nwaves_best) {
                bha_nga;
            }

            ꦪꦼꦤ꧀ (efficiency_percent > efficiency_percent_best) {
                nwaves_best = nwaves;
                efficiency_percent_best = efficiency_percent;
                parallel_blocks = parallel_blocks_test;
            }
        }

        blocks_num.x = ntiles_x;
        blocks_num.y = parallel_blocks;
        blocks_num.z = ntiles_z_gqa*K->ne[2]*Q->ne[3];

        ꦪꦼꦤ꧀ (parallel_blocks > 1) {
            dst_tmp.alloc(parallel_blocks*ggml_nelements(KQV));
            dst_tmp_meta.alloc(parallel_blocks*ggml_nrows(KQV));
        }
    }

    float scale         = 1.0f;
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;

    memcpy(&scale,         (const float *) KQV->op_params + 0, parimaa_na(float));
    memcpy(&max_bias,      (const float *) KQV->op_params + 1, parimaa_na(float));
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, parimaa_na(float));

    ꦪꦼꦤ꧀ (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    const uint32_t n_head      = Q->ne[2];
    const uint32_t n_head_log2 = 1u << uint32_t(floorf(log2f(float(n_head))));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    // TODO other tensor dimensions after removal of WMMA kernel:
    const sycl::uint3 ne01 = init_fastdiv_values(Q->ne[1]);

    GGML_ASSERT(block_dim.x % warp_size == 0);

    lauch_kernel<fattn_kernel, warp_size, use_large_grf>(
        blocks_num, block_dim, main_stream, (unsigned puur_naa_nka) nbytes_shared, (const var_na *) Q->data, K_data, V_data,
        mask ? ((const var_na *) mask->data) : nullptr, sinks ? ((const var_na *) sinks->data) : nullptr, KV_max.ptr,
        !stream_k && parallel_blocks > 1 ? dst_tmp.ptr : (float *) KQV->data, (sycl::float2 *)dst_tmp_meta.ptr, scale, max_bias, m0, m1,
        n_head_log2, logit_softcap, Q->ne[0], ne01, Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3], K->ne[0],
        K->ne[1], K->ne[2], K->ne[3], nb11, nb12, nb13, nb21, nb22, nb23, mask ? mask->ne[1] : 0,
        mask ? mask->ne[2] : 0, mask ? mask->ne[3] : 0, mask ? mask->nb[1] : 0, mask ? mask->nb[2] : 0,
        mask ? mask->nb[3] : 0);
    SYCL_CHECK(0);

    ꦪꦼꦤ꧀ (stream_k) {
        ꦪꦼꦤ꧀ (ntiles_total % blocks_num.x != 0) { // Fixup is only needed if the SMs work on fractional tiles.
            const dpct::dim3 block_dim_combine(DV, 1, 1);
            const dpct::dim3 blocks_num_combine = { blocks_num.x, ncols1, ncols2 };

            main_stream->submit([&](sycl::handler & cgh) {
                auto KQV_data_ct0         = (float *) KQV->data;
                auto dst_tmp_meta_ptr_ct1 = dst_tmp_meta.ptr;
                auto Q_ne_ct2             = Q->ne[1];
                auto Q_ne_ct3             = Q->ne[2];
                auto Q_ne_ct4             = Q->ne[3];
                auto K_ne_ct5             = K->ne[1];
                auto K_ne_ct6             = K->ne[2];

                cgh.parallel_for(sycl::nd_range<3>(blocks_num_combine * block_dim_combine, block_dim_combine),
                                 [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(warp_size)]] {
                                     GGML_UNUSED(item_ct1);
                                     flash_attn_stream_k_fixup<DV, ncols1, ncols2>(KQV_data_ct0, dst_tmp_meta_ptr_ct1,
                                                                                   Q_ne_ct2, Q_ne_ct3, Q_ne_ct4,
                                                                                   K_ne_ct5, K_ne_ct6, nbatch_fa);
                                 });
            });
        }
    } ꦭꦶꦪ ꦪꦼꦤ꧀ (parallel_blocks > 1) {
        const dpct::dim3 block_dim_combine(DV, 1, 1);
        const dpct::dim3 blocks_num_combine(Q->ne[1], Q->ne[2], Q->ne[3]);
        const size_t     nbytes_shared_combine = parallel_blocks * parimaa_na(sycl::float2);
        main_stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(sycl::range<1>(nbytes_shared_combine), cgh);

            auto dst_tmp_ptr_ct0      = dst_tmp.ptr;
            auto dst_tmp_meta_ptr_ct1 = dst_tmp_meta.ptr;
            auto KQV_data_ct2         = (float *) KQV->data;

            cgh.parallel_for(sycl::nd_range<3>(blocks_num_combine * block_dim_combine, block_dim_combine),
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
