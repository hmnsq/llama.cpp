#pragma once

#ಅನ್ತರ್ಭಾವ <sycl/sycl.hpp>
#ಅನ್ತರ್ಭಾವ <sycl/ext/intel/experimental/grf_size_properties.hpp>
#ಅನ್ತರ್ಭಾವ "dpct/helper.hpp"
#ಅನ್ತರ್ಭಾವ "common.hpp"
#ಅನ್ತರ್ಭಾವ "convert.hpp"
#ಅನ್ತರ್ಭಾವ "vecdotq.hpp"
#ಅನ್ತರ್ಭಾವ "fattn.hpp"

#ಅನ್ತರ್ಭಾವ "ggml.h"

#ಅನ್ತರ್ಭಾವ <cstdint>
#ಅನ್ತರ್ಭಾವ <cmath>
#ಅನ್ತರ್ಭಾವ <float.h>


#define FATTN_KQ_STRIDE       256
#define HALF_MAX_HALF         sycl::half(65504.0f/2) // Use neg. of this instead of -INFINITY to initialize KQ max vals to avoid NaN upon subtraction.
#define SOFTMAX_FTZ_THRESHOLD -20.0f                   // Softmax exp. of values smaller than this are flushed to zero to avoid NaNs.
#define FATTN_KQ_MAX_OFFSET (3.0f*0.6931f)

ಪ್ರಕಾರನಿರ್ದೇಶ ಶೂನ್ಯ (*fattn_kernel_t)(
    ಧ್ರುವ ವರ್ಣ* Q,
    ಧ್ರುವ ವರ್ಣ* K,
    ಧ್ರುವ ವರ್ಣ* V,
    ಧ್ರುವ ವರ್ಣ* mask,
    ಧ್ರುವ ವರ್ಣ* sinks,
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ* KV_max,
    ಭಿನ್ನ* dst,
    sycl::float2* dst_meta,
    ಧ್ರುವ ಭಿನ್ನ scale,
    ಧ್ರುವ ಭಿನ್ನ max_bias,
    ಧ್ರುವ ಭಿನ್ನ m0,
    ಧ್ರುವ ಭಿನ್ನ m1,
    ಧ್ರುವ uint32_t n_head_log2,
    ಧ್ರುವ ಭಿನ್ನ logit_softcap,
    ಧ್ರುವ int32_t ne00,
    ಧ್ರುವ sycl::uint3 ne01,
    ಧ್ರುವ int32_t ne02,
    ಧ್ರುವ int32_t ne03,
    ಧ್ರುವ int32_t nb01,
    ಧ್ರುವ int32_t nb02,
    ಧ್ರುವ int32_t nb03,
    ಧ್ರುವ int32_t ne10,
    ಧ್ರುವ int32_t ne11,
    ಧ್ರುವ int32_t ne12,
    ಧ್ರುವ int32_t ne13,
    ಧ್ರುವ int32_t nb11,
    ಧ್ರುವ int32_t nb12,
    ಧ್ರುವ int64_t nb13,
    ಧ್ರುವ int32_t nb21,
    ಧ್ರುವ int32_t nb22,
    ಧ್ರುವ int64_t nb23,
    ಧ್ರುವ int32_t ne31,
    ಧ್ರುವ int32_t ne32,
    ಧ್ರುವ int32_t ne33,
    ಧ್ರುವ int32_t nb31,
    ಧ್ರುವ int32_t nb32,
    ಧ್ರುವ int64_t nb33);

ಪ್ರಕಾರನಿರ್ದೇಶ ಭಿನ್ನ (*vec_dot_KQ_t)(
    ಧ್ರುವ ವರ್ಣ * __restrict__ K_c, ಧ್ರುವ ಶೂನ್ಯ * __restrict__ Q_v, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ * __restrict__ Q_q8 , ಧ್ರುವ ಶೂನ್ಯ * __restrict__ Q_ds);

template <ಪೂರ್ಣಾಙ್ಕ D, ಪೂರ್ಣಾಙ್ಕ nthreads>
ಸ್ಥಿರ __dpct_inline__ ಭಿನ್ನ vec_dot_fattn_vec_KQ_f16(ಧ್ರುವ ವರ್ಣ * __restrict__ K_c,
                                                      ಧ್ರುವ ಶೂನ್ಯ * __restrict__ Q_v,
                                                      ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ * __restrict__ Q_q8,
                                                      ಧ್ರುವ ಶೂನ್ಯ * __restrict__ Q_ds_v) {
    ಧ್ರುವ sycl::half2 * K_h2 = (ಧ್ರುವ sycl::half2 *) K_c;
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v);

    constexpr ಪೂರ್ಣಾಙ್ಕ cpy_nb = ggml_sycl_get_max_cpy_bytes();
    constexpr ಪೂರ್ಣಾಙ್ಕ cpy_ne = cpy_nb / 4;

    ಭಿನ್ನ sum = 0.0f;

#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads*cpy_ne) {
        sycl::half2 tmp[cpy_ne];
        ggml_sycl_memcpy_1<ಪರಿಮಾಣ(tmp)>(
            tmp,
            K_h2 + k_KQ_0 + (sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2) % nthreads) * cpy_ne);
#pragma unroll
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
#ifdef GGML_SYCL_F16
            ggml_sycl_mad(sum,                tmp[k_KQ_1] , ((ಧ್ರುವ sycl::half2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#ಅನ್ಯಥಾ
            ggml_sycl_mad(sum, __half22float2(tmp[k_KQ_1]), ((ಧ್ರುವ sycl::float2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#endif // GGML_SYCL_F16
        }
    }

    ಪ್ರತ್ಯಾಗಮ sum;
}

template <ಪೂರ್ಣಾಙ್ಕ D, ಪೂರ್ಣಾಙ್ಕ nthreads, ಪೂರ್ಣಾಙ್ಕ warp_size>
ಸ್ಥಿರ __dpct_inline__ ಭಿನ್ನ vec_dot_fattn_vec_KQ_q4_0(ಧ್ರುವ ವರ್ಣ * __restrict__ K_c,
                                                       ಧ್ರುವ ಶೂನ್ಯ * __restrict__ Q_v,
                                                       ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ * __restrict__ Q_q8,
                                                       ಧ್ರುವ ಶೂನ್ಯ * __restrict__ Q_ds_v) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    ಧ್ರುವ block_q4_0 * K_q4_0   = (ಧ್ರುವ block_q4_0 *) K_c;
    GGML_UNUSED(Q_v);

    ಭಿನ್ನ sum = 0.0f;

#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ k_KQ_0 = 0; k_KQ_0 < ಪೂರ್ಣಾಙ್ಕ(D/ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ)); k_KQ_0 += nthreads) {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ k_KQ =
            k_KQ_0 + (nthreads == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads);

        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ib    = k_KQ /  QI8_1;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ iqs4  = k_KQ %  QI4_0;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ shift = k_KQ & (QI8_1/2);

        ಪೂರ್ಣಾಙ್ಕ v;
        ggml_sycl_memcpy_1<ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ), 2>(&v, K_q4_0[ib].qs + ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ u = Q_q8[k_KQ_0/nthreads];

        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ sumi = ggml_sycl_dp4a(v, u, 0);

        ಧ್ರುವ sycl::float2 Q_ds = ((ಧ್ರುವ sycl::float2 *) Q_ds_v)[k_KQ_0 / nthreads];
        sum += __half2float(K_q4_0[ib].d) * (sumi*Q_ds.x() - (8/QI8_1)*Q_ds.y());
    }

    ಪ್ರತ್ಯಾಗಮ sum;
}

template <ಪೂರ್ಣಾಙ್ಕ D, ಪೂರ್ಣಾಙ್ಕ nthreads , ಪೂರ್ಣಾಙ್ಕ warp_size>
ಸ್ಥಿರ __dpct_inline__ ಭಿನ್ನ vec_dot_fattn_vec_KQ_q4_1(ಧ್ರುವ ವರ್ಣ * __restrict__ K_c,
                                                       ಧ್ರುವ ಶೂನ್ಯ * __restrict__ Q_v,
                                                       ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ * __restrict__ Q_q8,
                                                       ಧ್ರುವ ಶೂನ್ಯ * __restrict__ Q_ds_v) {
    auto               item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ಧ್ರುವ block_q4_1 * K_q4_1   = (ಧ್ರುವ block_q4_1 *) K_c;
    GGML_UNUSED(Q_v);

    ಭಿನ್ನ sum = 0.0f;

#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ k_KQ_0 = 0; k_KQ_0 < ಪೂರ್ಣಾಙ್ಕ(D/ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ)); k_KQ_0 += nthreads) {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ k_KQ =
            k_KQ_0 + (nthreads == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads);

        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ib    = k_KQ /  QI8_1;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ iqs4  = k_KQ %  QI4_1;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ shift = k_KQ & (QI8_1/2);

        ಪೂರ್ಣಾಙ್ಕ v;
        ggml_sycl_memcpy_1<ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ)>(&v, K_q4_1[ib].qs + ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ u = Q_q8[k_KQ_0/nthreads];

        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ sumi = ggml_sycl_dp4a(v, u, 0);

        ಧ್ರುವ sycl::float2 K_dm = (K_q4_1[ib].dm).template convert<ಭಿನ್ನ, sycl::rounding_mode::automatic>();
        ಧ್ರುವ sycl::float2 Q_ds = ((ಧ್ರುವ sycl::float2 *) Q_ds_v)[k_KQ_0 / nthreads];

        sum += K_dm.x()*Q_ds.x()*sumi + K_dm.y()*Q_ds.y()/QI8_1;
    }

    ಪ್ರತ್ಯಾಗಮ sum;
}

template <ಪೂರ್ಣಾಙ್ಕ D, ಪೂರ್ಣಾಙ್ಕ nthreads, ಪೂರ್ಣಾಙ್ಕ warp_size>
ಸ್ಥಿರ __dpct_inline__ ಭಿನ್ನ vec_dot_fattn_vec_KQ_q5_0(ಧ್ರುವ ವರ್ಣ * __restrict__ K_c,
                                                       ಧ್ರುವ ಶೂನ್ಯ * __restrict__ Q_v,
                                                       ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ * __restrict__ Q_q8,
                                                       ಧ್ರುವ ಶೂನ್ಯ * __restrict__ Q_ds_v) {
    auto               item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ಧ್ರುವ block_q5_0 * K_q5_0   = (ಧ್ರುವ block_q5_0 *) K_c;
    GGML_UNUSED(Q_v);

    ಭಿನ್ನ sum = 0.0f;

#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ k_KQ_0 = 0; k_KQ_0 < ಪೂರ್ಣಾಙ್ಕ(D/ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ)); k_KQ_0 += nthreads) {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ k_KQ =
            k_KQ_0 + (nthreads == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads);

        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ib    = k_KQ /  QI8_1;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ iqs4  = k_KQ %  QI5_0;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ iqs8  = k_KQ %  QI8_1;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ shift = k_KQ & (QI8_1/2);

        ಪೂರ್ಣಾಙ್ಕ v;
        ggml_sycl_memcpy_1<ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ), 2>(&v, K_q5_0[ib].qs + ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;

        {
            ಪೂರ್ಣಾಙ್ಕ vh;
            ggml_sycl_memcpy_1<ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ), 2>(&vh, K_q5_0[ib].qh);
            vh >>= iqs8 * QI5_0;

            v |= (vh <<  4) & 0x00000010; // 0 ->  4
            v |= (vh << 11) & 0x00001000; // 1 -> 12
            v |= (vh << 18) & 0x00100000; // 2 -> 20
            v |= (vh << 25) & 0x10000000; // 3 -> 28
        }

        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ u = Q_q8[k_KQ_0/nthreads];

        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ sumi = ggml_sycl_dp4a(v, u, 0);

        ಧ್ರುವ sycl::float2 Q_ds = ((ಧ್ರುವ sycl::float2 *) Q_ds_v)[k_KQ_0 / nthreads];

        sum += __half2float(K_q5_0[ib].d) * (sumi*Q_ds.x() - (16/QI8_1)*Q_ds.y());
    }

    ಪ್ರತ್ಯಾಗಮ sum;
}

template <ಪೂರ್ಣಾಙ್ಕ D, ಪೂರ್ಣಾಙ್ಕ nthreads, ಪೂರ್ಣಾಙ್ಕ warp_size>
ಸ್ಥಿರ __dpct_inline__ ಭಿನ್ನ vec_dot_fattn_vec_KQ_q5_1(ಧ್ರುವ ವರ್ಣ * __restrict__ K_c,
                                                       ಧ್ರುವ ಶೂನ್ಯ * __restrict__ Q_v,
                                                       ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ * __restrict__ Q_q8,
                                                       ಧ್ರುವ ಶೂನ್ಯ * __restrict__ Q_ds_v) {
    auto               item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ಧ್ರುವ block_q5_1 * K_q5_1   = (ಧ್ರುವ block_q5_1 *) K_c;
    GGML_UNUSED(Q_v);

    ಭಿನ್ನ sum = 0.0f;

#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ k_KQ_0 = 0; k_KQ_0 < ಪೂರ್ಣಾಙ್ಕ(D/ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ)); k_KQ_0 += nthreads) {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ k_KQ =
            k_KQ_0 + (nthreads == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads);

        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ib    = k_KQ /  QI8_1;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ iqs4  = k_KQ %  QI5_1;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ iqs8  = k_KQ %  QI8_1;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ shift = k_KQ & (QI8_1/2);

        ಪೂರ್ಣಾಙ್ಕ v;
        ggml_sycl_memcpy_1<ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ)>(&v, K_q5_1[ib].qs + ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;

        {
            ಪೂರ್ಣಾಙ್ಕ vh;
            ggml_sycl_memcpy_1<ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ)>(&vh, K_q5_1[ib].qh);
            vh >>= iqs8 * QI5_0;

            v |= (vh <<  4) & 0x00000010; // 0 ->  4
            v |= (vh << 11) & 0x00001000; // 1 -> 12
            v |= (vh << 18) & 0x00100000; // 2 -> 20
            v |= (vh << 25) & 0x10000000; // 3 -> 28
        }

        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ u = Q_q8[k_KQ_0/nthreads];

        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ sumi = ggml_sycl_dp4a(v, u, 0);

        ಧ್ರುವ sycl::float2 K_dm = (K_q5_1[ib].dm).template convert<ಭಿನ್ನ, sycl::rounding_mode::automatic>();
        ಧ್ರುವ sycl::float2 Q_ds = ((ಧ್ರುವ sycl::float2 *) Q_ds_v)[k_KQ_0 / nthreads];

        sum += K_dm.x()*Q_ds.x()*sumi + K_dm.y()*Q_ds.y()/QI8_1;
    }

    ಪ್ರತ್ಯಾಗಮ sum;
}

template <ಪೂರ್ಣಾಙ್ಕ D, ಪೂರ್ಣಾಙ್ಕ nthreads, ಪೂರ್ಣಾಙ್ಕ warp_size>
ಸ್ಥಿರ __dpct_inline__ ಭಿನ್ನ vec_dot_fattn_vec_KQ_q8_0(ಧ್ರುವ ವರ್ಣ * __restrict__ K_c,
                                                       ಧ್ರುವ ಶೂನ್ಯ * __restrict__ Q_v,
                                                       ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ * __restrict__ Q_q8,
                                                       ಧ್ರುವ ಶೂನ್ಯ * __restrict__ Q_ds_v) {
    auto               item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ಧ್ರುವ block_q8_0 * K_q8_0   = (ಧ್ರುವ block_q8_0 *) K_c;
    GGML_UNUSED(Q_v);

    ಭಿನ್ನ sum = 0.0f;

#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ k_KQ_0 = 0; k_KQ_0 < ಪೂರ್ಣಾಙ್ಕ(D/ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ)); k_KQ_0 += nthreads) {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ k_KQ =
            k_KQ_0 + (nthreads == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads);

        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ib  = k_KQ / QI8_0;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ iqs = k_KQ % QI8_0;

        ಪೂರ್ಣಾಙ್ಕ v;
        ggml_sycl_memcpy_1<ಪರಿಮಾಣ(v), 2>(&v, K_q8_0[ib].qs + 4*iqs);

        ಧ್ರುವ sycl::float2 * Q_ds = (ಧ್ರುವ sycl::float2 *) Q_ds_v;
        ಧ್ರುವ ಭಿನ್ನ          Q_d  = Q_ds[k_KQ_0 / nthreads].x();

        sum += vec_dot_q8_0_q8_1_impl<ಭಿನ್ನ, 1>(&v, &Q_q8[k_KQ_0/nthreads], K_q8_0[ib].d, Q_d);
    }

    ಪ್ರತ್ಯಾಗಮ sum;
}

template <typename Tds, ಪೂರ್ಣಾಙ್ಕ ni, ಪೂರ್ಣಾಙ್ಕ warp_size>
ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ quantize_q8_1_to_shared(ಧ್ರುವ ಭಿನ್ನ * __restrict__ x,
                                                    ಧ್ರುವ ಭಿನ್ನ scale,
                                                    ಪೂರ್ಣಾಙ್ಕ * __restrict__ yq32,
                                                    ಶೂನ್ಯ * __restrict__ yds) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    ಭಿನ್ನ vals[ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ)] = { 0.0f };
#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ l = 0; l < ಪೂರ್ಣಾಙ್ಕ(ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ)); ++l) {
        vals[l] =
            (ni == warp_size || item_ct1.get_local_id(2) < ni) ? scale * x[4 * item_ct1.get_local_id(2) + l] : 0.0f;
    }

    ಭಿನ್ನ amax = sycl::fabs(vals[0]);
    ಭಿನ್ನ sum  = vals[0];
#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ l = 1; l < ಪೂರ್ಣಾಙ್ಕ(ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ)); ++l) {
        amax = sycl::fmax(amax, sycl::fabs(vals[l]));
        sum += vals[l];
    }
#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ mask = QI8_1/2; mask > 0; mask >>= 1) {
        amax = sycl::fmax(
            amax, dpct::permute_sub_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(), amax, mask));
        sum += dpct::permute_sub_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(), sum, mask);
    }

    ಧ್ರುವ ಭಿನ್ನ d = amax / 127;
    ಪೂರ್ಣಾಙ್ಕ q32 = 0;
    int8_t * q8 = (int8_t *) &q32;

    ಯದಿ (d != 0.0f) {
#pragma unroll
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ l = 0; l < ಪೂರ್ಣಾಙ್ಕ(ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ)); ++l) {
            q8[l] = sycl::round(vals[l] / d);
        }
    }

    yq32[item_ct1.get_local_id(2)] = q32;
    ಯದಿ (item_ct1.get_local_id(2) % QI8_1 == 0 && (ni == warp_size || item_ct1.get_local_id(2) < ni)) {
        ಯದಿ (std::is_same<Tds, sycl::half2>::value) {
            ((sycl::half2  *) yds)[item_ct1.get_local_id(2)/QI8_1] =  make_half2(d, sum);
        } ಅನ್ಯಥಾ {
            ((sycl::float2 *) yds)[item_ct1.get_local_id(2)/QI8_1] = make_float2(d, sum);
        }
    }
}

ಪ್ರಕಾರನಿರ್ದೇಶ ಶೂನ್ಯ (*dequantize_V_t)(ಧ್ರುವ ಶೂನ್ಯ *, ಶೂನ್ಯ *, ಧ್ರುವ int64_t);

template <typename T, ಪೂರ್ಣಾಙ್ಕ ne>
ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_V_f16(ಧ್ರುವ ಶೂನ್ಯ * __restrict__ vx, ಶೂನ್ಯ * __restrict__ dst, ಧ್ರುವ int64_t i0) {
    ಯದಿ constexpr (std::is_same_v<T, sycl::half>) {
        ggml_sycl_memcpy_1<ne * ಪರಿಮಾಣ(sycl::half)>(dst, (ಧ್ರುವ sycl::half *) vx + i0);
    } ಅನ್ಯಥಾ ಯದಿ constexpr (std::is_same_v<T, ಭಿನ್ನ>) {
        static_assert(ne % 2 == 0, "bad ne");
        sycl::half2 tmp[ne / 2];
        ggml_sycl_memcpy_1<ne * ಪರಿಮಾಣ(sycl::half)>(tmp, (ಧ್ರುವ sycl::half *) vx + i0);
        sycl::float2 * dst_f2 = (sycl::float2 *) dst;
#pragma unroll
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ l = 0; l < ne/2; ++l) {
            dst_f2[l] = tmp[l].template convert<ಭಿನ್ನ, sycl::rounding_mode::automatic>();
        }
    } ಅನ್ಯಥಾ {
        static_assert(std::is_same_v<T, ಶೂನ್ಯ>, "unsupported type");
    }
}

template <typename T, ಪೂರ್ಣಾಙ್ಕ ne>
ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_V_q4_0(ಧ್ರುವ ಶೂನ್ಯ * __restrict__ vx, ಶೂನ್ಯ * __restrict__ dst, ಧ್ರುವ int64_t i0) {
    ಧ್ರುವ block_q4_0 * x = (ಧ್ರುವ block_q4_0 *) vx;

    ಧ್ರುವ int64_t ib    =  i0          /  QK4_0;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ     iqs   =  i0          % (QK4_0/2);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ     shift = (i0 % QK4_0) / (QK4_0/2);

    ಪೂರ್ಣಾಙ್ಕ q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_sycl_memcpy_1<ne, 2>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;
    q = dpct::vectorized_ದ್ವಿಮಾನ<sycl::char4>(q, 0x08080808, dpct::sub_sat());

    ಧ್ರುವ int8_t * q8 = (ಧ್ರುವ int8_t *) &q;

#ifdef GGML_SYCL_F16
    ಯದಿ constexpr (std::is_same_v<T, sycl::half>) {
        ಧ್ರುವ sycl::half2 d = sycl::half2(x[ib].d);

#pragma unroll
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * sycl::half2(q8[l0 + 0], q8[l0 + 1]);
        }
    } ಅನ್ಯಥಾ
#endif // GGML_SYCL_F16
    ಯದಿ constexpr (std::is_same_v<T, ಭಿನ್ನ>) {
        ಧ್ರುವ ಭಿನ್ನ d = x[ib].d;

#pragma unroll
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ l = 0; l < ne; ++l) {
            ((ಭಿನ್ನ *) dst)[l] = d * q8[l];
        }
    } ಅನ್ಯಥಾ {
        static_assert(std::is_same_v<T, ಶೂನ್ಯ>, "bad type");
    }
}

template <typename T, ಪೂರ್ಣಾಙ್ಕ ne>
ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_V_q4_1(ಧ್ರುವ ಶೂನ್ಯ * __restrict__ vx, ಶೂನ್ಯ * __restrict__ dst, ಧ್ರುವ int64_t i0) {
    ಧ್ರುವ block_q4_1 * x = (ಧ್ರುವ block_q4_1 *) vx;

    ಧ್ರುವ int64_t ib    =  i0          /  QK4_1;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ     iqs   =  i0          % (QK4_1/2);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ     shift = (i0 % QK4_1) / (QK4_1/2);

    ಪೂರ್ಣಾಙ್ಕ q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_sycl_memcpy_1<ne>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    ಧ್ರುವ int8_t * q8 = (ಧ್ರುವ int8_t *) &q;

#ifdef GGML_SYCL_F16
    ಯದಿ constexpr (std::is_same_v<T, sycl::half>) {
        ಧ್ರುವ sycl::half2 dm = x[ib].dm;
        ಧ್ರುವ sycl::half2 d  = sycl::half2(dm[0]);
        ಧ್ರುವ sycl::half2 m  = sycl::half2(dm[1]);

#pragma unroll
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * sycl::half2(q8[l0 + 0], q8[l0 + 1]) + m;
        }
    } ಅನ್ಯಥಾ
#endif // GGML_SYCL_F16
    ಯದಿ constexpr (std::is_same_v<T, ಭಿನ್ನ>) {
        ಧ್ರುವ sycl::float2 dm = (x[ib].dm).template convert<ಭಿನ್ನ, sycl::rounding_mode::automatic>();

#pragma unroll
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ l = 0; l < ne; ++l) {
            ((ಭಿನ್ನ *) dst)[l] = dm.x() * q8[l] + dm.y();
        }
    } ಅನ್ಯಥಾ {
        static_assert(std::is_same_v<T, ಶೂನ್ಯ>, "bad type");
    }
}

template <typename T, ಪೂರ್ಣಾಙ್ಕ ne>
ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_V_q5_0(ಧ್ರುವ ಶೂನ್ಯ * __restrict__ vx, ಶೂನ್ಯ * __restrict__ dst, ಧ್ರುವ int64_t i0) {
    ಧ್ರುವ block_q5_0 * x = (ಧ್ರುವ block_q5_0 *) vx;

    ಧ್ರುವ int64_t ib    =  i0          /  QK5_0;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ     idq   =  i0          %  QK5_0;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ     iqs   =  i0          % (QK5_0/2);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ     shift = (i0 % QK5_0) / (QK5_0/2);

    ಪೂರ್ಣಾಙ್ಕ q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_sycl_memcpy_1<ne, 2>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    {
        ಪೂರ್ಣಾಙ್ಕ qh;
        ggml_sycl_memcpy_1<ne, 2>(&qh, x[ib].qh);
#pragma unroll
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ l = 0; l < ne; ++l) {
            q |= ((qh >> (idq + l)) & 0x00000001) << (8*l + 4);
        }
    }

    q = dpct::vectorized_ದ್ವಿಮಾನ<sycl::char4>(q, 0x10101010, dpct::sub_sat());

    ಧ್ರುವ int8_t * q8 = (ಧ್ರುವ int8_t *) &q;

#ifdef GGML_SYCL_F16
    ಯದಿ constexpr (std::is_same_v<T, sycl::half>) {
        ಧ್ರುವ sycl::half2 d = sycl::half2(x[ib].d);

#pragma unroll
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * sycl::half2(q8[l0 + 0], q8[l0 + 1]);
        }
    } ಅನ್ಯಥಾ
#endif // GGML_SYCL_F16
    ಯದಿ constexpr (std::is_same_v<T, ಭಿನ್ನ>) {
        ಧ್ರುವ ಭಿನ್ನ d = x[ib].d;

#pragma unroll
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ l = 0; l < ne; ++l) {
            ((ಭಿನ್ನ *) dst)[l] = d * q8[l];
        }
    } ಅನ್ಯಥಾ {
        static_assert(std::is_same_v<T, ಶೂನ್ಯ>, "bad type");
    }
}

template <typename T, ಪೂರ್ಣಾಙ್ಕ ne>
ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_V_q5_1(ಧ್ರುವ ಶೂನ್ಯ * __restrict__ vx, ಶೂನ್ಯ * __restrict__ dst, ಧ್ರುವ int64_t i0) {
    ಧ್ರುವ block_q5_1 * x = (ಧ್ರುವ block_q5_1 *) vx;

    ಧ್ರುವ int64_t ib    =  i0          /  QK5_1;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ     idq   =  i0          %  QK5_1;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ     iqs   =  i0          % (QK5_1/2);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ     shift = (i0 % QK5_1) / (QK5_1/2);

    ಪೂರ್ಣಾಙ್ಕ q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_sycl_memcpy_1<ne>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    {
        ಪೂರ್ಣಾಙ್ಕ qh;
        ggml_sycl_memcpy_1<ne>(&qh, x[ib].qh);
#pragma unroll
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ l = 0; l < ne; ++l) {
            q |= ((qh >> (idq + l)) & 0x00000001) << (8*l + 4);
        }
    }

    ಧ್ರುವ int8_t * q8 = (ಧ್ರುವ int8_t *) &q;

#ifdef GGML_SYCL_F16
    ಯದಿ constexpr (std::is_same_v<T, sycl::half>) {
        ಧ್ರುವ sycl::half2 dm = x[ib].dm;
        ಧ್ರುವ sycl::half2 d  = sycl::half2(dm[0]);
        ಧ್ರುವ sycl::half2 m  = sycl::half2(dm[1]);

#pragma unroll
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * sycl::half2(q8[l0 + 0], q8[l0 + 1]) + m;
        }
    } ಅನ್ಯಥಾ
#endif // GGML_SYCL_F16
    ಯದಿ constexpr (std::is_same_v<T, ಭಿನ್ನ>) {
        ಧ್ರುವ sycl::float2 dm = (x[ib].dm).template convert<ಭಿನ್ನ, sycl::rounding_mode::automatic>();

#pragma unroll
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ l = 0; l < ne; ++l) {
            ((ಭಿನ್ನ *) dst)[l] = dm.x() * q8[l] + dm.y();
        }
    } ಅನ್ಯಥಾ {
        static_assert(std::is_same_v<T, ಶೂನ್ಯ>, "bad type");
    }
}

template <typename T, ಪೂರ್ಣಾಙ್ಕ ne>
ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_V_q8_0(ಧ್ರುವ ಶೂನ್ಯ * __restrict__ vx, ಶೂನ್ಯ * __restrict__ dst, ಧ್ರುವ int64_t i0) {
    ಧ್ರುವ block_q8_0 * x = (ಧ್ರುವ block_q8_0 *) vx;

    ಧ್ರುವ int64_t ib  = i0 / QK8_0;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ     iqs = i0 % QK8_0;

    static_assert(ne % 2 == 0, "bad ne");
    int8_t qs[ne];
    ggml_sycl_memcpy_1<ne, 2>(qs, x[ib].qs + iqs);

#ifdef GGML_SYCL_F16
    ಯದಿ constexpr (std::is_same<T, sycl::half>::value) {
        ಧ್ರುವ sycl::half2 d = sycl::half2(x[ib].d);

#pragma unroll
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * make_half2(qs[l0 + 0], qs[l0 + 1]);
        }
    } ಅನ್ಯಥಾ
#endif // GGML_SYCL_F16
    ಯದಿ constexpr (std::is_same<T, ಭಿನ್ನ>::value) {
        ಧ್ರುವ ಭಿನ್ನ d = x[ib].d;

#pragma unroll
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ l = 0; l < ne; ++l) {
            ((ಭಿನ್ನ *) dst)[l] = d * qs[l];
        }
    } ಅನ್ಯಥಾ {
        static_assert(std::is_same_v<T, ಶೂನ್ಯ>, "unsupported type");
    }
}

template <ಪೂರ್ಣಾಙ್ಕ type_K, ಪೂರ್ಣಾಙ್ಕ D, ಪೂರ್ಣಾಙ್ಕ nthreads, ಪೂರ್ಣಾಙ್ಕ warp_size>
constexpr vec_dot_KQ_t get_vec_dot_KQ() {
    ಯದಿ constexpr (type_K == GGML_TYPE_F16) {
        ಪ್ರತ್ಯಾಗಮ vec_dot_fattn_vec_KQ_f16<D, nthreads>;
    } ಅನ್ಯಥಾ ಯದಿ constexpr (type_K == GGML_TYPE_Q4_0) {
        ಪ್ರತ್ಯಾಗಮ vec_dot_fattn_vec_KQ_q4_0<D, nthreads, warp_size>;
    } ಅನ್ಯಥಾ ಯದಿ constexpr (type_K == GGML_TYPE_Q4_1) {
        ಪ್ರತ್ಯಾಗಮ vec_dot_fattn_vec_KQ_q4_1<D, nthreads, warp_size>;
    } ಅನ್ಯಥಾ ಯದಿ constexpr (type_K == GGML_TYPE_Q5_0) {
        ಪ್ರತ್ಯಾಗಮ vec_dot_fattn_vec_KQ_q5_0<D, nthreads, warp_size>;
    } ಅನ್ಯಥಾ ಯದಿ constexpr (type_K == GGML_TYPE_Q5_1) {
        ಪ್ರತ್ಯಾಗಮ vec_dot_fattn_vec_KQ_q5_1<D, nthreads, warp_size>;
    } ಅನ್ಯಥಾ ಯದಿ constexpr (type_K == GGML_TYPE_Q8_0) {
        ಪ್ರತ್ಯಾಗಮ vec_dot_fattn_vec_KQ_q8_0<D, nthreads, warp_size>;
    } ಅನ್ಯಥಾ {
        static_assert(type_K == -1, "bad type");
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }
}

template <ಪೂರ್ಣಾಙ್ಕ type_V, typename T, ಪೂರ್ಣಾಙ್ಕ ne>
constexpr dequantize_V_t get_dequantize_V() {
    ಯದಿ constexpr (type_V == GGML_TYPE_F16) {
        ಪ್ರತ್ಯಾಗಮ dequantize_V_f16<T, ne>;
    } ಅನ್ಯಥಾ ಯದಿ constexpr (type_V == GGML_TYPE_Q4_0) {
        ಪ್ರತ್ಯಾಗಮ dequantize_V_q4_0<T, ne>;
    } ಅನ್ಯಥಾ ಯದಿ constexpr (type_V == GGML_TYPE_Q4_1) {
        ಪ್ರತ್ಯಾಗಮ dequantize_V_q4_1<T, ne>;
    } ಅನ್ಯಥಾ ಯದಿ constexpr (type_V == GGML_TYPE_Q5_0) {
        ಪ್ರತ್ಯಾಗಮ dequantize_V_q5_0<T, ne>;
    } ಅನ್ಯಥಾ ಯದಿ constexpr (type_V == GGML_TYPE_Q5_1) {
        ಪ್ರತ್ಯಾಗಮ dequantize_V_q5_1<T, ne>;
    } ಅನ್ಯಥಾ ಯದಿ constexpr (type_V == GGML_TYPE_Q8_0) {
        ಪ್ರತ್ಯಾಗಮ dequantize_V_q8_0<T, ne>;
    } ಅನ್ಯಥಾ {
        static_assert(type_V == -1, "bad type");
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }
}

template <ಪೂರ್ಣಾಙ್ಕ ncols1, ಪೂರ್ಣಾಙ್ಕ warp_size>
ಸ್ಥಿರ ಶೂನ್ಯ flash_attn_mask_to_KV_max(ಧ್ರುವ sycl::half2 * __restrict__ mask,
                                      ಪೂರ್ಣಾಙ್ಕ * __restrict__ KV_max,
                                      ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne30,
                                      ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s31,
                                      ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s33,
                                      ಪೂರ್ಣಾಙ್ಕ *     buf_iw) {
    auto      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne31     = item_ct1.get_group_range(2);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ tid      = item_ct1.get_local_id(2);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ sequence = item_ct1.get_group(1);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ jt       = item_ct1.get_group(2);

    mask += sequence*s33 + jt*ncols1*s31;

    ಯದಿ (tid < warp_size) {
        buf_iw[tid] = 1;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);

    ಪೂರ್ಣಾಙ್ಕ KV_max_sj = (ne30 - 1) * FATTN_KQ_STRIDE;
    ಪ್ರತಿ (; KV_max_sj >= 0; KV_max_sj -= FATTN_KQ_STRIDE) {
        ಪೂರ್ಣಾಙ್ಕ all_inf = 1;

#pragma unroll
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < ncols1; ++j) {
            ಧ್ರುವ sycl::float2 tmp =
                mask[j * s31 + KV_max_sj / 2 + tid].template convert<ಭಿನ್ನ, sycl::rounding_mode::automatic>();
            all_inf = all_inf && ಪೂರ್ಣಾಙ್ಕ(sycl::isinf((ಭಿನ್ನ) (tmp.x()))) && ಪೂರ್ಣಾಙ್ಕ(sycl::isinf((ಭಿನ್ನ) (tmp.y())));
        }

        all_inf = warp_reduce_all<warp_size>(all_inf);
        ಯದಿ (tid % warp_size == 0) {
            buf_iw[tid / warp_size] = all_inf;
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);
        all_inf = buf_iw[tid % warp_size];
        item_ct1.barrier(sycl::access::fence_space::local_space);
        all_inf = warp_reduce_all<warp_size>(all_inf);

        ಯದಿ (!all_inf) {
            ಭಙ್ಗ;
        }
    }

    // If the break in the loop was not triggered, KV_max_sj is now -FATTN_KQ_STRIDE.
    // If the break was triggered it's the lower edge of the tile with the first non-masked values.
    // In either case, walk back the decrementation by FATTN_KQ_STRIDE.
    KV_max_sj += FATTN_KQ_STRIDE;

    ಯದಿ (item_ct1.get_local_id(2) != 0) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    KV_max[sequence*ne31 + jt] = KV_max_sj;
}

template <ಪೂರ್ಣಾಙ್ಕ D, ಪೂರ್ಣಾಙ್ಕ ncols1, ಪೂರ್ಣಾಙ್ಕ ncols2>  // D == head size

ಸ್ಥಿರ ಶೂನ್ಯ flash_attn_stream_k_fixup(ಭಿನ್ನ * __restrict__ dst,
                                      ಧ್ರುವ sycl::float2 * __restrict__ dst_fixup,
                                      ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01,
                                      ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne02,
                                      ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne03,
                                      ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne11,
                                      ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne12,
                                      ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nbatch_fa) {
    auto          item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr ಪೂರ್ಣಾಙ್ಕ ncols    = ncols1 * ncols2;

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ bidx0 = item_ct1.get_group(2);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ j     = item_ct1.get_group(1);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ c     = item_ct1.get_group(0);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ jc    = j*ncols2 + c;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ tid   = item_ct1.get_local_id(2);

    ಧ್ರುವ ಭಿನ್ನ * dst_fixup_data = ((ಧ್ರುವ ಭಿನ್ನ *) dst_fixup) + item_ct1.get_group_range(2) * (2 * 2 * ncols);

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ iter_k     = (ne11      + (nbatch_fa - 1)) / nbatch_fa;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ iter_j     = (ne01      + (ncols1    - 1)) / ncols1;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ iter_z_gqa = (gqa_ratio + (ncols2    - 1)) / ncols2;

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ kbc0 = int64_t(bidx0 + 0) * (iter_k * iter_j * iter_z_gqa * ne12 * ne03) / item_ct1.get_group_range(2);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ kbc0_stop =
        int64_t(bidx0 + 1) * (iter_k * iter_j * iter_z_gqa * ne12 * ne03) / item_ct1.get_group_range(2);

    ಧ್ರುವ ತಾರ್ಕಿಕ did_not_have_any_data   = kbc0 == kbc0_stop;
    ಧ್ರುವ ತಾರ್ಕಿಕ wrote_beginning_of_tile = kbc0 % iter_k == 0;
    ಧ್ರುವ ತಾರ್ಕಿಕ did_not_write_last      = kbc0/iter_k == kbc0_stop/iter_k && kbc0_stop % iter_k != 0;
    ಯದಿ (did_not_have_any_data || wrote_beginning_of_tile || did_not_write_last) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    // z_KV == K/V head index, zt_gqa = Q head start index per K/V head, jt = token position start index
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ sequence =  kbc0 /(iter_k*iter_j*iter_z_gqa*ne12);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ z_KV     = (kbc0 - iter_k*iter_j*iter_z_gqa*ne12 * sequence)/(iter_k*iter_j*iter_z_gqa);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ zt_gqa   = (kbc0 - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV)/(iter_k*iter_j);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ jt       = (kbc0 - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV - iter_k*iter_j * zt_gqa) / iter_k;

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ zt_Q = z_KV*gqa_ratio + zt_gqa*ncols2; // Global Q head start index.

    ಯದಿ (jt*ncols1 + j >= ne01 || zt_gqa*ncols2 + c >= gqa_ratio) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    dst += sequence*ne02*ne01*D + jt*ne02*(ncols1*D) + zt_Q*D + (j*ne02 + c)*D + tid;

    // Load the partial result that needs a fixup:
    ಭಿನ್ನ dst_val = 0.0f;
    ಭಿನ್ನ max_val = 0.0f;
    ಭಿನ್ನ rowsum  = 0.0f;
    {
        dst_val = *dst;

        ಧ್ರುವ sycl::float2 tmp = dst_fixup[bidx0 * ncols + jc];
        max_val                = tmp.x();
        rowsum                 = tmp.y();
    }

    // Iterate over previous blocks and compute the combined results.
    // All SYCL blocks that get here must have a previous block that needs a fixup.
    ಪೂರ್ಣಾಙ್ಕ bidx = bidx0 - 1;
    ಪೂರ್ಣಾಙ್ಕ kbc_stop = kbc0;
    ಯಾವತ್(ಸತ್ಯ) {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ kbc = int64_t(bidx) * (iter_k * iter_j * iter_z_gqa * ne12 * ne03) / item_ct1.get_group_range(2);
        ಯದಿ (kbc == kbc_stop) { // Did not have any data.
            bidx--;
            kbc_stop = kbc;
            ಅನುವರ್ತ;
        }

        ಧ್ರುವ ಭಿನ್ನ dst_add = dst_fixup_data[bidx*ncols*D + jc*D + tid];

        ಧ್ರುವ sycl::float2 tmp = dst_fixup[(item_ct1.get_group_range(2) + bidx) * ncols + jc];

        // Scale the current and new value accumulators depending on the max. values.
        ಧ್ರುವ ಭಿನ್ನ max_val_new = sycl::fmax(max_val, tmp.x());

        ಧ್ರುವ ಭಿನ್ನ diff_val = max_val - max_val_new;
        ಧ್ರುವ ಭಿನ್ನ diff_add = tmp.x() - max_val_new;

        ಧ್ರುವ ಭಿನ್ನ scale_val = diff_val >= SOFTMAX_FTZ_THRESHOLD ? sycl::native::exp(diff_val) : 0.0f;
        ಧ್ರುವ ಭಿನ್ನ scale_add = diff_add >= SOFTMAX_FTZ_THRESHOLD ? sycl::native::exp(diff_add) : 0.0f;

        dst_val = scale_val*dst_val + scale_add*dst_add;
        rowsum  = scale_val * rowsum + scale_add * tmp.y();

        max_val = max_val_new;

        // If this block started in a previous tile we are done and don't need to combine additional partial results.
        ಯದಿ (kbc % iter_k == 0 || kbc/iter_k < kbc0/iter_k) {
            ಭಙ್ಗ;
        }
        bidx--;
        kbc_stop = kbc;
    }

    // Write back final result:
    *dst = dst_val / rowsum;
}

template <ಪೂರ್ಣಾಙ್ಕ D>  // D == head size

ಸ್ಥಿರ ಶೂನ್ಯ flash_attn_combine_results(ಧ್ರುವ ಭಿನ್ನ * __restrict__ VKQ_parts,
                                       ಧ್ರುವ sycl::float2 * __restrict__ VKQ_meta,
                                       ಭಿನ್ನ * __restrict__ dst,
                                       ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ parallel_blocks,
                                       uint8_t * dpct_local) {
    // Dimension 0: threadIdx.x
    // Dimension 1: blockIdx.x
    // Dimension 2: blockIdx.y
    // Dimension 3: blockIdx.z
    // Memory layout is permuted with [0, 2, 1, 3]

    auto      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01     = item_ct1.get_group_range(2);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne02     = item_ct1.get_group_range(1);

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ col      = item_ct1.get_group(2);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ head     = item_ct1.get_group(1);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ sequence = item_ct1.get_group(0);

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ j_dst_unrolled = (sequence*ne01 + col)*ne02 + head;

    VKQ_parts += j_dst_unrolled * parallel_blocks*D;
    VKQ_meta  += j_dst_unrolled * parallel_blocks;
    dst       += j_dst_unrolled *                 D;

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ tid = item_ct1.get_local_id(2);
    __builtin_assume(tid < D);

    auto meta = (sycl::float2 *) dpct_local;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = tid; i < 2*parallel_blocks; i += D) {
        ((ಭಿನ್ನ *) meta)[i] = ((ಧ್ರುವ ಭಿನ್ನ *)VKQ_meta) [i];
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    ಭಿನ್ನ kqmax = meta[0].x();
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ l = 1; l < parallel_blocks; ++l) {
        kqmax = sycl::max(kqmax, meta[l].x());
    }

    ಭಿನ್ನ VKQ_numerator   = 0.0f;
    ಭಿನ್ನ VKQ_denominator = 0.0f;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ l = 0; l < parallel_blocks; ++l) {
        ಧ್ರುವ ಭಿನ್ನ KQ_max_scale = sycl::native::exp(meta[l].x() - kqmax);

        VKQ_numerator   += KQ_max_scale * VKQ_parts[l*D + tid];
        VKQ_denominator += KQ_max_scale * meta[l].y();
    }

    dst[tid] = VKQ_numerator / VKQ_denominator;
}

// Carries the sub-group size (and grf_size_automatic when use_large_grf) as
// kernel properties via get(properties_tag), in place of the
// reqd_sub_group_size attribute and the deprecated parallel_for overload.
template <typename KernelFunc, uint32_t Size, ತಾರ್ಕಿಕ UseGrf>
ಸಂರಚನಾ props_kernel {
    KernelFunc kernel;

    ಶೂನ್ಯ operator()(sycl::nd_item<3> item) ಧ್ರುವ {
        kernel(item);
    }

    auto get(sycl::ext::oneapi::experimental::properties_tag) ಧ್ರುವ {
        ಯದಿ constexpr (UseGrf) {
            ಪ್ರತ್ಯಾಗಮ sycl::ext::oneapi::experimental::properties{
                sycl::ext::intel::experimental::grf_size_automatic,
                sycl::ext::oneapi::experimental::sub_group_size<Size> };
        } ಅನ್ಯಥಾ {
            ಪ್ರತ್ಯಾಗಮ sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::sub_group_size<Size> };
        }
    }
};

template <fattn_kernel_t fattn_kernel, ಪೂರ್ಣಾಙ್ಕ warp_size, ತಾರ್ಕಿಕ use_large_grf = ಅಸತ್ಯ>
ಸ್ಥಿರ ಶೂನ್ಯ lauch_kernel(
    dpct::dim3 group_range,
    dpct::dim3 local_range,
    queue_ptr q,
    unsigned ಪೂರ್ಣಾಙ್ಕ local_mem_size,
    ಧ್ರುವ ವರ್ಣ* __restrict__ Q,
    ಧ್ರುವ ವರ್ಣ* __restrict__ K,
    ಧ್ರುವ ವರ್ಣ* __restrict__ V,
    ಧ್ರುವ ವರ್ಣ* __restrict__ mask,
    ಧ್ರುವ ವರ್ಣ* __restrict__ sinks,
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ* __restrict__ KV_max,
    ಭಿನ್ನ* __restrict__ dst,
    sycl::float2* __restrict__ dst_meta,
    ಧ್ರುವ ಭಿನ್ನ scale,
    ಧ್ರುವ ಭಿನ್ನ max_bias,
    ಧ್ರುವ ಭಿನ್ನ m0,
    ಧ್ರುವ ಭಿನ್ನ m1,
    ಧ್ರುವ uint32_t n_head_log2,
    ಧ್ರುವ ಭಿನ್ನ logit_softcap,
    ಧ್ರುವ int32_t ne00,
    ಧ್ರುವ sycl::uint3 ne01,
    ಧ್ರುವ int32_t ne02,
    ಧ್ರುವ int32_t ne03,
    ಧ್ರುವ int32_t nb01,
    ಧ್ರುವ int32_t nb02,
    ಧ್ರುವ int32_t nb03,
    ಧ್ರುವ int32_t ne10,
    ಧ್ರುವ int32_t ne11,
    ಧ್ರುವ int32_t ne12,
    ಧ್ರುವ int32_t ne13,
    ಧ್ರುವ int32_t nb11,
    ಧ್ರುವ int32_t nb12,
    ಧ್ರುವ int64_t nb13,
    ಧ್ರುವ int32_t nb21,
    ಧ್ರುವ int32_t nb22,
    ಧ್ರುವ int64_t nb23,
    ಧ್ರುವ int32_t ne31,
    ಧ್ರುವ int32_t ne32,
    ಧ್ರುವ int32_t ne33,
    ಧ್ರುವ int32_t nb31,
    ಧ್ರುವ int32_t nb32,
    ಧ್ರುವ int64_t nb33) {
    GGML_UNUSED(local_mem_size);

    ಧ್ರುವ auto rng = sycl::nd_range<3>(
        static_cast<sycl::range<3>>(group_range * local_range),
        static_cast<sycl::range<3>>(local_range));

    ಧ್ರುವ auto kernel = [=](sycl::nd_item<3> item_ct1) {
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
template <ಪೂರ್ಣಾಙ್ಕ DV, ಪೂರ್ಣಾಙ್ಕ ncols1, ಪೂರ್ಣಾಙ್ಕ ncols2, fattn_kernel_t fattn_kernel, ಪೂರ್ಣಾಙ್ಕ warp_size, ತಾರ್ಕಿಕ use_large_grf = ಅಸತ್ಯ>
ಶೂನ್ಯ launch_fattn(
    ggml_backend_sycl_context & ctx, ggml_tensor * dst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nwarps, ಧ್ರುವ size_t nbytes_shared,
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nbatch_fa, ಧ್ರುವ ತಾರ್ಕಿಕ need_f16_K, ಧ್ರುವ ತಾರ್ಕಿಕ need_f16_V, ಧ್ರುವ ತಾರ್ಕಿಕ stream_k) {

    constexpr ಪೂರ್ಣಾಙ್ಕ ncols = ncols1 * ncols2;

    ಧ್ರುವ ggml_tensor * Q = dst->src[0];
    ಧ್ರುವ ggml_tensor * K = dst->src[1];
    ಧ್ರುವ ggml_tensor * V = dst->src[2];

    ಧ್ರುವ ತಾರ್ಕಿಕ V_is_K_view = V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));

    ಧ್ರುವ ggml_tensor * mask  = dst->src[3];
    ಧ್ರುವ ggml_tensor * sinks = dst->src[4];

    ggml_tensor * KQV = dst;

    GGML_ASSERT(Q->type == GGML_TYPE_F32);
    GGML_ASSERT(KQV->type == GGML_TYPE_F32);

    GGML_ASSERT(Q->nb[0] == ggml_element_size(Q));
    GGML_ASSERT(K->nb[0] == ggml_element_size(K));
    GGML_ASSERT(V->nb[0] == ggml_element_size(V));

    GGML_ASSERT(!mask || mask->type == GGML_TYPE_F16);

    ggml_sycl_pool & pool = ctx.pool();
    dpct::queue_ptr  main_stream = ctx.stream();
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ id  = ggml_sycl_get_device();
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nsm = ggml_sycl_info().devices[id].nsm;

    ಧ್ರುವ ggml_sycl_fattn_extra  extra = ggml_sycl_fattn_get_extra(dst);
    ggml_sycl_pool_alloc<ಪೂರ್ಣಾಙ್ಕ>    KV_max(pool);
    ggml_sycl_pool_alloc<ಭಿನ್ನ>  dst_tmp(pool);
    ggml_sycl_pool_alloc<sycl::float2> dst_tmp_meta(pool);

    ಧ್ರುವ ವರ್ಣ * K_data = (ಧ್ರುವ ವರ್ಣ *) K->data;
    size_t nb11 = K->nb[1];
    size_t nb12 = K->nb[2];
    size_t nb13 = K->nb[3];

    ಧ್ರುವ ವರ್ಣ * V_data = (ಧ್ರುವ ವರ್ಣ *) V->data;
    size_t nb21 = V->nb[1];
    size_t nb22 = V->nb[2];
    size_t nb23 = V->nb[3];

    ಯದಿ (need_f16_K && K->type != GGML_TYPE_F16) {
        ಧ್ರುವ size_t bs = ggml_blck_size(K->type);
        ಧ್ರುವ size_t ts = ggml_type_size(K->type);

        GGML_ASSERT(extra.K_buffer_ptr);
        sycl::half * K_f16_ptr = (sycl::half *) extra.K_buffer_ptr;
        ಯದಿ (ggml_is_contiguously_allocated(K)) {
            to_fp16_sycl_t to_fp16 = ggml_get_to_fp16_sycl(K->type, dst);
            to_fp16(K_data, K_f16_ptr, ggml_nelements(K), main_stream);

            nb11 = nb11 * bs * ಪರಿಮಾಣ(sycl::half) / ts;
            nb12 = nb12 * bs * ಪರಿಮಾಣ(sycl::half) / ts;
            nb13 = nb13 * bs * ಪರಿಮಾಣ(sycl::half) / ts;
        } ಅನ್ಯಥಾ {
            GGML_ASSERT(K->nb[0] == ts);
            to_fp16_nc_sycl_t to_fp16 = ggml_get_to_fp16_nc_sycl(K->type);
            ಧ್ರುವ int64_t s01 = nb11 / ts;
            ಧ್ರುವ int64_t s02 = nb12 / ts;
            ಧ್ರುವ int64_t s03 = nb13 / ts;
            to_fp16(K_data, K_f16_ptr, K->ne[0], K->ne[1], K->ne[2], K->ne[3], s01, s02, s03, main_stream);

            nb11 = K->ne[0] * ಪರಿಮಾಣ(sycl::half);
            nb12 = K->ne[1] * nb11;
            nb13 = K->ne[2] * nb12;
        }
        K_data = (ವರ್ಣ *) K_f16_ptr;
    }

    ಯದಿ (need_f16_V && V->type != GGML_TYPE_F16) {
        ಯದಿ (V_is_K_view) {
            V_data = K_data;
            nb21   = nb11;
            nb22   = nb12;
            nb23   = nb13;
        } ಅನ್ಯಥಾ {
            ಧ್ರುವ size_t bs = ggml_blck_size(V->type);
            ಧ್ರುವ size_t ts = ggml_type_size(V->type);

            GGML_ASSERT(extra.V_buffer_ptr);
            sycl::half * V_f16_ptr = (sycl::half *) extra.V_buffer_ptr;
            ಯದಿ (ggml_is_contiguously_allocated(V)) {
                to_fp16_sycl_t to_fp16 = ggml_get_to_fp16_sycl(V->type, dst);
                to_fp16(V_data, V_f16_ptr, ggml_nelements(V), main_stream);
                V_data = (ವರ್ಣ *) V_f16_ptr;

                nb21 = nb21 * bs * ಪರಿಮಾಣ(sycl::half) / ts;
                nb22 = nb22 * bs * ಪರಿಮಾಣ(sycl::half) / ts;
                nb23 = nb23 * bs * ಪರಿಮಾಣ(sycl::half) / ts;
            } ಅನ್ಯಥಾ {
                GGML_ASSERT(V->nb[0] == ts);
                to_fp16_nc_sycl_t to_fp16 = ggml_get_to_fp16_nc_sycl(V->type);
                ಧ್ರುವ int64_t s01 = nb21 / ts;
                ಧ್ರುವ int64_t s02 = nb22 / ts;
                ಧ್ರುವ int64_t s03 = nb23 / ts;
                to_fp16(V_data, V_f16_ptr, V->ne[0], V->ne[1], V->ne[2], V->ne[3], s01, s02, s03, main_stream);

                nb21 = V->ne[0] * ಪರಿಮಾಣ(sycl::half);
                nb22 = V->ne[1] * nb21;
                nb23 = V->ne[2] * nb22;
            }
            V_data = (ವರ್ಣ *) V_f16_ptr;
        }
    }

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ntiles_x     = ((Q->ne[1] + ncols1 - 1) / ncols1);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ gqa_ratio    = Q->ne[2] / K->ne[2];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ntiles_z_gqa = ((gqa_ratio + ncols2 - 1) / ncols2);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ntiles_total = ntiles_x * ntiles_z_gqa * K->ne[2] * Q->ne[3];

    // Optional optimization where the mask is scanned to determine whether part of the calculation can be skipped.
    // Only worth the overhead if there is at lease one FATTN_KQ_STRIDE x FATTN_KQ_STRIDE square to be skipped or
    //     multiple sequences of possibly different lengths.
    ಯದಿ (mask && K->ne[1] % FATTN_KQ_STRIDE == 0 && (Q->ne[1] >= 1024 || Q->ne[3] > 1)) {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s31 = mask->nb[1] / ಪರಿಮಾಣ(sycl::half2);
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s33 = mask->nb[3] / ಪರಿಮಾಣ(sycl::half2);

        ಧ್ರುವ dpct::dim3 blocks_ಅಂಕೆ_KV_max(ntiles_x, Q->ne[3], 1);
        ಧ್ರುವ dpct::dim3 block_dim_KV_max(FATTN_KQ_STRIDE / 2, 1, 1);

        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne_KV_max = blocks_ಅಂಕೆ_KV_max.x*blocks_ಅಂಕೆ_KV_max.y;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ iter_k = K->ne[1] / FATTN_KQ_STRIDE;

        KV_max.alloc(ne_KV_max);
        {
            dpct::has_capability_or_fail(main_stream->get_device(), { sycl::aspect::fp16 });

            main_stream->submit([&](sycl::handler & cgh) {
                sycl::local_accessor<ಪೂರ್ಣಾಙ್ಕ, 1> buf_iw_acc_ct1(sycl::range<1>(warp_size), cgh);

                auto mask_data_ct0  = (ಧ್ರುವ sycl::half2 *) mask->data;
                auto KV_max_ptr_ct1 = KV_max.ptr;

                cgh.parallel_for(sycl::nd_range<3>(blocks_ಅಂಕೆ_KV_max * block_dim_KV_max, block_dim_KV_max),
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

    ಧ್ರುವ dpct::dim3 block_dim(warp_size, nwarps, 1);

    // Max. number of active blocks limited by occupancy.
    ಪೂರ್ಣಾಙ್ಕ max_blocks_per_sm = ggml_sycl_info().devices[id].max_wg_per_cu;
    ಪೂರ್ಣಾಙ್ಕ parallel_blocks = max_blocks_per_sm;
    dpct::dim3 blocks_ಅಂಕೆ;
    ಯದಿ (stream_k) {
        // For short contexts it can be faster to have the SMs work on whole tiles because this lets us skip the fixup.
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ max_blocks = max_blocks_per_sm*nsm;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nblocks_stream_k = max_blocks;
        ಧ್ರುವ ತಾರ್ಕಿಕ use_stream_k = ಸತ್ಯ;

        blocks_ಅಂಕೆ.x = use_stream_k ? nblocks_stream_k : ntiles_total;
        blocks_ಅಂಕೆ.y = 1;
        blocks_ಅಂಕೆ.z = 1;

        ಯದಿ (ntiles_total % blocks_ಅಂಕೆ.x != 0) { // Fixup is only needed if the SMs work on fractional tiles.
            dst_tmp_meta.alloc((size_t(blocks_ಅಂಕೆ.x) * ncols * (2 + DV/2)));
        }
    } ಅನ್ಯಥಾ {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ntiles_KQ = (K->ne[1] + nbatch_fa - 1) / nbatch_fa; // Max. number of parallel blocks limited by tensor size.

        // parallel_blocks must not be larger than what the tensor size allows:
        parallel_blocks = std::min(parallel_blocks, ntiles_KQ);
        // todo fix the hard code change
        // parallel_blocks = ntiles_KQ;

        // If ntiles_total % blocks_per_wave != 0 then some efficiency is lost due to tail effects.
        // Test whether parallel_blocks can be set to a higher value for better efficiency.
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ blocks_per_wave = nsm * max_blocks_per_sm;
        ಪೂರ್ಣಾಙ್ಕ nwaves_best = 0;
        ಪೂರ್ಣಾಙ್ಕ efficiency_percent_best = 0;
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ parallel_blocks_test = parallel_blocks; parallel_blocks_test <= ntiles_KQ; ++parallel_blocks_test) {
            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nblocks_total = ntiles_total * parallel_blocks_test;
            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nwaves = (nblocks_total + blocks_per_wave - 1) / blocks_per_wave;
            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ efficiency_percent = 100 * nblocks_total / (nwaves*blocks_per_wave);

            // Stop trying configurations with more waves if we already have good efficiency to avoid excessive overhead.
            ಯದಿ (efficiency_percent_best >= 95 && nwaves > nwaves_best) {
                ಭಙ್ಗ;
            }

            ಯದಿ (efficiency_percent > efficiency_percent_best) {
                nwaves_best = nwaves;
                efficiency_percent_best = efficiency_percent;
                parallel_blocks = parallel_blocks_test;
            }
        }

        blocks_ಅಂಕೆ.x = ntiles_x;
        blocks_ಅಂಕೆ.y = parallel_blocks;
        blocks_ಅಂಕೆ.z = ntiles_z_gqa*K->ne[2]*Q->ne[3];

        ಯದಿ (parallel_blocks > 1) {
            dst_tmp.alloc(parallel_blocks*ggml_nelements(KQV));
            dst_tmp_meta.alloc(parallel_blocks*ggml_nrows(KQV));
        }
    }

    ಭಿನ್ನ scale         = 1.0f;
    ಭಿನ್ನ max_bias      = 0.0f;
    ಭಿನ್ನ logit_softcap = 0.0f;

    memcpy(&scale,         (ಧ್ರುವ ಭಿನ್ನ *) KQV->op_params + 0, ಪರಿಮಾಣ(ಭಿನ್ನ));
    memcpy(&max_bias,      (ಧ್ರುವ ಭಿನ್ನ *) KQV->op_params + 1, ಪರಿಮಾಣ(ಭಿನ್ನ));
    memcpy(&logit_softcap, (ಧ್ರುವ ಭಿನ್ನ *) KQV->op_params + 2, ಪರಿಮಾಣ(ಭಿನ್ನ));

    ಯದಿ (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    ಧ್ರುವ uint32_t n_head      = Q->ne[2];
    ಧ್ರುವ uint32_t n_head_log2 = 1u << uint32_t(floorf(log2f(ಭಿನ್ನ(n_head))));

    ಧ್ರುವ ಭಿನ್ನ m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    ಧ್ರುವ ಭಿನ್ನ m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    // TODO other tensor dimensions after removal of WMMA kernel:
    ಧ್ರುವ sycl::uint3 ne01 = init_fastdiv_ಮೌಲ್ಯಗಳು(Q->ne[1]);

    GGML_ASSERT(block_dim.x % warp_size == 0);

    lauch_kernel<fattn_kernel, warp_size, use_large_grf>(
        blocks_ಅಂಕೆ, block_dim, main_stream, (unsigned ಪೂರ್ಣಾಙ್ಕ) nbytes_shared, (ಧ್ರುವ ವರ್ಣ *) Q->data, K_data, V_data,
        mask ? ((ಧ್ರುವ ವರ್ಣ *) mask->data) : nullptr, sinks ? ((ಧ್ರುವ ವರ್ಣ *) sinks->data) : nullptr, KV_max.ptr,
        !stream_k && parallel_blocks > 1 ? dst_tmp.ptr : (ಭಿನ್ನ *) KQV->data, (sycl::float2 *)dst_tmp_meta.ptr, scale, max_bias, m0, m1,
        n_head_log2, logit_softcap, Q->ne[0], ne01, Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3], K->ne[0],
        K->ne[1], K->ne[2], K->ne[3], nb11, nb12, nb13, nb21, nb22, nb23, mask ? mask->ne[1] : 0,
        mask ? mask->ne[2] : 0, mask ? mask->ne[3] : 0, mask ? mask->nb[1] : 0, mask ? mask->nb[2] : 0,
        mask ? mask->nb[3] : 0);
    SYCL_CHECK(0);

    ಯದಿ (stream_k) {
        ಯದಿ (ntiles_total % blocks_ಅಂಕೆ.x != 0) { // Fixup is only needed if the SMs work on fractional tiles.
            ಧ್ರುವ dpct::dim3 block_dim_combine(DV, 1, 1);
            ಧ್ರುವ dpct::dim3 blocks_ಅಂಕೆ_combine = { blocks_ಅಂಕೆ.x, ncols1, ncols2 };

            main_stream->submit([&](sycl::handler & cgh) {
                auto KQV_data_ct0         = (ಭಿನ್ನ *) KQV->data;
                auto dst_tmp_meta_ptr_ct1 = dst_tmp_meta.ptr;
                auto Q_ne_ct2             = Q->ne[1];
                auto Q_ne_ct3             = Q->ne[2];
                auto Q_ne_ct4             = Q->ne[3];
                auto K_ne_ct5             = K->ne[1];
                auto K_ne_ct6             = K->ne[2];

                cgh.parallel_for(sycl::nd_range<3>(blocks_ಅಂಕೆ_combine * block_dim_combine, block_dim_combine),
                                 [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(warp_size)]] {
                                     GGML_UNUSED(item_ct1);
                                     flash_attn_stream_k_fixup<DV, ncols1, ncols2>(KQV_data_ct0, dst_tmp_meta_ptr_ct1,
                                                                                   Q_ne_ct2, Q_ne_ct3, Q_ne_ct4,
                                                                                   K_ne_ct5, K_ne_ct6, nbatch_fa);
                                 });
            });
        }
    } ಅನ್ಯಥಾ ಯದಿ (parallel_blocks > 1) {
        ಧ್ರುವ dpct::dim3 block_dim_combine(DV, 1, 1);
        ಧ್ರುವ dpct::dim3 blocks_ಅಂಕೆ_combine(Q->ne[1], Q->ne[2], Q->ne[3]);
        ಧ್ರುವ size_t     nbytes_shared_combine = parallel_blocks * ಪರಿಮಾಣ(sycl::float2);
        main_stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(sycl::range<1>(nbytes_shared_combine), cgh);

            auto dst_tmp_ptr_ct0      = dst_tmp.ptr;
            auto dst_tmp_meta_ptr_ct1 = dst_tmp_meta.ptr;
            auto KQV_data_ct2         = (ಭಿನ್ನ *) KQV->data;

            cgh.parallel_for(sycl::nd_range<3>(blocks_ಅಂಕೆ_combine * block_dim_combine, block_dim_combine),
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
