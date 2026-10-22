#ಸೂಚನೆ once

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


#ವ್ಯಾಖ್ಯಾನಿಸು FATTN_KQ_STRIDE       256
#ವ್ಯಾಖ್ಯಾನಿಸು HALF_MAX_HALF         sycl::half(65504.0f/2) // Use neg. of this instead of -INFINITY to initialize KQ max vals to avoid NaN upon subtraction.
#ವ್ಯಾಖ್ಯಾನಿಸು SOFTMAX_FTZ_ಮಿತಿ -20.0f                   // Softmax exp. of values smaller than this are flushed to zero to avoid NaNs.
#ವ್ಯಾಖ್ಯಾನಿಸು FATTN_KQ_MAX_ಸ್ಥಾನಾಂತರ (3.0f*0.6931f)

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಶೂನ್ಯ (*fattn_kernel_t)(
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* Q,
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* K,
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* V,
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* ಮುಖವಾಡ,
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* sinks,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ* KV_max,
    ದಶಮಾಂಶ* dst,
    sycl::float2* dst_meta,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ max_bias,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ m0,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ m1,
    ಸ್ಥಿರಾಂಕ uint32_t n_ತಲೆ_log2,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ logit_softcap,
    ಸ್ಥಿರಾಂಕ int32_t ne00,
    ಸ್ಥಿರಾಂಕ sycl::uint3 ne01,
    ಸ್ಥಿರಾಂಕ int32_t ne02,
    ಸ್ಥಿರಾಂಕ int32_t ne03,
    ಸ್ಥಿರಾಂಕ int32_t nb01,
    ಸ್ಥಿರಾಂಕ int32_t nb02,
    ಸ್ಥಿರಾಂಕ int32_t nb03,
    ಸ್ಥಿರಾಂಕ int32_t ne10,
    ಸ್ಥಿರಾಂಕ int32_t ne11,
    ಸ್ಥಿರಾಂಕ int32_t ne12,
    ಸ್ಥಿರಾಂಕ int32_t ne13,
    ಸ್ಥಿರಾಂಕ int32_t nb11,
    ಸ್ಥಿರಾಂಕ int32_t nb12,
    ಸ್ಥಿರಾಂಕ int64_t nb13,
    ಸ್ಥಿರಾಂಕ int32_t nb21,
    ಸ್ಥಿರಾಂಕ int32_t nb22,
    ಸ್ಥಿರಾಂಕ int64_t nb23,
    ಸ್ಥಿರಾಂಕ int32_t ne31,
    ಸ್ಥಿರಾಂಕ int32_t ne32,
    ಸ್ಥಿರಾಂಕ int32_t ne33,
    ಸ್ಥಿರಾಂಕ int32_t nb31,
    ಸ್ಥಿರಾಂಕ int32_t nb32,
    ಸ್ಥಿರಾಂಕ int64_t nb33);

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ದಶಮಾಂಶ (*vec_dot_KQ_t)(
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * __restrict__ K_c, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ Q_v, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ * __restrict__ Q_q8 , ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ Q_ds);

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ D, ಪೂರ್ಣಾಂಕ nthreads>
ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ vec_dot_fattn_vec_KQ_f16(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * __restrict__ K_c,
                                                      ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ Q_v,
                                                      ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ * __restrict__ Q_q8,
                                                      ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ Q_ds_v) {
    ಸ್ಥಿರಾಂಕ sycl::half2 * K_h2 = (ಸ್ಥಿರಾಂಕ sycl::half2 *) K_c;
    GGML_ಬಳಸದ(Q_q8);
    GGML_ಬಳಸದ(Q_ds_v);

    constexpr ಪೂರ್ಣಾಂಕ cpy_nb = ggml_sycl_get_max_cpy_ಬೈಟುಗಳು();
    constexpr ಪೂರ್ಣಾಂಕ cpy_ne = cpy_nb / 4;

    ದಶಮಾಂಶ ಮೊತ್ತ = 0.0f;

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads*cpy_ne) {
        sycl::half2 tmp[cpy_ne];
        ggml_sycl_memcpy_1<ಗಾತ್ರ(tmp)>(
            tmp,
            K_h2 + k_KQ_0 + (sycl::ext::oneapi::this_work_ಅಂಶ::get_nd_ಅಂಶ<3>().get_local_id(2) % nthreads) * cpy_ne);
#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_F16
            ggml_sycl_mad(ಮೊತ್ತ,                tmp[k_KQ_1] , ((ಸ್ಥಿರಾಂಕ sycl::half2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#ಇಲ್ಲದಿದ್ದರೆ
            ggml_sycl_mad(ಮೊತ್ತ, __half22float2(tmp[k_KQ_1]), ((ಸ್ಥಿರಾಂಕ sycl::float2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#ಆದರೆ_ಅಂತ್ಯ // GGML_SYCL_F16
        }
    }

    ಹಿಂತಿರುಗಿ ಮೊತ್ತ;
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ D, ಪೂರ್ಣಾಂಕ nthreads, ಪೂರ್ಣಾಂಕ warp_ಗಾತ್ರ>
ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ vec_dot_fattn_vec_KQ_q4_0(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * __restrict__ K_c,
                                                       ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ Q_v,
                                                       ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ * __restrict__ Q_q8,
                                                       ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ Q_ds_v) {
    ಸ್ವಯಂಚಾಲಿತ ಅಂಶ_ct1 = sycl::ext::oneapi::this_work_ಅಂಶ::get_nd_ಅಂಶ<3>();

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0 * K_q4_0   = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0 *) K_c;
    GGML_ಬಳಸದ(Q_v);

    ದಶಮಾಂಶ ಮೊತ್ತ = 0.0f;

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k_KQ_0 = 0; k_KQ_0 < ಪೂರ್ಣಾಂಕ(D/ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ)); k_KQ_0 += nthreads) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ k_KQ =
            k_KQ_0 + (nthreads == warp_ಗಾತ್ರ ? ಅಂಶ_ct1.get_local_id(2) : ಅಂಶ_ct1.get_local_id(2) % nthreads);

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ib    = k_KQ /  QI8_1;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs4  = k_KQ %  QI4_0;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಸರಿಸು = k_KQ & (QI8_1/2);

        ಪೂರ್ಣಾಂಕ v;
        ggml_sycl_memcpy_1<ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ), 2>(&v, K_q4_0[ib].qs + ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ)*iqs4);
        v = (v >> ಸರಿಸು) & 0x0F0F0F0F;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ u = Q_q8[k_KQ_0/nthreads];

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ sumi = ggml_sycl_dp4a(v, u, 0);

        ಸ್ಥಿರಾಂಕ sycl::float2 Q_ds = ((ಸ್ಥಿರಾಂಕ sycl::float2 *) Q_ds_v)[k_KQ_0 / nthreads];
        ಮೊತ್ತ += __half2float(K_q4_0[ib].d) * (sumi*Q_ds.x() - (8/QI8_1)*Q_ds.y());
    }

    ಹಿಂತಿರುಗಿ ಮೊತ್ತ;
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ D, ಪೂರ್ಣಾಂಕ nthreads , ಪೂರ್ಣಾಂಕ warp_ಗಾತ್ರ>
ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ vec_dot_fattn_vec_KQ_q4_1(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * __restrict__ K_c,
                                                       ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ Q_v,
                                                       ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ * __restrict__ Q_q8,
                                                       ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ Q_ds_v) {
    ಸ್ವಯಂಚಾಲಿತ               ಅಂಶ_ct1 = sycl::ext::oneapi::this_work_ಅಂಶ::get_nd_ಅಂಶ<3>();
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_1 * K_q4_1   = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_1 *) K_c;
    GGML_ಬಳಸದ(Q_v);

    ದಶಮಾಂಶ ಮೊತ್ತ = 0.0f;

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k_KQ_0 = 0; k_KQ_0 < ಪೂರ್ಣಾಂಕ(D/ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ)); k_KQ_0 += nthreads) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ k_KQ =
            k_KQ_0 + (nthreads == warp_ಗಾತ್ರ ? ಅಂಶ_ct1.get_local_id(2) : ಅಂಶ_ct1.get_local_id(2) % nthreads);

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ib    = k_KQ /  QI8_1;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs4  = k_KQ %  QI4_1;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಸರಿಸು = k_KQ & (QI8_1/2);

        ಪೂರ್ಣಾಂಕ v;
        ggml_sycl_memcpy_1<ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ)>(&v, K_q4_1[ib].qs + ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ)*iqs4);
        v = (v >> ಸರಿಸು) & 0x0F0F0F0F;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ u = Q_q8[k_KQ_0/nthreads];

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ sumi = ggml_sycl_dp4a(v, u, 0);

        ಸ್ಥಿರಾಂಕ sycl::float2 K_dm = (K_q4_1[ib].dm).ಮಾದರಿ convert<ದಶಮಾಂಶ, sycl::rounding_ಬಹುಲಕ::automatic>();
        ಸ್ಥಿರಾಂಕ sycl::float2 Q_ds = ((ಸ್ಥಿರಾಂಕ sycl::float2 *) Q_ds_v)[k_KQ_0 / nthreads];

        ಮೊತ್ತ += K_dm.x()*Q_ds.x()*sumi + K_dm.y()*Q_ds.y()/QI8_1;
    }

    ಹಿಂತಿರುಗಿ ಮೊತ್ತ;
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ D, ಪೂರ್ಣಾಂಕ nthreads, ಪೂರ್ಣಾಂಕ warp_ಗಾತ್ರ>
ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ vec_dot_fattn_vec_KQ_q5_0(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * __restrict__ K_c,
                                                       ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ Q_v,
                                                       ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ * __restrict__ Q_q8,
                                                       ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ Q_ds_v) {
    ಸ್ವಯಂಚಾಲಿತ               ಅಂಶ_ct1 = sycl::ext::oneapi::this_work_ಅಂಶ::get_nd_ಅಂಶ<3>();
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_0 * K_q5_0   = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_0 *) K_c;
    GGML_ಬಳಸದ(Q_v);

    ದಶಮಾಂಶ ಮೊತ್ತ = 0.0f;

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k_KQ_0 = 0; k_KQ_0 < ಪೂರ್ಣಾಂಕ(D/ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ)); k_KQ_0 += nthreads) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ k_KQ =
            k_KQ_0 + (nthreads == warp_ಗಾತ್ರ ? ಅಂಶ_ct1.get_local_id(2) : ಅಂಶ_ct1.get_local_id(2) % nthreads);

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ib    = k_KQ /  QI8_1;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs4  = k_KQ %  QI5_0;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs8  = k_KQ %  QI8_1;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಸರಿಸು = k_KQ & (QI8_1/2);

        ಪೂರ್ಣಾಂಕ v;
        ggml_sycl_memcpy_1<ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ), 2>(&v, K_q5_0[ib].qs + ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ)*iqs4);
        v = (v >> ಸರಿಸು) & 0x0F0F0F0F;

        {
            ಪೂರ್ಣಾಂಕ vh;
            ggml_sycl_memcpy_1<ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ), 2>(&vh, K_q5_0[ib].qh);
            vh >>= iqs8 * QI5_0;

            v |= (vh <<  4) & 0x00000010; // 0 ->  4
            v |= (vh << 11) & 0x00001000; // 1 -> 12
            v |= (vh << 18) & 0x00100000; // 2 -> 20
            v |= (vh << 25) & 0x10000000; // 3 -> 28
        }

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ u = Q_q8[k_KQ_0/nthreads];

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ sumi = ggml_sycl_dp4a(v, u, 0);

        ಸ್ಥಿರಾಂಕ sycl::float2 Q_ds = ((ಸ್ಥಿರಾಂಕ sycl::float2 *) Q_ds_v)[k_KQ_0 / nthreads];

        ಮೊತ್ತ += __half2float(K_q5_0[ib].d) * (sumi*Q_ds.x() - (16/QI8_1)*Q_ds.y());
    }

    ಹಿಂತಿರುಗಿ ಮೊತ್ತ;
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ D, ಪೂರ್ಣಾಂಕ nthreads, ಪೂರ್ಣಾಂಕ warp_ಗಾತ್ರ>
ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ vec_dot_fattn_vec_KQ_q5_1(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * __restrict__ K_c,
                                                       ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ Q_v,
                                                       ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ * __restrict__ Q_q8,
                                                       ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ Q_ds_v) {
    ಸ್ವಯಂಚಾಲಿತ               ಅಂಶ_ct1 = sycl::ext::oneapi::this_work_ಅಂಶ::get_nd_ಅಂಶ<3>();
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_1 * K_q5_1   = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_1 *) K_c;
    GGML_ಬಳಸದ(Q_v);

    ದಶಮಾಂಶ ಮೊತ್ತ = 0.0f;

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k_KQ_0 = 0; k_KQ_0 < ಪೂರ್ಣಾಂಕ(D/ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ)); k_KQ_0 += nthreads) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ k_KQ =
            k_KQ_0 + (nthreads == warp_ಗಾತ್ರ ? ಅಂಶ_ct1.get_local_id(2) : ಅಂಶ_ct1.get_local_id(2) % nthreads);

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ib    = k_KQ /  QI8_1;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs4  = k_KQ %  QI5_1;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs8  = k_KQ %  QI8_1;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಸರಿಸು = k_KQ & (QI8_1/2);

        ಪೂರ್ಣಾಂಕ v;
        ggml_sycl_memcpy_1<ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ)>(&v, K_q5_1[ib].qs + ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ)*iqs4);
        v = (v >> ಸರಿಸು) & 0x0F0F0F0F;

        {
            ಪೂರ್ಣಾಂಕ vh;
            ggml_sycl_memcpy_1<ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ)>(&vh, K_q5_1[ib].qh);
            vh >>= iqs8 * QI5_0;

            v |= (vh <<  4) & 0x00000010; // 0 ->  4
            v |= (vh << 11) & 0x00001000; // 1 -> 12
            v |= (vh << 18) & 0x00100000; // 2 -> 20
            v |= (vh << 25) & 0x10000000; // 3 -> 28
        }

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ u = Q_q8[k_KQ_0/nthreads];

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ sumi = ggml_sycl_dp4a(v, u, 0);

        ಸ್ಥಿರಾಂಕ sycl::float2 K_dm = (K_q5_1[ib].dm).ಮಾದರಿ convert<ದಶಮಾಂಶ, sycl::rounding_ಬಹುಲಕ::automatic>();
        ಸ್ಥಿರಾಂಕ sycl::float2 Q_ds = ((ಸ್ಥಿರಾಂಕ sycl::float2 *) Q_ds_v)[k_KQ_0 / nthreads];

        ಮೊತ್ತ += K_dm.x()*Q_ds.x()*sumi + K_dm.y()*Q_ds.y()/QI8_1;
    }

    ಹಿಂತಿರುಗಿ ಮೊತ್ತ;
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ D, ಪೂರ್ಣಾಂಕ nthreads, ಪೂರ್ಣಾಂಕ warp_ಗಾತ್ರ>
ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ vec_dot_fattn_vec_KQ_q8_0(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * __restrict__ K_c,
                                                       ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ Q_v,
                                                       ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ * __restrict__ Q_q8,
                                                       ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ Q_ds_v) {
    ಸ್ವಯಂಚಾಲಿತ               ಅಂಶ_ct1 = sycl::ext::oneapi::this_work_ಅಂಶ::get_nd_ಅಂಶ<3>();
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 * K_q8_0   = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 *) K_c;
    GGML_ಬಳಸದ(Q_v);

    ದಶಮಾಂಶ ಮೊತ್ತ = 0.0f;

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k_KQ_0 = 0; k_KQ_0 < ಪೂರ್ಣಾಂಕ(D/ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ)); k_KQ_0 += nthreads) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ k_KQ =
            k_KQ_0 + (nthreads == warp_ಗಾತ್ರ ? ಅಂಶ_ct1.get_local_id(2) : ಅಂಶ_ct1.get_local_id(2) % nthreads);

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ib  = k_KQ / QI8_0;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs = k_KQ % QI8_0;

        ಪೂರ್ಣಾಂಕ v;
        ggml_sycl_memcpy_1<ಗಾತ್ರ(v), 2>(&v, K_q8_0[ib].qs + 4*iqs);

        ಸ್ಥಿರಾಂಕ sycl::float2 * Q_ds = (ಸ್ಥಿರಾಂಕ sycl::float2 *) Q_ds_v;
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ          Q_d  = Q_ds[k_KQ_0 / nthreads].x();

        ಮೊತ್ತ += vec_dot_q8_0_q8_1_impl<ದಶಮಾಂಶ, 1>(&v, &Q_q8[k_KQ_0/nthreads], K_q8_0[ib].d, Q_d);
    }

    ಹಿಂತಿರುಗಿ ಮೊತ್ತ;
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು Tds, ಪೂರ್ಣಾಂಕ ni, ಪೂರ್ಣಾಂಕ warp_ಗಾತ್ರ>
ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ quantize_q8_1_ಗಮ್ಯ_shared(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * __restrict__ x,
                                                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale,
                                                    ಪೂರ್ಣಾಂಕ * __restrict__ yq32,
                                                    ಶೂನ್ಯ * __restrict__ yds) {
    ಸ್ವಯಂಚಾಲಿತ ಅಂಶ_ct1 = sycl::ext::oneapi::this_work_ಅಂಶ::get_nd_ಅಂಶ<3>();

    ದಶಮಾಂಶ vals[ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ)] = { 0.0f };
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < ಪೂರ್ಣಾಂಕ(ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ)); ++l) {
        vals[l] =
            (ni == warp_ಗಾತ್ರ || ಅಂಶ_ct1.get_local_id(2) < ni) ? scale * x[4 * ಅಂಶ_ct1.get_local_id(2) + l] : 0.0f;
    }

    ದಶಮಾಂಶ amax = sycl::fabs(vals[0]);
    ದಶಮಾಂಶ ಮೊತ್ತ  = vals[0];
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 1; l < ಪೂರ್ಣಾಂಕ(ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ)); ++l) {
        amax = sycl::fmax(amax, sycl::fabs(vals[l]));
        ಮೊತ್ತ += vals[l];
    }
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಮುಖವಾಡ = QI8_1/2; ಮುಖವಾಡ > 0; ಮುಖವಾಡ >>= 1) {
        amax = sycl::fmax(
            amax, dpct::ಕ್ರಮಬದಲಾವಣೆ_sub_ಗುಂಪು_by_xor(sycl::ext::oneapi::this_work_ಅಂಶ::get_sub_ಗುಂಪು(), amax, ಮುಖವಾಡ));
        ಮೊತ್ತ += dpct::ಕ್ರಮಬದಲಾವಣೆ_sub_ಗುಂಪು_by_xor(sycl::ext::oneapi::this_work_ಅಂಶ::get_sub_ಗುಂಪು(), ಮೊತ್ತ, ಮುಖವಾಡ);
    }

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = amax / 127;
    ಪೂರ್ಣಾಂಕ q32 = 0;
    int8_t * q8 = (int8_t *) &q32;

    ಆದರೆ (d != 0.0f) {
#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < ಪೂರ್ಣಾಂಕ(ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ)); ++l) {
            q8[l] = sycl::round(vals[l] / d);
        }
    }

    yq32[ಅಂಶ_ct1.get_local_id(2)] = q32;
    ಆದರೆ (ಅಂಶ_ct1.get_local_id(2) % QI8_1 == 0 && (ni == warp_ಗಾತ್ರ || ಅಂಶ_ct1.get_local_id(2) < ni)) {
        ಆದರೆ (std::is_same<Tds, sycl::half2>::value) {
            ((sycl::half2  *) yds)[ಅಂಶ_ct1.get_local_id(2)/QI8_1] =  make_half2(d, ಮೊತ್ತ);
        } ಇಲ್ಲದಿದ್ದರೆ {
            ((sycl::float2 *) yds)[ಅಂಶ_ct1.get_local_id(2)/QI8_1] = make_float2(d, ಮೊತ್ತ);
        }
    }
}

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಶೂನ್ಯ (*dequantize_V_t)(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *, ಶೂನ್ಯ *, ಸ್ಥಿರಾಂಕ int64_t);

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T, ಪೂರ್ಣಾಂಕ ne>
ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_V_f16(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vx, ಶೂನ್ಯ * __restrict__ dst, ಸ್ಥಿರಾಂಕ int64_t i0) {
    ಆದರೆ constexpr (std::is_same_v<T, sycl::half>) {
        ggml_sycl_memcpy_1<ne * ಗಾತ್ರ(sycl::half)>(dst, (ಸ್ಥಿರಾಂಕ sycl::half *) vx + i0);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (std::is_same_v<T, ದಶಮಾಂಶ>) {
        static_assert(ne % 2 == 0, "bad ne");
        sycl::half2 tmp[ne / 2];
        ggml_sycl_memcpy_1<ne * ಗಾತ್ರ(sycl::half)>(tmp, (ಸ್ಥಿರಾಂಕ sycl::half *) vx + i0);
        sycl::float2 * dst_f2 = (sycl::float2 *) dst;
#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < ne/2; ++l) {
            dst_f2[l] = tmp[l].ಮಾದರಿ convert<ದಶಮಾಂಶ, sycl::rounding_ಬಹುಲಕ::automatic>();
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        static_assert(std::is_same_v<T, ಶೂನ್ಯ>, "unsupported type");
    }
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T, ಪೂರ್ಣಾಂಕ ne>
ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_V_q4_0(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vx, ಶೂನ್ಯ * __restrict__ dst, ಸ್ಥಿರಾಂಕ int64_t i0) {
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0 * x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0 *) vx;

    ಸ್ಥಿರಾಂಕ int64_t ib    =  i0          /  QK4_0;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ     iqs   =  i0          % (QK4_0/2);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ     ಸರಿಸು = (i0 % QK4_0) / (QK4_0/2);

    ಪೂರ್ಣಾಂಕ q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_sycl_memcpy_1<ne, 2>(&q, x[ib].qs + iqs);
    q >>= 4*ಸರಿಸು;
    q &= 0x0F0F0F0F;
    q = dpct::vectorized_ದ್ವಿಮಾನ<sycl::char4>(q, 0x08080808, dpct::sub_sat());

    ಸ್ಥಿರಾಂಕ int8_t * q8 = (ಸ್ಥಿರಾಂಕ int8_t *) &q;

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_F16
    ಆದರೆ constexpr (std::is_same_v<T, sycl::half>) {
        ಸ್ಥಿರಾಂಕ sycl::half2 d = sycl::half2(x[ib].d);

#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * sycl::half2(q8[l0 + 0], q8[l0 + 1]);
        }
    } ಇಲ್ಲದಿದ್ದರೆ
#ಆದರೆ_ಅಂತ್ಯ // GGML_SYCL_F16
    ಆದರೆ constexpr (std::is_same_v<T, ದಶಮಾಂಶ>) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = x[ib].d;

#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < ne; ++l) {
            ((ದಶಮಾಂಶ *) dst)[l] = d * q8[l];
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        static_assert(std::is_same_v<T, ಶೂನ್ಯ>, "bad type");
    }
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T, ಪೂರ್ಣಾಂಕ ne>
ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_V_q4_1(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vx, ಶೂನ್ಯ * __restrict__ dst, ಸ್ಥಿರಾಂಕ int64_t i0) {
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_1 * x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_1 *) vx;

    ಸ್ಥಿರಾಂಕ int64_t ib    =  i0          /  QK4_1;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ     iqs   =  i0          % (QK4_1/2);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ     ಸರಿಸು = (i0 % QK4_1) / (QK4_1/2);

    ಪೂರ್ಣಾಂಕ q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_sycl_memcpy_1<ne>(&q, x[ib].qs + iqs);
    q >>= 4*ಸರಿಸು;
    q &= 0x0F0F0F0F;

    ಸ್ಥಿರಾಂಕ int8_t * q8 = (ಸ್ಥಿರಾಂಕ int8_t *) &q;

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_F16
    ಆದರೆ constexpr (std::is_same_v<T, sycl::half>) {
        ಸ್ಥಿರಾಂಕ sycl::half2 dm = x[ib].dm;
        ಸ್ಥಿರಾಂಕ sycl::half2 d  = sycl::half2(dm[0]);
        ಸ್ಥಿರಾಂಕ sycl::half2 m  = sycl::half2(dm[1]);

#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * sycl::half2(q8[l0 + 0], q8[l0 + 1]) + m;
        }
    } ಇಲ್ಲದಿದ್ದರೆ
#ಆದರೆ_ಅಂತ್ಯ // GGML_SYCL_F16
    ಆದರೆ constexpr (std::is_same_v<T, ದಶಮಾಂಶ>) {
        ಸ್ಥಿರಾಂಕ sycl::float2 dm = (x[ib].dm).ಮಾದರಿ convert<ದಶಮಾಂಶ, sycl::rounding_ಬಹುಲಕ::automatic>();

#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < ne; ++l) {
            ((ದಶಮಾಂಶ *) dst)[l] = dm.x() * q8[l] + dm.y();
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        static_assert(std::is_same_v<T, ಶೂನ್ಯ>, "bad type");
    }
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T, ಪೂರ್ಣಾಂಕ ne>
ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_V_q5_0(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vx, ಶೂನ್ಯ * __restrict__ dst, ಸ್ಥಿರಾಂಕ int64_t i0) {
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_0 * x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_0 *) vx;

    ಸ್ಥಿರಾಂಕ int64_t ib    =  i0          /  QK5_0;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ     idq   =  i0          %  QK5_0;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ     iqs   =  i0          % (QK5_0/2);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ     ಸರಿಸು = (i0 % QK5_0) / (QK5_0/2);

    ಪೂರ್ಣಾಂಕ q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_sycl_memcpy_1<ne, 2>(&q, x[ib].qs + iqs);
    q >>= 4*ಸರಿಸು;
    q &= 0x0F0F0F0F;

    {
        ಪೂರ್ಣಾಂಕ qh;
        ggml_sycl_memcpy_1<ne, 2>(&qh, x[ib].qh);
#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < ne; ++l) {
            q |= ((qh >> (idq + l)) & 0x00000001) << (8*l + 4);
        }
    }

    q = dpct::vectorized_ದ್ವಿಮಾನ<sycl::char4>(q, 0x10101010, dpct::sub_sat());

    ಸ್ಥಿರಾಂಕ int8_t * q8 = (ಸ್ಥಿರಾಂಕ int8_t *) &q;

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_F16
    ಆದರೆ constexpr (std::is_same_v<T, sycl::half>) {
        ಸ್ಥಿರಾಂಕ sycl::half2 d = sycl::half2(x[ib].d);

#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * sycl::half2(q8[l0 + 0], q8[l0 + 1]);
        }
    } ಇಲ್ಲದಿದ್ದರೆ
#ಆದರೆ_ಅಂತ್ಯ // GGML_SYCL_F16
    ಆದರೆ constexpr (std::is_same_v<T, ದಶಮಾಂಶ>) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = x[ib].d;

#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < ne; ++l) {
            ((ದಶಮಾಂಶ *) dst)[l] = d * q8[l];
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        static_assert(std::is_same_v<T, ಶೂನ್ಯ>, "bad type");
    }
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T, ಪೂರ್ಣಾಂಕ ne>
ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_V_q5_1(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vx, ಶೂನ್ಯ * __restrict__ dst, ಸ್ಥಿರಾಂಕ int64_t i0) {
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_1 * x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_1 *) vx;

    ಸ್ಥಿರಾಂಕ int64_t ib    =  i0          /  QK5_1;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ     idq   =  i0          %  QK5_1;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ     iqs   =  i0          % (QK5_1/2);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ     ಸರಿಸು = (i0 % QK5_1) / (QK5_1/2);

    ಪೂರ್ಣಾಂಕ q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_sycl_memcpy_1<ne>(&q, x[ib].qs + iqs);
    q >>= 4*ಸರಿಸು;
    q &= 0x0F0F0F0F;

    {
        ಪೂರ್ಣಾಂಕ qh;
        ggml_sycl_memcpy_1<ne>(&qh, x[ib].qh);
#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < ne; ++l) {
            q |= ((qh >> (idq + l)) & 0x00000001) << (8*l + 4);
        }
    }

    ಸ್ಥಿರಾಂಕ int8_t * q8 = (ಸ್ಥಿರಾಂಕ int8_t *) &q;

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_F16
    ಆದರೆ constexpr (std::is_same_v<T, sycl::half>) {
        ಸ್ಥಿರಾಂಕ sycl::half2 dm = x[ib].dm;
        ಸ್ಥಿರಾಂಕ sycl::half2 d  = sycl::half2(dm[0]);
        ಸ್ಥಿರಾಂಕ sycl::half2 m  = sycl::half2(dm[1]);

#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * sycl::half2(q8[l0 + 0], q8[l0 + 1]) + m;
        }
    } ಇಲ್ಲದಿದ್ದರೆ
#ಆದರೆ_ಅಂತ್ಯ // GGML_SYCL_F16
    ಆದರೆ constexpr (std::is_same_v<T, ದಶಮಾಂಶ>) {
        ಸ್ಥಿರಾಂಕ sycl::float2 dm = (x[ib].dm).ಮಾದರಿ convert<ದಶಮಾಂಶ, sycl::rounding_ಬಹುಲಕ::automatic>();

#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < ne; ++l) {
            ((ದಶಮಾಂಶ *) dst)[l] = dm.x() * q8[l] + dm.y();
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        static_assert(std::is_same_v<T, ಶೂನ್ಯ>, "bad type");
    }
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T, ಪೂರ್ಣಾಂಕ ne>
ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_V_q8_0(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vx, ಶೂನ್ಯ * __restrict__ dst, ಸ್ಥಿರಾಂಕ int64_t i0) {
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 * x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 *) vx;

    ಸ್ಥಿರಾಂಕ int64_t ib  = i0 / QK8_0;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ     iqs = i0 % QK8_0;

    static_assert(ne % 2 == 0, "bad ne");
    int8_t qs[ne];
    ggml_sycl_memcpy_1<ne, 2>(qs, x[ib].qs + iqs);

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_F16
    ಆದರೆ constexpr (std::is_same<T, sycl::half>::value) {
        ಸ್ಥಿರಾಂಕ sycl::half2 d = sycl::half2(x[ib].d);

#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * make_half2(qs[l0 + 0], qs[l0 + 1]);
        }
    } ಇಲ್ಲದಿದ್ದರೆ
#ಆದರೆ_ಅಂತ್ಯ // GGML_SYCL_F16
    ಆದರೆ constexpr (std::is_same<T, ದಶಮಾಂಶ>::value) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = x[ib].d;

#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < ne; ++l) {
            ((ದಶಮಾಂಶ *) dst)[l] = d * qs[l];
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        static_assert(std::is_same_v<T, ಶೂನ್ಯ>, "unsupported type");
    }
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ type_K, ಪೂರ್ಣಾಂಕ D, ಪೂರ್ಣಾಂಕ nthreads, ಪೂರ್ಣಾಂಕ warp_ಗಾತ್ರ>
constexpr vec_dot_KQ_t get_vec_dot_KQ() {
    ಆದರೆ constexpr (type_K == GGML_TYPE_F16) {
        ಹಿಂತಿರುಗಿ vec_dot_fattn_vec_KQ_f16<D, nthreads>;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (type_K == GGML_TYPE_Q4_0) {
        ಹಿಂತಿರುಗಿ vec_dot_fattn_vec_KQ_q4_0<D, nthreads, warp_ಗಾತ್ರ>;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (type_K == GGML_TYPE_Q4_1) {
        ಹಿಂತಿರುಗಿ vec_dot_fattn_vec_KQ_q4_1<D, nthreads, warp_ಗಾತ್ರ>;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (type_K == GGML_TYPE_Q5_0) {
        ಹಿಂತಿರುಗಿ vec_dot_fattn_vec_KQ_q5_0<D, nthreads, warp_ಗಾತ್ರ>;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (type_K == GGML_TYPE_Q5_1) {
        ಹಿಂತಿರುಗಿ vec_dot_fattn_vec_KQ_q5_1<D, nthreads, warp_ಗಾತ್ರ>;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (type_K == GGML_TYPE_Q8_0) {
        ಹಿಂತಿರುಗಿ vec_dot_fattn_vec_KQ_q8_0<D, nthreads, warp_ಗಾತ್ರ>;
    } ಇಲ್ಲದಿದ್ದರೆ {
        static_assert(type_K == -1, "bad type");
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ type_V, ಬಗೆ_ಹೆಸರು T, ಪೂರ್ಣಾಂಕ ne>
constexpr dequantize_V_t get_dequantize_V() {
    ಆದರೆ constexpr (type_V == GGML_TYPE_F16) {
        ಹಿಂತಿರುಗಿ dequantize_V_f16<T, ne>;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (type_V == GGML_TYPE_Q4_0) {
        ಹಿಂತಿರುಗಿ dequantize_V_q4_0<T, ne>;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (type_V == GGML_TYPE_Q4_1) {
        ಹಿಂತಿರುಗಿ dequantize_V_q4_1<T, ne>;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (type_V == GGML_TYPE_Q5_0) {
        ಹಿಂತಿರುಗಿ dequantize_V_q5_0<T, ne>;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (type_V == GGML_TYPE_Q5_1) {
        ಹಿಂತಿರುಗಿ dequantize_V_q5_1<T, ne>;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (type_V == GGML_TYPE_Q8_0) {
        ಹಿಂತಿರುಗಿ dequantize_V_q8_0<T, ne>;
    } ಇಲ್ಲದಿದ್ದರೆ {
        static_assert(type_V == -1, "bad type");
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ ncols1, ಪೂರ್ಣಾಂಕ warp_ಗಾತ್ರ>
ಸ್ಥಿರ ಶೂನ್ಯ flash_attn_ಮುಖವಾಡ_ಗಮ್ಯ_KV_max(ಸ್ಥಿರಾಂಕ sycl::half2 * __restrict__ ಮುಖವಾಡ,
                                      ಪೂರ್ಣಾಂಕ * __restrict__ KV_max,
                                      ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne30,
                                      ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s31,
                                      ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s33,
                                      ಪೂರ್ಣಾಂಕ *     buf_iw) {
    ಸ್ವಯಂಚಾಲಿತ      ಅಂಶ_ct1 = sycl::ext::oneapi::this_work_ಅಂಶ::get_nd_ಅಂಶ<3>();
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne31     = ಅಂಶ_ct1.get_ಗುಂಪು_range(2);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ tid      = ಅಂಶ_ct1.get_local_id(2);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ sequence = ಅಂಶ_ct1.get_ಗುಂಪು(1);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ jt       = ಅಂಶ_ct1.get_ಗುಂಪು(2);

    ಮುಖವಾಡ += sequence*s33 + jt*ncols1*s31;

    ಆದರೆ (tid < warp_ಗಾತ್ರ) {
        buf_iw[tid] = 1;
    }
    ಅಂಶ_ct1.barrier(sycl::access::fence_space::local_space);

    ಪೂರ್ಣಾಂಕ KV_max_sj = (ne30 - 1) * FATTN_KQ_STRIDE;
    ಗಾಗಿ (; KV_max_sj >= 0; KV_max_sj -= FATTN_KQ_STRIDE) {
        ಪೂರ್ಣಾಂಕ all_inf = 1;

#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < ncols1; ++j) {
            ಸ್ಥಿರಾಂಕ sycl::float2 tmp =
                ಮುಖವಾಡ[j * s31 + KV_max_sj / 2 + tid].ಮಾದರಿ convert<ದಶಮಾಂಶ, sycl::rounding_ಬಹುಲಕ::automatic>();
            all_inf = all_inf && ಪೂರ್ಣಾಂಕ(sycl::isinf((ದಶಮಾಂಶ) (tmp.x()))) && ಪೂರ್ಣಾಂಕ(sycl::isinf((ದಶಮಾಂಶ) (tmp.y())));
        }

        all_inf = warp_reduce_all<warp_ಗಾತ್ರ>(all_inf);
        ಆದರೆ (tid % warp_ಗಾತ್ರ == 0) {
            buf_iw[tid / warp_ಗಾತ್ರ] = all_inf;
        }
        ಅಂಶ_ct1.barrier(sycl::access::fence_space::local_space);
        all_inf = buf_iw[tid % warp_ಗಾತ್ರ];
        ಅಂಶ_ct1.barrier(sycl::access::fence_space::local_space);
        all_inf = warp_reduce_all<warp_ಗಾತ್ರ>(all_inf);

        ಆದರೆ (!all_inf) {
            ನಿಲ್ಲಿಸು;
        }
    }

    // If the break in the loop was not triggered, KV_max_sj is now -FATTN_KQ_STRIDE.
    // If the break was triggered it's the lower edge of the tile with the first non-masked values.
    // In either case, walk back the decrementation by FATTN_KQ_STRIDE.
    KV_max_sj += FATTN_KQ_STRIDE;

    ಆದರೆ (ಅಂಶ_ct1.get_local_id(2) != 0) {
        ಹಿಂತಿರುಗಿ;
    }

    KV_max[sequence*ne31 + jt] = KV_max_sj;
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ D, ಪೂರ್ಣಾಂಕ ncols1, ಪೂರ್ಣಾಂಕ ncols2>  // D == head size

ಸ್ಥಿರ ಶೂನ್ಯ flash_attn_ಹರಿವು_k_fixup(ದಶಮಾಂಶ * __restrict__ dst,
                                      ಸ್ಥಿರಾಂಕ sycl::float2 * __restrict__ dst_fixup,
                                      ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne01,
                                      ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne02,
                                      ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne03,
                                      ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne11,
                                      ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne12,
                                      ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nbatch_fa) {
    ಸ್ವಯಂಚಾಲಿತ          ಅಂಶ_ct1 = sycl::ext::oneapi::this_work_ಅಂಶ::get_nd_ಅಂಶ<3>();
    constexpr ಪೂರ್ಣಾಂಕ ncols    = ncols1 * ncols2;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ bidx0 = ಅಂಶ_ct1.get_ಗುಂಪು(2);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ j     = ಅಂಶ_ct1.get_ಗುಂಪು(1);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ c     = ಅಂಶ_ct1.get_ಗುಂಪು(0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ jc    = j*ncols2 + c;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ tid   = ಅಂಶ_ct1.get_local_id(2);

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * dst_fixup_ದತ್ತಾಂಶ = ((ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) dst_fixup) + ಅಂಶ_ct1.get_ಗುಂಪು_range(2) * (2 * 2 * ncols);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ gqa_ಅನುಪಾತ = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iter_k     = (ne11      + (nbatch_fa - 1)) / nbatch_fa;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iter_j     = (ne01      + (ncols1    - 1)) / ncols1;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iter_z_gqa = (gqa_ಅನುಪಾತ + (ncols2    - 1)) / ncols2;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kbc0 = int64_t(bidx0 + 0) * (iter_k * iter_j * iter_z_gqa * ne12 * ne03) / ಅಂಶ_ct1.get_ಗುಂಪು_range(2);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kbc0_stop =
        int64_t(bidx0 + 1) * (iter_k * iter_j * iter_z_gqa * ne12 * ne03) / ಅಂಶ_ct1.get_ಗುಂಪು_range(2);

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ did_not_have_any_ದತ್ತಾಂಶ   = kbc0 == kbc0_stop;
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ wrote_beginning_ದ_tile = kbc0 % iter_k == 0;
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ did_not_write_ಕೊನೆಯ      = kbc0/iter_k == kbc0_stop/iter_k && kbc0_stop % iter_k != 0;
    ಆದರೆ (did_not_have_any_ದತ್ತಾಂಶ || wrote_beginning_ದ_tile || did_not_write_ಕೊನೆಯ) {
        ಹಿಂತಿರುಗಿ;
    }

    // z_KV == K/V head index, zt_gqa = Q head start index per K/V head, jt = token position start index
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ sequence =  kbc0 /(iter_k*iter_j*iter_z_gqa*ne12);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ z_KV     = (kbc0 - iter_k*iter_j*iter_z_gqa*ne12 * sequence)/(iter_k*iter_j*iter_z_gqa);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ zt_gqa   = (kbc0 - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV)/(iter_k*iter_j);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ jt       = (kbc0 - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV - iter_k*iter_j * zt_gqa) / iter_k;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ zt_Q = z_KV*gqa_ಅನುಪಾತ + zt_gqa*ncols2; // Global Q head start index.

    ಆದರೆ (jt*ncols1 + j >= ne01 || zt_gqa*ncols2 + c >= gqa_ಅನುಪಾತ) {
        ಹಿಂತಿರುಗಿ;
    }

    dst += sequence*ne02*ne01*D + jt*ne02*(ncols1*D) + zt_Q*D + (j*ne02 + c)*D + tid;

    // Load the partial result that needs a fixup:
    ದಶಮಾಂಶ dst_val = 0.0f;
    ದಶಮಾಂಶ max_val = 0.0f;
    ದಶಮಾಂಶ rowsum  = 0.0f;
    {
        dst_val = *dst;

        ಸ್ಥಿರಾಂಕ sycl::float2 tmp = dst_fixup[bidx0 * ncols + jc];
        max_val                = tmp.x();
        rowsum                 = tmp.y();
    }

    // Iterate over previous blocks and compute the combined results.
    // All SYCL blocks that get here must have a previous block that needs a fixup.
    ಪೂರ್ಣಾಂಕ bidx = bidx0 - 1;
    ಪೂರ್ಣಾಂಕ kbc_stop = kbc0;
    ಆಗ(ನಿಜ) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kbc = int64_t(bidx) * (iter_k * iter_j * iter_z_gqa * ne12 * ne03) / ಅಂಶ_ct1.get_ಗುಂಪು_range(2);
        ಆದರೆ (kbc == kbc_stop) { // Did not have any data.
            bidx--;
            kbc_stop = kbc;
            ಮುಂದುವರಿ;
        }

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dst_ಕೂಡಿಸು = dst_fixup_ದತ್ತಾಂಶ[bidx*ncols*D + jc*D + tid];

        ಸ್ಥಿರಾಂಕ sycl::float2 tmp = dst_fixup[(ಅಂಶ_ct1.get_ಗುಂಪು_range(2) + bidx) * ncols + jc];

        // Scale the current and new value accumulators depending on the max. values.
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ max_val_new = sycl::fmax(max_val, tmp.x());

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ diff_val = max_val - max_val_new;
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ diff_ಕೂಡಿಸು = tmp.x() - max_val_new;

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale_val = diff_val >= SOFTMAX_FTZ_ಮಿತಿ ? sycl::native::ಘಾತಾಂಕ(diff_val) : 0.0f;
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale_ಕೂಡಿಸು = diff_ಕೂಡಿಸು >= SOFTMAX_FTZ_ಮಿತಿ ? sycl::native::ಘಾತಾಂಕ(diff_ಕೂಡಿಸು) : 0.0f;

        dst_val = scale_val*dst_val + scale_ಕೂಡಿಸು*dst_ಕೂಡಿಸು;
        rowsum  = scale_val * rowsum + scale_ಕೂಡಿಸು * tmp.y();

        max_val = max_val_new;

        // If this block started in a previous tile we are done and don't need to combine additional partial results.
        ಆದರೆ (kbc % iter_k == 0 || kbc/iter_k < kbc0/iter_k) {
            ನಿಲ್ಲಿಸು;
        }
        bidx--;
        kbc_stop = kbc;
    }

    // Write back final result:
    *dst = dst_val / rowsum;
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ D>  // D == head size

ಸ್ಥಿರ ಶೂನ್ಯ flash_attn_combine_results(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * __restrict__ VKQ_parts,
                                       ಸ್ಥಿರಾಂಕ sycl::float2 * __restrict__ VKQ_meta,
                                       ದಶಮಾಂಶ * __restrict__ dst,
                                       ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ parallel_ಖಂಡಗಳು,
                                       uint8_t * dpct_local) {
    // Dimension 0: threadIdx.x
    // Dimension 1: blockIdx.x
    // Dimension 2: blockIdx.y
    // Dimension 3: blockIdx.z
    // Memory layout is permuted with [0, 2, 1, 3]

    ಸ್ವಯಂಚಾಲಿತ      ಅಂಶ_ct1 = sycl::ext::oneapi::this_work_ಅಂಶ::get_nd_ಅಂಶ<3>();
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne01     = ಅಂಶ_ct1.get_ಗುಂಪು_range(2);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne02     = ಅಂಶ_ct1.get_ಗುಂಪು_range(1);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು      = ಅಂಶ_ct1.get_ಗುಂಪು(2);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ತಲೆ     = ಅಂಶ_ct1.get_ಗುಂಪು(1);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ sequence = ಅಂಶ_ct1.get_ಗುಂಪು(0);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ j_dst_unrolled = (sequence*ne01 + ಲಂಬಸಾಲು)*ne02 + ತಲೆ;

    VKQ_parts += j_dst_unrolled * parallel_ಖಂಡಗಳು*D;
    VKQ_meta  += j_dst_unrolled * parallel_ಖಂಡಗಳು;
    dst       += j_dst_unrolled *                 D;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ tid = ಅಂಶ_ct1.get_local_id(2);
    __builtin_assume(tid < D);

    ಸ್ವಯಂಚಾಲಿತ meta = (sycl::float2 *) dpct_local;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = tid; i < 2*parallel_ಖಂಡಗಳು; i += D) {
        ((ದಶಮಾಂಶ *) meta)[i] = ((ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)VKQ_meta) [i];
    }

    ಅಂಶ_ct1.barrier(sycl::access::fence_space::local_space);

    ದಶಮಾಂಶ kqmax = meta[0].x();
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 1; l < parallel_ಖಂಡಗಳು; ++l) {
        kqmax = sycl::max(kqmax, meta[l].x());
    }

    ದಶಮಾಂಶ VKQ_numerator   = 0.0f;
    ದಶಮಾಂಶ VKQ_denominator = 0.0f;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < parallel_ಖಂಡಗಳು; ++l) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ KQ_max_scale = sycl::native::ಘಾತಾಂಕ(meta[l].x() - kqmax);

        VKQ_numerator   += KQ_max_scale * VKQ_parts[l*D + tid];
        VKQ_denominator += KQ_max_scale * meta[l].y();
    }

    dst[tid] = VKQ_numerator / VKQ_denominator;
}

// Carries the sub-group size (and grf_size_automatic when use_large_grf) as
// kernel properties via get(properties_tag), in place of the
// reqd_sub_group_size attribute and the deprecated parallel_for overload.
ಮಾದರಿ <ಬಗೆ_ಹೆಸರು KernelFunc, uint32_t ಗಾತ್ರ_4, ತಾರ್ಕಿಕ UseGrf>
ರಚನೆ props_kernel {
    KernelFunc kernel;

    ಶೂನ್ಯ ಕ್ರಿಯಾಕಾರಕ()(sycl::nd_ಅಂಶ<3> ಅಂಶ) ಸ್ಥಿರಾಂಕ {
        kernel(ಅಂಶ);
    }

    ಸ್ವಯಂಚಾಲಿತ get(sycl::ext::oneapi::experimental::properties_ಟ್ಯಾಗ್) ಸ್ಥಿರಾಂಕ {
        ಆದರೆ constexpr (UseGrf) {
            ಹಿಂತಿರುಗಿ sycl::ext::oneapi::experimental::properties{
                sycl::ext::intel::experimental::grf_ಗಾತ್ರ_automatic,
                sycl::ext::oneapi::experimental::sub_ಗುಂಪು_ಗಾತ್ರ<ಗಾತ್ರ_4> };
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಹಿಂತಿರುಗಿ sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::sub_ಗುಂಪು_ಗಾತ್ರ<ಗಾತ್ರ_4> };
        }
    }
};

ಮಾದರಿ <fattn_kernel_t fattn_kernel, ಪೂರ್ಣಾಂಕ warp_ಗಾತ್ರ, ತಾರ್ಕಿಕ use_large_grf = ಸುಳ್ಳು>
ಸ್ಥಿರ ಶೂನ್ಯ lauch_kernel(
    dpct::dim3 ಗುಂಪು_range,
    dpct::dim3 local_range,
    ಸರತಿ_ptr q,
    ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ local_mem_ಗಾತ್ರ_2,
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* __restrict__ Q,
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* __restrict__ K,
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* __restrict__ V,
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* __restrict__ ಮುಖವಾಡ,
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* __restrict__ sinks,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ* __restrict__ KV_max,
    ದಶಮಾಂಶ* __restrict__ dst,
    sycl::float2* __restrict__ dst_meta,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ max_bias,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ m0,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ m1,
    ಸ್ಥಿರಾಂಕ uint32_t n_ತಲೆ_log2,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ logit_softcap,
    ಸ್ಥಿರಾಂಕ int32_t ne00,
    ಸ್ಥಿರಾಂಕ sycl::uint3 ne01,
    ಸ್ಥಿರಾಂಕ int32_t ne02,
    ಸ್ಥಿರಾಂಕ int32_t ne03,
    ಸ್ಥಿರಾಂಕ int32_t nb01,
    ಸ್ಥಿರಾಂಕ int32_t nb02,
    ಸ್ಥಿರಾಂಕ int32_t nb03,
    ಸ್ಥಿರಾಂಕ int32_t ne10,
    ಸ್ಥಿರಾಂಕ int32_t ne11,
    ಸ್ಥಿರಾಂಕ int32_t ne12,
    ಸ್ಥಿರಾಂಕ int32_t ne13,
    ಸ್ಥಿರಾಂಕ int32_t nb11,
    ಸ್ಥಿರಾಂಕ int32_t nb12,
    ಸ್ಥಿರಾಂಕ int64_t nb13,
    ಸ್ಥಿರಾಂಕ int32_t nb21,
    ಸ್ಥಿರಾಂಕ int32_t nb22,
    ಸ್ಥಿರಾಂಕ int64_t nb23,
    ಸ್ಥಿರಾಂಕ int32_t ne31,
    ಸ್ಥಿರಾಂಕ int32_t ne32,
    ಸ್ಥಿರಾಂಕ int32_t ne33,
    ಸ್ಥಿರಾಂಕ int32_t nb31,
    ಸ್ಥಿರಾಂಕ int32_t nb32,
    ಸ್ಥಿರಾಂಕ int64_t nb33) {
    GGML_ಬಳಸದ(local_mem_ಗಾತ್ರ_2);

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ rng = sycl::nd_range<3>(
        static_cast<sycl::range<3>>(ಗುಂಪು_range * local_range),
        static_cast<sycl::range<3>>(local_range));

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ kernel = [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
        GGML_ಬಳಸದ(ಅಂಶ_ct1);
        fattn_kernel(Q, K, V, ಮುಖವಾಡ, sinks, KV_max, dst, dst_meta, scale,
                     max_bias, m0, m1, n_ತಲೆ_log2, logit_softcap, ne00,
                     ne01, ne02, ne03, nb01, nb02, nb03, ne10, ne11,
                     ne12, ne13, nb11, nb12, nb13, nb21, nb22, nb23,
                     ne31, ne32, ne33, nb31, nb32, nb33);
    };

    q->submit([&](sycl::handler &cgh) {
        // grf_size_automatic lets the compiler use the large register file when the kernel needs it. grf_size<256> would say the same but is undefined on devices that do not have it.
        cgh.parallel_for(rng, props_kernel<decltype(kernel), warp_ಗಾತ್ರ, use_large_grf>{kernel});
    });
}

// use_large_grf lets the compiler choose the large register file. Opt in where spilling limits the kernel, not thread occupancy.
ಮಾದರಿ <ಪೂರ್ಣಾಂಕ DV, ಪೂರ್ಣಾಂಕ ncols1, ಪೂರ್ಣಾಂಕ ncols2, fattn_kernel_t fattn_kernel, ಪೂರ್ಣಾಂಕ warp_ಗಾತ್ರ, ತಾರ್ಕಿಕ use_large_grf = ಸುಳ್ಳು>
ಶೂನ್ಯ launch_fattn(
    ggml_backend_sycl_context & ctx, ggml_tensor * dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nwarps, ಸ್ಥಿರಾಂಕ size_t nbytes_shared,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nbatch_fa, ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ need_f16_K, ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ need_f16_V, ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ ಹರಿವು_k) {

    constexpr ಪೂರ್ಣಾಂಕ ncols = ncols1 * ncols2;

    ಸ್ಥಿರಾಂಕ ggml_tensor * Q = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * K = dst->src[1];
    ಸ್ಥಿರಾಂಕ ggml_tensor * V = dst->src[2];

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ V_is_K_view = V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));

    ಸ್ಥಿರಾಂಕ ggml_tensor * ಮುಖವಾಡ  = dst->src[3];
    ಸ್ಥಿರಾಂಕ ggml_tensor * sinks = dst->src[4];

    ggml_tensor * KQV = dst;

    GGML_ASSERT(Q->type == GGML_TYPE_F32);
    GGML_ASSERT(KQV->type == GGML_TYPE_F32);

    GGML_ASSERT(Q->nb[0] == ggml_element_ಗಾತ್ರ(Q));
    GGML_ASSERT(K->nb[0] == ggml_element_ಗಾತ್ರ(K));
    GGML_ASSERT(V->nb[0] == ggml_element_ಗಾತ್ರ(V));

    GGML_ASSERT(!ಮುಖವಾಡ || ಮುಖವಾಡ->type == GGML_TYPE_F16);

    ggml_sycl_pool & pool = ctx.pool();
    dpct::ಸರತಿ_ptr  main_ಹರಿವು = ctx.stream();
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ id  = ggml_sycl_get_device();
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nsm = ggml_sycl_info().devices[id].nsm;

    ಸ್ಥಿರಾಂಕ ggml_sycl_fattn_extra  extra = ggml_sycl_fattn_get_extra(dst);
    ggml_sycl_pool_alloc<ಪೂರ್ಣಾಂಕ>    KV_max(pool);
    ggml_sycl_pool_alloc<ದಶಮಾಂಶ>  dst_tmp(pool);
    ggml_sycl_pool_alloc<sycl::float2> dst_tmp_meta(pool);

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * K_ದತ್ತಾಂಶ = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) K->ದತ್ತಾಂಶ;
    size_t nb11 = K->nb[1];
    size_t nb12 = K->nb[2];
    size_t nb13 = K->nb[3];

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * V_ದತ್ತಾಂಶ = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) V->ದತ್ತಾಂಶ;
    size_t nb21 = V->nb[1];
    size_t nb22 = V->nb[2];
    size_t nb23 = V->nb[3];

    ಆದರೆ (need_f16_K && K->type != GGML_TYPE_F16) {
        ಸ್ಥಿರಾಂಕ size_t bs = ggml_blck_ಗಾತ್ರ(K->type);
        ಸ್ಥಿರಾಂಕ size_t ts = ggml_type_ಗಾತ್ರ(K->type);

        GGML_ASSERT(extra.K_ಬಫರ್_ptr);
        sycl::half * K_f16_ptr = (sycl::half *) extra.K_ಬಫರ್_ptr;
        ಆದರೆ (ggml_is_contiguously_allocated(K)) {
            ಗಮ್ಯ_fp16_sycl_t ಗಮ್ಯ_fp16 = ggml_get_ಗಮ್ಯ_fp16_sycl(K->type, dst);
            ಗಮ್ಯ_fp16(K_ದತ್ತಾಂಶ, K_f16_ptr, ggml_nelements(K), main_ಹರಿವು);

            nb11 = nb11 * bs * ಗಾತ್ರ(sycl::half) / ts;
            nb12 = nb12 * bs * ಗಾತ್ರ(sycl::half) / ts;
            nb13 = nb13 * bs * ಗಾತ್ರ(sycl::half) / ts;
        } ಇಲ್ಲದಿದ್ದರೆ {
            GGML_ASSERT(K->nb[0] == ts);
            ಗಮ್ಯ_fp16_nc_sycl_t ಗಮ್ಯ_fp16 = ggml_get_ಗಮ್ಯ_fp16_nc_sycl(K->type);
            ಸ್ಥಿರಾಂಕ int64_t s01 = nb11 / ts;
            ಸ್ಥಿರಾಂಕ int64_t s02 = nb12 / ts;
            ಸ್ಥಿರಾಂಕ int64_t s03 = nb13 / ts;
            ಗಮ್ಯ_fp16(K_ದತ್ತಾಂಶ, K_f16_ptr, K->ne[0], K->ne[1], K->ne[2], K->ne[3], s01, s02, s03, main_ಹರಿವು);

            nb11 = K->ne[0] * ಗಾತ್ರ(sycl::half);
            nb12 = K->ne[1] * nb11;
            nb13 = K->ne[2] * nb12;
        }
        K_ದತ್ತಾಂಶ = (ಅಕ್ಷರ *) K_f16_ptr;
    }

    ಆದರೆ (need_f16_V && V->type != GGML_TYPE_F16) {
        ಆದರೆ (V_is_K_view) {
            V_ದತ್ತಾಂಶ = K_ದತ್ತಾಂಶ;
            nb21   = nb11;
            nb22   = nb12;
            nb23   = nb13;
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಸ್ಥಿರಾಂಕ size_t bs = ggml_blck_ಗಾತ್ರ(V->type);
            ಸ್ಥಿರಾಂಕ size_t ts = ggml_type_ಗಾತ್ರ(V->type);

            GGML_ASSERT(extra.V_ಬಫರ್_ptr);
            sycl::half * V_f16_ptr = (sycl::half *) extra.V_ಬಫರ್_ptr;
            ಆದರೆ (ggml_is_contiguously_allocated(V)) {
                ಗಮ್ಯ_fp16_sycl_t ಗಮ್ಯ_fp16 = ggml_get_ಗಮ್ಯ_fp16_sycl(V->type, dst);
                ಗಮ್ಯ_fp16(V_ದತ್ತಾಂಶ, V_f16_ptr, ggml_nelements(V), main_ಹರಿವು);
                V_ದತ್ತಾಂಶ = (ಅಕ್ಷರ *) V_f16_ptr;

                nb21 = nb21 * bs * ಗಾತ್ರ(sycl::half) / ts;
                nb22 = nb22 * bs * ಗಾತ್ರ(sycl::half) / ts;
                nb23 = nb23 * bs * ಗಾತ್ರ(sycl::half) / ts;
            } ಇಲ್ಲದಿದ್ದರೆ {
                GGML_ASSERT(V->nb[0] == ts);
                ಗಮ್ಯ_fp16_nc_sycl_t ಗಮ್ಯ_fp16 = ggml_get_ಗಮ್ಯ_fp16_nc_sycl(V->type);
                ಸ್ಥಿರಾಂಕ int64_t s01 = nb21 / ts;
                ಸ್ಥಿರಾಂಕ int64_t s02 = nb22 / ts;
                ಸ್ಥಿರಾಂಕ int64_t s03 = nb23 / ts;
                ಗಮ್ಯ_fp16(V_ದತ್ತಾಂಶ, V_f16_ptr, V->ne[0], V->ne[1], V->ne[2], V->ne[3], s01, s02, s03, main_ಹರಿವು);

                nb21 = V->ne[0] * ಗಾತ್ರ(sycl::half);
                nb22 = V->ne[1] * nb21;
                nb23 = V->ne[2] * nb22;
            }
            V_ದತ್ತಾಂಶ = (ಅಕ್ಷರ *) V_f16_ptr;
        }
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ntiles_x     = ((Q->ne[1] + ncols1 - 1) / ncols1);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ gqa_ಅನುಪಾತ    = Q->ne[2] / K->ne[2];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ntiles_z_gqa = ((gqa_ಅನುಪಾತ + ncols2 - 1) / ncols2);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ntiles_ಒಟ್ಟು = ntiles_x * ntiles_z_gqa * K->ne[2] * Q->ne[3];

    // Optional optimization where the mask is scanned to determine whether part of the calculation can be skipped.
    // Only worth the overhead if there is at lease one FATTN_KQ_STRIDE x FATTN_KQ_STRIDE square to be skipped or
    //     multiple sequences of possibly different lengths.
    ಆದರೆ (ಮುಖವಾಡ && K->ne[1] % FATTN_KQ_STRIDE == 0 && (Q->ne[1] >= 1024 || Q->ne[3] > 1)) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s31 = ಮುಖವಾಡ->nb[1] / ಗಾತ್ರ(sycl::half2);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s33 = ಮುಖವಾಡ->nb[3] / ಗಾತ್ರ(sycl::half2);

        ಸ್ಥಿರಾಂಕ dpct::dim3 ಖಂಡಗಳು_ಅಂಕೆ_KV_max(ntiles_x, Q->ne[3], 1);
        ಸ್ಥಿರಾಂಕ dpct::dim3 ಖಂಡ_dim_KV_max(FATTN_KQ_STRIDE / 2, 1, 1);

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne_KV_max = ಖಂಡಗಳು_ಅಂಕೆ_KV_max.x*ಖಂಡಗಳು_ಅಂಕೆ_KV_max.y;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iter_k = K->ne[1] / FATTN_KQ_STRIDE;

        KV_max.alloc(ne_KV_max);
        {
            dpct::has_capability_or_fail(main_ಹರಿವು->get_device(), { sycl::aspect::fp16 });

            main_ಹರಿವು->submit([&](sycl::handler & cgh) {
                sycl::local_accessor<ಪೂರ್ಣಾಂಕ, 1> buf_iw_acc_ct1(sycl::range<1>(warp_ಗಾತ್ರ), cgh);

                ಸ್ವಯಂಚಾಲಿತ ಮುಖವಾಡ_ದತ್ತಾಂಶ_ct0  = (ಸ್ಥಿರಾಂಕ sycl::half2 *) ಮುಖವಾಡ->ದತ್ತಾಂಶ;
                ಸ್ವಯಂಚಾಲಿತ KV_max_ptr_ct1 = KV_max.ptr;

                cgh.parallel_for(sycl::nd_range<3>(ಖಂಡಗಳು_ಅಂಕೆ_KV_max * ಖಂಡ_dim_KV_max, ಖಂಡ_dim_KV_max),
                                 [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(warp_ಗಾತ್ರ)]] {
                                     GGML_ಬಳಸದ(ಅಂಶ_ct1);
                                     flash_attn_ಮುಖವಾಡ_ಗಮ್ಯ_KV_max<ncols1, warp_ಗಾತ್ರ>(
                                         ಮುಖವಾಡ_ದತ್ತಾಂಶ_ct0, KV_max_ptr_ct1, iter_k, s31, s33,
                                         buf_iw_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                                 });
            });
        }
        SYCL_CHECK(0);
    }

    ಸ್ಥಿರಾಂಕ dpct::dim3 ಖಂಡ_dim(warp_ಗಾತ್ರ, nwarps, 1);

    // Max. number of active blocks limited by occupancy.
    ಪೂರ್ಣಾಂಕ max_ಖಂಡಗಳು_ಪ್ರತಿ_sm = ggml_sycl_info().devices[id].max_wg_ಪ್ರತಿ_cu;
    ಪೂರ್ಣಾಂಕ parallel_ಖಂಡಗಳು = max_ಖಂಡಗಳು_ಪ್ರತಿ_sm;
    dpct::dim3 ಖಂಡಗಳು_ಅಂಕೆ;
    ಆದರೆ (ಹರಿವು_k) {
        // For short contexts it can be faster to have the SMs work on whole tiles because this lets us skip the fixup.
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ max_ಖಂಡಗಳು = max_ಖಂಡಗಳು_ಪ್ರತಿ_sm*nsm;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nblocks_ಹರಿವು_k = max_ಖಂಡಗಳು;
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ use_ಹರಿವು_k = ನಿಜ;

        ಖಂಡಗಳು_ಅಂಕೆ.x = use_ಹರಿವು_k ? nblocks_ಹರಿವು_k : ntiles_ಒಟ್ಟು;
        ಖಂಡಗಳು_ಅಂಕೆ.y = 1;
        ಖಂಡಗಳು_ಅಂಕೆ.z = 1;

        ಆದರೆ (ntiles_ಒಟ್ಟು % ಖಂಡಗಳು_ಅಂಕೆ.x != 0) { // Fixup is only needed if the SMs work on fractional tiles.
            dst_tmp_meta.alloc((size_t(ಖಂಡಗಳು_ಅಂಕೆ.x) * ncols * (2 + DV/2)));
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ntiles_KQ = (K->ne[1] + nbatch_fa - 1) / nbatch_fa; // Max. number of parallel blocks limited by tensor size.

        // parallel_blocks must not be larger than what the tensor size allows:
        parallel_ಖಂಡಗಳು = std::min(parallel_ಖಂಡಗಳು, ntiles_KQ);
        // todo fix the hard code change
        // parallel_blocks = ntiles_KQ;

        // If ntiles_total % blocks_per_wave != 0 then some efficiency is lost due to tail effects.
        // Test whether parallel_blocks can be set to a higher value for better efficiency.
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡಗಳು_ಪ್ರತಿ_wave = nsm * max_ಖಂಡಗಳು_ಪ್ರತಿ_sm;
        ಪೂರ್ಣಾಂಕ nwaves_ಉತ್ತಮ = 0;
        ಪೂರ್ಣಾಂಕ efficiency_percent_ಉತ್ತಮ = 0;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ parallel_ಖಂಡಗಳು_test = parallel_ಖಂಡಗಳು; parallel_ಖಂಡಗಳು_test <= ntiles_KQ; ++parallel_ಖಂಡಗಳು_test) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nblocks_ಒಟ್ಟು = ntiles_ಒಟ್ಟು * parallel_ಖಂಡಗಳು_test;
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nwaves = (nblocks_ಒಟ್ಟು + ಖಂಡಗಳು_ಪ್ರತಿ_wave - 1) / ಖಂಡಗಳು_ಪ್ರತಿ_wave;
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ efficiency_percent = 100 * nblocks_ಒಟ್ಟು / (nwaves*ಖಂಡಗಳು_ಪ್ರತಿ_wave);

            // Stop trying configurations with more waves if we already have good efficiency to avoid excessive overhead.
            ಆದರೆ (efficiency_percent_ಉತ್ತಮ >= 95 && nwaves > nwaves_ಉತ್ತಮ) {
                ನಿಲ್ಲಿಸು;
            }

            ಆದರೆ (efficiency_percent > efficiency_percent_ಉತ್ತಮ) {
                nwaves_ಉತ್ತಮ = nwaves;
                efficiency_percent_ಉತ್ತಮ = efficiency_percent;
                parallel_ಖಂಡಗಳು = parallel_ಖಂಡಗಳು_test;
            }
        }

        ಖಂಡಗಳು_ಅಂಕೆ.x = ntiles_x;
        ಖಂಡಗಳು_ಅಂಕೆ.y = parallel_ಖಂಡಗಳು;
        ಖಂಡಗಳು_ಅಂಕೆ.z = ntiles_z_gqa*K->ne[2]*Q->ne[3];

        ಆದರೆ (parallel_ಖಂಡಗಳು > 1) {
            dst_tmp.alloc(parallel_ಖಂಡಗಳು*ggml_nelements(KQV));
            dst_tmp_meta.alloc(parallel_ಖಂಡಗಳು*ggml_nrows(KQV));
        }
    }

    ದಶಮಾಂಶ scale         = 1.0f;
    ದಶಮಾಂಶ max_bias      = 0.0f;
    ದಶಮಾಂಶ logit_softcap = 0.0f;

    ಸ್ಮರಣೆ_ನಕಲು(&scale,         (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) KQV->op_params + 0, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&max_bias,      (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) KQV->op_params + 1, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&logit_softcap, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) KQV->op_params + 2, ಗಾತ್ರ(ದಶಮಾಂಶ));

    ಆದರೆ (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    ಸ್ಥಿರಾಂಕ uint32_t n_ತಲೆ      = Q->ne[2];
    ಸ್ಥಿರಾಂಕ uint32_t n_ತಲೆ_log2 = 1u << uint32_t(floorf(log2f(ದಶಮಾಂಶ(n_ತಲೆ))));

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ m0 = powf(2.0f, -(max_bias       ) / n_ತಲೆ_log2);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ m1 = powf(2.0f, -(max_bias / 2.0f) / n_ತಲೆ_log2);

    // TODO other tensor dimensions after removal of WMMA kernel:
    ಸ್ಥಿರಾಂಕ sycl::uint3 ne01 = init_fastdiv_ಮೌಲ್ಯಗಳು(Q->ne[1]);

    GGML_ASSERT(ಖಂಡ_dim.x % warp_ಗಾತ್ರ == 0);

    lauch_kernel<fattn_kernel, warp_ಗಾತ್ರ, use_large_grf>(
        ಖಂಡಗಳು_ಅಂಕೆ, ಖಂಡ_dim, main_ಹರಿವು, (ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ) nbytes_shared, (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) Q->ದತ್ತಾಂಶ, K_ದತ್ತಾಂಶ, V_ದತ್ತಾಂಶ,
        ಮುಖವಾಡ ? ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) ಮುಖವಾಡ->ದತ್ತಾಂಶ) : ಶೂನ್ಯಸೂಚಕ, sinks ? ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) sinks->ದತ್ತಾಂಶ) : ಶೂನ್ಯಸೂಚಕ, KV_max.ptr,
        !ಹರಿವು_k && parallel_ಖಂಡಗಳು > 1 ? dst_tmp.ptr : (ದಶಮಾಂಶ *) KQV->ದತ್ತಾಂಶ, (sycl::float2 *)dst_tmp_meta.ptr, scale, max_bias, m0, m1,
        n_ತಲೆ_log2, logit_softcap, Q->ne[0], ne01, Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3], K->ne[0],
        K->ne[1], K->ne[2], K->ne[3], nb11, nb12, nb13, nb21, nb22, nb23, ಮುಖವಾಡ ? ಮುಖವಾಡ->ne[1] : 0,
        ಮುಖವಾಡ ? ಮುಖವಾಡ->ne[2] : 0, ಮುಖವಾಡ ? ಮುಖವಾಡ->ne[3] : 0, ಮುಖವಾಡ ? ಮುಖವಾಡ->nb[1] : 0, ಮುಖವಾಡ ? ಮುಖವಾಡ->nb[2] : 0,
        ಮುಖವಾಡ ? ಮುಖವಾಡ->nb[3] : 0);
    SYCL_CHECK(0);

    ಆದರೆ (ಹರಿವು_k) {
        ಆದರೆ (ntiles_ಒಟ್ಟು % ಖಂಡಗಳು_ಅಂಕೆ.x != 0) { // Fixup is only needed if the SMs work on fractional tiles.
            ಸ್ಥಿರಾಂಕ dpct::dim3 ಖಂಡ_dim_combine(DV, 1, 1);
            ಸ್ಥಿರಾಂಕ dpct::dim3 ಖಂಡಗಳು_ಅಂಕೆ_combine = { ಖಂಡಗಳು_ಅಂಕೆ.x, ncols1, ncols2 };

            main_ಹರಿವು->submit([&](sycl::handler & cgh) {
                ಸ್ವಯಂಚಾಲಿತ KQV_ದತ್ತಾಂಶ_ct0         = (ದಶಮಾಂಶ *) KQV->ದತ್ತಾಂಶ;
                ಸ್ವಯಂಚಾಲಿತ dst_tmp_meta_ptr_ct1 = dst_tmp_meta.ptr;
                ಸ್ವಯಂಚಾಲಿತ Q_ne_ct2             = Q->ne[1];
                ಸ್ವಯಂಚಾಲಿತ Q_ne_ct3             = Q->ne[2];
                ಸ್ವಯಂಚಾಲಿತ Q_ne_ct4             = Q->ne[3];
                ಸ್ವಯಂಚಾಲಿತ K_ne_ct5             = K->ne[1];
                ಸ್ವಯಂಚಾಲಿತ K_ne_ct6             = K->ne[2];

                cgh.parallel_for(sycl::nd_range<3>(ಖಂಡಗಳು_ಅಂಕೆ_combine * ಖಂಡ_dim_combine, ಖಂಡ_dim_combine),
                                 [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(warp_ಗಾತ್ರ)]] {
                                     GGML_ಬಳಸದ(ಅಂಶ_ct1);
                                     flash_attn_ಹರಿವು_k_fixup<DV, ncols1, ncols2>(KQV_ದತ್ತಾಂಶ_ct0, dst_tmp_meta_ptr_ct1,
                                                                                   Q_ne_ct2, Q_ne_ct3, Q_ne_ct4,
                                                                                   K_ne_ct5, K_ne_ct6, nbatch_fa);
                                 });
            });
        }
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (parallel_ಖಂಡಗಳು > 1) {
        ಸ್ಥಿರಾಂಕ dpct::dim3 ಖಂಡ_dim_combine(DV, 1, 1);
        ಸ್ಥಿರಾಂಕ dpct::dim3 ಖಂಡಗಳು_ಅಂಕೆ_combine(Q->ne[1], Q->ne[2], Q->ne[3]);
        ಸ್ಥಿರಾಂಕ size_t     nbytes_shared_combine = parallel_ಖಂಡಗಳು * ಗಾತ್ರ(sycl::float2);
        main_ಹರಿವು->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(sycl::range<1>(nbytes_shared_combine), cgh);

            ಸ್ವಯಂಚಾಲಿತ dst_tmp_ptr_ct0      = dst_tmp.ptr;
            ಸ್ವಯಂಚಾಲಿತ dst_tmp_meta_ptr_ct1 = dst_tmp_meta.ptr;
            ಸ್ವಯಂಚಾಲಿತ KQV_ದತ್ತಾಂಶ_ct2         = (ದಶಮಾಂಶ *) KQV->ದತ್ತಾಂಶ;

            cgh.parallel_for(sycl::nd_range<3>(ಖಂಡಗಳು_ಅಂಕೆ_combine * ಖಂಡ_dim_combine, ಖಂಡ_dim_combine),
                             [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(warp_ಗಾತ್ರ)]] {
                                 GGML_ಬಳಸದ(ಅಂಶ_ct1);
                                 flash_attn_combine_results<DV>(
                                     dst_tmp_ptr_ct0, dst_tmp_meta_ptr_ct1, KQV_ದತ್ತಾಂಶ_ct2, parallel_ಖಂಡಗಳು,
                                     dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                             });
        });
    }
    SYCL_CHECK(0);
}
