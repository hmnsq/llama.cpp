#ਹਦਾਇਤ once

#ਅਨ੍ਤਰ੍ਭਾਵ <sycl/sycl.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <sycl/ext/intel/experimental/grf_size_properties.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ "dpct/helper.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "common.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "convert.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "vecdotq.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "fattn.hpp"

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <cstdint>
#ਅਨ੍ਤਰ੍ਭਾਵ <cmath>
#ਅਨ੍ਤਰ੍ਭਾਵ <float.h>


#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ FATTN_KQ_STRIDE       256
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HALF_MAX_HALF         sycl::half(65504.0f/2) // Use neg. of this instead of -INFINITY to initialize KQ max vals to avoid NaN upon subtraction.
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SOFTMAX_FTZ_ਹੱਦ -20.0f                   // Softmax exp. of values smaller than this are flushed to zero to avoid NaNs.
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ FATTN_KQ_MAX_ਵਿੱਥ (3.0f*0.6931f)

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਖਾਲੀ (*fattn_kernel_t)(
    ਅਚਲ ਅੱਖਰ* Q,
    ਅਚਲ ਅੱਖਰ* K,
    ਅਚਲ ਅੱਖਰ* V,
    ਅਚਲ ਅੱਖਰ* ਨਕਾਬ,
    ਅਚਲ ਅੱਖਰ* sinks,
    ਅਚਲ ਪੂਰਨ* KV_max,
    ਦਸ਼ਮਲਵ* dst,
    sycl::float2* dst_meta,
    ਅਚਲ ਦਸ਼ਮਲਵ scale,
    ਅਚਲ ਦਸ਼ਮਲਵ max_bias,
    ਅਚਲ ਦਸ਼ਮਲਵ m0,
    ਅਚਲ ਦਸ਼ਮਲਵ m1,
    ਅਚਲ uint32_t n_ਸਿਰ_log2,
    ਅਚਲ ਦਸ਼ਮਲਵ logit_softcap,
    ਅਚਲ int32_t ne00,
    ਅਚਲ sycl::uint3 ne01,
    ਅਚਲ int32_t ne02,
    ਅਚਲ int32_t ne03,
    ਅਚਲ int32_t nb01,
    ਅਚਲ int32_t nb02,
    ਅਚਲ int32_t nb03,
    ਅਚਲ int32_t ne10,
    ਅਚਲ int32_t ne11,
    ਅਚਲ int32_t ne12,
    ਅਚਲ int32_t ne13,
    ਅਚਲ int32_t nb11,
    ਅਚਲ int32_t nb12,
    ਅਚਲ int64_t nb13,
    ਅਚਲ int32_t nb21,
    ਅਚਲ int32_t nb22,
    ਅਚਲ int64_t nb23,
    ਅਚਲ int32_t ne31,
    ਅਚਲ int32_t ne32,
    ਅਚਲ int32_t ne33,
    ਅਚਲ int32_t nb31,
    ਅਚਲ int32_t nb32,
    ਅਚਲ int64_t nb33);

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਦਸ਼ਮਲਵ (*vec_dot_KQ_t)(
    ਅਚਲ ਅੱਖਰ * __restrict__ K_c, ਅਚਲ ਖਾਲੀ * __restrict__ Q_v, ਅਚਲ ਪੂਰਨ * __restrict__ Q_q8 , ਅਚਲ ਖਾਲੀ * __restrict__ Q_ds);

ਸਾਂਚਾ <ਪੂਰਨ D, ਪੂਰਨ nthreads>
ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ vec_dot_fattn_vec_KQ_f16(ਅਚਲ ਅੱਖਰ * __restrict__ K_c,
                                                      ਅਚਲ ਖਾਲੀ * __restrict__ Q_v,
                                                      ਅਚਲ ਪੂਰਨ * __restrict__ Q_q8,
                                                      ਅਚਲ ਖਾਲੀ * __restrict__ Q_ds_v) {
    ਅਚਲ sycl::half2 * K_h2 = (ਅਚਲ sycl::half2 *) K_c;
    GGML_ਅਣਵਰਤਿਆ(Q_q8);
    GGML_ਅਣਵਰਤਿਆ(Q_ds_v);

    constexpr ਪੂਰਨ cpy_nb = ggml_sycl_get_max_cpy_ਬਾਈਟਾਂ();
    constexpr ਪੂਰਨ cpy_ne = cpy_nb / 4;

    ਦਸ਼ਮਲਵ ਜੋੜ = 0.0f;

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads*cpy_ne) {
        sycl::half2 tmp[cpy_ne];
        ggml_sycl_memcpy_1<ਆਕਾਰ(tmp)>(
            tmp,
            K_h2 + k_KQ_0 + (sycl::ext::oneapi::this_work_ਵਸਤੂ::get_nd_ਵਸਤੂ<3>().get_local_id(2) % nthreads) * cpy_ne);
#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_F16
            ggml_sycl_mad(ਜੋੜ,                tmp[k_KQ_1] , ((ਅਚਲ sycl::half2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#ਨਹੀਂਤਾਂ
            ggml_sycl_mad(ਜੋੜ, __half22float2(tmp[k_KQ_1]), ((ਅਚਲ sycl::float2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#ਜੇ_ਅੰਤ // GGML_SYCL_F16
        }
    }

    ਵਾਪਸ ਜੋੜ;
}

ਸਾਂਚਾ <ਪੂਰਨ D, ਪੂਰਨ nthreads, ਪੂਰਨ warp_ਮਾਪ>
ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ vec_dot_fattn_vec_KQ_q4_0(ਅਚਲ ਅੱਖਰ * __restrict__ K_c,
                                                       ਅਚਲ ਖਾਲੀ * __restrict__ Q_v,
                                                       ਅਚਲ ਪੂਰਨ * __restrict__ Q_q8,
                                                       ਅਚਲ ਖਾਲੀ * __restrict__ Q_ds_v) {
    ਸਵੈਚਾਲਿਤ ਵਸਤੂ_ct1 = sycl::ext::oneapi::this_work_ਵਸਤੂ::get_nd_ਵਸਤੂ<3>();

    ਅਚਲ ਖੰਡ_q4_0 * K_q4_0   = (ਅਚਲ ਖੰਡ_q4_0 *) K_c;
    GGML_ਅਣਵਰਤਿਆ(Q_v);

    ਦਸ਼ਮਲਵ ਜੋੜ = 0.0f;

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ k_KQ_0 = 0; k_KQ_0 < ਪੂਰਨ(D/ਆਕਾਰ(ਪੂਰਨ)); k_KQ_0 += nthreads) {
        ਅਚਲ ਪੂਰਨ k_KQ =
            k_KQ_0 + (nthreads == warp_ਮਾਪ ? ਵਸਤੂ_ct1.get_local_id(2) : ਵਸਤੂ_ct1.get_local_id(2) % nthreads);

        ਅਚਲ ਪੂਰਨ ib    = k_KQ /  QI8_1;
        ਅਚਲ ਪੂਰਨ iqs4  = k_KQ %  QI4_0;
        ਅਚਲ ਪੂਰਨ ਖਿਸਕਾਓ = k_KQ & (QI8_1/2);

        ਪੂਰਨ v;
        ggml_sycl_memcpy_1<ਆਕਾਰ(ਪੂਰਨ), 2>(&v, K_q4_0[ib].qs + ਆਕਾਰ(ਪੂਰਨ)*iqs4);
        v = (v >> ਖਿਸਕਾਓ) & 0x0F0F0F0F;
        ਅਚਲ ਪੂਰਨ u = Q_q8[k_KQ_0/nthreads];

        ਅਚਲ ਪੂਰਨ sumi = ggml_sycl_dp4a(v, u, 0);

        ਅਚਲ sycl::float2 Q_ds = ((ਅਚਲ sycl::float2 *) Q_ds_v)[k_KQ_0 / nthreads];
        ਜੋੜ += __half2float(K_q4_0[ib].d) * (sumi*Q_ds.x() - (8/QI8_1)*Q_ds.y());
    }

    ਵਾਪਸ ਜੋੜ;
}

ਸਾਂਚਾ <ਪੂਰਨ D, ਪੂਰਨ nthreads , ਪੂਰਨ warp_ਮਾਪ>
ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ vec_dot_fattn_vec_KQ_q4_1(ਅਚਲ ਅੱਖਰ * __restrict__ K_c,
                                                       ਅਚਲ ਖਾਲੀ * __restrict__ Q_v,
                                                       ਅਚਲ ਪੂਰਨ * __restrict__ Q_q8,
                                                       ਅਚਲ ਖਾਲੀ * __restrict__ Q_ds_v) {
    ਸਵੈਚਾਲਿਤ               ਵਸਤੂ_ct1 = sycl::ext::oneapi::this_work_ਵਸਤੂ::get_nd_ਵਸਤੂ<3>();
    ਅਚਲ ਖੰਡ_q4_1 * K_q4_1   = (ਅਚਲ ਖੰਡ_q4_1 *) K_c;
    GGML_ਅਣਵਰਤਿਆ(Q_v);

    ਦਸ਼ਮਲਵ ਜੋੜ = 0.0f;

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ k_KQ_0 = 0; k_KQ_0 < ਪੂਰਨ(D/ਆਕਾਰ(ਪੂਰਨ)); k_KQ_0 += nthreads) {
        ਅਚਲ ਪੂਰਨ k_KQ =
            k_KQ_0 + (nthreads == warp_ਮਾਪ ? ਵਸਤੂ_ct1.get_local_id(2) : ਵਸਤੂ_ct1.get_local_id(2) % nthreads);

        ਅਚਲ ਪੂਰਨ ib    = k_KQ /  QI8_1;
        ਅਚਲ ਪੂਰਨ iqs4  = k_KQ %  QI4_1;
        ਅਚਲ ਪੂਰਨ ਖਿਸਕਾਓ = k_KQ & (QI8_1/2);

        ਪੂਰਨ v;
        ggml_sycl_memcpy_1<ਆਕਾਰ(ਪੂਰਨ)>(&v, K_q4_1[ib].qs + ਆਕਾਰ(ਪੂਰਨ)*iqs4);
        v = (v >> ਖਿਸਕਾਓ) & 0x0F0F0F0F;
        ਅਚਲ ਪੂਰਨ u = Q_q8[k_KQ_0/nthreads];

        ਅਚਲ ਪੂਰਨ sumi = ggml_sycl_dp4a(v, u, 0);

        ਅਚਲ sycl::float2 K_dm = (K_q4_1[ib].dm).ਸਾਂਚਾ convert<ਦਸ਼ਮਲਵ, sycl::rounding_ਬਹੁਲਕ::automatic>();
        ਅਚਲ sycl::float2 Q_ds = ((ਅਚਲ sycl::float2 *) Q_ds_v)[k_KQ_0 / nthreads];

        ਜੋੜ += K_dm.x()*Q_ds.x()*sumi + K_dm.y()*Q_ds.y()/QI8_1;
    }

    ਵਾਪਸ ਜੋੜ;
}

ਸਾਂਚਾ <ਪੂਰਨ D, ਪੂਰਨ nthreads, ਪੂਰਨ warp_ਮਾਪ>
ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ vec_dot_fattn_vec_KQ_q5_0(ਅਚਲ ਅੱਖਰ * __restrict__ K_c,
                                                       ਅਚਲ ਖਾਲੀ * __restrict__ Q_v,
                                                       ਅਚਲ ਪੂਰਨ * __restrict__ Q_q8,
                                                       ਅਚਲ ਖਾਲੀ * __restrict__ Q_ds_v) {
    ਸਵੈਚਾਲਿਤ               ਵਸਤੂ_ct1 = sycl::ext::oneapi::this_work_ਵਸਤੂ::get_nd_ਵਸਤੂ<3>();
    ਅਚਲ ਖੰਡ_q5_0 * K_q5_0   = (ਅਚਲ ਖੰਡ_q5_0 *) K_c;
    GGML_ਅਣਵਰਤਿਆ(Q_v);

    ਦਸ਼ਮਲਵ ਜੋੜ = 0.0f;

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ k_KQ_0 = 0; k_KQ_0 < ਪੂਰਨ(D/ਆਕਾਰ(ਪੂਰਨ)); k_KQ_0 += nthreads) {
        ਅਚਲ ਪੂਰਨ k_KQ =
            k_KQ_0 + (nthreads == warp_ਮਾਪ ? ਵਸਤੂ_ct1.get_local_id(2) : ਵਸਤੂ_ct1.get_local_id(2) % nthreads);

        ਅਚਲ ਪੂਰਨ ib    = k_KQ /  QI8_1;
        ਅਚਲ ਪੂਰਨ iqs4  = k_KQ %  QI5_0;
        ਅਚਲ ਪੂਰਨ iqs8  = k_KQ %  QI8_1;
        ਅਚਲ ਪੂਰਨ ਖਿਸਕਾਓ = k_KQ & (QI8_1/2);

        ਪੂਰਨ v;
        ggml_sycl_memcpy_1<ਆਕਾਰ(ਪੂਰਨ), 2>(&v, K_q5_0[ib].qs + ਆਕਾਰ(ਪੂਰਨ)*iqs4);
        v = (v >> ਖਿਸਕਾਓ) & 0x0F0F0F0F;

        {
            ਪੂਰਨ vh;
            ggml_sycl_memcpy_1<ਆਕਾਰ(ਪੂਰਨ), 2>(&vh, K_q5_0[ib].qh);
            vh >>= iqs8 * QI5_0;

            v |= (vh <<  4) & 0x00000010; // 0 ->  4
            v |= (vh << 11) & 0x00001000; // 1 -> 12
            v |= (vh << 18) & 0x00100000; // 2 -> 20
            v |= (vh << 25) & 0x10000000; // 3 -> 28
        }

        ਅਚਲ ਪੂਰਨ u = Q_q8[k_KQ_0/nthreads];

        ਅਚਲ ਪੂਰਨ sumi = ggml_sycl_dp4a(v, u, 0);

        ਅਚਲ sycl::float2 Q_ds = ((ਅਚਲ sycl::float2 *) Q_ds_v)[k_KQ_0 / nthreads];

        ਜੋੜ += __half2float(K_q5_0[ib].d) * (sumi*Q_ds.x() - (16/QI8_1)*Q_ds.y());
    }

    ਵਾਪਸ ਜੋੜ;
}

ਸਾਂਚਾ <ਪੂਰਨ D, ਪੂਰਨ nthreads, ਪੂਰਨ warp_ਮਾਪ>
ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ vec_dot_fattn_vec_KQ_q5_1(ਅਚਲ ਅੱਖਰ * __restrict__ K_c,
                                                       ਅਚਲ ਖਾਲੀ * __restrict__ Q_v,
                                                       ਅਚਲ ਪੂਰਨ * __restrict__ Q_q8,
                                                       ਅਚਲ ਖਾਲੀ * __restrict__ Q_ds_v) {
    ਸਵੈਚਾਲਿਤ               ਵਸਤੂ_ct1 = sycl::ext::oneapi::this_work_ਵਸਤੂ::get_nd_ਵਸਤੂ<3>();
    ਅਚਲ ਖੰਡ_q5_1 * K_q5_1   = (ਅਚਲ ਖੰਡ_q5_1 *) K_c;
    GGML_ਅਣਵਰਤਿਆ(Q_v);

    ਦਸ਼ਮਲਵ ਜੋੜ = 0.0f;

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ k_KQ_0 = 0; k_KQ_0 < ਪੂਰਨ(D/ਆਕਾਰ(ਪੂਰਨ)); k_KQ_0 += nthreads) {
        ਅਚਲ ਪੂਰਨ k_KQ =
            k_KQ_0 + (nthreads == warp_ਮਾਪ ? ਵਸਤੂ_ct1.get_local_id(2) : ਵਸਤੂ_ct1.get_local_id(2) % nthreads);

        ਅਚਲ ਪੂਰਨ ib    = k_KQ /  QI8_1;
        ਅਚਲ ਪੂਰਨ iqs4  = k_KQ %  QI5_1;
        ਅਚਲ ਪੂਰਨ iqs8  = k_KQ %  QI8_1;
        ਅਚਲ ਪੂਰਨ ਖਿਸਕਾਓ = k_KQ & (QI8_1/2);

        ਪੂਰਨ v;
        ggml_sycl_memcpy_1<ਆਕਾਰ(ਪੂਰਨ)>(&v, K_q5_1[ib].qs + ਆਕਾਰ(ਪੂਰਨ)*iqs4);
        v = (v >> ਖਿਸਕਾਓ) & 0x0F0F0F0F;

        {
            ਪੂਰਨ vh;
            ggml_sycl_memcpy_1<ਆਕਾਰ(ਪੂਰਨ)>(&vh, K_q5_1[ib].qh);
            vh >>= iqs8 * QI5_0;

            v |= (vh <<  4) & 0x00000010; // 0 ->  4
            v |= (vh << 11) & 0x00001000; // 1 -> 12
            v |= (vh << 18) & 0x00100000; // 2 -> 20
            v |= (vh << 25) & 0x10000000; // 3 -> 28
        }

        ਅਚਲ ਪੂਰਨ u = Q_q8[k_KQ_0/nthreads];

        ਅਚਲ ਪੂਰਨ sumi = ggml_sycl_dp4a(v, u, 0);

        ਅਚਲ sycl::float2 K_dm = (K_q5_1[ib].dm).ਸਾਂਚਾ convert<ਦਸ਼ਮਲਵ, sycl::rounding_ਬਹੁਲਕ::automatic>();
        ਅਚਲ sycl::float2 Q_ds = ((ਅਚਲ sycl::float2 *) Q_ds_v)[k_KQ_0 / nthreads];

        ਜੋੜ += K_dm.x()*Q_ds.x()*sumi + K_dm.y()*Q_ds.y()/QI8_1;
    }

    ਵਾਪਸ ਜੋੜ;
}

ਸਾਂਚਾ <ਪੂਰਨ D, ਪੂਰਨ nthreads, ਪੂਰਨ warp_ਮਾਪ>
ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ vec_dot_fattn_vec_KQ_q8_0(ਅਚਲ ਅੱਖਰ * __restrict__ K_c,
                                                       ਅਚਲ ਖਾਲੀ * __restrict__ Q_v,
                                                       ਅਚਲ ਪੂਰਨ * __restrict__ Q_q8,
                                                       ਅਚਲ ਖਾਲੀ * __restrict__ Q_ds_v) {
    ਸਵੈਚਾਲਿਤ               ਵਸਤੂ_ct1 = sycl::ext::oneapi::this_work_ਵਸਤੂ::get_nd_ਵਸਤੂ<3>();
    ਅਚਲ ਖੰਡ_q8_0 * K_q8_0   = (ਅਚਲ ਖੰਡ_q8_0 *) K_c;
    GGML_ਅਣਵਰਤਿਆ(Q_v);

    ਦਸ਼ਮਲਵ ਜੋੜ = 0.0f;

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ k_KQ_0 = 0; k_KQ_0 < ਪੂਰਨ(D/ਆਕਾਰ(ਪੂਰਨ)); k_KQ_0 += nthreads) {
        ਅਚਲ ਪੂਰਨ k_KQ =
            k_KQ_0 + (nthreads == warp_ਮਾਪ ? ਵਸਤੂ_ct1.get_local_id(2) : ਵਸਤੂ_ct1.get_local_id(2) % nthreads);

        ਅਚਲ ਪੂਰਨ ib  = k_KQ / QI8_0;
        ਅਚਲ ਪੂਰਨ iqs = k_KQ % QI8_0;

        ਪੂਰਨ v;
        ggml_sycl_memcpy_1<ਆਕਾਰ(v), 2>(&v, K_q8_0[ib].qs + 4*iqs);

        ਅਚਲ sycl::float2 * Q_ds = (ਅਚਲ sycl::float2 *) Q_ds_v;
        ਅਚਲ ਦਸ਼ਮਲਵ          Q_d  = Q_ds[k_KQ_0 / nthreads].x();

        ਜੋੜ += vec_dot_q8_0_q8_1_impl<ਦਸ਼ਮਲਵ, 1>(&v, &Q_q8[k_KQ_0/nthreads], K_q8_0[ib].d, Q_d);
    }

    ਵਾਪਸ ਜੋੜ;
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ Tds, ਪੂਰਨ ni, ਪੂਰਨ warp_ਮਾਪ>
ਸਥਿਰ __dpct_inline__ ਖਾਲੀ quantize_q8_1_ਮੰਜ਼ਿਲ_shared(ਅਚਲ ਦਸ਼ਮਲਵ * __restrict__ x,
                                                    ਅਚਲ ਦਸ਼ਮਲਵ scale,
                                                    ਪੂਰਨ * __restrict__ yq32,
                                                    ਖਾਲੀ * __restrict__ yds) {
    ਸਵੈਚਾਲਿਤ ਵਸਤੂ_ct1 = sycl::ext::oneapi::this_work_ਵਸਤੂ::get_nd_ਵਸਤੂ<3>();

    ਦਸ਼ਮਲਵ vals[ਆਕਾਰ(ਪੂਰਨ)] = { 0.0f };
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ l = 0; l < ਪੂਰਨ(ਆਕਾਰ(ਪੂਰਨ)); ++l) {
        vals[l] =
            (ni == warp_ਮਾਪ || ਵਸਤੂ_ct1.get_local_id(2) < ni) ? scale * x[4 * ਵਸਤੂ_ct1.get_local_id(2) + l] : 0.0f;
    }

    ਦਸ਼ਮਲਵ amax = sycl::fabs(vals[0]);
    ਦਸ਼ਮਲਵ ਜੋੜ  = vals[0];
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ l = 1; l < ਪੂਰਨ(ਆਕਾਰ(ਪੂਰਨ)); ++l) {
        amax = sycl::fmax(amax, sycl::fabs(vals[l]));
        ਜੋੜ += vals[l];
    }
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ ਨਕਾਬ = QI8_1/2; ਨਕਾਬ > 0; ਨਕਾਬ >>= 1) {
        amax = sycl::fmax(
            amax, dpct::ਕ੍ਰਮ_ਬਦਲੋ_sub_ਸਮੂਹ_by_xor(sycl::ext::oneapi::this_work_ਵਸਤੂ::get_sub_ਸਮੂਹ(), amax, ਨਕਾਬ));
        ਜੋੜ += dpct::ਕ੍ਰਮ_ਬਦਲੋ_sub_ਸਮੂਹ_by_xor(sycl::ext::oneapi::this_work_ਵਸਤੂ::get_sub_ਸਮੂਹ(), ਜੋੜ, ਨਕਾਬ);
    }

    ਅਚਲ ਦਸ਼ਮਲਵ d = amax / 127;
    ਪੂਰਨ q32 = 0;
    int8_t * q8 = (int8_t *) &q32;

    ਜੇ (d != 0.0f) {
#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ l = 0; l < ਪੂਰਨ(ਆਕਾਰ(ਪੂਰਨ)); ++l) {
            q8[l] = sycl::round(vals[l] / d);
        }
    }

    yq32[ਵਸਤੂ_ct1.get_local_id(2)] = q32;
    ਜੇ (ਵਸਤੂ_ct1.get_local_id(2) % QI8_1 == 0 && (ni == warp_ਮਾਪ || ਵਸਤੂ_ct1.get_local_id(2) < ni)) {
        ਜੇ (std::is_same<Tds, sycl::half2>::value) {
            ((sycl::half2  *) yds)[ਵਸਤੂ_ct1.get_local_id(2)/QI8_1] =  make_half2(d, ਜੋੜ);
        } ਨਹੀਂਤਾਂ {
            ((sycl::float2 *) yds)[ਵਸਤੂ_ct1.get_local_id(2)/QI8_1] = make_float2(d, ਜੋੜ);
        }
    }
}

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਖਾਲੀ (*dequantize_V_t)(ਅਚਲ ਖਾਲੀ *, ਖਾਲੀ *, ਅਚਲ int64_t);

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T, ਪੂਰਨ ne>
ਸਥਿਰ __dpct_inline__ ਖਾਲੀ dequantize_V_f16(ਅਚਲ ਖਾਲੀ * __restrict__ vx, ਖਾਲੀ * __restrict__ dst, ਅਚਲ int64_t i0) {
    ਜੇ constexpr (std::is_same_v<T, sycl::half>) {
        ggml_sycl_memcpy_1<ne * ਆਕਾਰ(sycl::half)>(dst, (ਅਚਲ sycl::half *) vx + i0);
    } ਨਹੀਂਤਾਂ ਜੇ constexpr (std::is_same_v<T, ਦਸ਼ਮਲਵ>) {
        static_assert(ne % 2 == 0, "bad ne");
        sycl::half2 tmp[ne / 2];
        ggml_sycl_memcpy_1<ne * ਆਕਾਰ(sycl::half)>(tmp, (ਅਚਲ sycl::half *) vx + i0);
        sycl::float2 * dst_f2 = (sycl::float2 *) dst;
#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ l = 0; l < ne/2; ++l) {
            dst_f2[l] = tmp[l].ਸਾਂਚਾ convert<ਦਸ਼ਮਲਵ, sycl::rounding_ਬਹੁਲਕ::automatic>();
        }
    } ਨਹੀਂਤਾਂ {
        static_assert(std::is_same_v<T, ਖਾਲੀ>, "unsupported type");
    }
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T, ਪੂਰਨ ne>
ਸਥਿਰ __dpct_inline__ ਖਾਲੀ dequantize_V_q4_0(ਅਚਲ ਖਾਲੀ * __restrict__ vx, ਖਾਲੀ * __restrict__ dst, ਅਚਲ int64_t i0) {
    ਅਚਲ ਖੰਡ_q4_0 * x = (ਅਚਲ ਖੰਡ_q4_0 *) vx;

    ਅਚਲ int64_t ib    =  i0          /  QK4_0;
    ਅਚਲ ਪੂਰਨ     iqs   =  i0          % (QK4_0/2);
    ਅਚਲ ਪੂਰਨ     ਖਿਸਕਾਓ = (i0 % QK4_0) / (QK4_0/2);

    ਪੂਰਨ q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_sycl_memcpy_1<ne, 2>(&q, x[ib].qs + iqs);
    q >>= 4*ਖਿਸਕਾਓ;
    q &= 0x0F0F0F0F;
    q = dpct::vectorized_ਦੁਆਧਾਰੀ<sycl::char4>(q, 0x08080808, dpct::sub_sat());

    ਅਚਲ int8_t * q8 = (ਅਚਲ int8_t *) &q;

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_F16
    ਜੇ constexpr (std::is_same_v<T, sycl::half>) {
        ਅਚਲ sycl::half2 d = sycl::half2(x[ib].d);

#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * sycl::half2(q8[l0 + 0], q8[l0 + 1]);
        }
    } ਨਹੀਂਤਾਂ
#ਜੇ_ਅੰਤ // GGML_SYCL_F16
    ਜੇ constexpr (std::is_same_v<T, ਦਸ਼ਮਲਵ>) {
        ਅਚਲ ਦਸ਼ਮਲਵ d = x[ib].d;

#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ l = 0; l < ne; ++l) {
            ((ਦਸ਼ਮਲਵ *) dst)[l] = d * q8[l];
        }
    } ਨਹੀਂਤਾਂ {
        static_assert(std::is_same_v<T, ਖਾਲੀ>, "bad type");
    }
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T, ਪੂਰਨ ne>
ਸਥਿਰ __dpct_inline__ ਖਾਲੀ dequantize_V_q4_1(ਅਚਲ ਖਾਲੀ * __restrict__ vx, ਖਾਲੀ * __restrict__ dst, ਅਚਲ int64_t i0) {
    ਅਚਲ ਖੰਡ_q4_1 * x = (ਅਚਲ ਖੰਡ_q4_1 *) vx;

    ਅਚਲ int64_t ib    =  i0          /  QK4_1;
    ਅਚਲ ਪੂਰਨ     iqs   =  i0          % (QK4_1/2);
    ਅਚਲ ਪੂਰਨ     ਖਿਸਕਾਓ = (i0 % QK4_1) / (QK4_1/2);

    ਪੂਰਨ q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_sycl_memcpy_1<ne>(&q, x[ib].qs + iqs);
    q >>= 4*ਖਿਸਕਾਓ;
    q &= 0x0F0F0F0F;

    ਅਚਲ int8_t * q8 = (ਅਚਲ int8_t *) &q;

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_F16
    ਜੇ constexpr (std::is_same_v<T, sycl::half>) {
        ਅਚਲ sycl::half2 dm = x[ib].dm;
        ਅਚਲ sycl::half2 d  = sycl::half2(dm[0]);
        ਅਚਲ sycl::half2 m  = sycl::half2(dm[1]);

#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * sycl::half2(q8[l0 + 0], q8[l0 + 1]) + m;
        }
    } ਨਹੀਂਤਾਂ
#ਜੇ_ਅੰਤ // GGML_SYCL_F16
    ਜੇ constexpr (std::is_same_v<T, ਦਸ਼ਮਲਵ>) {
        ਅਚਲ sycl::float2 dm = (x[ib].dm).ਸਾਂਚਾ convert<ਦਸ਼ਮਲਵ, sycl::rounding_ਬਹੁਲਕ::automatic>();

#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ l = 0; l < ne; ++l) {
            ((ਦਸ਼ਮਲਵ *) dst)[l] = dm.x() * q8[l] + dm.y();
        }
    } ਨਹੀਂਤਾਂ {
        static_assert(std::is_same_v<T, ਖਾਲੀ>, "bad type");
    }
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T, ਪੂਰਨ ne>
ਸਥਿਰ __dpct_inline__ ਖਾਲੀ dequantize_V_q5_0(ਅਚਲ ਖਾਲੀ * __restrict__ vx, ਖਾਲੀ * __restrict__ dst, ਅਚਲ int64_t i0) {
    ਅਚਲ ਖੰਡ_q5_0 * x = (ਅਚਲ ਖੰਡ_q5_0 *) vx;

    ਅਚਲ int64_t ib    =  i0          /  QK5_0;
    ਅਚਲ ਪੂਰਨ     idq   =  i0          %  QK5_0;
    ਅਚਲ ਪੂਰਨ     iqs   =  i0          % (QK5_0/2);
    ਅਚਲ ਪੂਰਨ     ਖਿਸਕਾਓ = (i0 % QK5_0) / (QK5_0/2);

    ਪੂਰਨ q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_sycl_memcpy_1<ne, 2>(&q, x[ib].qs + iqs);
    q >>= 4*ਖਿਸਕਾਓ;
    q &= 0x0F0F0F0F;

    {
        ਪੂਰਨ qh;
        ggml_sycl_memcpy_1<ne, 2>(&qh, x[ib].qh);
#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ l = 0; l < ne; ++l) {
            q |= ((qh >> (idq + l)) & 0x00000001) << (8*l + 4);
        }
    }

    q = dpct::vectorized_ਦੁਆਧਾਰੀ<sycl::char4>(q, 0x10101010, dpct::sub_sat());

    ਅਚਲ int8_t * q8 = (ਅਚਲ int8_t *) &q;

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_F16
    ਜੇ constexpr (std::is_same_v<T, sycl::half>) {
        ਅਚਲ sycl::half2 d = sycl::half2(x[ib].d);

#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * sycl::half2(q8[l0 + 0], q8[l0 + 1]);
        }
    } ਨਹੀਂਤਾਂ
#ਜੇ_ਅੰਤ // GGML_SYCL_F16
    ਜੇ constexpr (std::is_same_v<T, ਦਸ਼ਮਲਵ>) {
        ਅਚਲ ਦਸ਼ਮਲਵ d = x[ib].d;

#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ l = 0; l < ne; ++l) {
            ((ਦਸ਼ਮਲਵ *) dst)[l] = d * q8[l];
        }
    } ਨਹੀਂਤਾਂ {
        static_assert(std::is_same_v<T, ਖਾਲੀ>, "bad type");
    }
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T, ਪੂਰਨ ne>
ਸਥਿਰ __dpct_inline__ ਖਾਲੀ dequantize_V_q5_1(ਅਚਲ ਖਾਲੀ * __restrict__ vx, ਖਾਲੀ * __restrict__ dst, ਅਚਲ int64_t i0) {
    ਅਚਲ ਖੰਡ_q5_1 * x = (ਅਚਲ ਖੰਡ_q5_1 *) vx;

    ਅਚਲ int64_t ib    =  i0          /  QK5_1;
    ਅਚਲ ਪੂਰਨ     idq   =  i0          %  QK5_1;
    ਅਚਲ ਪੂਰਨ     iqs   =  i0          % (QK5_1/2);
    ਅਚਲ ਪੂਰਨ     ਖਿਸਕਾਓ = (i0 % QK5_1) / (QK5_1/2);

    ਪੂਰਨ q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_sycl_memcpy_1<ne>(&q, x[ib].qs + iqs);
    q >>= 4*ਖਿਸਕਾਓ;
    q &= 0x0F0F0F0F;

    {
        ਪੂਰਨ qh;
        ggml_sycl_memcpy_1<ne>(&qh, x[ib].qh);
#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ l = 0; l < ne; ++l) {
            q |= ((qh >> (idq + l)) & 0x00000001) << (8*l + 4);
        }
    }

    ਅਚਲ int8_t * q8 = (ਅਚਲ int8_t *) &q;

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_F16
    ਜੇ constexpr (std::is_same_v<T, sycl::half>) {
        ਅਚਲ sycl::half2 dm = x[ib].dm;
        ਅਚਲ sycl::half2 d  = sycl::half2(dm[0]);
        ਅਚਲ sycl::half2 m  = sycl::half2(dm[1]);

#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * sycl::half2(q8[l0 + 0], q8[l0 + 1]) + m;
        }
    } ਨਹੀਂਤਾਂ
#ਜੇ_ਅੰਤ // GGML_SYCL_F16
    ਜੇ constexpr (std::is_same_v<T, ਦਸ਼ਮਲਵ>) {
        ਅਚਲ sycl::float2 dm = (x[ib].dm).ਸਾਂਚਾ convert<ਦਸ਼ਮਲਵ, sycl::rounding_ਬਹੁਲਕ::automatic>();

#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ l = 0; l < ne; ++l) {
            ((ਦਸ਼ਮਲਵ *) dst)[l] = dm.x() * q8[l] + dm.y();
        }
    } ਨਹੀਂਤਾਂ {
        static_assert(std::is_same_v<T, ਖਾਲੀ>, "bad type");
    }
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T, ਪੂਰਨ ne>
ਸਥਿਰ __dpct_inline__ ਖਾਲੀ dequantize_V_q8_0(ਅਚਲ ਖਾਲੀ * __restrict__ vx, ਖਾਲੀ * __restrict__ dst, ਅਚਲ int64_t i0) {
    ਅਚਲ ਖੰਡ_q8_0 * x = (ਅਚਲ ਖੰਡ_q8_0 *) vx;

    ਅਚਲ int64_t ib  = i0 / QK8_0;
    ਅਚਲ ਪੂਰਨ     iqs = i0 % QK8_0;

    static_assert(ne % 2 == 0, "bad ne");
    int8_t qs[ne];
    ggml_sycl_memcpy_1<ne, 2>(qs, x[ib].qs + iqs);

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_F16
    ਜੇ constexpr (std::is_same<T, sycl::half>::value) {
        ਅਚਲ sycl::half2 d = sycl::half2(x[ib].d);

#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * make_half2(qs[l0 + 0], qs[l0 + 1]);
        }
    } ਨਹੀਂਤਾਂ
#ਜੇ_ਅੰਤ // GGML_SYCL_F16
    ਜੇ constexpr (std::is_same<T, ਦਸ਼ਮਲਵ>::value) {
        ਅਚਲ ਦਸ਼ਮਲਵ d = x[ib].d;

#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ l = 0; l < ne; ++l) {
            ((ਦਸ਼ਮਲਵ *) dst)[l] = d * qs[l];
        }
    } ਨਹੀਂਤਾਂ {
        static_assert(std::is_same_v<T, ਖਾਲੀ>, "unsupported type");
    }
}

ਸਾਂਚਾ <ਪੂਰਨ type_K, ਪੂਰਨ D, ਪੂਰਨ nthreads, ਪੂਰਨ warp_ਮਾਪ>
constexpr vec_dot_KQ_t get_vec_dot_KQ() {
    ਜੇ constexpr (type_K == GGML_TYPE_F16) {
        ਵਾਪਸ vec_dot_fattn_vec_KQ_f16<D, nthreads>;
    } ਨਹੀਂਤਾਂ ਜੇ constexpr (type_K == GGML_TYPE_Q4_0) {
        ਵਾਪਸ vec_dot_fattn_vec_KQ_q4_0<D, nthreads, warp_ਮਾਪ>;
    } ਨਹੀਂਤਾਂ ਜੇ constexpr (type_K == GGML_TYPE_Q4_1) {
        ਵਾਪਸ vec_dot_fattn_vec_KQ_q4_1<D, nthreads, warp_ਮਾਪ>;
    } ਨਹੀਂਤਾਂ ਜੇ constexpr (type_K == GGML_TYPE_Q5_0) {
        ਵਾਪਸ vec_dot_fattn_vec_KQ_q5_0<D, nthreads, warp_ਮਾਪ>;
    } ਨਹੀਂਤਾਂ ਜੇ constexpr (type_K == GGML_TYPE_Q5_1) {
        ਵਾਪਸ vec_dot_fattn_vec_KQ_q5_1<D, nthreads, warp_ਮਾਪ>;
    } ਨਹੀਂਤਾਂ ਜੇ constexpr (type_K == GGML_TYPE_Q8_0) {
        ਵਾਪਸ vec_dot_fattn_vec_KQ_q8_0<D, nthreads, warp_ਮਾਪ>;
    } ਨਹੀਂਤਾਂ {
        static_assert(type_K == -1, "bad type");
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }
}

ਸਾਂਚਾ <ਪੂਰਨ type_V, ਕਿਸਮ_ਨਾਂ T, ਪੂਰਨ ne>
constexpr dequantize_V_t get_dequantize_V() {
    ਜੇ constexpr (type_V == GGML_TYPE_F16) {
        ਵਾਪਸ dequantize_V_f16<T, ne>;
    } ਨਹੀਂਤਾਂ ਜੇ constexpr (type_V == GGML_TYPE_Q4_0) {
        ਵਾਪਸ dequantize_V_q4_0<T, ne>;
    } ਨਹੀਂਤਾਂ ਜੇ constexpr (type_V == GGML_TYPE_Q4_1) {
        ਵਾਪਸ dequantize_V_q4_1<T, ne>;
    } ਨਹੀਂਤਾਂ ਜੇ constexpr (type_V == GGML_TYPE_Q5_0) {
        ਵਾਪਸ dequantize_V_q5_0<T, ne>;
    } ਨਹੀਂਤਾਂ ਜੇ constexpr (type_V == GGML_TYPE_Q5_1) {
        ਵਾਪਸ dequantize_V_q5_1<T, ne>;
    } ਨਹੀਂਤਾਂ ਜੇ constexpr (type_V == GGML_TYPE_Q8_0) {
        ਵਾਪਸ dequantize_V_q8_0<T, ne>;
    } ਨਹੀਂਤਾਂ {
        static_assert(type_V == -1, "bad type");
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }
}

ਸਾਂਚਾ <ਪੂਰਨ ncols1, ਪੂਰਨ warp_ਮਾਪ>
ਸਥਿਰ ਖਾਲੀ flash_attn_ਨਕਾਬ_ਮੰਜ਼ਿਲ_KV_max(ਅਚਲ sycl::half2 * __restrict__ ਨਕਾਬ,
                                      ਪੂਰਨ * __restrict__ KV_max,
                                      ਅਚਲ ਪੂਰਨ ne30,
                                      ਅਚਲ ਪੂਰਨ s31,
                                      ਅਚਲ ਪੂਰਨ s33,
                                      ਪੂਰਨ *     buf_iw) {
    ਸਵੈਚਾਲਿਤ      ਵਸਤੂ_ct1 = sycl::ext::oneapi::this_work_ਵਸਤੂ::get_nd_ਵਸਤੂ<3>();
    ਅਚਲ ਪੂਰਨ ne31     = ਵਸਤੂ_ct1.get_ਸਮੂਹ_range(2);
    ਅਚਲ ਪੂਰਨ tid      = ਵਸਤੂ_ct1.get_local_id(2);
    ਅਚਲ ਪੂਰਨ sequence = ਵਸਤੂ_ct1.get_ਸਮੂਹ(1);
    ਅਚਲ ਪੂਰਨ jt       = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2);

    ਨਕਾਬ += sequence*s33 + jt*ncols1*s31;

    ਜੇ (tid < warp_ਮਾਪ) {
        buf_iw[tid] = 1;
    }
    ਵਸਤੂ_ct1.barrier(sycl::access::fence_space::local_space);

    ਪੂਰਨ KV_max_sj = (ne30 - 1) * FATTN_KQ_STRIDE;
    ਲਈ (; KV_max_sj >= 0; KV_max_sj -= FATTN_KQ_STRIDE) {
        ਪੂਰਨ all_inf = 1;

#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ j = 0; j < ncols1; ++j) {
            ਅਚਲ sycl::float2 tmp =
                ਨਕਾਬ[j * s31 + KV_max_sj / 2 + tid].ਸਾਂਚਾ convert<ਦਸ਼ਮਲਵ, sycl::rounding_ਬਹੁਲਕ::automatic>();
            all_inf = all_inf && ਪੂਰਨ(sycl::isinf((ਦਸ਼ਮਲਵ) (tmp.x()))) && ਪੂਰਨ(sycl::isinf((ਦਸ਼ਮਲਵ) (tmp.y())));
        }

        all_inf = warp_reduce_all<warp_ਮਾਪ>(all_inf);
        ਜੇ (tid % warp_ਮਾਪ == 0) {
            buf_iw[tid / warp_ਮਾਪ] = all_inf;
        }
        ਵਸਤੂ_ct1.barrier(sycl::access::fence_space::local_space);
        all_inf = buf_iw[tid % warp_ਮਾਪ];
        ਵਸਤੂ_ct1.barrier(sycl::access::fence_space::local_space);
        all_inf = warp_reduce_all<warp_ਮਾਪ>(all_inf);

        ਜੇ (!all_inf) {
            ਰੋਕੋ;
        }
    }

    // If the break in the loop was not triggered, KV_max_sj is now -FATTN_KQ_STRIDE.
    // If the break was triggered it's the lower edge of the tile with the first non-masked values.
    // In either case, walk back the decrementation by FATTN_KQ_STRIDE.
    KV_max_sj += FATTN_KQ_STRIDE;

    ਜੇ (ਵਸਤੂ_ct1.get_local_id(2) != 0) {
        ਵਾਪਸ;
    }

    KV_max[sequence*ne31 + jt] = KV_max_sj;
}

ਸਾਂਚਾ <ਪੂਰਨ D, ਪੂਰਨ ncols1, ਪੂਰਨ ncols2>  // D == head size

ਸਥਿਰ ਖਾਲੀ flash_attn_ਧਾਰਾ_k_fixup(ਦਸ਼ਮਲਵ * __restrict__ dst,
                                      ਅਚਲ sycl::float2 * __restrict__ dst_fixup,
                                      ਅਚਲ ਪੂਰਨ ne01,
                                      ਅਚਲ ਪੂਰਨ ne02,
                                      ਅਚਲ ਪੂਰਨ ne03,
                                      ਅਚਲ ਪੂਰਨ ne11,
                                      ਅਚਲ ਪੂਰਨ ne12,
                                      ਅਚਲ ਪੂਰਨ nbatch_fa) {
    ਸਵੈਚਾਲਿਤ          ਵਸਤੂ_ct1 = sycl::ext::oneapi::this_work_ਵਸਤੂ::get_nd_ਵਸਤੂ<3>();
    constexpr ਪੂਰਨ ncols    = ncols1 * ncols2;

    ਅਚਲ ਪੂਰਨ bidx0 = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2);
    ਅਚਲ ਪੂਰਨ j     = ਵਸਤੂ_ct1.get_ਸਮੂਹ(1);
    ਅਚਲ ਪੂਰਨ c     = ਵਸਤੂ_ct1.get_ਸਮੂਹ(0);
    ਅਚਲ ਪੂਰਨ jc    = j*ncols2 + c;
    ਅਚਲ ਪੂਰਨ tid   = ਵਸਤੂ_ct1.get_local_id(2);

    ਅਚਲ ਦਸ਼ਮਲਵ * dst_fixup_ਅੰਕੜੇ = ((ਅਚਲ ਦਸ਼ਮਲਵ *) dst_fixup) + ਵਸਤੂ_ct1.get_ਸਮੂਹ_range(2) * (2 * 2 * ncols);

    ਅਚਲ ਪੂਰਨ gqa_ਅਨੁਪਾਤ = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.

    ਅਚਲ ਪੂਰਨ iter_k     = (ne11      + (nbatch_fa - 1)) / nbatch_fa;
    ਅਚਲ ਪੂਰਨ iter_j     = (ne01      + (ncols1    - 1)) / ncols1;
    ਅਚਲ ਪੂਰਨ iter_z_gqa = (gqa_ਅਨੁਪਾਤ + (ncols2    - 1)) / ncols2;

    ਅਚਲ ਪੂਰਨ kbc0 = int64_t(bidx0 + 0) * (iter_k * iter_j * iter_z_gqa * ne12 * ne03) / ਵਸਤੂ_ct1.get_ਸਮੂਹ_range(2);
    ਅਚਲ ਪੂਰਨ kbc0_stop =
        int64_t(bidx0 + 1) * (iter_k * iter_j * iter_z_gqa * ne12 * ne03) / ਵਸਤੂ_ct1.get_ਸਮੂਹ_range(2);

    ਅਚਲ ਤਾਰਕਿਕ did_not_have_any_ਅੰਕੜੇ   = kbc0 == kbc0_stop;
    ਅਚਲ ਤਾਰਕਿਕ wrote_beginning_ਦਾ_tile = kbc0 % iter_k == 0;
    ਅਚਲ ਤਾਰਕਿਕ did_not_write_ਆਖਰੀ      = kbc0/iter_k == kbc0_stop/iter_k && kbc0_stop % iter_k != 0;
    ਜੇ (did_not_have_any_ਅੰਕੜੇ || wrote_beginning_ਦਾ_tile || did_not_write_ਆਖਰੀ) {
        ਵਾਪਸ;
    }

    // z_KV == K/V head index, zt_gqa = Q head start index per K/V head, jt = token position start index
    ਅਚਲ ਪੂਰਨ sequence =  kbc0 /(iter_k*iter_j*iter_z_gqa*ne12);
    ਅਚਲ ਪੂਰਨ z_KV     = (kbc0 - iter_k*iter_j*iter_z_gqa*ne12 * sequence)/(iter_k*iter_j*iter_z_gqa);
    ਅਚਲ ਪੂਰਨ zt_gqa   = (kbc0 - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV)/(iter_k*iter_j);
    ਅਚਲ ਪੂਰਨ jt       = (kbc0 - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV - iter_k*iter_j * zt_gqa) / iter_k;

    ਅਚਲ ਪੂਰਨ zt_Q = z_KV*gqa_ਅਨੁਪਾਤ + zt_gqa*ncols2; // Global Q head start index.

    ਜੇ (jt*ncols1 + j >= ne01 || zt_gqa*ncols2 + c >= gqa_ਅਨੁਪਾਤ) {
        ਵਾਪਸ;
    }

    dst += sequence*ne02*ne01*D + jt*ne02*(ncols1*D) + zt_Q*D + (j*ne02 + c)*D + tid;

    // Load the partial result that needs a fixup:
    ਦਸ਼ਮਲਵ dst_val = 0.0f;
    ਦਸ਼ਮਲਵ max_val = 0.0f;
    ਦਸ਼ਮਲਵ rowsum  = 0.0f;
    {
        dst_val = *dst;

        ਅਚਲ sycl::float2 tmp = dst_fixup[bidx0 * ncols + jc];
        max_val                = tmp.x();
        rowsum                 = tmp.y();
    }

    // Iterate over previous blocks and compute the combined results.
    // All SYCL blocks that get here must have a previous block that needs a fixup.
    ਪੂਰਨ bidx = bidx0 - 1;
    ਪੂਰਨ kbc_stop = kbc0;
    ਜਦੋਂ(ਸੱਚ) {
        ਅਚਲ ਪੂਰਨ kbc = int64_t(bidx) * (iter_k * iter_j * iter_z_gqa * ne12 * ne03) / ਵਸਤੂ_ct1.get_ਸਮੂਹ_range(2);
        ਜੇ (kbc == kbc_stop) { // Did not have any data.
            bidx--;
            kbc_stop = kbc;
            ਜਾਰੀ;
        }

        ਅਚਲ ਦਸ਼ਮਲਵ dst_ਜੋੜੋ = dst_fixup_ਅੰਕੜੇ[bidx*ncols*D + jc*D + tid];

        ਅਚਲ sycl::float2 tmp = dst_fixup[(ਵਸਤੂ_ct1.get_ਸਮੂਹ_range(2) + bidx) * ncols + jc];

        // Scale the current and new value accumulators depending on the max. values.
        ਅਚਲ ਦਸ਼ਮਲਵ max_val_new = sycl::fmax(max_val, tmp.x());

        ਅਚਲ ਦਸ਼ਮਲਵ diff_val = max_val - max_val_new;
        ਅਚਲ ਦਸ਼ਮਲਵ diff_ਜੋੜੋ = tmp.x() - max_val_new;

        ਅਚਲ ਦਸ਼ਮਲਵ scale_val = diff_val >= SOFTMAX_FTZ_ਹੱਦ ? sycl::native::ਘਾਤਾਂਕ(diff_val) : 0.0f;
        ਅਚਲ ਦਸ਼ਮਲਵ scale_ਜੋੜੋ = diff_ਜੋੜੋ >= SOFTMAX_FTZ_ਹੱਦ ? sycl::native::ਘਾਤਾਂਕ(diff_ਜੋੜੋ) : 0.0f;

        dst_val = scale_val*dst_val + scale_ਜੋੜੋ*dst_ਜੋੜੋ;
        rowsum  = scale_val * rowsum + scale_ਜੋੜੋ * tmp.y();

        max_val = max_val_new;

        // If this block started in a previous tile we are done and don't need to combine additional partial results.
        ਜੇ (kbc % iter_k == 0 || kbc/iter_k < kbc0/iter_k) {
            ਰੋਕੋ;
        }
        bidx--;
        kbc_stop = kbc;
    }

    // Write back final result:
    *dst = dst_val / rowsum;
}

ਸਾਂਚਾ <ਪੂਰਨ D>  // D == head size

ਸਥਿਰ ਖਾਲੀ flash_attn_combine_results(ਅਚਲ ਦਸ਼ਮਲਵ * __restrict__ VKQ_parts,
                                       ਅਚਲ sycl::float2 * __restrict__ VKQ_meta,
                                       ਦਸ਼ਮਲਵ * __restrict__ dst,
                                       ਅਚਲ ਪੂਰਨ parallel_ਖੰਡਾਂ,
                                       uint8_t * dpct_local) {
    // Dimension 0: threadIdx.x
    // Dimension 1: blockIdx.x
    // Dimension 2: blockIdx.y
    // Dimension 3: blockIdx.z
    // Memory layout is permuted with [0, 2, 1, 3]

    ਸਵੈਚਾਲਿਤ      ਵਸਤੂ_ct1 = sycl::ext::oneapi::this_work_ਵਸਤੂ::get_nd_ਵਸਤੂ<3>();
    ਅਚਲ ਪੂਰਨ ne01     = ਵਸਤੂ_ct1.get_ਸਮੂਹ_range(2);
    ਅਚਲ ਪੂਰਨ ne02     = ਵਸਤੂ_ct1.get_ਸਮੂਹ_range(1);

    ਅਚਲ ਪੂਰਨ ਖਾਨਾ      = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2);
    ਅਚਲ ਪੂਰਨ ਸਿਰ     = ਵਸਤੂ_ct1.get_ਸਮੂਹ(1);
    ਅਚਲ ਪੂਰਨ sequence = ਵਸਤੂ_ct1.get_ਸਮੂਹ(0);

    ਅਚਲ ਪੂਰਨ j_dst_unrolled = (sequence*ne01 + ਖਾਨਾ)*ne02 + ਸਿਰ;

    VKQ_parts += j_dst_unrolled * parallel_ਖੰਡਾਂ*D;
    VKQ_meta  += j_dst_unrolled * parallel_ਖੰਡਾਂ;
    dst       += j_dst_unrolled *                 D;

    ਅਚਲ ਪੂਰਨ tid = ਵਸਤੂ_ct1.get_local_id(2);
    __builtin_assume(tid < D);

    ਸਵੈਚਾਲਿਤ meta = (sycl::float2 *) dpct_local;
    ਲਈ (ਪੂਰਨ i = tid; i < 2*parallel_ਖੰਡਾਂ; i += D) {
        ((ਦਸ਼ਮਲਵ *) meta)[i] = ((ਅਚਲ ਦਸ਼ਮਲਵ *)VKQ_meta) [i];
    }

    ਵਸਤੂ_ct1.barrier(sycl::access::fence_space::local_space);

    ਦਸ਼ਮਲਵ kqmax = meta[0].x();
    ਲਈ (ਪੂਰਨ l = 1; l < parallel_ਖੰਡਾਂ; ++l) {
        kqmax = sycl::max(kqmax, meta[l].x());
    }

    ਦਸ਼ਮਲਵ VKQ_numerator   = 0.0f;
    ਦਸ਼ਮਲਵ VKQ_denominator = 0.0f;
    ਲਈ (ਪੂਰਨ l = 0; l < parallel_ਖੰਡਾਂ; ++l) {
        ਅਚਲ ਦਸ਼ਮਲਵ KQ_max_scale = sycl::native::ਘਾਤਾਂਕ(meta[l].x() - kqmax);

        VKQ_numerator   += KQ_max_scale * VKQ_parts[l*D + tid];
        VKQ_denominator += KQ_max_scale * meta[l].y();
    }

    dst[tid] = VKQ_numerator / VKQ_denominator;
}

// Carries the sub-group size (and grf_size_automatic when use_large_grf) as
// kernel properties via get(properties_tag), in place of the
// reqd_sub_group_size attribute and the deprecated parallel_for overload.
ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ KernelFunc, uint32_t ਮਾਪ_3, ਤਾਰਕਿਕ UseGrf>
ਬਣਤਰ props_kernel {
    KernelFunc kernel;

    ਖਾਲੀ ਸੰਚਾਲਕ()(sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ) ਅਚਲ {
        kernel(ਵਸਤੂ);
    }

    ਸਵੈਚਾਲਿਤ get(sycl::ext::oneapi::experimental::properties_ਟੈਗ) ਅਚਲ {
        ਜੇ constexpr (UseGrf) {
            ਵਾਪਸ sycl::ext::oneapi::experimental::properties{
                sycl::ext::intel::experimental::grf_ਮਾਪ_automatic,
                sycl::ext::oneapi::experimental::sub_ਸਮੂਹ_ਮਾਪ<ਮਾਪ_3> };
        } ਨਹੀਂਤਾਂ {
            ਵਾਪਸ sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::sub_ਸਮੂਹ_ਮਾਪ<ਮਾਪ_3> };
        }
    }
};

ਸਾਂਚਾ <fattn_kernel_t fattn_kernel, ਪੂਰਨ warp_ਮਾਪ, ਤਾਰਕਿਕ use_large_grf = ਝੂਠ>
ਸਥਿਰ ਖਾਲੀ lauch_kernel(
    dpct::dim3 ਸਮੂਹ_range,
    dpct::dim3 local_range,
    ਕਤਾਰ_ptr q,
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ local_mem_ਮਾਪ_2,
    ਅਚਲ ਅੱਖਰ* __restrict__ Q,
    ਅਚਲ ਅੱਖਰ* __restrict__ K,
    ਅਚਲ ਅੱਖਰ* __restrict__ V,
    ਅਚਲ ਅੱਖਰ* __restrict__ ਨਕਾਬ,
    ਅਚਲ ਅੱਖਰ* __restrict__ sinks,
    ਅਚਲ ਪੂਰਨ* __restrict__ KV_max,
    ਦਸ਼ਮਲਵ* __restrict__ dst,
    sycl::float2* __restrict__ dst_meta,
    ਅਚਲ ਦਸ਼ਮਲਵ scale,
    ਅਚਲ ਦਸ਼ਮਲਵ max_bias,
    ਅਚਲ ਦਸ਼ਮਲਵ m0,
    ਅਚਲ ਦਸ਼ਮਲਵ m1,
    ਅਚਲ uint32_t n_ਸਿਰ_log2,
    ਅਚਲ ਦਸ਼ਮਲਵ logit_softcap,
    ਅਚਲ int32_t ne00,
    ਅਚਲ sycl::uint3 ne01,
    ਅਚਲ int32_t ne02,
    ਅਚਲ int32_t ne03,
    ਅਚਲ int32_t nb01,
    ਅਚਲ int32_t nb02,
    ਅਚਲ int32_t nb03,
    ਅਚਲ int32_t ne10,
    ਅਚਲ int32_t ne11,
    ਅਚਲ int32_t ne12,
    ਅਚਲ int32_t ne13,
    ਅਚਲ int32_t nb11,
    ਅਚਲ int32_t nb12,
    ਅਚਲ int64_t nb13,
    ਅਚਲ int32_t nb21,
    ਅਚਲ int32_t nb22,
    ਅਚਲ int64_t nb23,
    ਅਚਲ int32_t ne31,
    ਅਚਲ int32_t ne32,
    ਅਚਲ int32_t ne33,
    ਅਚਲ int32_t nb31,
    ਅਚਲ int32_t nb32,
    ਅਚਲ int64_t nb33) {
    GGML_ਅਣਵਰਤਿਆ(local_mem_ਮਾਪ_2);

    ਅਚਲ ਸਵੈਚਾਲਿਤ rng = sycl::nd_range<3>(
        static_cast<sycl::range<3>>(ਸਮੂਹ_range * local_range),
        static_cast<sycl::range<3>>(local_range));

    ਅਚਲ ਸਵੈਚਾਲਿਤ kernel = [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
        GGML_ਅਣਵਰਤਿਆ(ਵਸਤੂ_ct1);
        fattn_kernel(Q, K, V, ਨਕਾਬ, sinks, KV_max, dst, dst_meta, scale,
                     max_bias, m0, m1, n_ਸਿਰ_log2, logit_softcap, ne00,
                     ne01, ne02, ne03, nb01, nb02, nb03, ne10, ne11,
                     ne12, ne13, nb11, nb12, nb13, nb21, nb22, nb23,
                     ne31, ne32, ne33, nb31, nb32, nb33);
    };

    q->submit([&](sycl::handler &cgh) {
        // grf_size_automatic lets the compiler use the large register file when the kernel needs it. grf_size<256> would say the same but is undefined on devices that do not have it.
        cgh.parallel_for(rng, props_kernel<decltype(kernel), warp_ਮਾਪ, use_large_grf>{kernel});
    });
}

// use_large_grf lets the compiler choose the large register file. Opt in where spilling limits the kernel, not thread occupancy.
ਸਾਂਚਾ <ਪੂਰਨ DV, ਪੂਰਨ ncols1, ਪੂਰਨ ncols2, fattn_kernel_t fattn_kernel, ਪੂਰਨ warp_ਮਾਪ, ਤਾਰਕਿਕ use_large_grf = ਝੂਠ>
ਖਾਲੀ launch_fattn(
    ggml_backend_sycl_context & ctx, ggml_tensor * dst, ਅਚਲ ਪੂਰਨ nwarps, ਅਚਲ size_t nbytes_shared,
    ਅਚਲ ਪੂਰਨ nbatch_fa, ਅਚਲ ਤਾਰਕਿਕ need_f16_K, ਅਚਲ ਤਾਰਕਿਕ need_f16_V, ਅਚਲ ਤਾਰਕਿਕ ਧਾਰਾ_k) {

    constexpr ਪੂਰਨ ncols = ncols1 * ncols2;

    ਅਚਲ ggml_tensor * Q = dst->src[0];
    ਅਚਲ ggml_tensor * K = dst->src[1];
    ਅਚਲ ggml_tensor * V = dst->src[2];

    ਅਚਲ ਤਾਰਕਿਕ V_is_K_view = V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));

    ਅਚਲ ggml_tensor * ਨਕਾਬ  = dst->src[3];
    ਅਚਲ ggml_tensor * sinks = dst->src[4];

    ggml_tensor * KQV = dst;

    GGML_ASSERT(Q->type == GGML_TYPE_F32);
    GGML_ASSERT(KQV->type == GGML_TYPE_F32);

    GGML_ASSERT(Q->nb[0] == ggml_element_ਮਾਪ(Q));
    GGML_ASSERT(K->nb[0] == ggml_element_ਮਾਪ(K));
    GGML_ASSERT(V->nb[0] == ggml_element_ਮਾਪ(V));

    GGML_ASSERT(!ਨਕਾਬ || ਨਕਾਬ->type == GGML_TYPE_F16);

    ggml_sycl_pool & pool = ctx.pool();
    dpct::ਕਤਾਰ_ptr  main_ਧਾਰਾ = ctx.stream();
    ਅਚਲ ਪੂਰਨ id  = ggml_sycl_get_device();
    ਅਚਲ ਪੂਰਨ nsm = ggml_sycl_info().devices[id].nsm;

    ਅਚਲ ggml_sycl_fattn_extra  extra = ggml_sycl_fattn_get_extra(dst);
    ggml_sycl_pool_alloc<ਪੂਰਨ>    KV_max(pool);
    ggml_sycl_pool_alloc<ਦਸ਼ਮਲਵ>  dst_tmp(pool);
    ggml_sycl_pool_alloc<sycl::float2> dst_tmp_meta(pool);

    ਅਚਲ ਅੱਖਰ * K_ਅੰਕੜੇ = (ਅਚਲ ਅੱਖਰ *) K->ਅੰਕੜੇ;
    size_t nb11 = K->nb[1];
    size_t nb12 = K->nb[2];
    size_t nb13 = K->nb[3];

    ਅਚਲ ਅੱਖਰ * V_ਅੰਕੜੇ = (ਅਚਲ ਅੱਖਰ *) V->ਅੰਕੜੇ;
    size_t nb21 = V->nb[1];
    size_t nb22 = V->nb[2];
    size_t nb23 = V->nb[3];

    ਜੇ (need_f16_K && K->type != GGML_TYPE_F16) {
        ਅਚਲ size_t bs = ggml_blck_ਮਾਪ(K->type);
        ਅਚਲ size_t ts = ggml_type_ਮਾਪ(K->type);

        GGML_ASSERT(extra.K_ਬਫਰ_ptr);
        sycl::half * K_f16_ptr = (sycl::half *) extra.K_ਬਫਰ_ptr;
        ਜੇ (ggml_is_contiguously_allocated(K)) {
            ਮੰਜ਼ਿਲ_fp16_sycl_t ਮੰਜ਼ਿਲ_fp16 = ggml_get_ਮੰਜ਼ਿਲ_fp16_sycl(K->type, dst);
            ਮੰਜ਼ਿਲ_fp16(K_ਅੰਕੜੇ, K_f16_ptr, ggml_nelements(K), main_ਧਾਰਾ);

            nb11 = nb11 * bs * ਆਕਾਰ(sycl::half) / ts;
            nb12 = nb12 * bs * ਆਕਾਰ(sycl::half) / ts;
            nb13 = nb13 * bs * ਆਕਾਰ(sycl::half) / ts;
        } ਨਹੀਂਤਾਂ {
            GGML_ASSERT(K->nb[0] == ts);
            ਮੰਜ਼ਿਲ_fp16_nc_sycl_t ਮੰਜ਼ਿਲ_fp16 = ggml_get_ਮੰਜ਼ਿਲ_fp16_nc_sycl(K->type);
            ਅਚਲ int64_t s01 = nb11 / ts;
            ਅਚਲ int64_t s02 = nb12 / ts;
            ਅਚਲ int64_t s03 = nb13 / ts;
            ਮੰਜ਼ਿਲ_fp16(K_ਅੰਕੜੇ, K_f16_ptr, K->ne[0], K->ne[1], K->ne[2], K->ne[3], s01, s02, s03, main_ਧਾਰਾ);

            nb11 = K->ne[0] * ਆਕਾਰ(sycl::half);
            nb12 = K->ne[1] * nb11;
            nb13 = K->ne[2] * nb12;
        }
        K_ਅੰਕੜੇ = (ਅੱਖਰ *) K_f16_ptr;
    }

    ਜੇ (need_f16_V && V->type != GGML_TYPE_F16) {
        ਜੇ (V_is_K_view) {
            V_ਅੰਕੜੇ = K_ਅੰਕੜੇ;
            nb21   = nb11;
            nb22   = nb12;
            nb23   = nb13;
        } ਨਹੀਂਤਾਂ {
            ਅਚਲ size_t bs = ggml_blck_ਮਾਪ(V->type);
            ਅਚਲ size_t ts = ggml_type_ਮਾਪ(V->type);

            GGML_ASSERT(extra.V_ਬਫਰ_ptr);
            sycl::half * V_f16_ptr = (sycl::half *) extra.V_ਬਫਰ_ptr;
            ਜੇ (ggml_is_contiguously_allocated(V)) {
                ਮੰਜ਼ਿਲ_fp16_sycl_t ਮੰਜ਼ਿਲ_fp16 = ggml_get_ਮੰਜ਼ਿਲ_fp16_sycl(V->type, dst);
                ਮੰਜ਼ਿਲ_fp16(V_ਅੰਕੜੇ, V_f16_ptr, ggml_nelements(V), main_ਧਾਰਾ);
                V_ਅੰਕੜੇ = (ਅੱਖਰ *) V_f16_ptr;

                nb21 = nb21 * bs * ਆਕਾਰ(sycl::half) / ts;
                nb22 = nb22 * bs * ਆਕਾਰ(sycl::half) / ts;
                nb23 = nb23 * bs * ਆਕਾਰ(sycl::half) / ts;
            } ਨਹੀਂਤਾਂ {
                GGML_ASSERT(V->nb[0] == ts);
                ਮੰਜ਼ਿਲ_fp16_nc_sycl_t ਮੰਜ਼ਿਲ_fp16 = ggml_get_ਮੰਜ਼ਿਲ_fp16_nc_sycl(V->type);
                ਅਚਲ int64_t s01 = nb21 / ts;
                ਅਚਲ int64_t s02 = nb22 / ts;
                ਅਚਲ int64_t s03 = nb23 / ts;
                ਮੰਜ਼ਿਲ_fp16(V_ਅੰਕੜੇ, V_f16_ptr, V->ne[0], V->ne[1], V->ne[2], V->ne[3], s01, s02, s03, main_ਧਾਰਾ);

                nb21 = V->ne[0] * ਆਕਾਰ(sycl::half);
                nb22 = V->ne[1] * nb21;
                nb23 = V->ne[2] * nb22;
            }
            V_ਅੰਕੜੇ = (ਅੱਖਰ *) V_f16_ptr;
        }
    }

    ਅਚਲ ਪੂਰਨ ntiles_x     = ((Q->ne[1] + ncols1 - 1) / ncols1);
    ਅਚਲ ਪੂਰਨ gqa_ਅਨੁਪਾਤ    = Q->ne[2] / K->ne[2];
    ਅਚਲ ਪੂਰਨ ntiles_z_gqa = ((gqa_ਅਨੁਪਾਤ + ncols2 - 1) / ncols2);
    ਅਚਲ ਪੂਰਨ ntiles_ਕੁੱਲ = ntiles_x * ntiles_z_gqa * K->ne[2] * Q->ne[3];

    // Optional optimization where the mask is scanned to determine whether part of the calculation can be skipped.
    // Only worth the overhead if there is at lease one FATTN_KQ_STRIDE x FATTN_KQ_STRIDE square to be skipped or
    //     multiple sequences of possibly different lengths.
    ਜੇ (ਨਕਾਬ && K->ne[1] % FATTN_KQ_STRIDE == 0 && (Q->ne[1] >= 1024 || Q->ne[3] > 1)) {
        ਅਚਲ ਪੂਰਨ s31 = ਨਕਾਬ->nb[1] / ਆਕਾਰ(sycl::half2);
        ਅਚਲ ਪੂਰਨ s33 = ਨਕਾਬ->nb[3] / ਆਕਾਰ(sycl::half2);

        ਅਚਲ dpct::dim3 ਖੰਡਾਂ_ਅੰਕ_KV_max(ntiles_x, Q->ne[3], 1);
        ਅਚਲ dpct::dim3 ਖੰਡ_dim_KV_max(FATTN_KQ_STRIDE / 2, 1, 1);

        ਅਚਲ ਪੂਰਨ ne_KV_max = ਖੰਡਾਂ_ਅੰਕ_KV_max.x*ਖੰਡਾਂ_ਅੰਕ_KV_max.y;
        ਅਚਲ ਪੂਰਨ iter_k = K->ne[1] / FATTN_KQ_STRIDE;

        KV_max.alloc(ne_KV_max);
        {
            dpct::has_capability_or_fail(main_ਧਾਰਾ->get_device(), { sycl::aspect::fp16 });

            main_ਧਾਰਾ->submit([&](sycl::handler & cgh) {
                sycl::local_accessor<ਪੂਰਨ, 1> buf_iw_acc_ct1(sycl::range<1>(warp_ਮਾਪ), cgh);

                ਸਵੈਚਾਲਿਤ ਨਕਾਬ_ਅੰਕੜੇ_ct0  = (ਅਚਲ sycl::half2 *) ਨਕਾਬ->ਅੰਕੜੇ;
                ਸਵੈਚਾਲਿਤ KV_max_ptr_ct1 = KV_max.ptr;

                cgh.parallel_for(sycl::nd_range<3>(ਖੰਡਾਂ_ਅੰਕ_KV_max * ਖੰਡ_dim_KV_max, ਖੰਡ_dim_KV_max),
                                 [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(warp_ਮਾਪ)]] {
                                     GGML_ਅਣਵਰਤਿਆ(ਵਸਤੂ_ct1);
                                     flash_attn_ਨਕਾਬ_ਮੰਜ਼ਿਲ_KV_max<ncols1, warp_ਮਾਪ>(
                                         ਨਕਾਬ_ਅੰਕੜੇ_ct0, KV_max_ptr_ct1, iter_k, s31, s33,
                                         buf_iw_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                                 });
            });
        }
        SYCL_CHECK(0);
    }

    ਅਚਲ dpct::dim3 ਖੰਡ_dim(warp_ਮਾਪ, nwarps, 1);

    // Max. number of active blocks limited by occupancy.
    ਪੂਰਨ max_ਖੰਡਾਂ_ਪ੍ਰਤੀ_sm = ggml_sycl_info().devices[id].max_wg_ਪ੍ਰਤੀ_cu;
    ਪੂਰਨ parallel_ਖੰਡਾਂ = max_ਖੰਡਾਂ_ਪ੍ਰਤੀ_sm;
    dpct::dim3 ਖੰਡਾਂ_ਅੰਕ;
    ਜੇ (ਧਾਰਾ_k) {
        // For short contexts it can be faster to have the SMs work on whole tiles because this lets us skip the fixup.
        ਅਚਲ ਪੂਰਨ max_ਖੰਡਾਂ = max_ਖੰਡਾਂ_ਪ੍ਰਤੀ_sm*nsm;
        ਅਚਲ ਪੂਰਨ nblocks_ਧਾਰਾ_k = max_ਖੰਡਾਂ;
        ਅਚਲ ਤਾਰਕਿਕ use_ਧਾਰਾ_k = ਸੱਚ;

        ਖੰਡਾਂ_ਅੰਕ.x = use_ਧਾਰਾ_k ? nblocks_ਧਾਰਾ_k : ntiles_ਕੁੱਲ;
        ਖੰਡਾਂ_ਅੰਕ.y = 1;
        ਖੰਡਾਂ_ਅੰਕ.z = 1;

        ਜੇ (ntiles_ਕੁੱਲ % ਖੰਡਾਂ_ਅੰਕ.x != 0) { // Fixup is only needed if the SMs work on fractional tiles.
            dst_tmp_meta.alloc((size_t(ਖੰਡਾਂ_ਅੰਕ.x) * ncols * (2 + DV/2)));
        }
    } ਨਹੀਂਤਾਂ {
        ਅਚਲ ਪੂਰਨ ntiles_KQ = (K->ne[1] + nbatch_fa - 1) / nbatch_fa; // Max. number of parallel blocks limited by tensor size.

        // parallel_blocks must not be larger than what the tensor size allows:
        parallel_ਖੰਡਾਂ = std::min(parallel_ਖੰਡਾਂ, ntiles_KQ);
        // todo fix the hard code change
        // parallel_blocks = ntiles_KQ;

        // If ntiles_total % blocks_per_wave != 0 then some efficiency is lost due to tail effects.
        // Test whether parallel_blocks can be set to a higher value for better efficiency.
        ਅਚਲ ਪੂਰਨ ਖੰਡਾਂ_ਪ੍ਰਤੀ_wave = nsm * max_ਖੰਡਾਂ_ਪ੍ਰਤੀ_sm;
        ਪੂਰਨ nwaves_ਉੱਤਮ = 0;
        ਪੂਰਨ efficiency_percent_ਉੱਤਮ = 0;
        ਲਈ (ਪੂਰਨ parallel_ਖੰਡਾਂ_test = parallel_ਖੰਡਾਂ; parallel_ਖੰਡਾਂ_test <= ntiles_KQ; ++parallel_ਖੰਡਾਂ_test) {
            ਅਚਲ ਪੂਰਨ nblocks_ਕੁੱਲ = ntiles_ਕੁੱਲ * parallel_ਖੰਡਾਂ_test;
            ਅਚਲ ਪੂਰਨ nwaves = (nblocks_ਕੁੱਲ + ਖੰਡਾਂ_ਪ੍ਰਤੀ_wave - 1) / ਖੰਡਾਂ_ਪ੍ਰਤੀ_wave;
            ਅਚਲ ਪੂਰਨ efficiency_percent = 100 * nblocks_ਕੁੱਲ / (nwaves*ਖੰਡਾਂ_ਪ੍ਰਤੀ_wave);

            // Stop trying configurations with more waves if we already have good efficiency to avoid excessive overhead.
            ਜੇ (efficiency_percent_ਉੱਤਮ >= 95 && nwaves > nwaves_ਉੱਤਮ) {
                ਰੋਕੋ;
            }

            ਜੇ (efficiency_percent > efficiency_percent_ਉੱਤਮ) {
                nwaves_ਉੱਤਮ = nwaves;
                efficiency_percent_ਉੱਤਮ = efficiency_percent;
                parallel_ਖੰਡਾਂ = parallel_ਖੰਡਾਂ_test;
            }
        }

        ਖੰਡਾਂ_ਅੰਕ.x = ntiles_x;
        ਖੰਡਾਂ_ਅੰਕ.y = parallel_ਖੰਡਾਂ;
        ਖੰਡਾਂ_ਅੰਕ.z = ntiles_z_gqa*K->ne[2]*Q->ne[3];

        ਜੇ (parallel_ਖੰਡਾਂ > 1) {
            dst_tmp.alloc(parallel_ਖੰਡਾਂ*ggml_nelements(KQV));
            dst_tmp_meta.alloc(parallel_ਖੰਡਾਂ*ggml_nrows(KQV));
        }
    }

    ਦਸ਼ਮਲਵ scale         = 1.0f;
    ਦਸ਼ਮਲਵ max_bias      = 0.0f;
    ਦਸ਼ਮਲਵ logit_softcap = 0.0f;

    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&scale,         (ਅਚਲ ਦਸ਼ਮਲਵ *) KQV->op_params + 0, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&max_bias,      (ਅਚਲ ਦਸ਼ਮਲਵ *) KQV->op_params + 1, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&logit_softcap, (ਅਚਲ ਦਸ਼ਮਲਵ *) KQV->op_params + 2, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਜੇ (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    ਅਚਲ uint32_t n_ਸਿਰ      = Q->ne[2];
    ਅਚਲ uint32_t n_ਸਿਰ_log2 = 1u << uint32_t(floorf(log2f(ਦਸ਼ਮਲਵ(n_ਸਿਰ))));

    ਅਚਲ ਦਸ਼ਮਲਵ m0 = powf(2.0f, -(max_bias       ) / n_ਸਿਰ_log2);
    ਅਚਲ ਦਸ਼ਮਲਵ m1 = powf(2.0f, -(max_bias / 2.0f) / n_ਸਿਰ_log2);

    // TODO other tensor dimensions after removal of WMMA kernel:
    ਅਚਲ sycl::uint3 ne01 = init_fastdiv_ਮੁੱਲਾਂ(Q->ne[1]);

    GGML_ASSERT(ਖੰਡ_dim.x % warp_ਮਾਪ == 0);

    lauch_kernel<fattn_kernel, warp_ਮਾਪ, use_large_grf>(
        ਖੰਡਾਂ_ਅੰਕ, ਖੰਡ_dim, main_ਧਾਰਾ, (ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ) nbytes_shared, (ਅਚਲ ਅੱਖਰ *) Q->ਅੰਕੜੇ, K_ਅੰਕੜੇ, V_ਅੰਕੜੇ,
        ਨਕਾਬ ? ((ਅਚਲ ਅੱਖਰ *) ਨਕਾਬ->ਅੰਕੜੇ) : ਖਾਲੀ_ਸੂਚਕ, sinks ? ((ਅਚਲ ਅੱਖਰ *) sinks->ਅੰਕੜੇ) : ਖਾਲੀ_ਸੂਚਕ, KV_max.ptr,
        !ਧਾਰਾ_k && parallel_ਖੰਡਾਂ > 1 ? dst_tmp.ptr : (ਦਸ਼ਮਲਵ *) KQV->ਅੰਕੜੇ, (sycl::float2 *)dst_tmp_meta.ptr, scale, max_bias, m0, m1,
        n_ਸਿਰ_log2, logit_softcap, Q->ne[0], ne01, Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3], K->ne[0],
        K->ne[1], K->ne[2], K->ne[3], nb11, nb12, nb13, nb21, nb22, nb23, ਨਕਾਬ ? ਨਕਾਬ->ne[1] : 0,
        ਨਕਾਬ ? ਨਕਾਬ->ne[2] : 0, ਨਕਾਬ ? ਨਕਾਬ->ne[3] : 0, ਨਕਾਬ ? ਨਕਾਬ->nb[1] : 0, ਨਕਾਬ ? ਨਕਾਬ->nb[2] : 0,
        ਨਕਾਬ ? ਨਕਾਬ->nb[3] : 0);
    SYCL_CHECK(0);

    ਜੇ (ਧਾਰਾ_k) {
        ਜੇ (ntiles_ਕੁੱਲ % ਖੰਡਾਂ_ਅੰਕ.x != 0) { // Fixup is only needed if the SMs work on fractional tiles.
            ਅਚਲ dpct::dim3 ਖੰਡ_dim_combine(DV, 1, 1);
            ਅਚਲ dpct::dim3 ਖੰਡਾਂ_ਅੰਕ_combine = { ਖੰਡਾਂ_ਅੰਕ.x, ncols1, ncols2 };

            main_ਧਾਰਾ->submit([&](sycl::handler & cgh) {
                ਸਵੈਚਾਲਿਤ KQV_ਅੰਕੜੇ_ct0         = (ਦਸ਼ਮਲਵ *) KQV->ਅੰਕੜੇ;
                ਸਵੈਚਾਲਿਤ dst_tmp_meta_ptr_ct1 = dst_tmp_meta.ptr;
                ਸਵੈਚਾਲਿਤ Q_ne_ct2             = Q->ne[1];
                ਸਵੈਚਾਲਿਤ Q_ne_ct3             = Q->ne[2];
                ਸਵੈਚਾਲਿਤ Q_ne_ct4             = Q->ne[3];
                ਸਵੈਚਾਲਿਤ K_ne_ct5             = K->ne[1];
                ਸਵੈਚਾਲਿਤ K_ne_ct6             = K->ne[2];

                cgh.parallel_for(sycl::nd_range<3>(ਖੰਡਾਂ_ਅੰਕ_combine * ਖੰਡ_dim_combine, ਖੰਡ_dim_combine),
                                 [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(warp_ਮਾਪ)]] {
                                     GGML_ਅਣਵਰਤਿਆ(ਵਸਤੂ_ct1);
                                     flash_attn_ਧਾਰਾ_k_fixup<DV, ncols1, ncols2>(KQV_ਅੰਕੜੇ_ct0, dst_tmp_meta_ptr_ct1,
                                                                                   Q_ne_ct2, Q_ne_ct3, Q_ne_ct4,
                                                                                   K_ne_ct5, K_ne_ct6, nbatch_fa);
                                 });
            });
        }
    } ਨਹੀਂਤਾਂ ਜੇ (parallel_ਖੰਡਾਂ > 1) {
        ਅਚਲ dpct::dim3 ਖੰਡ_dim_combine(DV, 1, 1);
        ਅਚਲ dpct::dim3 ਖੰਡਾਂ_ਅੰਕ_combine(Q->ne[1], Q->ne[2], Q->ne[3]);
        ਅਚਲ size_t     nbytes_shared_combine = parallel_ਖੰਡਾਂ * ਆਕਾਰ(sycl::float2);
        main_ਧਾਰਾ->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(sycl::range<1>(nbytes_shared_combine), cgh);

            ਸਵੈਚਾਲਿਤ dst_tmp_ptr_ct0      = dst_tmp.ptr;
            ਸਵੈਚਾਲਿਤ dst_tmp_meta_ptr_ct1 = dst_tmp_meta.ptr;
            ਸਵੈਚਾਲਿਤ KQV_ਅੰਕੜੇ_ct2         = (ਦਸ਼ਮਲਵ *) KQV->ਅੰਕੜੇ;

            cgh.parallel_for(sycl::nd_range<3>(ਖੰਡਾਂ_ਅੰਕ_combine * ਖੰਡ_dim_combine, ਖੰਡ_dim_combine),
                             [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(warp_ਮਾਪ)]] {
                                 GGML_ਅਣਵਰਤਿਆ(ਵਸਤੂ_ct1);
                                 flash_attn_combine_results<DV>(
                                     dst_tmp_ptr_ct0, dst_tmp_meta_ptr_ct1, KQV_ਅੰਕੜੇ_ct2, parallel_ਖੰਡਾਂ,
                                     dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                             });
        });
    }
    SYCL_CHECK(0);
}
