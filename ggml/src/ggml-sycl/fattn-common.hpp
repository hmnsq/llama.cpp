#pragma once

#അന്തര്ഭാവ <sycl/sycl.hpp>
#അന്തര്ഭാവ <sycl/ext/intel/experimental/grf_size_properties.hpp>
#അന്തര്ഭാവ "dpct/helper.hpp"
#അന്തര്ഭാവ "common.hpp"
#അന്തര്ഭാവ "convert.hpp"
#അന്തര്ഭാവ "vecdotq.hpp"
#അന്തര്ഭാവ "fattn.hpp"

#അന്തര്ഭാവ "ggml.h"

#അന്തര്ഭാവ <cstdint>
#അന്തര്ഭാവ <cmath>
#അന്തര്ഭാവ <float.h>


#define FATTN_KQ_STRIDE       256
#define HALF_MAX_HALF         sycl::half(65504.0f/2) // Use neg. of this instead of -INFINITY to initialize KQ max vals to avoid NaN upon subtraction.
#define SOFTMAX_FTZ_THRESHOLD -20.0f                   // Softmax exp. of values smaller than this are flushed to zero to avoid NaNs.
#define FATTN_KQ_MAX_OFFSET (3.0f*0.6931f)

തരനിർവചനം ശൂന്യം (*fattn_kernel_t)(
    സ്ഥിരം അക്ഷരം* Q,
    സ്ഥിരം അക്ഷരം* K,
    സ്ഥിരം അക്ഷരം* V,
    സ്ഥിരം അക്ഷരം* mask,
    സ്ഥിരം അക്ഷരം* sinks,
    സ്ഥിരം പൂർണ്ണസംഖ്യ* KV_max,
    ദശാംശം* dst,
    sycl::float2* dst_meta,
    സ്ഥിരം ദശാംശം scale,
    സ്ഥിരം ദശാംശം max_bias,
    സ്ഥിരം ദശാംശം m0,
    സ്ഥിരം ദശാംശം m1,
    സ്ഥിരം uint32_t n_head_log2,
    സ്ഥിരം ദശാംശം logit_softcap,
    സ്ഥിരം int32_t ne00,
    സ്ഥിരം sycl::uint3 ne01,
    സ്ഥിരം int32_t ne02,
    സ്ഥിരം int32_t ne03,
    സ്ഥിരം int32_t nb01,
    സ്ഥിരം int32_t nb02,
    സ്ഥിരം int32_t nb03,
    സ്ഥിരം int32_t ne10,
    സ്ഥിരം int32_t ne11,
    സ്ഥിരം int32_t ne12,
    സ്ഥിരം int32_t ne13,
    സ്ഥിരം int32_t nb11,
    സ്ഥിരം int32_t nb12,
    സ്ഥിരം int64_t nb13,
    സ്ഥിരം int32_t nb21,
    സ്ഥിരം int32_t nb22,
    സ്ഥിരം int64_t nb23,
    സ്ഥിരം int32_t ne31,
    സ്ഥിരം int32_t ne32,
    സ്ഥിരം int32_t ne33,
    സ്ഥിരം int32_t nb31,
    സ്ഥിരം int32_t nb32,
    സ്ഥിരം int64_t nb33);

തരനിർവചനം ദശാംശം (*vec_dot_KQ_t)(
    സ്ഥിരം അക്ഷരം * __restrict__ K_c, സ്ഥിരം ശൂന്യം * __restrict__ Q_v, സ്ഥിരം പൂർണ്ണസംഖ്യ * __restrict__ Q_q8 , സ്ഥിരം ശൂന്യം * __restrict__ Q_ds);

template <പൂർണ്ണസംഖ്യ D, പൂർണ്ണസംഖ്യ nthreads>
സ്ഥിതം __dpct_inline__ ദശാംശം vec_dot_fattn_vec_KQ_f16(സ്ഥിരം അക്ഷരം * __restrict__ K_c,
                                                      സ്ഥിരം ശൂന്യം * __restrict__ Q_v,
                                                      സ്ഥിരം പൂർണ്ണസംഖ്യ * __restrict__ Q_q8,
                                                      സ്ഥിരം ശൂന്യം * __restrict__ Q_ds_v) {
    സ്ഥിരം sycl::half2 * K_h2 = (സ്ഥിരം sycl::half2 *) K_c;
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v);

    constexpr പൂർണ്ണസംഖ്യ cpy_nb = ggml_sycl_get_max_cpy_bytes();
    constexpr പൂർണ്ണസംഖ്യ cpy_ne = cpy_nb / 4;

    ദശാംശം sum = 0.0f;

#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads*cpy_ne) {
        sycl::half2 tmp[cpy_ne];
        ggml_sycl_memcpy_1<വലുപ്പം(tmp)>(
            tmp,
            K_h2 + k_KQ_0 + (sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2) % nthreads) * cpy_ne);
#pragma unroll
        വേണ്ടി (പൂർണ്ണസംഖ്യ k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
#ifdef GGML_SYCL_F16
            ggml_sycl_mad(sum,                tmp[k_KQ_1] , ((സ്ഥിരം sycl::half2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#അല്ലെങ്കിൽ
            ggml_sycl_mad(sum, __half22float2(tmp[k_KQ_1]), ((സ്ഥിരം sycl::float2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#endif // GGML_SYCL_F16
        }
    }

    തിരികെ sum;
}

template <പൂർണ്ണസംഖ്യ D, പൂർണ്ണസംഖ്യ nthreads, പൂർണ്ണസംഖ്യ warp_size>
സ്ഥിതം __dpct_inline__ ദശാംശം vec_dot_fattn_vec_KQ_q4_0(സ്ഥിരം അക്ഷരം * __restrict__ K_c,
                                                       സ്ഥിരം ശൂന്യം * __restrict__ Q_v,
                                                       സ്ഥിരം പൂർണ്ണസംഖ്യ * __restrict__ Q_q8,
                                                       സ്ഥിരം ശൂന്യം * __restrict__ Q_ds_v) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    സ്ഥിരം block_q4_0 * K_q4_0   = (സ്ഥിരം block_q4_0 *) K_c;
    GGML_UNUSED(Q_v);

    ദശാംശം sum = 0.0f;

#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ k_KQ_0 = 0; k_KQ_0 < പൂർണ്ണസംഖ്യ(D/വലുപ്പം(പൂർണ്ണസംഖ്യ)); k_KQ_0 += nthreads) {
        സ്ഥിരം പൂർണ്ണസംഖ്യ k_KQ =
            k_KQ_0 + (nthreads == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads);

        സ്ഥിരം പൂർണ്ണസംഖ്യ ib    = k_KQ /  QI8_1;
        സ്ഥിരം പൂർണ്ണസംഖ്യ iqs4  = k_KQ %  QI4_0;
        സ്ഥിരം പൂർണ്ണസംഖ്യ shift = k_KQ & (QI8_1/2);

        പൂർണ്ണസംഖ്യ v;
        ggml_sycl_memcpy_1<വലുപ്പം(പൂർണ്ണസംഖ്യ), 2>(&v, K_q4_0[ib].qs + വലുപ്പം(പൂർണ്ണസംഖ്യ)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;
        സ്ഥിരം പൂർണ്ണസംഖ്യ u = Q_q8[k_KQ_0/nthreads];

        സ്ഥിരം പൂർണ്ണസംഖ്യ sumi = ggml_sycl_dp4a(v, u, 0);

        സ്ഥിരം sycl::float2 Q_ds = ((സ്ഥിരം sycl::float2 *) Q_ds_v)[k_KQ_0 / nthreads];
        sum += __half2float(K_q4_0[ib].d) * (sumi*Q_ds.x() - (8/QI8_1)*Q_ds.y());
    }

    തിരികെ sum;
}

template <പൂർണ്ണസംഖ്യ D, പൂർണ്ണസംഖ്യ nthreads , പൂർണ്ണസംഖ്യ warp_size>
സ്ഥിതം __dpct_inline__ ദശാംശം vec_dot_fattn_vec_KQ_q4_1(സ്ഥിരം അക്ഷരം * __restrict__ K_c,
                                                       സ്ഥിരം ശൂന്യം * __restrict__ Q_v,
                                                       സ്ഥിരം പൂർണ്ണസംഖ്യ * __restrict__ Q_q8,
                                                       സ്ഥിരം ശൂന്യം * __restrict__ Q_ds_v) {
    auto               item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    സ്ഥിരം block_q4_1 * K_q4_1   = (സ്ഥിരം block_q4_1 *) K_c;
    GGML_UNUSED(Q_v);

    ദശാംശം sum = 0.0f;

#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ k_KQ_0 = 0; k_KQ_0 < പൂർണ്ണസംഖ്യ(D/വലുപ്പം(പൂർണ്ണസംഖ്യ)); k_KQ_0 += nthreads) {
        സ്ഥിരം പൂർണ്ണസംഖ്യ k_KQ =
            k_KQ_0 + (nthreads == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads);

        സ്ഥിരം പൂർണ്ണസംഖ്യ ib    = k_KQ /  QI8_1;
        സ്ഥിരം പൂർണ്ണസംഖ്യ iqs4  = k_KQ %  QI4_1;
        സ്ഥിരം പൂർണ്ണസംഖ്യ shift = k_KQ & (QI8_1/2);

        പൂർണ്ണസംഖ്യ v;
        ggml_sycl_memcpy_1<വലുപ്പം(പൂർണ്ണസംഖ്യ)>(&v, K_q4_1[ib].qs + വലുപ്പം(പൂർണ്ണസംഖ്യ)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;
        സ്ഥിരം പൂർണ്ണസംഖ്യ u = Q_q8[k_KQ_0/nthreads];

        സ്ഥിരം പൂർണ്ണസംഖ്യ sumi = ggml_sycl_dp4a(v, u, 0);

        സ്ഥിരം sycl::float2 K_dm = (K_q4_1[ib].dm).template convert<ദശാംശം, sycl::rounding_mode::automatic>();
        സ്ഥിരം sycl::float2 Q_ds = ((സ്ഥിരം sycl::float2 *) Q_ds_v)[k_KQ_0 / nthreads];

        sum += K_dm.x()*Q_ds.x()*sumi + K_dm.y()*Q_ds.y()/QI8_1;
    }

    തിരികെ sum;
}

template <പൂർണ്ണസംഖ്യ D, പൂർണ്ണസംഖ്യ nthreads, പൂർണ്ണസംഖ്യ warp_size>
സ്ഥിതം __dpct_inline__ ദശാംശം vec_dot_fattn_vec_KQ_q5_0(സ്ഥിരം അക്ഷരം * __restrict__ K_c,
                                                       സ്ഥിരം ശൂന്യം * __restrict__ Q_v,
                                                       സ്ഥിരം പൂർണ്ണസംഖ്യ * __restrict__ Q_q8,
                                                       സ്ഥിരം ശൂന്യം * __restrict__ Q_ds_v) {
    auto               item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    സ്ഥിരം block_q5_0 * K_q5_0   = (സ്ഥിരം block_q5_0 *) K_c;
    GGML_UNUSED(Q_v);

    ദശാംശം sum = 0.0f;

#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ k_KQ_0 = 0; k_KQ_0 < പൂർണ്ണസംഖ്യ(D/വലുപ്പം(പൂർണ്ണസംഖ്യ)); k_KQ_0 += nthreads) {
        സ്ഥിരം പൂർണ്ണസംഖ്യ k_KQ =
            k_KQ_0 + (nthreads == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads);

        സ്ഥിരം പൂർണ്ണസംഖ്യ ib    = k_KQ /  QI8_1;
        സ്ഥിരം പൂർണ്ണസംഖ്യ iqs4  = k_KQ %  QI5_0;
        സ്ഥിരം പൂർണ്ണസംഖ്യ iqs8  = k_KQ %  QI8_1;
        സ്ഥിരം പൂർണ്ണസംഖ്യ shift = k_KQ & (QI8_1/2);

        പൂർണ്ണസംഖ്യ v;
        ggml_sycl_memcpy_1<വലുപ്പം(പൂർണ്ണസംഖ്യ), 2>(&v, K_q5_0[ib].qs + വലുപ്പം(പൂർണ്ണസംഖ്യ)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;

        {
            പൂർണ്ണസംഖ്യ vh;
            ggml_sycl_memcpy_1<വലുപ്പം(പൂർണ്ണസംഖ്യ), 2>(&vh, K_q5_0[ib].qh);
            vh >>= iqs8 * QI5_0;

            v |= (vh <<  4) & 0x00000010; // 0 ->  4
            v |= (vh << 11) & 0x00001000; // 1 -> 12
            v |= (vh << 18) & 0x00100000; // 2 -> 20
            v |= (vh << 25) & 0x10000000; // 3 -> 28
        }

        സ്ഥിരം പൂർണ്ണസംഖ്യ u = Q_q8[k_KQ_0/nthreads];

        സ്ഥിരം പൂർണ്ണസംഖ്യ sumi = ggml_sycl_dp4a(v, u, 0);

        സ്ഥിരം sycl::float2 Q_ds = ((സ്ഥിരം sycl::float2 *) Q_ds_v)[k_KQ_0 / nthreads];

        sum += __half2float(K_q5_0[ib].d) * (sumi*Q_ds.x() - (16/QI8_1)*Q_ds.y());
    }

    തിരികെ sum;
}

template <പൂർണ്ണസംഖ്യ D, പൂർണ്ണസംഖ്യ nthreads, പൂർണ്ണസംഖ്യ warp_size>
സ്ഥിതം __dpct_inline__ ദശാംശം vec_dot_fattn_vec_KQ_q5_1(സ്ഥിരം അക്ഷരം * __restrict__ K_c,
                                                       സ്ഥിരം ശൂന്യം * __restrict__ Q_v,
                                                       സ്ഥിരം പൂർണ്ണസംഖ്യ * __restrict__ Q_q8,
                                                       സ്ഥിരം ശൂന്യം * __restrict__ Q_ds_v) {
    auto               item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    സ്ഥിരം block_q5_1 * K_q5_1   = (സ്ഥിരം block_q5_1 *) K_c;
    GGML_UNUSED(Q_v);

    ദശാംശം sum = 0.0f;

#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ k_KQ_0 = 0; k_KQ_0 < പൂർണ്ണസംഖ്യ(D/വലുപ്പം(പൂർണ്ണസംഖ്യ)); k_KQ_0 += nthreads) {
        സ്ഥിരം പൂർണ്ണസംഖ്യ k_KQ =
            k_KQ_0 + (nthreads == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads);

        സ്ഥിരം പൂർണ്ണസംഖ്യ ib    = k_KQ /  QI8_1;
        സ്ഥിരം പൂർണ്ണസംഖ്യ iqs4  = k_KQ %  QI5_1;
        സ്ഥിരം പൂർണ്ണസംഖ്യ iqs8  = k_KQ %  QI8_1;
        സ്ഥിരം പൂർണ്ണസംഖ്യ shift = k_KQ & (QI8_1/2);

        പൂർണ്ണസംഖ്യ v;
        ggml_sycl_memcpy_1<വലുപ്പം(പൂർണ്ണസംഖ്യ)>(&v, K_q5_1[ib].qs + വലുപ്പം(പൂർണ്ണസംഖ്യ)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;

        {
            പൂർണ്ണസംഖ്യ vh;
            ggml_sycl_memcpy_1<വലുപ്പം(പൂർണ്ണസംഖ്യ)>(&vh, K_q5_1[ib].qh);
            vh >>= iqs8 * QI5_0;

            v |= (vh <<  4) & 0x00000010; // 0 ->  4
            v |= (vh << 11) & 0x00001000; // 1 -> 12
            v |= (vh << 18) & 0x00100000; // 2 -> 20
            v |= (vh << 25) & 0x10000000; // 3 -> 28
        }

        സ്ഥിരം പൂർണ്ണസംഖ്യ u = Q_q8[k_KQ_0/nthreads];

        സ്ഥിരം പൂർണ്ണസംഖ്യ sumi = ggml_sycl_dp4a(v, u, 0);

        സ്ഥിരം sycl::float2 K_dm = (K_q5_1[ib].dm).template convert<ദശാംശം, sycl::rounding_mode::automatic>();
        സ്ഥിരം sycl::float2 Q_ds = ((സ്ഥിരം sycl::float2 *) Q_ds_v)[k_KQ_0 / nthreads];

        sum += K_dm.x()*Q_ds.x()*sumi + K_dm.y()*Q_ds.y()/QI8_1;
    }

    തിരികെ sum;
}

template <പൂർണ്ണസംഖ്യ D, പൂർണ്ണസംഖ്യ nthreads, പൂർണ്ണസംഖ്യ warp_size>
സ്ഥിതം __dpct_inline__ ദശാംശം vec_dot_fattn_vec_KQ_q8_0(സ്ഥിരം അക്ഷരം * __restrict__ K_c,
                                                       സ്ഥിരം ശൂന്യം * __restrict__ Q_v,
                                                       സ്ഥിരം പൂർണ്ണസംഖ്യ * __restrict__ Q_q8,
                                                       സ്ഥിരം ശൂന്യം * __restrict__ Q_ds_v) {
    auto               item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    സ്ഥിരം block_q8_0 * K_q8_0   = (സ്ഥിരം block_q8_0 *) K_c;
    GGML_UNUSED(Q_v);

    ദശാംശം sum = 0.0f;

#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ k_KQ_0 = 0; k_KQ_0 < പൂർണ്ണസംഖ്യ(D/വലുപ്പം(പൂർണ്ണസംഖ്യ)); k_KQ_0 += nthreads) {
        സ്ഥിരം പൂർണ്ണസംഖ്യ k_KQ =
            k_KQ_0 + (nthreads == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads);

        സ്ഥിരം പൂർണ്ണസംഖ്യ ib  = k_KQ / QI8_0;
        സ്ഥിരം പൂർണ്ണസംഖ്യ iqs = k_KQ % QI8_0;

        പൂർണ്ണസംഖ്യ v;
        ggml_sycl_memcpy_1<വലുപ്പം(v), 2>(&v, K_q8_0[ib].qs + 4*iqs);

        സ്ഥിരം sycl::float2 * Q_ds = (സ്ഥിരം sycl::float2 *) Q_ds_v;
        സ്ഥിരം ദശാംശം          Q_d  = Q_ds[k_KQ_0 / nthreads].x();

        sum += vec_dot_q8_0_q8_1_impl<ദശാംശം, 1>(&v, &Q_q8[k_KQ_0/nthreads], K_q8_0[ib].d, Q_d);
    }

    തിരികെ sum;
}

template <typename Tds, പൂർണ്ണസംഖ്യ ni, പൂർണ്ണസംഖ്യ warp_size>
സ്ഥിതം __dpct_inline__ ശൂന്യം quantize_q8_1_to_shared(സ്ഥിരം ദശാംശം * __restrict__ x,
                                                    സ്ഥിരം ദശാംശം scale,
                                                    പൂർണ്ണസംഖ്യ * __restrict__ yq32,
                                                    ശൂന്യം * __restrict__ yds) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    ദശാംശം vals[വലുപ്പം(പൂർണ്ണസംഖ്യ)] = { 0.0f };
#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < പൂർണ്ണസംഖ്യ(വലുപ്പം(പൂർണ്ണസംഖ്യ)); ++l) {
        vals[l] =
            (ni == warp_size || item_ct1.get_local_id(2) < ni) ? scale * x[4 * item_ct1.get_local_id(2) + l] : 0.0f;
    }

    ദശാംശം amax = sycl::fabs(vals[0]);
    ദശാംശം sum  = vals[0];
#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ l = 1; l < പൂർണ്ണസംഖ്യ(വലുപ്പം(പൂർണ്ണസംഖ്യ)); ++l) {
        amax = sycl::fmax(amax, sycl::fabs(vals[l]));
        sum += vals[l];
    }
#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ mask = QI8_1/2; mask > 0; mask >>= 1) {
        amax = sycl::fmax(
            amax, dpct::permute_sub_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(), amax, mask));
        sum += dpct::permute_sub_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(), sum, mask);
    }

    സ്ഥിരം ദശാംശം d = amax / 127;
    പൂർണ്ണസംഖ്യ q32 = 0;
    int8_t * q8 = (int8_t *) &q32;

    എങ്കിൽ (d != 0.0f) {
#pragma unroll
        വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < പൂർണ്ണസംഖ്യ(വലുപ്പം(പൂർണ്ണസംഖ്യ)); ++l) {
            q8[l] = sycl::round(vals[l] / d);
        }
    }

    yq32[item_ct1.get_local_id(2)] = q32;
    എങ്കിൽ (item_ct1.get_local_id(2) % QI8_1 == 0 && (ni == warp_size || item_ct1.get_local_id(2) < ni)) {
        എങ്കിൽ (std::is_same<Tds, sycl::half2>::value) {
            ((sycl::half2  *) yds)[item_ct1.get_local_id(2)/QI8_1] =  make_half2(d, sum);
        } അല്ലെങ്കിൽ {
            ((sycl::float2 *) yds)[item_ct1.get_local_id(2)/QI8_1] = make_float2(d, sum);
        }
    }
}

തരനിർവചനം ശൂന്യം (*dequantize_V_t)(സ്ഥിരം ശൂന്യം *, ശൂന്യം *, സ്ഥിരം int64_t);

template <typename T, പൂർണ്ണസംഖ്യ ne>
സ്ഥിതം __dpct_inline__ ശൂന്യം dequantize_V_f16(സ്ഥിരം ശൂന്യം * __restrict__ vx, ശൂന്യം * __restrict__ dst, സ്ഥിരം int64_t i0) {
    എങ്കിൽ constexpr (std::is_same_v<T, sycl::half>) {
        ggml_sycl_memcpy_1<ne * വലുപ്പം(sycl::half)>(dst, (സ്ഥിരം sycl::half *) vx + i0);
    } അല്ലെങ്കിൽ എങ്കിൽ constexpr (std::is_same_v<T, ദശാംശം>) {
        static_assert(ne % 2 == 0, "bad ne");
        sycl::half2 tmp[ne / 2];
        ggml_sycl_memcpy_1<ne * വലുപ്പം(sycl::half)>(tmp, (സ്ഥിരം sycl::half *) vx + i0);
        sycl::float2 * dst_f2 = (sycl::float2 *) dst;
#pragma unroll
        വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < ne/2; ++l) {
            dst_f2[l] = tmp[l].template convert<ദശാംശം, sycl::rounding_mode::automatic>();
        }
    } അല്ലെങ്കിൽ {
        static_assert(std::is_same_v<T, ശൂന്യം>, "unsupported type");
    }
}

template <typename T, പൂർണ്ണസംഖ്യ ne>
സ്ഥിതം __dpct_inline__ ശൂന്യം dequantize_V_q4_0(സ്ഥിരം ശൂന്യം * __restrict__ vx, ശൂന്യം * __restrict__ dst, സ്ഥിരം int64_t i0) {
    സ്ഥിരം block_q4_0 * x = (സ്ഥിരം block_q4_0 *) vx;

    സ്ഥിരം int64_t ib    =  i0          /  QK4_0;
    സ്ഥിരം പൂർണ്ണസംഖ്യ     iqs   =  i0          % (QK4_0/2);
    സ്ഥിരം പൂർണ്ണസംഖ്യ     shift = (i0 % QK4_0) / (QK4_0/2);

    പൂർണ്ണസംഖ്യ q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_sycl_memcpy_1<ne, 2>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;
    q = dpct::vectorized_ദ്വിമാന<sycl::char4>(q, 0x08080808, dpct::sub_sat());

    സ്ഥിരം int8_t * q8 = (സ്ഥിരം int8_t *) &q;

#ifdef GGML_SYCL_F16
    എങ്കിൽ constexpr (std::is_same_v<T, sycl::half>) {
        സ്ഥിരം sycl::half2 d = sycl::half2(x[ib].d);

#pragma unroll
        വേണ്ടി (പൂർണ്ണസംഖ്യ l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * sycl::half2(q8[l0 + 0], q8[l0 + 1]);
        }
    } അല്ലെങ്കിൽ
#endif // GGML_SYCL_F16
    എങ്കിൽ constexpr (std::is_same_v<T, ദശാംശം>) {
        സ്ഥിരം ദശാംശം d = x[ib].d;

#pragma unroll
        വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < ne; ++l) {
            ((ദശാംശം *) dst)[l] = d * q8[l];
        }
    } അല്ലെങ്കിൽ {
        static_assert(std::is_same_v<T, ശൂന്യം>, "bad type");
    }
}

template <typename T, പൂർണ്ണസംഖ്യ ne>
സ്ഥിതം __dpct_inline__ ശൂന്യം dequantize_V_q4_1(സ്ഥിരം ശൂന്യം * __restrict__ vx, ശൂന്യം * __restrict__ dst, സ്ഥിരം int64_t i0) {
    സ്ഥിരം block_q4_1 * x = (സ്ഥിരം block_q4_1 *) vx;

    സ്ഥിരം int64_t ib    =  i0          /  QK4_1;
    സ്ഥിരം പൂർണ്ണസംഖ്യ     iqs   =  i0          % (QK4_1/2);
    സ്ഥിരം പൂർണ്ണസംഖ്യ     shift = (i0 % QK4_1) / (QK4_1/2);

    പൂർണ്ണസംഖ്യ q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_sycl_memcpy_1<ne>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    സ്ഥിരം int8_t * q8 = (സ്ഥിരം int8_t *) &q;

#ifdef GGML_SYCL_F16
    എങ്കിൽ constexpr (std::is_same_v<T, sycl::half>) {
        സ്ഥിരം sycl::half2 dm = x[ib].dm;
        സ്ഥിരം sycl::half2 d  = sycl::half2(dm[0]);
        സ്ഥിരം sycl::half2 m  = sycl::half2(dm[1]);

#pragma unroll
        വേണ്ടി (പൂർണ്ണസംഖ്യ l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * sycl::half2(q8[l0 + 0], q8[l0 + 1]) + m;
        }
    } അല്ലെങ്കിൽ
#endif // GGML_SYCL_F16
    എങ്കിൽ constexpr (std::is_same_v<T, ദശാംശം>) {
        സ്ഥിരം sycl::float2 dm = (x[ib].dm).template convert<ദശാംശം, sycl::rounding_mode::automatic>();

#pragma unroll
        വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < ne; ++l) {
            ((ദശാംശം *) dst)[l] = dm.x() * q8[l] + dm.y();
        }
    } അല്ലെങ്കിൽ {
        static_assert(std::is_same_v<T, ശൂന്യം>, "bad type");
    }
}

template <typename T, പൂർണ്ണസംഖ്യ ne>
സ്ഥിതം __dpct_inline__ ശൂന്യം dequantize_V_q5_0(സ്ഥിരം ശൂന്യം * __restrict__ vx, ശൂന്യം * __restrict__ dst, സ്ഥിരം int64_t i0) {
    സ്ഥിരം block_q5_0 * x = (സ്ഥിരം block_q5_0 *) vx;

    സ്ഥിരം int64_t ib    =  i0          /  QK5_0;
    സ്ഥിരം പൂർണ്ണസംഖ്യ     idq   =  i0          %  QK5_0;
    സ്ഥിരം പൂർണ്ണസംഖ്യ     iqs   =  i0          % (QK5_0/2);
    സ്ഥിരം പൂർണ്ണസംഖ്യ     shift = (i0 % QK5_0) / (QK5_0/2);

    പൂർണ്ണസംഖ്യ q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_sycl_memcpy_1<ne, 2>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    {
        പൂർണ്ണസംഖ്യ qh;
        ggml_sycl_memcpy_1<ne, 2>(&qh, x[ib].qh);
#pragma unroll
        വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < ne; ++l) {
            q |= ((qh >> (idq + l)) & 0x00000001) << (8*l + 4);
        }
    }

    q = dpct::vectorized_ദ്വിമാന<sycl::char4>(q, 0x10101010, dpct::sub_sat());

    സ്ഥിരം int8_t * q8 = (സ്ഥിരം int8_t *) &q;

#ifdef GGML_SYCL_F16
    എങ്കിൽ constexpr (std::is_same_v<T, sycl::half>) {
        സ്ഥിരം sycl::half2 d = sycl::half2(x[ib].d);

#pragma unroll
        വേണ്ടി (പൂർണ്ണസംഖ്യ l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * sycl::half2(q8[l0 + 0], q8[l0 + 1]);
        }
    } അല്ലെങ്കിൽ
#endif // GGML_SYCL_F16
    എങ്കിൽ constexpr (std::is_same_v<T, ദശാംശം>) {
        സ്ഥിരം ദശാംശം d = x[ib].d;

#pragma unroll
        വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < ne; ++l) {
            ((ദശാംശം *) dst)[l] = d * q8[l];
        }
    } അല്ലെങ്കിൽ {
        static_assert(std::is_same_v<T, ശൂന്യം>, "bad type");
    }
}

template <typename T, പൂർണ്ണസംഖ്യ ne>
സ്ഥിതം __dpct_inline__ ശൂന്യം dequantize_V_q5_1(സ്ഥിരം ശൂന്യം * __restrict__ vx, ശൂന്യം * __restrict__ dst, സ്ഥിരം int64_t i0) {
    സ്ഥിരം block_q5_1 * x = (സ്ഥിരം block_q5_1 *) vx;

    സ്ഥിരം int64_t ib    =  i0          /  QK5_1;
    സ്ഥിരം പൂർണ്ണസംഖ്യ     idq   =  i0          %  QK5_1;
    സ്ഥിരം പൂർണ്ണസംഖ്യ     iqs   =  i0          % (QK5_1/2);
    സ്ഥിരം പൂർണ്ണസംഖ്യ     shift = (i0 % QK5_1) / (QK5_1/2);

    പൂർണ്ണസംഖ്യ q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_sycl_memcpy_1<ne>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    {
        പൂർണ്ണസംഖ്യ qh;
        ggml_sycl_memcpy_1<ne>(&qh, x[ib].qh);
#pragma unroll
        വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < ne; ++l) {
            q |= ((qh >> (idq + l)) & 0x00000001) << (8*l + 4);
        }
    }

    സ്ഥിരം int8_t * q8 = (സ്ഥിരം int8_t *) &q;

#ifdef GGML_SYCL_F16
    എങ്കിൽ constexpr (std::is_same_v<T, sycl::half>) {
        സ്ഥിരം sycl::half2 dm = x[ib].dm;
        സ്ഥിരം sycl::half2 d  = sycl::half2(dm[0]);
        സ്ഥിരം sycl::half2 m  = sycl::half2(dm[1]);

#pragma unroll
        വേണ്ടി (പൂർണ്ണസംഖ്യ l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * sycl::half2(q8[l0 + 0], q8[l0 + 1]) + m;
        }
    } അല്ലെങ്കിൽ
#endif // GGML_SYCL_F16
    എങ്കിൽ constexpr (std::is_same_v<T, ദശാംശം>) {
        സ്ഥിരം sycl::float2 dm = (x[ib].dm).template convert<ദശാംശം, sycl::rounding_mode::automatic>();

#pragma unroll
        വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < ne; ++l) {
            ((ദശാംശം *) dst)[l] = dm.x() * q8[l] + dm.y();
        }
    } അല്ലെങ്കിൽ {
        static_assert(std::is_same_v<T, ശൂന്യം>, "bad type");
    }
}

template <typename T, പൂർണ്ണസംഖ്യ ne>
സ്ഥിതം __dpct_inline__ ശൂന്യം dequantize_V_q8_0(സ്ഥിരം ശൂന്യം * __restrict__ vx, ശൂന്യം * __restrict__ dst, സ്ഥിരം int64_t i0) {
    സ്ഥിരം block_q8_0 * x = (സ്ഥിരം block_q8_0 *) vx;

    സ്ഥിരം int64_t ib  = i0 / QK8_0;
    സ്ഥിരം പൂർണ്ണസംഖ്യ     iqs = i0 % QK8_0;

    static_assert(ne % 2 == 0, "bad ne");
    int8_t qs[ne];
    ggml_sycl_memcpy_1<ne, 2>(qs, x[ib].qs + iqs);

#ifdef GGML_SYCL_F16
    എങ്കിൽ constexpr (std::is_same<T, sycl::half>::value) {
        സ്ഥിരം sycl::half2 d = sycl::half2(x[ib].d);

#pragma unroll
        വേണ്ടി (പൂർണ്ണസംഖ്യ l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * make_half2(qs[l0 + 0], qs[l0 + 1]);
        }
    } അല്ലെങ്കിൽ
#endif // GGML_SYCL_F16
    എങ്കിൽ constexpr (std::is_same<T, ദശാംശം>::value) {
        സ്ഥിരം ദശാംശം d = x[ib].d;

#pragma unroll
        വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < ne; ++l) {
            ((ദശാംശം *) dst)[l] = d * qs[l];
        }
    } അല്ലെങ്കിൽ {
        static_assert(std::is_same_v<T, ശൂന്യം>, "unsupported type");
    }
}

template <പൂർണ്ണസംഖ്യ type_K, പൂർണ്ണസംഖ്യ D, പൂർണ്ണസംഖ്യ nthreads, പൂർണ്ണസംഖ്യ warp_size>
constexpr vec_dot_KQ_t get_vec_dot_KQ() {
    എങ്കിൽ constexpr (type_K == GGML_TYPE_F16) {
        തിരികെ vec_dot_fattn_vec_KQ_f16<D, nthreads>;
    } അല്ലെങ്കിൽ എങ്കിൽ constexpr (type_K == GGML_TYPE_Q4_0) {
        തിരികെ vec_dot_fattn_vec_KQ_q4_0<D, nthreads, warp_size>;
    } അല്ലെങ്കിൽ എങ്കിൽ constexpr (type_K == GGML_TYPE_Q4_1) {
        തിരികെ vec_dot_fattn_vec_KQ_q4_1<D, nthreads, warp_size>;
    } അല്ലെങ്കിൽ എങ്കിൽ constexpr (type_K == GGML_TYPE_Q5_0) {
        തിരികെ vec_dot_fattn_vec_KQ_q5_0<D, nthreads, warp_size>;
    } അല്ലെങ്കിൽ എങ്കിൽ constexpr (type_K == GGML_TYPE_Q5_1) {
        തിരികെ vec_dot_fattn_vec_KQ_q5_1<D, nthreads, warp_size>;
    } അല്ലെങ്കിൽ എങ്കിൽ constexpr (type_K == GGML_TYPE_Q8_0) {
        തിരികെ vec_dot_fattn_vec_KQ_q8_0<D, nthreads, warp_size>;
    } അല്ലെങ്കിൽ {
        static_assert(type_K == -1, "bad type");
        തിരികെ nullptr;
    }
}

template <പൂർണ്ണസംഖ്യ type_V, typename T, പൂർണ്ണസംഖ്യ ne>
constexpr dequantize_V_t get_dequantize_V() {
    എങ്കിൽ constexpr (type_V == GGML_TYPE_F16) {
        തിരികെ dequantize_V_f16<T, ne>;
    } അല്ലെങ്കിൽ എങ്കിൽ constexpr (type_V == GGML_TYPE_Q4_0) {
        തിരികെ dequantize_V_q4_0<T, ne>;
    } അല്ലെങ്കിൽ എങ്കിൽ constexpr (type_V == GGML_TYPE_Q4_1) {
        തിരികെ dequantize_V_q4_1<T, ne>;
    } അല്ലെങ്കിൽ എങ്കിൽ constexpr (type_V == GGML_TYPE_Q5_0) {
        തിരികെ dequantize_V_q5_0<T, ne>;
    } അല്ലെങ്കിൽ എങ്കിൽ constexpr (type_V == GGML_TYPE_Q5_1) {
        തിരികെ dequantize_V_q5_1<T, ne>;
    } അല്ലെങ്കിൽ എങ്കിൽ constexpr (type_V == GGML_TYPE_Q8_0) {
        തിരികെ dequantize_V_q8_0<T, ne>;
    } അല്ലെങ്കിൽ {
        static_assert(type_V == -1, "bad type");
        തിരികെ nullptr;
    }
}

template <പൂർണ്ണസംഖ്യ ncols1, പൂർണ്ണസംഖ്യ warp_size>
സ്ഥിതം ശൂന്യം flash_attn_mask_to_KV_max(സ്ഥിരം sycl::half2 * __restrict__ mask,
                                      പൂർണ്ണസംഖ്യ * __restrict__ KV_max,
                                      സ്ഥിരം പൂർണ്ണസംഖ്യ ne30,
                                      സ്ഥിരം പൂർണ്ണസംഖ്യ s31,
                                      സ്ഥിരം പൂർണ്ണസംഖ്യ s33,
                                      പൂർണ്ണസംഖ്യ *     buf_iw) {
    auto      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    സ്ഥിരം പൂർണ്ണസംഖ്യ ne31     = item_ct1.get_group_range(2);
    സ്ഥിരം പൂർണ്ണസംഖ്യ tid      = item_ct1.get_local_id(2);
    സ്ഥിരം പൂർണ്ണസംഖ്യ sequence = item_ct1.get_group(1);
    സ്ഥിരം പൂർണ്ണസംഖ്യ jt       = item_ct1.get_group(2);

    mask += sequence*s33 + jt*ncols1*s31;

    എങ്കിൽ (tid < warp_size) {
        buf_iw[tid] = 1;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);

    പൂർണ്ണസംഖ്യ KV_max_sj = (ne30 - 1) * FATTN_KQ_STRIDE;
    വേണ്ടി (; KV_max_sj >= 0; KV_max_sj -= FATTN_KQ_STRIDE) {
        പൂർണ്ണസംഖ്യ all_inf = 1;

#pragma unroll
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < ncols1; ++j) {
            സ്ഥിരം sycl::float2 tmp =
                mask[j * s31 + KV_max_sj / 2 + tid].template convert<ദശാംശം, sycl::rounding_mode::automatic>();
            all_inf = all_inf && പൂർണ്ണസംഖ്യ(sycl::isinf((ദശാംശം) (tmp.x()))) && പൂർണ്ണസംഖ്യ(sycl::isinf((ദശാംശം) (tmp.y())));
        }

        all_inf = warp_reduce_all<warp_size>(all_inf);
        എങ്കിൽ (tid % warp_size == 0) {
            buf_iw[tid / warp_size] = all_inf;
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);
        all_inf = buf_iw[tid % warp_size];
        item_ct1.barrier(sycl::access::fence_space::local_space);
        all_inf = warp_reduce_all<warp_size>(all_inf);

        എങ്കിൽ (!all_inf) {
            നിർത്തുക;
        }
    }

    // If the break in the loop was not triggered, KV_max_sj is now -FATTN_KQ_STRIDE.
    // If the break was triggered it's the lower edge of the tile with the first non-masked values.
    // In either case, walk back the decrementation by FATTN_KQ_STRIDE.
    KV_max_sj += FATTN_KQ_STRIDE;

    എങ്കിൽ (item_ct1.get_local_id(2) != 0) {
        തിരികെ;
    }

    KV_max[sequence*ne31 + jt] = KV_max_sj;
}

template <പൂർണ്ണസംഖ്യ D, പൂർണ്ണസംഖ്യ ncols1, പൂർണ്ണസംഖ്യ ncols2>  // D == head size

സ്ഥിതം ശൂന്യം flash_attn_stream_k_fixup(ദശാംശം * __restrict__ dst,
                                      സ്ഥിരം sycl::float2 * __restrict__ dst_fixup,
                                      സ്ഥിരം പൂർണ്ണസംഖ്യ ne01,
                                      സ്ഥിരം പൂർണ്ണസംഖ്യ ne02,
                                      സ്ഥിരം പൂർണ്ണസംഖ്യ ne03,
                                      സ്ഥിരം പൂർണ്ണസംഖ്യ ne11,
                                      സ്ഥിരം പൂർണ്ണസംഖ്യ ne12,
                                      സ്ഥിരം പൂർണ്ണസംഖ്യ nbatch_fa) {
    auto          item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr പൂർണ്ണസംഖ്യ ncols    = ncols1 * ncols2;

    സ്ഥിരം പൂർണ്ണസംഖ്യ bidx0 = item_ct1.get_group(2);
    സ്ഥിരം പൂർണ്ണസംഖ്യ j     = item_ct1.get_group(1);
    സ്ഥിരം പൂർണ്ണസംഖ്യ c     = item_ct1.get_group(0);
    സ്ഥിരം പൂർണ്ണസംഖ്യ jc    = j*ncols2 + c;
    സ്ഥിരം പൂർണ്ണസംഖ്യ tid   = item_ct1.get_local_id(2);

    സ്ഥിരം ദശാംശം * dst_fixup_data = ((സ്ഥിരം ദശാംശം *) dst_fixup) + item_ct1.get_group_range(2) * (2 * 2 * ncols);

    സ്ഥിരം പൂർണ്ണസംഖ്യ gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.

    സ്ഥിരം പൂർണ്ണസംഖ്യ iter_k     = (ne11      + (nbatch_fa - 1)) / nbatch_fa;
    സ്ഥിരം പൂർണ്ണസംഖ്യ iter_j     = (ne01      + (ncols1    - 1)) / ncols1;
    സ്ഥിരം പൂർണ്ണസംഖ്യ iter_z_gqa = (gqa_ratio + (ncols2    - 1)) / ncols2;

    സ്ഥിരം പൂർണ്ണസംഖ്യ kbc0 = int64_t(bidx0 + 0) * (iter_k * iter_j * iter_z_gqa * ne12 * ne03) / item_ct1.get_group_range(2);
    സ്ഥിരം പൂർണ്ണസംഖ്യ kbc0_stop =
        int64_t(bidx0 + 1) * (iter_k * iter_j * iter_z_gqa * ne12 * ne03) / item_ct1.get_group_range(2);

    സ്ഥിരം തർക്കം did_not_have_any_data   = kbc0 == kbc0_stop;
    സ്ഥിരം തർക്കം wrote_beginning_of_tile = kbc0 % iter_k == 0;
    സ്ഥിരം തർക്കം did_not_write_last      = kbc0/iter_k == kbc0_stop/iter_k && kbc0_stop % iter_k != 0;
    എങ്കിൽ (did_not_have_any_data || wrote_beginning_of_tile || did_not_write_last) {
        തിരികെ;
    }

    // z_KV == K/V head index, zt_gqa = Q head start index per K/V head, jt = token position start index
    സ്ഥിരം പൂർണ്ണസംഖ്യ sequence =  kbc0 /(iter_k*iter_j*iter_z_gqa*ne12);
    സ്ഥിരം പൂർണ്ണസംഖ്യ z_KV     = (kbc0 - iter_k*iter_j*iter_z_gqa*ne12 * sequence)/(iter_k*iter_j*iter_z_gqa);
    സ്ഥിരം പൂർണ്ണസംഖ്യ zt_gqa   = (kbc0 - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV)/(iter_k*iter_j);
    സ്ഥിരം പൂർണ്ണസംഖ്യ jt       = (kbc0 - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV - iter_k*iter_j * zt_gqa) / iter_k;

    സ്ഥിരം പൂർണ്ണസംഖ്യ zt_Q = z_KV*gqa_ratio + zt_gqa*ncols2; // Global Q head start index.

    എങ്കിൽ (jt*ncols1 + j >= ne01 || zt_gqa*ncols2 + c >= gqa_ratio) {
        തിരികെ;
    }

    dst += sequence*ne02*ne01*D + jt*ne02*(ncols1*D) + zt_Q*D + (j*ne02 + c)*D + tid;

    // Load the partial result that needs a fixup:
    ദശാംശം dst_val = 0.0f;
    ദശാംശം max_val = 0.0f;
    ദശാംശം rowsum  = 0.0f;
    {
        dst_val = *dst;

        സ്ഥിരം sycl::float2 tmp = dst_fixup[bidx0 * ncols + jc];
        max_val                = tmp.x();
        rowsum                 = tmp.y();
    }

    // Iterate over previous blocks and compute the combined results.
    // All SYCL blocks that get here must have a previous block that needs a fixup.
    പൂർണ്ണസംഖ്യ bidx = bidx0 - 1;
    പൂർണ്ണസംഖ്യ kbc_stop = kbc0;
    ആകുമ്പോൾ(സത്യം) {
        സ്ഥിരം പൂർണ്ണസംഖ്യ kbc = int64_t(bidx) * (iter_k * iter_j * iter_z_gqa * ne12 * ne03) / item_ct1.get_group_range(2);
        എങ്കിൽ (kbc == kbc_stop) { // Did not have any data.
            bidx--;
            kbc_stop = kbc;
            തുടരുക;
        }

        സ്ഥിരം ദശാംശം dst_add = dst_fixup_data[bidx*ncols*D + jc*D + tid];

        സ്ഥിരം sycl::float2 tmp = dst_fixup[(item_ct1.get_group_range(2) + bidx) * ncols + jc];

        // Scale the current and new value accumulators depending on the max. values.
        സ്ഥിരം ദശാംശം max_val_new = sycl::fmax(max_val, tmp.x());

        സ്ഥിരം ദശാംശം diff_val = max_val - max_val_new;
        സ്ഥിരം ദശാംശം diff_add = tmp.x() - max_val_new;

        സ്ഥിരം ദശാംശം scale_val = diff_val >= SOFTMAX_FTZ_THRESHOLD ? sycl::native::exp(diff_val) : 0.0f;
        സ്ഥിരം ദശാംശം scale_add = diff_add >= SOFTMAX_FTZ_THRESHOLD ? sycl::native::exp(diff_add) : 0.0f;

        dst_val = scale_val*dst_val + scale_add*dst_add;
        rowsum  = scale_val * rowsum + scale_add * tmp.y();

        max_val = max_val_new;

        // If this block started in a previous tile we are done and don't need to combine additional partial results.
        എങ്കിൽ (kbc % iter_k == 0 || kbc/iter_k < kbc0/iter_k) {
            നിർത്തുക;
        }
        bidx--;
        kbc_stop = kbc;
    }

    // Write back final result:
    *dst = dst_val / rowsum;
}

template <പൂർണ്ണസംഖ്യ D>  // D == head size

സ്ഥിതം ശൂന്യം flash_attn_combine_results(സ്ഥിരം ദശാംശം * __restrict__ VKQ_parts,
                                       സ്ഥിരം sycl::float2 * __restrict__ VKQ_meta,
                                       ദശാംശം * __restrict__ dst,
                                       സ്ഥിരം പൂർണ്ണസംഖ്യ parallel_blocks,
                                       uint8_t * dpct_local) {
    // Dimension 0: threadIdx.x
    // Dimension 1: blockIdx.x
    // Dimension 2: blockIdx.y
    // Dimension 3: blockIdx.z
    // Memory layout is permuted with [0, 2, 1, 3]

    auto      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    സ്ഥിരം പൂർണ്ണസംഖ്യ ne01     = item_ct1.get_group_range(2);
    സ്ഥിരം പൂർണ്ണസംഖ്യ ne02     = item_ct1.get_group_range(1);

    സ്ഥിരം പൂർണ്ണസംഖ്യ col      = item_ct1.get_group(2);
    സ്ഥിരം പൂർണ്ണസംഖ്യ head     = item_ct1.get_group(1);
    സ്ഥിരം പൂർണ്ണസംഖ്യ sequence = item_ct1.get_group(0);

    സ്ഥിരം പൂർണ്ണസംഖ്യ j_dst_unrolled = (sequence*ne01 + col)*ne02 + head;

    VKQ_parts += j_dst_unrolled * parallel_blocks*D;
    VKQ_meta  += j_dst_unrolled * parallel_blocks;
    dst       += j_dst_unrolled *                 D;

    സ്ഥിരം പൂർണ്ണസംഖ്യ tid = item_ct1.get_local_id(2);
    __builtin_assume(tid < D);

    auto meta = (sycl::float2 *) dpct_local;
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = tid; i < 2*parallel_blocks; i += D) {
        ((ദശാംശം *) meta)[i] = ((സ്ഥിരം ദശാംശം *)VKQ_meta) [i];
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    ദശാംശം kqmax = meta[0].x();
    വേണ്ടി (പൂർണ്ണസംഖ്യ l = 1; l < parallel_blocks; ++l) {
        kqmax = sycl::max(kqmax, meta[l].x());
    }

    ദശാംശം VKQ_numerator   = 0.0f;
    ദശാംശം VKQ_denominator = 0.0f;
    വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < parallel_blocks; ++l) {
        സ്ഥിരം ദശാംശം KQ_max_scale = sycl::native::exp(meta[l].x() - kqmax);

        VKQ_numerator   += KQ_max_scale * VKQ_parts[l*D + tid];
        VKQ_denominator += KQ_max_scale * meta[l].y();
    }

    dst[tid] = VKQ_numerator / VKQ_denominator;
}

// Carries the sub-group size (and grf_size_automatic when use_large_grf) as
// kernel properties via get(properties_tag), in place of the
// reqd_sub_group_size attribute and the deprecated parallel_for overload.
template <typename KernelFunc, uint32_t Size, തർക്കം UseGrf>
ഘടന props_kernel {
    KernelFunc kernel;

    ശൂന്യം operator()(sycl::nd_item<3> item) സ്ഥിരം {
        kernel(item);
    }

    auto get(sycl::ext::oneapi::experimental::properties_tag) സ്ഥിരം {
        എങ്കിൽ constexpr (UseGrf) {
            തിരികെ sycl::ext::oneapi::experimental::properties{
                sycl::ext::intel::experimental::grf_size_automatic,
                sycl::ext::oneapi::experimental::sub_group_size<Size> };
        } അല്ലെങ്കിൽ {
            തിരികെ sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::sub_group_size<Size> };
        }
    }
};

template <fattn_kernel_t fattn_kernel, പൂർണ്ണസംഖ്യ warp_size, തർക്കം use_large_grf = അസത്യം>
സ്ഥിതം ശൂന്യം lauch_kernel(
    dpct::dim3 group_range,
    dpct::dim3 local_range,
    queue_ptr q,
    unsigned പൂർണ്ണസംഖ്യ local_mem_size,
    സ്ഥിരം അക്ഷരം* __restrict__ Q,
    സ്ഥിരം അക്ഷരം* __restrict__ K,
    സ്ഥിരം അക്ഷരം* __restrict__ V,
    സ്ഥിരം അക്ഷരം* __restrict__ mask,
    സ്ഥിരം അക്ഷരം* __restrict__ sinks,
    സ്ഥിരം പൂർണ്ണസംഖ്യ* __restrict__ KV_max,
    ദശാംശം* __restrict__ dst,
    sycl::float2* __restrict__ dst_meta,
    സ്ഥിരം ദശാംശം scale,
    സ്ഥിരം ദശാംശം max_bias,
    സ്ഥിരം ദശാംശം m0,
    സ്ഥിരം ദശാംശം m1,
    സ്ഥിരം uint32_t n_head_log2,
    സ്ഥിരം ദശാംശം logit_softcap,
    സ്ഥിരം int32_t ne00,
    സ്ഥിരം sycl::uint3 ne01,
    സ്ഥിരം int32_t ne02,
    സ്ഥിരം int32_t ne03,
    സ്ഥിരം int32_t nb01,
    സ്ഥിരം int32_t nb02,
    സ്ഥിരം int32_t nb03,
    സ്ഥിരം int32_t ne10,
    സ്ഥിരം int32_t ne11,
    സ്ഥിരം int32_t ne12,
    സ്ഥിരം int32_t ne13,
    സ്ഥിരം int32_t nb11,
    സ്ഥിരം int32_t nb12,
    സ്ഥിരം int64_t nb13,
    സ്ഥിരം int32_t nb21,
    സ്ഥിരം int32_t nb22,
    സ്ഥിരം int64_t nb23,
    സ്ഥിരം int32_t ne31,
    സ്ഥിരം int32_t ne32,
    സ്ഥിരം int32_t ne33,
    സ്ഥിരം int32_t nb31,
    സ്ഥിരം int32_t nb32,
    സ്ഥിരം int64_t nb33) {
    GGML_UNUSED(local_mem_size);

    സ്ഥിരം auto rng = sycl::nd_range<3>(
        static_cast<sycl::range<3>>(group_range * local_range),
        static_cast<sycl::range<3>>(local_range));

    സ്ഥിരം auto kernel = [=](sycl::nd_item<3> item_ct1) {
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
template <പൂർണ്ണസംഖ്യ DV, പൂർണ്ണസംഖ്യ ncols1, പൂർണ്ണസംഖ്യ ncols2, fattn_kernel_t fattn_kernel, പൂർണ്ണസംഖ്യ warp_size, തർക്കം use_large_grf = അസത്യം>
ശൂന്യം launch_fattn(
    ggml_backend_sycl_context & ctx, ggml_tensor * dst, സ്ഥിരം പൂർണ്ണസംഖ്യ nwarps, സ്ഥിരം size_t nbytes_shared,
    സ്ഥിരം പൂർണ്ണസംഖ്യ nbatch_fa, സ്ഥിരം തർക്കം need_f16_K, സ്ഥിരം തർക്കം need_f16_V, സ്ഥിരം തർക്കം stream_k) {

    constexpr പൂർണ്ണസംഖ്യ ncols = ncols1 * ncols2;

    സ്ഥിരം ggml_tensor * Q = dst->src[0];
    സ്ഥിരം ggml_tensor * K = dst->src[1];
    സ്ഥിരം ggml_tensor * V = dst->src[2];

    സ്ഥിരം തർക്കം V_is_K_view = V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));

    സ്ഥിരം ggml_tensor * mask  = dst->src[3];
    സ്ഥിരം ggml_tensor * sinks = dst->src[4];

    ggml_tensor * KQV = dst;

    GGML_ASSERT(Q->type == GGML_TYPE_F32);
    GGML_ASSERT(KQV->type == GGML_TYPE_F32);

    GGML_ASSERT(Q->nb[0] == ggml_element_size(Q));
    GGML_ASSERT(K->nb[0] == ggml_element_size(K));
    GGML_ASSERT(V->nb[0] == ggml_element_size(V));

    GGML_ASSERT(!mask || mask->type == GGML_TYPE_F16);

    ggml_sycl_pool & pool = ctx.pool();
    dpct::queue_ptr  main_stream = ctx.stream();
    സ്ഥിരം പൂർണ്ണസംഖ്യ id  = ggml_sycl_get_device();
    സ്ഥിരം പൂർണ്ണസംഖ്യ nsm = ggml_sycl_info().devices[id].nsm;

    സ്ഥിരം ggml_sycl_fattn_extra  extra = ggml_sycl_fattn_get_extra(dst);
    ggml_sycl_pool_alloc<പൂർണ്ണസംഖ്യ>    KV_max(pool);
    ggml_sycl_pool_alloc<ദശാംശം>  dst_tmp(pool);
    ggml_sycl_pool_alloc<sycl::float2> dst_tmp_meta(pool);

    സ്ഥിരം അക്ഷരം * K_data = (സ്ഥിരം അക്ഷരം *) K->data;
    size_t nb11 = K->nb[1];
    size_t nb12 = K->nb[2];
    size_t nb13 = K->nb[3];

    സ്ഥിരം അക്ഷരം * V_data = (സ്ഥിരം അക്ഷരം *) V->data;
    size_t nb21 = V->nb[1];
    size_t nb22 = V->nb[2];
    size_t nb23 = V->nb[3];

    എങ്കിൽ (need_f16_K && K->type != GGML_TYPE_F16) {
        സ്ഥിരം size_t bs = ggml_blck_size(K->type);
        സ്ഥിരം size_t ts = ggml_type_size(K->type);

        GGML_ASSERT(extra.K_buffer_ptr);
        sycl::half * K_f16_ptr = (sycl::half *) extra.K_buffer_ptr;
        എങ്കിൽ (ggml_is_contiguously_allocated(K)) {
            to_fp16_sycl_t to_fp16 = ggml_get_to_fp16_sycl(K->type, dst);
            to_fp16(K_data, K_f16_ptr, ggml_nelements(K), main_stream);

            nb11 = nb11 * bs * വലുപ്പം(sycl::half) / ts;
            nb12 = nb12 * bs * വലുപ്പം(sycl::half) / ts;
            nb13 = nb13 * bs * വലുപ്പം(sycl::half) / ts;
        } അല്ലെങ്കിൽ {
            GGML_ASSERT(K->nb[0] == ts);
            to_fp16_nc_sycl_t to_fp16 = ggml_get_to_fp16_nc_sycl(K->type);
            സ്ഥിരം int64_t s01 = nb11 / ts;
            സ്ഥിരം int64_t s02 = nb12 / ts;
            സ്ഥിരം int64_t s03 = nb13 / ts;
            to_fp16(K_data, K_f16_ptr, K->ne[0], K->ne[1], K->ne[2], K->ne[3], s01, s02, s03, main_stream);

            nb11 = K->ne[0] * വലുപ്പം(sycl::half);
            nb12 = K->ne[1] * nb11;
            nb13 = K->ne[2] * nb12;
        }
        K_data = (അക്ഷരം *) K_f16_ptr;
    }

    എങ്കിൽ (need_f16_V && V->type != GGML_TYPE_F16) {
        എങ്കിൽ (V_is_K_view) {
            V_data = K_data;
            nb21   = nb11;
            nb22   = nb12;
            nb23   = nb13;
        } അല്ലെങ്കിൽ {
            സ്ഥിരം size_t bs = ggml_blck_size(V->type);
            സ്ഥിരം size_t ts = ggml_type_size(V->type);

            GGML_ASSERT(extra.V_buffer_ptr);
            sycl::half * V_f16_ptr = (sycl::half *) extra.V_buffer_ptr;
            എങ്കിൽ (ggml_is_contiguously_allocated(V)) {
                to_fp16_sycl_t to_fp16 = ggml_get_to_fp16_sycl(V->type, dst);
                to_fp16(V_data, V_f16_ptr, ggml_nelements(V), main_stream);
                V_data = (അക്ഷരം *) V_f16_ptr;

                nb21 = nb21 * bs * വലുപ്പം(sycl::half) / ts;
                nb22 = nb22 * bs * വലുപ്പം(sycl::half) / ts;
                nb23 = nb23 * bs * വലുപ്പം(sycl::half) / ts;
            } അല്ലെങ്കിൽ {
                GGML_ASSERT(V->nb[0] == ts);
                to_fp16_nc_sycl_t to_fp16 = ggml_get_to_fp16_nc_sycl(V->type);
                സ്ഥിരം int64_t s01 = nb21 / ts;
                സ്ഥിരം int64_t s02 = nb22 / ts;
                സ്ഥിരം int64_t s03 = nb23 / ts;
                to_fp16(V_data, V_f16_ptr, V->ne[0], V->ne[1], V->ne[2], V->ne[3], s01, s02, s03, main_stream);

                nb21 = V->ne[0] * വലുപ്പം(sycl::half);
                nb22 = V->ne[1] * nb21;
                nb23 = V->ne[2] * nb22;
            }
            V_data = (അക്ഷരം *) V_f16_ptr;
        }
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ ntiles_x     = ((Q->ne[1] + ncols1 - 1) / ncols1);
    സ്ഥിരം പൂർണ്ണസംഖ്യ gqa_ratio    = Q->ne[2] / K->ne[2];
    സ്ഥിരം പൂർണ്ണസംഖ്യ ntiles_z_gqa = ((gqa_ratio + ncols2 - 1) / ncols2);
    സ്ഥിരം പൂർണ്ണസംഖ്യ ntiles_total = ntiles_x * ntiles_z_gqa * K->ne[2] * Q->ne[3];

    // Optional optimization where the mask is scanned to determine whether part of the calculation can be skipped.
    // Only worth the overhead if there is at lease one FATTN_KQ_STRIDE x FATTN_KQ_STRIDE square to be skipped or
    //     multiple sequences of possibly different lengths.
    എങ്കിൽ (mask && K->ne[1] % FATTN_KQ_STRIDE == 0 && (Q->ne[1] >= 1024 || Q->ne[3] > 1)) {
        സ്ഥിരം പൂർണ്ണസംഖ്യ s31 = mask->nb[1] / വലുപ്പം(sycl::half2);
        സ്ഥിരം പൂർണ്ണസംഖ്യ s33 = mask->nb[3] / വലുപ്പം(sycl::half2);

        സ്ഥിരം dpct::dim3 blocks_അക്കം_KV_max(ntiles_x, Q->ne[3], 1);
        സ്ഥിരം dpct::dim3 block_dim_KV_max(FATTN_KQ_STRIDE / 2, 1, 1);

        സ്ഥിരം പൂർണ്ണസംഖ്യ ne_KV_max = blocks_അക്കം_KV_max.x*blocks_അക്കം_KV_max.y;
        സ്ഥിരം പൂർണ്ണസംഖ്യ iter_k = K->ne[1] / FATTN_KQ_STRIDE;

        KV_max.alloc(ne_KV_max);
        {
            dpct::has_capability_or_fail(main_stream->get_device(), { sycl::aspect::fp16 });

            main_stream->submit([&](sycl::handler & cgh) {
                sycl::local_accessor<പൂർണ്ണസംഖ്യ, 1> buf_iw_acc_ct1(sycl::range<1>(warp_size), cgh);

                auto mask_data_ct0  = (സ്ഥിരം sycl::half2 *) mask->data;
                auto KV_max_ptr_ct1 = KV_max.ptr;

                cgh.parallel_for(sycl::nd_range<3>(blocks_അക്കം_KV_max * block_dim_KV_max, block_dim_KV_max),
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

    സ്ഥിരം dpct::dim3 block_dim(warp_size, nwarps, 1);

    // Max. number of active blocks limited by occupancy.
    പൂർണ്ണസംഖ്യ max_blocks_per_sm = ggml_sycl_info().devices[id].max_wg_per_cu;
    പൂർണ്ണസംഖ്യ parallel_blocks = max_blocks_per_sm;
    dpct::dim3 blocks_അക്കം;
    എങ്കിൽ (stream_k) {
        // For short contexts it can be faster to have the SMs work on whole tiles because this lets us skip the fixup.
        സ്ഥിരം പൂർണ്ണസംഖ്യ max_blocks = max_blocks_per_sm*nsm;
        സ്ഥിരം പൂർണ്ണസംഖ്യ nblocks_stream_k = max_blocks;
        സ്ഥിരം തർക്കം use_stream_k = സത്യം;

        blocks_അക്കം.x = use_stream_k ? nblocks_stream_k : ntiles_total;
        blocks_അക്കം.y = 1;
        blocks_അക്കം.z = 1;

        എങ്കിൽ (ntiles_total % blocks_അക്കം.x != 0) { // Fixup is only needed if the SMs work on fractional tiles.
            dst_tmp_meta.alloc((size_t(blocks_അക്കം.x) * ncols * (2 + DV/2)));
        }
    } അല്ലെങ്കിൽ {
        സ്ഥിരം പൂർണ്ണസംഖ്യ ntiles_KQ = (K->ne[1] + nbatch_fa - 1) / nbatch_fa; // Max. number of parallel blocks limited by tensor size.

        // parallel_blocks must not be larger than what the tensor size allows:
        parallel_blocks = std::min(parallel_blocks, ntiles_KQ);
        // todo fix the hard code change
        // parallel_blocks = ntiles_KQ;

        // If ntiles_total % blocks_per_wave != 0 then some efficiency is lost due to tail effects.
        // Test whether parallel_blocks can be set to a higher value for better efficiency.
        സ്ഥിരം പൂർണ്ണസംഖ്യ blocks_per_wave = nsm * max_blocks_per_sm;
        പൂർണ്ണസംഖ്യ nwaves_best = 0;
        പൂർണ്ണസംഖ്യ efficiency_percent_best = 0;
        വേണ്ടി (പൂർണ്ണസംഖ്യ parallel_blocks_test = parallel_blocks; parallel_blocks_test <= ntiles_KQ; ++parallel_blocks_test) {
            സ്ഥിരം പൂർണ്ണസംഖ്യ nblocks_total = ntiles_total * parallel_blocks_test;
            സ്ഥിരം പൂർണ്ണസംഖ്യ nwaves = (nblocks_total + blocks_per_wave - 1) / blocks_per_wave;
            സ്ഥിരം പൂർണ്ണസംഖ്യ efficiency_percent = 100 * nblocks_total / (nwaves*blocks_per_wave);

            // Stop trying configurations with more waves if we already have good efficiency to avoid excessive overhead.
            എങ്കിൽ (efficiency_percent_best >= 95 && nwaves > nwaves_best) {
                നിർത്തുക;
            }

            എങ്കിൽ (efficiency_percent > efficiency_percent_best) {
                nwaves_best = nwaves;
                efficiency_percent_best = efficiency_percent;
                parallel_blocks = parallel_blocks_test;
            }
        }

        blocks_അക്കം.x = ntiles_x;
        blocks_അക്കം.y = parallel_blocks;
        blocks_അക്കം.z = ntiles_z_gqa*K->ne[2]*Q->ne[3];

        എങ്കിൽ (parallel_blocks > 1) {
            dst_tmp.alloc(parallel_blocks*ggml_nelements(KQV));
            dst_tmp_meta.alloc(parallel_blocks*ggml_nrows(KQV));
        }
    }

    ദശാംശം scale         = 1.0f;
    ദശാംശം max_bias      = 0.0f;
    ദശാംശം logit_softcap = 0.0f;

    memcpy(&scale,         (സ്ഥിരം ദശാംശം *) KQV->op_params + 0, വലുപ്പം(ദശാംശം));
    memcpy(&max_bias,      (സ്ഥിരം ദശാംശം *) KQV->op_params + 1, വലുപ്പം(ദശാംശം));
    memcpy(&logit_softcap, (സ്ഥിരം ദശാംശം *) KQV->op_params + 2, വലുപ്പം(ദശാംശം));

    എങ്കിൽ (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    സ്ഥിരം uint32_t n_head      = Q->ne[2];
    സ്ഥിരം uint32_t n_head_log2 = 1u << uint32_t(floorf(log2f(ദശാംശം(n_head))));

    സ്ഥിരം ദശാംശം m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    സ്ഥിരം ദശാംശം m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    // TODO other tensor dimensions after removal of WMMA kernel:
    സ്ഥിരം sycl::uint3 ne01 = init_fastdiv_മൂല്യങ്ങൾ(Q->ne[1]);

    GGML_ASSERT(block_dim.x % warp_size == 0);

    lauch_kernel<fattn_kernel, warp_size, use_large_grf>(
        blocks_അക്കം, block_dim, main_stream, (unsigned പൂർണ്ണസംഖ്യ) nbytes_shared, (സ്ഥിരം അക്ഷരം *) Q->data, K_data, V_data,
        mask ? ((സ്ഥിരം അക്ഷരം *) mask->data) : nullptr, sinks ? ((സ്ഥിരം അക്ഷരം *) sinks->data) : nullptr, KV_max.ptr,
        !stream_k && parallel_blocks > 1 ? dst_tmp.ptr : (ദശാംശം *) KQV->data, (sycl::float2 *)dst_tmp_meta.ptr, scale, max_bias, m0, m1,
        n_head_log2, logit_softcap, Q->ne[0], ne01, Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3], K->ne[0],
        K->ne[1], K->ne[2], K->ne[3], nb11, nb12, nb13, nb21, nb22, nb23, mask ? mask->ne[1] : 0,
        mask ? mask->ne[2] : 0, mask ? mask->ne[3] : 0, mask ? mask->nb[1] : 0, mask ? mask->nb[2] : 0,
        mask ? mask->nb[3] : 0);
    SYCL_CHECK(0);

    എങ്കിൽ (stream_k) {
        എങ്കിൽ (ntiles_total % blocks_അക്കം.x != 0) { // Fixup is only needed if the SMs work on fractional tiles.
            സ്ഥിരം dpct::dim3 block_dim_combine(DV, 1, 1);
            സ്ഥിരം dpct::dim3 blocks_അക്കം_combine = { blocks_അക്കം.x, ncols1, ncols2 };

            main_stream->submit([&](sycl::handler & cgh) {
                auto KQV_data_ct0         = (ദശാംശം *) KQV->data;
                auto dst_tmp_meta_ptr_ct1 = dst_tmp_meta.ptr;
                auto Q_ne_ct2             = Q->ne[1];
                auto Q_ne_ct3             = Q->ne[2];
                auto Q_ne_ct4             = Q->ne[3];
                auto K_ne_ct5             = K->ne[1];
                auto K_ne_ct6             = K->ne[2];

                cgh.parallel_for(sycl::nd_range<3>(blocks_അക്കം_combine * block_dim_combine, block_dim_combine),
                                 [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(warp_size)]] {
                                     GGML_UNUSED(item_ct1);
                                     flash_attn_stream_k_fixup<DV, ncols1, ncols2>(KQV_data_ct0, dst_tmp_meta_ptr_ct1,
                                                                                   Q_ne_ct2, Q_ne_ct3, Q_ne_ct4,
                                                                                   K_ne_ct5, K_ne_ct6, nbatch_fa);
                                 });
            });
        }
    } അല്ലെങ്കിൽ എങ്കിൽ (parallel_blocks > 1) {
        സ്ഥിരം dpct::dim3 block_dim_combine(DV, 1, 1);
        സ്ഥിരം dpct::dim3 blocks_അക്കം_combine(Q->ne[1], Q->ne[2], Q->ne[3]);
        സ്ഥിരം size_t     nbytes_shared_combine = parallel_blocks * വലുപ്പം(sycl::float2);
        main_stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(sycl::range<1>(nbytes_shared_combine), cgh);

            auto dst_tmp_ptr_ct0      = dst_tmp.ptr;
            auto dst_tmp_meta_ptr_ct1 = dst_tmp_meta.ptr;
            auto KQV_data_ct2         = (ദശാംശം *) KQV->data;

            cgh.parallel_for(sycl::nd_range<3>(blocks_അക്കം_combine * block_dim_combine, block_dim_combine),
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
