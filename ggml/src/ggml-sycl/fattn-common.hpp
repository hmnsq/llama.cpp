#pragma once

#ଅନ୍ତର୍ଭାଵ <sycl/sycl.hpp>
#ଅନ୍ତର୍ଭାଵ <sycl/ext/intel/experimental/grf_size_properties.hpp>
#ଅନ୍ତର୍ଭାଵ "dpct/helper.hpp"
#ଅନ୍ତର୍ଭାଵ "common.hpp"
#ଅନ୍ତର୍ଭାଵ "convert.hpp"
#ଅନ୍ତର୍ଭାଵ "vecdotq.hpp"
#ଅନ୍ତର୍ଭାଵ "fattn.hpp"

#ଅନ୍ତର୍ଭାଵ "ggml.h"

#ଅନ୍ତର୍ଭାଵ <cstdint>
#ଅନ୍ତର୍ଭାଵ <cmath>
#ଅନ୍ତର୍ଭାଵ <float.h>


#define FATTN_KQ_STRIDE       256
#define HALF_MAX_HALF         sycl::half(65504.0f/2) // Use neg. of this instead of -INFINITY to initialize KQ max vals to avoid NaN upon subtraction.
#define SOFTMAX_FTZ_THRESHOLD -20.0f                   // Softmax exp. of values smaller than this are flushed to zero to avoid NaNs.
#define FATTN_KQ_MAX_OFFSET (3.0f*0.6931f)

ପ୍ରକାରବ୍ୟାଖ୍ୟା ଶୂନ୍ୟ (*fattn_kernel_t)(
    ଅଚଳ ଅକ୍ଷର* Q,
    ଅଚଳ ଅକ୍ଷର* K,
    ଅଚଳ ଅକ୍ଷର* V,
    ଅଚଳ ଅକ୍ଷର* mask,
    ଅଚଳ ଅକ୍ଷର* sinks,
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା* KV_max,
    ଦଶମିକ* dst,
    sycl::float2* dst_meta,
    ଅଚଳ ଦଶମିକ scale,
    ଅଚଳ ଦଶମିକ max_bias,
    ଅଚଳ ଦଶମିକ m0,
    ଅଚଳ ଦଶମିକ m1,
    ଅଚଳ uint32_t n_head_log2,
    ଅଚଳ ଦଶମିକ logit_softcap,
    ଅଚଳ int32_t ne00,
    ଅଚଳ sycl::uint3 ne01,
    ଅଚଳ int32_t ne02,
    ଅଚଳ int32_t ne03,
    ଅଚଳ int32_t nb01,
    ଅଚଳ int32_t nb02,
    ଅଚଳ int32_t nb03,
    ଅଚଳ int32_t ne10,
    ଅଚଳ int32_t ne11,
    ଅଚଳ int32_t ne12,
    ଅଚଳ int32_t ne13,
    ଅଚଳ int32_t nb11,
    ଅଚଳ int32_t nb12,
    ଅଚଳ int64_t nb13,
    ଅଚଳ int32_t nb21,
    ଅଚଳ int32_t nb22,
    ଅଚଳ int64_t nb23,
    ଅଚଳ int32_t ne31,
    ଅଚଳ int32_t ne32,
    ଅଚଳ int32_t ne33,
    ଅଚଳ int32_t nb31,
    ଅଚଳ int32_t nb32,
    ଅଚଳ int64_t nb33);

ପ୍ରକାରବ୍ୟାଖ୍ୟା ଦଶମିକ (*vec_dot_KQ_t)(
    ଅଚଳ ଅକ୍ଷର * __restrict__ K_c, ଅଚଳ ଶୂନ୍ୟ * __restrict__ Q_v, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା * __restrict__ Q_q8 , ଅଚଳ ଶୂନ୍ୟ * __restrict__ Q_ds);

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା D, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nthreads>
ସ୍ଥିର __dpct_inline__ ଦଶମିକ vec_dot_fattn_vec_KQ_f16(ଅଚଳ ଅକ୍ଷର * __restrict__ K_c,
                                                      ଅଚଳ ଶୂନ୍ୟ * __restrict__ Q_v,
                                                      ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା * __restrict__ Q_q8,
                                                      ଅଚଳ ଶୂନ୍ୟ * __restrict__ Q_ds_v) {
    ଅଚଳ sycl::half2 * K_h2 = (ଅଚଳ sycl::half2 *) K_c;
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v);

    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା cpy_nb = ggml_sycl_get_max_cpy_bytes();
    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା cpy_ne = cpy_nb / 4;

    ଦଶମିକ sum = 0.0f;

#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads*cpy_ne) {
        sycl::half2 tmp[cpy_ne];
        ggml_sycl_memcpy_1<ଆକାର(tmp)>(
            tmp,
            K_h2 + k_KQ_0 + (sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2) % nthreads) * cpy_ne);
#pragma unroll
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
#ifdef GGML_SYCL_F16
            ggml_sycl_mad(sum,                tmp[k_KQ_1] , ((ଅଚଳ sycl::half2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#ଅନ୍ୟଥା
            ggml_sycl_mad(sum, __half22float2(tmp[k_KQ_1]), ((ଅଚଳ sycl::float2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#endif // GGML_SYCL_F16
        }
    }

    ଫେରନ୍ତୁ sum;
}

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା D, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nthreads, ପୂର୍ଣ୍ଣସଂଖ୍ୟା warp_size>
ସ୍ଥିର __dpct_inline__ ଦଶମିକ vec_dot_fattn_vec_KQ_q4_0(ଅଚଳ ଅକ୍ଷର * __restrict__ K_c,
                                                       ଅଚଳ ଶୂନ୍ୟ * __restrict__ Q_v,
                                                       ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା * __restrict__ Q_q8,
                                                       ଅଚଳ ଶୂନ୍ୟ * __restrict__ Q_ds_v) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    ଅଚଳ block_q4_0 * K_q4_0   = (ଅଚଳ block_q4_0 *) K_c;
    GGML_UNUSED(Q_v);

    ଦଶମିକ sum = 0.0f;

#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k_KQ_0 = 0; k_KQ_0 < ପୂର୍ଣ୍ଣସଂଖ୍ୟା(D/ଆକାର(ପୂର୍ଣ୍ଣସଂଖ୍ୟା)); k_KQ_0 += nthreads) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା k_KQ =
            k_KQ_0 + (nthreads == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads);

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ib    = k_KQ /  QI8_1;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iqs4  = k_KQ %  QI4_0;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା shift = k_KQ & (QI8_1/2);

        ପୂର୍ଣ୍ଣସଂଖ୍ୟା v;
        ggml_sycl_memcpy_1<ଆକାର(ପୂର୍ଣ୍ଣସଂଖ୍ୟା), 2>(&v, K_q4_0[ib].qs + ଆକାର(ପୂର୍ଣ୍ଣସଂଖ୍ୟା)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା u = Q_q8[k_KQ_0/nthreads];

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା sumi = ggml_sycl_dp4a(v, u, 0);

        ଅଚଳ sycl::float2 Q_ds = ((ଅଚଳ sycl::float2 *) Q_ds_v)[k_KQ_0 / nthreads];
        sum += __half2float(K_q4_0[ib].d) * (sumi*Q_ds.x() - (8/QI8_1)*Q_ds.y());
    }

    ଫେରନ୍ତୁ sum;
}

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା D, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nthreads , ପୂର୍ଣ୍ଣସଂଖ୍ୟା warp_size>
ସ୍ଥିର __dpct_inline__ ଦଶମିକ vec_dot_fattn_vec_KQ_q4_1(ଅଚଳ ଅକ୍ଷର * __restrict__ K_c,
                                                       ଅଚଳ ଶୂନ୍ୟ * __restrict__ Q_v,
                                                       ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା * __restrict__ Q_q8,
                                                       ଅଚଳ ଶୂନ୍ୟ * __restrict__ Q_ds_v) {
    auto               item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ଅଚଳ block_q4_1 * K_q4_1   = (ଅଚଳ block_q4_1 *) K_c;
    GGML_UNUSED(Q_v);

    ଦଶମିକ sum = 0.0f;

#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k_KQ_0 = 0; k_KQ_0 < ପୂର୍ଣ୍ଣସଂଖ୍ୟା(D/ଆକାର(ପୂର୍ଣ୍ଣସଂଖ୍ୟା)); k_KQ_0 += nthreads) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା k_KQ =
            k_KQ_0 + (nthreads == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads);

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ib    = k_KQ /  QI8_1;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iqs4  = k_KQ %  QI4_1;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା shift = k_KQ & (QI8_1/2);

        ପୂର୍ଣ୍ଣସଂଖ୍ୟା v;
        ggml_sycl_memcpy_1<ଆକାର(ପୂର୍ଣ୍ଣସଂଖ୍ୟା)>(&v, K_q4_1[ib].qs + ଆକାର(ପୂର୍ଣ୍ଣସଂଖ୍ୟା)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା u = Q_q8[k_KQ_0/nthreads];

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା sumi = ggml_sycl_dp4a(v, u, 0);

        ଅଚଳ sycl::float2 K_dm = (K_q4_1[ib].dm).template convert<ଦଶମିକ, sycl::rounding_mode::automatic>();
        ଅଚଳ sycl::float2 Q_ds = ((ଅଚଳ sycl::float2 *) Q_ds_v)[k_KQ_0 / nthreads];

        sum += K_dm.x()*Q_ds.x()*sumi + K_dm.y()*Q_ds.y()/QI8_1;
    }

    ଫେରନ୍ତୁ sum;
}

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା D, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nthreads, ପୂର୍ଣ୍ଣସଂଖ୍ୟା warp_size>
ସ୍ଥିର __dpct_inline__ ଦଶମିକ vec_dot_fattn_vec_KQ_q5_0(ଅଚଳ ଅକ୍ଷର * __restrict__ K_c,
                                                       ଅଚଳ ଶୂନ୍ୟ * __restrict__ Q_v,
                                                       ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା * __restrict__ Q_q8,
                                                       ଅଚଳ ଶୂନ୍ୟ * __restrict__ Q_ds_v) {
    auto               item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ଅଚଳ block_q5_0 * K_q5_0   = (ଅଚଳ block_q5_0 *) K_c;
    GGML_UNUSED(Q_v);

    ଦଶମିକ sum = 0.0f;

#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k_KQ_0 = 0; k_KQ_0 < ପୂର୍ଣ୍ଣସଂଖ୍ୟା(D/ଆକାର(ପୂର୍ଣ୍ଣସଂଖ୍ୟା)); k_KQ_0 += nthreads) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା k_KQ =
            k_KQ_0 + (nthreads == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads);

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ib    = k_KQ /  QI8_1;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iqs4  = k_KQ %  QI5_0;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iqs8  = k_KQ %  QI8_1;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା shift = k_KQ & (QI8_1/2);

        ପୂର୍ଣ୍ଣସଂଖ୍ୟା v;
        ggml_sycl_memcpy_1<ଆକାର(ପୂର୍ଣ୍ଣସଂଖ୍ୟା), 2>(&v, K_q5_0[ib].qs + ଆକାର(ପୂର୍ଣ୍ଣସଂଖ୍ୟା)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;

        {
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା vh;
            ggml_sycl_memcpy_1<ଆକାର(ପୂର୍ଣ୍ଣସଂଖ୍ୟା), 2>(&vh, K_q5_0[ib].qh);
            vh >>= iqs8 * QI5_0;

            v |= (vh <<  4) & 0x00000010; // 0 ->  4
            v |= (vh << 11) & 0x00001000; // 1 -> 12
            v |= (vh << 18) & 0x00100000; // 2 -> 20
            v |= (vh << 25) & 0x10000000; // 3 -> 28
        }

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା u = Q_q8[k_KQ_0/nthreads];

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା sumi = ggml_sycl_dp4a(v, u, 0);

        ଅଚଳ sycl::float2 Q_ds = ((ଅଚଳ sycl::float2 *) Q_ds_v)[k_KQ_0 / nthreads];

        sum += __half2float(K_q5_0[ib].d) * (sumi*Q_ds.x() - (16/QI8_1)*Q_ds.y());
    }

    ଫେରନ୍ତୁ sum;
}

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା D, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nthreads, ପୂର୍ଣ୍ଣସଂଖ୍ୟା warp_size>
ସ୍ଥିର __dpct_inline__ ଦଶମିକ vec_dot_fattn_vec_KQ_q5_1(ଅଚଳ ଅକ୍ଷର * __restrict__ K_c,
                                                       ଅଚଳ ଶୂନ୍ୟ * __restrict__ Q_v,
                                                       ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା * __restrict__ Q_q8,
                                                       ଅଚଳ ଶୂନ୍ୟ * __restrict__ Q_ds_v) {
    auto               item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ଅଚଳ block_q5_1 * K_q5_1   = (ଅଚଳ block_q5_1 *) K_c;
    GGML_UNUSED(Q_v);

    ଦଶମିକ sum = 0.0f;

#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k_KQ_0 = 0; k_KQ_0 < ପୂର୍ଣ୍ଣସଂଖ୍ୟା(D/ଆକାର(ପୂର୍ଣ୍ଣସଂଖ୍ୟା)); k_KQ_0 += nthreads) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା k_KQ =
            k_KQ_0 + (nthreads == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads);

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ib    = k_KQ /  QI8_1;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iqs4  = k_KQ %  QI5_1;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iqs8  = k_KQ %  QI8_1;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା shift = k_KQ & (QI8_1/2);

        ପୂର୍ଣ୍ଣସଂଖ୍ୟା v;
        ggml_sycl_memcpy_1<ଆକାର(ପୂର୍ଣ୍ଣସଂଖ୍ୟା)>(&v, K_q5_1[ib].qs + ଆକାର(ପୂର୍ଣ୍ଣସଂଖ୍ୟା)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;

        {
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା vh;
            ggml_sycl_memcpy_1<ଆକାର(ପୂର୍ଣ୍ଣସଂଖ୍ୟା)>(&vh, K_q5_1[ib].qh);
            vh >>= iqs8 * QI5_0;

            v |= (vh <<  4) & 0x00000010; // 0 ->  4
            v |= (vh << 11) & 0x00001000; // 1 -> 12
            v |= (vh << 18) & 0x00100000; // 2 -> 20
            v |= (vh << 25) & 0x10000000; // 3 -> 28
        }

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା u = Q_q8[k_KQ_0/nthreads];

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା sumi = ggml_sycl_dp4a(v, u, 0);

        ଅଚଳ sycl::float2 K_dm = (K_q5_1[ib].dm).template convert<ଦଶମିକ, sycl::rounding_mode::automatic>();
        ଅଚଳ sycl::float2 Q_ds = ((ଅଚଳ sycl::float2 *) Q_ds_v)[k_KQ_0 / nthreads];

        sum += K_dm.x()*Q_ds.x()*sumi + K_dm.y()*Q_ds.y()/QI8_1;
    }

    ଫେରନ୍ତୁ sum;
}

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା D, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nthreads, ପୂର୍ଣ୍ଣସଂଖ୍ୟା warp_size>
ସ୍ଥିର __dpct_inline__ ଦଶମିକ vec_dot_fattn_vec_KQ_q8_0(ଅଚଳ ଅକ୍ଷର * __restrict__ K_c,
                                                       ଅଚଳ ଶୂନ୍ୟ * __restrict__ Q_v,
                                                       ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା * __restrict__ Q_q8,
                                                       ଅଚଳ ଶୂନ୍ୟ * __restrict__ Q_ds_v) {
    auto               item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ଅଚଳ block_q8_0 * K_q8_0   = (ଅଚଳ block_q8_0 *) K_c;
    GGML_UNUSED(Q_v);

    ଦଶମିକ sum = 0.0f;

#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k_KQ_0 = 0; k_KQ_0 < ପୂର୍ଣ୍ଣସଂଖ୍ୟା(D/ଆକାର(ପୂର୍ଣ୍ଣସଂଖ୍ୟା)); k_KQ_0 += nthreads) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା k_KQ =
            k_KQ_0 + (nthreads == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads);

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ib  = k_KQ / QI8_0;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iqs = k_KQ % QI8_0;

        ପୂର୍ଣ୍ଣସଂଖ୍ୟା v;
        ggml_sycl_memcpy_1<ଆକାର(v), 2>(&v, K_q8_0[ib].qs + 4*iqs);

        ଅଚଳ sycl::float2 * Q_ds = (ଅଚଳ sycl::float2 *) Q_ds_v;
        ଅଚଳ ଦଶମିକ          Q_d  = Q_ds[k_KQ_0 / nthreads].x();

        sum += vec_dot_q8_0_q8_1_impl<ଦଶମିକ, 1>(&v, &Q_q8[k_KQ_0/nthreads], K_q8_0[ib].d, Q_d);
    }

    ଫେରନ୍ତୁ sum;
}

template <typename Tds, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ni, ପୂର୍ଣ୍ଣସଂଖ୍ୟା warp_size>
ସ୍ଥିର __dpct_inline__ ଶୂନ୍ୟ quantize_q8_1_to_shared(ଅଚଳ ଦଶମିକ * __restrict__ x,
                                                    ଅଚଳ ଦଶମିକ scale,
                                                    ପୂର୍ଣ୍ଣସଂଖ୍ୟା * __restrict__ yq32,
                                                    ଶୂନ୍ୟ * __restrict__ yds) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    ଦଶମିକ vals[ଆକାର(ପୂର୍ଣ୍ଣସଂଖ୍ୟା)] = { 0.0f };
#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା l = 0; l < ପୂର୍ଣ୍ଣସଂଖ୍ୟା(ଆକାର(ପୂର୍ଣ୍ଣସଂଖ୍ୟା)); ++l) {
        vals[l] =
            (ni == warp_size || item_ct1.get_local_id(2) < ni) ? scale * x[4 * item_ct1.get_local_id(2) + l] : 0.0f;
    }

    ଦଶମିକ amax = sycl::fabs(vals[0]);
    ଦଶମିକ sum  = vals[0];
#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା l = 1; l < ପୂର୍ଣ୍ଣସଂଖ୍ୟା(ଆକାର(ପୂର୍ଣ୍ଣସଂଖ୍ୟା)); ++l) {
        amax = sycl::fmax(amax, sycl::fabs(vals[l]));
        sum += vals[l];
    }
#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା mask = QI8_1/2; mask > 0; mask >>= 1) {
        amax = sycl::fmax(
            amax, dpct::permute_sub_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(), amax, mask));
        sum += dpct::permute_sub_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(), sum, mask);
    }

    ଅଚଳ ଦଶମିକ d = amax / 127;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା q32 = 0;
    int8_t * q8 = (int8_t *) &q32;

    ଯଦି (d != 0.0f) {
#pragma unroll
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା l = 0; l < ପୂର୍ଣ୍ଣସଂଖ୍ୟା(ଆକାର(ପୂର୍ଣ୍ଣସଂଖ୍ୟା)); ++l) {
            q8[l] = sycl::round(vals[l] / d);
        }
    }

    yq32[item_ct1.get_local_id(2)] = q32;
    ଯଦି (item_ct1.get_local_id(2) % QI8_1 == 0 && (ni == warp_size || item_ct1.get_local_id(2) < ni)) {
        ଯଦି (std::is_same<Tds, sycl::half2>::value) {
            ((sycl::half2  *) yds)[item_ct1.get_local_id(2)/QI8_1] =  make_half2(d, sum);
        } ଅନ୍ୟଥା {
            ((sycl::float2 *) yds)[item_ct1.get_local_id(2)/QI8_1] = make_float2(d, sum);
        }
    }
}

ପ୍ରକାରବ୍ୟାଖ୍ୟା ଶୂନ୍ୟ (*dequantize_V_t)(ଅଚଳ ଶୂନ୍ୟ *, ଶୂନ୍ୟ *, ଅଚଳ int64_t);

template <typename T, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne>
ସ୍ଥିର __dpct_inline__ ଶୂନ୍ୟ dequantize_V_f16(ଅଚଳ ଶୂନ୍ୟ * __restrict__ vx, ଶୂନ୍ୟ * __restrict__ dst, ଅଚଳ int64_t i0) {
    ଯଦି constexpr (std::is_same_v<T, sycl::half>) {
        ggml_sycl_memcpy_1<ne * ଆକାର(sycl::half)>(dst, (ଅଚଳ sycl::half *) vx + i0);
    } ଅନ୍ୟଥା ଯଦି constexpr (std::is_same_v<T, ଦଶମିକ>) {
        static_assert(ne % 2 == 0, "bad ne");
        sycl::half2 tmp[ne / 2];
        ggml_sycl_memcpy_1<ne * ଆକାର(sycl::half)>(tmp, (ଅଚଳ sycl::half *) vx + i0);
        sycl::float2 * dst_f2 = (sycl::float2 *) dst;
#pragma unroll
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା l = 0; l < ne/2; ++l) {
            dst_f2[l] = tmp[l].template convert<ଦଶମିକ, sycl::rounding_mode::automatic>();
        }
    } ଅନ୍ୟଥା {
        static_assert(std::is_same_v<T, ଶୂନ୍ୟ>, "unsupported type");
    }
}

template <typename T, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne>
ସ୍ଥିର __dpct_inline__ ଶୂନ୍ୟ dequantize_V_q4_0(ଅଚଳ ଶୂନ୍ୟ * __restrict__ vx, ଶୂନ୍ୟ * __restrict__ dst, ଅଚଳ int64_t i0) {
    ଅଚଳ block_q4_0 * x = (ଅଚଳ block_q4_0 *) vx;

    ଅଚଳ int64_t ib    =  i0          /  QK4_0;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା     iqs   =  i0          % (QK4_0/2);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା     shift = (i0 % QK4_0) / (QK4_0/2);

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_sycl_memcpy_1<ne, 2>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;
    q = dpct::vectorized_ଦ୍ୱିମିକ<sycl::char4>(q, 0x08080808, dpct::sub_sat());

    ଅଚଳ int8_t * q8 = (ଅଚଳ int8_t *) &q;

#ifdef GGML_SYCL_F16
    ଯଦି constexpr (std::is_same_v<T, sycl::half>) {
        ଅଚଳ sycl::half2 d = sycl::half2(x[ib].d);

#pragma unroll
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * sycl::half2(q8[l0 + 0], q8[l0 + 1]);
        }
    } ଅନ୍ୟଥା
#endif // GGML_SYCL_F16
    ଯଦି constexpr (std::is_same_v<T, ଦଶମିକ>) {
        ଅଚଳ ଦଶମିକ d = x[ib].d;

#pragma unroll
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା l = 0; l < ne; ++l) {
            ((ଦଶମିକ *) dst)[l] = d * q8[l];
        }
    } ଅନ୍ୟଥା {
        static_assert(std::is_same_v<T, ଶୂନ୍ୟ>, "bad type");
    }
}

template <typename T, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne>
ସ୍ଥିର __dpct_inline__ ଶୂନ୍ୟ dequantize_V_q4_1(ଅଚଳ ଶୂନ୍ୟ * __restrict__ vx, ଶୂନ୍ୟ * __restrict__ dst, ଅଚଳ int64_t i0) {
    ଅଚଳ block_q4_1 * x = (ଅଚଳ block_q4_1 *) vx;

    ଅଚଳ int64_t ib    =  i0          /  QK4_1;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା     iqs   =  i0          % (QK4_1/2);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା     shift = (i0 % QK4_1) / (QK4_1/2);

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_sycl_memcpy_1<ne>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    ଅଚଳ int8_t * q8 = (ଅଚଳ int8_t *) &q;

#ifdef GGML_SYCL_F16
    ଯଦି constexpr (std::is_same_v<T, sycl::half>) {
        ଅଚଳ sycl::half2 dm = x[ib].dm;
        ଅଚଳ sycl::half2 d  = sycl::half2(dm[0]);
        ଅଚଳ sycl::half2 m  = sycl::half2(dm[1]);

#pragma unroll
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * sycl::half2(q8[l0 + 0], q8[l0 + 1]) + m;
        }
    } ଅନ୍ୟଥା
#endif // GGML_SYCL_F16
    ଯଦି constexpr (std::is_same_v<T, ଦଶମିକ>) {
        ଅଚଳ sycl::float2 dm = (x[ib].dm).template convert<ଦଶମିକ, sycl::rounding_mode::automatic>();

#pragma unroll
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା l = 0; l < ne; ++l) {
            ((ଦଶମିକ *) dst)[l] = dm.x() * q8[l] + dm.y();
        }
    } ଅନ୍ୟଥା {
        static_assert(std::is_same_v<T, ଶୂନ୍ୟ>, "bad type");
    }
}

template <typename T, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne>
ସ୍ଥିର __dpct_inline__ ଶୂନ୍ୟ dequantize_V_q5_0(ଅଚଳ ଶୂନ୍ୟ * __restrict__ vx, ଶୂନ୍ୟ * __restrict__ dst, ଅଚଳ int64_t i0) {
    ଅଚଳ block_q5_0 * x = (ଅଚଳ block_q5_0 *) vx;

    ଅଚଳ int64_t ib    =  i0          /  QK5_0;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା     idq   =  i0          %  QK5_0;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା     iqs   =  i0          % (QK5_0/2);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା     shift = (i0 % QK5_0) / (QK5_0/2);

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_sycl_memcpy_1<ne, 2>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    {
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା qh;
        ggml_sycl_memcpy_1<ne, 2>(&qh, x[ib].qh);
#pragma unroll
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା l = 0; l < ne; ++l) {
            q |= ((qh >> (idq + l)) & 0x00000001) << (8*l + 4);
        }
    }

    q = dpct::vectorized_ଦ୍ୱିମିକ<sycl::char4>(q, 0x10101010, dpct::sub_sat());

    ଅଚଳ int8_t * q8 = (ଅଚଳ int8_t *) &q;

#ifdef GGML_SYCL_F16
    ଯଦି constexpr (std::is_same_v<T, sycl::half>) {
        ଅଚଳ sycl::half2 d = sycl::half2(x[ib].d);

#pragma unroll
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * sycl::half2(q8[l0 + 0], q8[l0 + 1]);
        }
    } ଅନ୍ୟଥା
#endif // GGML_SYCL_F16
    ଯଦି constexpr (std::is_same_v<T, ଦଶମିକ>) {
        ଅଚଳ ଦଶମିକ d = x[ib].d;

#pragma unroll
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା l = 0; l < ne; ++l) {
            ((ଦଶମିକ *) dst)[l] = d * q8[l];
        }
    } ଅନ୍ୟଥା {
        static_assert(std::is_same_v<T, ଶୂନ୍ୟ>, "bad type");
    }
}

template <typename T, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne>
ସ୍ଥିର __dpct_inline__ ଶୂନ୍ୟ dequantize_V_q5_1(ଅଚଳ ଶୂନ୍ୟ * __restrict__ vx, ଶୂନ୍ୟ * __restrict__ dst, ଅଚଳ int64_t i0) {
    ଅଚଳ block_q5_1 * x = (ଅଚଳ block_q5_1 *) vx;

    ଅଚଳ int64_t ib    =  i0          /  QK5_1;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା     idq   =  i0          %  QK5_1;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା     iqs   =  i0          % (QK5_1/2);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା     shift = (i0 % QK5_1) / (QK5_1/2);

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_sycl_memcpy_1<ne>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    {
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା qh;
        ggml_sycl_memcpy_1<ne>(&qh, x[ib].qh);
#pragma unroll
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା l = 0; l < ne; ++l) {
            q |= ((qh >> (idq + l)) & 0x00000001) << (8*l + 4);
        }
    }

    ଅଚଳ int8_t * q8 = (ଅଚଳ int8_t *) &q;

#ifdef GGML_SYCL_F16
    ଯଦି constexpr (std::is_same_v<T, sycl::half>) {
        ଅଚଳ sycl::half2 dm = x[ib].dm;
        ଅଚଳ sycl::half2 d  = sycl::half2(dm[0]);
        ଅଚଳ sycl::half2 m  = sycl::half2(dm[1]);

#pragma unroll
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * sycl::half2(q8[l0 + 0], q8[l0 + 1]) + m;
        }
    } ଅନ୍ୟଥା
#endif // GGML_SYCL_F16
    ଯଦି constexpr (std::is_same_v<T, ଦଶମିକ>) {
        ଅଚଳ sycl::float2 dm = (x[ib].dm).template convert<ଦଶମିକ, sycl::rounding_mode::automatic>();

#pragma unroll
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା l = 0; l < ne; ++l) {
            ((ଦଶମିକ *) dst)[l] = dm.x() * q8[l] + dm.y();
        }
    } ଅନ୍ୟଥା {
        static_assert(std::is_same_v<T, ଶୂନ୍ୟ>, "bad type");
    }
}

template <typename T, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne>
ସ୍ଥିର __dpct_inline__ ଶୂନ୍ୟ dequantize_V_q8_0(ଅଚଳ ଶୂନ୍ୟ * __restrict__ vx, ଶୂନ୍ୟ * __restrict__ dst, ଅଚଳ int64_t i0) {
    ଅଚଳ block_q8_0 * x = (ଅଚଳ block_q8_0 *) vx;

    ଅଚଳ int64_t ib  = i0 / QK8_0;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା     iqs = i0 % QK8_0;

    static_assert(ne % 2 == 0, "bad ne");
    int8_t qs[ne];
    ggml_sycl_memcpy_1<ne, 2>(qs, x[ib].qs + iqs);

#ifdef GGML_SYCL_F16
    ଯଦି constexpr (std::is_same<T, sycl::half>::value) {
        ଅଚଳ sycl::half2 d = sycl::half2(x[ib].d);

#pragma unroll
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * make_half2(qs[l0 + 0], qs[l0 + 1]);
        }
    } ଅନ୍ୟଥା
#endif // GGML_SYCL_F16
    ଯଦି constexpr (std::is_same<T, ଦଶମିକ>::value) {
        ଅଚଳ ଦଶମିକ d = x[ib].d;

#pragma unroll
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା l = 0; l < ne; ++l) {
            ((ଦଶମିକ *) dst)[l] = d * qs[l];
        }
    } ଅନ୍ୟଥା {
        static_assert(std::is_same_v<T, ଶୂନ୍ୟ>, "unsupported type");
    }
}

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା type_K, ପୂର୍ଣ୍ଣସଂଖ୍ୟା D, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nthreads, ପୂର୍ଣ୍ଣସଂଖ୍ୟା warp_size>
constexpr vec_dot_KQ_t get_vec_dot_KQ() {
    ଯଦି constexpr (type_K == GGML_TYPE_F16) {
        ଫେରନ୍ତୁ vec_dot_fattn_vec_KQ_f16<D, nthreads>;
    } ଅନ୍ୟଥା ଯଦି constexpr (type_K == GGML_TYPE_Q4_0) {
        ଫେରନ୍ତୁ vec_dot_fattn_vec_KQ_q4_0<D, nthreads, warp_size>;
    } ଅନ୍ୟଥା ଯଦି constexpr (type_K == GGML_TYPE_Q4_1) {
        ଫେରନ୍ତୁ vec_dot_fattn_vec_KQ_q4_1<D, nthreads, warp_size>;
    } ଅନ୍ୟଥା ଯଦି constexpr (type_K == GGML_TYPE_Q5_0) {
        ଫେରନ୍ତୁ vec_dot_fattn_vec_KQ_q5_0<D, nthreads, warp_size>;
    } ଅନ୍ୟଥା ଯଦି constexpr (type_K == GGML_TYPE_Q5_1) {
        ଫେରନ୍ତୁ vec_dot_fattn_vec_KQ_q5_1<D, nthreads, warp_size>;
    } ଅନ୍ୟଥା ଯଦି constexpr (type_K == GGML_TYPE_Q8_0) {
        ଫେରନ୍ତୁ vec_dot_fattn_vec_KQ_q8_0<D, nthreads, warp_size>;
    } ଅନ୍ୟଥା {
        static_assert(type_K == -1, "bad type");
        ଫେରନ୍ତୁ nullptr;
    }
}

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା type_V, typename T, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne>
constexpr dequantize_V_t get_dequantize_V() {
    ଯଦି constexpr (type_V == GGML_TYPE_F16) {
        ଫେରନ୍ତୁ dequantize_V_f16<T, ne>;
    } ଅନ୍ୟଥା ଯଦି constexpr (type_V == GGML_TYPE_Q4_0) {
        ଫେରନ୍ତୁ dequantize_V_q4_0<T, ne>;
    } ଅନ୍ୟଥା ଯଦି constexpr (type_V == GGML_TYPE_Q4_1) {
        ଫେରନ୍ତୁ dequantize_V_q4_1<T, ne>;
    } ଅନ୍ୟଥା ଯଦି constexpr (type_V == GGML_TYPE_Q5_0) {
        ଫେରନ୍ତୁ dequantize_V_q5_0<T, ne>;
    } ଅନ୍ୟଥା ଯଦି constexpr (type_V == GGML_TYPE_Q5_1) {
        ଫେରନ୍ତୁ dequantize_V_q5_1<T, ne>;
    } ଅନ୍ୟଥା ଯଦି constexpr (type_V == GGML_TYPE_Q8_0) {
        ଫେରନ୍ତୁ dequantize_V_q8_0<T, ne>;
    } ଅନ୍ୟଥା {
        static_assert(type_V == -1, "bad type");
        ଫେରନ୍ତୁ nullptr;
    }
}

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols1, ପୂର୍ଣ୍ଣସଂଖ୍ୟା warp_size>
ସ୍ଥିର ଶୂନ୍ୟ flash_attn_mask_to_KV_max(ଅଚଳ sycl::half2 * __restrict__ mask,
                                      ପୂର୍ଣ୍ଣସଂଖ୍ୟା * __restrict__ KV_max,
                                      ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne30,
                                      ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s31,
                                      ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s33,
                                      ପୂର୍ଣ୍ଣସଂଖ୍ୟା *     buf_iw) {
    auto      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne31     = item_ct1.get_group_range(2);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା tid      = item_ct1.get_local_id(2);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା sequence = item_ct1.get_group(1);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା jt       = item_ct1.get_group(2);

    mask += sequence*s33 + jt*ncols1*s31;

    ଯଦି (tid < warp_size) {
        buf_iw[tid] = 1;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା KV_max_sj = (ne30 - 1) * FATTN_KQ_STRIDE;
    ପାଇଁ (; KV_max_sj >= 0; KV_max_sj -= FATTN_KQ_STRIDE) {
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା all_inf = 1;

#pragma unroll
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < ncols1; ++j) {
            ଅଚଳ sycl::float2 tmp =
                mask[j * s31 + KV_max_sj / 2 + tid].template convert<ଦଶମିକ, sycl::rounding_mode::automatic>();
            all_inf = all_inf && ପୂର୍ଣ୍ଣସଂଖ୍ୟା(sycl::isinf((ଦଶମିକ) (tmp.x()))) && ପୂର୍ଣ୍ଣସଂଖ୍ୟା(sycl::isinf((ଦଶମିକ) (tmp.y())));
        }

        all_inf = warp_reduce_all<warp_size>(all_inf);
        ଯଦି (tid % warp_size == 0) {
            buf_iw[tid / warp_size] = all_inf;
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);
        all_inf = buf_iw[tid % warp_size];
        item_ct1.barrier(sycl::access::fence_space::local_space);
        all_inf = warp_reduce_all<warp_size>(all_inf);

        ଯଦି (!all_inf) {
            ଭାଙ୍ଗ;
        }
    }

    // If the break in the loop was not triggered, KV_max_sj is now -FATTN_KQ_STRIDE.
    // If the break was triggered it's the lower edge of the tile with the first non-masked values.
    // In either case, walk back the decrementation by FATTN_KQ_STRIDE.
    KV_max_sj += FATTN_KQ_STRIDE;

    ଯଦି (item_ct1.get_local_id(2) != 0) {
        ଫେରନ୍ତୁ;
    }

    KV_max[sequence*ne31 + jt] = KV_max_sj;
}

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା D, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols1, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols2>  // D == head size

ସ୍ଥିର ଶୂନ୍ୟ flash_attn_stream_k_fixup(ଦଶମିକ * __restrict__ dst,
                                      ଅଚଳ sycl::float2 * __restrict__ dst_fixup,
                                      ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01,
                                      ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02,
                                      ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne03,
                                      ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11,
                                      ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne12,
                                      ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nbatch_fa) {
    auto          item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols    = ncols1 * ncols2;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା bidx0 = item_ct1.get_group(2);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା j     = item_ct1.get_group(1);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା c     = item_ct1.get_group(0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା jc    = j*ncols2 + c;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା tid   = item_ct1.get_local_id(2);

    ଅଚଳ ଦଶମିକ * dst_fixup_data = ((ଅଚଳ ଦଶମିକ *) dst_fixup) + item_ct1.get_group_range(2) * (2 * 2 * ncols);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iter_k     = (ne11      + (nbatch_fa - 1)) / nbatch_fa;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iter_j     = (ne01      + (ncols1    - 1)) / ncols1;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iter_z_gqa = (gqa_ratio + (ncols2    - 1)) / ncols2;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା kbc0 = int64_t(bidx0 + 0) * (iter_k * iter_j * iter_z_gqa * ne12 * ne03) / item_ct1.get_group_range(2);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା kbc0_stop =
        int64_t(bidx0 + 1) * (iter_k * iter_j * iter_z_gqa * ne12 * ne03) / item_ct1.get_group_range(2);

    ଅଚଳ ତାର୍କିକ did_not_have_any_data   = kbc0 == kbc0_stop;
    ଅଚଳ ତାର୍କିକ wrote_beginning_of_tile = kbc0 % iter_k == 0;
    ଅଚଳ ତାର୍କିକ did_not_write_last      = kbc0/iter_k == kbc0_stop/iter_k && kbc0_stop % iter_k != 0;
    ଯଦି (did_not_have_any_data || wrote_beginning_of_tile || did_not_write_last) {
        ଫେରନ୍ତୁ;
    }

    // z_KV == K/V head index, zt_gqa = Q head start index per K/V head, jt = token position start index
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା sequence =  kbc0 /(iter_k*iter_j*iter_z_gqa*ne12);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା z_KV     = (kbc0 - iter_k*iter_j*iter_z_gqa*ne12 * sequence)/(iter_k*iter_j*iter_z_gqa);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା zt_gqa   = (kbc0 - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV)/(iter_k*iter_j);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା jt       = (kbc0 - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV - iter_k*iter_j * zt_gqa) / iter_k;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା zt_Q = z_KV*gqa_ratio + zt_gqa*ncols2; // Global Q head start index.

    ଯଦି (jt*ncols1 + j >= ne01 || zt_gqa*ncols2 + c >= gqa_ratio) {
        ଫେରନ୍ତୁ;
    }

    dst += sequence*ne02*ne01*D + jt*ne02*(ncols1*D) + zt_Q*D + (j*ne02 + c)*D + tid;

    // Load the partial result that needs a fixup:
    ଦଶମିକ dst_val = 0.0f;
    ଦଶମିକ max_val = 0.0f;
    ଦଶମିକ rowsum  = 0.0f;
    {
        dst_val = *dst;

        ଅଚଳ sycl::float2 tmp = dst_fixup[bidx0 * ncols + jc];
        max_val                = tmp.x();
        rowsum                 = tmp.y();
    }

    // Iterate over previous blocks and compute the combined results.
    // All SYCL blocks that get here must have a previous block that needs a fixup.
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା bidx = bidx0 - 1;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା kbc_stop = kbc0;
    ଯେତେବେଳେ(ସତ୍ୟ) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା kbc = int64_t(bidx) * (iter_k * iter_j * iter_z_gqa * ne12 * ne03) / item_ct1.get_group_range(2);
        ଯଦି (kbc == kbc_stop) { // Did not have any data.
            bidx--;
            kbc_stop = kbc;
            ଜାରି;
        }

        ଅଚଳ ଦଶମିକ dst_add = dst_fixup_data[bidx*ncols*D + jc*D + tid];

        ଅଚଳ sycl::float2 tmp = dst_fixup[(item_ct1.get_group_range(2) + bidx) * ncols + jc];

        // Scale the current and new value accumulators depending on the max. values.
        ଅଚଳ ଦଶମିକ max_val_new = sycl::fmax(max_val, tmp.x());

        ଅଚଳ ଦଶମିକ diff_val = max_val - max_val_new;
        ଅଚଳ ଦଶମିକ diff_add = tmp.x() - max_val_new;

        ଅଚଳ ଦଶମିକ scale_val = diff_val >= SOFTMAX_FTZ_THRESHOLD ? sycl::native::exp(diff_val) : 0.0f;
        ଅଚଳ ଦଶମିକ scale_add = diff_add >= SOFTMAX_FTZ_THRESHOLD ? sycl::native::exp(diff_add) : 0.0f;

        dst_val = scale_val*dst_val + scale_add*dst_add;
        rowsum  = scale_val * rowsum + scale_add * tmp.y();

        max_val = max_val_new;

        // If this block started in a previous tile we are done and don't need to combine additional partial results.
        ଯଦି (kbc % iter_k == 0 || kbc/iter_k < kbc0/iter_k) {
            ଭାଙ୍ଗ;
        }
        bidx--;
        kbc_stop = kbc;
    }

    // Write back final result:
    *dst = dst_val / rowsum;
}

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା D>  // D == head size

ସ୍ଥିର ଶୂନ୍ୟ flash_attn_combine_results(ଅଚଳ ଦଶମିକ * __restrict__ VKQ_parts,
                                       ଅଚଳ sycl::float2 * __restrict__ VKQ_meta,
                                       ଦଶମିକ * __restrict__ dst,
                                       ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା parallel_blocks,
                                       uint8_t * dpct_local) {
    // Dimension 0: threadIdx.x
    // Dimension 1: blockIdx.x
    // Dimension 2: blockIdx.y
    // Dimension 3: blockIdx.z
    // Memory layout is permuted with [0, 2, 1, 3]

    auto      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01     = item_ct1.get_group_range(2);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02     = item_ct1.get_group_range(1);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା col      = item_ct1.get_group(2);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା head     = item_ct1.get_group(1);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା sequence = item_ct1.get_group(0);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା j_dst_unrolled = (sequence*ne01 + col)*ne02 + head;

    VKQ_parts += j_dst_unrolled * parallel_blocks*D;
    VKQ_meta  += j_dst_unrolled * parallel_blocks;
    dst       += j_dst_unrolled *                 D;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା tid = item_ct1.get_local_id(2);
    __builtin_assume(tid < D);

    auto meta = (sycl::float2 *) dpct_local;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = tid; i < 2*parallel_blocks; i += D) {
        ((ଦଶମିକ *) meta)[i] = ((ଅଚଳ ଦଶମିକ *)VKQ_meta) [i];
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    ଦଶମିକ kqmax = meta[0].x();
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା l = 1; l < parallel_blocks; ++l) {
        kqmax = sycl::max(kqmax, meta[l].x());
    }

    ଦଶମିକ VKQ_numerator   = 0.0f;
    ଦଶମିକ VKQ_denominator = 0.0f;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା l = 0; l < parallel_blocks; ++l) {
        ଅଚଳ ଦଶମିକ KQ_max_scale = sycl::native::exp(meta[l].x() - kqmax);

        VKQ_numerator   += KQ_max_scale * VKQ_parts[l*D + tid];
        VKQ_denominator += KQ_max_scale * meta[l].y();
    }

    dst[tid] = VKQ_numerator / VKQ_denominator;
}

// Carries the sub-group size (and grf_size_automatic when use_large_grf) as
// kernel properties via get(properties_tag), in place of the
// reqd_sub_group_size attribute and the deprecated parallel_for overload.
template <typename KernelFunc, uint32_t Size, ତାର୍କିକ UseGrf>
ସଂରଚନା props_kernel {
    KernelFunc kernel;

    ଶୂନ୍ୟ operator()(sycl::nd_item<3> item) ଅଚଳ {
        kernel(item);
    }

    auto get(sycl::ext::oneapi::experimental::properties_tag) ଅଚଳ {
        ଯଦି constexpr (UseGrf) {
            ଫେରନ୍ତୁ sycl::ext::oneapi::experimental::properties{
                sycl::ext::intel::experimental::grf_size_automatic,
                sycl::ext::oneapi::experimental::sub_group_size<Size> };
        } ଅନ୍ୟଥା {
            ଫେରନ୍ତୁ sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::sub_group_size<Size> };
        }
    }
};

template <fattn_kernel_t fattn_kernel, ପୂର୍ଣ୍ଣସଂଖ୍ୟା warp_size, ତାର୍କିକ use_large_grf = ମିଥ୍ୟା>
ସ୍ଥିର ଶୂନ୍ୟ lauch_kernel(
    dpct::dim3 group_range,
    dpct::dim3 local_range,
    queue_ptr q,
    unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା local_mem_size,
    ଅଚଳ ଅକ୍ଷର* __restrict__ Q,
    ଅଚଳ ଅକ୍ଷର* __restrict__ K,
    ଅଚଳ ଅକ୍ଷର* __restrict__ V,
    ଅଚଳ ଅକ୍ଷର* __restrict__ mask,
    ଅଚଳ ଅକ୍ଷର* __restrict__ sinks,
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା* __restrict__ KV_max,
    ଦଶମିକ* __restrict__ dst,
    sycl::float2* __restrict__ dst_meta,
    ଅଚଳ ଦଶମିକ scale,
    ଅଚଳ ଦଶମିକ max_bias,
    ଅଚଳ ଦଶମିକ m0,
    ଅଚଳ ଦଶମିକ m1,
    ଅଚଳ uint32_t n_head_log2,
    ଅଚଳ ଦଶମିକ logit_softcap,
    ଅଚଳ int32_t ne00,
    ଅଚଳ sycl::uint3 ne01,
    ଅଚଳ int32_t ne02,
    ଅଚଳ int32_t ne03,
    ଅଚଳ int32_t nb01,
    ଅଚଳ int32_t nb02,
    ଅଚଳ int32_t nb03,
    ଅଚଳ int32_t ne10,
    ଅଚଳ int32_t ne11,
    ଅଚଳ int32_t ne12,
    ଅଚଳ int32_t ne13,
    ଅଚଳ int32_t nb11,
    ଅଚଳ int32_t nb12,
    ଅଚଳ int64_t nb13,
    ଅଚଳ int32_t nb21,
    ଅଚଳ int32_t nb22,
    ଅଚଳ int64_t nb23,
    ଅଚଳ int32_t ne31,
    ଅଚଳ int32_t ne32,
    ଅଚଳ int32_t ne33,
    ଅଚଳ int32_t nb31,
    ଅଚଳ int32_t nb32,
    ଅଚଳ int64_t nb33) {
    GGML_UNUSED(local_mem_size);

    ଅଚଳ auto rng = sycl::nd_range<3>(
        static_cast<sycl::range<3>>(group_range * local_range),
        static_cast<sycl::range<3>>(local_range));

    ଅଚଳ auto kernel = [=](sycl::nd_item<3> item_ct1) {
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
template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା DV, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols1, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols2, fattn_kernel_t fattn_kernel, ପୂର୍ଣ୍ଣସଂଖ୍ୟା warp_size, ତାର୍କିକ use_large_grf = ମିଥ୍ୟା>
ଶୂନ୍ୟ launch_fattn(
    ggml_backend_sycl_context & ctx, ggml_tensor * dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nwarps, ଅଚଳ size_t nbytes_shared,
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nbatch_fa, ଅଚଳ ତାର୍କିକ need_f16_K, ଅଚଳ ତାର୍କିକ need_f16_V, ଅଚଳ ତାର୍କିକ stream_k) {

    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols = ncols1 * ncols2;

    ଅଚଳ ggml_tensor * Q = dst->src[0];
    ଅଚଳ ggml_tensor * K = dst->src[1];
    ଅଚଳ ggml_tensor * V = dst->src[2];

    ଅଚଳ ତାର୍କିକ V_is_K_view = V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));

    ଅଚଳ ggml_tensor * mask  = dst->src[3];
    ଅଚଳ ggml_tensor * sinks = dst->src[4];

    ggml_tensor * KQV = dst;

    GGML_ASSERT(Q->type == GGML_TYPE_F32);
    GGML_ASSERT(KQV->type == GGML_TYPE_F32);

    GGML_ASSERT(Q->nb[0] == ggml_element_size(Q));
    GGML_ASSERT(K->nb[0] == ggml_element_size(K));
    GGML_ASSERT(V->nb[0] == ggml_element_size(V));

    GGML_ASSERT(!mask || mask->type == GGML_TYPE_F16);

    ggml_sycl_pool & pool = ctx.pool();
    dpct::queue_ptr  main_stream = ctx.stream();
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା id  = ggml_sycl_get_device();
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nsm = ggml_sycl_info().devices[id].nsm;

    ଅଚଳ ggml_sycl_fattn_extra  extra = ggml_sycl_fattn_get_extra(dst);
    ggml_sycl_pool_alloc<ପୂର୍ଣ୍ଣସଂଖ୍ୟା>    KV_max(pool);
    ggml_sycl_pool_alloc<ଦଶମିକ>  dst_tmp(pool);
    ggml_sycl_pool_alloc<sycl::float2> dst_tmp_meta(pool);

    ଅଚଳ ଅକ୍ଷର * K_data = (ଅଚଳ ଅକ୍ଷର *) K->data;
    size_t nb11 = K->nb[1];
    size_t nb12 = K->nb[2];
    size_t nb13 = K->nb[3];

    ଅଚଳ ଅକ୍ଷର * V_data = (ଅଚଳ ଅକ୍ଷର *) V->data;
    size_t nb21 = V->nb[1];
    size_t nb22 = V->nb[2];
    size_t nb23 = V->nb[3];

    ଯଦି (need_f16_K && K->type != GGML_TYPE_F16) {
        ଅଚଳ size_t bs = ggml_blck_size(K->type);
        ଅଚଳ size_t ts = ggml_type_size(K->type);

        GGML_ASSERT(extra.K_buffer_ptr);
        sycl::half * K_f16_ptr = (sycl::half *) extra.K_buffer_ptr;
        ଯଦି (ggml_is_contiguously_allocated(K)) {
            to_fp16_sycl_t to_fp16 = ggml_get_to_fp16_sycl(K->type, dst);
            to_fp16(K_data, K_f16_ptr, ggml_nelements(K), main_stream);

            nb11 = nb11 * bs * ଆକାର(sycl::half) / ts;
            nb12 = nb12 * bs * ଆକାର(sycl::half) / ts;
            nb13 = nb13 * bs * ଆକାର(sycl::half) / ts;
        } ଅନ୍ୟଥା {
            GGML_ASSERT(K->nb[0] == ts);
            to_fp16_nc_sycl_t to_fp16 = ggml_get_to_fp16_nc_sycl(K->type);
            ଅଚଳ int64_t s01 = nb11 / ts;
            ଅଚଳ int64_t s02 = nb12 / ts;
            ଅଚଳ int64_t s03 = nb13 / ts;
            to_fp16(K_data, K_f16_ptr, K->ne[0], K->ne[1], K->ne[2], K->ne[3], s01, s02, s03, main_stream);

            nb11 = K->ne[0] * ଆକାର(sycl::half);
            nb12 = K->ne[1] * nb11;
            nb13 = K->ne[2] * nb12;
        }
        K_data = (ଅକ୍ଷର *) K_f16_ptr;
    }

    ଯଦି (need_f16_V && V->type != GGML_TYPE_F16) {
        ଯଦି (V_is_K_view) {
            V_data = K_data;
            nb21   = nb11;
            nb22   = nb12;
            nb23   = nb13;
        } ଅନ୍ୟଥା {
            ଅଚଳ size_t bs = ggml_blck_size(V->type);
            ଅଚଳ size_t ts = ggml_type_size(V->type);

            GGML_ASSERT(extra.V_buffer_ptr);
            sycl::half * V_f16_ptr = (sycl::half *) extra.V_buffer_ptr;
            ଯଦି (ggml_is_contiguously_allocated(V)) {
                to_fp16_sycl_t to_fp16 = ggml_get_to_fp16_sycl(V->type, dst);
                to_fp16(V_data, V_f16_ptr, ggml_nelements(V), main_stream);
                V_data = (ଅକ୍ଷର *) V_f16_ptr;

                nb21 = nb21 * bs * ଆକାର(sycl::half) / ts;
                nb22 = nb22 * bs * ଆକାର(sycl::half) / ts;
                nb23 = nb23 * bs * ଆକାର(sycl::half) / ts;
            } ଅନ୍ୟଥା {
                GGML_ASSERT(V->nb[0] == ts);
                to_fp16_nc_sycl_t to_fp16 = ggml_get_to_fp16_nc_sycl(V->type);
                ଅଚଳ int64_t s01 = nb21 / ts;
                ଅଚଳ int64_t s02 = nb22 / ts;
                ଅଚଳ int64_t s03 = nb23 / ts;
                to_fp16(V_data, V_f16_ptr, V->ne[0], V->ne[1], V->ne[2], V->ne[3], s01, s02, s03, main_stream);

                nb21 = V->ne[0] * ଆକାର(sycl::half);
                nb22 = V->ne[1] * nb21;
                nb23 = V->ne[2] * nb22;
            }
            V_data = (ଅକ୍ଷର *) V_f16_ptr;
        }
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ntiles_x     = ((Q->ne[1] + ncols1 - 1) / ncols1);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା gqa_ratio    = Q->ne[2] / K->ne[2];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ntiles_z_gqa = ((gqa_ratio + ncols2 - 1) / ncols2);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ntiles_total = ntiles_x * ntiles_z_gqa * K->ne[2] * Q->ne[3];

    // Optional optimization where the mask is scanned to determine whether part of the calculation can be skipped.
    // Only worth the overhead if there is at lease one FATTN_KQ_STRIDE x FATTN_KQ_STRIDE square to be skipped or
    //     multiple sequences of possibly different lengths.
    ଯଦି (mask && K->ne[1] % FATTN_KQ_STRIDE == 0 && (Q->ne[1] >= 1024 || Q->ne[3] > 1)) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s31 = mask->nb[1] / ଆକାର(sycl::half2);
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s33 = mask->nb[3] / ଆକାର(sycl::half2);

        ଅଚଳ dpct::dim3 blocks_ଅଙ୍କ_KV_max(ntiles_x, Q->ne[3], 1);
        ଅଚଳ dpct::dim3 block_dim_KV_max(FATTN_KQ_STRIDE / 2, 1, 1);

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne_KV_max = blocks_ଅଙ୍କ_KV_max.x*blocks_ଅଙ୍କ_KV_max.y;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iter_k = K->ne[1] / FATTN_KQ_STRIDE;

        KV_max.alloc(ne_KV_max);
        {
            dpct::has_capability_or_fail(main_stream->get_device(), { sycl::aspect::fp16 });

            main_stream->submit([&](sycl::handler & cgh) {
                sycl::local_accessor<ପୂର୍ଣ୍ଣସଂଖ୍ୟା, 1> buf_iw_acc_ct1(sycl::range<1>(warp_size), cgh);

                auto mask_data_ct0  = (ଅଚଳ sycl::half2 *) mask->data;
                auto KV_max_ptr_ct1 = KV_max.ptr;

                cgh.parallel_for(sycl::nd_range<3>(blocks_ଅଙ୍କ_KV_max * block_dim_KV_max, block_dim_KV_max),
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

    ଅଚଳ dpct::dim3 block_dim(warp_size, nwarps, 1);

    // Max. number of active blocks limited by occupancy.
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା max_blocks_per_sm = ggml_sycl_info().devices[id].max_wg_per_cu;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା parallel_blocks = max_blocks_per_sm;
    dpct::dim3 blocks_ଅଙ୍କ;
    ଯଦି (stream_k) {
        // For short contexts it can be faster to have the SMs work on whole tiles because this lets us skip the fixup.
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା max_blocks = max_blocks_per_sm*nsm;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nblocks_stream_k = max_blocks;
        ଅଚଳ ତାର୍କିକ use_stream_k = ସତ୍ୟ;

        blocks_ଅଙ୍କ.x = use_stream_k ? nblocks_stream_k : ntiles_total;
        blocks_ଅଙ୍କ.y = 1;
        blocks_ଅଙ୍କ.z = 1;

        ଯଦି (ntiles_total % blocks_ଅଙ୍କ.x != 0) { // Fixup is only needed if the SMs work on fractional tiles.
            dst_tmp_meta.alloc((size_t(blocks_ଅଙ୍କ.x) * ncols * (2 + DV/2)));
        }
    } ଅନ୍ୟଥା {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ntiles_KQ = (K->ne[1] + nbatch_fa - 1) / nbatch_fa; // Max. number of parallel blocks limited by tensor size.

        // parallel_blocks must not be larger than what the tensor size allows:
        parallel_blocks = std::min(parallel_blocks, ntiles_KQ);
        // todo fix the hard code change
        // parallel_blocks = ntiles_KQ;

        // If ntiles_total % blocks_per_wave != 0 then some efficiency is lost due to tail effects.
        // Test whether parallel_blocks can be set to a higher value for better efficiency.
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା blocks_per_wave = nsm * max_blocks_per_sm;
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା nwaves_best = 0;
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା efficiency_percent_best = 0;
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା parallel_blocks_test = parallel_blocks; parallel_blocks_test <= ntiles_KQ; ++parallel_blocks_test) {
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nblocks_total = ntiles_total * parallel_blocks_test;
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nwaves = (nblocks_total + blocks_per_wave - 1) / blocks_per_wave;
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା efficiency_percent = 100 * nblocks_total / (nwaves*blocks_per_wave);

            // Stop trying configurations with more waves if we already have good efficiency to avoid excessive overhead.
            ଯଦି (efficiency_percent_best >= 95 && nwaves > nwaves_best) {
                ଭାଙ୍ଗ;
            }

            ଯଦି (efficiency_percent > efficiency_percent_best) {
                nwaves_best = nwaves;
                efficiency_percent_best = efficiency_percent;
                parallel_blocks = parallel_blocks_test;
            }
        }

        blocks_ଅଙ୍କ.x = ntiles_x;
        blocks_ଅଙ୍କ.y = parallel_blocks;
        blocks_ଅଙ୍କ.z = ntiles_z_gqa*K->ne[2]*Q->ne[3];

        ଯଦି (parallel_blocks > 1) {
            dst_tmp.alloc(parallel_blocks*ggml_nelements(KQV));
            dst_tmp_meta.alloc(parallel_blocks*ggml_nrows(KQV));
        }
    }

    ଦଶମିକ scale         = 1.0f;
    ଦଶମିକ max_bias      = 0.0f;
    ଦଶମିକ logit_softcap = 0.0f;

    memcpy(&scale,         (ଅଚଳ ଦଶମିକ *) KQV->op_params + 0, ଆକାର(ଦଶମିକ));
    memcpy(&max_bias,      (ଅଚଳ ଦଶମିକ *) KQV->op_params + 1, ଆକାର(ଦଶମିକ));
    memcpy(&logit_softcap, (ଅଚଳ ଦଶମିକ *) KQV->op_params + 2, ଆକାର(ଦଶମିକ));

    ଯଦି (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    ଅଚଳ uint32_t n_head      = Q->ne[2];
    ଅଚଳ uint32_t n_head_log2 = 1u << uint32_t(floorf(log2f(ଦଶମିକ(n_head))));

    ଅଚଳ ଦଶମିକ m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    ଅଚଳ ଦଶମିକ m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    // TODO other tensor dimensions after removal of WMMA kernel:
    ଅଚଳ sycl::uint3 ne01 = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ(Q->ne[1]);

    GGML_ASSERT(block_dim.x % warp_size == 0);

    lauch_kernel<fattn_kernel, warp_size, use_large_grf>(
        blocks_ଅଙ୍କ, block_dim, main_stream, (unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା) nbytes_shared, (ଅଚଳ ଅକ୍ଷର *) Q->data, K_data, V_data,
        mask ? ((ଅଚଳ ଅକ୍ଷର *) mask->data) : nullptr, sinks ? ((ଅଚଳ ଅକ୍ଷର *) sinks->data) : nullptr, KV_max.ptr,
        !stream_k && parallel_blocks > 1 ? dst_tmp.ptr : (ଦଶମିକ *) KQV->data, (sycl::float2 *)dst_tmp_meta.ptr, scale, max_bias, m0, m1,
        n_head_log2, logit_softcap, Q->ne[0], ne01, Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3], K->ne[0],
        K->ne[1], K->ne[2], K->ne[3], nb11, nb12, nb13, nb21, nb22, nb23, mask ? mask->ne[1] : 0,
        mask ? mask->ne[2] : 0, mask ? mask->ne[3] : 0, mask ? mask->nb[1] : 0, mask ? mask->nb[2] : 0,
        mask ? mask->nb[3] : 0);
    SYCL_CHECK(0);

    ଯଦି (stream_k) {
        ଯଦି (ntiles_total % blocks_ଅଙ୍କ.x != 0) { // Fixup is only needed if the SMs work on fractional tiles.
            ଅଚଳ dpct::dim3 block_dim_combine(DV, 1, 1);
            ଅଚଳ dpct::dim3 blocks_ଅଙ୍କ_combine = { blocks_ଅଙ୍କ.x, ncols1, ncols2 };

            main_stream->submit([&](sycl::handler & cgh) {
                auto KQV_data_ct0         = (ଦଶମିକ *) KQV->data;
                auto dst_tmp_meta_ptr_ct1 = dst_tmp_meta.ptr;
                auto Q_ne_ct2             = Q->ne[1];
                auto Q_ne_ct3             = Q->ne[2];
                auto Q_ne_ct4             = Q->ne[3];
                auto K_ne_ct5             = K->ne[1];
                auto K_ne_ct6             = K->ne[2];

                cgh.parallel_for(sycl::nd_range<3>(blocks_ଅଙ୍କ_combine * block_dim_combine, block_dim_combine),
                                 [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(warp_size)]] {
                                     GGML_UNUSED(item_ct1);
                                     flash_attn_stream_k_fixup<DV, ncols1, ncols2>(KQV_data_ct0, dst_tmp_meta_ptr_ct1,
                                                                                   Q_ne_ct2, Q_ne_ct3, Q_ne_ct4,
                                                                                   K_ne_ct5, K_ne_ct6, nbatch_fa);
                                 });
            });
        }
    } ଅନ୍ୟଥା ଯଦି (parallel_blocks > 1) {
        ଅଚଳ dpct::dim3 block_dim_combine(DV, 1, 1);
        ଅଚଳ dpct::dim3 blocks_ଅଙ୍କ_combine(Q->ne[1], Q->ne[2], Q->ne[3]);
        ଅଚଳ size_t     nbytes_shared_combine = parallel_blocks * ଆକାର(sycl::float2);
        main_stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(sycl::range<1>(nbytes_shared_combine), cgh);

            auto dst_tmp_ptr_ct0      = dst_tmp.ptr;
            auto dst_tmp_meta_ptr_ct1 = dst_tmp_meta.ptr;
            auto KQV_data_ct2         = (ଦଶମିକ *) KQV->data;

            cgh.parallel_for(sycl::nd_range<3>(blocks_ଅଙ୍କ_combine * block_dim_combine, block_dim_combine),
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
