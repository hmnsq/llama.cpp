#pragma once

#অন্তর্ভাব <sycl/sycl.hpp>
#অন্তর্ভাব <sycl/ext/intel/experimental/grf_size_properties.hpp>
#অন্তর্ভাব "dpct/helper.hpp"
#অন্তর্ভাব "common.hpp"
#অন্তর্ভাব "convert.hpp"
#অন্তর্ভাব "vecdotq.hpp"
#অন্তর্ভাব "fattn.hpp"

#অন্তর্ভাব "ggml.h"

#অন্তর্ভাব <cstdint>
#অন্তর্ভাব <cmath>
#অন্তর্ভাব <float.h>


#define FATTN_KQ_STRIDE       256
#define HALF_MAX_HALF         sycl::half(65504.0f/2) // Use neg. of this instead of -INFINITY to initialize KQ max vals to avoid NaN upon subtraction.
#define SOFTMAX_FTZ_THRESHOLD -20.0f                   // Softmax exp. of values smaller than this are flushed to zero to avoid NaNs.
#define FATTN_KQ_MAX_OFFSET (3.0f*0.6931f)

প্রকারলেখ শূন্য (*fattn_kernel_t)(
    ধ্রুবক অক্ষর* Q,
    ধ্রুবক অক্ষর* K,
    ধ্রুবক অক্ষর* V,
    ধ্রুবক অক্ষর* mask,
    ধ্রুবক অক্ষর* sinks,
    ধ্রুবক পূর্ণসংখ্যা* KV_max,
    দশমিক* dst,
    sycl::float2* dst_meta,
    ধ্রুবক দশমিক scale,
    ধ্রুবক দশমিক max_bias,
    ধ্রুবক দশমিক m0,
    ধ্রুবক দশমিক m1,
    ধ্রুবক uint32_t n_শীর্ষ_log2,
    ধ্রুবক দশমিক logit_softcap,
    ধ্রুবক int32_t ne00,
    ধ্রুবক sycl::uint3 ne01,
    ধ্রুবক int32_t ne02,
    ধ্রুবক int32_t ne03,
    ধ্রুবক int32_t nb01,
    ধ্রুবক int32_t nb02,
    ধ্রুবক int32_t nb03,
    ধ্রুবক int32_t ne10,
    ধ্রুবক int32_t ne11,
    ধ্রুবক int32_t ne12,
    ধ্রুবক int32_t ne13,
    ধ্রুবক int32_t nb11,
    ধ্রুবক int32_t nb12,
    ধ্রুবক int64_t nb13,
    ধ্রুবক int32_t nb21,
    ধ্রুবক int32_t nb22,
    ধ্রুবক int64_t nb23,
    ধ্রুবক int32_t ne31,
    ধ্রুবক int32_t ne32,
    ধ্রুবক int32_t ne33,
    ধ্রুবক int32_t nb31,
    ধ্রুবক int32_t nb32,
    ধ্রুবক int64_t nb33);

প্রকারলেখ দশমিক (*vec_dot_KQ_t)(
    ধ্রুবক অক্ষর * __restrict__ K_c, ধ্রুবক শূন্য * __restrict__ Q_v, ধ্রুবক পূর্ণসংখ্যা * __restrict__ Q_q8 , ধ্রুবক শূন্য * __restrict__ Q_ds);

template <পূর্ণসংখ্যা D, পূর্ণসংখ্যা nthreads>
স্থির __dpct_inline__ দশমিক vec_dot_fattn_vec_KQ_f16(ধ্রুবক অক্ষর * __restrict__ K_c,
                                                      ধ্রুবক শূন্য * __restrict__ Q_v,
                                                      ধ্রুবক পূর্ণসংখ্যা * __restrict__ Q_q8,
                                                      ধ্রুবক শূন্য * __restrict__ Q_ds_v) {
    ধ্রুবক sycl::half2 * K_h2 = (ধ্রুবক sycl::half2 *) K_c;
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v);

    constexpr পূর্ণসংখ্যা cpy_nb = ggml_sycl_get_max_cpy_bytes();
    constexpr পূর্ণসংখ্যা cpy_ne = cpy_nb / 4;

    দশমিক যোগ_2 = 0.0f;

#pragma unroll
    জন্য (পূর্ণসংখ্যা k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads*cpy_ne) {
        sycl::half2 tmp[cpy_ne];
        ggml_sycl_memcpy_1<আকার(tmp)>(
            tmp,
            K_h2 + k_KQ_0 + (sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2) % nthreads) * cpy_ne);
#pragma unroll
        জন্য (পূর্ণসংখ্যা k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
#ifdef GGML_SYCL_F16
            ggml_sycl_mad(যোগ_2,                tmp[k_KQ_1] , ((ধ্রুবক sycl::half2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#অন্যথা
            ggml_sycl_mad(যোগ_2, __half22float2(tmp[k_KQ_1]), ((ধ্রুবক sycl::float2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#endif // GGML_SYCL_F16
        }
    }

    ফেরত যোগ_2;
}

template <পূর্ণসংখ্যা D, পূর্ণসংখ্যা nthreads, পূর্ণসংখ্যা warp_আকার>
স্থির __dpct_inline__ দশমিক vec_dot_fattn_vec_KQ_q4_0(ধ্রুবক অক্ষর * __restrict__ K_c,
                                                       ধ্রুবক শূন্য * __restrict__ Q_v,
                                                       ধ্রুবক পূর্ণসংখ্যা * __restrict__ Q_q8,
                                                       ধ্রুবক শূন্য * __restrict__ Q_ds_v) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    ধ্রুবক block_q4_0 * K_q4_0   = (ধ্রুবক block_q4_0 *) K_c;
    GGML_UNUSED(Q_v);

    দশমিক যোগ_2 = 0.0f;

#pragma unroll
    জন্য (পূর্ণসংখ্যা k_KQ_0 = 0; k_KQ_0 < পূর্ণসংখ্যা(D/আকার(পূর্ণসংখ্যা)); k_KQ_0 += nthreads) {
        ধ্রুবক পূর্ণসংখ্যা k_KQ =
            k_KQ_0 + (nthreads == warp_আকার ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads);

        ধ্রুবক পূর্ণসংখ্যা ib    = k_KQ /  QI8_1;
        ধ্রুবক পূর্ণসংখ্যা iqs4  = k_KQ %  QI4_0;
        ধ্রুবক পূর্ণসংখ্যা shift = k_KQ & (QI8_1/2);

        পূর্ণসংখ্যা v;
        ggml_sycl_memcpy_1<আকার(পূর্ণসংখ্যা), 2>(&v, K_q4_0[ib].qs + আকার(পূর্ণসংখ্যা)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;
        ধ্রুবক পূর্ণসংখ্যা u = Q_q8[k_KQ_0/nthreads];

        ধ্রুবক পূর্ণসংখ্যা sumi = ggml_sycl_dp4a(v, u, 0);

        ধ্রুবক sycl::float2 Q_ds = ((ধ্রুবক sycl::float2 *) Q_ds_v)[k_KQ_0 / nthreads];
        যোগ_2 += __half2float(K_q4_0[ib].d) * (sumi*Q_ds.x() - (8/QI8_1)*Q_ds.y());
    }

    ফেরত যোগ_2;
}

template <পূর্ণসংখ্যা D, পূর্ণসংখ্যা nthreads , পূর্ণসংখ্যা warp_আকার>
স্থির __dpct_inline__ দশমিক vec_dot_fattn_vec_KQ_q4_1(ধ্রুবক অক্ষর * __restrict__ K_c,
                                                       ধ্রুবক শূন্য * __restrict__ Q_v,
                                                       ধ্রুবক পূর্ণসংখ্যা * __restrict__ Q_q8,
                                                       ধ্রুবক শূন্য * __restrict__ Q_ds_v) {
    auto               item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ধ্রুবক block_q4_1 * K_q4_1   = (ধ্রুবক block_q4_1 *) K_c;
    GGML_UNUSED(Q_v);

    দশমিক যোগ_2 = 0.0f;

#pragma unroll
    জন্য (পূর্ণসংখ্যা k_KQ_0 = 0; k_KQ_0 < পূর্ণসংখ্যা(D/আকার(পূর্ণসংখ্যা)); k_KQ_0 += nthreads) {
        ধ্রুবক পূর্ণসংখ্যা k_KQ =
            k_KQ_0 + (nthreads == warp_আকার ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads);

        ধ্রুবক পূর্ণসংখ্যা ib    = k_KQ /  QI8_1;
        ধ্রুবক পূর্ণসংখ্যা iqs4  = k_KQ %  QI4_1;
        ধ্রুবক পূর্ণসংখ্যা shift = k_KQ & (QI8_1/2);

        পূর্ণসংখ্যা v;
        ggml_sycl_memcpy_1<আকার(পূর্ণসংখ্যা)>(&v, K_q4_1[ib].qs + আকার(পূর্ণসংখ্যা)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;
        ধ্রুবক পূর্ণসংখ্যা u = Q_q8[k_KQ_0/nthreads];

        ধ্রুবক পূর্ণসংখ্যা sumi = ggml_sycl_dp4a(v, u, 0);

        ধ্রুবক sycl::float2 K_dm = (K_q4_1[ib].dm).template convert<দশমিক, sycl::rounding_mode::automatic>();
        ধ্রুবক sycl::float2 Q_ds = ((ধ্রুবক sycl::float2 *) Q_ds_v)[k_KQ_0 / nthreads];

        যোগ_2 += K_dm.x()*Q_ds.x()*sumi + K_dm.y()*Q_ds.y()/QI8_1;
    }

    ফেরত যোগ_2;
}

template <পূর্ণসংখ্যা D, পূর্ণসংখ্যা nthreads, পূর্ণসংখ্যা warp_আকার>
স্থির __dpct_inline__ দশমিক vec_dot_fattn_vec_KQ_q5_0(ধ্রুবক অক্ষর * __restrict__ K_c,
                                                       ধ্রুবক শূন্য * __restrict__ Q_v,
                                                       ধ্রুবক পূর্ণসংখ্যা * __restrict__ Q_q8,
                                                       ধ্রুবক শূন্য * __restrict__ Q_ds_v) {
    auto               item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ধ্রুবক block_q5_0 * K_q5_0   = (ধ্রুবক block_q5_0 *) K_c;
    GGML_UNUSED(Q_v);

    দশমিক যোগ_2 = 0.0f;

#pragma unroll
    জন্য (পূর্ণসংখ্যা k_KQ_0 = 0; k_KQ_0 < পূর্ণসংখ্যা(D/আকার(পূর্ণসংখ্যা)); k_KQ_0 += nthreads) {
        ধ্রুবক পূর্ণসংখ্যা k_KQ =
            k_KQ_0 + (nthreads == warp_আকার ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads);

        ধ্রুবক পূর্ণসংখ্যা ib    = k_KQ /  QI8_1;
        ধ্রুবক পূর্ণসংখ্যা iqs4  = k_KQ %  QI5_0;
        ধ্রুবক পূর্ণসংখ্যা iqs8  = k_KQ %  QI8_1;
        ধ্রুবক পূর্ণসংখ্যা shift = k_KQ & (QI8_1/2);

        পূর্ণসংখ্যা v;
        ggml_sycl_memcpy_1<আকার(পূর্ণসংখ্যা), 2>(&v, K_q5_0[ib].qs + আকার(পূর্ণসংখ্যা)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;

        {
            পূর্ণসংখ্যা vh;
            ggml_sycl_memcpy_1<আকার(পূর্ণসংখ্যা), 2>(&vh, K_q5_0[ib].qh);
            vh >>= iqs8 * QI5_0;

            v |= (vh <<  4) & 0x00000010; // 0 ->  4
            v |= (vh << 11) & 0x00001000; // 1 -> 12
            v |= (vh << 18) & 0x00100000; // 2 -> 20
            v |= (vh << 25) & 0x10000000; // 3 -> 28
        }

        ধ্রুবক পূর্ণসংখ্যা u = Q_q8[k_KQ_0/nthreads];

        ধ্রুবক পূর্ণসংখ্যা sumi = ggml_sycl_dp4a(v, u, 0);

        ধ্রুবক sycl::float2 Q_ds = ((ধ্রুবক sycl::float2 *) Q_ds_v)[k_KQ_0 / nthreads];

        যোগ_2 += __half2float(K_q5_0[ib].d) * (sumi*Q_ds.x() - (16/QI8_1)*Q_ds.y());
    }

    ফেরত যোগ_2;
}

template <পূর্ণসংখ্যা D, পূর্ণসংখ্যা nthreads, পূর্ণসংখ্যা warp_আকার>
স্থির __dpct_inline__ দশমিক vec_dot_fattn_vec_KQ_q5_1(ধ্রুবক অক্ষর * __restrict__ K_c,
                                                       ধ্রুবক শূন্য * __restrict__ Q_v,
                                                       ধ্রুবক পূর্ণসংখ্যা * __restrict__ Q_q8,
                                                       ধ্রুবক শূন্য * __restrict__ Q_ds_v) {
    auto               item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ধ্রুবক block_q5_1 * K_q5_1   = (ধ্রুবক block_q5_1 *) K_c;
    GGML_UNUSED(Q_v);

    দশমিক যোগ_2 = 0.0f;

#pragma unroll
    জন্য (পূর্ণসংখ্যা k_KQ_0 = 0; k_KQ_0 < পূর্ণসংখ্যা(D/আকার(পূর্ণসংখ্যা)); k_KQ_0 += nthreads) {
        ধ্রুবক পূর্ণসংখ্যা k_KQ =
            k_KQ_0 + (nthreads == warp_আকার ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads);

        ধ্রুবক পূর্ণসংখ্যা ib    = k_KQ /  QI8_1;
        ধ্রুবক পূর্ণসংখ্যা iqs4  = k_KQ %  QI5_1;
        ধ্রুবক পূর্ণসংখ্যা iqs8  = k_KQ %  QI8_1;
        ধ্রুবক পূর্ণসংখ্যা shift = k_KQ & (QI8_1/2);

        পূর্ণসংখ্যা v;
        ggml_sycl_memcpy_1<আকার(পূর্ণসংখ্যা)>(&v, K_q5_1[ib].qs + আকার(পূর্ণসংখ্যা)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;

        {
            পূর্ণসংখ্যা vh;
            ggml_sycl_memcpy_1<আকার(পূর্ণসংখ্যা)>(&vh, K_q5_1[ib].qh);
            vh >>= iqs8 * QI5_0;

            v |= (vh <<  4) & 0x00000010; // 0 ->  4
            v |= (vh << 11) & 0x00001000; // 1 -> 12
            v |= (vh << 18) & 0x00100000; // 2 -> 20
            v |= (vh << 25) & 0x10000000; // 3 -> 28
        }

        ধ্রুবক পূর্ণসংখ্যা u = Q_q8[k_KQ_0/nthreads];

        ধ্রুবক পূর্ণসংখ্যা sumi = ggml_sycl_dp4a(v, u, 0);

        ধ্রুবক sycl::float2 K_dm = (K_q5_1[ib].dm).template convert<দশমিক, sycl::rounding_mode::automatic>();
        ধ্রুবক sycl::float2 Q_ds = ((ধ্রুবক sycl::float2 *) Q_ds_v)[k_KQ_0 / nthreads];

        যোগ_2 += K_dm.x()*Q_ds.x()*sumi + K_dm.y()*Q_ds.y()/QI8_1;
    }

    ফেরত যোগ_2;
}

template <পূর্ণসংখ্যা D, পূর্ণসংখ্যা nthreads, পূর্ণসংখ্যা warp_আকার>
স্থির __dpct_inline__ দশমিক vec_dot_fattn_vec_KQ_q8_0(ধ্রুবক অক্ষর * __restrict__ K_c,
                                                       ধ্রুবক শূন্য * __restrict__ Q_v,
                                                       ধ্রুবক পূর্ণসংখ্যা * __restrict__ Q_q8,
                                                       ধ্রুবক শূন্য * __restrict__ Q_ds_v) {
    auto               item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ধ্রুবক block_q8_0 * K_q8_0   = (ধ্রুবক block_q8_0 *) K_c;
    GGML_UNUSED(Q_v);

    দশমিক যোগ_2 = 0.0f;

#pragma unroll
    জন্য (পূর্ণসংখ্যা k_KQ_0 = 0; k_KQ_0 < পূর্ণসংখ্যা(D/আকার(পূর্ণসংখ্যা)); k_KQ_0 += nthreads) {
        ধ্রুবক পূর্ণসংখ্যা k_KQ =
            k_KQ_0 + (nthreads == warp_আকার ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads);

        ধ্রুবক পূর্ণসংখ্যা ib  = k_KQ / QI8_0;
        ধ্রুবক পূর্ণসংখ্যা iqs = k_KQ % QI8_0;

        পূর্ণসংখ্যা v;
        ggml_sycl_memcpy_1<আকার(v), 2>(&v, K_q8_0[ib].qs + 4*iqs);

        ধ্রুবক sycl::float2 * Q_ds = (ধ্রুবক sycl::float2 *) Q_ds_v;
        ধ্রুবক দশমিক          Q_d  = Q_ds[k_KQ_0 / nthreads].x();

        যোগ_2 += vec_dot_q8_0_q8_1_impl<দশমিক, 1>(&v, &Q_q8[k_KQ_0/nthreads], K_q8_0[ib].d, Q_d);
    }

    ফেরত যোগ_2;
}

template <typename Tds, পূর্ণসংখ্যা ni, পূর্ণসংখ্যা warp_আকার>
স্থির __dpct_inline__ শূন্য quantize_q8_1_to_shared(ধ্রুবক দশমিক * __restrict__ x,
                                                    ধ্রুবক দশমিক scale,
                                                    পূর্ণসংখ্যা * __restrict__ yq32,
                                                    শূন্য * __restrict__ yds) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    দশমিক vals[আকার(পূর্ণসংখ্যা)] = { 0.0f };
#pragma unroll
    জন্য (পূর্ণসংখ্যা l = 0; l < পূর্ণসংখ্যা(আকার(পূর্ণসংখ্যা)); ++l) {
        vals[l] =
            (ni == warp_আকার || item_ct1.get_local_id(2) < ni) ? scale * x[4 * item_ct1.get_local_id(2) + l] : 0.0f;
    }

    দশমিক amax = sycl::fabs(vals[0]);
    দশমিক যোগ_2  = vals[0];
#pragma unroll
    জন্য (পূর্ণসংখ্যা l = 1; l < পূর্ণসংখ্যা(আকার(পূর্ণসংখ্যা)); ++l) {
        amax = sycl::fmax(amax, sycl::fabs(vals[l]));
        যোগ_2 += vals[l];
    }
#pragma unroll
    জন্য (পূর্ণসংখ্যা mask = QI8_1/2; mask > 0; mask >>= 1) {
        amax = sycl::fmax(
            amax, dpct::permute_sub_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(), amax, mask));
        যোগ_2 += dpct::permute_sub_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(), যোগ_2, mask);
    }

    ধ্রুবক দশমিক d = amax / 127;
    পূর্ণসংখ্যা q32 = 0;
    int8_t * q8 = (int8_t *) &q32;

    যদি (d != 0.0f) {
#pragma unroll
        জন্য (পূর্ণসংখ্যা l = 0; l < পূর্ণসংখ্যা(আকার(পূর্ণসংখ্যা)); ++l) {
            q8[l] = sycl::round(vals[l] / d);
        }
    }

    yq32[item_ct1.get_local_id(2)] = q32;
    যদি (item_ct1.get_local_id(2) % QI8_1 == 0 && (ni == warp_আকার || item_ct1.get_local_id(2) < ni)) {
        যদি (std::is_same<Tds, sycl::half2>::value) {
            ((sycl::half2  *) yds)[item_ct1.get_local_id(2)/QI8_1] =  make_half2(d, যোগ_2);
        } অন্যথা {
            ((sycl::float2 *) yds)[item_ct1.get_local_id(2)/QI8_1] = make_float2(d, যোগ_2);
        }
    }
}

প্রকারলেখ শূন্য (*dequantize_V_t)(ধ্রুবক শূন্য *, শূন্য *, ধ্রুবক int64_t);

template <typename T, পূর্ণসংখ্যা ne>
স্থির __dpct_inline__ শূন্য dequantize_V_f16(ধ্রুবক শূন্য * __restrict__ vx, শূন্য * __restrict__ dst, ধ্রুবক int64_t i0) {
    যদি constexpr (std::is_same_v<T, sycl::half>) {
        ggml_sycl_memcpy_1<ne * আকার(sycl::half)>(dst, (ধ্রুবক sycl::half *) vx + i0);
    } অন্যথা যদি constexpr (std::is_same_v<T, দশমিক>) {
        static_assert(ne % 2 == 0, "bad ne");
        sycl::half2 tmp[ne / 2];
        ggml_sycl_memcpy_1<ne * আকার(sycl::half)>(tmp, (ধ্রুবক sycl::half *) vx + i0);
        sycl::float2 * dst_f2 = (sycl::float2 *) dst;
#pragma unroll
        জন্য (পূর্ণসংখ্যা l = 0; l < ne/2; ++l) {
            dst_f2[l] = tmp[l].template convert<দশমিক, sycl::rounding_mode::automatic>();
        }
    } অন্যথা {
        static_assert(std::is_same_v<T, শূন্য>, "unsupported type");
    }
}

template <typename T, পূর্ণসংখ্যা ne>
স্থির __dpct_inline__ শূন্য dequantize_V_q4_0(ধ্রুবক শূন্য * __restrict__ vx, শূন্য * __restrict__ dst, ধ্রুবক int64_t i0) {
    ধ্রুবক block_q4_0 * x = (ধ্রুবক block_q4_0 *) vx;

    ধ্রুবক int64_t ib    =  i0          /  QK4_0;
    ধ্রুবক পূর্ণসংখ্যা     iqs   =  i0          % (QK4_0/2);
    ধ্রুবক পূর্ণসংখ্যা     shift = (i0 % QK4_0) / (QK4_0/2);

    পূর্ণসংখ্যা q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_sycl_memcpy_1<ne, 2>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;
    q = dpct::vectorized_দ্বিমিক<sycl::char4>(q, 0x08080808, dpct::sub_sat());

    ধ্রুবক int8_t * q8 = (ধ্রুবক int8_t *) &q;

#ifdef GGML_SYCL_F16
    যদি constexpr (std::is_same_v<T, sycl::half>) {
        ধ্রুবক sycl::half2 d = sycl::half2(x[ib].d);

#pragma unroll
        জন্য (পূর্ণসংখ্যা l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * sycl::half2(q8[l0 + 0], q8[l0 + 1]);
        }
    } অন্যথা
#endif // GGML_SYCL_F16
    যদি constexpr (std::is_same_v<T, দশমিক>) {
        ধ্রুবক দশমিক d = x[ib].d;

#pragma unroll
        জন্য (পূর্ণসংখ্যা l = 0; l < ne; ++l) {
            ((দশমিক *) dst)[l] = d * q8[l];
        }
    } অন্যথা {
        static_assert(std::is_same_v<T, শূন্য>, "bad type");
    }
}

template <typename T, পূর্ণসংখ্যা ne>
স্থির __dpct_inline__ শূন্য dequantize_V_q4_1(ধ্রুবক শূন্য * __restrict__ vx, শূন্য * __restrict__ dst, ধ্রুবক int64_t i0) {
    ধ্রুবক block_q4_1 * x = (ধ্রুবক block_q4_1 *) vx;

    ধ্রুবক int64_t ib    =  i0          /  QK4_1;
    ধ্রুবক পূর্ণসংখ্যা     iqs   =  i0          % (QK4_1/2);
    ধ্রুবক পূর্ণসংখ্যা     shift = (i0 % QK4_1) / (QK4_1/2);

    পূর্ণসংখ্যা q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_sycl_memcpy_1<ne>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    ধ্রুবক int8_t * q8 = (ধ্রুবক int8_t *) &q;

#ifdef GGML_SYCL_F16
    যদি constexpr (std::is_same_v<T, sycl::half>) {
        ধ্রুবক sycl::half2 dm = x[ib].dm;
        ধ্রুবক sycl::half2 d  = sycl::half2(dm[0]);
        ধ্রুবক sycl::half2 m  = sycl::half2(dm[1]);

#pragma unroll
        জন্য (পূর্ণসংখ্যা l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * sycl::half2(q8[l0 + 0], q8[l0 + 1]) + m;
        }
    } অন্যথা
#endif // GGML_SYCL_F16
    যদি constexpr (std::is_same_v<T, দশমিক>) {
        ধ্রুবক sycl::float2 dm = (x[ib].dm).template convert<দশমিক, sycl::rounding_mode::automatic>();

#pragma unroll
        জন্য (পূর্ণসংখ্যা l = 0; l < ne; ++l) {
            ((দশমিক *) dst)[l] = dm.x() * q8[l] + dm.y();
        }
    } অন্যথা {
        static_assert(std::is_same_v<T, শূন্য>, "bad type");
    }
}

template <typename T, পূর্ণসংখ্যা ne>
স্থির __dpct_inline__ শূন্য dequantize_V_q5_0(ধ্রুবক শূন্য * __restrict__ vx, শূন্য * __restrict__ dst, ধ্রুবক int64_t i0) {
    ধ্রুবক block_q5_0 * x = (ধ্রুবক block_q5_0 *) vx;

    ধ্রুবক int64_t ib    =  i0          /  QK5_0;
    ধ্রুবক পূর্ণসংখ্যা     idq   =  i0          %  QK5_0;
    ধ্রুবক পূর্ণসংখ্যা     iqs   =  i0          % (QK5_0/2);
    ধ্রুবক পূর্ণসংখ্যা     shift = (i0 % QK5_0) / (QK5_0/2);

    পূর্ণসংখ্যা q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_sycl_memcpy_1<ne, 2>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    {
        পূর্ণসংখ্যা qh;
        ggml_sycl_memcpy_1<ne, 2>(&qh, x[ib].qh);
#pragma unroll
        জন্য (পূর্ণসংখ্যা l = 0; l < ne; ++l) {
            q |= ((qh >> (idq + l)) & 0x00000001) << (8*l + 4);
        }
    }

    q = dpct::vectorized_দ্বিমিক<sycl::char4>(q, 0x10101010, dpct::sub_sat());

    ধ্রুবক int8_t * q8 = (ধ্রুবক int8_t *) &q;

#ifdef GGML_SYCL_F16
    যদি constexpr (std::is_same_v<T, sycl::half>) {
        ধ্রুবক sycl::half2 d = sycl::half2(x[ib].d);

#pragma unroll
        জন্য (পূর্ণসংখ্যা l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * sycl::half2(q8[l0 + 0], q8[l0 + 1]);
        }
    } অন্যথা
#endif // GGML_SYCL_F16
    যদি constexpr (std::is_same_v<T, দশমিক>) {
        ধ্রুবক দশমিক d = x[ib].d;

#pragma unroll
        জন্য (পূর্ণসংখ্যা l = 0; l < ne; ++l) {
            ((দশমিক *) dst)[l] = d * q8[l];
        }
    } অন্যথা {
        static_assert(std::is_same_v<T, শূন্য>, "bad type");
    }
}

template <typename T, পূর্ণসংখ্যা ne>
স্থির __dpct_inline__ শূন্য dequantize_V_q5_1(ধ্রুবক শূন্য * __restrict__ vx, শূন্য * __restrict__ dst, ধ্রুবক int64_t i0) {
    ধ্রুবক block_q5_1 * x = (ধ্রুবক block_q5_1 *) vx;

    ধ্রুবক int64_t ib    =  i0          /  QK5_1;
    ধ্রুবক পূর্ণসংখ্যা     idq   =  i0          %  QK5_1;
    ধ্রুবক পূর্ণসংখ্যা     iqs   =  i0          % (QK5_1/2);
    ধ্রুবক পূর্ণসংখ্যা     shift = (i0 % QK5_1) / (QK5_1/2);

    পূর্ণসংখ্যা q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_sycl_memcpy_1<ne>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    {
        পূর্ণসংখ্যা qh;
        ggml_sycl_memcpy_1<ne>(&qh, x[ib].qh);
#pragma unroll
        জন্য (পূর্ণসংখ্যা l = 0; l < ne; ++l) {
            q |= ((qh >> (idq + l)) & 0x00000001) << (8*l + 4);
        }
    }

    ধ্রুবক int8_t * q8 = (ধ্রুবক int8_t *) &q;

#ifdef GGML_SYCL_F16
    যদি constexpr (std::is_same_v<T, sycl::half>) {
        ধ্রুবক sycl::half2 dm = x[ib].dm;
        ধ্রুবক sycl::half2 d  = sycl::half2(dm[0]);
        ধ্রুবক sycl::half2 m  = sycl::half2(dm[1]);

#pragma unroll
        জন্য (পূর্ণসংখ্যা l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * sycl::half2(q8[l0 + 0], q8[l0 + 1]) + m;
        }
    } অন্যথা
#endif // GGML_SYCL_F16
    যদি constexpr (std::is_same_v<T, দশমিক>) {
        ধ্রুবক sycl::float2 dm = (x[ib].dm).template convert<দশমিক, sycl::rounding_mode::automatic>();

#pragma unroll
        জন্য (পূর্ণসংখ্যা l = 0; l < ne; ++l) {
            ((দশমিক *) dst)[l] = dm.x() * q8[l] + dm.y();
        }
    } অন্যথা {
        static_assert(std::is_same_v<T, শূন্য>, "bad type");
    }
}

template <typename T, পূর্ণসংখ্যা ne>
স্থির __dpct_inline__ শূন্য dequantize_V_q8_0(ধ্রুবক শূন্য * __restrict__ vx, শূন্য * __restrict__ dst, ধ্রুবক int64_t i0) {
    ধ্রুবক block_q8_0 * x = (ধ্রুবক block_q8_0 *) vx;

    ধ্রুবক int64_t ib  = i0 / QK8_0;
    ধ্রুবক পূর্ণসংখ্যা     iqs = i0 % QK8_0;

    static_assert(ne % 2 == 0, "bad ne");
    int8_t qs[ne];
    ggml_sycl_memcpy_1<ne, 2>(qs, x[ib].qs + iqs);

#ifdef GGML_SYCL_F16
    যদি constexpr (std::is_same<T, sycl::half>::value) {
        ধ্রুবক sycl::half2 d = sycl::half2(x[ib].d);

#pragma unroll
        জন্য (পূর্ণসংখ্যা l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * make_half2(qs[l0 + 0], qs[l0 + 1]);
        }
    } অন্যথা
#endif // GGML_SYCL_F16
    যদি constexpr (std::is_same<T, দশমিক>::value) {
        ধ্রুবক দশমিক d = x[ib].d;

#pragma unroll
        জন্য (পূর্ণসংখ্যা l = 0; l < ne; ++l) {
            ((দশমিক *) dst)[l] = d * qs[l];
        }
    } অন্যথা {
        static_assert(std::is_same_v<T, শূন্য>, "unsupported type");
    }
}

template <পূর্ণসংখ্যা type_K, পূর্ণসংখ্যা D, পূর্ণসংখ্যা nthreads, পূর্ণসংখ্যা warp_আকার>
constexpr vec_dot_KQ_t get_vec_dot_KQ() {
    যদি constexpr (type_K == GGML_TYPE_F16) {
        ফেরত vec_dot_fattn_vec_KQ_f16<D, nthreads>;
    } অন্যথা যদি constexpr (type_K == GGML_TYPE_Q4_0) {
        ফেরত vec_dot_fattn_vec_KQ_q4_0<D, nthreads, warp_আকার>;
    } অন্যথা যদি constexpr (type_K == GGML_TYPE_Q4_1) {
        ফেরত vec_dot_fattn_vec_KQ_q4_1<D, nthreads, warp_আকার>;
    } অন্যথা যদি constexpr (type_K == GGML_TYPE_Q5_0) {
        ফেরত vec_dot_fattn_vec_KQ_q5_0<D, nthreads, warp_আকার>;
    } অন্যথা যদি constexpr (type_K == GGML_TYPE_Q5_1) {
        ফেরত vec_dot_fattn_vec_KQ_q5_1<D, nthreads, warp_আকার>;
    } অন্যথা যদি constexpr (type_K == GGML_TYPE_Q8_0) {
        ফেরত vec_dot_fattn_vec_KQ_q8_0<D, nthreads, warp_আকার>;
    } অন্যথা {
        static_assert(type_K == -1, "bad type");
        ফেরত nullptr;
    }
}

template <পূর্ণসংখ্যা type_V, typename T, পূর্ণসংখ্যা ne>
constexpr dequantize_V_t get_dequantize_V() {
    যদি constexpr (type_V == GGML_TYPE_F16) {
        ফেরত dequantize_V_f16<T, ne>;
    } অন্যথা যদি constexpr (type_V == GGML_TYPE_Q4_0) {
        ফেরত dequantize_V_q4_0<T, ne>;
    } অন্যথা যদি constexpr (type_V == GGML_TYPE_Q4_1) {
        ফেরত dequantize_V_q4_1<T, ne>;
    } অন্যথা যদি constexpr (type_V == GGML_TYPE_Q5_0) {
        ফেরত dequantize_V_q5_0<T, ne>;
    } অন্যথা যদি constexpr (type_V == GGML_TYPE_Q5_1) {
        ফেরত dequantize_V_q5_1<T, ne>;
    } অন্যথা যদি constexpr (type_V == GGML_TYPE_Q8_0) {
        ফেরত dequantize_V_q8_0<T, ne>;
    } অন্যথা {
        static_assert(type_V == -1, "bad type");
        ফেরত nullptr;
    }
}

template <পূর্ণসংখ্যা ncols1, পূর্ণসংখ্যা warp_আকার>
স্থির শূন্য flash_attn_mask_to_KV_max(ধ্রুবক sycl::half2 * __restrict__ mask,
                                      পূর্ণসংখ্যা * __restrict__ KV_max,
                                      ধ্রুবক পূর্ণসংখ্যা ne30,
                                      ধ্রুবক পূর্ণসংখ্যা s31,
                                      ধ্রুবক পূর্ণসংখ্যা s33,
                                      পূর্ণসংখ্যা *     buf_iw) {
    auto      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ধ্রুবক পূর্ণসংখ্যা ne31     = item_ct1.get_group_range(2);
    ধ্রুবক পূর্ণসংখ্যা tid      = item_ct1.get_local_id(2);
    ধ্রুবক পূর্ণসংখ্যা sequence = item_ct1.get_group(1);
    ধ্রুবক পূর্ণসংখ্যা jt       = item_ct1.get_group(2);

    mask += sequence*s33 + jt*ncols1*s31;

    যদি (tid < warp_আকার) {
        buf_iw[tid] = 1;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);

    পূর্ণসংখ্যা KV_max_sj = (ne30 - 1) * FATTN_KQ_STRIDE;
    জন্য (; KV_max_sj >= 0; KV_max_sj -= FATTN_KQ_STRIDE) {
        পূর্ণসংখ্যা all_inf = 1;

#pragma unroll
        জন্য (পূর্ণসংখ্যা j = 0; j < ncols1; ++j) {
            ধ্রুবক sycl::float2 tmp =
                mask[j * s31 + KV_max_sj / 2 + tid].template convert<দশমিক, sycl::rounding_mode::automatic>();
            all_inf = all_inf && পূর্ণসংখ্যা(sycl::isinf((দশমিক) (tmp.x()))) && পূর্ণসংখ্যা(sycl::isinf((দশমিক) (tmp.y())));
        }

        all_inf = warp_reduce_all<warp_আকার>(all_inf);
        যদি (tid % warp_আকার == 0) {
            buf_iw[tid / warp_আকার] = all_inf;
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);
        all_inf = buf_iw[tid % warp_আকার];
        item_ct1.barrier(sycl::access::fence_space::local_space);
        all_inf = warp_reduce_all<warp_আকার>(all_inf);

        যদি (!all_inf) {
            বিরতি;
        }
    }

    // If the break in the loop was not triggered, KV_max_sj is now -FATTN_KQ_STRIDE.
    // If the break was triggered it's the lower edge of the tile with the first non-masked values.
    // In either case, walk back the decrementation by FATTN_KQ_STRIDE.
    KV_max_sj += FATTN_KQ_STRIDE;

    যদি (item_ct1.get_local_id(2) != 0) {
        ফেরত;
    }

    KV_max[sequence*ne31 + jt] = KV_max_sj;
}

template <পূর্ণসংখ্যা D, পূর্ণসংখ্যা ncols1, পূর্ণসংখ্যা ncols2>  // D == head size

স্থির শূন্য flash_attn_stream_k_fixup(দশমিক * __restrict__ dst,
                                      ধ্রুবক sycl::float2 * __restrict__ dst_fixup,
                                      ধ্রুবক পূর্ণসংখ্যা ne01,
                                      ধ্রুবক পূর্ণসংখ্যা ne02,
                                      ধ্রুবক পূর্ণসংখ্যা ne03,
                                      ধ্রুবক পূর্ণসংখ্যা ne11,
                                      ধ্রুবক পূর্ণসংখ্যা ne12,
                                      ধ্রুবক পূর্ণসংখ্যা nbatch_fa) {
    auto          item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr পূর্ণসংখ্যা ncols    = ncols1 * ncols2;

    ধ্রুবক পূর্ণসংখ্যা bidx0 = item_ct1.get_group(2);
    ধ্রুবক পূর্ণসংখ্যা j     = item_ct1.get_group(1);
    ধ্রুবক পূর্ণসংখ্যা c     = item_ct1.get_group(0);
    ধ্রুবক পূর্ণসংখ্যা jc    = j*ncols2 + c;
    ধ্রুবক পূর্ণসংখ্যা tid   = item_ct1.get_local_id(2);

    ধ্রুবক দশমিক * dst_fixup_তথ্য = ((ধ্রুবক দশমিক *) dst_fixup) + item_ct1.get_group_range(2) * (2 * 2 * ncols);

    ধ্রুবক পূর্ণসংখ্যা gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.

    ধ্রুবক পূর্ণসংখ্যা iter_k     = (ne11      + (nbatch_fa - 1)) / nbatch_fa;
    ধ্রুবক পূর্ণসংখ্যা iter_j     = (ne01      + (ncols1    - 1)) / ncols1;
    ধ্রুবক পূর্ণসংখ্যা iter_z_gqa = (gqa_ratio + (ncols2    - 1)) / ncols2;

    ধ্রুবক পূর্ণসংখ্যা kbc0 = int64_t(bidx0 + 0) * (iter_k * iter_j * iter_z_gqa * ne12 * ne03) / item_ct1.get_group_range(2);
    ধ্রুবক পূর্ণসংখ্যা kbc0_stop =
        int64_t(bidx0 + 1) * (iter_k * iter_j * iter_z_gqa * ne12 * ne03) / item_ct1.get_group_range(2);

    ধ্রুবক যৌক্তিক did_not_have_any_তথ্য   = kbc0 == kbc0_stop;
    ধ্রুবক যৌক্তিক wrote_beginning_of_tile = kbc0 % iter_k == 0;
    ধ্রুবক যৌক্তিক did_not_write_last      = kbc0/iter_k == kbc0_stop/iter_k && kbc0_stop % iter_k != 0;
    যদি (did_not_have_any_তথ্য || wrote_beginning_of_tile || did_not_write_last) {
        ফেরত;
    }

    // z_KV == K/V head index, zt_gqa = Q head start index per K/V head, jt = token position start index
    ধ্রুবক পূর্ণসংখ্যা sequence =  kbc0 /(iter_k*iter_j*iter_z_gqa*ne12);
    ধ্রুবক পূর্ণসংখ্যা z_KV     = (kbc0 - iter_k*iter_j*iter_z_gqa*ne12 * sequence)/(iter_k*iter_j*iter_z_gqa);
    ধ্রুবক পূর্ণসংখ্যা zt_gqa   = (kbc0 - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV)/(iter_k*iter_j);
    ধ্রুবক পূর্ণসংখ্যা jt       = (kbc0 - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV - iter_k*iter_j * zt_gqa) / iter_k;

    ধ্রুবক পূর্ণসংখ্যা zt_Q = z_KV*gqa_ratio + zt_gqa*ncols2; // Global Q head start index.

    যদি (jt*ncols1 + j >= ne01 || zt_gqa*ncols2 + c >= gqa_ratio) {
        ফেরত;
    }

    dst += sequence*ne02*ne01*D + jt*ne02*(ncols1*D) + zt_Q*D + (j*ne02 + c)*D + tid;

    // Load the partial result that needs a fixup:
    দশমিক dst_val = 0.0f;
    দশমিক max_val = 0.0f;
    দশমিক rowsum  = 0.0f;
    {
        dst_val = *dst;

        ধ্রুবক sycl::float2 tmp = dst_fixup[bidx0 * ncols + jc];
        max_val                = tmp.x();
        rowsum                 = tmp.y();
    }

    // Iterate over previous blocks and compute the combined results.
    // All SYCL blocks that get here must have a previous block that needs a fixup.
    পূর্ণসংখ্যা bidx = bidx0 - 1;
    পূর্ণসংখ্যা kbc_stop = kbc0;
    যখন(সত্য) {
        ধ্রুবক পূর্ণসংখ্যা kbc = int64_t(bidx) * (iter_k * iter_j * iter_z_gqa * ne12 * ne03) / item_ct1.get_group_range(2);
        যদি (kbc == kbc_stop) { // Did not have any data.
            bidx--;
            kbc_stop = kbc;
            চালু;
        }

        ধ্রুবক দশমিক dst_add = dst_fixup_তথ্য[bidx*ncols*D + jc*D + tid];

        ধ্রুবক sycl::float2 tmp = dst_fixup[(item_ct1.get_group_range(2) + bidx) * ncols + jc];

        // Scale the current and new value accumulators depending on the max. values.
        ধ্রুবক দশমিক max_val_new = sycl::fmax(max_val, tmp.x());

        ধ্রুবক দশমিক diff_val = max_val - max_val_new;
        ধ্রুবক দশমিক diff_add = tmp.x() - max_val_new;

        ধ্রুবক দশমিক scale_val = diff_val >= SOFTMAX_FTZ_THRESHOLD ? sycl::native::exp(diff_val) : 0.0f;
        ধ্রুবক দশমিক scale_add = diff_add >= SOFTMAX_FTZ_THRESHOLD ? sycl::native::exp(diff_add) : 0.0f;

        dst_val = scale_val*dst_val + scale_add*dst_add;
        rowsum  = scale_val * rowsum + scale_add * tmp.y();

        max_val = max_val_new;

        // If this block started in a previous tile we are done and don't need to combine additional partial results.
        যদি (kbc % iter_k == 0 || kbc/iter_k < kbc0/iter_k) {
            বিরতি;
        }
        bidx--;
        kbc_stop = kbc;
    }

    // Write back final result:
    *dst = dst_val / rowsum;
}

template <পূর্ণসংখ্যা D>  // D == head size

স্থির শূন্য flash_attn_combine_results(ধ্রুবক দশমিক * __restrict__ VKQ_parts,
                                       ধ্রুবক sycl::float2 * __restrict__ VKQ_meta,
                                       দশমিক * __restrict__ dst,
                                       ধ্রুবক পূর্ণসংখ্যা parallel_blocks,
                                       uint8_t * dpct_local) {
    // Dimension 0: threadIdx.x
    // Dimension 1: blockIdx.x
    // Dimension 2: blockIdx.y
    // Dimension 3: blockIdx.z
    // Memory layout is permuted with [0, 2, 1, 3]

    auto      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ধ্রুবক পূর্ণসংখ্যা ne01     = item_ct1.get_group_range(2);
    ধ্রুবক পূর্ণসংখ্যা ne02     = item_ct1.get_group_range(1);

    ধ্রুবক পূর্ণসংখ্যা col      = item_ct1.get_group(2);
    ধ্রুবক পূর্ণসংখ্যা শীর্ষ     = item_ct1.get_group(1);
    ধ্রুবক পূর্ণসংখ্যা sequence = item_ct1.get_group(0);

    ধ্রুবক পূর্ণসংখ্যা j_dst_unrolled = (sequence*ne01 + col)*ne02 + শীর্ষ;

    VKQ_parts += j_dst_unrolled * parallel_blocks*D;
    VKQ_meta  += j_dst_unrolled * parallel_blocks;
    dst       += j_dst_unrolled *                 D;

    ধ্রুবক পূর্ণসংখ্যা tid = item_ct1.get_local_id(2);
    __builtin_assume(tid < D);

    auto meta = (sycl::float2 *) dpct_local;
    জন্য (পূর্ণসংখ্যা i = tid; i < 2*parallel_blocks; i += D) {
        ((দশমিক *) meta)[i] = ((ধ্রুবক দশমিক *)VKQ_meta) [i];
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    দশমিক kqmax = meta[0].x();
    জন্য (পূর্ণসংখ্যা l = 1; l < parallel_blocks; ++l) {
        kqmax = sycl::max(kqmax, meta[l].x());
    }

    দশমিক VKQ_numerator   = 0.0f;
    দশমিক VKQ_denominator = 0.0f;
    জন্য (পূর্ণসংখ্যা l = 0; l < parallel_blocks; ++l) {
        ধ্রুবক দশমিক KQ_max_scale = sycl::native::exp(meta[l].x() - kqmax);

        VKQ_numerator   += KQ_max_scale * VKQ_parts[l*D + tid];
        VKQ_denominator += KQ_max_scale * meta[l].y();
    }

    dst[tid] = VKQ_numerator / VKQ_denominator;
}

// Carries the sub-group size (and grf_size_automatic when use_large_grf) as
// kernel properties via get(properties_tag), in place of the
// reqd_sub_group_size attribute and the deprecated parallel_for overload.
template <typename KernelFunc, uint32_t আকার_4, যৌক্তিক UseGrf>
কাঠামো props_kernel {
    KernelFunc kernel;

    শূন্য operator()(sycl::nd_item<3> item) ধ্রুবক {
        kernel(item);
    }

    auto get(sycl::ext::oneapi::experimental::properties_tag) ধ্রুবক {
        যদি constexpr (UseGrf) {
            ফেরত sycl::ext::oneapi::experimental::properties{
                sycl::ext::intel::experimental::grf_আকার_automatic,
                sycl::ext::oneapi::experimental::sub_group_আকার<আকার_4> };
        } অন্যথা {
            ফেরত sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::sub_group_আকার<আকার_4> };
        }
    }
};

template <fattn_kernel_t fattn_kernel, পূর্ণসংখ্যা warp_আকার, যৌক্তিক use_large_grf = মিথ্যা>
স্থির শূন্য lauch_kernel(
    dpct::dim3 group_range,
    dpct::dim3 local_range,
    queue_ptr q,
    unsigned পূর্ণসংখ্যা local_mem_আকার_2,
    ধ্রুবক অক্ষর* __restrict__ Q,
    ধ্রুবক অক্ষর* __restrict__ K,
    ধ্রুবক অক্ষর* __restrict__ V,
    ধ্রুবক অক্ষর* __restrict__ mask,
    ধ্রুবক অক্ষর* __restrict__ sinks,
    ধ্রুবক পূর্ণসংখ্যা* __restrict__ KV_max,
    দশমিক* __restrict__ dst,
    sycl::float2* __restrict__ dst_meta,
    ধ্রুবক দশমিক scale,
    ধ্রুবক দশমিক max_bias,
    ধ্রুবক দশমিক m0,
    ধ্রুবক দশমিক m1,
    ধ্রুবক uint32_t n_শীর্ষ_log2,
    ধ্রুবক দশমিক logit_softcap,
    ধ্রুবক int32_t ne00,
    ধ্রুবক sycl::uint3 ne01,
    ধ্রুবক int32_t ne02,
    ধ্রুবক int32_t ne03,
    ধ্রুবক int32_t nb01,
    ধ্রুবক int32_t nb02,
    ধ্রুবক int32_t nb03,
    ধ্রুবক int32_t ne10,
    ধ্রুবক int32_t ne11,
    ধ্রুবক int32_t ne12,
    ধ্রুবক int32_t ne13,
    ধ্রুবক int32_t nb11,
    ধ্রুবক int32_t nb12,
    ধ্রুবক int64_t nb13,
    ধ্রুবক int32_t nb21,
    ধ্রুবক int32_t nb22,
    ধ্রুবক int64_t nb23,
    ধ্রুবক int32_t ne31,
    ধ্রুবক int32_t ne32,
    ধ্রুবক int32_t ne33,
    ধ্রুবক int32_t nb31,
    ধ্রুবক int32_t nb32,
    ধ্রুবক int64_t nb33) {
    GGML_UNUSED(local_mem_আকার_2);

    ধ্রুবক auto rng = sycl::nd_range<3>(
        static_cast<sycl::range<3>>(group_range * local_range),
        static_cast<sycl::range<3>>(local_range));

    ধ্রুবক auto kernel = [=](sycl::nd_item<3> item_ct1) {
        GGML_UNUSED(item_ct1);
        fattn_kernel(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
                     max_bias, m0, m1, n_শীর্ষ_log2, logit_softcap, ne00,
                     ne01, ne02, ne03, nb01, nb02, nb03, ne10, ne11,
                     ne12, ne13, nb11, nb12, nb13, nb21, nb22, nb23,
                     ne31, ne32, ne33, nb31, nb32, nb33);
    };

    q->submit([&](sycl::handler &cgh) {
        // grf_size_automatic lets the compiler use the large register file when the kernel needs it. grf_size<256> would say the same but is undefined on devices that do not have it.
        cgh.parallel_for(rng, props_kernel<decltype(kernel), warp_আকার, use_large_grf>{kernel});
    });
}

// use_large_grf lets the compiler choose the large register file. Opt in where spilling limits the kernel, not thread occupancy.
template <পূর্ণসংখ্যা DV, পূর্ণসংখ্যা ncols1, পূর্ণসংখ্যা ncols2, fattn_kernel_t fattn_kernel, পূর্ণসংখ্যা warp_আকার, যৌক্তিক use_large_grf = মিথ্যা>
শূন্য launch_fattn(
    ggml_backend_sycl_context & ctx, ggml_tensor * dst, ধ্রুবক পূর্ণসংখ্যা nwarps, ধ্রুবক size_t nbytes_shared,
    ধ্রুবক পূর্ণসংখ্যা nbatch_fa, ধ্রুবক যৌক্তিক need_f16_K, ধ্রুবক যৌক্তিক need_f16_V, ধ্রুবক যৌক্তিক stream_k) {

    constexpr পূর্ণসংখ্যা ncols = ncols1 * ncols2;

    ধ্রুবক ggml_tensor * Q = dst->src[0];
    ধ্রুবক ggml_tensor * K = dst->src[1];
    ধ্রুবক ggml_tensor * V = dst->src[2];

    ধ্রুবক যৌক্তিক V_is_K_view = V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));

    ধ্রুবক ggml_tensor * mask  = dst->src[3];
    ধ্রুবক ggml_tensor * sinks = dst->src[4];

    ggml_tensor * KQV = dst;

    GGML_ASSERT(Q->type == GGML_TYPE_F32);
    GGML_ASSERT(KQV->type == GGML_TYPE_F32);

    GGML_ASSERT(Q->nb[0] == ggml_element_আকার(Q));
    GGML_ASSERT(K->nb[0] == ggml_element_আকার(K));
    GGML_ASSERT(V->nb[0] == ggml_element_আকার(V));

    GGML_ASSERT(!mask || mask->type == GGML_TYPE_F16);

    ggml_sycl_pool & pool = ctx.pool();
    dpct::queue_ptr  main_stream = ctx.stream();
    ধ্রুবক পূর্ণসংখ্যা id  = ggml_sycl_get_device();
    ধ্রুবক পূর্ণসংখ্যা nsm = ggml_sycl_info().devices[id].nsm;

    ধ্রুবক ggml_sycl_fattn_extra  extra = ggml_sycl_fattn_get_extra(dst);
    ggml_sycl_pool_alloc<পূর্ণসংখ্যা>    KV_max(pool);
    ggml_sycl_pool_alloc<দশমিক>  dst_tmp(pool);
    ggml_sycl_pool_alloc<sycl::float2> dst_tmp_meta(pool);

    ধ্রুবক অক্ষর * K_তথ্য = (ধ্রুবক অক্ষর *) K->তথ্য;
    size_t nb11 = K->nb[1];
    size_t nb12 = K->nb[2];
    size_t nb13 = K->nb[3];

    ধ্রুবক অক্ষর * V_তথ্য = (ধ্রুবক অক্ষর *) V->তথ্য;
    size_t nb21 = V->nb[1];
    size_t nb22 = V->nb[2];
    size_t nb23 = V->nb[3];

    যদি (need_f16_K && K->type != GGML_TYPE_F16) {
        ধ্রুবক size_t bs = ggml_blck_আকার(K->type);
        ধ্রুবক size_t ts = ggml_type_আকার(K->type);

        GGML_ASSERT(extra.K_বাফার_ptr);
        sycl::half * K_f16_ptr = (sycl::half *) extra.K_বাফার_ptr;
        যদি (ggml_is_contiguously_allocated(K)) {
            to_fp16_sycl_t to_fp16 = ggml_get_to_fp16_sycl(K->type, dst);
            to_fp16(K_তথ্য, K_f16_ptr, ggml_nelements(K), main_stream);

            nb11 = nb11 * bs * আকার(sycl::half) / ts;
            nb12 = nb12 * bs * আকার(sycl::half) / ts;
            nb13 = nb13 * bs * আকার(sycl::half) / ts;
        } অন্যথা {
            GGML_ASSERT(K->nb[0] == ts);
            to_fp16_nc_sycl_t to_fp16 = ggml_get_to_fp16_nc_sycl(K->type);
            ধ্রুবক int64_t s01 = nb11 / ts;
            ধ্রুবক int64_t s02 = nb12 / ts;
            ধ্রুবক int64_t s03 = nb13 / ts;
            to_fp16(K_তথ্য, K_f16_ptr, K->ne[0], K->ne[1], K->ne[2], K->ne[3], s01, s02, s03, main_stream);

            nb11 = K->ne[0] * আকার(sycl::half);
            nb12 = K->ne[1] * nb11;
            nb13 = K->ne[2] * nb12;
        }
        K_তথ্য = (অক্ষর *) K_f16_ptr;
    }

    যদি (need_f16_V && V->type != GGML_TYPE_F16) {
        যদি (V_is_K_view) {
            V_তথ্য = K_তথ্য;
            nb21   = nb11;
            nb22   = nb12;
            nb23   = nb13;
        } অন্যথা {
            ধ্রুবক size_t bs = ggml_blck_আকার(V->type);
            ধ্রুবক size_t ts = ggml_type_আকার(V->type);

            GGML_ASSERT(extra.V_বাফার_ptr);
            sycl::half * V_f16_ptr = (sycl::half *) extra.V_বাফার_ptr;
            যদি (ggml_is_contiguously_allocated(V)) {
                to_fp16_sycl_t to_fp16 = ggml_get_to_fp16_sycl(V->type, dst);
                to_fp16(V_তথ্য, V_f16_ptr, ggml_nelements(V), main_stream);
                V_তথ্য = (অক্ষর *) V_f16_ptr;

                nb21 = nb21 * bs * আকার(sycl::half) / ts;
                nb22 = nb22 * bs * আকার(sycl::half) / ts;
                nb23 = nb23 * bs * আকার(sycl::half) / ts;
            } অন্যথা {
                GGML_ASSERT(V->nb[0] == ts);
                to_fp16_nc_sycl_t to_fp16 = ggml_get_to_fp16_nc_sycl(V->type);
                ধ্রুবক int64_t s01 = nb21 / ts;
                ধ্রুবক int64_t s02 = nb22 / ts;
                ধ্রুবক int64_t s03 = nb23 / ts;
                to_fp16(V_তথ্য, V_f16_ptr, V->ne[0], V->ne[1], V->ne[2], V->ne[3], s01, s02, s03, main_stream);

                nb21 = V->ne[0] * আকার(sycl::half);
                nb22 = V->ne[1] * nb21;
                nb23 = V->ne[2] * nb22;
            }
            V_তথ্য = (অক্ষর *) V_f16_ptr;
        }
    }

    ধ্রুবক পূর্ণসংখ্যা ntiles_x     = ((Q->ne[1] + ncols1 - 1) / ncols1);
    ধ্রুবক পূর্ণসংখ্যা gqa_ratio    = Q->ne[2] / K->ne[2];
    ধ্রুবক পূর্ণসংখ্যা ntiles_z_gqa = ((gqa_ratio + ncols2 - 1) / ncols2);
    ধ্রুবক পূর্ণসংখ্যা ntiles_যোগ = ntiles_x * ntiles_z_gqa * K->ne[2] * Q->ne[3];

    // Optional optimization where the mask is scanned to determine whether part of the calculation can be skipped.
    // Only worth the overhead if there is at lease one FATTN_KQ_STRIDE x FATTN_KQ_STRIDE square to be skipped or
    //     multiple sequences of possibly different lengths.
    যদি (mask && K->ne[1] % FATTN_KQ_STRIDE == 0 && (Q->ne[1] >= 1024 || Q->ne[3] > 1)) {
        ধ্রুবক পূর্ণসংখ্যা s31 = mask->nb[1] / আকার(sycl::half2);
        ধ্রুবক পূর্ণসংখ্যা s33 = mask->nb[3] / আকার(sycl::half2);

        ধ্রুবক dpct::dim3 blocks_অঙ্ক_KV_max(ntiles_x, Q->ne[3], 1);
        ধ্রুবক dpct::dim3 block_dim_KV_max(FATTN_KQ_STRIDE / 2, 1, 1);

        ধ্রুবক পূর্ণসংখ্যা ne_KV_max = blocks_অঙ্ক_KV_max.x*blocks_অঙ্ক_KV_max.y;
        ধ্রুবক পূর্ণসংখ্যা iter_k = K->ne[1] / FATTN_KQ_STRIDE;

        KV_max.alloc(ne_KV_max);
        {
            dpct::has_capability_or_fail(main_stream->get_device(), { sycl::aspect::fp16 });

            main_stream->submit([&](sycl::handler & cgh) {
                sycl::local_accessor<পূর্ণসংখ্যা, 1> buf_iw_acc_ct1(sycl::range<1>(warp_আকার), cgh);

                auto mask_তথ্য_ct0  = (ধ্রুবক sycl::half2 *) mask->তথ্য;
                auto KV_max_ptr_ct1 = KV_max.ptr;

                cgh.parallel_for(sycl::nd_range<3>(blocks_অঙ্ক_KV_max * block_dim_KV_max, block_dim_KV_max),
                                 [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_আকার(warp_আকার)]] {
                                     GGML_UNUSED(item_ct1);
                                     flash_attn_mask_to_KV_max<ncols1, warp_আকার>(
                                         mask_তথ্য_ct0, KV_max_ptr_ct1, iter_k, s31, s33,
                                         buf_iw_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                                 });
            });
        }
        SYCL_CHECK(0);
    }

    ধ্রুবক dpct::dim3 block_dim(warp_আকার, nwarps, 1);

    // Max. number of active blocks limited by occupancy.
    পূর্ণসংখ্যা max_blocks_per_sm = ggml_sycl_info().devices[id].max_wg_per_cu;
    পূর্ণসংখ্যা parallel_blocks = max_blocks_per_sm;
    dpct::dim3 blocks_অঙ্ক;
    যদি (stream_k) {
        // For short contexts it can be faster to have the SMs work on whole tiles because this lets us skip the fixup.
        ধ্রুবক পূর্ণসংখ্যা max_blocks = max_blocks_per_sm*nsm;
        ধ্রুবক পূর্ণসংখ্যা nblocks_stream_k = max_blocks;
        ধ্রুবক যৌক্তিক use_stream_k = সত্য;

        blocks_অঙ্ক.x = use_stream_k ? nblocks_stream_k : ntiles_যোগ;
        blocks_অঙ্ক.y = 1;
        blocks_অঙ্ক.z = 1;

        যদি (ntiles_যোগ % blocks_অঙ্ক.x != 0) { // Fixup is only needed if the SMs work on fractional tiles.
            dst_tmp_meta.alloc((size_t(blocks_অঙ্ক.x) * ncols * (2 + DV/2)));
        }
    } অন্যথা {
        ধ্রুবক পূর্ণসংখ্যা ntiles_KQ = (K->ne[1] + nbatch_fa - 1) / nbatch_fa; // Max. number of parallel blocks limited by tensor size.

        // parallel_blocks must not be larger than what the tensor size allows:
        parallel_blocks = std::min(parallel_blocks, ntiles_KQ);
        // todo fix the hard code change
        // parallel_blocks = ntiles_KQ;

        // If ntiles_total % blocks_per_wave != 0 then some efficiency is lost due to tail effects.
        // Test whether parallel_blocks can be set to a higher value for better efficiency.
        ধ্রুবক পূর্ণসংখ্যা blocks_per_wave = nsm * max_blocks_per_sm;
        পূর্ণসংখ্যা nwaves_best = 0;
        পূর্ণসংখ্যা efficiency_percent_best = 0;
        জন্য (পূর্ণসংখ্যা parallel_blocks_test = parallel_blocks; parallel_blocks_test <= ntiles_KQ; ++parallel_blocks_test) {
            ধ্রুবক পূর্ণসংখ্যা nblocks_যোগ = ntiles_যোগ * parallel_blocks_test;
            ধ্রুবক পূর্ণসংখ্যা nwaves = (nblocks_যোগ + blocks_per_wave - 1) / blocks_per_wave;
            ধ্রুবক পূর্ণসংখ্যা efficiency_percent = 100 * nblocks_যোগ / (nwaves*blocks_per_wave);

            // Stop trying configurations with more waves if we already have good efficiency to avoid excessive overhead.
            যদি (efficiency_percent_best >= 95 && nwaves > nwaves_best) {
                বিরতি;
            }

            যদি (efficiency_percent > efficiency_percent_best) {
                nwaves_best = nwaves;
                efficiency_percent_best = efficiency_percent;
                parallel_blocks = parallel_blocks_test;
            }
        }

        blocks_অঙ্ক.x = ntiles_x;
        blocks_অঙ্ক.y = parallel_blocks;
        blocks_অঙ্ক.z = ntiles_z_gqa*K->ne[2]*Q->ne[3];

        যদি (parallel_blocks > 1) {
            dst_tmp.alloc(parallel_blocks*ggml_nelements(KQV));
            dst_tmp_meta.alloc(parallel_blocks*ggml_nrows(KQV));
        }
    }

    দশমিক scale         = 1.0f;
    দশমিক max_bias      = 0.0f;
    দশমিক logit_softcap = 0.0f;

    memcpy(&scale,         (ধ্রুবক দশমিক *) KQV->op_params + 0, আকার(দশমিক));
    memcpy(&max_bias,      (ধ্রুবক দশমিক *) KQV->op_params + 1, আকার(দশমিক));
    memcpy(&logit_softcap, (ধ্রুবক দশমিক *) KQV->op_params + 2, আকার(দশমিক));

    যদি (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    ধ্রুবক uint32_t n_শীর্ষ      = Q->ne[2];
    ধ্রুবক uint32_t n_শীর্ষ_log2 = 1u << uint32_t(floorf(log2f(দশমিক(n_শীর্ষ))));

    ধ্রুবক দশমিক m0 = powf(2.0f, -(max_bias       ) / n_শীর্ষ_log2);
    ধ্রুবক দশমিক m1 = powf(2.0f, -(max_bias / 2.0f) / n_শীর্ষ_log2);

    // TODO other tensor dimensions after removal of WMMA kernel:
    ধ্রুবক sycl::uint3 ne01 = init_fastdiv_মানসমূহ(Q->ne[1]);

    GGML_ASSERT(block_dim.x % warp_আকার == 0);

    lauch_kernel<fattn_kernel, warp_আকার, use_large_grf>(
        blocks_অঙ্ক, block_dim, main_stream, (unsigned পূর্ণসংখ্যা) nbytes_shared, (ধ্রুবক অক্ষর *) Q->তথ্য, K_তথ্য, V_তথ্য,
        mask ? ((ধ্রুবক অক্ষর *) mask->তথ্য) : nullptr, sinks ? ((ধ্রুবক অক্ষর *) sinks->তথ্য) : nullptr, KV_max.ptr,
        !stream_k && parallel_blocks > 1 ? dst_tmp.ptr : (দশমিক *) KQV->তথ্য, (sycl::float2 *)dst_tmp_meta.ptr, scale, max_bias, m0, m1,
        n_শীর্ষ_log2, logit_softcap, Q->ne[0], ne01, Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3], K->ne[0],
        K->ne[1], K->ne[2], K->ne[3], nb11, nb12, nb13, nb21, nb22, nb23, mask ? mask->ne[1] : 0,
        mask ? mask->ne[2] : 0, mask ? mask->ne[3] : 0, mask ? mask->nb[1] : 0, mask ? mask->nb[2] : 0,
        mask ? mask->nb[3] : 0);
    SYCL_CHECK(0);

    যদি (stream_k) {
        যদি (ntiles_যোগ % blocks_অঙ্ক.x != 0) { // Fixup is only needed if the SMs work on fractional tiles.
            ধ্রুবক dpct::dim3 block_dim_combine(DV, 1, 1);
            ধ্রুবক dpct::dim3 blocks_অঙ্ক_combine = { blocks_অঙ্ক.x, ncols1, ncols2 };

            main_stream->submit([&](sycl::handler & cgh) {
                auto KQV_তথ্য_ct0         = (দশমিক *) KQV->তথ্য;
                auto dst_tmp_meta_ptr_ct1 = dst_tmp_meta.ptr;
                auto Q_ne_ct2             = Q->ne[1];
                auto Q_ne_ct3             = Q->ne[2];
                auto Q_ne_ct4             = Q->ne[3];
                auto K_ne_ct5             = K->ne[1];
                auto K_ne_ct6             = K->ne[2];

                cgh.parallel_for(sycl::nd_range<3>(blocks_অঙ্ক_combine * block_dim_combine, block_dim_combine),
                                 [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_আকার(warp_আকার)]] {
                                     GGML_UNUSED(item_ct1);
                                     flash_attn_stream_k_fixup<DV, ncols1, ncols2>(KQV_তথ্য_ct0, dst_tmp_meta_ptr_ct1,
                                                                                   Q_ne_ct2, Q_ne_ct3, Q_ne_ct4,
                                                                                   K_ne_ct5, K_ne_ct6, nbatch_fa);
                                 });
            });
        }
    } অন্যথা যদি (parallel_blocks > 1) {
        ধ্রুবক dpct::dim3 block_dim_combine(DV, 1, 1);
        ধ্রুবক dpct::dim3 blocks_অঙ্ক_combine(Q->ne[1], Q->ne[2], Q->ne[3]);
        ধ্রুবক size_t     nbytes_shared_combine = parallel_blocks * আকার(sycl::float2);
        main_stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(sycl::range<1>(nbytes_shared_combine), cgh);

            auto dst_tmp_ptr_ct0      = dst_tmp.ptr;
            auto dst_tmp_meta_ptr_ct1 = dst_tmp_meta.ptr;
            auto KQV_তথ্য_ct2         = (দশমিক *) KQV->তথ্য;

            cgh.parallel_for(sycl::nd_range<3>(blocks_অঙ্ক_combine * block_dim_combine, block_dim_combine),
                             [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_আকার(warp_আকার)]] {
                                 GGML_UNUSED(item_ct1);
                                 flash_attn_combine_results<DV>(
                                     dst_tmp_ptr_ct0, dst_tmp_meta_ptr_ct1, KQV_তথ্য_ct2, parallel_blocks,
                                     dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                             });
        });
    }
    SYCL_CHECK(0);
}
