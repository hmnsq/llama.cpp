#आशय once

#समावेश <sycl/sycl.hpp>
#समावेश <sycl/ext/intel/experimental/grf_size_properties.hpp>
#समावेश "dpct/helper.hpp"
#समावेश "common.hpp"
#समावेश "convert.hpp"
#समावेश "vecdotq.hpp"
#समावेश "fattn.hpp"

#समावेश "ggml.h"

#समावेश <cstdint>
#समावेश <cmath>
#समावेश <float.h>


#घोषणा FATTN_KQ_STRIDE       256
#घोषणा HALF_MAX_HALF         sycl::half(65504.0f/2) // Use neg. of this instead of -INFINITY to initialize KQ max vals to avoid NaN upon subtraction.
#घोषणा SOFTMAX_FTZ_THRESHOLD -20.0f                   // Softmax exp. of values smaller than this are flushed to zero to avoid NaNs.
#घोषणा FATTN_KQ_MAX_OFFSET (3.0f*0.6931f)

प्रकार व्योम (*fattn_kernel_t)(
    स्थिर अक्षर* Q,
    स्थिर अक्षर* K,
    स्थिर अक्षर* V,
    स्थिर अक्षर* mask,
    स्थिर अक्षर* sinks,
    स्थिर पूर्णांक* KV_max,
    भग्न* dst,
    sycl::float2* dst_meta,
    स्थिर भग्न scale,
    स्थिर भग्न max_bias,
    स्थिर भग्न m0,
    स्थिर भग्न m1,
    स्थिर uint32_t n_शीर्ष_log2,
    स्थिर भग्न logit_softcap,
    स्थिर int32_t ne00,
    स्थिर sycl::uint3 ne01,
    स्थिर int32_t ne02,
    स्थिर int32_t ne03,
    स्थिर int32_t nb01,
    स्थिर int32_t nb02,
    स्थिर int32_t nb03,
    स्थिर int32_t ne10,
    स्थिर int32_t ne11,
    स्थिर int32_t ne12,
    स्थिर int32_t ne13,
    स्थिर int32_t nb11,
    स्थिर int32_t nb12,
    स्थिर int64_t nb13,
    स्थिर int32_t nb21,
    स्थिर int32_t nb22,
    स्थिर int64_t nb23,
    स्थिर int32_t ne31,
    स्थिर int32_t ne32,
    स्थिर int32_t ne33,
    स्थिर int32_t nb31,
    स्थिर int32_t nb32,
    स्थिर int64_t nb33);

प्रकार भग्न (*vec_dot_KQ_t)(
    स्थिर अक्षर * __restrict__ K_c, स्थिर व्योम * __restrict__ Q_v, स्थिर पूर्णांक * __restrict__ Q_q8 , स्थिर व्योम * __restrict__ Q_ds);

ढाँचा <पूर्णांक D, पूर्णांक nthreads>
अटल __dpct_inline__ भग्न vec_dot_fattn_vec_KQ_f16(स्थिर अक्षर * __restrict__ K_c,
                                                      स्थिर व्योम * __restrict__ Q_v,
                                                      स्थिर पूर्णांक * __restrict__ Q_q8,
                                                      स्थिर व्योम * __restrict__ Q_ds_v) {
    स्थिर sycl::half2 * K_h2 = (स्थिर sycl::half2 *) K_c;
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v);

    constexpr पूर्णांक cpy_nb = ggml_sycl_get_max_cpy_bytes();
    constexpr पूर्णांक cpy_ne = cpy_nb / 4;

    भग्न योग_2 = 0.0f;

#आशय unroll
    क्रम (पूर्णांक k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads*cpy_ne) {
        sycl::half2 tmp[cpy_ne];
        ggml_sycl_memcpy_1<माप(tmp)>(
            tmp,
            K_h2 + k_KQ_0 + (sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2) % nthreads) * cpy_ne);
#आशय unroll
        क्रम (पूर्णांक k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
#अगर_घोषित GGML_SYCL_F16
            ggml_sycl_mad(योग_2,                tmp[k_KQ_1] , ((स्थिर sycl::half2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#अन्यथा
            ggml_sycl_mad(योग_2, __half22float2(tmp[k_KQ_1]), ((स्थिर sycl::float2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#पूर्ण_अगर // GGML_SYCL_F16
        }
    }

    वापस योग_2;
}

ढाँचा <पूर्णांक D, पूर्णांक nthreads, पूर्णांक warp_आकार>
अटल __dpct_inline__ भग्न vec_dot_fattn_vec_KQ_q4_0(स्थिर अक्षर * __restrict__ K_c,
                                                       स्थिर व्योम * __restrict__ Q_v,
                                                       स्थिर पूर्णांक * __restrict__ Q_q8,
                                                       स्थिर व्योम * __restrict__ Q_ds_v) {
    स्वतः item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    स्थिर block_q4_0 * K_q4_0   = (स्थिर block_q4_0 *) K_c;
    GGML_UNUSED(Q_v);

    भग्न योग_2 = 0.0f;

#आशय unroll
    क्रम (पूर्णांक k_KQ_0 = 0; k_KQ_0 < पूर्णांक(D/माप(पूर्णांक)); k_KQ_0 += nthreads) {
        स्थिर पूर्णांक k_KQ =
            k_KQ_0 + (nthreads == warp_आकार ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads);

        स्थिर पूर्णांक ib    = k_KQ /  QI8_1;
        स्थिर पूर्णांक iqs4  = k_KQ %  QI4_0;
        स्थिर पूर्णांक shift = k_KQ & (QI8_1/2);

        पूर्णांक v;
        ggml_sycl_memcpy_1<माप(पूर्णांक), 2>(&v, K_q4_0[ib].qs + माप(पूर्णांक)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;
        स्थिर पूर्णांक u = Q_q8[k_KQ_0/nthreads];

        स्थिर पूर्णांक sumi = ggml_sycl_dp4a(v, u, 0);

        स्थिर sycl::float2 Q_ds = ((स्थिर sycl::float2 *) Q_ds_v)[k_KQ_0 / nthreads];
        योग_2 += __half2float(K_q4_0[ib].d) * (sumi*Q_ds.x() - (8/QI8_1)*Q_ds.y());
    }

    वापस योग_2;
}

ढाँचा <पूर्णांक D, पूर्णांक nthreads , पूर्णांक warp_आकार>
अटल __dpct_inline__ भग्न vec_dot_fattn_vec_KQ_q4_1(स्थिर अक्षर * __restrict__ K_c,
                                                       स्थिर व्योम * __restrict__ Q_v,
                                                       स्थिर पूर्णांक * __restrict__ Q_q8,
                                                       स्थिर व्योम * __restrict__ Q_ds_v) {
    स्वतः               item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    स्थिर block_q4_1 * K_q4_1   = (स्थिर block_q4_1 *) K_c;
    GGML_UNUSED(Q_v);

    भग्न योग_2 = 0.0f;

#आशय unroll
    क्रम (पूर्णांक k_KQ_0 = 0; k_KQ_0 < पूर्णांक(D/माप(पूर्णांक)); k_KQ_0 += nthreads) {
        स्थिर पूर्णांक k_KQ =
            k_KQ_0 + (nthreads == warp_आकार ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads);

        स्थिर पूर्णांक ib    = k_KQ /  QI8_1;
        स्थिर पूर्णांक iqs4  = k_KQ %  QI4_1;
        स्थिर पूर्णांक shift = k_KQ & (QI8_1/2);

        पूर्णांक v;
        ggml_sycl_memcpy_1<माप(पूर्णांक)>(&v, K_q4_1[ib].qs + माप(पूर्णांक)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;
        स्थिर पूर्णांक u = Q_q8[k_KQ_0/nthreads];

        स्थिर पूर्णांक sumi = ggml_sycl_dp4a(v, u, 0);

        स्थिर sycl::float2 K_dm = (K_q4_1[ib].dm).ढाँचा convert<भग्न, sycl::rounding_mode::automatic>();
        स्थिर sycl::float2 Q_ds = ((स्थिर sycl::float2 *) Q_ds_v)[k_KQ_0 / nthreads];

        योग_2 += K_dm.x()*Q_ds.x()*sumi + K_dm.y()*Q_ds.y()/QI8_1;
    }

    वापस योग_2;
}

ढाँचा <पूर्णांक D, पूर्णांक nthreads, पूर्णांक warp_आकार>
अटल __dpct_inline__ भग्न vec_dot_fattn_vec_KQ_q5_0(स्थिर अक्षर * __restrict__ K_c,
                                                       स्थिर व्योम * __restrict__ Q_v,
                                                       स्थिर पूर्णांक * __restrict__ Q_q8,
                                                       स्थिर व्योम * __restrict__ Q_ds_v) {
    स्वतः               item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    स्थिर block_q5_0 * K_q5_0   = (स्थिर block_q5_0 *) K_c;
    GGML_UNUSED(Q_v);

    भग्न योग_2 = 0.0f;

#आशय unroll
    क्रम (पूर्णांक k_KQ_0 = 0; k_KQ_0 < पूर्णांक(D/माप(पूर्णांक)); k_KQ_0 += nthreads) {
        स्थिर पूर्णांक k_KQ =
            k_KQ_0 + (nthreads == warp_आकार ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads);

        स्थिर पूर्णांक ib    = k_KQ /  QI8_1;
        स्थिर पूर्णांक iqs4  = k_KQ %  QI5_0;
        स्थिर पूर्णांक iqs8  = k_KQ %  QI8_1;
        स्थिर पूर्णांक shift = k_KQ & (QI8_1/2);

        पूर्णांक v;
        ggml_sycl_memcpy_1<माप(पूर्णांक), 2>(&v, K_q5_0[ib].qs + माप(पूर्णांक)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;

        {
            पूर्णांक vh;
            ggml_sycl_memcpy_1<माप(पूर्णांक), 2>(&vh, K_q5_0[ib].qh);
            vh >>= iqs8 * QI5_0;

            v |= (vh <<  4) & 0x00000010; // 0 ->  4
            v |= (vh << 11) & 0x00001000; // 1 -> 12
            v |= (vh << 18) & 0x00100000; // 2 -> 20
            v |= (vh << 25) & 0x10000000; // 3 -> 28
        }

        स्थिर पूर्णांक u = Q_q8[k_KQ_0/nthreads];

        स्थिर पूर्णांक sumi = ggml_sycl_dp4a(v, u, 0);

        स्थिर sycl::float2 Q_ds = ((स्थिर sycl::float2 *) Q_ds_v)[k_KQ_0 / nthreads];

        योग_2 += __half2float(K_q5_0[ib].d) * (sumi*Q_ds.x() - (16/QI8_1)*Q_ds.y());
    }

    वापस योग_2;
}

ढाँचा <पूर्णांक D, पूर्णांक nthreads, पूर्णांक warp_आकार>
अटल __dpct_inline__ भग्न vec_dot_fattn_vec_KQ_q5_1(स्थिर अक्षर * __restrict__ K_c,
                                                       स्थिर व्योम * __restrict__ Q_v,
                                                       स्थिर पूर्णांक * __restrict__ Q_q8,
                                                       स्थिर व्योम * __restrict__ Q_ds_v) {
    स्वतः               item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    स्थिर block_q5_1 * K_q5_1   = (स्थिर block_q5_1 *) K_c;
    GGML_UNUSED(Q_v);

    भग्न योग_2 = 0.0f;

#आशय unroll
    क्रम (पूर्णांक k_KQ_0 = 0; k_KQ_0 < पूर्णांक(D/माप(पूर्णांक)); k_KQ_0 += nthreads) {
        स्थिर पूर्णांक k_KQ =
            k_KQ_0 + (nthreads == warp_आकार ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads);

        स्थिर पूर्णांक ib    = k_KQ /  QI8_1;
        स्थिर पूर्णांक iqs4  = k_KQ %  QI5_1;
        स्थिर पूर्णांक iqs8  = k_KQ %  QI8_1;
        स्थिर पूर्णांक shift = k_KQ & (QI8_1/2);

        पूर्णांक v;
        ggml_sycl_memcpy_1<माप(पूर्णांक)>(&v, K_q5_1[ib].qs + माप(पूर्णांक)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;

        {
            पूर्णांक vh;
            ggml_sycl_memcpy_1<माप(पूर्णांक)>(&vh, K_q5_1[ib].qh);
            vh >>= iqs8 * QI5_0;

            v |= (vh <<  4) & 0x00000010; // 0 ->  4
            v |= (vh << 11) & 0x00001000; // 1 -> 12
            v |= (vh << 18) & 0x00100000; // 2 -> 20
            v |= (vh << 25) & 0x10000000; // 3 -> 28
        }

        स्थिर पूर्णांक u = Q_q8[k_KQ_0/nthreads];

        स्थिर पूर्णांक sumi = ggml_sycl_dp4a(v, u, 0);

        स्थिर sycl::float2 K_dm = (K_q5_1[ib].dm).ढाँचा convert<भग्न, sycl::rounding_mode::automatic>();
        स्थिर sycl::float2 Q_ds = ((स्थिर sycl::float2 *) Q_ds_v)[k_KQ_0 / nthreads];

        योग_2 += K_dm.x()*Q_ds.x()*sumi + K_dm.y()*Q_ds.y()/QI8_1;
    }

    वापस योग_2;
}

ढाँचा <पूर्णांक D, पूर्णांक nthreads, पूर्णांक warp_आकार>
अटल __dpct_inline__ भग्न vec_dot_fattn_vec_KQ_q8_0(स्थिर अक्षर * __restrict__ K_c,
                                                       स्थिर व्योम * __restrict__ Q_v,
                                                       स्थिर पूर्णांक * __restrict__ Q_q8,
                                                       स्थिर व्योम * __restrict__ Q_ds_v) {
    स्वतः               item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    स्थिर block_q8_0 * K_q8_0   = (स्थिर block_q8_0 *) K_c;
    GGML_UNUSED(Q_v);

    भग्न योग_2 = 0.0f;

#आशय unroll
    क्रम (पूर्णांक k_KQ_0 = 0; k_KQ_0 < पूर्णांक(D/माप(पूर्णांक)); k_KQ_0 += nthreads) {
        स्थिर पूर्णांक k_KQ =
            k_KQ_0 + (nthreads == warp_आकार ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads);

        स्थिर पूर्णांक ib  = k_KQ / QI8_0;
        स्थिर पूर्णांक iqs = k_KQ % QI8_0;

        पूर्णांक v;
        ggml_sycl_memcpy_1<माप(v), 2>(&v, K_q8_0[ib].qs + 4*iqs);

        स्थिर sycl::float2 * Q_ds = (स्थिर sycl::float2 *) Q_ds_v;
        स्थिर भग्न          Q_d  = Q_ds[k_KQ_0 / nthreads].x();

        योग_2 += vec_dot_q8_0_q8_1_impl<भग्न, 1>(&v, &Q_q8[k_KQ_0/nthreads], K_q8_0[ib].d, Q_d);
    }

    वापस योग_2;
}

ढाँचा <typename Tds, पूर्णांक ni, पूर्णांक warp_आकार>
अटल __dpct_inline__ व्योम quantize_q8_1_to_shared(स्थिर भग्न * __restrict__ x,
                                                    स्थिर भग्न scale,
                                                    पूर्णांक * __restrict__ yq32,
                                                    व्योम * __restrict__ yds) {
    स्वतः item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    भग्न vals[माप(पूर्णांक)] = { 0.0f };
#आशय unroll
    क्रम (पूर्णांक l = 0; l < पूर्णांक(माप(पूर्णांक)); ++l) {
        vals[l] =
            (ni == warp_आकार || item_ct1.get_local_id(2) < ni) ? scale * x[4 * item_ct1.get_local_id(2) + l] : 0.0f;
    }

    भग्न amax = sycl::भ_असल(vals[0]);
    भग्न योग_2  = vals[0];
#आशय unroll
    क्रम (पूर्णांक l = 1; l < पूर्णांक(माप(पूर्णांक)); ++l) {
        amax = sycl::fmax(amax, sycl::भ_असल(vals[l]));
        योग_2 += vals[l];
    }
#आशय unroll
    क्रम (पूर्णांक mask = QI8_1/2; mask > 0; mask >>= 1) {
        amax = sycl::fmax(
            amax, dpct::permute_sub_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(), amax, mask));
        योग_2 += dpct::permute_sub_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(), योग_2, mask);
    }

    स्थिर भग्न d = amax / 127;
    पूर्णांक q32 = 0;
    int8_t * q8 = (int8_t *) &q32;

    अगर (d != 0.0f) {
#आशय unroll
        क्रम (पूर्णांक l = 0; l < पूर्णांक(माप(पूर्णांक)); ++l) {
            q8[l] = sycl::round(vals[l] / d);
        }
    }

    yq32[item_ct1.get_local_id(2)] = q32;
    अगर (item_ct1.get_local_id(2) % QI8_1 == 0 && (ni == warp_आकार || item_ct1.get_local_id(2) < ni)) {
        अगर (std::is_same<Tds, sycl::half2>::value) {
            ((sycl::half2  *) yds)[item_ct1.get_local_id(2)/QI8_1] =  make_half2(d, योग_2);
        } अन्यथा {
            ((sycl::float2 *) yds)[item_ct1.get_local_id(2)/QI8_1] = make_float2(d, योग_2);
        }
    }
}

प्रकार व्योम (*dequantize_V_t)(स्थिर व्योम *, व्योम *, स्थिर int64_t);

ढाँचा <typename T, पूर्णांक ne>
अटल __dpct_inline__ व्योम dequantize_V_f16(स्थिर व्योम * __restrict__ vx, व्योम * __restrict__ dst, स्थिर int64_t i0) {
    अगर constexpr (std::is_same_v<T, sycl::half>) {
        ggml_sycl_memcpy_1<ne * माप(sycl::half)>(dst, (स्थिर sycl::half *) vx + i0);
    } अन्यथा अगर constexpr (std::is_same_v<T, भग्न>) {
        static_assert(ne % 2 == 0, "bad ne");
        sycl::half2 tmp[ne / 2];
        ggml_sycl_memcpy_1<ne * माप(sycl::half)>(tmp, (स्थिर sycl::half *) vx + i0);
        sycl::float2 * dst_f2 = (sycl::float2 *) dst;
#आशय unroll
        क्रम (पूर्णांक l = 0; l < ne/2; ++l) {
            dst_f2[l] = tmp[l].ढाँचा convert<भग्न, sycl::rounding_mode::automatic>();
        }
    } अन्यथा {
        static_assert(std::is_same_v<T, व्योम>, "unsupported type");
    }
}

ढाँचा <typename T, पूर्णांक ne>
अटल __dpct_inline__ व्योम dequantize_V_q4_0(स्थिर व्योम * __restrict__ vx, व्योम * __restrict__ dst, स्थिर int64_t i0) {
    स्थिर block_q4_0 * x = (स्थिर block_q4_0 *) vx;

    स्थिर int64_t ib    =  i0          /  QK4_0;
    स्थिर पूर्णांक     iqs   =  i0          % (QK4_0/2);
    स्थिर पूर्णांक     shift = (i0 % QK4_0) / (QK4_0/2);

    पूर्णांक q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_sycl_memcpy_1<ne, 2>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;
    q = dpct::vectorized_द्विआधारी<sycl::char4>(q, 0x08080808, dpct::sub_sat());

    स्थिर int8_t * q8 = (स्थिर int8_t *) &q;

#अगर_घोषित GGML_SYCL_F16
    अगर constexpr (std::is_same_v<T, sycl::half>) {
        स्थिर sycl::half2 d = sycl::half2(x[ib].d);

#आशय unroll
        क्रम (पूर्णांक l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * sycl::half2(q8[l0 + 0], q8[l0 + 1]);
        }
    } अन्यथा
#पूर्ण_अगर // GGML_SYCL_F16
    अगर constexpr (std::is_same_v<T, भग्न>) {
        स्थिर भग्न d = x[ib].d;

#आशय unroll
        क्रम (पूर्णांक l = 0; l < ne; ++l) {
            ((भग्न *) dst)[l] = d * q8[l];
        }
    } अन्यथा {
        static_assert(std::is_same_v<T, व्योम>, "bad type");
    }
}

ढाँचा <typename T, पूर्णांक ne>
अटल __dpct_inline__ व्योम dequantize_V_q4_1(स्थिर व्योम * __restrict__ vx, व्योम * __restrict__ dst, स्थिर int64_t i0) {
    स्थिर block_q4_1 * x = (स्थिर block_q4_1 *) vx;

    स्थिर int64_t ib    =  i0          /  QK4_1;
    स्थिर पूर्णांक     iqs   =  i0          % (QK4_1/2);
    स्थिर पूर्णांक     shift = (i0 % QK4_1) / (QK4_1/2);

    पूर्णांक q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_sycl_memcpy_1<ne>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    स्थिर int8_t * q8 = (स्थिर int8_t *) &q;

#अगर_घोषित GGML_SYCL_F16
    अगर constexpr (std::is_same_v<T, sycl::half>) {
        स्थिर sycl::half2 dm = x[ib].dm;
        स्थिर sycl::half2 d  = sycl::half2(dm[0]);
        स्थिर sycl::half2 m  = sycl::half2(dm[1]);

#आशय unroll
        क्रम (पूर्णांक l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * sycl::half2(q8[l0 + 0], q8[l0 + 1]) + m;
        }
    } अन्यथा
#पूर्ण_अगर // GGML_SYCL_F16
    अगर constexpr (std::is_same_v<T, भग्न>) {
        स्थिर sycl::float2 dm = (x[ib].dm).ढाँचा convert<भग्न, sycl::rounding_mode::automatic>();

#आशय unroll
        क्रम (पूर्णांक l = 0; l < ne; ++l) {
            ((भग्न *) dst)[l] = dm.x() * q8[l] + dm.y();
        }
    } अन्यथा {
        static_assert(std::is_same_v<T, व्योम>, "bad type");
    }
}

ढाँचा <typename T, पूर्णांक ne>
अटल __dpct_inline__ व्योम dequantize_V_q5_0(स्थिर व्योम * __restrict__ vx, व्योम * __restrict__ dst, स्थिर int64_t i0) {
    स्थिर block_q5_0 * x = (स्थिर block_q5_0 *) vx;

    स्थिर int64_t ib    =  i0          /  QK5_0;
    स्थिर पूर्णांक     idq   =  i0          %  QK5_0;
    स्थिर पूर्णांक     iqs   =  i0          % (QK5_0/2);
    स्थिर पूर्णांक     shift = (i0 % QK5_0) / (QK5_0/2);

    पूर्णांक q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_sycl_memcpy_1<ne, 2>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    {
        पूर्णांक qh;
        ggml_sycl_memcpy_1<ne, 2>(&qh, x[ib].qh);
#आशय unroll
        क्रम (पूर्णांक l = 0; l < ne; ++l) {
            q |= ((qh >> (idq + l)) & 0x00000001) << (8*l + 4);
        }
    }

    q = dpct::vectorized_द्विआधारी<sycl::char4>(q, 0x10101010, dpct::sub_sat());

    स्थिर int8_t * q8 = (स्थिर int8_t *) &q;

#अगर_घोषित GGML_SYCL_F16
    अगर constexpr (std::is_same_v<T, sycl::half>) {
        स्थिर sycl::half2 d = sycl::half2(x[ib].d);

#आशय unroll
        क्रम (पूर्णांक l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * sycl::half2(q8[l0 + 0], q8[l0 + 1]);
        }
    } अन्यथा
#पूर्ण_अगर // GGML_SYCL_F16
    अगर constexpr (std::is_same_v<T, भग्न>) {
        स्थिर भग्न d = x[ib].d;

#आशय unroll
        क्रम (पूर्णांक l = 0; l < ne; ++l) {
            ((भग्न *) dst)[l] = d * q8[l];
        }
    } अन्यथा {
        static_assert(std::is_same_v<T, व्योम>, "bad type");
    }
}

ढाँचा <typename T, पूर्णांक ne>
अटल __dpct_inline__ व्योम dequantize_V_q5_1(स्थिर व्योम * __restrict__ vx, व्योम * __restrict__ dst, स्थिर int64_t i0) {
    स्थिर block_q5_1 * x = (स्थिर block_q5_1 *) vx;

    स्थिर int64_t ib    =  i0          /  QK5_1;
    स्थिर पूर्णांक     idq   =  i0          %  QK5_1;
    स्थिर पूर्णांक     iqs   =  i0          % (QK5_1/2);
    स्थिर पूर्णांक     shift = (i0 % QK5_1) / (QK5_1/2);

    पूर्णांक q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_sycl_memcpy_1<ne>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    {
        पूर्णांक qh;
        ggml_sycl_memcpy_1<ne>(&qh, x[ib].qh);
#आशय unroll
        क्रम (पूर्णांक l = 0; l < ne; ++l) {
            q |= ((qh >> (idq + l)) & 0x00000001) << (8*l + 4);
        }
    }

    स्थिर int8_t * q8 = (स्थिर int8_t *) &q;

#अगर_घोषित GGML_SYCL_F16
    अगर constexpr (std::is_same_v<T, sycl::half>) {
        स्थिर sycl::half2 dm = x[ib].dm;
        स्थिर sycl::half2 d  = sycl::half2(dm[0]);
        स्थिर sycl::half2 m  = sycl::half2(dm[1]);

#आशय unroll
        क्रम (पूर्णांक l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * sycl::half2(q8[l0 + 0], q8[l0 + 1]) + m;
        }
    } अन्यथा
#पूर्ण_अगर // GGML_SYCL_F16
    अगर constexpr (std::is_same_v<T, भग्न>) {
        स्थिर sycl::float2 dm = (x[ib].dm).ढाँचा convert<भग्न, sycl::rounding_mode::automatic>();

#आशय unroll
        क्रम (पूर्णांक l = 0; l < ne; ++l) {
            ((भग्न *) dst)[l] = dm.x() * q8[l] + dm.y();
        }
    } अन्यथा {
        static_assert(std::is_same_v<T, व्योम>, "bad type");
    }
}

ढाँचा <typename T, पूर्णांक ne>
अटल __dpct_inline__ व्योम dequantize_V_q8_0(स्थिर व्योम * __restrict__ vx, व्योम * __restrict__ dst, स्थिर int64_t i0) {
    स्थिर block_q8_0 * x = (स्थिर block_q8_0 *) vx;

    स्थिर int64_t ib  = i0 / QK8_0;
    स्थिर पूर्णांक     iqs = i0 % QK8_0;

    static_assert(ne % 2 == 0, "bad ne");
    int8_t qs[ne];
    ggml_sycl_memcpy_1<ne, 2>(qs, x[ib].qs + iqs);

#अगर_घोषित GGML_SYCL_F16
    अगर constexpr (std::is_same<T, sycl::half>::value) {
        स्थिर sycl::half2 d = sycl::half2(x[ib].d);

#आशय unroll
        क्रम (पूर्णांक l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * make_half2(qs[l0 + 0], qs[l0 + 1]);
        }
    } अन्यथा
#पूर्ण_अगर // GGML_SYCL_F16
    अगर constexpr (std::is_same<T, भग्न>::value) {
        स्थिर भग्न d = x[ib].d;

#आशय unroll
        क्रम (पूर्णांक l = 0; l < ne; ++l) {
            ((भग्न *) dst)[l] = d * qs[l];
        }
    } अन्यथा {
        static_assert(std::is_same_v<T, व्योम>, "unsupported type");
    }
}

ढाँचा <पूर्णांक type_K, पूर्णांक D, पूर्णांक nthreads, पूर्णांक warp_आकार>
constexpr vec_dot_KQ_t get_vec_dot_KQ() {
    अगर constexpr (type_K == GGML_TYPE_F16) {
        वापस vec_dot_fattn_vec_KQ_f16<D, nthreads>;
    } अन्यथा अगर constexpr (type_K == GGML_TYPE_Q4_0) {
        वापस vec_dot_fattn_vec_KQ_q4_0<D, nthreads, warp_आकार>;
    } अन्यथा अगर constexpr (type_K == GGML_TYPE_Q4_1) {
        वापस vec_dot_fattn_vec_KQ_q4_1<D, nthreads, warp_आकार>;
    } अन्यथा अगर constexpr (type_K == GGML_TYPE_Q5_0) {
        वापस vec_dot_fattn_vec_KQ_q5_0<D, nthreads, warp_आकार>;
    } अन्यथा अगर constexpr (type_K == GGML_TYPE_Q5_1) {
        वापस vec_dot_fattn_vec_KQ_q5_1<D, nthreads, warp_आकार>;
    } अन्यथा अगर constexpr (type_K == GGML_TYPE_Q8_0) {
        वापस vec_dot_fattn_vec_KQ_q8_0<D, nthreads, warp_आकार>;
    } अन्यथा {
        static_assert(type_K == -1, "bad type");
        वापस nullptr;
    }
}

ढाँचा <पूर्णांक type_V, typename T, पूर्णांक ne>
constexpr dequantize_V_t get_dequantize_V() {
    अगर constexpr (type_V == GGML_TYPE_F16) {
        वापस dequantize_V_f16<T, ne>;
    } अन्यथा अगर constexpr (type_V == GGML_TYPE_Q4_0) {
        वापस dequantize_V_q4_0<T, ne>;
    } अन्यथा अगर constexpr (type_V == GGML_TYPE_Q4_1) {
        वापस dequantize_V_q4_1<T, ne>;
    } अन्यथा अगर constexpr (type_V == GGML_TYPE_Q5_0) {
        वापस dequantize_V_q5_0<T, ne>;
    } अन्यथा अगर constexpr (type_V == GGML_TYPE_Q5_1) {
        वापस dequantize_V_q5_1<T, ne>;
    } अन्यथा अगर constexpr (type_V == GGML_TYPE_Q8_0) {
        वापस dequantize_V_q8_0<T, ne>;
    } अन्यथा {
        static_assert(type_V == -1, "bad type");
        वापस nullptr;
    }
}

ढाँचा <पूर्णांक ncols1, पूर्णांक warp_आकार>
अटल व्योम flash_attn_mask_to_KV_max(स्थिर sycl::half2 * __restrict__ mask,
                                      पूर्णांक * __restrict__ KV_max,
                                      स्थिर पूर्णांक ne30,
                                      स्थिर पूर्णांक s31,
                                      स्थिर पूर्णांक s33,
                                      पूर्णांक *     buf_iw) {
    स्वतः      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    स्थिर पूर्णांक ne31     = item_ct1.get_group_range(2);
    स्थिर पूर्णांक tid      = item_ct1.get_local_id(2);
    स्थिर पूर्णांक sequence = item_ct1.get_group(1);
    स्थिर पूर्णांक jt       = item_ct1.get_group(2);

    mask += sequence*s33 + jt*ncols1*s31;

    अगर (tid < warp_आकार) {
        buf_iw[tid] = 1;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);

    पूर्णांक KV_max_sj = (ne30 - 1) * FATTN_KQ_STRIDE;
    क्रम (; KV_max_sj >= 0; KV_max_sj -= FATTN_KQ_STRIDE) {
        पूर्णांक all_inf = 1;

#आशय unroll
        क्रम (पूर्णांक j = 0; j < ncols1; ++j) {
            स्थिर sycl::float2 tmp =
                mask[j * s31 + KV_max_sj / 2 + tid].ढाँचा convert<भग्न, sycl::rounding_mode::automatic>();
            all_inf = all_inf && पूर्णांक(sycl::isinf((भग्न) (tmp.x()))) && पूर्णांक(sycl::isinf((भग्न) (tmp.y())));
        }

        all_inf = warp_reduce_all<warp_आकार>(all_inf);
        अगर (tid % warp_आकार == 0) {
            buf_iw[tid / warp_आकार] = all_inf;
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);
        all_inf = buf_iw[tid % warp_आकार];
        item_ct1.barrier(sycl::access::fence_space::local_space);
        all_inf = warp_reduce_all<warp_आकार>(all_inf);

        अगर (!all_inf) {
            अवरोध;
        }
    }

    // If the break in the loop was not triggered, KV_max_sj is now -FATTN_KQ_STRIDE.
    // If the break was triggered it's the lower edge of the tile with the first non-masked values.
    // In either case, walk back the decrementation by FATTN_KQ_STRIDE.
    KV_max_sj += FATTN_KQ_STRIDE;

    अगर (item_ct1.get_local_id(2) != 0) {
        वापस;
    }

    KV_max[sequence*ne31 + jt] = KV_max_sj;
}

ढाँचा <पूर्णांक D, पूर्णांक ncols1, पूर्णांक ncols2>  // D == head size

अटल व्योम flash_attn_stream_k_fixup(भग्न * __restrict__ dst,
                                      स्थिर sycl::float2 * __restrict__ dst_fixup,
                                      स्थिर पूर्णांक ne01,
                                      स्थिर पूर्णांक ne02,
                                      स्थिर पूर्णांक ne03,
                                      स्थिर पूर्णांक ne11,
                                      स्थिर पूर्णांक ne12,
                                      स्थिर पूर्णांक nbatch_fa) {
    स्वतः          item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr पूर्णांक ncols    = ncols1 * ncols2;

    स्थिर पूर्णांक bidx0 = item_ct1.get_group(2);
    स्थिर पूर्णांक j     = item_ct1.get_group(1);
    स्थिर पूर्णांक c     = item_ct1.get_group(0);
    स्थिर पूर्णांक jc    = j*ncols2 + c;
    स्थिर पूर्णांक tid   = item_ct1.get_local_id(2);

    स्थिर भग्न * dst_fixup_तथ्य = ((स्थिर भग्न *) dst_fixup) + item_ct1.get_group_range(2) * (2 * 2 * ncols);

    स्थिर पूर्णांक gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.

    स्थिर पूर्णांक iter_k     = (ne11      + (nbatch_fa - 1)) / nbatch_fa;
    स्थिर पूर्णांक iter_j     = (ne01      + (ncols1    - 1)) / ncols1;
    स्थिर पूर्णांक iter_z_gqa = (gqa_ratio + (ncols2    - 1)) / ncols2;

    स्थिर पूर्णांक kbc0 = int64_t(bidx0 + 0) * (iter_k * iter_j * iter_z_gqa * ne12 * ne03) / item_ct1.get_group_range(2);
    स्थिर पूर्णांक kbc0_stop =
        int64_t(bidx0 + 1) * (iter_k * iter_j * iter_z_gqa * ne12 * ne03) / item_ct1.get_group_range(2);

    स्थिर bool did_not_have_any_तथ्य   = kbc0 == kbc0_stop;
    स्थिर bool wrote_beginning_of_tile = kbc0 % iter_k == 0;
    स्थिर bool did_not_write_last      = kbc0/iter_k == kbc0_stop/iter_k && kbc0_stop % iter_k != 0;
    अगर (did_not_have_any_तथ्य || wrote_beginning_of_tile || did_not_write_last) {
        वापस;
    }

    // z_KV == K/V head index, zt_gqa = Q head start index per K/V head, jt = token position start index
    स्थिर पूर्णांक sequence =  kbc0 /(iter_k*iter_j*iter_z_gqa*ne12);
    स्थिर पूर्णांक z_KV     = (kbc0 - iter_k*iter_j*iter_z_gqa*ne12 * sequence)/(iter_k*iter_j*iter_z_gqa);
    स्थिर पूर्णांक zt_gqa   = (kbc0 - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV)/(iter_k*iter_j);
    स्थिर पूर्णांक jt       = (kbc0 - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV - iter_k*iter_j * zt_gqa) / iter_k;

    स्थिर पूर्णांक zt_Q = z_KV*gqa_ratio + zt_gqa*ncols2; // Global Q head start index.

    अगर (jt*ncols1 + j >= ne01 || zt_gqa*ncols2 + c >= gqa_ratio) {
        वापस;
    }

    dst += sequence*ne02*ne01*D + jt*ne02*(ncols1*D) + zt_Q*D + (j*ne02 + c)*D + tid;

    // Load the partial result that needs a fixup:
    भग्न dst_val = 0.0f;
    भग्न max_val = 0.0f;
    भग्न rowsum  = 0.0f;
    {
        dst_val = *dst;

        स्थिर sycl::float2 tmp = dst_fixup[bidx0 * ncols + jc];
        max_val                = tmp.x();
        rowsum                 = tmp.y();
    }

    // Iterate over previous blocks and compute the combined results.
    // All SYCL blocks that get here must have a previous block that needs a fixup.
    पूर्णांक bidx = bidx0 - 1;
    पूर्णांक kbc_stop = kbc0;
    जबतक(true) {
        स्थिर पूर्णांक kbc = int64_t(bidx) * (iter_k * iter_j * iter_z_gqa * ne12 * ne03) / item_ct1.get_group_range(2);
        अगर (kbc == kbc_stop) { // Did not have any data.
            bidx--;
            kbc_stop = kbc;
            जारी;
        }

        स्थिर भग्न dst_add = dst_fixup_तथ्य[bidx*ncols*D + jc*D + tid];

        स्थिर sycl::float2 tmp = dst_fixup[(item_ct1.get_group_range(2) + bidx) * ncols + jc];

        // Scale the current and new value accumulators depending on the max. values.
        स्थिर भग्न max_val_new = sycl::fmax(max_val, tmp.x());

        स्थिर भग्न diff_val = max_val - max_val_new;
        स्थिर भग्न diff_add = tmp.x() - max_val_new;

        स्थिर भग्न scale_val = diff_val >= SOFTMAX_FTZ_THRESHOLD ? sycl::native::exp(diff_val) : 0.0f;
        स्थिर भग्न scale_add = diff_add >= SOFTMAX_FTZ_THRESHOLD ? sycl::native::exp(diff_add) : 0.0f;

        dst_val = scale_val*dst_val + scale_add*dst_add;
        rowsum  = scale_val * rowsum + scale_add * tmp.y();

        max_val = max_val_new;

        // If this block started in a previous tile we are done and don't need to combine additional partial results.
        अगर (kbc % iter_k == 0 || kbc/iter_k < kbc0/iter_k) {
            अवरोध;
        }
        bidx--;
        kbc_stop = kbc;
    }

    // Write back final result:
    *dst = dst_val / rowsum;
}

ढाँचा <पूर्णांक D>  // D == head size

अटल व्योम flash_attn_combine_results(स्थिर भग्न * __restrict__ VKQ_parts,
                                       स्थिर sycl::float2 * __restrict__ VKQ_meta,
                                       भग्न * __restrict__ dst,
                                       स्थिर पूर्णांक parallel_blocks,
                                       uint8_t * dpct_local) {
    // Dimension 0: threadIdx.x
    // Dimension 1: blockIdx.x
    // Dimension 2: blockIdx.y
    // Dimension 3: blockIdx.z
    // Memory layout is permuted with [0, 2, 1, 3]

    स्वतः      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    स्थिर पूर्णांक ne01     = item_ct1.get_group_range(2);
    स्थिर पूर्णांक ne02     = item_ct1.get_group_range(1);

    स्थिर पूर्णांक col      = item_ct1.get_group(2);
    स्थिर पूर्णांक शीर्ष     = item_ct1.get_group(1);
    स्थिर पूर्णांक sequence = item_ct1.get_group(0);

    स्थिर पूर्णांक j_dst_unrolled = (sequence*ne01 + col)*ne02 + शीर्ष;

    VKQ_parts += j_dst_unrolled * parallel_blocks*D;
    VKQ_meta  += j_dst_unrolled * parallel_blocks;
    dst       += j_dst_unrolled *                 D;

    स्थिर पूर्णांक tid = item_ct1.get_local_id(2);
    __builtin_assume(tid < D);

    स्वतः meta = (sycl::float2 *) dpct_local;
    क्रम (पूर्णांक i = tid; i < 2*parallel_blocks; i += D) {
        ((भग्न *) meta)[i] = ((स्थिर भग्न *)VKQ_meta) [i];
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    भग्न kqmax = meta[0].x();
    क्रम (पूर्णांक l = 1; l < parallel_blocks; ++l) {
        kqmax = sycl::max(kqmax, meta[l].x());
    }

    भग्न VKQ_numerator   = 0.0f;
    भग्न VKQ_denominator = 0.0f;
    क्रम (पूर्णांक l = 0; l < parallel_blocks; ++l) {
        स्थिर भग्न KQ_max_scale = sycl::native::exp(meta[l].x() - kqmax);

        VKQ_numerator   += KQ_max_scale * VKQ_parts[l*D + tid];
        VKQ_denominator += KQ_max_scale * meta[l].y();
    }

    dst[tid] = VKQ_numerator / VKQ_denominator;
}

// Carries the sub-group size (and grf_size_automatic when use_large_grf) as
// kernel properties via get(properties_tag), in place of the
// reqd_sub_group_size attribute and the deprecated parallel_for overload.
ढाँचा <typename KernelFunc, uint32_t आकार_3, bool UseGrf>
काष्ठा props_kernel {
    KernelFunc kernel;

    व्योम चालक()(sycl::nd_item<3> item) स्थिर {
        kernel(item);
    }

    स्वतः लो(sycl::ext::oneapi::experimental::properties_tag) स्थिर {
        अगर constexpr (UseGrf) {
            वापस sycl::ext::oneapi::experimental::properties{
                sycl::ext::intel::experimental::grf_आकार_automatic,
                sycl::ext::oneapi::experimental::sub_group_आकार<आकार_3> };
        } अन्यथा {
            वापस sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::sub_group_आकार<आकार_3> };
        }
    }
};

ढाँचा <fattn_kernel_t fattn_kernel, पूर्णांक warp_आकार, bool use_large_grf = false>
अटल व्योम lauch_kernel(
    dpct::dim3 group_range,
    dpct::dim3 local_range,
    queue_ptr q,
    अचिन्हित पूर्णांक local_mem_आकार_2,
    स्थिर अक्षर* __restrict__ Q,
    स्थिर अक्षर* __restrict__ K,
    स्थिर अक्षर* __restrict__ V,
    स्थिर अक्षर* __restrict__ mask,
    स्थिर अक्षर* __restrict__ sinks,
    स्थिर पूर्णांक* __restrict__ KV_max,
    भग्न* __restrict__ dst,
    sycl::float2* __restrict__ dst_meta,
    स्थिर भग्न scale,
    स्थिर भग्न max_bias,
    स्थिर भग्न m0,
    स्थिर भग्न m1,
    स्थिर uint32_t n_शीर्ष_log2,
    स्थिर भग्न logit_softcap,
    स्थिर int32_t ne00,
    स्थिर sycl::uint3 ne01,
    स्थिर int32_t ne02,
    स्थिर int32_t ne03,
    स्थिर int32_t nb01,
    स्थिर int32_t nb02,
    स्थिर int32_t nb03,
    स्थिर int32_t ne10,
    स्थिर int32_t ne11,
    स्थिर int32_t ne12,
    स्थिर int32_t ne13,
    स्थिर int32_t nb11,
    स्थिर int32_t nb12,
    स्थिर int64_t nb13,
    स्थिर int32_t nb21,
    स्थिर int32_t nb22,
    स्थिर int64_t nb23,
    स्थिर int32_t ne31,
    स्थिर int32_t ne32,
    स्थिर int32_t ne33,
    स्थिर int32_t nb31,
    स्थिर int32_t nb32,
    स्थिर int64_t nb33) {
    GGML_UNUSED(local_mem_आकार_2);

    स्थिर स्वतः rng = sycl::nd_range<3>(
        static_cast<sycl::range<3>>(group_range * local_range),
        static_cast<sycl::range<3>>(local_range));

    स्थिर स्वतः kernel = [=](sycl::nd_item<3> item_ct1) {
        GGML_UNUSED(item_ct1);
        fattn_kernel(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
                     max_bias, m0, m1, n_शीर्ष_log2, logit_softcap, ne00,
                     ne01, ne02, ne03, nb01, nb02, nb03, ne10, ne11,
                     ne12, ne13, nb11, nb12, nb13, nb21, nb22, nb23,
                     ne31, ne32, ne33, nb31, nb32, nb33);
    };

    q->submit([&](sycl::handler &cgh) {
        // grf_size_automatic lets the compiler use the large register file when the kernel needs it. grf_size<256> would say the same but is undefined on devices that do not have it.
        cgh.parallel_for(rng, props_kernel<decltype(kernel), warp_आकार, use_large_grf>{kernel});
    });
}

// use_large_grf lets the compiler choose the large register file. Opt in where spilling limits the kernel, not thread occupancy.
ढाँचा <पूर्णांक DV, पूर्णांक ncols1, पूर्णांक ncols2, fattn_kernel_t fattn_kernel, पूर्णांक warp_आकार, bool use_large_grf = false>
व्योम launch_fattn(
    ggml_backend_sycl_context & ctx, ggml_tensor * dst, स्थिर पूर्णांक nwarps, स्थिर माप_प्रकार nbytes_shared,
    स्थिर पूर्णांक nbatch_fa, स्थिर bool need_f16_K, स्थिर bool need_f16_V, स्थिर bool stream_k) {

    constexpr पूर्णांक ncols = ncols1 * ncols2;

    स्थिर ggml_tensor * Q = dst->src[0];
    स्थिर ggml_tensor * K = dst->src[1];
    स्थिर ggml_tensor * V = dst->src[2];

    स्थिर bool V_is_K_view = V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));

    स्थिर ggml_tensor * mask  = dst->src[3];
    स्थिर ggml_tensor * sinks = dst->src[4];

    ggml_tensor * KQV = dst;

    GGML_ASSERT(Q->type == GGML_TYPE_F32);
    GGML_ASSERT(KQV->type == GGML_TYPE_F32);

    GGML_ASSERT(Q->nb[0] == ggml_element_आकार(Q));
    GGML_ASSERT(K->nb[0] == ggml_element_आकार(K));
    GGML_ASSERT(V->nb[0] == ggml_element_आकार(V));

    GGML_ASSERT(!mask || mask->type == GGML_TYPE_F16);

    ggml_sycl_pool & pool = ctx.pool();
    dpct::queue_ptr  main_stream = ctx.stream();
    स्थिर पूर्णांक id  = ggml_sycl_get_device();
    स्थिर पूर्णांक nsm = ggml_sycl_info().devices[id].nsm;

    स्थिर ggml_sycl_fattn_extra  extra = ggml_sycl_fattn_get_extra(dst);
    ggml_sycl_pool_alloc<पूर्णांक>    KV_max(pool);
    ggml_sycl_pool_alloc<भग्न>  dst_tmp(pool);
    ggml_sycl_pool_alloc<sycl::float2> dst_tmp_meta(pool);

    स्थिर अक्षर * K_तथ्य = (स्थिर अक्षर *) K->तथ्य;
    माप_प्रकार nb11 = K->nb[1];
    माप_प्रकार nb12 = K->nb[2];
    माप_प्रकार nb13 = K->nb[3];

    स्थिर अक्षर * V_तथ्य = (स्थिर अक्षर *) V->तथ्य;
    माप_प्रकार nb21 = V->nb[1];
    माप_प्रकार nb22 = V->nb[2];
    माप_प्रकार nb23 = V->nb[3];

    अगर (need_f16_K && K->type != GGML_TYPE_F16) {
        स्थिर माप_प्रकार bs = ggml_blck_आकार(K->type);
        स्थिर माप_प्रकार ts = ggml_type_आकार(K->type);

        GGML_ASSERT(extra.K_बफर_ptr);
        sycl::half * K_f16_ptr = (sycl::half *) extra.K_बफर_ptr;
        अगर (ggml_is_contiguously_allocated(K)) {
            to_fp16_sycl_t to_fp16 = ggml_get_to_fp16_sycl(K->type, dst);
            to_fp16(K_तथ्य, K_f16_ptr, ggml_nelements(K), main_stream);

            nb11 = nb11 * bs * माप(sycl::half) / ts;
            nb12 = nb12 * bs * माप(sycl::half) / ts;
            nb13 = nb13 * bs * माप(sycl::half) / ts;
        } अन्यथा {
            GGML_ASSERT(K->nb[0] == ts);
            to_fp16_nc_sycl_t to_fp16 = ggml_get_to_fp16_nc_sycl(K->type);
            स्थिर int64_t s01 = nb11 / ts;
            स्थिर int64_t s02 = nb12 / ts;
            स्थिर int64_t s03 = nb13 / ts;
            to_fp16(K_तथ्य, K_f16_ptr, K->ne[0], K->ne[1], K->ne[2], K->ne[3], s01, s02, s03, main_stream);

            nb11 = K->ne[0] * माप(sycl::half);
            nb12 = K->ne[1] * nb11;
            nb13 = K->ne[2] * nb12;
        }
        K_तथ्य = (अक्षर *) K_f16_ptr;
    }

    अगर (need_f16_V && V->type != GGML_TYPE_F16) {
        अगर (V_is_K_view) {
            V_तथ्य = K_तथ्य;
            nb21   = nb11;
            nb22   = nb12;
            nb23   = nb13;
        } अन्यथा {
            स्थिर माप_प्रकार bs = ggml_blck_आकार(V->type);
            स्थिर माप_प्रकार ts = ggml_type_आकार(V->type);

            GGML_ASSERT(extra.V_बफर_ptr);
            sycl::half * V_f16_ptr = (sycl::half *) extra.V_बफर_ptr;
            अगर (ggml_is_contiguously_allocated(V)) {
                to_fp16_sycl_t to_fp16 = ggml_get_to_fp16_sycl(V->type, dst);
                to_fp16(V_तथ्य, V_f16_ptr, ggml_nelements(V), main_stream);
                V_तथ्य = (अक्षर *) V_f16_ptr;

                nb21 = nb21 * bs * माप(sycl::half) / ts;
                nb22 = nb22 * bs * माप(sycl::half) / ts;
                nb23 = nb23 * bs * माप(sycl::half) / ts;
            } अन्यथा {
                GGML_ASSERT(V->nb[0] == ts);
                to_fp16_nc_sycl_t to_fp16 = ggml_get_to_fp16_nc_sycl(V->type);
                स्थिर int64_t s01 = nb21 / ts;
                स्थिर int64_t s02 = nb22 / ts;
                स्थिर int64_t s03 = nb23 / ts;
                to_fp16(V_तथ्य, V_f16_ptr, V->ne[0], V->ne[1], V->ne[2], V->ne[3], s01, s02, s03, main_stream);

                nb21 = V->ne[0] * माप(sycl::half);
                nb22 = V->ne[1] * nb21;
                nb23 = V->ne[2] * nb22;
            }
            V_तथ्य = (अक्षर *) V_f16_ptr;
        }
    }

    स्थिर पूर्णांक ntiles_x     = ((Q->ne[1] + ncols1 - 1) / ncols1);
    स्थिर पूर्णांक gqa_ratio    = Q->ne[2] / K->ne[2];
    स्थिर पूर्णांक ntiles_z_gqa = ((gqa_ratio + ncols2 - 1) / ncols2);
    स्थिर पूर्णांक ntiles_योग = ntiles_x * ntiles_z_gqa * K->ne[2] * Q->ne[3];

    // Optional optimization where the mask is scanned to determine whether part of the calculation can be skipped.
    // Only worth the overhead if there is at lease one FATTN_KQ_STRIDE x FATTN_KQ_STRIDE square to be skipped or
    //     multiple sequences of possibly different lengths.
    अगर (mask && K->ne[1] % FATTN_KQ_STRIDE == 0 && (Q->ne[1] >= 1024 || Q->ne[3] > 1)) {
        स्थिर पूर्णांक s31 = mask->nb[1] / माप(sycl::half2);
        स्थिर पूर्णांक s33 = mask->nb[3] / माप(sycl::half2);

        स्थिर dpct::dim3 blocks_अंक_KV_max(ntiles_x, Q->ne[3], 1);
        स्थिर dpct::dim3 block_dim_KV_max(FATTN_KQ_STRIDE / 2, 1, 1);

        स्थिर पूर्णांक ne_KV_max = blocks_अंक_KV_max.x*blocks_अंक_KV_max.y;
        स्थिर पूर्णांक iter_k = K->ne[1] / FATTN_KQ_STRIDE;

        KV_max.alloc(ne_KV_max);
        {
            dpct::has_capability_or_fail(main_stream->get_device(), { sycl::aspect::fp16 });

            main_stream->submit([&](sycl::handler & cgh) {
                sycl::local_accessor<पूर्णांक, 1> buf_iw_acc_ct1(sycl::range<1>(warp_आकार), cgh);

                स्वतः mask_तथ्य_ct0  = (स्थिर sycl::half2 *) mask->तथ्य;
                स्वतः KV_max_ptr_ct1 = KV_max.ptr;

                cgh.parallel_for(sycl::nd_range<3>(blocks_अंक_KV_max * block_dim_KV_max, block_dim_KV_max),
                                 [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_आकार(warp_आकार)]] {
                                     GGML_UNUSED(item_ct1);
                                     flash_attn_mask_to_KV_max<ncols1, warp_आकार>(
                                         mask_तथ्य_ct0, KV_max_ptr_ct1, iter_k, s31, s33,
                                         buf_iw_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().लो());
                                 });
            });
        }
        SYCL_CHECK(0);
    }

    स्थिर dpct::dim3 block_dim(warp_आकार, nwarps, 1);

    // Max. number of active blocks limited by occupancy.
    पूर्णांक max_blocks_per_sm = ggml_sycl_info().devices[id].max_wg_per_cu;
    पूर्णांक parallel_blocks = max_blocks_per_sm;
    dpct::dim3 blocks_अंक;
    अगर (stream_k) {
        // For short contexts it can be faster to have the SMs work on whole tiles because this lets us skip the fixup.
        स्थिर पूर्णांक max_blocks = max_blocks_per_sm*nsm;
        स्थिर पूर्णांक nblocks_stream_k = max_blocks;
        स्थिर bool use_stream_k = true;

        blocks_अंक.x = use_stream_k ? nblocks_stream_k : ntiles_योग;
        blocks_अंक.y = 1;
        blocks_अंक.z = 1;

        अगर (ntiles_योग % blocks_अंक.x != 0) { // Fixup is only needed if the SMs work on fractional tiles.
            dst_tmp_meta.alloc((माप_प्रकार(blocks_अंक.x) * ncols * (2 + DV/2)));
        }
    } अन्यथा {
        स्थिर पूर्णांक ntiles_KQ = (K->ne[1] + nbatch_fa - 1) / nbatch_fa; // Max. number of parallel blocks limited by tensor size.

        // parallel_blocks must not be larger than what the tensor size allows:
        parallel_blocks = std::min(parallel_blocks, ntiles_KQ);
        // todo fix the hard code change
        // parallel_blocks = ntiles_KQ;

        // If ntiles_total % blocks_per_wave != 0 then some efficiency is lost due to tail effects.
        // Test whether parallel_blocks can be set to a higher value for better efficiency.
        स्थिर पूर्णांक blocks_per_wave = nsm * max_blocks_per_sm;
        पूर्णांक nwaves_best = 0;
        पूर्णांक efficiency_percent_best = 0;
        क्रम (पूर्णांक parallel_blocks_test = parallel_blocks; parallel_blocks_test <= ntiles_KQ; ++parallel_blocks_test) {
            स्थिर पूर्णांक nblocks_योग = ntiles_योग * parallel_blocks_test;
            स्थिर पूर्णांक nwaves = (nblocks_योग + blocks_per_wave - 1) / blocks_per_wave;
            स्थिर पूर्णांक efficiency_percent = 100 * nblocks_योग / (nwaves*blocks_per_wave);

            // Stop trying configurations with more waves if we already have good efficiency to avoid excessive overhead.
            अगर (efficiency_percent_best >= 95 && nwaves > nwaves_best) {
                अवरोध;
            }

            अगर (efficiency_percent > efficiency_percent_best) {
                nwaves_best = nwaves;
                efficiency_percent_best = efficiency_percent;
                parallel_blocks = parallel_blocks_test;
            }
        }

        blocks_अंक.x = ntiles_x;
        blocks_अंक.y = parallel_blocks;
        blocks_अंक.z = ntiles_z_gqa*K->ne[2]*Q->ne[3];

        अगर (parallel_blocks > 1) {
            dst_tmp.alloc(parallel_blocks*ggml_nelements(KQV));
            dst_tmp_meta.alloc(parallel_blocks*ggml_nrows(KQV));
        }
    }

    भग्न scale         = 1.0f;
    भग्न max_bias      = 0.0f;
    भग्न logit_softcap = 0.0f;

    स_नकल(&scale,         (स्थिर भग्न *) KQV->op_params + 0, माप(भग्न));
    स_नकल(&max_bias,      (स्थिर भग्न *) KQV->op_params + 1, माप(भग्न));
    स_नकल(&logit_softcap, (स्थिर भग्न *) KQV->op_params + 2, माप(भग्न));

    अगर (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    स्थिर uint32_t n_शीर्ष      = Q->ne[2];
    स्थिर uint32_t n_शीर्ष_log2 = 1u << uint32_t(floorf(log2f(भग्न(n_शीर्ष))));

    स्थिर भग्न m0 = powf(2.0f, -(max_bias       ) / n_शीर्ष_log2);
    स्थिर भग्न m1 = powf(2.0f, -(max_bias / 2.0f) / n_शीर्ष_log2);

    // TODO other tensor dimensions after removal of WMMA kernel:
    स्थिर sycl::uint3 ne01 = init_fastdiv_मूल्यावली(Q->ne[1]);

    GGML_ASSERT(block_dim.x % warp_आकार == 0);

    lauch_kernel<fattn_kernel, warp_आकार, use_large_grf>(
        blocks_अंक, block_dim, main_stream, (अचिन्हित पूर्णांक) nbytes_shared, (स्थिर अक्षर *) Q->तथ्य, K_तथ्य, V_तथ्य,
        mask ? ((स्थिर अक्षर *) mask->तथ्य) : nullptr, sinks ? ((स्थिर अक्षर *) sinks->तथ्य) : nullptr, KV_max.ptr,
        !stream_k && parallel_blocks > 1 ? dst_tmp.ptr : (भग्न *) KQV->तथ्य, (sycl::float2 *)dst_tmp_meta.ptr, scale, max_bias, m0, m1,
        n_शीर्ष_log2, logit_softcap, Q->ne[0], ne01, Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3], K->ne[0],
        K->ne[1], K->ne[2], K->ne[3], nb11, nb12, nb13, nb21, nb22, nb23, mask ? mask->ne[1] : 0,
        mask ? mask->ne[2] : 0, mask ? mask->ne[3] : 0, mask ? mask->nb[1] : 0, mask ? mask->nb[2] : 0,
        mask ? mask->nb[3] : 0);
    SYCL_CHECK(0);

    अगर (stream_k) {
        अगर (ntiles_योग % blocks_अंक.x != 0) { // Fixup is only needed if the SMs work on fractional tiles.
            स्थिर dpct::dim3 block_dim_combine(DV, 1, 1);
            स्थिर dpct::dim3 blocks_अंक_combine = { blocks_अंक.x, ncols1, ncols2 };

            main_stream->submit([&](sycl::handler & cgh) {
                स्वतः KQV_तथ्य_ct0         = (भग्न *) KQV->तथ्य;
                स्वतः dst_tmp_meta_ptr_ct1 = dst_tmp_meta.ptr;
                स्वतः Q_ne_ct2             = Q->ne[1];
                स्वतः Q_ne_ct3             = Q->ne[2];
                स्वतः Q_ne_ct4             = Q->ne[3];
                स्वतः K_ne_ct5             = K->ne[1];
                स्वतः K_ne_ct6             = K->ne[2];

                cgh.parallel_for(sycl::nd_range<3>(blocks_अंक_combine * block_dim_combine, block_dim_combine),
                                 [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_आकार(warp_आकार)]] {
                                     GGML_UNUSED(item_ct1);
                                     flash_attn_stream_k_fixup<DV, ncols1, ncols2>(KQV_तथ्य_ct0, dst_tmp_meta_ptr_ct1,
                                                                                   Q_ne_ct2, Q_ne_ct3, Q_ne_ct4,
                                                                                   K_ne_ct5, K_ne_ct6, nbatch_fa);
                                 });
            });
        }
    } अन्यथा अगर (parallel_blocks > 1) {
        स्थिर dpct::dim3 block_dim_combine(DV, 1, 1);
        स्थिर dpct::dim3 blocks_अंक_combine(Q->ne[1], Q->ne[2], Q->ne[3]);
        स्थिर माप_प्रकार     nbytes_shared_combine = parallel_blocks * माप(sycl::float2);
        main_stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(sycl::range<1>(nbytes_shared_combine), cgh);

            स्वतः dst_tmp_ptr_ct0      = dst_tmp.ptr;
            स्वतः dst_tmp_meta_ptr_ct1 = dst_tmp_meta.ptr;
            स्वतः KQV_तथ्य_ct2         = (भग्न *) KQV->तथ्य;

            cgh.parallel_for(sycl::nd_range<3>(blocks_अंक_combine * block_dim_combine, block_dim_combine),
                             [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_आकार(warp_आकार)]] {
                                 GGML_UNUSED(item_ct1);
                                 flash_attn_combine_results<DV>(
                                     dst_tmp_ptr_ct0, dst_tmp_meta_ptr_ct1, KQV_तथ्य_ct2, parallel_blocks,
                                     dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().लो());
                             });
        });
    }
    SYCL_CHECK(0);
}
