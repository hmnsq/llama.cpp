#निर्देशः once

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


#परिभाषय FATTN_KQ_STRIDE       256
#परिभाषय HALF_MAX_HALF         sycl::half(65504.0f/2) // Use neg. of this instead of -INFINITY to initialize KQ max vals to avoid NaN upon subtraction.
#परिभाषय SOFTMAX_FTZ_देहली -20.0f                   // Softmax exp. of values smaller than this are flushed to zero to avoid NaNs.
#परिभाषय FATTN_KQ_MAX_विस्थापनम् (3.0f*0.6931f)

प्रकारनिर्देश शून्य (*fattn_kernel_t)(
    ध्रुव वर्ण* Q,
    ध्रुव वर्ण* K,
    ध्रुव वर्ण* V,
    ध्रुव वर्ण* आवरणम्,
    ध्रुव वर्ण* sinks,
    ध्रुव पूर्णाङ्क* KV_max,
    भिन्न* dst,
    sycl::float2* dst_meta,
    ध्रुव भिन्न scale,
    ध्रुव भिन्न max_bias,
    ध्रुव भिन्न m0,
    ध्रुव भिन्न m1,
    ध्रुव uint32_t n_शीर्ष_log2,
    ध्रुव भिन्न logit_softcap,
    ध्रुव int32_t ne00,
    ध्रुव sycl::uint3 ne01,
    ध्रुव int32_t ne02,
    ध्रुव int32_t ne03,
    ध्रुव int32_t nb01,
    ध्रुव int32_t nb02,
    ध्रुव int32_t nb03,
    ध्रुव int32_t ne10,
    ध्रुव int32_t ne11,
    ध्रुव int32_t ne12,
    ध्रुव int32_t ne13,
    ध्रुव int32_t nb11,
    ध्रुव int32_t nb12,
    ध्रुव int64_t nb13,
    ध्रुव int32_t nb21,
    ध्रुव int32_t nb22,
    ध्रुव int64_t nb23,
    ध्रुव int32_t ne31,
    ध्रुव int32_t ne32,
    ध्रुव int32_t ne33,
    ध्रुव int32_t nb31,
    ध्रुव int32_t nb32,
    ध्रुव int64_t nb33);

प्रकारनिर्देश भिन्न (*vec_dot_KQ_t)(
    ध्रुव वर्ण * __restrict__ K_c, ध्रुव शून्य * __restrict__ Q_v, ध्रुव पूर्णाङ्क * __restrict__ Q_q8 , ध्रुव शून्य * __restrict__ Q_ds);

प्रतिरूपम् <पूर्णाङ्क D, पूर्णाङ्क nthreads>
स्थिर __dpct_inline__ भिन्न vec_dot_fattn_vec_KQ_f16(ध्रुव वर्ण * __restrict__ K_c,
                                                      ध्रुव शून्य * __restrict__ Q_v,
                                                      ध्रुव पूर्णाङ्क * __restrict__ Q_q8,
                                                      ध्रुव शून्य * __restrict__ Q_ds_v) {
    ध्रुव sycl::half2 * K_h2 = (ध्रुव sycl::half2 *) K_c;
    GGML_अप्रयुक्तम्(Q_q8);
    GGML_अप्रयुक्तम्(Q_ds_v);

    constexpr पूर्णाङ्क cpy_nb = ggml_sycl_get_max_cpy_अष्टकानि();
    constexpr पूर्णाङ्क cpy_ne = cpy_nb / 4;

    भिन्न योग_2 = 0.0f;

#निर्देशः unroll
    प्रति (पूर्णाङ्क k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads*cpy_ne) {
        sycl::half2 tmp[cpy_ne];
        ggml_sycl_memcpy_1<परिमाण(tmp)>(
            tmp,
            K_h2 + k_KQ_0 + (sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2) % nthreads) * cpy_ne);
#निर्देशः unroll
        प्रति (पूर्णाङ्क k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
#यदि_परिभाषितम् GGML_SYCL_F16
            ggml_sycl_mad(योग_2,                tmp[k_KQ_1] , ((ध्रुव sycl::half2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#अन्यथा
            ggml_sycl_mad(योग_2, __half22float2(tmp[k_KQ_1]), ((ध्रुव sycl::float2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#यदि_अन्तः // GGML_SYCL_F16
        }
    }

    प्रत्यागम योग_2;
}

प्रतिरूपम् <पूर्णाङ्क D, पूर्णाङ्क nthreads, पूर्णाङ्क warp_आकार>
स्थिर __dpct_inline__ भिन्न vec_dot_fattn_vec_KQ_q4_0(ध्रुव वर्ण * __restrict__ K_c,
                                                       ध्रुव शून्य * __restrict__ Q_v,
                                                       ध्रुव पूर्णाङ्क * __restrict__ Q_q8,
                                                       ध्रुव शून्य * __restrict__ Q_ds_v) {
    स्वचालित item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    ध्रुव खण्डम्_q4_0 * K_q4_0   = (ध्रुव खण्डम्_q4_0 *) K_c;
    GGML_अप्रयुक्तम्(Q_v);

    भिन्न योग_2 = 0.0f;

#निर्देशः unroll
    प्रति (पूर्णाङ्क k_KQ_0 = 0; k_KQ_0 < पूर्णाङ्क(D/परिमाण(पूर्णाङ्क)); k_KQ_0 += nthreads) {
        ध्रुव पूर्णाङ्क k_KQ =
            k_KQ_0 + (nthreads == warp_आकार ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads);

        ध्रुव पूर्णाङ्क ib    = k_KQ /  QI8_1;
        ध्रुव पूर्णाङ्क iqs4  = k_KQ %  QI4_0;
        ध्रुव पूर्णाङ्क सरणम् = k_KQ & (QI8_1/2);

        पूर्णाङ्क v;
        ggml_sycl_memcpy_1<परिमाण(पूर्णाङ्क), 2>(&v, K_q4_0[ib].qs + परिमाण(पूर्णाङ्क)*iqs4);
        v = (v >> सरणम्) & 0x0F0F0F0F;
        ध्रुव पूर्णाङ्क u = Q_q8[k_KQ_0/nthreads];

        ध्रुव पूर्णाङ्क sumi = ggml_sycl_dp4a(v, u, 0);

        ध्रुव sycl::float2 Q_ds = ((ध्रुव sycl::float2 *) Q_ds_v)[k_KQ_0 / nthreads];
        योग_2 += __half2float(K_q4_0[ib].d) * (sumi*Q_ds.x() - (8/QI8_1)*Q_ds.y());
    }

    प्रत्यागम योग_2;
}

प्रतिरूपम् <पूर्णाङ्क D, पूर्णाङ्क nthreads , पूर्णाङ्क warp_आकार>
स्थिर __dpct_inline__ भिन्न vec_dot_fattn_vec_KQ_q4_1(ध्रुव वर्ण * __restrict__ K_c,
                                                       ध्रुव शून्य * __restrict__ Q_v,
                                                       ध्रुव पूर्णाङ्क * __restrict__ Q_q8,
                                                       ध्रुव शून्य * __restrict__ Q_ds_v) {
    स्वचालित               item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ध्रुव खण्डम्_q4_1 * K_q4_1   = (ध्रुव खण्डम्_q4_1 *) K_c;
    GGML_अप्रयुक्तम्(Q_v);

    भिन्न योग_2 = 0.0f;

#निर्देशः unroll
    प्रति (पूर्णाङ्क k_KQ_0 = 0; k_KQ_0 < पूर्णाङ्क(D/परिमाण(पूर्णाङ्क)); k_KQ_0 += nthreads) {
        ध्रुव पूर्णाङ्क k_KQ =
            k_KQ_0 + (nthreads == warp_आकार ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads);

        ध्रुव पूर्णाङ्क ib    = k_KQ /  QI8_1;
        ध्रुव पूर्णाङ्क iqs4  = k_KQ %  QI4_1;
        ध्रुव पूर्णाङ्क सरणम् = k_KQ & (QI8_1/2);

        पूर्णाङ्क v;
        ggml_sycl_memcpy_1<परिमाण(पूर्णाङ्क)>(&v, K_q4_1[ib].qs + परिमाण(पूर्णाङ्क)*iqs4);
        v = (v >> सरणम्) & 0x0F0F0F0F;
        ध्रुव पूर्णाङ्क u = Q_q8[k_KQ_0/nthreads];

        ध्रुव पूर्णाङ्क sumi = ggml_sycl_dp4a(v, u, 0);

        ध्रुव sycl::float2 K_dm = (K_q4_1[ib].dm).प्रतिरूपम् convert<भिन्न, sycl::rounding_बहुलकम्::automatic>();
        ध्रुव sycl::float2 Q_ds = ((ध्रुव sycl::float2 *) Q_ds_v)[k_KQ_0 / nthreads];

        योग_2 += K_dm.x()*Q_ds.x()*sumi + K_dm.y()*Q_ds.y()/QI8_1;
    }

    प्रत्यागम योग_2;
}

प्रतिरूपम् <पूर्णाङ्क D, पूर्णाङ्क nthreads, पूर्णाङ्क warp_आकार>
स्थिर __dpct_inline__ भिन्न vec_dot_fattn_vec_KQ_q5_0(ध्रुव वर्ण * __restrict__ K_c,
                                                       ध्रुव शून्य * __restrict__ Q_v,
                                                       ध्रुव पूर्णाङ्क * __restrict__ Q_q8,
                                                       ध्रुव शून्य * __restrict__ Q_ds_v) {
    स्वचालित               item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ध्रुव खण्डम्_q5_0 * K_q5_0   = (ध्रुव खण्डम्_q5_0 *) K_c;
    GGML_अप्रयुक्तम्(Q_v);

    भिन्न योग_2 = 0.0f;

#निर्देशः unroll
    प्रति (पूर्णाङ्क k_KQ_0 = 0; k_KQ_0 < पूर्णाङ्क(D/परिमाण(पूर्णाङ्क)); k_KQ_0 += nthreads) {
        ध्रुव पूर्णाङ्क k_KQ =
            k_KQ_0 + (nthreads == warp_आकार ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads);

        ध्रुव पूर्णाङ्क ib    = k_KQ /  QI8_1;
        ध्रुव पूर्णाङ्क iqs4  = k_KQ %  QI5_0;
        ध्रुव पूर्णाङ्क iqs8  = k_KQ %  QI8_1;
        ध्रुव पूर्णाङ्क सरणम् = k_KQ & (QI8_1/2);

        पूर्णाङ्क v;
        ggml_sycl_memcpy_1<परिमाण(पूर्णाङ्क), 2>(&v, K_q5_0[ib].qs + परिमाण(पूर्णाङ्क)*iqs4);
        v = (v >> सरणम्) & 0x0F0F0F0F;

        {
            पूर्णाङ्क vh;
            ggml_sycl_memcpy_1<परिमाण(पूर्णाङ्क), 2>(&vh, K_q5_0[ib].qh);
            vh >>= iqs8 * QI5_0;

            v |= (vh <<  4) & 0x00000010; // 0 ->  4
            v |= (vh << 11) & 0x00001000; // 1 -> 12
            v |= (vh << 18) & 0x00100000; // 2 -> 20
            v |= (vh << 25) & 0x10000000; // 3 -> 28
        }

        ध्रुव पूर्णाङ्क u = Q_q8[k_KQ_0/nthreads];

        ध्रुव पूर्णाङ्क sumi = ggml_sycl_dp4a(v, u, 0);

        ध्रुव sycl::float2 Q_ds = ((ध्रुव sycl::float2 *) Q_ds_v)[k_KQ_0 / nthreads];

        योग_2 += __half2float(K_q5_0[ib].d) * (sumi*Q_ds.x() - (16/QI8_1)*Q_ds.y());
    }

    प्रत्यागम योग_2;
}

प्रतिरूपम् <पूर्णाङ्क D, पूर्णाङ्क nthreads, पूर्णाङ्क warp_आकार>
स्थिर __dpct_inline__ भिन्न vec_dot_fattn_vec_KQ_q5_1(ध्रुव वर्ण * __restrict__ K_c,
                                                       ध्रुव शून्य * __restrict__ Q_v,
                                                       ध्रुव पूर्णाङ्क * __restrict__ Q_q8,
                                                       ध्रुव शून्य * __restrict__ Q_ds_v) {
    स्वचालित               item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ध्रुव खण्डम्_q5_1 * K_q5_1   = (ध्रुव खण्डम्_q5_1 *) K_c;
    GGML_अप्रयुक्तम्(Q_v);

    भिन्न योग_2 = 0.0f;

#निर्देशः unroll
    प्रति (पूर्णाङ्क k_KQ_0 = 0; k_KQ_0 < पूर्णाङ्क(D/परिमाण(पूर्णाङ्क)); k_KQ_0 += nthreads) {
        ध्रुव पूर्णाङ्क k_KQ =
            k_KQ_0 + (nthreads == warp_आकार ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads);

        ध्रुव पूर्णाङ्क ib    = k_KQ /  QI8_1;
        ध्रुव पूर्णाङ्क iqs4  = k_KQ %  QI5_1;
        ध्रुव पूर्णाङ्क iqs8  = k_KQ %  QI8_1;
        ध्रुव पूर्णाङ्क सरणम् = k_KQ & (QI8_1/2);

        पूर्णाङ्क v;
        ggml_sycl_memcpy_1<परिमाण(पूर्णाङ्क)>(&v, K_q5_1[ib].qs + परिमाण(पूर्णाङ्क)*iqs4);
        v = (v >> सरणम्) & 0x0F0F0F0F;

        {
            पूर्णाङ्क vh;
            ggml_sycl_memcpy_1<परिमाण(पूर्णाङ्क)>(&vh, K_q5_1[ib].qh);
            vh >>= iqs8 * QI5_0;

            v |= (vh <<  4) & 0x00000010; // 0 ->  4
            v |= (vh << 11) & 0x00001000; // 1 -> 12
            v |= (vh << 18) & 0x00100000; // 2 -> 20
            v |= (vh << 25) & 0x10000000; // 3 -> 28
        }

        ध्रुव पूर्णाङ्क u = Q_q8[k_KQ_0/nthreads];

        ध्रुव पूर्णाङ्क sumi = ggml_sycl_dp4a(v, u, 0);

        ध्रुव sycl::float2 K_dm = (K_q5_1[ib].dm).प्रतिरूपम् convert<भिन्न, sycl::rounding_बहुलकम्::automatic>();
        ध्रुव sycl::float2 Q_ds = ((ध्रुव sycl::float2 *) Q_ds_v)[k_KQ_0 / nthreads];

        योग_2 += K_dm.x()*Q_ds.x()*sumi + K_dm.y()*Q_ds.y()/QI8_1;
    }

    प्रत्यागम योग_2;
}

प्रतिरूपम् <पूर्णाङ्क D, पूर्णाङ्क nthreads, पूर्णाङ्क warp_आकार>
स्थिर __dpct_inline__ भिन्न vec_dot_fattn_vec_KQ_q8_0(ध्रुव वर्ण * __restrict__ K_c,
                                                       ध्रुव शून्य * __restrict__ Q_v,
                                                       ध्रुव पूर्णाङ्क * __restrict__ Q_q8,
                                                       ध्रुव शून्य * __restrict__ Q_ds_v) {
    स्वचालित               item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ध्रुव खण्डम्_q8_0 * K_q8_0   = (ध्रुव खण्डम्_q8_0 *) K_c;
    GGML_अप्रयुक्तम्(Q_v);

    भिन्न योग_2 = 0.0f;

#निर्देशः unroll
    प्रति (पूर्णाङ्क k_KQ_0 = 0; k_KQ_0 < पूर्णाङ्क(D/परिमाण(पूर्णाङ्क)); k_KQ_0 += nthreads) {
        ध्रुव पूर्णाङ्क k_KQ =
            k_KQ_0 + (nthreads == warp_आकार ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads);

        ध्रुव पूर्णाङ्क ib  = k_KQ / QI8_0;
        ध्रुव पूर्णाङ्क iqs = k_KQ % QI8_0;

        पूर्णाङ्क v;
        ggml_sycl_memcpy_1<परिमाण(v), 2>(&v, K_q8_0[ib].qs + 4*iqs);

        ध्रुव sycl::float2 * Q_ds = (ध्रुव sycl::float2 *) Q_ds_v;
        ध्रुव भिन्न          Q_d  = Q_ds[k_KQ_0 / nthreads].x();

        योग_2 += vec_dot_q8_0_q8_1_impl<भिन्न, 1>(&v, &Q_q8[k_KQ_0/nthreads], K_q8_0[ib].d, Q_d);
    }

    प्रत्यागम योग_2;
}

प्रतिरूपम् <प्रकारनाम Tds, पूर्णाङ्क ni, पूर्णाङ्क warp_आकार>
स्थिर __dpct_inline__ शून्य quantize_q8_1_गन्तव्यम्_shared(ध्रुव भिन्न * __restrict__ x,
                                                    ध्रुव भिन्न scale,
                                                    पूर्णाङ्क * __restrict__ yq32,
                                                    शून्य * __restrict__ yds) {
    स्वचालित item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    भिन्न vals[परिमाण(पूर्णाङ्क)] = { 0.0f };
#निर्देशः unroll
    प्रति (पूर्णाङ्क l = 0; l < पूर्णाङ्क(परिमाण(पूर्णाङ्क)); ++l) {
        vals[l] =
            (ni == warp_आकार || item_ct1.get_local_id(2) < ni) ? scale * x[4 * item_ct1.get_local_id(2) + l] : 0.0f;
    }

    भिन्न amax = sycl::fabs(vals[0]);
    भिन्न योग_2  = vals[0];
#निर्देशः unroll
    प्रति (पूर्णाङ्क l = 1; l < पूर्णाङ्क(परिमाण(पूर्णाङ्क)); ++l) {
        amax = sycl::fmax(amax, sycl::fabs(vals[l]));
        योग_2 += vals[l];
    }
#निर्देशः unroll
    प्रति (पूर्णाङ्क आवरणम् = QI8_1/2; आवरणम् > 0; आवरणम् >>= 1) {
        amax = sycl::fmax(
            amax, dpct::permute_sub_गणः_by_xor(sycl::ext::oneapi::this_work_item::get_sub_गणः(), amax, आवरणम्));
        योग_2 += dpct::permute_sub_गणः_by_xor(sycl::ext::oneapi::this_work_item::get_sub_गणः(), योग_2, आवरणम्);
    }

    ध्रुव भिन्न d = amax / 127;
    पूर्णाङ्क q32 = 0;
    int8_t * q8 = (int8_t *) &q32;

    यदि (d != 0.0f) {
#निर्देशः unroll
        प्रति (पूर्णाङ्क l = 0; l < पूर्णाङ्क(परिमाण(पूर्णाङ्क)); ++l) {
            q8[l] = sycl::round(vals[l] / d);
        }
    }

    yq32[item_ct1.get_local_id(2)] = q32;
    यदि (item_ct1.get_local_id(2) % QI8_1 == 0 && (ni == warp_आकार || item_ct1.get_local_id(2) < ni)) {
        यदि (std::is_same<Tds, sycl::half2>::value) {
            ((sycl::half2  *) yds)[item_ct1.get_local_id(2)/QI8_1] =  make_half2(d, योग_2);
        } अन्यथा {
            ((sycl::float2 *) yds)[item_ct1.get_local_id(2)/QI8_1] = make_float2(d, योग_2);
        }
    }
}

प्रकारनिर्देश शून्य (*dequantize_V_t)(ध्रुव शून्य *, शून्य *, ध्रुव int64_t);

प्रतिरूपम् <प्रकारनाम T, पूर्णाङ्क ne>
स्थिर __dpct_inline__ शून्य dequantize_V_f16(ध्रुव शून्य * __restrict__ vx, शून्य * __restrict__ dst, ध्रुव int64_t i0) {
    यदि constexpr (std::is_same_v<T, sycl::half>) {
        ggml_sycl_memcpy_1<ne * परिमाण(sycl::half)>(dst, (ध्रुव sycl::half *) vx + i0);
    } अन्यथा यदि constexpr (std::is_same_v<T, भिन्न>) {
        static_assert(ne % 2 == 0, "bad ne");
        sycl::half2 tmp[ne / 2];
        ggml_sycl_memcpy_1<ne * परिमाण(sycl::half)>(tmp, (ध्रुव sycl::half *) vx + i0);
        sycl::float2 * dst_f2 = (sycl::float2 *) dst;
#निर्देशः unroll
        प्रति (पूर्णाङ्क l = 0; l < ne/2; ++l) {
            dst_f2[l] = tmp[l].प्रतिरूपम् convert<भिन्न, sycl::rounding_बहुलकम्::automatic>();
        }
    } अन्यथा {
        static_assert(std::is_same_v<T, शून्य>, "unsupported type");
    }
}

प्रतिरूपम् <प्रकारनाम T, पूर्णाङ्क ne>
स्थिर __dpct_inline__ शून्य dequantize_V_q4_0(ध्रुव शून्य * __restrict__ vx, शून्य * __restrict__ dst, ध्रुव int64_t i0) {
    ध्रुव खण्डम्_q4_0 * x = (ध्रुव खण्डम्_q4_0 *) vx;

    ध्रुव int64_t ib    =  i0          /  QK4_0;
    ध्रुव पूर्णाङ्क     iqs   =  i0          % (QK4_0/2);
    ध्रुव पूर्णाङ्क     सरणम् = (i0 % QK4_0) / (QK4_0/2);

    पूर्णाङ्क q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_sycl_memcpy_1<ne, 2>(&q, x[ib].qs + iqs);
    q >>= 4*सरणम्;
    q &= 0x0F0F0F0F;
    q = dpct::vectorized_द्विमान<sycl::char4>(q, 0x08080808, dpct::sub_sat());

    ध्रुव int8_t * q8 = (ध्रुव int8_t *) &q;

#यदि_परिभाषितम् GGML_SYCL_F16
    यदि constexpr (std::is_same_v<T, sycl::half>) {
        ध्रुव sycl::half2 d = sycl::half2(x[ib].d);

#निर्देशः unroll
        प्रति (पूर्णाङ्क l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * sycl::half2(q8[l0 + 0], q8[l0 + 1]);
        }
    } अन्यथा
#यदि_अन्तः // GGML_SYCL_F16
    यदि constexpr (std::is_same_v<T, भिन्न>) {
        ध्रुव भिन्न d = x[ib].d;

#निर्देशः unroll
        प्रति (पूर्णाङ्क l = 0; l < ne; ++l) {
            ((भिन्न *) dst)[l] = d * q8[l];
        }
    } अन्यथा {
        static_assert(std::is_same_v<T, शून्य>, "bad type");
    }
}

प्रतिरूपम् <प्रकारनाम T, पूर्णाङ्क ne>
स्थिर __dpct_inline__ शून्य dequantize_V_q4_1(ध्रुव शून्य * __restrict__ vx, शून्य * __restrict__ dst, ध्रुव int64_t i0) {
    ध्रुव खण्डम्_q4_1 * x = (ध्रुव खण्डम्_q4_1 *) vx;

    ध्रुव int64_t ib    =  i0          /  QK4_1;
    ध्रुव पूर्णाङ्क     iqs   =  i0          % (QK4_1/2);
    ध्रुव पूर्णाङ्क     सरणम् = (i0 % QK4_1) / (QK4_1/2);

    पूर्णाङ्क q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_sycl_memcpy_1<ne>(&q, x[ib].qs + iqs);
    q >>= 4*सरणम्;
    q &= 0x0F0F0F0F;

    ध्रुव int8_t * q8 = (ध्रुव int8_t *) &q;

#यदि_परिभाषितम् GGML_SYCL_F16
    यदि constexpr (std::is_same_v<T, sycl::half>) {
        ध्रुव sycl::half2 dm = x[ib].dm;
        ध्रुव sycl::half2 d  = sycl::half2(dm[0]);
        ध्रुव sycl::half2 m  = sycl::half2(dm[1]);

#निर्देशः unroll
        प्रति (पूर्णाङ्क l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * sycl::half2(q8[l0 + 0], q8[l0 + 1]) + m;
        }
    } अन्यथा
#यदि_अन्तः // GGML_SYCL_F16
    यदि constexpr (std::is_same_v<T, भिन्न>) {
        ध्रुव sycl::float2 dm = (x[ib].dm).प्रतिरूपम् convert<भिन्न, sycl::rounding_बहुलकम्::automatic>();

#निर्देशः unroll
        प्रति (पूर्णाङ्क l = 0; l < ne; ++l) {
            ((भिन्न *) dst)[l] = dm.x() * q8[l] + dm.y();
        }
    } अन्यथा {
        static_assert(std::is_same_v<T, शून्य>, "bad type");
    }
}

प्रतिरूपम् <प्रकारनाम T, पूर्णाङ्क ne>
स्थिर __dpct_inline__ शून्य dequantize_V_q5_0(ध्रुव शून्य * __restrict__ vx, शून्य * __restrict__ dst, ध्रुव int64_t i0) {
    ध्रुव खण्डम्_q5_0 * x = (ध्रुव खण्डम्_q5_0 *) vx;

    ध्रुव int64_t ib    =  i0          /  QK5_0;
    ध्रुव पूर्णाङ्क     idq   =  i0          %  QK5_0;
    ध्रुव पूर्णाङ्क     iqs   =  i0          % (QK5_0/2);
    ध्रुव पूर्णाङ्क     सरणम् = (i0 % QK5_0) / (QK5_0/2);

    पूर्णाङ्क q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_sycl_memcpy_1<ne, 2>(&q, x[ib].qs + iqs);
    q >>= 4*सरणम्;
    q &= 0x0F0F0F0F;

    {
        पूर्णाङ्क qh;
        ggml_sycl_memcpy_1<ne, 2>(&qh, x[ib].qh);
#निर्देशः unroll
        प्रति (पूर्णाङ्क l = 0; l < ne; ++l) {
            q |= ((qh >> (idq + l)) & 0x00000001) << (8*l + 4);
        }
    }

    q = dpct::vectorized_द्विमान<sycl::char4>(q, 0x10101010, dpct::sub_sat());

    ध्रुव int8_t * q8 = (ध्रुव int8_t *) &q;

#यदि_परिभाषितम् GGML_SYCL_F16
    यदि constexpr (std::is_same_v<T, sycl::half>) {
        ध्रुव sycl::half2 d = sycl::half2(x[ib].d);

#निर्देशः unroll
        प्रति (पूर्णाङ्क l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * sycl::half2(q8[l0 + 0], q8[l0 + 1]);
        }
    } अन्यथा
#यदि_अन्तः // GGML_SYCL_F16
    यदि constexpr (std::is_same_v<T, भिन्न>) {
        ध्रुव भिन्न d = x[ib].d;

#निर्देशः unroll
        प्रति (पूर्णाङ्क l = 0; l < ne; ++l) {
            ((भिन्न *) dst)[l] = d * q8[l];
        }
    } अन्यथा {
        static_assert(std::is_same_v<T, शून्य>, "bad type");
    }
}

प्रतिरूपम् <प्रकारनाम T, पूर्णाङ्क ne>
स्थिर __dpct_inline__ शून्य dequantize_V_q5_1(ध्रुव शून्य * __restrict__ vx, शून्य * __restrict__ dst, ध्रुव int64_t i0) {
    ध्रुव खण्डम्_q5_1 * x = (ध्रुव खण्डम्_q5_1 *) vx;

    ध्रुव int64_t ib    =  i0          /  QK5_1;
    ध्रुव पूर्णाङ्क     idq   =  i0          %  QK5_1;
    ध्रुव पूर्णाङ्क     iqs   =  i0          % (QK5_1/2);
    ध्रुव पूर्णाङ्क     सरणम् = (i0 % QK5_1) / (QK5_1/2);

    पूर्णाङ्क q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_sycl_memcpy_1<ne>(&q, x[ib].qs + iqs);
    q >>= 4*सरणम्;
    q &= 0x0F0F0F0F;

    {
        पूर्णाङ्क qh;
        ggml_sycl_memcpy_1<ne>(&qh, x[ib].qh);
#निर्देशः unroll
        प्रति (पूर्णाङ्क l = 0; l < ne; ++l) {
            q |= ((qh >> (idq + l)) & 0x00000001) << (8*l + 4);
        }
    }

    ध्रुव int8_t * q8 = (ध्रुव int8_t *) &q;

#यदि_परिभाषितम् GGML_SYCL_F16
    यदि constexpr (std::is_same_v<T, sycl::half>) {
        ध्रुव sycl::half2 dm = x[ib].dm;
        ध्रुव sycl::half2 d  = sycl::half2(dm[0]);
        ध्रुव sycl::half2 m  = sycl::half2(dm[1]);

#निर्देशः unroll
        प्रति (पूर्णाङ्क l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * sycl::half2(q8[l0 + 0], q8[l0 + 1]) + m;
        }
    } अन्यथा
#यदि_अन्तः // GGML_SYCL_F16
    यदि constexpr (std::is_same_v<T, भिन्न>) {
        ध्रुव sycl::float2 dm = (x[ib].dm).प्रतिरूपम् convert<भिन्न, sycl::rounding_बहुलकम्::automatic>();

#निर्देशः unroll
        प्रति (पूर्णाङ्क l = 0; l < ne; ++l) {
            ((भिन्न *) dst)[l] = dm.x() * q8[l] + dm.y();
        }
    } अन्यथा {
        static_assert(std::is_same_v<T, शून्य>, "bad type");
    }
}

प्रतिरूपम् <प्रकारनाम T, पूर्णाङ्क ne>
स्थिर __dpct_inline__ शून्य dequantize_V_q8_0(ध्रुव शून्य * __restrict__ vx, शून्य * __restrict__ dst, ध्रुव int64_t i0) {
    ध्रुव खण्डम्_q8_0 * x = (ध्रुव खण्डम्_q8_0 *) vx;

    ध्रुव int64_t ib  = i0 / QK8_0;
    ध्रुव पूर्णाङ्क     iqs = i0 % QK8_0;

    static_assert(ne % 2 == 0, "bad ne");
    int8_t qs[ne];
    ggml_sycl_memcpy_1<ne, 2>(qs, x[ib].qs + iqs);

#यदि_परिभाषितम् GGML_SYCL_F16
    यदि constexpr (std::is_same<T, sycl::half>::value) {
        ध्रुव sycl::half2 d = sycl::half2(x[ib].d);

#निर्देशः unroll
        प्रति (पूर्णाङ्क l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * make_half2(qs[l0 + 0], qs[l0 + 1]);
        }
    } अन्यथा
#यदि_अन्तः // GGML_SYCL_F16
    यदि constexpr (std::is_same<T, भिन्न>::value) {
        ध्रुव भिन्न d = x[ib].d;

#निर्देशः unroll
        प्रति (पूर्णाङ्क l = 0; l < ne; ++l) {
            ((भिन्न *) dst)[l] = d * qs[l];
        }
    } अन्यथा {
        static_assert(std::is_same_v<T, शून्य>, "unsupported type");
    }
}

प्रतिरूपम् <पूर्णाङ्क type_K, पूर्णाङ्क D, पूर्णाङ्क nthreads, पूर्णाङ्क warp_आकार>
constexpr vec_dot_KQ_t get_vec_dot_KQ() {
    यदि constexpr (type_K == GGML_TYPE_F16) {
        प्रत्यागम vec_dot_fattn_vec_KQ_f16<D, nthreads>;
    } अन्यथा यदि constexpr (type_K == GGML_TYPE_Q4_0) {
        प्रत्यागम vec_dot_fattn_vec_KQ_q4_0<D, nthreads, warp_आकार>;
    } अन्यथा यदि constexpr (type_K == GGML_TYPE_Q4_1) {
        प्रत्यागम vec_dot_fattn_vec_KQ_q4_1<D, nthreads, warp_आकार>;
    } अन्यथा यदि constexpr (type_K == GGML_TYPE_Q5_0) {
        प्रत्यागम vec_dot_fattn_vec_KQ_q5_0<D, nthreads, warp_आकार>;
    } अन्यथा यदि constexpr (type_K == GGML_TYPE_Q5_1) {
        प्रत्यागम vec_dot_fattn_vec_KQ_q5_1<D, nthreads, warp_आकार>;
    } अन्यथा यदि constexpr (type_K == GGML_TYPE_Q8_0) {
        प्रत्यागम vec_dot_fattn_vec_KQ_q8_0<D, nthreads, warp_आकार>;
    } अन्यथा {
        static_assert(type_K == -1, "bad type");
        प्रत्यागम शून्यसूचकः;
    }
}

प्रतिरूपम् <पूर्णाङ्क type_V, प्रकारनाम T, पूर्णाङ्क ne>
constexpr dequantize_V_t get_dequantize_V() {
    यदि constexpr (type_V == GGML_TYPE_F16) {
        प्रत्यागम dequantize_V_f16<T, ne>;
    } अन्यथा यदि constexpr (type_V == GGML_TYPE_Q4_0) {
        प्रत्यागम dequantize_V_q4_0<T, ne>;
    } अन्यथा यदि constexpr (type_V == GGML_TYPE_Q4_1) {
        प्रत्यागम dequantize_V_q4_1<T, ne>;
    } अन्यथा यदि constexpr (type_V == GGML_TYPE_Q5_0) {
        प्रत्यागम dequantize_V_q5_0<T, ne>;
    } अन्यथा यदि constexpr (type_V == GGML_TYPE_Q5_1) {
        प्रत्यागम dequantize_V_q5_1<T, ne>;
    } अन्यथा यदि constexpr (type_V == GGML_TYPE_Q8_0) {
        प्रत्यागम dequantize_V_q8_0<T, ne>;
    } अन्यथा {
        static_assert(type_V == -1, "bad type");
        प्रत्यागम शून्यसूचकः;
    }
}

प्रतिरूपम् <पूर्णाङ्क ncols1, पूर्णाङ्क warp_आकार>
स्थिर शून्य flash_attn_आवरणम्_गन्तव्यम्_KV_max(ध्रुव sycl::half2 * __restrict__ आवरणम्,
                                      पूर्णाङ्क * __restrict__ KV_max,
                                      ध्रुव पूर्णाङ्क ne30,
                                      ध्रुव पूर्णाङ्क s31,
                                      ध्रुव पूर्णाङ्क s33,
                                      पूर्णाङ्क *     buf_iw) {
    स्वचालित      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ध्रुव पूर्णाङ्क ne31     = item_ct1.get_गणः_range(2);
    ध्रुव पूर्णाङ्क tid      = item_ct1.get_local_id(2);
    ध्रुव पूर्णाङ्क sequence = item_ct1.get_गणः(1);
    ध्रुव पूर्णाङ्क jt       = item_ct1.get_गणः(2);

    आवरणम् += sequence*s33 + jt*ncols1*s31;

    यदि (tid < warp_आकार) {
        buf_iw[tid] = 1;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);

    पूर्णाङ्क KV_max_sj = (ne30 - 1) * FATTN_KQ_STRIDE;
    प्रति (; KV_max_sj >= 0; KV_max_sj -= FATTN_KQ_STRIDE) {
        पूर्णाङ्क all_inf = 1;

#निर्देशः unroll
        प्रति (पूर्णाङ्क j = 0; j < ncols1; ++j) {
            ध्रुव sycl::float2 tmp =
                आवरणम्[j * s31 + KV_max_sj / 2 + tid].प्रतिरूपम् convert<भिन्न, sycl::rounding_बहुलकम्::automatic>();
            all_inf = all_inf && पूर्णाङ्क(sycl::isinf((भिन्न) (tmp.x()))) && पूर्णाङ्क(sycl::isinf((भिन्न) (tmp.y())));
        }

        all_inf = warp_reduce_all<warp_आकार>(all_inf);
        यदि (tid % warp_आकार == 0) {
            buf_iw[tid / warp_आकार] = all_inf;
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);
        all_inf = buf_iw[tid % warp_आकार];
        item_ct1.barrier(sycl::access::fence_space::local_space);
        all_inf = warp_reduce_all<warp_आकार>(all_inf);

        यदि (!all_inf) {
            भङ्ग;
        }
    }

    // If the break in the loop was not triggered, KV_max_sj is now -FATTN_KQ_STRIDE.
    // If the break was triggered it's the lower edge of the tile with the first non-masked values.
    // In either case, walk back the decrementation by FATTN_KQ_STRIDE.
    KV_max_sj += FATTN_KQ_STRIDE;

    यदि (item_ct1.get_local_id(2) != 0) {
        प्रत्यागम;
    }

    KV_max[sequence*ne31 + jt] = KV_max_sj;
}

प्रतिरूपम् <पूर्णाङ्क D, पूर्णाङ्क ncols1, पूर्णाङ्क ncols2>  // D == head size

स्थिर शून्य flash_attn_प्रवाहः_k_fixup(भिन्न * __restrict__ dst,
                                      ध्रुव sycl::float2 * __restrict__ dst_fixup,
                                      ध्रुव पूर्णाङ्क ne01,
                                      ध्रुव पूर्णाङ्क ne02,
                                      ध्रुव पूर्णाङ्क ne03,
                                      ध्रुव पूर्णाङ्क ne11,
                                      ध्रुव पूर्णाङ्क ne12,
                                      ध्रुव पूर्णाङ्क nbatch_fa) {
    स्वचालित          item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr पूर्णाङ्क ncols    = ncols1 * ncols2;

    ध्रुव पूर्णाङ्क bidx0 = item_ct1.get_गणः(2);
    ध्रुव पूर्णाङ्क j     = item_ct1.get_गणः(1);
    ध्रुव पूर्णाङ्क c     = item_ct1.get_गणः(0);
    ध्रुव पूर्णाङ्क jc    = j*ncols2 + c;
    ध्रुव पूर्णाङ्क tid   = item_ct1.get_local_id(2);

    ध्रुव भिन्न * dst_fixup_दत्त = ((ध्रुव भिन्न *) dst_fixup) + item_ct1.get_गणः_range(2) * (2 * 2 * ncols);

    ध्रुव पूर्णाङ्क gqa_अनुपातः = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.

    ध्रुव पूर्णाङ्क iter_k     = (ne11      + (nbatch_fa - 1)) / nbatch_fa;
    ध्रुव पूर्णाङ्क iter_j     = (ne01      + (ncols1    - 1)) / ncols1;
    ध्रुव पूर्णाङ्क iter_z_gqa = (gqa_अनुपातः + (ncols2    - 1)) / ncols2;

    ध्रुव पूर्णाङ्क kbc0 = int64_t(bidx0 + 0) * (iter_k * iter_j * iter_z_gqa * ne12 * ne03) / item_ct1.get_गणः_range(2);
    ध्रुव पूर्णाङ्क kbc0_stop =
        int64_t(bidx0 + 1) * (iter_k * iter_j * iter_z_gqa * ne12 * ne03) / item_ct1.get_गणः_range(2);

    ध्रुव तार्किक did_not_have_any_दत्त   = kbc0 == kbc0_stop;
    ध्रुव तार्किक wrote_beginning_सम्बन्धि_tile = kbc0 % iter_k == 0;
    ध्रुव तार्किक did_not_write_चरमः      = kbc0/iter_k == kbc0_stop/iter_k && kbc0_stop % iter_k != 0;
    यदि (did_not_have_any_दत्त || wrote_beginning_सम्बन्धि_tile || did_not_write_चरमः) {
        प्रत्यागम;
    }

    // z_KV == K/V head index, zt_gqa = Q head start index per K/V head, jt = token position start index
    ध्रुव पूर्णाङ्क sequence =  kbc0 /(iter_k*iter_j*iter_z_gqa*ne12);
    ध्रुव पूर्णाङ्क z_KV     = (kbc0 - iter_k*iter_j*iter_z_gqa*ne12 * sequence)/(iter_k*iter_j*iter_z_gqa);
    ध्रुव पूर्णाङ्क zt_gqa   = (kbc0 - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV)/(iter_k*iter_j);
    ध्रुव पूर्णाङ्क jt       = (kbc0 - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV - iter_k*iter_j * zt_gqa) / iter_k;

    ध्रुव पूर्णाङ्क zt_Q = z_KV*gqa_अनुपातः + zt_gqa*ncols2; // Global Q head start index.

    यदि (jt*ncols1 + j >= ne01 || zt_gqa*ncols2 + c >= gqa_अनुपातः) {
        प्रत्यागम;
    }

    dst += sequence*ne02*ne01*D + jt*ne02*(ncols1*D) + zt_Q*D + (j*ne02 + c)*D + tid;

    // Load the partial result that needs a fixup:
    भिन्न dst_val = 0.0f;
    भिन्न max_val = 0.0f;
    भिन्न rowsum  = 0.0f;
    {
        dst_val = *dst;

        ध्रुव sycl::float2 tmp = dst_fixup[bidx0 * ncols + jc];
        max_val                = tmp.x();
        rowsum                 = tmp.y();
    }

    // Iterate over previous blocks and compute the combined results.
    // All SYCL blocks that get here must have a previous block that needs a fixup.
    पूर्णाङ्क bidx = bidx0 - 1;
    पूर्णाङ्क kbc_stop = kbc0;
    यावत्(सत्य) {
        ध्रुव पूर्णाङ्क kbc = int64_t(bidx) * (iter_k * iter_j * iter_z_gqa * ne12 * ne03) / item_ct1.get_गणः_range(2);
        यदि (kbc == kbc_stop) { // Did not have any data.
            bidx--;
            kbc_stop = kbc;
            अनुवर्त;
        }

        ध्रुव भिन्न dst_सङ्कलय = dst_fixup_दत्त[bidx*ncols*D + jc*D + tid];

        ध्रुव sycl::float2 tmp = dst_fixup[(item_ct1.get_गणः_range(2) + bidx) * ncols + jc];

        // Scale the current and new value accumulators depending on the max. values.
        ध्रुव भिन्न max_val_new = sycl::fmax(max_val, tmp.x());

        ध्रुव भिन्न diff_val = max_val - max_val_new;
        ध्रुव भिन्न diff_सङ्कलय = tmp.x() - max_val_new;

        ध्रुव भिन्न scale_val = diff_val >= SOFTMAX_FTZ_देहली ? sycl::native::घातांकः(diff_val) : 0.0f;
        ध्रुव भिन्न scale_सङ्कलय = diff_सङ्कलय >= SOFTMAX_FTZ_देहली ? sycl::native::घातांकः(diff_सङ्कलय) : 0.0f;

        dst_val = scale_val*dst_val + scale_सङ्कलय*dst_सङ्कलय;
        rowsum  = scale_val * rowsum + scale_सङ्कलय * tmp.y();

        max_val = max_val_new;

        // If this block started in a previous tile we are done and don't need to combine additional partial results.
        यदि (kbc % iter_k == 0 || kbc/iter_k < kbc0/iter_k) {
            भङ्ग;
        }
        bidx--;
        kbc_stop = kbc;
    }

    // Write back final result:
    *dst = dst_val / rowsum;
}

प्रतिरूपम् <पूर्णाङ्क D>  // D == head size

स्थिर शून्य flash_attn_combine_results(ध्रुव भिन्न * __restrict__ VKQ_parts,
                                       ध्रुव sycl::float2 * __restrict__ VKQ_meta,
                                       भिन्न * __restrict__ dst,
                                       ध्रुव पूर्णाङ्क parallel_खण्डानि,
                                       uint8_t * dpct_local) {
    // Dimension 0: threadIdx.x
    // Dimension 1: blockIdx.x
    // Dimension 2: blockIdx.y
    // Dimension 3: blockIdx.z
    // Memory layout is permuted with [0, 2, 1, 3]

    स्वचालित      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ध्रुव पूर्णाङ्क ne01     = item_ct1.get_गणः_range(2);
    ध्रुव पूर्णाङ्क ne02     = item_ct1.get_गणः_range(1);

    ध्रुव पूर्णाङ्क col      = item_ct1.get_गणः(2);
    ध्रुव पूर्णाङ्क शीर्ष     = item_ct1.get_गणः(1);
    ध्रुव पूर्णाङ्क sequence = item_ct1.get_गणः(0);

    ध्रुव पूर्णाङ्क j_dst_unrolled = (sequence*ne01 + col)*ne02 + शीर्ष;

    VKQ_parts += j_dst_unrolled * parallel_खण्डानि*D;
    VKQ_meta  += j_dst_unrolled * parallel_खण्डानि;
    dst       += j_dst_unrolled *                 D;

    ध्रुव पूर्णाङ्क tid = item_ct1.get_local_id(2);
    __builtin_assume(tid < D);

    स्वचालित meta = (sycl::float2 *) dpct_local;
    प्रति (पूर्णाङ्क i = tid; i < 2*parallel_खण्डानि; i += D) {
        ((भिन्न *) meta)[i] = ((ध्रुव भिन्न *)VKQ_meta) [i];
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    भिन्न kqmax = meta[0].x();
    प्रति (पूर्णाङ्क l = 1; l < parallel_खण्डानि; ++l) {
        kqmax = sycl::max(kqmax, meta[l].x());
    }

    भिन्न VKQ_numerator   = 0.0f;
    भिन्न VKQ_denominator = 0.0f;
    प्रति (पूर्णाङ्क l = 0; l < parallel_खण्डानि; ++l) {
        ध्रुव भिन्न KQ_max_scale = sycl::native::घातांकः(meta[l].x() - kqmax);

        VKQ_numerator   += KQ_max_scale * VKQ_parts[l*D + tid];
        VKQ_denominator += KQ_max_scale * meta[l].y();
    }

    dst[tid] = VKQ_numerator / VKQ_denominator;
}

// Carries the sub-group size (and grf_size_automatic when use_large_grf) as
// kernel properties via get(properties_tag), in place of the
// reqd_sub_group_size attribute and the deprecated parallel_for overload.
प्रतिरूपम् <प्रकारनाम KernelFunc, uint32_t आकार_3, तार्किक UseGrf>
संरचना props_kernel {
    KernelFunc kernel;

    शून्य संक्रियकः()(sycl::nd_item<3> item) ध्रुव {
        kernel(item);
    }

    स्वचालित get(sycl::ext::oneapi::experimental::properties_चिह्नकम्) ध्रुव {
        यदि constexpr (UseGrf) {
            प्रत्यागम sycl::ext::oneapi::experimental::properties{
                sycl::ext::intel::experimental::grf_आकार_automatic,
                sycl::ext::oneapi::experimental::sub_गणः_आकार<आकार_3> };
        } अन्यथा {
            प्रत्यागम sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::sub_गणः_आकार<आकार_3> };
        }
    }
};

प्रतिरूपम् <fattn_kernel_t fattn_kernel, पूर्णाङ्क warp_आकार, तार्किक use_large_grf = असत्य>
स्थिर शून्य lauch_kernel(
    dpct::dim3 गणः_range,
    dpct::dim3 local_range,
    queue_ptr q,
    अचिह्नित पूर्णाङ्क local_mem_आकार_2,
    ध्रुव वर्ण* __restrict__ Q,
    ध्रुव वर्ण* __restrict__ K,
    ध्रुव वर्ण* __restrict__ V,
    ध्रुव वर्ण* __restrict__ आवरणम्,
    ध्रुव वर्ण* __restrict__ sinks,
    ध्रुव पूर्णाङ्क* __restrict__ KV_max,
    भिन्न* __restrict__ dst,
    sycl::float2* __restrict__ dst_meta,
    ध्रुव भिन्न scale,
    ध्रुव भिन्न max_bias,
    ध्रुव भिन्न m0,
    ध्रुव भिन्न m1,
    ध्रुव uint32_t n_शीर्ष_log2,
    ध्रुव भिन्न logit_softcap,
    ध्रुव int32_t ne00,
    ध्रुव sycl::uint3 ne01,
    ध्रुव int32_t ne02,
    ध्रुव int32_t ne03,
    ध्रुव int32_t nb01,
    ध्रुव int32_t nb02,
    ध्रुव int32_t nb03,
    ध्रुव int32_t ne10,
    ध्रुव int32_t ne11,
    ध्रुव int32_t ne12,
    ध्रुव int32_t ne13,
    ध्रुव int32_t nb11,
    ध्रुव int32_t nb12,
    ध्रुव int64_t nb13,
    ध्रुव int32_t nb21,
    ध्रुव int32_t nb22,
    ध्रुव int64_t nb23,
    ध्रुव int32_t ne31,
    ध्रुव int32_t ne32,
    ध्रुव int32_t ne33,
    ध्रुव int32_t nb31,
    ध्रुव int32_t nb32,
    ध्रुव int64_t nb33) {
    GGML_अप्रयुक्तम्(local_mem_आकार_2);

    ध्रुव स्वचालित rng = sycl::nd_range<3>(
        static_cast<sycl::range<3>>(गणः_range * local_range),
        static_cast<sycl::range<3>>(local_range));

    ध्रुव स्वचालित kernel = [=](sycl::nd_item<3> item_ct1) {
        GGML_अप्रयुक्तम्(item_ct1);
        fattn_kernel(Q, K, V, आवरणम्, sinks, KV_max, dst, dst_meta, scale,
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
प्रतिरूपम् <पूर्णाङ्क DV, पूर्णाङ्क ncols1, पूर्णाङ्क ncols2, fattn_kernel_t fattn_kernel, पूर्णाङ्क warp_आकार, तार्किक use_large_grf = असत्य>
शून्य launch_fattn(
    ggml_backend_sycl_context & ctx, ggml_tensor * dst, ध्रुव पूर्णाङ्क nwarps, ध्रुव size_t nbytes_shared,
    ध्रुव पूर्णाङ्क nbatch_fa, ध्रुव तार्किक need_f16_K, ध्रुव तार्किक need_f16_V, ध्रुव तार्किक प्रवाहः_k) {

    constexpr पूर्णाङ्क ncols = ncols1 * ncols2;

    ध्रुव ggml_tensor * Q = dst->src[0];
    ध्रुव ggml_tensor * K = dst->src[1];
    ध्रुव ggml_tensor * V = dst->src[2];

    ध्रुव तार्किक V_is_K_view = V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));

    ध्रुव ggml_tensor * आवरणम्  = dst->src[3];
    ध्रुव ggml_tensor * sinks = dst->src[4];

    ggml_tensor * KQV = dst;

    GGML_ASSERT(Q->type == GGML_TYPE_F32);
    GGML_ASSERT(KQV->type == GGML_TYPE_F32);

    GGML_ASSERT(Q->nb[0] == ggml_element_आकार(Q));
    GGML_ASSERT(K->nb[0] == ggml_element_आकार(K));
    GGML_ASSERT(V->nb[0] == ggml_element_आकार(V));

    GGML_ASSERT(!आवरणम् || आवरणम्->type == GGML_TYPE_F16);

    ggml_sycl_pool & pool = ctx.pool();
    dpct::queue_ptr  main_प्रवाहः = ctx.stream();
    ध्रुव पूर्णाङ्क id  = ggml_sycl_get_device();
    ध्रुव पूर्णाङ्क nsm = ggml_sycl_info().devices[id].nsm;

    ध्रुव ggml_sycl_fattn_extra  extra = ggml_sycl_fattn_get_extra(dst);
    ggml_sycl_pool_alloc<पूर्णाङ्क>    KV_max(pool);
    ggml_sycl_pool_alloc<भिन्न>  dst_tmp(pool);
    ggml_sycl_pool_alloc<sycl::float2> dst_tmp_meta(pool);

    ध्रुव वर्ण * K_दत्त = (ध्रुव वर्ण *) K->दत्त;
    size_t nb11 = K->nb[1];
    size_t nb12 = K->nb[2];
    size_t nb13 = K->nb[3];

    ध्रुव वर्ण * V_दत्त = (ध्रुव वर्ण *) V->दत्त;
    size_t nb21 = V->nb[1];
    size_t nb22 = V->nb[2];
    size_t nb23 = V->nb[3];

    यदि (need_f16_K && K->type != GGML_TYPE_F16) {
        ध्रुव size_t bs = ggml_blck_आकार(K->type);
        ध्रुव size_t ts = ggml_type_आकार(K->type);

        GGML_ASSERT(extra.K_बफर_ptr);
        sycl::half * K_f16_ptr = (sycl::half *) extra.K_बफर_ptr;
        यदि (ggml_is_contiguously_allocated(K)) {
            गन्तव्यम्_fp16_sycl_t गन्तव्यम्_fp16 = ggml_get_गन्तव्यम्_fp16_sycl(K->type, dst);
            गन्तव्यम्_fp16(K_दत्त, K_f16_ptr, ggml_nelements(K), main_प्रवाहः);

            nb11 = nb11 * bs * परिमाण(sycl::half) / ts;
            nb12 = nb12 * bs * परिमाण(sycl::half) / ts;
            nb13 = nb13 * bs * परिमाण(sycl::half) / ts;
        } अन्यथा {
            GGML_ASSERT(K->nb[0] == ts);
            गन्तव्यम्_fp16_nc_sycl_t गन्तव्यम्_fp16 = ggml_get_गन्तव्यम्_fp16_nc_sycl(K->type);
            ध्रुव int64_t s01 = nb11 / ts;
            ध्रुव int64_t s02 = nb12 / ts;
            ध्रुव int64_t s03 = nb13 / ts;
            गन्तव्यम्_fp16(K_दत्त, K_f16_ptr, K->ne[0], K->ne[1], K->ne[2], K->ne[3], s01, s02, s03, main_प्रवाहः);

            nb11 = K->ne[0] * परिमाण(sycl::half);
            nb12 = K->ne[1] * nb11;
            nb13 = K->ne[2] * nb12;
        }
        K_दत्त = (वर्ण *) K_f16_ptr;
    }

    यदि (need_f16_V && V->type != GGML_TYPE_F16) {
        यदि (V_is_K_view) {
            V_दत्त = K_दत्त;
            nb21   = nb11;
            nb22   = nb12;
            nb23   = nb13;
        } अन्यथा {
            ध्रुव size_t bs = ggml_blck_आकार(V->type);
            ध्रुव size_t ts = ggml_type_आकार(V->type);

            GGML_ASSERT(extra.V_बफर_ptr);
            sycl::half * V_f16_ptr = (sycl::half *) extra.V_बफर_ptr;
            यदि (ggml_is_contiguously_allocated(V)) {
                गन्तव्यम्_fp16_sycl_t गन्तव्यम्_fp16 = ggml_get_गन्तव्यम्_fp16_sycl(V->type, dst);
                गन्तव्यम्_fp16(V_दत्त, V_f16_ptr, ggml_nelements(V), main_प्रवाहः);
                V_दत्त = (वर्ण *) V_f16_ptr;

                nb21 = nb21 * bs * परिमाण(sycl::half) / ts;
                nb22 = nb22 * bs * परिमाण(sycl::half) / ts;
                nb23 = nb23 * bs * परिमाण(sycl::half) / ts;
            } अन्यथा {
                GGML_ASSERT(V->nb[0] == ts);
                गन्तव्यम्_fp16_nc_sycl_t गन्तव्यम्_fp16 = ggml_get_गन्तव्यम्_fp16_nc_sycl(V->type);
                ध्रुव int64_t s01 = nb21 / ts;
                ध्रुव int64_t s02 = nb22 / ts;
                ध्रुव int64_t s03 = nb23 / ts;
                गन्तव्यम्_fp16(V_दत्त, V_f16_ptr, V->ne[0], V->ne[1], V->ne[2], V->ne[3], s01, s02, s03, main_प्रवाहः);

                nb21 = V->ne[0] * परिमाण(sycl::half);
                nb22 = V->ne[1] * nb21;
                nb23 = V->ne[2] * nb22;
            }
            V_दत्त = (वर्ण *) V_f16_ptr;
        }
    }

    ध्रुव पूर्णाङ्क ntiles_x     = ((Q->ne[1] + ncols1 - 1) / ncols1);
    ध्रुव पूर्णाङ्क gqa_अनुपातः    = Q->ne[2] / K->ne[2];
    ध्रुव पूर्णाङ्क ntiles_z_gqa = ((gqa_अनुपातः + ncols2 - 1) / ncols2);
    ध्रुव पूर्णाङ्क ntiles_योग = ntiles_x * ntiles_z_gqa * K->ne[2] * Q->ne[3];

    // Optional optimization where the mask is scanned to determine whether part of the calculation can be skipped.
    // Only worth the overhead if there is at lease one FATTN_KQ_STRIDE x FATTN_KQ_STRIDE square to be skipped or
    //     multiple sequences of possibly different lengths.
    यदि (आवरणम् && K->ne[1] % FATTN_KQ_STRIDE == 0 && (Q->ne[1] >= 1024 || Q->ne[3] > 1)) {
        ध्रुव पूर्णाङ्क s31 = आवरणम्->nb[1] / परिमाण(sycl::half2);
        ध्रुव पूर्णाङ्क s33 = आवरणम्->nb[3] / परिमाण(sycl::half2);

        ध्रुव dpct::dim3 खण्डानि_अङ्कः_KV_max(ntiles_x, Q->ne[3], 1);
        ध्रुव dpct::dim3 खण्डम्_dim_KV_max(FATTN_KQ_STRIDE / 2, 1, 1);

        ध्रुव पूर्णाङ्क ne_KV_max = खण्डानि_अङ्कः_KV_max.x*खण्डानि_अङ्कः_KV_max.y;
        ध्रुव पूर्णाङ्क iter_k = K->ne[1] / FATTN_KQ_STRIDE;

        KV_max.alloc(ne_KV_max);
        {
            dpct::has_capability_or_fail(main_प्रवाहः->get_device(), { sycl::aspect::fp16 });

            main_प्रवाहः->submit([&](sycl::handler & cgh) {
                sycl::local_accessor<पूर्णाङ्क, 1> buf_iw_acc_ct1(sycl::range<1>(warp_आकार), cgh);

                स्वचालित आवरणम्_दत्त_ct0  = (ध्रुव sycl::half2 *) आवरणम्->दत्त;
                स्वचालित KV_max_ptr_ct1 = KV_max.ptr;

                cgh.parallel_for(sycl::nd_range<3>(खण्डानि_अङ्कः_KV_max * खण्डम्_dim_KV_max, खण्डम्_dim_KV_max),
                                 [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_गणः_आकार(warp_आकार)]] {
                                     GGML_अप्रयुक्तम्(item_ct1);
                                     flash_attn_आवरणम्_गन्तव्यम्_KV_max<ncols1, warp_आकार>(
                                         आवरणम्_दत्त_ct0, KV_max_ptr_ct1, iter_k, s31, s33,
                                         buf_iw_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                                 });
            });
        }
        SYCL_CHECK(0);
    }

    ध्रुव dpct::dim3 खण्डम्_dim(warp_आकार, nwarps, 1);

    // Max. number of active blocks limited by occupancy.
    पूर्णाङ्क max_खण्डानि_प्रत्येकम्_sm = ggml_sycl_info().devices[id].max_wg_प्रत्येकम्_cu;
    पूर्णाङ्क parallel_खण्डानि = max_खण्डानि_प्रत्येकम्_sm;
    dpct::dim3 खण्डानि_अङ्कः;
    यदि (प्रवाहः_k) {
        // For short contexts it can be faster to have the SMs work on whole tiles because this lets us skip the fixup.
        ध्रुव पूर्णाङ्क max_खण्डानि = max_खण्डानि_प्रत्येकम्_sm*nsm;
        ध्रुव पूर्णाङ्क nblocks_प्रवाहः_k = max_खण्डानि;
        ध्रुव तार्किक use_प्रवाहः_k = सत्य;

        खण्डानि_अङ्कः.x = use_प्रवाहः_k ? nblocks_प्रवाहः_k : ntiles_योग;
        खण्डानि_अङ्कः.y = 1;
        खण्डानि_अङ्कः.z = 1;

        यदि (ntiles_योग % खण्डानि_अङ्कः.x != 0) { // Fixup is only needed if the SMs work on fractional tiles.
            dst_tmp_meta.alloc((size_t(खण्डानि_अङ्कः.x) * ncols * (2 + DV/2)));
        }
    } अन्यथा {
        ध्रुव पूर्णाङ्क ntiles_KQ = (K->ne[1] + nbatch_fa - 1) / nbatch_fa; // Max. number of parallel blocks limited by tensor size.

        // parallel_blocks must not be larger than what the tensor size allows:
        parallel_खण्डानि = std::min(parallel_खण्डानि, ntiles_KQ);
        // todo fix the hard code change
        // parallel_blocks = ntiles_KQ;

        // If ntiles_total % blocks_per_wave != 0 then some efficiency is lost due to tail effects.
        // Test whether parallel_blocks can be set to a higher value for better efficiency.
        ध्रुव पूर्णाङ्क खण्डानि_प्रत्येकम्_wave = nsm * max_खण्डानि_प्रत्येकम्_sm;
        पूर्णाङ्क nwaves_best = 0;
        पूर्णाङ्क efficiency_percent_best = 0;
        प्रति (पूर्णाङ्क parallel_खण्डानि_test = parallel_खण्डानि; parallel_खण्डानि_test <= ntiles_KQ; ++parallel_खण्डानि_test) {
            ध्रुव पूर्णाङ्क nblocks_योग = ntiles_योग * parallel_खण्डानि_test;
            ध्रुव पूर्णाङ्क nwaves = (nblocks_योग + खण्डानि_प्रत्येकम्_wave - 1) / खण्डानि_प्रत्येकम्_wave;
            ध्रुव पूर्णाङ्क efficiency_percent = 100 * nblocks_योग / (nwaves*खण्डानि_प्रत्येकम्_wave);

            // Stop trying configurations with more waves if we already have good efficiency to avoid excessive overhead.
            यदि (efficiency_percent_best >= 95 && nwaves > nwaves_best) {
                भङ्ग;
            }

            यदि (efficiency_percent > efficiency_percent_best) {
                nwaves_best = nwaves;
                efficiency_percent_best = efficiency_percent;
                parallel_खण्डानि = parallel_खण्डानि_test;
            }
        }

        खण्डानि_अङ्कः.x = ntiles_x;
        खण्डानि_अङ्कः.y = parallel_खण्डानि;
        खण्डानि_अङ्कः.z = ntiles_z_gqa*K->ne[2]*Q->ne[3];

        यदि (parallel_खण्डानि > 1) {
            dst_tmp.alloc(parallel_खण्डानि*ggml_nelements(KQV));
            dst_tmp_meta.alloc(parallel_खण्डानि*ggml_nrows(KQV));
        }
    }

    भिन्न scale         = 1.0f;
    भिन्न max_bias      = 0.0f;
    भिन्न logit_softcap = 0.0f;

    स्मृतिं_प्रतिलिख(&scale,         (ध्रुव भिन्न *) KQV->op_params + 0, परिमाण(भिन्न));
    स्मृतिं_प्रतिलिख(&max_bias,      (ध्रुव भिन्न *) KQV->op_params + 1, परिमाण(भिन्न));
    स्मृतिं_प्रतिलिख(&logit_softcap, (ध्रुव भिन्न *) KQV->op_params + 2, परिमाण(भिन्न));

    यदि (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    ध्रुव uint32_t n_शीर्ष      = Q->ne[2];
    ध्रुव uint32_t n_शीर्ष_log2 = 1u << uint32_t(floorf(log2f(भिन्न(n_शीर्ष))));

    ध्रुव भिन्न m0 = powf(2.0f, -(max_bias       ) / n_शीर्ष_log2);
    ध्रुव भिन्न m1 = powf(2.0f, -(max_bias / 2.0f) / n_शीर्ष_log2);

    // TODO other tensor dimensions after removal of WMMA kernel:
    ध्रुव sycl::uint3 ne01 = init_fastdiv_मानानि(Q->ne[1]);

    GGML_ASSERT(खण्डम्_dim.x % warp_आकार == 0);

    lauch_kernel<fattn_kernel, warp_आकार, use_large_grf>(
        खण्डानि_अङ्कः, खण्डम्_dim, main_प्रवाहः, (अचिह्नित पूर्णाङ्क) nbytes_shared, (ध्रुव वर्ण *) Q->दत्त, K_दत्त, V_दत्त,
        आवरणम् ? ((ध्रुव वर्ण *) आवरणम्->दत्त) : शून्यसूचकः, sinks ? ((ध्रुव वर्ण *) sinks->दत्त) : शून्यसूचकः, KV_max.ptr,
        !प्रवाहः_k && parallel_खण्डानि > 1 ? dst_tmp.ptr : (भिन्न *) KQV->दत्त, (sycl::float2 *)dst_tmp_meta.ptr, scale, max_bias, m0, m1,
        n_शीर्ष_log2, logit_softcap, Q->ne[0], ne01, Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3], K->ne[0],
        K->ne[1], K->ne[2], K->ne[3], nb11, nb12, nb13, nb21, nb22, nb23, आवरणम् ? आवरणम्->ne[1] : 0,
        आवरणम् ? आवरणम्->ne[2] : 0, आवरणम् ? आवरणम्->ne[3] : 0, आवरणम् ? आवरणम्->nb[1] : 0, आवरणम् ? आवरणम्->nb[2] : 0,
        आवरणम् ? आवरणम्->nb[3] : 0);
    SYCL_CHECK(0);

    यदि (प्रवाहः_k) {
        यदि (ntiles_योग % खण्डानि_अङ्कः.x != 0) { // Fixup is only needed if the SMs work on fractional tiles.
            ध्रुव dpct::dim3 खण्डम्_dim_combine(DV, 1, 1);
            ध्रुव dpct::dim3 खण्डानि_अङ्कः_combine = { खण्डानि_अङ्कः.x, ncols1, ncols2 };

            main_प्रवाहः->submit([&](sycl::handler & cgh) {
                स्वचालित KQV_दत्त_ct0         = (भिन्न *) KQV->दत्त;
                स्वचालित dst_tmp_meta_ptr_ct1 = dst_tmp_meta.ptr;
                स्वचालित Q_ne_ct2             = Q->ne[1];
                स्वचालित Q_ne_ct3             = Q->ne[2];
                स्वचालित Q_ne_ct4             = Q->ne[3];
                स्वचालित K_ne_ct5             = K->ne[1];
                स्वचालित K_ne_ct6             = K->ne[2];

                cgh.parallel_for(sycl::nd_range<3>(खण्डानि_अङ्कः_combine * खण्डम्_dim_combine, खण्डम्_dim_combine),
                                 [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_गणः_आकार(warp_आकार)]] {
                                     GGML_अप्रयुक्तम्(item_ct1);
                                     flash_attn_प्रवाहः_k_fixup<DV, ncols1, ncols2>(KQV_दत्त_ct0, dst_tmp_meta_ptr_ct1,
                                                                                   Q_ne_ct2, Q_ne_ct3, Q_ne_ct4,
                                                                                   K_ne_ct5, K_ne_ct6, nbatch_fa);
                                 });
            });
        }
    } अन्यथा यदि (parallel_खण्डानि > 1) {
        ध्रुव dpct::dim3 खण्डम्_dim_combine(DV, 1, 1);
        ध्रुव dpct::dim3 खण्डानि_अङ्कः_combine(Q->ne[1], Q->ne[2], Q->ne[3]);
        ध्रुव size_t     nbytes_shared_combine = parallel_खण्डानि * परिमाण(sycl::float2);
        main_प्रवाहः->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(sycl::range<1>(nbytes_shared_combine), cgh);

            स्वचालित dst_tmp_ptr_ct0      = dst_tmp.ptr;
            स्वचालित dst_tmp_meta_ptr_ct1 = dst_tmp_meta.ptr;
            स्वचालित KQV_दत्त_ct2         = (भिन्न *) KQV->दत्त;

            cgh.parallel_for(sycl::nd_range<3>(खण्डानि_अङ्कः_combine * खण्डम्_dim_combine, खण्डम्_dim_combine),
                             [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_गणः_आकार(warp_आकार)]] {
                                 GGML_अप्रयुक्तम्(item_ct1);
                                 flash_attn_combine_results<DV>(
                                     dst_tmp_ptr_ct0, dst_tmp_meta_ptr_ct1, KQV_दत्त_ct2, parallel_खण्डानि,
                                     dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                             });
        });
    }
    SYCL_CHECK(0);
}
