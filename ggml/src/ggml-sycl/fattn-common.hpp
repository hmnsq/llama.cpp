#సూచన once

#అన్తర్భావ <sycl/sycl.hpp>
#అన్తర్భావ <sycl/ext/intel/experimental/grf_size_properties.hpp>
#అన్తర్భావ "dpct/helper.hpp"
#అన్తర్భావ "common.hpp"
#అన్తర్భావ "convert.hpp"
#అన్తర్భావ "vecdotq.hpp"
#అన్తర్భావ "fattn.hpp"

#అన్తర్భావ "ggml.h"

#అన్తర్భావ <cstdint>
#అన్తర్భావ <cmath>
#అన్తర్భావ <float.h>


#నిర్వచించు FATTN_KQ_STRIDE       256
#నిర్వచించు HALF_MAX_HALF         sycl::half(65504.0f/2) // Use neg. of this instead of -INFINITY to initialize KQ max vals to avoid NaN upon subtraction.
#నిర్వచించు SOFTMAX_FTZ_పరిమితి -20.0f                   // Softmax exp. of values smaller than this are flushed to zero to avoid NaNs.
#నిర్వచించు FATTN_KQ_MAX_స్థానభ్రంశం (3.0f*0.6931f)

రకనిర్వచనం శూన్య (*fattn_kernel_t)(
    స్థిరరాశి అక్షరం* Q,
    స్థిరరాశి అక్షరం* K,
    స్థిరరాశి అక్షరం* V,
    స్థిరరాశి అక్షరం* ముసుగు,
    స్థిరరాశి అక్షరం* sinks,
    స్థిరరాశి పూర్ణాంకం* KV_max,
    దశాంశం* dst,
    sycl::float2* dst_meta,
    స్థిరరాశి దశాంశం scale,
    స్థిరరాశి దశాంశం max_bias,
    స్థిరరాశి దశాంశం m0,
    స్థిరరాశి దశాంశం m1,
    స్థిరరాశి uint32_t n_తల_log2,
    స్థిరరాశి దశాంశం logit_softcap,
    స్థిరరాశి int32_t ne00,
    స్థిరరాశి sycl::uint3 ne01,
    స్థిరరాశి int32_t ne02,
    స్థిరరాశి int32_t ne03,
    స్థిరరాశి int32_t nb01,
    స్థిరరాశి int32_t nb02,
    స్థిరరాశి int32_t nb03,
    స్థిరరాశి int32_t ne10,
    స్థిరరాశి int32_t ne11,
    స్థిరరాశి int32_t ne12,
    స్థిరరాశి int32_t ne13,
    స్థిరరాశి int32_t nb11,
    స్థిరరాశి int32_t nb12,
    స్థిరరాశి int64_t nb13,
    స్థిరరాశి int32_t nb21,
    స్థిరరాశి int32_t nb22,
    స్థిరరాశి int64_t nb23,
    స్థిరరాశి int32_t ne31,
    స్థిరరాశి int32_t ne32,
    స్థిరరాశి int32_t ne33,
    స్థిరరాశి int32_t nb31,
    స్థిరరాశి int32_t nb32,
    స్థిరరాశి int64_t nb33);

రకనిర్వచనం దశాంశం (*vec_dot_KQ_t)(
    స్థిరరాశి అక్షరం * __restrict__ K_c, స్థిరరాశి శూన్య * __restrict__ Q_v, స్థిరరాశి పూర్ణాంకం * __restrict__ Q_q8 , స్థిరరాశి శూన్య * __restrict__ Q_ds);

మూస <పూర్ణాంకం D, పూర్ణాంకం nthreads>
స్థిర __dpct_inline__ దశాంశం vec_dot_fattn_vec_KQ_f16(స్థిరరాశి అక్షరం * __restrict__ K_c,
                                                      స్థిరరాశి శూన్య * __restrict__ Q_v,
                                                      స్థిరరాశి పూర్ణాంకం * __restrict__ Q_q8,
                                                      స్థిరరాశి శూన్య * __restrict__ Q_ds_v) {
    స్థిరరాశి sycl::half2 * K_h2 = (స్థిరరాశి sycl::half2 *) K_c;
    GGML_వాడని(Q_q8);
    GGML_వాడని(Q_ds_v);

    constexpr పూర్ణాంకం cpy_nb = ggml_sycl_get_max_cpy_బైట్లు();
    constexpr పూర్ణాంకం cpy_ne = cpy_nb / 4;

    దశాంశం కూడిక = 0.0f;

#సూచన unroll
    కొరకు (పూర్ణాంకం k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads*cpy_ne) {
        sycl::half2 tmp[cpy_ne];
        ggml_sycl_memcpy_1<పరిమాణం(tmp)>(
            tmp,
            K_h2 + k_KQ_0 + (sycl::ext::oneapi::this_work_అంశం::get_nd_అంశం<3>().get_local_id(2) % nthreads) * cpy_ne);
#సూచన unroll
        కొరకు (పూర్ణాంకం k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
#నిర్వచితమైతే GGML_SYCL_F16
            ggml_sycl_mad(కూడిక,                tmp[k_KQ_1] , ((స్థిరరాశి sycl::half2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#లేకపోతే
            ggml_sycl_mad(కూడిక, __half22float2(tmp[k_KQ_1]), ((స్థిరరాశి sycl::float2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#అయితే_ముగింపు // GGML_SYCL_F16
        }
    }

    తిరిగి కూడిక;
}

మూస <పూర్ణాంకం D, పూర్ణాంకం nthreads, పూర్ణాంకం warp_కొలత>
స్థిర __dpct_inline__ దశాంశం vec_dot_fattn_vec_KQ_q4_0(స్థిరరాశి అక్షరం * __restrict__ K_c,
                                                       స్థిరరాశి శూన్య * __restrict__ Q_v,
                                                       స్థిరరాశి పూర్ణాంకం * __restrict__ Q_q8,
                                                       స్థిరరాశి శూన్య * __restrict__ Q_ds_v) {
    స్వయంచాలక అంశం_ct1 = sycl::ext::oneapi::this_work_అంశం::get_nd_అంశం<3>();

    స్థిరరాశి ఖండం_q4_0 * K_q4_0   = (స్థిరరాశి ఖండం_q4_0 *) K_c;
    GGML_వాడని(Q_v);

    దశాంశం కూడిక = 0.0f;

#సూచన unroll
    కొరకు (పూర్ణాంకం k_KQ_0 = 0; k_KQ_0 < పూర్ణాంకం(D/పరిమాణం(పూర్ణాంకం)); k_KQ_0 += nthreads) {
        స్థిరరాశి పూర్ణాంకం k_KQ =
            k_KQ_0 + (nthreads == warp_కొలత ? అంశం_ct1.get_local_id(2) : అంశం_ct1.get_local_id(2) % nthreads);

        స్థిరరాశి పూర్ణాంకం ib    = k_KQ /  QI8_1;
        స్థిరరాశి పూర్ణాంకం iqs4  = k_KQ %  QI4_0;
        స్థిరరాశి పూర్ణాంకం జరుపు = k_KQ & (QI8_1/2);

        పూర్ణాంకం v;
        ggml_sycl_memcpy_1<పరిమాణం(పూర్ణాంకం), 2>(&v, K_q4_0[ib].qs + పరిమాణం(పూర్ణాంకం)*iqs4);
        v = (v >> జరుపు) & 0x0F0F0F0F;
        స్థిరరాశి పూర్ణాంకం u = Q_q8[k_KQ_0/nthreads];

        స్థిరరాశి పూర్ణాంకం sumi = ggml_sycl_dp4a(v, u, 0);

        స్థిరరాశి sycl::float2 Q_ds = ((స్థిరరాశి sycl::float2 *) Q_ds_v)[k_KQ_0 / nthreads];
        కూడిక += __half2float(K_q4_0[ib].d) * (sumi*Q_ds.x() - (8/QI8_1)*Q_ds.y());
    }

    తిరిగి కూడిక;
}

మూస <పూర్ణాంకం D, పూర్ణాంకం nthreads , పూర్ణాంకం warp_కొలత>
స్థిర __dpct_inline__ దశాంశం vec_dot_fattn_vec_KQ_q4_1(స్థిరరాశి అక్షరం * __restrict__ K_c,
                                                       స్థిరరాశి శూన్య * __restrict__ Q_v,
                                                       స్థిరరాశి పూర్ణాంకం * __restrict__ Q_q8,
                                                       స్థిరరాశి శూన్య * __restrict__ Q_ds_v) {
    స్వయంచాలక               అంశం_ct1 = sycl::ext::oneapi::this_work_అంశం::get_nd_అంశం<3>();
    స్థిరరాశి ఖండం_q4_1 * K_q4_1   = (స్థిరరాశి ఖండం_q4_1 *) K_c;
    GGML_వాడని(Q_v);

    దశాంశం కూడిక = 0.0f;

#సూచన unroll
    కొరకు (పూర్ణాంకం k_KQ_0 = 0; k_KQ_0 < పూర్ణాంకం(D/పరిమాణం(పూర్ణాంకం)); k_KQ_0 += nthreads) {
        స్థిరరాశి పూర్ణాంకం k_KQ =
            k_KQ_0 + (nthreads == warp_కొలత ? అంశం_ct1.get_local_id(2) : అంశం_ct1.get_local_id(2) % nthreads);

        స్థిరరాశి పూర్ణాంకం ib    = k_KQ /  QI8_1;
        స్థిరరాశి పూర్ణాంకం iqs4  = k_KQ %  QI4_1;
        స్థిరరాశి పూర్ణాంకం జరుపు = k_KQ & (QI8_1/2);

        పూర్ణాంకం v;
        ggml_sycl_memcpy_1<పరిమాణం(పూర్ణాంకం)>(&v, K_q4_1[ib].qs + పరిమాణం(పూర్ణాంకం)*iqs4);
        v = (v >> జరుపు) & 0x0F0F0F0F;
        స్థిరరాశి పూర్ణాంకం u = Q_q8[k_KQ_0/nthreads];

        స్థిరరాశి పూర్ణాంకం sumi = ggml_sycl_dp4a(v, u, 0);

        స్థిరరాశి sycl::float2 K_dm = (K_q4_1[ib].dm).మూస convert<దశాంశం, sycl::rounding_బహుళకం::automatic>();
        స్థిరరాశి sycl::float2 Q_ds = ((స్థిరరాశి sycl::float2 *) Q_ds_v)[k_KQ_0 / nthreads];

        కూడిక += K_dm.x()*Q_ds.x()*sumi + K_dm.y()*Q_ds.y()/QI8_1;
    }

    తిరిగి కూడిక;
}

మూస <పూర్ణాంకం D, పూర్ణాంకం nthreads, పూర్ణాంకం warp_కొలత>
స్థిర __dpct_inline__ దశాంశం vec_dot_fattn_vec_KQ_q5_0(స్థిరరాశి అక్షరం * __restrict__ K_c,
                                                       స్థిరరాశి శూన్య * __restrict__ Q_v,
                                                       స్థిరరాశి పూర్ణాంకం * __restrict__ Q_q8,
                                                       స్థిరరాశి శూన్య * __restrict__ Q_ds_v) {
    స్వయంచాలక               అంశం_ct1 = sycl::ext::oneapi::this_work_అంశం::get_nd_అంశం<3>();
    స్థిరరాశి ఖండం_q5_0 * K_q5_0   = (స్థిరరాశి ఖండం_q5_0 *) K_c;
    GGML_వాడని(Q_v);

    దశాంశం కూడిక = 0.0f;

#సూచన unroll
    కొరకు (పూర్ణాంకం k_KQ_0 = 0; k_KQ_0 < పూర్ణాంకం(D/పరిమాణం(పూర్ణాంకం)); k_KQ_0 += nthreads) {
        స్థిరరాశి పూర్ణాంకం k_KQ =
            k_KQ_0 + (nthreads == warp_కొలత ? అంశం_ct1.get_local_id(2) : అంశం_ct1.get_local_id(2) % nthreads);

        స్థిరరాశి పూర్ణాంకం ib    = k_KQ /  QI8_1;
        స్థిరరాశి పూర్ణాంకం iqs4  = k_KQ %  QI5_0;
        స్థిరరాశి పూర్ణాంకం iqs8  = k_KQ %  QI8_1;
        స్థిరరాశి పూర్ణాంకం జరుపు = k_KQ & (QI8_1/2);

        పూర్ణాంకం v;
        ggml_sycl_memcpy_1<పరిమాణం(పూర్ణాంకం), 2>(&v, K_q5_0[ib].qs + పరిమాణం(పూర్ణాంకం)*iqs4);
        v = (v >> జరుపు) & 0x0F0F0F0F;

        {
            పూర్ణాంకం vh;
            ggml_sycl_memcpy_1<పరిమాణం(పూర్ణాంకం), 2>(&vh, K_q5_0[ib].qh);
            vh >>= iqs8 * QI5_0;

            v |= (vh <<  4) & 0x00000010; // 0 ->  4
            v |= (vh << 11) & 0x00001000; // 1 -> 12
            v |= (vh << 18) & 0x00100000; // 2 -> 20
            v |= (vh << 25) & 0x10000000; // 3 -> 28
        }

        స్థిరరాశి పూర్ణాంకం u = Q_q8[k_KQ_0/nthreads];

        స్థిరరాశి పూర్ణాంకం sumi = ggml_sycl_dp4a(v, u, 0);

        స్థిరరాశి sycl::float2 Q_ds = ((స్థిరరాశి sycl::float2 *) Q_ds_v)[k_KQ_0 / nthreads];

        కూడిక += __half2float(K_q5_0[ib].d) * (sumi*Q_ds.x() - (16/QI8_1)*Q_ds.y());
    }

    తిరిగి కూడిక;
}

మూస <పూర్ణాంకం D, పూర్ణాంకం nthreads, పూర్ణాంకం warp_కొలత>
స్థిర __dpct_inline__ దశాంశం vec_dot_fattn_vec_KQ_q5_1(స్థిరరాశి అక్షరం * __restrict__ K_c,
                                                       స్థిరరాశి శూన్య * __restrict__ Q_v,
                                                       స్థిరరాశి పూర్ణాంకం * __restrict__ Q_q8,
                                                       స్థిరరాశి శూన్య * __restrict__ Q_ds_v) {
    స్వయంచాలక               అంశం_ct1 = sycl::ext::oneapi::this_work_అంశం::get_nd_అంశం<3>();
    స్థిరరాశి ఖండం_q5_1 * K_q5_1   = (స్థిరరాశి ఖండం_q5_1 *) K_c;
    GGML_వాడని(Q_v);

    దశాంశం కూడిక = 0.0f;

#సూచన unroll
    కొరకు (పూర్ణాంకం k_KQ_0 = 0; k_KQ_0 < పూర్ణాంకం(D/పరిమాణం(పూర్ణాంకం)); k_KQ_0 += nthreads) {
        స్థిరరాశి పూర్ణాంకం k_KQ =
            k_KQ_0 + (nthreads == warp_కొలత ? అంశం_ct1.get_local_id(2) : అంశం_ct1.get_local_id(2) % nthreads);

        స్థిరరాశి పూర్ణాంకం ib    = k_KQ /  QI8_1;
        స్థిరరాశి పూర్ణాంకం iqs4  = k_KQ %  QI5_1;
        స్థిరరాశి పూర్ణాంకం iqs8  = k_KQ %  QI8_1;
        స్థిరరాశి పూర్ణాంకం జరుపు = k_KQ & (QI8_1/2);

        పూర్ణాంకం v;
        ggml_sycl_memcpy_1<పరిమాణం(పూర్ణాంకం)>(&v, K_q5_1[ib].qs + పరిమాణం(పూర్ణాంకం)*iqs4);
        v = (v >> జరుపు) & 0x0F0F0F0F;

        {
            పూర్ణాంకం vh;
            ggml_sycl_memcpy_1<పరిమాణం(పూర్ణాంకం)>(&vh, K_q5_1[ib].qh);
            vh >>= iqs8 * QI5_0;

            v |= (vh <<  4) & 0x00000010; // 0 ->  4
            v |= (vh << 11) & 0x00001000; // 1 -> 12
            v |= (vh << 18) & 0x00100000; // 2 -> 20
            v |= (vh << 25) & 0x10000000; // 3 -> 28
        }

        స్థిరరాశి పూర్ణాంకం u = Q_q8[k_KQ_0/nthreads];

        స్థిరరాశి పూర్ణాంకం sumi = ggml_sycl_dp4a(v, u, 0);

        స్థిరరాశి sycl::float2 K_dm = (K_q5_1[ib].dm).మూస convert<దశాంశం, sycl::rounding_బహుళకం::automatic>();
        స్థిరరాశి sycl::float2 Q_ds = ((స్థిరరాశి sycl::float2 *) Q_ds_v)[k_KQ_0 / nthreads];

        కూడిక += K_dm.x()*Q_ds.x()*sumi + K_dm.y()*Q_ds.y()/QI8_1;
    }

    తిరిగి కూడిక;
}

మూస <పూర్ణాంకం D, పూర్ణాంకం nthreads, పూర్ణాంకం warp_కొలత>
స్థిర __dpct_inline__ దశాంశం vec_dot_fattn_vec_KQ_q8_0(స్థిరరాశి అక్షరం * __restrict__ K_c,
                                                       స్థిరరాశి శూన్య * __restrict__ Q_v,
                                                       స్థిరరాశి పూర్ణాంకం * __restrict__ Q_q8,
                                                       స్థిరరాశి శూన్య * __restrict__ Q_ds_v) {
    స్వయంచాలక               అంశం_ct1 = sycl::ext::oneapi::this_work_అంశం::get_nd_అంశం<3>();
    స్థిరరాశి ఖండం_q8_0 * K_q8_0   = (స్థిరరాశి ఖండం_q8_0 *) K_c;
    GGML_వాడని(Q_v);

    దశాంశం కూడిక = 0.0f;

#సూచన unroll
    కొరకు (పూర్ణాంకం k_KQ_0 = 0; k_KQ_0 < పూర్ణాంకం(D/పరిమాణం(పూర్ణాంకం)); k_KQ_0 += nthreads) {
        స్థిరరాశి పూర్ణాంకం k_KQ =
            k_KQ_0 + (nthreads == warp_కొలత ? అంశం_ct1.get_local_id(2) : అంశం_ct1.get_local_id(2) % nthreads);

        స్థిరరాశి పూర్ణాంకం ib  = k_KQ / QI8_0;
        స్థిరరాశి పూర్ణాంకం iqs = k_KQ % QI8_0;

        పూర్ణాంకం v;
        ggml_sycl_memcpy_1<పరిమాణం(v), 2>(&v, K_q8_0[ib].qs + 4*iqs);

        స్థిరరాశి sycl::float2 * Q_ds = (స్థిరరాశి sycl::float2 *) Q_ds_v;
        స్థిరరాశి దశాంశం          Q_d  = Q_ds[k_KQ_0 / nthreads].x();

        కూడిక += vec_dot_q8_0_q8_1_impl<దశాంశం, 1>(&v, &Q_q8[k_KQ_0/nthreads], K_q8_0[ib].d, Q_d);
    }

    తిరిగి కూడిక;
}

మూస <రకం_పేరు Tds, పూర్ణాంకం ni, పూర్ణాంకం warp_కొలత>
స్థిర __dpct_inline__ శూన్య quantize_q8_1_గమ్యం_shared(స్థిరరాశి దశాంశం * __restrict__ x,
                                                    స్థిరరాశి దశాంశం scale,
                                                    పూర్ణాంకం * __restrict__ yq32,
                                                    శూన్య * __restrict__ yds) {
    స్వయంచాలక అంశం_ct1 = sycl::ext::oneapi::this_work_అంశం::get_nd_అంశం<3>();

    దశాంశం vals[పరిమాణం(పూర్ణాంకం)] = { 0.0f };
#సూచన unroll
    కొరకు (పూర్ణాంకం l = 0; l < పూర్ణాంకం(పరిమాణం(పూర్ణాంకం)); ++l) {
        vals[l] =
            (ni == warp_కొలత || అంశం_ct1.get_local_id(2) < ni) ? scale * x[4 * అంశం_ct1.get_local_id(2) + l] : 0.0f;
    }

    దశాంశం amax = sycl::fabs(vals[0]);
    దశాంశం కూడిక  = vals[0];
#సూచన unroll
    కొరకు (పూర్ణాంకం l = 1; l < పూర్ణాంకం(పరిమాణం(పూర్ణాంకం)); ++l) {
        amax = sycl::fmax(amax, sycl::fabs(vals[l]));
        కూడిక += vals[l];
    }
#సూచన unroll
    కొరకు (పూర్ణాంకం ముసుగు = QI8_1/2; ముసుగు > 0; ముసుగు >>= 1) {
        amax = sycl::fmax(
            amax, dpct::క్రమమార్పు_sub_సమూహం_by_xor(sycl::ext::oneapi::this_work_అంశం::get_sub_సమూహం(), amax, ముసుగు));
        కూడిక += dpct::క్రమమార్పు_sub_సమూహం_by_xor(sycl::ext::oneapi::this_work_అంశం::get_sub_సమూహం(), కూడిక, ముసుగు);
    }

    స్థిరరాశి దశాంశం d = amax / 127;
    పూర్ణాంకం q32 = 0;
    int8_t * q8 = (int8_t *) &q32;

    ఒకవేళ (d != 0.0f) {
#సూచన unroll
        కొరకు (పూర్ణాంకం l = 0; l < పూర్ణాంకం(పరిమాణం(పూర్ణాంకం)); ++l) {
            q8[l] = sycl::round(vals[l] / d);
        }
    }

    yq32[అంశం_ct1.get_local_id(2)] = q32;
    ఒకవేళ (అంశం_ct1.get_local_id(2) % QI8_1 == 0 && (ni == warp_కొలత || అంశం_ct1.get_local_id(2) < ni)) {
        ఒకవేళ (std::is_same<Tds, sycl::half2>::value) {
            ((sycl::half2  *) yds)[అంశం_ct1.get_local_id(2)/QI8_1] =  make_half2(d, కూడిక);
        } లేకపోతే {
            ((sycl::float2 *) yds)[అంశం_ct1.get_local_id(2)/QI8_1] = make_float2(d, కూడిక);
        }
    }
}

రకనిర్వచనం శూన్య (*dequantize_V_t)(స్థిరరాశి శూన్య *, శూన్య *, స్థిరరాశి int64_t);

మూస <రకం_పేరు T, పూర్ణాంకం ne>
స్థిర __dpct_inline__ శూన్య dequantize_V_f16(స్థిరరాశి శూన్య * __restrict__ vx, శూన్య * __restrict__ dst, స్థిరరాశి int64_t i0) {
    ఒకవేళ constexpr (std::is_same_v<T, sycl::half>) {
        ggml_sycl_memcpy_1<ne * పరిమాణం(sycl::half)>(dst, (స్థిరరాశి sycl::half *) vx + i0);
    } లేకపోతే ఒకవేళ constexpr (std::is_same_v<T, దశాంశం>) {
        static_assert(ne % 2 == 0, "bad ne");
        sycl::half2 tmp[ne / 2];
        ggml_sycl_memcpy_1<ne * పరిమాణం(sycl::half)>(tmp, (స్థిరరాశి sycl::half *) vx + i0);
        sycl::float2 * dst_f2 = (sycl::float2 *) dst;
#సూచన unroll
        కొరకు (పూర్ణాంకం l = 0; l < ne/2; ++l) {
            dst_f2[l] = tmp[l].మూస convert<దశాంశం, sycl::rounding_బహుళకం::automatic>();
        }
    } లేకపోతే {
        static_assert(std::is_same_v<T, శూన్య>, "unsupported type");
    }
}

మూస <రకం_పేరు T, పూర్ణాంకం ne>
స్థిర __dpct_inline__ శూన్య dequantize_V_q4_0(స్థిరరాశి శూన్య * __restrict__ vx, శూన్య * __restrict__ dst, స్థిరరాశి int64_t i0) {
    స్థిరరాశి ఖండం_q4_0 * x = (స్థిరరాశి ఖండం_q4_0 *) vx;

    స్థిరరాశి int64_t ib    =  i0          /  QK4_0;
    స్థిరరాశి పూర్ణాంకం     iqs   =  i0          % (QK4_0/2);
    స్థిరరాశి పూర్ణాంకం     జరుపు = (i0 % QK4_0) / (QK4_0/2);

    పూర్ణాంకం q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_sycl_memcpy_1<ne, 2>(&q, x[ib].qs + iqs);
    q >>= 4*జరుపు;
    q &= 0x0F0F0F0F;
    q = dpct::vectorized_ద్విమాన<sycl::char4>(q, 0x08080808, dpct::sub_sat());

    స్థిరరాశి int8_t * q8 = (స్థిరరాశి int8_t *) &q;

#నిర్వచితమైతే GGML_SYCL_F16
    ఒకవేళ constexpr (std::is_same_v<T, sycl::half>) {
        స్థిరరాశి sycl::half2 d = sycl::half2(x[ib].d);

#సూచన unroll
        కొరకు (పూర్ణాంకం l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * sycl::half2(q8[l0 + 0], q8[l0 + 1]);
        }
    } లేకపోతే
#అయితే_ముగింపు // GGML_SYCL_F16
    ఒకవేళ constexpr (std::is_same_v<T, దశాంశం>) {
        స్థిరరాశి దశాంశం d = x[ib].d;

#సూచన unroll
        కొరకు (పూర్ణాంకం l = 0; l < ne; ++l) {
            ((దశాంశం *) dst)[l] = d * q8[l];
        }
    } లేకపోతే {
        static_assert(std::is_same_v<T, శూన్య>, "bad type");
    }
}

మూస <రకం_పేరు T, పూర్ణాంకం ne>
స్థిర __dpct_inline__ శూన్య dequantize_V_q4_1(స్థిరరాశి శూన్య * __restrict__ vx, శూన్య * __restrict__ dst, స్థిరరాశి int64_t i0) {
    స్థిరరాశి ఖండం_q4_1 * x = (స్థిరరాశి ఖండం_q4_1 *) vx;

    స్థిరరాశి int64_t ib    =  i0          /  QK4_1;
    స్థిరరాశి పూర్ణాంకం     iqs   =  i0          % (QK4_1/2);
    స్థిరరాశి పూర్ణాంకం     జరుపు = (i0 % QK4_1) / (QK4_1/2);

    పూర్ణాంకం q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_sycl_memcpy_1<ne>(&q, x[ib].qs + iqs);
    q >>= 4*జరుపు;
    q &= 0x0F0F0F0F;

    స్థిరరాశి int8_t * q8 = (స్థిరరాశి int8_t *) &q;

#నిర్వచితమైతే GGML_SYCL_F16
    ఒకవేళ constexpr (std::is_same_v<T, sycl::half>) {
        స్థిరరాశి sycl::half2 dm = x[ib].dm;
        స్థిరరాశి sycl::half2 d  = sycl::half2(dm[0]);
        స్థిరరాశి sycl::half2 m  = sycl::half2(dm[1]);

#సూచన unroll
        కొరకు (పూర్ణాంకం l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * sycl::half2(q8[l0 + 0], q8[l0 + 1]) + m;
        }
    } లేకపోతే
#అయితే_ముగింపు // GGML_SYCL_F16
    ఒకవేళ constexpr (std::is_same_v<T, దశాంశం>) {
        స్థిరరాశి sycl::float2 dm = (x[ib].dm).మూస convert<దశాంశం, sycl::rounding_బహుళకం::automatic>();

#సూచన unroll
        కొరకు (పూర్ణాంకం l = 0; l < ne; ++l) {
            ((దశాంశం *) dst)[l] = dm.x() * q8[l] + dm.y();
        }
    } లేకపోతే {
        static_assert(std::is_same_v<T, శూన్య>, "bad type");
    }
}

మూస <రకం_పేరు T, పూర్ణాంకం ne>
స్థిర __dpct_inline__ శూన్య dequantize_V_q5_0(స్థిరరాశి శూన్య * __restrict__ vx, శూన్య * __restrict__ dst, స్థిరరాశి int64_t i0) {
    స్థిరరాశి ఖండం_q5_0 * x = (స్థిరరాశి ఖండం_q5_0 *) vx;

    స్థిరరాశి int64_t ib    =  i0          /  QK5_0;
    స్థిరరాశి పూర్ణాంకం     idq   =  i0          %  QK5_0;
    స్థిరరాశి పూర్ణాంకం     iqs   =  i0          % (QK5_0/2);
    స్థిరరాశి పూర్ణాంకం     జరుపు = (i0 % QK5_0) / (QK5_0/2);

    పూర్ణాంకం q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_sycl_memcpy_1<ne, 2>(&q, x[ib].qs + iqs);
    q >>= 4*జరుపు;
    q &= 0x0F0F0F0F;

    {
        పూర్ణాంకం qh;
        ggml_sycl_memcpy_1<ne, 2>(&qh, x[ib].qh);
#సూచన unroll
        కొరకు (పూర్ణాంకం l = 0; l < ne; ++l) {
            q |= ((qh >> (idq + l)) & 0x00000001) << (8*l + 4);
        }
    }

    q = dpct::vectorized_ద్విమాన<sycl::char4>(q, 0x10101010, dpct::sub_sat());

    స్థిరరాశి int8_t * q8 = (స్థిరరాశి int8_t *) &q;

#నిర్వచితమైతే GGML_SYCL_F16
    ఒకవేళ constexpr (std::is_same_v<T, sycl::half>) {
        స్థిరరాశి sycl::half2 d = sycl::half2(x[ib].d);

#సూచన unroll
        కొరకు (పూర్ణాంకం l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * sycl::half2(q8[l0 + 0], q8[l0 + 1]);
        }
    } లేకపోతే
#అయితే_ముగింపు // GGML_SYCL_F16
    ఒకవేళ constexpr (std::is_same_v<T, దశాంశం>) {
        స్థిరరాశి దశాంశం d = x[ib].d;

#సూచన unroll
        కొరకు (పూర్ణాంకం l = 0; l < ne; ++l) {
            ((దశాంశం *) dst)[l] = d * q8[l];
        }
    } లేకపోతే {
        static_assert(std::is_same_v<T, శూన్య>, "bad type");
    }
}

మూస <రకం_పేరు T, పూర్ణాంకం ne>
స్థిర __dpct_inline__ శూన్య dequantize_V_q5_1(స్థిరరాశి శూన్య * __restrict__ vx, శూన్య * __restrict__ dst, స్థిరరాశి int64_t i0) {
    స్థిరరాశి ఖండం_q5_1 * x = (స్థిరరాశి ఖండం_q5_1 *) vx;

    స్థిరరాశి int64_t ib    =  i0          /  QK5_1;
    స్థిరరాశి పూర్ణాంకం     idq   =  i0          %  QK5_1;
    స్థిరరాశి పూర్ణాంకం     iqs   =  i0          % (QK5_1/2);
    స్థిరరాశి పూర్ణాంకం     జరుపు = (i0 % QK5_1) / (QK5_1/2);

    పూర్ణాంకం q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_sycl_memcpy_1<ne>(&q, x[ib].qs + iqs);
    q >>= 4*జరుపు;
    q &= 0x0F0F0F0F;

    {
        పూర్ణాంకం qh;
        ggml_sycl_memcpy_1<ne>(&qh, x[ib].qh);
#సూచన unroll
        కొరకు (పూర్ణాంకం l = 0; l < ne; ++l) {
            q |= ((qh >> (idq + l)) & 0x00000001) << (8*l + 4);
        }
    }

    స్థిరరాశి int8_t * q8 = (స్థిరరాశి int8_t *) &q;

#నిర్వచితమైతే GGML_SYCL_F16
    ఒకవేళ constexpr (std::is_same_v<T, sycl::half>) {
        స్థిరరాశి sycl::half2 dm = x[ib].dm;
        స్థిరరాశి sycl::half2 d  = sycl::half2(dm[0]);
        స్థిరరాశి sycl::half2 m  = sycl::half2(dm[1]);

#సూచన unroll
        కొరకు (పూర్ణాంకం l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * sycl::half2(q8[l0 + 0], q8[l0 + 1]) + m;
        }
    } లేకపోతే
#అయితే_ముగింపు // GGML_SYCL_F16
    ఒకవేళ constexpr (std::is_same_v<T, దశాంశం>) {
        స్థిరరాశి sycl::float2 dm = (x[ib].dm).మూస convert<దశాంశం, sycl::rounding_బహుళకం::automatic>();

#సూచన unroll
        కొరకు (పూర్ణాంకం l = 0; l < ne; ++l) {
            ((దశాంశం *) dst)[l] = dm.x() * q8[l] + dm.y();
        }
    } లేకపోతే {
        static_assert(std::is_same_v<T, శూన్య>, "bad type");
    }
}

మూస <రకం_పేరు T, పూర్ణాంకం ne>
స్థిర __dpct_inline__ శూన్య dequantize_V_q8_0(స్థిరరాశి శూన్య * __restrict__ vx, శూన్య * __restrict__ dst, స్థిరరాశి int64_t i0) {
    స్థిరరాశి ఖండం_q8_0 * x = (స్థిరరాశి ఖండం_q8_0 *) vx;

    స్థిరరాశి int64_t ib  = i0 / QK8_0;
    స్థిరరాశి పూర్ణాంకం     iqs = i0 % QK8_0;

    static_assert(ne % 2 == 0, "bad ne");
    int8_t qs[ne];
    ggml_sycl_memcpy_1<ne, 2>(qs, x[ib].qs + iqs);

#నిర్వచితమైతే GGML_SYCL_F16
    ఒకవేళ constexpr (std::is_same<T, sycl::half>::value) {
        స్థిరరాశి sycl::half2 d = sycl::half2(x[ib].d);

#సూచన unroll
        కొరకు (పూర్ణాంకం l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * make_half2(qs[l0 + 0], qs[l0 + 1]);
        }
    } లేకపోతే
#అయితే_ముగింపు // GGML_SYCL_F16
    ఒకవేళ constexpr (std::is_same<T, దశాంశం>::value) {
        స్థిరరాశి దశాంశం d = x[ib].d;

#సూచన unroll
        కొరకు (పూర్ణాంకం l = 0; l < ne; ++l) {
            ((దశాంశం *) dst)[l] = d * qs[l];
        }
    } లేకపోతే {
        static_assert(std::is_same_v<T, శూన్య>, "unsupported type");
    }
}

మూస <పూర్ణాంకం type_K, పూర్ణాంకం D, పూర్ణాంకం nthreads, పూర్ణాంకం warp_కొలత>
constexpr vec_dot_KQ_t get_vec_dot_KQ() {
    ఒకవేళ constexpr (type_K == GGML_TYPE_F16) {
        తిరిగి vec_dot_fattn_vec_KQ_f16<D, nthreads>;
    } లేకపోతే ఒకవేళ constexpr (type_K == GGML_TYPE_Q4_0) {
        తిరిగి vec_dot_fattn_vec_KQ_q4_0<D, nthreads, warp_కొలత>;
    } లేకపోతే ఒకవేళ constexpr (type_K == GGML_TYPE_Q4_1) {
        తిరిగి vec_dot_fattn_vec_KQ_q4_1<D, nthreads, warp_కొలత>;
    } లేకపోతే ఒకవేళ constexpr (type_K == GGML_TYPE_Q5_0) {
        తిరిగి vec_dot_fattn_vec_KQ_q5_0<D, nthreads, warp_కొలత>;
    } లేకపోతే ఒకవేళ constexpr (type_K == GGML_TYPE_Q5_1) {
        తిరిగి vec_dot_fattn_vec_KQ_q5_1<D, nthreads, warp_కొలత>;
    } లేకపోతే ఒకవేళ constexpr (type_K == GGML_TYPE_Q8_0) {
        తిరిగి vec_dot_fattn_vec_KQ_q8_0<D, nthreads, warp_కొలత>;
    } లేకపోతే {
        static_assert(type_K == -1, "bad type");
        తిరిగి శూన్యసూచిక;
    }
}

మూస <పూర్ణాంకం type_V, రకం_పేరు T, పూర్ణాంకం ne>
constexpr dequantize_V_t get_dequantize_V() {
    ఒకవేళ constexpr (type_V == GGML_TYPE_F16) {
        తిరిగి dequantize_V_f16<T, ne>;
    } లేకపోతే ఒకవేళ constexpr (type_V == GGML_TYPE_Q4_0) {
        తిరిగి dequantize_V_q4_0<T, ne>;
    } లేకపోతే ఒకవేళ constexpr (type_V == GGML_TYPE_Q4_1) {
        తిరిగి dequantize_V_q4_1<T, ne>;
    } లేకపోతే ఒకవేళ constexpr (type_V == GGML_TYPE_Q5_0) {
        తిరిగి dequantize_V_q5_0<T, ne>;
    } లేకపోతే ఒకవేళ constexpr (type_V == GGML_TYPE_Q5_1) {
        తిరిగి dequantize_V_q5_1<T, ne>;
    } లేకపోతే ఒకవేళ constexpr (type_V == GGML_TYPE_Q8_0) {
        తిరిగి dequantize_V_q8_0<T, ne>;
    } లేకపోతే {
        static_assert(type_V == -1, "bad type");
        తిరిగి శూన్యసూచిక;
    }
}

మూస <పూర్ణాంకం ncols1, పూర్ణాంకం warp_కొలత>
స్థిర శూన్య flash_attn_ముసుగు_గమ్యం_KV_max(స్థిరరాశి sycl::half2 * __restrict__ ముసుగు,
                                      పూర్ణాంకం * __restrict__ KV_max,
                                      స్థిరరాశి పూర్ణాంకం ne30,
                                      స్థిరరాశి పూర్ణాంకం s31,
                                      స్థిరరాశి పూర్ణాంకం s33,
                                      పూర్ణాంకం *     buf_iw) {
    స్వయంచాలక      అంశం_ct1 = sycl::ext::oneapi::this_work_అంశం::get_nd_అంశం<3>();
    స్థిరరాశి పూర్ణాంకం ne31     = అంశం_ct1.get_సమూహం_range(2);
    స్థిరరాశి పూర్ణాంకం tid      = అంశం_ct1.get_local_id(2);
    స్థిరరాశి పూర్ణాంకం sequence = అంశం_ct1.get_సమూహం(1);
    స్థిరరాశి పూర్ణాంకం jt       = అంశం_ct1.get_సమూహం(2);

    ముసుగు += sequence*s33 + jt*ncols1*s31;

    ఒకవేళ (tid < warp_కొలత) {
        buf_iw[tid] = 1;
    }
    అంశం_ct1.barrier(sycl::access::fence_space::local_space);

    పూర్ణాంకం KV_max_sj = (ne30 - 1) * FATTN_KQ_STRIDE;
    కొరకు (; KV_max_sj >= 0; KV_max_sj -= FATTN_KQ_STRIDE) {
        పూర్ణాంకం all_inf = 1;

#సూచన unroll
        కొరకు (పూర్ణాంకం j = 0; j < ncols1; ++j) {
            స్థిరరాశి sycl::float2 tmp =
                ముసుగు[j * s31 + KV_max_sj / 2 + tid].మూస convert<దశాంశం, sycl::rounding_బహుళకం::automatic>();
            all_inf = all_inf && పూర్ణాంకం(sycl::isinf((దశాంశం) (tmp.x()))) && పూర్ణాంకం(sycl::isinf((దశాంశం) (tmp.y())));
        }

        all_inf = warp_reduce_all<warp_కొలత>(all_inf);
        ఒకవేళ (tid % warp_కొలత == 0) {
            buf_iw[tid / warp_కొలత] = all_inf;
        }
        అంశం_ct1.barrier(sycl::access::fence_space::local_space);
        all_inf = buf_iw[tid % warp_కొలత];
        అంశం_ct1.barrier(sycl::access::fence_space::local_space);
        all_inf = warp_reduce_all<warp_కొలత>(all_inf);

        ఒకవేళ (!all_inf) {
            ఆపు;
        }
    }

    // If the break in the loop was not triggered, KV_max_sj is now -FATTN_KQ_STRIDE.
    // If the break was triggered it's the lower edge of the tile with the first non-masked values.
    // In either case, walk back the decrementation by FATTN_KQ_STRIDE.
    KV_max_sj += FATTN_KQ_STRIDE;

    ఒకవేళ (అంశం_ct1.get_local_id(2) != 0) {
        తిరిగి;
    }

    KV_max[sequence*ne31 + jt] = KV_max_sj;
}

మూస <పూర్ణాంకం D, పూర్ణాంకం ncols1, పూర్ణాంకం ncols2>  // D == head size

స్థిర శూన్య flash_attn_ప్రవాహం_k_fixup(దశాంశం * __restrict__ dst,
                                      స్థిరరాశి sycl::float2 * __restrict__ dst_fixup,
                                      స్థిరరాశి పూర్ణాంకం ne01,
                                      స్థిరరాశి పూర్ణాంకం ne02,
                                      స్థిరరాశి పూర్ణాంకం ne03,
                                      స్థిరరాశి పూర్ణాంకం ne11,
                                      స్థిరరాశి పూర్ణాంకం ne12,
                                      స్థిరరాశి పూర్ణాంకం nbatch_fa) {
    స్వయంచాలక          అంశం_ct1 = sycl::ext::oneapi::this_work_అంశం::get_nd_అంశం<3>();
    constexpr పూర్ణాంకం ncols    = ncols1 * ncols2;

    స్థిరరాశి పూర్ణాంకం bidx0 = అంశం_ct1.get_సమూహం(2);
    స్థిరరాశి పూర్ణాంకం j     = అంశం_ct1.get_సమూహం(1);
    స్థిరరాశి పూర్ణాంకం c     = అంశం_ct1.get_సమూహం(0);
    స్థిరరాశి పూర్ణాంకం jc    = j*ncols2 + c;
    స్థిరరాశి పూర్ణాంకం tid   = అంశం_ct1.get_local_id(2);

    స్థిరరాశి దశాంశం * dst_fixup_దత్తాంశం = ((స్థిరరాశి దశాంశం *) dst_fixup) + అంశం_ct1.get_సమూహం_range(2) * (2 * 2 * ncols);

    స్థిరరాశి పూర్ణాంకం gqa_నిష్పత్తి = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.

    స్థిరరాశి పూర్ణాంకం iter_k     = (ne11      + (nbatch_fa - 1)) / nbatch_fa;
    స్థిరరాశి పూర్ణాంకం iter_j     = (ne01      + (ncols1    - 1)) / ncols1;
    స్థిరరాశి పూర్ణాంకం iter_z_gqa = (gqa_నిష్పత్తి + (ncols2    - 1)) / ncols2;

    స్థిరరాశి పూర్ణాంకం kbc0 = int64_t(bidx0 + 0) * (iter_k * iter_j * iter_z_gqa * ne12 * ne03) / అంశం_ct1.get_సమూహం_range(2);
    స్థిరరాశి పూర్ణాంకం kbc0_stop =
        int64_t(bidx0 + 1) * (iter_k * iter_j * iter_z_gqa * ne12 * ne03) / అంశం_ct1.get_సమూహం_range(2);

    స్థిరరాశి తార్కిక did_not_have_any_దత్తాంశం   = kbc0 == kbc0_stop;
    స్థిరరాశి తార్కిక wrote_beginning_యొక్క_tile = kbc0 % iter_k == 0;
    స్థిరరాశి తార్కిక did_not_write_చివరి      = kbc0/iter_k == kbc0_stop/iter_k && kbc0_stop % iter_k != 0;
    ఒకవేళ (did_not_have_any_దత్తాంశం || wrote_beginning_యొక్క_tile || did_not_write_చివరి) {
        తిరిగి;
    }

    // z_KV == K/V head index, zt_gqa = Q head start index per K/V head, jt = token position start index
    స్థిరరాశి పూర్ణాంకం sequence =  kbc0 /(iter_k*iter_j*iter_z_gqa*ne12);
    స్థిరరాశి పూర్ణాంకం z_KV     = (kbc0 - iter_k*iter_j*iter_z_gqa*ne12 * sequence)/(iter_k*iter_j*iter_z_gqa);
    స్థిరరాశి పూర్ణాంకం zt_gqa   = (kbc0 - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV)/(iter_k*iter_j);
    స్థిరరాశి పూర్ణాంకం jt       = (kbc0 - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV - iter_k*iter_j * zt_gqa) / iter_k;

    స్థిరరాశి పూర్ణాంకం zt_Q = z_KV*gqa_నిష్పత్తి + zt_gqa*ncols2; // Global Q head start index.

    ఒకవేళ (jt*ncols1 + j >= ne01 || zt_gqa*ncols2 + c >= gqa_నిష్పత్తి) {
        తిరిగి;
    }

    dst += sequence*ne02*ne01*D + jt*ne02*(ncols1*D) + zt_Q*D + (j*ne02 + c)*D + tid;

    // Load the partial result that needs a fixup:
    దశాంశం dst_val = 0.0f;
    దశాంశం max_val = 0.0f;
    దశాంశం rowsum  = 0.0f;
    {
        dst_val = *dst;

        స్థిరరాశి sycl::float2 tmp = dst_fixup[bidx0 * ncols + jc];
        max_val                = tmp.x();
        rowsum                 = tmp.y();
    }

    // Iterate over previous blocks and compute the combined results.
    // All SYCL blocks that get here must have a previous block that needs a fixup.
    పూర్ణాంకం bidx = bidx0 - 1;
    పూర్ణాంకం kbc_stop = kbc0;
    అయితే(నిజం) {
        స్థిరరాశి పూర్ణాంకం kbc = int64_t(bidx) * (iter_k * iter_j * iter_z_gqa * ne12 * ne03) / అంశం_ct1.get_సమూహం_range(2);
        ఒకవేళ (kbc == kbc_stop) { // Did not have any data.
            bidx--;
            kbc_stop = kbc;
            కొనసాగు;
        }

        స్థిరరాశి దశాంశం dst_కూడు = dst_fixup_దత్తాంశం[bidx*ncols*D + jc*D + tid];

        స్థిరరాశి sycl::float2 tmp = dst_fixup[(అంశం_ct1.get_సమూహం_range(2) + bidx) * ncols + jc];

        // Scale the current and new value accumulators depending on the max. values.
        స్థిరరాశి దశాంశం max_val_new = sycl::fmax(max_val, tmp.x());

        స్థిరరాశి దశాంశం diff_val = max_val - max_val_new;
        స్థిరరాశి దశాంశం diff_కూడు = tmp.x() - max_val_new;

        స్థిరరాశి దశాంశం scale_val = diff_val >= SOFTMAX_FTZ_పరిమితి ? sycl::native::ఘాతాంకం(diff_val) : 0.0f;
        స్థిరరాశి దశాంశం scale_కూడు = diff_కూడు >= SOFTMAX_FTZ_పరిమితి ? sycl::native::ఘాతాంకం(diff_కూడు) : 0.0f;

        dst_val = scale_val*dst_val + scale_కూడు*dst_కూడు;
        rowsum  = scale_val * rowsum + scale_కూడు * tmp.y();

        max_val = max_val_new;

        // If this block started in a previous tile we are done and don't need to combine additional partial results.
        ఒకవేళ (kbc % iter_k == 0 || kbc/iter_k < kbc0/iter_k) {
            ఆపు;
        }
        bidx--;
        kbc_stop = kbc;
    }

    // Write back final result:
    *dst = dst_val / rowsum;
}

మూస <పూర్ణాంకం D>  // D == head size

స్థిర శూన్య flash_attn_combine_results(స్థిరరాశి దశాంశం * __restrict__ VKQ_parts,
                                       స్థిరరాశి sycl::float2 * __restrict__ VKQ_meta,
                                       దశాంశం * __restrict__ dst,
                                       స్థిరరాశి పూర్ణాంకం parallel_ఖండాలు,
                                       uint8_t * dpct_local) {
    // Dimension 0: threadIdx.x
    // Dimension 1: blockIdx.x
    // Dimension 2: blockIdx.y
    // Dimension 3: blockIdx.z
    // Memory layout is permuted with [0, 2, 1, 3]

    స్వయంచాలక      అంశం_ct1 = sycl::ext::oneapi::this_work_అంశం::get_nd_అంశం<3>();
    స్థిరరాశి పూర్ణాంకం ne01     = అంశం_ct1.get_సమూహం_range(2);
    స్థిరరాశి పూర్ణాంకం ne02     = అంశం_ct1.get_సమూహం_range(1);

    స్థిరరాశి పూర్ణాంకం నిలువు      = అంశం_ct1.get_సమూహం(2);
    స్థిరరాశి పూర్ణాంకం తల     = అంశం_ct1.get_సమూహం(1);
    స్థిరరాశి పూర్ణాంకం sequence = అంశం_ct1.get_సమూహం(0);

    స్థిరరాశి పూర్ణాంకం j_dst_unrolled = (sequence*ne01 + నిలువు)*ne02 + తల;

    VKQ_parts += j_dst_unrolled * parallel_ఖండాలు*D;
    VKQ_meta  += j_dst_unrolled * parallel_ఖండాలు;
    dst       += j_dst_unrolled *                 D;

    స్థిరరాశి పూర్ణాంకం tid = అంశం_ct1.get_local_id(2);
    __builtin_assume(tid < D);

    స్వయంచాలక meta = (sycl::float2 *) dpct_local;
    కొరకు (పూర్ణాంకం i = tid; i < 2*parallel_ఖండాలు; i += D) {
        ((దశాంశం *) meta)[i] = ((స్థిరరాశి దశాంశం *)VKQ_meta) [i];
    }

    అంశం_ct1.barrier(sycl::access::fence_space::local_space);

    దశాంశం kqmax = meta[0].x();
    కొరకు (పూర్ణాంకం l = 1; l < parallel_ఖండాలు; ++l) {
        kqmax = sycl::max(kqmax, meta[l].x());
    }

    దశాంశం VKQ_numerator   = 0.0f;
    దశాంశం VKQ_denominator = 0.0f;
    కొరకు (పూర్ణాంకం l = 0; l < parallel_ఖండాలు; ++l) {
        స్థిరరాశి దశాంశం KQ_max_scale = sycl::native::ఘాతాంకం(meta[l].x() - kqmax);

        VKQ_numerator   += KQ_max_scale * VKQ_parts[l*D + tid];
        VKQ_denominator += KQ_max_scale * meta[l].y();
    }

    dst[tid] = VKQ_numerator / VKQ_denominator;
}

// Carries the sub-group size (and grf_size_automatic when use_large_grf) as
// kernel properties via get(properties_tag), in place of the
// reqd_sub_group_size attribute and the deprecated parallel_for overload.
మూస <రకం_పేరు KernelFunc, uint32_t కొలత_3, తార్కిక UseGrf>
నిర్మాణం props_kernel {
    KernelFunc kernel;

    శూన్య క్రియాకారకం()(sycl::nd_అంశం<3> అంశం) స్థిరరాశి {
        kernel(అంశం);
    }

    స్వయంచాలక get(sycl::ext::oneapi::experimental::properties_ట్యాగ్) స్థిరరాశి {
        ఒకవేళ constexpr (UseGrf) {
            తిరిగి sycl::ext::oneapi::experimental::properties{
                sycl::ext::intel::experimental::grf_కొలత_automatic,
                sycl::ext::oneapi::experimental::sub_సమూహం_కొలత<కొలత_3> };
        } లేకపోతే {
            తిరిగి sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::sub_సమూహం_కొలత<కొలత_3> };
        }
    }
};

మూస <fattn_kernel_t fattn_kernel, పూర్ణాంకం warp_కొలత, తార్కిక use_large_grf = అబద్ధం>
స్థిర శూన్య lauch_kernel(
    dpct::dim3 సమూహం_range,
    dpct::dim3 local_range,
    వరుస_ptr q,
    గుర్తులేని పూర్ణాంకం local_mem_కొలత_2,
    స్థిరరాశి అక్షరం* __restrict__ Q,
    స్థిరరాశి అక్షరం* __restrict__ K,
    స్థిరరాశి అక్షరం* __restrict__ V,
    స్థిరరాశి అక్షరం* __restrict__ ముసుగు,
    స్థిరరాశి అక్షరం* __restrict__ sinks,
    స్థిరరాశి పూర్ణాంకం* __restrict__ KV_max,
    దశాంశం* __restrict__ dst,
    sycl::float2* __restrict__ dst_meta,
    స్థిరరాశి దశాంశం scale,
    స్థిరరాశి దశాంశం max_bias,
    స్థిరరాశి దశాంశం m0,
    స్థిరరాశి దశాంశం m1,
    స్థిరరాశి uint32_t n_తల_log2,
    స్థిరరాశి దశాంశం logit_softcap,
    స్థిరరాశి int32_t ne00,
    స్థిరరాశి sycl::uint3 ne01,
    స్థిరరాశి int32_t ne02,
    స్థిరరాశి int32_t ne03,
    స్థిరరాశి int32_t nb01,
    స్థిరరాశి int32_t nb02,
    స్థిరరాశి int32_t nb03,
    స్థిరరాశి int32_t ne10,
    స్థిరరాశి int32_t ne11,
    స్థిరరాశి int32_t ne12,
    స్థిరరాశి int32_t ne13,
    స్థిరరాశి int32_t nb11,
    స్థిరరాశి int32_t nb12,
    స్థిరరాశి int64_t nb13,
    స్థిరరాశి int32_t nb21,
    స్థిరరాశి int32_t nb22,
    స్థిరరాశి int64_t nb23,
    స్థిరరాశి int32_t ne31,
    స్థిరరాశి int32_t ne32,
    స్థిరరాశి int32_t ne33,
    స్థిరరాశి int32_t nb31,
    స్థిరరాశి int32_t nb32,
    స్థిరరాశి int64_t nb33) {
    GGML_వాడని(local_mem_కొలత_2);

    స్థిరరాశి స్వయంచాలక rng = sycl::nd_range<3>(
        static_cast<sycl::range<3>>(సమూహం_range * local_range),
        static_cast<sycl::range<3>>(local_range));

    స్థిరరాశి స్వయంచాలక kernel = [=](sycl::nd_అంశం<3> అంశం_ct1) {
        GGML_వాడని(అంశం_ct1);
        fattn_kernel(Q, K, V, ముసుగు, sinks, KV_max, dst, dst_meta, scale,
                     max_bias, m0, m1, n_తల_log2, logit_softcap, ne00,
                     ne01, ne02, ne03, nb01, nb02, nb03, ne10, ne11,
                     ne12, ne13, nb11, nb12, nb13, nb21, nb22, nb23,
                     ne31, ne32, ne33, nb31, nb32, nb33);
    };

    q->submit([&](sycl::handler &cgh) {
        // grf_size_automatic lets the compiler use the large register file when the kernel needs it. grf_size<256> would say the same but is undefined on devices that do not have it.
        cgh.parallel_for(rng, props_kernel<decltype(kernel), warp_కొలత, use_large_grf>{kernel});
    });
}

// use_large_grf lets the compiler choose the large register file. Opt in where spilling limits the kernel, not thread occupancy.
మూస <పూర్ణాంకం DV, పూర్ణాంకం ncols1, పూర్ణాంకం ncols2, fattn_kernel_t fattn_kernel, పూర్ణాంకం warp_కొలత, తార్కిక use_large_grf = అబద్ధం>
శూన్య launch_fattn(
    ggml_backend_sycl_context & ctx, ggml_tensor * dst, స్థిరరాశి పూర్ణాంకం nwarps, స్థిరరాశి size_t nbytes_shared,
    స్థిరరాశి పూర్ణాంకం nbatch_fa, స్థిరరాశి తార్కిక need_f16_K, స్థిరరాశి తార్కిక need_f16_V, స్థిరరాశి తార్కిక ప్రవాహం_k) {

    constexpr పూర్ణాంకం ncols = ncols1 * ncols2;

    స్థిరరాశి ggml_tensor * Q = dst->src[0];
    స్థిరరాశి ggml_tensor * K = dst->src[1];
    స్థిరరాశి ggml_tensor * V = dst->src[2];

    స్థిరరాశి తార్కిక V_is_K_view = V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));

    స్థిరరాశి ggml_tensor * ముసుగు  = dst->src[3];
    స్థిరరాశి ggml_tensor * sinks = dst->src[4];

    ggml_tensor * KQV = dst;

    GGML_ASSERT(Q->type == GGML_TYPE_F32);
    GGML_ASSERT(KQV->type == GGML_TYPE_F32);

    GGML_ASSERT(Q->nb[0] == ggml_element_కొలత(Q));
    GGML_ASSERT(K->nb[0] == ggml_element_కొలత(K));
    GGML_ASSERT(V->nb[0] == ggml_element_కొలత(V));

    GGML_ASSERT(!ముసుగు || ముసుగు->type == GGML_TYPE_F16);

    ggml_sycl_pool & pool = ctx.pool();
    dpct::వరుస_ptr  main_ప్రవాహం = ctx.stream();
    స్థిరరాశి పూర్ణాంకం id  = ggml_sycl_get_device();
    స్థిరరాశి పూర్ణాంకం nsm = ggml_sycl_info().devices[id].nsm;

    స్థిరరాశి ggml_sycl_fattn_extra  extra = ggml_sycl_fattn_get_extra(dst);
    ggml_sycl_pool_alloc<పూర్ణాంకం>    KV_max(pool);
    ggml_sycl_pool_alloc<దశాంశం>  dst_tmp(pool);
    ggml_sycl_pool_alloc<sycl::float2> dst_tmp_meta(pool);

    స్థిరరాశి అక్షరం * K_దత్తాంశం = (స్థిరరాశి అక్షరం *) K->దత్తాంశం;
    size_t nb11 = K->nb[1];
    size_t nb12 = K->nb[2];
    size_t nb13 = K->nb[3];

    స్థిరరాశి అక్షరం * V_దత్తాంశం = (స్థిరరాశి అక్షరం *) V->దత్తాంశం;
    size_t nb21 = V->nb[1];
    size_t nb22 = V->nb[2];
    size_t nb23 = V->nb[3];

    ఒకవేళ (need_f16_K && K->type != GGML_TYPE_F16) {
        స్థిరరాశి size_t bs = ggml_blck_కొలత(K->type);
        స్థిరరాశి size_t ts = ggml_type_కొలత(K->type);

        GGML_ASSERT(extra.K_బఫరు_ptr);
        sycl::half * K_f16_ptr = (sycl::half *) extra.K_బఫరు_ptr;
        ఒకవేళ (ggml_is_contiguously_allocated(K)) {
            గమ్యం_fp16_sycl_t గమ్యం_fp16 = ggml_get_గమ్యం_fp16_sycl(K->type, dst);
            గమ్యం_fp16(K_దత్తాంశం, K_f16_ptr, ggml_nelements(K), main_ప్రవాహం);

            nb11 = nb11 * bs * పరిమాణం(sycl::half) / ts;
            nb12 = nb12 * bs * పరిమాణం(sycl::half) / ts;
            nb13 = nb13 * bs * పరిమాణం(sycl::half) / ts;
        } లేకపోతే {
            GGML_ASSERT(K->nb[0] == ts);
            గమ్యం_fp16_nc_sycl_t గమ్యం_fp16 = ggml_get_గమ్యం_fp16_nc_sycl(K->type);
            స్థిరరాశి int64_t s01 = nb11 / ts;
            స్థిరరాశి int64_t s02 = nb12 / ts;
            స్థిరరాశి int64_t s03 = nb13 / ts;
            గమ్యం_fp16(K_దత్తాంశం, K_f16_ptr, K->ne[0], K->ne[1], K->ne[2], K->ne[3], s01, s02, s03, main_ప్రవాహం);

            nb11 = K->ne[0] * పరిమాణం(sycl::half);
            nb12 = K->ne[1] * nb11;
            nb13 = K->ne[2] * nb12;
        }
        K_దత్తాంశం = (అక్షరం *) K_f16_ptr;
    }

    ఒకవేళ (need_f16_V && V->type != GGML_TYPE_F16) {
        ఒకవేళ (V_is_K_view) {
            V_దత్తాంశం = K_దత్తాంశం;
            nb21   = nb11;
            nb22   = nb12;
            nb23   = nb13;
        } లేకపోతే {
            స్థిరరాశి size_t bs = ggml_blck_కొలత(V->type);
            స్థిరరాశి size_t ts = ggml_type_కొలత(V->type);

            GGML_ASSERT(extra.V_బఫరు_ptr);
            sycl::half * V_f16_ptr = (sycl::half *) extra.V_బఫరు_ptr;
            ఒకవేళ (ggml_is_contiguously_allocated(V)) {
                గమ్యం_fp16_sycl_t గమ్యం_fp16 = ggml_get_గమ్యం_fp16_sycl(V->type, dst);
                గమ్యం_fp16(V_దత్తాంశం, V_f16_ptr, ggml_nelements(V), main_ప్రవాహం);
                V_దత్తాంశం = (అక్షరం *) V_f16_ptr;

                nb21 = nb21 * bs * పరిమాణం(sycl::half) / ts;
                nb22 = nb22 * bs * పరిమాణం(sycl::half) / ts;
                nb23 = nb23 * bs * పరిమాణం(sycl::half) / ts;
            } లేకపోతే {
                GGML_ASSERT(V->nb[0] == ts);
                గమ్యం_fp16_nc_sycl_t గమ్యం_fp16 = ggml_get_గమ్యం_fp16_nc_sycl(V->type);
                స్థిరరాశి int64_t s01 = nb21 / ts;
                స్థిరరాశి int64_t s02 = nb22 / ts;
                స్థిరరాశి int64_t s03 = nb23 / ts;
                గమ్యం_fp16(V_దత్తాంశం, V_f16_ptr, V->ne[0], V->ne[1], V->ne[2], V->ne[3], s01, s02, s03, main_ప్రవాహం);

                nb21 = V->ne[0] * పరిమాణం(sycl::half);
                nb22 = V->ne[1] * nb21;
                nb23 = V->ne[2] * nb22;
            }
            V_దత్తాంశం = (అక్షరం *) V_f16_ptr;
        }
    }

    స్థిరరాశి పూర్ణాంకం ntiles_x     = ((Q->ne[1] + ncols1 - 1) / ncols1);
    స్థిరరాశి పూర్ణాంకం gqa_నిష్పత్తి    = Q->ne[2] / K->ne[2];
    స్థిరరాశి పూర్ణాంకం ntiles_z_gqa = ((gqa_నిష్పత్తి + ncols2 - 1) / ncols2);
    స్థిరరాశి పూర్ణాంకం ntiles_మొత్తం = ntiles_x * ntiles_z_gqa * K->ne[2] * Q->ne[3];

    // Optional optimization where the mask is scanned to determine whether part of the calculation can be skipped.
    // Only worth the overhead if there is at lease one FATTN_KQ_STRIDE x FATTN_KQ_STRIDE square to be skipped or
    //     multiple sequences of possibly different lengths.
    ఒకవేళ (ముసుగు && K->ne[1] % FATTN_KQ_STRIDE == 0 && (Q->ne[1] >= 1024 || Q->ne[3] > 1)) {
        స్థిరరాశి పూర్ణాంకం s31 = ముసుగు->nb[1] / పరిమాణం(sycl::half2);
        స్థిరరాశి పూర్ణాంకం s33 = ముసుగు->nb[3] / పరిమాణం(sycl::half2);

        స్థిరరాశి dpct::dim3 ఖండాలు_అంకె_KV_max(ntiles_x, Q->ne[3], 1);
        స్థిరరాశి dpct::dim3 ఖండం_dim_KV_max(FATTN_KQ_STRIDE / 2, 1, 1);

        స్థిరరాశి పూర్ణాంకం ne_KV_max = ఖండాలు_అంకె_KV_max.x*ఖండాలు_అంకె_KV_max.y;
        స్థిరరాశి పూర్ణాంకం iter_k = K->ne[1] / FATTN_KQ_STRIDE;

        KV_max.alloc(ne_KV_max);
        {
            dpct::has_capability_or_fail(main_ప్రవాహం->get_device(), { sycl::aspect::fp16 });

            main_ప్రవాహం->submit([&](sycl::handler & cgh) {
                sycl::local_accessor<పూర్ణాంకం, 1> buf_iw_acc_ct1(sycl::range<1>(warp_కొలత), cgh);

                స్వయంచాలక ముసుగు_దత్తాంశం_ct0  = (స్థిరరాశి sycl::half2 *) ముసుగు->దత్తాంశం;
                స్వయంచాలక KV_max_ptr_ct1 = KV_max.ptr;

                cgh.parallel_for(sycl::nd_range<3>(ఖండాలు_అంకె_KV_max * ఖండం_dim_KV_max, ఖండం_dim_KV_max),
                                 [=](sycl::nd_అంశం<3> అంశం_ct1) [[sycl::reqd_sub_సమూహం_కొలత(warp_కొలత)]] {
                                     GGML_వాడని(అంశం_ct1);
                                     flash_attn_ముసుగు_గమ్యం_KV_max<ncols1, warp_కొలత>(
                                         ముసుగు_దత్తాంశం_ct0, KV_max_ptr_ct1, iter_k, s31, s33,
                                         buf_iw_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                                 });
            });
        }
        SYCL_CHECK(0);
    }

    స్థిరరాశి dpct::dim3 ఖండం_dim(warp_కొలత, nwarps, 1);

    // Max. number of active blocks limited by occupancy.
    పూర్ణాంకం max_ఖండాలు_చొప్పున_sm = ggml_sycl_info().devices[id].max_wg_చొప్పున_cu;
    పూర్ణాంకం parallel_ఖండాలు = max_ఖండాలు_చొప్పున_sm;
    dpct::dim3 ఖండాలు_అంకె;
    ఒకవేళ (ప్రవాహం_k) {
        // For short contexts it can be faster to have the SMs work on whole tiles because this lets us skip the fixup.
        స్థిరరాశి పూర్ణాంకం max_ఖండాలు = max_ఖండాలు_చొప్పున_sm*nsm;
        స్థిరరాశి పూర్ణాంకం nblocks_ప్రవాహం_k = max_ఖండాలు;
        స్థిరరాశి తార్కిక use_ప్రవాహం_k = నిజం;

        ఖండాలు_అంకె.x = use_ప్రవాహం_k ? nblocks_ప్రవాహం_k : ntiles_మొత్తం;
        ఖండాలు_అంకె.y = 1;
        ఖండాలు_అంకె.z = 1;

        ఒకవేళ (ntiles_మొత్తం % ఖండాలు_అంకె.x != 0) { // Fixup is only needed if the SMs work on fractional tiles.
            dst_tmp_meta.alloc((size_t(ఖండాలు_అంకె.x) * ncols * (2 + DV/2)));
        }
    } లేకపోతే {
        స్థిరరాశి పూర్ణాంకం ntiles_KQ = (K->ne[1] + nbatch_fa - 1) / nbatch_fa; // Max. number of parallel blocks limited by tensor size.

        // parallel_blocks must not be larger than what the tensor size allows:
        parallel_ఖండాలు = std::min(parallel_ఖండాలు, ntiles_KQ);
        // todo fix the hard code change
        // parallel_blocks = ntiles_KQ;

        // If ntiles_total % blocks_per_wave != 0 then some efficiency is lost due to tail effects.
        // Test whether parallel_blocks can be set to a higher value for better efficiency.
        స్థిరరాశి పూర్ణాంకం ఖండాలు_చొప్పున_wave = nsm * max_ఖండాలు_చొప్పున_sm;
        పూర్ణాంకం nwaves_ఉత్తమ = 0;
        పూర్ణాంకం efficiency_percent_ఉత్తమ = 0;
        కొరకు (పూర్ణాంకం parallel_ఖండాలు_test = parallel_ఖండాలు; parallel_ఖండాలు_test <= ntiles_KQ; ++parallel_ఖండాలు_test) {
            స్థిరరాశి పూర్ణాంకం nblocks_మొత్తం = ntiles_మొత్తం * parallel_ఖండాలు_test;
            స్థిరరాశి పూర్ణాంకం nwaves = (nblocks_మొత్తం + ఖండాలు_చొప్పున_wave - 1) / ఖండాలు_చొప్పున_wave;
            స్థిరరాశి పూర్ణాంకం efficiency_percent = 100 * nblocks_మొత్తం / (nwaves*ఖండాలు_చొప్పున_wave);

            // Stop trying configurations with more waves if we already have good efficiency to avoid excessive overhead.
            ఒకవేళ (efficiency_percent_ఉత్తమ >= 95 && nwaves > nwaves_ఉత్తమ) {
                ఆపు;
            }

            ఒకవేళ (efficiency_percent > efficiency_percent_ఉత్తమ) {
                nwaves_ఉత్తమ = nwaves;
                efficiency_percent_ఉత్తమ = efficiency_percent;
                parallel_ఖండాలు = parallel_ఖండాలు_test;
            }
        }

        ఖండాలు_అంకె.x = ntiles_x;
        ఖండాలు_అంకె.y = parallel_ఖండాలు;
        ఖండాలు_అంకె.z = ntiles_z_gqa*K->ne[2]*Q->ne[3];

        ఒకవేళ (parallel_ఖండాలు > 1) {
            dst_tmp.alloc(parallel_ఖండాలు*ggml_nelements(KQV));
            dst_tmp_meta.alloc(parallel_ఖండాలు*ggml_nrows(KQV));
        }
    }

    దశాంశం scale         = 1.0f;
    దశాంశం max_bias      = 0.0f;
    దశాంశం logit_softcap = 0.0f;

    జ్ఞాపకం_నకలు(&scale,         (స్థిరరాశి దశాంశం *) KQV->op_params + 0, పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&max_bias,      (స్థిరరాశి దశాంశం *) KQV->op_params + 1, పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&logit_softcap, (స్థిరరాశి దశాంశం *) KQV->op_params + 2, పరిమాణం(దశాంశం));

    ఒకవేళ (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    స్థిరరాశి uint32_t n_తల      = Q->ne[2];
    స్థిరరాశి uint32_t n_తల_log2 = 1u << uint32_t(floorf(log2f(దశాంశం(n_తల))));

    స్థిరరాశి దశాంశం m0 = powf(2.0f, -(max_bias       ) / n_తల_log2);
    స్థిరరాశి దశాంశం m1 = powf(2.0f, -(max_bias / 2.0f) / n_తల_log2);

    // TODO other tensor dimensions after removal of WMMA kernel:
    స్థిరరాశి sycl::uint3 ne01 = init_fastdiv_విలువలు(Q->ne[1]);

    GGML_ASSERT(ఖండం_dim.x % warp_కొలత == 0);

    lauch_kernel<fattn_kernel, warp_కొలత, use_large_grf>(
        ఖండాలు_అంకె, ఖండం_dim, main_ప్రవాహం, (గుర్తులేని పూర్ణాంకం) nbytes_shared, (స్థిరరాశి అక్షరం *) Q->దత్తాంశం, K_దత్తాంశం, V_దత్తాంశం,
        ముసుగు ? ((స్థిరరాశి అక్షరం *) ముసుగు->దత్తాంశం) : శూన్యసూచిక, sinks ? ((స్థిరరాశి అక్షరం *) sinks->దత్తాంశం) : శూన్యసూచిక, KV_max.ptr,
        !ప్రవాహం_k && parallel_ఖండాలు > 1 ? dst_tmp.ptr : (దశాంశం *) KQV->దత్తాంశం, (sycl::float2 *)dst_tmp_meta.ptr, scale, max_bias, m0, m1,
        n_తల_log2, logit_softcap, Q->ne[0], ne01, Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3], K->ne[0],
        K->ne[1], K->ne[2], K->ne[3], nb11, nb12, nb13, nb21, nb22, nb23, ముసుగు ? ముసుగు->ne[1] : 0,
        ముసుగు ? ముసుగు->ne[2] : 0, ముసుగు ? ముసుగు->ne[3] : 0, ముసుగు ? ముసుగు->nb[1] : 0, ముసుగు ? ముసుగు->nb[2] : 0,
        ముసుగు ? ముసుగు->nb[3] : 0);
    SYCL_CHECK(0);

    ఒకవేళ (ప్రవాహం_k) {
        ఒకవేళ (ntiles_మొత్తం % ఖండాలు_అంకె.x != 0) { // Fixup is only needed if the SMs work on fractional tiles.
            స్థిరరాశి dpct::dim3 ఖండం_dim_combine(DV, 1, 1);
            స్థిరరాశి dpct::dim3 ఖండాలు_అంకె_combine = { ఖండాలు_అంకె.x, ncols1, ncols2 };

            main_ప్రవాహం->submit([&](sycl::handler & cgh) {
                స్వయంచాలక KQV_దత్తాంశం_ct0         = (దశాంశం *) KQV->దత్తాంశం;
                స్వయంచాలక dst_tmp_meta_ptr_ct1 = dst_tmp_meta.ptr;
                స్వయంచాలక Q_ne_ct2             = Q->ne[1];
                స్వయంచాలక Q_ne_ct3             = Q->ne[2];
                స్వయంచాలక Q_ne_ct4             = Q->ne[3];
                స్వయంచాలక K_ne_ct5             = K->ne[1];
                స్వయంచాలక K_ne_ct6             = K->ne[2];

                cgh.parallel_for(sycl::nd_range<3>(ఖండాలు_అంకె_combine * ఖండం_dim_combine, ఖండం_dim_combine),
                                 [=](sycl::nd_అంశం<3> అంశం_ct1) [[sycl::reqd_sub_సమూహం_కొలత(warp_కొలత)]] {
                                     GGML_వాడని(అంశం_ct1);
                                     flash_attn_ప్రవాహం_k_fixup<DV, ncols1, ncols2>(KQV_దత్తాంశం_ct0, dst_tmp_meta_ptr_ct1,
                                                                                   Q_ne_ct2, Q_ne_ct3, Q_ne_ct4,
                                                                                   K_ne_ct5, K_ne_ct6, nbatch_fa);
                                 });
            });
        }
    } లేకపోతే ఒకవేళ (parallel_ఖండాలు > 1) {
        స్థిరరాశి dpct::dim3 ఖండం_dim_combine(DV, 1, 1);
        స్థిరరాశి dpct::dim3 ఖండాలు_అంకె_combine(Q->ne[1], Q->ne[2], Q->ne[3]);
        స్థిరరాశి size_t     nbytes_shared_combine = parallel_ఖండాలు * పరిమాణం(sycl::float2);
        main_ప్రవాహం->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(sycl::range<1>(nbytes_shared_combine), cgh);

            స్వయంచాలక dst_tmp_ptr_ct0      = dst_tmp.ptr;
            స్వయంచాలక dst_tmp_meta_ptr_ct1 = dst_tmp_meta.ptr;
            స్వయంచాలక KQV_దత్తాంశం_ct2         = (దశాంశం *) KQV->దత్తాంశం;

            cgh.parallel_for(sycl::nd_range<3>(ఖండాలు_అంకె_combine * ఖండం_dim_combine, ఖండం_dim_combine),
                             [=](sycl::nd_అంశం<3> అంశం_ct1) [[sycl::reqd_sub_సమూహం_కొలత(warp_కొలత)]] {
                                 GGML_వాడని(అంశం_ct1);
                                 flash_attn_combine_results<DV>(
                                     dst_tmp_ptr_ct0, dst_tmp_meta_ptr_ct1, KQV_దత్తాంశం_ct2, parallel_ఖండాలు,
                                     dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                             });
        });
    }
    SYCL_CHECK(0);
}
