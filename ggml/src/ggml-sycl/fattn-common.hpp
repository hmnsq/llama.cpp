#ہدایت once

#شامل <sycl/sycl.hpp>
#شامل <sycl/ext/intel/experimental/grf_size_properties.hpp>
#شامل "dpct/helper.hpp"
#شامل "common.hpp"
#شامل "convert.hpp"
#شامل "vecdotq.hpp"
#شامل "fattn.hpp"

#شامل "ggml.h"

#شامل <cstdint>
#شامل <cmath>
#شامل <float.h>


#تعریف_کرو FATTN_KQ_STRIDE       256
#تعریف_کرو HALF_MAX_HALF         sycl::half(65504.0f/2) // Use neg. of this instead of -INFINITY to initialize KQ max vals to avoid NaN upon subtraction.
#تعریف_کرو SOFTMAX_FTZ_حد -20.0f                   // Softmax exp. of values smaller than this are flushed to zero to avoid NaNs.
#تعریف_کرو FATTN_KQ_MAX_انحراف (3.0f*0.6931f)

قسمتعریف خالی (*fattn_kernel_t)(
    ثابت حرف* Q,
    ثابت حرف* K,
    ثابت حرف* V,
    ثابت حرف* نقاب,
    ثابت حرف* sinks,
    ثابت صحیح* KV_max,
    اعشاری* dst,
    sycl::float2* dst_meta,
    ثابت اعشاری scale,
    ثابت اعشاری max_bias,
    ثابت اعشاری m0,
    ثابت اعشاری m1,
    ثابت uint32_t n_سر_log2,
    ثابت اعشاری logit_softcap,
    ثابت int32_t ne00,
    ثابت sycl::uint3 ne01,
    ثابت int32_t ne02,
    ثابت int32_t ne03,
    ثابت int32_t nb01,
    ثابت int32_t nb02,
    ثابت int32_t nb03,
    ثابت int32_t ne10,
    ثابت int32_t ne11,
    ثابت int32_t ne12,
    ثابت int32_t ne13,
    ثابت int32_t nb11,
    ثابت int32_t nb12,
    ثابت int64_t nb13,
    ثابت int32_t nb21,
    ثابت int32_t nb22,
    ثابت int64_t nb23,
    ثابت int32_t ne31,
    ثابت int32_t ne32,
    ثابت int32_t ne33,
    ثابت int32_t nb31,
    ثابت int32_t nb32,
    ثابت int64_t nb33);

قسمتعریف اعشاری (*vec_dot_KQ_t)(
    ثابت حرف * __restrict__ K_c, ثابت خالی * __restrict__ Q_v, ثابت صحیح * __restrict__ Q_q8 , ثابت خالی * __restrict__ Q_ds);

سانچہ <صحیح D, صحیح nthreads>
ساکن __dpct_inline__ اعشاری vec_dot_fattn_vec_KQ_f16(ثابت حرف * __restrict__ K_c,
                                                      ثابت خالی * __restrict__ Q_v,
                                                      ثابت صحیح * __restrict__ Q_q8,
                                                      ثابت خالی * __restrict__ Q_ds_v) {
    ثابت sycl::half2 * K_h2 = (ثابت sycl::half2 *) K_c;
    GGML_غیر_مستعمل(Q_q8);
    GGML_غیر_مستعمل(Q_ds_v);

    constexpr صحیح cpy_nb = ggml_sycl_get_max_cpy_بائٹیں();
    constexpr صحیح cpy_ne = cpy_nb / 4;

    اعشاری مجموع = 0.0f;

#ہدایت unroll
    برائے (صحیح k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads*cpy_ne) {
        sycl::half2 tmp[cpy_ne];
        ggml_sycl_memcpy_1<حجم(tmp)>(
            tmp,
            K_h2 + k_KQ_0 + (sycl::ext::oneapi::this_work_شے::get_nd_شے<3>().get_local_id(2) % nthreads) * cpy_ne);
#ہدایت unroll
        برائے (صحیح k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
#اگر_تعریف_شدہ GGML_SYCL_F16
            ggml_sycl_mad(مجموع,                tmp[k_KQ_1] , ((ثابت sycl::half2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#ورنہ
            ggml_sycl_mad(مجموع, __half22float2(tmp[k_KQ_1]), ((ثابت sycl::float2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#اگر_ختم // GGML_SYCL_F16
        }
    }

    واپس مجموع;
}

سانچہ <صحیح D, صحیح nthreads, صحیح warp_سائز>
ساکن __dpct_inline__ اعشاری vec_dot_fattn_vec_KQ_q4_0(ثابت حرف * __restrict__ K_c,
                                                       ثابت خالی * __restrict__ Q_v,
                                                       ثابت صحیح * __restrict__ Q_q8,
                                                       ثابت خالی * __restrict__ Q_ds_v) {
    خودکار شے_ct1 = sycl::ext::oneapi::this_work_شے::get_nd_شے<3>();

    ثابت بلاک_q4_0 * K_q4_0   = (ثابت بلاک_q4_0 *) K_c;
    GGML_غیر_مستعمل(Q_v);

    اعشاری مجموع = 0.0f;

#ہدایت unroll
    برائے (صحیح k_KQ_0 = 0; k_KQ_0 < صحیح(D/حجم(صحیح)); k_KQ_0 += nthreads) {
        ثابت صحیح k_KQ =
            k_KQ_0 + (nthreads == warp_سائز ? شے_ct1.get_local_id(2) : شے_ct1.get_local_id(2) % nthreads);

        ثابت صحیح ib    = k_KQ /  QI8_1;
        ثابت صحیح iqs4  = k_KQ %  QI4_0;
        ثابت صحیح سرکاؤ = k_KQ & (QI8_1/2);

        صحیح v;
        ggml_sycl_memcpy_1<حجم(صحیح), 2>(&v, K_q4_0[ib].qs + حجم(صحیح)*iqs4);
        v = (v >> سرکاؤ) & 0x0F0F0F0F;
        ثابت صحیح u = Q_q8[k_KQ_0/nthreads];

        ثابت صحیح sumi = ggml_sycl_dp4a(v, u, 0);

        ثابت sycl::float2 Q_ds = ((ثابت sycl::float2 *) Q_ds_v)[k_KQ_0 / nthreads];
        مجموع += __half2float(K_q4_0[ib].d) * (sumi*Q_ds.x() - (8/QI8_1)*Q_ds.y());
    }

    واپس مجموع;
}

سانچہ <صحیح D, صحیح nthreads , صحیح warp_سائز>
ساکن __dpct_inline__ اعشاری vec_dot_fattn_vec_KQ_q4_1(ثابت حرف * __restrict__ K_c,
                                                       ثابت خالی * __restrict__ Q_v,
                                                       ثابت صحیح * __restrict__ Q_q8,
                                                       ثابت خالی * __restrict__ Q_ds_v) {
    خودکار               شے_ct1 = sycl::ext::oneapi::this_work_شے::get_nd_شے<3>();
    ثابت بلاک_q4_1 * K_q4_1   = (ثابت بلاک_q4_1 *) K_c;
    GGML_غیر_مستعمل(Q_v);

    اعشاری مجموع = 0.0f;

#ہدایت unroll
    برائے (صحیح k_KQ_0 = 0; k_KQ_0 < صحیح(D/حجم(صحیح)); k_KQ_0 += nthreads) {
        ثابت صحیح k_KQ =
            k_KQ_0 + (nthreads == warp_سائز ? شے_ct1.get_local_id(2) : شے_ct1.get_local_id(2) % nthreads);

        ثابت صحیح ib    = k_KQ /  QI8_1;
        ثابت صحیح iqs4  = k_KQ %  QI4_1;
        ثابت صحیح سرکاؤ = k_KQ & (QI8_1/2);

        صحیح v;
        ggml_sycl_memcpy_1<حجم(صحیح)>(&v, K_q4_1[ib].qs + حجم(صحیح)*iqs4);
        v = (v >> سرکاؤ) & 0x0F0F0F0F;
        ثابت صحیح u = Q_q8[k_KQ_0/nthreads];

        ثابت صحیح sumi = ggml_sycl_dp4a(v, u, 0);

        ثابت sycl::float2 K_dm = (K_q4_1[ib].dm).سانچہ convert<اعشاری, sycl::rounding_منوال::automatic>();
        ثابت sycl::float2 Q_ds = ((ثابت sycl::float2 *) Q_ds_v)[k_KQ_0 / nthreads];

        مجموع += K_dm.x()*Q_ds.x()*sumi + K_dm.y()*Q_ds.y()/QI8_1;
    }

    واپس مجموع;
}

سانچہ <صحیح D, صحیح nthreads, صحیح warp_سائز>
ساکن __dpct_inline__ اعشاری vec_dot_fattn_vec_KQ_q5_0(ثابت حرف * __restrict__ K_c,
                                                       ثابت خالی * __restrict__ Q_v,
                                                       ثابت صحیح * __restrict__ Q_q8,
                                                       ثابت خالی * __restrict__ Q_ds_v) {
    خودکار               شے_ct1 = sycl::ext::oneapi::this_work_شے::get_nd_شے<3>();
    ثابت بلاک_q5_0 * K_q5_0   = (ثابت بلاک_q5_0 *) K_c;
    GGML_غیر_مستعمل(Q_v);

    اعشاری مجموع = 0.0f;

#ہدایت unroll
    برائے (صحیح k_KQ_0 = 0; k_KQ_0 < صحیح(D/حجم(صحیح)); k_KQ_0 += nthreads) {
        ثابت صحیح k_KQ =
            k_KQ_0 + (nthreads == warp_سائز ? شے_ct1.get_local_id(2) : شے_ct1.get_local_id(2) % nthreads);

        ثابت صحیح ib    = k_KQ /  QI8_1;
        ثابت صحیح iqs4  = k_KQ %  QI5_0;
        ثابت صحیح iqs8  = k_KQ %  QI8_1;
        ثابت صحیح سرکاؤ = k_KQ & (QI8_1/2);

        صحیح v;
        ggml_sycl_memcpy_1<حجم(صحیح), 2>(&v, K_q5_0[ib].qs + حجم(صحیح)*iqs4);
        v = (v >> سرکاؤ) & 0x0F0F0F0F;

        {
            صحیح vh;
            ggml_sycl_memcpy_1<حجم(صحیح), 2>(&vh, K_q5_0[ib].qh);
            vh >>= iqs8 * QI5_0;

            v |= (vh <<  4) & 0x00000010; // 0 ->  4
            v |= (vh << 11) & 0x00001000; // 1 -> 12
            v |= (vh << 18) & 0x00100000; // 2 -> 20
            v |= (vh << 25) & 0x10000000; // 3 -> 28
        }

        ثابت صحیح u = Q_q8[k_KQ_0/nthreads];

        ثابت صحیح sumi = ggml_sycl_dp4a(v, u, 0);

        ثابت sycl::float2 Q_ds = ((ثابت sycl::float2 *) Q_ds_v)[k_KQ_0 / nthreads];

        مجموع += __half2float(K_q5_0[ib].d) * (sumi*Q_ds.x() - (16/QI8_1)*Q_ds.y());
    }

    واپس مجموع;
}

سانچہ <صحیح D, صحیح nthreads, صحیح warp_سائز>
ساکن __dpct_inline__ اعشاری vec_dot_fattn_vec_KQ_q5_1(ثابت حرف * __restrict__ K_c,
                                                       ثابت خالی * __restrict__ Q_v,
                                                       ثابت صحیح * __restrict__ Q_q8,
                                                       ثابت خالی * __restrict__ Q_ds_v) {
    خودکار               شے_ct1 = sycl::ext::oneapi::this_work_شے::get_nd_شے<3>();
    ثابت بلاک_q5_1 * K_q5_1   = (ثابت بلاک_q5_1 *) K_c;
    GGML_غیر_مستعمل(Q_v);

    اعشاری مجموع = 0.0f;

#ہدایت unroll
    برائے (صحیح k_KQ_0 = 0; k_KQ_0 < صحیح(D/حجم(صحیح)); k_KQ_0 += nthreads) {
        ثابت صحیح k_KQ =
            k_KQ_0 + (nthreads == warp_سائز ? شے_ct1.get_local_id(2) : شے_ct1.get_local_id(2) % nthreads);

        ثابت صحیح ib    = k_KQ /  QI8_1;
        ثابت صحیح iqs4  = k_KQ %  QI5_1;
        ثابت صحیح iqs8  = k_KQ %  QI8_1;
        ثابت صحیح سرکاؤ = k_KQ & (QI8_1/2);

        صحیح v;
        ggml_sycl_memcpy_1<حجم(صحیح)>(&v, K_q5_1[ib].qs + حجم(صحیح)*iqs4);
        v = (v >> سرکاؤ) & 0x0F0F0F0F;

        {
            صحیح vh;
            ggml_sycl_memcpy_1<حجم(صحیح)>(&vh, K_q5_1[ib].qh);
            vh >>= iqs8 * QI5_0;

            v |= (vh <<  4) & 0x00000010; // 0 ->  4
            v |= (vh << 11) & 0x00001000; // 1 -> 12
            v |= (vh << 18) & 0x00100000; // 2 -> 20
            v |= (vh << 25) & 0x10000000; // 3 -> 28
        }

        ثابت صحیح u = Q_q8[k_KQ_0/nthreads];

        ثابت صحیح sumi = ggml_sycl_dp4a(v, u, 0);

        ثابت sycl::float2 K_dm = (K_q5_1[ib].dm).سانچہ convert<اعشاری, sycl::rounding_منوال::automatic>();
        ثابت sycl::float2 Q_ds = ((ثابت sycl::float2 *) Q_ds_v)[k_KQ_0 / nthreads];

        مجموع += K_dm.x()*Q_ds.x()*sumi + K_dm.y()*Q_ds.y()/QI8_1;
    }

    واپس مجموع;
}

سانچہ <صحیح D, صحیح nthreads, صحیح warp_سائز>
ساکن __dpct_inline__ اعشاری vec_dot_fattn_vec_KQ_q8_0(ثابت حرف * __restrict__ K_c,
                                                       ثابت خالی * __restrict__ Q_v,
                                                       ثابت صحیح * __restrict__ Q_q8,
                                                       ثابت خالی * __restrict__ Q_ds_v) {
    خودکار               شے_ct1 = sycl::ext::oneapi::this_work_شے::get_nd_شے<3>();
    ثابت بلاک_q8_0 * K_q8_0   = (ثابت بلاک_q8_0 *) K_c;
    GGML_غیر_مستعمل(Q_v);

    اعشاری مجموع = 0.0f;

#ہدایت unroll
    برائے (صحیح k_KQ_0 = 0; k_KQ_0 < صحیح(D/حجم(صحیح)); k_KQ_0 += nthreads) {
        ثابت صحیح k_KQ =
            k_KQ_0 + (nthreads == warp_سائز ? شے_ct1.get_local_id(2) : شے_ct1.get_local_id(2) % nthreads);

        ثابت صحیح ib  = k_KQ / QI8_0;
        ثابت صحیح iqs = k_KQ % QI8_0;

        صحیح v;
        ggml_sycl_memcpy_1<حجم(v), 2>(&v, K_q8_0[ib].qs + 4*iqs);

        ثابت sycl::float2 * Q_ds = (ثابت sycl::float2 *) Q_ds_v;
        ثابت اعشاری          Q_d  = Q_ds[k_KQ_0 / nthreads].x();

        مجموع += vec_dot_q8_0_q8_1_impl<اعشاری, 1>(&v, &Q_q8[k_KQ_0/nthreads], K_q8_0[ib].d, Q_d);
    }

    واپس مجموع;
}

سانچہ <قسم_نام Tds, صحیح ni, صحیح warp_سائز>
ساکن __dpct_inline__ خالی quantize_q8_1_مقصد_shared(ثابت اعشاری * __restrict__ x,
                                                    ثابت اعشاری scale,
                                                    صحیح * __restrict__ yq32,
                                                    خالی * __restrict__ yds) {
    خودکار شے_ct1 = sycl::ext::oneapi::this_work_شے::get_nd_شے<3>();

    اعشاری vals[حجم(صحیح)] = { 0.0f };
#ہدایت unroll
    برائے (صحیح l = 0; l < صحیح(حجم(صحیح)); ++l) {
        vals[l] =
            (ni == warp_سائز || شے_ct1.get_local_id(2) < ni) ? scale * x[4 * شے_ct1.get_local_id(2) + l] : 0.0f;
    }

    اعشاری amax = sycl::fabs(vals[0]);
    اعشاری مجموع  = vals[0];
#ہدایت unroll
    برائے (صحیح l = 1; l < صحیح(حجم(صحیح)); ++l) {
        amax = sycl::fmax(amax, sycl::fabs(vals[l]));
        مجموع += vals[l];
    }
#ہدایت unroll
    برائے (صحیح نقاب = QI8_1/2; نقاب > 0; نقاب >>= 1) {
        amax = sycl::fmax(
            amax, dpct::ترتیب_بدلو_sub_گروہ_by_xor(sycl::ext::oneapi::this_work_شے::get_sub_گروہ(), amax, نقاب));
        مجموع += dpct::ترتیب_بدلو_sub_گروہ_by_xor(sycl::ext::oneapi::this_work_شے::get_sub_گروہ(), مجموع, نقاب);
    }

    ثابت اعشاری d = amax / 127;
    صحیح q32 = 0;
    int8_t * q8 = (int8_t *) &q32;

    اگر (d != 0.0f) {
#ہدایت unroll
        برائے (صحیح l = 0; l < صحیح(حجم(صحیح)); ++l) {
            q8[l] = sycl::round(vals[l] / d);
        }
    }

    yq32[شے_ct1.get_local_id(2)] = q32;
    اگر (شے_ct1.get_local_id(2) % QI8_1 == 0 && (ni == warp_سائز || شے_ct1.get_local_id(2) < ni)) {
        اگر (std::is_same<Tds, sycl::half2>::value) {
            ((sycl::half2  *) yds)[شے_ct1.get_local_id(2)/QI8_1] =  make_half2(d, مجموع);
        } ورنہ {
            ((sycl::float2 *) yds)[شے_ct1.get_local_id(2)/QI8_1] = make_float2(d, مجموع);
        }
    }
}

قسمتعریف خالی (*dequantize_V_t)(ثابت خالی *, خالی *, ثابت int64_t);

سانچہ <قسم_نام T, صحیح ne>
ساکن __dpct_inline__ خالی dequantize_V_f16(ثابت خالی * __restrict__ vx, خالی * __restrict__ dst, ثابت int64_t i0) {
    اگر constexpr (std::is_same_v<T, sycl::half>) {
        ggml_sycl_memcpy_1<ne * حجم(sycl::half)>(dst, (ثابت sycl::half *) vx + i0);
    } ورنہ اگر constexpr (std::is_same_v<T, اعشاری>) {
        static_assert(ne % 2 == 0, "bad ne");
        sycl::half2 tmp[ne / 2];
        ggml_sycl_memcpy_1<ne * حجم(sycl::half)>(tmp, (ثابت sycl::half *) vx + i0);
        sycl::float2 * dst_f2 = (sycl::float2 *) dst;
#ہدایت unroll
        برائے (صحیح l = 0; l < ne/2; ++l) {
            dst_f2[l] = tmp[l].سانچہ convert<اعشاری, sycl::rounding_منوال::automatic>();
        }
    } ورنہ {
        static_assert(std::is_same_v<T, خالی>, "unsupported type");
    }
}

سانچہ <قسم_نام T, صحیح ne>
ساکن __dpct_inline__ خالی dequantize_V_q4_0(ثابت خالی * __restrict__ vx, خالی * __restrict__ dst, ثابت int64_t i0) {
    ثابت بلاک_q4_0 * x = (ثابت بلاک_q4_0 *) vx;

    ثابت int64_t ib    =  i0          /  QK4_0;
    ثابت صحیح     iqs   =  i0          % (QK4_0/2);
    ثابت صحیح     سرکاؤ = (i0 % QK4_0) / (QK4_0/2);

    صحیح q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_sycl_memcpy_1<ne, 2>(&q, x[ib].qs + iqs);
    q >>= 4*سرکاؤ;
    q &= 0x0F0F0F0F;
    q = dpct::vectorized_ثنائی<sycl::char4>(q, 0x08080808, dpct::sub_sat());

    ثابت int8_t * q8 = (ثابت int8_t *) &q;

#اگر_تعریف_شدہ GGML_SYCL_F16
    اگر constexpr (std::is_same_v<T, sycl::half>) {
        ثابت sycl::half2 d = sycl::half2(x[ib].d);

#ہدایت unroll
        برائے (صحیح l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * sycl::half2(q8[l0 + 0], q8[l0 + 1]);
        }
    } ورنہ
#اگر_ختم // GGML_SYCL_F16
    اگر constexpr (std::is_same_v<T, اعشاری>) {
        ثابت اعشاری d = x[ib].d;

#ہدایت unroll
        برائے (صحیح l = 0; l < ne; ++l) {
            ((اعشاری *) dst)[l] = d * q8[l];
        }
    } ورنہ {
        static_assert(std::is_same_v<T, خالی>, "bad type");
    }
}

سانچہ <قسم_نام T, صحیح ne>
ساکن __dpct_inline__ خالی dequantize_V_q4_1(ثابت خالی * __restrict__ vx, خالی * __restrict__ dst, ثابت int64_t i0) {
    ثابت بلاک_q4_1 * x = (ثابت بلاک_q4_1 *) vx;

    ثابت int64_t ib    =  i0          /  QK4_1;
    ثابت صحیح     iqs   =  i0          % (QK4_1/2);
    ثابت صحیح     سرکاؤ = (i0 % QK4_1) / (QK4_1/2);

    صحیح q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_sycl_memcpy_1<ne>(&q, x[ib].qs + iqs);
    q >>= 4*سرکاؤ;
    q &= 0x0F0F0F0F;

    ثابت int8_t * q8 = (ثابت int8_t *) &q;

#اگر_تعریف_شدہ GGML_SYCL_F16
    اگر constexpr (std::is_same_v<T, sycl::half>) {
        ثابت sycl::half2 dm = x[ib].dm;
        ثابت sycl::half2 d  = sycl::half2(dm[0]);
        ثابت sycl::half2 m  = sycl::half2(dm[1]);

#ہدایت unroll
        برائے (صحیح l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * sycl::half2(q8[l0 + 0], q8[l0 + 1]) + m;
        }
    } ورنہ
#اگر_ختم // GGML_SYCL_F16
    اگر constexpr (std::is_same_v<T, اعشاری>) {
        ثابت sycl::float2 dm = (x[ib].dm).سانچہ convert<اعشاری, sycl::rounding_منوال::automatic>();

#ہدایت unroll
        برائے (صحیح l = 0; l < ne; ++l) {
            ((اعشاری *) dst)[l] = dm.x() * q8[l] + dm.y();
        }
    } ورنہ {
        static_assert(std::is_same_v<T, خالی>, "bad type");
    }
}

سانچہ <قسم_نام T, صحیح ne>
ساکن __dpct_inline__ خالی dequantize_V_q5_0(ثابت خالی * __restrict__ vx, خالی * __restrict__ dst, ثابت int64_t i0) {
    ثابت بلاک_q5_0 * x = (ثابت بلاک_q5_0 *) vx;

    ثابت int64_t ib    =  i0          /  QK5_0;
    ثابت صحیح     idq   =  i0          %  QK5_0;
    ثابت صحیح     iqs   =  i0          % (QK5_0/2);
    ثابت صحیح     سرکاؤ = (i0 % QK5_0) / (QK5_0/2);

    صحیح q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_sycl_memcpy_1<ne, 2>(&q, x[ib].qs + iqs);
    q >>= 4*سرکاؤ;
    q &= 0x0F0F0F0F;

    {
        صحیح qh;
        ggml_sycl_memcpy_1<ne, 2>(&qh, x[ib].qh);
#ہدایت unroll
        برائے (صحیح l = 0; l < ne; ++l) {
            q |= ((qh >> (idq + l)) & 0x00000001) << (8*l + 4);
        }
    }

    q = dpct::vectorized_ثنائی<sycl::char4>(q, 0x10101010, dpct::sub_sat());

    ثابت int8_t * q8 = (ثابت int8_t *) &q;

#اگر_تعریف_شدہ GGML_SYCL_F16
    اگر constexpr (std::is_same_v<T, sycl::half>) {
        ثابت sycl::half2 d = sycl::half2(x[ib].d);

#ہدایت unroll
        برائے (صحیح l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * sycl::half2(q8[l0 + 0], q8[l0 + 1]);
        }
    } ورنہ
#اگر_ختم // GGML_SYCL_F16
    اگر constexpr (std::is_same_v<T, اعشاری>) {
        ثابت اعشاری d = x[ib].d;

#ہدایت unroll
        برائے (صحیح l = 0; l < ne; ++l) {
            ((اعشاری *) dst)[l] = d * q8[l];
        }
    } ورنہ {
        static_assert(std::is_same_v<T, خالی>, "bad type");
    }
}

سانچہ <قسم_نام T, صحیح ne>
ساکن __dpct_inline__ خالی dequantize_V_q5_1(ثابت خالی * __restrict__ vx, خالی * __restrict__ dst, ثابت int64_t i0) {
    ثابت بلاک_q5_1 * x = (ثابت بلاک_q5_1 *) vx;

    ثابت int64_t ib    =  i0          /  QK5_1;
    ثابت صحیح     idq   =  i0          %  QK5_1;
    ثابت صحیح     iqs   =  i0          % (QK5_1/2);
    ثابت صحیح     سرکاؤ = (i0 % QK5_1) / (QK5_1/2);

    صحیح q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_sycl_memcpy_1<ne>(&q, x[ib].qs + iqs);
    q >>= 4*سرکاؤ;
    q &= 0x0F0F0F0F;

    {
        صحیح qh;
        ggml_sycl_memcpy_1<ne>(&qh, x[ib].qh);
#ہدایت unroll
        برائے (صحیح l = 0; l < ne; ++l) {
            q |= ((qh >> (idq + l)) & 0x00000001) << (8*l + 4);
        }
    }

    ثابت int8_t * q8 = (ثابت int8_t *) &q;

#اگر_تعریف_شدہ GGML_SYCL_F16
    اگر constexpr (std::is_same_v<T, sycl::half>) {
        ثابت sycl::half2 dm = x[ib].dm;
        ثابت sycl::half2 d  = sycl::half2(dm[0]);
        ثابت sycl::half2 m  = sycl::half2(dm[1]);

#ہدایت unroll
        برائے (صحیح l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * sycl::half2(q8[l0 + 0], q8[l0 + 1]) + m;
        }
    } ورنہ
#اگر_ختم // GGML_SYCL_F16
    اگر constexpr (std::is_same_v<T, اعشاری>) {
        ثابت sycl::float2 dm = (x[ib].dm).سانچہ convert<اعشاری, sycl::rounding_منوال::automatic>();

#ہدایت unroll
        برائے (صحیح l = 0; l < ne; ++l) {
            ((اعشاری *) dst)[l] = dm.x() * q8[l] + dm.y();
        }
    } ورنہ {
        static_assert(std::is_same_v<T, خالی>, "bad type");
    }
}

سانچہ <قسم_نام T, صحیح ne>
ساکن __dpct_inline__ خالی dequantize_V_q8_0(ثابت خالی * __restrict__ vx, خالی * __restrict__ dst, ثابت int64_t i0) {
    ثابت بلاک_q8_0 * x = (ثابت بلاک_q8_0 *) vx;

    ثابت int64_t ib  = i0 / QK8_0;
    ثابت صحیح     iqs = i0 % QK8_0;

    static_assert(ne % 2 == 0, "bad ne");
    int8_t qs[ne];
    ggml_sycl_memcpy_1<ne, 2>(qs, x[ib].qs + iqs);

#اگر_تعریف_شدہ GGML_SYCL_F16
    اگر constexpr (std::is_same<T, sycl::half>::value) {
        ثابت sycl::half2 d = sycl::half2(x[ib].d);

#ہدایت unroll
        برائے (صحیح l0 = 0; l0 < ne; l0 += 2) {
            ((sycl::half2 *) dst)[l0 / 2] = d * make_half2(qs[l0 + 0], qs[l0 + 1]);
        }
    } ورنہ
#اگر_ختم // GGML_SYCL_F16
    اگر constexpr (std::is_same<T, اعشاری>::value) {
        ثابت اعشاری d = x[ib].d;

#ہدایت unroll
        برائے (صحیح l = 0; l < ne; ++l) {
            ((اعشاری *) dst)[l] = d * qs[l];
        }
    } ورنہ {
        static_assert(std::is_same_v<T, خالی>, "unsupported type");
    }
}

سانچہ <صحیح type_K, صحیح D, صحیح nthreads, صحیح warp_سائز>
constexpr vec_dot_KQ_t get_vec_dot_KQ() {
    اگر constexpr (type_K == GGML_TYPE_F16) {
        واپس vec_dot_fattn_vec_KQ_f16<D, nthreads>;
    } ورنہ اگر constexpr (type_K == GGML_TYPE_Q4_0) {
        واپس vec_dot_fattn_vec_KQ_q4_0<D, nthreads, warp_سائز>;
    } ورنہ اگر constexpr (type_K == GGML_TYPE_Q4_1) {
        واپس vec_dot_fattn_vec_KQ_q4_1<D, nthreads, warp_سائز>;
    } ورنہ اگر constexpr (type_K == GGML_TYPE_Q5_0) {
        واپس vec_dot_fattn_vec_KQ_q5_0<D, nthreads, warp_سائز>;
    } ورنہ اگر constexpr (type_K == GGML_TYPE_Q5_1) {
        واپس vec_dot_fattn_vec_KQ_q5_1<D, nthreads, warp_سائز>;
    } ورنہ اگر constexpr (type_K == GGML_TYPE_Q8_0) {
        واپس vec_dot_fattn_vec_KQ_q8_0<D, nthreads, warp_سائز>;
    } ورنہ {
        static_assert(type_K == -1, "bad type");
        واپس خالی_اشاریہ;
    }
}

سانچہ <صحیح type_V, قسم_نام T, صحیح ne>
constexpr dequantize_V_t get_dequantize_V() {
    اگر constexpr (type_V == GGML_TYPE_F16) {
        واپس dequantize_V_f16<T, ne>;
    } ورنہ اگر constexpr (type_V == GGML_TYPE_Q4_0) {
        واپس dequantize_V_q4_0<T, ne>;
    } ورنہ اگر constexpr (type_V == GGML_TYPE_Q4_1) {
        واپس dequantize_V_q4_1<T, ne>;
    } ورنہ اگر constexpr (type_V == GGML_TYPE_Q5_0) {
        واپس dequantize_V_q5_0<T, ne>;
    } ورنہ اگر constexpr (type_V == GGML_TYPE_Q5_1) {
        واپس dequantize_V_q5_1<T, ne>;
    } ورنہ اگر constexpr (type_V == GGML_TYPE_Q8_0) {
        واپس dequantize_V_q8_0<T, ne>;
    } ورنہ {
        static_assert(type_V == -1, "bad type");
        واپس خالی_اشاریہ;
    }
}

سانچہ <صحیح ncols1, صحیح warp_سائز>
ساکن خالی flash_attn_نقاب_مقصد_KV_max(ثابت sycl::half2 * __restrict__ نقاب,
                                      صحیح * __restrict__ KV_max,
                                      ثابت صحیح ne30,
                                      ثابت صحیح s31,
                                      ثابت صحیح s33,
                                      صحیح *     buf_iw) {
    خودکار      شے_ct1 = sycl::ext::oneapi::this_work_شے::get_nd_شے<3>();
    ثابت صحیح ne31     = شے_ct1.get_گروہ_range(2);
    ثابت صحیح tid      = شے_ct1.get_local_id(2);
    ثابت صحیح sequence = شے_ct1.get_گروہ(1);
    ثابت صحیح jt       = شے_ct1.get_گروہ(2);

    نقاب += sequence*s33 + jt*ncols1*s31;

    اگر (tid < warp_سائز) {
        buf_iw[tid] = 1;
    }
    شے_ct1.barrier(sycl::access::fence_space::local_space);

    صحیح KV_max_sj = (ne30 - 1) * FATTN_KQ_STRIDE;
    برائے (; KV_max_sj >= 0; KV_max_sj -= FATTN_KQ_STRIDE) {
        صحیح all_inf = 1;

#ہدایت unroll
        برائے (صحیح j = 0; j < ncols1; ++j) {
            ثابت sycl::float2 tmp =
                نقاب[j * s31 + KV_max_sj / 2 + tid].سانچہ convert<اعشاری, sycl::rounding_منوال::automatic>();
            all_inf = all_inf && صحیح(sycl::isinf((اعشاری) (tmp.x()))) && صحیح(sycl::isinf((اعشاری) (tmp.y())));
        }

        all_inf = warp_reduce_all<warp_سائز>(all_inf);
        اگر (tid % warp_سائز == 0) {
            buf_iw[tid / warp_سائز] = all_inf;
        }
        شے_ct1.barrier(sycl::access::fence_space::local_space);
        all_inf = buf_iw[tid % warp_سائز];
        شے_ct1.barrier(sycl::access::fence_space::local_space);
        all_inf = warp_reduce_all<warp_سائز>(all_inf);

        اگر (!all_inf) {
            توڑو;
        }
    }

    // If the break in the loop was not triggered, KV_max_sj is now -FATTN_KQ_STRIDE.
    // If the break was triggered it's the lower edge of the tile with the first non-masked values.
    // In either case, walk back the decrementation by FATTN_KQ_STRIDE.
    KV_max_sj += FATTN_KQ_STRIDE;

    اگر (شے_ct1.get_local_id(2) != 0) {
        واپس;
    }

    KV_max[sequence*ne31 + jt] = KV_max_sj;
}

سانچہ <صحیح D, صحیح ncols1, صحیح ncols2>  // D == head size

ساکن خالی flash_attn_دھارا_k_fixup(اعشاری * __restrict__ dst,
                                      ثابت sycl::float2 * __restrict__ dst_fixup,
                                      ثابت صحیح ne01,
                                      ثابت صحیح ne02,
                                      ثابت صحیح ne03,
                                      ثابت صحیح ne11,
                                      ثابت صحیح ne12,
                                      ثابت صحیح nbatch_fa) {
    خودکار          شے_ct1 = sycl::ext::oneapi::this_work_شے::get_nd_شے<3>();
    constexpr صحیح ncols    = ncols1 * ncols2;

    ثابت صحیح bidx0 = شے_ct1.get_گروہ(2);
    ثابت صحیح j     = شے_ct1.get_گروہ(1);
    ثابت صحیح c     = شے_ct1.get_گروہ(0);
    ثابت صحیح jc    = j*ncols2 + c;
    ثابت صحیح tid   = شے_ct1.get_local_id(2);

    ثابت اعشاری * dst_fixup_ڈیٹا = ((ثابت اعشاری *) dst_fixup) + شے_ct1.get_گروہ_range(2) * (2 * 2 * ncols);

    ثابت صحیح gqa_تناسب = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.

    ثابت صحیح iter_k     = (ne11      + (nbatch_fa - 1)) / nbatch_fa;
    ثابت صحیح iter_j     = (ne01      + (ncols1    - 1)) / ncols1;
    ثابت صحیح iter_z_gqa = (gqa_تناسب + (ncols2    - 1)) / ncols2;

    ثابت صحیح kbc0 = int64_t(bidx0 + 0) * (iter_k * iter_j * iter_z_gqa * ne12 * ne03) / شے_ct1.get_گروہ_range(2);
    ثابت صحیح kbc0_stop =
        int64_t(bidx0 + 1) * (iter_k * iter_j * iter_z_gqa * ne12 * ne03) / شے_ct1.get_گروہ_range(2);

    ثابت منطقی did_not_have_any_ڈیٹا   = kbc0 == kbc0_stop;
    ثابت منطقی wrote_beginning_کا_tile = kbc0 % iter_k == 0;
    ثابت منطقی did_not_write_آخری      = kbc0/iter_k == kbc0_stop/iter_k && kbc0_stop % iter_k != 0;
    اگر (did_not_have_any_ڈیٹا || wrote_beginning_کا_tile || did_not_write_آخری) {
        واپس;
    }

    // z_KV == K/V head index, zt_gqa = Q head start index per K/V head, jt = token position start index
    ثابت صحیح sequence =  kbc0 /(iter_k*iter_j*iter_z_gqa*ne12);
    ثابت صحیح z_KV     = (kbc0 - iter_k*iter_j*iter_z_gqa*ne12 * sequence)/(iter_k*iter_j*iter_z_gqa);
    ثابت صحیح zt_gqa   = (kbc0 - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV)/(iter_k*iter_j);
    ثابت صحیح jt       = (kbc0 - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV - iter_k*iter_j * zt_gqa) / iter_k;

    ثابت صحیح zt_Q = z_KV*gqa_تناسب + zt_gqa*ncols2; // Global Q head start index.

    اگر (jt*ncols1 + j >= ne01 || zt_gqa*ncols2 + c >= gqa_تناسب) {
        واپس;
    }

    dst += sequence*ne02*ne01*D + jt*ne02*(ncols1*D) + zt_Q*D + (j*ne02 + c)*D + tid;

    // Load the partial result that needs a fixup:
    اعشاری dst_val = 0.0f;
    اعشاری max_val = 0.0f;
    اعشاری rowsum  = 0.0f;
    {
        dst_val = *dst;

        ثابت sycl::float2 tmp = dst_fixup[bidx0 * ncols + jc];
        max_val                = tmp.x();
        rowsum                 = tmp.y();
    }

    // Iterate over previous blocks and compute the combined results.
    // All SYCL blocks that get here must have a previous block that needs a fixup.
    صحیح bidx = bidx0 - 1;
    صحیح kbc_stop = kbc0;
    جب(سچ) {
        ثابت صحیح kbc = int64_t(bidx) * (iter_k * iter_j * iter_z_gqa * ne12 * ne03) / شے_ct1.get_گروہ_range(2);
        اگر (kbc == kbc_stop) { // Did not have any data.
            bidx--;
            kbc_stop = kbc;
            جاری;
        }

        ثابت اعشاری dst_جوڑو = dst_fixup_ڈیٹا[bidx*ncols*D + jc*D + tid];

        ثابت sycl::float2 tmp = dst_fixup[(شے_ct1.get_گروہ_range(2) + bidx) * ncols + jc];

        // Scale the current and new value accumulators depending on the max. values.
        ثابت اعشاری max_val_new = sycl::fmax(max_val, tmp.x());

        ثابت اعشاری diff_val = max_val - max_val_new;
        ثابت اعشاری diff_جوڑو = tmp.x() - max_val_new;

        ثابت اعشاری scale_val = diff_val >= SOFTMAX_FTZ_حد ? sycl::native::قوت_نما(diff_val) : 0.0f;
        ثابت اعشاری scale_جوڑو = diff_جوڑو >= SOFTMAX_FTZ_حد ? sycl::native::قوت_نما(diff_جوڑو) : 0.0f;

        dst_val = scale_val*dst_val + scale_جوڑو*dst_جوڑو;
        rowsum  = scale_val * rowsum + scale_جوڑو * tmp.y();

        max_val = max_val_new;

        // If this block started in a previous tile we are done and don't need to combine additional partial results.
        اگر (kbc % iter_k == 0 || kbc/iter_k < kbc0/iter_k) {
            توڑو;
        }
        bidx--;
        kbc_stop = kbc;
    }

    // Write back final result:
    *dst = dst_val / rowsum;
}

سانچہ <صحیح D>  // D == head size

ساکن خالی flash_attn_combine_results(ثابت اعشاری * __restrict__ VKQ_parts,
                                       ثابت sycl::float2 * __restrict__ VKQ_meta,
                                       اعشاری * __restrict__ dst,
                                       ثابت صحیح parallel_بلاکیں,
                                       uint8_t * dpct_local) {
    // Dimension 0: threadIdx.x
    // Dimension 1: blockIdx.x
    // Dimension 2: blockIdx.y
    // Dimension 3: blockIdx.z
    // Memory layout is permuted with [0, 2, 1, 3]

    خودکار      شے_ct1 = sycl::ext::oneapi::this_work_شے::get_nd_شے<3>();
    ثابت صحیح ne01     = شے_ct1.get_گروہ_range(2);
    ثابت صحیح ne02     = شے_ct1.get_گروہ_range(1);

    ثابت صحیح ستون      = شے_ct1.get_گروہ(2);
    ثابت صحیح سر     = شے_ct1.get_گروہ(1);
    ثابت صحیح sequence = شے_ct1.get_گروہ(0);

    ثابت صحیح j_dst_unrolled = (sequence*ne01 + ستون)*ne02 + سر;

    VKQ_parts += j_dst_unrolled * parallel_بلاکیں*D;
    VKQ_meta  += j_dst_unrolled * parallel_بلاکیں;
    dst       += j_dst_unrolled *                 D;

    ثابت صحیح tid = شے_ct1.get_local_id(2);
    __builtin_assume(tid < D);

    خودکار meta = (sycl::float2 *) dpct_local;
    برائے (صحیح i = tid; i < 2*parallel_بلاکیں; i += D) {
        ((اعشاری *) meta)[i] = ((ثابت اعشاری *)VKQ_meta) [i];
    }

    شے_ct1.barrier(sycl::access::fence_space::local_space);

    اعشاری kqmax = meta[0].x();
    برائے (صحیح l = 1; l < parallel_بلاکیں; ++l) {
        kqmax = sycl::max(kqmax, meta[l].x());
    }

    اعشاری VKQ_numerator   = 0.0f;
    اعشاری VKQ_denominator = 0.0f;
    برائے (صحیح l = 0; l < parallel_بلاکیں; ++l) {
        ثابت اعشاری KQ_max_scale = sycl::native::قوت_نما(meta[l].x() - kqmax);

        VKQ_numerator   += KQ_max_scale * VKQ_parts[l*D + tid];
        VKQ_denominator += KQ_max_scale * meta[l].y();
    }

    dst[tid] = VKQ_numerator / VKQ_denominator;
}

// Carries the sub-group size (and grf_size_automatic when use_large_grf) as
// kernel properties via get(properties_tag), in place of the
// reqd_sub_group_size attribute and the deprecated parallel_for overload.
سانچہ <قسم_نام KernelFunc, uint32_t سائز_3, منطقی UseGrf>
ساخت props_kernel {
    KernelFunc kernel;

    خالی عامل()(sycl::nd_شے<3> شے) ثابت {
        kernel(شے);
    }

    خودکار get(sycl::ext::oneapi::experimental::properties_لیبل) ثابت {
        اگر constexpr (UseGrf) {
            واپس sycl::ext::oneapi::experimental::properties{
                sycl::ext::intel::experimental::grf_سائز_automatic,
                sycl::ext::oneapi::experimental::sub_گروہ_سائز<سائز_3> };
        } ورنہ {
            واپس sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::sub_گروہ_سائز<سائز_3> };
        }
    }
};

سانچہ <fattn_kernel_t fattn_kernel, صحیح warp_سائز, منطقی use_large_grf = جھوٹ>
ساکن خالی lauch_kernel(
    dpct::dim3 گروہ_range,
    dpct::dim3 local_range,
    قطار_ptr q,
    غیرعلامتی صحیح local_mem_سائز_2,
    ثابت حرف* __restrict__ Q,
    ثابت حرف* __restrict__ K,
    ثابت حرف* __restrict__ V,
    ثابت حرف* __restrict__ نقاب,
    ثابت حرف* __restrict__ sinks,
    ثابت صحیح* __restrict__ KV_max,
    اعشاری* __restrict__ dst,
    sycl::float2* __restrict__ dst_meta,
    ثابت اعشاری scale,
    ثابت اعشاری max_bias,
    ثابت اعشاری m0,
    ثابت اعشاری m1,
    ثابت uint32_t n_سر_log2,
    ثابت اعشاری logit_softcap,
    ثابت int32_t ne00,
    ثابت sycl::uint3 ne01,
    ثابت int32_t ne02,
    ثابت int32_t ne03,
    ثابت int32_t nb01,
    ثابت int32_t nb02,
    ثابت int32_t nb03,
    ثابت int32_t ne10,
    ثابت int32_t ne11,
    ثابت int32_t ne12,
    ثابت int32_t ne13,
    ثابت int32_t nb11,
    ثابت int32_t nb12,
    ثابت int64_t nb13,
    ثابت int32_t nb21,
    ثابت int32_t nb22,
    ثابت int64_t nb23,
    ثابت int32_t ne31,
    ثابت int32_t ne32,
    ثابت int32_t ne33,
    ثابت int32_t nb31,
    ثابت int32_t nb32,
    ثابت int64_t nb33) {
    GGML_غیر_مستعمل(local_mem_سائز_2);

    ثابت خودکار rng = sycl::nd_range<3>(
        static_cast<sycl::range<3>>(گروہ_range * local_range),
        static_cast<sycl::range<3>>(local_range));

    ثابت خودکار kernel = [=](sycl::nd_شے<3> شے_ct1) {
        GGML_غیر_مستعمل(شے_ct1);
        fattn_kernel(Q, K, V, نقاب, sinks, KV_max, dst, dst_meta, scale,
                     max_bias, m0, m1, n_سر_log2, logit_softcap, ne00,
                     ne01, ne02, ne03, nb01, nb02, nb03, ne10, ne11,
                     ne12, ne13, nb11, nb12, nb13, nb21, nb22, nb23,
                     ne31, ne32, ne33, nb31, nb32, nb33);
    };

    q->submit([&](sycl::handler &cgh) {
        // grf_size_automatic lets the compiler use the large register file when the kernel needs it. grf_size<256> would say the same but is undefined on devices that do not have it.
        cgh.parallel_for(rng, props_kernel<decltype(kernel), warp_سائز, use_large_grf>{kernel});
    });
}

// use_large_grf lets the compiler choose the large register file. Opt in where spilling limits the kernel, not thread occupancy.
سانچہ <صحیح DV, صحیح ncols1, صحیح ncols2, fattn_kernel_t fattn_kernel, صحیح warp_سائز, منطقی use_large_grf = جھوٹ>
خالی launch_fattn(
    ggml_backend_sycl_context & ctx, ggml_tensor * dst, ثابت صحیح nwarps, ثابت size_t nbytes_shared,
    ثابت صحیح nbatch_fa, ثابت منطقی need_f16_K, ثابت منطقی need_f16_V, ثابت منطقی دھارا_k) {

    constexpr صحیح ncols = ncols1 * ncols2;

    ثابت ggml_tensor * Q = dst->src[0];
    ثابت ggml_tensor * K = dst->src[1];
    ثابت ggml_tensor * V = dst->src[2];

    ثابت منطقی V_is_K_view = V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));

    ثابت ggml_tensor * نقاب  = dst->src[3];
    ثابت ggml_tensor * sinks = dst->src[4];

    ggml_tensor * KQV = dst;

    GGML_ASSERT(Q->type == GGML_TYPE_F32);
    GGML_ASSERT(KQV->type == GGML_TYPE_F32);

    GGML_ASSERT(Q->nb[0] == ggml_element_سائز(Q));
    GGML_ASSERT(K->nb[0] == ggml_element_سائز(K));
    GGML_ASSERT(V->nb[0] == ggml_element_سائز(V));

    GGML_ASSERT(!نقاب || نقاب->type == GGML_TYPE_F16);

    ggml_sycl_pool & pool = ctx.pool();
    dpct::قطار_ptr  main_دھارا = ctx.stream();
    ثابت صحیح id  = ggml_sycl_get_device();
    ثابت صحیح nsm = ggml_sycl_info().devices[id].nsm;

    ثابت ggml_sycl_fattn_extra  extra = ggml_sycl_fattn_get_extra(dst);
    ggml_sycl_pool_alloc<صحیح>    KV_max(pool);
    ggml_sycl_pool_alloc<اعشاری>  dst_tmp(pool);
    ggml_sycl_pool_alloc<sycl::float2> dst_tmp_meta(pool);

    ثابت حرف * K_ڈیٹا = (ثابت حرف *) K->ڈیٹا;
    size_t nb11 = K->nb[1];
    size_t nb12 = K->nb[2];
    size_t nb13 = K->nb[3];

    ثابت حرف * V_ڈیٹا = (ثابت حرف *) V->ڈیٹا;
    size_t nb21 = V->nb[1];
    size_t nb22 = V->nb[2];
    size_t nb23 = V->nb[3];

    اگر (need_f16_K && K->type != GGML_TYPE_F16) {
        ثابت size_t bs = ggml_blck_سائز(K->type);
        ثابت size_t ts = ggml_type_سائز(K->type);

        GGML_ASSERT(extra.K_بفر_ptr);
        sycl::half * K_f16_ptr = (sycl::half *) extra.K_بفر_ptr;
        اگر (ggml_is_contiguously_allocated(K)) {
            مقصد_fp16_sycl_t مقصد_fp16 = ggml_get_مقصد_fp16_sycl(K->type, dst);
            مقصد_fp16(K_ڈیٹا, K_f16_ptr, ggml_nelements(K), main_دھارا);

            nb11 = nb11 * bs * حجم(sycl::half) / ts;
            nb12 = nb12 * bs * حجم(sycl::half) / ts;
            nb13 = nb13 * bs * حجم(sycl::half) / ts;
        } ورنہ {
            GGML_ASSERT(K->nb[0] == ts);
            مقصد_fp16_nc_sycl_t مقصد_fp16 = ggml_get_مقصد_fp16_nc_sycl(K->type);
            ثابت int64_t s01 = nb11 / ts;
            ثابت int64_t s02 = nb12 / ts;
            ثابت int64_t s03 = nb13 / ts;
            مقصد_fp16(K_ڈیٹا, K_f16_ptr, K->ne[0], K->ne[1], K->ne[2], K->ne[3], s01, s02, s03, main_دھارا);

            nb11 = K->ne[0] * حجم(sycl::half);
            nb12 = K->ne[1] * nb11;
            nb13 = K->ne[2] * nb12;
        }
        K_ڈیٹا = (حرف *) K_f16_ptr;
    }

    اگر (need_f16_V && V->type != GGML_TYPE_F16) {
        اگر (V_is_K_view) {
            V_ڈیٹا = K_ڈیٹا;
            nb21   = nb11;
            nb22   = nb12;
            nb23   = nb13;
        } ورنہ {
            ثابت size_t bs = ggml_blck_سائز(V->type);
            ثابت size_t ts = ggml_type_سائز(V->type);

            GGML_ASSERT(extra.V_بفر_ptr);
            sycl::half * V_f16_ptr = (sycl::half *) extra.V_بفر_ptr;
            اگر (ggml_is_contiguously_allocated(V)) {
                مقصد_fp16_sycl_t مقصد_fp16 = ggml_get_مقصد_fp16_sycl(V->type, dst);
                مقصد_fp16(V_ڈیٹا, V_f16_ptr, ggml_nelements(V), main_دھارا);
                V_ڈیٹا = (حرف *) V_f16_ptr;

                nb21 = nb21 * bs * حجم(sycl::half) / ts;
                nb22 = nb22 * bs * حجم(sycl::half) / ts;
                nb23 = nb23 * bs * حجم(sycl::half) / ts;
            } ورنہ {
                GGML_ASSERT(V->nb[0] == ts);
                مقصد_fp16_nc_sycl_t مقصد_fp16 = ggml_get_مقصد_fp16_nc_sycl(V->type);
                ثابت int64_t s01 = nb21 / ts;
                ثابت int64_t s02 = nb22 / ts;
                ثابت int64_t s03 = nb23 / ts;
                مقصد_fp16(V_ڈیٹا, V_f16_ptr, V->ne[0], V->ne[1], V->ne[2], V->ne[3], s01, s02, s03, main_دھارا);

                nb21 = V->ne[0] * حجم(sycl::half);
                nb22 = V->ne[1] * nb21;
                nb23 = V->ne[2] * nb22;
            }
            V_ڈیٹا = (حرف *) V_f16_ptr;
        }
    }

    ثابت صحیح ntiles_x     = ((Q->ne[1] + ncols1 - 1) / ncols1);
    ثابت صحیح gqa_تناسب    = Q->ne[2] / K->ne[2];
    ثابت صحیح ntiles_z_gqa = ((gqa_تناسب + ncols2 - 1) / ncols2);
    ثابت صحیح ntiles_کل = ntiles_x * ntiles_z_gqa * K->ne[2] * Q->ne[3];

    // Optional optimization where the mask is scanned to determine whether part of the calculation can be skipped.
    // Only worth the overhead if there is at lease one FATTN_KQ_STRIDE x FATTN_KQ_STRIDE square to be skipped or
    //     multiple sequences of possibly different lengths.
    اگر (نقاب && K->ne[1] % FATTN_KQ_STRIDE == 0 && (Q->ne[1] >= 1024 || Q->ne[3] > 1)) {
        ثابت صحیح s31 = نقاب->nb[1] / حجم(sycl::half2);
        ثابت صحیح s33 = نقاب->nb[3] / حجم(sycl::half2);

        ثابت dpct::dim3 بلاکیں_عدد_KV_max(ntiles_x, Q->ne[3], 1);
        ثابت dpct::dim3 بلاک_dim_KV_max(FATTN_KQ_STRIDE / 2, 1, 1);

        ثابت صحیح ne_KV_max = بلاکیں_عدد_KV_max.x*بلاکیں_عدد_KV_max.y;
        ثابت صحیح iter_k = K->ne[1] / FATTN_KQ_STRIDE;

        KV_max.alloc(ne_KV_max);
        {
            dpct::has_capability_or_fail(main_دھارا->get_device(), { sycl::aspect::fp16 });

            main_دھارا->submit([&](sycl::handler & cgh) {
                sycl::local_accessor<صحیح, 1> buf_iw_acc_ct1(sycl::range<1>(warp_سائز), cgh);

                خودکار نقاب_ڈیٹا_ct0  = (ثابت sycl::half2 *) نقاب->ڈیٹا;
                خودکار KV_max_ptr_ct1 = KV_max.ptr;

                cgh.parallel_for(sycl::nd_range<3>(بلاکیں_عدد_KV_max * بلاک_dim_KV_max, بلاک_dim_KV_max),
                                 [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(warp_سائز)]] {
                                     GGML_غیر_مستعمل(شے_ct1);
                                     flash_attn_نقاب_مقصد_KV_max<ncols1, warp_سائز>(
                                         نقاب_ڈیٹا_ct0, KV_max_ptr_ct1, iter_k, s31, s33,
                                         buf_iw_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                                 });
            });
        }
        SYCL_CHECK(0);
    }

    ثابت dpct::dim3 بلاک_dim(warp_سائز, nwarps, 1);

    // Max. number of active blocks limited by occupancy.
    صحیح max_بلاکیں_فی_sm = ggml_sycl_info().devices[id].max_wg_فی_cu;
    صحیح parallel_بلاکیں = max_بلاکیں_فی_sm;
    dpct::dim3 بلاکیں_عدد;
    اگر (دھارا_k) {
        // For short contexts it can be faster to have the SMs work on whole tiles because this lets us skip the fixup.
        ثابت صحیح max_بلاکیں = max_بلاکیں_فی_sm*nsm;
        ثابت صحیح nblocks_دھارا_k = max_بلاکیں;
        ثابت منطقی use_دھارا_k = سچ;

        بلاکیں_عدد.x = use_دھارا_k ? nblocks_دھارا_k : ntiles_کل;
        بلاکیں_عدد.y = 1;
        بلاکیں_عدد.z = 1;

        اگر (ntiles_کل % بلاکیں_عدد.x != 0) { // Fixup is only needed if the SMs work on fractional tiles.
            dst_tmp_meta.alloc((size_t(بلاکیں_عدد.x) * ncols * (2 + DV/2)));
        }
    } ورنہ {
        ثابت صحیح ntiles_KQ = (K->ne[1] + nbatch_fa - 1) / nbatch_fa; // Max. number of parallel blocks limited by tensor size.

        // parallel_blocks must not be larger than what the tensor size allows:
        parallel_بلاکیں = std::min(parallel_بلاکیں, ntiles_KQ);
        // todo fix the hard code change
        // parallel_blocks = ntiles_KQ;

        // If ntiles_total % blocks_per_wave != 0 then some efficiency is lost due to tail effects.
        // Test whether parallel_blocks can be set to a higher value for better efficiency.
        ثابت صحیح بلاکیں_فی_wave = nsm * max_بلاکیں_فی_sm;
        صحیح nwaves_بہترین = 0;
        صحیح efficiency_percent_بہترین = 0;
        برائے (صحیح parallel_بلاکیں_test = parallel_بلاکیں; parallel_بلاکیں_test <= ntiles_KQ; ++parallel_بلاکیں_test) {
            ثابت صحیح nblocks_کل = ntiles_کل * parallel_بلاکیں_test;
            ثابت صحیح nwaves = (nblocks_کل + بلاکیں_فی_wave - 1) / بلاکیں_فی_wave;
            ثابت صحیح efficiency_percent = 100 * nblocks_کل / (nwaves*بلاکیں_فی_wave);

            // Stop trying configurations with more waves if we already have good efficiency to avoid excessive overhead.
            اگر (efficiency_percent_بہترین >= 95 && nwaves > nwaves_بہترین) {
                توڑو;
            }

            اگر (efficiency_percent > efficiency_percent_بہترین) {
                nwaves_بہترین = nwaves;
                efficiency_percent_بہترین = efficiency_percent;
                parallel_بلاکیں = parallel_بلاکیں_test;
            }
        }

        بلاکیں_عدد.x = ntiles_x;
        بلاکیں_عدد.y = parallel_بلاکیں;
        بلاکیں_عدد.z = ntiles_z_gqa*K->ne[2]*Q->ne[3];

        اگر (parallel_بلاکیں > 1) {
            dst_tmp.alloc(parallel_بلاکیں*ggml_nelements(KQV));
            dst_tmp_meta.alloc(parallel_بلاکیں*ggml_nrows(KQV));
        }
    }

    اعشاری scale         = 1.0f;
    اعشاری max_bias      = 0.0f;
    اعشاری logit_softcap = 0.0f;

    حافظہ_نقل(&scale,         (ثابت اعشاری *) KQV->op_params + 0, حجم(اعشاری));
    حافظہ_نقل(&max_bias,      (ثابت اعشاری *) KQV->op_params + 1, حجم(اعشاری));
    حافظہ_نقل(&logit_softcap, (ثابت اعشاری *) KQV->op_params + 2, حجم(اعشاری));

    اگر (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    ثابت uint32_t n_سر      = Q->ne[2];
    ثابت uint32_t n_سر_log2 = 1u << uint32_t(floorf(log2f(اعشاری(n_سر))));

    ثابت اعشاری m0 = powf(2.0f, -(max_bias       ) / n_سر_log2);
    ثابت اعشاری m1 = powf(2.0f, -(max_bias / 2.0f) / n_سر_log2);

    // TODO other tensor dimensions after removal of WMMA kernel:
    ثابت sycl::uint3 ne01 = init_fastdiv_اقدار(Q->ne[1]);

    GGML_ASSERT(بلاک_dim.x % warp_سائز == 0);

    lauch_kernel<fattn_kernel, warp_سائز, use_large_grf>(
        بلاکیں_عدد, بلاک_dim, main_دھارا, (غیرعلامتی صحیح) nbytes_shared, (ثابت حرف *) Q->ڈیٹا, K_ڈیٹا, V_ڈیٹا,
        نقاب ? ((ثابت حرف *) نقاب->ڈیٹا) : خالی_اشاریہ, sinks ? ((ثابت حرف *) sinks->ڈیٹا) : خالی_اشاریہ, KV_max.ptr,
        !دھارا_k && parallel_بلاکیں > 1 ? dst_tmp.ptr : (اعشاری *) KQV->ڈیٹا, (sycl::float2 *)dst_tmp_meta.ptr, scale, max_bias, m0, m1,
        n_سر_log2, logit_softcap, Q->ne[0], ne01, Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3], K->ne[0],
        K->ne[1], K->ne[2], K->ne[3], nb11, nb12, nb13, nb21, nb22, nb23, نقاب ? نقاب->ne[1] : 0,
        نقاب ? نقاب->ne[2] : 0, نقاب ? نقاب->ne[3] : 0, نقاب ? نقاب->nb[1] : 0, نقاب ? نقاب->nb[2] : 0,
        نقاب ? نقاب->nb[3] : 0);
    SYCL_CHECK(0);

    اگر (دھارا_k) {
        اگر (ntiles_کل % بلاکیں_عدد.x != 0) { // Fixup is only needed if the SMs work on fractional tiles.
            ثابت dpct::dim3 بلاک_dim_combine(DV, 1, 1);
            ثابت dpct::dim3 بلاکیں_عدد_combine = { بلاکیں_عدد.x, ncols1, ncols2 };

            main_دھارا->submit([&](sycl::handler & cgh) {
                خودکار KQV_ڈیٹا_ct0         = (اعشاری *) KQV->ڈیٹا;
                خودکار dst_tmp_meta_ptr_ct1 = dst_tmp_meta.ptr;
                خودکار Q_ne_ct2             = Q->ne[1];
                خودکار Q_ne_ct3             = Q->ne[2];
                خودکار Q_ne_ct4             = Q->ne[3];
                خودکار K_ne_ct5             = K->ne[1];
                خودکار K_ne_ct6             = K->ne[2];

                cgh.parallel_for(sycl::nd_range<3>(بلاکیں_عدد_combine * بلاک_dim_combine, بلاک_dim_combine),
                                 [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(warp_سائز)]] {
                                     GGML_غیر_مستعمل(شے_ct1);
                                     flash_attn_دھارا_k_fixup<DV, ncols1, ncols2>(KQV_ڈیٹا_ct0, dst_tmp_meta_ptr_ct1,
                                                                                   Q_ne_ct2, Q_ne_ct3, Q_ne_ct4,
                                                                                   K_ne_ct5, K_ne_ct6, nbatch_fa);
                                 });
            });
        }
    } ورنہ اگر (parallel_بلاکیں > 1) {
        ثابت dpct::dim3 بلاک_dim_combine(DV, 1, 1);
        ثابت dpct::dim3 بلاکیں_عدد_combine(Q->ne[1], Q->ne[2], Q->ne[3]);
        ثابت size_t     nbytes_shared_combine = parallel_بلاکیں * حجم(sycl::float2);
        main_دھارا->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(sycl::range<1>(nbytes_shared_combine), cgh);

            خودکار dst_tmp_ptr_ct0      = dst_tmp.ptr;
            خودکار dst_tmp_meta_ptr_ct1 = dst_tmp_meta.ptr;
            خودکار KQV_ڈیٹا_ct2         = (اعشاری *) KQV->ڈیٹا;

            cgh.parallel_for(sycl::nd_range<3>(بلاکیں_عدد_combine * بلاک_dim_combine, بلاک_dim_combine),
                             [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(warp_سائز)]] {
                                 GGML_غیر_مستعمل(شے_ct1);
                                 flash_attn_combine_results<DV>(
                                     dst_tmp_ptr_ct0, dst_tmp_meta_ptr_ct1, KQV_ڈیٹا_ct2, parallel_بلاکیں,
                                     dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                             });
        });
    }
    SYCL_CHECK(0);
}
