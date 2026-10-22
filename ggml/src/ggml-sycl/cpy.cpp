#شامل "cpy.hpp"

#شامل <float.h>
#شامل <vector>

#شامل "dequantize.hpp"
#شامل "ggml-sycl/common.hpp"
#شامل "ggml-sycl/presets.hpp"
#شامل "ggml.h"

ساکن خالی cpy_1_f32_f32(ثابت حرف * cxi, حرف * cdsti) {
    ثابت اعشاری * xi   = (ثابت اعشاری *) cxi;
    اعشاری *       dsti = (اعشاری *) cdsti;

    *dsti = *xi;
}

ساکن خالی cpy_1_f32_f16(ثابت حرف * cxi, حرف * cdsti) {
    ثابت اعشاری * xi   = (ثابت اعشاری *) cxi;
    sycl::half *  dsti = (sycl::half *) cdsti;

    *dsti = sycl::vec<اعشاری, 1>(*xi).convert<sycl::half, sycl::rounding_منوال::automatic>()[0];
}

ساکن خالی cpy_1_f16_f16(ثابت حرف * cxi, حرف * cdsti) {
    ثابت sycl::half * xi   = (ثابت sycl::half *) cxi;
    sycl::half *       dsti = (sycl::half *) cdsti;

    *dsti = *xi;
}

ساکن خالی cpy_1_f16_f32(ثابت حرف * cxi, حرف * cdsti) {
    ثابت sycl::half * xi   = (ثابت sycl::half *) cxi;
    اعشاری *            dsti = (اعشاری *) cdsti;

    *dsti = *xi;
}

ساکن خالی cpy_1_i16_i16(ثابت حرف * cxi, حرف * cdsti) {
    ثابت int16_t * xi   = (ثابت int16_t *) cxi;
    int16_t *       dsti = (int16_t *) cdsti;

    *dsti = *xi;
}

ساکن خالی cpy_1_i32_i32(ثابت حرف * cxi, حرف * cdsti) {
    ثابت int32_t * xi   = (ثابت int32_t *) cxi;
    int32_t *       dsti = (int32_t *) cdsti;

    *dsti = *xi;
}

ساکن خالی cpy_1_f32_i32(ثابت حرف * cxi, حرف * cdsti) {
    ثابت اعشاری * xi     = (ثابت اعشاری *) cxi;
    int32_t *     dsti   = (int32_t *) cdsti;

    *dsti = (int32_t) *xi;
}

ساکن خالی cpy_1_i32_f32(ثابت حرف * cxi, حرف * cdsti) {
    ثابت int32_t * xi   = (ثابت int32_t *) cxi;
    اعشاری *         dsti = (اعشاری *) cdsti;

    *dsti = (اعشاری) *xi;
}

#اگر_تعریف_شدہ GGML_SYCL_HAS_BF16
ساکن خالی cpy_1_f32_bf16(ثابت حرف * cxi, حرف * cdsti) {
    ثابت اعشاری * xi   = (ثابت اعشاری *) cxi;
    sycl::ext::oneapi::bfloat16 *  dsti = (sycl::ext::oneapi::bfloat16 *) cdsti;

    *dsti = sycl::ext::oneapi::bfloat16(*xi);
}

ساکن خالی cpy_1_bf16_f32(ثابت حرف * cxi, حرف * cdsti) {
    ثابت sycl::ext::oneapi::bfloat16 * xi   = (ثابت sycl::ext::oneapi::bfloat16 *) cxi;
    اعشاری *            dsti = (اعشاری *) cdsti;

    *dsti = static_cast<اعشاری>(*xi);
}

ساکن خالی cpy_1_bf16_bf16(ثابت حرف * cxi, حرف * cdsti) {
    ثابت sycl::ext::oneapi::bfloat16 * xi   = (ثابت sycl::ext::oneapi::bfloat16 *) cxi;
    sycl::ext::oneapi::bfloat16 *       dsti = (sycl::ext::oneapi::bfloat16 *) cdsti;

    *dsti = *xi;
}

ساکن خالی cpy_1_f16_bf16(ثابت حرف * cxi, حرف * cdsti) {
    ثابت sycl::half * xi   = (ثابت sycl::half *) cxi;
    sycl::ext::oneapi::bfloat16 *  dsti = (sycl::ext::oneapi::bfloat16 *) cdsti;

    *dsti = sycl::ext::oneapi::bfloat16(static_cast<اعشاری>(*xi));
}

ساکن خالی cpy_1_bf16_f16(ثابت حرف * cxi, حرف * cdsti) {
    ثابت sycl::ext::oneapi::bfloat16 * xi   = (ثابت sycl::ext::oneapi::bfloat16 *) cxi;
    sycl::half *            dsti = (sycl::half *) cdsti;

    *dsti = sycl::half(static_cast<اعشاری>(*xi));
}
#اگر_ختم

سانچہ <cpy_kernel_t cpy_1>
ساکن خالی cpy_f32_f16(ثابت حرف * cx, حرف * cdst, ثابت صحیح ne, ثابت صحیح ne00, ثابت صحیح ne01, ثابت صحیح ne02,
                        ثابت صحیح nb00, ثابت صحیح nb01, ثابت صحیح nb02, ثابت صحیح nb03, ثابت صحیح ne10, ثابت صحیح ne11,
                        ثابت صحیح ne12, ثابت صحیح nb10, ثابت صحیح nb11, ثابت صحیح nb12, ثابت صحیح nb13,
                        ثابت sycl::nd_شے<3> & شے_ct1) {
    ثابت صحیح i = شے_ct1.get_local_range(2) * شے_ct1.get_گروہ(2) + شے_ct1.get_local_id(2);

    اگر (i >= ne) {
        واپس;
    }

    // determine indices i02/i12, i01/i11, i00/i10 as a function of index i of flattened tensor
    // then combine those indices with the corresponding byte offsets to get the total offsets
    ثابت صحیح i03      = i / (ne00 * ne01 * ne02);
    ثابت صحیح i02      = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    ثابت صحیح i01      = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
    ثابت صحیح i00      = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
    ثابت صحیح x_انحراف = i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;

    ثابت صحیح i13        = i / (ne10 * ne11 * ne12);
    ثابت صحیح i12        = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
    ثابت صحیح i11        = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
    ثابت صحیح i10        = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
    ثابت صحیح dst_انحراف = i10 * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

    cpy_1(cx + x_انحراف, cdst + dst_انحراف);
}


/* quantized type same copy */
سانچہ<قسم_نام T>
ساکن خالی cpy_blck_q_q(ثابت حرف * cxi, حرف * cdsti) {
    ثابت T * xi = (ثابت T *) cxi;
    T * dsti = (T *) cdsti;
    *dsti = *xi;
}


ساکن خالی cpy_blck_q8_0_f32(ثابت حرف * cxi, حرف * cdsti) {
    اعشاری * cdstf = (اعشاری *) (cdsti);

    برائے (صحیح j = 0; j < QK8_0; j += 2) {
        dfloat2 dq;
        dequantize_q8_0(cxi, 0, j, dq);
        *(cdstf + j)     = dq.x();
        *(cdstf + j + 1) = dq.y();
    }
}

ساکن خالی cpy_blck_q2_0_f32(ثابت حرف * cxi, حرف * cdsti) {
    ثابت بلاک_q2_0 * xi = (ثابت بلاک_q2_0 *) cxi;
    اعشاری * cdstf = (اعشاری *) cdsti;

    ثابت اعشاری d = xi->d;

    برائے (صحیح j = 0; j < QK2_0; ++j) {
        ثابت صحیح byte_اشاریہ = j / 4;
        ثابت صحیح ثنائی_ہندسہ_انحراف = (j % 4) * 2;
        ثابت صحیح q = (xi->qs[byte_اشاریہ] >> ثنائی_ہندسہ_انحراف) & 0x3;
        cdstf[j] = (اعشاری) (q - 1) * d;
    }
}



سانچہ <dequantize_kernel_t dequant, صحیح qk> ساکن خالی cpy_blck_q_f32(ثابت حرف * cxi, حرف * cdsti) {
    اعشاری * cdstf = (اعشاری *) (cdsti);

    برائے (صحیح j = 0; j < qk / 2; j++) {
        dfloat2 dq;
        dequant(cxi, 0, j, dq);
        *(cdstf + j)          = dq.x();
        *(cdstf + j + qk / 2) = dq.y();
    }
}


سانچہ <قسم_نام T, صحیح qk>
ساکن خالی cpy_q_q(ثابت حرف * cx, حرف * cdst, ثابت صحیح ne, ثابت صحیح ne00, ثابت صحیح ne01, ثابت صحیح ne02,
                      ثابت صحیح nb00, ثابت صحیح nb01, ثابت صحیح nb02, ثابت صحیح nb03, ثابت صحیح ne10, ثابت صحیح ne11,
                      ثابت صحیح ne12, ثابت صحیح nb10, ثابت صحیح nb11, ثابت صحیح nb12, ثابت صحیح nb13,
                      ثابت sycl::nd_شے<3> & شے_ct1) {
    ثابت صحیح i = (شے_ct1.get_local_range(2) * شے_ct1.get_گروہ(2) + شے_ct1.get_local_id(2)) * qk;

    اگر (i >= ne) {
        واپس;
    }

    ثابت صحیح i03      = i / (ne00 * ne01 * ne02);
    ثابت صحیح i02      = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    ثابت صحیح i01      = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
    ثابت صحیح i00      = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
    ثابت صحیح x_انحراف = (i00 / qk) * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;


    ثابت صحیح i13        = i / (ne10 * ne11 * ne12);
    ثابت صحیح i12        = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
    ثابت صحیح i11        = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
    ثابت صحیح i10        = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
    ثابت صحیح dst_انحراف = (i10 / qk) * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

    cpy_blck_q_q<T>(cx + x_انحراف, cdst + dst_انحراف);
}

سانچہ <cpy_kernel_t cpy_blck, صحیح qk>
ساکن خالی cpy_f32_q(ثابت حرف * cx, حرف * cdst, ثابت صحیح ne, ثابت صحیح ne00, ثابت صحیح ne01, ثابت صحیح ne02,
                      ثابت صحیح nb00, ثابت صحیح nb01, ثابت صحیح nb02, ثابت صحیح nb03, ثابت صحیح ne10, ثابت صحیح ne11,
                      ثابت صحیح ne12, ثابت صحیح nb10, ثابت صحیح nb11, ثابت صحیح nb12, ثابت صحیح nb13,
                      ثابت sycl::nd_شے<3> & شے_ct1) {
    ثابت صحیح i = (شے_ct1.get_local_range(2) * شے_ct1.get_گروہ(2) + شے_ct1.get_local_id(2)) * qk;

    اگر (i >= ne) {
        واپس;
    }


    ثابت صحیح i03      = i / (ne00 * ne01 * ne02);
    ثابت صحیح i02      = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    ثابت صحیح i01      = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
    ثابت صحیح i00      = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
    ثابت صحیح x_انحراف = i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;

    ثابت صحیح i13        = i / (ne10 * ne11 * ne12);
    ثابت صحیح i12        = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
    ثابت صحیح i11        = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
    ثابت صحیح i10        = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
    ثابت صحیح dst_انحراف = (i10 / qk) * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

    cpy_blck(cx + x_انحراف, cdst + dst_انحراف);
}

سانچہ <cpy_kernel_t cpy_blck, صحیح qk>
ساکن خالی cpy_q_f32(ثابت حرف * cx, حرف * cdst, ثابت صحیح ne, ثابت صحیح ne00, ثابت صحیح ne01, ثابت صحیح ne02,
                      ثابت صحیح nb00, ثابت صحیح nb01, ثابت صحیح nb02, ثابت صحیح nb03, ثابت صحیح ne10, ثابت صحیح ne11,
                      ثابت صحیح ne12, ثابت صحیح nb10, ثابت صحیح nb11, ثابت صحیح nb12, ثابت صحیح nb13,
                      ثابت sycl::nd_شے<3> & شے_ct1) {
    ثابت صحیح i = (شے_ct1.get_local_range(2) * شے_ct1.get_گروہ(2) + شے_ct1.get_local_id(2)) * qk;

    اگر (i >= ne) {
        واپس;
    }

    ثابت صحیح i03      = i / (ne00 * ne01 * ne02);
    ثابت صحیح i02      = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    ثابت صحیح i01      = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
    ثابت صحیح i00      = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
    ثابت صحیح x_انحراف = (i00 / qk) * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;

    ثابت صحیح i13        = i / (ne10 * ne11 * ne12);
    ثابت صحیح i12        = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
    ثابت صحیح i11        = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
    ثابت صحیح i10        = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
    ثابت صحیح dst_انحراف = i10 * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

    cpy_blck(cx + x_انحراف, cdst + dst_انحراف);
}

ساکن خالی ggml_cpy_f16_f32_sycl(ثابت حرف * cx, حرف * cdst, ثابت صحیح ne, ثابت صحیح ne00, ثابت صحیح ne01,
                                  ثابت صحیح ne02, ثابت صحیح nb00, ثابت صحیح nb01, ثابت صحیح nb02, ثابت صحیح nb03,
                                  ثابت صحیح ne10, ثابت صحیح ne11, ثابت صحیح ne12, ثابت صحیح nb10, ثابت صحیح nb11,
                                  ثابت صحیح nb12, ثابت صحیح nb13, قطار_ptr stream) {
    ثابت صحیح عدد_بلاکیں = (ne + SYCL_CPY_بلاک_سائز - 1) / SYCL_CPY_بلاک_سائز;
    {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, عدد_بلاکیں) * sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز),
                              sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز)),
            [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]]{
                cpy_f32_f16<cpy_1_f16_f32>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, شے_ct1);
            });
    }
}

ساکن خالی ggml_cpy_f32_f32_sycl(ثابت حرف * cx, حرف * cdst, ثابت صحیح ne, ثابت صحیح ne00, ثابت صحیح ne01,
                                  ثابت صحیح ne02, ثابت صحیح nb00, ثابت صحیح nb01, ثابت صحیح nb02, ثابت صحیح nb03,
                                  ثابت صحیح ne10, ثابت صحیح ne11, ثابت صحیح ne12, ثابت صحیح nb10, ثابت صحیح nb11,
                                  ثابت صحیح nb12, ثابت صحیح nb13, قطار_ptr stream) {
    ثابت صحیح عدد_بلاکیں = (ne + SYCL_CPY_بلاک_سائز - 1) / SYCL_CPY_بلاک_سائز;
    {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, عدد_بلاکیں) * sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز),
                              sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز)),
            [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]]{
                cpy_f32_f16<cpy_1_f32_f32>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, شے_ct1);
            });
    }
}

ساکن خالی ggml_cpy_f32_f16_sycl(ثابت حرف * cx, حرف * cdst, ثابت صحیح ne, ثابت صحیح ne00, ثابت صحیح ne01,
                                  ثابت صحیح ne02, ثابت صحیح nb00, ثابت صحیح nb01, ثابت صحیح nb02, ثابت صحیح nb03,
                                  ثابت صحیح ne10, ثابت صحیح ne11, ثابت صحیح ne12, ثابت صحیح nb10, ثابت صحیح nb11,
                                  ثابت صحیح nb12, ثابت صحیح nb13, قطار_ptr stream) {
    ثابت صحیح عدد_بلاکیں = (ne + SYCL_CPY_بلاک_سائز - 1) / SYCL_CPY_بلاک_سائز;
    {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, عدد_بلاکیں) * sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز),
                              sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز)),
            [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]]{
                cpy_f32_f16<cpy_1_f32_f16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, شے_ct1);
            });
    }
}

ساکن خالی ggml_cpy_f32_i32_sycl(ثابت حرف * cx, حرف * cdst, ثابت صحیح ne, ثابت صحیح ne00, ثابت صحیح ne01,
                                  ثابت صحیح ne02, ثابت صحیح nb00, ثابت صحیح nb01, ثابت صحیح nb02, ثابت صحیح nb03,
                                  ثابت صحیح ne10, ثابت صحیح ne11, ثابت صحیح ne12, ثابت صحیح nb10, ثابت صحیح nb11,
                                  ثابت صحیح nb12, ثابت صحیح nb13, قطار_ptr stream) {
    ثابت صحیح عدد_بلاکیں = (ne + SYCL_CPY_بلاک_سائز - 1) / SYCL_CPY_بلاک_سائز;
    {
        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, عدد_بلاکیں) * sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز),
                              sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز)),
            [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]]{
                cpy_f32_f16<cpy_1_f32_i32>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, شے_ct1);
            });
    }
}

ساکن خالی ggml_cpy_i32_f32_sycl(ثابت حرف * cx, حرف * cdst, ثابت صحیح ne, ثابت صحیح ne00, ثابت صحیح ne01,
                                  ثابت صحیح ne02, ثابت صحیح nb00, ثابت صحیح nb01, ثابت صحیح nb02, ثابت صحیح nb03,
                                  ثابت صحیح ne10, ثابت صحیح ne11, ثابت صحیح ne12, ثابت صحیح nb10, ثابت صحیح nb11,
                                  ثابت صحیح nb12, ثابت صحیح nb13, قطار_ptr stream) {
    ثابت صحیح عدد_بلاکیں = (ne + SYCL_CPY_بلاک_سائز - 1) / SYCL_CPY_بلاک_سائز;
    {
        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, عدد_بلاکیں) * sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز),
                              sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز)),
            [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]]{
                cpy_f32_f16<cpy_1_i32_f32>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, شے_ct1);
            });
    }
}

ساکن خالی ggml_cpy_f32_q8_0_sycl(ثابت حرف * cx, حرف * cdst, ثابت صحیح ne, ثابت صحیح ne00, ثابت صحیح ne01,
                                   ثابت صحیح ne02, ثابت صحیح nb00, ثابت صحیح nb01, ثابت صحیح nb02, ثابت صحیح nb03,
                                   ثابت صحیح ne10, ثابت صحیح ne11, ثابت صحیح ne12, ثابت صحیح nb10, ثابت صحیح nb11,
                                   ثابت صحیح nb12, ثابت صحیح nb13, قطار_ptr stream) {
    GGML_ASSERT(ne % QK8_0 == 0);
    ثابت صحیح عدد_بلاکیں = ceil_div(ne / QK8_0, SYCL_CPY_بلاک_سائز);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, عدد_بلاکیں) * sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز),
                                           sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز)),
                         [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]]{
                             cpy_f32_q<cpy_blck_f32_q8_0, QK8_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, شے_ct1);
                         });
}

ساکن خالی ggml_cpy_q8_0_f32_sycl(ثابت حرف * cx, حرف * cdst, ثابت صحیح ne, ثابت صحیح ne00, ثابت صحیح ne01,
                                   ثابت صحیح ne02, ثابت صحیح nb00, ثابت صحیح nb01, ثابت صحیح nb02, ثابت صحیح nb03,
                                   ثابت صحیح ne10, ثابت صحیح ne11, ثابت صحیح ne12, ثابت صحیح nb10, ثابت صحیح nb11,
                                   ثابت صحیح nb12, ثابت صحیح nb13, قطار_ptr stream) {
    GGML_ASSERT(ne % QK8_0 == 0);
    ثابت صحیح عدد_بلاکیں = ceil_div(ne / QK8_0, SYCL_CPY_بلاک_سائز);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, عدد_بلاکیں) * sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز),
                                           sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز)),
                         [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]]{
                             cpy_q_f32<cpy_blck_q8_0_f32, QK8_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, شے_ct1);
                         });
}

ساکن خالی ggml_cpy_q2_0_f32_sycl(ثابت حرف * cx, حرف * cdst, ثابت صحیح ne, ثابت صحیح ne00, ثابت صحیح ne01,
                                   ثابت صحیح ne02, ثابت صحیح nb00, ثابت صحیح nb01, ثابت صحیح nb02, ثابت صحیح nb03,
                                   ثابت صحیح ne10, ثابت صحیح ne11, ثابت صحیح ne12, ثابت صحیح nb10, ثابت صحیح nb11,
                                   ثابت صحیح nb12, ثابت صحیح nb13, قطار_ptr stream) {
    GGML_ASSERT(ne % QK2_0 == 0);
    ثابت صحیح عدد_بلاکیں = ceil_div(ne / QK2_0, SYCL_CPY_بلاک_سائز);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, عدد_بلاکیں) * sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز),
                          sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز)),
        [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
            cpy_q_f32<cpy_blck_q2_0_f32, QK2_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11,
                                                ne12, nb10, nb11, nb12, nb13, شے_ct1);
        });
}

ساکن خالی ggml_cpy_f32_q4_0_sycl(ثابت حرف * cx, حرف * cdst, ثابت صحیح ne, ثابت صحیح ne00, ثابت صحیح ne01,
                                   ثابت صحیح ne02, ثابت صحیح nb00, ثابت صحیح nb01, ثابت صحیح nb02, ثابت صحیح nb03,
                                   ثابت صحیح ne10, ثابت صحیح ne11, ثابت صحیح ne12, ثابت صحیح nb10, ثابت صحیح nb11,
                                   ثابت صحیح nb12, ثابت صحیح nb13, قطار_ptr stream) {
    GGML_ASSERT(ne % QK4_0 == 0);
    ثابت صحیح عدد_بلاکیں = ceil_div(ne / QK4_0, SYCL_CPY_بلاک_سائز);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, عدد_بلاکیں) * sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز),
                                           sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز)),
                         [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]]{
                             cpy_f32_q<cpy_blck_f32_q4_0, QK4_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, شے_ct1);
                         });
}

ساکن خالی ggml_cpy_q4_0_f32_sycl(ثابت حرف * cx, حرف * cdst, ثابت صحیح ne, ثابت صحیح ne00, ثابت صحیح ne01,
                                   ثابت صحیح ne02, ثابت صحیح nb00, ثابت صحیح nb01, ثابت صحیح nb02, ثابت صحیح nb03,
                                   ثابت صحیح ne10, ثابت صحیح ne11, ثابت صحیح ne12, ثابت صحیح nb10, ثابت صحیح nb11,
                                   ثابت صحیح nb12, ثابت صحیح nb13, قطار_ptr stream) {
    GGML_ASSERT(ne % QK4_0 == 0);
    ثابت صحیح عدد_بلاکیں = ceil_div(ne / QK4_0, SYCL_CPY_بلاک_سائز);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, عدد_بلاکیں) * sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز),
                          sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز)),
        [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]]{
            cpy_q_f32<cpy_blck_q_f32<dequantize_q4_0, QK4_0>, QK4_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02,
                                                                     nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                                                     شے_ct1);
        });
}

ساکن خالی ggml_cpy_f32_q4_1_sycl(ثابت حرف * cx, حرف * cdst, ثابت صحیح ne, ثابت صحیح ne00, ثابت صحیح ne01,
                                   ثابت صحیح ne02, ثابت صحیح nb00, ثابت صحیح nb01, ثابت صحیح nb02, ثابت صحیح nb03,
                                   ثابت صحیح ne10, ثابت صحیح ne11, ثابت صحیح ne12, ثابت صحیح nb10, ثابت صحیح nb11,
                                   ثابت صحیح nb12, ثابت صحیح nb13, قطار_ptr stream) {
    GGML_ASSERT(ne % QK4_1 == 0);
    ثابت صحیح عدد_بلاکیں = ceil_div(ne / QK4_1, SYCL_CPY_بلاک_سائز);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, عدد_بلاکیں) * sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز),
                                           sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز)),
                         [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]]{
                             cpy_f32_q<cpy_blck_f32_q4_1, QK4_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, شے_ct1);
                         });
}

ساکن خالی ggml_cpy_q4_1_f32_sycl(ثابت حرف * cx, حرف * cdst, ثابت صحیح ne, ثابت صحیح ne00, ثابت صحیح ne01,
                                   ثابت صحیح ne02, ثابت صحیح nb00, ثابت صحیح nb01, ثابت صحیح nb02, ثابت صحیح nb03,
                                   ثابت صحیح ne10, ثابت صحیح ne11, ثابت صحیح ne12, ثابت صحیح nb10, ثابت صحیح nb11,
                                   ثابت صحیح nb12, ثابت صحیح nb13, قطار_ptr stream) {
    GGML_ASSERT(ne % QK4_1 == 0);
    ثابت صحیح عدد_بلاکیں = ceil_div(ne / QK4_1, SYCL_CPY_بلاک_سائز);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, عدد_بلاکیں) * sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز),
                          sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز)),
        [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]]{
            cpy_q_f32<cpy_blck_q_f32<dequantize_q4_1, QK4_1>, QK4_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02,
                                                                     nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                                                     شے_ct1);
        });
}

ساکن خالی ggml_cpy_f32_q5_0_sycl(ثابت حرف * cx, حرف * cdst, ثابت صحیح ne, ثابت صحیح ne00, ثابت صحیح ne01,
                                   ثابت صحیح ne02, ثابت صحیح nb00, ثابت صحیح nb01, ثابت صحیح nb02, ثابت صحیح nb03,
                                   ثابت صحیح ne10, ثابت صحیح ne11, ثابت صحیح ne12, ثابت صحیح nb10, ثابت صحیح nb11,
                                   ثابت صحیح nb12, ثابت صحیح nb13, قطار_ptr stream) {
    GGML_ASSERT(ne % QK5_0 == 0);
    ثابت صحیح عدد_بلاکیں = ceil_div(ne / QK5_0, SYCL_CPY_بلاک_سائز);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, عدد_بلاکیں) * sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز),
                                           sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز)),
                         [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                             cpy_f32_q<cpy_blck_f32_q5_0, QK5_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, شے_ct1);
                         });
}

ساکن خالی ggml_cpy_q5_0_f32_sycl(ثابت حرف * cx, حرف * cdst, ثابت صحیح ne, ثابت صحیح ne00, ثابت صحیح ne01,
                                   ثابت صحیح ne02, ثابت صحیح nb00, ثابت صحیح nb01, ثابت صحیح nb02, ثابت صحیح nb03,
                                   ثابت صحیح ne10, ثابت صحیح ne11, ثابت صحیح ne12, ثابت صحیح nb10, ثابت صحیح nb11,
                                   ثابت صحیح nb12, ثابت صحیح nb13, قطار_ptr stream) {
    GGML_ASSERT(ne % QK5_0 == 0);
    ثابت صحیح عدد_بلاکیں = ceil_div(ne / QK5_0, SYCL_CPY_بلاک_سائز);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, عدد_بلاکیں) * sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز),
                          sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز)),
        [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]]{
            cpy_q_f32<cpy_blck_q_f32<dequantize_q5_0, QK5_0>, QK5_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02,
                                                                     nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                                                     شے_ct1);
        });
}

ساکن خالی ggml_cpy_f32_q5_1_sycl(ثابت حرف * cx, حرف * cdst, ثابت صحیح ne, ثابت صحیح ne00, ثابت صحیح ne01,
                                   ثابت صحیح ne02, ثابت صحیح nb00, ثابت صحیح nb01, ثابت صحیح nb02, ثابت صحیح nb03,
                                   ثابت صحیح ne10, ثابت صحیح ne11, ثابت صحیح ne12, ثابت صحیح nb10, ثابت صحیح nb11,
                                   ثابت صحیح nb12, ثابت صحیح nb13, قطار_ptr stream) {
    GGML_ASSERT(ne % QK5_1 == 0);
    ثابت صحیح عدد_بلاکیں = ceil_div(ne / QK5_1, SYCL_CPY_بلاک_سائز);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, عدد_بلاکیں) * sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز),
                                           sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز)),
                         [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]]{
                             cpy_f32_q<cpy_blck_f32_q5_1, QK5_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, شے_ct1);
                         });
}

ساکن خالی ggml_cpy_q5_1_f32_sycl(ثابت حرف * cx, حرف * cdst, ثابت صحیح ne, ثابت صحیح ne00, ثابت صحیح ne01,
                                   ثابت صحیح ne02, ثابت صحیح nb00, ثابت صحیح nb01, ثابت صحیح nb02, ثابت صحیح nb03,
                                   ثابت صحیح ne10, ثابت صحیح ne11, ثابت صحیح ne12, ثابت صحیح nb10, ثابت صحیح nb11,
                                   ثابت صحیح nb12, ثابت صحیح nb13, قطار_ptr stream) {
    GGML_ASSERT(ne % QK5_1 == 0);
    ثابت صحیح عدد_بلاکیں = ceil_div(ne / QK5_1, SYCL_CPY_بلاک_سائز);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, عدد_بلاکیں) * sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز),
                          sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز)),
        [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]]{
            cpy_q_f32<cpy_blck_q_f32<dequantize_q5_1, QK5_1>, QK5_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02,
                                                                     nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                                                     شے_ct1);
        });
}

ساکن خالی ggml_cpy_mxfp4_f32_sycl(ثابت حرف * cx, حرف * cdst, ثابت صحیح ne, ثابت صحیح ne00, ثابت صحیح ne01,
                                    ثابت صحیح ne02, ثابت صحیح nb00, ثابت صحیح nb01, ثابت صحیح nb02, ثابت صحیح nb03,
                                    ثابت صحیح ne10, ثابت صحیح ne11, ثابت صحیح ne12, ثابت صحیح nb10, ثابت صحیح nb11,
                                    ثابت صحیح nb12, ثابت صحیح nb13, قطار_ptr stream) {
    GGML_ASSERT(ne % QK_MXFP4 == 0);
    ثابت صحیح عدد_بلاکیں = ceil_div(ne / QK_MXFP4, SYCL_CPY_بلاک_سائز);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, عدد_بلاکیں) * sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز),
                          sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز)),
        [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
            cpy_q_f32<cpy_blck_q_f32<dequantize_mxfp4, QK_MXFP4>, QK_MXFP4>(cx, cdst, ne, ne00, ne01, ne02, nb00,
                                                                             nb01, nb02, nb03, ne10, ne11, ne12,
                                                                             nb10, nb11, nb12, nb13, شے_ct1);
        });
}

ساکن خالی ggml_cpy_f32_iq4_nl_sycl(ثابت حرف * cx, حرف * cdst, ثابت صحیح ne, ثابت صحیح ne00, ثابت صحیح ne01,
                                     ثابت صحیح ne02, ثابت صحیح nb00, ثابت صحیح nb01, ثابت صحیح nb02, ثابت صحیح nb03,
                                     ثابت صحیح ne10, ثابت صحیح ne11, ثابت صحیح ne12, ثابت صحیح nb10, ثابت صحیح nb11,
                                     ثابت صحیح nb12, ثابت صحیح nb13, قطار_ptr stream) {
    GGML_ASSERT(ne % QK4_NL == 0);
    ثابت صحیح عدد_بلاکیں = ceil_div(ne / QK4_NL, SYCL_CPY_بلاک_سائز);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, عدد_بلاکیں) * sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز),
                          sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز)),
        [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
            cpy_f32_q<cpy_blck_f32_iq4_nl, QK4_NL>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11,
                                                   ne12, nb10, nb11, nb12, nb13, شے_ct1);
        });
}

ساکن خالی cpy_blck_f16_q4_0(ثابت حرف * cxi, حرف * cdsti) {
    ثابت sycl::half * xi = (ثابت sycl::half *) cxi;
    اعشاری xf[QK4_0];

    برائے (صحیح j = 0; j < QK4_0; ++j) {
        xf[j] = (اعشاری) xi[j];
    }

    cpy_blck_f32_q4_0((ثابت حرف *) xf, cdsti);
}

ساکن خالی cpy_blck_f16_q4_1(ثابت حرف * cxi, حرف * cdsti) {
    ثابت sycl::half * xi = (ثابت sycl::half *) cxi;
    اعشاری xf[QK4_1];

    برائے (صحیح j = 0; j < QK4_1; ++j) {
        xf[j] = (اعشاری) xi[j];
    }

    cpy_blck_f32_q4_1((ثابت حرف *) xf, cdsti);
}

ساکن خالی cpy_blck_f16_q5_0(ثابت حرف * cxi, حرف * cdsti) {
    ثابت sycl::half * xi = (ثابت sycl::half *) cxi;
    اعشاری xf[QK5_0];

    برائے (صحیح j = 0; j < QK5_0; ++j) {
        xf[j] = (اعشاری) xi[j];
    }

    cpy_blck_f32_q5_0((ثابت حرف *) xf, cdsti);
}

ساکن خالی ggml_cpy_f16_q4_0_sycl(ثابت حرف * cx, حرف * cdst, ثابت صحیح ne, ثابت صحیح ne00, ثابت صحیح ne01,
                                   ثابت صحیح ne02, ثابت صحیح nb00, ثابت صحیح nb01, ثابت صحیح nb02, ثابت صحیح nb03,
                                   ثابت صحیح ne10, ثابت صحیح ne11, ثابت صحیح ne12, ثابت صحیح nb10, ثابت صحیح nb11,
                                   ثابت صحیح nb12, ثابت صحیح nb13, قطار_ptr stream) {
    GGML_ASSERT(ne % QK4_0 == 0);
    ثابت صحیح عدد_بلاکیں = ceil_div(ne / QK4_0, SYCL_CPY_بلاک_سائز);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, عدد_بلاکیں) * sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز),
                                           sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز)),
                         [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]]{
                             cpy_f32_q<cpy_blck_f16_q4_0, QK4_0>(cx, cdst, ne, ne00, ne01, ne02,
                                                                 nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, شے_ct1);
                         });
}

ساکن خالی ggml_cpy_f16_q4_1_sycl(ثابت حرف * cx, حرف * cdst, ثابت صحیح ne, ثابت صحیح ne00, ثابت صحیح ne01,
                                   ثابت صحیح ne02, ثابت صحیح nb00, ثابت صحیح nb01, ثابت صحیح nb02, ثابت صحیح nb03,
                                   ثابت صحیح ne10, ثابت صحیح ne11, ثابت صحیح ne12, ثابت صحیح nb10, ثابت صحیح nb11,
                                   ثابت صحیح nb12, ثابت صحیح nb13, قطار_ptr stream) {
    GGML_ASSERT(ne % QK4_1 == 0);
    ثابت صحیح عدد_بلاکیں = ceil_div(ne / QK4_1, SYCL_CPY_بلاک_سائز);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, عدد_بلاکیں) * sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز),
                                           sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز)),
                         [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]]{
                             cpy_f32_q<cpy_blck_f16_q4_1, QK4_1>(cx, cdst, ne, ne00, ne01, ne02,
                                                                 nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, شے_ct1);
                         });
}

ساکن خالی ggml_cpy_f16_q5_0_sycl(ثابت حرف * cx, حرف * cdst, ثابت صحیح ne, ثابت صحیح ne00, ثابت صحیح ne01,
                                   ثابت صحیح ne02, ثابت صحیح nb00, ثابت صحیح nb01, ثابت صحیح nb02, ثابت صحیح nb03,
                                   ثابت صحیح ne10, ثابت صحیح ne11, ثابت صحیح ne12, ثابت صحیح nb10, ثابت صحیح nb11,
                                   ثابت صحیح nb12, ثابت صحیح nb13, قطار_ptr stream) {
    GGML_ASSERT(ne % QK5_0 == 0);
    ثابت صحیح عدد_بلاکیں = ceil_div(ne / QK5_0, SYCL_CPY_بلاک_سائز);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, عدد_بلاکیں) * sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز),
                                           sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز)),
                         [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]]{
                             cpy_f32_q<cpy_blck_f16_q5_0, QK5_0>(cx, cdst, ne, ne00, ne01, ne02,
                                                                 nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, شے_ct1);
                         });
}

ساکن منطقی ggml_sycl_is_quantized_type(شمار ggml_type type) {
    انتخاب (type) {
        حالت GGML_TYPE_Q1_0:
        حالت GGML_TYPE_Q2_0:
        حالت GGML_TYPE_Q4_0:
        حالت GGML_TYPE_Q4_1:
        حالت GGML_TYPE_Q5_0:
        حالت GGML_TYPE_Q5_1:
        حالت GGML_TYPE_Q8_0:
        حالت GGML_TYPE_MXFP4:
        حالت GGML_TYPE_NVFP4:
        حالت GGML_TYPE_Q2_K:
        حالت GGML_TYPE_Q3_K:
        حالت GGML_TYPE_Q4_K:
        حالت GGML_TYPE_Q5_K:
        حالت GGML_TYPE_Q6_K:
        حالت GGML_TYPE_IQ2_XXS:
        حالت GGML_TYPE_IQ2_XS:
        حالت GGML_TYPE_IQ2_S:
        حالت GGML_TYPE_IQ3_XXS:
        حالت GGML_TYPE_IQ3_S:
        حالت GGML_TYPE_IQ1_S:
        حالت GGML_TYPE_IQ1_M:
        حالت GGML_TYPE_IQ4_NL:
        حالت GGML_TYPE_IQ4_XS:
            واپس سچ;
        طےشدہ:
            واپس جھوٹ;
    }
}

ساکن منطقی ggml_sycl_can_quantize_صفیں_sycl(شمار ggml_type type) {
    انتخاب (type) {
        حالت GGML_TYPE_Q1_0:
        حالت GGML_TYPE_Q2_0:
        حالت GGML_TYPE_Q4_0:
        حالت GGML_TYPE_Q4_1:
        حالت GGML_TYPE_Q5_0:
        حالت GGML_TYPE_Q5_1:
        حالت GGML_TYPE_Q8_0:
        حالت GGML_TYPE_MXFP4:
        حالت GGML_TYPE_NVFP4:
        حالت GGML_TYPE_Q2_K:
        حالت GGML_TYPE_Q3_K:
        حالت GGML_TYPE_Q4_K:
        حالت GGML_TYPE_Q5_K:
        حالت GGML_TYPE_Q6_K:
        حالت GGML_TYPE_IQ4_NL:
        حالت GGML_TYPE_IQ4_XS:
            واپس سچ;
        طےشدہ:
            واپس جھوٹ;
    }
}

سانچہ <قسم_نام SrcScalar>
ساکن اندرونی اعشاری ggml_sycl_src_مقصد_f32(ثابت SrcScalar & x) {
    واپس (اعشاری) x;
}

#اگر_تعریف_شدہ GGML_SYCL_HAS_BF16
سانچہ <>
اندرونی اعشاری ggml_sycl_src_مقصد_f32<sycl::ext::oneapi::bfloat16>(ثابت sycl::ext::oneapi::bfloat16 & x) {
    واپس static_cast<اعشاری>(x);
}

سانچہ <>
اندرونی اعشاری ggml_sycl_src_مقصد_f32<ggml_bf16_t>(ثابت ggml_bf16_t & x) {
    اتحاد {
        uint32_t u32;
        اعشاری f32;
    } value;

    value.u32 = (uint32_t) x.بٹ << 16;
    واپس value.f32;
}
#اگر_ختم

سانچہ <قسم_نام SrcScalar, cpy_kernel_t quantize_بلاک, صحیح qk>
ساکن خالی ggml_sycl_quantize_صفیں_q(ثابت حرف * cx, حرف * cdst, ثابت int64_t ne,
                                      ثابت int64_t ne00, ثابت int64_t ne01, ثابت int64_t ne02,
                                      ثابت size_t nb00, ثابت size_t nb01, ثابت size_t nb02, ثابت size_t nb03,
                                      ثابت int64_t ne10, ثابت int64_t ne11, ثابت int64_t ne12,
                                      ثابت size_t nb10, ثابت size_t nb11, ثابت size_t nb12, ثابت size_t nb13,
                                      قطار_ptr stream) {
    GGML_ASSERT(ne % qk == 0);
    GGML_ASSERT(ne00 % qk == 0);

    ثابت int64_t کل_بلاکیں = ne / qk;
    constexpr صحیح بلاک_سائز_3 = 256;
    ثابت int64_t grid_سائز = ceil_div(کل_بلاکیں, (int64_t) بلاک_سائز_3);

    stream->parallel_for(sycl::nd_range<1>(grid_سائز * بلاک_سائز_3, بلاک_سائز_3),
                        [=](sycl::nd_شے<1> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
        ثابت int64_t بلاک_idx = شے_ct1.get_global_خطی_id();
        اگر (بلاک_idx >= کل_بلاکیں) {
            واپس;
        }

        ثابت int64_t i = بلاک_idx * qk;

        ثابت int64_t i03 = i / (ne00 * ne01 * ne02);
        ثابت int64_t i02 = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
        ثابت int64_t i01 = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
        ثابت int64_t i00 = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
        ثابت size_t x_انحراف = i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;

        ثابت int64_t i13 = i / (ne10 * ne11 * ne12);
        ثابت int64_t i12 = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
        ثابت int64_t i11 = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
        ثابت int64_t i10 = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
        ثابت size_t dst_انحراف = (i10 / qk) * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

        اعشاری xf[qk];
        اگر (nb00 == حجم(SrcScalar)) {
            ثابت SrcScalar * src_صف = (ثابت SrcScalar *) (cx + x_انحراف);
            برائے (صحیح j = 0; j < qk; ++j) {
                xf[j] = ggml_sycl_src_مقصد_f32(src_صف[j]);
            }
        } ورنہ {
            برائے (صحیح j = 0; j < qk; ++j) {
                ثابت SrcScalar * src_val = (ثابت SrcScalar *) (cx + x_انحراف + j * nb00);
                xf[j] = ggml_sycl_src_مقصد_f32(*src_val);
            }
        }

        quantize_بلاک((ثابت حرف *) xf, cdst + dst_انحراف);
    });
}

سانچہ <قسم_نام SrcScalar>
ساکن خالی ggml_sycl_quantize_صفیں_sycl(ثابت حرف * cx, حرف * cdst, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1,
                                         ثابت int64_t ne, ثابت int64_t ne00, ثابت int64_t ne01, ثابت int64_t ne02,
                                         ثابت size_t nb00, ثابت size_t nb01, ثابت size_t nb02, ثابت size_t nb03,
                                         ثابت int64_t ne10, ثابت int64_t ne11, ثابت int64_t ne12, ثابت size_t nb10,
                                         ثابت size_t nb11, ثابت size_t nb12, ثابت size_t nb13, قطار_ptr stream) {
    GGML_غیر_مستعمل(src0);
    GGML_غیر_مستعمل(src1);

    انتخاب (src1->type) {
        حالت GGML_TYPE_Q8_0:
            ggml_sycl_quantize_صفیں_q<SrcScalar, cpy_blck_f32_q8_0, QK8_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            توڑو;
        حالت GGML_TYPE_Q1_0:
            ggml_sycl_quantize_صفیں_q<SrcScalar, cpy_blck_f32_q1_0, QK1_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            توڑو;
        حالت GGML_TYPE_Q2_0:
            ggml_sycl_quantize_صفیں_q<SrcScalar, cpy_blck_f32_q2_0, QK2_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            توڑو;
        حالت GGML_TYPE_Q5_1:
            ggml_sycl_quantize_صفیں_q<SrcScalar, cpy_blck_f32_q5_1, QK5_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            توڑو;
        حالت GGML_TYPE_Q5_0:
            ggml_sycl_quantize_صفیں_q<SrcScalar, cpy_blck_f32_q5_0, QK5_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            توڑو;
        حالت GGML_TYPE_Q4_1:
            ggml_sycl_quantize_صفیں_q<SrcScalar, cpy_blck_f32_q4_1, QK4_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            توڑو;
        حالت GGML_TYPE_Q4_0:
            ggml_sycl_quantize_صفیں_q<SrcScalar, cpy_blck_f32_q4_0, QK4_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            توڑو;
        حالت GGML_TYPE_IQ4_NL:
            ggml_sycl_quantize_صفیں_q<SrcScalar, cpy_blck_f32_iq4_nl, QK4_NL>(cx, cdst, ne, ne00, ne01, ne02, nb00,
                                                                               nb01, nb02, nb03, ne10, ne11, ne12,
                                                                               nb10, nb11, nb12, nb13, stream);
            توڑو;
        حالت GGML_TYPE_MXFP4:
            ggml_sycl_quantize_صفیں_q<SrcScalar, cpy_blck_f32_mxfp4, QK_MXFP4>(cx, cdst, ne, ne00, ne01, ne02, nb00,
                                                                                nb01, nb02, nb03, ne10, ne11, ne12,
                                                                                nb10, nb11, nb12, nb13, stream);
            توڑو;
        حالت GGML_TYPE_NVFP4:
            ggml_sycl_quantize_صفیں_q<SrcScalar, cpy_blck_f32_nvfp4, QK_NVFP4>(cx, cdst, ne, ne00, ne01, ne02, nb00,
                                                                                nb01, nb02, nb03, ne10, ne11, ne12,
                                                                                nb10, nb11, nb12, nb13, stream);
            توڑو;
        طےشدہ:
            GGML_ABORT("unsupported quantized target type in sycl quantizer src1->type=%s\n",
                       ggml_type_name(src1->type));
    }
}

ساکن خالی ggml_cpy_f16_f16_sycl(ثابت حرف * cx, حرف * cdst, ثابت صحیح ne, ثابت صحیح ne00, ثابت صحیح ne01,
                                  ثابت صحیح ne02, ثابت صحیح nb00, ثابت صحیح nb01, ثابت صحیح nb02, ثابت صحیح nb03,
                                  ثابت صحیح ne10, ثابت صحیح ne11, ثابت صحیح ne12, ثابت صحیح nb10, ثابت صحیح nb11,
                                  ثابت صحیح nb12, ثابت صحیح nb13, قطار_ptr stream) {
    ثابت صحیح عدد_بلاکیں = (ne + SYCL_CPY_بلاک_سائز - 1) / SYCL_CPY_بلاک_سائز;
    {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, عدد_بلاکیں) * sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز),
                              sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز)),
            [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]]{
                cpy_f32_f16<cpy_1_f16_f16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, شے_ct1);
            });
    }
}

ساکن خالی ggml_cpy_i16_i16_sycl(ثابت حرف * cx, حرف * cdst, ثابت صحیح ne, ثابت صحیح ne00, ثابت صحیح ne01,
                                  ثابت صحیح ne02, ثابت صحیح nb00, ثابت صحیح nb01, ثابت صحیح nb02, ثابت صحیح nb03,
                                  ثابت صحیح ne10, ثابت صحیح ne11, ثابت صحیح ne12, ثابت صحیح nb10, ثابت صحیح nb11,
                                  ثابت صحیح nb12, ثابت صحیح nb13, قطار_ptr stream) {
    ثابت صحیح عدد_بلاکیں = (ne + SYCL_CPY_بلاک_سائز - 1) / SYCL_CPY_بلاک_سائز;
    {
        // dpct::has_capability_or_fail(stream->get_device(),
        //                              {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, عدد_بلاکیں) * sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز),
                              sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز)),
            [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]]{
                cpy_f32_f16<cpy_1_i16_i16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, شے_ct1);
            });
    }
}

ساکن خالی ggml_cpy_i32_i32_sycl(ثابت حرف * cx, حرف * cdst, ثابت صحیح ne, ثابت صحیح ne00, ثابت صحیح ne01,
                                  ثابت صحیح ne02, ثابت صحیح nb00, ثابت صحیح nb01, ثابت صحیح nb02, ثابت صحیح nb03,
                                  ثابت صحیح ne10, ثابت صحیح ne11, ثابت صحیح ne12, ثابت صحیح nb10, ثابت صحیح nb11,
                                  ثابت صحیح nb12, ثابت صحیح nb13, قطار_ptr stream) {
    ثابت صحیح عدد_بلاکیں = (ne + SYCL_CPY_بلاک_سائز - 1) / SYCL_CPY_بلاک_سائز;
    {
        // dpct::has_capability_or_fail(stream->get_device(),
        //                              {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, عدد_بلاکیں) * sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز),
                              sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز)),
            [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]]{
                cpy_f32_f16<cpy_1_i32_i32>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, شے_ct1);
            });
    }
}

ساکن خالی ggml_cpy_q8_0_q8_0(ثابت حرف * cx, حرف * cdst, ثابت صحیح ne, ثابت صحیح ne00, ثابت صحیح ne01,
                                   ثابت صحیح ne02, ثابت صحیح nb00, ثابت صحیح nb01, ثابت صحیح nb02, ثابت صحیح nb03,
                                   ثابت صحیح ne10, ثابت صحیح ne11, ثابت صحیح ne12, ثابت صحیح nb10, ثابت صحیح nb11,
                                   ثابت صحیح nb12, ثابت صحیح nb13, قطار_ptr stream) {
    GGML_ASSERT(ne % QK8_0 == 0);
    ثابت صحیح عدد_بلاکیں = ceil_div(ne / QK8_0, SYCL_CPY_بلاک_سائز);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, عدد_بلاکیں) * sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز),
                              sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز)),
                              [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]]{
            cpy_q_q<بلاک_q8_0, QK8_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, شے_ct1);
        });
}


ساکن خالی ggml_cpy_q5_0_q5_0(ثابت حرف * cx, حرف * cdst, ثابت صحیح ne, ثابت صحیح ne00, ثابت صحیح ne01,
                                   ثابت صحیح ne02, ثابت صحیح nb00, ثابت صحیح nb01, ثابت صحیح nb02, ثابت صحیح nb03,
                                   ثابت صحیح ne10, ثابت صحیح ne11, ثابت صحیح ne12, ثابت صحیح nb10, ثابت صحیح nb11,
                                   ثابت صحیح nb12, ثابت صحیح nb13, قطار_ptr stream) {
    GGML_ASSERT(ne % QK5_0 == 0);
    ثابت صحیح عدد_بلاکیں = ceil_div(ne / QK5_0, SYCL_CPY_بلاک_سائز);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, عدد_بلاکیں) * sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز),
                              sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز)),
                              [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]]{
            cpy_q_q<بلاک_q5_0, QK5_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, شے_ct1);
        });
}


ساکن خالی ggml_cpy_q5_1_q5_1(ثابت حرف * cx, حرف * cdst, ثابت صحیح ne, ثابت صحیح ne00, ثابت صحیح ne01,
                                   ثابت صحیح ne02, ثابت صحیح nb00, ثابت صحیح nb01, ثابت صحیح nb02, ثابت صحیح nb03,
                                   ثابت صحیح ne10, ثابت صحیح ne11, ثابت صحیح ne12, ثابت صحیح nb10, ثابت صحیح nb11,
                                   ثابت صحیح nb12, ثابت صحیح nb13, قطار_ptr stream) {
    GGML_ASSERT(ne % QK5_1 == 0);
    ثابت صحیح عدد_بلاکیں = ceil_div(ne / QK5_1, SYCL_CPY_بلاک_سائز);

    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, عدد_بلاکیں) * sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز),
                              sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز)),
                              [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]]{
            cpy_q_q<بلاک_q5_1, QK5_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, شے_ct1);
        });
}


ساکن خالی ggml_cpy_q4_0_q4_0(ثابت حرف * cx, حرف * cdst, ثابت صحیح ne, ثابت صحیح ne00, ثابت صحیح ne01,
                                   ثابت صحیح ne02, ثابت صحیح nb00, ثابت صحیح nb01, ثابت صحیح nb02, ثابت صحیح nb03,
                                   ثابت صحیح ne10, ثابت صحیح ne11, ثابت صحیح ne12, ثابت صحیح nb10, ثابت صحیح nb11,
                                   ثابت صحیح nb12, ثابت صحیح nb13, قطار_ptr stream) {
    GGML_ASSERT(ne % QK4_0 == 0);
    ثابت صحیح عدد_بلاکیں = ceil_div(ne / QK4_0, SYCL_CPY_بلاک_سائز);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, عدد_بلاکیں) * sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز), sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز)),
        [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]]{
            cpy_q_q<بلاک_q4_0, QK4_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, شے_ct1);
        });
}


ساکن خالی ggml_cpy_q4_1_q4_1(ثابت حرف * cx, حرف * cdst, ثابت صحیح ne, ثابت صحیح ne00, ثابت صحیح ne01,
                                   ثابت صحیح ne02, ثابت صحیح nb00, ثابت صحیح nb01, ثابت صحیح nb02, ثابت صحیح nb03,
                                   ثابت صحیح ne10, ثابت صحیح ne11, ثابت صحیح ne12, ثابت صحیح nb10, ثابت صحیح nb11,
                                   ثابت صحیح nb12, ثابت صحیح nb13, قطار_ptr stream) {

    GGML_ASSERT(ne % QK4_1 == 0);
    ثابت صحیح عدد_بلاکیں = ceil_div(ne / QK4_1, SYCL_CPY_بلاک_سائز);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, عدد_بلاکیں) * sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز), sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز)),
        [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]]{
            cpy_q_q<بلاک_q4_1, QK4_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, شے_ct1);
        });
}

ساکن خالی ggml_cpy_q1_0_q1_0(ثابت حرف * cx, حرف * cdst, ثابت صحیح ne, ثابت صحیح ne00, ثابت صحیح ne01,
                                   ثابت صحیح ne02, ثابت صحیح nb00, ثابت صحیح nb01, ثابت صحیح nb02, ثابت صحیح nb03,
                                   ثابت صحیح ne10, ثابت صحیح ne11, ثابت صحیح ne12, ثابت صحیح nb10, ثابت صحیح nb11,
                                   ثابت صحیح nb12, ثابت صحیح nb13, قطار_ptr stream) {
    GGML_ASSERT(ne % QK1_0 == 0);
    ثابت صحیح عدد_بلاکیں = ceil_div(ne / QK1_0, SYCL_CPY_بلاک_سائز);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, عدد_بلاکیں) * sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز), sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز)),
        [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
            cpy_q_q<بلاک_q1_0, QK1_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, شے_ct1);
        });
}

ساکن خالی ggml_cpy_q2_0_q2_0(ثابت حرف * cx, حرف * cdst, ثابت صحیح ne, ثابت صحیح ne00, ثابت صحیح ne01,
                                   ثابت صحیح ne02, ثابت صحیح nb00, ثابت صحیح nb01, ثابت صحیح nb02, ثابت صحیح nb03,
                                   ثابت صحیح ne10, ثابت صحیح ne11, ثابت صحیح ne12, ثابت صحیح nb10, ثابت صحیح nb11,
                                   ثابت صحیح nb12, ثابت صحیح nb13, قطار_ptr stream) {
    GGML_ASSERT(ne % QK2_0 == 0);
    ثابت صحیح عدد_بلاکیں = ceil_div(ne / QK2_0, SYCL_CPY_بلاک_سائز);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, عدد_بلاکیں) * sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز), sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز)),
        [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]]{
            cpy_q_q<بلاک_q2_0, QK2_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, شے_ct1);
        });
}

ساکن خالی ggml_cpy_mxfp4_mxfp4(ثابت حرف * cx, حرف * cdst, ثابت صحیح ne, ثابت صحیح ne00, ثابت صحیح ne01,
                                   ثابت صحیح ne02, ثابت صحیح nb00, ثابت صحیح nb01, ثابت صحیح nb02, ثابت صحیح nb03,
                                   ثابت صحیح ne10, ثابت صحیح ne11, ثابت صحیح ne12, ثابت صحیح nb10, ثابت صحیح nb11,
                                   ثابت صحیح nb12, ثابت صحیح nb13, قطار_ptr stream) {
    GGML_ASSERT(ne % QK_MXFP4 == 0);
    ثابت صحیح عدد_بلاکیں = ceil_div(ne / QK_MXFP4, SYCL_CPY_بلاک_سائز);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, عدد_بلاکیں) * sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز), sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز)),
        [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
            cpy_q_q<بلاک_mxfp4, QK_MXFP4>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, شے_ct1);
        });
}

ساکن خالی ggml_cpy_nvfp4_nvfp4(ثابت حرف * cx, حرف * cdst, ثابت صحیح ne, ثابت صحیح ne00, ثابت صحیح ne01,
                                   ثابت صحیح ne02, ثابت صحیح nb00, ثابت صحیح nb01, ثابت صحیح nb02, ثابت صحیح nb03,
                                   ثابت صحیح ne10, ثابت صحیح ne11, ثابت صحیح ne12, ثابت صحیح nb10, ثابت صحیح nb11,
                                   ثابت صحیح nb12, ثابت صحیح nb13, قطار_ptr stream) {
    GGML_ASSERT(ne % QK_NVFP4 == 0);
    ثابت صحیح عدد_بلاکیں = ceil_div(ne / QK_NVFP4, SYCL_CPY_بلاک_سائز);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, عدد_بلاکیں) * sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز), sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز)),
        [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]]{
            cpy_q_q<بلاک_nvfp4, QK_NVFP4>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, شے_ct1);
        });
}

ساکن خالی ggml_cpy_q2_K_q2_K(ثابت حرف * cx, حرف * cdst, ثابت صحیح ne, ثابت صحیح ne00, ثابت صحیح ne01,
                                   ثابت صحیح ne02, ثابت صحیح nb00, ثابت صحیح nb01, ثابت صحیح nb02, ثابت صحیح nb03,
                                   ثابت صحیح ne10, ثابت صحیح ne11, ثابت صحیح ne12, ثابت صحیح nb10, ثابت صحیح nb11,
                                   ثابت صحیح nb12, ثابت صحیح nb13, قطار_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ثابت صحیح عدد_بلاکیں = ceil_div(ne / QK_K, SYCL_CPY_بلاک_سائز);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, عدد_بلاکیں) * sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز), sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز)),
        [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]]{
            cpy_q_q<بلاک_q2_K, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, شے_ct1);
        });
}

ساکن خالی ggml_cpy_q3_K_q3_K(ثابت حرف * cx, حرف * cdst, ثابت صحیح ne, ثابت صحیح ne00, ثابت صحیح ne01,
                                   ثابت صحیح ne02, ثابت صحیح nb00, ثابت صحیح nb01, ثابت صحیح nb02, ثابت صحیح nb03,
                                   ثابت صحیح ne10, ثابت صحیح ne11, ثابت صحیح ne12, ثابت صحیح nb10, ثابت صحیح nb11,
                                   ثابت صحیح nb12, ثابت صحیح nb13, قطار_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ثابت صحیح عدد_بلاکیں = ceil_div(ne / QK_K, SYCL_CPY_بلاک_سائز);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, عدد_بلاکیں) * sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز), sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز)),
        [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]]{
            cpy_q_q<بلاک_q3_K, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, شے_ct1);
        });
}

ساکن خالی ggml_cpy_q4_K_q4_K(ثابت حرف * cx, حرف * cdst, ثابت صحیح ne, ثابت صحیح ne00, ثابت صحیح ne01,
                                   ثابت صحیح ne02, ثابت صحیح nb00, ثابت صحیح nb01, ثابت صحیح nb02, ثابت صحیح nb03,
                                   ثابت صحیح ne10, ثابت صحیح ne11, ثابت صحیح ne12, ثابت صحیح nb10, ثابت صحیح nb11,
                                   ثابت صحیح nb12, ثابت صحیح nb13, قطار_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ثابت صحیح عدد_بلاکیں = ceil_div(ne / QK_K, SYCL_CPY_بلاک_سائز);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, عدد_بلاکیں) * sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز), sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز)),
        [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]]{
            cpy_q_q<بلاک_q4_K, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, شے_ct1);
        });
}

ساکن خالی ggml_cpy_q5_K_q5_K(ثابت حرف * cx, حرف * cdst, ثابت صحیح ne, ثابت صحیح ne00, ثابت صحیح ne01,
                                   ثابت صحیح ne02, ثابت صحیح nb00, ثابت صحیح nb01, ثابت صحیح nb02, ثابت صحیح nb03,
                                   ثابت صحیح ne10, ثابت صحیح ne11, ثابت صحیح ne12, ثابت صحیح nb10, ثابت صحیح nb11,
                                   ثابت صحیح nb12, ثابت صحیح nb13, قطار_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ثابت صحیح عدد_بلاکیں = ceil_div(ne / QK_K, SYCL_CPY_بلاک_سائز);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, عدد_بلاکیں) * sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز), sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز)),
        [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]]{
            cpy_q_q<بلاک_q5_K, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, شے_ct1);
        });
}

ساکن خالی ggml_cpy_q6_K_q6_K(ثابت حرف * cx, حرف * cdst, ثابت صحیح ne, ثابت صحیح ne00, ثابت صحیح ne01,
                                   ثابت صحیح ne02, ثابت صحیح nb00, ثابت صحیح nb01, ثابت صحیح nb02, ثابت صحیح nb03,
                                   ثابت صحیح ne10, ثابت صحیح ne11, ثابت صحیح ne12, ثابت صحیح nb10, ثابت صحیح nb11,
                                   ثابت صحیح nb12, ثابت صحیح nb13, قطار_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ثابت صحیح عدد_بلاکیں = ceil_div(ne / QK_K, SYCL_CPY_بلاک_سائز);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, عدد_بلاکیں) * sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز), sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز)),
        [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]]{
            cpy_q_q<بلاک_q6_K, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, شے_ct1);
        });
}

ساکن خالی ggml_cpy_iq2_xxs_iq2_xxs(ثابت حرف * cx, حرف * cdst, ثابت صحیح ne, ثابت صحیح ne00, ثابت صحیح ne01,
                                   ثابت صحیح ne02, ثابت صحیح nb00, ثابت صحیح nb01, ثابت صحیح nb02, ثابت صحیح nb03,
                                   ثابت صحیح ne10, ثابت صحیح ne11, ثابت صحیح ne12, ثابت صحیح nb10, ثابت صحیح nb11,
                                   ثابت صحیح nb12, ثابت صحیح nb13, قطار_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ثابت صحیح عدد_بلاکیں = ceil_div(ne / QK_K, SYCL_CPY_بلاک_سائز);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, عدد_بلاکیں) * sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز), sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز)),
        [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]]{
            cpy_q_q<بلاک_iq2_xxs, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, شے_ct1);
        });
}

ساکن خالی ggml_cpy_iq2_xs_iq2_xs(ثابت حرف * cx, حرف * cdst, ثابت صحیح ne, ثابت صحیح ne00, ثابت صحیح ne01,
                                   ثابت صحیح ne02, ثابت صحیح nb00, ثابت صحیح nb01, ثابت صحیح nb02, ثابت صحیح nb03,
                                   ثابت صحیح ne10, ثابت صحیح ne11, ثابت صحیح ne12, ثابت صحیح nb10, ثابت صحیح nb11,
                                   ثابت صحیح nb12, ثابت صحیح nb13, قطار_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ثابت صحیح عدد_بلاکیں = ceil_div(ne / QK_K, SYCL_CPY_بلاک_سائز);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, عدد_بلاکیں) * sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز), sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز)),
        [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]]{
            cpy_q_q<بلاک_iq2_xs, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, شے_ct1);
        });
}

ساکن خالی ggml_cpy_iq2_s_iq2_s(ثابت حرف * cx, حرف * cdst, ثابت صحیح ne, ثابت صحیح ne00, ثابت صحیح ne01,
                                   ثابت صحیح ne02, ثابت صحیح nb00, ثابت صحیح nb01, ثابت صحیح nb02, ثابت صحیح nb03,
                                   ثابت صحیح ne10, ثابت صحیح ne11, ثابت صحیح ne12, ثابت صحیح nb10, ثابت صحیح nb11,
                                   ثابت صحیح nb12, ثابت صحیح nb13, قطار_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ثابت صحیح عدد_بلاکیں = ceil_div(ne / QK_K, SYCL_CPY_بلاک_سائز);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, عدد_بلاکیں) * sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز), sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز)),
        [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]]{
            cpy_q_q<بلاک_iq2_s, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, شے_ct1);
        });
}

ساکن خالی ggml_cpy_iq3_xxs_iq3_xxs(ثابت حرف * cx, حرف * cdst, ثابت صحیح ne, ثابت صحیح ne00, ثابت صحیح ne01,
                                   ثابت صحیح ne02, ثابت صحیح nb00, ثابت صحیح nb01, ثابت صحیح nb02, ثابت صحیح nb03,
                                   ثابت صحیح ne10, ثابت صحیح ne11, ثابت صحیح ne12, ثابت صحیح nb10, ثابت صحیح nb11,
                                   ثابت صحیح nb12, ثابت صحیح nb13, قطار_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ثابت صحیح عدد_بلاکیں = ceil_div(ne / QK_K, SYCL_CPY_بلاک_سائز);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, عدد_بلاکیں) * sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز), sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز)),
        [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]]{
            cpy_q_q<بلاک_iq3_xxs, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, شے_ct1);
        });
}

ساکن خالی ggml_cpy_iq1_s_iq1_s(ثابت حرف * cx, حرف * cdst, ثابت صحیح ne, ثابت صحیح ne00, ثابت صحیح ne01,
                                   ثابت صحیح ne02, ثابت صحیح nb00, ثابت صحیح nb01, ثابت صحیح nb02, ثابت صحیح nb03,
                                   ثابت صحیح ne10, ثابت صحیح ne11, ثابت صحیح ne12, ثابت صحیح nb10, ثابت صحیح nb11,
                                   ثابت صحیح nb12, ثابت صحیح nb13, قطار_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ثابت صحیح عدد_بلاکیں = ceil_div(ne / QK_K, SYCL_CPY_بلاک_سائز);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, عدد_بلاکیں) * sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز), sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز)),
        [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]]{
            cpy_q_q<بلاک_iq1_s, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, شے_ct1);
        });
}

ساکن خالی ggml_cpy_iq1_m_iq1_m(ثابت حرف * cx, حرف * cdst, ثابت صحیح ne, ثابت صحیح ne00, ثابت صحیح ne01,
                                   ثابت صحیح ne02, ثابت صحیح nb00, ثابت صحیح nb01, ثابت صحیح nb02, ثابت صحیح nb03,
                                   ثابت صحیح ne10, ثابت صحیح ne11, ثابت صحیح ne12, ثابت صحیح nb10, ثابت صحیح nb11,
                                   ثابت صحیح nb12, ثابت صحیح nb13, قطار_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ثابت صحیح عدد_بلاکیں = ceil_div(ne / QK_K, SYCL_CPY_بلاک_سائز);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, عدد_بلاکیں) * sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز), sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز)),
        [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]]{
            cpy_q_q<بلاک_iq1_m, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, شے_ct1);
        });
}

ساکن خالی ggml_cpy_iq4_nl_iq4_nl(ثابت حرف * cx, حرف * cdst, ثابت صحیح ne, ثابت صحیح ne00, ثابت صحیح ne01,
                                   ثابت صحیح ne02, ثابت صحیح nb00, ثابت صحیح nb01, ثابت صحیح nb02, ثابت صحیح nb03,
                                   ثابت صحیح ne10, ثابت صحیح ne11, ثابت صحیح ne12, ثابت صحیح nb10, ثابت صحیح nb11,
                                   ثابت صحیح nb12, ثابت صحیح nb13, قطار_ptr stream) {
    GGML_ASSERT(ne % QK4_NL == 0);
    ثابت صحیح عدد_بلاکیں = ceil_div(ne / QK4_NL, SYCL_CPY_بلاک_سائز);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, عدد_بلاکیں) * sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز), sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز)),
        [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]]{
            cpy_q_q<بلاک_iq4_nl, QK4_NL>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, شے_ct1);
        });
}

ساکن خالی ggml_cpy_iq3_s_iq3_s(ثابت حرف * cx, حرف * cdst, ثابت صحیح ne, ثابت صحیح ne00, ثابت صحیح ne01,
                                   ثابت صحیح ne02, ثابت صحیح nb00, ثابت صحیح nb01, ثابت صحیح nb02, ثابت صحیح nb03,
                                   ثابت صحیح ne10, ثابت صحیح ne11, ثابت صحیح ne12, ثابت صحیح nb10, ثابت صحیح nb11,
                                   ثابت صحیح nb12, ثابت صحیح nb13, قطار_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ثابت صحیح عدد_بلاکیں = ceil_div(ne / QK_K, SYCL_CPY_بلاک_سائز);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, عدد_بلاکیں) * sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز), sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز)),
        [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]]{
            cpy_q_q<بلاک_iq3_s, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, شے_ct1);
        });
}

ساکن خالی ggml_cpy_iq4_xs_iq4_xs(ثابت حرف * cx, حرف * cdst, ثابت صحیح ne, ثابت صحیح ne00, ثابت صحیح ne01,
                                   ثابت صحیح ne02, ثابت صحیح nb00, ثابت صحیح nb01, ثابت صحیح nb02, ثابت صحیح nb03,
                                   ثابت صحیح ne10, ثابت صحیح ne11, ثابت صحیح ne12, ثابت صحیح nb10, ثابت صحیح nb11,
                                   ثابت صحیح nb12, ثابت صحیح nb13, قطار_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ثابت صحیح عدد_بلاکیں = ceil_div(ne / QK_K, SYCL_CPY_بلاک_سائز);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, عدد_بلاکیں) * sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز), sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز)),
        [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]]{
            cpy_q_q<بلاک_iq4_xs, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, شے_ct1);
        });
}

#اگر_تعریف_شدہ GGML_SYCL_HAS_BF16
ساکن خالی ggml_cpy_f32_bf16_sycl(ثابت حرف * cx, حرف * cdst, ثابت صحیح ne, ثابت صحیح ne00, ثابت صحیح ne01,
                                  ثابت صحیح ne02, ثابت صحیح nb00, ثابت صحیح nb01, ثابت صحیح nb02, ثابت صحیح nb03,
                                  ثابت صحیح ne10, ثابت صحیح ne11, ثابت صحیح ne12, ثابت صحیح nb10, ثابت صحیح nb11,
                                  ثابت صحیح nb12, ثابت صحیح nb13, قطار_ptr stream) {
    ثابت صحیح عدد_بلاکیں = (ne + SYCL_CPY_بلاک_سائز - 1) / SYCL_CPY_بلاک_سائز;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, عدد_بلاکیں) * sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز),
                          sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز)),
        [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]]{
            cpy_f32_f16<cpy_1_f32_bf16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, شے_ct1);
        });
}

ساکن خالی ggml_cpy_bf16_f32_sycl(ثابت حرف * cx, حرف * cdst, ثابت صحیح ne, ثابت صحیح ne00, ثابت صحیح ne01,
                                  ثابت صحیح ne02, ثابت صحیح nb00, ثابت صحیح nb01, ثابت صحیح nb02, ثابت صحیح nb03,
                                  ثابت صحیح ne10, ثابت صحیح ne11, ثابت صحیح ne12, ثابت صحیح nb10, ثابت صحیح nb11,
                                  ثابت صحیح nb12, ثابت صحیح nb13, قطار_ptr stream) {
    ثابت صحیح عدد_بلاکیں = (ne + SYCL_CPY_بلاک_سائز - 1) / SYCL_CPY_بلاک_سائز;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, عدد_بلاکیں) * sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز),
                          sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز)),
        [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]]{
            cpy_f32_f16<cpy_1_bf16_f32>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, شے_ct1);
        });
}

ساکن خالی ggml_cpy_bf16_bf16_sycl(ثابت حرف * cx, حرف * cdst, ثابت صحیح ne, ثابت صحیح ne00, ثابت صحیح ne01,
                                  ثابت صحیح ne02, ثابت صحیح nb00, ثابت صحیح nb01, ثابت صحیح nb02, ثابت صحیح nb03,
                                  ثابت صحیح ne10, ثابت صحیح ne11, ثابت صحیح ne12, ثابت صحیح nb10, ثابت صحیح nb11,
                                  ثابت صحیح nb12, ثابت صحیح nb13, قطار_ptr stream) {
    ثابت صحیح عدد_بلاکیں = (ne + SYCL_CPY_بلاک_سائز - 1) / SYCL_CPY_بلاک_سائز;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, عدد_بلاکیں) * sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز),
                          sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز)),
        [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]]{
            cpy_f32_f16<cpy_1_bf16_bf16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                            nb10, nb11, nb12, nb13, شے_ct1);
        });
}

ساکن خالی ggml_cpy_f16_bf16_sycl(ثابت حرف * cx, حرف * cdst, ثابت صحیح ne, ثابت صحیح ne00, ثابت صحیح ne01,
                                  ثابت صحیح ne02, ثابت صحیح nb00, ثابت صحیح nb01, ثابت صحیح nb02, ثابت صحیح nb03,
                                  ثابت صحیح ne10, ثابت صحیح ne11, ثابت صحیح ne12, ثابت صحیح nb10, ثابت صحیح nb11,
                                  ثابت صحیح nb12, ثابت صحیح nb13, قطار_ptr stream) {
    ثابت صحیح عدد_بلاکیں = (ne + SYCL_CPY_بلاک_سائز - 1) / SYCL_CPY_بلاک_سائز;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, عدد_بلاکیں) * sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز),
                          sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز)),
        [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]]{
            cpy_f32_f16<cpy_1_f16_bf16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, شے_ct1);
        });
}

ساکن خالی ggml_cpy_bf16_f16_sycl(ثابت حرف * cx, حرف * cdst, ثابت صحیح ne, ثابت صحیح ne00, ثابت صحیح ne01,
                                  ثابت صحیح ne02, ثابت صحیح nb00, ثابت صحیح nb01, ثابت صحیح nb02, ثابت صحیح nb03,
                                  ثابت صحیح ne10, ثابت صحیح ne11, ثابت صحیح ne12, ثابت صحیح nb10, ثابت صحیح nb11,
                                  ثابت صحیح nb12, ثابت صحیح nb13, قطار_ptr stream) {
    ثابت صحیح عدد_بلاکیں = (ne + SYCL_CPY_بلاک_سائز - 1) / SYCL_CPY_بلاک_سائز;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, عدد_بلاکیں) * sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز),
                          sycl::range<3>(1, 1, SYCL_CPY_بلاک_سائز)),
        [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]]{
            cpy_f32_f16<cpy_1_bf16_f16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, شے_ct1);
        });
}
#اگر_ختم

خالی ggml_sycl_cpy(ggml_backend_sycl_context & ctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1) کوشش {
    // Unlike other operators ggml_sycl_cpy takes 2 distinct tensors instead of a dst ggml_tensor and rely on its src field
    GGML_SYCL_DEBUG("ggml_sycl_cpy: src0->type=%s, src1->type=%s\n",
        ggml_type_name(src0->type), ggml_type_name(src1->type));
    scope_op_debug_print scope_dbg_print(__func__, src1, /*num_src=*/0, debug_get_tensor_str("\tsrc0", src0));
    ثابت int64_t ne = ggml_nelements(src0);
    GGML_ASSERT(ne == ggml_nelements(src1));
    // the cpy kernels index in 32-bit
    GGML_ASSERT(ne <= INT32_MAX);

    GGML_TENSOR_ثنائی_OP_LOCALS01;

    SYCL_CHECK(ggml_sycl_سیٹ_device(ctx.device));
    قطار_ptr main_دھارا = ctx.stream();

    حرف * src0_ddc = (حرف *) src0->ڈیٹا;
    حرف * src1_ddc = (حرف *) src1->ڈیٹا;
    اگر ((src0->type == src1->type) && (ggml_is_contiguous(src0) && ggml_is_contiguous(src1))) {
        GGML_SYCL_DEBUG("%s: memcpy path\n", __func__);
        main_دھارا->حافظہ_نقل(src1_ddc, src0_ddc, ggml_nbytes(src0));
    } ورنہ اگر (src0->type == GGML_TYPE_F32 && ggml_sycl_is_quantized_type(src1->type)) {
        GGML_ASSERT(ggml_sycl_can_quantize_صفیں_sycl(src1->type));
        ggml_sycl_quantize_صفیں_sycl<اعشاری>(src0_ddc, src1_ddc, src0, src1, ne, ne00, ne01, ne02, nb00, nb01,
                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_دھارا);
    } ورنہ اگر (src0->type == GGML_TYPE_F16 && ggml_sycl_is_quantized_type(src1->type)) {
        GGML_ASSERT(ggml_sycl_can_quantize_صفیں_sycl(src1->type));
        ggml_sycl_quantize_صفیں_sycl<sycl::half>(src0_ddc, src1_ddc, src0, src1, ne, ne00, ne01, ne02, nb00,
                                                 nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                                 main_دھارا);
#اگر_تعریف_شدہ GGML_SYCL_HAS_BF16
    } ورنہ اگر (src0->type == GGML_TYPE_BF16 && ggml_sycl_is_quantized_type(src1->type)) {
        GGML_ASSERT(ggml_sycl_can_quantize_صفیں_sycl(src1->type));
        ggml_sycl_quantize_صفیں_sycl<ggml_bf16_t>(src0_ddc, src1_ddc, src0, src1, ne, ne00, ne01, ne02,
                                                  nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                  nb12, nb13, main_دھارا);
#اگر_ختم
    } ورنہ اگر (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_f32_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_دھارا);
    } ورنہ اگر (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F16) {
        ggml_cpy_f32_f16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_دھارا);
    } ورنہ اگر (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_I32) {
        ggml_cpy_f32_i32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_دھارا);
    } ورنہ اگر (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q8_0) {
        ggml_cpy_f32_q8_0_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_دھارا);
    } ورنہ اگر (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q4_0) {
        ggml_cpy_f32_q4_0_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_دھارا);
    } ورنہ اگر (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q4_1) {
        ggml_cpy_f32_q4_1_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_دھارا);
    } ورنہ اگر (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_f16_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_دھارا);
    } ورنہ اگر (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16) {
        ggml_cpy_f16_f16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_دھارا);
    } ورنہ اگر (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_Q4_0) {
        ggml_cpy_f16_q4_0_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02,
                               nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_دھارا);
    } ورنہ اگر (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_Q4_1) {
        ggml_cpy_f16_q4_1_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02,
                               nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_دھارا);
    } ورنہ اگر (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_Q5_0) {
        ggml_cpy_f16_q5_0_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02,
                               nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_دھارا);
    } ورنہ اگر (src0->type == GGML_TYPE_I16 && src1->type == GGML_TYPE_I16) {
        ggml_cpy_i16_i16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_دھارا);
    } ورنہ اگر (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_I32) {
        ggml_cpy_i32_i32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_دھارا);
    } ورنہ اگر (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_i32_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_دھارا);
    } ورنہ اگر (src0->type == GGML_TYPE_Q4_0 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q4_0_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_دھارا);
    } ورنہ اگر (src0->type == GGML_TYPE_Q4_1 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q4_1_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_دھارا);
    } ورنہ اگر (src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q8_0_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_دھارا);
    } ورنہ اگر (src0->type == GGML_TYPE_Q2_0 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q2_0_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                               nb10, nb11, nb12, nb13, main_دھارا);
    } ورنہ اگر (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q5_0) {
        ggml_cpy_f32_q5_0_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_دھارا);
    } ورنہ اگر (src0->type == GGML_TYPE_Q5_0 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q5_0_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_دھارا);
    } ورنہ اگر (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q5_1) {
        ggml_cpy_f32_q5_1_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_دھارا);
    } ورنہ اگر (src0->type == GGML_TYPE_Q5_1 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q5_1_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_دھارا);
    } ورنہ اگر (src0->type == GGML_TYPE_MXFP4 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_mxfp4_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                nb10, nb11, nb12, nb13, main_دھارا);
    } ورنہ اگر (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_IQ4_NL) {
        ggml_cpy_f32_iq4_nl_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                 nb10, nb11, nb12, nb13, main_دھارا);
    } ورنہ اگر (src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_Q8_0) {
        ggml_cpy_q8_0_q8_0(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_دھارا);
    } ورنہ اگر (src0->type == GGML_TYPE_Q5_0 && src1->type == GGML_TYPE_Q5_0) {
        ggml_cpy_q5_0_q5_0(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_دھارا);
    } ورنہ اگر (src0->type == GGML_TYPE_Q5_1 && src1->type == GGML_TYPE_Q5_1) {
        ggml_cpy_q5_1_q5_1(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_دھارا);
    } ورنہ اگر (src0->type == GGML_TYPE_Q4_0 && src1->type == GGML_TYPE_Q4_0) {
        ggml_cpy_q4_0_q4_0(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_دھارا);
    } ورنہ اگر (src0->type == GGML_TYPE_Q4_1 && src1->type == GGML_TYPE_Q4_1) {
        ggml_cpy_q4_1_q4_1(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_دھارا);
    } ورنہ اگر (src0->type == GGML_TYPE_Q1_0 && src1->type == GGML_TYPE_Q1_0) {
        ggml_cpy_q1_0_q1_0(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_دھارا);
    } ورنہ اگر (src0->type == GGML_TYPE_Q2_0 && src1->type == GGML_TYPE_Q2_0) {
        ggml_cpy_q2_0_q2_0(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_دھارا);
    } ورنہ اگر (src0->type == GGML_TYPE_MXFP4 && src1->type == GGML_TYPE_MXFP4) {
        ggml_cpy_mxfp4_mxfp4(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_دھارا);
    } ورنہ اگر (src0->type == GGML_TYPE_NVFP4 && src1->type == GGML_TYPE_NVFP4) {
        ggml_cpy_nvfp4_nvfp4(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_دھارا);
    } ورنہ اگر (src0->type == GGML_TYPE_Q2_K && src1->type == GGML_TYPE_Q2_K) {
        ggml_cpy_q2_K_q2_K(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_دھارا);
    } ورنہ اگر (src0->type == GGML_TYPE_Q3_K && src1->type == GGML_TYPE_Q3_K) {
        ggml_cpy_q3_K_q3_K(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_دھارا);
    } ورنہ اگر (src0->type == GGML_TYPE_Q4_K && src1->type == GGML_TYPE_Q4_K) {
        ggml_cpy_q4_K_q4_K(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_دھارا);
    } ورنہ اگر (src0->type == GGML_TYPE_Q5_K && src1->type == GGML_TYPE_Q5_K) {
        ggml_cpy_q5_K_q5_K(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_دھارا);
    } ورنہ اگر (src0->type == GGML_TYPE_Q6_K && src1->type == GGML_TYPE_Q6_K) {
        ggml_cpy_q6_K_q6_K(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_دھارا);
    } ورنہ اگر (src0->type == GGML_TYPE_IQ2_XXS && src1->type == GGML_TYPE_IQ2_XXS) {
        ggml_cpy_iq2_xxs_iq2_xxs(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_دھارا);
    } ورنہ اگر (src0->type == GGML_TYPE_IQ2_XS && src1->type == GGML_TYPE_IQ2_XS) {
        ggml_cpy_iq2_xs_iq2_xs(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_دھارا);
    } ورنہ اگر (src0->type == GGML_TYPE_IQ2_S && src1->type == GGML_TYPE_IQ2_S) {
        ggml_cpy_iq2_s_iq2_s(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_دھارا);
    } ورنہ اگر (src0->type == GGML_TYPE_IQ3_XXS && src1->type == GGML_TYPE_IQ3_XXS) {
        ggml_cpy_iq3_xxs_iq3_xxs(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_دھارا);
    } ورنہ اگر (src0->type == GGML_TYPE_IQ1_S && src1->type == GGML_TYPE_IQ1_S) {
        ggml_cpy_iq1_s_iq1_s(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_دھارا);
    } ورنہ اگر (src0->type == GGML_TYPE_IQ1_M && src1->type == GGML_TYPE_IQ1_M) {
        ggml_cpy_iq1_m_iq1_m(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_دھارا);
    } ورنہ اگر (src0->type == GGML_TYPE_IQ4_NL && src1->type == GGML_TYPE_IQ4_NL) {
        ggml_cpy_iq4_nl_iq4_nl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_دھارا);
    } ورنہ اگر (src0->type == GGML_TYPE_IQ3_S && src1->type == GGML_TYPE_IQ3_S) {
        ggml_cpy_iq3_s_iq3_s(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_دھارا);
    } ورنہ اگر (src0->type == GGML_TYPE_IQ4_XS && src1->type == GGML_TYPE_IQ4_XS) {
        ggml_cpy_iq4_xs_iq4_xs(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_دھارا);
#اگر_تعریف_شدہ GGML_SYCL_HAS_BF16
    } ورنہ اگر (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_BF16) {
        ggml_cpy_f32_bf16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_دھارا);
    } ورنہ اگر (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_bf16_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_دھارا);
    } ورنہ اگر (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16) {
        ggml_cpy_bf16_bf16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_دھارا);
    } ورنہ اگر (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_BF16) {
        ggml_cpy_f16_bf16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_دھارا);
    } ورنہ اگر (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F16) {
        ggml_cpy_bf16_f16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_دھارا);
#اگر_ختم
    } ورنہ {
        GGML_LOG_ERROR("%s: unsupported type combination (%s to %s)\n", __func__, ggml_type_name(src0->type),
                       ggml_type_name(src1->type));
        GGML_ABORT("fatal error");
    }
} پکڑو (ثابت sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::سطر_ختم;
    std::باہر_نکلو(1);
}

خالی ggml_sycl_dup(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_cpy(ctx, dst->src[0], dst);
}
