#شامل "set_rows.hpp"
#شامل "cpy.hpp"

#شامل "ggml-quants.h"

#شامل <vector>

نامخلا utils {
سانچہ<قسم_نام T>
ساکن constexpr منطقی is_arithmetic_v() {
    واپس std::is_arithmetic_v<T> || std::is_same_v<T, sycl::half>
#اگر_تعریف_شدہ GGML_SYCL_HAS_BF16
        || std::is_same_v<T, sycl::ext::oneapi::bfloat16>
#اگر_ختم
        ;
}
}

سانچہ<قسم_نام TIn, قسم_نام TOut>
ساکن اندرونی std::enable_if_t<utils::is_arithmetic_v<TIn>() && utils::is_arithmetic_v<TOut>(), خالی>
convert (ثابت حرف* src, حرف* dst) {
    خودکار src_val = *reinterpret_cast<ثابت TIn*>(src);
    خودکار dst_val = sycl::vec<TIn, 1>(src_val).سانچہ convert<TOut, sycl::rounding_منوال::automatic>()[0];
   *reinterpret_cast<TOut*>(dst) = dst_val;
}

#اگر_تعریف_شدہ GGML_SYCL_HAS_BF16
// sycl::vec::convert does not provide a half -> bfloat16 path, so route through float.
سانچہ<>
اندرونی خالی convert<sycl::half, sycl::ext::oneapi::bfloat16>(ثابت حرف* src, حرف* dst) {
    ثابت اعشاری tmp = sycl::vec<sycl::half, 1>(*reinterpret_cast<ثابت sycl::half*>(src))
                          .سانچہ convert<اعشاری, sycl::rounding_منوال::automatic>()[0];
    *reinterpret_cast<sycl::ext::oneapi::bfloat16*>(dst) = sycl::ext::oneapi::bfloat16(tmp);
}
#اگر_ختم

سانچہ <قسم_نام TIn, قسم_نام TIdx, قسم_نام بلاک_Type, صحیح qk, cpy_kernel_t cpyblck>
ساکن خالی سیٹ_صفیں_sycl_q(ثابت حرف * __restrict__ src0_d,
                            ثابت TIdx * __restrict__ src1_d,
                            بلاک_Type * __restrict__ dst_d,
                            // tensor dimensions src0 and src1
                            ثابت int64_t ne00,
                            ثابت int64_t ne01,
                            ثابت int64_t ne02,
                            ثابت int64_t ne03,
                            ثابت int64_t ne10,
                            ثابت int64_t ne11,
                            ثابت int64_t ne12,
                            ثابت int64_t ne13,
                            // strides for src0
                            ثابت size_t  nb00,
                            ثابت size_t  nb01,
                            ثابت size_t  nb02,
                            ثابت size_t  nb03,
                            // strides for src1
                            ثابت size_t  nb10,
                            ثابت size_t  nb11,
                            ثابت size_t  nb12,
                            ثابت size_t  nb13,
                            // strides for dst
                            ثابت size_t  nb1,
                            ثابت size_t  nb2,
                            ثابت size_t  nb3,
                            قطار_ptr     stream) {
    ثابت int64_t کل_بلاکیں = (ne00 * ne01 * ne02 * ne03) / qk;
    constexpr صحیح بلاک_سائز_3   = 256;
    ثابت int64_t grid_سائز    = ceil_div(کل_بلاکیں, بلاک_سائز_3);

    stream->parallel_for(sycl::nd_range<1>(grid_سائز * بلاک_سائز_3, بلاک_سائز_3), [=](sycl::nd_شے<1> شے_ct1) {
        ثابت int64_t i = شے_ct1.get_global_خطی_id();
        اگر (i >= کل_بلاکیں) {
            واپس;
        }
        ثابت int64_t i_بنیاد      = i * qk;
        ثابت int64_t i03         = i_بنیاد / (ne00 * ne01 * ne02);
        ثابت int64_t rem1        = i_بنیاد - i03 * (ne00 * ne01 * ne02);
        ثابت int64_t i02         = rem1 / (ne00 * ne01);
        ثابت int64_t rem2        = rem1 - i02 * ne00 * ne01;
        ثابت int64_t i01         = rem2 / ne00;
        ثابت int64_t i00         = rem2 - i01 * ne00;
        ثابت int64_t i12         = i03 % ne12;
        ثابت int64_t i11         = i02 % ne11;
        ثابت int64_t i10         = i01;
        ثابت size_t  src_انحراف  = calculate_انحراف<3>({ nb01, nb02, nb03 }, { i01, i02, i03 });
        ثابت حرف *  src_بلاک   = src0_d + src_انحراف + i00 * حجم(TIn);
        ثابت size_t  src1_انحراف = calculate_انحراف<3>({ nb10, nb11, nb12 }, { i10, i11, i12 });
        ثابت int64_t dst_صف     = src1_d[src1_انحراف / حجم(TIdx)];
        ثابت size_t  dst_انحراف =
            calculate_انحراف<3>({ nb1, nb2, nb3 }, { dst_صف, i02, i03 }) + (i00 / qk) * حجم(بلاک_Type);
        حرف * dst_بلاک = reinterpret_cast<حرف *>(reinterpret_cast<حرف *>(dst_d) + dst_انحراف);
        اگر constexpr (std::is_same_v<TIn, اعشاری>) {
            cpyblck(src_بلاک, dst_بلاک);
        } ورنہ {
            اعشاری src_بلاک_f32[qk];
            ثابت TIn * src_بلاک_t = reinterpret_cast<ثابت TIn *>(src_بلاک);
            برائے (صحیح j = 0; j < qk; ++j) {
                src_بلاک_f32[j] = (اعشاری) src_بلاک_t[j];
            }
            cpyblck(reinterpret_cast<ثابت حرف *>(src_بلاک_f32), dst_بلاک);
        }
    });
    GGML_غیر_مستعمل(ne10);
    GGML_غیر_مستعمل(ne13);
    GGML_غیر_مستعمل(nb00);
    GGML_غیر_مستعمل(nb13);
}

سانچہ<قسم_نام بلاک_Type>
استعمال quantize_صف_qk_t = خالی (*)(ثابت اعشاری *, بلاک_Type *, int64_t);

استعمال quantize_صفیں_f_t = size_t (*)(ثابت اعشاری *, خالی *, int64_t, int64_t, ثابت اعشاری *);

سانچہ <قسم_نام TIn, قسم_نام TIdx, قسم_نام بلاک_Type, صحیح qk, quantize_صف_qk_t<بلاک_Type> quantize_صف>
ساکن خالی سیٹ_صفیں_sycl_qk_host(
        ثابت ggml_tensor * src0,
        ثابت ggml_tensor * src1,
        ggml_tensor * dst,
        ثابت int64_t ne00,
        ثابت int64_t ne01,
        ثابت int64_t ne02,
        ثابت int64_t ne03,
        ثابت int64_t ne11,
        ثابت int64_t ne12,
        ثابت size_t nb01,
        ثابت size_t nb02,
        ثابت size_t nb03,
        ثابت size_t nb10,
        ثابت size_t nb11,
        ثابت size_t nb12,
        ثابت size_t nb1,
        ثابت size_t nb2,
        ثابت size_t nb3,
        قطار_ptr stream) {
    GGML_ASSERT(ne00 % qk == 0);

    ثابت size_t src0_بائٹیں = ggml_nbytes(src0);
    ثابت size_t src1_بائٹیں = ggml_nbytes(src1);

    std::vector<حرف> src0_host(src0_بائٹیں);
    std::vector<حرف> src1_host(src1_بائٹیں);

    stream->حافظہ_نقل(src0_host.ڈیٹا(), src0->ڈیٹا, src0_بائٹیں);
    stream->حافظہ_نقل(src1_host.ڈیٹا(), src1->ڈیٹا, src1_بائٹیں);
    stream->wait();

    std::vector<اعشاری> src_صف_f32(ne00);
    ثابت int64_t nblocks = ne00 / qk;
    std::vector<بلاک_Type> dst_صف_q(nblocks);

    برائے (int64_t i03 = 0; i03 < ne03; ++i03) {
        برائے (int64_t i02 = 0; i02 < ne02; ++i02) {
            برائے (int64_t i01 = 0; i01 < ne01; ++i01) {
                ثابت int64_t i12 = i03 % ne12;
                ثابت int64_t i11 = i02 % ne11;
                ثابت int64_t i10 = i01;

                ثابت size_t src1_انحراف = calculate_انحراف<3>({ nb10, nb11, nb12 }, { i10, i11, i12 });
                ثابت int64_t dst_صف = *(ثابت TIdx *) (src1_host.ڈیٹا() + src1_انحراف);

                ثابت size_t src0_صف_انحراف = calculate_انحراف<3>({ nb01, nb02, nb03 }, { i01, i02, i03 });
                ثابت TIn * src_صف = reinterpret_cast<ثابت TIn *>(src0_host.ڈیٹا() + src0_صف_انحراف);

                برائے (int64_t i00 = 0; i00 < ne00; ++i00) {
                    src_صف_f32[i00] = (اعشاری) src_صف[i00];
                }

                quantize_صف(src_صف_f32.ڈیٹا(), dst_صف_q.ڈیٹا(), ne00);

                ثابت size_t dst_انحراف = calculate_انحراف<3>({ nb1, nb2, nb3 }, { dst_صف, i02, i03 });
                stream->حافظہ_نقل((حرف *) dst->ڈیٹا + dst_انحراف, dst_صف_q.ڈیٹا(), nblocks * حجم(بلاک_Type));
                stream->wait();
            }
        }
    }
}

سانچہ <قسم_نام TIn, قسم_نام TIdx, قسم_نام بلاک_Type, صحیح qk, quantize_صفیں_f_t quantize_صفیں>
ساکن خالی سیٹ_صفیں_sycl_iq_host(
        ثابت ggml_tensor * src0,
        ثابت ggml_tensor * src1,
        ggml_tensor * dst,
        ثابت int64_t ne00,
        ثابت int64_t ne01,
        ثابت int64_t ne02,
        ثابت int64_t ne03,
        ثابت int64_t ne11,
        ثابت int64_t ne12,
        ثابت size_t nb01,
        ثابت size_t nb02,
        ثابت size_t nb03,
        ثابت size_t nb10,
        ثابت size_t nb11,
        ثابت size_t nb12,
        ثابت size_t nb1,
        ثابت size_t nb2,
        ثابت size_t nb3,
        قطار_ptr stream) {
    GGML_ASSERT(ne00 % qk == 0);

    ثابت size_t src0_بائٹیں = ggml_nbytes(src0);
    ثابت size_t src1_بائٹیں = ggml_nbytes(src1);

    std::vector<حرف> src0_host(src0_بائٹیں);
    std::vector<حرف> src1_host(src1_بائٹیں);

    stream->حافظہ_نقل(src0_host.ڈیٹا(), src0->ڈیٹا, src0_بائٹیں);
    stream->حافظہ_نقل(src1_host.ڈیٹا(), src1->ڈیٹا, src1_بائٹیں);
    stream->wait();

    std::vector<اعشاری> src_صف_f32(ne00);
    ثابت int64_t nblocks = ne00 / qk;
    std::vector<بلاک_Type> dst_صف_q(nblocks);

    برائے (int64_t i03 = 0; i03 < ne03; ++i03) {
        برائے (int64_t i02 = 0; i02 < ne02; ++i02) {
            برائے (int64_t i01 = 0; i01 < ne01; ++i01) {
                ثابت int64_t i12 = i03 % ne12;
                ثابت int64_t i11 = i02 % ne11;
                ثابت int64_t i10 = i01;

                ثابت size_t src1_انحراف = calculate_انحراف<3>({ nb10, nb11, nb12 }, { i10, i11, i12 });
                ثابت int64_t dst_صف = *(ثابت TIdx *) (src1_host.ڈیٹا() + src1_انحراف);

                ثابت size_t src0_صف_انحراف = calculate_انحراف<3>({ nb01, nb02, nb03 }, { i01, i02, i03 });
                ثابت TIn * src_صف = reinterpret_cast<ثابت TIn *>(src0_host.ڈیٹا() + src0_صف_انحراف);

                برائے (int64_t i00 = 0; i00 < ne00; ++i00) {
                    src_صف_f32[i00] = (اعشاری) src_صف[i00];
                }

                quantize_صفیں(src_صف_f32.ڈیٹا(), dst_صف_q.ڈیٹا(), 1, ne00, خالی_اشاریہ);

                ثابت size_t dst_انحراف = calculate_انحراف<3>({ nb1, nb2, nb3 }, { dst_صف, i02, i03 });
                stream->حافظہ_نقل((حرف *) dst->ڈیٹا + dst_انحراف, dst_صف_q.ڈیٹا(), nblocks * حجم(بلاک_Type));
                stream->wait();
            }
        }
    }
}

سانچہ<قسم_نام TIn, قسم_نام TIdx, قسم_نام TOut>
ساکن خالی k_سیٹ_صفیں(
        ثابت حرف * __restrict__ src0, ثابت TIdx * __restrict__ src1, حرف * __restrict__ dst,
        ثابت int64_t ne00, ثابت int64_t ne01, ثابت int64_t ne02,
        ثابت int64_t ne11, ثابت int64_t ne12,
        ثابت size_t nb01, ثابت size_t nb02, ثابت size_t nb03,
        ثابت size_t nb10, ثابت size_t nb11, ثابت size_t nb12,
        ثابت size_t nb1, ثابت size_t nb2, ثابت size_t nb3,
        ثابت size_t src_type_سائز, ثابت size_t dst_type_سائز,
        ثابت int64_t کل_elements,
        ثابت sycl::nd_شے<1> & شے_ct1) {

    ثابت int64_t i = شے_ct1.get_global_خطی_id();
    اگر (i >= کل_elements) {
        واپس;
    }

    ثابت int64_t i03 = i / (ne00 * ne01 * ne02);
    ثابت int64_t i02 = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    ثابت int64_t i01 = (i - i03 * ne00 * ne01 * ne02 - i02 * ne00 * ne01) / ne00;
    ثابت int64_t i00 = i - i03 * ne00 * ne01 * ne02 - i02 * ne00 * ne01 - i01 * ne00;

    ثابت int64_t i12 = i03 % ne12;
    ثابت int64_t i11 = i02 % ne11;
    ثابت int64_t i10 = i01;

    ثابت int64_t dst_صف = *(ثابت TIdx *)((ثابت حرف *)src1 + calculate_انحراف<3>({nb10, nb11, nb12}, {i10, i11, i12}));

    ثابت حرف * src0_صف = src0 + calculate_انحراف<3>({nb01, nb02, nb03}, {i01, i02, i03});
    ثابت حرف * src_elem = src0_صف + i00 * src_type_سائز;
    حرف * dst_صف_ptr = dst + dst_صف*nb1 + i02*nb2 + i03*nb3;
    حرف * dst_elem = dst_صف_ptr + i00 * dst_type_سائز;

    convert<TIn, TOut>(src_elem, dst_elem);
}

سانچہ<قسم_نام TIn, قسم_نام TIdx, قسم_نام TOut>
ساکن خالی سیٹ_صفیں_sycl(
        ثابت حرف * src0_d, ثابت TIdx * src1_d, حرف * dst_d,
        ثابت int64_t ne00, ثابت int64_t ne01, ثابت int64_t ne02, ثابت int64_t ne03,
        ثابت int64_t ne11, ثابت int64_t ne12, ثابت size_t nb01, ثابت size_t nb02, ثابت size_t nb03,
        ثابت size_t nb10, ثابت size_t nb11, ثابت size_t nb12,
        ثابت size_t nb1, ثابت size_t nb2, ثابت size_t nb3,
        ثابت size_t src_type_سائز, ثابت size_t dst_type_سائز,
        قطار_ptr stream) {

    ثابت int64_t کل_elements = ne00 * ne01 * ne02 * ne03;

    constexpr صحیح بلاک_سائز_3 = 64;
    ثابت int64_t grid_سائز = ceil_div(کل_elements, بلاک_سائز_3);

    stream->parallel_for(
        sycl::nd_range<1>(grid_سائز * بلاک_سائز_3, بلاک_سائز_3),
        [=](sycl::nd_شے<1> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
            k_سیٹ_صفیں<TIn, TIdx, TOut>(
                src0_d, src1_d, dst_d,
                ne00, ne01, ne02,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                src_type_سائز, dst_type_سائز,
                کل_elements,
                شے_ct1
            );
        }
    );
}

سانچہ<قسم_نام TIn, قسم_نام TIdx>
ساکن خالی سیٹ_صفیں_sycl(ggml_backend_sycl_context & ctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst) {
    ثابت حرف * src0_d = (ثابت حرف *)src0->ڈیٹا;
    ثابت TIdx * src1_d = (ثابت TIdx *)src1->ڈیٹا;

    GGML_TENSOR_ثنائی_OP_LOCALS

    dpct::قطار_ptr stream = ctx.stream();
    انتخاب (dst->type) {
        حالت GGML_TYPE_F32:
            سیٹ_صفیں_sycl<TIn, TIdx, اعشاری>(
                src0_d, src1_d, (حرف *)dst->ڈیٹا,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                حجم(TIn), حجم(اعشاری),
                stream
            );
            توڑو;
        حالت GGML_TYPE_F16:
            dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
            سیٹ_صفیں_sycl<TIn, TIdx, sycl::half>(
                src0_d, src1_d, (حرف *)dst->ڈیٹا,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                حجم(TIn), حجم(sycl::half),
                stream
            );
            توڑو;
#اگر_تعریف_شدہ GGML_SYCL_HAS_BF16
        حالت GGML_TYPE_BF16:
            سیٹ_صفیں_sycl<TIn, TIdx, sycl::ext::oneapi::bfloat16>(
                src0_d, src1_d, (حرف *)dst->ڈیٹا,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                حجم(TIn), حجم(sycl::ext::oneapi::bfloat16),
                stream
            );
            توڑو;
#اگر_ختم
        حالت GGML_TYPE_Q8_0:
            سیٹ_صفیں_sycl_q<TIn, TIdx, بلاک_q8_0, QK8_0, cpy_blck_f32_q8_0>(
                src0_d, src1_d, (بلاک_q8_0 *) dst->ڈیٹا, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            توڑو;
        حالت GGML_TYPE_Q1_0:
            سیٹ_صفیں_sycl_q<TIn, TIdx, بلاک_q1_0, QK1_0, cpy_blck_f32_q1_0>(
                src0_d, src1_d, (بلاک_q1_0 *) dst->ڈیٹا, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            توڑو;
        حالت GGML_TYPE_Q2_0:
            سیٹ_صفیں_sycl_q<TIn, TIdx, بلاک_q2_0, QK2_0, cpy_blck_f32_q2_0>(
                src0_d, src1_d, (بلاک_q2_0 *) dst->ڈیٹا, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            توڑو;
        حالت GGML_TYPE_Q5_1:
            سیٹ_صفیں_sycl_q<TIn, TIdx, بلاک_q5_1, QK5_1, cpy_blck_f32_q5_1>(
                src0_d, src1_d, (بلاک_q5_1 *) dst->ڈیٹا, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            توڑو;
        حالت GGML_TYPE_Q5_0:
            سیٹ_صفیں_sycl_q<TIn, TIdx, بلاک_q5_0, QK5_0, cpy_blck_f32_q5_0>(
                src0_d, src1_d, (بلاک_q5_0 *) dst->ڈیٹا, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            توڑو;
        حالت GGML_TYPE_Q4_1:
            سیٹ_صفیں_sycl_q<TIn, TIdx, بلاک_q4_1, QK4_1, cpy_blck_f32_q4_1>(
                src0_d, src1_d, (بلاک_q4_1 *) dst->ڈیٹا, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            توڑو;
        حالت GGML_TYPE_Q4_0:
            سیٹ_صفیں_sycl_q<TIn, TIdx, بلاک_q4_0, QK4_0, cpy_blck_f32_q4_0>(
                src0_d, src1_d, (بلاک_q4_0 *) dst->ڈیٹا, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            توڑو;
        حالت GGML_TYPE_IQ4_NL:
            سیٹ_صفیں_sycl_q<TIn, TIdx, بلاک_iq4_nl, QK4_NL, cpy_blck_f32_iq4_nl>(
                src0_d, src1_d, (بلاک_iq4_nl *) dst->ڈیٹا, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            توڑو;
        حالت GGML_TYPE_MXFP4:
            سیٹ_صفیں_sycl_q<TIn, TIdx, بلاک_mxfp4, QK_MXFP4, cpy_blck_f32_mxfp4>(
                src0_d, src1_d, (بلاک_mxfp4 *) dst->ڈیٹا, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            توڑو;
        حالت GGML_TYPE_NVFP4:
            سیٹ_صفیں_sycl_q<TIn, TIdx, بلاک_nvfp4, QK_NVFP4, cpy_blck_f32_nvfp4>(
                src0_d, src1_d, (بلاک_nvfp4 *) dst->ڈیٹا, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            توڑو;
        حالت GGML_TYPE_Q2_K:
            سیٹ_صفیں_sycl_qk_host<TIn, TIdx, بلاک_q2_K, QK_K, quantize_صف_q2_K_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            توڑو;
        حالت GGML_TYPE_Q3_K:
            سیٹ_صفیں_sycl_qk_host<TIn, TIdx, بلاک_q3_K, QK_K, quantize_صف_q3_K_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            توڑو;
        حالت GGML_TYPE_Q4_K:
            سیٹ_صفیں_sycl_qk_host<TIn, TIdx, بلاک_q4_K, QK_K, quantize_صف_q4_K_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            توڑو;
        حالت GGML_TYPE_Q5_K:
            سیٹ_صفیں_sycl_qk_host<TIn, TIdx, بلاک_q5_K, QK_K, quantize_صف_q5_K_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            توڑو;
        حالت GGML_TYPE_Q6_K:
            سیٹ_صفیں_sycl_qk_host<TIn, TIdx, بلاک_q6_K, QK_K, quantize_صف_q6_K_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            توڑو;
        حالت GGML_TYPE_IQ2_XXS:
            سیٹ_صفیں_sycl_iq_host<TIn, TIdx, بلاک_iq2_xxs, QK_K, quantize_iq2_xxs>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            توڑو;
        حالت GGML_TYPE_IQ2_XS:
            سیٹ_صفیں_sycl_iq_host<TIn, TIdx, بلاک_iq2_xs, QK_K, quantize_iq2_xs>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            توڑو;
        حالت GGML_TYPE_IQ2_S:
            سیٹ_صفیں_sycl_iq_host<TIn, TIdx, بلاک_iq2_s, QK_K, quantize_iq2_s>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            توڑو;
        حالت GGML_TYPE_IQ3_XXS:
            سیٹ_صفیں_sycl_qk_host<TIn, TIdx, بلاک_iq3_xxs, QK_K, quantize_صف_iq3_xxs_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            توڑو;
        حالت GGML_TYPE_IQ3_S:
            سیٹ_صفیں_sycl_qk_host<TIn, TIdx, بلاک_iq3_s, QK_K, quantize_صف_iq3_s_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            توڑو;
        حالت GGML_TYPE_IQ1_S:
            سیٹ_صفیں_sycl_iq_host<TIn, TIdx, بلاک_iq1_s, QK_K, quantize_iq1_s>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            توڑو;
        حالت GGML_TYPE_IQ1_M:
            سیٹ_صفیں_sycl_iq_host<TIn, TIdx, بلاک_iq1_m, QK_K, quantize_iq1_m>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            توڑو;
        حالت GGML_TYPE_IQ4_XS:
            سیٹ_صفیں_sycl_qk_host<TIn, TIdx, بلاک_iq4_xs, QK_K, quantize_صف_iq4_xs_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            توڑو;
        طےشدہ:
            GGML_ABORT("Unsupported tensor type: src0 %s src1 %s dst %s", ggml_type_name(dst->src[0]->type),
                ggml_type_name(dst->src[1]->type), ggml_type_name(dst->type));
            توڑو;
    }
}

خالی ggml_sycl_op_سیٹ_صفیں(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->src[1]->type == GGML_TYPE_I64 || dst->src[1]->type == GGML_TYPE_I32);

    // dispatch on the index type (src1) and the source value type (src0)
    اگر (src0->type == GGML_TYPE_F16) {
        اگر (src1->type == GGML_TYPE_I64) {
            سیٹ_صفیں_sycl<sycl::half, int64_t>(ctx, src0, src1, dst);
        } ورنہ {
            سیٹ_صفیں_sycl<sycl::half, int32_t>(ctx, src0, src1, dst);
        }
    } ورنہ {
        اگر (src1->type == GGML_TYPE_I64) {
            سیٹ_صفیں_sycl<اعشاری, int64_t>(ctx, src0, src1, dst);
        } ورنہ {
            سیٹ_صفیں_sycl<اعشاری, int32_t>(ctx, src0, src1, dst);
        }
    }
}
