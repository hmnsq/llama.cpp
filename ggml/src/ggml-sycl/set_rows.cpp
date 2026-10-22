#ضمن "set_rows.hpp"
#ضمن "cpy.hpp"

#ضمن "ggml-quants.h"

#ضمن <vector>

نطاق utils {
سانچو<قسم_نالو T>
ساكن constexpr منطقي is_arithmetic_v() {
    أرجع std::is_arithmetic_v<T> || std::is_same_v<T, sycl::half>
#جيڪڏهن_وصف_ٿيل GGML_SYCL_HAS_BF16
        || std::is_same_v<T, sycl::ext::oneapi::bfloat16>
#جيڪڏهن_پورو
        ;
}
}

سانچو<قسم_نالو TIn, قسم_نالو TOut>
ساكن مضمن std::enable_if_t<utils::is_arithmetic_v<TIn>() && utils::is_arithmetic_v<TOut>(), فراغ>
convert (ثابت حرف* src, حرف* dst) {
    خودڪار src_val = *reinterpret_cast<ثابت TIn*>(src);
    خودڪار dst_val = sycl::vec<TIn, 1>(src_val).سانچو convert<TOut, sycl::rounding_گهڻائي::automatic>()[0];
   *reinterpret_cast<TOut*>(dst) = dst_val;
}

#جيڪڏهن_وصف_ٿيل GGML_SYCL_HAS_BF16
// sycl::vec::convert does not provide a half -> bfloat16 path, so route through float.
سانچو<>
مضمن فراغ convert<sycl::half, sycl::ext::oneapi::bfloat16>(ثابت حرف* src, حرف* dst) {
    ثابت عائم tmp = sycl::vec<sycl::half, 1>(*reinterpret_cast<ثابت sycl::half*>(src))
                          .سانچو convert<عائم, sycl::rounding_گهڻائي::automatic>()[0];
    *reinterpret_cast<sycl::ext::oneapi::bfloat16*>(dst) = sycl::ext::oneapi::bfloat16(tmp);
}
#جيڪڏهن_پورو

سانچو <قسم_نالو TIn, قسم_نالو TIdx, قسم_نالو بلاڪ_Type, صحيح qk, cpy_kernel_t cpyblck>
ساكن فراغ سيٽ_صفون_sycl_q(ثابت حرف * __restrict__ src0_d,
                            ثابت TIdx * __restrict__ src1_d,
                            بلاڪ_Type * __restrict__ dst_d,
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
    ثابت int64_t ڪل_بلاڪون = (ne00 * ne01 * ne02 * ne03) / qk;
    constexpr صحيح بلاڪ_سائيز_3   = 256;
    ثابت int64_t grid_سائيز    = ceil_div(ڪل_بلاڪون, بلاڪ_سائيز_3);

    stream->parallel_for(sycl::nd_range<1>(grid_سائيز * بلاڪ_سائيز_3, بلاڪ_سائيز_3), [=](sycl::nd_شي<1> شي_ct1) {
        ثابت int64_t i = شي_ct1.get_global_لڪيري_id();
        إذا (i >= ڪل_بلاڪون) {
            أرجع;
        }
        ثابت int64_t i_بنياد      = i * qk;
        ثابت int64_t i03         = i_بنياد / (ne00 * ne01 * ne02);
        ثابت int64_t rem1        = i_بنياد - i03 * (ne00 * ne01 * ne02);
        ثابت int64_t i02         = rem1 / (ne00 * ne01);
        ثابت int64_t rem2        = rem1 - i02 * ne00 * ne01;
        ثابت int64_t i01         = rem2 / ne00;
        ثابت int64_t i00         = rem2 - i01 * ne00;
        ثابت int64_t i12         = i03 % ne12;
        ثابت int64_t i11         = i02 % ne11;
        ثابت int64_t i10         = i01;
        ثابت size_t  src_وٿي  = calculate_وٿي<3>({ nb01, nb02, nb03 }, { i01, i02, i03 });
        ثابت حرف *  src_بلاڪ   = src0_d + src_وٿي + i00 * حجم(TIn);
        ثابت size_t  src1_وٿي = calculate_وٿي<3>({ nb10, nb11, nb12 }, { i10, i11, i12 });
        ثابت int64_t dst_صف     = src1_d[src1_وٿي / حجم(TIdx)];
        ثابت size_t  dst_وٿي =
            calculate_وٿي<3>({ nb1, nb2, nb3 }, { dst_صف, i02, i03 }) + (i00 / qk) * حجم(بلاڪ_Type);
        حرف * dst_بلاڪ = reinterpret_cast<حرف *>(reinterpret_cast<حرف *>(dst_d) + dst_وٿي);
        إذا constexpr (std::is_same_v<TIn, عائم>) {
            cpyblck(src_بلاڪ, dst_بلاڪ);
        } وإلا {
            عائم src_بلاڪ_f32[qk];
            ثابت TIn * src_بلاڪ_t = reinterpret_cast<ثابت TIn *>(src_بلاڪ);
            لكل (صحيح j = 0; j < qk; ++j) {
                src_بلاڪ_f32[j] = (عائم) src_بلاڪ_t[j];
            }
            cpyblck(reinterpret_cast<ثابت حرف *>(src_بلاڪ_f32), dst_بلاڪ);
        }
    });
    GGML_اڻ_استعمال_ٿيل(ne10);
    GGML_اڻ_استعمال_ٿيل(ne13);
    GGML_اڻ_استعمال_ٿيل(nb00);
    GGML_اڻ_استعمال_ٿيل(nb13);
}

سانچو<قسم_نالو بلاڪ_Type>
استعمال quantize_صف_qk_t = فراغ (*)(ثابت عائم *, بلاڪ_Type *, int64_t);

استعمال quantize_صفون_f_t = size_t (*)(ثابت عائم *, فراغ *, int64_t, int64_t, ثابت عائم *);

سانچو <قسم_نالو TIn, قسم_نالو TIdx, قسم_نالو بلاڪ_Type, صحيح qk, quantize_صف_qk_t<بلاڪ_Type> quantize_صف>
ساكن فراغ سيٽ_صفون_sycl_qk_host(
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

    ثابت size_t src0_بائيٽون = ggml_nbytes(src0);
    ثابت size_t src1_بائيٽون = ggml_nbytes(src1);

    std::vector<حرف> src0_host(src0_بائيٽون);
    std::vector<حرف> src1_host(src1_بائيٽون);

    stream->يادگيري_نقل(src0_host.معلومات(), src0->معلومات, src0_بائيٽون);
    stream->يادگيري_نقل(src1_host.معلومات(), src1->معلومات, src1_بائيٽون);
    stream->wait();

    std::vector<عائم> src_صف_f32(ne00);
    ثابت int64_t nblocks = ne00 / qk;
    std::vector<بلاڪ_Type> dst_صف_q(nblocks);

    لكل (int64_t i03 = 0; i03 < ne03; ++i03) {
        لكل (int64_t i02 = 0; i02 < ne02; ++i02) {
            لكل (int64_t i01 = 0; i01 < ne01; ++i01) {
                ثابت int64_t i12 = i03 % ne12;
                ثابت int64_t i11 = i02 % ne11;
                ثابت int64_t i10 = i01;

                ثابت size_t src1_وٿي = calculate_وٿي<3>({ nb10, nb11, nb12 }, { i10, i11, i12 });
                ثابت int64_t dst_صف = *(ثابت TIdx *) (src1_host.معلومات() + src1_وٿي);

                ثابت size_t src0_صف_وٿي = calculate_وٿي<3>({ nb01, nb02, nb03 }, { i01, i02, i03 });
                ثابت TIn * src_صف = reinterpret_cast<ثابت TIn *>(src0_host.معلومات() + src0_صف_وٿي);

                لكل (int64_t i00 = 0; i00 < ne00; ++i00) {
                    src_صف_f32[i00] = (عائم) src_صف[i00];
                }

                quantize_صف(src_صف_f32.معلومات(), dst_صف_q.معلومات(), ne00);

                ثابت size_t dst_وٿي = calculate_وٿي<3>({ nb1, nb2, nb3 }, { dst_صف, i02, i03 });
                stream->يادگيري_نقل((حرف *) dst->معلومات + dst_وٿي, dst_صف_q.معلومات(), nblocks * حجم(بلاڪ_Type));
                stream->wait();
            }
        }
    }
}

سانچو <قسم_نالو TIn, قسم_نالو TIdx, قسم_نالو بلاڪ_Type, صحيح qk, quantize_صفون_f_t quantize_صفون>
ساكن فراغ سيٽ_صفون_sycl_iq_host(
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

    ثابت size_t src0_بائيٽون = ggml_nbytes(src0);
    ثابت size_t src1_بائيٽون = ggml_nbytes(src1);

    std::vector<حرف> src0_host(src0_بائيٽون);
    std::vector<حرف> src1_host(src1_بائيٽون);

    stream->يادگيري_نقل(src0_host.معلومات(), src0->معلومات, src0_بائيٽون);
    stream->يادگيري_نقل(src1_host.معلومات(), src1->معلومات, src1_بائيٽون);
    stream->wait();

    std::vector<عائم> src_صف_f32(ne00);
    ثابت int64_t nblocks = ne00 / qk;
    std::vector<بلاڪ_Type> dst_صف_q(nblocks);

    لكل (int64_t i03 = 0; i03 < ne03; ++i03) {
        لكل (int64_t i02 = 0; i02 < ne02; ++i02) {
            لكل (int64_t i01 = 0; i01 < ne01; ++i01) {
                ثابت int64_t i12 = i03 % ne12;
                ثابت int64_t i11 = i02 % ne11;
                ثابت int64_t i10 = i01;

                ثابت size_t src1_وٿي = calculate_وٿي<3>({ nb10, nb11, nb12 }, { i10, i11, i12 });
                ثابت int64_t dst_صف = *(ثابت TIdx *) (src1_host.معلومات() + src1_وٿي);

                ثابت size_t src0_صف_وٿي = calculate_وٿي<3>({ nb01, nb02, nb03 }, { i01, i02, i03 });
                ثابت TIn * src_صف = reinterpret_cast<ثابت TIn *>(src0_host.معلومات() + src0_صف_وٿي);

                لكل (int64_t i00 = 0; i00 < ne00; ++i00) {
                    src_صف_f32[i00] = (عائم) src_صف[i00];
                }

                quantize_صفون(src_صف_f32.معلومات(), dst_صف_q.معلومات(), 1, ne00, خالي_اشارو);

                ثابت size_t dst_وٿي = calculate_وٿي<3>({ nb1, nb2, nb3 }, { dst_صف, i02, i03 });
                stream->يادگيري_نقل((حرف *) dst->معلومات + dst_وٿي, dst_صف_q.معلومات(), nblocks * حجم(بلاڪ_Type));
                stream->wait();
            }
        }
    }
}

سانچو<قسم_نالو TIn, قسم_نالو TIdx, قسم_نالو TOut>
ساكن فراغ k_سيٽ_صفون(
        ثابت حرف * __restrict__ src0, ثابت TIdx * __restrict__ src1, حرف * __restrict__ dst,
        ثابت int64_t ne00, ثابت int64_t ne01, ثابت int64_t ne02,
        ثابت int64_t ne11, ثابت int64_t ne12,
        ثابت size_t nb01, ثابت size_t nb02, ثابت size_t nb03,
        ثابت size_t nb10, ثابت size_t nb11, ثابت size_t nb12,
        ثابت size_t nb1, ثابت size_t nb2, ثابت size_t nb3,
        ثابت size_t src_type_سائيز, ثابت size_t dst_type_سائيز,
        ثابت int64_t ڪل_elements,
        ثابت sycl::nd_شي<1> & شي_ct1) {

    ثابت int64_t i = شي_ct1.get_global_لڪيري_id();
    إذا (i >= ڪل_elements) {
        أرجع;
    }

    ثابت int64_t i03 = i / (ne00 * ne01 * ne02);
    ثابت int64_t i02 = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    ثابت int64_t i01 = (i - i03 * ne00 * ne01 * ne02 - i02 * ne00 * ne01) / ne00;
    ثابت int64_t i00 = i - i03 * ne00 * ne01 * ne02 - i02 * ne00 * ne01 - i01 * ne00;

    ثابت int64_t i12 = i03 % ne12;
    ثابت int64_t i11 = i02 % ne11;
    ثابت int64_t i10 = i01;

    ثابت int64_t dst_صف = *(ثابت TIdx *)((ثابت حرف *)src1 + calculate_وٿي<3>({nb10, nb11, nb12}, {i10, i11, i12}));

    ثابت حرف * src0_صف = src0 + calculate_وٿي<3>({nb01, nb02, nb03}, {i01, i02, i03});
    ثابت حرف * src_elem = src0_صف + i00 * src_type_سائيز;
    حرف * dst_صف_ptr = dst + dst_صف*nb1 + i02*nb2 + i03*nb3;
    حرف * dst_elem = dst_صف_ptr + i00 * dst_type_سائيز;

    convert<TIn, TOut>(src_elem, dst_elem);
}

سانچو<قسم_نالو TIn, قسم_نالو TIdx, قسم_نالو TOut>
ساكن فراغ سيٽ_صفون_sycl(
        ثابت حرف * src0_d, ثابت TIdx * src1_d, حرف * dst_d,
        ثابت int64_t ne00, ثابت int64_t ne01, ثابت int64_t ne02, ثابت int64_t ne03,
        ثابت int64_t ne11, ثابت int64_t ne12, ثابت size_t nb01, ثابت size_t nb02, ثابت size_t nb03,
        ثابت size_t nb10, ثابت size_t nb11, ثابت size_t nb12,
        ثابت size_t nb1, ثابت size_t nb2, ثابت size_t nb3,
        ثابت size_t src_type_سائيز, ثابت size_t dst_type_سائيز,
        قطار_ptr stream) {

    ثابت int64_t ڪل_elements = ne00 * ne01 * ne02 * ne03;

    constexpr صحيح بلاڪ_سائيز_3 = 64;
    ثابت int64_t grid_سائيز = ceil_div(ڪل_elements, بلاڪ_سائيز_3);

    stream->parallel_for(
        sycl::nd_range<1>(grid_سائيز * بلاڪ_سائيز_3, بلاڪ_سائيز_3),
        [=](sycl::nd_شي<1> شي_ct1) [[sycl::reqd_sub_گروهه_سائيز(WARP_سائيز)]] {
            k_سيٽ_صفون<TIn, TIdx, TOut>(
                src0_d, src1_d, dst_d,
                ne00, ne01, ne02,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                src_type_سائيز, dst_type_سائيز,
                ڪل_elements,
                شي_ct1
            );
        }
    );
}

سانچو<قسم_نالو TIn, قسم_نالو TIdx>
ساكن فراغ سيٽ_صفون_sycl(ggml_backend_sycl_context & ctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst) {
    ثابت حرف * src0_d = (ثابت حرف *)src0->معلومات;
    ثابت TIdx * src1_d = (ثابت TIdx *)src1->معلومات;

    GGML_TENSOR_ٻٽو_OP_LOCALS

    dpct::قطار_ptr stream = ctx.stream();
    بدل (dst->type) {
        حالة GGML_TYPE_F32:
            سيٽ_صفون_sycl<TIn, TIdx, عائم>(
                src0_d, src1_d, (حرف *)dst->معلومات,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                حجم(TIn), حجم(عائم),
                stream
            );
            اكسر;
        حالة GGML_TYPE_F16:
            dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
            سيٽ_صفون_sycl<TIn, TIdx, sycl::half>(
                src0_d, src1_d, (حرف *)dst->معلومات,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                حجم(TIn), حجم(sycl::half),
                stream
            );
            اكسر;
#جيڪڏهن_وصف_ٿيل GGML_SYCL_HAS_BF16
        حالة GGML_TYPE_BF16:
            سيٽ_صفون_sycl<TIn, TIdx, sycl::ext::oneapi::bfloat16>(
                src0_d, src1_d, (حرف *)dst->معلومات,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                حجم(TIn), حجم(sycl::ext::oneapi::bfloat16),
                stream
            );
            اكسر;
#جيڪڏهن_پورو
        حالة GGML_TYPE_Q8_0:
            سيٽ_صفون_sycl_q<TIn, TIdx, بلاڪ_q8_0, QK8_0, cpy_blck_f32_q8_0>(
                src0_d, src1_d, (بلاڪ_q8_0 *) dst->معلومات, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            اكسر;
        حالة GGML_TYPE_Q1_0:
            سيٽ_صفون_sycl_q<TIn, TIdx, بلاڪ_q1_0, QK1_0, cpy_blck_f32_q1_0>(
                src0_d, src1_d, (بلاڪ_q1_0 *) dst->معلومات, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            اكسر;
        حالة GGML_TYPE_Q2_0:
            سيٽ_صفون_sycl_q<TIn, TIdx, بلاڪ_q2_0, QK2_0, cpy_blck_f32_q2_0>(
                src0_d, src1_d, (بلاڪ_q2_0 *) dst->معلومات, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            اكسر;
        حالة GGML_TYPE_Q5_1:
            سيٽ_صفون_sycl_q<TIn, TIdx, بلاڪ_q5_1, QK5_1, cpy_blck_f32_q5_1>(
                src0_d, src1_d, (بلاڪ_q5_1 *) dst->معلومات, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            اكسر;
        حالة GGML_TYPE_Q5_0:
            سيٽ_صفون_sycl_q<TIn, TIdx, بلاڪ_q5_0, QK5_0, cpy_blck_f32_q5_0>(
                src0_d, src1_d, (بلاڪ_q5_0 *) dst->معلومات, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            اكسر;
        حالة GGML_TYPE_Q4_1:
            سيٽ_صفون_sycl_q<TIn, TIdx, بلاڪ_q4_1, QK4_1, cpy_blck_f32_q4_1>(
                src0_d, src1_d, (بلاڪ_q4_1 *) dst->معلومات, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            اكسر;
        حالة GGML_TYPE_Q4_0:
            سيٽ_صفون_sycl_q<TIn, TIdx, بلاڪ_q4_0, QK4_0, cpy_blck_f32_q4_0>(
                src0_d, src1_d, (بلاڪ_q4_0 *) dst->معلومات, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            اكسر;
        حالة GGML_TYPE_IQ4_NL:
            سيٽ_صفون_sycl_q<TIn, TIdx, بلاڪ_iq4_nl, QK4_NL, cpy_blck_f32_iq4_nl>(
                src0_d, src1_d, (بلاڪ_iq4_nl *) dst->معلومات, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            اكسر;
        حالة GGML_TYPE_MXFP4:
            سيٽ_صفون_sycl_q<TIn, TIdx, بلاڪ_mxfp4, QK_MXFP4, cpy_blck_f32_mxfp4>(
                src0_d, src1_d, (بلاڪ_mxfp4 *) dst->معلومات, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            اكسر;
        حالة GGML_TYPE_NVFP4:
            سيٽ_صفون_sycl_q<TIn, TIdx, بلاڪ_nvfp4, QK_NVFP4, cpy_blck_f32_nvfp4>(
                src0_d, src1_d, (بلاڪ_nvfp4 *) dst->معلومات, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            اكسر;
        حالة GGML_TYPE_Q2_K:
            سيٽ_صفون_sycl_qk_host<TIn, TIdx, بلاڪ_q2_K, QK_K, quantize_صف_q2_K_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            اكسر;
        حالة GGML_TYPE_Q3_K:
            سيٽ_صفون_sycl_qk_host<TIn, TIdx, بلاڪ_q3_K, QK_K, quantize_صف_q3_K_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            اكسر;
        حالة GGML_TYPE_Q4_K:
            سيٽ_صفون_sycl_qk_host<TIn, TIdx, بلاڪ_q4_K, QK_K, quantize_صف_q4_K_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            اكسر;
        حالة GGML_TYPE_Q5_K:
            سيٽ_صفون_sycl_qk_host<TIn, TIdx, بلاڪ_q5_K, QK_K, quantize_صف_q5_K_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            اكسر;
        حالة GGML_TYPE_Q6_K:
            سيٽ_صفون_sycl_qk_host<TIn, TIdx, بلاڪ_q6_K, QK_K, quantize_صف_q6_K_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            اكسر;
        حالة GGML_TYPE_IQ2_XXS:
            سيٽ_صفون_sycl_iq_host<TIn, TIdx, بلاڪ_iq2_xxs, QK_K, quantize_iq2_xxs>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            اكسر;
        حالة GGML_TYPE_IQ2_XS:
            سيٽ_صفون_sycl_iq_host<TIn, TIdx, بلاڪ_iq2_xs, QK_K, quantize_iq2_xs>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            اكسر;
        حالة GGML_TYPE_IQ2_S:
            سيٽ_صفون_sycl_iq_host<TIn, TIdx, بلاڪ_iq2_s, QK_K, quantize_iq2_s>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            اكسر;
        حالة GGML_TYPE_IQ3_XXS:
            سيٽ_صفون_sycl_qk_host<TIn, TIdx, بلاڪ_iq3_xxs, QK_K, quantize_صف_iq3_xxs_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            اكسر;
        حالة GGML_TYPE_IQ3_S:
            سيٽ_صفون_sycl_qk_host<TIn, TIdx, بلاڪ_iq3_s, QK_K, quantize_صف_iq3_s_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            اكسر;
        حالة GGML_TYPE_IQ1_S:
            سيٽ_صفون_sycl_iq_host<TIn, TIdx, بلاڪ_iq1_s, QK_K, quantize_iq1_s>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            اكسر;
        حالة GGML_TYPE_IQ1_M:
            سيٽ_صفون_sycl_iq_host<TIn, TIdx, بلاڪ_iq1_m, QK_K, quantize_iq1_m>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            اكسر;
        حالة GGML_TYPE_IQ4_XS:
            سيٽ_صفون_sycl_qk_host<TIn, TIdx, بلاڪ_iq4_xs, QK_K, quantize_صف_iq4_xs_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            اكسر;
        افتراضي:
            GGML_ABORT("Unsupported tensor type: src0 %s src1 %s dst %s", ggml_type_name(dst->src[0]->type),
                ggml_type_name(dst->src[1]->type), ggml_type_name(dst->type));
            اكسر;
    }
}

فراغ ggml_sycl_op_سيٽ_صفون(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->src[1]->type == GGML_TYPE_I64 || dst->src[1]->type == GGML_TYPE_I32);

    // dispatch on the index type (src1) and the source value type (src0)
    إذا (src0->type == GGML_TYPE_F16) {
        إذا (src1->type == GGML_TYPE_I64) {
            سيٽ_صفون_sycl<sycl::half, int64_t>(ctx, src0, src1, dst);
        } وإلا {
            سيٽ_صفون_sycl<sycl::half, int32_t>(ctx, src0, src1, dst);
        }
    } وإلا {
        إذا (src1->type == GGML_TYPE_I64) {
            سيٽ_صفون_sycl<عائم, int64_t>(ctx, src0, src1, dst);
        } وإلا {
            سيٽ_صفون_sycl<عائم, int32_t>(ctx, src0, src1, dst);
        }
    }
}
