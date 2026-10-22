#ضمن "set_rows.hpp"
#ضمن "cpy.hpp"

#ضمن "ggml-quants.h"

#ضمن <vector>

نطاق utils {
کينډۍ<ډول_نوم T>
ساكن constexpr منطقي is_arithmetic_v() {
    أرجع std::is_arithmetic_v<T> || std::is_same_v<T, sycl::half>
#که_تعریف_شوی GGML_SYCL_HAS_BF16
        || std::is_same_v<T, sycl::ext::oneapi::bfloat16>
#د_که_پای
        ;
}
}

کينډۍ<ډول_نوم TIn, ډول_نوم TOut>
ساكن مضمن std::enable_if_t<utils::is_arithmetic_v<TIn>() && utils::is_arithmetic_v<TOut>(), فراغ>
convert (ثابت حرف* src, حرف* dst) {
    خپلکاری src_val = *reinterpret_cast<ثابت TIn*>(src);
    خپلکاری dst_val = sycl::vec<TIn, 1>(src_val).کينډۍ convert<TOut, sycl::rounding_ډېرپېښ::automatic>()[0];
   *reinterpret_cast<TOut*>(dst) = dst_val;
}

#که_تعریف_شوی GGML_SYCL_HAS_BF16
// sycl::vec::convert does not provide a half -> bfloat16 path, so route through float.
کينډۍ<>
مضمن فراغ convert<sycl::half, sycl::ext::oneapi::bfloat16>(ثابت حرف* src, حرف* dst) {
    ثابت عائم tmp = sycl::vec<sycl::half, 1>(*reinterpret_cast<ثابت sycl::half*>(src))
                          .کينډۍ convert<عائم, sycl::rounding_ډېرپېښ::automatic>()[0];
    *reinterpret_cast<sycl::ext::oneapi::bfloat16*>(dst) = sycl::ext::oneapi::bfloat16(tmp);
}
#د_که_پای

کينډۍ <ډول_نوم TIn, ډول_نوم TIdx, ډول_نوم بلاک_Type, صحيح qk, cpy_kernel_t cpyblck>
ساكن فراغ سېټ_لیکې_sycl_q(ثابت حرف * __restrict__ src0_d,
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
                            کتار_ptr     stream) {
    ثابت int64_t ټول_بلاکونه = (ne00 * ne01 * ne02 * ne03) / qk;
    constexpr صحيح بلاک_اندازه_3   = 256;
    ثابت int64_t grid_اندازه    = ceil_div(ټول_بلاکونه, بلاک_اندازه_3);

    stream->parallel_for(sycl::nd_range<1>(grid_اندازه * بلاک_اندازه_3, بلاک_اندازه_3), [=](sycl::nd_توکی<1> توکی_ct1) {
        ثابت int64_t i = توکی_ct1.get_global_خطي_id();
        إذا (i >= ټول_بلاکونه) {
            أرجع;
        }
        ثابت int64_t i_بنسټ      = i * qk;
        ثابت int64_t i03         = i_بنسټ / (ne00 * ne01 * ne02);
        ثابت int64_t rem1        = i_بنسټ - i03 * (ne00 * ne01 * ne02);
        ثابت int64_t i02         = rem1 / (ne00 * ne01);
        ثابت int64_t rem2        = rem1 - i02 * ne00 * ne01;
        ثابت int64_t i01         = rem2 / ne00;
        ثابت int64_t i00         = rem2 - i01 * ne00;
        ثابت int64_t i12         = i03 % ne12;
        ثابت int64_t i11         = i02 % ne11;
        ثابت int64_t i10         = i01;
        ثابت size_t  src_دننه_واټن  = calculate_دننه_واټن<3>({ nb01, nb02, nb03 }, { i01, i02, i03 });
        ثابت حرف *  src_بلاک   = src0_d + src_دننه_واټن + i00 * حجم(TIn);
        ثابت size_t  src1_دننه_واټن = calculate_دننه_واټن<3>({ nb10, nb11, nb12 }, { i10, i11, i12 });
        ثابت int64_t dst_لیکه     = src1_d[src1_دننه_واټن / حجم(TIdx)];
        ثابت size_t  dst_دننه_واټن =
            calculate_دننه_واټن<3>({ nb1, nb2, nb3 }, { dst_لیکه, i02, i03 }) + (i00 / qk) * حجم(بلاک_Type);
        حرف * dst_بلاک = reinterpret_cast<حرف *>(reinterpret_cast<حرف *>(dst_d) + dst_دننه_واټن);
        إذا constexpr (std::is_same_v<TIn, عائم>) {
            cpyblck(src_بلاک, dst_بلاک);
        } وإلا {
            عائم src_بلاک_f32[qk];
            ثابت TIn * src_بلاک_t = reinterpret_cast<ثابت TIn *>(src_بلاک);
            لكل (صحيح j = 0; j < qk; ++j) {
                src_بلاک_f32[j] = (عائم) src_بلاک_t[j];
            }
            cpyblck(reinterpret_cast<ثابت حرف *>(src_بلاک_f32), dst_بلاک);
        }
    });
    GGML_ناکارول_شوی(ne10);
    GGML_ناکارول_شوی(ne13);
    GGML_ناکارول_شوی(nb00);
    GGML_ناکارول_شوی(nb13);
}

کينډۍ<ډول_نوم بلاک_Type>
کارول quantize_لیکه_qk_t = فراغ (*)(ثابت عائم *, بلاک_Type *, int64_t);

کارول quantize_لیکې_f_t = size_t (*)(ثابت عائم *, فراغ *, int64_t, int64_t, ثابت عائم *);

کينډۍ <ډول_نوم TIn, ډول_نوم TIdx, ډول_نوم بلاک_Type, صحيح qk, quantize_لیکه_qk_t<بلاک_Type> quantize_لیکه>
ساكن فراغ سېټ_لیکې_sycl_qk_host(
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
        کتار_ptr stream) {
    GGML_ASSERT(ne00 % qk == 0);

    ثابت size_t src0_بایټونه = ggml_nbytes(src0);
    ثابت size_t src1_بایټونه = ggml_nbytes(src1);

    std::vector<حرف> src0_host(src0_بایټونه);
    std::vector<حرف> src1_host(src1_بایټونه);

    stream->حافظه_نقل(src0_host.معلومات(), src0->معلومات, src0_بایټونه);
    stream->حافظه_نقل(src1_host.معلومات(), src1->معلومات, src1_بایټونه);
    stream->wait();

    std::vector<عائم> src_لیکه_f32(ne00);
    ثابت int64_t nblocks = ne00 / qk;
    std::vector<بلاک_Type> dst_لیکه_q(nblocks);

    لكل (int64_t i03 = 0; i03 < ne03; ++i03) {
        لكل (int64_t i02 = 0; i02 < ne02; ++i02) {
            لكل (int64_t i01 = 0; i01 < ne01; ++i01) {
                ثابت int64_t i12 = i03 % ne12;
                ثابت int64_t i11 = i02 % ne11;
                ثابت int64_t i10 = i01;

                ثابت size_t src1_دننه_واټن = calculate_دننه_واټن<3>({ nb10, nb11, nb12 }, { i10, i11, i12 });
                ثابت int64_t dst_لیکه = *(ثابت TIdx *) (src1_host.معلومات() + src1_دننه_واټن);

                ثابت size_t src0_لیکه_دننه_واټن = calculate_دننه_واټن<3>({ nb01, nb02, nb03 }, { i01, i02, i03 });
                ثابت TIn * src_لیکه = reinterpret_cast<ثابت TIn *>(src0_host.معلومات() + src0_لیکه_دننه_واټن);

                لكل (int64_t i00 = 0; i00 < ne00; ++i00) {
                    src_لیکه_f32[i00] = (عائم) src_لیکه[i00];
                }

                quantize_لیکه(src_لیکه_f32.معلومات(), dst_لیکه_q.معلومات(), ne00);

                ثابت size_t dst_دننه_واټن = calculate_دننه_واټن<3>({ nb1, nb2, nb3 }, { dst_لیکه, i02, i03 });
                stream->حافظه_نقل((حرف *) dst->معلومات + dst_دننه_واټن, dst_لیکه_q.معلومات(), nblocks * حجم(بلاک_Type));
                stream->wait();
            }
        }
    }
}

کينډۍ <ډول_نوم TIn, ډول_نوم TIdx, ډول_نوم بلاک_Type, صحيح qk, quantize_لیکې_f_t quantize_لیکې>
ساكن فراغ سېټ_لیکې_sycl_iq_host(
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
        کتار_ptr stream) {
    GGML_ASSERT(ne00 % qk == 0);

    ثابت size_t src0_بایټونه = ggml_nbytes(src0);
    ثابت size_t src1_بایټونه = ggml_nbytes(src1);

    std::vector<حرف> src0_host(src0_بایټونه);
    std::vector<حرف> src1_host(src1_بایټونه);

    stream->حافظه_نقل(src0_host.معلومات(), src0->معلومات, src0_بایټونه);
    stream->حافظه_نقل(src1_host.معلومات(), src1->معلومات, src1_بایټونه);
    stream->wait();

    std::vector<عائم> src_لیکه_f32(ne00);
    ثابت int64_t nblocks = ne00 / qk;
    std::vector<بلاک_Type> dst_لیکه_q(nblocks);

    لكل (int64_t i03 = 0; i03 < ne03; ++i03) {
        لكل (int64_t i02 = 0; i02 < ne02; ++i02) {
            لكل (int64_t i01 = 0; i01 < ne01; ++i01) {
                ثابت int64_t i12 = i03 % ne12;
                ثابت int64_t i11 = i02 % ne11;
                ثابت int64_t i10 = i01;

                ثابت size_t src1_دننه_واټن = calculate_دننه_واټن<3>({ nb10, nb11, nb12 }, { i10, i11, i12 });
                ثابت int64_t dst_لیکه = *(ثابت TIdx *) (src1_host.معلومات() + src1_دننه_واټن);

                ثابت size_t src0_لیکه_دننه_واټن = calculate_دننه_واټن<3>({ nb01, nb02, nb03 }, { i01, i02, i03 });
                ثابت TIn * src_لیکه = reinterpret_cast<ثابت TIn *>(src0_host.معلومات() + src0_لیکه_دننه_واټن);

                لكل (int64_t i00 = 0; i00 < ne00; ++i00) {
                    src_لیکه_f32[i00] = (عائم) src_لیکه[i00];
                }

                quantize_لیکې(src_لیکه_f32.معلومات(), dst_لیکه_q.معلومات(), 1, ne00, تش_اشاره);

                ثابت size_t dst_دننه_واټن = calculate_دننه_واټن<3>({ nb1, nb2, nb3 }, { dst_لیکه, i02, i03 });
                stream->حافظه_نقل((حرف *) dst->معلومات + dst_دننه_واټن, dst_لیکه_q.معلومات(), nblocks * حجم(بلاک_Type));
                stream->wait();
            }
        }
    }
}

کينډۍ<ډول_نوم TIn, ډول_نوم TIdx, ډول_نوم TOut>
ساكن فراغ k_سېټ_لیکې(
        ثابت حرف * __restrict__ src0, ثابت TIdx * __restrict__ src1, حرف * __restrict__ dst,
        ثابت int64_t ne00, ثابت int64_t ne01, ثابت int64_t ne02,
        ثابت int64_t ne11, ثابت int64_t ne12,
        ثابت size_t nb01, ثابت size_t nb02, ثابت size_t nb03,
        ثابت size_t nb10, ثابت size_t nb11, ثابت size_t nb12,
        ثابت size_t nb1, ثابت size_t nb2, ثابت size_t nb3,
        ثابت size_t src_type_اندازه, ثابت size_t dst_type_اندازه,
        ثابت int64_t ټول_elements,
        ثابت sycl::nd_توکی<1> & توکی_ct1) {

    ثابت int64_t i = توکی_ct1.get_global_خطي_id();
    إذا (i >= ټول_elements) {
        أرجع;
    }

    ثابت int64_t i03 = i / (ne00 * ne01 * ne02);
    ثابت int64_t i02 = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    ثابت int64_t i01 = (i - i03 * ne00 * ne01 * ne02 - i02 * ne00 * ne01) / ne00;
    ثابت int64_t i00 = i - i03 * ne00 * ne01 * ne02 - i02 * ne00 * ne01 - i01 * ne00;

    ثابت int64_t i12 = i03 % ne12;
    ثابت int64_t i11 = i02 % ne11;
    ثابت int64_t i10 = i01;

    ثابت int64_t dst_لیکه = *(ثابت TIdx *)((ثابت حرف *)src1 + calculate_دننه_واټن<3>({nb10, nb11, nb12}, {i10, i11, i12}));

    ثابت حرف * src0_لیکه = src0 + calculate_دننه_واټن<3>({nb01, nb02, nb03}, {i01, i02, i03});
    ثابت حرف * src_elem = src0_لیکه + i00 * src_type_اندازه;
    حرف * dst_لیکه_ptr = dst + dst_لیکه*nb1 + i02*nb2 + i03*nb3;
    حرف * dst_elem = dst_لیکه_ptr + i00 * dst_type_اندازه;

    convert<TIn, TOut>(src_elem, dst_elem);
}

کينډۍ<ډول_نوم TIn, ډول_نوم TIdx, ډول_نوم TOut>
ساكن فراغ سېټ_لیکې_sycl(
        ثابت حرف * src0_d, ثابت TIdx * src1_d, حرف * dst_d,
        ثابت int64_t ne00, ثابت int64_t ne01, ثابت int64_t ne02, ثابت int64_t ne03,
        ثابت int64_t ne11, ثابت int64_t ne12, ثابت size_t nb01, ثابت size_t nb02, ثابت size_t nb03,
        ثابت size_t nb10, ثابت size_t nb11, ثابت size_t nb12,
        ثابت size_t nb1, ثابت size_t nb2, ثابت size_t nb3,
        ثابت size_t src_type_اندازه, ثابت size_t dst_type_اندازه,
        کتار_ptr stream) {

    ثابت int64_t ټول_elements = ne00 * ne01 * ne02 * ne03;

    constexpr صحيح بلاک_اندازه_3 = 64;
    ثابت int64_t grid_اندازه = ceil_div(ټول_elements, بلاک_اندازه_3);

    stream->parallel_for(
        sycl::nd_range<1>(grid_اندازه * بلاک_اندازه_3, بلاک_اندازه_3),
        [=](sycl::nd_توکی<1> توکی_ct1) [[sycl::reqd_sub_ډله_اندازه(WARP_اندازه)]] {
            k_سېټ_لیکې<TIn, TIdx, TOut>(
                src0_d, src1_d, dst_d,
                ne00, ne01, ne02,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                src_type_اندازه, dst_type_اندازه,
                ټول_elements,
                توکی_ct1
            );
        }
    );
}

کينډۍ<ډول_نوم TIn, ډول_نوم TIdx>
ساكن فراغ سېټ_لیکې_sycl(ggml_backend_sycl_context & ctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst) {
    ثابت حرف * src0_d = (ثابت حرف *)src0->معلومات;
    ثابت TIdx * src1_d = (ثابت TIdx *)src1->معلومات;

    GGML_TENSOR_دوګونی_OP_LOCALS

    dpct::کتار_ptr stream = ctx.stream();
    بدل (dst->type) {
        حالة GGML_TYPE_F32:
            سېټ_لیکې_sycl<TIn, TIdx, عائم>(
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
            سېټ_لیکې_sycl<TIn, TIdx, sycl::half>(
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
#که_تعریف_شوی GGML_SYCL_HAS_BF16
        حالة GGML_TYPE_BF16:
            سېټ_لیکې_sycl<TIn, TIdx, sycl::ext::oneapi::bfloat16>(
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
#د_که_پای
        حالة GGML_TYPE_Q8_0:
            سېټ_لیکې_sycl_q<TIn, TIdx, بلاک_q8_0, QK8_0, cpy_blck_f32_q8_0>(
                src0_d, src1_d, (بلاک_q8_0 *) dst->معلومات, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            اكسر;
        حالة GGML_TYPE_Q1_0:
            سېټ_لیکې_sycl_q<TIn, TIdx, بلاک_q1_0, QK1_0, cpy_blck_f32_q1_0>(
                src0_d, src1_d, (بلاک_q1_0 *) dst->معلومات, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            اكسر;
        حالة GGML_TYPE_Q2_0:
            سېټ_لیکې_sycl_q<TIn, TIdx, بلاک_q2_0, QK2_0, cpy_blck_f32_q2_0>(
                src0_d, src1_d, (بلاک_q2_0 *) dst->معلومات, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            اكسر;
        حالة GGML_TYPE_Q5_1:
            سېټ_لیکې_sycl_q<TIn, TIdx, بلاک_q5_1, QK5_1, cpy_blck_f32_q5_1>(
                src0_d, src1_d, (بلاک_q5_1 *) dst->معلومات, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            اكسر;
        حالة GGML_TYPE_Q5_0:
            سېټ_لیکې_sycl_q<TIn, TIdx, بلاک_q5_0, QK5_0, cpy_blck_f32_q5_0>(
                src0_d, src1_d, (بلاک_q5_0 *) dst->معلومات, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            اكسر;
        حالة GGML_TYPE_Q4_1:
            سېټ_لیکې_sycl_q<TIn, TIdx, بلاک_q4_1, QK4_1, cpy_blck_f32_q4_1>(
                src0_d, src1_d, (بلاک_q4_1 *) dst->معلومات, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            اكسر;
        حالة GGML_TYPE_Q4_0:
            سېټ_لیکې_sycl_q<TIn, TIdx, بلاک_q4_0, QK4_0, cpy_blck_f32_q4_0>(
                src0_d, src1_d, (بلاک_q4_0 *) dst->معلومات, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            اكسر;
        حالة GGML_TYPE_IQ4_NL:
            سېټ_لیکې_sycl_q<TIn, TIdx, بلاک_iq4_nl, QK4_NL, cpy_blck_f32_iq4_nl>(
                src0_d, src1_d, (بلاک_iq4_nl *) dst->معلومات, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            اكسر;
        حالة GGML_TYPE_MXFP4:
            سېټ_لیکې_sycl_q<TIn, TIdx, بلاک_mxfp4, QK_MXFP4, cpy_blck_f32_mxfp4>(
                src0_d, src1_d, (بلاک_mxfp4 *) dst->معلومات, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            اكسر;
        حالة GGML_TYPE_NVFP4:
            سېټ_لیکې_sycl_q<TIn, TIdx, بلاک_nvfp4, QK_NVFP4, cpy_blck_f32_nvfp4>(
                src0_d, src1_d, (بلاک_nvfp4 *) dst->معلومات, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            اكسر;
        حالة GGML_TYPE_Q2_K:
            سېټ_لیکې_sycl_qk_host<TIn, TIdx, بلاک_q2_K, QK_K, quantize_لیکه_q2_K_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            اكسر;
        حالة GGML_TYPE_Q3_K:
            سېټ_لیکې_sycl_qk_host<TIn, TIdx, بلاک_q3_K, QK_K, quantize_لیکه_q3_K_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            اكسر;
        حالة GGML_TYPE_Q4_K:
            سېټ_لیکې_sycl_qk_host<TIn, TIdx, بلاک_q4_K, QK_K, quantize_لیکه_q4_K_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            اكسر;
        حالة GGML_TYPE_Q5_K:
            سېټ_لیکې_sycl_qk_host<TIn, TIdx, بلاک_q5_K, QK_K, quantize_لیکه_q5_K_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            اكسر;
        حالة GGML_TYPE_Q6_K:
            سېټ_لیکې_sycl_qk_host<TIn, TIdx, بلاک_q6_K, QK_K, quantize_لیکه_q6_K_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            اكسر;
        حالة GGML_TYPE_IQ2_XXS:
            سېټ_لیکې_sycl_iq_host<TIn, TIdx, بلاک_iq2_xxs, QK_K, quantize_iq2_xxs>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            اكسر;
        حالة GGML_TYPE_IQ2_XS:
            سېټ_لیکې_sycl_iq_host<TIn, TIdx, بلاک_iq2_xs, QK_K, quantize_iq2_xs>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            اكسر;
        حالة GGML_TYPE_IQ2_S:
            سېټ_لیکې_sycl_iq_host<TIn, TIdx, بلاک_iq2_s, QK_K, quantize_iq2_s>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            اكسر;
        حالة GGML_TYPE_IQ3_XXS:
            سېټ_لیکې_sycl_qk_host<TIn, TIdx, بلاک_iq3_xxs, QK_K, quantize_لیکه_iq3_xxs_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            اكسر;
        حالة GGML_TYPE_IQ3_S:
            سېټ_لیکې_sycl_qk_host<TIn, TIdx, بلاک_iq3_s, QK_K, quantize_لیکه_iq3_s_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            اكسر;
        حالة GGML_TYPE_IQ1_S:
            سېټ_لیکې_sycl_iq_host<TIn, TIdx, بلاک_iq1_s, QK_K, quantize_iq1_s>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            اكسر;
        حالة GGML_TYPE_IQ1_M:
            سېټ_لیکې_sycl_iq_host<TIn, TIdx, بلاک_iq1_m, QK_K, quantize_iq1_m>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            اكسر;
        حالة GGML_TYPE_IQ4_XS:
            سېټ_لیکې_sycl_qk_host<TIn, TIdx, بلاک_iq4_xs, QK_K, quantize_لیکه_iq4_xs_ref>(
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

فراغ ggml_sycl_op_سېټ_لیکې(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->src[1]->type == GGML_TYPE_I64 || dst->src[1]->type == GGML_TYPE_I32);

    // dispatch on the index type (src1) and the source value type (src0)
    إذا (src0->type == GGML_TYPE_F16) {
        إذا (src1->type == GGML_TYPE_I64) {
            سېټ_لیکې_sycl<sycl::half, int64_t>(ctx, src0, src1, dst);
        } وإلا {
            سېټ_لیکې_sycl<sycl::half, int32_t>(ctx, src0, src1, dst);
        }
    } وإلا {
        إذا (src1->type == GGML_TYPE_I64) {
            سېټ_لیکې_sycl<عائم, int64_t>(ctx, src0, src1, dst);
        } وإلا {
            سېټ_لیکې_sycl<عائم, int32_t>(ctx, src0, src1, dst);
        }
    }
}
