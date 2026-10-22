#ضمن "set_rows.hpp"
#ضمن "cpy.hpp"

#ضمن "ggml-quants.h"

#ضمن <vector>

نطاق utils {
قالب<نام_نوع T>
ساكن constexpr منطقي is_arithmetic_v() {
    أرجع std::is_arithmetic_v<T> || std::is_same_v<T, sycl::half>
#اگر_تعریف_شده GGML_SYCL_HAS_BF16
        || std::is_same_v<T, sycl::ext::oneapi::bfloat16>
#پایان_اگر
        ;
}
}

قالب<نام_نوع TIn, نام_نوع TOut>
ساكن مضمن std::enable_if_t<utils::is_arithmetic_v<TIn>() && utils::is_arithmetic_v<TOut>(), فراغ>
convert (ثابت حرف* src, حرف* dst) {
    خودکار src_val = *reinterpret_cast<ثابت TIn*>(src);
    خودکار dst_val = sycl::vec<TIn, 1>(src_val).قالب convert<TOut, sycl::rounding_نما::automatic>()[0];
   *reinterpret_cast<TOut*>(dst) = dst_val;
}

#اگر_تعریف_شده GGML_SYCL_HAS_BF16
// sycl::vec::convert does not provide a half -> bfloat16 path, so route through float.
قالب<>
مضمن فراغ convert<sycl::half, sycl::ext::oneapi::bfloat16>(ثابت حرف* src, حرف* dst) {
    ثابت عائم tmp = sycl::vec<sycl::half, 1>(*reinterpret_cast<ثابت sycl::half*>(src))
                          .قالب convert<عائم, sycl::rounding_نما::automatic>()[0];
    *reinterpret_cast<sycl::ext::oneapi::bfloat16*>(dst) = sycl::ext::oneapi::bfloat16(tmp);
}
#پایان_اگر

قالب <نام_نوع TIn, نام_نوع TIdx, نام_نوع بلوک_Type, صحيح qk, cpy_kernel_t cpyblck>
ساكن فراغ مجموعه_ردیفها_sycl_q(ثابت حرف * __restrict__ src0_d,
                            ثابت TIdx * __restrict__ src1_d,
                            بلوک_Type * __restrict__ dst_d,
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
                            صف_ptr     stream) {
    ثابت int64_t مجموع_بلوکها = (ne00 * ne01 * ne02 * ne03) / qk;
    constexpr صحيح بلوک_اندازه_3   = 256;
    ثابت int64_t grid_اندازه    = ceil_div(مجموع_بلوکها, بلوک_اندازه_3);

    stream->parallel_for(sycl::nd_range<1>(grid_اندازه * بلوک_اندازه_3, بلوک_اندازه_3), [=](sycl::nd_قلم<1> قلم_ct1) {
        ثابت int64_t i = قلم_ct1.get_global_خطی_id();
        إذا (i >= مجموع_بلوکها) {
            أرجع;
        }
        ثابت int64_t i_پایه      = i * qk;
        ثابت int64_t i03         = i_پایه / (ne00 * ne01 * ne02);
        ثابت int64_t rem1        = i_پایه - i03 * (ne00 * ne01 * ne02);
        ثابت int64_t i02         = rem1 / (ne00 * ne01);
        ثابت int64_t rem2        = rem1 - i02 * ne00 * ne01;
        ثابت int64_t i01         = rem2 / ne00;
        ثابت int64_t i00         = rem2 - i01 * ne00;
        ثابت int64_t i12         = i03 % ne12;
        ثابت int64_t i11         = i02 % ne11;
        ثابت int64_t i10         = i01;
        ثابت size_t  src_جابه_جایی  = calculate_جابه_جایی<3>({ nb01, nb02, nb03 }, { i01, i02, i03 });
        ثابت حرف *  src_بلوک   = src0_d + src_جابه_جایی + i00 * حجم(TIn);
        ثابت size_t  src1_جابه_جایی = calculate_جابه_جایی<3>({ nb10, nb11, nb12 }, { i10, i11, i12 });
        ثابت int64_t dst_ردیف     = src1_d[src1_جابه_جایی / حجم(TIdx)];
        ثابت size_t  dst_جابه_جایی =
            calculate_جابه_جایی<3>({ nb1, nb2, nb3 }, { dst_ردیف, i02, i03 }) + (i00 / qk) * حجم(بلوک_Type);
        حرف * dst_بلوک = reinterpret_cast<حرف *>(reinterpret_cast<حرف *>(dst_d) + dst_جابه_جایی);
        إذا constexpr (std::is_same_v<TIn, عائم>) {
            cpyblck(src_بلوک, dst_بلوک);
        } وإلا {
            عائم src_بلوک_f32[qk];
            ثابت TIn * src_بلوک_t = reinterpret_cast<ثابت TIn *>(src_بلوک);
            لكل (صحيح j = 0; j < qk; ++j) {
                src_بلوک_f32[j] = (عائم) src_بلوک_t[j];
            }
            cpyblck(reinterpret_cast<ثابت حرف *>(src_بلوک_f32), dst_بلوک);
        }
    });
    GGML_استفادهنشده(ne10);
    GGML_استفادهنشده(ne13);
    GGML_استفادهنشده(nb00);
    GGML_استفادهنشده(nb13);
}

قالب<نام_نوع بلوک_Type>
استفاده quantize_ردیف_qk_t = فراغ (*)(ثابت عائم *, بلوک_Type *, int64_t);

استفاده quantize_ردیفها_f_t = size_t (*)(ثابت عائم *, فراغ *, int64_t, int64_t, ثابت عائم *);

قالب <نام_نوع TIn, نام_نوع TIdx, نام_نوع بلوک_Type, صحيح qk, quantize_ردیف_qk_t<بلوک_Type> quantize_ردیف>
ساكن فراغ مجموعه_ردیفها_sycl_qk_host(
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
        صف_ptr stream) {
    GGML_ASSERT(ne00 % qk == 0);

    ثابت size_t src0_بایتها = ggml_nbytes(src0);
    ثابت size_t src1_بایتها = ggml_nbytes(src1);

    std::vector<حرف> src0_host(src0_بایتها);
    std::vector<حرف> src1_host(src1_بایتها);

    stream->کپی_حافظه(src0_host.داده(), src0->داده, src0_بایتها);
    stream->کپی_حافظه(src1_host.داده(), src1->داده, src1_بایتها);
    stream->wait();

    std::vector<عائم> src_ردیف_f32(ne00);
    ثابت int64_t nblocks = ne00 / qk;
    std::vector<بلوک_Type> dst_ردیف_q(nblocks);

    لكل (int64_t i03 = 0; i03 < ne03; ++i03) {
        لكل (int64_t i02 = 0; i02 < ne02; ++i02) {
            لكل (int64_t i01 = 0; i01 < ne01; ++i01) {
                ثابت int64_t i12 = i03 % ne12;
                ثابت int64_t i11 = i02 % ne11;
                ثابت int64_t i10 = i01;

                ثابت size_t src1_جابه_جایی = calculate_جابه_جایی<3>({ nb10, nb11, nb12 }, { i10, i11, i12 });
                ثابت int64_t dst_ردیف = *(ثابت TIdx *) (src1_host.داده() + src1_جابه_جایی);

                ثابت size_t src0_ردیف_جابه_جایی = calculate_جابه_جایی<3>({ nb01, nb02, nb03 }, { i01, i02, i03 });
                ثابت TIn * src_ردیف = reinterpret_cast<ثابت TIn *>(src0_host.داده() + src0_ردیف_جابه_جایی);

                لكل (int64_t i00 = 0; i00 < ne00; ++i00) {
                    src_ردیف_f32[i00] = (عائم) src_ردیف[i00];
                }

                quantize_ردیف(src_ردیف_f32.داده(), dst_ردیف_q.داده(), ne00);

                ثابت size_t dst_جابه_جایی = calculate_جابه_جایی<3>({ nb1, nb2, nb3 }, { dst_ردیف, i02, i03 });
                stream->کپی_حافظه((حرف *) dst->داده + dst_جابه_جایی, dst_ردیف_q.داده(), nblocks * حجم(بلوک_Type));
                stream->wait();
            }
        }
    }
}

قالب <نام_نوع TIn, نام_نوع TIdx, نام_نوع بلوک_Type, صحيح qk, quantize_ردیفها_f_t quantize_ردیفها>
ساكن فراغ مجموعه_ردیفها_sycl_iq_host(
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
        صف_ptr stream) {
    GGML_ASSERT(ne00 % qk == 0);

    ثابت size_t src0_بایتها = ggml_nbytes(src0);
    ثابت size_t src1_بایتها = ggml_nbytes(src1);

    std::vector<حرف> src0_host(src0_بایتها);
    std::vector<حرف> src1_host(src1_بایتها);

    stream->کپی_حافظه(src0_host.داده(), src0->داده, src0_بایتها);
    stream->کپی_حافظه(src1_host.داده(), src1->داده, src1_بایتها);
    stream->wait();

    std::vector<عائم> src_ردیف_f32(ne00);
    ثابت int64_t nblocks = ne00 / qk;
    std::vector<بلوک_Type> dst_ردیف_q(nblocks);

    لكل (int64_t i03 = 0; i03 < ne03; ++i03) {
        لكل (int64_t i02 = 0; i02 < ne02; ++i02) {
            لكل (int64_t i01 = 0; i01 < ne01; ++i01) {
                ثابت int64_t i12 = i03 % ne12;
                ثابت int64_t i11 = i02 % ne11;
                ثابت int64_t i10 = i01;

                ثابت size_t src1_جابه_جایی = calculate_جابه_جایی<3>({ nb10, nb11, nb12 }, { i10, i11, i12 });
                ثابت int64_t dst_ردیف = *(ثابت TIdx *) (src1_host.داده() + src1_جابه_جایی);

                ثابت size_t src0_ردیف_جابه_جایی = calculate_جابه_جایی<3>({ nb01, nb02, nb03 }, { i01, i02, i03 });
                ثابت TIn * src_ردیف = reinterpret_cast<ثابت TIn *>(src0_host.داده() + src0_ردیف_جابه_جایی);

                لكل (int64_t i00 = 0; i00 < ne00; ++i00) {
                    src_ردیف_f32[i00] = (عائم) src_ردیف[i00];
                }

                quantize_ردیفها(src_ردیف_f32.داده(), dst_ردیف_q.داده(), 1, ne00, اشاره_تهی);

                ثابت size_t dst_جابه_جایی = calculate_جابه_جایی<3>({ nb1, nb2, nb3 }, { dst_ردیف, i02, i03 });
                stream->کپی_حافظه((حرف *) dst->داده + dst_جابه_جایی, dst_ردیف_q.داده(), nblocks * حجم(بلوک_Type));
                stream->wait();
            }
        }
    }
}

قالب<نام_نوع TIn, نام_نوع TIdx, نام_نوع TOut>
ساكن فراغ k_مجموعه_ردیفها(
        ثابت حرف * __restrict__ src0, ثابت TIdx * __restrict__ src1, حرف * __restrict__ dst,
        ثابت int64_t ne00, ثابت int64_t ne01, ثابت int64_t ne02,
        ثابت int64_t ne11, ثابت int64_t ne12,
        ثابت size_t nb01, ثابت size_t nb02, ثابت size_t nb03,
        ثابت size_t nb10, ثابت size_t nb11, ثابت size_t nb12,
        ثابت size_t nb1, ثابت size_t nb2, ثابت size_t nb3,
        ثابت size_t src_type_اندازه, ثابت size_t dst_type_اندازه,
        ثابت int64_t مجموع_elements,
        ثابت sycl::nd_قلم<1> & قلم_ct1) {

    ثابت int64_t i = قلم_ct1.get_global_خطی_id();
    إذا (i >= مجموع_elements) {
        أرجع;
    }

    ثابت int64_t i03 = i / (ne00 * ne01 * ne02);
    ثابت int64_t i02 = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    ثابت int64_t i01 = (i - i03 * ne00 * ne01 * ne02 - i02 * ne00 * ne01) / ne00;
    ثابت int64_t i00 = i - i03 * ne00 * ne01 * ne02 - i02 * ne00 * ne01 - i01 * ne00;

    ثابت int64_t i12 = i03 % ne12;
    ثابت int64_t i11 = i02 % ne11;
    ثابت int64_t i10 = i01;

    ثابت int64_t dst_ردیف = *(ثابت TIdx *)((ثابت حرف *)src1 + calculate_جابه_جایی<3>({nb10, nb11, nb12}, {i10, i11, i12}));

    ثابت حرف * src0_ردیف = src0 + calculate_جابه_جایی<3>({nb01, nb02, nb03}, {i01, i02, i03});
    ثابت حرف * src_elem = src0_ردیف + i00 * src_type_اندازه;
    حرف * dst_ردیف_ptr = dst + dst_ردیف*nb1 + i02*nb2 + i03*nb3;
    حرف * dst_elem = dst_ردیف_ptr + i00 * dst_type_اندازه;

    convert<TIn, TOut>(src_elem, dst_elem);
}

قالب<نام_نوع TIn, نام_نوع TIdx, نام_نوع TOut>
ساكن فراغ مجموعه_ردیفها_sycl(
        ثابت حرف * src0_d, ثابت TIdx * src1_d, حرف * dst_d,
        ثابت int64_t ne00, ثابت int64_t ne01, ثابت int64_t ne02, ثابت int64_t ne03,
        ثابت int64_t ne11, ثابت int64_t ne12, ثابت size_t nb01, ثابت size_t nb02, ثابت size_t nb03,
        ثابت size_t nb10, ثابت size_t nb11, ثابت size_t nb12,
        ثابت size_t nb1, ثابت size_t nb2, ثابت size_t nb3,
        ثابت size_t src_type_اندازه, ثابت size_t dst_type_اندازه,
        صف_ptr stream) {

    ثابت int64_t مجموع_elements = ne00 * ne01 * ne02 * ne03;

    constexpr صحيح بلوک_اندازه_3 = 64;
    ثابت int64_t grid_اندازه = ceil_div(مجموع_elements, بلوک_اندازه_3);

    stream->parallel_for(
        sycl::nd_range<1>(grid_اندازه * بلوک_اندازه_3, بلوک_اندازه_3),
        [=](sycl::nd_قلم<1> قلم_ct1) [[sycl::reqd_sub_گروه_اندازه(WARP_اندازه)]] {
            k_مجموعه_ردیفها<TIn, TIdx, TOut>(
                src0_d, src1_d, dst_d,
                ne00, ne01, ne02,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                src_type_اندازه, dst_type_اندازه,
                مجموع_elements,
                قلم_ct1
            );
        }
    );
}

قالب<نام_نوع TIn, نام_نوع TIdx>
ساكن فراغ مجموعه_ردیفها_sycl(ggml_backend_sycl_context & ctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst) {
    ثابت حرف * src0_d = (ثابت حرف *)src0->داده;
    ثابت TIdx * src1_d = (ثابت TIdx *)src1->داده;

    GGML_TENSOR_دودویی_OP_LOCALS

    dpct::صف_ptr stream = ctx.stream();
    بدل (dst->type) {
        حالة GGML_TYPE_F32:
            مجموعه_ردیفها_sycl<TIn, TIdx, عائم>(
                src0_d, src1_d, (حرف *)dst->داده,
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
            مجموعه_ردیفها_sycl<TIn, TIdx, sycl::half>(
                src0_d, src1_d, (حرف *)dst->داده,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                حجم(TIn), حجم(sycl::half),
                stream
            );
            اكسر;
#اگر_تعریف_شده GGML_SYCL_HAS_BF16
        حالة GGML_TYPE_BF16:
            مجموعه_ردیفها_sycl<TIn, TIdx, sycl::ext::oneapi::bfloat16>(
                src0_d, src1_d, (حرف *)dst->داده,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                حجم(TIn), حجم(sycl::ext::oneapi::bfloat16),
                stream
            );
            اكسر;
#پایان_اگر
        حالة GGML_TYPE_Q8_0:
            مجموعه_ردیفها_sycl_q<TIn, TIdx, بلوک_q8_0, QK8_0, cpy_blck_f32_q8_0>(
                src0_d, src1_d, (بلوک_q8_0 *) dst->داده, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            اكسر;
        حالة GGML_TYPE_Q1_0:
            مجموعه_ردیفها_sycl_q<TIn, TIdx, بلوک_q1_0, QK1_0, cpy_blck_f32_q1_0>(
                src0_d, src1_d, (بلوک_q1_0 *) dst->داده, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            اكسر;
        حالة GGML_TYPE_Q2_0:
            مجموعه_ردیفها_sycl_q<TIn, TIdx, بلوک_q2_0, QK2_0, cpy_blck_f32_q2_0>(
                src0_d, src1_d, (بلوک_q2_0 *) dst->داده, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            اكسر;
        حالة GGML_TYPE_Q5_1:
            مجموعه_ردیفها_sycl_q<TIn, TIdx, بلوک_q5_1, QK5_1, cpy_blck_f32_q5_1>(
                src0_d, src1_d, (بلوک_q5_1 *) dst->داده, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            اكسر;
        حالة GGML_TYPE_Q5_0:
            مجموعه_ردیفها_sycl_q<TIn, TIdx, بلوک_q5_0, QK5_0, cpy_blck_f32_q5_0>(
                src0_d, src1_d, (بلوک_q5_0 *) dst->داده, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            اكسر;
        حالة GGML_TYPE_Q4_1:
            مجموعه_ردیفها_sycl_q<TIn, TIdx, بلوک_q4_1, QK4_1, cpy_blck_f32_q4_1>(
                src0_d, src1_d, (بلوک_q4_1 *) dst->داده, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            اكسر;
        حالة GGML_TYPE_Q4_0:
            مجموعه_ردیفها_sycl_q<TIn, TIdx, بلوک_q4_0, QK4_0, cpy_blck_f32_q4_0>(
                src0_d, src1_d, (بلوک_q4_0 *) dst->داده, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            اكسر;
        حالة GGML_TYPE_IQ4_NL:
            مجموعه_ردیفها_sycl_q<TIn, TIdx, بلوک_iq4_nl, QK4_NL, cpy_blck_f32_iq4_nl>(
                src0_d, src1_d, (بلوک_iq4_nl *) dst->داده, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            اكسر;
        حالة GGML_TYPE_MXFP4:
            مجموعه_ردیفها_sycl_q<TIn, TIdx, بلوک_mxfp4, QK_MXFP4, cpy_blck_f32_mxfp4>(
                src0_d, src1_d, (بلوک_mxfp4 *) dst->داده, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            اكسر;
        حالة GGML_TYPE_NVFP4:
            مجموعه_ردیفها_sycl_q<TIn, TIdx, بلوک_nvfp4, QK_NVFP4, cpy_blck_f32_nvfp4>(
                src0_d, src1_d, (بلوک_nvfp4 *) dst->داده, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            اكسر;
        حالة GGML_TYPE_Q2_K:
            مجموعه_ردیفها_sycl_qk_host<TIn, TIdx, بلوک_q2_K, QK_K, quantize_ردیف_q2_K_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            اكسر;
        حالة GGML_TYPE_Q3_K:
            مجموعه_ردیفها_sycl_qk_host<TIn, TIdx, بلوک_q3_K, QK_K, quantize_ردیف_q3_K_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            اكسر;
        حالة GGML_TYPE_Q4_K:
            مجموعه_ردیفها_sycl_qk_host<TIn, TIdx, بلوک_q4_K, QK_K, quantize_ردیف_q4_K_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            اكسر;
        حالة GGML_TYPE_Q5_K:
            مجموعه_ردیفها_sycl_qk_host<TIn, TIdx, بلوک_q5_K, QK_K, quantize_ردیف_q5_K_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            اكسر;
        حالة GGML_TYPE_Q6_K:
            مجموعه_ردیفها_sycl_qk_host<TIn, TIdx, بلوک_q6_K, QK_K, quantize_ردیف_q6_K_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            اكسر;
        حالة GGML_TYPE_IQ2_XXS:
            مجموعه_ردیفها_sycl_iq_host<TIn, TIdx, بلوک_iq2_xxs, QK_K, quantize_iq2_xxs>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            اكسر;
        حالة GGML_TYPE_IQ2_XS:
            مجموعه_ردیفها_sycl_iq_host<TIn, TIdx, بلوک_iq2_xs, QK_K, quantize_iq2_xs>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            اكسر;
        حالة GGML_TYPE_IQ2_S:
            مجموعه_ردیفها_sycl_iq_host<TIn, TIdx, بلوک_iq2_s, QK_K, quantize_iq2_s>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            اكسر;
        حالة GGML_TYPE_IQ3_XXS:
            مجموعه_ردیفها_sycl_qk_host<TIn, TIdx, بلوک_iq3_xxs, QK_K, quantize_ردیف_iq3_xxs_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            اكسر;
        حالة GGML_TYPE_IQ3_S:
            مجموعه_ردیفها_sycl_qk_host<TIn, TIdx, بلوک_iq3_s, QK_K, quantize_ردیف_iq3_s_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            اكسر;
        حالة GGML_TYPE_IQ1_S:
            مجموعه_ردیفها_sycl_iq_host<TIn, TIdx, بلوک_iq1_s, QK_K, quantize_iq1_s>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            اكسر;
        حالة GGML_TYPE_IQ1_M:
            مجموعه_ردیفها_sycl_iq_host<TIn, TIdx, بلوک_iq1_m, QK_K, quantize_iq1_m>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            اكسر;
        حالة GGML_TYPE_IQ4_XS:
            مجموعه_ردیفها_sycl_qk_host<TIn, TIdx, بلوک_iq4_xs, QK_K, quantize_ردیف_iq4_xs_ref>(
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

فراغ ggml_sycl_op_مجموعه_ردیفها(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->src[1]->type == GGML_TYPE_I64 || dst->src[1]->type == GGML_TYPE_I32);

    // dispatch on the index type (src1) and the source value type (src0)
    إذا (src0->type == GGML_TYPE_F16) {
        إذا (src1->type == GGML_TYPE_I64) {
            مجموعه_ردیفها_sycl<sycl::half, int64_t>(ctx, src0, src1, dst);
        } وإلا {
            مجموعه_ردیفها_sycl<sycl::half, int32_t>(ctx, src0, src1, dst);
        }
    } وإلا {
        إذا (src1->type == GGML_TYPE_I64) {
            مجموعه_ردیفها_sycl<عائم, int64_t>(ctx, src0, src1, dst);
        } وإلا {
            مجموعه_ردیفها_sycl<عائم, int32_t>(ctx, src0, src1, dst);
        }
    }
}
