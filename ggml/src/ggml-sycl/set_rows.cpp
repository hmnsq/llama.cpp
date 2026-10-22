#ಅನ್ತರ್ಭಾವ "set_rows.hpp"
#ಅನ್ತರ್ಭಾವ "cpy.hpp"

#ಅನ್ತರ್ಭಾವ "ggml-quants.h"

#ಅನ್ತರ್ಭಾವ <vector>

ನಾಮಾಕಾಶ utils {
ಮಾದರಿ<ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ constexpr ತಾರ್ಕಿಕ is_arithmetic_v() {
    ಹಿಂತಿರುಗಿ std::is_arithmetic_v<T> || std::is_same_v<T, sycl::half>
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_HAS_BF16
        || std::is_same_v<T, sycl::ext::oneapi::bfloat16>
#ಆದರೆ_ಅಂತ್ಯ
        ;
}
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು TIn, ಬಗೆ_ಹೆಸರು TOut>
ಸ್ಥಿರ ಸಾಲೊಳಗೆ std::enable_if_t<utils::is_arithmetic_v<TIn>() && utils::is_arithmetic_v<TOut>(), ಶೂನ್ಯ>
convert (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* src, ಅಕ್ಷರ* dst) {
    ಸ್ವಯಂಚಾಲಿತ src_val = *reinterpret_cast<ಸ್ಥಿರಾಂಕ TIn*>(src);
    ಸ್ವಯಂಚಾಲಿತ dst_val = sycl::vec<TIn, 1>(src_val).ಮಾದರಿ convert<TOut, sycl::rounding_ಬಹುಲಕ::automatic>()[0];
   *reinterpret_cast<TOut*>(dst) = dst_val;
}

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_HAS_BF16
// sycl::vec::convert does not provide a half -> bfloat16 path, so route through float.
ಮಾದರಿ<>
ಸಾಲೊಳಗೆ ಶೂನ್ಯ convert<sycl::half, sycl::ext::oneapi::bfloat16>(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* src, ಅಕ್ಷರ* dst) {
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ tmp = sycl::vec<sycl::half, 1>(*reinterpret_cast<ಸ್ಥಿರಾಂಕ sycl::half*>(src))
                          .ಮಾದರಿ convert<ದಶಮಾಂಶ, sycl::rounding_ಬಹುಲಕ::automatic>()[0];
    *reinterpret_cast<sycl::ext::oneapi::bfloat16*>(dst) = sycl::ext::oneapi::bfloat16(tmp);
}
#ಆದರೆ_ಅಂತ್ಯ

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು TIn, ಬಗೆ_ಹೆಸರು TIdx, ಬಗೆ_ಹೆಸರು ಖಂಡ_Type, ಪೂರ್ಣಾಂಕ qk, cpy_kernel_t cpyblck>
ಸ್ಥಿರ ಶೂನ್ಯ ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_sycl_q(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * __restrict__ src0_d,
                            ಸ್ಥಿರಾಂಕ TIdx * __restrict__ src1_d,
                            ಖಂಡ_Type * __restrict__ dst_d,
                            // tensor dimensions src0 and src1
                            ಸ್ಥಿರಾಂಕ int64_t ne00,
                            ಸ್ಥಿರಾಂಕ int64_t ne01,
                            ಸ್ಥಿರಾಂಕ int64_t ne02,
                            ಸ್ಥಿರಾಂಕ int64_t ne03,
                            ಸ್ಥಿರಾಂಕ int64_t ne10,
                            ಸ್ಥಿರಾಂಕ int64_t ne11,
                            ಸ್ಥಿರಾಂಕ int64_t ne12,
                            ಸ್ಥಿರಾಂಕ int64_t ne13,
                            // strides for src0
                            ಸ್ಥಿರಾಂಕ size_t  nb00,
                            ಸ್ಥಿರಾಂಕ size_t  nb01,
                            ಸ್ಥಿರಾಂಕ size_t  nb02,
                            ಸ್ಥಿರಾಂಕ size_t  nb03,
                            // strides for src1
                            ಸ್ಥಿರಾಂಕ size_t  nb10,
                            ಸ್ಥಿರಾಂಕ size_t  nb11,
                            ಸ್ಥಿರಾಂಕ size_t  nb12,
                            ಸ್ಥಿರಾಂಕ size_t  nb13,
                            // strides for dst
                            ಸ್ಥಿರಾಂಕ size_t  nb1,
                            ಸ್ಥಿರಾಂಕ size_t  nb2,
                            ಸ್ಥಿರಾಂಕ size_t  nb3,
                            ಸರತಿ_ptr     stream) {
    ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_ಖಂಡಗಳು = (ne00 * ne01 * ne02 * ne03) / qk;
    constexpr ಪೂರ್ಣಾಂಕ ಖಂಡ_ಗಾತ್ರ_3   = 256;
    ಸ್ಥಿರಾಂಕ int64_t grid_ಗಾತ್ರ    = ceil_div(ಒಟ್ಟು_ಖಂಡಗಳು, ಖಂಡ_ಗಾತ್ರ_3);

    stream->parallel_for(sycl::nd_range<1>(grid_ಗಾತ್ರ * ಖಂಡ_ಗಾತ್ರ_3, ಖಂಡ_ಗಾತ್ರ_3), [=](sycl::nd_ಅಂಶ<1> ಅಂಶ_ct1) {
        ಸ್ಥಿರಾಂಕ int64_t i = ಅಂಶ_ct1.get_global_ರೇಖೀಯ_id();
        ಆದರೆ (i >= ಒಟ್ಟು_ಖಂಡಗಳು) {
            ಹಿಂತಿರುಗಿ;
        }
        ಸ್ಥಿರಾಂಕ int64_t i_ತಳ      = i * qk;
        ಸ್ಥಿರಾಂಕ int64_t i03         = i_ತಳ / (ne00 * ne01 * ne02);
        ಸ್ಥಿರಾಂಕ int64_t rem1        = i_ತಳ - i03 * (ne00 * ne01 * ne02);
        ಸ್ಥಿರಾಂಕ int64_t i02         = rem1 / (ne00 * ne01);
        ಸ್ಥಿರಾಂಕ int64_t rem2        = rem1 - i02 * ne00 * ne01;
        ಸ್ಥಿರಾಂಕ int64_t i01         = rem2 / ne00;
        ಸ್ಥಿರಾಂಕ int64_t i00         = rem2 - i01 * ne00;
        ಸ್ಥಿರಾಂಕ int64_t i12         = i03 % ne12;
        ಸ್ಥಿರಾಂಕ int64_t i11         = i02 % ne11;
        ಸ್ಥಿರಾಂಕ int64_t i10         = i01;
        ಸ್ಥಿರಾಂಕ size_t  src_ಸ್ಥಾನಾಂತರ  = calculate_ಸ್ಥಾನಾಂತರ<3>({ nb01, nb02, nb03 }, { i01, i02, i03 });
        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *  src_ಖಂಡ   = src0_d + src_ಸ್ಥಾನಾಂತರ + i00 * ಗಾತ್ರ(TIn);
        ಸ್ಥಿರಾಂಕ size_t  src1_ಸ್ಥಾನಾಂತರ = calculate_ಸ್ಥಾನಾಂತರ<3>({ nb10, nb11, nb12 }, { i10, i11, i12 });
        ಸ್ಥಿರಾಂಕ int64_t dst_ಅಡ್ಡಸಾಲು     = src1_d[src1_ಸ್ಥಾನಾಂತರ / ಗಾತ್ರ(TIdx)];
        ಸ್ಥಿರಾಂಕ size_t  dst_ಸ್ಥಾನಾಂತರ =
            calculate_ಸ್ಥಾನಾಂತರ<3>({ nb1, nb2, nb3 }, { dst_ಅಡ್ಡಸಾಲು, i02, i03 }) + (i00 / qk) * ಗಾತ್ರ(ಖಂಡ_Type);
        ಅಕ್ಷರ * dst_ಖಂಡ = reinterpret_cast<ಅಕ್ಷರ *>(reinterpret_cast<ಅಕ್ಷರ *>(dst_d) + dst_ಸ್ಥಾನಾಂತರ);
        ಆದರೆ constexpr (std::is_same_v<TIn, ದಶಮಾಂಶ>) {
            cpyblck(src_ಖಂಡ, dst_ಖಂಡ);
        } ಇಲ್ಲದಿದ್ದರೆ {
            ದಶಮಾಂಶ src_ಖಂಡ_f32[qk];
            ಸ್ಥಿರಾಂಕ TIn * src_ಖಂಡ_t = reinterpret_cast<ಸ್ಥಿರಾಂಕ TIn *>(src_ಖಂಡ);
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < qk; ++j) {
                src_ಖಂಡ_f32[j] = (ದಶಮಾಂಶ) src_ಖಂಡ_t[j];
            }
            cpyblck(reinterpret_cast<ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *>(src_ಖಂಡ_f32), dst_ಖಂಡ);
        }
    });
    GGML_ಬಳಸದ(ne10);
    GGML_ಬಳಸದ(ne13);
    GGML_ಬಳಸದ(nb00);
    GGML_ಬಳಸದ(nb13);
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು ಖಂಡ_Type>
ಬಳಸಿ quantize_ಅಡ್ಡಸಾಲು_qk_t = ಶೂನ್ಯ (*)(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *, ಖಂಡ_Type *, int64_t);

ಬಳಸಿ quantize_ಅಡ್ಡಸಾಲುಗಳು_f_t = size_t (*)(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *, ಶೂನ್ಯ *, int64_t, int64_t, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *);

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು TIn, ಬಗೆ_ಹೆಸರು TIdx, ಬಗೆ_ಹೆಸರು ಖಂಡ_Type, ಪೂರ್ಣಾಂಕ qk, quantize_ಅಡ್ಡಸಾಲು_qk_t<ಖಂಡ_Type> quantize_ಅಡ್ಡಸಾಲು>
ಸ್ಥಿರ ಶೂನ್ಯ ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_sycl_qk_host(
        ಸ್ಥಿರಾಂಕ ggml_tensor * src0,
        ಸ್ಥಿರಾಂಕ ggml_tensor * src1,
        ggml_tensor * dst,
        ಸ್ಥಿರಾಂಕ int64_t ne00,
        ಸ್ಥಿರಾಂಕ int64_t ne01,
        ಸ್ಥಿರಾಂಕ int64_t ne02,
        ಸ್ಥಿರಾಂಕ int64_t ne03,
        ಸ್ಥಿರಾಂಕ int64_t ne11,
        ಸ್ಥಿರಾಂಕ int64_t ne12,
        ಸ್ಥಿರಾಂಕ size_t nb01,
        ಸ್ಥಿರಾಂಕ size_t nb02,
        ಸ್ಥಿರಾಂಕ size_t nb03,
        ಸ್ಥಿರಾಂಕ size_t nb10,
        ಸ್ಥಿರಾಂಕ size_t nb11,
        ಸ್ಥಿರಾಂಕ size_t nb12,
        ಸ್ಥಿರಾಂಕ size_t nb1,
        ಸ್ಥಿರಾಂಕ size_t nb2,
        ಸ್ಥಿರಾಂಕ size_t nb3,
        ಸರತಿ_ptr stream) {
    GGML_ASSERT(ne00 % qk == 0);

    ಸ್ಥಿರಾಂಕ size_t src0_ಬೈಟುಗಳು = ggml_nbytes(src0);
    ಸ್ಥಿರಾಂಕ size_t src1_ಬೈಟುಗಳು = ggml_nbytes(src1);

    std::vector<ಅಕ್ಷರ> src0_host(src0_ಬೈಟುಗಳು);
    std::vector<ಅಕ್ಷರ> src1_host(src1_ಬೈಟುಗಳು);

    stream->ಸ್ಮರಣೆ_ನಕಲು(src0_host.ದತ್ತಾಂಶ(), src0->ದತ್ತಾಂಶ, src0_ಬೈಟುಗಳು);
    stream->ಸ್ಮರಣೆ_ನಕಲು(src1_host.ದತ್ತಾಂಶ(), src1->ದತ್ತಾಂಶ, src1_ಬೈಟುಗಳು);
    stream->wait();

    std::vector<ದಶಮಾಂಶ> src_ಅಡ್ಡಸಾಲು_f32(ne00);
    ಸ್ಥಿರಾಂಕ int64_t nblocks = ne00 / qk;
    std::vector<ಖಂಡ_Type> dst_ಅಡ್ಡಸಾಲು_q(nblocks);

    ಗಾಗಿ (int64_t i03 = 0; i03 < ne03; ++i03) {
        ಗಾಗಿ (int64_t i02 = 0; i02 < ne02; ++i02) {
            ಗಾಗಿ (int64_t i01 = 0; i01 < ne01; ++i01) {
                ಸ್ಥಿರಾಂಕ int64_t i12 = i03 % ne12;
                ಸ್ಥಿರಾಂಕ int64_t i11 = i02 % ne11;
                ಸ್ಥಿರಾಂಕ int64_t i10 = i01;

                ಸ್ಥಿರಾಂಕ size_t src1_ಸ್ಥಾನಾಂತರ = calculate_ಸ್ಥಾನಾಂತರ<3>({ nb10, nb11, nb12 }, { i10, i11, i12 });
                ಸ್ಥಿರಾಂಕ int64_t dst_ಅಡ್ಡಸಾಲು = *(ಸ್ಥಿರಾಂಕ TIdx *) (src1_host.ದತ್ತಾಂಶ() + src1_ಸ್ಥಾನಾಂತರ);

                ಸ್ಥಿರಾಂಕ size_t src0_ಅಡ್ಡಸಾಲು_ಸ್ಥಾನಾಂತರ = calculate_ಸ್ಥಾನಾಂತರ<3>({ nb01, nb02, nb03 }, { i01, i02, i03 });
                ಸ್ಥಿರಾಂಕ TIn * src_ಅಡ್ಡಸಾಲು = reinterpret_cast<ಸ್ಥಿರಾಂಕ TIn *>(src0_host.ದತ್ತಾಂಶ() + src0_ಅಡ್ಡಸಾಲು_ಸ್ಥಾನಾಂತರ);

                ಗಾಗಿ (int64_t i00 = 0; i00 < ne00; ++i00) {
                    src_ಅಡ್ಡಸಾಲು_f32[i00] = (ದಶಮಾಂಶ) src_ಅಡ್ಡಸಾಲು[i00];
                }

                quantize_ಅಡ್ಡಸಾಲು(src_ಅಡ್ಡಸಾಲು_f32.ದತ್ತಾಂಶ(), dst_ಅಡ್ಡಸಾಲು_q.ದತ್ತಾಂಶ(), ne00);

                ಸ್ಥಿರಾಂಕ size_t dst_ಸ್ಥಾನಾಂತರ = calculate_ಸ್ಥಾನಾಂತರ<3>({ nb1, nb2, nb3 }, { dst_ಅಡ್ಡಸಾಲು, i02, i03 });
                stream->ಸ್ಮರಣೆ_ನಕಲು((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + dst_ಸ್ಥಾನಾಂತರ, dst_ಅಡ್ಡಸಾಲು_q.ದತ್ತಾಂಶ(), nblocks * ಗಾತ್ರ(ಖಂಡ_Type));
                stream->wait();
            }
        }
    }
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು TIn, ಬಗೆ_ಹೆಸರು TIdx, ಬಗೆ_ಹೆಸರು ಖಂಡ_Type, ಪೂರ್ಣಾಂಕ qk, quantize_ಅಡ್ಡಸಾಲುಗಳು_f_t quantize_ಅಡ್ಡಸಾಲುಗಳು>
ಸ್ಥಿರ ಶೂನ್ಯ ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_sycl_iq_host(
        ಸ್ಥಿರಾಂಕ ggml_tensor * src0,
        ಸ್ಥಿರಾಂಕ ggml_tensor * src1,
        ggml_tensor * dst,
        ಸ್ಥಿರಾಂಕ int64_t ne00,
        ಸ್ಥಿರಾಂಕ int64_t ne01,
        ಸ್ಥಿರಾಂಕ int64_t ne02,
        ಸ್ಥಿರಾಂಕ int64_t ne03,
        ಸ್ಥಿರಾಂಕ int64_t ne11,
        ಸ್ಥಿರಾಂಕ int64_t ne12,
        ಸ್ಥಿರಾಂಕ size_t nb01,
        ಸ್ಥಿರಾಂಕ size_t nb02,
        ಸ್ಥಿರಾಂಕ size_t nb03,
        ಸ್ಥಿರಾಂಕ size_t nb10,
        ಸ್ಥಿರಾಂಕ size_t nb11,
        ಸ್ಥಿರಾಂಕ size_t nb12,
        ಸ್ಥಿರಾಂಕ size_t nb1,
        ಸ್ಥಿರಾಂಕ size_t nb2,
        ಸ್ಥಿರಾಂಕ size_t nb3,
        ಸರತಿ_ptr stream) {
    GGML_ASSERT(ne00 % qk == 0);

    ಸ್ಥಿರಾಂಕ size_t src0_ಬೈಟುಗಳು = ggml_nbytes(src0);
    ಸ್ಥಿರಾಂಕ size_t src1_ಬೈಟುಗಳು = ggml_nbytes(src1);

    std::vector<ಅಕ್ಷರ> src0_host(src0_ಬೈಟುಗಳು);
    std::vector<ಅಕ್ಷರ> src1_host(src1_ಬೈಟುಗಳು);

    stream->ಸ್ಮರಣೆ_ನಕಲು(src0_host.ದತ್ತಾಂಶ(), src0->ದತ್ತಾಂಶ, src0_ಬೈಟುಗಳು);
    stream->ಸ್ಮರಣೆ_ನಕಲು(src1_host.ದತ್ತಾಂಶ(), src1->ದತ್ತಾಂಶ, src1_ಬೈಟುಗಳು);
    stream->wait();

    std::vector<ದಶಮಾಂಶ> src_ಅಡ್ಡಸಾಲು_f32(ne00);
    ಸ್ಥಿರಾಂಕ int64_t nblocks = ne00 / qk;
    std::vector<ಖಂಡ_Type> dst_ಅಡ್ಡಸಾಲು_q(nblocks);

    ಗಾಗಿ (int64_t i03 = 0; i03 < ne03; ++i03) {
        ಗಾಗಿ (int64_t i02 = 0; i02 < ne02; ++i02) {
            ಗಾಗಿ (int64_t i01 = 0; i01 < ne01; ++i01) {
                ಸ್ಥಿರಾಂಕ int64_t i12 = i03 % ne12;
                ಸ್ಥಿರಾಂಕ int64_t i11 = i02 % ne11;
                ಸ್ಥಿರಾಂಕ int64_t i10 = i01;

                ಸ್ಥಿರಾಂಕ size_t src1_ಸ್ಥಾನಾಂತರ = calculate_ಸ್ಥಾನಾಂತರ<3>({ nb10, nb11, nb12 }, { i10, i11, i12 });
                ಸ್ಥಿರಾಂಕ int64_t dst_ಅಡ್ಡಸಾಲು = *(ಸ್ಥಿರಾಂಕ TIdx *) (src1_host.ದತ್ತಾಂಶ() + src1_ಸ್ಥಾನಾಂತರ);

                ಸ್ಥಿರಾಂಕ size_t src0_ಅಡ್ಡಸಾಲು_ಸ್ಥಾನಾಂತರ = calculate_ಸ್ಥಾನಾಂತರ<3>({ nb01, nb02, nb03 }, { i01, i02, i03 });
                ಸ್ಥಿರಾಂಕ TIn * src_ಅಡ್ಡಸಾಲು = reinterpret_cast<ಸ್ಥಿರಾಂಕ TIn *>(src0_host.ದತ್ತಾಂಶ() + src0_ಅಡ್ಡಸಾಲು_ಸ್ಥಾನಾಂತರ);

                ಗಾಗಿ (int64_t i00 = 0; i00 < ne00; ++i00) {
                    src_ಅಡ್ಡಸಾಲು_f32[i00] = (ದಶಮಾಂಶ) src_ಅಡ್ಡಸಾಲು[i00];
                }

                quantize_ಅಡ್ಡಸಾಲುಗಳು(src_ಅಡ್ಡಸಾಲು_f32.ದತ್ತಾಂಶ(), dst_ಅಡ್ಡಸಾಲು_q.ದತ್ತಾಂಶ(), 1, ne00, ಶೂನ್ಯಸೂಚಕ);

                ಸ್ಥಿರಾಂಕ size_t dst_ಸ್ಥಾನಾಂತರ = calculate_ಸ್ಥಾನಾಂತರ<3>({ nb1, nb2, nb3 }, { dst_ಅಡ್ಡಸಾಲು, i02, i03 });
                stream->ಸ್ಮರಣೆ_ನಕಲು((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + dst_ಸ್ಥಾನಾಂತರ, dst_ಅಡ್ಡಸಾಲು_q.ದತ್ತಾಂಶ(), nblocks * ಗಾತ್ರ(ಖಂಡ_Type));
                stream->wait();
            }
        }
    }
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು TIn, ಬಗೆ_ಹೆಸರು TIdx, ಬಗೆ_ಹೆಸರು TOut>
ಸ್ಥಿರ ಶೂನ್ಯ k_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು(
        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * __restrict__ src0, ಸ್ಥಿರಾಂಕ TIdx * __restrict__ src1, ಅಕ್ಷರ * __restrict__ dst,
        ಸ್ಥಿರಾಂಕ int64_t ne00, ಸ್ಥಿರಾಂಕ int64_t ne01, ಸ್ಥಿರಾಂಕ int64_t ne02,
        ಸ್ಥಿರಾಂಕ int64_t ne11, ಸ್ಥಿರಾಂಕ int64_t ne12,
        ಸ್ಥಿರಾಂಕ size_t nb01, ಸ್ಥಿರಾಂಕ size_t nb02, ಸ್ಥಿರಾಂಕ size_t nb03,
        ಸ್ಥಿರಾಂಕ size_t nb10, ಸ್ಥಿರಾಂಕ size_t nb11, ಸ್ಥಿರಾಂಕ size_t nb12,
        ಸ್ಥಿರಾಂಕ size_t nb1, ಸ್ಥಿರಾಂಕ size_t nb2, ಸ್ಥಿರಾಂಕ size_t nb3,
        ಸ್ಥಿರಾಂಕ size_t src_type_ಗಾತ್ರ, ಸ್ಥಿರಾಂಕ size_t dst_type_ಗಾತ್ರ,
        ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_elements,
        ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<1> & ಅಂಶ_ct1) {

    ಸ್ಥಿರಾಂಕ int64_t i = ಅಂಶ_ct1.get_global_ರೇಖೀಯ_id();
    ಆದರೆ (i >= ಒಟ್ಟು_elements) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ int64_t i03 = i / (ne00 * ne01 * ne02);
    ಸ್ಥಿರಾಂಕ int64_t i02 = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    ಸ್ಥಿರಾಂಕ int64_t i01 = (i - i03 * ne00 * ne01 * ne02 - i02 * ne00 * ne01) / ne00;
    ಸ್ಥಿರಾಂಕ int64_t i00 = i - i03 * ne00 * ne01 * ne02 - i02 * ne00 * ne01 - i01 * ne00;

    ಸ್ಥಿರಾಂಕ int64_t i12 = i03 % ne12;
    ಸ್ಥಿರಾಂಕ int64_t i11 = i02 % ne11;
    ಸ್ಥಿರಾಂಕ int64_t i10 = i01;

    ಸ್ಥಿರಾಂಕ int64_t dst_ಅಡ್ಡಸಾಲು = *(ಸ್ಥಿರಾಂಕ TIdx *)((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)src1 + calculate_ಸ್ಥಾನಾಂತರ<3>({nb10, nb11, nb12}, {i10, i11, i12}));

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * src0_ಅಡ್ಡಸಾಲು = src0 + calculate_ಸ್ಥಾನಾಂತರ<3>({nb01, nb02, nb03}, {i01, i02, i03});
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * src_elem = src0_ಅಡ್ಡಸಾಲು + i00 * src_type_ಗಾತ್ರ;
    ಅಕ್ಷರ * dst_ಅಡ್ಡಸಾಲು_ptr = dst + dst_ಅಡ್ಡಸಾಲು*nb1 + i02*nb2 + i03*nb3;
    ಅಕ್ಷರ * dst_elem = dst_ಅಡ್ಡಸಾಲು_ptr + i00 * dst_type_ಗಾತ್ರ;

    convert<TIn, TOut>(src_elem, dst_elem);
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು TIn, ಬಗೆ_ಹೆಸರು TIdx, ಬಗೆ_ಹೆಸರು TOut>
ಸ್ಥಿರ ಶೂನ್ಯ ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_sycl(
        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * src0_d, ಸ್ಥಿರಾಂಕ TIdx * src1_d, ಅಕ್ಷರ * dst_d,
        ಸ್ಥಿರಾಂಕ int64_t ne00, ಸ್ಥಿರಾಂಕ int64_t ne01, ಸ್ಥಿರಾಂಕ int64_t ne02, ಸ್ಥಿರಾಂಕ int64_t ne03,
        ಸ್ಥಿರಾಂಕ int64_t ne11, ಸ್ಥಿರಾಂಕ int64_t ne12, ಸ್ಥಿರಾಂಕ size_t nb01, ಸ್ಥಿರಾಂಕ size_t nb02, ಸ್ಥಿರಾಂಕ size_t nb03,
        ಸ್ಥಿರಾಂಕ size_t nb10, ಸ್ಥಿರಾಂಕ size_t nb11, ಸ್ಥಿರಾಂಕ size_t nb12,
        ಸ್ಥಿರಾಂಕ size_t nb1, ಸ್ಥಿರಾಂಕ size_t nb2, ಸ್ಥಿರಾಂಕ size_t nb3,
        ಸ್ಥಿರಾಂಕ size_t src_type_ಗಾತ್ರ, ಸ್ಥಿರಾಂಕ size_t dst_type_ಗಾತ್ರ,
        ಸರತಿ_ptr stream) {

    ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_elements = ne00 * ne01 * ne02 * ne03;

    constexpr ಪೂರ್ಣಾಂಕ ಖಂಡ_ಗಾತ್ರ_3 = 64;
    ಸ್ಥಿರಾಂಕ int64_t grid_ಗಾತ್ರ = ceil_div(ಒಟ್ಟು_elements, ಖಂಡ_ಗಾತ್ರ_3);

    stream->parallel_for(
        sycl::nd_range<1>(grid_ಗಾತ್ರ * ಖಂಡ_ಗಾತ್ರ_3, ಖಂಡ_ಗಾತ್ರ_3),
        [=](sycl::nd_ಅಂಶ<1> ಅಂಶ_ct1) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
            k_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು<TIn, TIdx, TOut>(
                src0_d, src1_d, dst_d,
                ne00, ne01, ne02,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                src_type_ಗಾತ್ರ, dst_type_ಗಾತ್ರ,
                ಒಟ್ಟು_elements,
                ಅಂಶ_ct1
            );
        }
    );
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು TIn, ಬಗೆ_ಹೆಸರು TIdx>
ಸ್ಥಿರ ಶೂನ್ಯ ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_sycl(ggml_backend_sycl_context & ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * src0_d = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)src0->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ TIdx * src1_d = (ಸ್ಥಿರಾಂಕ TIdx *)src1->ದತ್ತಾಂಶ;

    GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS

    dpct::ಸರತಿ_ptr stream = ctx.stream();
    ಆಯ್ಕೆ (dst->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_sycl<TIn, TIdx, ದಶಮಾಂಶ>(
                src0_d, src1_d, (ಅಕ್ಷರ *)dst->ದತ್ತಾಂಶ,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                ಗಾತ್ರ(TIn), ಗಾತ್ರ(ದಶಮಾಂಶ),
                stream
            );
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F16:
            dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
            ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_sycl<TIn, TIdx, sycl::half>(
                src0_d, src1_d, (ಅಕ್ಷರ *)dst->ದತ್ತಾಂಶ,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                ಗಾತ್ರ(TIn), ಗಾತ್ರ(sycl::half),
                stream
            );
            ನಿಲ್ಲಿಸು;
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_HAS_BF16
        ಸ್ಥಿತಿ GGML_TYPE_BF16:
            ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_sycl<TIn, TIdx, sycl::ext::oneapi::bfloat16>(
                src0_d, src1_d, (ಅಕ್ಷರ *)dst->ದತ್ತಾಂಶ,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                ಗಾತ್ರ(TIn), ಗಾತ್ರ(sycl::ext::oneapi::bfloat16),
                stream
            );
            ನಿಲ್ಲಿಸು;
#ಆದರೆ_ಅಂತ್ಯ
        ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
            ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_sycl_q<TIn, TIdx, ಖಂಡ_q8_0, QK8_0, cpy_blck_f32_q8_0>(
                src0_d, src1_d, (ಖಂಡ_q8_0 *) dst->ದತ್ತಾಂಶ, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q1_0:
            ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_sycl_q<TIn, TIdx, ಖಂಡ_q1_0, QK1_0, cpy_blck_f32_q1_0>(
                src0_d, src1_d, (ಖಂಡ_q1_0 *) dst->ದತ್ತಾಂಶ, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q2_0:
            ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_sycl_q<TIn, TIdx, ಖಂಡ_q2_0, QK2_0, cpy_blck_f32_q2_0>(
                src0_d, src1_d, (ಖಂಡ_q2_0 *) dst->ದತ್ತಾಂಶ, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q5_1:
            ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_sycl_q<TIn, TIdx, ಖಂಡ_q5_1, QK5_1, cpy_blck_f32_q5_1>(
                src0_d, src1_d, (ಖಂಡ_q5_1 *) dst->ದತ್ತಾಂಶ, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q5_0:
            ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_sycl_q<TIn, TIdx, ಖಂಡ_q5_0, QK5_0, cpy_blck_f32_q5_0>(
                src0_d, src1_d, (ಖಂಡ_q5_0 *) dst->ದತ್ತಾಂಶ, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q4_1:
            ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_sycl_q<TIn, TIdx, ಖಂಡ_q4_1, QK4_1, cpy_blck_f32_q4_1>(
                src0_d, src1_d, (ಖಂಡ_q4_1 *) dst->ದತ್ತಾಂಶ, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
            ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_sycl_q<TIn, TIdx, ಖಂಡ_q4_0, QK4_0, cpy_blck_f32_q4_0>(
                src0_d, src1_d, (ಖಂಡ_q4_0 *) dst->ದತ್ತಾಂಶ, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_IQ4_NL:
            ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_sycl_q<TIn, TIdx, ಖಂಡ_iq4_nl, QK4_NL, cpy_blck_f32_iq4_nl>(
                src0_d, src1_d, (ಖಂಡ_iq4_nl *) dst->ದತ್ತಾಂಶ, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_MXFP4:
            ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_sycl_q<TIn, TIdx, ಖಂಡ_mxfp4, QK_MXFP4, cpy_blck_f32_mxfp4>(
                src0_d, src1_d, (ಖಂಡ_mxfp4 *) dst->ದತ್ತಾಂಶ, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_NVFP4:
            ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_sycl_q<TIn, TIdx, ಖಂಡ_nvfp4, QK_NVFP4, cpy_blck_f32_nvfp4>(
                src0_d, src1_d, (ಖಂಡ_nvfp4 *) dst->ದತ್ತಾಂಶ, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q2_K:
            ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_sycl_qk_host<TIn, TIdx, ಖಂಡ_q2_K, QK_K, quantize_ಅಡ್ಡಸಾಲು_q2_K_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q3_K:
            ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_sycl_qk_host<TIn, TIdx, ಖಂಡ_q3_K, QK_K, quantize_ಅಡ್ಡಸಾಲು_q3_K_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q4_K:
            ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_sycl_qk_host<TIn, TIdx, ಖಂಡ_q4_K, QK_K, quantize_ಅಡ್ಡಸಾಲು_q4_K_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q5_K:
            ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_sycl_qk_host<TIn, TIdx, ಖಂಡ_q5_K, QK_K, quantize_ಅಡ್ಡಸಾಲು_q5_K_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q6_K:
            ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_sycl_qk_host<TIn, TIdx, ಖಂಡ_q6_K, QK_K, quantize_ಅಡ್ಡಸಾಲು_q6_K_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_IQ2_XXS:
            ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_sycl_iq_host<TIn, TIdx, ಖಂಡ_iq2_xxs, QK_K, quantize_iq2_xxs>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_IQ2_XS:
            ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_sycl_iq_host<TIn, TIdx, ಖಂಡ_iq2_xs, QK_K, quantize_iq2_xs>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_IQ2_S:
            ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_sycl_iq_host<TIn, TIdx, ಖಂಡ_iq2_s, QK_K, quantize_iq2_s>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_IQ3_XXS:
            ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_sycl_qk_host<TIn, TIdx, ಖಂಡ_iq3_xxs, QK_K, quantize_ಅಡ್ಡಸಾಲು_iq3_xxs_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_IQ3_S:
            ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_sycl_qk_host<TIn, TIdx, ಖಂಡ_iq3_s, QK_K, quantize_ಅಡ್ಡಸಾಲು_iq3_s_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_IQ1_S:
            ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_sycl_iq_host<TIn, TIdx, ಖಂಡ_iq1_s, QK_K, quantize_iq1_s>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_IQ1_M:
            ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_sycl_iq_host<TIn, TIdx, ಖಂಡ_iq1_m, QK_K, quantize_iq1_m>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_IQ4_XS:
            ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_sycl_qk_host<TIn, TIdx, ಖಂಡ_iq4_xs, QK_K, quantize_ಅಡ್ಡಸಾಲು_iq4_xs_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            GGML_ABORT("Unsupported tensor type: src0 %s src1 %s dst %s", ggml_type_name(dst->src[0]->type),
                ggml_type_name(dst->src[1]->type), ggml_type_name(dst->type));
            ನಿಲ್ಲಿಸು;
    }
}

ಶೂನ್ಯ ggml_sycl_op_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->src[1]->type == GGML_TYPE_I64 || dst->src[1]->type == GGML_TYPE_I32);

    // dispatch on the index type (src1) and the source value type (src0)
    ಆದರೆ (src0->type == GGML_TYPE_F16) {
        ಆದರೆ (src1->type == GGML_TYPE_I64) {
            ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_sycl<sycl::half, int64_t>(ctx, src0, src1, dst);
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_sycl<sycl::half, int32_t>(ctx, src0, src1, dst);
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಆದರೆ (src1->type == GGML_TYPE_I64) {
            ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_sycl<ದಶಮಾಂಶ, int64_t>(ctx, src0, src1, dst);
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_sycl<ದಶಮಾಂಶ, int32_t>(ctx, src0, src1, dst);
        }
    }
}
