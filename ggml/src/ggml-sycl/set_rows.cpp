#ਅਨ੍ਤਰ੍ਭਾਵ "set_rows.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "cpy.hpp"

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-quants.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <vector>

ਨਾਮਾਕਾਸ਼ utils {
ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ constexpr ਤਾਰਕਿਕ is_arithmetic_v() {
    ਵਾਪਸ std::is_arithmetic_v<T> || std::is_same_v<T, sycl::half>
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_HAS_BF16
        || std::is_same_v<T, sycl::ext::oneapi::bfloat16>
#ਜੇ_ਅੰਤ
        ;
}
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ TIn, ਕਿਸਮ_ਨਾਂ TOut>
ਸਥਿਰ ਸਤਰਅੰਦਰ std::enable_if_t<utils::is_arithmetic_v<TIn>() && utils::is_arithmetic_v<TOut>(), ਖਾਲੀ>
convert (ਅਚਲ ਅੱਖਰ* src, ਅੱਖਰ* dst) {
    ਸਵੈਚਾਲਿਤ src_val = *reinterpret_cast<ਅਚਲ TIn*>(src);
    ਸਵੈਚਾਲਿਤ dst_val = sycl::vec<TIn, 1>(src_val).ਸਾਂਚਾ convert<TOut, sycl::rounding_ਬਹੁਲਕ::automatic>()[0];
   *reinterpret_cast<TOut*>(dst) = dst_val;
}

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_HAS_BF16
// sycl::vec::convert does not provide a half -> bfloat16 path, so route through float.
ਸਾਂਚਾ<>
ਸਤਰਅੰਦਰ ਖਾਲੀ convert<sycl::half, sycl::ext::oneapi::bfloat16>(ਅਚਲ ਅੱਖਰ* src, ਅੱਖਰ* dst) {
    ਅਚਲ ਦਸ਼ਮਲਵ tmp = sycl::vec<sycl::half, 1>(*reinterpret_cast<ਅਚਲ sycl::half*>(src))
                          .ਸਾਂਚਾ convert<ਦਸ਼ਮਲਵ, sycl::rounding_ਬਹੁਲਕ::automatic>()[0];
    *reinterpret_cast<sycl::ext::oneapi::bfloat16*>(dst) = sycl::ext::oneapi::bfloat16(tmp);
}
#ਜੇ_ਅੰਤ

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ TIn, ਕਿਸਮ_ਨਾਂ TIdx, ਕਿਸਮ_ਨਾਂ ਖੰਡ_Type, ਪੂਰਨ qk, cpy_kernel_t cpyblck>
ਸਥਿਰ ਖਾਲੀ ਸੈੱਟ_ਪੰਕਤੀਆਂ_sycl_q(ਅਚਲ ਅੱਖਰ * __restrict__ src0_d,
                            ਅਚਲ TIdx * __restrict__ src1_d,
                            ਖੰਡ_Type * __restrict__ dst_d,
                            // tensor dimensions src0 and src1
                            ਅਚਲ int64_t ne00,
                            ਅਚਲ int64_t ne01,
                            ਅਚਲ int64_t ne02,
                            ਅਚਲ int64_t ne03,
                            ਅਚਲ int64_t ne10,
                            ਅਚਲ int64_t ne11,
                            ਅਚਲ int64_t ne12,
                            ਅਚਲ int64_t ne13,
                            // strides for src0
                            ਅਚਲ size_t  nb00,
                            ਅਚਲ size_t  nb01,
                            ਅਚਲ size_t  nb02,
                            ਅਚਲ size_t  nb03,
                            // strides for src1
                            ਅਚਲ size_t  nb10,
                            ਅਚਲ size_t  nb11,
                            ਅਚਲ size_t  nb12,
                            ਅਚਲ size_t  nb13,
                            // strides for dst
                            ਅਚਲ size_t  nb1,
                            ਅਚਲ size_t  nb2,
                            ਅਚਲ size_t  nb3,
                            ਕਤਾਰ_ptr     stream) {
    ਅਚਲ int64_t ਕੁੱਲ_ਖੰਡਾਂ = (ne00 * ne01 * ne02 * ne03) / qk;
    constexpr ਪੂਰਨ ਖੰਡ_ਮਾਪ_3   = 256;
    ਅਚਲ int64_t grid_ਮਾਪ    = ceil_div(ਕੁੱਲ_ਖੰਡਾਂ, ਖੰਡ_ਮਾਪ_3);

    stream->parallel_for(sycl::nd_range<1>(grid_ਮਾਪ * ਖੰਡ_ਮਾਪ_3, ਖੰਡ_ਮਾਪ_3), [=](sycl::nd_ਵਸਤੂ<1> ਵਸਤੂ_ct1) {
        ਅਚਲ int64_t i = ਵਸਤੂ_ct1.get_global_ਰੇਖਿਕ_id();
        ਜੇ (i >= ਕੁੱਲ_ਖੰਡਾਂ) {
            ਵਾਪਸ;
        }
        ਅਚਲ int64_t i_ਅਧਾਰ      = i * qk;
        ਅਚਲ int64_t i03         = i_ਅਧਾਰ / (ne00 * ne01 * ne02);
        ਅਚਲ int64_t rem1        = i_ਅਧਾਰ - i03 * (ne00 * ne01 * ne02);
        ਅਚਲ int64_t i02         = rem1 / (ne00 * ne01);
        ਅਚਲ int64_t rem2        = rem1 - i02 * ne00 * ne01;
        ਅਚਲ int64_t i01         = rem2 / ne00;
        ਅਚਲ int64_t i00         = rem2 - i01 * ne00;
        ਅਚਲ int64_t i12         = i03 % ne12;
        ਅਚਲ int64_t i11         = i02 % ne11;
        ਅਚਲ int64_t i10         = i01;
        ਅਚਲ size_t  src_ਵਿੱਥ  = calculate_ਵਿੱਥ<3>({ nb01, nb02, nb03 }, { i01, i02, i03 });
        ਅਚਲ ਅੱਖਰ *  src_ਖੰਡ   = src0_d + src_ਵਿੱਥ + i00 * ਆਕਾਰ(TIn);
        ਅਚਲ size_t  src1_ਵਿੱਥ = calculate_ਵਿੱਥ<3>({ nb10, nb11, nb12 }, { i10, i11, i12 });
        ਅਚਲ int64_t dst_ਪੰਕਤੀ     = src1_d[src1_ਵਿੱਥ / ਆਕਾਰ(TIdx)];
        ਅਚਲ size_t  dst_ਵਿੱਥ =
            calculate_ਵਿੱਥ<3>({ nb1, nb2, nb3 }, { dst_ਪੰਕਤੀ, i02, i03 }) + (i00 / qk) * ਆਕਾਰ(ਖੰਡ_Type);
        ਅੱਖਰ * dst_ਖੰਡ = reinterpret_cast<ਅੱਖਰ *>(reinterpret_cast<ਅੱਖਰ *>(dst_d) + dst_ਵਿੱਥ);
        ਜੇ constexpr (std::is_same_v<TIn, ਦਸ਼ਮਲਵ>) {
            cpyblck(src_ਖੰਡ, dst_ਖੰਡ);
        } ਨਹੀਂਤਾਂ {
            ਦਸ਼ਮਲਵ src_ਖੰਡ_f32[qk];
            ਅਚਲ TIn * src_ਖੰਡ_t = reinterpret_cast<ਅਚਲ TIn *>(src_ਖੰਡ);
            ਲਈ (ਪੂਰਨ j = 0; j < qk; ++j) {
                src_ਖੰਡ_f32[j] = (ਦਸ਼ਮਲਵ) src_ਖੰਡ_t[j];
            }
            cpyblck(reinterpret_cast<ਅਚਲ ਅੱਖਰ *>(src_ਖੰਡ_f32), dst_ਖੰਡ);
        }
    });
    GGML_ਅਣਵਰਤਿਆ(ne10);
    GGML_ਅਣਵਰਤਿਆ(ne13);
    GGML_ਅਣਵਰਤਿਆ(nb00);
    GGML_ਅਣਵਰਤਿਆ(nb13);
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ ਖੰਡ_Type>
ਵਰਤੋਂ quantize_ਪੰਕਤੀ_qk_t = ਖਾਲੀ (*)(ਅਚਲ ਦਸ਼ਮਲਵ *, ਖੰਡ_Type *, int64_t);

ਵਰਤੋਂ quantize_ਪੰਕਤੀਆਂ_f_t = size_t (*)(ਅਚਲ ਦਸ਼ਮਲਵ *, ਖਾਲੀ *, int64_t, int64_t, ਅਚਲ ਦਸ਼ਮਲਵ *);

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ TIn, ਕਿਸਮ_ਨਾਂ TIdx, ਕਿਸਮ_ਨਾਂ ਖੰਡ_Type, ਪੂਰਨ qk, quantize_ਪੰਕਤੀ_qk_t<ਖੰਡ_Type> quantize_ਪੰਕਤੀ>
ਸਥਿਰ ਖਾਲੀ ਸੈੱਟ_ਪੰਕਤੀਆਂ_sycl_qk_host(
        ਅਚਲ ggml_tensor * src0,
        ਅਚਲ ggml_tensor * src1,
        ggml_tensor * dst,
        ਅਚਲ int64_t ne00,
        ਅਚਲ int64_t ne01,
        ਅਚਲ int64_t ne02,
        ਅਚਲ int64_t ne03,
        ਅਚਲ int64_t ne11,
        ਅਚਲ int64_t ne12,
        ਅਚਲ size_t nb01,
        ਅਚਲ size_t nb02,
        ਅਚਲ size_t nb03,
        ਅਚਲ size_t nb10,
        ਅਚਲ size_t nb11,
        ਅਚਲ size_t nb12,
        ਅਚਲ size_t nb1,
        ਅਚਲ size_t nb2,
        ਅਚਲ size_t nb3,
        ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ne00 % qk == 0);

    ਅਚਲ size_t src0_ਬਾਈਟਾਂ = ggml_nbytes(src0);
    ਅਚਲ size_t src1_ਬਾਈਟਾਂ = ggml_nbytes(src1);

    std::vector<ਅੱਖਰ> src0_host(src0_ਬਾਈਟਾਂ);
    std::vector<ਅੱਖਰ> src1_host(src1_ਬਾਈਟਾਂ);

    stream->ਸਮ੍ਰਿਤੀ_ਨਕਲ(src0_host.ਅੰਕੜੇ(), src0->ਅੰਕੜੇ, src0_ਬਾਈਟਾਂ);
    stream->ਸਮ੍ਰਿਤੀ_ਨਕਲ(src1_host.ਅੰਕੜੇ(), src1->ਅੰਕੜੇ, src1_ਬਾਈਟਾਂ);
    stream->wait();

    std::vector<ਦਸ਼ਮਲਵ> src_ਪੰਕਤੀ_f32(ne00);
    ਅਚਲ int64_t nblocks = ne00 / qk;
    std::vector<ਖੰਡ_Type> dst_ਪੰਕਤੀ_q(nblocks);

    ਲਈ (int64_t i03 = 0; i03 < ne03; ++i03) {
        ਲਈ (int64_t i02 = 0; i02 < ne02; ++i02) {
            ਲਈ (int64_t i01 = 0; i01 < ne01; ++i01) {
                ਅਚਲ int64_t i12 = i03 % ne12;
                ਅਚਲ int64_t i11 = i02 % ne11;
                ਅਚਲ int64_t i10 = i01;

                ਅਚਲ size_t src1_ਵਿੱਥ = calculate_ਵਿੱਥ<3>({ nb10, nb11, nb12 }, { i10, i11, i12 });
                ਅਚਲ int64_t dst_ਪੰਕਤੀ = *(ਅਚਲ TIdx *) (src1_host.ਅੰਕੜੇ() + src1_ਵਿੱਥ);

                ਅਚਲ size_t src0_ਪੰਕਤੀ_ਵਿੱਥ = calculate_ਵਿੱਥ<3>({ nb01, nb02, nb03 }, { i01, i02, i03 });
                ਅਚਲ TIn * src_ਪੰਕਤੀ = reinterpret_cast<ਅਚਲ TIn *>(src0_host.ਅੰਕੜੇ() + src0_ਪੰਕਤੀ_ਵਿੱਥ);

                ਲਈ (int64_t i00 = 0; i00 < ne00; ++i00) {
                    src_ਪੰਕਤੀ_f32[i00] = (ਦਸ਼ਮਲਵ) src_ਪੰਕਤੀ[i00];
                }

                quantize_ਪੰਕਤੀ(src_ਪੰਕਤੀ_f32.ਅੰਕੜੇ(), dst_ਪੰਕਤੀ_q.ਅੰਕੜੇ(), ne00);

                ਅਚਲ size_t dst_ਵਿੱਥ = calculate_ਵਿੱਥ<3>({ nb1, nb2, nb3 }, { dst_ਪੰਕਤੀ, i02, i03 });
                stream->ਸਮ੍ਰਿਤੀ_ਨਕਲ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + dst_ਵਿੱਥ, dst_ਪੰਕਤੀ_q.ਅੰਕੜੇ(), nblocks * ਆਕਾਰ(ਖੰਡ_Type));
                stream->wait();
            }
        }
    }
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ TIn, ਕਿਸਮ_ਨਾਂ TIdx, ਕਿਸਮ_ਨਾਂ ਖੰਡ_Type, ਪੂਰਨ qk, quantize_ਪੰਕਤੀਆਂ_f_t quantize_ਪੰਕਤੀਆਂ>
ਸਥਿਰ ਖਾਲੀ ਸੈੱਟ_ਪੰਕਤੀਆਂ_sycl_iq_host(
        ਅਚਲ ggml_tensor * src0,
        ਅਚਲ ggml_tensor * src1,
        ggml_tensor * dst,
        ਅਚਲ int64_t ne00,
        ਅਚਲ int64_t ne01,
        ਅਚਲ int64_t ne02,
        ਅਚਲ int64_t ne03,
        ਅਚਲ int64_t ne11,
        ਅਚਲ int64_t ne12,
        ਅਚਲ size_t nb01,
        ਅਚਲ size_t nb02,
        ਅਚਲ size_t nb03,
        ਅਚਲ size_t nb10,
        ਅਚਲ size_t nb11,
        ਅਚਲ size_t nb12,
        ਅਚਲ size_t nb1,
        ਅਚਲ size_t nb2,
        ਅਚਲ size_t nb3,
        ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ne00 % qk == 0);

    ਅਚਲ size_t src0_ਬਾਈਟਾਂ = ggml_nbytes(src0);
    ਅਚਲ size_t src1_ਬਾਈਟਾਂ = ggml_nbytes(src1);

    std::vector<ਅੱਖਰ> src0_host(src0_ਬਾਈਟਾਂ);
    std::vector<ਅੱਖਰ> src1_host(src1_ਬਾਈਟਾਂ);

    stream->ਸਮ੍ਰਿਤੀ_ਨਕਲ(src0_host.ਅੰਕੜੇ(), src0->ਅੰਕੜੇ, src0_ਬਾਈਟਾਂ);
    stream->ਸਮ੍ਰਿਤੀ_ਨਕਲ(src1_host.ਅੰਕੜੇ(), src1->ਅੰਕੜੇ, src1_ਬਾਈਟਾਂ);
    stream->wait();

    std::vector<ਦਸ਼ਮਲਵ> src_ਪੰਕਤੀ_f32(ne00);
    ਅਚਲ int64_t nblocks = ne00 / qk;
    std::vector<ਖੰਡ_Type> dst_ਪੰਕਤੀ_q(nblocks);

    ਲਈ (int64_t i03 = 0; i03 < ne03; ++i03) {
        ਲਈ (int64_t i02 = 0; i02 < ne02; ++i02) {
            ਲਈ (int64_t i01 = 0; i01 < ne01; ++i01) {
                ਅਚਲ int64_t i12 = i03 % ne12;
                ਅਚਲ int64_t i11 = i02 % ne11;
                ਅਚਲ int64_t i10 = i01;

                ਅਚਲ size_t src1_ਵਿੱਥ = calculate_ਵਿੱਥ<3>({ nb10, nb11, nb12 }, { i10, i11, i12 });
                ਅਚਲ int64_t dst_ਪੰਕਤੀ = *(ਅਚਲ TIdx *) (src1_host.ਅੰਕੜੇ() + src1_ਵਿੱਥ);

                ਅਚਲ size_t src0_ਪੰਕਤੀ_ਵਿੱਥ = calculate_ਵਿੱਥ<3>({ nb01, nb02, nb03 }, { i01, i02, i03 });
                ਅਚਲ TIn * src_ਪੰਕਤੀ = reinterpret_cast<ਅਚਲ TIn *>(src0_host.ਅੰਕੜੇ() + src0_ਪੰਕਤੀ_ਵਿੱਥ);

                ਲਈ (int64_t i00 = 0; i00 < ne00; ++i00) {
                    src_ਪੰਕਤੀ_f32[i00] = (ਦਸ਼ਮਲਵ) src_ਪੰਕਤੀ[i00];
                }

                quantize_ਪੰਕਤੀਆਂ(src_ਪੰਕਤੀ_f32.ਅੰਕੜੇ(), dst_ਪੰਕਤੀ_q.ਅੰਕੜੇ(), 1, ne00, ਖਾਲੀ_ਸੂਚਕ);

                ਅਚਲ size_t dst_ਵਿੱਥ = calculate_ਵਿੱਥ<3>({ nb1, nb2, nb3 }, { dst_ਪੰਕਤੀ, i02, i03 });
                stream->ਸਮ੍ਰਿਤੀ_ਨਕਲ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + dst_ਵਿੱਥ, dst_ਪੰਕਤੀ_q.ਅੰਕੜੇ(), nblocks * ਆਕਾਰ(ਖੰਡ_Type));
                stream->wait();
            }
        }
    }
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ TIn, ਕਿਸਮ_ਨਾਂ TIdx, ਕਿਸਮ_ਨਾਂ TOut>
ਸਥਿਰ ਖਾਲੀ k_ਸੈੱਟ_ਪੰਕਤੀਆਂ(
        ਅਚਲ ਅੱਖਰ * __restrict__ src0, ਅਚਲ TIdx * __restrict__ src1, ਅੱਖਰ * __restrict__ dst,
        ਅਚਲ int64_t ne00, ਅਚਲ int64_t ne01, ਅਚਲ int64_t ne02,
        ਅਚਲ int64_t ne11, ਅਚਲ int64_t ne12,
        ਅਚਲ size_t nb01, ਅਚਲ size_t nb02, ਅਚਲ size_t nb03,
        ਅਚਲ size_t nb10, ਅਚਲ size_t nb11, ਅਚਲ size_t nb12,
        ਅਚਲ size_t nb1, ਅਚਲ size_t nb2, ਅਚਲ size_t nb3,
        ਅਚਲ size_t src_type_ਮਾਪ, ਅਚਲ size_t dst_type_ਮਾਪ,
        ਅਚਲ int64_t ਕੁੱਲ_elements,
        ਅਚਲ sycl::nd_ਵਸਤੂ<1> & ਵਸਤੂ_ct1) {

    ਅਚਲ int64_t i = ਵਸਤੂ_ct1.get_global_ਰੇਖਿਕ_id();
    ਜੇ (i >= ਕੁੱਲ_elements) {
        ਵਾਪਸ;
    }

    ਅਚਲ int64_t i03 = i / (ne00 * ne01 * ne02);
    ਅਚਲ int64_t i02 = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    ਅਚਲ int64_t i01 = (i - i03 * ne00 * ne01 * ne02 - i02 * ne00 * ne01) / ne00;
    ਅਚਲ int64_t i00 = i - i03 * ne00 * ne01 * ne02 - i02 * ne00 * ne01 - i01 * ne00;

    ਅਚਲ int64_t i12 = i03 % ne12;
    ਅਚਲ int64_t i11 = i02 % ne11;
    ਅਚਲ int64_t i10 = i01;

    ਅਚਲ int64_t dst_ਪੰਕਤੀ = *(ਅਚਲ TIdx *)((ਅਚਲ ਅੱਖਰ *)src1 + calculate_ਵਿੱਥ<3>({nb10, nb11, nb12}, {i10, i11, i12}));

    ਅਚਲ ਅੱਖਰ * src0_ਪੰਕਤੀ = src0 + calculate_ਵਿੱਥ<3>({nb01, nb02, nb03}, {i01, i02, i03});
    ਅਚਲ ਅੱਖਰ * src_elem = src0_ਪੰਕਤੀ + i00 * src_type_ਮਾਪ;
    ਅੱਖਰ * dst_ਪੰਕਤੀ_ptr = dst + dst_ਪੰਕਤੀ*nb1 + i02*nb2 + i03*nb3;
    ਅੱਖਰ * dst_elem = dst_ਪੰਕਤੀ_ptr + i00 * dst_type_ਮਾਪ;

    convert<TIn, TOut>(src_elem, dst_elem);
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ TIn, ਕਿਸਮ_ਨਾਂ TIdx, ਕਿਸਮ_ਨਾਂ TOut>
ਸਥਿਰ ਖਾਲੀ ਸੈੱਟ_ਪੰਕਤੀਆਂ_sycl(
        ਅਚਲ ਅੱਖਰ * src0_d, ਅਚਲ TIdx * src1_d, ਅੱਖਰ * dst_d,
        ਅਚਲ int64_t ne00, ਅਚਲ int64_t ne01, ਅਚਲ int64_t ne02, ਅਚਲ int64_t ne03,
        ਅਚਲ int64_t ne11, ਅਚਲ int64_t ne12, ਅਚਲ size_t nb01, ਅਚਲ size_t nb02, ਅਚਲ size_t nb03,
        ਅਚਲ size_t nb10, ਅਚਲ size_t nb11, ਅਚਲ size_t nb12,
        ਅਚਲ size_t nb1, ਅਚਲ size_t nb2, ਅਚਲ size_t nb3,
        ਅਚਲ size_t src_type_ਮਾਪ, ਅਚਲ size_t dst_type_ਮਾਪ,
        ਕਤਾਰ_ptr stream) {

    ਅਚਲ int64_t ਕੁੱਲ_elements = ne00 * ne01 * ne02 * ne03;

    constexpr ਪੂਰਨ ਖੰਡ_ਮਾਪ_3 = 64;
    ਅਚਲ int64_t grid_ਮਾਪ = ceil_div(ਕੁੱਲ_elements, ਖੰਡ_ਮਾਪ_3);

    stream->parallel_for(
        sycl::nd_range<1>(grid_ਮਾਪ * ਖੰਡ_ਮਾਪ_3, ਖੰਡ_ਮਾਪ_3),
        [=](sycl::nd_ਵਸਤੂ<1> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
            k_ਸੈੱਟ_ਪੰਕਤੀਆਂ<TIn, TIdx, TOut>(
                src0_d, src1_d, dst_d,
                ne00, ne01, ne02,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                src_type_ਮਾਪ, dst_type_ਮਾਪ,
                ਕੁੱਲ_elements,
                ਵਸਤੂ_ct1
            );
        }
    );
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ TIn, ਕਿਸਮ_ਨਾਂ TIdx>
ਸਥਿਰ ਖਾਲੀ ਸੈੱਟ_ਪੰਕਤੀਆਂ_sycl(ggml_backend_sycl_context & ctx, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ggml_tensor * dst) {
    ਅਚਲ ਅੱਖਰ * src0_d = (ਅਚਲ ਅੱਖਰ *)src0->ਅੰਕੜੇ;
    ਅਚਲ TIdx * src1_d = (ਅਚਲ TIdx *)src1->ਅੰਕੜੇ;

    GGML_TENSOR_ਦੁਆਧਾਰੀ_OP_LOCALS

    dpct::ਕਤਾਰ_ptr stream = ctx.stream();
    ਚੋਣ (dst->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            ਸੈੱਟ_ਪੰਕਤੀਆਂ_sycl<TIn, TIdx, ਦਸ਼ਮਲਵ>(
                src0_d, src1_d, (ਅੱਖਰ *)dst->ਅੰਕੜੇ,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                ਆਕਾਰ(TIn), ਆਕਾਰ(ਦਸ਼ਮਲਵ),
                stream
            );
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F16:
            dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
            ਸੈੱਟ_ਪੰਕਤੀਆਂ_sycl<TIn, TIdx, sycl::half>(
                src0_d, src1_d, (ਅੱਖਰ *)dst->ਅੰਕੜੇ,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                ਆਕਾਰ(TIn), ਆਕਾਰ(sycl::half),
                stream
            );
            ਰੋਕੋ;
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_HAS_BF16
        ਹਾਲਤ GGML_TYPE_BF16:
            ਸੈੱਟ_ਪੰਕਤੀਆਂ_sycl<TIn, TIdx, sycl::ext::oneapi::bfloat16>(
                src0_d, src1_d, (ਅੱਖਰ *)dst->ਅੰਕੜੇ,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                ਆਕਾਰ(TIn), ਆਕਾਰ(sycl::ext::oneapi::bfloat16),
                stream
            );
            ਰੋਕੋ;
#ਜੇ_ਅੰਤ
        ਹਾਲਤ GGML_TYPE_Q8_0:
            ਸੈੱਟ_ਪੰਕਤੀਆਂ_sycl_q<TIn, TIdx, ਖੰਡ_q8_0, QK8_0, cpy_blck_f32_q8_0>(
                src0_d, src1_d, (ਖੰਡ_q8_0 *) dst->ਅੰਕੜੇ, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q1_0:
            ਸੈੱਟ_ਪੰਕਤੀਆਂ_sycl_q<TIn, TIdx, ਖੰਡ_q1_0, QK1_0, cpy_blck_f32_q1_0>(
                src0_d, src1_d, (ਖੰਡ_q1_0 *) dst->ਅੰਕੜੇ, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q2_0:
            ਸੈੱਟ_ਪੰਕਤੀਆਂ_sycl_q<TIn, TIdx, ਖੰਡ_q2_0, QK2_0, cpy_blck_f32_q2_0>(
                src0_d, src1_d, (ਖੰਡ_q2_0 *) dst->ਅੰਕੜੇ, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q5_1:
            ਸੈੱਟ_ਪੰਕਤੀਆਂ_sycl_q<TIn, TIdx, ਖੰਡ_q5_1, QK5_1, cpy_blck_f32_q5_1>(
                src0_d, src1_d, (ਖੰਡ_q5_1 *) dst->ਅੰਕੜੇ, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q5_0:
            ਸੈੱਟ_ਪੰਕਤੀਆਂ_sycl_q<TIn, TIdx, ਖੰਡ_q5_0, QK5_0, cpy_blck_f32_q5_0>(
                src0_d, src1_d, (ਖੰਡ_q5_0 *) dst->ਅੰਕੜੇ, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q4_1:
            ਸੈੱਟ_ਪੰਕਤੀਆਂ_sycl_q<TIn, TIdx, ਖੰਡ_q4_1, QK4_1, cpy_blck_f32_q4_1>(
                src0_d, src1_d, (ਖੰਡ_q4_1 *) dst->ਅੰਕੜੇ, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q4_0:
            ਸੈੱਟ_ਪੰਕਤੀਆਂ_sycl_q<TIn, TIdx, ਖੰਡ_q4_0, QK4_0, cpy_blck_f32_q4_0>(
                src0_d, src1_d, (ਖੰਡ_q4_0 *) dst->ਅੰਕੜੇ, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_IQ4_NL:
            ਸੈੱਟ_ਪੰਕਤੀਆਂ_sycl_q<TIn, TIdx, ਖੰਡ_iq4_nl, QK4_NL, cpy_blck_f32_iq4_nl>(
                src0_d, src1_d, (ਖੰਡ_iq4_nl *) dst->ਅੰਕੜੇ, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_MXFP4:
            ਸੈੱਟ_ਪੰਕਤੀਆਂ_sycl_q<TIn, TIdx, ਖੰਡ_mxfp4, QK_MXFP4, cpy_blck_f32_mxfp4>(
                src0_d, src1_d, (ਖੰਡ_mxfp4 *) dst->ਅੰਕੜੇ, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_NVFP4:
            ਸੈੱਟ_ਪੰਕਤੀਆਂ_sycl_q<TIn, TIdx, ਖੰਡ_nvfp4, QK_NVFP4, cpy_blck_f32_nvfp4>(
                src0_d, src1_d, (ਖੰਡ_nvfp4 *) dst->ਅੰਕੜੇ, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q2_K:
            ਸੈੱਟ_ਪੰਕਤੀਆਂ_sycl_qk_host<TIn, TIdx, ਖੰਡ_q2_K, QK_K, quantize_ਪੰਕਤੀ_q2_K_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q3_K:
            ਸੈੱਟ_ਪੰਕਤੀਆਂ_sycl_qk_host<TIn, TIdx, ਖੰਡ_q3_K, QK_K, quantize_ਪੰਕਤੀ_q3_K_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q4_K:
            ਸੈੱਟ_ਪੰਕਤੀਆਂ_sycl_qk_host<TIn, TIdx, ਖੰਡ_q4_K, QK_K, quantize_ਪੰਕਤੀ_q4_K_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q5_K:
            ਸੈੱਟ_ਪੰਕਤੀਆਂ_sycl_qk_host<TIn, TIdx, ਖੰਡ_q5_K, QK_K, quantize_ਪੰਕਤੀ_q5_K_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q6_K:
            ਸੈੱਟ_ਪੰਕਤੀਆਂ_sycl_qk_host<TIn, TIdx, ਖੰਡ_q6_K, QK_K, quantize_ਪੰਕਤੀ_q6_K_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_IQ2_XXS:
            ਸੈੱਟ_ਪੰਕਤੀਆਂ_sycl_iq_host<TIn, TIdx, ਖੰਡ_iq2_xxs, QK_K, quantize_iq2_xxs>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_IQ2_XS:
            ਸੈੱਟ_ਪੰਕਤੀਆਂ_sycl_iq_host<TIn, TIdx, ਖੰਡ_iq2_xs, QK_K, quantize_iq2_xs>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_IQ2_S:
            ਸੈੱਟ_ਪੰਕਤੀਆਂ_sycl_iq_host<TIn, TIdx, ਖੰਡ_iq2_s, QK_K, quantize_iq2_s>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_IQ3_XXS:
            ਸੈੱਟ_ਪੰਕਤੀਆਂ_sycl_qk_host<TIn, TIdx, ਖੰਡ_iq3_xxs, QK_K, quantize_ਪੰਕਤੀ_iq3_xxs_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_IQ3_S:
            ਸੈੱਟ_ਪੰਕਤੀਆਂ_sycl_qk_host<TIn, TIdx, ਖੰਡ_iq3_s, QK_K, quantize_ਪੰਕਤੀ_iq3_s_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_IQ1_S:
            ਸੈੱਟ_ਪੰਕਤੀਆਂ_sycl_iq_host<TIn, TIdx, ਖੰਡ_iq1_s, QK_K, quantize_iq1_s>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_IQ1_M:
            ਸੈੱਟ_ਪੰਕਤੀਆਂ_sycl_iq_host<TIn, TIdx, ਖੰਡ_iq1_m, QK_K, quantize_iq1_m>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_IQ4_XS:
            ਸੈੱਟ_ਪੰਕਤੀਆਂ_sycl_qk_host<TIn, TIdx, ਖੰਡ_iq4_xs, QK_K, quantize_ਪੰਕਤੀ_iq4_xs_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            ਰੋਕੋ;
        ਮੂਲ:
            GGML_ABORT("Unsupported tensor type: src0 %s src1 %s dst %s", ggml_type_name(dst->src[0]->type),
                ggml_type_name(dst->src[1]->type), ggml_type_name(dst->type));
            ਰੋਕੋ;
    }
}

ਖਾਲੀ ggml_sycl_op_ਸੈੱਟ_ਪੰਕਤੀਆਂ(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->src[1]->type == GGML_TYPE_I64 || dst->src[1]->type == GGML_TYPE_I32);

    // dispatch on the index type (src1) and the source value type (src0)
    ਜੇ (src0->type == GGML_TYPE_F16) {
        ਜੇ (src1->type == GGML_TYPE_I64) {
            ਸੈੱਟ_ਪੰਕਤੀਆਂ_sycl<sycl::half, int64_t>(ctx, src0, src1, dst);
        } ਨਹੀਂਤਾਂ {
            ਸੈੱਟ_ਪੰਕਤੀਆਂ_sycl<sycl::half, int32_t>(ctx, src0, src1, dst);
        }
    } ਨਹੀਂਤਾਂ {
        ਜੇ (src1->type == GGML_TYPE_I64) {
            ਸੈੱਟ_ਪੰਕਤੀਆਂ_sycl<ਦਸ਼ਮਲਵ, int64_t>(ctx, src0, src1, dst);
        } ਨਹੀਂਤਾਂ {
            ਸੈੱਟ_ਪੰਕਤੀਆਂ_sycl<ਦਸ਼ਮਲਵ, int32_t>(ctx, src0, src1, dst);
        }
    }
}
