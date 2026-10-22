#ਅਨ੍ਤਰ੍ਭਾਵ "repeat_back.hpp"

#ਅਨ੍ਤਰ੍ਭਾਵ "common.hpp"

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_ASSERT_TENSOR_ਸਮਾਉਂਦਾ_INT(t) \
    GGML_ASSERT((t)->ne[0] < ਵੱਧ_ਤੋਂ_ਵੱਧ_ਅੰਕ && (t)->ne[1] < ਵੱਧ_ਤੋਂ_ਵੱਧ_ਅੰਕ && (t)->ne[2] < ਵੱਧ_ਤੋਂ_ਵੱਧ_ਅੰਕ && (t)->ne[3] < ਵੱਧ_ਤੋਂ_ਵੱਧ_ਅੰਕ)

ਖਾਲੀ ggml_sycl_op_repeat_ਪਿੱਠ(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    ਅਚਲ ਦਸ਼ਮਲਵ * src0_dd = (ਅਚਲ ਦਸ਼ਮਲਵ *) dst->src[0]->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ *       dst_dd  = (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ;

    GGML_ASSERT_TENSOR_ਸਮਾਉਂਦਾ_INT(dst);
    GGML_ASSERT_TENSOR_ਸਮਾਉਂਦਾ_INT(dst->src[0]);

    ਅਚਲ ਪੂਰਨ ne0 = dst->ne[0], ne1 = dst->ne[1], ne2 = dst->ne[2], ne3 = dst->ne[3];
    ਅਚਲ ਪੂਰਨ ne00 = dst->src[0]->ne[0], ne01 = dst->src[0]->ne[1], ne02 = dst->src[0]->ne[2],
              ne03 = dst->src[0]->ne[3];

    ਅਚਲ ਪੂਰਨ nr0 = ne00 / ne0;
    ਅਚਲ ਪੂਰਨ nr1 = ne01 / ne1;
    ਅਚਲ ਪੂਰਨ nr2 = ne02 / ne2;
    ਅਚਲ ਪੂਰਨ nr3 = ne03 / ne3;

    ਅਚਲ ਪੂਰਨ nb0 = dst->src[0]->nb[0];
    ਅਚਲ ਪੂਰਨ nb1 = dst->src[0]->nb[1];
    ਅਚਲ ਪੂਰਨ nb2 = dst->src[0]->nb[2];
    ਅਚਲ ਪੂਰਨ nb3 = dst->src[0]->nb[3];

    ਅਚਲ ਅੱਖਰ * ਅਧਾਰ = (ਅਚਲ ਅੱਖਰ *) src0_dd;

    ਅਚਲ size_t  ਕੁੱਲ      = (size_t) ne0 * ne1 * ne2 * ne3;
    constexpr ਪੂਰਨ ਖੰਡ_ਮਾਪ_2 = 256;
    ਅਚਲ ਪੂਰਨ     ਅੰਕ_ਖੰਡਾਂ = (ਕੁੱਲ + ਖੰਡ_ਮਾਪ_2 - 1) / ਖੰਡ_ਮਾਪ_2;

    ਅਚਲ ਦਸ਼ਮਲਵ inv_ne0      = 1.0f / ne0;
    ਅਚਲ ਦਸ਼ਮਲਵ inv_ne_01    = 1.0f / (ne0 * ne1);
    ਅਚਲ ਦਸ਼ਮਲਵ inv_ne_012   = 1.0f / (ne0 * ne1 * ne2);
    ਅਚਲ ਪੂਰਨ   repeat_ਗਿਣਤੀ = nr0 * nr1 * nr2 * nr3;

    ਕਤਾਰ_ptr stream = ctx.stream();

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(ਅੰਕ_ਖੰਡਾਂ * ਖੰਡ_ਮਾਪ_2), sycl::range<1>(ਖੰਡ_ਮਾਪ_2)),
        [=](sycl::nd_ਵਸਤੂ<1> ਵਸਤੂ_ct1) {
            ਅਚਲ size_t i = ਵਸਤੂ_ct1.get_global_ਰੇਖਿਕ_id();
            ਜੇ (i >= ਕੁੱਲ) {
                ਵਾਪਸ;
            }

            ਅਚਲ ਪੂਰਨ i3 = (ਪੂਰਨ) (i * inv_ne_012);
            ਅਚਲ ਪੂਰਨ i2 = (ਪੂਰਨ) (i * inv_ne_01) - i3 * ne2;
            ਅਚਲ ਪੂਰਨ i1 = (ਪੂਰਨ) (i * inv_ne0) - (ਪੂਰਨ) (i * inv_ne_01) * ne1;
            ਅਚਲ ਪੂਰਨ i0 = i - (ਪੂਰਨ) (i * inv_ne0) * ne0;

            ਪੂਰਨ   j0 = 0, j1 = 0, j2 = 0, j3 = 0;
            ਦਸ਼ਮਲਵ acc = 0.0f;

            ਲਈ (ਪੂਰਨ j = 0; j < repeat_ਗਿਣਤੀ; ++j) {
                ਅਚਲ ਦਸ਼ਮਲਵ * ptr = (ਅਚਲ ਦਸ਼ਮਲਵ *) (ਅਧਾਰ + (i0 + j0 * ne0) * nb0 + (i1 + j1 * ne1) * nb1 +
                    (i2 + j2 * ne2) * nb2 + (i3 + j3 * ne3) * nb3);
                acc += *ptr;

                ਪੂਰਨ ਹਾਸਲ = (++j0 >= nr0);
                j0 -= ਹਾਸਲ * nr0;
                ਹਾਸਲ = (ਹਾਸਲ && (++j1 >= nr1));
                j1 -= ਹਾਸਲ * nr1;
                ਹਾਸਲ = (ਹਾਸਲ && (++j2 >= nr2));
                j2 -= ਹਾਸਲ * nr2;
                j3 += ਹਾਸਲ;
            }
            dst_dd[i] = acc;
        });
}
