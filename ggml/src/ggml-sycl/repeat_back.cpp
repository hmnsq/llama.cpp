#अन्तर्भाव "repeat_back.hpp"

#अन्तर्भाव "common.hpp"

#define GGML_ASSERT_TENSOR_FITS_INT(t) \
    GGML_ASSERT((t)->ne[0] < INT_MAX && (t)->ne[1] < INT_MAX && (t)->ne[2] < INT_MAX && (t)->ne[3] < INT_MAX)

शून्य ggml_sycl_op_repeat_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    ध्रुव भिन्न * src0_dd = (ध्रुव भिन्न *) dst->src[0]->data;
    भिन्न *       dst_dd  = (भिन्न *) dst->data;

    GGML_ASSERT_TENSOR_FITS_INT(dst);
    GGML_ASSERT_TENSOR_FITS_INT(dst->src[0]);

    ध्रुव पूर्णाङ्क ne0 = dst->ne[0], ne1 = dst->ne[1], ne2 = dst->ne[2], ne3 = dst->ne[3];
    ध्रुव पूर्णाङ्क ne00 = dst->src[0]->ne[0], ne01 = dst->src[0]->ne[1], ne02 = dst->src[0]->ne[2],
              ne03 = dst->src[0]->ne[3];

    ध्रुव पूर्णाङ्क nr0 = ne00 / ne0;
    ध्रुव पूर्णाङ्क nr1 = ne01 / ne1;
    ध्रुव पूर्णाङ्क nr2 = ne02 / ne2;
    ध्रुव पूर्णाङ्क nr3 = ne03 / ne3;

    ध्रुव पूर्णाङ्क nb0 = dst->src[0]->nb[0];
    ध्रुव पूर्णाङ्क nb1 = dst->src[0]->nb[1];
    ध्रुव पूर्णाङ्क nb2 = dst->src[0]->nb[2];
    ध्रुव पूर्णाङ्क nb3 = dst->src[0]->nb[3];

    ध्रुव वर्ण * base = (ध्रुव वर्ण *) src0_dd;

    ध्रुव size_t  total      = (size_t) ne0 * ne1 * ne2 * ne3;
    constexpr पूर्णाङ्क BLOCK_SIZE = 256;
    ध्रुव पूर्णाङ्क     अङ्कः_blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

    ध्रुव भिन्न inv_ne0      = 1.0f / ne0;
    ध्रुव भिन्न inv_ne_01    = 1.0f / (ne0 * ne1);
    ध्रुव भिन्न inv_ne_012   = 1.0f / (ne0 * ne1 * ne2);
    ध्रुव पूर्णाङ्क   repeat_गणनम् = nr0 * nr1 * nr2 * nr3;

    queue_ptr stream = ctx.stream();

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(अङ्कः_blocks * BLOCK_SIZE), sycl::range<1>(BLOCK_SIZE)),
        [=](sycl::nd_item<1> item_ct1) {
            ध्रुव size_t i = item_ct1.get_global_linear_id();
            सचे (i >= total) {
                पटिनिवत्त;
            }

            ध्रुव पूर्णाङ्क i3 = (पूर्णाङ्क) (i * inv_ne_012);
            ध्रुव पूर्णाङ्क i2 = (पूर्णाङ्क) (i * inv_ne_01) - i3 * ne2;
            ध्रुव पूर्णाङ्क i1 = (पूर्णाङ्क) (i * inv_ne0) - (पूर्णाङ्क) (i * inv_ne_01) * ne1;
            ध्रुव पूर्णाङ्क i0 = i - (पूर्णाङ्क) (i * inv_ne0) * ne0;

            पूर्णाङ्क   j0 = 0, j1 = 0, j2 = 0, j3 = 0;
            भिन्न acc = 0.0f;

            अत्थाय (पूर्णाङ्क j = 0; j < repeat_गणनम्; ++j) {
                ध्रुव भिन्न * ptr = (ध्रुव भिन्न *) (base + (i0 + j0 * ne0) * nb0 + (i1 + j1 * ne1) * nb1 +
                    (i2 + j2 * ne2) * nb2 + (i3 + j3 * ne3) * nb3);
                acc += *ptr;

                पूर्णाङ्क carry = (++j0 >= nr0);
                j0 -= carry * nr0;
                carry = (carry && (++j1 >= nr1));
                j1 -= carry * nr1;
                carry = (carry && (++j2 >= nr2));
                j2 -= carry * nr2;
                j3 += carry;
            }
            dst_dd[i] = acc;
        });
}
