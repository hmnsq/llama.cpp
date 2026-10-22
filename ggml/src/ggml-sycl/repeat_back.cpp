#અન્તર્ભાવ "repeat_back.hpp"

#અન્તર્ભાવ "common.hpp"

#define GGML_ASSERT_TENSOR_FITS_INT(t) \
    GGML_ASSERT((t)->ne[0] < INT_MAX && (t)->ne[1] < INT_MAX && (t)->ne[2] < INT_MAX && (t)->ne[3] < INT_MAX)

શૂન્ય ggml_sycl_op_repeat_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    અચલ દશાંશ * src0_dd = (અચલ દશાંશ *) dst->src[0]->data;
    દશાંશ *       dst_dd  = (દશાંશ *) dst->data;

    GGML_ASSERT_TENSOR_FITS_INT(dst);
    GGML_ASSERT_TENSOR_FITS_INT(dst->src[0]);

    અચલ પૂર્ણાંક ne0 = dst->ne[0], ne1 = dst->ne[1], ne2 = dst->ne[2], ne3 = dst->ne[3];
    અચલ પૂર્ણાંક ne00 = dst->src[0]->ne[0], ne01 = dst->src[0]->ne[1], ne02 = dst->src[0]->ne[2],
              ne03 = dst->src[0]->ne[3];

    અચલ પૂર્ણાંક nr0 = ne00 / ne0;
    અચલ પૂર્ણાંક nr1 = ne01 / ne1;
    અચલ પૂર્ણાંક nr2 = ne02 / ne2;
    અચલ પૂર્ણાંક nr3 = ne03 / ne3;

    અચલ પૂર્ણાંક nb0 = dst->src[0]->nb[0];
    અચલ પૂર્ણાંક nb1 = dst->src[0]->nb[1];
    અચલ પૂર્ણાંક nb2 = dst->src[0]->nb[2];
    અચલ પૂર્ણાંક nb3 = dst->src[0]->nb[3];

    અચલ અક્ષર * base = (અચલ અક્ષર *) src0_dd;

    અચલ size_t  total      = (size_t) ne0 * ne1 * ne2 * ne3;
    constexpr પૂર્ણાંક BLOCK_SIZE = 256;
    અચલ પૂર્ણાંક     અંક_blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

    અચલ દશાંશ inv_ne0      = 1.0f / ne0;
    અચલ દશાંશ inv_ne_01    = 1.0f / (ne0 * ne1);
    અચલ દશાંશ inv_ne_012   = 1.0f / (ne0 * ne1 * ne2);
    અચલ પૂર્ણાંક   repeat_ગણતરી = nr0 * nr1 * nr2 * nr3;

    queue_ptr stream = ctx.stream();

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(અંક_blocks * BLOCK_SIZE), sycl::range<1>(BLOCK_SIZE)),
        [=](sycl::nd_item<1> item_ct1) {
            અચલ size_t i = item_ct1.get_global_linear_id();
            જો (i >= total) {
                પાછા;
            }

            અચલ પૂર્ણાંક i3 = (પૂર્ણાંક) (i * inv_ne_012);
            અચલ પૂર્ણાંક i2 = (પૂર્ણાંક) (i * inv_ne_01) - i3 * ne2;
            અચલ પૂર્ણાંક i1 = (પૂર્ણાંક) (i * inv_ne0) - (પૂર્ણાંક) (i * inv_ne_01) * ne1;
            અચલ પૂર્ણાંક i0 = i - (પૂર્ણાંક) (i * inv_ne0) * ne0;

            પૂર્ણાંક   j0 = 0, j1 = 0, j2 = 0, j3 = 0;
            દશાંશ acc = 0.0f;

            માટે (પૂર્ણાંક j = 0; j < repeat_ગણતરી; ++j) {
                અચલ દશાંશ * ptr = (અચલ દશાંશ *) (base + (i0 + j0 * ne0) * nb0 + (i1 + j1 * ne1) * nb1 +
                    (i2 + j2 * ne2) * nb2 + (i3 + j3 * ne3) * nb3);
                acc += *ptr;

                પૂર્ણાંક carry = (++j0 >= nr0);
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
