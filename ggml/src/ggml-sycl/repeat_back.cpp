#ଅନ୍ତର୍ଭାଵ "repeat_back.hpp"

#ଅନ୍ତର୍ଭାଵ "common.hpp"

#define GGML_ASSERT_TENSOR_FITS_INT(t) \
    GGML_ASSERT((t)->ne[0] < INT_MAX && (t)->ne[1] < INT_MAX && (t)->ne[2] < INT_MAX && (t)->ne[3] < INT_MAX)

ଶୂନ୍ୟ ggml_sycl_op_repeat_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    ଅଚଳ ଦଶମିକ * src0_dd = (ଅଚଳ ଦଶମିକ *) dst->src[0]->data;
    ଦଶମିକ *       dst_dd  = (ଦଶମିକ *) dst->data;

    GGML_ASSERT_TENSOR_FITS_INT(dst);
    GGML_ASSERT_TENSOR_FITS_INT(dst->src[0]);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne0 = dst->ne[0], ne1 = dst->ne[1], ne2 = dst->ne[2], ne3 = dst->ne[3];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00 = dst->src[0]->ne[0], ne01 = dst->src[0]->ne[1], ne02 = dst->src[0]->ne[2],
              ne03 = dst->src[0]->ne[3];

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr0 = ne00 / ne0;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr1 = ne01 / ne1;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr2 = ne02 / ne2;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr3 = ne03 / ne3;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb0 = dst->src[0]->nb[0];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb1 = dst->src[0]->nb[1];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb2 = dst->src[0]->nb[2];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb3 = dst->src[0]->nb[3];

    ଅଚଳ ଅକ୍ଷର * base = (ଅଚଳ ଅକ୍ଷର *) src0_dd;

    ଅଚଳ size_t  total      = (size_t) ne0 * ne1 * ne2 * ne3;
    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା BLOCK_SIZE = 256;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା     ଅଙ୍କ_blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

    ଅଚଳ ଦଶମିକ inv_ne0      = 1.0f / ne0;
    ଅଚଳ ଦଶମିକ inv_ne_01    = 1.0f / (ne0 * ne1);
    ଅଚଳ ଦଶମିକ inv_ne_012   = 1.0f / (ne0 * ne1 * ne2);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା   repeat_ଗଣନା = nr0 * nr1 * nr2 * nr3;

    queue_ptr stream = ctx.stream();

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(ଅଙ୍କ_blocks * BLOCK_SIZE), sycl::range<1>(BLOCK_SIZE)),
        [=](sycl::nd_item<1> item_ct1) {
            ଅଚଳ size_t i = item_ct1.get_global_linear_id();
            ଯଦି (i >= total) {
                ଫେରନ୍ତୁ;
            }

            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i3 = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) (i * inv_ne_012);
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i2 = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) (i * inv_ne_01) - i3 * ne2;
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1 = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) (i * inv_ne0) - (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) (i * inv_ne_01) * ne1;
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i0 = i - (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) (i * inv_ne0) * ne0;

            ପୂର୍ଣ୍ଣସଂଖ୍ୟା   j0 = 0, j1 = 0, j2 = 0, j3 = 0;
            ଦଶମିକ acc = 0.0f;

            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < repeat_ଗଣନା; ++j) {
                ଅଚଳ ଦଶମିକ * ptr = (ଅଚଳ ଦଶମିକ *) (base + (i0 + j0 * ne0) * nb0 + (i1 + j1 * ne1) * nb1 +
                    (i2 + j2 * ne2) * nb2 + (i3 + j3 * ne3) * nb3);
                acc += *ptr;

                ପୂର୍ଣ୍ଣସଂଖ୍ୟା carry = (++j0 >= nr0);
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
