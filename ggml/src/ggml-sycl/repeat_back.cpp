#অন্তর্ভাব "repeat_back.hpp"

#অন্তর্ভাব "common.hpp"

#define GGML_ASSERT_TENSOR_FITS_INT(t) \
    GGML_ASSERT((t)->ne[0] < INT_MAX && (t)->ne[1] < INT_MAX && (t)->ne[2] < INT_MAX && (t)->ne[3] < INT_MAX)

শূন্য ggml_sycl_op_repeat_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    ধ্রুব ভিন্ন * src0_dd = (ধ্রুব ভিন্ন *) dst->src[0]->data;
    ভিন্ন *       dst_dd  = (ভিন্ন *) dst->data;

    GGML_ASSERT_TENSOR_FITS_INT(dst);
    GGML_ASSERT_TENSOR_FITS_INT(dst->src[0]);

    ধ্রুব পূর্ণাঙ্ক ne0 = dst->ne[0], ne1 = dst->ne[1], ne2 = dst->ne[2], ne3 = dst->ne[3];
    ধ্রুব পূর্ণাঙ্ক ne00 = dst->src[0]->ne[0], ne01 = dst->src[0]->ne[1], ne02 = dst->src[0]->ne[2],
              ne03 = dst->src[0]->ne[3];

    ধ্রুব পূর্ণাঙ্ক nr0 = ne00 / ne0;
    ধ্রুব পূর্ণাঙ্ক nr1 = ne01 / ne1;
    ধ্রুব পূর্ণাঙ্ক nr2 = ne02 / ne2;
    ধ্রুব পূর্ণাঙ্ক nr3 = ne03 / ne3;

    ধ্রুব পূর্ণাঙ্ক nb0 = dst->src[0]->nb[0];
    ধ্রুব পূর্ণাঙ্ক nb1 = dst->src[0]->nb[1];
    ধ্রুব পূর্ণাঙ্ক nb2 = dst->src[0]->nb[2];
    ধ্রুব পূর্ণাঙ্ক nb3 = dst->src[0]->nb[3];

    ধ্রুব বর্ণ * base = (ধ্রুব বর্ণ *) src0_dd;

    ধ্রুব size_t  total      = (size_t) ne0 * ne1 * ne2 * ne3;
    constexpr পূর্ণাঙ্ক BLOCK_SIZE = 256;
    ধ্রুব পূর্ণাঙ্ক     অঙ্ক_blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

    ধ্রুব ভিন্ন inv_ne0      = 1.0f / ne0;
    ধ্রুব ভিন্ন inv_ne_01    = 1.0f / (ne0 * ne1);
    ধ্রুব ভিন্ন inv_ne_012   = 1.0f / (ne0 * ne1 * ne2);
    ধ্রুব পূর্ণাঙ্ক   repeat_গুনতি = nr0 * nr1 * nr2 * nr3;

    queue_ptr stream = ctx.stream();

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(অঙ্ক_blocks * BLOCK_SIZE), sycl::range<1>(BLOCK_SIZE)),
        [=](sycl::nd_item<1> item_ct1) {
            ধ্রুব size_t i = item_ct1.get_global_linear_id();
            যদি (i >= total) {
                প্রত্যাগম;
            }

            ধ্রুব পূর্ণাঙ্ক i3 = (পূর্ণাঙ্ক) (i * inv_ne_012);
            ধ্রুব পূর্ণাঙ্ক i2 = (পূর্ণাঙ্ক) (i * inv_ne_01) - i3 * ne2;
            ধ্রুব পূর্ণাঙ্ক i1 = (পূর্ণাঙ্ক) (i * inv_ne0) - (পূর্ণাঙ্ক) (i * inv_ne_01) * ne1;
            ধ্রুব পূর্ণাঙ্ক i0 = i - (পূর্ণাঙ্ক) (i * inv_ne0) * ne0;

            পূর্ণাঙ্ক   j0 = 0, j1 = 0, j2 = 0, j3 = 0;
            ভিন্ন acc = 0.0f;

            প্রতি (পূর্ণাঙ্ক j = 0; j < repeat_গুনতি; ++j) {
                ধ্রুব ভিন্ন * ptr = (ধ্রুব ভিন্ন *) (base + (i0 + j0 * ne0) * nb0 + (i1 + j1 * ne1) * nb1 +
                    (i2 + j2 * ne2) * nb2 + (i3 + j3 * ne3) * nb3);
                acc += *ptr;

                পূর্ণাঙ্ক carry = (++j0 >= nr0);
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
