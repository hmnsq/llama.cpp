#অন্তর্ভাব "repeat_back.hpp"

#অন্তর্ভাব "common.hpp"

#define GGML_ASSERT_TENSOR_FITS_INT(t) \
    GGML_ASSERT((t)->ne[0] < INT_MAX && (t)->ne[1] < INT_MAX && (t)->ne[2] < INT_MAX && (t)->ne[3] < INT_MAX)

শূন্য ggml_sycl_op_repeat_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    ধ্রুবক দশমিক * src0_dd = (ধ্রুবক দশমিক *) dst->src[0]->তথ্য;
    দশমিক *       dst_dd  = (দশমিক *) dst->তথ্য;

    GGML_ASSERT_TENSOR_FITS_INT(dst);
    GGML_ASSERT_TENSOR_FITS_INT(dst->src[0]);

    ধ্রুবক পূর্ণসংখ্যা ne0 = dst->ne[0], ne1 = dst->ne[1], ne2 = dst->ne[2], ne3 = dst->ne[3];
    ধ্রুবক পূর্ণসংখ্যা ne00 = dst->src[0]->ne[0], ne01 = dst->src[0]->ne[1], ne02 = dst->src[0]->ne[2],
              ne03 = dst->src[0]->ne[3];

    ধ্রুবক পূর্ণসংখ্যা nr0 = ne00 / ne0;
    ধ্রুবক পূর্ণসংখ্যা nr1 = ne01 / ne1;
    ধ্রুবক পূর্ণসংখ্যা nr2 = ne02 / ne2;
    ধ্রুবক পূর্ণসংখ্যা nr3 = ne03 / ne3;

    ধ্রুবক পূর্ণসংখ্যা nb0 = dst->src[0]->nb[0];
    ধ্রুবক পূর্ণসংখ্যা nb1 = dst->src[0]->nb[1];
    ধ্রুবক পূর্ণসংখ্যা nb2 = dst->src[0]->nb[2];
    ধ্রুবক পূর্ণসংখ্যা nb3 = dst->src[0]->nb[3];

    ধ্রুবক অক্ষর * base = (ধ্রুবক অক্ষর *) src0_dd;

    ধ্রুবক size_t  যোগ      = (size_t) ne0 * ne1 * ne2 * ne3;
    constexpr পূর্ণসংখ্যা BLOCK_আকার = 256;
    ধ্রুবক পূর্ণসংখ্যা     অঙ্ক_blocks = (যোগ + BLOCK_আকার - 1) / BLOCK_আকার;

    ধ্রুবক দশমিক inv_ne0      = 1.0f / ne0;
    ধ্রুবক দশমিক inv_ne_01    = 1.0f / (ne0 * ne1);
    ধ্রুবক দশমিক inv_ne_012   = 1.0f / (ne0 * ne1 * ne2);
    ধ্রুবক পূর্ণসংখ্যা   repeat_গুনতি = nr0 * nr1 * nr2 * nr3;

    queue_ptr stream = ctx.stream();

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(অঙ্ক_blocks * BLOCK_আকার), sycl::range<1>(BLOCK_আকার)),
        [=](sycl::nd_item<1> item_ct1) {
            ধ্রুবক size_t i = item_ct1.get_global_linear_id();
            যদি (i >= যোগ) {
                ফেরত;
            }

            ধ্রুবক পূর্ণসংখ্যা i3 = (পূর্ণসংখ্যা) (i * inv_ne_012);
            ধ্রুবক পূর্ণসংখ্যা i2 = (পূর্ণসংখ্যা) (i * inv_ne_01) - i3 * ne2;
            ধ্রুবক পূর্ণসংখ্যা i1 = (পূর্ণসংখ্যা) (i * inv_ne0) - (পূর্ণসংখ্যা) (i * inv_ne_01) * ne1;
            ধ্রুবক পূর্ণসংখ্যা i0 = i - (পূর্ণসংখ্যা) (i * inv_ne0) * ne0;

            পূর্ণসংখ্যা   j0 = 0, j1 = 0, j2 = 0, j3 = 0;
            দশমিক acc = 0.0f;

            জন্য (পূর্ণসংখ্যা j = 0; j < repeat_গুনতি; ++j) {
                ধ্রুবক দশমিক * ptr = (ধ্রুবক দশমিক *) (base + (i0 + j0 * ne0) * nb0 + (i1 + j1 * ne1) * nb1 +
                    (i2 + j2 * ne2) * nb2 + (i3 + j3 * ne3) * nb3);
                acc += *ptr;

                পূর্ণসংখ্যা carry = (++j0 >= nr0);
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
