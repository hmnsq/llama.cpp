#شامل "repeat_back.hpp"

#شامل "common.hpp"

#define GGML_ASSERT_TENSOR_FITS_INT(t) \
    GGML_ASSERT((t)->ne[0] < INT_MAX && (t)->ne[1] < INT_MAX && (t)->ne[2] < INT_MAX && (t)->ne[3] < INT_MAX)

تهی ggml_sycl_op_repeat_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    ثابت اعشاری * src0_dd = (ثابت اعشاری *) dst->src[0]->data;
    اعشاری *       dst_dd  = (اعشاری *) dst->data;

    GGML_ASSERT_TENSOR_FITS_INT(dst);
    GGML_ASSERT_TENSOR_FITS_INT(dst->src[0]);

    ثابت صحیح ne0 = dst->ne[0], ne1 = dst->ne[1], ne2 = dst->ne[2], ne3 = dst->ne[3];
    ثابت صحیح ne00 = dst->src[0]->ne[0], ne01 = dst->src[0]->ne[1], ne02 = dst->src[0]->ne[2],
              ne03 = dst->src[0]->ne[3];

    ثابت صحیح nr0 = ne00 / ne0;
    ثابت صحیح nr1 = ne01 / ne1;
    ثابت صحیح nr2 = ne02 / ne2;
    ثابت صحیح nr3 = ne03 / ne3;

    ثابت صحیح nb0 = dst->src[0]->nb[0];
    ثابت صحیح nb1 = dst->src[0]->nb[1];
    ثابت صحیح nb2 = dst->src[0]->nb[2];
    ثابت صحیح nb3 = dst->src[0]->nb[3];

    ثابت نویسه * base = (ثابت نویسه *) src0_dd;

    ثابت size_t  total      = (size_t) ne0 * ne1 * ne2 * ne3;
    constexpr صحیح BLOCK_SIZE = 256;
    ثابت صحیح     رقم_blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

    ثابت اعشاری inv_ne0      = 1.0f / ne0;
    ثابت اعشاری inv_ne_01    = 1.0f / (ne0 * ne1);
    ثابت اعشاری inv_ne_012   = 1.0f / (ne0 * ne1 * ne2);
    ثابت صحیح   repeat_شمار = nr0 * nr1 * nr2 * nr3;

    queue_ptr stream = ctx.stream();

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(رقم_blocks * BLOCK_SIZE), sycl::range<1>(BLOCK_SIZE)),
        [=](sycl::nd_item<1> item_ct1) {
            ثابت size_t i = item_ct1.get_global_linear_id();
            اگر (i >= total) {
                بازگشت;
            }

            ثابت صحیح i3 = (صحیح) (i * inv_ne_012);
            ثابت صحیح i2 = (صحیح) (i * inv_ne_01) - i3 * ne2;
            ثابت صحیح i1 = (صحیح) (i * inv_ne0) - (صحیح) (i * inv_ne_01) * ne1;
            ثابت صحیح i0 = i - (صحیح) (i * inv_ne0) * ne0;

            صحیح   j0 = 0, j1 = 0, j2 = 0, j3 = 0;
            اعشاری acc = 0.0f;

            برای (صحیح j = 0; j < repeat_شمار; ++j) {
                ثابت اعشاری * ptr = (ثابت اعشاری *) (base + (i0 + j0 * ne0) * nb0 + (i1 + j1 * ne1) * nb1 +
                    (i2 + j2 * ne2) * nb2 + (i3 + j3 * ne3) * nb3);
                acc += *ptr;

                صحیح carry = (++j0 >= nr0);
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
