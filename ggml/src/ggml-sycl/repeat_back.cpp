#அந்தர்பாவ "repeat_back.hpp"

#அந்தர்பாவ "common.hpp"

#define GGML_ASSERT_TENSOR_FITS_INT(t) \
    GGML_ASSERT((t)->ne[0] < INT_MAX && (t)->ne[1] < INT_MAX && (t)->ne[2] < INT_MAX && (t)->ne[3] < INT_MAX)

வெறுமை ggml_sycl_op_repeat_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    மாறிலி தசம * src0_dd = (மாறிலி தசம *) dst->src[0]->data;
    தசம *       dst_dd  = (தசம *) dst->data;

    GGML_ASSERT_TENSOR_FITS_INT(dst);
    GGML_ASSERT_TENSOR_FITS_INT(dst->src[0]);

    மாறிலி முழு ne0 = dst->ne[0], ne1 = dst->ne[1], ne2 = dst->ne[2], ne3 = dst->ne[3];
    மாறிலி முழு ne00 = dst->src[0]->ne[0], ne01 = dst->src[0]->ne[1], ne02 = dst->src[0]->ne[2],
              ne03 = dst->src[0]->ne[3];

    மாறிலி முழு nr0 = ne00 / ne0;
    மாறிலி முழு nr1 = ne01 / ne1;
    மாறிலி முழு nr2 = ne02 / ne2;
    மாறிலி முழு nr3 = ne03 / ne3;

    மாறிலி முழு nb0 = dst->src[0]->nb[0];
    மாறிலி முழு nb1 = dst->src[0]->nb[1];
    மாறிலி முழு nb2 = dst->src[0]->nb[2];
    மாறிலி முழு nb3 = dst->src[0]->nb[3];

    மாறிலி எழுத்து * base = (மாறிலி எழுத்து *) src0_dd;

    மாறிலி size_t  total      = (size_t) ne0 * ne1 * ne2 * ne3;
    constexpr முழு BLOCK_SIZE = 256;
    மாறிலி முழு     இலக்கம்_blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

    மாறிலி தசம inv_ne0      = 1.0f / ne0;
    மாறிலி தசம inv_ne_01    = 1.0f / (ne0 * ne1);
    மாறிலி தசம inv_ne_012   = 1.0f / (ne0 * ne1 * ne2);
    மாறிலி முழு   repeat_மொத்தம் = nr0 * nr1 * nr2 * nr3;

    queue_ptr stream = ctx.stream();

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(இலக்கம்_blocks * BLOCK_SIZE), sycl::range<1>(BLOCK_SIZE)),
        [=](sycl::nd_item<1> item_ct1) {
            மாறிலி size_t i = item_ct1.get_global_linear_id();
            எனில் (i >= total) {
                திரும்பு;
            }

            மாறிலி முழு i3 = (முழு) (i * inv_ne_012);
            மாறிலி முழு i2 = (முழு) (i * inv_ne_01) - i3 * ne2;
            மாறிலி முழு i1 = (முழு) (i * inv_ne0) - (முழு) (i * inv_ne_01) * ne1;
            மாறிலி முழு i0 = i - (முழு) (i * inv_ne0) * ne0;

            முழு   j0 = 0, j1 = 0, j2 = 0, j3 = 0;
            தசம acc = 0.0f;

            க்கு (முழு j = 0; j < repeat_மொத்தம்; ++j) {
                மாறிலி தசம * ptr = (மாறிலி தசம *) (base + (i0 + j0 * ne0) * nb0 + (i1 + j1 * ne1) * nb1 +
                    (i2 + j2 * ne2) * nb2 + (i3 + j3 * ne3) * nb3);
                acc += *ptr;

                முழு carry = (++j0 >= nr0);
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
