#include "repeat_back.hpp"

#include "common.hpp"

#define GGML_ASSERT_TENSOR_FITS_INT(t) \
    GGML_ASSERT((t)->ne[0] < INT_MAX && (t)->ne[1] < INT_MAX && (t)->ne[2] < INT_MAX && (t)->ne[3] < INT_MAX)

void ggml_sycl_op_repeat_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const float * src0_dd = (const float *) dst->src[0]->data;
    float *       dst_dd  = (float *) dst->data;

    GGML_ASSERT_TENSOR_FITS_INT(dst);
    GGML_ASSERT_TENSOR_FITS_INT(dst->src[0]);

    const පූර්ණ_සංඛ්‍යා ne0 = dst->ne[0], ne1 = dst->ne[1], ne2 = dst->ne[2], ne3 = dst->ne[3];
    const පූර්ණ_සංඛ්‍යා ne00 = dst->src[0]->ne[0], ne01 = dst->src[0]->ne[1], ne02 = dst->src[0]->ne[2],
              ne03 = dst->src[0]->ne[3];

    const පූර්ණ_සංඛ්‍යා nr0 = ne00 / ne0;
    const පූර්ණ_සංඛ්‍යා nr1 = ne01 / ne1;
    const පූර්ණ_සංඛ්‍යා nr2 = ne02 / ne2;
    const පූර්ණ_සංඛ්‍යා nr3 = ne03 / ne3;

    const පූර්ණ_සංඛ්‍යා nb0 = dst->src[0]->nb[0];
    const පූර්ණ_සංඛ්‍යා nb1 = dst->src[0]->nb[1];
    const පූර්ණ_සංඛ්‍යා nb2 = dst->src[0]->nb[2];
    const පූර්ණ_සංඛ්‍යා nb3 = dst->src[0]->nb[3];

    const var_na * base = (const var_na *) src0_dd;

    const size_t  total      = (size_t) ne0 * ne1 * ne2 * ne3;
    constexpr පූර්ණ_සංඛ්‍යා BLOCK_SIZE = 256;
    const පූර්ණ_සංඛ්‍යා     num_blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

    const float inv_ne0      = 1.0f / ne0;
    const float inv_ne_01    = 1.0f / (ne0 * ne1);
    const float inv_ne_012   = 1.0f / (ne0 * ne1 * ne2);
    const පූර්ණ_සංඛ්‍යා   repeat_count = nr0 * nr1 * nr2 * nr3;

    queue_ptr stream = ctx.stream();

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(num_blocks * BLOCK_SIZE), sycl::range<1>(BLOCK_SIZE)),
        [=](sycl::nd_item<1> item_ct1) {
            const size_t i = item_ct1.get_global_linear_id();
            නම් (i >= total) {
                ආපසු;
            }

            const පූර්ණ_සංඛ්‍යා i3 = (පූර්ණ_සංඛ්‍යා) (i * inv_ne_012);
            const පූර්ණ_සංඛ්‍යා i2 = (පූර්ණ_සංඛ්‍යා) (i * inv_ne_01) - i3 * ne2;
            const පූර්ණ_සංඛ්‍යා i1 = (පූර්ණ_සංඛ්‍යා) (i * inv_ne0) - (පූර්ණ_සංඛ්‍යා) (i * inv_ne_01) * ne1;
            const පූර්ණ_සංඛ්‍යා i0 = i - (පූර්ණ_සංඛ්‍යා) (i * inv_ne0) * ne0;

            පූර්ණ_සංඛ්‍යා   j0 = 0, j1 = 0, j2 = 0, j3 = 0;
            float acc = 0.0f;

            සඳහා (පූර්ණ_සංඛ්‍යා j = 0; j < repeat_count; ++j) {
                const float * ptr = (const float *) (base + (i0 + j0 * ne0) * nb0 + (i1 + j1 * ne1) * nb1 +
                    (i2 + j2 * ne2) * nb2 + (i3 + j3 * ne3) * nb3);
                acc += *ptr;

                පූර්ණ_සංඛ්‍යා carry = (++j0 >= nr0);
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
