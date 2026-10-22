#समावेश "repeat_back.hpp"

#समावेश "common.hpp"

#घोषणा GGML_ASSERT_TENSOR_FITS_INT(t) \
    GGML_ASSERT((t)->ne[0] < पूर्णांक_उच्च && (t)->ne[1] < पूर्णांक_उच्च && (t)->ne[2] < पूर्णांक_उच्च && (t)->ne[3] < पूर्णांक_उच्च)

व्योम ggml_sycl_op_repeat_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    स्थिर भग्न * src0_dd = (स्थिर भग्न *) dst->src[0]->तथ्य;
    भग्न *       dst_dd  = (भग्न *) dst->तथ्य;

    GGML_ASSERT_TENSOR_FITS_INT(dst);
    GGML_ASSERT_TENSOR_FITS_INT(dst->src[0]);

    स्थिर पूर्णांक ne0 = dst->ne[0], ne1 = dst->ne[1], ne2 = dst->ne[2], ne3 = dst->ne[3];
    स्थिर पूर्णांक ne00 = dst->src[0]->ne[0], ne01 = dst->src[0]->ne[1], ne02 = dst->src[0]->ne[2],
              ne03 = dst->src[0]->ne[3];

    स्थिर पूर्णांक nr0 = ne00 / ne0;
    स्थिर पूर्णांक nr1 = ne01 / ne1;
    स्थिर पूर्णांक nr2 = ne02 / ne2;
    स्थिर पूर्णांक nr3 = ne03 / ne3;

    स्थिर पूर्णांक nb0 = dst->src[0]->nb[0];
    स्थिर पूर्णांक nb1 = dst->src[0]->nb[1];
    स्थिर पूर्णांक nb2 = dst->src[0]->nb[2];
    स्थिर पूर्णांक nb3 = dst->src[0]->nb[3];

    स्थिर अक्षर * base = (स्थिर अक्षर *) src0_dd;

    स्थिर माप_प्रकार  योग      = (माप_प्रकार) ne0 * ne1 * ne2 * ne3;
    constexpr पूर्णांक BLOCK_आकार = 256;
    स्थिर पूर्णांक     अंक_blocks = (योग + BLOCK_आकार - 1) / BLOCK_आकार;

    स्थिर भग्न inv_ne0      = 1.0f / ne0;
    स्थिर भग्न inv_ne_01    = 1.0f / (ne0 * ne1);
    स्थिर भग्न inv_ne_012   = 1.0f / (ne0 * ne1 * ne2);
    स्थिर पूर्णांक   repeat_गिनती = nr0 * nr1 * nr2 * nr3;

    queue_ptr stream = ctx.stream();

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(अंक_blocks * BLOCK_आकार), sycl::range<1>(BLOCK_आकार)),
        [=](sycl::nd_item<1> item_ct1) {
            स्थिर माप_प्रकार i = item_ct1.get_global_linear_id();
            अगर (i >= योग) {
                वापस;
            }

            स्थिर पूर्णांक i3 = (पूर्णांक) (i * inv_ne_012);
            स्थिर पूर्णांक i2 = (पूर्णांक) (i * inv_ne_01) - i3 * ne2;
            स्थिर पूर्णांक i1 = (पूर्णांक) (i * inv_ne0) - (पूर्णांक) (i * inv_ne_01) * ne1;
            स्थिर पूर्णांक i0 = i - (पूर्णांक) (i * inv_ne0) * ne0;

            पूर्णांक   j0 = 0, j1 = 0, j2 = 0, j3 = 0;
            भग्न acc = 0.0f;

            क्रम (पूर्णांक j = 0; j < repeat_गिनती; ++j) {
                स्थिर भग्न * ptr = (स्थिर भग्न *) (base + (i0 + j0 * ne0) * nb0 + (i1 + j1 * ne1) * nb1 +
                    (i2 + j2 * ne2) * nb2 + (i3 + j3 * ne3) * nb3);
                acc += *ptr;

                पूर्णांक carry = (++j0 >= nr0);
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
