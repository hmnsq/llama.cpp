#ಅನ್ತರ್ಭಾವ "repeat_back.hpp"

#ಅನ್ತರ್ಭಾವ "common.hpp"

#define GGML_ASSERT_TENSOR_FITS_INT(t) \
    GGML_ASSERT((t)->ne[0] < INT_MAX && (t)->ne[1] < INT_MAX && (t)->ne[2] < INT_MAX && (t)->ne[3] < INT_MAX)

ಶೂನ್ಯ ggml_sycl_op_repeat_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    ಧ್ರುವ ಭಿನ್ನ * src0_dd = (ಧ್ರುವ ಭಿನ್ನ *) dst->src[0]->data;
    ಭಿನ್ನ *       dst_dd  = (ಭಿನ್ನ *) dst->data;

    GGML_ASSERT_TENSOR_FITS_INT(dst);
    GGML_ASSERT_TENSOR_FITS_INT(dst->src[0]);

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne0 = dst->ne[0], ne1 = dst->ne[1], ne2 = dst->ne[2], ne3 = dst->ne[3];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00 = dst->src[0]->ne[0], ne01 = dst->src[0]->ne[1], ne02 = dst->src[0]->ne[2],
              ne03 = dst->src[0]->ne[3];

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nr0 = ne00 / ne0;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nr1 = ne01 / ne1;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nr2 = ne02 / ne2;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nr3 = ne03 / ne3;

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb0 = dst->src[0]->nb[0];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb1 = dst->src[0]->nb[1];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb2 = dst->src[0]->nb[2];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb3 = dst->src[0]->nb[3];

    ಧ್ರುವ ವರ್ಣ * base = (ಧ್ರುವ ವರ್ಣ *) src0_dd;

    ಧ್ರುವ size_t  total      = (size_t) ne0 * ne1 * ne2 * ne3;
    constexpr ಪೂರ್ಣಾಙ್ಕ BLOCK_SIZE = 256;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ     ಅಂಕೆ_blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

    ಧ್ರುವ ಭಿನ್ನ inv_ne0      = 1.0f / ne0;
    ಧ್ರುವ ಭಿನ್ನ inv_ne_01    = 1.0f / (ne0 * ne1);
    ಧ್ರುವ ಭಿನ್ನ inv_ne_012   = 1.0f / (ne0 * ne1 * ne2);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ   repeat_ಎಣಿಕೆ = nr0 * nr1 * nr2 * nr3;

    queue_ptr stream = ctx.stream();

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(ಅಂಕೆ_blocks * BLOCK_SIZE), sycl::range<1>(BLOCK_SIZE)),
        [=](sycl::nd_item<1> item_ct1) {
            ಧ್ರುವ size_t i = item_ct1.get_global_linear_id();
            ಯದಿ (i >= total) {
                ಪ್ರತ್ಯಾಗಮ;
            }

            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i3 = (ಪೂರ್ಣಾಙ್ಕ) (i * inv_ne_012);
            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i2 = (ಪೂರ್ಣಾಙ್ಕ) (i * inv_ne_01) - i3 * ne2;
            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i1 = (ಪೂರ್ಣಾಙ್ಕ) (i * inv_ne0) - (ಪೂರ್ಣಾಙ್ಕ) (i * inv_ne_01) * ne1;
            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i0 = i - (ಪೂರ್ಣಾಙ್ಕ) (i * inv_ne0) * ne0;

            ಪೂರ್ಣಾಙ್ಕ   j0 = 0, j1 = 0, j2 = 0, j3 = 0;
            ಭಿನ್ನ acc = 0.0f;

            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < repeat_ಎಣಿಕೆ; ++j) {
                ಧ್ರುವ ಭಿನ್ನ * ptr = (ಧ್ರುವ ಭಿನ್ನ *) (base + (i0 + j0 * ne0) * nb0 + (i1 + j1 * ne1) * nb1 +
                    (i2 + j2 * ne2) * nb2 + (i3 + j3 * ne3) * nb3);
                acc += *ptr;

                ಪೂರ್ಣಾಙ್ಕ carry = (++j0 >= nr0);
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
