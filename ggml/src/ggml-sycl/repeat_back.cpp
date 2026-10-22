#ಅನ್ತರ್ಭಾವ "repeat_back.hpp"

#ಅನ್ತರ್ಭಾವ "common.hpp"

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_ASSERT_TENSOR_ಹೊಂದುತ್ತದೆ_INT(t) \
    GGML_ASSERT((t)->ne[0] < ಗರಿಷ್ಠ_ಪೂರ್ಣಾಂಕ && (t)->ne[1] < ಗರಿಷ್ಠ_ಪೂರ್ಣಾಂಕ && (t)->ne[2] < ಗರಿಷ್ಠ_ಪೂರ್ಣಾಂಕ && (t)->ne[3] < ಗರಿಷ್ಠ_ಪೂರ್ಣಾಂಕ)

ಶೂನ್ಯ ggml_sycl_op_repeat_ಹಿಂಭಾಗ(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src0_dd = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) dst->src[0]->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ *       dst_dd  = (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ;

    GGML_ASSERT_TENSOR_ಹೊಂದುತ್ತದೆ_INT(dst);
    GGML_ASSERT_TENSOR_ಹೊಂದುತ್ತದೆ_INT(dst->src[0]);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne0 = dst->ne[0], ne1 = dst->ne[1], ne2 = dst->ne[2], ne3 = dst->ne[3];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne00 = dst->src[0]->ne[0], ne01 = dst->src[0]->ne[1], ne02 = dst->src[0]->ne[2],
              ne03 = dst->src[0]->ne[3];

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr0 = ne00 / ne0;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr1 = ne01 / ne1;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr2 = ne02 / ne2;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr3 = ne03 / ne3;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb0 = dst->src[0]->nb[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb1 = dst->src[0]->nb[1];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb2 = dst->src[0]->nb[2];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb3 = dst->src[0]->nb[3];

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ತಳ = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src0_dd;

    ಸ್ಥಿರಾಂಕ size_t  ಒಟ್ಟು      = (size_t) ne0 * ne1 * ne2 * ne3;
    constexpr ಪೂರ್ಣಾಂಕ ಖಂಡ_ಗಾತ್ರ_2 = 256;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ     ಅಂಕೆ_ಖಂಡಗಳು = (ಒಟ್ಟು + ಖಂಡ_ಗಾತ್ರ_2 - 1) / ಖಂಡ_ಗಾತ್ರ_2;

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ inv_ne0      = 1.0f / ne0;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ inv_ne_01    = 1.0f / (ne0 * ne1);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ inv_ne_012   = 1.0f / (ne0 * ne1 * ne2);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ   repeat_ಎಣಿಕೆ = nr0 * nr1 * nr2 * nr3;

    ಸರತಿ_ptr stream = ctx.stream();

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(ಅಂಕೆ_ಖಂಡಗಳು * ಖಂಡ_ಗಾತ್ರ_2), sycl::range<1>(ಖಂಡ_ಗಾತ್ರ_2)),
        [=](sycl::nd_ಅಂಶ<1> ಅಂಶ_ct1) {
            ಸ್ಥಿರಾಂಕ size_t i = ಅಂಶ_ct1.get_global_ರೇಖೀಯ_id();
            ಆದರೆ (i >= ಒಟ್ಟು) {
                ಹಿಂತಿರುಗಿ;
            }

            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i3 = (ಪೂರ್ಣಾಂಕ) (i * inv_ne_012);
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i2 = (ಪೂರ್ಣಾಂಕ) (i * inv_ne_01) - i3 * ne2;
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i1 = (ಪೂರ್ಣಾಂಕ) (i * inv_ne0) - (ಪೂರ್ಣಾಂಕ) (i * inv_ne_01) * ne1;
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i0 = i - (ಪೂರ್ಣಾಂಕ) (i * inv_ne0) * ne0;

            ಪೂರ್ಣಾಂಕ   j0 = 0, j1 = 0, j2 = 0, j3 = 0;
            ದಶಮಾಂಶ acc = 0.0f;

            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < repeat_ಎಣಿಕೆ; ++j) {
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ptr = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) (ತಳ + (i0 + j0 * ne0) * nb0 + (i1 + j1 * ne1) * nb1 +
                    (i2 + j2 * ne2) * nb2 + (i3 + j3 * ne3) * nb3);
                acc += *ptr;

                ಪೂರ್ಣಾಂಕ ವಾಹಕ = (++j0 >= nr0);
                j0 -= ವಾಹಕ * nr0;
                ವಾಹಕ = (ವಾಹಕ && (++j1 >= nr1));
                j1 -= ವಾಹಕ * nr1;
                ವಾಹಕ = (ವಾಹಕ && (++j2 >= nr2));
                j2 -= ವಾಹಕ * nr2;
                j3 += ವಾಹಕ;
            }
            dst_dd[i] = acc;
        });
}
