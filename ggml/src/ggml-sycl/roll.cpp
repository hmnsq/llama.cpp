#ಅನ್ತರ್ಭಾವ "roll.hpp"
#ಅನ್ತರ್ಭಾವ "common.hpp"

ಬಳಸಿ ನಾಮಾಕಾಶ sycl;

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಪೂರ್ಣಾಂಕ wrap_ಕೂಡಿಸು(ಪೂರ್ಣಾಂಕ i, ಪೂರ್ಣಾಂಕ ಸರಿಸು, ಪೂರ್ಣಾಂಕ n) {

    ಪೂರ್ಣಾಂಕ s = i + ಸರಿಸು;
    ಹಿಂತಿರುಗಿ (s >= n) ? (s - n) : s;
}

ಸ್ಥಿರ ಶೂನ್ಯ kernel_roll_fused_i0_i1(
    queue &q,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *src_d,
    ದಶಮಾಂಶ *dst_d,
    ಪೂರ್ಣಾಂಕ ne0, ಪೂರ್ಣಾಂಕ ne1, ಪೂರ್ಣಾಂಕ ne2, ಪೂರ್ಣಾಂಕ ne3,
    ಪೂರ್ಣಾಂಕ sh0, ಪೂರ್ಣಾಂಕ sh1, ಪೂರ್ಣಾಂಕ sh2, ಪೂರ್ಣಾಂಕ sh3)
{
    ಆದರೆ (ne0 == 0 || ne1 == 0 || ne2 == 0 || ne3 == 0) ಹಿಂತಿರುಗಿ;


    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride1 = ne0;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride2 = ne0 * ne1;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride3 = ne0 * ne1 * ne2;


    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ shNe0 = (ne0 - sh0) % ne0;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ shNe1 = (ne1 - sh1) % ne1;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ shNe2 = (ne2 - sh2) % ne2;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ shNe3 = (ne3 - sh3) % ne3;


    ಸ್ಥಿರಾಂಕ size_t g0 = (size_t) ne3;
    ಸ್ಥಿರಾಂಕ size_t g1 = (size_t) ne2;
    ಸ್ಥಿರಾಂಕ size_t g2 = (size_t) (ne1 * ne0);

    ಸ್ಥಿರಾಂಕ range<3> global{ g0, g1, g2 };

    q.submit([&](handler &h) {
        h.parallel_for(global, [=](id<3> idx) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i3 = (ಪೂರ್ಣಾಂಕ) idx[0];
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i2 = (ಪೂರ್ಣಾಂಕ) idx[1];

            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ fused = (ಪೂರ್ಣಾಂಕ) idx[2];
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i1 = fused / ne0;
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i0 = fused - i1 * ne0;  // fused % ne0


            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ idx_dst = i0
                              + i1 * stride1
                              + i2 * stride2
                              + i3 * stride3;


            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s0 = wrap_ಕೂಡಿಸು(i0, shNe0, ne0);
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s1 = wrap_ಕೂಡಿಸು(i1, shNe1, ne1);
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s2 = wrap_ಕೂಡಿಸು(i2, shNe2, ne2);
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s3 = wrap_ಕೂಡಿಸು(i3, shNe3, ne3);

            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ idx_src = s0
                              + s1 * stride1
                              + s2 * stride2
                              + s3 * stride3;

            dst_d[idx_dst] = src_d[idx_src];
        });
    });
}

ಶೂನ್ಯ ggml_sycl_roll(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    ಸ್ಥಿರಾಂಕ ggml_tensor *src = dst->src[0];
    GGML_ASSERT(src && src->type == GGML_TYPE_F32);

    // the kernel indexes in 32-bit
    GGML_ASSERT(dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3] <= INT32_MAX);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne0 = (ಪೂರ್ಣಾಂಕ) dst->ne[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne1 = (ಪೂರ್ಣಾಂಕ) dst->ne[1];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne2 = (ಪೂರ್ಣಾಂಕ) dst->ne[2];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne3 = (ಪೂರ್ಣಾಂಕ) dst->ne[3];

    ಸ್ಥಿರಾಂಕ int32_t *params = (ಸ್ಥಿರಾಂಕ int32_t *) dst->op_params;
    ಪೂರ್ಣಾಂಕ shift0 = params[0];
    ಪೂರ್ಣಾಂಕ shift1 = params[1];
    ಪೂರ್ಣಾಂಕ shift2 = params[2];
    ಪೂರ್ಣಾಂಕ shift3 = params[3];


    ಆದರೆ ((shift0 | shift1 | shift2 | shift3) == 0) {
        ಸ್ಥಿರಾಂಕ size_t nb = ggml_nbytes(src);
        queue *q = ctx.stream();
        SYCL_CHECK(CHECK_TRY_ERROR(q->ಸ್ಮರಣೆ_ನಕಲು(dst->ದತ್ತಾಂಶ, src->ದತ್ತಾಂಶ, nb)));
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ವಯಂಚಾಲಿತ norm = [](ಪೂರ್ಣಾಂಕ sh, ಪೂರ್ಣಾಂಕ n) -> ಪೂರ್ಣಾಂಕ {
        ಆದರೆ (n <= 0) ಹಿಂತಿರುಗಿ 0;
        sh %= n;
        ಆದರೆ (sh < 0) sh += n;
        ಹಿಂತಿರುಗಿ sh;
    };
    shift0 = norm(shift0, ne0);
    shift1 = norm(shift1, ne1);
    shift2 = norm(shift2, ne2);
    shift3 = norm(shift3, ne3);

    ಪ್ರಯತ್ನಿಸು {
        queue *q = ctx.stream();

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *src_d = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src->ದತ್ತಾಂಶ;
        ದಶಮಾಂಶ *dst_d = (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ;
        GGML_ASSERT(src_d && dst_d);

        kernel_roll_fused_i0_i1(
            *q, src_d, dst_d,
            ne0, ne1, ne2, ne3,
            shift0, shift1, shift2, shift3
        );
    } ಹಿಡಿ (ಸ್ಥಿರಾಂಕ std::exception &e) {
        std::fprintf(stderr, "[SYCL-ROLL] ERROR: %s\n", e.what());
        ಎಸೆ;
    }
}
