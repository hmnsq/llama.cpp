#ಅನ್ತರ್ಭಾವ "roll.hpp"
#ಅನ್ತರ್ಭಾವ "common.hpp"

using ನಾಮಾಕಾಶ sycl;

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಪೂರ್ಣಾಙ್ಕ wrap_add(ಪೂರ್ಣಾಙ್ಕ i, ಪೂರ್ಣಾಙ್ಕ shift, ಪೂರ್ಣಾಙ್ಕ n) {

    ಪೂರ್ಣಾಙ್ಕ s = i + shift;
    ಪ್ರತ್ಯಾಗಮ (s >= n) ? (s - n) : s;
}

ಸ್ಥಿರ ಶೂನ್ಯ kernel_roll_fused_i0_i1(
    queue &q,
    ಧ್ರುವ ಭಿನ್ನ *src_d,
    ಭಿನ್ನ *dst_d,
    ಪೂರ್ಣಾಙ್ಕ ne0, ಪೂರ್ಣಾಙ್ಕ ne1, ಪೂರ್ಣಾಙ್ಕ ne2, ಪೂರ್ಣಾಙ್ಕ ne3,
    ಪೂರ್ಣಾಙ್ಕ sh0, ಪೂರ್ಣಾಙ್ಕ sh1, ಪೂರ್ಣಾಙ್ಕ sh2, ಪೂರ್ಣಾಙ್ಕ sh3)
{
    ಯದಿ (ne0 == 0 || ne1 == 0 || ne2 == 0 || ne3 == 0) ಪ್ರತ್ಯಾಗಮ;


    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ stride1 = ne0;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ stride2 = ne0 * ne1;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ stride3 = ne0 * ne1 * ne2;


    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ shNe0 = (ne0 - sh0) % ne0;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ shNe1 = (ne1 - sh1) % ne1;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ shNe2 = (ne2 - sh2) % ne2;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ shNe3 = (ne3 - sh3) % ne3;


    ಧ್ರುವ size_t g0 = (size_t) ne3;
    ಧ್ರುವ size_t g1 = (size_t) ne2;
    ಧ್ರುವ size_t g2 = (size_t) (ne1 * ne0);

    ಧ್ರುವ range<3> global{ g0, g1, g2 };

    q.submit([&](handler &h) {
        h.parallel_for(global, [=](id<3> idx) {
            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i3 = (ಪೂರ್ಣಾಙ್ಕ) idx[0];
            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i2 = (ಪೂರ್ಣಾಙ್ಕ) idx[1];

            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ fused = (ಪೂರ್ಣಾಙ್ಕ) idx[2];
            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i1 = fused / ne0;
            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i0 = fused - i1 * ne0;  // fused % ne0


            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ idx_dst = i0
                              + i1 * stride1
                              + i2 * stride2
                              + i3 * stride3;


            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s0 = wrap_add(i0, shNe0, ne0);
            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s1 = wrap_add(i1, shNe1, ne1);
            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s2 = wrap_add(i2, shNe2, ne2);
            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s3 = wrap_add(i3, shNe3, ne3);

            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ idx_src = s0
                              + s1 * stride1
                              + s2 * stride2
                              + s3 * stride3;

            dst_d[idx_dst] = src_d[idx_src];
        });
    });
}

ಶೂನ್ಯ ggml_sycl_roll(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    ಧ್ರುವ ggml_tensor *src = dst->src[0];
    GGML_ASSERT(src && src->type == GGML_TYPE_F32);

    // the kernel indexes in 32-bit
    GGML_ASSERT(dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3] <= INT32_MAX);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne0 = (ಪೂರ್ಣಾಙ್ಕ) dst->ne[0];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne1 = (ಪೂರ್ಣಾಙ್ಕ) dst->ne[1];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne2 = (ಪೂರ್ಣಾಙ್ಕ) dst->ne[2];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne3 = (ಪೂರ್ಣಾಙ್ಕ) dst->ne[3];

    ಧ್ರುವ int32_t *params = (ಧ್ರುವ int32_t *) dst->op_params;
    ಪೂರ್ಣಾಙ್ಕ shift0 = params[0];
    ಪೂರ್ಣಾಙ್ಕ shift1 = params[1];
    ಪೂರ್ಣಾಙ್ಕ shift2 = params[2];
    ಪೂರ್ಣಾಙ್ಕ shift3 = params[3];


    ಯದಿ ((shift0 | shift1 | shift2 | shift3) == 0) {
        ಧ್ರುವ size_t nb = ggml_nbytes(src);
        queue *q = ctx.stream();
        SYCL_CHECK(CHECK_TRY_ERROR(q->memcpy(dst->data, src->data, nb)));
        ಪ್ರತ್ಯಾಗಮ;
    }

    auto norm = [](ಪೂರ್ಣಾಙ್ಕ sh, ಪೂರ್ಣಾಙ್ಕ n) -> ಪೂರ್ಣಾಙ್ಕ {
        ಯದಿ (n <= 0) ಪ್ರತ್ಯಾಗಮ 0;
        sh %= n;
        ಯದಿ (sh < 0) sh += n;
        ಪ್ರತ್ಯಾಗಮ sh;
    };
    shift0 = norm(shift0, ne0);
    shift1 = norm(shift1, ne1);
    shift2 = norm(shift2, ne2);
    shift3 = norm(shift3, ne3);

    ಯತ್ನ {
        queue *q = ctx.stream();

        ಧ್ರುವ ಭಿನ್ನ *src_d = (ಧ್ರುವ ಭಿನ್ನ *) src->data;
        ಭಿನ್ನ *dst_d = (ಭಿನ್ನ *) dst->data;
        GGML_ASSERT(src_d && dst_d);

        kernel_roll_fused_i0_i1(
            *q, src_d, dst_d,
            ne0, ne1, ne2, ne3,
            shift0, shift1, shift2, shift3
        );
    } ಗೃಹಾಣ (ಧ್ರುವ std::exception &e) {
        std::fprintf(stderr, "[SYCL-ROLL] ERROR: %s\n", e.what());
        ಕ್ಷಿಪ;
    }
}
