#अन्तर्भाव "roll.hpp"
#अन्तर्भाव "common.hpp"

using नामाकाश sycl;

स्थिर अन्तःस्थ पूर्णाङ्क wrap_add(पूर्णाङ्क i, पूर्णाङ्क shift, पूर्णाङ्क n) {

    पूर्णाङ्क s = i + shift;
    फर्काऊ (s >= n) ? (s - n) : s;
}

स्थिर शून्य kernel_roll_fused_i0_i1(
    queue &q,
    अचर भिन्न *src_d,
    भिन्न *dst_d,
    पूर्णाङ्क ne0, पूर्णाङ्क ne1, पूर्णाङ्क ne2, पूर्णाङ्क ne3,
    पूर्णाङ्क sh0, पूर्णाङ्क sh1, पूर्णाङ्क sh2, पूर्णाङ्क sh3)
{
    यदि (ne0 == 0 || ne1 == 0 || ne2 == 0 || ne3 == 0) फर्काऊ;


    अचर पूर्णाङ्क stride1 = ne0;
    अचर पूर्णाङ्क stride2 = ne0 * ne1;
    अचर पूर्णाङ्क stride3 = ne0 * ne1 * ne2;


    अचर पूर्णाङ्क shNe0 = (ne0 - sh0) % ne0;
    अचर पूर्णाङ्क shNe1 = (ne1 - sh1) % ne1;
    अचर पूर्णाङ्क shNe2 = (ne2 - sh2) % ne2;
    अचर पूर्णाङ्क shNe3 = (ne3 - sh3) % ne3;


    अचर माप_प्रकार g0 = (माप_प्रकार) ne3;
    अचर माप_प्रकार g1 = (माप_प्रकार) ne2;
    अचर माप_प्रकार g2 = (माप_प्रकार) (ne1 * ne0);

    अचर range<3> global{ g0, g1, g2 };

    q.submit([&](handler &h) {
        h.parallel_for(global, [=](id<3> idx) {
            अचर पूर्णाङ्क i3 = (पूर्णाङ्क) idx[0];
            अचर पूर्णाङ्क i2 = (पूर्णाङ्क) idx[1];

            अचर पूर्णाङ्क fused = (पूर्णाङ्क) idx[2];
            अचर पूर्णाङ्क i1 = fused / ne0;
            अचर पूर्णाङ्क i0 = fused - i1 * ne0;  // fused % ne0


            अचर पूर्णाङ्क idx_dst = i0
                              + i1 * stride1
                              + i2 * stride2
                              + i3 * stride3;


            अचर पूर्णाङ्क s0 = wrap_add(i0, shNe0, ne0);
            अचर पूर्णाङ्क s1 = wrap_add(i1, shNe1, ne1);
            अचर पूर्णाङ्क s2 = wrap_add(i2, shNe2, ne2);
            अचर पूर्णाङ्क s3 = wrap_add(i3, shNe3, ne3);

            अचर पूर्णाङ्क idx_src = s0
                              + s1 * stride1
                              + s2 * stride2
                              + s3 * stride3;

            dst_d[idx_dst] = src_d[idx_src];
        });
    });
}

शून्य ggml_sycl_roll(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    अचर ggml_tensor *src = dst->src[0];
    GGML_ASSERT(src && src->type == GGML_TYPE_F32);

    // the kernel indexes in 32-bit
    GGML_ASSERT(dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3] <= INT32_MAX);
    अचर पूर्णाङ्क ne0 = (पूर्णाङ्क) dst->ne[0];
    अचर पूर्णाङ्क ne1 = (पूर्णाङ्क) dst->ne[1];
    अचर पूर्णाङ्क ne2 = (पूर्णाङ्क) dst->ne[2];
    अचर पूर्णाङ्क ne3 = (पूर्णाङ्क) dst->ne[3];

    अचर int32_t *params = (अचर int32_t *) dst->op_params;
    पूर्णाङ्क shift0 = params[0];
    पूर्णाङ्क shift1 = params[1];
    पूर्णाङ्क shift2 = params[2];
    पूर्णाङ्क shift3 = params[3];


    यदि ((shift0 | shift1 | shift2 | shift3) == 0) {
        अचर माप_प्रकार nb = ggml_nbytes(src);
        queue *q = ctx.stream();
        SYCL_CHECK(CHECK_TRY_ERROR(q->स_नकल(dst->data, src->data, nb)));
        फर्काऊ;
    }

    स्वतः norm = [](पूर्णाङ्क sh, पूर्णाङ्क n) -> पूर्णाङ्क {
        यदि (n <= 0) फर्काऊ 0;
        sh %= n;
        यदि (sh < 0) sh += n;
        फर्काऊ sh;
    };
    shift0 = norm(shift0, ne0);
    shift1 = norm(shift1, ne1);
    shift2 = norm(shift2, ne2);
    shift3 = norm(shift3, ne3);

    यत्न {
        queue *q = ctx.stream();

        अचर भिन्न *src_d = (अचर भिन्न *) src->data;
        भिन्न *dst_d = (भिन्न *) dst->data;
        GGML_ASSERT(src_d && dst_d);

        kernel_roll_fused_i0_i1(
            *q, src_d, dst_d,
            ne0, ne1, ne2, ne3,
            shift0, shift1, shift2, shift3
        );
    } गृहाण (अचर std::exception &e) {
        std::ख_लिखो(मानक_त्रुटि, "[SYCL-ROLL] ERROR: %s\n", e.what());
        क्षिप;
    }
}
