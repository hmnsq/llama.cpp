#समावेश "roll.hpp"
#समावेश "common.hpp"

using namespace sycl;

अटल अंतरभूत पूर्णांक wrap_add(पूर्णांक i, पूर्णांक shift, पूर्णांक n) {

    पूर्णांक s = i + shift;
    वापस (s >= n) ? (s - n) : s;
}

अटल व्योम kernel_roll_fused_i0_i1(
    queue &q,
    स्थिर भग्न *src_d,
    भग्न *dst_d,
    पूर्णांक ne0, पूर्णांक ne1, पूर्णांक ne2, पूर्णांक ne3,
    पूर्णांक sh0, पूर्णांक sh1, पूर्णांक sh2, पूर्णांक sh3)
{
    अगर (ne0 == 0 || ne1 == 0 || ne2 == 0 || ne3 == 0) वापस;


    स्थिर पूर्णांक stride1 = ne0;
    स्थिर पूर्णांक stride2 = ne0 * ne1;
    स्थिर पूर्णांक stride3 = ne0 * ne1 * ne2;


    स्थिर पूर्णांक shNe0 = (ne0 - sh0) % ne0;
    स्थिर पूर्णांक shNe1 = (ne1 - sh1) % ne1;
    स्थिर पूर्णांक shNe2 = (ne2 - sh2) % ne2;
    स्थिर पूर्णांक shNe3 = (ne3 - sh3) % ne3;


    स्थिर माप_प्रकार g0 = (माप_प्रकार) ne3;
    स्थिर माप_प्रकार g1 = (माप_प्रकार) ne2;
    स्थिर माप_प्रकार g2 = (माप_प्रकार) (ne1 * ne0);

    स्थिर range<3> global{ g0, g1, g2 };

    q.submit([&](handler &h) {
        h.parallel_for(global, [=](id<3> idx) {
            स्थिर पूर्णांक i3 = (पूर्णांक) idx[0];
            स्थिर पूर्णांक i2 = (पूर्णांक) idx[1];

            स्थिर पूर्णांक fused = (पूर्णांक) idx[2];
            स्थिर पूर्णांक i1 = fused / ne0;
            स्थिर पूर्णांक i0 = fused - i1 * ne0;  // fused % ne0


            स्थिर पूर्णांक idx_dst = i0
                              + i1 * stride1
                              + i2 * stride2
                              + i3 * stride3;


            स्थिर पूर्णांक s0 = wrap_add(i0, shNe0, ne0);
            स्थिर पूर्णांक s1 = wrap_add(i1, shNe1, ne1);
            स्थिर पूर्णांक s2 = wrap_add(i2, shNe2, ne2);
            स्थिर पूर्णांक s3 = wrap_add(i3, shNe3, ne3);

            स्थिर पूर्णांक idx_src = s0
                              + s1 * stride1
                              + s2 * stride2
                              + s3 * stride3;

            dst_d[idx_dst] = src_d[idx_src];
        });
    });
}

व्योम ggml_sycl_roll(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    स्थिर ggml_tensor *src = dst->src[0];
    GGML_ASSERT(src && src->type == GGML_TYPE_F32);

    // the kernel indexes in 32-bit
    GGML_ASSERT(dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3] <= INT32_MAX);
    स्थिर पूर्णांक ne0 = (पूर्णांक) dst->ne[0];
    स्थिर पूर्णांक ne1 = (पूर्णांक) dst->ne[1];
    स्थिर पूर्णांक ne2 = (पूर्णांक) dst->ne[2];
    स्थिर पूर्णांक ne3 = (पूर्णांक) dst->ne[3];

    स्थिर int32_t *params = (स्थिर int32_t *) dst->op_params;
    पूर्णांक shift0 = params[0];
    पूर्णांक shift1 = params[1];
    पूर्णांक shift2 = params[2];
    पूर्णांक shift3 = params[3];


    अगर ((shift0 | shift1 | shift2 | shift3) == 0) {
        स्थिर माप_प्रकार nb = ggml_nbytes(src);
        queue *q = ctx.stream();
        SYCL_CHECK(CHECK_TRY_ERROR(q->स_नकल(dst->तथ्य, src->तथ्य, nb)));
        वापस;
    }

    स्वतः norm = [](पूर्णांक sh, पूर्णांक n) -> पूर्णांक {
        अगर (n <= 0) वापस 0;
        sh %= n;
        अगर (sh < 0) sh += n;
        वापस sh;
    };
    shift0 = norm(shift0, ne0);
    shift1 = norm(shift1, ne1);
    shift2 = norm(shift2, ne2);
    shift3 = norm(shift3, ne3);

    प्रयास {
        queue *q = ctx.stream();

        स्थिर भग्न *src_d = (स्थिर भग्न *) src->तथ्य;
        भग्न *dst_d = (भग्न *) dst->तथ्य;
        GGML_ASSERT(src_d && dst_d);

        kernel_roll_fused_i0_i1(
            *q, src_d, dst_d,
            ne0, ne1, ne2, ne3,
            shift0, shift1, shift2, shift3
        );
    } पकड़ो (स्थिर std::exception &e) {
        std::ख_लिखो(मानक_त्रुटि, "[SYCL-ROLL] ERROR: %s\n", e.what());
        फेंको;
    }
}
