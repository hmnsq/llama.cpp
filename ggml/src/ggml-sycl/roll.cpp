#ضمن "roll.hpp"
#ضمن "common.hpp"

استفاده نطاق sycl;

ساكن مضمن صحيح wrap_جمع_کن(صحيح i, صحيح لغزش, صحيح n) {

    صحيح s = i + لغزش;
    أرجع (s >= n) ? (s - n) : s;
}

ساكن فراغ kernel_roll_fused_i0_i1(
    queue &q,
    ثابت عائم *src_d,
    عائم *dst_d,
    صحيح ne0, صحيح ne1, صحيح ne2, صحيح ne3,
    صحيح sh0, صحيح sh1, صحيح sh2, صحيح sh3)
{
    إذا (ne0 == 0 || ne1 == 0 || ne2 == 0 || ne3 == 0) أرجع;


    ثابت صحيح stride1 = ne0;
    ثابت صحيح stride2 = ne0 * ne1;
    ثابت صحيح stride3 = ne0 * ne1 * ne2;


    ثابت صحيح shNe0 = (ne0 - sh0) % ne0;
    ثابت صحيح shNe1 = (ne1 - sh1) % ne1;
    ثابت صحيح shNe2 = (ne2 - sh2) % ne2;
    ثابت صحيح shNe3 = (ne3 - sh3) % ne3;


    ثابت size_t g0 = (size_t) ne3;
    ثابت size_t g1 = (size_t) ne2;
    ثابت size_t g2 = (size_t) (ne1 * ne0);

    ثابت range<3> global{ g0, g1, g2 };

    q.submit([&](handler &h) {
        h.parallel_for(global, [=](id<3> idx) {
            ثابت صحيح i3 = (صحيح) idx[0];
            ثابت صحيح i2 = (صحيح) idx[1];

            ثابت صحيح fused = (صحيح) idx[2];
            ثابت صحيح i1 = fused / ne0;
            ثابت صحيح i0 = fused - i1 * ne0;  // fused % ne0


            ثابت صحيح idx_dst = i0
                              + i1 * stride1
                              + i2 * stride2
                              + i3 * stride3;


            ثابت صحيح s0 = wrap_جمع_کن(i0, shNe0, ne0);
            ثابت صحيح s1 = wrap_جمع_کن(i1, shNe1, ne1);
            ثابت صحيح s2 = wrap_جمع_کن(i2, shNe2, ne2);
            ثابت صحيح s3 = wrap_جمع_کن(i3, shNe3, ne3);

            ثابت صحيح idx_src = s0
                              + s1 * stride1
                              + s2 * stride2
                              + s3 * stride3;

            dst_d[idx_dst] = src_d[idx_src];
        });
    });
}

فراغ ggml_sycl_roll(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    ثابت ggml_tensor *src = dst->src[0];
    GGML_ASSERT(src && src->type == GGML_TYPE_F32);

    // the kernel indexes in 32-bit
    GGML_ASSERT(dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3] <= INT32_MAX);
    ثابت صحيح ne0 = (صحيح) dst->ne[0];
    ثابت صحيح ne1 = (صحيح) dst->ne[1];
    ثابت صحيح ne2 = (صحيح) dst->ne[2];
    ثابت صحيح ne3 = (صحيح) dst->ne[3];

    ثابت int32_t *params = (ثابت int32_t *) dst->op_params;
    صحيح shift0 = params[0];
    صحيح shift1 = params[1];
    صحيح shift2 = params[2];
    صحيح shift3 = params[3];


    إذا ((shift0 | shift1 | shift2 | shift3) == 0) {
        ثابت size_t nb = ggml_nbytes(src);
        queue *q = ctx.stream();
        SYCL_CHECK(CHECK_TRY_ERROR(q->کپی_حافظه(dst->داده, src->داده, nb)));
        أرجع;
    }

    خودکار norm = [](صحيح sh, صحيح n) -> صحيح {
        إذا (n <= 0) أرجع 0;
        sh %= n;
        إذا (sh < 0) sh += n;
        أرجع sh;
    };
    shift0 = norm(shift0, ne0);
    shift1 = norm(shift1, ne1);
    shift2 = norm(shift2, ne2);
    shift3 = norm(shift3, ne3);

    حاول {
        queue *q = ctx.stream();

        ثابت عائم *src_d = (ثابت عائم *) src->داده;
        عائم *dst_d = (عائم *) dst->داده;
        GGML_ASSERT(src_d && dst_d);

        kernel_roll_fused_i0_i1(
            *q, src_d, dst_d,
            ne0, ne1, ne2, ne3,
            shift0, shift1, shift2, shift3
        );
    } امسك (ثابت std::exception &e) {
        std::fprintf(stderr, "[SYCL-ROLL] ERROR: %s\n", e.what());
        ارم;
    }
}
