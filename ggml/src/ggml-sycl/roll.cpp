#شامل "roll.hpp"
#شامل "common.hpp"

استعمال نامخلا sycl;

ساکن اندرونی صحیح wrap_جوڑو(صحیح i, صحیح سرکاؤ, صحیح n) {

    صحیح s = i + سرکاؤ;
    واپس (s >= n) ? (s - n) : s;
}

ساکن خالی kernel_roll_fused_i0_i1(
    queue &q,
    ثابت اعشاری *src_d,
    اعشاری *dst_d,
    صحیح ne0, صحیح ne1, صحیح ne2, صحیح ne3,
    صحیح sh0, صحیح sh1, صحیح sh2, صحیح sh3)
{
    اگر (ne0 == 0 || ne1 == 0 || ne2 == 0 || ne3 == 0) واپس;


    ثابت صحیح stride1 = ne0;
    ثابت صحیح stride2 = ne0 * ne1;
    ثابت صحیح stride3 = ne0 * ne1 * ne2;


    ثابت صحیح shNe0 = (ne0 - sh0) % ne0;
    ثابت صحیح shNe1 = (ne1 - sh1) % ne1;
    ثابت صحیح shNe2 = (ne2 - sh2) % ne2;
    ثابت صحیح shNe3 = (ne3 - sh3) % ne3;


    ثابت size_t g0 = (size_t) ne3;
    ثابت size_t g1 = (size_t) ne2;
    ثابت size_t g2 = (size_t) (ne1 * ne0);

    ثابت range<3> global{ g0, g1, g2 };

    q.submit([&](handler &h) {
        h.parallel_for(global, [=](id<3> idx) {
            ثابت صحیح i3 = (صحیح) idx[0];
            ثابت صحیح i2 = (صحیح) idx[1];

            ثابت صحیح fused = (صحیح) idx[2];
            ثابت صحیح i1 = fused / ne0;
            ثابت صحیح i0 = fused - i1 * ne0;  // fused % ne0


            ثابت صحیح idx_dst = i0
                              + i1 * stride1
                              + i2 * stride2
                              + i3 * stride3;


            ثابت صحیح s0 = wrap_جوڑو(i0, shNe0, ne0);
            ثابت صحیح s1 = wrap_جوڑو(i1, shNe1, ne1);
            ثابت صحیح s2 = wrap_جوڑو(i2, shNe2, ne2);
            ثابت صحیح s3 = wrap_جوڑو(i3, shNe3, ne3);

            ثابت صحیح idx_src = s0
                              + s1 * stride1
                              + s2 * stride2
                              + s3 * stride3;

            dst_d[idx_dst] = src_d[idx_src];
        });
    });
}

خالی ggml_sycl_roll(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    ثابت ggml_tensor *src = dst->src[0];
    GGML_ASSERT(src && src->type == GGML_TYPE_F32);

    // the kernel indexes in 32-bit
    GGML_ASSERT(dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3] <= INT32_MAX);
    ثابت صحیح ne0 = (صحیح) dst->ne[0];
    ثابت صحیح ne1 = (صحیح) dst->ne[1];
    ثابت صحیح ne2 = (صحیح) dst->ne[2];
    ثابت صحیح ne3 = (صحیح) dst->ne[3];

    ثابت int32_t *params = (ثابت int32_t *) dst->op_params;
    صحیح shift0 = params[0];
    صحیح shift1 = params[1];
    صحیح shift2 = params[2];
    صحیح shift3 = params[3];


    اگر ((shift0 | shift1 | shift2 | shift3) == 0) {
        ثابت size_t nb = ggml_nbytes(src);
        queue *q = ctx.stream();
        SYCL_CHECK(CHECK_TRY_ERROR(q->حافظہ_نقل(dst->ڈیٹا, src->ڈیٹا, nb)));
        واپس;
    }

    خودکار norm = [](صحیح sh, صحیح n) -> صحیح {
        اگر (n <= 0) واپس 0;
        sh %= n;
        اگر (sh < 0) sh += n;
        واپس sh;
    };
    shift0 = norm(shift0, ne0);
    shift1 = norm(shift1, ne1);
    shift2 = norm(shift2, ne2);
    shift3 = norm(shift3, ne3);

    کوشش {
        queue *q = ctx.stream();

        ثابت اعشاری *src_d = (ثابت اعشاری *) src->ڈیٹا;
        اعشاری *dst_d = (اعشاری *) dst->ڈیٹا;
        GGML_ASSERT(src_d && dst_d);

        kernel_roll_fused_i0_i1(
            *q, src_d, dst_d,
            ne0, ne1, ne2, ne3,
            shift0, shift1, shift2, shift3
        );
    } پکڑو (ثابت std::exception &e) {
        std::fprintf(stderr, "[SYCL-ROLL] ERROR: %s\n", e.what());
        پھینکو;
    }
}
