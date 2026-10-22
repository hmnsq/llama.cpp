#অন্তর্ভাব "roll.hpp"
#অন্তর্ভাব "common.hpp"

using নামাকাশ sycl;

স্থির অন্তঃরেখ পূর্ণসংখ্যা wrap_add(পূর্ণসংখ্যা i, পূর্ণসংখ্যা shift, পূর্ণসংখ্যা n) {

    পূর্ণসংখ্যা s = i + shift;
    ফেরত (s >= n) ? (s - n) : s;
}

স্থির শূন্য kernel_roll_fused_i0_i1(
    queue &q,
    ধ্রুবক দশমিক *src_d,
    দশমিক *dst_d,
    পূর্ণসংখ্যা ne0, পূর্ণসংখ্যা ne1, পূর্ণসংখ্যা ne2, পূর্ণসংখ্যা ne3,
    পূর্ণসংখ্যা sh0, পূর্ণসংখ্যা sh1, পূর্ণসংখ্যা sh2, পূর্ণসংখ্যা sh3)
{
    যদি (ne0 == 0 || ne1 == 0 || ne2 == 0 || ne3 == 0) ফেরত;


    ধ্রুবক পূর্ণসংখ্যা stride1 = ne0;
    ধ্রুবক পূর্ণসংখ্যা stride2 = ne0 * ne1;
    ধ্রুবক পূর্ণসংখ্যা stride3 = ne0 * ne1 * ne2;


    ধ্রুবক পূর্ণসংখ্যা shNe0 = (ne0 - sh0) % ne0;
    ধ্রুবক পূর্ণসংখ্যা shNe1 = (ne1 - sh1) % ne1;
    ধ্রুবক পূর্ণসংখ্যা shNe2 = (ne2 - sh2) % ne2;
    ধ্রুবক পূর্ণসংখ্যা shNe3 = (ne3 - sh3) % ne3;


    ধ্রুবক size_t g0 = (size_t) ne3;
    ধ্রুবক size_t g1 = (size_t) ne2;
    ধ্রুবক size_t g2 = (size_t) (ne1 * ne0);

    ধ্রুবক range<3> global{ g0, g1, g2 };

    q.submit([&](handler &h) {
        h.parallel_for(global, [=](id<3> idx) {
            ধ্রুবক পূর্ণসংখ্যা i3 = (পূর্ণসংখ্যা) idx[0];
            ধ্রুবক পূর্ণসংখ্যা i2 = (পূর্ণসংখ্যা) idx[1];

            ধ্রুবক পূর্ণসংখ্যা fused = (পূর্ণসংখ্যা) idx[2];
            ধ্রুবক পূর্ণসংখ্যা i1 = fused / ne0;
            ধ্রুবক পূর্ণসংখ্যা i0 = fused - i1 * ne0;  // fused % ne0


            ধ্রুবক পূর্ণসংখ্যা idx_dst = i0
                              + i1 * stride1
                              + i2 * stride2
                              + i3 * stride3;


            ধ্রুবক পূর্ণসংখ্যা s0 = wrap_add(i0, shNe0, ne0);
            ধ্রুবক পূর্ণসংখ্যা s1 = wrap_add(i1, shNe1, ne1);
            ধ্রুবক পূর্ণসংখ্যা s2 = wrap_add(i2, shNe2, ne2);
            ধ্রুবক পূর্ণসংখ্যা s3 = wrap_add(i3, shNe3, ne3);

            ধ্রুবক পূর্ণসংখ্যা idx_src = s0
                              + s1 * stride1
                              + s2 * stride2
                              + s3 * stride3;

            dst_d[idx_dst] = src_d[idx_src];
        });
    });
}

শূন্য ggml_sycl_roll(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    ধ্রুবক ggml_tensor *src = dst->src[0];
    GGML_ASSERT(src && src->type == GGML_TYPE_F32);

    // the kernel indexes in 32-bit
    GGML_ASSERT(dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3] <= INT32_MAX);
    ধ্রুবক পূর্ণসংখ্যা ne0 = (পূর্ণসংখ্যা) dst->ne[0];
    ধ্রুবক পূর্ণসংখ্যা ne1 = (পূর্ণসংখ্যা) dst->ne[1];
    ধ্রুবক পূর্ণসংখ্যা ne2 = (পূর্ণসংখ্যা) dst->ne[2];
    ধ্রুবক পূর্ণসংখ্যা ne3 = (পূর্ণসংখ্যা) dst->ne[3];

    ধ্রুবক int32_t *params = (ধ্রুবক int32_t *) dst->op_params;
    পূর্ণসংখ্যা shift0 = params[0];
    পূর্ণসংখ্যা shift1 = params[1];
    পূর্ণসংখ্যা shift2 = params[2];
    পূর্ণসংখ্যা shift3 = params[3];


    যদি ((shift0 | shift1 | shift2 | shift3) == 0) {
        ধ্রুবক size_t nb = ggml_nbytes(src);
        queue *q = ctx.stream();
        SYCL_CHECK(CHECK_TRY_ERROR(q->memcpy(dst->তথ্য, src->তথ্য, nb)));
        ফেরত;
    }

    auto norm = [](পূর্ণসংখ্যা sh, পূর্ণসংখ্যা n) -> পূর্ণসংখ্যা {
        যদি (n <= 0) ফেরত 0;
        sh %= n;
        যদি (sh < 0) sh += n;
        ফেরত sh;
    };
    shift0 = norm(shift0, ne0);
    shift1 = norm(shift1, ne1);
    shift2 = norm(shift2, ne2);
    shift3 = norm(shift3, ne3);

    চেষ্টা {
        queue *q = ctx.stream();

        ধ্রুবক দশমিক *src_d = (ধ্রুবক দশমিক *) src->তথ্য;
        দশমিক *dst_d = (দশমিক *) dst->তথ্য;
        GGML_ASSERT(src_d && dst_d);

        kernel_roll_fused_i0_i1(
            *q, src_d, dst_d,
            ne0, ne1, ne2, ne3,
            shift0, shift1, shift2, shift3
        );
    } ধরো (ধ্রুবক std::exception &e) {
        std::fprintf(stderr, "[SYCL-ROLL] ERROR: %s\n", e.what());
        ছোড়ো;
    }
}
