#অন্তর্ভাৱ "roll.hpp"
#অন্তর্ভাৱ "common.hpp"

using নামাকাশ sycl;

স্থির অন্তঃস্থ পূর্ণাঙ্ক wrap_add(পূর্ণাঙ্ক i, পূর্ণাঙ্ক shift, পূর্ণাঙ্ক n) {

    পূর্ণাঙ্ক s = i + shift;
    ঘূৰাই (s >= n) ? (s - n) : s;
}

স্থির শূন্য kernel_roll_fused_i0_i1(
    queue &q,
    ধ্রুৱ ভিন্ন *src_d,
    ভিন্ন *dst_d,
    পূর্ণাঙ্ক ne0, পূর্ণাঙ্ক ne1, পূর্ণাঙ্ক ne2, পূর্ণাঙ্ক ne3,
    পূর্ণাঙ্ক sh0, পূর্ণাঙ্ক sh1, পূর্ণাঙ্ক sh2, পূর্ণাঙ্ক sh3)
{
    যদি (ne0 == 0 || ne1 == 0 || ne2 == 0 || ne3 == 0) ঘূৰাই;


    ধ্রুৱ পূর্ণাঙ্ক stride1 = ne0;
    ধ্রুৱ পূর্ণাঙ্ক stride2 = ne0 * ne1;
    ধ্রুৱ পূর্ণাঙ্ক stride3 = ne0 * ne1 * ne2;


    ধ্রুৱ পূর্ণাঙ্ক shNe0 = (ne0 - sh0) % ne0;
    ধ্রুৱ পূর্ণাঙ্ক shNe1 = (ne1 - sh1) % ne1;
    ধ্রুৱ পূর্ণাঙ্ক shNe2 = (ne2 - sh2) % ne2;
    ধ্রুৱ পূর্ণাঙ্ক shNe3 = (ne3 - sh3) % ne3;


    ধ্রুৱ size_t g0 = (size_t) ne3;
    ধ্রুৱ size_t g1 = (size_t) ne2;
    ধ্রুৱ size_t g2 = (size_t) (ne1 * ne0);

    ধ্রুৱ range<3> global{ g0, g1, g2 };

    q.submit([&](handler &h) {
        h.parallel_for(global, [=](id<3> idx) {
            ধ্রুৱ পূর্ণাঙ্ক i3 = (পূর্ণাঙ্ক) idx[0];
            ধ্রুৱ পূর্ণাঙ্ক i2 = (পূর্ণাঙ্ক) idx[1];

            ধ্রুৱ পূর্ণাঙ্ক fused = (পূর্ণাঙ্ক) idx[2];
            ধ্রুৱ পূর্ণাঙ্ক i1 = fused / ne0;
            ধ্রুৱ পূর্ণাঙ্ক i0 = fused - i1 * ne0;  // fused % ne0


            ধ্রুৱ পূর্ণাঙ্ক idx_dst = i0
                              + i1 * stride1
                              + i2 * stride2
                              + i3 * stride3;


            ধ্রুৱ পূর্ণাঙ্ক s0 = wrap_add(i0, shNe0, ne0);
            ধ্রুৱ পূর্ণাঙ্ক s1 = wrap_add(i1, shNe1, ne1);
            ধ্রুৱ পূর্ণাঙ্ক s2 = wrap_add(i2, shNe2, ne2);
            ধ্রুৱ পূর্ণাঙ্ক s3 = wrap_add(i3, shNe3, ne3);

            ধ্রুৱ পূর্ণাঙ্ক idx_src = s0
                              + s1 * stride1
                              + s2 * stride2
                              + s3 * stride3;

            dst_d[idx_dst] = src_d[idx_src];
        });
    });
}

শূন্য ggml_sycl_roll(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    ধ্রুৱ ggml_tensor *src = dst->src[0];
    GGML_ASSERT(src && src->type == GGML_TYPE_F32);

    // the kernel indexes in 32-bit
    GGML_ASSERT(dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3] <= INT32_MAX);
    ধ্রুৱ পূর্ণাঙ্ক ne0 = (পূর্ণাঙ্ক) dst->ne[0];
    ধ্রুৱ পূর্ণাঙ্ক ne1 = (পূর্ণাঙ্ক) dst->ne[1];
    ধ্রুৱ পূর্ণাঙ্ক ne2 = (পূর্ণাঙ্ক) dst->ne[2];
    ধ্রুৱ পূর্ণাঙ্ক ne3 = (পূর্ণাঙ্ক) dst->ne[3];

    ধ্রুৱ int32_t *params = (ধ্রুৱ int32_t *) dst->op_params;
    পূর্ণাঙ্ক shift0 = params[0];
    পূর্ণাঙ্ক shift1 = params[1];
    পূর্ণাঙ্ক shift2 = params[2];
    পূর্ণাঙ্ক shift3 = params[3];


    যদি ((shift0 | shift1 | shift2 | shift3) == 0) {
        ধ্রুৱ size_t nb = ggml_nbytes(src);
        queue *q = ctx.stream();
        SYCL_CHECK(CHECK_TRY_ERROR(q->memcpy(dst->data, src->data, nb)));
        ঘূৰাই;
    }

    auto norm = [](পূর্ণাঙ্ক sh, পূর্ণাঙ্ক n) -> পূর্ণাঙ্ক {
        যদি (n <= 0) ঘূৰাই 0;
        sh %= n;
        যদি (sh < 0) sh += n;
        ঘূৰাই sh;
    };
    shift0 = norm(shift0, ne0);
    shift1 = norm(shift1, ne1);
    shift2 = norm(shift2, ne2);
    shift3 = norm(shift3, ne3);

    যত্ন {
        queue *q = ctx.stream();

        ধ্রুৱ ভিন্ন *src_d = (ধ্রুৱ ভিন্ন *) src->data;
        ভিন্ন *dst_d = (ভিন্ন *) dst->data;
        GGML_ASSERT(src_d && dst_d);

        kernel_roll_fused_i0_i1(
            *q, src_d, dst_d,
            ne0, ne1, ne2, ne3,
            shift0, shift1, shift2, shift3
        );
    } গৃহাণ (ধ্রুৱ std::exception &e) {
        std::fprintf(stderr, "[SYCL-ROLL] ERROR: %s\n", e.what());
        ক্ষিপ;
    }
}
