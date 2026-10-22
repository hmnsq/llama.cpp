#ଅନ୍ତର୍ଭାଵ "roll.hpp"
#ଅନ୍ତର୍ଭାଵ "common.hpp"

using ନାମାକାଶ sycl;

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ପୂର୍ଣ୍ଣସଂଖ୍ୟା wrap_add(ପୂର୍ଣ୍ଣସଂଖ୍ୟା i, ପୂର୍ଣ୍ଣସଂଖ୍ୟା shift, ପୂର୍ଣ୍ଣସଂଖ୍ୟା n) {

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା s = i + shift;
    ଫେରନ୍ତୁ (s >= n) ? (s - n) : s;
}

ସ୍ଥିର ଶୂନ୍ୟ kernel_roll_fused_i0_i1(
    queue &q,
    ଅଚଳ ଦଶମିକ *src_d,
    ଦଶମିକ *dst_d,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne0, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne1, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne2, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne3,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା sh0, ପୂର୍ଣ୍ଣସଂଖ୍ୟା sh1, ପୂର୍ଣ୍ଣସଂଖ୍ୟା sh2, ପୂର୍ଣ୍ଣସଂଖ୍ୟା sh3)
{
    ଯଦି (ne0 == 0 || ne1 == 0 || ne2 == 0 || ne3 == 0) ଫେରନ୍ତୁ;


    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride1 = ne0;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride2 = ne0 * ne1;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride3 = ne0 * ne1 * ne2;


    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା shNe0 = (ne0 - sh0) % ne0;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା shNe1 = (ne1 - sh1) % ne1;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା shNe2 = (ne2 - sh2) % ne2;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା shNe3 = (ne3 - sh3) % ne3;


    ଅଚଳ size_t g0 = (size_t) ne3;
    ଅଚଳ size_t g1 = (size_t) ne2;
    ଅଚଳ size_t g2 = (size_t) (ne1 * ne0);

    ଅଚଳ range<3> global{ g0, g1, g2 };

    q.submit([&](handler &h) {
        h.parallel_for(global, [=](id<3> idx) {
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i3 = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) idx[0];
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i2 = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) idx[1];

            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା fused = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) idx[2];
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1 = fused / ne0;
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i0 = fused - i1 * ne0;  // fused % ne0


            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା idx_dst = i0
                              + i1 * stride1
                              + i2 * stride2
                              + i3 * stride3;


            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s0 = wrap_add(i0, shNe0, ne0);
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s1 = wrap_add(i1, shNe1, ne1);
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s2 = wrap_add(i2, shNe2, ne2);
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s3 = wrap_add(i3, shNe3, ne3);

            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା idx_src = s0
                              + s1 * stride1
                              + s2 * stride2
                              + s3 * stride3;

            dst_d[idx_dst] = src_d[idx_src];
        });
    });
}

ଶୂନ୍ୟ ggml_sycl_roll(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    ଅଚଳ ggml_tensor *src = dst->src[0];
    GGML_ASSERT(src && src->type == GGML_TYPE_F32);

    // the kernel indexes in 32-bit
    GGML_ASSERT(dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3] <= INT32_MAX);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne0 = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) dst->ne[0];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne1 = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) dst->ne[1];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne2 = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) dst->ne[2];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne3 = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) dst->ne[3];

    ଅଚଳ int32_t *params = (ଅଚଳ int32_t *) dst->op_params;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା shift0 = params[0];
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା shift1 = params[1];
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା shift2 = params[2];
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା shift3 = params[3];


    ଯଦି ((shift0 | shift1 | shift2 | shift3) == 0) {
        ଅଚଳ size_t nb = ggml_nbytes(src);
        queue *q = ctx.stream();
        SYCL_CHECK(CHECK_TRY_ERROR(q->memcpy(dst->data, src->data, nb)));
        ଫେରନ୍ତୁ;
    }

    auto norm = [](ପୂର୍ଣ୍ଣସଂଖ୍ୟା sh, ପୂର୍ଣ୍ଣସଂଖ୍ୟା n) -> ପୂର୍ଣ୍ଣସଂଖ୍ୟା {
        ଯଦି (n <= 0) ଫେରନ୍ତୁ 0;
        sh %= n;
        ଯଦି (sh < 0) sh += n;
        ଫେରନ୍ତୁ sh;
    };
    shift0 = norm(shift0, ne0);
    shift1 = norm(shift1, ne1);
    shift2 = norm(shift2, ne2);
    shift3 = norm(shift3, ne3);

    ଚେଷ୍ଟା {
        queue *q = ctx.stream();

        ଅଚଳ ଦଶମିକ *src_d = (ଅଚଳ ଦଶମିକ *) src->data;
        ଦଶମିକ *dst_d = (ଦଶମିକ *) dst->data;
        GGML_ASSERT(src_d && dst_d);

        kernel_roll_fused_i0_i1(
            *q, src_d, dst_d,
            ne0, ne1, ne2, ne3,
            shift0, shift1, shift2, shift3
        );
    } ଧର (ଅଚଳ std::exception &e) {
        std::fprintf(stderr, "[SYCL-ROLL] ERROR: %s\n", e.what());
        ଫିଙ୍ଗ;
    }
}
