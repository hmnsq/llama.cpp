#અન્તર્ભાવ "roll.hpp"
#અન્તર્ભાવ "common.hpp"

using નામાકાશ sycl;

સ્થિર અંતઃરેખ પૂર્ણાંક wrap_add(પૂર્ણાંક i, પૂર્ણાંક shift, પૂર્ણાંક n) {

    પૂર્ણાંક s = i + shift;
    પાછા (s >= n) ? (s - n) : s;
}

સ્થિર શૂન્ય kernel_roll_fused_i0_i1(
    queue &q,
    અચલ દશાંશ *src_d,
    દશાંશ *dst_d,
    પૂર્ણાંક ne0, પૂર્ણાંક ne1, પૂર્ણાંક ne2, પૂર્ણાંક ne3,
    પૂર્ણાંક sh0, પૂર્ણાંક sh1, પૂર્ણાંક sh2, પૂર્ણાંક sh3)
{
    જો (ne0 == 0 || ne1 == 0 || ne2 == 0 || ne3 == 0) પાછા;


    અચલ પૂર્ણાંક stride1 = ne0;
    અચલ પૂર્ણાંક stride2 = ne0 * ne1;
    અચલ પૂર્ણાંક stride3 = ne0 * ne1 * ne2;


    અચલ પૂર્ણાંક shNe0 = (ne0 - sh0) % ne0;
    અચલ પૂર્ણાંક shNe1 = (ne1 - sh1) % ne1;
    અચલ પૂર્ણાંક shNe2 = (ne2 - sh2) % ne2;
    અચલ પૂર્ણાંક shNe3 = (ne3 - sh3) % ne3;


    અચલ size_t g0 = (size_t) ne3;
    અચલ size_t g1 = (size_t) ne2;
    અચલ size_t g2 = (size_t) (ne1 * ne0);

    અચલ range<3> global{ g0, g1, g2 };

    q.submit([&](handler &h) {
        h.parallel_for(global, [=](id<3> idx) {
            અચલ પૂર્ણાંક i3 = (પૂર્ણાંક) idx[0];
            અચલ પૂર્ણાંક i2 = (પૂર્ણાંક) idx[1];

            અચલ પૂર્ણાંક fused = (પૂર્ણાંક) idx[2];
            અચલ પૂર્ણાંક i1 = fused / ne0;
            અચલ પૂર્ણાંક i0 = fused - i1 * ne0;  // fused % ne0


            અચલ પૂર્ણાંક idx_dst = i0
                              + i1 * stride1
                              + i2 * stride2
                              + i3 * stride3;


            અચલ પૂર્ણાંક s0 = wrap_add(i0, shNe0, ne0);
            અચલ પૂર્ણાંક s1 = wrap_add(i1, shNe1, ne1);
            અચલ પૂર્ણાંક s2 = wrap_add(i2, shNe2, ne2);
            અચલ પૂર્ણાંક s3 = wrap_add(i3, shNe3, ne3);

            અચલ પૂર્ણાંક idx_src = s0
                              + s1 * stride1
                              + s2 * stride2
                              + s3 * stride3;

            dst_d[idx_dst] = src_d[idx_src];
        });
    });
}

શૂન્ય ggml_sycl_roll(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    અચલ ggml_tensor *src = dst->src[0];
    GGML_ASSERT(src && src->type == GGML_TYPE_F32);

    // the kernel indexes in 32-bit
    GGML_ASSERT(dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3] <= INT32_MAX);
    અચલ પૂર્ણાંક ne0 = (પૂર્ણાંક) dst->ne[0];
    અચલ પૂર્ણાંક ne1 = (પૂર્ણાંક) dst->ne[1];
    અચલ પૂર્ણાંક ne2 = (પૂર્ણાંક) dst->ne[2];
    અચલ પૂર્ણાંક ne3 = (પૂર્ણાંક) dst->ne[3];

    અચલ int32_t *params = (અચલ int32_t *) dst->op_params;
    પૂર્ણાંક shift0 = params[0];
    પૂર્ણાંક shift1 = params[1];
    પૂર્ણાંક shift2 = params[2];
    પૂર્ણાંક shift3 = params[3];


    જો ((shift0 | shift1 | shift2 | shift3) == 0) {
        અચલ size_t nb = ggml_nbytes(src);
        queue *q = ctx.stream();
        SYCL_CHECK(CHECK_TRY_ERROR(q->memcpy(dst->data, src->data, nb)));
        પાછા;
    }

    auto norm = [](પૂર્ણાંક sh, પૂર્ણાંક n) -> પૂર્ણાંક {
        જો (n <= 0) પાછા 0;
        sh %= n;
        જો (sh < 0) sh += n;
        પાછા sh;
    };
    shift0 = norm(shift0, ne0);
    shift1 = norm(shift1, ne1);
    shift2 = norm(shift2, ne2);
    shift3 = norm(shift3, ne3);

    પ્રયાસ {
        queue *q = ctx.stream();

        અચલ દશાંશ *src_d = (અચલ દશાંશ *) src->data;
        દશાંશ *dst_d = (દશાંશ *) dst->data;
        GGML_ASSERT(src_d && dst_d);

        kernel_roll_fused_i0_i1(
            *q, src_d, dst_d,
            ne0, ne1, ne2, ne3,
            shift0, shift1, shift2, shift3
        );
    } પકડો (અચલ std::exception &e) {
        std::fprintf(stderr, "[SYCL-ROLL] ERROR: %s\n", e.what());
        ફેંકો;
    }
}
