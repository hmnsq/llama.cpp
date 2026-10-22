#అన్తర్భావ "roll.hpp"
#అన్తర్భావ "common.hpp"

ఉపయోగించి నామాకాశ sycl;

స్థిర అంతఃవరుస పూర్ణాంకం wrap_కూడు(పూర్ణాంకం i, పూర్ణాంకం జరుపు, పూర్ణాంకం n) {

    పూర్ణాంకం s = i + జరుపు;
    తిరిగి (s >= n) ? (s - n) : s;
}

స్థిర శూన్య kernel_roll_fused_i0_i1(
    queue &q,
    స్థిరరాశి దశాంశం *src_d,
    దశాంశం *dst_d,
    పూర్ణాంకం ne0, పూర్ణాంకం ne1, పూర్ణాంకం ne2, పూర్ణాంకం ne3,
    పూర్ణాంకం sh0, పూర్ణాంకం sh1, పూర్ణాంకం sh2, పూర్ణాంకం sh3)
{
    ఒకవేళ (ne0 == 0 || ne1 == 0 || ne2 == 0 || ne3 == 0) తిరిగి;


    స్థిరరాశి పూర్ణాంకం stride1 = ne0;
    స్థిరరాశి పూర్ణాంకం stride2 = ne0 * ne1;
    స్థిరరాశి పూర్ణాంకం stride3 = ne0 * ne1 * ne2;


    స్థిరరాశి పూర్ణాంకం shNe0 = (ne0 - sh0) % ne0;
    స్థిరరాశి పూర్ణాంకం shNe1 = (ne1 - sh1) % ne1;
    స్థిరరాశి పూర్ణాంకం shNe2 = (ne2 - sh2) % ne2;
    స్థిరరాశి పూర్ణాంకం shNe3 = (ne3 - sh3) % ne3;


    స్థిరరాశి size_t g0 = (size_t) ne3;
    స్థిరరాశి size_t g1 = (size_t) ne2;
    స్థిరరాశి size_t g2 = (size_t) (ne1 * ne0);

    స్థిరరాశి range<3> global{ g0, g1, g2 };

    q.submit([&](handler &h) {
        h.parallel_for(global, [=](id<3> idx) {
            స్థిరరాశి పూర్ణాంకం i3 = (పూర్ణాంకం) idx[0];
            స్థిరరాశి పూర్ణాంకం i2 = (పూర్ణాంకం) idx[1];

            స్థిరరాశి పూర్ణాంకం fused = (పూర్ణాంకం) idx[2];
            స్థిరరాశి పూర్ణాంకం i1 = fused / ne0;
            స్థిరరాశి పూర్ణాంకం i0 = fused - i1 * ne0;  // fused % ne0


            స్థిరరాశి పూర్ణాంకం idx_dst = i0
                              + i1 * stride1
                              + i2 * stride2
                              + i3 * stride3;


            స్థిరరాశి పూర్ణాంకం s0 = wrap_కూడు(i0, shNe0, ne0);
            స్థిరరాశి పూర్ణాంకం s1 = wrap_కూడు(i1, shNe1, ne1);
            స్థిరరాశి పూర్ణాంకం s2 = wrap_కూడు(i2, shNe2, ne2);
            స్థిరరాశి పూర్ణాంకం s3 = wrap_కూడు(i3, shNe3, ne3);

            స్థిరరాశి పూర్ణాంకం idx_src = s0
                              + s1 * stride1
                              + s2 * stride2
                              + s3 * stride3;

            dst_d[idx_dst] = src_d[idx_src];
        });
    });
}

శూన్య ggml_sycl_roll(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    స్థిరరాశి ggml_tensor *src = dst->src[0];
    GGML_ASSERT(src && src->type == GGML_TYPE_F32);

    // the kernel indexes in 32-bit
    GGML_ASSERT(dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3] <= INT32_MAX);
    స్థిరరాశి పూర్ణాంకం ne0 = (పూర్ణాంకం) dst->ne[0];
    స్థిరరాశి పూర్ణాంకం ne1 = (పూర్ణాంకం) dst->ne[1];
    స్థిరరాశి పూర్ణాంకం ne2 = (పూర్ణాంకం) dst->ne[2];
    స్థిరరాశి పూర్ణాంకం ne3 = (పూర్ణాంకం) dst->ne[3];

    స్థిరరాశి int32_t *params = (స్థిరరాశి int32_t *) dst->op_params;
    పూర్ణాంకం shift0 = params[0];
    పూర్ణాంకం shift1 = params[1];
    పూర్ణాంకం shift2 = params[2];
    పూర్ణాంకం shift3 = params[3];


    ఒకవేళ ((shift0 | shift1 | shift2 | shift3) == 0) {
        స్థిరరాశి size_t nb = ggml_nbytes(src);
        queue *q = ctx.stream();
        SYCL_CHECK(CHECK_TRY_ERROR(q->జ్ఞాపకం_నకలు(dst->దత్తాంశం, src->దత్తాంశం, nb)));
        తిరిగి;
    }

    స్వయంచాలక norm = [](పూర్ణాంకం sh, పూర్ణాంకం n) -> పూర్ణాంకం {
        ఒకవేళ (n <= 0) తిరిగి 0;
        sh %= n;
        ఒకవేళ (sh < 0) sh += n;
        తిరిగి sh;
    };
    shift0 = norm(shift0, ne0);
    shift1 = norm(shift1, ne1);
    shift2 = norm(shift2, ne2);
    shift3 = norm(shift3, ne3);

    ప్రయత్నించు {
        queue *q = ctx.stream();

        స్థిరరాశి దశాంశం *src_d = (స్థిరరాశి దశాంశం *) src->దత్తాంశం;
        దశాంశం *dst_d = (దశాంశం *) dst->దత్తాంశం;
        GGML_ASSERT(src_d && dst_d);

        kernel_roll_fused_i0_i1(
            *q, src_d, dst_d,
            ne0, ne1, ne2, ne3,
            shift0, shift1, shift2, shift3
        );
    } పట్టుకో (స్థిరరాశి std::exception &e) {
        std::fprintf(stderr, "[SYCL-ROLL] ERROR: %s\n", e.what());
        విసురు;
    }
}
