#ਅਨ੍ਤਰ੍ਭਾਵ "roll.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "common.hpp"

ਵਰਤੋਂ ਨਾਮਾਕਾਸ਼ sycl;

ਸਥਿਰ ਸਤਰਅੰਦਰ ਪੂਰਨ wrap_ਜੋੜੋ(ਪੂਰਨ i, ਪੂਰਨ ਖਿਸਕਾਓ, ਪੂਰਨ n) {

    ਪੂਰਨ s = i + ਖਿਸਕਾਓ;
    ਵਾਪਸ (s >= n) ? (s - n) : s;
}

ਸਥਿਰ ਖਾਲੀ kernel_roll_fused_i0_i1(
    queue &q,
    ਅਚਲ ਦਸ਼ਮਲਵ *src_d,
    ਦਸ਼ਮਲਵ *dst_d,
    ਪੂਰਨ ne0, ਪੂਰਨ ne1, ਪੂਰਨ ne2, ਪੂਰਨ ne3,
    ਪੂਰਨ sh0, ਪੂਰਨ sh1, ਪੂਰਨ sh2, ਪੂਰਨ sh3)
{
    ਜੇ (ne0 == 0 || ne1 == 0 || ne2 == 0 || ne3 == 0) ਵਾਪਸ;


    ਅਚਲ ਪੂਰਨ stride1 = ne0;
    ਅਚਲ ਪੂਰਨ stride2 = ne0 * ne1;
    ਅਚਲ ਪੂਰਨ stride3 = ne0 * ne1 * ne2;


    ਅਚਲ ਪੂਰਨ shNe0 = (ne0 - sh0) % ne0;
    ਅਚਲ ਪੂਰਨ shNe1 = (ne1 - sh1) % ne1;
    ਅਚਲ ਪੂਰਨ shNe2 = (ne2 - sh2) % ne2;
    ਅਚਲ ਪੂਰਨ shNe3 = (ne3 - sh3) % ne3;


    ਅਚਲ size_t g0 = (size_t) ne3;
    ਅਚਲ size_t g1 = (size_t) ne2;
    ਅਚਲ size_t g2 = (size_t) (ne1 * ne0);

    ਅਚਲ range<3> global{ g0, g1, g2 };

    q.submit([&](handler &h) {
        h.parallel_for(global, [=](id<3> idx) {
            ਅਚਲ ਪੂਰਨ i3 = (ਪੂਰਨ) idx[0];
            ਅਚਲ ਪੂਰਨ i2 = (ਪੂਰਨ) idx[1];

            ਅਚਲ ਪੂਰਨ fused = (ਪੂਰਨ) idx[2];
            ਅਚਲ ਪੂਰਨ i1 = fused / ne0;
            ਅਚਲ ਪੂਰਨ i0 = fused - i1 * ne0;  // fused % ne0


            ਅਚਲ ਪੂਰਨ idx_dst = i0
                              + i1 * stride1
                              + i2 * stride2
                              + i3 * stride3;


            ਅਚਲ ਪੂਰਨ s0 = wrap_ਜੋੜੋ(i0, shNe0, ne0);
            ਅਚਲ ਪੂਰਨ s1 = wrap_ਜੋੜੋ(i1, shNe1, ne1);
            ਅਚਲ ਪੂਰਨ s2 = wrap_ਜੋੜੋ(i2, shNe2, ne2);
            ਅਚਲ ਪੂਰਨ s3 = wrap_ਜੋੜੋ(i3, shNe3, ne3);

            ਅਚਲ ਪੂਰਨ idx_src = s0
                              + s1 * stride1
                              + s2 * stride2
                              + s3 * stride3;

            dst_d[idx_dst] = src_d[idx_src];
        });
    });
}

ਖਾਲੀ ggml_sycl_roll(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    ਅਚਲ ggml_tensor *src = dst->src[0];
    GGML_ASSERT(src && src->type == GGML_TYPE_F32);

    // the kernel indexes in 32-bit
    GGML_ASSERT(dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3] <= INT32_MAX);
    ਅਚਲ ਪੂਰਨ ne0 = (ਪੂਰਨ) dst->ne[0];
    ਅਚਲ ਪੂਰਨ ne1 = (ਪੂਰਨ) dst->ne[1];
    ਅਚਲ ਪੂਰਨ ne2 = (ਪੂਰਨ) dst->ne[2];
    ਅਚਲ ਪੂਰਨ ne3 = (ਪੂਰਨ) dst->ne[3];

    ਅਚਲ int32_t *params = (ਅਚਲ int32_t *) dst->op_params;
    ਪੂਰਨ shift0 = params[0];
    ਪੂਰਨ shift1 = params[1];
    ਪੂਰਨ shift2 = params[2];
    ਪੂਰਨ shift3 = params[3];


    ਜੇ ((shift0 | shift1 | shift2 | shift3) == 0) {
        ਅਚਲ size_t nb = ggml_nbytes(src);
        queue *q = ctx.stream();
        SYCL_CHECK(CHECK_TRY_ERROR(q->ਸਮ੍ਰਿਤੀ_ਨਕਲ(dst->ਅੰਕੜੇ, src->ਅੰਕੜੇ, nb)));
        ਵਾਪਸ;
    }

    ਸਵੈਚਾਲਿਤ norm = [](ਪੂਰਨ sh, ਪੂਰਨ n) -> ਪੂਰਨ {
        ਜੇ (n <= 0) ਵਾਪਸ 0;
        sh %= n;
        ਜੇ (sh < 0) sh += n;
        ਵਾਪਸ sh;
    };
    shift0 = norm(shift0, ne0);
    shift1 = norm(shift1, ne1);
    shift2 = norm(shift2, ne2);
    shift3 = norm(shift3, ne3);

    ਕੋਸ਼ਿਸ਼ {
        queue *q = ctx.stream();

        ਅਚਲ ਦਸ਼ਮਲਵ *src_d = (ਅਚਲ ਦਸ਼ਮਲਵ *) src->ਅੰਕੜੇ;
        ਦਸ਼ਮਲਵ *dst_d = (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ;
        GGML_ASSERT(src_d && dst_d);

        kernel_roll_fused_i0_i1(
            *q, src_d, dst_d,
            ne0, ne1, ne2, ne3,
            shift0, shift1, shift2, shift3
        );
    } ਫੜੋ (ਅਚਲ std::exception &e) {
        std::fprintf(stderr, "[SYCL-ROLL] ERROR: %s\n", e.what());
        ਸੁੱਟੋ;
    }
}
