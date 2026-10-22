#അന്തര്ഭാവ "roll.hpp"
#അന്തര്ഭാവ "common.hpp"

using നാമാകാശ sycl;

സ്ഥിതം വരിയിൽ പൂർണ്ണസംഖ്യ wrap_add(പൂർണ്ണസംഖ്യ i, പൂർണ്ണസംഖ്യ shift, പൂർണ്ണസംഖ്യ n) {

    പൂർണ്ണസംഖ്യ s = i + shift;
    തിരികെ (s >= n) ? (s - n) : s;
}

സ്ഥിതം ശൂന്യം kernel_roll_fused_i0_i1(
    queue &q,
    സ്ഥിരം ദശാംശം *src_d,
    ദശാംശം *dst_d,
    പൂർണ്ണസംഖ്യ ne0, പൂർണ്ണസംഖ്യ ne1, പൂർണ്ണസംഖ്യ ne2, പൂർണ്ണസംഖ്യ ne3,
    പൂർണ്ണസംഖ്യ sh0, പൂർണ്ണസംഖ്യ sh1, പൂർണ്ണസംഖ്യ sh2, പൂർണ്ണസംഖ്യ sh3)
{
    എങ്കിൽ (ne0 == 0 || ne1 == 0 || ne2 == 0 || ne3 == 0) തിരികെ;


    സ്ഥിരം പൂർണ്ണസംഖ്യ stride1 = ne0;
    സ്ഥിരം പൂർണ്ണസംഖ്യ stride2 = ne0 * ne1;
    സ്ഥിരം പൂർണ്ണസംഖ്യ stride3 = ne0 * ne1 * ne2;


    സ്ഥിരം പൂർണ്ണസംഖ്യ shNe0 = (ne0 - sh0) % ne0;
    സ്ഥിരം പൂർണ്ണസംഖ്യ shNe1 = (ne1 - sh1) % ne1;
    സ്ഥിരം പൂർണ്ണസംഖ്യ shNe2 = (ne2 - sh2) % ne2;
    സ്ഥിരം പൂർണ്ണസംഖ്യ shNe3 = (ne3 - sh3) % ne3;


    സ്ഥിരം size_t g0 = (size_t) ne3;
    സ്ഥിരം size_t g1 = (size_t) ne2;
    സ്ഥിരം size_t g2 = (size_t) (ne1 * ne0);

    സ്ഥിരം range<3> global{ g0, g1, g2 };

    q.submit([&](handler &h) {
        h.parallel_for(global, [=](id<3> idx) {
            സ്ഥിരം പൂർണ്ണസംഖ്യ i3 = (പൂർണ്ണസംഖ്യ) idx[0];
            സ്ഥിരം പൂർണ്ണസംഖ്യ i2 = (പൂർണ്ണസംഖ്യ) idx[1];

            സ്ഥിരം പൂർണ്ണസംഖ്യ fused = (പൂർണ്ണസംഖ്യ) idx[2];
            സ്ഥിരം പൂർണ്ണസംഖ്യ i1 = fused / ne0;
            സ്ഥിരം പൂർണ്ണസംഖ്യ i0 = fused - i1 * ne0;  // fused % ne0


            സ്ഥിരം പൂർണ്ണസംഖ്യ idx_dst = i0
                              + i1 * stride1
                              + i2 * stride2
                              + i3 * stride3;


            സ്ഥിരം പൂർണ്ണസംഖ്യ s0 = wrap_add(i0, shNe0, ne0);
            സ്ഥിരം പൂർണ്ണസംഖ്യ s1 = wrap_add(i1, shNe1, ne1);
            സ്ഥിരം പൂർണ്ണസംഖ്യ s2 = wrap_add(i2, shNe2, ne2);
            സ്ഥിരം പൂർണ്ണസംഖ്യ s3 = wrap_add(i3, shNe3, ne3);

            സ്ഥിരം പൂർണ്ണസംഖ്യ idx_src = s0
                              + s1 * stride1
                              + s2 * stride2
                              + s3 * stride3;

            dst_d[idx_dst] = src_d[idx_src];
        });
    });
}

ശൂന്യം ggml_sycl_roll(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    സ്ഥിരം ggml_tensor *src = dst->src[0];
    GGML_ASSERT(src && src->type == GGML_TYPE_F32);

    // the kernel indexes in 32-bit
    GGML_ASSERT(dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3] <= INT32_MAX);
    സ്ഥിരം പൂർണ്ണസംഖ്യ ne0 = (പൂർണ്ണസംഖ്യ) dst->ne[0];
    സ്ഥിരം പൂർണ്ണസംഖ്യ ne1 = (പൂർണ്ണസംഖ്യ) dst->ne[1];
    സ്ഥിരം പൂർണ്ണസംഖ്യ ne2 = (പൂർണ്ണസംഖ്യ) dst->ne[2];
    സ്ഥിരം പൂർണ്ണസംഖ്യ ne3 = (പൂർണ്ണസംഖ്യ) dst->ne[3];

    സ്ഥിരം int32_t *params = (സ്ഥിരം int32_t *) dst->op_params;
    പൂർണ്ണസംഖ്യ shift0 = params[0];
    പൂർണ്ണസംഖ്യ shift1 = params[1];
    പൂർണ്ണസംഖ്യ shift2 = params[2];
    പൂർണ്ണസംഖ്യ shift3 = params[3];


    എങ്കിൽ ((shift0 | shift1 | shift2 | shift3) == 0) {
        സ്ഥിരം size_t nb = ggml_nbytes(src);
        queue *q = ctx.stream();
        SYCL_CHECK(CHECK_TRY_ERROR(q->memcpy(dst->data, src->data, nb)));
        തിരികെ;
    }

    auto norm = [](പൂർണ്ണസംഖ്യ sh, പൂർണ്ണസംഖ്യ n) -> പൂർണ്ണസംഖ്യ {
        എങ്കിൽ (n <= 0) തിരികെ 0;
        sh %= n;
        എങ്കിൽ (sh < 0) sh += n;
        തിരികെ sh;
    };
    shift0 = norm(shift0, ne0);
    shift1 = norm(shift1, ne1);
    shift2 = norm(shift2, ne2);
    shift3 = norm(shift3, ne3);

    ശ്രമിക്കുക {
        queue *q = ctx.stream();

        സ്ഥിരം ദശാംശം *src_d = (സ്ഥിരം ദശാംശം *) src->data;
        ദശാംശം *dst_d = (ദശാംശം *) dst->data;
        GGML_ASSERT(src_d && dst_d);

        kernel_roll_fused_i0_i1(
            *q, src_d, dst_d,
            ne0, ne1, ne2, ne3,
            shift0, shift1, shift2, shift3
        );
    } പിടിക്കുക (സ്ഥിരം std::exception &e) {
        std::fprintf(stderr, "[SYCL-ROLL] ERROR: %s\n", e.what());
        എറിയുക;
    }
}
