#அந்தர்பாவ "roll.hpp"
#அந்தர்பாவ "common.hpp"

using நாமாகாஶ sycl;

நிலையான உள்வரி முழு wrap_add(முழு i, முழு shift, முழு n) {

    முழு s = i + shift;
    திரும்பு (s >= n) ? (s - n) : s;
}

நிலையான வெறுமை kernel_roll_fused_i0_i1(
    queue &q,
    மாறிலி தசம *src_d,
    தசம *dst_d,
    முழு ne0, முழு ne1, முழு ne2, முழு ne3,
    முழு sh0, முழு sh1, முழு sh2, முழு sh3)
{
    எனில் (ne0 == 0 || ne1 == 0 || ne2 == 0 || ne3 == 0) திரும்பு;


    மாறிலி முழு stride1 = ne0;
    மாறிலி முழு stride2 = ne0 * ne1;
    மாறிலி முழு stride3 = ne0 * ne1 * ne2;


    மாறிலி முழு shNe0 = (ne0 - sh0) % ne0;
    மாறிலி முழு shNe1 = (ne1 - sh1) % ne1;
    மாறிலி முழு shNe2 = (ne2 - sh2) % ne2;
    மாறிலி முழு shNe3 = (ne3 - sh3) % ne3;


    மாறிலி size_t g0 = (size_t) ne3;
    மாறிலி size_t g1 = (size_t) ne2;
    மாறிலி size_t g2 = (size_t) (ne1 * ne0);

    மாறிலி range<3> global{ g0, g1, g2 };

    q.submit([&](handler &h) {
        h.parallel_for(global, [=](id<3> idx) {
            மாறிலி முழு i3 = (முழு) idx[0];
            மாறிலி முழு i2 = (முழு) idx[1];

            மாறிலி முழு fused = (முழு) idx[2];
            மாறிலி முழு i1 = fused / ne0;
            மாறிலி முழு i0 = fused - i1 * ne0;  // fused % ne0


            மாறிலி முழு idx_dst = i0
                              + i1 * stride1
                              + i2 * stride2
                              + i3 * stride3;


            மாறிலி முழு s0 = wrap_add(i0, shNe0, ne0);
            மாறிலி முழு s1 = wrap_add(i1, shNe1, ne1);
            மாறிலி முழு s2 = wrap_add(i2, shNe2, ne2);
            மாறிலி முழு s3 = wrap_add(i3, shNe3, ne3);

            மாறிலி முழு idx_src = s0
                              + s1 * stride1
                              + s2 * stride2
                              + s3 * stride3;

            dst_d[idx_dst] = src_d[idx_src];
        });
    });
}

வெறுமை ggml_sycl_roll(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    மாறிலி ggml_tensor *src = dst->src[0];
    GGML_ASSERT(src && src->type == GGML_TYPE_F32);

    // the kernel indexes in 32-bit
    GGML_ASSERT(dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3] <= INT32_MAX);
    மாறிலி முழு ne0 = (முழு) dst->ne[0];
    மாறிலி முழு ne1 = (முழு) dst->ne[1];
    மாறிலி முழு ne2 = (முழு) dst->ne[2];
    மாறிலி முழு ne3 = (முழு) dst->ne[3];

    மாறிலி int32_t *params = (மாறிலி int32_t *) dst->op_params;
    முழு shift0 = params[0];
    முழு shift1 = params[1];
    முழு shift2 = params[2];
    முழு shift3 = params[3];


    எனில் ((shift0 | shift1 | shift2 | shift3) == 0) {
        மாறிலி size_t nb = ggml_nbytes(src);
        queue *q = ctx.stream();
        SYCL_CHECK(CHECK_TRY_ERROR(q->memcpy(dst->data, src->data, nb)));
        திரும்பு;
    }

    auto norm = [](முழு sh, முழு n) -> முழு {
        எனில் (n <= 0) திரும்பு 0;
        sh %= n;
        எனில் (sh < 0) sh += n;
        திரும்பு sh;
    };
    shift0 = norm(shift0, ne0);
    shift1 = norm(shift1, ne1);
    shift2 = norm(shift2, ne2);
    shift3 = norm(shift3, ne3);

    முயல் {
        queue *q = ctx.stream();

        மாறிலி தசம *src_d = (மாறிலி தசம *) src->data;
        தசம *dst_d = (தசம *) dst->data;
        GGML_ASSERT(src_d && dst_d);

        kernel_roll_fused_i0_i1(
            *q, src_d, dst_d,
            ne0, ne1, ne2, ne3,
            shift0, shift1, shift2, shift3
        );
    } பிடி (மாறிலி std::exception &e) {
        std::fprintf(stderr, "[SYCL-ROLL] ERROR: %s\n", e.what());
        எறி;
    }
}
