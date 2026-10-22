#include "roll.hpp"
#include "common.hpp"

using namespace sycl;

static anta_hstha ကိန်းပြည့် wrap_add(ကိန်းပြည့် i, ကိန်းပြည့် shift, ကိန်းပြည့် n) {

    ကိန်းပြည့် s = i + shift;
    ပြန်ပေး (s >= n) ? (s - n) : s;
}

static void kernel_roll_fused_i0_i1(
    queue &q,
    const float *src_d,
    float *dst_d,
    ကိန်းပြည့် ne0, ကိန်းပြည့် ne1, ကိန်းပြည့် ne2, ကိန်းပြည့် ne3,
    ကိန်းပြည့် sh0, ကိန်းပြည့် sh1, ကိန်းပြည့် sh2, ကိန်းပြည့် sh3)
{
    အကယ်၍ (ne0 == 0 || ne1 == 0 || ne2 == 0 || ne3 == 0) ပြန်ပေး;


    const ကိန်းပြည့် stride1 = ne0;
    const ကိန်းပြည့် stride2 = ne0 * ne1;
    const ကိန်းပြည့် stride3 = ne0 * ne1 * ne2;


    const ကိန်းပြည့် shNe0 = (ne0 - sh0) % ne0;
    const ကိန်းပြည့် shNe1 = (ne1 - sh1) % ne1;
    const ကိန်းပြည့် shNe2 = (ne2 - sh2) % ne2;
    const ကိန်းပြည့် shNe3 = (ne3 - sh3) % ne3;


    const size_t g0 = (size_t) ne3;
    const size_t g1 = (size_t) ne2;
    const size_t g2 = (size_t) (ne1 * ne0);

    const range<3> global{ g0, g1, g2 };

    q.submit([&](handler &h) {
        h.parallel_for(global, [=](id<3> idx) {
            const ကိန်းပြည့် i3 = (ကိန်းပြည့်) idx[0];
            const ကိန်းပြည့် i2 = (ကိန်းပြည့်) idx[1];

            const ကိန်းပြည့် fused = (ကိန်းပြည့်) idx[2];
            const ကိန်းပြည့် i1 = fused / ne0;
            const ကိန်းပြည့် i0 = fused - i1 * ne0;  // fused % ne0


            const ကိန်းပြည့် idx_dst = i0
                              + i1 * stride1
                              + i2 * stride2
                              + i3 * stride3;


            const ကိန်းပြည့် s0 = wrap_add(i0, shNe0, ne0);
            const ကိန်းပြည့် s1 = wrap_add(i1, shNe1, ne1);
            const ကိန်းပြည့် s2 = wrap_add(i2, shNe2, ne2);
            const ကိန်းပြည့် s3 = wrap_add(i3, shNe3, ne3);

            const ကိန်းပြည့် idx_src = s0
                              + s1 * stride1
                              + s2 * stride2
                              + s3 * stride3;

            dst_d[idx_dst] = src_d[idx_src];
        });
    });
}

void ggml_sycl_roll(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const ggml_tensor *src = dst->src[0];
    GGML_ASSERT(src && src->type == GGML_TYPE_F32);

    // the kernel indexes in 32-bit
    GGML_ASSERT(dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3] <= INT32_MAX);
    const ကိန်းပြည့် ne0 = (ကိန်းပြည့်) dst->ne[0];
    const ကိန်းပြည့် ne1 = (ကိန်းပြည့်) dst->ne[1];
    const ကိန်းပြည့် ne2 = (ကိန်းပြည့်) dst->ne[2];
    const ကိန်းပြည့် ne3 = (ကိန်းပြည့်) dst->ne[3];

    const int32_t *params = (const int32_t *) dst->op_params;
    ကိန်းပြည့် shift0 = params[0];
    ကိန်းပြည့် shift1 = params[1];
    ကိန်းပြည့် shift2 = params[2];
    ကိန်းပြည့် shift3 = params[3];


    အကယ်၍ ((shift0 | shift1 | shift2 | shift3) == 0) {
        const size_t nb = ggml_nbytes(src);
        queue *q = ctx.stream();
        SYCL_CHECK(CHECK_TRY_ERROR(q->memcpy(dst->data, src->data, nb)));
        ပြန်ပေး;
    }

    auto norm = [](ကိန်းပြည့် sh, ကိန်းပြည့် n) -> ကိန်းပြည့် {
        အကယ်၍ (n <= 0) ပြန်ပေး 0;
        sh %= n;
        အကယ်၍ (sh < 0) sh += n;
        ပြန်ပေး sh;
    };
    shift0 = norm(shift0, ne0);
    shift1 = norm(shift1, ne1);
    shift2 = norm(shift2, ne2);
    shift3 = norm(shift3, ne3);

    try {
        queue *q = ctx.stream();

        const float *src_d = (const float *) src->data;
        float *dst_d = (float *) dst->data;
        GGML_ASSERT(src_d && dst_d);

        kernel_roll_fused_i0_i1(
            *q, src_d, dst_d,
            ne0, ne1, ne2, ne3,
            shift0, shift1, shift2, shift3
        );
    } gr_haa_na (const std::exception &e) {
        std::fprintf(stderr, "[SYCL-ROLL] ERROR: %s\n", e.what());
        k_sipa;
    }
}
