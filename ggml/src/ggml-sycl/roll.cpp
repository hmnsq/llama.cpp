#include "roll.hpp"
#include "common.hpp"

using namespace sycl;

static anta_hstha පූර්ණ_සංඛ්‍යා wrap_add(පූර්ණ_සංඛ්‍යා i, පූර්ණ_සංඛ්‍යා shift, පූර්ණ_සංඛ්‍යා n) {

    පූර්ණ_සංඛ්‍යා s = i + shift;
    ආපසු (s >= n) ? (s - n) : s;
}

static void kernel_roll_fused_i0_i1(
    queue &q,
    const float *src_d,
    float *dst_d,
    පූර්ණ_සංඛ්‍යා ne0, පූර්ණ_සංඛ්‍යා ne1, පූර්ණ_සංඛ්‍යා ne2, පූර්ණ_සංඛ්‍යා ne3,
    පූර්ණ_සංඛ්‍යා sh0, පූර්ණ_සංඛ්‍යා sh1, පූර්ණ_සංඛ්‍යා sh2, පූර්ණ_සංඛ්‍යා sh3)
{
    නම් (ne0 == 0 || ne1 == 0 || ne2 == 0 || ne3 == 0) ආපසු;


    const පූර්ණ_සංඛ්‍යා stride1 = ne0;
    const පූර්ණ_සංඛ්‍යා stride2 = ne0 * ne1;
    const පූර්ණ_සංඛ්‍යා stride3 = ne0 * ne1 * ne2;


    const පූර්ණ_සංඛ්‍යා shNe0 = (ne0 - sh0) % ne0;
    const පූර්ණ_සංඛ්‍යා shNe1 = (ne1 - sh1) % ne1;
    const පූර්ණ_සංඛ්‍යා shNe2 = (ne2 - sh2) % ne2;
    const පූර්ණ_සංඛ්‍යා shNe3 = (ne3 - sh3) % ne3;


    const size_t g0 = (size_t) ne3;
    const size_t g1 = (size_t) ne2;
    const size_t g2 = (size_t) (ne1 * ne0);

    const range<3> global{ g0, g1, g2 };

    q.submit([&](handler &h) {
        h.parallel_for(global, [=](id<3> idx) {
            const පූර්ණ_සංඛ්‍යා i3 = (පූර්ණ_සංඛ්‍යා) idx[0];
            const පූර්ණ_සංඛ්‍යා i2 = (පූර්ණ_සංඛ්‍යා) idx[1];

            const පූර්ණ_සංඛ්‍යා fused = (පූර්ණ_සංඛ්‍යා) idx[2];
            const පූර්ණ_සංඛ්‍යා i1 = fused / ne0;
            const පූර්ණ_සංඛ්‍යා i0 = fused - i1 * ne0;  // fused % ne0


            const පූර්ණ_සංඛ්‍යා idx_dst = i0
                              + i1 * stride1
                              + i2 * stride2
                              + i3 * stride3;


            const පූර්ණ_සංඛ්‍යා s0 = wrap_add(i0, shNe0, ne0);
            const පූර්ණ_සංඛ්‍යා s1 = wrap_add(i1, shNe1, ne1);
            const පූර්ණ_සංඛ්‍යා s2 = wrap_add(i2, shNe2, ne2);
            const පූර්ණ_සංඛ්‍යා s3 = wrap_add(i3, shNe3, ne3);

            const පූර්ණ_සංඛ්‍යා idx_src = s0
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
    const පූර්ණ_සංඛ්‍යා ne0 = (පූර්ණ_සංඛ්‍යා) dst->ne[0];
    const පූර්ණ_සංඛ්‍යා ne1 = (පූර්ණ_සංඛ්‍යා) dst->ne[1];
    const පූර්ණ_සංඛ්‍යා ne2 = (පූර්ණ_සංඛ්‍යා) dst->ne[2];
    const පූර්ණ_සංඛ්‍යා ne3 = (පූර්ණ_සංඛ්‍යා) dst->ne[3];

    const int32_t *params = (const int32_t *) dst->op_params;
    පූර්ණ_සංඛ්‍යා shift0 = params[0];
    පූර්ණ_සංඛ්‍යා shift1 = params[1];
    පූර්ණ_සංඛ්‍යා shift2 = params[2];
    පූර්ණ_සංඛ්‍යා shift3 = params[3];


    නම් ((shift0 | shift1 | shift2 | shift3) == 0) {
        const size_t nb = ggml_nbytes(src);
        queue *q = ctx.stream();
        SYCL_CHECK(CHECK_TRY_ERROR(q->memcpy(dst->data, src->data, nb)));
        ආපසු;
    }

    auto norm = [](පූර්ණ_සංඛ්‍යා sh, පූර්ණ_සංඛ්‍යා n) -> පූර්ණ_සංඛ්‍යා {
        නම් (n <= 0) ආපසු 0;
        sh %= n;
        නම් (sh < 0) sh += n;
        ආපසු sh;
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
