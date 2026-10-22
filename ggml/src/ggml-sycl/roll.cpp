#include "roll.hpp"
#include "common.hpp"

using namespace sycl;

static inline цяло wrap_add(цяло i, цяло shift, цяло n) {

    цяло s = i + shift;
    върни (s >= n) ? (s - n) : s;
}

static void kernel_roll_fused_i0_i1(
    queue &q,
    const float *src_d,
    float *dst_d,
    цяло ne0, цяло ne1, цяло ne2, цяло ne3,
    цяло sh0, цяло sh1, цяло sh2, цяло sh3)
{
    ако (ne0 == 0 || ne1 == 0 || ne2 == 0 || ne3 == 0) върни;


    const цяло stride1 = ne0;
    const цяло stride2 = ne0 * ne1;
    const цяло stride3 = ne0 * ne1 * ne2;


    const цяло shNe0 = (ne0 - sh0) % ne0;
    const цяло shNe1 = (ne1 - sh1) % ne1;
    const цяло shNe2 = (ne2 - sh2) % ne2;
    const цяло shNe3 = (ne3 - sh3) % ne3;


    const size_t g0 = (size_t) ne3;
    const size_t g1 = (size_t) ne2;
    const size_t g2 = (size_t) (ne1 * ne0);

    const range<3> global{ g0, g1, g2 };

    q.submit([&](handler &h) {
        h.parallel_for(global, [=](id<3> idx) {
            const цяло i3 = (цяло) idx[0];
            const цяло i2 = (цяло) idx[1];

            const цяло fused = (цяло) idx[2];
            const цяло i1 = fused / ne0;
            const цяло i0 = fused - i1 * ne0;  // fused % ne0


            const цяло idx_dst = i0
                              + i1 * stride1
                              + i2 * stride2
                              + i3 * stride3;


            const цяло s0 = wrap_add(i0, shNe0, ne0);
            const цяло s1 = wrap_add(i1, shNe1, ne1);
            const цяло s2 = wrap_add(i2, shNe2, ne2);
            const цяло s3 = wrap_add(i3, shNe3, ne3);

            const цяло idx_src = s0
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
    const цяло ne0 = (цяло) dst->ne[0];
    const цяло ne1 = (цяло) dst->ne[1];
    const цяло ne2 = (цяло) dst->ne[2];
    const цяло ne3 = (цяло) dst->ne[3];

    const int32_t *params = (const int32_t *) dst->op_params;
    цяло shift0 = params[0];
    цяло shift1 = params[1];
    цяло shift2 = params[2];
    цяло shift3 = params[3];


    ако ((shift0 | shift1 | shift2 | shift3) == 0) {
        const size_t nb = ggml_nbytes(src);
        queue *q = ctx.stream();
        SYCL_CHECK(CHECK_TRY_ERROR(q->memcpy(dst->data, src->data, nb)));
        върни;
    }

    auto norm = [](цяло sh, цяло n) -> цяло {
        ако (n <= 0) върни 0;
        sh %= n;
        ако (sh < 0) sh += n;
        върни sh;
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
    } catch (const std::exception &e) {
        std::fprintf(stderr, "[SYCL-ROLL] ERROR: %s\n", e.what());
        throw;
    }
}
