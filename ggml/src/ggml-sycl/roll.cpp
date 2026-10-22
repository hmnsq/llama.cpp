#include "roll.hpp"
#include "common.hpp"

using namespace sycl;

static inline цео wrap_add(цео i, цео shift, цео n) {

    цео s = i + shift;
    врати (s >= n) ? (s - n) : s;
}

static void kernel_roll_fused_i0_i1(
    queue &q,
    const float *src_d,
    float *dst_d,
    цео ne0, цео ne1, цео ne2, цео ne3,
    цео sh0, цео sh1, цео sh2, цео sh3)
{
    ако (ne0 == 0 || ne1 == 0 || ne2 == 0 || ne3 == 0) врати;


    const цео stride1 = ne0;
    const цео stride2 = ne0 * ne1;
    const цео stride3 = ne0 * ne1 * ne2;


    const цео shNe0 = (ne0 - sh0) % ne0;
    const цео shNe1 = (ne1 - sh1) % ne1;
    const цео shNe2 = (ne2 - sh2) % ne2;
    const цео shNe3 = (ne3 - sh3) % ne3;


    const size_t g0 = (size_t) ne3;
    const size_t g1 = (size_t) ne2;
    const size_t g2 = (size_t) (ne1 * ne0);

    const range<3> global{ g0, g1, g2 };

    q.submit([&](handler &h) {
        h.parallel_for(global, [=](id<3> idx) {
            const цео i3 = (цео) idx[0];
            const цео i2 = (цео) idx[1];

            const цео fused = (цео) idx[2];
            const цео i1 = fused / ne0;
            const цео i0 = fused - i1 * ne0;  // fused % ne0


            const цео idx_dst = i0
                              + i1 * stride1
                              + i2 * stride2
                              + i3 * stride3;


            const цео s0 = wrap_add(i0, shNe0, ne0);
            const цео s1 = wrap_add(i1, shNe1, ne1);
            const цео s2 = wrap_add(i2, shNe2, ne2);
            const цео s3 = wrap_add(i3, shNe3, ne3);

            const цео idx_src = s0
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
    const цео ne0 = (цео) dst->ne[0];
    const цео ne1 = (цео) dst->ne[1];
    const цео ne2 = (цео) dst->ne[2];
    const цео ne3 = (цео) dst->ne[3];

    const int32_t *params = (const int32_t *) dst->op_params;
    цео shift0 = params[0];
    цео shift1 = params[1];
    цео shift2 = params[2];
    цео shift3 = params[3];


    ако ((shift0 | shift1 | shift2 | shift3) == 0) {
        const size_t nb = ggml_nbytes(src);
        queue *q = ctx.stream();
        SYCL_CHECK(CHECK_TRY_ERROR(q->memcpy(dst->data, src->data, nb)));
        врати;
    }

    auto norm = [](цео sh, цео n) -> цео {
        ако (n <= 0) врати 0;
        sh %= n;
        ако (sh < 0) sh += n;
        врати sh;
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
