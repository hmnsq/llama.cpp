#include "roll.hpp"
#include "common.hpp"

using namespace sycl;

static inline целое wrap_add(целое i, целое shift, целое n) {

    целое s = i + shift;
    вернуть (s >= n) ? (s - n) : s;
}

static void kernel_roll_fused_i0_i1(
    queue &q,
    const float *src_d,
    float *dst_d,
    целое ne0, целое ne1, целое ne2, целое ne3,
    целое sh0, целое sh1, целое sh2, целое sh3)
{
    если (ne0 == 0 || ne1 == 0 || ne2 == 0 || ne3 == 0) вернуть;


    const целое stride1 = ne0;
    const целое stride2 = ne0 * ne1;
    const целое stride3 = ne0 * ne1 * ne2;


    const целое shNe0 = (ne0 - sh0) % ne0;
    const целое shNe1 = (ne1 - sh1) % ne1;
    const целое shNe2 = (ne2 - sh2) % ne2;
    const целое shNe3 = (ne3 - sh3) % ne3;


    const size_t g0 = (size_t) ne3;
    const size_t g1 = (size_t) ne2;
    const size_t g2 = (size_t) (ne1 * ne0);

    const range<3> global{ g0, g1, g2 };

    q.submit([&](handler &h) {
        h.parallel_for(global, [=](id<3> idx) {
            const целое i3 = (целое) idx[0];
            const целое i2 = (целое) idx[1];

            const целое fused = (целое) idx[2];
            const целое i1 = fused / ne0;
            const целое i0 = fused - i1 * ne0;  // fused % ne0


            const целое idx_dst = i0
                              + i1 * stride1
                              + i2 * stride2
                              + i3 * stride3;


            const целое s0 = wrap_add(i0, shNe0, ne0);
            const целое s1 = wrap_add(i1, shNe1, ne1);
            const целое s2 = wrap_add(i2, shNe2, ne2);
            const целое s3 = wrap_add(i3, shNe3, ne3);

            const целое idx_src = s0
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
    const целое ne0 = (целое) dst->ne[0];
    const целое ne1 = (целое) dst->ne[1];
    const целое ne2 = (целое) dst->ne[2];
    const целое ne3 = (целое) dst->ne[3];

    const int32_t *params = (const int32_t *) dst->op_params;
    целое shift0 = params[0];
    целое shift1 = params[1];
    целое shift2 = params[2];
    целое shift3 = params[3];


    если ((shift0 | shift1 | shift2 | shift3) == 0) {
        const size_t nb = ggml_nbytes(src);
        queue *q = ctx.stream();
        SYCL_CHECK(CHECK_TRY_ERROR(q->memcpy(dst->data, src->data, nb)));
        вернуть;
    }

    auto norm = [](целое sh, целое n) -> целое {
        если (n <= 0) вернуть 0;
        sh %= n;
        если (sh < 0) sh += n;
        вернуть sh;
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
