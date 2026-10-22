#include "roll.hpp"
#include "common.hpp"

using namespace sycl;

static inline số_nguyên wrap_add(số_nguyên i, số_nguyên shift, số_nguyên n) {

    số_nguyên s = i + shift;
    trả_về (s >= n) ? (s - n) : s;
}

static void kernel_roll_fused_i0_i1(
    queue &q,
    const float *src_d,
    float *dst_d,
    số_nguyên ne0, số_nguyên ne1, số_nguyên ne2, số_nguyên ne3,
    số_nguyên sh0, số_nguyên sh1, số_nguyên sh2, số_nguyên sh3)
{
    nếu (ne0 == 0 || ne1 == 0 || ne2 == 0 || ne3 == 0) trả_về;


    const số_nguyên stride1 = ne0;
    const số_nguyên stride2 = ne0 * ne1;
    const số_nguyên stride3 = ne0 * ne1 * ne2;


    const số_nguyên shNe0 = (ne0 - sh0) % ne0;
    const số_nguyên shNe1 = (ne1 - sh1) % ne1;
    const số_nguyên shNe2 = (ne2 - sh2) % ne2;
    const số_nguyên shNe3 = (ne3 - sh3) % ne3;


    const size_t g0 = (size_t) ne3;
    const size_t g1 = (size_t) ne2;
    const size_t g2 = (size_t) (ne1 * ne0);

    const range<3> global{ g0, g1, g2 };

    q.submit([&](handler &h) {
        h.parallel_for(global, [=](id<3> idx) {
            const số_nguyên i3 = (số_nguyên) idx[0];
            const số_nguyên i2 = (số_nguyên) idx[1];

            const số_nguyên fused = (số_nguyên) idx[2];
            const số_nguyên i1 = fused / ne0;
            const số_nguyên i0 = fused - i1 * ne0;  // fused % ne0


            const số_nguyên idx_dst = i0
                              + i1 * stride1
                              + i2 * stride2
                              + i3 * stride3;


            const số_nguyên s0 = wrap_add(i0, shNe0, ne0);
            const số_nguyên s1 = wrap_add(i1, shNe1, ne1);
            const số_nguyên s2 = wrap_add(i2, shNe2, ne2);
            const số_nguyên s3 = wrap_add(i3, shNe3, ne3);

            const số_nguyên idx_src = s0
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
    const số_nguyên ne0 = (số_nguyên) dst->ne[0];
    const số_nguyên ne1 = (số_nguyên) dst->ne[1];
    const số_nguyên ne2 = (số_nguyên) dst->ne[2];
    const số_nguyên ne3 = (số_nguyên) dst->ne[3];

    const int32_t *params = (const int32_t *) dst->op_params;
    số_nguyên shift0 = params[0];
    số_nguyên shift1 = params[1];
    số_nguyên shift2 = params[2];
    số_nguyên shift3 = params[3];


    nếu ((shift0 | shift1 | shift2 | shift3) == 0) {
        const size_t nb = ggml_nbytes(src);
        queue *q = ctx.stream();
        SYCL_CHECK(CHECK_TRY_ERROR(q->memcpy(dst->data, src->data, nb)));
        trả_về;
    }

    auto norm = [](số_nguyên sh, số_nguyên n) -> số_nguyên {
        nếu (n <= 0) trả_về 0;
        sh %= n;
        nếu (sh < 0) sh += n;
        trả_về sh;
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
