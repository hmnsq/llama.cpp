#include "roll.hpp"
#include "common.hpp"

using namespace sycl;

static inline 정수 wrap_add(정수 i, 정수 shift, 정수 n) {

    정수 s = i + shift;
    반환 (s >= n) ? (s - n) : s;
}

static void kernel_roll_fused_i0_i1(
    queue &q,
    const float *src_d,
    float *dst_d,
    정수 ne0, 정수 ne1, 정수 ne2, 정수 ne3,
    정수 sh0, 정수 sh1, 정수 sh2, 정수 sh3)
{
    만약 (ne0 == 0 || ne1 == 0 || ne2 == 0 || ne3 == 0) 반환;


    const 정수 stride1 = ne0;
    const 정수 stride2 = ne0 * ne1;
    const 정수 stride3 = ne0 * ne1 * ne2;


    const 정수 shNe0 = (ne0 - sh0) % ne0;
    const 정수 shNe1 = (ne1 - sh1) % ne1;
    const 정수 shNe2 = (ne2 - sh2) % ne2;
    const 정수 shNe3 = (ne3 - sh3) % ne3;


    const size_t g0 = (size_t) ne3;
    const size_t g1 = (size_t) ne2;
    const size_t g2 = (size_t) (ne1 * ne0);

    const range<3> global{ g0, g1, g2 };

    q.submit([&](handler &h) {
        h.parallel_for(global, [=](id<3> idx) {
            const 정수 i3 = (정수) idx[0];
            const 정수 i2 = (정수) idx[1];

            const 정수 fused = (정수) idx[2];
            const 정수 i1 = fused / ne0;
            const 정수 i0 = fused - i1 * ne0;  // fused % ne0


            const 정수 idx_dst = i0
                              + i1 * stride1
                              + i2 * stride2
                              + i3 * stride3;


            const 정수 s0 = wrap_add(i0, shNe0, ne0);
            const 정수 s1 = wrap_add(i1, shNe1, ne1);
            const 정수 s2 = wrap_add(i2, shNe2, ne2);
            const 정수 s3 = wrap_add(i3, shNe3, ne3);

            const 정수 idx_src = s0
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
    const 정수 ne0 = (정수) dst->ne[0];
    const 정수 ne1 = (정수) dst->ne[1];
    const 정수 ne2 = (정수) dst->ne[2];
    const 정수 ne3 = (정수) dst->ne[3];

    const int32_t *params = (const int32_t *) dst->op_params;
    정수 shift0 = params[0];
    정수 shift1 = params[1];
    정수 shift2 = params[2];
    정수 shift3 = params[3];


    만약 ((shift0 | shift1 | shift2 | shift3) == 0) {
        const size_t nb = ggml_nbytes(src);
        queue *q = ctx.stream();
        SYCL_CHECK(CHECK_TRY_ERROR(q->memcpy(dst->data, src->data, nb)));
        반환;
    }

    auto norm = [](정수 sh, 정수 n) -> 정수 {
        만약 (n <= 0) 반환 0;
        sh %= n;
        만약 (sh < 0) sh += n;
        반환 sh;
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
