#include "roll.hpp"
#include "common.hpp"

using namespace sycl;

static inline 整数 wrap_add(整数 i, 整数 shift, 整数 n) {

    整数 s = i + shift;
    戻る (s >= n) ? (s - n) : s;
}

static void kernel_roll_fused_i0_i1(
    queue &q,
    const float *src_d,
    float *dst_d,
    整数 ne0, 整数 ne1, 整数 ne2, 整数 ne3,
    整数 sh0, 整数 sh1, 整数 sh2, 整数 sh3)
{
    もし (ne0 == 0 || ne1 == 0 || ne2 == 0 || ne3 == 0) 戻る;


    const 整数 stride1 = ne0;
    const 整数 stride2 = ne0 * ne1;
    const 整数 stride3 = ne0 * ne1 * ne2;


    const 整数 shNe0 = (ne0 - sh0) % ne0;
    const 整数 shNe1 = (ne1 - sh1) % ne1;
    const 整数 shNe2 = (ne2 - sh2) % ne2;
    const 整数 shNe3 = (ne3 - sh3) % ne3;


    const size_t g0 = (size_t) ne3;
    const size_t g1 = (size_t) ne2;
    const size_t g2 = (size_t) (ne1 * ne0);

    const range<3> global{ g0, g1, g2 };

    q.submit([&](handler &h) {
        h.parallel_for(global, [=](id<3> idx) {
            const 整数 i3 = (整数) idx[0];
            const 整数 i2 = (整数) idx[1];

            const 整数 fused = (整数) idx[2];
            const 整数 i1 = fused / ne0;
            const 整数 i0 = fused - i1 * ne0;  // fused % ne0


            const 整数 idx_dst = i0
                              + i1 * stride1
                              + i2 * stride2
                              + i3 * stride3;


            const 整数 s0 = wrap_add(i0, shNe0, ne0);
            const 整数 s1 = wrap_add(i1, shNe1, ne1);
            const 整数 s2 = wrap_add(i2, shNe2, ne2);
            const 整数 s3 = wrap_add(i3, shNe3, ne3);

            const 整数 idx_src = s0
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
    const 整数 ne0 = (整数) dst->ne[0];
    const 整数 ne1 = (整数) dst->ne[1];
    const 整数 ne2 = (整数) dst->ne[2];
    const 整数 ne3 = (整数) dst->ne[3];

    const int32_t *params = (const int32_t *) dst->op_params;
    整数 shift0 = params[0];
    整数 shift1 = params[1];
    整数 shift2 = params[2];
    整数 shift3 = params[3];


    もし ((shift0 | shift1 | shift2 | shift3) == 0) {
        const size_t nb = ggml_nbytes(src);
        queue *q = ctx.stream();
        SYCL_CHECK(CHECK_TRY_ERROR(q->memcpy(dst->data, src->data, nb)));
        戻る;
    }

    auto norm = [](整数 sh, 整数 n) -> 整数 {
        もし (n <= 0) 戻る 0;
        sh %= n;
        もし (sh < 0) sh += n;
        戻る sh;
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
