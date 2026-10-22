#include "roll.hpp"
#include "common.hpp"

using namespace sycl;

static anta_hstha ຈຳນວນເຕັມ wrap_add(ຈຳນວນເຕັມ i, ຈຳນວນເຕັມ shift, ຈຳນວນເຕັມ n) {

    ຈຳນວນເຕັມ s = i + shift;
    ສົ່ງຄືນ (s >= n) ? (s - n) : s;
}

static void kernel_roll_fused_i0_i1(
    queue &q,
    const float *src_d,
    float *dst_d,
    ຈຳນວນເຕັມ ne0, ຈຳນວນເຕັມ ne1, ຈຳນວນເຕັມ ne2, ຈຳນວນເຕັມ ne3,
    ຈຳນວນເຕັມ sh0, ຈຳນວນເຕັມ sh1, ຈຳນວນເຕັມ sh2, ຈຳນວນເຕັມ sh3)
{
    ຖ້າ (ne0 == 0 || ne1 == 0 || ne2 == 0 || ne3 == 0) ສົ່ງຄືນ;


    const ຈຳນວນເຕັມ stride1 = ne0;
    const ຈຳນວນເຕັມ stride2 = ne0 * ne1;
    const ຈຳນວນເຕັມ stride3 = ne0 * ne1 * ne2;


    const ຈຳນວນເຕັມ shNe0 = (ne0 - sh0) % ne0;
    const ຈຳນວນເຕັມ shNe1 = (ne1 - sh1) % ne1;
    const ຈຳນວນເຕັມ shNe2 = (ne2 - sh2) % ne2;
    const ຈຳນວນເຕັມ shNe3 = (ne3 - sh3) % ne3;


    const size_t g0 = (size_t) ne3;
    const size_t g1 = (size_t) ne2;
    const size_t g2 = (size_t) (ne1 * ne0);

    const range<3> global{ g0, g1, g2 };

    q.submit([&](handler &h) {
        h.parallel_for(global, [=](id<3> idx) {
            const ຈຳນວນເຕັມ i3 = (ຈຳນວນເຕັມ) idx[0];
            const ຈຳນວນເຕັມ i2 = (ຈຳນວນເຕັມ) idx[1];

            const ຈຳນວນເຕັມ fused = (ຈຳນວນເຕັມ) idx[2];
            const ຈຳນວນເຕັມ i1 = fused / ne0;
            const ຈຳນວນເຕັມ i0 = fused - i1 * ne0;  // fused % ne0


            const ຈຳນວນເຕັມ idx_dst = i0
                              + i1 * stride1
                              + i2 * stride2
                              + i3 * stride3;


            const ຈຳນວນເຕັມ s0 = wrap_add(i0, shNe0, ne0);
            const ຈຳນວນເຕັມ s1 = wrap_add(i1, shNe1, ne1);
            const ຈຳນວນເຕັມ s2 = wrap_add(i2, shNe2, ne2);
            const ຈຳນວນເຕັມ s3 = wrap_add(i3, shNe3, ne3);

            const ຈຳນວນເຕັມ idx_src = s0
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
    const ຈຳນວນເຕັມ ne0 = (ຈຳນວນເຕັມ) dst->ne[0];
    const ຈຳນວນເຕັມ ne1 = (ຈຳນວນເຕັມ) dst->ne[1];
    const ຈຳນວນເຕັມ ne2 = (ຈຳນວນເຕັມ) dst->ne[2];
    const ຈຳນວນເຕັມ ne3 = (ຈຳນວນເຕັມ) dst->ne[3];

    const int32_t *params = (const int32_t *) dst->op_params;
    ຈຳນວນເຕັມ shift0 = params[0];
    ຈຳນວນເຕັມ shift1 = params[1];
    ຈຳນວນເຕັມ shift2 = params[2];
    ຈຳນວນເຕັມ shift3 = params[3];


    ຖ້າ ((shift0 | shift1 | shift2 | shift3) == 0) {
        const size_t nb = ggml_nbytes(src);
        queue *q = ctx.stream();
        SYCL_CHECK(CHECK_TRY_ERROR(q->memcpy(dst->data, src->data, nb)));
        ສົ່ງຄືນ;
    }

    auto norm = [](ຈຳນວນເຕັມ sh, ຈຳນວນເຕັມ n) -> ຈຳນວນເຕັມ {
        ຖ້າ (n <= 0) ສົ່ງຄືນ 0;
        sh %= n;
        ຖ້າ (sh < 0) sh += n;
        ສົ່ງຄືນ sh;
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
