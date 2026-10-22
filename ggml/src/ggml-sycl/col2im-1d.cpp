#include "col2im-1d.hpp"

template <typename T>
static void col2im_1d_sycl(
        const T * col,
        T * dst,
        const ကိန်းပြည့် T_in,
        const sycl::uint3 T_out_fd,
        const ကိန်းပြည့် K,
        const ကိန်းပြည့် K_OC,
        const int32_t s0,
        const int32_t p0,
        const ကိန်းပြည့် total,
        dpct::queue_ptr stream) {

    const uint32_t block_size = SYCL_COL2IM_1D_BLOCK_SIZE;
    const uint32_t num_blocks = (uint32_t) ((total + block_size - 1) / block_size);

    stream->parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, num_blocks * block_size),
            sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) {
            const ကိန်းပြည့် idx = (ကိန်းပြည့်) item_ct1.get_global_id(2);
            အကယ်၍ (idx >= total) {
                ပြန်ပေး;
            }

            const sycl::uint2 qr = fast_div_modulo((uint32_t) idx, T_out_fd);
            const ကိန်းပြည့် oc    = (ကိန်းပြည့်) qr.x();
            const ကိန်းပြည့် t_out = (ကိန်းပြည့်) qr.y();
            const ကိန်းပြည့် t_abs = t_out + p0;

            ကိန်းပြည့် t_in_min = (t_abs - K + s0) / s0;
            အကယ်၍ (t_in_min < 0) {
                t_in_min = 0;
            }
            ကိန်းပြည့် t_in_max = t_abs / s0;
            အကယ်၍ (t_in_max >= T_in) {
                t_in_max = T_in - 1;
            }

            float sum = 0.0f;
            အတွက် (ကိန်းပြည့် t_in = t_in_min; t_in <= t_in_max; ++t_in) {
                const ကိန်းပြည့် k = t_abs - t_in * s0;
                sum += static_cast<float>(col[(oc * K + k) + t_in * K_OC]);
            }

            dst[idx] = static_cast<T>(sum);
        });
}

void ggml_sycl_op_col2im_1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0 != nullptr);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(src0->type == dst->type);

    const int32_t s0 = ((const int32_t *) dst->op_params)[0];
    const int32_t OC = ((const int32_t *) dst->op_params)[1];
    const int32_t p0 = ((const int32_t *) dst->op_params)[2];

    const ကိန်းပြည့် K_OC  = (ကိန်းပြည့်) src0->ne[0];
    const ကိန်းပြည့် T_in  = (ကိန်းပြည့်) src0->ne[1];
    const ကိန်းပြည့် K     = K_OC / OC;
    const ကိန်းပြည့် T_out = (ကိန်းပြည့်) dst->ne[0];

    GGML_ASSERT(OC > 0);
    GGML_ASSERT(K_OC % OC == 0);

    const sycl::uint3 T_out_fd = init_fastdiv_values((uint32_t) T_out);

    const ကိန်းပြည့် total = T_out * OC;

    dpct::queue_ptr stream = ctx.stream();

    switch (src0->type) {
        pak_sa GGML_TYPE_F32:
            col2im_1d_sycl<float>(
                (const float *) src0->data,
                (float *) dst->data,
                T_in, T_out_fd, K, K_OC, s0, p0, total, stream);
            bha_nga;
        pak_sa GGML_TYPE_F16:
            col2im_1d_sycl<sycl::half>(
                (const sycl::half *) src0->data,
                (sycl::half *) dst->data,
                T_in, T_out_fd, K, K_OC, s0, p0, total, stream);
            bha_nga;
#ifdef GGML_SYCL_HAS_BF16
        pak_sa GGML_TYPE_BF16:
            col2im_1d_sycl<sycl::ext::oneapi::bfloat16>(
                (const sycl::ext::oneapi::bfloat16 *) src0->data,
                (sycl::ext::oneapi::bfloat16 *) dst->data,
                T_in, T_out_fd, K, K_OC, s0, p0, total, stream);
            bha_nga;
#endif
        default:
            GGML_ABORT("col2im_1d: unsupported type %d", src0->type);
    }
}
