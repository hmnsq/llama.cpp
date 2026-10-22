#include "col2im-1d.hpp"

template <typename T>
static void col2im_1d_sycl(
        const T * col,
        T * dst,
        const số_nguyên T_in,
        const sycl::uint3 T_out_fd,
        const số_nguyên K,
        const số_nguyên K_OC,
        const int32_t s0,
        const int32_t p0,
        const số_nguyên total,
        dpct::queue_ptr stream) {

    const uint32_t block_size = SYCL_COL2IM_1D_BLOCK_SIZE;
    const uint32_t num_blocks = (uint32_t) ((total + block_size - 1) / block_size);

    stream->parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, num_blocks * block_size),
            sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) {
            const số_nguyên idx = (số_nguyên) item_ct1.get_global_id(2);
            nếu (idx >= total) {
                trả_về;
            }

            const sycl::uint2 qr = fast_div_modulo((uint32_t) idx, T_out_fd);
            const số_nguyên oc    = (số_nguyên) qr.x();
            const số_nguyên t_out = (số_nguyên) qr.y();
            const số_nguyên t_abs = t_out + p0;

            số_nguyên t_in_min = (t_abs - K + s0) / s0;
            nếu (t_in_min < 0) {
                t_in_min = 0;
            }
            số_nguyên t_in_max = t_abs / s0;
            nếu (t_in_max >= T_in) {
                t_in_max = T_in - 1;
            }

            float sum = 0.0f;
            cho (số_nguyên t_in = t_in_min; t_in <= t_in_max; ++t_in) {
                const số_nguyên k = t_abs - t_in * s0;
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

    const số_nguyên K_OC  = (số_nguyên) src0->ne[0];
    const số_nguyên T_in  = (số_nguyên) src0->ne[1];
    const số_nguyên K     = K_OC / OC;
    const số_nguyên T_out = (số_nguyên) dst->ne[0];

    GGML_ASSERT(OC > 0);
    GGML_ASSERT(K_OC % OC == 0);

    const sycl::uint3 T_out_fd = init_fastdiv_values((uint32_t) T_out);

    const số_nguyên total = T_out * OC;

    dpct::queue_ptr stream = ctx.stream();

    switch (src0->type) {
        case GGML_TYPE_F32:
            col2im_1d_sycl<float>(
                (const float *) src0->data,
                (float *) dst->data,
                T_in, T_out_fd, K, K_OC, s0, p0, total, stream);
            break;
        case GGML_TYPE_F16:
            col2im_1d_sycl<sycl::half>(
                (const sycl::half *) src0->data,
                (sycl::half *) dst->data,
                T_in, T_out_fd, K, K_OC, s0, p0, total, stream);
            break;
#ifdef GGML_SYCL_HAS_BF16
        case GGML_TYPE_BF16:
            col2im_1d_sycl<sycl::ext::oneapi::bfloat16>(
                (const sycl::ext::oneapi::bfloat16 *) src0->data,
                (sycl::ext::oneapi::bfloat16 *) dst->data,
                T_in, T_out_fd, K, K_OC, s0, p0, total, stream);
            break;
#endif
        default:
            GGML_ABORT("col2im_1d: unsupported type %d", src0->type);
    }
}
