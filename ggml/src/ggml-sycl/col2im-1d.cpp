#ضمن "col2im-1d.hpp"

template <typename T>
ساكن فراغ col2im_1d_sycl(
        ثابت T * col,
        T * dst,
        ثابت صحيح T_in,
        ثابت sycl::uint3 T_out_fd,
        ثابت صحيح K,
        ثابت صحيح K_OC,
        ثابت int32_t s0,
        ثابت int32_t p0,
        ثابت صحيح total,
        dpct::queue_ptr stream) {

    ثابت uint32_t block_size = SYCL_COL2IM_1D_BLOCK_SIZE;
    ثابت uint32_t num_blocks = (uint32_t) ((total + block_size - 1) / block_size);

    stream->parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, num_blocks * block_size),
            sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) {
            ثابت صحيح idx = (صحيح) item_ct1.get_global_id(2);
            ܐܢ (idx >= total) {
                ܦܢܐ;
            }

            ثابت sycl::uint2 qr = fast_div_modulo((uint32_t) idx, T_out_fd);
            ثابت صحيح oc    = (صحيح) qr.x();
            ثابت صحيح t_out = (صحيح) qr.y();
            ثابت صحيح t_abs = t_out + p0;

            صحيح t_in_min = (t_abs - K + s0) / s0;
            ܐܢ (t_in_min < 0) {
                t_in_min = 0;
            }
            صحيح t_in_max = t_abs / s0;
            ܐܢ (t_in_max >= T_in) {
                t_in_max = T_in - 1;
            }

            عائم sum = 0.0f;
            لكل (صحيح t_in = t_in_min; t_in <= t_in_max; ++t_in) {
                ثابت صحيح k = t_abs - t_in * s0;
                sum += static_cast<عائم>(col[(oc * K + k) + t_in * K_OC]);
            }

            dst[idx] = static_cast<T>(sum);
        });
}

فراغ ggml_sycl_op_col2im_1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0 != nullptr);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(src0->type == dst->type);

    ثابت int32_t s0 = ((ثابت int32_t *) dst->op_params)[0];
    ثابت int32_t OC = ((ثابت int32_t *) dst->op_params)[1];
    ثابت int32_t p0 = ((ثابت int32_t *) dst->op_params)[2];

    ثابت صحيح K_OC  = (صحيح) src0->ne[0];
    ثابت صحيح T_in  = (صحيح) src0->ne[1];
    ثابت صحيح K     = K_OC / OC;
    ثابت صحيح T_out = (صحيح) dst->ne[0];

    GGML_ASSERT(OC > 0);
    GGML_ASSERT(K_OC % OC == 0);

    ثابت sycl::uint3 T_out_fd = init_fastdiv_values((uint32_t) T_out);

    ثابت صحيح total = T_out * OC;

    dpct::queue_ptr stream = ctx.stream();

    بدل (src0->type) {
        حالة GGML_TYPE_F32:
            col2im_1d_sycl<عائم>(
                (ثابت عائم *) src0->data,
                (عائم *) dst->data,
                T_in, T_out_fd, K, K_OC, s0, p0, total, stream);
            اكسر;
        حالة GGML_TYPE_F16:
            col2im_1d_sycl<sycl::half>(
                (ثابت sycl::half *) src0->data,
                (sycl::half *) dst->data,
                T_in, T_out_fd, K, K_OC, s0, p0, total, stream);
            اكسر;
#ifdef GGML_SYCL_HAS_BF16
        حالة GGML_TYPE_BF16:
            col2im_1d_sycl<sycl::ext::oneapi::bfloat16>(
                (ثابت sycl::ext::oneapi::bfloat16 *) src0->data,
                (sycl::ext::oneapi::bfloat16 *) dst->data,
                T_in, T_out_fd, K, K_OC, s0, p0, total, stream);
            اكسر;
#endif
        افتراضي:
            GGML_ABORT("col2im_1d: unsupported type %d", src0->type);
    }
}
