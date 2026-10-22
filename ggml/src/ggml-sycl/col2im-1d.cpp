#अन्तर्भाव "col2im-1d.hpp"

ढाँचा <typename T>
स्थिर शून्य col2im_1d_sycl(
        अचर T * col,
        T * dst,
        अचर पूर्णाङ्क T_in,
        अचर sycl::uint3 T_out_fd,
        अचर पूर्णाङ्क K,
        अचर पूर्णाङ्क K_OC,
        अचर int32_t s0,
        अचर int32_t p0,
        अचर पूर्णाङ्क total,
        dpct::queue_ptr stream) {

    अचर uint32_t block_size = SYCL_COL2IM_1D_BLOCK_SIZE;
    अचर uint32_t अङ्क_blocks = (uint32_t) ((total + block_size - 1) / block_size);

    stream->parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, अङ्क_blocks * block_size),
            sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) {
            अचर पूर्णाङ्क idx = (पूर्णाङ्क) item_ct1.get_global_id(2);
            यदि (idx >= total) {
                फर्काऊ;
            }

            अचर sycl::uint2 qr = fast_div_modulo((uint32_t) idx, T_out_fd);
            अचर पूर्णाङ्क oc    = (पूर्णाङ्क) qr.x();
            अचर पूर्णाङ्क t_out = (पूर्णाङ्क) qr.y();
            अचर पूर्णाङ्क t_abs = t_out + p0;

            पूर्णाङ्क t_in_min = (t_abs - K + s0) / s0;
            यदि (t_in_min < 0) {
                t_in_min = 0;
            }
            पूर्णाङ्क t_in_max = t_abs / s0;
            यदि (t_in_max >= T_in) {
                t_in_max = T_in - 1;
            }

            भिन्न sum = 0.0f;
            लागि (पूर्णाङ्क t_in = t_in_min; t_in <= t_in_max; ++t_in) {
                अचर पूर्णाङ्क k = t_abs - t_in * s0;
                sum += static_cast<भिन्न>(col[(oc * K + k) + t_in * K_OC]);
            }

            dst[idx] = static_cast<T>(sum);
        });
}

शून्य ggml_sycl_op_col2im_1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    अचर ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0 != nullptr);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(src0->type == dst->type);

    अचर int32_t s0 = ((अचर int32_t *) dst->op_params)[0];
    अचर int32_t OC = ((अचर int32_t *) dst->op_params)[1];
    अचर int32_t p0 = ((अचर int32_t *) dst->op_params)[2];

    अचर पूर्णाङ्क K_OC  = (पूर्णाङ्क) src0->ne[0];
    अचर पूर्णाङ्क T_in  = (पूर्णाङ्क) src0->ne[1];
    अचर पूर्णाङ्क K     = K_OC / OC;
    अचर पूर्णाङ्क T_out = (पूर्णाङ्क) dst->ne[0];

    GGML_ASSERT(OC > 0);
    GGML_ASSERT(K_OC % OC == 0);

    अचर sycl::uint3 T_out_fd = init_fastdiv_मानहरू((uint32_t) T_out);

    अचर पूर्णाङ्क total = T_out * OC;

    dpct::queue_ptr stream = ctx.stream();

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            col2im_1d_sycl<भिन्न>(
                (अचर भिन्न *) src0->data,
                (भिन्न *) dst->data,
                T_in, T_out_fd, K, K_OC, s0, p0, total, stream);
            रोक;
        पक्ष GGML_TYPE_F16:
            col2im_1d_sycl<sycl::half>(
                (अचर sycl::half *) src0->data,
                (sycl::half *) dst->data,
                T_in, T_out_fd, K, K_OC, s0, p0, total, stream);
            रोक;
#ifdef GGML_SYCL_HAS_BF16
        पक्ष GGML_TYPE_BF16:
            col2im_1d_sycl<sycl::ext::oneapi::bfloat16>(
                (अचर sycl::ext::oneapi::bfloat16 *) src0->data,
                (sycl::ext::oneapi::bfloat16 *) dst->data,
                T_in, T_out_fd, K, K_OC, s0, p0, total, stream);
            रोक;
#endif
        स्वभाव:
            GGML_ABORT("col2im_1d: unsupported type %d", src0->type);
    }
}
