#अन्तर्भाव "col2im-1d.hpp"

template <typename T>
स्थिर शून्य col2im_1d_sycl(
        ध्रुव T * col,
        T * dst,
        ध्रुव पूर्णाङ्क T_in,
        ध्रुव sycl::uint3 T_out_fd,
        ध्रुव पूर्णाङ्क K,
        ध्रुव पूर्णाङ्क K_OC,
        ध्रुव int32_t s0,
        ध्रुव int32_t p0,
        ध्रुव पूर्णाङ्क total,
        dpct::queue_ptr stream) {

    ध्रुव uint32_t block_size = SYCL_COL2IM_1D_BLOCK_SIZE;
    ध्रुव uint32_t अंक_blocks = (uint32_t) ((total + block_size - 1) / block_size);

    stream->parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, अंक_blocks * block_size),
            sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) {
            ध्रुव पूर्णाङ्क idx = (पूर्णाङ्क) item_ct1.get_global_id(2);
            यदि (idx >= total) {
                प्रत्यागम;
            }

            ध्रुव sycl::uint2 qr = fast_div_modulo((uint32_t) idx, T_out_fd);
            ध्रुव पूर्णाङ्क oc    = (पूर्णाङ्क) qr.x();
            ध्रुव पूर्णाङ्क t_out = (पूर्णाङ्क) qr.y();
            ध्रुव पूर्णाङ्क t_abs = t_out + p0;

            पूर्णाङ्क t_in_min = (t_abs - K + s0) / s0;
            यदि (t_in_min < 0) {
                t_in_min = 0;
            }
            पूर्णाङ्क t_in_max = t_abs / s0;
            यदि (t_in_max >= T_in) {
                t_in_max = T_in - 1;
            }

            भिन्न sum = 0.0f;
            प्रति (पूर्णाङ्क t_in = t_in_min; t_in <= t_in_max; ++t_in) {
                ध्रुव पूर्णाङ्क k = t_abs - t_in * s0;
                sum += static_cast<भिन्न>(col[(oc * K + k) + t_in * K_OC]);
            }

            dst[idx] = static_cast<T>(sum);
        });
}

शून्य ggml_sycl_op_col2im_1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ध्रुव ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0 != nullptr);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(src0->type == dst->type);

    ध्रुव int32_t s0 = ((ध्रुव int32_t *) dst->op_params)[0];
    ध्रुव int32_t OC = ((ध्रुव int32_t *) dst->op_params)[1];
    ध्रुव int32_t p0 = ((ध्रुव int32_t *) dst->op_params)[2];

    ध्रुव पूर्णाङ्क K_OC  = (पूर्णाङ्क) src0->ne[0];
    ध्रुव पूर्णाङ्क T_in  = (पूर्णाङ्क) src0->ne[1];
    ध्रुव पूर्णाङ्क K     = K_OC / OC;
    ध्रुव पूर्णाङ्क T_out = (पूर्णाङ्क) dst->ne[0];

    GGML_ASSERT(OC > 0);
    GGML_ASSERT(K_OC % OC == 0);

    ध्रुव sycl::uint3 T_out_fd = init_fastdiv_मूल्ये((uint32_t) T_out);

    ध्रुव पूर्णाङ्क total = T_out * OC;

    dpct::queue_ptr stream = ctx.stream();

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            col2im_1d_sycl<भिन्न>(
                (ध्रुव भिन्न *) src0->data,
                (भिन्न *) dst->data,
                T_in, T_out_fd, K, K_OC, s0, p0, total, stream);
            भङ्ग;
        पक्ष GGML_TYPE_F16:
            col2im_1d_sycl<sycl::half>(
                (ध्रुव sycl::half *) src0->data,
                (sycl::half *) dst->data,
                T_in, T_out_fd, K, K_OC, s0, p0, total, stream);
            भङ्ग;
#ifdef GGML_SYCL_HAS_BF16
        पक्ष GGML_TYPE_BF16:
            col2im_1d_sycl<sycl::ext::oneapi::bfloat16>(
                (ध्रुव sycl::ext::oneapi::bfloat16 *) src0->data,
                (sycl::ext::oneapi::bfloat16 *) dst->data,
                T_in, T_out_fd, K, K_OC, s0, p0, total, stream);
            भङ्ग;
#endif
        स्वभाव:
            GGML_ABORT("col2im_1d: unsupported type %d", src0->type);
    }
}
