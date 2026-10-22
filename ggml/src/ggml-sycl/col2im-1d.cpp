#अन्तर्भाव "col2im-1d.hpp"

प्रतिरूपम् <प्रकारनाम T>
स्थिर शून्य col2im_1d_sycl(
        ध्रुव T * col,
        T * dst,
        ध्रुव पूर्णाङ्क T_अन्तर्,
        ध्रुव sycl::uint3 T_बहिर्गमनम्_fd,
        ध्रुव पूर्णाङ्क K,
        ध्रुव पूर्णाङ्क K_OC,
        ध्रुव int32_t s0,
        ध्रुव int32_t p0,
        ध्रुव पूर्णाङ्क योग,
        dpct::queue_ptr stream) {

    ध्रुव uint32_t खण्डम्_आकार_3 = SYCL_COL2_IM_1_D_खण्डम्_आकार;
    ध्रुव uint32_t अङ्कः_खण्डानि = (uint32_t) ((योग + खण्डम्_आकार_3 - 1) / खण्डम्_आकार_3);

    stream->parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, अङ्कः_खण्डानि * खण्डम्_आकार_3),
            sycl::range<3>(1, 1, खण्डम्_आकार_3)),
        [=](sycl::nd_item<3> item_ct1) {
            ध्रुव पूर्णाङ्क idx = (पूर्णाङ्क) item_ct1.get_global_id(2);
            यदि (idx >= योग) {
                प्रत्यागम;
            }

            ध्रुव sycl::uint2 qr = fast_div_modulo((uint32_t) idx, T_बहिर्गमनम्_fd);
            ध्रुव पूर्णाङ्क oc    = (पूर्णाङ्क) qr.x();
            ध्रुव पूर्णाङ्क t_बहिर्गमनम् = (पूर्णाङ्क) qr.y();
            ध्रुव पूर्णाङ्क t_abs = t_बहिर्गमनम् + p0;

            पूर्णाङ्क t_अन्तर्_min = (t_abs - K + s0) / s0;
            यदि (t_अन्तर्_min < 0) {
                t_अन्तर्_min = 0;
            }
            पूर्णाङ्क t_अन्तर्_max = t_abs / s0;
            यदि (t_अन्तर्_max >= T_अन्तर्) {
                t_अन्तर्_max = T_अन्तर् - 1;
            }

            भिन्न योग_2 = 0.0f;
            प्रति (पूर्णाङ्क t_अन्तर् = t_अन्तर्_min; t_अन्तर् <= t_अन्तर्_max; ++t_अन्तर्) {
                ध्रुव पूर्णाङ्क k = t_abs - t_अन्तर् * s0;
                योग_2 += static_cast<भिन्न>(col[(oc * K + k) + t_अन्तर् * K_OC]);
            }

            dst[idx] = static_cast<T>(योग_2);
        });
}

शून्य ggml_sycl_op_col2im_1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ध्रुव ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0 != शून्यसूचकः);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(src0->type == dst->type);

    ध्रुव int32_t s0 = ((ध्रुव int32_t *) dst->op_params)[0];
    ध्रुव int32_t OC = ((ध्रुव int32_t *) dst->op_params)[1];
    ध्रुव int32_t p0 = ((ध्रुव int32_t *) dst->op_params)[2];

    ध्रुव पूर्णाङ्क K_OC  = (पूर्णाङ्क) src0->ne[0];
    ध्रुव पूर्णाङ्क T_अन्तर्  = (पूर्णाङ्क) src0->ne[1];
    ध्रुव पूर्णाङ्क K     = K_OC / OC;
    ध्रुव पूर्णाङ्क T_बहिर्गमनम् = (पूर्णाङ्क) dst->ne[0];

    GGML_ASSERT(OC > 0);
    GGML_ASSERT(K_OC % OC == 0);

    ध्रुव sycl::uint3 T_बहिर्गमनम्_fd = init_fastdiv_मानानि((uint32_t) T_बहिर्गमनम्);

    ध्रुव पूर्णाङ्क योग = T_बहिर्गमनम् * OC;

    dpct::queue_ptr stream = ctx.stream();

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            col2im_1d_sycl<भिन्न>(
                (ध्रुव भिन्न *) src0->दत्त,
                (भिन्न *) dst->दत्त,
                T_अन्तर्, T_बहिर्गमनम्_fd, K, K_OC, s0, p0, योग, stream);
            भङ्ग;
        पक्ष GGML_TYPE_F16:
            col2im_1d_sycl<sycl::half>(
                (ध्रुव sycl::half *) src0->दत्त,
                (sycl::half *) dst->दत्त,
                T_अन्तर्, T_बहिर्गमनम्_fd, K, K_OC, s0, p0, योग, stream);
            भङ्ग;
#यदि_परिभाषितम् GGML_SYCL_HAS_BF16
        पक्ष GGML_TYPE_BF16:
            col2im_1d_sycl<sycl::ext::oneapi::bfloat16>(
                (ध्रुव sycl::ext::oneapi::bfloat16 *) src0->दत्त,
                (sycl::ext::oneapi::bfloat16 *) dst->दत्त,
                T_अन्तर्, T_बहिर्गमनम्_fd, K, K_OC, s0, p0, योग, stream);
            भङ्ग;
#यदि_अन्तः
        स्वभाव:
            GGML_ABORT("col2im_1d: unsupported type %d", src0->type);
    }
}
