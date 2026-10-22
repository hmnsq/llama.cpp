#ಅನ್ತರ್ಭಾವ "col2im-1d.hpp"

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ ಶೂನ್ಯ col2im_1d_sycl(
        ಸ್ಥಿರಾಂಕ T * ಲಂಬಸಾಲು,
        T * dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ T_ಒಳ,
        ಸ್ಥಿರಾಂಕ sycl::uint3 T_ಹೊರಹೋಗು_fd,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ K,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ K_OC,
        ಸ್ಥಿರಾಂಕ int32_t s0,
        ಸ್ಥಿರಾಂಕ int32_t p0,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಒಟ್ಟು,
        dpct::ಸರತಿ_ptr stream) {

    ಸ್ಥಿರಾಂಕ uint32_t ಖಂಡ_ಗಾತ್ರ_3 = SYCL_COL2_IM_1_D_ಖಂಡ_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_ಖಂಡಗಳು = (uint32_t) ((ಒಟ್ಟು + ಖಂಡ_ಗಾತ್ರ_3 - 1) / ಖಂಡ_ಗಾತ್ರ_3);

    stream->parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, ಅಂಕೆ_ಖಂಡಗಳು * ಖಂಡ_ಗಾತ್ರ_3),
            sycl::range<3>(1, 1, ಖಂಡ_ಗಾತ್ರ_3)),
        [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ idx = (ಪೂರ್ಣಾಂಕ) ಅಂಶ_ct1.get_global_id(2);
            ಆದರೆ (idx >= ಒಟ್ಟು) {
                ಹಿಂತಿರುಗಿ;
            }

            ಸ್ಥಿರಾಂಕ sycl::uint2 qr = fast_div_modulo((uint32_t) idx, T_ಹೊರಹೋಗು_fd);
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ oc    = (ಪೂರ್ಣಾಂಕ) qr.x();
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ t_ಹೊರಹೋಗು = (ಪೂರ್ಣಾಂಕ) qr.y();
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ t_abs = t_ಹೊರಹೋಗು + p0;

            ಪೂರ್ಣಾಂಕ t_ಒಳ_min = (t_abs - K + s0) / s0;
            ಆದರೆ (t_ಒಳ_min < 0) {
                t_ಒಳ_min = 0;
            }
            ಪೂರ್ಣಾಂಕ t_ಒಳ_max = t_abs / s0;
            ಆದರೆ (t_ಒಳ_max >= T_ಒಳ) {
                t_ಒಳ_max = T_ಒಳ - 1;
            }

            ದಶಮಾಂಶ ಮೊತ್ತ = 0.0f;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ t_ಒಳ = t_ಒಳ_min; t_ಒಳ <= t_ಒಳ_max; ++t_ಒಳ) {
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ k = t_abs - t_ಒಳ * s0;
                ಮೊತ್ತ += static_cast<ದಶಮಾಂಶ>(ಲಂಬಸಾಲು[(oc * K + k) + t_ಒಳ * K_OC]);
            }

            dst[idx] = static_cast<T>(ಮೊತ್ತ);
        });
}

ಶೂನ್ಯ ggml_sycl_op_col2im_1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0 != ಶೂನ್ಯಸೂಚಕ);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(src0->type == dst->type);

    ಸ್ಥಿರಾಂಕ int32_t s0 = ((ಸ್ಥಿರಾಂಕ int32_t *) dst->op_params)[0];
    ಸ್ಥಿರಾಂಕ int32_t OC = ((ಸ್ಥಿರಾಂಕ int32_t *) dst->op_params)[1];
    ಸ್ಥಿರಾಂಕ int32_t p0 = ((ಸ್ಥಿರಾಂಕ int32_t *) dst->op_params)[2];

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ K_OC  = (ಪೂರ್ಣಾಂಕ) src0->ne[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ T_ಒಳ  = (ಪೂರ್ಣಾಂಕ) src0->ne[1];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ K     = K_OC / OC;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ T_ಹೊರಹೋಗು = (ಪೂರ್ಣಾಂಕ) dst->ne[0];

    GGML_ASSERT(OC > 0);
    GGML_ASSERT(K_OC % OC == 0);

    ಸ್ಥಿರಾಂಕ sycl::uint3 T_ಹೊರಹೋಗು_fd = init_fastdiv_ಮೌಲ್ಯಗಳು((uint32_t) T_ಹೊರಹೋಗು);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಒಟ್ಟು = T_ಹೊರಹೋಗು * OC;

    dpct::ಸರತಿ_ptr stream = ctx.stream();

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            col2im_1d_sycl<ದಶಮಾಂಶ>(
                (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src0->ದತ್ತಾಂಶ,
                (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ,
                T_ಒಳ, T_ಹೊರಹೋಗು_fd, K, K_OC, s0, p0, ಒಟ್ಟು, stream);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F16:
            col2im_1d_sycl<sycl::half>(
                (ಸ್ಥಿರಾಂಕ sycl::half *) src0->ದತ್ತಾಂಶ,
                (sycl::half *) dst->ದತ್ತಾಂಶ,
                T_ಒಳ, T_ಹೊರಹೋಗು_fd, K, K_OC, s0, p0, ಒಟ್ಟು, stream);
            ನಿಲ್ಲಿಸು;
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_HAS_BF16
        ಸ್ಥಿತಿ GGML_TYPE_BF16:
            col2im_1d_sycl<sycl::ext::oneapi::bfloat16>(
                (ಸ್ಥಿರಾಂಕ sycl::ext::oneapi::bfloat16 *) src0->ದತ್ತಾಂಶ,
                (sycl::ext::oneapi::bfloat16 *) dst->ದತ್ತಾಂಶ,
                T_ಒಳ, T_ಹೊರಹೋಗು_fd, K, K_OC, s0, p0, ಒಟ್ಟು, stream);
            ನಿಲ್ಲಿಸು;
#ಆದರೆ_ಅಂತ್ಯ
        ಪೂರ್ವನಿಯೋಜಿತ:
            GGML_ABORT("col2im_1d: unsupported type %d", src0->type);
    }
}
