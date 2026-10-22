#ಅನ್ತರ್ಭಾವ "fill.hpp"
#ಅನ್ತರ್ಭಾವ "common.hpp"

#ವ್ಯಾಖ್ಯಾನಿಸು SYCL_FILL_ಖಂಡ_ಗಾತ್ರ 256

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ ಶೂನ್ಯ fill_kernel(T * dst, ಸ್ಥಿರಾಂಕ int64_t k, ಸ್ಥಿರಾಂಕ T value,
                        ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<1> & ಅಂಶ) {
    ಸ್ಥಿರಾಂಕ int64_t i = (int64_t)ಅಂಶ.get_global_id(0);
    ಆದರೆ (i >= k) {
        ಹಿಂತಿರುಗಿ;
    }
    dst[i] = value;
}

ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_sycl_op_fill(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_is_contiguous(dst));

    dpct::ಸರತಿ_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_ಗಣ_device(ctx.device));

    ದಶಮಾಂಶ value;
    ಸ್ಮರಣೆ_ನಕಲು(&value, dst->op_params, ಗಾತ್ರ(ದಶಮಾಂಶ));

    ಸ್ಥಿರಾಂಕ int64_t k = ggml_nelements(dst);
    ಸ್ಥಿರಾಂಕ int64_t ಅಂಕೆ_ಖಂಡಗಳು = (k + SYCL_FILL_ಖಂಡ_ಗಾತ್ರ - 1) / SYCL_FILL_ಖಂಡ_ಗಾತ್ರ;
    ಶೂನ್ಯ * dst_d = dst->ದತ್ತಾಂಶ;

    ಆಯ್ಕೆ (dst->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            stream->parallel_for(
                sycl::nd_range<1>(ಅಂಕೆ_ಖಂಡಗಳು * SYCL_FILL_ಖಂಡ_ಗಾತ್ರ, SYCL_FILL_ಖಂಡ_ಗಾತ್ರ),
                [=](sycl::nd_ಅಂಶ<1> ಅಂಶ) {
                    fill_kernel(static_cast<ದಶಮಾಂಶ *>(dst_d), k, value, ಅಂಶ);
                });
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F16:
            {
                sycl::half h_ಮೌಲ್ಯ = sycl::half(value);
                stream->parallel_for(
                    sycl::nd_range<1>(ಅಂಕೆ_ಖಂಡಗಳು * SYCL_FILL_ಖಂಡ_ಗಾತ್ರ, SYCL_FILL_ಖಂಡ_ಗಾತ್ರ),
                    [=](sycl::nd_ಅಂಶ<1> ಅಂಶ) {
                        fill_kernel(static_cast<sycl::half *>(dst_d), k, h_ಮೌಲ್ಯ, ಅಂಶ);
                    });
            }
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            GGML_ABORT("unsupported type");
    }
}

ಶೂನ್ಯ ggml_sycl_fill(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/0);
    ggml_sycl_op_fill(ctx, dst);
}
