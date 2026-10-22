#ਅਨ੍ਤਰ੍ਭਾਵ "fill.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "common.hpp"

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SYCL_FILL_ਖੰਡ_ਮਾਪ 256

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ ਖਾਲੀ fill_kernel(T * dst, ਅਚਲ int64_t k, ਅਚਲ T value,
                        ਅਚਲ sycl::nd_ਵਸਤੂ<1> & ਵਸਤੂ) {
    ਅਚਲ int64_t i = (int64_t)ਵਸਤੂ.get_global_id(0);
    ਜੇ (i >= k) {
        ਵਾਪਸ;
    }
    dst[i] = value;
}

ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_sycl_op_fill(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_is_contiguous(dst));

    dpct::ਕਤਾਰ_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_ਸੈੱਟ_device(ctx.device));

    ਦਸ਼ਮਲਵ value;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&value, dst->op_params, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਅਚਲ int64_t k = ggml_nelements(dst);
    ਅਚਲ int64_t ਅੰਕ_ਖੰਡਾਂ = (k + SYCL_FILL_ਖੰਡ_ਮਾਪ - 1) / SYCL_FILL_ਖੰਡ_ਮਾਪ;
    ਖਾਲੀ * dst_d = dst->ਅੰਕੜੇ;

    ਚੋਣ (dst->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            stream->parallel_for(
                sycl::nd_range<1>(ਅੰਕ_ਖੰਡਾਂ * SYCL_FILL_ਖੰਡ_ਮਾਪ, SYCL_FILL_ਖੰਡ_ਮਾਪ),
                [=](sycl::nd_ਵਸਤੂ<1> ਵਸਤੂ) {
                    fill_kernel(static_cast<ਦਸ਼ਮਲਵ *>(dst_d), k, value, ਵਸਤੂ);
                });
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F16:
            {
                sycl::half h_ਮੁੱਲ = sycl::half(value);
                stream->parallel_for(
                    sycl::nd_range<1>(ਅੰਕ_ਖੰਡਾਂ * SYCL_FILL_ਖੰਡ_ਮਾਪ, SYCL_FILL_ਖੰਡ_ਮਾਪ),
                    [=](sycl::nd_ਵਸਤੂ<1> ਵਸਤੂ) {
                        fill_kernel(static_cast<sycl::half *>(dst_d), k, h_ਮੁੱਲ, ਵਸਤੂ);
                    });
            }
            ਰੋਕੋ;
        ਮੂਲ:
            GGML_ABORT("unsupported type");
    }
}

ਖਾਲੀ ggml_sycl_fill(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/0);
    ggml_sycl_op_fill(ctx, dst);
}
