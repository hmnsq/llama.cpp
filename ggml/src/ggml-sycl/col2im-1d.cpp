#ਅਨ੍ਤਰ੍ਭਾਵ "col2im-1d.hpp"

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ ਖਾਲੀ col2im_1d_sycl(
        ਅਚਲ T * ਖਾਨਾ,
        T * dst,
        ਅਚਲ ਪੂਰਨ T_ਵਿੱਚ,
        ਅਚਲ sycl::uint3 T_ਬਾਹਰਲਾ_fd,
        ਅਚਲ ਪੂਰਨ K,
        ਅਚਲ ਪੂਰਨ K_OC,
        ਅਚਲ int32_t s0,
        ਅਚਲ int32_t p0,
        ਅਚਲ ਪੂਰਨ ਕੁੱਲ,
        dpct::ਕਤਾਰ_ptr stream) {

    ਅਚਲ uint32_t ਖੰਡ_ਮਾਪ_3 = SYCL_COL2_IM_1_D_ਖੰਡ_ਮਾਪ;
    ਅਚਲ uint32_t ਅੰਕ_ਖੰਡਾਂ = (uint32_t) ((ਕੁੱਲ + ਖੰਡ_ਮਾਪ_3 - 1) / ਖੰਡ_ਮਾਪ_3);

    stream->parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, ਅੰਕ_ਖੰਡਾਂ * ਖੰਡ_ਮਾਪ_3),
            sycl::range<3>(1, 1, ਖੰਡ_ਮਾਪ_3)),
        [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
            ਅਚਲ ਪੂਰਨ idx = (ਪੂਰਨ) ਵਸਤੂ_ct1.get_global_id(2);
            ਜੇ (idx >= ਕੁੱਲ) {
                ਵਾਪਸ;
            }

            ਅਚਲ sycl::uint2 qr = fast_div_modulo((uint32_t) idx, T_ਬਾਹਰਲਾ_fd);
            ਅਚਲ ਪੂਰਨ oc    = (ਪੂਰਨ) qr.x();
            ਅਚਲ ਪੂਰਨ t_ਬਾਹਰਲਾ = (ਪੂਰਨ) qr.y();
            ਅਚਲ ਪੂਰਨ t_abs = t_ਬਾਹਰਲਾ + p0;

            ਪੂਰਨ t_ਵਿੱਚ_min = (t_abs - K + s0) / s0;
            ਜੇ (t_ਵਿੱਚ_min < 0) {
                t_ਵਿੱਚ_min = 0;
            }
            ਪੂਰਨ t_ਵਿੱਚ_max = t_abs / s0;
            ਜੇ (t_ਵਿੱਚ_max >= T_ਵਿੱਚ) {
                t_ਵਿੱਚ_max = T_ਵਿੱਚ - 1;
            }

            ਦਸ਼ਮਲਵ ਜੋੜ = 0.0f;
            ਲਈ (ਪੂਰਨ t_ਵਿੱਚ = t_ਵਿੱਚ_min; t_ਵਿੱਚ <= t_ਵਿੱਚ_max; ++t_ਵਿੱਚ) {
                ਅਚਲ ਪੂਰਨ k = t_abs - t_ਵਿੱਚ * s0;
                ਜੋੜ += static_cast<ਦਸ਼ਮਲਵ>(ਖਾਨਾ[(oc * K + k) + t_ਵਿੱਚ * K_OC]);
            }

            dst[idx] = static_cast<T>(ਜੋੜ);
        });
}

ਖਾਲੀ ggml_sycl_op_col2im_1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0 != ਖਾਲੀ_ਸੂਚਕ);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(src0->type == dst->type);

    ਅਚਲ int32_t s0 = ((ਅਚਲ int32_t *) dst->op_params)[0];
    ਅਚਲ int32_t OC = ((ਅਚਲ int32_t *) dst->op_params)[1];
    ਅਚਲ int32_t p0 = ((ਅਚਲ int32_t *) dst->op_params)[2];

    ਅਚਲ ਪੂਰਨ K_OC  = (ਪੂਰਨ) src0->ne[0];
    ਅਚਲ ਪੂਰਨ T_ਵਿੱਚ  = (ਪੂਰਨ) src0->ne[1];
    ਅਚਲ ਪੂਰਨ K     = K_OC / OC;
    ਅਚਲ ਪੂਰਨ T_ਬਾਹਰਲਾ = (ਪੂਰਨ) dst->ne[0];

    GGML_ASSERT(OC > 0);
    GGML_ASSERT(K_OC % OC == 0);

    ਅਚਲ sycl::uint3 T_ਬਾਹਰਲਾ_fd = init_fastdiv_ਮੁੱਲਾਂ((uint32_t) T_ਬਾਹਰਲਾ);

    ਅਚਲ ਪੂਰਨ ਕੁੱਲ = T_ਬਾਹਰਲਾ * OC;

    dpct::ਕਤਾਰ_ptr stream = ctx.stream();

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            col2im_1d_sycl<ਦਸ਼ਮਲਵ>(
                (ਅਚਲ ਦਸ਼ਮਲਵ *) src0->ਅੰਕੜੇ,
                (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ,
                T_ਵਿੱਚ, T_ਬਾਹਰਲਾ_fd, K, K_OC, s0, p0, ਕੁੱਲ, stream);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F16:
            col2im_1d_sycl<sycl::half>(
                (ਅਚਲ sycl::half *) src0->ਅੰਕੜੇ,
                (sycl::half *) dst->ਅੰਕੜੇ,
                T_ਵਿੱਚ, T_ਬਾਹਰਲਾ_fd, K, K_OC, s0, p0, ਕੁੱਲ, stream);
            ਰੋਕੋ;
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_HAS_BF16
        ਹਾਲਤ GGML_TYPE_BF16:
            col2im_1d_sycl<sycl::ext::oneapi::bfloat16>(
                (ਅਚਲ sycl::ext::oneapi::bfloat16 *) src0->ਅੰਕੜੇ,
                (sycl::ext::oneapi::bfloat16 *) dst->ਅੰਕੜੇ,
                T_ਵਿੱਚ, T_ਬਾਹਰਲਾ_fd, K, K_OC, s0, p0, ਕੁੱਲ, stream);
            ਰੋਕੋ;
#ਜੇ_ਅੰਤ
        ਮੂਲ:
            GGML_ABORT("col2im_1d: unsupported type %d", src0->type);
    }
}
