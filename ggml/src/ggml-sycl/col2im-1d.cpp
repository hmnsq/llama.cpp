#ಅನ್ತರ್ಭಾವ "col2im-1d.hpp"

template <typename T>
ಸ್ಥಿರ ಶೂನ್ಯ col2im_1d_sycl(
        ಧ್ರುವ T * col,
        T * dst,
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ T_in,
        ಧ್ರುವ sycl::uint3 T_out_fd,
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ K,
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ K_OC,
        ಧ್ರುವ int32_t s0,
        ಧ್ರುವ int32_t p0,
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ total,
        dpct::queue_ptr stream) {

    ಧ್ರುವ uint32_t block_size = SYCL_COL2IM_1D_BLOCK_SIZE;
    ಧ್ರುವ uint32_t ಅಂಕೆ_blocks = (uint32_t) ((total + block_size - 1) / block_size);

    stream->parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, ಅಂಕೆ_blocks * block_size),
            sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) {
            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ idx = (ಪೂರ್ಣಾಙ್ಕ) item_ct1.get_global_id(2);
            ಯದಿ (idx >= total) {
                ಪ್ರತ್ಯಾಗಮ;
            }

            ಧ್ರುವ sycl::uint2 qr = fast_div_modulo((uint32_t) idx, T_out_fd);
            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ oc    = (ಪೂರ್ಣಾಙ್ಕ) qr.x();
            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ t_out = (ಪೂರ್ಣಾಙ್ಕ) qr.y();
            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ t_abs = t_out + p0;

            ಪೂರ್ಣಾಙ್ಕ t_in_min = (t_abs - K + s0) / s0;
            ಯದಿ (t_in_min < 0) {
                t_in_min = 0;
            }
            ಪೂರ್ಣಾಙ್ಕ t_in_max = t_abs / s0;
            ಯದಿ (t_in_max >= T_in) {
                t_in_max = T_in - 1;
            }

            ಭಿನ್ನ sum = 0.0f;
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ t_in = t_in_min; t_in <= t_in_max; ++t_in) {
                ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ k = t_abs - t_in * s0;
                sum += static_cast<ಭಿನ್ನ>(col[(oc * K + k) + t_in * K_OC]);
            }

            dst[idx] = static_cast<T>(sum);
        });
}

ಶೂನ್ಯ ggml_sycl_op_col2im_1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ಧ್ರುವ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0 != nullptr);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(src0->type == dst->type);

    ಧ್ರುವ int32_t s0 = ((ಧ್ರುವ int32_t *) dst->op_params)[0];
    ಧ್ರುವ int32_t OC = ((ಧ್ರುವ int32_t *) dst->op_params)[1];
    ಧ್ರುವ int32_t p0 = ((ಧ್ರುವ int32_t *) dst->op_params)[2];

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ K_OC  = (ಪೂರ್ಣಾಙ್ಕ) src0->ne[0];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ T_in  = (ಪೂರ್ಣಾಙ್ಕ) src0->ne[1];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ K     = K_OC / OC;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ T_out = (ಪೂರ್ಣಾಙ್ಕ) dst->ne[0];

    GGML_ASSERT(OC > 0);
    GGML_ASSERT(K_OC % OC == 0);

    ಧ್ರುವ sycl::uint3 T_out_fd = init_fastdiv_ಮೌಲ್ಯಗಳು((uint32_t) T_out);

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ total = T_out * OC;

    dpct::queue_ptr stream = ctx.stream();

    ವಿಕಲ್ಪ (src0->type) {
        ಪಕ್ಷ GGML_TYPE_F32:
            col2im_1d_sycl<ಭಿನ್ನ>(
                (ಧ್ರುವ ಭಿನ್ನ *) src0->data,
                (ಭಿನ್ನ *) dst->data,
                T_in, T_out_fd, K, K_OC, s0, p0, total, stream);
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_F16:
            col2im_1d_sycl<sycl::half>(
                (ಧ್ರುವ sycl::half *) src0->data,
                (sycl::half *) dst->data,
                T_in, T_out_fd, K, K_OC, s0, p0, total, stream);
            ಭಙ್ಗ;
#ifdef GGML_SYCL_HAS_BF16
        ಪಕ್ಷ GGML_TYPE_BF16:
            col2im_1d_sycl<sycl::ext::oneapi::bfloat16>(
                (ಧ್ರುವ sycl::ext::oneapi::bfloat16 *) src0->data,
                (sycl::ext::oneapi::bfloat16 *) dst->data,
                T_in, T_out_fd, K, K_OC, s0, p0, total, stream);
            ಭಙ್ಗ;
#endif
        ಸ್ವಭಾವ:
            GGML_ABORT("col2im_1d: unsupported type %d", src0->type);
    }
}
