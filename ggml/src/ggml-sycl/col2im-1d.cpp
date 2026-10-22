#அந்தர்பாவ "col2im-1d.hpp"

template <typename T>
நிலையான வெறுமை col2im_1d_sycl(
        மாறிலி T * col,
        T * dst,
        மாறிலி முழு T_in,
        மாறிலி sycl::uint3 T_out_fd,
        மாறிலி முழு K,
        மாறிலி முழு K_OC,
        மாறிலி int32_t s0,
        மாறிலி int32_t p0,
        மாறிலி முழு total,
        dpct::queue_ptr stream) {

    மாறிலி uint32_t block_size = SYCL_COL2IM_1D_BLOCK_SIZE;
    மாறிலி uint32_t இலக்கம்_blocks = (uint32_t) ((total + block_size - 1) / block_size);

    stream->parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, இலக்கம்_blocks * block_size),
            sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) {
            மாறிலி முழு idx = (முழு) item_ct1.get_global_id(2);
            எனில் (idx >= total) {
                திரும்பு;
            }

            மாறிலி sycl::uint2 qr = fast_div_modulo((uint32_t) idx, T_out_fd);
            மாறிலி முழு oc    = (முழு) qr.x();
            மாறிலி முழு t_out = (முழு) qr.y();
            மாறிலி முழு t_abs = t_out + p0;

            முழு t_in_min = (t_abs - K + s0) / s0;
            எனில் (t_in_min < 0) {
                t_in_min = 0;
            }
            முழு t_in_max = t_abs / s0;
            எனில் (t_in_max >= T_in) {
                t_in_max = T_in - 1;
            }

            தசம sum = 0.0f;
            க்கு (முழு t_in = t_in_min; t_in <= t_in_max; ++t_in) {
                மாறிலி முழு k = t_abs - t_in * s0;
                sum += static_cast<தசம>(col[(oc * K + k) + t_in * K_OC]);
            }

            dst[idx] = static_cast<T>(sum);
        });
}

வெறுமை ggml_sycl_op_col2im_1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    மாறிலி ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0 != nullptr);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(src0->type == dst->type);

    மாறிலி int32_t s0 = ((மாறிலி int32_t *) dst->op_params)[0];
    மாறிலி int32_t OC = ((மாறிலி int32_t *) dst->op_params)[1];
    மாறிலி int32_t p0 = ((மாறிலி int32_t *) dst->op_params)[2];

    மாறிலி முழு K_OC  = (முழு) src0->ne[0];
    மாறிலி முழு T_in  = (முழு) src0->ne[1];
    மாறிலி முழு K     = K_OC / OC;
    மாறிலி முழு T_out = (முழு) dst->ne[0];

    GGML_ASSERT(OC > 0);
    GGML_ASSERT(K_OC % OC == 0);

    மாறிலி sycl::uint3 T_out_fd = init_fastdiv_மதிப்புகள்((uint32_t) T_out);

    மாறிலி முழு total = T_out * OC;

    dpct::queue_ptr stream = ctx.stream();

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F32:
            col2im_1d_sycl<தசம>(
                (மாறிலி தசம *) src0->data,
                (தசம *) dst->data,
                T_in, T_out_fd, K, K_OC, s0, p0, total, stream);
            நிறுத்து;
        நிலை GGML_TYPE_F16:
            col2im_1d_sycl<sycl::half>(
                (மாறிலி sycl::half *) src0->data,
                (sycl::half *) dst->data,
                T_in, T_out_fd, K, K_OC, s0, p0, total, stream);
            நிறுத்து;
#ifdef GGML_SYCL_HAS_BF16
        நிலை GGML_TYPE_BF16:
            col2im_1d_sycl<sycl::ext::oneapi::bfloat16>(
                (மாறிலி sycl::ext::oneapi::bfloat16 *) src0->data,
                (sycl::ext::oneapi::bfloat16 *) dst->data,
                T_in, T_out_fd, K, K_OC, s0, p0, total, stream);
            நிறுத்து;
#endif
        இயல்பு:
            GGML_ABORT("col2im_1d: unsupported type %d", src0->type);
    }
}
