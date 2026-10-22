#ଅନ୍ତର୍ଭାଵ "col2im-1d.hpp"

template <typename T>
ସ୍ଥିର ଶୂନ୍ୟ col2im_1d_sycl(
        ଅଚଳ T * col,
        T * dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା T_in,
        ଅଚଳ sycl::uint3 T_out_fd,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା K,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା K_OC,
        ଅଚଳ int32_t s0,
        ଅଚଳ int32_t p0,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା total,
        dpct::queue_ptr stream) {

    ଅଚଳ uint32_t block_size = SYCL_COL2IM_1D_BLOCK_SIZE;
    ଅଚଳ uint32_t ଅଙ୍କ_blocks = (uint32_t) ((total + block_size - 1) / block_size);

    stream->parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, ଅଙ୍କ_blocks * block_size),
            sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) {
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା idx = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) item_ct1.get_global_id(2);
            ଯଦି (idx >= total) {
                ଫେରନ୍ତୁ;
            }

            ଅଚଳ sycl::uint2 qr = fast_div_modulo((uint32_t) idx, T_out_fd);
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା oc    = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) qr.x();
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା t_out = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) qr.y();
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା t_abs = t_out + p0;

            ପୂର୍ଣ୍ଣସଂଖ୍ୟା t_in_min = (t_abs - K + s0) / s0;
            ଯଦି (t_in_min < 0) {
                t_in_min = 0;
            }
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା t_in_max = t_abs / s0;
            ଯଦି (t_in_max >= T_in) {
                t_in_max = T_in - 1;
            }

            ଦଶମିକ sum = 0.0f;
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା t_in = t_in_min; t_in <= t_in_max; ++t_in) {
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = t_abs - t_in * s0;
                sum += static_cast<ଦଶମିକ>(col[(oc * K + k) + t_in * K_OC]);
            }

            dst[idx] = static_cast<T>(sum);
        });
}

ଶୂନ୍ୟ ggml_sycl_op_col2im_1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0 != nullptr);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(src0->type == dst->type);

    ଅଚଳ int32_t s0 = ((ଅଚଳ int32_t *) dst->op_params)[0];
    ଅଚଳ int32_t OC = ((ଅଚଳ int32_t *) dst->op_params)[1];
    ଅଚଳ int32_t p0 = ((ଅଚଳ int32_t *) dst->op_params)[2];

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା K_OC  = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) src0->ne[0];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା T_in  = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) src0->ne[1];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା K     = K_OC / OC;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା T_out = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) dst->ne[0];

    GGML_ASSERT(OC > 0);
    GGML_ASSERT(K_OC % OC == 0);

    ଅଚଳ sycl::uint3 T_out_fd = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ((uint32_t) T_out);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା total = T_out * OC;

    dpct::queue_ptr stream = ctx.stream();

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            col2im_1d_sycl<ଦଶମିକ>(
                (ଅଚଳ ଦଶମିକ *) src0->data,
                (ଦଶମିକ *) dst->data,
                T_in, T_out_fd, K, K_OC, s0, p0, total, stream);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_F16:
            col2im_1d_sycl<sycl::half>(
                (ଅଚଳ sycl::half *) src0->data,
                (sycl::half *) dst->data,
                T_in, T_out_fd, K, K_OC, s0, p0, total, stream);
            ଭାଙ୍ଗ;
#ifdef GGML_SYCL_HAS_BF16
        ସ୍ଥିତି GGML_TYPE_BF16:
            col2im_1d_sycl<sycl::ext::oneapi::bfloat16>(
                (ଅଚଳ sycl::ext::oneapi::bfloat16 *) src0->data,
                (sycl::ext::oneapi::bfloat16 *) dst->data,
                T_in, T_out_fd, K, K_OC, s0, p0, total, stream);
            ଭାଙ୍ଗ;
#endif
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            GGML_ABORT("col2im_1d: unsupported type %d", src0->type);
    }
}
