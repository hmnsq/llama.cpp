#અન્તર્ભાવ "col2im-1d.hpp"

template <typename T>
સ્થિર શૂન્ય col2im_1d_sycl(
        અચલ T * col,
        T * dst,
        અચલ પૂર્ણાંક T_in,
        અચલ sycl::uint3 T_out_fd,
        અચલ પૂર્ણાંક K,
        અચલ પૂર્ણાંક K_OC,
        અચલ int32_t s0,
        અચલ int32_t p0,
        અચલ પૂર્ણાંક total,
        dpct::queue_ptr stream) {

    અચલ uint32_t block_size = SYCL_COL2IM_1D_BLOCK_SIZE;
    અચલ uint32_t અંક_blocks = (uint32_t) ((total + block_size - 1) / block_size);

    stream->parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, અંક_blocks * block_size),
            sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) {
            અચલ પૂર્ણાંક idx = (પૂર્ણાંક) item_ct1.get_global_id(2);
            જો (idx >= total) {
                પાછા;
            }

            અચલ sycl::uint2 qr = fast_div_modulo((uint32_t) idx, T_out_fd);
            અચલ પૂર્ણાંક oc    = (પૂર્ણાંક) qr.x();
            અચલ પૂર્ણાંક t_out = (પૂર્ણાંક) qr.y();
            અચલ પૂર્ણાંક t_abs = t_out + p0;

            પૂર્ણાંક t_in_min = (t_abs - K + s0) / s0;
            જો (t_in_min < 0) {
                t_in_min = 0;
            }
            પૂર્ણાંક t_in_max = t_abs / s0;
            જો (t_in_max >= T_in) {
                t_in_max = T_in - 1;
            }

            દશાંશ sum = 0.0f;
            માટે (પૂર્ણાંક t_in = t_in_min; t_in <= t_in_max; ++t_in) {
                અચલ પૂર્ણાંક k = t_abs - t_in * s0;
                sum += static_cast<દશાંશ>(col[(oc * K + k) + t_in * K_OC]);
            }

            dst[idx] = static_cast<T>(sum);
        });
}

શૂન્ય ggml_sycl_op_col2im_1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    અચલ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0 != nullptr);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(src0->type == dst->type);

    અચલ int32_t s0 = ((અચલ int32_t *) dst->op_params)[0];
    અચલ int32_t OC = ((અચલ int32_t *) dst->op_params)[1];
    અચલ int32_t p0 = ((અચલ int32_t *) dst->op_params)[2];

    અચલ પૂર્ણાંક K_OC  = (પૂર્ણાંક) src0->ne[0];
    અચલ પૂર્ણાંક T_in  = (પૂર્ણાંક) src0->ne[1];
    અચલ પૂર્ણાંક K     = K_OC / OC;
    અચલ પૂર્ણાંક T_out = (પૂર્ણાંક) dst->ne[0];

    GGML_ASSERT(OC > 0);
    GGML_ASSERT(K_OC % OC == 0);

    અચલ sycl::uint3 T_out_fd = init_fastdiv_કિંમતો((uint32_t) T_out);

    અચલ પૂર્ણાંક total = T_out * OC;

    dpct::queue_ptr stream = ctx.stream();

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F32:
            col2im_1d_sycl<દશાંશ>(
                (અચલ દશાંશ *) src0->data,
                (દશાંશ *) dst->data,
                T_in, T_out_fd, K, K_OC, s0, p0, total, stream);
            તોડો;
        સ્થિતિ GGML_TYPE_F16:
            col2im_1d_sycl<sycl::half>(
                (અચલ sycl::half *) src0->data,
                (sycl::half *) dst->data,
                T_in, T_out_fd, K, K_OC, s0, p0, total, stream);
            તોડો;
#ifdef GGML_SYCL_HAS_BF16
        સ્થિતિ GGML_TYPE_BF16:
            col2im_1d_sycl<sycl::ext::oneapi::bfloat16>(
                (અચલ sycl::ext::oneapi::bfloat16 *) src0->data,
                (sycl::ext::oneapi::bfloat16 *) dst->data,
                T_in, T_out_fd, K, K_OC, s0, p0, total, stream);
            તોડો;
#endif
        મૂળભૂત:
            GGML_ABORT("col2im_1d: unsupported type %d", src0->type);
    }
}
