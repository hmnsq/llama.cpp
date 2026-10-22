#অন্তর্ভাব "col2im-1d.hpp"

template <typename T>
স্থির শূন্য col2im_1d_sycl(
        ধ্রুব T * col,
        T * dst,
        ধ্রুব পূর্ণাঙ্ক T_in,
        ধ্রুব sycl::uint3 T_out_fd,
        ধ্রুব পূর্ণাঙ্ক K,
        ধ্রুব পূর্ণাঙ্ক K_OC,
        ধ্রুব int32_t s0,
        ধ্রুব int32_t p0,
        ধ্রুব পূর্ণাঙ্ক total,
        dpct::queue_ptr stream) {

    ধ্রুব uint32_t block_size = SYCL_COL2IM_1D_BLOCK_SIZE;
    ধ্রুব uint32_t অঙ্ক_blocks = (uint32_t) ((total + block_size - 1) / block_size);

    stream->parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, অঙ্ক_blocks * block_size),
            sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) {
            ধ্রুব পূর্ণাঙ্ক idx = (পূর্ণাঙ্ক) item_ct1.get_global_id(2);
            যদি (idx >= total) {
                প্রত্যাগম;
            }

            ধ্রুব sycl::uint2 qr = fast_div_modulo((uint32_t) idx, T_out_fd);
            ধ্রুব পূর্ণাঙ্ক oc    = (পূর্ণাঙ্ক) qr.x();
            ধ্রুব পূর্ণাঙ্ক t_out = (পূর্ণাঙ্ক) qr.y();
            ধ্রুব পূর্ণাঙ্ক t_abs = t_out + p0;

            পূর্ণাঙ্ক t_in_min = (t_abs - K + s0) / s0;
            যদি (t_in_min < 0) {
                t_in_min = 0;
            }
            পূর্ণাঙ্ক t_in_max = t_abs / s0;
            যদি (t_in_max >= T_in) {
                t_in_max = T_in - 1;
            }

            ভিন্ন sum = 0.0f;
            প্রতি (পূর্ণাঙ্ক t_in = t_in_min; t_in <= t_in_max; ++t_in) {
                ধ্রুব পূর্ণাঙ্ক k = t_abs - t_in * s0;
                sum += static_cast<ভিন্ন>(col[(oc * K + k) + t_in * K_OC]);
            }

            dst[idx] = static_cast<T>(sum);
        });
}

শূন্য ggml_sycl_op_col2im_1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ধ্রুব ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0 != nullptr);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(src0->type == dst->type);

    ধ্রুব int32_t s0 = ((ধ্রুব int32_t *) dst->op_params)[0];
    ধ্রুব int32_t OC = ((ধ্রুব int32_t *) dst->op_params)[1];
    ধ্রুব int32_t p0 = ((ধ্রুব int32_t *) dst->op_params)[2];

    ধ্রুব পূর্ণাঙ্ক K_OC  = (পূর্ণাঙ্ক) src0->ne[0];
    ধ্রুব পূর্ণাঙ্ক T_in  = (পূর্ণাঙ্ক) src0->ne[1];
    ধ্রুব পূর্ণাঙ্ক K     = K_OC / OC;
    ধ্রুব পূর্ণাঙ্ক T_out = (পূর্ণাঙ্ক) dst->ne[0];

    GGML_ASSERT(OC > 0);
    GGML_ASSERT(K_OC % OC == 0);

    ধ্রুব sycl::uint3 T_out_fd = init_fastdiv_মানসমূহ((uint32_t) T_out);

    ধ্রুব পূর্ণাঙ্ক total = T_out * OC;

    dpct::queue_ptr stream = ctx.stream();

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F32:
            col2im_1d_sycl<ভিন্ন>(
                (ধ্রুব ভিন্ন *) src0->data,
                (ভিন্ন *) dst->data,
                T_in, T_out_fd, K, K_OC, s0, p0, total, stream);
            ভঙ্গ;
        পক্ষ GGML_TYPE_F16:
            col2im_1d_sycl<sycl::half>(
                (ধ্রুব sycl::half *) src0->data,
                (sycl::half *) dst->data,
                T_in, T_out_fd, K, K_OC, s0, p0, total, stream);
            ভঙ্গ;
#ifdef GGML_SYCL_HAS_BF16
        পক্ষ GGML_TYPE_BF16:
            col2im_1d_sycl<sycl::ext::oneapi::bfloat16>(
                (ধ্রুব sycl::ext::oneapi::bfloat16 *) src0->data,
                (sycl::ext::oneapi::bfloat16 *) dst->data,
                T_in, T_out_fd, K, K_OC, s0, p0, total, stream);
            ভঙ্গ;
#endif
        স্বভাব:
            GGML_ABORT("col2im_1d: unsupported type %d", src0->type);
    }
}
