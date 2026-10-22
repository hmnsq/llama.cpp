#অন্তর্ভাব "col2im-1d.hpp"

template <typename T>
স্থির শূন্য col2im_1d_sycl(
        ধ্রুবক T * col,
        T * dst,
        ধ্রুবক পূর্ণসংখ্যা T_in,
        ধ্রুবক sycl::uint3 T_out_fd,
        ধ্রুবক পূর্ণসংখ্যা K,
        ধ্রুবক পূর্ণসংখ্যা K_OC,
        ধ্রুবক int32_t s0,
        ধ্রুবক int32_t p0,
        ধ্রুবক পূর্ণসংখ্যা যোগ,
        dpct::queue_ptr stream) {

    ধ্রুবক uint32_t block_আকার_2 = SYCL_COL2_IM_1_D_BLOCK_আকার;
    ধ্রুবক uint32_t অঙ্ক_blocks = (uint32_t) ((যোগ + block_আকার_2 - 1) / block_আকার_2);

    stream->parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, অঙ্ক_blocks * block_আকার_2),
            sycl::range<3>(1, 1, block_আকার_2)),
        [=](sycl::nd_item<3> item_ct1) {
            ধ্রুবক পূর্ণসংখ্যা idx = (পূর্ণসংখ্যা) item_ct1.get_global_id(2);
            যদি (idx >= যোগ) {
                ফেরত;
            }

            ধ্রুবক sycl::uint2 qr = fast_div_modulo((uint32_t) idx, T_out_fd);
            ধ্রুবক পূর্ণসংখ্যা oc    = (পূর্ণসংখ্যা) qr.x();
            ধ্রুবক পূর্ণসংখ্যা t_out = (পূর্ণসংখ্যা) qr.y();
            ধ্রুবক পূর্ণসংখ্যা t_abs = t_out + p0;

            পূর্ণসংখ্যা t_in_min = (t_abs - K + s0) / s0;
            যদি (t_in_min < 0) {
                t_in_min = 0;
            }
            পূর্ণসংখ্যা t_in_max = t_abs / s0;
            যদি (t_in_max >= T_in) {
                t_in_max = T_in - 1;
            }

            দশমিক যোগ_2 = 0.0f;
            জন্য (পূর্ণসংখ্যা t_in = t_in_min; t_in <= t_in_max; ++t_in) {
                ধ্রুবক পূর্ণসংখ্যা k = t_abs - t_in * s0;
                যোগ_2 += static_cast<দশমিক>(col[(oc * K + k) + t_in * K_OC]);
            }

            dst[idx] = static_cast<T>(যোগ_2);
        });
}

শূন্য ggml_sycl_op_col2im_1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ধ্রুবক ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0 != nullptr);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(src0->type == dst->type);

    ধ্রুবক int32_t s0 = ((ধ্রুবক int32_t *) dst->op_params)[0];
    ধ্রুবক int32_t OC = ((ধ্রুবক int32_t *) dst->op_params)[1];
    ধ্রুবক int32_t p0 = ((ধ্রুবক int32_t *) dst->op_params)[2];

    ধ্রুবক পূর্ণসংখ্যা K_OC  = (পূর্ণসংখ্যা) src0->ne[0];
    ধ্রুবক পূর্ণসংখ্যা T_in  = (পূর্ণসংখ্যা) src0->ne[1];
    ধ্রুবক পূর্ণসংখ্যা K     = K_OC / OC;
    ধ্রুবক পূর্ণসংখ্যা T_out = (পূর্ণসংখ্যা) dst->ne[0];

    GGML_ASSERT(OC > 0);
    GGML_ASSERT(K_OC % OC == 0);

    ধ্রুবক sycl::uint3 T_out_fd = init_fastdiv_মানসমূহ((uint32_t) T_out);

    ধ্রুবক পূর্ণসংখ্যা যোগ = T_out * OC;

    dpct::queue_ptr stream = ctx.stream();

    নির্বাচন (src0->type) {
        ক্ষেত্র GGML_TYPE_F32:
            col2im_1d_sycl<দশমিক>(
                (ধ্রুবক দশমিক *) src0->তথ্য,
                (দশমিক *) dst->তথ্য,
                T_in, T_out_fd, K, K_OC, s0, p0, যোগ, stream);
            বিরতি;
        ক্ষেত্র GGML_TYPE_F16:
            col2im_1d_sycl<sycl::half>(
                (ধ্রুবক sycl::half *) src0->তথ্য,
                (sycl::half *) dst->তথ্য,
                T_in, T_out_fd, K, K_OC, s0, p0, যোগ, stream);
            বিরতি;
#ifdef GGML_SYCL_HAS_BF16
        ক্ষেত্র GGML_TYPE_BF16:
            col2im_1d_sycl<sycl::ext::oneapi::bfloat16>(
                (ধ্রুবক sycl::ext::oneapi::bfloat16 *) src0->তথ্য,
                (sycl::ext::oneapi::bfloat16 *) dst->তথ্য,
                T_in, T_out_fd, K, K_OC, s0, p0, যোগ, stream);
            বিরতি;
#endif
        পূর্বনির্ধারিত:
            GGML_ABORT("col2im_1d: unsupported type %d", src0->type);
    }
}
