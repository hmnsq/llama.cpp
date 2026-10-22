#شامل "col2im-1d.hpp"

template <typename T>
ایستا تهی col2im_1d_sycl(
        ثابت T * col,
        T * dst,
        ثابت صحیح T_in,
        ثابت sycl::uint3 T_out_fd,
        ثابت صحیح K,
        ثابت صحیح K_OC,
        ثابت int32_t s0,
        ثابت int32_t p0,
        ثابت صحیح total,
        dpct::queue_ptr stream) {

    ثابت uint32_t block_size = SYCL_COL2IM_1D_BLOCK_SIZE;
    ثابت uint32_t رقم_blocks = (uint32_t) ((total + block_size - 1) / block_size);

    stream->parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, رقم_blocks * block_size),
            sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) {
            ثابت صحیح idx = (صحیح) item_ct1.get_global_id(2);
            اگر (idx >= total) {
                بازگشت;
            }

            ثابت sycl::uint2 qr = fast_div_modulo((uint32_t) idx, T_out_fd);
            ثابت صحیح oc    = (صحیح) qr.x();
            ثابت صحیح t_out = (صحیح) qr.y();
            ثابت صحیح t_abs = t_out + p0;

            صحیح t_in_min = (t_abs - K + s0) / s0;
            اگر (t_in_min < 0) {
                t_in_min = 0;
            }
            صحیح t_in_max = t_abs / s0;
            اگر (t_in_max >= T_in) {
                t_in_max = T_in - 1;
            }

            اعشاری sum = 0.0f;
            برای (صحیح t_in = t_in_min; t_in <= t_in_max; ++t_in) {
                ثابت صحیح k = t_abs - t_in * s0;
                sum += static_cast<اعشاری>(col[(oc * K + k) + t_in * K_OC]);
            }

            dst[idx] = static_cast<T>(sum);
        });
}

تهی ggml_sycl_op_col2im_1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0 != nullptr);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(src0->type == dst->type);

    ثابت int32_t s0 = ((ثابت int32_t *) dst->op_params)[0];
    ثابت int32_t OC = ((ثابت int32_t *) dst->op_params)[1];
    ثابت int32_t p0 = ((ثابت int32_t *) dst->op_params)[2];

    ثابت صحیح K_OC  = (صحیح) src0->ne[0];
    ثابت صحیح T_in  = (صحیح) src0->ne[1];
    ثابت صحیح K     = K_OC / OC;
    ثابت صحیح T_out = (صحیح) dst->ne[0];

    GGML_ASSERT(OC > 0);
    GGML_ASSERT(K_OC % OC == 0);

    ثابت sycl::uint3 T_out_fd = init_fastdiv_مقادیر((uint32_t) T_out);

    ثابت صحیح total = T_out * OC;

    dpct::queue_ptr stream = ctx.stream();

    انتخاب (src0->type) {
        حالت GGML_TYPE_F32:
            col2im_1d_sycl<اعشاری>(
                (ثابت اعشاری *) src0->data,
                (اعشاری *) dst->data,
                T_in, T_out_fd, K, K_OC, s0, p0, total, stream);
            بشکن;
        حالت GGML_TYPE_F16:
            col2im_1d_sycl<sycl::half>(
                (ثابت sycl::half *) src0->data,
                (sycl::half *) dst->data,
                T_in, T_out_fd, K, K_OC, s0, p0, total, stream);
            بشکن;
#ifdef GGML_SYCL_HAS_BF16
        حالت GGML_TYPE_BF16:
            col2im_1d_sycl<sycl::ext::oneapi::bfloat16>(
                (ثابت sycl::ext::oneapi::bfloat16 *) src0->data,
                (sycl::ext::oneapi::bfloat16 *) dst->data,
                T_in, T_out_fd, K, K_OC, s0, p0, total, stream);
            بشکن;
#endif
        پیشفرض:
            GGML_ABORT("col2im_1d: unsupported type %d", src0->type);
    }
}
