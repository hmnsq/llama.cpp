#समावेश "col2im-1d.hpp"

ढाँचा <typename T>
अटल व्योम col2im_1d_sycl(
        स्थिर T * col,
        T * dst,
        स्थिर पूर्णांक T_in,
        स्थिर sycl::uint3 T_out_fd,
        स्थिर पूर्णांक K,
        स्थिर पूर्णांक K_OC,
        स्थिर int32_t s0,
        स्थिर int32_t p0,
        स्थिर पूर्णांक योग,
        dpct::queue_ptr stream) {

    स्थिर uint32_t block_आकार_2 = SYCL_COL2_IM_1_D_BLOCK_आकार;
    स्थिर uint32_t अंक_blocks = (uint32_t) ((योग + block_आकार_2 - 1) / block_आकार_2);

    stream->parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, अंक_blocks * block_आकार_2),
            sycl::range<3>(1, 1, block_आकार_2)),
        [=](sycl::nd_item<3> item_ct1) {
            स्थिर पूर्णांक idx = (पूर्णांक) item_ct1.get_global_id(2);
            अगर (idx >= योग) {
                वापस;
            }

            स्थिर sycl::uint2 qr = fast_div_modulo((uint32_t) idx, T_out_fd);
            स्थिर पूर्णांक oc    = (पूर्णांक) qr.x();
            स्थिर पूर्णांक t_out = (पूर्णांक) qr.y();
            स्थिर पूर्णांक t_abs = t_out + p0;

            पूर्णांक t_in_min = (t_abs - K + s0) / s0;
            अगर (t_in_min < 0) {
                t_in_min = 0;
            }
            पूर्णांक t_in_max = t_abs / s0;
            अगर (t_in_max >= T_in) {
                t_in_max = T_in - 1;
            }

            भग्न योग_2 = 0.0f;
            क्रम (पूर्णांक t_in = t_in_min; t_in <= t_in_max; ++t_in) {
                स्थिर पूर्णांक k = t_abs - t_in * s0;
                योग_2 += static_cast<भग्न>(col[(oc * K + k) + t_in * K_OC]);
            }

            dst[idx] = static_cast<T>(योग_2);
        });
}

व्योम ggml_sycl_op_col2im_1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    स्थिर ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0 != nullptr);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(src0->type == dst->type);

    स्थिर int32_t s0 = ((स्थिर int32_t *) dst->op_params)[0];
    स्थिर int32_t OC = ((स्थिर int32_t *) dst->op_params)[1];
    स्थिर int32_t p0 = ((स्थिर int32_t *) dst->op_params)[2];

    स्थिर पूर्णांक K_OC  = (पूर्णांक) src0->ne[0];
    स्थिर पूर्णांक T_in  = (पूर्णांक) src0->ne[1];
    स्थिर पूर्णांक K     = K_OC / OC;
    स्थिर पूर्णांक T_out = (पूर्णांक) dst->ne[0];

    GGML_ASSERT(OC > 0);
    GGML_ASSERT(K_OC % OC == 0);

    स्थिर sycl::uint3 T_out_fd = init_fastdiv_मूल्यावली((uint32_t) T_out);

    स्थिर पूर्णांक योग = T_out * OC;

    dpct::queue_ptr stream = ctx.stream();

    चयन (src0->type) {
        हाल GGML_TYPE_F32:
            col2im_1d_sycl<भग्न>(
                (स्थिर भग्न *) src0->तथ्य,
                (भग्न *) dst->तथ्य,
                T_in, T_out_fd, K, K_OC, s0, p0, योग, stream);
            अवरोध;
        हाल GGML_TYPE_F16:
            col2im_1d_sycl<sycl::half>(
                (स्थिर sycl::half *) src0->तथ्य,
                (sycl::half *) dst->तथ्य,
                T_in, T_out_fd, K, K_OC, s0, p0, योग, stream);
            अवरोध;
#अगर_घोषित GGML_SYCL_HAS_BF16
        हाल GGML_TYPE_BF16:
            col2im_1d_sycl<sycl::ext::oneapi::bfloat16>(
                (स्थिर sycl::ext::oneapi::bfloat16 *) src0->तथ्य,
                (sycl::ext::oneapi::bfloat16 *) dst->तथ्य,
                T_in, T_out_fd, K, K_OC, s0, p0, योग, stream);
            अवरोध;
#पूर्ण_अगर
        शेष:
            GGML_ABORT("col2im_1d: unsupported type %d", src0->type);
    }
}
