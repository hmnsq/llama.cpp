#അന്തര്ഭാവ "col2im-1d.hpp"

template <typename T>
സ്ഥിതം ശൂന്യം col2im_1d_sycl(
        സ്ഥിരം T * col,
        T * dst,
        സ്ഥിരം പൂർണ്ണസംഖ്യ T_in,
        സ്ഥിരം sycl::uint3 T_out_fd,
        സ്ഥിരം പൂർണ്ണസംഖ്യ K,
        സ്ഥിരം പൂർണ്ണസംഖ്യ K_OC,
        സ്ഥിരം int32_t s0,
        സ്ഥിരം int32_t p0,
        സ്ഥിരം പൂർണ്ണസംഖ്യ total,
        dpct::queue_ptr stream) {

    സ്ഥിരം uint32_t block_size = SYCL_COL2IM_1D_BLOCK_SIZE;
    സ്ഥിരം uint32_t അക്കം_blocks = (uint32_t) ((total + block_size - 1) / block_size);

    stream->parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, അക്കം_blocks * block_size),
            sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) {
            സ്ഥിരം പൂർണ്ണസംഖ്യ idx = (പൂർണ്ണസംഖ്യ) item_ct1.get_global_id(2);
            എങ്കിൽ (idx >= total) {
                തിരികെ;
            }

            സ്ഥിരം sycl::uint2 qr = fast_div_modulo((uint32_t) idx, T_out_fd);
            സ്ഥിരം പൂർണ്ണസംഖ്യ oc    = (പൂർണ്ണസംഖ്യ) qr.x();
            സ്ഥിരം പൂർണ്ണസംഖ്യ t_out = (പൂർണ്ണസംഖ്യ) qr.y();
            സ്ഥിരം പൂർണ്ണസംഖ്യ t_abs = t_out + p0;

            പൂർണ്ണസംഖ്യ t_in_min = (t_abs - K + s0) / s0;
            എങ്കിൽ (t_in_min < 0) {
                t_in_min = 0;
            }
            പൂർണ്ണസംഖ്യ t_in_max = t_abs / s0;
            എങ്കിൽ (t_in_max >= T_in) {
                t_in_max = T_in - 1;
            }

            ദശാംശം sum = 0.0f;
            വേണ്ടി (പൂർണ്ണസംഖ്യ t_in = t_in_min; t_in <= t_in_max; ++t_in) {
                സ്ഥിരം പൂർണ്ണസംഖ്യ k = t_abs - t_in * s0;
                sum += static_cast<ദശാംശം>(col[(oc * K + k) + t_in * K_OC]);
            }

            dst[idx] = static_cast<T>(sum);
        });
}

ശൂന്യം ggml_sycl_op_col2im_1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0 != nullptr);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(src0->type == dst->type);

    സ്ഥിരം int32_t s0 = ((സ്ഥിരം int32_t *) dst->op_params)[0];
    സ്ഥിരം int32_t OC = ((സ്ഥിരം int32_t *) dst->op_params)[1];
    സ്ഥിരം int32_t p0 = ((സ്ഥിരം int32_t *) dst->op_params)[2];

    സ്ഥിരം പൂർണ്ണസംഖ്യ K_OC  = (പൂർണ്ണസംഖ്യ) src0->ne[0];
    സ്ഥിരം പൂർണ്ണസംഖ്യ T_in  = (പൂർണ്ണസംഖ്യ) src0->ne[1];
    സ്ഥിരം പൂർണ്ണസംഖ്യ K     = K_OC / OC;
    സ്ഥിരം പൂർണ്ണസംഖ്യ T_out = (പൂർണ്ണസംഖ്യ) dst->ne[0];

    GGML_ASSERT(OC > 0);
    GGML_ASSERT(K_OC % OC == 0);

    സ്ഥിരം sycl::uint3 T_out_fd = init_fastdiv_മൂല്യങ്ങൾ((uint32_t) T_out);

    സ്ഥിരം പൂർണ്ണസംഖ്യ total = T_out * OC;

    dpct::queue_ptr stream = ctx.stream();

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F32:
            col2im_1d_sycl<ദശാംശം>(
                (സ്ഥിരം ദശാംശം *) src0->data,
                (ദശാംശം *) dst->data,
                T_in, T_out_fd, K, K_OC, s0, p0, total, stream);
            നിർത്തുക;
        അവസ്ഥ GGML_TYPE_F16:
            col2im_1d_sycl<sycl::half>(
                (സ്ഥിരം sycl::half *) src0->data,
                (sycl::half *) dst->data,
                T_in, T_out_fd, K, K_OC, s0, p0, total, stream);
            നിർത്തുക;
#ifdef GGML_SYCL_HAS_BF16
        അവസ്ഥ GGML_TYPE_BF16:
            col2im_1d_sycl<sycl::ext::oneapi::bfloat16>(
                (സ്ഥിരം sycl::ext::oneapi::bfloat16 *) src0->data,
                (sycl::ext::oneapi::bfloat16 *) dst->data,
                T_in, T_out_fd, K, K_OC, s0, p0, total, stream);
            നിർത്തുക;
#endif
        സ്ഥിരസ്ഥിതി:
            GGML_ABORT("col2im_1d: unsupported type %d", src0->type);
    }
}
