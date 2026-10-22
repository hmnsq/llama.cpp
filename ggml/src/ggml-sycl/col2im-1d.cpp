#ضمن "col2im-1d.hpp"

سانچو <قسم_نالو T>
ساكن فراغ col2im_1d_sycl(
        ثابت T * کانو,
        T * dst,
        ثابت صحيح T_اندريون,
        ثابت sycl::uint3 T_ٻاهريون_fd,
        ثابت صحيح K,
        ثابت صحيح K_OC,
        ثابت int32_t s0,
        ثابت int32_t p0,
        ثابت صحيح ڪل,
        dpct::قطار_ptr stream) {

    ثابت uint32_t بلاڪ_سائيز_3 = SYCL_COL2_IM_1_D_بلاڪ_سائيز;
    ثابت uint32_t انگ_بلاڪون = (uint32_t) ((ڪل + بلاڪ_سائيز_3 - 1) / بلاڪ_سائيز_3);

    stream->parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, انگ_بلاڪون * بلاڪ_سائيز_3),
            sycl::range<3>(1, 1, بلاڪ_سائيز_3)),
        [=](sycl::nd_شي<3> شي_ct1) {
            ثابت صحيح idx = (صحيح) شي_ct1.get_global_id(2);
            إذا (idx >= ڪل) {
                أرجع;
            }

            ثابت sycl::uint2 qr = fast_div_modulo((uint32_t) idx, T_ٻاهريون_fd);
            ثابت صحيح oc    = (صحيح) qr.x();
            ثابت صحيح t_ٻاهريون = (صحيح) qr.y();
            ثابت صحيح t_abs = t_ٻاهريون + p0;

            صحيح t_اندريون_min = (t_abs - K + s0) / s0;
            إذا (t_اندريون_min < 0) {
                t_اندريون_min = 0;
            }
            صحيح t_اندريون_max = t_abs / s0;
            إذا (t_اندريون_max >= T_اندريون) {
                t_اندريون_max = T_اندريون - 1;
            }

            عائم جوڙ = 0.0f;
            لكل (صحيح t_اندريون = t_اندريون_min; t_اندريون <= t_اندريون_max; ++t_اندريون) {
                ثابت صحيح k = t_abs - t_اندريون * s0;
                جوڙ += static_cast<عائم>(کانو[(oc * K + k) + t_اندريون * K_OC]);
            }

            dst[idx] = static_cast<T>(جوڙ);
        });
}

فراغ ggml_sycl_op_col2im_1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0 != خالي_اشارو);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(src0->type == dst->type);

    ثابت int32_t s0 = ((ثابت int32_t *) dst->op_params)[0];
    ثابت int32_t OC = ((ثابت int32_t *) dst->op_params)[1];
    ثابت int32_t p0 = ((ثابت int32_t *) dst->op_params)[2];

    ثابت صحيح K_OC  = (صحيح) src0->ne[0];
    ثابت صحيح T_اندريون  = (صحيح) src0->ne[1];
    ثابت صحيح K     = K_OC / OC;
    ثابت صحيح T_ٻاهريون = (صحيح) dst->ne[0];

    GGML_ASSERT(OC > 0);
    GGML_ASSERT(K_OC % OC == 0);

    ثابت sycl::uint3 T_ٻاهريون_fd = init_fastdiv_قدرون((uint32_t) T_ٻاهريون);

    ثابت صحيح ڪل = T_ٻاهريون * OC;

    dpct::قطار_ptr stream = ctx.stream();

    بدل (src0->type) {
        حالة GGML_TYPE_F32:
            col2im_1d_sycl<عائم>(
                (ثابت عائم *) src0->معلومات,
                (عائم *) dst->معلومات,
                T_اندريون, T_ٻاهريون_fd, K, K_OC, s0, p0, ڪل, stream);
            اكسر;
        حالة GGML_TYPE_F16:
            col2im_1d_sycl<sycl::half>(
                (ثابت sycl::half *) src0->معلومات,
                (sycl::half *) dst->معلومات,
                T_اندريون, T_ٻاهريون_fd, K, K_OC, s0, p0, ڪل, stream);
            اكسر;
#جيڪڏهن_وصف_ٿيل GGML_SYCL_HAS_BF16
        حالة GGML_TYPE_BF16:
            col2im_1d_sycl<sycl::ext::oneapi::bfloat16>(
                (ثابت sycl::ext::oneapi::bfloat16 *) src0->معلومات,
                (sycl::ext::oneapi::bfloat16 *) dst->معلومات,
                T_اندريون, T_ٻاهريون_fd, K, K_OC, s0, p0, ڪل, stream);
            اكسر;
#جيڪڏهن_پورو
        افتراضي:
            GGML_ABORT("col2im_1d: unsupported type %d", src0->type);
    }
}
