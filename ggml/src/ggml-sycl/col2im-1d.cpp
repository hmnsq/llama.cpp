#ضمن "col2im-1d.hpp"

کينډۍ <ډول_نوم T>
ساكن فراغ col2im_1d_sycl(
        ثابت T * ستون,
        T * dst,
        ثابت صحيح T_دننه,
        ثابت sycl::uint3 T_بهر_fd,
        ثابت صحيح K,
        ثابت صحيح K_OC,
        ثابت int32_t s0,
        ثابت int32_t p0,
        ثابت صحيح ټول,
        dpct::کتار_ptr stream) {

    ثابت uint32_t بلاک_اندازه_3 = SYCL_COL2_IM_1_D_بلاک_اندازه;
    ثابت uint32_t عدد_بلاکونه = (uint32_t) ((ټول + بلاک_اندازه_3 - 1) / بلاک_اندازه_3);

    stream->parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, عدد_بلاکونه * بلاک_اندازه_3),
            sycl::range<3>(1, 1, بلاک_اندازه_3)),
        [=](sycl::nd_توکی<3> توکی_ct1) {
            ثابت صحيح idx = (صحيح) توکی_ct1.get_global_id(2);
            إذا (idx >= ټول) {
                أرجع;
            }

            ثابت sycl::uint2 qr = fast_div_modulo((uint32_t) idx, T_بهر_fd);
            ثابت صحيح oc    = (صحيح) qr.x();
            ثابت صحيح t_بهر = (صحيح) qr.y();
            ثابت صحيح t_abs = t_بهر + p0;

            صحيح t_دننه_min = (t_abs - K + s0) / s0;
            إذا (t_دننه_min < 0) {
                t_دننه_min = 0;
            }
            صحيح t_دننه_max = t_abs / s0;
            إذا (t_دننه_max >= T_دننه) {
                t_دننه_max = T_دننه - 1;
            }

            عائم مجموعه = 0.0f;
            لكل (صحيح t_دننه = t_دننه_min; t_دننه <= t_دننه_max; ++t_دننه) {
                ثابت صحيح k = t_abs - t_دننه * s0;
                مجموعه += static_cast<عائم>(ستون[(oc * K + k) + t_دننه * K_OC]);
            }

            dst[idx] = static_cast<T>(مجموعه);
        });
}

فراغ ggml_sycl_op_col2im_1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0 != تش_اشاره);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(src0->type == dst->type);

    ثابت int32_t s0 = ((ثابت int32_t *) dst->op_params)[0];
    ثابت int32_t OC = ((ثابت int32_t *) dst->op_params)[1];
    ثابت int32_t p0 = ((ثابت int32_t *) dst->op_params)[2];

    ثابت صحيح K_OC  = (صحيح) src0->ne[0];
    ثابت صحيح T_دننه  = (صحيح) src0->ne[1];
    ثابت صحيح K     = K_OC / OC;
    ثابت صحيح T_بهر = (صحيح) dst->ne[0];

    GGML_ASSERT(OC > 0);
    GGML_ASSERT(K_OC % OC == 0);

    ثابت sycl::uint3 T_بهر_fd = init_fastdiv_ارزښتونه((uint32_t) T_بهر);

    ثابت صحيح ټول = T_بهر * OC;

    dpct::کتار_ptr stream = ctx.stream();

    بدل (src0->type) {
        حالة GGML_TYPE_F32:
            col2im_1d_sycl<عائم>(
                (ثابت عائم *) src0->معلومات,
                (عائم *) dst->معلومات,
                T_دننه, T_بهر_fd, K, K_OC, s0, p0, ټول, stream);
            اكسر;
        حالة GGML_TYPE_F16:
            col2im_1d_sycl<sycl::half>(
                (ثابت sycl::half *) src0->معلومات,
                (sycl::half *) dst->معلومات,
                T_دننه, T_بهر_fd, K, K_OC, s0, p0, ټول, stream);
            اكسر;
#که_تعریف_شوی GGML_SYCL_HAS_BF16
        حالة GGML_TYPE_BF16:
            col2im_1d_sycl<sycl::ext::oneapi::bfloat16>(
                (ثابت sycl::ext::oneapi::bfloat16 *) src0->معلومات,
                (sycl::ext::oneapi::bfloat16 *) dst->معلومات,
                T_دننه, T_بهر_fd, K, K_OC, s0, p0, ټول, stream);
            اكسر;
#د_که_پای
        افتراضي:
            GGML_ABORT("col2im_1d: unsupported type %d", src0->type);
    }
}
