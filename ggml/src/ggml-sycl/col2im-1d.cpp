#ضمن "col2im-1d.hpp"

قالب <نام_نوع T>
ساكن فراغ col2im_1d_sycl(
        ثابت T * ستون,
        T * dst,
        ثابت صحيح T_درون,
        ثابت sycl::uint3 T_بیرون_fd,
        ثابت صحيح K,
        ثابت صحيح K_OC,
        ثابت int32_t s0,
        ثابت int32_t p0,
        ثابت صحيح مجموع,
        dpct::صف_ptr stream) {

    ثابت uint32_t بلوک_اندازه_3 = SYCL_COL2_IM_1_D_بلوک_اندازه;
    ثابت uint32_t رقم_بلوکها = (uint32_t) ((مجموع + بلوک_اندازه_3 - 1) / بلوک_اندازه_3);

    stream->parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, رقم_بلوکها * بلوک_اندازه_3),
            sycl::range<3>(1, 1, بلوک_اندازه_3)),
        [=](sycl::nd_قلم<3> قلم_ct1) {
            ثابت صحيح idx = (صحيح) قلم_ct1.get_global_id(2);
            إذا (idx >= مجموع) {
                أرجع;
            }

            ثابت sycl::uint2 qr = fast_div_modulo((uint32_t) idx, T_بیرون_fd);
            ثابت صحيح oc    = (صحيح) qr.x();
            ثابت صحيح t_بیرون = (صحيح) qr.y();
            ثابت صحيح t_abs = t_بیرون + p0;

            صحيح t_درون_min = (t_abs - K + s0) / s0;
            إذا (t_درون_min < 0) {
                t_درون_min = 0;
            }
            صحيح t_درون_max = t_abs / s0;
            إذا (t_درون_max >= T_درون) {
                t_درون_max = T_درون - 1;
            }

            عائم جمع = 0.0f;
            لكل (صحيح t_درون = t_درون_min; t_درون <= t_درون_max; ++t_درون) {
                ثابت صحيح k = t_abs - t_درون * s0;
                جمع += static_cast<عائم>(ستون[(oc * K + k) + t_درون * K_OC]);
            }

            dst[idx] = static_cast<T>(جمع);
        });
}

فراغ ggml_sycl_op_col2im_1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0 != اشاره_تهی);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(src0->type == dst->type);

    ثابت int32_t s0 = ((ثابت int32_t *) dst->op_params)[0];
    ثابت int32_t OC = ((ثابت int32_t *) dst->op_params)[1];
    ثابت int32_t p0 = ((ثابت int32_t *) dst->op_params)[2];

    ثابت صحيح K_OC  = (صحيح) src0->ne[0];
    ثابت صحيح T_درون  = (صحيح) src0->ne[1];
    ثابت صحيح K     = K_OC / OC;
    ثابت صحيح T_بیرون = (صحيح) dst->ne[0];

    GGML_ASSERT(OC > 0);
    GGML_ASSERT(K_OC % OC == 0);

    ثابت sycl::uint3 T_بیرون_fd = init_fastdiv_مقادیر((uint32_t) T_بیرون);

    ثابت صحيح مجموع = T_بیرون * OC;

    dpct::صف_ptr stream = ctx.stream();

    بدل (src0->type) {
        حالة GGML_TYPE_F32:
            col2im_1d_sycl<عائم>(
                (ثابت عائم *) src0->داده,
                (عائم *) dst->داده,
                T_درون, T_بیرون_fd, K, K_OC, s0, p0, مجموع, stream);
            اكسر;
        حالة GGML_TYPE_F16:
            col2im_1d_sycl<sycl::half>(
                (ثابت sycl::half *) src0->داده,
                (sycl::half *) dst->داده,
                T_درون, T_بیرون_fd, K, K_OC, s0, p0, مجموع, stream);
            اكسر;
#اگر_تعریف_شده GGML_SYCL_HAS_BF16
        حالة GGML_TYPE_BF16:
            col2im_1d_sycl<sycl::ext::oneapi::bfloat16>(
                (ثابت sycl::ext::oneapi::bfloat16 *) src0->داده,
                (sycl::ext::oneapi::bfloat16 *) dst->داده,
                T_درون, T_بیرون_fd, K, K_OC, s0, p0, مجموع, stream);
            اكسر;
#پایان_اگر
        افتراضي:
            GGML_ABORT("col2im_1d: unsupported type %d", src0->type);
    }
}
