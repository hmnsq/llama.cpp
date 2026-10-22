#ضمن "col2im-1d.hpp"

سانچا <قسم_ناں T>
ساكن فراغ col2im_1d_sycl(
        ثابت T * خانہ,
        T * dst,
        ثابت صحيح T_وچ,
        ثابت sycl::uint3 T_باہرلا_fd,
        ثابت صحيح K,
        ثابت صحيح K_OC,
        ثابت int32_t s0,
        ثابت int32_t p0,
        ثابت صحيح کل,
        dpct::قطار_ptr stream) {

    ثابت uint32_t بلاک_ناپ_3 = SYCL_COL2_IM_1_D_بلاک_ناپ;
    ثابت uint32_t عدد_بلاکاں = (uint32_t) ((کل + بلاک_ناپ_3 - 1) / بلاک_ناپ_3);

    stream->parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, عدد_بلاکاں * بلاک_ناپ_3),
            sycl::range<3>(1, 1, بلاک_ناپ_3)),
        [=](sycl::nd_شے<3> شے_ct1) {
            ثابت صحيح idx = (صحيح) شے_ct1.get_global_id(2);
            إذا (idx >= کل) {
                أرجع;
            }

            ثابت sycl::uint2 qr = fast_div_modulo((uint32_t) idx, T_باہرلا_fd);
            ثابت صحيح oc    = (صحيح) qr.x();
            ثابت صحيح t_باہرلا = (صحيح) qr.y();
            ثابت صحيح t_abs = t_باہرلا + p0;

            صحيح t_وچ_min = (t_abs - K + s0) / s0;
            إذا (t_وچ_min < 0) {
                t_وچ_min = 0;
            }
            صحيح t_وچ_max = t_abs / s0;
            إذا (t_وچ_max >= T_وچ) {
                t_وچ_max = T_وچ - 1;
            }

            عائم جوڑ = 0.0f;
            لكل (صحيح t_وچ = t_وچ_min; t_وچ <= t_وچ_max; ++t_وچ) {
                ثابت صحيح k = t_abs - t_وچ * s0;
                جوڑ += static_cast<عائم>(خانہ[(oc * K + k) + t_وچ * K_OC]);
            }

            dst[idx] = static_cast<T>(جوڑ);
        });
}

فراغ ggml_sycl_op_col2im_1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0 != خالی_سوچک);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(src0->type == dst->type);

    ثابت int32_t s0 = ((ثابت int32_t *) dst->op_params)[0];
    ثابت int32_t OC = ((ثابت int32_t *) dst->op_params)[1];
    ثابت int32_t p0 = ((ثابت int32_t *) dst->op_params)[2];

    ثابت صحيح K_OC  = (صحيح) src0->ne[0];
    ثابت صحيح T_وچ  = (صحيح) src0->ne[1];
    ثابت صحيح K     = K_OC / OC;
    ثابت صحيح T_باہرلا = (صحيح) dst->ne[0];

    GGML_ASSERT(OC > 0);
    GGML_ASSERT(K_OC % OC == 0);

    ثابت sycl::uint3 T_باہرلا_fd = init_fastdiv_اقدار((uint32_t) T_باہرلا);

    ثابت صحيح کل = T_باہرلا * OC;

    dpct::قطار_ptr stream = ctx.stream();

    بدل (src0->type) {
        حالة GGML_TYPE_F32:
            col2im_1d_sycl<عائم>(
                (ثابت عائم *) src0->انکڑے,
                (عائم *) dst->انکڑے,
                T_وچ, T_باہرلا_fd, K, K_OC, s0, p0, کل, stream);
            اكسر;
        حالة GGML_TYPE_F16:
            col2im_1d_sycl<sycl::half>(
                (ثابت sycl::half *) src0->انکڑے,
                (sycl::half *) dst->انکڑے,
                T_وچ, T_باہرلا_fd, K, K_OC, s0, p0, کل, stream);
            اكسر;
#جے_تعریف_شدہ GGML_SYCL_HAS_BF16
        حالة GGML_TYPE_BF16:
            col2im_1d_sycl<sycl::ext::oneapi::bfloat16>(
                (ثابت sycl::ext::oneapi::bfloat16 *) src0->انکڑے,
                (sycl::ext::oneapi::bfloat16 *) dst->انکڑے,
                T_وچ, T_باہرلا_fd, K, K_OC, s0, p0, کل, stream);
            اكسر;
#جے_انت
        افتراضي:
            GGML_ABORT("col2im_1d: unsupported type %d", src0->type);
    }
}
