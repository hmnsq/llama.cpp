#شامل "col2im-1d.hpp"

سانچہ <قسم_نام T>
ساکن خالی col2im_1d_sycl(
        ثابت T * ستون,
        T * dst,
        ثابت صحیح T_اندر,
        ثابت sycl::uint3 T_برآمد_fd,
        ثابت صحیح K,
        ثابت صحیح K_OC,
        ثابت int32_t s0,
        ثابت int32_t p0,
        ثابت صحیح کل,
        dpct::قطار_ptr stream) {

    ثابت uint32_t بلاک_سائز_3 = SYCL_COL2_IM_1_D_بلاک_سائز;
    ثابت uint32_t عدد_بلاکیں = (uint32_t) ((کل + بلاک_سائز_3 - 1) / بلاک_سائز_3);

    stream->parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, عدد_بلاکیں * بلاک_سائز_3),
            sycl::range<3>(1, 1, بلاک_سائز_3)),
        [=](sycl::nd_شے<3> شے_ct1) {
            ثابت صحیح idx = (صحیح) شے_ct1.get_global_id(2);
            اگر (idx >= کل) {
                واپس;
            }

            ثابت sycl::uint2 qr = fast_div_modulo((uint32_t) idx, T_برآمد_fd);
            ثابت صحیح oc    = (صحیح) qr.x();
            ثابت صحیح t_برآمد = (صحیح) qr.y();
            ثابت صحیح t_abs = t_برآمد + p0;

            صحیح t_اندر_min = (t_abs - K + s0) / s0;
            اگر (t_اندر_min < 0) {
                t_اندر_min = 0;
            }
            صحیح t_اندر_max = t_abs / s0;
            اگر (t_اندر_max >= T_اندر) {
                t_اندر_max = T_اندر - 1;
            }

            اعشاری مجموع = 0.0f;
            برائے (صحیح t_اندر = t_اندر_min; t_اندر <= t_اندر_max; ++t_اندر) {
                ثابت صحیح k = t_abs - t_اندر * s0;
                مجموع += static_cast<اعشاری>(ستون[(oc * K + k) + t_اندر * K_OC]);
            }

            dst[idx] = static_cast<T>(مجموع);
        });
}

خالی ggml_sycl_op_col2im_1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0 != خالی_اشاریہ);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(src0->type == dst->type);

    ثابت int32_t s0 = ((ثابت int32_t *) dst->op_params)[0];
    ثابت int32_t OC = ((ثابت int32_t *) dst->op_params)[1];
    ثابت int32_t p0 = ((ثابت int32_t *) dst->op_params)[2];

    ثابت صحیح K_OC  = (صحیح) src0->ne[0];
    ثابت صحیح T_اندر  = (صحیح) src0->ne[1];
    ثابت صحیح K     = K_OC / OC;
    ثابت صحیح T_برآمد = (صحیح) dst->ne[0];

    GGML_ASSERT(OC > 0);
    GGML_ASSERT(K_OC % OC == 0);

    ثابت sycl::uint3 T_برآمد_fd = init_fastdiv_اقدار((uint32_t) T_برآمد);

    ثابت صحیح کل = T_برآمد * OC;

    dpct::قطار_ptr stream = ctx.stream();

    انتخاب (src0->type) {
        حالت GGML_TYPE_F32:
            col2im_1d_sycl<اعشاری>(
                (ثابت اعشاری *) src0->ڈیٹا,
                (اعشاری *) dst->ڈیٹا,
                T_اندر, T_برآمد_fd, K, K_OC, s0, p0, کل, stream);
            توڑو;
        حالت GGML_TYPE_F16:
            col2im_1d_sycl<sycl::half>(
                (ثابت sycl::half *) src0->ڈیٹا,
                (sycl::half *) dst->ڈیٹا,
                T_اندر, T_برآمد_fd, K, K_OC, s0, p0, کل, stream);
            توڑو;
#اگر_تعریف_شدہ GGML_SYCL_HAS_BF16
        حالت GGML_TYPE_BF16:
            col2im_1d_sycl<sycl::ext::oneapi::bfloat16>(
                (ثابت sycl::ext::oneapi::bfloat16 *) src0->ڈیٹا,
                (sycl::ext::oneapi::bfloat16 *) dst->ڈیٹا,
                T_اندر, T_برآمد_fd, K, K_OC, s0, p0, کل, stream);
            توڑو;
#اگر_ختم
        طےشدہ:
            GGML_ABORT("col2im_1d: unsupported type %d", src0->type);
    }
}
