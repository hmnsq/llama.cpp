//******************************************************************************
// CUMSUM F32 Kernel
// Computes an inclusive prefix sum along dim 0 for each row in higher dims.
// First-pass implementation: scalar and row-contiguous input/output only.
//******************************************************************************

#ضمن "ggml_tensor.h"
#ضمن "platform.h"

#ضمن <stdint.h>

بنية ggml_et_cumsum_params {
    بنية ggml_tensor src0;
    بنية ggml_tensor dst;
};

صحيح مدخل_نقطه(بنية ggml_et_cumsum_params * params, فراغ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    إذا (!kernel_env) {
        أرجع -1;
    }

    صحيح thread_id   = get_relative_thread_id(kernel_env->shire_ماسک);
    صحيح رقم_نخها = get_رقم_نخها(kernel_env->shire_ماسک);

    إذا (thread_id < 0) {
        أرجع 0;
    }

    إذا (params == 0 || ((uint64_t) params & 0x7) != 0) {
        أرجع -1;
    }

    بنية ggml_tensor * src0 = &params->src0;
    بنية ggml_tensor * dst  = &params->dst;

    إذا (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        أرجع -1;
    }

    عائم * src0_داده = (عائم *) src0->داده;
    عائم * dst_داده  = (عائم *) dst->داده;

    إذا (!src0_داده || !dst_داده) {
        أرجع -1;
    }

    ثابت int64_t ne0 = src0->ne[0];
    ثابت int64_t ne1 = src0->ne[1];
    ثابت int64_t ne2 = src0->ne[2];
    ثابت int64_t ne3 = src0->ne[3];

    ثابت size_t snb0 = src0->nb[0];
    ثابت size_t snb1 = src0->nb[1];
    ثابت size_t snb2 = src0->nb[2];
    ثابت size_t snb3 = src0->nb[3];

    ثابت size_t dnb0 = dst->nb[0];
    ثابت size_t dnb1 = dst->nb[1];
    ثابت size_t dnb2 = dst->nb[2];
    ثابت size_t dnb3 = dst->nb[3];

    إذا (snb0 != حجم(عائم) || dnb0 != حجم(عائم)) {
        أرجع -1;
    }

    ثابت int64_t مجموع_ردیفها     = ne1 * ne2 * ne3;
    ثابت int64_t ردیفها_در_هر_گروه = et_ردیفها_در_هر_cacheline_گروه(ne0, حجم(عائم));
    ثابت int64_t مجموع_groups   = (مجموع_ردیفها + ردیفها_در_هر_گروه - 1) / ردیفها_در_هر_گروه;

    لكل (int64_t grp = thread_id; grp < مجموع_groups; grp += رقم_نخها) {
        ثابت int64_t ردیف_شروع = grp * ردیفها_در_هر_گروه;
        int64_t       ردیف_انتها   = ردیف_شروع + ردیفها_در_هر_گروه;
        إذا (ردیف_انتها > مجموع_ردیفها) {
            ردیف_انتها = مجموع_ردیفها;
        }

        لكل (int64_t ردیف = ردیف_شروع; ردیف < ردیف_انتها; ++ردیف) {
            int64_t i1 = ردیف % ne1;
            int64_t i2 = (ردیف / ne1) % ne2;
            int64_t i3 = ردیف / (ne1 * ne2);

            ثابت عائم * src_ردیف = (ثابت عائم *) ((ثابت حرف *) src0_داده + i1 * snb1 + i2 * snb2 + i3 * snb3);
            عائم *       dst_ردیف = (عائم *) ((حرف *) dst_داده + i1 * dnb1 + i2 * dnb2 + i3 * dnb3);

            عائم acc = 0.0f;
            لكل (int64_t i0 = 0; i0 < ne0; ++i0) {
                acc += src_ردیف[i0];
                dst_ردیف[i0] = acc;
            }
        }
    }

    أرجع 0;
}
