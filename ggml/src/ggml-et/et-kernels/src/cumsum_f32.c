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

صحيح ثبت_ټکی(بنية ggml_et_cumsum_params * params, فراغ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    إذا (!kernel_env) {
        أرجع -1;
    }

    صحيح thread_id   = get_relative_thread_id(kernel_env->shire_ماسک);
    صحيح عدد_تارونه = get_عدد_تارونه(kernel_env->shire_ماسک);

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

    عائم * src0_معلومات = (عائم *) src0->معلومات;
    عائم * dst_معلومات  = (عائم *) dst->معلومات;

    إذا (!src0_معلومات || !dst_معلومات) {
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

    ثابت int64_t ټول_لیکې     = ne1 * ne2 * ne3;
    ثابت int64_t لیکې_فی_ډله = et_لیکې_فی_cacheline_ډله(ne0, حجم(عائم));
    ثابت int64_t ټول_groups   = (ټول_لیکې + لیکې_فی_ډله - 1) / لیکې_فی_ډله;

    لكل (int64_t grp = thread_id; grp < ټول_groups; grp += عدد_تارونه) {
        ثابت int64_t لیکه_پیل = grp * لیکې_فی_ډله;
        int64_t       لیکه_پای_ټکی   = لیکه_پیل + لیکې_فی_ډله;
        إذا (لیکه_پای_ټکی > ټول_لیکې) {
            لیکه_پای_ټکی = ټول_لیکې;
        }

        لكل (int64_t لیکه = لیکه_پیل; لیکه < لیکه_پای_ټکی; ++لیکه) {
            int64_t i1 = لیکه % ne1;
            int64_t i2 = (لیکه / ne1) % ne2;
            int64_t i3 = لیکه / (ne1 * ne2);

            ثابت عائم * src_لیکه = (ثابت عائم *) ((ثابت حرف *) src0_معلومات + i1 * snb1 + i2 * snb2 + i3 * snb3);
            عائم *       dst_لیکه = (عائم *) ((حرف *) dst_معلومات + i1 * dnb1 + i2 * dnb2 + i3 * dnb3);

            عائم acc = 0.0f;
            لكل (int64_t i0 = 0; i0 < ne0; ++i0) {
                acc += src_لیکه[i0];
                dst_لیکه[i0] = acc;
            }
        }
    }

    أرجع 0;
}
