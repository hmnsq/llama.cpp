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

صحيح اندراج_نقطہ(بنية ggml_et_cumsum_params * params, فراغ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    إذا (!kernel_env) {
        أرجع -1;
    }

    صحيح thread_id   = get_relative_thread_id(kernel_env->shire_نقاب);
    صحيح عدد_دھاگے = get_عدد_دھاگے(kernel_env->shire_نقاب);

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

    عائم * src0_انکڑے = (عائم *) src0->انکڑے;
    عائم * dst_انکڑے  = (عائم *) dst->انکڑے;

    إذا (!src0_انکڑے || !dst_انکڑے) {
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

    ثابت int64_t کل_صفاں     = ne1 * ne2 * ne3;
    ثابت int64_t صفاں_فی_گروہ = et_صفاں_فی_cacheline_گروہ(ne0, حجم(عائم));
    ثابت int64_t کل_groups   = (کل_صفاں + صفاں_فی_گروہ - 1) / صفاں_فی_گروہ;

    لكل (int64_t grp = thread_id; grp < کل_groups; grp += عدد_دھاگے) {
        ثابت int64_t صف_شروع = grp * صفاں_فی_گروہ;
        int64_t       صف_سرا   = صف_شروع + صفاں_فی_گروہ;
        إذا (صف_سرا > کل_صفاں) {
            صف_سرا = کل_صفاں;
        }

        لكل (int64_t صف = صف_شروع; صف < صف_سرا; ++صف) {
            int64_t i1 = صف % ne1;
            int64_t i2 = (صف / ne1) % ne2;
            int64_t i3 = صف / (ne1 * ne2);

            ثابت عائم * src_صف = (ثابت عائم *) ((ثابت حرف *) src0_انکڑے + i1 * snb1 + i2 * snb2 + i3 * snb3);
            عائم *       dst_صف = (عائم *) ((حرف *) dst_انکڑے + i1 * dnb1 + i2 * dnb2 + i3 * dnb3);

            عائم acc = 0.0f;
            لكل (int64_t i0 = 0; i0 < ne0; ++i0) {
                acc += src_صف[i0];
                dst_صف[i0] = acc;
            }
        }
    }

    أرجع 0;
}
