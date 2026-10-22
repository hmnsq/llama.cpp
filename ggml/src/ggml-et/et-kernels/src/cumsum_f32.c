//******************************************************************************
// CUMSUM F32 Kernel
// Computes an inclusive prefix sum along dim 0 for each row in higher dims.
// First-pass implementation: scalar and row-contiguous input/output only.
//******************************************************************************

#شامل "ggml_tensor.h"
#شامل "platform.h"

#شامل <stdint.h>

ساخت ggml_et_cumsum_params {
    ساخت ggml_tensor src0;
    ساخت ggml_tensor dst;
};

صحیح اندراج_نقطہ(ساخت ggml_et_cumsum_params * params, خالی * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    اگر (!kernel_env) {
        واپس -1;
    }

    صحیح thread_id   = get_relative_thread_id(kernel_env->shire_نقاب);
    صحیح عدد_دھاگے = get_عدد_دھاگے(kernel_env->shire_نقاب);

    اگر (thread_id < 0) {
        واپس 0;
    }

    اگر (params == 0 || ((uint64_t) params & 0x7) != 0) {
        واپس -1;
    }

    ساخت ggml_tensor * src0 = &params->src0;
    ساخت ggml_tensor * dst  = &params->dst;

    اگر (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        واپس -1;
    }

    اعشاری * src0_ڈیٹا = (اعشاری *) src0->ڈیٹا;
    اعشاری * dst_ڈیٹا  = (اعشاری *) dst->ڈیٹا;

    اگر (!src0_ڈیٹا || !dst_ڈیٹا) {
        واپس -1;
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

    اگر (snb0 != حجم(اعشاری) || dnb0 != حجم(اعشاری)) {
        واپس -1;
    }

    ثابت int64_t کل_صفیں     = ne1 * ne2 * ne3;
    ثابت int64_t صفیں_فی_گروہ = et_صفیں_فی_cacheline_گروہ(ne0, حجم(اعشاری));
    ثابت int64_t کل_groups   = (کل_صفیں + صفیں_فی_گروہ - 1) / صفیں_فی_گروہ;

    برائے (int64_t grp = thread_id; grp < کل_groups; grp += عدد_دھاگے) {
        ثابت int64_t صف_آغاز = grp * صفیں_فی_گروہ;
        int64_t       صف_انتہا   = صف_آغاز + صفیں_فی_گروہ;
        اگر (صف_انتہا > کل_صفیں) {
            صف_انتہا = کل_صفیں;
        }

        برائے (int64_t صف = صف_آغاز; صف < صف_انتہا; ++صف) {
            int64_t i1 = صف % ne1;
            int64_t i2 = (صف / ne1) % ne2;
            int64_t i3 = صف / (ne1 * ne2);

            ثابت اعشاری * src_صف = (ثابت اعشاری *) ((ثابت حرف *) src0_ڈیٹا + i1 * snb1 + i2 * snb2 + i3 * snb3);
            اعشاری *       dst_صف = (اعشاری *) ((حرف *) dst_ڈیٹا + i1 * dnb1 + i2 * dnb2 + i3 * dnb3);

            اعشاری acc = 0.0f;
            برائے (int64_t i0 = 0; i0 < ne0; ++i0) {
                acc += src_صف[i0];
                dst_صف[i0] = acc;
            }
        }
    }

    واپس 0;
}
