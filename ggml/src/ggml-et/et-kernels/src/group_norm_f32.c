//******************************************************************************
// GROUP_NORM F32 Kernel
// Baseline scalar implementation:
//   normalize over (ne0 * ne1 * channels_in_group) for each (group, batch).
//
// Parallelization:
// - Work is partitioned across (group, batch) pairs.
// - For non-cache-aligned ne0, writes are emitted in row-groups so each thread's
//   destination write footprint still spans an integer number of cache lines.
//******************************************************************************

#ضمن "ggml_tensor.h"
#ضمن "math_fp.h"
#ضمن "platform.h"

#ضمن <stdint.h>

بنية ggml_et_ډله_norm_params {
    بنية ggml_tensor src0;
    بنية ggml_tensor dst;
    int32_t            n_groups;
    عائم              eps;
};

صحيح ثبت_ټکی(بنية ggml_et_ډله_norm_params * params, فراغ * env) {
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

    ثابت عائم * src0_معلومات = (ثابت عائم *) src0->معلومات;
    عائم *       dst_معلومات  = (عائم *) dst->معلومات;

    إذا (!src0_معلومات || !dst_معلومات) {
        أرجع -1;
    }

    ثابت int32_t n_groups = params->n_groups;
    ثابت عائم   eps      = params->eps;

    إذا (n_groups <= 0 || eps < 0.0f) {
        أرجع -1;
    }

    ثابت int64_t ne0 = dst->ne[0];
    ثابت int64_t ne1 = dst->ne[1];
    ثابت int64_t ne2 = dst->ne[2];
    ثابت int64_t ne3 = dst->ne[3];

    إذا (src0->ne[0] != ne0 || src0->ne[1] != ne1 || src0->ne[2] != ne2 || src0->ne[3] != ne3) {
        أرجع -1;
    }

    ثابت int64_t nb1  = dst->nb[1];
    ثابت int64_t nb2  = dst->nb[2];
    ثابت int64_t nb3  = dst->nb[3];
    ثابت int64_t nb01 = src0->nb[1];
    ثابت int64_t nb02 = src0->nb[2];
    ثابت int64_t nb03 = src0->nb[3];

    ثابت int64_t channels_فی_ډله = (ne2 + n_groups - 1) / n_groups;
    إذا (channels_فی_ډله <= 0) {
        أرجع -1;
    }

    ثابت int64_t فعال_groups        = (ne2 + channels_فی_ډله - 1) / channels_فی_ډله;
    ثابت int64_t ټول_work           = فعال_groups * ne3;
    ثابت int64_t لیکې_فی_write_ډله = et_لیکې_فی_cacheline_ډله(ne0, حجم(عائم));

    لكل (int64_t work = thread_id; work < ټول_work; work += عدد_تارونه) {
        ثابت int64_t i3        = work / فعال_groups;
        ثابت int64_t ډله_idx = work % فعال_groups;

        ثابت int64_t channel_پیل = ډله_idx * channels_فی_ډله;
        int64_t       channel_پای_ټکی   = channel_پیل + channels_فی_ډله;
        إذا (channel_پای_ټکی > ne2) {
            channel_پای_ټکی = ne2;
        }

        ثابت int64_t channel_شمېر = channel_پای_ټکی - channel_پیل;
        إذا (channel_شمېر <= 0) {
            استمر;
        }

        عائم مجموعه   = 0.0f;
        عائم denom = 0.0f;
        لكل (int64_t i2 = channel_پیل; i2 < channel_پای_ټکی; ++i2) {
            لكل (int64_t i1 = 0; i1 < ne1; ++i1) {
                ثابت عائم * src_لیکه = (ثابت عائم *) ((ثابت حرف *) src0_معلومات + i3 * nb03 + i2 * nb02 + i1 * nb01);
                لكل (int64_t i0 = 0; i0 < ne0; ++i0) {
                    مجموعه += src_لیکه[i0];
                    denom += 1.0f;
                }
            }
        }

        ثابت عائم منځنۍ_2 = et_fdiv(مجموعه, denom);

        عائم var_مجموعه = 0.0f;
        لكل (int64_t i2 = channel_پیل; i2 < channel_پای_ټکی; ++i2) {
            لكل (int64_t i1 = 0; i1 < ne1; ++i1) {
                ثابت عائم * src_لیکه = (ثابت عائم *) ((ثابت حرف *) src0_معلومات + i3 * nb03 + i2 * nb02 + i1 * nb01);
                لكل (int64_t i0 = 0; i0 < ne0; ++i0) {
                    ثابت عائم centered = src_لیکه[i0] - منځنۍ_2;
                    var_مجموعه += centered * centered;
                }
            }
        }

        ثابت عائم variance = et_fdiv(var_مجموعه, denom);
        ثابت عائم scale    = et_fdiv(1.0f, et_sqrtf(variance + eps));

        إذا (ne0 % 16 == 0) {
            لكل (int64_t i2 = channel_پیل; i2 < channel_پای_ټکی; ++i2) {
                لكل (int64_t i1 = 0; i1 < ne1; ++i1) {
                    ثابت عائم * src_لیکه =
                        (ثابت عائم *) ((ثابت حرف *) src0_معلومات + i3 * nb03 + i2 * nb02 + i1 * nb01);
                    عائم * dst_لیکه = (عائم *) ((حرف *) dst_معلومات + i3 * nb3 + i2 * nb2 + i1 * nb1);
                    لكل (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_لیکه[i0] = (src_لیکه[i0] - منځنۍ_2) * scale;
                    }
                }
            }
        } وإلا {
            ثابت int64_t ټول_لیکې_دننه_ډله = channel_شمېر * ne1;
            ثابت int64_t ټول_write_groups  = (ټول_لیکې_دننه_ډله + لیکې_فی_write_ډله - 1) / لیکې_فی_write_ډله;

            لكل (int64_t write_ډله = 0; write_ډله < ټول_write_groups; ++write_ډله) {
                ثابت int64_t لیکه_پیل = write_ډله * لیکې_فی_write_ډله;
                int64_t       لیکه_پای_ټکی   = لیکه_پیل + لیکې_فی_write_ډله;
                إذا (لیکه_پای_ټکی > ټول_لیکې_دننه_ډله) {
                    لیکه_پای_ټکی = ټول_لیکې_دننه_ډله;
                }

                لكل (int64_t لیکه = لیکه_پیل; لیکه < لیکه_پای_ټکی; ++لیکه) {
                    ثابت int64_t local_i2 = لیکه / ne1;
                    ثابت int64_t i1       = لیکه % ne1;
                    ثابت int64_t i2       = channel_پیل + local_i2;

                    ثابت عائم * src_لیکه =
                        (ثابت عائم *) ((ثابت حرف *) src0_معلومات + i3 * nb03 + i2 * nb02 + i1 * nb01);
                    عائم * dst_لیکه = (عائم *) ((حرف *) dst_معلومات + i3 * nb3 + i2 * nb2 + i1 * nb1);
                    لكل (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_لیکه[i0] = (src_لیکه[i0] - منځنۍ_2) * scale;
                    }
                }
            }
        }
    }

    أرجع 0;
}
