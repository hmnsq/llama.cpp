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

بنية ggml_et_گروه_norm_params {
    بنية ggml_tensor src0;
    بنية ggml_tensor dst;
    int32_t            n_groups;
    عائم              eps;
};

صحيح مدخل_نقطه(بنية ggml_et_گروه_norm_params * params, فراغ * env) {
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

    ثابت عائم * src0_داده = (ثابت عائم *) src0->داده;
    عائم *       dst_داده  = (عائم *) dst->داده;

    إذا (!src0_داده || !dst_داده) {
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

    ثابت int64_t channels_در_هر_گروه = (ne2 + n_groups - 1) / n_groups;
    إذا (channels_در_هر_گروه <= 0) {
        أرجع -1;
    }

    ثابت int64_t فعال_groups        = (ne2 + channels_در_هر_گروه - 1) / channels_در_هر_گروه;
    ثابت int64_t مجموع_work           = فعال_groups * ne3;
    ثابت int64_t ردیفها_در_هر_write_گروه = et_ردیفها_در_هر_cacheline_گروه(ne0, حجم(عائم));

    لكل (int64_t work = thread_id; work < مجموع_work; work += رقم_نخها) {
        ثابت int64_t i3        = work / فعال_groups;
        ثابت int64_t گروه_idx = work % فعال_groups;

        ثابت int64_t channel_شروع = گروه_idx * channels_در_هر_گروه;
        int64_t       channel_انتها   = channel_شروع + channels_در_هر_گروه;
        إذا (channel_انتها > ne2) {
            channel_انتها = ne2;
        }

        ثابت int64_t channel_شمار = channel_انتها - channel_شروع;
        إذا (channel_شمار <= 0) {
            استمر;
        }

        عائم جمع   = 0.0f;
        عائم denom = 0.0f;
        لكل (int64_t i2 = channel_شروع; i2 < channel_انتها; ++i2) {
            لكل (int64_t i1 = 0; i1 < ne1; ++i1) {
                ثابت عائم * src_ردیف = (ثابت عائم *) ((ثابت حرف *) src0_داده + i3 * nb03 + i2 * nb02 + i1 * nb01);
                لكل (int64_t i0 = 0; i0 < ne0; ++i0) {
                    جمع += src_ردیف[i0];
                    denom += 1.0f;
                }
            }
        }

        ثابت عائم میانگین_2 = et_fdiv(جمع, denom);

        عائم var_جمع = 0.0f;
        لكل (int64_t i2 = channel_شروع; i2 < channel_انتها; ++i2) {
            لكل (int64_t i1 = 0; i1 < ne1; ++i1) {
                ثابت عائم * src_ردیف = (ثابت عائم *) ((ثابت حرف *) src0_داده + i3 * nb03 + i2 * nb02 + i1 * nb01);
                لكل (int64_t i0 = 0; i0 < ne0; ++i0) {
                    ثابت عائم centered = src_ردیف[i0] - میانگین_2;
                    var_جمع += centered * centered;
                }
            }
        }

        ثابت عائم variance = et_fdiv(var_جمع, denom);
        ثابت عائم scale    = et_fdiv(1.0f, et_sqrtf(variance + eps));

        إذا (ne0 % 16 == 0) {
            لكل (int64_t i2 = channel_شروع; i2 < channel_انتها; ++i2) {
                لكل (int64_t i1 = 0; i1 < ne1; ++i1) {
                    ثابت عائم * src_ردیف =
                        (ثابت عائم *) ((ثابت حرف *) src0_داده + i3 * nb03 + i2 * nb02 + i1 * nb01);
                    عائم * dst_ردیف = (عائم *) ((حرف *) dst_داده + i3 * nb3 + i2 * nb2 + i1 * nb1);
                    لكل (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_ردیف[i0] = (src_ردیف[i0] - میانگین_2) * scale;
                    }
                }
            }
        } وإلا {
            ثابت int64_t مجموع_ردیفها_درون_گروه = channel_شمار * ne1;
            ثابت int64_t مجموع_write_groups  = (مجموع_ردیفها_درون_گروه + ردیفها_در_هر_write_گروه - 1) / ردیفها_در_هر_write_گروه;

            لكل (int64_t write_گروه = 0; write_گروه < مجموع_write_groups; ++write_گروه) {
                ثابت int64_t ردیف_شروع = write_گروه * ردیفها_در_هر_write_گروه;
                int64_t       ردیف_انتها   = ردیف_شروع + ردیفها_در_هر_write_گروه;
                إذا (ردیف_انتها > مجموع_ردیفها_درون_گروه) {
                    ردیف_انتها = مجموع_ردیفها_درون_گروه;
                }

                لكل (int64_t ردیف = ردیف_شروع; ردیف < ردیف_انتها; ++ردیف) {
                    ثابت int64_t local_i2 = ردیف / ne1;
                    ثابت int64_t i1       = ردیف % ne1;
                    ثابت int64_t i2       = channel_شروع + local_i2;

                    ثابت عائم * src_ردیف =
                        (ثابت عائم *) ((ثابت حرف *) src0_داده + i3 * nb03 + i2 * nb02 + i1 * nb01);
                    عائم * dst_ردیف = (عائم *) ((حرف *) dst_داده + i3 * nb3 + i2 * nb2 + i1 * nb1);
                    لكل (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_ردیف[i0] = (src_ردیف[i0] - میانگین_2) * scale;
                    }
                }
            }
        }
    }

    أرجع 0;
}
