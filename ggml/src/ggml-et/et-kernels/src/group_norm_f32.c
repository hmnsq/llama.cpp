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

#شامل "ggml_tensor.h"
#شامل "math_fp.h"
#شامل "platform.h"

#شامل <stdint.h>

ساخت ggml_et_گروہ_norm_params {
    ساخت ggml_tensor src0;
    ساخت ggml_tensor dst;
    int32_t            n_groups;
    اعشاری              eps;
};

صحیح اندراج_نقطہ(ساخت ggml_et_گروہ_norm_params * params, خالی * env) {
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

    ثابت اعشاری * src0_ڈیٹا = (ثابت اعشاری *) src0->ڈیٹا;
    اعشاری *       dst_ڈیٹا  = (اعشاری *) dst->ڈیٹا;

    اگر (!src0_ڈیٹا || !dst_ڈیٹا) {
        واپس -1;
    }

    ثابت int32_t n_groups = params->n_groups;
    ثابت اعشاری   eps      = params->eps;

    اگر (n_groups <= 0 || eps < 0.0f) {
        واپس -1;
    }

    ثابت int64_t ne0 = dst->ne[0];
    ثابت int64_t ne1 = dst->ne[1];
    ثابت int64_t ne2 = dst->ne[2];
    ثابت int64_t ne3 = dst->ne[3];

    اگر (src0->ne[0] != ne0 || src0->ne[1] != ne1 || src0->ne[2] != ne2 || src0->ne[3] != ne3) {
        واپس -1;
    }

    ثابت int64_t nb1  = dst->nb[1];
    ثابت int64_t nb2  = dst->nb[2];
    ثابت int64_t nb3  = dst->nb[3];
    ثابت int64_t nb01 = src0->nb[1];
    ثابت int64_t nb02 = src0->nb[2];
    ثابت int64_t nb03 = src0->nb[3];

    ثابت int64_t channels_فی_گروہ = (ne2 + n_groups - 1) / n_groups;
    اگر (channels_فی_گروہ <= 0) {
        واپس -1;
    }

    ثابت int64_t فعال_groups        = (ne2 + channels_فی_گروہ - 1) / channels_فی_گروہ;
    ثابت int64_t کل_work           = فعال_groups * ne3;
    ثابت int64_t صفیں_فی_write_گروہ = et_صفیں_فی_cacheline_گروہ(ne0, حجم(اعشاری));

    برائے (int64_t work = thread_id; work < کل_work; work += عدد_دھاگے) {
        ثابت int64_t i3        = work / فعال_groups;
        ثابت int64_t گروہ_idx = work % فعال_groups;

        ثابت int64_t channel_آغاز = گروہ_idx * channels_فی_گروہ;
        int64_t       channel_انتہا   = channel_آغاز + channels_فی_گروہ;
        اگر (channel_انتہا > ne2) {
            channel_انتہا = ne2;
        }

        ثابت int64_t channel_گنتی = channel_انتہا - channel_آغاز;
        اگر (channel_گنتی <= 0) {
            جاری;
        }

        اعشاری مجموع   = 0.0f;
        اعشاری denom = 0.0f;
        برائے (int64_t i2 = channel_آغاز; i2 < channel_انتہا; ++i2) {
            برائے (int64_t i1 = 0; i1 < ne1; ++i1) {
                ثابت اعشاری * src_صف = (ثابت اعشاری *) ((ثابت حرف *) src0_ڈیٹا + i3 * nb03 + i2 * nb02 + i1 * nb01);
                برائے (int64_t i0 = 0; i0 < ne0; ++i0) {
                    مجموع += src_صف[i0];
                    denom += 1.0f;
                }
            }
        }

        ثابت اعشاری اوسط_2 = et_fdiv(مجموع, denom);

        اعشاری var_مجموع = 0.0f;
        برائے (int64_t i2 = channel_آغاز; i2 < channel_انتہا; ++i2) {
            برائے (int64_t i1 = 0; i1 < ne1; ++i1) {
                ثابت اعشاری * src_صف = (ثابت اعشاری *) ((ثابت حرف *) src0_ڈیٹا + i3 * nb03 + i2 * nb02 + i1 * nb01);
                برائے (int64_t i0 = 0; i0 < ne0; ++i0) {
                    ثابت اعشاری centered = src_صف[i0] - اوسط_2;
                    var_مجموع += centered * centered;
                }
            }
        }

        ثابت اعشاری variance = et_fdiv(var_مجموع, denom);
        ثابت اعشاری scale    = et_fdiv(1.0f, et_sqrtf(variance + eps));

        اگر (ne0 % 16 == 0) {
            برائے (int64_t i2 = channel_آغاز; i2 < channel_انتہا; ++i2) {
                برائے (int64_t i1 = 0; i1 < ne1; ++i1) {
                    ثابت اعشاری * src_صف =
                        (ثابت اعشاری *) ((ثابت حرف *) src0_ڈیٹا + i3 * nb03 + i2 * nb02 + i1 * nb01);
                    اعشاری * dst_صف = (اعشاری *) ((حرف *) dst_ڈیٹا + i3 * nb3 + i2 * nb2 + i1 * nb1);
                    برائے (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_صف[i0] = (src_صف[i0] - اوسط_2) * scale;
                    }
                }
            }
        } ورنہ {
            ثابت int64_t کل_صفیں_اندر_گروہ = channel_گنتی * ne1;
            ثابت int64_t کل_write_groups  = (کل_صفیں_اندر_گروہ + صفیں_فی_write_گروہ - 1) / صفیں_فی_write_گروہ;

            برائے (int64_t write_گروہ = 0; write_گروہ < کل_write_groups; ++write_گروہ) {
                ثابت int64_t صف_آغاز = write_گروہ * صفیں_فی_write_گروہ;
                int64_t       صف_انتہا   = صف_آغاز + صفیں_فی_write_گروہ;
                اگر (صف_انتہا > کل_صفیں_اندر_گروہ) {
                    صف_انتہا = کل_صفیں_اندر_گروہ;
                }

                برائے (int64_t صف = صف_آغاز; صف < صف_انتہا; ++صف) {
                    ثابت int64_t local_i2 = صف / ne1;
                    ثابت int64_t i1       = صف % ne1;
                    ثابت int64_t i2       = channel_آغاز + local_i2;

                    ثابت اعشاری * src_صف =
                        (ثابت اعشاری *) ((ثابت حرف *) src0_ڈیٹا + i3 * nb03 + i2 * nb02 + i1 * nb01);
                    اعشاری * dst_صف = (اعشاری *) ((حرف *) dst_ڈیٹا + i3 * nb3 + i2 * nb2 + i1 * nb1);
                    برائے (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_صف[i0] = (src_صف[i0] - اوسط_2) * scale;
                    }
                }
            }
        }
    }

    واپس 0;
}
