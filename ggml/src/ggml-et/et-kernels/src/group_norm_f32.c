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

بنية ggml_et_گروهه_norm_params {
    بنية ggml_tensor src0;
    بنية ggml_tensor dst;
    int32_t            n_groups;
    عائم              eps;
};

صحيح داخلا_نقطو(بنية ggml_et_گروهه_norm_params * params, فراغ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    إذا (!kernel_env) {
        أرجع -1;
    }

    صحيح thread_id   = get_relative_thread_id(kernel_env->shire_نقاب);
    صحيح انگ_ڌاڳا = get_انگ_ڌاڳا(kernel_env->shire_نقاب);

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

    ثابت int64_t channels_في_گروهه = (ne2 + n_groups - 1) / n_groups;
    إذا (channels_في_گروهه <= 0) {
        أرجع -1;
    }

    ثابت int64_t سرگرم_groups        = (ne2 + channels_في_گروهه - 1) / channels_في_گروهه;
    ثابت int64_t ڪل_work           = سرگرم_groups * ne3;
    ثابت int64_t صفون_في_write_گروهه = et_صفون_في_cacheline_گروهه(ne0, حجم(عائم));

    لكل (int64_t work = thread_id; work < ڪل_work; work += انگ_ڌاڳا) {
        ثابت int64_t i3        = work / سرگرم_groups;
        ثابت int64_t گروهه_idx = work % سرگرم_groups;

        ثابت int64_t channel_شروعات = گروهه_idx * channels_في_گروهه;
        int64_t       channel_پڇاڙي   = channel_شروعات + channels_في_گروهه;
        إذا (channel_پڇاڙي > ne2) {
            channel_پڇاڙي = ne2;
        }

        ثابت int64_t channel_ڳڻپ = channel_پڇاڙي - channel_شروعات;
        إذا (channel_ڳڻپ <= 0) {
            استمر;
        }

        عائم جوڙ   = 0.0f;
        عائم denom = 0.0f;
        لكل (int64_t i2 = channel_شروعات; i2 < channel_پڇاڙي; ++i2) {
            لكل (int64_t i1 = 0; i1 < ne1; ++i1) {
                ثابت عائم * src_صف = (ثابت عائم *) ((ثابت حرف *) src0_معلومات + i3 * nb03 + i2 * nb02 + i1 * nb01);
                لكل (int64_t i0 = 0; i0 < ne0; ++i0) {
                    جوڙ += src_صف[i0];
                    denom += 1.0f;
                }
            }
        }

        ثابت عائم سراسري_2 = et_fdiv(جوڙ, denom);

        عائم var_جوڙ = 0.0f;
        لكل (int64_t i2 = channel_شروعات; i2 < channel_پڇاڙي; ++i2) {
            لكل (int64_t i1 = 0; i1 < ne1; ++i1) {
                ثابت عائم * src_صف = (ثابت عائم *) ((ثابت حرف *) src0_معلومات + i3 * nb03 + i2 * nb02 + i1 * nb01);
                لكل (int64_t i0 = 0; i0 < ne0; ++i0) {
                    ثابت عائم centered = src_صف[i0] - سراسري_2;
                    var_جوڙ += centered * centered;
                }
            }
        }

        ثابت عائم variance = et_fdiv(var_جوڙ, denom);
        ثابت عائم scale    = et_fdiv(1.0f, et_sqrtf(variance + eps));

        إذا (ne0 % 16 == 0) {
            لكل (int64_t i2 = channel_شروعات; i2 < channel_پڇاڙي; ++i2) {
                لكل (int64_t i1 = 0; i1 < ne1; ++i1) {
                    ثابت عائم * src_صف =
                        (ثابت عائم *) ((ثابت حرف *) src0_معلومات + i3 * nb03 + i2 * nb02 + i1 * nb01);
                    عائم * dst_صف = (عائم *) ((حرف *) dst_معلومات + i3 * nb3 + i2 * nb2 + i1 * nb1);
                    لكل (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_صف[i0] = (src_صف[i0] - سراسري_2) * scale;
                    }
                }
            }
        } وإلا {
            ثابت int64_t ڪل_صفون_اندريون_گروهه = channel_ڳڻپ * ne1;
            ثابت int64_t ڪل_write_groups  = (ڪل_صفون_اندريون_گروهه + صفون_في_write_گروهه - 1) / صفون_في_write_گروهه;

            لكل (int64_t write_گروهه = 0; write_گروهه < ڪل_write_groups; ++write_گروهه) {
                ثابت int64_t صف_شروعات = write_گروهه * صفون_في_write_گروهه;
                int64_t       صف_پڇاڙي   = صف_شروعات + صفون_في_write_گروهه;
                إذا (صف_پڇاڙي > ڪل_صفون_اندريون_گروهه) {
                    صف_پڇاڙي = ڪل_صفون_اندريون_گروهه;
                }

                لكل (int64_t صف = صف_شروعات; صف < صف_پڇاڙي; ++صف) {
                    ثابت int64_t local_i2 = صف / ne1;
                    ثابت int64_t i1       = صف % ne1;
                    ثابت int64_t i2       = channel_شروعات + local_i2;

                    ثابت عائم * src_صف =
                        (ثابت عائم *) ((ثابت حرف *) src0_معلومات + i3 * nb03 + i2 * nb02 + i1 * nb01);
                    عائم * dst_صف = (عائم *) ((حرف *) dst_معلومات + i3 * nb3 + i2 * nb2 + i1 * nb1);
                    لكل (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_صف[i0] = (src_صف[i0] - سراسري_2) * scale;
                    }
                }
            }
        }
    }

    أرجع 0;
}
