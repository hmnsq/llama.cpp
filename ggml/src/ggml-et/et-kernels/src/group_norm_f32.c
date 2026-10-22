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

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml_tensor.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "math_fp.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "platform.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>

ਬਣਤਰ ggml_et_ਸਮੂਹ_norm_params {
    ਬਣਤਰ ggml_tensor src0;
    ਬਣਤਰ ggml_tensor dst;
    int32_t            n_groups;
    ਦਸ਼ਮਲਵ              eps;
};

ਪੂਰਨ ਇੰਦਰਾਜ_ਬਿੰਦੂ(ਬਣਤਰ ggml_et_ਸਮੂਹ_norm_params * params, ਖਾਲੀ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ਜੇ (!kernel_env) {
        ਵਾਪਸ -1;
    }

    ਪੂਰਨ thread_id   = get_relative_thread_id(kernel_env->shire_ਨਕਾਬ);
    ਪੂਰਨ ਅੰਕ_ਧਾਗੇ = get_ਅੰਕ_ਧਾਗੇ(kernel_env->shire_ਨਕਾਬ);

    ਜੇ (thread_id < 0) {
        ਵਾਪਸ 0;
    }

    ਜੇ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ਵਾਪਸ -1;
    }

    ਬਣਤਰ ggml_tensor * src0 = &params->src0;
    ਬਣਤਰ ggml_tensor * dst  = &params->dst;

    ਜੇ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ਵਾਪਸ -1;
    }

    ਅਚਲ ਦਸ਼ਮਲਵ * src0_ਅੰਕੜੇ = (ਅਚਲ ਦਸ਼ਮਲਵ *) src0->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ *       dst_ਅੰਕੜੇ  = (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ;

    ਜੇ (!src0_ਅੰਕੜੇ || !dst_ਅੰਕੜੇ) {
        ਵਾਪਸ -1;
    }

    ਅਚਲ int32_t n_groups = params->n_groups;
    ਅਚਲ ਦਸ਼ਮਲਵ   eps      = params->eps;

    ਜੇ (n_groups <= 0 || eps < 0.0f) {
        ਵਾਪਸ -1;
    }

    ਅਚਲ int64_t ne0 = dst->ne[0];
    ਅਚਲ int64_t ne1 = dst->ne[1];
    ਅਚਲ int64_t ne2 = dst->ne[2];
    ਅਚਲ int64_t ne3 = dst->ne[3];

    ਜੇ (src0->ne[0] != ne0 || src0->ne[1] != ne1 || src0->ne[2] != ne2 || src0->ne[3] != ne3) {
        ਵਾਪਸ -1;
    }

    ਅਚਲ int64_t nb1  = dst->nb[1];
    ਅਚਲ int64_t nb2  = dst->nb[2];
    ਅਚਲ int64_t nb3  = dst->nb[3];
    ਅਚਲ int64_t nb01 = src0->nb[1];
    ਅਚਲ int64_t nb02 = src0->nb[2];
    ਅਚਲ int64_t nb03 = src0->nb[3];

    ਅਚਲ int64_t channels_ਪ੍ਰਤੀ_ਸਮੂਹ = (ne2 + n_groups - 1) / n_groups;
    ਜੇ (channels_ਪ੍ਰਤੀ_ਸਮੂਹ <= 0) {
        ਵਾਪਸ -1;
    }

    ਅਚਲ int64_t ਸਰਗਰਮ_groups        = (ne2 + channels_ਪ੍ਰਤੀ_ਸਮੂਹ - 1) / channels_ਪ੍ਰਤੀ_ਸਮੂਹ;
    ਅਚਲ int64_t ਕੁੱਲ_work           = ਸਰਗਰਮ_groups * ne3;
    ਅਚਲ int64_t ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_write_ਸਮੂਹ = et_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_cacheline_ਸਮੂਹ(ne0, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਲਈ (int64_t work = thread_id; work < ਕੁੱਲ_work; work += ਅੰਕ_ਧਾਗੇ) {
        ਅਚਲ int64_t i3        = work / ਸਰਗਰਮ_groups;
        ਅਚਲ int64_t ਸਮੂਹ_idx = work % ਸਰਗਰਮ_groups;

        ਅਚਲ int64_t channel_ਸ਼ੁਰੂ = ਸਮੂਹ_idx * channels_ਪ੍ਰਤੀ_ਸਮੂਹ;
        int64_t       channel_ਸਿਰਾ   = channel_ਸ਼ੁਰੂ + channels_ਪ੍ਰਤੀ_ਸਮੂਹ;
        ਜੇ (channel_ਸਿਰਾ > ne2) {
            channel_ਸਿਰਾ = ne2;
        }

        ਅਚਲ int64_t channel_ਗਿਣਤੀ = channel_ਸਿਰਾ - channel_ਸ਼ੁਰੂ;
        ਜੇ (channel_ਗਿਣਤੀ <= 0) {
            ਜਾਰੀ;
        }

        ਦਸ਼ਮਲਵ ਜੋੜ   = 0.0f;
        ਦਸ਼ਮਲਵ denom = 0.0f;
        ਲਈ (int64_t i2 = channel_ਸ਼ੁਰੂ; i2 < channel_ਸਿਰਾ; ++i2) {
            ਲਈ (int64_t i1 = 0; i1 < ne1; ++i1) {
                ਅਚਲ ਦਸ਼ਮਲਵ * src_ਪੰਕਤੀ = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) src0_ਅੰਕੜੇ + i3 * nb03 + i2 * nb02 + i1 * nb01);
                ਲਈ (int64_t i0 = 0; i0 < ne0; ++i0) {
                    ਜੋੜ += src_ਪੰਕਤੀ[i0];
                    denom += 1.0f;
                }
            }
        }

        ਅਚਲ ਦਸ਼ਮਲਵ ਔਸਤ_2 = et_fdiv(ਜੋੜ, denom);

        ਦਸ਼ਮਲਵ var_ਜੋੜ = 0.0f;
        ਲਈ (int64_t i2 = channel_ਸ਼ੁਰੂ; i2 < channel_ਸਿਰਾ; ++i2) {
            ਲਈ (int64_t i1 = 0; i1 < ne1; ++i1) {
                ਅਚਲ ਦਸ਼ਮਲਵ * src_ਪੰਕਤੀ = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) src0_ਅੰਕੜੇ + i3 * nb03 + i2 * nb02 + i1 * nb01);
                ਲਈ (int64_t i0 = 0; i0 < ne0; ++i0) {
                    ਅਚਲ ਦਸ਼ਮਲਵ centered = src_ਪੰਕਤੀ[i0] - ਔਸਤ_2;
                    var_ਜੋੜ += centered * centered;
                }
            }
        }

        ਅਚਲ ਦਸ਼ਮਲਵ variance = et_fdiv(var_ਜੋੜ, denom);
        ਅਚਲ ਦਸ਼ਮਲਵ scale    = et_fdiv(1.0f, et_sqrtf(variance + eps));

        ਜੇ (ne0 % 16 == 0) {
            ਲਈ (int64_t i2 = channel_ਸ਼ੁਰੂ; i2 < channel_ਸਿਰਾ; ++i2) {
                ਲਈ (int64_t i1 = 0; i1 < ne1; ++i1) {
                    ਅਚਲ ਦਸ਼ਮਲਵ * src_ਪੰਕਤੀ =
                        (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) src0_ਅੰਕੜੇ + i3 * nb03 + i2 * nb02 + i1 * nb01);
                    ਦਸ਼ਮਲਵ * dst_ਪੰਕਤੀ = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst_ਅੰਕੜੇ + i3 * nb3 + i2 * nb2 + i1 * nb1);
                    ਲਈ (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_ਪੰਕਤੀ[i0] = (src_ਪੰਕਤੀ[i0] - ਔਸਤ_2) * scale;
                    }
                }
            }
        } ਨਹੀਂਤਾਂ {
            ਅਚਲ int64_t ਕੁੱਲ_ਪੰਕਤੀਆਂ_ਵਿੱਚ_ਸਮੂਹ = channel_ਗਿਣਤੀ * ne1;
            ਅਚਲ int64_t ਕੁੱਲ_write_groups  = (ਕੁੱਲ_ਪੰਕਤੀਆਂ_ਵਿੱਚ_ਸਮੂਹ + ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_write_ਸਮੂਹ - 1) / ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_write_ਸਮੂਹ;

            ਲਈ (int64_t write_ਸਮੂਹ = 0; write_ਸਮੂਹ < ਕੁੱਲ_write_groups; ++write_ਸਮੂਹ) {
                ਅਚਲ int64_t ਪੰਕਤੀ_ਸ਼ੁਰੂ = write_ਸਮੂਹ * ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_write_ਸਮੂਹ;
                int64_t       ਪੰਕਤੀ_ਸਿਰਾ   = ਪੰਕਤੀ_ਸ਼ੁਰੂ + ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_write_ਸਮੂਹ;
                ਜੇ (ਪੰਕਤੀ_ਸਿਰਾ > ਕੁੱਲ_ਪੰਕਤੀਆਂ_ਵਿੱਚ_ਸਮੂਹ) {
                    ਪੰਕਤੀ_ਸਿਰਾ = ਕੁੱਲ_ਪੰਕਤੀਆਂ_ਵਿੱਚ_ਸਮੂਹ;
                }

                ਲਈ (int64_t ਪੰਕਤੀ = ਪੰਕਤੀ_ਸ਼ੁਰੂ; ਪੰਕਤੀ < ਪੰਕਤੀ_ਸਿਰਾ; ++ਪੰਕਤੀ) {
                    ਅਚਲ int64_t local_i2 = ਪੰਕਤੀ / ne1;
                    ਅਚਲ int64_t i1       = ਪੰਕਤੀ % ne1;
                    ਅਚਲ int64_t i2       = channel_ਸ਼ੁਰੂ + local_i2;

                    ਅਚਲ ਦਸ਼ਮਲਵ * src_ਪੰਕਤੀ =
                        (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) src0_ਅੰਕੜੇ + i3 * nb03 + i2 * nb02 + i1 * nb01);
                    ਦਸ਼ਮਲਵ * dst_ਪੰਕਤੀ = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst_ਅੰਕੜੇ + i3 * nb3 + i2 * nb2 + i1 * nb1);
                    ਲਈ (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_ਪੰਕਤੀ[i0] = (src_ਪੰਕਤੀ[i0] - ਔਸਤ_2) * scale;
                    }
                }
            }
        }
    }

    ਵਾਪਸ 0;
}
