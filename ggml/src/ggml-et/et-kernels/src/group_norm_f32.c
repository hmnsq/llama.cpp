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

#ಅನ್ತರ್ಭಾವ "ggml_tensor.h"
#ಅನ್ತರ್ಭಾವ "math_fp.h"
#ಅನ್ತರ್ಭಾವ "platform.h"

#ಅನ್ತರ್ಭಾವ <stdint.h>

ರಚನೆ ggml_et_ಗುಂಪು_norm_params {
    ರಚನೆ ggml_tensor src0;
    ರಚನೆ ggml_tensor dst;
    int32_t            n_groups;
    ದಶಮಾಂಶ              eps;
};

ಪೂರ್ಣಾಂಕ ನಮೂದು_ಬಿಂದು(ರಚನೆ ggml_et_ಗುಂಪು_norm_params * params, ಶೂನ್ಯ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ಆದರೆ (!kernel_env) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಪೂರ್ಣಾಂಕ thread_id   = get_relative_thread_id(kernel_env->shire_ಮುಖವಾಡ);
    ಪೂರ್ಣಾಂಕ ಅಂಕೆ_ಎಳೆಗಳು = get_ಅಂಕೆ_ಎಳೆಗಳು(kernel_env->shire_ಮುಖವಾಡ);

    ಆದರೆ (thread_id < 0) {
        ಹಿಂತಿರುಗಿ 0;
    }

    ಆದರೆ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ರಚನೆ ggml_tensor * src0 = &params->src0;
    ರಚನೆ ggml_tensor * dst  = &params->dst;

    ಆದರೆ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src0_ದತ್ತಾಂಶ = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src0->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ *       dst_ದತ್ತಾಂಶ  = (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ;

    ಆದರೆ (!src0_ದತ್ತಾಂಶ || !dst_ದತ್ತಾಂಶ) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಸ್ಥಿರಾಂಕ int32_t n_groups = params->n_groups;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ   eps      = params->eps;

    ಆದರೆ (n_groups <= 0 || eps < 0.0f) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಸ್ಥಿರಾಂಕ int64_t ne0 = dst->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t ne1 = dst->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t ne2 = dst->ne[2];
    ಸ್ಥಿರಾಂಕ int64_t ne3 = dst->ne[3];

    ಆದರೆ (src0->ne[0] != ne0 || src0->ne[1] != ne1 || src0->ne[2] != ne2 || src0->ne[3] != ne3) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಸ್ಥಿರಾಂಕ int64_t nb1  = dst->nb[1];
    ಸ್ಥಿರಾಂಕ int64_t nb2  = dst->nb[2];
    ಸ್ಥಿರಾಂಕ int64_t nb3  = dst->nb[3];
    ಸ್ಥಿರಾಂಕ int64_t nb01 = src0->nb[1];
    ಸ್ಥಿರಾಂಕ int64_t nb02 = src0->nb[2];
    ಸ್ಥಿರಾಂಕ int64_t nb03 = src0->nb[3];

    ಸ್ಥಿರಾಂಕ int64_t channels_ಪ್ರತಿ_ಗುಂಪು = (ne2 + n_groups - 1) / n_groups;
    ಆದರೆ (channels_ಪ್ರತಿ_ಗುಂಪು <= 0) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಸ್ಥಿರಾಂಕ int64_t ಸಕ್ರಿಯ_groups        = (ne2 + channels_ಪ್ರತಿ_ಗುಂಪು - 1) / channels_ಪ್ರತಿ_ಗುಂಪು;
    ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_work           = ಸಕ್ರಿಯ_groups * ne3;
    ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_write_ಗುಂಪು = et_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_cacheline_ಗುಂಪು(ne0, ಗಾತ್ರ(ದಶಮಾಂಶ));

    ಗಾಗಿ (int64_t work = thread_id; work < ಒಟ್ಟು_work; work += ಅಂಕೆ_ಎಳೆಗಳು) {
        ಸ್ಥಿರಾಂಕ int64_t i3        = work / ಸಕ್ರಿಯ_groups;
        ಸ್ಥಿರಾಂಕ int64_t ಗುಂಪು_idx = work % ಸಕ್ರಿಯ_groups;

        ಸ್ಥಿರಾಂಕ int64_t channel_ಆರಂಭ = ಗುಂಪು_idx * channels_ಪ್ರತಿ_ಗುಂಪು;
        int64_t       channel_ಕೊನೆ   = channel_ಆರಂಭ + channels_ಪ್ರತಿ_ಗುಂಪು;
        ಆದರೆ (channel_ಕೊನೆ > ne2) {
            channel_ಕೊನೆ = ne2;
        }

        ಸ್ಥಿರಾಂಕ int64_t channel_ಎಣಿಕೆ = channel_ಕೊನೆ - channel_ಆರಂಭ;
        ಆದರೆ (channel_ಎಣಿಕೆ <= 0) {
            ಮುಂದುವರಿ;
        }

        ದಶಮಾಂಶ ಮೊತ್ತ   = 0.0f;
        ದಶಮಾಂಶ denom = 0.0f;
        ಗಾಗಿ (int64_t i2 = channel_ಆರಂಭ; i2 < channel_ಕೊನೆ; ++i2) {
            ಗಾಗಿ (int64_t i1 = 0; i1 < ne1; ++i1) {
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src_ಅಡ್ಡಸಾಲು = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src0_ದತ್ತಾಂಶ + i3 * nb03 + i2 * nb02 + i1 * nb01);
                ಗಾಗಿ (int64_t i0 = 0; i0 < ne0; ++i0) {
                    ಮೊತ್ತ += src_ಅಡ್ಡಸಾಲು[i0];
                    denom += 1.0f;
                }
            }
        }

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ಸರಾಸರಿ_2 = et_fdiv(ಮೊತ್ತ, denom);

        ದಶಮಾಂಶ var_ಮೊತ್ತ = 0.0f;
        ಗಾಗಿ (int64_t i2 = channel_ಆರಂಭ; i2 < channel_ಕೊನೆ; ++i2) {
            ಗಾಗಿ (int64_t i1 = 0; i1 < ne1; ++i1) {
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src_ಅಡ್ಡಸಾಲು = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src0_ದತ್ತಾಂಶ + i3 * nb03 + i2 * nb02 + i1 * nb01);
                ಗಾಗಿ (int64_t i0 = 0; i0 < ne0; ++i0) {
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ centered = src_ಅಡ್ಡಸಾಲು[i0] - ಸರಾಸರಿ_2;
                    var_ಮೊತ್ತ += centered * centered;
                }
            }
        }

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ variance = et_fdiv(var_ಮೊತ್ತ, denom);
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale    = et_fdiv(1.0f, et_sqrtf(variance + eps));

        ಆದರೆ (ne0 % 16 == 0) {
            ಗಾಗಿ (int64_t i2 = channel_ಆರಂಭ; i2 < channel_ಕೊನೆ; ++i2) {
                ಗಾಗಿ (int64_t i1 = 0; i1 < ne1; ++i1) {
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src_ಅಡ್ಡಸಾಲು =
                        (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src0_ದತ್ತಾಂಶ + i3 * nb03 + i2 * nb02 + i1 * nb01);
                    ದಶಮಾಂಶ * dst_ಅಡ್ಡಸಾಲು = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst_ದತ್ತಾಂಶ + i3 * nb3 + i2 * nb2 + i1 * nb1);
                    ಗಾಗಿ (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_ಅಡ್ಡಸಾಲು[i0] = (src_ಅಡ್ಡಸಾಲು[i0] - ಸರಾಸರಿ_2) * scale;
                    }
                }
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು_ಒಳ_ಗುಂಪು = channel_ಎಣಿಕೆ * ne1;
            ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_write_groups  = (ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು_ಒಳ_ಗುಂಪು + ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_write_ಗುಂಪು - 1) / ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_write_ಗುಂಪು;

            ಗಾಗಿ (int64_t write_ಗುಂಪು = 0; write_ಗುಂಪು < ಒಟ್ಟು_write_groups; ++write_ಗುಂಪು) {
                ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲು_ಆರಂಭ = write_ಗುಂಪು * ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_write_ಗುಂಪು;
                int64_t       ಅಡ್ಡಸಾಲು_ಕೊನೆ   = ಅಡ್ಡಸಾಲು_ಆರಂಭ + ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_write_ಗುಂಪು;
                ಆದರೆ (ಅಡ್ಡಸಾಲು_ಕೊನೆ > ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು_ಒಳ_ಗುಂಪು) {
                    ಅಡ್ಡಸಾಲು_ಕೊನೆ = ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು_ಒಳ_ಗುಂಪು;
                }

                ಗಾಗಿ (int64_t ಅಡ್ಡಸಾಲು = ಅಡ್ಡಸಾಲು_ಆರಂಭ; ಅಡ್ಡಸಾಲು < ಅಡ್ಡಸಾಲು_ಕೊನೆ; ++ಅಡ್ಡಸಾಲು) {
                    ಸ್ಥಿರಾಂಕ int64_t local_i2 = ಅಡ್ಡಸಾಲು / ne1;
                    ಸ್ಥಿರಾಂಕ int64_t i1       = ಅಡ್ಡಸಾಲು % ne1;
                    ಸ್ಥಿರಾಂಕ int64_t i2       = channel_ಆರಂಭ + local_i2;

                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src_ಅಡ್ಡಸಾಲು =
                        (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src0_ದತ್ತಾಂಶ + i3 * nb03 + i2 * nb02 + i1 * nb01);
                    ದಶಮಾಂಶ * dst_ಅಡ್ಡಸಾಲು = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst_ದತ್ತಾಂಶ + i3 * nb3 + i2 * nb2 + i1 * nb1);
                    ಗಾಗಿ (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_ಅಡ್ಡಸಾಲು[i0] = (src_ಅಡ್ಡಸಾಲು[i0] - ಸರಾಸರಿ_2) * scale;
                    }
                }
            }
        }
    }

    ಹಿಂತಿರುಗಿ 0;
}
