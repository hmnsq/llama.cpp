//******************************************************************************
// IM2COL Kernel
// Baseline scalar implementation for:
//   src1: [N, IC, IH, IW] -> dst: [N, OH, OW, IC*KH*KW] (2D)
//   src1: [N, IC,    IW] -> dst: [N,  1, OW, IC*   KW] (1D)
//
// Work is distributed by row-groups so threads own cache-line-aligned chunks of
// destination rows even when ne0 is not cache aligned.
//******************************************************************************

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml_tensor.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "math_fp.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "platform.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ im2col_store_elem(ਖਾਲੀ * dst_ਅਧਾਰ, ਗਿਣਤੀ ggml_type dst_type, int64_t idx, ਦਸ਼ਮਲਵ ਮੁੱਲ) {
    ਜੇ (dst_type == GGML_TYPE_F32) {
        ((ਦਸ਼ਮਲਵ *) dst_ਅਧਾਰ)[idx] = ਮੁੱਲ;
    } ਨਹੀਂਤਾਂ {
        ((uint16_t *) dst_ਅਧਾਰ)[idx] = fp32_ਮੰਜ਼ਿਲ_fp16(ਮੁੱਲ);
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ im2col_ਲੱਦੋ_src_elem(ਅਚਲ ਖਾਲੀ * src_ਅਧਾਰ, ਗਿਣਤੀ ggml_type src_type, int64_t idx) {
    ਜੇ (src_type == GGML_TYPE_F32) {
        ਵਾਪਸ ((ਅਚਲ ਦਸ਼ਮਲਵ *) src_ਅਧਾਰ)[idx];
    }

    ਵਾਪਸ fp16_ਮੰਜ਼ਿਲ_fp32(((ਅਚਲ uint16_t *) src_ਅਧਾਰ)[idx]);
}

ਪੂਰਨ ਇੰਦਰਾਜ_ਬਿੰਦੂ(ਬਣਤਰ ggml_et_ਦੁਆਧਾਰੀ_params * params, ਖਾਲੀ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ਜੇ (!kernel_env || params == 0 || ((uint64_t) params & 0x7) != 0) {
        ਵਾਪਸ -1;
    }

    ਪੂਰਨ thread_id   = get_relative_thread_id(kernel_env->shire_ਨਕਾਬ);
    ਪੂਰਨ ਅੰਕ_ਧਾਗੇ = get_ਅੰਕ_ਧਾਗੇ(kernel_env->shire_ਨਕਾਬ);

    ਜੇ (thread_id < 0) {
        ਵਾਪਸ 0;
    }

    ਬਣਤਰ ggml_tensor * src0 = &params->src0;
    ਬਣਤਰ ggml_tensor * src1 = &params->src1;
    ਬਣਤਰ ggml_tensor * dst  = &params->dst;

    ਜੇ (!src1->ਅੰਕੜੇ || !dst->ਅੰਕੜੇ) {
        ਵਾਪਸ -1;
    }

    ਜੇ (!((dst->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) ||
          (dst->type == GGML_TYPE_F16 && (src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32)))) {
        ਵਾਪਸ -1;
    }

    ਅਚਲ int32_t s0    = ((ਅਚਲ int32_t *) dst->op_params)[0];
    ਅਚਲ int32_t s1    = ((ਅਚਲ int32_t *) dst->op_params)[1];
    ਅਚਲ int32_t p0    = ((ਅਚਲ int32_t *) dst->op_params)[2];
    ਅਚਲ int32_t p1    = ((ਅਚਲ int32_t *) dst->op_params)[3];
    ਅਚਲ int32_t d0    = ((ਅਚਲ int32_t *) dst->op_params)[4];
    ਅਚਲ int32_t d1    = ((ਅਚਲ int32_t *) dst->op_params)[5];
    ਅਚਲ int32_t is_2d = ((ਅਚਲ int32_t *) dst->op_params)[6];

    ਅਚਲ int64_t N  = is_2d ? src1->ne[3] : src1->ne[2];
    ਅਚਲ int64_t IC = is_2d ? src1->ne[2] : src1->ne[1];
    ਅਚਲ int64_t IH = is_2d ? src1->ne[1] : 1;
    ਅਚਲ int64_t IW = src1->ne[0];

    ਅਚਲ int64_t KH = is_2d ? src0->ne[1] : 1;
    ਅਚਲ int64_t KW = src0->ne[0];

    ਅਚਲ int64_t OH         = is_2d ? dst->ne[2] : 1;
    ਅਚਲ int64_t OW         = dst->ne[1];
    ਅਚਲ int64_t ਪੰਕਤੀ_elems  = dst->ne[0];
    ਅਚਲ int64_t ਕੁੱਲ_ਪੰਕਤੀਆਂ = OW * OH * N;

    ਅਚਲ size_t src_batch_stride   = is_2d ? src1->nb[3] : src1->nb[2];
    ਅਚਲ size_t src_channel_stride = is_2d ? src1->nb[2] : src1->nb[1];

    ਅਚਲ size_t dst_ਪੰਕਤੀ_stride   = dst->nb[1];
    ਅਚਲ size_t dst_plane_stride = is_2d ? dst->nb[2] : 0;
    ਅਚਲ size_t dst_batch_stride = is_2d ? dst->nb[3] : dst->nb[2];

    ਅਚਲ int64_t dst_elem_ਮਾਪ  = (dst->type == GGML_TYPE_F32) ? (int64_t) ਆਕਾਰ(ਦਸ਼ਮਲਵ) : (int64_t) ਆਕਾਰ(uint16_t);
    ਅਚਲ int64_t ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_ਸਮੂਹ = et_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_cacheline_ਸਮੂਹ(ਪੰਕਤੀ_elems, dst_elem_ਮਾਪ);
    ਅਚਲ int64_t ਕੁੱਲ_groups   = (ਕੁੱਲ_ਪੰਕਤੀਆਂ + ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_ਸਮੂਹ - 1) / ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_ਸਮੂਹ;

    ਲਈ (int64_t grp = thread_id; grp < ਕੁੱਲ_groups; grp += ਅੰਕ_ਧਾਗੇ) {
        ਅਚਲ int64_t ਪੰਕਤੀ_ਸ਼ੁਰੂ = grp * ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_ਸਮੂਹ;
        int64_t       ਪੰਕਤੀ_ਸਿਰਾ   = ਪੰਕਤੀ_ਸ਼ੁਰੂ + ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_ਸਮੂਹ;
        ਜੇ (ਪੰਕਤੀ_ਸਿਰਾ > ਕੁੱਲ_ਪੰਕਤੀਆਂ) {
            ਪੰਕਤੀ_ਸਿਰਾ = ਕੁੱਲ_ਪੰਕਤੀਆਂ;
        }

        ਲਈ (int64_t ਪੰਕਤੀ = ਪੰਕਤੀ_ਸ਼ੁਰੂ; ਪੰਕਤੀ < ਪੰਕਤੀ_ਸਿਰਾ; ++ਪੰਕਤੀ) {
            ਅਚਲ int64_t ਵਿੱਚ  = ਪੰਕਤੀ / (OH * OW);
            ਅਚਲ int64_t rem = ਪੰਕਤੀ % (OH * OW);
            ਅਚਲ int64_t ioh = rem / OW;
            ਅਚਲ int64_t iow = rem % OW;

            ਖਾਲੀ * dst_ਪੰਕਤੀ = (ਅੱਖਰ *) dst->ਅੰਕੜੇ + ਵਿੱਚ * dst_batch_stride + ioh * dst_plane_stride + iow * dst_ਪੰਕਤੀ_stride;

            ਲਈ (int64_t iic = 0; iic < IC; ++iic) {
                ਅਚਲ ਖਾਲੀ * src_channel = (ਅਚਲ ਅੱਖਰ *) src1->ਅੰਕੜੇ + ਵਿੱਚ * src_batch_stride + iic * src_channel_stride;

                ਲਈ (int64_t ikh = 0; ikh < KH; ++ikh) {
                    ਲਈ (int64_t ikw = 0; ikw < KW; ++ikw) {
                        ਅਚਲ int64_t iiw     = iow * s0 + ikw * d0 - p0;
                        ਅਚਲ int64_t iih     = ioh * s1 + ikh * d1 - p1;
                        ਅਚਲ int64_t dst_idx = iic * (KH * KW) + ikh * KW + ikw;

                        ਜੇ (iiw < 0 || iiw >= IW || iih < 0 || iih >= IH) {
                            im2col_store_elem(dst_ਪੰਕਤੀ, dst->type, dst_idx, 0.0f);
                        } ਨਹੀਂਤਾਂ {
                            ਅਚਲ int64_t src_idx = iih * IW + iiw;
                            ਅਚਲ ਦਸ਼ਮਲਵ   ਮੁੱਲ   = im2col_ਲੱਦੋ_src_elem(src_channel, src1->type, src_idx);
                            im2col_store_elem(dst_ਪੰਕਤੀ, dst->type, dst_idx, ਮੁੱਲ);
                        }
                    }
                }
            }
        }
    }

    ਵਾਪਸ 0;
}
