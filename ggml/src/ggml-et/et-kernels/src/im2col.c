//******************************************************************************
// IM2COL Kernel
// Baseline scalar implementation for:
//   src1: [N, IC, IH, IW] -> dst: [N, OH, OW, IC*KH*KW] (2D)
//   src1: [N, IC,    IW] -> dst: [N,  1, OW, IC*   KW] (1D)
//
// Work is distributed by row-groups so threads own cache-line-aligned chunks of
// destination rows even when ne0 is not cache aligned.
//******************************************************************************

#شامل "ggml_tensor.h"
#شامل "math_fp.h"
#شامل "platform.h"

#شامل <stdint.h>

ساکن اندرونی خالی im2col_store_elem(خالی * dst_بنیاد, شمار ggml_type dst_type, int64_t idx, اعشاری قدر) {
    اگر (dst_type == GGML_TYPE_F32) {
        ((اعشاری *) dst_بنیاد)[idx] = قدر;
    } ورنہ {
        ((uint16_t *) dst_بنیاد)[idx] = fp32_مقصد_fp16(قدر);
    }
}

ساکن اندرونی اعشاری im2col_بھرو_src_elem(ثابت خالی * src_بنیاد, شمار ggml_type src_type, int64_t idx) {
    اگر (src_type == GGML_TYPE_F32) {
        واپس ((ثابت اعشاری *) src_بنیاد)[idx];
    }

    واپس fp16_مقصد_fp32(((ثابت uint16_t *) src_بنیاد)[idx]);
}

صحیح اندراج_نقطہ(ساخت ggml_et_ثنائی_params * params, خالی * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    اگر (!kernel_env || params == 0 || ((uint64_t) params & 0x7) != 0) {
        واپس -1;
    }

    صحیح thread_id   = get_relative_thread_id(kernel_env->shire_نقاب);
    صحیح عدد_دھاگے = get_عدد_دھاگے(kernel_env->shire_نقاب);

    اگر (thread_id < 0) {
        واپس 0;
    }

    ساخت ggml_tensor * src0 = &params->src0;
    ساخت ggml_tensor * src1 = &params->src1;
    ساخت ggml_tensor * dst  = &params->dst;

    اگر (!src1->ڈیٹا || !dst->ڈیٹا) {
        واپس -1;
    }

    اگر (!((dst->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) ||
          (dst->type == GGML_TYPE_F16 && (src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32)))) {
        واپس -1;
    }

    ثابت int32_t s0    = ((ثابت int32_t *) dst->op_params)[0];
    ثابت int32_t s1    = ((ثابت int32_t *) dst->op_params)[1];
    ثابت int32_t p0    = ((ثابت int32_t *) dst->op_params)[2];
    ثابت int32_t p1    = ((ثابت int32_t *) dst->op_params)[3];
    ثابت int32_t d0    = ((ثابت int32_t *) dst->op_params)[4];
    ثابت int32_t d1    = ((ثابت int32_t *) dst->op_params)[5];
    ثابت int32_t is_2d = ((ثابت int32_t *) dst->op_params)[6];

    ثابت int64_t N  = is_2d ? src1->ne[3] : src1->ne[2];
    ثابت int64_t IC = is_2d ? src1->ne[2] : src1->ne[1];
    ثابت int64_t IH = is_2d ? src1->ne[1] : 1;
    ثابت int64_t IW = src1->ne[0];

    ثابت int64_t KH = is_2d ? src0->ne[1] : 1;
    ثابت int64_t KW = src0->ne[0];

    ثابت int64_t OH         = is_2d ? dst->ne[2] : 1;
    ثابت int64_t OW         = dst->ne[1];
    ثابت int64_t صف_elems  = dst->ne[0];
    ثابت int64_t کل_صفیں = OW * OH * N;

    ثابت size_t src_batch_stride   = is_2d ? src1->nb[3] : src1->nb[2];
    ثابت size_t src_channel_stride = is_2d ? src1->nb[2] : src1->nb[1];

    ثابت size_t dst_صف_stride   = dst->nb[1];
    ثابت size_t dst_plane_stride = is_2d ? dst->nb[2] : 0;
    ثابت size_t dst_batch_stride = is_2d ? dst->nb[3] : dst->nb[2];

    ثابت int64_t dst_elem_سائز  = (dst->type == GGML_TYPE_F32) ? (int64_t) حجم(اعشاری) : (int64_t) حجم(uint16_t);
    ثابت int64_t صفیں_فی_گروہ = et_صفیں_فی_cacheline_گروہ(صف_elems, dst_elem_سائز);
    ثابت int64_t کل_groups   = (کل_صفیں + صفیں_فی_گروہ - 1) / صفیں_فی_گروہ;

    برائے (int64_t grp = thread_id; grp < کل_groups; grp += عدد_دھاگے) {
        ثابت int64_t صف_آغاز = grp * صفیں_فی_گروہ;
        int64_t       صف_انتہا   = صف_آغاز + صفیں_فی_گروہ;
        اگر (صف_انتہا > کل_صفیں) {
            صف_انتہا = کل_صفیں;
        }

        برائے (int64_t صف = صف_آغاز; صف < صف_انتہا; ++صف) {
            ثابت int64_t اندر  = صف / (OH * OW);
            ثابت int64_t rem = صف % (OH * OW);
            ثابت int64_t ioh = rem / OW;
            ثابت int64_t iow = rem % OW;

            خالی * dst_صف = (حرف *) dst->ڈیٹا + اندر * dst_batch_stride + ioh * dst_plane_stride + iow * dst_صف_stride;

            برائے (int64_t iic = 0; iic < IC; ++iic) {
                ثابت خالی * src_channel = (ثابت حرف *) src1->ڈیٹا + اندر * src_batch_stride + iic * src_channel_stride;

                برائے (int64_t ikh = 0; ikh < KH; ++ikh) {
                    برائے (int64_t ikw = 0; ikw < KW; ++ikw) {
                        ثابت int64_t iiw     = iow * s0 + ikw * d0 - p0;
                        ثابت int64_t iih     = ioh * s1 + ikh * d1 - p1;
                        ثابت int64_t dst_idx = iic * (KH * KW) + ikh * KW + ikw;

                        اگر (iiw < 0 || iiw >= IW || iih < 0 || iih >= IH) {
                            im2col_store_elem(dst_صف, dst->type, dst_idx, 0.0f);
                        } ورنہ {
                            ثابت int64_t src_idx = iih * IW + iiw;
                            ثابت اعشاری   قدر   = im2col_بھرو_src_elem(src_channel, src1->type, src_idx);
                            im2col_store_elem(dst_صف, dst->type, dst_idx, قدر);
                        }
                    }
                }
            }
        }
    }

    واپس 0;
}
