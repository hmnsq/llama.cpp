//******************************************************************************
// IM2COL Kernel
// Baseline scalar implementation for:
//   src1: [N, IC, IH, IW] -> dst: [N, OH, OW, IC*KH*KW] (2D)
//   src1: [N, IC,    IW] -> dst: [N,  1, OW, IC*   KW] (1D)
//
// Work is distributed by row-groups so threads own cache-line-aligned chunks of
// destination rows even when ne0 is not cache aligned.
//******************************************************************************

#ضمن "ggml_tensor.h"
#ضمن "math_fp.h"
#ضمن "platform.h"

#ضمن <stdint.h>

ساكن مضمن فراغ im2col_store_elem(فراغ * dst_ادھار, تعداد ggml_type dst_type, int64_t idx, عائم قدر) {
    إذا (dst_type == GGML_TYPE_F32) {
        ((عائم *) dst_ادھار)[idx] = قدر;
    } وإلا {
        ((uint16_t *) dst_ادھار)[idx] = fp32_منزل_fp16(قدر);
    }
}

ساكن مضمن عائم im2col_لدو_src_elem(ثابت فراغ * src_ادھار, تعداد ggml_type src_type, int64_t idx) {
    إذا (src_type == GGML_TYPE_F32) {
        أرجع ((ثابت عائم *) src_ادھار)[idx];
    }

    أرجع fp16_منزل_fp32(((ثابت uint16_t *) src_ادھار)[idx]);
}

صحيح اندراج_نقطہ(بنية ggml_et_ثنائی_params * params, فراغ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    إذا (!kernel_env || params == 0 || ((uint64_t) params & 0x7) != 0) {
        أرجع -1;
    }

    صحيح thread_id   = get_relative_thread_id(kernel_env->shire_نقاب);
    صحيح عدد_دھاگے = get_عدد_دھاگے(kernel_env->shire_نقاب);

    إذا (thread_id < 0) {
        أرجع 0;
    }

    بنية ggml_tensor * src0 = &params->src0;
    بنية ggml_tensor * src1 = &params->src1;
    بنية ggml_tensor * dst  = &params->dst;

    إذا (!src1->انکڑے || !dst->انکڑے) {
        أرجع -1;
    }

    إذا (!((dst->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) ||
          (dst->type == GGML_TYPE_F16 && (src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32)))) {
        أرجع -1;
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
    ثابت int64_t کل_صفاں = OW * OH * N;

    ثابت size_t src_batch_stride   = is_2d ? src1->nb[3] : src1->nb[2];
    ثابت size_t src_channel_stride = is_2d ? src1->nb[2] : src1->nb[1];

    ثابت size_t dst_صف_stride   = dst->nb[1];
    ثابت size_t dst_plane_stride = is_2d ? dst->nb[2] : 0;
    ثابت size_t dst_batch_stride = is_2d ? dst->nb[3] : dst->nb[2];

    ثابت int64_t dst_elem_ناپ  = (dst->type == GGML_TYPE_F32) ? (int64_t) حجم(عائم) : (int64_t) حجم(uint16_t);
    ثابت int64_t صفاں_فی_گروہ = et_صفاں_فی_cacheline_گروہ(صف_elems, dst_elem_ناپ);
    ثابت int64_t کل_groups   = (کل_صفاں + صفاں_فی_گروہ - 1) / صفاں_فی_گروہ;

    لكل (int64_t grp = thread_id; grp < کل_groups; grp += عدد_دھاگے) {
        ثابت int64_t صف_شروع = grp * صفاں_فی_گروہ;
        int64_t       صف_سرا   = صف_شروع + صفاں_فی_گروہ;
        إذا (صف_سرا > کل_صفاں) {
            صف_سرا = کل_صفاں;
        }

        لكل (int64_t صف = صف_شروع; صف < صف_سرا; ++صف) {
            ثابت int64_t وچ  = صف / (OH * OW);
            ثابت int64_t rem = صف % (OH * OW);
            ثابت int64_t ioh = rem / OW;
            ثابت int64_t iow = rem % OW;

            فراغ * dst_صف = (حرف *) dst->انکڑے + وچ * dst_batch_stride + ioh * dst_plane_stride + iow * dst_صف_stride;

            لكل (int64_t iic = 0; iic < IC; ++iic) {
                ثابت فراغ * src_channel = (ثابت حرف *) src1->انکڑے + وچ * src_batch_stride + iic * src_channel_stride;

                لكل (int64_t ikh = 0; ikh < KH; ++ikh) {
                    لكل (int64_t ikw = 0; ikw < KW; ++ikw) {
                        ثابت int64_t iiw     = iow * s0 + ikw * d0 - p0;
                        ثابت int64_t iih     = ioh * s1 + ikh * d1 - p1;
                        ثابت int64_t dst_idx = iic * (KH * KW) + ikh * KW + ikw;

                        إذا (iiw < 0 || iiw >= IW || iih < 0 || iih >= IH) {
                            im2col_store_elem(dst_صف, dst->type, dst_idx, 0.0f);
                        } وإلا {
                            ثابت int64_t src_idx = iih * IW + iiw;
                            ثابت عائم   قدر   = im2col_لدو_src_elem(src_channel, src1->type, src_idx);
                            im2col_store_elem(dst_صف, dst->type, dst_idx, قدر);
                        }
                    }
                }
            }
        }
    }

    أرجع 0;
}
