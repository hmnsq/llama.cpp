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

ساكن مضمن فراغ im2col_store_elem(فراغ * dst_بنسټ, تعداد ggml_type dst_type, int64_t idx, عائم ارزښت) {
    إذا (dst_type == GGML_TYPE_F32) {
        ((عائم *) dst_بنسټ)[idx] = ارزښت;
    } وإلا {
        ((uint16_t *) dst_بنسټ)[idx] = fp32_منزل_fp16(ارزښت);
    }
}

ساكن مضمن عائم im2col_بارول_src_elem(ثابت فراغ * src_بنسټ, تعداد ggml_type src_type, int64_t idx) {
    إذا (src_type == GGML_TYPE_F32) {
        أرجع ((ثابت عائم *) src_بنسټ)[idx];
    }

    أرجع fp16_منزل_fp32(((ثابت uint16_t *) src_بنسټ)[idx]);
}

صحيح ثبت_ټکی(بنية ggml_et_دوګونی_params * params, فراغ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    إذا (!kernel_env || params == 0 || ((uint64_t) params & 0x7) != 0) {
        أرجع -1;
    }

    صحيح thread_id   = get_relative_thread_id(kernel_env->shire_ماسک);
    صحيح عدد_تارونه = get_عدد_تارونه(kernel_env->shire_ماسک);

    إذا (thread_id < 0) {
        أرجع 0;
    }

    بنية ggml_tensor * src0 = &params->src0;
    بنية ggml_tensor * src1 = &params->src1;
    بنية ggml_tensor * dst  = &params->dst;

    إذا (!src1->معلومات || !dst->معلومات) {
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
    ثابت int64_t لیکه_elems  = dst->ne[0];
    ثابت int64_t ټول_لیکې = OW * OH * N;

    ثابت size_t src_batch_stride   = is_2d ? src1->nb[3] : src1->nb[2];
    ثابت size_t src_channel_stride = is_2d ? src1->nb[2] : src1->nb[1];

    ثابت size_t dst_لیکه_stride   = dst->nb[1];
    ثابت size_t dst_plane_stride = is_2d ? dst->nb[2] : 0;
    ثابت size_t dst_batch_stride = is_2d ? dst->nb[3] : dst->nb[2];

    ثابت int64_t dst_elem_اندازه  = (dst->type == GGML_TYPE_F32) ? (int64_t) حجم(عائم) : (int64_t) حجم(uint16_t);
    ثابت int64_t لیکې_فی_ډله = et_لیکې_فی_cacheline_ډله(لیکه_elems, dst_elem_اندازه);
    ثابت int64_t ټول_groups   = (ټول_لیکې + لیکې_فی_ډله - 1) / لیکې_فی_ډله;

    لكل (int64_t grp = thread_id; grp < ټول_groups; grp += عدد_تارونه) {
        ثابت int64_t لیکه_پیل = grp * لیکې_فی_ډله;
        int64_t       لیکه_پای_ټکی   = لیکه_پیل + لیکې_فی_ډله;
        إذا (لیکه_پای_ټکی > ټول_لیکې) {
            لیکه_پای_ټکی = ټول_لیکې;
        }

        لكل (int64_t لیکه = لیکه_پیل; لیکه < لیکه_پای_ټکی; ++لیکه) {
            ثابت int64_t دننه  = لیکه / (OH * OW);
            ثابت int64_t rem = لیکه % (OH * OW);
            ثابت int64_t ioh = rem / OW;
            ثابت int64_t iow = rem % OW;

            فراغ * dst_لیکه = (حرف *) dst->معلومات + دننه * dst_batch_stride + ioh * dst_plane_stride + iow * dst_لیکه_stride;

            لكل (int64_t iic = 0; iic < IC; ++iic) {
                ثابت فراغ * src_channel = (ثابت حرف *) src1->معلومات + دننه * src_batch_stride + iic * src_channel_stride;

                لكل (int64_t ikh = 0; ikh < KH; ++ikh) {
                    لكل (int64_t ikw = 0; ikw < KW; ++ikw) {
                        ثابت int64_t iiw     = iow * s0 + ikw * d0 - p0;
                        ثابت int64_t iih     = ioh * s1 + ikh * d1 - p1;
                        ثابت int64_t dst_idx = iic * (KH * KW) + ikh * KW + ikw;

                        إذا (iiw < 0 || iiw >= IW || iih < 0 || iih >= IH) {
                            im2col_store_elem(dst_لیکه, dst->type, dst_idx, 0.0f);
                        } وإلا {
                            ثابت int64_t src_idx = iih * IW + iiw;
                            ثابت عائم   ارزښت   = im2col_بارول_src_elem(src_channel, src1->type, src_idx);
                            im2col_store_elem(dst_لیکه, dst->type, dst_idx, ارزښت);
                        }
                    }
                }
            }
        }
    }

    أرجع 0;
}
