//******************************************************************************
// IM2COL Kernel
// Baseline scalar implementation for:
//   src1: [N, IC, IH, IW] -> dst: [N, OH, OW, IC*KH*KW] (2D)
//   src1: [N, IC,    IW] -> dst: [N,  1, OW, IC*   KW] (1D)
//
// Work is distributed by row-groups so threads own cache-line-aligned chunks of
// destination rows even when ne0 is not cache aligned.
//******************************************************************************

#അന്തര്ഭാവ "ggml_tensor.h"
#അന്തര്ഭാവ "math_fp.h"
#അന്തര്ഭാവ "platform.h"

#അന്തര്ഭാവ <stdint.h>

സ്ഥിതം വരിയിൽ ശൂന്യം im2col_store_elem(ശൂന്യം * dst_base, എണ്ണൽ ggml_type dst_type, int64_t idx, ദശാംശം മൂല്യം) {
    എങ്കിൽ (dst_type == GGML_TYPE_F32) {
        ((ദശാംശം *) dst_base)[idx] = മൂല്യം;
    } അല്ലെങ്കിൽ {
        ((uint16_t *) dst_base)[idx] = fp32_to_fp16(മൂല്യം);
    }
}

സ്ഥിതം വരിയിൽ ദശാംശം im2col_load_src_elem(സ്ഥിരം ശൂന്യം * src_base, എണ്ണൽ ggml_type src_type, int64_t idx) {
    എങ്കിൽ (src_type == GGML_TYPE_F32) {
        തിരികെ ((സ്ഥിരം ദശാംശം *) src_base)[idx];
    }

    തിരികെ fp16_to_fp32(((സ്ഥിരം uint16_t *) src_base)[idx]);
}

പൂർണ്ണസംഖ്യ entry_point(ഘടന ggml_et_ദ്വിമാന_params * params, ശൂന്യം * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    എങ്കിൽ (!kernel_env || params == 0 || ((uint64_t) params & 0x7) != 0) {
        തിരികെ -1;
    }

    പൂർണ്ണസംഖ്യ thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    പൂർണ്ണസംഖ്യ അക്കം_threads = get_അക്കം_threads(kernel_env->shire_mask);

    എങ്കിൽ (thread_id < 0) {
        തിരികെ 0;
    }

    ഘടന ggml_tensor * src0 = &params->src0;
    ഘടന ggml_tensor * src1 = &params->src1;
    ഘടന ggml_tensor * dst  = &params->dst;

    എങ്കിൽ (!src1->data || !dst->data) {
        തിരികെ -1;
    }

    എങ്കിൽ (!((dst->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) ||
          (dst->type == GGML_TYPE_F16 && (src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32)))) {
        തിരികെ -1;
    }

    സ്ഥിരം int32_t s0    = ((സ്ഥിരം int32_t *) dst->op_params)[0];
    സ്ഥിരം int32_t s1    = ((സ്ഥിരം int32_t *) dst->op_params)[1];
    സ്ഥിരം int32_t p0    = ((സ്ഥിരം int32_t *) dst->op_params)[2];
    സ്ഥിരം int32_t p1    = ((സ്ഥിരം int32_t *) dst->op_params)[3];
    സ്ഥിരം int32_t d0    = ((സ്ഥിരം int32_t *) dst->op_params)[4];
    സ്ഥിരം int32_t d1    = ((സ്ഥിരം int32_t *) dst->op_params)[5];
    സ്ഥിരം int32_t is_2d = ((സ്ഥിരം int32_t *) dst->op_params)[6];

    സ്ഥിരം int64_t N  = is_2d ? src1->ne[3] : src1->ne[2];
    സ്ഥിരം int64_t IC = is_2d ? src1->ne[2] : src1->ne[1];
    സ്ഥിരം int64_t IH = is_2d ? src1->ne[1] : 1;
    സ്ഥിരം int64_t IW = src1->ne[0];

    സ്ഥിരം int64_t KH = is_2d ? src0->ne[1] : 1;
    സ്ഥിരം int64_t KW = src0->ne[0];

    സ്ഥിരം int64_t OH         = is_2d ? dst->ne[2] : 1;
    സ്ഥിരം int64_t OW         = dst->ne[1];
    സ്ഥിരം int64_t row_elems  = dst->ne[0];
    സ്ഥിരം int64_t total_rows = OW * OH * N;

    സ്ഥിരം size_t src_batch_stride   = is_2d ? src1->nb[3] : src1->nb[2];
    സ്ഥിരം size_t src_channel_stride = is_2d ? src1->nb[2] : src1->nb[1];

    സ്ഥിരം size_t dst_row_stride   = dst->nb[1];
    സ്ഥിരം size_t dst_plane_stride = is_2d ? dst->nb[2] : 0;
    സ്ഥിരം size_t dst_batch_stride = is_2d ? dst->nb[3] : dst->nb[2];

    സ്ഥിരം int64_t dst_elem_size  = (dst->type == GGML_TYPE_F32) ? (int64_t) വലുപ്പം(ദശാംശം) : (int64_t) വലുപ്പം(uint16_t);
    സ്ഥിരം int64_t rows_per_group = et_rows_per_cacheline_group(row_elems, dst_elem_size);
    സ്ഥിരം int64_t total_groups   = (total_rows + rows_per_group - 1) / rows_per_group;

    വേണ്ടി (int64_t grp = thread_id; grp < total_groups; grp += അക്കം_threads) {
        സ്ഥിരം int64_t row_start = grp * rows_per_group;
        int64_t       row_end   = row_start + rows_per_group;
        എങ്കിൽ (row_end > total_rows) {
            row_end = total_rows;
        }

        വേണ്ടി (int64_t row = row_start; row < row_end; ++row) {
            സ്ഥിരം int64_t in  = row / (OH * OW);
            സ്ഥിരം int64_t rem = row % (OH * OW);
            സ്ഥിരം int64_t ioh = rem / OW;
            സ്ഥിരം int64_t iow = rem % OW;

            ശൂന്യം * dst_row = (അക്ഷരം *) dst->data + in * dst_batch_stride + ioh * dst_plane_stride + iow * dst_row_stride;

            വേണ്ടി (int64_t iic = 0; iic < IC; ++iic) {
                സ്ഥിരം ശൂന്യം * src_channel = (സ്ഥിരം അക്ഷരം *) src1->data + in * src_batch_stride + iic * src_channel_stride;

                വേണ്ടി (int64_t ikh = 0; ikh < KH; ++ikh) {
                    വേണ്ടി (int64_t ikw = 0; ikw < KW; ++ikw) {
                        സ്ഥിരം int64_t iiw     = iow * s0 + ikw * d0 - p0;
                        സ്ഥിരം int64_t iih     = ioh * s1 + ikh * d1 - p1;
                        സ്ഥിരം int64_t dst_idx = iic * (KH * KW) + ikh * KW + ikw;

                        എങ്കിൽ (iiw < 0 || iiw >= IW || iih < 0 || iih >= IH) {
                            im2col_store_elem(dst_row, dst->type, dst_idx, 0.0f);
                        } അല്ലെങ്കിൽ {
                            സ്ഥിരം int64_t src_idx = iih * IW + iiw;
                            സ്ഥിരം ദശാംശം   മൂല്യം   = im2col_load_src_elem(src_channel, src1->type, src_idx);
                            im2col_store_elem(dst_row, dst->type, dst_idx, മൂല്യം);
                        }
                    }
                }
            }
        }
    }

    തിരികെ 0;
}
