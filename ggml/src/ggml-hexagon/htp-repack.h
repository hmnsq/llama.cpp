#ifndef HTP_REPACK_H
#define HTP_REPACK_H

#ضمن <stddef.h>
#ضمن <stdint.h>

#ضمن "ggml.h"
#ضمن "htp/htp-tensor.h"

ساكن مضمن bool htp_is_tiled_weight_type(تعداد ggml_type type) {
    החזר type == GGML_TYPE_Q4_0  || type == GGML_TYPE_Q4_1   ||
           type == GGML_TYPE_Q8_0  || type == GGML_TYPE_IQ4_NL ||
           type == GGML_TYPE_MXFP4 || type == GGML_TYPE_Q1_0   ||
           type == GGML_TYPE_Q6_K   ||
           type == GGML_TYPE_Q4_K  || type == GGML_TYPE_Q5_K   ||
           type == GGML_TYPE_Q3_K  || type == GGML_TYPE_Q2_K   ||
           type == GGML_TYPE_Q2_0  ||
           type == GGML_TYPE_F16   || type == GGML_TYPE_F32    ||
           type == GGML_TYPE_BF16;
}

ساكن مضمن bool htp_is_tiled_weight_tensor(ثابت بنية ggml_tensor * t) {
    אם (!ggml_is_contiguous(t) || ggml_is_view(t)) {
        החזר false;
    }
    אם (!htp_is_tiled_weight_type(t->type)) {
        החזר false;
    }
    אם (t->ne[0] < 32) {
        החזר false;
    }
    אם (!ggml_is_quantized(t->type)) {
        החזר (t->ne[0] % 32 == 0) && (t->ne[1] % 32 == 0) && (t->ne[2] == 1) && (t->ne[3] == 1);
    }
    החזר t->ne[3] == 1;
}

// Size of one tiled row in the DSP tiled layout. The Q3_K..Q6_K tiles store uncompressed scales/mins,
// so they are larger than the ggml blocks. For the other tiled types (Q2_K included) the tile has the same size.
ساكن مضمن size_t htp_tiled_row_size(تعداد ggml_type type, int64_t ne0) {
    אם (type == GGML_TYPE_F16 || type == GGML_TYPE_F32 || type == GGML_TYPE_BF16) {
        החזר (size_t) ne0 * حجم(ggml_fp16_t);
    }
    אם (type == GGML_TYPE_Q6_K) {
        החזר (size_t) (ne0 / 32) * (HTP_TENSOR_TILE_SIZE_Q6_K / 32);
    }
    אם (type == GGML_TYPE_Q2_0) {
        החזר (size_t) (ne0 / 64) * 18;
    }
    אם (type == GGML_TYPE_Q4_K) {
        החזר (size_t) (ne0 / 32) * (HTP_TENSOR_TILE_SIZE_Q4_K / 32);
    }
    אם (type == GGML_TYPE_Q5_K) {
        החזר (size_t) (ne0 / 32) * (HTP_TENSOR_TILE_SIZE_Q5_K / 32);
    }
    אם (type == GGML_TYPE_Q3_K) {
        החזר (size_t) (ne0 / 32) * (HTP_TENSOR_TILE_SIZE_Q3_K / 32);
    }
    החזר ggml_row_size(type, ne0);
}

ساكن مضمن bool htp_is_hmx_weight_type(تعداد ggml_type type) {
    החזר htp_is_tiled_weight_type(type);
}

فراغ htp_repack_tensor_tiled(بنية ggml_tensor * t, ثابت فراغ * data, size_t size);
فراغ htp_unpack_tensor_tiled(فراغ * data, ثابت بنية ggml_tensor * t, size_t offset, size_t size);

#endif // HTP_REPACK_H
