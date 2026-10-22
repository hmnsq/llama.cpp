#ifndef HTP_REPACK_H
#define HTP_REPACK_H

#شامل <stddef.h>
#شامل <stdint.h>

#شامل "ggml.h"
#شامل "htp/htp-tensor.h"

ایستا درونخطی bool htp_is_tiled_weight_type(شمارش ggml_type type) {
    بازگشت type == GGML_TYPE_Q4_0  || type == GGML_TYPE_Q4_1   ||
           type == GGML_TYPE_Q8_0  || type == GGML_TYPE_IQ4_NL ||
           type == GGML_TYPE_MXFP4 || type == GGML_TYPE_Q1_0   ||
           type == GGML_TYPE_Q6_K   ||
           type == GGML_TYPE_Q4_K  || type == GGML_TYPE_Q5_K   ||
           type == GGML_TYPE_Q3_K  || type == GGML_TYPE_Q2_K   ||
           type == GGML_TYPE_Q2_0  ||
           type == GGML_TYPE_F16   || type == GGML_TYPE_F32    ||
           type == GGML_TYPE_BF16;
}

ایستا درونخطی bool htp_is_tiled_weight_tensor(ثابت ساختار ggml_tensor * t) {
    اگر (!ggml_is_contiguous(t) || ggml_is_view(t)) {
        بازگشت false;
    }
    اگر (!htp_is_tiled_weight_type(t->type)) {
        بازگشت false;
    }
    اگر (t->ne[0] < 32) {
        بازگشت false;
    }
    اگر (!ggml_is_quantized(t->type)) {
        بازگشت (t->ne[0] % 32 == 0) && (t->ne[1] % 32 == 0) && (t->ne[2] == 1) && (t->ne[3] == 1);
    }
    بازگشت t->ne[3] == 1;
}

// Size of one tiled row in the DSP tiled layout. The Q3_K..Q6_K tiles store uncompressed scales/mins,
// so they are larger than the ggml blocks. For the other tiled types (Q2_K included) the tile has the same size.
ایستا درونخطی size_t htp_tiled_row_size(شمارش ggml_type type, int64_t ne0) {
    اگر (type == GGML_TYPE_F16 || type == GGML_TYPE_F32 || type == GGML_TYPE_BF16) {
        بازگشت (size_t) ne0 * اندازه(ggml_fp16_t);
    }
    اگر (type == GGML_TYPE_Q6_K) {
        بازگشت (size_t) (ne0 / 32) * (HTP_TENSOR_TILE_SIZE_Q6_K / 32);
    }
    اگر (type == GGML_TYPE_Q2_0) {
        بازگشت (size_t) (ne0 / 64) * 18;
    }
    اگر (type == GGML_TYPE_Q4_K) {
        بازگشت (size_t) (ne0 / 32) * (HTP_TENSOR_TILE_SIZE_Q4_K / 32);
    }
    اگر (type == GGML_TYPE_Q5_K) {
        بازگشت (size_t) (ne0 / 32) * (HTP_TENSOR_TILE_SIZE_Q5_K / 32);
    }
    اگر (type == GGML_TYPE_Q3_K) {
        بازگشت (size_t) (ne0 / 32) * (HTP_TENSOR_TILE_SIZE_Q3_K / 32);
    }
    بازگشت ggml_row_size(type, ne0);
}

ایستا درونخطی bool htp_is_hmx_weight_type(شمارش ggml_type type) {
    بازگشت htp_is_tiled_weight_type(type);
}

تهی htp_repack_tensor_tiled(ساختار ggml_tensor * t, ثابت تهی * data, size_t size);
تهی htp_unpack_tensor_tiled(تهی * data, ثابت ساختار ggml_tensor * t, size_t offset, size_t size);

#endif // HTP_REPACK_H
