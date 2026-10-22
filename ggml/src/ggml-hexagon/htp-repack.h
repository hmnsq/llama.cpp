#اگر_غیر_تعریف_شدہ HTP_REPACK_H
#تعریف_کرو HTP_REPACK_H

#شامل <stddef.h>
#شامل <stdint.h>

#شامل "ggml.h"
#شامل "htp/htp-tensor.h"

ساکن اندرونی bool htp_is_tiled_وزن_type(شمار ggml_type type) {
    واپس type == GGML_TYPE_Q4_0  || type == GGML_TYPE_Q4_1   ||
           type == GGML_TYPE_Q8_0  || type == GGML_TYPE_IQ4_NL ||
           type == GGML_TYPE_MXFP4 || type == GGML_TYPE_Q1_0   ||
           type == GGML_TYPE_Q6_K   ||
           type == GGML_TYPE_Q4_K  || type == GGML_TYPE_Q5_K   ||
           type == GGML_TYPE_Q3_K  || type == GGML_TYPE_Q2_K   ||
           type == GGML_TYPE_Q2_0  ||
           type == GGML_TYPE_F16   || type == GGML_TYPE_F32    ||
           type == GGML_TYPE_BF16;
}

ساکن اندرونی bool htp_is_tiled_وزن_tensor(ثابت ساخت ggml_tensor * t) {
    اگر (!ggml_is_contiguous(t) || ggml_is_view(t)) {
        واپس false;
    }
    اگر (!htp_is_tiled_وزن_type(t->type)) {
        واپس false;
    }
    اگر (t->ne[0] < 32) {
        واپس false;
    }
    اگر (!ggml_is_quantized(t->type)) {
        واپس (t->ne[0] % 32 == 0) && (t->ne[1] % 32 == 0) && (t->ne[2] == 1) && (t->ne[3] == 1);
    }
    واپس t->ne[3] == 1;
}

// Size of one tiled row in the DSP tiled layout. The Q3_K..Q6_K tiles store uncompressed scales/mins,
// so they are larger than the ggml blocks. For the other tiled types (Q2_K included) the tile has the same size.
ساکن اندرونی size_t htp_tiled_صف_سائز(شمار ggml_type type, int64_t ne0) {
    اگر (type == GGML_TYPE_F16 || type == GGML_TYPE_F32 || type == GGML_TYPE_BF16) {
        واپس (size_t) ne0 * حجم(ggml_fp16_t);
    }
    اگر (type == GGML_TYPE_Q6_K) {
        واپس (size_t) (ne0 / 32) * (HTP_TENSOR_TILE_سائز_Q6_K / 32);
    }
    اگر (type == GGML_TYPE_Q2_0) {
        واپس (size_t) (ne0 / 64) * 18;
    }
    اگر (type == GGML_TYPE_Q4_K) {
        واپس (size_t) (ne0 / 32) * (HTP_TENSOR_TILE_سائز_Q4_K / 32);
    }
    اگر (type == GGML_TYPE_Q5_K) {
        واپس (size_t) (ne0 / 32) * (HTP_TENSOR_TILE_سائز_Q5_K / 32);
    }
    اگر (type == GGML_TYPE_Q3_K) {
        واپس (size_t) (ne0 / 32) * (HTP_TENSOR_TILE_سائز_Q3_K / 32);
    }
    واپس ggml_صف_سائز(type, ne0);
}

ساکن اندرونی bool htp_is_hmx_وزن_type(شمار ggml_type type) {
    واپس htp_is_tiled_وزن_type(type);
}

خالی htp_repack_tensor_tiled(ساخت ggml_tensor * t, ثابت خالی * ڈیٹا, size_t سائز);
خالی htp_unpack_tensor_tiled(خالی * ڈیٹا, ثابت ساخت ggml_tensor * t, size_t انحراف, size_t سائز);

#اگر_ختم // HTP_REPACK_H
