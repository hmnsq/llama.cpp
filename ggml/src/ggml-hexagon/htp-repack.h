#که_نه_تعریف_شوی HTP_REPACK_H
#تعریف_کړه HTP_REPACK_H

#ضمن <stddef.h>
#ضمن <stdint.h>

#ضمن "ggml.h"
#ضمن "htp/htp-tensor.h"

ساكن مضمن bool htp_is_tiled_وزن_type(تعداد ggml_type type) {
    أرجع type == GGML_TYPE_Q4_0  || type == GGML_TYPE_Q4_1   ||
           type == GGML_TYPE_Q8_0  || type == GGML_TYPE_IQ4_NL ||
           type == GGML_TYPE_MXFP4 || type == GGML_TYPE_Q1_0   ||
           type == GGML_TYPE_Q6_K   ||
           type == GGML_TYPE_Q4_K  || type == GGML_TYPE_Q5_K   ||
           type == GGML_TYPE_Q3_K  || type == GGML_TYPE_Q2_K   ||
           type == GGML_TYPE_Q2_0  ||
           type == GGML_TYPE_F16   || type == GGML_TYPE_F32    ||
           type == GGML_TYPE_BF16;
}

ساكن مضمن bool htp_is_tiled_وزن_tensor(ثابت بنية ggml_tensor * t) {
    إذا (!ggml_is_contiguous(t) || ggml_is_view(t)) {
        أرجع false;
    }
    إذا (!htp_is_tiled_وزن_type(t->type)) {
        أرجع false;
    }
    إذا (t->ne[0] < 32) {
        أرجع false;
    }
    إذا (!ggml_is_quantized(t->type)) {
        أرجع (t->ne[0] % 32 == 0) && (t->ne[1] % 32 == 0) && (t->ne[2] == 1) && (t->ne[3] == 1);
    }
    أرجع t->ne[3] == 1;
}

// Size of one tiled row in the DSP tiled layout. The Q3_K..Q6_K tiles store uncompressed scales/mins,
// so they are larger than the ggml blocks. For the other tiled types (Q2_K included) the tile has the same size.
ساكن مضمن size_t htp_tiled_لیکه_اندازه(تعداد ggml_type type, int64_t ne0) {
    إذا (type == GGML_TYPE_F16 || type == GGML_TYPE_F32 || type == GGML_TYPE_BF16) {
        أرجع (size_t) ne0 * حجم(ggml_fp16_t);
    }
    إذا (type == GGML_TYPE_Q6_K) {
        أرجع (size_t) (ne0 / 32) * (HTP_TENSOR_TILE_اندازه_Q6_K / 32);
    }
    إذا (type == GGML_TYPE_Q2_0) {
        أرجع (size_t) (ne0 / 64) * 18;
    }
    إذا (type == GGML_TYPE_Q4_K) {
        أرجع (size_t) (ne0 / 32) * (HTP_TENSOR_TILE_اندازه_Q4_K / 32);
    }
    إذا (type == GGML_TYPE_Q5_K) {
        أرجع (size_t) (ne0 / 32) * (HTP_TENSOR_TILE_اندازه_Q5_K / 32);
    }
    إذا (type == GGML_TYPE_Q3_K) {
        أرجع (size_t) (ne0 / 32) * (HTP_TENSOR_TILE_اندازه_Q3_K / 32);
    }
    أرجع ggml_لیکه_اندازه(type, ne0);
}

ساكن مضمن bool htp_is_hmx_وزن_type(تعداد ggml_type type) {
    أرجع htp_is_tiled_وزن_type(type);
}

فراغ htp_repack_tensor_tiled(بنية ggml_tensor * t, ثابت فراغ * معلومات, size_t اندازه);
فراغ htp_unpack_tensor_tiled(فراغ * معلومات, ثابت بنية ggml_tensor * t, size_t دننه_واټن, size_t اندازه);

#د_که_پای // HTP_REPACK_H
