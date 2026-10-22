#ifndef HTP_REPACK_H
#define HTP_REPACK_H

#അന്തര്ഭാവ <stddef.h>
#അന്തര്ഭാവ <stdint.h>

#അന്തര്ഭാവ "ggml.h"
#അന്തര്ഭാവ "htp/htp-tensor.h"

സ്ഥിതം വരിയിൽ bool htp_is_tiled_weight_type(എണ്ണൽ ggml_type type) {
    തിരികെ type == GGML_TYPE_Q4_0  || type == GGML_TYPE_Q4_1   ||
           type == GGML_TYPE_Q8_0  || type == GGML_TYPE_IQ4_NL ||
           type == GGML_TYPE_MXFP4 || type == GGML_TYPE_Q1_0   ||
           type == GGML_TYPE_Q6_K   ||
           type == GGML_TYPE_Q4_K  || type == GGML_TYPE_Q5_K   ||
           type == GGML_TYPE_Q3_K  || type == GGML_TYPE_Q2_K   ||
           type == GGML_TYPE_Q2_0  ||
           type == GGML_TYPE_F16   || type == GGML_TYPE_F32    ||
           type == GGML_TYPE_BF16;
}

സ്ഥിതം വരിയിൽ bool htp_is_tiled_weight_tensor(സ്ഥിരം ഘടന ggml_tensor * t) {
    എങ്കിൽ (!ggml_is_contiguous(t) || ggml_is_view(t)) {
        തിരികെ false;
    }
    എങ്കിൽ (!htp_is_tiled_weight_type(t->type)) {
        തിരികെ false;
    }
    എങ്കിൽ (t->ne[0] < 32) {
        തിരികെ false;
    }
    എങ്കിൽ (!ggml_is_quantized(t->type)) {
        തിരികെ (t->ne[0] % 32 == 0) && (t->ne[1] % 32 == 0) && (t->ne[2] == 1) && (t->ne[3] == 1);
    }
    തിരികെ t->ne[3] == 1;
}

// Size of one tiled row in the DSP tiled layout. The Q3_K..Q6_K tiles store uncompressed scales/mins,
// so they are larger than the ggml blocks. For the other tiled types (Q2_K included) the tile has the same size.
സ്ഥിതം വരിയിൽ size_t htp_tiled_row_size(എണ്ണൽ ggml_type type, int64_t ne0) {
    എങ്കിൽ (type == GGML_TYPE_F16 || type == GGML_TYPE_F32 || type == GGML_TYPE_BF16) {
        തിരികെ (size_t) ne0 * വലുപ്പം(ggml_fp16_t);
    }
    എങ്കിൽ (type == GGML_TYPE_Q6_K) {
        തിരികെ (size_t) (ne0 / 32) * (HTP_TENSOR_TILE_SIZE_Q6_K / 32);
    }
    എങ്കിൽ (type == GGML_TYPE_Q2_0) {
        തിരികെ (size_t) (ne0 / 64) * 18;
    }
    എങ്കിൽ (type == GGML_TYPE_Q4_K) {
        തിരികെ (size_t) (ne0 / 32) * (HTP_TENSOR_TILE_SIZE_Q4_K / 32);
    }
    എങ്കിൽ (type == GGML_TYPE_Q5_K) {
        തിരികെ (size_t) (ne0 / 32) * (HTP_TENSOR_TILE_SIZE_Q5_K / 32);
    }
    എങ്കിൽ (type == GGML_TYPE_Q3_K) {
        തിരികെ (size_t) (ne0 / 32) * (HTP_TENSOR_TILE_SIZE_Q3_K / 32);
    }
    തിരികെ ggml_row_size(type, ne0);
}

സ്ഥിതം വരിയിൽ bool htp_is_hmx_weight_type(എണ്ണൽ ggml_type type) {
    തിരികെ htp_is_tiled_weight_type(type);
}

ശൂന്യം htp_repack_tensor_tiled(ഘടന ggml_tensor * t, സ്ഥിരം ശൂന്യം * data, size_t size);
ശൂന്യം htp_unpack_tensor_tiled(ശൂന്യം * data, സ്ഥിരം ഘടന ggml_tensor * t, size_t offset, size_t size);

#endif // HTP_REPACK_H
