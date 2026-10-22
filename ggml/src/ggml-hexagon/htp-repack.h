#ifndef HTP_REPACK_H
#define HTP_REPACK_H

#અન્તર્ભાવ <stddef.h>
#અન્તર્ભાવ <stdint.h>

#અન્તર્ભાવ "ggml.h"
#અન્તર્ભાવ "htp/htp-tensor.h"

સ્થિર અંતઃરેખ bool htp_is_tiled_weight_type(ગણના ggml_type type) {
    પાછા type == GGML_TYPE_Q4_0  || type == GGML_TYPE_Q4_1   ||
           type == GGML_TYPE_Q8_0  || type == GGML_TYPE_IQ4_NL ||
           type == GGML_TYPE_MXFP4 || type == GGML_TYPE_Q1_0   ||
           type == GGML_TYPE_Q6_K   ||
           type == GGML_TYPE_Q4_K  || type == GGML_TYPE_Q5_K   ||
           type == GGML_TYPE_Q3_K  || type == GGML_TYPE_Q2_K   ||
           type == GGML_TYPE_Q2_0  ||
           type == GGML_TYPE_F16   || type == GGML_TYPE_F32    ||
           type == GGML_TYPE_BF16;
}

સ્થિર અંતઃરેખ bool htp_is_tiled_weight_tensor(અચલ સંરચના ggml_tensor * t) {
    જો (!ggml_is_contiguous(t) || ggml_is_view(t)) {
        પાછા false;
    }
    જો (!htp_is_tiled_weight_type(t->type)) {
        પાછા false;
    }
    જો (t->ne[0] < 32) {
        પાછા false;
    }
    જો (!ggml_is_quantized(t->type)) {
        પાછા (t->ne[0] % 32 == 0) && (t->ne[1] % 32 == 0) && (t->ne[2] == 1) && (t->ne[3] == 1);
    }
    પાછા t->ne[3] == 1;
}

// Size of one tiled row in the DSP tiled layout. The Q3_K..Q6_K tiles store uncompressed scales/mins,
// so they are larger than the ggml blocks. For the other tiled types (Q2_K included) the tile has the same size.
સ્થિર અંતઃરેખ size_t htp_tiled_row_size(ગણના ggml_type type, int64_t ne0) {
    જો (type == GGML_TYPE_F16 || type == GGML_TYPE_F32 || type == GGML_TYPE_BF16) {
        પાછા (size_t) ne0 * કદ(ggml_fp16_t);
    }
    જો (type == GGML_TYPE_Q6_K) {
        પાછા (size_t) (ne0 / 32) * (HTP_TENSOR_TILE_SIZE_Q6_K / 32);
    }
    જો (type == GGML_TYPE_Q2_0) {
        પાછા (size_t) (ne0 / 64) * 18;
    }
    જો (type == GGML_TYPE_Q4_K) {
        પાછા (size_t) (ne0 / 32) * (HTP_TENSOR_TILE_SIZE_Q4_K / 32);
    }
    જો (type == GGML_TYPE_Q5_K) {
        પાછા (size_t) (ne0 / 32) * (HTP_TENSOR_TILE_SIZE_Q5_K / 32);
    }
    જો (type == GGML_TYPE_Q3_K) {
        પાછા (size_t) (ne0 / 32) * (HTP_TENSOR_TILE_SIZE_Q3_K / 32);
    }
    પાછા ggml_row_size(type, ne0);
}

સ્થિર અંતઃરેખ bool htp_is_hmx_weight_type(ગણના ggml_type type) {
    પાછા htp_is_tiled_weight_type(type);
}

શૂન્ય htp_repack_tensor_tiled(સંરચના ggml_tensor * t, અચલ શૂન્ય * data, size_t size);
શૂન્ય htp_unpack_tensor_tiled(શૂન્ય * data, અચલ સંરચના ggml_tensor * t, size_t offset, size_t size);

#endif // HTP_REPACK_H
