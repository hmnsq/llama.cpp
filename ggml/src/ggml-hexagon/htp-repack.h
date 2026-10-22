#अगर_अघोषित HTP_REPACK_H
#घोषणा HTP_REPACK_H

#समावेश <stddef.h>
#समावेश <stdint.h>

#समावेश "ggml.h"
#समावेश "htp/htp-tensor.h"

अटल अंतरभूत bool htp_is_tiled_weight_type(क्रमागत ggml_type type) {
    वापस type == GGML_TYPE_Q4_0  || type == GGML_TYPE_Q4_1   ||
           type == GGML_TYPE_Q8_0  || type == GGML_TYPE_IQ4_NL ||
           type == GGML_TYPE_MXFP4 || type == GGML_TYPE_Q1_0   ||
           type == GGML_TYPE_Q6_K   ||
           type == GGML_TYPE_Q4_K  || type == GGML_TYPE_Q5_K   ||
           type == GGML_TYPE_Q3_K  || type == GGML_TYPE_Q2_K   ||
           type == GGML_TYPE_Q2_0  ||
           type == GGML_TYPE_F16   || type == GGML_TYPE_F32    ||
           type == GGML_TYPE_BF16;
}

अटल अंतरभूत bool htp_is_tiled_weight_tensor(स्थिर काष्ठा ggml_tensor * t) {
    अगर (!ggml_is_contiguous(t) || ggml_is_view(t)) {
        वापस false;
    }
    अगर (!htp_is_tiled_weight_type(t->type)) {
        वापस false;
    }
    अगर (t->ne[0] < 32) {
        वापस false;
    }
    अगर (!ggml_is_quantized(t->type)) {
        वापस (t->ne[0] % 32 == 0) && (t->ne[1] % 32 == 0) && (t->ne[2] == 1) && (t->ne[3] == 1);
    }
    वापस t->ne[3] == 1;
}

// Size of one tiled row in the DSP tiled layout. The Q3_K..Q6_K tiles store uncompressed scales/mins,
// so they are larger than the ggml blocks. For the other tiled types (Q2_K included) the tile has the same size.
अटल अंतरभूत माप_प्रकार htp_tiled_row_आकार(क्रमागत ggml_type type, int64_t ne0) {
    अगर (type == GGML_TYPE_F16 || type == GGML_TYPE_F32 || type == GGML_TYPE_BF16) {
        वापस (माप_प्रकार) ne0 * माप(ggml_fp16_t);
    }
    अगर (type == GGML_TYPE_Q6_K) {
        वापस (माप_प्रकार) (ne0 / 32) * (HTP_TENSOR_TILE_आकार_Q6_K / 32);
    }
    अगर (type == GGML_TYPE_Q2_0) {
        वापस (माप_प्रकार) (ne0 / 64) * 18;
    }
    अगर (type == GGML_TYPE_Q4_K) {
        वापस (माप_प्रकार) (ne0 / 32) * (HTP_TENSOR_TILE_आकार_Q4_K / 32);
    }
    अगर (type == GGML_TYPE_Q5_K) {
        वापस (माप_प्रकार) (ne0 / 32) * (HTP_TENSOR_TILE_आकार_Q5_K / 32);
    }
    अगर (type == GGML_TYPE_Q3_K) {
        वापस (माप_प्रकार) (ne0 / 32) * (HTP_TENSOR_TILE_आकार_Q3_K / 32);
    }
    वापस ggml_row_आकार(type, ne0);
}

अटल अंतरभूत bool htp_is_hmx_weight_type(क्रमागत ggml_type type) {
    वापस htp_is_tiled_weight_type(type);
}

व्योम htp_repack_tensor_tiled(काष्ठा ggml_tensor * t, स्थिर व्योम * तथ्य, माप_प्रकार आकार);
व्योम htp_unpack_tensor_tiled(व्योम * तथ्य, स्थिर काष्ठा ggml_tensor * t, माप_प्रकार offset, माप_प्रकार आकार);

#पूर्ण_अगर // HTP_REPACK_H
