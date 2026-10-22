#ifndef HTP_REPACK_H
#define HTP_REPACK_H

#अन्तर्भाव <stddef.h>
#अन्तर्भाव <stdint.h>

#अन्तर्भाव "ggml.h"
#अन्तर्भाव "htp/htp-tensor.h"

स्थिर अन्तःस्थ bool htp_is_tiled_weight_type(गणना ggml_type type) {
    पटिनिवत्त type == GGML_TYPE_Q4_0  || type == GGML_TYPE_Q4_1   ||
           type == GGML_TYPE_Q8_0  || type == GGML_TYPE_IQ4_NL ||
           type == GGML_TYPE_MXFP4 || type == GGML_TYPE_Q1_0   ||
           type == GGML_TYPE_Q6_K   ||
           type == GGML_TYPE_Q4_K  || type == GGML_TYPE_Q5_K   ||
           type == GGML_TYPE_Q3_K  || type == GGML_TYPE_Q2_K   ||
           type == GGML_TYPE_Q2_0  ||
           type == GGML_TYPE_F16   || type == GGML_TYPE_F32    ||
           type == GGML_TYPE_BF16;
}

स्थिर अन्तःस्थ bool htp_is_tiled_weight_tensor(ध्रुव संरचना ggml_tensor * t) {
    सचे (!ggml_is_contiguous(t) || ggml_is_view(t)) {
        पटिनिवत्त false;
    }
    सचे (!htp_is_tiled_weight_type(t->type)) {
        पटिनिवत्त false;
    }
    सचे (t->ne[0] < 32) {
        पटिनिवत्त false;
    }
    सचे (!ggml_is_quantized(t->type)) {
        पटिनिवत्त (t->ne[0] % 32 == 0) && (t->ne[1] % 32 == 0) && (t->ne[2] == 1) && (t->ne[3] == 1);
    }
    पटिनिवत्त t->ne[3] == 1;
}

// Size of one tiled row in the DSP tiled layout. The Q3_K..Q6_K tiles store uncompressed scales/mins,
// so they are larger than the ggml blocks. For the other tiled types (Q2_K included) the tile has the same size.
स्थिर अन्तःस्थ size_t htp_tiled_row_size(गणना ggml_type type, int64_t ne0) {
    सचे (type == GGML_TYPE_F16 || type == GGML_TYPE_F32 || type == GGML_TYPE_BF16) {
        पटिनिवत्त (size_t) ne0 * परिमाण(ggml_fp16_t);
    }
    सचे (type == GGML_TYPE_Q6_K) {
        पटिनिवत्त (size_t) (ne0 / 32) * (HTP_TENSOR_TILE_SIZE_Q6_K / 32);
    }
    सचे (type == GGML_TYPE_Q2_0) {
        पटिनिवत्त (size_t) (ne0 / 64) * 18;
    }
    सचे (type == GGML_TYPE_Q4_K) {
        पटिनिवत्त (size_t) (ne0 / 32) * (HTP_TENSOR_TILE_SIZE_Q4_K / 32);
    }
    सचे (type == GGML_TYPE_Q5_K) {
        पटिनिवत्त (size_t) (ne0 / 32) * (HTP_TENSOR_TILE_SIZE_Q5_K / 32);
    }
    सचे (type == GGML_TYPE_Q3_K) {
        पटिनिवत्त (size_t) (ne0 / 32) * (HTP_TENSOR_TILE_SIZE_Q3_K / 32);
    }
    पटिनिवत्त ggml_row_size(type, ne0);
}

स्थिर अन्तःस्थ bool htp_is_hmx_weight_type(गणना ggml_type type) {
    पटिनिवत्त htp_is_tiled_weight_type(type);
}

शून्य htp_repack_tensor_tiled(संरचना ggml_tensor * t, ध्रुव शून्य * data, size_t size);
शून्य htp_unpack_tensor_tiled(शून्य * data, ध्रुव संरचना ggml_tensor * t, size_t offset, size_t size);

#endif // HTP_REPACK_H
