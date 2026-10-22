#ifndef HTP_REPACK_H
#define HTP_REPACK_H

#ಅನ್ತರ್ಭಾವ <stddef.h>
#ಅನ್ತರ್ಭಾವ <stdint.h>

#ಅನ್ತರ್ಭಾವ "ggml.h"
#ಅನ್ತರ್ಭಾವ "htp/htp-tensor.h"

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ bool htp_is_tiled_weight_type(ಗಣನಾ ggml_type type) {
    ಪ್ರತ್ಯಾಗಮ type == GGML_TYPE_Q4_0  || type == GGML_TYPE_Q4_1   ||
           type == GGML_TYPE_Q8_0  || type == GGML_TYPE_IQ4_NL ||
           type == GGML_TYPE_MXFP4 || type == GGML_TYPE_Q1_0   ||
           type == GGML_TYPE_Q6_K   ||
           type == GGML_TYPE_Q4_K  || type == GGML_TYPE_Q5_K   ||
           type == GGML_TYPE_Q3_K  || type == GGML_TYPE_Q2_K   ||
           type == GGML_TYPE_Q2_0  ||
           type == GGML_TYPE_F16   || type == GGML_TYPE_F32    ||
           type == GGML_TYPE_BF16;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ bool htp_is_tiled_weight_tensor(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * t) {
    ಯದಿ (!ggml_is_contiguous(t) || ggml_is_view(t)) {
        ಪ್ರತ್ಯಾಗಮ false;
    }
    ಯದಿ (!htp_is_tiled_weight_type(t->type)) {
        ಪ್ರತ್ಯಾಗಮ false;
    }
    ಯದಿ (t->ne[0] < 32) {
        ಪ್ರತ್ಯಾಗಮ false;
    }
    ಯದಿ (!ggml_is_quantized(t->type)) {
        ಪ್ರತ್ಯಾಗಮ (t->ne[0] % 32 == 0) && (t->ne[1] % 32 == 0) && (t->ne[2] == 1) && (t->ne[3] == 1);
    }
    ಪ್ರತ್ಯಾಗಮ t->ne[3] == 1;
}

// Size of one tiled row in the DSP tiled layout. The Q3_K..Q6_K tiles store uncompressed scales/mins,
// so they are larger than the ggml blocks. For the other tiled types (Q2_K included) the tile has the same size.
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ size_t htp_tiled_row_size(ಗಣನಾ ggml_type type, int64_t ne0) {
    ಯದಿ (type == GGML_TYPE_F16 || type == GGML_TYPE_F32 || type == GGML_TYPE_BF16) {
        ಪ್ರತ್ಯಾಗಮ (size_t) ne0 * ಪರಿಮಾಣ(ggml_fp16_t);
    }
    ಯದಿ (type == GGML_TYPE_Q6_K) {
        ಪ್ರತ್ಯಾಗಮ (size_t) (ne0 / 32) * (HTP_TENSOR_TILE_SIZE_Q6_K / 32);
    }
    ಯದಿ (type == GGML_TYPE_Q2_0) {
        ಪ್ರತ್ಯಾಗಮ (size_t) (ne0 / 64) * 18;
    }
    ಯದಿ (type == GGML_TYPE_Q4_K) {
        ಪ್ರತ್ಯಾಗಮ (size_t) (ne0 / 32) * (HTP_TENSOR_TILE_SIZE_Q4_K / 32);
    }
    ಯದಿ (type == GGML_TYPE_Q5_K) {
        ಪ್ರತ್ಯಾಗಮ (size_t) (ne0 / 32) * (HTP_TENSOR_TILE_SIZE_Q5_K / 32);
    }
    ಯದಿ (type == GGML_TYPE_Q3_K) {
        ಪ್ರತ್ಯಾಗಮ (size_t) (ne0 / 32) * (HTP_TENSOR_TILE_SIZE_Q3_K / 32);
    }
    ಪ್ರತ್ಯಾಗಮ ggml_row_size(type, ne0);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ bool htp_is_hmx_weight_type(ಗಣನಾ ggml_type type) {
    ಪ್ರತ್ಯಾಗಮ htp_is_tiled_weight_type(type);
}

ಶೂನ್ಯ htp_repack_tensor_tiled(ಸಂರಚನಾ ggml_tensor * t, ಧ್ರುವ ಶೂನ್ಯ * data, size_t size);
ಶೂನ್ಯ htp_unpack_tensor_tiled(ಶೂನ್ಯ * data, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * t, size_t offset, size_t size);

#endif // HTP_REPACK_H
