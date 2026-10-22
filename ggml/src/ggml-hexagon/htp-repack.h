#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ HTP_REPACK_H
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_REPACK_H

#ಅನ್ತರ್ಭಾವ <stddef.h>
#ಅನ್ತರ್ಭಾವ <stdint.h>

#ಅನ್ತರ್ಭಾವ "ggml.h"
#ಅನ್ತರ್ಭಾವ "htp/htp-tensor.h"

ಸ್ಥಿರ ಸಾಲೊಳಗೆ bool htp_is_tiled_ತೂಕ_type(ಎಣಿಕೆ ggml_type type) {
    ಹಿಂತಿರುಗಿ type == GGML_TYPE_Q4_0  || type == GGML_TYPE_Q4_1   ||
           type == GGML_TYPE_Q8_0  || type == GGML_TYPE_IQ4_NL ||
           type == GGML_TYPE_MXFP4 || type == GGML_TYPE_Q1_0   ||
           type == GGML_TYPE_Q6_K   ||
           type == GGML_TYPE_Q4_K  || type == GGML_TYPE_Q5_K   ||
           type == GGML_TYPE_Q3_K  || type == GGML_TYPE_Q2_K   ||
           type == GGML_TYPE_Q2_0  ||
           type == GGML_TYPE_F16   || type == GGML_TYPE_F32    ||
           type == GGML_TYPE_BF16;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ bool htp_is_tiled_ತೂಕ_tensor(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * t) {
    ಆದರೆ (!ggml_is_contiguous(t) || ggml_is_view(t)) {
        ಹಿಂತಿರುಗಿ false;
    }
    ಆದರೆ (!htp_is_tiled_ತೂಕ_type(t->type)) {
        ಹಿಂತಿರುಗಿ false;
    }
    ಆದರೆ (t->ne[0] < 32) {
        ಹಿಂತಿರುಗಿ false;
    }
    ಆದರೆ (!ggml_is_quantized(t->type)) {
        ಹಿಂತಿರುಗಿ (t->ne[0] % 32 == 0) && (t->ne[1] % 32 == 0) && (t->ne[2] == 1) && (t->ne[3] == 1);
    }
    ಹಿಂತಿರುಗಿ t->ne[3] == 1;
}

// Size of one tiled row in the DSP tiled layout. The Q3_K..Q6_K tiles store uncompressed scales/mins,
// so they are larger than the ggml blocks. For the other tiled types (Q2_K included) the tile has the same size.
ಸ್ಥಿರ ಸಾಲೊಳಗೆ size_t htp_tiled_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(ಎಣಿಕೆ ggml_type type, int64_t ne0) {
    ಆದರೆ (type == GGML_TYPE_F16 || type == GGML_TYPE_F32 || type == GGML_TYPE_BF16) {
        ಹಿಂತಿರುಗಿ (size_t) ne0 * ಗಾತ್ರ(ggml_fp16_t);
    }
    ಆದರೆ (type == GGML_TYPE_Q6_K) {
        ಹಿಂತಿರುಗಿ (size_t) (ne0 / 32) * (HTP_TENSOR_TILE_ಗಾತ್ರ_Q6_K / 32);
    }
    ಆದರೆ (type == GGML_TYPE_Q2_0) {
        ಹಿಂತಿರುಗಿ (size_t) (ne0 / 64) * 18;
    }
    ಆದರೆ (type == GGML_TYPE_Q4_K) {
        ಹಿಂತಿರುಗಿ (size_t) (ne0 / 32) * (HTP_TENSOR_TILE_ಗಾತ್ರ_Q4_K / 32);
    }
    ಆದರೆ (type == GGML_TYPE_Q5_K) {
        ಹಿಂತಿರುಗಿ (size_t) (ne0 / 32) * (HTP_TENSOR_TILE_ಗಾತ್ರ_Q5_K / 32);
    }
    ಆದರೆ (type == GGML_TYPE_Q3_K) {
        ಹಿಂತಿರುಗಿ (size_t) (ne0 / 32) * (HTP_TENSOR_TILE_ಗಾತ್ರ_Q3_K / 32);
    }
    ಹಿಂತಿರುಗಿ ggml_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(type, ne0);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ bool htp_is_hmx_ತೂಕ_type(ಎಣಿಕೆ ggml_type type) {
    ಹಿಂತಿರುಗಿ htp_is_tiled_ತೂಕ_type(type);
}

ಶೂನ್ಯ htp_repack_tensor_tiled(ರಚನೆ ggml_tensor * t, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಗಾತ್ರ_2);
ಶೂನ್ಯ htp_unpack_tensor_tiled(ಶೂನ್ಯ * ದತ್ತಾಂಶ, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * t, size_t ಸ್ಥಾನಾಂತರ, size_t ಗಾತ್ರ_2);

#ಆದರೆ_ಅಂತ್ಯ // HTP_REPACK_H
