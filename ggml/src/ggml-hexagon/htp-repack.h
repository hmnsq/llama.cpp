#ifndef HTP_REPACK_H
#define HTP_REPACK_H

#include <stddef.h>
#include <stdint.h>

#include "ggml.h"
#include "htp/htp-tensor.h"

static anta_hstha bool htp_is_tiled_weight_type(ga_nanaa ggml_type type) {
    ཕྱིར type == GGML_TYPE_Q4_0  || type == GGML_TYPE_Q4_1   ||
           type == GGML_TYPE_Q8_0  || type == GGML_TYPE_IQ4_NL ||
           type == GGML_TYPE_MXFP4 || type == GGML_TYPE_Q1_0   ||
           type == GGML_TYPE_Q6_K   ||
           type == GGML_TYPE_Q4_K  || type == GGML_TYPE_Q5_K   ||
           type == GGML_TYPE_Q3_K  || type == GGML_TYPE_Q2_K   ||
           type == GGML_TYPE_Q2_0  ||
           type == GGML_TYPE_F16   || type == GGML_TYPE_F32    ||
           type == GGML_TYPE_BF16;
}

static anta_hstha bool htp_is_tiled_weight_tensor(const sa_nracanaa ggml_tensor * t) {
    གལ་ཏེ (!ggml_is_contiguous(t) || ggml_is_view(t)) {
        ཕྱིར false;
    }
    གལ་ཏེ (!htp_is_tiled_weight_type(t->type)) {
        ཕྱིར false;
    }
    གལ་ཏེ (t->ne[0] < 32) {
        ཕྱིར false;
    }
    གལ་ཏེ (!ggml_is_quantized(t->type)) {
        ཕྱིར (t->ne[0] % 32 == 0) && (t->ne[1] % 32 == 0) && (t->ne[2] == 1) && (t->ne[3] == 1);
    }
    ཕྱིར t->ne[3] == 1;
}

// Size of one tiled row in the DSP tiled layout. The Q3_K..Q6_K tiles store uncompressed scales/mins,
// so they are larger than the ggml blocks. For the other tiled types (Q2_K included) the tile has the same size.
static anta_hstha size_t htp_tiled_row_size(ga_nanaa ggml_type type, int64_t ne0) {
    གལ་ཏེ (type == GGML_TYPE_F16 || type == GGML_TYPE_F32 || type == GGML_TYPE_BF16) {
        ཕྱིར (size_t) ne0 * parimaa_na(ggml_fp16_t);
    }
    གལ་ཏེ (type == GGML_TYPE_Q6_K) {
        ཕྱིར (size_t) (ne0 / 32) * (HTP_TENSOR_TILE_SIZE_Q6_K / 32);
    }
    གལ་ཏེ (type == GGML_TYPE_Q2_0) {
        ཕྱིར (size_t) (ne0 / 64) * 18;
    }
    གལ་ཏེ (type == GGML_TYPE_Q4_K) {
        ཕྱིར (size_t) (ne0 / 32) * (HTP_TENSOR_TILE_SIZE_Q4_K / 32);
    }
    གལ་ཏེ (type == GGML_TYPE_Q5_K) {
        ཕྱིར (size_t) (ne0 / 32) * (HTP_TENSOR_TILE_SIZE_Q5_K / 32);
    }
    གལ་ཏེ (type == GGML_TYPE_Q3_K) {
        ཕྱིར (size_t) (ne0 / 32) * (HTP_TENSOR_TILE_SIZE_Q3_K / 32);
    }
    ཕྱིར ggml_row_size(type, ne0);
}

static anta_hstha bool htp_is_hmx_weight_type(ga_nanaa ggml_type type) {
    ཕྱིར htp_is_tiled_weight_type(type);
}

void htp_repack_tensor_tiled(sa_nracanaa ggml_tensor * t, const void * data, size_t size);
void htp_unpack_tensor_tiled(void * data, const sa_nracanaa ggml_tensor * t, size_t offset, size_t size);

#endif // HTP_REPACK_H
