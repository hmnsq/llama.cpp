#ifndef HTP_REPACK_H
#define HTP_REPACK_H

#அந்தர்பாவ <stddef.h>
#அந்தர்பாவ <stdint.h>

#அந்தர்பாவ "ggml.h"
#அந்தர்பாவ "htp/htp-tensor.h"

நிலையான உள்வரி bool htp_is_tiled_weight_type(எண்ணிக்கை ggml_type type) {
    திரும்பு type == GGML_TYPE_Q4_0  || type == GGML_TYPE_Q4_1   ||
           type == GGML_TYPE_Q8_0  || type == GGML_TYPE_IQ4_NL ||
           type == GGML_TYPE_MXFP4 || type == GGML_TYPE_Q1_0   ||
           type == GGML_TYPE_Q6_K   ||
           type == GGML_TYPE_Q4_K  || type == GGML_TYPE_Q5_K   ||
           type == GGML_TYPE_Q3_K  || type == GGML_TYPE_Q2_K   ||
           type == GGML_TYPE_Q2_0  ||
           type == GGML_TYPE_F16   || type == GGML_TYPE_F32    ||
           type == GGML_TYPE_BF16;
}

நிலையான உள்வரி bool htp_is_tiled_weight_tensor(மாறிலி கட்டமைப்பு ggml_tensor * t) {
    எனில் (!ggml_is_contiguous(t) || ggml_is_view(t)) {
        திரும்பு false;
    }
    எனில் (!htp_is_tiled_weight_type(t->type)) {
        திரும்பு false;
    }
    எனில் (t->ne[0] < 32) {
        திரும்பு false;
    }
    எனில் (!ggml_is_quantized(t->type)) {
        திரும்பு (t->ne[0] % 32 == 0) && (t->ne[1] % 32 == 0) && (t->ne[2] == 1) && (t->ne[3] == 1);
    }
    திரும்பு t->ne[3] == 1;
}

// Size of one tiled row in the DSP tiled layout. The Q3_K..Q6_K tiles store uncompressed scales/mins,
// so they are larger than the ggml blocks. For the other tiled types (Q2_K included) the tile has the same size.
நிலையான உள்வரி size_t htp_tiled_row_size(எண்ணிக்கை ggml_type type, int64_t ne0) {
    எனில் (type == GGML_TYPE_F16 || type == GGML_TYPE_F32 || type == GGML_TYPE_BF16) {
        திரும்பு (size_t) ne0 * அளவு(ggml_fp16_t);
    }
    எனில் (type == GGML_TYPE_Q6_K) {
        திரும்பு (size_t) (ne0 / 32) * (HTP_TENSOR_TILE_SIZE_Q6_K / 32);
    }
    எனில் (type == GGML_TYPE_Q2_0) {
        திரும்பு (size_t) (ne0 / 64) * 18;
    }
    எனில் (type == GGML_TYPE_Q4_K) {
        திரும்பு (size_t) (ne0 / 32) * (HTP_TENSOR_TILE_SIZE_Q4_K / 32);
    }
    எனில் (type == GGML_TYPE_Q5_K) {
        திரும்பு (size_t) (ne0 / 32) * (HTP_TENSOR_TILE_SIZE_Q5_K / 32);
    }
    எனில் (type == GGML_TYPE_Q3_K) {
        திரும்பு (size_t) (ne0 / 32) * (HTP_TENSOR_TILE_SIZE_Q3_K / 32);
    }
    திரும்பு ggml_row_size(type, ne0);
}

நிலையான உள்வரி bool htp_is_hmx_weight_type(எண்ணிக்கை ggml_type type) {
    திரும்பு htp_is_tiled_weight_type(type);
}

வெறுமை htp_repack_tensor_tiled(கட்டமைப்பு ggml_tensor * t, மாறிலி வெறுமை * data, size_t size);
வெறுமை htp_unpack_tensor_tiled(வெறுமை * data, மாறிலி கட்டமைப்பு ggml_tensor * t, size_t offset, size_t size);

#endif // HTP_REPACK_H
