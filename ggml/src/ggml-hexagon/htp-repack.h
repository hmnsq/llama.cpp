#ifndef HTP_REPACK_H
#define HTP_REPACK_H

#ଅନ୍ତର୍ଭାଵ <stddef.h>
#ଅନ୍ତର୍ଭାଵ <stdint.h>

#ଅନ୍ତର୍ଭାଵ "ggml.h"
#ଅନ୍ତର୍ଭାଵ "htp/htp-tensor.h"

ସ୍ଥିର ଧାଡ଼ିଭିତରେ bool htp_is_tiled_weight_type(ଗଣନା ggml_type type) {
    ଫେରନ୍ତୁ type == GGML_TYPE_Q4_0  || type == GGML_TYPE_Q4_1   ||
           type == GGML_TYPE_Q8_0  || type == GGML_TYPE_IQ4_NL ||
           type == GGML_TYPE_MXFP4 || type == GGML_TYPE_Q1_0   ||
           type == GGML_TYPE_Q6_K   ||
           type == GGML_TYPE_Q4_K  || type == GGML_TYPE_Q5_K   ||
           type == GGML_TYPE_Q3_K  || type == GGML_TYPE_Q2_K   ||
           type == GGML_TYPE_Q2_0  ||
           type == GGML_TYPE_F16   || type == GGML_TYPE_F32    ||
           type == GGML_TYPE_BF16;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ bool htp_is_tiled_weight_tensor(ଅଚଳ ସଂରଚନା ggml_tensor * t) {
    ଯଦି (!ggml_is_contiguous(t) || ggml_is_view(t)) {
        ଫେରନ୍ତୁ false;
    }
    ଯଦି (!htp_is_tiled_weight_type(t->type)) {
        ଫେରନ୍ତୁ false;
    }
    ଯଦି (t->ne[0] < 32) {
        ଫେରନ୍ତୁ false;
    }
    ଯଦି (!ggml_is_quantized(t->type)) {
        ଫେରନ୍ତୁ (t->ne[0] % 32 == 0) && (t->ne[1] % 32 == 0) && (t->ne[2] == 1) && (t->ne[3] == 1);
    }
    ଫେରନ୍ତୁ t->ne[3] == 1;
}

// Size of one tiled row in the DSP tiled layout. The Q3_K..Q6_K tiles store uncompressed scales/mins,
// so they are larger than the ggml blocks. For the other tiled types (Q2_K included) the tile has the same size.
ସ୍ଥିର ଧାଡ଼ିଭିତରେ size_t htp_tiled_row_size(ଗଣନା ggml_type type, int64_t ne0) {
    ଯଦି (type == GGML_TYPE_F16 || type == GGML_TYPE_F32 || type == GGML_TYPE_BF16) {
        ଫେରନ୍ତୁ (size_t) ne0 * ଆକାର(ggml_fp16_t);
    }
    ଯଦି (type == GGML_TYPE_Q6_K) {
        ଫେରନ୍ତୁ (size_t) (ne0 / 32) * (HTP_TENSOR_TILE_SIZE_Q6_K / 32);
    }
    ଯଦି (type == GGML_TYPE_Q2_0) {
        ଫେରନ୍ତୁ (size_t) (ne0 / 64) * 18;
    }
    ଯଦି (type == GGML_TYPE_Q4_K) {
        ଫେରନ୍ତୁ (size_t) (ne0 / 32) * (HTP_TENSOR_TILE_SIZE_Q4_K / 32);
    }
    ଯଦି (type == GGML_TYPE_Q5_K) {
        ଫେରନ୍ତୁ (size_t) (ne0 / 32) * (HTP_TENSOR_TILE_SIZE_Q5_K / 32);
    }
    ଯଦି (type == GGML_TYPE_Q3_K) {
        ଫେରନ୍ତୁ (size_t) (ne0 / 32) * (HTP_TENSOR_TILE_SIZE_Q3_K / 32);
    }
    ଫେରନ୍ତୁ ggml_row_size(type, ne0);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ bool htp_is_hmx_weight_type(ଗଣନା ggml_type type) {
    ଫେରନ୍ତୁ htp_is_tiled_weight_type(type);
}

ଶୂନ୍ୟ htp_repack_tensor_tiled(ସଂରଚନା ggml_tensor * t, ଅଚଳ ଶୂନ୍ୟ * data, size_t size);
ଶୂନ୍ୟ htp_unpack_tensor_tiled(ଶୂନ୍ୟ * data, ଅଚଳ ସଂରଚନା ggml_tensor * t, size_t offset, size_t size);

#endif // HTP_REPACK_H
