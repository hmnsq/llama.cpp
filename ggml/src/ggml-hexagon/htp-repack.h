#ifndef HTP_REPACK_H
#define HTP_REPACK_H

#অন্তর্ভাব <stddef.h>
#অন্তর্ভাব <stdint.h>

#অন্তর্ভাব "ggml.h"
#অন্তর্ভাব "htp/htp-tensor.h"

স্থির অন্তঃরেখ bool htp_is_tiled_weight_type(গণনা ggml_type type) {
    ফেরত type == GGML_TYPE_Q4_0  || type == GGML_TYPE_Q4_1   ||
           type == GGML_TYPE_Q8_0  || type == GGML_TYPE_IQ4_NL ||
           type == GGML_TYPE_MXFP4 || type == GGML_TYPE_Q1_0   ||
           type == GGML_TYPE_Q6_K   ||
           type == GGML_TYPE_Q4_K  || type == GGML_TYPE_Q5_K   ||
           type == GGML_TYPE_Q3_K  || type == GGML_TYPE_Q2_K   ||
           type == GGML_TYPE_Q2_0  ||
           type == GGML_TYPE_F16   || type == GGML_TYPE_F32    ||
           type == GGML_TYPE_BF16;
}

স্থির অন্তঃরেখ bool htp_is_tiled_weight_tensor(ধ্রুবক কাঠামো ggml_tensor * t) {
    যদি (!ggml_is_contiguous(t) || ggml_is_view(t)) {
        ফেরত false;
    }
    যদি (!htp_is_tiled_weight_type(t->type)) {
        ফেরত false;
    }
    যদি (t->ne[0] < 32) {
        ফেরত false;
    }
    যদি (!ggml_is_quantized(t->type)) {
        ফেরত (t->ne[0] % 32 == 0) && (t->ne[1] % 32 == 0) && (t->ne[2] == 1) && (t->ne[3] == 1);
    }
    ফেরত t->ne[3] == 1;
}

// Size of one tiled row in the DSP tiled layout. The Q3_K..Q6_K tiles store uncompressed scales/mins,
// so they are larger than the ggml blocks. For the other tiled types (Q2_K included) the tile has the same size.
স্থির অন্তঃরেখ size_t htp_tiled_row_আকার(গণনা ggml_type type, int64_t ne0) {
    যদি (type == GGML_TYPE_F16 || type == GGML_TYPE_F32 || type == GGML_TYPE_BF16) {
        ফেরত (size_t) ne0 * আকার(ggml_fp16_t);
    }
    যদি (type == GGML_TYPE_Q6_K) {
        ফেরত (size_t) (ne0 / 32) * (HTP_TENSOR_TILE_আকার_Q6_K / 32);
    }
    যদি (type == GGML_TYPE_Q2_0) {
        ফেরত (size_t) (ne0 / 64) * 18;
    }
    যদি (type == GGML_TYPE_Q4_K) {
        ফেরত (size_t) (ne0 / 32) * (HTP_TENSOR_TILE_আকার_Q4_K / 32);
    }
    যদি (type == GGML_TYPE_Q5_K) {
        ফেরত (size_t) (ne0 / 32) * (HTP_TENSOR_TILE_আকার_Q5_K / 32);
    }
    যদি (type == GGML_TYPE_Q3_K) {
        ফেরত (size_t) (ne0 / 32) * (HTP_TENSOR_TILE_আকার_Q3_K / 32);
    }
    ফেরত ggml_row_আকার(type, ne0);
}

স্থির অন্তঃরেখ bool htp_is_hmx_weight_type(গণনা ggml_type type) {
    ফেরত htp_is_tiled_weight_type(type);
}

শূন্য htp_repack_tensor_tiled(কাঠামো ggml_tensor * t, ধ্রুবক শূন্য * তথ্য, size_t আকার_2);
শূন্য htp_unpack_tensor_tiled(শূন্য * তথ্য, ধ্রুবক কাঠামো ggml_tensor * t, size_t offset, size_t আকার_2);

#endif // HTP_REPACK_H
