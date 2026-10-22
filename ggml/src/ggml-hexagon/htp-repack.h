#ifndef HTP_REPACK_H
#define HTP_REPACK_H

#অন্তর্ভাৱ <stddef.h>
#অন্তর্ভাৱ <stdint.h>

#অন্তর্ভাৱ "ggml.h"
#অন্তর্ভাৱ "htp/htp-tensor.h"

স্থির অন্তঃস্থ bool htp_is_tiled_weight_type(গণনা ggml_type type) {
    ঘূৰাই type == GGML_TYPE_Q4_0  || type == GGML_TYPE_Q4_1   ||
           type == GGML_TYPE_Q8_0  || type == GGML_TYPE_IQ4_NL ||
           type == GGML_TYPE_MXFP4 || type == GGML_TYPE_Q1_0   ||
           type == GGML_TYPE_Q6_K   ||
           type == GGML_TYPE_Q4_K  || type == GGML_TYPE_Q5_K   ||
           type == GGML_TYPE_Q3_K  || type == GGML_TYPE_Q2_K   ||
           type == GGML_TYPE_Q2_0  ||
           type == GGML_TYPE_F16   || type == GGML_TYPE_F32    ||
           type == GGML_TYPE_BF16;
}

স্থির অন্তঃস্থ bool htp_is_tiled_weight_tensor(ধ্রুৱ সংরচনা ggml_tensor * t) {
    যদি (!ggml_is_contiguous(t) || ggml_is_view(t)) {
        ঘূৰাই false;
    }
    যদি (!htp_is_tiled_weight_type(t->type)) {
        ঘূৰাই false;
    }
    যদি (t->ne[0] < 32) {
        ঘূৰাই false;
    }
    যদি (!ggml_is_quantized(t->type)) {
        ঘূৰাই (t->ne[0] % 32 == 0) && (t->ne[1] % 32 == 0) && (t->ne[2] == 1) && (t->ne[3] == 1);
    }
    ঘূৰাই t->ne[3] == 1;
}

// Size of one tiled row in the DSP tiled layout. The Q3_K..Q6_K tiles store uncompressed scales/mins,
// so they are larger than the ggml blocks. For the other tiled types (Q2_K included) the tile has the same size.
স্থির অন্তঃস্থ size_t htp_tiled_row_size(গণনা ggml_type type, int64_t ne0) {
    যদি (type == GGML_TYPE_F16 || type == GGML_TYPE_F32 || type == GGML_TYPE_BF16) {
        ঘূৰাই (size_t) ne0 * পরিমাণ(ggml_fp16_t);
    }
    যদি (type == GGML_TYPE_Q6_K) {
        ঘূৰাই (size_t) (ne0 / 32) * (HTP_TENSOR_TILE_SIZE_Q6_K / 32);
    }
    যদি (type == GGML_TYPE_Q2_0) {
        ঘূৰাই (size_t) (ne0 / 64) * 18;
    }
    যদি (type == GGML_TYPE_Q4_K) {
        ঘূৰাই (size_t) (ne0 / 32) * (HTP_TENSOR_TILE_SIZE_Q4_K / 32);
    }
    যদি (type == GGML_TYPE_Q5_K) {
        ঘূৰাই (size_t) (ne0 / 32) * (HTP_TENSOR_TILE_SIZE_Q5_K / 32);
    }
    যদি (type == GGML_TYPE_Q3_K) {
        ঘূৰাই (size_t) (ne0 / 32) * (HTP_TENSOR_TILE_SIZE_Q3_K / 32);
    }
    ঘূৰাই ggml_row_size(type, ne0);
}

স্থির অন্তঃস্থ bool htp_is_hmx_weight_type(গণনা ggml_type type) {
    ঘূৰাই htp_is_tiled_weight_type(type);
}

শূন্য htp_repack_tensor_tiled(সংরচনা ggml_tensor * t, ধ্রুৱ শূন্য * data, size_t size);
শূন্য htp_unpack_tensor_tiled(শূন্য * data, ধ্রুৱ সংরচনা ggml_tensor * t, size_t offset, size_t size);

#endif // HTP_REPACK_H
