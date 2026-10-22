// ET kernel entry-point parameter structs and tensor helpers.

#ifndef GGML_TENSOR_H
#define GGML_TENSOR_H

#அந்தர்பாவ <stddef.h>
#அந்தர்பாவ <stdint.h>

#அந்தர்பாவ "ggml.h"

கட்டமைப்பு ggml_et_இருமம்_params {
    கட்டமைப்பு ggml_tensor src0;
    கட்டமைப்பு ggml_tensor src1;
    கட்டமைப்பு ggml_tensor dst;
};

// bias.data == NULL -> unfused MUL_MAT; otherwise dst = mat_mul(...) + bias.
கட்டமைப்பு ggml_et_mm_q8_params {
    கட்டமைப்பு ggml_tensor src0;
    கட்டமைப்பு ggml_tensor src1;
    கட்டமைப்பு ggml_tensor dst;
    கட்டமைப்பு ggml_tensor bias;
};

கட்டமைப்பு ggml_et_mul_mat_id_params {
    கட்டமைப்பு ggml_tensor src0;  // [K, M, n_expert]
    கட்டமைப்பு ggml_tensor src1;  // [K, n_expert_used, batch]
    கட்டமைப்பு ggml_tensor src2;  // [n_expert_used, batch] (I32 expert indices)
    கட்டமைப்பு ggml_tensor dst;   // [M, n_expert_used, batch, 1]
};

// ne[i] == 1 axes are skipped: their stride is unobservable.
நிலையான உள்வரி முழு ggml_tensor_is_contiguous(மாறிலி கட்டமைப்பு ggml_tensor * t, முழு type_size) {
    int64_t expected = type_size;
    க்கு (முழு i = 0; i < GGML_MAX_DIMS; i++) {
        எனில் (t->ne[i] > 1 && (int64_t) t->nb[i] != expected) {
            திரும்பு 0;
        }
        expected *= t->ne[i];
    }
    திரும்பு 1;
}

#endif  // GGML_TENSOR_H
