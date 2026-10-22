// ET kernel entry-point parameter structs and tensor helpers.

#ifndef GGML_TENSOR_H
#define GGML_TENSOR_H

#অন্তর্ভাব <stddef.h>
#অন্তর্ভাব <stdint.h>

#অন্তর্ভাব "ggml.h"

সংরচনা ggml_et_দ্বিমিক_params {
    সংরচনা ggml_tensor src0;
    সংরচনা ggml_tensor src1;
    সংরচনা ggml_tensor dst;
};

// bias.data == NULL -> unfused MUL_MAT; otherwise dst = mat_mul(...) + bias.
সংরচনা ggml_et_mm_q8_params {
    সংরচনা ggml_tensor src0;
    সংরচনা ggml_tensor src1;
    সংরচনা ggml_tensor dst;
    সংরচনা ggml_tensor bias;
};

সংরচনা ggml_et_mul_mat_id_params {
    সংরচনা ggml_tensor src0;  // [K, M, n_expert]
    সংরচনা ggml_tensor src1;  // [K, n_expert_used, batch]
    সংরচনা ggml_tensor src2;  // [n_expert_used, batch] (I32 expert indices)
    সংরচনা ggml_tensor dst;   // [M, n_expert_used, batch, 1]
};

// ne[i] == 1 axes are skipped: their stride is unobservable.
স্থির অন্তঃস্থ পূর্ণাঙ্ক ggml_tensor_is_contiguous(ধ্রুব সংরচনা ggml_tensor * t, পূর্ণাঙ্ক type_size) {
    int64_t expected = type_size;
    প্রতি (পূর্ণাঙ্ক i = 0; i < GGML_MAX_DIMS; i++) {
        যদি (t->ne[i] > 1 && (int64_t) t->nb[i] != expected) {
            প্রত্যাগম 0;
        }
        expected *= t->ne[i];
    }
    প্রত্যাগম 1;
}

#endif  // GGML_TENSOR_H
