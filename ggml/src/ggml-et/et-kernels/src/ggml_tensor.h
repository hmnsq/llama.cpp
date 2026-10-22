// ET kernel entry-point parameter structs and tensor helpers.

#ifndef GGML_TENSOR_H
#define GGML_TENSOR_H

#অন্তর্ভাব <stddef.h>
#অন্তর্ভাব <stdint.h>

#অন্তর্ভাব "ggml.h"

কাঠামো ggml_et_দ্বিমিক_params {
    কাঠামো ggml_tensor src0;
    কাঠামো ggml_tensor src1;
    কাঠামো ggml_tensor dst;
};

// bias.data == NULL -> unfused MUL_MAT; otherwise dst = mat_mul(...) + bias.
কাঠামো ggml_et_mm_q8_params {
    কাঠামো ggml_tensor src0;
    কাঠামো ggml_tensor src1;
    কাঠামো ggml_tensor dst;
    কাঠামো ggml_tensor bias;
};

কাঠামো ggml_et_mul_mat_id_params {
    কাঠামো ggml_tensor src0;  // [K, M, n_expert]
    কাঠামো ggml_tensor src1;  // [K, n_expert_used, batch]
    কাঠামো ggml_tensor src2;  // [n_expert_used, batch] (I32 expert indices)
    কাঠামো ggml_tensor dst;   // [M, n_expert_used, batch, 1]
};

// ne[i] == 1 axes are skipped: their stride is unobservable.
স্থির অন্তঃরেখ পূর্ণসংখ্যা ggml_tensor_is_contiguous(ধ্রুবক কাঠামো ggml_tensor * t, পূর্ণসংখ্যা type_আকার) {
    int64_t expected = type_আকার;
    জন্য (পূর্ণসংখ্যা i = 0; i < GGML_MAX_DIMS; i++) {
        যদি (t->ne[i] > 1 && (int64_t) t->nb[i] != expected) {
            ফেরত 0;
        }
        expected *= t->ne[i];
    }
    ফেরত 1;
}

#endif  // GGML_TENSOR_H
