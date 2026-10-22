// ET kernel entry-point parameter structs and tensor helpers.

#ifndef GGML_TENSOR_H
#define GGML_TENSOR_H

#ಅನ್ತರ್ಭಾವ <stddef.h>
#ಅನ್ತರ್ಭಾವ <stdint.h>

#ಅನ್ತರ್ಭಾವ "ggml.h"

ಸಂರಚನಾ ggml_et_ದ್ವಿಮಾನ_params {
    ಸಂರಚನಾ ggml_tensor src0;
    ಸಂರಚನಾ ggml_tensor src1;
    ಸಂರಚನಾ ggml_tensor dst;
};

// bias.data == NULL -> unfused MUL_MAT; otherwise dst = mat_mul(...) + bias.
ಸಂರಚನಾ ggml_et_mm_q8_params {
    ಸಂರಚನಾ ggml_tensor src0;
    ಸಂರಚನಾ ggml_tensor src1;
    ಸಂರಚನಾ ggml_tensor dst;
    ಸಂರಚನಾ ggml_tensor bias;
};

ಸಂರಚನಾ ggml_et_mul_mat_id_params {
    ಸಂರಚನಾ ggml_tensor src0;  // [K, M, n_expert]
    ಸಂರಚನಾ ggml_tensor src1;  // [K, n_expert_used, batch]
    ಸಂರಚನಾ ggml_tensor src2;  // [n_expert_used, batch] (I32 expert indices)
    ಸಂರಚನಾ ggml_tensor dst;   // [M, n_expert_used, batch, 1]
};

// ne[i] == 1 axes are skipped: their stride is unobservable.
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಪೂರ್ಣಾಙ್ಕ ggml_tensor_is_contiguous(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * t, ಪೂರ್ಣಾಙ್ಕ type_size) {
    int64_t expected = type_size;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < GGML_MAX_DIMS; i++) {
        ಯದಿ (t->ne[i] > 1 && (int64_t) t->nb[i] != expected) {
            ಪ್ರತ್ಯಾಗಮ 0;
        }
        expected *= t->ne[i];
    }
    ಪ್ರತ್ಯಾಗಮ 1;
}

#endif  // GGML_TENSOR_H
