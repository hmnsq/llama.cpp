// ET kernel entry-point parameter structs and tensor helpers.

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ GGML_TENSOR_H
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_TENSOR_H

#ಅನ್ತರ್ಭಾವ <stddef.h>
#ಅನ್ತರ್ಭಾವ <stdint.h>

#ಅನ್ತರ್ಭಾವ "ggml.h"

ರಚನೆ ggml_et_ದ್ವಿಮಾನ_params {
    ರಚನೆ ggml_tensor src0;
    ರಚನೆ ggml_tensor src1;
    ರಚನೆ ggml_tensor dst;
};

// bias.data == NULL -> unfused MUL_MAT; otherwise dst = mat_mul(...) + bias.
ರಚನೆ ggml_et_mm_q8_params {
    ರಚನೆ ggml_tensor src0;
    ರಚನೆ ggml_tensor src1;
    ರಚನೆ ggml_tensor dst;
    ರಚನೆ ggml_tensor bias;
};

ರಚನೆ ggml_et_mul_mat_id_params {
    ರಚನೆ ggml_tensor src0;  // [K, M, n_expert]
    ರಚನೆ ggml_tensor src1;  // [K, n_expert_used, batch]
    ರಚನೆ ggml_tensor src2;  // [n_expert_used, batch] (I32 expert indices)
    ರಚನೆ ggml_tensor dst;   // [M, n_expert_used, batch, 1]
};

// ne[i] == 1 axes are skipped: their stride is unobservable.
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಪೂರ್ಣಾಂಕ ggml_tensor_is_contiguous(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * t, ಪೂರ್ಣಾಂಕ type_ಗಾತ್ರ) {
    int64_t expected = type_ಗಾತ್ರ;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_DIMS; i++) {
        ಆದರೆ (t->ne[i] > 1 && (int64_t) t->nb[i] != expected) {
            ಹಿಂತಿರುಗಿ 0;
        }
        expected *= t->ne[i];
    }
    ಹಿಂತಿರುಗಿ 1;
}

#ಆದರೆ_ಅಂತ್ಯ  // GGML_TENSOR_H
