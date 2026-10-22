// ET kernel entry-point parameter structs and tensor helpers.

#ifndef GGML_TENSOR_H
#define GGML_TENSOR_H

#अन्तर्भाव <stddef.h>
#अन्तर्भाव <stdint.h>

#अन्तर्भाव "ggml.h"

संरचना ggml_et_द्विमान_params {
    संरचना ggml_tensor src0;
    संरचना ggml_tensor src1;
    संरचना ggml_tensor dst;
};

// bias.data == NULL -> unfused MUL_MAT; otherwise dst = mat_mul(...) + bias.
संरचना ggml_et_mm_q8_params {
    संरचना ggml_tensor src0;
    संरचना ggml_tensor src1;
    संरचना ggml_tensor dst;
    संरचना ggml_tensor bias;
};

संरचना ggml_et_mul_mat_id_params {
    संरचना ggml_tensor src0;  // [K, M, n_expert]
    संरचना ggml_tensor src1;  // [K, n_expert_used, batch]
    संरचना ggml_tensor src2;  // [n_expert_used, batch] (I32 expert indices)
    संरचना ggml_tensor dst;   // [M, n_expert_used, batch, 1]
};

// ne[i] == 1 axes are skipped: their stride is unobservable.
स्थिर ओळीत पूर्णांक ggml_tensor_is_contiguous(स्थिरांक संरचना ggml_tensor * t, पूर्णांक type_size) {
    int64_t expected = type_size;
    साठी (पूर्णांक i = 0; i < GGML_MAX_DIMS; i++) {
        जर (t->ne[i] > 1 && (int64_t) t->nb[i] != expected) {
            परत 0;
        }
        expected *= t->ne[i];
    }
    परत 1;
}

#endif  // GGML_TENSOR_H
