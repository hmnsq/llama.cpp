// ET kernel entry-point parameter structs and tensor helpers.

#अगर_अघोषित GGML_TENSOR_H
#घोषणा GGML_TENSOR_H

#समावेश <stddef.h>
#समावेश <stdint.h>

#समावेश "ggml.h"

काष्ठा ggml_et_द्विआधारी_params {
    काष्ठा ggml_tensor src0;
    काष्ठा ggml_tensor src1;
    काष्ठा ggml_tensor dst;
};

// bias.data == NULL -> unfused MUL_MAT; otherwise dst = mat_mul(...) + bias.
काष्ठा ggml_et_mm_q8_params {
    काष्ठा ggml_tensor src0;
    काष्ठा ggml_tensor src1;
    काष्ठा ggml_tensor dst;
    काष्ठा ggml_tensor bias;
};

काष्ठा ggml_et_mul_mat_id_params {
    काष्ठा ggml_tensor src0;  // [K, M, n_expert]
    काष्ठा ggml_tensor src1;  // [K, n_expert_used, batch]
    काष्ठा ggml_tensor src2;  // [n_expert_used, batch] (I32 expert indices)
    काष्ठा ggml_tensor dst;   // [M, n_expert_used, batch, 1]
};

// ne[i] == 1 axes are skipped: their stride is unobservable.
अटल अंतरभूत पूर्णांक ggml_tensor_is_contiguous(स्थिर काष्ठा ggml_tensor * t, पूर्णांक type_आकार) {
    int64_t expected = type_आकार;
    क्रम (पूर्णांक i = 0; i < GGML_MAX_DIMS; i++) {
        अगर (t->ne[i] > 1 && (int64_t) t->nb[i] != expected) {
            वापस 0;
        }
        expected *= t->ne[i];
    }
    वापस 1;
}

#पूर्ण_अगर  // GGML_TENSOR_H
