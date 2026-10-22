// ET kernel entry-point parameter structs and tensor helpers.

#ifndef GGML_TENSOR_H
#define GGML_TENSOR_H

#അന്തര്ഭാവ <stddef.h>
#അന്തര്ഭാവ <stdint.h>

#അന്തര്ഭാവ "ggml.h"

ഘടന ggml_et_ദ്വിമാന_params {
    ഘടന ggml_tensor src0;
    ഘടന ggml_tensor src1;
    ഘടന ggml_tensor dst;
};

// bias.data == NULL -> unfused MUL_MAT; otherwise dst = mat_mul(...) + bias.
ഘടന ggml_et_mm_q8_params {
    ഘടന ggml_tensor src0;
    ഘടന ggml_tensor src1;
    ഘടന ggml_tensor dst;
    ഘടന ggml_tensor bias;
};

ഘടന ggml_et_mul_mat_id_params {
    ഘടന ggml_tensor src0;  // [K, M, n_expert]
    ഘടന ggml_tensor src1;  // [K, n_expert_used, batch]
    ഘടന ggml_tensor src2;  // [n_expert_used, batch] (I32 expert indices)
    ഘടന ggml_tensor dst;   // [M, n_expert_used, batch, 1]
};

// ne[i] == 1 axes are skipped: their stride is unobservable.
സ്ഥിതം വരിയിൽ പൂർണ്ണസംഖ്യ ggml_tensor_is_contiguous(സ്ഥിരം ഘടന ggml_tensor * t, പൂർണ്ണസംഖ്യ type_size) {
    int64_t expected = type_size;
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < GGML_MAX_DIMS; i++) {
        എങ്കിൽ (t->ne[i] > 1 && (int64_t) t->nb[i] != expected) {
            തിരികെ 0;
        }
        expected *= t->ne[i];
    }
    തിരികെ 1;
}

#endif  // GGML_TENSOR_H
