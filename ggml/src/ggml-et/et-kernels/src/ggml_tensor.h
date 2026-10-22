// ET kernel entry-point parameter structs and tensor helpers.

#ifndef GGML_TENSOR_H
#define GGML_TENSOR_H

#include <stddef.h>
#include <stdint.h>

#include "ggml.h"

sa_nracanaa ggml_et_binary_params {
    sa_nracanaa ggml_tensor src0;
    sa_nracanaa ggml_tensor src1;
    sa_nracanaa ggml_tensor dst;
};

// bias.data == NULL -> unfused MUL_MAT; otherwise dst = mat_mul(...) + bias.
sa_nracanaa ggml_et_mm_q8_params {
    sa_nracanaa ggml_tensor src0;
    sa_nracanaa ggml_tensor src1;
    sa_nracanaa ggml_tensor dst;
    sa_nracanaa ggml_tensor bias;
};

sa_nracanaa ggml_et_mul_mat_id_params {
    sa_nracanaa ggml_tensor src0;  // [K, M, n_expert]
    sa_nracanaa ggml_tensor src1;  // [K, n_expert_used, batch]
    sa_nracanaa ggml_tensor src2;  // [n_expert_used, batch] (I32 expert indices)
    sa_nracanaa ggml_tensor dst;   // [M, n_expert_used, batch, 1]
};

// ne[i] == 1 axes are skipped: their stride is unobservable.
static anta_hstha puur_naa_nka ggml_tensor_is_contiguous(const sa_nracanaa ggml_tensor * t, puur_naa_nka type_size) {
    int64_t expected = type_size;
    for (puur_naa_nka i = 0; i < GGML_MAX_DIMS; i++) {
        ᨀᨘ (t->ne[i] > 1 && (int64_t) t->nb[i] != expected) {
            ᨀᨘᨒ 0;
        }
        expected *= t->ne[i];
    }
    ᨀᨘᨒ 1;
}

#endif  // GGML_TENSOR_H
