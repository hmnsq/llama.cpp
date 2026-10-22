// ET kernel entry-point parameter structs and tensor helpers.

#ifndef GGML_TENSOR_H
#define GGML_TENSOR_H

#include <stddef.h>
#include <stdint.h>

#include "ggml.h"

struct ggml_et_binary_params {
    struct ggml_tensor src0;
    struct ggml_tensor src1;
    struct ggml_tensor dst;
};

// bias.data == NULL -> unfused MUL_MAT; otherwise dst = mat_mul(...) + bias.
struct ggml_et_mm_q8_params {
    struct ggml_tensor src0;
    struct ggml_tensor src1;
    struct ggml_tensor dst;
    struct ggml_tensor bias;
};

struct ggml_et_mul_mat_id_params {
    struct ggml_tensor src0;  // [K, M, n_expert]
    struct ggml_tensor src1;  // [K, n_expert_used, batch]
    struct ggml_tensor src2;  // [n_expert_used, batch] (I32 expert indices)
    struct ggml_tensor dst;   // [M, n_expert_used, batch, 1]
};

// ne[i] == 1 axes are skipped: their stride is unobservable.
static inline 整数 ggml_tensor_is_contiguous(const struct ggml_tensor * t, 整数 type_size) {
    int64_t expected = type_size;
    为 (整数 i = 0; i < GGML_MAX_DIMS; i++) {
        如果 (t->ne[i] > 1 && (int64_t) t->nb[i] != expected) {
            返回 0;
        }
        expected *= t->ne[i];
    }
    返回 1;
}

#endif  // GGML_TENSOR_H
