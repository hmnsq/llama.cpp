// ET kernel entry-point parameter structs and tensor helpers.

#ifndef GGML_TENSOR_H
#define GGML_TENSOR_H

#شامل <stddef.h>
#شامل <stdint.h>

#شامل "ggml.h"

ساختار ggml_et_دودویی_params {
    ساختار ggml_tensor src0;
    ساختار ggml_tensor src1;
    ساختار ggml_tensor dst;
};

// bias.data == NULL -> unfused MUL_MAT; otherwise dst = mat_mul(...) + bias.
ساختار ggml_et_mm_q8_params {
    ساختار ggml_tensor src0;
    ساختار ggml_tensor src1;
    ساختار ggml_tensor dst;
    ساختار ggml_tensor bias;
};

ساختار ggml_et_mul_mat_id_params {
    ساختار ggml_tensor src0;  // [K, M, n_expert]
    ساختار ggml_tensor src1;  // [K, n_expert_used, batch]
    ساختار ggml_tensor src2;  // [n_expert_used, batch] (I32 expert indices)
    ساختار ggml_tensor dst;   // [M, n_expert_used, batch, 1]
};

// ne[i] == 1 axes are skipped: their stride is unobservable.
ایستا درونخطی صحیح ggml_tensor_is_contiguous(ثابت ساختار ggml_tensor * t, صحیح type_size) {
    int64_t expected = type_size;
    برای (صحیح i = 0; i < GGML_MAX_DIMS; i++) {
        اگر (t->ne[i] > 1 && (int64_t) t->nb[i] != expected) {
            بازگشت 0;
        }
        expected *= t->ne[i];
    }
    بازگشت 1;
}

#endif  // GGML_TENSOR_H
