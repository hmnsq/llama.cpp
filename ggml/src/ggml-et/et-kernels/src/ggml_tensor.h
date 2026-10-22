// ET kernel entry-point parameter structs and tensor helpers.

#اگر_غیر_تعریف_شدہ GGML_TENSOR_H
#تعریف_کرو GGML_TENSOR_H

#شامل <stddef.h>
#شامل <stdint.h>

#شامل "ggml.h"

ساخت ggml_et_ثنائی_params {
    ساخت ggml_tensor src0;
    ساخت ggml_tensor src1;
    ساخت ggml_tensor dst;
};

// bias.data == NULL -> unfused MUL_MAT; otherwise dst = mat_mul(...) + bias.
ساخت ggml_et_mm_q8_params {
    ساخت ggml_tensor src0;
    ساخت ggml_tensor src1;
    ساخت ggml_tensor dst;
    ساخت ggml_tensor bias;
};

ساخت ggml_et_mul_mat_id_params {
    ساخت ggml_tensor src0;  // [K, M, n_expert]
    ساخت ggml_tensor src1;  // [K, n_expert_used, batch]
    ساخت ggml_tensor src2;  // [n_expert_used, batch] (I32 expert indices)
    ساخت ggml_tensor dst;   // [M, n_expert_used, batch, 1]
};

// ne[i] == 1 axes are skipped: their stride is unobservable.
ساکن اندرونی صحیح ggml_tensor_is_contiguous(ثابت ساخت ggml_tensor * t, صحیح type_سائز) {
    int64_t expected = type_سائز;
    برائے (صحیح i = 0; i < GGML_MAX_DIMS; i++) {
        اگر (t->ne[i] > 1 && (int64_t) t->nb[i] != expected) {
            واپس 0;
        }
        expected *= t->ne[i];
    }
    واپس 1;
}

#اگر_ختم  // GGML_TENSOR_H
