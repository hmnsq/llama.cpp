// ET kernel entry-point parameter structs and tensor helpers.

#که_نه_تعریف_شوی GGML_TENSOR_H
#تعریف_کړه GGML_TENSOR_H

#ضمن <stddef.h>
#ضمن <stdint.h>

#ضمن "ggml.h"

بنية ggml_et_دوګونی_params {
    بنية ggml_tensor src0;
    بنية ggml_tensor src1;
    بنية ggml_tensor dst;
};

// bias.data == NULL -> unfused MUL_MAT; otherwise dst = mat_mul(...) + bias.
بنية ggml_et_mm_q8_params {
    بنية ggml_tensor src0;
    بنية ggml_tensor src1;
    بنية ggml_tensor dst;
    بنية ggml_tensor bias;
};

بنية ggml_et_mul_mat_id_params {
    بنية ggml_tensor src0;  // [K, M, n_expert]
    بنية ggml_tensor src1;  // [K, n_expert_used, batch]
    بنية ggml_tensor src2;  // [n_expert_used, batch] (I32 expert indices)
    بنية ggml_tensor dst;   // [M, n_expert_used, batch, 1]
};

// ne[i] == 1 axes are skipped: their stride is unobservable.
ساكن مضمن صحيح ggml_tensor_is_contiguous(ثابت بنية ggml_tensor * t, صحيح type_اندازه) {
    int64_t expected = type_اندازه;
    لكل (صحيح i = 0; i < GGML_MAX_DIMS; i++) {
        إذا (t->ne[i] > 1 && (int64_t) t->nb[i] != expected) {
            أرجع 0;
        }
        expected *= t->ne[i];
    }
    أرجع 1;
}

#د_که_پای  // GGML_TENSOR_H
