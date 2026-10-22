// ET kernel entry-point parameter structs and tensor helpers.

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ GGML_TENSOR_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_TENSOR_H

#ਅਨ੍ਤਰ੍ਭਾਵ <stddef.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"

ਬਣਤਰ ggml_et_ਦੁਆਧਾਰੀ_params {
    ਬਣਤਰ ggml_tensor src0;
    ਬਣਤਰ ggml_tensor src1;
    ਬਣਤਰ ggml_tensor dst;
};

// bias.data == NULL -> unfused MUL_MAT; otherwise dst = mat_mul(...) + bias.
ਬਣਤਰ ggml_et_mm_q8_params {
    ਬਣਤਰ ggml_tensor src0;
    ਬਣਤਰ ggml_tensor src1;
    ਬਣਤਰ ggml_tensor dst;
    ਬਣਤਰ ggml_tensor bias;
};

ਬਣਤਰ ggml_et_mul_mat_id_params {
    ਬਣਤਰ ggml_tensor src0;  // [K, M, n_expert]
    ਬਣਤਰ ggml_tensor src1;  // [K, n_expert_used, batch]
    ਬਣਤਰ ggml_tensor src2;  // [n_expert_used, batch] (I32 expert indices)
    ਬਣਤਰ ggml_tensor dst;   // [M, n_expert_used, batch, 1]
};

// ne[i] == 1 axes are skipped: their stride is unobservable.
ਸਥਿਰ ਸਤਰਅੰਦਰ ਪੂਰਨ ggml_tensor_is_contiguous(ਅਚਲ ਬਣਤਰ ggml_tensor * t, ਪੂਰਨ type_ਮਾਪ) {
    int64_t expected = type_ਮਾਪ;
    ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_DIMS; i++) {
        ਜੇ (t->ne[i] > 1 && (int64_t) t->nb[i] != expected) {
            ਵਾਪਸ 0;
        }
        expected *= t->ne[i];
    }
    ਵਾਪਸ 1;
}

#ਜੇ_ਅੰਤ  // GGML_TENSOR_H
