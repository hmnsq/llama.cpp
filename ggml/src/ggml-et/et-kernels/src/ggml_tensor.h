// ET kernel entry-point parameter structs and tensor helpers.

#ifndef GGML_TENSOR_H
#define GGML_TENSOR_H

#ଅନ୍ତର୍ଭାଵ <stddef.h>
#ଅନ୍ତର୍ଭାଵ <stdint.h>

#ଅନ୍ତର୍ଭାଵ "ggml.h"

ସଂରଚନା ggml_et_ଦ୍ୱିମିକ_params {
    ସଂରଚନା ggml_tensor src0;
    ସଂରଚନା ggml_tensor src1;
    ସଂରଚନା ggml_tensor dst;
};

// bias.data == NULL -> unfused MUL_MAT; otherwise dst = mat_mul(...) + bias.
ସଂରଚନା ggml_et_mm_q8_params {
    ସଂରଚନା ggml_tensor src0;
    ସଂରଚନା ggml_tensor src1;
    ସଂରଚନା ggml_tensor dst;
    ସଂରଚନା ggml_tensor bias;
};

ସଂରଚନା ggml_et_mul_mat_id_params {
    ସଂରଚନା ggml_tensor src0;  // [K, M, n_expert]
    ସଂରଚନା ggml_tensor src1;  // [K, n_expert_used, batch]
    ସଂରଚନା ggml_tensor src2;  // [n_expert_used, batch] (I32 expert indices)
    ସଂରଚନା ggml_tensor dst;   // [M, n_expert_used, batch, 1]
};

// ne[i] == 1 axes are skipped: their stride is unobservable.
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_tensor_is_contiguous(ଅଚଳ ସଂରଚନା ggml_tensor * t, ପୂର୍ଣ୍ଣସଂଖ୍ୟା type_size) {
    int64_t expected = type_size;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < GGML_MAX_DIMS; i++) {
        ଯଦି (t->ne[i] > 1 && (int64_t) t->nb[i] != expected) {
            ଫେରନ୍ତୁ 0;
        }
        expected *= t->ne[i];
    }
    ଫେରନ୍ତୁ 1;
}

#endif  // GGML_TENSOR_H
