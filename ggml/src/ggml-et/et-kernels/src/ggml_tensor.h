// ET kernel entry-point parameter structs and tensor helpers.

#నిర్వచితం_కాకపోతే GGML_TENSOR_H
#నిర్వచించు GGML_TENSOR_H

#అన్తర్భావ <stddef.h>
#అన్తర్భావ <stdint.h>

#అన్తర్భావ "ggml.h"

నిర్మాణం ggml_et_ద్విమాన_params {
    నిర్మాణం ggml_tensor src0;
    నిర్మాణం ggml_tensor src1;
    నిర్మాణం ggml_tensor dst;
};

// bias.data == NULL -> unfused MUL_MAT; otherwise dst = mat_mul(...) + bias.
నిర్మాణం ggml_et_mm_q8_params {
    నిర్మాణం ggml_tensor src0;
    నిర్మాణం ggml_tensor src1;
    నిర్మాణం ggml_tensor dst;
    నిర్మాణం ggml_tensor bias;
};

నిర్మాణం ggml_et_mul_mat_id_params {
    నిర్మాణం ggml_tensor src0;  // [K, M, n_expert]
    నిర్మాణం ggml_tensor src1;  // [K, n_expert_used, batch]
    నిర్మాణం ggml_tensor src2;  // [n_expert_used, batch] (I32 expert indices)
    నిర్మాణం ggml_tensor dst;   // [M, n_expert_used, batch, 1]
};

// ne[i] == 1 axes are skipped: their stride is unobservable.
స్థిర అంతఃవరుస పూర్ణాంకం ggml_tensor_is_contiguous(స్థిరరాశి నిర్మాణం ggml_tensor * t, పూర్ణాంకం type_కొలత) {
    int64_t expected = type_కొలత;
    కొరకు (పూర్ణాంకం i = 0; i < GGML_MAX_DIMS; i++) {
        ఒకవేళ (t->ne[i] > 1 && (int64_t) t->nb[i] != expected) {
            తిరిగి 0;
        }
        expected *= t->ne[i];
    }
    తిరిగి 1;
}

#అయితే_ముగింపు  // GGML_TENSOR_H
