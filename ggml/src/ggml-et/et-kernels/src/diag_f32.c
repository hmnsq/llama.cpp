//******************************************************************************
// Diag F32 Kernel
// Creates a diagonal matrix from a 1D vector.
// dst[i][j] = (i == j) ? src0[i] : 0.0f
//
// src0: [N, 1, ne2, ne3]   (1D vector per batch)
// dst:  [N, N, ne2, ne3]   (diagonal matrix per batch)
//******************************************************************************

#अन्तर्भाव "ggml_tensor.h"
#अन्तर्भाव "platform.h"

#अन्तर्भाव <stdint.h>

संरचना ggml_et_diag_params {
    संरचना ggml_tensor src0;  // F32 input vector
    संरचना ggml_tensor dst;   // F32 output diagonal matrix
};

पूर्णाङ्क प्रविष्टिः_point(संरचना ggml_et_diag_params * params, शून्य * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    यदि (!kernel_env) {
        प्रत्यागम -1;
    }

    पूर्णाङ्क thread_id   = get_relative_thread_id(kernel_env->shire_आवरणम्);
    पूर्णाङ्क अङ्कः_तन्तवः = get_अङ्कः_तन्तवः(kernel_env->shire_आवरणम्);

    यदि (thread_id < 0) {
        प्रत्यागम 0;
    }

    यदि (params == 0 || ((uint64_t) params & 0x7) != 0) {
        प्रत्यागम -1;
    }

    संरचना ggml_tensor * src0 = &params->src0;
    संरचना ggml_tensor * dst  = &params->dst;

    यदि (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        प्रत्यागम -1;
    }

    भिन्न * src0_दत्त = (भिन्न *) src0->दत्त;
    भिन्न * dst_दत्त  = (भिन्न *) dst->दत्त;

    यदि (!src0_दत्त || !dst_दत्त) {
        प्रत्यागम -1;
    }

    ध्रुव int64_t ne0 = dst->ne[0];  // N (row width = column count)
    ध्रुव int64_t ne1 = dst->ne[1];  // N (number of rows)
    ध्रुव int64_t ne2 = dst->ne[2];
    ध्रुव int64_t ne3 = dst->ne[3];

    ध्रुव size_t nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];
    ध्रुव size_t nb02 = src0->nb[2], nb03 = src0->nb[3];

    // Total rows across all batches — parallelize over these
    ध्रुव int64_t योग_पङ्क्तिकाः = ne1 * ne2 * ne3;

    // Prepare zero vector for SIMD zeroing
    भिन्न zero = 0.0f;
    __asm__ चञ्चल("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

    प्रति (int64_t row = thread_id; row < योग_पङ्क्तिकाः; row += अङ्कः_तन्तवः) {
        int64_t i1 = row % ne1;
        int64_t i2 = (row / ne1) % ne2;
        int64_t i3 = row / (ne1 * ne2);

        भिन्न * dst_row = (भिन्न *) ((वर्ण *) dst_दत्त + i1 * nb1 + i2 * nb2 + i3 * nb3);

        // Zero the entire row with SIMD
        int64_t       i0      = 0;
        ध्रुव int64_t vec_अन्तः = (ne0 / 8) * 8;
        प्रति (; i0 < vec_अन्तः; i0 += 8) {
            __asm__ चञ्चल("fsw.ps f10, %[d]\n" : [d] "=m"(*(भिन्न (*)[8]) & dst_row[i0])::"f10");
        }
        प्रति (; i0 < ne0; i0++) {
            dst_row[i0] = 0.0f;
        }

        // Place the diagonal element: dst[i1][i1] = src0[i1]
        ध्रुव भिन्न * src_ptr = (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src0_दत्त + i2 * nb02 + i3 * nb03);
        dst_row[i1]           = src_ptr[i1];
    }

    प्रत्यागम 0;
}
