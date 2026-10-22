//******************************************************************************
// Diag F32 Kernel
// Creates a diagonal matrix from a 1D vector.
// dst[i][j] = (i == j) ? src0[i] : 0.0f
//
// src0: [N, 1, ne2, ne3]   (1D vector per batch)
// dst:  [N, N, ne2, ne3]   (diagonal matrix per batch)
//******************************************************************************

#समावेश "ggml_tensor.h"
#समावेश "platform.h"

#समावेश <stdint.h>

काष्ठा ggml_et_diag_params {
    काष्ठा ggml_tensor src0;  // F32 input vector
    काष्ठा ggml_tensor dst;   // F32 output diagonal matrix
};

पूर्णांक entry_point(काष्ठा ggml_et_diag_params * params, व्योम * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    अगर (!kernel_env) {
        वापस -1;
    }

    पूर्णांक thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    पूर्णांक अंक_threads = get_अंक_threads(kernel_env->shire_mask);

    अगर (thread_id < 0) {
        वापस 0;
    }

    अगर (params == 0 || ((uint64_t) params & 0x7) != 0) {
        वापस -1;
    }

    काष्ठा ggml_tensor * src0 = &params->src0;
    काष्ठा ggml_tensor * dst  = &params->dst;

    अगर (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        वापस -1;
    }

    भग्न * src0_तथ्य = (भग्न *) src0->तथ्य;
    भग्न * dst_तथ्य  = (भग्न *) dst->तथ्य;

    अगर (!src0_तथ्य || !dst_तथ्य) {
        वापस -1;
    }

    स्थिर int64_t ne0 = dst->ne[0];  // N (row width = column count)
    स्थिर int64_t ne1 = dst->ne[1];  // N (number of rows)
    स्थिर int64_t ne2 = dst->ne[2];
    स्थिर int64_t ne3 = dst->ne[3];

    स्थिर माप_प्रकार nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];
    स्थिर माप_प्रकार nb02 = src0->nb[2], nb03 = src0->nb[3];

    // Total rows across all batches — parallelize over these
    स्थिर int64_t योग_rows = ne1 * ne2 * ne3;

    // Prepare zero vector for SIMD zeroing
    भग्न zero = 0.0f;
    __asm__ अस्थिर("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

    क्रम (int64_t row = thread_id; row < योग_rows; row += अंक_threads) {
        int64_t i1 = row % ne1;
        int64_t i2 = (row / ne1) % ne2;
        int64_t i3 = row / (ne1 * ne2);

        भग्न * dst_row = (भग्न *) ((अक्षर *) dst_तथ्य + i1 * nb1 + i2 * nb2 + i3 * nb3);

        // Zero the entire row with SIMD
        int64_t       i0      = 0;
        स्थिर int64_t vec_end = (ne0 / 8) * 8;
        क्रम (; i0 < vec_end; i0 += 8) {
            __asm__ अस्थिर("fsw.ps f10, %[d]\n" : [d] "=m"(*(भग्न (*)[8]) & dst_row[i0])::"f10");
        }
        क्रम (; i0 < ne0; i0++) {
            dst_row[i0] = 0.0f;
        }

        // Place the diagonal element: dst[i1][i1] = src0[i1]
        स्थिर भग्न * src_ptr = (स्थिर भग्न *) ((स्थिर अक्षर *) src0_तथ्य + i2 * nb02 + i3 * nb03);
        dst_row[i1]           = src_ptr[i1];
    }

    वापस 0;
}
