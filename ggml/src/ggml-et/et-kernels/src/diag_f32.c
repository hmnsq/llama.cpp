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

पूर्णांक entry_point(संरचना ggml_et_diag_params * params, शून्य * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    जर (!kernel_env) {
        परत -1;
    }

    पूर्णांक thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    पूर्णांक अंक_threads = get_अंक_threads(kernel_env->shire_mask);

    जर (thread_id < 0) {
        परत 0;
    }

    जर (params == 0 || ((uint64_t) params & 0x7) != 0) {
        परत -1;
    }

    संरचना ggml_tensor * src0 = &params->src0;
    संरचना ggml_tensor * dst  = &params->dst;

    जर (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        परत -1;
    }

    दशांश * src0_data = (दशांश *) src0->data;
    दशांश * dst_data  = (दशांश *) dst->data;

    जर (!src0_data || !dst_data) {
        परत -1;
    }

    स्थिरांक int64_t ne0 = dst->ne[0];  // N (row width = column count)
    स्थिरांक int64_t ne1 = dst->ne[1];  // N (number of rows)
    स्थिरांक int64_t ne2 = dst->ne[2];
    स्थिरांक int64_t ne3 = dst->ne[3];

    स्थिरांक माप_प्रकार nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];
    स्थिरांक माप_प्रकार nb02 = src0->nb[2], nb03 = src0->nb[3];

    // Total rows across all batches — parallelize over these
    स्थिरांक int64_t total_rows = ne1 * ne2 * ne3;

    // Prepare zero vector for SIMD zeroing
    दशांश zero = 0.0f;
    __asm__ अस्थिर("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

    साठी (int64_t row = thread_id; row < total_rows; row += अंक_threads) {
        int64_t i1 = row % ne1;
        int64_t i2 = (row / ne1) % ne2;
        int64_t i3 = row / (ne1 * ne2);

        दशांश * dst_row = (दशांश *) ((अक्षर *) dst_data + i1 * nb1 + i2 * nb2 + i3 * nb3);

        // Zero the entire row with SIMD
        int64_t       i0      = 0;
        स्थिरांक int64_t vec_end = (ne0 / 8) * 8;
        साठी (; i0 < vec_end; i0 += 8) {
            __asm__ अस्थिर("fsw.ps f10, %[d]\n" : [d] "=m"(*(दशांश (*)[8]) & dst_row[i0])::"f10");
        }
        साठी (; i0 < ne0; i0++) {
            dst_row[i0] = 0.0f;
        }

        // Place the diagonal element: dst[i1][i1] = src0[i1]
        स्थिरांक दशांश * src_ptr = (स्थिरांक दशांश *) ((स्थिरांक अक्षर *) src0_data + i2 * nb02 + i3 * nb03);
        dst_row[i1]           = src_ptr[i1];
    }

    परत 0;
}
