//******************************************************************************
// Bare Metal CONT F16 Kernel
// Converts non-contiguous F16 tensors to contiguous memory layout
//
// Note: F16 is represented as uint16_t (IEEE 754 binary16 format)
//******************************************************************************

#அந்தர்பாவ "ggml_tensor.h"
#அந்தர்பாவ "platform.h"

#அந்தர்பாவ <assert.h>
#அந்தர்பாவ <stdbool.h>
#அந்தர்பாவ <stdint.h>

கட்டமைப்பு ggml_et_cont_params {
    கட்டமைப்பு ggml_tensor src0;  // F16 input tensor (non-contiguous)
    கட்டமைப்பு ggml_tensor dst;   // F16 output tensor (contiguous)
};

முழு entry_point(கட்டமைப்பு ggml_et_cont_params * params, வெறுமை * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    எனில் (!kernel_env) {
        திரும்பு -1;
    }

    முழு thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    முழு இலக்கம்_threads = 2048;  //get_num_threads(kernel_env->shire_mask);

    எனில் (thread_id < 0) {
        திரும்பு 0;
    }

    எனில் (params == 0 || ((uint64_t) params & 0x7) != 0) {
        திரும்பு -1;  // Invalid pointer
    }

    கட்டமைப்பு ggml_tensor * src0 = &params->src0;  // Non-contiguous input
    கட்டமைப்பு ggml_tensor * dst  = &params->dst;   // Contiguous output

    எனில் (src0->type != GGML_TYPE_F16 || dst->type != GGML_TYPE_F16) {
        திரும்பு -1;  // Unsupported type combination
    }

    uint16_t * src0_data = (uint16_t *) src0->data;
    uint16_t * dst_data  = (uint16_t *) dst->data;

    எனில் (!src0_data || !dst_data) {
        திரும்பு -1;  // Null data pointer
    }

    மாறிலி int64_t src_elements = src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3];
    மாறிலி int64_t dst_elements = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
    எனில் (src_elements != dst_elements) {
        திரும்பு -1;  // Element count mismatch
    }

    // Source tensor dimensions and strides
    மாறிலி int64_t ne00 = src0->ne[0];
    மாறிலி int64_t ne01 = src0->ne[1];
    மாறிலி int64_t ne02 = src0->ne[2];
    மாறிலி int64_t ne03 = src0->ne[3];

    மாறிலி int64_t nb00 = src0->nb[0];
    மாறிலி int64_t nb01 = src0->nb[1];
    மாறிலி int64_t nb02 = src0->nb[2];
    மாறிலி int64_t nb03 = src0->nb[3];

    // Parallelize by rows (dimension 1)
    மாறிலி int64_t total_rows      = ne01;
    மாறிலி int64_t rows_per_thread = (total_rows + இலக்கம்_threads - 1) / இலக்கம்_threads;
    மாறிலி int64_t start_row       = thread_id * rows_per_thread;
    மாறிலி int64_t end_row = (start_row + rows_per_thread < total_rows) ? (start_row + rows_per_thread) : total_rows;

    எனில் (start_row >= total_rows) {
        திரும்பு 0;
    }

    // Iterate over source tensor dimensions
    க்கு (int64_t i03 = 0; i03 < ne03; i03++) {
        க்கு (int64_t i02 = 0; i02 < ne02; i02++) {
            // Calculate base linear index for this (i03, i02) slice in destination
            மாறிலி int64_t dst_linear_base = i03 * ne02 * ne01 * ne00 + i02 * ne01 * ne00;

            // Process this thread's assigned rows
            க்கு (int64_t i01 = start_row; i01 < end_row; i01++) {
                // Linear index for start of this row in destination
                மாறிலி int64_t dst_linear_row_base = dst_linear_base + i01 * ne00;

                // Inner loop over dimension 0
                க்கு (int64_t i00 = 0; i00 < ne00; i00++) {
                    // Source offset using non-contiguous strides
                    மாறிலி int64_t    src_offset_bytes = i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;
                    மாறிலி uint16_t * src_ptr = (மாறிலி uint16_t *) ((மாறிலி எழுத்து *) src0_data + src_offset_bytes);

                    // Destination linear index (contiguous layout)
                    மாறிலி int64_t dst_linear_idx = dst_linear_row_base + i00;

                    // Use atomic store for thread safety
                    atomic_store_f16((நிலையற்ற uint16_t *) &dst_data[dst_linear_idx], *src_ptr);
                }
            }
        }
    }

    திரும்பு 0;
}
