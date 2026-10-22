//******************************************************************************
// Bare Metal CONT F16 Kernel
// Converts non-contiguous F16 tensors to contiguous memory layout
//
// Note: F16 is represented as uint16_t (IEEE 754 binary16 format)
//******************************************************************************

#ಅನ್ತರ್ಭಾವ "ggml_tensor.h"
#ಅನ್ತರ್ಭಾವ "platform.h"

#ಅನ್ತರ್ಭಾವ <assert.h>
#ಅನ್ತರ್ಭಾವ <stdbool.h>
#ಅನ್ತರ್ಭಾವ <stdint.h>

ಸಂರಚನಾ ggml_et_cont_params {
    ಸಂರಚನಾ ggml_tensor src0;  // F16 input tensor (non-contiguous)
    ಸಂರಚನಾ ggml_tensor dst;   // F16 output tensor (contiguous)
};

ಪೂರ್ಣಾಙ್ಕ entry_point(ಸಂರಚನಾ ggml_et_cont_params * params, ಶೂನ್ಯ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ಯದಿ (!kernel_env) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಪೂರ್ಣಾಙ್ಕ thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_threads = 2048;  //get_num_threads(kernel_env->shire_mask);

    ಯದಿ (thread_id < 0) {
        ಪ್ರತ್ಯಾಗಮ 0;
    }

    ಯದಿ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ಪ್ರತ್ಯಾಗಮ -1;  // Invalid pointer
    }

    ಸಂರಚನಾ ggml_tensor * src0 = &params->src0;  // Non-contiguous input
    ಸಂರಚನಾ ggml_tensor * dst  = &params->dst;   // Contiguous output

    ಯದಿ (src0->type != GGML_TYPE_F16 || dst->type != GGML_TYPE_F16) {
        ಪ್ರತ್ಯಾಗಮ -1;  // Unsupported type combination
    }

    uint16_t * src0_data = (uint16_t *) src0->data;
    uint16_t * dst_data  = (uint16_t *) dst->data;

    ಯದಿ (!src0_data || !dst_data) {
        ಪ್ರತ್ಯಾಗಮ -1;  // Null data pointer
    }

    ಧ್ರುವ int64_t src_elements = src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3];
    ಧ್ರುವ int64_t dst_elements = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
    ಯದಿ (src_elements != dst_elements) {
        ಪ್ರತ್ಯಾಗಮ -1;  // Element count mismatch
    }

    // Source tensor dimensions and strides
    ಧ್ರುವ int64_t ne00 = src0->ne[0];
    ಧ್ರುವ int64_t ne01 = src0->ne[1];
    ಧ್ರುವ int64_t ne02 = src0->ne[2];
    ಧ್ರುವ int64_t ne03 = src0->ne[3];

    ಧ್ರುವ int64_t nb00 = src0->nb[0];
    ಧ್ರುವ int64_t nb01 = src0->nb[1];
    ಧ್ರುವ int64_t nb02 = src0->nb[2];
    ಧ್ರುವ int64_t nb03 = src0->nb[3];

    // Parallelize by rows (dimension 1)
    ಧ್ರುವ int64_t total_rows      = ne01;
    ಧ್ರುವ int64_t rows_per_thread = (total_rows + ಅಂಕೆ_threads - 1) / ಅಂಕೆ_threads;
    ಧ್ರುವ int64_t start_row       = thread_id * rows_per_thread;
    ಧ್ರುವ int64_t end_row = (start_row + rows_per_thread < total_rows) ? (start_row + rows_per_thread) : total_rows;

    ಯದಿ (start_row >= total_rows) {
        ಪ್ರತ್ಯಾಗಮ 0;
    }

    // Iterate over source tensor dimensions
    ಪ್ರತಿ (int64_t i03 = 0; i03 < ne03; i03++) {
        ಪ್ರತಿ (int64_t i02 = 0; i02 < ne02; i02++) {
            // Calculate base linear index for this (i03, i02) slice in destination
            ಧ್ರುವ int64_t dst_linear_base = i03 * ne02 * ne01 * ne00 + i02 * ne01 * ne00;

            // Process this thread's assigned rows
            ಪ್ರತಿ (int64_t i01 = start_row; i01 < end_row; i01++) {
                // Linear index for start of this row in destination
                ಧ್ರುವ int64_t dst_linear_row_base = dst_linear_base + i01 * ne00;

                // Inner loop over dimension 0
                ಪ್ರತಿ (int64_t i00 = 0; i00 < ne00; i00++) {
                    // Source offset using non-contiguous strides
                    ಧ್ರುವ int64_t    src_offset_bytes = i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;
                    ಧ್ರುವ uint16_t * src_ptr = (ಧ್ರುವ uint16_t *) ((ಧ್ರುವ ವರ್ಣ *) src0_data + src_offset_bytes);

                    // Destination linear index (contiguous layout)
                    ಧ್ರುವ int64_t dst_linear_idx = dst_linear_row_base + i00;

                    // Use atomic store for thread safety
                    atomic_store_f16((ಚಞ್ಚಲ uint16_t *) &dst_data[dst_linear_idx], *src_ptr);
                }
            }
        }
    }

    ಪ್ರತ್ಯಾಗಮ 0;
}
