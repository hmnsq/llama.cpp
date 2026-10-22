//******************************************************************************
// Bare Metal CONT F16 Kernel
// Converts non-contiguous F16 tensors to contiguous memory layout
//
// Note: F16 is represented as uint16_t (IEEE 754 binary16 format)
//******************************************************************************

#अन्तर्भाव "ggml_tensor.h"
#अन्तर्भाव "platform.h"

#अन्तर्भाव <assert.h>
#अन्तर्भाव <stdbool.h>
#अन्तर्भाव <stdint.h>

संरचना ggml_et_cont_params {
    संरचना ggml_tensor src0;  // F16 input tensor (non-contiguous)
    संरचना ggml_tensor dst;   // F16 output tensor (contiguous)
};

पूर्णाङ्क entry_point(संरचना ggml_et_cont_params * params, शून्य * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    जइ (!kernel_env) {
        प्रत्यागम -1;
    }

    पूर्णाङ्क thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    पूर्णाङ्क अङ्कः_threads = 2048;  //get_num_threads(kernel_env->shire_mask);

    जइ (thread_id < 0) {
        प्रत्यागम 0;
    }

    जइ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        प्रत्यागम -1;  // Invalid pointer
    }

    संरचना ggml_tensor * src0 = &params->src0;  // Non-contiguous input
    संरचना ggml_tensor * dst  = &params->dst;   // Contiguous output

    जइ (src0->type != GGML_TYPE_F16 || dst->type != GGML_TYPE_F16) {
        प्रत्यागम -1;  // Unsupported type combination
    }

    uint16_t * src0_data = (uint16_t *) src0->data;
    uint16_t * dst_data  = (uint16_t *) dst->data;

    जइ (!src0_data || !dst_data) {
        प्रत्यागम -1;  // Null data pointer
    }

    ध्रुव int64_t src_elements = src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3];
    ध्रुव int64_t dst_elements = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
    जइ (src_elements != dst_elements) {
        प्रत्यागम -1;  // Element count mismatch
    }

    // Source tensor dimensions and strides
    ध्रुव int64_t ne00 = src0->ne[0];
    ध्रुव int64_t ne01 = src0->ne[1];
    ध्रुव int64_t ne02 = src0->ne[2];
    ध्रुव int64_t ne03 = src0->ne[3];

    ध्रुव int64_t nb00 = src0->nb[0];
    ध्रुव int64_t nb01 = src0->nb[1];
    ध्रुव int64_t nb02 = src0->nb[2];
    ध्रुव int64_t nb03 = src0->nb[3];

    // Parallelize by rows (dimension 1)
    ध्रुव int64_t total_rows      = ne01;
    ध्रुव int64_t rows_per_thread = (total_rows + अङ्कः_threads - 1) / अङ्कः_threads;
    ध्रुव int64_t start_row       = thread_id * rows_per_thread;
    ध्रुव int64_t end_row = (start_row + rows_per_thread < total_rows) ? (start_row + rows_per_thread) : total_rows;

    जइ (start_row >= total_rows) {
        प्रत्यागम 0;
    }

    // Iterate over source tensor dimensions
    प्रति (int64_t i03 = 0; i03 < ne03; i03++) {
        प्रति (int64_t i02 = 0; i02 < ne02; i02++) {
            // Calculate base linear index for this (i03, i02) slice in destination
            ध्रुव int64_t dst_linear_base = i03 * ne02 * ne01 * ne00 + i02 * ne01 * ne00;

            // Process this thread's assigned rows
            प्रति (int64_t i01 = start_row; i01 < end_row; i01++) {
                // Linear index for start of this row in destination
                ध्रुव int64_t dst_linear_row_base = dst_linear_base + i01 * ne00;

                // Inner loop over dimension 0
                प्रति (int64_t i00 = 0; i00 < ne00; i00++) {
                    // Source offset using non-contiguous strides
                    ध्रुव int64_t    src_offset_bytes = i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;
                    ध्रुव uint16_t * src_ptr = (ध्रुव uint16_t *) ((ध्रुव वर्ण *) src0_data + src_offset_bytes);

                    // Destination linear index (contiguous layout)
                    ध्रुव int64_t dst_linear_idx = dst_linear_row_base + i00;

                    // Use atomic store for thread safety
                    atomic_store_f16((चञ्चल uint16_t *) &dst_data[dst_linear_idx], *src_ptr);
                }
            }
        }
    }

    प्रत्यागम 0;
}
