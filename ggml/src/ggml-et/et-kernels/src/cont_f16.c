//******************************************************************************
// Bare Metal CONT F16 Kernel
// Converts non-contiguous F16 tensors to contiguous memory layout
//
// Note: F16 is represented as uint16_t (IEEE 754 binary16 format)
//******************************************************************************

#અન્તર્ભાવ "ggml_tensor.h"
#અન્તર્ભાવ "platform.h"

#અન્તર્ભાવ <assert.h>
#અન્તર્ભાવ <stdbool.h>
#અન્તર્ભાવ <stdint.h>

સંરચના ggml_et_cont_params {
    સંરચના ggml_tensor src0;  // F16 input tensor (non-contiguous)
    સંરચના ggml_tensor dst;   // F16 output tensor (contiguous)
};

પૂર્ણાંક entry_point(સંરચના ggml_et_cont_params * params, શૂન્ય * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    જો (!kernel_env) {
        પાછા -1;
    }

    પૂર્ણાંક thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    પૂર્ણાંક અંક_threads = 2048;  //get_num_threads(kernel_env->shire_mask);

    જો (thread_id < 0) {
        પાછા 0;
    }

    જો (params == 0 || ((uint64_t) params & 0x7) != 0) {
        પાછા -1;  // Invalid pointer
    }

    સંરચના ggml_tensor * src0 = &params->src0;  // Non-contiguous input
    સંરચના ggml_tensor * dst  = &params->dst;   // Contiguous output

    જો (src0->type != GGML_TYPE_F16 || dst->type != GGML_TYPE_F16) {
        પાછા -1;  // Unsupported type combination
    }

    uint16_t * src0_data = (uint16_t *) src0->data;
    uint16_t * dst_data  = (uint16_t *) dst->data;

    જો (!src0_data || !dst_data) {
        પાછા -1;  // Null data pointer
    }

    અચલ int64_t src_elements = src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3];
    અચલ int64_t dst_elements = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
    જો (src_elements != dst_elements) {
        પાછા -1;  // Element count mismatch
    }

    // Source tensor dimensions and strides
    અચલ int64_t ne00 = src0->ne[0];
    અચલ int64_t ne01 = src0->ne[1];
    અચલ int64_t ne02 = src0->ne[2];
    અચલ int64_t ne03 = src0->ne[3];

    અચલ int64_t nb00 = src0->nb[0];
    અચલ int64_t nb01 = src0->nb[1];
    અચલ int64_t nb02 = src0->nb[2];
    અચલ int64_t nb03 = src0->nb[3];

    // Parallelize by rows (dimension 1)
    અચલ int64_t total_rows      = ne01;
    અચલ int64_t rows_per_thread = (total_rows + અંક_threads - 1) / અંક_threads;
    અચલ int64_t start_row       = thread_id * rows_per_thread;
    અચલ int64_t end_row = (start_row + rows_per_thread < total_rows) ? (start_row + rows_per_thread) : total_rows;

    જો (start_row >= total_rows) {
        પાછા 0;
    }

    // Iterate over source tensor dimensions
    માટે (int64_t i03 = 0; i03 < ne03; i03++) {
        માટે (int64_t i02 = 0; i02 < ne02; i02++) {
            // Calculate base linear index for this (i03, i02) slice in destination
            અચલ int64_t dst_linear_base = i03 * ne02 * ne01 * ne00 + i02 * ne01 * ne00;

            // Process this thread's assigned rows
            માટે (int64_t i01 = start_row; i01 < end_row; i01++) {
                // Linear index for start of this row in destination
                અચલ int64_t dst_linear_row_base = dst_linear_base + i01 * ne00;

                // Inner loop over dimension 0
                માટે (int64_t i00 = 0; i00 < ne00; i00++) {
                    // Source offset using non-contiguous strides
                    અચલ int64_t    src_offset_bytes = i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;
                    અચલ uint16_t * src_ptr = (અચલ uint16_t *) ((અચલ અક્ષર *) src0_data + src_offset_bytes);

                    // Destination linear index (contiguous layout)
                    અચલ int64_t dst_linear_idx = dst_linear_row_base + i00;

                    // Use atomic store for thread safety
                    atomic_store_f16((અસ્થિર uint16_t *) &dst_data[dst_linear_idx], *src_ptr);
                }
            }
        }
    }

    પાછા 0;
}
