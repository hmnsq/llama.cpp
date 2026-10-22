//******************************************************************************
// Bare Metal CONT F16 Kernel
// Converts non-contiguous F16 tensors to contiguous memory layout
//
// Note: F16 is represented as uint16_t (IEEE 754 binary16 format)
//******************************************************************************

#include "ggml_tensor.h"
#include "platform.h"

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>

sa_nracanaa ggml_et_cont_params {
    sa_nracanaa ggml_tensor src0;  // F16 input tensor (non-contiguous)
    sa_nracanaa ggml_tensor dst;   // F16 output tensor (contiguous)
};

ចំនួនគត់ entry_point(sa_nracanaa ggml_et_cont_params * params, void * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ប្រសិនបើ (!kernel_env) {
        ត្រឡប់ -1;
    }

    ចំនួនគត់ thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    ចំនួនគត់ num_threads = 2048;  //get_num_threads(kernel_env->shire_mask);

    ប្រសិនបើ (thread_id < 0) {
        ត្រឡប់ 0;
    }

    ប្រសិនបើ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ត្រឡប់ -1;  // Invalid pointer
    }

    sa_nracanaa ggml_tensor * src0 = &params->src0;  // Non-contiguous input
    sa_nracanaa ggml_tensor * dst  = &params->dst;   // Contiguous output

    ប្រសិនបើ (src0->type != GGML_TYPE_F16 || dst->type != GGML_TYPE_F16) {
        ត្រឡប់ -1;  // Unsupported type combination
    }

    uint16_t * src0_data = (uint16_t *) src0->data;
    uint16_t * dst_data  = (uint16_t *) dst->data;

    ប្រសិនបើ (!src0_data || !dst_data) {
        ត្រឡប់ -1;  // Null data pointer
    }

    const int64_t src_elements = src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3];
    const int64_t dst_elements = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
    ប្រសិនបើ (src_elements != dst_elements) {
        ត្រឡប់ -1;  // Element count mismatch
    }

    // Source tensor dimensions and strides
    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const int64_t nb00 = src0->nb[0];
    const int64_t nb01 = src0->nb[1];
    const int64_t nb02 = src0->nb[2];
    const int64_t nb03 = src0->nb[3];

    // Parallelize by rows (dimension 1)
    const int64_t total_rows      = ne01;
    const int64_t rows_per_thread = (total_rows + num_threads - 1) / num_threads;
    const int64_t start_row       = thread_id * rows_per_thread;
    const int64_t end_row = (start_row + rows_per_thread < total_rows) ? (start_row + rows_per_thread) : total_rows;

    ប្រសិនបើ (start_row >= total_rows) {
        ត្រឡប់ 0;
    }

    // Iterate over source tensor dimensions
    សម្រាប់ (int64_t i03 = 0; i03 < ne03; i03++) {
        សម្រាប់ (int64_t i02 = 0; i02 < ne02; i02++) {
            // Calculate base linear index for this (i03, i02) slice in destination
            const int64_t dst_linear_base = i03 * ne02 * ne01 * ne00 + i02 * ne01 * ne00;

            // Process this thread's assigned rows
            សម្រាប់ (int64_t i01 = start_row; i01 < end_row; i01++) {
                // Linear index for start of this row in destination
                const int64_t dst_linear_row_base = dst_linear_base + i01 * ne00;

                // Inner loop over dimension 0
                សម្រាប់ (int64_t i00 = 0; i00 < ne00; i00++) {
                    // Source offset using non-contiguous strides
                    const int64_t    src_offset_bytes = i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;
                    const uint16_t * src_ptr = (const uint16_t *) ((const var_na *) src0_data + src_offset_bytes);

                    // Destination linear index (contiguous layout)
                    const int64_t dst_linear_idx = dst_linear_row_base + i00;

                    // Use atomic store for thread safety
                    atomic_store_f16((ca_ncala uint16_t *) &dst_data[dst_linear_idx], *src_ptr);
                }
            }
        }
    }

    ត្រឡប់ 0;
}
