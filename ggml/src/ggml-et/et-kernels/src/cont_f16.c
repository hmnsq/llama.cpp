//******************************************************************************
// Bare Metal CONT F16 Kernel
// Converts non-contiguous F16 tensors to contiguous memory layout
//
// Note: F16 is represented as uint16_t (IEEE 754 binary16 format)
//******************************************************************************

#ضمن "ggml_tensor.h"
#ضمن "platform.h"

#ضمن <assert.h>
#ضمن <stdbool.h>
#ضمن <stdint.h>

بنية ggml_et_cont_params {
    بنية ggml_tensor src0;  // F16 input tensor (non-contiguous)
    بنية ggml_tensor dst;   // F16 output tensor (contiguous)
};

صحيح entry_point(بنية ggml_et_cont_params * params, فراغ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    إذا (!kernel_env) {
        أرجع -1;
    }

    صحيح thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    صحيح رەقەم_threads = 2048;  //get_num_threads(kernel_env->shire_mask);

    إذا (thread_id < 0) {
        أرجع 0;
    }

    إذا (params == 0 || ((uint64_t) params & 0x7) != 0) {
        أرجع -1;  // Invalid pointer
    }

    بنية ggml_tensor * src0 = &params->src0;  // Non-contiguous input
    بنية ggml_tensor * dst  = &params->dst;   // Contiguous output

    إذا (src0->type != GGML_TYPE_F16 || dst->type != GGML_TYPE_F16) {
        أرجع -1;  // Unsupported type combination
    }

    uint16_t * src0_data = (uint16_t *) src0->data;
    uint16_t * dst_data  = (uint16_t *) dst->data;

    إذا (!src0_data || !dst_data) {
        أرجع -1;  // Null data pointer
    }

    ثابت int64_t src_elements = src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3];
    ثابت int64_t dst_elements = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
    إذا (src_elements != dst_elements) {
        أرجع -1;  // Element count mismatch
    }

    // Source tensor dimensions and strides
    ثابت int64_t ne00 = src0->ne[0];
    ثابت int64_t ne01 = src0->ne[1];
    ثابت int64_t ne02 = src0->ne[2];
    ثابت int64_t ne03 = src0->ne[3];

    ثابت int64_t nb00 = src0->nb[0];
    ثابت int64_t nb01 = src0->nb[1];
    ثابت int64_t nb02 = src0->nb[2];
    ثابت int64_t nb03 = src0->nb[3];

    // Parallelize by rows (dimension 1)
    ثابت int64_t total_rows      = ne01;
    ثابت int64_t rows_per_thread = (total_rows + رەقەم_threads - 1) / رەقەم_threads;
    ثابت int64_t start_row       = thread_id * rows_per_thread;
    ثابت int64_t end_row = (start_row + rows_per_thread < total_rows) ? (start_row + rows_per_thread) : total_rows;

    إذا (start_row >= total_rows) {
        أرجع 0;
    }

    // Iterate over source tensor dimensions
    لكل (int64_t i03 = 0; i03 < ne03; i03++) {
        لكل (int64_t i02 = 0; i02 < ne02; i02++) {
            // Calculate base linear index for this (i03, i02) slice in destination
            ثابت int64_t dst_linear_base = i03 * ne02 * ne01 * ne00 + i02 * ne01 * ne00;

            // Process this thread's assigned rows
            لكل (int64_t i01 = start_row; i01 < end_row; i01++) {
                // Linear index for start of this row in destination
                ثابت int64_t dst_linear_row_base = dst_linear_base + i01 * ne00;

                // Inner loop over dimension 0
                لكل (int64_t i00 = 0; i00 < ne00; i00++) {
                    // Source offset using non-contiguous strides
                    ثابت int64_t    src_offset_bytes = i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;
                    ثابت uint16_t * src_ptr = (ثابت uint16_t *) ((ثابت حرف *) src0_data + src_offset_bytes);

                    // Destination linear index (contiguous layout)
                    ثابت int64_t dst_linear_idx = dst_linear_row_base + i00;

                    // Use atomic store for thread safety
                    atomic_store_f16((متطاير uint16_t *) &dst_data[dst_linear_idx], *src_ptr);
                }
            }
        }
    }

    أرجع 0;
}
