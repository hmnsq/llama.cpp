//******************************************************************************
// Bare Metal CONT F16 Kernel
// Converts non-contiguous F16 tensors to contiguous memory layout
//
// Note: F16 is represented as uint16_t (IEEE 754 binary16 format)
//******************************************************************************

#অন্তর্ভাব "ggml_tensor.h"
#অন্তর্ভাব "platform.h"

#অন্তর্ভাব <assert.h>
#অন্তর্ভাব <stdbool.h>
#অন্তর্ভাব <stdint.h>

কাঠামো ggml_et_cont_params {
    কাঠামো ggml_tensor src0;  // F16 input tensor (non-contiguous)
    কাঠামো ggml_tensor dst;   // F16 output tensor (contiguous)
};

পূর্ণসংখ্যা entry_point(কাঠামো ggml_et_cont_params * params, শূন্য * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    যদি (!kernel_env) {
        ফেরত -1;
    }

    পূর্ণসংখ্যা thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    পূর্ণসংখ্যা অঙ্ক_threads = 2048;  //get_num_threads(kernel_env->shire_mask);

    যদি (thread_id < 0) {
        ফেরত 0;
    }

    যদি (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ফেরত -1;  // Invalid pointer
    }

    কাঠামো ggml_tensor * src0 = &params->src0;  // Non-contiguous input
    কাঠামো ggml_tensor * dst  = &params->dst;   // Contiguous output

    যদি (src0->type != GGML_TYPE_F16 || dst->type != GGML_TYPE_F16) {
        ফেরত -1;  // Unsupported type combination
    }

    uint16_t * src0_তথ্য = (uint16_t *) src0->তথ্য;
    uint16_t * dst_তথ্য  = (uint16_t *) dst->তথ্য;

    যদি (!src0_তথ্য || !dst_তথ্য) {
        ফেরত -1;  // Null data pointer
    }

    ধ্রুবক int64_t src_elements = src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3];
    ধ্রুবক int64_t dst_elements = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
    যদি (src_elements != dst_elements) {
        ফেরত -1;  // Element count mismatch
    }

    // Source tensor dimensions and strides
    ধ্রুবক int64_t ne00 = src0->ne[0];
    ধ্রুবক int64_t ne01 = src0->ne[1];
    ধ্রুবক int64_t ne02 = src0->ne[2];
    ধ্রুবক int64_t ne03 = src0->ne[3];

    ধ্রুবক int64_t nb00 = src0->nb[0];
    ধ্রুবক int64_t nb01 = src0->nb[1];
    ধ্রুবক int64_t nb02 = src0->nb[2];
    ধ্রুবক int64_t nb03 = src0->nb[3];

    // Parallelize by rows (dimension 1)
    ধ্রুবক int64_t যোগ_rows      = ne01;
    ধ্রুবক int64_t rows_per_thread = (যোগ_rows + অঙ্ক_threads - 1) / অঙ্ক_threads;
    ধ্রুবক int64_t start_row       = thread_id * rows_per_thread;
    ধ্রুবক int64_t end_row = (start_row + rows_per_thread < যোগ_rows) ? (start_row + rows_per_thread) : যোগ_rows;

    যদি (start_row >= যোগ_rows) {
        ফেরত 0;
    }

    // Iterate over source tensor dimensions
    জন্য (int64_t i03 = 0; i03 < ne03; i03++) {
        জন্য (int64_t i02 = 0; i02 < ne02; i02++) {
            // Calculate base linear index for this (i03, i02) slice in destination
            ধ্রুবক int64_t dst_linear_base = i03 * ne02 * ne01 * ne00 + i02 * ne01 * ne00;

            // Process this thread's assigned rows
            জন্য (int64_t i01 = start_row; i01 < end_row; i01++) {
                // Linear index for start of this row in destination
                ধ্রুবক int64_t dst_linear_row_base = dst_linear_base + i01 * ne00;

                // Inner loop over dimension 0
                জন্য (int64_t i00 = 0; i00 < ne00; i00++) {
                    // Source offset using non-contiguous strides
                    ধ্রুবক int64_t    src_offset_bytes = i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;
                    ধ্রুবক uint16_t * src_ptr = (ধ্রুবক uint16_t *) ((ধ্রুবক অক্ষর *) src0_তথ্য + src_offset_bytes);

                    // Destination linear index (contiguous layout)
                    ধ্রুবক int64_t dst_linear_idx = dst_linear_row_base + i00;

                    // Use atomic store for thread safety
                    atomic_store_f16((অস্থির uint16_t *) &dst_তথ্য[dst_linear_idx], *src_ptr);
                }
            }
        }
    }

    ফেরত 0;
}
