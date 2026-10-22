//******************************************************************************
// Bare Metal CONT F16 Kernel
// Converts non-contiguous F16 tensors to contiguous memory layout
//
// Note: F16 is represented as uint16_t (IEEE 754 binary16 format)
//******************************************************************************

#समावेश "ggml_tensor.h"
#समावेश "platform.h"

#समावेश <assert.h>
#समावेश <stdbool.h>
#समावेश <stdint.h>

काष्ठा ggml_et_cont_params {
    काष्ठा ggml_tensor src0;  // F16 input tensor (non-contiguous)
    काष्ठा ggml_tensor dst;   // F16 output tensor (contiguous)
};

पूर्णांक entry_point(काष्ठा ggml_et_cont_params * params, व्योम * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    अगर (!kernel_env) {
        वापस -1;
    }

    पूर्णांक thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    पूर्णांक अंक_threads = 2048;  //get_num_threads(kernel_env->shire_mask);

    अगर (thread_id < 0) {
        वापस 0;
    }

    अगर (params == 0 || ((uint64_t) params & 0x7) != 0) {
        वापस -1;  // Invalid pointer
    }

    काष्ठा ggml_tensor * src0 = &params->src0;  // Non-contiguous input
    काष्ठा ggml_tensor * dst  = &params->dst;   // Contiguous output

    अगर (src0->type != GGML_TYPE_F16 || dst->type != GGML_TYPE_F16) {
        वापस -1;  // Unsupported type combination
    }

    uint16_t * src0_तथ्य = (uint16_t *) src0->तथ्य;
    uint16_t * dst_तथ्य  = (uint16_t *) dst->तथ्य;

    अगर (!src0_तथ्य || !dst_तथ्य) {
        वापस -1;  // Null data pointer
    }

    स्थिर int64_t src_elements = src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3];
    स्थिर int64_t dst_elements = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
    अगर (src_elements != dst_elements) {
        वापस -1;  // Element count mismatch
    }

    // Source tensor dimensions and strides
    स्थिर int64_t ne00 = src0->ne[0];
    स्थिर int64_t ne01 = src0->ne[1];
    स्थिर int64_t ne02 = src0->ne[2];
    स्थिर int64_t ne03 = src0->ne[3];

    स्थिर int64_t nb00 = src0->nb[0];
    स्थिर int64_t nb01 = src0->nb[1];
    स्थिर int64_t nb02 = src0->nb[2];
    स्थिर int64_t nb03 = src0->nb[3];

    // Parallelize by rows (dimension 1)
    स्थिर int64_t योग_rows      = ne01;
    स्थिर int64_t rows_per_thread = (योग_rows + अंक_threads - 1) / अंक_threads;
    स्थिर int64_t start_row       = thread_id * rows_per_thread;
    स्थिर int64_t end_row = (start_row + rows_per_thread < योग_rows) ? (start_row + rows_per_thread) : योग_rows;

    अगर (start_row >= योग_rows) {
        वापस 0;
    }

    // Iterate over source tensor dimensions
    क्रम (int64_t i03 = 0; i03 < ne03; i03++) {
        क्रम (int64_t i02 = 0; i02 < ne02; i02++) {
            // Calculate base linear index for this (i03, i02) slice in destination
            स्थिर int64_t dst_linear_base = i03 * ne02 * ne01 * ne00 + i02 * ne01 * ne00;

            // Process this thread's assigned rows
            क्रम (int64_t i01 = start_row; i01 < end_row; i01++) {
                // Linear index for start of this row in destination
                स्थिर int64_t dst_linear_row_base = dst_linear_base + i01 * ne00;

                // Inner loop over dimension 0
                क्रम (int64_t i00 = 0; i00 < ne00; i00++) {
                    // Source offset using non-contiguous strides
                    स्थिर int64_t    src_offset_bytes = i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;
                    स्थिर uint16_t * src_ptr = (स्थिर uint16_t *) ((स्थिर अक्षर *) src0_तथ्य + src_offset_bytes);

                    // Destination linear index (contiguous layout)
                    स्थिर int64_t dst_linear_idx = dst_linear_row_base + i00;

                    // Use atomic store for thread safety
                    atomic_store_f16((अस्थिर uint16_t *) &dst_तथ्य[dst_linear_idx], *src_ptr);
                }
            }
        }
    }

    वापस 0;
}
