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

صحيح داخلا_نقطو(بنية ggml_et_cont_params * params, فراغ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    إذا (!kernel_env) {
        أرجع -1;
    }

    صحيح thread_id   = get_relative_thread_id(kernel_env->shire_نقاب);
    صحيح انگ_ڌاڳا = 2048;  //get_num_threads(kernel_env->shire_mask);

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

    uint16_t * src0_معلومات = (uint16_t *) src0->معلومات;
    uint16_t * dst_معلومات  = (uint16_t *) dst->معلومات;

    إذا (!src0_معلومات || !dst_معلومات) {
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
    ثابت int64_t ڪل_صفون      = ne01;
    ثابت int64_t صفون_في_thread = (ڪل_صفون + انگ_ڌاڳا - 1) / انگ_ڌاڳا;
    ثابت int64_t شروعات_صف       = thread_id * صفون_في_thread;
    ثابت int64_t پڇاڙي_صف = (شروعات_صف + صفون_في_thread < ڪل_صفون) ? (شروعات_صف + صفون_في_thread) : ڪل_صفون;

    إذا (شروعات_صف >= ڪل_صفون) {
        أرجع 0;
    }

    // Iterate over source tensor dimensions
    لكل (int64_t i03 = 0; i03 < ne03; i03++) {
        لكل (int64_t i02 = 0; i02 < ne02; i02++) {
            // Calculate base linear index for this (i03, i02) slice in destination
            ثابت int64_t dst_لڪيري_بنياد = i03 * ne02 * ne01 * ne00 + i02 * ne01 * ne00;

            // Process this thread's assigned rows
            لكل (int64_t i01 = شروعات_صف; i01 < پڇاڙي_صف; i01++) {
                // Linear index for start of this row in destination
                ثابت int64_t dst_لڪيري_صف_بنياد = dst_لڪيري_بنياد + i01 * ne00;

                // Inner loop over dimension 0
                لكل (int64_t i00 = 0; i00 < ne00; i00++) {
                    // Source offset using non-contiguous strides
                    ثابت int64_t    src_وٿي_بائيٽون = i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;
                    ثابت uint16_t * src_ptr = (ثابت uint16_t *) ((ثابت حرف *) src0_معلومات + src_وٿي_بائيٽون);

                    // Destination linear index (contiguous layout)
                    ثابت int64_t dst_لڪيري_idx = dst_لڪيري_صف_بنياد + i00;

                    // Use atomic store for thread safety
                    atomic_store_f16((متطاير uint16_t *) &dst_معلومات[dst_لڪيري_idx], *src_ptr);
                }
            }
        }
    }

    أرجع 0;
}
