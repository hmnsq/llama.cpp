//******************************************************************************
// Bare Metal CONT F16 Kernel
// Converts non-contiguous F16 tensors to contiguous memory layout
//
// Note: F16 is represented as uint16_t (IEEE 754 binary16 format)
//******************************************************************************

#شامل "ggml_tensor.h"
#شامل "platform.h"

#شامل <assert.h>
#شامل <stdbool.h>
#شامل <stdint.h>

ساخت ggml_et_cont_params {
    ساخت ggml_tensor src0;  // F16 input tensor (non-contiguous)
    ساخت ggml_tensor dst;   // F16 output tensor (contiguous)
};

صحیح اندراج_نقطہ(ساخت ggml_et_cont_params * params, خالی * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    اگر (!kernel_env) {
        واپس -1;
    }

    صحیح thread_id   = get_relative_thread_id(kernel_env->shire_نقاب);
    صحیح عدد_دھاگے = 2048;  //get_num_threads(kernel_env->shire_mask);

    اگر (thread_id < 0) {
        واپس 0;
    }

    اگر (params == 0 || ((uint64_t) params & 0x7) != 0) {
        واپس -1;  // Invalid pointer
    }

    ساخت ggml_tensor * src0 = &params->src0;  // Non-contiguous input
    ساخت ggml_tensor * dst  = &params->dst;   // Contiguous output

    اگر (src0->type != GGML_TYPE_F16 || dst->type != GGML_TYPE_F16) {
        واپس -1;  // Unsupported type combination
    }

    uint16_t * src0_ڈیٹا = (uint16_t *) src0->ڈیٹا;
    uint16_t * dst_ڈیٹا  = (uint16_t *) dst->ڈیٹا;

    اگر (!src0_ڈیٹا || !dst_ڈیٹا) {
        واپس -1;  // Null data pointer
    }

    ثابت int64_t src_elements = src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3];
    ثابت int64_t dst_elements = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
    اگر (src_elements != dst_elements) {
        واپس -1;  // Element count mismatch
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
    ثابت int64_t کل_صفیں      = ne01;
    ثابت int64_t صفیں_فی_thread = (کل_صفیں + عدد_دھاگے - 1) / عدد_دھاگے;
    ثابت int64_t آغاز_صف       = thread_id * صفیں_فی_thread;
    ثابت int64_t انتہا_صف = (آغاز_صف + صفیں_فی_thread < کل_صفیں) ? (آغاز_صف + صفیں_فی_thread) : کل_صفیں;

    اگر (آغاز_صف >= کل_صفیں) {
        واپس 0;
    }

    // Iterate over source tensor dimensions
    برائے (int64_t i03 = 0; i03 < ne03; i03++) {
        برائے (int64_t i02 = 0; i02 < ne02; i02++) {
            // Calculate base linear index for this (i03, i02) slice in destination
            ثابت int64_t dst_خطی_بنیاد = i03 * ne02 * ne01 * ne00 + i02 * ne01 * ne00;

            // Process this thread's assigned rows
            برائے (int64_t i01 = آغاز_صف; i01 < انتہا_صف; i01++) {
                // Linear index for start of this row in destination
                ثابت int64_t dst_خطی_صف_بنیاد = dst_خطی_بنیاد + i01 * ne00;

                // Inner loop over dimension 0
                برائے (int64_t i00 = 0; i00 < ne00; i00++) {
                    // Source offset using non-contiguous strides
                    ثابت int64_t    src_انحراف_بائٹیں = i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;
                    ثابت uint16_t * src_ptr = (ثابت uint16_t *) ((ثابت حرف *) src0_ڈیٹا + src_انحراف_بائٹیں);

                    // Destination linear index (contiguous layout)
                    ثابت int64_t dst_خطی_idx = dst_خطی_صف_بنیاد + i00;

                    // Use atomic store for thread safety
                    atomic_store_f16((متلون uint16_t *) &dst_ڈیٹا[dst_خطی_idx], *src_ptr);
                }
            }
        }
    }

    واپس 0;
}
