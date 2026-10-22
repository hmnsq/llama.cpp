//******************************************************************************
// Bare Metal CONT F16 Kernel
// Converts non-contiguous F16 tensors to contiguous memory layout
//
// Note: F16 is represented as uint16_t (IEEE 754 binary16 format)
//******************************************************************************

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml_tensor.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "platform.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <assert.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdbool.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>

ਬਣਤਰ ggml_et_cont_params {
    ਬਣਤਰ ggml_tensor src0;  // F16 input tensor (non-contiguous)
    ਬਣਤਰ ggml_tensor dst;   // F16 output tensor (contiguous)
};

ਪੂਰਨ ਇੰਦਰਾਜ_ਬਿੰਦੂ(ਬਣਤਰ ggml_et_cont_params * params, ਖਾਲੀ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ਜੇ (!kernel_env) {
        ਵਾਪਸ -1;
    }

    ਪੂਰਨ thread_id   = get_relative_thread_id(kernel_env->shire_ਨਕਾਬ);
    ਪੂਰਨ ਅੰਕ_ਧਾਗੇ = 2048;  //get_num_threads(kernel_env->shire_mask);

    ਜੇ (thread_id < 0) {
        ਵਾਪਸ 0;
    }

    ਜੇ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ਵਾਪਸ -1;  // Invalid pointer
    }

    ਬਣਤਰ ggml_tensor * src0 = &params->src0;  // Non-contiguous input
    ਬਣਤਰ ggml_tensor * dst  = &params->dst;   // Contiguous output

    ਜੇ (src0->type != GGML_TYPE_F16 || dst->type != GGML_TYPE_F16) {
        ਵਾਪਸ -1;  // Unsupported type combination
    }

    uint16_t * src0_ਅੰਕੜੇ = (uint16_t *) src0->ਅੰਕੜੇ;
    uint16_t * dst_ਅੰਕੜੇ  = (uint16_t *) dst->ਅੰਕੜੇ;

    ਜੇ (!src0_ਅੰਕੜੇ || !dst_ਅੰਕੜੇ) {
        ਵਾਪਸ -1;  // Null data pointer
    }

    ਅਚਲ int64_t src_elements = src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3];
    ਅਚਲ int64_t dst_elements = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
    ਜੇ (src_elements != dst_elements) {
        ਵਾਪਸ -1;  // Element count mismatch
    }

    // Source tensor dimensions and strides
    ਅਚਲ int64_t ne00 = src0->ne[0];
    ਅਚਲ int64_t ne01 = src0->ne[1];
    ਅਚਲ int64_t ne02 = src0->ne[2];
    ਅਚਲ int64_t ne03 = src0->ne[3];

    ਅਚਲ int64_t nb00 = src0->nb[0];
    ਅਚਲ int64_t nb01 = src0->nb[1];
    ਅਚਲ int64_t nb02 = src0->nb[2];
    ਅਚਲ int64_t nb03 = src0->nb[3];

    // Parallelize by rows (dimension 1)
    ਅਚਲ int64_t ਕੁੱਲ_ਪੰਕਤੀਆਂ      = ne01;
    ਅਚਲ int64_t ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread = (ਕੁੱਲ_ਪੰਕਤੀਆਂ + ਅੰਕ_ਧਾਗੇ - 1) / ਅੰਕ_ਧਾਗੇ;
    ਅਚਲ int64_t ਸ਼ੁਰੂ_ਪੰਕਤੀ       = thread_id * ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread;
    ਅਚਲ int64_t ਸਿਰਾ_ਪੰਕਤੀ = (ਸ਼ੁਰੂ_ਪੰਕਤੀ + ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread < ਕੁੱਲ_ਪੰਕਤੀਆਂ) ? (ਸ਼ੁਰੂ_ਪੰਕਤੀ + ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread) : ਕੁੱਲ_ਪੰਕਤੀਆਂ;

    ਜੇ (ਸ਼ੁਰੂ_ਪੰਕਤੀ >= ਕੁੱਲ_ਪੰਕਤੀਆਂ) {
        ਵਾਪਸ 0;
    }

    // Iterate over source tensor dimensions
    ਲਈ (int64_t i03 = 0; i03 < ne03; i03++) {
        ਲਈ (int64_t i02 = 0; i02 < ne02; i02++) {
            // Calculate base linear index for this (i03, i02) slice in destination
            ਅਚਲ int64_t dst_ਰੇਖਿਕ_ਅਧਾਰ = i03 * ne02 * ne01 * ne00 + i02 * ne01 * ne00;

            // Process this thread's assigned rows
            ਲਈ (int64_t i01 = ਸ਼ੁਰੂ_ਪੰਕਤੀ; i01 < ਸਿਰਾ_ਪੰਕਤੀ; i01++) {
                // Linear index for start of this row in destination
                ਅਚਲ int64_t dst_ਰੇਖਿਕ_ਪੰਕਤੀ_ਅਧਾਰ = dst_ਰੇਖਿਕ_ਅਧਾਰ + i01 * ne00;

                // Inner loop over dimension 0
                ਲਈ (int64_t i00 = 0; i00 < ne00; i00++) {
                    // Source offset using non-contiguous strides
                    ਅਚਲ int64_t    src_ਵਿੱਥ_ਬਾਈਟਾਂ = i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;
                    ਅਚਲ uint16_t * src_ptr = (ਅਚਲ uint16_t *) ((ਅਚਲ ਅੱਖਰ *) src0_ਅੰਕੜੇ + src_ਵਿੱਥ_ਬਾਈਟਾਂ);

                    // Destination linear index (contiguous layout)
                    ਅਚਲ int64_t dst_ਰੇਖਿਕ_idx = dst_ਰੇਖਿਕ_ਪੰਕਤੀ_ਅਧਾਰ + i00;

                    // Use atomic store for thread safety
                    atomic_store_f16((ਅਸਥਿਰ uint16_t *) &dst_ਅੰਕੜੇ[dst_ਰੇਖਿਕ_idx], *src_ptr);
                }
            }
        }
    }

    ਵਾਪਸ 0;
}
