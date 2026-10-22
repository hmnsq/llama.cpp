//******************************************************************************
// Bare Metal CONT F16 Kernel
// Converts non-contiguous F16 tensors to contiguous memory layout
//
// Note: F16 is represented as uint16_t (IEEE 754 binary16 format)
//******************************************************************************

#అన్తర్భావ "ggml_tensor.h"
#అన్తర్భావ "platform.h"

#అన్తర్భావ <assert.h>
#అన్తర్భావ <stdbool.h>
#అన్తర్భావ <stdint.h>

నిర్మాణం ggml_et_cont_params {
    నిర్మాణం ggml_tensor src0;  // F16 input tensor (non-contiguous)
    నిర్మాణం ggml_tensor dst;   // F16 output tensor (contiguous)
};

పూర్ణాంకం నమోదు_బిందువు(నిర్మాణం ggml_et_cont_params * params, శూన్య * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ఒకవేళ (!kernel_env) {
        తిరిగి -1;
    }

    పూర్ణాంకం thread_id   = get_relative_thread_id(kernel_env->shire_ముసుగు);
    పూర్ణాంకం అంకె_దారాలు = 2048;  //get_num_threads(kernel_env->shire_mask);

    ఒకవేళ (thread_id < 0) {
        తిరిగి 0;
    }

    ఒకవేళ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        తిరిగి -1;  // Invalid pointer
    }

    నిర్మాణం ggml_tensor * src0 = &params->src0;  // Non-contiguous input
    నిర్మాణం ggml_tensor * dst  = &params->dst;   // Contiguous output

    ఒకవేళ (src0->type != GGML_TYPE_F16 || dst->type != GGML_TYPE_F16) {
        తిరిగి -1;  // Unsupported type combination
    }

    uint16_t * src0_దత్తాంశం = (uint16_t *) src0->దత్తాంశం;
    uint16_t * dst_దత్తాంశం  = (uint16_t *) dst->దత్తాంశం;

    ఒకవేళ (!src0_దత్తాంశం || !dst_దత్తాంశం) {
        తిరిగి -1;  // Null data pointer
    }

    స్థిరరాశి int64_t src_elements = src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3];
    స్థిరరాశి int64_t dst_elements = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
    ఒకవేళ (src_elements != dst_elements) {
        తిరిగి -1;  // Element count mismatch
    }

    // Source tensor dimensions and strides
    స్థిరరాశి int64_t ne00 = src0->ne[0];
    స్థిరరాశి int64_t ne01 = src0->ne[1];
    స్థిరరాశి int64_t ne02 = src0->ne[2];
    స్థిరరాశి int64_t ne03 = src0->ne[3];

    స్థిరరాశి int64_t nb00 = src0->nb[0];
    స్థిరరాశి int64_t nb01 = src0->nb[1];
    స్థిరరాశి int64_t nb02 = src0->nb[2];
    స్థిరరాశి int64_t nb03 = src0->nb[3];

    // Parallelize by rows (dimension 1)
    స్థిరరాశి int64_t మొత్తం_అడ్డువరుసలు      = ne01;
    స్థిరరాశి int64_t అడ్డువరుసలు_చొప్పున_thread = (మొత్తం_అడ్డువరుసలు + అంకె_దారాలు - 1) / అంకె_దారాలు;
    స్థిరరాశి int64_t ప్రారంభం_అడ్డువరుస       = thread_id * అడ్డువరుసలు_చొప్పున_thread;
    స్థిరరాశి int64_t చివర_అడ్డువరుస = (ప్రారంభం_అడ్డువరుస + అడ్డువరుసలు_చొప్పున_thread < మొత్తం_అడ్డువరుసలు) ? (ప్రారంభం_అడ్డువరుస + అడ్డువరుసలు_చొప్పున_thread) : మొత్తం_అడ్డువరుసలు;

    ఒకవేళ (ప్రారంభం_అడ్డువరుస >= మొత్తం_అడ్డువరుసలు) {
        తిరిగి 0;
    }

    // Iterate over source tensor dimensions
    కొరకు (int64_t i03 = 0; i03 < ne03; i03++) {
        కొరకు (int64_t i02 = 0; i02 < ne02; i02++) {
            // Calculate base linear index for this (i03, i02) slice in destination
            స్థిరరాశి int64_t dst_రేఖీయ_పునాది = i03 * ne02 * ne01 * ne00 + i02 * ne01 * ne00;

            // Process this thread's assigned rows
            కొరకు (int64_t i01 = ప్రారంభం_అడ్డువరుస; i01 < చివర_అడ్డువరుస; i01++) {
                // Linear index for start of this row in destination
                స్థిరరాశి int64_t dst_రేఖీయ_అడ్డువరుస_పునాది = dst_రేఖీయ_పునాది + i01 * ne00;

                // Inner loop over dimension 0
                కొరకు (int64_t i00 = 0; i00 < ne00; i00++) {
                    // Source offset using non-contiguous strides
                    స్థిరరాశి int64_t    src_స్థానభ్రంశం_బైట్లు = i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;
                    స్థిరరాశి uint16_t * src_ptr = (స్థిరరాశి uint16_t *) ((స్థిరరాశి అక్షరం *) src0_దత్తాంశం + src_స్థానభ్రంశం_బైట్లు);

                    // Destination linear index (contiguous layout)
                    స్థిరరాశి int64_t dst_రేఖీయ_idx = dst_రేఖీయ_అడ్డువరుస_పునాది + i00;

                    // Use atomic store for thread safety
                    atomic_store_f16((అస్థిర uint16_t *) &dst_దత్తాంశం[dst_రేఖీయ_idx], *src_ptr);
                }
            }
        }
    }

    తిరిగి 0;
}
