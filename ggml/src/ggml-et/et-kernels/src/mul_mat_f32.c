#అన్తర్భావ "block_ops.h"
#అన్తర్భావ "ggml_tensor.h"
#అన్తర్భావ "platform.h"
#అన్తర్భావ "quants.h"

#అన్తర్భావ <etsoc/common/utils.h>
#అన్తర్భావ <stdint.h>
#అన్తర్భావ <stdio.h>

పూర్ణాంకం నమోదు_బిందువు(నిర్మాణం ggml_et_ద్విమాన_params * params, శూన్య * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ఒకవేళ (!kernel_env || params == 0 || ((uint64_t) params & 0x7) != 0) {
        తిరిగి -1;
    }

    // Thread coordination
    పూర్ణాంకం thread_id   = get_relative_thread_id(kernel_env->shire_ముసుగు);
    పూర్ణాంకం అంకె_దారాలు = get_అంకె_దారాలు(kernel_env->shire_ముసుగు);

    ఒకవేళ (thread_id < 0 || (thread_id & 1)) {
        తిరిగి 0;  // Skip odd threads to avoid resource contention
    }

    పూర్ణాంకం effective_thread_id   = thread_id / 2;
    పూర్ణాంకం effective_అంకె_దారాలు = (అంకె_దారాలు + 1) / 2;

    // Extract tensor references
    నిర్మాణం ggml_tensor * src0 = &params->src0;  // Weight matrix A (F32)
    నిర్మాణం ggml_tensor * src1 = &params->src1;  // Activation matrix B (F16/F32)
    నిర్మాణం ggml_tensor * dst  = &params->dst;   // Output matrix C (F32)

    // Generic non-matrix-engine path: F32 x (F16/F32) -> F32
    ఒకవేళ (src0->type != GGML_TYPE_F32 || (src1->type != GGML_TYPE_F16 && src1->type != GGML_TYPE_F32) ||
        dst->type != GGML_TYPE_F32) {
        తిరిగి -1;
    }

    స్థిరరాశి దశాంశం * src0_దత్తాంశం = (స్థిరరాశి దశాంశం *) src0->దత్తాంశం;
    దశాంశం *       dst_దత్తాంశం  = (దశాంశం *) dst->దత్తాంశం;

    // Dimensions and Strides
    స్థిరరాశి int64_t K = src0->ne[0];
    స్థిరరాశి int64_t M = src0->ne[1];
    స్థిరరాశి int64_t N = src1->ne[1];

    స్థిరరాశి int64_t ne02 = src0->ne[2], ne03 = src0->ne[3];
    స్థిరరాశి int64_t ne12 = src1->ne[2], ne13 = src1->ne[3];
    స్థిరరాశి int64_t ne2 = dst->ne[2], ne3 = dst->ne[3];

    స్థిరరాశి size_t nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    స్థిరరాశి size_t nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];
    స్థిరరాశి size_t nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];

    // F32 specific block size and counts
    స్థిరరాశి పూర్ణాంకం     ఖండం_కొలత_3  = QK_F32;
    స్థిరరాశి int64_t K_ఖండాలు    = K / ఖండం_కొలత_3;
    స్థిరరాశి int64_t K_remainder = K % ఖండం_కొలత_3;

    // Threading distribution
    స్థిరరాశి uint64_t మొత్తం_elements = M * N * ne2 * ne3;
    స్థిరరాశి uint64_t చొప్పున_thread     = 16;
    స్థిరరాశి uint64_t దారాలు_stride = చొప్పున_thread * effective_అంకె_దారాలు;

    ఒకవేళ (effective_thread_id * చొప్పున_thread >= మొత్తం_elements) {
        తిరిగి 0;
    }

    // Broadcasting support
    స్థిరరాశి int64_t r2 = ne12 / ne02;
    స్థిరరాశి int64_t r3 = ne13 / ne03;

    కొరకు (uint64_t పునాది_idx = effective_thread_id * చొప్పున_thread; పునాది_idx < మొత్తం_elements; పునాది_idx += దారాలు_stride) {
        కొరకు (uint64_t j = 0; j < చొప్పున_thread; j++) {
            స్థిరరాశి uint64_t idx = పునాది_idx + j;
            ఒకవేళ (idx >= మొత్తం_elements) {
                ఆపు;
            }

            // Index decoding
            స్థిరరాశి int64_t i3   = idx / (M * N * ne2);
            స్థిరరాశి int64_t rem3 = idx % (M * N * ne2);
            స్థిరరాశి int64_t i2   = rem3 / (M * N);
            స్థిరరాశి int64_t rem2 = rem3 % (M * N);
            స్థిరరాశి int64_t n    = rem2 / M;
            స్థిరరాశి int64_t m    = rem2 % M;

            స్థిరరాశి int64_t i03 = i3 / r3, i02 = i2 / r2;
            స్థిరరాశి int64_t i13 = (ne13 > 1) ? i3 : 0, i12 = (ne12 > 1) ? i2 : 0;

            దశాంశం         కూడిక     = 0.0f;
            స్థిరరాశి దశాంశం * f32_అడ్డువరుస = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) src0_దత్తాంశం + m * nb01 + i02 * nb02 + i03 * nb03);

            ఒకవేళ (src1->type == GGML_TYPE_F32) {
                స్థిరరాశి దశాంశం * src1_దత్తాంశం = (స్థిరరాశి దశాంశం *) src1->దత్తాంశం;

                కొరకు (int64_t kb = 0; kb < K_ఖండాలు; kb++) {
                    స్థిరరాశి దశాంశం * b_నిలువు_ptr =
                        (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) src1_దత్తాంశం + (kb * ఖండం_కొలత_3) * పరిమాణం(దశాంశం) + n * nb11 +
                                         i12 * nb12 + i13 * nb13);
                    కూడిక += compute_ఖండం_dot_లబ్ధం_f32(&f32_అడ్డువరుస[kb * ఖండం_కొలత_3], b_నిలువు_ptr);
                }

                ఒకవేళ (K_remainder > 0) {
                    స్థిరరాశి int64_t స్థానభ్రంశం    = K_ఖండాలు * ఖండం_కొలత_3;
                    స్థిరరాశి దశాంశం * b_నిలువు_ptr = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) src1_దత్తాంశం + స్థానభ్రంశం * పరిమాణం(దశాంశం) +
                                                               n * nb11 + i12 * nb12 + i13 * nb13);
                    కూడిక += compute_ఖండం_dot_లబ్ధం_f32_partial(&f32_అడ్డువరుస[స్థానభ్రంశం], b_నిలువు_ptr, K_remainder);
                }
            } లేకపోతే {
                స్థిరరాశి uint16_t * src1_దత్తాంశం = (స్థిరరాశి uint16_t *) src1->దత్తాంశం;

                కొరకు (int64_t kb = 0; kb < K_ఖండాలు; kb++) {
                    స్థిరరాశి uint16_t * b_నిలువు_ptr =
                        (స్థిరరాశి uint16_t *) ((స్థిరరాశి అక్షరం *) src1_దత్తాంశం + (kb * ఖండం_కొలత_3) * పరిమాణం(uint16_t) + n * nb11 +
                                            i12 * nb12 + i13 * nb13);
                    కూడిక += compute_ఖండం_dot_లబ్ధం_f32_f16_partial(&f32_అడ్డువరుస[kb * ఖండం_కొలత_3], b_నిలువు_ptr, ఖండం_కొలత_3);
                }

                ఒకవేళ (K_remainder > 0) {
                    స్థిరరాశి int64_t    స్థానభ్రంశం = K_ఖండాలు * ఖండం_కొలత_3;
                    స్థిరరాశి uint16_t * b_నిలువు_ptr =
                        (స్థిరరాశి uint16_t *) ((స్థిరరాశి అక్షరం *) src1_దత్తాంశం + స్థానభ్రంశం * పరిమాణం(uint16_t) + n * nb11 +
                                            i12 * nb12 + i13 * nb13);
                    కూడిక += compute_ఖండం_dot_లబ్ధం_f32_f16_partial(&f32_అడ్డువరుస[స్థానభ్రంశం], b_నిలువు_ptr, K_remainder);
                }
            }

            // Atomic store for output
            అస్థిర దశాంశం * c_element =
                (అస్థిర దశాంశం *) ((అక్షరం *) dst_దత్తాంశం + m * dst->nb[0] + n * nb1 + i2 * nb2 + i3 * nb3);
            atomic_store_f32(c_element, కూడిక);
        }
    }

    తిరిగి 0;
}
