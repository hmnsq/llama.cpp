#अन्तर्भाव "block_ops.h"
#अन्तर्भाव "ggml_tensor.h"
#अन्तर्भाव "platform.h"
#अन्तर्भाव "quants.h"

#अन्तर्भाव <etsoc/common/utils.h>
#अन्तर्भाव <stdint.h>
#अन्तर्भाव <stdio.h>

पूर्णाङ्क प्रविष्टिः_point(संरचना ggml_et_द्विमान_params * params, शून्य * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    यदि (!kernel_env || params == 0 || ((uint64_t) params & 0x7) != 0) {
        प्रत्यागम -1;
    }

    // Thread coordination
    पूर्णाङ्क thread_id   = get_relative_thread_id(kernel_env->shire_आवरणम्);
    पूर्णाङ्क अङ्कः_तन्तवः = get_अङ्कः_तन्तवः(kernel_env->shire_आवरणम्);

    यदि (thread_id < 0 || (thread_id & 1)) {
        प्रत्यागम 0;  // Skip odd threads to avoid resource contention
    }

    पूर्णाङ्क effective_thread_id   = thread_id / 2;
    पूर्णाङ्क effective_अङ्कः_तन्तवः = (अङ्कः_तन्तवः + 1) / 2;

    // Extract tensor references
    संरचना ggml_tensor * src0 = &params->src0;  // Weight matrix A (F32)
    संरचना ggml_tensor * src1 = &params->src1;  // Activation matrix B (F16/F32)
    संरचना ggml_tensor * dst  = &params->dst;   // Output matrix C (F32)

    // Generic non-matrix-engine path: F32 x (F16/F32) -> F32
    यदि (src0->type != GGML_TYPE_F32 || (src1->type != GGML_TYPE_F16 && src1->type != GGML_TYPE_F32) ||
        dst->type != GGML_TYPE_F32) {
        प्रत्यागम -1;
    }

    ध्रुव भिन्न * src0_दत्त = (ध्रुव भिन्न *) src0->दत्त;
    भिन्न *       dst_दत्त  = (भिन्न *) dst->दत्त;

    // Dimensions and Strides
    ध्रुव int64_t K = src0->ne[0];
    ध्रुव int64_t M = src0->ne[1];
    ध्रुव int64_t N = src1->ne[1];

    ध्रुव int64_t ne02 = src0->ne[2], ne03 = src0->ne[3];
    ध्रुव int64_t ne12 = src1->ne[2], ne13 = src1->ne[3];
    ध्रुव int64_t ne2 = dst->ne[2], ne3 = dst->ne[3];

    ध्रुव size_t nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    ध्रुव size_t nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];
    ध्रुव size_t nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];

    // F32 specific block size and counts
    ध्रुव पूर्णाङ्क     खण्डम्_आकार_3  = QK_F32;
    ध्रुव int64_t K_खण्डानि    = K / खण्डम्_आकार_3;
    ध्रुव int64_t K_remainder = K % खण्डम्_आकार_3;

    // Threading distribution
    ध्रुव uint64_t योग_elements = M * N * ne2 * ne3;
    ध्रुव uint64_t प्रत्येकम्_thread     = 16;
    ध्रुव uint64_t तन्तवः_stride = प्रत्येकम्_thread * effective_अङ्कः_तन्तवः;

    यदि (effective_thread_id * प्रत्येकम्_thread >= योग_elements) {
        प्रत्यागम 0;
    }

    // Broadcasting support
    ध्रुव int64_t r2 = ne12 / ne02;
    ध्रुव int64_t r3 = ne13 / ne03;

    प्रति (uint64_t आधारः_idx = effective_thread_id * प्रत्येकम्_thread; आधारः_idx < योग_elements; आधारः_idx += तन्तवः_stride) {
        प्रति (uint64_t j = 0; j < प्रत्येकम्_thread; j++) {
            ध्रुव uint64_t idx = आधारः_idx + j;
            यदि (idx >= योग_elements) {
                भङ्ग;
            }

            // Index decoding
            ध्रुव int64_t i3   = idx / (M * N * ne2);
            ध्रुव int64_t rem3 = idx % (M * N * ne2);
            ध्रुव int64_t i2   = rem3 / (M * N);
            ध्रुव int64_t rem2 = rem3 % (M * N);
            ध्रुव int64_t n    = rem2 / M;
            ध्रुव int64_t m    = rem2 % M;

            ध्रुव int64_t i03 = i3 / r3, i02 = i2 / r2;
            ध्रुव int64_t i13 = (ne13 > 1) ? i3 : 0, i12 = (ne12 > 1) ? i2 : 0;

            भिन्न         योग_2     = 0.0f;
            ध्रुव भिन्न * f32_row = (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src0_दत्त + m * nb01 + i02 * nb02 + i03 * nb03);

            यदि (src1->type == GGML_TYPE_F32) {
                ध्रुव भिन्न * src1_दत्त = (ध्रुव भिन्न *) src1->दत्त;

                प्रति (int64_t kb = 0; kb < K_खण्डानि; kb++) {
                    ध्रुव भिन्न * b_col_ptr =
                        (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src1_दत्त + (kb * खण्डम्_आकार_3) * परिमाण(भिन्न) + n * nb11 +
                                         i12 * nb12 + i13 * nb13);
                    योग_2 += compute_खण्डम्_dot_गुणनफलम्_f32(&f32_row[kb * खण्डम्_आकार_3], b_col_ptr);
                }

                यदि (K_remainder > 0) {
                    ध्रुव int64_t विस्थापनम्    = K_खण्डानि * खण्डम्_आकार_3;
                    ध्रुव भिन्न * b_col_ptr = (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src1_दत्त + विस्थापनम् * परिमाण(भिन्न) +
                                                               n * nb11 + i12 * nb12 + i13 * nb13);
                    योग_2 += compute_खण्डम्_dot_गुणनफलम्_f32_partial(&f32_row[विस्थापनम्], b_col_ptr, K_remainder);
                }
            } अन्यथा {
                ध्रुव uint16_t * src1_दत्त = (ध्रुव uint16_t *) src1->दत्त;

                प्रति (int64_t kb = 0; kb < K_खण्डानि; kb++) {
                    ध्रुव uint16_t * b_col_ptr =
                        (ध्रुव uint16_t *) ((ध्रुव वर्ण *) src1_दत्त + (kb * खण्डम्_आकार_3) * परिमाण(uint16_t) + n * nb11 +
                                            i12 * nb12 + i13 * nb13);
                    योग_2 += compute_खण्डम्_dot_गुणनफलम्_f32_f16_partial(&f32_row[kb * खण्डम्_आकार_3], b_col_ptr, खण्डम्_आकार_3);
                }

                यदि (K_remainder > 0) {
                    ध्रुव int64_t    विस्थापनम् = K_खण्डानि * खण्डम्_आकार_3;
                    ध्रुव uint16_t * b_col_ptr =
                        (ध्रुव uint16_t *) ((ध्रुव वर्ण *) src1_दत्त + विस्थापनम् * परिमाण(uint16_t) + n * nb11 +
                                            i12 * nb12 + i13 * nb13);
                    योग_2 += compute_खण्डम्_dot_गुणनफलम्_f32_f16_partial(&f32_row[विस्थापनम्], b_col_ptr, K_remainder);
                }
            }

            // Atomic store for output
            चञ्चल भिन्न * c_element =
                (चञ्चल भिन्न *) ((वर्ण *) dst_दत्त + m * dst->nb[0] + n * nb1 + i2 * nb2 + i3 * nb3);
            atomic_store_f32(c_element, योग_2);
        }
    }

    प्रत्यागम 0;
}
