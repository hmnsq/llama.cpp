#अन्तर्भाव "block_ops.h"
#अन्तर्भाव "ggml_tensor.h"
#अन्तर्भाव "platform.h"
#अन्तर्भाव "quants.h"

#अन्तर्भाव <etsoc/common/utils.h>
#अन्तर्भाव <stdint.h>
#अन्तर्भाव <stdio.h>

पूर्णाङ्क entry_point(संरचना ggml_et_द्विआधारी_params * params, शून्य * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    यदि (!kernel_env || params == 0 || ((uint64_t) params & 0x7) != 0) {
        प्रत्यागम -1;
    }

    // Thread coordination
    पूर्णाङ्क thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    पूर्णाङ्क अंक_threads = get_अंक_threads(kernel_env->shire_mask);

    यदि (thread_id < 0 || (thread_id & 1)) {
        प्रत्यागम 0;  // Skip odd threads to avoid resource contention
    }

    पूर्णाङ्क effective_thread_id   = thread_id / 2;
    पूर्णाङ्क effective_अंक_threads = (अंक_threads + 1) / 2;

    // Extract tensor references
    संरचना ggml_tensor * src0 = &params->src0;  // Weight matrix A (F32)
    संरचना ggml_tensor * src1 = &params->src1;  // Activation matrix B (F16/F32)
    संरचना ggml_tensor * dst  = &params->dst;   // Output matrix C (F32)

    // Generic non-matrix-engine path: F32 x (F16/F32) -> F32
    यदि (src0->type != GGML_TYPE_F32 || (src1->type != GGML_TYPE_F16 && src1->type != GGML_TYPE_F32) ||
        dst->type != GGML_TYPE_F32) {
        प्रत्यागम -1;
    }

    ध्रुव भिन्न * src0_data = (ध्रुव भिन्न *) src0->data;
    भिन्न *       dst_data  = (भिन्न *) dst->data;

    // Dimensions and Strides
    ध्रुव int64_t K = src0->ne[0];
    ध्रुव int64_t M = src0->ne[1];
    ध्रुव int64_t N = src1->ne[1];

    ध्रुव int64_t ne02 = src0->ne[2], ne03 = src0->ne[3];
    ध्रुव int64_t ne12 = src1->ne[2], ne13 = src1->ne[3];
    ध्रुव int64_t ne2 = dst->ne[2], ne3 = dst->ne[3];

    ध्रुव माप_प्रकार nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    ध्रुव माप_प्रकार nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];
    ध्रुव माप_प्रकार nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];

    // F32 specific block size and counts
    ध्रुव पूर्णाङ्क     block_size  = QK_F32;
    ध्रुव int64_t K_blocks    = K / block_size;
    ध्रुव int64_t K_remainder = K % block_size;

    // Threading distribution
    ध्रुव uint64_t total_elements = M * N * ne2 * ne3;
    ध्रुव uint64_t per_thread     = 16;
    ध्रुव uint64_t threads_stride = per_thread * effective_अंक_threads;

    यदि (effective_thread_id * per_thread >= total_elements) {
        प्रत्यागम 0;
    }

    // Broadcasting support
    ध्रुव int64_t r2 = ne12 / ne02;
    ध्रुव int64_t r3 = ne13 / ne03;

    प्रति (uint64_t base_idx = effective_thread_id * per_thread; base_idx < total_elements; base_idx += threads_stride) {
        प्रति (uint64_t j = 0; j < per_thread; j++) {
            ध्रुव uint64_t idx = base_idx + j;
            यदि (idx >= total_elements) {
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

            भिन्न         sum     = 0.0f;
            ध्रुव भिन्न * f32_row = (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src0_data + m * nb01 + i02 * nb02 + i03 * nb03);

            यदि (src1->type == GGML_TYPE_F32) {
                ध्रुव भिन्न * src1_data = (ध्रुव भिन्न *) src1->data;

                प्रति (int64_t kb = 0; kb < K_blocks; kb++) {
                    ध्रुव भिन्न * b_col_ptr =
                        (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src1_data + (kb * block_size) * परिमाण(भिन्न) + n * nb11 +
                                         i12 * nb12 + i13 * nb13);
                    sum += compute_block_dot_product_f32(&f32_row[kb * block_size], b_col_ptr);
                }

                यदि (K_remainder > 0) {
                    ध्रुव int64_t offset    = K_blocks * block_size;
                    ध्रुव भिन्न * b_col_ptr = (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src1_data + offset * परिमाण(भिन्न) +
                                                               n * nb11 + i12 * nb12 + i13 * nb13);
                    sum += compute_block_dot_product_f32_partial(&f32_row[offset], b_col_ptr, K_remainder);
                }
            } अन्यथा {
                ध्रुव uint16_t * src1_data = (ध्रुव uint16_t *) src1->data;

                प्रति (int64_t kb = 0; kb < K_blocks; kb++) {
                    ध्रुव uint16_t * b_col_ptr =
                        (ध्रुव uint16_t *) ((ध्रुव वर्ण *) src1_data + (kb * block_size) * परिमाण(uint16_t) + n * nb11 +
                                            i12 * nb12 + i13 * nb13);
                    sum += compute_block_dot_product_f32_f16_partial(&f32_row[kb * block_size], b_col_ptr, block_size);
                }

                यदि (K_remainder > 0) {
                    ध्रुव int64_t    offset = K_blocks * block_size;
                    ध्रुव uint16_t * b_col_ptr =
                        (ध्रुव uint16_t *) ((ध्रुव वर्ण *) src1_data + offset * परिमाण(uint16_t) + n * nb11 +
                                            i12 * nb12 + i13 * nb13);
                    sum += compute_block_dot_product_f32_f16_partial(&f32_row[offset], b_col_ptr, K_remainder);
                }
            }

            // Atomic store for output
            चञ्चल भिन्न * c_element =
                (चञ्चल भिन्न *) ((वर्ण *) dst_data + m * dst->nb[0] + n * nb1 + i2 * nb2 + i3 * nb3);
            atomic_store_f32(c_element, sum);
        }
    }

    प्रत्यागम 0;
}
