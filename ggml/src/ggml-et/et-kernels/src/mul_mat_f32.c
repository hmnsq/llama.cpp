#ಅನ್ತರ್ಭಾವ "block_ops.h"
#ಅನ್ತರ್ಭಾವ "ggml_tensor.h"
#ಅನ್ತರ್ಭಾವ "platform.h"
#ಅನ್ತರ್ಭಾವ "quants.h"

#ಅನ್ತರ್ಭಾವ <etsoc/common/utils.h>
#ಅನ್ತರ್ಭಾವ <stdint.h>
#ಅನ್ತರ್ಭಾವ <stdio.h>

ಪೂರ್ಣಾಙ್ಕ entry_point(ಸಂರಚನಾ ggml_et_ದ್ವಿಮಾನ_params * params, ಶೂನ್ಯ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ಯದಿ (!kernel_env || params == 0 || ((uint64_t) params & 0x7) != 0) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    // Thread coordination
    ಪೂರ್ಣಾಙ್ಕ thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_threads = get_ಅಂಕೆ_threads(kernel_env->shire_mask);

    ಯದಿ (thread_id < 0 || (thread_id & 1)) {
        ಪ್ರತ್ಯಾಗಮ 0;  // Skip odd threads to avoid resource contention
    }

    ಪೂರ್ಣಾಙ್ಕ effective_thread_id   = thread_id / 2;
    ಪೂರ್ಣಾಙ್ಕ effective_ಅಂಕೆ_threads = (ಅಂಕೆ_threads + 1) / 2;

    // Extract tensor references
    ಸಂರಚನಾ ggml_tensor * src0 = &params->src0;  // Weight matrix A (F32)
    ಸಂರಚನಾ ggml_tensor * src1 = &params->src1;  // Activation matrix B (F16/F32)
    ಸಂರಚನಾ ggml_tensor * dst  = &params->dst;   // Output matrix C (F32)

    // Generic non-matrix-engine path: F32 x (F16/F32) -> F32
    ಯದಿ (src0->type != GGML_TYPE_F32 || (src1->type != GGML_TYPE_F16 && src1->type != GGML_TYPE_F32) ||
        dst->type != GGML_TYPE_F32) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಧ್ರುವ ಭಿನ್ನ * src0_data = (ಧ್ರುವ ಭಿನ್ನ *) src0->data;
    ಭಿನ್ನ *       dst_data  = (ಭಿನ್ನ *) dst->data;

    // Dimensions and Strides
    ಧ್ರುವ int64_t K = src0->ne[0];
    ಧ್ರುವ int64_t M = src0->ne[1];
    ಧ್ರುವ int64_t N = src1->ne[1];

    ಧ್ರುವ int64_t ne02 = src0->ne[2], ne03 = src0->ne[3];
    ಧ್ರುವ int64_t ne12 = src1->ne[2], ne13 = src1->ne[3];
    ಧ್ರುವ int64_t ne2 = dst->ne[2], ne3 = dst->ne[3];

    ಧ್ರುವ size_t nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    ಧ್ರುವ size_t nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];
    ಧ್ರುವ size_t nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];

    // F32 specific block size and counts
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ     block_size  = QK_F32;
    ಧ್ರುವ int64_t K_blocks    = K / block_size;
    ಧ್ರುವ int64_t K_remainder = K % block_size;

    // Threading distribution
    ಧ್ರುವ uint64_t total_elements = M * N * ne2 * ne3;
    ಧ್ರುವ uint64_t per_thread     = 16;
    ಧ್ರುವ uint64_t threads_stride = per_thread * effective_ಅಂಕೆ_threads;

    ಯದಿ (effective_thread_id * per_thread >= total_elements) {
        ಪ್ರತ್ಯಾಗಮ 0;
    }

    // Broadcasting support
    ಧ್ರುವ int64_t r2 = ne12 / ne02;
    ಧ್ರುವ int64_t r3 = ne13 / ne03;

    ಪ್ರತಿ (uint64_t base_idx = effective_thread_id * per_thread; base_idx < total_elements; base_idx += threads_stride) {
        ಪ್ರತಿ (uint64_t j = 0; j < per_thread; j++) {
            ಧ್ರುವ uint64_t idx = base_idx + j;
            ಯದಿ (idx >= total_elements) {
                ಭಙ್ಗ;
            }

            // Index decoding
            ಧ್ರುವ int64_t i3   = idx / (M * N * ne2);
            ಧ್ರುವ int64_t rem3 = idx % (M * N * ne2);
            ಧ್ರುವ int64_t i2   = rem3 / (M * N);
            ಧ್ರುವ int64_t rem2 = rem3 % (M * N);
            ಧ್ರುವ int64_t n    = rem2 / M;
            ಧ್ರುವ int64_t m    = rem2 % M;

            ಧ್ರುವ int64_t i03 = i3 / r3, i02 = i2 / r2;
            ಧ್ರುವ int64_t i13 = (ne13 > 1) ? i3 : 0, i12 = (ne12 > 1) ? i2 : 0;

            ಭಿನ್ನ         sum     = 0.0f;
            ಧ್ರುವ ಭಿನ್ನ * f32_row = (ಧ್ರುವ ಭಿನ್ನ *) ((ಧ್ರುವ ವರ್ಣ *) src0_data + m * nb01 + i02 * nb02 + i03 * nb03);

            ಯದಿ (src1->type == GGML_TYPE_F32) {
                ಧ್ರುವ ಭಿನ್ನ * src1_data = (ಧ್ರುವ ಭಿನ್ನ *) src1->data;

                ಪ್ರತಿ (int64_t kb = 0; kb < K_blocks; kb++) {
                    ಧ್ರುವ ಭಿನ್ನ * b_col_ptr =
                        (ಧ್ರುವ ಭಿನ್ನ *) ((ಧ್ರುವ ವರ್ಣ *) src1_data + (kb * block_size) * ಪರಿಮಾಣ(ಭಿನ್ನ) + n * nb11 +
                                         i12 * nb12 + i13 * nb13);
                    sum += compute_block_dot_product_f32(&f32_row[kb * block_size], b_col_ptr);
                }

                ಯದಿ (K_remainder > 0) {
                    ಧ್ರುವ int64_t offset    = K_blocks * block_size;
                    ಧ್ರುವ ಭಿನ್ನ * b_col_ptr = (ಧ್ರುವ ಭಿನ್ನ *) ((ಧ್ರುವ ವರ್ಣ *) src1_data + offset * ಪರಿಮಾಣ(ಭಿನ್ನ) +
                                                               n * nb11 + i12 * nb12 + i13 * nb13);
                    sum += compute_block_dot_product_f32_partial(&f32_row[offset], b_col_ptr, K_remainder);
                }
            } ಅನ್ಯಥಾ {
                ಧ್ರುವ uint16_t * src1_data = (ಧ್ರುವ uint16_t *) src1->data;

                ಪ್ರತಿ (int64_t kb = 0; kb < K_blocks; kb++) {
                    ಧ್ರುವ uint16_t * b_col_ptr =
                        (ಧ್ರುವ uint16_t *) ((ಧ್ರುವ ವರ್ಣ *) src1_data + (kb * block_size) * ಪರಿಮಾಣ(uint16_t) + n * nb11 +
                                            i12 * nb12 + i13 * nb13);
                    sum += compute_block_dot_product_f32_f16_partial(&f32_row[kb * block_size], b_col_ptr, block_size);
                }

                ಯದಿ (K_remainder > 0) {
                    ಧ್ರುವ int64_t    offset = K_blocks * block_size;
                    ಧ್ರುವ uint16_t * b_col_ptr =
                        (ಧ್ರುವ uint16_t *) ((ಧ್ರುವ ವರ್ಣ *) src1_data + offset * ಪರಿಮಾಣ(uint16_t) + n * nb11 +
                                            i12 * nb12 + i13 * nb13);
                    sum += compute_block_dot_product_f32_f16_partial(&f32_row[offset], b_col_ptr, K_remainder);
                }
            }

            // Atomic store for output
            ಚಞ್ಚಲ ಭಿನ್ನ * c_element =
                (ಚಞ್ಚಲ ಭಿನ್ನ *) ((ವರ್ಣ *) dst_data + m * dst->nb[0] + n * nb1 + i2 * nb2 + i3 * nb3);
            atomic_store_f32(c_element, sum);
        }
    }

    ಪ್ರತ್ಯಾಗಮ 0;
}
