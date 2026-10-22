#ಅನ್ತರ್ಭಾವ "block_ops.h"
#ಅನ್ತರ್ಭಾವ "ggml_tensor.h"
#ಅನ್ತರ್ಭಾವ "platform.h"
#ಅನ್ತರ್ಭಾವ "quants.h"

#ಅನ್ತರ್ಭಾವ <etsoc/common/utils.h>
#ಅನ್ತರ್ಭಾವ <stdint.h>
#ಅನ್ತರ್ಭಾವ <stdio.h>

ಪೂರ್ಣಾಂಕ ನಮೂದು_ಬಿಂದು(ರಚನೆ ggml_et_ದ್ವಿಮಾನ_params * params, ಶೂನ್ಯ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ಆದರೆ (!kernel_env || params == 0 || ((uint64_t) params & 0x7) != 0) {
        ಹಿಂತಿರುಗಿ -1;
    }

    // Thread coordination
    ಪೂರ್ಣಾಂಕ thread_id   = get_relative_thread_id(kernel_env->shire_ಮುಖವಾಡ);
    ಪೂರ್ಣಾಂಕ ಅಂಕೆ_ಎಳೆಗಳು = get_ಅಂಕೆ_ಎಳೆಗಳು(kernel_env->shire_ಮುಖವಾಡ);

    ಆದರೆ (thread_id < 0 || (thread_id & 1)) {
        ಹಿಂತಿರುಗಿ 0;  // Skip odd threads to avoid resource contention
    }

    ಪೂರ್ಣಾಂಕ effective_thread_id   = thread_id / 2;
    ಪೂರ್ಣಾಂಕ effective_ಅಂಕೆ_ಎಳೆಗಳು = (ಅಂಕೆ_ಎಳೆಗಳು + 1) / 2;

    // Extract tensor references
    ರಚನೆ ggml_tensor * src0 = &params->src0;  // Weight matrix A (F32)
    ರಚನೆ ggml_tensor * src1 = &params->src1;  // Activation matrix B (F16/F32)
    ರಚನೆ ggml_tensor * dst  = &params->dst;   // Output matrix C (F32)

    // Generic non-matrix-engine path: F32 x (F16/F32) -> F32
    ಆದರೆ (src0->type != GGML_TYPE_F32 || (src1->type != GGML_TYPE_F16 && src1->type != GGML_TYPE_F32) ||
        dst->type != GGML_TYPE_F32) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src0_ದತ್ತಾಂಶ = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src0->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ *       dst_ದತ್ತಾಂಶ  = (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ;

    // Dimensions and Strides
    ಸ್ಥಿರಾಂಕ int64_t K = src0->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t M = src0->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t N = src1->ne[1];

    ಸ್ಥಿರಾಂಕ int64_t ne02 = src0->ne[2], ne03 = src0->ne[3];
    ಸ್ಥಿರಾಂಕ int64_t ne12 = src1->ne[2], ne13 = src1->ne[3];
    ಸ್ಥಿರಾಂಕ int64_t ne2 = dst->ne[2], ne3 = dst->ne[3];

    ಸ್ಥಿರಾಂಕ size_t nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    ಸ್ಥಿರಾಂಕ size_t nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];
    ಸ್ಥಿರಾಂಕ size_t nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];

    // F32 specific block size and counts
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ     ಖಂಡ_ಗಾತ್ರ_3  = QK_F32;
    ಸ್ಥಿರಾಂಕ int64_t K_ಖಂಡಗಳು    = K / ಖಂಡ_ಗಾತ್ರ_3;
    ಸ್ಥಿರಾಂಕ int64_t K_remainder = K % ಖಂಡ_ಗಾತ್ರ_3;

    // Threading distribution
    ಸ್ಥಿರಾಂಕ uint64_t ಒಟ್ಟು_elements = M * N * ne2 * ne3;
    ಸ್ಥಿರಾಂಕ uint64_t ಪ್ರತಿ_thread     = 16;
    ಸ್ಥಿರಾಂಕ uint64_t ಎಳೆಗಳು_stride = ಪ್ರತಿ_thread * effective_ಅಂಕೆ_ಎಳೆಗಳು;

    ಆದರೆ (effective_thread_id * ಪ್ರತಿ_thread >= ಒಟ್ಟು_elements) {
        ಹಿಂತಿರುಗಿ 0;
    }

    // Broadcasting support
    ಸ್ಥಿರಾಂಕ int64_t r2 = ne12 / ne02;
    ಸ್ಥಿರಾಂಕ int64_t r3 = ne13 / ne03;

    ಗಾಗಿ (uint64_t ತಳ_idx = effective_thread_id * ಪ್ರತಿ_thread; ತಳ_idx < ಒಟ್ಟು_elements; ತಳ_idx += ಎಳೆಗಳು_stride) {
        ಗಾಗಿ (uint64_t j = 0; j < ಪ್ರತಿ_thread; j++) {
            ಸ್ಥಿರಾಂಕ uint64_t idx = ತಳ_idx + j;
            ಆದರೆ (idx >= ಒಟ್ಟು_elements) {
                ನಿಲ್ಲಿಸು;
            }

            // Index decoding
            ಸ್ಥಿರಾಂಕ int64_t i3   = idx / (M * N * ne2);
            ಸ್ಥಿರಾಂಕ int64_t rem3 = idx % (M * N * ne2);
            ಸ್ಥಿರಾಂಕ int64_t i2   = rem3 / (M * N);
            ಸ್ಥಿರಾಂಕ int64_t rem2 = rem3 % (M * N);
            ಸ್ಥಿರಾಂಕ int64_t n    = rem2 / M;
            ಸ್ಥಿರಾಂಕ int64_t m    = rem2 % M;

            ಸ್ಥಿರಾಂಕ int64_t i03 = i3 / r3, i02 = i2 / r2;
            ಸ್ಥಿರಾಂಕ int64_t i13 = (ne13 > 1) ? i3 : 0, i12 = (ne12 > 1) ? i2 : 0;

            ದಶಮಾಂಶ         ಮೊತ್ತ     = 0.0f;
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * f32_ಅಡ್ಡಸಾಲು = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src0_ದತ್ತಾಂಶ + m * nb01 + i02 * nb02 + i03 * nb03);

            ಆದರೆ (src1->type == GGML_TYPE_F32) {
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src1_ದತ್ತಾಂಶ = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src1->ದತ್ತಾಂಶ;

                ಗಾಗಿ (int64_t kb = 0; kb < K_ಖಂಡಗಳು; kb++) {
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * b_ಲಂಬಸಾಲು_ptr =
                        (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src1_ದತ್ತಾಂಶ + (kb * ಖಂಡ_ಗಾತ್ರ_3) * ಗಾತ್ರ(ದಶಮಾಂಶ) + n * nb11 +
                                         i12 * nb12 + i13 * nb13);
                    ಮೊತ್ತ += compute_ಖಂಡ_dot_ಗುಣಲಬ್ಧ_f32(&f32_ಅಡ್ಡಸಾಲು[kb * ಖಂಡ_ಗಾತ್ರ_3], b_ಲಂಬಸಾಲು_ptr);
                }

                ಆದರೆ (K_remainder > 0) {
                    ಸ್ಥಿರಾಂಕ int64_t ಸ್ಥಾನಾಂತರ    = K_ಖಂಡಗಳು * ಖಂಡ_ಗಾತ್ರ_3;
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * b_ಲಂಬಸಾಲು_ptr = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src1_ದತ್ತಾಂಶ + ಸ್ಥಾನಾಂತರ * ಗಾತ್ರ(ದಶಮಾಂಶ) +
                                                               n * nb11 + i12 * nb12 + i13 * nb13);
                    ಮೊತ್ತ += compute_ಖಂಡ_dot_ಗುಣಲಬ್ಧ_f32_partial(&f32_ಅಡ್ಡಸಾಲು[ಸ್ಥಾನಾಂತರ], b_ಲಂಬಸಾಲು_ptr, K_remainder);
                }
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಸ್ಥಿರಾಂಕ uint16_t * src1_ದತ್ತಾಂಶ = (ಸ್ಥಿರಾಂಕ uint16_t *) src1->ದತ್ತಾಂಶ;

                ಗಾಗಿ (int64_t kb = 0; kb < K_ಖಂಡಗಳು; kb++) {
                    ಸ್ಥಿರಾಂಕ uint16_t * b_ಲಂಬಸಾಲು_ptr =
                        (ಸ್ಥಿರಾಂಕ uint16_t *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src1_ದತ್ತಾಂಶ + (kb * ಖಂಡ_ಗಾತ್ರ_3) * ಗಾತ್ರ(uint16_t) + n * nb11 +
                                            i12 * nb12 + i13 * nb13);
                    ಮೊತ್ತ += compute_ಖಂಡ_dot_ಗುಣಲಬ್ಧ_f32_f16_partial(&f32_ಅಡ್ಡಸಾಲು[kb * ಖಂಡ_ಗಾತ್ರ_3], b_ಲಂಬಸಾಲು_ptr, ಖಂಡ_ಗಾತ್ರ_3);
                }

                ಆದರೆ (K_remainder > 0) {
                    ಸ್ಥಿರಾಂಕ int64_t    ಸ್ಥಾನಾಂತರ = K_ಖಂಡಗಳು * ಖಂಡ_ಗಾತ್ರ_3;
                    ಸ್ಥಿರಾಂಕ uint16_t * b_ಲಂಬಸಾಲು_ptr =
                        (ಸ್ಥಿರಾಂಕ uint16_t *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src1_ದತ್ತಾಂಶ + ಸ್ಥಾನಾಂತರ * ಗಾತ್ರ(uint16_t) + n * nb11 +
                                            i12 * nb12 + i13 * nb13);
                    ಮೊತ್ತ += compute_ಖಂಡ_dot_ಗುಣಲಬ್ಧ_f32_f16_partial(&f32_ಅಡ್ಡಸಾಲು[ಸ್ಥಾನಾಂತರ], b_ಲಂಬಸಾಲು_ptr, K_remainder);
                }
            }

            // Atomic store for output
            ಅಸ್ಥಿರ ದಶಮಾಂಶ * c_element =
                (ಅಸ್ಥಿರ ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst_ದತ್ತಾಂಶ + m * dst->nb[0] + n * nb1 + i2 * nb2 + i3 * nb3);
            atomic_store_f32(c_element, ಮೊತ್ತ);
        }
    }

    ಹಿಂತಿರುಗಿ 0;
}
