#ਅਨ੍ਤਰ੍ਭਾਵ "block_ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml_tensor.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "platform.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "quants.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <etsoc/common/utils.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdio.h>

ਪੂਰਨ ਇੰਦਰਾਜ_ਬਿੰਦੂ(ਬਣਤਰ ggml_et_ਦੁਆਧਾਰੀ_params * params, ਖਾਲੀ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ਜੇ (!kernel_env || params == 0 || ((uint64_t) params & 0x7) != 0) {
        ਵਾਪਸ -1;
    }

    // Thread coordination
    ਪੂਰਨ thread_id   = get_relative_thread_id(kernel_env->shire_ਨਕਾਬ);
    ਪੂਰਨ ਅੰਕ_ਧਾਗੇ = get_ਅੰਕ_ਧਾਗੇ(kernel_env->shire_ਨਕਾਬ);

    ਜੇ (thread_id < 0 || (thread_id & 1)) {
        ਵਾਪਸ 0;  // Skip odd threads to avoid resource contention
    }

    ਪੂਰਨ effective_thread_id   = thread_id / 2;
    ਪੂਰਨ effective_ਅੰਕ_ਧਾਗੇ = (ਅੰਕ_ਧਾਗੇ + 1) / 2;

    // Extract tensor references
    ਬਣਤਰ ggml_tensor * src0 = &params->src0;  // Weight matrix A (F32)
    ਬਣਤਰ ggml_tensor * src1 = &params->src1;  // Activation matrix B (F16/F32)
    ਬਣਤਰ ggml_tensor * dst  = &params->dst;   // Output matrix C (F32)

    // Generic non-matrix-engine path: F32 x (F16/F32) -> F32
    ਜੇ (src0->type != GGML_TYPE_F32 || (src1->type != GGML_TYPE_F16 && src1->type != GGML_TYPE_F32) ||
        dst->type != GGML_TYPE_F32) {
        ਵਾਪਸ -1;
    }

    ਅਚਲ ਦਸ਼ਮਲਵ * src0_ਅੰਕੜੇ = (ਅਚਲ ਦਸ਼ਮਲਵ *) src0->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ *       dst_ਅੰਕੜੇ  = (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ;

    // Dimensions and Strides
    ਅਚਲ int64_t K = src0->ne[0];
    ਅਚਲ int64_t M = src0->ne[1];
    ਅਚਲ int64_t N = src1->ne[1];

    ਅਚਲ int64_t ne02 = src0->ne[2], ne03 = src0->ne[3];
    ਅਚਲ int64_t ne12 = src1->ne[2], ne13 = src1->ne[3];
    ਅਚਲ int64_t ne2 = dst->ne[2], ne3 = dst->ne[3];

    ਅਚਲ size_t nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    ਅਚਲ size_t nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];
    ਅਚਲ size_t nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];

    // F32 specific block size and counts
    ਅਚਲ ਪੂਰਨ     ਖੰਡ_ਮਾਪ_3  = QK_F32;
    ਅਚਲ int64_t K_ਖੰਡਾਂ    = K / ਖੰਡ_ਮਾਪ_3;
    ਅਚਲ int64_t K_remainder = K % ਖੰਡ_ਮਾਪ_3;

    // Threading distribution
    ਅਚਲ uint64_t ਕੁੱਲ_elements = M * N * ne2 * ne3;
    ਅਚਲ uint64_t ਪ੍ਰਤੀ_thread     = 16;
    ਅਚਲ uint64_t ਧਾਗੇ_stride = ਪ੍ਰਤੀ_thread * effective_ਅੰਕ_ਧਾਗੇ;

    ਜੇ (effective_thread_id * ਪ੍ਰਤੀ_thread >= ਕੁੱਲ_elements) {
        ਵਾਪਸ 0;
    }

    // Broadcasting support
    ਅਚਲ int64_t r2 = ne12 / ne02;
    ਅਚਲ int64_t r3 = ne13 / ne03;

    ਲਈ (uint64_t ਅਧਾਰ_idx = effective_thread_id * ਪ੍ਰਤੀ_thread; ਅਧਾਰ_idx < ਕੁੱਲ_elements; ਅਧਾਰ_idx += ਧਾਗੇ_stride) {
        ਲਈ (uint64_t j = 0; j < ਪ੍ਰਤੀ_thread; j++) {
            ਅਚਲ uint64_t idx = ਅਧਾਰ_idx + j;
            ਜੇ (idx >= ਕੁੱਲ_elements) {
                ਰੋਕੋ;
            }

            // Index decoding
            ਅਚਲ int64_t i3   = idx / (M * N * ne2);
            ਅਚਲ int64_t rem3 = idx % (M * N * ne2);
            ਅਚਲ int64_t i2   = rem3 / (M * N);
            ਅਚਲ int64_t rem2 = rem3 % (M * N);
            ਅਚਲ int64_t n    = rem2 / M;
            ਅਚਲ int64_t m    = rem2 % M;

            ਅਚਲ int64_t i03 = i3 / r3, i02 = i2 / r2;
            ਅਚਲ int64_t i13 = (ne13 > 1) ? i3 : 0, i12 = (ne12 > 1) ? i2 : 0;

            ਦਸ਼ਮਲਵ         ਜੋੜ     = 0.0f;
            ਅਚਲ ਦਸ਼ਮਲਵ * f32_ਪੰਕਤੀ = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) src0_ਅੰਕੜੇ + m * nb01 + i02 * nb02 + i03 * nb03);

            ਜੇ (src1->type == GGML_TYPE_F32) {
                ਅਚਲ ਦਸ਼ਮਲਵ * src1_ਅੰਕੜੇ = (ਅਚਲ ਦਸ਼ਮਲਵ *) src1->ਅੰਕੜੇ;

                ਲਈ (int64_t kb = 0; kb < K_ਖੰਡਾਂ; kb++) {
                    ਅਚਲ ਦਸ਼ਮਲਵ * b_ਖਾਨਾ_ptr =
                        (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) src1_ਅੰਕੜੇ + (kb * ਖੰਡ_ਮਾਪ_3) * ਆਕਾਰ(ਦਸ਼ਮਲਵ) + n * nb11 +
                                         i12 * nb12 + i13 * nb13);
                    ਜੋੜ += compute_ਖੰਡ_dot_ਗੁਣਨਫਲ_f32(&f32_ਪੰਕਤੀ[kb * ਖੰਡ_ਮਾਪ_3], b_ਖਾਨਾ_ptr);
                }

                ਜੇ (K_remainder > 0) {
                    ਅਚਲ int64_t ਵਿੱਥ    = K_ਖੰਡਾਂ * ਖੰਡ_ਮਾਪ_3;
                    ਅਚਲ ਦਸ਼ਮਲਵ * b_ਖਾਨਾ_ptr = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) src1_ਅੰਕੜੇ + ਵਿੱਥ * ਆਕਾਰ(ਦਸ਼ਮਲਵ) +
                                                               n * nb11 + i12 * nb12 + i13 * nb13);
                    ਜੋੜ += compute_ਖੰਡ_dot_ਗੁਣਨਫਲ_f32_partial(&f32_ਪੰਕਤੀ[ਵਿੱਥ], b_ਖਾਨਾ_ptr, K_remainder);
                }
            } ਨਹੀਂਤਾਂ {
                ਅਚਲ uint16_t * src1_ਅੰਕੜੇ = (ਅਚਲ uint16_t *) src1->ਅੰਕੜੇ;

                ਲਈ (int64_t kb = 0; kb < K_ਖੰਡਾਂ; kb++) {
                    ਅਚਲ uint16_t * b_ਖਾਨਾ_ptr =
                        (ਅਚਲ uint16_t *) ((ਅਚਲ ਅੱਖਰ *) src1_ਅੰਕੜੇ + (kb * ਖੰਡ_ਮਾਪ_3) * ਆਕਾਰ(uint16_t) + n * nb11 +
                                            i12 * nb12 + i13 * nb13);
                    ਜੋੜ += compute_ਖੰਡ_dot_ਗੁਣਨਫਲ_f32_f16_partial(&f32_ਪੰਕਤੀ[kb * ਖੰਡ_ਮਾਪ_3], b_ਖਾਨਾ_ptr, ਖੰਡ_ਮਾਪ_3);
                }

                ਜੇ (K_remainder > 0) {
                    ਅਚਲ int64_t    ਵਿੱਥ = K_ਖੰਡਾਂ * ਖੰਡ_ਮਾਪ_3;
                    ਅਚਲ uint16_t * b_ਖਾਨਾ_ptr =
                        (ਅਚਲ uint16_t *) ((ਅਚਲ ਅੱਖਰ *) src1_ਅੰਕੜੇ + ਵਿੱਥ * ਆਕਾਰ(uint16_t) + n * nb11 +
                                            i12 * nb12 + i13 * nb13);
                    ਜੋੜ += compute_ਖੰਡ_dot_ਗੁਣਨਫਲ_f32_f16_partial(&f32_ਪੰਕਤੀ[ਵਿੱਥ], b_ਖਾਨਾ_ptr, K_remainder);
                }
            }

            // Atomic store for output
            ਅਸਥਿਰ ਦਸ਼ਮਲਵ * c_element =
                (ਅਸਥਿਰ ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst_ਅੰਕੜੇ + m * dst->nb[0] + n * nb1 + i2 * nb2 + i3 * nb3);
            atomic_store_f32(c_element, ਜੋੜ);
        }
    }

    ਵਾਪਸ 0;
}
