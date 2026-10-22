#અન્તર્ભાવ "block_ops.h"
#અન્તર્ભાવ "ggml_tensor.h"
#અન્તર્ભાવ "platform.h"
#અન્તર્ભાવ "quants.h"

#અન્તર્ભાવ <etsoc/common/utils.h>
#અન્તર્ભાવ <stdint.h>
#અન્તર્ભાવ <stdio.h>

પૂર્ણાંક entry_point(સંરચના ggml_et_દ્વિઅંકી_params * params, શૂન્ય * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    જો (!kernel_env || params == 0 || ((uint64_t) params & 0x7) != 0) {
        પાછા -1;
    }

    // Thread coordination
    પૂર્ણાંક thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    પૂર્ણાંક અંક_threads = get_અંક_threads(kernel_env->shire_mask);

    જો (thread_id < 0 || (thread_id & 1)) {
        પાછા 0;  // Skip odd threads to avoid resource contention
    }

    પૂર્ણાંક effective_thread_id   = thread_id / 2;
    પૂર્ણાંક effective_અંક_threads = (અંક_threads + 1) / 2;

    // Extract tensor references
    સંરચના ggml_tensor * src0 = &params->src0;  // Weight matrix A (F32)
    સંરચના ggml_tensor * src1 = &params->src1;  // Activation matrix B (F16/F32)
    સંરચના ggml_tensor * dst  = &params->dst;   // Output matrix C (F32)

    // Generic non-matrix-engine path: F32 x (F16/F32) -> F32
    જો (src0->type != GGML_TYPE_F32 || (src1->type != GGML_TYPE_F16 && src1->type != GGML_TYPE_F32) ||
        dst->type != GGML_TYPE_F32) {
        પાછા -1;
    }

    અચલ દશાંશ * src0_data = (અચલ દશાંશ *) src0->data;
    દશાંશ *       dst_data  = (દશાંશ *) dst->data;

    // Dimensions and Strides
    અચલ int64_t K = src0->ne[0];
    અચલ int64_t M = src0->ne[1];
    અચલ int64_t N = src1->ne[1];

    અચલ int64_t ne02 = src0->ne[2], ne03 = src0->ne[3];
    અચલ int64_t ne12 = src1->ne[2], ne13 = src1->ne[3];
    અચલ int64_t ne2 = dst->ne[2], ne3 = dst->ne[3];

    અચલ size_t nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    અચલ size_t nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];
    અચલ size_t nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];

    // F32 specific block size and counts
    અચલ પૂર્ણાંક     block_size  = QK_F32;
    અચલ int64_t K_blocks    = K / block_size;
    અચલ int64_t K_remainder = K % block_size;

    // Threading distribution
    અચલ uint64_t total_elements = M * N * ne2 * ne3;
    અચલ uint64_t per_thread     = 16;
    અચલ uint64_t threads_stride = per_thread * effective_અંક_threads;

    જો (effective_thread_id * per_thread >= total_elements) {
        પાછા 0;
    }

    // Broadcasting support
    અચલ int64_t r2 = ne12 / ne02;
    અચલ int64_t r3 = ne13 / ne03;

    માટે (uint64_t base_idx = effective_thread_id * per_thread; base_idx < total_elements; base_idx += threads_stride) {
        માટે (uint64_t j = 0; j < per_thread; j++) {
            અચલ uint64_t idx = base_idx + j;
            જો (idx >= total_elements) {
                તોડો;
            }

            // Index decoding
            અચલ int64_t i3   = idx / (M * N * ne2);
            અચલ int64_t rem3 = idx % (M * N * ne2);
            અચલ int64_t i2   = rem3 / (M * N);
            અચલ int64_t rem2 = rem3 % (M * N);
            અચલ int64_t n    = rem2 / M;
            અચલ int64_t m    = rem2 % M;

            અચલ int64_t i03 = i3 / r3, i02 = i2 / r2;
            અચલ int64_t i13 = (ne13 > 1) ? i3 : 0, i12 = (ne12 > 1) ? i2 : 0;

            દશાંશ         sum     = 0.0f;
            અચલ દશાંશ * f32_row = (અચલ દશાંશ *) ((અચલ અક્ષર *) src0_data + m * nb01 + i02 * nb02 + i03 * nb03);

            જો (src1->type == GGML_TYPE_F32) {
                અચલ દશાંશ * src1_data = (અચલ દશાંશ *) src1->data;

                માટે (int64_t kb = 0; kb < K_blocks; kb++) {
                    અચલ દશાંશ * b_col_ptr =
                        (અચલ દશાંશ *) ((અચલ અક્ષર *) src1_data + (kb * block_size) * કદ(દશાંશ) + n * nb11 +
                                         i12 * nb12 + i13 * nb13);
                    sum += compute_block_dot_product_f32(&f32_row[kb * block_size], b_col_ptr);
                }

                જો (K_remainder > 0) {
                    અચલ int64_t offset    = K_blocks * block_size;
                    અચલ દશાંશ * b_col_ptr = (અચલ દશાંશ *) ((અચલ અક્ષર *) src1_data + offset * કદ(દશાંશ) +
                                                               n * nb11 + i12 * nb12 + i13 * nb13);
                    sum += compute_block_dot_product_f32_partial(&f32_row[offset], b_col_ptr, K_remainder);
                }
            } નહીંતર {
                અચલ uint16_t * src1_data = (અચલ uint16_t *) src1->data;

                માટે (int64_t kb = 0; kb < K_blocks; kb++) {
                    અચલ uint16_t * b_col_ptr =
                        (અચલ uint16_t *) ((અચલ અક્ષર *) src1_data + (kb * block_size) * કદ(uint16_t) + n * nb11 +
                                            i12 * nb12 + i13 * nb13);
                    sum += compute_block_dot_product_f32_f16_partial(&f32_row[kb * block_size], b_col_ptr, block_size);
                }

                જો (K_remainder > 0) {
                    અચલ int64_t    offset = K_blocks * block_size;
                    અચલ uint16_t * b_col_ptr =
                        (અચલ uint16_t *) ((અચલ અક્ષર *) src1_data + offset * કદ(uint16_t) + n * nb11 +
                                            i12 * nb12 + i13 * nb13);
                    sum += compute_block_dot_product_f32_f16_partial(&f32_row[offset], b_col_ptr, K_remainder);
                }
            }

            // Atomic store for output
            અસ્થિર દશાંશ * c_element =
                (અસ્થિર દશાંશ *) ((અક્ષર *) dst_data + m * dst->nb[0] + n * nb1 + i2 * nb2 + i3 * nb3);
            atomic_store_f32(c_element, sum);
        }
    }

    પાછા 0;
}
