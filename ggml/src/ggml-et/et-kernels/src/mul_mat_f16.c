//******************************************************************************
// MUL_MAT Kernel
// Matrix multiplication: C[M,N] = A[M,K] * B[K,N]
//******************************************************************************

#ଅନ୍ତର୍ଭାଵ "block_ops.h"
#ଅନ୍ତର୍ଭାଵ "ggml_tensor.h"
#ଅନ୍ତର୍ଭାଵ "math_fp.h"
#ଅନ୍ତର୍ଭାଵ "platform.h"
#ଅନ୍ତର୍ଭାଵ "quants.h"

#ଅନ୍ତର୍ଭାଵ <stdint.h>

ପୂର୍ଣ୍ଣସଂଖ୍ୟା entry_point(ସଂରଚନା ggml_et_ଦ୍ୱିମିକ_params * params, ଶୂନ୍ୟ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ଯଦି (!kernel_env || params == 0 || ((uint64_t) params & 0x7) != 0) {
        ଫେରନ୍ତୁ -1;
    }

    // Thread coordination
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_threads = get_ଅଙ୍କ_threads(kernel_env->shire_mask);

    ଯଦି (thread_id < 0 || (thread_id & 1)) {
        ଫେରନ୍ତୁ 0;  // Skip odd threads to avoid resource contention
    }

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା effective_thread_id   = thread_id / 2;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା effective_ଅଙ୍କ_threads = (ଅଙ୍କ_threads + 1) / 2;

    // Extract tensor references
    ସଂରଚନା ggml_tensor * src0 = &params->src0;  // Weight matrix A (F16)
    ସଂରଚନା ggml_tensor * src1 = &params->src1;  // Activation matrix B (F16/F32)
    ସଂରଚନା ggml_tensor * dst  = &params->dst;   // Output matrix C (F32)

    // Generic non-matrix-engine path: F16 x (F16/F32) -> F32
    ଯଦି (src0->type != GGML_TYPE_F16 || (src1->type != GGML_TYPE_F16 && src1->type != GGML_TYPE_F32) ||
        dst->type != GGML_TYPE_F32) {
        ଫେରନ୍ତୁ -1;
    }

    ଅଚଳ uint16_t * src0_data = (ଅଚଳ uint16_t *) src0->data;
    ଦଶମିକ *          dst_data  = (ଦଶମିକ *) dst->data;

    // Dimensions and Strides
    ଅଚଳ int64_t K = src0->ne[0];
    ଅଚଳ int64_t M = src0->ne[1];
    ଅଚଳ int64_t N = src1->ne[1];

    ଅଚଳ int64_t ne02 = src0->ne[2], ne03 = src0->ne[3];
    ଅଚଳ int64_t ne12 = src1->ne[2], ne13 = src1->ne[3];
    ଅଚଳ int64_t ne2 = dst->ne[2], ne3 = dst->ne[3];

    ଅଚଳ size_t nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    ଅଚଳ size_t nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];
    ଅଚଳ size_t nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];

    // F16 specific block size (Usually QK_F16)
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା     block_size  = QK_F16;
    ଅଚଳ int64_t K_blocks    = K / block_size;
    ଅଚଳ int64_t K_remainder = K % block_size;

    // Threading distribution
    ଅଚଳ uint64_t total_elements = M * N * ne2 * ne3;
    ଅଚଳ uint64_t per_thread     = 16;
    ଅଚଳ uint64_t threads_stride = per_thread * effective_ଅଙ୍କ_threads;

    ଯଦି (effective_thread_id * per_thread >= total_elements) {
        ଫେରନ୍ତୁ 0;
    }

    // Broadcasting support
    ଅଚଳ int64_t r2 = ne12 / ne02;
    ଅଚଳ int64_t r3 = ne13 / ne03;

    ପାଇଁ (uint64_t base_idx = effective_thread_id * per_thread; base_idx < total_elements; base_idx += threads_stride) {
        ପାଇଁ (uint64_t j = 0; j < per_thread; j++) {
            ଅଚଳ uint64_t idx = base_idx + j;
            ଯଦି (idx >= total_elements) {
                ଭାଙ୍ଗ;
            }

            // Index decoding
            ଅଚଳ int64_t i3   = idx / (M * N * ne2);
            ଅଚଳ int64_t rem3 = idx % (M * N * ne2);
            ଅଚଳ int64_t i2   = rem3 / (M * N);
            ଅଚଳ int64_t rem2 = rem3 % (M * N);
            ଅଚଳ int64_t n    = rem2 / M;
            ଅଚଳ int64_t m    = rem2 % M;

            ଅଚଳ int64_t i03 = i3 / r3, i02 = i2 / r2;
            ଅଚଳ int64_t i13 = (ne13 > 1) ? i3 : 0, i12 = (ne12 > 1) ? i2 : 0;

            ଦଶମିକ            sum = 0.0f;
            ଅଚଳ uint16_t * f16_row =
                (ଅଚଳ uint16_t *) ((ଅଚଳ ଅକ୍ଷର *) src0_data + m * nb01 + i02 * nb02 + i03 * nb03);

            ଯଦି (src1->type == GGML_TYPE_F32) {
                ଅଚଳ ଦଶମିକ * src1_data = (ଅଚଳ ଦଶମିକ *) src1->data;

                ପାଇଁ (int64_t kb = 0; kb < K_blocks; kb++) {
                    ଅଚଳ ଦଶମିକ * b_col_ptr =
                        (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) src1_data + (kb * block_size) * ଆକାର(ଦଶମିକ) + n * nb11 +
                                         i12 * nb12 + i13 * nb13);
                    sum += compute_block_dot_product_f16_naive(&f16_row[kb * block_size], b_col_ptr);
                }

                ଯଦି (K_remainder > 0) {
                    ଅଚଳ int64_t offset    = K_blocks * block_size;
                    ଅଚଳ ଦଶମିକ * b_col_ptr = (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) src1_data + offset * ଆକାର(ଦଶମିକ) +
                                                               n * nb11 + i12 * nb12 + i13 * nb13);
                    sum += compute_block_dot_product_f16_partial(&f16_row[offset], b_col_ptr, K_remainder);
                }
            } ଅନ୍ୟଥା {
                ଅଚଳ uint16_t * src1_data = (ଅଚଳ uint16_t *) src1->data;

                ପାଇଁ (int64_t kb = 0; kb < K_blocks; kb++) {
                    ଅଚଳ uint16_t * b_col_ptr =
                        (ଅଚଳ uint16_t *) ((ଅଚଳ ଅକ୍ଷର *) src1_data + (kb * block_size) * ଆକାର(uint16_t) + n * nb11 +
                                            i12 * nb12 + i13 * nb13);
                    sum += compute_block_dot_product_f16_f16_partial(&f16_row[kb * block_size], b_col_ptr, block_size);
                }

                ଯଦି (K_remainder > 0) {
                    ଅଚଳ int64_t    offset = K_blocks * block_size;
                    ଅଚଳ uint16_t * b_col_ptr =
                        (ଅଚଳ uint16_t *) ((ଅଚଳ ଅକ୍ଷର *) src1_data + offset * ଆକାର(uint16_t) + n * nb11 +
                                            i12 * nb12 + i13 * nb13);
                    sum += compute_block_dot_product_f16_f16_partial(&f16_row[offset], b_col_ptr, K_remainder);
                }
            }

            // Atomic store for output
            ଅସ୍ଥିର ଦଶମିକ * c_element =
                (ଅସ୍ଥିର ଦଶମିକ *) ((ଅକ୍ଷର *) dst_data + m * dst->nb[0] + n * nb1 + i2 * nb2 + i3 * nb3);
            atomic_store_f32(c_element, sum);
        }
    }

    ଫେରନ୍ତୁ 0;
}
