#அந்தர்பாவ "block_ops.h"
#அந்தர்பாவ "ggml_tensor.h"
#அந்தர்பாவ "platform.h"
#அந்தர்பாவ "quants.h"

#அந்தர்பாவ <etsoc/common/utils.h>
#அந்தர்பாவ <stdint.h>
#அந்தர்பாவ <stdio.h>

முழு entry_point(கட்டமைப்பு ggml_et_இருமம்_params * params, வெறுமை * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    எனில் (!kernel_env || params == 0 || ((uint64_t) params & 0x7) != 0) {
        திரும்பு -1;
    }

    // Thread coordination
    முழு thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    முழு இலக்கம்_threads = get_இலக்கம்_threads(kernel_env->shire_mask);

    எனில் (thread_id < 0 || (thread_id & 1)) {
        திரும்பு 0;  // Skip odd threads to avoid resource contention
    }

    முழு effective_thread_id   = thread_id / 2;
    முழு effective_இலக்கம்_threads = (இலக்கம்_threads + 1) / 2;

    // Extract tensor references
    கட்டமைப்பு ggml_tensor * src0 = &params->src0;  // Weight matrix A (F32)
    கட்டமைப்பு ggml_tensor * src1 = &params->src1;  // Activation matrix B (F16/F32)
    கட்டமைப்பு ggml_tensor * dst  = &params->dst;   // Output matrix C (F32)

    // Generic non-matrix-engine path: F32 x (F16/F32) -> F32
    எனில் (src0->type != GGML_TYPE_F32 || (src1->type != GGML_TYPE_F16 && src1->type != GGML_TYPE_F32) ||
        dst->type != GGML_TYPE_F32) {
        திரும்பு -1;
    }

    மாறிலி தசம * src0_data = (மாறிலி தசம *) src0->data;
    தசம *       dst_data  = (தசம *) dst->data;

    // Dimensions and Strides
    மாறிலி int64_t K = src0->ne[0];
    மாறிலி int64_t M = src0->ne[1];
    மாறிலி int64_t N = src1->ne[1];

    மாறிலி int64_t ne02 = src0->ne[2], ne03 = src0->ne[3];
    மாறிலி int64_t ne12 = src1->ne[2], ne13 = src1->ne[3];
    மாறிலி int64_t ne2 = dst->ne[2], ne3 = dst->ne[3];

    மாறிலி size_t nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    மாறிலி size_t nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];
    மாறிலி size_t nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];

    // F32 specific block size and counts
    மாறிலி முழு     block_size  = QK_F32;
    மாறிலி int64_t K_blocks    = K / block_size;
    மாறிலி int64_t K_remainder = K % block_size;

    // Threading distribution
    மாறிலி uint64_t total_elements = M * N * ne2 * ne3;
    மாறிலி uint64_t per_thread     = 16;
    மாறிலி uint64_t threads_stride = per_thread * effective_இலக்கம்_threads;

    எனில் (effective_thread_id * per_thread >= total_elements) {
        திரும்பு 0;
    }

    // Broadcasting support
    மாறிலி int64_t r2 = ne12 / ne02;
    மாறிலி int64_t r3 = ne13 / ne03;

    க்கு (uint64_t base_idx = effective_thread_id * per_thread; base_idx < total_elements; base_idx += threads_stride) {
        க்கு (uint64_t j = 0; j < per_thread; j++) {
            மாறிலி uint64_t idx = base_idx + j;
            எனில் (idx >= total_elements) {
                நிறுத்து;
            }

            // Index decoding
            மாறிலி int64_t i3   = idx / (M * N * ne2);
            மாறிலி int64_t rem3 = idx % (M * N * ne2);
            மாறிலி int64_t i2   = rem3 / (M * N);
            மாறிலி int64_t rem2 = rem3 % (M * N);
            மாறிலி int64_t n    = rem2 / M;
            மாறிலி int64_t m    = rem2 % M;

            மாறிலி int64_t i03 = i3 / r3, i02 = i2 / r2;
            மாறிலி int64_t i13 = (ne13 > 1) ? i3 : 0, i12 = (ne12 > 1) ? i2 : 0;

            தசம         sum     = 0.0f;
            மாறிலி தசம * f32_row = (மாறிலி தசம *) ((மாறிலி எழுத்து *) src0_data + m * nb01 + i02 * nb02 + i03 * nb03);

            எனில் (src1->type == GGML_TYPE_F32) {
                மாறிலி தசம * src1_data = (மாறிலி தசம *) src1->data;

                க்கு (int64_t kb = 0; kb < K_blocks; kb++) {
                    மாறிலி தசம * b_col_ptr =
                        (மாறிலி தசம *) ((மாறிலி எழுத்து *) src1_data + (kb * block_size) * அளவு(தசம) + n * nb11 +
                                         i12 * nb12 + i13 * nb13);
                    sum += compute_block_dot_product_f32(&f32_row[kb * block_size], b_col_ptr);
                }

                எனில் (K_remainder > 0) {
                    மாறிலி int64_t offset    = K_blocks * block_size;
                    மாறிலி தசம * b_col_ptr = (மாறிலி தசம *) ((மாறிலி எழுத்து *) src1_data + offset * அளவு(தசம) +
                                                               n * nb11 + i12 * nb12 + i13 * nb13);
                    sum += compute_block_dot_product_f32_partial(&f32_row[offset], b_col_ptr, K_remainder);
                }
            } இல்லையெனில் {
                மாறிலி uint16_t * src1_data = (மாறிலி uint16_t *) src1->data;

                க்கு (int64_t kb = 0; kb < K_blocks; kb++) {
                    மாறிலி uint16_t * b_col_ptr =
                        (மாறிலி uint16_t *) ((மாறிலி எழுத்து *) src1_data + (kb * block_size) * அளவு(uint16_t) + n * nb11 +
                                            i12 * nb12 + i13 * nb13);
                    sum += compute_block_dot_product_f32_f16_partial(&f32_row[kb * block_size], b_col_ptr, block_size);
                }

                எனில் (K_remainder > 0) {
                    மாறிலி int64_t    offset = K_blocks * block_size;
                    மாறிலி uint16_t * b_col_ptr =
                        (மாறிலி uint16_t *) ((மாறிலி எழுத்து *) src1_data + offset * அளவு(uint16_t) + n * nb11 +
                                            i12 * nb12 + i13 * nb13);
                    sum += compute_block_dot_product_f32_f16_partial(&f32_row[offset], b_col_ptr, K_remainder);
                }
            }

            // Atomic store for output
            நிலையற்ற தசம * c_element =
                (நிலையற்ற தசம *) ((எழுத்து *) dst_data + m * dst->nb[0] + n * nb1 + i2 * nb2 + i3 * nb3);
            atomic_store_f32(c_element, sum);
        }
    }

    திரும்பு 0;
}
