//******************************************************************************
// MUL_MAT Kernel
// Matrix multiplication: C[M,N] = A[M,K] * B[K,N]
//******************************************************************************

#অন্তর্ভাব "block_ops.h"
#অন্তর্ভাব "ggml_tensor.h"
#অন্তর্ভাব "math_fp.h"
#অন্তর্ভাব "platform.h"
#অন্তর্ভাব "quants.h"

#অন্তর্ভাব <stdint.h>

পূর্ণাঙ্ক entry_point(সংরচনা ggml_et_দ্বিমিক_params * params, শূন্য * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    যদি (!kernel_env || params == 0 || ((uint64_t) params & 0x7) != 0) {
        প্রত্যাগম -1;
    }

    // Thread coordination
    পূর্ণাঙ্ক thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    পূর্ণাঙ্ক অঙ্ক_threads = get_অঙ্ক_threads(kernel_env->shire_mask);

    যদি (thread_id < 0 || (thread_id & 1)) {
        প্রত্যাগম 0;  // Skip odd threads to avoid resource contention
    }

    পূর্ণাঙ্ক effective_thread_id   = thread_id / 2;
    পূর্ণাঙ্ক effective_অঙ্ক_threads = (অঙ্ক_threads + 1) / 2;

    // Extract tensor references
    সংরচনা ggml_tensor * src0 = &params->src0;  // Weight matrix A (F16)
    সংরচনা ggml_tensor * src1 = &params->src1;  // Activation matrix B (F16/F32)
    সংরচনা ggml_tensor * dst  = &params->dst;   // Output matrix C (F32)

    // Generic non-matrix-engine path: F16 x (F16/F32) -> F32
    যদি (src0->type != GGML_TYPE_F16 || (src1->type != GGML_TYPE_F16 && src1->type != GGML_TYPE_F32) ||
        dst->type != GGML_TYPE_F32) {
        প্রত্যাগম -1;
    }

    ধ্রুব uint16_t * src0_data = (ধ্রুব uint16_t *) src0->data;
    ভিন্ন *          dst_data  = (ভিন্ন *) dst->data;

    // Dimensions and Strides
    ধ্রুব int64_t K = src0->ne[0];
    ধ্রুব int64_t M = src0->ne[1];
    ধ্রুব int64_t N = src1->ne[1];

    ধ্রুব int64_t ne02 = src0->ne[2], ne03 = src0->ne[3];
    ধ্রুব int64_t ne12 = src1->ne[2], ne13 = src1->ne[3];
    ধ্রুব int64_t ne2 = dst->ne[2], ne3 = dst->ne[3];

    ধ্রুব size_t nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    ধ্রুব size_t nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];
    ধ্রুব size_t nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];

    // F16 specific block size (Usually QK_F16)
    ধ্রুব পূর্ণাঙ্ক     block_size  = QK_F16;
    ধ্রুব int64_t K_blocks    = K / block_size;
    ধ্রুব int64_t K_remainder = K % block_size;

    // Threading distribution
    ধ্রুব uint64_t total_elements = M * N * ne2 * ne3;
    ধ্রুব uint64_t per_thread     = 16;
    ধ্রুব uint64_t threads_stride = per_thread * effective_অঙ্ক_threads;

    যদি (effective_thread_id * per_thread >= total_elements) {
        প্রত্যাগম 0;
    }

    // Broadcasting support
    ধ্রুব int64_t r2 = ne12 / ne02;
    ধ্রুব int64_t r3 = ne13 / ne03;

    প্রতি (uint64_t base_idx = effective_thread_id * per_thread; base_idx < total_elements; base_idx += threads_stride) {
        প্রতি (uint64_t j = 0; j < per_thread; j++) {
            ধ্রুব uint64_t idx = base_idx + j;
            যদি (idx >= total_elements) {
                ভঙ্গ;
            }

            // Index decoding
            ধ্রুব int64_t i3   = idx / (M * N * ne2);
            ধ্রুব int64_t rem3 = idx % (M * N * ne2);
            ধ্রুব int64_t i2   = rem3 / (M * N);
            ধ্রুব int64_t rem2 = rem3 % (M * N);
            ধ্রুব int64_t n    = rem2 / M;
            ধ্রুব int64_t m    = rem2 % M;

            ধ্রুব int64_t i03 = i3 / r3, i02 = i2 / r2;
            ধ্রুব int64_t i13 = (ne13 > 1) ? i3 : 0, i12 = (ne12 > 1) ? i2 : 0;

            ভিন্ন            sum = 0.0f;
            ধ্রুব uint16_t * f16_row =
                (ধ্রুব uint16_t *) ((ধ্রুব বর্ণ *) src0_data + m * nb01 + i02 * nb02 + i03 * nb03);

            যদি (src1->type == GGML_TYPE_F32) {
                ধ্রুব ভিন্ন * src1_data = (ধ্রুব ভিন্ন *) src1->data;

                প্রতি (int64_t kb = 0; kb < K_blocks; kb++) {
                    ধ্রুব ভিন্ন * b_col_ptr =
                        (ধ্রুব ভিন্ন *) ((ধ্রুব বর্ণ *) src1_data + (kb * block_size) * পরিমাণ(ভিন্ন) + n * nb11 +
                                         i12 * nb12 + i13 * nb13);
                    sum += compute_block_dot_product_f16_naive(&f16_row[kb * block_size], b_col_ptr);
                }

                যদি (K_remainder > 0) {
                    ধ্রুব int64_t offset    = K_blocks * block_size;
                    ধ্রুব ভিন্ন * b_col_ptr = (ধ্রুব ভিন্ন *) ((ধ্রুব বর্ণ *) src1_data + offset * পরিমাণ(ভিন্ন) +
                                                               n * nb11 + i12 * nb12 + i13 * nb13);
                    sum += compute_block_dot_product_f16_partial(&f16_row[offset], b_col_ptr, K_remainder);
                }
            } অন্যথা {
                ধ্রুব uint16_t * src1_data = (ধ্রুব uint16_t *) src1->data;

                প্রতি (int64_t kb = 0; kb < K_blocks; kb++) {
                    ধ্রুব uint16_t * b_col_ptr =
                        (ধ্রুব uint16_t *) ((ধ্রুব বর্ণ *) src1_data + (kb * block_size) * পরিমাণ(uint16_t) + n * nb11 +
                                            i12 * nb12 + i13 * nb13);
                    sum += compute_block_dot_product_f16_f16_partial(&f16_row[kb * block_size], b_col_ptr, block_size);
                }

                যদি (K_remainder > 0) {
                    ধ্রুব int64_t    offset = K_blocks * block_size;
                    ধ্রুব uint16_t * b_col_ptr =
                        (ধ্রুব uint16_t *) ((ধ্রুব বর্ণ *) src1_data + offset * পরিমাণ(uint16_t) + n * nb11 +
                                            i12 * nb12 + i13 * nb13);
                    sum += compute_block_dot_product_f16_f16_partial(&f16_row[offset], b_col_ptr, K_remainder);
                }
            }

            // Atomic store for output
            চঞ্চল ভিন্ন * c_element =
                (চঞ্চল ভিন্ন *) ((বর্ণ *) dst_data + m * dst->nb[0] + n * nb1 + i2 * nb2 + i3 * nb3);
            atomic_store_f32(c_element, sum);
        }
    }

    প্রত্যাগম 0;
}
