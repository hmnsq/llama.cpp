#অন্তর্ভাব "block_ops.h"
#অন্তর্ভাব "ggml_tensor.h"
#অন্তর্ভাব "platform.h"
#অন্তর্ভাব "quants.h"

#অন্তর্ভাব <etsoc/common/utils.h>
#অন্তর্ভাব <stdint.h>
#অন্তর্ভাব <stdio.h>

পূর্ণসংখ্যা entry_point(কাঠামো ggml_et_দ্বিমিক_params * params, শূন্য * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    যদি (!kernel_env || params == 0 || ((uint64_t) params & 0x7) != 0) {
        ফেরত -1;
    }

    // Thread coordination
    পূর্ণসংখ্যা thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    পূর্ণসংখ্যা অঙ্ক_threads = get_অঙ্ক_threads(kernel_env->shire_mask);

    যদি (thread_id < 0 || (thread_id & 1)) {
        ফেরত 0;  // Skip odd threads to avoid resource contention
    }

    পূর্ণসংখ্যা effective_thread_id   = thread_id / 2;
    পূর্ণসংখ্যা effective_অঙ্ক_threads = (অঙ্ক_threads + 1) / 2;

    // Extract tensor references
    কাঠামো ggml_tensor * src0 = &params->src0;  // Weight matrix A (F32)
    কাঠামো ggml_tensor * src1 = &params->src1;  // Activation matrix B (F16/F32)
    কাঠামো ggml_tensor * dst  = &params->dst;   // Output matrix C (F32)

    // Generic non-matrix-engine path: F32 x (F16/F32) -> F32
    যদি (src0->type != GGML_TYPE_F32 || (src1->type != GGML_TYPE_F16 && src1->type != GGML_TYPE_F32) ||
        dst->type != GGML_TYPE_F32) {
        ফেরত -1;
    }

    ধ্রুবক দশমিক * src0_তথ্য = (ধ্রুবক দশমিক *) src0->তথ্য;
    দশমিক *       dst_তথ্য  = (দশমিক *) dst->তথ্য;

    // Dimensions and Strides
    ধ্রুবক int64_t K = src0->ne[0];
    ধ্রুবক int64_t M = src0->ne[1];
    ধ্রুবক int64_t N = src1->ne[1];

    ধ্রুবক int64_t ne02 = src0->ne[2], ne03 = src0->ne[3];
    ধ্রুবক int64_t ne12 = src1->ne[2], ne13 = src1->ne[3];
    ধ্রুবক int64_t ne2 = dst->ne[2], ne3 = dst->ne[3];

    ধ্রুবক size_t nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    ধ্রুবক size_t nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];
    ধ্রুবক size_t nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];

    // F32 specific block size and counts
    ধ্রুবক পূর্ণসংখ্যা     block_আকার_2  = QK_F32;
    ধ্রুবক int64_t K_blocks    = K / block_আকার_2;
    ধ্রুবক int64_t K_remainder = K % block_আকার_2;

    // Threading distribution
    ধ্রুবক uint64_t যোগ_elements = M * N * ne2 * ne3;
    ধ্রুবক uint64_t per_thread     = 16;
    ধ্রুবক uint64_t threads_stride = per_thread * effective_অঙ্ক_threads;

    যদি (effective_thread_id * per_thread >= যোগ_elements) {
        ফেরত 0;
    }

    // Broadcasting support
    ধ্রুবক int64_t r2 = ne12 / ne02;
    ধ্রুবক int64_t r3 = ne13 / ne03;

    জন্য (uint64_t base_idx = effective_thread_id * per_thread; base_idx < যোগ_elements; base_idx += threads_stride) {
        জন্য (uint64_t j = 0; j < per_thread; j++) {
            ধ্রুবক uint64_t idx = base_idx + j;
            যদি (idx >= যোগ_elements) {
                বিরতি;
            }

            // Index decoding
            ধ্রুবক int64_t i3   = idx / (M * N * ne2);
            ধ্রুবক int64_t rem3 = idx % (M * N * ne2);
            ধ্রুবক int64_t i2   = rem3 / (M * N);
            ধ্রুবক int64_t rem2 = rem3 % (M * N);
            ধ্রুবক int64_t n    = rem2 / M;
            ধ্রুবক int64_t m    = rem2 % M;

            ধ্রুবক int64_t i03 = i3 / r3, i02 = i2 / r2;
            ধ্রুবক int64_t i13 = (ne13 > 1) ? i3 : 0, i12 = (ne12 > 1) ? i2 : 0;

            দশমিক         যোগ_2     = 0.0f;
            ধ্রুবক দশমিক * f32_row = (ধ্রুবক দশমিক *) ((ধ্রুবক অক্ষর *) src0_তথ্য + m * nb01 + i02 * nb02 + i03 * nb03);

            যদি (src1->type == GGML_TYPE_F32) {
                ধ্রুবক দশমিক * src1_তথ্য = (ধ্রুবক দশমিক *) src1->তথ্য;

                জন্য (int64_t kb = 0; kb < K_blocks; kb++) {
                    ধ্রুবক দশমিক * b_col_ptr =
                        (ধ্রুবক দশমিক *) ((ধ্রুবক অক্ষর *) src1_তথ্য + (kb * block_আকার_2) * আকার(দশমিক) + n * nb11 +
                                         i12 * nb12 + i13 * nb13);
                    যোগ_2 += compute_block_dot_product_f32(&f32_row[kb * block_আকার_2], b_col_ptr);
                }

                যদি (K_remainder > 0) {
                    ধ্রুবক int64_t offset    = K_blocks * block_আকার_2;
                    ধ্রুবক দশমিক * b_col_ptr = (ধ্রুবক দশমিক *) ((ধ্রুবক অক্ষর *) src1_তথ্য + offset * আকার(দশমিক) +
                                                               n * nb11 + i12 * nb12 + i13 * nb13);
                    যোগ_2 += compute_block_dot_product_f32_partial(&f32_row[offset], b_col_ptr, K_remainder);
                }
            } অন্যথা {
                ধ্রুবক uint16_t * src1_তথ্য = (ধ্রুবক uint16_t *) src1->তথ্য;

                জন্য (int64_t kb = 0; kb < K_blocks; kb++) {
                    ধ্রুবক uint16_t * b_col_ptr =
                        (ধ্রুবক uint16_t *) ((ধ্রুবক অক্ষর *) src1_তথ্য + (kb * block_আকার_2) * আকার(uint16_t) + n * nb11 +
                                            i12 * nb12 + i13 * nb13);
                    যোগ_2 += compute_block_dot_product_f32_f16_partial(&f32_row[kb * block_আকার_2], b_col_ptr, block_আকার_2);
                }

                যদি (K_remainder > 0) {
                    ধ্রুবক int64_t    offset = K_blocks * block_আকার_2;
                    ধ্রুবক uint16_t * b_col_ptr =
                        (ধ্রুবক uint16_t *) ((ধ্রুবক অক্ষর *) src1_তথ্য + offset * আকার(uint16_t) + n * nb11 +
                                            i12 * nb12 + i13 * nb13);
                    যোগ_2 += compute_block_dot_product_f32_f16_partial(&f32_row[offset], b_col_ptr, K_remainder);
                }
            }

            // Atomic store for output
            অস্থির দশমিক * c_element =
                (অস্থির দশমিক *) ((অক্ষর *) dst_তথ্য + m * dst->nb[0] + n * nb1 + i2 * nb2 + i3 * nb3);
            atomic_store_f32(c_element, যোগ_2);
        }
    }

    ফেরত 0;
}
