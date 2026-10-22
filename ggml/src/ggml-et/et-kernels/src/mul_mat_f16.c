//******************************************************************************
// MUL_MAT Kernel
// Matrix multiplication: C[M,N] = A[M,K] * B[K,N]
//******************************************************************************

#ضمن "block_ops.h"
#ضمن "ggml_tensor.h"
#ضمن "math_fp.h"
#ضمن "platform.h"
#ضمن "quants.h"

#ضمن <stdint.h>

صحيح entry_point(بنية ggml_et_دووانەیی_params * params, فراغ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    إذا (!kernel_env || params == 0 || ((uint64_t) params & 0x7) != 0) {
        أرجع -1;
    }

    // Thread coordination
    صحيح thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    صحيح ژمارە_threads = get_ژمارە_threads(kernel_env->shire_mask);

    إذا (thread_id < 0 || (thread_id & 1)) {
        أرجع 0;  // Skip odd threads to avoid resource contention
    }

    صحيح effective_thread_id   = thread_id / 2;
    صحيح effective_ژمارە_threads = (ژمارە_threads + 1) / 2;

    // Extract tensor references
    بنية ggml_tensor * src0 = &params->src0;  // Weight matrix A (F16)
    بنية ggml_tensor * src1 = &params->src1;  // Activation matrix B (F16/F32)
    بنية ggml_tensor * dst  = &params->dst;   // Output matrix C (F32)

    // Generic non-matrix-engine path: F16 x (F16/F32) -> F32
    إذا (src0->type != GGML_TYPE_F16 || (src1->type != GGML_TYPE_F16 && src1->type != GGML_TYPE_F32) ||
        dst->type != GGML_TYPE_F32) {
        أرجع -1;
    }

    ثابت uint16_t * src0_data = (ثابت uint16_t *) src0->data;
    عائم *          dst_data  = (عائم *) dst->data;

    // Dimensions and Strides
    ثابت int64_t K = src0->ne[0];
    ثابت int64_t M = src0->ne[1];
    ثابت int64_t N = src1->ne[1];

    ثابت int64_t ne02 = src0->ne[2], ne03 = src0->ne[3];
    ثابت int64_t ne12 = src1->ne[2], ne13 = src1->ne[3];
    ثابت int64_t ne2 = dst->ne[2], ne3 = dst->ne[3];

    ثابت size_t nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    ثابت size_t nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];
    ثابت size_t nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];

    // F16 specific block size (Usually QK_F16)
    ثابت صحيح     block_size  = QK_F16;
    ثابت int64_t K_blocks    = K / block_size;
    ثابت int64_t K_remainder = K % block_size;

    // Threading distribution
    ثابت uint64_t total_elements = M * N * ne2 * ne3;
    ثابت uint64_t per_thread     = 16;
    ثابت uint64_t threads_stride = per_thread * effective_ژمارە_threads;

    إذا (effective_thread_id * per_thread >= total_elements) {
        أرجع 0;
    }

    // Broadcasting support
    ثابت int64_t r2 = ne12 / ne02;
    ثابت int64_t r3 = ne13 / ne03;

    لكل (uint64_t base_idx = effective_thread_id * per_thread; base_idx < total_elements; base_idx += threads_stride) {
        لكل (uint64_t j = 0; j < per_thread; j++) {
            ثابت uint64_t idx = base_idx + j;
            إذا (idx >= total_elements) {
                اكسر;
            }

            // Index decoding
            ثابت int64_t i3   = idx / (M * N * ne2);
            ثابت int64_t rem3 = idx % (M * N * ne2);
            ثابت int64_t i2   = rem3 / (M * N);
            ثابت int64_t rem2 = rem3 % (M * N);
            ثابت int64_t n    = rem2 / M;
            ثابت int64_t m    = rem2 % M;

            ثابت int64_t i03 = i3 / r3, i02 = i2 / r2;
            ثابت int64_t i13 = (ne13 > 1) ? i3 : 0, i12 = (ne12 > 1) ? i2 : 0;

            عائم            sum = 0.0f;
            ثابت uint16_t * f16_row =
                (ثابت uint16_t *) ((ثابت حرف *) src0_data + m * nb01 + i02 * nb02 + i03 * nb03);

            إذا (src1->type == GGML_TYPE_F32) {
                ثابت عائم * src1_data = (ثابت عائم *) src1->data;

                لكل (int64_t kb = 0; kb < K_blocks; kb++) {
                    ثابت عائم * b_col_ptr =
                        (ثابت عائم *) ((ثابت حرف *) src1_data + (kb * block_size) * حجم(عائم) + n * nb11 +
                                         i12 * nb12 + i13 * nb13);
                    sum += compute_block_dot_product_f16_naive(&f16_row[kb * block_size], b_col_ptr);
                }

                إذا (K_remainder > 0) {
                    ثابت int64_t offset    = K_blocks * block_size;
                    ثابت عائم * b_col_ptr = (ثابت عائم *) ((ثابت حرف *) src1_data + offset * حجم(عائم) +
                                                               n * nb11 + i12 * nb12 + i13 * nb13);
                    sum += compute_block_dot_product_f16_partial(&f16_row[offset], b_col_ptr, K_remainder);
                }
            } وإلا {
                ثابت uint16_t * src1_data = (ثابت uint16_t *) src1->data;

                لكل (int64_t kb = 0; kb < K_blocks; kb++) {
                    ثابت uint16_t * b_col_ptr =
                        (ثابت uint16_t *) ((ثابت حرف *) src1_data + (kb * block_size) * حجم(uint16_t) + n * nb11 +
                                            i12 * nb12 + i13 * nb13);
                    sum += compute_block_dot_product_f16_f16_partial(&f16_row[kb * block_size], b_col_ptr, block_size);
                }

                إذا (K_remainder > 0) {
                    ثابت int64_t    offset = K_blocks * block_size;
                    ثابت uint16_t * b_col_ptr =
                        (ثابت uint16_t *) ((ثابت حرف *) src1_data + offset * حجم(uint16_t) + n * nb11 +
                                            i12 * nb12 + i13 * nb13);
                    sum += compute_block_dot_product_f16_f16_partial(&f16_row[offset], b_col_ptr, K_remainder);
                }
            }

            // Atomic store for output
            متطاير عائم * c_element =
                (متطاير عائم *) ((حرف *) dst_data + m * dst->nb[0] + n * nb1 + i2 * nb2 + i3 * nb3);
            atomic_store_f32(c_element, sum);
        }
    }

    أرجع 0;
}
