//******************************************************************************
// MUL_MAT Kernel
// Matrix multiplication: C[M,N] = A[M,K] * B[K,N]
//******************************************************************************

#include "block_ops.h"
#include "ggml_tensor.h"
#include "math_fp.h"
#include "platform.h"
#include "quants.h"

#include <stdint.h>

整数 entry_point(struct ggml_et_binary_params * params, void * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    如果 (!kernel_env || params == 0 || ((uint64_t) params & 0x7) != 0) {
        返回 -1;
    }

    // Thread coordination
    整数 thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    整数 num_threads = get_num_threads(kernel_env->shire_mask);

    如果 (thread_id < 0 || (thread_id & 1)) {
        返回 0;  // Skip odd threads to avoid resource contention
    }

    整数 effective_thread_id   = thread_id / 2;
    整数 effective_num_threads = (num_threads + 1) / 2;

    // Extract tensor references
    struct ggml_tensor * src0 = &params->src0;  // Weight matrix A (F16)
    struct ggml_tensor * src1 = &params->src1;  // Activation matrix B (F16/F32)
    struct ggml_tensor * dst  = &params->dst;   // Output matrix C (F32)

    // Generic non-matrix-engine path: F16 x (F16/F32) -> F32
    如果 (src0->type != GGML_TYPE_F16 || (src1->type != GGML_TYPE_F16 && src1->type != GGML_TYPE_F32) ||
        dst->type != GGML_TYPE_F32) {
        返回 -1;
    }

    const uint16_t * src0_data = (const uint16_t *) src0->data;
    float *          dst_data  = (float *) dst->data;

    // Dimensions and Strides
    const int64_t K = src0->ne[0];
    const int64_t M = src0->ne[1];
    const int64_t N = src1->ne[1];

    const int64_t ne02 = src0->ne[2], ne03 = src0->ne[3];
    const int64_t ne12 = src1->ne[2], ne13 = src1->ne[3];
    const int64_t ne2 = dst->ne[2], ne3 = dst->ne[3];

    const size_t nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    const size_t nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];
    const size_t nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];

    // F16 specific block size (Usually QK_F16)
    const 整数     block_size  = QK_F16;
    const int64_t K_blocks    = K / block_size;
    const int64_t K_remainder = K % block_size;

    // Threading distribution
    const uint64_t total_elements = M * N * ne2 * ne3;
    const uint64_t per_thread     = 16;
    const uint64_t threads_stride = per_thread * effective_num_threads;

    如果 (effective_thread_id * per_thread >= total_elements) {
        返回 0;
    }

    // Broadcasting support
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    为 (uint64_t base_idx = effective_thread_id * per_thread; base_idx < total_elements; base_idx += threads_stride) {
        为 (uint64_t j = 0; j < per_thread; j++) {
            const uint64_t idx = base_idx + j;
            如果 (idx >= total_elements) {
                break;
            }

            // Index decoding
            const int64_t i3   = idx / (M * N * ne2);
            const int64_t rem3 = idx % (M * N * ne2);
            const int64_t i2   = rem3 / (M * N);
            const int64_t rem2 = rem3 % (M * N);
            const int64_t n    = rem2 / M;
            const int64_t m    = rem2 % M;

            const int64_t i03 = i3 / r3, i02 = i2 / r2;
            const int64_t i13 = (ne13 > 1) ? i3 : 0, i12 = (ne12 > 1) ? i2 : 0;

            float            sum = 0.0f;
            const uint16_t * f16_row =
                (const uint16_t *) ((const char *) src0_data + m * nb01 + i02 * nb02 + i03 * nb03);

            如果 (src1->type == GGML_TYPE_F32) {
                const float * src1_data = (const float *) src1->data;

                为 (int64_t kb = 0; kb < K_blocks; kb++) {
                    const float * b_col_ptr =
                        (const float *) ((const char *) src1_data + (kb * block_size) * sizeof(float) + n * nb11 +
                                         i12 * nb12 + i13 * nb13);
                    sum += compute_block_dot_product_f16_naive(&f16_row[kb * block_size], b_col_ptr);
                }

                如果 (K_remainder > 0) {
                    const int64_t offset    = K_blocks * block_size;
                    const float * b_col_ptr = (const float *) ((const char *) src1_data + offset * sizeof(float) +
                                                               n * nb11 + i12 * nb12 + i13 * nb13);
                    sum += compute_block_dot_product_f16_partial(&f16_row[offset], b_col_ptr, K_remainder);
                }
            } 否则 {
                const uint16_t * src1_data = (const uint16_t *) src1->data;

                为 (int64_t kb = 0; kb < K_blocks; kb++) {
                    const uint16_t * b_col_ptr =
                        (const uint16_t *) ((const char *) src1_data + (kb * block_size) * sizeof(uint16_t) + n * nb11 +
                                            i12 * nb12 + i13 * nb13);
                    sum += compute_block_dot_product_f16_f16_partial(&f16_row[kb * block_size], b_col_ptr, block_size);
                }

                如果 (K_remainder > 0) {
                    const int64_t    offset = K_blocks * block_size;
                    const uint16_t * b_col_ptr =
                        (const uint16_t *) ((const char *) src1_data + offset * sizeof(uint16_t) + n * nb11 +
                                            i12 * nb12 + i13 * nb13);
                    sum += compute_block_dot_product_f16_f16_partial(&f16_row[offset], b_col_ptr, K_remainder);
                }
            }

            // Atomic store for output
            volatile float * c_element =
                (volatile float *) ((char *) dst_data + m * dst->nb[0] + n * nb1 + i2 * nb2 + i3 * nb3);
            atomic_store_f32(c_element, sum);
        }
    }

    返回 0;
}
