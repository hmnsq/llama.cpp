//******************************************************************************
// MUL_MAT Kernel
// Matrix multiplication: C[M,N] = A[M,K] * B[K,N]
//******************************************************************************

#شامل "block_ops.h"
#شامل "ggml_tensor.h"
#شامل "math_fp.h"
#شامل "platform.h"
#شامل "quants.h"

#شامل <stdint.h>

صحیح entry_point(ساختار ggml_et_دودویی_params * params, تهی * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    اگر (!kernel_env || params == 0 || ((uint64_t) params & 0x7) != 0) {
        بازگشت -1;
    }

    // Thread coordination
    صحیح thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    صحیح رقم_threads = get_رقم_threads(kernel_env->shire_mask);

    اگر (thread_id < 0 || (thread_id & 1)) {
        بازگشت 0;  // Skip odd threads to avoid resource contention
    }

    صحیح effective_thread_id   = thread_id / 2;
    صحیح effective_رقم_threads = (رقم_threads + 1) / 2;

    // Extract tensor references
    ساختار ggml_tensor * src0 = &params->src0;  // Weight matrix A (F16)
    ساختار ggml_tensor * src1 = &params->src1;  // Activation matrix B (F16/F32)
    ساختار ggml_tensor * dst  = &params->dst;   // Output matrix C (F32)

    // Generic non-matrix-engine path: F16 x (F16/F32) -> F32
    اگر (src0->type != GGML_TYPE_F16 || (src1->type != GGML_TYPE_F16 && src1->type != GGML_TYPE_F32) ||
        dst->type != GGML_TYPE_F32) {
        بازگشت -1;
    }

    ثابت uint16_t * src0_data = (ثابت uint16_t *) src0->data;
    اعشاری *          dst_data  = (اعشاری *) dst->data;

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
    ثابت صحیح     block_size  = QK_F16;
    ثابت int64_t K_blocks    = K / block_size;
    ثابت int64_t K_remainder = K % block_size;

    // Threading distribution
    ثابت uint64_t total_elements = M * N * ne2 * ne3;
    ثابت uint64_t per_thread     = 16;
    ثابت uint64_t threads_stride = per_thread * effective_رقم_threads;

    اگر (effective_thread_id * per_thread >= total_elements) {
        بازگشت 0;
    }

    // Broadcasting support
    ثابت int64_t r2 = ne12 / ne02;
    ثابت int64_t r3 = ne13 / ne03;

    برای (uint64_t base_idx = effective_thread_id * per_thread; base_idx < total_elements; base_idx += threads_stride) {
        برای (uint64_t j = 0; j < per_thread; j++) {
            ثابت uint64_t idx = base_idx + j;
            اگر (idx >= total_elements) {
                بشکن;
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

            اعشاری            sum = 0.0f;
            ثابت uint16_t * f16_row =
                (ثابت uint16_t *) ((ثابت نویسه *) src0_data + m * nb01 + i02 * nb02 + i03 * nb03);

            اگر (src1->type == GGML_TYPE_F32) {
                ثابت اعشاری * src1_data = (ثابت اعشاری *) src1->data;

                برای (int64_t kb = 0; kb < K_blocks; kb++) {
                    ثابت اعشاری * b_col_ptr =
                        (ثابت اعشاری *) ((ثابت نویسه *) src1_data + (kb * block_size) * اندازه(اعشاری) + n * nb11 +
                                         i12 * nb12 + i13 * nb13);
                    sum += compute_block_dot_product_f16_naive(&f16_row[kb * block_size], b_col_ptr);
                }

                اگر (K_remainder > 0) {
                    ثابت int64_t offset    = K_blocks * block_size;
                    ثابت اعشاری * b_col_ptr = (ثابت اعشاری *) ((ثابت نویسه *) src1_data + offset * اندازه(اعشاری) +
                                                               n * nb11 + i12 * nb12 + i13 * nb13);
                    sum += compute_block_dot_product_f16_partial(&f16_row[offset], b_col_ptr, K_remainder);
                }
            } وگرنه {
                ثابت uint16_t * src1_data = (ثابت uint16_t *) src1->data;

                برای (int64_t kb = 0; kb < K_blocks; kb++) {
                    ثابت uint16_t * b_col_ptr =
                        (ثابت uint16_t *) ((ثابت نویسه *) src1_data + (kb * block_size) * اندازه(uint16_t) + n * nb11 +
                                            i12 * nb12 + i13 * nb13);
                    sum += compute_block_dot_product_f16_f16_partial(&f16_row[kb * block_size], b_col_ptr, block_size);
                }

                اگر (K_remainder > 0) {
                    ثابت int64_t    offset = K_blocks * block_size;
                    ثابت uint16_t * b_col_ptr =
                        (ثابت uint16_t *) ((ثابت نویسه *) src1_data + offset * اندازه(uint16_t) + n * nb11 +
                                            i12 * nb12 + i13 * nb13);
                    sum += compute_block_dot_product_f16_f16_partial(&f16_row[offset], b_col_ptr, K_remainder);
                }
            }

            // Atomic store for output
            فرار اعشاری * c_element =
                (فرار اعشاری *) ((نویسه *) dst_data + m * dst->nb[0] + n * nb1 + i2 * nb2 + i3 * nb3);
            atomic_store_f32(c_element, sum);
        }
    }

    بازگشت 0;
}
