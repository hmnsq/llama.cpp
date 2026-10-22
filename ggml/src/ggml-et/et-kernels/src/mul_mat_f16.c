//******************************************************************************
// MUL_MAT Kernel
// Matrix multiplication: C[M,N] = A[M,K] * B[K,N]
//******************************************************************************

#അന്തര്ഭാവ "block_ops.h"
#അന്തര്ഭാവ "ggml_tensor.h"
#അന്തര്ഭാവ "math_fp.h"
#അന്തര്ഭാവ "platform.h"
#അന്തര്ഭാവ "quants.h"

#അന്തര്ഭാവ <stdint.h>

പൂർണ്ണസംഖ്യ entry_point(ഘടന ggml_et_ദ്വിമാന_params * params, ശൂന്യം * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    എങ്കിൽ (!kernel_env || params == 0 || ((uint64_t) params & 0x7) != 0) {
        തിരികെ -1;
    }

    // Thread coordination
    പൂർണ്ണസംഖ്യ thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    പൂർണ്ണസംഖ്യ അക്കം_threads = get_അക്കം_threads(kernel_env->shire_mask);

    എങ്കിൽ (thread_id < 0 || (thread_id & 1)) {
        തിരികെ 0;  // Skip odd threads to avoid resource contention
    }

    പൂർണ്ണസംഖ്യ effective_thread_id   = thread_id / 2;
    പൂർണ്ണസംഖ്യ effective_അക്കം_threads = (അക്കം_threads + 1) / 2;

    // Extract tensor references
    ഘടന ggml_tensor * src0 = &params->src0;  // Weight matrix A (F16)
    ഘടന ggml_tensor * src1 = &params->src1;  // Activation matrix B (F16/F32)
    ഘടന ggml_tensor * dst  = &params->dst;   // Output matrix C (F32)

    // Generic non-matrix-engine path: F16 x (F16/F32) -> F32
    എങ്കിൽ (src0->type != GGML_TYPE_F16 || (src1->type != GGML_TYPE_F16 && src1->type != GGML_TYPE_F32) ||
        dst->type != GGML_TYPE_F32) {
        തിരികെ -1;
    }

    സ്ഥിരം uint16_t * src0_data = (സ്ഥിരം uint16_t *) src0->data;
    ദശാംശം *          dst_data  = (ദശാംശം *) dst->data;

    // Dimensions and Strides
    സ്ഥിരം int64_t K = src0->ne[0];
    സ്ഥിരം int64_t M = src0->ne[1];
    സ്ഥിരം int64_t N = src1->ne[1];

    സ്ഥിരം int64_t ne02 = src0->ne[2], ne03 = src0->ne[3];
    സ്ഥിരം int64_t ne12 = src1->ne[2], ne13 = src1->ne[3];
    സ്ഥിരം int64_t ne2 = dst->ne[2], ne3 = dst->ne[3];

    സ്ഥിരം size_t nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    സ്ഥിരം size_t nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];
    സ്ഥിരം size_t nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];

    // F16 specific block size (Usually QK_F16)
    സ്ഥിരം പൂർണ്ണസംഖ്യ     block_size  = QK_F16;
    സ്ഥിരം int64_t K_blocks    = K / block_size;
    സ്ഥിരം int64_t K_remainder = K % block_size;

    // Threading distribution
    സ്ഥിരം uint64_t total_elements = M * N * ne2 * ne3;
    സ്ഥിരം uint64_t per_thread     = 16;
    സ്ഥിരം uint64_t threads_stride = per_thread * effective_അക്കം_threads;

    എങ്കിൽ (effective_thread_id * per_thread >= total_elements) {
        തിരികെ 0;
    }

    // Broadcasting support
    സ്ഥിരം int64_t r2 = ne12 / ne02;
    സ്ഥിരം int64_t r3 = ne13 / ne03;

    വേണ്ടി (uint64_t base_idx = effective_thread_id * per_thread; base_idx < total_elements; base_idx += threads_stride) {
        വേണ്ടി (uint64_t j = 0; j < per_thread; j++) {
            സ്ഥിരം uint64_t idx = base_idx + j;
            എങ്കിൽ (idx >= total_elements) {
                നിർത്തുക;
            }

            // Index decoding
            സ്ഥിരം int64_t i3   = idx / (M * N * ne2);
            സ്ഥിരം int64_t rem3 = idx % (M * N * ne2);
            സ്ഥിരം int64_t i2   = rem3 / (M * N);
            സ്ഥിരം int64_t rem2 = rem3 % (M * N);
            സ്ഥിരം int64_t n    = rem2 / M;
            സ്ഥിരം int64_t m    = rem2 % M;

            സ്ഥിരം int64_t i03 = i3 / r3, i02 = i2 / r2;
            സ്ഥിരം int64_t i13 = (ne13 > 1) ? i3 : 0, i12 = (ne12 > 1) ? i2 : 0;

            ദശാംശം            sum = 0.0f;
            സ്ഥിരം uint16_t * f16_row =
                (സ്ഥിരം uint16_t *) ((സ്ഥിരം അക്ഷരം *) src0_data + m * nb01 + i02 * nb02 + i03 * nb03);

            എങ്കിൽ (src1->type == GGML_TYPE_F32) {
                സ്ഥിരം ദശാംശം * src1_data = (സ്ഥിരം ദശാംശം *) src1->data;

                വേണ്ടി (int64_t kb = 0; kb < K_blocks; kb++) {
                    സ്ഥിരം ദശാംശം * b_col_ptr =
                        (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) src1_data + (kb * block_size) * വലുപ്പം(ദശാംശം) + n * nb11 +
                                         i12 * nb12 + i13 * nb13);
                    sum += compute_block_dot_product_f16_naive(&f16_row[kb * block_size], b_col_ptr);
                }

                എങ്കിൽ (K_remainder > 0) {
                    സ്ഥിരം int64_t offset    = K_blocks * block_size;
                    സ്ഥിരം ദശാംശം * b_col_ptr = (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) src1_data + offset * വലുപ്പം(ദശാംശം) +
                                                               n * nb11 + i12 * nb12 + i13 * nb13);
                    sum += compute_block_dot_product_f16_partial(&f16_row[offset], b_col_ptr, K_remainder);
                }
            } അല്ലെങ്കിൽ {
                സ്ഥിരം uint16_t * src1_data = (സ്ഥിരം uint16_t *) src1->data;

                വേണ്ടി (int64_t kb = 0; kb < K_blocks; kb++) {
                    സ്ഥിരം uint16_t * b_col_ptr =
                        (സ്ഥിരം uint16_t *) ((സ്ഥിരം അക്ഷരം *) src1_data + (kb * block_size) * വലുപ്പം(uint16_t) + n * nb11 +
                                            i12 * nb12 + i13 * nb13);
                    sum += compute_block_dot_product_f16_f16_partial(&f16_row[kb * block_size], b_col_ptr, block_size);
                }

                എങ്കിൽ (K_remainder > 0) {
                    സ്ഥിരം int64_t    offset = K_blocks * block_size;
                    സ്ഥിരം uint16_t * b_col_ptr =
                        (സ്ഥിരം uint16_t *) ((സ്ഥിരം അക്ഷരം *) src1_data + offset * വലുപ്പം(uint16_t) + n * nb11 +
                                            i12 * nb12 + i13 * nb13);
                    sum += compute_block_dot_product_f16_f16_partial(&f16_row[offset], b_col_ptr, K_remainder);
                }
            }

            // Atomic store for output
            അസ്ഥിര ദശാംശം * c_element =
                (അസ്ഥിര ദശാംശം *) ((അക്ഷരം *) dst_data + m * dst->nb[0] + n * nb1 + i2 * nb2 + i3 * nb3);
            atomic_store_f32(c_element, sum);
        }
    }

    തിരികെ 0;
}
