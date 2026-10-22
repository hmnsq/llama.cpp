//******************************************************************************
// MUL_MAT Kernel
// Matrix multiplication: C[M,N] = A[M,K] * B[K,N]
//******************************************************************************

#अन्तर्भाव "block_ops.h"
#अन्तर्भाव "ggml_tensor.h"
#अन्तर्भाव "math_fp.h"
#अन्तर्भाव "platform.h"
#अन्तर्भाव "quants.h"

#अन्तर्भाव <stdint.h>

पूर्णांक entry_point(संरचना ggml_et_द्विमान_params * params, शून्य * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    जर (!kernel_env || params == 0 || ((uint64_t) params & 0x7) != 0) {
        परत -1;
    }

    // Thread coordination
    पूर्णांक thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    पूर्णांक अंक_threads = get_अंक_threads(kernel_env->shire_mask);

    जर (thread_id < 0 || (thread_id & 1)) {
        परत 0;  // Skip odd threads to avoid resource contention
    }

    पूर्णांक effective_thread_id   = thread_id / 2;
    पूर्णांक effective_अंक_threads = (अंक_threads + 1) / 2;

    // Extract tensor references
    संरचना ggml_tensor * src0 = &params->src0;  // Weight matrix A (F16)
    संरचना ggml_tensor * src1 = &params->src1;  // Activation matrix B (F16/F32)
    संरचना ggml_tensor * dst  = &params->dst;   // Output matrix C (F32)

    // Generic non-matrix-engine path: F16 x (F16/F32) -> F32
    जर (src0->type != GGML_TYPE_F16 || (src1->type != GGML_TYPE_F16 && src1->type != GGML_TYPE_F32) ||
        dst->type != GGML_TYPE_F32) {
        परत -1;
    }

    स्थिरांक uint16_t * src0_data = (स्थिरांक uint16_t *) src0->data;
    दशांश *          dst_data  = (दशांश *) dst->data;

    // Dimensions and Strides
    स्थिरांक int64_t K = src0->ne[0];
    स्थिरांक int64_t M = src0->ne[1];
    स्थिरांक int64_t N = src1->ne[1];

    स्थिरांक int64_t ne02 = src0->ne[2], ne03 = src0->ne[3];
    स्थिरांक int64_t ne12 = src1->ne[2], ne13 = src1->ne[3];
    स्थिरांक int64_t ne2 = dst->ne[2], ne3 = dst->ne[3];

    स्थिरांक माप_प्रकार nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    स्थिरांक माप_प्रकार nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];
    स्थिरांक माप_प्रकार nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];

    // F16 specific block size (Usually QK_F16)
    स्थिरांक पूर्णांक     block_size  = QK_F16;
    स्थिरांक int64_t K_blocks    = K / block_size;
    स्थिरांक int64_t K_remainder = K % block_size;

    // Threading distribution
    स्थिरांक uint64_t total_elements = M * N * ne2 * ne3;
    स्थिरांक uint64_t per_thread     = 16;
    स्थिरांक uint64_t threads_stride = per_thread * effective_अंक_threads;

    जर (effective_thread_id * per_thread >= total_elements) {
        परत 0;
    }

    // Broadcasting support
    स्थिरांक int64_t r2 = ne12 / ne02;
    स्थिरांक int64_t r3 = ne13 / ne03;

    साठी (uint64_t base_idx = effective_thread_id * per_thread; base_idx < total_elements; base_idx += threads_stride) {
        साठी (uint64_t j = 0; j < per_thread; j++) {
            स्थिरांक uint64_t idx = base_idx + j;
            जर (idx >= total_elements) {
                थांबा;
            }

            // Index decoding
            स्थिरांक int64_t i3   = idx / (M * N * ne2);
            स्थिरांक int64_t rem3 = idx % (M * N * ne2);
            स्थिरांक int64_t i2   = rem3 / (M * N);
            स्थिरांक int64_t rem2 = rem3 % (M * N);
            स्थिरांक int64_t n    = rem2 / M;
            स्थिरांक int64_t m    = rem2 % M;

            स्थिरांक int64_t i03 = i3 / r3, i02 = i2 / r2;
            स्थिरांक int64_t i13 = (ne13 > 1) ? i3 : 0, i12 = (ne12 > 1) ? i2 : 0;

            दशांश            sum = 0.0f;
            स्थिरांक uint16_t * f16_row =
                (स्थिरांक uint16_t *) ((स्थिरांक अक्षर *) src0_data + m * nb01 + i02 * nb02 + i03 * nb03);

            जर (src1->type == GGML_TYPE_F32) {
                स्थिरांक दशांश * src1_data = (स्थिरांक दशांश *) src1->data;

                साठी (int64_t kb = 0; kb < K_blocks; kb++) {
                    स्थिरांक दशांश * b_col_ptr =
                        (स्थिरांक दशांश *) ((स्थिरांक अक्षर *) src1_data + (kb * block_size) * आकार(दशांश) + n * nb11 +
                                         i12 * nb12 + i13 * nb13);
                    sum += compute_block_dot_product_f16_naive(&f16_row[kb * block_size], b_col_ptr);
                }

                जर (K_remainder > 0) {
                    स्थिरांक int64_t offset    = K_blocks * block_size;
                    स्थिरांक दशांश * b_col_ptr = (स्थिरांक दशांश *) ((स्थिरांक अक्षर *) src1_data + offset * आकार(दशांश) +
                                                               n * nb11 + i12 * nb12 + i13 * nb13);
                    sum += compute_block_dot_product_f16_partial(&f16_row[offset], b_col_ptr, K_remainder);
                }
            } अन्यथा {
                स्थिरांक uint16_t * src1_data = (स्थिरांक uint16_t *) src1->data;

                साठी (int64_t kb = 0; kb < K_blocks; kb++) {
                    स्थिरांक uint16_t * b_col_ptr =
                        (स्थिरांक uint16_t *) ((स्थिरांक अक्षर *) src1_data + (kb * block_size) * आकार(uint16_t) + n * nb11 +
                                            i12 * nb12 + i13 * nb13);
                    sum += compute_block_dot_product_f16_f16_partial(&f16_row[kb * block_size], b_col_ptr, block_size);
                }

                जर (K_remainder > 0) {
                    स्थिरांक int64_t    offset = K_blocks * block_size;
                    स्थिरांक uint16_t * b_col_ptr =
                        (स्थिरांक uint16_t *) ((स्थिरांक अक्षर *) src1_data + offset * आकार(uint16_t) + n * nb11 +
                                            i12 * nb12 + i13 * nb13);
                    sum += compute_block_dot_product_f16_f16_partial(&f16_row[offset], b_col_ptr, K_remainder);
                }
            }

            // Atomic store for output
            अस्थिर दशांश * c_element =
                (अस्थिर दशांश *) ((अक्षर *) dst_data + m * dst->nb[0] + n * nb1 + i2 * nb2 + i3 * nb3);
            atomic_store_f32(c_element, sum);
        }
    }

    परत 0;
}
