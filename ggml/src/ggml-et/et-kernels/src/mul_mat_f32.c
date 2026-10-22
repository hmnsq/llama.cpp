#समावेश "block_ops.h"
#समावेश "ggml_tensor.h"
#समावेश "platform.h"
#समावेश "quants.h"

#समावेश <etsoc/common/utils.h>
#समावेश <stdint.h>
#समावेश <stdio.h>

पूर्णांक entry_point(काष्ठा ggml_et_द्विआधारी_params * params, व्योम * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    अगर (!kernel_env || params == 0 || ((uint64_t) params & 0x7) != 0) {
        वापस -1;
    }

    // Thread coordination
    पूर्णांक thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    पूर्णांक अंक_threads = get_अंक_threads(kernel_env->shire_mask);

    अगर (thread_id < 0 || (thread_id & 1)) {
        वापस 0;  // Skip odd threads to avoid resource contention
    }

    पूर्णांक effective_thread_id   = thread_id / 2;
    पूर्णांक effective_अंक_threads = (अंक_threads + 1) / 2;

    // Extract tensor references
    काष्ठा ggml_tensor * src0 = &params->src0;  // Weight matrix A (F32)
    काष्ठा ggml_tensor * src1 = &params->src1;  // Activation matrix B (F16/F32)
    काष्ठा ggml_tensor * dst  = &params->dst;   // Output matrix C (F32)

    // Generic non-matrix-engine path: F32 x (F16/F32) -> F32
    अगर (src0->type != GGML_TYPE_F32 || (src1->type != GGML_TYPE_F16 && src1->type != GGML_TYPE_F32) ||
        dst->type != GGML_TYPE_F32) {
        वापस -1;
    }

    स्थिर भग्न * src0_तथ्य = (स्थिर भग्न *) src0->तथ्य;
    भग्न *       dst_तथ्य  = (भग्न *) dst->तथ्य;

    // Dimensions and Strides
    स्थिर int64_t K = src0->ne[0];
    स्थिर int64_t M = src0->ne[1];
    स्थिर int64_t N = src1->ne[1];

    स्थिर int64_t ne02 = src0->ne[2], ne03 = src0->ne[3];
    स्थिर int64_t ne12 = src1->ne[2], ne13 = src1->ne[3];
    स्थिर int64_t ne2 = dst->ne[2], ne3 = dst->ne[3];

    स्थिर माप_प्रकार nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    स्थिर माप_प्रकार nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];
    स्थिर माप_प्रकार nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];

    // F32 specific block size and counts
    स्थिर पूर्णांक     block_आकार_2  = QK_F32;
    स्थिर int64_t K_blocks    = K / block_आकार_2;
    स्थिर int64_t K_remainder = K % block_आकार_2;

    // Threading distribution
    स्थिर uint64_t योग_elements = M * N * ne2 * ne3;
    स्थिर uint64_t per_thread     = 16;
    स्थिर uint64_t threads_stride = per_thread * effective_अंक_threads;

    अगर (effective_thread_id * per_thread >= योग_elements) {
        वापस 0;
    }

    // Broadcasting support
    स्थिर int64_t r2 = ne12 / ne02;
    स्थिर int64_t r3 = ne13 / ne03;

    क्रम (uint64_t base_idx = effective_thread_id * per_thread; base_idx < योग_elements; base_idx += threads_stride) {
        क्रम (uint64_t j = 0; j < per_thread; j++) {
            स्थिर uint64_t idx = base_idx + j;
            अगर (idx >= योग_elements) {
                अवरोध;
            }

            // Index decoding
            स्थिर int64_t i3   = idx / (M * N * ne2);
            स्थिर int64_t rem3 = idx % (M * N * ne2);
            स्थिर int64_t i2   = rem3 / (M * N);
            स्थिर int64_t rem2 = rem3 % (M * N);
            स्थिर int64_t n    = rem2 / M;
            स्थिर int64_t m    = rem2 % M;

            स्थिर int64_t i03 = i3 / r3, i02 = i2 / r2;
            स्थिर int64_t i13 = (ne13 > 1) ? i3 : 0, i12 = (ne12 > 1) ? i2 : 0;

            भग्न         योग_2     = 0.0f;
            स्थिर भग्न * f32_row = (स्थिर भग्न *) ((स्थिर अक्षर *) src0_तथ्य + m * nb01 + i02 * nb02 + i03 * nb03);

            अगर (src1->type == GGML_TYPE_F32) {
                स्थिर भग्न * src1_तथ्य = (स्थिर भग्न *) src1->तथ्य;

                क्रम (int64_t kb = 0; kb < K_blocks; kb++) {
                    स्थिर भग्न * b_col_ptr =
                        (स्थिर भग्न *) ((स्थिर अक्षर *) src1_तथ्य + (kb * block_आकार_2) * माप(भग्न) + n * nb11 +
                                         i12 * nb12 + i13 * nb13);
                    योग_2 += compute_block_dot_product_f32(&f32_row[kb * block_आकार_2], b_col_ptr);
                }

                अगर (K_remainder > 0) {
                    स्थिर int64_t offset    = K_blocks * block_आकार_2;
                    स्थिर भग्न * b_col_ptr = (स्थिर भग्न *) ((स्थिर अक्षर *) src1_तथ्य + offset * माप(भग्न) +
                                                               n * nb11 + i12 * nb12 + i13 * nb13);
                    योग_2 += compute_block_dot_product_f32_partial(&f32_row[offset], b_col_ptr, K_remainder);
                }
            } अन्यथा {
                स्थिर uint16_t * src1_तथ्य = (स्थिर uint16_t *) src1->तथ्य;

                क्रम (int64_t kb = 0; kb < K_blocks; kb++) {
                    स्थिर uint16_t * b_col_ptr =
                        (स्थिर uint16_t *) ((स्थिर अक्षर *) src1_तथ्य + (kb * block_आकार_2) * माप(uint16_t) + n * nb11 +
                                            i12 * nb12 + i13 * nb13);
                    योग_2 += compute_block_dot_product_f32_f16_partial(&f32_row[kb * block_आकार_2], b_col_ptr, block_आकार_2);
                }

                अगर (K_remainder > 0) {
                    स्थिर int64_t    offset = K_blocks * block_आकार_2;
                    स्थिर uint16_t * b_col_ptr =
                        (स्थिर uint16_t *) ((स्थिर अक्षर *) src1_तथ्य + offset * माप(uint16_t) + n * nb11 +
                                            i12 * nb12 + i13 * nb13);
                    योग_2 += compute_block_dot_product_f32_f16_partial(&f32_row[offset], b_col_ptr, K_remainder);
                }
            }

            // Atomic store for output
            अस्थिर भग्न * c_element =
                (अस्थिर भग्न *) ((अक्षर *) dst_तथ्य + m * dst->nb[0] + n * nb1 + i2 * nb2 + i3 * nb3);
            atomic_store_f32(c_element, योग_2);
        }
    }

    वापस 0;
}
