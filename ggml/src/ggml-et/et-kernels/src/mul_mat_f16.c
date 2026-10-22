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

صحیح اندراج_نقطہ(ساخت ggml_et_ثنائی_params * params, خالی * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    اگر (!kernel_env || params == 0 || ((uint64_t) params & 0x7) != 0) {
        واپس -1;
    }

    // Thread coordination
    صحیح thread_id   = get_relative_thread_id(kernel_env->shire_نقاب);
    صحیح عدد_دھاگے = get_عدد_دھاگے(kernel_env->shire_نقاب);

    اگر (thread_id < 0 || (thread_id & 1)) {
        واپس 0;  // Skip odd threads to avoid resource contention
    }

    صحیح effective_thread_id   = thread_id / 2;
    صحیح effective_عدد_دھاگے = (عدد_دھاگے + 1) / 2;

    // Extract tensor references
    ساخت ggml_tensor * src0 = &params->src0;  // Weight matrix A (F16)
    ساخت ggml_tensor * src1 = &params->src1;  // Activation matrix B (F16/F32)
    ساخت ggml_tensor * dst  = &params->dst;   // Output matrix C (F32)

    // Generic non-matrix-engine path: F16 x (F16/F32) -> F32
    اگر (src0->type != GGML_TYPE_F16 || (src1->type != GGML_TYPE_F16 && src1->type != GGML_TYPE_F32) ||
        dst->type != GGML_TYPE_F32) {
        واپس -1;
    }

    ثابت uint16_t * src0_ڈیٹا = (ثابت uint16_t *) src0->ڈیٹا;
    اعشاری *          dst_ڈیٹا  = (اعشاری *) dst->ڈیٹا;

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
    ثابت صحیح     بلاک_سائز_3  = QK_F16;
    ثابت int64_t K_بلاکیں    = K / بلاک_سائز_3;
    ثابت int64_t K_remainder = K % بلاک_سائز_3;

    // Threading distribution
    ثابت uint64_t کل_elements = M * N * ne2 * ne3;
    ثابت uint64_t فی_thread     = 16;
    ثابت uint64_t دھاگے_stride = فی_thread * effective_عدد_دھاگے;

    اگر (effective_thread_id * فی_thread >= کل_elements) {
        واپس 0;
    }

    // Broadcasting support
    ثابت int64_t r2 = ne12 / ne02;
    ثابت int64_t r3 = ne13 / ne03;

    برائے (uint64_t بنیاد_idx = effective_thread_id * فی_thread; بنیاد_idx < کل_elements; بنیاد_idx += دھاگے_stride) {
        برائے (uint64_t j = 0; j < فی_thread; j++) {
            ثابت uint64_t idx = بنیاد_idx + j;
            اگر (idx >= کل_elements) {
                توڑو;
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

            اعشاری            مجموع = 0.0f;
            ثابت uint16_t * f16_صف =
                (ثابت uint16_t *) ((ثابت حرف *) src0_ڈیٹا + m * nb01 + i02 * nb02 + i03 * nb03);

            اگر (src1->type == GGML_TYPE_F32) {
                ثابت اعشاری * src1_ڈیٹا = (ثابت اعشاری *) src1->ڈیٹا;

                برائے (int64_t kb = 0; kb < K_بلاکیں; kb++) {
                    ثابت اعشاری * b_ستون_ptr =
                        (ثابت اعشاری *) ((ثابت حرف *) src1_ڈیٹا + (kb * بلاک_سائز_3) * حجم(اعشاری) + n * nb11 +
                                         i12 * nb12 + i13 * nb13);
                    مجموع += compute_بلاک_dot_حاصل_ضرب_f16_naive(&f16_صف[kb * بلاک_سائز_3], b_ستون_ptr);
                }

                اگر (K_remainder > 0) {
                    ثابت int64_t انحراف    = K_بلاکیں * بلاک_سائز_3;
                    ثابت اعشاری * b_ستون_ptr = (ثابت اعشاری *) ((ثابت حرف *) src1_ڈیٹا + انحراف * حجم(اعشاری) +
                                                               n * nb11 + i12 * nb12 + i13 * nb13);
                    مجموع += compute_بلاک_dot_حاصل_ضرب_f16_partial(&f16_صف[انحراف], b_ستون_ptr, K_remainder);
                }
            } ورنہ {
                ثابت uint16_t * src1_ڈیٹا = (ثابت uint16_t *) src1->ڈیٹا;

                برائے (int64_t kb = 0; kb < K_بلاکیں; kb++) {
                    ثابت uint16_t * b_ستون_ptr =
                        (ثابت uint16_t *) ((ثابت حرف *) src1_ڈیٹا + (kb * بلاک_سائز_3) * حجم(uint16_t) + n * nb11 +
                                            i12 * nb12 + i13 * nb13);
                    مجموع += compute_بلاک_dot_حاصل_ضرب_f16_f16_partial(&f16_صف[kb * بلاک_سائز_3], b_ستون_ptr, بلاک_سائز_3);
                }

                اگر (K_remainder > 0) {
                    ثابت int64_t    انحراف = K_بلاکیں * بلاک_سائز_3;
                    ثابت uint16_t * b_ستون_ptr =
                        (ثابت uint16_t *) ((ثابت حرف *) src1_ڈیٹا + انحراف * حجم(uint16_t) + n * nb11 +
                                            i12 * nb12 + i13 * nb13);
                    مجموع += compute_بلاک_dot_حاصل_ضرب_f16_f16_partial(&f16_صف[انحراف], b_ستون_ptr, K_remainder);
                }
            }

            // Atomic store for output
            متلون اعشاری * c_element =
                (متلون اعشاری *) ((حرف *) dst_ڈیٹا + m * dst->nb[0] + n * nb1 + i2 * nb2 + i3 * nb3);
            atomic_store_f32(c_element, مجموع);
        }
    }

    واپس 0;
}
