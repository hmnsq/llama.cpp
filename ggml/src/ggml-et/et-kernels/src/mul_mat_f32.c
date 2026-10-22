#ضمن "block_ops.h"
#ضمن "ggml_tensor.h"
#ضمن "platform.h"
#ضمن "quants.h"

#ضمن <etsoc/common/utils.h>
#ضمن <stdint.h>
#ضمن <stdio.h>

صحيح ثبت_ټکی(بنية ggml_et_دوګونی_params * params, فراغ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    إذا (!kernel_env || params == 0 || ((uint64_t) params & 0x7) != 0) {
        أرجع -1;
    }

    // Thread coordination
    صحيح thread_id   = get_relative_thread_id(kernel_env->shire_ماسک);
    صحيح عدد_تارونه = get_عدد_تارونه(kernel_env->shire_ماسک);

    إذا (thread_id < 0 || (thread_id & 1)) {
        أرجع 0;  // Skip odd threads to avoid resource contention
    }

    صحيح effective_thread_id   = thread_id / 2;
    صحيح effective_عدد_تارونه = (عدد_تارونه + 1) / 2;

    // Extract tensor references
    بنية ggml_tensor * src0 = &params->src0;  // Weight matrix A (F32)
    بنية ggml_tensor * src1 = &params->src1;  // Activation matrix B (F16/F32)
    بنية ggml_tensor * dst  = &params->dst;   // Output matrix C (F32)

    // Generic non-matrix-engine path: F32 x (F16/F32) -> F32
    إذا (src0->type != GGML_TYPE_F32 || (src1->type != GGML_TYPE_F16 && src1->type != GGML_TYPE_F32) ||
        dst->type != GGML_TYPE_F32) {
        أرجع -1;
    }

    ثابت عائم * src0_معلومات = (ثابت عائم *) src0->معلومات;
    عائم *       dst_معلومات  = (عائم *) dst->معلومات;

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

    // F32 specific block size and counts
    ثابت صحيح     بلاک_اندازه_3  = QK_F32;
    ثابت int64_t K_بلاکونه    = K / بلاک_اندازه_3;
    ثابت int64_t K_remainder = K % بلاک_اندازه_3;

    // Threading distribution
    ثابت uint64_t ټول_elements = M * N * ne2 * ne3;
    ثابت uint64_t فی_thread     = 16;
    ثابت uint64_t تارونه_stride = فی_thread * effective_عدد_تارونه;

    إذا (effective_thread_id * فی_thread >= ټول_elements) {
        أرجع 0;
    }

    // Broadcasting support
    ثابت int64_t r2 = ne12 / ne02;
    ثابت int64_t r3 = ne13 / ne03;

    لكل (uint64_t بنسټ_idx = effective_thread_id * فی_thread; بنسټ_idx < ټول_elements; بنسټ_idx += تارونه_stride) {
        لكل (uint64_t j = 0; j < فی_thread; j++) {
            ثابت uint64_t idx = بنسټ_idx + j;
            إذا (idx >= ټول_elements) {
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

            عائم         مجموعه     = 0.0f;
            ثابت عائم * f32_لیکه = (ثابت عائم *) ((ثابت حرف *) src0_معلومات + m * nb01 + i02 * nb02 + i03 * nb03);

            إذا (src1->type == GGML_TYPE_F32) {
                ثابت عائم * src1_معلومات = (ثابت عائم *) src1->معلومات;

                لكل (int64_t kb = 0; kb < K_بلاکونه; kb++) {
                    ثابت عائم * b_ستون_ptr =
                        (ثابت عائم *) ((ثابت حرف *) src1_معلومات + (kb * بلاک_اندازه_3) * حجم(عائم) + n * nb11 +
                                         i12 * nb12 + i13 * nb13);
                    مجموعه += compute_بلاک_dot_حاصل_ضرب_f32(&f32_لیکه[kb * بلاک_اندازه_3], b_ستون_ptr);
                }

                إذا (K_remainder > 0) {
                    ثابت int64_t دننه_واټن    = K_بلاکونه * بلاک_اندازه_3;
                    ثابت عائم * b_ستون_ptr = (ثابت عائم *) ((ثابت حرف *) src1_معلومات + دننه_واټن * حجم(عائم) +
                                                               n * nb11 + i12 * nb12 + i13 * nb13);
                    مجموعه += compute_بلاک_dot_حاصل_ضرب_f32_partial(&f32_لیکه[دننه_واټن], b_ستون_ptr, K_remainder);
                }
            } وإلا {
                ثابت uint16_t * src1_معلومات = (ثابت uint16_t *) src1->معلومات;

                لكل (int64_t kb = 0; kb < K_بلاکونه; kb++) {
                    ثابت uint16_t * b_ستون_ptr =
                        (ثابت uint16_t *) ((ثابت حرف *) src1_معلومات + (kb * بلاک_اندازه_3) * حجم(uint16_t) + n * nb11 +
                                            i12 * nb12 + i13 * nb13);
                    مجموعه += compute_بلاک_dot_حاصل_ضرب_f32_f16_partial(&f32_لیکه[kb * بلاک_اندازه_3], b_ستون_ptr, بلاک_اندازه_3);
                }

                إذا (K_remainder > 0) {
                    ثابت int64_t    دننه_واټن = K_بلاکونه * بلاک_اندازه_3;
                    ثابت uint16_t * b_ستون_ptr =
                        (ثابت uint16_t *) ((ثابت حرف *) src1_معلومات + دننه_واټن * حجم(uint16_t) + n * nb11 +
                                            i12 * nb12 + i13 * nb13);
                    مجموعه += compute_بلاک_dot_حاصل_ضرب_f32_f16_partial(&f32_لیکه[دننه_واټن], b_ستون_ptr, K_remainder);
                }
            }

            // Atomic store for output
            متطاير عائم * c_element =
                (متطاير عائم *) ((حرف *) dst_معلومات + m * dst->nb[0] + n * nb1 + i2 * nb2 + i3 * nb3);
            atomic_store_f32(c_element, مجموعه);
        }
    }

    أرجع 0;
}
