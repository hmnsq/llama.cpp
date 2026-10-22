//******************************************************************************
// Bare Metal MUL_MAT_ID Kernel (Mixture of Experts)
//
// ALGORITHM:
// MUL_MAT_ID performs batched matrix multiplication with expert routing.
// Each output element selects which expert matrix to use based on an index tensor.
//
// INPUTS:
//   src0 (as):  Expert weight matrices [K, M, n_expert]
//               - Stack of n_expert matrices, each of size [K, M]
//   src1 (b):   Activation vectors [K, n_cols, batch]
//               - n_cols can be 1 (broadcast) or n_expert_used (per-expert inputs)
//   src2 (ids): Expert selection indices [n_expert_used, batch] (int32)
//               - For each (slot, batch), specifies which expert from src0 to use
//
// OUTPUT:
//   dst:        Result [M, n_expert_used, batch, 1]
//
// COMPUTATION:
//   For each output position (m, slot, batch):
//     expert_id = ids[slot, batch]           // Which expert to use (0..n_expert-1)
//     col_idx = slot % src1.ne[1]            // Which column in src1 (handles broadcasting)
//     dst[m, slot, batch] = dot_product(
//       src0[0:K, m, expert_id],             // Row m from selected expert matrix
//       src1[0:K, col_idx, batch]            // Column from activations (may broadcast)
//     )
//
// BROADCASTING:
//   - When src1.ne[1] == 1: All expert slots use the same activation column
//   - When src1.ne[1] == n_expert_used: Each slot has its own activation column
//   - General case: col_idx = slot % src1.ne[1] (modulo handles both cases)
//
// MATH NOTATION:
//   C[m, s, b] = Sum(k=0 to K-1) A[k, m, ids[s,b]] x B[k, s % ne11, b]
//   where:
//     m: [0, M)               - output feature index
//     s: [0, n_expert_used)   - expert slot index
//     b: [0, batch)           - batch index
//     k: [0, K)               - hidden dimension
//     ne11 = src1->ne[1]       - number of columns in src1
//******************************************************************************

#ضمن "block_ops.h"
#ضمن "ggml_tensor.h"
#ضمن "math_fp.h"
#ضمن "platform.h"
#ضمن "quants.h"

#ضمن <stdint.h>

// Main entry point for MUL_MAT_ID kernel (Mixture of Experts)
صحيح داخلا_نقطو(بنية ggml_et_mul_mat_id_params * params, فراغ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    إذا (!kernel_env) {
        أرجع -1;
    }

    // Get thread coordination info
    صحيح thread_id   = get_relative_thread_id(kernel_env->shire_نقاب);
    صحيح انگ_ڌاڳا = get_انگ_ڌاڳا(kernel_env->shire_نقاب);

    إذا (thread_id < 0) {
        أرجع -1;
    }

    // Use even threads only to avoid resource contention
    // Each minion has 2 threads sharing instruction/data cache, NOC to RAM, and FPU
    // Odd threads return immediately to avoid fighting for shared resources
    إذا (thread_id & 1) {
        أرجع 0;  // Odd thread - skip work
    }

    // Adjust thread count and ID for even-only threading
    صحيح effective_thread_id   = thread_id / 2;
    صحيح effective_انگ_ڌاڳا = (انگ_ڌاڳا + 1) / 2;  // Ceiling division

    // Validate params
    إذا (params == 0 || ((uint64_t) params & 0x7) != 0) {
        أرجع -1;
    }

    // Extract tensor references
    بنية ggml_tensor * src0 = &params->src0;  // Expert weight matrices [K, M, n_expert]
    بنية ggml_tensor * src1 = &params->src1;  // Activations [K, n_expert_used, batch]
    بنية ggml_tensor * src2 = &params->src2;  // Expert indices [n_expert_used, batch] (I32)
    بنية ggml_tensor * dst  = &params->dst;   // Output [M, n_expert_used, batch, 1]

    // Validate tensor types
    إذا (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32) {
        أرجع -1;
    }

    // Get data pointers
    ثابت فراغ *    src0_معلومات = src0->معلومات;                    // Expert matrices (Q8_0/F16/F32)
    ثابت عائم *   src1_معلومات = (ثابت عائم *) src1->معلومات;    // Activations (F32)
    ثابت int32_t * src2_معلومات = (ثابت int32_t *) src2->معلومات;  // Expert IDs (I32)
    عائم *         dst_معلومات  = (عائم *) dst->معلومات;           // Output (F32)

    إذا (!src0_معلومات || !src1_معلومات || !src2_معلومات || !dst_معلومات) {
        أرجع -1;
    }

    // Determine block size based on src0 type
    صحيح بلاڪ_سائيز_3;
    بدل (src0->type) {
        حالة GGML_TYPE_Q8_0:
            بلاڪ_سائيز_3 = QK8_0;
            اكسر;
        حالة GGML_TYPE_Q4_0:
            بلاڪ_سائيز_3 = QK4_0;
            اكسر;
        حالة GGML_TYPE_F16:
            بلاڪ_سائيز_3 = QK_F16;
            اكسر;
        حالة GGML_TYPE_F32:
            بلاڪ_سائيز_3 = QK_F32;
            اكسر;
        افتراضي:
            أرجع -1;
    }

    // Get dimensions
    // src0: [K, M, n_expert] - expert weight matrices
    // src1: [K, n_expert_used, batch] - activations
    // src2: [n_expert_used, batch] - expert indices
    // dst:  [M, n_expert_used, batch, 1] - output
    ثابت int64_t K             = src0->ne[0];  // Hidden dimension
    ثابت int64_t M             = src0->ne[1];  // Output features
    ثابت int64_t n_expert      = src0->ne[2];  // Number of experts
    ثابت int64_t n_expert_استعمال_ٿيل = src2->ne[0];  // Experts used per token
    ثابت int64_t batch         = src2->ne[1];  // Batch size

    // Strides (in bytes)
    ثابت size_t nb01 = src0->nb[1];  // src0 row stride
    ثابت size_t nb02 = src0->nb[2];  // src0 expert stride
    ثابت size_t nb11 = src1->nb[1];  // src1 column stride
    ثابت size_t nb12 = src1->nb[2];  // src1 batch stride
    ثابت size_t nb20 = src2->nb[0];  // src2 element stride
    ثابت size_t nb21 = src2->nb[1];  // src2 batch stride
    ثابت size_t nb1  = dst->nb[1];   // dst column stride
    ثابت size_t nb2  = dst->nb[2];   // dst batch stride

    // Verify K dimension alignment for quantization
    // Q8_0 requires strict alignment (quantized data must be block-aligned)
    // F32 and F16 can handle partial blocks with scalar remainders
    إذا ((src0->type == GGML_TYPE_Q8_0 || src0->type == GGML_TYPE_Q4_0) && K % بلاڪ_سائيز_3 != 0) {
        أرجع -1;  // Q8_0 requires K to be multiple of block_size
    }

    // Verify first dimension is contiguous
    size_t expected_element_سائيز_src0;
    إذا (src0->type == GGML_TYPE_Q8_0) {
        expected_element_سائيز_src0 = حجم(بلاڪ_q8_0);
    } وإلا إذا (src0->type == GGML_TYPE_Q4_0) {
        expected_element_سائيز_src0 = حجم(بلاڪ_q4_0);
    } وإلا إذا (src0->type == GGML_TYPE_F16) {
        expected_element_سائيز_src0 = حجم(uint16_t);
    } وإلا إذا (src0->type == GGML_TYPE_F32) {
        expected_element_سائيز_src0 = حجم(عائم);
    } وإلا {
        أرجع -1;
    }

    إذا (src0->nb[0] != expected_element_سائيز_src0 || src1->nb[0] != حجم(عائم) || src2->nb[0] != حجم(int32_t) ||
        dst->nb[0] != حجم(عائم)) {
        أرجع -1;
    }

    ثابت int64_t K_بلاڪون = K / بلاڪ_سائيز_3;

    // Threading: distribute output elements across threads
    // Total output elements = M * n_expert_used * batch
    ثابت uint64_t ڪل_elements = M * n_expert_استعمال_ٿيل * batch;

    ثابت uint64_t في_thread     = 16;
    ثابت uint64_t ڌاڳا_stride = في_thread * effective_انگ_ڌاڳا;

    إذا (effective_thread_id * في_thread >= ڪل_elements) {
        أرجع 0;
    }

    // Process elements assigned to this thread
    لكل (uint64_t بنياد_idx = effective_thread_id * في_thread; بنياد_idx < ڪل_elements; بنياد_idx += ڌاڳا_stride) {
        لكل (uint64_t j = 0; j < في_thread; j++) {
            ثابت uint64_t idx = بنياد_idx + j;

            إذا (idx >= ڪل_elements) {
                اكسر;
            }

            // Decode linear index to (m, n_idx, batch_idx)
            // Layout: m + M * (n_idx + n_expert_used * batch_idx)
            ثابت int64_t batch_idx = idx / (M * n_expert_استعمال_ٿيل);
            ثابت int64_t rem       = idx % (M * n_expert_استعمال_ٿيل);
            ثابت int64_t n_idx     = rem / M;
            ثابت int64_t m         = rem % M;

            // Get expert ID from src2[n_idx, batch_idx]
            ثابت int32_t expert_id = *(ثابت int32_t *) ((ثابت حرف *) src2_معلومات + n_idx * nb20 + batch_idx * nb21);

            // Validate expert ID
            إذا (expert_id < 0 || expert_id >= n_expert) {
                // Invalid expert ID - write zero and continue
                متطاير عائم * dst_element =
                    (متطاير عائم *) ((حرف *) dst_معلومات + m * dst->nb[0] + n_idx * nb1 + batch_idx * nb2);
                atomic_store_f32(dst_element, 0.0f);
                استمر;
            }

            // Compute dot product: expert_matrix[m, :] x activations[:, col_idx, batch_idx]
            // Use modulo to handle broadcasting: when src1 has fewer columns than expert slots,
            // multiple slots share the same activation column (col_idx = n_idx % src1->ne[1])
            ثابت int64_t کانو_idx = n_idx % src1->ne[1];
            عائم         جوڙ     = 0.0f;

            // Type switch hoisted outside block loop: one branch per element, not per block
            ثابت حرف * expert_صف_بنياد = (ثابت حرف *) src0_معلومات + m * nb01 + expert_id * nb02;

            بدل (src0->type) {
                حالة GGML_TYPE_Q8_0:
                    {
                        ثابت بلاڪ_q8_0 * q8_صف = (ثابت بلاڪ_q8_0 *) expert_صف_بنياد;
                        ثابت عائم *      b_کانو_بنياد =
                            (ثابت عائم *) ((ثابت حرف *) src1_معلومات + کانو_idx * nb11 + batch_idx * nb12);
                        جوڙ += compute_صف_dot_q8_0(q8_صف, b_کانو_بنياد, K_بلاڪون);
                        اكسر;
                    }
                حالة GGML_TYPE_Q4_0:
                    {
                        ثابت بلاڪ_q4_0 * q4_صف = (ثابت بلاڪ_q4_0 *) expert_صف_بنياد;
                        ثابت عائم *      b_کانو_بنياد =
                            (ثابت عائم *) ((ثابت حرف *) src1_معلومات + کانو_idx * nb11 + batch_idx * nb12);
                        جوڙ += compute_صف_dot_q4_0(q4_صف, b_کانو_بنياد, K_بلاڪون);
                        اكسر;
                    }
                حالة GGML_TYPE_F16:
                    {
                        ثابت uint16_t * f16_صف     = (ثابت uint16_t *) expert_صف_بنياد;
                        ثابت int64_t    K_remainder = K % بلاڪ_سائيز_3;
                        لكل (int64_t kb = 0; kb < K_بلاڪون; kb++) {
                            ثابت عائم * b_کانو_ptr =
                                (ثابت عائم *) ((ثابت حرف *) src1_معلومات + (kb * بلاڪ_سائيز_3) * حجم(عائم) +
                                                 کانو_idx * nb11 + batch_idx * nb12);
                            جوڙ += compute_بلاڪ_dot_حاصل_ضرب_f16_naive(&f16_صف[kb * بلاڪ_سائيز_3], b_کانو_ptr);
                        }
                        إذا (K_remainder > 0) {
                            ثابت int64_t وٿي = K_بلاڪون * بلاڪ_سائيز_3;
                            ثابت عائم * b_کانو_ptr =
                                (ثابت عائم *) ((ثابت حرف *) src1_معلومات + وٿي * حجم(عائم) + کانو_idx * nb11 +
                                                 batch_idx * nb12);
                            جوڙ += compute_بلاڪ_dot_حاصل_ضرب_f16_partial(&f16_صف[وٿي], b_کانو_ptr, K_remainder);
                        }
                        اكسر;
                    }
                حالة GGML_TYPE_F32:
                    {
                        ثابت عائم * f32_صف     = (ثابت عائم *) expert_صف_بنياد;
                        ثابت int64_t K_remainder = K % بلاڪ_سائيز_3;
                        لكل (int64_t kb = 0; kb < K_بلاڪون; kb++) {
                            ثابت عائم * b_کانو_ptr =
                                (ثابت عائم *) ((ثابت حرف *) src1_معلومات + (kb * بلاڪ_سائيز_3) * حجم(عائم) +
                                                 کانو_idx * nb11 + batch_idx * nb12);
                            جوڙ += compute_بلاڪ_dot_حاصل_ضرب_f32(&f32_صف[kb * بلاڪ_سائيز_3], b_کانو_ptr);
                        }
                        إذا (K_remainder > 0) {
                            ثابت int64_t وٿي = K_بلاڪون * بلاڪ_سائيز_3;
                            ثابت عائم * b_کانو_ptr =
                                (ثابت عائم *) ((ثابت حرف *) src1_معلومات + وٿي * حجم(عائم) + کانو_idx * nb11 +
                                                 batch_idx * nb12);
                            جوڙ += compute_بلاڪ_dot_حاصل_ضرب_f32_partial(&f32_صف[وٿي], b_کانو_ptr, K_remainder);
                        }
                        اكسر;
                    }
                افتراضي:
                    أرجع -1;
            }

            // Store result using atomic store to avoid cache coherency issues
            // when multiple threads write to the same cache line (64 bytes = 16 floats)
            متطاير عائم * dst_element =
                (متطاير عائم *) ((حرف *) dst_معلومات + m * dst->nb[0] + n_idx * nb1 + batch_idx * nb2);
            atomic_store_f32(dst_element, جوڙ);
        }
    }

    أرجع 0;
}
