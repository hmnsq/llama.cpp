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

#شامل "block_ops.h"
#شامل "ggml_tensor.h"
#شامل "math_fp.h"
#شامل "platform.h"
#شامل "quants.h"

#شامل <stdint.h>

// Main entry point for MUL_MAT_ID kernel (Mixture of Experts)
صحیح اندراج_نقطہ(ساخت ggml_et_mul_mat_id_params * params, خالی * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    اگر (!kernel_env) {
        واپس -1;
    }

    // Get thread coordination info
    صحیح thread_id   = get_relative_thread_id(kernel_env->shire_نقاب);
    صحیح عدد_دھاگے = get_عدد_دھاگے(kernel_env->shire_نقاب);

    اگر (thread_id < 0) {
        واپس -1;
    }

    // Use even threads only to avoid resource contention
    // Each minion has 2 threads sharing instruction/data cache, NOC to RAM, and FPU
    // Odd threads return immediately to avoid fighting for shared resources
    اگر (thread_id & 1) {
        واپس 0;  // Odd thread - skip work
    }

    // Adjust thread count and ID for even-only threading
    صحیح effective_thread_id   = thread_id / 2;
    صحیح effective_عدد_دھاگے = (عدد_دھاگے + 1) / 2;  // Ceiling division

    // Validate params
    اگر (params == 0 || ((uint64_t) params & 0x7) != 0) {
        واپس -1;
    }

    // Extract tensor references
    ساخت ggml_tensor * src0 = &params->src0;  // Expert weight matrices [K, M, n_expert]
    ساخت ggml_tensor * src1 = &params->src1;  // Activations [K, n_expert_used, batch]
    ساخت ggml_tensor * src2 = &params->src2;  // Expert indices [n_expert_used, batch] (I32)
    ساخت ggml_tensor * dst  = &params->dst;   // Output [M, n_expert_used, batch, 1]

    // Validate tensor types
    اگر (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32) {
        واپس -1;
    }

    // Get data pointers
    ثابت خالی *    src0_ڈیٹا = src0->ڈیٹا;                    // Expert matrices (Q8_0/F16/F32)
    ثابت اعشاری *   src1_ڈیٹا = (ثابت اعشاری *) src1->ڈیٹا;    // Activations (F32)
    ثابت int32_t * src2_ڈیٹا = (ثابت int32_t *) src2->ڈیٹا;  // Expert IDs (I32)
    اعشاری *         dst_ڈیٹا  = (اعشاری *) dst->ڈیٹا;           // Output (F32)

    اگر (!src0_ڈیٹا || !src1_ڈیٹا || !src2_ڈیٹا || !dst_ڈیٹا) {
        واپس -1;
    }

    // Determine block size based on src0 type
    صحیح بلاک_سائز_3;
    انتخاب (src0->type) {
        حالت GGML_TYPE_Q8_0:
            بلاک_سائز_3 = QK8_0;
            توڑو;
        حالت GGML_TYPE_Q4_0:
            بلاک_سائز_3 = QK4_0;
            توڑو;
        حالت GGML_TYPE_F16:
            بلاک_سائز_3 = QK_F16;
            توڑو;
        حالت GGML_TYPE_F32:
            بلاک_سائز_3 = QK_F32;
            توڑو;
        طےشدہ:
            واپس -1;
    }

    // Get dimensions
    // src0: [K, M, n_expert] - expert weight matrices
    // src1: [K, n_expert_used, batch] - activations
    // src2: [n_expert_used, batch] - expert indices
    // dst:  [M, n_expert_used, batch, 1] - output
    ثابت int64_t K             = src0->ne[0];  // Hidden dimension
    ثابت int64_t M             = src0->ne[1];  // Output features
    ثابت int64_t n_expert      = src0->ne[2];  // Number of experts
    ثابت int64_t n_expert_مستعمل = src2->ne[0];  // Experts used per token
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
    اگر ((src0->type == GGML_TYPE_Q8_0 || src0->type == GGML_TYPE_Q4_0) && K % بلاک_سائز_3 != 0) {
        واپس -1;  // Q8_0 requires K to be multiple of block_size
    }

    // Verify first dimension is contiguous
    size_t expected_element_سائز_src0;
    اگر (src0->type == GGML_TYPE_Q8_0) {
        expected_element_سائز_src0 = حجم(بلاک_q8_0);
    } ورنہ اگر (src0->type == GGML_TYPE_Q4_0) {
        expected_element_سائز_src0 = حجم(بلاک_q4_0);
    } ورنہ اگر (src0->type == GGML_TYPE_F16) {
        expected_element_سائز_src0 = حجم(uint16_t);
    } ورنہ اگر (src0->type == GGML_TYPE_F32) {
        expected_element_سائز_src0 = حجم(اعشاری);
    } ورنہ {
        واپس -1;
    }

    اگر (src0->nb[0] != expected_element_سائز_src0 || src1->nb[0] != حجم(اعشاری) || src2->nb[0] != حجم(int32_t) ||
        dst->nb[0] != حجم(اعشاری)) {
        واپس -1;
    }

    ثابت int64_t K_بلاکیں = K / بلاک_سائز_3;

    // Threading: distribute output elements across threads
    // Total output elements = M * n_expert_used * batch
    ثابت uint64_t کل_elements = M * n_expert_مستعمل * batch;

    ثابت uint64_t فی_thread     = 16;
    ثابت uint64_t دھاگے_stride = فی_thread * effective_عدد_دھاگے;

    اگر (effective_thread_id * فی_thread >= کل_elements) {
        واپس 0;
    }

    // Process elements assigned to this thread
    برائے (uint64_t بنیاد_idx = effective_thread_id * فی_thread; بنیاد_idx < کل_elements; بنیاد_idx += دھاگے_stride) {
        برائے (uint64_t j = 0; j < فی_thread; j++) {
            ثابت uint64_t idx = بنیاد_idx + j;

            اگر (idx >= کل_elements) {
                توڑو;
            }

            // Decode linear index to (m, n_idx, batch_idx)
            // Layout: m + M * (n_idx + n_expert_used * batch_idx)
            ثابت int64_t batch_idx = idx / (M * n_expert_مستعمل);
            ثابت int64_t rem       = idx % (M * n_expert_مستعمل);
            ثابت int64_t n_idx     = rem / M;
            ثابت int64_t m         = rem % M;

            // Get expert ID from src2[n_idx, batch_idx]
            ثابت int32_t expert_id = *(ثابت int32_t *) ((ثابت حرف *) src2_ڈیٹا + n_idx * nb20 + batch_idx * nb21);

            // Validate expert ID
            اگر (expert_id < 0 || expert_id >= n_expert) {
                // Invalid expert ID - write zero and continue
                متلون اعشاری * dst_element =
                    (متلون اعشاری *) ((حرف *) dst_ڈیٹا + m * dst->nb[0] + n_idx * nb1 + batch_idx * nb2);
                atomic_store_f32(dst_element, 0.0f);
                جاری;
            }

            // Compute dot product: expert_matrix[m, :] x activations[:, col_idx, batch_idx]
            // Use modulo to handle broadcasting: when src1 has fewer columns than expert slots,
            // multiple slots share the same activation column (col_idx = n_idx % src1->ne[1])
            ثابت int64_t ستون_idx = n_idx % src1->ne[1];
            اعشاری         مجموع     = 0.0f;

            // Type switch hoisted outside block loop: one branch per element, not per block
            ثابت حرف * expert_صف_بنیاد = (ثابت حرف *) src0_ڈیٹا + m * nb01 + expert_id * nb02;

            انتخاب (src0->type) {
                حالت GGML_TYPE_Q8_0:
                    {
                        ثابت بلاک_q8_0 * q8_صف = (ثابت بلاک_q8_0 *) expert_صف_بنیاد;
                        ثابت اعشاری *      b_ستون_بنیاد =
                            (ثابت اعشاری *) ((ثابت حرف *) src1_ڈیٹا + ستون_idx * nb11 + batch_idx * nb12);
                        مجموع += compute_صف_dot_q8_0(q8_صف, b_ستون_بنیاد, K_بلاکیں);
                        توڑو;
                    }
                حالت GGML_TYPE_Q4_0:
                    {
                        ثابت بلاک_q4_0 * q4_صف = (ثابت بلاک_q4_0 *) expert_صف_بنیاد;
                        ثابت اعشاری *      b_ستون_بنیاد =
                            (ثابت اعشاری *) ((ثابت حرف *) src1_ڈیٹا + ستون_idx * nb11 + batch_idx * nb12);
                        مجموع += compute_صف_dot_q4_0(q4_صف, b_ستون_بنیاد, K_بلاکیں);
                        توڑو;
                    }
                حالت GGML_TYPE_F16:
                    {
                        ثابت uint16_t * f16_صف     = (ثابت uint16_t *) expert_صف_بنیاد;
                        ثابت int64_t    K_remainder = K % بلاک_سائز_3;
                        برائے (int64_t kb = 0; kb < K_بلاکیں; kb++) {
                            ثابت اعشاری * b_ستون_ptr =
                                (ثابت اعشاری *) ((ثابت حرف *) src1_ڈیٹا + (kb * بلاک_سائز_3) * حجم(اعشاری) +
                                                 ستون_idx * nb11 + batch_idx * nb12);
                            مجموع += compute_بلاک_dot_حاصل_ضرب_f16_naive(&f16_صف[kb * بلاک_سائز_3], b_ستون_ptr);
                        }
                        اگر (K_remainder > 0) {
                            ثابت int64_t انحراف = K_بلاکیں * بلاک_سائز_3;
                            ثابت اعشاری * b_ستون_ptr =
                                (ثابت اعشاری *) ((ثابت حرف *) src1_ڈیٹا + انحراف * حجم(اعشاری) + ستون_idx * nb11 +
                                                 batch_idx * nb12);
                            مجموع += compute_بلاک_dot_حاصل_ضرب_f16_partial(&f16_صف[انحراف], b_ستون_ptr, K_remainder);
                        }
                        توڑو;
                    }
                حالت GGML_TYPE_F32:
                    {
                        ثابت اعشاری * f32_صف     = (ثابت اعشاری *) expert_صف_بنیاد;
                        ثابت int64_t K_remainder = K % بلاک_سائز_3;
                        برائے (int64_t kb = 0; kb < K_بلاکیں; kb++) {
                            ثابت اعشاری * b_ستون_ptr =
                                (ثابت اعشاری *) ((ثابت حرف *) src1_ڈیٹا + (kb * بلاک_سائز_3) * حجم(اعشاری) +
                                                 ستون_idx * nb11 + batch_idx * nb12);
                            مجموع += compute_بلاک_dot_حاصل_ضرب_f32(&f32_صف[kb * بلاک_سائز_3], b_ستون_ptr);
                        }
                        اگر (K_remainder > 0) {
                            ثابت int64_t انحراف = K_بلاکیں * بلاک_سائز_3;
                            ثابت اعشاری * b_ستون_ptr =
                                (ثابت اعشاری *) ((ثابت حرف *) src1_ڈیٹا + انحراف * حجم(اعشاری) + ستون_idx * nb11 +
                                                 batch_idx * nb12);
                            مجموع += compute_بلاک_dot_حاصل_ضرب_f32_partial(&f32_صف[انحراف], b_ستون_ptr, K_remainder);
                        }
                        توڑو;
                    }
                طےشدہ:
                    واپس -1;
            }

            // Store result using atomic store to avoid cache coherency issues
            // when multiple threads write to the same cache line (64 bytes = 16 floats)
            متلون اعشاری * dst_element =
                (متلون اعشاری *) ((حرف *) dst_ڈیٹا + m * dst->nb[0] + n_idx * nb1 + batch_idx * nb2);
            atomic_store_f32(dst_element, مجموع);
        }
    }

    واپس 0;
}
