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
صحيح entry_point(بنية ggml_et_mul_mat_id_params * params, فراغ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    אם (!kernel_env) {
        החזר -1;
    }

    // Get thread coordination info
    صحيح thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    صحيح num_threads = get_num_threads(kernel_env->shire_mask);

    אם (thread_id < 0) {
        החזר -1;
    }

    // Use even threads only to avoid resource contention
    // Each minion has 2 threads sharing instruction/data cache, NOC to RAM, and FPU
    // Odd threads return immediately to avoid fighting for shared resources
    אם (thread_id & 1) {
        החזר 0;  // Odd thread - skip work
    }

    // Adjust thread count and ID for even-only threading
    صحيح effective_thread_id   = thread_id / 2;
    صحيح effective_num_threads = (num_threads + 1) / 2;  // Ceiling division

    // Validate params
    אם (params == 0 || ((uint64_t) params & 0x7) != 0) {
        החזר -1;
    }

    // Extract tensor references
    بنية ggml_tensor * src0 = &params->src0;  // Expert weight matrices [K, M, n_expert]
    بنية ggml_tensor * src1 = &params->src1;  // Activations [K, n_expert_used, batch]
    بنية ggml_tensor * src2 = &params->src2;  // Expert indices [n_expert_used, batch] (I32)
    بنية ggml_tensor * dst  = &params->dst;   // Output [M, n_expert_used, batch, 1]

    // Validate tensor types
    אם (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32) {
        החזר -1;
    }

    // Get data pointers
    ثابت فراغ *    src0_data = src0->data;                    // Expert matrices (Q8_0/F16/F32)
    ثابت عائم *   src1_data = (ثابت عائم *) src1->data;    // Activations (F32)
    ثابت int32_t * src2_data = (ثابت int32_t *) src2->data;  // Expert IDs (I32)
    عائم *         dst_data  = (عائم *) dst->data;           // Output (F32)

    אם (!src0_data || !src1_data || !src2_data || !dst_data) {
        החזר -1;
    }

    // Determine block size based on src0 type
    صحيح block_size;
    بدل (src0->type) {
        حالة GGML_TYPE_Q8_0:
            block_size = QK8_0;
            اكسر;
        حالة GGML_TYPE_Q4_0:
            block_size = QK4_0;
            اكسر;
        حالة GGML_TYPE_F16:
            block_size = QK_F16;
            اكسر;
        حالة GGML_TYPE_F32:
            block_size = QK_F32;
            اكسر;
        افتراضي:
            החזר -1;
    }

    // Get dimensions
    // src0: [K, M, n_expert] - expert weight matrices
    // src1: [K, n_expert_used, batch] - activations
    // src2: [n_expert_used, batch] - expert indices
    // dst:  [M, n_expert_used, batch, 1] - output
    ثابت int64_t K             = src0->ne[0];  // Hidden dimension
    ثابت int64_t M             = src0->ne[1];  // Output features
    ثابت int64_t n_expert      = src0->ne[2];  // Number of experts
    ثابت int64_t n_expert_used = src2->ne[0];  // Experts used per token
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
    אם ((src0->type == GGML_TYPE_Q8_0 || src0->type == GGML_TYPE_Q4_0) && K % block_size != 0) {
        החזר -1;  // Q8_0 requires K to be multiple of block_size
    }

    // Verify first dimension is contiguous
    size_t expected_element_size_src0;
    אם (src0->type == GGML_TYPE_Q8_0) {
        expected_element_size_src0 = حجم(block_q8_0);
    } אחרת אם (src0->type == GGML_TYPE_Q4_0) {
        expected_element_size_src0 = حجم(block_q4_0);
    } אחרת אם (src0->type == GGML_TYPE_F16) {
        expected_element_size_src0 = حجم(uint16_t);
    } אחרת אם (src0->type == GGML_TYPE_F32) {
        expected_element_size_src0 = حجم(عائم);
    } אחרת {
        החזר -1;
    }

    אם (src0->nb[0] != expected_element_size_src0 || src1->nb[0] != حجم(عائم) || src2->nb[0] != حجم(int32_t) ||
        dst->nb[0] != حجم(عائم)) {
        החזר -1;
    }

    ثابت int64_t K_blocks = K / block_size;

    // Threading: distribute output elements across threads
    // Total output elements = M * n_expert_used * batch
    ثابت uint64_t total_elements = M * n_expert_used * batch;

    ثابت uint64_t per_thread     = 16;
    ثابت uint64_t threads_stride = per_thread * effective_num_threads;

    אם (effective_thread_id * per_thread >= total_elements) {
        החזר 0;
    }

    // Process elements assigned to this thread
    עבור (uint64_t base_idx = effective_thread_id * per_thread; base_idx < total_elements; base_idx += threads_stride) {
        עבור (uint64_t j = 0; j < per_thread; j++) {
            ثابت uint64_t idx = base_idx + j;

            אם (idx >= total_elements) {
                اكسر;
            }

            // Decode linear index to (m, n_idx, batch_idx)
            // Layout: m + M * (n_idx + n_expert_used * batch_idx)
            ثابت int64_t batch_idx = idx / (M * n_expert_used);
            ثابت int64_t rem       = idx % (M * n_expert_used);
            ثابت int64_t n_idx     = rem / M;
            ثابت int64_t m         = rem % M;

            // Get expert ID from src2[n_idx, batch_idx]
            ثابت int32_t expert_id = *(ثابت int32_t *) ((ثابت حرف *) src2_data + n_idx * nb20 + batch_idx * nb21);

            // Validate expert ID
            אם (expert_id < 0 || expert_id >= n_expert) {
                // Invalid expert ID - write zero and continue
                متطاير عائم * dst_element =
                    (متطاير عائم *) ((حرف *) dst_data + m * dst->nb[0] + n_idx * nb1 + batch_idx * nb2);
                atomic_store_f32(dst_element, 0.0f);
                استمر;
            }

            // Compute dot product: expert_matrix[m, :] x activations[:, col_idx, batch_idx]
            // Use modulo to handle broadcasting: when src1 has fewer columns than expert slots,
            // multiple slots share the same activation column (col_idx = n_idx % src1->ne[1])
            ثابت int64_t col_idx = n_idx % src1->ne[1];
            عائم         sum     = 0.0f;

            // Type switch hoisted outside block loop: one branch per element, not per block
            ثابت حرف * expert_row_base = (ثابت حرف *) src0_data + m * nb01 + expert_id * nb02;

            بدل (src0->type) {
                حالة GGML_TYPE_Q8_0:
                    {
                        ثابت block_q8_0 * q8_row = (ثابت block_q8_0 *) expert_row_base;
                        ثابت عائم *      b_col_base =
                            (ثابت عائم *) ((ثابت حرف *) src1_data + col_idx * nb11 + batch_idx * nb12);
                        sum += compute_row_dot_q8_0(q8_row, b_col_base, K_blocks);
                        اكسر;
                    }
                حالة GGML_TYPE_Q4_0:
                    {
                        ثابت block_q4_0 * q4_row = (ثابت block_q4_0 *) expert_row_base;
                        ثابت عائم *      b_col_base =
                            (ثابت عائم *) ((ثابت حرف *) src1_data + col_idx * nb11 + batch_idx * nb12);
                        sum += compute_row_dot_q4_0(q4_row, b_col_base, K_blocks);
                        اكسر;
                    }
                حالة GGML_TYPE_F16:
                    {
                        ثابت uint16_t * f16_row     = (ثابت uint16_t *) expert_row_base;
                        ثابت int64_t    K_remainder = K % block_size;
                        עבור (int64_t kb = 0; kb < K_blocks; kb++) {
                            ثابت عائم * b_col_ptr =
                                (ثابت عائم *) ((ثابت حرف *) src1_data + (kb * block_size) * حجم(عائم) +
                                                 col_idx * nb11 + batch_idx * nb12);
                            sum += compute_block_dot_product_f16_naive(&f16_row[kb * block_size], b_col_ptr);
                        }
                        אם (K_remainder > 0) {
                            ثابت int64_t offset = K_blocks * block_size;
                            ثابت عائم * b_col_ptr =
                                (ثابت عائم *) ((ثابت حرف *) src1_data + offset * حجم(عائم) + col_idx * nb11 +
                                                 batch_idx * nb12);
                            sum += compute_block_dot_product_f16_partial(&f16_row[offset], b_col_ptr, K_remainder);
                        }
                        اكسر;
                    }
                حالة GGML_TYPE_F32:
                    {
                        ثابت عائم * f32_row     = (ثابت عائم *) expert_row_base;
                        ثابت int64_t K_remainder = K % block_size;
                        עבור (int64_t kb = 0; kb < K_blocks; kb++) {
                            ثابت عائم * b_col_ptr =
                                (ثابت عائم *) ((ثابت حرف *) src1_data + (kb * block_size) * حجم(عائم) +
                                                 col_idx * nb11 + batch_idx * nb12);
                            sum += compute_block_dot_product_f32(&f32_row[kb * block_size], b_col_ptr);
                        }
                        אם (K_remainder > 0) {
                            ثابت int64_t offset = K_blocks * block_size;
                            ثابت عائم * b_col_ptr =
                                (ثابت عائم *) ((ثابت حرف *) src1_data + offset * حجم(عائم) + col_idx * nb11 +
                                                 batch_idx * nb12);
                            sum += compute_block_dot_product_f32_partial(&f32_row[offset], b_col_ptr, K_remainder);
                        }
                        اكسر;
                    }
                افتراضي:
                    החזר -1;
            }

            // Store result using atomic store to avoid cache coherency issues
            // when multiple threads write to the same cache line (64 bytes = 16 floats)
            متطاير عائم * dst_element =
                (متطاير عائم *) ((حرف *) dst_data + m * dst->nb[0] + n_idx * nb1 + batch_idx * nb2);
            atomic_store_f32(dst_element, sum);
        }
    }

    החזר 0;
}
