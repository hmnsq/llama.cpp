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

#অন্তর্ভাব "block_ops.h"
#অন্তর্ভাব "ggml_tensor.h"
#অন্তর্ভাব "math_fp.h"
#অন্তর্ভাব "platform.h"
#অন্তর্ভাব "quants.h"

#অন্তর্ভাব <stdint.h>

// Main entry point for MUL_MAT_ID kernel (Mixture of Experts)
পূর্ণসংখ্যা entry_point(কাঠামো ggml_et_mul_mat_id_params * params, শূন্য * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    যদি (!kernel_env) {
        ফেরত -1;
    }

    // Get thread coordination info
    পূর্ণসংখ্যা thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    পূর্ণসংখ্যা অঙ্ক_threads = get_অঙ্ক_threads(kernel_env->shire_mask);

    যদি (thread_id < 0) {
        ফেরত -1;
    }

    // Use even threads only to avoid resource contention
    // Each minion has 2 threads sharing instruction/data cache, NOC to RAM, and FPU
    // Odd threads return immediately to avoid fighting for shared resources
    যদি (thread_id & 1) {
        ফেরত 0;  // Odd thread - skip work
    }

    // Adjust thread count and ID for even-only threading
    পূর্ণসংখ্যা effective_thread_id   = thread_id / 2;
    পূর্ণসংখ্যা effective_অঙ্ক_threads = (অঙ্ক_threads + 1) / 2;  // Ceiling division

    // Validate params
    যদি (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ফেরত -1;
    }

    // Extract tensor references
    কাঠামো ggml_tensor * src0 = &params->src0;  // Expert weight matrices [K, M, n_expert]
    কাঠামো ggml_tensor * src1 = &params->src1;  // Activations [K, n_expert_used, batch]
    কাঠামো ggml_tensor * src2 = &params->src2;  // Expert indices [n_expert_used, batch] (I32)
    কাঠামো ggml_tensor * dst  = &params->dst;   // Output [M, n_expert_used, batch, 1]

    // Validate tensor types
    যদি (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32) {
        ফেরত -1;
    }

    // Get data pointers
    ধ্রুবক শূন্য *    src0_তথ্য = src0->তথ্য;                    // Expert matrices (Q8_0/F16/F32)
    ধ্রুবক দশমিক *   src1_তথ্য = (ধ্রুবক দশমিক *) src1->তথ্য;    // Activations (F32)
    ধ্রুবক int32_t * src2_তথ্য = (ধ্রুবক int32_t *) src2->তথ্য;  // Expert IDs (I32)
    দশমিক *         dst_তথ্য  = (দশমিক *) dst->তথ্য;           // Output (F32)

    যদি (!src0_তথ্য || !src1_তথ্য || !src2_তথ্য || !dst_তথ্য) {
        ফেরত -1;
    }

    // Determine block size based on src0 type
    পূর্ণসংখ্যা block_আকার_2;
    নির্বাচন (src0->type) {
        ক্ষেত্র GGML_TYPE_Q8_0:
            block_আকার_2 = QK8_0;
            বিরতি;
        ক্ষেত্র GGML_TYPE_Q4_0:
            block_আকার_2 = QK4_0;
            বিরতি;
        ক্ষেত্র GGML_TYPE_F16:
            block_আকার_2 = QK_F16;
            বিরতি;
        ক্ষেত্র GGML_TYPE_F32:
            block_আকার_2 = QK_F32;
            বিরতি;
        পূর্বনির্ধারিত:
            ফেরত -1;
    }

    // Get dimensions
    // src0: [K, M, n_expert] - expert weight matrices
    // src1: [K, n_expert_used, batch] - activations
    // src2: [n_expert_used, batch] - expert indices
    // dst:  [M, n_expert_used, batch, 1] - output
    ধ্রুবক int64_t K             = src0->ne[0];  // Hidden dimension
    ধ্রুবক int64_t M             = src0->ne[1];  // Output features
    ধ্রুবক int64_t n_expert      = src0->ne[2];  // Number of experts
    ধ্রুবক int64_t n_expert_used = src2->ne[0];  // Experts used per token
    ধ্রুবক int64_t batch         = src2->ne[1];  // Batch size

    // Strides (in bytes)
    ধ্রুবক size_t nb01 = src0->nb[1];  // src0 row stride
    ধ্রুবক size_t nb02 = src0->nb[2];  // src0 expert stride
    ধ্রুবক size_t nb11 = src1->nb[1];  // src1 column stride
    ধ্রুবক size_t nb12 = src1->nb[2];  // src1 batch stride
    ধ্রুবক size_t nb20 = src2->nb[0];  // src2 element stride
    ধ্রুবক size_t nb21 = src2->nb[1];  // src2 batch stride
    ধ্রুবক size_t nb1  = dst->nb[1];   // dst column stride
    ধ্রুবক size_t nb2  = dst->nb[2];   // dst batch stride

    // Verify K dimension alignment for quantization
    // Q8_0 requires strict alignment (quantized data must be block-aligned)
    // F32 and F16 can handle partial blocks with scalar remainders
    যদি ((src0->type == GGML_TYPE_Q8_0 || src0->type == GGML_TYPE_Q4_0) && K % block_আকার_2 != 0) {
        ফেরত -1;  // Q8_0 requires K to be multiple of block_size
    }

    // Verify first dimension is contiguous
    size_t expected_element_আকার_src0;
    যদি (src0->type == GGML_TYPE_Q8_0) {
        expected_element_আকার_src0 = আকার(block_q8_0);
    } অন্যথা যদি (src0->type == GGML_TYPE_Q4_0) {
        expected_element_আকার_src0 = আকার(block_q4_0);
    } অন্যথা যদি (src0->type == GGML_TYPE_F16) {
        expected_element_আকার_src0 = আকার(uint16_t);
    } অন্যথা যদি (src0->type == GGML_TYPE_F32) {
        expected_element_আকার_src0 = আকার(দশমিক);
    } অন্যথা {
        ফেরত -1;
    }

    যদি (src0->nb[0] != expected_element_আকার_src0 || src1->nb[0] != আকার(দশমিক) || src2->nb[0] != আকার(int32_t) ||
        dst->nb[0] != আকার(দশমিক)) {
        ফেরত -1;
    }

    ধ্রুবক int64_t K_blocks = K / block_আকার_2;

    // Threading: distribute output elements across threads
    // Total output elements = M * n_expert_used * batch
    ধ্রুবক uint64_t যোগ_elements = M * n_expert_used * batch;

    ধ্রুবক uint64_t per_thread     = 16;
    ধ্রুবক uint64_t threads_stride = per_thread * effective_অঙ্ক_threads;

    যদি (effective_thread_id * per_thread >= যোগ_elements) {
        ফেরত 0;
    }

    // Process elements assigned to this thread
    জন্য (uint64_t base_idx = effective_thread_id * per_thread; base_idx < যোগ_elements; base_idx += threads_stride) {
        জন্য (uint64_t j = 0; j < per_thread; j++) {
            ধ্রুবক uint64_t idx = base_idx + j;

            যদি (idx >= যোগ_elements) {
                বিরতি;
            }

            // Decode linear index to (m, n_idx, batch_idx)
            // Layout: m + M * (n_idx + n_expert_used * batch_idx)
            ধ্রুবক int64_t batch_idx = idx / (M * n_expert_used);
            ধ্রুবক int64_t rem       = idx % (M * n_expert_used);
            ধ্রুবক int64_t n_idx     = rem / M;
            ধ্রুবক int64_t m         = rem % M;

            // Get expert ID from src2[n_idx, batch_idx]
            ধ্রুবক int32_t expert_id = *(ধ্রুবক int32_t *) ((ধ্রুবক অক্ষর *) src2_তথ্য + n_idx * nb20 + batch_idx * nb21);

            // Validate expert ID
            যদি (expert_id < 0 || expert_id >= n_expert) {
                // Invalid expert ID - write zero and continue
                অস্থির দশমিক * dst_element =
                    (অস্থির দশমিক *) ((অক্ষর *) dst_তথ্য + m * dst->nb[0] + n_idx * nb1 + batch_idx * nb2);
                atomic_store_f32(dst_element, 0.0f);
                চালু;
            }

            // Compute dot product: expert_matrix[m, :] x activations[:, col_idx, batch_idx]
            // Use modulo to handle broadcasting: when src1 has fewer columns than expert slots,
            // multiple slots share the same activation column (col_idx = n_idx % src1->ne[1])
            ধ্রুবক int64_t col_idx = n_idx % src1->ne[1];
            দশমিক         যোগ_2     = 0.0f;

            // Type switch hoisted outside block loop: one branch per element, not per block
            ধ্রুবক অক্ষর * expert_row_base = (ধ্রুবক অক্ষর *) src0_তথ্য + m * nb01 + expert_id * nb02;

            নির্বাচন (src0->type) {
                ক্ষেত্র GGML_TYPE_Q8_0:
                    {
                        ধ্রুবক block_q8_0 * q8_row = (ধ্রুবক block_q8_0 *) expert_row_base;
                        ধ্রুবক দশমিক *      b_col_base =
                            (ধ্রুবক দশমিক *) ((ধ্রুবক অক্ষর *) src1_তথ্য + col_idx * nb11 + batch_idx * nb12);
                        যোগ_2 += compute_row_dot_q8_0(q8_row, b_col_base, K_blocks);
                        বিরতি;
                    }
                ক্ষেত্র GGML_TYPE_Q4_0:
                    {
                        ধ্রুবক block_q4_0 * q4_row = (ধ্রুবক block_q4_0 *) expert_row_base;
                        ধ্রুবক দশমিক *      b_col_base =
                            (ধ্রুবক দশমিক *) ((ধ্রুবক অক্ষর *) src1_তথ্য + col_idx * nb11 + batch_idx * nb12);
                        যোগ_2 += compute_row_dot_q4_0(q4_row, b_col_base, K_blocks);
                        বিরতি;
                    }
                ক্ষেত্র GGML_TYPE_F16:
                    {
                        ধ্রুবক uint16_t * f16_row     = (ধ্রুবক uint16_t *) expert_row_base;
                        ধ্রুবক int64_t    K_remainder = K % block_আকার_2;
                        জন্য (int64_t kb = 0; kb < K_blocks; kb++) {
                            ধ্রুবক দশমিক * b_col_ptr =
                                (ধ্রুবক দশমিক *) ((ধ্রুবক অক্ষর *) src1_তথ্য + (kb * block_আকার_2) * আকার(দশমিক) +
                                                 col_idx * nb11 + batch_idx * nb12);
                            যোগ_2 += compute_block_dot_product_f16_naive(&f16_row[kb * block_আকার_2], b_col_ptr);
                        }
                        যদি (K_remainder > 0) {
                            ধ্রুবক int64_t offset = K_blocks * block_আকার_2;
                            ধ্রুবক দশমিক * b_col_ptr =
                                (ধ্রুবক দশমিক *) ((ধ্রুবক অক্ষর *) src1_তথ্য + offset * আকার(দশমিক) + col_idx * nb11 +
                                                 batch_idx * nb12);
                            যোগ_2 += compute_block_dot_product_f16_partial(&f16_row[offset], b_col_ptr, K_remainder);
                        }
                        বিরতি;
                    }
                ক্ষেত্র GGML_TYPE_F32:
                    {
                        ধ্রুবক দশমিক * f32_row     = (ধ্রুবক দশমিক *) expert_row_base;
                        ধ্রুবক int64_t K_remainder = K % block_আকার_2;
                        জন্য (int64_t kb = 0; kb < K_blocks; kb++) {
                            ধ্রুবক দশমিক * b_col_ptr =
                                (ধ্রুবক দশমিক *) ((ধ্রুবক অক্ষর *) src1_তথ্য + (kb * block_আকার_2) * আকার(দশমিক) +
                                                 col_idx * nb11 + batch_idx * nb12);
                            যোগ_2 += compute_block_dot_product_f32(&f32_row[kb * block_আকার_2], b_col_ptr);
                        }
                        যদি (K_remainder > 0) {
                            ধ্রুবক int64_t offset = K_blocks * block_আকার_2;
                            ধ্রুবক দশমিক * b_col_ptr =
                                (ধ্রুবক দশমিক *) ((ধ্রুবক অক্ষর *) src1_তথ্য + offset * আকার(দশমিক) + col_idx * nb11 +
                                                 batch_idx * nb12);
                            যোগ_2 += compute_block_dot_product_f32_partial(&f32_row[offset], b_col_ptr, K_remainder);
                        }
                        বিরতি;
                    }
                পূর্বনির্ধারিত:
                    ফেরত -1;
            }

            // Store result using atomic store to avoid cache coherency issues
            // when multiple threads write to the same cache line (64 bytes = 16 floats)
            অস্থির দশমিক * dst_element =
                (অস্থির দশমিক *) ((অক্ষর *) dst_তথ্য + m * dst->nb[0] + n_idx * nb1 + batch_idx * nb2);
            atomic_store_f32(dst_element, যোগ_2);
        }
    }

    ফেরত 0;
}
