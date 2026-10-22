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

#ଅନ୍ତର୍ଭାଵ "block_ops.h"
#ଅନ୍ତର୍ଭାଵ "ggml_tensor.h"
#ଅନ୍ତର୍ଭାଵ "math_fp.h"
#ଅନ୍ତର୍ଭାଵ "platform.h"
#ଅନ୍ତର୍ଭାଵ "quants.h"

#ଅନ୍ତର୍ଭାଵ <stdint.h>

// Main entry point for MUL_MAT_ID kernel (Mixture of Experts)
ପୂର୍ଣ୍ଣସଂଖ୍ୟା entry_point(ସଂରଚନା ggml_et_mul_mat_id_params * params, ଶୂନ୍ୟ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ଯଦି (!kernel_env) {
        ଫେରନ୍ତୁ -1;
    }

    // Get thread coordination info
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_threads = get_ଅଙ୍କ_threads(kernel_env->shire_mask);

    ଯଦି (thread_id < 0) {
        ଫେରନ୍ତୁ -1;
    }

    // Use even threads only to avoid resource contention
    // Each minion has 2 threads sharing instruction/data cache, NOC to RAM, and FPU
    // Odd threads return immediately to avoid fighting for shared resources
    ଯଦି (thread_id & 1) {
        ଫେରନ୍ତୁ 0;  // Odd thread - skip work
    }

    // Adjust thread count and ID for even-only threading
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା effective_thread_id   = thread_id / 2;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା effective_ଅଙ୍କ_threads = (ଅଙ୍କ_threads + 1) / 2;  // Ceiling division

    // Validate params
    ଯଦି (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ଫେରନ୍ତୁ -1;
    }

    // Extract tensor references
    ସଂରଚନା ggml_tensor * src0 = &params->src0;  // Expert weight matrices [K, M, n_expert]
    ସଂରଚନା ggml_tensor * src1 = &params->src1;  // Activations [K, n_expert_used, batch]
    ସଂରଚନା ggml_tensor * src2 = &params->src2;  // Expert indices [n_expert_used, batch] (I32)
    ସଂରଚନା ggml_tensor * dst  = &params->dst;   // Output [M, n_expert_used, batch, 1]

    // Validate tensor types
    ଯଦି (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32) {
        ଫେରନ୍ତୁ -1;
    }

    // Get data pointers
    ଅଚଳ ଶୂନ୍ୟ *    src0_data = src0->data;                    // Expert matrices (Q8_0/F16/F32)
    ଅଚଳ ଦଶମିକ *   src1_data = (ଅଚଳ ଦଶମିକ *) src1->data;    // Activations (F32)
    ଅଚଳ int32_t * src2_data = (ଅଚଳ int32_t *) src2->data;  // Expert IDs (I32)
    ଦଶମିକ *         dst_data  = (ଦଶମିକ *) dst->data;           // Output (F32)

    ଯଦି (!src0_data || !src1_data || !src2_data || !dst_data) {
        ଫେରନ୍ତୁ -1;
    }

    // Determine block size based on src0 type
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_size;
    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_Q8_0:
            block_size = QK8_0;
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q4_0:
            block_size = QK4_0;
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_F16:
            block_size = QK_F16;
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_F32:
            block_size = QK_F32;
            ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            ଫେରନ୍ତୁ -1;
    }

    // Get dimensions
    // src0: [K, M, n_expert] - expert weight matrices
    // src1: [K, n_expert_used, batch] - activations
    // src2: [n_expert_used, batch] - expert indices
    // dst:  [M, n_expert_used, batch, 1] - output
    ଅଚଳ int64_t K             = src0->ne[0];  // Hidden dimension
    ଅଚଳ int64_t M             = src0->ne[1];  // Output features
    ଅଚଳ int64_t n_expert      = src0->ne[2];  // Number of experts
    ଅଚଳ int64_t n_expert_used = src2->ne[0];  // Experts used per token
    ଅଚଳ int64_t batch         = src2->ne[1];  // Batch size

    // Strides (in bytes)
    ଅଚଳ size_t nb01 = src0->nb[1];  // src0 row stride
    ଅଚଳ size_t nb02 = src0->nb[2];  // src0 expert stride
    ଅଚଳ size_t nb11 = src1->nb[1];  // src1 column stride
    ଅଚଳ size_t nb12 = src1->nb[2];  // src1 batch stride
    ଅଚଳ size_t nb20 = src2->nb[0];  // src2 element stride
    ଅଚଳ size_t nb21 = src2->nb[1];  // src2 batch stride
    ଅଚଳ size_t nb1  = dst->nb[1];   // dst column stride
    ଅଚଳ size_t nb2  = dst->nb[2];   // dst batch stride

    // Verify K dimension alignment for quantization
    // Q8_0 requires strict alignment (quantized data must be block-aligned)
    // F32 and F16 can handle partial blocks with scalar remainders
    ଯଦି ((src0->type == GGML_TYPE_Q8_0 || src0->type == GGML_TYPE_Q4_0) && K % block_size != 0) {
        ଫେରନ୍ତୁ -1;  // Q8_0 requires K to be multiple of block_size
    }

    // Verify first dimension is contiguous
    size_t expected_element_size_src0;
    ଯଦି (src0->type == GGML_TYPE_Q8_0) {
        expected_element_size_src0 = ଆକାର(block_q8_0);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_Q4_0) {
        expected_element_size_src0 = ଆକାର(block_q4_0);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_F16) {
        expected_element_size_src0 = ଆକାର(uint16_t);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_F32) {
        expected_element_size_src0 = ଆକାର(ଦଶମିକ);
    } ଅନ୍ୟଥା {
        ଫେରନ୍ତୁ -1;
    }

    ଯଦି (src0->nb[0] != expected_element_size_src0 || src1->nb[0] != ଆକାର(ଦଶମିକ) || src2->nb[0] != ଆକାର(int32_t) ||
        dst->nb[0] != ଆକାର(ଦଶମିକ)) {
        ଫେରନ୍ତୁ -1;
    }

    ଅଚଳ int64_t K_blocks = K / block_size;

    // Threading: distribute output elements across threads
    // Total output elements = M * n_expert_used * batch
    ଅଚଳ uint64_t total_elements = M * n_expert_used * batch;

    ଅଚଳ uint64_t per_thread     = 16;
    ଅଚଳ uint64_t threads_stride = per_thread * effective_ଅଙ୍କ_threads;

    ଯଦି (effective_thread_id * per_thread >= total_elements) {
        ଫେରନ୍ତୁ 0;
    }

    // Process elements assigned to this thread
    ପାଇଁ (uint64_t base_idx = effective_thread_id * per_thread; base_idx < total_elements; base_idx += threads_stride) {
        ପାଇଁ (uint64_t j = 0; j < per_thread; j++) {
            ଅଚଳ uint64_t idx = base_idx + j;

            ଯଦି (idx >= total_elements) {
                ଭାଙ୍ଗ;
            }

            // Decode linear index to (m, n_idx, batch_idx)
            // Layout: m + M * (n_idx + n_expert_used * batch_idx)
            ଅଚଳ int64_t batch_idx = idx / (M * n_expert_used);
            ଅଚଳ int64_t rem       = idx % (M * n_expert_used);
            ଅଚଳ int64_t n_idx     = rem / M;
            ଅଚଳ int64_t m         = rem % M;

            // Get expert ID from src2[n_idx, batch_idx]
            ଅଚଳ int32_t expert_id = *(ଅଚଳ int32_t *) ((ଅଚଳ ଅକ୍ଷର *) src2_data + n_idx * nb20 + batch_idx * nb21);

            // Validate expert ID
            ଯଦି (expert_id < 0 || expert_id >= n_expert) {
                // Invalid expert ID - write zero and continue
                ଅସ୍ଥିର ଦଶମିକ * dst_element =
                    (ଅସ୍ଥିର ଦଶମିକ *) ((ଅକ୍ଷର *) dst_data + m * dst->nb[0] + n_idx * nb1 + batch_idx * nb2);
                atomic_store_f32(dst_element, 0.0f);
                ଜାରି;
            }

            // Compute dot product: expert_matrix[m, :] x activations[:, col_idx, batch_idx]
            // Use modulo to handle broadcasting: when src1 has fewer columns than expert slots,
            // multiple slots share the same activation column (col_idx = n_idx % src1->ne[1])
            ଅଚଳ int64_t col_idx = n_idx % src1->ne[1];
            ଦଶମିକ         sum     = 0.0f;

            // Type switch hoisted outside block loop: one branch per element, not per block
            ଅଚଳ ଅକ୍ଷର * expert_row_base = (ଅଚଳ ଅକ୍ଷର *) src0_data + m * nb01 + expert_id * nb02;

            ବାଛ (src0->type) {
                ସ୍ଥିତି GGML_TYPE_Q8_0:
                    {
                        ଅଚଳ block_q8_0 * q8_row = (ଅଚଳ block_q8_0 *) expert_row_base;
                        ଅଚଳ ଦଶମିକ *      b_col_base =
                            (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) src1_data + col_idx * nb11 + batch_idx * nb12);
                        sum += compute_row_dot_q8_0(q8_row, b_col_base, K_blocks);
                        ଭାଙ୍ଗ;
                    }
                ସ୍ଥିତି GGML_TYPE_Q4_0:
                    {
                        ଅଚଳ block_q4_0 * q4_row = (ଅଚଳ block_q4_0 *) expert_row_base;
                        ଅଚଳ ଦଶମିକ *      b_col_base =
                            (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) src1_data + col_idx * nb11 + batch_idx * nb12);
                        sum += compute_row_dot_q4_0(q4_row, b_col_base, K_blocks);
                        ଭାଙ୍ଗ;
                    }
                ସ୍ଥିତି GGML_TYPE_F16:
                    {
                        ଅଚଳ uint16_t * f16_row     = (ଅଚଳ uint16_t *) expert_row_base;
                        ଅଚଳ int64_t    K_remainder = K % block_size;
                        ପାଇଁ (int64_t kb = 0; kb < K_blocks; kb++) {
                            ଅଚଳ ଦଶମିକ * b_col_ptr =
                                (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) src1_data + (kb * block_size) * ଆକାର(ଦଶମିକ) +
                                                 col_idx * nb11 + batch_idx * nb12);
                            sum += compute_block_dot_product_f16_naive(&f16_row[kb * block_size], b_col_ptr);
                        }
                        ଯଦି (K_remainder > 0) {
                            ଅଚଳ int64_t offset = K_blocks * block_size;
                            ଅଚଳ ଦଶମିକ * b_col_ptr =
                                (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) src1_data + offset * ଆକାର(ଦଶମିକ) + col_idx * nb11 +
                                                 batch_idx * nb12);
                            sum += compute_block_dot_product_f16_partial(&f16_row[offset], b_col_ptr, K_remainder);
                        }
                        ଭାଙ୍ଗ;
                    }
                ସ୍ଥିତି GGML_TYPE_F32:
                    {
                        ଅଚଳ ଦଶମିକ * f32_row     = (ଅଚଳ ଦଶମିକ *) expert_row_base;
                        ଅଚଳ int64_t K_remainder = K % block_size;
                        ପାଇଁ (int64_t kb = 0; kb < K_blocks; kb++) {
                            ଅଚଳ ଦଶମିକ * b_col_ptr =
                                (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) src1_data + (kb * block_size) * ଆକାର(ଦଶମିକ) +
                                                 col_idx * nb11 + batch_idx * nb12);
                            sum += compute_block_dot_product_f32(&f32_row[kb * block_size], b_col_ptr);
                        }
                        ଯଦି (K_remainder > 0) {
                            ଅଚଳ int64_t offset = K_blocks * block_size;
                            ଅଚଳ ଦଶମିକ * b_col_ptr =
                                (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) src1_data + offset * ଆକାର(ଦଶମିକ) + col_idx * nb11 +
                                                 batch_idx * nb12);
                            sum += compute_block_dot_product_f32_partial(&f32_row[offset], b_col_ptr, K_remainder);
                        }
                        ଭାଙ୍ଗ;
                    }
                ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                    ଫେରନ୍ତୁ -1;
            }

            // Store result using atomic store to avoid cache coherency issues
            // when multiple threads write to the same cache line (64 bytes = 16 floats)
            ଅସ୍ଥିର ଦଶମିକ * dst_element =
                (ଅସ୍ଥିର ଦଶମିକ *) ((ଅକ୍ଷର *) dst_data + m * dst->nb[0] + n_idx * nb1 + batch_idx * nb2);
            atomic_store_f32(dst_element, sum);
        }
    }

    ଫେରନ୍ତୁ 0;
}
