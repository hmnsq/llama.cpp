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

#समावेश "block_ops.h"
#समावेश "ggml_tensor.h"
#समावेश "math_fp.h"
#समावेश "platform.h"
#समावेश "quants.h"

#समावेश <stdint.h>

// Main entry point for MUL_MAT_ID kernel (Mixture of Experts)
पूर्णांक entry_point(काष्ठा ggml_et_mul_mat_id_params * params, व्योम * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    अगर (!kernel_env) {
        वापस -1;
    }

    // Get thread coordination info
    पूर्णांक thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    पूर्णांक अंक_threads = get_अंक_threads(kernel_env->shire_mask);

    अगर (thread_id < 0) {
        वापस -1;
    }

    // Use even threads only to avoid resource contention
    // Each minion has 2 threads sharing instruction/data cache, NOC to RAM, and FPU
    // Odd threads return immediately to avoid fighting for shared resources
    अगर (thread_id & 1) {
        वापस 0;  // Odd thread - skip work
    }

    // Adjust thread count and ID for even-only threading
    पूर्णांक effective_thread_id   = thread_id / 2;
    पूर्णांक effective_अंक_threads = (अंक_threads + 1) / 2;  // Ceiling division

    // Validate params
    अगर (params == 0 || ((uint64_t) params & 0x7) != 0) {
        वापस -1;
    }

    // Extract tensor references
    काष्ठा ggml_tensor * src0 = &params->src0;  // Expert weight matrices [K, M, n_expert]
    काष्ठा ggml_tensor * src1 = &params->src1;  // Activations [K, n_expert_used, batch]
    काष्ठा ggml_tensor * src2 = &params->src2;  // Expert indices [n_expert_used, batch] (I32)
    काष्ठा ggml_tensor * dst  = &params->dst;   // Output [M, n_expert_used, batch, 1]

    // Validate tensor types
    अगर (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32) {
        वापस -1;
    }

    // Get data pointers
    स्थिर व्योम *    src0_तथ्य = src0->तथ्य;                    // Expert matrices (Q8_0/F16/F32)
    स्थिर भग्न *   src1_तथ्य = (स्थिर भग्न *) src1->तथ्य;    // Activations (F32)
    स्थिर int32_t * src2_तथ्य = (स्थिर int32_t *) src2->तथ्य;  // Expert IDs (I32)
    भग्न *         dst_तथ्य  = (भग्न *) dst->तथ्य;           // Output (F32)

    अगर (!src0_तथ्य || !src1_तथ्य || !src2_तथ्य || !dst_तथ्य) {
        वापस -1;
    }

    // Determine block size based on src0 type
    पूर्णांक block_आकार_2;
    चयन (src0->type) {
        हाल GGML_TYPE_Q8_0:
            block_आकार_2 = QK8_0;
            अवरोध;
        हाल GGML_TYPE_Q4_0:
            block_आकार_2 = QK4_0;
            अवरोध;
        हाल GGML_TYPE_F16:
            block_आकार_2 = QK_F16;
            अवरोध;
        हाल GGML_TYPE_F32:
            block_आकार_2 = QK_F32;
            अवरोध;
        शेष:
            वापस -1;
    }

    // Get dimensions
    // src0: [K, M, n_expert] - expert weight matrices
    // src1: [K, n_expert_used, batch] - activations
    // src2: [n_expert_used, batch] - expert indices
    // dst:  [M, n_expert_used, batch, 1] - output
    स्थिर int64_t K             = src0->ne[0];  // Hidden dimension
    स्थिर int64_t M             = src0->ne[1];  // Output features
    स्थिर int64_t n_expert      = src0->ne[2];  // Number of experts
    स्थिर int64_t n_expert_used = src2->ne[0];  // Experts used per token
    स्थिर int64_t batch         = src2->ne[1];  // Batch size

    // Strides (in bytes)
    स्थिर माप_प्रकार nb01 = src0->nb[1];  // src0 row stride
    स्थिर माप_प्रकार nb02 = src0->nb[2];  // src0 expert stride
    स्थिर माप_प्रकार nb11 = src1->nb[1];  // src1 column stride
    स्थिर माप_प्रकार nb12 = src1->nb[2];  // src1 batch stride
    स्थिर माप_प्रकार nb20 = src2->nb[0];  // src2 element stride
    स्थिर माप_प्रकार nb21 = src2->nb[1];  // src2 batch stride
    स्थिर माप_प्रकार nb1  = dst->nb[1];   // dst column stride
    स्थिर माप_प्रकार nb2  = dst->nb[2];   // dst batch stride

    // Verify K dimension alignment for quantization
    // Q8_0 requires strict alignment (quantized data must be block-aligned)
    // F32 and F16 can handle partial blocks with scalar remainders
    अगर ((src0->type == GGML_TYPE_Q8_0 || src0->type == GGML_TYPE_Q4_0) && K % block_आकार_2 != 0) {
        वापस -1;  // Q8_0 requires K to be multiple of block_size
    }

    // Verify first dimension is contiguous
    माप_प्रकार expected_element_आकार_src0;
    अगर (src0->type == GGML_TYPE_Q8_0) {
        expected_element_आकार_src0 = माप(block_q8_0);
    } अन्यथा अगर (src0->type == GGML_TYPE_Q4_0) {
        expected_element_आकार_src0 = माप(block_q4_0);
    } अन्यथा अगर (src0->type == GGML_TYPE_F16) {
        expected_element_आकार_src0 = माप(uint16_t);
    } अन्यथा अगर (src0->type == GGML_TYPE_F32) {
        expected_element_आकार_src0 = माप(भग्न);
    } अन्यथा {
        वापस -1;
    }

    अगर (src0->nb[0] != expected_element_आकार_src0 || src1->nb[0] != माप(भग्न) || src2->nb[0] != माप(int32_t) ||
        dst->nb[0] != माप(भग्न)) {
        वापस -1;
    }

    स्थिर int64_t K_blocks = K / block_आकार_2;

    // Threading: distribute output elements across threads
    // Total output elements = M * n_expert_used * batch
    स्थिर uint64_t योग_elements = M * n_expert_used * batch;

    स्थिर uint64_t per_thread     = 16;
    स्थिर uint64_t threads_stride = per_thread * effective_अंक_threads;

    अगर (effective_thread_id * per_thread >= योग_elements) {
        वापस 0;
    }

    // Process elements assigned to this thread
    क्रम (uint64_t base_idx = effective_thread_id * per_thread; base_idx < योग_elements; base_idx += threads_stride) {
        क्रम (uint64_t j = 0; j < per_thread; j++) {
            स्थिर uint64_t idx = base_idx + j;

            अगर (idx >= योग_elements) {
                अवरोध;
            }

            // Decode linear index to (m, n_idx, batch_idx)
            // Layout: m + M * (n_idx + n_expert_used * batch_idx)
            स्थिर int64_t batch_idx = idx / (M * n_expert_used);
            स्थिर int64_t rem       = idx % (M * n_expert_used);
            स्थिर int64_t n_idx     = rem / M;
            स्थिर int64_t m         = rem % M;

            // Get expert ID from src2[n_idx, batch_idx]
            स्थिर int32_t expert_id = *(स्थिर int32_t *) ((स्थिर अक्षर *) src2_तथ्य + n_idx * nb20 + batch_idx * nb21);

            // Validate expert ID
            अगर (expert_id < 0 || expert_id >= n_expert) {
                // Invalid expert ID - write zero and continue
                अस्थिर भग्न * dst_element =
                    (अस्थिर भग्न *) ((अक्षर *) dst_तथ्य + m * dst->nb[0] + n_idx * nb1 + batch_idx * nb2);
                atomic_store_f32(dst_element, 0.0f);
                जारी;
            }

            // Compute dot product: expert_matrix[m, :] x activations[:, col_idx, batch_idx]
            // Use modulo to handle broadcasting: when src1 has fewer columns than expert slots,
            // multiple slots share the same activation column (col_idx = n_idx % src1->ne[1])
            स्थिर int64_t col_idx = n_idx % src1->ne[1];
            भग्न         योग_2     = 0.0f;

            // Type switch hoisted outside block loop: one branch per element, not per block
            स्थिर अक्षर * expert_row_base = (स्थिर अक्षर *) src0_तथ्य + m * nb01 + expert_id * nb02;

            चयन (src0->type) {
                हाल GGML_TYPE_Q8_0:
                    {
                        स्थिर block_q8_0 * q8_row = (स्थिर block_q8_0 *) expert_row_base;
                        स्थिर भग्न *      b_col_base =
                            (स्थिर भग्न *) ((स्थिर अक्षर *) src1_तथ्य + col_idx * nb11 + batch_idx * nb12);
                        योग_2 += compute_row_dot_q8_0(q8_row, b_col_base, K_blocks);
                        अवरोध;
                    }
                हाल GGML_TYPE_Q4_0:
                    {
                        स्थिर block_q4_0 * q4_row = (स्थिर block_q4_0 *) expert_row_base;
                        स्थिर भग्न *      b_col_base =
                            (स्थिर भग्न *) ((स्थिर अक्षर *) src1_तथ्य + col_idx * nb11 + batch_idx * nb12);
                        योग_2 += compute_row_dot_q4_0(q4_row, b_col_base, K_blocks);
                        अवरोध;
                    }
                हाल GGML_TYPE_F16:
                    {
                        स्थिर uint16_t * f16_row     = (स्थिर uint16_t *) expert_row_base;
                        स्थिर int64_t    K_remainder = K % block_आकार_2;
                        क्रम (int64_t kb = 0; kb < K_blocks; kb++) {
                            स्थिर भग्न * b_col_ptr =
                                (स्थिर भग्न *) ((स्थिर अक्षर *) src1_तथ्य + (kb * block_आकार_2) * माप(भग्न) +
                                                 col_idx * nb11 + batch_idx * nb12);
                            योग_2 += compute_block_dot_product_f16_naive(&f16_row[kb * block_आकार_2], b_col_ptr);
                        }
                        अगर (K_remainder > 0) {
                            स्थिर int64_t offset = K_blocks * block_आकार_2;
                            स्थिर भग्न * b_col_ptr =
                                (स्थिर भग्न *) ((स्थिर अक्षर *) src1_तथ्य + offset * माप(भग्न) + col_idx * nb11 +
                                                 batch_idx * nb12);
                            योग_2 += compute_block_dot_product_f16_partial(&f16_row[offset], b_col_ptr, K_remainder);
                        }
                        अवरोध;
                    }
                हाल GGML_TYPE_F32:
                    {
                        स्थिर भग्न * f32_row     = (स्थिर भग्न *) expert_row_base;
                        स्थिर int64_t K_remainder = K % block_आकार_2;
                        क्रम (int64_t kb = 0; kb < K_blocks; kb++) {
                            स्थिर भग्न * b_col_ptr =
                                (स्थिर भग्न *) ((स्थिर अक्षर *) src1_तथ्य + (kb * block_आकार_2) * माप(भग्न) +
                                                 col_idx * nb11 + batch_idx * nb12);
                            योग_2 += compute_block_dot_product_f32(&f32_row[kb * block_आकार_2], b_col_ptr);
                        }
                        अगर (K_remainder > 0) {
                            स्थिर int64_t offset = K_blocks * block_आकार_2;
                            स्थिर भग्न * b_col_ptr =
                                (स्थिर भग्न *) ((स्थिर अक्षर *) src1_तथ्य + offset * माप(भग्न) + col_idx * nb11 +
                                                 batch_idx * nb12);
                            योग_2 += compute_block_dot_product_f32_partial(&f32_row[offset], b_col_ptr, K_remainder);
                        }
                        अवरोध;
                    }
                शेष:
                    वापस -1;
            }

            // Store result using atomic store to avoid cache coherency issues
            // when multiple threads write to the same cache line (64 bytes = 16 floats)
            अस्थिर भग्न * dst_element =
                (अस्थिर भग्न *) ((अक्षर *) dst_तथ्य + m * dst->nb[0] + n_idx * nb1 + batch_idx * nb2);
            atomic_store_f32(dst_element, योग_2);
        }
    }

    वापस 0;
}
