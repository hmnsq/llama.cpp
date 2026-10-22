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

#अन्तर्भाव "block_ops.h"
#अन्तर्भाव "ggml_tensor.h"
#अन्तर्भाव "math_fp.h"
#अन्तर्भाव "platform.h"
#अन्तर्भाव "quants.h"

#अन्तर्भाव <stdint.h>

// Main entry point for MUL_MAT_ID kernel (Mixture of Experts)
पूर्णाङ्क प्रविष्टिः_point(संरचना ggml_et_mul_mat_id_params * params, शून्य * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    यदि (!kernel_env) {
        प्रत्यागम -1;
    }

    // Get thread coordination info
    पूर्णाङ्क thread_id   = get_relative_thread_id(kernel_env->shire_आवरणम्);
    पूर्णाङ्क अङ्कः_तन्तवः = get_अङ्कः_तन्तवः(kernel_env->shire_आवरणम्);

    यदि (thread_id < 0) {
        प्रत्यागम -1;
    }

    // Use even threads only to avoid resource contention
    // Each minion has 2 threads sharing instruction/data cache, NOC to RAM, and FPU
    // Odd threads return immediately to avoid fighting for shared resources
    यदि (thread_id & 1) {
        प्रत्यागम 0;  // Odd thread - skip work
    }

    // Adjust thread count and ID for even-only threading
    पूर्णाङ्क effective_thread_id   = thread_id / 2;
    पूर्णाङ्क effective_अङ्कः_तन्तवः = (अङ्कः_तन्तवः + 1) / 2;  // Ceiling division

    // Validate params
    यदि (params == 0 || ((uint64_t) params & 0x7) != 0) {
        प्रत्यागम -1;
    }

    // Extract tensor references
    संरचना ggml_tensor * src0 = &params->src0;  // Expert weight matrices [K, M, n_expert]
    संरचना ggml_tensor * src1 = &params->src1;  // Activations [K, n_expert_used, batch]
    संरचना ggml_tensor * src2 = &params->src2;  // Expert indices [n_expert_used, batch] (I32)
    संरचना ggml_tensor * dst  = &params->dst;   // Output [M, n_expert_used, batch, 1]

    // Validate tensor types
    यदि (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32) {
        प्रत्यागम -1;
    }

    // Get data pointers
    ध्रुव शून्य *    src0_दत्त = src0->दत्त;                    // Expert matrices (Q8_0/F16/F32)
    ध्रुव भिन्न *   src1_दत्त = (ध्रुव भिन्न *) src1->दत्त;    // Activations (F32)
    ध्रुव int32_t * src2_दत्त = (ध्रुव int32_t *) src2->दत्त;  // Expert IDs (I32)
    भिन्न *         dst_दत्त  = (भिन्न *) dst->दत्त;           // Output (F32)

    यदि (!src0_दत्त || !src1_दत्त || !src2_दत्त || !dst_दत्त) {
        प्रत्यागम -1;
    }

    // Determine block size based on src0 type
    पूर्णाङ्क खण्डम्_आकार_3;
    विकल्प (src0->type) {
        पक्ष GGML_TYPE_Q8_0:
            खण्डम्_आकार_3 = QK8_0;
            भङ्ग;
        पक्ष GGML_TYPE_Q4_0:
            खण्डम्_आकार_3 = QK4_0;
            भङ्ग;
        पक्ष GGML_TYPE_F16:
            खण्डम्_आकार_3 = QK_F16;
            भङ्ग;
        पक्ष GGML_TYPE_F32:
            खण्डम्_आकार_3 = QK_F32;
            भङ्ग;
        स्वभाव:
            प्रत्यागम -1;
    }

    // Get dimensions
    // src0: [K, M, n_expert] - expert weight matrices
    // src1: [K, n_expert_used, batch] - activations
    // src2: [n_expert_used, batch] - expert indices
    // dst:  [M, n_expert_used, batch, 1] - output
    ध्रुव int64_t K             = src0->ne[0];  // Hidden dimension
    ध्रुव int64_t M             = src0->ne[1];  // Output features
    ध्रुव int64_t n_expert      = src0->ne[2];  // Number of experts
    ध्रुव int64_t n_expert_प्रयुक्तम् = src2->ne[0];  // Experts used per token
    ध्रुव int64_t batch         = src2->ne[1];  // Batch size

    // Strides (in bytes)
    ध्रुव size_t nb01 = src0->nb[1];  // src0 row stride
    ध्रुव size_t nb02 = src0->nb[2];  // src0 expert stride
    ध्रुव size_t nb11 = src1->nb[1];  // src1 column stride
    ध्रुव size_t nb12 = src1->nb[2];  // src1 batch stride
    ध्रुव size_t nb20 = src2->nb[0];  // src2 element stride
    ध्रुव size_t nb21 = src2->nb[1];  // src2 batch stride
    ध्रुव size_t nb1  = dst->nb[1];   // dst column stride
    ध्रुव size_t nb2  = dst->nb[2];   // dst batch stride

    // Verify K dimension alignment for quantization
    // Q8_0 requires strict alignment (quantized data must be block-aligned)
    // F32 and F16 can handle partial blocks with scalar remainders
    यदि ((src0->type == GGML_TYPE_Q8_0 || src0->type == GGML_TYPE_Q4_0) && K % खण्डम्_आकार_3 != 0) {
        प्रत्यागम -1;  // Q8_0 requires K to be multiple of block_size
    }

    // Verify first dimension is contiguous
    size_t expected_element_आकार_src0;
    यदि (src0->type == GGML_TYPE_Q8_0) {
        expected_element_आकार_src0 = परिमाण(खण्डम्_q8_0);
    } अन्यथा यदि (src0->type == GGML_TYPE_Q4_0) {
        expected_element_आकार_src0 = परिमाण(खण्डम्_q4_0);
    } अन्यथा यदि (src0->type == GGML_TYPE_F16) {
        expected_element_आकार_src0 = परिमाण(uint16_t);
    } अन्यथा यदि (src0->type == GGML_TYPE_F32) {
        expected_element_आकार_src0 = परिमाण(भिन्न);
    } अन्यथा {
        प्रत्यागम -1;
    }

    यदि (src0->nb[0] != expected_element_आकार_src0 || src1->nb[0] != परिमाण(भिन्न) || src2->nb[0] != परिमाण(int32_t) ||
        dst->nb[0] != परिमाण(भिन्न)) {
        प्रत्यागम -1;
    }

    ध्रुव int64_t K_खण्डानि = K / खण्डम्_आकार_3;

    // Threading: distribute output elements across threads
    // Total output elements = M * n_expert_used * batch
    ध्रुव uint64_t योग_elements = M * n_expert_प्रयुक्तम् * batch;

    ध्रुव uint64_t प्रत्येकम्_thread     = 16;
    ध्रुव uint64_t तन्तवः_stride = प्रत्येकम्_thread * effective_अङ्कः_तन्तवः;

    यदि (effective_thread_id * प्रत्येकम्_thread >= योग_elements) {
        प्रत्यागम 0;
    }

    // Process elements assigned to this thread
    प्रति (uint64_t आधारः_idx = effective_thread_id * प्रत्येकम्_thread; आधारः_idx < योग_elements; आधारः_idx += तन्तवः_stride) {
        प्रति (uint64_t j = 0; j < प्रत्येकम्_thread; j++) {
            ध्रुव uint64_t idx = आधारः_idx + j;

            यदि (idx >= योग_elements) {
                भङ्ग;
            }

            // Decode linear index to (m, n_idx, batch_idx)
            // Layout: m + M * (n_idx + n_expert_used * batch_idx)
            ध्रुव int64_t batch_idx = idx / (M * n_expert_प्रयुक्तम्);
            ध्रुव int64_t rem       = idx % (M * n_expert_प्रयुक्तम्);
            ध्रुव int64_t n_idx     = rem / M;
            ध्रुव int64_t m         = rem % M;

            // Get expert ID from src2[n_idx, batch_idx]
            ध्रुव int32_t expert_id = *(ध्रुव int32_t *) ((ध्रुव वर्ण *) src2_दत्त + n_idx * nb20 + batch_idx * nb21);

            // Validate expert ID
            यदि (expert_id < 0 || expert_id >= n_expert) {
                // Invalid expert ID - write zero and continue
                चञ्चल भिन्न * dst_element =
                    (चञ्चल भिन्न *) ((वर्ण *) dst_दत्त + m * dst->nb[0] + n_idx * nb1 + batch_idx * nb2);
                atomic_store_f32(dst_element, 0.0f);
                अनुवर्त;
            }

            // Compute dot product: expert_matrix[m, :] x activations[:, col_idx, batch_idx]
            // Use modulo to handle broadcasting: when src1 has fewer columns than expert slots,
            // multiple slots share the same activation column (col_idx = n_idx % src1->ne[1])
            ध्रुव int64_t col_idx = n_idx % src1->ne[1];
            भिन्न         योग_2     = 0.0f;

            // Type switch hoisted outside block loop: one branch per element, not per block
            ध्रुव वर्ण * expert_row_आधारः = (ध्रुव वर्ण *) src0_दत्त + m * nb01 + expert_id * nb02;

            विकल्प (src0->type) {
                पक्ष GGML_TYPE_Q8_0:
                    {
                        ध्रुव खण्डम्_q8_0 * q8_row = (ध्रुव खण्डम्_q8_0 *) expert_row_आधारः;
                        ध्रुव भिन्न *      b_col_आधारः =
                            (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src1_दत्त + col_idx * nb11 + batch_idx * nb12);
                        योग_2 += compute_row_dot_q8_0(q8_row, b_col_आधारः, K_खण्डानि);
                        भङ्ग;
                    }
                पक्ष GGML_TYPE_Q4_0:
                    {
                        ध्रुव खण्डम्_q4_0 * q4_row = (ध्रुव खण्डम्_q4_0 *) expert_row_आधारः;
                        ध्रुव भिन्न *      b_col_आधारः =
                            (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src1_दत्त + col_idx * nb11 + batch_idx * nb12);
                        योग_2 += compute_row_dot_q4_0(q4_row, b_col_आधारः, K_खण्डानि);
                        भङ्ग;
                    }
                पक्ष GGML_TYPE_F16:
                    {
                        ध्रुव uint16_t * f16_row     = (ध्रुव uint16_t *) expert_row_आधारः;
                        ध्रुव int64_t    K_remainder = K % खण्डम्_आकार_3;
                        प्रति (int64_t kb = 0; kb < K_खण्डानि; kb++) {
                            ध्रुव भिन्न * b_col_ptr =
                                (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src1_दत्त + (kb * खण्डम्_आकार_3) * परिमाण(भिन्न) +
                                                 col_idx * nb11 + batch_idx * nb12);
                            योग_2 += compute_खण्डम्_dot_गुणनफलम्_f16_naive(&f16_row[kb * खण्डम्_आकार_3], b_col_ptr);
                        }
                        यदि (K_remainder > 0) {
                            ध्रुव int64_t विस्थापनम् = K_खण्डानि * खण्डम्_आकार_3;
                            ध्रुव भिन्न * b_col_ptr =
                                (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src1_दत्त + विस्थापनम् * परिमाण(भिन्न) + col_idx * nb11 +
                                                 batch_idx * nb12);
                            योग_2 += compute_खण्डम्_dot_गुणनफलम्_f16_partial(&f16_row[विस्थापनम्], b_col_ptr, K_remainder);
                        }
                        भङ्ग;
                    }
                पक्ष GGML_TYPE_F32:
                    {
                        ध्रुव भिन्न * f32_row     = (ध्रुव भिन्न *) expert_row_आधारः;
                        ध्रुव int64_t K_remainder = K % खण्डम्_आकार_3;
                        प्रति (int64_t kb = 0; kb < K_खण्डानि; kb++) {
                            ध्रुव भिन्न * b_col_ptr =
                                (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src1_दत्त + (kb * खण्डम्_आकार_3) * परिमाण(भिन्न) +
                                                 col_idx * nb11 + batch_idx * nb12);
                            योग_2 += compute_खण्डम्_dot_गुणनफलम्_f32(&f32_row[kb * खण्डम्_आकार_3], b_col_ptr);
                        }
                        यदि (K_remainder > 0) {
                            ध्रुव int64_t विस्थापनम् = K_खण्डानि * खण्डम्_आकार_3;
                            ध्रुव भिन्न * b_col_ptr =
                                (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src1_दत्त + विस्थापनम् * परिमाण(भिन्न) + col_idx * nb11 +
                                                 batch_idx * nb12);
                            योग_2 += compute_खण्डम्_dot_गुणनफलम्_f32_partial(&f32_row[विस्थापनम्], b_col_ptr, K_remainder);
                        }
                        भङ्ग;
                    }
                स्वभाव:
                    प्रत्यागम -1;
            }

            // Store result using atomic store to avoid cache coherency issues
            // when multiple threads write to the same cache line (64 bytes = 16 floats)
            चञ्चल भिन्न * dst_element =
                (चञ्चल भिन्न *) ((वर्ण *) dst_दत्त + m * dst->nb[0] + n_idx * nb1 + batch_idx * nb2);
            atomic_store_f32(dst_element, योग_2);
        }
    }

    प्रत्यागम 0;
}
