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

#అన్తర్భావ "block_ops.h"
#అన్తర్భావ "ggml_tensor.h"
#అన్తర్భావ "math_fp.h"
#అన్తర్భావ "platform.h"
#అన్తర్భావ "quants.h"

#అన్తర్భావ <stdint.h>

// Main entry point for MUL_MAT_ID kernel (Mixture of Experts)
పూర్ణాంకం నమోదు_బిందువు(నిర్మాణం ggml_et_mul_mat_id_params * params, శూన్య * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ఒకవేళ (!kernel_env) {
        తిరిగి -1;
    }

    // Get thread coordination info
    పూర్ణాంకం thread_id   = get_relative_thread_id(kernel_env->shire_ముసుగు);
    పూర్ణాంకం అంకె_దారాలు = get_అంకె_దారాలు(kernel_env->shire_ముసుగు);

    ఒకవేళ (thread_id < 0) {
        తిరిగి -1;
    }

    // Use even threads only to avoid resource contention
    // Each minion has 2 threads sharing instruction/data cache, NOC to RAM, and FPU
    // Odd threads return immediately to avoid fighting for shared resources
    ఒకవేళ (thread_id & 1) {
        తిరిగి 0;  // Odd thread - skip work
    }

    // Adjust thread count and ID for even-only threading
    పూర్ణాంకం effective_thread_id   = thread_id / 2;
    పూర్ణాంకం effective_అంకె_దారాలు = (అంకె_దారాలు + 1) / 2;  // Ceiling division

    // Validate params
    ఒకవేళ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        తిరిగి -1;
    }

    // Extract tensor references
    నిర్మాణం ggml_tensor * src0 = &params->src0;  // Expert weight matrices [K, M, n_expert]
    నిర్మాణం ggml_tensor * src1 = &params->src1;  // Activations [K, n_expert_used, batch]
    నిర్మాణం ggml_tensor * src2 = &params->src2;  // Expert indices [n_expert_used, batch] (I32)
    నిర్మాణం ggml_tensor * dst  = &params->dst;   // Output [M, n_expert_used, batch, 1]

    // Validate tensor types
    ఒకవేళ (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32) {
        తిరిగి -1;
    }

    // Get data pointers
    స్థిరరాశి శూన్య *    src0_దత్తాంశం = src0->దత్తాంశం;                    // Expert matrices (Q8_0/F16/F32)
    స్థిరరాశి దశాంశం *   src1_దత్తాంశం = (స్థిరరాశి దశాంశం *) src1->దత్తాంశం;    // Activations (F32)
    స్థిరరాశి int32_t * src2_దత్తాంశం = (స్థిరరాశి int32_t *) src2->దత్తాంశం;  // Expert IDs (I32)
    దశాంశం *         dst_దత్తాంశం  = (దశాంశం *) dst->దత్తాంశం;           // Output (F32)

    ఒకవేళ (!src0_దత్తాంశం || !src1_దత్తాంశం || !src2_దత్తాంశం || !dst_దత్తాంశం) {
        తిరిగి -1;
    }

    // Determine block size based on src0 type
    పూర్ణాంకం ఖండం_కొలత_3;
    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_Q8_0:
            ఖండం_కొలత_3 = QK8_0;
            ఆపు;
        స్థితి GGML_TYPE_Q4_0:
            ఖండం_కొలత_3 = QK4_0;
            ఆపు;
        స్థితి GGML_TYPE_F16:
            ఖండం_కొలత_3 = QK_F16;
            ఆపు;
        స్థితి GGML_TYPE_F32:
            ఖండం_కొలత_3 = QK_F32;
            ఆపు;
        అప్రమేయ:
            తిరిగి -1;
    }

    // Get dimensions
    // src0: [K, M, n_expert] - expert weight matrices
    // src1: [K, n_expert_used, batch] - activations
    // src2: [n_expert_used, batch] - expert indices
    // dst:  [M, n_expert_used, batch, 1] - output
    స్థిరరాశి int64_t K             = src0->ne[0];  // Hidden dimension
    స్థిరరాశి int64_t M             = src0->ne[1];  // Output features
    స్థిరరాశి int64_t n_expert      = src0->ne[2];  // Number of experts
    స్థిరరాశి int64_t n_expert_వాడిన = src2->ne[0];  // Experts used per token
    స్థిరరాశి int64_t batch         = src2->ne[1];  // Batch size

    // Strides (in bytes)
    స్థిరరాశి size_t nb01 = src0->nb[1];  // src0 row stride
    స్థిరరాశి size_t nb02 = src0->nb[2];  // src0 expert stride
    స్థిరరాశి size_t nb11 = src1->nb[1];  // src1 column stride
    స్థిరరాశి size_t nb12 = src1->nb[2];  // src1 batch stride
    స్థిరరాశి size_t nb20 = src2->nb[0];  // src2 element stride
    స్థిరరాశి size_t nb21 = src2->nb[1];  // src2 batch stride
    స్థిరరాశి size_t nb1  = dst->nb[1];   // dst column stride
    స్థిరరాశి size_t nb2  = dst->nb[2];   // dst batch stride

    // Verify K dimension alignment for quantization
    // Q8_0 requires strict alignment (quantized data must be block-aligned)
    // F32 and F16 can handle partial blocks with scalar remainders
    ఒకవేళ ((src0->type == GGML_TYPE_Q8_0 || src0->type == GGML_TYPE_Q4_0) && K % ఖండం_కొలత_3 != 0) {
        తిరిగి -1;  // Q8_0 requires K to be multiple of block_size
    }

    // Verify first dimension is contiguous
    size_t expected_element_కొలత_src0;
    ఒకవేళ (src0->type == GGML_TYPE_Q8_0) {
        expected_element_కొలత_src0 = పరిమాణం(ఖండం_q8_0);
    } లేకపోతే ఒకవేళ (src0->type == GGML_TYPE_Q4_0) {
        expected_element_కొలత_src0 = పరిమాణం(ఖండం_q4_0);
    } లేకపోతే ఒకవేళ (src0->type == GGML_TYPE_F16) {
        expected_element_కొలత_src0 = పరిమాణం(uint16_t);
    } లేకపోతే ఒకవేళ (src0->type == GGML_TYPE_F32) {
        expected_element_కొలత_src0 = పరిమాణం(దశాంశం);
    } లేకపోతే {
        తిరిగి -1;
    }

    ఒకవేళ (src0->nb[0] != expected_element_కొలత_src0 || src1->nb[0] != పరిమాణం(దశాంశం) || src2->nb[0] != పరిమాణం(int32_t) ||
        dst->nb[0] != పరిమాణం(దశాంశం)) {
        తిరిగి -1;
    }

    స్థిరరాశి int64_t K_ఖండాలు = K / ఖండం_కొలత_3;

    // Threading: distribute output elements across threads
    // Total output elements = M * n_expert_used * batch
    స్థిరరాశి uint64_t మొత్తం_elements = M * n_expert_వాడిన * batch;

    స్థిరరాశి uint64_t చొప్పున_thread     = 16;
    స్థిరరాశి uint64_t దారాలు_stride = చొప్పున_thread * effective_అంకె_దారాలు;

    ఒకవేళ (effective_thread_id * చొప్పున_thread >= మొత్తం_elements) {
        తిరిగి 0;
    }

    // Process elements assigned to this thread
    కొరకు (uint64_t పునాది_idx = effective_thread_id * చొప్పున_thread; పునాది_idx < మొత్తం_elements; పునాది_idx += దారాలు_stride) {
        కొరకు (uint64_t j = 0; j < చొప్పున_thread; j++) {
            స్థిరరాశి uint64_t idx = పునాది_idx + j;

            ఒకవేళ (idx >= మొత్తం_elements) {
                ఆపు;
            }

            // Decode linear index to (m, n_idx, batch_idx)
            // Layout: m + M * (n_idx + n_expert_used * batch_idx)
            స్థిరరాశి int64_t batch_idx = idx / (M * n_expert_వాడిన);
            స్థిరరాశి int64_t rem       = idx % (M * n_expert_వాడిన);
            స్థిరరాశి int64_t n_idx     = rem / M;
            స్థిరరాశి int64_t m         = rem % M;

            // Get expert ID from src2[n_idx, batch_idx]
            స్థిరరాశి int32_t expert_id = *(స్థిరరాశి int32_t *) ((స్థిరరాశి అక్షరం *) src2_దత్తాంశం + n_idx * nb20 + batch_idx * nb21);

            // Validate expert ID
            ఒకవేళ (expert_id < 0 || expert_id >= n_expert) {
                // Invalid expert ID - write zero and continue
                అస్థిర దశాంశం * dst_element =
                    (అస్థిర దశాంశం *) ((అక్షరం *) dst_దత్తాంశం + m * dst->nb[0] + n_idx * nb1 + batch_idx * nb2);
                atomic_store_f32(dst_element, 0.0f);
                కొనసాగు;
            }

            // Compute dot product: expert_matrix[m, :] x activations[:, col_idx, batch_idx]
            // Use modulo to handle broadcasting: when src1 has fewer columns than expert slots,
            // multiple slots share the same activation column (col_idx = n_idx % src1->ne[1])
            స్థిరరాశి int64_t నిలువు_idx = n_idx % src1->ne[1];
            దశాంశం         కూడిక     = 0.0f;

            // Type switch hoisted outside block loop: one branch per element, not per block
            స్థిరరాశి అక్షరం * expert_అడ్డువరుస_పునాది = (స్థిరరాశి అక్షరం *) src0_దత్తాంశం + m * nb01 + expert_id * nb02;

            ఎంపిక (src0->type) {
                స్థితి GGML_TYPE_Q8_0:
                    {
                        స్థిరరాశి ఖండం_q8_0 * q8_అడ్డువరుస = (స్థిరరాశి ఖండం_q8_0 *) expert_అడ్డువరుస_పునాది;
                        స్థిరరాశి దశాంశం *      b_నిలువు_పునాది =
                            (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) src1_దత్తాంశం + నిలువు_idx * nb11 + batch_idx * nb12);
                        కూడిక += compute_అడ్డువరుస_dot_q8_0(q8_అడ్డువరుస, b_నిలువు_పునాది, K_ఖండాలు);
                        ఆపు;
                    }
                స్థితి GGML_TYPE_Q4_0:
                    {
                        స్థిరరాశి ఖండం_q4_0 * q4_అడ్డువరుస = (స్థిరరాశి ఖండం_q4_0 *) expert_అడ్డువరుస_పునాది;
                        స్థిరరాశి దశాంశం *      b_నిలువు_పునాది =
                            (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) src1_దత్తాంశం + నిలువు_idx * nb11 + batch_idx * nb12);
                        కూడిక += compute_అడ్డువరుస_dot_q4_0(q4_అడ్డువరుస, b_నిలువు_పునాది, K_ఖండాలు);
                        ఆపు;
                    }
                స్థితి GGML_TYPE_F16:
                    {
                        స్థిరరాశి uint16_t * f16_అడ్డువరుస     = (స్థిరరాశి uint16_t *) expert_అడ్డువరుస_పునాది;
                        స్థిరరాశి int64_t    K_remainder = K % ఖండం_కొలత_3;
                        కొరకు (int64_t kb = 0; kb < K_ఖండాలు; kb++) {
                            స్థిరరాశి దశాంశం * b_నిలువు_ptr =
                                (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) src1_దత్తాంశం + (kb * ఖండం_కొలత_3) * పరిమాణం(దశాంశం) +
                                                 నిలువు_idx * nb11 + batch_idx * nb12);
                            కూడిక += compute_ఖండం_dot_లబ్ధం_f16_naive(&f16_అడ్డువరుస[kb * ఖండం_కొలత_3], b_నిలువు_ptr);
                        }
                        ఒకవేళ (K_remainder > 0) {
                            స్థిరరాశి int64_t స్థానభ్రంశం = K_ఖండాలు * ఖండం_కొలత_3;
                            స్థిరరాశి దశాంశం * b_నిలువు_ptr =
                                (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) src1_దత్తాంశం + స్థానభ్రంశం * పరిమాణం(దశాంశం) + నిలువు_idx * nb11 +
                                                 batch_idx * nb12);
                            కూడిక += compute_ఖండం_dot_లబ్ధం_f16_partial(&f16_అడ్డువరుస[స్థానభ్రంశం], b_నిలువు_ptr, K_remainder);
                        }
                        ఆపు;
                    }
                స్థితి GGML_TYPE_F32:
                    {
                        స్థిరరాశి దశాంశం * f32_అడ్డువరుస     = (స్థిరరాశి దశాంశం *) expert_అడ్డువరుస_పునాది;
                        స్థిరరాశి int64_t K_remainder = K % ఖండం_కొలత_3;
                        కొరకు (int64_t kb = 0; kb < K_ఖండాలు; kb++) {
                            స్థిరరాశి దశాంశం * b_నిలువు_ptr =
                                (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) src1_దత్తాంశం + (kb * ఖండం_కొలత_3) * పరిమాణం(దశాంశం) +
                                                 నిలువు_idx * nb11 + batch_idx * nb12);
                            కూడిక += compute_ఖండం_dot_లబ్ధం_f32(&f32_అడ్డువరుస[kb * ఖండం_కొలత_3], b_నిలువు_ptr);
                        }
                        ఒకవేళ (K_remainder > 0) {
                            స్థిరరాశి int64_t స్థానభ్రంశం = K_ఖండాలు * ఖండం_కొలత_3;
                            స్థిరరాశి దశాంశం * b_నిలువు_ptr =
                                (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) src1_దత్తాంశం + స్థానభ్రంశం * పరిమాణం(దశాంశం) + నిలువు_idx * nb11 +
                                                 batch_idx * nb12);
                            కూడిక += compute_ఖండం_dot_లబ్ధం_f32_partial(&f32_అడ్డువరుస[స్థానభ్రంశం], b_నిలువు_ptr, K_remainder);
                        }
                        ఆపు;
                    }
                అప్రమేయ:
                    తిరిగి -1;
            }

            // Store result using atomic store to avoid cache coherency issues
            // when multiple threads write to the same cache line (64 bytes = 16 floats)
            అస్థిర దశాంశం * dst_element =
                (అస్థిర దశాంశం *) ((అక్షరం *) dst_దత్తాంశం + m * dst->nb[0] + n_idx * nb1 + batch_idx * nb2);
            atomic_store_f32(dst_element, కూడిక);
        }
    }

    తిరిగి 0;
}
