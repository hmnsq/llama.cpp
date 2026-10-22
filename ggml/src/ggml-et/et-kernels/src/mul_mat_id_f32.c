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

#ಅನ್ತರ್ಭಾವ "block_ops.h"
#ಅನ್ತರ್ಭಾವ "ggml_tensor.h"
#ಅನ್ತರ್ಭಾವ "math_fp.h"
#ಅನ್ತರ್ಭಾವ "platform.h"
#ಅನ್ತರ್ಭಾವ "quants.h"

#ಅನ್ತರ್ಭಾವ <stdint.h>

// Main entry point for MUL_MAT_ID kernel (Mixture of Experts)
ಪೂರ್ಣಾಂಕ ನಮೂದು_ಬಿಂದು(ರಚನೆ ggml_et_mul_mat_id_params * params, ಶೂನ್ಯ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ಆದರೆ (!kernel_env) {
        ಹಿಂತಿರುಗಿ -1;
    }

    // Get thread coordination info
    ಪೂರ್ಣಾಂಕ thread_id   = get_relative_thread_id(kernel_env->shire_ಮುಖವಾಡ);
    ಪೂರ್ಣಾಂಕ ಅಂಕೆ_ಎಳೆಗಳು = get_ಅಂಕೆ_ಎಳೆಗಳು(kernel_env->shire_ಮುಖವಾಡ);

    ಆದರೆ (thread_id < 0) {
        ಹಿಂತಿರುಗಿ -1;
    }

    // Use even threads only to avoid resource contention
    // Each minion has 2 threads sharing instruction/data cache, NOC to RAM, and FPU
    // Odd threads return immediately to avoid fighting for shared resources
    ಆದರೆ (thread_id & 1) {
        ಹಿಂತಿರುಗಿ 0;  // Odd thread - skip work
    }

    // Adjust thread count and ID for even-only threading
    ಪೂರ್ಣಾಂಕ effective_thread_id   = thread_id / 2;
    ಪೂರ್ಣಾಂಕ effective_ಅಂಕೆ_ಎಳೆಗಳು = (ಅಂಕೆ_ಎಳೆಗಳು + 1) / 2;  // Ceiling division

    // Validate params
    ಆದರೆ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ಹಿಂತಿರುಗಿ -1;
    }

    // Extract tensor references
    ರಚನೆ ggml_tensor * src0 = &params->src0;  // Expert weight matrices [K, M, n_expert]
    ರಚನೆ ggml_tensor * src1 = &params->src1;  // Activations [K, n_expert_used, batch]
    ರಚನೆ ggml_tensor * src2 = &params->src2;  // Expert indices [n_expert_used, batch] (I32)
    ರಚನೆ ggml_tensor * dst  = &params->dst;   // Output [M, n_expert_used, batch, 1]

    // Validate tensor types
    ಆದರೆ (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32) {
        ಹಿಂತಿರುಗಿ -1;
    }

    // Get data pointers
    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *    src0_ದತ್ತಾಂಶ = src0->ದತ್ತಾಂಶ;                    // Expert matrices (Q8_0/F16/F32)
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *   src1_ದತ್ತಾಂಶ = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src1->ದತ್ತಾಂಶ;    // Activations (F32)
    ಸ್ಥಿರಾಂಕ int32_t * src2_ದತ್ತಾಂಶ = (ಸ್ಥಿರಾಂಕ int32_t *) src2->ದತ್ತಾಂಶ;  // Expert IDs (I32)
    ದಶಮಾಂಶ *         dst_ದತ್ತಾಂಶ  = (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ;           // Output (F32)

    ಆದರೆ (!src0_ದತ್ತಾಂಶ || !src1_ದತ್ತಾಂಶ || !src2_ದತ್ತಾಂಶ || !dst_ದತ್ತಾಂಶ) {
        ಹಿಂತಿರುಗಿ -1;
    }

    // Determine block size based on src0 type
    ಪೂರ್ಣಾಂಕ ಖಂಡ_ಗಾತ್ರ_3;
    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
            ಖಂಡ_ಗಾತ್ರ_3 = QK8_0;
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
            ಖಂಡ_ಗಾತ್ರ_3 = QK4_0;
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F16:
            ಖಂಡ_ಗಾತ್ರ_3 = QK_F16;
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            ಖಂಡ_ಗಾತ್ರ_3 = QK_F32;
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ -1;
    }

    // Get dimensions
    // src0: [K, M, n_expert] - expert weight matrices
    // src1: [K, n_expert_used, batch] - activations
    // src2: [n_expert_used, batch] - expert indices
    // dst:  [M, n_expert_used, batch, 1] - output
    ಸ್ಥಿರಾಂಕ int64_t K             = src0->ne[0];  // Hidden dimension
    ಸ್ಥಿರಾಂಕ int64_t M             = src0->ne[1];  // Output features
    ಸ್ಥಿರಾಂಕ int64_t n_expert      = src0->ne[2];  // Number of experts
    ಸ್ಥಿರಾಂಕ int64_t n_expert_ಬಳಸಿದ = src2->ne[0];  // Experts used per token
    ಸ್ಥಿರಾಂಕ int64_t batch         = src2->ne[1];  // Batch size

    // Strides (in bytes)
    ಸ್ಥಿರಾಂಕ size_t nb01 = src0->nb[1];  // src0 row stride
    ಸ್ಥಿರಾಂಕ size_t nb02 = src0->nb[2];  // src0 expert stride
    ಸ್ಥಿರಾಂಕ size_t nb11 = src1->nb[1];  // src1 column stride
    ಸ್ಥಿರಾಂಕ size_t nb12 = src1->nb[2];  // src1 batch stride
    ಸ್ಥಿರಾಂಕ size_t nb20 = src2->nb[0];  // src2 element stride
    ಸ್ಥಿರಾಂಕ size_t nb21 = src2->nb[1];  // src2 batch stride
    ಸ್ಥಿರಾಂಕ size_t nb1  = dst->nb[1];   // dst column stride
    ಸ್ಥಿರಾಂಕ size_t nb2  = dst->nb[2];   // dst batch stride

    // Verify K dimension alignment for quantization
    // Q8_0 requires strict alignment (quantized data must be block-aligned)
    // F32 and F16 can handle partial blocks with scalar remainders
    ಆದರೆ ((src0->type == GGML_TYPE_Q8_0 || src0->type == GGML_TYPE_Q4_0) && K % ಖಂಡ_ಗಾತ್ರ_3 != 0) {
        ಹಿಂತಿರುಗಿ -1;  // Q8_0 requires K to be multiple of block_size
    }

    // Verify first dimension is contiguous
    size_t expected_element_ಗಾತ್ರ_src0;
    ಆದರೆ (src0->type == GGML_TYPE_Q8_0) {
        expected_element_ಗಾತ್ರ_src0 = ಗಾತ್ರ(ಖಂಡ_q8_0);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0->type == GGML_TYPE_Q4_0) {
        expected_element_ಗಾತ್ರ_src0 = ಗಾತ್ರ(ಖಂಡ_q4_0);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0->type == GGML_TYPE_F16) {
        expected_element_ಗಾತ್ರ_src0 = ಗಾತ್ರ(uint16_t);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0->type == GGML_TYPE_F32) {
        expected_element_ಗಾತ್ರ_src0 = ಗಾತ್ರ(ದಶಮಾಂಶ);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಆದರೆ (src0->nb[0] != expected_element_ಗಾತ್ರ_src0 || src1->nb[0] != ಗಾತ್ರ(ದಶಮಾಂಶ) || src2->nb[0] != ಗಾತ್ರ(int32_t) ||
        dst->nb[0] != ಗಾತ್ರ(ದಶಮಾಂಶ)) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಸ್ಥಿರಾಂಕ int64_t K_ಖಂಡಗಳು = K / ಖಂಡ_ಗಾತ್ರ_3;

    // Threading: distribute output elements across threads
    // Total output elements = M * n_expert_used * batch
    ಸ್ಥಿರಾಂಕ uint64_t ಒಟ್ಟು_elements = M * n_expert_ಬಳಸಿದ * batch;

    ಸ್ಥಿರಾಂಕ uint64_t ಪ್ರತಿ_thread     = 16;
    ಸ್ಥಿರಾಂಕ uint64_t ಎಳೆಗಳು_stride = ಪ್ರತಿ_thread * effective_ಅಂಕೆ_ಎಳೆಗಳು;

    ಆದರೆ (effective_thread_id * ಪ್ರತಿ_thread >= ಒಟ್ಟು_elements) {
        ಹಿಂತಿರುಗಿ 0;
    }

    // Process elements assigned to this thread
    ಗಾಗಿ (uint64_t ತಳ_idx = effective_thread_id * ಪ್ರತಿ_thread; ತಳ_idx < ಒಟ್ಟು_elements; ತಳ_idx += ಎಳೆಗಳು_stride) {
        ಗಾಗಿ (uint64_t j = 0; j < ಪ್ರತಿ_thread; j++) {
            ಸ್ಥಿರಾಂಕ uint64_t idx = ತಳ_idx + j;

            ಆದರೆ (idx >= ಒಟ್ಟು_elements) {
                ನಿಲ್ಲಿಸು;
            }

            // Decode linear index to (m, n_idx, batch_idx)
            // Layout: m + M * (n_idx + n_expert_used * batch_idx)
            ಸ್ಥಿರಾಂಕ int64_t batch_idx = idx / (M * n_expert_ಬಳಸಿದ);
            ಸ್ಥಿರಾಂಕ int64_t rem       = idx % (M * n_expert_ಬಳಸಿದ);
            ಸ್ಥಿರಾಂಕ int64_t n_idx     = rem / M;
            ಸ್ಥಿರಾಂಕ int64_t m         = rem % M;

            // Get expert ID from src2[n_idx, batch_idx]
            ಸ್ಥಿರಾಂಕ int32_t expert_id = *(ಸ್ಥಿರಾಂಕ int32_t *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src2_ದತ್ತಾಂಶ + n_idx * nb20 + batch_idx * nb21);

            // Validate expert ID
            ಆದರೆ (expert_id < 0 || expert_id >= n_expert) {
                // Invalid expert ID - write zero and continue
                ಅಸ್ಥಿರ ದಶಮಾಂಶ * dst_element =
                    (ಅಸ್ಥಿರ ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst_ದತ್ತಾಂಶ + m * dst->nb[0] + n_idx * nb1 + batch_idx * nb2);
                atomic_store_f32(dst_element, 0.0f);
                ಮುಂದುವರಿ;
            }

            // Compute dot product: expert_matrix[m, :] x activations[:, col_idx, batch_idx]
            // Use modulo to handle broadcasting: when src1 has fewer columns than expert slots,
            // multiple slots share the same activation column (col_idx = n_idx % src1->ne[1])
            ಸ್ಥಿರಾಂಕ int64_t ಲಂಬಸಾಲು_idx = n_idx % src1->ne[1];
            ದಶಮಾಂಶ         ಮೊತ್ತ     = 0.0f;

            // Type switch hoisted outside block loop: one branch per element, not per block
            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * expert_ಅಡ್ಡಸಾಲು_ತಳ = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src0_ದತ್ತಾಂಶ + m * nb01 + expert_id * nb02;

            ಆಯ್ಕೆ (src0->type) {
                ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
                    {
                        ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 * q8_ಅಡ್ಡಸಾಲು = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 *) expert_ಅಡ್ಡಸಾಲು_ತಳ;
                        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *      b_ಲಂಬಸಾಲು_ತಳ =
                            (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src1_ದತ್ತಾಂಶ + ಲಂಬಸಾಲು_idx * nb11 + batch_idx * nb12);
                        ಮೊತ್ತ += compute_ಅಡ್ಡಸಾಲು_dot_q8_0(q8_ಅಡ್ಡಸಾಲು, b_ಲಂಬಸಾಲು_ತಳ, K_ಖಂಡಗಳು);
                        ನಿಲ್ಲಿಸು;
                    }
                ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
                    {
                        ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0 * q4_ಅಡ್ಡಸಾಲು = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0 *) expert_ಅಡ್ಡಸಾಲು_ತಳ;
                        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *      b_ಲಂಬಸಾಲು_ತಳ =
                            (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src1_ದತ್ತಾಂಶ + ಲಂಬಸಾಲು_idx * nb11 + batch_idx * nb12);
                        ಮೊತ್ತ += compute_ಅಡ್ಡಸಾಲು_dot_q4_0(q4_ಅಡ್ಡಸಾಲು, b_ಲಂಬಸಾಲು_ತಳ, K_ಖಂಡಗಳು);
                        ನಿಲ್ಲಿಸು;
                    }
                ಸ್ಥಿತಿ GGML_TYPE_F16:
                    {
                        ಸ್ಥಿರಾಂಕ uint16_t * f16_ಅಡ್ಡಸಾಲು     = (ಸ್ಥಿರಾಂಕ uint16_t *) expert_ಅಡ್ಡಸಾಲು_ತಳ;
                        ಸ್ಥಿರಾಂಕ int64_t    K_remainder = K % ಖಂಡ_ಗಾತ್ರ_3;
                        ಗಾಗಿ (int64_t kb = 0; kb < K_ಖಂಡಗಳು; kb++) {
                            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * b_ಲಂಬಸಾಲು_ptr =
                                (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src1_ದತ್ತಾಂಶ + (kb * ಖಂಡ_ಗಾತ್ರ_3) * ಗಾತ್ರ(ದಶಮಾಂಶ) +
                                                 ಲಂಬಸಾಲು_idx * nb11 + batch_idx * nb12);
                            ಮೊತ್ತ += compute_ಖಂಡ_dot_ಗುಣಲಬ್ಧ_f16_naive(&f16_ಅಡ್ಡಸಾಲು[kb * ಖಂಡ_ಗಾತ್ರ_3], b_ಲಂಬಸಾಲು_ptr);
                        }
                        ಆದರೆ (K_remainder > 0) {
                            ಸ್ಥಿರಾಂಕ int64_t ಸ್ಥಾನಾಂತರ = K_ಖಂಡಗಳು * ಖಂಡ_ಗಾತ್ರ_3;
                            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * b_ಲಂಬಸಾಲು_ptr =
                                (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src1_ದತ್ತಾಂಶ + ಸ್ಥಾನಾಂತರ * ಗಾತ್ರ(ದಶಮಾಂಶ) + ಲಂಬಸಾಲು_idx * nb11 +
                                                 batch_idx * nb12);
                            ಮೊತ್ತ += compute_ಖಂಡ_dot_ಗುಣಲಬ್ಧ_f16_partial(&f16_ಅಡ್ಡಸಾಲು[ಸ್ಥಾನಾಂತರ], b_ಲಂಬಸಾಲು_ptr, K_remainder);
                        }
                        ನಿಲ್ಲಿಸು;
                    }
                ಸ್ಥಿತಿ GGML_TYPE_F32:
                    {
                        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * f32_ಅಡ್ಡಸಾಲು     = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) expert_ಅಡ್ಡಸಾಲು_ತಳ;
                        ಸ್ಥಿರಾಂಕ int64_t K_remainder = K % ಖಂಡ_ಗಾತ್ರ_3;
                        ಗಾಗಿ (int64_t kb = 0; kb < K_ಖಂಡಗಳು; kb++) {
                            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * b_ಲಂಬಸಾಲು_ptr =
                                (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src1_ದತ್ತಾಂಶ + (kb * ಖಂಡ_ಗಾತ್ರ_3) * ಗಾತ್ರ(ದಶಮಾಂಶ) +
                                                 ಲಂಬಸಾಲು_idx * nb11 + batch_idx * nb12);
                            ಮೊತ್ತ += compute_ಖಂಡ_dot_ಗುಣಲಬ್ಧ_f32(&f32_ಅಡ್ಡಸಾಲು[kb * ಖಂಡ_ಗಾತ್ರ_3], b_ಲಂಬಸಾಲು_ptr);
                        }
                        ಆದರೆ (K_remainder > 0) {
                            ಸ್ಥಿರಾಂಕ int64_t ಸ್ಥಾನಾಂತರ = K_ಖಂಡಗಳು * ಖಂಡ_ಗಾತ್ರ_3;
                            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * b_ಲಂಬಸಾಲು_ptr =
                                (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src1_ದತ್ತಾಂಶ + ಸ್ಥಾನಾಂತರ * ಗಾತ್ರ(ದಶಮಾಂಶ) + ಲಂಬಸಾಲು_idx * nb11 +
                                                 batch_idx * nb12);
                            ಮೊತ್ತ += compute_ಖಂಡ_dot_ಗುಣಲಬ್ಧ_f32_partial(&f32_ಅಡ್ಡಸಾಲು[ಸ್ಥಾನಾಂತರ], b_ಲಂಬಸಾಲು_ptr, K_remainder);
                        }
                        ನಿಲ್ಲಿಸು;
                    }
                ಪೂರ್ವನಿಯೋಜಿತ:
                    ಹಿಂತಿರುಗಿ -1;
            }

            // Store result using atomic store to avoid cache coherency issues
            // when multiple threads write to the same cache line (64 bytes = 16 floats)
            ಅಸ್ಥಿರ ದಶಮಾಂಶ * dst_element =
                (ಅಸ್ಥಿರ ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst_ದತ್ತಾಂಶ + m * dst->nb[0] + n_idx * nb1 + batch_idx * nb2);
            atomic_store_f32(dst_element, ಮೊತ್ತ);
        }
    }

    ಹಿಂತಿರುಗಿ 0;
}
