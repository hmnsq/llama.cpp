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

#ਅਨ੍ਤਰ੍ਭਾਵ "block_ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml_tensor.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "math_fp.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "platform.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "quants.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>

// Main entry point for MUL_MAT_ID kernel (Mixture of Experts)
ਪੂਰਨ ਇੰਦਰਾਜ_ਬਿੰਦੂ(ਬਣਤਰ ggml_et_mul_mat_id_params * params, ਖਾਲੀ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ਜੇ (!kernel_env) {
        ਵਾਪਸ -1;
    }

    // Get thread coordination info
    ਪੂਰਨ thread_id   = get_relative_thread_id(kernel_env->shire_ਨਕਾਬ);
    ਪੂਰਨ ਅੰਕ_ਧਾਗੇ = get_ਅੰਕ_ਧਾਗੇ(kernel_env->shire_ਨਕਾਬ);

    ਜੇ (thread_id < 0) {
        ਵਾਪਸ -1;
    }

    // Use even threads only to avoid resource contention
    // Each minion has 2 threads sharing instruction/data cache, NOC to RAM, and FPU
    // Odd threads return immediately to avoid fighting for shared resources
    ਜੇ (thread_id & 1) {
        ਵਾਪਸ 0;  // Odd thread - skip work
    }

    // Adjust thread count and ID for even-only threading
    ਪੂਰਨ effective_thread_id   = thread_id / 2;
    ਪੂਰਨ effective_ਅੰਕ_ਧਾਗੇ = (ਅੰਕ_ਧਾਗੇ + 1) / 2;  // Ceiling division

    // Validate params
    ਜੇ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ਵਾਪਸ -1;
    }

    // Extract tensor references
    ਬਣਤਰ ggml_tensor * src0 = &params->src0;  // Expert weight matrices [K, M, n_expert]
    ਬਣਤਰ ggml_tensor * src1 = &params->src1;  // Activations [K, n_expert_used, batch]
    ਬਣਤਰ ggml_tensor * src2 = &params->src2;  // Expert indices [n_expert_used, batch] (I32)
    ਬਣਤਰ ggml_tensor * dst  = &params->dst;   // Output [M, n_expert_used, batch, 1]

    // Validate tensor types
    ਜੇ (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32) {
        ਵਾਪਸ -1;
    }

    // Get data pointers
    ਅਚਲ ਖਾਲੀ *    src0_ਅੰਕੜੇ = src0->ਅੰਕੜੇ;                    // Expert matrices (Q8_0/F16/F32)
    ਅਚਲ ਦਸ਼ਮਲਵ *   src1_ਅੰਕੜੇ = (ਅਚਲ ਦਸ਼ਮਲਵ *) src1->ਅੰਕੜੇ;    // Activations (F32)
    ਅਚਲ int32_t * src2_ਅੰਕੜੇ = (ਅਚਲ int32_t *) src2->ਅੰਕੜੇ;  // Expert IDs (I32)
    ਦਸ਼ਮਲਵ *         dst_ਅੰਕੜੇ  = (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ;           // Output (F32)

    ਜੇ (!src0_ਅੰਕੜੇ || !src1_ਅੰਕੜੇ || !src2_ਅੰਕੜੇ || !dst_ਅੰਕੜੇ) {
        ਵਾਪਸ -1;
    }

    // Determine block size based on src0 type
    ਪੂਰਨ ਖੰਡ_ਮਾਪ_3;
    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_Q8_0:
            ਖੰਡ_ਮਾਪ_3 = QK8_0;
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q4_0:
            ਖੰਡ_ਮਾਪ_3 = QK4_0;
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F16:
            ਖੰਡ_ਮਾਪ_3 = QK_F16;
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F32:
            ਖੰਡ_ਮਾਪ_3 = QK_F32;
            ਰੋਕੋ;
        ਮੂਲ:
            ਵਾਪਸ -1;
    }

    // Get dimensions
    // src0: [K, M, n_expert] - expert weight matrices
    // src1: [K, n_expert_used, batch] - activations
    // src2: [n_expert_used, batch] - expert indices
    // dst:  [M, n_expert_used, batch, 1] - output
    ਅਚਲ int64_t K             = src0->ne[0];  // Hidden dimension
    ਅਚਲ int64_t M             = src0->ne[1];  // Output features
    ਅਚਲ int64_t n_expert      = src0->ne[2];  // Number of experts
    ਅਚਲ int64_t n_expert_ਵਰਤਿਆ = src2->ne[0];  // Experts used per token
    ਅਚਲ int64_t batch         = src2->ne[1];  // Batch size

    // Strides (in bytes)
    ਅਚਲ size_t nb01 = src0->nb[1];  // src0 row stride
    ਅਚਲ size_t nb02 = src0->nb[2];  // src0 expert stride
    ਅਚਲ size_t nb11 = src1->nb[1];  // src1 column stride
    ਅਚਲ size_t nb12 = src1->nb[2];  // src1 batch stride
    ਅਚਲ size_t nb20 = src2->nb[0];  // src2 element stride
    ਅਚਲ size_t nb21 = src2->nb[1];  // src2 batch stride
    ਅਚਲ size_t nb1  = dst->nb[1];   // dst column stride
    ਅਚਲ size_t nb2  = dst->nb[2];   // dst batch stride

    // Verify K dimension alignment for quantization
    // Q8_0 requires strict alignment (quantized data must be block-aligned)
    // F32 and F16 can handle partial blocks with scalar remainders
    ਜੇ ((src0->type == GGML_TYPE_Q8_0 || src0->type == GGML_TYPE_Q4_0) && K % ਖੰਡ_ਮਾਪ_3 != 0) {
        ਵਾਪਸ -1;  // Q8_0 requires K to be multiple of block_size
    }

    // Verify first dimension is contiguous
    size_t expected_element_ਮਾਪ_src0;
    ਜੇ (src0->type == GGML_TYPE_Q8_0) {
        expected_element_ਮਾਪ_src0 = ਆਕਾਰ(ਖੰਡ_q8_0);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_Q4_0) {
        expected_element_ਮਾਪ_src0 = ਆਕਾਰ(ਖੰਡ_q4_0);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_F16) {
        expected_element_ਮਾਪ_src0 = ਆਕਾਰ(uint16_t);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_F32) {
        expected_element_ਮਾਪ_src0 = ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    } ਨਹੀਂਤਾਂ {
        ਵਾਪਸ -1;
    }

    ਜੇ (src0->nb[0] != expected_element_ਮਾਪ_src0 || src1->nb[0] != ਆਕਾਰ(ਦਸ਼ਮਲਵ) || src2->nb[0] != ਆਕਾਰ(int32_t) ||
        dst->nb[0] != ਆਕਾਰ(ਦਸ਼ਮਲਵ)) {
        ਵਾਪਸ -1;
    }

    ਅਚਲ int64_t K_ਖੰਡਾਂ = K / ਖੰਡ_ਮਾਪ_3;

    // Threading: distribute output elements across threads
    // Total output elements = M * n_expert_used * batch
    ਅਚਲ uint64_t ਕੁੱਲ_elements = M * n_expert_ਵਰਤਿਆ * batch;

    ਅਚਲ uint64_t ਪ੍ਰਤੀ_thread     = 16;
    ਅਚਲ uint64_t ਧਾਗੇ_stride = ਪ੍ਰਤੀ_thread * effective_ਅੰਕ_ਧਾਗੇ;

    ਜੇ (effective_thread_id * ਪ੍ਰਤੀ_thread >= ਕੁੱਲ_elements) {
        ਵਾਪਸ 0;
    }

    // Process elements assigned to this thread
    ਲਈ (uint64_t ਅਧਾਰ_idx = effective_thread_id * ਪ੍ਰਤੀ_thread; ਅਧਾਰ_idx < ਕੁੱਲ_elements; ਅਧਾਰ_idx += ਧਾਗੇ_stride) {
        ਲਈ (uint64_t j = 0; j < ਪ੍ਰਤੀ_thread; j++) {
            ਅਚਲ uint64_t idx = ਅਧਾਰ_idx + j;

            ਜੇ (idx >= ਕੁੱਲ_elements) {
                ਰੋਕੋ;
            }

            // Decode linear index to (m, n_idx, batch_idx)
            // Layout: m + M * (n_idx + n_expert_used * batch_idx)
            ਅਚਲ int64_t batch_idx = idx / (M * n_expert_ਵਰਤਿਆ);
            ਅਚਲ int64_t rem       = idx % (M * n_expert_ਵਰਤਿਆ);
            ਅਚਲ int64_t n_idx     = rem / M;
            ਅਚਲ int64_t m         = rem % M;

            // Get expert ID from src2[n_idx, batch_idx]
            ਅਚਲ int32_t expert_id = *(ਅਚਲ int32_t *) ((ਅਚਲ ਅੱਖਰ *) src2_ਅੰਕੜੇ + n_idx * nb20 + batch_idx * nb21);

            // Validate expert ID
            ਜੇ (expert_id < 0 || expert_id >= n_expert) {
                // Invalid expert ID - write zero and continue
                ਅਸਥਿਰ ਦਸ਼ਮਲਵ * dst_element =
                    (ਅਸਥਿਰ ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst_ਅੰਕੜੇ + m * dst->nb[0] + n_idx * nb1 + batch_idx * nb2);
                atomic_store_f32(dst_element, 0.0f);
                ਜਾਰੀ;
            }

            // Compute dot product: expert_matrix[m, :] x activations[:, col_idx, batch_idx]
            // Use modulo to handle broadcasting: when src1 has fewer columns than expert slots,
            // multiple slots share the same activation column (col_idx = n_idx % src1->ne[1])
            ਅਚਲ int64_t ਖਾਨਾ_idx = n_idx % src1->ne[1];
            ਦਸ਼ਮਲਵ         ਜੋੜ     = 0.0f;

            // Type switch hoisted outside block loop: one branch per element, not per block
            ਅਚਲ ਅੱਖਰ * expert_ਪੰਕਤੀ_ਅਧਾਰ = (ਅਚਲ ਅੱਖਰ *) src0_ਅੰਕੜੇ + m * nb01 + expert_id * nb02;

            ਚੋਣ (src0->type) {
                ਹਾਲਤ GGML_TYPE_Q8_0:
                    {
                        ਅਚਲ ਖੰਡ_q8_0 * q8_ਪੰਕਤੀ = (ਅਚਲ ਖੰਡ_q8_0 *) expert_ਪੰਕਤੀ_ਅਧਾਰ;
                        ਅਚਲ ਦਸ਼ਮਲਵ *      b_ਖਾਨਾ_ਅਧਾਰ =
                            (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) src1_ਅੰਕੜੇ + ਖਾਨਾ_idx * nb11 + batch_idx * nb12);
                        ਜੋੜ += compute_ਪੰਕਤੀ_dot_q8_0(q8_ਪੰਕਤੀ, b_ਖਾਨਾ_ਅਧਾਰ, K_ਖੰਡਾਂ);
                        ਰੋਕੋ;
                    }
                ਹਾਲਤ GGML_TYPE_Q4_0:
                    {
                        ਅਚਲ ਖੰਡ_q4_0 * q4_ਪੰਕਤੀ = (ਅਚਲ ਖੰਡ_q4_0 *) expert_ਪੰਕਤੀ_ਅਧਾਰ;
                        ਅਚਲ ਦਸ਼ਮਲਵ *      b_ਖਾਨਾ_ਅਧਾਰ =
                            (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) src1_ਅੰਕੜੇ + ਖਾਨਾ_idx * nb11 + batch_idx * nb12);
                        ਜੋੜ += compute_ਪੰਕਤੀ_dot_q4_0(q4_ਪੰਕਤੀ, b_ਖਾਨਾ_ਅਧਾਰ, K_ਖੰਡਾਂ);
                        ਰੋਕੋ;
                    }
                ਹਾਲਤ GGML_TYPE_F16:
                    {
                        ਅਚਲ uint16_t * f16_ਪੰਕਤੀ     = (ਅਚਲ uint16_t *) expert_ਪੰਕਤੀ_ਅਧਾਰ;
                        ਅਚਲ int64_t    K_remainder = K % ਖੰਡ_ਮਾਪ_3;
                        ਲਈ (int64_t kb = 0; kb < K_ਖੰਡਾਂ; kb++) {
                            ਅਚਲ ਦਸ਼ਮਲਵ * b_ਖਾਨਾ_ptr =
                                (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) src1_ਅੰਕੜੇ + (kb * ਖੰਡ_ਮਾਪ_3) * ਆਕਾਰ(ਦਸ਼ਮਲਵ) +
                                                 ਖਾਨਾ_idx * nb11 + batch_idx * nb12);
                            ਜੋੜ += compute_ਖੰਡ_dot_ਗੁਣਨਫਲ_f16_naive(&f16_ਪੰਕਤੀ[kb * ਖੰਡ_ਮਾਪ_3], b_ਖਾਨਾ_ptr);
                        }
                        ਜੇ (K_remainder > 0) {
                            ਅਚਲ int64_t ਵਿੱਥ = K_ਖੰਡਾਂ * ਖੰਡ_ਮਾਪ_3;
                            ਅਚਲ ਦਸ਼ਮਲਵ * b_ਖਾਨਾ_ptr =
                                (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) src1_ਅੰਕੜੇ + ਵਿੱਥ * ਆਕਾਰ(ਦਸ਼ਮਲਵ) + ਖਾਨਾ_idx * nb11 +
                                                 batch_idx * nb12);
                            ਜੋੜ += compute_ਖੰਡ_dot_ਗੁਣਨਫਲ_f16_partial(&f16_ਪੰਕਤੀ[ਵਿੱਥ], b_ਖਾਨਾ_ptr, K_remainder);
                        }
                        ਰੋਕੋ;
                    }
                ਹਾਲਤ GGML_TYPE_F32:
                    {
                        ਅਚਲ ਦਸ਼ਮਲਵ * f32_ਪੰਕਤੀ     = (ਅਚਲ ਦਸ਼ਮਲਵ *) expert_ਪੰਕਤੀ_ਅਧਾਰ;
                        ਅਚਲ int64_t K_remainder = K % ਖੰਡ_ਮਾਪ_3;
                        ਲਈ (int64_t kb = 0; kb < K_ਖੰਡਾਂ; kb++) {
                            ਅਚਲ ਦਸ਼ਮਲਵ * b_ਖਾਨਾ_ptr =
                                (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) src1_ਅੰਕੜੇ + (kb * ਖੰਡ_ਮਾਪ_3) * ਆਕਾਰ(ਦਸ਼ਮਲਵ) +
                                                 ਖਾਨਾ_idx * nb11 + batch_idx * nb12);
                            ਜੋੜ += compute_ਖੰਡ_dot_ਗੁਣਨਫਲ_f32(&f32_ਪੰਕਤੀ[kb * ਖੰਡ_ਮਾਪ_3], b_ਖਾਨਾ_ptr);
                        }
                        ਜੇ (K_remainder > 0) {
                            ਅਚਲ int64_t ਵਿੱਥ = K_ਖੰਡਾਂ * ਖੰਡ_ਮਾਪ_3;
                            ਅਚਲ ਦਸ਼ਮਲਵ * b_ਖਾਨਾ_ptr =
                                (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) src1_ਅੰਕੜੇ + ਵਿੱਥ * ਆਕਾਰ(ਦਸ਼ਮਲਵ) + ਖਾਨਾ_idx * nb11 +
                                                 batch_idx * nb12);
                            ਜੋੜ += compute_ਖੰਡ_dot_ਗੁਣਨਫਲ_f32_partial(&f32_ਪੰਕਤੀ[ਵਿੱਥ], b_ਖਾਨਾ_ptr, K_remainder);
                        }
                        ਰੋਕੋ;
                    }
                ਮੂਲ:
                    ਵਾਪਸ -1;
            }

            // Store result using atomic store to avoid cache coherency issues
            // when multiple threads write to the same cache line (64 bytes = 16 floats)
            ਅਸਥਿਰ ਦਸ਼ਮਲਵ * dst_element =
                (ਅਸਥਿਰ ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst_ਅੰਕੜੇ + m * dst->nb[0] + n_idx * nb1 + batch_idx * nb2);
            atomic_store_f32(dst_element, ਜੋੜ);
        }
    }

    ਵਾਪਸ 0;
}
