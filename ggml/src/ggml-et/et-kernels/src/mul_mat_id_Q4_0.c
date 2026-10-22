//******************************************************************************
// MUL_MAT_ID kernel specialized for Q4_0 weights (Mixture of Experts).
//
// C[m, s, b] = Sum(k=0..K-1) A[k, m, ids[s,b]] * B[k, s % ne11, b]
//   A: Q4_0  [K, M, n_expert]   weights
//   B: F32   [K, n_cols, batch] activations
//   ids: I32 [n_expert_used, batch]
//   C: F32   [M, n_expert_used, batch]
//
// Strategy: All harts active. Flat m-major output partition allows amortized
// expert lookups and 2-row x2 dot products.
//******************************************************************************

#ಅನ್ತರ್ಭಾವ "block_ops.h"
#ಅನ್ತರ್ಭಾವ "ggml_tensor.h"
#ಅನ್ತರ್ಭಾವ "math_fp.h"
#ಅನ್ತರ್ಭಾವ "platform.h"
#ಅನ್ತರ್ಭಾವ "quants.h"

#ಅನ್ತರ್ಭಾವ <stdint.h>

ಪೂರ್ಣಾಂಕ ನಮೂದು_ಬಿಂದು(ರಚನೆ ggml_et_mul_mat_id_params * params, ಶೂನ್ಯ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;
    ಆದರೆ (!kernel_env || !params) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಪೂರ್ಣಾಂಕ thread_id   = get_relative_thread_id(kernel_env->shire_ಮುಖವಾಡ);
    ಪೂರ್ಣಾಂಕ ಅಂಕೆ_ಎಳೆಗಳು = get_ಅಂಕೆ_ಎಳೆಗಳು(kernel_env->shire_ಮುಖವಾಡ);
    ಆದರೆ (thread_id < 0) {
        ಹಿಂತಿರುಗಿ 0;
    }

    ರಚನೆ ggml_tensor * src0 = &params->src0;
    ರಚನೆ ggml_tensor * src1 = &params->src1;
    ರಚನೆ ggml_tensor * src2 = &params->src2;
    ರಚನೆ ggml_tensor * dst  = &params->dst;

    ಆದರೆ (src0->type != GGML_TYPE_Q4_0 || src1->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32 ||
        dst->type != GGML_TYPE_F32) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *    src0_ದತ್ತಾಂಶ = src0->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *   src1_ದತ್ತಾಂಶ = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src1->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ int32_t * src2_ದತ್ತಾಂಶ = (ಸ್ಥಿರಾಂಕ int32_t *) src2->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ *         dst_ದತ್ತಾಂಶ  = (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ;
    ಆದರೆ (!src0_ದತ್ತಾಂಶ || !src1_ದತ್ತಾಂಶ || !src2_ದತ್ತಾಂಶ || !dst_ದತ್ತಾಂಶ) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಸ್ಥಿರಾಂಕ int64_t K             = src0->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t M             = src0->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t n_expert      = src0->ne[2];
    ಸ್ಥಿರಾಂಕ int64_t n_expert_ಬಳಸಿದ = src2->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t batch         = src2->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t ne11          = src1->ne[1];

    ಆದರೆ (K % QK4_0 != 0) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಸ್ಥಿರಾಂಕ size_t nb01 = src0->nb[1];  // Q4_0 row stride
    ಸ್ಥಿರಾಂಕ size_t nb02 = src0->nb[2];  // expert stride
    ಸ್ಥಿರಾಂಕ size_t nb11 = src1->nb[1];  // activation column stride
    ಸ್ಥಿರಾಂಕ size_t nb12 = src1->nb[2];  // activation batch stride
    ಸ್ಥಿರಾಂಕ size_t nb20 = src2->nb[0];
    ಸ್ಥಿರಾಂಕ size_t nb21 = src2->nb[1];
    ಸ್ಥಿರಾಂಕ size_t nbd0 = dst->nb[0];
    ಸ್ಥಿರಾಂಕ size_t nbd1 = dst->nb[1];
    ಸ್ಥಿರಾಂಕ size_t nbd2 = dst->nb[2];

    ಆದರೆ (src0->nb[0] != ಗಾತ್ರ(ಖಂಡ_q4_0) || src1->nb[0] != ಗಾತ್ರ(ದಶಮಾಂಶ) || src2->nb[0] != ಗಾತ್ರ(int32_t) ||
        nbd0 != ಗಾತ್ರ(ದಶಮಾಂಶ)) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಸ್ಥಿರಾಂಕ int64_t K_ಖಂಡಗಳು = K / QK4_0;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ     use_x2   = ((nb01 & 31) == 0);

    ಸ್ಥಿರಾಂಕ uint64_t ಒಟ್ಟು_outputs = (uint64_t) M * (uint64_t) n_expert_ಬಳಸಿದ * (uint64_t) batch;
    ಆದರೆ (ಒಟ್ಟು_outputs == 0) {
        ಹಿಂತಿರುಗಿ 0;
    }

    // Even partition: hart h owns outputs [h*chunk, (h+1)*chunk).
    ಸ್ಥಿರಾಂಕ uint64_t chunk    = (ಒಟ್ಟು_outputs + (uint64_t) ಅಂಕೆ_ಎಳೆಗಳು - 1) / (uint64_t) ಅಂಕೆ_ಎಳೆಗಳು;
    ಸ್ಥಿರಾಂಕ uint64_t my_ಆರಂಭ = (uint64_t) thread_id * chunk;
    ಆದರೆ (my_ಆರಂಭ >= ಒಟ್ಟು_outputs) {
        ಹಿಂತಿರುಗಿ 0;
    }
    uint64_t my_ಕೊನೆ = my_ಆರಂಭ + chunk;
    ಆದರೆ (my_ಕೊನೆ > ಒಟ್ಟು_outputs) {
        my_ಕೊನೆ = ಒಟ್ಟು_outputs;
    }

    // Save mask register once; full lanes for vector dot.
    q4_dot_state q4_state;
    q4_dot_begin(&q4_state);

    ಸ್ಥಿರಾಂಕ uint64_t ಪ್ರತಿ_batch = (uint64_t) M * (uint64_t) n_expert_ಬಳಸಿದ;

    uint64_t idx = my_ಆರಂಭ;
    ಆಗ (idx < my_ಕೊನೆ) {
        // Decode (m, slot, batch) from the m-major linear index.
        ಸ್ಥಿರಾಂಕ int64_t  batch_idx = (int64_t) (idx / ಪ್ರತಿ_batch);
        ಸ್ಥಿರಾಂಕ uint64_t rem       = idx - (uint64_t) batch_idx * ಪ್ರತಿ_batch;
        ಸ್ಥಿರಾಂಕ int64_t  ಸ್ಥಾನ_idx  = (int64_t) (rem / (uint64_t) M);
        ಸ್ಥಿರಾಂಕ int64_t  m0        = (int64_t) (rem - (uint64_t) ಸ್ಥಾನ_idx * (uint64_t) M);

        // How many outputs left in this (slot, batch) run AND in my range.
        ಸ್ಥಿರಾಂಕ uint64_t ಸತತ_ಕೊನೆ_global =
            (uint64_t) batch_idx * ಪ್ರತಿ_batch + (uint64_t) ಸ್ಥಾನ_idx * (uint64_t) M + (uint64_t) M;
        ಸ್ಥಿರಾಂಕ uint64_t ಕೊನೆ_ಒಳ_my = (ಸತತ_ಕೊನೆ_global < my_ಕೊನೆ) ? ಸತತ_ಕೊನೆ_global : my_ಕೊನೆ;
        int64_t        ಸತತ_len   = (int64_t) (ಕೊನೆ_ಒಳ_my - idx);

        // Resolve expert + B column + dst slot for this run.
        ಸ್ಥಿರಾಂಕ int32_t expert_id =
            *(ಸ್ಥಿರಾಂಕ int32_t *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src2_ದತ್ತಾಂಶ + ಸ್ಥಾನ_idx * (int64_t) nb20 + batch_idx * (int64_t) nb21);

        ಅಕ್ಷರ * dst_ಸ್ಥಾನ = (ಅಕ್ಷರ *) dst_ದತ್ತಾಂಶ + ಸ್ಥಾನ_idx * (int64_t) nbd1 + batch_idx * (int64_t) nbd2;

        ಆದರೆ (expert_id < 0 || expert_id >= n_expert) {
            // Invalid expert id — zero out this run's outputs.
            int64_t m = m0;
            ಗಾಗಿ (int64_t i = 0; i < ಸತತ_len; i++, m++) {
                atomic_store_f32((ಅಸ್ಥಿರ ದಶಮಾಂಶ *) (dst_ಸ್ಥಾನ + m * (int64_t) nbd0), 0.0f);
            }
            idx += (uint64_t) ಸತತ_len;
            ಮುಂದುವರಿ;
        }

        ಸ್ಥಿರಾಂಕ int64_t ಲಂಬಸಾಲು_idx = ಸ್ಥಾನ_idx % ne11;
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * b_ಲಂಬಸಾಲು_ತಳ =
            (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src1_ದತ್ತಾಂಶ + ಲಂಬಸಾಲು_idx * (int64_t) nb11 + batch_idx * (int64_t) nb12);
        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * expert_ತಳ = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src0_ದತ್ತಾಂಶ + expert_id * (int64_t) nb02;

        int64_t m    = m0;
        int64_t ಎಡ_2 = ಸತತ_len;

        // Paired-row dots: halves B bandwidth for runs >= 2.
        ಆದರೆ (use_x2) {
            ಆಗ (ಎಡ_2 >= 2) {
                ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0 * row0 = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0 *) (expert_ತಳ + m * (int64_t) nb01);
                ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0 * row1 = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0 *) (expert_ತಳ + (m + 1) * (int64_t) nb01);
                ದಶಮಾಂಶ              s0, s1;
                q4_dot_compute_x2_aligned(row0, row1, b_ಲಂಬಸಾಲು_ತಳ, K_ಖಂಡಗಳು, &s0, &s1);
                atomic_store_f32((ಅಸ್ಥಿರ ದಶಮಾಂಶ *) (dst_ಸ್ಥಾನ + m * (int64_t) nbd0), s0);
                atomic_store_f32((ಅಸ್ಥಿರ ದಶಮಾಂಶ *) (dst_ಸ್ಥಾನ + (m + 1) * (int64_t) nbd0), s1);
                m += 2;
                ಎಡ_2 -= 2;
            }
        }

        // Tail / non-aligned fallback: single-row dots.
        ಆಗ (ಎಡ_2 > 0) {
            ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0 * ಅಡ್ಡಸಾಲು = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0 *) (expert_ತಳ + m * (int64_t) nb01);
            ದಶಮಾಂಶ              s   = q4_dot_compute(ಅಡ್ಡಸಾಲು, b_ಲಂಬಸಾಲು_ತಳ, K_ಖಂಡಗಳು);
            atomic_store_f32((ಅಸ್ಥಿರ ದಶಮಾಂಶ *) (dst_ಸ್ಥಾನ + m * (int64_t) nbd0), s);
            m++;
            ಎಡ_2--;
        }

        idx += (uint64_t) ಸತತ_len;
    }

    q4_dot_ಕೊನೆ(&q4_state);
    ಹಿಂತಿರುಗಿ 0;
}
