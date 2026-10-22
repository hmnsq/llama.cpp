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

#अन्तर्भाव "block_ops.h"
#अन्तर्भाव "ggml_tensor.h"
#अन्तर्भाव "math_fp.h"
#अन्तर्भाव "platform.h"
#अन्तर्भाव "quants.h"

#अन्तर्भाव <stdint.h>

पूर्णाङ्क entry_point(संरचना ggml_et_mul_mat_id_params * params, शून्य * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;
    जइ (!kernel_env || !params) {
        प्रत्यागम -1;
    }

    पूर्णाङ्क thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    पूर्णाङ्क अङ्कः_threads = get_अङ्कः_threads(kernel_env->shire_mask);
    जइ (thread_id < 0) {
        प्रत्यागम 0;
    }

    संरचना ggml_tensor * src0 = &params->src0;
    संरचना ggml_tensor * src1 = &params->src1;
    संरचना ggml_tensor * src2 = &params->src2;
    संरचना ggml_tensor * dst  = &params->dst;

    जइ (src0->type != GGML_TYPE_Q4_0 || src1->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32 ||
        dst->type != GGML_TYPE_F32) {
        प्रत्यागम -1;
    }

    ध्रुव शून्य *    src0_data = src0->data;
    ध्रुव भिन्न *   src1_data = (ध्रुव भिन्न *) src1->data;
    ध्रुव int32_t * src2_data = (ध्रुव int32_t *) src2->data;
    भिन्न *         dst_data  = (भिन्न *) dst->data;
    जइ (!src0_data || !src1_data || !src2_data || !dst_data) {
        प्रत्यागम -1;
    }

    ध्रुव int64_t K             = src0->ne[0];
    ध्रुव int64_t M             = src0->ne[1];
    ध्रुव int64_t n_expert      = src0->ne[2];
    ध्रुव int64_t n_expert_used = src2->ne[0];
    ध्रुव int64_t batch         = src2->ne[1];
    ध्रुव int64_t ne11          = src1->ne[1];

    जइ (K % QK4_0 != 0) {
        प्रत्यागम -1;
    }

    ध्रुव size_t nb01 = src0->nb[1];  // Q4_0 row stride
    ध्रुव size_t nb02 = src0->nb[2];  // expert stride
    ध्रुव size_t nb11 = src1->nb[1];  // activation column stride
    ध्रुव size_t nb12 = src1->nb[2];  // activation batch stride
    ध्रुव size_t nb20 = src2->nb[0];
    ध्रुव size_t nb21 = src2->nb[1];
    ध्रुव size_t nbd0 = dst->nb[0];
    ध्रुव size_t nbd1 = dst->nb[1];
    ध्रुव size_t nbd2 = dst->nb[2];

    जइ (src0->nb[0] != परिमाण(block_q4_0) || src1->nb[0] != परिमाण(भिन्न) || src2->nb[0] != परिमाण(int32_t) ||
        nbd0 != परिमाण(भिन्न)) {
        प्रत्यागम -1;
    }

    ध्रुव int64_t K_blocks = K / QK4_0;
    ध्रुव पूर्णाङ्क     use_x2   = ((nb01 & 31) == 0);

    ध्रुव uint64_t total_outputs = (uint64_t) M * (uint64_t) n_expert_used * (uint64_t) batch;
    जइ (total_outputs == 0) {
        प्रत्यागम 0;
    }

    // Even partition: hart h owns outputs [h*chunk, (h+1)*chunk).
    ध्रुव uint64_t chunk    = (total_outputs + (uint64_t) अङ्कः_threads - 1) / (uint64_t) अङ्कः_threads;
    ध्रुव uint64_t my_start = (uint64_t) thread_id * chunk;
    जइ (my_start >= total_outputs) {
        प्रत्यागम 0;
    }
    uint64_t my_end = my_start + chunk;
    जइ (my_end > total_outputs) {
        my_end = total_outputs;
    }

    // Save mask register once; full lanes for vector dot.
    q4_dot_state q4_state;
    q4_dot_begin(&q4_state);

    ध्रुव uint64_t per_batch = (uint64_t) M * (uint64_t) n_expert_used;

    uint64_t idx = my_start;
    जाव (idx < my_end) {
        // Decode (m, slot, batch) from the m-major linear index.
        ध्रुव int64_t  batch_idx = (int64_t) (idx / per_batch);
        ध्रुव uint64_t rem       = idx - (uint64_t) batch_idx * per_batch;
        ध्रुव int64_t  slot_idx  = (int64_t) (rem / (uint64_t) M);
        ध्रुव int64_t  m0        = (int64_t) (rem - (uint64_t) slot_idx * (uint64_t) M);

        // How many outputs left in this (slot, batch) run AND in my range.
        ध्रुव uint64_t run_end_global =
            (uint64_t) batch_idx * per_batch + (uint64_t) slot_idx * (uint64_t) M + (uint64_t) M;
        ध्रुव uint64_t end_in_my = (run_end_global < my_end) ? run_end_global : my_end;
        int64_t        run_len   = (int64_t) (end_in_my - idx);

        // Resolve expert + B column + dst slot for this run.
        ध्रुव int32_t expert_id =
            *(ध्रुव int32_t *) ((ध्रुव वर्ण *) src2_data + slot_idx * (int64_t) nb20 + batch_idx * (int64_t) nb21);

        वर्ण * dst_slot = (वर्ण *) dst_data + slot_idx * (int64_t) nbd1 + batch_idx * (int64_t) nbd2;

        जइ (expert_id < 0 || expert_id >= n_expert) {
            // Invalid expert id — zero out this run's outputs.
            int64_t m = m0;
            प्रति (int64_t i = 0; i < run_len; i++, m++) {
                atomic_store_f32((चञ्चल भिन्न *) (dst_slot + m * (int64_t) nbd0), 0.0f);
            }
            idx += (uint64_t) run_len;
            अनुवर्त;
        }

        ध्रुव int64_t col_idx = slot_idx % ne11;
        ध्रुव भिन्न * b_col_base =
            (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src1_data + col_idx * (int64_t) nb11 + batch_idx * (int64_t) nb12);
        ध्रुव वर्ण * expert_base = (ध्रुव वर्ण *) src0_data + expert_id * (int64_t) nb02;

        int64_t m    = m0;
        int64_t वामः = run_len;

        // Paired-row dots: halves B bandwidth for runs >= 2.
        जइ (use_x2) {
            जाव (वामः >= 2) {
                ध्रुव block_q4_0 * row0 = (ध्रुव block_q4_0 *) (expert_base + m * (int64_t) nb01);
                ध्रुव block_q4_0 * row1 = (ध्रुव block_q4_0 *) (expert_base + (m + 1) * (int64_t) nb01);
                भिन्न              s0, s1;
                q4_dot_compute_x2_aligned(row0, row1, b_col_base, K_blocks, &s0, &s1);
                atomic_store_f32((चञ्चल भिन्न *) (dst_slot + m * (int64_t) nbd0), s0);
                atomic_store_f32((चञ्चल भिन्न *) (dst_slot + (m + 1) * (int64_t) nbd0), s1);
                m += 2;
                वामः -= 2;
            }
        }

        // Tail / non-aligned fallback: single-row dots.
        जाव (वामः > 0) {
            ध्रुव block_q4_0 * row = (ध्रुव block_q4_0 *) (expert_base + m * (int64_t) nb01);
            भिन्न              s   = q4_dot_compute(row, b_col_base, K_blocks);
            atomic_store_f32((चञ्चल भिन्न *) (dst_slot + m * (int64_t) nbd0), s);
            m++;
            वामः--;
        }

        idx += (uint64_t) run_len;
    }

    q4_dot_end(&q4_state);
    प्रत्यागम 0;
}
