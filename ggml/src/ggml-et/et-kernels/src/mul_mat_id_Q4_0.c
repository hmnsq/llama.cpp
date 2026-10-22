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

#અન્તર્ભાવ "block_ops.h"
#અન્તર્ભાવ "ggml_tensor.h"
#અન્તર્ભાવ "math_fp.h"
#અન્તર્ભાવ "platform.h"
#અન્તર્ભાવ "quants.h"

#અન્તર્ભાવ <stdint.h>

પૂર્ણાંક entry_point(સંરચના ggml_et_mul_mat_id_params * params, શૂન્ય * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;
    જો (!kernel_env || !params) {
        પાછા -1;
    }

    પૂર્ણાંક thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    પૂર્ણાંક અંક_threads = get_અંક_threads(kernel_env->shire_mask);
    જો (thread_id < 0) {
        પાછા 0;
    }

    સંરચના ggml_tensor * src0 = &params->src0;
    સંરચના ggml_tensor * src1 = &params->src1;
    સંરચના ggml_tensor * src2 = &params->src2;
    સંરચના ggml_tensor * dst  = &params->dst;

    જો (src0->type != GGML_TYPE_Q4_0 || src1->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32 ||
        dst->type != GGML_TYPE_F32) {
        પાછા -1;
    }

    અચલ શૂન્ય *    src0_data = src0->data;
    અચલ દશાંશ *   src1_data = (અચલ દશાંશ *) src1->data;
    અચલ int32_t * src2_data = (અચલ int32_t *) src2->data;
    દશાંશ *         dst_data  = (દશાંશ *) dst->data;
    જો (!src0_data || !src1_data || !src2_data || !dst_data) {
        પાછા -1;
    }

    અચલ int64_t K             = src0->ne[0];
    અચલ int64_t M             = src0->ne[1];
    અચલ int64_t n_expert      = src0->ne[2];
    અચલ int64_t n_expert_used = src2->ne[0];
    અચલ int64_t batch         = src2->ne[1];
    અચલ int64_t ne11          = src1->ne[1];

    જો (K % QK4_0 != 0) {
        પાછા -1;
    }

    અચલ size_t nb01 = src0->nb[1];  // Q4_0 row stride
    અચલ size_t nb02 = src0->nb[2];  // expert stride
    અચલ size_t nb11 = src1->nb[1];  // activation column stride
    અચલ size_t nb12 = src1->nb[2];  // activation batch stride
    અચલ size_t nb20 = src2->nb[0];
    અચલ size_t nb21 = src2->nb[1];
    અચલ size_t nbd0 = dst->nb[0];
    અચલ size_t nbd1 = dst->nb[1];
    અચલ size_t nbd2 = dst->nb[2];

    જો (src0->nb[0] != કદ(block_q4_0) || src1->nb[0] != કદ(દશાંશ) || src2->nb[0] != કદ(int32_t) ||
        nbd0 != કદ(દશાંશ)) {
        પાછા -1;
    }

    અચલ int64_t K_blocks = K / QK4_0;
    અચલ પૂર્ણાંક     use_x2   = ((nb01 & 31) == 0);

    અચલ uint64_t total_outputs = (uint64_t) M * (uint64_t) n_expert_used * (uint64_t) batch;
    જો (total_outputs == 0) {
        પાછા 0;
    }

    // Even partition: hart h owns outputs [h*chunk, (h+1)*chunk).
    અચલ uint64_t chunk    = (total_outputs + (uint64_t) અંક_threads - 1) / (uint64_t) અંક_threads;
    અચલ uint64_t my_start = (uint64_t) thread_id * chunk;
    જો (my_start >= total_outputs) {
        પાછા 0;
    }
    uint64_t my_end = my_start + chunk;
    જો (my_end > total_outputs) {
        my_end = total_outputs;
    }

    // Save mask register once; full lanes for vector dot.
    q4_dot_state q4_state;
    q4_dot_begin(&q4_state);

    અચલ uint64_t per_batch = (uint64_t) M * (uint64_t) n_expert_used;

    uint64_t idx = my_start;
    જ્યારે (idx < my_end) {
        // Decode (m, slot, batch) from the m-major linear index.
        અચલ int64_t  batch_idx = (int64_t) (idx / per_batch);
        અચલ uint64_t rem       = idx - (uint64_t) batch_idx * per_batch;
        અચલ int64_t  slot_idx  = (int64_t) (rem / (uint64_t) M);
        અચલ int64_t  m0        = (int64_t) (rem - (uint64_t) slot_idx * (uint64_t) M);

        // How many outputs left in this (slot, batch) run AND in my range.
        અચલ uint64_t run_end_global =
            (uint64_t) batch_idx * per_batch + (uint64_t) slot_idx * (uint64_t) M + (uint64_t) M;
        અચલ uint64_t end_in_my = (run_end_global < my_end) ? run_end_global : my_end;
        int64_t        run_len   = (int64_t) (end_in_my - idx);

        // Resolve expert + B column + dst slot for this run.
        અચલ int32_t expert_id =
            *(અચલ int32_t *) ((અચલ અક્ષર *) src2_data + slot_idx * (int64_t) nb20 + batch_idx * (int64_t) nb21);

        અક્ષર * dst_slot = (અક્ષર *) dst_data + slot_idx * (int64_t) nbd1 + batch_idx * (int64_t) nbd2;

        જો (expert_id < 0 || expert_id >= n_expert) {
            // Invalid expert id — zero out this run's outputs.
            int64_t m = m0;
            માટે (int64_t i = 0; i < run_len; i++, m++) {
                atomic_store_f32((અસ્થિર દશાંશ *) (dst_slot + m * (int64_t) nbd0), 0.0f);
            }
            idx += (uint64_t) run_len;
            ચાલુ;
        }

        અચલ int64_t col_idx = slot_idx % ne11;
        અચલ દશાંશ * b_col_base =
            (અચલ દશાંશ *) ((અચલ અક્ષર *) src1_data + col_idx * (int64_t) nb11 + batch_idx * (int64_t) nb12);
        અચલ અક્ષર * expert_base = (અચલ અક્ષર *) src0_data + expert_id * (int64_t) nb02;

        int64_t m    = m0;
        int64_t ડાબું = run_len;

        // Paired-row dots: halves B bandwidth for runs >= 2.
        જો (use_x2) {
            જ્યારે (ડાબું >= 2) {
                અચલ block_q4_0 * row0 = (અચલ block_q4_0 *) (expert_base + m * (int64_t) nb01);
                અચલ block_q4_0 * row1 = (અચલ block_q4_0 *) (expert_base + (m + 1) * (int64_t) nb01);
                દશાંશ              s0, s1;
                q4_dot_compute_x2_aligned(row0, row1, b_col_base, K_blocks, &s0, &s1);
                atomic_store_f32((અસ્થિર દશાંશ *) (dst_slot + m * (int64_t) nbd0), s0);
                atomic_store_f32((અસ્થિર દશાંશ *) (dst_slot + (m + 1) * (int64_t) nbd0), s1);
                m += 2;
                ડાબું -= 2;
            }
        }

        // Tail / non-aligned fallback: single-row dots.
        જ્યારે (ડાબું > 0) {
            અચલ block_q4_0 * row = (અચલ block_q4_0 *) (expert_base + m * (int64_t) nb01);
            દશાંશ              s   = q4_dot_compute(row, b_col_base, K_blocks);
            atomic_store_f32((અસ્થિર દશાંશ *) (dst_slot + m * (int64_t) nbd0), s);
            m++;
            ડાબું--;
        }

        idx += (uint64_t) run_len;
    }

    q4_dot_end(&q4_state);
    પાછા 0;
}
