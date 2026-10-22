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

#অন্তর্ভাব "block_ops.h"
#অন্তর্ভাব "ggml_tensor.h"
#অন্তর্ভাব "math_fp.h"
#অন্তর্ভাব "platform.h"
#অন্তর্ভাব "quants.h"

#অন্তর্ভাব <stdint.h>

পূর্ণাঙ্ক entry_point(সংরচনা ggml_et_mul_mat_id_params * params, শূন্য * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;
    যদি (!kernel_env || !params) {
        প্রত্যাগম -1;
    }

    পূর্ণাঙ্ক thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    পূর্ণাঙ্ক অঙ্ক_threads = get_অঙ্ক_threads(kernel_env->shire_mask);
    যদি (thread_id < 0) {
        প্রত্যাগম 0;
    }

    সংরচনা ggml_tensor * src0 = &params->src0;
    সংরচনা ggml_tensor * src1 = &params->src1;
    সংরচনা ggml_tensor * src2 = &params->src2;
    সংরচনা ggml_tensor * dst  = &params->dst;

    যদি (src0->type != GGML_TYPE_Q4_0 || src1->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32 ||
        dst->type != GGML_TYPE_F32) {
        প্রত্যাগম -1;
    }

    ধ্রুব শূন্য *    src0_data = src0->data;
    ধ্রুব ভিন্ন *   src1_data = (ধ্রুব ভিন্ন *) src1->data;
    ধ্রুব int32_t * src2_data = (ধ্রুব int32_t *) src2->data;
    ভিন্ন *         dst_data  = (ভিন্ন *) dst->data;
    যদি (!src0_data || !src1_data || !src2_data || !dst_data) {
        প্রত্যাগম -1;
    }

    ধ্রুব int64_t K             = src0->ne[0];
    ধ্রুব int64_t M             = src0->ne[1];
    ধ্রুব int64_t n_expert      = src0->ne[2];
    ধ্রুব int64_t n_expert_used = src2->ne[0];
    ধ্রুব int64_t batch         = src2->ne[1];
    ধ্রুব int64_t ne11          = src1->ne[1];

    যদি (K % QK4_0 != 0) {
        প্রত্যাগম -1;
    }

    ধ্রুব size_t nb01 = src0->nb[1];  // Q4_0 row stride
    ধ্রুব size_t nb02 = src0->nb[2];  // expert stride
    ধ্রুব size_t nb11 = src1->nb[1];  // activation column stride
    ধ্রুব size_t nb12 = src1->nb[2];  // activation batch stride
    ধ্রুব size_t nb20 = src2->nb[0];
    ধ্রুব size_t nb21 = src2->nb[1];
    ধ্রুব size_t nbd0 = dst->nb[0];
    ধ্রুব size_t nbd1 = dst->nb[1];
    ধ্রুব size_t nbd2 = dst->nb[2];

    যদি (src0->nb[0] != পরিমাণ(block_q4_0) || src1->nb[0] != পরিমাণ(ভিন্ন) || src2->nb[0] != পরিমাণ(int32_t) ||
        nbd0 != পরিমাণ(ভিন্ন)) {
        প্রত্যাগম -1;
    }

    ধ্রুব int64_t K_blocks = K / QK4_0;
    ধ্রুব পূর্ণাঙ্ক     use_x2   = ((nb01 & 31) == 0);

    ধ্রুব uint64_t total_outputs = (uint64_t) M * (uint64_t) n_expert_used * (uint64_t) batch;
    যদি (total_outputs == 0) {
        প্রত্যাগম 0;
    }

    // Even partition: hart h owns outputs [h*chunk, (h+1)*chunk).
    ধ্রুব uint64_t chunk    = (total_outputs + (uint64_t) অঙ্ক_threads - 1) / (uint64_t) অঙ্ক_threads;
    ধ্রুব uint64_t my_start = (uint64_t) thread_id * chunk;
    যদি (my_start >= total_outputs) {
        প্রত্যাগম 0;
    }
    uint64_t my_end = my_start + chunk;
    যদি (my_end > total_outputs) {
        my_end = total_outputs;
    }

    // Save mask register once; full lanes for vector dot.
    q4_dot_state q4_state;
    q4_dot_begin(&q4_state);

    ধ্রুব uint64_t per_batch = (uint64_t) M * (uint64_t) n_expert_used;

    uint64_t idx = my_start;
    যাবত্ (idx < my_end) {
        // Decode (m, slot, batch) from the m-major linear index.
        ধ্রুব int64_t  batch_idx = (int64_t) (idx / per_batch);
        ধ্রুব uint64_t rem       = idx - (uint64_t) batch_idx * per_batch;
        ধ্রুব int64_t  slot_idx  = (int64_t) (rem / (uint64_t) M);
        ধ্রুব int64_t  m0        = (int64_t) (rem - (uint64_t) slot_idx * (uint64_t) M);

        // How many outputs left in this (slot, batch) run AND in my range.
        ধ্রুব uint64_t run_end_global =
            (uint64_t) batch_idx * per_batch + (uint64_t) slot_idx * (uint64_t) M + (uint64_t) M;
        ধ্রুব uint64_t end_in_my = (run_end_global < my_end) ? run_end_global : my_end;
        int64_t        run_len   = (int64_t) (end_in_my - idx);

        // Resolve expert + B column + dst slot for this run.
        ধ্রুব int32_t expert_id =
            *(ধ্রুব int32_t *) ((ধ্রুব বর্ণ *) src2_data + slot_idx * (int64_t) nb20 + batch_idx * (int64_t) nb21);

        বর্ণ * dst_slot = (বর্ণ *) dst_data + slot_idx * (int64_t) nbd1 + batch_idx * (int64_t) nbd2;

        যদি (expert_id < 0 || expert_id >= n_expert) {
            // Invalid expert id — zero out this run's outputs.
            int64_t m = m0;
            প্রতি (int64_t i = 0; i < run_len; i++, m++) {
                atomic_store_f32((চঞ্চল ভিন্ন *) (dst_slot + m * (int64_t) nbd0), 0.0f);
            }
            idx += (uint64_t) run_len;
            অনুবর্ত;
        }

        ধ্রুব int64_t col_idx = slot_idx % ne11;
        ধ্রুব ভিন্ন * b_col_base =
            (ধ্রুব ভিন্ন *) ((ধ্রুব বর্ণ *) src1_data + col_idx * (int64_t) nb11 + batch_idx * (int64_t) nb12);
        ধ্রুব বর্ণ * expert_base = (ধ্রুব বর্ণ *) src0_data + expert_id * (int64_t) nb02;

        int64_t m    = m0;
        int64_t বাম = run_len;

        // Paired-row dots: halves B bandwidth for runs >= 2.
        যদি (use_x2) {
            যাবত্ (বাম >= 2) {
                ধ্রুব block_q4_0 * row0 = (ধ্রুব block_q4_0 *) (expert_base + m * (int64_t) nb01);
                ধ্রুব block_q4_0 * row1 = (ধ্রুব block_q4_0 *) (expert_base + (m + 1) * (int64_t) nb01);
                ভিন্ন              s0, s1;
                q4_dot_compute_x2_aligned(row0, row1, b_col_base, K_blocks, &s0, &s1);
                atomic_store_f32((চঞ্চল ভিন্ন *) (dst_slot + m * (int64_t) nbd0), s0);
                atomic_store_f32((চঞ্চল ভিন্ন *) (dst_slot + (m + 1) * (int64_t) nbd0), s1);
                m += 2;
                বাম -= 2;
            }
        }

        // Tail / non-aligned fallback: single-row dots.
        যাবত্ (বাম > 0) {
            ধ্রুব block_q4_0 * row = (ধ্রুব block_q4_0 *) (expert_base + m * (int64_t) nb01);
            ভিন্ন              s   = q4_dot_compute(row, b_col_base, K_blocks);
            atomic_store_f32((চঞ্চল ভিন্ন *) (dst_slot + m * (int64_t) nbd0), s);
            m++;
            বাম--;
        }

        idx += (uint64_t) run_len;
    }

    q4_dot_end(&q4_state);
    প্রত্যাগম 0;
}
