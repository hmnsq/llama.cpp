//******************************************************************************
// MUL_MAT_ID kernel specialized for Q8_0 weights (Mixture of Experts).
//
// C[m, s, b] = Sum(k=0..K-1) A[k, m, ids[s,b]] * B[k, s % ne11, b]
//   A: Q8_0  [K, M, n_expert]   weights
//   B: F32   [K, n_cols, batch] activations
//   ids: I32 [n_expert_used, batch]
//   C: F32   [M, n_expert_used, batch]
//
// Strategy mirrors mul_mat_id_Q4_0.c.
//******************************************************************************

#অন্তর্ভাৱ "block_ops.h"
#অন্তর্ভাৱ "ggml_tensor.h"
#অন্তর্ভাৱ "math_fp.h"
#অন্তর্ভাৱ "platform.h"
#অন্তর্ভাৱ "quants.h"

#অন্তর্ভাৱ <stdint.h>

পূর্ণাঙ্ক entry_point(সংরচনা ggml_et_mul_mat_id_params * params, শূন্য * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;
    যদি (!kernel_env || !params) {
        ঘূৰাই -1;
    }

    পূর্ণাঙ্ক thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    পূর্ণাঙ্ক অঙ্ক_threads = get_অঙ্ক_threads(kernel_env->shire_mask);
    যদি (thread_id < 0) {
        ঘূৰাই 0;
    }

    সংরচনা ggml_tensor * src0 = &params->src0;
    সংরচনা ggml_tensor * src1 = &params->src1;
    সংরচনা ggml_tensor * src2 = &params->src2;
    সংরচনা ggml_tensor * dst  = &params->dst;

    যদি (src0->type != GGML_TYPE_Q8_0 || src1->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32 ||
        dst->type != GGML_TYPE_F32) {
        ঘূৰাই -1;
    }

    ধ্রুৱ শূন্য *    src0_data = src0->data;
    ধ্রুৱ ভিন্ন *   src1_data = (ধ্রুৱ ভিন্ন *) src1->data;
    ধ্রুৱ int32_t * src2_data = (ধ্রুৱ int32_t *) src2->data;
    ভিন্ন *         dst_data  = (ভিন্ন *) dst->data;
    যদি (!src0_data || !src1_data || !src2_data || !dst_data) {
        ঘূৰাই -1;
    }

    ধ্রুৱ int64_t K             = src0->ne[0];
    ধ্রুৱ int64_t M             = src0->ne[1];
    ধ্রুৱ int64_t n_expert      = src0->ne[2];
    ধ্রুৱ int64_t n_expert_used = src2->ne[0];
    ধ্রুৱ int64_t batch         = src2->ne[1];
    ধ্রুৱ int64_t ne11          = src1->ne[1];

    যদি (K % QK8_0 != 0) {
        ঘূৰাই -1;
    }

    ধ্রুৱ size_t nb01 = src0->nb[1];
    ধ্রুৱ size_t nb02 = src0->nb[2];
    ধ্রুৱ size_t nb11 = src1->nb[1];
    ধ্রুৱ size_t nb12 = src1->nb[2];
    ধ্রুৱ size_t nb20 = src2->nb[0];
    ধ্রুৱ size_t nb21 = src2->nb[1];
    ধ্রুৱ size_t nbd0 = dst->nb[0];
    ধ্রুৱ size_t nbd1 = dst->nb[1];
    ধ্রুৱ size_t nbd2 = dst->nb[2];

    যদি (src0->nb[0] != পরিমাণ(block_q8_0) || src1->nb[0] != পরিমাণ(ভিন্ন) || src2->nb[0] != পরিমাণ(int32_t) ||
        nbd0 != পরিমাণ(ভিন্ন)) {
        ঘূৰাই -1;
    }

    ধ্রুৱ int64_t K_blocks = K / QK8_0;
    ধ্রুৱ পূর্ণাঙ্ক     use_x2   = ((nb01 & 31) == 0);

    ধ্রুৱ uint64_t total_outputs = (uint64_t) M * (uint64_t) n_expert_used * (uint64_t) batch;
    যদি (total_outputs == 0) {
        ঘূৰাই 0;
    }

    ধ্রুৱ uint64_t chunk    = (total_outputs + (uint64_t) অঙ্ক_threads - 1) / (uint64_t) অঙ্ক_threads;
    ধ্রুৱ uint64_t my_start = (uint64_t) thread_id * chunk;
    যদি (my_start >= total_outputs) {
        ঘূৰাই 0;
    }
    uint64_t my_end = my_start + chunk;
    যদি (my_end > total_outputs) {
        my_end = total_outputs;
    }

    q8_dot_state q8_state;
    q8_dot_begin(&q8_state);

    ধ্রুৱ uint64_t per_batch = (uint64_t) M * (uint64_t) n_expert_used;

    uint64_t idx = my_start;
    যেতিয়ালৈ (idx < my_end) {
        ধ্রুৱ int64_t  batch_idx = (int64_t) (idx / per_batch);
        ধ্রুৱ uint64_t rem       = idx - (uint64_t) batch_idx * per_batch;
        ধ্রুৱ int64_t  slot_idx  = (int64_t) (rem / (uint64_t) M);
        ধ্রুৱ int64_t  m0        = (int64_t) (rem - (uint64_t) slot_idx * (uint64_t) M);

        ধ্রুৱ uint64_t run_end_global =
            (uint64_t) batch_idx * per_batch + (uint64_t) slot_idx * (uint64_t) M + (uint64_t) M;
        ধ্রুৱ uint64_t end_in_my = (run_end_global < my_end) ? run_end_global : my_end;
        int64_t        run_len   = (int64_t) (end_in_my - idx);

        ধ্রুৱ int32_t expert_id =
            *(ধ্রুৱ int32_t *) ((ধ্রুৱ ৱর্ণ *) src2_data + slot_idx * (int64_t) nb20 + batch_idx * (int64_t) nb21);

        ৱর্ণ * dst_slot = (ৱর্ণ *) dst_data + slot_idx * (int64_t) nbd1 + batch_idx * (int64_t) nbd2;

        যদি (expert_id < 0 || expert_id >= n_expert) {
            int64_t m = m0;
            বাবে (int64_t i = 0; i < run_len; i++, m++) {
                atomic_store_f32((চঞ্চল ভিন্ন *) (dst_slot + m * (int64_t) nbd0), 0.0f);
            }
            idx += (uint64_t) run_len;
            অনুৱর্ত;
        }

        ধ্রুৱ int64_t col_idx = slot_idx % ne11;
        ধ্রুৱ ভিন্ন * b_col_base =
            (ধ্রুৱ ভিন্ন *) ((ধ্রুৱ ৱর্ণ *) src1_data + col_idx * (int64_t) nb11 + batch_idx * (int64_t) nb12);
        ধ্রুৱ ৱর্ণ * expert_base = (ধ্রুৱ ৱর্ণ *) src0_data + expert_id * (int64_t) nb02;

        int64_t m    = m0;
        int64_t বাম = run_len;

        যদি (use_x2) {
            যেতিয়ালৈ (বাম >= 2) {
                ধ্রুৱ block_q8_0 * row0 = (ধ্রুৱ block_q8_0 *) (expert_base + m * (int64_t) nb01);
                ধ্রুৱ block_q8_0 * row1 = (ধ্রুৱ block_q8_0 *) (expert_base + (m + 1) * (int64_t) nb01);
                ভিন্ন              s0, s1;
                q8_dot_compute_x2_aligned(row0, row1, b_col_base, K_blocks, &s0, &s1);
                atomic_store_f32((চঞ্চল ভিন্ন *) (dst_slot + m * (int64_t) nbd0), s0);
                atomic_store_f32((চঞ্চল ভিন্ন *) (dst_slot + (m + 1) * (int64_t) nbd0), s1);
                m += 2;
                বাম -= 2;
            }
        }

        যেতিয়ালৈ (বাম > 0) {
            ধ্রুৱ block_q8_0 * row = (ধ্রুৱ block_q8_0 *) (expert_base + m * (int64_t) nb01);
            ভিন্ন              s   = q8_dot_compute(row, b_col_base, K_blocks);
            atomic_store_f32((চঞ্চল ভিন্ন *) (dst_slot + m * (int64_t) nbd0), s);
            m++;
            বাম--;
        }

        idx += (uint64_t) run_len;
    }

    q8_dot_end(&q8_state);
    ঘূৰাই 0;
}
