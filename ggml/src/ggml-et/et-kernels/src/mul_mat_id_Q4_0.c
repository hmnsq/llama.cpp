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

#شامل "block_ops.h"
#شامل "ggml_tensor.h"
#شامل "math_fp.h"
#شامل "platform.h"
#شامل "quants.h"

#شامل <stdint.h>

صحیح entry_point(ساختار ggml_et_mul_mat_id_params * params, تهی * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;
    اگر (!kernel_env || !params) {
        بازگشت -1;
    }

    صحیح thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    صحیح رقم_threads = get_رقم_threads(kernel_env->shire_mask);
    اگر (thread_id < 0) {
        بازگشت 0;
    }

    ساختار ggml_tensor * src0 = &params->src0;
    ساختار ggml_tensor * src1 = &params->src1;
    ساختار ggml_tensor * src2 = &params->src2;
    ساختار ggml_tensor * dst  = &params->dst;

    اگر (src0->type != GGML_TYPE_Q4_0 || src1->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32 ||
        dst->type != GGML_TYPE_F32) {
        بازگشت -1;
    }

    ثابت تهی *    src0_data = src0->data;
    ثابت اعشاری *   src1_data = (ثابت اعشاری *) src1->data;
    ثابت int32_t * src2_data = (ثابت int32_t *) src2->data;
    اعشاری *         dst_data  = (اعشاری *) dst->data;
    اگر (!src0_data || !src1_data || !src2_data || !dst_data) {
        بازگشت -1;
    }

    ثابت int64_t K             = src0->ne[0];
    ثابت int64_t M             = src0->ne[1];
    ثابت int64_t n_expert      = src0->ne[2];
    ثابت int64_t n_expert_used = src2->ne[0];
    ثابت int64_t batch         = src2->ne[1];
    ثابت int64_t ne11          = src1->ne[1];

    اگر (K % QK4_0 != 0) {
        بازگشت -1;
    }

    ثابت size_t nb01 = src0->nb[1];  // Q4_0 row stride
    ثابت size_t nb02 = src0->nb[2];  // expert stride
    ثابت size_t nb11 = src1->nb[1];  // activation column stride
    ثابت size_t nb12 = src1->nb[2];  // activation batch stride
    ثابت size_t nb20 = src2->nb[0];
    ثابت size_t nb21 = src2->nb[1];
    ثابت size_t nbd0 = dst->nb[0];
    ثابت size_t nbd1 = dst->nb[1];
    ثابت size_t nbd2 = dst->nb[2];

    اگر (src0->nb[0] != اندازه(block_q4_0) || src1->nb[0] != اندازه(اعشاری) || src2->nb[0] != اندازه(int32_t) ||
        nbd0 != اندازه(اعشاری)) {
        بازگشت -1;
    }

    ثابت int64_t K_blocks = K / QK4_0;
    ثابت صحیح     use_x2   = ((nb01 & 31) == 0);

    ثابت uint64_t total_outputs = (uint64_t) M * (uint64_t) n_expert_used * (uint64_t) batch;
    اگر (total_outputs == 0) {
        بازگشت 0;
    }

    // Even partition: hart h owns outputs [h*chunk, (h+1)*chunk).
    ثابت uint64_t chunk    = (total_outputs + (uint64_t) رقم_threads - 1) / (uint64_t) رقم_threads;
    ثابت uint64_t my_start = (uint64_t) thread_id * chunk;
    اگر (my_start >= total_outputs) {
        بازگشت 0;
    }
    uint64_t my_end = my_start + chunk;
    اگر (my_end > total_outputs) {
        my_end = total_outputs;
    }

    // Save mask register once; full lanes for vector dot.
    q4_dot_state q4_state;
    q4_dot_begin(&q4_state);

    ثابت uint64_t per_batch = (uint64_t) M * (uint64_t) n_expert_used;

    uint64_t idx = my_start;
    وقتی (idx < my_end) {
        // Decode (m, slot, batch) from the m-major linear index.
        ثابت int64_t  batch_idx = (int64_t) (idx / per_batch);
        ثابت uint64_t rem       = idx - (uint64_t) batch_idx * per_batch;
        ثابت int64_t  slot_idx  = (int64_t) (rem / (uint64_t) M);
        ثابت int64_t  m0        = (int64_t) (rem - (uint64_t) slot_idx * (uint64_t) M);

        // How many outputs left in this (slot, batch) run AND in my range.
        ثابت uint64_t run_end_global =
            (uint64_t) batch_idx * per_batch + (uint64_t) slot_idx * (uint64_t) M + (uint64_t) M;
        ثابت uint64_t end_in_my = (run_end_global < my_end) ? run_end_global : my_end;
        int64_t        run_len   = (int64_t) (end_in_my - idx);

        // Resolve expert + B column + dst slot for this run.
        ثابت int32_t expert_id =
            *(ثابت int32_t *) ((ثابت نویسه *) src2_data + slot_idx * (int64_t) nb20 + batch_idx * (int64_t) nb21);

        نویسه * dst_slot = (نویسه *) dst_data + slot_idx * (int64_t) nbd1 + batch_idx * (int64_t) nbd2;

        اگر (expert_id < 0 || expert_id >= n_expert) {
            // Invalid expert id — zero out this run's outputs.
            int64_t m = m0;
            برای (int64_t i = 0; i < run_len; i++, m++) {
                atomic_store_f32((فرار اعشاری *) (dst_slot + m * (int64_t) nbd0), 0.0f);
            }
            idx += (uint64_t) run_len;
            ادامه;
        }

        ثابت int64_t col_idx = slot_idx % ne11;
        ثابت اعشاری * b_col_base =
            (ثابت اعشاری *) ((ثابت نویسه *) src1_data + col_idx * (int64_t) nb11 + batch_idx * (int64_t) nb12);
        ثابت نویسه * expert_base = (ثابت نویسه *) src0_data + expert_id * (int64_t) nb02;

        int64_t m    = m0;
        int64_t چپ = run_len;

        // Paired-row dots: halves B bandwidth for runs >= 2.
        اگر (use_x2) {
            وقتی (چپ >= 2) {
                ثابت block_q4_0 * row0 = (ثابت block_q4_0 *) (expert_base + m * (int64_t) nb01);
                ثابت block_q4_0 * row1 = (ثابت block_q4_0 *) (expert_base + (m + 1) * (int64_t) nb01);
                اعشاری              s0, s1;
                q4_dot_compute_x2_aligned(row0, row1, b_col_base, K_blocks, &s0, &s1);
                atomic_store_f32((فرار اعشاری *) (dst_slot + m * (int64_t) nbd0), s0);
                atomic_store_f32((فرار اعشاری *) (dst_slot + (m + 1) * (int64_t) nbd0), s1);
                m += 2;
                چپ -= 2;
            }
        }

        // Tail / non-aligned fallback: single-row dots.
        وقتی (چپ > 0) {
            ثابت block_q4_0 * row = (ثابت block_q4_0 *) (expert_base + m * (int64_t) nb01);
            اعشاری              s   = q4_dot_compute(row, b_col_base, K_blocks);
            atomic_store_f32((فرار اعشاری *) (dst_slot + m * (int64_t) nbd0), s);
            m++;
            چپ--;
        }

        idx += (uint64_t) run_len;
    }

    q4_dot_end(&q4_state);
    بازگشت 0;
}
