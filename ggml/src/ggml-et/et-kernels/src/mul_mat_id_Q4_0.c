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

#அந்தர்பாவ "block_ops.h"
#அந்தர்பாவ "ggml_tensor.h"
#அந்தர்பாவ "math_fp.h"
#அந்தர்பாவ "platform.h"
#அந்தர்பாவ "quants.h"

#அந்தர்பாவ <stdint.h>

முழு entry_point(கட்டமைப்பு ggml_et_mul_mat_id_params * params, வெறுமை * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;
    எனில் (!kernel_env || !params) {
        திரும்பு -1;
    }

    முழு thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    முழு இலக்கம்_threads = get_இலக்கம்_threads(kernel_env->shire_mask);
    எனில் (thread_id < 0) {
        திரும்பு 0;
    }

    கட்டமைப்பு ggml_tensor * src0 = &params->src0;
    கட்டமைப்பு ggml_tensor * src1 = &params->src1;
    கட்டமைப்பு ggml_tensor * src2 = &params->src2;
    கட்டமைப்பு ggml_tensor * dst  = &params->dst;

    எனில் (src0->type != GGML_TYPE_Q4_0 || src1->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32 ||
        dst->type != GGML_TYPE_F32) {
        திரும்பு -1;
    }

    மாறிலி வெறுமை *    src0_data = src0->data;
    மாறிலி தசம *   src1_data = (மாறிலி தசம *) src1->data;
    மாறிலி int32_t * src2_data = (மாறிலி int32_t *) src2->data;
    தசம *         dst_data  = (தசம *) dst->data;
    எனில் (!src0_data || !src1_data || !src2_data || !dst_data) {
        திரும்பு -1;
    }

    மாறிலி int64_t K             = src0->ne[0];
    மாறிலி int64_t M             = src0->ne[1];
    மாறிலி int64_t n_expert      = src0->ne[2];
    மாறிலி int64_t n_expert_used = src2->ne[0];
    மாறிலி int64_t batch         = src2->ne[1];
    மாறிலி int64_t ne11          = src1->ne[1];

    எனில் (K % QK4_0 != 0) {
        திரும்பு -1;
    }

    மாறிலி size_t nb01 = src0->nb[1];  // Q4_0 row stride
    மாறிலி size_t nb02 = src0->nb[2];  // expert stride
    மாறிலி size_t nb11 = src1->nb[1];  // activation column stride
    மாறிலி size_t nb12 = src1->nb[2];  // activation batch stride
    மாறிலி size_t nb20 = src2->nb[0];
    மாறிலி size_t nb21 = src2->nb[1];
    மாறிலி size_t nbd0 = dst->nb[0];
    மாறிலி size_t nbd1 = dst->nb[1];
    மாறிலி size_t nbd2 = dst->nb[2];

    எனில் (src0->nb[0] != அளவு(block_q4_0) || src1->nb[0] != அளவு(தசம) || src2->nb[0] != அளவு(int32_t) ||
        nbd0 != அளவு(தசம)) {
        திரும்பு -1;
    }

    மாறிலி int64_t K_blocks = K / QK4_0;
    மாறிலி முழு     use_x2   = ((nb01 & 31) == 0);

    மாறிலி uint64_t total_outputs = (uint64_t) M * (uint64_t) n_expert_used * (uint64_t) batch;
    எனில் (total_outputs == 0) {
        திரும்பு 0;
    }

    // Even partition: hart h owns outputs [h*chunk, (h+1)*chunk).
    மாறிலி uint64_t chunk    = (total_outputs + (uint64_t) இலக்கம்_threads - 1) / (uint64_t) இலக்கம்_threads;
    மாறிலி uint64_t my_start = (uint64_t) thread_id * chunk;
    எனில் (my_start >= total_outputs) {
        திரும்பு 0;
    }
    uint64_t my_end = my_start + chunk;
    எனில் (my_end > total_outputs) {
        my_end = total_outputs;
    }

    // Save mask register once; full lanes for vector dot.
    q4_dot_state q4_state;
    q4_dot_begin(&q4_state);

    மாறிலி uint64_t per_batch = (uint64_t) M * (uint64_t) n_expert_used;

    uint64_t idx = my_start;
    போது (idx < my_end) {
        // Decode (m, slot, batch) from the m-major linear index.
        மாறிலி int64_t  batch_idx = (int64_t) (idx / per_batch);
        மாறிலி uint64_t rem       = idx - (uint64_t) batch_idx * per_batch;
        மாறிலி int64_t  slot_idx  = (int64_t) (rem / (uint64_t) M);
        மாறிலி int64_t  m0        = (int64_t) (rem - (uint64_t) slot_idx * (uint64_t) M);

        // How many outputs left in this (slot, batch) run AND in my range.
        மாறிலி uint64_t run_end_global =
            (uint64_t) batch_idx * per_batch + (uint64_t) slot_idx * (uint64_t) M + (uint64_t) M;
        மாறிலி uint64_t end_in_my = (run_end_global < my_end) ? run_end_global : my_end;
        int64_t        run_len   = (int64_t) (end_in_my - idx);

        // Resolve expert + B column + dst slot for this run.
        மாறிலி int32_t expert_id =
            *(மாறிலி int32_t *) ((மாறிலி எழுத்து *) src2_data + slot_idx * (int64_t) nb20 + batch_idx * (int64_t) nb21);

        எழுத்து * dst_slot = (எழுத்து *) dst_data + slot_idx * (int64_t) nbd1 + batch_idx * (int64_t) nbd2;

        எனில் (expert_id < 0 || expert_id >= n_expert) {
            // Invalid expert id — zero out this run's outputs.
            int64_t m = m0;
            க்கு (int64_t i = 0; i < run_len; i++, m++) {
                atomic_store_f32((நிலையற்ற தசம *) (dst_slot + m * (int64_t) nbd0), 0.0f);
            }
            idx += (uint64_t) run_len;
            தொடர்;
        }

        மாறிலி int64_t col_idx = slot_idx % ne11;
        மாறிலி தசம * b_col_base =
            (மாறிலி தசம *) ((மாறிலி எழுத்து *) src1_data + col_idx * (int64_t) nb11 + batch_idx * (int64_t) nb12);
        மாறிலி எழுத்து * expert_base = (மாறிலி எழுத்து *) src0_data + expert_id * (int64_t) nb02;

        int64_t m    = m0;
        int64_t இடது = run_len;

        // Paired-row dots: halves B bandwidth for runs >= 2.
        எனில் (use_x2) {
            போது (இடது >= 2) {
                மாறிலி block_q4_0 * row0 = (மாறிலி block_q4_0 *) (expert_base + m * (int64_t) nb01);
                மாறிலி block_q4_0 * row1 = (மாறிலி block_q4_0 *) (expert_base + (m + 1) * (int64_t) nb01);
                தசம              s0, s1;
                q4_dot_compute_x2_aligned(row0, row1, b_col_base, K_blocks, &s0, &s1);
                atomic_store_f32((நிலையற்ற தசம *) (dst_slot + m * (int64_t) nbd0), s0);
                atomic_store_f32((நிலையற்ற தசம *) (dst_slot + (m + 1) * (int64_t) nbd0), s1);
                m += 2;
                இடது -= 2;
            }
        }

        // Tail / non-aligned fallback: single-row dots.
        போது (இடது > 0) {
            மாறிலி block_q4_0 * row = (மாறிலி block_q4_0 *) (expert_base + m * (int64_t) nb01);
            தசம              s   = q4_dot_compute(row, b_col_base, K_blocks);
            atomic_store_f32((நிலையற்ற தசம *) (dst_slot + m * (int64_t) nbd0), s);
            m++;
            இடது--;
        }

        idx += (uint64_t) run_len;
    }

    q4_dot_end(&q4_state);
    திரும்பு 0;
}
