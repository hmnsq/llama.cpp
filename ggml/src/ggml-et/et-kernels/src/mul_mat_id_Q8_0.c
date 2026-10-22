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

#ಅನ್ತರ್ಭಾವ "block_ops.h"
#ಅನ್ತರ್ಭಾವ "ggml_tensor.h"
#ಅನ್ತರ್ಭಾವ "math_fp.h"
#ಅನ್ತರ್ಭಾವ "platform.h"
#ಅನ್ತರ್ಭಾವ "quants.h"

#ಅನ್ತರ್ಭಾವ <stdint.h>

ಪೂರ್ಣಾಙ್ಕ entry_point(ಸಂರಚನಾ ggml_et_mul_mat_id_params * params, ಶೂನ್ಯ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;
    ಯದಿ (!kernel_env || !params) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಪೂರ್ಣಾಙ್ಕ thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_threads = get_ಅಂಕೆ_threads(kernel_env->shire_mask);
    ಯದಿ (thread_id < 0) {
        ಪ್ರತ್ಯಾಗಮ 0;
    }

    ಸಂರಚನಾ ggml_tensor * src0 = &params->src0;
    ಸಂರಚನಾ ggml_tensor * src1 = &params->src1;
    ಸಂರಚನಾ ggml_tensor * src2 = &params->src2;
    ಸಂರಚನಾ ggml_tensor * dst  = &params->dst;

    ಯದಿ (src0->type != GGML_TYPE_Q8_0 || src1->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32 ||
        dst->type != GGML_TYPE_F32) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಧ್ರುವ ಶೂನ್ಯ *    src0_data = src0->data;
    ಧ್ರುವ ಭಿನ್ನ *   src1_data = (ಧ್ರುವ ಭಿನ್ನ *) src1->data;
    ಧ್ರುವ int32_t * src2_data = (ಧ್ರುವ int32_t *) src2->data;
    ಭಿನ್ನ *         dst_data  = (ಭಿನ್ನ *) dst->data;
    ಯದಿ (!src0_data || !src1_data || !src2_data || !dst_data) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಧ್ರುವ int64_t K             = src0->ne[0];
    ಧ್ರುವ int64_t M             = src0->ne[1];
    ಧ್ರುವ int64_t n_expert      = src0->ne[2];
    ಧ್ರುವ int64_t n_expert_used = src2->ne[0];
    ಧ್ರುವ int64_t batch         = src2->ne[1];
    ಧ್ರುವ int64_t ne11          = src1->ne[1];

    ಯದಿ (K % QK8_0 != 0) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಧ್ರುವ size_t nb01 = src0->nb[1];
    ಧ್ರುವ size_t nb02 = src0->nb[2];
    ಧ್ರುವ size_t nb11 = src1->nb[1];
    ಧ್ರುವ size_t nb12 = src1->nb[2];
    ಧ್ರುವ size_t nb20 = src2->nb[0];
    ಧ್ರುವ size_t nb21 = src2->nb[1];
    ಧ್ರುವ size_t nbd0 = dst->nb[0];
    ಧ್ರುವ size_t nbd1 = dst->nb[1];
    ಧ್ರುವ size_t nbd2 = dst->nb[2];

    ಯದಿ (src0->nb[0] != ಪರಿಮಾಣ(block_q8_0) || src1->nb[0] != ಪರಿಮಾಣ(ಭಿನ್ನ) || src2->nb[0] != ಪರಿಮಾಣ(int32_t) ||
        nbd0 != ಪರಿಮಾಣ(ಭಿನ್ನ)) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಧ್ರುವ int64_t K_blocks = K / QK8_0;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ     use_x2   = ((nb01 & 31) == 0);

    ಧ್ರುವ uint64_t total_outputs = (uint64_t) M * (uint64_t) n_expert_used * (uint64_t) batch;
    ಯದಿ (total_outputs == 0) {
        ಪ್ರತ್ಯಾಗಮ 0;
    }

    ಧ್ರುವ uint64_t chunk    = (total_outputs + (uint64_t) ಅಂಕೆ_threads - 1) / (uint64_t) ಅಂಕೆ_threads;
    ಧ್ರುವ uint64_t my_start = (uint64_t) thread_id * chunk;
    ಯದಿ (my_start >= total_outputs) {
        ಪ್ರತ್ಯಾಗಮ 0;
    }
    uint64_t my_end = my_start + chunk;
    ಯದಿ (my_end > total_outputs) {
        my_end = total_outputs;
    }

    q8_dot_state q8_state;
    q8_dot_begin(&q8_state);

    ಧ್ರುವ uint64_t per_batch = (uint64_t) M * (uint64_t) n_expert_used;

    uint64_t idx = my_start;
    ಯಾವತ್ (idx < my_end) {
        ಧ್ರುವ int64_t  batch_idx = (int64_t) (idx / per_batch);
        ಧ್ರುವ uint64_t rem       = idx - (uint64_t) batch_idx * per_batch;
        ಧ್ರುವ int64_t  slot_idx  = (int64_t) (rem / (uint64_t) M);
        ಧ್ರುವ int64_t  m0        = (int64_t) (rem - (uint64_t) slot_idx * (uint64_t) M);

        ಧ್ರುವ uint64_t run_end_global =
            (uint64_t) batch_idx * per_batch + (uint64_t) slot_idx * (uint64_t) M + (uint64_t) M;
        ಧ್ರುವ uint64_t end_in_my = (run_end_global < my_end) ? run_end_global : my_end;
        int64_t        run_len   = (int64_t) (end_in_my - idx);

        ಧ್ರುವ int32_t expert_id =
            *(ಧ್ರುವ int32_t *) ((ಧ್ರುವ ವರ್ಣ *) src2_data + slot_idx * (int64_t) nb20 + batch_idx * (int64_t) nb21);

        ವರ್ಣ * dst_slot = (ವರ್ಣ *) dst_data + slot_idx * (int64_t) nbd1 + batch_idx * (int64_t) nbd2;

        ಯದಿ (expert_id < 0 || expert_id >= n_expert) {
            int64_t m = m0;
            ಪ್ರತಿ (int64_t i = 0; i < run_len; i++, m++) {
                atomic_store_f32((ಚಞ್ಚಲ ಭಿನ್ನ *) (dst_slot + m * (int64_t) nbd0), 0.0f);
            }
            idx += (uint64_t) run_len;
            ಅನುವರ್ತ;
        }

        ಧ್ರುವ int64_t col_idx = slot_idx % ne11;
        ಧ್ರುವ ಭಿನ್ನ * b_col_base =
            (ಧ್ರುವ ಭಿನ್ನ *) ((ಧ್ರುವ ವರ್ಣ *) src1_data + col_idx * (int64_t) nb11 + batch_idx * (int64_t) nb12);
        ಧ್ರುವ ವರ್ಣ * expert_base = (ಧ್ರುವ ವರ್ಣ *) src0_data + expert_id * (int64_t) nb02;

        int64_t m    = m0;
        int64_t ಎಡ = run_len;

        ಯದಿ (use_x2) {
            ಯಾವತ್ (ಎಡ >= 2) {
                ಧ್ರುವ block_q8_0 * row0 = (ಧ್ರುವ block_q8_0 *) (expert_base + m * (int64_t) nb01);
                ಧ್ರುವ block_q8_0 * row1 = (ಧ್ರುವ block_q8_0 *) (expert_base + (m + 1) * (int64_t) nb01);
                ಭಿನ್ನ              s0, s1;
                q8_dot_compute_x2_aligned(row0, row1, b_col_base, K_blocks, &s0, &s1);
                atomic_store_f32((ಚಞ್ಚಲ ಭಿನ್ನ *) (dst_slot + m * (int64_t) nbd0), s0);
                atomic_store_f32((ಚಞ್ಚಲ ಭಿನ್ನ *) (dst_slot + (m + 1) * (int64_t) nbd0), s1);
                m += 2;
                ಎಡ -= 2;
            }
        }

        ಯಾವತ್ (ಎಡ > 0) {
            ಧ್ರುವ block_q8_0 * row = (ಧ್ರುವ block_q8_0 *) (expert_base + m * (int64_t) nb01);
            ಭಿನ್ನ              s   = q8_dot_compute(row, b_col_base, K_blocks);
            atomic_store_f32((ಚಞ್ಚಲ ಭಿನ್ನ *) (dst_slot + m * (int64_t) nbd0), s);
            m++;
            ಎಡ--;
        }

        idx += (uint64_t) run_len;
    }

    q8_dot_end(&q8_state);
    ಪ್ರತ್ಯಾಗಮ 0;
}
