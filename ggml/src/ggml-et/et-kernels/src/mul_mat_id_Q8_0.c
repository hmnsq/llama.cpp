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

#ضمن "block_ops.h"
#ضمن "ggml_tensor.h"
#ضمن "math_fp.h"
#ضمن "platform.h"
#ضمن "quants.h"

#ضمن <stdint.h>

صحيح entry_point(بنية ggml_et_mul_mat_id_params * params, فراغ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;
    𐡄𐡍 (!kernel_env || !params) {
        أرجع -1;
    }

    صحيح thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    صحيح num_threads = get_num_threads(kernel_env->shire_mask);
    𐡄𐡍 (thread_id < 0) {
        أرجع 0;
    }

    بنية ggml_tensor * src0 = &params->src0;
    بنية ggml_tensor * src1 = &params->src1;
    بنية ggml_tensor * src2 = &params->src2;
    بنية ggml_tensor * dst  = &params->dst;

    𐡄𐡍 (src0->type != GGML_TYPE_Q8_0 || src1->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32 ||
        dst->type != GGML_TYPE_F32) {
        أرجع -1;
    }

    ثابت فراغ *    src0_data = src0->data;
    ثابت عائم *   src1_data = (ثابت عائم *) src1->data;
    ثابت int32_t * src2_data = (ثابت int32_t *) src2->data;
    عائم *         dst_data  = (عائم *) dst->data;
    𐡄𐡍 (!src0_data || !src1_data || !src2_data || !dst_data) {
        أرجع -1;
    }

    ثابت int64_t K             = src0->ne[0];
    ثابت int64_t M             = src0->ne[1];
    ثابت int64_t n_expert      = src0->ne[2];
    ثابت int64_t n_expert_used = src2->ne[0];
    ثابت int64_t batch         = src2->ne[1];
    ثابت int64_t ne11          = src1->ne[1];

    𐡄𐡍 (K % QK8_0 != 0) {
        أرجع -1;
    }

    ثابت size_t nb01 = src0->nb[1];
    ثابت size_t nb02 = src0->nb[2];
    ثابت size_t nb11 = src1->nb[1];
    ثابت size_t nb12 = src1->nb[2];
    ثابت size_t nb20 = src2->nb[0];
    ثابت size_t nb21 = src2->nb[1];
    ثابت size_t nbd0 = dst->nb[0];
    ثابت size_t nbd1 = dst->nb[1];
    ثابت size_t nbd2 = dst->nb[2];

    𐡄𐡍 (src0->nb[0] != حجم(block_q8_0) || src1->nb[0] != حجم(عائم) || src2->nb[0] != حجم(int32_t) ||
        nbd0 != حجم(عائم)) {
        أرجع -1;
    }

    ثابت int64_t K_blocks = K / QK8_0;
    ثابت صحيح     use_x2   = ((nb01 & 31) == 0);

    ثابت uint64_t total_outputs = (uint64_t) M * (uint64_t) n_expert_used * (uint64_t) batch;
    𐡄𐡍 (total_outputs == 0) {
        أرجع 0;
    }

    ثابت uint64_t chunk    = (total_outputs + (uint64_t) num_threads - 1) / (uint64_t) num_threads;
    ثابت uint64_t my_start = (uint64_t) thread_id * chunk;
    𐡄𐡍 (my_start >= total_outputs) {
        أرجع 0;
    }
    uint64_t my_end = my_start + chunk;
    𐡄𐡍 (my_end > total_outputs) {
        my_end = total_outputs;
    }

    q8_dot_state q8_state;
    q8_dot_begin(&q8_state);

    ثابت uint64_t per_batch = (uint64_t) M * (uint64_t) n_expert_used;

    uint64_t idx = my_start;
    طالما (idx < my_end) {
        ثابت int64_t  batch_idx = (int64_t) (idx / per_batch);
        ثابت uint64_t rem       = idx - (uint64_t) batch_idx * per_batch;
        ثابت int64_t  slot_idx  = (int64_t) (rem / (uint64_t) M);
        ثابت int64_t  m0        = (int64_t) (rem - (uint64_t) slot_idx * (uint64_t) M);

        ثابت uint64_t run_end_global =
            (uint64_t) batch_idx * per_batch + (uint64_t) slot_idx * (uint64_t) M + (uint64_t) M;
        ثابت uint64_t end_in_my = (run_end_global < my_end) ? run_end_global : my_end;
        int64_t        run_len   = (int64_t) (end_in_my - idx);

        ثابت int32_t expert_id =
            *(ثابت int32_t *) ((ثابت حرف *) src2_data + slot_idx * (int64_t) nb20 + batch_idx * (int64_t) nb21);

        حرف * dst_slot = (حرف *) dst_data + slot_idx * (int64_t) nbd1 + batch_idx * (int64_t) nbd2;

        𐡄𐡍 (expert_id < 0 || expert_id >= n_expert) {
            int64_t m = m0;
            لكل (int64_t i = 0; i < run_len; i++, m++) {
                atomic_store_f32((متطاير عائم *) (dst_slot + m * (int64_t) nbd0), 0.0f);
            }
            idx += (uint64_t) run_len;
            استمر;
        }

        ثابت int64_t col_idx = slot_idx % ne11;
        ثابت عائم * b_col_base =
            (ثابت عائم *) ((ثابت حرف *) src1_data + col_idx * (int64_t) nb11 + batch_idx * (int64_t) nb12);
        ثابت حرف * expert_base = (ثابت حرف *) src0_data + expert_id * (int64_t) nb02;

        int64_t m    = m0;
        int64_t left = run_len;

        𐡄𐡍 (use_x2) {
            طالما (left >= 2) {
                ثابت block_q8_0 * row0 = (ثابت block_q8_0 *) (expert_base + m * (int64_t) nb01);
                ثابت block_q8_0 * row1 = (ثابت block_q8_0 *) (expert_base + (m + 1) * (int64_t) nb01);
                عائم              s0, s1;
                q8_dot_compute_x2_aligned(row0, row1, b_col_base, K_blocks, &s0, &s1);
                atomic_store_f32((متطاير عائم *) (dst_slot + m * (int64_t) nbd0), s0);
                atomic_store_f32((متطاير عائم *) (dst_slot + (m + 1) * (int64_t) nbd0), s1);
                m += 2;
                left -= 2;
            }
        }

        طالما (left > 0) {
            ثابت block_q8_0 * row = (ثابت block_q8_0 *) (expert_base + m * (int64_t) nb01);
            عائم              s   = q8_dot_compute(row, b_col_base, K_blocks);
            atomic_store_f32((متطاير عائم *) (dst_slot + m * (int64_t) nbd0), s);
            m++;
            left--;
        }

        idx += (uint64_t) run_len;
    }

    q8_dot_end(&q8_state);
    أرجع 0;
}
