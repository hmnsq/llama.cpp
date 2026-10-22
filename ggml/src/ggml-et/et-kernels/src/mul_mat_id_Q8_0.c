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

#അന്തര്ഭാവ "block_ops.h"
#അന്തര്ഭാവ "ggml_tensor.h"
#അന്തര്ഭാവ "math_fp.h"
#അന്തര്ഭാവ "platform.h"
#അന്തര്ഭാവ "quants.h"

#അന്തര്ഭാവ <stdint.h>

പൂർണ്ണസംഖ്യ entry_point(ഘടന ggml_et_mul_mat_id_params * params, ശൂന്യം * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;
    എങ്കിൽ (!kernel_env || !params) {
        തിരികെ -1;
    }

    പൂർണ്ണസംഖ്യ thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    പൂർണ്ണസംഖ്യ അക്കം_threads = get_അക്കം_threads(kernel_env->shire_mask);
    എങ്കിൽ (thread_id < 0) {
        തിരികെ 0;
    }

    ഘടന ggml_tensor * src0 = &params->src0;
    ഘടന ggml_tensor * src1 = &params->src1;
    ഘടന ggml_tensor * src2 = &params->src2;
    ഘടന ggml_tensor * dst  = &params->dst;

    എങ്കിൽ (src0->type != GGML_TYPE_Q8_0 || src1->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32 ||
        dst->type != GGML_TYPE_F32) {
        തിരികെ -1;
    }

    സ്ഥിരം ശൂന്യം *    src0_data = src0->data;
    സ്ഥിരം ദശാംശം *   src1_data = (സ്ഥിരം ദശാംശം *) src1->data;
    സ്ഥിരം int32_t * src2_data = (സ്ഥിരം int32_t *) src2->data;
    ദശാംശം *         dst_data  = (ദശാംശം *) dst->data;
    എങ്കിൽ (!src0_data || !src1_data || !src2_data || !dst_data) {
        തിരികെ -1;
    }

    സ്ഥിരം int64_t K             = src0->ne[0];
    സ്ഥിരം int64_t M             = src0->ne[1];
    സ്ഥിരം int64_t n_expert      = src0->ne[2];
    സ്ഥിരം int64_t n_expert_used = src2->ne[0];
    സ്ഥിരം int64_t batch         = src2->ne[1];
    സ്ഥിരം int64_t ne11          = src1->ne[1];

    എങ്കിൽ (K % QK8_0 != 0) {
        തിരികെ -1;
    }

    സ്ഥിരം size_t nb01 = src0->nb[1];
    സ്ഥിരം size_t nb02 = src0->nb[2];
    സ്ഥിരം size_t nb11 = src1->nb[1];
    സ്ഥിരം size_t nb12 = src1->nb[2];
    സ്ഥിരം size_t nb20 = src2->nb[0];
    സ്ഥിരം size_t nb21 = src2->nb[1];
    സ്ഥിരം size_t nbd0 = dst->nb[0];
    സ്ഥിരം size_t nbd1 = dst->nb[1];
    സ്ഥിരം size_t nbd2 = dst->nb[2];

    എങ്കിൽ (src0->nb[0] != വലുപ്പം(block_q8_0) || src1->nb[0] != വലുപ്പം(ദശാംശം) || src2->nb[0] != വലുപ്പം(int32_t) ||
        nbd0 != വലുപ്പം(ദശാംശം)) {
        തിരികെ -1;
    }

    സ്ഥിരം int64_t K_blocks = K / QK8_0;
    സ്ഥിരം പൂർണ്ണസംഖ്യ     use_x2   = ((nb01 & 31) == 0);

    സ്ഥിരം uint64_t total_outputs = (uint64_t) M * (uint64_t) n_expert_used * (uint64_t) batch;
    എങ്കിൽ (total_outputs == 0) {
        തിരികെ 0;
    }

    സ്ഥിരം uint64_t chunk    = (total_outputs + (uint64_t) അക്കം_threads - 1) / (uint64_t) അക്കം_threads;
    സ്ഥിരം uint64_t my_start = (uint64_t) thread_id * chunk;
    എങ്കിൽ (my_start >= total_outputs) {
        തിരികെ 0;
    }
    uint64_t my_end = my_start + chunk;
    എങ്കിൽ (my_end > total_outputs) {
        my_end = total_outputs;
    }

    q8_dot_state q8_state;
    q8_dot_begin(&q8_state);

    സ്ഥിരം uint64_t per_batch = (uint64_t) M * (uint64_t) n_expert_used;

    uint64_t idx = my_start;
    ആകുമ്പോൾ (idx < my_end) {
        സ്ഥിരം int64_t  batch_idx = (int64_t) (idx / per_batch);
        സ്ഥിരം uint64_t rem       = idx - (uint64_t) batch_idx * per_batch;
        സ്ഥിരം int64_t  slot_idx  = (int64_t) (rem / (uint64_t) M);
        സ്ഥിരം int64_t  m0        = (int64_t) (rem - (uint64_t) slot_idx * (uint64_t) M);

        സ്ഥിരം uint64_t run_end_global =
            (uint64_t) batch_idx * per_batch + (uint64_t) slot_idx * (uint64_t) M + (uint64_t) M;
        സ്ഥിരം uint64_t end_in_my = (run_end_global < my_end) ? run_end_global : my_end;
        int64_t        run_len   = (int64_t) (end_in_my - idx);

        സ്ഥിരം int32_t expert_id =
            *(സ്ഥിരം int32_t *) ((സ്ഥിരം അക്ഷരം *) src2_data + slot_idx * (int64_t) nb20 + batch_idx * (int64_t) nb21);

        അക്ഷരം * dst_slot = (അക്ഷരം *) dst_data + slot_idx * (int64_t) nbd1 + batch_idx * (int64_t) nbd2;

        എങ്കിൽ (expert_id < 0 || expert_id >= n_expert) {
            int64_t m = m0;
            വേണ്ടി (int64_t i = 0; i < run_len; i++, m++) {
                atomic_store_f32((അസ്ഥിര ദശാംശം *) (dst_slot + m * (int64_t) nbd0), 0.0f);
            }
            idx += (uint64_t) run_len;
            തുടരുക;
        }

        സ്ഥിരം int64_t col_idx = slot_idx % ne11;
        സ്ഥിരം ദശാംശം * b_col_base =
            (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) src1_data + col_idx * (int64_t) nb11 + batch_idx * (int64_t) nb12);
        സ്ഥിരം അക്ഷരം * expert_base = (സ്ഥിരം അക്ഷരം *) src0_data + expert_id * (int64_t) nb02;

        int64_t m    = m0;
        int64_t ഇടത് = run_len;

        എങ്കിൽ (use_x2) {
            ആകുമ്പോൾ (ഇടത് >= 2) {
                സ്ഥിരം block_q8_0 * row0 = (സ്ഥിരം block_q8_0 *) (expert_base + m * (int64_t) nb01);
                സ്ഥിരം block_q8_0 * row1 = (സ്ഥിരം block_q8_0 *) (expert_base + (m + 1) * (int64_t) nb01);
                ദശാംശം              s0, s1;
                q8_dot_compute_x2_aligned(row0, row1, b_col_base, K_blocks, &s0, &s1);
                atomic_store_f32((അസ്ഥിര ദശാംശം *) (dst_slot + m * (int64_t) nbd0), s0);
                atomic_store_f32((അസ്ഥിര ദശാംശം *) (dst_slot + (m + 1) * (int64_t) nbd0), s1);
                m += 2;
                ഇടത് -= 2;
            }
        }

        ആകുമ്പോൾ (ഇടത് > 0) {
            സ്ഥിരം block_q8_0 * row = (സ്ഥിരം block_q8_0 *) (expert_base + m * (int64_t) nb01);
            ദശാംശം              s   = q8_dot_compute(row, b_col_base, K_blocks);
            atomic_store_f32((അസ്ഥിര ദശാംശം *) (dst_slot + m * (int64_t) nbd0), s);
            m++;
            ഇടത്--;
        }

        idx += (uint64_t) run_len;
    }

    q8_dot_end(&q8_state);
    തിരികെ 0;
}
