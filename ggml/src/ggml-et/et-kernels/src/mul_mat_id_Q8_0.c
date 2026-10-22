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

#अन्तर्भाव "block_ops.h"
#अन्तर्भाव "ggml_tensor.h"
#अन्तर्भाव "math_fp.h"
#अन्तर्भाव "platform.h"
#अन्तर्भाव "quants.h"

#अन्तर्भाव <stdint.h>

पूर्णाङ्क entry_point(संरचना ggml_et_mul_mat_id_params * params, शून्य * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;
    यदि (!kernel_env || !params) {
        फर्काऊ -1;
    }

    पूर्णाङ्क thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    पूर्णाङ्क अङ्क_threads = get_अङ्क_threads(kernel_env->shire_mask);
    यदि (thread_id < 0) {
        फर्काऊ 0;
    }

    संरचना ggml_tensor * src0 = &params->src0;
    संरचना ggml_tensor * src1 = &params->src1;
    संरचना ggml_tensor * src2 = &params->src2;
    संरचना ggml_tensor * dst  = &params->dst;

    यदि (src0->type != GGML_TYPE_Q8_0 || src1->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32 ||
        dst->type != GGML_TYPE_F32) {
        फर्काऊ -1;
    }

    अचर शून्य *    src0_data = src0->data;
    अचर भिन्न *   src1_data = (अचर भिन्न *) src1->data;
    अचर int32_t * src2_data = (अचर int32_t *) src2->data;
    भिन्न *         dst_data  = (भिन्न *) dst->data;
    यदि (!src0_data || !src1_data || !src2_data || !dst_data) {
        फर्काऊ -1;
    }

    अचर int64_t K             = src0->ne[0];
    अचर int64_t M             = src0->ne[1];
    अचर int64_t n_expert      = src0->ne[2];
    अचर int64_t n_expert_used = src2->ne[0];
    अचर int64_t batch         = src2->ne[1];
    अचर int64_t ne11          = src1->ne[1];

    यदि (K % QK8_0 != 0) {
        फर्काऊ -1;
    }

    अचर माप_प्रकार nb01 = src0->nb[1];
    अचर माप_प्रकार nb02 = src0->nb[2];
    अचर माप_प्रकार nb11 = src1->nb[1];
    अचर माप_प्रकार nb12 = src1->nb[2];
    अचर माप_प्रकार nb20 = src2->nb[0];
    अचर माप_प्रकार nb21 = src2->nb[1];
    अचर माप_प्रकार nbd0 = dst->nb[0];
    अचर माप_प्रकार nbd1 = dst->nb[1];
    अचर माप_प्रकार nbd2 = dst->nb[2];

    यदि (src0->nb[0] != परिमाण(block_q8_0) || src1->nb[0] != परिमाण(भिन्न) || src2->nb[0] != परिमाण(int32_t) ||
        nbd0 != परिमाण(भिन्न)) {
        फर्काऊ -1;
    }

    अचर int64_t K_blocks = K / QK8_0;
    अचर पूर्णाङ्क     use_x2   = ((nb01 & 31) == 0);

    अचर uint64_t total_outputs = (uint64_t) M * (uint64_t) n_expert_used * (uint64_t) batch;
    यदि (total_outputs == 0) {
        फर्काऊ 0;
    }

    अचर uint64_t chunk    = (total_outputs + (uint64_t) अङ्क_threads - 1) / (uint64_t) अङ्क_threads;
    अचर uint64_t my_start = (uint64_t) thread_id * chunk;
    यदि (my_start >= total_outputs) {
        फर्काऊ 0;
    }
    uint64_t my_end = my_start + chunk;
    यदि (my_end > total_outputs) {
        my_end = total_outputs;
    }

    q8_dot_state q8_state;
    q8_dot_begin(&q8_state);

    अचर uint64_t per_batch = (uint64_t) M * (uint64_t) n_expert_used;

    uint64_t idx = my_start;
    जबसम्म (idx < my_end) {
        अचर int64_t  batch_idx = (int64_t) (idx / per_batch);
        अचर uint64_t rem       = idx - (uint64_t) batch_idx * per_batch;
        अचर int64_t  slot_idx  = (int64_t) (rem / (uint64_t) M);
        अचर int64_t  m0        = (int64_t) (rem - (uint64_t) slot_idx * (uint64_t) M);

        अचर uint64_t run_end_global =
            (uint64_t) batch_idx * per_batch + (uint64_t) slot_idx * (uint64_t) M + (uint64_t) M;
        अचर uint64_t end_in_my = (run_end_global < my_end) ? run_end_global : my_end;
        int64_t        run_len   = (int64_t) (end_in_my - idx);

        अचर int32_t expert_id =
            *(अचर int32_t *) ((अचर वर्ण *) src2_data + slot_idx * (int64_t) nb20 + batch_idx * (int64_t) nb21);

        वर्ण * dst_slot = (वर्ण *) dst_data + slot_idx * (int64_t) nbd1 + batch_idx * (int64_t) nbd2;

        यदि (expert_id < 0 || expert_id >= n_expert) {
            int64_t m = m0;
            लागि (int64_t i = 0; i < run_len; i++, m++) {
                atomic_store_f32((चञ्चल भिन्न *) (dst_slot + m * (int64_t) nbd0), 0.0f);
            }
            idx += (uint64_t) run_len;
            जारी;
        }

        अचर int64_t col_idx = slot_idx % ne11;
        अचर भिन्न * b_col_base =
            (अचर भिन्न *) ((अचर वर्ण *) src1_data + col_idx * (int64_t) nb11 + batch_idx * (int64_t) nb12);
        अचर वर्ण * expert_base = (अचर वर्ण *) src0_data + expert_id * (int64_t) nb02;

        int64_t m    = m0;
        int64_t बायाँ = run_len;

        यदि (use_x2) {
            जबसम्म (बायाँ >= 2) {
                अचर block_q8_0 * row0 = (अचर block_q8_0 *) (expert_base + m * (int64_t) nb01);
                अचर block_q8_0 * row1 = (अचर block_q8_0 *) (expert_base + (m + 1) * (int64_t) nb01);
                भिन्न              s0, s1;
                q8_dot_compute_x2_aligned(row0, row1, b_col_base, K_blocks, &s0, &s1);
                atomic_store_f32((चञ्चल भिन्न *) (dst_slot + m * (int64_t) nbd0), s0);
                atomic_store_f32((चञ्चल भिन्न *) (dst_slot + (m + 1) * (int64_t) nbd0), s1);
                m += 2;
                बायाँ -= 2;
            }
        }

        जबसम्म (बायाँ > 0) {
            अचर block_q8_0 * row = (अचर block_q8_0 *) (expert_base + m * (int64_t) nb01);
            भिन्न              s   = q8_dot_compute(row, b_col_base, K_blocks);
            atomic_store_f32((चञ्चल भिन्न *) (dst_slot + m * (int64_t) nbd0), s);
            m++;
            बायाँ--;
        }

        idx += (uint64_t) run_len;
    }

    q8_dot_end(&q8_state);
    फर्काऊ 0;
}
