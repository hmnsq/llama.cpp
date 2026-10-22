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

पूर्णाङ्क प्रविष्टिः_point(संरचना ggml_et_mul_mat_id_params * params, शून्य * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;
    यदि (!kernel_env || !params) {
        प्रत्यागम -1;
    }

    पूर्णाङ्क thread_id   = get_relative_thread_id(kernel_env->shire_आवरणम्);
    पूर्णाङ्क अङ्कः_तन्तवः = get_अङ्कः_तन्तवः(kernel_env->shire_आवरणम्);
    यदि (thread_id < 0) {
        प्रत्यागम 0;
    }

    संरचना ggml_tensor * src0 = &params->src0;
    संरचना ggml_tensor * src1 = &params->src1;
    संरचना ggml_tensor * src2 = &params->src2;
    संरचना ggml_tensor * dst  = &params->dst;

    यदि (src0->type != GGML_TYPE_Q8_0 || src1->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32 ||
        dst->type != GGML_TYPE_F32) {
        प्रत्यागम -1;
    }

    ध्रुव शून्य *    src0_दत्त = src0->दत्त;
    ध्रुव भिन्न *   src1_दत्त = (ध्रुव भिन्न *) src1->दत्त;
    ध्रुव int32_t * src2_दत्त = (ध्रुव int32_t *) src2->दत्त;
    भिन्न *         dst_दत्त  = (भिन्न *) dst->दत्त;
    यदि (!src0_दत्त || !src1_दत्त || !src2_दत्त || !dst_दत्त) {
        प्रत्यागम -1;
    }

    ध्रुव int64_t K             = src0->ne[0];
    ध्रुव int64_t M             = src0->ne[1];
    ध्रुव int64_t n_expert      = src0->ne[2];
    ध्रुव int64_t n_expert_प्रयुक्तम् = src2->ne[0];
    ध्रुव int64_t batch         = src2->ne[1];
    ध्रुव int64_t ne11          = src1->ne[1];

    यदि (K % QK8_0 != 0) {
        प्रत्यागम -1;
    }

    ध्रुव size_t nb01 = src0->nb[1];
    ध्रुव size_t nb02 = src0->nb[2];
    ध्रुव size_t nb11 = src1->nb[1];
    ध्रुव size_t nb12 = src1->nb[2];
    ध्रुव size_t nb20 = src2->nb[0];
    ध्रुव size_t nb21 = src2->nb[1];
    ध्रुव size_t nbd0 = dst->nb[0];
    ध्रुव size_t nbd1 = dst->nb[1];
    ध्रुव size_t nbd2 = dst->nb[2];

    यदि (src0->nb[0] != परिमाण(खण्डम्_q8_0) || src1->nb[0] != परिमाण(भिन्न) || src2->nb[0] != परिमाण(int32_t) ||
        nbd0 != परिमाण(भिन्न)) {
        प्रत्यागम -1;
    }

    ध्रुव int64_t K_खण्डानि = K / QK8_0;
    ध्रुव पूर्णाङ्क     use_x2   = ((nb01 & 31) == 0);

    ध्रुव uint64_t योग_outputs = (uint64_t) M * (uint64_t) n_expert_प्रयुक्तम् * (uint64_t) batch;
    यदि (योग_outputs == 0) {
        प्रत्यागम 0;
    }

    ध्रुव uint64_t chunk    = (योग_outputs + (uint64_t) अङ्कः_तन्तवः - 1) / (uint64_t) अङ्कः_तन्तवः;
    ध्रुव uint64_t my_start = (uint64_t) thread_id * chunk;
    यदि (my_start >= योग_outputs) {
        प्रत्यागम 0;
    }
    uint64_t my_अन्तः = my_start + chunk;
    यदि (my_अन्तः > योग_outputs) {
        my_अन्तः = योग_outputs;
    }

    q8_dot_state q8_state;
    q8_dot_begin(&q8_state);

    ध्रुव uint64_t प्रत्येकम्_batch = (uint64_t) M * (uint64_t) n_expert_प्रयुक्तम्;

    uint64_t idx = my_start;
    यावत् (idx < my_अन्तः) {
        ध्रुव int64_t  batch_idx = (int64_t) (idx / प्रत्येकम्_batch);
        ध्रुव uint64_t rem       = idx - (uint64_t) batch_idx * प्रत्येकम्_batch;
        ध्रुव int64_t  slot_idx  = (int64_t) (rem / (uint64_t) M);
        ध्रुव int64_t  m0        = (int64_t) (rem - (uint64_t) slot_idx * (uint64_t) M);

        ध्रुव uint64_t सततम्_अन्तः_global =
            (uint64_t) batch_idx * प्रत्येकम्_batch + (uint64_t) slot_idx * (uint64_t) M + (uint64_t) M;
        ध्रुव uint64_t अन्तः_अन्तर्_my = (सततम्_अन्तः_global < my_अन्तः) ? सततम्_अन्तः_global : my_अन्तः;
        int64_t        सततम्_len   = (int64_t) (अन्तः_अन्तर्_my - idx);

        ध्रुव int32_t expert_id =
            *(ध्रुव int32_t *) ((ध्रुव वर्ण *) src2_दत्त + slot_idx * (int64_t) nb20 + batch_idx * (int64_t) nb21);

        वर्ण * dst_slot = (वर्ण *) dst_दत्त + slot_idx * (int64_t) nbd1 + batch_idx * (int64_t) nbd2;

        यदि (expert_id < 0 || expert_id >= n_expert) {
            int64_t m = m0;
            प्रति (int64_t i = 0; i < सततम्_len; i++, m++) {
                atomic_store_f32((चञ्चल भिन्न *) (dst_slot + m * (int64_t) nbd0), 0.0f);
            }
            idx += (uint64_t) सततम्_len;
            अनुवर्त;
        }

        ध्रुव int64_t col_idx = slot_idx % ne11;
        ध्रुव भिन्न * b_col_आधारः =
            (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src1_दत्त + col_idx * (int64_t) nb11 + batch_idx * (int64_t) nb12);
        ध्रुव वर्ण * expert_आधारः = (ध्रुव वर्ण *) src0_दत्त + expert_id * (int64_t) nb02;

        int64_t m    = m0;
        int64_t वामः = सततम्_len;

        यदि (use_x2) {
            यावत् (वामः >= 2) {
                ध्रुव खण्डम्_q8_0 * row0 = (ध्रुव खण्डम्_q8_0 *) (expert_आधारः + m * (int64_t) nb01);
                ध्रुव खण्डम्_q8_0 * row1 = (ध्रुव खण्डम्_q8_0 *) (expert_आधारः + (m + 1) * (int64_t) nb01);
                भिन्न              s0, s1;
                q8_dot_compute_x2_aligned(row0, row1, b_col_आधारः, K_खण्डानि, &s0, &s1);
                atomic_store_f32((चञ्चल भिन्न *) (dst_slot + m * (int64_t) nbd0), s0);
                atomic_store_f32((चञ्चल भिन्न *) (dst_slot + (m + 1) * (int64_t) nbd0), s1);
                m += 2;
                वामः -= 2;
            }
        }

        यावत् (वामः > 0) {
            ध्रुव खण्डम्_q8_0 * row = (ध्रुव खण्डम्_q8_0 *) (expert_आधारः + m * (int64_t) nb01);
            भिन्न              s   = q8_dot_compute(row, b_col_आधारः, K_खण्डानि);
            atomic_store_f32((चञ्चल भिन्न *) (dst_slot + m * (int64_t) nbd0), s);
            m++;
            वामः--;
        }

        idx += (uint64_t) सततम्_len;
    }

    q8_dot_अन्तः(&q8_state);
    प्रत्यागम 0;
}
