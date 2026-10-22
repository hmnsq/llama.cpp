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

#समावेश "block_ops.h"
#समावेश "ggml_tensor.h"
#समावेश "math_fp.h"
#समावेश "platform.h"
#समावेश "quants.h"

#समावेश <stdint.h>

पूर्णांक entry_point(काष्ठा ggml_et_mul_mat_id_params * params, व्योम * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;
    अगर (!kernel_env || !params) {
        वापस -1;
    }

    पूर्णांक thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    पूर्णांक अंक_threads = get_अंक_threads(kernel_env->shire_mask);
    अगर (thread_id < 0) {
        वापस 0;
    }

    काष्ठा ggml_tensor * src0 = &params->src0;
    काष्ठा ggml_tensor * src1 = &params->src1;
    काष्ठा ggml_tensor * src2 = &params->src2;
    काष्ठा ggml_tensor * dst  = &params->dst;

    अगर (src0->type != GGML_TYPE_Q8_0 || src1->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32 ||
        dst->type != GGML_TYPE_F32) {
        वापस -1;
    }

    स्थिर व्योम *    src0_तथ्य = src0->तथ्य;
    स्थिर भग्न *   src1_तथ्य = (स्थिर भग्न *) src1->तथ्य;
    स्थिर int32_t * src2_तथ्य = (स्थिर int32_t *) src2->तथ्य;
    भग्न *         dst_तथ्य  = (भग्न *) dst->तथ्य;
    अगर (!src0_तथ्य || !src1_तथ्य || !src2_तथ्य || !dst_तथ्य) {
        वापस -1;
    }

    स्थिर int64_t K             = src0->ne[0];
    स्थिर int64_t M             = src0->ne[1];
    स्थिर int64_t n_expert      = src0->ne[2];
    स्थिर int64_t n_expert_used = src2->ne[0];
    स्थिर int64_t batch         = src2->ne[1];
    स्थिर int64_t ne11          = src1->ne[1];

    अगर (K % QK8_0 != 0) {
        वापस -1;
    }

    स्थिर माप_प्रकार nb01 = src0->nb[1];
    स्थिर माप_प्रकार nb02 = src0->nb[2];
    स्थिर माप_प्रकार nb11 = src1->nb[1];
    स्थिर माप_प्रकार nb12 = src1->nb[2];
    स्थिर माप_प्रकार nb20 = src2->nb[0];
    स्थिर माप_प्रकार nb21 = src2->nb[1];
    स्थिर माप_प्रकार nbd0 = dst->nb[0];
    स्थिर माप_प्रकार nbd1 = dst->nb[1];
    स्थिर माप_प्रकार nbd2 = dst->nb[2];

    अगर (src0->nb[0] != माप(block_q8_0) || src1->nb[0] != माप(भग्न) || src2->nb[0] != माप(int32_t) ||
        nbd0 != माप(भग्न)) {
        वापस -1;
    }

    स्थिर int64_t K_blocks = K / QK8_0;
    स्थिर पूर्णांक     use_x2   = ((nb01 & 31) == 0);

    स्थिर uint64_t योग_outputs = (uint64_t) M * (uint64_t) n_expert_used * (uint64_t) batch;
    अगर (योग_outputs == 0) {
        वापस 0;
    }

    स्थिर uint64_t chunk    = (योग_outputs + (uint64_t) अंक_threads - 1) / (uint64_t) अंक_threads;
    स्थिर uint64_t my_start = (uint64_t) thread_id * chunk;
    अगर (my_start >= योग_outputs) {
        वापस 0;
    }
    uint64_t my_end = my_start + chunk;
    अगर (my_end > योग_outputs) {
        my_end = योग_outputs;
    }

    q8_dot_state q8_state;
    q8_dot_begin(&q8_state);

    स्थिर uint64_t per_batch = (uint64_t) M * (uint64_t) n_expert_used;

    uint64_t idx = my_start;
    जबतक (idx < my_end) {
        स्थिर int64_t  batch_idx = (int64_t) (idx / per_batch);
        स्थिर uint64_t rem       = idx - (uint64_t) batch_idx * per_batch;
        स्थिर int64_t  slot_idx  = (int64_t) (rem / (uint64_t) M);
        स्थिर int64_t  m0        = (int64_t) (rem - (uint64_t) slot_idx * (uint64_t) M);

        स्थिर uint64_t run_end_global =
            (uint64_t) batch_idx * per_batch + (uint64_t) slot_idx * (uint64_t) M + (uint64_t) M;
        स्थिर uint64_t end_in_my = (run_end_global < my_end) ? run_end_global : my_end;
        int64_t        run_len   = (int64_t) (end_in_my - idx);

        स्थिर int32_t expert_id =
            *(स्थिर int32_t *) ((स्थिर अक्षर *) src2_तथ्य + slot_idx * (int64_t) nb20 + batch_idx * (int64_t) nb21);

        अक्षर * dst_slot = (अक्षर *) dst_तथ्य + slot_idx * (int64_t) nbd1 + batch_idx * (int64_t) nbd2;

        अगर (expert_id < 0 || expert_id >= n_expert) {
            int64_t m = m0;
            क्रम (int64_t i = 0; i < run_len; i++, m++) {
                atomic_store_f32((अस्थिर भग्न *) (dst_slot + m * (int64_t) nbd0), 0.0f);
            }
            idx += (uint64_t) run_len;
            जारी;
        }

        स्थिर int64_t col_idx = slot_idx % ne11;
        स्थिर भग्न * b_col_base =
            (स्थिर भग्न *) ((स्थिर अक्षर *) src1_तथ्य + col_idx * (int64_t) nb11 + batch_idx * (int64_t) nb12);
        स्थिर अक्षर * expert_base = (स्थिर अक्षर *) src0_तथ्य + expert_id * (int64_t) nb02;

        int64_t m    = m0;
        int64_t बायाँ = run_len;

        अगर (use_x2) {
            जबतक (बायाँ >= 2) {
                स्थिर block_q8_0 * row0 = (स्थिर block_q8_0 *) (expert_base + m * (int64_t) nb01);
                स्थिर block_q8_0 * row1 = (स्थिर block_q8_0 *) (expert_base + (m + 1) * (int64_t) nb01);
                भग्न              s0, s1;
                q8_dot_compute_x2_aligned(row0, row1, b_col_base, K_blocks, &s0, &s1);
                atomic_store_f32((अस्थिर भग्न *) (dst_slot + m * (int64_t) nbd0), s0);
                atomic_store_f32((अस्थिर भग्न *) (dst_slot + (m + 1) * (int64_t) nbd0), s1);
                m += 2;
                बायाँ -= 2;
            }
        }

        जबतक (बायाँ > 0) {
            स्थिर block_q8_0 * row = (स्थिर block_q8_0 *) (expert_base + m * (int64_t) nb01);
            भग्न              s   = q8_dot_compute(row, b_col_base, K_blocks);
            atomic_store_f32((अस्थिर भग्न *) (dst_slot + m * (int64_t) nbd0), s);
            m++;
            बायाँ--;
        }

        idx += (uint64_t) run_len;
    }

    q8_dot_end(&q8_state);
    वापस 0;
}
