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

पूर्णांक entry_point(संरचना ggml_et_mul_mat_id_params * params, शून्य * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;
    जर (!kernel_env || !params) {
        परत -1;
    }

    पूर्णांक thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    पूर्णांक अंक_threads = get_अंक_threads(kernel_env->shire_mask);
    जर (thread_id < 0) {
        परत 0;
    }

    संरचना ggml_tensor * src0 = &params->src0;
    संरचना ggml_tensor * src1 = &params->src1;
    संरचना ggml_tensor * src2 = &params->src2;
    संरचना ggml_tensor * dst  = &params->dst;

    जर (src0->type != GGML_TYPE_Q8_0 || src1->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32 ||
        dst->type != GGML_TYPE_F32) {
        परत -1;
    }

    स्थिरांक शून्य *    src0_data = src0->data;
    स्थिरांक दशांश *   src1_data = (स्थिरांक दशांश *) src1->data;
    स्थिरांक int32_t * src2_data = (स्थिरांक int32_t *) src2->data;
    दशांश *         dst_data  = (दशांश *) dst->data;
    जर (!src0_data || !src1_data || !src2_data || !dst_data) {
        परत -1;
    }

    स्थिरांक int64_t K             = src0->ne[0];
    स्थिरांक int64_t M             = src0->ne[1];
    स्थिरांक int64_t n_expert      = src0->ne[2];
    स्थिरांक int64_t n_expert_used = src2->ne[0];
    स्थिरांक int64_t batch         = src2->ne[1];
    स्थिरांक int64_t ne11          = src1->ne[1];

    जर (K % QK8_0 != 0) {
        परत -1;
    }

    स्थिरांक माप_प्रकार nb01 = src0->nb[1];
    स्थिरांक माप_प्रकार nb02 = src0->nb[2];
    स्थिरांक माप_प्रकार nb11 = src1->nb[1];
    स्थिरांक माप_प्रकार nb12 = src1->nb[2];
    स्थिरांक माप_प्रकार nb20 = src2->nb[0];
    स्थिरांक माप_प्रकार nb21 = src2->nb[1];
    स्थिरांक माप_प्रकार nbd0 = dst->nb[0];
    स्थिरांक माप_प्रकार nbd1 = dst->nb[1];
    स्थिरांक माप_प्रकार nbd2 = dst->nb[2];

    जर (src0->nb[0] != आकार(block_q8_0) || src1->nb[0] != आकार(दशांश) || src2->nb[0] != आकार(int32_t) ||
        nbd0 != आकार(दशांश)) {
        परत -1;
    }

    स्थिरांक int64_t K_blocks = K / QK8_0;
    स्थिरांक पूर्णांक     use_x2   = ((nb01 & 31) == 0);

    स्थिरांक uint64_t total_outputs = (uint64_t) M * (uint64_t) n_expert_used * (uint64_t) batch;
    जर (total_outputs == 0) {
        परत 0;
    }

    स्थिरांक uint64_t chunk    = (total_outputs + (uint64_t) अंक_threads - 1) / (uint64_t) अंक_threads;
    स्थिरांक uint64_t my_start = (uint64_t) thread_id * chunk;
    जर (my_start >= total_outputs) {
        परत 0;
    }
    uint64_t my_end = my_start + chunk;
    जर (my_end > total_outputs) {
        my_end = total_outputs;
    }

    q8_dot_state q8_state;
    q8_dot_begin(&q8_state);

    स्थिरांक uint64_t per_batch = (uint64_t) M * (uint64_t) n_expert_used;

    uint64_t idx = my_start;
    पर्यंत (idx < my_end) {
        स्थिरांक int64_t  batch_idx = (int64_t) (idx / per_batch);
        स्थिरांक uint64_t rem       = idx - (uint64_t) batch_idx * per_batch;
        स्थिरांक int64_t  slot_idx  = (int64_t) (rem / (uint64_t) M);
        स्थिरांक int64_t  m0        = (int64_t) (rem - (uint64_t) slot_idx * (uint64_t) M);

        स्थिरांक uint64_t run_end_global =
            (uint64_t) batch_idx * per_batch + (uint64_t) slot_idx * (uint64_t) M + (uint64_t) M;
        स्थिरांक uint64_t end_in_my = (run_end_global < my_end) ? run_end_global : my_end;
        int64_t        run_len   = (int64_t) (end_in_my - idx);

        स्थिरांक int32_t expert_id =
            *(स्थिरांक int32_t *) ((स्थिरांक अक्षर *) src2_data + slot_idx * (int64_t) nb20 + batch_idx * (int64_t) nb21);

        अक्षर * dst_slot = (अक्षर *) dst_data + slot_idx * (int64_t) nbd1 + batch_idx * (int64_t) nbd2;

        जर (expert_id < 0 || expert_id >= n_expert) {
            int64_t m = m0;
            साठी (int64_t i = 0; i < run_len; i++, m++) {
                atomic_store_f32((अस्थिर दशांश *) (dst_slot + m * (int64_t) nbd0), 0.0f);
            }
            idx += (uint64_t) run_len;
            सुरूठेवा;
        }

        स्थिरांक int64_t col_idx = slot_idx % ne11;
        स्थिरांक दशांश * b_col_base =
            (स्थिरांक दशांश *) ((स्थिरांक अक्षर *) src1_data + col_idx * (int64_t) nb11 + batch_idx * (int64_t) nb12);
        स्थिरांक अक्षर * expert_base = (स्थिरांक अक्षर *) src0_data + expert_id * (int64_t) nb02;

        int64_t m    = m0;
        int64_t डावा = run_len;

        जर (use_x2) {
            पर्यंत (डावा >= 2) {
                स्थिरांक block_q8_0 * row0 = (स्थिरांक block_q8_0 *) (expert_base + m * (int64_t) nb01);
                स्थिरांक block_q8_0 * row1 = (स्थिरांक block_q8_0 *) (expert_base + (m + 1) * (int64_t) nb01);
                दशांश              s0, s1;
                q8_dot_compute_x2_aligned(row0, row1, b_col_base, K_blocks, &s0, &s1);
                atomic_store_f32((अस्थिर दशांश *) (dst_slot + m * (int64_t) nbd0), s0);
                atomic_store_f32((अस्थिर दशांश *) (dst_slot + (m + 1) * (int64_t) nbd0), s1);
                m += 2;
                डावा -= 2;
            }
        }

        पर्यंत (डावा > 0) {
            स्थिरांक block_q8_0 * row = (स्थिरांक block_q8_0 *) (expert_base + m * (int64_t) nb01);
            दशांश              s   = q8_dot_compute(row, b_col_base, K_blocks);
            atomic_store_f32((अस्थिर दशांश *) (dst_slot + m * (int64_t) nbd0), s);
            m++;
            डावा--;
        }

        idx += (uint64_t) run_len;
    }

    q8_dot_end(&q8_state);
    परत 0;
}
