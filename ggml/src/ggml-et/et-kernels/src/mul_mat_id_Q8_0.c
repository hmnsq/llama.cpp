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

#ਅਨ੍ਤਰ੍ਭਾਵ "block_ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml_tensor.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "math_fp.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "platform.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "quants.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>

ਪੂਰਨ ਇੰਦਰਾਜ_ਬਿੰਦੂ(ਬਣਤਰ ggml_et_mul_mat_id_params * params, ਖਾਲੀ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;
    ਜੇ (!kernel_env || !params) {
        ਵਾਪਸ -1;
    }

    ਪੂਰਨ thread_id   = get_relative_thread_id(kernel_env->shire_ਨਕਾਬ);
    ਪੂਰਨ ਅੰਕ_ਧਾਗੇ = get_ਅੰਕ_ਧਾਗੇ(kernel_env->shire_ਨਕਾਬ);
    ਜੇ (thread_id < 0) {
        ਵਾਪਸ 0;
    }

    ਬਣਤਰ ggml_tensor * src0 = &params->src0;
    ਬਣਤਰ ggml_tensor * src1 = &params->src1;
    ਬਣਤਰ ggml_tensor * src2 = &params->src2;
    ਬਣਤਰ ggml_tensor * dst  = &params->dst;

    ਜੇ (src0->type != GGML_TYPE_Q8_0 || src1->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32 ||
        dst->type != GGML_TYPE_F32) {
        ਵਾਪਸ -1;
    }

    ਅਚਲ ਖਾਲੀ *    src0_ਅੰਕੜੇ = src0->ਅੰਕੜੇ;
    ਅਚਲ ਦਸ਼ਮਲਵ *   src1_ਅੰਕੜੇ = (ਅਚਲ ਦਸ਼ਮਲਵ *) src1->ਅੰਕੜੇ;
    ਅਚਲ int32_t * src2_ਅੰਕੜੇ = (ਅਚਲ int32_t *) src2->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ *         dst_ਅੰਕੜੇ  = (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ;
    ਜੇ (!src0_ਅੰਕੜੇ || !src1_ਅੰਕੜੇ || !src2_ਅੰਕੜੇ || !dst_ਅੰਕੜੇ) {
        ਵਾਪਸ -1;
    }

    ਅਚਲ int64_t K             = src0->ne[0];
    ਅਚਲ int64_t M             = src0->ne[1];
    ਅਚਲ int64_t n_expert      = src0->ne[2];
    ਅਚਲ int64_t n_expert_ਵਰਤਿਆ = src2->ne[0];
    ਅਚਲ int64_t batch         = src2->ne[1];
    ਅਚਲ int64_t ne11          = src1->ne[1];

    ਜੇ (K % QK8_0 != 0) {
        ਵਾਪਸ -1;
    }

    ਅਚਲ size_t nb01 = src0->nb[1];
    ਅਚਲ size_t nb02 = src0->nb[2];
    ਅਚਲ size_t nb11 = src1->nb[1];
    ਅਚਲ size_t nb12 = src1->nb[2];
    ਅਚਲ size_t nb20 = src2->nb[0];
    ਅਚਲ size_t nb21 = src2->nb[1];
    ਅਚਲ size_t nbd0 = dst->nb[0];
    ਅਚਲ size_t nbd1 = dst->nb[1];
    ਅਚਲ size_t nbd2 = dst->nb[2];

    ਜੇ (src0->nb[0] != ਆਕਾਰ(ਖੰਡ_q8_0) || src1->nb[0] != ਆਕਾਰ(ਦਸ਼ਮਲਵ) || src2->nb[0] != ਆਕਾਰ(int32_t) ||
        nbd0 != ਆਕਾਰ(ਦਸ਼ਮਲਵ)) {
        ਵਾਪਸ -1;
    }

    ਅਚਲ int64_t K_ਖੰਡਾਂ = K / QK8_0;
    ਅਚਲ ਪੂਰਨ     use_x2   = ((nb01 & 31) == 0);

    ਅਚਲ uint64_t ਕੁੱਲ_outputs = (uint64_t) M * (uint64_t) n_expert_ਵਰਤਿਆ * (uint64_t) batch;
    ਜੇ (ਕੁੱਲ_outputs == 0) {
        ਵਾਪਸ 0;
    }

    ਅਚਲ uint64_t chunk    = (ਕੁੱਲ_outputs + (uint64_t) ਅੰਕ_ਧਾਗੇ - 1) / (uint64_t) ਅੰਕ_ਧਾਗੇ;
    ਅਚਲ uint64_t my_ਸ਼ੁਰੂ = (uint64_t) thread_id * chunk;
    ਜੇ (my_ਸ਼ੁਰੂ >= ਕੁੱਲ_outputs) {
        ਵਾਪਸ 0;
    }
    uint64_t my_ਸਿਰਾ = my_ਸ਼ੁਰੂ + chunk;
    ਜੇ (my_ਸਿਰਾ > ਕੁੱਲ_outputs) {
        my_ਸਿਰਾ = ਕੁੱਲ_outputs;
    }

    q8_dot_state q8_state;
    q8_dot_begin(&q8_state);

    ਅਚਲ uint64_t ਪ੍ਰਤੀ_batch = (uint64_t) M * (uint64_t) n_expert_ਵਰਤਿਆ;

    uint64_t idx = my_ਸ਼ੁਰੂ;
    ਜਦੋਂ (idx < my_ਸਿਰਾ) {
        ਅਚਲ int64_t  batch_idx = (int64_t) (idx / ਪ੍ਰਤੀ_batch);
        ਅਚਲ uint64_t rem       = idx - (uint64_t) batch_idx * ਪ੍ਰਤੀ_batch;
        ਅਚਲ int64_t  ਥਾਂ_idx  = (int64_t) (rem / (uint64_t) M);
        ਅਚਲ int64_t  m0        = (int64_t) (rem - (uint64_t) ਥਾਂ_idx * (uint64_t) M);

        ਅਚਲ uint64_t ਲੜੀਵਾਰ_ਸਿਰਾ_global =
            (uint64_t) batch_idx * ਪ੍ਰਤੀ_batch + (uint64_t) ਥਾਂ_idx * (uint64_t) M + (uint64_t) M;
        ਅਚਲ uint64_t ਸਿਰਾ_ਵਿੱਚ_my = (ਲੜੀਵਾਰ_ਸਿਰਾ_global < my_ਸਿਰਾ) ? ਲੜੀਵਾਰ_ਸਿਰਾ_global : my_ਸਿਰਾ;
        int64_t        ਲੜੀਵਾਰ_len   = (int64_t) (ਸਿਰਾ_ਵਿੱਚ_my - idx);

        ਅਚਲ int32_t expert_id =
            *(ਅਚਲ int32_t *) ((ਅਚਲ ਅੱਖਰ *) src2_ਅੰਕੜੇ + ਥਾਂ_idx * (int64_t) nb20 + batch_idx * (int64_t) nb21);

        ਅੱਖਰ * dst_ਥਾਂ = (ਅੱਖਰ *) dst_ਅੰਕੜੇ + ਥਾਂ_idx * (int64_t) nbd1 + batch_idx * (int64_t) nbd2;

        ਜੇ (expert_id < 0 || expert_id >= n_expert) {
            int64_t m = m0;
            ਲਈ (int64_t i = 0; i < ਲੜੀਵਾਰ_len; i++, m++) {
                atomic_store_f32((ਅਸਥਿਰ ਦਸ਼ਮਲਵ *) (dst_ਥਾਂ + m * (int64_t) nbd0), 0.0f);
            }
            idx += (uint64_t) ਲੜੀਵਾਰ_len;
            ਜਾਰੀ;
        }

        ਅਚਲ int64_t ਖਾਨਾ_idx = ਥਾਂ_idx % ne11;
        ਅਚਲ ਦਸ਼ਮਲਵ * b_ਖਾਨਾ_ਅਧਾਰ =
            (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) src1_ਅੰਕੜੇ + ਖਾਨਾ_idx * (int64_t) nb11 + batch_idx * (int64_t) nb12);
        ਅਚਲ ਅੱਖਰ * expert_ਅਧਾਰ = (ਅਚਲ ਅੱਖਰ *) src0_ਅੰਕੜੇ + expert_id * (int64_t) nb02;

        int64_t m    = m0;
        int64_t ਖੱਬਾ = ਲੜੀਵਾਰ_len;

        ਜੇ (use_x2) {
            ਜਦੋਂ (ਖੱਬਾ >= 2) {
                ਅਚਲ ਖੰਡ_q8_0 * row0 = (ਅਚਲ ਖੰਡ_q8_0 *) (expert_ਅਧਾਰ + m * (int64_t) nb01);
                ਅਚਲ ਖੰਡ_q8_0 * row1 = (ਅਚਲ ਖੰਡ_q8_0 *) (expert_ਅਧਾਰ + (m + 1) * (int64_t) nb01);
                ਦਸ਼ਮਲਵ              s0, s1;
                q8_dot_compute_x2_aligned(row0, row1, b_ਖਾਨਾ_ਅਧਾਰ, K_ਖੰਡਾਂ, &s0, &s1);
                atomic_store_f32((ਅਸਥਿਰ ਦਸ਼ਮਲਵ *) (dst_ਥਾਂ + m * (int64_t) nbd0), s0);
                atomic_store_f32((ਅਸਥਿਰ ਦਸ਼ਮਲਵ *) (dst_ਥਾਂ + (m + 1) * (int64_t) nbd0), s1);
                m += 2;
                ਖੱਬਾ -= 2;
            }
        }

        ਜਦੋਂ (ਖੱਬਾ > 0) {
            ਅਚਲ ਖੰਡ_q8_0 * ਪੰਕਤੀ = (ਅਚਲ ਖੰਡ_q8_0 *) (expert_ਅਧਾਰ + m * (int64_t) nb01);
            ਦਸ਼ਮਲਵ              s   = q8_dot_compute(ਪੰਕਤੀ, b_ਖਾਨਾ_ਅਧਾਰ, K_ਖੰਡਾਂ);
            atomic_store_f32((ਅਸਥਿਰ ਦਸ਼ਮਲਵ *) (dst_ਥਾਂ + m * (int64_t) nbd0), s);
            m++;
            ਖੱਬਾ--;
        }

        idx += (uint64_t) ਲੜੀਵਾਰ_len;
    }

    q8_dot_ਸਿਰਾ(&q8_state);
    ਵਾਪਸ 0;
}
