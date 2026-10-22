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

#అన్తర్భావ "block_ops.h"
#అన్తర్భావ "ggml_tensor.h"
#అన్తర్భావ "math_fp.h"
#అన్తర్భావ "platform.h"
#అన్తర్భావ "quants.h"

#అన్తర్భావ <stdint.h>

పూర్ణాంకం నమోదు_బిందువు(నిర్మాణం ggml_et_mul_mat_id_params * params, శూన్య * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;
    ఒకవేళ (!kernel_env || !params) {
        తిరిగి -1;
    }

    పూర్ణాంకం thread_id   = get_relative_thread_id(kernel_env->shire_ముసుగు);
    పూర్ణాంకం అంకె_దారాలు = get_అంకె_దారాలు(kernel_env->shire_ముసుగు);
    ఒకవేళ (thread_id < 0) {
        తిరిగి 0;
    }

    నిర్మాణం ggml_tensor * src0 = &params->src0;
    నిర్మాణం ggml_tensor * src1 = &params->src1;
    నిర్మాణం ggml_tensor * src2 = &params->src2;
    నిర్మాణం ggml_tensor * dst  = &params->dst;

    ఒకవేళ (src0->type != GGML_TYPE_Q8_0 || src1->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32 ||
        dst->type != GGML_TYPE_F32) {
        తిరిగి -1;
    }

    స్థిరరాశి శూన్య *    src0_దత్తాంశం = src0->దత్తాంశం;
    స్థిరరాశి దశాంశం *   src1_దత్తాంశం = (స్థిరరాశి దశాంశం *) src1->దత్తాంశం;
    స్థిరరాశి int32_t * src2_దత్తాంశం = (స్థిరరాశి int32_t *) src2->దత్తాంశం;
    దశాంశం *         dst_దత్తాంశం  = (దశాంశం *) dst->దత్తాంశం;
    ఒకవేళ (!src0_దత్తాంశం || !src1_దత్తాంశం || !src2_దత్తాంశం || !dst_దత్తాంశం) {
        తిరిగి -1;
    }

    స్థిరరాశి int64_t K             = src0->ne[0];
    స్థిరరాశి int64_t M             = src0->ne[1];
    స్థిరరాశి int64_t n_expert      = src0->ne[2];
    స్థిరరాశి int64_t n_expert_వాడిన = src2->ne[0];
    స్థిరరాశి int64_t batch         = src2->ne[1];
    స్థిరరాశి int64_t ne11          = src1->ne[1];

    ఒకవేళ (K % QK8_0 != 0) {
        తిరిగి -1;
    }

    స్థిరరాశి size_t nb01 = src0->nb[1];
    స్థిరరాశి size_t nb02 = src0->nb[2];
    స్థిరరాశి size_t nb11 = src1->nb[1];
    స్థిరరాశి size_t nb12 = src1->nb[2];
    స్థిరరాశి size_t nb20 = src2->nb[0];
    స్థిరరాశి size_t nb21 = src2->nb[1];
    స్థిరరాశి size_t nbd0 = dst->nb[0];
    స్థిరరాశి size_t nbd1 = dst->nb[1];
    స్థిరరాశి size_t nbd2 = dst->nb[2];

    ఒకవేళ (src0->nb[0] != పరిమాణం(ఖండం_q8_0) || src1->nb[0] != పరిమాణం(దశాంశం) || src2->nb[0] != పరిమాణం(int32_t) ||
        nbd0 != పరిమాణం(దశాంశం)) {
        తిరిగి -1;
    }

    స్థిరరాశి int64_t K_ఖండాలు = K / QK8_0;
    స్థిరరాశి పూర్ణాంకం     use_x2   = ((nb01 & 31) == 0);

    స్థిరరాశి uint64_t మొత్తం_outputs = (uint64_t) M * (uint64_t) n_expert_వాడిన * (uint64_t) batch;
    ఒకవేళ (మొత్తం_outputs == 0) {
        తిరిగి 0;
    }

    స్థిరరాశి uint64_t chunk    = (మొత్తం_outputs + (uint64_t) అంకె_దారాలు - 1) / (uint64_t) అంకె_దారాలు;
    స్థిరరాశి uint64_t my_ప్రారంభం = (uint64_t) thread_id * chunk;
    ఒకవేళ (my_ప్రారంభం >= మొత్తం_outputs) {
        తిరిగి 0;
    }
    uint64_t my_చివర = my_ప్రారంభం + chunk;
    ఒకవేళ (my_చివర > మొత్తం_outputs) {
        my_చివర = మొత్తం_outputs;
    }

    q8_dot_state q8_state;
    q8_dot_begin(&q8_state);

    స్థిరరాశి uint64_t చొప్పున_batch = (uint64_t) M * (uint64_t) n_expert_వాడిన;

    uint64_t idx = my_ప్రారంభం;
    అయితే (idx < my_చివర) {
        స్థిరరాశి int64_t  batch_idx = (int64_t) (idx / చొప్పున_batch);
        స్థిరరాశి uint64_t rem       = idx - (uint64_t) batch_idx * చొప్పున_batch;
        స్థిరరాశి int64_t  స్థానం_idx  = (int64_t) (rem / (uint64_t) M);
        స్థిరరాశి int64_t  m0        = (int64_t) (rem - (uint64_t) స్థానం_idx * (uint64_t) M);

        స్థిరరాశి uint64_t నిరంతరం_చివర_global =
            (uint64_t) batch_idx * చొప్పున_batch + (uint64_t) స్థానం_idx * (uint64_t) M + (uint64_t) M;
        స్థిరరాశి uint64_t చివర_లోపల_my = (నిరంతరం_చివర_global < my_చివర) ? నిరంతరం_చివర_global : my_చివర;
        int64_t        నిరంతరం_len   = (int64_t) (చివర_లోపల_my - idx);

        స్థిరరాశి int32_t expert_id =
            *(స్థిరరాశి int32_t *) ((స్థిరరాశి అక్షరం *) src2_దత్తాంశం + స్థానం_idx * (int64_t) nb20 + batch_idx * (int64_t) nb21);

        అక్షరం * dst_స్థానం = (అక్షరం *) dst_దత్తాంశం + స్థానం_idx * (int64_t) nbd1 + batch_idx * (int64_t) nbd2;

        ఒకవేళ (expert_id < 0 || expert_id >= n_expert) {
            int64_t m = m0;
            కొరకు (int64_t i = 0; i < నిరంతరం_len; i++, m++) {
                atomic_store_f32((అస్థిర దశాంశం *) (dst_స్థానం + m * (int64_t) nbd0), 0.0f);
            }
            idx += (uint64_t) నిరంతరం_len;
            కొనసాగు;
        }

        స్థిరరాశి int64_t నిలువు_idx = స్థానం_idx % ne11;
        స్థిరరాశి దశాంశం * b_నిలువు_పునాది =
            (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) src1_దత్తాంశం + నిలువు_idx * (int64_t) nb11 + batch_idx * (int64_t) nb12);
        స్థిరరాశి అక్షరం * expert_పునాది = (స్థిరరాశి అక్షరం *) src0_దత్తాంశం + expert_id * (int64_t) nb02;

        int64_t m    = m0;
        int64_t ఎడమ_2 = నిరంతరం_len;

        ఒకవేళ (use_x2) {
            అయితే (ఎడమ_2 >= 2) {
                స్థిరరాశి ఖండం_q8_0 * row0 = (స్థిరరాశి ఖండం_q8_0 *) (expert_పునాది + m * (int64_t) nb01);
                స్థిరరాశి ఖండం_q8_0 * row1 = (స్థిరరాశి ఖండం_q8_0 *) (expert_పునాది + (m + 1) * (int64_t) nb01);
                దశాంశం              s0, s1;
                q8_dot_compute_x2_aligned(row0, row1, b_నిలువు_పునాది, K_ఖండాలు, &s0, &s1);
                atomic_store_f32((అస్థిర దశాంశం *) (dst_స్థానం + m * (int64_t) nbd0), s0);
                atomic_store_f32((అస్థిర దశాంశం *) (dst_స్థానం + (m + 1) * (int64_t) nbd0), s1);
                m += 2;
                ఎడమ_2 -= 2;
            }
        }

        అయితే (ఎడమ_2 > 0) {
            స్థిరరాశి ఖండం_q8_0 * అడ్డువరుస = (స్థిరరాశి ఖండం_q8_0 *) (expert_పునాది + m * (int64_t) nb01);
            దశాంశం              s   = q8_dot_compute(అడ్డువరుస, b_నిలువు_పునాది, K_ఖండాలు);
            atomic_store_f32((అస్థిర దశాంశం *) (dst_స్థానం + m * (int64_t) nbd0), s);
            m++;
            ఎడమ_2--;
        }

        idx += (uint64_t) నిరంతరం_len;
    }

    q8_dot_చివర(&q8_state);
    తిరిగి 0;
}
