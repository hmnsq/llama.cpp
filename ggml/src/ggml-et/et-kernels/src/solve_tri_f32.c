//******************************************************************************
// Solve Triangular F32 Kernel
// Forward substitution: solve AX = B where A is lower-triangular.
//
// src0 (A): [n, n, B1, B2]  lower-triangular matrix
// src1 (B): [k, n, B1, B2]  right-hand side
// dst  (X): [k, n, B1, B2]  solution
//
// For each column j (parallelized across threads):
//   For i = 0..n-1:
//     X[i,j] = (B[i,j] - dot(A[i,0..i-1], X[0..i-1,j])) / A[i,i]
//
// Lower-triangular, left-side, non-unit variant implemented.
//******************************************************************************

#అన్తర్భావ "ggml_tensor.h"
#అన్తర్భావ "math_fp.h"
#అన్తర్భావ "platform.h"

#అన్తర్భావ <stdint.h>

నిర్మాణం ggml_et_solve_tri_params {
    నిర్మాణం ggml_tensor src0;  // A: lower-triangular [n, n, B1, B2]
    నిర్మాణం ggml_tensor src1;  // B: RHS [k, n, B1, B2]
    నిర్మాణం ggml_tensor dst;   // X: solution [k, n, B1, B2]
};

పూర్ణాంకం నమోదు_బిందువు(నిర్మాణం ggml_et_solve_tri_params * params, శూన్య * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ఒకవేళ (!kernel_env) {
        తిరిగి -1;
    }

    పూర్ణాంకం thread_id   = get_relative_thread_id(kernel_env->shire_ముసుగు);
    పూర్ణాంకం అంకె_దారాలు = get_అంకె_దారాలు(kernel_env->shire_ముసుగు);

    ఒకవేళ (thread_id < 0) {
        తిరిగి 0;
    }

    ఒకవేళ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        తిరిగి -1;
    }

    నిర్మాణం ggml_tensor * src0 = &params->src0;  // A
    నిర్మాణం ggml_tensor * src1 = &params->src1;  // B
    నిర్మాణం ggml_tensor * dst  = &params->dst;   // X

    ఒకవేళ (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        తిరిగి -1;
    }

    స్థిరరాశి దశాంశం * A_దత్తాంశం = (స్థిరరాశి దశాంశం *) src0->దత్తాంశం;
    స్థిరరాశి దశాంశం * B_దత్తాంశం = (స్థిరరాశి దశాంశం *) src1->దత్తాంశం;
    దశాంశం *       X_దత్తాంశం = (దశాంశం *) dst->దత్తాంశం;

    ఒకవేళ (!A_దత్తాంశం || !B_దత్తాంశం || !X_దత్తాంశం) {
        తిరిగి -1;
    }

    స్థిరరాశి int64_t n   = src0->ne[1];  // A is n×n
    స్థిరరాశి int64_t k   = src1->ne[0];  // number of RHS columns
    స్థిరరాశి int64_t ne2 = src0->ne[2];
    స్థిరరాశి int64_t ne3 = src0->ne[3];

    // Strides in bytes
    స్థిరరాశి size_t nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    స్థిరరాశి size_t nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];
    స్థిరరాశి size_t nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];

    // k % 16 == 0 guaranteed by supports_op. Rows are cache-line aligned,
    // so column groups of 16 map to exclusive cache lines.
    // TODO: Vectorize the thing
    స్థిరరాశి int64_t నిలువుల_సంఖ్య_చొప్పున_cl    = 16;
    స్థిరరాశి int64_t అంకె_నిలువు_groups = k / నిలువుల_సంఖ్య_చొప్పున_cl;
    స్థిరరాశి int64_t మొత్తం_work     = అంకె_నిలువు_groups * ne2 * ne3;

    కొరకు (int64_t work = thread_id; work < మొత్తం_work; work += అంకె_దారాలు) {
        స్థిరరాశి int64_t cg = work % అంకె_నిలువు_groups;
        స్థిరరాశి int64_t i2 = (work / అంకె_నిలువు_groups) % ne2;
        స్థిరరాశి int64_t i3 = work / (అంకె_నిలువు_groups * ne2);

        స్థిరరాశి int64_t j_ప్రారంభం = cg * నిలువుల_సంఖ్య_చొప్పున_cl;
        స్థిరరాశి int64_t j_చివర   = j_ప్రారంభం + నిలువుల_సంఖ్య_చొప్పున_cl;

        స్థిరరాశి దశాంశం * A_batch = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) A_దత్తాంశం + i2 * nb02 + i3 * nb03);
        స్థిరరాశి దశాంశం * B_batch = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) B_దత్తాంశం + i2 * nb12 + i3 * nb13);
        దశాంశం *       X_batch = (దశాంశం *) ((అక్షరం *) X_దత్తాంశం + i2 * nb2 + i3 * nb3);

        కొరకు (int64_t j = j_ప్రారంభం; j < j_చివర; j++) {
            కొరకు (int64_t i = 0; i < n; i++) {
                స్థిరరాశి దశాంశం * A_అడ్డువరుస = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) A_batch + i * nb01);
                దశాంశం *       X_అడ్డువరుస = (దశాంశం *) ((అక్షరం *) X_batch + i * nb1);
                స్థిరరాశి దశాంశం * B_అడ్డువరుస = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) B_batch + i * nb11);

                దశాంశం కూడిక = 0.0f;
                కొరకు (int64_t t = 0; t < i; t++) {
                    స్థిరరాశి దశాంశం * X_t = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) X_batch + t * nb1);
                    కూడిక += A_అడ్డువరుస[t] * X_t[j];
                }

                X_అడ్డువరుస[j] = et_fdiv(B_అడ్డువరుస[j] - కూడిక, A_అడ్డువరుస[i]);
            }
        }
    }

    తిరిగి 0;
}
