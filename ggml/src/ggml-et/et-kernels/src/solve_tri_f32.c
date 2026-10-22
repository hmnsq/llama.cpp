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

#شامل "ggml_tensor.h"
#شامل "math_fp.h"
#شامل "platform.h"

#شامل <stdint.h>

ساخت ggml_et_solve_tri_params {
    ساخت ggml_tensor src0;  // A: lower-triangular [n, n, B1, B2]
    ساخت ggml_tensor src1;  // B: RHS [k, n, B1, B2]
    ساخت ggml_tensor dst;   // X: solution [k, n, B1, B2]
};

صحیح اندراج_نقطہ(ساخت ggml_et_solve_tri_params * params, خالی * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    اگر (!kernel_env) {
        واپس -1;
    }

    صحیح thread_id   = get_relative_thread_id(kernel_env->shire_نقاب);
    صحیح عدد_دھاگے = get_عدد_دھاگے(kernel_env->shire_نقاب);

    اگر (thread_id < 0) {
        واپس 0;
    }

    اگر (params == 0 || ((uint64_t) params & 0x7) != 0) {
        واپس -1;
    }

    ساخت ggml_tensor * src0 = &params->src0;  // A
    ساخت ggml_tensor * src1 = &params->src1;  // B
    ساخت ggml_tensor * dst  = &params->dst;   // X

    اگر (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        واپس -1;
    }

    ثابت اعشاری * A_ڈیٹا = (ثابت اعشاری *) src0->ڈیٹا;
    ثابت اعشاری * B_ڈیٹا = (ثابت اعشاری *) src1->ڈیٹا;
    اعشاری *       X_ڈیٹا = (اعشاری *) dst->ڈیٹا;

    اگر (!A_ڈیٹا || !B_ڈیٹا || !X_ڈیٹا) {
        واپس -1;
    }

    ثابت int64_t n   = src0->ne[1];  // A is n×n
    ثابت int64_t k   = src1->ne[0];  // number of RHS columns
    ثابت int64_t ne2 = src0->ne[2];
    ثابت int64_t ne3 = src0->ne[3];

    // Strides in bytes
    ثابت size_t nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    ثابت size_t nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];
    ثابت size_t nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];

    // k % 16 == 0 guaranteed by supports_op. Rows are cache-line aligned,
    // so column groups of 16 map to exclusive cache lines.
    // TODO: Vectorize the thing
    ثابت int64_t ستون_تعداد_فی_cl    = 16;
    ثابت int64_t عدد_ستون_groups = k / ستون_تعداد_فی_cl;
    ثابت int64_t کل_work     = عدد_ستون_groups * ne2 * ne3;

    برائے (int64_t work = thread_id; work < کل_work; work += عدد_دھاگے) {
        ثابت int64_t cg = work % عدد_ستون_groups;
        ثابت int64_t i2 = (work / عدد_ستون_groups) % ne2;
        ثابت int64_t i3 = work / (عدد_ستون_groups * ne2);

        ثابت int64_t j_آغاز = cg * ستون_تعداد_فی_cl;
        ثابت int64_t j_انتہا   = j_آغاز + ستون_تعداد_فی_cl;

        ثابت اعشاری * A_batch = (ثابت اعشاری *) ((ثابت حرف *) A_ڈیٹا + i2 * nb02 + i3 * nb03);
        ثابت اعشاری * B_batch = (ثابت اعشاری *) ((ثابت حرف *) B_ڈیٹا + i2 * nb12 + i3 * nb13);
        اعشاری *       X_batch = (اعشاری *) ((حرف *) X_ڈیٹا + i2 * nb2 + i3 * nb3);

        برائے (int64_t j = j_آغاز; j < j_انتہا; j++) {
            برائے (int64_t i = 0; i < n; i++) {
                ثابت اعشاری * A_صف = (ثابت اعشاری *) ((ثابت حرف *) A_batch + i * nb01);
                اعشاری *       X_صف = (اعشاری *) ((حرف *) X_batch + i * nb1);
                ثابت اعشاری * B_صف = (ثابت اعشاری *) ((ثابت حرف *) B_batch + i * nb11);

                اعشاری مجموع = 0.0f;
                برائے (int64_t t = 0; t < i; t++) {
                    ثابت اعشاری * X_t = (ثابت اعشاری *) ((ثابت حرف *) X_batch + t * nb1);
                    مجموع += A_صف[t] * X_t[j];
                }

                X_صف[j] = et_fdiv(B_صف[j] - مجموع, A_صف[i]);
            }
        }
    }

    واپس 0;
}
