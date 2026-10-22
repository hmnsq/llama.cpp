#شامل "ggml_tensor.h"
#شامل "math_fp.h"
#شامل "platform.h"

#شامل <stdint.h>

ساخت ggml_et_ssm_جائزہ_params {
    ساخت ggml_tensor src0;  // s:   [d_state, head_dim, n_head, n_seqs]
    ساخت ggml_tensor src1;  // x:   [head_dim, n_head, n_seq_tokens, n_seqs]
    ساخت ggml_tensor src2;  // dt:  [n_head, n_seq_tokens, n_seqs]
    ساخت ggml_tensor src3;  // A:   [d_state, n_head] or [1, n_head]
    ساخت ggml_tensor src4;  // B:   [d_state, n_group, n_seq_tokens, n_seqs]
    ساخت ggml_tensor src5;  // C:   [d_state, n_group, n_seq_tokens, n_seqs]
    ساخت ggml_tensor src6;  // ids: [n_seqs] i32
    ساخت ggml_tensor dst;   // packed [y, states]
    int32_t            K;
};

ساکن اندرونی اعشاری softplus_f32(اعشاری x) {
    واپس x <= 20.0f ? et_logf(1.0f + et_expf(x)) : x;
}

صحیح اندراج_نقطہ(ساخت ggml_et_ssm_جائزہ_params * params, خالی * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    اگر (!kernel_env) {
        واپس -1;
    }

    ثابت صحیح thread_id   = get_relative_thread_id(kernel_env->shire_نقاب);
    ثابت صحیح عدد_دھاگے = get_عدد_دھاگے(kernel_env->shire_نقاب);

    اگر (thread_id < 0) {
        واپس 0;
    }

    اگر (params == 0 || ((uint64_t) params & 0x7) != 0) {
        واپس -1;
    }

    ساخت ggml_tensor * src0 = &params->src0;
    ساخت ggml_tensor * src1 = &params->src1;
    ساخت ggml_tensor * src2 = &params->src2;
    ساخت ggml_tensor * src3 = &params->src3;
    ساخت ggml_tensor * src4 = &params->src4;
    ساخت ggml_tensor * src5 = &params->src5;
    ساخت ggml_tensor * src6 = &params->src6;
    ساخت ggml_tensor * dst  = &params->dst;

    اگر (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_F32 ||
        src3->type != GGML_TYPE_F32 || src4->type != GGML_TYPE_F32 || src5->type != GGML_TYPE_F32 ||
        src6->type != GGML_TYPE_I32 || dst->type != GGML_TYPE_F32) {
        واپس -1;
    }

    ثابت اعشاری *   s_ڈیٹا   = (ثابت اعشاری *) src0->ڈیٹا;
    ثابت اعشاری *   x_ڈیٹا   = (ثابت اعشاری *) src1->ڈیٹا;
    ثابت اعشاری *   dt_ڈیٹا  = (ثابت اعشاری *) src2->ڈیٹا;
    ثابت اعشاری *   A_ڈیٹا   = (ثابت اعشاری *) src3->ڈیٹا;
    ثابت اعشاری *   B_ڈیٹا   = (ثابت اعشاری *) src4->ڈیٹا;
    ثابت اعشاری *   C_ڈیٹا   = (ثابت اعشاری *) src5->ڈیٹا;
    ثابت int32_t * ids      = (ثابت int32_t *) src6->ڈیٹا;
    اعشاری *         dst_ڈیٹا = (اعشاری *) dst->ڈیٹا;

    اگر (!s_ڈیٹا || !x_ڈیٹا || !dt_ڈیٹا || !A_ڈیٹا || !B_ڈیٹا || !C_ڈیٹا || !ids || !dst_ڈیٹا) {
        واپس -1;
    }

    ثابت int64_t d_state      = src0->ne[0];
    ثابت int64_t سر_dim     = src0->ne[1];
    ثابت int64_t n_سر       = src1->ne[1];
    ثابت int64_t n_گروہ      = src4->ne[1];
    ثابت int64_t n_seq_علامات = src1->ne[2];
    ثابت int64_t n_seqs       = src1->ne[3];
    ثابت int64_t y_elems      = src1->ne[0] * src1->ne[1] * src1->ne[2] * src1->ne[3];
    ثابت int64_t K            = params->K;

    اگر (src0->nb[0] != حجم(اعشاری) || src1->nb[0] != حجم(اعشاری) || src2->nb[0] != حجم(اعشاری) ||
        src3->nb[0] != حجم(اعشاری) || src4->nb[0] != حجم(اعشاری) || src5->nb[0] != حجم(اعشاری) ||
        src6->nb[0] != حجم(int32_t) || dst->nb[0] != حجم(اعشاری)) {
        واپس -1;
    }

    اگر (K < 1 || n_گروہ <= 0 || n_سر % n_گروہ != 0) {
        واپس -1;
    }

    // Cache-line bundling on the dst output (1 dst float per (head, dim, token)).
    //   - When head_dim < 16: bundle 16/head_dim heads per work-unit (1 line of dst).
    //   - When head_dim >= 16: each head's dim slice spans head_dim/16 lines, so we
    //     can split dims into chunks of 16 across threads without false sharing.
    ثابت int64_t dst_lanes_فی_cl    = 16;
    ثابت int64_t heads_فی_cacheline = سر_dim >= dst_lanes_فی_cl ? 1 : (dst_lanes_فی_cl / سر_dim);
    ثابت int64_t heads_فی_بلاک     = heads_فی_cacheline > 0 ? heads_فی_cacheline : 1;
    ثابت int64_t بلاکیں_فی_seq      = (n_سر + heads_فی_بلاک - 1) / heads_فی_بلاک;
    ثابت int64_t dim_chunk_lanes     = سر_dim >= dst_lanes_فی_cl ? dst_lanes_فی_cl : سر_dim;
    ثابت int64_t dim_chunks_فی_سر = (سر_dim + dim_chunk_lanes - 1) / dim_chunk_lanes;

    // A "unit" = (seq, head_block, dim_chunk). This expands the parallelism by a
    // factor of dim_chunks_per_head over the prior block-only scheme; for Mamba-2
    // shapes (head_dim=64) that's a 4x bump in active threads.
    ثابت int64_t units_فی_seq    = بلاکیں_فی_seq * dim_chunks_فی_سر;
    ثابت int64_t کل_units      = n_seqs * units_فی_seq;
    ثابت int64_t units_فی_thread = (کل_units + عدد_دھاگے - 1) / عدد_دھاگے;
    ثابت int64_t unit_begin       = (int64_t) thread_id * units_فی_thread;
    int64_t       unit_انتہا         = unit_begin + units_فی_thread;

    اگر (unit_begin >= کل_units) {
        واپس 0;
    }

    اگر (unit_انتہا > کل_units) {
        unit_انتہا = کل_units;
    }

    ثابت صحیح     A_نشر = (src3->ne[0] == 1);
    ثابت int64_t d_state_vec = (d_state / 8) * 8;  // largest multiple of 8 <= d_state
    ثابت اعشاری   log2e_const = 1.4426950408889634f;

    برائے (int64_t unit = unit_begin; unit < unit_انتہا; ++unit) {
        ثابت int64_t seq_idx       = unit / units_فی_seq;
        ثابت int64_t unit_اندر_seq   = unit % units_فی_seq;
        ثابت int64_t بلاک_اندر_seq  = unit_اندر_seq / dim_chunks_فی_سر;
        ثابت int64_t dim_chunk_idx = unit_اندر_seq % dim_chunks_فی_سر;
        ثابت int64_t سر_begin    = بلاک_اندر_seq * heads_فی_بلاک;
        int64_t       سر_انتہا      = سر_begin + heads_فی_بلاک;

        اگر (سر_انتہا > n_سر) {
            سر_انتہا = n_سر;
        }

        ثابت int64_t dim_begin = dim_chunk_idx * dim_chunk_lanes;
        int64_t       dim_انتہا   = dim_begin + dim_chunk_lanes;
        اگر (dim_انتہا > سر_dim) {
            dim_انتہا = سر_dim;
        }

        ثابت int32_t state_seq = ids[seq_idx];

        برائے (int64_t سر_idx = سر_begin; سر_idx < سر_انتہا; ++سر_idx) {
            ثابت int64_t گروہ_idx = سر_idx / (n_سر / n_گروہ);

            // A pointer for this head: contiguous over state_idx when not broadcast
            ثابت اعشاری * A_صف = (ثابت اعشاری *) ((ثابت حرف *) A_ڈیٹا + (size_t) سر_idx * src3->nb[1]);

            برائے (int64_t dim_idx = dim_begin; dim_idx < dim_انتہا; ++dim_idx) {
                ثابت اعشاری * state_src =
                    (ثابت اعشاری *) ((ثابت حرف *) s_ڈیٹا + (size_t) dim_idx * src0->nb[1] +
                                     (size_t) سر_idx * src0->nb[2] + (size_t) state_seq * src0->nb[3]);

                اعشاری * state_dst =
                    (اعشاری *) ((حرف *) dst_ڈیٹا + (size_t) y_elems * حجم(اعشاری) + (size_t) dim_idx * src0->nb[1] +
                               (size_t) سر_idx * src0->nb[2] + (size_t) seq_idx * src0->nb[3]);

                برائے (int64_t token_idx = 0; token_idx < n_seq_علامات; ++token_idx) {
                    ثابت اعشاری * x_ptr =
                        (ثابت اعشاری *) ((ثابت حرف *) x_ڈیٹا + (size_t) dim_idx * src1->nb[0] +
                                         (size_t) سر_idx * src1->nb[1] + (size_t) token_idx * src1->nb[2] +
                                         (size_t) seq_idx * src1->nb[3]);

                    ثابت اعشاری * dt_ptr =
                        (ثابت اعشاری *) ((ثابت حرف *) dt_ڈیٹا + (size_t) سر_idx * src2->nb[0] +
                                         (size_t) token_idx * src2->nb[1] + (size_t) seq_idx * src2->nb[2]);

                    ثابت اعشاری * B_صف =
                        (ثابت اعشاری *) ((ثابت حرف *) B_ڈیٹا + (size_t) گروہ_idx * src4->nb[1] +
                                         (size_t) token_idx * src4->nb[2] + (size_t) seq_idx * src4->nb[3]);

                    ثابت اعشاری * C_صف =
                        (ثابت اعشاری *) ((ثابت حرف *) C_ڈیٹا + (size_t) گروہ_idx * src5->nb[1] +
                                         (size_t) token_idx * src5->nb[2] + (size_t) seq_idx * src5->nb[3]);

                    ثابت اعشاری dt_softplus = softplus_f32(*dt_ptr);
                    ثابت اعشاری x_dt        = (*x_ptr) * dt_softplus;
                    ثابت اعشاری dt_log2e    = dt_softplus * log2e_const;

                    // Source of "previous state" for this token: input state on token 0,
                    // last token's state thereafter (we wrote it into state_dst).
                    ثابت اعشاری * پچھلا_صف = (token_idx == 0) ? state_src : state_dst;

                    اعشاری   sumf      = 0.0f;
                    int64_t state_idx = 0;

                    اگر (d_state_vec > 0) {
                        // Save mask, enable all 8 vector lanes for the state loop.
                        غیرعلامتی طویل saved_نقاب;
                        __asm__ متلون("mova.x.m %0" : "=r"(saved_نقاب));
                        __asm__ متلون("mov.m.x m0, x0, 0xFF");

                        // Per-token broadcasts:
                        //   f20 = x_dt     (B*x_dt)
                        //   f21 = dt_log2e (for fexp.ps when A is per-state)
                        //   f22 = dA       (only when A is broadcast scalar)
                        //   f23 = sum-of-products accumulator (zeroed)
                        __asm__ متلون(
                            "fbc.ps f20, %[xdt]\n\t"
                            "fbc.ps f21, %[dtl]\n\t"
                            "fbci.pi f23, 0\n\t"
                            :
                            : [xdt] "m"(x_dt), [dtl] "m"(dt_log2e)
                            : "f20", "f21", "f23");

                        اگر (A_نشر) {
                            // dA is a per-head scalar — compute once and splat.
                            ثابت اعشاری dA_scalar = et_expf(dt_softplus * (*A_صف));
                            __asm__ متلون("fbc.ps f22, %[da]\n\t" : : [da] "m"(dA_scalar) : "f22");
                        }

                        برائے (; state_idx < d_state_vec; state_idx += 8) {
                            اگر (!A_نشر) {
                                // f22 = exp(dt_softplus * A[state..state+7])
                                //     = 2^((dt_softplus * A) * log2e) via fexp.ps
                                __asm__ متلون(
                                    "flw.ps  f24, %[av]\n\t"
                                    "fmul.ps f24, f24, f21\n\t"  // A * dt_log2e
                                    "fexp.ps f22, f24\n\t"       // dA = 2^(...)
                                    :
                                    : [av] "m"(*(ثابت اعشاری (*)[8]) & A_صف[state_idx])
                                    : "f22", "f24");
                            }

                            // state = prev * dA + B * x_dt
                            // sumf += state * C
                            // Reads prev before writing state_dst — safe even when
                            // prev_row == state_dst (write-after-read, same index).
                            __asm__ متلون(
                                "flw.ps  f25, %[prev]\n\t"
                                "flw.ps  f26, %[bv]\n\t"
                                "flw.ps  f27, %[cv]\n\t"
                                "fmul.ps f26, f26, f20\n\t"        // B * x_dt
                                "fmadd.ps f25, f25, f22, f26\n\t"  // state = prev*dA + B*x_dt
                                "fsw.ps  f25, %[sd]\n\t"
                                "fmadd.ps f23, f25, f27, f23\n\t"  // sum += state*C
                                : [sd] "=m"(*(اعشاری (*)[8]) & state_dst[state_idx])
                                : [پچھلا] "m"(*(ثابت اعشاری (*)[8]) & پچھلا_صف[state_idx]),
                                  [bv] "m"(*(ثابت اعشاری (*)[8]) & B_صف[state_idx]),
                                  [cv] "m"(*(ثابت اعشاری (*)[8]) & C_صف[state_idx])
                                : "f25", "f26", "f27");
                        }

                        // Horizontal reduce f23 (8 lanes) -> scalar sumf.
                        __asm__ متلون(
                            "fswizz.ps f1, f23, 0xB1\n\t"
                            "fadd.ps   f2, f23, f1, rne\n\t"
                            "fswizz.ps f3, f2, 0x4E\n\t"
                            "fadd.ps   f4, f2, f3, rne\n\t"
                            "fmvz.x.ps t0, f4, 4\n\t"
                            "fbcx.ps   f5, t0\n\t"
                            "fadd.ps   %[vout], f4, f5, rne\n\t"
                            : [vout] "=f"(sumf)::"t0", "f1", "f2", "f3", "f4", "f5");

                        __asm__ متلون("mova.m.x %0" ::"r"(saved_نقاب));
                    }

                    // Scalar tail (d_state not a multiple of 8).
                    برائے (; state_idx < d_state; ++state_idx) {
                        ثابت اعشاری پچھلا_state = پچھلا_صف[state_idx];
                        ثابت اعشاری A_val      = A_نشر ? *A_صف : A_صف[state_idx];
                        ثابت اعشاری dA         = et_expf(dt_softplus * A_val);
                        ثابت اعشاری st         = پچھلا_state * dA + B_صف[state_idx] * x_dt;
                        state_dst[state_idx]   = st;
                        sumf += st * C_صف[state_idx];
                    }

                    ثابت int64_t خانہ = n_seq_علامات - 1 - token_idx;
                    اگر (خانہ > 0 && خانہ < K) {
                        اعشاری * state_snapshot =
                            (اعشاری *) ((حرف *) state_dst + (size_t) خانہ * n_seqs * src0->nb[3]);
                        برائے (int64_t i = 0; i < d_state; ++i) {
                            state_snapshot[i] = state_dst[i];
                        }
                    }

                    dst_ڈیٹا[seq_idx * (n_seq_علامات * n_سر * سر_dim) + token_idx * (n_سر * سر_dim) +
                             سر_idx * سر_dim + dim_idx] = sumf;
                }
            }
        }
    }

    واپس 0;
}
