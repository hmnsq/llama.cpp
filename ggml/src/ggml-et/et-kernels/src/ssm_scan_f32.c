#ضمن "ggml_tensor.h"
#ضمن "math_fp.h"
#ضمن "platform.h"

#ضمن <stdint.h>

بنية ggml_et_ssm_پیمایش_params {
    بنية ggml_tensor src0;  // s:   [d_state, head_dim, n_head, n_seqs]
    بنية ggml_tensor src1;  // x:   [head_dim, n_head, n_seq_tokens, n_seqs]
    بنية ggml_tensor src2;  // dt:  [n_head, n_seq_tokens, n_seqs]
    بنية ggml_tensor src3;  // A:   [d_state, n_head] or [1, n_head]
    بنية ggml_tensor src4;  // B:   [d_state, n_group, n_seq_tokens, n_seqs]
    بنية ggml_tensor src5;  // C:   [d_state, n_group, n_seq_tokens, n_seqs]
    بنية ggml_tensor src6;  // ids: [n_seqs] i32
    بنية ggml_tensor dst;   // packed [y, states]
    int32_t            K;
};

ساكن مضمن عائم softplus_f32(عائم x) {
    أرجع x <= 20.0f ? et_logf(1.0f + et_expf(x)) : x;
}

صحيح مدخل_نقطه(بنية ggml_et_ssm_پیمایش_params * params, فراغ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    إذا (!kernel_env) {
        أرجع -1;
    }

    ثابت صحيح thread_id   = get_relative_thread_id(kernel_env->shire_ماسک);
    ثابت صحيح رقم_نخها = get_رقم_نخها(kernel_env->shire_ماسک);

    إذا (thread_id < 0) {
        أرجع 0;
    }

    إذا (params == 0 || ((uint64_t) params & 0x7) != 0) {
        أرجع -1;
    }

    بنية ggml_tensor * src0 = &params->src0;
    بنية ggml_tensor * src1 = &params->src1;
    بنية ggml_tensor * src2 = &params->src2;
    بنية ggml_tensor * src3 = &params->src3;
    بنية ggml_tensor * src4 = &params->src4;
    بنية ggml_tensor * src5 = &params->src5;
    بنية ggml_tensor * src6 = &params->src6;
    بنية ggml_tensor * dst  = &params->dst;

    إذا (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_F32 ||
        src3->type != GGML_TYPE_F32 || src4->type != GGML_TYPE_F32 || src5->type != GGML_TYPE_F32 ||
        src6->type != GGML_TYPE_I32 || dst->type != GGML_TYPE_F32) {
        أرجع -1;
    }

    ثابت عائم *   s_داده   = (ثابت عائم *) src0->داده;
    ثابت عائم *   x_داده   = (ثابت عائم *) src1->داده;
    ثابت عائم *   dt_داده  = (ثابت عائم *) src2->داده;
    ثابت عائم *   A_داده   = (ثابت عائم *) src3->داده;
    ثابت عائم *   B_داده   = (ثابت عائم *) src4->داده;
    ثابت عائم *   C_داده   = (ثابت عائم *) src5->داده;
    ثابت int32_t * ids      = (ثابت int32_t *) src6->داده;
    عائم *         dst_داده = (عائم *) dst->داده;

    إذا (!s_داده || !x_داده || !dt_داده || !A_داده || !B_داده || !C_داده || !ids || !dst_داده) {
        أرجع -1;
    }

    ثابت int64_t d_state      = src0->ne[0];
    ثابت int64_t سر_dim     = src0->ne[1];
    ثابت int64_t n_سر       = src1->ne[1];
    ثابت int64_t n_گروه      = src4->ne[1];
    ثابت int64_t n_seq_نشانهها = src1->ne[2];
    ثابت int64_t n_seqs       = src1->ne[3];
    ثابت int64_t y_elems      = src1->ne[0] * src1->ne[1] * src1->ne[2] * src1->ne[3];
    ثابت int64_t K            = params->K;

    إذا (src0->nb[0] != حجم(عائم) || src1->nb[0] != حجم(عائم) || src2->nb[0] != حجم(عائم) ||
        src3->nb[0] != حجم(عائم) || src4->nb[0] != حجم(عائم) || src5->nb[0] != حجم(عائم) ||
        src6->nb[0] != حجم(int32_t) || dst->nb[0] != حجم(عائم)) {
        أرجع -1;
    }

    إذا (K < 1 || n_گروه <= 0 || n_سر % n_گروه != 0) {
        أرجع -1;
    }

    // Cache-line bundling on the dst output (1 dst float per (head, dim, token)).
    //   - When head_dim < 16: bundle 16/head_dim heads per work-unit (1 line of dst).
    //   - When head_dim >= 16: each head's dim slice spans head_dim/16 lines, so we
    //     can split dims into chunks of 16 across threads without false sharing.
    ثابت int64_t dst_lanes_در_هر_cl    = 16;
    ثابت int64_t heads_در_هر_cacheline = سر_dim >= dst_lanes_در_هر_cl ? 1 : (dst_lanes_در_هر_cl / سر_dim);
    ثابت int64_t heads_در_هر_بلوک     = heads_در_هر_cacheline > 0 ? heads_در_هر_cacheline : 1;
    ثابت int64_t بلوکها_در_هر_seq      = (n_سر + heads_در_هر_بلوک - 1) / heads_در_هر_بلوک;
    ثابت int64_t dim_chunk_lanes     = سر_dim >= dst_lanes_در_هر_cl ? dst_lanes_در_هر_cl : سر_dim;
    ثابت int64_t dim_chunks_در_هر_سر = (سر_dim + dim_chunk_lanes - 1) / dim_chunk_lanes;

    // A "unit" = (seq, head_block, dim_chunk). This expands the parallelism by a
    // factor of dim_chunks_per_head over the prior block-only scheme; for Mamba-2
    // shapes (head_dim=64) that's a 4x bump in active threads.
    ثابت int64_t units_در_هر_seq    = بلوکها_در_هر_seq * dim_chunks_در_هر_سر;
    ثابت int64_t مجموع_units      = n_seqs * units_در_هر_seq;
    ثابت int64_t units_در_هر_thread = (مجموع_units + رقم_نخها - 1) / رقم_نخها;
    ثابت int64_t unit_begin       = (int64_t) thread_id * units_در_هر_thread;
    int64_t       unit_انتها         = unit_begin + units_در_هر_thread;

    إذا (unit_begin >= مجموع_units) {
        أرجع 0;
    }

    إذا (unit_انتها > مجموع_units) {
        unit_انتها = مجموع_units;
    }

    ثابت صحيح     A_پخش = (src3->ne[0] == 1);
    ثابت int64_t d_state_vec = (d_state / 8) * 8;  // largest multiple of 8 <= d_state
    ثابت عائم   log2e_const = 1.4426950408889634f;

    لكل (int64_t unit = unit_begin; unit < unit_انتها; ++unit) {
        ثابت int64_t seq_idx       = unit / units_در_هر_seq;
        ثابت int64_t unit_درون_seq   = unit % units_در_هر_seq;
        ثابت int64_t بلوک_درون_seq  = unit_درون_seq / dim_chunks_در_هر_سر;
        ثابت int64_t dim_chunk_idx = unit_درون_seq % dim_chunks_در_هر_سر;
        ثابت int64_t سر_begin    = بلوک_درون_seq * heads_در_هر_بلوک;
        int64_t       سر_انتها      = سر_begin + heads_در_هر_بلوک;

        إذا (سر_انتها > n_سر) {
            سر_انتها = n_سر;
        }

        ثابت int64_t dim_begin = dim_chunk_idx * dim_chunk_lanes;
        int64_t       dim_انتها   = dim_begin + dim_chunk_lanes;
        إذا (dim_انتها > سر_dim) {
            dim_انتها = سر_dim;
        }

        ثابت int32_t state_seq = ids[seq_idx];

        لكل (int64_t سر_idx = سر_begin; سر_idx < سر_انتها; ++سر_idx) {
            ثابت int64_t گروه_idx = سر_idx / (n_سر / n_گروه);

            // A pointer for this head: contiguous over state_idx when not broadcast
            ثابت عائم * A_ردیف = (ثابت عائم *) ((ثابت حرف *) A_داده + (size_t) سر_idx * src3->nb[1]);

            لكل (int64_t dim_idx = dim_begin; dim_idx < dim_انتها; ++dim_idx) {
                ثابت عائم * state_src =
                    (ثابت عائم *) ((ثابت حرف *) s_داده + (size_t) dim_idx * src0->nb[1] +
                                     (size_t) سر_idx * src0->nb[2] + (size_t) state_seq * src0->nb[3]);

                عائم * state_dst =
                    (عائم *) ((حرف *) dst_داده + (size_t) y_elems * حجم(عائم) + (size_t) dim_idx * src0->nb[1] +
                               (size_t) سر_idx * src0->nb[2] + (size_t) seq_idx * src0->nb[3]);

                لكل (int64_t token_idx = 0; token_idx < n_seq_نشانهها; ++token_idx) {
                    ثابت عائم * x_ptr =
                        (ثابت عائم *) ((ثابت حرف *) x_داده + (size_t) dim_idx * src1->nb[0] +
                                         (size_t) سر_idx * src1->nb[1] + (size_t) token_idx * src1->nb[2] +
                                         (size_t) seq_idx * src1->nb[3]);

                    ثابت عائم * dt_ptr =
                        (ثابت عائم *) ((ثابت حرف *) dt_داده + (size_t) سر_idx * src2->nb[0] +
                                         (size_t) token_idx * src2->nb[1] + (size_t) seq_idx * src2->nb[2]);

                    ثابت عائم * B_ردیف =
                        (ثابت عائم *) ((ثابت حرف *) B_داده + (size_t) گروه_idx * src4->nb[1] +
                                         (size_t) token_idx * src4->nb[2] + (size_t) seq_idx * src4->nb[3]);

                    ثابت عائم * C_ردیف =
                        (ثابت عائم *) ((ثابت حرف *) C_داده + (size_t) گروه_idx * src5->nb[1] +
                                         (size_t) token_idx * src5->nb[2] + (size_t) seq_idx * src5->nb[3]);

                    ثابت عائم dt_softplus = softplus_f32(*dt_ptr);
                    ثابت عائم x_dt        = (*x_ptr) * dt_softplus;
                    ثابت عائم dt_log2e    = dt_softplus * log2e_const;

                    // Source of "previous state" for this token: input state on token 0,
                    // last token's state thereafter (we wrote it into state_dst).
                    ثابت عائم * prev_ردیف = (token_idx == 0) ? state_src : state_dst;

                    عائم   sumf      = 0.0f;
                    int64_t state_idx = 0;

                    إذا (d_state_vec > 0) {
                        // Save mask, enable all 8 vector lanes for the state loop.
                        بیعلامت دراز saved_ماسک;
                        __asm__ متطاير("mova.x.m %0" : "=r"(saved_ماسک));
                        __asm__ متطاير("mov.m.x m0, x0, 0xFF");

                        // Per-token broadcasts:
                        //   f20 = x_dt     (B*x_dt)
                        //   f21 = dt_log2e (for fexp.ps when A is per-state)
                        //   f22 = dA       (only when A is broadcast scalar)
                        //   f23 = sum-of-products accumulator (zeroed)
                        __asm__ متطاير(
                            "fbc.ps f20, %[xdt]\n\t"
                            "fbc.ps f21, %[dtl]\n\t"
                            "fbci.pi f23, 0\n\t"
                            :
                            : [xdt] "m"(x_dt), [dtl] "m"(dt_log2e)
                            : "f20", "f21", "f23");

                        إذا (A_پخش) {
                            // dA is a per-head scalar — compute once and splat.
                            ثابت عائم dA_scalar = et_expf(dt_softplus * (*A_ردیف));
                            __asm__ متطاير("fbc.ps f22, %[da]\n\t" : : [da] "m"(dA_scalar) : "f22");
                        }

                        لكل (; state_idx < d_state_vec; state_idx += 8) {
                            إذا (!A_پخش) {
                                // f22 = exp(dt_softplus * A[state..state+7])
                                //     = 2^((dt_softplus * A) * log2e) via fexp.ps
                                __asm__ متطاير(
                                    "flw.ps  f24, %[av]\n\t"
                                    "fmul.ps f24, f24, f21\n\t"  // A * dt_log2e
                                    "fexp.ps f22, f24\n\t"       // dA = 2^(...)
                                    :
                                    : [av] "m"(*(ثابت عائم (*)[8]) & A_ردیف[state_idx])
                                    : "f22", "f24");
                            }

                            // state = prev * dA + B * x_dt
                            // sumf += state * C
                            // Reads prev before writing state_dst — safe even when
                            // prev_row == state_dst (write-after-read, same index).
                            __asm__ متطاير(
                                "flw.ps  f25, %[prev]\n\t"
                                "flw.ps  f26, %[bv]\n\t"
                                "flw.ps  f27, %[cv]\n\t"
                                "fmul.ps f26, f26, f20\n\t"        // B * x_dt
                                "fmadd.ps f25, f25, f22, f26\n\t"  // state = prev*dA + B*x_dt
                                "fsw.ps  f25, %[sd]\n\t"
                                "fmadd.ps f23, f25, f27, f23\n\t"  // sum += state*C
                                : [sd] "=m"(*(عائم (*)[8]) & state_dst[state_idx])
                                : [prev] "m"(*(ثابت عائم (*)[8]) & prev_ردیف[state_idx]),
                                  [bv] "m"(*(ثابت عائم (*)[8]) & B_ردیف[state_idx]),
                                  [cv] "m"(*(ثابت عائم (*)[8]) & C_ردیف[state_idx])
                                : "f25", "f26", "f27");
                        }

                        // Horizontal reduce f23 (8 lanes) -> scalar sumf.
                        __asm__ متطاير(
                            "fswizz.ps f1, f23, 0xB1\n\t"
                            "fadd.ps   f2, f23, f1, rne\n\t"
                            "fswizz.ps f3, f2, 0x4E\n\t"
                            "fadd.ps   f4, f2, f3, rne\n\t"
                            "fmvz.x.ps t0, f4, 4\n\t"
                            "fbcx.ps   f5, t0\n\t"
                            "fadd.ps   %[vout], f4, f5, rne\n\t"
                            : [vout] "=f"(sumf)::"t0", "f1", "f2", "f3", "f4", "f5");

                        __asm__ متطاير("mova.m.x %0" ::"r"(saved_ماسک));
                    }

                    // Scalar tail (d_state not a multiple of 8).
                    لكل (; state_idx < d_state; ++state_idx) {
                        ثابت عائم prev_state = prev_ردیف[state_idx];
                        ثابت عائم A_val      = A_پخش ? *A_ردیف : A_ردیف[state_idx];
                        ثابت عائم dA         = et_expf(dt_softplus * A_val);
                        ثابت عائم st         = prev_state * dA + B_ردیف[state_idx] * x_dt;
                        state_dst[state_idx]   = st;
                        sumf += st * C_ردیف[state_idx];
                    }

                    ثابت int64_t جایگاه = n_seq_نشانهها - 1 - token_idx;
                    إذا (جایگاه > 0 && جایگاه < K) {
                        عائم * state_snapshot =
                            (عائم *) ((حرف *) state_dst + (size_t) جایگاه * n_seqs * src0->nb[3]);
                        لكل (int64_t i = 0; i < d_state; ++i) {
                            state_snapshot[i] = state_dst[i];
                        }
                    }

                    dst_داده[seq_idx * (n_seq_نشانهها * n_سر * سر_dim) + token_idx * (n_سر * سر_dim) +
                             سر_idx * سر_dim + dim_idx] = sumf;
                }
            }
        }
    }

    أرجع 0;
}
