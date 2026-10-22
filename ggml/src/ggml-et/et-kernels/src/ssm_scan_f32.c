#అన్తర్భావ "ggml_tensor.h"
#అన్తర్భావ "math_fp.h"
#అన్తర్భావ "platform.h"

#అన్తర్భావ <stdint.h>

నిర్మాణం ggml_et_ssm_పరిశీలన_params {
    నిర్మాణం ggml_tensor src0;  // s:   [d_state, head_dim, n_head, n_seqs]
    నిర్మాణం ggml_tensor src1;  // x:   [head_dim, n_head, n_seq_tokens, n_seqs]
    నిర్మాణం ggml_tensor src2;  // dt:  [n_head, n_seq_tokens, n_seqs]
    నిర్మాణం ggml_tensor src3;  // A:   [d_state, n_head] or [1, n_head]
    నిర్మాణం ggml_tensor src4;  // B:   [d_state, n_group, n_seq_tokens, n_seqs]
    నిర్మాణం ggml_tensor src5;  // C:   [d_state, n_group, n_seq_tokens, n_seqs]
    నిర్మాణం ggml_tensor src6;  // ids: [n_seqs] i32
    నిర్మాణం ggml_tensor dst;   // packed [y, states]
    int32_t            K;
};

స్థిర అంతఃవరుస దశాంశం softplus_f32(దశాంశం x) {
    తిరిగి x <= 20.0f ? et_logf(1.0f + et_expf(x)) : x;
}

పూర్ణాంకం నమోదు_బిందువు(నిర్మాణం ggml_et_ssm_పరిశీలన_params * params, శూన్య * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ఒకవేళ (!kernel_env) {
        తిరిగి -1;
    }

    స్థిరరాశి పూర్ణాంకం thread_id   = get_relative_thread_id(kernel_env->shire_ముసుగు);
    స్థిరరాశి పూర్ణాంకం అంకె_దారాలు = get_అంకె_దారాలు(kernel_env->shire_ముసుగు);

    ఒకవేళ (thread_id < 0) {
        తిరిగి 0;
    }

    ఒకవేళ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        తిరిగి -1;
    }

    నిర్మాణం ggml_tensor * src0 = &params->src0;
    నిర్మాణం ggml_tensor * src1 = &params->src1;
    నిర్మాణం ggml_tensor * src2 = &params->src2;
    నిర్మాణం ggml_tensor * src3 = &params->src3;
    నిర్మాణం ggml_tensor * src4 = &params->src4;
    నిర్మాణం ggml_tensor * src5 = &params->src5;
    నిర్మాణం ggml_tensor * src6 = &params->src6;
    నిర్మాణం ggml_tensor * dst  = &params->dst;

    ఒకవేళ (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_F32 ||
        src3->type != GGML_TYPE_F32 || src4->type != GGML_TYPE_F32 || src5->type != GGML_TYPE_F32 ||
        src6->type != GGML_TYPE_I32 || dst->type != GGML_TYPE_F32) {
        తిరిగి -1;
    }

    స్థిరరాశి దశాంశం *   s_దత్తాంశం   = (స్థిరరాశి దశాంశం *) src0->దత్తాంశం;
    స్థిరరాశి దశాంశం *   x_దత్తాంశం   = (స్థిరరాశి దశాంశం *) src1->దత్తాంశం;
    స్థిరరాశి దశాంశం *   dt_దత్తాంశం  = (స్థిరరాశి దశాంశం *) src2->దత్తాంశం;
    స్థిరరాశి దశాంశం *   A_దత్తాంశం   = (స్థిరరాశి దశాంశం *) src3->దత్తాంశం;
    స్థిరరాశి దశాంశం *   B_దత్తాంశం   = (స్థిరరాశి దశాంశం *) src4->దత్తాంశం;
    స్థిరరాశి దశాంశం *   C_దత్తాంశం   = (స్థిరరాశి దశాంశం *) src5->దత్తాంశం;
    స్థిరరాశి int32_t * ids      = (స్థిరరాశి int32_t *) src6->దత్తాంశం;
    దశాంశం *         dst_దత్తాంశం = (దశాంశం *) dst->దత్తాంశం;

    ఒకవేళ (!s_దత్తాంశం || !x_దత్తాంశం || !dt_దత్తాంశం || !A_దత్తాంశం || !B_దత్తాంశం || !C_దత్తాంశం || !ids || !dst_దత్తాంశం) {
        తిరిగి -1;
    }

    స్థిరరాశి int64_t d_state      = src0->ne[0];
    స్థిరరాశి int64_t తల_dim     = src0->ne[1];
    స్థిరరాశి int64_t n_తల       = src1->ne[1];
    స్థిరరాశి int64_t n_సమూహం      = src4->ne[1];
    స్థిరరాశి int64_t n_seq_టోకెన్లు = src1->ne[2];
    స్థిరరాశి int64_t n_seqs       = src1->ne[3];
    స్థిరరాశి int64_t y_elems      = src1->ne[0] * src1->ne[1] * src1->ne[2] * src1->ne[3];
    స్థిరరాశి int64_t K            = params->K;

    ఒకవేళ (src0->nb[0] != పరిమాణం(దశాంశం) || src1->nb[0] != పరిమాణం(దశాంశం) || src2->nb[0] != పరిమాణం(దశాంశం) ||
        src3->nb[0] != పరిమాణం(దశాంశం) || src4->nb[0] != పరిమాణం(దశాంశం) || src5->nb[0] != పరిమాణం(దశాంశం) ||
        src6->nb[0] != పరిమాణం(int32_t) || dst->nb[0] != పరిమాణం(దశాంశం)) {
        తిరిగి -1;
    }

    ఒకవేళ (K < 1 || n_సమూహం <= 0 || n_తల % n_సమూహం != 0) {
        తిరిగి -1;
    }

    // Cache-line bundling on the dst output (1 dst float per (head, dim, token)).
    //   - When head_dim < 16: bundle 16/head_dim heads per work-unit (1 line of dst).
    //   - When head_dim >= 16: each head's dim slice spans head_dim/16 lines, so we
    //     can split dims into chunks of 16 across threads without false sharing.
    స్థిరరాశి int64_t dst_lanes_చొప్పున_cl    = 16;
    స్థిరరాశి int64_t heads_చొప్పున_cacheline = తల_dim >= dst_lanes_చొప్పున_cl ? 1 : (dst_lanes_చొప్పున_cl / తల_dim);
    స్థిరరాశి int64_t heads_చొప్పున_ఖండం     = heads_చొప్పున_cacheline > 0 ? heads_చొప్పున_cacheline : 1;
    స్థిరరాశి int64_t ఖండాలు_చొప్పున_seq      = (n_తల + heads_చొప్పున_ఖండం - 1) / heads_చొప్పున_ఖండం;
    స్థిరరాశి int64_t dim_chunk_lanes     = తల_dim >= dst_lanes_చొప్పున_cl ? dst_lanes_చొప్పున_cl : తల_dim;
    స్థిరరాశి int64_t dim_chunks_చొప్పున_తల = (తల_dim + dim_chunk_lanes - 1) / dim_chunk_lanes;

    // A "unit" = (seq, head_block, dim_chunk). This expands the parallelism by a
    // factor of dim_chunks_per_head over the prior block-only scheme; for Mamba-2
    // shapes (head_dim=64) that's a 4x bump in active threads.
    స్థిరరాశి int64_t units_చొప్పున_seq    = ఖండాలు_చొప్పున_seq * dim_chunks_చొప్పున_తల;
    స్థిరరాశి int64_t మొత్తం_units      = n_seqs * units_చొప్పున_seq;
    స్థిరరాశి int64_t units_చొప్పున_thread = (మొత్తం_units + అంకె_దారాలు - 1) / అంకె_దారాలు;
    స్థిరరాశి int64_t unit_begin       = (int64_t) thread_id * units_చొప్పున_thread;
    int64_t       unit_చివర         = unit_begin + units_చొప్పున_thread;

    ఒకవేళ (unit_begin >= మొత్తం_units) {
        తిరిగి 0;
    }

    ఒకవేళ (unit_చివర > మొత్తం_units) {
        unit_చివర = మొత్తం_units;
    }

    స్థిరరాశి పూర్ణాంకం     A_ప్రసారం = (src3->ne[0] == 1);
    స్థిరరాశి int64_t d_state_vec = (d_state / 8) * 8;  // largest multiple of 8 <= d_state
    స్థిరరాశి దశాంశం   log2e_const = 1.4426950408889634f;

    కొరకు (int64_t unit = unit_begin; unit < unit_చివర; ++unit) {
        స్థిరరాశి int64_t seq_idx       = unit / units_చొప్పున_seq;
        స్థిరరాశి int64_t unit_లోపల_seq   = unit % units_చొప్పున_seq;
        స్థిరరాశి int64_t ఖండం_లోపల_seq  = unit_లోపల_seq / dim_chunks_చొప్పున_తల;
        స్థిరరాశి int64_t dim_chunk_idx = unit_లోపల_seq % dim_chunks_చొప్పున_తల;
        స్థిరరాశి int64_t తల_begin    = ఖండం_లోపల_seq * heads_చొప్పున_ఖండం;
        int64_t       తల_చివర      = తల_begin + heads_చొప్పున_ఖండం;

        ఒకవేళ (తల_చివర > n_తల) {
            తల_చివర = n_తల;
        }

        స్థిరరాశి int64_t dim_begin = dim_chunk_idx * dim_chunk_lanes;
        int64_t       dim_చివర   = dim_begin + dim_chunk_lanes;
        ఒకవేళ (dim_చివర > తల_dim) {
            dim_చివర = తల_dim;
        }

        స్థిరరాశి int32_t state_seq = ids[seq_idx];

        కొరకు (int64_t తల_idx = తల_begin; తల_idx < తల_చివర; ++తల_idx) {
            స్థిరరాశి int64_t సమూహం_idx = తల_idx / (n_తల / n_సమూహం);

            // A pointer for this head: contiguous over state_idx when not broadcast
            స్థిరరాశి దశాంశం * A_అడ్డువరుస = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) A_దత్తాంశం + (size_t) తల_idx * src3->nb[1]);

            కొరకు (int64_t dim_idx = dim_begin; dim_idx < dim_చివర; ++dim_idx) {
                స్థిరరాశి దశాంశం * state_src =
                    (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) s_దత్తాంశం + (size_t) dim_idx * src0->nb[1] +
                                     (size_t) తల_idx * src0->nb[2] + (size_t) state_seq * src0->nb[3]);

                దశాంశం * state_dst =
                    (దశాంశం *) ((అక్షరం *) dst_దత్తాంశం + (size_t) y_elems * పరిమాణం(దశాంశం) + (size_t) dim_idx * src0->nb[1] +
                               (size_t) తల_idx * src0->nb[2] + (size_t) seq_idx * src0->nb[3]);

                కొరకు (int64_t token_idx = 0; token_idx < n_seq_టోకెన్లు; ++token_idx) {
                    స్థిరరాశి దశాంశం * x_ptr =
                        (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) x_దత్తాంశం + (size_t) dim_idx * src1->nb[0] +
                                         (size_t) తల_idx * src1->nb[1] + (size_t) token_idx * src1->nb[2] +
                                         (size_t) seq_idx * src1->nb[3]);

                    స్థిరరాశి దశాంశం * dt_ptr =
                        (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) dt_దత్తాంశం + (size_t) తల_idx * src2->nb[0] +
                                         (size_t) token_idx * src2->nb[1] + (size_t) seq_idx * src2->nb[2]);

                    స్థిరరాశి దశాంశం * B_అడ్డువరుస =
                        (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) B_దత్తాంశం + (size_t) సమూహం_idx * src4->nb[1] +
                                         (size_t) token_idx * src4->nb[2] + (size_t) seq_idx * src4->nb[3]);

                    స్థిరరాశి దశాంశం * C_అడ్డువరుస =
                        (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) C_దత్తాంశం + (size_t) సమూహం_idx * src5->nb[1] +
                                         (size_t) token_idx * src5->nb[2] + (size_t) seq_idx * src5->nb[3]);

                    స్థిరరాశి దశాంశం dt_softplus = softplus_f32(*dt_ptr);
                    స్థిరరాశి దశాంశం x_dt        = (*x_ptr) * dt_softplus;
                    స్థిరరాశి దశాంశం dt_log2e    = dt_softplus * log2e_const;

                    // Source of "previous state" for this token: input state on token 0,
                    // last token's state thereafter (we wrote it into state_dst).
                    స్థిరరాశి దశాంశం * prev_అడ్డువరుస = (token_idx == 0) ? state_src : state_dst;

                    దశాంశం   sumf      = 0.0f;
                    int64_t state_idx = 0;

                    ఒకవేళ (d_state_vec > 0) {
                        // Save mask, enable all 8 vector lanes for the state loop.
                        గుర్తులేని దీర్ఘ saved_ముసుగు;
                        __asm__ అస్థిర("mova.x.m %0" : "=r"(saved_ముసుగు));
                        __asm__ అస్థిర("mov.m.x m0, x0, 0xFF");

                        // Per-token broadcasts:
                        //   f20 = x_dt     (B*x_dt)
                        //   f21 = dt_log2e (for fexp.ps when A is per-state)
                        //   f22 = dA       (only when A is broadcast scalar)
                        //   f23 = sum-of-products accumulator (zeroed)
                        __asm__ అస్థిర(
                            "fbc.ps f20, %[xdt]\n\t"
                            "fbc.ps f21, %[dtl]\n\t"
                            "fbci.pi f23, 0\n\t"
                            :
                            : [xdt] "m"(x_dt), [dtl] "m"(dt_log2e)
                            : "f20", "f21", "f23");

                        ఒకవేళ (A_ప్రసారం) {
                            // dA is a per-head scalar — compute once and splat.
                            స్థిరరాశి దశాంశం dA_scalar = et_expf(dt_softplus * (*A_అడ్డువరుస));
                            __asm__ అస్థిర("fbc.ps f22, %[da]\n\t" : : [da] "m"(dA_scalar) : "f22");
                        }

                        కొరకు (; state_idx < d_state_vec; state_idx += 8) {
                            ఒకవేళ (!A_ప్రసారం) {
                                // f22 = exp(dt_softplus * A[state..state+7])
                                //     = 2^((dt_softplus * A) * log2e) via fexp.ps
                                __asm__ అస్థిర(
                                    "flw.ps  f24, %[av]\n\t"
                                    "fmul.ps f24, f24, f21\n\t"  // A * dt_log2e
                                    "fexp.ps f22, f24\n\t"       // dA = 2^(...)
                                    :
                                    : [av] "m"(*(స్థిరరాశి దశాంశం (*)[8]) & A_అడ్డువరుస[state_idx])
                                    : "f22", "f24");
                            }

                            // state = prev * dA + B * x_dt
                            // sumf += state * C
                            // Reads prev before writing state_dst — safe even when
                            // prev_row == state_dst (write-after-read, same index).
                            __asm__ అస్థిర(
                                "flw.ps  f25, %[prev]\n\t"
                                "flw.ps  f26, %[bv]\n\t"
                                "flw.ps  f27, %[cv]\n\t"
                                "fmul.ps f26, f26, f20\n\t"        // B * x_dt
                                "fmadd.ps f25, f25, f22, f26\n\t"  // state = prev*dA + B*x_dt
                                "fsw.ps  f25, %[sd]\n\t"
                                "fmadd.ps f23, f25, f27, f23\n\t"  // sum += state*C
                                : [sd] "=m"(*(దశాంశం (*)[8]) & state_dst[state_idx])
                                : [prev] "m"(*(స్థిరరాశి దశాంశం (*)[8]) & prev_అడ్డువరుస[state_idx]),
                                  [bv] "m"(*(స్థిరరాశి దశాంశం (*)[8]) & B_అడ్డువరుస[state_idx]),
                                  [cv] "m"(*(స్థిరరాశి దశాంశం (*)[8]) & C_అడ్డువరుస[state_idx])
                                : "f25", "f26", "f27");
                        }

                        // Horizontal reduce f23 (8 lanes) -> scalar sumf.
                        __asm__ అస్థిర(
                            "fswizz.ps f1, f23, 0xB1\n\t"
                            "fadd.ps   f2, f23, f1, rne\n\t"
                            "fswizz.ps f3, f2, 0x4E\n\t"
                            "fadd.ps   f4, f2, f3, rne\n\t"
                            "fmvz.x.ps t0, f4, 4\n\t"
                            "fbcx.ps   f5, t0\n\t"
                            "fadd.ps   %[vout], f4, f5, rne\n\t"
                            : [vout] "=f"(sumf)::"t0", "f1", "f2", "f3", "f4", "f5");

                        __asm__ అస్థిర("mova.m.x %0" ::"r"(saved_ముసుగు));
                    }

                    // Scalar tail (d_state not a multiple of 8).
                    కొరకు (; state_idx < d_state; ++state_idx) {
                        స్థిరరాశి దశాంశం prev_state = prev_అడ్డువరుస[state_idx];
                        స్థిరరాశి దశాంశం A_val      = A_ప్రసారం ? *A_అడ్డువరుస : A_అడ్డువరుస[state_idx];
                        స్థిరరాశి దశాంశం dA         = et_expf(dt_softplus * A_val);
                        స్థిరరాశి దశాంశం st         = prev_state * dA + B_అడ్డువరుస[state_idx] * x_dt;
                        state_dst[state_idx]   = st;
                        sumf += st * C_అడ్డువరుస[state_idx];
                    }

                    స్థిరరాశి int64_t స్థానం = n_seq_టోకెన్లు - 1 - token_idx;
                    ఒకవేళ (స్థానం > 0 && స్థానం < K) {
                        దశాంశం * state_snapshot =
                            (దశాంశం *) ((అక్షరం *) state_dst + (size_t) స్థానం * n_seqs * src0->nb[3]);
                        కొరకు (int64_t i = 0; i < d_state; ++i) {
                            state_snapshot[i] = state_dst[i];
                        }
                    }

                    dst_దత్తాంశం[seq_idx * (n_seq_టోకెన్లు * n_తల * తల_dim) + token_idx * (n_తల * తల_dim) +
                             తల_idx * తల_dim + dim_idx] = sumf;
                }
            }
        }
    }

    తిరిగి 0;
}
