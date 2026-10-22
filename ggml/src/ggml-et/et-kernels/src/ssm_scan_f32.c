#অন্তর্ভাব "ggml_tensor.h"
#অন্তর্ভাব "math_fp.h"
#অন্তর্ভাব "platform.h"

#অন্তর্ভাব <stdint.h>

কাঠামো ggml_et_ssm_scan_params {
    কাঠামো ggml_tensor src0;  // s:   [d_state, head_dim, n_head, n_seqs]
    কাঠামো ggml_tensor src1;  // x:   [head_dim, n_head, n_seq_tokens, n_seqs]
    কাঠামো ggml_tensor src2;  // dt:  [n_head, n_seq_tokens, n_seqs]
    কাঠামো ggml_tensor src3;  // A:   [d_state, n_head] or [1, n_head]
    কাঠামো ggml_tensor src4;  // B:   [d_state, n_group, n_seq_tokens, n_seqs]
    কাঠামো ggml_tensor src5;  // C:   [d_state, n_group, n_seq_tokens, n_seqs]
    কাঠামো ggml_tensor src6;  // ids: [n_seqs] i32
    কাঠামো ggml_tensor dst;   // packed [y, states]
    int32_t            K;
};

স্থির অন্তঃরেখ দশমিক softplus_f32(দশমিক x) {
    ফেরত x <= 20.0f ? et_logf(1.0f + et_expf(x)) : x;
}

পূর্ণসংখ্যা entry_point(কাঠামো ggml_et_ssm_scan_params * params, শূন্য * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    যদি (!kernel_env) {
        ফেরত -1;
    }

    ধ্রুবক পূর্ণসংখ্যা thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    ধ্রুবক পূর্ণসংখ্যা অঙ্ক_threads = get_অঙ্ক_threads(kernel_env->shire_mask);

    যদি (thread_id < 0) {
        ফেরত 0;
    }

    যদি (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ফেরত -1;
    }

    কাঠামো ggml_tensor * src0 = &params->src0;
    কাঠামো ggml_tensor * src1 = &params->src1;
    কাঠামো ggml_tensor * src2 = &params->src2;
    কাঠামো ggml_tensor * src3 = &params->src3;
    কাঠামো ggml_tensor * src4 = &params->src4;
    কাঠামো ggml_tensor * src5 = &params->src5;
    কাঠামো ggml_tensor * src6 = &params->src6;
    কাঠামো ggml_tensor * dst  = &params->dst;

    যদি (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_F32 ||
        src3->type != GGML_TYPE_F32 || src4->type != GGML_TYPE_F32 || src5->type != GGML_TYPE_F32 ||
        src6->type != GGML_TYPE_I32 || dst->type != GGML_TYPE_F32) {
        ফেরত -1;
    }

    ধ্রুবক দশমিক *   s_তথ্য   = (ধ্রুবক দশমিক *) src0->তথ্য;
    ধ্রুবক দশমিক *   x_তথ্য   = (ধ্রুবক দশমিক *) src1->তথ্য;
    ধ্রুবক দশমিক *   dt_তথ্য  = (ধ্রুবক দশমিক *) src2->তথ্য;
    ধ্রুবক দশমিক *   A_তথ্য   = (ধ্রুবক দশমিক *) src3->তথ্য;
    ধ্রুবক দশমিক *   B_তথ্য   = (ধ্রুবক দশমিক *) src4->তথ্য;
    ধ্রুবক দশমিক *   C_তথ্য   = (ধ্রুবক দশমিক *) src5->তথ্য;
    ধ্রুবক int32_t * ids      = (ধ্রুবক int32_t *) src6->তথ্য;
    দশমিক *         dst_তথ্য = (দশমিক *) dst->তথ্য;

    যদি (!s_তথ্য || !x_তথ্য || !dt_তথ্য || !A_তথ্য || !B_তথ্য || !C_তথ্য || !ids || !dst_তথ্য) {
        ফেরত -1;
    }

    ধ্রুবক int64_t d_state      = src0->ne[0];
    ধ্রুবক int64_t শীর্ষ_dim     = src0->ne[1];
    ধ্রুবক int64_t n_শীর্ষ       = src1->ne[1];
    ধ্রুবক int64_t n_group      = src4->ne[1];
    ধ্রুবক int64_t n_seq_tokens = src1->ne[2];
    ধ্রুবক int64_t n_seqs       = src1->ne[3];
    ধ্রুবক int64_t y_elems      = src1->ne[0] * src1->ne[1] * src1->ne[2] * src1->ne[3];
    ধ্রুবক int64_t K            = params->K;

    যদি (src0->nb[0] != আকার(দশমিক) || src1->nb[0] != আকার(দশমিক) || src2->nb[0] != আকার(দশমিক) ||
        src3->nb[0] != আকার(দশমিক) || src4->nb[0] != আকার(দশমিক) || src5->nb[0] != আকার(দশমিক) ||
        src6->nb[0] != আকার(int32_t) || dst->nb[0] != আকার(দশমিক)) {
        ফেরত -1;
    }

    যদি (K < 1 || n_group <= 0 || n_শীর্ষ % n_group != 0) {
        ফেরত -1;
    }

    // Cache-line bundling on the dst output (1 dst float per (head, dim, token)).
    //   - When head_dim < 16: bundle 16/head_dim heads per work-unit (1 line of dst).
    //   - When head_dim >= 16: each head's dim slice spans head_dim/16 lines, so we
    //     can split dims into chunks of 16 across threads without false sharing.
    ধ্রুবক int64_t dst_lanes_per_cl    = 16;
    ধ্রুবক int64_t heads_per_cacheline = শীর্ষ_dim >= dst_lanes_per_cl ? 1 : (dst_lanes_per_cl / শীর্ষ_dim);
    ধ্রুবক int64_t heads_per_block     = heads_per_cacheline > 0 ? heads_per_cacheline : 1;
    ধ্রুবক int64_t blocks_per_seq      = (n_শীর্ষ + heads_per_block - 1) / heads_per_block;
    ধ্রুবক int64_t dim_chunk_lanes     = শীর্ষ_dim >= dst_lanes_per_cl ? dst_lanes_per_cl : শীর্ষ_dim;
    ধ্রুবক int64_t dim_chunks_per_শীর্ষ = (শীর্ষ_dim + dim_chunk_lanes - 1) / dim_chunk_lanes;

    // A "unit" = (seq, head_block, dim_chunk). This expands the parallelism by a
    // factor of dim_chunks_per_head over the prior block-only scheme; for Mamba-2
    // shapes (head_dim=64) that's a 4x bump in active threads.
    ধ্রুবক int64_t units_per_seq    = blocks_per_seq * dim_chunks_per_শীর্ষ;
    ধ্রুবক int64_t যোগ_units      = n_seqs * units_per_seq;
    ধ্রুবক int64_t units_per_thread = (যোগ_units + অঙ্ক_threads - 1) / অঙ্ক_threads;
    ধ্রুবক int64_t unit_begin       = (int64_t) thread_id * units_per_thread;
    int64_t       unit_end         = unit_begin + units_per_thread;

    যদি (unit_begin >= যোগ_units) {
        ফেরত 0;
    }

    যদি (unit_end > যোগ_units) {
        unit_end = যোগ_units;
    }

    ধ্রুবক পূর্ণসংখ্যা     A_broadcast = (src3->ne[0] == 1);
    ধ্রুবক int64_t d_state_vec = (d_state / 8) * 8;  // largest multiple of 8 <= d_state
    ধ্রুবক দশমিক   log2e_const = 1.4426950408889634f;

    জন্য (int64_t unit = unit_begin; unit < unit_end; ++unit) {
        ধ্রুবক int64_t seq_idx       = unit / units_per_seq;
        ধ্রুবক int64_t unit_in_seq   = unit % units_per_seq;
        ধ্রুবক int64_t block_in_seq  = unit_in_seq / dim_chunks_per_শীর্ষ;
        ধ্রুবক int64_t dim_chunk_idx = unit_in_seq % dim_chunks_per_শীর্ষ;
        ধ্রুবক int64_t শীর্ষ_begin    = block_in_seq * heads_per_block;
        int64_t       শীর্ষ_end      = শীর্ষ_begin + heads_per_block;

        যদি (শীর্ষ_end > n_শীর্ষ) {
            শীর্ষ_end = n_শীর্ষ;
        }

        ধ্রুবক int64_t dim_begin = dim_chunk_idx * dim_chunk_lanes;
        int64_t       dim_end   = dim_begin + dim_chunk_lanes;
        যদি (dim_end > শীর্ষ_dim) {
            dim_end = শীর্ষ_dim;
        }

        ধ্রুবক int32_t state_seq = ids[seq_idx];

        জন্য (int64_t শীর্ষ_idx = শীর্ষ_begin; শীর্ষ_idx < শীর্ষ_end; ++শীর্ষ_idx) {
            ধ্রুবক int64_t group_idx = শীর্ষ_idx / (n_শীর্ষ / n_group);

            // A pointer for this head: contiguous over state_idx when not broadcast
            ধ্রুবক দশমিক * A_row = (ধ্রুবক দশমিক *) ((ধ্রুবক অক্ষর *) A_তথ্য + (size_t) শীর্ষ_idx * src3->nb[1]);

            জন্য (int64_t dim_idx = dim_begin; dim_idx < dim_end; ++dim_idx) {
                ধ্রুবক দশমিক * state_src =
                    (ধ্রুবক দশমিক *) ((ধ্রুবক অক্ষর *) s_তথ্য + (size_t) dim_idx * src0->nb[1] +
                                     (size_t) শীর্ষ_idx * src0->nb[2] + (size_t) state_seq * src0->nb[3]);

                দশমিক * state_dst =
                    (দশমিক *) ((অক্ষর *) dst_তথ্য + (size_t) y_elems * আকার(দশমিক) + (size_t) dim_idx * src0->nb[1] +
                               (size_t) শীর্ষ_idx * src0->nb[2] + (size_t) seq_idx * src0->nb[3]);

                জন্য (int64_t token_idx = 0; token_idx < n_seq_tokens; ++token_idx) {
                    ধ্রুবক দশমিক * x_ptr =
                        (ধ্রুবক দশমিক *) ((ধ্রুবক অক্ষর *) x_তথ্য + (size_t) dim_idx * src1->nb[0] +
                                         (size_t) শীর্ষ_idx * src1->nb[1] + (size_t) token_idx * src1->nb[2] +
                                         (size_t) seq_idx * src1->nb[3]);

                    ধ্রুবক দশমিক * dt_ptr =
                        (ধ্রুবক দশমিক *) ((ধ্রুবক অক্ষর *) dt_তথ্য + (size_t) শীর্ষ_idx * src2->nb[0] +
                                         (size_t) token_idx * src2->nb[1] + (size_t) seq_idx * src2->nb[2]);

                    ধ্রুবক দশমিক * B_row =
                        (ধ্রুবক দশমিক *) ((ধ্রুবক অক্ষর *) B_তথ্য + (size_t) group_idx * src4->nb[1] +
                                         (size_t) token_idx * src4->nb[2] + (size_t) seq_idx * src4->nb[3]);

                    ধ্রুবক দশমিক * C_row =
                        (ধ্রুবক দশমিক *) ((ধ্রুবক অক্ষর *) C_তথ্য + (size_t) group_idx * src5->nb[1] +
                                         (size_t) token_idx * src5->nb[2] + (size_t) seq_idx * src5->nb[3]);

                    ধ্রুবক দশমিক dt_softplus = softplus_f32(*dt_ptr);
                    ধ্রুবক দশমিক x_dt        = (*x_ptr) * dt_softplus;
                    ধ্রুবক দশমিক dt_log2e    = dt_softplus * log2e_const;

                    // Source of "previous state" for this token: input state on token 0,
                    // last token's state thereafter (we wrote it into state_dst).
                    ধ্রুবক দশমিক * পূর্ব_row = (token_idx == 0) ? state_src : state_dst;

                    দশমিক   sumf      = 0.0f;
                    int64_t state_idx = 0;

                    যদি (d_state_vec > 0) {
                        // Save mask, enable all 8 vector lanes for the state loop.
                        unsigned long saved_mask;
                        __asm__ অস্থির("mova.x.m %0" : "=r"(saved_mask));
                        __asm__ অস্থির("mov.m.x m0, x0, 0xFF");

                        // Per-token broadcasts:
                        //   f20 = x_dt     (B*x_dt)
                        //   f21 = dt_log2e (for fexp.ps when A is per-state)
                        //   f22 = dA       (only when A is broadcast scalar)
                        //   f23 = sum-of-products accumulator (zeroed)
                        __asm__ অস্থির(
                            "fbc.ps f20, %[xdt]\n\t"
                            "fbc.ps f21, %[dtl]\n\t"
                            "fbci.pi f23, 0\n\t"
                            :
                            : [xdt] "m"(x_dt), [dtl] "m"(dt_log2e)
                            : "f20", "f21", "f23");

                        যদি (A_broadcast) {
                            // dA is a per-head scalar — compute once and splat.
                            ধ্রুবক দশমিক dA_scalar = et_expf(dt_softplus * (*A_row));
                            __asm__ অস্থির("fbc.ps f22, %[da]\n\t" : : [da] "m"(dA_scalar) : "f22");
                        }

                        জন্য (; state_idx < d_state_vec; state_idx += 8) {
                            যদি (!A_broadcast) {
                                // f22 = exp(dt_softplus * A[state..state+7])
                                //     = 2^((dt_softplus * A) * log2e) via fexp.ps
                                __asm__ অস্থির(
                                    "flw.ps  f24, %[av]\n\t"
                                    "fmul.ps f24, f24, f21\n\t"  // A * dt_log2e
                                    "fexp.ps f22, f24\n\t"       // dA = 2^(...)
                                    :
                                    : [av] "m"(*(ধ্রুবক দশমিক (*)[8]) & A_row[state_idx])
                                    : "f22", "f24");
                            }

                            // state = prev * dA + B * x_dt
                            // sumf += state * C
                            // Reads prev before writing state_dst — safe even when
                            // prev_row == state_dst (write-after-read, same index).
                            __asm__ অস্থির(
                                "flw.ps  f25, %[prev]\n\t"
                                "flw.ps  f26, %[bv]\n\t"
                                "flw.ps  f27, %[cv]\n\t"
                                "fmul.ps f26, f26, f20\n\t"        // B * x_dt
                                "fmadd.ps f25, f25, f22, f26\n\t"  // state = prev*dA + B*x_dt
                                "fsw.ps  f25, %[sd]\n\t"
                                "fmadd.ps f23, f25, f27, f23\n\t"  // sum += state*C
                                : [sd] "=m"(*(দশমিক (*)[8]) & state_dst[state_idx])
                                : [পূর্ব] "m"(*(ধ্রুবক দশমিক (*)[8]) & পূর্ব_row[state_idx]),
                                  [bv] "m"(*(ধ্রুবক দশমিক (*)[8]) & B_row[state_idx]),
                                  [cv] "m"(*(ধ্রুবক দশমিক (*)[8]) & C_row[state_idx])
                                : "f25", "f26", "f27");
                        }

                        // Horizontal reduce f23 (8 lanes) -> scalar sumf.
                        __asm__ অস্থির(
                            "fswizz.ps f1, f23, 0xB1\n\t"
                            "fadd.ps   f2, f23, f1, rne\n\t"
                            "fswizz.ps f3, f2, 0x4E\n\t"
                            "fadd.ps   f4, f2, f3, rne\n\t"
                            "fmvz.x.ps t0, f4, 4\n\t"
                            "fbcx.ps   f5, t0\n\t"
                            "fadd.ps   %[vout], f4, f5, rne\n\t"
                            : [vout] "=f"(sumf)::"t0", "f1", "f2", "f3", "f4", "f5");

                        __asm__ অস্থির("mova.m.x %0" ::"r"(saved_mask));
                    }

                    // Scalar tail (d_state not a multiple of 8).
                    জন্য (; state_idx < d_state; ++state_idx) {
                        ধ্রুবক দশমিক পূর্ব_state = পূর্ব_row[state_idx];
                        ধ্রুবক দশমিক A_val      = A_broadcast ? *A_row : A_row[state_idx];
                        ধ্রুবক দশমিক dA         = et_expf(dt_softplus * A_val);
                        ধ্রুবক দশমিক st         = পূর্ব_state * dA + B_row[state_idx] * x_dt;
                        state_dst[state_idx]   = st;
                        sumf += st * C_row[state_idx];
                    }

                    ধ্রুবক int64_t slot = n_seq_tokens - 1 - token_idx;
                    যদি (slot > 0 && slot < K) {
                        দশমিক * state_snapshot =
                            (দশমিক *) ((অক্ষর *) state_dst + (size_t) slot * n_seqs * src0->nb[3]);
                        জন্য (int64_t i = 0; i < d_state; ++i) {
                            state_snapshot[i] = state_dst[i];
                        }
                    }

                    dst_তথ্য[seq_idx * (n_seq_tokens * n_শীর্ষ * শীর্ষ_dim) + token_idx * (n_শীর্ষ * শীর্ষ_dim) +
                             শীর্ষ_idx * শীর্ষ_dim + dim_idx] = sumf;
                }
            }
        }
    }

    ফেরত 0;
}
