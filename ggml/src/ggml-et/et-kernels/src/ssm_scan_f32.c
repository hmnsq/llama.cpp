#شامل "ggml_tensor.h"
#شامل "math_fp.h"
#شامل "platform.h"

#شامل <stdint.h>

ساختار ggml_et_ssm_scan_params {
    ساختار ggml_tensor src0;  // s:   [d_state, head_dim, n_head, n_seqs]
    ساختار ggml_tensor src1;  // x:   [head_dim, n_head, n_seq_tokens, n_seqs]
    ساختار ggml_tensor src2;  // dt:  [n_head, n_seq_tokens, n_seqs]
    ساختار ggml_tensor src3;  // A:   [d_state, n_head] or [1, n_head]
    ساختار ggml_tensor src4;  // B:   [d_state, n_group, n_seq_tokens, n_seqs]
    ساختار ggml_tensor src5;  // C:   [d_state, n_group, n_seq_tokens, n_seqs]
    ساختار ggml_tensor src6;  // ids: [n_seqs] i32
    ساختار ggml_tensor dst;   // packed [y, states]
    int32_t            K;
};

ایستا درونخطی اعشاری softplus_f32(اعشاری x) {
    بازگشت x <= 20.0f ? et_logf(1.0f + et_expf(x)) : x;
}

صحیح entry_point(ساختار ggml_et_ssm_scan_params * params, تهی * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    اگر (!kernel_env) {
        بازگشت -1;
    }

    ثابت صحیح thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    ثابت صحیح رقم_threads = get_رقم_threads(kernel_env->shire_mask);

    اگر (thread_id < 0) {
        بازگشت 0;
    }

    اگر (params == 0 || ((uint64_t) params & 0x7) != 0) {
        بازگشت -1;
    }

    ساختار ggml_tensor * src0 = &params->src0;
    ساختار ggml_tensor * src1 = &params->src1;
    ساختار ggml_tensor * src2 = &params->src2;
    ساختار ggml_tensor * src3 = &params->src3;
    ساختار ggml_tensor * src4 = &params->src4;
    ساختار ggml_tensor * src5 = &params->src5;
    ساختار ggml_tensor * src6 = &params->src6;
    ساختار ggml_tensor * dst  = &params->dst;

    اگر (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_F32 ||
        src3->type != GGML_TYPE_F32 || src4->type != GGML_TYPE_F32 || src5->type != GGML_TYPE_F32 ||
        src6->type != GGML_TYPE_I32 || dst->type != GGML_TYPE_F32) {
        بازگشت -1;
    }

    ثابت اعشاری *   s_data   = (ثابت اعشاری *) src0->data;
    ثابت اعشاری *   x_data   = (ثابت اعشاری *) src1->data;
    ثابت اعشاری *   dt_data  = (ثابت اعشاری *) src2->data;
    ثابت اعشاری *   A_data   = (ثابت اعشاری *) src3->data;
    ثابت اعشاری *   B_data   = (ثابت اعشاری *) src4->data;
    ثابت اعشاری *   C_data   = (ثابت اعشاری *) src5->data;
    ثابت int32_t * ids      = (ثابت int32_t *) src6->data;
    اعشاری *         dst_data = (اعشاری *) dst->data;

    اگر (!s_data || !x_data || !dt_data || !A_data || !B_data || !C_data || !ids || !dst_data) {
        بازگشت -1;
    }

    ثابت int64_t d_state      = src0->ne[0];
    ثابت int64_t head_dim     = src0->ne[1];
    ثابت int64_t n_head       = src1->ne[1];
    ثابت int64_t n_group      = src4->ne[1];
    ثابت int64_t n_seq_tokens = src1->ne[2];
    ثابت int64_t n_seqs       = src1->ne[3];
    ثابت int64_t y_elems      = src1->ne[0] * src1->ne[1] * src1->ne[2] * src1->ne[3];
    ثابت int64_t K            = params->K;

    اگر (src0->nb[0] != اندازه(اعشاری) || src1->nb[0] != اندازه(اعشاری) || src2->nb[0] != اندازه(اعشاری) ||
        src3->nb[0] != اندازه(اعشاری) || src4->nb[0] != اندازه(اعشاری) || src5->nb[0] != اندازه(اعشاری) ||
        src6->nb[0] != اندازه(int32_t) || dst->nb[0] != اندازه(اعشاری)) {
        بازگشت -1;
    }

    اگر (K < 1 || n_group <= 0 || n_head % n_group != 0) {
        بازگشت -1;
    }

    // Cache-line bundling on the dst output (1 dst float per (head, dim, token)).
    //   - When head_dim < 16: bundle 16/head_dim heads per work-unit (1 line of dst).
    //   - When head_dim >= 16: each head's dim slice spans head_dim/16 lines, so we
    //     can split dims into chunks of 16 across threads without false sharing.
    ثابت int64_t dst_lanes_per_cl    = 16;
    ثابت int64_t heads_per_cacheline = head_dim >= dst_lanes_per_cl ? 1 : (dst_lanes_per_cl / head_dim);
    ثابت int64_t heads_per_block     = heads_per_cacheline > 0 ? heads_per_cacheline : 1;
    ثابت int64_t blocks_per_seq      = (n_head + heads_per_block - 1) / heads_per_block;
    ثابت int64_t dim_chunk_lanes     = head_dim >= dst_lanes_per_cl ? dst_lanes_per_cl : head_dim;
    ثابت int64_t dim_chunks_per_head = (head_dim + dim_chunk_lanes - 1) / dim_chunk_lanes;

    // A "unit" = (seq, head_block, dim_chunk). This expands the parallelism by a
    // factor of dim_chunks_per_head over the prior block-only scheme; for Mamba-2
    // shapes (head_dim=64) that's a 4x bump in active threads.
    ثابت int64_t units_per_seq    = blocks_per_seq * dim_chunks_per_head;
    ثابت int64_t total_units      = n_seqs * units_per_seq;
    ثابت int64_t units_per_thread = (total_units + رقم_threads - 1) / رقم_threads;
    ثابت int64_t unit_begin       = (int64_t) thread_id * units_per_thread;
    int64_t       unit_end         = unit_begin + units_per_thread;

    اگر (unit_begin >= total_units) {
        بازگشت 0;
    }

    اگر (unit_end > total_units) {
        unit_end = total_units;
    }

    ثابت صحیح     A_broadcast = (src3->ne[0] == 1);
    ثابت int64_t d_state_vec = (d_state / 8) * 8;  // largest multiple of 8 <= d_state
    ثابت اعشاری   log2e_const = 1.4426950408889634f;

    برای (int64_t unit = unit_begin; unit < unit_end; ++unit) {
        ثابت int64_t seq_idx       = unit / units_per_seq;
        ثابت int64_t unit_in_seq   = unit % units_per_seq;
        ثابت int64_t block_in_seq  = unit_in_seq / dim_chunks_per_head;
        ثابت int64_t dim_chunk_idx = unit_in_seq % dim_chunks_per_head;
        ثابت int64_t head_begin    = block_in_seq * heads_per_block;
        int64_t       head_end      = head_begin + heads_per_block;

        اگر (head_end > n_head) {
            head_end = n_head;
        }

        ثابت int64_t dim_begin = dim_chunk_idx * dim_chunk_lanes;
        int64_t       dim_end   = dim_begin + dim_chunk_lanes;
        اگر (dim_end > head_dim) {
            dim_end = head_dim;
        }

        ثابت int32_t state_seq = ids[seq_idx];

        برای (int64_t head_idx = head_begin; head_idx < head_end; ++head_idx) {
            ثابت int64_t group_idx = head_idx / (n_head / n_group);

            // A pointer for this head: contiguous over state_idx when not broadcast
            ثابت اعشاری * A_row = (ثابت اعشاری *) ((ثابت نویسه *) A_data + (size_t) head_idx * src3->nb[1]);

            برای (int64_t dim_idx = dim_begin; dim_idx < dim_end; ++dim_idx) {
                ثابت اعشاری * state_src =
                    (ثابت اعشاری *) ((ثابت نویسه *) s_data + (size_t) dim_idx * src0->nb[1] +
                                     (size_t) head_idx * src0->nb[2] + (size_t) state_seq * src0->nb[3]);

                اعشاری * state_dst =
                    (اعشاری *) ((نویسه *) dst_data + (size_t) y_elems * اندازه(اعشاری) + (size_t) dim_idx * src0->nb[1] +
                               (size_t) head_idx * src0->nb[2] + (size_t) seq_idx * src0->nb[3]);

                برای (int64_t token_idx = 0; token_idx < n_seq_tokens; ++token_idx) {
                    ثابت اعشاری * x_ptr =
                        (ثابت اعشاری *) ((ثابت نویسه *) x_data + (size_t) dim_idx * src1->nb[0] +
                                         (size_t) head_idx * src1->nb[1] + (size_t) token_idx * src1->nb[2] +
                                         (size_t) seq_idx * src1->nb[3]);

                    ثابت اعشاری * dt_ptr =
                        (ثابت اعشاری *) ((ثابت نویسه *) dt_data + (size_t) head_idx * src2->nb[0] +
                                         (size_t) token_idx * src2->nb[1] + (size_t) seq_idx * src2->nb[2]);

                    ثابت اعشاری * B_row =
                        (ثابت اعشاری *) ((ثابت نویسه *) B_data + (size_t) group_idx * src4->nb[1] +
                                         (size_t) token_idx * src4->nb[2] + (size_t) seq_idx * src4->nb[3]);

                    ثابت اعشاری * C_row =
                        (ثابت اعشاری *) ((ثابت نویسه *) C_data + (size_t) group_idx * src5->nb[1] +
                                         (size_t) token_idx * src5->nb[2] + (size_t) seq_idx * src5->nb[3]);

                    ثابت اعشاری dt_softplus = softplus_f32(*dt_ptr);
                    ثابت اعشاری x_dt        = (*x_ptr) * dt_softplus;
                    ثابت اعشاری dt_log2e    = dt_softplus * log2e_const;

                    // Source of "previous state" for this token: input state on token 0,
                    // last token's state thereafter (we wrote it into state_dst).
                    ثابت اعشاری * prev_row = (token_idx == 0) ? state_src : state_dst;

                    اعشاری   sumf      = 0.0f;
                    int64_t state_idx = 0;

                    اگر (d_state_vec > 0) {
                        // Save mask, enable all 8 vector lanes for the state loop.
                        unsigned long saved_mask;
                        __asm__ فرار("mova.x.m %0" : "=r"(saved_mask));
                        __asm__ فرار("mov.m.x m0, x0, 0xFF");

                        // Per-token broadcasts:
                        //   f20 = x_dt     (B*x_dt)
                        //   f21 = dt_log2e (for fexp.ps when A is per-state)
                        //   f22 = dA       (only when A is broadcast scalar)
                        //   f23 = sum-of-products accumulator (zeroed)
                        __asm__ فرار(
                            "fbc.ps f20, %[xdt]\n\t"
                            "fbc.ps f21, %[dtl]\n\t"
                            "fbci.pi f23, 0\n\t"
                            :
                            : [xdt] "m"(x_dt), [dtl] "m"(dt_log2e)
                            : "f20", "f21", "f23");

                        اگر (A_broadcast) {
                            // dA is a per-head scalar — compute once and splat.
                            ثابت اعشاری dA_scalar = et_expf(dt_softplus * (*A_row));
                            __asm__ فرار("fbc.ps f22, %[da]\n\t" : : [da] "m"(dA_scalar) : "f22");
                        }

                        برای (; state_idx < d_state_vec; state_idx += 8) {
                            اگر (!A_broadcast) {
                                // f22 = exp(dt_softplus * A[state..state+7])
                                //     = 2^((dt_softplus * A) * log2e) via fexp.ps
                                __asm__ فرار(
                                    "flw.ps  f24, %[av]\n\t"
                                    "fmul.ps f24, f24, f21\n\t"  // A * dt_log2e
                                    "fexp.ps f22, f24\n\t"       // dA = 2^(...)
                                    :
                                    : [av] "m"(*(ثابت اعشاری (*)[8]) & A_row[state_idx])
                                    : "f22", "f24");
                            }

                            // state = prev * dA + B * x_dt
                            // sumf += state * C
                            // Reads prev before writing state_dst — safe even when
                            // prev_row == state_dst (write-after-read, same index).
                            __asm__ فرار(
                                "flw.ps  f25, %[prev]\n\t"
                                "flw.ps  f26, %[bv]\n\t"
                                "flw.ps  f27, %[cv]\n\t"
                                "fmul.ps f26, f26, f20\n\t"        // B * x_dt
                                "fmadd.ps f25, f25, f22, f26\n\t"  // state = prev*dA + B*x_dt
                                "fsw.ps  f25, %[sd]\n\t"
                                "fmadd.ps f23, f25, f27, f23\n\t"  // sum += state*C
                                : [sd] "=m"(*(اعشاری (*)[8]) & state_dst[state_idx])
                                : [prev] "m"(*(ثابت اعشاری (*)[8]) & prev_row[state_idx]),
                                  [bv] "m"(*(ثابت اعشاری (*)[8]) & B_row[state_idx]),
                                  [cv] "m"(*(ثابت اعشاری (*)[8]) & C_row[state_idx])
                                : "f25", "f26", "f27");
                        }

                        // Horizontal reduce f23 (8 lanes) -> scalar sumf.
                        __asm__ فرار(
                            "fswizz.ps f1, f23, 0xB1\n\t"
                            "fadd.ps   f2, f23, f1, rne\n\t"
                            "fswizz.ps f3, f2, 0x4E\n\t"
                            "fadd.ps   f4, f2, f3, rne\n\t"
                            "fmvz.x.ps t0, f4, 4\n\t"
                            "fbcx.ps   f5, t0\n\t"
                            "fadd.ps   %[vout], f4, f5, rne\n\t"
                            : [vout] "=f"(sumf)::"t0", "f1", "f2", "f3", "f4", "f5");

                        __asm__ فرار("mova.m.x %0" ::"r"(saved_mask));
                    }

                    // Scalar tail (d_state not a multiple of 8).
                    برای (; state_idx < d_state; ++state_idx) {
                        ثابت اعشاری prev_state = prev_row[state_idx];
                        ثابت اعشاری A_val      = A_broadcast ? *A_row : A_row[state_idx];
                        ثابت اعشاری dA         = et_expf(dt_softplus * A_val);
                        ثابت اعشاری st         = prev_state * dA + B_row[state_idx] * x_dt;
                        state_dst[state_idx]   = st;
                        sumf += st * C_row[state_idx];
                    }

                    ثابت int64_t slot = n_seq_tokens - 1 - token_idx;
                    اگر (slot > 0 && slot < K) {
                        اعشاری * state_snapshot =
                            (اعشاری *) ((نویسه *) state_dst + (size_t) slot * n_seqs * src0->nb[3]);
                        برای (int64_t i = 0; i < d_state; ++i) {
                            state_snapshot[i] = state_dst[i];
                        }
                    }

                    dst_data[seq_idx * (n_seq_tokens * n_head * head_dim) + token_idx * (n_head * head_dim) +
                             head_idx * head_dim + dim_idx] = sumf;
                }
            }
        }
    }

    بازگشت 0;
}
