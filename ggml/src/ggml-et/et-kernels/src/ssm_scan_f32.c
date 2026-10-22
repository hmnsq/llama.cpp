#അന്തര്ഭാവ "ggml_tensor.h"
#അന്തര്ഭാവ "math_fp.h"
#അന്തര്ഭാവ "platform.h"

#അന്തര്ഭാവ <stdint.h>

ഘടന ggml_et_ssm_scan_params {
    ഘടന ggml_tensor src0;  // s:   [d_state, head_dim, n_head, n_seqs]
    ഘടന ggml_tensor src1;  // x:   [head_dim, n_head, n_seq_tokens, n_seqs]
    ഘടന ggml_tensor src2;  // dt:  [n_head, n_seq_tokens, n_seqs]
    ഘടന ggml_tensor src3;  // A:   [d_state, n_head] or [1, n_head]
    ഘടന ggml_tensor src4;  // B:   [d_state, n_group, n_seq_tokens, n_seqs]
    ഘടന ggml_tensor src5;  // C:   [d_state, n_group, n_seq_tokens, n_seqs]
    ഘടന ggml_tensor src6;  // ids: [n_seqs] i32
    ഘടന ggml_tensor dst;   // packed [y, states]
    int32_t            K;
};

സ്ഥിതം വരിയിൽ ദശാംശം softplus_f32(ദശാംശം x) {
    തിരികെ x <= 20.0f ? et_logf(1.0f + et_expf(x)) : x;
}

പൂർണ്ണസംഖ്യ entry_point(ഘടന ggml_et_ssm_scan_params * params, ശൂന്യം * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    എങ്കിൽ (!kernel_env) {
        തിരികെ -1;
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    സ്ഥിരം പൂർണ്ണസംഖ്യ അക്കം_threads = get_അക്കം_threads(kernel_env->shire_mask);

    എങ്കിൽ (thread_id < 0) {
        തിരികെ 0;
    }

    എങ്കിൽ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        തിരികെ -1;
    }

    ഘടന ggml_tensor * src0 = &params->src0;
    ഘടന ggml_tensor * src1 = &params->src1;
    ഘടന ggml_tensor * src2 = &params->src2;
    ഘടന ggml_tensor * src3 = &params->src3;
    ഘടന ggml_tensor * src4 = &params->src4;
    ഘടന ggml_tensor * src5 = &params->src5;
    ഘടന ggml_tensor * src6 = &params->src6;
    ഘടന ggml_tensor * dst  = &params->dst;

    എങ്കിൽ (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_F32 ||
        src3->type != GGML_TYPE_F32 || src4->type != GGML_TYPE_F32 || src5->type != GGML_TYPE_F32 ||
        src6->type != GGML_TYPE_I32 || dst->type != GGML_TYPE_F32) {
        തിരികെ -1;
    }

    സ്ഥിരം ദശാംശം *   s_data   = (സ്ഥിരം ദശാംശം *) src0->data;
    സ്ഥിരം ദശാംശം *   x_data   = (സ്ഥിരം ദശാംശം *) src1->data;
    സ്ഥിരം ദശാംശം *   dt_data  = (സ്ഥിരം ദശാംശം *) src2->data;
    സ്ഥിരം ദശാംശം *   A_data   = (സ്ഥിരം ദശാംശം *) src3->data;
    സ്ഥിരം ദശാംശം *   B_data   = (സ്ഥിരം ദശാംശം *) src4->data;
    സ്ഥിരം ദശാംശം *   C_data   = (സ്ഥിരം ദശാംശം *) src5->data;
    സ്ഥിരം int32_t * ids      = (സ്ഥിരം int32_t *) src6->data;
    ദശാംശം *         dst_data = (ദശാംശം *) dst->data;

    എങ്കിൽ (!s_data || !x_data || !dt_data || !A_data || !B_data || !C_data || !ids || !dst_data) {
        തിരികെ -1;
    }

    സ്ഥിരം int64_t d_state      = src0->ne[0];
    സ്ഥിരം int64_t head_dim     = src0->ne[1];
    സ്ഥിരം int64_t n_head       = src1->ne[1];
    സ്ഥിരം int64_t n_group      = src4->ne[1];
    സ്ഥിരം int64_t n_seq_tokens = src1->ne[2];
    സ്ഥിരം int64_t n_seqs       = src1->ne[3];
    സ്ഥിരം int64_t y_elems      = src1->ne[0] * src1->ne[1] * src1->ne[2] * src1->ne[3];
    സ്ഥിരം int64_t K            = params->K;

    എങ്കിൽ (src0->nb[0] != വലുപ്പം(ദശാംശം) || src1->nb[0] != വലുപ്പം(ദശാംശം) || src2->nb[0] != വലുപ്പം(ദശാംശം) ||
        src3->nb[0] != വലുപ്പം(ദശാംശം) || src4->nb[0] != വലുപ്പം(ദശാംശം) || src5->nb[0] != വലുപ്പം(ദശാംശം) ||
        src6->nb[0] != വലുപ്പം(int32_t) || dst->nb[0] != വലുപ്പം(ദശാംശം)) {
        തിരികെ -1;
    }

    എങ്കിൽ (K < 1 || n_group <= 0 || n_head % n_group != 0) {
        തിരികെ -1;
    }

    // Cache-line bundling on the dst output (1 dst float per (head, dim, token)).
    //   - When head_dim < 16: bundle 16/head_dim heads per work-unit (1 line of dst).
    //   - When head_dim >= 16: each head's dim slice spans head_dim/16 lines, so we
    //     can split dims into chunks of 16 across threads without false sharing.
    സ്ഥിരം int64_t dst_lanes_per_cl    = 16;
    സ്ഥിരം int64_t heads_per_cacheline = head_dim >= dst_lanes_per_cl ? 1 : (dst_lanes_per_cl / head_dim);
    സ്ഥിരം int64_t heads_per_block     = heads_per_cacheline > 0 ? heads_per_cacheline : 1;
    സ്ഥിരം int64_t blocks_per_seq      = (n_head + heads_per_block - 1) / heads_per_block;
    സ്ഥിരം int64_t dim_chunk_lanes     = head_dim >= dst_lanes_per_cl ? dst_lanes_per_cl : head_dim;
    സ്ഥിരം int64_t dim_chunks_per_head = (head_dim + dim_chunk_lanes - 1) / dim_chunk_lanes;

    // A "unit" = (seq, head_block, dim_chunk). This expands the parallelism by a
    // factor of dim_chunks_per_head over the prior block-only scheme; for Mamba-2
    // shapes (head_dim=64) that's a 4x bump in active threads.
    സ്ഥിരം int64_t units_per_seq    = blocks_per_seq * dim_chunks_per_head;
    സ്ഥിരം int64_t total_units      = n_seqs * units_per_seq;
    സ്ഥിരം int64_t units_per_thread = (total_units + അക്കം_threads - 1) / അക്കം_threads;
    സ്ഥിരം int64_t unit_begin       = (int64_t) thread_id * units_per_thread;
    int64_t       unit_end         = unit_begin + units_per_thread;

    എങ്കിൽ (unit_begin >= total_units) {
        തിരികെ 0;
    }

    എങ്കിൽ (unit_end > total_units) {
        unit_end = total_units;
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ     A_broadcast = (src3->ne[0] == 1);
    സ്ഥിരം int64_t d_state_vec = (d_state / 8) * 8;  // largest multiple of 8 <= d_state
    സ്ഥിരം ദശാംശം   log2e_const = 1.4426950408889634f;

    വേണ്ടി (int64_t unit = unit_begin; unit < unit_end; ++unit) {
        സ്ഥിരം int64_t seq_idx       = unit / units_per_seq;
        സ്ഥിരം int64_t unit_in_seq   = unit % units_per_seq;
        സ്ഥിരം int64_t block_in_seq  = unit_in_seq / dim_chunks_per_head;
        സ്ഥിരം int64_t dim_chunk_idx = unit_in_seq % dim_chunks_per_head;
        സ്ഥിരം int64_t head_begin    = block_in_seq * heads_per_block;
        int64_t       head_end      = head_begin + heads_per_block;

        എങ്കിൽ (head_end > n_head) {
            head_end = n_head;
        }

        സ്ഥിരം int64_t dim_begin = dim_chunk_idx * dim_chunk_lanes;
        int64_t       dim_end   = dim_begin + dim_chunk_lanes;
        എങ്കിൽ (dim_end > head_dim) {
            dim_end = head_dim;
        }

        സ്ഥിരം int32_t state_seq = ids[seq_idx];

        വേണ്ടി (int64_t head_idx = head_begin; head_idx < head_end; ++head_idx) {
            സ്ഥിരം int64_t group_idx = head_idx / (n_head / n_group);

            // A pointer for this head: contiguous over state_idx when not broadcast
            സ്ഥിരം ദശാംശം * A_row = (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) A_data + (size_t) head_idx * src3->nb[1]);

            വേണ്ടി (int64_t dim_idx = dim_begin; dim_idx < dim_end; ++dim_idx) {
                സ്ഥിരം ദശാംശം * state_src =
                    (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) s_data + (size_t) dim_idx * src0->nb[1] +
                                     (size_t) head_idx * src0->nb[2] + (size_t) state_seq * src0->nb[3]);

                ദശാംശം * state_dst =
                    (ദശാംശം *) ((അക്ഷരം *) dst_data + (size_t) y_elems * വലുപ്പം(ദശാംശം) + (size_t) dim_idx * src0->nb[1] +
                               (size_t) head_idx * src0->nb[2] + (size_t) seq_idx * src0->nb[3]);

                വേണ്ടി (int64_t token_idx = 0; token_idx < n_seq_tokens; ++token_idx) {
                    സ്ഥിരം ദശാംശം * x_ptr =
                        (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) x_data + (size_t) dim_idx * src1->nb[0] +
                                         (size_t) head_idx * src1->nb[1] + (size_t) token_idx * src1->nb[2] +
                                         (size_t) seq_idx * src1->nb[3]);

                    സ്ഥിരം ദശാംശം * dt_ptr =
                        (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) dt_data + (size_t) head_idx * src2->nb[0] +
                                         (size_t) token_idx * src2->nb[1] + (size_t) seq_idx * src2->nb[2]);

                    സ്ഥിരം ദശാംശം * B_row =
                        (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) B_data + (size_t) group_idx * src4->nb[1] +
                                         (size_t) token_idx * src4->nb[2] + (size_t) seq_idx * src4->nb[3]);

                    സ്ഥിരം ദശാംശം * C_row =
                        (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) C_data + (size_t) group_idx * src5->nb[1] +
                                         (size_t) token_idx * src5->nb[2] + (size_t) seq_idx * src5->nb[3]);

                    സ്ഥിരം ദശാംശം dt_softplus = softplus_f32(*dt_ptr);
                    സ്ഥിരം ദശാംശം x_dt        = (*x_ptr) * dt_softplus;
                    സ്ഥിരം ദശാംശം dt_log2e    = dt_softplus * log2e_const;

                    // Source of "previous state" for this token: input state on token 0,
                    // last token's state thereafter (we wrote it into state_dst).
                    സ്ഥിരം ദശാംശം * prev_row = (token_idx == 0) ? state_src : state_dst;

                    ദശാംശം   sumf      = 0.0f;
                    int64_t state_idx = 0;

                    എങ്കിൽ (d_state_vec > 0) {
                        // Save mask, enable all 8 vector lanes for the state loop.
                        unsigned long saved_mask;
                        __asm__ അസ്ഥിര("mova.x.m %0" : "=r"(saved_mask));
                        __asm__ അസ്ഥിര("mov.m.x m0, x0, 0xFF");

                        // Per-token broadcasts:
                        //   f20 = x_dt     (B*x_dt)
                        //   f21 = dt_log2e (for fexp.ps when A is per-state)
                        //   f22 = dA       (only when A is broadcast scalar)
                        //   f23 = sum-of-products accumulator (zeroed)
                        __asm__ അസ്ഥിര(
                            "fbc.ps f20, %[xdt]\n\t"
                            "fbc.ps f21, %[dtl]\n\t"
                            "fbci.pi f23, 0\n\t"
                            :
                            : [xdt] "m"(x_dt), [dtl] "m"(dt_log2e)
                            : "f20", "f21", "f23");

                        എങ്കിൽ (A_broadcast) {
                            // dA is a per-head scalar — compute once and splat.
                            സ്ഥിരം ദശാംശം dA_scalar = et_expf(dt_softplus * (*A_row));
                            __asm__ അസ്ഥിര("fbc.ps f22, %[da]\n\t" : : [da] "m"(dA_scalar) : "f22");
                        }

                        വേണ്ടി (; state_idx < d_state_vec; state_idx += 8) {
                            എങ്കിൽ (!A_broadcast) {
                                // f22 = exp(dt_softplus * A[state..state+7])
                                //     = 2^((dt_softplus * A) * log2e) via fexp.ps
                                __asm__ അസ്ഥിര(
                                    "flw.ps  f24, %[av]\n\t"
                                    "fmul.ps f24, f24, f21\n\t"  // A * dt_log2e
                                    "fexp.ps f22, f24\n\t"       // dA = 2^(...)
                                    :
                                    : [av] "m"(*(സ്ഥിരം ദശാംശം (*)[8]) & A_row[state_idx])
                                    : "f22", "f24");
                            }

                            // state = prev * dA + B * x_dt
                            // sumf += state * C
                            // Reads prev before writing state_dst — safe even when
                            // prev_row == state_dst (write-after-read, same index).
                            __asm__ അസ്ഥിര(
                                "flw.ps  f25, %[prev]\n\t"
                                "flw.ps  f26, %[bv]\n\t"
                                "flw.ps  f27, %[cv]\n\t"
                                "fmul.ps f26, f26, f20\n\t"        // B * x_dt
                                "fmadd.ps f25, f25, f22, f26\n\t"  // state = prev*dA + B*x_dt
                                "fsw.ps  f25, %[sd]\n\t"
                                "fmadd.ps f23, f25, f27, f23\n\t"  // sum += state*C
                                : [sd] "=m"(*(ദശാംശം (*)[8]) & state_dst[state_idx])
                                : [prev] "m"(*(സ്ഥിരം ദശാംശം (*)[8]) & prev_row[state_idx]),
                                  [bv] "m"(*(സ്ഥിരം ദശാംശം (*)[8]) & B_row[state_idx]),
                                  [cv] "m"(*(സ്ഥിരം ദശാംശം (*)[8]) & C_row[state_idx])
                                : "f25", "f26", "f27");
                        }

                        // Horizontal reduce f23 (8 lanes) -> scalar sumf.
                        __asm__ അസ്ഥിര(
                            "fswizz.ps f1, f23, 0xB1\n\t"
                            "fadd.ps   f2, f23, f1, rne\n\t"
                            "fswizz.ps f3, f2, 0x4E\n\t"
                            "fadd.ps   f4, f2, f3, rne\n\t"
                            "fmvz.x.ps t0, f4, 4\n\t"
                            "fbcx.ps   f5, t0\n\t"
                            "fadd.ps   %[vout], f4, f5, rne\n\t"
                            : [vout] "=f"(sumf)::"t0", "f1", "f2", "f3", "f4", "f5");

                        __asm__ അസ്ഥിര("mova.m.x %0" ::"r"(saved_mask));
                    }

                    // Scalar tail (d_state not a multiple of 8).
                    വേണ്ടി (; state_idx < d_state; ++state_idx) {
                        സ്ഥിരം ദശാംശം prev_state = prev_row[state_idx];
                        സ്ഥിരം ദശാംശം A_val      = A_broadcast ? *A_row : A_row[state_idx];
                        സ്ഥിരം ദശാംശം dA         = et_expf(dt_softplus * A_val);
                        സ്ഥിരം ദശാംശം st         = prev_state * dA + B_row[state_idx] * x_dt;
                        state_dst[state_idx]   = st;
                        sumf += st * C_row[state_idx];
                    }

                    സ്ഥിരം int64_t slot = n_seq_tokens - 1 - token_idx;
                    എങ്കിൽ (slot > 0 && slot < K) {
                        ദശാംശം * state_snapshot =
                            (ദശാംശം *) ((അക്ഷരം *) state_dst + (size_t) slot * n_seqs * src0->nb[3]);
                        വേണ്ടി (int64_t i = 0; i < d_state; ++i) {
                            state_snapshot[i] = state_dst[i];
                        }
                    }

                    dst_data[seq_idx * (n_seq_tokens * n_head * head_dim) + token_idx * (n_head * head_dim) +
                             head_idx * head_dim + dim_idx] = sumf;
                }
            }
        }
    }

    തിരികെ 0;
}
