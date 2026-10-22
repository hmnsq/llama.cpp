#ਅਨ੍ਤਰ੍ਭਾਵ "ggml_tensor.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "math_fp.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "platform.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>

ਬਣਤਰ ggml_et_ssm_ਜਾਂਚ_params {
    ਬਣਤਰ ggml_tensor src0;  // s:   [d_state, head_dim, n_head, n_seqs]
    ਬਣਤਰ ggml_tensor src1;  // x:   [head_dim, n_head, n_seq_tokens, n_seqs]
    ਬਣਤਰ ggml_tensor src2;  // dt:  [n_head, n_seq_tokens, n_seqs]
    ਬਣਤਰ ggml_tensor src3;  // A:   [d_state, n_head] or [1, n_head]
    ਬਣਤਰ ggml_tensor src4;  // B:   [d_state, n_group, n_seq_tokens, n_seqs]
    ਬਣਤਰ ggml_tensor src5;  // C:   [d_state, n_group, n_seq_tokens, n_seqs]
    ਬਣਤਰ ggml_tensor src6;  // ids: [n_seqs] i32
    ਬਣਤਰ ggml_tensor dst;   // packed [y, states]
    int32_t            K;
};

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ softplus_f32(ਦਸ਼ਮਲਵ x) {
    ਵਾਪਸ x <= 20.0f ? et_logf(1.0f + et_expf(x)) : x;
}

ਪੂਰਨ ਇੰਦਰਾਜ_ਬਿੰਦੂ(ਬਣਤਰ ggml_et_ssm_ਜਾਂਚ_params * params, ਖਾਲੀ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ਜੇ (!kernel_env) {
        ਵਾਪਸ -1;
    }

    ਅਚਲ ਪੂਰਨ thread_id   = get_relative_thread_id(kernel_env->shire_ਨਕਾਬ);
    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਧਾਗੇ = get_ਅੰਕ_ਧਾਗੇ(kernel_env->shire_ਨਕਾਬ);

    ਜੇ (thread_id < 0) {
        ਵਾਪਸ 0;
    }

    ਜੇ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ਵਾਪਸ -1;
    }

    ਬਣਤਰ ggml_tensor * src0 = &params->src0;
    ਬਣਤਰ ggml_tensor * src1 = &params->src1;
    ਬਣਤਰ ggml_tensor * src2 = &params->src2;
    ਬਣਤਰ ggml_tensor * src3 = &params->src3;
    ਬਣਤਰ ggml_tensor * src4 = &params->src4;
    ਬਣਤਰ ggml_tensor * src5 = &params->src5;
    ਬਣਤਰ ggml_tensor * src6 = &params->src6;
    ਬਣਤਰ ggml_tensor * dst  = &params->dst;

    ਜੇ (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_F32 ||
        src3->type != GGML_TYPE_F32 || src4->type != GGML_TYPE_F32 || src5->type != GGML_TYPE_F32 ||
        src6->type != GGML_TYPE_I32 || dst->type != GGML_TYPE_F32) {
        ਵਾਪਸ -1;
    }

    ਅਚਲ ਦਸ਼ਮਲਵ *   s_ਅੰਕੜੇ   = (ਅਚਲ ਦਸ਼ਮਲਵ *) src0->ਅੰਕੜੇ;
    ਅਚਲ ਦਸ਼ਮਲਵ *   x_ਅੰਕੜੇ   = (ਅਚਲ ਦਸ਼ਮਲਵ *) src1->ਅੰਕੜੇ;
    ਅਚਲ ਦਸ਼ਮਲਵ *   dt_ਅੰਕੜੇ  = (ਅਚਲ ਦਸ਼ਮਲਵ *) src2->ਅੰਕੜੇ;
    ਅਚਲ ਦਸ਼ਮਲਵ *   A_ਅੰਕੜੇ   = (ਅਚਲ ਦਸ਼ਮਲਵ *) src3->ਅੰਕੜੇ;
    ਅਚਲ ਦਸ਼ਮਲਵ *   B_ਅੰਕੜੇ   = (ਅਚਲ ਦਸ਼ਮਲਵ *) src4->ਅੰਕੜੇ;
    ਅਚਲ ਦਸ਼ਮਲਵ *   C_ਅੰਕੜੇ   = (ਅਚਲ ਦਸ਼ਮਲਵ *) src5->ਅੰਕੜੇ;
    ਅਚਲ int32_t * ids      = (ਅਚਲ int32_t *) src6->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ *         dst_ਅੰਕੜੇ = (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ;

    ਜੇ (!s_ਅੰਕੜੇ || !x_ਅੰਕੜੇ || !dt_ਅੰਕੜੇ || !A_ਅੰਕੜੇ || !B_ਅੰਕੜੇ || !C_ਅੰਕੜੇ || !ids || !dst_ਅੰਕੜੇ) {
        ਵਾਪਸ -1;
    }

    ਅਚਲ int64_t d_state      = src0->ne[0];
    ਅਚਲ int64_t ਸਿਰ_dim     = src0->ne[1];
    ਅਚਲ int64_t n_ਸਿਰ       = src1->ne[1];
    ਅਚਲ int64_t n_ਸਮੂਹ      = src4->ne[1];
    ਅਚਲ int64_t n_seq_ਟੋਕਨ = src1->ne[2];
    ਅਚਲ int64_t n_seqs       = src1->ne[3];
    ਅਚਲ int64_t y_elems      = src1->ne[0] * src1->ne[1] * src1->ne[2] * src1->ne[3];
    ਅਚਲ int64_t K            = params->K;

    ਜੇ (src0->nb[0] != ਆਕਾਰ(ਦਸ਼ਮਲਵ) || src1->nb[0] != ਆਕਾਰ(ਦਸ਼ਮਲਵ) || src2->nb[0] != ਆਕਾਰ(ਦਸ਼ਮਲਵ) ||
        src3->nb[0] != ਆਕਾਰ(ਦਸ਼ਮਲਵ) || src4->nb[0] != ਆਕਾਰ(ਦਸ਼ਮਲਵ) || src5->nb[0] != ਆਕਾਰ(ਦਸ਼ਮਲਵ) ||
        src6->nb[0] != ਆਕਾਰ(int32_t) || dst->nb[0] != ਆਕਾਰ(ਦਸ਼ਮਲਵ)) {
        ਵਾਪਸ -1;
    }

    ਜੇ (K < 1 || n_ਸਮੂਹ <= 0 || n_ਸਿਰ % n_ਸਮੂਹ != 0) {
        ਵਾਪਸ -1;
    }

    // Cache-line bundling on the dst output (1 dst float per (head, dim, token)).
    //   - When head_dim < 16: bundle 16/head_dim heads per work-unit (1 line of dst).
    //   - When head_dim >= 16: each head's dim slice spans head_dim/16 lines, so we
    //     can split dims into chunks of 16 across threads without false sharing.
    ਅਚਲ int64_t dst_lanes_ਪ੍ਰਤੀ_cl    = 16;
    ਅਚਲ int64_t heads_ਪ੍ਰਤੀ_cacheline = ਸਿਰ_dim >= dst_lanes_ਪ੍ਰਤੀ_cl ? 1 : (dst_lanes_ਪ੍ਰਤੀ_cl / ਸਿਰ_dim);
    ਅਚਲ int64_t heads_ਪ੍ਰਤੀ_ਖੰਡ     = heads_ਪ੍ਰਤੀ_cacheline > 0 ? heads_ਪ੍ਰਤੀ_cacheline : 1;
    ਅਚਲ int64_t ਖੰਡਾਂ_ਪ੍ਰਤੀ_seq      = (n_ਸਿਰ + heads_ਪ੍ਰਤੀ_ਖੰਡ - 1) / heads_ਪ੍ਰਤੀ_ਖੰਡ;
    ਅਚਲ int64_t dim_chunk_lanes     = ਸਿਰ_dim >= dst_lanes_ਪ੍ਰਤੀ_cl ? dst_lanes_ਪ੍ਰਤੀ_cl : ਸਿਰ_dim;
    ਅਚਲ int64_t dim_chunks_ਪ੍ਰਤੀ_ਸਿਰ = (ਸਿਰ_dim + dim_chunk_lanes - 1) / dim_chunk_lanes;

    // A "unit" = (seq, head_block, dim_chunk). This expands the parallelism by a
    // factor of dim_chunks_per_head over the prior block-only scheme; for Mamba-2
    // shapes (head_dim=64) that's a 4x bump in active threads.
    ਅਚਲ int64_t units_ਪ੍ਰਤੀ_seq    = ਖੰਡਾਂ_ਪ੍ਰਤੀ_seq * dim_chunks_ਪ੍ਰਤੀ_ਸਿਰ;
    ਅਚਲ int64_t ਕੁੱਲ_units      = n_seqs * units_ਪ੍ਰਤੀ_seq;
    ਅਚਲ int64_t units_ਪ੍ਰਤੀ_thread = (ਕੁੱਲ_units + ਅੰਕ_ਧਾਗੇ - 1) / ਅੰਕ_ਧਾਗੇ;
    ਅਚਲ int64_t unit_begin       = (int64_t) thread_id * units_ਪ੍ਰਤੀ_thread;
    int64_t       unit_ਸਿਰਾ         = unit_begin + units_ਪ੍ਰਤੀ_thread;

    ਜੇ (unit_begin >= ਕੁੱਲ_units) {
        ਵਾਪਸ 0;
    }

    ਜੇ (unit_ਸਿਰਾ > ਕੁੱਲ_units) {
        unit_ਸਿਰਾ = ਕੁੱਲ_units;
    }

    ਅਚਲ ਪੂਰਨ     A_ਪ੍ਰਸਾਰਣ = (src3->ne[0] == 1);
    ਅਚਲ int64_t d_state_vec = (d_state / 8) * 8;  // largest multiple of 8 <= d_state
    ਅਚਲ ਦਸ਼ਮਲਵ   log2e_const = 1.4426950408889634f;

    ਲਈ (int64_t unit = unit_begin; unit < unit_ਸਿਰਾ; ++unit) {
        ਅਚਲ int64_t seq_idx       = unit / units_ਪ੍ਰਤੀ_seq;
        ਅਚਲ int64_t unit_ਵਿੱਚ_seq   = unit % units_ਪ੍ਰਤੀ_seq;
        ਅਚਲ int64_t ਖੰਡ_ਵਿੱਚ_seq  = unit_ਵਿੱਚ_seq / dim_chunks_ਪ੍ਰਤੀ_ਸਿਰ;
        ਅਚਲ int64_t dim_chunk_idx = unit_ਵਿੱਚ_seq % dim_chunks_ਪ੍ਰਤੀ_ਸਿਰ;
        ਅਚਲ int64_t ਸਿਰ_begin    = ਖੰਡ_ਵਿੱਚ_seq * heads_ਪ੍ਰਤੀ_ਖੰਡ;
        int64_t       ਸਿਰ_ਸਿਰਾ      = ਸਿਰ_begin + heads_ਪ੍ਰਤੀ_ਖੰਡ;

        ਜੇ (ਸਿਰ_ਸਿਰਾ > n_ਸਿਰ) {
            ਸਿਰ_ਸਿਰਾ = n_ਸਿਰ;
        }

        ਅਚਲ int64_t dim_begin = dim_chunk_idx * dim_chunk_lanes;
        int64_t       dim_ਸਿਰਾ   = dim_begin + dim_chunk_lanes;
        ਜੇ (dim_ਸਿਰਾ > ਸਿਰ_dim) {
            dim_ਸਿਰਾ = ਸਿਰ_dim;
        }

        ਅਚਲ int32_t state_seq = ids[seq_idx];

        ਲਈ (int64_t ਸਿਰ_idx = ਸਿਰ_begin; ਸਿਰ_idx < ਸਿਰ_ਸਿਰਾ; ++ਸਿਰ_idx) {
            ਅਚਲ int64_t ਸਮੂਹ_idx = ਸਿਰ_idx / (n_ਸਿਰ / n_ਸਮੂਹ);

            // A pointer for this head: contiguous over state_idx when not broadcast
            ਅਚਲ ਦਸ਼ਮਲਵ * A_ਪੰਕਤੀ = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) A_ਅੰਕੜੇ + (size_t) ਸਿਰ_idx * src3->nb[1]);

            ਲਈ (int64_t dim_idx = dim_begin; dim_idx < dim_ਸਿਰਾ; ++dim_idx) {
                ਅਚਲ ਦਸ਼ਮਲਵ * state_src =
                    (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) s_ਅੰਕੜੇ + (size_t) dim_idx * src0->nb[1] +
                                     (size_t) ਸਿਰ_idx * src0->nb[2] + (size_t) state_seq * src0->nb[3]);

                ਦਸ਼ਮਲਵ * state_dst =
                    (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst_ਅੰਕੜੇ + (size_t) y_elems * ਆਕਾਰ(ਦਸ਼ਮਲਵ) + (size_t) dim_idx * src0->nb[1] +
                               (size_t) ਸਿਰ_idx * src0->nb[2] + (size_t) seq_idx * src0->nb[3]);

                ਲਈ (int64_t token_idx = 0; token_idx < n_seq_ਟੋਕਨ; ++token_idx) {
                    ਅਚਲ ਦਸ਼ਮਲਵ * x_ptr =
                        (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) x_ਅੰਕੜੇ + (size_t) dim_idx * src1->nb[0] +
                                         (size_t) ਸਿਰ_idx * src1->nb[1] + (size_t) token_idx * src1->nb[2] +
                                         (size_t) seq_idx * src1->nb[3]);

                    ਅਚਲ ਦਸ਼ਮਲਵ * dt_ptr =
                        (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) dt_ਅੰਕੜੇ + (size_t) ਸਿਰ_idx * src2->nb[0] +
                                         (size_t) token_idx * src2->nb[1] + (size_t) seq_idx * src2->nb[2]);

                    ਅਚਲ ਦਸ਼ਮਲਵ * B_ਪੰਕਤੀ =
                        (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) B_ਅੰਕੜੇ + (size_t) ਸਮੂਹ_idx * src4->nb[1] +
                                         (size_t) token_idx * src4->nb[2] + (size_t) seq_idx * src4->nb[3]);

                    ਅਚਲ ਦਸ਼ਮਲਵ * C_ਪੰਕਤੀ =
                        (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) C_ਅੰਕੜੇ + (size_t) ਸਮੂਹ_idx * src5->nb[1] +
                                         (size_t) token_idx * src5->nb[2] + (size_t) seq_idx * src5->nb[3]);

                    ਅਚਲ ਦਸ਼ਮਲਵ dt_softplus = softplus_f32(*dt_ptr);
                    ਅਚਲ ਦਸ਼ਮਲਵ x_dt        = (*x_ptr) * dt_softplus;
                    ਅਚਲ ਦਸ਼ਮਲਵ dt_log2e    = dt_softplus * log2e_const;

                    // Source of "previous state" for this token: input state on token 0,
                    // last token's state thereafter (we wrote it into state_dst).
                    ਅਚਲ ਦਸ਼ਮਲਵ * prev_ਪੰਕਤੀ = (token_idx == 0) ? state_src : state_dst;

                    ਦਸ਼ਮਲਵ   sumf      = 0.0f;
                    int64_t state_idx = 0;

                    ਜੇ (d_state_vec > 0) {
                        // Save mask, enable all 8 vector lanes for the state loop.
                        ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ saved_ਨਕਾਬ;
                        __asm__ ਅਸਥਿਰ("mova.x.m %0" : "=r"(saved_ਨਕਾਬ));
                        __asm__ ਅਸਥਿਰ("mov.m.x m0, x0, 0xFF");

                        // Per-token broadcasts:
                        //   f20 = x_dt     (B*x_dt)
                        //   f21 = dt_log2e (for fexp.ps when A is per-state)
                        //   f22 = dA       (only when A is broadcast scalar)
                        //   f23 = sum-of-products accumulator (zeroed)
                        __asm__ ਅਸਥਿਰ(
                            "fbc.ps f20, %[xdt]\n\t"
                            "fbc.ps f21, %[dtl]\n\t"
                            "fbci.pi f23, 0\n\t"
                            :
                            : [xdt] "m"(x_dt), [dtl] "m"(dt_log2e)
                            : "f20", "f21", "f23");

                        ਜੇ (A_ਪ੍ਰਸਾਰਣ) {
                            // dA is a per-head scalar — compute once and splat.
                            ਅਚਲ ਦਸ਼ਮਲਵ dA_scalar = et_expf(dt_softplus * (*A_ਪੰਕਤੀ));
                            __asm__ ਅਸਥਿਰ("fbc.ps f22, %[da]\n\t" : : [da] "m"(dA_scalar) : "f22");
                        }

                        ਲਈ (; state_idx < d_state_vec; state_idx += 8) {
                            ਜੇ (!A_ਪ੍ਰਸਾਰਣ) {
                                // f22 = exp(dt_softplus * A[state..state+7])
                                //     = 2^((dt_softplus * A) * log2e) via fexp.ps
                                __asm__ ਅਸਥਿਰ(
                                    "flw.ps  f24, %[av]\n\t"
                                    "fmul.ps f24, f24, f21\n\t"  // A * dt_log2e
                                    "fexp.ps f22, f24\n\t"       // dA = 2^(...)
                                    :
                                    : [av] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & A_ਪੰਕਤੀ[state_idx])
                                    : "f22", "f24");
                            }

                            // state = prev * dA + B * x_dt
                            // sumf += state * C
                            // Reads prev before writing state_dst — safe even when
                            // prev_row == state_dst (write-after-read, same index).
                            __asm__ ਅਸਥਿਰ(
                                "flw.ps  f25, %[prev]\n\t"
                                "flw.ps  f26, %[bv]\n\t"
                                "flw.ps  f27, %[cv]\n\t"
                                "fmul.ps f26, f26, f20\n\t"        // B * x_dt
                                "fmadd.ps f25, f25, f22, f26\n\t"  // state = prev*dA + B*x_dt
                                "fsw.ps  f25, %[sd]\n\t"
                                "fmadd.ps f23, f25, f27, f23\n\t"  // sum += state*C
                                : [sd] "=m"(*(ਦਸ਼ਮਲਵ (*)[8]) & state_dst[state_idx])
                                : [prev] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & prev_ਪੰਕਤੀ[state_idx]),
                                  [bv] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & B_ਪੰਕਤੀ[state_idx]),
                                  [cv] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & C_ਪੰਕਤੀ[state_idx])
                                : "f25", "f26", "f27");
                        }

                        // Horizontal reduce f23 (8 lanes) -> scalar sumf.
                        __asm__ ਅਸਥਿਰ(
                            "fswizz.ps f1, f23, 0xB1\n\t"
                            "fadd.ps   f2, f23, f1, rne\n\t"
                            "fswizz.ps f3, f2, 0x4E\n\t"
                            "fadd.ps   f4, f2, f3, rne\n\t"
                            "fmvz.x.ps t0, f4, 4\n\t"
                            "fbcx.ps   f5, t0\n\t"
                            "fadd.ps   %[vout], f4, f5, rne\n\t"
                            : [vout] "=f"(sumf)::"t0", "f1", "f2", "f3", "f4", "f5");

                        __asm__ ਅਸਥਿਰ("mova.m.x %0" ::"r"(saved_ਨਕਾਬ));
                    }

                    // Scalar tail (d_state not a multiple of 8).
                    ਲਈ (; state_idx < d_state; ++state_idx) {
                        ਅਚਲ ਦਸ਼ਮਲਵ prev_state = prev_ਪੰਕਤੀ[state_idx];
                        ਅਚਲ ਦਸ਼ਮਲਵ A_val      = A_ਪ੍ਰਸਾਰਣ ? *A_ਪੰਕਤੀ : A_ਪੰਕਤੀ[state_idx];
                        ਅਚਲ ਦਸ਼ਮਲਵ dA         = et_expf(dt_softplus * A_val);
                        ਅਚਲ ਦਸ਼ਮਲਵ st         = prev_state * dA + B_ਪੰਕਤੀ[state_idx] * x_dt;
                        state_dst[state_idx]   = st;
                        sumf += st * C_ਪੰਕਤੀ[state_idx];
                    }

                    ਅਚਲ int64_t ਥਾਂ = n_seq_ਟੋਕਨ - 1 - token_idx;
                    ਜੇ (ਥਾਂ > 0 && ਥਾਂ < K) {
                        ਦਸ਼ਮਲਵ * state_snapshot =
                            (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) state_dst + (size_t) ਥਾਂ * n_seqs * src0->nb[3]);
                        ਲਈ (int64_t i = 0; i < d_state; ++i) {
                            state_snapshot[i] = state_dst[i];
                        }
                    }

                    dst_ਅੰਕੜੇ[seq_idx * (n_seq_ਟੋਕਨ * n_ਸਿਰ * ਸਿਰ_dim) + token_idx * (n_ਸਿਰ * ਸਿਰ_dim) +
                             ਸਿਰ_idx * ਸਿਰ_dim + dim_idx] = sumf;
                }
            }
        }
    }

    ਵਾਪਸ 0;
}
