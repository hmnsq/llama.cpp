#ಅನ್ತರ್ಭಾವ "ggml_tensor.h"
#ಅನ್ತರ್ಭಾವ "math_fp.h"
#ಅನ್ತರ್ಭಾವ "platform.h"

#ಅನ್ತರ್ಭಾವ <stdint.h>

ರಚನೆ ggml_et_ssm_ಪರಿಶೀಲನೆ_params {
    ರಚನೆ ggml_tensor src0;  // s:   [d_state, head_dim, n_head, n_seqs]
    ರಚನೆ ggml_tensor src1;  // x:   [head_dim, n_head, n_seq_tokens, n_seqs]
    ರಚನೆ ggml_tensor src2;  // dt:  [n_head, n_seq_tokens, n_seqs]
    ರಚನೆ ggml_tensor src3;  // A:   [d_state, n_head] or [1, n_head]
    ರಚನೆ ggml_tensor src4;  // B:   [d_state, n_group, n_seq_tokens, n_seqs]
    ರಚನೆ ggml_tensor src5;  // C:   [d_state, n_group, n_seq_tokens, n_seqs]
    ರಚನೆ ggml_tensor src6;  // ids: [n_seqs] i32
    ರಚನೆ ggml_tensor dst;   // packed [y, states]
    int32_t            K;
};

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ softplus_f32(ದಶಮಾಂಶ x) {
    ಹಿಂತಿರುಗಿ x <= 20.0f ? et_logf(1.0f + et_expf(x)) : x;
}

ಪೂರ್ಣಾಂಕ ನಮೂದು_ಬಿಂದು(ರಚನೆ ggml_et_ssm_ಪರಿಶೀಲನೆ_params * params, ಶೂನ್ಯ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ಆದರೆ (!kernel_env) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ thread_id   = get_relative_thread_id(kernel_env->shire_ಮುಖವಾಡ);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_ಎಳೆಗಳು = get_ಅಂಕೆ_ಎಳೆಗಳು(kernel_env->shire_ಮುಖವಾಡ);

    ಆದರೆ (thread_id < 0) {
        ಹಿಂತಿರುಗಿ 0;
    }

    ಆದರೆ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ರಚನೆ ggml_tensor * src0 = &params->src0;
    ರಚನೆ ggml_tensor * src1 = &params->src1;
    ರಚನೆ ggml_tensor * src2 = &params->src2;
    ರಚನೆ ggml_tensor * src3 = &params->src3;
    ರಚನೆ ggml_tensor * src4 = &params->src4;
    ರಚನೆ ggml_tensor * src5 = &params->src5;
    ರಚನೆ ggml_tensor * src6 = &params->src6;
    ರಚನೆ ggml_tensor * dst  = &params->dst;

    ಆದರೆ (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_F32 ||
        src3->type != GGML_TYPE_F32 || src4->type != GGML_TYPE_F32 || src5->type != GGML_TYPE_F32 ||
        src6->type != GGML_TYPE_I32 || dst->type != GGML_TYPE_F32) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *   s_ದತ್ತಾಂಶ   = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src0->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *   x_ದತ್ತಾಂಶ   = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src1->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *   dt_ದತ್ತಾಂಶ  = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src2->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *   A_ದತ್ತಾಂಶ   = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src3->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *   B_ದತ್ತಾಂಶ   = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src4->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *   C_ದತ್ತಾಂಶ   = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src5->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ int32_t * ids      = (ಸ್ಥಿರಾಂಕ int32_t *) src6->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ *         dst_ದತ್ತಾಂಶ = (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ;

    ಆದರೆ (!s_ದತ್ತಾಂಶ || !x_ದತ್ತಾಂಶ || !dt_ದತ್ತಾಂಶ || !A_ದತ್ತಾಂಶ || !B_ದತ್ತಾಂಶ || !C_ದತ್ತಾಂಶ || !ids || !dst_ದತ್ತಾಂಶ) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಸ್ಥಿರಾಂಕ int64_t d_state      = src0->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t ತಲೆ_dim     = src0->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t n_ತಲೆ       = src1->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t n_ಗುಂಪು      = src4->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t n_seq_ಟೋಕನುಗಳು = src1->ne[2];
    ಸ್ಥಿರಾಂಕ int64_t n_seqs       = src1->ne[3];
    ಸ್ಥಿರಾಂಕ int64_t y_elems      = src1->ne[0] * src1->ne[1] * src1->ne[2] * src1->ne[3];
    ಸ್ಥಿರಾಂಕ int64_t K            = params->K;

    ಆದರೆ (src0->nb[0] != ಗಾತ್ರ(ದಶಮಾಂಶ) || src1->nb[0] != ಗಾತ್ರ(ದಶಮಾಂಶ) || src2->nb[0] != ಗಾತ್ರ(ದಶಮಾಂಶ) ||
        src3->nb[0] != ಗಾತ್ರ(ದಶಮಾಂಶ) || src4->nb[0] != ಗಾತ್ರ(ದಶಮಾಂಶ) || src5->nb[0] != ಗಾತ್ರ(ದಶಮಾಂಶ) ||
        src6->nb[0] != ಗಾತ್ರ(int32_t) || dst->nb[0] != ಗಾತ್ರ(ದಶಮಾಂಶ)) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಆದರೆ (K < 1 || n_ಗುಂಪು <= 0 || n_ತಲೆ % n_ಗುಂಪು != 0) {
        ಹಿಂತಿರುಗಿ -1;
    }

    // Cache-line bundling on the dst output (1 dst float per (head, dim, token)).
    //   - When head_dim < 16: bundle 16/head_dim heads per work-unit (1 line of dst).
    //   - When head_dim >= 16: each head's dim slice spans head_dim/16 lines, so we
    //     can split dims into chunks of 16 across threads without false sharing.
    ಸ್ಥಿರಾಂಕ int64_t dst_lanes_ಪ್ರತಿ_cl    = 16;
    ಸ್ಥಿರಾಂಕ int64_t heads_ಪ್ರತಿ_cacheline = ತಲೆ_dim >= dst_lanes_ಪ್ರತಿ_cl ? 1 : (dst_lanes_ಪ್ರತಿ_cl / ತಲೆ_dim);
    ಸ್ಥಿರಾಂಕ int64_t heads_ಪ್ರತಿ_ಖಂಡ     = heads_ಪ್ರತಿ_cacheline > 0 ? heads_ಪ್ರತಿ_cacheline : 1;
    ಸ್ಥಿರಾಂಕ int64_t ಖಂಡಗಳು_ಪ್ರತಿ_seq      = (n_ತಲೆ + heads_ಪ್ರತಿ_ಖಂಡ - 1) / heads_ಪ್ರತಿ_ಖಂಡ;
    ಸ್ಥಿರಾಂಕ int64_t dim_chunk_lanes     = ತಲೆ_dim >= dst_lanes_ಪ್ರತಿ_cl ? dst_lanes_ಪ್ರತಿ_cl : ತಲೆ_dim;
    ಸ್ಥಿರಾಂಕ int64_t dim_chunks_ಪ್ರತಿ_ತಲೆ = (ತಲೆ_dim + dim_chunk_lanes - 1) / dim_chunk_lanes;

    // A "unit" = (seq, head_block, dim_chunk). This expands the parallelism by a
    // factor of dim_chunks_per_head over the prior block-only scheme; for Mamba-2
    // shapes (head_dim=64) that's a 4x bump in active threads.
    ಸ್ಥಿರಾಂಕ int64_t units_ಪ್ರತಿ_seq    = ಖಂಡಗಳು_ಪ್ರತಿ_seq * dim_chunks_ಪ್ರತಿ_ತಲೆ;
    ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_units      = n_seqs * units_ಪ್ರತಿ_seq;
    ಸ್ಥಿರಾಂಕ int64_t units_ಪ್ರತಿ_thread = (ಒಟ್ಟು_units + ಅಂಕೆ_ಎಳೆಗಳು - 1) / ಅಂಕೆ_ಎಳೆಗಳು;
    ಸ್ಥಿರಾಂಕ int64_t unit_begin       = (int64_t) thread_id * units_ಪ್ರತಿ_thread;
    int64_t       unit_ಕೊನೆ         = unit_begin + units_ಪ್ರತಿ_thread;

    ಆದರೆ (unit_begin >= ಒಟ್ಟು_units) {
        ಹಿಂತಿರುಗಿ 0;
    }

    ಆದರೆ (unit_ಕೊನೆ > ಒಟ್ಟು_units) {
        unit_ಕೊನೆ = ಒಟ್ಟು_units;
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ     A_ಪ್ರಸಾರ = (src3->ne[0] == 1);
    ಸ್ಥಿರಾಂಕ int64_t d_state_vec = (d_state / 8) * 8;  // largest multiple of 8 <= d_state
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ   log2e_const = 1.4426950408889634f;

    ಗಾಗಿ (int64_t unit = unit_begin; unit < unit_ಕೊನೆ; ++unit) {
        ಸ್ಥಿರಾಂಕ int64_t seq_idx       = unit / units_ಪ್ರತಿ_seq;
        ಸ್ಥಿರಾಂಕ int64_t unit_ಒಳ_seq   = unit % units_ಪ್ರತಿ_seq;
        ಸ್ಥಿರಾಂಕ int64_t ಖಂಡ_ಒಳ_seq  = unit_ಒಳ_seq / dim_chunks_ಪ್ರತಿ_ತಲೆ;
        ಸ್ಥಿರಾಂಕ int64_t dim_chunk_idx = unit_ಒಳ_seq % dim_chunks_ಪ್ರತಿ_ತಲೆ;
        ಸ್ಥಿರಾಂಕ int64_t ತಲೆ_begin    = ಖಂಡ_ಒಳ_seq * heads_ಪ್ರತಿ_ಖಂಡ;
        int64_t       ತಲೆ_ಕೊನೆ      = ತಲೆ_begin + heads_ಪ್ರತಿ_ಖಂಡ;

        ಆದರೆ (ತಲೆ_ಕೊನೆ > n_ತಲೆ) {
            ತಲೆ_ಕೊನೆ = n_ತಲೆ;
        }

        ಸ್ಥಿರಾಂಕ int64_t dim_begin = dim_chunk_idx * dim_chunk_lanes;
        int64_t       dim_ಕೊನೆ   = dim_begin + dim_chunk_lanes;
        ಆದರೆ (dim_ಕೊನೆ > ತಲೆ_dim) {
            dim_ಕೊನೆ = ತಲೆ_dim;
        }

        ಸ್ಥಿರಾಂಕ int32_t state_seq = ids[seq_idx];

        ಗಾಗಿ (int64_t ತಲೆ_idx = ತಲೆ_begin; ತಲೆ_idx < ತಲೆ_ಕೊನೆ; ++ತಲೆ_idx) {
            ಸ್ಥಿರಾಂಕ int64_t ಗುಂಪು_idx = ತಲೆ_idx / (n_ತಲೆ / n_ಗುಂಪು);

            // A pointer for this head: contiguous over state_idx when not broadcast
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * A_ಅಡ್ಡಸಾಲು = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) A_ದತ್ತಾಂಶ + (size_t) ತಲೆ_idx * src3->nb[1]);

            ಗಾಗಿ (int64_t dim_idx = dim_begin; dim_idx < dim_ಕೊನೆ; ++dim_idx) {
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * state_src =
                    (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) s_ದತ್ತಾಂಶ + (size_t) dim_idx * src0->nb[1] +
                                     (size_t) ತಲೆ_idx * src0->nb[2] + (size_t) state_seq * src0->nb[3]);

                ದಶಮಾಂಶ * state_dst =
                    (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst_ದತ್ತಾಂಶ + (size_t) y_elems * ಗಾತ್ರ(ದಶಮಾಂಶ) + (size_t) dim_idx * src0->nb[1] +
                               (size_t) ತಲೆ_idx * src0->nb[2] + (size_t) seq_idx * src0->nb[3]);

                ಗಾಗಿ (int64_t token_idx = 0; token_idx < n_seq_ಟೋಕನುಗಳು; ++token_idx) {
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * x_ptr =
                        (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) x_ದತ್ತಾಂಶ + (size_t) dim_idx * src1->nb[0] +
                                         (size_t) ತಲೆ_idx * src1->nb[1] + (size_t) token_idx * src1->nb[2] +
                                         (size_t) seq_idx * src1->nb[3]);

                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * dt_ptr =
                        (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) dt_ದತ್ತಾಂಶ + (size_t) ತಲೆ_idx * src2->nb[0] +
                                         (size_t) token_idx * src2->nb[1] + (size_t) seq_idx * src2->nb[2]);

                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * B_ಅಡ್ಡಸಾಲು =
                        (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) B_ದತ್ತಾಂಶ + (size_t) ಗುಂಪು_idx * src4->nb[1] +
                                         (size_t) token_idx * src4->nb[2] + (size_t) seq_idx * src4->nb[3]);

                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * C_ಅಡ್ಡಸಾಲು =
                        (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) C_ದತ್ತಾಂಶ + (size_t) ಗುಂಪು_idx * src5->nb[1] +
                                         (size_t) token_idx * src5->nb[2] + (size_t) seq_idx * src5->nb[3]);

                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dt_softplus = softplus_f32(*dt_ptr);
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x_dt        = (*x_ptr) * dt_softplus;
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dt_log2e    = dt_softplus * log2e_const;

                    // Source of "previous state" for this token: input state on token 0,
                    // last token's state thereafter (we wrote it into state_dst).
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * prev_ಅಡ್ಡಸಾಲು = (token_idx == 0) ? state_src : state_dst;

                    ದಶಮಾಂಶ   sumf      = 0.0f;
                    int64_t state_idx = 0;

                    ಆದರೆ (d_state_vec > 0) {
                        // Save mask, enable all 8 vector lanes for the state loop.
                        ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ saved_ಮುಖವಾಡ;
                        __asm__ ಅಸ್ಥಿರ("mova.x.m %0" : "=r"(saved_ಮುಖವಾಡ));
                        __asm__ ಅಸ್ಥಿರ("mov.m.x m0, x0, 0xFF");

                        // Per-token broadcasts:
                        //   f20 = x_dt     (B*x_dt)
                        //   f21 = dt_log2e (for fexp.ps when A is per-state)
                        //   f22 = dA       (only when A is broadcast scalar)
                        //   f23 = sum-of-products accumulator (zeroed)
                        __asm__ ಅಸ್ಥಿರ(
                            "fbc.ps f20, %[xdt]\n\t"
                            "fbc.ps f21, %[dtl]\n\t"
                            "fbci.pi f23, 0\n\t"
                            :
                            : [xdt] "m"(x_dt), [dtl] "m"(dt_log2e)
                            : "f20", "f21", "f23");

                        ಆದರೆ (A_ಪ್ರಸಾರ) {
                            // dA is a per-head scalar — compute once and splat.
                            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dA_scalar = et_expf(dt_softplus * (*A_ಅಡ್ಡಸಾಲು));
                            __asm__ ಅಸ್ಥಿರ("fbc.ps f22, %[da]\n\t" : : [da] "m"(dA_scalar) : "f22");
                        }

                        ಗಾಗಿ (; state_idx < d_state_vec; state_idx += 8) {
                            ಆದರೆ (!A_ಪ್ರಸಾರ) {
                                // f22 = exp(dt_softplus * A[state..state+7])
                                //     = 2^((dt_softplus * A) * log2e) via fexp.ps
                                __asm__ ಅಸ್ಥಿರ(
                                    "flw.ps  f24, %[av]\n\t"
                                    "fmul.ps f24, f24, f21\n\t"  // A * dt_log2e
                                    "fexp.ps f22, f24\n\t"       // dA = 2^(...)
                                    :
                                    : [av] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[8]) & A_ಅಡ್ಡಸಾಲು[state_idx])
                                    : "f22", "f24");
                            }

                            // state = prev * dA + B * x_dt
                            // sumf += state * C
                            // Reads prev before writing state_dst — safe even when
                            // prev_row == state_dst (write-after-read, same index).
                            __asm__ ಅಸ್ಥಿರ(
                                "flw.ps  f25, %[prev]\n\t"
                                "flw.ps  f26, %[bv]\n\t"
                                "flw.ps  f27, %[cv]\n\t"
                                "fmul.ps f26, f26, f20\n\t"        // B * x_dt
                                "fmadd.ps f25, f25, f22, f26\n\t"  // state = prev*dA + B*x_dt
                                "fsw.ps  f25, %[sd]\n\t"
                                "fmadd.ps f23, f25, f27, f23\n\t"  // sum += state*C
                                : [sd] "=m"(*(ದಶಮಾಂಶ (*)[8]) & state_dst[state_idx])
                                : [prev] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[8]) & prev_ಅಡ್ಡಸಾಲು[state_idx]),
                                  [bv] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[8]) & B_ಅಡ್ಡಸಾಲು[state_idx]),
                                  [cv] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[8]) & C_ಅಡ್ಡಸಾಲು[state_idx])
                                : "f25", "f26", "f27");
                        }

                        // Horizontal reduce f23 (8 lanes) -> scalar sumf.
                        __asm__ ಅಸ್ಥಿರ(
                            "fswizz.ps f1, f23, 0xB1\n\t"
                            "fadd.ps   f2, f23, f1, rne\n\t"
                            "fswizz.ps f3, f2, 0x4E\n\t"
                            "fadd.ps   f4, f2, f3, rne\n\t"
                            "fmvz.x.ps t0, f4, 4\n\t"
                            "fbcx.ps   f5, t0\n\t"
                            "fadd.ps   %[vout], f4, f5, rne\n\t"
                            : [vout] "=f"(sumf)::"t0", "f1", "f2", "f3", "f4", "f5");

                        __asm__ ಅಸ್ಥಿರ("mova.m.x %0" ::"r"(saved_ಮುಖವಾಡ));
                    }

                    // Scalar tail (d_state not a multiple of 8).
                    ಗಾಗಿ (; state_idx < d_state; ++state_idx) {
                        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ prev_state = prev_ಅಡ್ಡಸಾಲು[state_idx];
                        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ A_val      = A_ಪ್ರಸಾರ ? *A_ಅಡ್ಡಸಾಲು : A_ಅಡ್ಡಸಾಲು[state_idx];
                        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dA         = et_expf(dt_softplus * A_val);
                        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ st         = prev_state * dA + B_ಅಡ್ಡಸಾಲು[state_idx] * x_dt;
                        state_dst[state_idx]   = st;
                        sumf += st * C_ಅಡ್ಡಸಾಲು[state_idx];
                    }

                    ಸ್ಥಿರಾಂಕ int64_t ಸ್ಥಾನ = n_seq_ಟೋಕನುಗಳು - 1 - token_idx;
                    ಆದರೆ (ಸ್ಥಾನ > 0 && ಸ್ಥಾನ < K) {
                        ದಶಮಾಂಶ * state_snapshot =
                            (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) state_dst + (size_t) ಸ್ಥಾನ * n_seqs * src0->nb[3]);
                        ಗಾಗಿ (int64_t i = 0; i < d_state; ++i) {
                            state_snapshot[i] = state_dst[i];
                        }
                    }

                    dst_ದತ್ತಾಂಶ[seq_idx * (n_seq_ಟೋಕನುಗಳು * n_ತಲೆ * ತಲೆ_dim) + token_idx * (n_ತಲೆ * ತಲೆ_dim) +
                             ತಲೆ_idx * ತಲೆ_dim + dim_idx] = sumf;
                }
            }
        }
    }

    ಹಿಂತಿರುಗಿ 0;
}
